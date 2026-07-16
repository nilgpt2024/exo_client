"""Unlimited-OCR 分片模型封装

直接复用 transformers 的 AutoModelForCausalLM，加载完整模型后按 Shard 层范围裁剪，
使其适配 exo 的分片分布式推理框架。
"""
import json
import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any

# 兼容性补丁：Unlimited-OCR 的 remote code 基于 transformers 4.46，
# 使用了 5.x 中已移除的 is_torch_fx_available
import transformers.utils.import_utils as _import_utils
if not hasattr(_import_utils, "is_torch_fx_available"):
    _import_utils.is_torch_fx_available = lambda: False

logger = logging.getLogger(__name__)


@dataclass
class ShardedModelOutput:
    """分片模型输出：统一 logits / last_hidden_state / past_key_values"""
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None


class IdentityBlock(nn.Module):
    """占位模块：不参与计算，直接返回隐藏状态和原 past_key_values"""

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
        # 保持 transformers 层返回 (hidden_states, past_key_values) 的约定
        past_key_values = kwargs.get("past_key_values", None)
        return hidden_states, past_key_values


class ShardedUnlimitedOCRModel(nn.Module):
    """按 Shard 裁剪的 Unlimited-OCR 模型包装器"""

    def __init__(
        self,
        model_path: Union[str, Path],
        shard: "Shard",
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.shard = shard
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype if dtype is not None else torch.float32
        self.model_path = Path(model_path)

        # 1. 加载配置
        from transformers import AutoConfig
        self.config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code
        )
        # 把分片信息注入 config，便于后续调试/校验
        self.config.shard = {
            "model_id": shard.model_id,
            "start_layer": shard.start_layer,
            "end_layer": shard.end_layer,
            "n_layers": shard.n_layers,
        }

        # 2. 加载完整模型到 meta device（不分配权重内存）
        # 注意：该模型的 auto_map 中 AutoModel 指向 UnlimitedOCRForCausalLM，
        # 因此使用 AutoModel 即可加载完整因果模型。
        from transformers import AutoModel
        try:
            full_model = AutoModel.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=trust_remote_code,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="meta",
            )
        except Exception as e:
            logger.warning(f"[UnlimitedOCR] device_map='meta' 加载失败: {e}，回退到 CPU 全量加载")
            full_model = AutoModel.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=trust_remote_code,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )

        # 3. 保存引用并裁剪（不在这里移动设备，由调用方通过 to_empty + load_state_dict 处理）
        self.model = full_model
        self._prune_modules()

    def _get_base_model(self):
        """获取语言模型的基座部分（embed_tokens + layers + norm）"""
        # 标准结构：ForCausalLM.model 是 BaseModel
        if hasattr(self.model, "model"):
            return self.model.model
        return self.model

    def _prune_modules(self):
        """根据 shard 裁剪不需要的模块"""
        base = self._get_base_model()
        start_layer = self.shard.start_layer
        end_layer = self.shard.end_layer

        # 裁剪 decoder layers：范围外的替换为 IdentityBlock
        if hasattr(base, "layers"):
            for i, layer in enumerate(base.layers):
                if i < start_layer or i > end_layer:
                    base.layers[i] = IdentityBlock()

        # 首分片保留 embed_tokens / vision / projector；非首分片删除
        if not self.shard.is_first_layer():
            for name in ["embed_tokens", "vision_model", "sam_model", "projector", "image_newline", "view_seperator"]:
                if hasattr(base, name):
                    setattr(base, name, None)
            # 某些 remote code 把视觉模块挂在 ForCausalLM 上
            for name in ["vision_model", "sam_model", "projector"]:
                if hasattr(self.model, name):
                    setattr(self.model, name, None)

        # 尾分片保留 norm 和 lm_head；非尾分片把 norm 替换为 Identity（避免 remote code 调用时报错）
        if not self.shard.is_last_layer():
            if hasattr(base, "norm") and base.norm is not None:
                base.norm = nn.Identity()
            if hasattr(self.model, "lm_head"):
                self.model.lm_head = None

    @classmethod
    def sanitize(cls, state_dict: Dict[str, Any], shard: "Shard") -> Dict[str, Any]:
        """按分片范围过滤权重键

        过滤规则：
        - 仅首分片保留 embed_tokens / vision_model / sam_model / projector / image_newline / view_seperator
        - 仅尾分片保留 model.norm / lm_head
        - 仅保留 [start_layer, end_layer] 范围内的 model.layers.N.*
        """
        start_layer = shard.start_layer
        end_layer = shard.end_layer
        filtered = {}

        for key, tensor in state_dict.items():
            keep = False

            # 语言模型层
            if key.startswith("model.layers."):
                try:
                    layer_idx = int(key.split(".")[2])
                    if start_layer <= layer_idx <= end_layer:
                        keep = True
                except (ValueError, IndexError):
                    pass

            # 嵌入层、视觉/投影相关（仅首分片）
            elif shard.is_first_layer() and (
                key.startswith("model.embed_tokens.")
                or key.startswith("model.vision_model.")
                or key.startswith("model.sam_model.")
                or key.startswith("model.projector.")
                or key == "model.image_newline"
                or key == "model.view_seperator"
            ):
                keep = True

            # 最终 norm（仅尾分片）
            elif shard.is_last_layer() and key.startswith("model.norm."):
                keep = True

            # lm_head（仅尾分片）
            elif shard.is_last_layer() and key.startswith("lm_head."):
                keep = True

            if keep:
                filtered[key] = tensor

        removed = len(state_dict) - len(filtered)
        logger.info(f"[UnlimitedOCR] sanitize 保留 {len(filtered)}/{len(state_dict)} 个参数，移除 {removed} 个")
        return filtered

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        return_dict: bool = True,
        **kwargs
    ):
        """分片感知前向传播

        - 首分片：input_ids 有效，走完整 embedding + 视觉路径
        - 中间/尾分片：inputs_embeds 为上阶段隐藏状态
        - 非尾分片返回 last_hidden_state
        - 尾分片返回 logits（取最后一个位置）
        """
        base = self._get_base_model()

        # 构造传给 base model 的参数
        base_kwargs = {
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "return_dict": return_dict,
        }
        if input_ids is not None:
            base_kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            base_kwargs["inputs_embeds"] = inputs_embeds

        # 透传视觉/其它字段（首分片时有效）
        for key in ["pixel_values", "image_grid_thw", "image_sizes", "deepip_pixel_values", "sam_pixel_values"]:
            if key in kwargs and kwargs[key] is not None:
                base_kwargs[key] = kwargs[key]

        base_outputs = base(**base_kwargs)

        # 取隐藏状态
        if hasattr(base_outputs, "last_hidden_state"):
            hidden_states = base_outputs.last_hidden_state
        else:
            hidden_states = base_outputs[0]

        # 取更新后的 KV 缓存
        updated_past = getattr(base_outputs, "past_key_values", past_key_values)

        # 非尾分片直接返回隐藏状态
        if not self.shard.is_last_layer():
            return ShardedModelOutput(
                last_hidden_state=hidden_states,
                past_key_values=updated_past,
            )

        # 尾分片：应用 norm 和 lm_head 生成 logits
        if hasattr(base, "norm") and base.norm is not None:
            hidden_states = base.norm(hidden_states)

        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            logits = self.model.lm_head(hidden_states)
        else:
            logits = hidden_states

        # 取最后一个位置用于生成
        if logits.dim() >= 2:
            logits = logits[:, -1, :]

        return ShardedModelOutput(
            logits=logits,
            past_key_values=updated_past,
        )
