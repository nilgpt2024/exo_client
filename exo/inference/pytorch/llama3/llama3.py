from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from exo.inference.shard import Shard
from exo.inference.pytorch.models.base import IdentityBlock


@dataclass
class ModelArgs(LlamaConfig):
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        # 处理shard配置
        if isinstance(self.shard, Shard):
            # 确保内部属性已初始化
            self._init_internal_attrs()
            return
        if not isinstance(self.shard, dict):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

        self.shard = Shard(**self.shard)
        # 确保内部属性已初始化
        self._init_internal_attrs()

    def _init_internal_attrs(self):
        """初始化transformers需要的内部属性"""
        # 初始化注意力实现相关属性
        if not hasattr(self, '_attn_implementation_internal'):
            # 从attn_implementation或默认值获取
            attn_impl = getattr(self, 'attn_implementation', None)
            if attn_impl is None:
                attn_impl = 'eager'  # 默认使用eager实现
            object.__setattr__(self, '_attn_implementation_internal', attn_impl)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建 ModelArgs 实例"""
        # 提取 shard 配置
        shard_config = config_dict.pop("shard", {})

        # 创建 LlamaConfig 实例，让它处理自己的参数
        llama_config = LlamaConfig.from_dict(config_dict)
        
        # 转换为 ModelArgs 实例
        instance = cls.from_config(llama_config)

        # 设置 shard - 提供默认值以避免参数缺失错误
        if isinstance(shard_config, dict):
            # 确保 shard 配置包含必要的参数
            if not shard_config:
                instance.shard = Shard("", 0, 0, 0)  # 使用默认值
            else:
                instance.shard = Shard(**shard_config)
        elif isinstance(shard_config, Shard):
            instance.shard = shard_config

        return instance
        
    @classmethod
    def from_config(cls, config: LlamaConfig):
        """从 LlamaConfig 实例创建 ModelArgs 实例"""
        # 创建一个空的 ModelArgs 实例
        instance = cls()
        
        # 复制所有 LlamaConfig 的属性
        for key, value in config.__dict__.items():
            if not key.startswith('_'):  # 跳过私有属性
                setattr(instance, key, value)
        
        # 确保 shard 有默认值
        if not hasattr(instance, 'shard'):
            instance.shard = Shard("", 0, 0, 0)
        
        return instance


class Llama3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        # 创建旋转位置嵌入
        self.rotary_emb = LlamaRotaryEmbedding(config=args)

        # 只在第一层或最后层（且权重共享）时创建嵌入层
        if args.shard.is_first_layer() or (args.shard.is_last_layer() and args.tie_word_embeddings):
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

            # 根据分片配置构建层
        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            if args.shard.start_layer <= i <= args.shard.end_layer:
                # 在分片范围内使用真实的 Transformer 层
                self.layers.append(LlamaDecoderLayer(args, layer_idx=i))
            else:
                # 在分片范围外使用占位符
                self.layers.append(IdentityBlock())

                # 只在最后层创建归一化层
        if args.shard.is_last_layer():
            self.norm = LlamaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        # 处理输入嵌入
        if inputs_embeds is None:
            if self.args.shard.is_first_layer():
                hidden_states = self.embed_tokens(input_ids)
            else:
                # 对于非第一层分片，input_ids 实际上是前一层的隐藏状态
                hidden_states = input_ids.float()
        else:
            hidden_states = inputs_embeds

        # 创建注意力掩码 - 参考Tinygrad的实现，考虑past_key_values长度
        if attention_mask is None and hidden_states.shape[1] > 1:
            batch_size, seq_len = hidden_states.shape[:2]
            
            # 计算总的序列长度（包括过去的key-values）
            past_seq_len = 0
            if past_key_values is not None:
                # 检查是否是DynamicCache对象
                if hasattr(past_key_values, 'get_seq_length'):
                    past_seq_len = past_key_values.get_seq_length()
                elif len(past_key_values) > 0:
                    # 从past_key_values列表中获取过去的序列长度
                    for past_kv in past_key_values:
                        if past_kv is not None and hasattr(past_kv, 'shape') and len(past_kv.shape) >= 3:
                            past_seq_len = past_kv.shape[2]  # 假设形状为 [batch, heads, seq_len, head_dim]
                            break
                        elif past_kv is not None and hasattr(past_kv, 'key') and hasattr(past_kv.key, 'shape'):
                            past_seq_len = past_kv.key.shape[2]
                            break
            
            total_seq_len = past_seq_len + seq_len
            
            # 创建扩展的因果掩码，形状为 [batch_size, 1, seq_len, total_seq_len]
            # 这允许当前序列关注所有之前的token（包括缓存中的）
            # 使用0表示可以关注的位，使用-inf表示需要屏蔽的位
            attention_mask = torch.zeros(
                (seq_len, total_seq_len),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            # 创建下三角掩码，屏蔽未来的位置
            if total_seq_len > 0:
                # 对于每个查询位置，只允许关注键值位置 <= 查询位置 + past_seq_len
                for q_pos in range(seq_len):
                    # 查询位置q_pos可以关注所有键值位置 <= q_pos + past_seq_len
                    max_kv_pos = q_pos + past_seq_len
                    if max_kv_pos + 1 < total_seq_len:
                        # 屏蔽max_kv_pos之后的位置
                        attention_mask[q_pos, max_kv_pos + 1:] = float('-inf')

            # 添加batch和head维度
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # 计算旋转位置编码 - 新版本transformers需要显式计算
        # 获取序列长度和kv长度
        seq_len = hidden_states.shape[1]
        past_seq_len = 0
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                past_seq_len = past_key_values.get_seq_length()
            elif len(past_key_values) > 0:
                for past_kv in past_key_values:
                    if past_kv is not None and hasattr(past_kv, 'shape') and len(past_kv.shape) >= 3:
                        past_seq_len = past_kv.shape[2]
                        break
                    elif past_kv is not None and hasattr(past_kv, 'key') and hasattr(past_kv.key, 'shape'):
                        past_seq_len = past_kv.key.shape[2]
                        break

        total_seq_len = past_seq_len + seq_len

        # 确保 position_ids 正确 - 必须考虑缓存长度
        if position_ids is None:
            batch_size = hidden_states.shape[0]
            # position_ids 必须包含历史位置，从 past_seq_len 开始
            position_ids = torch.arange(
                past_seq_len, total_seq_len,
                dtype=torch.long,
                device=hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)

        # 使用rotary_emb计算位置编码
        # position_embeddings 是一个元组 (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 检查是否是Cache对象（如DynamicCache）
        is_cache_object = past_key_values is not None and hasattr(past_key_values, 'get_seq_length')
        original_cache = past_key_values if is_cache_object else None

        # 通过各层处理
        for i, layer in enumerate(self.layers):
            if isinstance(layer, IdentityBlock):
                hidden_states = layer(hidden_states)
            else:
                cache_position = torch.arange(
                    past_seq_len, past_seq_len + seq_len,
                    dtype=torch.long,
                    device=hidden_states.device
                )
                
                if is_cache_object:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=original_cache,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

        # 最后层的归一化
        if self.args.shard.is_last_layer():
            hidden_states = self.norm(hidden_states)

        # 返回隐藏状态和更新后的KV缓存
        return hidden_states, past_key_values if use_cache else None


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = getattr(args, 'model_type', 'llama')
        self.model = Llama3Model(args)

        # 只在最后层创建输出层
        if args.shard.is_last_layer():
            if not args.tie_word_embeddings:
                self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 解析返回值
        if isinstance(outputs, tuple):
            hidden_states, updated_cache = outputs
        else:
            hidden_states = outputs
            updated_cache = past_key_values if use_cache else None

        class ModelOutput:
            def __init__(self, hidden_states, logits=None, past_key_values=None):
                self.hidden_states = hidden_states
                self.logits = logits
                self.past_key_values = past_key_values

        # 只在最后层生成 logits
        if self.args.shard.is_last_layer():
            if self.args.tie_word_embeddings:
                logits = torch.nn.functional.linear(hidden_states, self.model.embed_tokens.weight)
            else:
                logits = self.lm_head(hidden_states)
            return ModelOutput(hidden_states, logits, updated_cache)

        return ModelOutput(hidden_states, None, updated_cache)

    def sanitize(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """过滤权重，只保留当前分片需要的权重"""
        shard_state_dict = {}

        for key, value in weights.items():
            # 跳过不需要的权重
            if "rotary_emb.inv_freq" in key:
                continue

            # 处理层权重
            if key.startswith('model.layers.'):
                try:
                    layer_num = int(key.split('.')[2])
                    if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                        shard_state_dict[key] = value
                except (IndexError, ValueError):
                    # 如果无法解析层号，跳过
                    continue
            # 处理嵌入层权重
            elif key.startswith('model.embed_tokens'):
                # 第一层需要embed_tokens进行嵌入
                # 最后一层在tie_word_embeddings=True时也需要embed_tokens的权重用于生成logits
                if self.args.shard.is_first_layer() or (self.args.shard.is_last_layer() and self.args.tie_word_embeddings):
                    shard_state_dict[key] = value
            # 处理独立输出层权重
            elif key.startswith('lm_head'):
                # 只在最后一层且不使用权重共享时需要lm_head
                if self.args.shard.is_last_layer() and not self.args.tie_word_embeddings:
                    shard_state_dict[key] = value
            # 处理归一化层权重
            elif key.startswith('model.norm'):
                if self.args.shard.is_last_layer():
                    shard_state_dict[key] = value

        return shard_state_dict

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return getattr(self.args, 'head_dim', None) or self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return getattr(self.args, 'num_key_value_heads', self.args.num_attention_heads)

    # 确保类可以被正确导入


__all__ = ['Model', 'ModelArgs']