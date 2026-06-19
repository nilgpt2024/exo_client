#!/usr/bin/env python3
"""
统一的Qwen3模型实现 - 合并分布式推理支持和改进组件
整合了两个文件的优势：
1. 来自qwen3.py的分布式推理支持和分片机制
2. 来自qwen3_model_core.py的改进组件和注意力掩码修复
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
import logging
import math

import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding, Qwen3DecoderLayer, Qwen3RMSNorm
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from exo.inference.shard import Shard
from exo.inference.pytorch.models.base import IdentityBlock


@dataclass
class ModelArgs(Qwen3Config):
    """
    ModelArgs类，用于配置Qwen3模型。
    直接继承Qwen3Config，符合exo标准模式。
    """
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_cache: bool = True
    use_return_dict: bool = True

    def __post_init__(self):
        """后初始化处理，设置默认值和验证分片"""
        # 处理分片参数
        if isinstance(self.shard, Shard):
            pass
        elif isinstance(self.shard, dict):
            self.shard = Shard(**self.shard)
        else:
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

        # 确保注意力实现相关属性存在
        if not hasattr(self, '_attn_implementation_internal'):
            self._attn_implementation_internal = getattr(self, '_attn_implementation', 'eager') or 'eager'

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建 ModelArgs 实例"""
        shard_config = config_dict.pop("shard", {})
        qwen3_config = Qwen3Config.from_dict(config_dict)
        instance = cls.from_config(qwen3_config)

        # 设置shard
        if isinstance(shard_config, dict):
            if not shard_config:
                instance.shard = Shard("", 0, 0, 0)
            else:
                shard_defaults = {
                    'model_id': '',
                    'start_layer': 0,
                    'end_layer': 0,
                    'n_layers': 0
                }
                for key, value in shard_config.items():
                    if key in shard_defaults:
                        shard_defaults[key] = value
                instance.shard = Shard(**shard_defaults)
        elif isinstance(shard_config, Shard):
            instance.shard = shard_config

        return instance

    @classmethod
    def from_config(cls, config: Qwen3Config):
        """从Qwen3Config实例创建ModelArgs实例"""
        instance = object.__new__(cls)

        # 复制所有Qwen3Config的属性
        for key, value in config.__dict__.items():
            if not key.startswith('_'):
                setattr(instance, key, value)

        # 设置shard默认值
        if not hasattr(instance, 'shard'):
            instance.shard = Shard("", 0, 0, 0)

        # 确保注意力实现相关属性存在
        if not hasattr(instance, '_attn_implementation_internal'):
            instance._attn_implementation_internal = getattr(instance, '_attn_implementation', 'eager') or 'eager'

        # 确保输出配置属性存在
        if not hasattr(instance, 'output_attentions'):
            instance.output_attentions = False
        if not hasattr(instance, 'output_hidden_states'):
            instance.output_hidden_states = False
        if not hasattr(instance, 'use_cache'):
            instance.use_cache = True
        if not hasattr(instance, 'use_return_dict'):
            instance.use_return_dict = True

        instance.__post_init__()
        return instance

class Qwen3Model(nn.Module):
    """统一的Qwen3模型 - 支持分片和改进组件"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        # 只在第一层或最后层(且权重共享)时创建嵌入层
        if args.shard.is_first_layer() or (args.shard.is_last_layer() and getattr(args, 'tie_word_embeddings', False)):
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # 创建旋转位置编码
        self.rotary_emb = Qwen3RotaryEmbedding(args)

        # 根据分片配置构建层
        # 关键修复：使用本地层索引（从0开始），而不是全局层索引
        # 这样 DynamicCache 的层索引就能正确对应
        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            in_shard_range = args.shard.start_layer <= i <= args.shard.end_layer
            if in_shard_range:
                # 使用本地层索引：全局索引 - 分片起始层
                local_layer_idx = i - args.shard.start_layer
                self.layers.append(Qwen3DecoderLayer(args, layer_idx=local_layer_idx))
            else:
                self.layers.append(IdentityBlock())

        # 只在最后层创建归一化层
        if self.args.shard.is_last_layer():
            self.norm = Qwen3RMSNorm(args.hidden_size, eps=args.rms_norm_eps)



    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        # 处理输入嵌入
        if inputs_embeds is None:
            if self.args.shard.is_first_layer():
                hidden_states = self.embed_tokens(input_ids)
                pass  # logging.info(f"[DIM-DEBUG] After embed_tokens: input_ids.shape={input_ids.shape}, hidden_states.shape={hidden_states.shape}")
            else:
                hidden_states = input_ids.float()
                pass  # DIM-DEBUG logs suppressed
        else:
            hidden_states = inputs_embeds

        batch_size, seq_length = hidden_states.shape[:2]
        device = hidden_states.device

        # 初始化缓存 - 与标准transformers库保持一致
        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache(config=self.args)

        # 处理缓存位置 - 使用transformers库标准方式
        # 关键修复：由于Qwen3DecoderLayer现在使用本地层索引（从0开始），
        # 缓存长度检测也应该使用本地层索引0
        if cache_position is None:
            if past_key_values is not None:
                # 使用本地层索引0来获取缓存长度（因为Qwen3DecoderLayer使用本地索引）
                local_layer_idx = 0
                # 对于新版DynamicCache，需要指定层索引
                if hasattr(past_key_values, 'layers'):
                    # 新版DynamicCache: 检查指定层是否初始化
                    if local_layer_idx < len(past_key_values.layers):
                        layer = past_key_values.layers[local_layer_idx]
                        past_seen_tokens = layer.get_seq_length() if hasattr(layer, 'is_initialized') and layer.is_initialized else 0
                    else:
                        past_seen_tokens = 0
                elif hasattr(past_key_values, 'key_cache') and past_key_values.key_cache:
                    # 旧版DynamicCache: 使用本地层索引0
                    past_seen_tokens = past_key_values.key_cache[0].shape[2] if len(past_key_values.key_cache) > 0 and past_key_values.key_cache[0] is not None else 0
                else:
                    # 尝试使用get_seq_length方法
                    try:
                        past_seen_tokens = past_key_values.get_seq_length(local_layer_idx)
                    except:
                        past_seen_tokens = 0
            else:
                past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=hidden_states.device
            )

        # 创建位置ID（如果没有提供） - 使用transformers库标准方式
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 使用transformers库的标准注意力掩码准备 - 关键修复
        from transformers.models.qwen3.modeling_qwen3 import create_causal_mask

        # 准备掩码参数 - 使用transformers库标准格式
        mask_kwargs = {
            "config": self.args,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        # 创建因果掩码 - 使用标准transformers库函数
        causal_mask = create_causal_mask(**mask_kwargs)

        # 生成位置嵌入 - 使用Transformers库的标准方式
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 通过各层处理 - 使用transformers库标准机制
        local_layer_idx = 0  # 本地层索引，从0开始
        for i, layer in enumerate(self.layers):
            if isinstance(layer, IdentityBlock):
                hidden_states = layer(hidden_states)
            else:
                # 计算本地层索引（相对于分片的起始层）
                current_local_layer_idx = local_layer_idx
                
                if use_cache and past_key_values is not None:
                    _cache_len_before = 0
                    if hasattr(past_key_values, 'layers') and past_key_values.layers:
                        # 使用本地层索引访问缓存
                        if current_local_layer_idx < len(past_key_values.layers):
                            _layer = past_key_values.layers[current_local_layer_idx]
                            if _layer is not None:
                                if hasattr(_layer, 'is_initialized') and _layer.is_initialized and hasattr(_layer, 'get_seq_length'):
                                    _cache_len_before = _layer.get_seq_length()
                                elif hasattr(_layer, 'key_cache') and _layer.key_cache is not None:
                                    try:
                                        _kc = _layer.key_cache
                                        if isinstance(_kc, tuple) and len(_kc) > 0:
                                            _cache_len_before = _kc[0].shape[2] if len(_kc[0].shape) >= 3 else 0
                                    except Exception:
                                        pass
                                # 调试：打印缓存的keys形状
                                if hasattr(_layer, 'keys') and _layer.keys is not None:
                                    pass  # logging.info(f"[SHAPE-DEBUG] Layer {i}(local={current_local_layer_idx}) BEFORE: cache.keys.shape={_layer.keys.shape}, hidden_states.shape={hidden_states.shape}")

                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions or False,
                    use_cache=use_cache or False,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                )

                if use_cache and past_key_values is not None:
                    _cache_len_after = 0
                    _total_layers = 0
                    if hasattr(past_key_values, 'layers'):
                        _total_layers = len(past_key_values.layers)
                        if current_local_layer_idx < _total_layers:
                            _layer = past_key_values.layers[current_local_layer_idx]
                            if _layer is not None:
                                if hasattr(_layer, 'is_initialized') and _layer.is_initialized and hasattr(_layer, 'get_seq_length'):
                                    _cache_len_after = _layer.get_seq_length()
                                elif hasattr(_layer, 'key_cache') and _layer.key_cache is not None:
                                    try:
                                        _kc = _layer.key_cache
                                        if isinstance(_kc, tuple) and len(_kc) > 0:
                                            _cache_len_after = _kc[0].shape[2] if len(_kc[0].shape) >= 3 else 0
                                    except Exception:
                                        pass
                    pass  # logging.info(f"[MODEL-LAYER] layer_idx={i}(global), local_idx={current_local_layer_idx}, cache_layers={_total_layers}, cache_len_around_layer: {_cache_len_before} -> {_cache_len_after}")
                
                # 递增本地层索引
                local_layer_idx += 1

        # 最后层的归一化
        if self.args.shard.is_last_layer():
            hidden_states = self.norm(hidden_states)

        # 返回隐藏状态和KV缓存（使用transformers库的标准格式）
        if use_cache:
            # 注意：在transformers库中，past_key_values是在层处理过程中内部更新的
            # 我们需要返回当前的past_key_values（现在已经被更新）
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
            )
        else:
            return hidden_states

class Model(nn.Module):
    """统一的Qwen3模型封装 - 支持分片和所有改进功能"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = getattr(args, 'model_type', 'qwen3')
        
        # 使用统一的模型实现
        self.model = Qwen3Model(args)
        self.use_transformers_impl = False
        
        # 加载generation配置
        self.generation_config = getattr(args, 'generation_config', {
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'do_sample': True
        })
        
        # 只在最后层且不共享权重时创建独立的输出层
        if args.shard.is_last_layer() and not getattr(args, 'tie_word_embeddings', False):
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
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        # 默认使用缓存如果未指定
        if use_cache is None:
            use_cache = True
        
        # 验证输入参数
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds")
        
        # 保存输入的past_key_values用于返回
        input_past_key_values = past_key_values
        
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
            cache_position=cache_position,  # 传递缓存位置参数
        )

        # 处理Qwen3Model的返回值 - 现在使用BaseModelOutputWithPast
        if hasattr(outputs, 'last_hidden_state'):
            # BaseModelOutputWithPast格式
            hidden_states = outputs.last_hidden_state
            new_past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            # 向后兼容旧的元组格式
            hidden_states, new_past_key_values = outputs
        else:
            # 兼容直接返回张量的格式
            hidden_states = outputs
            new_past_key_values = None

        # 只在最后层生成 logits
        if self.args.shard.is_last_layer():
            if getattr(self.args, 'tie_word_embeddings', False):
                # 使用共享的嵌入权重
                # 诊断：检查权重是否已正确加载（非零且非随机）
                if hasattr(self.model, 'embed_tokens'):
                    w = self.model.embed_tokens.weight
                    pass  # logging.info(f"[LOGITS-DEBUG] embed_tokens.weight: shape={w.shape}, dtype={w.dtype}, mean={w.float().mean().item():.6f}, std={w.float().std().item():.6f}, has_nan={torch.isnan(w).any().item()}")
                else:
                    pass  # logging.error(f"[LOGITS-DEBUG] ERROR: self.model does not have embed_tokens attribute!")
                logits = torch.nn.functional.linear(hidden_states, self.model.embed_tokens.weight)
                pass  # logging.info(f"[LOGITS-DEBUG] logits: shape={logits.shape}, dtype={logits.dtype}, mean={logits.float().mean().item():.6f}, std={logits.float().std().item():.6f}")
                if logits.dim() >= 2:
                    top5_vals, top5_idx = torch.topk(logits[0, -1, :], k=5)
                    pass  # logging.info(f"[LOGITS-DEBUG] top-5 tokens: {top5_idx.tolist()}, values: {[f'{v:.4f}' for v in top5_vals.tolist()]}")
            else:
                # 使用独立的输出层
                logits = self.lm_head(hidden_states)
            
            # 创建兼容Transformers接口的输出对象
            class ModelOutput:
                def __init__(self, logits, past_key_values=None):
                    self.logits = logits
                    self.past_key_values = past_key_values
                
                def __getitem__(self, key):
                    """支持字典式访问"""
                    if key == "logits":
                        return self.logits
                    elif key == "past_key_values":
                        return self.past_key_values
                    raise KeyError(f"Key {key} not found")
                
                def __contains__(self, key):
                    """支持in操作符"""
                    return key in ["logits", "past_key_values"]
                
                def get(self, key, default=None):
                    """支持get方法"""
                    try:
                        return self[key]
                    except KeyError:
                        return default
                
                def keys(self):
                    """返回所有可用的键"""
                    keys = ["logits"]
                    if self.past_key_values is not None:
                        keys.append("past_key_values")
                    return keys
                
                def items(self):
                    """返回键值对"""
                    items = [("logits", self.logits)]
                    if self.past_key_values is not None:
                        items.append(("past_key_values", self.past_key_values))
                    return items
                
                def __iter__(self):
                    """支持迭代"""
                    return iter(self.keys())
                
                def __len__(self):
                    """返回键的数量"""
                    return len(self.keys())
                
                def __repr__(self):
                    """字符串表示"""
                    return f"ModelOutput(logits_shape={self.logits.shape if hasattr(self.logits, 'shape') else 'unknown'}, past_key_values={'present' if self.past_key_values is not None else 'None'})"
                
                def to_tuple(self):
                    """转换为元组"""
                    return (self.logits, self.past_key_values)
            
            # 使用从Qwen3Model获取的新KV缓存，如果没有则使用输入的缓存
            if new_past_key_values is None:
                new_past_key_values = input_past_key_values
            
            return ModelOutput(logits, past_key_values=new_past_key_values)

        # 关键修复：即使不是最后一层，也要返回KV缓存
        # 创建一个包含hidden_states和past_key_values的输出对象
        class HiddenStateOutput:
            def __init__(self, hidden_states, past_key_values=None):
                self.last_hidden_state = hidden_states
                self.past_key_values = past_key_values
                # 为了兼容性，也提供logits属性（虽然为空）
                self.logits = None
            
            def __getitem__(self, key):
                if key == "last_hidden_state":
                    return self.last_hidden_state
                elif key == "past_key_values":
                    return self.past_key_values
                elif key == "logits":
                    return self.logits
                raise KeyError(f"Key {key} not found")
            
            def __contains__(self, key):
                return key in ["last_hidden_state", "past_key_values", "logits"]
            
            def get(self, key, default=None):
                try:
                    return self[key]
                except KeyError:
                    return default
        
        # 确保返回KV缓存，即使不是最后一层
        if new_past_key_values is None:
            new_past_key_values = input_past_key_values
        
        return HiddenStateOutput(hidden_states, past_key_values=new_past_key_values)

    def to(self, *args, **kwargs):
        """移动模型到设备 - 遵循exo标准做法"""
        super().to(*args, **kwargs)
        return self
    
    def cuda(self, *args, **kwargs):
        """移动模型到CUDA - 遵循exo标准做法"""
        super().cuda(*args, **kwargs)
        return self
    
    def cpu(self, *args, **kwargs):
        """移动模型到CPU - 遵循exo标准做法"""
        super().cpu(*args, **kwargs)
        return self

    def sanitize(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        根据当前分片的配置，从完整权重字典中提取相关的权重。
        支持改进的组件名称映射。
        
        Args:
            weights (dict): 完整的模型权重字典。

        Returns:
            dict: 当前分片需要的权重字典。
        """
        shard_state_dict = {}
        skipped_keys = []
        
        # 添加详细调试信息
        print(f"[DEBUG] sanitize: shard info start_layer={self.args.shard.start_layer}, end_layer={self.args.shard.end_layer}")
        print(f"[DEBUG] sanitize: is_first_layer={self.args.shard.is_first_layer()}, is_last_layer={self.args.shard.is_last_layer()}")
        print(f"[DEBUG] sanitize: tie_word_embeddings={getattr(self.args, 'tie_word_embeddings', 'N/A')}")

        # 统计各类权重的数量
        layer_weights = 0
        embed_tokens_weights = 0
        lm_head_weights = 0
        norm_weights = 0
        other_weights = 0
        
        for key, value in weights.items():
            # 跳过所有旋转位置编码的 inv_freq 参数
            if "rotary_emb.inv_freq" in key:
                skipped_keys.append(key)
                continue
            
            # 当权重共享时，跳过lm_head.weight
            if getattr(self.args, 'tie_word_embeddings', False) and key == "lm_head.weight":
                continue

            # 处理层权重
            if key.startswith('model.layers.'):
                try:
                    layer_num = int(key.split('.')[2])
                    if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                        shard_state_dict[key] = value
                        layer_weights += 1
                except (IndexError, ValueError):
                    continue
            # 处理嵌入层权重
            elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
                embed_tokens_weights += 1
                print(f"[KEEP] embed_tokens weight: {key}")
            elif (self.args.shard.is_last_layer() and getattr(self.args, 'tie_word_embeddings', False)) and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
                embed_tokens_weights += 1
                print(f"[KEEP] tie_word_embeddings embed_tokens: {key}")
            # 处理lm_head权重
            elif (self.args.shard.is_last_layer() and not getattr(self.args, 'tie_word_embeddings', False)) and key.startswith('lm_head'):
                shard_state_dict[key] = value
                lm_head_weights += 1
                print(f"[KEEP] lm_head weight: {key}")
            # 处理归一化层权重
            elif self.args.shard.is_last_layer() and key.startswith('model.norm'):
                shard_state_dict[key] = value
                norm_weights += 1
                print(f"[KEEP] norm weight: {key}")
            else:
                # 记录未匹配的权重
                other_weights += 1
                if other_weights <= 5:  # 只显示前5个
                    print(f"[UNMATCHED] weight: {key}")

        # 详细统计输出
        print(f"[STATS] Weight classification:")
        print(f"  layer weights: {layer_weights}")
        print(f"  embed_tokens: {embed_tokens_weights}")
        print(f"  lm_head: {lm_head_weights}")
        print(f"  norm: {norm_weights}")
        print(f"  unmatched: {other_weights}")
        print(f"  total kept: {len(shard_state_dict)}")
        print(f"  total skipped: {len(weights) - len(shard_state_dict)}")

        # 调试输出
        if skipped_keys:
            print(f"调试: 跳过 {len(skipped_keys)} 个旋转位置编码键: {skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}")
        
        print(f"Sanitize: 保留了 {len(shard_state_dict)} 个权重，跳过了 {len(weights) - len(shard_state_dict)} 个权重")
        
        # 如果权重共享，确保移除lm_head.weight（与MLX实现一致）
        if getattr(self.args, 'tie_word_embeddings', False):
            removed_lm_head = shard_state_dict.pop("lm_head.weight", None)
            if removed_lm_head is not None:
                print(f"[REMOVE] Removed shared weight lm_head.weight")
        
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


# 保持向后兼容性
__all__ = ['Model', 'ModelArgs', 'Qwen3Model']