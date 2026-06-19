#!/usr/bin/env python3
"""
Qwen3-TTS 模型实现 - 支持语音合成推理
基于 transformers 的 Qwen3TTSForConditionalGeneration 架构
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import logging

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding, Qwen3DecoderLayer, Qwen3RMSNorm
)
from transformers.cache_utils import DynamicCache

from exo.inference.shard import Shard
from exo.inference.pytorch.models.base import IdentityBlock

logger = logging.getLogger(__name__)


@dataclass
class Qwen3TTSArgs:
    """Qwen3-TTS 模型参数类"""
    # 模型架构参数
    vocab_size: int = 3072  # codec vocab size
    text_vocab_size: int = 151936
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 6144
    max_position_embeddings: int = 4096
    
    # 归一化参数
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-6
    
    # 注意力参数
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    
    # 分片配置
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))
    
    # 输出配置
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_cache: bool = True
    use_return_dict: bool = True
    
    # TTS 特殊配置
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    tts_pad_token_id: int = 151671
    assistant_token_id: int = 77091
    
    # 注意力实现
    _attn_implementation: str = 'eager'
    _attn_implementation_internal: str = 'eager'
    
    # 额外需要的属性
    attention_bias: bool = False
    hidden_act: str = 'silu'
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    layer_types: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Qwen3TTSArgs':
        """从配置字典创建参数"""
        # 提取 talker_config
        talker_config = config_dict.get('talker_config', {})
        
        # 合并配置
        merged_config = {**config_dict, **talker_config}
        
        # 创建参数实例
        args = cls()
        
        # 设置基本参数
        args.vocab_size = merged_config.get('vocab_size', 3072)
        args.text_vocab_size = merged_config.get('text_vocab_size', 151936)
        args.hidden_size = merged_config.get('hidden_size', 2048)
        args.num_hidden_layers = merged_config.get('num_hidden_layers', 28)
        args.num_attention_heads = merged_config.get('num_attention_heads', 16)
        args.num_key_value_heads = merged_config.get('num_key_value_heads', 8)
        args.intermediate_size = merged_config.get('intermediate_size', 6144)
        args.max_position_embeddings = merged_config.get('max_position_embeddings', 4096)
        args.rms_norm_eps = merged_config.get('rms_norm_eps', 1e-6)
        args.layer_norm_eps = merged_config.get('layer_norm_eps', 1e-6)
        args.attention_dropout = merged_config.get('attention_dropout', 0.0)
        args.rope_theta = merged_config.get('rope_theta', 10000.0)
        args.rope_scaling = merged_config.get('rope_scaling', None)
        
        # 额外参数
        args.attention_bias = merged_config.get('attention_bias', False)
        args.hidden_act = merged_config.get('hidden_act', 'silu')
        args.initializer_range = merged_config.get('initializer_range', 0.02)
        
        # 层类型
        args.layer_types = merged_config.get('layer_types', None)
        if args.layer_types is None:
            # 默认所有层都是标准注意力
            args.layer_types = ['attention'] * args.num_hidden_layers
        
        # TTS 特殊 token
        args.tts_bos_token_id = merged_config.get('tts_bos_token_id', 151672)
        args.tts_eos_token_id = merged_config.get('tts_eos_token_id', 151673)
        args.tts_pad_token_id = merged_config.get('tts_pad_token_id', 151671)
        args.assistant_token_id = merged_config.get('assistant_token_id', 77091)
        
        return args
    
    def __post_init__(self):
        """后初始化处理"""
        if isinstance(self.shard, dict):
            self.shard = Shard(**self.shard)


class Qwen3TTSModel(nn.Module):
    """
    Qwen3-TTS 模型核心实现
    支持文本到语音合成的推理
    """
    
    def __init__(self, args: Qwen3TTSArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        
        # 文本嵌入层 (文本词汇表大小: 151936)
        if args.shard.is_first_layer():
            self.embed_tokens = nn.Embedding(args.text_vocab_size, args.hidden_size)
        
        # 旋转位置编码
        self.rotary_emb = Qwen3RotaryEmbedding(args)
        
        # 根据分片配置构建层
        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            in_shard_range = args.shard.start_layer <= i <= args.shard.end_layer
            if in_shard_range:
                self.layers.append(Qwen3DecoderLayer(args, layer_idx=i))
            else:
                self.layers.append(IdentityBlock())
        
        # 最后层的归一化
        if args.shard.is_last_layer():
            self.norm = Qwen3RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            # Codec 输出头 (codec词汇表大小: 3072)
            self.codec_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """前向传播"""
        # 处理输入嵌入
        if inputs_embeds is None:
            if self.args.shard.is_first_layer():
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                # 如果不是第一层，input_ids 实际上是隐藏状态
                inputs_embeds = input_ids
        
        hidden_states = inputs_embeds
        
        # 计算位置编码
        if position_ids is None:
            if past_key_values is not None:
                past_length = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0
                position_ids = torch.arange(
                    past_length, hidden_states.shape[1] + past_length,
                    dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)
            else:
                position_ids = torch.arange(
                    hidden_states.shape[1],
                    dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)
        
        # 计算旋转位置编码
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 检查是否是Cache对象
        is_cache_object = hasattr(past_key_values, '__iter__') and not isinstance(past_key_values, (list, tuple))
        original_cache = past_key_values if is_cache_object else None
        
        # 初始化缓存列表
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        elif is_cache_object:
            # 如果是Cache对象（如DynamicCache），直接使用它
            past_key_values = [original_cache] * len(self.layers)
        
        # 通过各层处理
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if isinstance(layer, IdentityBlock):
                # 占位符层直接传递输入
                hidden_states = layer(hidden_states)
            else:
                # 真实的 Transformer 层
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    layer_idx=i,
                )
                # 获取更新后的隐藏状态
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
        
        # 最后层处理
        if self.args.shard.is_last_layer():
            hidden_states = self.norm(hidden_states)
            logits = self.codec_head(hidden_states)
        else:
            # 如果不是最后层，返回隐藏状态
            logits = hidden_states
        
        if return_dict:
            return {
                'logits': logits,
                'past_key_values': original_cache if is_cache_object else None,
                'hidden_states': hidden_states,
            }
        
        return logits


class Qwen3TTSForConditionalGeneration(nn.Module):
    """Qwen3-TTS 条件生成模型封装"""
    
    def __init__(self, args: Qwen3TTSArgs):
        super().__init__()
        self.args = args
        self.model = Qwen3TTSModel(args)
        
        # 加载 generation 配置
        self.generation_config = {
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'do_sample': True,
            'max_new_tokens': 1000,
        }
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        if return_dict:
            return outputs
        
        return outputs['logits']
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """为生成准备输入"""
        # 如果存在 past_key_values，只使用最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            'input_ids': input_ids,
            'past_key_values': past_key_values,
            'use_cache': True,
            'attention_mask': attention_mask,
        }
