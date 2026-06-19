#!/usr/bin/env python3
"""
Fara-7B (基于 Qwen2.5-VL) 分片模型实现
支持分布式推理架构
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

import logging
logger = logging.getLogger(__name__)
logging.getLogger("exo.inference.pytorch.qwen2_5vl.qwen2_5vl").setLevel(logging.WARNING)


class ShardedQwen2_5VlTextModel(nn.Module):
    """
    分片版本的 Qwen2.5-VL 文本模型
    只初始化当前分片需要的层，节省GPU内存
    """

    def __init__(self, config, shard=None):
        """
        Args:
            config: 模型配置 (Qwen2_5_VLTextConfig)
            shard: Shard对象，定义分片范围。如果为None，则加载所有层
        """
        super().__init__()
        self.config = config
        self.shard = shard

        # 确保config有_attn_implementation属性
        if not hasattr(self.config, '_attn_implementation') or self.config._attn_implementation is None:
            self.config._attn_implementation = "eager"

        self.padding_idx = getattr(config, 'pad_token_id', 151643)
        self.vocab_size = getattr(config, 'vocab_size', 152064)

        # 根据分片配置决定是否创建嵌入层
        if shard is None or shard.is_first_layer():
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
            logger.debug(f"[ShardedFaraTextModel] 创建embed_tokens层")
        else:
            self.embed_tokens = None
            logger.debug(f"[ShardedFaraTextModel] 跳过embed_tokens层（非首分片）")

        # 根据分片配置创建Transformer层
        if shard is None:
            # 无分片模式：创建所有层
            start_layer = 0
            end_layer = config.num_hidden_layers - 1
            layer_indices = range(config.num_hidden_layers)
            logger.debug(f"[ShardedFaraTextModel] 无分片模式，创建所有{config.num_hidden_layers}层")
        else:
            # 分片模式：只创建当前分片的层
            start_layer = shard.start_layer
            end_layer = shard.end_layer
            layer_indices = range(start_layer, end_layer + 1)
            logger.debug(f"[ShardedFaraTextModel] 分片模式，创建层{start_layer}-{end_layer}")

        # 使用官方Qwen2.5VL的层实现
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in layer_indices]
        )

        # 存储实际的层索引映射
        self.layer_idx_map = {i: idx for i, idx in enumerate(layer_indices)}
        self.start_layer = start_layer
        self.end_layer = end_layer

        # 根据分片配置决定是否创建最终的norm层
        if shard is None or shard.is_last_layer():
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            logger.debug(f"[ShardedFaraTextModel] 创建最终norm层")
        else:
            self.norm = None
            logger.debug(f"[ShardedFaraTextModel] 跳过最终norm层（非尾分片）")

        # 始终创建rotary_emb（每个分片都需要）
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif "RMSNorm" in module.__class__.__name__ or "LayerNorm" in module.__class__.__name__:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def is_first_layer(self):
        """判断是否为第一个分片"""
        return self.shard is None or self.shard.is_first_layer()

    def is_last_layer(self):
        """判断是否为最后一个分片"""
        return self.shard is None or self.shard.is_last_layer()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_causal_mask(
        self,
        input_embeds,
        attention_mask=None,
        cache_position=None,
        past_key_values=None,
    ):
        """创建因果掩码"""
        batch_size, seq_len = input_embeds.shape[:2]
        device = input_embeds.device

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        total_seq_len = past_seen_tokens + seq_len

        # 创建4D因果掩码
        if seq_len == 1:
            # 单token生成，不需要因果掩码
            return None

        # 创建因果掩码
        causal_mask = torch.zeros((batch_size, 1, seq_len, total_seq_len), device=device)

        # 应用因果约束
        if seq_len > 1:
            local_causal = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
            causal_mask[:, :, :, past_seen_tokens:] = local_causal.unsqueeze(0).unsqueeze(0)

        return causal_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """前向传播"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 初始化KV缓存
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # 首分片需要处理输入嵌入
        if self.is_first_layer():
            if inputs_embeds is None:
                if self.embed_tokens is None:
                    raise ValueError("embed_tokens is None but this is the first layer")
                inputs_embeds = self.embed_tokens(input_ids)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            # 处理position_ids (Qwen2.5-VL使用mRoPE)
            if position_ids is None:
                # mRoPE: 3D position_ids (3, batch, seq_len)
                position_ids_2d = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
                position_ids = position_ids_2d[None, ...].expand(3, -1, -1)

            # 创建因果掩码
            if attention_mask is None or attention_mask.dim() != 4:
                attention_mask = self._prepare_causal_mask(
                    inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                )

            hidden_states = inputs_embeds
        else:
            # 非首分片：直接接收hidden_states作为输入
            if inputs_embeds is None:
                raise ValueError("Non-first shard requires inputs_embeds (hidden_states from previous shard)")
            hidden_states = inputs_embeds

            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

            # 处理position_ids
            if position_ids is None:
                position_ids_2d = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_len,
                    dtype=torch.long, device=hidden_states.device
                )
                position_ids_2d = position_ids_2d.unsqueeze(0).expand(batch_size, -1)
                position_ids = position_ids_2d[None, ...].expand(3, -1, -1)

            # 创建因果掩码
            if attention_mask is None:
                if seq_len > 1:
                    # 多token输入：需要因果掩码
                    total_seq_len = past_seen_tokens + seq_len
                    attention_mask = torch.zeros((batch_size, 1, seq_len, total_seq_len), device=hidden_states.device)
                    local_causal = torch.triu(
                        torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device),
                        diagonal=1
                    )
                    attention_mask[:, :, :, past_seen_tokens:] = local_causal.unsqueeze(0).unsqueeze(0)
                else:
                    # 单token输入：不需要因果掩码，使用None让模型自动处理
                    attention_mask = None

            if cache_position is None:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device
                )

        # 创建位置嵌入 (mRoPE)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 通过Transformer层
        for layer_idx, decoder_layer in enumerate(self.layers):
            actual_layer_idx = self.layer_idx_map[layer_idx]
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # Qwen2_5_VLDecoderLayer 返回 tuple，第一个元素是 hidden_states
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        # 尾分片应用最终的norm
        if self.is_last_layer():
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def sanitize(self, state_dict):
        """权重过滤方法 - 只保留当前分片需要的权重"""
        sanitized = {}

        for key, value in state_dict.items():
            # 处理embed_tokens权重 - 只有首分片需要
            if "embed_tokens" in key:
                if self.is_first_layer():
                    new_key = key.replace("model.language_model.embed_tokens.", "embed_tokens.")
                    new_key = new_key.replace("language_model.embed_tokens.", "embed_tokens.")
                    new_key = new_key.replace("model.embed_tokens.", "embed_tokens.")
                    sanitized[new_key] = value
                    logger.debug(f"[sanitize] 保留embed_tokens权重: {key} -> {new_key}")
                else:
                    logger.debug(f"[sanitize] 跳过embed_tokens权重（非首分片）: {key}")
                continue

            # 处理Transformer层权重
            if "layers." in key:
                # 提取层号
                parts = key.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        break

                if layer_idx is not None:
                    # 检查该层是否在当前分片范围内
                    if self.start_layer <= layer_idx <= self.end_layer:
                        # 重新映射层索引
                        new_layer_idx = layer_idx - self.start_layer
                        new_key = key.replace(f"layers.{layer_idx}.", f"layers.{new_layer_idx}.")
                        # 移除前缀
                        new_key = new_key.replace("model.language_model.", "")
                        new_key = new_key.replace("language_model.", "")
                        new_key = new_key.replace("model.", "")
                        sanitized[new_key] = value
                        logger.debug(f"[sanitize] 保留层{layer_idx}权重: {key} -> {new_key}")
                    else:
                        logger.debug(f"[sanitize] 跳过层{layer_idx}权重（不在分片范围{self.start_layer}-{self.end_layer}）: {key}")
                continue

            # 处理最终norm权重 - 只有尾分片需要
            if "norm.weight" in key and "layernorm" not in key and "input_layernorm" not in key and "post_attention_layernorm" not in key and "layers." not in key:
                if self.is_last_layer():
                    new_key = key.replace("model.language_model.norm.", "norm.")
                    new_key = new_key.replace("language_model.norm.", "norm.")
                    new_key = new_key.replace("model.norm.", "norm.")
                    sanitized[new_key] = value
                    logger.debug(f"[sanitize] 保留最终norm权重: {key} -> {new_key}")
                else:
                    logger.debug(f"[sanitize] 跳过最终norm权重（非尾分片）: {key}")
                continue

            # 处理rotary_emb权重
            if "rotary_emb" in key:
                new_key = key.replace("model.language_model.", "")
                new_key = new_key.replace("language_model.", "")
                new_key = new_key.replace("model.", "")
                sanitized[new_key] = value
                logger.debug(f"[sanitize] 保留rotary_emb权重: {key} -> {new_key}")
                continue

            # 处理lm_head权重 - 只有尾分片需要
            if "lm_head" in key:
                if self.is_last_layer():
                    logger.debug(f"[sanitize] 保留lm_head权重（在FaraModel中加载）: {key}")
                else:
                    logger.debug(f"[sanitize] 跳过lm_head权重（非尾分片）: {key}")
                continue

            # 其他权重跳过
            logger.debug(f"[sanitize] 跳过其他权重: {key}")

        logger.debug(f"[sanitize] 过滤结果: {len(state_dict)} -> {len(sanitized)}")
        return sanitized

    def load_state_dict_with_sanitize(self, state_dict, strict=True):
        """加载权重并应用sanitize过滤"""
        logger.debug(f"[ShardedFaraTextModel] 开始加载权重并应用sanitize过滤...")

        sanitized_dict = self.sanitize(state_dict)
        logger.debug(f"[ShardedFaraTextModel] 权重过滤完成: {len(state_dict)} -> {len(sanitized_dict)}")

        missing_keys, unexpected_keys = self.load_state_dict(sanitized_dict, strict=False)

        if missing_keys:
            logger.warning(f"[ShardedFaraTextModel] 缺失的权重键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"[ShardedFaraTextModel] 意外的权重键: {unexpected_keys}")

        logger.debug("[ShardedFaraTextModel] 权重加载完成")
        return missing_keys, unexpected_keys


class Qwen2_5VlModel(nn.Module):
    """完整的Qwen2.5-VL推理模型"""

    def __init__(self, config, shard=None, tokenizer_vocab_size=None):
        super().__init__()
        self.config = config
        self.shard = shard

        # 使用官方词汇表大小或提供的
        if tokenizer_vocab_size is not None:
            self.tokenizer_vocab_size = tokenizer_vocab_size
        elif hasattr(config, 'text_config'):
            self.tokenizer_vocab_size = config.text_config.vocab_size
        else:
            self.tokenizer_vocab_size = getattr(config, 'vocab_size', 152064)

        # 使用分片版本的文本模型
        if hasattr(config, 'text_config'):
            text_config = config.text_config
        else:
            text_config = config

        self.model = ShardedQwen2_5VlTextModel(text_config, shard=shard)

        # lm_head
        self.lm_head = nn.Linear(text_config.hidden_size, self.tokenizer_vocab_size, bias=False)

        # 视觉模型（如果配置中包含且是首分片）
        if hasattr(config, 'vision_config') and config.vision_config is not None and (shard is None or shard.is_first_layer()):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
            self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
            logger.debug("使用官方Qwen2_5_VisionTransformerPretrainedModel")
        else:
            self.visual = None
            if shard is not None and not shard.is_first_layer():
                logger.debug("非首分片，跳过视觉模型")
            else:
                logger.debug("未检测到视觉配置")

    def load_state_dict_with_sanitize(self, state_dict, strict=True, target_device=None, target_dtype=None):
        """加载权重并应用sanitize过滤
        
        Args:
            state_dict: 权重字典
            strict: 是否严格匹配
            target_device: 目标设备（用于视觉模型）
            target_dtype: 目标数据类型（用于视觉模型）
        """
        logger.debug("开始加载权重并应用sanitize过滤...")

        # 使用模型内置的sanitize方法过滤文本模型权重
        text_missing, text_unexpected = self.model.load_state_dict_with_sanitize(state_dict, strict=False)

        # 处理视觉模型权重（只有首分片需要）
        vision_dict = {}
        if self.model.is_first_layer() and self.visual is not None:
            for key, value in state_dict.items():
                if key.startswith("model.visual."):
                    new_key = key.replace("model.visual.", "")
                    vision_dict[new_key] = value
                elif key.startswith("visual."):
                    new_key = key.replace("visual.", "")
                    vision_dict[new_key] = value

            if vision_dict:
                # 将视觉模型权重转换到目标设备和dtype
                if target_device is not None or target_dtype is not None:
                    for k, v in vision_dict.items():
                        if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                            vision_dict[k] = v.to(device=target_device, dtype=target_dtype)
                        elif target_device is not None:
                            vision_dict[k] = v.to(device=target_device)
                
                vision_missing, vision_unexpected = self.visual.load_state_dict(vision_dict, strict=False)
                logger.debug(f"视觉模型权重加载完成: {len(vision_dict)} 个参数")
                
                # 确保视觉模型在正确的设备上
                if target_device is not None:
                    self.visual = self.visual.to(device=target_device)
                
                if vision_missing:
                    logger.warning(f"视觉模型缺失权重: {len(vision_missing)} 个")
                if vision_unexpected:
                    logger.warning(f"视觉模型意外权重: {len(vision_unexpected)} 个")
        else:
            logger.debug(f"跳过视觉模型权重（非首分片或无视觉模型）")

        # 设置权重共享：lm_head.weight = embed_tokens.weight
        # 注意：保持与目标dtype一致，避免float32转换
        if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
            print(f"[Fara] 设置权重共享: lm_head.weight = embed_tokens.weight")
        else:
            # 非首分片，从state_dict加载lm_head权重
            lm_head_key = None
            embed_tokens_key = None
            for key in state_dict.keys():
                if 'lm_head.weight' in key:
                    lm_head_key = key
                if 'embed_tokens.weight' in key:
                    embed_tokens_key = key

            if lm_head_key:
                self.lm_head.weight.data = state_dict[lm_head_key]
                print(f"[Fara] 从state_dict加载lm_head权重: {lm_head_key}")
            elif embed_tokens_key:
                self.lm_head.weight.data = state_dict[embed_tokens_key]
                print(f"[Fara] 从state_dict加载embed_tokens权重到lm_head: {embed_tokens_key}")

        logger.info("权重加载完成")
        return list(text_missing) + list(vision_dict.keys()), text_unexpected

    def _get_required_weight_keys(self):
        """获取当前分片需要的权重键列表"""
        required_keys = set()

        # 嵌入层（首分片）
        if self.model.is_first_layer():
            required_keys.update([
                'model.embed_tokens.weight',
                'model.language_model.embed_tokens.weight',
                'language_model.embed_tokens.weight'
            ])

        # Transformer层
        for layer_idx in range(self.model.start_layer, self.model.end_layer + 1):
            required_keys.add(f'model.layers.{layer_idx}.')
            required_keys.add(f'model.language_model.layers.{layer_idx}.')
            required_keys.add(f'language_model.layers.{layer_idx}.')

        # 最终norm（尾分片）
        if self.model.is_last_layer():
            required_keys.update([
                'model.norm.weight',
                'model.language_model.norm.weight',
                'language_model.norm.weight'
            ])

        # lm_head 和 embed_tokens（尾分片）
        # 注意：lm_head 通常和 embed_tokens 共享权重，所以需要同时加载
        if self.model.is_last_layer():
            required_keys.add('lm_head.weight')
            required_keys.add('embed_tokens.weight')  # 用于权重共享

        # rotary_emb（所有分片）
        required_keys.add('rotary_emb')

        # 视觉模型（首分片）
        if self.model.is_first_layer() and self.visual is not None:
            required_keys.add('model.visual.')
            required_keys.add('visual.')

        return required_keys

    def _should_load_key(self, key: str, required_patterns: set) -> bool:
        """检查是否应该加载某个权重键"""
        for pattern in required_patterns:
            # 支持多种匹配方式
            if pattern in key:
                return True
            # 特殊处理层匹配 - 检查 key 中是否包含该层
            if "layers." in pattern and "layers." in key:
                # 提取 pattern 中的层号
                import re
                pattern_match = re.search(r'layers\.(\d+)\.', pattern)
                if pattern_match:
                    layer_num = pattern_match.group(1)
                    # 检查 key 中是否包含相同的层号
                    if f"layers.{layer_num}." in key:
                        return True
        return False

    def load_pretrained_weights(self, pretrained_model_name_or_path, target_device=None, target_dtype=None, **kwargs):
        """从预训练模型加载权重 - 按需加载，减少内存占用
        
        Args:
            pretrained_model_name_or_path: 模型路径
            target_device: 目标设备，如果指定则直接加载到该设备
            target_dtype: 目标数据类型，如果指定则直接转换（仅对浮点张量）
        """
        print(f"[Fara] 从 {pretrained_model_name_or_path} 加载预训练权重...")
        print(f"[Fara] 分片范围: 层 {self.model.start_layer}-{self.model.end_layer}")
        if target_device is not None:
            print(f"[Fara] 目标设备: {target_device}, 目标精度: {target_dtype}")

        try:
            import safetensors.torch
            from safetensors import safe_open

            model_path = Path(pretrained_model_name_or_path)
            safetensors_files = sorted(model_path.glob("*.safetensors"))

            # 获取需要的权重键模式
            required_patterns = self._get_required_weight_keys()
            print(f"[Fara] 需要的权重模式: {len(required_patterns)} 个")

            if safetensors_files:
                print(f"[Fara] 发现 {len(safetensors_files)} 个safetensors文件，按需加载...")

                total_loaded = 0
                total_skipped = 0

                for file_path in safetensors_files:
                    # 使用 safe_open 按需读取，不加载整个文件
                    with safe_open(file_path, framework="pt") as f:
                        keys_in_file = list(f.keys())
                        keys_to_load = []
                        keys_skipped = []

                        for key in keys_in_file:
                            if self._should_load_key(key, required_patterns):
                                keys_to_load.append(key)
                            else:
                                keys_skipped.append(key)

                        if keys_to_load:
                            print(f"[Fara] {file_path.name}: 加载 {len(keys_to_load)} 个权重, 跳过 {len(keys_skipped)} 个")

                            # 只加载需要的权重，并直接转换到目标设备和精度
                            file_dict = {}
                            # 优先使用传入的目标设备和精度，避免二次转换
                            device = target_device if target_device is not None else next(self.parameters()).device
                            dtype = target_dtype if target_dtype is not None else next(self.parameters()).dtype
                            for key in keys_to_load:
                                tensor = f.get_tensor(key)
                                # 直接转换到目标设备和精度，减少内存占用
                                # 只对浮点类型张量进行dtype转换，保留量化权重(int8等)的原始类型
                                if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                    tensor = tensor.to(dtype=dtype, device=device)
                                else:
                                    tensor = tensor.to(device=device)
                                file_dict[key] = tensor

                            # 立即应用sanitize并加载到模型
                            self.load_state_dict_with_sanitize(file_dict, target_device=device, target_dtype=dtype)

                            # 清理临时字典，释放内存
                            del file_dict
                            import gc
                            gc.collect()

                            total_loaded += len(keys_to_load)
                        else:
                            print(f"[Fara] {file_path.name}: 跳过全部 {len(keys_skipped)} 个权重")

                        total_skipped += len(keys_skipped)

                print(f"[Fara] 权重加载完成: 加载 {total_loaded} 个, 跳过 {total_skipped} 个")

            else:
                # 回退到 PyTorch 格式
                pytorch_files = list(model_path.glob("pytorch_model*.bin"))
                if pytorch_files:
                    print(f"[Fara] 发现 {len(pytorch_files)} 个PyTorch权重文件")
                    state_dict = {}
                    device = target_device if target_device is not None else next(self.parameters()).device
                    dtype = target_dtype if target_dtype is not None else next(self.parameters()).dtype
                    for file in pytorch_files:
                        file_dict = torch.load(file, map_location='cpu')
                        # 过滤权重并转换到目标设备和dtype
                        for k, v in file_dict.items():
                            if self._should_load_key(k, required_patterns):
                                if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                    state_dict[k] = v.to(device=device, dtype=dtype)
                                else:
                                    state_dict[k] = v.to(device=device)
                    self.load_state_dict_with_sanitize(state_dict, target_device=device, target_dtype=dtype)
                else:
                    raise ValueError(f"未找到权重文件: {model_path}")

            # 注意：模型已在创建时移动到目标设备，无需再次移动
            # 视觉模型已在 load_state_dict_with_sanitize 中处理

            print("[Fara] 预训练权重加载成功！")
            return True

        except Exception as e:
            print(f"[Fara] 加载预训练权重失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """FaraModel 前向传播

        1. 调用 ShardedFaraTextModel 获取 hidden_states
        2. 如果是尾分片，通过 lm_head 计算 logits
        """
        # 处理视觉输入（仅首分片）
        if pixel_values is not None and self.visual is not None and self.model.is_first_layer():
            # 使用视觉模型处理图像
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # TODO: 将 image_embeds 与 text_embeds 合并
            # 这里简化处理，假设视觉特征已经通过其他方式处理
            pass

        # 调用文本模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # 尾分片：计算 logits
        if self.model.is_last_layer():
            logits = self.lm_head(hidden_states)
            # 返回类似 CausalLMOutputWithPast 的结构
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=None,
            )
        else:
            # 中间分片：返回 hidden_states
            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=outputs.past_key_values,
                hidden_states=None,
                attentions=None,
            )

    def set_decoder(self, decoder):
        self.model = decoder

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor):
        """处理图像特征"""
        if self.visual is None:
            raise ValueError("视觉模型未初始化")

        # 转换数据类型
        visual_dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.to(visual_dtype)

        # 通过视觉模型处理
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        # 根据grid_thw分割图像嵌入
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)

        return image_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """前向传播"""
        if inputs_embeds is None and pixel_values is not None and self.visual is not None:
            # 处理图像输入
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)

            # 获取文本嵌入
            if input_ids is not None and self.model.embed_tokens is not None:
                inputs_embeds = self.model.embed_tokens(input_ids)

                # 合并图像和文本嵌入
                # 找到图像token的位置并替换
                image_token_id = getattr(self.config, 'image_token_id', 151655)
                image_mask = (input_ids == image_token_id)

                if image_mask.any():
                    # 替换图像token嵌入
                    for batch_idx in range(input_ids.shape[0]):
                        batch_image_embeds = image_embeds[batch_idx] if batch_idx < len(image_embeds) else image_embeds[0]
                        image_positions = image_mask[batch_idx].nonzero(as_tuple=True)[0]

                        for i, pos in enumerate(image_positions):
                            if i < batch_image_embeds.shape[0]:
                                inputs_embeds[batch_idx, pos] = batch_image_embeds[i]

        # 通过文本模型
        outputs = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # 尾分片计算logits
        if self.model.is_last_layer():
            logits = self.lm_head(hidden_states)
        else:
            logits = hidden_states

        if return_dict:
            return {
                'logits': logits,
                'past_key_values': outputs.past_key_values,
                'hidden_states': hidden_states,
            }
        else:
            return logits
