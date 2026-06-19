#!/usr/bin/env python3
"""
分片版本的Qwen3VL模型实现 V2
直接使用官方Qwen3VLTextModel并修改以支持分片
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLTextConfig,
    create_causal_mask
)

import logging
logger = logging.getLogger(__name__)
logging.getLogger("exo.inference.pytorch.qwen3vl.qwen3_vl").setLevel(logging.WARNING)

class ShardedQwen3VLTextModel(nn.Module):
    """
    分片版本的Qwen3VLTextModel
    只初始化当前分片需要的层，节省GPU内存
    """

    def __init__(self, config: Qwen3VLTextConfig, shard=None):
        """
        Args:
            config: Qwen3VLTextConfig配置
            shard: Shard对象，定义分片范围。如果为None，则加载所有层
        """
        super().__init__()
        self.config = config
        self.shard = shard
        
        # 确保config有_attn_implementation属性
        if not hasattr(self.config, '_attn_implementation') or self.config._attn_implementation is None:
            self.config._attn_implementation = "sdpa"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 根据分片配置决定是否创建嵌入层
        if shard is None or shard.is_first_layer():
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            logger.debug(f"[ShardedQwen3VLTextModel] 创建embed_tokens层")
        else:
            self.embed_tokens = None
            logger.debug(f"[ShardedQwen3VLTextModel] 跳过embed_tokens层（非首分片）")

        # 根据分片配置创建Transformer层
        if shard is None:
            # 无分片模式：创建所有层
            start_layer = 0
            end_layer = config.num_hidden_layers - 1
            layer_indices = range(config.num_hidden_layers)
            logger.debug(f"[ShardedQwen3VLTextModel] 无分片模式，创建所有{config.num_hidden_layers}层")
        else:
            # 分片模式：只创建当前分片的层
            start_layer = shard.start_layer
            end_layer = shard.end_layer
            layer_indices = range(start_layer, end_layer + 1)
            logger.debug(f"[ShardedQwen3VLTextModel] 分片模式，创建层{start_layer}-{end_layer}")

        # 使用官方Qwen3VLTextModel的层实现
        # 关键修复：使用相对层索引（local_layer_idx）而不是绝对层号
        # 这样 DynamicCache 的 layers 列表索引就能正确对应
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx=i) for i, _ in enumerate(layer_indices)]
        )

        # 存储实际的层索引映射
        self.layer_idx_map = {i: idx for i, idx in enumerate(layer_indices)}
        self.start_layer = start_layer
        self.end_layer = end_layer

        # 根据分片配置决定是否创建最终的norm层
        if shard is None or shard.is_last_layer():
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
            self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            logger.debug(f"[ShardedQwen3VLTextModel] 创建最终norm层")
        else:
            self.norm = None
            logger.debug(f"[ShardedQwen3VLTextModel] 跳过最终norm层（非尾分片）")

        # 始终创建rotary_emb（每个分片都需要）
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

        # 初始化权重（与官方模型保持一致）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化权重，与PreTrainedModel的_init_weights保持一致
        """
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
            # RMSNorm和LayerNorm的权重初始化为1.0
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
        """
        前向传播
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")



        # 初始化KV缓存（如果use_cache=True且past_key_values为None）
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
            logger.info(f"[forward] 初始化DynamicCache")

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

            # 处理position_ids
            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                position_ids = position_ids[1:]
            else:
                text_position_ids = position_ids[0]

            # 创建因果掩码
            # 如果传入了4D mask（由引擎创建），直接使用
            if attention_mask is not None and attention_mask.dim() == 4:
                causal_mask = attention_mask
            else:
                causal_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )
            
            attention_mask = causal_mask
            hidden_states = inputs_embeds
        else:
            # 非首分片：直接接收hidden_states作为输入
            if inputs_embeds is None:
                raise ValueError("Non-first shard requires inputs_embeds (hidden_states from previous shard)")
            hidden_states = inputs_embeds

            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            total_seq_len = past_seen_tokens + seq_len

            if cache_position is None:
                cache_position = torch.arange(
                    past_seen_tokens, total_seq_len, device=hidden_states.device
                )

            if position_ids is None:
                position_ids = cache_position.view(1, 1, -1).expand(4, batch_size, -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                position_ids = position_ids[1:]
            else:
                text_position_ids = position_ids[0]

            if attention_mask is not None and attention_mask.dim() == 4:
                pass
            else:
                attention_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )

        # 创建位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 通过Transformer层
        for layer_idx, decoder_layer in enumerate(self.layers):
            actual_layer_idx = self.layer_idx_map[layer_idx]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

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
        """
        权重过滤方法 - 只保留当前分片需要的权重
        """
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

            # 处理Transformer层权重（必须在norm.weight之前处理）
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
                        # 重新映射层索引（因为ModuleList从0开始）
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
            # 注意：这里要排除层内的norm（如q_norm, k_norm, v_norm）
            if "norm.weight" in key and "layernorm" not in key and "input_layernorm" not in key and "post_attention_layernorm" not in key and "layers." not in key:
                if self.is_last_layer():
                    # 尾分片需要加载最终的norm权重
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
                new_key = new_key.replace("model.language_model.", "")
                new_key = new_key.replace("language_model.", "")
                new_key = new_key.replace("model.", "")
                sanitized[new_key] = value
                logger.debug(f"[sanitize] 保留rotary_emb权重: {key} -> {new_key}")
                continue

            # 处理lm_head权重 - 只有尾分片需要
            if "lm_head" in key:
                if self.is_last_layer():
                    # 尾分片需要保留lm_head权重，但不在ShardedQwen3VLTextModel中加载
                    # 而是在Qwen3VLModel中加载
                    logger.debug(f"[sanitize] 保留lm_head权重（在Qwen3VLModel中加载）: {key}")
                else:
                    logger.debug(f"[sanitize] 跳过lm_head权重（非尾分片）: {key}")
                continue

            # 其他权重跳过
            logger.debug(f"[sanitize] 跳过其他权重: {key}")

        logger.debug(f"[sanitize] 过滤结果: {len(state_dict)} -> {len(sanitized)}")
        return sanitized

    def load_state_dict_with_sanitize(self, state_dict, strict=True):
        """
        加载权重并应用sanitize过滤
        只加载当前分片需要的权重
        """
        logger.debug(f"[ShardedQwen3VLTextModel] 开始加载权重并应用sanitize过滤...")

        # 应用sanitize过滤
        sanitized_dict = self.sanitize(state_dict)
        logger.debug(f"[ShardedQwen3VLTextModel] 权重过滤完成: {len(state_dict)} -> {len(sanitized_dict)}")

        # 加载过滤后的权重
        missing_keys, unexpected_keys = self.load_state_dict(sanitized_dict, strict=False)

        if missing_keys:
            logger.warning(f"[ShardedQwen3VLTextModel] 缺失的权重键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"[ShardedQwen3VLTextModel] 意外的权重键: {unexpected_keys}")

        logger.debug("[ShardedQwen3VLTextModel] 权重加载完成")
        return missing_keys, unexpected_keys

class Qwen3VLModel(nn.Module):
    """完整的Qwen3VL推理模型，使用分片版本的ShardedQwen3VLTextModel"""
    
    def __init__(self, config, shard=None, tokenizer_vocab_size=None):
        super().__init__()
        self.config = config
        self.shard = shard
        
        # 使用官方词汇表大小或提供的
        self.tokenizer_vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else config.text_config.vocab_size
        
        # 使用分片版本的文本模型 - 只初始化当前分片需要的层
        self.model = ShardedQwen3VLTextModel(config.text_config, shard=shard)
        
        # 使用权重共享：lm_head.weight = embed_tokens.weight
        # 这样可以减少参数数量，并且与官方模型保持一致
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.tokenizer_vocab_size, bias=False)
        
        # 视觉模型（如果配置中包含）
        if hasattr(config, 'vision_config'):
            # 使用 _from_config 方法初始化（与官方模型一致）
            self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
            logger.debug("使用官方Qwen3VLVisionModel (_from_config)")
            logger.debug(f"视觉模型输出维度: {config.vision_config.out_hidden_size}")
            logger.debug(f"文本模型隐藏维度: {config.text_config.hidden_size}")
        else:
            self.visual = None
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
        
        # 处理视觉模型权重
        # 视觉模型权重在state_dict中的键名是 "model.visual.xxx"
        # 但视觉模型实例(self.visual)的state_dict键名直接是 "xxx" (不带前缀)
        vision_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.visual."):
                # 将 "model.visual.xxx" 转换为 "xxx"
                new_key = key.replace("model.visual.", "")
                vision_dict[new_key] = value
            elif key.startswith("visual."):
                # 将 "visual.xxx" 转换为 "xxx"
                new_key = key.replace("visual.", "")
                vision_dict[new_key] = value
        
        if vision_dict and self.visual is not None:
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
                logger.warning(f"  前10个: {list(vision_missing)[:10]}")
            if vision_unexpected:
                logger.warning(f"视觉模型意外权重: {len(vision_unexpected)} 个")
                logger.warning(f"  前10个: {list(vision_unexpected)[:10]}")
        
        # 设置权重共享：lm_head.weight = embed_tokens.weight
        # 注意：保持与目标dtype一致，避免float32转换
        print(f"[Qwen3VL] 检查embed_tokens: hasattr={hasattr(self.model, 'embed_tokens')}, embed_tokens={self.model.embed_tokens is not None}")
        if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
            print(f"[Qwen3VL] 设置权重共享: lm_head.weight = embed_tokens.weight (形状: {self.lm_head.weight.shape})")
        else:
            # 如果没有embed_tokens（非首分片），直接从state_dict加载lm_head权重
            # 查找lm_head权重
            lm_head_key = None
            embed_tokens_key = None
            print(f"[Qwen3VL] 查找lm_head和embed_tokens权重，state_dict共有{len(state_dict)}个键")
            for key in state_dict.keys():
                if 'lm_head.weight' in key:
                    lm_head_key = key
                    print(f"[Qwen3VL] 找到lm_head权重: {key}")
                if 'embed_tokens.weight' in key:
                    embed_tokens_key = key
                    print(f"[Qwen3VL] 找到embed_tokens权重: {key}")
            
            if lm_head_key:
                self.lm_head.weight.data = state_dict[lm_head_key]
                print(f"[Qwen3VL] 从state_dict加载lm_head权重: {lm_head_key}, 形状: {self.lm_head.weight.shape}")
            elif embed_tokens_key:
                self.lm_head.weight.data = state_dict[embed_tokens_key]
                print(f"[Qwen3VL] 从state_dict加载embed_tokens权重到lm_head: {embed_tokens_key}, 形状: {self.lm_head.weight.shape}")
            else:
                print(f"[Qwen3VL] 错误: 无法找到lm_head或embed_tokens权重!")
                # 打印所有可用的键名
                matching_keys = [k for k in state_dict.keys() if 'lm_head' in k or 'embed_tokens' in k]
                print(f"[Qwen3VL] 匹配的键名: {matching_keys}")
                print(f"[Qwen3VL] 所有键名(前20个): {list(state_dict.keys())[:20]}")
        
        logger.info("权重加载完成")
        # 将 dict_keys 转换为列表
        all_missing = list(text_missing) + list(vision_dict.keys())
        return all_missing, text_unexpected
    
    def load_pretrained_weights(self, pretrained_model_name_or_path, target_device=None, target_dtype=None, **kwargs):
        """从预训练模型加载权重
        
        Args:
            pretrained_model_name_or_path: 模型路径
            target_device: 目标设备，如果指定则直接加载到该设备
            target_dtype: 目标数据类型，如果指定则直接转换（仅对浮点张量）
        """
        print(f"[Qwen3VL] 从 {pretrained_model_name_or_path} 加载预训练权重...")
        if target_device is not None:
            print(f"[Qwen3VL] 目标设备: {target_device}, 目标精度: {target_dtype}")
        
        try:
            # 尝试加载权重文件
            import safetensors.torch
            
            # 检查是否有safetensors格式的权重
            model_path = Path(pretrained_model_name_or_path)
            safetensors_files = list(model_path.glob("*.safetensors"))
            
            if safetensors_files:
                logger.debug(f"发现 {len(safetensors_files)} 个safetensors文件")
                state_dict = {}
                device = target_device if target_device is not None else next(self.parameters()).device
                dtype = target_dtype if target_dtype is not None else next(self.parameters()).dtype
                
                for file in safetensors_files:
                    logger.debug(f"加载 {file.name}...")
                    file_dict = safetensors.torch.load_file(file)
                    # 转换到目标设备和dtype
                    for k, v in file_dict.items():
                        if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                            state_dict[k] = v.to(device=device, dtype=dtype)
                        else:
                            state_dict[k] = v.to(device=device)
                    # 立即释放原始权重内存
                    del file_dict
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # 尝试加载PyTorch权重文件
                pytorch_files = list(model_path.glob("pytorch_model*.bin"))
                if pytorch_files:
                    logger.debug(f"发现 {len(pytorch_files)} 个PyTorch权重文件")
                    state_dict = {}
                    device = target_device if target_device is not None else next(self.parameters()).device
                    dtype = target_dtype if target_dtype is not None else next(self.parameters()).dtype
                    
                    for file in pytorch_files:
                        logger.debug(f"加载 {file.name}...")
                        file_dict = torch.load(file, map_location='cpu')
                        # 转换到目标设备和dtype
                        for k, v in file_dict.items():
                            if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                state_dict[k] = v.to(device=device, dtype=dtype)
                            else:
                                state_dict[k] = v.to(device=device)
                        # 立即释放原始权重内存
                        del file_dict
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # 尝试从transformers加载
                    from transformers import AutoModelForCausalLM
                    logger.debug("从transformers加载权重...")
                    
                    pretrained_model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path,
                        torch_dtype=target_dtype if target_dtype else torch.float16,
                        device_map=target_device if target_device else "cpu",
                        **kwargs
                    )
                    state_dict = pretrained_model.state_dict()
                    
                    # 清理内存
                    del pretrained_model
                    import gc
                    gc.collect()
            
            logger.debug(f"权重加载完成，共 {len(state_dict)} 个参数")
            
            # 应用sanitize并加载权重
            self.load_state_dict_with_sanitize(state_dict, target_device=target_device, target_dtype=target_dtype)
            
            # 清理临时权重内存
            del state_dict
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("[Qwen3VL] 预训练权重加载成功！")
            return True
            
        except Exception as e:
            print(f"[Qwen3VL] 加载预训练权重失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 其他方法保持不变...
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor):
        if self.visual is None:
            raise ValueError("视觉模型未初始化")
        
        visual_dtype = self.visual.dtype
        pixel_values = pixel_values.type(visual_dtype)
        
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, vision_output.deepstack_features
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        pixel_values: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        hidden_states: torch.Tensor = None,
        **kwargs
    ):
        """前向传播方法 - 支持分片推理"""
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.shard is None:
            return self._forward_full_model(
                input_ids, attention_mask, position_ids, past_key_values,
                inputs_embeds, labels, use_cache, output_attentions,
                output_hidden_states, return_dict, pixel_values,
                pixel_values_videos, image_grid_thw, video_grid_thw, **kwargs
            )
        
        return self._forward_sharded(
            input_ids, attention_mask, position_ids, past_key_values,
            inputs_embeds, use_cache, output_attentions,
            output_hidden_states, return_dict, pixel_values,
            image_grid_thw, hidden_states, **kwargs
        )
    
    def _forward_full_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        pixel_values: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        **kwargs
    ):
        """完整模型前向传播（非分片模式）"""
        
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            
            image_token_id = getattr(self.config, 'image_token_id', 151655)
            image_mask = input_ids == image_token_id
            
            if image_mask.sum() > 0:
                num_image_tokens = image_mask.sum().item()
                if image_embeds.shape[0] > num_image_tokens:
                    image_embeds = image_embeds[:num_image_tokens]
                image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask_3d, image_embeds)
        
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        
        return {
            'logits': logits,
            'past_key_values': outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }
    
    def _forward_sharded(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list = None,
        inputs_embeds: torch.Tensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        hidden_states: torch.Tensor = None,
        cache_position: torch.LongTensor = None,
        **kwargs
    ):
        """分片推理前向传播 - 使用ShardedQwen3VLTextModel"""
        
        # 设置use_cache默认值
        if use_cache is None:
            use_cache = True  # 分片推理默认使用缓存
        
        logger.info(f"[FORWARD_SHARDED] 开始")
        
        is_first_layer = self.shard.is_first_layer()
        is_last_layer = self.shard.is_last_layer()
        
        logger.info(f"[FORWARD_SHARDED] 分片: start={self.shard.start_layer}, end={self.shard.end_layer}")
        logger.info(f"[FORWARD_SHARDED] is_first={is_first_layer}, is_last={is_last_layer}")
        
        if is_first_layer:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("First shard requires either input_ids or inputs_embeds")
                inputs_embeds = self.get_input_embeddings()(input_ids)
            
            if pixel_values is not None:
                image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                
                image_token_id = getattr(self.config, 'image_token_id', 151655)
                image_mask = input_ids == image_token_id
                
                if image_mask.sum() > 0:
                    num_image_tokens = image_mask.sum().item()
                    if image_embeds.shape[0] > num_image_tokens:
                        image_embeds = image_embeds[:num_image_tokens]
                    image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask_3d, image_embeds)
            
            hidden_states = inputs_embeds
        else:
            if hidden_states is not None:
                pass
            elif inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                raise ValueError("Non-first shard requires hidden_states or inputs_embeds as input")
        
        # 创建cache_position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
        
        # attention_mask 交给模型内部的 create_causal_mask 处理
        # 不在这里手动创建 float 类型的 causal mask，避免 dtype 不匹配
        
        # 调用分片文本模型的forward方法
        outputs = self.model(
            input_ids=None,  # 已经转换为inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=hidden_states,  # 传入hidden_states作为inputs_embeds
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs.last_hidden_state
        
        if is_last_layer:
            logits = self.lm_head(hidden_states)
        else:
            logits = None
        
        if not return_dict:
            if is_last_layer:
                return (logits,) + (outputs.past_key_values if use_cache else None,)
            else:
                return (hidden_states,)
        
        result = {
            'hidden_states': hidden_states,
        }
        
        if is_last_layer:
            result['logits'] = logits
        
        if use_cache:
            result['past_key_values'] = outputs.past_key_values
        
        return result
    
    def generate(self, input_ids=None, pixel_values=None, image_grid_thw=None, 
                 max_new_tokens=50, temperature=0.3, top_k=20, top_p=0.8, 
                 repetition_penalty=1.3, do_sample=True, pad_token_id=None, 
                 eos_token_id=None, tokenizer=None, **kwargs):
        """
        生成方法 - 兼容Hugging Face生成接口
        
        Args:
            input_ids: 输入token IDs
            pixel_values: 像素值（用于视觉模型）
            image_grid_thw: 图像网格配置
            max_new_tokens: 最大新生成token数
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P采样
            repetition_penalty: 重复惩罚
            do_sample: 是否使用采样
            pad_token_id: padding token ID
            eos_token_id: 结束token ID
            tokenizer: 分词器（用于调试输出）
            **kwargs: 其他参数
            
        Returns:
            生成的token IDs
        """
        self.eval()
        
        logger.info(f"GENERATE - max_tokens={max_new_tokens}, temp={temperature}, input_shape={input_ids.shape if input_ids is not None else 'None'}")
        
        with torch.no_grad():
            generated_ids = input_ids.clone()
            batch_size = input_ids.size(0)
            past_key_values = None
            
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = self(
                        input_ids=generated_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        use_cache=True,
                        past_key_values=None,
                        **kwargs
                    )
                    past_key_values = outputs['past_key_values']
                else:
                    outputs = self(
                        input_ids=next_tokens.unsqueeze(-1),
                        pixel_values=None,
                        image_grid_thw=None,
                        use_cache=True,
                        past_key_values=past_key_values,
                        **kwargs
                    )
                    past_key_values = outputs['past_key_values']
                
                logits = outputs['logits'][:, -1, :]
                
                if outputs['logits'] is None:
                    logger.error("[GENERATE] logits为None")
                    return generated_ids
                
                # 应用重复惩罚
                if repetition_penalty != 1.0 and step > 0:
                    for i in range(batch_size):
                        for token_id in generated_ids[i]:
                            logits[i, token_id] /= repetition_penalty
                
                # 应用温度
                if temperature != 1.0:
                    logits = logits / temperature
                
                # 应用Top-K过滤
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # 应用Top-P过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 找到累积概率超过top_p的位置
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 创建掩码并应用到原始logits
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                # 转换为概率并采样
                probs = torch.softmax(logits, dim=-1)
                
                if do_sample:
                    # 采样下一个token
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    # 贪婪解码
                    next_tokens = torch.argmax(probs, dim=-1)
                
                # 添加到生成的序列
                generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
                # 检查是否生成了结束符
                if eos_token_id is not None:
                    if (next_tokens == eos_token_id).all():
                        break
            
            logger.info(f"GENERATE 完成 - 总token数: {generated_ids.size(1)}")
            return generated_ids
