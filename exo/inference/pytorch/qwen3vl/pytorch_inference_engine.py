#!/usr/bin/env python3
"""
Qwen3VL 推理引擎 - 支持外部传入分片
每个引擎实例只负责一个分片的推理，通过隐藏状态传递实现多分片协作
"""

import torch
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Tuple
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from transformers import AutoConfig, AutoProcessor
from exo.inference.pytorch.qwen3vl.qwen3_vl import Qwen3VLModel

class PyTorchQwen3VLInferenceEngine(InferenceEngine):
    """Qwen3VL 推理引擎 - 单分片版本
    
    每个引擎实例只加载和执行一个分片，支持通过隐藏状态传递实现多分片协作。
    这是exo框架的标准模式：外部控制分片，引擎只负责执行分配的分片。
    """

    def __init__(self, shard_downloader: ShardDownloader, model_path: str = None, **kwargs):
        super().__init__()
        self.shard_downloader = shard_downloader
        self.model_path = model_path
        self.model = None  # 单个分片模型
        self.shard = None  # 当前分片配置
        self.tokenizer = None
        self.config = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 检查 GPU 是否真正支持 BF16 (需要 Compute Capability >= 8.0)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = major * 10 + minor
            # Ampere (8.0+) 才支持 BF16, P100 (6.0) 和 V100 (7.0) 不支持
            if compute_capability >= 80 and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                print(f"[Qwen3VL] 使用 BF16 精度 (Compute Capability {major}.{minor})")
            else:
                self.dtype = torch.float16
                print(f"[Qwen3VL] 使用 FP16 精度 (Compute Capability {major}.{minor}, BF16需要>=8.0)")
        else:
            self.dtype = torch.float16
            print(f"[Qwen3VL] 使用 FP16 精度 (CPU模式)")
        self._shard_lock = asyncio.Lock()  # 添加异步锁防止并发加载
        self._executor = ThreadPoolExecutor(max_workers=1)  # 用于执行同步阻塞操作

    def _load_checkpoint_sync(self, shard: Shard, path: str):
        """同步加载检查点 - 在线程池中执行（使用 meta device 优化）
        
        Args:
            shard: 分片配置，定义了当前引擎负责处理的层范围
            path: 模型路径
        """
        import time
        import safetensors.torch
        from pathlib import Path
        
        load_start = time.time()
        
        self.model_path = path
        self.shard = shard
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        # 安全提取 tokenizer：不同模型的 processor 结构可能不同
        self.tokenizer = getattr(self.processor, 'tokenizer', None) or getattr(self.processor, '_tokenizer', None)
        if self.tokenizer is None:
            print(f"[Qwen3VL] ⚠️ processor 没有 tokenizer 属性，尝试从路径直接加载")
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            except Exception as tok_err:
                print(f"[Qwen3VL] ⚠️ Tokenizer 加载失败: {tok_err}，继续使用 processor 作为 tokenizer")
                self.tokenizer = self.processor  # 最后回退

        # 步骤1: 使用 meta device 创建模型（不分配内存，不初始化参数）
        # 这比标准初始化快数百倍（0.几秒 vs 数分钟）
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            print(f"[Qwen3VL] 使用 meta device 创建模型...")
            with torch.device("meta"):
                self.model = Qwen3VLModel(config=self.config, shard=shard)
        finally:
            torch.set_default_dtype(original_dtype)
        
        meta_time = time.time() - load_start
        print(f"[Qwen3VL] meta device 模型创建完成，耗时: {meta_time:.2f}s")

        # 步骤2: 使用 load_file 高效加载权重
        print(f"[Qwen3VL] 加载预训练权重...")
        weight_start = time.time()
        
        model_path = Path(path)
        safetensors_files = list(model_path.glob("*.safetensors"))
        
        if not safetensors_files:
            raise RuntimeError(f"未找到safetensors权重文件: {path}")
        
        # 使用 load_file 一次性加载所有权重
        state_dict = {}
        for sf in safetensors_files:
            file_weights = safetensors.torch.load_file(sf)
            state_dict.update(file_weights)
            del file_weights
        
        # 键名映射：Qwen3VL权重文件中的键名与模型参数名不一致
        # 权重文件: model.language_model.embed_tokens.weight -> 模型: model.embed_tokens.weight
        # 权重文件: model.language_model.layers.X -> 模型: model.layers.X
        # 权重文件: model.language_model.norm -> 模型: model.norm
        # 权重文件: model.visual -> 模型: visual
        # 关键：对于非首分片，还需要重映射层索引
        # 例如 node2 (层14-27): model.layers.14.xxx -> model.layers.0.xxx
        mapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model."):]
            elif key.startswith("model.visual."):
                new_key = "visual." + key[len("model.visual."):]
            elif key == "lm_head.weight" and not key.startswith("model."):
                pass
            
            if shard is not None and "model.layers." in new_key:
                parts = new_key.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        break
                
                if layer_idx is not None:
                    if shard.start_layer <= layer_idx <= shard.end_layer:
                        new_layer_idx = layer_idx - shard.start_layer
                        new_key = new_key.replace(f"layers.{layer_idx}.", f"layers.{new_layer_idx}.")
                    else:
                        continue
            
            mapped_state_dict[new_key] = value
        
        state_dict = mapped_state_dict
        
        weight_time = time.time() - weight_start
        print(f"[Qwen3VL] 权重文件加载完成，共 {len(state_dict)} 个参数，耗时: {weight_time:.2f}s")

        # 步骤3: 直接替换 meta device 上的参数到目标设备
        print(f"[Qwen3VL] 替换参数到设备: {self.device}...")
        replace_start = time.time()
        
        target_device = self.device
        loaded_count = 0
        unmatched_params = []
        
        for name, param in self.model.named_parameters():
            if name in state_dict:
                weight = state_dict[name].to(device=target_device, dtype=self.dtype)
                parts = name.split('.')
                obj = self.model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], torch.nn.Parameter(weight, requires_grad=False))
                loaded_count += 1
            elif name == "lm_head.weight" and "model.embed_tokens.weight" in state_dict:
                weight = state_dict["model.embed_tokens.weight"].to(device=target_device, dtype=self.dtype)
                parts = name.split('.')
                obj = self.model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], torch.nn.Parameter(weight, requires_grad=False))
                loaded_count += 1
            else:
                unmatched_params.append(name)
        
        if unmatched_params:
            print(f"[Qwen3VL] 未匹配的参数 ({len(unmatched_params)}): {unmatched_params[:10]}...")
        
        total_params = sum(1 for _ in self.model.named_parameters())
        print(f"[Qwen3VL] 参数匹配: {loaded_count}/{total_params} 已加载, {len(unmatched_params)} 未匹配")
        
        # 处理 meta device 上的 buffers
        for name, buffer in list(self.model.named_buffers()):
            if buffer.device.type == "meta":
                parts = name.split('.')
                obj = self.model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                if 'inv_freq' in name:
                    try:
                        if hasattr(obj, 'dim') and hasattr(obj, 'theta') and not hasattr(getattr(type(obj), 'compute_default_rope_parameters', None), '__call__'):
                            inv_freq = 1.0 / (obj.theta ** (torch.arange(0, obj.dim, 2, dtype=torch.float32, device=target_device) / obj.dim))
                            obj.register_buffer(parts[-1], inv_freq)
                        else:
                            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
                            rope_config = obj.config if hasattr(obj, 'config') else self.config.text_config
                            inv_freq, attention_scaling = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(
                                rope_config, device=target_device
                            )
                            obj.register_buffer(parts[-1], inv_freq.clone())
                            if parts[-1] == "inv_freq" and hasattr(obj, 'attention_scaling'):
                                obj.attention_scaling = attention_scaling
                    except Exception as e:
                        print(f"[Qwen3VL] 重新初始化 {name} 失败: {e}，使用空张量")
                        new_buffer = torch.empty(buffer.shape, dtype=buffer.dtype, device=target_device)
                        obj.register_buffer(parts[-1], new_buffer)
                else:
                    new_buffer = torch.empty(buffer.shape, dtype=buffer.dtype, device=target_device)
                    obj.register_buffer(parts[-1], new_buffer)
        
        replace_time = time.time() - replace_start
        print(f"[Qwen3VL] 参数替换完成，加载 {loaded_count} 个参数，耗时: {replace_time:.2f}s")
        
        # 清理权重字典
        del state_dict
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 设置为评估模式
        self.model.eval()

        total_time = time.time() - load_start
        print(f"[Qwen3VL] 模型加载完成！总耗时: {total_time:.2f}s (meta创建: {meta_time:.2f}s, 权重加载: {weight_time:.2f}s, 参数替换: {replace_time:.2f}s)")

        return True

    async def load_checkpoint(self, shard: Shard, path: str):
        """加载检查点 - 只加载指定的分片（异步包装，避免阻塞事件循环）
        
        Args:
            shard: 分片配置，定义了当前引擎负责处理的层范围
            path: 权重文件的路径（本地目录、本地文件、或 ModelScope/HuggingFace repo ID）
        
        说明:
            - 本地目录/文件: 直接加载
            - HuggingFace repo ID (如 "Qwen/Qwen3-VL-2B-Instruct"): 先检查本地缓存
              本地缓存位置: ~/.cache/exo/downloads/{repo_id.replace('/', '--')}
        """
        import os
        from pathlib import Path as PathLib
        
        print(f"[Qwen3VL] load_checkpoint被调用: shard={shard}, path={path}")
        
        actual_path = path
        
        if not os.path.exists(path):
            is_repo_id = (
                '/' in path and 
                not path.startswith('.') and 
                not path.startswith('/') and 
                '\\' not in path and
                not path.endswith('.pt') and
                not path.endswith('.bin') and
                not path.endswith('.safetensors')
            )
            
            if is_repo_id:
                print(f"[Qwen3VL] 检测到 Repo ID: {path}，解析本地缓存路径...")
                try:
                    cache_base = PathLib.home() / ".cache" / "exo" / "downloads"
                    local_dir_name = path.replace("/", "--")
                    resolved_path = cache_base / local_dir_name
                    
                    if resolved_path.exists() and resolved_path.is_dir():
                        actual_path = str(resolved_path)
                        print(f"[Qwen3VL] ✅ 找到本地缓存路径: {actual_path}")
                    else:
                        print(f"[Qwen3VL] 本地缓存不存在，使用 shard_downloader 获取路径...")
                        actual_path = str(await self.shard_downloader.ensure_shard(shard, self.__class__.__name__))
                        print(f"[Qwen3VL] shard_downloader 返回路径: {actual_path}")
                except Exception as e:
                    print(f"[Qwen3VL] ⚠️ 通过 shard_downloader 获取路径失败: {e}，尝试直接 from_pretrained")
                    actual_path = path
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_checkpoint_sync, shard, actual_path)
        print(f"[Qwen3VL] load_checkpoint完成")

    def run_forward(self, input_ids=None, inputs_embeds=None, pixel_values=None, image_grid_thw=None,
                   position_ids=None, attention_mask=None, past_key_values=None, use_cache=True, return_dict=True):
        """
        执行单个分片的前向传播

        Args:
            input_ids: 输入token（仅首分片需要）
            inputs_embeds: 输入嵌入/隐藏状态（非首分片使用）
            pixel_values: 图像像素值（仅首分片需要）
            image_grid_thw: 图像网格信息（仅首分片需要）
            position_ids: 位置编码
            attention_mask: 注意力掩码
            past_key_values: 当前分片的KV缓存
            use_cache: 是否使用缓存
            return_dict: 是否返回字典格式

        Returns:
            分片输出（尾分片返回logits，中间分片返回hidden_states）
        """
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        
        return outputs

    async def encode(self, shard: Shard, prompt: str, enable_thinking: bool = False) -> np.ndarray:
        """编码提示文本"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # 检查 prompt 是否已经包含聊天模板标记（如 <|im_start|>、<|user|> 等）
        # 如果已经格式化，直接使用，避免双重模板化
        is_already_formatted = any(marker in prompt for marker in ['<|im_start|>', '###', '`', '[INST]', '<s>[INST]'])
        
        if is_already_formatted:
            # Prompt 已经格式化，直接编码
            if self.processor is not None:
                inputs = self.processor(text=prompt, return_tensors="pt")
                input_ids = inputs['input_ids']
            else:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        elif self.processor is not None:
            # 优先使用 processor（如果可用），因为 processor 包含完整的预处理逻辑
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(text=text, return_tensors="pt")
            input_ids = inputs['input_ids']
        elif hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            input_ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
        else:
            text = prompt
            input_ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

        # 返回 2D 数组 (1, seq_len)
        return input_ids.cpu().numpy()

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """解码 token 序列"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        if isinstance(tokens, np.ndarray):
            if tokens.ndim > 1:
                tokens = tokens.squeeze()
                if tokens.ndim > 1:
                    tokens = tokens[0]

        token_list = tokens.tolist() if isinstance(tokens, np.ndarray) else list(tokens)
        return self.tokenizer.decode(token_list, skip_special_tokens=True)

    async def get_embedding(self, token_tensor: torch.Tensor, shard: Shard) -> Optional[np.ndarray]:
        """将 token 转换为嵌入向量 - 用于生成循环中的 token→embedding 转换"""
        try:
            with torch.no_grad():
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                    embedding = self.model.model.embed_tokens(token_tensor.to(self.device))
                elif hasattr(self.model, 'embed_tokens'):
                    embedding = self.model.embed_tokens(token_tensor.to(self.device))
                else:
                    return None
                return embedding.cpu().numpy()
        except Exception as e:
            print(f"[Qwen3VL get_embedding] Error: {e}")
            return None

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str,
                           inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        """执行提示推理 - 支持文本和图片"""
        if inference_state is None:
            inference_state = {}

        enable_thinking = inference_state.get("enable_thinking", False)
        image = inference_state.get("image", None)

        # 如果有图片，使用processor统一处理文本+图片（支持多轮对话）
        if image is not None and self.processor is not None:
            # 优先使用完整的 messages 列表（包含多轮对话历史和图片位置信息）
            raw_messages = inference_state.get("messages", None)
            if raw_messages is not None:
                # 将 base64 图片 URL 转换为 PIL Image 对象
                messages_with_image = []
                for msg in raw_messages:
                    msg_copy = dict(msg)
                    if isinstance(msg_copy.get("content"), list):
                        new_content = []
                        for item in msg_copy["content"]:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                new_content.append({"type": "image", "image": image})
                            else:
                                new_content.append(item)
                        msg_copy["content"] = new_content
                    messages_with_image.append(msg_copy)

                text = self.processor.apply_chat_template(messages_with_image, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=text, images=[image], return_tensors="pt")
            else:
                # 回退：只使用单轮对话（兼容旧逻辑）
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": inference_state.get("original_prompt", "描述图片")},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=text, images=[image], return_tensors="pt")
            # 保持2D形状 (1, seq_len) 与 test_qwen3vl_sharded_inference.py 一致
            input_ids = inputs['input_ids']
            tokens = input_ids.cpu().numpy()
            # 将pixel_values和image_grid_thw放入inference_state供infer_tensor使用
            inference_state['pixel_values'] = inputs.get('pixel_values', None)
            inference_state['image_grid_thw'] = inputs.get('image_grid_thw', None)
        else:
            # 纯文本推理
            # 检查 prompt 是否已经包含聊天模板标记
            is_already_formatted = any(marker in prompt for marker in ['<|im_start|>', '<|user|>', '<|assistant|>', '[INST]', '<s>[INST]'])
            
            if is_already_formatted:
                if self.processor is not None:
                    inputs = self.processor(text=prompt, return_tensors="pt")
                    tokens = inputs['input_ids'].cpu().numpy()
                else:
                    tokens = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).cpu().numpy()
            else:
                # 需要应用模板
                tokens = await self.encode(shard, prompt, enable_thinking)

        output_data, inference_state = await self.infer_tensor(request_id, shard, tokens, inference_state)

        return output_data, inference_state

    async def sample(self, x: np.ndarray, temp: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                     repetition_penalty: float = 1.0, generated_tokens: List[int] = None, shard=None) -> np.ndarray:
        """采样下一个 token"""
        if isinstance(x, np.ndarray):
            logits = torch.from_numpy(x).float()
        else:
            logits = x

        # 应用重复惩罚
        if repetition_penalty != 1.0 and generated_tokens is not None and len(generated_tokens) > 0:
            # 统计每个 token 出现的次数
            from collections import Counter
            token_counts = Counter(generated_tokens)
            for token_id, count in token_counts.items():
                if 0 <= token_id < logits.size(-1):
                    # 对重复出现的 token 降低概率
                    penalty = repetition_penalty ** count
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= penalty
                    else:
                        logits[0, token_id] *= penalty

        if temp <= 0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temp

            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token.cpu().numpy()

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray,
                          inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        """执行推理 - 支持外部传入分片和隐藏状态传递
        
        Args:
            request_id: 请求ID
            shard: 分片配置（由外部传入，定义当前引擎负责的层范围）
            input_data: 输入数据
                - 首分片: input_ids (token indices)
                - 非首分片: hidden_states (来自前一个分片的输出)
            inference_state: 推理状态，包含:
                - past_key_values: 当前分片的KV缓存
                - pixel_values: 图像像素值（首分片使用）
                - image_grid_thw: 图像网格信息（首分片使用）
                - hidden_states: 隐藏状态（用于分片间传递）
        
        Returns:
            output_data: 输出数据
                - 尾分片: logits (用于采样)
                - 非尾分片: hidden_states (传递给下一个分片)
            inference_state: 更新后的推理状态
        """
        if inference_state is None:
            inference_state = {}
        
        # 检查是否需要重新加载模型（如果shard变化）
        # 优化：只在 shard 变化或模型未加载时才获取锁（生成循环中 shard 通常不变）
        needs_reload = self.shard != shard or self.model is None
        if needs_reload:
            async with self._shard_lock:
                if self.shard != shard or self.model is None:
                    if self.model_path is None:
                        await self.ensure_shard(shard)
                    else:
                        await self.load_checkpoint(shard, self.model_path)

        # 获取当前分片的KV缓存
        past_key_values = inference_state.get('past_key_values', None)

        # 判断是否为尾分片
        is_last_layer = shard.is_last_layer()
        is_first_layer = shard.is_first_layer()

        # 准备输入
        # 检测输入类型：3D数组是embedding/hidden_states，2D数组是input_ids
        input_is_embedding = False
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 3:
                input_is_embedding = True
            elif input_data.ndim == 2 and input_data.dtype in [np.float16, np.float32, np.float64]:
                input_is_embedding = True
        
        if is_first_layer and not input_is_embedding:
            # 首分片：input_data 是 input_ids
            if isinstance(input_data, np.ndarray):
                input_ids = torch.from_numpy(input_data).long().to(self.device)
            else:
                input_ids = input_data.long().to(self.device)
            
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            batch_size, seq_len = input_ids.shape
            inputs_embeds = None
        else:
            # 非首分片或首分片但输入是embedding（生成循环中token→embedding）
            if isinstance(input_data, np.ndarray):
                inputs_embeds = torch.from_numpy(input_data).to(self.device, dtype=self.dtype)
            else:
                inputs_embeds = input_data.to(self.device, dtype=self.dtype)
            
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            
            batch_size, seq_len, hidden_size = inputs_embeds.shape
            input_ids = None

        # 计算 position_ids - 考虑 past_key_values 的序列长度
        # 修复后：DynamicCache 使用相对层索引（0-based），
        # 所以 layers[0] 就是当前分片的第一层，get_seq_length() 能正确返回序列长度
        if past_key_values is not None:
            if len(past_key_values.layers) > 0:
                past_seq_len = past_key_values.layers[0].get_seq_length()
            else:
                past_seq_len = 0
            position_ids_2d = torch.arange(past_seq_len, past_seq_len + seq_len, dtype=torch.long, device=self.device)
        else:
            past_seq_len = 0
            position_ids_2d = torch.arange(seq_len, dtype=torch.long, device=self.device)

        position_ids_2d = position_ids_2d.unsqueeze(0).expand(batch_size, -1)
        # Qwen3VL uses 4D position_ids: (text, temporal, height, width)
        # The hard coded `4` is for text, temporal, height and width.
        position_ids = position_ids_2d[None, ...].expand(4, -1, -1)
        
        # 不在引擎层创建 causal mask，而是让模型内部的 create_causal_mask 处理
        # 这样可以确保 mask 的 dtype 和格式与 SDPA 注意力实现兼容
        # create_causal_mask 在 SDPA 模式下通常会返回 None（使用 is_causal=True）
        # 这避免了 "invalid dtype for bias" 错误
        causal_mask = None

        # 处理 pixel_values 和 image_grid_thw（仅首分片且第一次调用时）
        pixel_values = None
        image_grid_thw = None
        if is_first_layer and (past_key_values is None):
            pixel_values = inference_state.get('pixel_values')
            image_grid_thw = inference_state.get('image_grid_thw')
            if pixel_values is not None:
                # 支持 numpy 数组和 torch.Tensor
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = torch.from_numpy(pixel_values).to(device=self.device, dtype=self.dtype)
                else:
                    pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            if image_grid_thw is not None:
                # 支持 numpy 数组和 torch.Tensor
                if isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw).to(device=self.device)
                else:
                    image_grid_thw = image_grid_thw.to(device=self.device)

        def _sync_forward():
            with torch.no_grad():
                return self.run_forward(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    position_ids=position_ids,
                    attention_mask=causal_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

        outputs = await self._run_in_executor(_sync_forward)

        inference_state['past_key_values'] = outputs.get('past_key_values', None)

        if is_last_layer:
            logits = outputs['logits']
            logits_last = logits[:, -1, :].cpu().float().numpy()
            return logits_last, inference_state
        else:
            hidden_states = outputs['hidden_states']
            hidden_states_np = hidden_states.cpu().float().numpy()
            return hidden_states_np, inference_state

    async def ensure_shard(self, shard: Shard):
        """确保分片已加载 - 与exo框架标准接口兼容
        
        Args:
            shard: 分片配置
        """
        # 使用异步锁确保并发安全，防止竞态条件导致重复加载
        async with self._shard_lock:
            # 双重检查，避免在等待锁期间分片已被其他协程加载
            if self.shard == shard and self.model is not None:
                return
            
            print(f"[Qwen3VL] 加载模型: {shard.model_id}")
            self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
     
            # 加载检查点
            await self.load_checkpoint(shard, self.model_path)
            print(f"[Qwen3VL] 模型加载完成: {shard.model_id}")

    async def cleanup(self):
        """清理推理引擎资源，释放GPU内存"""
        # 清理模型和分词器
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        self.shard = None
        
        # 关闭线程池
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        # 彻底清理GPU内存
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                if hasattr(torch.cuda, 'release_memory'):
                    torch.cuda.release_memory(torch.cuda.memory_reserved())
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        print("[Qwen3VL] 推理引擎资源已清理")
