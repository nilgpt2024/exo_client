"""Unlimited-OCR PyTorch 推理引擎

直接复用 transformers AutoModelForCausalLM，支持 exo 分片分布式推理。
"""
import asyncio
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from exo import DEBUG
from exo.download.shard_download import ShardDownloader
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard

logger = logging.getLogger(__name__)


class PyTorchUnlimitedOCRInferenceEngine(InferenceEngine):
    """Unlimited-OCR 推理引擎 - 基于 transformers AutoModel"""

    def __init__(self, shard_downloader: ShardDownloader, **kwargs):
        super().__init__()
        self.shard_downloader = shard_downloader
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = self._get_best_device("auto")
        self.shard = None
        self.use_amp = torch.cuda.is_available() or torch.backends.mps.is_available()
        self.states = OrderedDict()
        self._shard_lock = asyncio.Lock()

        # 默认 BF16
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.use_bf16:
            logger.info("[UnlimitedOCR] BF16 detected, enabling BF16 optimized inference")
        else:
            logger.info("[UnlimitedOCR] BF16 not supported, using default precision")

    def _get_best_device(self, device_hint: str = "auto") -> torch.device:
        """自动选择最佳设备"""
        if device_hint == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                logger.info(f"[UnlimitedOCR] 优先使用GPU: {gpu_name} ({memory_gb:.1f}GB)")
            elif torch.backends.mps.is_available():
                device_name = "mps"
                logger.info("[UnlimitedOCR] 使用 Apple Silicon GPU (MPS)")
            else:
                device_name = "cpu"
                logger.info("[UnlimitedOCR] 回退到CPU推理")
        else:
            device_name = device_hint
        return torch.device(device_name)

    async def ensure_shard(self, shard: Shard):
        """确保模型分片已加载"""
        if self.shard is not None and self.shard == shard:
            return

        async with self._shard_lock:
            if self.shard is not None and self.shard == shard:
                return

            model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)

            from .sharded_utils import load_shard

            model_shard, tokenizer = await load_shard(
                model_path=str(model_path),
                shard=shard,
                model_config={},
                lazy=False,
                executor=None,
                device=self.device,
                use_bf16=self.use_bf16,
            )

            self.tokenizer = tokenizer
            self.model = model_shard
            self.config = model_shard.config
            self.shard = shard

            # 设置默认 eos_token_id
            if self.tokenizer is not None:
                if not hasattr(self.tokenizer, "eos_token_id") or self.tokenizer.eos_token_id is None:
                    self.tokenizer.eos_token_id = 1  # Deepseek 默认 eos

            logger.info(f"[UnlimitedOCR] 分片加载完成: {shard.model_id} layers {shard.start_layer}-{shard.end_layer}")

    async def encode(self, shard: Shard, prompt: str, enable_thinking: bool = False) -> np.ndarray:
        """编码提示文本"""
        await self.ensure_shard(shard)
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # 应用 chat template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"[UnlimitedOCR] apply_chat_template 失败: {e}，使用原始 prompt")
                text = prompt
        else:
            text = prompt

        tokens = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
        return tokens[0].cpu().numpy()

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """解码 token 序列"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        if isinstance(tokens, np.ndarray):
            if tokens.ndim > 1:
                tokens = tokens.squeeze()
                if tokens.ndim > 1:
                    tokens = tokens[0]

        try:
            text = self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            return text.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as e:
            return f"<decode_error: {e}>"

    async def get_embedding(self, token_tensor: torch.Tensor, shard: Shard) -> Optional[np.ndarray]:
        """获取 token embedding"""
        await self.ensure_shard(shard)
        try:
            with torch.no_grad():
                base = self.model._get_base_model()
                if hasattr(base, "embed_tokens") and base.embed_tokens is not None:
                    embedding = base.embed_tokens(token_tensor.to(self.device))
                    return embedding.cpu().numpy()
                return None
        except Exception as e:
            logger.warning(f"[UnlimitedOCR] get_embedding 失败: {e}")
            return None

    async def sample(self, x: np.ndarray, temp: float = 0.7, top_p: float = 0.9,
                     top_k: int = 50, repetition_penalty: float = 1.0,
                     generated_tokens: List[int] = None, shard: Shard = None) -> np.ndarray:
        """采样下一个 token"""
        if isinstance(x, np.ndarray):
            logits = torch.from_numpy(x).float()
        else:
            logits = x

        if logits.dim() > 1:
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            elif logits.dim() == 2:
                pass
            else:
                logits = logits[-1]

        if logits.dim() > 1:
            logits = logits.squeeze()
            if logits.dim() > 1:
                logits = logits[0]

        if repetition_penalty != 1.0 and generated_tokens and len(generated_tokens) > 0:
            for token_id in set(generated_tokens[-50:]):
                if token_id < len(logits):
                    logits[token_id] /= repetition_penalty

        if temp <= 0:
            next_token = int(torch.argmax(logits, dim=-1).item())
        else:
            if temp != 1.0:
                logits = logits / temp

            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(0, top_k_indices, top_k_logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        return np.array([next_token], dtype=np.int32)

    async def poll_state(self, request_id: str, shard: Optional[Shard] = None, max_states: int = 2):
        """LRU 缓存状态管理"""
        if request_id in self.states:
            self.states.move_to_end(request_id)
            cache_hit = True
        else:
            cache_hit = False
            if len(self.states) >= max_states:
                oldest_request_id, oldest_state = self.states.popitem(last=False)
                if hasattr(oldest_state, "cache") and oldest_state.cache is not None:
                    oldest_state.cache = None

            from .stateful_model import make_prompt_state

            if self.model is not None and hasattr(self.model, "config"):
                cfg = self.model.config
                # 优先从 language_config 取语言模型参数
                lang_cfg = getattr(cfg, "language_config", cfg)
                n_kv_heads = getattr(lang_cfg, "num_key_value_heads", getattr(lang_cfg, "num_attention_heads", 8))
                head_dim = getattr(lang_cfg, "hidden_size", 1280) // getattr(lang_cfg, "num_attention_heads", 10)
                n_layers = getattr(lang_cfg, "num_hidden_layers", 12)
            else:
                n_kv_heads = 10
                head_dim = 128
                n_layers = 12

            state = make_prompt_state(
                batch_size=1,
                max_seq_len=1024,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                n_layers=n_layers,
                device=self.device,
                shard=shard,
            )
            self.states[request_id] = state
            logger.info(f"[UnlimitedOCR] 创建状态: {request_id}, layers {shard.start_layer}-{shard.end_layer}")

        state = self.states[request_id]
        return {
            "start_pos": state.position,
            "cache": state.cache,
            "cache_hit": cache_hit,
            "cache_count": len(self.states),
            "request_id": request_id,
        }

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> Tuple[dict, dict]:
        """处理提示词推理"""
        await self.ensure_shard(shard)
        input_tokens = await self.encode(shard, prompt)
        return await self.infer_tensor(request_id, shard, input_tokens, inference_state)

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """分片张量推理"""
        await self.ensure_shard(shard)

        if inference_state and "past_key_values" in inference_state:
            cache = inference_state["past_key_values"]
            state_info = None
        else:
            state_info = await self.poll_state(request_id, shard=shard)
            cache = state_info["cache"]

        def _infer():
            is_token_ids = input_data.dtype in [np.int32, np.int64]
            if is_token_ids:
                input_tensor = torch.from_numpy(input_data).long()
            else:
                input_tensor = torch.from_numpy(input_data).float()

            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            elif input_tensor.dim() == 2 and not is_token_ids:
                input_tensor = input_tensor.unsqueeze(0)

            target_device = torch.device(self.device) if isinstance(self.device, str) else self.device
            input_tensor = input_tensor.to(target_device, non_blocking=True)

            # 收集视觉字段（仅首分片有效）
            vision_kwargs = {}
            if inference_state:
                for key in ["pixel_values", "image_grid_thw", "image_sizes", "deepip_pixel_values", "sam_pixel_values"]:
                    val = inference_state.get(key)
                    if val is not None:
                        if isinstance(val, np.ndarray):
                            val = torch.from_numpy(val)
                        vision_kwargs[key] = val.to(target_device, dtype=self.model.dtype if hasattr(self.model, "dtype") else torch.float32)

            local_cache = cache
            if local_cache is not None and hasattr(local_cache, "to"):
                try:
                    local_cache = local_cache.to(target_device)
                except Exception:
                    pass

            # 计算 cache_position
            actual_cache_len = 0
            if local_cache is not None:
                if hasattr(local_cache, "get_seq_length"):
                    actual_cache_len = local_cache.get_seq_length()
                elif hasattr(local_cache, "key_cache") and local_cache.key_cache:
                    if len(local_cache.key_cache) > 0 and local_cache.key_cache[0] is not None:
                        actual_cache_len = local_cache.key_cache[0].shape[2] if len(local_cache.key_cache[0].shape) >= 3 else 0

            seq_length = input_tensor.shape[1]

            def _run_forward(**extra_kwargs):
                if is_token_ids:
                    return self.model(
                        input_ids=input_tensor,
                        inputs_embeds=None,
                        past_key_values=local_cache,
                        use_cache=True,
                        **vision_kwargs,
                        **extra_kwargs,
                    )
                else:
                    return self.model(
                        input_ids=None,
                        inputs_embeds=input_tensor,
                        past_key_values=local_cache,
                        use_cache=True,
                        **vision_kwargs,
                        **extra_kwargs,
                    )

            with torch.no_grad():
                # 尝试传入 cache_position（transformers 4.3x+ 推荐）
                try:
                    cache_position = torch.arange(actual_cache_len, actual_cache_len + seq_length, device=target_device)
                    outputs = _run_forward(cache_position=cache_position)
                except TypeError:
                    outputs = _run_forward()

            # 更新 KV 缓存
            updated_cache = getattr(outputs, "past_key_values", local_cache)
            if request_id in self.states:
                self.states[request_id].cache = updated_cache
                self.states[request_id].position += input_tensor.shape[1]
                self.states.move_to_end(request_id)

            # 提取 logits 或隐藏状态
            if hasattr(outputs, "logits") and outputs.logits is not None:
                out_tensor = outputs.logits
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                out_tensor = outputs.last_hidden_state
            else:
                out_tensor = outputs

            if self.use_bf16 and out_tensor.dtype == torch.bfloat16:
                out_tensor = out_tensor.float()

            enhanced_state = (inference_state or {}).copy()
            cache_hit = state_info["cache_hit"] if state_info is not None else True
            current_position = self.states[request_id].position if request_id in self.states else actual_cache_len + seq_length
            enhanced_state.update({
                "cache_hit": cache_hit,
                "position": current_position,
                "request_id": request_id,
                "past_key_values": updated_cache,
            })

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            return out_tensor.detach().cpu().numpy(), enhanced_state

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _infer)

    async def save_checkpoint(self, shard: Shard, path: str):
        """保存模型权重"""
        await self.ensure_shard(shard)
        logger.info(f"[UnlimitedOCR] 保存检查点到: {path}")
        if hasattr(self.model.model, "save_pretrained"):
            self.model.model.save_pretrained(path)
        else:
            torch.save(self.model.model.state_dict(), path)

    async def load_checkpoint(self, shard: Shard, path: str):
        """从路径加载检查点"""
        logger.info(f"[UnlimitedOCR] 从 {path} 加载检查点")
        await self.ensure_shard(shard)
        actual_path = path
        if not os.path.exists(path):
            # 尝试解析为本地缓存路径
            from pathlib import Path as PathLib
            cache_base = PathLib.home() / ".cache" / "exo" / "downloads"
            local_dir_name = path.replace("/", "--")
            resolved = cache_base / local_dir_name
            if resolved.exists():
                actual_path = str(resolved)

        if os.path.isdir(actual_path):
            from transformers import AutoModelForCausalLM
            self.model.model = AutoModelForCausalLM.from_pretrained(
                actual_path, config=self.config, trust_remote_code=True
            ).to(self.device)
        elif os.path.isfile(actual_path):
            state_dict = torch.load(actual_path, map_location=self.device)
            self.model.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"无法加载检查点: {path}")
