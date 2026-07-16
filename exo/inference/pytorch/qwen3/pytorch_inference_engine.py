#!/usr/bin/env python3
"""
完整的 Qwen3 推理引擎 - 类似 pytorch_inference_engine.py
整合所有组件，提供完整的推理接口，支持 exo 框架标准
"""

import asyncio
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import json
import time
from collections import OrderedDict
import logging
import numpy as np
from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.inference.shard import Shard
from .stateful_model import ModelState
from exo import DEBUG

logger = logging.getLogger(__name__)


class PyTorchQwen3InferenceEngine(InferenceEngine):
    """完整的 Qwen3 推理引擎 - 支持 exo 框架标准接口"""

    def __init__(self, shard_downloader: ShardDownloader, **kwargs):
        super().__init__()  # 调用父类构造函数
        self.shard_downloader = shard_downloader
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = self._get_best_device("auto")
        self.shard = None
        self.use_amp = torch.cuda.is_available() or torch.backends.mps.is_available()
        self.states = OrderedDict()
        # 状态管理 - 使用ModelState模式
        self.states = OrderedDict()  # 存储ModelState对象

        # BF16优化支持
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.use_bf16:
            logger.info("[OK] BF16 detected, enabling BF16 optimized inference")
        else:
            logger.info("ℹ️ 未检测到BF16支持，使用默认精度")

        # 模型路径
        self.model_path = None
        # 创建异步锁用于保护模型分片加载，防止竞态条件
        self._shard_lock = asyncio.Lock()

    def _get_best_device(self, device_hint: str = "auto") -> torch.device:
        """自动选择最佳设备 - 优先使用GPU"""
        if device_hint == "auto":
            # 优先使用CUDA GPU
            if torch.cuda.is_available():
                device_name = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                logger.info(f"优先使用GPU: {gpu_name} ({memory_gb:.1f}GB)")
            # 其次使用Apple Silicon GPU
            elif torch.backends.mps.is_available():
                device_name = "mps"
                logger.info("使用Apple Silicon GPU (MPS)")
            else:
                device_name = "cpu"
                logger.info("回退到CPU推理")
        else:
            device_name = device_hint

        return torch.device(device_name)



    async def ensure_shard(self, shard: Shard):
        # ✅ 支持多实例模型的智能匹配（基于 base_model_id + instance_id + 层范围）
        if self.shard is not None:
            # 1️⃣ 完全匹配（最快路径）
            if self.shard == shard:
                return
            
            # 2️⃣ 多实例智能匹配：忽略 model_id 中的实例后缀差异
            if (hasattr(self.shard, 'base_model_id') and hasattr(shard, 'base_model_id') and
                hasattr(self.shard, 'instance_id') and hasattr(shard, 'instance_id')):
                is_match = (
                    self.shard.base_model_id == shard.base_model_id and
                    self.shard.instance_id == shard.instance_id and
                    self.shard.start_layer == shard.start_layer and
                    self.shard.end_layer == shard.end_layer and
                    self.shard.n_layers == shard.n_layers
                )
                if is_match:
                    logging.info(f"✅ [ensure_shard] 多实例智能匹配成功: "
                               f"self={self.shard.model_id}::{self.shard.instance_id} "
                               f"<-> req={shard.model_id}::{shard.instance_id}")
                    return
                
                # 3️⃣ 兼容性回退：只检查基础模型和层范围（忽略实例ID差异）
                if (self.shard.base_model_id == shard.base_model_id and
                    self.shard.start_layer == shard.start_layer and
                    self.shard.end_layer == shard.end_layer and
                    self.shard.n_layers == shard.n_layers):
                    logging.warning(f"⚠️ [ensure_shard] 基础匹配成功但实例ID不同: "
                                   f"self={self.shard.instance_id} vs req={shard.instance_id}")
                    return
        
        async with self._shard_lock:
            # 在锁内再次检查（防止并发加载）
            if self.shard is not None:
                if self.shard == shard:
                    return
                # 锁内也应用多实例智能匹配
                if (hasattr(self.shard, 'base_model_id') and hasattr(shard, 'base_model_id') and
                    self.shard.base_model_id == shard.base_model_id and
                    self.shard.start_layer == shard.start_layer and
                    self.shard.end_layer == shard.end_layer and
                    self.shard.n_layers == shard.n_layers):
                    logging.info(f"✅ [ensure_shard] 锁内多实例匹配成功，跳过重复加载")
                    return
            model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
            if self.shard != shard:
                # 使用异步 load_shard 函数进行完全异步加载
                from .sharded_utils import load_shard

                model_shard, tokenizer = await load_shard(
                    model_path=str(model_path),
                    shard=shard,
                    model_config={},
                    lazy=False,
                    executor=None,
                    device=self.device,
                    use_bf16=self.use_bf16
                )

                # 设置分词器并处理特殊token的回退
                self.tokenizer = tokenizer
                if self.tokenizer:
                    # 获取tokenizer的词汇表大小
                    tokenizer_vocab_size = getattr(self.tokenizer, 'vocab_size', None)

                    # 确保EOS token ID在有效范围内
                    if not hasattr(self.tokenizer, 'eos_token_id') or self.tokenizer.eos_token_id is None:
                        self.tokenizer.eos_token_id = 151645  # Qwen3 默认 eos_token_id
                    elif tokenizer_vocab_size and self.tokenizer.eos_token_id >= tokenizer_vocab_size:
                        if DEBUG >= 2:
                            logging.warning(
                                f"EOS token ID {self.tokenizer.eos_token_id} 超出词汇表大小 {tokenizer_vocab_size}")

                # 确保模型在正确的设备上 - 在加载阶段完成设备同步，避免推理时重复操作
                logging.info(f"准备移动模型到设备: {self.device}")
                try:
                    # 确定目标数据类型
                    target_dtype = torch.bfloat16 if self.use_bf16 and "cuda" in str(self.device) else None

                    # 一次性完成设备移动和类型转换，避免多次内存分配
                    if target_dtype is not None:
                        logging.info(f"启用BF16优化，移动模型到设备 {self.device} 并转换精度...")
                        model_shard = model_shard.to(device=self.device, dtype=target_dtype)
                        logging.info("模型已成功移动到设备并转换为BF16精度")
                    else:
                        model_shard = model_shard.to(self.device)
                        if DEBUG >= 2:
                            logging.info(f"模型已成功移动到设备: {self.device}")

                    # 验证模型设备一致性
                    if hasattr(model_shard, 'parameters'):
                        first_param = next(model_shard.parameters(), None)
                        if first_param is not None:
                            actual_device = first_param.device
                            actual_dtype = first_param.dtype
                            if DEBUG >= 2:
                                logging.info(f"验证模型设备: 期望 {self.device}, 实际 {actual_device}, 精度 {actual_dtype}")
                            if str(actual_device) != str(self.device):
                                if DEBUG >= 2:
                                    logging.warning(f"模型设备不一致: 期望 {self.device}, 实际 {actual_device}")
                                # 尝试再次移动，保留当前数据类型
                                model_shard = model_shard.to(self.device)

                except Exception as e:
                    logging.warning(f"模型设备移动或精度转换失败: {e}，使用默认设备")

                self.shard = shard
                self.model = model_shard

                # 最终验证：确保模型确实在正确的设备上
                if hasattr(self.model, 'parameters'):
                    first_param = next(self.model.parameters(), None)
                    if first_param is not None:
                        final_device = first_param.device
                        final_dtype = first_param.dtype
                        if str(final_device) != str(self.device):
                            if DEBUG >= 2:
                                logging.error(f"严重错误：模型最终设备验证失败！期望 {self.device}, 实际 {final_device}")
                            # 强制再次移动
                            self.model = self.model.to(self.device)
                        else:
                            if DEBUG >= 2:
                                logging.info(f"模型最终设备验证通过: {final_device}, 精度: {final_dtype}")

    def _ensure_model_state_device(self, state):
        """确保 ModelState 的 KV 缓存在正确的设备上 - 简化版本"""
        if state is None or not hasattr(state, 'cache') or state.cache is None:
            return state

        target_device = torch.device(self.device) if isinstance(self.device, str) else self.device

        # DynamicCache 会自动处理设备管理，只有在需要时才移动
        if hasattr(state.cache, 'to'):
            try:
                state.cache = state.cache.to(target_device)
            except Exception:
                # 如果移动失败，保持原样
                pass

        return state

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = {}

        if torch.cuda.is_available():
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024 ** 3
            memory_info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            memory_info["gpu_utilization"] = memory_info["gpu_allocated_gb"] / memory_info["gpu_total_gb"] * 100

        return memory_info

    def _manage_kv_cache(self, request_id: str, clear: bool = False) -> None:
        """增强KV缓存管理 - 支持LRU清理、内存监控和缓存压缩"""
        if clear and request_id in self.states:
            logger.info(f"清除请求 {request_id} 的状态")

            # 获取状态对象进行清理
            state = self.states[request_id]

            # 清理KV缓存内存
            if hasattr(state, 'cache') and state.cache is not None:
                # 清理DynamicCache中的张量
                if hasattr(state.cache, 'key_cache') and state.cache.key_cache:
                    for layer_cache in state.cache.key_cache:
                        if hasattr(layer_cache, 'data'):
                            del layer_cache.data
                        elif torch.is_tensor(layer_cache):
                            if layer_cache.is_cuda:
                                layer_cache.data = torch.empty(0, device='cpu')
                            del layer_cache

                if hasattr(state.cache, 'value_cache') and state.cache.value_cache:
                    for layer_cache in state.cache.value_cache:
                        if hasattr(layer_cache, 'data'):
                            del layer_cache.data
                        elif torch.is_tensor(layer_cache):
                            if layer_cache.is_cuda:
                                layer_cache.data = torch.empty(0, device='cpu')
                            del layer_cache

            # 从states中移除
            del self.states[request_id]

            # 强制垃圾回收和CUDA缓存清理
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info(f"请求 {request_id} 的状态已完全清理")

    def _get_cache_stats(self) -> dict:
        """获取KV缓存统计信息"""
        stats = {
            'total_caches': len(self.states),
            'cache_keys': list(self.states.keys()),
            'gpu_memory_gb': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'states_detail': {}
        }

        # 详细分析每个缓存状态
        for req_id, state in self.states.items():
            state_info = {
                'position': getattr(state, 'position', 0),
                'has_cache': hasattr(state, 'cache') and state.cache is not None
            }

            if state_info['has_cache']:
                cache = state.cache
                if hasattr(cache, 'key_cache') and cache.key_cache:
                    state_info['key_cache_layers'] = len(cache.key_cache)
                    if cache.key_cache and len(cache.key_cache) > 0:
                        # 获取第一个有效层的缓存形状
                        first_layer = cache.key_cache[0]
                        if torch.is_tensor(first_layer):
                            state_info['cache_shape'] = list(first_layer.shape)
                        elif hasattr(first_layer, 'shape'):
                            state_info['cache_shape'] = list(first_layer.shape)

                if hasattr(cache, 'value_cache') and cache.value_cache:
                    state_info['value_cache_layers'] = len(cache.value_cache)

            stats['states_detail'][req_id] = state_info

        return stats

    async def encode(self, shard: Shard, prompt: str, enable_thinking: bool = False) -> np.ndarray:
        """编码提示文本 - 返回numpy数组"""
        await self.ensure_shard(shard)

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # 应用 chat template（如果支持）
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        else:
            text = prompt

        # 分词
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        return tokens[0].cpu().numpy()

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """解码 token 序列 - 接收numpy数组"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # 确保 tokens 是一维的
        if isinstance(tokens, np.ndarray):
            if tokens.ndim > 1:
                tokens = tokens.squeeze()
                if tokens.ndim > 1:
                    tokens = tokens[0]

        # 解码
        try:
            text = self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            return text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            return f"<decode_error: {e}>"

    async def get_embedding(self, token_tensor: torch.Tensor, shard: Shard) -> Optional[np.ndarray]:
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
            print(f"[Qwen3 get_embedding] Error: {e}")
            return None

    async def sample(self, x: np.ndarray, temp: float = 0.7, top_p: float = 0.9,
                     top_k: int = 50, repetition_penalty: float = 1.0,
                     generated_tokens: List[int] = None) -> np.ndarray:
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
            # 贪心解码
            next_token = int(torch.argmax(logits, dim=-1).item())
        else:
            # 温度调节
            if temp != 1.0:
                logits = logits / temp

            # Top-k过滤
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(0, top_k_indices, top_k_logits)

            # Top-p过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # 概率采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        # 返回numpy数组
        result = np.array([next_token], dtype=np.int32)
        logging.info(f"[sample] END: next_token={next_token}")
        return result

    # 移除违反exo设计原则的状态共享机制
    # 每个引擎应该独立管理自己的KV缓存，通过传递隐藏状态协同工作

    async def poll_state(self, request_id: str, shard: Optional["Shard"] = None, max_states: int = 2):
        """增强LRU缓存状态管理 - 支持分片感知的缓存创建

        Args:
            request_id: 请求的唯一标识符
            shard: 分片信息，用于创建分片感知的部分缓存
            max_states: 最大缓存数量，默认为 2

        Returns:
            状态字典，包含start和cache
        """
        # 记录当前缓存状态
        initial_cache_count = len(self.states)

        # 遵循exo设计原则：每个引擎独立管理自己的KV缓存
        # 完全移除状态共享机制，只使用本地缓存
        if request_id in self.states:
            # 缓存命中 - 移到末尾表示最近使用
            self.states.move_to_end(request_id)
            cache_hit = True
            logging.info(f"LRU缓存命中: {request_id} (当前缓存数: {len(self.states)})")
        else:
            # 缓存未命中 - 需要创建新状态
            cache_hit = False

            # 检查内存使用情况
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                if current_memory > 10.0:  # 超过10GB内存使用
                    logging.warning(f"内存使用过高({current_memory:.1f}GB)，考虑清理缓存")

            # 检查是否需要驱逐最旧的缓存
            if len(self.states) >= max_states:
                # 获取最旧的请求ID
                oldest_request_id, oldest_state = self.states.popitem(last=False)
                logging.info(f"LRU缓存驱逐: {oldest_request_id} (释放内存)")

                # 清理被驱逐的状态（可选的内存优化）
                if hasattr(oldest_state, 'cache') and oldest_state.cache is not None:
                    # 清理缓存张量引用 - 适配新版transformers 4.56+的DynamicCache API
                    try:
                        # 新版transformers使用layers属性
                        if hasattr(oldest_state.cache, 'layers'):
                            oldest_state.cache.layers.clear()
                        # 旧版transformers使用key_cache/value_cache（只读属性，不能设置）
                        # 直接将cache对象置为None
                        oldest_state.cache = None
                    except (AttributeError, TypeError) as e:
                        logging.debug(f"清理缓存时遇到预期错误: {e}")
                        oldest_state.cache = None

            # 创建新状态 - 参考Tinygrad模式
            from .stateful_model import make_prompt_state

            # 获取模型配置参数
            if self.model is not None and hasattr(self.model, 'args'):
                args = self.model.args
                n_kv_heads = getattr(args, 'n_kv_heads', getattr(args, 'num_key_value_heads', 8))
                head_dim = getattr(args, 'head_dim', getattr(args, 'hidden_size', 2048) // getattr(args, 'num_attention_heads', 16))
                n_layers = getattr(args, 'n_layers', getattr(args, 'num_hidden_layers', 32))
            else:
                # 使用合理的默认值
                n_kv_heads = 8
                head_dim = 128
                n_layers = 32

            # 创建新状态 - 使用分片感知的部分KV缓存
            # 关键：每个引擎只创建自己负责的分片范围的KV缓存
            state = make_prompt_state(
                    batch_size=1,
                    max_seq_len=1024,  # 进一步减少最大序列长度以节省内存
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    n_layers=n_layers,
                    device=self.device,
                    shard=shard  # 传递分片信息，创建部分缓存
                )

            # 记录缓存创建信息 - 显示实际分片范围
            if shard:
                logging.info(f"创建分片状态: {request_id} (层 {shard.start_layer}-{shard.end_layer}, 共{shard.end_layer - shard.start_layer + 1}层)")
            else:
                logging.info(f"创建完整状态: {request_id} (所有{n_layers}层)")
            self.states[request_id] = state
            logging.info(f"LRU缓存创建: {request_id} (缓存数: {initial_cache_count} -> {len(self.states)})")

        # 获取最终状态
        state = self.states[request_id]

        # 返回状态信息，包含缓存统计和分片信息
        result = {
            "start_pos": state.position,  # 关键：使用start_pos而不是start，与Tinygrad保持一致
            "cache": state.cache,
            "cache_hit": cache_hit,
            "cache_count": len(self.states),
            "request_id": request_id,
            "shard_info": {
                "start_layer": getattr(state.cache, 'start_layer', 0),
                "end_layer": getattr(state.cache, 'end_layer', state.cache.n_layers - 1 if hasattr(state.cache, 'n_layers') else 0),
                "n_layers": getattr(state.cache, 'n_layers', 0)
            } if hasattr(state.cache, 'start_layer') else None
        }

        # 如果启用了详细日志，输出缓存统计
        if logging.getLogger().level <= logging.DEBUG:
            cache_stats = self._get_cache_stats()
            logging.debug(f"KV缓存统计: {cache_stats}")

        return result

    async def infer_prompt(self, request_id: str, shard: "Shard", prompt: str, inference_state: Optional[dict] = None) -> tuple[dict, dict]:
        """处理提示词推理 - 使用分片感知的缓存管理

        Args:
            request_id: 请求的唯一标识符
            shard: 模型分片信息
            prompt: 输入提示词
            inference_state: 推理状态字典

        Returns:
            推理结果和增强的状态信息
        """
        # 确保分片已加载
        await self.ensure_shard(shard)

        # 从inference_state中提取enable_thinking参数
        enable_thinking = None
        if inference_state and "enable_thinking" in inference_state:
            enable_thinking = inference_state["enable_thinking"]
            logging.info(f"从inference_state获取思考模式: {'启用' if enable_thinking else '禁用'}")
        else:
            # 默认禁用思考模式以避免特殊token
            enable_thinking = False
            logging.info("默认禁用思考模式")

        # 编码提示文本，传入enable_thinking参数
        input_tokens = await self.encode(shard, prompt, enable_thinking)

        # 执行张量推理
        logits, updated_state = await self.infer_tensor(
            request_id,
            shard,
            input_tokens,
            inference_state
        )

        # 返回推理结果和状态 (与基础接口保持一致，只返回2个值)
        return logits, updated_state

    async def infer_tensor(
            self,
            request_id: str,
            shard: Shard,
            input_data: np.ndarray,
            inference_state: Optional[dict] = None,
            pixel_values: Optional[np.ndarray] = None,
            image_grid_thw: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """增强KV缓存管理的推理逻辑"""
        await self.ensure_shard(shard)

        if inference_state and 'past_key_values' in inference_state:
            cache = inference_state['past_key_values']
            state_info = None
        else:
            state_info = await self.poll_state(request_id, shard=shard)
            cache = state_info['cache']

        def _infer():
            is_token_ids = input_data.dtype in [np.int32, np.int64]
            input_tensor = torch.from_numpy(input_data).long() if is_token_ids else torch.from_numpy(input_data).float()

            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            elif input_tensor.dim() == 2 and not is_token_ids:
                input_tensor = input_tensor.unsqueeze(0)

            target_device = torch.device(self.device) if isinstance(self.device, str) else self.device
            input_tensor = input_tensor.to(target_device, non_blocking=True)
            local_cache = cache

            cache_length_before = 0
            if local_cache is not None:
                if hasattr(local_cache, 'layers') and local_cache.layers:
                    if len(local_cache.layers) > 0:
                        layer = local_cache.layers[0]
                        if hasattr(layer, 'is_initialized') and layer.is_initialized and hasattr(layer, 'get_seq_length'):
                            cache_length_before = layer.get_seq_length()
                        elif hasattr(layer, 'key_cache') and layer.key_cache is not None:
                            try:
                                kc = layer.key_cache
                                if isinstance(kc, tuple) and len(kc) > 0:
                                    cache_length_before = kc[0].shape[2] if len(kc[0].shape) >= 3 else 0
                            except Exception:
                                pass
                elif hasattr(local_cache, 'key_cache') and local_cache.key_cache:
                    if len(local_cache.key_cache) > 0 and local_cache.key_cache[0] is not None:
                        cache_length_before = local_cache.key_cache[0].shape[2] if len(local_cache.key_cache[0].shape) >= 3 else 0

            model_kwargs = {
                'past_key_values': local_cache,
                'use_cache': True,
                'return_dict': True,
                'output_hidden_states': False,
                'output_attentions': False,
            }

            if local_cache is not None and hasattr(local_cache, 'to'):
                try:
                    local_cache = local_cache.to(target_device)
                except Exception:
                    pass

            if hasattr(self.model, 'config'):
                self.model.config.use_cache = True

            cache_position = None
            actual_cache_len = 0
            if local_cache is not None:
                if hasattr(local_cache, 'layers') and local_cache.layers:
                    if len(local_cache.layers) > 0:
                        layer = local_cache.layers[0]
                        if hasattr(layer, 'is_initialized') and layer.is_initialized and hasattr(layer, 'get_seq_length'):
                            actual_cache_len = layer.get_seq_length()
                elif hasattr(local_cache, 'key_cache') and local_cache.key_cache:
                    if len(local_cache.key_cache) > 0 and local_cache.key_cache[0] is not None:
                        actual_cache_len = local_cache.key_cache[0].shape[2] if len(local_cache.key_cache[0].shape) >= 3 else 0
            
            seq_length = input_tensor.shape[1]

            if actual_cache_len > 0 or (inference_state is None or 'position' not in inference_state):
                cache_position = torch.arange(
                    actual_cache_len, actual_cache_len + seq_length, device=target_device
                )
            else:
                current_position = inference_state['position']
                if actual_cache_len == 0 and current_position > 0:
                    cache_position = torch.arange(0, seq_length, device=target_device)
                else:
                    cache_position = torch.arange(
                        current_position, current_position + seq_length, device=target_device
                    )

            if cache_position is not None:
                model_kwargs['cache_position'] = cache_position

            # 遵循exo设计原则：不手动传递position_ids，让模型根据KV缓存自动计算位置
            # 每个引擎独立管理自己的KV缓存，模型会根据past_key_values.get_seq_length()计算正确的位置

            if is_token_ids:
                model_kwargs['input_ids'] = input_tensor
                model_kwargs['inputs_embeds'] = None
            else:
                model_kwargs['inputs_embeds'] = input_tensor
                model_kwargs['input_ids'] = None

            with torch.no_grad():
                if self.use_bf16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model(**model_kwargs)
                else:
                    outputs = self.model(**model_kwargs)

            old_position = self.states[request_id].position if request_id in self.states else 0
            self.states[request_id].position += input_tensor.shape[1]
            new_position = self.states[request_id].position

            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                local_cache = outputs.past_key_values
                if request_id in self.states:
                    self.states[request_id].cache = local_cache

            if request_id in self.states:
                self.states.move_to_end(request_id)

            # 提取logits或隐藏状态
            # 关键修复：处理非最后一层分片的输出（返回的是hidden_states而不是logits）
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logits = outputs.logits
            elif hasattr(outputs, 'last_hidden_state'):
                # 非最后一层分片返回的是隐藏状态
                logits = outputs.last_hidden_state
            else:
                # 向后兼容
                logits = outputs[0]

            # 关键修复：BF16输出转换为float32再转numpy，避免ScalarType错误
            if self.use_bf16 and logits.dtype == torch.bfloat16:
                logits = logits.float()  # 转换为float32
            
            # 诊断：记录输出张量的统计信息
            pass  # logging.info(f"[OUTPUT-DEBUG] Output tensor: shape={logits.shape}, dtype={logits.dtype}, mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
            if logits.dim() >= 3:
                pass  # logging.info(f"[OUTPUT-DEBUG] Last token stats: mean={logits[0,-1,:].mean().item():.6f}, std={logits[0,-1,:].std().item():.6f}")

            # 从模型中获取shard信息
            shard = None
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'shard'):
                shard = self.model.args.shard
                pass  # logging.info(f"[OUTPUT-DEBUG] Model has args.shard: {shard}")
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'shard'):
                shard = self.model.config.shard
                pass  # logging.info(f"[OUTPUT-DEBUG] Model has config.shard: {shard}")
            else:
                pass  # logging.info(f"[OUTPUT-DEBUG] Model type: {type(self.model).__name__}, attrs: {[a for a in dir(self.model) if not a.startswith('_')][:10]}")

            # 增强返回信息 - 包含分片信息
            enhanced_state = inference_state.copy() if inference_state else {}
            # 关键修复：使用更新后的位置，而不是旧的位置
            if request_id in self.states:
                final_position = self.states[request_id].position
            elif state_info is not None:
                final_position = state_info.get('start_pos', 0) + input_tensor.shape[1]
            else:
                final_position = cache_length_before + input_tensor.shape[1]

            enhanced_state.update({
                'cache_hit': True,
                'cache_length': cache_length_before,
                'position': final_position,  # 使用更新后的位置
                'request_id': request_id,
                'past_key_values': local_cache,  # 关键：返回更新后的缓存
                'shard_info': {
                    'start_layer': getattr(local_cache, 'start_layer', 0),
                    'end_layer': getattr(local_cache, 'end_layer', local_cache.n_layers - 1 if hasattr(local_cache, 'n_layers') else 0),
                    'n_layers': getattr(local_cache, 'n_layers', 0)
                } if hasattr(local_cache, 'start_layer') else None
            })

            #logging.info(f"推理完成: {request_id} (最终位置: {final_position})")

            # 推理后清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.debug(f"推理后已同步并清理GPU缓存: {request_id}")

            return logits.detach().cpu().numpy(), enhanced_state

        # 在线程池中执行推理函数，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _infer)


    async def save_checkpoint(self, shard: Shard, path: str):
        """保存模型权重到指定路径

        Args:
            shard: 模型分片
            path: 保存权重的文件路径
        """
        logger.info(f"保存检查点到: {path}")
        await self.ensure_shard(shard)  # 确保模型分片已加载

        # 在PyTorch中保存权重
        if hasattr(self.model, 'save_pretrained'):
            # 使用transformers的保存方法
            self.model.save_pretrained(path)
        else:
            # 手动保存状态字典
            torch.save(self.model.state_dict(), path)

        logger.info(f"检查点保存成功: {path}")

    async def load_checkpoint(self, shard: Shard, path: str):
        """从指定路径加载模型权重

        Args:
            shard: 模型分片
            path: 权重文件的路径（本地目录、本地文件、或 ModelScope/HuggingFace repo ID）
        
        说明:
            - 本地目录/文件: 直接加载
            - ModelScope repo ID (如 "Qwen/Qwen3-0.6B"): 通过 shard_downloader 解析为本地缓存路径
              本地缓存位置: ~/.cache/exo/downloads/{repo_id.replace('/', '--')}
        """
        logger.info(f"从 {path} 加载检查点")
        
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
                logger.info(f"检测到 Repo ID: {path}，通过 shard_downloader 解析本地路径...")
                try:
                    from pathlib import Path as PathLib
                    cache_base = PathLib.home() / ".cache" / "exo" / "downloads"
                    local_name = path.replace("/", "--")
                    
                    # 兼容多种本地目录格式：
                    # - exo 内部命名：microsoft--Fara-7B
                    # - ModelScope SDK 原始结构：microsoft/Fara-7B
                    # - ModelScope SDK HF 风格缓存：models/microsoft--Fara-7B/snapshots/master
                    candidate_paths = [
                        cache_base / local_name,
                        cache_base / path,
                        cache_base / "models" / local_name / "snapshots" / "master",
                        cache_base / "models" / path / "snapshots" / "master",
                        cache_base / "models" / local_name / "snapshots" / "main",
                        cache_base / "models" / path / "snapshots" / "main",
                    ]
                    
                    resolved_path = None
                    for candidate in candidate_paths:
                        if candidate.exists() and candidate.is_dir():
                            resolved_path = candidate
                            break
                    
                    if resolved_path is not None:
                        actual_path = str(resolved_path)
                        logger.info(f"找到本地缓存路径: {actual_path}")
                    else:
                        logger.info(f"使用 shard_downloader 下载/获取路径...")
                        actual_path = str(await self.shard_downloader.ensure_shard(shard, self.__class__.__name__))
                        logger.info(f"shard_downloader 返回路径: {actual_path}")
                except Exception as e:
                    logger.warning(f"通过 shard_downloader 获取路径失败: {e}，尝试直接 from_pretrained")
                    actual_path = path
        
        await self.ensure_shard(shard)

        try:
            if os.path.isdir(actual_path):
                logger.info(f"从本地目录加载: {actual_path}")
                if hasattr(self.model, 'from_pretrained'):
                    self.model = self.model.from_pretrained(actual_path)
                    self.model = self.model.to(self.device)
            elif os.path.isfile(actual_path):
                logger.info(f"从本地文件加载: {actual_path}")
                state_dict = torch.load(actual_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                logger.info(f"路径不存在，尝试直接 from_pretrained: {actual_path}")
                if hasattr(self.model, 'from_pretrained'):
                    self.model = self.model.from_pretrained(actual_path)
                    self.model = self.model.to(self.device)
                else:
                    raise FileNotFoundError(f"无法加载模型，路径不存在: {actual_path}")

            logger.info(f"检查点加载成功: {actual_path}")

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise
