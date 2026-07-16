"""Unlimited-OCR PyTorch KV 缓存状态管理"""
import torch
import logging
from typing import List, Tuple, Optional


def create_kv_cache(
    batch_size: int,
    max_seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
    start_layer: int = 0,
    end_layer: int = None
) -> "DynamicCache":
    """
    创建 KV 缓存，返回 DynamicCache 对象。

    为了支持分片推理，在缓存对象上附加分片范围信息。
    """
    from transformers.cache_utils import DynamicCache

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if end_layer is None:
        end_layer = n_layers - 1

    cache = DynamicCache()
    cache.start_layer = start_layer
    cache.end_layer = end_layer
    cache.n_layers = end_layer - start_layer + 1

    logging.info(f"[UnlimitedOCR] Created DynamicCache for layers {start_layer}-{end_layer} (total {cache.n_layers} layers)")

    return cache


class ModelState:
    """模型状态管理类 - 支持分片感知的层级隔离缓存"""

    def __init__(self, cache: "DynamicCache", position: int = 0, shard: Optional["Shard"] = None):
        self.cache = cache
        self.position = position
        self.shard = shard

    def get_local_layer_idx(self, global_layer_idx: int) -> int:
        """将全局层索引转换为分片本地索引"""
        if self.shard is None:
            return global_layer_idx

        if global_layer_idx < self.shard.start_layer or global_layer_idx > self.shard.end_layer:
            raise ValueError(f"Layer {global_layer_idx} not in shard range {self.shard.start_layer}-{self.shard.end_layer}")

        return global_layer_idx - self.shard.start_layer

    def get_global_layer_idx(self, local_layer_idx: int) -> int:
        """将分片本地索引转换为全局层索引"""
        if self.shard is None:
            return local_layer_idx

        shard_layer_count = self.shard.end_layer - self.shard.start_layer + 1
        if local_layer_idx < 0 or local_layer_idx >= shard_layer_count:
            raise ValueError(f"Local layer {local_layer_idx} out of range for {shard_layer_count} layers in shard")

        return local_layer_idx + self.shard.start_layer

    def is_layer_in_shard(self, layer_idx: int) -> bool:
        """检查指定层是否在当前分片范围内"""
        if self.shard is None:
            return True
        return self.shard.start_layer <= layer_idx <= self.shard.end_layer


def make_prompt_state(
    batch_size: int,
    max_seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
    shard: Optional["Shard"] = None
) -> ModelState:
    """为新的 prompt 创建初始状态"""
    start_layer = 0
    end_layer = n_layers - 1

    if shard is not None:
        start_layer = shard.start_layer
        end_layer = shard.end_layer

    cache = create_kv_cache(
        batch_size, max_seq_len, n_kv_heads, head_dim, n_layers, dtype, device,
        start_layer=start_layer, end_layer=end_layer
    )
    return ModelState(cache, position=0, shard=shard)
