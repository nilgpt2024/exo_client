"""PyTorch KV 缓存状态管理 - Fara-7B版本"""
import torch
from typing import List, Tuple, Optional
from collections import OrderedDict


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
    为指定层范围创建 KV 缓存，返回 DynamicCache 对象
    """
    from transformers.cache_utils import DynamicCache

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置默认的end_layer为最后一层
    if end_layer is None:
        end_layer = n_layers - 1

    # 验证层范围
    assert 0 <= start_layer <= end_layer < n_layers, f"Invalid layer range: {start_layer}-{end_layer} for {n_layers} layers"

    # 创建空的DynamicCache对象
    cache = DynamicCache()

    # 只为分片范围内的层创建缓存
    for layer_idx in range(start_layer, end_layer + 1):
        # 创建空的key和value张量
        key_cache = torch.empty((batch_size, n_kv_heads, 0, head_dim), dtype=dtype, device=device)
        value_cache = torch.empty((batch_size, n_kv_heads, 0, head_dim), dtype=dtype, device=device)
        cache.update(key_cache, value_cache, layer_idx=layer_idx)

    # 存储层范围信息用于后续验证
    cache.start_layer = start_layer
    cache.end_layer = end_layer
    cache.n_layers = end_layer - start_layer + 1

    return cache


class ModelState:
    """模型状态管理类 - 支持分片感知的层级隔离缓存"""
    def __init__(self, cache: List[Tuple[torch.Tensor, torch.Tensor]], position: int = 0, shard: Optional["Shard"] = None):
        self.cache = cache
        self.position = position
        self.shard = shard  # 存储分片信息用于层索引映射

    def get_local_layer_idx(self, global_layer_idx: int) -> int:
        """将全局层索引转换为分片本地索引"""
        if self.shard is None:
            return global_layer_idx

        # 验证层索引是否在分片范围内
        if global_layer_idx < self.shard.start_layer or global_layer_idx > self.shard.end_layer:
            raise ValueError(f"Layer {global_layer_idx} not in shard range {self.shard.start_layer}-{self.shard.end_layer}")

        # 计算本地索引：全局索引 - 分片起始层
        return global_layer_idx - self.shard.start_layer

    def get_global_layer_idx(self, local_layer_idx: int) -> int:
        """将分片本地索引转换为全局层索引"""
        if self.shard is None:
            return local_layer_idx

        # 验证本地索引是否有效（分片内的层数）
        shard_layer_count = self.shard.end_layer - self.shard.start_layer + 1
        if local_layer_idx < 0 or local_layer_idx >= shard_layer_count:
            raise ValueError(f"Local layer {local_layer_idx} out of range for {shard_layer_count} layers in shard")

        # 计算全局索引：本地索引 + 分片起始层
        return local_layer_idx + self.shard.start_layer

    def is_layer_in_shard(self, layer_idx: int) -> bool:
        """检查指定层是否在当前分片范围内"""
        if self.shard is None:
            return True  # 无分片信息时认为所有层都在范围内

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
    """为新的 prompt 创建初始状态，支持分片感知的部分缓存创建"""

    # 根据分片信息确定层范围
    start_layer = 0
    end_layer = n_layers - 1

    if shard is not None:
        start_layer = shard.start_layer
        end_layer = shard.end_layer
        print(f"[make_prompt_state] Creating partial cache for layers {start_layer}-{end_layer} (shard: {shard.start_layer}-{shard.end_layer})")
    else:
        print(f"[make_prompt_state] Creating full cache for all {n_layers} layers")

    cache = create_kv_cache(
        batch_size, max_seq_len, n_kv_heads, head_dim, n_layers, dtype, device,
        start_layer=start_layer, end_layer=end_layer
    )
    return ModelState(cache, position=0, shard=shard)
