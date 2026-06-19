#!/usr/bin/env python3
"""
Qwen3-TTS 状态管理
支持 KV 缓存和生成状态管理
"""

import torch
from typing import List, Tuple, Optional
from transformers.cache_utils import DynamicCache


class ModelState:
    """
    Qwen3-TTS 模型状态管理类
    支持 KV 缓存和生成进度跟踪
    """
    
    def __init__(
        self,
        batch_size: int = 1,
        max_seq_len: int = 2048,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        n_layers: int = 28,
        device: torch.device = None,
        shard: Optional["Shard"] = None
    ):
        """
        初始化模型状态
        
        Args:
            batch_size: 批次大小
            max_seq_len: 最大序列长度
            n_kv_heads: KV 头数
            head_dim: 头维度
            n_layers: 层数
            device: 计算设备
            shard: 模型分片信息
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.shard = shard
        self.position = 0
        
        # 创建 DynamicCache
        self.cache = DynamicCache()
        
        # 存储配置信息
        self.config = {
            'batch_size': batch_size,
            'max_seq_len': max_seq_len,
            'n_kv_heads': n_kv_heads,
            'head_dim': head_dim,
            'n_layers': n_layers,
        }
    
    def get_seq_length(self) -> int:
        """获取当前序列长度"""
        return self.cache.get_seq_length() if hasattr(self.cache, 'get_seq_length') else 0
    
    def update_position(self, new_tokens: int):
        """更新生成位置"""
        self.position += new_tokens
    
    def clear(self):
        """清空状态"""
        self.cache = DynamicCache()
        self.position = 0


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
) -> DynamicCache:
    """
    创建 KV 缓存
    
    Args:
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        n_kv_heads: KV 头数
        head_dim: 头维度
        n_layers: 层数
        dtype: 数据类型
        device: 计算设备
        start_layer: 起始层
        end_layer: 结束层
    
    Returns:
        DynamicCache 对象
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if end_layer is None:
        end_layer = n_layers - 1
    
    # 创建空的 DynamicCache
    cache = DynamicCache()
    
    # 存储层范围信息
    cache.start_layer = start_layer
    cache.end_layer = end_layer
    cache.n_layers = n_layers
    
    return cache


def make_prompt_state(
    batch_size: int = 1,
    max_seq_len: int = 2048,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    n_layers: int = 28,
    dtype: torch.dtype = torch.float16,
    device: torch.device = None,
    shard: Optional["Shard"] = None
) -> ModelState:
    """
    创建新的 prompt 状态
    
    Args:
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        n_kv_heads: KV 头数
        head_dim: 头维度
        n_layers: 层数
        dtype: 数据类型
        device: 计算设备
        shard: 模型分片信息
    
    Returns:
        ModelState 对象
    """
    return ModelState(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_layers=n_layers,
        device=device,
        shard=shard
    )
