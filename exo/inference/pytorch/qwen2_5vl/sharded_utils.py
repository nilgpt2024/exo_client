#!/usr/bin/env python3
"""
Qwen2.5-VL 分片权重加载工具 - 支持分片模型加载和权重过滤
基于 exo 框架的分布式推理架构设计
"""

import glob
import json
import logging
import torch
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from concurrent.futures import ThreadPoolExecutor
import safetensors.torch
from safetensors import safe_open
from fnmatch import fnmatch
import re

from exo import DEBUG
from exo.inference.tokenizers import resolve_tokenizer
from exo.inference.shard import Shard
from typing import Optional as TypingOptional

# Fara特定导入
try:
    from transformers import AutoTokenizer, AutoConfig, AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers库未安装，Qwen2.5-VL功能将受限")

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _get_weight_files(model_path: Path) -> List[Path]:
    """获取模型权重文件列表"""
    # 优先查找safetensors格式
    safetensor_files = list(model_path.glob("*.safetensors"))
    if safetensor_files:
        return safetensor_files

    # 回退到pytorch格式
    pytorch_files = list(model_path.glob("*.bin"))
    if pytorch_files:
        return pytorch_files

    # 检查索引文件
    index_files = list(model_path.glob("*.index.json"))
    if index_files:
        return index_files

    return []


def _load_partial_safetensor_weights(file_path: Path, required_keys: List[str], device: Union[str, torch.device] = "cpu", use_bf16: bool = False) -> Dict[str, torch.Tensor]:
    """按需加载safetensors格式权重，只加载指定的权重键"""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("请安装safetensors: pip install safetensors")

    logger.info(f"按需加载safetensors文件: {file_path} (需要 {len(required_keys)} 个权重) 到设备: {device}, BF16: {use_bf16}")

    partial_weights = {}
    device_str = str(device)

    # safe_open的device参数只支持CUDA设备
    if device_str.startswith("cuda"):
        with safe_open(file_path, framework="pt") as f:
            available_keys = set(f.keys())
            keys_to_load = [key for key in required_keys if key in available_keys]

            for key in keys_to_load:
                try:
                    tensor = f.get_tensor(key)
                    if device_str != "cpu":
                        tensor = tensor.to(device)
                    if use_bf16 and tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    partial_weights[key] = tensor
                except Exception as e:
                    logger.error(f"加载权重 {key} 失败: {e}")
    else:
        with safe_open(file_path, framework="pt") as f:
            available_keys = set(f.keys())
            keys_to_load = [key for key in required_keys if key in available_keys]

            for key in keys_to_load:
                try:
                    tensor = f.get_tensor(key)
                    if use_bf16 and tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    partial_weights[key] = tensor
                except Exception as e:
                    logger.error(f"加载权重 {key} 失败: {e}")

    logger.info(f"成功加载 {len(partial_weights)} 个权重到设备: {device}")
    return partial_weights


def _analyze_required_weights(config, shard: Optional[Shard] = None) -> List[str]:
    """分析需要加载的权重键列表"""
    required_patterns = []

    # 获取总层数
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        total_layers = config.text_config.num_hidden_layers
    elif hasattr(config, 'num_hidden_layers'):
        total_layers = config.num_hidden_layers
    else:
        total_layers = getattr(config, 'n_layers', 28)

    # 如果没有分片信息，需要所有层
    if not shard or not shard.model_id:
        logger.info("未指定分片，需要加载所有层")
        return ["*"]  # 通配符，表示需要所有权重

    # 根据分片配置精确过滤
    logger.info(f"分片配置: 层 {shard.start_layer} 到 {shard.end_layer} (总层数: {total_layers})")

    # 基础权重（根据分片位置决定）
    if shard.start_layer == 0:
        # 第一个分片需要嵌入层
        required_patterns.extend([
            "model.embed_tokens",
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens",
            "model.language_model.embed_tokens.weight"
        ])

    if shard.end_layer == total_layers - 1:
        # 最后一个分片需要输出层
        required_patterns.extend([
            "model.norm",
            "model.norm.weight",
            "model.language_model.norm",
            "model.language_model.norm.weight",
            "lm_head",
            "lm_head.weight"
        ])

    # 视觉模型权重（如果首分片包含视觉处理）
    if shard.start_layer == 0:
        required_patterns.extend([
            "model.visual.*",
            "visual.*"
        ])

    # 特定层的权重
    for layer_idx in range(shard.start_layer, shard.end_layer + 1):
        required_patterns.extend([
            f"model.layers.{layer_idx}.*",
            f"model.language_model.layers.{layer_idx}.*",
            f"language_model.layers.{layer_idx}.*"
        ])

    return required_patterns


def load_shard_weights(
    model_path: Union[str, Path],
    shard: Optional[Shard] = None,
    device: Union[str, torch.device] = "cpu",
    use_bf16: bool = False
) -> Dict[str, torch.Tensor]:
    """
    加载分片权重

    Args:
        model_path: 模型路径
        shard: 分片配置
        device: 目标设备
        use_bf16: 是否使用BF16

    Returns:
        权重字典
    """
    model_path = Path(model_path)

    # 加载配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 分析需要的权重
    required_patterns = _analyze_required_weights(config, shard)

    # 如果是通配符，加载所有权重
    if required_patterns == ["*"]:
        logger.info("加载所有权重")
        required_keys = None
    else:
        # 将模式转换为具体的键名（在加载时过滤）
        required_keys = required_patterns

    # 获取权重文件
    weight_files = _get_weight_files(model_path)

    if not weight_files:
        raise ModelNotFoundError(f"在 {model_path} 中未找到权重文件")

    # 加载权重
    all_weights = {}
    for file_path in weight_files:
        if file_path.suffix == ".safetensors":
            if required_keys:
                # 按需加载
                weights = _load_partial_safetensor_weights(file_path, required_keys, device, use_bf16)
            else:
                # 加载所有
                weights = safetensors.torch.load_file(file_path)
                if use_bf16:
                    weights = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in weights.items()}
            all_weights.update(weights)
        elif file_path.suffix == ".bin":
            weights = torch.load(file_path, map_location=device)
            if use_bf16:
                weights = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in weights.items()}
            all_weights.update(weights)

    logger.info(f"成功加载 {len(all_weights)} 个权重参数")
    return all_weights


async def load_shard_weights_async(
    model_path: Union[str, Path],
    shard: Optional[Shard] = None,
    device: Union[str, torch.device] = "cpu",
    use_bf16: bool = False
) -> Dict[str, torch.Tensor]:
    """异步加载分片权重"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor,
            load_shard_weights,
            model_path,
            shard,
            device,
            use_bf16
        )
