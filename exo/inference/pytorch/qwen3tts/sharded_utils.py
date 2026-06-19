#!/usr/bin/env python3
"""
Qwen3-TTS 分片工具
支持模型分片加载和权重管理
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

from exo import DEBUG
from exo.inference.tokenizers import resolve_tokenizer
from exo.inference.shard import Shard

logger = logging.getLogger(__name__)


def _get_classes(config: dict):
    """
    获取模型类
    
    Args:
        config: 模型配置
    
    Returns:
        (Model类, ModelArgs类)
    """
    model_type = config.get("model_type", "")
    
    if model_type == "qwen3_tts":
        from .qwen3tts_model import Qwen3TTSForConditionalGeneration, Qwen3TTSArgs
        return Qwen3TTSForConditionalGeneration, Qwen3TTSArgs
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def load_config(model_path: Path) -> dict:
    """加载模型配置"""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"加载配置: {config_path}")
    else:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    # 加载 generation_config.json
    gen_config_path = model_path / "generation_config.json"
    if gen_config_path.exists():
        try:
            with open(gen_config_path, "r", encoding='utf-8') as f:
                gen_config = json.load(f)
                config['generation_config'] = gen_config
                logger.info(f"加载生成配置: temperature={gen_config.get('temperature')}")
        except Exception as e:
            logger.warning(f"加载 generation_config.json 失败: {e}")
    
    return config


async def load_shard(
    model_path: str,
    shard: Shard,
    model_config: dict = None,
    lazy: bool = False,
    executor: ThreadPoolExecutor = None,
    device: Optional[Union[str, torch.device]] = None,
    use_bf16: bool = False
) -> tuple:
    """
    异步加载模型分片
    
    Args:
        model_path: 模型路径
        shard: 模型分片
        model_config: 模型配置
        lazy: 是否延迟加载
        executor: 线程池执行器
        device: 计算设备
        use_bf16: 是否使用 BF16
    
    Returns:
        (model, tokenizer)
    """
    model_path = Path(model_path)
    
    # 加载配置
    if model_config is None or len(model_config) == 0:
        config_dict = load_config(model_path)
    else:
        config_dict = model_config
    
    # 获取模型类
    ModelClass, ArgsClass = _get_classes(config_dict)
    
    # 创建模型参数
    args = ArgsClass.from_dict(config_dict)
    args.shard = shard
    
    # 创建模型
    model = ModelClass(args)
    
    # 加载权重
    if not lazy:
        await _load_weights_async(model, model_path, shard, executor, device)
    
    # 移动模型到设备
    if device is not None:
        device = torch.device(device) if isinstance(device, str) else device
        model = model.to(device)
    
    # 启用 BF16
    if use_bf16:
        model = model.to(torch.bfloat16)
    
    # 设置为评估模式
    model.eval()
    
    # 加载 tokenizer
    from .tts_tokenizer import Qwen3TTSTokenizer
    tokenizer = Qwen3TTSTokenizer(str(model_path))
    
    logger.info(f"✅ 分片加载完成: layers {shard.start_layer}-{shard.end_layer}")
    
    return model, tokenizer


async def _load_weights_async(
    model: torch.nn.Module,
    model_path: Path,
    shard: Shard,
    executor: ThreadPoolExecutor = None,
    device: Optional[torch.device] = None
):
    """异步加载权重"""
    loop = asyncio.get_event_loop()
    
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1)
    
    await loop.run_in_executor(
        executor,
        _load_weights_sync,
        model,
        model_path,
        shard,
        device
    )


def _load_weights_sync(
    model: torch.nn.Module,
    model_path: Path,
    shard: Shard,
    device: Optional[torch.device] = None
):
    """同步加载权重"""
    # 查找权重文件
    weight_files = list(model_path.glob("*.safetensors"))
    weight_files = [f for f in weight_files if 'speech_tokenizer' not in str(f)]
    
    if not weight_files:
        raise FileNotFoundError(f"未找到权重文件: {model_path}")
    
    logger.info(f"找到 {len(weight_files)} 个权重文件")
    
    # 加载所有权重
    state_dict = {}
    for weight_file in weight_files:
        logger.info(f"加载: {weight_file.name}")
        file_state_dict = safetensors.torch.load_file(weight_file)
        state_dict.update(file_state_dict)
    
    # 过滤权重
    filtered_state_dict = _filter_weights_for_shard(state_dict, shard)
    
    # 加载到模型
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"缺失的键: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"意外的键: {unexpected_keys[:5]}...")
    
    logger.info(f"加载了 {len(filtered_state_dict)} 个权重")


def _filter_weights_for_shard(state_dict: Dict[str, torch.Tensor], shard: Shard) -> Dict[str, torch.Tensor]:
    """根据分片过滤权重"""
    filtered = {}
    
    for key, value in state_dict.items():
        # 检查是否在分片范围内
        if 'layers.' in key:
            # 提取层索引
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if shard.start_layer <= layer_idx <= shard.end_layer:
                            filtered[key] = value
                        break
                    except ValueError:
                        continue
        else:
            # 非层权重
            if shard.is_first_layer() and 'embed_tokens' in key:
                filtered[key] = value
            elif shard.is_last_layer() and ('norm' in key or 'codec_head' in key or 'lm_head' in key):
                filtered[key] = value
            elif 'embed_tokens' not in key and 'norm' not in key and 'codec_head' not in key and 'lm_head' not in key:
                filtered[key] = value
    
    return filtered


def sanitize(weights: Dict[str, torch.Tensor], shard: Shard) -> Dict[str, torch.Tensor]:
    """清理和过滤权重"""
    return _filter_weights_for_shard(weights, shard)
