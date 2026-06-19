#!/usr/bin/env python3
"""
Qwen3VL 分片权重加载工具 - 支持分片模型加载和权重过滤
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
from typing import Optional as TypingOptional  # 避免与transformers的Optional冲突

# Qwen3VL特定导入
try:
    from transformers import AutoTokenizer, AutoConfig, AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers库未安装，Qwen3VL功能将受限")

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    
    # 添加调试信息
    print(f"[DEBUG] _get_classes:原始model_type={config.get('model_type', 'not found')}")
    print(f"[DEBUG] _get_classes:处理后model_type={model_type}")
    print(f"[DEBUG] _get_classes:配置中的关键字段={list(config.keys())}")

    try:
        if model_type == "qwen3":
            from .models.qwen3 import Model, ModelArgs
            return Model, ModelArgs
        elif model_type == "qwen3_vl":
            from .models.qwen3_vl import Model, ModelArgs
            return Model, ModelArgs
        elif model_type == "llama":
            from .models.llama import Model, ModelArgs
            return Model, ModelArgs
        else:
            # 可以扩展支持其他模型类型
            raise ValueError(f"Model type {model_type} not supported in PyTorch engine.")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)


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
    """按需加载safetensors格式权重，只加载指定的权重键 - 支持BF16和直接设备加载"""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("请安装safetensors: pip install safetensors")
    
    logger.info(f"按需加载safetensors文件: {file_path} (需要 {len(required_keys)} 个权重) 到设备: {device}, BF16: {use_bf16}")
    
    partial_weights = {}
    
    # 将device转换为字符串以便比较
    device_str = str(device)
    
    # safe_open的device参数只支持CUDA设备，不支持CPU
    if device_str.startswith("cuda"):
        # 对于CUDA设备，先加载到CPU，然后移动到GPU
        with safe_open(file_path, framework="pt") as f:
            available_keys = set(f.keys())
            keys_to_load = [key for key in required_keys if key in available_keys]
            
            for key in keys_to_load:
                try:
                    tensor = f.get_tensor(key)
                    # 移动到目标设备并应用BF16转换
                    if device_str != "cpu":
                        tensor = tensor.to(device)
                    if use_bf16 and tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    partial_weights[key] = tensor
                except Exception as e:
                    logger.error(f"加载权重 {key} 失败: {e}")
    else:
        # 对于CPU设备，不使用device参数
        with safe_open(file_path, framework="pt") as f:
            available_keys = set(f.keys())
            keys_to_load = [key for key in required_keys if key in available_keys]
            
            for key in keys_to_load:
                try:
                    tensor = f.get_tensor(key)
                    # 应用BF16转换（如果启用）
                    if use_bf16 and tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    partial_weights[key] = tensor
                except Exception as e:
                    logger.error(f"加载权重 {key} 失败: {e}")
    
    logger.info(f"成功加载 {len(partial_weights)} 个权重到设备: {device}")
    return partial_weights


def _analyze_required_weights(config, shard: Optional[Shard] = None) -> List[str]:
    """分析需要加载的权重键列表 - 修复版支持Qwen3VL结构"""
    required_patterns = []
    
    # 获取总层数
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        total_layers = config.text_config.num_hidden_layers
    elif hasattr(config, 'num_hidden_layers'):
        total_layers = config.num_hidden_layers
    else:
        total_layers = getattr(config, 'n_layers', 32)
    
    # 如果没有分片信息，需要所有层
    if not shard or not shard.model_id:
        logger.info("未指定分片，需要加载所有层")
        return ["*"]  # 通配符，表示需要所有权重
    
    # 根据分片配置精确过滤
    logger.info(f"分片配置: 层 {shard.start_layer} 到 {shard.end_layer} (总层数: {total_layers})")
    
    # 基础权重（根据分片位置决定）- 支持多种格式
    if shard.start_layer == 0:
        # 第一个分片需要嵌入层
        required_patterns.extend([
            # 标准格式
            "model.embed_tokens",
            "model.embed_tokens.weight",
            # Qwen3VL格式
            "model.language_model.embed_tokens",
            "model.language_model.embed_tokens.weight"
        ])
    
    if shard.end_layer == total_layers - 1:
        # 最后一个分片需要输出层
        required_patterns.extend([
            # 标准格式
            "model.norm",
            "model.norm.weight",
            "model.norm.bias",
            "lm_head",
            "lm_head.weight",
            "lm_head.bias",
            # Qwen3VL格式
            "model.language_model.norm",
            "model.language_model.norm.weight",
            "model.language_model.norm.bias",
            "model.language_model.lm_head",
            "model.language_model.lm_head.weight",
            "model.language_model.lm_head.bias"
        ])
    
    # 视觉相关权重（多模态模型需要）- 只在第一层分片时加载
    visual_patterns = []
    if shard.start_layer == 0:
        # 只有第一个分片需要视觉权重
        visual_patterns = [
            # Qwen3VL实际格式 - 使用精确前缀匹配
            "model.visual.",
            "model.vision_tower.",
        ]
        required_patterns.extend(visual_patterns)
        logger.info(f"第一层分片，添加视觉权重模式: {len(visual_patterns)} 个")
    else:
        logger.info(f"非第一层分片（{shard.start_layer}），跳过视觉权重加载")
    
    # 分片层权重（支持多种格式）
    layer_patterns = [
        "model.layers.",           # 标准格式
        "transformer.h.",          # 替代格式
        "model.transformer.h.",    # 替代格式
        "model.language_model.layers.",  # Qwen3VL格式
    ]
    
    for layer_pattern in layer_patterns:
        for layer_idx in range(shard.start_layer, min(shard.end_layer + 1, total_layers)):
            required_patterns.append(f"{layer_pattern}{layer_idx}.")
    
    logger.info(f"分析完成: 需要 {len(required_patterns)} 个权重模式")
    if DEBUG >= 2:
        logger.info(f"权重模式: {required_patterns}")
    
    return required_patterns


def _load_weights_on_demand(model_path: Path, required_patterns: List[str], device: Union[str, torch.device], use_bf16: bool = False) -> Dict[str, torch.Tensor]:
    """按需加载权重 - 只加载需要的权重，支持BF16和设备优化"""
    logger.info(f"开始按需加载权重到设备: {device}, BF16: {use_bf16}")
    
    # 在权重加载前进行内存清理，为大规模权重加载预留空间
    if str(device).startswith("cuda") and torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("✓ 权重加载前GPU内存清理完成")
    
    # 获取权重文件
    weight_files = _get_weight_files(model_path)
    if not weight_files:
        raise RuntimeError("未找到权重文件")
    
    loaded_weights = {}
    
    # 处理每个权重文件
    for file_path in weight_files:
        if file_path.suffix == ".safetensors":
            # safetensors格式支持按需加载
            file_weights = _load_matching_safetensor_weights(file_path, required_patterns, device, use_bf16)
            loaded_weights.update(file_weights)
            
        elif file_path.suffix == ".bin":
            # PyTorch格式需要加载整个文件再过滤
            file_weights = _load_matching_pytorch_weights(file_path, required_patterns, device, use_bf16)
            loaded_weights.update(file_weights)
            
        elif file_path.suffix == ".json" and "index" in file_path.name:
            # 处理索引文件
            index_weights = _load_matching_index_weights(file_path, required_patterns, device, use_bf16)
            loaded_weights.update(index_weights)
    
    logger.info(f"按需加载完成: 成功加载 {len(loaded_weights)} 个权重")
    
    # 在权重加载完成后进行内存清理，释放临时内存
    if str(device).startswith("cuda") and torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("✓ 权重加载后GPU内存清理完成")
    
    return loaded_weights


def _load_matching_safetensor_weights(file_path: Path, required_patterns: List[str], device: Union[str, torch.device], use_bf16: bool = False) -> Dict[str, torch.Tensor]:
    """按需加载匹配的safetensors权重 - 精确匹配，支持BF16转换"""
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors未安装，跳过文件")
        return {}
    
    matching_weights = {}
    
    # 将device转换为字符串以便比较
    device_str = str(device)
    
    # 特殊处理通配符
    if "*" in required_patterns:
        logger.info("检测到通配符模式，加载所有权重")
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                try:
                    tensor = f.get_tensor(key)
                    if device_str != "cpu" and not device_str.startswith("cuda"):
                        tensor = tensor.to(device)
                    # 应用BF16转换（如果启用且是float32）
                    if use_bf16 and tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    matching_weights[key] = tensor
                except Exception as e:
                    logger.error(f"加载权重 {key} 失败: {e}")
        return matching_weights
    
    with safe_open(file_path, framework="pt") as f:
        available_keys = list(f.keys())
        
        # 精确匹配逻辑
        matching_keys = []
        for key in available_keys:
            key_matched = False
            
            # 检查每个模式
            for pattern in required_patterns:
                if pattern.endswith("."):
                    # 层模式匹配：检查键是否以该模式开头
                    if key.startswith(pattern):
                        key_matched = True
                        break
                else:
                    # 精确模式匹配：检查键是否包含该模式
                    if pattern in key:
                        key_matched = True
                        break
            
            if key_matched:
                matching_keys.append(key)
        
        logger.info(f"文件 {file_path.name}: 匹配到 {len(matching_keys)} 个权重")
        
        # 加载匹配的权重
        for key in matching_keys:
            try:
                tensor = f.get_tensor(key)
                if device_str != "cpu" and not device_str.startswith("cuda"):
                    tensor = tensor.to(device)
                # 应用BF16转换（如果启用且是float32）
                if use_bf16 and tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.bfloat16)
                matching_weights[key] = tensor
            except Exception as e:
                logger.error(f"加载权重 {key} 失败: {e}")
    
    return matching_weights


def _load_matching_pytorch_weights(file_path: Path, required_patterns: List[str], device: Union[str, torch.device], use_bf16: bool = False) -> Dict[str, torch.Tensor]:
    """加载匹配的PyTorch权重 - 支持直接设备加载和BF16转换"""
    try:
        # 将device转换为字符串以便比较
        device_str = str(device)
        
        # 直接加载到目标设备（如果支持）
        if device_str != "cpu" and torch.cuda.is_available():
            all_weights = torch.load(file_path, map_location=device, weights_only=True)
        else:
            all_weights = torch.load(file_path, map_location="cpu", weights_only=True)
        
        # 过滤需要的权重
        matching_weights = {}
        for key, tensor in all_weights.items():
            for pattern in required_patterns:
                if pattern in key or key.startswith(pattern.replace(".", "")):
                    # 应用BF16转换（如果启用且是float32）
                    if use_bf16 and tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    # 移动到目标设备（如果还没在目标设备上）
                    if device_str != "cpu" and tensor.device != torch.device(device):
                        tensor = tensor.to(device)
                    matching_weights[key] = tensor
                    break
        
        logger.info(f"文件 {file_path.name}: 匹配到 {len(matching_weights)} 个权重, BF16: {use_bf16}")
        return matching_weights
        
    except Exception as e:
        logger.error(f"加载PyTorch权重文件失败: {e}")
        return {}


def _load_matching_index_weights(index_file: Path, required_patterns: List[str], device: Union[str, torch.device] = "cpu", use_bf16: bool = False) -> Dict[str, torch.Tensor]:
    """处理索引文件，按需加载权重"""
    try:
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get('weight_map', {})
        base_path = index_file.parent
        
        # 找到需要的文件
        required_files = set()
        for weight_key, filename in weight_map.items():
            for pattern in required_patterns:
                if pattern in weight_key or weight_key.startswith(pattern.replace(".", "")):
                    required_files.add(filename)
                    break
        
        logger.info(f"索引文件 {index_file.name}: 需要 {len(required_files)} 个文件")
        
        # 按需加载这些文件
        loaded_weights = {}
        for filename in required_files:
            file_path = base_path / filename
            if not file_path.exists():
                logger.warning(f"权重文件不存在: {file_path}")
                continue
            
            if filename.endswith(".safetensors"):
                file_weights = _load_matching_safetensor_weights(file_path, required_patterns, device, use_bf16)
            else:
                file_weights = _load_matching_pytorch_weights(file_path, required_patterns, device, use_bf16)
            
            loaded_weights.update(file_weights)
        
        return loaded_weights
        
    except Exception as e:
        logger.error(f"处理索引文件失败: {e}")
        return {}


def _load_partial_pytorch_weights(file_path: Path, required_keys: List[str], device: Union[str, torch.device] = "cpu") -> Dict[str, torch.Tensor]:
    """按需加载PyTorch格式权重，只加载指定的权重键"""
    try:
        # 将device转换为字符串以便比较
        device_str = str(device)
        
        # 加载整个文件
        all_weights = torch.load(file_path, map_location="cpu", weights_only=True)
        
        # 过滤需要的权重
        filtered_weights = {}
        for key in required_keys:
            if key in all_weights:
                filtered_weights[key] = all_weights[key].to(device) if device_str != "cpu" else all_weights[key]
        
        logger.info(f"从 {file_path} 加载了 {len(filtered_weights)} 个权重")
        return filtered_weights
        
    except Exception as e:
        logger.error(f"加载PyTorch权重文件失败: {e}")
        return {}


def load_qwen3vl_model_shard(
    model_path: Path,
    shard: Shard,
    device: Optional[Union[str, torch.device]] = None,
    use_bf16: bool = False
) -> tuple:
    """
    加载Qwen3VL模型分片 - 支持分片和权重过滤
    
    Args:
        model_path: 模型路径
        shard: 分片信息
        device: 目标设备
        use_bf16: 是否使用BF16精度
        
    Returns:
        (model, tokenizer, processor) 三元组
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers库未安装，无法加载Qwen3VL模型")
    
    logger.info("开始Qwen3VL模型分片加载流程...")
    
    # 步骤1: 加载配置
    logger.info("步骤1: 加载模型配置...")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    logger.info(f"✓ 配置加载完成: {config.model_type}")
    
    # 步骤2: 加载分词器和处理器
    logger.info("步骤2: 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 只在需要时加载处理器（避免torchvision依赖）
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        logger.info("✓ 处理器加载完成")
    except ImportError as e:
        logger.warning(f"处理器加载失败（可能缺少torchvision），跳过: {e}")
    logger.info("✓ 分词器加载完成")
    
    # 步骤3: 分析需要加载的权重
    logger.info("步骤3: 分析需要加载的权重...")
    required_weights = _analyze_required_weights(config, shard)
    logger.info(f"✓ 分析完成，需要加载 {len(required_weights)} 个权重参数")
    
    # 步骤4: 按需加载权重（而不是加载完整权重）
    logger.info("步骤4: 按需加载权重...")
    filtered_weights = _load_weights_on_demand(model_path, required_weights, device, use_bf16)
    logger.info(f"✓ 成功加载 {len(filtered_weights)} 个权重参数")
    
    # 步骤5: 创建模型 - 分片优化版本
    logger.info("步骤5: 创建Qwen3VL模型分片...")
    try:
        # 获取模型类
        Model, ModelArgs = _get_classes(config.to_dict())
        
        # 获取 tokenizer 的 vocab_size
        tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', None)
        logger.info(f"Tokenizer vocab_size: {tokenizer_vocab_size}")
        
        # 创建分片模型（只包含必要的层）
        try:
            model = Model(config, shard=shard, tokenizer_vocab_size=tokenizer_vocab_size)
            logger.info(f"✓ 分片模型创建完成: {type(model)}")
        except Exception as e:
            # 如果Model类不支持tokenizer_vocab_size参数，尝试标准方式
            logger.warning(f"Model类不支持tokenizer_vocab_size参数，尝试标准实例化: {type(e).__name__}: {e}")
            model = Model(config, shard=shard)
            logger.info(f"✓ 模型创建完成（标准方式）: {type(model)}")
        
        # 内存清理
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        raise
    
    # 步骤7: 加载过滤后的权重
    logger.info("步骤7: 加载过滤后的权重...")
    try:
        # 在权重加载前进行内存清理，减少峰值内存使用
        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("✓ 权重加载前GPU内存清理完成")
        
        # 清理权重（如果模型有sanitize方法）- 关键步骤！
        if hasattr(model, "sanitize"):
            logger.warning(f"sanitize前权重数量: {len(filtered_weights)}")
            filtered_weights = model.sanitize(filtered_weights)
            logger.warning(f"sanitize后权重数量: {len(filtered_weights)}")
        else:
            logger.warning(f"模型没有sanitize方法，跳过权重适配")
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_weights, strict=False)
        
        if missing_keys:
            logger.warning(f"缺失的权重键: {len(missing_keys)} 个")
            for key in missing_keys[:5]:
                logger.warning(f"  - {key}")
            if len(missing_keys) > 5:
                logger.warning(f"  ... 还有 {len(missing_keys) - 5} 个")
        
        if unexpected_keys:
            logger.warning(f"意外的权重键: {len(unexpected_keys)} 个")
            for key in unexpected_keys[:5]:
                logger.warning(f"  - {key}")
            if len(unexpected_keys) > 5:
                logger.warning(f"  ... 还有 {len(unexpected_keys) - 5} 个")
        
        logger.info("✓ 权重加载完成")
    except Exception as e:
        logger.error(f"权重加载失败: {e}")
        raise
    
    # 步骤8: 移动到目标设备 - 分片优化
    if device and str(device) != "cpu":
        logger.info(f"步骤8: 移动到设备 {device} - 分片优化...")
        try:
            # 在移动模型前进行内存清理，为大规模模型移动预留空间
            if str(device).startswith("cuda") and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info("✓ 模型移动前GPU内存清理完成")
            
            # 只移动必要的层到GPU，减少内存使用
            if shard and shard.start_layer is not None and shard.end_layer is not None:
                logger.info(f"只移动分片层 {shard.start_layer}-{shard.end_layer} 到 {device}")
                
                # 获取实际的模型对象（处理嵌套结构）
                actual_model = model
                if hasattr(model, 'model'):
                    actual_model = model.model
                    if hasattr(actual_model, 'model'):
                        actual_model = actual_model.model
                
                # 移动嵌入层（如果需要）
                if shard.start_layer == 0:
                    if hasattr(actual_model, 'embed_tokens'):
                        actual_model.embed_tokens = actual_model.embed_tokens.to(device)
                        logger.info(f"✓ 嵌入层已移动到 {device}")
                
                # 移动分片的transformer层
                if hasattr(actual_model, 'layers'):
                    for layer_idx in range(shard.start_layer, min(shard.end_layer + 1, len(actual_model.layers))):
                        if layer_idx < len(actual_model.layers):
                            actual_model.layers[layer_idx] = actual_model.layers[layer_idx].to(device)
                    logger.info(f"✓ 分片层已移动到 {device}")
                
                # 移动输出层（如果需要）
                if shard.end_layer == (getattr(config, 'num_hidden_layers', 36) - 1):
                    if hasattr(actual_model, 'norm'):
                        actual_model.norm = actual_model.norm.to(device)
                        logger.info(f"✓ 归一化层已移动到 {device}")
                    if hasattr(model, 'lm_head'):
                        model.lm_head = model.lm_head.to(device)
                        logger.info(f"✓ LM Head已移动到 {device}")
                
                # 移动视觉组件
                if hasattr(actual_model, 'visual'):
                    actual_model.visual = actual_model.visual.to(device)
                    logger.info(f"✓ 视觉组件已移动到 {device}")
                    
            else:
                # 没有分片信息，移动整个模型
                logger.info(f"移动整个模型到 {device}")
                model = model.to(device)
                
            logger.info("✓ 模型移动完成")
            
            # 在移动完成后进行内存清理，释放临时内存
            if str(device).startswith("cuda") and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info("✓ 模型移动后GPU内存清理完成")
            
        except Exception as e:
            logger.error(f"模型移动失败: {e}")
            logger.warning("继续使用CPU设备")
            # 内存清理
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 设置模型为评估模式
    model.eval()
    
    # 显示内存使用情况
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"✓ Qwen3VL模型分片加载成功，设备: {device}")
        logger.info(f"  GPU内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
    else:
        logger.info(f"✓ Qwen3VL模型分片加载成功，设备: {device}")
    
    return model, tokenizer, processor







async def load_shard(
        model_path: str,
        shard: Shard,
        tokenizer_config={},
        model_config={},
        adapter_path: Optional[str] = None,
        lazy: bool = False,
        executor: Optional[ThreadPoolExecutor] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_bf16: bool = False,
):
    """
    加载模型分片和分词器异步 - 支持Qwen3VL和标准模型
    
    Args:
        model_path: 模型路径
        shard: 分片配置
        tokenizer_config: 分词器配置
        model_config: 模型配置
        adapter_path: 适配器路径（未使用）
        lazy: 是否延迟加载权重
        executor: 线程池执行器
        device: 目标设备
        use_bf16: 是否使用BF16精度
        
    Returns:
        (model, tokenizer, processor) - 对于Qwen3VL返回三元组，对于标准模型返回(model, tokenizer, None)
    """
    # 运行模型加载在线程池中使其完全异步
    loop = asyncio.get_running_loop()
    
    def load_and_move_model():
        # 在模型加载前彻底清理GPU内存
        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 对于Qwen3VL模型，直接使用专用加载函数，不回退
        model, tokenizer, processor = load_qwen3vl_model_shard(
            Path(model_path),
            shard,
            device=device,
            use_bf16=use_bf16
        )
        return model, tokenizer, processor
    
    if executor is None:
        # 使用临时executor，确保正确清理资源
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pytorch_load") as temp_executor:
            result = await loop.run_in_executor(temp_executor, load_and_move_model)
    else:
        # 使用调用者提供的executor
        result = await loop.run_in_executor(executor, load_and_move_model)
    
    return result