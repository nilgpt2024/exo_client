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
        if model_type == "llama":
            from .llama3 import Model, ModelArgs
            return Model, ModelArgs
        else:
            # 可以扩展支持其他模型类型
            raise ValueError(f"Model type {model_type} not supported in PyTorch engine.")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)


# 只使用本地配置加载，移除 transformers 依赖
def load_config(model_path: Path) -> dict:
    """加载模型配置文件"""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"[DEBUG] load_config: 从{config_path}加载配置")
        print(f"[DEBUG] load_config: 原始model_type={config.get('model_type', 'not found')}")
    else:
        raise FileNotFoundError(f"Config file not found in {model_path}")

    # 加载generation_config.json（如果存在）
    gen_config_path = model_path / "generation_config.json"
    if gen_config_path.exists():
        try:
            with open(gen_config_path, "r", encoding="utf-8") as f:
                gen_config_data = json.load(f)
                config['generation_config'] = gen_config_data
                print(f"已加载generation配置: temperature={gen_config_data.get('temperature')}, top_p={gen_config_data.get('top_p')}")
        except Exception as e:
            print(f"加载generation_config.json失败: {e}")

    return config


#
def load_model_shard(
        model_path: Path,
        shard: Shard,
        lazy: bool = False,
        model_config: dict = {},
        device: Optional[Union[str, torch.device]] = None,
        use_bf16: bool = False
) -> torch.nn.Module:
    """
    Load and initialize the model shard from a given path.

    Args:
        model_path (Path): The path to load the model from.
        shard (Shard): The shard configuration.
        lazy (bool): If False, load weights immediately. Default: False
        model_config (dict, optional): Configuration parameters for the model.
        device (Optional[Union[str, torch.device]]): Target device for the model.
        use_bf16 (bool): Whether to use BF16 precision.

    Returns:
        torch.nn.Module: The loaded and initialized model shard.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found.
    """
    config = load_config(model_path)
    config.update(model_config)

    # 添加调试输出验证配置
    if DEBUG >= 2:
        logging.info(f"Model config: hidden_size={config['hidden_size']}, "
                    f"num_attention_heads={config['num_attention_heads']}, "
                    f"num_key_value_heads={config.get('num_key_value_heads', 'not set')}")

    # 添加分片配置
    config["shard"] = {
        "model_id": shard.model_id,
        "start_layer": shard.start_layer,
        "end_layer": shard.end_layer,
        "n_layers": shard.n_layers,
    }

    # 获取模型类
    print(f"[DEBUG] load_model_shard: 即将调用_get_classes，config.model_type={config.get('model_type')}")
    model_class, model_args_class = _get_classes(config=config)

    # 创建模型参数
    model_args = model_args_class.from_dict(config)

    # 确定目标设备和精度
    target_device = device if device else "cpu"
    target_dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    # 使用目标 dtype 创建模型，先在 CPU 上创建和加载权重，避免显存占用翻倍
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        # 创建模型实例（先在 CPU 上）
        model = model_class(model_args)
        # 注意：先不在此移动到 GPU，而是在加载权重后再移动
        # 这样可以避免随机初始化的模型和预训练权重同时存在于显存中
    finally:
        torch.set_default_dtype(original_dtype)

    if not lazy:
        # 加载权重到 CPU，然后再移动到目标设备
        load_model_weights(model, model_path, shard, device="cpu")
        
        # 将加载好权重的模型移动到目标设备
        if target_device != "cpu":
            model = model.to(device=target_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logging.info(f"模型已成功加载到设备: {target_device}, 精度: {target_dtype}")

    # 设置模型为评估模式
    model.eval()
    return model


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
    Load model shard and tokenizer asynchronously.

    Args:
        model_path (str): Path to the model.
        shard (Shard): Shard configuration.
        tokenizer_config (dict): Tokenizer configuration.
        model_config (dict): Model configuration.
        adapter_path (Optional[str]): Path to adapter (not used in PyTorch engine).
        lazy (bool): Whether to load weights lazily.
        executor (Optional[ThreadPoolExecutor]): Thread pool executor for async operations.
        device (Optional[Union[str, torch.device]]): Target device for the model.
        use_bf16 (bool): Whether to use BF16 precision.

    Returns:
        Tuple of (model, tokenizer).
    """
    # Run model loading in thread pool to make it fully async
    loop = asyncio.get_running_loop()
    
    # 定义一个包装函数，包含模型加载和设备移动
    def load_and_move_model():
        # 在模型加载前彻底清理GPU内存
        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 确定目标设备用于权重加载
        target_device = device if device else "cpu"
        
        model = load_model_shard(
            Path(model_path),
            shard,
            lazy,
            model_config,
            device=device,
            use_bf16=use_bf16
        )
        
        # 在线程池中执行设备移动和精度转换，避免阻塞主线程
        if device is not None:
            try:
                # BF16优化：如果支持BF16，将模型转换为BF16精度
                if use_bf16:
                    logging.info("启用BF16优化，转换模型精度...")
                    model = model.to(torch.bfloat16)
                    logging.info("模型已转换为BF16精度")
                
                # 移动模型到指定设备
                model = model.to(device)
                logging.info(f"模型已成功移动到设备: {device}")
                
                # 模型加载后彻底清理GPU内存
                if torch.cuda.is_available():
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    
                    # 同步所有CUDA操作
                    torch.cuda.synchronize()
                    
                    # 重置CUDA内存分配器
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # 获取当前GPU内存使用情况
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    
                    # 尝试释放保留内存（可选的激进清理）
                    try:
                        # 释放所有保留的缓存内存
                        if hasattr(torch.cuda, 'release_memory'):
                            torch.cuda.release_memory(torch.cuda.memory_reserved())
                        # 再次清理缓存
                        torch.cuda.empty_cache()
                        
                        # 重新获取内存使用情况
                        memory_allocated_after = torch.cuda.memory_allocated() / 1024**3  # GB
                        memory_reserved_after = torch.cuda.memory_reserved() / 1024**3   # GB
                        
                        logging.info(f"模型加载后已彻底清理GPU内存 - 清理前: 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")
                        logging.info(f"模型加载后已彻底清理GPU内存 - 清理后: 已分配: {memory_allocated_after:.2f}GB, 已保留: {memory_reserved_after:.2f}GB")
                    except Exception as e:
                        logging.warning(f"尝试释放保留内存失败: {e}")
                        logging.info(f"模型加载后已彻底清理GPU内存 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")
            except Exception as e:
                logging.warning(f"模型设备移动失败: {e}，使用默认设备")
        
        return model
    
    if executor is None:
        # 使用临时executor，确保正确清理资源
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pytorch_load") as temp_executor:
            model = await loop.run_in_executor(
                temp_executor,
                load_and_move_model
            )
    else:
        # 使用调用者提供的executor
        model = await loop.run_in_executor(
            executor,
            load_and_move_model
        )

    # Load tokenizer asynchronously
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        tokenizer = await resolve_tokenizer(model_path)
    
    return model, tokenizer

def load_model_weights(model: torch.nn.Module, model_path: Path, shard: Optional[Shard] = None, device: str = "cpu"):
    """加载模型权重 - 支持.index.json分片加载"""
    logger.info(f"开始加载模型权重到设备: {device}...")

    # 首先检查是否存在.index.json文件（分片加载）
    index_files = list(model_path.glob("*.index.json"))
    if index_files:
        logger.info(f"检测到.index.json文件，使用分片加载: {index_files[0]}")
        _load_indexed_weights(model, index_files[0], shard, device)
        return

    # 回退到传统的文件扫描方式
    safetensor_files = list(model_path.glob("*.safetensors"))
    pytorch_files = list(model_path.glob("*.bin"))

    if safetensor_files:
        _load_safetensor_weights(model, safetensor_files, shard, device)
    elif pytorch_files:
        _load_pytorch_weights(model, pytorch_files, shard, device)
    else:
        raise FileNotFoundError("未找到权重文件")

def _load_indexed_weights(model: torch.nn.Module, index_file: Path, shard: Optional[Shard] = None, device: str = "cpu"):
    """通过.index.json文件加载分片权重 - 类似Tinygrad的实现，支持按需加载"""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("请安装safetensors: pip install safetensors")

    logger.info(f"使用索引文件加载分片权重到设备: {device}")
    
    # 读取索引文件
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data.get('weight_map', {})
    logger.info(f"索引文件中总权重数量: {len(weight_map)}")
    
    # 过滤权重映射 - 根据分片信息只保留需要的权重
    filtered_weight_map = {}
    excluded_keys = []

    for key, filename in weight_map.items():
        # 如果提供了分片信息，进行层级别过滤
        if shard and key.startswith("model.layers."):
            try:
                # 提取层号: model.layers.X....
                layer_num = int(key.split('.')[2])
                if layer_num < shard.start_layer or layer_num > shard.end_layer:
                    excluded_keys.append(key)
                    continue
            except (ValueError, IndexError):
                # 如果解析失败，包含该权重（保守策略）
                pass
        # 处理非层权重（embed_tokens, norm, lm_head等）
        elif key.startswith("model.embed_tokens"):
            # 第一层需要embed_tokens进行嵌入
            # 最后一层在tie_word_embeddings=True时也需要embed_tokens的权重用于生成logits
            if shard and not (shard.is_first_layer() or shard.is_last_layer()):
                excluded_keys.append(key)
                continue
        elif key.startswith("model.norm"):
            # 只在最后层需要norm
            if shard and not shard.is_last_layer():
                excluded_keys.append(key)
                continue
        elif key.startswith("lm_head"):
            # 只在最后一层且不使用权重共享时需要lm_head
            # 注意：这里无法确定tie_word_embeddings，所以保守地保留
            # 让模型的sanitize方法来最终决定是否使用
            if shard and not shard.is_last_layer():
                excluded_keys.append(key)
                continue
        # 确保视觉相关的权重总是被包含 - 修复视觉权重加载问题
        elif (key.startswith("model.visual") or
              key.startswith("visual") or
              "visual.merger" in key or
              "vision" in key.lower() or
              "patch_embedding" in key or
              "position_embedding" in key):
            # 总是包含视觉相关的权重，不论分片设置
            # Qwen3VL模型使用merger而不是linear projection来实现视觉投影
            # 扩展检测模式以包含更多视觉相关权重
            pass

        filtered_weight_map[key] = filename
    
    if DEBUG >= 2:
        logger.info(f"分片 {shard} 排除的权重键数量: {len(excluded_keys)}")
        logger.info(f"分片 {shard} 保留的权重键数量: {len(filtered_weight_map)}")
    
    # 按文件分组权重
    file_to_keys = {}
    for key, filename in filtered_weight_map.items():
        if filename not in file_to_keys:
            file_to_keys[filename] = []
        file_to_keys[filename].append(key)
    
    # 按需加载需要的文件
    state_dict = {}
    base_path = index_file.parent
    
    for filename, keys in file_to_keys.items():
        file_path = base_path / filename
        if not file_path.exists():
            logger.warning(f"权重文件不存在: {file_path}")
            continue
            
        logger.info(f"按需加载权重文件: {filename} (需要 {len(keys)} 个权重) 到设备: {device}")
        
        try:
            # 根据文件扩展名选择按需加载方式
            if filename.endswith(".safetensors"):
                file_weights = _load_partial_safetensor_weights(file_path, keys, device=device)
            else:
                file_weights = _load_partial_pytorch_weights(file_path, keys, device=device)
                    
            state_dict.update(file_weights)
            
        except Exception as e:
            logger.error(f"加载文件 {filename} 失败: {e}")
            continue
    
    logger.info(f"总共加载的权重数量: {len(state_dict)} 到设备: {device}")
    
    # 清理权重（如果模型有sanitize方法）
    if hasattr(model, "sanitize"):
        logger.warning(f"sanitize前权重数量: {len(state_dict)}")
        state_dict = model.sanitize(state_dict)
        logger.warning(f"sanitize后权重数量: {len(state_dict)}")
    else:
        logger.warning(f"模型没有sanitize方法，跳过权重适配")
    
    # 加载权重到模型
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"缺失的权重键: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"意外的权重键: {unexpected_keys}")
    
    # 清理临时权重内存
    del state_dict
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"成功加载分片权重到设备: {device}")

def _load_partial_safetensor_weights(file_path: Path, required_keys: List[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    按需加载safetensors格式权重，只加载指定的权重键
    
    Args:
        file_path: safetensors文件路径
        required_keys: 需要加载的权重键列表
        device: 目标设备，默认为"cpu"，可设为"cuda"直接加载到GPU
        
    Returns:
        包含指定权重的字典
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("请安装safetensors: pip install safetensors")
    
    logger.info(f"按需加载safetensors文件: {file_path} (需要 {len(required_keys)} 个权重) 到设备: {device}")
    
    # 确保device是字符串
    device_str = str(device)
    
    partial_weights = {}
    
    # safe_open的device参数只支持CUDA设备，不支持CPU
    if device_str.startswith("cuda"):
        # 对于CUDA设备，先加载到CPU，然后移动到GPU
        with safe_open(file_path, framework="pt") as f:
            # 检查文件中包含哪些键
            available_keys = set(f.keys())
            
            # 找出实际需要的键
            keys_to_load = [key for key in required_keys if key in available_keys]
            missing_keys = [key for key in required_keys if key not in available_keys]
            
            if missing_keys:
                logger.warning(f"文件 {file_path} 中缺失的权重键: {missing_keys}")
            
            # 按需加载权重并移动到GPU - 增强错误处理
            for key in keys_to_load:
                try:
                    tensor = f.get_tensor(key)
                    # 检查是否为视觉权重，添加特殊处理
                    if any(key.startswith(pattern) for pattern in ['model.visual', 'visual', 'model.vision']):
                        logger.debug(f"加载视觉权重: {key}, shape: {tensor.shape}, dtype: {tensor.dtype}")
                    
                    # 移动到GPU
                    partial_weights[key] = tensor.to(device_str)
                except Exception as e:
                    logger.error(f"加载权重 {key} 失败: {e}")
                    # 对于视觉权重，记录更详细的错误信息
                    if any(key.startswith(pattern) for pattern in ['model.visual', 'visual', 'model.vision']):
                        logger.error(f"视觉权重加载失败 - {key}: {e}")
    else:
        # 对于CPU设备，不使用device参数
        with safe_open(file_path, framework="pt") as f:
            # 检查文件中包含哪些键
            available_keys = set(f.keys())
            
            # 找出实际需要的键
            keys_to_load = [key for key in required_keys if key in available_keys]
            missing_keys = [key for key in required_keys if key not in available_keys]
            
            if missing_keys:
                logger.warning(f"文件 {file_path} 中缺失的权重键: {missing_keys}")
            
            # 按需加载权重
            for key in keys_to_load:
                try:
                    tensor = f.get_tensor(key)
                    # 手动移动到CPU设备
                    partial_weights[key] = tensor.to(device_str)
                except Exception as e:
                    logger.error(f"加载权重 {key} 失败: {e}")
    
    logger.info(f"成功加载 {len(partial_weights)} 个权重到设备: {device_str}")
    return partial_weights

def _load_partial_pytorch_weights(file_path: Path, required_keys: List[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    按需加载PyTorch格式权重，只加载指定的权重键
    
    Args:
        file_path: PyTorch权重文件路径
        required_keys: 需要加载的权重键列表
        device: 目标设备，默认为"cpu"，可设为"cuda"直接加载到GPU
        
    Returns:
        包含指定权重的字典
    """
    logger.info(f"按需加载PyTorch文件: {file_path} (需要 {len(required_keys)} 个权重) 到设备: {device}")
    
    try:
        # 加载整个文件到目标设备
        file_weights = torch.load(file_path, map_location=device, weights_only=True)
        
        # 只保留需要的权重
        partial_weights = {}
        missing_keys = []
        
        for key in required_keys:
            if key in file_weights:
                partial_weights[key] = file_weights[key]
            else:
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"文件 {file_path} 中缺失的权重键: {missing_keys}")
        
        logger.info(f"成功加载 {len(partial_weights)} 个权重到设备: {device}")
        return partial_weights
        
    except Exception as e:
        logger.error(f"加载PyTorch文件 {file_path} 失败: {e}")
        return {}

def _load_safetensor_weights(model: torch.nn.Module, files: List[Path], shard: Optional[Shard] = None, device: str = "cpu"):
    """加载Safetensor格式权重 - 支持分片过滤和按需加载"""
    state_dict = {}
    
    # 如果提供了分片信息，先确定需要的权重键
    if shard:
        # 先收集所有可能的权重键
        all_keys = set()
        for file in files:
            try:
                from safetensors import safe_open
                # safetensors的safe_open函数不支持device参数，移除它
                with safe_open(file, framework="pt") as f:
                    all_keys.update(f.keys())
            except Exception as e:
                logger.warning(f"无法读取文件 {file} 的键列表: {e}")
        
        # 根据分片信息过滤需要的权重键
        required_keys = []
        for key in all_keys:
            if key.startswith("model.layers."):
                try:
                    layer_num = int(key.split('.')[2])
                    if shard.start_layer <= layer_num <= shard.end_layer:
                        required_keys.append(key)
                except (ValueError, IndexError):
                    required_keys.append(key)  # 保守策略：包含无法解析的权重
            else:
                required_keys.append(key)  # 非层权重全部包含
        
        # 按需加载每个文件
        for file in files:
            logger.info(f"按需加载safetensors文件: {file}")
            file_weights = _load_partial_safetensor_weights(file, required_keys, device=device)
            state_dict.update(file_weights)
    else:
        # 没有分片信息，使用传统方式加载
        try:
            from safetensors.torch import load_file
            for file in files:
                logger.info(f"加载权重文件: {file} 到设备: {device}")
                file_weights = load_file(file, device=device)
                state_dict.update(file_weights)
        except ImportError:
            raise ImportError("请安装safetensors: pip install safetensors")

    # 清理权重
    if hasattr(model, "sanitize"):
        logger.warning(f"sanitize前权重数量: {len(state_dict)}")
        state_dict = model.sanitize(state_dict)
        logger.warning(f"sanitize方法完成，处理了{len(state_dict)}个权重")
    else:
        logger.warning(f"模型没有sanitize方法，跳过权重适配")

    # 直接加载权重
    result = model.load_state_dict(state_dict, strict=False)
    
    # 处理不同的返回值格式
    if result is None:
        logger.info("权重加载完成（无返回信息）")
        missing_keys = []
        unexpected_keys = []
    else:
        if isinstance(result, tuple) and len(result) == 2:
            missing_keys, unexpected_keys = result
        else:
            logger.warning(f"load_state_dict返回了意外的格式: {type(result)}")
            missing_keys = []
            unexpected_keys = []

    if missing_keys:
        logger.warning(f"缺失的权重键: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"意外的权重键: {unexpected_keys}")
    
    # 清理临时权重内存
    del state_dict
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _load_pytorch_weights(model: torch.nn.Module, files: List[Path], shard: Optional[Shard] = None, device: str = "cpu"):
    """加载PyTorch格式权重 - 支持分片过滤和按需加载"""
    state_dict = {}
    
    # 如果提供了分片信息，先确定需要的权重键
    if shard:
        # 先收集所有可能的权重键
        all_keys = set()
        for file in files:
            try:
                file_weights = torch.load(file, map_location="cpu", weights_only=True)
                all_keys.update(file_weights.keys())
            except Exception as e:
                logger.warning(f"无法读取文件 {file} 的键列表: {e}")
        
        # 根据分片信息过滤需要的权重键
        required_keys = []
        for key in all_keys:
            if key.startswith("model.layers."):
                try:
                    layer_num = int(key.split('.')[2])
                    if shard.start_layer <= layer_num <= shard.end_layer:
                        required_keys.append(key)
                except (ValueError, IndexError):
                    required_keys.append(key)  # 保守策略：包含无法解析的权重
            else:
                required_keys.append(key)  # 非层权重全部包含
        
        # 按需加载每个文件到目标设备
        for file in files:
            logger.info(f"按需加载PyTorch文件: {file} 到设备: {device}")
            file_weights = _load_partial_pytorch_weights(file, required_keys, device)
            state_dict.update(file_weights)
    else:
        # 没有分片信息，使用传统方式加载到目标设备
        for file in files:
            logger.info(f"加载权重文件: {file} 到设备: {device}")
            file_weights = torch.load(file, map_location=device, weights_only=True)
            state_dict.update(file_weights)

    # 清理权重
    if hasattr(model, "sanitize"):
        logger.warning(f"sanitize前权重数量: {len(state_dict)}")
        state_dict = model.sanitize(state_dict)
        logger.warning(f"sanitize方法完成，处理了{len(state_dict)}个权重")
    else:
        logger.warning(f"模型没有sanitize方法，跳过权重适配")

    # 直接加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.warning(f"缺失的权重键: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"意外的权重键: {unexpected_keys}")
    
    # 清理临时权重内存
    del state_dict
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()