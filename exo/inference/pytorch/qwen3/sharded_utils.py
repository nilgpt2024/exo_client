import glob
import json
import logging
import torch
import torch.nn as nn
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
        if model_type == "qwen3":
            from .qwen3 import Model, ModelArgs
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
    
    # 使用 meta device 创建模型（不分配内存，不初始化参数）
    # 这比标准初始化快 580 倍（0.3 秒 vs 177 秒）
    # 因为 nn.Embedding 的随机初始化是主要瓶颈
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        with torch.device("meta"):
            model = model_class(model_args)
    finally:
        torch.set_default_dtype(original_dtype)

    if not lazy:
        # 加载权重并替换 meta device 上的参数
        load_model_weights(model, model_path, shard, device=target_device, target_dtype=target_dtype)
        
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
        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        model = load_model_shard(
            Path(model_path),
            shard,
            lazy,
            model_config,
            device=device,
            use_bf16=use_bf16
        )
        
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"模型加载后 GPU 内存: {memory_allocated:.2f}GB")
        
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

def load_model_weights(model: torch.nn.Module, model_path: Path, shard: Optional[Shard] = None, device: str = "cpu", target_dtype: Optional[torch.dtype] = None):
    """加载模型权重 - 支持.index.json分片加载和 meta device 模型"""
    is_meta = any(p.device.type == "meta" for p in model.parameters())
    if is_meta:
        logger.info(f"检测到 meta device 模型，使用高效加载方式...")
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
    """通过.index.json文件加载分片权重 - 使用 load_file 高效加载"""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("请安装safetensors: pip install safetensors")

    is_meta = any(p.device.type == "meta" for p in model.parameters())
    logger.info(f"使用索引文件加载分片权重到设备: {device} (meta_device={is_meta})")
    
    # 读取索引文件
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data.get('weight_map', {})
    logger.info(f"索引文件中总权重数量: {len(weight_map)}")
    
    # 过滤权重映射 - 根据分片信息只保留需要的权重
    filtered_keys = set()
    
    for key in weight_map:
        if shard and key.startswith("model.layers."):
            try:
                layer_num = int(key.split('.')[2])
                if layer_num < shard.start_layer or layer_num > shard.end_layer:
                    continue
            except (ValueError, IndexError):
                pass
        filtered_keys.add(key)
    
    logger.info(f"分片 {shard} 保留的权重键数量: {len(filtered_keys)}")
    
    # 使用 load_file 一次性加载每个文件，然后过滤需要的权重
    # 这比 safe_open 逐个读取快 100+ 倍
    state_dict = {}
    base_path = index_file.parent
    
    # 收集需要加载的文件
    files_to_load = set()
    for key in filtered_keys:
        files_to_load.add(weight_map[key])
    
    for filename in files_to_load:
        file_path = base_path / filename
        if not file_path.exists():
            logger.warning(f"权重文件不存在: {file_path}")
            continue
        
        logger.info(f"加载权重文件: {filename}")
        
        try:
            if filename.endswith(".safetensors"):
                file_weights = load_file(file_path)
            else:
                file_weights = torch.load(file_path, map_location="cpu", weights_only=True)
            
            # 只保留需要的权重
            for key in filtered_keys:
                if key in file_weights:
                    state_dict[key] = file_weights[key]
            
            del file_weights
            
        except Exception as e:
            logger.error(f"加载文件 {filename} 失败: {e}")
            continue
    
    logger.info(f"总共加载的权重数量: {len(state_dict)}")
    
    # 清理权重（如果模型有sanitize方法）
    if hasattr(model, "sanitize"):
        logger.info(f"sanitize前权重数量: {len(state_dict)}")
        state_dict = model.sanitize(state_dict)
        logger.info(f"sanitize后权重数量: {len(state_dict)}")
    else:
        logger.info(f"模型没有sanitize方法，跳过权重适配")
    
    # 加载权重到模型
    if is_meta:
        # meta device 模型需要特殊处理：逐个替换参数
        logger.info(f"使用 meta device 加载方式，逐个替换参数到设备: {device}")
        model = _load_weights_to_meta_model(model, state_dict, device)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"缺失的权重键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"意外的权重键: {unexpected_keys}")
    
    del state_dict
    import gc
    gc.collect()
    
    logger.info(f"成功加载分片权重到设备: {device}")

def _load_weights_to_meta_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], device: str = "cpu") -> torch.nn.Module:
    """将权重加载到 meta device 模型 - 直接替换参数到目标设备（绕过 load_state_dict）"""
    import gc
    
    target_device = torch.device(device)
    
    loaded_count = 0
    
    # 直接替换参数 - 比 load_state_dict 快 2500 倍
    for name, param in model.named_parameters():
        if name in state_dict:
            weight = state_dict[name].to(device=target_device)
            parts = name.split('.')
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], nn.Parameter(weight, requires_grad=False))
            loaded_count += 1
    
    # 处理 meta device 上的 buffers（如 rotary_emb 的 inv_freq）
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type == "meta":
            parts = name.split('.')
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            if 'inv_freq' in name:
                try:
                    if hasattr(obj, 'dim') and hasattr(obj, 'theta') and not hasattr(getattr(type(obj), 'compute_default_rope_parameters', None), '__call__'):
                        inv_freq = 1.0 / (obj.theta ** (torch.arange(0, obj.dim, 2, dtype=torch.float32, device=target_device) / obj.dim))
                        obj.register_buffer(parts[-1], inv_freq)
                        if hasattr(obj, 'original_inv_freq'):
                            obj.register_buffer("original_inv_freq", inv_freq.clone())
                    elif hasattr(obj, 'config'):
                        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
                        inv_freq, attention_scaling = Qwen3RotaryEmbedding.compute_default_rope_parameters(obj.config, device=target_device)
                        obj.register_buffer("inv_freq", inv_freq)
                        obj.attention_scaling = attention_scaling
                        if hasattr(obj, 'original_inv_freq'):
                            obj.register_buffer("original_inv_freq", inv_freq.clone())
                    else:
                        raise AttributeError("无法确定 RoPE 初始化方式")
                except Exception as e:
                    logger.warning(f"重新初始化 {name} 失败: {e}")
                    new_buffer = torch.empty(buffer.shape, dtype=buffer.dtype, device=target_device)
                    obj.register_buffer(parts[-1], new_buffer)
            else:
                new_buffer = torch.empty(buffer.shape, dtype=buffer.dtype, device=target_device)
                obj.register_buffer(parts[-1], new_buffer)
    
    logger.info(f"成功加载 {loaded_count} 个参数到设备: {device}")
    
    gc.collect()
    if target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model

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
    """加载Safetensor格式权重 - 使用 load_file 高效加载"""
    is_meta = any(p.device.type == "meta" for p in model.parameters())
    
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("请安装safetensors: pip install safetensors")
    
    state_dict = {}
    
    # 使用 load_file 一次性加载每个文件
    for file in files:
        logger.info(f"加载权重文件: {file}")
        try:
            file_weights = load_file(file)
            state_dict.update(file_weights)
            del file_weights
        except Exception as e:
            logger.error(f"加载文件 {file} 失败: {e}")
            continue
    
    # 根据分片信息过滤权重
    if shard:
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.layers."):
                try:
                    layer_num = int(key.split('.')[2])
                    if shard.start_layer <= layer_num <= shard.end_layer:
                        if hasattr(model, "sanitize"):
                            # sanitize will handle index remapping later
                            filtered_state_dict[key] = value
                        else:
                            # No sanitize method, remap layer indices here
                            remapped_key = key.replace(
                                f"model.layers.{layer_num}.",
                                f"model.layers.{layer_num - shard.start_layer}.",
                                1
                            )
                            filtered_state_dict[remapped_key] = value
                except (ValueError, IndexError):
                    filtered_state_dict[key] = value
            else:
                filtered_state_dict[key] = value
        state_dict = filtered_state_dict

    # 清理权重
    if hasattr(model, "sanitize"):
        logger.info(f"sanitize前权重数量: {len(state_dict)}")
        state_dict = model.sanitize(state_dict)
        logger.info(f"sanitize后权重数量: {len(state_dict)}")
    else:
        logger.info(f"模型没有sanitize方法，跳过权重适配")

    # 加载权重到模型
    if is_meta:
        logger.info(f"使用 meta device 加载方式，逐个替换参数到设备: {device}")
        model = _load_weights_to_meta_model(model, state_dict, device)
    else:
        result = model.load_state_dict(state_dict, strict=False)
        if result is None:
            logger.info("权重加载完成（无返回信息）")
        elif isinstance(result, tuple) and len(result) == 2:
            missing_keys, unexpected_keys = result
            if missing_keys:
                logger.warning(f"缺失的权重键: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"意外的权重键: {unexpected_keys}")
    
    del state_dict
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _load_pytorch_weights(model: torch.nn.Module, files: List[Path], shard: Optional[Shard] = None, device: str = "cpu"):
    """加载PyTorch格式权重 - 支持分片过滤和 meta device"""
    is_meta = any(p.device.type == "meta" for p in model.parameters())
    state_dict = {}
    
    for file in files:
        logger.info(f"加载权重文件: {file}")
        try:
            file_weights = torch.load(file, map_location="cpu", weights_only=True)
            state_dict.update(file_weights)
            del file_weights
        except Exception as e:
            logger.error(f"加载文件 {file} 失败: {e}")
            continue
    
    # 根据分片信息过滤权重
    if shard:
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.layers."):
                try:
                    layer_num = int(key.split('.')[2])
                    if shard.start_layer <= layer_num <= shard.end_layer:
                        if hasattr(model, "sanitize"):
                            # sanitize will handle index remapping later
                            filtered_state_dict[key] = value
                        else:
                            # No sanitize method, remap layer indices here
                            remapped_key = key.replace(
                                f"model.layers.{layer_num}.",
                                f"model.layers.{layer_num - shard.start_layer}.",
                                1
                            )
                            filtered_state_dict[remapped_key] = value
                except (ValueError, IndexError):
                    filtered_state_dict[key] = value
            else:
                filtered_state_dict[key] = value
        state_dict = filtered_state_dict

    # 清理权重
    if hasattr(model, "sanitize"):
        logger.info(f"sanitize前权重数量: {len(state_dict)}")
        state_dict = model.sanitize(state_dict)
        logger.info(f"sanitize后权重数量: {len(state_dict)}")
    else:
        logger.info(f"模型没有sanitize方法，跳过权重适配")

    # 加载权重到模型
    if is_meta:
        logger.info(f"使用 meta device 加载方式，逐个替换参数到设备: {device}")
        model = _load_weights_to_meta_model(model, state_dict, device)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"缺失的权重键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"意外的权重键: {unexpected_keys}")
    
    del state_dict
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()