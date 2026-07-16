"""Unlimited-OCR 分片加载工具"""
import json
import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

import torch
from safetensors import safe_open

from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from .unlimited_ocr_model import ShardedUnlimitedOCRModel

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def load_config(model_path: Union[str, Path]) -> Dict[str, Any]:
    """加载模型配置文件"""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in {model_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def _detect_weight_files(model_path: Path) -> Tuple[list, str]:
    """探测权重文件列表，返回 (文件列表, 格式)"""
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if safetensors_files:
        return safetensors_files, "safetensors"
    bin_files = sorted(model_path.glob("*.bin"))
    if bin_files:
        return bin_files, "bin"
    raise FileNotFoundError(f"No weight files found in {model_path}")


def _load_safetensors_file(path: Path) -> Dict[str, torch.Tensor]:
    """加载单个 safetensors 文件"""
    state_dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def _load_bin_file(path: Path) -> Dict[str, torch.Tensor]:
    """加载单个 bin 文件"""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _load_full_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
    """加载完整 state_dict"""
    weight_files, fmt = _detect_weight_files(model_path)
    logger.info(f"[UnlimitedOCR] 检测到 {fmt} 权重文件: {len(weight_files)} 个")

    state_dict = {}
    for wf in weight_files:
        logger.info(f"[UnlimitedOCR] 加载权重文件: {wf.name}")
        if fmt == "safetensors":
            partial = _load_safetensors_file(wf)
        else:
            partial = _load_bin_file(wf)
        state_dict.update(partial)

    return state_dict


def _init_meta_buffers(model: torch.nn.Module, device: torch.device) -> None:
    """初始化仍留在 meta device 上的 buffer（如 inv_freq 等）"""
    meta_count = 0
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == "meta":
            try:
                new_buffer = torch.zeros(buffer.shape, dtype=buffer.dtype, device=device)
                parent_module, attr_name = _get_parent_module_and_attr(model, name)
                setattr(parent_module, attr_name, new_buffer)
                meta_count += 1
            except Exception as e:
                logger.warning(f"[UnlimitedOCR] 初始化 meta buffer '{name}' 失败: {e}")
    if meta_count > 0:
        logger.warning(f"[UnlimitedOCR] 共初始化 {meta_count} 个 meta buffer，可能影响数值正确性")


def _init_meta_parameters(model: torch.nn.Module, device: torch.device) -> None:
    """初始化仍留在 meta device 上的参数"""
    meta_count = 0
    for name, param in model.named_parameters():
        if param is not None and param.device.type == "meta":
            try:
                new_param = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device=device),
                    requires_grad=param.requires_grad,
                )
                parent_module, attr_name = _get_parent_module_and_attr(model, name)
                setattr(parent_module, attr_name, new_param)
                meta_count += 1
            except Exception as e:
                logger.warning(f"[UnlimitedOCR] 初始化 meta parameter '{name}' 失败: {e}")
    if meta_count > 0:
        logger.warning(f"[UnlimitedOCR] 共初始化 {meta_count} 个 meta parameter，可能影响数值正确性")


def _get_parent_module_and_attr(model: torch.nn.Module, full_name: str) -> Tuple[torch.nn.Module, str]:
    """根据全名获取父模块和属性名"""
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def load_model_shard(
    model_path: Union[str, Path],
    shard: Shard,
    lazy: bool = False,
    model_config: Dict[str, Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    use_bf16: bool = False,
    trust_remote_code: bool = True,
) -> ShardedUnlimitedOCRModel:
    """加载并裁剪 Unlimited-OCR 模型分片"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(f"Model path not found: {model_path}")

    config = load_config(model_path)
    config.update(model_config or {})

    # 确定设备和精度
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device) if isinstance(device, str) else device
    target_dtype = torch.bfloat16 if use_bf16 and target_device.type == "cuda" else torch.float32

    logger.info(
        f"[UnlimitedOCR] 加载分片: {shard.model_id} layers {shard.start_layer}-{shard.end_layer}, "
        f"device={target_device}, dtype={target_dtype}"
    )

    # 创建裁剪后的 meta device 模型
    model = ShardedUnlimitedOCRModel(
        model_path=model_path,
        shard=shard,
        device=target_device,
        dtype=target_dtype,
        trust_remote_code=trust_remote_code,
    )

    # 关键：先把 meta device 上的模型骨架移动到目标设备（不初始化数据），
    # 否则 load_state_dict 无法把 meta tensor 直接搬到真实设备。
    logger.info(f"[UnlimitedOCR] 移动模型骨架到 {target_device}")
    model.model = model.model.to_empty(device=target_device)

    # 加载并过滤权重
    logger.info("[UnlimitedOCR] 开始加载完整权重并过滤...")
    full_state_dict = _load_full_state_dict(model_path)
    filtered_state_dict = ShardedUnlimitedOCRModel.sanitize(full_state_dict, shard)

    # 把权重移到目标设备（state_dict 默认在 CPU）
    filtered_state_dict = {
        k: v.to(target_device, dtype=target_dtype) if torch.is_tensor(v) else v
        for k, v in filtered_state_dict.items()
    }

    # 清理内存
    del full_state_dict

    # 加载到模型（assign=True 用于替换 meta tensor）
    missing, unexpected = model.model.load_state_dict(filtered_state_dict, strict=False, assign=True)
    if missing:
        logger.warning(f"[UnlimitedOCR] 加载后缺少参数: {missing[:10]}...")
    if unexpected:
        logger.warning(f"[UnlimitedOCR] 加载后多余参数: {unexpected[:10]}...")

    del filtered_state_dict

    # 初始化仍缺失的 meta buffer / 参数
    _init_meta_buffers(model.model, target_device)
    _init_meta_parameters(model.model, target_device)

    model.model.eval()
    logger.info("[UnlimitedOCR] 分片加载完成")
    return model


async def load_shard(
    model_path: str,
    shard: Shard,
    model_config: Dict[str, Any] = None,
    lazy: bool = False,
    executor=None,
    device: Optional[Union[str, torch.device]] = None,
    use_bf16: bool = False,
) -> Tuple[ShardedUnlimitedOCRModel, Any]:
    """异步加载模型分片和 tokenizer

    返回 (model_shard, tokenizer)
    """
    # 同步加载模型
    model_shard = load_model_shard(
        model_path=Path(model_path),
        shard=shard,
        lazy=lazy,
        model_config=model_config or {},
        device=device,
        use_bf16=use_bf16,
    )

    # 加载 tokenizer / processor
    try:
        from transformers import AutoProcessor, AutoTokenizer
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"[UnlimitedOCR] AutoProcessor 加载失败: {e}，回退到 AutoTokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = None

    # 把 processor 挂在 tokenizer 上，方便 engine 透传图像字段
    if processor is not None and tokenizer is not None:
        tokenizer._processor = processor

    return model_shard, tokenizer
