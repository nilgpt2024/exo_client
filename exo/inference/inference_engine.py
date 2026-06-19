import numpy as np
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from numpy import ndarray

from exo.helpers import DEBUG

logger = logging.getLogger(__name__)

from typing import Tuple, Optional, List, Any
from abc import ABC, abstractmethod
from .shard import Shard
from exo.download.shard_download import ShardDownloader


class InferenceEngine(ABC):
  _executor: ThreadPoolExecutor | None = None

  def __init__(self):
    self.session = {}
    self.on_model_loaded_callback = None

  @classmethod
  def _get_executor(cls) -> ThreadPoolExecutor:
    if cls._executor is None:
      cls._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="exo_infer")
    return cls._executor

  async def _run_in_executor(self, fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self._get_executor(), lambda: fn(*args, **kwargs))

  def set_on_model_loaded_callback(self, callback):
    """设置模型加载回调函数"""
    self.on_model_loaded_callback = callback

  def _notify_model_loaded(self, shard: Shard):
    """通知模型已加载"""
    if self.on_model_loaded_callback:
      try:
        self.on_model_loaded_callback(shard)
      except Exception as e:
        print(f"[InferenceEngine] Error calling model loaded callback: {e}")

  @abstractmethod
  async def encode(self, shard: Shard, prompt: str, enable_thinking: bool) -> np.ndarray:
    pass

  @abstractmethod
  async def sample(self, x: np.ndarray, temp: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                   repetition_penalty: float = 1.0, generated_tokens: list = None, shard: 'Shard' = None) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    pass

  @abstractmethod
  async def load_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_session(self, key, value):
    self.session[key] = value

  async def clear_session(self):
      """清空会话状态"""
      self.session.clear()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> \
          tuple[ndarray, dict | None]:
    enable_thinking = None
    if inference_state and "enable_thinking" in inference_state:
        enable_thinking = inference_state["enable_thinking"]
        logging.info(f"从inference_state获取思考模式: {'启用' if enable_thinking else '禁用'}")
    tokens = await self.encode(shard, prompt,enable_thinking)

    # 针对不同模型类型处理tokens形状
    if shard.model_id == 'stable-diffusion-2-1-base':
        # stable-diffusion模型保持原始形状
        x = tokens

    output_data, inference_state = await self.infer_tensor(request_id, shard, tokens, inference_state)

    return output_data, inference_state


inference_engine_classes = {
  "pytorch": "PyTorchInferenceEngine",
  "dummy": "DummyInferenceEngine",
}


def get_inference_engine(inference_engine_name, shard_downloader):
    """获取推理引擎
    
    Args:
        inference_engine_name: 引擎类型名称 (pytorch, dummy)
        shard_downloader: 分片下载器
    
    Returns:
        InferenceEngine 实例
    """
    if inference_engine_name == "dummy":
        from exo.inference.dummy_inference_engine import DummyInferenceEngine
        return DummyInferenceEngine()
    
    elif inference_engine_name == "pytorch":
        # 使用统一的 PyTorch 推理引擎，它会根据模型ID自动选择具体实现
        try:
            from exo.inference.pytorch.pytorch_inference_engine import PyTorchInferenceEngine
            return PyTorchInferenceEngine(shard_downloader)
        except Exception as e:
            import traceback, sys
            print(f"[ERROR] 初始化 PyTorch 引擎失败: {e}", file=sys.stderr)
            traceback.print_exc()
            raise
    
    else:
        raise ValueError(f"Unknown or unsupported inference engine: {inference_engine_name}")
