from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, AsyncIterator
from pathlib import Path
from exo.inference.shard import Shard
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem


class ShardDownloader(ABC):
  @abstractmethod
  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    """
        Ensures that the shard is downloaded.
        Does not allow multiple overlapping downloads at once.
        If you try to download a Shard which overlaps a Shard that is already being downloaded,
        the download will be cancelled and a new download will start.

        Args:
            shard (Shard): The shard to download.
            inference_engine_name (str): The inference engine used on the node hosting the shard
        """
    pass

  @property
  @abstractmethod
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    pass

  @abstractmethod
  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    """Get the download status of shards.
    
    Returns:
        Optional[Dict[str, float]]: A dictionary mapping shard IDs to their download percentage (0-100),
        or None if status cannot be determined
    """
    pass


class NoopShardDownloader(ShardDownloader):
  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    # 返回已存在的模型路径，根据shard的model_id构建正确的路径
    import os
    from pathlib import Path
    
    # 优先使用环境变量中的路径
    env_path = os.environ.get('MODEL_PATH')
    if env_path:
      return Path(env_path)
    
    # 根据shard的model_id构建路径
    if shard and shard.model_id:
      # 处理模型ID格式，将/Qwen/Qwen3-VL-2B-Instruct转换为Qwen--Qwen3-VL-2B-Instruct
      model_id = shard.model_id
      if '/' in model_id:
        # 移除前缀，只保留模型名称部分
        model_name = model_id.split('/')[-1]
      else:
        model_name = model_id
      
      # 构建本地路径
      base_path = Path(r'C:\Users\nil\.cache\exo\downloads')
      model_path = base_path / f'Qwen--{model_name}'
      
      # 如果路径不存在，尝试其他可能的格式
      if not model_path.exists():
        # 尝试直接使用model_id作为目录名
        alt_path = base_path / model_id.replace('/', '--')
        if alt_path.exists():
          return alt_path
      
      return model_path
    
    # 回退到默认路径
    return Path(r'C:\Users\nil\.cache\exo\downloads\Qwen--Qwen3-0.6B')

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return AsyncCallbackSystem()

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    if False: yield
