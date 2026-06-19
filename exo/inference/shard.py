from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class Shard:
  model_id: str
  start_layer: int
  end_layer: int
  n_layers: int
  repo_id: str = ""
  tie_word_embeddings: bool = True
  instance_id: str = "default"

  def __post_init__(self):
    # Auto-parse instance_id from model_id if not explicitly set
    if self.instance_id == "default" and "::" in self.model_id:
      parts = self.model_id.split("::", 1)
      if len(parts) == 2 and parts[1]:
        object.__setattr__(self, 'instance_id', parts[1])

  @property
  def base_model_id(self) -> str:
    """获取基础模型ID（不含实例信息）"""
    if "::" in self.model_id:
      return self.model_id.split("::")[0]
    return self.model_id

  @property
  def full_model_id(self) -> str:
    """获取完整模型标识（包含实例信息）"""
    if self.instance_id and self.instance_id != "default":
      if "::" in self.model_id:
        return self.model_id
      return f"{self.model_id}::{self.instance_id}"
    return self.model_id

  def __hash__(self):
    return hash((self.base_model_id, self.start_layer, self.end_layer, self.n_layers, self.tie_word_embeddings, self.instance_id))

  def is_first_layer(self) -> bool:
    return self.start_layer == 0

  def is_last_layer(self) -> bool:
    return self.end_layer == self.n_layers - 1

  def get_layer_count(self) -> int:
    return self.end_layer - self.start_layer + 1

  def to_dict(self) -> dict:
    return {
      "model_id": self.model_id,
      "base_model_id": self.base_model_id,
      "full_model_id": self.full_model_id,
      "start_layer": self.start_layer,
      "end_layer": self.end_layer,
      "n_layers": self.n_layers,
      "tie_word_embeddings": self.tie_word_embeddings,
      "instance_id": self.instance_id,
    }

  @staticmethod
  def from_dict(data: dict) -> 'Shard':
    kwargs = {
      "model_id": data.get("model_id", ""),
      "start_layer": data.get("start_layer", 0),
      "end_layer": data.get("end_layer", 0),
      "n_layers": data.get("n_layers", 0),
      "tie_word_embeddings": data.get("tie_word_embeddings", True),
      "instance_id": data.get("instance_id", "default"),
    }
    
    if "repo_id" in data:
      kwargs["repo_id"] = data["repo_id"]
    
    return Shard(**kwargs)

  def overlaps(self, other: 'Shard') -> bool:
    return shards_overlap(self, other)


@dataclass
class ModelLoadState:
  model_id: str
  shard: Shard
  loaded_at: float = field(default_factory=time.time)

  @property
  def base_model_id(self) -> str:
    return self.shard.base_model_id

  @property
  def full_model_id(self) -> str:
    return self.shard.full_model_id

  @property
  def instance_id(self) -> str:
    return self.shard.instance_id

  def to_dict(self) -> dict:
    return {
      "model_id": self.model_id,
      "base_model_id": self.base_model_id,
      "full_model_id": self.full_model_id,
      "instance_id": self.instance_id,
      "shard": self.shard.to_dict(),
      "loaded_at": self.loaded_at
    }

  @staticmethod
  def from_dict(data: dict) -> 'ModelLoadState':
    return ModelLoadState(
      model_id=data.get("model_id", ""),
      shard=Shard.from_dict(data.get("shard", {})),
      loaded_at=data.get("loaded_at", time.time())
    )


def shards_overlap(shard1: Shard, shard2: Shard) -> bool:
  return (shard1.model_id == shard2.model_id and max(shard1.start_layer, shard2.start_layer) <= min(shard1.end_layer, shard2.end_layer))
