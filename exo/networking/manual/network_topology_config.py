from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ValidationError, ConfigDict

from exo.topology.device_capabilities import DeviceCapabilities


class ShardConfig(BaseModel):
  """模型分片配置"""
  model_config = ConfigDict(protected_namespaces=())

  model_id: str
  start_layer: int
  end_layer: int
  n_layers: int
  repo_id: Optional[str] = ""


class PeerConfig(BaseModel):
  address: str
  port: int
  device_capabilities: DeviceCapabilities
  shard: Optional[Union[ShardConfig, List[ShardConfig]]] = None  # 支持单个或多个模型分片配置


class NetworkTopology(BaseModel):
  """Configuration of the network. A collection outlining all nodes in the network, including the node this is running from."""

  peers: Dict[str, PeerConfig]
  """
  node_id to PeerConfig. The node_id is used to identify the peer in the discovery process. The node that this is running from should be included in this dict.
  """
  @classmethod
  def from_path(cls, path: str) -> "NetworkTopology":
    try:
      with open(path, "r") as f:
        config_data = f.read()
    except FileNotFoundError as e:
      raise FileNotFoundError(f"Config file not found at {path}") from e

    try:
      return cls.model_validate_json(config_data)
    except ValidationError as e:
      raise ValueError(f"Error validating network topology config from {path}: {e}") from e
