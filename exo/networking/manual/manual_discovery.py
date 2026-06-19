import os
import asyncio
from typing import Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from exo.networking.discovery import Discovery
from exo.topology.device_capabilities import DeviceCapabilities
from exo.networking.manual.network_topology_config import NetworkTopology, PeerConfig, ShardConfig
from exo.inference.shard import Shard
from exo.helpers import DEBUG_DISCOVERY
from exo.networking.peer_handle import PeerHandle


class ManualDiscovery(Discovery):
  def __init__(
    self,
    network_config_path: Optional[str],
    node_id: str,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
  ):
    self.network_config_path = network_config_path
    self.node_id = node_id
    self.create_peer_handle = create_peer_handle

    self.listen_task = None
    self.known_peers: Dict[str, PeerHandle] = {}

    self._cached_peers: Dict[str, PeerConfig] = {}
    self._last_modified_time: Optional[float] = None
    self._file_executor = ThreadPoolExecutor(max_workers=1)
    self._current_node_config: Optional[PeerConfig] = None  # 当前节点的配置

  async def load_node_config(self) -> Optional[PeerConfig]:
    """主动加载当前节点配置"""
    if not self.network_config_path:
      print(f"[ManualDiscovery] No config path provided, skipping config loading")
      return None
    
    try:
      loop = asyncio.get_running_loop()
      topology = await loop.run_in_executor(self._file_executor, NetworkTopology.from_path, self.network_config_path)
      
      if self.node_id not in topology.peers:
        print(f"[ManualDiscovery] Warning: Node {self.node_id} not found in config")
        return None
      
      self._current_node_config = topology.peers[self.node_id]

      if self._current_node_config.shard:
        shard_config = self._current_node_config.shard
        if isinstance(shard_config, list):
          print(f"[ManualDiscovery] Node {self.node_id} configured with {len(shard_config)} shards:")
          for sc in shard_config:
            print(f"  - model={sc.model_id}, layers={sc.start_layer}-{sc.end_layer}/{sc.n_layers}")
        else:
          print(f"[ManualDiscovery] Node {self.node_id} configured with shard: "
                f"model={shard_config.model_id}, layers={shard_config.start_layer}-{shard_config.end_layer}/{shard_config.n_layers}")

      return self._current_node_config
    except Exception as e:
      print(f"[ManualDiscovery] Error loading node config: {e}")
      return None

  def get_current_node_shard(self, model_id: Optional[str] = None) -> Optional[Shard]:
    """获取当前节点配置的分片信息

    Args:
      model_id: 可选，指定要获取的模型ID。如果配置了多个shard且未指定model_id，返回第一个shard

    Returns:
      Shard对象或None
    """
    if self._current_node_config and self._current_node_config.shard:
      shard_config = self._current_node_config.shard

      # 处理多个shard的情况
      if isinstance(shard_config, list):
        if model_id:
          # 根据model_id查找对应的shard
          for sc in shard_config:
            if sc.model_id == model_id:
              return Shard(
                model_id=sc.model_id,
                start_layer=sc.start_layer,
                end_layer=sc.end_layer,
                n_layers=sc.n_layers,
                repo_id=sc.repo_id or ""
              )
          return None
        else:
          # 未指定model_id，返回第一个shard
          sc = shard_config[0]
          return Shard(
            model_id=sc.model_id,
            start_layer=sc.start_layer,
            end_layer=sc.end_layer,
            n_layers=sc.n_layers,
            repo_id=sc.repo_id or ""
          )
      else:
        # 单个shard的情况
        return Shard(
          model_id=shard_config.model_id,
          start_layer=shard_config.start_layer,
          end_layer=shard_config.end_layer,
          n_layers=shard_config.n_layers,
          repo_id=shard_config.repo_id or ""
        )
    return None

  def get_current_node_shards(self) -> List[Shard]:
    """获取当前节点配置的所有分片信息"""
    shards = []
    if self._current_node_config and self._current_node_config.shard:
      shard_config = self._current_node_config.shard

      if isinstance(shard_config, list):
        for sc in shard_config:
          shards.append(Shard(
            model_id=sc.model_id,
            start_layer=sc.start_layer,
            end_layer=sc.end_layer,
            n_layers=sc.n_layers,
            repo_id=sc.repo_id or ""
          ))
      else:
        shards.append(Shard(
          model_id=shard_config.model_id,
          start_layer=shard_config.start_layer,
          end_layer=shard_config.end_layer,
          n_layers=shard_config.n_layers,
          repo_id=shard_config.repo_id or ""
        ))
    return shards

  async def start(self) -> None:
    self.listen_task = asyncio.create_task(self.task_find_peers_from_config())

  async def stop(self) -> None:
    if self.listen_task: self.listen_task.cancel()
    self._file_executor.shutdown(wait=True)

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2: print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    if DEBUG_DISCOVERY >= 2: print(f"Discovered peers: {[peer.id() for peer in self.known_peers.values()]}")
    return list(self.known_peers.values())

  def add_known_node(self, node_id: str, address: str, port: int, description: str = "Manager", device_capabilities: Optional[DeviceCapabilities] = None):
    if node_id == self.node_id or node_id in self.known_peers:
      return
    
    peer_config = PeerConfig(
      id=node_id,
      address=address,
      port=port,
      device_capabilities=device_capabilities or DeviceCapabilities()
    )
    
    peer = self.create_peer_handle(node_id, f"{address}:{port}", "MAN", peer_config.device_capabilities)
    self.known_peers[node_id] = peer
    print(f"[ManualDiscovery] 添加已知节点: {node_id} @ {address}:{port} (来源: {description})")

  async def _check_peer_health(self, peer_id: str, peer_config: PeerConfig) -> Optional[PeerHandle]:
    """检查单个节点的健康状态，带超时"""
    try:
      peer = self.known_peers.get(peer_id)
      if not peer:
        if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} not found in known peers. Adding.")
        peer = self.create_peer_handle(peer_id, f"{peer_config.address}:{peer_config.port}", "MAN", peer_config.device_capabilities)
      
      # 设置 3 秒超时进行 health check
      is_healthy = await asyncio.wait_for(peer.health_check(), timeout=3.0)
      
      if is_healthy:
        if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is healthy.")
        return peer
      else:
        if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is not healthy.")
        return None
    except asyncio.TimeoutError:
      if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} health check timeout.")
      return None
    except Exception as e:
      if DEBUG_DISCOVERY >= 2: print(f"Exception occurred when checking {peer_id=}: {e}")
      return None

  async def task_find_peers_from_config(self):
    if DEBUG_DISCOVERY >= 2: print("Starting task to find peers from config...")
    
    # 首次运行时立即检查一次
    first_run = True
    
    while True:
      peers_from_config = await self._get_peers()
      
      if DEBUG_DISCOVERY >= 2: 
        print(f"Checking {len(peers_from_config)} peers from config...")
      
      # 并行检查所有节点的健康状态
      if peers_from_config:
        health_check_tasks = [
          self._check_peer_health(peer_id, peer_config)
          for peer_id, peer_config in peers_from_config.items()
        ]
        peer_results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
        
        # 收集健康的节点
        new_known_peers = {}
        for peer_id, result in zip(peers_from_config.keys(), peer_results):
          if isinstance(result, Exception):
            if DEBUG_DISCOVERY >= 2: print(f"Health check failed for {peer_id=}: {result}")
          elif result is not None:
            new_known_peers[peer_id] = result
        
        # 检测新上线的节点
        new_peers = set(new_known_peers.keys()) - set(self.known_peers.keys())
        if new_peers:
          print(f"[ManualDiscovery] New peers online: {new_peers}")
        
        # 检测下线的节点
        removed_peers = set(self.known_peers.keys()) - set(new_known_peers.keys())
        if removed_peers:
          print(f"[ManualDiscovery] Peers offline: {removed_peers}")
        
        self.known_peers = new_known_peers
      else:
        self.known_peers = {}
      
      if DEBUG_DISCOVERY >= 2: 
        print(f"Current known peers: {[peer.id() for peer in self.known_peers.values()]}")
      
      # 首次运行后，使用更短的间隔
      if first_run:
        await asyncio.sleep(1.0)
        first_run = False
      else:
        await asyncio.sleep(2.0)

  async def _get_peers(self):
    if not self.network_config_path:
      return {}
    
    try:
      loop = asyncio.get_running_loop()
      current_mtime = await loop.run_in_executor(self._file_executor, os.path.getmtime, self.network_config_path)

      if (self._cached_peers is not None and self._last_modified_time is not None and current_mtime <= self._last_modified_time):
        return self._cached_peers

      topology = await loop.run_in_executor(self._file_executor, NetworkTopology.from_path, self.network_config_path)

      if self.node_id not in topology.peers:
        raise ValueError(
          f"Node ID {self.node_id} not found in network config file "
          f"{self.network_config_path}. Please run with `node_id` set to "
          f"one of the keys in the config file: {[k for k, _ in topology.peers]}"
        )

      # 保存当前节点的配置
      self._current_node_config = topology.peers[self.node_id]

      # 如果有分片配置，打印日志
      if self._current_node_config.shard:
        shard_config = self._current_node_config.shard
        if isinstance(shard_config, list):
          print(f"[ManualDiscovery] Node {self.node_id} configured with {len(shard_config)} shards:")
          for sc in shard_config:
            print(f"  - model={sc.model_id}, layers={sc.start_layer}-{sc.end_layer}/{sc.n_layers}")
        else:
          print(f"[ManualDiscovery] Node {self.node_id} configured with shard: "
                f"model={shard_config.model_id}, layers={shard_config.start_layer}-{shard_config.end_layer}/{shard_config.n_layers}")

      peers_in_network = topology.peers.copy()
      peers_in_network.pop(self.node_id)

      self._cached_peers = peers_in_network
      self._last_modified_time = current_mtime

      return peers_in_network

    except Exception as e:
      if DEBUG_DISCOVERY >= 2:
        print(f"Error when loading network config file from {self.network_config_path}. "
              f"Please update the config file in order to successfully discover peers. "
              f"Exception: {e}")
      return self._cached_peers
