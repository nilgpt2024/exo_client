import numpy as np
import json
import asyncio
import uuid
import time
import traceback
import logging
from typing import List, Dict, Optional, Tuple, Union, Set, Any
from exo.networking import Discovery, PeerHandle, Server
from exo.inference.inference_engine import InferenceEngine, Shard
from exo.inference.shard import ModelLoadState
from exo.topology.topology import Topology
from exo.topology.device_capabilities import device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards
from exo.topology.model_aware_partitioning_strategy import ModelAwarePartitioningStrategy
from exo import DEBUG
from exo.helpers import AsyncCallbackSystem
from exo.viz.topology_viz import TopologyViz
from exo.download.download_progress import RepoProgressEvent
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.download.shard_download import ShardDownloader

# GPU池管理器（可选依赖，导入失败不影响核心功能）
try:
  from exo.topology.gpu_pool_api import GPUPoolAPI
  _GPU_POOL_AVAILABLE = True
except ImportError as e:
  GPUPoolAPI = None
  _GPU_POOL_AVAILABLE = False
  import logging
  logging.getLogger(__name__).warning(f"GPU Pool API unavailable: {e}")

# WebSocket 增强版管理器（可选依赖，导入失败不影响核心功能）
try:
  from exo.networking.websocket_optimized import (
    MessagePriority,
    EnhancedWebSocketManager
  )
  _WS_V2_AVAILABLE = True
except ImportError as e:
  MessagePriority = None
  EnhancedWebSocketManager = None
  _WS_V2_AVAILABLE = False
  import logging
  logging.getLogger(__name__).warning(f"WebSocket V2 Manager unavailable: {e}")

class Node:
  def __init__(
    self,
    _id: str,
    server: Server,
    inference_engine: InferenceEngine,
    discovery: Discovery,
    shard_downloader: ShardDownloader,
    partitioning_strategy: PartitioningStrategy = None,
    max_generate_tokens: int = 1024,
    default_sample_temperature: float = 0.7,
    topology_viz: Optional[TopologyViz] = None,
    manager_url: Optional[str] = None,
    chatgpt_api_port: int = 52415,
    auto_connect: bool = True,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.server = server
    self.discovery = discovery
    self.shard_downloader = shard_downloader
    self.manager_url = manager_url
    self.chatgpt_api_port = chatgpt_api_port
    self.auto_connect = auto_connect

    if manager_url:
      from exo.models import init_models
      init_models(manager_url)
    # 订阅下载进度事件
    self.shard_downloader.on_progress.register("node_progress").on_next(self.on_download_progress)
    self.partitioning_strategy = partitioning_strategy
    self.peers: List[PeerHandle] = {}
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}
    # 存储每个请求的 KV 缓存，让每个节点自己维护自己的 KV 缓存状态
    self.request_kv_cache: Dict[str, any] = {}

    # [STAR] 多实例推理引擎池：支持每个 instance_id 拥有独立的引擎实例
    # key: instance_id (如 "default", "worker-1", "worker-2")
    # value: InferenceEngine 实例
    self.inference_engines: Dict[str, InferenceEngine] = {"default": inference_engine}
    inference_engine._instance_id = "default"

    # 设置模型加载回调（仅对默认引擎）
    self.inference_engine.set_on_model_loaded_callback(self.on_model_loaded)
    
    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self.default_sample_temperature = default_sample_temperature
    self.default_top_p = 0.9
    self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.outstanding_requests = {}
    
    # 节点性能统计
    self.node_stats: Dict[str, Dict] = {}  # 每个节点的性能统计
    
    # 节点分片配置信息（从广播收集）
    self.node_shards: Dict[str, Shard] = {}  # node_id -> Shard (向后兼容，保存第一个分片)
    self.node_shards_multi: Dict[str, List[Shard]] = {}  # node_id -> List[Shard] (多模型分片支持)
    
    # 模型加载状态跟踪
    self.my_loaded_models: Dict[str, ModelLoadState] = {}  # 自己加载的模型
    self.node_loaded_models: Dict[str, Dict[str, ModelLoadState]] = {}  # node_id -> {model_id -> ModelLoadState}
    
    # 统一GPU显存池管理器（用于统一调配所有节点的模型权重）
    self.gpu_pool: Optional[Any] = None  # 延迟初始化，在 start() 中创建

    # 推理状态标志位 - 用于优化：推理期间暂停后台拓扑更新
    self.is_inferencing: bool = False
    self._inference_count: int = 0  # 当前活跃的推理请求数量
    self._topology_update_task: Optional[asyncio.Task] = None  # 当前运行的拓扑更新任务

  async def _enter_inference(self):
    """标记进入推理状态（支持嵌套调用）"""
    self._inference_count += 1
    if self._inference_count == 1:
      self.is_inferencing = True
      
      # 取消正在进行的拓扑更新任务，立即释放资源给推理
      if self._topology_update_task and not self._topology_update_task.done():
        self._topology_update_task.cancel()
        logging.info("[InferenceState] [FAST] 已取消拓扑更新任务，优先处理推理")
      
      logging.info(f"[InferenceState] 进入推理模式 (active_requests={self._inference_count})")

  async def _exit_inference(self):
    """标记退出推理状态"""
    self._inference_count = max(0, self._inference_count - 1)
    if self._inference_count == 0:
      self.is_inferencing = False
      logging.info(f"[InferenceState] 退出推理模式 (active_requests={self._inference_count})")

  def get_engine(self, instance_id: str = "default") -> InferenceEngine:
    """获取指定实例的推理引擎

    Args:
      instance_id: 实例ID（如 "default", "worker-1", "worker-2"）

    Returns:
      对应的 InferenceEngine 实例

    说明:
      - 如果实例不存在，会自动创建新的引擎实例
      - 每个实例拥有独立的模型权重和状态，互不干扰
      - 向后兼容：instance_id="default" 返回原始引擎
    """
    if instance_id not in self.inference_engines:
      print(f"[Node] [STAR] Create new engine instance: {instance_id} (pool: {list(self.inference_engines.keys())})")
      new_engine = self._create_engine_instance(instance_id)
      self.inference_engines[instance_id] = new_engine
      print(f"[Node] [OK] 引擎池当前实例数: {len(self.inference_engines)}")
    else:
      engine = self.inference_engines[instance_id]
      if hasattr(engine, 'dump_state'):
        state = engine.dump_state()
        if state.get('call_count', 0) > 0 and len(state.get('cached_models', [])) == 0:
          print(f"[Node] [WARN] 引擎 {instance_id} 已使用 {state['call_count']} 次但缓存为空! "
                f"历史: {state.get('recent_history', [])[-3:]}")

    return self.inference_engines[instance_id]

  def _create_engine_instance(self, instance_id: str) -> InferenceEngine:
    """创建新的推理引擎实例（与默认引擎相同的配置）

    Args:
      instance_id: 新实例的ID

    Returns:
      新创建的 InferenceEngine 实例
    """
    import copy

    # 获取默认引擎的类型和构造参数
    default_engine = self.inference_engines["default"]
    engine_class = type(default_engine)

    print(f"[Node] [INIT] 创建引擎实例 [{instance_id}]，类型: {engine_class.__name__}")

    # 通过反射获取构造函数参数
    # 大多数引擎需要: shard_downloader, model_path, device
    try:
      if hasattr(default_engine, 'shard_downloader'):
        new_engine = engine_class(
          shard_downloader=default_engine.shard_downloader,
          model_path=getattr(default_engine, 'model_path', None),
          device=str(getattr(default_engine, 'device', 'cuda'))
        )
      else:
        # 备用方案：直接实例化（适用于简单引擎）
        new_engine = engine_class()

      # 设置模型加载回调
      new_engine.set_on_model_loaded_callback(self.on_model_loaded)

      new_engine._instance_id = instance_id

      print(f"[Node] [OK] 引擎实例 [{instance_id}] 创建成功")
      return new_engine

    except Exception as e:
      print(f"[Node] [FAIL] 创建引擎实例 [{instance_id}] 失败: {e}")
      raise

  async def start(self, wait_for_peers: int = 0) -> None:
    # 只有在设备能力未被手动设置时才进行自动检测
    if self.device_capabilities == UNKNOWN_DEVICE_CAPABILITIES:
      self.device_capabilities = await device_capabilities()
    if self.server is not None:
      await self.server.start()
    await self.discovery.start()
    await self.update_peers(wait_for_peers)
    await self.collect_topology(set())
    print(f"Collected topology: {self.topology}")
    
    # 主动注册到 EXO Manager（如果配置了）- 延迟2秒确保server完全就绪
    if self.manager_url:
      asyncio.create_task(self._delayed_register_to_manager(delay=2.0))
    
    # 初始化统一GPU显存池管理器（可选功能）
    if _GPU_POOL_AVAILABLE and GPUPoolAPI is not None:
      try:
        self.gpu_pool = GPUPoolAPI(self)
        await self.gpu_pool.initialize()
        print(f"[Node] GPU显存池管理器已初始化")
      except Exception as e:
        print(f"[Node] GPU显存池管理器初始化失败（不影响其他功能）: {e}")
        self.gpu_pool = None
    else:
      print("[Node] GPU显存池模块不可用，跳过初始化")
    
    # 广播自己的分片配置（如果配置了）- 延迟一点确保peers已经连接
    asyncio.create_task(self._periodic_broadcast_shard_config())
    # 广播已加载的模型
    asyncio.create_task(self._periodic_broadcast_loaded_models())
    # 延迟一点后，请求其他节点的已加载模型信息
    asyncio.create_task(self._request_loaded_models_after_delay(delay=4.0))

    # 启动周期性拓扑收集（优化：使用30秒间隔，推理期间自动跳过）
    asyncio.create_task(self.periodic_topology_collection(30.0))  # 从2秒改为30秒

    # [STAR] 启动 WebSocket 连接到 Manager（用于内网穿透场景）- 使用增强版管理器
    if self.manager_url:
      asyncio.create_task(self._connect_to_manager_websocket_v2(delay=3.0))

  async def _delayed_register_to_manager(self, delay: float = 2.0):
    """延迟注册，确保 gRPC server 完全就绪"""
    await asyncio.sleep(delay)
    await self._register_to_manager()

  async def _register_to_manager(self, max_retries: int = 3, retry_interval: float = 5.0):
    """
    主动注册到 EXO Manager
    
    向 Manager 发送自己的节点信息，包括：
    - node_id
    - address (gRPC 监听地址)
    - port (gRPC 端口)
    - device_capabilities
    """
    if not self.manager_url:
      return
    
    import requests as http_requests
    
    manager_api = f"{self.manager_url.rstrip('/')}/api/nodes"
    
    for attempt in range(1, max_retries + 1):
      try:
        # 获取本节点的监听地址和端口
        my_address = "127.0.0.1"
        my_port = 50051

        if hasattr(self.server, 'host') and self.server.host:
          my_address = self.server.host
        if hasattr(self.server, 'port') and self.server.port:
          my_port = self.server.port

        # 🔑 关键优化：如果使用 FRP P2P 模式，向 Manager 注册 FRP 地址
        # 这样其他节点通过 Manager 发现时能获得正确的 P2P 连接地址
        if hasattr(self.discovery, 'get_my_address_info'):
          frp_addr_info = self.discovery.get_my_address_info()
          if frp_addr_info:
            original_addr = f"{my_address}:{my_port}"
            my_address = frp_addr_info["address"]
            my_port = frp_addr_info["port"]
            logging.info(
              f"[Node] [Manager] Address override for FRP P2P:\n"
              f"   Original: {original_addr}\n"
              f"   FRP P2P:  {my_address}:{my_port}"
            )

        # 如果绑定的是 0.0.0.0，尝试获取本机 IP
        if my_address == "0.0.0.0":
          try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            my_address = s.getsockname()[0]
            s.close()
          except Exception:
            my_address = "127.0.0.1"
        
        # 构建注册请求
        payload = {
          "node_id": self.id,
          "address": my_address,
          "port": my_port,
          "chatgpt_api_port": self.chatgpt_api_port,
          "device_info": self.device_capabilities.to_dict() if hasattr(self.device_capabilities, 'to_dict') else {}
        }
        
        print(f"[Node] [Manager] 正在注册到 Manager ({attempt}/{max_retries})...")
        print(f"[Node] [Manager]   URL: {manager_api}")
        print(f"[Node] [Manager]   Node: {self.id} @ {my_address}:{my_port} (HTTP:{self.chatgpt_api_port})")
        
        response = http_requests.post(manager_api, json=payload, timeout=30)
        
        if response.status_code == 200:
          result = response.json()
          if result.get("success"):
            print(f"[Node] [Manager] [OK] 注册成功! {result.get('message', '')}")
            
            # 启动心跳保活
            asyncio.create_task(self._manager_heartbeat())
            return
          else:
            print(f"[Node] [Manager] [WARN] 注册返回失败: {result.get('message', 'Unknown error')}")
        else:
          print(f"[Node] [Manager] [FAIL] HTTP {response.status_code}: {response.text[:200]}")
          
      except http_requests.exceptions.ConnectionError:
        print(f"[Node] [Manager] [FAIL] 无法连接到 Manager ({self.manager_url})")
      except Exception as e:
        print(f"[Node] [Manager] [FAIL] 注册异常: {e}")
      
      if attempt < max_retries:
        print(f"[Node] [Manager]   {retry_interval}秒后重试...")
        await asyncio.sleep(retry_interval)
    
    print(f"[Node] [Manager] [WARN] 注册失败 ({max_retries}次尝试)，将独立运行")

  async def _manager_heartbeat(self, interval: float = 30.0):
    """
    定期向 Manager 发送心跳，保持注册状态

    [STAR] 双向通信：
    - 上报：loaded_models, GPU 显存等实时状态
    - 拉取：待处理的任务（Pull 模式）

    在 FRP/内网场景下，这是 Node → Manager 的唯一通信通道
    """
    import requests as http_requests
    health_url = f"{self.manager_url.rstrip('/')}/api/nodes/{self.id}/health-check"
    
    while True:
      try:
        await asyncio.sleep(interval)
        
        # [OK] 构建心跳 payload（携带节点状态 + 网络地址）
        # 注意: 地址信息用于 Manager 自动注册时构建正确的连接 URL
        heartbeat_payload = {
          "timestamp": time.time(),
          "node_id": self.id,
          "address": my_address,
          "port": my_port,
          "chatgpt_api_port": self.chatgpt_api_port,
        }
        
        # 添加已加载模型信息
        if self.my_loaded_models:
          heartbeat_payload["loaded_models"] = [
            {
              "model_id": model_id,
              "shard": load_state.shard.to_dict() if hasattr(load_state.shard, 'to_dict') else {}
            }
            for model_id, load_state in self.my_loaded_models.items()
          ]
        
        # 添加 GPU 显存信息
        gpu_memory = self._get_realtime_gpu_memory()
        if gpu_memory:
          heartbeat_payload["gpu_memory"] = gpu_memory
        
        # POST 时带上状态数据
        response = http_requests.post(health_url, json=heartbeat_payload, timeout=5)
        if response.status_code == 200:
          result = response.json()
          logging.debug(f"[Node] [Manager] 心跳成功")
          
          # [STAR] 检查是否有待处理的任务
          pending_tasks = result.get("data", {}).get("pending_tasks", [])
          if pending_tasks:
            print(f"[Node] [Manager] [RECV] 收到 {len(pending_tasks)} 个待处理任务（Pull 模式）")
            
            for task in pending_tasks:
              task_type = task.get("type", "shard_config")
              task_id = task.get("task_id", "unknown")
              
              try:
                if task_type == "shard_config" or "shard" in task:
                  model_id = task.get("model_id")
                  model_path = task.get("model_path")
                  shard = task.get("shard", {})
                  peer_list = task.get("peer_list", [])
                  
                  print(f"[Node] [Manager] [EXEC] 执行分片配置任务: {task_id}")
                  print(f"   模型: {model_id}, 层: {shard.get('start_layer')}-{shard.get('end_layer')}")
                  
                  # 注册 peer 列表
                  if peer_list:
                    asyncio.create_task(self._register_peers_from_manager(peer_list))
                  
                  # 延迟加载模型
                  async def _delayed_load():
                    await asyncio.sleep(1.0)
                    await self._handle_manager_load(
                      model_id, model_path,
                      shard.get("start_layer", 0),
                      shard.get("end_layer", 0),
                      shard.get("n_layers", 0)
                    )
                  asyncio.create_task(_delayed_load())
                  
                  print(f"[OK] [Node] [Manager] 任务 {task_id} 已启动执行")
                
                else:
                  print(f"[WARN] [Node] [Manager] 未知任务类型: {task_type}")
                  
              except Exception as e:
                print(f"[FAIL] [Node] [Manager] 执行任务失败 ({task_id}): {e}")
                if DEBUG >= 1: traceback.print_exc()
          
        else:
          logging.warning(f"[Node] [Manager] 心跳失败: HTTP {response.status_code}")
      except Exception as e:
        logging.debug(f"[Node] [Manager] 心跳异常: {e}")

  async def _connect_to_manager_websocket(self, delay: float = 3.0):
    """
    通过 WebSocket 连接到 Manager（用于内网穿透）
    
    建立持久连接后：
    1. 发送注册消息
    2. 接收推理请求
    3. 调用本地 ChatGPT API 处理
    4. 流式返回结果
    """
    if not self.manager_url:
      return
    
    await asyncio.sleep(delay)
    
    import websockets
    
    # 构建 WebSocket URL (http -> ws, https -> wss)
    base_url = self.manager_url.rstrip('/')
    if base_url.startswith('https://'):
      ws_url = base_url.replace('https://', 'wss://')
    else:
      ws_url = base_url.replace('http://', 'ws://')
    
    ws_endpoint = f"{ws_url}/ws/node/{self.id}"
    
    retry_interval = 5.0
    attempt = 0
    
    # [STAR] 无限重连模式（确保 WS 始终可用）
    while True:
      attempt += 1
      try:
        print(f"[NodeWS] [CONNECT] 正在连接到 Manager WebSocket (第{attempt}次)...")
        print(f"[NodeWS]   URL: {ws_endpoint}")
        
        # [STAR] 优化：增加心跳超时时间，避免推理过程中断连
        async with websockets.connect(
          ws_endpoint,
          ping_interval=60,              # 心跳间隔（从30s增加到60s）
          ping_timeout=120,             # 心跳超时（从60s增加到120s，容忍GPU计算延迟）
          close_timeout=10,
          max_size=10 * 1024 * 1024
        ) as websocket:
          
          print(f"[NodeWS] [OK] 已连接到 Manager!")
          
          # 发送注册消息
          register_msg = {
            "type": "register",
            "node_id": self.id,
            "chatgpt_api_port": self.chatgpt_api_port
          }
          await websocket.send(json.dumps(register_msg))
          print(f"[NodeWS] [SEND] 已发送注册消息")
          
          # 等待注册确认
          response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
          ack_data = json.loads(response)
          
          if ack_data.get("type") == "register_ack" and ack_data.get("status") == "success":
            print(f"[NodeWS] [OK] 注册成功: {ack_data.get('message')}")
          else:
            print(f"[NodeWS] [WARN] 注册响应异常: {ack_data}")
            continue
          
          # 保持连接，处理 Manager 发来的请求
          print(f"[NodeWS] [TARGET] 进入消息循环，等待推理请求...")
          
          while True:
            try:
              message = await asyncio.wait_for(websocket.recv(), timeout=None)
              data = json.loads(message)
              
              msg_type = data.get("type")
              
              if msg_type == "inference_request":
                # 收到推理请求，异步处理（不阻塞消息循环）
                request_id = data.get("request_id")
                print(f"[NodeWS] [RECV] 收到推理请求: {request_id}")
                
                asyncio.create_task(
                  self._handle_ws_inference_request(websocket, data)
                )
                
              elif msg_type == "model_load":
                # [STAR] Received model load request
                task_id = data.get("task_id", "unknown")
                model_id = data.get("model_id", "")
                print(f"[NodeWS] [LOAD] 收到模型加载任务: {task_id}, model={model_id}")
                
                asyncio.create_task(
                  self._handle_ws_model_load(websocket, data)
                )
                
              elif msg_type == "model_unload":
                # [STAR] 收到模型卸载请求
                unload_model_id = data.get("model_id", "")
                print(f"[NodeWS] [DELETE] 收到模型卸载请求: {unload_model_id}")
                
                asyncio.create_task(
                  self._handle_ws_model_unload(websocket, unload_model_id)
                )
                
              elif msg_type == "heartbeat":
                # 心跳检测
                await websocket.send(json.dumps({
                  "type": "pong",
                  "timestamp": time.time(),
                  "node_id": self.id
                }))
                
              else:
                print(f"[NodeWS] [WARN] 未知消息类型: {msg_type}")
                
            except websockets.exceptions.ConnectionClosed:
              print(f"[NodeWS] 🔌 WebSocket 连接已关闭")
              break
            except Exception as e:
              print(f"[NodeWS] [FAIL] 消息处理错误: {e}")
              if DEBUG >= 1: traceback.print_exc()
              break
      
      except Exception as e:
        print(f"[NodeWS] [FAIL] 连接失败 (第{attempt}次): {e}")
        print(f"[NodeWS]   {retry_interval:.1f}秒后重试...")
        await asyncio.sleep(retry_interval)
        retry_interval = min(retry_interval * 1.5, 60.0)  # 指数退避，最大60秒
    
    # 不会执行到这里（无限循环）
    # print(f"[NodeWS] [WARN] WebSocket 连接失败 ({max_retries}次尝试)，将仅使用 HTTP 心跳模式")

  async def _connect_to_manager_websocket_v2(self, delay: float = 3.0):
    """
    [STAR] 增强版 WebSocket 连接（使用优化管理器）
    
    新特性：
    - 消息队列缓冲（防止丢消息）
    - 背压控制（防止连接过载）
    - 断线自动重传（保证可靠性）
    - 智能心跳保活（更健康）
    - 连接状态监控（可观测性）
    - 优先级消息（QoS保障）
    
    使用方式：
      asyncio.create_task(self._connect_to_manager_websocket_v2())
    """
    if not self.manager_url:
      return
    
    await asyncio.sleep(delay)
    
    try:
      from exo.networking.websocket_optimized import create_ws_manager
      
      print(f"[NodeWS-V2] [FAST] 启动增强版 WebSocket 管理器...")
      
      # 创建增强版管理器
      # [STAR] 优化：增加心跳间隔和超时时间，避免推理过程中因GPU计算阻塞导致断连
      # 原因：大模型推理时GPU计算会阻塞事件循环，无法及时响应ping/pong
      self.ws_manager_v2: EnhancedWebSocketManager = await create_ws_manager(
        node_id=self.id,
        manager_url=self.manager_url,
        max_queue_size=2000,           # 队列容量
        ping_interval=60.0,            # 心跳间隔（从20s增加到60s，减少频率）
        ping_timeout=120.0,            # 心跳超时（从40s增加到120s，容忍GPU计算延迟）
        reconnect_base_delay=2.0,      # 基础重连延迟（从1s增加到2s）
        reconnect_max_delay=60.0,      # 最大重连延迟
        max_retries=5,                 # 最大重试次数（从3次增加到5次）
        enable_compression=True,       # 启用压缩
        stats_callback=self._on_ws_stats_update  # 统计回调
      )
      
      # 设置事件回调
      self.ws_manager_v2.on_connect = self._on_ws_connected
      self.ws_manager_v2.on_disconnect = self._on_ws_disconnected
      self.ws_manager_v2.on_message = self._on_ws_message_received
      self.ws_manager_v2.on_error = self._on_ws_error
      
      # 启动管理器
      await self.ws_manager_v2.start()
      
      print(f"[OK] [NodeWS-V2] 增强版 WebSocket 已启动!")
      
      # 启动消息处理循环
      asyncio.create_task(self._ws_v2_message_loop())
      
    except ImportError as e:
      print(f"[WARN] [NodeWS-V2] 无法导入优化模块，回退到旧版本: {e}")
      await self._connect_to_manager_websocket(delay=0)
    except Exception as e:
      print(f"[FAIL] [NodeWS-V2] 启动失败: {e}")
      if DEBUG >= 1: traceback.print_exc()
      await self._connect_to_manager_websocket(delay=0)

  async def _ws_v2_message_loop(self):
    """
    V2 版本的消息处理循环
    
    从增强版管理器的接收队列中取出消息并处理
    """
    if not hasattr(self, 'ws_manager_v2'):
      return
    
    try:
      async for message in self.ws_manager_v2.receive_stream():
        if not self.ws_manager_v2.is_running:
          break

        msg_type = message.msg_type

        # 🔍 [诊断] 记录每条收到的消息类型（排查消息丢失）
        print(f"🔍 [NodeWS-V2-DISPATCH] 收到消息: type={msg_type}, msg_keys={list(message.payload.keys())[:8]}")
        
        if msg_type == "inference_request":
          # 推理请求 - 异步处理
          request_id = message.payload.get("request_id", "unknown")
          model_for_task = message.payload.get("model_id", "unknown")
          print(f"[NodeWS-V2] [RECV] 收到推理请求: {request_id}")
          
          asyncio.create_task(
            self._handle_ws_inference_request_v2(message.payload),
            name=f"ws-inference-{request_id[:8]}-{model_for_task.split('::')[0][:10]}"
          )
          
        elif msg_type == "model_load":
          # 模型加载请求
          task_id = message.payload.get("task_id", "unknown")
          model_id = message.payload.get("model_id", "")
          print(f"[NodeWS-V2] [LOAD] 收到模型加载任务: {task_id}, model={model_id}")
          
          asyncio.create_task(
            self._handle_ws_model_load_v2(message.payload)
          )
          
        elif msg_type == "model_unload":
          # 模型卸载请求
          unload_model_id = message.payload.get("model_id", "")
          print(f"[NodeWS-V2] [DELETE] 收到模型卸载请求: {unload_model_id}")
          
          asyncio.create_task(
            self._handle_ws_model_unload(unload_model_id)
          )
          
        elif msg_type == "heartbeat":
          # 心跳响应已由管理器自动处理
          pass
          
        elif msg_type in ["register_ack", "node_registered", "registration_success"]:
          # 注册确认消息 - 记录日志即可
          print(f"[NodeWS-V2] [OK] 收到注册确认: {msg_type}")
          
        elif msg_type in ["broadcast", "system_message", "notification"]:
          # 广播/系统消息
          content = message.payload.get("content", message.payload.get("message", ""))
          sender = message.payload.get("sender", "manager")
          print(f"[NodeWS-V2] [BROADCAST] 收到广播消息 (来自 {sender}): {content[:100]}")
          
        elif msg_type in ["health_check", "ping", "heartbeat_request"]:
          print(f"[NodeWS-V2] [HEART] 收到健康检查: {msg_type}")
          if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected:
            try:
              response = {
                "type": "health_check_response",
                "status": "healthy",
                "node_id": self.id,
                "timestamp": time.time()
              }
              await self.ws_manager_v2.send(response, priority=MessagePriority.NORMAL)
              print(f"[NodeWS-V2] [OK] 已响应健康检查")
            except Exception as e:
              print(f"[NodeWS-V2] [WARN] 响应健康检查失败: {e}")

        elif msg_type == "peer_list":
          # 📢 Manager 推送的在线节点列表（用于 FRP P2P 发现）
          peers = message.payload.get("peers", [])
          if peers:
            print(f"[NodeWS-V2] [PEER] 收到 Manager 推送的 {len(peers)} 个在线节点")
            await self._register_peers_from_manager(peers)
          else:
            print(f"[NodeWS-V2] [PEER] Manager 推送的节点列表为空")

        elif msg_type == "grpc_relay":
          # 管理平台通过 WebSocket 转发的 gRPC 请求（当直连 gRPC 不可达时）
          method = message.payload.get("method", "")
          source_node_id = message.payload.get("source_node_id", "")
          payload = message.payload.get("payload", {})
          print(f"[NodeWS-V2] [GRPC-RELAY] 收到转发请求: method={method}, from={source_node_id}")

          try:
            if method == "HealthCheck":
              response = {
                "type": "grpc_relay_response",
                "method": "HealthCheck",
                "source_node_id": source_node_id,
                "success": True,
                "payload": {"is_healthy": True, "node_id": self.id}
              }
              await self.ws_manager_v2.send(response, priority=MessagePriority.NORMAL)
              print(f"[NodeWS-V2] [GRPC-RELAY] HealthCheck 已响应")

            elif method == "CollectTopology":
              topology_data = await self._build_topology_response(payload)
              response = {
                "type": "grpc_relay_response",
                "method": "CollectTopology",
                "source_node_id": source_node_id,
                "success": True,
                "payload": topology_data
              }
              await self.ws_manager_v2.send(response, priority=MessagePriority.NORMAL)
              print(f"[NodeWS-V2] [GRPC-RELAY] CollectTopology 已响应")

            else:
              print(f"[NodeWS-V2] [WARN] 未知的 gRPC relay 方法: {method}")

          except Exception as e:
            print(f"[NodeWS-V2] [ERROR] grpc_relay 处理失败: {e}")
            error_resp = {
              "type": "grpc_relay_response",
              "method": method,
              "source_node_id": source_node_id,
              "success": False,
              "error": str(e)
            }
            try:
              await self.ws_manager_v2.send(error_resp, priority=MessagePriority.NORMAL)
            except:
              pass

        else:
          print(f"[NodeWS-V2] [WARN] 未知消息类型: {msg_type}")
          
    except asyncio.CancelledError:
      print(f"[NodeWS-V2] [STOP] 消息循环已停止")
    except Exception as e:
      print(f"[NodeWS-V2] [FAIL] 消息循环错误: {e}")
      if DEBUG >= 1: traceback.print_exc()

  async def _handle_ws_inference_request_v2(self, request_data: Dict):
    """
    V2 版本：处理推理请求（使用增强功能）
    
    改进：
    - 自动流式返回结果
    - 更好的错误处理
    - 进度报告
    - 背压控制
    """
    import aiohttp
    
    request_id = request_data.get("request_id", "unknown")
    model_id = request_data.get("model_id", "")
    messages = request_data.get("messages", [])
    stream = request_data.get("stream", True)
    
    print(f"[NodeWS-V2] [EXEC] 开始处理推理 (V2): {request_id}, model={model_id}")

    start_time = time.time()
    total_tokens_used = 0
    chunk_count = 0
    
    try:
      # 构建本地 ChatGPT API 请求 URL
      local_api_url = f"http://127.0.0.1:{self.chatgpt_api_port}/v1/chat/completions"
      
      # 转换消息格式 - 支持多模态（文本+图片）
      # OpenAI 格式: content 可以是 string 或 [{"type": "image_url", ...}, {"type": "text", ...}]
      api_messages = []
      has_image = False
      for msg in messages:
        if isinstance(msg, dict):
          role = msg.get("role", "user")
          content = msg.get("content", "")
          
          if isinstance(content, list):
            has_image = True
          
          api_messages.append({
            "role": role,
            "content": content
          })
      
      if has_image:
        print(f"[NodeWS-V2] [IMAGE] 检测到图片内容，使用多模态模式")
      
      max_tokens_default = 2048 if has_image else 512
      
      payload = {
        "model": model_id,
        "messages": api_messages,
        "temperature": request_data.get("temperature", 0.7),
        "max_tokens": request_data.get("max_tokens", max_tokens_default),
        "stream": stream
      }
      
      # 发送开始通知
      if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
        await self.ws_manager_v2.send({
          "type": "inference_start",
          "request_id": request_id,
          "timestamp": time.time()
        }, priority=MessagePriority.HIGH)
      
      async with aiohttp.ClientSession() as session:
        async with session.post(
          local_api_url,
          json=payload,
          headers={"Content-Type": "application/json"},
          timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
          
          if response.status != 200:
            error_text = await response.text()
            print(f"[NodeWS-V2] [FAIL] 本地 API 错误 {response.status}: {error_text[:200]}")
            
            error_msg = {
              "type": "inference_error",
              "request_id": request_id,
              "error": f"Local API error ({response.status}): {error_text[:200]}"
            }
            
            if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
              await self.ws_manager_v2.send(error_msg, priority=MessagePriority.CRITICAL, require_ack=True)
            return
          
          # 流式读取本地 API 的响应
          chunk_count = 0
          async for line in response.content:
            line_text = line.decode('utf-8').strip()
            
            if not line_text:
              continue
            
            # 发送数据块给 Manager
            chunk_msg = {
              "type": "inference_chunk",
              "request_id": request_id,
              "data": line_text,
              "chunk_index": chunk_count
            }
            
            if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
              await self.ws_manager_v2.send(chunk_msg, priority=MessagePriority.HIGH)
            
            chunk_count += 1
            
            # 统计 token 使用量
            if line_text.startswith('data: ') and '[DONE]' not in line_text:
              try:
                data = json.loads(line_text[6:])
                choices = data.get('choices', [])
                if choices:
                  delta = choices[0].get('delta', {}).get('content', '')
                  if delta:
                    total_tokens_used += len(delta) // 4
              except:
                pass
      
      # 计算耗时
      elapsed_time = time.time() - start_time
      
      # 发送完成消息
      complete_msg = {
        "type": "inference_complete",
        "request_id": request_id,
        "tokens_used": total_tokens_used,
        "elapsed_time": elapsed_time,
        "chunks_sent": chunk_count
      }
      
      if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
        await self.ws_manager_v2.send(complete_msg, priority=MessagePriority.HIGH, require_ack=True)
      
      print(f"[NodeWS-V2] [OK] 推理完成: {request_id}, tokens≈{total_tokens_used}, 耗时={elapsed_time:.2f}s")
      
    except Exception as e:
      print(f"[NodeWS-V2] [FAIL] 推理处理失败 {request_id}: {e}")
      if DEBUG >= 1: traceback.print_exc()
      
      error_msg = {
        "type": "inference_error",
        "request_id": request_id,
        "error": str(e)
      }
      
      if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
        try:
          await self.ws_manager_v2.send(error_msg, priority=MessagePriority.CRITICAL, require_ack=True)
        except:
          pass

  async def _handle_ws_model_load_v2(self, task_data: Dict):
    """
    V2 版本：处理模型加载请求（使用增强功能）
    """
    task_id = task_data.get("task_id", "unknown")
    model_id = task_data.get("model_id", "")
    model_path = task_data.get("model_path", "")  # 获取模型路径
    shard = task_data.get("shard", {})            # 获取分片配置
    peer_list = task_data.get("peer_list", [])   # 获取 Peer 列表
    instance_id = task_data.get("instance_id", None)  # [OK] 获取实例ID
    
    print(f"[NodeWS-V2] [LOAD-START] 开始模型加载 (V2): {task_id}, model={model_id}, path={model_path}, instance={instance_id}")
    if instance_id is None:
      print(f"[NodeWS-V2] [ROUTE-CHECK] 收到的完整数据 keys: {list(task_data.keys())}")
    
    try:
      # 发送开始消息
      if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
        await self.ws_manager_v2.send({
          "type": "model_load_start",
          "task_id": task_id,
          "model_id": model_id,
          "node_id": self.id
        }, priority=MessagePriority.HIGH)
      
      # 执行模型加载逻辑（使用 Manager 指定的分片配置）
      if model_path:
        # 🔧 关键修复：使用 Manager 分发的具体分片信息，而不是重新分配
        if shard and "start_layer" in shard and "end_layer" in shard:
          # Manager 已指定分片 → 直接加载该节点的专属层
          print(f"[NodeWS-V2] [SHARD] 使用 Manager 指定分片: 层 {shard['start_layer']}-{shard['end_layer']}")

          # 🔍 重复加载检测：检查是否已存在相同的模型+实例
          base_model_id = model_id.split("::")[0] if "::" in model_id else model_id
          check_key = f"{base_model_id}::{instance_id}" if instance_id else base_model_id

          if hasattr(self, 'gpu_pool') and self.gpu_pool and hasattr(self.gpu_pool.manager, 'pool_models'):
            existing_models = self.gpu_pool.manager.pool_models
            if check_key in existing_models:
              existing_state = existing_models[check_key].state
              if existing_state in ['loaded', 'partial', 'loading']:
                print(f"[NodeWS-V2] [DUPLICATE] 模型 {check_key} 已存在 (状态: {existing_state})，跳过重复加载")
                result = {"success": True, "message": "模型已加载", "duplicate": True}
              else:
                print(f"[NodeWS-V2] [RELOAD] 模型 {check_key} 状态为 {existing_state}，重新加载...")
                # 继续下面的加载流程
                result = None
            else:
              result = None  # 不存在，需要加载
          else:
            result = None  # 无法检查，继续加载

          # 只有在需要加载时才执行
          if result is None:
            from exo.inference.shard import Shard as ShardType

            custom_shards = {
              self.id: ShardType(
                model_id=base_model_id,
                start_layer=int(shard["start_layer"]),
                end_layer=int(shard["end_layer"]),
                n_layers=int(shard.get("n_layers", 32)),
                repo_id="",
                tie_word_embeddings=True,
                instance_id=instance_id
              )
            }

            result = await self.gpu_pool.load(
              model_id=model_id,
              model_path=model_path,
              n_layers=int(shard.get("n_layers", 32)),
              custom_shards=custom_shards,
              strategy="custom",
              instance_id=instance_id
            )
        else:
          # 无分片信息 → 回退到自动分配模式
          print(f"[NodeWS-V2] [AUTO] 未收到分片信息，使用自动分配模式")
          result = await self.pool_load_model(
            model_id=model_id,
            model_path=model_path,
            nodes=peer_list if peer_list else None,
            n_layers=shard.get("n_layers") if shard else None,
            instance_id=instance_id
          )
      else:
        options = task_data.get("options", {})
        print(f"[NodeWS-V2] [WARN] 未提供 model_path，使用 options: {list(options.keys())}")
        
        if "model_path" in options:
          result = await self.pool_load_model(
            model_id=model_id,
            model_path=options["model_path"],
            instance_id=options.get("instance_id") or instance_id,  # [OK] 传递实例ID
            **{k: v for k, v in options.items() if k not in ["model_path", "instance_id"]}
          )
        else:
          raise ValueError("缺少必需参数: model_path")
      
      # 发送完成消息
      if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
        # 构建 loaded_models 列表（与 V1 格式一致）
        loaded = []
        if result and isinstance(result, dict) and result.get("success"):
          base_model_id = model_id.split("::")[0] if "::" in model_id else model_id
          loaded.append({
            "model_id": model_id,
            "shard": {
              "start_layer": shard.get("start_layer") if isinstance(shard, dict) else None,
              "end_layer": shard.get("end_layer") if isinstance(shard, dict) else None,
              "n_layers": shard.get("n_layers") if isinstance(shard, dict) else None,
              "instance_id": instance_id
            }
          })

        await self.ws_manager_v2.send({
          "type": "model_load_complete",
          "task_id": task_id,
          "node_id": self.id,
          "success": True,
          "loaded_models": loaded,
          "result": result
        }, priority=MessagePriority.HIGH, require_ack=True)
      
      print(f"[NodeWS-V2] [OK] 模型加载完成: {task_id}")
      
    except Exception as e:
      print(f"[NodeWS-V2] [FAIL] 模型加载失败: {task_id}: {e}")
      
      if hasattr(self, 'ws_manager_v2') and self.ws_manager_v2.is_connected and MessagePriority:
        await self.ws_manager_v2.send({
          "type": "model_load_error",
          "task_id": task_id,
          "error": str(e)
        }, priority=MessagePriority.CRITICAL, require_ack=True)

  # ========== V2 回调方法 ==========
  
  async def _on_ws_connected(self):
    """WebSocket 连接成功回调"""
    print(f"[NodeWS-V2] [LINK] 已连接到 Manager!")
    
    # 可以在这里执行额外的初始化操作
    # 例如：同步状态、发送初始数据等

  async def _on_ws_disconnected(self):
    """WebSocket 断开连接回调"""
    print(f"[NodeWS-V2] 🔌 与 Manager 的连接已断开")
    
    # 管理器会自动尝试重连，这里可以做一些清理工作

  async def _on_ws_message_received(self, message):
    """收到消息的通用回调（可选）"""
    pass  # 主要在 _ws_v2_message_loop 中处理

  async def _on_ws_error(self, error: Exception):
    """错误回调"""
    print(f"[NodeWS-V2] [FAIL] 错误: {error}")

  async def _on_ws_stats_update(self, stats):
    """
    统计信息更新回调（用于监控）
    
    Args:
      stats: ConnectionStats 对象
    """
    # 可以定期打印统计信息或发送到监控系统
    if DEBUG >= 3:
      print(f"[NodeWS-V2] [STATS] 统计: sent={stats.messages_sent}, "
            f"recv={stats.messages_received}, "
            f"latency={stats.avg_latency*1000:.1f}ms")

  async def get_ws_status(self) -> Optional[Dict]:
    """
    获取 WebSocket 连接状态（用于 API 和监控）
    
    Returns:
      Dict or None: 连接状态信息
    """
    if not hasattr(self, 'ws_manager_v2'):
      return None
    
    if not self.ws_manager_v2:
      return None
    
    base_status = self.ws_manager_v2.status
    
    # 添加健康状态
    health = self.ws_manager_v2.get_health_status()
    
    return {
      **base_status,
      "health": health
    }

  async def _handle_ws_inference_request(self, websocket, request_data: Dict):
    """
    处理通过 WebSocket 收到的推理请求
    
    调用本地 ChatGPT API 执行推理，并流式返回结果
    """
    import aiohttp
    
    request_id = request_data.get("request_id", "unknown")
    model_id = request_data.get("model_id", "")
    messages = request_data.get("messages", [])
    stream = request_data.get("stream", True)
    
    print(f"[NodeWS] [EXEC] 开始处理推理: {request_id}, model={model_id}")
    
    heartbeat_task = None
    
    try:
      # 构建本地 ChatGPT API 请求 URL
      local_api_url = f"http://127.0.0.1:{self.chatgpt_api_port}/v1/chat/completions"
      
      # 转换消息格式
      api_messages = []
      for msg in messages:
        if isinstance(msg, dict):
          api_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
          })
      
      payload = {
        "model": model_id,
        "messages": api_messages,
        "temperature": request_data.get("temperature", 0.7),
        "max_tokens": request_data.get("max_tokens", 512),
        "stream": stream
      }
      
      total_tokens_used = 0
      
      async with aiohttp.ClientSession() as session:
        async with session.post(
          local_api_url,
          json=payload,
          headers={"Content-Type": "application/json"},
          timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
          
          if response.status != 200:
            error_text = await response.text()
            print(f"[NodeWS] [FAIL] 本地 API 错误 {response.status}: {error_text[:200]}")
            
            error_msg = {
              "type": "inference_error",
              "request_id": request_id,
              "error": f"Local API error ({response.status}): {error_text[:200]}"
            }
            await websocket.send(json.dumps(error_msg))
            return
          
          # 流式读取本地 API 的响应
          async for line in response.content:
            line_text = line.decode('utf-8').strip()
            
            if not line_text:
              continue
            
            # 直接转发 SSE 数据块给 Manager
            chunk_msg = {
              "type": "inference_chunk",
              "request_id": request_id,
              "data": line_text
            }
            await websocket.send(json.dumps(chunk_msg))
            
            # 统计 token 使用量（简单估算）
            if line_text.startswith('data: ') and '[DONE]' not in line_text:
              try:
                data = json.loads(line_text[6:])
                choices = data.get('choices', [])
                if choices:
                  delta = choices[0].get('delta', {}).get('content', '')
                  if delta:
                    total_tokens_used += len(delta) // 4
              except:
                pass
      
      # 发送完成消息
      complete_msg = {
        "type": "inference_complete",
        "request_id": request_id,
        "tokens_used": total_tokens_used
      }
      await websocket.send(json.dumps(complete_msg))
      
      print(f"[NodeWS] [OK] 推理完成: {request_id}, tokens≈{total_tokens_used}")
      
    except Exception as e:
      print(f"[NodeWS] [FAIL] 推理处理失败 {request_id}: {e}")
      if DEBUG >= 1: traceback.print_exc()
      
      error_msg = {
        "type": "inference_error",
        "request_id": request_id,
        "error": str(e)
      }
      try:
        await websocket.send(json.dumps(error_msg))
      except:
        pass
    
    finally:
      # [STAR] 取消心跳保活任务
      if heartbeat_task:
        heartbeat_task.cancel()
        try:
          await heartbeat_task
        except asyncio.CancelledError:
          pass

  async def _handle_ws_model_load(self, websocket, task_data: Dict):
    """
    处理通过 WebSocket 收到的模型加载请求

    执行模型下载和加载，并通过 WebSocket 返回结果
    [STAR] 新增：模型加载期间发送心跳保活，防止 WebSocket 断开
    [STAR] 新增：Node 自动根据 model_id 解析 model_path（不依赖 Server）
    """
    from exo.inference.shard import Shard
    from exo.models import DEFAULT_MODEL_CARDS
    
    task_id = task_data.get("task_id", "unknown")
    model_id = task_data.get("model_id", "")
    model_path = task_data.get("model_path", "")
    shard_info = task_data.get("shard", {})
    peer_list = task_data.get("peer_list", [])

    # [STAR] 多实例支持：从任务数据中提取 instance_id
    instance_id = shard_info.get("instance_id", task_data.get("instance_id", "default"))

    print(f"[NodeWS] [EXEC] 开始处理模型加载: {task_id}, model={model_id} (实例: {instance_id})")
    
    # [STAR] 如果 model_path 为空，Node 自己根据 model_id 查找路径
    if not model_path and model_id:
      print(f"[NodeWS] [ROUTE-CHECK] model_path 为空，尝试从本地配置解析...")
      
      # 从 DEFAULT_MODEL_CARDS 中查找
      if model_id in DEFAULT_MODEL_CARDS:
        config = DEFAULT_MODEL_CARDS[model_id]
        repo_info = config.get("repo", {})
        
        if isinstance(repo_info, dict):
          engine_name = self.inference_engine.__class__.__name__
          if engine_name in repo_info:
            model_path = repo_info[engine_name]
            print(f"[NodeWS] [OK] 从配置中找到路径 (引擎 {engine_name}): {model_path}")
          else:
            for eng_name, repo in repo_info.items():
              if 'PyTorch' in eng_name or 'Dummy' in eng_name:
                model_path = repo
                print(f"[NodeWS] [OK] 回退到引擎 {eng_name}: {model_path}")
                break
        
        if not model_path:
          print(f"[NodeWS] [WARN] 配置中未找到 {model_id} 的路径")
      else:
        print(f"[NodeWS] [WARN] {model_id} 不在 DEFAULT_MODEL_CARDS 中")
    
    # [STAR] Start heartbeat task (prevent WS disconnection during long model loading)
    heartbeat_task = None
    try:
      async def _keepalive():
        """模型加载期间的心跳保活"""
        while True:
          await asyncio.sleep(15)  # 每 15 秒发一次心跳
          try:
            await websocket.send(json.dumps({
              "type": "model_loading_progress",
              "task_id": task_id,
              "node_id": self.id,
              "status": "loading",
              "timestamp": time.time()
            }))
          except Exception:
            break
      
      heartbeat_task = asyncio.create_task(_keepalive())
      
      # 1. 注册 peer 列表（如果提供）
      if peer_list:
        print(f"[NodeWS] [INFO] 注册 {len(peer_list)} 个 peers")
        await self._register_peers_from_manager(peer_list)
      
      # 2. 构建分片对象
      start_layer = shard_info.get("start_layer", 0)
      end_layer = shard_info.get("end_layer", 0)
      n_layers = shard_info.get("n_layers", 32)

      shard = Shard(
        model_id=model_id,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        instance_id=instance_id
      )

      # 3. [STAR] 使用对应实例的引擎加载模型（支持多实例）
      target_engine = self.get_engine(instance_id)
      print(f"[NodeWS] [INIT] 正在加载模型分片 (层 {start_layer}-{end_layer}, 实例: {instance_id})...")
      try:
        await target_engine.ensure_shard(shard)
        print(f"[NodeWS] [OK] 模型加载成功！(实例: {instance_id})")
      except Exception as e:
        print(f"[NodeWS] [FAIL] 模型加载失败 {task_id}: {e}")
        raise
      
      # 4. 注意：不需要手动更新 my_loaded_models 和 node_shards
      #    因为 on_model_loaded 回调已被 ensure_shard 自动触发，已完成这些操作
      
      # 5. 发送成功响应
      success_msg = {
        "type": "model_load_complete",
        "task_id": task_id,
        "node_id": self.id,
        "success": True,
        "loaded_models": [
          {
            "model_id": model_id,
            "shard": shard.to_dict() if hasattr(shard, 'to_dict') else {}
          }
        ],
        "message": f"Model {model_id} loaded successfully (layers {start_layer}-{end_layer})"
      }
      await websocket.send(json.dumps(success_msg))
      
      print(f"[NodeWS] [OK] 模型加载完成: {task_id}, model={model_id}")
      
      # 6. 同时上报状态更新
      status_msg = {
        "type": "model_status_update",
        "node_id": self.id,
        "loaded_models": [
          {
            "model_id": mid,
            "shard": ls.shard.to_dict() if hasattr(ls, 'shard') and hasattr(ls.shard, 'to_dict') else {}
          }
          for mid, ls in self.my_loaded_models.items()
        ],
        "gpu_memory": self._get_realtime_gpu_memory()
      }
      await websocket.send(json.dumps(status_msg))
      
    except Exception as e:
      print(f"[NodeWS] [FAIL] 模型加载失败 {task_id}: {e}")
      if DEBUG >= 1: traceback.print_exc()
      
      error_msg = {
        "type": "model_load_complete",
        "task_id": task_id,
        "node_id": self.id,
        "success": False,
        "error": str(e),
        "loaded_models": []
      }
      try:
        await websocket.send(json.dumps(error_msg))
      except:
        pass
    
    finally:
      # [STAR] 取消心跳保活任务
      if heartbeat_task:
        heartbeat_task.cancel()
        try:
          await heartbeat_task
        except asyncio.CancelledError:
          pass

  async def _handle_ws_model_unload(self, websocket, model_id: str):
    """
    处理通过 WebSocket 收到的模型卸载请求
    """
    print(f"[NodeWS] [DELETE] 开始卸载模型: {model_id}")
    
    try:
      # 调用推理引擎卸载模型
      if hasattr(self.inference_engine, 'unload_model'):
        success = await self.inference_engine.unload_model(model_id)
        
        if success:
          # 更新本地状态
          if model_id in self.my_loaded_models:
            del self.my_loaded_models[model_id]
          
          if model_id in self.node_shards:
            del self.node_shards[model_id]
          
          complete_msg = {
            "type": "model_unload_complete",
            "node_id": self.id,
            "model_id": model_id,
            "success": True,
            "message": f"Model {model_id} unloaded successfully"
          }
          await websocket.send(json.dumps(complete_msg))
          
          print(f"[NodeWS] [OK] 模型卸载完成: {model_id}")
          
          # 上报状态更新
          status_msg = {
            "type": "model_status_update",
            "node_id": self.id,
            "loaded_models": [
              {
                "model_id": mid,
                "shard": ls.shard.to_dict() if hasattr(ls, 'shard') and hasattr(ls.shard, 'to_dict') else {}
              }
              for mid, ls in self.my_loaded_models.items()
            ]
          }
          await websocket.send(json.dumps(status_msg))
        else:
          raise Exception("unload_model returned False")
      else:
        raise Exception("Inference engine does not support unload_model")
        
    except Exception as e:
      print(f"[NodeWS] [FAIL] 模型卸载失败 ({model_id}): {e}")
      
      error_msg = {
        "type": "model_unload_complete",
        "node_id": self.id,
        "model_id": model_id,
        "success": False,
        "error": str(e)
      }
      try:
        await websocket.send(json.dumps(error_msg))
      except:
        pass

  async def _request_loaded_models_after_delay(self, delay: float = 4.0):
    """延迟后请求其他节点的已加载模型信息"""
    await asyncio.sleep(delay)
    await self._request_loaded_models_from_peers()

  async def _periodic_broadcast_shard_config(self, initial_delay: float = 2.0, interval: float = 5.0):
    """定期广播分片配置，确保所有节点都能收到"""
    await asyncio.sleep(initial_delay)
    await self._broadcast_shard_config()
    
    # 如果 discovery 不支持分片配置，不需要定期重播
    if not (hasattr(self.discovery, 'get_current_node_shards') or hasattr(self.discovery, 'get_current_node_shard')):
      return
    
    # 定期重播，确保新连接的节点也能收到
    while True:
      await asyncio.sleep(interval)
      if self.node_shards.get(self.id) and len(self.peers) > 0:
        # 检查是否所有peers都有我们的分片信息
        missing_peers = [p.id() for p in self.peers if p.id() not in self.node_shards]
        if missing_peers:
          print(f"[Node] Peers missing shard info: {missing_peers}, re-broadcasting")
          await self._broadcast_shard_config()

  async def _broadcast_shard_config(self):
    """广播自己的分片配置给其他节点"""
    if hasattr(self.discovery, 'get_current_node_shards'):
      print(f"[Node] Broadcasting shard config for {self.id}...")
      shards = self.discovery.get_current_node_shards()
      if DEBUG >= 2: print(f"[Node] Current node shards: {shards}")
      if shards:
        self.node_shards[self.id] = shards[0]
        self.node_shards_multi[self.id] = shards
        shard_message = json.dumps({
          "type": "node_shard_config",
          "node_id": self.id,
          "shard": shards[0].to_dict(),
          "shards": [s.to_dict() for s in shards]
        })
        if DEBUG >= 2: print(f"[Node] Broadcasting message: {shard_message}")
        await self.broadcast_opaque_status("", shard_message)
        print(f"[Node] Broadcasted {len(shards)} shard config(s)")
      else:
        if DEBUG >= 2: print(f"[Node] No shard configured for {self.id}, skipping broadcast")
    elif hasattr(self.discovery, 'get_current_node_shard'):
      print(f"[Node] Broadcasting shard config for {self.id}...")
      shard = self.discovery.get_current_node_shard()
      if DEBUG >= 2: print(f"[Node] Current node shard: {shard}")
      if shard:
        self.node_shards[self.id] = shard
        shard_message = json.dumps({
          "type": "node_shard_config",
          "node_id": self.id,
          "shard": shard.to_dict()
        })
        if DEBUG >= 2: print(f"[Node] Broadcasting message: {shard_message}")
        await self.broadcast_opaque_status("", shard_message)
        print(f"[Node] Broadcasted shard config: {shard}")
      else:
        if DEBUG >= 2: print(f"[Node] No shard configured for {self.id}, skipping broadcast")
    else:
      if DEBUG >= 2: print(f"[Node] Discovery does not support get_current_node_shard, skipping shard broadcast")

  async def _periodic_broadcast_loaded_models(self, initial_delay: float = 3.0, interval: float = 10.0):
    """定期广播已加载的模型，确保所有节点都能收到"""
    await asyncio.sleep(initial_delay)
    await self._broadcast_loaded_models()
    
    # 定期重播，确保新连接的节点也能收到
    while True:
      await asyncio.sleep(interval)
      if self.my_loaded_models and len(self.peers) > 0:
        await self._broadcast_loaded_models()

  async def _broadcast_loaded_models(self, gpu_memory=None):
    """广播自己已加载的模型给其他节点"""
    if not self.my_loaded_models:
      return
    
    print(f"[Node] Broadcasting loaded models for {self.id}: {list(self.my_loaded_models.keys())}")
    load_states = [state.to_dict() for state in self.my_loaded_models.values()]
    
    # 构建消息，包含实时 GPU 显存信息
    message_data = {
      "type": "node_loaded_models",
      "node_id": self.id,
      "loaded_models": load_states
    }
    
    # 如果有 GPU 显存数据，附加到消息中
    if gpu_memory:
      message_data["gpu_memory"] = gpu_memory
      print(f"[Node] 附加GPU显存: {gpu_memory.get('used', 0)}/{gpu_memory.get('total', 0)} MB")
    
    message = json.dumps(message_data)
    await self.broadcast_opaque_status("", message)
    
    if self.node_shards.get(self.id):
      shard = self.node_shards[self.id]
      shard_message = json.dumps({
        "type": "node_shard_config",
        "node_id": self.id,
        "shard": shard.to_dict(),
        "shards": [s.to_dict() for s in self.node_shards_multi.get(self.id, [shard])],
        "gpu_memory": gpu_memory or {}  # 也包含在分片配置中
      })
      await self.broadcast_opaque_status("", shard_message)
      print(f"[Node] Also broadcasted shard config: {shard}")

  async def _request_loaded_models_from_peers(self):
    """新节点启动后，主动请求所有peers的已加载模型信息"""
    if not self.peers:
      return
    
    print(f"[Node] Requesting loaded models from peers: {[p.id() for p in self.peers]}")
    
    request_data = {
      "type": "request_loaded_models",
      "node_id": self.id
    }
    
    if hasattr(self.discovery, 'get_my_address_info'):
      addr_info = self.discovery.get_my_address_info()
      if addr_info:
        request_data["address"] = addr_info.get("address")
        request_data["port"] = addr_info.get("port")
    
    request_message = json.dumps(request_data)
    await self.broadcast_opaque_status("", request_message)

  async def _respond_with_loaded_models(self, requester_id: str):
    """响应其他节点的已加载模型请求"""
    if not self.my_loaded_models:
      return
    
    print(f"[Node] Sending loaded models to {requester_id}: {list(self.my_loaded_models.keys())}")
    load_states = [state.to_dict() for state in self.my_loaded_models.values()]
    message = json.dumps({
      "type": "node_loaded_models",
      "node_id": self.id,
      "loaded_models": load_states
    })
    # 直接发送给请求者，而不是广播给所有人
    for peer in self.peers:
      if peer.id() == requester_id:
        try:
          await peer.send_opaque_status("", message)
        except Exception as e:
          print(f"[Node] Error sending loaded models to {requester_id}: {e}")
        break

  def on_model_loaded(self, shard: Shard):
    """当本节点加载模型时调用，记录并广播"""
    if shard.instance_id and shard.instance_id != "default":
      expected_suffix = f"::{shard.instance_id}"
      if shard.model_id.endswith(expected_suffix):
        model_id = shard.model_id
      else:
        model_id = f"{shard.model_id}::{shard.instance_id}"
    else:
      model_id = shard.model_id
    
    self.my_loaded_models[model_id] = ModelLoadState(model_id=model_id, shard=shard)
    
    # 同时更新 node_shards 和 node_shards_multi，确保数据一致性
    self.node_shards[self.id] = shard
    if self.id not in self.node_shards_multi:
      self.node_shards_multi[self.id] = []
    
    # [OK] 使用完整 model_id 检查和更新分片
    updated = False
    for i, existing_shard in enumerate(self.node_shards_multi[self.id]):
      expected_suffix = f"::{existing_shard.instance_id}" if existing_shard.instance_id and existing_shard.instance_id != "default" else ""
      if expected_suffix and existing_shard.model_id.endswith(expected_suffix):
        existing_full_id = existing_shard.model_id
      elif existing_shard.instance_id and existing_shard.instance_id != "default":
        existing_full_id = f"{existing_shard.model_id}::{existing_shard.instance_id}"
      else:
        existing_full_id = existing_shard.model_id
      if existing_full_id == model_id:
        self.node_shards_multi[self.id][i] = shard
        updated = True
        break
    if not updated:
      self.node_shards_multi[self.id].append(shard)
    
    # 更新分区策略中的模型位置
    if isinstance(self.partitioning_strategy, ModelAwarePartitioningStrategy):
      self.partitioning_strategy.update_model_location(self.id, self.my_loaded_models[model_id])
    
    # 广播新加载的模型
    asyncio.create_task(self._broadcast_loaded_models(), name=f"broadcast-loaded-models-{model_id}")
    print(f"[Node] Model loaded and broadcasted: {model_id} (instance={shard.instance_id})")

  async def stop(self) -> None:
    await self.discovery.stop()
    if self.server is not None:
      await self.server.stop()

  async def switch_discovery(self, new_discovery: Discovery) -> None:
    """
    运行时切换发现模块
    
    Args:
        new_discovery: 新的发现模块实例
    """
    print(f"[Node] 切换发现模块: {type(self.discovery).__name__} -> {type(new_discovery).__name__}")
    
    # 1. 停止当前的 discovery
    try:
      await self.discovery.stop()
      print(f"[Node] 已停止旧的发现模块")
    except Exception as e:
      print(f"[Node] 停止旧发现模块时出错: {e}")
    
    # 2. 更新 discovery 实例
    self.discovery = new_discovery
    
    # 3. 清理旧的 peers
    self.peers = {}
    self.topology = Topology()
    self.node_shards = {}
    self.node_shards_multi = {}
    self.node_loaded_models = {}
    print(f"[Node] 已清理旧的节点信息")
    
    # 4. 启动新的 discovery
    try:
      await self.discovery.start()
      print(f"[Node] 已启动新的发现模块")
    except Exception as e:
      print(f"[Node] 启动新发现模块时出错: {e}")
      raise
    
    # 5. 重新发现 peers
    await self.update_peers(0)
    await self.collect_topology(set())
    print(f"[Node] 发现模块切换完成，当前节点数: {len(self.peers) + 1}")
    
    # 6. 重新广播分片配置和已加载模型
    if self.node_shards.get(self.id):
      await self._broadcast_shard_config()
    if self.my_loaded_models:
      await self._broadcast_loaded_models()

  def on_node_status(self, request_id, opaque_status):
    try:
      if DEBUG >= 2: print(f"[DEBUG] on_node_status called with: {request_id=} {opaque_status=}")
      status_data = json.loads(opaque_status)
      status_type = status_data.get("type", "")
      if DEBUG >= 2: print(f"[DEBUG] Parsed status type: {status_type}")
      # 对于分片配置消息，总是打印日志
      if status_type == "node_shard_config":
        print(f"[Node] on_node_status received node_shard_config: {status_data}")
      
      # 首先获取 node_id（所有消息类型都需要）
      node_id = status_data.get("node_id")
      
      if status_type == "supported_inference_engines":
        engines = status_data.get("engines", [])
        self.topology_inference_engines_pool.append(engines)
      elif status_type == "caller_register":
        caller_id = status_data.get("node_id")
        caller_address = status_data.get("address")
        caller_port = status_data.get("port")
        caller_caps = status_data.get("device_capabilities", {})
        if DEBUG >= 2: print(f"[Node] Received caller_register from {caller_id} @ {caller_address}:{caller_port}")
        
        if caller_id and caller_id != self.id:
          if hasattr(self.discovery, 'add_known_node') and caller_address and caller_port:
            from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops, DeviceMemory
            flops_data = caller_caps.get("flops", {})
            memory_detail_data = caller_caps.get("memory_detail")
            
            caller_device_caps = DeviceCapabilities(
              model=caller_caps.get("model", "unknown"),
              chip=caller_caps.get("chip", "unknown"),
              memory=caller_caps.get("memory", 0),
              flops=DeviceFlops(
                fp32=flops_data.get("fp32", 0),
                fp16=flops_data.get("fp16", 0),
                int8=flops_data.get("int8", 0)
              ),
              memory_detail=DeviceMemory(
                total=memory_detail_data.get("total", 0),
                free=memory_detail_data.get("free", 0),
                used=memory_detail_data.get("used", 0)
              ) if memory_detail_data else None
            )
            self.discovery.add_known_node(
              node_id=caller_id,
              address=caller_address,
              port=caller_port,
              description="Incoming",
              device_capabilities=caller_device_caps
            )
            if DEBUG >= 2: print(f"[Node] Registered caller {caller_id} as known node")
            asyncio.create_task(self._update_peers_and_broadcast(), name=f"update-peers-{caller_id}")
      elif status_type == "node_status":
        status = status_data.get("status", "")
        if DEBUG >= 2: print(f"[DEBUG] Processing node_status: {status=} {node_id=}")
        if status.startswith("start_"):
          if DEBUG >= 2: print(f"[DEBUG] Setting active_node_id to: {node_id}")
          self.current_topology.active_node_id = node_id
          if DEBUG >= 2: print(f"[DEBUG] Active node ID is now: {self.current_topology.active_node_id}")
        elif status.startswith("end_"):
          if DEBUG >= 2: print(f"[DEBUG] Checking end status: current={self.current_topology.active_node_id}, received={node_id}")
          if node_id == self.current_topology.active_node_id:
            if DEBUG >= 2: print(f"[DEBUG] Resetting active_node_id to None")
            self.current_topology.active_node_id = None
            if DEBUG >= 2: print(f"[DEBUG] Active node ID is now: {self.current_topology.active_node_id}")

      download_progress = None
      if status_type == "download_progress":
        if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
        download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
        self.node_download_progress[status_data.get('node_id')] = download_progress

      # 收集节点性能统计 (从其他节点广播的统计信息)
      if status_type == "node_performance_stats" and node_id:
        stats = status_data.get("stats")
        if stats:
          self.node_stats[node_id] = stats
      
      # 收集节点分片配置 (从其他节点广播的分片信息)
      if status_type == "node_shard_config":
        node_id_from_msg = status_data.get("node_id")
        print(f"[Node] Received node_shard_config, node_id_from_msg={node_id_from_msg}, current node_id={node_id}")
        shard_data = status_data.get("shard")
        shards_data = status_data.get("shards")  # 获取所有分片信息（多模型分片支持）
        print(f"[Node] Shard data: {shard_data}, shards_data: {shards_data}")
        if shard_data and node_id_from_msg:
          from exo.inference.shard import Shard
          shard = Shard.from_dict(shard_data)
          self.node_shards[node_id_from_msg] = shard
          print(f"[Node] Received and stored shard config from {node_id_from_msg}: {shard}")
          # 如果有多分片信息，也保存到 node_shards_multi（用于多模型分片查询）
          if shards_data:
            self.node_shards_multi[node_id_from_msg] = [Shard.from_dict(s) for s in shards_data]
            print(f"[Node] Received and stored {len(shards_data)} shard config(s) from {node_id_from_msg}")
          print(f"[Node] Current node_shards: {self.node_shards}")

      # 收集节点已加载的模型信息
      if status_type == "node_loaded_models":
        node_id_from_msg = status_data.get("node_id")
        loaded_models_data = status_data.get("loaded_models", [])
        print(f"[Node] Received node_loaded_models from {node_id_from_msg}: {[m['model_id'] for m in loaded_models_data]}")
        
        if node_id_from_msg:
          if node_id_from_msg not in self.node_loaded_models:
            self.node_loaded_models[node_id_from_msg] = {}
          
          for model_data in loaded_models_data:
            load_state = ModelLoadState.from_dict(model_data)
            self.node_loaded_models[node_id_from_msg][load_state.model_id] = load_state
            
            # 更新分区策略中的模型位置
            if isinstance(self.partitioning_strategy, ModelAwarePartitioningStrategy):
              self.partitioning_strategy.update_model_location(node_id_from_msg, load_state)
          
          print(f"[Node] Updated node_loaded_models for {node_id_from_msg}: {list(self.node_loaded_models[node_id_from_msg].keys())}")

      # 处理已加载模型请求
      if status_type == "request_loaded_models":
        requester_id = status_data.get("node_id")
        requester_address = status_data.get("address")
        requester_port = status_data.get("port")
        print(f"[Node] Received request_loaded_models from {requester_id}")
        
        if requester_id and requester_id != self.id:
          if hasattr(self.discovery, 'add_known_node') and requester_address and requester_port:
            if requester_id not in [p.id() for p in self.peers]:
              self.discovery.add_known_node(
                node_id=requester_id,
                address=requester_address,
                port=requester_port,
                description="Incoming"
              )
          
          asyncio.create_task(self._respond_with_loaded_models(requester_id), name=f"respond-loaded-models-{requester_id}")

      # ==================== 处理来自 EXO Manager 的命令 ====================
      if status_type == "manager_shard_config":
        print(f"[Node] [Manager] 收到分片配置命令: {status_data.get('model_id')}")
        
        cmd_model_id = status_data.get("model_id")
        cmd_model_path = status_data.get("model_path")
        cmd_shard = status_data.get("shard", {})
        cmd_start_layer = cmd_shard.get("start_layer", 0)
        cmd_end_layer = cmd_shard.get("end_layer", 0)
        cmd_n_layers = cmd_shard.get("n_layers", 0)
        
        peer_list = status_data.get("peer_list", [])
        if peer_list:
          asyncio.create_task(self._register_peers_from_manager(peer_list), name=f"register-peers-from-manager")
        
        # 延迟启动模型加载，确保 P2P 注册完成
        async def _delayed_load():
          await asyncio.sleep(1.0)
          await self._handle_manager_load(cmd_model_id, cmd_model_path, cmd_start_layer, cmd_end_layer, cmd_n_layers)
        asyncio.create_task(_delayed_load(), name=f"delayed-load-{cmd_model_id}")
      
      elif status_type == "manager_unload_model":
        cmd_model_id = status_data.get("model_id")
        cmd_unload_all = status_data.get("unload_all_instances", False)
        print(f"[Node] [Manager] 收到卸载模型命令: {cmd_model_id} (卸载所有实例={cmd_unload_all})")

        asyncio.create_task(self._handle_manager_unload(cmd_model_id, unload_all_instances=cmd_unload_all))
      
      # 处理 Manager 请求实时 GPU 显存
      elif status_type == "request_gpu_status":
        print(f"[Node] [Manager] 收到 GPU 显存请求")
        asyncio.create_task(self._handle_gpu_status_request())
      
      elif status_type == "manager_peer_topology":
        peer_list = status_data.get("peer_list", [])
        if peer_list:
          asyncio.create_task(self._register_peers_from_manager(peer_list))

      if self.topology_viz:
        self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id, self.node_download_progress)
    except Exception as e:
      if DEBUG >= 1: print(f"Error on_node_status: {e}")
      if DEBUG >= 1: traceback.print_exc()

  async def _handle_manager_load(self, model_id, model_path, start_layer, end_layer, n_layers):
    """处理 Manager 发来的分片加载命令"""
    try:
      from exo.inference.shard import Shard
      
      # 提取 base_model_id 和 instance_id（避免文件系统路径污染）
      if "::" in model_id:
        base_model_id = model_id.split("::")[0]
        instance_id = model_id.split("::")[1]
      else:
        base_model_id = model_id
        instance_id = "default"
      
      # 创建 Shard 时使用 base_model_id（用于文件系统操作）
      shard = Shard(
        model_id=base_model_id,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        instance_id=instance_id
      )
      
      print(f"[Node] [Manager] 配置分片: {shard} (完整ID: {model_id})")
      
      self.node_shards[self.id] = shard
      
      if not model_path:
        model_path = self._resolve_model_path(base_model_id)  # 使用 base_model_id 解析路径
        if model_path:
          print(f"[Node] [Manager] 解析到本地路径: {model_path}")
        else:
          print(f"[Node] [Manager] 本地不存在，启动后台下载任务...")
          asyncio.create_task(self._download_and_load_shard(shard))
          print(f"[Node] [Manager] 后台下载已启动，返回响应避免阻塞")
          return
      
      if self.gpu_pool and model_path:
        load_result = await self.gpu_pool.load_single_shard(
          model_id=model_id,  # 完整ID用于管理
          base_model_id=base_model_id,  # 基础ID用于文件系统
          model_path=model_path,
          shard=shard
        )
        
        if load_result.get("success"):
          print(f"[Node] [Manager] 分片加载成功: {model_id} L{start_layer}-L{end_layer}")
          
          self.on_model_loaded(shard)
          
          gpu_memory = self._get_realtime_gpu_memory()
          await self._broadcast_shard_config()
          await self._broadcast_loaded_models(gpu_memory)
        else:
          print(f"[Node] [Manager] 分片加载失败: {load_result.get('error')}")
      else:
        if not self.gpu_pool:
          print(f"[Node] [Manager] GPU池未初始化，无法加载分片")
        elif not model_path:
          print(f"[Node] [Manager] 无法确定模型路径，跳过加载")
        
    except Exception as e:
      print(f"[Node] [Manager] 处理分片配置异常: {e}")
      traceback.print_exc()
  
  async def _register_peers_from_manager(self, peer_list: list):
    """注册 Manager 通知的 P2P 节点"""
    try:
      from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops, DeviceMemory
      
      registered_count = 0
      for peer_info in peer_list:
        peer_id = peer_info.get("node_id")
        peer_address = peer_info.get("address")
        peer_port = peer_info.get("port")
        
        if not peer_id or peer_id == self.id or not peer_address or not peer_port:
          continue
        
        if hasattr(self.discovery, 'add_known_node'):
          caps_data = peer_info.get("device_capabilities", {})
          flops_data = caps_data.get("flops", {})
          mem_data = caps_data.get("memory_detail")
          
          device_caps = DeviceCapabilities(
            model=caps_data.get("model", "unknown"),
            chip=caps_data.get("chip", "unknown"),
            memory=caps_data.get("memory", 0),
            flops=DeviceFlops(
              fp32=flops_data.get("fp32", 0),
              fp16=flops_data.get("fp16", 0),
              int8=flops_data.get("int8", 0)
            ),
            memory_detail=DeviceMemory(
              total=mem_data.get("total", 0) if mem_data else 0,
              free=mem_data.get("free", 0) if mem_data else 0,
              used=mem_data.get("used", 0) if mem_data else 0
            ) if mem_data else None
          )
          
          self.discovery.add_known_node(
            node_id=peer_id,
            address=peer_address,
            port=int(peer_port),
            description="Manager-P2P",
            device_capabilities=device_caps
          )
          registered_count += 1
      
      print(f"[Node] [P2P] 从 Manager 注册了 {registered_count} 个 P2P 节点: {[p['node_id'] for p in peer_list]}")
      
      if registered_count > 0:
        if hasattr(self, '_update_peers_and_broadcast'):
          await self._update_peers_and_broadcast()
        
        await self.collect_topology(set())
        logging.info(f"[Node] [P2P] Topology 更新完成, nodes={list(self.topology.nodes.keys())}")
        
    except Exception as e:
      print(f"[Node] [P2P] 注册 P2P 节点失败: {e}")

  async def _build_topology_response(self, request_payload: dict) -> dict:
    """构建 CollectTopology 的响应数据（供 WS grpc_relay 使用）"""
    topology = self.current_topology
    nodes = {}

    # 获取实时 GPU 显存和利用率
    realtime_memory = None
    gpu_utilization = None
    try:
      import pynvml
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
      realtime_memory = {
        "total": gpu_mem.total // 2**20,
        "free": gpu_mem.free // 2**20,
        "used": gpu_mem.used // 2**20
      }
      try:
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = {"gpu": util_rates.gpu, "memory": util_rates.memory}
      except Exception:
        pass
    except Exception:
      pass

    for node_id, cap in topology.nodes.items():
      flops_data = {}
      if hasattr(cap.flops, 'fp32'):
        flops_data = {"fp32": cap.flops.fp32, "fp16": cap.flops.fp16, "int8": cap.flops.int8}
      elif isinstance(cap.flops, dict):
        flops_data = cap.flops

      mem_detail = None
      if realtime_memory and node_id == self.id:
        mem_detail = realtime_memory
      elif hasattr(cap, 'memory_detail') and cap.memory_detail:
        if hasattr(cap.memory_detail, 'total'):
          mem_detail = {"total": cap.memory_detail.total, "free": cap.memory_detail.free, "used": cap.memory_detail.used}
        elif isinstance(cap.memory_detail, dict):
          mem_detail = cap.memory_detail

      # 已加载模型列表
      loaded_models = []
      if hasattr(self, 'my_loaded_models') and self.my_loaded_models:
        for model_id, load_state in self.my_loaded_models.items():
          shard_obj = load_state.shard if hasattr(load_state, 'shard') else None
          loaded_models.append({
            "model_id": model_id,
            "start_layer": shard_obj.start_layer if shard_obj else 0,
            "end_layer": shard_obj.end_layer if shard_obj else 0,
            "n_layers": shard_obj.n_layers if shard_obj else 0
          })

      nodes[node_id] = {
        "model": cap.model,
        "chip": cap.chip,
        "memory": cap.memory,
        "flops": flops_data,
        "memory_detail": mem_detail,
        "loaded_models": loaded_models,
      }

    peer_graph = {}
    for node_id, connections in topology.peer_graph.items():
      peer_graph[node_id] = [
        {"to_id": conn.to_id, "description": conn.description} for conn in connections
      ]

    return {
      "nodes": nodes,
      "peer_graph": peer_graph,
    }

  async def _download_and_load_shard(self, shard: Shard):
    """
    后台下载并加载分片（非阻塞）
    
    这个方法会在后台异步执行，避免阻塞主事件循环，
    确保 Node 在下载过程中仍能响应 Manager 的心跳和状态查询。
    
    重要：使用 base_model_id 进行文件系统操作，避免 instance_id 污染路径
    """
    try:
      base_model_id = shard.base_model_id
      full_model_id = shard.full_model_id
      start_layer = shard.start_layer
      end_layer = shard.end_layer
      
      print(f"[Node] [Download] 开始下载模型: {full_model_id} (基础ID: {base_model_id}) L{start_layer}-L{end_layer}")
      
      # 检查 inference_engine 和 shard_downloader 是否可用
      if not hasattr(self, 'inference_engine') or not self.inference_engine:
        print(f"[Node] [Download] [FAIL] inference_engine 未初始化")
        return
      
      if not hasattr(self.inference_engine, 'shard_downloader') or not self.inference_engine.shard_downloader:
        print(f"[Node] [Download] [FAIL] shard_downloader 未初始化")
        return
      
      # 创建用于下载的 Shard（使用 base_model_id，避免路径污染）
      download_shard = Shard(
        model_id=base_model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
        repo_id=shard.repo_id,
        tie_word_embeddings=shard.tie_word_embeddings
      )
      
      # 执行下载（这是耗时操作，但在独立的后台任务中运行）
      downloaded_path = await self.inference_engine.shard_downloader.ensure_shard(
        download_shard,
        self.inference_engine.__class__.__name__
      )
      
      model_path = str(downloaded_path)
      print(f"[Node] [Download] [OK] 模型下载完成: {model_path} (实例: {full_model_id})")
      
      # 下载完成后，加载分片到 GPU 池（传入完整信息用于管理）
      if self.gpu_pool:
        print(f"[Node] [Download] 开始加载分片到 GPU 池...")
        load_result = await self.gpu_pool.load_single_shard(
          model_id=full_model_id,
          base_model_id=base_model_id,
          model_path=model_path,
          shard=shard
        )
        
        if load_result.get("success"):
          print(f"[Node] [Download] [OK] 分片加载成功: {full_model_id} L{start_layer}-L{end_layer}")
          
          # 广播更新后的状态
          gpu_memory = self._get_realtime_gpu_memory()
          await self._broadcast_shard_config()
          await self._broadcast_loaded_models(gpu_memory)
        else:
          print(f"[Node] [Download] [FAIL] 分片加载失败: {load_result.get('error')}")
      else:
        print(f"[Node] [Download] [WARN] GPU池未初始化，无法加载分片")
        
    except Exception as e:
      print(f"[Node] [Download] [FAIL] 下载或加载过程异常: {e}")
      import traceback
      traceback.print_exc()
  
  def _resolve_model_path(self, model_id):
    """
    根据 model_id 解析本地模型路径
    
    流程:
    1. 从 exo.models.model_cards 获取 repo_id (如 "Qwen/Qwen3-0.6B")
    2. 将 repo_id 转换为本地路径 (~/.cache/exo/downloads/Qwen--Qwen3-0.6B)
    
    Args:
        model_id: 模型标识符 (如 "qwen-3-0.6b")
    
    Returns:
        本地模型路径字符串，如果找不到则返回 None
    """
    try:
      from exo.models import model_cards
      from exo.download.new_shard_download import exo_home
      
      if model_id not in model_cards:
        print(f"[Node] [Manager] 模型 {model_id} 不在标准配置中")
        return None
      
      config = model_cards[model_id]
      repo_info = config.get("repo", {})
      
      # 获取 PyTorch 或 Dummy 引擎的 repo ID
      repo_id = None
      for engine_name, repo in repo_info.items():
        if 'PyTorch' in engine_name or 'Dummy' in engine_name:
          repo_id = repo
          break
      
      if not repo_id:
        print(f"[Node] [Manager] 模型 {model_id} 没有 PyTorch/Dummy 引擎配置")
        return None
      
      # 转换为本地路径: Qwen/Qwen3-0.6B → ~/.cache/exo/downloads/Qwen--Qwen3-0.6B
      repo_id_for_path = repo_id.replace("/", "--")
      local_path = exo_home() / "downloads" / repo_id_for_path
      
      # 检查是否存在嵌套目录（某些模型的特殊结构）
      if local_path.exists():
        parts = repo_id.split("/")
        if len(parts) >= 2:
          nested_path = local_path / parts[-2] / parts[-1]
          if nested_path.exists():
            print(f"[Node] [Manager] 发现嵌套目录: {nested_path}")
            return str(nested_path)
        
        return str(local_path)
      else:
        print(f"[Node] [Manager] 本地路径不存在: {local_path}")
        return None
        
    except Exception as e:
      print(f"[Node] [Manager] 解析模型路径失败: {e}")
      import traceback
      traceback.print_exc()
      return None
  
  def _get_realtime_gpu_memory(self):
    """
    获取实时 GPU 显存使用情况
    
    使用 pynvml 获取当前 GPU 的:
    - total: 总显存 (MB)
    - free: 可用显存 (MB)  
    - used: 已用显存 (MB)
    
    Returns:
        dict: {"total": int, "free": int, "used": int} 或 None
    """
    try:
      import pynvml
      
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      
      gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8') if isinstance(pynvml.nvmlDeviceGetName(handle), bytes) else pynvml.nvmlDeviceGetName(handle)
      gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      
      memory_total_mb = gpu_memory_info.total // 2**20
      memory_free_mb = gpu_memory_info.free // 2**20
      memory_used_mb = gpu_memory_info.used // 2**20
      
      result = {
        "total": memory_total_mb,
        "free": memory_free_mb,
        "used": memory_used_mb,
        "gpu_name": gpu_name,
        "memory_utilization_percent": round(memory_used_mb / memory_total_mb * 100, 1) if memory_total_mb > 0 else 0
      }
      
      try:
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        result["gpu_utilization_percent"] = util_rates.gpu
        result["memory_controller_utilization"] = util_rates.memory
      except:
        result["gpu_utilization_percent"] = 0
      
      print(f"[Node] [GPU] 实时显存: {gpu_name}, {memory_used_mb}/{memory_total_mb} MB "
            f"(显存使用率: {result['memory_utilization_percent']}%, "
            f"GPU计算利用率: {result.get('gpu_utilization_percent', 'N/A')}%)")
      
      return result
      
    except Exception as e:
      print(f"[Node] [GPU] 获取GPU显存失败: {e}")
      return None
  
  async def _handle_gpu_status_request(self):
    """处理 Manager 发来的 GPU 显存请求"""
    try:
      gpu_memory = self._get_realtime_gpu_memory()
      
      if gpu_memory:
        response_msg = json.dumps({
          "type": "gpu_status_response",
          "node_id": self.id,
          "gpu_memory": gpu_memory,
          "timestamp": time.time()
        })
        await self.broadcast_opaque_status("", response_msg)
        print(f"[Node] [Manager] 已返回 GPU 显存: {gpu_memory.get('used', 0)}/{gpu_memory.get('total', 0)} MB")
        
    except Exception as e:
      print(f"[Node] [Manager] 处理GPU显存请求异常: {e}")

  async def _handle_manager_unload(self, model_id, unload_all_instances=False):
    """处理 Manager 发来的卸载模型命令（支持多实例）"""
    try:
      if self.gpu_pool:
        unload_result = await self.gpu_pool.unload(
          model_id=model_id,
          unload_all_instances=unload_all_instances
        )

        if unload_result.get("success"):
          print(f"[Node] [Manager] 模型卸载成功: {model_id}")

          if model_id in self.node_shards:
            del self.node_shards[model_id]

          # [STAR] 清理对应的推理引擎实例（释放显存）
          if "::" in model_id:
            parts = model_id.split("::", 1)
            instance_id = parts[1]
          else:
            instance_id = "default"

          if unload_all_instances:
            keys_to_remove = [
              k for k in self.inference_engines.keys()
              if k != "default" and (model_id in k or k == instance_id)
            ]
            for key in keys_to_remove:
              if key in self.inference_engines:
                engine = self.inference_engines[key]
                if hasattr(engine, 'unload_model'):
                  try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                      await engine.unload_model(model_id)
                    else:
                      loop.run_until_complete(engine.unload_model(model_id))
                    print(f"[Node] [Manager] [OK] 已调用引擎 {key} 的卸载方法释放资源")
                  except Exception as e:
                    print(f"[Node] [Manager] [WARN] 引擎 {key} 卸载方法执行异常: {e}")
                
                del self.inference_engines[key]
                print(f"[Node] [Manager] [OK] 已清理引擎实例: {key}")
            print(f"[Node] [Manager] [STAR] 已清理 {len(keys_to_remove)} 个引擎实例")
          else:
            if instance_id in self.inference_engines and instance_id != "default":
              engine = self.inference_engines[instance_id]
              if hasattr(engine, 'unload_model'):
                try:
                  loop = asyncio.get_event_loop()
                  if loop.is_running():
                    await engine.unload_model(model_id)
                  else:
                    loop.run_until_complete(engine.unload_model(model_id))
                  print(f"[Node] [Manager] [OK] 已调用引擎 {instance_id} 的卸载方法释放资源")
                except Exception as e:
                  print(f"[Node] [Manager] [WARN] 引擎 {instance_id} 卸载方法执行异常: {e}")
              
              del self.inference_engines[instance_id]
              print(f"[Node] [Manager] [OK] 已清理引擎实例: {instance_id}")

          await self._broadcast_shard_config()
          await self._broadcast_loaded_models()
        else:
          print(f"[Node] [Manager] 模型卸载失败: {unload_result.get('error')}")
      else:
        print(f"[Node] [Manager] GPU池未初始化，无法卸载模型")
        
    except Exception as e:
      print(f"[Node] [Manager] 处理卸载命令异常: {e}")

  def get_supported_inference_engines(self):
    supported_engine_names = []
    if self.inference_engine is None:
      if DEBUG >= 1: print("inference_engine is None, returning empty supported engines list")
      return supported_engine_names
    engine_class_name = self.inference_engine.__class__.__name__
    if engine_class_name == 'MLXDynamicShardInferenceEngine':
      supported_engine_names.append('mlx')
    elif engine_class_name ==  'PyTorchInferenceEngine':
      supported_engine_names.append('pytorch')
    # 只返回实际支持的引擎，不再默认添加tinygrad
    return supported_engine_names

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool
  
  token_count = 0
  first_token_time = 0

  async def process_inference_result(
    self,
    shard,
    result: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ):
    try:
      my_shard = await self.get_current_shard(shard)
      pass  # logging.info(f"[process_inference_result] START: shard={shard}, my_shard={my_shard}, request_id={request_id}, my_shard.is_last_layer()={my_shard.is_last_layer()}, node_id={self.id}")
    except Exception as e:
      pass  # logging.warning(f"[process_inference_result] Failed to get_current_shard, using input shard: {e}")
      my_shard = shard

    if shard.model_id != 'stable-diffusion-2-1-base' and not shard.model_id.startswith('qwen-3-tts'):
      if request_id not in self.buffered_token_output:
        self.buffered_token_output[request_id] = ([], False)
        pass  # logging.info(f"[process_inference_result] Created new buffered_token_output for {request_id}")
      else:
        pass  # logging.info(f"[process_inference_result] Existing buffered_token_output for {request_id}: {len(self.buffered_token_output[request_id][0])} tokens")
      
      # 优先使用请求中的 max_tokens，否则使用节点默认值
      max_tokens = self.max_generate_tokens
      if inference_state and 'max_tokens' in inference_state:
        max_tokens = inference_state['max_tokens']
        if DEBUG >= 2: print(f"[{request_id}] Using max_tokens from inference_state: {max_tokens}")
      else:
        if DEBUG >= 2: print(f"[{request_id}] Using default max_tokens: {max_tokens}, inference_state={inference_state}")
      
      is_finished = len(self.buffered_token_output[request_id][0]) >= max_tokens
      pass  # logging.info(f"[process_inference_result] Initial is_finished={is_finished}, buffered_tokens={len(self.buffered_token_output[request_id][0])}, max_tokens={max_tokens}")
      
      if my_shard.is_last_layer() and not is_finished:
        generated_tokens = self.buffered_token_output[request_id][0]
        temp = inference_state.get('temperature', self.default_sample_temperature)
        top_k = inference_state.get('top_k', 50)
        top_p = inference_state.get('top_p', 0.9)
        pass  # logging.info(f"[process_inference_result] About to call sample()...")
        try:
            # [STAR] 多实例支持：使用对应实例的引擎进行采样
            sample_instance_id = getattr(my_shard, 'instance_id', None) or "default"
            sample_engine = self.get_engine(sample_instance_id)
            
            print(f"[[ROUTE-CHECK] ROUTE] sample(): instance={sample_instance_id}, "
                  f"sample_id={id(sample_engine)}, default_id={id(self.inference_engine)}, "
                  f"is_default={id(sample_engine) == id(self.inference_engine)}")
            
            token = await sample_engine.sample(
                result,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.2,
                generated_tokens=generated_tokens if generated_tokens else None,
                shard=my_shard
            )
            pass  # logging.info(f"[process_inference_result] sample() completed: token={token.item()}")
        except Exception as sample_err:
            pass  # logging.error(f"[process_inference_result] sample() FAILED: {sample_err}")
            import traceback
            traceback.print_exc()
            raise
        pass  # logging.info(f"[process_inference_result] Calling ensure_shard after sampling...")
        # [STAR] Multi-instance support: use corresponding engine instance
        target_instance_id = getattr(my_shard, 'instance_id', None) or "default"
        target_engine = self.get_engine(target_instance_id)
        
        current_task = asyncio.current_task()
        task_name = current_task.get_name() if current_task else "no-task"
        
        print(f"[[ROUTE-CHECK] ROUTE] process_inference_result ensure_shard: instance={target_instance_id}, "
              f"target_id={id(target_engine)}, default_id={id(self.inference_engine)}, "
              f"is_default={id(target_engine) == id(self.inference_engine)}, task={task_name}")
        
        await target_engine.ensure_shard(my_shard)
        pass  # logging.info(f"[process_inference_result] ensure_shard completed, appending token...")
        self.buffered_token_output[request_id][0].append(token.item())
        tokenizer = target_engine.get_tokenizer(my_shard.model_id)
        pass  # logging.info(f"[process_inference_result] Tokenizer type: {type(tokenizer).__name__}, token_id={token.item()}, eos_token_id={tokenizer.eos_token_id if tokenizer else 'None'}")
        eos_tokens = {tokenizer.eos_token_id} if tokenizer else {151645}
        eos_tokens.add(151643)
        is_finished = token.item() in eos_tokens or is_finished or len(self.buffered_token_output[request_id][0]) >= max_tokens
        pass  # logging.info(f"[process_inference_result] After sampling: token={token.item()}, eos_token_id={tokenizer.eos_token_id if tokenizer else 'None'}, is_finished={is_finished}, buffered_tokens={len(self.buffered_token_output[request_id][0])}")
        if DEBUG >= 2: print(f"[{request_id}] result size: {result.size}, is finished: {is_finished}, buffered tokens: {len(self.buffered_token_output[request_id][0])}")
        forward = token.reshape(1, -1)
        intermediate_result = [self.buffered_token_output[request_id][0][-1]]
      else:
        pass  # logging.info(f"[process_inference_result] Not last layer (my_shard.is_last_layer={my_shard.is_last_layer()}) or already finished, forwarding result directly")
        forward = result
        intermediate_result = None
    else:
      # [STAR] Multi-instance support: use corresponding engine instance
      else_instance_id = getattr(shard, 'instance_id', None) or "default"
      else_engine = self.get_engine(else_instance_id)
      
      else_task = asyncio.current_task()
      else_task_name = else_task.get_name() if else_task else "no-task"
      
      print(f"[[ROUTE-CHECK] ROUTE] process_inference_result else-ensure: instance={else_instance_id}, "
            f"target_id={id(else_engine)}, default_id={id(self.inference_engine)}, "
            f"is_default={id(else_engine) == id(self.inference_engine)}, task={else_task_name}")
      
      await else_engine.ensure_shard(shard)
      is_finished = inference_state.get("is_finished", False)
      intermediate_result, inference_state = self.handle_stable_diffusion(inference_state, result)
      forward = result
    if shard.model_id.startswith('qwen-3-tts'):
      is_finished = True
      intermediate_result = result
      forward = result
    if my_shard.is_last_layer():
      self.trigger_on_token_callbacks(request_id, intermediate_result, is_finished)
      asyncio.create_task(self.broadcast_result(request_id, intermediate_result, is_finished))

    if is_finished:
      if shard.model_id != 'stable-diffusion-2-1-base' and not shard.model_id.startswith('qwen-3-tts'):
        self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
      self.outstanding_requests.pop(request_id)
      if request_id in self.request_kv_cache:
        del self.request_kv_cache[request_id]
        if DEBUG >= 2: print(f"[{request_id}] Cleaned up KV cache")
      if shard.model_id.startswith('qwen-3-tts'):
        return intermediate_result
      return np.array(self.buffered_token_output[request_id][0]) if shard.model_id != 'stable-diffusion-2-1-base' else intermediate_result
    elif my_shard.is_last_layer() and intermediate_result is not None:
      pass  # logging.info(f"[process_inference_result] Last layer sampled token, returning to caller for next generation step")
      return forward
    else:
      pass  # logging.info(f"[process_inference_result] Not finished, forwarding to next partition (my_shard.is_last_layer={my_shard.is_last_layer()})")
      self.outstanding_requests[request_id] = "waiting"
      next_partition_index = self.get_partition_index(offset = 1)
      pass  # logging.info(f"[process_inference_result] next_partition_index={next_partition_index}, forward.shape={forward.shape if hasattr(forward, 'shape') else 'N/A'}")
      if inference_state is None:
        inference_state = {}
      inference_state['is_finished'] = is_finished
      inference_state['generated_tokens'] = self.buffered_token_output.get(request_id, ([], False))[0]
      
      # 关键修复：分片间传递隐藏状态时，不传递 position 字段
      # 每个节点应根据自己的 KV 缓存独立计算 cache_position
      # 只有在生成循环中（单token推理）才由调用方显式设置 position
      # 删除从其他节点 KV 缓存读取 position 的逻辑，避免位置编码错误
      if 'position' in inference_state:
          pass  # logging.info(f"[POS-CLEAR] [{request_id}] Removing position from inference_state before cross-shard forward (value was: {inference_state['position']})")
          del inference_state['position']

      pass  # logging.info(f"[process_inference_result] Awaiting forward_tensor to partition {next_partition_index} with my_shard={my_shard}...")
      forward_result = await self.forward_tensor(my_shard, forward, request_id, next_partition_index, inference_state)
      pass  # logging.info(f"[process_inference_result] forward_tensor returned: shape={forward_result.shape if forward_result is not None else None}")
      
      # 如果下一层是最后一层，forward_result 可能包含采样的 token 或 logits
      # 需要将结果返回给调用者继续处理
      return forward_result

  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = {},
  ) -> Optional[np.ndarray]:
    await self._enter_inference()
    try:
        start_time = time.perf_counter_ns()
        start_status = json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": base_shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
        })
        if DEBUG >= 2: print(f"[DEBUG] Broadcasting start status: {start_status}")
        asyncio.create_task(
          self.broadcast_opaque_status(
            request_id,
            start_status,
          )
        )
        if DEBUG >= 2: print(f"[DEBUG] Start status broadcast scheduled")

        start_time = time.perf_counter_ns()
        resp = await self._process_prompt(base_shard, prompt, request_id, inference_state)
        end_time = time.perf_counter_ns()
        elapsed_time_ns = end_time - start_time

        end_status = json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": base_shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        })
        if DEBUG >= 2: print(f"[DEBUG] Broadcasting end status: {end_status}")
        asyncio.create_task(
          self.broadcast_opaque_status(
            request_id,
            end_status,
          )
        )
        if DEBUG >= 2: print(f"[DEBUG] End status broadcast scheduled")
        if DEBUG >= 2: print(f"[{request_id}] process prompt:  {elapsed_time_ns}")
        return resp
    finally:
      await self._exit_inference()

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None,
                            inference_state: Optional[dict] = None) -> Optional[np.ndarray]:
      if request_id is None:
          request_id = str(uuid.uuid4())

      print(f"[[ROUTE-CHECK] ROUTE] _process_prompt 入口: "
            f"shard={base_shard.model_id}::{getattr(base_shard, 'instance_id', '?')}, "
            f"引擎池={list(self.inference_engines.keys())}, "
            f"default_id={id(self.inference_engine)}")

      shard = await self.get_current_shard(base_shard)

      if not shard.is_first_layer():
          if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
          self.outstanding_requests[request_id] = "waiting"

          # 检查是否有图像数据需要处理
          # 如果有图像，先在本地编码为 input_ids，然后作为 tensor 转发
          # 因为图像数据无法通过 prompt 字符串传递
          if inference_state and inference_state.get("image") is not None and self.inference_engine is not None:
              if DEBUG >= 2: print(f"[{request_id}] image detected, encoding locally before forwarding")
              try:
                  # 获取首分片（层 0 开始的分片）
                  partitions = self.partitioning_strategy.partition(self.topology)
                  first_shard = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)[0]

                  # 使用 inference_engine 的 processor 编码图像和文本
                  # 注意：这里不需要加载模型权重，只需要 tokenizer 和 processor
                  image = inference_state.get("image")
                  original_prompt = inference_state.get("original_prompt", prompt)

                  # 检查是否已经有 processor，如果没有，尝试加载
                  processor = self.inference_engine.get_processor(first_shard.model_id)
                  if processor is None:
                      if DEBUG >= 2: print(f"[{request_id}] loading tokenizer and processor for image encoding")
                      # 使用模型路径加载 processor（不加载模型权重）
                      model_path = await self.shard_downloader.ensure_shard(first_shard, self.inference_engine.__class__.__name__)
                      from transformers import AutoProcessor
                      processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                      # 设置到对应的模型引擎
                      self.inference_engine.set_processor_and_tokenizer(first_shard.model_id, processor)

                  processor = self.inference_engine.get_processor(first_shard.model_id)
                  if image is not None and processor is not None:
                      messages = [
                          {
                              "role": "user",
                              "content": [
                                  {"type": "image", "image": image},
                                  {"type": "text", "text": original_prompt},
                              ],
                          }
                      ]
                      text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                      inputs = processor(text=text, images=[image], return_tensors="pt")
                      input_ids = inputs['input_ids']
                      tokens = input_ids.cpu().numpy()

                      # 将 pixel_values 和 image_grid_thw 放入 inference_state
                      # 注意：需要转换为 numpy 数组以便序列化传输
                      pixel_values = inputs.get('pixel_values', None)
                      image_grid_thw = inputs.get('image_grid_thw', None)
                      if pixel_values is not None:
                          inference_state['pixel_values'] = pixel_values.cpu().numpy()
                      if image_grid_thw is not None:
                          inference_state['image_grid_thw'] = image_grid_thw.cpu().numpy()

                      if DEBUG >= 2: print(f"[{request_id}] encoded tokens shape: {tokens.shape}")

                      # 使用 forward_tensor 转发编码后的 input_ids 到首分片节点
                      resp = await self.forward_tensor(first_shard, tokens, request_id, 0, inference_state)
                      return None
              except Exception as e:
                  if DEBUG >= 1: print(f"[{request_id}] error encoding prompt with image: {e}")
                  traceback.print_exc()
                  # 如果编码失败，回退到原来的 forward_prompt

          resp = await self.forward_prompt(shard, prompt, request_id, 0, inference_state)
          return None
      else:
          # [STAR] 多实例支持：根据 shard.instance_id 选择对应的引擎
          instance_id = getattr(shard, 'instance_id', None) or "default"
          target_engine = self.get_engine(instance_id)

          if target_engine is None:
              print(f"[{request_id}] inference_engine is None (instance: {instance_id}), cannot process prompt")
              return None

          # [ROUTE-CHECK] 关键诊断：验证引擎实例一致性（无条件输出）
          is_default_engine = (id(target_engine) == id(self.inference_engine))
          cached_models = target_engine.get_loaded_models() if hasattr(target_engine, 'get_loaded_models') else []
          print(f"[[ROUTE-CHECK] ROUTE] infer_prompt: instance={instance_id}, "
                f"target_id={id(target_engine)}, default_id={id(self.inference_engine)}, "
                f"is_default={is_default_engine}, cached={cached_models}")

          if is_default_engine and instance_id != "default":
              print(f"[[WARN] ROUTE] 请求 {instance_id} → default引擎! 池={list(self.inference_engines.keys())}")

          self.outstanding_requests[request_id] = "processing"
          result, inference_state = await target_engine.infer_prompt(request_id, shard, prompt, inference_state)
          print(f"[[ROUTE-CHECK] ROUTE] infer_prompt returned (engine: {instance_id})")
          
          # TTS 模型：一次性生成音频，不需要 token 生成循环
          if shard.model_id.startswith('qwen-3-tts'):
              tts_task = asyncio.current_task()
              print(f"[[ROUTE-CHECK] ROUTE] TTS process_inference_result call, task={tts_task.get_name() if tts_task else 'no-task'}")
              ret = await self.process_inference_result(shard, result, request_id, inference_state)
              return ret
          
          # 保存 KV 缓存到节点缓存中，以便后续生成步骤使用
          if 'past_key_values' in inference_state:
              self.request_kv_cache[request_id] = inference_state['past_key_values']
              if DEBUG >= 2: print(f"[{request_id}] Saved initial past_key_values to cache from infer_prompt")
          
          ret = await self.process_inference_result(shard, result, request_id, inference_state)
          
          # 多节点联合推理：实现生成循环
          max_tokens = inference_state.get('max_tokens', self.max_generate_tokens)
          generation_step = 0
          
          # 关键修复：从本节点 KV 缓存获取正确的初始位置（prompt 的 token 数量）
          # 这是第一个新生成 token 应该所在的位置
          current_position = 0
          if request_id in self.request_kv_cache:
              cache = self.request_kv_cache[request_id]
              if hasattr(cache, 'layers') and cache.layers:
                  first_layer = cache.layers[0]
                  if hasattr(first_layer, 'is_initialized') and first_layer.is_initialized and hasattr(first_layer, 'get_seq_length'):
                      current_position = first_layer.get_seq_length()
              elif hasattr(cache, 'key_cache') and cache.key_cache:
                  if len(cache.key_cache) > 0 and cache.key_cache[0] is not None:
                      current_position = cache.key_cache[0].shape[2] if len(cache.key_cache[0].shape) >= 3 else 0
          logging.info(f"[process_prompt] Starting generation loop, initial_position={current_position}, max_tokens={max_tokens}")
          
          while generation_step < max_tokens:
              if request_id in self.buffered_token_output:
                  tokens, finished = self.buffered_token_output[request_id]
                  if finished or len(tokens) >= max_tokens:
                      break
              
              if ret is not None and hasattr(ret, 'shape') and len(ret.shape) > 0:
                  result = ret
                  generation_step += 1
                  
                  if ret.ndim == 1 and ret.dtype in [np.int32, np.int64]:
                      logging.info(f"[process_prompt] Received EOS signal: {len(ret)} tokens")
                      self.buffered_token_output[request_id] = (ret.tolist(), True)
                      break
                  
                  # 更新位置信息（关键修复：每步 +1）
                  current_position += 1
                  inference_state['position'] = current_position
                  
                  # 更新本地 buffered_token_output（从返回的 token）
                  if request_id not in self.buffered_token_output:
                      self.buffered_token_output[request_id] = ([], False)
                  if len(ret.shape) >= 2 and ret.shape[1] > 0:
                      token_val = int(ret[0][0]) if ret.ndim == 2 else int(ret.item())
                      self.buffered_token_output[request_id][0].append(token_val)
                      
                      tokens_so_far = self.buffered_token_output[request_id][0]
                      is_finished_now = len(tokens_so_far) >= max_tokens
                      if is_finished_now:
                          self.buffered_token_output[request_id] = (tokens_so_far, True)
                  
                  # 注意：不在此处 trigger_on_token_callbacks / broadcast_result
                  # 因为 last layer 节点(node-2)的 process_inference_result 已经触发过了
                  # 协调节点(node-1)重复触发会导致 API 层收到双倍 token 造成输出重复
                  
                  # 优化：直接传递 token 作为 input_ids，避免 token→embedding→numpy→torch 的无效往返
                  # infer_tensor 内部会自动识别 input_ids 并执行 embed_tokens 查表
                  if len(result.shape) == 2 and result.shape[1] == 1:
                      result = np.array([[token_val]], dtype=np.int64)

                  my_shard_loop = await self.get_current_shard(shard)
                  # [STAR] 多实例支持：generation loop 也必须使用对应实例的引擎
                  # 这是修复推理快速退出的关键！之前错误地使用了 self.inference_engine
                  loop_instance_id = getattr(my_shard_loop, 'instance_id', None) or instance_id
                  loop_engine = self.get_engine(loop_instance_id)

                  # [ROUTE-CHECK] 关键诊断：每次 infer_tensor 前都验证路由（不只是 step=0）
                  loop_is_default = (id(loop_engine) == id(self.inference_engine))
                  if generation_step == 0 or (loop_is_default and loop_instance_id != "default"):
                      loop_cached = loop_engine.get_loaded_models() if hasattr(loop_engine, 'get_loaded_models') else []
                      print(f"[[ROUTE-CHECK] ROUTE] gen_loop step={generation_step}: "
                            f"loop_instance={loop_instance_id}, "
                            f"loop_id={id(loop_engine)}, default_id={id(self.inference_engine)}, "
                            f"is_default={loop_is_default}, cached={loop_cached}, "
                            f"shard_instance={getattr(my_shard_loop, 'instance_id', '?')}")
                      
                      if loop_is_default and loop_instance_id != "default":
                          print(f"[[!!!ROUTE-ERROR!!!] ROUTE] [WARN][WARN][WARN] gen_loop 路由错误! "
                                f"{loop_instance_id} → default引擎! "
                                f"引擎池={list(self.inference_engines.keys())}")
                          import traceback
                          print("[[!!!ROUTE-ERROR!!!] ROUTE] 调用栈:")
                          for line in traceback.format_stack()[-8:-1]:
                              print(f"  {line.strip()}")

                  result, inference_state = await loop_engine.infer_tensor(request_id, my_shard_loop, result, inference_state)
                  ret = await self.process_inference_result(shard, result, request_id, inference_state)
              else:
                  logging.warning(f"[process_prompt] No valid result returned, stopping generation")
                  break
          
          if request_id in self.buffered_token_output:
              tokens, finished = self.buffered_token_output[request_id]
              logging.info(f"[process_prompt] Final result: tokens={len(tokens)}, finished={finished}")
              if not finished and len(tokens) > 0:
                  logging.info(f"[process_prompt] Sending final finished signal")
                  self.buffered_token_output[request_id] = (tokens, True)
                  self.trigger_on_token_callbacks(request_id, [], True)
                  asyncio.create_task(self.broadcast_result(request_id, [], True))
          
          return ret

  async def enqueue_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    request_id: Optional[str] = None,
    train: bool = False,
  ):
    shard = await self.get_current_shard(base_shard)
    if shard.is_first_layer():
      loss = await self.process_example(shard, example, target, length, train, request_id)
      return loss
    else:
      if request_id is None:
        request_id = str(uuid.uuid4())
      self.outstanding_requests[request_id] = "waiting"
      loss = await self.forward_example(shard, example, target, length, train, request_id, 0) 
    return loss

  async def coordinate_save(
    self,
    base_shard: Shard,
    iteration: int,
    destination: str,
  ):
    shard = await self.get_current_shard(base_shard)
    model = shard.model_id
    sid = shard.__hash__()
    path = f"{destination}/{model}/{sid}-{iteration}.safetensors"
    self.outstanding_requests[f"{sid}::{iteration}"] = "Checking"
    if model not in self.checkpoints:
      self.checkpoints[model] = {}
    if sid not in self.checkpoints[model]:
      self.checkpoints[model][sid] = []
    if len(self.checkpoints[model][sid]) < 1 or self.checkpoints[model][sid][-1] < iteration:
      print(f"Saving checkpoint to {path}")
      self.outstanding_requests[f"{sid}::{iteration}"] = "Saving"
      import os
      os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
      await self.inference_engine.save_checkpoint(shard, path)
      self.checkpoints[model][sid] = sorted(self.checkpoints[model][sid] + [iteration])
    self.outstanding_requests.pop(f"{sid}::{iteration}")

  async def process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ):
    shard = await self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"start_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "example_size": example.size,
          "example_shape": example.shape,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_example(shard, example, target, length, train, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"end_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    return resp

  async def _process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = await self.get_current_shard(base_shard)
    if DEBUG >= 1: print(f"[{request_id}] process_example: {example.shape=}")
    try:
      target = target.astype(int)
      if train:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "training"
          loss, grad = await self.inference_engine.train(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss, backgrad = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
          self.outstanding_requests[request_id] = "training"
          partial_loss, grad = await self.inference_engine.train(request_id, shard, example, backgrad, length, loss="back_gradient")
        self.outstanding_requests.pop(request_id)
        if shard.is_first_layer():
          return loss
        else:
          return loss, grad
      else:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "evaluating"
          loss = await self.inference_engine.evaluate(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
        self.outstanding_requests.pop(request_id)
        return loss
    except Exception as e:
      self.outstanding_requests.pop(request_id)
      print(f"Error processing example for shard {shard}: {e}")
      traceback.print_exc()
      return None
        
  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    await self._enter_inference()
    try:
      print(f"[DEBUG] process_tensor called: node_id={self.id}, received_shard={base_shard}, tensor_shape={tensor.shape}")

      start_time = time.perf_counter_ns()
      resp = await self._process_tensor(base_shard, tensor, request_id, inference_state)
      end_time = time.perf_counter_ns()
      elapsed_time_ns = end_time - start_time

      await self._broadcast_node_stats(elapsed_time_ns)

      print(f"[DEBUG] process_tensor completed: node_id={self.id}, elapsed_ms={elapsed_time_ns/1e6:.2f}, resp.shape={resp.shape if resp is not None else None}")
      if DEBUG >= 2: print(f"[{request_id}] process_tensor: {base_shard=} {tensor.size=} {tensor.shape=} {elapsed_time_ns=}")
      return resp
    finally:
      await self._exit_inference()
  
  async def _broadcast_node_stats(self, elapsed_time_ns: int):
    """广播本节点的性能统计给所有节点"""
    # 更新本节点统计
    if self.id not in self.node_stats:
      self.node_stats[self.id] = {
        "total_requests": 0,
        "total_tokens": 0,
        "total_time_ms": 0,
        "avg_time_per_request_ms": 0,
        "tokens_per_second": 0,
        "last_updated": time.time()
      }
    stats = self.node_stats[self.id]
    stats["total_requests"] += 1
    stats["total_time_ms"] += elapsed_time_ns / 1e6
    stats["avg_time_per_request_ms"] = stats["total_time_ms"] / stats["total_requests"]
    stats["last_updated"] = time.time()
    
    # 广播给所有节点
    stats_message = json.dumps({
      "type": "node_performance_stats",
      "node_id": self.id,
      "stats": stats
    })
    await self.broadcast_opaque_status("", stats_message)

  async def _process_tensor(
    self,
    shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    # 直接使用传入的 shard ，不需要重新计算

    try:
      self.outstanding_requests[request_id] = "processing"
      
      # 关键修复：获取当前节点的实际分片，而不是使用传入的分片
      # 传入的分片可能是其他节点的分片信息
      my_shard = await self.get_current_shard(shard)
      logging.info(f"[_process_tensor] node_id={self.id}, input_shard={shard}, my_shard={my_shard}")
      
      # 使用节点自己维护的 KV 缓存
      # 每个节点应该维护自己的 past_key_values，不应该在分片间传递
      if inference_state is None:
        inference_state = {}
      
      # 从节点的 KV 缓存中获取当前请求的 past_key_values
      # 注意：每个节点只维护自己的层的 KV 缓存
      if request_id in self.request_kv_cache:
        inference_state['past_key_values'] = self.request_kv_cache[request_id]
        logging.info(f"[process_tensor] [{request_id}] 使用缓存的 past_key_values (缓存类型: {type(self.request_kv_cache[request_id]).__name__})")
      else:
        logging.info(f"[process_tensor] [{request_id}] 创建新的 past_key_values (首次处理此请求)")
        inference_state.pop('past_key_values', None)
      
      # 使用 my_shard 而不是传入的 shard
      # [STAR] 多实例支持：使用对应实例的引擎进行 tensor 推理
      tensor_instance_id = getattr(my_shard, 'instance_id', None) or "default"
      tensor_engine = self.get_engine(tensor_instance_id)
      result, inference_state = await tensor_engine.infer_tensor(request_id, my_shard, tensor, inference_state)
      
      # 保存更新后的 KV 缓存
      if 'past_key_values' in inference_state:
        self.request_kv_cache[request_id] = inference_state['past_key_values']
        logging.info(f"[process_tensor] [{request_id}] 保存 past_key_values 到缓存 (缓存类型: {type(inference_state['past_key_values']).__name__})")
      
      ret = await self.process_inference_result(my_shard, result, request_id, inference_state)
      return ret
    except Exception as e:
      self.outstanding_requests.pop(request_id, None)
      logging.error(f"[_process_tensor] Error processing tensor for shard {shard}: {e}")
      import traceback
      traceback.print_exc()
      return None
  
  async def forward_example(
    self,
    base_shard: Shard,
    step: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    target_shard = await self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")
    target_peer = next((p for p in self.peers if p.id() == target_id), None)
    if not target_peer:
      raise ValueError(f"peer for {target_index} not found")
    if DEBUG >= 1: print(f"sending example to {target_peer.id()}: {step} => {target} ({length})")
    resp = await target_peer.send_example(target_shard, step, target, length, request_id=request_id, train=train)
    return resp

  async def forward_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = await self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
    if target_id == self.id:
      await self.process_prompt(next_shard, prompt, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending prompt to {target_peer.id()}: {prompt}")
      await target_peer.send_prompt(next_shard, prompt, request_id=request_id, inference_state=inference_state)
  
  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    logging.info(f"[forward_tensor] START: target_index={target_index}, tensor.shape={tensor.shape}, request_id={request_id}")
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    
    partitions = self.partitioning_strategy.partition(self.topology)
    logging.info(f"[forward_tensor] partitions={[p.node_id for p in partitions]}, target_index={target_index}")
    
    target_id = partitions[target_index].node_id
    logging.info(f"[forward_tensor] target_id={target_id}, self.id={self.id}")
    
    next_shard = await self.get_current_shard(base_shard, target_index)
    logging.info(f"[forward_tensor] next_shard={next_shard}")
    
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {next_shard}")
    if target_id == self.id:
      logging.info(f"[forward_tensor] Processing locally (target_id == self.id)")
      result = await self.process_tensor(next_shard, tensor, request_id, inference_state)
      if result is not None:
        logging.info(f"[forward_tensor] Received result from self: shape={result.shape}")
        return result
    else:
      logging.info(f"[forward_tensor] Looking for peer: target_id={target_id}, available_peers={[p.id() for p in self.peers]}")
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        logging.error(f"[forward_tensor] Peer not found! target_id={target_id}, available_peers={[p.id() for p in self.peers]}")
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: {tensor}")
      logging.info(f"[forward_tensor] Sending tensor to peer: {target_peer.id()}, shape={tensor.shape}, nbytes={tensor.nbytes}")
      result = await target_peer.send_tensor(next_shard, tensor, request_id=request_id, inference_state=inference_state)
      # 处理返回的结果（如果是token，需要继续生成）
      if result is not None:
        logging.info(f"[forward_tensor] Received result from {target_peer.id()}: shape={result.shape}")
        # 将结果返回给调用者（process_inference_result）
        # 注意：这里不需要继续转发，因为process_inference_result会处理返回的token
        # 并决定是否继续生成
        return result

  def get_partition_index(self, offset: int = 0):
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return None
    partitions = self.partitioning_strategy.partition(self.topology)
    
    # 调试信息
    logging.info(f"[get_partition_index] node_id={self.id}, partitions={[p.node_id for p in partitions]}")
    
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    
    logging.info(f"[get_partition_index] current_partition_index={current_partition_index}, offset={offset}, result={(current_partition_index + offset) % len(partitions)}")
    
    return (current_partition_index + offset) % len(partitions)

  async def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    # [OK] 支持多实例模型匹配
    if hasattr(self, 'my_loaded_models'):
      # 直接匹配完整 model_id（如 "qwen-3-0.6b::worker-1"）
      if base_shard.model_id in self.my_loaded_models:
        loaded_shard = self.my_loaded_models[base_shard.model_id].shard
        pass  # logging.info(f"[get_current_shard] [OK] 使用已加载分片 (精确匹配): {loaded_shard} (不重新计算)")
        return loaded_shard

      # 回退：尝试通过 base_model_id + instance_id 匹配
      if hasattr(base_shard, 'instance_id') and base_shard.instance_id and base_shard.instance_id != "default":
        base_model_id = base_shard.base_model_id if hasattr(base_shard, 'base_model_id') else base_shard.model_id.split("::")[0]
        full_id = f"{base_model_id}::{base_shard.instance_id}"
        if full_id in self.my_loaded_models:
          loaded_shard = self.my_loaded_models[full_id].shard
          pass  # logging.info(f"[get_current_shard] [OK] 使用已加载分片 (实例匹配): {loaded_shard} (不重新计算)")
          return loaded_shard

      # 最后回退：只使用基础 model_id 匹配
      base_model_id = base_shard.base_model_id if hasattr(base_shard, 'base_model_id') else base_shard.model_id.split("::")[0]
      if base_model_id in self.my_loaded_models:
        loaded_shard = self.my_loaded_models[base_model_id].shard
        pass  # logging.info(f"[get_current_shard] [WARN] 使用已加载分片 (基础ID回退): {loaded_shard}")
        return loaded_shard
      
      logging.warning(f"[get_current_shard] [FAIL] 模型未加载: {base_shard.model_id} (instance={getattr(base_shard, 'instance_id', 'N/A')})")
      logging.info(f"[get_current_shard] 已加载模型列表: {list(self.my_loaded_models.keys())}")

    topology_node_count = len(self.topology.nodes) if self.topology and hasattr(self.topology, 'nodes') else 0
    
    configured_shard = None
    if hasattr(self.discovery, 'get_current_node_shard'):
      logging.info(f"[get_current_shard] calling get_current_node_shard with model_id={base_shard.model_id}")
      result = self.discovery.get_current_node_shard(model_id=base_shard.model_id)
      logging.info(f"[get_current_shard] result type={type(result)}, result={result}")
      if result is not None and asyncio.iscoroutine(result):
        configured_shard = await result
      else:
        configured_shard = result

    if configured_shard is not None and topology_node_count <= 1:
      logging.info(f"[get_current_shard] node_id={self.id}, 单节点模式，使用配置分片: {configured_shard}")
      return Shard(
        model_id=base_shard.model_id,
        start_layer=configured_shard.start_layer,
        end_layer=configured_shard.end_layer,
        n_layers=base_shard.n_layers,
        repo_id=configured_shard.repo_id,
        instance_id=getattr(base_shard, 'instance_id', None)  # [OK] 传递实例ID
      )
    
    if configured_shard is not None and topology_node_count > 1:
      logging.info(f"[get_current_shard] node_id={self.id}, 检测到{topology_node_count}个节点，忽略固定配置分片，使用动态分区")
    
    if index is None:
      index = self.get_partition_index()
    partitions = self.partitioning_strategy.partition(self.topology)
    
    logging.info(f"[get_current_shard] topology.nodes={list(self.topology.nodes.keys())}")
    logging.info(f"[get_current_shard] partitions count={len(partitions)}")
    for i, p in enumerate(partitions):
      logging.info(f"[get_current_shard] partition[{i}]: node_id={p.node_id}, start={p.start}, end={p.end}")
    
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id,
                                       instance_id=getattr(base_shard, 'instance_id', None) or "default")
    
    logging.info(f"[get_current_shard] node_id={self.id}, index={index}, num_shards={len(shards)}")
    for i, s in enumerate(shards):
      logging.info(f"[get_current_shard] shard[{i}]: start={s.start_layer}, end={s.end_layer}")
    logging.info(f"[get_current_shard] returning shard[{index}]: start={shards[index].start_layer}, end={shards[index].end_layer}")
    
    if len(shards) == 1 and shards[0].start_layer == 0 and shards[0].end_layer == base_shard.n_layers - 1:
      other_nodes_with_shard = [
        node_id for node_id, shard in self.node_shards.items()
        if node_id != self.id and shard.model_id == base_shard.model_id
      ]
      other_nodes_with_model = [
        node_id for node_id, models in self.node_loaded_models.items()
        if node_id != self.id and base_shard.model_id in models
      ]
      if other_nodes_with_shard:
        print(f"[Node] 检测到分布式推理环境，其他节点已有分片: {other_nodes_with_shard}")
        print(f"[Node] 拓扑可能暂时不完整，但分片信息已同步，继续推理")
      elif other_nodes_with_model:
        print(f"[Node] 检测到分布式推理环境，其他节点已加载模型: {other_nodes_with_model}")
        print(f"[Node] 当前分区数=1，但期望分布式推理，等待远程节点恢复...")
        raise RuntimeError(f"Waiting for distributed inference peers to recover: {other_nodes_with_model}")
    
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    logging.info(f"[update_peers] Starting update_peers, current peers: {[p.id() for p in self.peers]}")
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    logging.info(f"[update_peers] discover_peers returned: {[p.id() for p in next_peers]}")

    if self.manager_url:
      try:
        import aiohttp
        from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
        from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops, UNKNOWN_DEVICE_CAPABILITIES

        async with aiohttp.ClientSession() as session:
          async with session.get(
            f"{self.manager_url.rstrip('/')}/api/nodes",
            timeout=aiohttp.ClientTimeout(total=5)
          ) as nodes_resp:
            if nodes_resp.status == 200:
              nodes_data = await nodes_resp.json()
              online_nodes = [n for n in nodes_data.get("data", []) if n.get("status") == "online" and n.get("node_id") != self.id]
              existing_peer_ids = {p.id() for p in next_peers}
              for node_info in online_nodes:
                peer_id = node_info.get("node_id")
                if peer_id and peer_id not in existing_peer_ids:
                  # 🔑 关键优化：FRP 模式下使用 P2P 地址而非 Manager 原始地址
                  raw_addr = f"{node_info.get('address', '127.0.0.1')}:{node_info.get('port', 50051)}"

                  # 检查 discovery 是否支持 FRP P2P 地址转换
                  if hasattr(self.discovery, 'get_frp_p2p_address'):
                    addr = self.discovery.get_frp_p2p_address(peer_id, raw_addr)
                    logging.info(f"[update_peers] Address conversion via FRP: {raw_addr} -> {addr}")
                  else:
                    addr = raw_addr

                  dev_info = node_info.get("device_info", {})
                  device_caps = UNKNOWN_DEVICE_CAPABILITIES
                  if dev_info:
                    flops_data = dev_info.get("flops", {})
                    device_caps = DeviceCapabilities(
                      model=dev_info.get("model", "unknown"),
                      chip=dev_info.get("chip", "unknown"),
                      memory=dev_info.get("memory", 0),
                      flops=DeviceFlops(
                        fp32=flops_data.get("fp32", 0),
                        fp16=flops_data.get("fp16", 0),
                        int8=flops_data.get("int8", 0)
                      ),
                    )
                  new_peer = GRPCPeerHandle(peer_id, addr, "manager-discovery", device_caps)
                  next_peers.append(new_peer)
                  logging.info(f"[update_peers] Manager-assisted discovery: added {peer_id}@{addr}")
      except Exception as e:
        logging.warning(f"[update_peers] Failed to fetch peers from Manager: {e}")

    logging.info(f"[update_peers] Final peers (UDP+Manager): {[p.id() for p in next_peers]}")
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_to_disconnect = [peer for peer in peers_removed if await peer.is_connected()]
    peers_to_connect = [peer for peer in peers_added + peers_updated + peers_unchanged if not await peer.is_connected()]

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{peer.id()}@{peer.addr()}" for peer in peers]

    if DEBUG >= 2:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_disconnect={peers_to_disconnect} to_connect={peers_to_connect}")

    async def disconnect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(peer, timeout=35):
      try:
        await asyncio.wait_for(peer.connect(), timeout)
        return True
      except Exception as e:
        print(f"Error connecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    disconnect_results = await asyncio.gather(*(disconnect_with_timeout(peer) for peer in peers_to_disconnect), return_exceptions=True)
    
    if self.auto_connect:
      connect_results = await asyncio.gather(*(connect_with_timeout(peer) for peer in peers_to_connect), return_exceptions=True)

      successful_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is True]
      failed_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is False]
      successful_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is True]
      failed_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is False]
      
      # [STAR] 自动回退：当直接 gRPC 连接失败时，尝试通过 Manager 中继
      relayed_peers = []
      failed_relay_peer_ids = set()
      if failed_connects and self.manager_url:
        if DEBUG >= 1:
          print(f"[update_peers] 🔀 尝试通过 Manager 中继 {len(failed_connects)} 个失败连接...")
        
        try:
          from exo.networking.grpc.manager_relay_peer_handle import ManagerRelayPeerHandle
          
          for peer in failed_connects:
            try:
              relay_peer = ManagerRelayPeerHandle(
                _id=peer.id(),
                manager_url=self.manager_url,
                desc=f"relay-{peer.description()}",
                device_capabilities=peer.device_capabilities(),
                source_node_id=self.id,
                timeout=120.0
              )
              
              # 尝试连接中继
              await asyncio.wait_for(relay_peer.connect(), timeout=10.0)
              
              # 替换原 peer 为中继 peer
              next_peers = [relay_peer if p.id() == peer.id() else p for p in next_peers]
              relayed_peers.append(relay_peer)
              
              if DEBUG >= 1:
                print(f"[OK] [Relay] 成功建立中继连接: {self.id} → Manager → {peer.id()}")
                
            except Exception as relay_error:
              if DEBUG >= 1:
                print(f"[FAIL] [Relay] 中继失败 {peer.id()}: {relay_error}")
              # 标记为完全无法连接，后续从 peers 列表中移除
              failed_relay_peer_ids.add(peer.id())
                
        except ImportError as e:
          if DEBUG >= 1:
            print(f"[WARN] [Relay] ManagerRelayPeerHandle 模块导入失败: {e}")
          failed_relay_peer_ids.update(p.id() for p in failed_connects)
        except Exception as e:
          if DEBUG >= 1:
            print(f"[WARN] [Relay] 自动回退失败: {e}")
          failed_relay_peer_ids.update(p.id() for p in failed_connects)
      
      # [STAR] 移除完全无法连接的 peers（直连失败 + 中继失败）
      if failed_relay_peer_ids:
        original_count = len(next_peers)
        next_peers = [p for p in next_peers if p.id() not in failed_relay_peer_ids]
        removed_count = original_count - len(next_peers)
        if DEBUG >= 1:
          print(f"[update_peers] 🗑️ 移除无法连接的 peers ({removed_count}): {failed_relay_peer_ids}")
          print(f"[update_peers]    这些节点将在下次 discovery 时重新尝试连接")
      
      if DEBUG >= 1:
        if successful_disconnects: print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
        if failed_disconnects: print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
        if successful_connects: print(f"Successfully connected peers: {_pretty(successful_connects)}")
        if failed_connects and not relayed_peers: 
          print(f"Failed to connect peers (direct): {_pretty(failed_connects)}")
        if relayed_peers:
          print(f"[OK] Relayed via Manager: {_pretty(relayed_peers)}")
    else:
      if DEBUG >= 1 or peers_to_connect:
        print(f"[update_peers] auto_connect=False, skipping auto-connect for {len(peers_to_connect)} peers: {_pretty(peers_to_connect)}")

    self.peers = next_peers
    
    logging.info(f"[update_peers] Updated peers list: {[p.id() for p in self.peers]}")
    
    if peers_removed:
      for peer in peers_removed:
        if peer.id() in self.node_loaded_models:
          print(f"[Node] 清理断开节点的已加载模型信息: {peer.id()}")
          del self.node_loaded_models[peer.id()]
    
    if peers_added:
      if hasattr(self.discovery, 'get_current_node_shard'):
        shard = self.discovery.get_current_node_shard()
        if shard:
          print(f"[Node] New peers connected: {[p.id() for p in peers_added]}, broadcasting shard config")
          await self._broadcast_shard_config()
      
      # 广播自己已加载的模型给新节点
      if self.my_loaded_models:
        print(f"[Node] New peers connected: {[p.id() for p in peers_added]}, broadcasting loaded models")
        await self._broadcast_loaded_models()
      
      # 请求新节点的已加载模型信息
      print(f"[Node] Requesting loaded models from new peers: {[p.id() for p in peers_added]}")
      await self._request_loaded_models_from_peers()
    
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def _update_peers_and_broadcast(self):
    """更新peers并广播自己的信息（用于被动接收连接时）"""
    try:
      await self.update_peers()
      if self.my_loaded_models:
        await self._broadcast_loaded_models()
    except Exception as e:
      print(f"[Node] Error in _update_peers_and_broadcast: {e}")

  async def select_best_inference_engine(self):
    if self.inference_engine.__class__.__name__ == 'DummyInferenceEngine': return
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)

    # 检查是否有支持的引擎
    if not supported_engines:
      print(f"[Node] 没有检测到支持的推理引擎，保持当前引擎: {self.inference_engine.__class__.__name__}")
      return

    topology_engines = self.get_topology_inference_engines()
    if len(topology_engines) > 0:
      best_engine = supported_engines[0]

      # 如果当前引擎已经是最佳引擎，不需要切换
      current_engine_name = self.inference_engine.__class__.__name__.replace('InferenceEngine', '').lower()
      if current_engine_name == best_engine.lower():
        if DEBUG >= 2: print(f"[Node] 当前引擎 {best_engine} 已经是最佳引擎，无需切换")
        return

      # 在替换推理引擎前，先清理旧推理引擎的资源
      if hasattr(self.inference_engine, 'cleanup'):
        try:
          await self.inference_engine.cleanup()
          print(f"已清理旧推理引擎资源: {self.inference_engine.__class__.__name__}")
        except Exception as e:
          print(f"清理旧推理引擎资源时出错: {e}")

      self.inference_engine = get_inference_engine(best_engine, self.shard_downloader)
      print(f"[Node] 已切换到推理引擎: {best_engine}")
 
  async def periodic_topology_collection(self, interval: float):
    """
    周期性拓扑收集（优化版）

    优化点：
    1. 推理期间自动跳过更新，避免阻塞gRPC请求
    2. 使用后台任务执行，不阻塞事件循环
    3. 增加间隔时间，减少CPU占用
    """
    skip_count = 0
    while True:
      await asyncio.sleep(interval)

      # 关键优化：如果正在推理，跳过本次更新
      if self.is_inferencing:
        skip_count += 1
        logging.info(f"[TopologyUpdate] [SKIP] 跳过拓扑更新（推理进行中）- 已连续跳过{skip_count}次")
        continue

      # 重置跳过计数器
      if skip_count > 0:
        logging.info(f"[TopologyUpdate] [OK] 恢复拓扑更新（之前跳过了{skip_count}次）")
        skip_count = 0

      try:
        # 完全异步执行：不await，让拓扑更新在后台运行
        self._topology_update_task = asyncio.create_task(self._background_topology_update())
        # 不阻塞等待，让 gRPC 请求可以立即响应
      except Exception as e:
        print(f"Error collecting topology: {e}")
        traceback.print_exc()

  async def _background_topology_update(self):
    """后台执行拓扑更新的具体逻辑（支持中断）"""
    try:
      # 在每个步骤前检查是否正在推理，如果是则放弃本次更新
      did_peers_change = await self.update_peers()
      
      # 检查是否被推理中断
      if self.is_inferencing:
        logging.info("[TopologyUpdate] [WARN] 推理开始，中断拓扑更新（阶段1完成）")
        return
      
      if DEBUG >= 2: print(f"{did_peers_change=}")
      await self.collect_topology(set())
      
      # 再次检查
      if self.is_inferencing:
        logging.info("[TopologyUpdate] [WARN] 推理开始，中断拓扑更新（阶段2完成）")
        return
      
      if did_peers_change:
        await self.select_best_inference_engine()
        
      logging.info("[TopologyUpdate] [OK] 拓扑更新完成")
    except asyncio.CancelledError:
      logging.info("[TopologyUpdate] [CANCEL] 拓扑更新被取消")
      raise
    except Exception as e:
      print(f"Error in background topology update: {e}")
      traceback.print_exc()

  async def collect_topology(self, visited: set[str], max_depth: int = 4) -> Topology:
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    logging.info(f"[collect_topology] Starting collection, self.peers={[p.id() for p in self.peers]}, max_depth={max_depth}")
    if DEBUG >= 2: print(f"Collecting topology {max_depth=} {visited=}")

    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    for peer in self.peers:
      logging.info(f"[collect_topology] Processing peer: {peer.id()}")
      next_topology.update_node(peer.id(), peer.device_capabilities())
      next_topology.add_edge(self.id, peer.id(), peer.description())
      logging.info(f"[collect_topology] Added peer {peer.id()} to next_topology, nodes now: {list(next_topology.nodes.keys())}")

      if peer.id() in prev_visited:
        continue

      if max_depth <= 0:
        if DEBUG >= 2: print("Max depth reached. Skipping...")
        continue

      try:
        my_addr_info = None
        if hasattr(self.discovery, 'get_my_address_info'):
          my_addr_info = self.discovery.get_my_address_info()
        
        if my_addr_info:
          caller_info = json.dumps({
            "type": "caller_register",
            "node_id": self.id,
            "address": my_addr_info.get("address"),
            "port": my_addr_info.get("port"),
            "device_capabilities": self.device_capabilities.to_dict()
          })
          try:
            await asyncio.wait_for(peer.send_opaque_status("", caller_info), timeout=5.0)
          except Exception as e:
            if DEBUG >= 2: print(f"Failed to send caller info to {peer.id()}: {e}")
        
        other_topology = await asyncio.wait_for(peer.collect_topology(visited, max_depth=max_depth - 1), timeout=60.0)
        if DEBUG >= 2: print(f"Collected topology from: {peer.id()}: {other_topology}")

        # 检测并同步真实的 Node ID（解决 Tailscale 设备名与 exo Node ID 不一致的问题）
        if peer.id() not in other_topology.nodes:
          # 对方返回的拓扑中不包含当前 peer_id，说明对方使用了不同的 ID
          # 尝试找到对方真实的 Node ID（通过查找 self 节点）
          for node_id in other_topology.nodes:
            if node_id != self.id and node_id not in next_topology.nodes:
              # 发现了一个未知的节点 ID，可能是对方的真实 ID
              old_peer_id = peer.id()
              print(f"[collect_topology] 🔗 Syncing peer ID: {old_peer_id} → {node_id} (from remote topology)")
              # 更新 peer 的内部 ID
              if hasattr(peer, '_id'):
                peer._id = node_id
              # 将旧 ID 的数据迁移到新 ID
              if old_peer_id in next_topology.nodes:
                cap = next_topology.nodes.pop(old_peer_id)
                next_topology.update_node(node_id, cap)
              # 更新边
              if self.id in next_topology.peer_graph:
                for conn in list(next_topology.peer_graph[self.id]):
                  if conn.to_id == old_peer_id:
                    conn.to_id = node_id
              visited.discard(old_peer_id)
              visited.add(node_id)
              break

        next_topology.merge(peer.id(), other_topology)
      except asyncio.TimeoutError as e:
        print(f"[WARN] Timeout collecting topology from {peer.id()}: {e} (peer may be loading model)")
        print(f"   Peer address: {peer.addr()}")
        print(f"   This is normal during model loading, will retry next cycle")
        if self.topology.nodes.get(peer.id()):
          next_topology.update_node(peer.id(), self.topology.nodes[peer.id()])
          if self.id in self.topology.peer_graph:
            for conn in self.topology.peer_graph[self.id]:
              if conn.to_id == peer.id():
                next_topology.add_edge(self.id, peer.id(), conn.description)
                break
      except Exception as e:
        print(f"[FAIL] Error collecting topology from {peer.id()}: {type(e).__name__}: {e}")
        print(f"   Peer address: {peer.addr()}")
        print(f"   Error details: {str(e)}")
        if DEBUG >= 2: traceback.print_exc()
        if self.topology.nodes.get(peer.id()):
          next_topology.update_node(peer.id(), self.topology.nodes[peer.id()])
          if self.id in self.topology.peer_graph:
            for conn in self.topology.peer_graph[self.id]:
              if conn.to_id == peer.id():
                next_topology.add_edge(self.id, peer.id(), conn.description)
                break

    next_topology.active_node_id = self.topology.active_node_id
    logging.info(f"[collect_topology] Collection complete, next_topology.nodes={list(next_topology.nodes.keys())}, current_topology.nodes={list(self.topology.nodes.keys())}")
    if len(next_topology.nodes) < len(self.topology.nodes):
      print(f"[Node] Topology collection incomplete, keeping previous topology")
      logging.warning(f"[collect_topology] Keeping previous topology (next has {len(next_topology.nodes)} nodes, current has {len(self.topology.nodes)})")
      self.topology.active_node_id = next_topology.active_node_id
    else:
      self.topology = next_topology
      logging.info(f"[collect_topology] Updated topology, nodes={list(self.topology.nodes.keys())}")
    if self.topology_viz:
      self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id)
    return self.topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Triggering all on_token callbacks with {request_id=} {tokens=} {is_finished=}")
    self.on_token.trigger_all(request_id, tokens, is_finished)
  
  async def broadcast_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Broadcasting result: {request_id=} {result=} {is_finished=}")
    async def send_result_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_result(request_id, result, is_finished), timeout=30.0)
      except asyncio.TimeoutError:
        print(f"Timeout broadcasting result to {peer.id()}")
      except Exception as e:
        print(f"Error broadcasting result to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_result_to_peer(peer) for peer in self.peers], return_exceptions=True)

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")
    if DEBUG >= 2: print(f"[DEBUG] Broadcasting status: {request_id=} {status=}")
    if DEBUG >= 2: print(f"[DEBUG] Current peers count: {len(self.peers)}")
    if DEBUG >= 2: print(f"[DEBUG] Will trigger own on_opaque_status: True")

    async def send_status_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_opaque_status(request_id, status), timeout=30.0)
      except asyncio.TimeoutError:
        print(f"Timeout sending opaque status to {peer.id()}")
      except Exception as e:
        print(f"Error sending opaque status to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_status_to_peer(peer) for peer in self.peers], return_exceptions=True)
    # in the case of opaque status, we also want to receive our own opaque statuses
    if DEBUG >= 2: print(f"[DEBUG] Triggering own on_opaque_status with: {request_id=} {status=}")
    self.on_opaque_status.trigger_all(request_id, status)
    if DEBUG >= 2: print(f"[DEBUG] Finished broadcasting opaque status")

  @property
  def current_topology(self) -> Topology:
    return self.topology

  def handle_stable_diffusion(self, inference_state, result):
    if inference_state.get('is_step_finished', False):
      inference_state['step'] = inference_state.get('step', 0) + 1
    progress = [inference_state.get('step', 0), inference_state.get('total_steps', 1)]
    intermediate_result = result
    if progress[0] == progress[1]:
      intermediate_result = result
    return intermediate_result, inference_state

  #添加下载进度处理方法
  def on_download_progress(self, shard_progress_tuple):
    shard, progress_event = shard_progress_tuple
    if DEBUG >= 8: print(f"Download progress update for {shard.model_id}: {progress_event}")
    # 将进度事件存储在 node_download_progress 字典中，使用模型ID作为键
    self.node_download_progress[shard.model_id] = progress_event
    # 使用asyncio.create_task来异步广播进度事件
    asyncio.create_task(self.broadcast_opaque_status("", json.dumps({
      "type": "download_progress",
      "node_id": self.id,
      "progress": progress_event.to_dict()
    })))
  
  # ========== 统一GPU显存池管理接口 ==========
  
  async def pool_load_model(
    self,
    model_id: str,
    model_path: str,
    *,
    nodes: Optional[List[str]] = None,
    strategy: str = "memory_weighted",
    n_layers: Optional[int] = None,
    instance_id: Optional[str] = None,  # [OK] 新增实例ID参数
    force: bool = False
  ) -> Dict:
    """
    通过统一GPU池加载模型（自动分片到所有节点）
    
    这是最主要的使用方式，将模型自动分配到所有可用节点
    
    Args:
        model_id: 模型标识符 (如 "Qwen/Qwen3-4B")
        model_path: 模型文件路径
        nodes: 指定使用的节点列表 (None=自动使用所有节点)
        strategy: 分配策略 ("memory_weighted", "uniform", "performance_weighted")
        n_layers: 手动指定总层数 (None=自动检测)
        instance_id: 实例ID (支持多实例，如 "worker-1", "worker-2")
        force: 强制重新加载
        
    Returns:
        操作结果字典
    """
    if not self.gpu_pool:
      return {"success": False, "error": "GPU池管理器未初始化"}
    
    result = await self.gpu_pool.load(
      model_id=model_id,
      model_path=model_path,
      nodes=nodes,
      strategy=strategy,
      n_layers=n_layers,
      instance_id=instance_id,  # [OK] 传递实例ID
      force=force
    )
    
    if result.get("success"):
      print(f"\n[OK] 模型 {model_id} 已成功加载到GPU池!")
      print(f"   分配详情: {result.get('allocations', {})}")
    else:
      print(f"\n[FAIL] 加载失败: {result.get('error', '未知错误')}")
    
    return result
  
  async def pool_unload_model(self, model_id: str, *, nodes: Optional[List[str]] = None) -> Dict:
    """
    从GPU池卸载模型
    
    Args:
        model_id: 模型ID
        nodes: 从哪些节点卸载 (None=所有节点)
        
    Returns:
        操作结果
    """
    if not self.gpu_pool:
      return {"success": False, "error": "GPU池管理器未初始化"}
    
    result = await self.gpu_pool.unload(model_id, nodes=nodes)
    
    if result.get("success"):
      print(f"\n[OK] 模型 {model_id} 已从GPU池卸载")
    else:
      print(f"\n[FAIL] 卸载失败: {result.get('error', '未知错误')}")
    
    return result
  
  async def pool_rebalance(self, model_id: str) -> Dict:
    """
    重新平衡模型在池中的分布
    
    当添加新节点或节点资源变化时使用
    
    Args:
        model_id: 模型ID
        
    Returns:
        新的分配信息
    """
    if not self.gpu_pool:
      return {"success": False, "error": "GPU池管理器未初始化"}
    
    result = await self.gpu_pool.rebalance(model_id)
    
    if result.get("success"):
      print(f"\n[OK] 模型 {model_id} 重平衡完成!")
      print(f"   新分配: {result.get('new_allocations', {})}")
    else:
      print(f"\n[FAIL] 重平衡失败: {result.get('error', '未知错误')}")
    
    return result
  
  def pool_status(self, *, model_id: Optional[str] = None) -> Dict:
    """
    获取GPU池状态
    
    Args:
        model_id: 特定模型ID (None=显示整个池的状态)
        
    Returns:
        状态字典
    """
    if not self.gpu_pool:
      return {"error": "GPU池管理器未初始化"}
    
    return self.gpu_pool.status(model_id=model_id)
  
  def pool_list_models(self) -> List[Dict]:
    """列出池中所有模型"""
    if not self.gpu_pool:
      return []
    return self.gpu_pool.list_models()
  
  def pool_list_nodes(self) -> List[Dict]:
    """列出池中所有节点及其负载"""
    if not self.gpu_pool:
      return []
    return self.gpu_pool.list_nodes()
  
  def pool_preview(self, model_id: str, n_layers: int, *, strategy: str = "memory_weighted") -> Dict:
    """
    预览模型的分配方案（不实际加载）
    
    Args:
        model_id: 模型ID
        n_layers: 总层数
        strategy: 分配策略
        
    Returns:
        分配预览
    """
    if not self.gpu_pool:
      return {"error": "GPU池管理器未初始化"}
    
    return self.gpu_pool.preview_allocation(model_id, n_layers, strategy=strategy)
  
  async def pool_command(self, command: str) -> Any:
    """
    执行GPU池管理命令（CLI风格）
    
    支持的命令:
        - load <model_id> <path> [--nodes ...] [--strategy ...]
        - unload <model_id>
        - rebalance <model_id>
        - status [--model ...]
        - list-models / ls
        - list-nodes / ln
        - preview <model_id> <n_layers>
        - help
        
    示例:
        await node.pool_command("load qwen3-4b ./models/qwen3-4b")
        await node.pool_command("status")
        await node.pool_command("list-models")
    """
    if not self.gpu_pool:
      return {"error": "GPU池管理器未初始化"}
    
    from exo.topology.gpu_pool_api import GPUBalancerCLI
    cli = GPUBalancerCLI(self.gpu_pool)
    return await cli.run_command(command)
