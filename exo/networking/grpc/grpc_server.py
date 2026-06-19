import grpc
from concurrent import futures
import numpy as np
from asyncio import CancelledError

import platform

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import node_service_pb2
import node_service_pb2_grpc
from exo import DEBUG
from exo.inference.shard import Shard
from exo.orchestration import Node
import json

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
  import mlx.core as mx
else:
  import numpy as mx


class GRPCServer(node_service_pb2_grpc.NodeServiceServicer):
  def __init__(self, node: Node, host: str, port: int):
    self.node = node
    self.host = host
    self.port = port
    self.server = None

  async def start(self) -> None:
    # 实现自动端口选择功能，在绑定失败时自动尝试附近端口
    import socket
    import platform
    import time
    
    # 创建gRPC服务器
    self.server = grpc.aio.server(
      futures.ThreadPoolExecutor(max_workers=32),
      options=[
        ("grpc.max_metadata_size", 32*1024*1024),
        ("grpc.max_send_message_length", 256*1024*1024),
        ("grpc.max_receive_message_length", 256*1024*1024),
        ("grpc.keepalive_time_ms", 10000),
        ("grpc.keepalive_timeout_ms", 60000),  # 增加到60秒，适应慢速网络
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ("grpc.max_concurrent_streams", 100),
        ("grpc.tcp_nodelay", 1),
        ("grpc.optimization_target", "throughput"),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_concurrent_streams", 0),  # Unlimited concurrent streams
      ],
      compression=grpc.Compression.Gzip  # 启用 Gzip 压缩支持（与客户端保持一致）
    )
    node_service_pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
    
    # 尝试在配置的端口或附近端口上绑定
    original_port = self.port
    max_port_attempts = 20  # 尝试最多20个连续端口
    bound_port = None
    
    for port_attempt in range(max_port_attempts):
      try:
        current_port = original_port + port_attempt
        listen_addr = f"{self.host}:{current_port}"
        
        # 在Windows系统上使用特殊处理
        if platform.system().lower() == "windows":
          # 先使用socket API设置SO_REUSEADDR并尝试绑定
          sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
          try:
            sock.bind((self.host, current_port))
            sock.close()
            # 给操作系统一点时间释放资源
            time.sleep(0.5)
          except Exception as e:
            sock.close()
            print(f"Pre-binding socket on port {current_port} failed: {e}")
            continue  # 尝试下一个端口
        
        # 尝试用gRPC服务器绑定端口
        bound_port = self.server.add_insecure_port(listen_addr)
        
        if bound_port == current_port:
          # 成功绑定到当前端口
          self.port = current_port  # 更新实际绑定的端口
          print(f"Successfully bound to port {current_port}")
          break
        else:
          print(f"Failed to bind to port {current_port}")
      except Exception as e:
        print(f"Port binding attempt on {current_port} failed: {e}")
        # 短暂延迟后尝试下一个端口
        time.sleep(1)
    
    if bound_port is None or bound_port != self.port:
      raise RuntimeError(f"Failed to bind to any port in range {original_port}-{original_port + max_port_attempts - 1}")
    
    # 启动服务器
    await self.server.start()
    if DEBUG >= 1: print(f"Server started, listening on {self.host}:{self.port}")
    
    # 通知节点端口已更改（如果有必要）
    if hasattr(self.node, 'on_port_bound') and callable(getattr(self.node, 'on_port_bound')):
      self.node.on_port_bound(self.port)

  async def stop(self) -> None:
    if self.server:
      try:
        await self.server.stop(grace=5)
        await self.server.wait_for_termination()
      except CancelledError:
        pass
      if DEBUG >= 1: print("Server stopped and all connections are closed")

  async def SendPrompt(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
      instance_id=request.shard.instance_id or None,
    )
    prompt = request.prompt
    request_id = request.request_id
    inference_state = None if request.inference_state is None else self.deserialize_inference_state(request.inference_state)
    result = await self.node.process_prompt(shard, prompt, request_id, inference_state)
    if DEBUG >= 5: print(f"SendPrompt {shard=} {prompt=} {request_id=} result: {result}")
    tensor_data = result.tobytes() if result is not None else None
    return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype)) if result is not None else node_service_pb2.Tensor()

  async def SendTensor(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
      instance_id=request.shard.instance_id or None,
    )
    print(f"[DEBUG SendTensor] request.tensor.shape: {request.tensor.shape}, dtype: {request.tensor.dtype}, data_len: {len(request.tensor.tensor_data)}")
    tensor = np.frombuffer(request.tensor.tensor_data, dtype=np.dtype(request.tensor.dtype)).reshape(request.tensor.shape)
    print(f"[DEBUG SendTensor] Reshaped tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    request_id = request.request_id

    inference_state = None if request.inference_state is None else self.deserialize_inference_state(request.inference_state)

    result = await self.node.process_tensor(shard, tensor, request_id, inference_state)
    print(f"[DEBUG SendTensor] Processed tensor, result shape={result.shape if result is not None else None}")
    if DEBUG >= 5: print(f"SendTensor tensor {shard=} {tensor=} {request_id=} result: {result}")
    if result is not None:
      tensor_data = result.tobytes()
      shape_list = list(result.shape)
      print(f"[DEBUG SendTensor] Returning tensor: shape={shape_list}, dtype={str(result.dtype)}")
      return node_service_pb2.Tensor(tensor_data=tensor_data, shape=shape_list, dtype=str(result.dtype))
    else:
      return node_service_pb2.Tensor()

  async def SendExample(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
      instance_id=request.shard.instance_id or None,
    )
    example = np.frombuffer(request.example.tensor_data, dtype=np.dtype(request.example.dtype)).reshape(request.example.shape)
    target = np.frombuffer(request.target.tensor_data, dtype=np.dtype(request.target.dtype)).reshape(request.target.shape)
    length = np.frombuffer(request.length.tensor_data, dtype=np.dtype(request.length.dtype)).reshape(request.length.shape)
    train = request.train
    request_id = request.request_id

    if train and not shard.is_first_layer():
      loss, grad = await self.node.process_example(shard, example, target, length, train, request_id)
      tensor_data = grad.tobytes()
      grad_tensor = node_service_pb2.Tensor(tensor_data=tensor_data, shape=grad.shape, dtype=str(grad.dtype))
      return node_service_pb2.Loss(loss=loss, grads=grad_tensor)
    else:
      loss = await self.node.process_example(shard, example, target, length, train, request_id)
      return node_service_pb2.Loss(loss=loss, grads=None)

  async def CollectTopology(self, request, context):
    max_depth = request.max_depth
    visited = set(request.visited)
    topology = self.node.current_topology
    nodes = {}
    
    # 获取实时 GPU 显存和利用率（用于覆盖静态数据）
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
        gpu_utilization = {
          "gpu": util_rates.gpu,
          "memory": util_rates.memory
        }
        if DEBUG >= 1: print(f"[CollectTopology] GPU利用率: compute={util_rates.gpu}%, memory={util_rates.memory}%")
      except:
        pass
      
      if DEBUG >= 1: print(f"[CollectTopology] 实时GPU显存: {realtime_memory['used']}/{realtime_memory['total']} MB")
    except Exception as e:
      if DEBUG >= 1: print(f"[CollectTopology] pynvml 获取显存失败: {e}")
    
    for node_id, cap in topology.nodes.items():
      if hasattr(cap.flops, 'fp32') and hasattr(cap.flops, 'fp16') and hasattr(cap.flops, 'int8'):
        device_flops = node_service_pb2.DeviceFlops(fp32=cap.flops.fp32, fp16=cap.flops.fp16, int8=cap.flops.int8)
      elif isinstance(cap.flops, dict):
        device_flops = node_service_pb2.DeviceFlops(fp32=cap.flops.get('fp32', 0), fp16=cap.flops.get('fp16', 0), int8=cap.flops.get('int8', 0))
      else:
        device_flops = node_service_pb2.DeviceFlops(fp32=0, fp16=0, int8=0)
      
      # 优先使用实时显存数据
      if realtime_memory:
        device_memory = node_service_pb2.DeviceMemory(
          total=realtime_memory["total"],
          free=realtime_memory["free"],
          used=realtime_memory["used"]
        )
      elif hasattr(cap, 'memory_detail') and cap.memory_detail:
        if hasattr(cap.memory_detail, 'total'):
          device_memory = node_service_pb2.DeviceMemory(
            total=cap.memory_detail.total,
            free=cap.memory_detail.free,
            used=cap.memory_detail.used
          )
        elif isinstance(cap.memory_detail, dict):
          device_memory = node_service_pb2.DeviceMemory(
            total=cap.memory_detail.get('total', 0),
            free=cap.memory_detail.get('free', 0),
            used=cap.memory_detail.get('used', 0)
          )
      else:
        device_memory = None
      
      # 构建已加载模型列表
      loaded_models_list = []
      if hasattr(self.node, 'my_loaded_models') and self.node.my_loaded_models:
        for model_id, load_state in self.node.my_loaded_models.items():
          shard = load_state.shard if hasattr(load_state, 'shard') else None
          if shard:
            loaded_models_list.append(node_service_pb2.LoadedModelInfo(
              model_id=model_id,
              start_layer=shard.start_layer,
              end_layer=shard.end_layer,
              n_layers=shard.n_layers
            ))
          else:
            loaded_models_list.append(node_service_pb2.LoadedModelInfo(
              model_id=model_id,
              start_layer=0,
              end_layer=0,
              n_layers=0
            ))
        if DEBUG >= 1: print(f"[CollectTopology] 已加载模型: {[m.model_id for m in loaded_models_list]}")
      
      nodes[node_id] = node_service_pb2.DeviceCapabilities(
        model=cap.model,
        chip=cap.chip,
        memory=cap.memory,
        flops=device_flops,
        memory_detail=device_memory,
        loaded_models=loaded_models_list,  # 新增：已加载模型列表
      )
    
    peer_graph = {
      node_id: node_service_pb2.PeerConnections(connections=[node_service_pb2.PeerConnection(to_id=conn.to_id, description=conn.description) for conn in connections])
      for node_id, connections in topology.peer_graph.items()
    }
    if DEBUG >= 5: print(f"CollectTopology {max_depth=} {visited=} {nodes=} {peer_graph=}")
    return node_service_pb2.Topology(nodes=nodes, peer_graph=peer_graph)

  async def SendResult(self, request, context):
    request_id = request.request_id
    result = request.result
    is_finished = request.is_finished
    img = request.tensor
    if DEBUG >= 5: print(f"Received SendResult request: {request_id=} {result=} {is_finished=}")
    result = list(result)
    if len(img.tensor_data) > 0:
      result = np.frombuffer(img.tensor_data, dtype=np.dtype(img.dtype)).reshape(img.shape)
    self.node.on_token.trigger_all(request_id, result, is_finished)
    return node_service_pb2.Empty()

  async def SendOpaqueStatus(self, request, context):
    request_id = request.request_id
    status = request.status
    if DEBUG >= 8: print(f"Received SendOpaqueStatus request: {request_id=} {status=}")
    self.node.on_opaque_status.trigger_all(request_id, status)
    return node_service_pb2.Empty()

  async def HealthCheck(self, request, context):
    print(f"[Server] 🩺 HealthCheck received from peer")
    print(f"[Server] 📋 Request details: caller_node_id={request.caller_node_id}, caller_address={request.caller_address}")
    response = node_service_pb2.HealthCheckResponse(is_healthy=True)
    print(f"[Server] ✅ Returning healthy=True")
    return response

  def deserialize_inference_state(self, inference_state_proto: node_service_pb2.InferenceState) -> dict:
    inference_state = {}

    for k, tensor_data in inference_state_proto.tensor_data.items():
      np_array = np.frombuffer(tensor_data.tensor_data, dtype=tensor_data.dtype).reshape(tensor_data.shape)
      # 根据平台选择数组类型：macOS ARM64 使用 mlx，其他平台使用 numpy
      if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
        inference_state[k] = mx.array(np_array)
      else:
        # 在非 macOS 平台上保持为 numpy 数组
        inference_state[k] = np_array

    for k, tensor_list in inference_state_proto.tensor_list_data.items():
      if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
        inference_state[k] = [mx.array(np.frombuffer(tensor.tensor_data, dtype=tensor.dtype).reshape(tensor.shape)) for tensor in tensor_list.tensors]
      else:
        # 在非 macOS 平台上保持为 numpy 数组
        inference_state[k] = [np.frombuffer(tensor.tensor_data, dtype=tensor.dtype).reshape(tensor.shape) for tensor in tensor_list.tensors]

    if inference_state_proto.other_data_json:
      other_data = json.loads(inference_state_proto.other_data_json)
      inference_state.update(other_data)

    return inference_state
