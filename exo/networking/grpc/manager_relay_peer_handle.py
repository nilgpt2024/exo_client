import asyncio
import aiohttp
import numpy as np
import base64
import json
from typing import Optional, List
from exo.networking.peer_handle import PeerHandle
from exo.inference.shard import Shard
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops, UNKNOWN_DEVICE_CAPABILITIES
from exo.topology.topology import Topology
from exo.helpers import DEBUG


class ManagerRelayPeerHandle(PeerHandle):
    """
    通过 EXO Manager 作为中继的 PeerHandle 实现
    
    当节点无法直接建立 gRPC 连接时（NAT/防火墙），
    使用此类通过 Manager 的 HTTP API 转发所有消息
    
    架构：
    Node A → HTTP POST → Manager → WebSocket → Node B
    """
    
    def __init__(
        self,
        _id: str,
        manager_url: str,
        desc: str = "manager-relay",
        device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
        source_node_id: str = "",
        timeout: float = 120.0
    ):
        self._id = _id
        self.manager_url = manager_url.rstrip('/')
        self.desc = desc
        self._device_capabilities = device_capabilities
        self.source_node_id = source_node_id
        self.timeout = timeout
        self._connected = False
        
    def id(self) -> str:
        return self._id
    
    def addr(self) -> str:
        return f"relay://{self.manager_url}"
    
    def description(self) -> str:
        return self.desc
    
    def device_capabilities(self) -> DeviceCapabilities:
        return self._device_capabilities
    
    async def connect(self) -> None:
        """连接到 Manager（验证目标节点是否在线）"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/relay/{self._id}/health-check"
                payload = {
                    "source_node_id": self.source_node_id
                }
                
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success") and data.get("is_healthy"):
                            self._connected = True
                            if DEBUG >= 1:
                                print(f"[ManagerRelay] ✅ 成功通过 Manager 连接到 {self._id}")
                            return
            
            self._connected = False
            raise ConnectionError(f"无法通过 Manager 连接到节点 {self._id}")
            
        except Exception as e:
            self._connected = False
            if DEBUG >= 2:
                print(f"[ManagerRelay] [ERROR] 连接失败: {e}")
            raise
    
    async def is_connected(self) -> bool:
        """检查是否已连接（轻量级检查）"""
        if not self._connected:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/nodes/{self._id}"
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except:
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """断开连接"""
        self._connected = False
        if DEBUG >= 1:
            print(f"[ManagerRelay] 🔌 断开与 {self._id} 的中继连接")
    
    async def health_check(self) -> bool:
        """健康检查（通过 Manager 中继）"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/relay/{self._id}/health-check"
                payload = {"source_node_id": self.source_node_id}
                
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("is_healthy", False)
                    return False
                    
        except Exception as e:
            if DEBUG >= 3:
                print(f"[ManagerRelay] 健康检查失败: {e}")
            return False
    
    async def send_prompt(
        self,
        shard: Shard,
        prompt: str,
        inference_state: Optional[dict] = None,
        request_id: Optional[str] = None
    ) -> None:
        """发送 Prompt（通过 Manager 中继）"""
        try:
            payload = {
                "source_node_id": self.source_node_id,
                "shard": {
                    "model_id": shard.model_id,
                    "start_layer": shard.start_layer,
                    "end_layer": shard.end_layer,
                    "n_layers": shard.n_layers,
                    "instance_id": getattr(shard, 'instance_id', None) or ""
                },
                "prompt": prompt,
                "request_id": request_id or ""
            }
            
            if inference_state:
                payload["inference_state"] = inference_state
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/relay/{self._id}/prompt"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"Prompt 中继失败 ({resp.status}): {error_text}")
                    
                    if DEBUG >= 2:
                        print(f"[ManagerRelay] Prompt 已中继到 {self._id}")
                        
        except Exception as e:
            if DEBUG >= 1:
                print(f"[ManagerRelay] ❌ Prompt 中继失败: {e}")
            raise
    
    async def send_tensor(
        self,
        shard: Shard,
        tensor: np.ndarray,
        inference_state: Optional[dict] = None,
        request_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        发送张量数据（隐藏状态传递）[STAR] 核心方法
        
        通过 Manager 中继 hidden state tensor
        这是分布式推理的关键操作！
        """
        try:
            tensor_base64 = base64.b64encode(tensor.tobytes()).decode('utf-8')
            
            payload = {
                "source_node_id": self.source_node_id,
                "shard": {
                    "model_id": shard.model_id,
                    "start_layer": shard.start_layer,
                    "end_layer": shard.end_layer,
                    "n_layers": shard.n_layers,
                    "instance_id": getattr(shard, 'instance_id', None) or ""
                },
                "tensor_data": tensor_base64,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "request_id": request_id or ""
            }
            
            if inference_state:
                payload["inference_state"] = inference_state
            
            if DEBUG >= 2:
                print(f"[ManagerRelay] [BOX] 发送张量到 {self._id}: "
                      f"shape={tensor.shape}, dtype={tensor.dtype}, "
                      f"size={tensor.nbytes/1024/1024:.2f}MB")
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/relay/{self._id}/tensor"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"张量中继失败 ({resp.status}): {error_text}")
                    
                    data = await resp.json()
                    
                    if not data.get("success"):
                        error_msg = data.get("error", "Unknown error")
                        raise Exception(f"张量中继返回错误: {error_msg}")
                    
                    # 解析响应张量
                    response_data = data.get("data")
                    if not response_data:
                        if DEBUG >= 2:
                            print(f"[ManagerRelay] 张量中继成功（无响应数据）")
                        return None
                    
                    result_tensor_b64 = response_data.get("tensor_data")
                    result_shape = response_data.get("shape")
                    result_dtype = response_data.get("dtype")
                    
                    if result_tensor_b64 and result_shape and result_dtype:
                        result_bytes = base64.b64decode(result_tensor_b64)
                        result = np.frombuffer(result_bytes, dtype=np.dtype(result_dtype)).reshape(result_shape)
                        
                        if DEBUG >= 2:
                            print(f"[ManagerRelay] ✅ 收到响应张量: shape={result.shape}, dtype={result.dtype}")
                        return result
                    
                    if DEBUG >= 2:
                        print(f"[ManagerRelay] 张量中继成功")
                    return None
                    
        except Exception as e:
            if DEBUG >= 1:
                print(f"[ManagerRelay] ❌ 张量中继失败: {e}")
            raise
    
    async def send_result(
        self,
        request_id: str,
        result: List[int],
        is_finished: bool
    ) -> None:
        """发送结果（通过 Manager 中继）"""
        try:
            payload = {
                "source_node_id": self.source_node_id,
                "request_id": request_id,
                "result": result,
                "is_finished": is_finished
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/relay/{self._id}/result"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 404:
                        # 如果没有专门的 result 端点，忽略（非关键）
                        if DEBUG >= 3:
                            print(f"[ManagerRelay] [WARN] Result 中继端点不存在（可忽略）")
                        return
                    
                    if resp.status != 200:
                        raise Exception(f"Result 中继失败 ({resp.status})")
                        
        except Exception as e:
            if DEBUG >= 2:
                print(f"[ManagerRelay] [WARN] Result 中继失败（非关键）: {e}")
            # 不抛出异常，因为 result 发送不是关键路径

    async def send_opaque_status(
        self,
        request_id: str,
        status: str
    ) -> None:
        """发送 Opaque Status（通过 Manager 中继）"""
        try:
            payload = {
                "source_node_id": self.source_node_id,
                "request_id": request_id,
                "status": status
            }

            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/relay/{self._id}/opaque_status"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 404:
                        if DEBUG >= 3:
                            print(f"[ManagerRelay] [WARN] Opaque status 中继端点不存在（可忽略）")
                        return

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"Opaque status 中继失败 ({resp.status}): {error_text}")

                    if DEBUG >= 2:
                        print(f"[ManagerRelay] Opaque status 已中继到 {self._id}")

        except Exception as e:
            if DEBUG >= 1:
                print(f"[ManagerRelay] ❌ Opaque status 中继失败: {e}")
            raise

    async def collect_topology(
        self,
        visited: set,
        max_depth: int = 4
    ) -> Topology:
        """收集拓扑信息（通过 Manager 中继）"""
        try:
            payload = {
                "source_node_id": self.source_node_id,
                "visited": list(visited),
                "max_depth": max_depth
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.manager_url}/api/relay/{self._id}/topology"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"拓扑收集中继失败 ({resp.status}): {error_text}")
                    
                    data = await resp.json()
                    topology_data = data.get("topology", {})
                    
                    # 解析拓扑数据
                    topology = Topology()
                    
                    nodes_data = topology_data.get("nodes", {})
                    for node_id, node_info in nodes_data.items():
                        from exo.topology.device_capabilities import DeviceMemory
                        
                        caps = node_info.get("device_capabilities", node_info)
                        flops = caps.get("flops", {})
                        
                        device_caps = DeviceCapabilities(
                            model=caps.get("model", "unknown"),
                            chip=caps.get("chip", "unknown"),
                            memory=caps.get("memory", 0),
                            flops=DeviceFlops(
                                fp32=flops.get("fp32", 0),
                                fp16=flops.get("fp16", 0),
                                int8=flops.get("int8", 0)
                            )
                        )
                        
                        topology.update_node(node_id, device_caps)
                    
                    peer_graph = topology_data.get("peer_graph", {})
                    for from_id, connections in peer_graph.items():
                        for conn in connections.get("connections", []):
                            topology.add_edge(from_id, conn.get("to_id"), conn.get("description"))
                    
                    if DEBUG >= 2:
                        print(f"[ManagerRelay] 📊 从 {self._id} 收集到拓扑: {len(topology.nodes)} 节点")
                    
                    return topology
                    
        except Exception as e:
            if DEBUG >= 1:
                print(f"[ManagerRelay] [ERROR] 拓扑收集失败: {e}")
            raise
    
    def __repr__(self):
        return f"ManagerRelayPeerHandle(id={self._id}, manager={self.manager_url})"
