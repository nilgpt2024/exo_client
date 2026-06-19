import asyncio
import json
import socket
import time
import traceback
import platform
from typing import List, Dict, Callable, Tuple, Coroutine, Optional

# 全局变量用于控制调试输出
DEBUG = False
DEBUG_DISCOVERY = 2

# Windows错误处理辅助函数
def is_windows_network_error(e: Exception, error_code: str = '1231') -> bool:
    """检查异常是否为特定的Windows网络错误"""
    return (platform.system() == "Windows" and 
            isinstance(e, OSError) and 
            error_code in str(e))
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY, get_all_ip_addresses_and_interfaces, get_interface_priority_and_type


class ListenProtocol(asyncio.DatagramProtocol):
  def __init__(self, on_message: Callable[[bytes, Tuple[str, int]], Coroutine]):
    super().__init__()
    self.on_message = on_message
    self.loop = asyncio.get_event_loop()

  def connection_made(self, transport):
    self.transport = transport

  def datagram_received(self, data, addr):
    asyncio.create_task(self.on_message(data, addr))


def get_broadcast_address(ip_addr: str) -> str:
  try:
    # Split IP into octets and create broadcast address for the subnet
    ip_parts = ip_addr.split('.')
    return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.255"
  except:
    return "255.255.255.255"


class BroadcastProtocol(asyncio.DatagramProtocol):
  def __init__(self, message: str, broadcast_port: int, source_ip: str):
    self.message = message
    self.broadcast_port = broadcast_port
    self.source_ip = source_ip

  def connection_made(self, transport):
    sock = transport.get_extra_info("socket")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # Try both subnet-specific and global broadcast
    broadcast_addr = get_broadcast_address(self.source_ip)
    transport.sendto(self.message.encode("utf-8"), (broadcast_addr, self.broadcast_port))
    if broadcast_addr != "255.255.255.255":
      transport.sendto(self.message.encode("utf-8"), ("255.255.255.255", self.broadcast_port))


class UDPDiscovery(Discovery):
  def __init__(
    self,
    node_id: str,
    node_port: int,
    listen_port: int,
    broadcast_port: int,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
    broadcast_interval: int = 2.5,
    discovery_timeout: int = 30,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
    allowed_node_ids: Optional[List[str]] = None,
    allowed_interface_types: Optional[List[str]] = None,
  ):
    self.node_id = node_id
    self.node_port = node_port
    self.listen_port = listen_port
    self.broadcast_port = broadcast_port
    self.create_peer_handle = create_peer_handle
    self.broadcast_interval = broadcast_interval
    self.discovery_timeout = discovery_timeout
    self.device_capabilities = device_capabilities
    self.allowed_node_ids = allowed_node_ids
    self.allowed_interface_types = allowed_interface_types
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float, int]] = {}
    self.broadcast_task = None
    self.listen_task = None
    self.cleanup_task = None

  async def start(self):
    self.device_capabilities = await device_capabilities()
    self.broadcast_task = asyncio.create_task(self.task_broadcast_presence())
    self.listen_task = asyncio.create_task(self.task_listen_for_peers())
    self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

  async def stop(self):
    if self.broadcast_task: self.broadcast_task.cancel()
    if self.listen_task: self.listen_task.cancel()
    if self.cleanup_task: self.cleanup_task.cancel()
    if self.broadcast_task or self.listen_task or self.cleanup_task:
      await asyncio.gather(self.broadcast_task, self.listen_task, self.cleanup_task, return_exceptions=True)

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2: print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    return [peer_handle for peer_handle, _, _, _ in self.known_peers.values()]

  async def task_broadcast_presence(self):
    """广播当前节点的存在信息到网络上的所有接口"""
    while True:
      try:
        # 获取所有网络接口和IP地址，并为每个接口创建单独的异步任务
        interfaces = get_all_ip_addresses_and_interfaces()
        interface_tasks = []
        
        for addr, interface_name in interfaces:
          # 跳过回环地址以避免不必要的错误
          if addr == '127.0.0.1' or addr == 'localhost':
            continue
          
          # 为每个有效接口创建一个单独的任务，并确保异常被捕获
          task = asyncio.create_task(
            self._safe_broadcast_to_interface(addr, interface_name)
          )
          interface_tasks.append(task)
        
        # 并发执行所有接口的广播任务，并确保返回所有异常
        if interface_tasks:
          results = await asyncio.gather(*interface_tasks, return_exceptions=True)
          
          # 处理每个任务的结果，记录重要错误
          for i, result in enumerate(results):
            if isinstance(result, Exception) and DEBUG_DISCOVERY >= 1:
              addr, interface_name = interfaces[i]
              print(f"接口广播任务出错 ({interface_name} - {addr}): {type(result).__name__}: {result}")
              
      except Exception as e:
        # 捕获并记录整个广播循环中的错误，但不中断任务
        if DEBUG_DISCOVERY >= 1:
          print(f"广播循环中发生未预期的错误: {type(e).__name__}: {e}")
          traceback.print_exc()
      finally:
        try:
          # 无论如何都要确保睡眠时间，防止在Windows上出现CPU占用过高
          await asyncio.sleep(self.broadcast_interval)
        except asyncio.CancelledError:
          # 处理任务取消
          raise
        except Exception as e:
          if DEBUG_DISCOVERY >= 2:
            print(f"睡眠期间出错: {type(e).__name__}: {e}")
            
  async def _safe_broadcast_to_interface(self, addr: str, interface_name: str):
    """安全地向指定接口发送广播消息，处理所有可能的异常"""
    try:
      # 获取接口优先级和类型
      interface_priority, interface_type = await get_interface_priority_and_type(interface_name)
      
      # 构建广播消息
      message = json.dumps({
        "type": "discovery",
        "node_id": self.node_id,
        "grpc_port": self.node_port,
        "device_capabilities": self.device_capabilities.to_dict(),
        "priority": interface_priority,
        "interface_name": interface_name,
        "interface_type": interface_type,
      })
      
      # 创建并配置套接字
      sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
      except AttributeError:
        pass  # 某些平台可能不支持
      
      # 在Windows上，需要特别处理绑定操作，忽略可能的网络位置错误
      try:
        sock.bind((addr, 0))
      except OSError as e:
        if is_windows_network_error(e):
          if DEBUG_DISCOVERY >= 1:
            print(f"跳过不可访问的网络接口 {interface_name} ({addr}): {e}")
          return
        raise  # 重新抛出非Windows特定错误
      
      # 使用异步事件循环创建数据报端点
      try:
        # 捕获并处理异步创建端点时的Windows网络错误
        loop = asyncio.get_running_loop()
        transport = await self._async_create_datagram_endpoint_safe(
          loop, lambda: BroadcastProtocol(message, self.broadcast_port, addr), sock=sock,
          interface_name=interface_name, addr=addr
        )
        
        if transport:
          try: 
            # 给传输一点时间完成发送，然后关闭
            await asyncio.sleep(0.1)
            transport.close()
          except Exception as e:
            if DEBUG_DISCOVERY >= 2:
              print(f"关闭传输时出错 ({interface_name} - {addr}): {type(e).__name__}: {e}")
      except Exception as e:
        if DEBUG_DISCOVERY >= 2:
          print(f"广播操作失败 ({interface_name} - {addr} - {interface_priority}): {type(e).__name__}: {e}")
    except Exception as e:
      if DEBUG_DISCOVERY >= 2:
        print(f"处理接口 {interface_name} ({addr}) 时出错: {type(e).__name__}: {e}")
    
  async def _async_create_datagram_endpoint_safe(self, loop, protocol_factory, sock=None, interface_name='', addr=''):
    """安全地创建异步数据报端点，捕获Windows特定的网络错误"""
    try:
      transport, protocol = await loop.create_datagram_endpoint(
        protocol_factory, sock=sock
      )
      return transport
    except OSError as e:
      # 捕获异步操作中的Windows网络位置错误
      if is_windows_network_error(e):
        if DEBUG_DISCOVERY >= 1:
          print(f"异步广播操作在接口 {interface_name} ({addr}) 上失败: {e}")
        return None
      # 对于其他类型的错误，我们仍然希望记录它们，但不中断执行
      if DEBUG_DISCOVERY >= 1:
        print(f"创建数据报端点失败: {type(e).__name__}: {e}")
      return None
    except Exception as e:
      if DEBUG_DISCOVERY >= 1:
        print(f"创建数据报端点时发生未预期的错误: {type(e).__name__}: {e}")
      return None

  async def on_listen_message(self, data, addr):
    if not data:
      return

    decoded_data = data.decode("utf-8", errors="ignore")

    # Check if the decoded data starts with a valid JSON character
    if not (decoded_data.strip() and decoded_data.strip()[0] in "{["):
      if DEBUG_DISCOVERY >= 2: print(f"Received invalid JSON data from {addr}: {decoded_data[:100]}")
      return

    try:
      decoder = json.JSONDecoder(strict=False)
      message = decoder.decode(decoded_data)
    except json.JSONDecodeError as e:
      if DEBUG_DISCOVERY >= 2: print(f"Error decoding JSON data from {addr}: {e}")
      return

    if DEBUG_DISCOVERY >= 2: print(f"received from peer {addr}: {message}")

    if message["type"] == "discovery" and message["node_id"] != self.node_id:
      peer_id = message["node_id"]
      
      # Skip if peer_id is not in allowed list
      if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
        if DEBUG_DISCOVERY >= 2: print(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list")
        return

      peer_host = addr[0]
      peer_port = message["grpc_port"]
      peer_prio = message["priority"]
      peer_interface_name = message["interface_name"]
      peer_interface_type = message["interface_type"]

      # Skip if interface type is not in allowed list
      if self.allowed_interface_types and peer_interface_type not in self.allowed_interface_types:
        if DEBUG_DISCOVERY >= 2: print(f"Ignoring peer {peer_id} as its interface type {peer_interface_type} is not in the allowed interface types list")
        return

      device_capabilities = DeviceCapabilities(**message["device_capabilities"])

      if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
        if peer_id in self.known_peers:
          existing_peer_prio = self.known_peers[peer_id][3]
          if existing_peer_prio >= peer_prio:
            if DEBUG >= 1:
              print(f"Ignoring peer {peer_id} at {peer_host}:{peer_port} with priority {peer_prio} because we already know about a peer with higher or equal priority: {existing_peer_prio}")
            return
        new_peer_handle = self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", f"{peer_interface_type} ({peer_interface_name})", device_capabilities)
        if not await new_peer_handle.health_check():
          if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Skipping.")
          return
        if DEBUG >= 1: print(f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
        self.known_peers[peer_id] = (new_peer_handle, time.time(), time.time(), peer_prio)
      else:
        if not await self.known_peers[peer_id][0].health_check():
          if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.")
          if peer_id in self.known_peers: del self.known_peers[peer_id]
          return
        if peer_id in self.known_peers: self.known_peers[peer_id] = (self.known_peers[peer_id][0], self.known_peers[peer_id][1], time.time(), peer_prio)

  async def task_listen_for_peers(self):
    """监听网络上的UDP广播消息以发现其他节点"""
    # 定义监听地址列表，首先尝试所有接口，然后回退到回环地址
    listen_addresses = [("0.0.0.0", self.listen_port), ("127.0.0.1", self.listen_port)]
    
    # 创建自定义异常处理的协程包装器
    async def create_listen_endpoint_safe(addr):
      """安全地创建监听端点，捕获和处理特定的Windows网络错误"""
      try:
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
          lambda: ListenProtocol(self.on_listen_message), 
          local_addr=addr
        )
        if DEBUG_DISCOVERY >= 2:
          print(f"成功在地址 {addr} 上启动监听任务")
        # 这个协程会一直运行直到被取消，不会主动返回
        while True:
          await asyncio.sleep(3600)  # 睡眠1小时，让任务保持活跃
      except OSError as e:
        # 捕获并检查Windows网络位置错误
        if is_windows_network_error(e):
          if DEBUG_DISCOVERY >= 1:
            print(f"在地址 {addr} 上监听失败，Windows网络接口错误: {e}")
          # 仅抛出异常让外层代码处理回退，而不是中断整个任务
          raise
        else:
          # 非Windows网络错误，直接抛出
          raise
    
    # 尝试在每个地址上创建监听端点，直到成功为止
    for addr in listen_addresses:
      try:
        await create_listen_endpoint_safe(addr)
        return  # 如果成功，任务会一直运行
      except Exception as e:
        # 只有在最后一个地址尝试失败时才打印详细错误
        is_last_address = addr == listen_addresses[-1]
        if DEBUG_DISCOVERY >= (1 if is_last_address else 2):
          print(f"尝试在地址 {addr} 上监听失败: {type(e).__name__}: {e}")
        
        # 如果不是Windows网络错误，或者已经尝试了所有地址，则重新抛出异常
        if not is_windows_network_error(e) or is_last_address:
          # 即使所有地址都失败，我们仍然不希望任务崩溃，而是继续尝试
          if DEBUG_DISCOVERY >= 1:
            print(f"所有监听地址尝试失败，将在 {self.broadcast_interval} 秒后重试...")
          
          # 等待一段时间后重试
          try:
            await asyncio.sleep(self.broadcast_interval)
          except asyncio.CancelledError:
            raise
          
          # 递归调用自身进行重试
          await self.task_listen_for_peers()
          return
        
        # 继续尝试下一个地址

  async def task_cleanup_peers(self):
    while True:
      try:
        current_time = time.time()
        peers_to_remove = []

        peer_ids = list(self.known_peers.keys())
        results = await asyncio.gather(*[self.check_peer(peer_id, current_time) for peer_id in peer_ids], return_exceptions=True)

        for peer_id, should_remove in zip(peer_ids, results):
          if should_remove: peers_to_remove.append(peer_id)

        if DEBUG_DISCOVERY >= 2:
          print(
            "Peer statuses:", {
              peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, health_check={(await peer_handle.health_check())[0]}, connected_at={connected_at}, last_seen={last_seen}, prio={prio}"
              for peer_handle, connected_at, last_seen, prio in self.known_peers.values()
            }
          )

        for peer_id in peers_to_remove:
          if peer_id in self.known_peers:
            del self.known_peers[peer_id]
            if DEBUG_DISCOVERY >= 2: print(f"Removed peer {peer_id} due to inactivity or failed health check.")
      except Exception as e:
        print(f"Error in cleanup peers: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.broadcast_interval)

  async def check_peer(self, peer_id: str, current_time: float) -> bool:
    peer_handle, connected_at, last_seen, prio = self.known_peers.get(peer_id, (None, None, None, None))
    if peer_handle is None: return False

    try:
      is_connected = await peer_handle.is_connected()
      health_ok = await peer_handle.health_check()
    except Exception as e:
      if DEBUG_DISCOVERY >= 2: print(f"Error checking peer {peer_id}: {e}")
      return True

    should_remove = ((not is_connected and current_time - connected_at > self.discovery_timeout) or (current_time - last_seen > self.discovery_timeout) or (not health_ok))
    return should_remove
