import asyncio
import os
import time
import traceback
from typing import List, Dict, Callable, Tuple, Optional
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY
from .tailscale_helpers import get_tailscale_devices_local, get_tailscale_devices, Device
from .tailscale_auto_config import get_tailscale_env, auto_configure_tailscale

# FRP 集成支持
try:
    from exo.networking.frp.frp_auto_manager import (
        get_frp_manager,
        initialize_frp_auto,
    )
    HAS_FRP_SUPPORT = True
except ImportError:
    HAS_FRP_SUPPORT = False
    if DEBUG >= 2:
        print("[Tailscale] ⚠️ FRP 模块不可用，将仅使用 Tailscale 直连")


class TailscaleDiscovery(Discovery):
  def __init__(
    self,
    node_id: str,
    node_port: int,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
    discovery_interval: int = 5,
    discovery_timeout: int = 30,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
    tailscale_api_key: str = None,
    tailnet: str = None,
    allowed_node_ids: List[str] = None,
    default_peer_port: Optional[int] = None,
  ):
    self.node_id = node_id
    self.node_port = node_port
    self.create_peer_handle = create_peer_handle
    self.discovery_interval = discovery_interval
    self.discovery_timeout = discovery_timeout
    self.device_capabilities = device_capabilities
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
    self.discovery_task = None
    self.cleanup_task = None
    self.tailscale_api_key = tailscale_api_key
    self.tailnet = tailnet
    self.allowed_node_ids = allowed_node_ids
    self.default_peer_port = default_peer_port or node_port
    self.use_local_discovery = not (tailscale_api_key and tailnet)

    # 网络环境信息（由 auto_config 填充）
    self.tailscale_env = None
    self.use_serve_mode = False  # 是否使用 Tailscale Serve 模式
    self.peer_domain_map = {}  # peer_name -> domain_name (Tailscale Serve 域名)

    # FRP 集成相关
    self.frp_manager = None
    self.use_frp_fallback = False  # 是否启用 FRP 回退
    self.frp_initialized = False   # FRP 是否已初始化
    self.direct_connect_failures = 0  # 直连失败次数计数器

  async def start(self):
    """
    启动 Tailscale 发现服务
    自动执行：
      1. 环境检测和自动配置
      2. 选择最佳连接策略（Tailscale / FRP）
      3. 启动设备发现循环
    """
    print("=" * 60)
    print("[Tailscale] 🚀 初始化 Tailscale 自动发现服务 (含 FRP 回退)")
    print("=" * 60)

    # Step 1: 自动检测和配置 Tailscale 网络
    print("\n[Tailscale] Step 1/4: 环境检测与自动配置...")
    await self._auto_configure_network()

    # Step 2: 获取设备能力信息
    print("\n[Tailscale] Step 2/4: 获取本机设备信息...")
    self.device_capabilities = await device_capabilities()

    if self.use_local_discovery:
      if DEBUG >= 1:
        print("[Tailscale] ✅ 使用本地 CLI 自动发现（无需 API Key）")
      print("[Tailscale] ✅ 两台机器只需加入同一 Tailscale 网络即可")
    else:
      if DEBUG >= 1:
        print(f"[Tailscale] 🔑 使用 API 发现模式 (tailnet: {self.tailnet})")

    # Step 3: 初始化 FRP 回退机制（可选但推荐用于容器环境）
    print("\n[Tailscale] Step 3/4: 检查 FRP 回退支持...")
    await self._initialize_frp_fallback()

    # Step 4: 显示连接策略
    print("\n[Tailscale] Step 4/4: 连接策略选择...")
    self._display_connection_strategy()

    # 启动发现和清理任务
    self.discovery_task = asyncio.create_task(self.task_discover_peers())
    self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

    print("\n" + "=" * 60)
    print("[Tailscale] ✅ 服务启动完成，开始发现对端节点...")
    print("=" * 60 + "\n")

  async def _auto_configure_network(self):
    """自动配置 Tailscale 网络"""
    try:
        self.tailscale_env = get_tailscale_env()
        config_success = await auto_configure_tailscale()

        if config_success and self.tailscale_env.configured:
            print("[Tailscale-Env] ✅ 环境配置成功")

            # 根据环境设置连接策略
            strategy = self.tailscale_env.get_connection_strategy()

            if strategy['use_socks5']:
                # 设置环境变量让 gRPC 客户端使用 SOCKS5
                os.environ['USE_TAILSCALE_SOCKS5'] = 'true'
                if strategy['socks5_address']:
                    host, port = strategy['socks5_address'].split(':')
                    os.environ['TAILSCALE_SOCKS5_HOST'] = host
                    os.environ['TAILSCALE_SOCKS5_PORT'] = port
                print(f"[Tailscale-Env] 🔌 已启用 SOCKS5 代理模式")

            elif not strategy['use_direct']:
                # 无法直连且无代理，提示使用 Tailscale Serve
                print("[Tailscale-Env] ⚠️  检测到无法 P2P 直连")
                print("[Tailscale-Env] 💡 将使用 Tailscale Serve 作为备用方案")
                self.use_serve_mode = True

        else:
            print("[Tailscale-Env] ℹ️  使用默认直连模式")

    except Exception as e:
        print(f"[Tailscale-Env] ⚠️  自动配置失败: {e}")
        print("[Tailscale-Env] ℹ️  将使用默认配置继续...")

  def _display_connection_strategy(self):
    """显示当前选择的连接策略"""
    if self.use_socks_proxy_enabled():
        mode = "🔌 SOCKS5 代理模式 (DERP 中继)"
        details = "流量将通过 Tailscale SOCKS5 代理转发"
    elif self.use_serve_mode:
        mode = "🌐 Tailscale Serve 模式"
        details = "使用 Tailscale 分配的域名 + HTTPS 端口"
    else:
        mode = "🔗 直连模式 (P2P)"
        details = "尝试直接建立 TCP 连接"

    print(f"[Tailscale] 📋 当前策略: {mode}")
    print(f"[Tailscale] 📋 详细说明: {details}")

    if DEBUG_DISCOVERY >= 2:
        if self.tailscale_env:
            env_info = self.tailscale_env.get_connection_strategy()
            print(f"[Tailscale] 📊 环境详情: {env_info}")

  def use_socks_proxy_enabled(self) -> bool:
    """检查是否启用了 SOCKS5 代理"""
    return os.environ.get('USE_TAILSCALE_SOCKS5', 'false').lower() == 'true'

  async def _initialize_frp_fallback(self):
    """
    初始化 FRP 回退机制
    
    触发条件：
    - 环境变量 EXO_USE_FRP=true
    - 检测到无法直连且无 SOCKS5 代理
    """
    if not HAS_FRP_SUPPORT:
      print("[Tailscale-FRP] ⚠️  FRP 模块不可用")
      return

    # 检查是否强制启用
    force_frp = os.environ.get('EXO_USE_FRP', 'false').lower() == 'true'

    # 检查是否应该自动启用（当 Tailscale 直连不可用时）
    auto_enable = (
      self.tailscale_env and
      not self.tailscale_env.can_direct_connect and
      not self.use_socks_proxy_enabled()
    )

    if not (force_frp or auto_enable):
      print("[Tailscale-FRP] ℹ️  FRP 未启用（直连可用或未配置）")
      return

    try:
      print("[Tailscale-FRP] 🚀 正在初始化 FRP 回退机制...")

      # 获取 FRP 服务端地址（优先从环境变量）
      frp_server = os.environ.get('EXO_FRP_SERVER_ADDR')

      success = await initialize_frp_auto(
        server_addr=frp_server,
        local_port=self.node_port,
        token=os.environ.get('EXO_FRP_TOKEN'),
        force_mode='client' if frp_server else None,
      )

      if success:
        self.frp_manager = get_frp_manager()
        self.frp_initialized = True
        self.use_frp_fallback = True

        status = self.frp_manager.get_status()
        print(f"[Tailscale-FRP] ✅ FRP 回退已就绪 (XTCP P2P 模式)！")
        print(f"[Tailscale-FRP] 📊 运行模式: {status['mode']}")
        print(f"[Tailscale-FRP] 🔗 连接策略: P2P 直连优先 → TCP 中转备用")
        if status.get('server_addr'):
          print(f"[Tailscale-FRP] 🌐 服务端: {status['server_addr']}:{status['server_port']}")
        print(f"[Tailscale-FRP] 💡 当 Tailscale 直连失败时自动启用")

      else:
        print("[Tailscale-FRP] ⚠️  FRP 初始化失败，将仅使用 Tailscale")

    except Exception as e:
      print(f"[Tailscale-FRP] ❌ FRP 初始化异常: {e}")
      import traceback
      traceback.print_exc()

  async def _discover_devices(self) -> Dict[str, Device]:
    """Discover devices using local CLI (preferred) or API (fallback)"""
    try:
      # Method 1: Local CLI discovery (automatic, no API key)
      if self.use_local_discovery:
        devices, _ = await get_tailscale_devices_local()
        return devices

      # Method 2: API-based discovery (fallback)
      devices = await get_tailscale_devices(self.tailscale_api_key, self.tailnet)
      return devices

    except Exception as e:
      print(f"[Tailscale] ❌ Discovery failed: {e}")
      return {}

  async def task_discover_peers(self):
    while True:
      try:
        devices: dict[str, Device] = await self._discover_devices()

        if not devices:
          if DEBUG_DISCOVERY >= 2:
            print("[Tailscale] No devices found. Waiting for peers to join Tailscale...")
          await asyncio.sleep(self.discovery_interval)
          continue

        current_time = time.time()
        print(f"[Tailscale] 🔄 Discovery cycle: {len(devices)} device(s) found, Self={self.node_id}")

        for device_name, device in devices.items():
          print(f"[Tailscale] 📝 Processing: {device_name} (name={device.name}, addr={device.addresses})")

          # Skip self
          if device_name == self.node_id or device.name == self.node_id:
            print(f"[Tailscale] ⏭️ Skipping self: {device_name} == {self.node_id}")
            continue

          peer_host = device.addresses[0] if device.addresses else None
          if not peer_host:
            print(f"[Tailscale] ❌ No address for {device_name}. Skip.")
            continue

          peer_id = device_name

          # Filter by allowed list (if specified)
          if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
            print(f"[Tailscale] ❌ {peer_id} not in allowed list: {self.allowed_node_ids}")
            continue

          peer_addr = f"{peer_host}:{self.default_peer_port}"
          print(f"[Tailscale] 🔗 Attempting to connect: {peer_id} @ {peer_addr}")

          # New peer or address changed
          if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != peer_addr:
            try:
              new_peer_handle = self.create_peer_handle(peer_id, peer_addr, "TS", UNKNOWN_DEVICE_CAPABILITIES)

              # Health check to verify it's an exo node
              print(f"[Tailscale] 💓 Health check for {peer_id}...")
              is_healthy = await new_peer_handle.health_check()
              print(f"[Tailscale] {'✅' if is_healthy else '❌'} Health check result: {is_healthy}")

              if not is_healthy:
                # 尝试 FRP 回退
                is_healthy = await self._try_frp_fallback(peer_id, peer_host, peer_addr, new_peer_handle)

                if not is_healthy:
                  print(f"[Tailscale] ⚠️ Peer {peer_id} at {peer_addr} is not an exo node (health check failed)")
                  continue

              if DEBUG >= 1:
                print(f"[Tailscale] ✅ Discovered exo node: {peer_id} at {peer_addr}")

              self.known_peers[peer_id] = (
                new_peer_handle,
                current_time,
                current_time,
              )
            except Exception as e:
              print(f"[Tailscale] ❌ Error connecting to {peer_id}: {type(e).__name__}: {e}")
              continue
          else:
            # Existing peer - health check
            try:
              is_healthy = await self.known_peers[peer_id][0].health_check()
              if not is_healthy:
                # 尝试 FRP 回退（重新创建连接）
                frp_peer_id = f"{peer_id}_frp"
                if frp_peer_id not in self.known_peers and self.use_frp_fallback:
                  is_healthy = await self._try_frp_reconnect(peer_id, peer_host)
                  if is_healthy:
                    current_time = time.time()

                if not is_healthy:
                  if DEBUG >= 1:
                    print(f"[Tailscale] ❌ Peer {peer_id} became unhealthy. Removing.")
                  if peer_id in self.known_peers:
                    del self.known_peers[peer_id]
                  continue

              # Update last seen timestamp
              self.known_peers[peer_id] = (
                self.known_peers[peer_id][0],
                self.known_peers[peer_id][1],
                current_time
              )
            except Exception as e:
              print(f"[Tailscale] ❌ Error checking existing peer {peer_id}: {e}")
              continue

        print(f"[Tailscale] ✅ Discovery cycle complete. Known peers: {list(self.known_peers.keys())}")

      except Exception as e:
        print(f"[Tailscale] ❌ Error in discover peers: {e}")
        import traceback
        if DEBUG_DISCOVERY >= 2:
          print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)

  async def _try_frp_fallback(self, peer_id: str, peer_host: str, original_addr: str, original_handle) -> bool:
    """
    尝试使用 FRP 进行健康检查回退
    
    Args:
      peer_id: 节点 ID
      peer_host: 对端主机地址
      original_addr: 原始地址
      original_handle: 原始 PeerHandle
      
    Returns:
      bool: FRP 连接是否成功
    """
    if not self.use_frp_fallback or not self.frp_manager:
      return False

    try:
      print(f"[Tailscale-FRP] 🔄 尝试通过 FRP (XTCP P2P) 连接 {peer_id}...")

      # 获取 FRP 转发地址
      frp_addr = self.frp_manager.get_peer_address(peer_id, f"{peer_host}:{self.default_peer_port}")

      # 创建新的 PeerHandle（使用 FRP 地址）
      frp_peer = self.create_peer_handle(peer_id, frp_addr, "FRP-P2P", UNKNOWN_DEVICE_CAPABILITIES)

      # 健康检查
      is_healthy = await asyncio.wait_for(frp_peer.health_check(), timeout=15.0)

      if is_healthy:
        print(f"[Tailscale-FRP] ✅ 通过 FRP P2P 成功连接 {peer_id} @ {frp_addr}")
        self.direct_connect_failures = 0

        # 更新 known_peers 中的 handle 为 FRP 版本
        current_time = time.time()
        self.known_peers[peer_id] = (frp_peer, current_time, current_time)

        return True
      else:
        print(f"[Tailscale-FRP] ❌ FRP 健康检查失败: {peer_id}")
        return False

    except Exception as e:
      print(f"[Tailscale-FRP] ❌ FRP 连接异常: {e}")
      return False

  async def _try_frp_reconnect(self, peer_id: str, peer_host: str) -> bool:
    """
    当已有节点连接断开时，尝试通过 FRP 重连
    
    Args:
      peer_id: 节点 ID
      peer_host: 对端主机地址
      
    Returns:
      bool: 是否重连成功
    """
    try:
      print(f"[Tailscale-FRP] 🔁 尝试通过 FRP 重连 {peer_id}...")

      frp_addr = self.frp_manager.get_peer_address(peer_id, f"{peer_host}:{self.default_peer_port}")
      frp_peer = self.create_peer_handle(f"{peer_id}_frp", frp_addr, "FRP-Fallback", UNKNOWN_DEVICE_CAPABILITIES)

      is_healthy = await asyncio.wait_for(frp_peer.health_check(), timeout=15.0)

      if is_healthy:
        print(f"[Tailscale-FRP] ✅ FRP 重连成功: {peer_id}")

        # 替换原有的失效节点
        current_time = time.time()
        if peer_id in self.known_peers:
          del self.known_peers[peer_id]
        self.known_peers[f"{peer_id}_frp"] = (frp_peer, current_time, current_time)

        return True

      return False

    except Exception as e:
      print(f"[Tailscale-FRP] ❌ 重连失败: {e}")
      return False

  async def stop(self):
    # 停止 FRP
    if self.frp_manager:
      print("[Tailscale] 正在停止 FRP 回退服务...")
      await self.frp_manager.stop()
      self.frp_manager = None
      print("[Tailscale] FRP 已停止")

    if self.discovery_task:
      self.discovery_task.cancel()
    if self.cleanup_task:
      self.cleanup_task.cancel()
    if self.discovery_task or self.cleanup_task:
      await asyncio.gather(self.discovery_task, self.cleanup_task, return_exceptions=True)

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2:
          print(f"[Tailscale] Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting...")
        await asyncio.sleep(0.1)
    return [peer_handle for peer_handle, _, _ in self.known_peers.values()]

  async def task_cleanup_peers(self):
    while True:
      try:
        current_time = time.time()
        peers_to_remove = []

        peer_ids = list(self.known_peers.keys())
        results = await asyncio.gather(
          *[self.check_peer(peer_id, current_time) for peer_id in peer_ids],
          return_exceptions=True
        )

        for peer_id, should_remove in zip(peer_ids, results):
          if should_remove:
            peers_to_remove.append(peer_id)

        if DEBUG_DISCOVERY >= 2:
          statuses = {}
          for peer_handle, connected_at, last_seen in self.known_peers.values():
            is_conn = await peer_handle.is_connected()
            health = await peer_handle.health_check()
            statuses[peer_handle.id()] = f"connected={is_conn}, healthy={health}"
          print(f"[Tailscale] Peer statuses: {statuses}")

        for peer_id in peers_to_remove:
          if peer_id in self.known_peers:
            del self.known_peers[peer_id]
            if DEBUG_DISCOVERY >= 2:
              print(f"[Tailscale] Removed inactive peer: {peer_id}")

      except Exception as e:
        print(f"[Tailscale] Error in cleanup: {e}")
        if DEBUG_DISCOVERY >= 2:
          print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)

  async def check_peer(self, peer_id: str, current_time: float) -> bool:
    peer_handle, connected_at, last_seen = self.known_peers.get(peer_id, (None, None, None))
    if peer_handle is None:
      return False

    try:
      is_connected = await peer_handle.is_connected()
      health_ok = await peer_handle.health_check()
    except Exception as e:
      if DEBUG_DISCOVERY >= 2:
        print(f"[Tailscale] Error checking peer {peer_id}: {e}")
      return True  # Don't remove on transient errors

    should_remove = (
      (not is_connected and current_time - connected_at > self.discovery_timeout) or
      (not health_ok and current_time - last_seen > self.discovery_timeout)
    )
    return should_remove
