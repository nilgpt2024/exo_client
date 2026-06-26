import os
import asyncio
import json
import hashlib
import logging
import time
from typing import Dict, List, Callable, Optional, Set
from pathlib import Path

from exo.networking.discovery import Discovery
from exo.topology.device_capabilities import DeviceCapabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.networking.peer_handle import PeerHandle
from exo.helpers import DEBUG_DISCOVERY
from exo.networking.frp.frp_downloader import ensure_frpc_installed, get_frpc_path
from exo.networking.frp.frp_config import FRPConfig
from exo.networking.frp.frp_process import FRPProcessManager


def calculate_remote_port(node_id: str) -> int:
    """根据 node_id 计算远程端口（与 frp_config.py 中的逻辑一致）"""
    hash_val = int(hashlib.md5(node_id.encode()).hexdigest()[:8], 16)
    return 30000 + (hash_val % 20000)


# ==================== P2P 对端信息持久化 ====================
# 每个 node 本地保存已发现的 P2P 对端，Manager 下线后仍可直连

# 缓存文件路径：~/.exo/peer_cache/<node_id>.json
_PEER_CACHE_DIR = Path.home() / ".exo" / "peer_cache"
_PEER_CACHE_TTL = 30 * 24 * 3600  # 缓存有效期 30 天


class NodeInfo:
    """节点信息"""
    def __init__(
        self,
        node_id: str,
        address: str,
        port: int,
        description: str = "FRP",
        device_capabilities: Optional[DeviceCapabilities] = None
    ):
        self.node_id = node_id
        self.address = address
        self.port = port
        self.description = description
        self.device_capabilities = device_capabilities or UNKNOWN_DEVICE_CAPABILITIES
    
    def to_dict(self) -> Dict:
        flops_data = {}
        if self.device_capabilities.flops:
            flops_obj = self.device_capabilities.flops
            if hasattr(flops_obj, 'model_dump'):
                flops_data = flops_obj.model_dump()
            elif hasattr(flops_obj, 'to_dict'):
                flops_data = flops_obj.to_dict()
            else:
                flops_data = {"fp32": 0, "fp16": 0, "int8": 0}
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "description": self.description,
            "device_capabilities": {
                "model": self.device_capabilities.model,
                "chip": self.device_capabilities.chip,
                "memory": self.device_capabilities.memory,
                "flops": flops_data
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "NodeInfo":
        dc = data.get("device_capabilities", {})
        return cls(
            node_id=data["node_id"],
            address=data["address"],
            port=data["port"],
            description=data.get("description", "FRP"),
            device_capabilities=DeviceCapabilities(
                model=dc.get("model", "unknown"),
                chip=dc.get("chip", "unknown"),
                memory=dc.get("memory", 0)
            )
        )


class FRPDiscovery(Discovery):
    """基于 frp 的跨公网节点发现 - 支持自动发现和种子节点"""

    def __init__(
        self,
        frp_server_addr: str,
        frp_server_port: int,
        node_id: str,
        local_port: int,
        create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
        frp_token: Optional[str] = None,
        frp_remote_port: Optional[int] = None,
        chatgpt_local_port: int = 52415,
        seed_peers: Optional[str] = None,
        discovery_timeout: int = 30,
        device_capabilities: Optional[DeviceCapabilities] = None,
        enable_p2p: bool = True,  # ✅ 默认启用 XTCP P2P
    ):
        """
        初始化 FRPDiscovery
        
        Args:
            frp_server_addr: frp 服务器地址
            frp_server_port: frp 服务器端口
            node_id: 本节点 ID
            local_port: 本节点本地服务端口
            create_peer_handle: 创建 PeerHandle 的回调函数
            frp_token: frp 认证 token（**必需**，与服务端保持一致）
            frp_remote_port: frp 远程端口（可选，不指定则自动生成）
            chatgpt_local_port: 本节点本地 ChatGPT API HTTP 端口
            seed_peers: 种子节点列表（可选）
            discovery_timeout: 发现超时时间（秒）
            device_capabilities: 本节点的设备能力信息
            enable_p2p: 是否启用 P2P (xtcp) 模式（默认 True）
        """
        self.frp_server_addr = frp_server_addr
        self.frp_server_port = frp_server_port
        self.node_id = node_id
        self.local_port = local_port
        self.chatgpt_local_port = chatgpt_local_port
        self.create_peer_handle = create_peer_handle
        self.frp_token = frp_token or "exo-frp-default-token"  # 🔐 默认 token
        self.frp_remote_port = frp_remote_port
        self.discovery_timeout = discovery_timeout
        self.device_capabilities = device_capabilities or DeviceCapabilities("unknown", "unknown", 0)
        self.enable_p2p = enable_p2p

        self.listen_task = None
        self.known_peers: Dict[str, PeerHandle] = {}
        self.known_node_infos: Dict[str, NodeInfo] = {}

        # 失败重试退避状态：{node_id: {"fail_count": int, "next_retry_at": float}}
        self.node_retry_state: Dict[str, Dict] = {}
        self.retry_base_delay = 5.0      # 基础退避（秒）
        self.retry_max_delay = 300.0     # 最大退避 5 分钟
        self.retry_backoff_factor = 2.0  # 指数退避乘数

        self.frp_config = FRPConfig()
        self.frp_process_manager: Optional[FRPProcessManager] = None
        self.my_remote_port: Optional[int] = None
        self.my_chatgpt_remote_port: Optional[int] = None
        self.my_address: Optional[str] = None

        # 解析种子节点
        self.seed_node_infos: List[NodeInfo] = self._parse_seed_peers(seed_peers)

    def _parse_seed_peers(self, seed_peers: Optional[str]) -> List[NodeInfo]:
        """解析种子节点字符串"""
        if not seed_peers:
            return []
        
        result = []
        for peer_str in seed_peers.split(","):
            peer_str = peer_str.strip()
            if not peer_str:
                continue
            
            try:
                # 格式: node_id@address:port
                if "@" in peer_str:
                    node_id_part, addr_part = peer_str.split("@", 1)
                    node_id = node_id_part.strip()
                else:
                    node_id = f"seed_{len(result)}"
                    addr_part = peer_str
                
                if ":" in addr_part:
                    address, port_str = addr_part.rsplit(":", 1)
                    port = int(port_str)
                else:
                    address = addr_part
                    port = 5678
                
                result.append(NodeInfo(node_id, address, port))
                print(f"[FRP] 添加种子节点: {node_id} @ {address}:{port}")
            except Exception as e:
                print(f"[FRP] 解析种子节点失败: {peer_str}, 错误: {e}")
        
        return result

    async def start(self) -> None:
        """启动 frp 发现"""
        print("=" * 60)
        print("  启动 FRP 发现模块 (XTCP P2P 模式)")
        print("=" * 60)
        print(f"[FRP] 节点 ID: {self.node_id}")
        print(f"[FRP] 服务端: {self.frp_server_addr}:{self.frp_server_port}")

        # 🔐 显示 Token 信息（脱敏 + 长度验证）
        token_display = self.frp_token
        if len(token_display) > 12:
            token_masked = f"{token_display[:6]}...{token_display[-4:]}"
        else:
            token_masked = "***"
        print(f"[FRP] 🔐 Token: {token_masked} (长度: {len(self.frp_token)} 字符)")

        # ⚠️ 警告：如果 Token 太短或包含特殊字符
        if len(self.frp_token) < 16:
            print(f"[FRP] ⚠️ 警告: Token 长度过短 ({len(self.frp_token)} 字符)，建议使用至少 16 字符的强 Token")
        if '$' in self.frp_token or '`' in self.frp_token:
            print(f"[FRP] ⚠️ 警告: Token 包含 PowerShell 特殊字符 ($ 或 `)，请确保使用单引号包裹")

        print(f"[FRP] 🔗 P2P 模式: {'✅ 启用' if self.enable_p2p else '❌ 禁用'}")
        
        # 1. 确保 frpc 已安装
        if not ensure_frpc_installed():
            print("[FRP] 错误: 无法安装或找到 frpc")
            return
        
        # 2. 生成并保存 frpc 配置
        frpc_config = self.frp_config.generate_frpc_config(
            server_addr=self.frp_server_addr,
            server_port=self.frp_server_port,
            node_id=self.node_id,
            local_port=self.local_port,
            remote_port=self.frp_remote_port,
            chatgpt_local_port=self.chatgpt_local_port,
            token=self.frp_token,  # ✅ 确保传递 token
            enable_p2p=self.enable_p2p
        )
        
        # 获取分配的远程端口（从 TCP fallback 代理获取，因为 XTCP 没有 remotePort）
        if frpc_config and "proxies" in frpc_config:
            self.my_address = self.frp_server_addr

            # 遍历所有代理，找到 gRPC 和 ChatGPT 的 remotePort
            for proxy in frpc_config["proxies"]:
                if proxy.get("remotePort"):
                    name = proxy.get("name", "")
                    if "chatgpt" in name:
                        self.my_chatgpt_remote_port = proxy.get("remotePort")
                    else:
                        self.my_remote_port = proxy.get("remotePort")

            # 如果都没找到（纯 P2P 模式），使用自动生成的端口
            if not self.my_remote_port:
                import hashlib
                hash_val = int(hashlib.md5(self.node_id.encode()).hexdigest()[:8], 16)
                self.my_remote_port = 30000 + (hash_val % 20000)

            print(f"[FRP] 本节点 gRPC 访问地址: {self.my_address}:{self.my_remote_port}")
            if self.my_chatgpt_remote_port:
                print(f"[FRP] 本节点 ChatGPT API 访问地址: {self.my_address}:{self.my_chatgpt_remote_port}")
        
        config_path = self.frp_config.get_frpc_config_path(self.node_id)
        self.frp_config.save_frpc_config(frpc_config, self.node_id)
        
        # 3. 启动 frpc 进程
        frpc_path = get_frpc_path()
        self.frp_process_manager = FRPProcessManager(frpc_path, config_path)
        self.frp_process_manager.start()
        
        # 4. 添加种子节点到已知节点列表
        for seed_info in self.seed_node_infos:
            if seed_info.node_id != self.node_id:
                self.known_node_infos[seed_info.node_id] = seed_info
        
        # 5. 🆕 从本地缓存恢复 P2P 对端（Manager 下线时仍可直连）
        self.load_peer_cache()
        
        # 6. 启动发现任务
        self.listen_task = asyncio.create_task(self._discovery_loop())
        
        logging.info(f"[FRP start] Discovery loop task created: {self.listen_task}")
        print("[FRP] FRP 发现模块启动成功")
        print()

    async def stop(self) -> None:
        """停止 frp 发现"""
        # 保存最终 P2P 对端信息到本地缓存
        self.save_peer_cache()
        
        if self.listen_task:
            self.listen_task.cancel()
        
        if self.frp_process_manager:
            self.frp_process_manager.stop()

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """发现对等节点"""
        logging.info(f"[FRP discover_peers] Called, known_peers={[p.id() for p in self.known_peers.values()]}")
        
        if not self.known_peers and self.known_node_infos:
            logging.info(f"[FRP discover_peers] known_peers is empty but known_node_infos has {len(self.known_node_infos)} nodes, waiting for health check...")
            max_wait = 5.0
            wait_interval = 0.1
            waited = 0.0
            while not self.known_peers and waited < max_wait:
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                if waited >= 3.0 and int(waited) % 5 == 0:  # 减少日志频率：每5秒输出一次（原每2秒）
                    logging.info(f"[FRP discover_peers] Still waiting for health check... ({waited:.1f}s)")
            
            if self.known_peers:
                logging.info(f"[FRP discover_peers] Health check completed, found {len(self.known_peers)} peers")
            else:
                logging.warning(f"[FRP discover_peers] Health check timeout after {max_wait}s, no peers found")
        
        if wait_for_peers > 0:
            while len(self.known_peers) < wait_for_peers:
                if DEBUG_DISCOVERY >= 2:
                    print(f"[FRP] 当前对等节点: {len(self.known_peers)}/{wait_for_peers}. 等待更多节点...")
                await asyncio.sleep(0.1)
        
        if DEBUG_DISCOVERY >= 2:
            print(f"[FRP] 发现的对等节点: {[peer.id() for peer in self.known_peers.values()]}")
        
        result = list(self.known_peers.values())
        logging.info(f"[FRP discover_peers] Returning {len(result)} peers: {[p.id() for p in result]}")
        return result

    def add_known_node(self, node_id: str, address: str, port: int, 
                       description: str = "Connected", 
                       device_capabilities: Optional[DeviceCapabilities] = None) -> bool:
        """
        添加已知节点（被动接收连接时调用）
        
        Args:
            node_id: 节点 ID
            address: 节点地址
            port: 节点端口
            description: 描述
            device_capabilities: 设备能力
            
        Returns:
            是否成功添加（如果已存在或是自己则返回 False）
        """
        logging.info(f"[FRP add_known_node] Called with node_id={node_id}, address={address}:{port}, self.node_id={self.node_id}")
        if node_id == self.node_id:
            logging.info(f"[FRP add_known_node] Skipping self node")
            return False

        if node_id in self.known_node_infos:
            logging.info(f"[FRP add_known_node] Node already known: {node_id}")
            return False

        # 🔑 核心优化：使用 FRP P2P 地址而非 Manager relay 地址
        # Manager 返回的是中继地址 (relay://xxx.app.cloudstudio.work)
        # FRP P2P 模式需要通过 frps 协调建立直连
        # 正确地址应该是: frp_server_addr + remotePort(基于node_id计算)
        if self.frp_server_addr and self.enable_p2p and address:
            original_addr = f"{address}:{port}"
            frp_port = calculate_remote_port(node_id)
            frp_address = self.frp_server_addr

            logging.info(
                f"[FRP add_known_node] Address conversion for {node_id}:\n"
                f"   Original (from Manager): {original_addr}\n"
                f"   Converted (FRP P2P):     {frp_address}:{frp_port}"
            )

            address = frp_address
            port = frp_port
            description = description or "FRP-P2P (via Manager discovery)"

        node_info = NodeInfo(
            node_id=node_id,
            address=address,
            port=port,
            description=description,
            device_capabilities=device_capabilities or UNKNOWN_DEVICE_CAPABILITIES
        )
        self.known_node_infos[node_id] = node_info
        logging.info(f"[FRP add_known_node] Added new node to discovery list: {node_id} @ {address}:{port}")
        print(f"[FRP] ✅ 添加新节点到发现列表: {node_id} @ {address}:{port}")
        
        # 新节点加入时持久化（确保 Manager 下线后仍可直连）
        self.save_peer_cache()
        
        return True

    def get_my_address_info(self) -> Optional[Dict[str, any]]:
        """获取本节点的 FRP 地址信息"""
        if self.my_address and self.my_remote_port:
            return {
                "address": self.my_address,
                "port": self.my_remote_port
            }
        return None

    def get_my_chatgpt_address_info(self) -> Optional[Dict[str, any]]:
        """获取本节点 ChatGPT API 的 FRP 地址信息"""
        if self.my_address and self.my_chatgpt_remote_port:
            return {
                "address": self.my_address,
                "port": self.my_chatgpt_remote_port
            }
        return None

    def get_frp_p2p_address(self, node_id: str, original_addr: str) -> str:
        """
        将 Manager 返回的原始地址转换为 FRP P2P 地址

        这是核心方法，被 node.py 的 update_peers() 调用，
        确保所有节点间通信都通过 FRP P2P 通道。

        Args:
            node_id: 目标节点 ID
            original_addr: Manager 返回的原始地址 (如 "10.2.25.205:50051")

        Returns:
            FRP P2P 地址 (如 "119.45.114.133:37765")
            或原始地址（如果 FRP 未启用）
        """
        # 如果 FRP P2P 模式已启用，进行地址转换
        if self.frp_server_addr and self.enable_p2p:
            frp_port = calculate_remote_port(node_id)
            frp_addr = f"{self.frp_server_addr}:{frp_port}"

            logging.info(
                f"[FRP get_frp_p2p_address] Converting address for {node_id}:\n"
                f"   Original (Manager): {original_addr}\n"
                f"   Converted (FRP P2P): {frp_addr}"
            )

            return frp_addr

        # FRP 未启用，返回原始地址
        logging.debug(f"[FRP get_frp_p2p_address] FRP not enabled, using original address: {original_addr}")
        return original_addr

    # ==================== P2P 对端信息持久化 ====================

    def _get_cache_file_path(self) -> Path:
        """获取当前节点的缓存文件路径"""
        return _PEER_CACHE_DIR / f"{self.node_id}.json"

    def save_peer_cache(self):
        """
        将已发现的 P2P 对端信息持久化到本地文件
        
        保存内容：
        - 所有 known_node_infos（不含自身）
        - FRP 服务端地址（用于地址计算）
        - 保存时间戳
        """
        try:
            cache_data = {
                "version": 1,
                "node_id": self.node_id,
                "frp_server_addr": self.frp_server_addr,
                "frp_server_port": self.frp_server_port,
                "enable_p2p": self.enable_p2p,
                "saved_at": time.time(),
                "peers": {}
            }

            for node_id, info in self.known_node_infos.items():
                if node_id == self.node_id:
                    continue
                cache_data["peers"][node_id] = info.to_dict()

            _PEER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = self._get_cache_file_path()
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            peer_count = len(cache_data["peers"])
            logging.info(f"💾 [PeerCache] P2P对端缓存已保存: {peer_count} 个节点 → {cache_path}")

        except Exception as e:
            logging.error(f"❌ [PeerCache] 保存缓存失败: {e}")

    def load_peer_cache(self) -> int:
        """
        从本地文件加载缓存的 P2P 对端信息

        Returns:
            成功恢复的节点数量
            缓存不存在或已过期则返回 0
        """
        cache_path = self._get_cache_file_path()
        if not cache_path.exists():
            logging.info("[PeerCache] 无本地缓存文件，跳过加载")
            return 0

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # 检查缓存是否过期
            saved_at = cache_data.get("saved_at", 0)
            age_seconds = time.time() - saved_at
            if age_seconds > _PEER_CACHE_TTL:
                logging.warning(
                    f"[PeerCache] 缓存已过期 (保存于 {age_seconds / 86400:.1f} 天前)，"
                    f"阈值 {_PEER_CACHE_TTL / 86400:.0f} 天，跳过加载"
                )
                return 0

            # 检查 FRP 配置是否匹配（避免使用旧配置的缓存）
            cached_frp_addr = cache_data.get("frp_server_addr", "")
            cached_frp_port = cache_data.get("frp_server_port", 0)
            if (cached_frp_addr and cached_frp_addr != self.frp_server_addr) or \
               (cached_frp_port and cached_frp_port != self.frp_server_port):
                logging.warning(
                    f"[PeerCache] FRP 配置不匹配 "
                    f"(缓存: {cached_frp_addr}:{cached_frp_port}, "
                    f"当前: {self.frp_server_addr}:{self.frp_server_port})，跳过加载"
                )
                return 0

            restored = 0
            for node_id, peer_data in cache_data.get("peers", {}).items():
                # 跳过自身和已在内存中的节点
                if node_id == self.node_id:
                    continue
                if node_id in self.known_node_infos:
                    continue

                try:
                    dc = peer_data.get("device_capabilities", {})
                    flops_data = dc.get("flops", {"fp32": 0, "fp16": 0, "int8": 0})
                    
                    from exo.topology.device_capabilities import DeviceFlops
                    node_info = NodeInfo(
                        node_id=node_id,
                        address=peer_data.get("address", ""),
                        port=peer_data.get("port", 0),
                        description=f"{peer_data.get('description', '')} (from cache)",
                        device_capabilities=DeviceCapabilities(
                            model=dc.get("model", "unknown"),
                            chip=dc.get("chip", "unknown"),
                            memory=dc.get("memory", 0),
                            flops=DeviceFlops(
                                fp32=flops_data.get("fp32", 0),
                                fp16=flops_data.get("fp16", 0),
                                int8=flops_data.get("int8", 0)
                            )
                        )
                    )
                    self.known_node_infos[node_id] = node_info
                    restored += 1
                except Exception as e:
                    logging.warning(f"[PeerCache] 恢复节点 {node_id} 失败: {e}")

            if restored > 0:
                logging.info(
                    f"📂 [PeerCache] 从本地缓存恢复了 {restored} 个 P2P 对端 "
                    f"(缓存年龄: {age_seconds / 3600:.1f} 小时)"
                )
                print(f"[FRP] 📂 从本地缓存恢复了 {restored} 个 P2P 对端 (Manager 离线时仍可直连)")

            return restored

        except Exception as e:
            logging.error(f"❌ [PeerCache] 加载缓存失败: {e}")
            return 0

    def _should_retry_node(self, node_id: str, now: Optional[float] = None) -> bool:
        """判断节点是否已过退避期，可以进行下一次重试"""
        now = now or time.time()
        state = self.node_retry_state.get(node_id)
        if not state:
            return True
        return now >= state.get("next_retry_at", 0)

    def _record_node_failure(self, node_id: str):
        """记录节点连接失败，并计算下一次重试时间（指数退避）"""
        now = time.time()
        state = self.node_retry_state.setdefault(node_id, {"fail_count": 0, "next_retry_at": now})
        state["fail_count"] += 1
        delay = min(
            self.retry_base_delay * (self.retry_backoff_factor ** (state["fail_count"] - 1)),
            self.retry_max_delay
        )
        state["next_retry_at"] = now + delay
        logging.debug(
            f"[FRP] 节点 {node_id} 连接失败 #{state['fail_count']}，"
            f"{delay:.0f}s 后重试"
        )

    def _record_node_success(self, node_id: str):
        """节点连接成功，重置失败退避状态"""
        if node_id in self.node_retry_state:
            self.node_retry_state[node_id]["fail_count"] = 0
            self.node_retry_state[node_id]["next_retry_at"] = 0

    async def _discovery_loop(self):
        """发现循环"""
        logging.info("[FRP _discovery_loop] Starting automatic node discovery...")
        print("[FRP] 开始自动节点发现...")
        
        await asyncio.sleep(3)
        
        last_online_peers = set()
        
        while True:
            try:
                logging.info(f"[FRP _discovery_loop] known_node_infos count: {len(self.known_node_infos)}, nodes: {list(self.known_node_infos.keys())}")
                if self.known_node_infos:
                    now = time.time()
                    nodes_to_check = [
                        node_info
                        for node_info in self.known_node_infos.values()
                        if node_info.node_id != self.node_id and self._should_retry_node(node_info.node_id, now)
                    ]
                    skipped_count = len(self.known_node_infos) - len(nodes_to_check)
                    if skipped_count > 0:
                        logging.debug(f"[FRP _discovery_loop] {skipped_count} 个节点处于退避期，跳过")

                    if DEBUG_DISCOVERY:
                        print(f"[FRP] 正在检查 {len(nodes_to_check)} 个已知节点（跳过 {skipped_count} 个）...")

                    health_check_tasks = [
                        self._check_and_update_node(node_info)
                        for node_info in nodes_to_check
                    ]
                    
                    if health_check_tasks:
                        results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
                        
                        new_known_peers = {}
                        for result in results:
                            if isinstance(result, PeerHandle):
                                new_known_peers[result.id()] = result
                        
                        logging.info(f"[FRP _discovery_loop] Health check results: {len(new_known_peers)} healthy peers")
                        
                        new_peers = set(new_known_peers.keys()) - set(self.known_peers.keys())
                        if new_peers:
                            print(f"[FRP] 新节点上线: {new_peers}")
                        
                        removed_peers = set(self.known_peers.keys()) - set(new_known_peers.keys())
                        if removed_peers:
                            print(f"[FRP] 节点下线: {removed_peers}")
                            logging.warning(f"[FRP _discovery_loop] Peers removed due to failed health check: {removed_peers}")
                        
                        self.known_peers = new_known_peers
                    
                    current_online = set(self.known_peers.keys())
                    if current_online != last_online_peers:
                        print(f"[FRP] 当前在线节点: {list(self.known_peers.keys())}")
                        last_online_peers = current_online
                else:
                    logging.info("[FRP _discovery_loop] No known nodes, waiting for discovery...")
                    if DEBUG_DISCOVERY:
                        print("[FRP] 没有已知节点，等待发现...")
                
            except Exception as e:
                logging.error(f"[FRP _discovery_loop] Error in discovery loop: {e}")
                print(f"[FRP] 发现循环出错: {e}")
                import traceback
                traceback.print_exc()
            
            await asyncio.sleep(5)

    async def _check_and_update_node(self, node_info: NodeInfo) -> Optional[PeerHandle]:
        """检查节点健康状态，并尝试获取该节点知道的其他节点"""
        try:
            peer = self.known_peers.get(node_info.node_id)
            is_reconnect = False
            if not peer:
                print(f"[FRP] 尝试连接节点: {node_info.node_id} @ {node_info.address}:{node_info.port}")
                peer = self.create_peer_handle(
                    node_info.node_id,
                    f"{node_info.address}:{node_info.port}",
                    node_info.description,
                    node_info.device_capabilities
                )
                is_reconnect = True
            
            logging.info(f"[FRP _check_and_update_node] Checking health for {node_info.node_id}")
            is_healthy = await asyncio.wait_for(peer.health_check(), timeout=30.0)
            logging.info(f"[FRP _check_and_update_node] Health check result for {node_info.node_id}: {is_healthy}")
            
            if is_healthy:
                actual_node_id = peer.id()
                if actual_node_id == self.node_id:
                    print(f"[FRP] 检测到自身节点，跳过: {node_info.node_id}")
                    if node_info.node_id in self.known_node_infos:
                        del self.known_node_infos[node_info.node_id]
                    return None
                
                if is_reconnect or DEBUG_DISCOVERY:
                    print(f"[FRP] 节点连接成功: {actual_node_id} (配置ID: {node_info.node_id})")
                
                try:
                    topology = await asyncio.wait_for(
                        peer.collect_topology(visited=set(), max_depth=2),
                        timeout=5.0
                    )
                    
                    new_nodes_found = 0
                    for node_id, capabilities in topology.nodes.items():
                        if node_id != self.node_id and node_id not in self.known_node_infos:
                            remote_port = calculate_remote_port(node_id)
                            
                            new_node_info = NodeInfo(
                                node_id=node_id,
                                address=self.frp_server_addr,
                                port=remote_port,
                                description="Auto-discovered",
                                device_capabilities=capabilities
                            )
                            self.known_node_infos[node_id] = new_node_info
                            new_nodes_found += 1
                            print(f"[FRP] 发现新节点: {node_id} @ {self.frp_server_addr}:{remote_port}")
                    
                    if new_nodes_found > 0:
                        print(f"[FRP] 从 {node_info.node_id} 发现 {new_nodes_found} 个新节点")
                        # 拓扑发现新节点后持久化（Manager 下线后仍可直连）
                        self.save_peer_cache()
                        
                except Exception as e:
                    print(f"[FRP] 从 {node_info.node_id} 获取拓扑信息失败: {e}")

                self._record_node_success(node_info.node_id)
                return peer
            else:
                print(f"[FRP] 节点不健康: {node_info.node_id}")
                self._record_node_failure(node_info.node_id)
                return None
                
        except asyncio.TimeoutError:
            print(f"[FRP] 连接超时: {node_info.node_id} @ {node_info.address}:{node_info.port}")
            logging.warning(f"[FRP _check_and_update_node] Timeout checking {node_info.node_id}")
            self._record_node_failure(node_info.node_id)
            return None
        except Exception as e:
            print(f"[FRP] 连接失败: {node_info.node_id}, 错误: {e}")
            logging.error(f"[FRP _check_and_update_node] Failed to check {node_info.node_id}: {e}")
            self._record_node_failure(node_info.node_id)
            return None
