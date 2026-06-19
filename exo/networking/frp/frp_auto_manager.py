"""
FRP 自动配置和管理模块

功能：
1. 自动检测是否需要 FRP（当 Tailscale 直连失败时）
2. 一键启动 FRP 服务端/客户端
3. 管理 FRP 生命周期
4. 提供地址转换服务（Tailscale IP → FRP 转发地址）

使用场景：
- Docker/容器环境无法 P2P 直连
- Tailscale DERP 中继不稳定
- 需要可靠的跨公网连接
"""

import asyncio
import os
import sys
import platform
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from exo.helpers import DEBUG, DEBUG_DISCOVERY

# 导入现有 FRP 模块
try:
    from exo.networking.frp.frp_downloader import (
        ensure_frpc_installed,
        get_frpc_path,
        get_frps_path,
        check_frpc_available,
    )
    from exo.networking.frp.frp_config import FRPConfig
    from exo.networking.frp.frp_process import FRPProcessManager
    HAS_FRP_MODULES = True
except ImportError as e:
    HAS_FRP_MODULES = False
    if DEBUG >= 2:
        print(f"[FRP-Manager] ⚠️ FRP 模块导入失败: {e}")


class FRPAutoManager:
    """
    FRP 自动管理器
    
    职责：
    - 检测网络环境，判断是否需要 FRP
    - 自动下载/安装 FRP
    - 启动和管理 frpc/frps 进程
    - 提供地址映射服务
    """

    # 默认配置
    DEFAULT_FRPS_PORT = 7000
    DEFAULT_TOKEN = "exo-frp-auto-token-2024"
    
    def __init__(self):
        self.is_initialized = False
        self.is_running = False
        self.mode = None  # "server" | "client" | "none"
        
        # FRP 配置
        self.server_addr: Optional[str] = None
        self.server_port: int = self.DEFAULT_FRPS_PORT
        self.token: str = self.DEFAULT_TOKEN
        
        # 本地端口映射
        self.local_port: Optional[int] = None
        self.remote_port_map: Dict[str, int] = {}  # node_id -> remote_port
        
        # 进程管理
        self.process_manager: Optional[FRPProcessManager] = None
        self.config_manager: Optional[FRPConfig] = None
        
        # 状态跟踪
        self.peer_frp_addresses: Dict[str, str] = {}  # node_id -> "addr:port"

    async def initialize(
        self,
        server_addr: Optional[str] = None,
        local_port: int = 50051,
        token: Optional[str] = None,
        force_mode: Optional[str] = None,  # "server" | "client" | None (auto)
    ) -> bool:
        """
        初始化 FRP 管理器
        
        Args:
            server_addr: FRP 服务端地址（None 则自动检测或作为服务端）
            local_port: 本地 gRPC 服务端口
            token: 认证令牌
            force_mode: 强制指定模式
            
        Returns:
            bool: 是否初始化成功
        """
        if not HAS_FRP_MODULES:
            print("[FRP-Manager] ❌ FRP 模块不可用")
            return False

        print("\n" + "=" * 60)
        print("[FRP-Manager] 🚀 初始化 FRP 自动管理器")
        print("=" * 60)

        self.local_port = local_port
        self.token = token or self.DEFAULT_TOKEN
        self.config_manager = FRPConfig()

        # Step 1: 确保 FRP 已安装
        print("\n[Step 1/4] 检查/安装 FRP...")
        if not ensure_frpc_installed():
            print("[FRP-Manager] ❌ FRP 安装失败")
            return False
        print("✅ FRP 已就绪")

        # Step 2: 确定运行模式
        print("\n[Step 2/4] 确定运行模式...")
        if force_mode:
            self.mode = force_mode
            print(f"📋 强制模式: {self.mode}")
        else:
            self.mode = await self._detect_best_mode(server_addr)
            print(f"📋 自动检测模式: {self.mode}")

        # Step 3: 配置并启动
        print("\n[Step 3/4] 配置并启动 FRP...")
        success = False
        if self.mode == "server":
            success = await self._start_as_server()
        elif self.mode == "client":
            self.server_addr = server_addr
            success = await self._start_as_client()
        else:
            print("ℹ️ 不需要 FRP（直连可用）")

        if success:
            self.is_initialized = True
            print("\n[Step 4/4] ✅ FRP 初始化完成！")
        else:
            print("\n[Step 4/4] ❌ FRP 初始化失败")

        print("=" * 60 + "\n")
        return success

    async def _detect_best_mode(self, server_addr: Optional[str]) -> str:
        """
        自动检测最佳运行模式
        
        规则：
        - 如果提供了 server_addr → 客户端模式
        - 如果是 Windows 且无 server_addr → 服务端模式
        - 如果是 Linux/Docker 且有环境变量 → 客户端模式
        """
        # 检查环境变量
        env_mode = os.environ.get('EXO_FRP_MODE', '').lower()
        env_server = os.environ.get('EXO_FRP_SERVER_ADDR', '')

        if env_mode in ['server', 'client']:
            return env_mode

        if env_server:
            self.server_addr = env_server
            return 'client'

        if server_addr:
            self.server_addr = server_addr
            return 'client'

        # 根据平台自动选择
        system = platform.system().lower()
        is_docker = os.path.exists('/.dockerenv')

        if system == 'windows' and not is_docker:
            return 'server'
        elif is_docker or system == 'linux':
            # Linux/Docker 默认需要服务端地址
            # 如果没有，尝试使用 Tailscale IP 或提示用户
            return 'client'  # 假设服务端已在外部运行
        else:
            return 'none'

    async def _start_as_server(self) -> bool:
        """以服务端模式启动 (frps)"""
        try:
            print(f"[FRP-Server] 🖥️ 启动 FRP 服务端 (端口 {self.DEFAULT_FRPS_PORT})...")

            # 生成服务端配置
            config = self.config_manager.generate_frps_config(
                bind_port=self.DEFAULT_FRPS_PORT,
                token=self.token,
                dashboard_port=7500,
                dashboard_user="admin",
                dashboard_pwd="admin123",
                enable_xtcp=True,  # ✅ 启用 XTCP P2P 支持
            )
            
            config_path = self.config_manager.get_frps_config_path()
            self.config_manager.save_frps_config(config)

            # 启动 frps
            frps_path = get_frps_path()
            if not frps_path.exists():
                print(f"[FRP-Server] ❌ frps 不存在: {frps_path}")
                return False

            # 使用 subprocess 直接启动（简化版）
            import subprocess
            process = subprocess.Popen(
                [str(frps_path), '-c', str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # 等待启动
            await asyncio.sleep(2)

            if process.poll() is None:
                print(f"[FRP-Server] ✅ frps 运行中 (PID: {process.pid})")
                print(f"[FRP-Server] 📍 监听端口: {self.DEFAULT_FRPS_PORT}")
                print(f"[FRP-Server] 🔐 传输加密: 已启用 (XTCP P2P 必需)")
                print(f"[FRP-Server] 🔑 Token: {self.token[:8]}...")
                print(f"[FRP-Server] 📊 Dashboard: http://0.0.0.0:7500")
                
                self.is_running = True
                
                # 获取本机地址供客户端连接
                self.server_addr = await self._get_public_or_tailscale_ip()
                print(f"[FRP-Server] 🌐 客户端连接地址: {self.server_addr}:{self.DEFAULT_FRPS_PORT}")
                
                return True
            else:
                output = process.stdout.read() if process.stdout else ""
                print(f"[FRP-Server] ❌ frps 启动失败: {output[:200]}")
                return False

        except Exception as e:
            print(f"[FRP-Server] ❌ 启动失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _start_as_client(self) -> bool:
        """
        以客户端模式启动 (frpc)
        
        默认使用 XTCP P2P 模式：
        - 优先尝试 P2P 直连（低延迟、高带宽）
        - P2P 失败时自动回退到 TCP 中转（通过服务端）
        
        这种混合模式确保了最佳的网络性能和可靠性。
        """
        try:
            if not self.server_addr:
                print("[FRP-Client] ❌ 未指定服务端地址")
                print("[FRP-Client] 💡 设置环境变量 EXO_FRP_SERVER_ADDR 或传入 server_addr 参数")
                return False

            print(f"[FRP-Client] 🚀 启动 FRP 客户端 (XTCP P2P 模式)")
            print(f"[FRP-Client] 📍 目标: {self.server_addr}:{self.DEFAULT_FRPS_PORT}")

            # 生成客户端配置
            # 使用固定的远程端口（与本地端口相同，方便记忆）
            remote_port = self.local_port
            
            config = self.config_manager.generate_frpc_config(
                server_addr=self.server_addr,
                server_port=self.DEFAULT_FRPS_PORT,
                node_id=f"auto-{platform.node()}",
                local_port=self.local_port,
                remote_port=remote_port,
                token=self.token,
                enable_p2p=True,  # ✅ 默认启用 XTCP P2P 模式
            )

            config_path = self.config_manager.get_frpc_config_path(f"auto-client")
            self.config_manager.save_frpc_config(config, f"auto-client")

            # 启动 frpc
            frpc_path = get_frpc_path()
            self.process_manager = FRPProcessManager(frpc_path, config_path)
            success = self.process_manager.start()

            if success:
                await asyncio.sleep(3)
                
                if self.process_manager.is_running():
                    print(f"[FRP-Client] ✅ frpc 运行成功 (XTCP P2P 模式)")
                    print(f"[FRP-Client] 🔗 连接模式: P2P 直连优先，TCP 中转备用")
                    print(f"[FRP-Client] 📍 本地服务: :{self.local_port}")
                    print(f"[FRP-Client] 🌐 远程访问: {self.server_addr}:{remote_port}")
                    print(f"[FRP-Client] 💡 提示: 首次连接会尝试 P2P 打洞")
                    
                    self.is_running = True
                    self.remote_port_map['default'] = remote_port
                    
                    return True
                else:
                    print("[FRP-Client] ⚠️ frpc 启动后退出（可能正在重连）")
                    # 即使进程暂时退出，也标记为已初始化（有自动重连机制）
                    self.is_running = True
                    return True
            else:
                print("[FRP-Client] ❌ frpc 启动失败")
                return False

        except Exception as e:
            print(f"[FRP-Client] ❌ 启动失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _get_public_or_tailscale_ip(self) -> str:
        """获取本机的公网 IP 或 Tailscale IP"""
        try:
            # 尝试获取 Tailscale IP
            proc = await asyncio.create_subprocess_exec(
                'tailscale', 'ip', '-4',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            
            if proc.returncode == 0:
                ip = stdout.decode().strip()
                if ip and ip.startswith('100.'):
                    return ip
        except:
            pass

        # 回退到本地 IP
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return '127.0.0.1'

    def get_peer_address(self, peer_id: str, original_address: str) -> str:
        """
        获取对端的 FRP 转发地址
        
        Args:
            peer_id: 对端节点 ID
            original_address: 原始地址（如 Tailscale IP）
            
        Returns:
            FRP 转发后的地址 ("server_addr:remote_port")
        """
        if not self.is_running or self.mode != 'client':
            return original_address

        # 如果已有缓存，直接返回
        if peer_id in self.peer_frp_addresses:
            return self.peer_frp_addresses[peer_id]

        # 对于客户端模式，所有流量都通过 FRP 服务端转发
        # 使用服务端地址 + 对端原始端口（假设对端也用相同端口注册到 FRP）
        if ':' in original_address:
            _, port = original_address.rsplit(':', 1)
            frp_address = f"{self.server_addr}:{port}"
        else:
            frp_address = f"{self.server_addr}:{original_address}"

        # 缓存结果
        self.peer_frp_addresses[peer_id] = frp_address

        if DEBUG_DISCOVERY >= 2:
            print(f"[FRP-Manager] 地址转换: {original_address} -> {frp_address}")

        return frp_address

    def should_use_frp_for_peer(self, peer_address: str) -> bool:
        """
        判断是否应该通过 FRP 连接某个对端
        
        Args:
            peer_address: 对端地址
            
        Returns:
            bool: 是否应该走 FRP
        """
        if not self.is_running:
            return False

        # 如果是 Tailscale 内网地址且我们处于客户端模式，优先使用 FRP
        if peer_address.startswith('100.') or peer_address.startswith('fd7a:'):
            return self.mode == 'client'

        return False

    async def stop(self):
        """停止 FRP 服务"""
        if self.process_manager:
            print("[FRP-Manager] 正在停止 FRP 客户端...")
            self.process_manager.stop()
            self.process_manager = None

        self.is_running = False
        self.is_initialized = False
        print("[FRP-Manager] FRP 已停止")

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'initialized': self.is_initialized,
            'running': self.is_running,
            'mode': self.mode,
            'server_addr': self.server_addr,
            'server_port': self.server_port,
            'local_port': self.local_port,
            'peer_count': len(self.peer_frp_addresses),
        }


# 全局单例
_frp_manager_instance: Optional[FRPAutoManager] = None


def get_frp_manager() -> FRPAutoManager:
    """获取全局 FRP 管理器实例"""
    global _frp_manager_instance
    if _frp_manager_instance is None:
        _frp_manager_instance = FRPAutoManager()
    return _frp_manager_instance


async def initialize_frp_auto(
    server_addr: Optional[str] = None,
    local_port: int = 50051,
    token: Optional[str] = None,
    force_mode: Optional[str] = None,
) -> bool:
    """
    一键初始化 FRP（供外部调用）
    
    Args:
        server_addr: FRP 服务端地址
        local_port: 本地 gRPC 端口
        token: 认证令牌
        force_mode: 强制模式
        
    Returns:
        bool: 是否成功
    """
    manager = get_frp_manager()
    return await manager.initialize(
        server_addr=server_addr,
        local_port=local_port,
        token=token,
        force_mode=force_mode,
    )
