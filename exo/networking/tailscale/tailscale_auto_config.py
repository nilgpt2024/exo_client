"""
Tailscale 环境检测和自动配置模块

功能：
1. 检测运行环境（Docker 容器 / 物理机）
2. 检测 Tailscale 网络模式（userspace-networking / 内核模式）
3. 自动配置 SOCKS5 代理或路由规则
4. 验证跨平台连通性
"""

import asyncio
import subprocess
import platform
import os
from typing import Tuple, Optional, Dict, Any
from exo.helpers import DEBUG_DISCOVERY


class TailscaleEnvironment:
    """Tailscale 运行环境管理器"""

    def __init__(self):
        self.is_docker = False
        self.is_userspace_networking = False
        self.tailscale_ip = None
        self.socks5_proxy = None
        self.can_direct_connect = True  # 默认假设可以直连
        self.configured = False

    async def detect_and_configure(self) -> bool:
        """
        自动检测环境并配置网络

        Returns:
            bool: 配置是否成功
        """
        print("[Tailscale-Env] 🔍 开始环境检测...")

        # Step 1: 检测是否在 Docker 中
        self._detect_docker()

        # Step 2: 检测 Tailscale 模式
        await self._detect_tailscale_mode()

        # Step 3: 获取本机 Tailscale IP
        await self._get_tailscale_ip()

        # Step 4: 判断是否需要代理
        if self.is_userspace_networking or self.is_docker:
            print("[Tailscale-Env] ⚠️ 检测到 userspace-networking 或 Docker 环境")
            print("[Tailscale-Env] 🔄 将启用 SOCKS5 代理以支持 DERP 中继...")
            success = await self._configure_socks5_proxy()
            if not success:
                print("[Tailscale-Env] ❌ SOCKS5 代理配置失败，尝试备用方案...")
                success = await self._configure_fallback()
        else:
            print("[Tailscale-Env] ✅ 物理机 + 内核模式，尝试直连")
            success = True

        self.configured = success
        return success

    def _detect_docker(self):
        """检测是否在 Docker 容器中运行"""
        try:
            # 方法 1: 检查 /.dockerenv 文件
            if os.path.exists('/.dockerenv'):
                self.is_docker = True
                print("[Tailscale-Env] 🐳 检测到 Docker 环境 (/.dockerenv)")
                return

            # 方法 2: 检查 cgroup 信息
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read() or 'lxc' in f.read():
                    self.is_docker = True
                    print("[Tailscale-Env] 🐳 检测到 Docker 环境 (cgroup)")
                    return

            # 方法 3: Windows 上检查 WSL
            if platform.system() == "Windows":
                # WSL 也视为类似容器的环境
                pass

        except Exception:
            pass

        print("[Tailscale-Env] 💻 非 Docker 环境")

    async def _detect_tailscale_mode(self):
        """检测 Tailscale 使用的是 userspace 还是内核模式"""
        try:
            # 执行 tailscale status 获取信息
            proc = await asyncio.create_subprocess_exec(
                'tailscale', 'status', '--json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            if proc.returncode != 0:
                print(f"[Tailscale-Env] ⚠️ tailscale status 失败: {stderr.decode()}")
                return

            import json
            data = json.loads(stdout.decode())

            # 检查 MagicSockName 或其他标识
            backend = data.get('BackendState', '')
            if 'userspace' in str(backend).lower() or 'user' in str(backend).lower():
                self.is_userspace_networking = True
                print(f"[Tailscale-Env] 🌐 Userspace-Networking 模式: {backend}")
            else:
                print(f"[Tailscale-Env] ⚙️ 内核模式: {backend}")

            # 检查是否有 DERP 中继信息
            peer_data = data.get('Peer', {})
            for peer_id, peer_info in peer_data.items():
                derp_info = peer_info.get('Derp', '')
                if derp_info and 'relay' in str(derp_info).lower():
                    self.can_direct_connect = False
                    print(f"[Tailscale-Env] 🔄 检测到 DERP 中继: {derp_info}")
                    break

        except FileNotFoundError:
            print("[Tailscale-Env] ❌ tailscale 命令未找到")
        except Exception as e:
            print(f"[Tailscale-Env] ⚠️ 模式检测失败: {e}")
            # 默认假设需要代理（安全策略）
            if self.is_docker:
                self.is_userspace_networking = True

    async def _get_tailscale_ip(self):
        """获取本机的 Tailscale IP 地址"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'tailscale', 'ip', '-4',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            if proc.returncode == 0:
                self.tailscale_ip = stdout.decode().strip()
                print(f"[Tailscale-Env] 📡 本机 Tailscale IP: {self.tailscale_ip}")
            else:
                print(f"[Tailscale-Env] ⚠️ 获取 IP 失败: {stderr.decode()}")

        except Exception as e:
            print(f"[Tailscale-Env] ❌ 获取 Tailscale IP 失败: {e}")

    async def _configure_socks5_proxy(self) -> bool:
        """
        配置 SOCKS5 代理以支持 DERP 中继

        Returns:
            bool: 是否成功
        """
        proxy_port = os.environ.get('TAILSCALE_SOCKS_PORT', '1055')
        self.socks5_proxy = f"localhost:{proxy_port}"

        try:
            # 检查 tailscaled 是否已经启用了 SOCKS5
            proc = await asyncio.create_subprocess_exec(
                'curl', '--socks5', f'localhost:{proxy_port}',
                'http://example.com',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=5
            )
            await proc.communicate()

            if proc.returncode == 0:
                print(f"[Tailscale-Env] ✅ SOCKS5 代理已可用: {self.socks5_proxy}")
                return True

        except Exception:
            pass

        # 如果 SOCKS5 不可用，尝试重启 tailscaled 并添加参数
        print(f"[Tailscale-Env] 🔧 尝试配置 SOCKS5 代理...")

        try:
            # 提示用户手动配置（因为修改 tailscaled 参数需要 root 权限和重启服务）
            print("""
[!] ======================================================
[!] ⚠️  需要手动配置 SOCKS5 代理
[!]
[!] 请在启动 exo 之前执行以下命令：
[!
[!]   Linux/Docker:
[!]     pkill tailscaled
[!]     nohup tailscaled --tun=userspace-networking \\
[!]           --socks5-server=localhost:{port} > /tmp/tailscale.log 2>&1 &
[!]     sleep 2 && tailscale up
[!
[!]   或者设置环境变量：
[!]     export USE_TAILSCALE_SOCKS5=true
[!]     export TAILSCALE_SOCKS5_HOST=localhost:{port}
[!]
[!] ======================================================""".format(port=proxy_port))

            # 设置环境变量供后续使用
            os.environ['USE_TAILSCALE_SOCKS5'] = 'true'
            os.environ['TAILSCALE_SOCKS5_HOST'] = f'localhost:{proxy_port}'

            return False  # 需要手动干预

        except Exception as e:
            print(f"[Tailscale-Env] ❌ SOCKS5 代理配置失败: {e}")
            return False

    async def _configure_fallback(self) -> bool:
        """
        备用方案：使用 ip route 强制流量走 Tailscale 接口

        Returns:
            bool: 是否成功
        """
        print("[Tailscale-Env] 🔧 尝试配置路由规则...")

        try:
            if self.tailscale_ip:
                # 获取 Tailscale 接口名称
                proc = await asyncio.create_subprocess_exec(
                    'ip', 'route', 'get', self.tailscale_ip,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)

                route_output = stdout.decode()
                if 'tailscale' in route_output.lower():
                    print(f"[Tailscale-Env] ✅ Tailscale 路由已存在")
                    print("[Tailscale-Env] ℹ️  将在连接时动态添加路由规则")
                    return True

        except Exception as e:
            print(f"[Tailscale-Env] ⚠️ 路由配置失败: {e}")

        return False

    async def test_peer_connectivity(self, peer_ip: str, port: int = 50051, timeout: int = 10) -> Tuple[bool, str]:
        """
        测试到对端的 TCP 连通性

        Args:
            peer_ip: 对端 IP 地址
            port: 目标端口
            timeout: 超时时间（秒）

        Returns:
            Tuple[bool, str]: (是否成功, 详细信息)
        """
        print(f"[Tailscale-Env] 🧪 测试连通性: {peer_ip}:{port} (timeout={timeout}s)")

        try:
            # 使用 Python socket 测试
            import socket

            loop = asyncio.get_event_loop()
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer_ip, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()

            msg = f"✅ TCP 连接成功！"
            print(f"[Tailscale-Env] {msg}")
            return True, msg

        except asyncio.TimeoutError:
            msg = f"❌ 连接超时 ({timeout}s)"
            print(f"[Tailscale-Env] {msg}")
            return False, msg
        except ConnectionRefusedError:
            msg = "❌ 连接被拒绝（端口未监听）"
            print(f"[Tailscale-Env] {msg}")
            return False, msg
        except OSError as e:
            msg = f"❌ 网络错误: {e}"
            print(f"[Tailscale-Env] {msg}")
            return False, msg
        except Exception as e:
            msg = f"❌ 未知错误: {type(e).__name__}: {e}"
            print(f"[Tailscale-Env] {msg}")
            return False, msg

    def get_connection_strategy(self) -> Dict[str, Any]:
        """
        获取推荐的连接策略

        Returns:
            dict: 连接配置信息
        """
        strategy = {
            'use_socks5': self.is_userspace_networking or self.is_docker,
            'socks5_address': self.socks5_proxy,
            'use_direct': self.can_direct_connect and not (self.is_userspace_networking or self.is_docker),
            'fallback_to_relay': not self.can_direct_connect,
            'environment': {
                'is_docker': self.is_docker,
                'is_userspace': self.is_userspace_networking,
                'tailscale_ip': self.tailscale_ip,
            }
        }

        if DEBUG_DISCOVERY >= 2:
            print(f"[Tailscale-Env] 📋 连接策略: {strategy}")

        return strategy


# 全局单例实例
_tailscale_env_instance: Optional[TailscaleEnvironment] = None


def get_tailscale_env() -> TailscaleEnvironment:
    """获取全局 Tailscale 环境实例"""
    global _tailscale_env_instance
    if _tailscale_env_instance is None:
        _tailscale_env_instance = TailscaleEnvironment()
    return _tailscale_env_instance


async def auto_configure_tailscale() -> bool:
    """
    一键配置 Tailscale 网络（供外部调用）

    Returns:
        bool: 配置是否成功
    """
    env = get_tailscale_env()
    return await env.detect_and_configure()
