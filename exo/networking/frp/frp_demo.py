#!/usr/bin/env python3
"""
FRP 集成使用示例

本脚本演示如何在 exo 中使用 FRP 进行跨公网节点发现

场景：
1. Windows 作为 FRP 服务端（frps）
2. Linux/Docker 作为 FRP 客户端（frpc）
3. 自动回退：当 Tailscale 直连失败时，自动切换到 FRP
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def demo_frp_server_mode():
    """
    演示 1: Windows 端启动 FRP 服务端
    
    适用场景：
    - Windows 有公网 IP 或 Tailscale IP 可达
    - Linux/Docker 在 NAT 后面，无法被主动连接
    """
    print("=" * 70)
    print("  场景 1: Windows 启动 FRP 服务端 (frps)")
    print("=" * 70)
    
    from exo.networking.frp.frp_auto_manager import initialize_frp_auto, get_frp_manager
    
    success = await initialize_frp_auto(
        server_addr=None,  # None 表示作为服务端运行
        local_port=50051,
        token="my-secure-token",
        force_mode='server',
    )
    
    if success:
        manager = get_frp_manager()
        status = manager.get_status()
        
        print("\n✅ FRP 服务端已启动！")
        print(f"   监听端口: {status['server_port']}")
        print(f"   服务地址: {status.get('server_addr', '待检测')}")
        print(f"\n📋 客户端连接信息:")
        print(f"   服务端地址: <你的公网IP或Tailscale_IP>")
        print(f"   服务端端口: {status['server_port']}")
        print(f"   Token: my-secure-token")
        print(f"\n💡 请在 Linux 端执行: 场景 2 的命令")
        
        # 保持运行（演示用）
        try:
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            await manager.stop()
    else:
        print("❌ FRP 服务端启动失败")


async def demo_frp_client_mode():
    """
    演示 2: Linux/Docker 端启动 FRP 客户端
    
    适用场景：
    - 连接到 Windows 的 FRP 服务端
    - 将本地 gRPC 服务暴露给外部
    """
    print("=" * 70)
    print("  场景 2: Linux 启动 FRP 客户端 (frpc)")
    print("=" * 70)
    
    # 从环境变量或参数获取服务端地址
    server_addr = os.environ.get('EXO_FRP_SERVER_ADDR')
    if not server_addr:
        print("❌ 未设置 EXO_FRP_SERVER_ADDR 环境变量")
        print("💡 示例: export EXO_FRP_SERVER_ADDR=100.x.x.x")
        return
    
    from exo.networking.frp.frp_auto_manager import initialize_frp_auto, get_frp_manager
    
    success = await initialize_frp_auto(
        server_addr=server_addr,
        local_port=50051,
        token="my-secure-token",
        force_mode='client',
    )
    
    if success:
        manager = get_frp_manager()
        status = manager.get_status()
        
        print("\n✅ FRP 客户端已连接！")
        print(f"   本地端口: {status['local_port']} -> 远程: {server_addr}:{status['local_port']}")
        print(f"\n💡 现在 Windows 可以通过 {server_addr}:50051 访问此节点")
        
        try:
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            await manager.stop()
    else:
        print("❌ FRP 客户端启动失败")


async def demo_tailscale_with_frp_fallback():
    """
    演示 3: Tailscale + FRP 自动回退
    
    工作流程：
    1. 首先尝试 Tailscale 直连
    2. 如果直连失败，自动启用 FRP 回退
    3. 对用户完全透明
    """
    print("=" * 70)
    print("  场景 3: Tailscale + FRP 自动回退 (推荐)")
    print("=" * 70)
    
    # 设置环境变量以启用 FRP 自动回退
    os.environ['EXO_USE_FRP'] = 'true'
    os.environ['EXO_FRP_SERVER_ADDR'] = '100.x.x.x'  # 替换为实际的服务端地址
    
    print("\n📋 配置:")
    print("   EXO_USE_FRP=true (启用 FRP 回退)")
    print(f"   EXO_FRP_SERVER_ADDR={os.environ['EXO_FRP_SERVER_ADDR']}")
    print("\n⚙️  启动 exo 时会:")
    print("   1. 尝试 Tailscale P2P 直连")
    print("   2. 失败后自动启动 FRP 客户端")
    print("   3. 通过 FRP 转发建立连接")
    print("\n💡 命令:")
    print("   python -m exo.main --disable-tui --discovery-module tailscale --node-port 50051")


def show_usage():
    """显示使用说明"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              exo FRP 集成 - 快速开始指南                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🚀 快速部署步骤                                                  ║
║  ─────────────────────────────────────────────────────          ║
║                                                                  ║
║  【Windows 端】(有公网 IP 或 Tailscale IP)                       ║
║  ─────────────────────────────────────────                      ║
║  1. 安装依赖:                                                    ║
║     pip install requests toml tqdm PySocks                       ║
║                                                                  ║
║  2. 启动 FRP 服务端 + exo:                                       ║
║     set EXO_FRP_MODE=server                                      ║
║     python -m exo.main --disable-tui \\                          ║
║         --discovery-module tailscale \\                          ║
║         --node-port 50051                                        ║
║                                                                  ║
║  【Linux/Docker 端】(NAT 后)                                     ║
║  ─────────────────────────────────────────                      ║
║  1. 设置环境变量:                                                ║
║     export EXO_USE_FRP=true                                      ║
║     export EXO_FRP_SERVER_ADDR=<Windows的IP>                     ║
║     export EXO_FRP_TOKEN=my-secure-token                         ║
║                                                                  ║
║  2. 启动 exo (FRP 会自动启动):                                   ║
║     python -m exo.main --disable-tui \\                          ║
║         --discovery-module tailscale \\                          ║
║         --node-port 50051                                        ║
║                                                                  ║
║  📊 架构图                                                       ║
║  ─────────────────────────────────────────                      ║
║                                                                  ║
║     [Linux Node]                    [Windows Node]               ║
║         |                                |                        ║
║         |  (Tailscale 直连失败)           |                        ║
║         v                                v                        ║
║     [frpc :50051] ----TCP----> [frps :7000]                      ║
║                                  |                               ║
║                                  v                               ║
║                            [gRPC Server]                         ║
║                                                                  ║
║  🔧 高级配置                                                      ║
║  ─────────────────────────────────────────                      ║
║                                                                  ║
║  环境变量:                                                        ║
║  • EXO_USE_FRP=true|false      (强制启用/禁用 FRP)                ║
║  • EXO_FRP_SERVER_ADDR=x.x.x.x (FRP 服务端地址)                  ║
║  • EXO_FRP_PORT=7000            (FRP 服务端端口)                  ║
║  • EXO_FRP_TOKEN=xxx            (认证令牌)                        ║
║  • EXO_FRP_MODE=server|client   (强制模式)                        ║
║                                                                  ║
║  🎯 推荐组合                                                     ║
║  ─────────────────────────────────────────                      ║
║                                                                  ║
║  ✅ 最佳方案: Tailscale (设备发现) + FRP (网络穿透)                ║
║     - Tailscale 用于获取节点列表                                  ║
║     - FRP 用于实际数据传输（当 P2P 不可用时）                     ║
║                                                                  ║
║  ✅ 简化方案: 仅使用 FRPDiscovery                                 ║
║     python -m exo.main --disable-tui \\                          ║
║         --discovery-module frp \\                                ║
║         --frp-server-addr x.x.x.x \\                             ║
║         --node-port 50051                                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == '--help':
        show_usage()
        sys.exit(0)
    
    mode = sys.argv[1].lower()
    
    if mode == 'server':
        asyncio.run(demo_frp_server_mode())
    elif mode == 'client':
        asyncio.run(demo_frp_client_mode())
    elif mode == 'demo':
        asyncio.run(demo_tailscale_with_frp_fallback())
    else:
        print(f"❌ 未知模式: {mode}")
        print("💡 用法: python frp_demo.py [server|client|demo|--help]")
        sys.exit(1)
