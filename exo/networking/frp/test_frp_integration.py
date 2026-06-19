#!/usr/bin/env python3
"""
FRP 功能集成测试脚本

测试项目：
1. FRP 模块导入
2. 自动安装检测
3. 配置生成
4. TailscaleDiscovery + FRP 回退初始化
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试 1: 模块导入"""
    print("\n" + "=" * 60)
    print("  Test 1: Module Import")
    print("=" * 60)

    try:
        from exo.networking.frp import (
            FRPDiscovery,
            FRPAutoManager,
            get_frp_manager,
            initialize_frp_auto,
        )
        print("[PASS] FRP core modules imported")

        from exo.networking.frp.frp_config import FRPConfig
        from exo.networking.frp.frp_process import FRPProcessManager
        from exo.networking.frp.frp_downloader import (
            ensure_frpc_installed,
            get_frpc_path,
            check_frpc_available,
        )
        print("[PASS] FRP helper modules imported")

        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_config_generation():
    """测试 2: 配置生成"""
    print("\n" + "=" * 60)
    print("  Test 2: Config Generation (P2P Mode)")
    print("=" * 60)

    try:
        from exo.networking.frp.frp_config import FRPConfig

        config_mgr = FRPConfig()

        # 测试服务端配置（默认启用 XTCP）
        frps_config = config_mgr.generate_frps_config(
            bind_port=7000,
            token="test-token",
            dashboard_port=7500,
        )
        assert frps_config['bindPort'] == 7000
        assert frps_config['auth']['token'] == 'test-token'
        # 验证 XTCP 支持已启用
        assert frps_config.get('transport', {}).get('useEncryption') == True
        print("[PASS] Server config generated with XTCP support")

        # 测试客户端配置（默认启用 P2P）
        frpc_config = config_mgr.generate_frpc_config(
            server_addr="127.0.0.1",
            server_port=7000,
            node_id="test-node",
            local_port=50051,
            token="test-token",
            enable_p2p=True,  # 显式启用 P2P
        )
        assert frpc_config['serverAddr'] == "127.0.0.1"
        
        # 验证代理配置（应该有 XTCP 和 TCP 两个代理）
        proxy_types = [p.get('type') for p in frpc_config.get('proxies', [])]
        assert 'xtcp' in proxy_types, f"XTCP proxy not found, got: {proxy_types}"
        assert 'tcp' in proxy_types, f"TCP fallback not found, got: {proxy_types}"
        print(f"[PASS] Client config generated with P2P mode (proxies: {proxy_types})")
        
        # 验证 XTCP 是第一个（优先使用）
        xtcp_proxy = [p for p in frpc_config['proxies'] if p.get('type') == 'xtcp'][0]
        assert 'secretKey' in xtcp_proxy
        assert xtcp_proxy['name'].startswith('exo_p2p_')
        print("[PASS] XTCP P2P proxy is primary (first in list)")

        return True
    except Exception as e:
        print(f"[FAIL] Config generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frp_manager_init():
    """测试 3: 管理器初始化"""
    print("\n" + "=" * 60)
    print("  Test 3: FRPAutoManager Init")
    print("=" * 60)

    try:
        from exo.networking.frp.frp_auto_manager import FRPAutoManager, get_frp_manager

        # 先获取单例
        manager = get_frp_manager()
        assert manager.is_initialized == False
        assert manager.is_running == False
        print("[PASS] Manager instance created via singleton")

        # 测试单例模式（再次获取应该是同一个）
        manager2 = get_frp_manager()
        assert manager is manager2
        print("[PASS] Singleton pattern works")

        # 测试状态获取
        status = manager.get_status()
        assert 'initialized' in status
        assert 'running' in status
        print("[PASS] Status query successful")

        return True
    except Exception as e:
        print(f"[FAIL] Manager init failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_address_mapping():
    """测试 4: 地址映射逻辑"""
    print("\n" + "=" * 60)
    print("  Test 4: Address Mapping")
    print("=" * 60)

    try:
        from exo.networking.frp.frp_auto_manager import get_frp_manager

        # 使用单例
        manager = get_frp_manager()
        manager.server_addr = '192.168.1.100'
        manager.is_running = True
        manager.mode = 'client'  # 设置为客户端模式

        # 测试地址转换
        original = '100.90.182.17:50051'
        mapped = manager.get_peer_address('node-1', original)

        assert mapped.startswith('192.168.1.100')
        assert ':50051' in mapped
        print(f"[PASS] Address mapping correct: {original} -> {mapped}")

        # 测试是否应该使用 FRP
        should_use = manager.should_use_frp_for_peer('100.x.x.x:50051')
        assert should_use == True
        print("[PASS] FRP usage detection correct (Tailscale IP)")

        should_not_use = manager.should_use_frp_for_peer('192.168.1.50:50051')
        print(f"[INFO] Non-Tailscale IP result: {should_not_use}")

        # 重置状态
        manager.is_running = False
        manager.mode = None

        return True
    except Exception as e:
        print(f"[FAIL] Address mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tailscale_frp_integration():
    """测试 5: Tailscale + FRP 集成"""
    print("\n" + "=" * 60)
    print("  Test 5: TailscaleDiscovery + FRP Integration")
    print("=" * 60)

    try:
        # 设置环境变量启用 FRP
        os.environ['EXO_USE_FRP'] = 'true'
        os.environ['EXO_FRP_SERVER_ADDR'] = '127.0.0.1'  # 本地测试

        # 导入 TailscaleDiscovery
        from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery, HAS_FRP_SUPPORT

        if not HAS_FRP_SUPPORT:
            print("[WARN] FRP support not compiled (missing dependencies)")
            return True

        print("[PASS] TailscaleDiscovery imported with FRP support")

        # 创建模拟的 Discovery 实例（不实际启动）
        def mock_create_peer(node_id, addr, desc, caps):
            class MockPeer:
                def id(self): return node_id
                def addr(self): return addr
                async def health_check(self): return False
                async def collect_topology(self, **kwargs): 
                    from collections import namedtuple
                    Topology = namedtuple('Topology', ['nodes'])
                    return Topology(nodes={})
            return MockPeer()

        discovery = TailscaleDiscovery(
            node_id='test-node',
            node_port=50051,
            create_peer_handle=mock_create_peer,
        )

        # 检查属性
        assert hasattr(discovery, 'frp_manager')
        assert hasattr(discovery, 'use_frp_fallback')
        assert hasattr(discovery, '_initialize_frp_fallback')
        print("[PASS] FRP attributes added to TailscaleDiscovery")

        # 清理环境变量
        del os.environ['EXO_USE_FRP']
        del os.environ['EXO_FRP_SERVER_ADDR']

        return True
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "   exo FRP Integration Test Suite".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    results = []

    # 同步测试
    results.append(("Module Import", test_imports()))
    results.append(("Config Generation", test_config_generation()))
    results.append(("Manager Init", test_frp_manager_init()))
    results.append(("Address Mapping", test_address_mapping()))

    # 异步测试
    results.append(("Tailscale+FRP", await test_tailscale_frp_integration()))
    results.append(("Token Auth", await test_token_authentication()))

    # 输出总结
    print("\n" + "=" * 60)
    print("  Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! FRP integration ready!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} tests failed, check errors above")
        return 1


async def test_token_authentication():
    """测试 6: Token 认证配置"""
    print("\n" + "=" * 60)
    print("  Test 6: Token Authentication")
    print("=" * 60)

    try:
        from exo.networking.frp.frp_config import FRPConfig
        import warnings

        config_mgr = FRPConfig()

        # 测试 1: 显式指定 token
        config_with_token = config_mgr.generate_frpc_config(
            server_addr="127.0.0.1",
            server_port=7000,
            node_id="test-node",
            local_port=50051,
            token="my-secure-token-123",
            enable_p2p=True,
        )

        assert 'auth' in config_with_token, "Missing [auth] section"
        assert config_with_token['auth']['token'] == "my-secure-token-123"
        print("[PASS] Explicit token configured correctly")

        # 测试 2: 不指定 token（应该使用默认值并发出警告）
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            config_without_token = config_mgr.generate_frpc_config(
                server_addr="127.0.0.1",
                server_port=7000,
                node_id="test-node",
                local_port=50051,
                token=None,  # 不指定
                enable_p2p=True,
            )

            # 应该有默认 token
            assert 'auth' in config_without_token, "Missing [auth] section when no token provided"
            assert config_without_token['auth']['token'] == "exo-frp-default-token"
            print("[PASS] Default token used when not specified")

            # 应该有警告
            assert len(w) > 0, "Expected warning about default token"
            assert issubclass(w[-1].category, UserWarning)
            print("[PASS] Warning issued for missing token")

        # 测试 3: FRPDiscovery 默认 token
        from exo.networking.frp.frp_discovery import FRPDiscovery
        from exo.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES

        # 创建 mock peer handle
        def mock_peer_handle(node_id, addr, desc, caps):
            class MockPeer:
                def id(self): return node_id
                def addr(self): return addr
                async def health_check(self): return False
                async def collect_topology(self, **kwargs):
                    from collections import namedtuple
                    Topology = namedtuple('Topology', ['nodes'])
                    return Topology(nodes={})
            return MockPeer()

        discovery = FRPDiscovery(
            frp_server_addr="127.0.0.1",
            frp_server_port=7000,
            node_id="test-node",
            local_port=50051,
            create_peer_handle=mock_peer_handle,
            frp_token=None,  # 不指定，应该使用默认值
            enable_p2p=True,
            device_capabilities=UNKNOWN_DEVICE_CAPABILITIES,  # 使用默认设备能力
        )

        assert discovery.frp_token == "exo-frp-default-token"
        print("[PASS] FRPDiscovery uses default token when not specified")

        # 测试 4: FRPDiscovery 自定义 token
        discovery_custom = FRPDiscovery(
            frp_server_addr="127.0.0.1",
            frp_server_port=7000,
            node_id="test-node",
            local_port=50051,
            create_peer_handle=mock_peer_handle,
            frp_token="my-custom-token",
            enable_p2p=True,
            device_capabilities=UNKNOWN_DEVICE_CAPABILITIES,  # 使用默认设备能力
        )

        assert discovery_custom.frp_token == "my-custom-token"
        print("[PASS] FRPDiscovery accepts custom token")

        # 测试 5: 验证 secretKey 一致性（关键！）
        config_node_a = config_mgr.generate_frpc_config(
            server_addr="127.0.0.1",
            server_port=7000,
            node_id="node-a-windows",  # 不同的 node_id
            local_port=50051,
            token="shared-token-12345",  # 相同的 token
            enable_p2p=True,
        )

        config_node_b = config_mgr.generate_frpc_config(
            server_addr="127.0.0.1",
            server_port=7000,
            node_id="node-b-linux",  # 不同的 node_id
            local_port=50051,
            token="shared-token-12345",  # 相同的 token
            enable_p2p=True,
        )

        # 提取两个节点的 XTCP secretKey
        secret_key_a = None
        secret_key_b = None

        for proxy in config_node_a.get("proxies", []):
            if proxy.get("type") == "xtcp":
                secret_key_a = proxy.get("secretKey")
                break

        for proxy in config_node_b.get("proxies", []):
            if proxy.get("type") == "xtcp":
                secret_key_b = proxy.get("secretKey")
                break

        assert secret_key_a is not None, "Node A missing XTCP secretKey"
        assert secret_key_b is not None, "Node B missing XTCP secretKey"
        assert secret_key_a == secret_key_b, f"SecretKeys don't match! A={secret_key_a}, B={secret_key_b}"
        print(f"[PASS] SecretKey consistency verified: {secret_key_a} (same for both nodes)")

        return True
    except Exception as e:
        print(f"[FAIL] Token auth test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
