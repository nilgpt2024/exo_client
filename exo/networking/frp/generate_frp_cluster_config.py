#!/usr/bin/env python3
"""
FRP 多节点部署配置生成器
========================

用于生成多个 exo 节点的统一 FRP 配置，确保：
1. 所有节点的 token 完全一致
2. secretKey 基于相同 token 生成（保证 P2P 可用）
3. 自动生成 seed_peers 配置
4. 避免 PowerShell 特殊字符问题

使用方法:
    python generate_frp_cluster_config.py --nodes "windows@119.45.114.133,linux@10.0.0.1" --token "your-token"
"""

import argparse
import hashlib
import json
import sys
from typing import List, Dict, Tuple


def generate_token() -> str:
    """生成安全的随机 token"""
    import secrets
    return secrets.token_urlsafe(32)


def calculate_secret_key(token: str) -> str:
    """根据 token 计算 XTCP secretKey"""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]


def calculate_remote_port(node_id: str) -> int:
    """根据 node_id 计算远程端口"""
    hash_val = int(hashlib.md5(node_id.encode()).hexdigest()[:8], 16)
    return 30000 + (hash_val % 20000)


def parse_nodes(nodes_str: str) -> List[Dict]:
    """解析节点列表"""
    nodes = []
    for node_str in nodes_str.split(","):
        node_str = node_str.strip()
        if not node_str:
            continue

        # 格式: node_id@address 或 node_id@address:port
        if "@" in node_str:
            name_part, addr_part = node_str.split("@", 1)
            node_id = name_part.strip()

            if ":" in addr_part:
                address, port = addr_part.rsplit(":", 1)
                port = int(port.strip())
            else:
                address = addr_part.strip()
                port = None  # 使用默认端口
        else:
            node_id = f"node-{len(nodes)}"
            address = node_str
            port = None

        nodes.append({
            "node_id": node_id,
            "address": address,
            "port": port,
        })

    return nodes


def generate_node_commands(
    nodes: List[Dict],
    token: str,
    server_addr: str,
    server_port: int,
    local_port: int,
    enable_p2p: bool = True,
) -> Dict[str, str]:
    """为每个节点生成启动命令"""

    # 计算统一的 secretKey
    secret_key = calculate_secret_key(token)

    # 生成所有节点的 seed_peers 列表（排除自己）
    commands = {}

    for node in nodes:
        node_id = node["node_id"]

        # 构建其他节点的 seed_peers 字符串
        other_nodes = [
            f"{n['node_id']}@{n['address']}:{n['port'] or local_port}"
            for n in nodes
            if n["node_id"] != node_id
        ]
        seed_peers = ",".join(other_nodes) if other_nodes else ""

        # 计算当前节点的 remote_port
        remote_port = calculate_remote_port(node_id)

        # 生成命令（PowerShell 兼容格式）
        if sys.platform == "win32":
            # Windows: 使用单引号避免 $ 被解析
            cmd = (
                f'python -m exo.main --disable-tui '
                f'--discovery-module frp '
                f'--node-id {node_id} '
                f'--node-port {local_port} '
                f'--frp-server-addr {server_addr} '
                f'--frp-server-port {server_port} '
                f"--frp-token '{token}' "
                f'--enable-p2p ' if enable_p2p else ''
            )

            if seed_peers:
                cmd += f'--seed-peers "{seed_peers}" '

            cmd = cmd.strip()
        else:
            # Linux/Mac: 正常双引号
            cmd_parts = [
                'python -m exo.main',
                '--disable-tui',
                '--discovery-module frp',
                f'--node-id {node_id}',
                f'--node-port {local_port}',
                f'--frp-server-addr {server_addr}',
                f'--frp-server-port {server_port}',
                f'--frp-token "{token}"',
            ]

            if enable_p2p:
                cmd_parts.append('--enable-p2p')

            if seed_peers:
                cmd_parts.append(f'--seed-peers "{seed_peers}"')

            cmd = ' \\\n    '.join(cmd_parts)

        commands[node_id] = {
            "command": cmd,
            "remote_port": remote_port,
            "secret_key": secret_key,
            "seed_peers": seed_peers,
        }

    return commands


def print_config_summary(
    nodes: List[Dict],
    token: str,
    server_addr: str,
    server_port: int,
    commands: Dict[str, dict],
):
    """打印配置摘要"""
    print("\n" + "=" * 70)
    print("  FRP Cluster Configuration Summary")
    print("=" * 70)

    print(f"\n📌 Server Configuration:")
    print(f"   Address: {server_addr}:{server_port}")
    print(f"   Token: {token[:8]}...{token[-4:] if len(token) > 12 else ''} ({len(token)} chars)")
    print(f"   SecretKey: {commands[list(commands.keys())[0]]['secret_key']}")
    print(f"   Nodes: {len(nodes)}")

    print(f"\n📋 Node Details:")
    print("-" * 70)
    print(f"{'Node ID':<20} {'Address':<20} {'Remote Port':<12} {'SecretKey'}")
    print("-" * 70)

    for node in nodes:
        node_id = node["node_id"]
        info = commands[node_id]
        print(f"{node_id:<20} {node['address']:<20} {info['remote_port']:<12} {info['secret_key']}")

    print("\n✅ Verification:")
    secret_keys = set(info["secret_key"] for info in commands.values())
    if len(secret_keys) == 1:
        print(f"   [PASS] All nodes use the same SecretKey: {secret_keys.pop()}")
    else:
        print(f"   [FAIL] SecretKeys are inconsistent: {secret_keys}")

    token_lengths = set()
    # 这里只是演示，实际应该检查真实传入的 token


def main():
    parser = argparse.ArgumentParser(
        description="Generate FRP cluster configuration for multiple exo nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config for 2 nodes with auto-generated token
  python %(prog)s --nodes "windows@192.168.1.100,linux@10.0.0.1"

  # Use custom token and server
  python %(prog)s --nodes "win@1.2.3.4,lin@5.6.7.8" --token "my-secret" --server 10.0.0.100 --port 7000

  # Generate and save to file
  python %(prog)s --nodes "a@ip1,b@ip2" --output cluster_config.json
        """
    )

    parser.add_argument(
        "--nodes", "-n",
        type=str,
        required=True,
        help="Comma-separated list of nodes (format: node_id@address[:port])"
    )
    parser.add_argument("--token", "-t", type=str, default=None, help="FRP authentication token")
    parser.add_argument("--server", "-s", type=str, default="127.0.0.1", help="FRP server address")
    parser.add_argument("--port", "-p", type=int, default=7000, help="FRP server port")
    parser.add_argument("--local-port", type=int, default=50051, help="Local gRPC port for all nodes")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file (JSON format)")
    parser.add_argument("--no-p2p", action="store_true", help="Disable P2P mode")

    args = parser.parse_args()

    # 解析节点列表
    try:
        nodes = parse_nodes(args.nodes)
    except Exception as e:
        print(f"[ERROR] Failed to parse nodes: {e}")
        sys.exit(1)

    if not nodes:
        print("[ERROR] No nodes specified")
        sys.exit(1)

    # 生成或使用指定的 token
    token = args.token or generate_token()

    # 为所有节点生成一致的配置
    commands = generate_node_commands(
        nodes=nodes,
        token=token,
        server_addr=args.server,
        server_port=args.port,
        local_port=args.local_port,
        enable_p2p=not args.no_p2p,
    )

    # 打印摘要
    print_config_summary(
        nodes=nodes,
        token=token,
        server_addr=args.server,
        server_port=args.port,
        commands=commands,
    )

    # 打印详细命令
    print("\n" + "=" * 70)
    print("  Startup Commands")
    print("=" * 70)

    for node_id, info in commands.items():
        print(f"\n{'─' * 70}")
        print(f"🖥️  Node: {node_id}")
        print(f"{'─' * 70}")
        print(info["command"])
        if info["seed_peers"]:
            print(f"\n   Seed Peers: {info['seed_peers']}")

    # 保存到文件（可选）
    if args.output:
        output_data = {
            "token": token,
            "server": args.server,
            "port": args.port,
            "local_port": args.local_port,
            "enable_p2p": not args.no_p2p,
            "secret_key": list(commands.values())[0]["secret_key"],
            "nodes": {
                node_id: {
                    "address": node["address"],
                    "port": node.get("port"),
                    "remote_port": info["remote_port"],
                    "command": info["command"],
                    "seed_peers": info["seed_peers"],
                }
                for node, (node_id, info) in zip(nodes, commands.items())
            }
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Configuration saved to: {args.output}")

    print("\n" + "=" * 70)
    print("  Next Steps:")
    print("=" * 70)
    print("""
1. Copy the command for each node to the corresponding machine
2. Ensure all nodes use the EXACT SAME token (check for $ characters!)
3. Start all nodes - they will automatically discover each other via seed_peers
4. Check logs for "[FRP] 新节点上线:" messages

⚠️  IMPORTANT (Windows/PowerShell):
   - Always use SINGLE QUOTES for token: --frp-token 'your$token'
   - Or escape $ with backtick: --frp-token "your`$token"
   - Verify token length is identical on all nodes
""")


if __name__ == "__main__":
    main()
