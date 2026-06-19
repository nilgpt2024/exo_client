import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False


class FRPConfig:
    """frp 配置管理器"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".exo" / "frp"
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def get_frpc_config_path(self, node_id: str) -> Path:
        """获取 frpc 配置文件路径"""
        if HAS_TOML:
            return self.config_dir / f"frpc_{node_id}.toml"
        else:
            return self.config_dir / f"frpc_{node_id}.ini"

    def get_frps_config_path(self) -> Path:
        """获取 frps 配置文件路径"""
        if HAS_TOML:
            return self.config_dir / "frps.toml"
        else:
            return self.config_dir / "frps.json"
    
    def generate_frps_config(
        self,
        bind_port: int = 7000,
        vhost_http_port: Optional[int] = None,
        vhost_https_port: Optional[int] = None,
        dashboard_port: Optional[int] = None,
        dashboard_user: Optional[str] = None,
        dashboard_pwd: Optional[str] = None,
        token: Optional[str] = None,
        enable_xtcp: bool = True,  # 默认启用 XTCP P2P 支持
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成 frps 服务端配置（支持 XTCP P2P 模式）

        Args:
            bind_port: 服务端监听端口
            vhost_http_port: HTTP 虚拟主机端口
            vhost_https_port: HTTPS 虚拟主机端口
            dashboard_port: Dashboard 端口
            dashboard_user: Dashboard 用户名
            dashboard_pwd: Dashboard 密码
            token: 认证令牌
            enable_xtcp: 是否启用 XTCP P2P 支持（默认 True）
        """
        config = {
            "bindPort": bind_port,
        }
        
        # 启用传输层加密和压缩（XTCP P2P 必需）
        if enable_xtcp:
            config["transport"] = {
                "useEncryption": True,
                "useCompression": True,
            }
        
        if vhost_http_port:
            config["vhostHTTPPort"] = vhost_http_port
        if vhost_https_port:
            config["vhostHTTPSPort"] = vhost_https_port
        if dashboard_port:
            config["webServer"] = {
                "addr": "0.0.0.0",
                "port": dashboard_port,
            }
            if dashboard_user:
                config["webServer"]["user"] = dashboard_user
            if dashboard_pwd:
                config["webServer"]["password"] = dashboard_pwd
        if token:
            config["auth"] = {
                "token": token
            }
        
        return config
    
    def generate_frpc_config(
        self,
        server_addr: str,
        server_port: int,
        node_id: str,
        local_port: int,
        remote_port: Optional[int] = None,
        token: Optional[str] = None,
        enable_p2p: bool = True,  # 默认启用 P2P (XTCP)
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成 frpc 客户端配置（优先使用 XTCP P2P 直连，TCP 中转作为备用）

        XTCP (eXtended TCP) 工作原理：
        - 两端都连接到 frps 服务端进行握手
        - 尝试建立 P2P 直连（打洞）
        - 如果 P2P 失败，自动回退到服务端中转

        Args:
            server_addr: FRP 服务端地址
            server_port: FRP 服务端端口
            node_id: 节点唯一标识（用于生成 secretKey）
            local_port: 本地 gRPC 服务端口
            remote_port: TCP 中转的远程端口（可选）
            token: 认证令牌
            enable_p2p: 是否启用 XTCP P2P 模式（默认 True）
        """
        # 如果未指定远程端口，自动生成一个
        if remote_port is None:
            # 使用 node_id 的哈希值生成端口，范围 30000-50000
            import hashlib
            hash_val = int(hashlib.md5(node_id.encode()).hexdigest()[:8], 16)
            remote_port = 30000 + (hash_val % 20000)

        # 🔐 生成 secret key（用于 XTCP P2P 认证）
        # 重要：必须使用与 frps 服务端一致的密钥，否则无法建立 P2P 连接
        # 方案：使用 FRP token 作为 secretKey，确保所有节点使用相同的密钥
        if not token:
            # 如果没有提供 token，使用默认值并警告
            import warnings
            warnings.warn(
                "FRP token not specified! Using default token for XTCP P2P. "
                "For production use, please provide a secure token via --frp-token parameter.",
                UserWarning,
                stacklevel=2
            )
            token = "exo-frp-default-token"

        # ⚠️ 检测特殊字符（PowerShell 兼容性）
        if '$' in token:
            import warnings
            warnings.warn(
                f"Token contains '$' character which may be interpreted as a variable in PowerShell. "
                f"Actual token length: {len(token)} characters. "
                f"If using PowerShell, consider using single quotes: --frp-token '{token}'",
                UserWarning,
                stacklevel=2
            )

        # ✅ 使用 token 的哈希作为 secretKey（保证一致性且安全性）
        import hashlib
        secret_key = hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]

        config = {
            "serverAddr": server_addr,
            "serverPort": server_port,
            "proxies": []
        }
        
        # 优先添加 XTCP P2P 代理（高性能直连）
        if enable_p2p:
            config["proxies"].append({
                "name": f"exo_p2p_{node_id}",
                "type": "xtcp",
                "secretKey": secret_key,
                "localIP": "127.0.0.1",
                "localPort": local_port,
            })
        
        # 添加 TCP 代理作为备用方案（当 P2P 打洞失败时自动回退）
        config["proxies"].append({
            "name": f"exo_fallback_{node_id}",
            "type": "tcp",
            "localIP": "127.0.0.1",
            "localPort": local_port,
            "remotePort": remote_port,
        })

        # 🔐 Token 认证配置（token 已在前面验证并设置默认值）
        config["auth"] = {
            "token": token
        }

        return config
    
    def save_config(self, config: Dict[str, Any], config_path: Path) -> bool:
        """保存配置到文件（TOML / JSON 格式）"""
        try:
            if HAS_TOML:
                with open(config_path, "w", encoding="utf-8") as f:
                    toml.dump(config, f)
                print(f"配置已保存 (TOML): {config_path}")
            elif str(config_path).endswith(".json"):
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"配置已保存 (JSON): {config_path}")
            else:
                ini_content = self._dict_to_ini(config)
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(ini_content)
                print(f"配置已保存 (INI): {config_path}")
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def _dict_to_ini(self, config: Dict[str, Any]) -> str:
        """将字典转换为 frpc 兼容的 INI 格式字符串

        FRP 标准 INI 格式（参考官方文档 https://github.com/fatedier/frp）：
        [common]
        server_addr = "x.x.x.x"
        server_port = 7000
        token = "xxx"              ← token 直接在 [common] 内！

        [proxy_name]               ← 每个 proxy 用名称作为段头
        type = xtcp
        local_ip = 127.0.0.1
        ...
        """
        lines = []

        # ---- [common] 段 ----
        lines.append("[common]")
        if "serverAddr" in config:
            lines.append(f'server_addr = "{config["serverAddr"]}"')
        if "serverPort" in config:
            lines.append(f"server_port = {config['serverPort']}")

        # 🔑 Token 直接放在 [common] 段内（FRP 标准 INI 格式要求）
        if "auth" in config and "token" in config["auth"]:
            lines.append(f'token = "{config["auth"]["token"]}"')

        # ---- 各 proxy 用独立 [name] 段 ----
        if "proxies" in config:
            for proxy in config["proxies"]:
                name = proxy.get("name", "unnamed_proxy")
                lines.append("")
                lines.append(f"[{name}]")
                if "type" in proxy:
                    lines.append(f'type = "{proxy["type"]}"')
                if "localIP" in proxy:
                    lines.append(f'local_ip = "{proxy["localIP"]}"')
                if "localPort" in proxy:
                    lines.append(f"local_port = {proxy['localPort']}")
                if "remotePort" in proxy:
                    lines.append(f"remote_port = {proxy['remotePort']}")
                if "secretKey" in proxy:
                    lines.append(f'secret_key = "{proxy["secretKey"]}"')

        return "\n".join(lines) + "\n"
    
    def load_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """从文件加载配置"""
        try:
            if not config_path.exists():
                print(f"配置文件不存在: {config_path}")
                return None
            
            if HAS_TOML and str(config_path).endswith(".toml"):
                with open(config_path, "r", encoding="utf-8") as f:
                    return toml.load(f)
            elif str(config_path).endswith(".ini"):
                return self._ini_to_dict(config_path)
            else:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载配置失败: {e}")
            return None
    
    def _ini_to_dict(self, config_path: Path) -> Dict[str, Any]:
        """从 INI 文件加载配置（兼容 frpc 标准 INI 格式）

        支持格式：
        [common]
        server_addr = "x.x.x.x"

        [auth]
        token = "xxx"

        [proxy_name]          ← 每个 proxy 是独立段
        type = xtcp
        ...
        """
        config = {}
        current_section = None
        current_proxy = None

        # INI snake_case → 内部 camelCase 键名映射
        key_map = {
            "server_addr": "serverAddr",
            "server_port": "serverPort",
            "local_ip": "localIP",
            "local_port": "localPort",
            "remote_port": "remotePort",
            "secret_key": "secretKey",
        }

        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue

                # [section_name] — 段头（包括 proxy 名称段）
                if line.startswith("[") and line.endswith("]"):
                    section_name = line[1:-1]

                    # [[proxies]] 旧格式兼容
                    if line.startswith("[["):
                        if "proxies" not in config:
                            config["proxies"] = []
                        proxy = {}
                        config["proxies"].append(proxy)
                        current_section = "proxies"
                        current_proxy = proxy
                    elif section_name == "common":
                        config["common"] = {}
                        current_section = "common"
                        current_proxy = None
                    elif section_name == "auth":
                        config["auth"] = {}
                        current_section = "auth"
                        current_proxy = None
                    else:
                        # 其他段名视为 proxy 定义
                        if "proxies" not in config:
                            config["proxies"] = []
                        proxy = {"name": section_name}
                        config["proxies"].append(proxy)
                        current_section = "proxy"
                        current_proxy = proxy

                elif "=" in line and current_section:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # 去引号
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    elif value.isdigit():
                        value = int(value)

                    # 键名映射
                    mapped_key = key_map.get(key, key)

                    # 🔑 特殊处理：[common] 中的 token → 内部 auth.token
                    if current_section == "common" and key == "token":
                        config.setdefault("auth", {})["token"] = value
                    elif current_section == "proxies" and current_proxy is not None:
                        current_proxy[mapped_key] = value
                    elif current_section == "proxy" and current_proxy is not None:
                        current_proxy[mapped_key] = value
                    else:
                        config.setdefault(current_section, {})[mapped_key] = value

        return config
    
    def save_frps_config(self, config: Dict[str, Any]) -> bool:
        """保存 frps 配置"""
        return self.save_config(config, self.get_frps_config_path())
    
    def save_frpc_config(self, config: Dict[str, Any], node_id: str) -> bool:
        """保存 frpc 配置"""
        self._cleanup_old_config_files(node_id)
        return self.save_config(config, self.get_frpc_config_path(node_id))
    
    def _cleanup_old_config_files(self, node_id: str):
        """清理旧格式的配置文件"""
        for ext in ['.toml', '.json', '.ini']:
            old_file = self.config_dir / f"frpc_{node_id}{ext}"
            if old_file.exists():
                try:
                    old_file.unlink()
                    print(f"[FRP] 已清理旧配置文件: {old_file}")
                except Exception as e:
                    print(f"[FRP] 清理旧配置文件失败: {e}")
