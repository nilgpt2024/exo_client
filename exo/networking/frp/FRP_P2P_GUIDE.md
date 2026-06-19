# FRP XTCP P2P 模式使用指南

## 🚀 核心特性

**默认启用 XTCP P2P 模式**，实现高性能直连：

- ✅ **优先 P2P 直连**：低延迟、高带宽、无中转瓶颈
- ✅ **自动回退 TCP 中转**：当 P2P 打洞失败时，自动切换到服务端中转
- ✅ **零配置**：开箱即用，无需手动配置 secretKey
- ✅ **加密传输**：传输层 AES 加密 + 压缩

---

## 📊 工作原理

```
[Node A]                    [FRPS Server]                    [Node B]
   |                             |                              |
   |  1. 连接 frps 注册          |                              |
   | <------------------------- |                              |
   |                             |                              |
   |  2. 请求连接 Node B        |                              |
   | -------------------------> |                              |
   |                             | --------------------------> |
   |                             |                              |
   |  3. P2P 打洞 (NAT 穿透)     |                              |
   | <------------------------> | <-------------------------> |
   |                             |                              |
   |  4. ✅ 建立 P2P 直连！      | (不再参与数据转发)            |
   | <=========================> |                              |
   
   如果打洞失败：
   |  5. 回退到 TCP 中转         |                              |
   | <--------------------->    | --------------------->       |
```

---

## 🎯 适用场景

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| 两台公网机器 | XTCP P2P | 直接建立连接，无需中转 |
| 一台公网 + 一台 NAT 后 | XTCP P2P | 尝试打洞，失败则中转 |
| 两台都在 NAT 后 | XTCP P2P | 大部分情况可成功打洞 |
| 严格防火墙环境 | TCP 中转 | 自动回退到服务端转发 |

---

## ⚙️ 配置示例

### 服务端配置 (frps.toml)

```toml
bindPort = 7000

# XTCP P2P 必需的传输层配置
[transport]
useEncryption = true      # 启用加密（XTCP 必需）
useCompression = true     # 启用压缩（减少带宽）

# 认证配置
[auth]
token = "your-secure-token"

# 可选：Web Dashboard
[webServer]
addr = "0.0.0.0"
port = 7500
user = "admin"
password = "admin123"
```

### 客户端配置 (frpc.toml)

```toml
serverAddr = "your-server-ip"
serverPort = 7000

# 认证
[auth]
token = "your-secure-token"

# 代理 1: XTCP P2P（优先使用）
[[proxies]]
name = "exo_p2p_mynode"
type = "xtcp"
secretKey = "auto-generated-by-node-id"  # 自动生成
localIP = "127.0.0.1"
localPort = 50051

# 代理 2: TCP 中转（备用）
[[proxies]]
name = "exo_fallback_mynode"
type = "tcp"
localIP = "127.0.0.1"
localPort = 50051
remotePort = 50051
```

---

## 🚀 快速启动

### Windows 端（作为 FRP Server）

```powershell
python -m exo.main --disable-tui --discovery-module tailscale --node-port 50051
```

**输出示例**：
```
[FRP-Server] ✅ frps 运行中 (PID: 12345)
[FRP-Server] 📍 监听端口: 7000
[FRP-Server] 🔐 传输加密: 已启用 (XTCP P2P 必需)
[FRP-Server] 🌐 客户端连接地址: 100.90.182.17:7000
```

### Linux/Docker 端（作为 FRP Client）

```bash
export EXO_USE_FRP=true
export EXO_FRP_SERVER_ADDR=100.90.182.17  # Windows 的 Tailscale IP
export EXO_FRP_TOKEN=your-secure-token

python -m exo.main --disable-tui --discovery-module tailscale --node-port 50051
```

**输出示例**：
```
[FRP-Client] ✅ frpc 运行成功 (XTCP P2P 模式)
[FRP-Client] 🔗 连接模式: P2P 直连优先，TCP 中转备用
[FRP-Client] 💡 提示: 首次连接会尝试 P2P 打洞
```

---

## 🔍 日志解读

### P2P 打洞成功

```
[FRP] [info] [xxx_xtcp_xxx] start a new connection [lan]
[FRP] [info] [xxx_xtcp_xxx] successfully connected to address [x.x.x.x:xxxxx] (through NAT hole punching)
```

**含义**：成功建立 P2P 直连，后续流量不经过服务器。

### P2P 打洞失败，回退到中转

```
[FRP] [info] [xxx_xtcp_xxx] start a new connection [fallback to server relay]
[FRP] [info] [xxx_fallback_xxx] proxy registered successfully
```

**含义**：P2P 打洞失败，已切换到 TCP 中转模式。

---

## 🛠️ 高级配置

### 禁用 P2P 模式（纯 TCP 中转）

```bash
# 在 exo 启动前设置
export EXO_FRP_DISABLE_P2P=true
```

或修改代码：
```python
config = self.config_manager.generate_frpc_config(
    ...,
    enable_p2p=False,  # 禁用 P2P
)
```

### 自定义 SecretKey

默认情况下，secretKey 基于 `node_id` 自动生成。如需自定义：

```bash
export EXO_FRP_SECRET_KEY=my-custom-secret-key
```

---

## 📈 性能对比

| 指标 | XTCP P2P | TCP 中转 | Tailscale DERP |
|------|----------|----------|----------------|
| **延迟** | ~5-20ms | ~30-100ms | ~50-150ms |
| **带宽** | 无限制 | 受限于服务器带宽 | 受限于 DERP 节点 |
| **稳定性** | 取决于网络 | ★★★★★ | ★★★★☆ |
| **适用场景** | 实时推理、大模型传输 | 备用方案 | 跨云服务商 |

---

## ❓ 常见问题

### Q: 为什么我的节点没有建立 P2P 连接？

**可能原因**：
1. 双方都在严格 NAT/防火墙后（对称型 NAT）
2. UDP 端口被阻止（XTCP 使用 UDP 打洞）
3. 网络运营商禁止 P2P 流量

**解决方案**：
- 系统会自动回退到 TCP 中转，无需手动干预
- 或禁用 P2P：`export EXO_FRP_DISABLE_P2P=true`

### Q: 如何确认当前是 P2P 还是中转模式？

查看 frpc/frps 日志中的 `[lan]` vs `[relay]` 关键字。

### Q: XTCP 和 STCP 有什么区别？

- **XTCP**：真正的 P2P（NAT 穿透），性能最优
- **STCP**：通过服务端转发，但可控制访问权限

推荐使用 **XTCP**。

---

## 📚 相关文档

- [FRP 官方文档 - XTCP](https://gofrp.org/docs/features/xtcp/)
- [NAT 穿透原理](https://en.wikipedia.org/wiki/NAT_traversal)
- [FRP 配置参考](https://gofrp.com/docs/reference/configuration/)
