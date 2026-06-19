# FRP 完整部署指南（含 Token 认证）

## ⚠️ 重要：Token 认证必须一致！

**frps 服务端**和**所有 frpc 客户端**的 token 必须完全相同，否则连接会被拒绝。

---

## 📋 配置文件示例

### 1️⃣ FRP Server (frps.toml) - 在独立服务器上运行

```toml
# ============================================================
#  FRP Server 配置 (frps.toml)
#  部署位置: 独立云服务器 / VPS / 有公网 IP 的机器
# ============================================================

# 监听端口（客户端连接到此端口）
bindPort = 7000

# ============================================================
#  🔐 认证配置（必需）
# ============================================================
[auth]
token = "your-secure-token-here"  # ⚠️ 所有客户端必须使用相同的 token

# ============================================================
#  🛡️ 传输层加密（XTCP P2P 模式必需）
# ============================================================
[transport]
useEncryption = true      # 启用 AES 加密
useCompression = true     # 启用数据压缩

# ============================================================
#  📊 Dashboard（可选，用于监控）
# ============================================================
[webServer]
addr = "0.0.0.0"
port = 7500
user = "admin"
password = "your-dashboard-password"
```

#### 启动命令：

```bash
# 前台运行（调试用）
./frps -c frps.toml

# 后台运行（生产环境）
nohup ./frps -c frps.toml > frps.log 2>&1 &

# 使用 systemd 管理（推荐）
sudo systemctl start frps
```

---

### 2️⃣ FRP Client A - Windows (frpc_A.toml)

```toml
# ============================================================
#  FRP Client 配置 - Windows 节点
#  文件位置: ~/.exo/frp/frpc_windows-node.toml
# ============================================================

# 连接到服务端
serverAddr = "your-frp-server-ip"
serverPort = 7000

# ============================================================
#  🔐 认证（必须与服务端一致！）
# ============================================================
[auth]
token = "your-secure-token-here"  # ✅ 与 frps.toml 相同

# ============================================================
#  代理 1: XTCP P2P 模式（优先使用，高性能直连）
# ============================================================
[[proxies]]
name = "exo_p2p_windows"
type = "xtcp"
secretKey = "auto-generated-by-exo"
localIP = "127.0.0.1"
localPort = 50051

# ============================================================
#  代理 2: TCP 中转模式（备用，当 P2P 失败时自动启用）
# ============================================================
[[proxies]]
name = "exo_fallback_windows"
type = "tcp"
localIP = "127.0.0.1"
localPort = 50051
remotePort = 50051
```

---

### 3️⃣ FRP Client B - Linux/Docker (frpc_B.toml)

```toml
# ============================================================
#  FRP Client 配置 - Linux 节点
#  文件位置: ~/.exo/frp/frpc_linux-node.toml
# ============================================================

# 连接到服务端（同一个服务端！）
serverAddr = "your-frp-server-ip"
serverPort = 7000

# ============================================================
#  🔐 认证（必须与服务端一致！）
# ============================================================
[auth]
token = "your-secure-token-here"  # ✅ 与 frps.toml 相同

# ============================================================
#  代理 1: XTCP P2P 模式
# ============================================================
[[proxies]]
name = "exo_p2p_linux"
type = "xtcp"
secretKey = "auto-generated-by-exo"
localIP = "127.0.0.1"
localPort = 50051

# ============================================================
#  代理 2: TCP 中转模式
# ============================================================
[[proxies]]
name = "exo_fallback_linux"
type = "tcp"
localIP = "127.0.0.1"
localPort = 50051
remotePort = 50051
```

---

## 🚀 使用 exo 自动生成配置（推荐）

### Windows 端启动：

```powershell
python -m exo.main --disable-tui `
    --discovery-module frp `
    --frp-server-addr <你的FRP服务器IP> `
    --frp-port 7000 `
    --frp-token "your-secure-token-here" `   # ⚠️ 必须指定！
    --enable-p2p `
    --node-port 50051
```

**输出示例**：
```
============================================================
  启动 FRP 发现模块 (XTCP P2P 模式)
============================================================
[FRP] 节点 ID: windows-node
[FRP] 服务端: 123.45.67.89:7000
[FRP] 🔐 Token: your-sec...-here    ← 确认 token 已设置
[FRP] 🔗 P2P 模式: ✅ 启用
```

### Linux 端启动：

```bash
python -m exo.main \
    --disable-tui \
    --discovery-module frp \
    --frp-server-addr 123.45.67.89 \
    --frp-port 7000 \
    --frp-token "your-secure-token-here" \   # ⚠️ 同一个 token！
    --enable-p2p \
    --node-port 50051
```

---

## ❌ 常见错误：Token 不匹配

### 错误日志示例：

```
# 服务端日志
[FRP] [error] login from client [x.x.x.x:xxxxx] failed: token is not correct

# 客户端日志
[FRP] [error] [xxx_xtcp_xxx] failed to register proxy: login to server failed: token is not correct
```

### 解决方法：

1. **检查所有配置文件的 `[auth].token` 是否完全相同**
2. **确保没有多余的空格或引号**
3. **重启 frps 和所有 frpc**

---

## 🔧 Token 最佳实践

### ✅ 推荐：使用强 Token

```bash
# 生成随机 token（Linux/Mac）
openssl rand -hex 32

# 或使用 Python
python -c "import secrets; print(secrets.token_hex(32))"
```

**示例输出**：
```
a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
```

### ⚠️ 不要这样做：

```toml
# ❌ 错误：使用弱 token
token = "123456"
token = "password"
token = "admin"

# ❌ 错误：不同客户端使用不同 token
# Windows: token = "abc"
# Linux:   token = "xyz"  ← 会失败！
```

---

## 📊 验证配置是否正确

### 检查生成的配置文件：

```bash
# 查看 exo 生成的 frpc 配置
cat ~/.exo/frp/frpc_*.toml | grep -A2 "\[auth\]"
```

**预期输出**：
```toml
[auth]
token = "your-secure-token-here"  # ✅ 已包含 auth 段
```

### 测试连接：

```bash
# 手动测试 frpc 能否连接到 frps
./frpc -c frpc.toml -v

# 如果看到以下日志说明成功：
# [INFO] [xxx_xtcp_xxx] proxy registered successfully
```

---

## 🎯 快速检查清单

在启动前，请确认：

- [x] **frps.toml** 中设置了 `[auth].token`
- [x] **所有 frpc 客户端** 的 token 与 frps **完全相同**
- [x] **frps** 已启动并监听在 `bindPort`（默认 7000）
- [x] **防火墙** 允许 7000 端口入站（UDP + TCP）
- [x] **客户端能访问** `frps-server-ip:7000`

---

## 💡 故障排查

### 问题 1：连接被拒绝

**症状**：`login to server failed: dial tcp x.x.x.x:7000: connect: connection refused`

**原因**：frps 未运行或防火墙阻止

**解决**：
```bash
# 检查 frps 是否在运行
ps aux | grep frps

# 检查端口监听
netstat -tlnp | grep 7000

# 检查防火墙
sudo ufw allow 7000/tcp
sudo ufw allow 7000/udp
```

### 问题 2：Token 认证失败

**症状**：`login to server failed: token is not correct`

**原因**：Token 不匹配

**解决**：
```bash
# 对比服务端和客户端的 token
grep "token" /path/to/frps.toml
grep "token" /path/to/frpc.toml

# 确保它们完全一致（包括大小写、空格）
```

### 问题 3：P2P 打洞失败

**症状**：只能通过中转模式连接

**原因**：NAT 类型限制或 UDP 被阻止

**解决**：
1. 检查 UDP 7000 端口是否开放
2. 尝试禁用 P2P：去掉 `--enable-p2p` 参数
3. 或接受 TCP 中转模式（性能稍差但更稳定）

---

## 📚 相关文档

- [FRP 官方认证文档](https://gofrp.com/docs/reference/configuration/#auth)
- [XTCP P2P 模式详解](./FRP_P2P_GUIDE.md)
- [完整部署教程](../README.md)
