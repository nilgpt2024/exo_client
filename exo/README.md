# Exo - 分布式推理框架

Exo 是一个分布式 AI 推理框架，通过将大型模型按层分片到多个设备上，实现多设备协同推理。

## 核心特性

- **模型分片 (Sharding)**：将大型模型按层分割成多个分片，每个设备只加载和执行部分层
- **隐藏状态传递 (Hidden State Passing)**：分片间通过传递隐藏状态张量协同工作
- **独立 KV 缓存管理**：每个引擎独立管理自己负责的层的 KV 缓存，使用 LRU 缓存策略
- **权重过滤和加载**：只加载当前分片需要的层权重，减少内存占用
- **gRPC 网络通信**：节点间通过 gRPC 进行高效通信

## 支持的模型

- LLaMA 3 (llama3)
- Qwen2.5-VL (qwen2_5vl)
- Qwen3 (qwen3)
- Qwen3-VL (qwen3vl)
- Qwen3-TTS (qwen3tts)

## 项目结构

```
exo/
├── api/                  # API 接口（ChatGPT API）
├── apputil/              # 应用工具（动画、基础图片）
├── download/             # 模型下载与分片下载
├── inference/            # 推理引擎核心
│   ├── pytorch/          # PyTorch 实现
│   │   ├── llama3/       # LLaMA 3 模型
│   │   ├── qwen2_5vl/    # Qwen2.5-VL 模型
│   │   ├── qwen3/        # Qwen3 模型
│   │   ├── qwen3vl/      # Qwen3-VL 模型
│   │   └── qwen3tts/     # Qwen3-TTS 模型
│   └── shard.py          # 分片定义
├── networking/           # 网络通信层
│   ├── grpc/             # gRPC 通信实现
│   ├── frp/              # FRP 内网穿透
│   ├── tailscale/        # Tailscale 组网
│   └── udp/              # UDP 发现
├── orchestration/        # 节点编排与调度
├── topology/             # 拓扑管理与分区策略
├── train/                # 训练数据集（LoRA）
├── tinychat/             # Web 聊天界面
└── viz/                  # 可视化工具
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python main.py
```

## 核心架构

### 1. 模型分片机制

每个节点只加载和执行部分层（`start_layer` 到 `end_layer`）：
- `is_first_layer()` 判断是否包含第一层（需要处理 token 嵌入）
- `is_last_layer()` 判断是否包含最后一层（需要生成输出 logits）

### 2. 隐藏状态传递流程

1. 第一个分片处理输入，输出隐藏状态（result）
2. 隐藏状态通过网络传递给下一个分片
3. 下一个分片接收隐藏状态作为输入，继续处理

### 3. 独立缓存管理

每个节点独立管理自己的状态：
- 最小化网络传输：只传递隐藏状态，不传递 KV 缓存
- 简化分布式协调：每个节点独立管理自己的状态
- 支持异构部署：不同节点可使用不同硬件
- 内存高效：每个节点只加载部分层权重

## 网络发现方式

- **Tailscale**：基于 Tailscale VPN 的自动发现
- **UDP**：局域网 UDP 广播发现
- **Manual**：手动配置网络拓扑
- **FRP**：通过 FRP 进行内网穿透

## 许可证

MIT License
