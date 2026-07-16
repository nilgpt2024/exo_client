# 方案 B：基于 transformers AutoModel 的 Unlimited-OCR 推理引擎支持计划

## 背景与目标

用户希望 `exo_client` 能够推理 `PaddlePaddle/Unlimited-OCR` 模型。该模型已确认无法在现有引擎上运行（基于 DeepseekV2 / `UnlimitedOCRForCausalLM`，带 MoE、DeepLiP/SAM 视觉编码器、MLP projector）。

用户选择 **方案 B**：不手写完整模型架构，而是直接复用 `transformers.AutoModel`/`AutoModelForCausalLM`，在加载完整模型后按 exo 的 `Shard` 层范围进行裁剪，使其适配 exo 的分片分布式推理框架。

目标：新增一个 `PyTorchUnlimitedOCRInferenceEngine`，使 `PaddlePaddle/Unlimited-OCR` 能在 exo 的分片机制下跑通文本生成，并为后续图像输入留出扩展点。

## 已确认的模型结构

从本地已下载权重 `C:\Users\nil\.cache\exo\downloads\PaddlePaddle--Unlimited-OCR` 分析：

- `config.json` 中 `model_type = "unlimited-ocr"`
- 语言模型：`language_config.architectures = ["DeepseekOCRForCausalLM"]`, `num_hidden_layers = 12`
- 权重键前缀：
  - `model.embed_tokens.*`
  - `model.layers.N.*`（N=0..11，MoE expert 权重在同一层内）
  - `model.norm.*`
  - `lm_head.weight`
  - `model.projector.*`
  - `model.vision_model.*`
  - `model.sam_model.*`
  - `model.image_newline`, `model.view_seperator`
- 加载需要 `trust_remote_code=True`

## 总体设计

新增目录 `exo/inference/pytorch/unlimited_ocr/`，实现一个基于 transformers 的分片封装引擎：

1. **模型封装**：`ShardedUnlimitedOCRModel`
   - 使用 `AutoConfig.from_pretrained(..., trust_remote_code=True)` 加载配置
   - 非尾分片：用 `AutoModel.from_pretrained(..., device_map="meta")` 加载基座
   - 尾分片：用 `AutoModelForCausalLM.from_pretrained(..., device_map="meta")` 加载完整因果模型
   - 根据 `shard` 裁剪模块：超出 `[start_layer, end_layer]` 的 decoder layer 替换为 `IdentityBlock`
   - 首分片保留 `embed_tokens` / `projector` / `vision_model` / `sam_model`
   - 尾分片保留 `norm` 和 `lm_head`

2. **前向传播**：统一 `forward(input_ids=None, inputs_embeds=None, **kwargs)`
   - 首分片：`input_ids` 有效，走完整 embedding + 视觉/投影路径
   - 中间/尾分片：`input_ids=None`，`inputs_embeds` 为上阶段隐藏状态
   - 非尾分片返回 `last_hidden_state`
   - 尾分片返回 `logits`（取最后一个位置）

3. **KV 缓存**：复用 `transformers.cache_utils.DynamicCache`
   - 每个请求独立缓存，由 `stateful_model.py` 创建分片感知缓存
   - 不跨节点传递缓存，符合 exo 设计原则

4. **权重加载**：`sharded_utils.py`
   - 扫描 `*.safetensors`（或 `.bin`）
   - 按权重键前缀和层号过滤，只加载 `[start_layer, end_layer]` 内权重
   - 仅首分片加载 `embed_tokens` / `vision_model` / `sam_model` / `projector` / `image_newline` / `view_seperator`
   - 仅尾分片加载 `model.norm` / `lm_head`

5. **Tokenizer / Processor**：
   - 通过 `AutoProcessor.from_pretrained(..., trust_remote_code=True)` 加载
   - 文本：`processor(text=..., return_tensors="pt")`
   - 图像（首期 stub）：把 `image` 字段放入 `inference_state`，首分片 `forward` 透传 `pixel_values` / `image_grid_thw` 等字段；若 processor 不支持则仅打印 warning，继续文本推理

## 需要创建的文件

| 文件 | 用途 |
|---|---|
| `exo/inference/pytorch/unlimited_ocr/__init__.py` | 包初始化 |
| `exo/inference/pytorch/unlimited_ocr/pytorch_inference_engine.py` | 引擎主类 `PyTorchUnlimitedOCRInferenceEngine`，实现 `InferenceEngine` 接口 |
| `exo/inference/pytorch/unlimited_ocr/unlimited_ocr_model.py` | `ShardedUnlimitedOCRModel` 包装器、裁剪逻辑、前向传播 |
| `exo/inference/pytorch/unlimited_ocr/sharded_utils.py` | `load_shard`、`load_model_shard`、权重过滤、meta buffer 初始化 |
| `exo/inference/pytorch/unlimited_ocr/stateful_model.py` | `ModelState`、`make_prompt_state`、基于 `DynamicCache` 的分片缓存 |
| `exo/inference/pytorch/unlimited_ocr/test_engine.py` | 本地单测：单分片、两分片、权重过滤、KV 缓存 |

## 需要修改的文件

| 文件 | 修改内容 |
|---|---|
| `exo/models.py` | 在 `DEFAULT_MODEL_CARDS` 中增加 `unlimited-ocr` 条目：`layers=12`，`repo={"PyTorchUnlimitedOCRInferenceEngine": "PaddlePaddle/Unlimited-OCR"}`；补充 `DEFAULT_PRETTY_NAME` |
| `exo/inference/pytorch/pytorch_inference_engine.py` | 添加引擎名到模块路径的显式映射：`ENGINE_MODULE_MAP["unlimitedocr"] = "unlimited_ocr"`（因为类名去掉前后缀得到 `unlimitedocr`，目录名为 `unlimited_ocr`） |

## 实现步骤

1. **新增目录与空文件**：创建 `unlimited_ocr/` 目录及上述 6 个新文件。
2. **实现 `stateful_model.py`**：
   - 定义 `ModelState`（`cache`、`position`、`shard`）
   - 实现 `make_prompt_state(batch_size, max_seq_len, n_kv_heads, head_dim, n_layers, device, shard)`，返回 `DynamicCache` 并附加 `start_layer/end_layer/n_layers`
3. **实现 `unlimited_ocr_model.py`**：
   - `IdentityBlock`：输入 `(hidden_states, past_key_values)`，返回原 hidden_states 和原 past_key_values
   - `ShardedUnlimitedOCRModel`：构造函数接收 `shard`、`device`、`dtype`，加载 meta device 模型并裁剪
   - `sanitize(state_dict)` 静态/类方法：按分片范围过滤权重
   - `forward(...)`：区分 `input_ids` 与 `inputs_embeds`，调用底层模型，返回 logits 或 last_hidden_state
4. **实现 `sharded_utils.py`**：
   - `load_config(model_path)`：读取 `config.json`
   - `load_model_shard(...)`：创建 `ShardedUnlimitedOCRModel`，调用权重加载
   - `load_shard(...)`：包装函数，返回 `(model, tokenizer)`
   - 权重加载逻辑：读取 `safetensors`/`bin`，按前缀和层号过滤，赋值到 meta device 参数；对未初始化的 meta buffer 初始化并 warning
5. **实现 `pytorch_inference_engine.py`**：
   - 类 `PyTorchUnlimitedOCRInferenceEngine`，继承 `InferenceEngine`
   - 实现 `ensure_shard`、`encode`、`decode`、`sample`、`infer_tensor`、`load_checkpoint`
   - `infer_tensor` 核心逻辑与 Qwen3 引擎保持一致：poll_state → forward → 更新 cache → 返回 logits/hidden_states + state
   - 处理 `inference_state` 中的图像字段并透传给首分片 forward
6. **修改 `models.py`**：注册模型卡片。
7. **修改 `exo/inference/pytorch/pytorch_inference_engine.py`**：添加模块名映射。
8. **编写 `test_engine.py`**：单分片 logits 形状、两分片等价性、权重过滤、KV 缓存递增验证。
9. **运行测试并修复问题**。

## 关键风险与回退

| 风险 | 应对措施 |
|---|---|
| `device_map="meta"` 对 `trust_remote_code=True` 模型不兼容 | 回退到 `low_cpu_mem_usage=True` 在 CPU 加载完整模型后再裁剪 |
| DeepseekV2 meta buffer（如 `inv_freq`、MLA 压缩参数）初始化复杂 | 回退 CPU 全量加载；或在权重加载后调用对应模块的初始化函数 |
| `AutoModel` 未注册，非尾分片无法取 hidden_states | 统一使用 `AutoModelForCausalLM`，非尾分片删除 `lm_head` 并返回 `model.model(...)` 的输出 |
| `inputs_embeds` 分支被远程代码忽略 | 检查远程代码 `forward`；必要时 monkey-patch 或改用 hook 方式 |
| 视觉编码器权重巨大导致首分片 OOM | 首期 stub 图像输入，仅跑纯文本；后续再实现图像路径 |
| 权重键前缀与预期不符 | 运行时打印前 30 个键名动态探测前缀，避免硬编码 |

## 验证计划

1. **单分片完整模型测试**：`Shard("PaddlePaddle/Unlimited-OCR", 0, 11, 12)`，编码 prompt 后调用 `infer_tensor`，断言输出 logits 形状为 `(1, vocab_size)`。
2. **两分片本地流水线测试**：引擎 A `[0,5]` 输出 hidden states 给引擎 B `[6,11]`，比较最终 logits 与单分片完整模型差异在合理浮点误差内。
3. **权重过滤验证**：各分片加载的参数集合正确，非首分片无 `embed_tokens`/`vision_model`/`sam_model`/`projector`，非尾分片无 `norm`/`lm_head`。
4. **KV 缓存递增验证**：同一 `request_id` 连续调用两次 `infer_tensor`，确认 `position` 和 cache seq length 正确累加。
5. **Tokenizer 一致性**：`encode` 后 `decode` 能恢复原始 prompt 文本。
6. **分布式集成测试**：启动两个 exo 节点分别加载上下分片，通过聊天接口生成文本。

## 首期范围建议

首期实现 **纯文本推理**，图像输入仅做 stub/透传框架，不强制要求 OCR 效果正确。这样可以在不处理复杂视觉 pipeline 的情况下，先验证 transformers AutoModel 分片封装这条路是否可行。图像 OCR 功能作为后续迭代。
