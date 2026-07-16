# 扩展 Qwen3-TTS 引擎以支持官方 Examples 功能

## 背景与目标

用户希望 `exo_client` 的 `PyTorchQwen3TTSInferenceEngine` 能够跑通 `F:\Qwen3-TTS\examples` 中的示例。当前引擎只支持最基础的 `voice_design` 单条推理，缺少大量参数和模式。

目标：扩展引擎和 HTTP 端点，使其支持：
- `generate_voice_design(text, language, instruct, max_new_tokens, ...)`
- `generate_custom_voice(text, speaker, language, instruct, ...)`
- `generate_voice_clone(text, language, ref_audio, ref_text, x_vector_only_mode, ...)`
- 所有生成参数透传：`max_new_tokens`、`top_k`、`top_p`、`temperature`、`repetition_penalty`、`subtalker_*`、`non_streaming_mode`
- 单条与批量输入
- 可选的两阶段 `create_voice_clone_prompt` -> `generate_voice_clone`

## 范围控制

首期聚焦 examples 中直接出现的功能：
1. VoiceDesign 单条 + 批量
2. CustomVoice 单条 + 批量
3. VoiceClone 直接调用（ref_audio/ref_text/x_vector_only_mode）
4. 所有生成参数透传
5. 新增 CustomVoice 和 Base 模型注册
6. 扩展 `/v1/audio/generations` HTTP 端点
7. 修复/更新 `test_engine.py`

以下作为后续迭代：
- `/v1/audio/clone_prompt` 两阶段接口
- 独立 `Qwen3TTSTokenizer` 编解码接口
- `exo_manager/openai_routes.py` 的 TTS 代理

## 已确认的 API 签名

`Qwen3TTSModel`（`F:\Qwen3-TTS\qwen_tts\inference\qwen3_tts_model.py`）的方法统一支持 `**gen_kwargs`：

```python
generate_voice_design(text, instruct, language=None, non_streaming_mode=True, **gen_kwargs)
generate_custom_voice(text, speaker, language=None, instruct=None, non_streaming_mode=True, **gen_kwargs)
generate_voice_clone(text, language=None, ref_audio=None, ref_text=None, x_vector_only_mode=False, voice_clone_prompt=None, non_streaming_mode=False, **gen_kwargs)
create_voice_clone_prompt(ref_audio, ref_text=None, x_vector_only_mode=False)
```

其中 `gen_kwargs` 包括：`do_sample`、`top_k`、`top_p`、`temperature`、`repetition_penalty`、`max_new_tokens`、`subtalker_dosample`、`subtalker_top_k`、`subtalker_top_p`、`subtalker_temperature`。

## 实现方案

### 1. 修改 `exo/inference/pytorch/qwen3tts/pytorch_inference_engine.py`

**输入解析**
- `infer_tensor` 中把 `input_data` 解码后尝试 `json.loads`；
- 若结果是 `list[str]` 则按批量处理；否则视为单条；
- 允许 `inference_state["texts"]` 覆盖 prompt。

**参数读取**
- 从 `inference_state` 读取所有新参数：`max_new_tokens`、`top_k`、`top_p`、`temperature`、`repetition_penalty`、`subtalker_dosample`、`subtalker_top_k`、`subtalker_top_p`、`subtalker_temperature`、`non_streaming_mode`、`ref_audio`、`ref_text`、`x_vector_only_mode`、`voice_clone_prompt`。
- 对标量参数进行广播：若 `texts` 为 list 且参数为标量，则复制为同长度 list；若参数为 list 则校验长度。

**生成路由**
- `voice_design` -> `generate_voice_design(...)`
- `custom_voice` -> `generate_custom_voice(...)`，修复当前错误地把 `voice_clone_prompt` 传给该方法的 bug
- `voice_clone`：
  - 若 `voice_clone_prompt` 存在，直接透传；
  - 否则使用 `ref_audio`/`ref_text`/`x_vector_only_mode` 直接调用；
  - 两者都缺时抛错。
- 所有调用通过 `**gen_kwargs` 透传生成参数，值为 `None` 时不传入。

**批量处理**
- `_sync_generate` 内部统一调用 wrapper 的批量能力；
- 返回 `(wavs: list[np.ndarray], sr: int)`；
- 外层包装为 `{"audio": wavs[0] if single else wavs, "sample_rate": sr, "is_batch": is_batch}`。

**新增 `create_voice_clone_prompt`**
- 可选实现 `async def create_voice_clone_prompt(self, shard, ref_audio, ref_text=None, x_vector_only_mode=False)`；
- 把返回的 `List[VoiceClonePromptItem]` 序列化为可 JSON 的字典（tensor 转 base64 + shape/dtype），供 HTTP 层返回。

### 2. 修改 `exo/api/chatgpt_api.py`

**扩展 `handle_post_audio_generations`**
- 读取新字段：`max_new_tokens`、`top_k`、`top_p`、`temperature`、`repetition_penalty`、`subtalker_dosample`、`subtalker_top_k`、`subtalker_top_p`、`subtalker_temperature`、`non_streaming_mode`、`ref_audio`、`ref_text`、`x_vector_only_mode`、`voice_clone_prompt`。
- `text` 支持 str 或 list[str]。
- 默认 `mode` 推断：
  - `qwen-3-tts-1.7b` -> `voice_design`
  - `qwen-3-tts-1.7b-custom` -> `custom_voice`
  - `qwen-3-tts-1.7b-base` -> `voice_clone`
- 将参数放入 `inference_state`。
- 当 `text` 为 list 时，用 `json.dumps(text)` 作为 `process_prompt` 的 prompt，并保留原始 list 到 `inference_state["texts"]`。

**响应处理**
- 单条继续返回 `audio/wav`；
- 批量返回 JSON：`{"sample_rate": sr, "audio": [base64_wav, ...]}`。

**新增 `POST /v1/audio/clone_prompt`（可选）**
- 解析 `model`、`ref_audio`、`ref_text`、`x_vector_only_mode`；
- 加载 `qwen-3-tts-1.7b-base` 分片；
- 调用引擎 `create_voice_clone_prompt`；
- 返回 JSON 序列化的 prompt。

### 3. 修改 `exo/models.py`

在 `DEFAULT_MODEL_CARDS` 中新增：
- `qwen-3-tts-1.7b-custom` -> `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `qwen-3-tts-1.7b-base` -> `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

在 `DEFAULT_PRETTY_NAME` 中补全 `qwen-3-tts-1.7b-base` 的展示名。

### 4. 修改/重写 `exo/inference/pytorch/qwen3tts/test_engine.py`

- 使用正确的 `infer_prompt(request_id, shard, prompt, inference_state=...)` 签名；
- 覆盖 VoiceDesign、CustomVoice、VoiceClone 直接调用；
- 覆盖批量输入；
- 通过环境变量或本地缓存路径加载模型；
- 保存输出 WAV 并校验非零样本比例。

## 需要修改的文件

| 文件 | 修改内容 |
|---|---|
| `exo/inference/pytorch/qwen3tts/pytorch_inference_engine.py` | 扩展 `infer_tensor`、修复 `custom_voice`、透传生成参数、支持批量、新增 `create_voice_clone_prompt` |
| `exo/api/chatgpt_api.py` | 扩展 `/v1/audio/generations` 参数、默认 mode 推断、批量响应、可选新增 `/v1/audio/clone_prompt` |
| `exo/models.py` | 注册 `qwen-3-tts-1.7b-custom` 和 `qwen-3-tts-1.7b-base` |
| `exo/inference/pytorch/qwen3tts/test_engine.py` | 重写为符合当前接口的测试 |

## 验证计划

1. **静态检查**：`python -m py_compile` 通过上述所有文件。
2. **模型注册检查**：启动节点后 `/v1/models` 包含三个 TTS 模型 ID。
3. **引擎直连测试**：运行重写后的 `test_engine.py`，验证：
   - VoiceDesign 单条与批量输出可播放 WAV
   - CustomVoice 使用 `speaker` 与 `instruct`
   - VoiceClone 使用 `ref_audio` + `ref_text` + `x_vector_only_mode`
   - 生成参数透传不抛错
4. **HTTP 端点测试**：使用 `curl` 验证 `/v1/audio/generations` 对三种模式返回正确格式。
5. **异常场景**：缺少必要参数时返回 400 并附带清晰错误。
