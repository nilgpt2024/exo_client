#!/usr/bin/env python3
r"""
TTS + ASR 端到端功能验证

让 Qwen3-TTS 生成音频，再用 Qwen3-ASR 识别回文本，通过文本相似度验证两个
推理引擎的功能是否真正可用。

运行方式:
    cd F:\\exoProject\\exo_client
    python exo\\inference\\pytorch\\test_tts_asr_e2e.py

环境变量:
    QWEN_TTS_MODEL_PATH: TTS 模型本地路径（可选）
    QWEN_ASR_MODEL_PATH: ASR 模型本地路径（可选）
    TEST_TEXT: 指定 TTS 合成文本，默认 "你好，这是语音合成与识别测试。"
"""

import asyncio
import difflib
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

# 将 exo_client 加入路径
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from exo.inference.pytorch.qwen3asr.pytorch_inference_engine import (
    PyTorchQwen3ASRInferenceEngine,
)
from exo.inference.pytorch.qwen3tts.pytorch_inference_engine import (
    PyTorchQwen3TTSInferenceEngine,
)
from exo.inference.shard import Shard


def _default_tts_model_path() -> str:
    repo_name = "Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    local = Path.home() / ".cache" / "exo" / "downloads" / repo_name
    if local.exists():
        return str(local)
    return repo_name.replace("--", "/")


def _default_asr_model_path() -> str:
    repo_name = "Qwen--Qwen3-ASR-0.6B"
    local = Path.home() / ".cache" / "exo" / "downloads" / repo_name
    if local.exists():
        return str(local)
    return repo_name.replace("--", "/")


async def _load_tts_engine():
    engine = PyTorchQwen3TTSInferenceEngine(shard_downloader=None)
    engine.model_path = os.environ.get("QWEN_TTS_MODEL_PATH", _default_tts_model_path())
    shard = Shard(model_id="qwen-3-tts-1.7b", start_layer=0, end_layer=35, n_layers=36)
    await engine.ensure_shard(shard)
    return engine, shard


async def _load_asr_engine():
    engine = PyTorchQwen3ASRInferenceEngine(shard_downloader=None)
    engine.model_path = os.environ.get("QWEN_ASR_MODEL_PATH", _default_asr_model_path())
    shard = Shard(model_id="qwen-3-asr-0.6b", start_layer=0, end_layer=27, n_layers=28)
    await engine.ensure_shard(shard)
    return engine, shard


def _save_wav(audio: np.ndarray, sr: int, path: Path):
    sf.write(str(path), audio, sr, format="WAV")


def _text_similarity(a: str, b: str) -> float:
    """返回两段中文/英文文本的相似度（0~1）。"""
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


async def main():
    start_time = time.time()
    test_text = os.environ.get("TEST_TEXT", "你好，这是语音合成与识别测试。")
    print(f"原文本: {test_text}")

    # 1. TTS 生成音频
    print("\n[1/3] 加载 TTS 模型并生成音频...")
    tts_engine, tts_shard = await _load_tts_engine()
    request_id = str(uuid.uuid4())
    inference_state = {
        "tts_mode": "voice_design",
        "language": "Chinese",
        "instruct": "一个年轻女性的声音，温柔而清晰",
        "max_new_tokens": 128,
    }
    audio, state = await tts_engine.infer_prompt(
        request_id, tts_shard, prompt=test_text, inference_state=inference_state
    )
    assert isinstance(audio, np.ndarray), f"TTS 输出类型异常: {type(audio)}"
    sr = state.get("sample_rate", 24000)
    print(f"TTS 生成完成: samples={len(audio)}, sample_rate={sr}")

    # 2. 保存音频到临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)
    _save_wav(audio, sr, audio_path)
    print(f"音频已保存: {audio_path}")

    # 3. ASR 识别音频
    print("\n[2/3] 加载 ASR 模型并识别音频...")
    asr_engine, asr_shard = await _load_asr_engine()
    input_data = await asr_engine.encode(asr_shard, prompt=str(audio_path))
    output, _ = await asr_engine.infer_tensor(
        request_id="asr-" + request_id,
        shard=asr_shard,
        input_data=input_data,
        inference_state={"language": "Chinese"},
    )
    recognized = await asr_engine.decode(asr_shard, output)
    print(f"ASR 识别结果: {recognized}")

    # 4. 相似度评估
    print("\n[3/3] 评估识别结果...")
    similarity = _text_similarity(test_text, recognized)
    print(f"文本相似度: {similarity:.2%}")

    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f}s")

    # 关闭 TTS 子进程服务
    await tts_engine.shutdown()

    # 简单阈值判断（允许同音字、标点等少量差异）
    if similarity >= 0.5:
        print("✅ TTS + ASR 端到端测试通过")
    else:
        print("❌ TTS + ASR 端到端测试未通过：识别结果与原文本差异过大")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
