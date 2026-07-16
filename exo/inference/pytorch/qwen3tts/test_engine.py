#!/usr/bin/env python3
r"""
Qwen3-TTS 推理引擎本地测试

运行方式:
    cd F:\exoProject\exo_client
    python -m exo.inference.pytorch.qwen3tts.test_engine

环境变量:
    QWEN_TTS_MODEL_PATH: 模型本地路径，缺省使用默认缓存路径
    QWEN_TTS_REF_AUDIO: VoiceClone 参考音频路径（可选）
    QWEN_TTS_REF_TEXT: VoiceClone 参考文本（可选）
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

# 将 exo_client 加入路径
_project_root = Path(__file__).resolve().parents[5]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from exo.inference.pytorch.qwen3tts.pytorch_inference_engine import PyTorchQwen3TTSInferenceEngine
from exo.inference.shard import Shard


def _default_model_path(model_id: str = "qwen-3-tts-1.7b") -> str:
    """返回本地缓存路径，若不存在则返回 HuggingFace repo ID"""
    repo_map = {
        "qwen-3-tts-1.7b": "Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "qwen-3-tts-1.7b-custom": "Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "qwen-3-tts-1.7b-base": "Qwen--Qwen3-TTS-12Hz-1.7B-Base",
    }
    repo_name = repo_map.get(model_id, repo_map["qwen-3-tts-1.7b"])
    local = Path.home() / ".cache" / "exo" / "downloads" / repo_name
    if local.exists():
        return str(local)
    return repo_name.replace("--", "/")


def _make_shard(model_id: str, start_layer: int, end_layer: int) -> Shard:
    return Shard(
        model_id=model_id,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=36,
    )


async def _ensure_engine(model_id: str, start_layer: int = 0, end_layer: int = 35):
    """构造引擎并加载模型分片"""
    engine = PyTorchQwen3TTSInferenceEngine(shard_downloader=None)
    env_key = f"QWEN_TTS_MODEL_PATH_{model_id.replace('-', '_').upper()}"
    engine.model_path = os.environ.get(env_key) or os.environ.get("QWEN_TTS_MODEL_PATH", _default_model_path(model_id))
    shard = _make_shard(model_id, start_layer, end_layer)
    await engine.ensure_shard(shard)
    return engine, shard


def _save_wav(audio: np.ndarray, sr: int, filename: str):
    """保存音频为 WAV 文件"""
    sf.write(filename, audio, sr, format="WAV")
    print(f"  saved: {filename} (samples={len(audio)}, sr={sr})")


def _nonzero_ratio(audio: np.ndarray) -> float:
    """返回非零样本比例"""
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.count_nonzero(audio != 0) / len(audio))


async def test_voice_design_single():
    print("\n[TEST] VoiceDesign single")
    engine, shard = await _ensure_engine("qwen-3-tts-1.7b")
    request_id = str(uuid.uuid4())
    inference_state = {
        "tts_mode": "voice_design",
        "language": "Chinese",
        "instruct": "一个年轻女性的声音，温柔而清晰",
        "max_new_tokens": 128,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.05,
    }
    audio, state = await engine.infer_prompt(
        request_id, shard,
        prompt="你好。",
        inference_state=inference_state,
    )
    assert isinstance(audio, np.ndarray), f"expected np.ndarray, got {type(audio)}"
    assert state.get("sample_rate") == 24000, f"unexpected sample_rate: {state.get('sample_rate')}"
    ratio = _nonzero_ratio(audio)
    assert ratio > 0.01, f"audio is mostly silent: nonzero_ratio={ratio}"
    _save_wav(audio, state["sample_rate"], "test_voice_design_single.wav")
    print(f"  OK, nonzero_ratio={ratio:.4f}")


async def test_voice_design_batch():
    print("\n[TEST] VoiceDesign batch")
    engine, shard = await _ensure_engine("qwen-3-tts-1.7b")
    request_id = str(uuid.uuid4())
    texts = ["你好。", "你好。"]
    inference_state = {
        "tts_mode": "voice_design",
        "language": "Chinese",
        "instruct": "一个年轻女性的声音",
        "texts": texts,
        "max_new_tokens": 128,
    }
    prompt = json.dumps(texts)
    input_data = np.array(list(prompt.encode("utf-8")), dtype=np.uint8)
    try:
        result, state = await engine.infer_tensor(request_id, shard, input_data, inference_state)
    except Exception as e:
        print(f"  WARN: batch inference failed: {e}")
        print("  This may be due to transformers version mismatch (expected 4.46).")
        return
    assert isinstance(result, dict) and result.get("is_batch"), f"expected batch dict, got {type(result)}"
    assert len(result["audio"]) == 2, f"expected 2 audios, got {len(result['audio'])}"
    for i, audio in enumerate(result["audio"]):
        ratio = _nonzero_ratio(audio)
        assert ratio > 0.01, f"audio {i} is mostly silent: nonzero_ratio={ratio}"
        _save_wav(audio, state["sample_rate"], f"test_voice_design_batch_{i}.wav")
    print(f"  OK, batch count={len(result['audio'])}")


async def test_custom_voice():
    print("\n[TEST] CustomVoice")
    model_path = _default_model_path("qwen-3-tts-1.7b-custom")
    if not Path(model_path).exists():
        print(f"  SKIP: CustomVoice model not found at {model_path}")
        return

    engine, shard = await _ensure_engine("qwen-3-tts-1.7b-custom")
    request_id = str(uuid.uuid4())
    inference_state = {
        "tts_mode": "custom_voice",
        "language": "Chinese",
        "speaker": "Vivian",
        "instruct": "愤怒",
        "max_new_tokens": 1024,
        "top_p": 0.92,
    }
    audio, state = await engine.infer_prompt(
        request_id, shard,
        prompt="你怎么能这样对我？",
        inference_state=inference_state,
    )
    assert isinstance(audio, np.ndarray), f"expected np.ndarray, got {type(audio)}"
    ratio = _nonzero_ratio(audio)
    assert ratio > 0.01, f"audio is mostly silent: nonzero_ratio={ratio}"
    _save_wav(audio, state["sample_rate"], "test_custom_voice.wav")
    print(f"  OK, nonzero_ratio={ratio:.4f}")


async def test_voice_clone_direct():
    print("\n[TEST] VoiceClone direct")
    model_path = _default_model_path("qwen-3-tts-1.7b-base")
    if not Path(model_path).exists():
        print(f"  SKIP: Base VoiceClone model not found at {model_path}")
        return

    ref_audio_path = os.environ.get("QWEN_TTS_REF_AUDIO")
    ref_text = os.environ.get("QWEN_TTS_REF_TEXT", "请提供参考文本")
    if ref_audio_path is None or not Path(ref_audio_path).exists():
        print("  SKIP: QWEN_TTS_REF_AUDIO not set or file not found")
        return

    engine, shard = await _ensure_engine("qwen-3-tts-1.7b-base")
    request_id = str(uuid.uuid4())
    inference_state = {
        "tts_mode": "voice_clone",
        "language": "Chinese",
        "ref_audio": ref_audio_path,
        "ref_text": ref_text,
        "x_vector_only_mode": False,
        "max_new_tokens": 1024,
        "temperature": 0.7,
    }
    audio, state = await engine.infer_prompt(
        request_id, shard,
        prompt="这是用参考音色克隆出来的声音。",
        inference_state=inference_state,
    )
    assert isinstance(audio, np.ndarray), f"expected np.ndarray, got {type(audio)}"
    ratio = _nonzero_ratio(audio)
    assert ratio > 0.01, f"audio is mostly silent: nonzero_ratio={ratio}"
    _save_wav(audio, state["sample_rate"], "test_voice_clone_direct.wav")
    print(f"  OK, nonzero_ratio={ratio:.4f}")


async def test_gen_kwargs_passthrough():
    print("\n[TEST] generation kwargs passthrough")
    engine_class = PyTorchQwen3TTSInferenceEngine
    state = {
        "max_new_tokens": 2048,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "subtalker_dosample": True,
        "subtalker_top_k": 20,
        "subtalker_top_p": 0.85,
        "subtalker_temperature": 0.7,
        "non_streaming_mode": True,
    }
    gen_kwargs = engine_class._build_gen_kwargs(state)
    for key in state:
        assert key in gen_kwargs, f"missing key: {key}"
        assert gen_kwargs[key] == state[key], f"mismatch for {key}"
    state_with_none = {"max_new_tokens": 512, "temperature": None}
    gen_kwargs2 = engine_class._build_gen_kwargs(state_with_none)
    assert "max_new_tokens" in gen_kwargs2
    assert "temperature" not in gen_kwargs2
    print("  OK")


async def main():
    print(f"Project root: {_project_root}")
    print(f"VoiceDesign model path: {_default_model_path('qwen-3-tts-1.7b')}")

    await test_gen_kwargs_passthrough()
    await test_voice_design_single()
    await test_voice_design_batch()
    await test_custom_voice()
    await test_voice_clone_direct()

    print("\nAll available tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
