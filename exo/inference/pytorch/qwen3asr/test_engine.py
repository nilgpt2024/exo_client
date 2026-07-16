#!/usr/bin/env python3
"""
Qwen3-ASR 推理引擎测试

使用本地缓存模型对示例音频进行转录测试。
如果没有示例音频，会生成一个极短的静音 wav 文件作为占位。
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# 将项目根目录加入路径，以便 import exo
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from exo.inference.shard import Shard
from exo.inference.pytorch.qwen3asr.pytorch_inference_engine import (
    PyTorchQwen3ASRInferenceEngine,
)


async def ensure_test_audio(path: Path) -> Path:
    """若不存在测试音频，则生成 1 秒 16kHz 单声道静音文件作为占位。"""
    if path.exists():
        return path

    print(f"未找到示例音频，生成占位静音文件: {path}")
    sample_rate = 16000
    duration = 1.0
    silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
    sf.write(path, silence, sample_rate)
    return path


async def main():
    start_time = time.time()

    # 打印环境信息
    try:
        import transformers
        import qwen_asr

        print(f"transformers 版本: {transformers.__version__}")
        print(f"qwen-asr 版本: {getattr(qwen_asr, '__version__', 'unknown')}")
    except Exception as e:
        print(f"打印版本信息失败: {e}")

    # 本地缓存路径
    cache_dir = Path.home() / ".cache" / "exo" / "downloads" / "Qwen--Qwen3-ASR-0.6B"
    if not cache_dir.exists():
        print(f"错误：未找到本地缓存模型: {cache_dir}")
        print("请先用以下命令下载模型：")
        print("  huggingface-cli download Qwen/Qwen3-ASR-0.6B")
        print("或设置环境变量 AUDIO_PATH 指定测试音频路径（模型路径仍需存在）。")
        sys.exit(1)

    # 测试音频路径，可通过环境变量 AUDIO_PATH 覆盖
    audio_path = Path(
        os.environ.get("AUDIO_PATH", cache_dir / ".." / "test_audio.wav")
    ).resolve()
    audio_path = await ensure_test_audio(audio_path)

    print(f"模型路径: {cache_dir}")
    print(f"测试音频: {audio_path}")

    # 构造引擎（不使用 shard_downloader，直接指定模型路径）
    engine = PyTorchQwen3ASRInferenceEngine(
        shard_downloader=None, model_path=str(cache_dir)
    )

    # 构造完整分片（单节点加载所有 28 层）
    shard = Shard(
        model_id="qwen-3-asr-0.6b", start_layer=0, end_layer=27, n_layers=28
    )

    # encode：prompt 为音频路径
    input_data = await engine.encode(shard, prompt=str(audio_path))
    print(f"encode 输出类型: {type(input_data)}, shape: {input_data.shape}")

    # infer_tensor
    output, state = await engine.infer_tensor(
        request_id="test-001",
        shard=shard,
        input_data=input_data,
        inference_state={"language": "Chinese"},
    )
    print(f"infer_tensor 输出类型: {type(output)}, shape: {output.shape}")

    # decode
    transcription = await engine.decode(shard, output)
    elapsed = time.time() - start_time
    print(f"识别结果: {transcription}")
    print(f"总耗时: {elapsed:.2f}s")

    if not transcription.strip():
        print("警告：识别结果为空（占位静音音频属于正常现象）")
    else:
        print("ASR 引擎测试通过")


if __name__ == "__main__":
    asyncio.run(main())
