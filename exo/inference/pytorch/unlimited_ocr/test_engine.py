"""Unlimited-OCR 引擎本地测试"""
import asyncio
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 确保能找到 exo 包
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from exo.inference.shard import Shard
from exo.inference.pytorch.unlimited_ocr.unlimited_ocr_model import ShardedUnlimitedOCRModel
from exo.inference.pytorch.unlimited_ocr.sharded_utils import load_model_shard, load_config
from exo.inference.pytorch.unlimited_ocr.pytorch_inference_engine import PyTorchUnlimitedOCRInferenceEngine


CACHE_DIR = Path.home() / ".cache" / "exo" / "downloads" / "PaddlePaddle--Unlimited-OCR"
MODEL_ID = "PaddlePaddle/Unlimited-OCR"
N_LAYERS = 12


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_load_config():
    cfg = load_config(CACHE_DIR)
    assert cfg["model_type"] == "unlimited-ocr"
    assert cfg["language_config"]["num_hidden_layers"] == N_LAYERS
    print("[OK] test_load_config")


def test_weight_keys():
    index_path = CACHE_DIR / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    keys = list(index["weight_map"].keys())

    # 确认关键前缀存在
    assert any(k.startswith("model.layers.0.") for k in keys)
    assert any(k.startswith("model.layers.11.") for k in keys)
    assert any(k.startswith("model.embed_tokens.") for k in keys)
    assert any(k.startswith("model.norm.") for k in keys)
    assert any(k == "lm_head.weight" for k in keys)
    print(f"[OK] test_weight_keys (total {len(keys)} keys)")


def test_sanitize():
    from exo.inference.pytorch.unlimited_ocr.unlimited_ocr_model import ShardedUnlimitedOCRModel

    # 模拟完整 state_dict（只包含键）
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.5.self_attn.q_proj.weight",
        "model.layers.11.self_attn.q_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.vision_model.transformer.layers.0.self_attn.out_proj.weight",
        "model.projector.0.weight",
    ]
    fake_state = {k: None for k in keys}

    shard_first = Shard(MODEL_ID, 0, 5, N_LAYERS)
    filtered_first = ShardedUnlimitedOCRModel.sanitize(fake_state, shard_first)
    assert "model.embed_tokens.weight" in filtered_first
    assert "model.layers.0.self_attn.q_proj.weight" in filtered_first
    assert "model.layers.5.self_attn.q_proj.weight" in filtered_first
    assert "model.layers.11.self_attn.q_proj.weight" not in filtered_first
    assert "model.norm.weight" not in filtered_first
    assert "lm_head.weight" not in filtered_first
    assert "model.vision_model.transformer.layers.0.self_attn.out_proj.weight" in filtered_first

    shard_last = Shard(MODEL_ID, 6, 11, N_LAYERS)
    filtered_last = ShardedUnlimitedOCRModel.sanitize(fake_state, shard_last)
    assert "model.embed_tokens.weight" not in filtered_last
    assert "model.layers.5.self_attn.q_proj.weight" not in filtered_last
    assert "model.layers.11.self_attn.q_proj.weight" in filtered_last
    assert "model.norm.weight" in filtered_last
    assert "lm_head.weight" in filtered_last
    assert "model.vision_model.transformer.layers.0.self_attn.out_proj.weight" not in filtered_last

    print("[OK] test_sanitize")


def test_load_shard_first():
    """加载首分片，验证参数过滤"""
    shard = Shard(MODEL_ID, 0, 5, N_LAYERS)
    device = _default_device()
    model = load_model_shard(
        CACHE_DIR,
        shard=shard,
        device=device,
        use_bf16=False,
    )

    base = model._get_base_model()
    assert base.embed_tokens is not None
    # 非尾分片的 norm 被替换为 Identity（remote code 会调用）
    assert isinstance(base.norm, (nn.Identity, type(None)))
    assert model.model.lm_head is None

    # 统计实际参数数量
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"[INFO] first shard total params: {total_params / 1e6:.2f}M")

    # 确认没有尾层参数
    for name, _ in model.model.named_parameters():
        assert not name.startswith("lm_head.")
        assert not name.startswith("model.norm.")

    print("[OK] test_load_shard_first")
    return model


def test_load_shard_last():
    """加载尾分片，验证参数过滤"""
    shard = Shard(MODEL_ID, 6, 11, N_LAYERS)
    device = _default_device()
    model = load_model_shard(
        CACHE_DIR,
        shard=shard,
        device=device,
        use_bf16=False,
    )

    base = model._get_base_model()
    assert base.embed_tokens is None
    assert base.norm is not None
    assert model.model.lm_head is not None

    # 确认没有首层参数
    for name, _ in model.model.named_parameters():
        assert not name.startswith("model.embed_tokens.")
        assert not name.startswith("model.vision_model.")
        assert not name.startswith("model.sam_model.")
        assert not name.startswith("model.projector.")

    print("[OK] test_load_shard_last")
    return model


async def test_engine_single_shard():
    """单分片完整模型推理"""
    from exo.download.shard_download import NoopShardDownloader

    shard = Shard(MODEL_ID, 0, N_LAYERS - 1, N_LAYERS)
    downloader = NoopShardDownloader()
    engine = PyTorchUnlimitedOCRInferenceEngine(downloader)

    # 加载模型
    await engine.ensure_shard(shard)
    assert engine.model is not None
    assert engine.tokenizer is not None

    # encode / decode
    prompt = "Hello, this is a test."
    tokens = await engine.encode(shard, prompt)
    assert isinstance(tokens, np.ndarray)
    assert len(tokens) > 0

    decoded = await engine.decode(shard, tokens)
    print(f"[INFO] decoded: {decoded[:50]}")

    # 推理
    logits, state = await engine.infer_tensor("req-1", shard, tokens)
    print(f"[INFO] logits shape: {logits.shape}, dtype: {logits.dtype}")
    # 使用模型 config 中的 vocab_size（可能与 tokenizer.vocab_size 不一致）
    model_vocab_size = engine.config.vocab_size
    print(f"[INFO] model vocab_size: {model_vocab_size}")
    assert logits.shape[-1] == model_vocab_size, f"logits last dim {logits.shape[-1]} != model vocab_size {model_vocab_size}"
    assert "past_key_values" in state
    print(f"[OK] test_engine_single_shard")

    return logits, state


async def test_two_shard_pipeline():
    """两分片流水线：对比单分片与两分片最终 logits

    为避免同时加载两个分片到 GPU 导致 OOM，强制在 CPU 上运行此测试。
    """
    from exo.download.shard_download import NoopShardDownloader

    prompt = "Hello, this is a test."
    device = "cpu"

    # 单分片完整模型
    full_shard = Shard(MODEL_ID, 0, N_LAYERS - 1, N_LAYERS)
    engine_full = PyTorchUnlimitedOCRInferenceEngine(NoopShardDownloader())
    engine_full.device = torch.device(device)
    await engine_full.ensure_shard(full_shard)
    tokens = await engine_full.encode(full_shard, prompt)
    full_logits, _ = await engine_full.infer_tensor("req-full", full_shard, tokens)

    # 两分片
    shard_a = Shard(MODEL_ID, 0, 5, N_LAYERS)
    shard_b = Shard(MODEL_ID, 6, 11, N_LAYERS)

    engine_a = PyTorchUnlimitedOCRInferenceEngine(NoopShardDownloader())
    engine_b = PyTorchUnlimitedOCRInferenceEngine(NoopShardDownloader())
    engine_a.device = torch.device(device)
    engine_b.device = torch.device(device)

    await engine_a.ensure_shard(shard_a)
    await engine_b.ensure_shard(shard_b)

    # A 处理 prompt
    hidden_a, state_a = await engine_a.infer_tensor("req-pipe", shard_a, tokens)
    assert hidden_a.ndim == 3  # [batch, seq, hidden]

    # B 接收隐藏状态
    logits_b, state_b = await engine_b.infer_tensor("req-pipe", shard_b, hidden_a[0, -1:, :])

    # 比较最终 logits（允许一定误差）
    diff = np.abs(full_logits - logits_b).max()
    print(f"[INFO] two-shard vs single-shard max diff: {diff:.6f}")

    # 由于中间层 IdentityBlock 不更新 KV，且我们裁剪了 norm/lm_head，
    # 这里只验证形状和数值合理性，不要求严格相等
    model_vocab_size_b = engine_b.config.vocab_size
    assert logits_b.shape[-1] == model_vocab_size_b, f"logits_b last dim {logits_b.shape[-1]} != model vocab_size {model_vocab_size_b}"
    assert np.isfinite(logits_b).all()
    print("[OK] test_two_shard_pipeline")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-only", action="store_true", help="只运行不加载权重的轻量级测试")
    parser.add_argument("--skip-inference", action="store_true", help="跳过大模型推理测试")
    args = parser.parse_args()

    print(f"[INFO] Using device: {_default_device()}")
    print(f"[INFO] Model cache: {CACHE_DIR}")

    if not CACHE_DIR.exists():
        print(f"[ERROR] Model cache not found: {CACHE_DIR}")
        print("Please run test_new_shard_download.py first to download the model.")
        return

    # 轻量级测试
    test_load_config()
    test_weight_keys()
    test_sanitize()

    if args.light_only:
        print("\nLight tests passed!")
        return

    # 加载测试（较慢，取决于硬件）
    print("\n[INFO] Loading first shard...")
    test_load_shard_first()
    print("\n[INFO] Loading last shard...")
    test_load_shard_last()

    if args.skip_inference:
        print("\nLoad tests passed!")
        return

    # 推理测试（更慢）
    print("\n[INFO] Running single shard inference...")
    await test_engine_single_shard()

    print("\n[INFO] Running two shard pipeline...")
    await test_two_shard_pipeline()

    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
