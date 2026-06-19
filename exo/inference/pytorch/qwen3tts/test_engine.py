#!/usr/bin/env python3
"""
测试 Qwen3-TTS 推理引擎
使用官方实现
"""

import sys
import os
import torch
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查 CUDA
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 导入引擎
from pytorch_inference_engine import PyTorchQwen3TTSInferenceEngine

def test_tts_inference():
    """测试 TTS 推理"""
    
    # 模型路径
    model_path = r'C:\Users\nil\.cache\exo\downloads\Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign'
    
    print("\n" + "="*60)
    print("初始化 Qwen3-TTS 引擎")
    print("="*60)
    
    # 创建引擎
    engine = PyTorchQwen3TTSInferenceEngine(
        model_path=model_path,
        device='cuda',
        use_bf16=True
    )
    
    print("\n✅ 模型加载成功")
    print(f"模型类型: Qwen3TTSForConditionalGeneration")
    print(f"设备: {engine.device}")
    
    # 测试文本
    test_text = "你好，这是Qwen3语音合成测试。"
    
    print("\n" + "="*60)
    print("测试 TTS 推理")
    print("="*60)
    print(f"\n输入文本: {test_text}")
    print("开始推理...")
    
    # 推理
    codec_tokens, audio = engine.infer_prompt(
        test_text,
        max_new_tokens=2000,
        temperature=0.9,
        top_p=0.95,
        language="chinese"
    )
    
    if len(audio) > 0:
        print(f"\n✅ 推理完成")
        print(f"音频长度: {len(audio)} samples")
        print(f"音频时长: {len(audio)/24000:.2f} seconds")
        
        # 保存音频
        output_path = "test_output_official.wav"
        engine.save_audio(audio, output_path, sample_rate=24000)
        print(f"\n音频已保存: {output_path}")
        
        # 验证音频
        import numpy as np
        print(f"\n音频统计:")
        print(f"  最小值: {audio.min():.4f}")
        print(f"  最大值: {audio.max():.4f}")
        print(f"  平均值: {audio.mean():.4f}")
        print(f"  非零样本: {(audio != 0).sum()} / {len(audio)} ({(audio != 0).sum() / len(audio) * 100:.2f}%)")
        
        if (audio != 0).sum() > 0:
            print("\n✅ 音频有声音！")
        else:
            print("\n❌ 音频是静音")
    else:
        print("\n❌ 推理失败，没有生成音频")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_tts_inference()
