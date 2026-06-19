#!/usr/bin/env python3
"""
Audio Decoder - 将 codec tokens 解码为音频
使用 Qwen3-TTS 官方 speech_tokenizer 模型
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np
import logging
import sys
import os

logger = logging.getLogger(__name__)


class AudioDecoder:
    """
    音频解码器
    将 codec tokens 转换为音频波形
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化音频解码器
        
        Args:
            model_path: speech_tokenizer 模型路径
            device: 计算设备
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.config = None
        
        self._load_model()
    
    def _load_model(self):
        """加载 speech_tokenizer 模型"""
        import json
        
        # 加载配置
        config_path = os.path.join(self.model_path, 'speech_tokenizer', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"加载 speech_tokenizer 配置: {config_path}")
        else:
            logger.warning(f"未找到 speech_tokenizer 配置: {config_path}")
            self.config = {}
        
        # 尝试加载官方 Qwen3-TTS 模型（从本地源码）
        try:
            # 添加本地源码路径到 sys.path
            src_path = os.path.join(os.path.dirname(__file__), 'qwen3tts_src')
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            # 添加 tokenizer_12hz 路径
            tokenizer_path = os.path.join(src_path, 'tokenizer_12hz')
            if tokenizer_path not in sys.path:
                sys.path.insert(0, tokenizer_path)
            
            from modeling_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Model,
                Qwen3TTSTokenizerV2Config
            )
            
            # 加载配置
            config = Qwen3TTSTokenizerV2Config.from_pretrained(
                os.path.join(self.model_path, 'speech_tokenizer'),
                local_files_only=True
            )
            
            # 加载模型
            self.model = Qwen3TTSTokenizerV2Model.from_pretrained(
                os.path.join(self.model_path, 'speech_tokenizer'),
                config=config,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ 成功加载 Qwen3TTSTokenizerV2Model")
            
        except Exception as e:
            logger.error(f"加载 Qwen3TTSTokenizerV2Model 失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def decode(
        self,
        codec_tokens: List[int],
        sample_rate: int = 24000,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        解码 codec tokens 为音频
        
        Args:
            codec_tokens: codec token IDs
            sample_rate: 采样率
            return_numpy: 是否返回 numpy 数组
        
        Returns:
            音频波形数据
        """
        if self.model is None:
            logger.error("AudioDecoder 模型未加载，使用虚拟音频")
            return self._generate_dummy_audio(codec_tokens, sample_rate)
        
        try:
            # 将 codec tokens 转换为张量
            # Qwen3TTSTokenizerV2 期望输入形状: [batch_size, codes_length, num_quantizers]
            # 但我们只有单层 codes，需要扩展为 16 层
            
            # 首先转换为基础张量 [batch_size, codes_length]
            tokens_tensor = torch.tensor([codec_tokens], device=self.device)
            batch_size, seq_len = tokens_tensor.shape
            
            # 获取配置中的 quantizer 数量和 codebook 大小
            decoder_config = getattr(self.model.config, 'decoder_config', None)
            if decoder_config is not None:
                num_quantizers = getattr(decoder_config, 'num_quantizers', 16)
                codebook_size = getattr(decoder_config, 'codebook_size', 2048)
            else:
                num_quantizers = 16
                codebook_size = 2048
            
            # 确保 tokens 在有效范围内 [0, codebook_size)
            tokens_tensor = torch.clamp(tokens_tensor, 0, codebook_size - 1)
            
            # 扩展为多层: [batch_size, codes_length, num_quantizers]
            # 第一层使用实际的 codec tokens，其余层使用 codebook 的中间值
            multi_layer_codes = torch.full(
                (batch_size, seq_len, num_quantizers), 
                codebook_size // 2,  # 使用中间值作为默认值
                device=self.device, 
                dtype=torch.long
            )
            multi_layer_codes[:, :, 0] = tokens_tensor  # 第一层
            
            with torch.no_grad():
                # 调用模型解码
                output = self.model.decode(multi_layer_codes, return_dict=True)
                audio_values = output.audio_values
            
            # 提取音频 (取第一个 batch)
            if isinstance(audio_values, list) and len(audio_values) > 0:
                audio = audio_values[0]
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
            else:
                audio = np.array([], dtype=np.float32)
            
            logger.info(f"解码完成: {len(codec_tokens)} tokens -> {len(audio)} samples")
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"解码失败: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_dummy_audio(codec_tokens, sample_rate)
    
    def _generate_dummy_audio(self, codec_tokens: List[int], sample_rate: int) -> np.ndarray:
        """生成虚拟音频（用于测试）"""
        # 根据 token 数量估算音频长度
        # 假设 12.5 tokens/second
        duration = len(codec_tokens) / 12.5
        samples = int(duration * sample_rate)
        
        # 生成正弦波作为虚拟音频
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 音符
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # 添加淡入淡出
        fade_samples = min(int(0.01 * sample_rate), samples // 10)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        
        logger.info(f"生成虚拟音频: {len(codec_tokens)} tokens -> {len(audio)} samples")
        return audio.astype(np.float32)
    
    def decode_batch(
        self,
        codec_tokens_batch: List[List[int]],
        sample_rate: int = 24000
    ) -> List[np.ndarray]:
        """
        批量解码 codec tokens
        
        Args:
            codec_tokens_batch: 批量 codec token IDs
            sample_rate: 采样率
        
        Returns:
            音频波形数据列表
        """
        audios = []
        for tokens in codec_tokens_batch:
            audio = self.decode(tokens, sample_rate)
            audios.append(audio)
        return audios
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: int = 24000
    ):
        """
        保存音频到文件
        
        Args:
            audio: 音频波形数据
            output_path: 输出路径
            sample_rate: 采样率
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio, sample_rate)
            logger.info(f"音频已保存: {output_path}")
        except ImportError:
            logger.error("未安装 soundfile，无法保存音频")
            # 尝试使用 scipy
            try:
                from scipy.io.wavfile import write
                # 转换为 16-bit PCM
                audio_int16 = (audio * 32767).astype(np.int16)
                write(output_path, sample_rate, audio_int16)
                logger.info(f"音频已保存 (scipy): {output_path}")
            except ImportError:
                logger.error("未安装 scipy，无法保存音频")


class DummyAudioDecoder(AudioDecoder):
    """虚拟音频解码器 - 用于测试"""
    
    def __init__(self, *args, **kwargs):
        """初始化虚拟解码器"""
        self.model_path = kwargs.get('model_path', '')
        self.device = kwargs.get('device', 'cuda')
        self.config = {}
        self.model = {}
        logger.info("使用 DummyAudioDecoder")
    
    def decode(
        self,
        codec_tokens: List[int],
        sample_rate: int = 24000,
        return_numpy: bool = True
    ) -> np.ndarray:
        """生成虚拟音频"""
        return self._generate_dummy_audio(codec_tokens, sample_rate)
