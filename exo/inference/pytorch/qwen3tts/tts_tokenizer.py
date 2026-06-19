#!/usr/bin/env python3
"""
Qwen3-TTS Tokenizer 处理
支持文本到 codec tokens 的转换
"""

import torch
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
import json
import logging

logger = logging.getLogger(__name__)


class Qwen3TTSTokenizer:
    """Qwen3-TTS 专用 Tokenizer 封装"""
    
    def __init__(self, model_path: str):
        """
        初始化 TTS Tokenizer
        
        Args:
            model_path: 模型路径，包含 tokenizer 配置文件
        """
        self.model_path = model_path
        
        # 加载文本 tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 加载配置
        self._load_config()
        
        logger.info(f"Qwen3TTSTokenizer 初始化完成")
        logger.info(f"  文本词汇表大小: {self.text_vocab_size}")
        logger.info(f"  Codec 词汇表大小: {self.codec_vocab_size}")
        logger.info(f"  TTS BOS token: {self.tts_bos_token_id}")
        logger.info(f"  TTS EOS token: {self.tts_eos_token_id}")
    
    def _load_config(self):
        """加载模型配置"""
        import os
        config_path = os.path.join(self.model_path, 'config.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        talker_config = config.get('talker_config', {})
        
        # Token IDs
        self.tts_bos_token_id = config.get('tts_bos_token_id', 151672)
        self.tts_eos_token_id = config.get('tts_eos_token_id', 151673)
        self.tts_pad_token_id = config.get('tts_pad_token_id', 151671)
        self.assistant_token_id = config.get('assistant_token_id', 77091)
        
        # 词汇表大小
        self.text_vocab_size = talker_config.get('text_vocab_size', 151936)
        self.codec_vocab_size = talker_config.get('vocab_size', 3072)
        
        # Codec 特殊 token IDs
        self.codec_bos_id = talker_config.get('codec_bos_id', 2149)
        self.codec_eos_token_id = talker_config.get('codec_eos_token_id', 2150)
        self.codec_pad_id = talker_config.get('codec_pad_id', 2148)
        self.codec_think_id = talker_config.get('codec_think_id', 2154)
        self.codec_nothink_id = talker_config.get('codec_nothink_id', 2155)
        
        # 语言 ID
        self.codec_language_id = talker_config.get('codec_language_id', {
            'chinese': 2055,
            'english': 2050,
        })
    
    def encode_text(
        self,
        text: str,
        language: str = 'chinese',
        add_special_tokens: bool = True,
        return_tensors: str = 'pt'
    ) -> torch.Tensor:
        """
        编码文本为输入 IDs
        
        Args:
            text: 输入文本
            language: 语言 ('chinese', 'english', 等)
            add_special_tokens: 是否添加特殊 token
            return_tensors: 返回张量格式
        
        Returns:
            编码后的 token IDs
        """
        # 构建 TTS 提示格式
        # 格式: tts_bos + language_id + text + assistant_token
        
        # 先编码文本
        text_tokens = self.text_tokenizer.encode(
            text,
            add_special_tokens=False
        )
        
        if add_special_tokens:
            # 获取语言 ID
            language_id = self.codec_language_id.get(language, 2055)
            
            # 构建完整序列
            input_ids = [
                self.tts_bos_token_id,  # TTS 开始
                language_id,             # 语言标记
            ] + text_tokens + [
                self.assistant_token_id, # Assistant 标记
            ]
        else:
            input_ids = text_tokens
        
        # 转换为张量
        if return_tensors == 'pt':
            return torch.tensor([input_ids], dtype=torch.long)
        return input_ids
    
    def decode_codec_tokens(
        self,
        codec_tokens: List[int],
        skip_special_tokens: bool = True
    ) -> List[int]:
        """
        解码 codec tokens
        
        Args:
            codec_tokens: codec token IDs
            skip_special_tokens: 是否跳过特殊 token
        
        Returns:
            处理后的 codec tokens
        """
        if skip_special_tokens:
            # 过滤掉特殊 token
            special_ids = {
                self.codec_bos_id,
                self.codec_eos_token_id,
                self.codec_pad_id,
                self.codec_think_id,
                self.codec_nothink_id,
            }
            codec_tokens = [t for t in codec_tokens if t not in special_ids]
        
        return codec_tokens
    
    def create_tts_prompt(
        self,
        text: str,
        language: str = 'chinese',
        voice_desc: Optional[str] = None
    ) -> str:
        """
        创建 TTS 提示
        
        Args:
            text: 要合成的文本
            language: 语言
            voice_desc: 声音描述（可选）
        
        Returns:
            格式化后的提示文本
        """
        # 构建提示格式
        if voice_desc:
            prompt = f"{voice_desc}\n{text}"
        else:
            prompt = text
        
        return prompt
    
    def batch_encode(
        self,
        texts: List[str],
        language: str = 'chinese',
        padding: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            language: 语言
            padding: 是否填充
            return_tensors: 返回张量格式
        
        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        encoded = []
        for text in texts:
            ids = self.encode_text(text, language, return_tensors=None)
            encoded.append(ids)
        
        # 填充
        if padding:
            max_len = max(len(ids) for ids in encoded)
            padded = []
            masks = []
            for ids in encoded:
                pad_len = max_len - len(ids)
                padded_ids = ids + [self.tts_pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                padded.append(padded_ids)
                masks.append(mask)
            
            if return_tensors == 'pt':
                return {
                    'input_ids': torch.tensor(padded, dtype=torch.long),
                    'attention_mask': torch.tensor(masks, dtype=torch.long),
                }
        
        return {'input_ids': encoded}
    
    @property
    def eos_token_id(self) -> int:
        """获取 EOS token ID"""
        return self.tts_eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """获取 BOS token ID"""
        return self.tts_bos_token_id
    
    @property
    def pad_token_id(self) -> int:
        """获取 PAD token ID"""
        return self.tts_pad_token_id
