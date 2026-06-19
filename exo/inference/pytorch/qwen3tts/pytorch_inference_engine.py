#!/usr/bin/env python3
"""
PyTorch Qwen3-TTS Inference Engine for exo
适配 exo 框架的 InferenceEngine 接口
"""

import asyncio
import concurrent.futures
import torch
import numpy as np
import logging
import sys
import os
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# 添加 Qwen3-TTS 源码路径
_qwen_tts_path = Path("F:/Qwen3-TTS")
if _qwen_tts_path.exists() and str(_qwen_tts_path) not in sys.path:
    sys.path.insert(0, str(_qwen_tts_path))

# 兼容性补丁: qwen_tts 依赖 transformers 4.57 的 check_model_inputs,
# 但当前环境 transformers 5.x 已移除该函数, 需要 monkey-patch
try:
    from transformers.utils.generic import check_model_inputs
except ImportError:
    import functools
    def _noop_decorator(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper
    import transformers.utils.generic as _generic
    _generic.check_model_inputs = _noop_decorator

# 兼容性补丁: Qwen3TTSTalkerConfig 缺少 pad_token_id,
# transformers 5.x 的 generate() 和模型初始化都需要此属性
def _patch_talker_config():
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
        # 类属性
        if not hasattr(Qwen3TTSTalkerConfig, 'pad_token_id') or getattr(Qwen3TTSTalkerConfig, 'pad_token_id') is None:
            Qwen3TTSTalkerConfig.pad_token_id = 151671
        # 实例属性：通过 __init__ 注入
        _original_init = Qwen3TTSTalkerConfig.__init__
        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            if not hasattr(self, 'pad_token_id') or self.pad_token_id is None:
                self.pad_token_id = 151671
        Qwen3TTSTalkerConfig.__init__ = _patched_init
    except Exception:
        pass

# 兼容性补丁: transformers 5.x 的 ROPE_INIT_FUNCTIONS 移除了 'default' 键,
# qwen_tts 代码中 rope_type="default" 在 5.x 中会 KeyError,
# 需要提供一个不依赖 config.rope_parameters 的 default RoPE 初始化函数
def _patch_rope_init_functions():
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if 'default' not in ROPE_INIT_FUNCTIONS:
            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
                if config is None:
                    raise ValueError("config is required")
                base = getattr(config, 'rope_theta', 10000.0)
                partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)
                head_dim = getattr(config, 'head_dim', None)
                if head_dim is None:
                    head_dim = config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
                attention_factor = 1.0
                return inv_freq, attention_factor
            ROPE_INIT_FUNCTIONS['default'] = _compute_default_rope_parameters
    except Exception:
        pass

_patch_rope_init_functions()
_patch_talker_config()

from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.inference.shard import Shard

logger = logging.getLogger(__name__)


class PyTorchQwen3TTSInferenceEngine(InferenceEngine):
    """
    Qwen3-TTS 推理引擎 - 适配 exo 框架
    支持完整的 TTS 功能，包括 VoiceDesign 和 CustomVoice
    """

    def __init__(self, shard_downloader: ShardDownloader, model_path: str = None, **kwargs):
        super().__init__()
        self.shard_downloader = shard_downloader
        self.model_path = model_path
        self.device = self._get_best_device()
        self.shard = None

        # 模型组件
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.speech_tokenizer = None

        # 加载锁
        self._shard_lock = asyncio.Lock()

        # BF16 支持
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.use_bf16:
            logger.info("✅ 检测到BF16支持，启用BF16优化推理")

        # 线程池执行器（用于同步 TTS 生成）
        self._executor = None

    def _get_executor(self):
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return self._executor

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._get_executor(), fn, *args)

    def _get_best_device(self) -> torch.device:
        """自动选择最佳设备"""
        if torch.cuda.is_available():
            device_name = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"使用GPU: {gpu_name} ({memory_gb:.1f}GB)")
        elif torch.backends.mps.is_available():
            device_name = "mps"
            logger.info("使用Apple Silicon GPU (MPS)")
        else:
            device_name = "cpu"
            logger.info("使用CPU推理")
        return torch.device(device_name)

    async def ensure_shard(self, shard: Shard):
        """确保模型分片已加载"""
        async with self._shard_lock:
            if self.shard == shard:
                return

            # 下载模型
            if self.shard_downloader is not None:
                model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
                self.model_path = str(model_path)

            if self.shard != shard:
                await self._load_model()
                self.shard = shard
                logger.info(f"✅ Qwen3-TTS 模型加载完成: {shard.model_id}")

    async def _load_model(self):
        """加载 Qwen3-TTS 模型"""
        try:
            logger.info(f"加载 Qwen3-TTS 模型: {self.model_path}")

            _patch_talker_config()
            _patch_rope_init_functions()

            # 导入 Qwen3-TTS 模块
            from qwen_tts import Qwen3TTSModel

            # 检查路径是否存在，如果存在则使用本地文件
            model_path = Path(self.model_path)
            if model_path.exists():
                # 本地路径
                logger.info(f"使用本地模型路径: {model_path}")
                self.model_wrapper = Qwen3TTSModel.from_pretrained(
                    str(model_path),
                    device_map=str(self.device),
                    dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
                    attn_implementation="eager",
                    local_files_only=True
                )
            else:
                # HuggingFace 模型 ID
                logger.info(f"从 HuggingFace 下载模型: {self.model_path}")
                self.model_wrapper = Qwen3TTSModel.from_pretrained(
                    self.model_path,
                    device_map=str(self.device),
                    dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
                    attn_implementation="eager"
                )

            self.model = self.model_wrapper.model
            self.processor = self.model_wrapper.processor
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

            logger.info("✅ 成功加载 Qwen3-TTS 模型")

        except Exception as e:
            logger.error(f"加载 Qwen3-TTS 模型失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def encode(self, shard: Shard, prompt: str, enable_thinking: bool = False) -> np.ndarray:
        """
        编码文本为输入张量
        TTS 模型不需要传统的 encode，返回文本本身
        """
        await self.ensure_shard(shard)
        # 返回文本的 UTF-8 编码字节数组
        return np.array(list(prompt.encode('utf-8')), dtype=np.uint8)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """
        解码 token 为文本
        TTS 模型输出是音频，这里返回空字符串
        """
        return ""

    async def sample(self, x: np.ndarray, temp: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                     repetition_penalty: float = 1.0, generated_tokens: list = None) -> np.ndarray:
        """
        采样 - TTS 模型不使用此接口
        """
        return x

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray,
                          inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """
        从张量推理 - TTS 主入口

        input_data: 编码后的文本（字节数组）
        返回: (音频数据, inference_state)
        """
        await self.ensure_shard(shard)

        # 解码文本
        text = input_data.tobytes().decode('utf-8')

        # 从 inference_state 获取参数
        mode = inference_state.get('tts_mode', 'voice_design') if inference_state else 'voice_design'
        language = inference_state.get('language', 'Chinese') if inference_state else 'Chinese'
        instruct = inference_state.get('instruct', '') if inference_state else ''
        speaker = inference_state.get('speaker', None) if inference_state else None

        def _sync_generate():
            # 在生成前清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if mode == 'voice_design':
                wavs, sr = self.model_wrapper.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct,
                )
            elif mode == 'custom_voice':
                voice_clone_prompt = inference_state.get('voice_clone_prompt') if inference_state else None
                if voice_clone_prompt:
                    wavs, sr = self.model_wrapper.generate_custom_voice(
                        text=text,
                        voice_clone_prompt=voice_clone_prompt
                    )
                else:
                    raise ValueError("custom_voice 模式需要提供 voice_clone_prompt")
            else:
                wavs, sr = self.model_wrapper.generate_voice_clone(
                    text=text,
                    speaker=speaker
                )

            audio = wavs[0] if isinstance(wavs, list) else wavs
            audio_array = np.array(audio, dtype=np.float32)
            
            # 生成后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return audio_array, sr

        try:
            audio_array, sr = await self._run_in_executor(_sync_generate)

            if inference_state is None:
                inference_state = {}
            inference_state['sample_rate'] = sr

            return audio_array, inference_state

        except Exception as e:
            logger.error(f"TTS 推理失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str,
                          inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """
        从文本提示推理 - TTS 便捷入口
        """
        await self.ensure_shard(shard)

        # 编码文本
        input_data = np.array(list(prompt.encode('utf-8')), dtype=np.uint8)

        return await self.infer_tensor(request_id, shard, input_data, inference_state)

    async def load_checkpoint(self, shard: Shard, path: str):
        """加载模型检查点"""
        await self.ensure_shard(shard)
        logger.info(f"Qwen3-TTS 模型检查点加载: {path}")

    async def save_checkpoint(self, shard: Shard, path: str):
        """保存模型检查点"""
        logger.info(f"Qwen3-TTS 模型检查点保存: {path}")

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024 ** 3
            memory_info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return memory_info
