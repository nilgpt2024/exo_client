#!/usr/bin/env python3
"""
PyTorch Qwen3-ASR Inference Engine for exo

参考 Qwen3-TTS 引擎的实现方式，使用官方 qwen-asr 包提供的 Qwen3ASRModel
完成音频到文本的转录。当前为单节点完整模型推理，尚未实现按层分片。

兼容性说明：
- qwen-asr 基于 transformers ~4.57.6 编写，当前 exo 锁定 transformers==5.3.0。
- 本模块在导入阶段以及每次加载模型前注入兼容性补丁，使 qwen-asr 能在 5.3.0 下运行。
"""

import asyncio
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 兼容性补丁工具函数
# ---------------------------------------------------------------------------

def _patch_check_model_inputs() -> None:
    """
    为 transformers 5.x 补全 check_model_inputs 符号。

    qwen-asr 的建模代码会执行：
        from transformers.utils.generic import ... check_model_inputs
    并在函数上装饰 @check_model_inputs(...)。
    transformers 5.3.0 已移除该符号，因此需要注入一个兼容装饰器工厂/装饰器的实现。
    """
    try:
        import transformers.utils.generic as _generic

        # 已经存在且可调用的原始实现先保存
        _orig = getattr(_generic, "_original_check_model_inputs", None)
        if _orig is None:
            _orig = getattr(_generic, "check_model_inputs", None)
            if callable(_orig):
                _generic._original_check_model_inputs = _orig

        def _compat_check_model_inputs(*args, **kwargs):
            # 直接用作 @check_model_inputs（无参数装饰器）
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def wrapper(fn):
                try:
                    if _orig is not None and callable(_orig):
                        return _orig(fn, *args, **kwargs)
                except Exception:
                    pass
                return fn

            return wrapper

        _generic.check_model_inputs = _compat_check_model_inputs
    except Exception as e:
        logger.debug(f"[ASR patch] check_model_inputs 补丁未生效: {e}")


def _compute_default_rope_parameters(
    config=None, device=None, seq_len=None, layer_type=None
):
    """
    兼容 transformers 5.3.0 与 qwen-asr 的 default RoPE 参数计算。

    优先从 config.rope_parameters（5.3.0 新位置）读取，再回退到 config.rope_theta /
    config.rope_scaling（旧位置）。
    """
    if config is None:
        raise ValueError("config is required")

    rope_params = getattr(config, "rope_parameters", None) or {}
    base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = rope_params.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", 1.0),
    )

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_attention_heads = getattr(config, "num_attention_heads", None)
        if hidden_size is None or num_attention_heads is None:
            raise ValueError(
                "config must provide head_dim or hidden_size + num_attention_heads"
            )
        head_dim = hidden_size // num_attention_heads

    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq, 1.0


def _patch_rope_init_functions() -> None:
    """
    为 transformers 5.x 的 ROPE_INIT_FUNCTIONS 回填 'default' 键。

    qwen-asr 的 Qwen3ASRThinkerTextRotaryEmbedding 在 __init__ 中通过
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    获取初始化函数。transformers 5.3.0 的字典里没有 'default' 键，会触发 KeyError。
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if not isinstance(ROPE_INIT_FUNCTIONS, dict):
            return

        if "default" in ROPE_INIT_FUNCTIONS:
            return

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
        logger.debug("[ASR patch] 已注入 ROPE_INIT_FUNCTIONS['default']")
    except Exception as e:
        logger.debug(f"[ASR patch] ROPE_INIT_FUNCTIONS 补丁未生效: {e}")


def _patch_asr_config() -> None:
    """
    为 Qwen3ASRThinkerConfig 补全 pad_token_id。

    transformers 5.3.0 的 generate() 与模型前向会访问 config.pad_token_id，
    而 qwen-asr 的 Qwen3ASRThinkerConfig 未定义该属性，导致 AttributeError。
    """
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRThinkerConfig,
        )

        default_pad = 151645  # 与 eos_token_id 保持一致

        # 类属性兜底
        if (
            not hasattr(Qwen3ASRThinkerConfig, "pad_token_id")
            or getattr(Qwen3ASRThinkerConfig, "pad_token_id", None) is None
        ):
            Qwen3ASRThinkerConfig.pad_token_id = default_pad

        # 实例属性兜底：在 __init__ 执行后补全
        _original_init = Qwen3ASRThinkerConfig.__init__

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            if (
                not hasattr(self, "pad_token_id")
                or self.pad_token_id is None
            ):
                self.pad_token_id = default_pad

        Qwen3ASRThinkerConfig.__init__ = _patched_init
    except Exception as e:
        logger.debug(f"[ASR patch] Qwen3ASRThinkerConfig 补丁未生效: {e}")


def _patch_asr_rotary_embedding() -> None:
    """
    为 Qwen3ASRThinkerTextRotaryEmbedding 补全 compute_default_rope_parameters。

    transformers 5.3.0 的通用 _init_weights 在处理 rope_type='default' 的
    RotaryEmbedding 时会调用 module.compute_default_rope_parameters(config)，
    而 qwen-asr 的 RotaryEmbedding 类没有该方法，导致 AttributeError。
    """
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextRotaryEmbedding,
        )

        if hasattr(Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):
            return

        Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = staticmethod(
            _compute_default_rope_parameters
        )
        logger.debug(
            "[ASR patch] 已为 Qwen3ASRThinkerTextRotaryEmbedding 注入 compute_default_rope_parameters"
        )
    except Exception as e:
        logger.debug(f"[ASR patch] Qwen3ASRThinkerTextRotaryEmbedding 补丁未生效: {e}")


def _patch_tokenizer_mistral_regex() -> None:
    """
    修复 transformers 5.3.0 中 fix_mistral_regex 重复传入的 TypeError。

    qwen-asr 在调用 AutoProcessor.from_pretrained 时显式传入 fix_mistral_regex=True。
    该参数会一路传递到 TokenizersBackend.__init__，而 5.3.0 的 __init__ 在调用
    _patch_mistral_regex 时同时显式传了 fix_mistral_regex 并通过 **kwargs 再次传入，
    导致重复关键字参数 TypeError。

    这里直接修补 TokenizersBackend.__init__ 的源码，在调用 _patch_mistral_regex 前
    从 kwargs 中弹出 fix_mistral_regex，再显式传入，从而消除重复。
    """
    try:
        import inspect
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        if getattr(TokenizersBackend, "_asr_mistral_regex_patched", False):
            return

        orig_source = inspect.getsource(TokenizersBackend.__init__)
        # 去掉方法定义行，保留方法体
        lines = orig_source.splitlines()
        body_lines = []
        for line in lines:
            if line.strip().startswith("def __init__("):
                continue
            # 去掉首行的 4 空格缩进（源码中方法体相对类定义缩进一层）
            if line.startswith("        "):
                body_lines.append(line[8:])
            else:
                body_lines.append(line)
        body = "\n".join(body_lines)

        target = (
            '    self._tokenizer = self._patch_mistral_regex(\n'
            '        self._tokenizer,\n'
            '        self.init_kwargs.get("name_or_path", None),\n'
            '        init_kwargs=self.init_kwargs,\n'
            '        fix_mistral_regex=kwargs.get("fix_mistral_regex"),\n'
            '        **kwargs,\n'
            '    )'
        )
        replacement = (
            '    _fix_mistral_regex = kwargs.pop("fix_mistral_regex", None)\n'
            '    self._tokenizer = self._patch_mistral_regex(\n'
            '        self._tokenizer,\n'
            '        self.init_kwargs.get("name_or_path", None),\n'
            '        init_kwargs=self.init_kwargs,\n'
            '        fix_mistral_regex=_fix_mistral_regex,\n'
            '        **kwargs,\n'
            '    )'
        )

        if target not in body:
            logger.debug("[ASR patch] 未找到 TokenizersBackend.__init__ 目标代码，跳过源码补丁")
            return

        new_body = body.replace(target, replacement)
        # exec 编译的函数中 super() 无法正确解析 __class__ cell，
        # 因此把 super().__init__ 替换为对 PreTrainedTokenizerBase 的显式调用。
        new_body = new_body.replace(
            "super().__init__(**kwargs)",
            "PreTrainedTokenizerBase.__init__(self, **kwargs)",
        )

        # 构造新的 __init__ 函数字符串
        new_func_source = "def __init__(self, *args, **kwargs):\n" + "\n".join(
            "    " + line for line in new_body.splitlines()
        )

        namespace = {}
        # 将 TokenizersBackend 模块中的符号导入命名空间，便于 exec 解析
        import transformers.tokenization_utils_tokenizers as _tu

        for _name in dir(_tu):
            if not _name.startswith("__"):
                namespace[_name] = getattr(_tu, _name)
        namespace["PreTrainedTokenizerBase"] = _tu.PreTrainedTokenizerBase
        namespace["logger"] = _tu.logger
        namespace["AddedToken"] = _tu.AddedToken

        exec(new_func_source, namespace)
        TokenizersBackend.__init__ = namespace["__init__"]
        TokenizersBackend._asr_mistral_regex_patched = True
        logger.debug("[ASR patch] 已修复 TokenizersBackend.__init__ 重复参数问题")
    except Exception as e:
        logger.debug(f"[ASR patch] fix_mistral_regex 补丁未生效: {e}")


# ---------------------------------------------------------------------------
# 模块加载时先执行一次补丁（快速路径）
# ---------------------------------------------------------------------------
_patch_check_model_inputs()
_patch_rope_init_functions()
_patch_asr_rotary_embedding()
_patch_tokenizer_mistral_regex()


class PyTorchQwen3ASRInferenceEngine(InferenceEngine):
    """
    Qwen3-ASR 推理引擎，支持 exo 框架标准接口。

    输入：音频文件路径（通过 prompt 传入）。
    输出：转录后的文本字符串。

    复用 qwen-asr 包的 Qwen3ASRModel，其内部已封装：
      - Whisper 风格 log-mel 特征提取
      - Audio Transformer (AuT) Encoder
      - Projector
      - Qwen3-0.6B 文本解码器与自回归生成
      - 长音频自动切分
    """

    def __init__(
        self,
        shard_downloader: ShardDownloader,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.shard_downloader = shard_downloader
        self.model_path = model_path
        self.device = self._get_best_device("auto")
        self.shard: Optional[Shard] = None
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._shard_lock = asyncio.Lock()

        logger.info(
            f"[PyTorchQwen3ASRInferenceEngine] 初始化完成，设备: {self.device}"
        )

    def _get_best_device(self, device_hint: str = "auto") -> torch.device:
        """自动选择最佳设备，优先 GPU。"""
        if device_hint == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                logger.info(f"优先使用GPU: {gpu_name} ({memory_gb:.1f}GB)")
            elif torch.backends.mps.is_available():
                device_name = "mps"
                logger.info("使用Apple Silicon GPU (MPS)")
            else:
                device_name = "cpu"
                logger.info("回退到CPU推理")
        else:
            device_name = device_hint
        return torch.device(device_name)

    async def ensure_shard(self, shard: Shard):
        """确保模型分片已加载。当前实现加载完整模型。"""
        # 快速路径：同一个 shard 不需要重复加载
        if self.shard is not None and self.shard == shard:
            return

        async with self._shard_lock:
            # 在锁内再次检查，防止并发重复加载
            if self.shard is not None and self.shard == shard:
                return

            if self.shard_downloader is not None:
                model_path = await self.shard_downloader.ensure_shard(
                    shard, self.__class__.__name__
                )
                self.model_path = str(model_path)
            elif self.model_path is None:
                raise RuntimeError(
                    "未提供 model_path 且 shard_downloader 为 None，无法加载模型"
                )

            if self.shard != shard or self.model is None:
                await self._load_model()
                self.shard = shard

    async def _load_model(self):
        """异步加载 Qwen3ASRModel。"""
        # 在 import qwen_asr 前再次执行补丁，确保即使被其他代码提前导入也能生效
        _patch_check_model_inputs()
        _patch_rope_init_functions()

        import qwen_asr  # noqa: F401  # 触发 AutoConfig/AutoModel/AutoProcessor 注册
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

        # 在实例化模型前修补 config、RoPE 类与 tokenizer
        _patch_asr_config()
        _patch_asr_rotary_embedding()
        _patch_tokenizer_mistral_regex()

        model_path = self.model_path
        if model_path is None:
            raise RuntimeError("model_path 为空，无法加载模型")

        # 支持传入 repo id 时自动解析本地缓存路径
        if not os.path.exists(model_path):
            is_repo_id = (
                "/" in model_path
                and not model_path.startswith(".")
                and not model_path.startswith("/")
                and "\\" not in model_path
            )
            if is_repo_id:
                cache_base = Path.home() / ".cache" / "exo" / "downloads"
                local_dir_name = model_path.replace("/", "--")
                resolved_path = cache_base / local_dir_name
                if resolved_path.exists() and resolved_path.is_dir():
                    model_path = str(resolved_path)
                    logger.info(f"[ASR] Repo ID 解析为本地缓存路径: {model_path}")

        logger.info(f"[ASR] 正在加载 Qwen3-ASR 模型: {model_path}")

        def _do_load():
            # 根据设备选择精度与加载方式
            use_cuda = self.device.type == "cuda"
            torch_dtype = (
                torch.bfloat16
                if use_cuda and torch.cuda.is_bf16_supported()
                else torch.float32
            )

            model = Qwen3ASRModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                local_files_only=True,
                device_map=None,  # 禁用自动 device_map，手动管理设备
            )

            # 手动将模型移动到目标设备（如果底层是 transformers 模型）
            inner_model = getattr(model, "model", None)
            if inner_model is not None:
                inner_model = inner_model.to(self.device)
                model.model = inner_model
            else:
                model = model.to(self.device)

            return model

        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, _do_load)
        except Exception as e:
            logger.error(f"[ASR] 加载 Qwen3-ASR 模型失败: {e}")
            traceback.print_exc()
            raise

        # 提取 processor / tokenizer 供后续 decode 使用
        self.processor = getattr(self.model, "processor", None)
        self.tokenizer = (
            getattr(self.processor, "tokenizer", None) if self.processor else None
        )
        if self.tokenizer is None:
            # 兜底：尝试直接从模型路径加载 tokenizer
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )

        logger.info("[ASR] Qwen3-ASR 模型加载完成")

    async def encode(
        self, shard: Shard, prompt: str, enable_thinking: bool = False
    ) -> np.ndarray:
        """
        编码输入。

        对于 ASR，prompt 应传入音频文件路径。这里将路径编码为 UTF-8 字节数组，
        与 TTS 保持一致，把实际音频解析推迟到 infer_tensor。
        """
        await self.ensure_shard(shard)
        return np.array(list(prompt.encode("utf-8")), dtype=np.uint8)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """解码 token 序列或文本字节为可读文本。"""
        await self.ensure_shard(shard)

        if tokens is None or len(tokens) == 0:
            return ""

        # 若 tokens 是文本字符串的字节表示（来自 infer_tensor 的文本输出），直接解码
        if tokens.dtype == np.uint8:
            try:
                return tokens.tobytes().decode("utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"[ASR decode] 字节解码失败: {e}")
                return ""

        # 否则按 token ID 解码
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer 未加载，无法解码")

        try:
            # 确保 tokens 是一维的
            if tokens.ndim > 1:
                tokens = tokens.squeeze()
                if tokens.ndim > 1:
                    tokens = tokens[0]
            token_list = tokens.tolist()
            # 去除 padding / special tokens
            if isinstance(token_list, int):
                token_list = [token_list]
            text = self.tokenizer.decode(token_list, skip_special_tokens=True)
            return text.encode("utf-8", errors="ignore").decode("utf-8")
        except Exception as e:
            logger.error(f"[ASR decode] Token 解码失败: {e}")
            return f"<decode_error: {e}>"

    async def sample(
        self,
        x: np.ndarray,
        temp: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        generated_tokens: list = None,
        shard: "Shard" = None,
    ) -> np.ndarray:
        """
        ASR 已经在模型内部完成自回归生成，sample 阶段直接透传。
        """
        return x

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        ASR 主推理入口。

        input_data: 音频文件路径的 UTF-8 字节数组。
        inference_state 可选字段：
          - language: 语言名称，如 "Chinese" / "English"
          - context: 上下文文本（可选）
          - max_new_tokens: 最大生成 token 数
          - return_timestamps: 是否返回时间戳（需要 forced_aligner，当前不启用）
        """
        await self.ensure_shard(shard)

        if self.model is None:
            raise RuntimeError("ASR 模型未加载")

        # 解析音频路径
        try:
            audio_path = input_data.tobytes().decode("utf-8", errors="ignore").strip()
        except Exception as e:
            raise ValueError(f"无法解析音频路径: {e}")

        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        inference_state = inference_state or {}
        language = inference_state.get("language") or inference_state.get("lang")
        context = inference_state.get("context", "")
        return_timestamps = inference_state.get("return_timestamps", False)
        max_new_tokens = inference_state.get("max_new_tokens", 512)

        # 动态更新模型 max_new_tokens
        if hasattr(self.model, "max_new_tokens") and max_new_tokens is not None:
            self.model.max_new_tokens = max_new_tokens

        logger.info(
            f"[ASR infer_tensor] request_id={request_id}, audio={audio_path}, "
            f"language={language}"
        )

        def _infer():
            results = self.model.transcribe(
                audio=audio_path,
                context=context,
                language=language,
                return_time_stamps=return_timestamps,
            )
            # results 是 List[ASRTranscription]
            if not results:
                return ""
            result = results[0]
            # 返回文本
            text = getattr(result, "text", "")
            return str(text)

        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(None, _infer)

        # 更新 inference_state
        enhanced_state = inference_state.copy()
        enhanced_state.update(
            {
                "request_id": request_id,
                "audio_path": audio_path,
                "transcription": transcription,
            }
        )

        # 将文本编码为 UTF-8 字节数组返回，与 encode 阶段对称，decode 可直接解析
        output = np.array(list(transcription.encode("utf-8")), dtype=np.uint8)
        return output, enhanced_state

    async def load_checkpoint(self, shard: Shard, path: str):
        """加载模型检查点。"""
        logger.info(f"[ASR] 加载检查点: {path}")
        self.model_path = path
        await self.ensure_shard(shard)

    async def get_embedding(self, token_tensor, shard: Shard = None):
        """ASR 不支持常规 token embedding 获取，返回 None。"""
        return None

    def get_processor(self, model_id: str = None):
        """获取 ASR processor。"""
        return self.processor

    def get_tokenizer(self, model_id: str = None):
        """获取 ASR tokenizer。"""
        return self.tokenizer
