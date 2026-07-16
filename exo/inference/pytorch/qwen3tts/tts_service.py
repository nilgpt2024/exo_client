#!/usr/bin/env python3
"""
Qwen3-TTS 独立子进程推理服务

设计目标：
- 在独立的 Python 虚拟环境中运行（transformers 4.57.3 + 官方 qwen-tts）
- 通过 HTTP 接口暴露 TTS 生成能力
- 主进程（exo）通过 HTTP 调用，避免 transformers 版本冲突

运行方式（在 venv 中）:
    F:\\Qwen3-TTS\\.venv_tts_4573\\Scripts\\python.exe tts_service.py --model_path <path> --port 0

--port 0 表示让系统自动分配端口，服务启动后会打印实际端口。
"""

import argparse
import json
import logging
import os
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

# Qwen3-TTS 源码路径（如果通过 pip install -e 安装则不需要）
_qwen_tts_path = Path("F:/Qwen3-TTS")
if _qwen_tts_path.exists() and str(_qwen_tts_path) not in sys.path:
    sys.path.insert(0, str(_qwen_tts_path))

# 兼容性补丁: qwen_tts 基于 transformers 4.46, 其中 check_model_inputs 是装饰器工厂
# @check_model_inputs()。transformers 4.57 中它仍是装饰器工厂但行为略有不同，且在某些
# 调用场景下会错误地过滤掉 forward 的 kwargs（如 inputs_embeds）。由于该装饰器仅用于
# 输入校验/输出记录，对 TTS 推理语义无影响，这里在 qwen_tts 任何模块导入前将其替换为
# 透传装饰器，避免运行时 TypeError。
try:
    import transformers.utils.generic as _generic

    def _compat_check_model_inputs(*args, **kwargs):
        # 直接用作 @check_model_inputs（不带括号）
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        # 用作装饰器工厂 @check_model_inputs(...) 时，返回一个直接透传原函数的装饰器
        def _transparent_wrapper(fn):
            return fn

        return _transparent_wrapper

    if hasattr(_generic, "check_model_inputs"):
        _orig = _generic.check_model_inputs
        if callable(_orig):
            _generic._original_check_model_inputs = _orig
    _generic.check_model_inputs = _compat_check_model_inputs
except Exception:
    pass

import torch
from qwen_tts import Qwen3TTSModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class TTSService:
    """包装 Qwen3TTSModel，提供线程安全的生成接口。"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Qwen3TTSModel] = None
        self._load_model()

    def _load_model(self):
        logger.info(f"加载 Qwen3-TTS 模型: {self.model_path}")
        logger.info(f"使用设备: {self.device}")

        # Qwen3-TTS 官方推荐使用 bfloat16；Pascal (P100) 不支持 BF16，回退 FP32
        use_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        logger.info(f"使用 dtype: {dtype}")

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
        )
        logger.info("Qwen3-TTS 模型加载完成")

    def generate(
        self,
        text: str,
        language: str = "Chinese",
        instruct: Optional[str] = None,
        mode: str = "voice_design",
        max_new_tokens: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        生成语音并保存到 output_path（若未提供则创建临时文件）。
        返回包含 output_path、sample_rate、duration 的字典。
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if mode != "voice_design":
            raise ValueError(f"当前仅支持 voice_design 模式，收到: {mode}")

        instruct = instruct or ""
        gen_kwargs = {}
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        t0 = time.time()
        wavs, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            **gen_kwargs,
        )
        t1 = time.time()

        audio = np.array(wavs[0], dtype=np.float32)
        if output_path is None:
            import tempfile

            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        output_path = str(output_path)
        self._save_wav(audio, sr, output_path)

        return {
            "output_path": output_path,
            "sample_rate": sr,
            "duration": float(len(audio) / sr),
            "generate_time": float(t1 - t0),
        }

    @staticmethod
    def _save_wav(audio: np.ndarray, sample_rate: int, path: str):
        """将 float32 音频保存为 16-bit PCM WAV。"""
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())


class RequestHandler(BaseHTTPRequestHandler):
    service: Optional[TTSService] = None

    def log_message(self, format, *args):
        logger.info(format % args)

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": f"未知路径: {self.path}"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            req = json.loads(body)
        except Exception as e:
            self._send_json(400, {"error": f"请求解析失败: {e}"})
            return

        try:
            text = req.get("text")
            if not text:
                raise ValueError("请求体必须包含 'text'")

            result = self.service.generate(
                text=text,
                language=req.get("language", "Chinese"),
                instruct=req.get("instruct"),
                mode=req.get("mode", "voice_design"),
                max_new_tokens=req.get("max_new_tokens"),
                output_path=req.get("output_path"),
            )
            self._send_json(200, result)
        except Exception as e:
            logger.exception("生成失败")
            self._send_json(500, {"error": f"生成失败: {e}"})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model_loaded": self.service is not None})
        else:
            self._send_json(404, {"error": f"未知路径: {self.path}"})


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 子进程推理服务")
    parser.add_argument("--model_path", required=True, help="Qwen3-TTS 模型本地路径")
    parser.add_argument("--port", type=int, default=0, help="HTTP 服务端口，0 表示自动分配")
    parser.add_argument("--device", default=None, help="推理设备，如 cuda:0 或 cpu")
    args = parser.parse_args()

    service = TTSService(args.model_path, device=args.device)
    RequestHandler.service = service

    server = HTTPServer(("127.0.0.1", args.port), RequestHandler)
    actual_port = server.server_address[1]

    # 打印端口信息到 stdout，方便父进程捕获
    print(f"TTS_SERVICE_PORT={actual_port}", flush=True)
    logger.info(f"TTS 服务已启动: http://127.0.0.1:{actual_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，关闭服务")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
