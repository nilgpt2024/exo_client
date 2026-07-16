#!/usr/bin/env python3
"""
PyTorch Qwen3-TTS Inference Engine for exo (IPC 子进程版本)

Qwen3-TTS 官方依赖 transformers 4.57.3，与 exo 主环境的 transformers 5.3.0 不兼容。
本引擎通过启动独立的 Python 子进程（运行 TTS 服务）并通过 HTTP 与其通信，
解决版本冲突问题，同时保持 exo InferenceEngine 接口不变。

当前子进程服务仅支持 voice_design 模式。其它模式如需支持，需同步扩展
 tts_service.py 的 generate 接口。
"""

import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import soundfile as sf
import torch

from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.inference.shard import Shard

logger = logging.getLogger(__name__)


class PyTorchQwen3TTSInferenceEngine(InferenceEngine):
    """
    Qwen3-TTS 推理引擎 - 通过子进程服务运行
    """

    # 子进程服务配置（硬编码路径，与当前部署环境一致）
    _VENV_PYTHON = Path("F:/Qwen3-TTS/.venv_tts_4573/Scripts/python.exe")
    _SERVICE_SCRIPT = Path(__file__).resolve().parent / "tts_service.py"

    def __init__(
        self,
        shard_downloader: ShardDownloader,
        model_path: str = None,
        use_subprocess: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.shard_downloader = shard_downloader
        self.model_path = model_path
        self.use_subprocess = use_subprocess
        self.device = self._get_best_device()
        self.shard = None

        # 子进程服务状态
        self._service_proc: Optional[asyncio.subprocess.Process] = None
        self._service_port: Optional[int] = None
        self._service_url: Optional[str] = None
        self._service_lock = asyncio.Lock()

        # 加载锁
        self._shard_lock = asyncio.Lock()

    def _get_best_device(self) -> torch.device:
        """自动选择最佳设备（仅用于决定子进程服务使用的设备参数）"""
        if torch.cuda.is_available():
            device_name = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"[TTS] 检测到GPU: {gpu_name} ({memory_gb:.1f}GB)，子进程将使用 cuda:0")
        else:
            device_name = "cpu"
            logger.info("[TTS] 未检测到GPU，子进程将使用 CPU")
        return torch.device(device_name)

    async def ensure_shard(self, shard: Shard):
        """确保模型分片已加载（即子进程服务已启动并加载模型）"""
        async with self._shard_lock:
            if self.shard == shard and self._service_url is not None:
                return

            # 下载/确定模型路径
            if self.shard_downloader is not None:
                model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
                self.model_path = str(model_path)

            if self.use_subprocess:
                await self._ensure_subprocess_service()
            else:
                raise RuntimeError("Qwen3-TTS 当前仅支持子进程模式，请将 use_subprocess 设为 True")

            self.shard = shard
            logger.info(f"✅ Qwen3-TTS 子进程服务准备就绪: {shard.model_id} @ {self.model_path}")

    async def _ensure_subprocess_service(self):
        """启动或复用子进程 TTS 服务"""
        async with self._service_lock:
            if self._service_url is not None:
                # 已有服务，先检查健康
                if await self._health_check():
                    return
                # 健康检查失败则重启
                await self._stop_subprocess_service()

            if not self._VENV_PYTHON.exists():
                raise RuntimeError(f"TTS 子进程 Python 不存在: {self._VENV_PYTHON}")
            if not self._SERVICE_SCRIPT.exists():
                raise RuntimeError(f"TTS 服务脚本不存在: {self._SERVICE_SCRIPT}")
            if not self.model_path:
                raise RuntimeError("model_path 未设置，无法启动 TTS 子进程服务")

            logger.info(f"[TTS] 启动子进程服务: {self._VENV_PYTHON} {self._SERVICE_SCRIPT}")
            logger.info(f"[TTS] 模型路径: {self.model_path}")

            cmd = [
                str(self._VENV_PYTHON),
                str(self._SERVICE_SCRIPT),
                "--model_path", str(self.model_path),
                "--port", "0",
                "--device", str(self.device),
            ]

            self._service_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # 启动 stderr 读取任务，避免子进程 stderr 缓冲区满而阻塞
            stderr_task = asyncio.create_task(self._drain_service_stderr())

            # 读取 stdout 直到获取 TTS_SERVICE_PORT=xxx（模型加载可能耗时较长，给 5 分钟）
            port = await self._read_service_port(timeout=300.0)

            # 端口已获取，取消 stderr  drain 任务（让它继续读取直到取消）
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

            if port is None:
                await self._stop_subprocess_service()
                raise RuntimeError("未能从 TTS 子进程服务获取端口号")

            self._service_port = port
            self._service_url = f"http://127.0.0.1:{port}"
            logger.info(f"[TTS] 子进程服务端口: {port}")

            # 等待健康检查通过
            if not await self._wait_for_health(timeout=300.0):
                await self._stop_subprocess_service()
                raise RuntimeError("TTS 子进程服务健康检查失败")

            logger.info(f"[TTS] 子进程服务健康检查通过: {self._service_url}")

    async def _drain_service_stderr(self):
        """持续读取子进程 stderr 并记录日志，防止缓冲区满"""
        assert self._service_proc is not None
        assert self._service_proc.stderr is not None
        try:
            while True:
                line = await self._service_proc.stderr.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str:
                    logger.info(f"[TTS service stderr] {line_str}")
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _read_service_port(self, timeout: float = 30.0) -> Optional[int]:
        """从子进程 stdout 读取 TTS_SERVICE_PORT=xxx"""
        assert self._service_proc is not None
        assert self._service_proc.stdout is not None

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = await asyncio.wait_for(
                    self._service_proc.stdout.readline(),
                    timeout=max(0.1, deadline - time.time()),
                )
            except asyncio.TimeoutError:
                break

            if not line:
                break

            line_str = line.decode("utf-8", errors="replace").strip()
            if line_str.startswith("TTS_SERVICE_PORT="):
                try:
                    return int(line_str.split("=", 1)[1])
                except ValueError:
                    logger.warning(f"[TTS] 无法解析服务端口行: {line_str}")
            else:
                # 其它 stdout 行作为信息日志
                logger.info(f"[TTS service stdout] {line_str}")

        return None

    async def _wait_for_health(self, timeout: float = 300.0, interval: float = 0.5) -> bool:
        """轮询 /health 直到服务就绪"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if await self._health_check():
                return True
            await asyncio.sleep(interval)
        return False

    async def _health_check(self) -> bool:
        """向子进程服务发送健康检查"""
        if self._service_url is None:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._service_url}/health", timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.debug(f"[TTS] 健康检查异常: {e}")
            return False

    async def _stop_subprocess_service(self):
        """停止子进程服务"""
        if self._service_proc is None:
            return

        proc = self._service_proc
        self._service_proc = None
        self._service_port = None
        self._service_url = None

        try:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
        except Exception as e:
            logger.warning(f"[TTS] 停止子进程服务时出错: {e}")

    async def encode(self, shard: Shard, prompt: str, enable_thinking: bool = False) -> np.ndarray:
        """编码文本为输入张量（TTS 只需把 prompt 编码为字节数组）"""
        await self.ensure_shard(shard)
        return np.array(list(prompt.encode("utf-8")), dtype=np.uint8)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        """解码 token 为文本（TTS 输出是音频，返回空字符串）"""
        return ""

    async def sample(self, x: np.ndarray, temp: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                     repetition_penalty: float = 1.0, generated_tokens: list = None) -> np.ndarray:
        """采样 - TTS 模型不使用此接口"""
        return x

    async def infer_tensor(
        self,
        request_id: str,
        shard: Shard,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None,
    ) -> Tuple[Union[np.ndarray, dict], Optional[dict]]:
        """从张量推理 - TTS 主入口"""
        await self.ensure_shard(shard)

        # 解码文本
        text_or_json = input_data.tobytes().decode("utf-8")

        # 允许 inference_state["texts"] 覆盖 prompt
        if inference_state and "texts" in inference_state:
            texts = inference_state["texts"]
        else:
            try:
                parsed = json.loads(text_or_json)
                if isinstance(parsed, list):
                    texts = parsed
                else:
                    texts = text_or_json
            except json.JSONDecodeError:
                texts = text_or_json

        is_batch = isinstance(texts, list)
        text_list = texts if is_batch else [texts]

        # 当前服务仅支持单条 voice_design
        if is_batch and len(text_list) > 1:
            raise NotImplementedError("子进程服务当前仅支持单条 TTS 生成")

        text = text_list[0]
        mode = (inference_state.get("tts_mode", "voice_design") if inference_state else "voice_design")
        if mode != "voice_design":
            raise NotImplementedError(f"子进程服务当前仅支持 voice_design 模式，收到: {mode}")

        language = inference_state.get("language", "Chinese") if inference_state else "Chinese"
        instruct = inference_state.get("instruct") if inference_state else None
        max_new_tokens = inference_state.get("max_new_tokens") if inference_state else None

        payload = {
            "text": text,
            "language": language,
            "mode": mode,
        }
        if instruct is not None:
            payload["instruct"] = instruct
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens

        # 调用子进程服务
        result = await self._call_generate(payload)
        output_path = result.get("output_path")
        if not output_path or not Path(output_path).exists():
            raise RuntimeError(f"TTS 子进程服务未返回有效音频文件: {output_path}")

        # 读取音频
        audio, sr = sf.read(output_path, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 清理临时文件（服务创建在系统 temp 目录，这里不删避免调试困难，可后续启用）
        # try:
        #     os.remove(output_path)
        # except Exception:
        #     pass

        if inference_state is None:
            inference_state = {}
        inference_state["sample_rate"] = int(sr)
        inference_state["duration"] = float(len(audio) / sr)
        inference_state["generate_time"] = result.get("generate_time", 0.0)

        return audio, inference_state

    async def _call_generate(self, payload: dict) -> dict:
        """向子进程服务发送 /generate 请求"""
        assert self._service_url is not None
        url = f"{self._service_url}/generate"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600.0),
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error = data.get("error", f"HTTP {resp.status}")
                        raise RuntimeError(f"TTS 子进程服务生成失败: {error}")
                    return data
        except aiohttp.ClientError as e:
            raise RuntimeError(f"TTS 子进程服务请求失败: {e}")

    async def infer_prompt(
        self,
        request_id: str,
        shard: Shard,
        prompt: str,
        inference_state: Optional[dict] = None,
    ) -> Tuple[Union[np.ndarray, dict], Optional[dict]]:
        """从文本提示推理 - TTS 便捷入口"""
        await self.ensure_shard(shard)
        input_data = np.array(list(prompt.encode("utf-8")), dtype=np.uint8)
        return await self.infer_tensor(request_id, shard, input_data, inference_state)

    async def create_voice_clone_prompt(
        self,
        shard: Shard,
        ref_audio: Union[str, List[str]],
        ref_text: Optional[Union[str, List[str]]] = None,
        x_vector_only_mode: bool = False,
    ) -> Dict[str, Any]:
        """两阶段 VoiceClone：当前子进程服务尚未支持"""
        raise NotImplementedError("VoiceClone prompt 创建当前未在 IPC 模式下实现")

    async def load_checkpoint(self, shard: Shard, path: str):
        """加载模型检查点（IPC 模式下无需额外操作）"""
        await self.ensure_shard(shard)
        logger.info(f"Qwen3-TTS IPC 模型检查点加载: {path}")

    async def save_checkpoint(self, shard: Shard, path: str):
        """保存模型检查点（IPC 模式下无需额外操作）"""
        logger.info(f"Qwen3-TTS IPC 模型检查点保存: {path}")

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024 ** 3
            memory_info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return memory_info

    async def shutdown(self):
        """关闭子进程服务"""
        await self._stop_subprocess_service()

    def __del__(self):
        """析构时尝试停止子进程服务（同步 kill，避免事件循环已关闭导致异常）"""
        if self._service_proc is not None:
            try:
                if self._service_proc.returncode is None:
                    self._service_proc.kill()
            except Exception:
                pass
