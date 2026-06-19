#!/usr/bin/env python3
"""
统一的 PyTorch 推理引擎入口
根据 models.py 中的配置自动选择正确的具体引擎实现
支持多模型并发加载
"""

import numpy as np
import logging
import time
from typing import Optional, Tuple, Dict, Any, List
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo import models

logger = logging.getLogger(__name__)


class PyTorchInferenceEngine(InferenceEngine):
    """统一的 PyTorch 推理引擎

    根据 models.py 的 repo 配置自动选择使用正确的引擎:
    - PyTorchQwen3VLInferenceEngine: 用于 Qwen3-VL 视觉模型
    - PyTorchQwen3InferenceEngine: 用于 Qwen3 文本模型和其他模型

    支持多模型并发加载，每个模型有独立的引擎实例
    """

    def __init__(self, shard_downloader: ShardDownloader, model_path: str = None, **kwargs):
        super().__init__()
        self.shard_downloader = shard_downloader
        self.model_path = model_path
        self._instance_create_time = time.time()
        self._instance_id = id(self)
        self._call_count = 0
        # 使用字典存储多个引擎实例: model_id -> engine
        self._engines: Dict[str, Any] = {}
        self._current_model_id: Optional[str] = None
        # 默认引擎（向后兼容）
        self._default_engine = None
        # 追踪每个模型的加载状态
        self._loaded_shards: Dict[str, Shard] = {}
        # 追踪引擎操作历史（用于调试空缓存问题）
        self._engine_history: List[str] = []

        print(f"[PyTorchInferenceEngine] [INIT] Instance created: id={self._instance_id}, "
              f"time={time.strftime('%H:%M:%S', time.localtime(self._instance_create_time))}")

    def _get_engine_for_model(self, model_id: str):
        """根据 models.py 配置获取对应的引擎类型

        动态导入引擎类，避免硬编码
        """
        safe_model_id = model_id.split("::")[0] if "::" in model_id else model_id
        # 从 models.py 获取该模型支持的引擎列表
        model_info = models.model_cards.get(safe_model_id, {})
        repo_config = model_info.get("repo", {})

        # 动态检测 PyTorch 引擎（以 'PyTorch' 开头，以 'InferenceEngine' 结尾）
        pytorch_engines = [
            k for k in repo_config.keys()
            if k.startswith("PyTorch") and k.endswith("InferenceEngine")
        ]

        if pytorch_engines:
            # 使用第一个找到的 PyTorch 引擎
            engine_class_name = pytorch_engines[0]
            # 将类名转换为模块路径
            # 例如: PyTorchFaraInferenceEngine -> exo.inference.pytorch.fara.pytorch_inference_engine
            # 或者: PyTorchQwen3VLInferenceEngine -> exo.inference.pytorch.qwen3vl.pytorch_inference_engine
            engine_type = engine_class_name.replace("PyTorch", "").replace("InferenceEngine", "").lower()
            module_path = f"exo.inference.pytorch.{engine_type}.pytorch_inference_engine"

            try:
                # 动态导入模块
                module = __import__(module_path, fromlist=[engine_class_name])
                # 获取引擎类
                engine_class = getattr(module, engine_class_name)
                # 创建引擎实例
                return engine_class(self.shard_downloader, model_path=self.model_path)
            except (ImportError, AttributeError) as e:
                print(f"[PyTorchInferenceEngine] 动态导入引擎失败: {engine_class_name}, 错误: {e}")
                # 回退到默认引擎
                pass

        # 默认使用 Qwen3VL 引擎作为兜底
        from exo.inference.pytorch.qwen3vl.pytorch_inference_engine import PyTorchQwen3VLInferenceEngine
        return PyTorchQwen3VLInferenceEngine(self.shard_downloader, model_path=self.model_path)

    def _ensure_engine(self, shard: Shard):
        """确保引擎已初始化（支持多模型）

        防护逻辑：
        1. 检查 _engines 缓存是否已有该 model_id 的引擎
        2. 如果已有引擎，检查其权重是否已加载
        3. 只有在必要时才创建新引擎，避免重复创建导致权重反复加载
        """
        self._call_count += 1
        model_id = shard.model_id
        call_num = self._call_count
        elapsed = time.time() - self._instance_create_time

        history_entry = f"[{call_num}] _ensure_engine({model_id}) engines={list(self._engines.keys())}"
        self._engine_history.append(history_entry)
        if len(self._engine_history) > 20:
            self._engine_history = self._engine_history[-20:]

        if model_id in self._engines:
            existing_engine = self._engines[model_id]
            has_model = getattr(existing_engine, 'model', None) is not None
            has_shard = getattr(existing_engine, 'shard', None) is not None

            if has_model or has_shard:
                logger.debug(f"[PyTorchInferenceEngine] [OK] Reuse existing engine: {model_id} "
                            f"(instance={self._instance_id}, call=#{call_num}, "
                            f"存活={elapsed:.1f}s, 已有模型={has_model}, 已有分片={has_shard})")
            else:
                logger.debug(f"[PyTorchInferenceEngine] ♻️ 引擎存在但未加载权重: {model_id}，将触发 ensure_shard")
        else:
            print(f"[PyTorchInferenceEngine] [NEW-ENGINE] Create new engine: {model_id} (instance={self._instance_id}, "
                  f"call=#{call_num}, 存活={elapsed:.1f}s, 当前已缓存: {list(self._engines.keys())})")

            if len(self._engines) == 0 and call_num > 1:
                print(f"[PyTorchInferenceEngine] [WARN] Cache empty but not first call! "
                      f"历史记录: {self._engine_history[-5:]}")

            import traceback
            print(f"[PyTorchInferenceEngine] [STACK] Call stack (instance={self._instance_id}):")
            for line in traceback.format_stack()[-8:-1]:
                print(f"  {line.strip()}")

            self._engines[model_id] = self._get_engine_for_model(model_id)

        self._current_model_id = model_id
        self._default_engine = self._engines[model_id]

    def dump_state(self) -> dict:
        """输出当前状态（用于调试）"""
        return {
            "instance_id": self._instance_id,
            "age_seconds": time.time() - self._instance_create_time,
            "call_count": self._call_count,
            "cached_models": list(self._engines.keys()),
            "loaded_shards": {k: str(v) for k, v in self._loaded_shards.items()},
            "current_model_id": self._current_model_id,
            "recent_history": self._engine_history[-10:] if self._engine_history else [],
        }

    def _get_engine(self, shard: Optional[Shard] = None) -> Any:
        """获取引擎实例"""
        if shard is not None and shard.model_id in self._engines:
            return self._engines[shard.model_id]
        if self._current_model_id and self._current_model_id in self._engines:
            return self._engines[self._current_model_id]
        if self._default_engine is not None:
            return self._default_engine
        return None

    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return list(self._engines.keys())

    def has_model(self, model_id: str) -> bool:
        """检查是否已加载指定模型"""
        return model_id in self._engines

    async def encode(self, shard: Shard, prompt: str, enable_thinking: bool = False) -> np.ndarray:
        self._ensure_engine(shard)
        await self.ensure_shard(shard)
        return await self._engines[shard.model_id].encode(shard, prompt, enable_thinking)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        self._ensure_engine(shard)
        await self.ensure_shard(shard)
        return await self._engines[shard.model_id].decode(shard, tokens)

    async def sample(self, x: np.ndarray, temp: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                     repetition_penalty: float = 1.0, generated_tokens: list = None, shard: Shard = None) -> np.ndarray:
        # 如果传入了 shard，使用对应的引擎；否则使用当前引擎
        if shard is not None and shard.model_id in self._engines:
            engine = self._engines[shard.model_id]
        else:
            engine = self._get_engine()
        if engine is None:
            raise RuntimeError("Engine not initialized. Call encode or infer_prompt first.")
        # sample 只需要 tokenizer，不需要重新加载模型
        # 直接调用底层引擎的 sample 方法，不调用 ensure_shard
        return await engine.sample(x, temp, top_p, top_k, repetition_penalty, generated_tokens)

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray,
                          inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        self._ensure_engine(shard)
        await self.ensure_shard(shard)
        return await self._engines[shard.model_id].infer_tensor(request_id, shard, input_data, inference_state)

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str,
                          inference_state: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        self._ensure_engine(shard)
        await self.ensure_shard(shard)
        return await self._engines[shard.model_id].infer_prompt(request_id, shard, prompt, inference_state)

    async def load_checkpoint(self, shard: Shard, path: str):
        """加载模型检查点（支持多模型）"""
        self._ensure_engine(shard)
        await self._engines[shard.model_id].load_checkpoint(shard, path)
        print(f"[PyTorchInferenceEngine] 模型 {shard.model_id} 加载完成，当前已加载模型: {list(self._engines.keys())}")

    async def ensure_shard(self, shard: Shard):
        """确保分片已加载（带去重逻辑）

        防护逻辑：
        1. 检查 _loaded_shards 是否已有该 model_id 的记录
        2. 比较新旧 shard 的关键属性（model_id, start_layer, end_layer, instance_id）
        3. 只有在分片确实变化时才重新加载权重
        """
        import traceback

        is_suspicious = (len(self._engines) == 0 and self._call_count > 0)

        import asyncio
        import sys
        current_task = asyncio.current_task()
        task_name = current_task.get_name() if current_task else "no-task"

        # [GLOBAL-INTERCEPTOR] Detect default engine being called with non-default shard!
        is_default_engine = (self._instance_id == "default" or not hasattr(self, '_instance_id'))
        shard_instance_id = getattr(shard, 'instance_id', None)
        is_misrouted = (is_default_engine and shard_instance_id and shard_instance_id != "default")
        
        if is_misrouted:
            print(f"\n{'='*60}")
            print(f"[BLOCKED] Default engine blocked non-default shard!")
            print(f"  Engine: {self._instance_id} (id={id(self)})")
            print(f"  Shard: model={shard.model_id}, instance={shard_instance_id}")
            print(f"  Cache: {list(self._engines.keys())}")
            print(f"  Task: {task_name}")
            print(f"{'='*60}\n")
            return

        print(f"[PyTorchInferenceEngine] [CHECK] ensure_shard entry (instance={self._instance_id}, "
              f"model={shard.model_id}, instance_id={getattr(shard, 'instance_id', '?')}, "
              f"current_cache={list(self._engines.keys())}, suspicious={is_suspicious}, task={task_name})")

        # [KEY-IMPROVEMENT] Lower diagnosis trigger condition
        # Reason: default engine first call has is_suspicious=False, _call_count=0, so Handle interceptor won't trigger!
        should_diagnose = (
            is_suspicious or 
            self._call_count <= 2 or 
            (len(self._engines) == 0 and getattr(shard, 'instance_id', None) != "default")  # New!
        )

        if should_diagnose:
            print(f"[PyTorchInferenceEngine] [STACK] ensure_shard full stack (instance={self._instance_id}, task={task_name}):")
            for line in traceback.format_stack()[-12:-1]:
                print(f"  {line.strip()}")
            
            # [DEEP-FRAME-DIAGNOSIS] Get caller's local variables
            try:
                print(f"[PyTorchInferenceEngine] [FRAME] Caller frame diagnosis (traverse up):")
                
                # 遍历前5个帧，寻找应用层代码
                for frame_depth in range(1, 6):
                    try:
                        frame = sys._getframe(frame_depth)
                        code = frame.f_code
                        locals_dict = frame.f_locals
                        
                        filename = code.co_filename
                        func_name = code.co_name
                        lineno = frame.f_lineno
                        
                        # Determine if it's application code
                        is_app_code = 'exo' in filename and 'asyncio' not in filename
                        
                        marker = "[APP-LAYER]" if is_app_code else "[FRAMEWORK]"
                        
                        print(f"  [{marker}] Frame#{frame_depth}: {filename}")
                        print(f"         Function: {func_name}, Line: {lineno}")
                        
                        if is_app_code:
                            app_vars = ['self', 'shard', 'request_id', 'instance_id', 
                                       'target_engine', 'my_shard', 'result']
                            for var in app_vars:
                                if var in locals_dict:
                                    val = locals_dict[var]
                                    val_type = type(val).__name__
                                    if hasattr(val, 'model_id'):
                                        print(f"         {var} = {val_type}(model_id={getattr(val, 'model_id', '?')})")
                                    elif var == 'self':
                                        print(f"         {var} = {val_type}(id={id(val)})")
                                    else:
                                        print(f"         {var} = {str(val)[:80]}")
                        
                    except ValueError:
                        break
                        
            except Exception as frame_err:
                print(f"[PyTorchInferenceEngine] [WARN] Frame diagnosis failed: {frame_err}")

            # [ULTIMATE-DIAGNOSIS] If it's a pure framework call, intercept Handle object
            if not any('exo' in sys._getframe(d).f_code.co_filename for d in range(1, 6)):
                print(f"[PyTorchInferenceEngine] [!!!] Pure framework call! Intercepting Handle...")
                
                # Method 1: Check current Task's creation stack
                if current_task:
                    try:
                        print(f"[PyTorchInferenceEngine] [INFO] Task info:")
                        print(f"  Task ID: {id(current_task)}")
                        print(f"  Task name: {current_task.get_name()}")
                        print(f"  Task done: {current_task.done()}")
                        
                        # Try to get coroutine object from task
                        coro = current_task.get_coro()
                        if coro:
                            print(f"  Coroutine: {coro}")
                            print(f"  Coroutine qualname: {coro.cr_code.co_qualname if hasattr(coro, 'cr_code') else 'N/A'}")
                            
                    except Exception as task_err:
                        print(f"[PyTorchInferenceEngine] [WARN] Task info retrieval failed: {task_err}")
                
                # Method 2: Traverse all pending Handles in event loop
                loop = asyncio.get_event_loop()
                try:
                    print(f"[PyTorchInferenceEngine] [INFO] Event loop status:")
                    print(f"  Loop running: {loop.is_running()}")
                    
                    # Check Handles in _ready queue
                    if hasattr(loop, '_ready'):
                        ready_handles = list(loop._ready)
                        print(f"  Ready queue: {len(ready_handles)} handles")
                        for i, handle in enumerate(ready_handles[:5]):  # Only show first 5
                            print(f"    [{i}] {handle}")
                            if hasattr(handle, '_callback'):
                                cb = handle._callback
                                print(f"        callback: {cb}")
                                print(f"        callback args: {handle._args if hasattr(handle, '_args') else 'N/A'}")
                                
                except Exception as loop_err:
                    print(f"[PyTorchInferenceEngine] [WARN] Loop status retrieval failed: {loop_err}")

        self._ensure_engine(shard)
        model_id = shard.model_id

        previous_shard = self._loaded_shards.get(model_id)

        def shards_equivalent(old_shard: Shard, new_shard: Shard) -> bool:
            if old_shard is None or new_shard is None:
                return False
            return (old_shard.model_id == new_shard.model_id and
                    old_shard.start_layer == new_shard.start_layer and
                    old_shard.end_layer == new_shard.end_layer and
                    getattr(old_shard, 'instance_id', None) == getattr(new_shard, 'instance_id', None))

        if previous_shard is not None and shards_equivalent(previous_shard, shard):
            existing_engine = self._engines.get(model_id)
            has_model = getattr(existing_engine, 'model', None) is not None
            if has_model:
                logging.debug(f"[PyTorchInferenceEngine] ⏭️ 跳过重复加载: {model_id} (分片未变化)")
                return

        if hasattr(self._engines[model_id], 'ensure_shard'):
            await self._engines[model_id].ensure_shard(shard)

        current_shard = getattr(self._engines[model_id], 'shard', None)
        if current_shard is None:
            current_shard = shard

        if previous_shard is None or not shards_equivalent(previous_shard, current_shard):
            self._loaded_shards[model_id] = current_shard
            print(f"[PyTorchInferenceEngine] 模型 {model_id} 加载/更新了分片: {current_shard}")
            self._notify_model_loaded(current_shard)

    @property
    def tokenizer(self):
        engine = self._get_engine()
        if engine is None:
            return None
        return engine.tokenizer

    def get_tokenizer(self, model_id: str = None):
        """获取指定模型的 tokenizer"""
        if model_id is not None and model_id in self._engines:
            return self._engines[model_id].tokenizer
        engine = self._get_engine()
        if engine is None:
            return None
        return engine.tokenizer

    async def get_embedding(self, token_tensor, shard: Shard = None):
        """将 token 转换为嵌入向量 - 委托给具体引擎"""
        self._ensure_engine(shard)
        model_id = shard.model_id if shard else None
        engine = self._engines.get(model_id) if model_id else self._get_engine()
        if engine and hasattr(engine, 'get_embedding'):
            return await engine.get_embedding(token_tensor, shard)
        return None

    def get_processor(self, model_id: str = None):
        """获取指定模型的 processor"""
        if model_id is not None and model_id in self._engines:
            return getattr(self._engines[model_id], 'processor', None)
        engine = self._get_engine()
        if engine is None:
            return None
        return getattr(engine, 'processor', None)

    def set_processor_and_tokenizer(self, model_id: str, processor, tokenizer=None):
        """设置指定模型的 processor 和 tokenizer"""
        if model_id in self._engines:
            self._engines[model_id].processor = processor
            if tokenizer is not None:
                self._engines[model_id].tokenizer = tokenizer
            elif hasattr(processor, 'tokenizer'):
                self._engines[model_id].tokenizer = processor.tokenizer

    @property
    def processor(self):
        engine = self._get_engine()
        if engine is None:
            return None
        return getattr(engine, 'processor', None)

    @property
    def model(self):
        engine = self._get_engine()
        if engine is None:
            return None
        return getattr(engine, 'model', None)

    @property
    def shard(self):
        engine = self._get_engine()
        if engine is None:
            return None
        return getattr(engine, 'shard', None)

    async def unload_model(self, model_id: str) -> bool:
        """卸载指定模型，释放内存"""
        import gc
        import torch

        if model_id not in self._engines:
            return False

        engine = self._engines[model_id]

        # 清理模型
        if hasattr(engine, 'model') and engine.model is not None:
            del engine.model

        # 清理 tokenizer
        if hasattr(engine, 'tokenizer') and engine.tokenizer is not None:
            del engine.tokenizer

        # 清理 processor
        if hasattr(engine, 'processor') and engine.processor is not None:
            del engine.processor

        # 清理 shard
        if hasattr(engine, 'shard') and engine.shard is not None:
            del engine.shard

        # 删除引擎实例
        del self._engines[model_id]

        # 如果卸载的是当前模型，重置状态
        if self._current_model_id == model_id:
            self._current_model_id = None
            self._default_engine = None

        # 强制垃圾回收
        gc.collect()

        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()

        print(f"[PyTorchInferenceEngine] 模型 {model_id} 已卸载，剩余模型: {list(self._engines.keys())}")
        return True
