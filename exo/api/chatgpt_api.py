import uuid
import time
import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Literal, Union, Dict, Optional
from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError, ClientError
import aiohttp_cors
import traceback
import signal
from exo import DEBUG, VERSION
from exo.helpers import PrefixDict, shutdown, get_exo_images_dir
from exo.inference.model_tokenizers import resolve_tokenizer
from exo.orchestration import Node
from exo.models import build_base_shard, build_full_shard, model_cards, get_repo, get_supported_models, get_pretty_name
from exo.inference.shard import Shard
from typing import Callable, Optional
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from exo.download.download_progress import RepoProgressEvent
import platform
from collections import defaultdict

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
  import mlx.core as mx
else:
  import numpy as mx


class Message:
  def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]], tools: Optional[List[Dict]] = None):
    self.role = role
    self.content = content
    self.tools = tools

  def to_dict(self):
    data = {"role": self.role, "content": self.content}
    if self.tools:
      data["tools"] = self.tools
    return data


class ChatCompletionRequest:
  def __init__(self, model: str, messages: List[Message], temperature: float, tools: Optional[List[Dict]] = None, enable_thinking: Optional[bool] = None, max_tokens: Optional[int] = None, top_k: Optional[int] = None, top_p: Optional[float] = None):
    self.model = model
    self.messages = messages
    self.temperature = temperature
    self.tools = tools
    # enable_thinking现在通过inference_state统一传递，这里仅保留接口兼容性
    self.enable_thinking = enable_thinking
    self.max_tokens = max_tokens
    self.top_k = top_k
    self.top_p = top_p

  def to_dict(self):
    result = {"model": self.model, "messages": [message.to_dict() for message in self.messages], "temperature": self.temperature, "tools": self.tools}
    # enable_thinking现在通过inference_state传递，不再包含在to_dict中
    return result


def generate_completion(
  chat_request: ChatCompletionRequest,
  tokenizer,
  prompt: str,
  request_id: str,
  tokens: List[int],
  stream: bool,
  finish_reason: Union[Literal["length", "stop"], None],
  object_type: Literal["chat.completion", "text_completion"],
  delta_content: Optional[str] = None,
) -> dict:
  # 确保tokens是适合解码的格式
  try:
    # 如果tokens是列表，确保它包含的是整数
    if isinstance(tokens, list):
      # 展平嵌套列表并转换为整数
      flat_tokens = []
      for token in tokens:
        if isinstance(token, list):
          flat_tokens.extend([int(t) for t in token])
        else:
          flat_tokens.append(int(token))
      tokens = flat_tokens
    
    # 解码tokens为文本
    decoded_content = tokenizer.decode(tokens, skip_special_tokens=True)
    
    # 额外清理可能残留的特殊标记（某些tokenizer的skip_special_tokens不完全）
    import re
    special_token_patterns = [
        r'<\|im_end\|>',
        r'<\|im_start\|>',
        r'<\|endoftext\|>',
        r'<\|end\|>',
    ]
    for pattern in special_token_patterns:
      decoded_content = re.sub(pattern, '', decoded_content)
    decoded_content = decoded_content.rstrip()
  except Exception as e:
    print(f"Token解码错误: {e}, tokens类型: {type(tokens)}, 内容: {tokens[:10] if tokens else 'empty'}")
    # 如果解码失败，使用备用方案
    decoded_content = str(tokens) if tokens else ""
  
  completion = {
    "id": f"chatcmpl-{request_id}",
    "object": object_type,
    "created": int(time.time()),
    "model": chat_request.model,
    "system_fingerprint": f"exo_{VERSION}",
    "choices": [{
      "index": 0,
      "message": {"role": "assistant", "content": decoded_content},
      "logprobs": None,
      "finish_reason": finish_reason,
    }],
  }

  if not stream:
    try:
      prompt_tokens = len(tokenizer.encode(prompt))
      completion_tokens = len(tokens) if tokens else 0
      completion["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
      }
    except Exception as e:
      print(f"计算token使用量错误: {e}")
      completion["usage"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
      }

  choice = completion["choices"][0]
  if object_type.startswith("chat.completion"):
    key_name = "delta" if stream else "message"
    content = delta_content if (stream and delta_content is not None) else decoded_content
    choice[key_name] = {"role": "assistant", "content": content}
  elif object_type == "text_completion":
    choice["text"] = decoded_content
  else:
    ValueError(f"Unsupported response type: {object_type}")

  return completion


def remap_messages(messages: List[Message]) -> List[Message]:
  remapped_messages = []
  last_image = None
  for message in messages:
    if not isinstance(message.content, list):
      remapped_messages.append(message)
      continue

    remapped_content = []
    for content in message.content:
      if isinstance(content, dict):
        if content.get("type") in ["image_url", "image"]:
          image_url = content.get("image_url", {}).get("url") or content.get("image")
          if image_url:
            last_image = {"type": "image", "image": image_url}
            remapped_content.append({"type": "text", "text": "[An image was uploaded but is not displayed here]"})
        else:
          remapped_content.append(content)
      else:
        remapped_content.append(content)
    remapped_messages.append(Message(role=message.role, content=remapped_content))

  if last_image:
    # Replace the last image placeholder with the actual image content
    for message in reversed(remapped_messages):
      for i, content in enumerate(message.content):
        if isinstance(content, dict):
          if content.get("type") == "text" and content.get("text") == "[An image was uploaded but is not displayed here]":
            message.content[i] = last_image
            return remapped_messages

  return remapped_messages


def build_prompt(tokenizer, _messages: List[Message], tools: Optional[List[Dict]] = None):
  messages = remap_messages(_messages)
  chat_template_args = {"conversation": [m.to_dict() for m in messages], "tokenize": False, "add_generation_prompt": True}
  if tools: 
    chat_template_args["tools"] = tools

  try:
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt: {prompt}")
    return prompt
  except UnicodeEncodeError:
    # Handle Unicode encoding by ensuring everything is UTF-8
    chat_template_args["conversation"] = [
      {k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v 
       for k, v in m.to_dict().items()}
      for m in messages
    ]
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt (UTF-8 encoded): {prompt}")
    return prompt


def parse_message(data: dict):
  if "role" not in data or "content" not in data:
    raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
  return Message(data["role"], data["content"], data.get("tools"))


def parse_chat_request(data: dict, default_model: str):
  return ChatCompletionRequest(
    data.get("model", default_model),
    [parse_message(msg) for msg in data["messages"]],
    data.get("temperature", 0.0),
    data.get("tools", None),
    data.get("enable_thinking", None),
    data.get("max_tokens", None),
    data.get("top_k", None),
    data.get("top_p", None),
  )


class PromptSession:
  def __init__(self, request_id: str, timestamp: int, prompt: str):
    self.request_id = request_id
    self.timestamp = timestamp
    self.prompt = prompt


class ChatGPTAPI:
  def __init__(
    self,
    node: Node,
    inference_engine_classname: str,
    response_timeout: int = 90,
    on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None,
    default_model: Optional[str] = None,
    system_prompt: Optional[str] = None
  ):
    self.node = node
    self.inference_engine_classname = inference_engine_classname
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    self.app = web.Application(client_max_size=100*1024*1024)  # 100MB to support image upload
    self.prompts: PrefixDict[str, PromptSession] = PrefixDict()
    self.prev_token_lens: Dict[str, int] = {}
    self.stream_tasks: Dict[str, asyncio.Task] = {}
    self.default_model = default_model or "llama-3.2-1b"
    self.token_queues = defaultdict(asyncio.Queue)
    # 流式输出累积 buffer：解决 byte-level tokenizer 单 token 解码产生 U+FFFD 乱码的问题
    self.stream_token_buffers: Dict[str, List[int]] = {}
    self.stream_text_buffers: Dict[str, str] = {}

    # Get the callback system and register our handler
    self.token_callback = node.on_token.register("chatgpt-api-token-handler")
    # 使用*args来捕获所有传递的参数，确保不会因为参数数量不匹配而导致错误
    self.token_callback.on_next(lambda *args: asyncio.create_task(self.handle_tokens(args[0], args[1], args[2])) if len(args) >= 3 else None)
    self.system_prompt = system_prompt

    cors = aiohttp_cors.setup(self.app)
    cors_options = aiohttp_cors.ResourceOptions(
      allow_credentials=True,
      expose_headers="*",
      allow_headers="*",
      allow_methods="*",
    )
    cors.add(self.app.router.add_get("/models", self.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models", self.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/image/generations", self.handle_post_image_generations), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/audio/generations", self.handle_post_audio_generations), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/audio/clone_prompt", self.handle_post_audio_clone_prompt), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/download/progress", self.handle_get_download_progress), {"*": cors_options})
    cors.add(self.app.router.add_get("/modelpool", self.handle_model_support), {"*": cors_options})
    cors.add(self.app.router.add_get("/healthcheck", self.handle_healthcheck), {"*": cors_options})
    cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})
    cors.add(self.app.router.add_get("/initial_models", self.handle_get_initial_models), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_get("/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/discovery/switch", self.handle_switch_discovery), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/discovery/status", self.handle_get_discovery_status), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/manager/shard-config", self.handle_manager_shard_config), {"*": cors_options})

    # Add static routes
    if "__compiled__" not in globals():
      self.static_dir = Path(__file__).parent.parent/"tinychat"
      self.app.router.add_get("/", self.handle_root)
      if self.static_dir.exists():
        self.app.router.add_static("/", self.static_dir, name="static")
      
    # Always add images route, regardless of compilation status
    self.images_dir = get_exo_images_dir()
    self.images_dir.mkdir(parents=True, exist_ok=True)
    self.app.router.add_static('/images/', self.images_dir, name='static_images')

    self.app.middlewares.append(self.timeout_middleware)
    self.app.middlewares.append(self.log_request)

  async def handle_quit(self, request):
    if DEBUG >= 1: print("Received quit signal")
    response = web.json_response({"detail": "Quit signal received"}, status=200)
    await response.prepare(request)
    await response.write_eof()
    await shutdown(signal.SIGINT, asyncio.get_event_loop(), self.node.server)

  async def timeout_middleware(self, app, handler):
    async def middleware(request):
      try:
        return await asyncio.wait_for(handler(request), timeout=self.response_timeout)
      except asyncio.TimeoutError:
        return web.json_response({"detail": "Request timed out"}, status=408)

    return middleware

  async def log_request(self, app, handler):
    async def middleware(request):
      if DEBUG >= 2: print(f"Received request: {request.method} {request.path}")
      return await handler(request)

    return middleware

  async def handle_root(self, request):
    return web.FileResponse(self.static_dir/"index.html")

  async def handle_healthcheck(self, request):
    return web.json_response({"status": "ok"})

  async def handle_model_support(self, request):
    try:
        response = web.StreamResponse(status=200, reason='OK', headers={ 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' })
        try:
            await response.prepare(request)
        except (ConnectionResetError, BrokenPipeError, RuntimeError, ClientConnectionResetError, ClientError) as conn_error:
            if DEBUG >= 2: print(f"Client disconnected before response: {conn_error}")
            return response

        try:
            # 获取已加载到内存的模型列表
            loaded_models = []
            if hasattr(self.node.inference_engine, 'get_loaded_models'):
                loaded_models = self.node.inference_engine.get_loaded_models()

            async for path, s in self.node.shard_downloader.get_shard_download_status(self.inference_engine_classname):
                # 检查响应是否仍然活跃
                if response.task is None or response.task.done():
                    if DEBUG >= 2: print("Client disconnected, stopping model support stream")
                    break

                # 增加额外的检查逻辑，避免显示错误的下载状态
                actual_downloaded = s.downloaded_bytes == s.total_bytes and s.total_bytes > 0
                # 如果total_bytes为0，则认为未下载
                if s.total_bytes == 0:
                    download_percentage = 0
                    downloaded = False
                else:
                    download_percentage = 100 if actual_downloaded else 100 * float(s.downloaded_bytes) / float(s.total_bytes)
                    downloaded = actual_downloaded

                # 如果模型已加载到内存，也标记为可用
                is_loaded = s.shard.model_id in loaded_models

                model_data = { s.shard.model_id: {
                    "downloaded": downloaded or is_loaded,  # 已下载或已加载都视为可用
                    "download_percentage": 100 if is_loaded else download_percentage,
                    "total_size": s.total_bytes,
                    "total_downloaded": s.total_bytes if is_loaded else s.downloaded_bytes,
                    "loaded_in_memory": is_loaded  # 新增字段：是否已加载到内存
                } }

                # 安全写入，处理连接已关闭的情况
                try:
                    await response.write(f"data: {json.dumps(model_data)}\n\n".encode('utf-8'))
                    await response.drain()
                except (ConnectionResetError, BrokenPipeError, RuntimeError, ClientConnectionResetError, ClientError) as write_error:
                    if DEBUG >= 2: print(f"Client disconnected during write: {write_error}")
                    break

            if response.task is not None and not response.task.done():
                try:
                    await response.write(b"data: [DONE]\n\n")
                except (ConnectionResetError, BrokenPipeError, RuntimeError, ClientConnectionResetError, ClientError) as write_error:
                    if DEBUG >= 2: print(f"Client disconnected during final write: {write_error}")

        except asyncio.CancelledError:
            if DEBUG >= 2: print("Model support stream cancelled")
            # 客户端取消请求是正常情况，不需要记录为错误
            pass
        finally:
            # 确保正确关闭响应
            try:
                await response.write_eof()
            except:
                # 如果连接已关闭，忽略写入EOF的错误
                pass

        return response

    except (ConnectionResetError, BrokenPipeError, RuntimeError, asyncio.CancelledError, ClientConnectionResetError, ClientError) as conn_error:
        if DEBUG >= 2: print(f"Connection error in handle_model_support: {conn_error}")
        return web.json_response({"detail": "Connection closed"}, status=499)
    except Exception as e:
        print(f"Error in handle_model_support: {str(e)}")
        traceback.print_exc()
        return web.json_response({"detail": f"Server error: {str(e)}"}, status=500)
        
  async def handle_get_models(self, request):
    models_list = [{"id": model_name, "object": "model", "owned_by": "exo", "ready": True} for model_name, _ in model_cards.items()]
    return web.json_response({"object": "list", "data": models_list})

  async def handle_post_chat_token_encode(self, request):
    data = await request.json()
    model = data.get("model", self.default_model)
    if model and model.startswith("gpt-"):
      model = self.default_model
    original_model = model
    base_model = model.split("::")[0] if "::" in model else model

    # [FIX] 支持用 HF repo ID 格式请求模型
    if base_model not in model_cards:
      resolved = None
      for key, info in model_cards.items():
        for engine_repo in info.get("repo", {}).values():
          if engine_repo == base_model:
            resolved = key
            break
        if resolved:
          break
      if resolved:
        if DEBUG >= 1: print(f"[ChatGPTAPI] 模型名解析: HF repo ID '{base_model}' → 短名 '{resolved}'")
        base_model = resolved
        model = resolved

    if not model or base_model not in model_cards:
      if DEBUG >= 1: print(f"Invalid model: {model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
      model = self.default_model
      base_model = model
    shard = build_base_shard(base_model, self.inference_engine_classname)
    if shard and "::" in original_model:
      instance_id = original_model.split("::")[1] if "::" in original_model else "default"
      shard = Shard(
        base_model,
        shard.start_layer,
        shard.end_layer,
        shard.n_layers,
        repo_id=shard.repo_id,
        tie_word_embeddings=shard.tie_word_embeddings,
        instance_id=instance_id
      )
      if DEBUG >= 1: print(f"[ChatGPTAPI] 多实例推理: model={original_model}, base={base_model}, instance={instance_id}")
    messages = [parse_message(msg) for msg in data.get("messages", [])]
    tokenizer = await resolve_tokenizer(get_repo(base_model, self.inference_engine_classname))
    if tokenizer is None:
      return web.json_response({"detail": f"Failed to load tokenizer for model {model}"}, status=500)
    prompt = build_prompt(tokenizer, messages, data.get("tools", None))
    tokens = tokenizer.encode(prompt)
    return web.json_response({
      "length": len(prompt),
      "num_tokens": len(tokens),
      "encoded_tokens": tokens,
      "encoded_prompt": prompt,
    })

  async def handle_get_download_progress(self, request):
    progress_data = {}
    for node_id, progress_event in self.node.node_download_progress.items():
      if isinstance(progress_event, RepoProgressEvent):
        if progress_event.status != "in_progress": continue
        progress_data[node_id] = progress_event.to_dict()
      else:
        print(f"Unknown progress event type: {type(progress_event)}. {progress_event}")
    return web.json_response(progress_data)

  async def handle_post_chat_completions(self, request):
    data = await request.json()
    if DEBUG >= 2: print(f"[ChatGPTAPI] Handling chat completions request from {request.remote}: {data}")
    stream = data.get("stream", False)
    chat_request = parse_chat_request(data, self.default_model)
    if chat_request.model and chat_request.model.startswith("gpt-"):
      chat_request.model = self.default_model
    base_model = chat_request.model.split("::")[0] if "::" in chat_request.model else chat_request.model
    original_model = chat_request.model

    # [FIX] 支持用 HF repo ID 格式请求模型（如 "Qwen/Qwen3-4B"）
    if base_model not in model_cards:
      resolved = None
      for key, info in model_cards.items():
        for engine_repo in info.get("repo", {}).values():
          if engine_repo == base_model:
            resolved = key
            break
        if resolved:
          break
      if resolved:
        if DEBUG >= 1: print(f"[ChatGPTAPI] 模型名解析: HF repo ID '{base_model}' → 短名 '{resolved}'")
        base_model = resolved
        chat_request.model = resolved

    if not chat_request.model or base_model not in model_cards:
      if DEBUG >= 1: print(f"[ChatGPTAPI] Invalid model: {chat_request.model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
      chat_request.model = self.default_model
      base_model = chat_request.model
    shard = build_base_shard(base_model, self.inference_engine_classname)
    if shard and "::" in original_model:
      instance_id = original_model.split("::")[1] if "::" in original_model else "default"
      shard = Shard(
        base_model, shard.start_layer, shard.end_layer, shard.n_layers,
        repo_id=shard.repo_id, tie_word_embeddings=shard.tie_word_embeddings,
        instance_id=instance_id
      )
      if DEBUG >= 1: print(f"[ChatGPTAPI] 多实例推理: model={original_model}, base={base_model}, instance={instance_id}")
    elif shard and not ("::" in original_model):
      # 请求使用 base model_id（如 qwen-3-0.6b），自动解析为节点的 full_model_id（如 qwen-3-0.6b::worker-1）
      if hasattr(self.node, 'my_loaded_models'):
        for loaded_id in self.node.my_loaded_models:
          loaded_base = loaded_id.split("::")[0] if "::" in loaded_id else loaded_id
          if loaded_base == base_model:
            chat_request.model = loaded_id
            # 同时更新为节点的实际分片，避免因 shard 不匹配导致重复加载
            actual_shard = self.node.my_loaded_models[loaded_id].shard
            if actual_shard:
              shard = Shard(
                base_model, actual_shard.start_layer, actual_shard.end_layer,
                actual_shard.n_layers, repo_id=actual_shard.repo_id,
                tie_word_embeddings=actual_shard.tie_word_embeddings,
                instance_id=actual_shard.instance_id
              )
            if DEBUG >= 1: print(f"[ChatGPTAPI] 模型ID解析: '{original_model}' → '{loaded_id}' (base→full), shard={shard}")
            break
    if not shard:
      supported_models = [model for model, info in model_cards.items() if self.inference_engine_classname in info.get("repo", {})]
      return web.json_response(
        {"detail": f"Unsupported model: {chat_request.model} with inference engine {self.inference_engine_classname}. Supported models for this engine: {supported_models}"},
        status=400,
      )

    tokenizer = await resolve_tokenizer(get_repo(base_model, self.inference_engine_classname))
    if tokenizer is None:
      return web.json_response({"detail": f"Failed to load tokenizer for model {chat_request.model}"}, status=500)

    # Add system prompt if set
    if self.system_prompt and not any(msg.role == "system" for msg in chat_request.messages):
      chat_request.messages.insert(0, Message("system", self.system_prompt))

    prompt = build_prompt(tokenizer, chat_request.messages, chat_request.tools)
    request_id = str(uuid.uuid4())
    if self.on_chat_completion_request:
      try:
        self.on_chat_completion_request(request_id, chat_request, prompt)
      except Exception as e:
        if DEBUG >= 2: traceback.print_exc()

    if DEBUG >= 2: print(f"[ChatGPTAPI] Processing prompt: {request_id=} {shard=} {prompt=}")

    try:
      # 创建inference_state字典，并添加temperature和enable_thinking参数
      inference_state = {}
      if hasattr(chat_request, 'temperature'):
          inference_state['temperature'] = chat_request.temperature
      if hasattr(chat_request, 'enable_thinking') and chat_request.enable_thinking is not None:
          inference_state['enable_thinking'] = chat_request.enable_thinking
      else:
          inference_state['enable_thinking'] = False
      if hasattr(chat_request, 'max_tokens') and chat_request.max_tokens is not None:
          inference_state['max_tokens'] = chat_request.max_tokens
      if hasattr(chat_request, 'top_k') and chat_request.top_k is not None:
          inference_state['top_k'] = chat_request.top_k
      if hasattr(chat_request, 'top_p') and chat_request.top_p is not None:
          inference_state['top_p'] = chat_request.top_p

      # 提取并读取图片内容
      image = self._extract_image_from_messages(chat_request.messages)
      if image is not None:
          inference_state['image'] = image
          # 传递完整的 messages 列表（含图片位置信息），用于多轮对话+图片的正确处理
          inference_state['messages'] = [m.to_dict() for m in chat_request.messages]
          if DEBUG >= 2: print(f"[ChatGPTAPI] Image extracted and messages list added to inference_state")

      if stream:
        # 流式模式：启动 process_prompt 作为后台任务，不等待完成
        # 立即开始从 token_queues 读取 token 并流式返回
        print(f"[ChatGPTAPI] Starting process_prompt as background task for streaming")
        process_task = asyncio.create_task(self.node.process_prompt(shard, prompt, request_id=request_id, inference_state=inference_state))
      else:
        # 非流式模式：等待 process_prompt 完成
        try:
            result = await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(shard, prompt, request_id=request_id, inference_state=inference_state))), timeout=self.response_timeout)
        except Exception as e:
            raise
        print(f"[ChatGPTAPI] process_prompt completed, waiting for response. timeout={self.response_timeout}s")

      if stream:
        response = web.StreamResponse(
          status=200,
          reason="OK",
          headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
          },
        )
        await response.prepare(request)

        try:
          # Stream tokens while waiting for inference to complete
          while True:
            print(f"[ChatGPTAPI] Waiting for token from queue: {request_id=}")
            tokens, is_finished = await asyncio.wait_for(
              self.token_queues[request_id].get(),
              timeout=self.response_timeout
            )
            print(f"[ChatGPTAPI] Got token from queue: {request_id=} {tokens=} {is_finished=}")

            eos_token_id = None
            if not eos_token_id and hasattr(tokenizer, "eos_token_id"): eos_token_id = tokenizer.eos_token_id
            if not eos_token_id and hasattr(tokenizer, "_tokenizer"): eos_token_id = tokenizer.special_tokens_map.get("eos_token_id")

            finish_reason = None
            if is_finished:
                if tokens and tokens[-1] == eos_token_id:
                    finish_reason = "stop"
                else:
                    finish_reason = "length"
            print(f"{eos_token_id=} {tokens[-1] if tokens else None} {finish_reason=}")

            # 累积流式 token，避免 byte-level tokenizer 在 token 边界处产生 U+FFFD 乱码
            if request_id not in self.stream_token_buffers:
                self.stream_token_buffers[request_id] = []
                self.stream_text_buffers[request_id] = ""
            self.stream_token_buffers[request_id].extend(tokens)
            accumulated_tokens = self.stream_token_buffers[request_id]

            full_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=True)
            special_token_patterns = [
                r'<\|im_end\|>',
                r'<\|im_start\|>',
                r'<\|endoftext\|>',
                r'<\|end\|>',
            ]
            for pattern in special_token_patterns:
                full_text = re.sub(pattern, '', full_text)
            full_text = full_text.rstrip()

            prev_text = self.stream_text_buffers[request_id]
            # 延迟发送策略：如果新增文本里包含 U+FFFD（不完整的 UTF-8 字节序列），
            # 则只发送 U+FFFD 之前的部分，剩余部分等后续 token 补齐后再一起发送。
            # 请求结束时（finish_reason 不为 None）强制发送剩余内容。
            new_part = full_text[len(prev_text):]
            if "\ufffd" in new_part and finish_reason is None:
                ufffd_pos = new_part.find("\ufffd")
                delta_text = new_part[:ufffd_pos]
            else:
                delta_text = new_part
            self.stream_text_buffers[request_id] = prev_text + delta_text

            completion = generate_completion(
              chat_request,
              tokenizer,
              prompt,
              request_id,
              accumulated_tokens,
              stream,
              finish_reason,
              "chat.completion",
              delta_content=delta_text,
            )

            # 安全写入，处理连接已关闭的情况
            try:
              await response.write(f"data: {json.dumps(completion)}\n\n".encode('utf-8'))
              # 立即刷新确保数据发送
              await response.drain()
            except (ConnectionResetError, BrokenPipeError, RuntimeError) as write_error:
              print(f"Client disconnected during chat completion write: {write_error}")
              break

            if is_finished:
              break

          # 只有在连接仍然活跃时才发送EOF
          try:
            await response.write_eof()
          except (ConnectionResetError, BrokenPipeError, RuntimeError) as eof_error:
            print(f"Client disconnected during EOF: {eof_error}")
            # 客户端已断开，忽略EOF错误
          return response

        except asyncio.TimeoutError:
          print(f"[ChatGPTAPI] Timeout waiting for token: {request_id=}")
          return web.json_response({"detail": "Response generation timed out"}, status=408)

        except asyncio.CancelledError:
          print(f"[ChatGPTAPI] Request cancelled: {request_id=}")
          # 请求被取消，清理资源并返回
          raise

        except Exception as e:
          print(f"[ChatGPTAPI] Error processing prompt: {e}")
          traceback.print_exc()
          return web.json_response(
            {"detail": f"Error processing prompt: {str(e)}"},
            status=500
          )

        finally:
          # Clean up the queue for this request
          if request_id in self.token_queues:
            print(f"[ChatGPTAPI] Cleaning up token queue: {request_id=}")
            del self.token_queues[request_id]
          # 清理流式累积 buffer
          if request_id in self.stream_token_buffers:
            del self.stream_token_buffers[request_id]
          if request_id in self.stream_text_buffers:
            del self.stream_text_buffers[request_id]
          # 等待后台任务完成（如果还在运行）
          if 'process_task' in dir() and not process_task.done():
            try:
              await asyncio.wait_for(process_task, timeout=5.0)
            except asyncio.TimeoutError:
              print(f"[ChatGPTAPI] Background process_task timeout, cancelling")
              process_task.cancel()
      else:
        tokens = []
        while True:
          _tokens, is_finished = await asyncio.wait_for(self.token_queues[request_id].get(), timeout=self.response_timeout)
          tokens.extend(_tokens)
          if is_finished:
            break
        finish_reason = "length"
        eos_token_id = None
        if not eos_token_id and hasattr(tokenizer, "eos_token_id"): eos_token_id = tokenizer.eos_token_id
        if not eos_token_id and hasattr(tokenizer, "_tokenizer"): eos_token_id = tokenizer.special_tokens_map.get("eos_token_id")
        print(f"Checking if end of tokens result {tokens[-1]=} is {eos_token_id=}")
        if tokens[-1] == eos_token_id:
          finish_reason = "stop"

        return web.json_response(generate_completion(chat_request, tokenizer, prompt, request_id, tokens, stream, finish_reason, "chat.completion"))
    except asyncio.TimeoutError:
      return web.json_response({"detail": "Response generation timed out"}, status=408)
    except Exception as e:
      print(f"[ChatGPTAPI] Error processing prompt: {e}")
      traceback.print_exc()
      return web.json_response({"detail": f"Error processing prompt: {str(e)}"}, status=500)

  async def handle_post_image_generations(self, request):
    data = await request.json()

    if DEBUG >= 2: print(f"Handling chat completions request from {request.remote}: {data}")
    stream = data.get("stream", False)
    model = data.get("model", "")
    prompt = data.get("prompt", "")
    image_url = data.get("image_url", "")
    if DEBUG >= 2: print(f"model: {model}, prompt: {prompt}, stream: {stream}")
    shard = build_base_shard(model, self.inference_engine_classname)
    if DEBUG >= 2: print(f"shard: {shard}")
    if not shard:
      return web.json_response({"error": f"Unsupported model: {model} with inference engine {self.inference_engine_classname}"}, status=400)

    request_id = str(uuid.uuid4())
    callback_id = f"chatgpt-api-wait-response-{request_id}"
    callback = self.node.on_token.register(callback_id)
    try:
      if image_url != "" and image_url != None:
        img = self.base64_decode(image_url)
      else:
        img = None
      await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(shard, prompt, request_id=request_id, inference_state={"image": img}))), timeout=self.response_timeout)

      response = web.StreamResponse(status=200, reason='OK', headers={
        'Content-Type': 'application/octet-stream',
        "Cache-Control": "no-cache",
      })
      await response.prepare(request)

      def get_progress_bar(current_step, total_steps, bar_length=50):
        # Calculate the percentage of completion
        percent = float(current_step)/total_steps
        # Calculate the number of hashes to display
        arrow = '-'*int(round(percent*bar_length) - 1) + '>'
        spaces = ' '*(bar_length - len(arrow))

        # Create the progress bar string
        progress_bar = f'Progress: [{arrow}{spaces}] {int(percent * 100)}% ({current_step}/{total_steps})'
        return progress_bar

      async def stream_image(_request_id: str, result, is_finished: bool):
        # 检查响应是否仍然活跃
        if response.task is None or response.task.done():
          if DEBUG >= 2: print("Client disconnected, stopping image stream")
          return
          
        try:
          if isinstance(result, list):
            await response.write(json.dumps({'progress': get_progress_bar((result[0]), (result[1]))}).encode('utf-8') + b'\n')

          elif isinstance(result, np.ndarray):
            try:
              im = Image.fromarray(np.array(result))
              # Save the image to a file
              image_filename = f"{_request_id}.png"
              image_path = self.images_dir/image_filename
              im.save(image_path)
              
              # Get URL for the saved image
              try:
                image_url = request.app.router['static_images'].url_for(filename=image_filename)
                base_url = f"{request.scheme}://{request.host}"
                full_image_url = base_url + str(image_url)
                
                await response.write(json.dumps({'images': [{'url': str(full_image_url), 'content_type': 'image/png'}]}).encode('utf-8') + b'\n')
              except KeyError as e:
                if DEBUG >= 2: print(f"Error getting image URL: {e}")
                # Fallback to direct file path if URL generation fails
                await response.write(json.dumps({'images': [{'url': str(image_path), 'content_type': 'image/png'}]}).encode('utf-8') + b'\n')
              
              if is_finished:
                try:
                  await response.write_eof()
                except (ConnectionResetError, BrokenPipeError, RuntimeError) as eof_error:
                  if DEBUG >= 2: print(f"Client disconnected during image EOF: {eof_error}")
              
            except Exception as e:
              if DEBUG >= 2: print(f"Error processing image: {e}")
              if DEBUG >= 2: traceback.print_exc()
              await response.write(json.dumps({'error': str(e)}).encode('utf-8') + b'\n')
              
        except (ConnectionResetError, BrokenPipeError, RuntimeError) as write_error:
          if DEBUG >= 2: print(f"Client disconnected during image write: {write_error}")
          # 客户端已断开，停止流

      stream_task = None

      def on_result(_request_id: str, result, is_finished: bool):
        nonlocal stream_task
        stream_task = asyncio.create_task(stream_image(_request_id, result, is_finished))
        return _request_id == request_id and is_finished

      await callback.wait(on_result, timeout=self.response_timeout*10)

      if stream_task:
        # Wait for the stream task to complete before returning
        await stream_task

      return response

    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"}, status=500)

  async def handle_post_audio_generations(self, request):
    data = await request.json()
    if DEBUG >= 2: print(f"Handling audio generations request from {request.remote}: {data}")

    model = data.get("model", "")
    text = data.get("text", data.get("prompt", ""))

    # 默认 mode 根据模型 ID 推断
    tts_mode = data.get("mode")
    if not tts_mode:
      if model.endswith("-custom"):
        tts_mode = "custom_voice"
      elif model.endswith("-base"):
        tts_mode = "voice_clone"
      else:
        tts_mode = "voice_design"

    language = data.get("language", "Chinese")
    instruct = data.get("instruct", None)
    speaker = data.get("speaker", None)
    ref_audio = data.get("ref_audio", None)
    ref_text = data.get("ref_text", None)
    x_vector_only_mode = data.get("x_vector_only_mode", False)
    voice_clone_prompt = data.get("voice_clone_prompt", None)

    # 生成参数
    max_new_tokens = data.get("max_new_tokens", None)
    do_sample = data.get("do_sample", None)
    top_k = data.get("top_k", None)
    top_p = data.get("top_p", None)
    temperature = data.get("temperature", None)
    repetition_penalty = data.get("repetition_penalty", None)
    subtalker_dosample = data.get("subtalker_dosample", None)
    subtalker_top_k = data.get("subtalker_top_k", None)
    subtalker_top_p = data.get("subtalker_top_p", None)
    subtalker_temperature = data.get("subtalker_temperature", None)
    non_streaming_mode = data.get("non_streaming_mode", None)

    shard = build_base_shard(model, self.inference_engine_classname)
    if not shard:
      return web.json_response({"detail": f"Unsupported model: {model}"}, status=400)

    # 参数校验
    if tts_mode == "custom_voice" and speaker is None:
      return web.json_response({"detail": "custom_voice mode requires 'speaker'"}, status=400)
    if tts_mode == "voice_clone" and ref_audio is None and voice_clone_prompt is None:
      return web.json_response({"detail": "voice_clone mode requires 'ref_audio' or 'voice_clone_prompt'"}, status=400)

    request_id = str(uuid.uuid4())
    callback_id = f"chatgpt-api-wait-response-{request_id}"
    callback = self.node.on_token.register(callback_id)

    inference_state = {
      "tts_mode": tts_mode,
      "language": language,
    }
    if instruct is not None:
      inference_state["instruct"] = instruct
    if speaker is not None:
      inference_state["speaker"] = speaker
    if ref_audio is not None:
      inference_state["ref_audio"] = ref_audio
    if ref_text is not None:
      inference_state["ref_text"] = ref_text
    if x_vector_only_mode is not None:
      inference_state["x_vector_only_mode"] = x_vector_only_mode
    if voice_clone_prompt is not None:
      inference_state["voice_clone_prompt"] = voice_clone_prompt

    for key in ["max_new_tokens", "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
                "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature",
                "non_streaming_mode"]:
      value = data.get(key)
      if value is not None:
        inference_state[key] = value

    # 支持批量文本：传入 list 时通过 texts 字段保留原始列表
    is_batch = isinstance(text, list)
    if is_batch:
      inference_state["texts"] = text
      prompt = json.dumps(text)
    else:
      prompt = text

    try:
      await asyncio.wait_for(
        asyncio.shield(asyncio.create_task(
          self.node.process_prompt(shard, prompt, request_id=request_id, inference_state=inference_state)
        )),
        timeout=self.response_timeout
      )

      audio_result = None
      sample_rate = 24000
      is_batch_result = False

      def on_result(_request_id: str, result, is_finished: bool):
        nonlocal audio_result, sample_rate, is_batch_result
        if _request_id == request_id:
          if isinstance(result, np.ndarray):
            audio_result = result
          if isinstance(result, dict):
            if "audio" in result:
              audio_result = result["audio"]
            if "sample_rate" in result:
              sample_rate = result["sample_rate"]
            if result.get("is_batch"):
              is_batch_result = True
          return is_finished
        return False

      await callback.wait(on_result, timeout=self.response_timeout * 10)

      if audio_result is None:
        return web.json_response({"detail": "No audio generated"}, status=500)

      import io
      import soundfile as sf

      # 批量返回 JSON（base64 wav 列表）
      if is_batch_result and isinstance(audio_result, list):
        encoded_audios = []
        for wav in audio_result:
          buffer = io.BytesIO()
          sf.write(buffer, wav, sample_rate, format='WAV')
          buffer.seek(0)
          encoded_audios.append(base64.b64encode(buffer.read()).decode('utf-8'))

        return web.json_response({
          "sample_rate": sample_rate,
          "audio": encoded_audios,
          "count": len(encoded_audios),
        })

      # 单条返回 audio/wav
      buffer = io.BytesIO()
      sf.write(buffer, audio_result, sample_rate, format='WAV')
      buffer.seek(0)

      return web.Response(
        body=buffer.read(),
        content_type='audio/wav',
        headers={
          "Content-Disposition": f"attachment; filename=tts_{request_id}.wav",
          "Cache-Control": "no-cache",
        }
      )

    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error processing TTS: {str(e)}"}, status=500)

  async def handle_post_audio_clone_prompt(self, request):
    """
    两阶段 VoiceClone 第一阶段：根据参考音频创建 voice_clone_prompt
    """
    data = await request.json()
    if DEBUG >= 2: print(f"Handling audio clone prompt request from {request.remote}: {data}")

    model = data.get("model", "")
    ref_audio = data.get("ref_audio", None)
    ref_text = data.get("ref_text", None)
    x_vector_only_mode = data.get("x_vector_only_mode", False)

    if ref_audio is None:
      return web.json_response({"detail": "ref_audio is required"}, status=400)

    shard = build_base_shard(model, self.inference_engine_classname)
    if not shard:
      return web.json_response({"detail": f"Unsupported model: {model}"}, status=400)

    try:
      # 确保引擎已加载
      await self.node.inference_engine.ensure_shard(shard)

      # 检查引擎是否支持 create_voice_clone_prompt
      engine = self.node.inference_engine
      if hasattr(engine, 'create_voice_clone_prompt'):
        prompt_data = await engine.create_voice_clone_prompt(shard, ref_audio, ref_text, x_vector_only_mode)
      elif hasattr(engine, 'inference_engine') and hasattr(engine.inference_engine, 'create_voice_clone_prompt'):
        prompt_data = await engine.inference_engine.create_voice_clone_prompt(shard, ref_audio, ref_text, x_vector_only_mode)
      else:
        return web.json_response({"detail": "Current inference engine does not support voice clone prompt"}, status=400)

      return web.json_response(prompt_data)

    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error creating voice clone prompt: {str(e)}"}, status=500)

  async def handle_get_initial_models(self, request):
    model_data = {}
    for model_id in get_supported_models([[self.inference_engine_classname]]):
      model_data[model_id] = {
        "name": get_pretty_name(model_id),
        "downloaded": None,  # Initially unknown
        "download_percentage": None,  # Change from 0 to null
        "total_size": None,
        "total_downloaded": None,
        "loading": True  # Add loading state
      }
    return web.json_response(model_data)

  def _get_gpu_memory_info(self):
    """获取实时的 GPU 内存信息"""
    try:
      import pynvml
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      return {
        "total": info.total // 2**20,
        "free": info.free // 2**20,
        "used": info.used // 2**20
      }
    except Exception as e:
      if DEBUG >= 2: print(f"[API] Error getting GPU memory info: {e}")
      return None

  async def handle_get_topology(self, request):
    try:
      topology = self.node.current_topology
      if topology:
        if DEBUG >= 2: print(f"[API] Getting topology with {len(topology.nodes)} nodes")
        result = topology.to_json()
        # 添加节点性能统计
        result["node_stats"] = self.node.node_stats
        
        # 从推理引擎直接获取当前加载的分片信息（最准确的来源）
        engine_loaded_shards = {}
        if hasattr(self.node.inference_engine, '_loaded_shards'):
          engine_loaded_shards = self.node.inference_engine._loaded_shards
        
        # 添加详细的模型加载状态（包含每个模型加载的层信息）- 这是最完整的信息
        result["node_loaded_models"] = {}
        
        # 自己的加载状态（优先使用 node.my_loaded_models，如果没有则从引擎补充）
        if hasattr(self.node, 'my_loaded_models'):
          result["node_loaded_models"][self.node.id] = {
            model_id: load_state.to_dict()
            for model_id, load_state in self.node.my_loaded_models.items()
          }
          
          # 从引擎补充未在 my_loaded_models 中的模型
          for model_id, shard in engine_loaded_shards.items():
            if model_id not in result["node_loaded_models"][self.node.id]:
              from exo.inference.shard import ModelLoadState
              load_state = ModelLoadState(model_id=model_id, shard=shard)
              result["node_loaded_models"][self.node.id][model_id] = load_state.to_dict()
              print(f"[API] 补充从引擎获取的模型加载状态: {model_id}")
        
        # 其他节点的加载状态
        if hasattr(self.node, 'node_loaded_models'):
          for node_id, models in self.node.node_loaded_models.items():
            if node_id not in result["node_loaded_models"]:
              result["node_loaded_models"][node_id] = {
                model_id: load_state.to_dict()
                for model_id, load_state in models.items()
              }
        
        # 从 node_loaded_models 生成向后兼容的 node_shards 和 node_shards_multi
        result["node_shards"] = {}
        result["node_shards_multi"] = {}
        
        for node_id, models in result["node_loaded_models"].items():
          shards_list = []
          for model_id, load_state in models.items():
            shards_list.append(load_state["shard"])
          
          result["node_shards_multi"][node_id] = shards_list
          if shards_list:
            result["node_shards"][node_id] = shards_list[0]
        
        # 实时更新 GPU 内存信息 - 只更新当前节点的内存信息
        memory_info = self._get_gpu_memory_info()
        if memory_info and self.node.id in result.get("nodes", {}):
          result["nodes"][self.node.id]["memory_detail"] = memory_info
          result["nodes"][self.node.id]["memory"] = memory_info["total"]
        
        if DEBUG >= 2: print(f"[API] Topology result: {result}")
        return web.json_response(result)
      else:
        if DEBUG >= 2: print("[API] No topology available, returning empty object")
        return web.json_response({})
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      print(f"❌ [API] Error getting topology: {type(e).__name__}: {e}")
      print(f"   Topology object: {self.node.current_topology}")
      print(f"   Topology type: {type(self.node.current_topology)}")
      if hasattr(self.node.current_topology, 'nodes'):
        print(f"   Nodes: {list(self.node.current_topology.nodes.keys()) if self.node.current_topology.nodes else 'empty'}")
      return web.json_response({"detail": f"Error getting topology: {str(e)}"}, status=500)

  async def handle_manager_shard_config(self, request):
    """处理 Manager 通过 HTTP 发送的分片配置（FRP/HTTP Fallback 模式）"""
    try:
      data = await request.json()
      
      if DEBUG >= 1:
        print(f"[API] [Manager] 收到 HTTP 分片配置命令: {data.get('model_id')}")
      
      cmd_model_id = data.get("model_id")
      cmd_model_path = data.get("model_path")
      cmd_shard = data.get("shard", {})
      cmd_start_layer = cmd_shard.get("start_layer", 0)
      cmd_end_layer = cmd_shard.get("end_layer", 0)
      cmd_n_layers = cmd_shard.get("n_layers", 0)
      peer_list = data.get("peer_list", [])
      
      if peer_list:
        asyncio.create_task(self.node._register_peers_from_manager(peer_list))
      
      async def _delayed_load():
        await asyncio.sleep(1.0)
        await self.node._handle_manager_load(
          cmd_model_id, cmd_model_path, 
          cmd_start_layer, cmd_end_layer, cmd_n_layers
        )
      asyncio.create_task(_delayed_load())
      
      return web.json_response({
        "success": True,
        "message": f"分片配置已接收，正在加载: {cmd_model_id}",
        "node_id": self.node.id,
        "shard": {
          "start_layer": cmd_start_layer,
          "end_layer": cmd_end_layer,
          "n_layers": cmd_n_layers
        }
      })
      
    except Exception as e:
      if DEBUG >= 1: traceback.print_exc()
      print(f"[ERROR] [API] [Manager] 处理分片配置失败: {e}")
      return web.json_response({
        "success": False,
        "error": str(e)
      }, status=500)

  async def handle_get_discovery_status(self, request):
    """获取当前发现模块的状态"""
    try:
      discovery = self.node.discovery
      discovery_type = type(discovery).__name__
      
      result = {
        "discovery_type": discovery_type,
        "node_id": self.node.id,
      }
      
      if discovery_type == "FRPDiscovery":
        result.update({
          "frp_server_addr": getattr(discovery, 'frp_server_addr', None),
          "frp_server_port": getattr(discovery, 'frp_server_port', None),
          "frp_remote_port": getattr(discovery, 'my_remote_port', None),
          "my_address": getattr(discovery, 'my_address', None),
          "enable_p2p": getattr(discovery, 'enable_p2p', False),
        })
      elif discovery_type == "UDPDiscovery":
        result.update({
          "node_port": getattr(discovery, 'node_port', None),
          "listen_port": getattr(discovery, 'listen_port', None),
          "broadcast_port": getattr(discovery, 'broadcast_port', None),
        })
      elif discovery_type == "TailscaleDiscovery":
        result.update({
          "node_port": getattr(discovery, 'node_port', None),
        })
      elif discovery_type == "ManualDiscovery":
        result.update({
          "config_path": getattr(discovery, 'config_path', None),
        })
      
      return web.json_response(result)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error getting discovery status: {str(e)}"}, status=500)

  async def handle_switch_discovery(self, request):
    """切换发现模块"""
    try:
      data = await request.json()
      discovery_type = data.get("discovery_type", "").lower()
      
      if discovery_type not in ["udp", "tailscale", "manual", "frp"]:
        return web.json_response({"detail": "Invalid discovery type. Must be one of: udp, tailscale, manual, frp"}, status=400)
      
      new_discovery = None
      
      if discovery_type == "frp":
        frp_server_addr = data.get("frp_server_addr")
        if not frp_server_addr:
          return web.json_response({"detail": "frp_server_addr is required for FRP discovery"}, status=400)
        
        from exo.networking.frp.frp_discovery import FRPDiscovery
        from exo.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
        
        new_discovery = FRPDiscovery(
          frp_server_addr=frp_server_addr,
          frp_server_port=data.get("frp_server_port", 7000),
          node_id=self.node.id,
          local_port=self.node.server.port if self.node.server else 5678,
          create_peer_handle=lambda peer_id, address, description, device_capabilities: self._create_grpc_peer_handle(peer_id, address, description, device_capabilities),
          frp_token=data.get("frp_token"),
          frp_remote_port=data.get("frp_remote_port"),
          seed_peers=data.get("seed_peers"),
          discovery_timeout=data.get("discovery_timeout", 30),
          device_capabilities=UNKNOWN_DEVICE_CAPABILITIES,
          enable_p2p=data.get("enable_p2p", False)
        )
      
      elif discovery_type == "udp":
        from exo.networking.udp.udp_discovery import UDPDiscovery
        
        new_discovery = UDPDiscovery(
          node_id=self.node.id,
          node_port=self.node.server.port if self.node.server else 5678,
          listen_port=data.get("listen_port", 5678),
          broadcast_port=data.get("broadcast_port", 5678),
          create_peer_handle=lambda peer_id, address, description, device_capabilities: self._create_grpc_peer_handle(peer_id, address, description, device_capabilities),
          discovery_timeout=data.get("discovery_timeout", 30),
          allowed_node_ids=data.get("allowed_node_ids"),
          allowed_interface_types=data.get("allowed_interface_types")
        )
      
      elif discovery_type == "tailscale":
        from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
        
        new_discovery = TailscaleDiscovery(
          node_id=self.node.id,
          node_port=self.node.server.port if self.node.server else 5678,
          create_peer_handle=lambda peer_id, address, description, device_capabilities: self._create_grpc_peer_handle(peer_id, address, description, device_capabilities),
          discovery_timeout=data.get("discovery_timeout", 30),
          tailscale_api_key=data.get("tailscale_api_key"),
          tailnet=data.get("tailnet_name"),
          allowed_node_ids=data.get("allowed_node_ids")
        )
      
      elif discovery_type == "manual":
        from exo.networking.manual.manual_discovery import ManualDiscovery
        
        new_discovery = ManualDiscovery(
          config_path=data.get("config_path"),
          node_id=self.node.id,
          create_peer_handle=lambda peer_id, address, description, device_capabilities: self._create_grpc_peer_handle(peer_id, address, description, device_capabilities)
        )
      
      if new_discovery:
        await self.node.switch_discovery(new_discovery)
        return web.json_response({
          "success": True,
          "message": f"Switched to {discovery_type} discovery",
          "discovery_type": discovery_type
        })
      else:
        return web.json_response({"detail": "Failed to create discovery instance"}, status=500)
        
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error switching discovery: {str(e)}"}, status=500)

  def _create_grpc_peer_handle(self, peer_id, address, description, device_capabilities):
    """创建 GRPC PeerHandle"""
    from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
    return GRPCPeerHandle(peer_id, address, description, device_capabilities)

  async def handle_tokens(self, request_id: str, tokens: List[int], is_finished: bool):
    await self.token_queues[request_id].put((tokens if tokens is not None else [], is_finished))

  async def run(self, host: str = "0.0.0.0", port: int = 52415):
    runner = web.AppRunner(self.app,access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

  def _extract_image_from_messages(self, messages: List[Message]):
    """从消息中提取图片内容

    Args:
        messages: 消息列表

    Returns:
        PIL Image 对象，如果没有图片则返回 None
    """
    from PIL import Image
    from io import BytesIO
    import base64

    for message in messages:
      if not isinstance(message.content, list):
        continue

      for content in message.content:
        if isinstance(content, dict) and content.get("type") in ["image_url", "image"]:
          image_url = content.get("image_url", {}).get("url") or content.get("image")
          if not image_url:
            continue

          try:
            # 处理 base64 编码的图片
            if image_url.startswith('data:image'):
              # 解码 base64 图片为 PIL Image
              if image_url.startswith('data:image'):
                image_url = image_url.split(',')[1]
              image_data = base64.b64decode(image_url)
              img = Image.open(BytesIO(image_data))
              return img
            elif image_url.startswith('http://') or image_url.startswith('https://'):
              # 从 URL 下载图片
              import requests
              response = requests.get(image_url, timeout=30)
              response.raise_for_status()
              img = Image.open(BytesIO(response.content))
              return img
            else:
              # 本地文件路径
              img = Image.open(image_url)
              return img
          except Exception as e:
            if DEBUG >= 2: print(f"[ChatGPTAPI] Error loading image: {e}")
            continue

    return None

  def _extract_text_from_messages(self, messages: List[Message]):
    """从消息中提取文本内容

    Args:
        messages: 消息列表

    Returns:
        文本内容，如果没有文本则返回空字符串
    """
    for message in messages:
      if not isinstance(message.content, list):
        if isinstance(message.content, str):
          return message.content
        continue

      for content in message.content:
        if isinstance(content, dict) and content.get("type") == "text":
          return content.get("text", "")

    return ""

  def base64_decode(self, base64_string):
    #decode and reshape image
    if base64_string.startswith('data:image'):
      base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_data))
    W, H = (dim - dim%64 for dim in (img.width, img.height))
    if W != img.width or H != img.height:
      if DEBUG >= 2: print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
      img = img.resize((W, H), Image.NEAREST)  # use desired downsampling filter
    img = mx.array(np.array(img))
    img = (img[:, :, :3].astype(mx.float32)/255)*2 - 1
    img = img[None]
    return img
