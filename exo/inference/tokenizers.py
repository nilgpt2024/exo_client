import traceback
from os import PathLike
from aiofiles import os as aios
from typing import Union, Optional
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
from exo.helpers import DEBUG
from exo.download.new_shard_download import ensure_downloads_dir


class DummyTokenizer:
  def __init__(self, model_type: str = "default"):
    if "qwen" in str(model_type).lower():
      self.eos_token_id = 151643
      self.vocab_size = 151936
      self.model_max_length = 32768
    else:
      self.eos_token_id = 69
      self.vocab_size = 1000
      self.model_max_length = 2048
    self.model_type = model_type

  def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
    if "qwen" in str(self.model_type).lower():
      parts = []
      for m in conversation:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
          content = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
      if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
      out = "".join(parts)
      if tokenize:
        return [1] * max(1, len(out) // 4)
      return out
    return "dummy_tokenized_prompt" if not tokenize else [1]

  def encode(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
    if isinstance(text, (list, tuple)):
      arr = np.array([[1] * max(1, len(str(x)) // 4) for x in text])
    else:
      arr = np.array([1] * max(1, len(str(text)) // 4))
    if return_tensors == "pt":
      try:
        import torch
        return torch.from_numpy(arr.reshape(1, -1)) if arr.ndim == 1 else torch.from_numpy(arr)
      except Exception:
        return arr.reshape(1, -1) if arr.ndim == 1 else arr
    return arr

  def decode(self, tokens, skip_special_tokens=True, **kwargs):
    try:
      import numpy as _np
      if isinstance(tokens, _np.ndarray):
        tokens = tokens.tolist()
    except Exception:
      pass
    if isinstance(tokens, (list, tuple)) and len(tokens) and not isinstance(tokens[0], int):
      try:
        tokens = [int(x) for x in tokens]
      except Exception:
        tokens = []
    n = len(tokens) if hasattr(tokens, '__len__') else 1
    return "dummy output " * max(1, n // 8)


async def resolve_tokenizer(repo_id: Union[str, PathLike], inference_engine_classname: Optional[str] = None):
  if repo_id is None:
    return DummyTokenizer()
  repo_id_str = str(repo_id)
  if not repo_id_str.strip():
    return DummyTokenizer()
  if repo_id_str == "dummy":
    return DummyTokenizer()

  # 检查repo_id是否为None或无效
  if repo_id is None or not str(repo_id).strip():
    if DEBUG >= 2: print(f"Invalid repo_id: {repo_id}, using direct resolution")
    try:
      return await _resolve_tokenizer(repo_id, engine_clsname=inference_engine_classname)
    except Exception:
      return DummyTokenizer()

  try:
    local_path = await ensure_downloads_dir()/str(repo_id).replace("/", "--")
    if DEBUG >= 2: print(f"Checking if local path exists to load tokenizer from local {local_path=}")
    if local_path and await aios.path.exists(local_path):
      if DEBUG >= 2: print(f"Resolving tokenizer for {repo_id=} from {local_path=}")
      try:
        return await _resolve_tokenizer(local_path, engine_clsname=inference_engine_classname)
      except Exception:
        pass
  except Exception as e:
    if DEBUG >= 5: print(f"Local check for {repo_id=} failed with error: {e}")
    if DEBUG >= 5: traceback.print_exc()

  # 如果本地检查失败，直接尝试使用repo_id
  if DEBUG >= 2: print(f"Resolving tokenizer for {repo_id=} directly")
  try:
    return await _resolve_tokenizer(repo_id, engine_clsname=inference_engine_classname)
  except Exception as e:
    if inference_engine_classname == "DummyInferenceEngine" or str(repo_id).startswith("dummy"):
      s = str(repo_id).lower()
      mt = "default"
      if "qwen" in s: mt = "qwen"
      elif "llama" in s: mt = "llama"
      elif "fara" in s: mt = "qwen"
      if DEBUG >= 1: print(f"[tokenizers] resolve failed ({e}), fallback DummyTokenizer({mt})")
      return DummyTokenizer(model_type=mt)
    raise


async def _resolve_tokenizer(repo_id_or_local_path: Union[str, PathLike], engine_clsname: Optional[str] = None):
  try:
    if DEBUG >= 4: print(f"Trying AutoProcessor for {repo_id_or_local_path}")
    processor = AutoProcessor.from_pretrained(repo_id_or_local_path, use_fast=True if "Mistral-Large" in f"{repo_id_or_local_path}" else False, trust_remote_code=True)
    # 安全访问 tokenizer 属性，避免 AttributeError
    _tok = getattr(processor, 'tokenizer', None) or getattr(processor, '_tokenizer', None) or processor
    if not hasattr(processor, 'eos_token_id') and hasattr(_tok, 'eos_token_id'):
      processor.eos_token_id = _tok.eos_token_id
    if not hasattr(processor, 'encode') and hasattr(_tok, 'encode'):
      processor.encode = _tok.encode
    if not hasattr(processor, 'decode') and hasattr(_tok, 'decode'):
      processor.decode = _tok.decode
    # 多模态 Processor 不继承 tokenizer 的 chat_template，手动同步
    if (not getattr(processor, 'chat_template', None)) and getattr(_tok, 'chat_template', None):
      processor.chat_template = _tok.chat_template
    return processor
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load processor for {repo_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {repo_id_or_local_path}")
    return AutoTokenizer.from_pretrained(repo_id_or_local_path, trust_remote_code=True)
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {repo_id_or_local_path}. Falling back to tinygrad tokenizer. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  if engine_clsname == "DummyInferenceEngine":
    s = str(repo_id_or_local_path).lower()
    mt = "default"
    if "qwen" in s: mt = "qwen"
    elif "llama" in s: mt = "llama"
    elif "fara" in s: mt = "qwen"
    return DummyTokenizer(model_type=mt)

  raise ValueError(f"[TODO] Unsupported model: {repo_id_or_local_path}")
