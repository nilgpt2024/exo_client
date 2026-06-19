# -*- coding: utf-8 -*-
"""模型分词器管理模块

该模块提供了分词器(Tokenizer)的加载和解析功能，支持从本地或在线加载Hugging Face格式的分词器，
特别包含了对Qwen3等特定模型的兼容处理逻辑。当无法加载实际分词器时，提供了虚拟分词器实现作为后备方案。
"""
import traceback
from os import PathLike
from aiofiles import os as aios
from typing import Union
import traceback
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
from exo.helpers import DEBUG
from exo.download.new_shard_download import ensure_downloads_dir


class DummyTokenizer:
  """虚拟分词器类
  
  当无法加载实际分词器时提供的后备实现，模拟分词器的基本功能。
  根据不同的模型类型提供不同的默认参数，特别对Qwen模型进行了适配。
  
  Args:
    model_type (str, optional): 模型类型，用于区分不同模型的分词器特性。默认为"default"。
  
  Attributes:
    eos_token_id: 结束标记ID
    vocab_size: 词汇表大小
    model_max_length: 模型支持的最大长度
    pad_token_id: 填充标记ID
    bos_token_id: 开始标记ID
    model_type: 模型类型标识
  """
  def __init__(self, model_type="default"):
    # 根据模型类型设置不同的分词器参数
    self.eos_token_id = 151643 if "qwen" in model_type.lower() else 69  # 结束标记ID
    self.vocab_size = 151936 if "qwen" in model_type.lower() else 1000  # 词汇表大小
    self.model_max_length = 32768 if "qwen" in model_type.lower() else 2048  # 最大序列长度
    self.pad_token_id = None  # 填充标记ID
    self.bos_token_id = 151643 if "qwen" in model_type.lower() else 1  # 开始标记ID
    self.model_type = model_type  # 模型类型标识

  def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
    """应用聊天模板格式化对话内容
    
    将对话内容转换为模型可接受的格式，根据不同模型类型使用不同的模板。
    特别对Qwen模型实现了特定的模板格式模拟。
    
    Args:
      conversation: 对话内容，可以是字符串、字典或列表
      tokenize (bool, optional): 是否对结果进行分词。默认为True
      add_generation_prompt (bool, optional): 是否添加生成提示。默认为True
      tools: 工具定义，此处未使用
      **kwargs: 其他参数
    
    Returns:
      Union[str, np.ndarray]: 格式化后的文本或分词后的token序列
    """
    if "qwen" in self.model_type.lower():
      # 模拟Qwen3的聊天模板格式
      messages = []
      # 处理输入的对话内容，统一转换为消息列表格式
      if isinstance(conversation, list):
        for msg in conversation:
          if isinstance(msg, dict):
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
          else:
            messages.append({"role": "user", "content": str(msg)})
      else:
        messages.append({"role": "user", "content": str(conversation)})
      
      # 如果需要，添加assistant角色的生成提示
      if add_generation_prompt:
        messages.append({"role": "assistant", "content": ""})
      
      # 构建符合Qwen3模型格式的对话模板
      prompt = ""
      for msg in messages:
        if msg["role"] == "system":
          prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
          prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
          prompt += f"<|im_start|>assistant\n{msg['content']}"
      
      # 根据tokenize参数决定返回分词后的结果还是原始文本
      return self.encode(prompt) if tokenize else prompt
    else:
      # 对于非Qwen模型，返回简单的虚拟分词结果
      return "dummy_tokenized_prompt"

  def encode(self, text):
    """将文本编码为token序列
    
    根据不同模型类型，将输入文本转换为对应的token ID序列。
    对Qwen模型实现了特定的编码逻辑模拟。
    
    Args:
      text (str): 要编码的文本
    
    Returns:
      np.ndarray: 编码后的token ID数组
    """
    if "qwen" in self.model_type.lower():
      # 模拟Qwen3的tokenizer行为
      # 简单起见，我们将文本转换为数字序列
      tokens = np.zeros(len(text) // 2 + 2, dtype=np.int64)  # 创建token数组
      tokens[0] = self.bos_token_id  # 添加开始标记
      # 将文本字符转换为token ID，每两个字符生成一个token
      for i in range(1, min(len(tokens)-1, len(text)//2 + 1)):
        tokens[i] = min(self.vocab_size - 1, ord(text[i*2-2]) + ord(text[i*2-1]) % (self.vocab_size - 2))
      tokens[-1] = self.eos_token_id  # 添加结束标记
      return tokens
    else:
      # 对于非Qwen模型，返回简单的token数组
      return np.array([1])

  def decode(self, tokens):
    """将token序列解码为文本
    
    根据不同模型类型，将输入的token ID序列转换为对应的文本。
    对Qwen和Llama-3模型实现了特定的解码逻辑模拟。
    
    Args:
      tokens: 要解码的token ID序列，可以是列表或数组
    
    Returns:
      str: 解码后的文本
    """
    if isinstance(tokens, (list, np.ndarray)):
      if len(tokens) == 0:
        return ""
      result = []
      for t in tokens:
        result.append(self._decode_single_token(int(t)))
      return "".join(result)
    else:
      return self._decode_single_token(int(tokens))
  
  def _decode_single_token(self, token_id: int) -> str:
    """解码单个token ID为文本"""
    if "qwen" in self.model_type.lower():
      if token_id == 0:
        return ""
      elif token_id == 1:
        return ""
      elif token_id == 2:
        return ""
      elif token_id == self.eos_token_id:
        return ""
      elif token_id == self.bos_token_id:
        return ""
      elif 3 <= token_id <= 10:
        return " " if token_id % 2 == 0 else ""
      elif 11 <= token_id <= 100:
        punctuation_map = {
          11: " ", 12: ".", 13: ",", 14: "!", 15: "?", 
          16: ";", 17: ":", 18: "-", 19: "'", 20: '"',
          21: "(", 22: ")", 23: "[", 24: "]", 25: "{", 26: "}"
        }
        return punctuation_map.get(token_id, " ")
      elif 101 <= token_id <= 200:
        return str((token_id - 101) % 10)
      elif 201 <= token_id <= 300:
        return chr(65 + (token_id - 201) % 26)
      elif 301 <= token_id <= 400:
        return chr(97 + (token_id - 301) % 26)
      else:
        char_type = token_id % 4
        if char_type == 0:
          return chr(97 + (token_id % 26))
        elif char_type == 1:
          return chr(65 + (token_id % 26))
        elif char_type == 2:
          return str(token_id % 10)
        else:
          return " "
    elif "llama3" in self.model_type.lower():
      if token_id == 1:
        return ""
      elif token_id == 128001:
        return ""
      elif 90 <= token_id <= 122:
        return chr(token_id)
      elif 32 <= token_id <= 47:
        return chr(token_id)
      elif 48 <= token_id <= 57:
        return chr(token_id)
      elif token_id % 26 == 0:
        return " "
      else:
        return chr(97 + (token_id % 26))
    else:
      return chr(97 + (token_id % 26))

  # 添加缺失的方法以兼容Qwen3 tokenizer的接口
  def __call__(self, text, **kwargs):
    """使分词器可调用，直接调用时执行编码操作
    
    Args:
      text (str): 要编码的文本
      **kwargs: 其他参数，此处未使用
    
    Returns:
      np.ndarray: 编码后的token ID数组
    """
    return self.encode(text)

  def convert_tokens_to_ids(self, tokens):
    """将token转换为对应的ID
    
    Args:
      tokens: 要转换的token，可以是字符串或token列表
    
    Returns:
      Union[np.ndarray, List[int]]: 转换后的token ID序列
    """
    # 如果是字符串则调用encode方法，否则返回相同长度的[1]数组
    return [1] * len(tokens) if not isinstance(tokens, str) else self.encode(tokens)

  def convert_ids_to_tokens(self, ids):
    """将token ID转换为对应的token表示
    
    Args:
      ids: 要转换的token ID，可以是整数或ID列表
    
    Returns:
      Union[str, List[str]]: 转换后的token表示，统一使用<unk>表示未知token
    """
    # 统一返回<unk>表示未知token
    return ["<unk>"] * len(ids) if not isinstance(ids, int) else "<unk>"


# 添加tokenizer缓存字典
_tokenizer_cache = {}

async def resolve_tokenizer(repo_id: Union[str, PathLike]):
  """解析并加载分词器的异步函数
  
  尝试优先从本地路径加载分词器，如果本地路径不存在或加载失败，则通过_resolve_tokenizer函数加载。
  特别处理了"dummy"仓库ID的情况，直接返回虚拟分词器。
  
  Args:
    repo_id (Union[str, PathLike]): 模型仓库ID或路径
  
  Returns:
    加载的分词器对象或虚拟分词器
  """
  # 特殊处理dummy仓库ID，直接返回虚拟分词器
  if repo_id == "dummy":
    return DummyTokenizer()
  
  # 检查缓存中是否已存在该repo_id的tokenizer
  repo_id_str = str(repo_id)
  if repo_id_str in _tokenizer_cache:
    if DEBUG >= 2: print(f"从缓存中获取tokenizer: {repo_id_str}")
    return _tokenizer_cache[repo_id_str]
  
  # 构建本地下载路径
  local_path = await ensure_downloads_dir()/repo_id_str.replace("/", "--")
  
  # 调试信息输出
  if DEBUG >= 2: print(f"Checking if local path exists to load tokenizer from local {local_path=}")
  
  try:
    # 检查本地路径是否存在，如果存在则尝试从本地加载
    if local_path and await aios.path.exists(local_path):
      if DEBUG >= 2: print(f"Resolving tokenizer for {repo_id=} from {local_path=}")
      tokenizer = await _resolve_tokenizer(local_path)
      # 存入缓存
      _tokenizer_cache[repo_id_str] = tokenizer
      return tokenizer
  except:
    # 本地路径检查失败时的错误处理
    if DEBUG >= 5: print(f"Local check for {local_path=} failed. Resolving tokenizer for {repo_id=} normally...")
    if DEBUG >= 5: traceback.print_exc()
  
  # 本地加载失败或路径不存在时，正常解析分词器
  tokenizer = await _resolve_tokenizer(repo_id)
  # 存入缓存
  _tokenizer_cache[repo_id_str] = tokenizer
  return tokenizer

# 添加清除缓存的函数，用于调试或特殊情况
def clear_tokenizer_cache():
  """清除所有缓存的tokenizer"""
  global _tokenizer_cache
  _tokenizer_cache = {}


async def _resolve_tokenizer(repo_id_or_local_path: Union[str, PathLike]):
  """解析并加载分词器的核心异步函数
  
  这是加载分词器的核心函数，实现了多种加载策略：
  1. 优先从本地路径加载
  2. 针对Qwen/Qwen3模型进行特殊处理和兼容性适配
  3. 尝试多种加载方法（AutoProcessor、AutoTokenizer等）
  4. 本地加载失败时尝试在线加载
  5. 所有方法失败时返回虚拟分词器作为后备方案
  
  Args:
    repo_id_or_local_path (Union[str, PathLike]): 模型仓库ID或本地路径
  
  Returns:
    加载的分词器对象或虚拟分词器
  """
  # 修复：处理 Qwen/Qwen3-0.6B 这种格式的模型ID并尝试从本地路径加载
  repo_id_str = str(repo_id_or_local_path)
  original_repo_id = repo_id_or_local_path
  
  # 检查是否是 Qwen/Qwen3-0.6B 格式，如果是则转换为 qwen-3-0.6b 格式
  if 'Qwen/' in repo_id_str:
    print(f"检测到Qwen模型格式: {repo_id_str}")
    # 提取 Qwen3-0.6B 部分并转换为小写，将 Qwen3 改为 qwen-3
    model_part = repo_id_str.split('/')[-1]
    if model_part.startswith('Qwen3'):
      model_part = model_part.replace('Qwen3', 'qwen-3').lower()
      print(f"转换模型ID为: {model_part}")
    # 更新用于日志和检测的值
    repo_id_str = model_part
  
  # 优先尝试从本地下载路径加载模型
  local_model_path = None
  try:
    # 构建本地模型路径：~/.cache/exo/downloads/Qwen--Qwen3-0.6B
    from exo.download.new_shard_download import exo_home
    
    # 处理仓库ID格式
    repo_id_for_path = str(original_repo_id).replace("/", "--")
    local_model_path = exo_home()/"downloads"/repo_id_for_path
    
    # 特殊处理Qwen模型的路径问题
    if 'Qwen' in str(original_repo_id) and local_model_path.exists():
      # 检查是否存在嵌套的模型目录结构
      # 修复：确保original_repo_id转换为字符串后再使用split
      original_repo_str = str(original_repo_id)
      repo_parts = original_repo_str.split('/')
      if len(repo_parts) >= 2:
        nested_model_path = local_model_path / repo_parts[-2] / repo_parts[-1]
        if nested_model_path.exists():
          print(f"发现嵌套的模型目录: {nested_model_path}")
          local_model_path = nested_model_path
  except Exception as e:
    print(f"处理本地路径时出错: {type(e).__name__}: {str(e)}")
  
  # 尝试所有可能的tokenizer加载方法
  # 1. 首先尝试本地加载
  if local_model_path and local_model_path.exists():
    print(f"发现本地模型路径: {local_model_path}")
    
    # 为Qwen模型添加特殊处理
    is_qwen = "Qwen" in repo_id_str or "qwen" in repo_id_str
    
    # 1.1 尝试AutoProcessor
    try:
      print(f"尝试从本地路径加载AutoProcessor，is_qwen={is_qwen}")
      processor = AutoProcessor.from_pretrained(
        str(local_model_path),
        use_fast=not is_qwen if "Mistral-Large" not in repo_id_str else True,
        trust_remote_code=True,
        local_files_only=True  # 强制只使用本地文件
      )
      # 确保processor对象具有必要的属性
      if not hasattr(processor, 'eos_token_id'):
        processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
      if not hasattr(processor, 'encode'):
        processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
      if not hasattr(processor, 'decode'):
        processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
      print(f"成功从本地路径加载AutoProcessor")
      return processor
    except Exception as e:
      print(f"从本地路径加载AutoProcessor失败: {type(e).__name__}: {str(e)}")
      
    # 1.2 尝试AutoTokenizer
    try:
      print(f"尝试从本地路径加载AutoTokenizer，is_qwen={is_qwen}")
      tokenizer = AutoTokenizer.from_pretrained(
        str(local_model_path),
        trust_remote_code=True,
        use_fast=not is_qwen,
        local_files_only=True  # 强制只使用本地文件
      )
      print(f"成功从本地路径加载AutoTokenizer")
      return tokenizer
    except Exception as e:
      print(f"从本地路径加载AutoTokenizer失败: {type(e).__name__}: {str(e)}")
      
    # 1.3 对于Qwen3模型的特殊处理
    if 'qwen-3' in repo_id_str or 'Qwen3' in repo_id_str:
      print(f"尝试Qwen3模型特殊处理路径")
      # 1.3.1 尝试Qwen2Tokenizer
      try:
        from transformers import Qwen2Tokenizer
        print("尝试直接使用Qwen2Tokenizer处理Qwen3模型")
        tokenizer = Qwen2Tokenizer.from_pretrained(
          str(local_model_path),
          trust_remote_code=True,
          local_files_only=True  # 强制只使用本地文件
        )
        print(f"成功从本地路径加载Qwen2Tokenizer处理Qwen3模型")
        return tokenizer
      except Exception as e:
        print(f"Qwen2Tokenizer本地加载失败: {type(e).__name__}: {str(e)}")

      # 1.3.2 尝试QwenTokenizer
      try:
        from transformers import QwenTokenizer
        print("尝试直接使用QwenTokenizer处理Qwen3模型")
        tokenizer = QwenTokenizer.from_pretrained(
          str(local_model_path),
          trust_remote_code=True,
          local_files_only=True  # 强制只使用本地文件
        )
        print(f"成功从本地路径加载QwenTokenizer处理Qwen3模型")
        return tokenizer
      except Exception as e:
        print(f"QwenTokenizer本地加载失败: {type(e).__name__}: {str(e)}")
        
      # 1.3.3 尝试通过修改配置文件绕过架构检查
      try:
        import json
        import os
        import shutil
        config_path = os.path.join(str(local_model_path), "config.json")
        if os.path.exists(config_path):
          print(f"尝试修改配置文件绕过架构检查: {config_path}")
          with open(config_path, 'r') as f:
            config = json.load(f)
          
          # 保存原始配置
          original_architectures = config.get("architectures", [])
          original_model_type = config.get("model_type", "")
          
          # 修改配置以使用Transformers可识别的架构
          if "Qwen3" in str(original_architectures) or "qwen3" == original_model_type:
            config["architectures"] = ["Qwen2ForCausalLM"]
            config["model_type"] = "qwen2"
            
            # 临时保存修改后的配置
            temp_config_path = os.path.join(str(local_model_path), "temp_config.json")
            with open(temp_config_path, 'w') as f:
              json.dump(config, f)
            
            try:
              print("尝试使用修改后的配置文件加载Qwen2Tokenizer")
              tokenizer = Qwen2Tokenizer.from_pretrained(
                str(local_model_path),
                trust_remote_code=True,
                local_files_only=True,
                config=temp_config_path
              )
              print(f"成功使用修改后的配置文件加载tokenizer")
              return tokenizer
            except Exception as e_inner:
              print(f"使用修改后的配置文件加载失败: {type(e_inner).__name__}: {str(e_inner)}")
              # 尝试使用AutoTokenizer替代
              try:
                print("尝试使用修改后的配置文件加载AutoTokenizer")
                tokenizer = AutoTokenizer.from_pretrained(
                  str(local_model_path),
                  trust_remote_code=True,
                  local_files_only=True,
                  config=temp_config_path
                )
                print(f"成功使用修改后的配置文件加载AutoTokenizer")
                return tokenizer
              except Exception as e_inner2:
                print(f"使用修改后的配置文件加载AutoTokenizer失败: {type(e_inner2).__name__}: {str(e_inner2)}")
            finally:
              # 清理临时文件
              if os.path.exists(temp_config_path):
                try:
                  os.remove(temp_config_path)
                except:
                  pass
        
        # 1.3.4 尝试直接使用tokenizer_config.json文件
        tokenizer_config_path = os.path.join(str(local_model_path), "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
          print(f"尝试直接使用tokenizer_config.json: {tokenizer_config_path}")
          try:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(local_model_path))
            print(f"成功使用PreTrainedTokenizerFast加载tokenizer")
            return tokenizer
          except Exception as e_inner:
            print(f"使用PreTrainedTokenizerFast加载失败: {type(e_inner).__name__}: {str(e_inner)}")
        
        # 1.3.5 尝试使用目录中的vocab文件
        if os.path.exists(os.path.join(str(local_model_path), "vocab.json")) and \
           os.path.exists(os.path.join(str(local_model_path), "merges.txt")):
          print(f"发现vocab.json和merges.txt，尝试使用它们加载tokenizer")
          try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(str(local_model_path), local_files_only=True)
            print(f"成功使用GPT2Tokenizer加载tokenizer")
            return tokenizer
          except Exception as e_inner:
            print(f"使用GPT2Tokenizer加载失败: {type(e_inner).__name__}: {str(e_inner)}")
      except Exception as e:
        print(f"修改配置文件尝试失败: {type(e).__name__}: {str(e)}")
  
  # 2. 本地加载失败，尝试在线加载
  # 2.1 尝试AutoProcessor
  try:
    print(f"尝试在线加载AutoProcessor")
    is_qwen = "Qwen" in repo_id_str or "qwen" in repo_id_str
    processor = AutoProcessor.from_pretrained(
      original_repo_id,
      use_fast=not is_qwen if "Mistral-Large" not in repo_id_str else True,
      trust_remote_code=True
    )
    # 确保processor对象具有必要的属性
    if not hasattr(processor, 'eos_token_id'):
      processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
    if not hasattr(processor, 'encode'):
      processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
    if not hasattr(processor, 'decode'):
      processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
    print(f"成功在线加载AutoProcessor")
    return processor
  except Exception as e:
    print(f"在线加载AutoProcessor失败: {type(e).__name__}: {str(e)}")
    
  # 2.2 尝试AutoTokenizer
  try:
    print(f"尝试在线加载AutoTokenizer")
    is_qwen = "Qwen" in repo_id_str or "qwen" in repo_id_str
    tokenizer = AutoTokenizer.from_pretrained(
      original_repo_id,
      trust_remote_code=True,
      use_fast=not is_qwen
    )
    print(f"成功在线加载AutoTokenizer")
    return tokenizer
  except Exception as e:
    print(f"在线加载AutoTokenizer失败: {type(e).__name__}: {str(e)}")
    
    # 2.3 对于Qwen3模型的特殊处理
    if 'qwen-3' in repo_id_str or 'Qwen3' in repo_id_str:
      print(f"尝试Qwen3模型在线特殊处理路径")
      # 2.3.1 尝试Qwen2Tokenizer
      try:
        from transformers import Qwen2Tokenizer
        print("尝试直接使用Qwen2Tokenizer在线处理Qwen3模型")
        tokenizer = Qwen2Tokenizer.from_pretrained(
          original_repo_id,
          trust_remote_code=True
        )
        print(f"成功在线加载Qwen2Tokenizer处理Qwen3模型")
        return tokenizer
      except Exception as e_inner:
        print(f"Qwen2Tokenizer在线加载失败: {type(e_inner).__name__}: {str(e_inner)}")
        
      # 2.3.2 尝试QwenTokenizer
      try:
        from transformers import QwenTokenizer
        print("尝试直接使用QwenTokenizer在线处理Qwen3模型")
        tokenizer = QwenTokenizer.from_pretrained(
          original_repo_id,
          trust_remote_code=True
        )
        print(f"成功在线加载QwenTokenizer处理Qwen3模型")
        return tokenizer
      except Exception as e_inner:
        print(f"QwenTokenizer在线加载失败: {type(e_inner).__name__}: {str(e_inner)}")
        
      # 2.3.3 尝试使用ignore_mismatched_sizes参数
      try:
        print("尝试使用ignore_mismatched_sizes参数加载tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
          original_repo_id,
          trust_remote_code=True,
          ignore_mismatched_sizes=True
        )
        print(f"成功使用ignore_mismatched_sizes参数加载tokenizer")
        return tokenizer
      except Exception as e_inner:
        print(f"使用ignore_mismatched_sizes参数加载失败: {type(e_inner).__name__}: {str(e_inner)}")
        
      # 2.3.4 尝试修改模型ID格式
      try:
        print("尝试修改模型ID格式")
        modified_repo_id = original_repo_id.replace("Qwen3-", "qwen-3-").lower()
        if modified_repo_id != original_repo_id:
          print(f"将模型ID从 {original_repo_id} 修改为 {modified_repo_id}")
          tokenizer = AutoTokenizer.from_pretrained(
            modified_repo_id,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
          )
          print(f"成功使用修改后的模型ID加载tokenizer")
          return tokenizer
      except Exception as e_inner:
        print(f"使用修改后的模型ID加载失败: {type(e_inner).__name__}: {str(e_inner)}")

  # 3. 所有尝试都失败，抛出错误而不是使用错误的后备tokenizer
  # 错误的tokenizer比没有tokenizer更糟糕，应该让错误正常传播
  raise ValueError(f"无法加载模型 {repo_id_or_local_path} 的分词器。所有加载尝试都失败。请确保模型文件完整且分词器配置正确。")