from exo.inference.shard import Shard
from typing import Optional, List
import requests
import json
import os

# 默认模型配置（本地回退）
DEFAULT_MODEL_CARDS = {
  "qwen-3-0.6b": {
    "layers": 28,
    "repo": {
       "PyTorchQwen3InferenceEngine": "Qwen/Qwen3-0.6B"
    },
  },
  "qwen-3-4b": {
    "layers": 36,
    "repo": {
       "PyTorchQwen3InferenceEngine": "Qwen/Qwen3-4B"
    },
  },
  "qwen-3-vl-2b": {
    "layers": 28,
    "repo": {
       "PyTorchQwen3VLInferenceEngine": "Qwen/Qwen3-VL-2B-Instruct"
    },
  },
  "qwen-3-vl-4b": {
    "layers": 36,
    "repo": {
       "PyTorchQwen3VLInferenceEngine": "Qwen/Qwen3-VL-4B-Instruct"
    },
  },
  "qwen-3-tts-1.7b": {
    "layers": 36,
    "repo": {
       "PyTorchQwen3TTSInferenceEngine": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    },
  },
  "fara-7b": {
    "layers": 28,
    "repo": {
       "PyTorchQwen2_5VlInferenceEngine": "microsoft/Fara-7B-INT8"
    },
  },
  "qwen-2.5-vl-3b": {
    "layers": 36,
    "repo": {
       "PyTorchQwen2_5VlInferenceEngine": "Qwen/Qwen2.5-VL-3B-Instruct"
    },
  },
  "llama-3.2-1b": {
    "layers": 16,
    "repo": {
      "PyTorchLlama3InferenceEngine": "unsloth/Llama-3.2-1B-Instruct"
    },
  },
  "dummy": {
    "layers": 8,
    "repo": {
      "DummyInferenceEngine": "dummy",
    },
  }, 
}

DEFAULT_PRETTY_NAME = {
  "llama-3.3-70b": "Llama 3.3 70B",
  "llama-3.2-1b": "Llama 3.2 1B",
  "llama-3.2-1b-8bit": "Llama 3.2 1B (8-bit)",
  "llama-3.2-3b": "Llama 3.2 3B",
  "llama-3.2-3b-8bit": "Llama 3.2 3B (8-bit)",
  "llama-3.2-3b-bf16": "Llama 3.2 3B (BF16)",
  "llama-3.1-8b": "Llama 3.1 8B",
  "llama-3.1-70b": "Llama 3.1 70B",
  "llama-3.1-70b-bf16": "Llama 3.1 70B (BF16)",
  "llama-3.1-405b": "Llama 3.1 405B",
  "llama-3.1-405b-8bit": "Llama 3.1 405B (8-bit)",
  "gemma2-9b": "Gemma2 9B",
  "gemma2-27b": "Gemma2 27B",
  "nemotron-70b": "Nemotron 70B",
  "nemotron-70b-bf16": "Nemotron 70B (BF16)",
  "mistral-nemo": "Mistral Nemo",
  "mistral-large": "Mistral Large",
  "deepseek-coder-v2-lite": "Deepseek Coder V2 Lite",
  "deepseek-coder-v2.5": "Deepseek Coder V2.5",
  "deepseek-v3": "Deepseek V3 (4-bit)",
  "deepseek-v3-3bit": "Deepseek V3 (3-bit)",
  "deepseek-r1": "Deepseek R1 (4-bit)",
  "deepseek-r1-3bit": "Deepseek R1 (3-bit)",
  "llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
  "qwen-2.5-vl-3b": "Qwen 2.5 VL 3B",
  "qwen-3-0.6b": "Qwen 3 0.6B",
  "qwen-3-4b": "Qwen 3 4B",
  "qwen-3-vl-2b": "Qwen 3 VL 2B",
  "qwen-3-vl-4b": "Qwen 3 VL 4B",
  "qwen-3-tts-1.7b": "Qwen 3 TTS 1.7B (VoiceDesign)",
  "qwen-3-tts-1.7b-custom": "Qwen 3 TTS 1.7B (CustomVoice)",
  "fara-7b": "Fara 7B (Microsoft Computer Use Agent)",
  "qwen-2.5-0.5b": "Qwen 2.5 0.5B",
  "qwen-2.5-1.5b": "Qwen 2.5 1.5B",
  "qwen-2.5-coder-1.5b": "Qwen 2.5 Coder 1.5B",
  "qwen-2.5-3b": "Qwen 2.5 3B",
  "qwen-2.5-coder-3b": "Qwen 2.5 Coder 3B",
  "qwen-2.5-7b": "Qwen 2.5 7B",
  "qwen-2.5-coder-7b": "Qwen 2.5 Coder 7B",
  "qwen-2.5-math-7b": "Qwen 2.5 7B (Math)",
  "qwen-2.5-14b": "Qwen 2.5 14B",
  "qwen-2.5-coder-14b": "Qwen 2.5 Coder 14B",
  "qwen-2.5-32b": "Qwen 2.5 32B",
  "qwen-2.5-coder-32b": "Qwen 2.5 Coder 32B",
  "qwen-2.5-72b": "Qwen 2.5 72B",
  "qwen-2.5-math-72b": "Qwen 2.5 72B (Math)",
  "phi-3.5-mini": "Phi-3.5 Mini",
  "phi-4": "Phi-4",
  "llama-3-8b": "Llama 3 8B",
  "llama-3-70b": "Llama 3 70B",
  "stable-diffusion-2-1-base": "Stable Diffusion 2.1",
  "deepseek-r1-distill-qwen-1.5b": "DeepSeek R1 Distill Qwen 1.5B",
  "deepseek-r1-distill-qwen-1.5b-3bit": "DeepSeek R1 Distill Qwen 1.5B (3-bit)",
  "deepseek-r1-distill-qwen-1.5b-6bit": "DeepSeek R1 Distill Qwen 1.5B (6-bit)",
  "deepseek-r1-distill-qwen-1.5b-8bit": "DeepSeek R1 Distill Qwen 1.5B (8-bit)",
  "deepseek-r1-distill-qwen-1.5b-bf16": "DeepSeek R1 Distill Qwen 1.5B (BF16)",
  "deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill Qwen 7B",
  "deepseek-r1-distill-qwen-7b-3bit": "DeepSeek R1 Distill Qwen 7B (3-bit)",
  "deepseek-r1-distill-qwen-7b-6bit": "DeepSeek R1 Distill Qwen 7B (6-bit)",
  "deepseek-r1-distill-qwen-7b-8bit": "DeepSeek R1 Distill Qwen 7B (8-bit)",
  "deepseek-r1-distill-qwen-7b-bf16": "DeepSeek R1 Distill Qwen 7B (BF16)",
  "deepseek-r1-distill-qwen-14b": "DeepSeek R1 Distill Qwen 14B",
  "deepseek-r1-distill-qwen-14b-3bit": "DeepSeek R1 Distill Qwen 14B (3-bit)",
  "deepseek-r1-distill-qwen-14b-6bit": "DeepSeek R1 Distill Qwen 14B (6-bit)",
  "deepseek-r1-distill-qwen-14b-8bit": "DeepSeek R1 Distill Qwen 14B (8-bit)",
  "deepseek-r1-distill-qwen-14b-bf16": "DeepSeek R1 Distill Qwen 14B (BF16)",
  "deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B",
  "deepseek-r1-distill-qwen-32b-3bit": "DeepSeek R1 Distill Qwen 32B (3-bit)",
  "deepseek-r1-distill-qwen-32b-8bit": "DeepSeek R1 Distill Qwen 32B (8-bit)",
  "deepseek-r1-distill-qwen-32b-bf16": "DeepSeek R1 Distill Qwen 32B (BF16)",
  "deepseek-r1-distill-llama-8b-8bit": "DeepSeek R1 Distill Llama 8B (8-bit)",
  "deepseek-r1-distill-llama-8b": "DeepSeek R1 Distill Llama 8B",
  "deepseek-r1-distill-llama-8b-3bit": "DeepSeek R1 Distill Llama 8B (3-bit)",
  "deepseek-r1-distill-llama-8b-6bit": "DeepSeek R1 Distill Llama 8B (6-bit)",
  "deepseek-r1-distill-llama-8b-bf16": "DeepSeek R1 Distill Llama 8B (BF16)",
  "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
  "deepseek-r1-distill-llama-70b-3bit": "DeepSeek R1 Distill Llama 70B (3-bit)",
  "deepseek-r1-distill-llama-70b-6bit": "DeepSeek R1 Distill Llama 70B (6-bit)",
  "deepseek-r1-distill-llama-70b-8bit": "DeepSeek R1 Distill Llama 70B (8-bit)",
  "deepseek-r1-distill-qwen-32b-6bit": "DeepSeek R1 Distill Qwen 32B (6-bit)",
}

# 远程模型配置 URL（回退用）
REMOTE_MODELS_URL = os.environ.get("EXO_MODELS_URL", "https://token.sygis.com/models.json")

# Manager URL（运行时设置）
_manager_url = None

# 全局变量（从远程加载）
model_cards = {}
pretty_name = {}

def set_manager_url(url: Optional[str]):
    """设置 Manager URL，用于后续加载模型配置"""
    global _manager_url
    _manager_url = url

def _parse_manager_response(data: dict):
    """解析 Manager /api/models/available 响应为本地格式"""
    cards = {}
    names = {}
    
    response_data = data.get("data", data)
    models_list = response_data.get("models", [])
    
    for m in models_list:
        model_id = m.get("model_id")
        if not model_id:
            continue
        
        layers = m.get("layers", 0)
        repo_name = m.get("repo", "")
        engines = m.get("engines", [])
        
        if repo_name and engines:
            repo_config = {}
            for engine in engines:
                if 'PyTorch' in engine or 'Dummy' in engine or 'MLX' in engine:
                    repo_config[engine] = repo_name
            
            if repo_config:
                cards[model_id] = {"layers": layers, "repo": repo_config}
        
        pretty = m.get("pretty_name")
        if pretty and pretty != model_id:
            names[model_id] = pretty
    
    return cards, names

def _load_from_manager():
    """从 Manager API 加载模型配置"""
    global _manager_url
    
    if not _manager_url:
        return None, None
    
    try:
        url = f"{_manager_url.rstrip('/')}/api/models/available"
        print(f"[Models] 正在从 Manager 获取模型配置: {url}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            cards, names = _parse_manager_response(data)
            
            print(f"[Models] 从 Manager 获取到 {len(cards)} 个模型配置")
            return cards, names
        else:
            print(f"[Models] Manager 配置获取失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"[Models] Manager 连接失败: {e}")
    
    return None, None

def _load_remote_models():
    """加载模型配置：优先从 Manager 获取，回退到远程 URL"""
    global model_cards, pretty_name
    
    # 先使用默认配置
    model_cards = DEFAULT_MODEL_CARDS.copy()
    pretty_name = DEFAULT_PRETTY_NAME.copy()
    
    # 策略1: 尝试从 Manager 获取
    manager_cards, manager_names = _load_from_manager()
    if manager_cards:
        model_cards.update(manager_cards)
        if manager_names:
            pretty_name.update(manager_names)
        print(f"[Models] 从 Manager 加载成功，总模型数: {len(model_cards)}")
        return
     
def init_models(manager_url: Optional[str] = None):
    """初始化模型配置（可在运行时调用以重新加载）"""
    if manager_url:
        set_manager_url(manager_url)
    _load_remote_models()

# 模块加载时自动获取配置（使用默认策略）
_load_remote_models()

def _get_pytorch_engines(repo_config: dict) -> list:
  """从 repo_config 中提取所有 PyTorch 引擎配置
  
  动态检测以 'PyTorch' 开头且以 'InferenceEngine' 结尾的引擎
  """
  return [k for k in repo_config.keys() if k.startswith("PyTorch") and k.endswith("InferenceEngine")]

def get_repo(model_id: str, inference_engine_classname: str) -> Optional[str]:
  model_info = model_cards.get(model_id, {})
  repo_config = model_info.get("repo", {})
  
  # 如果直接找到了配置的引擎，直接返回
  if inference_engine_classname in repo_config:
    return repo_config[inference_engine_classname]  
  # 如果是 PyTorchInferenceEngine（统一入口），返回第一个可用的 PyTorch 引擎配置
  if inference_engine_classname == "PyTorchInferenceEngine":
    pytorch_engines = _get_pytorch_engines(repo_config)
    if pytorch_engines:
      return repo_config[pytorch_engines[0]]
  
  return None

def get_pretty_name(model_id: str) -> Optional[str]:
  return pretty_name.get(model_id, None)

def build_base_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  repo = get_repo(model_id, inference_engine_classname)
  n_layers = model_cards.get(model_id, {}).get("layers", 0)
  if repo is None or n_layers < 1:
    return None
  # 返回完整的模型分片（0 到 n_layers-1），让 node.py 根据拓扑决定实际加载哪个分片
  return Shard(model_id, 0, n_layers - 1, n_layers)

def build_full_shard(model_id: str, inference_engine_classname: str) -> Optional[Shard]:
  base_shard = build_base_shard(model_id, inference_engine_classname)
  if base_shard is None: return None
  return Shard(base_shard.model_id, 0, base_shard.n_layers - 1, base_shard.n_layers)

def get_supported_models(supported_inference_engine_lists: Optional[List[List[str]]] = None) -> List[str]:
  if not supported_inference_engine_lists:
    return list(model_cards.keys())

  from exo.inference.inference_engine import inference_engine_classes
  supported_inference_engine_lists = [
    [inference_engine_classes[engine] if engine in inference_engine_classes else engine for engine in engine_list]
    for engine_list in supported_inference_engine_lists
  ]

  def has_any_engine(model_info: dict, engine_list: List[str]) -> bool:
    repo_config = model_info.get("repo", {})
    for engine in engine_list:
      # 直接匹配配置的引擎
      if engine in repo_config:
        return True
      # 如果是 PyTorchInferenceEngine，动态检查是否有任何 PyTorch 引擎配置
      if engine == "PyTorchInferenceEngine":
        pytorch_engines = _get_pytorch_engines(repo_config)
        if pytorch_engines:
          return True
    return False

  def supports_all_engine_lists(model_info: dict) -> bool:
    return all(has_any_engine(model_info, engine_list)
              for engine_list in supported_inference_engine_lists)

  return [
    model_id for model_id, model_info in model_cards.items()
    if supports_all_engine_lists(model_info)
  ]
