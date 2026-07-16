from exo.inference.shard import Shard
from typing import Optional, List
import requests
import json
import os

# 默认模型配置（本地回退）
DEFAULT_MODEL_CARDS = {
   
}

DEFAULT_PRETTY_NAME = {
   
}


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
