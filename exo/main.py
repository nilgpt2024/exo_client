# 导入标准库模块，用于命令行参数解析
import argparse
import sys
import os

# 确保项目根目录在 Python 搜索路径中，以便能正确导入 exo 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入异步编程相关模块
import asyncio
# 导入程序退出时执行清理操作的模块
import atexit
# 导入信号处理相关模块
import signal
# 导入JSON数据处理模块
import json
# 导入获取系统平台信息的模块
import platform
# 导入操作系统相关功能模块
import os
# 导入时间处理模块
import time
# 导入异常堆栈跟踪模块
import traceback
# 导入生成唯一标识符的模块
import uuid
# 导入科学计算库
import numpy as np
# 导入进度条显示模块
from tqdm import tqdm
# 导入日志模块
import logging
from pathlib import Path
from datetime import datetime

# 配置日志输出到文件和控制台
def setup_logging():
    """配置日志输出到文件和控制台"""
    # 获取日志文件路径 - 项目根目录下的 logs 文件夹
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用时间戳生成唯一的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"exo_{timestamp}.log"
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            # 文件处理器 - 输出到日志文件
            logging.FileHandler(log_file, encoding='utf-8', mode='a'),
            # 控制台处理器 - 输出到终端
            logging.StreamHandler()
        ]
    )
    
    print(f"日志文件: {log_file}")
    return log_file

# 初始化日志配置
log_file = setup_logging()

# 首先尝试解决protobuf版本不兼容问题
# 在导入任何可能使用transformers的模块之前先处理protobuf
print("正在尝试应用protobuf版本兼容补丁...")

# 添加protobuf版本补丁
# 这个补丁尝试解决transformers库中加载tensorflow proto时的版本不兼容问题
def monkey_patch_protobuf_for_tf():
    """猴子补丁修复protobuf与tensorflow proto的版本兼容性问题"""
    try:
        # 导入google.protobuf内部模块
        import google.protobuf.descriptor_pb2
        import google.protobuf.internal.decoder
        import google.protobuf.internal.encoder
        
        # 安全地检查_DecodeField属性是否存在
        if hasattr(google.protobuf.internal.decoder, '_DecodeField'):
            # 保存原始的解析函数
            original_decode_field = google.protobuf.internal.decoder._DecodeField
            
            # 创建一个替代函数，忽略版本不兼容警告
            def patched_decode_field(buffer, pos, end, message, field_dict, extension_dict, unknown_field_handlers):
                try:
                    return original_decode_field(buffer, pos, end, message, field_dict, extension_dict, unknown_field_handlers)
                except Exception as e:
                    # 检查是否是版本不兼容错误
                    if "incompatible Protobuf Gencode/Runtime versions" in str(e):
                        # 忽略这个错误，尝试继续执行
                        print(f"忽略protobuf版本不兼容错误: {e}")
                        return pos  # 返回当前位置，尝试继续解析
                    # 其他错误保持原样抛出
                    raise
            
            # 替换函数
            google.protobuf.internal.decoder._DecodeField = patched_decode_field
            print("成功应用protobuf版本兼容补丁")
        else:
            print("protobuf版本中没有_DecodeField属性，跳过补丁")
    except ImportError:
        print("无法找到需要修补的protobuf模块，可能是环境不同")
    except Exception as e:
        print(f"应用protobuf补丁时出错: {e}")

# 应用protobuf补丁
monkey_patch_protobuf_for_tf()
# 从自定义模块中导入数据集加载和批处理相关函数
from exo.train.dataset import load_dataset, iterate_batches
# 从自定义模块中导入手动发现相关类
from exo.networking.manual.manual_discovery import ManualDiscovery
# 从自定义模块中导入节点类
from exo.orchestration.node import Node
# 从自定义模块中导入GRPC服务器类
from exo.networking.grpc.grpc_server import GRPCServer
# 从自定义模块中导入UDP发现相关类
from exo.networking.udp.udp_discovery import UDPDiscovery
# 从自定义模块中导入Tailscale发现相关类
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
# 从自定义模块中导入FRP发现相关类
from exo.networking.frp.frp_discovery import FRPDiscovery
# 从自定义模块中导入GRPC对等节点处理类
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
# 从自定义模块中导入环形内存加权分区策略类
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
# 从自定义模块中导入ChatGPT API类
from exo.api import ChatGPTAPI
# 从自定义模块中导入分片下载相关类
from exo.download.shard_download import ShardDownloader, NoopShardDownloader
# 从自定义模块中导入仓库进度事件类
from exo.download.download_progress import RepoProgressEvent
# 从自定义模块中导入新的分片下载器相关函数和工具函数
from exo.download.new_shard_download import new_shard_downloader, has_exo_home_read_access, has_exo_home_write_access, ensure_exo_home, seed_models
# 从自定义模块中导入辅助函数
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses_and_interfaces, terminal_link, shutdown
# 从自定义模块中导入分片类
from exo.inference.shard import Shard
# 从自定义模块中导入推理引擎获取函数
from exo.inference.inference_engine import get_inference_engine
# 从自定义模块中导入模型分词器解析函数
from exo.inference.model_tokenizers import resolve_tokenizer
# 从自定义模块中导入构建基础分片和获取仓库的函数
from exo.models import build_base_shard, get_repo
# 从自定义模块中导入拓扑可视化类
from exo.viz.topology_viz import TopologyViz
# 尝试导入uvloop模块，用于提升异步性能
try:
    import uvloop
except:
    uvloop = None
# 导入线程池执行器模块
import concurrent.futures
# 导入系统监控和管理模块
import psutil

# TODO: figure out why this is happening
# 设置GRPC日志级别为错误
os.environ["GRPC_VERBOSITY"] = "error"
# 设置transformers库日志级别为错误
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# 启用tokenizers并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["DEBUG"] = "4"
# Configure uvloop for maximum performance
# 配置uvloop以获得最佳性能
def configure_uvloop():
    # 在 Windows 上优先使用默认事件循环
    if psutil.WINDOWS:
        loop = asyncio.ProactorEventLoop() if hasattr(asyncio, 'ProactorEventLoop') else asyncio.get_event_loop()
    else:
        # Unix 系统上使用 uvloop（如果可用）
        if uvloop is not None:
            uvloop.install()
        loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)

    # Increase file descriptor limits on Unix systems
    # 在Unix系统上增加文件描述符限制
    if not psutil.WINDOWS:
      import resource
      # 获取当前文件描述符限制
      soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
      try: 
        # 尝试将软限制和硬限制都设置为硬限制的值
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
      except ValueError:
        try: 
          # 若失败，尝试将软限制设置为8192
          resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        except ValueError: 
          pass

    # 设置默认的线程池执行器
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4)))
    return loop

# 解析命令行参数
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("command", nargs="?", choices=["run", "eval", "train"], help="Command to run")
parser.add_argument("model_name", nargs="?", help="Model name to run")
parser.add_argument("--default-model", type=str, default=None, help="Default model")
parser.add_argument("--iters", type=int, default=100, help="Training iterations")
parser.add_argument("--save-every", type=int, default=5, help="Save the model every N iterations.")
parser.add_argument("--data", type=str, default="exo/train/data/lora", help="Directory where training data lives")
parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size.")
parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a custom checkpoint to load")
parser.add_argument("--save-checkpoint-dir", type=str, default="checkpoints", help="Path to a folder where checkpoints are stored")
parser.add_argument("--node-id", type=str, default=None, help="Node ID (auto-generated if not specified)")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=None, help="Node port")
parser.add_argument("--models-seed-dir", type=str, default=None, help="Model seed directory")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--download-quick-check", action="store_true", help="Quick check local path for model shards download")
parser.add_argument("--max-parallel-downloads", type=int, default=8, help="Max parallel downloads for model shards download")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual", "frp"], default="manual", help="Discovery module to use")
parser.add_argument("--frp-server-addr", type=str, default=None, help="FRP server address (for frp discovery)")
parser.add_argument("--frp-server-port", type=int, default=7000, help="FRP server port (for frp discovery)")
parser.add_argument("--frp-token", type=str, default=None, help="FRP authentication token (for frp discovery)")
parser.add_argument("--frp-remote-port", type=int, default=None, help="FRP remote port (for frp discovery, optional)")
parser.add_argument("--seed-peers", type=str, default=None, help="Seed peers for frp discovery, format: 'node1@addr:port,node2@addr:port'")
parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
parser.add_argument("--disable-p2p", action=argparse.BooleanOptionalAction, default=False, help="Disable P2P (xtcp) mode for FRP discovery (default: enabled)")
parser.add_argument("--discovery-config-path", type=str, default=None, help="Path to discovery config json file")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=52415, help="ChatGPT API port")
parser.add_argument("--chatgpt-api-response-timeout", type=int, default=900, help="ChatGPT API response timeout in seconds")
parser.add_argument("--max-generate-tokens", type=int, default=1024, help="Max tokens to generate in each request")
parser.add_argument("--inference-engine", type=str, default="pytorch", help="Inference engine to use (mlx, pytorch, or dummy)")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="Disable TUI")
parser.add_argument("--run-model", type=str, help="Specify a model to run directly")
parser.add_argument("--prompt", type=str, help="Prompt for the model when using --run-model", default="Who are you?")
parser.add_argument("--default-temp", type=float, help="Default token sampling temperature", default=0.0)
parser.add_argument("--tailscale-api-key", type=str, default=None, help="Tailscale API key")
parser.add_argument("--tailnet-name", type=str, default=None, help="Tailnet name")
parser.add_argument("--node-id-filter", type=str, default=None, help="Comma separated list of allowed node IDs (only for UDP and Tailscale discovery)")
parser.add_argument("--interface-type-filter", type=str, default=None, help="Comma separated list of allowed interface types (only for UDP discovery)")
parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for the ChatGPT API")
parser.add_argument("--manager", type=str, default=None, help="EXO Manager address (e.g., http://localhost:8080) to auto-register on startup")
parser.add_argument("--auto-connect", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable auto-connect to discovered peers (default: enabled)")
args = parser.parse_args()
print(f"Selected inference engine: {args.inference_engine}")

def _load_config_for_node(config_path: str, node_id: str):
  """从 network_config 中自动读取当前节点的 port 和 default_model"""
  try:
    import json
    with open(config_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    peer = data.get("peers", {}).get(node_id)
    if not peer:
      return None, None
    port = peer.get("port")
    model_id = None
    shard = peer.get("shard")
    if shard:
      if isinstance(shard, list) and len(shard) > 0:
        model_id = shard[0].get("model_id")
      elif isinstance(shard, dict):
        model_id = shard.get("model_id")
    return port, model_id
  except Exception as e:
    if DEBUG >= 1: print(f"[Main] Warning: failed to read config from {config_path}: {e}")
    return None, None

if args.discovery_module == "manual" and args.discovery_config_path:
  cfg_port, cfg_model = _load_config_for_node(args.discovery_config_path, args.node_id)
  if cfg_port is not None and args.node_port is None:
    args.node_port = cfg_port
    if DEBUG >= 1: print(f"[Main] Auto-set node_port from config: {args.node_port}")
  if cfg_model is not None and args.default_model is None:
    args.default_model = cfg_model
    if DEBUG >= 1: print(f"[Main] Auto-set default_model from config: {args.default_model}")

# 打印黄色的Exo标识
print_yellow_exo()

# 获取系统信息
system_info = get_system_info()
print(f"Detected system: {system_info}")

# 根据推理引擎类型选择分片下载器
shard_downloader: ShardDownloader = new_shard_downloader(args.max_parallel_downloads) if args.inference_engine != "dummy" else NoopShardDownloader()
# 自动选择推理引擎：如果用户未指定，在Apple Silicon Mac上使用mlx，其他系统使用pytorch
# 现在也支持pytorch作为可选的推理引擎
inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "pytorch")
print(f"Inference engine name after selection: {inference_engine_name}")

# 获取推理引擎实例
inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

# 如果未指定节点端口，自动查找可用端口
if args.node_port is None:
  args.node_port = find_available_port(args.node_host)
  if DEBUG >= 1: print(f"Using available port: {args.node_port}")

# 如果未指定节点ID，获取或创建一个节点ID
args.node_id = args.node_id or get_or_create_node_id()
# 生成ChatGPT API端点列表
chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
# 生成Web聊天URL列表
web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]
if DEBUG >= 0:
  print("Chat interface started:")
  for web_chat_url in web_chat_urls:
    print(f" - {terminal_link(web_chat_url)}")
  print("ChatGPT API endpoint served at:")
  for chatgpt_api_endpoint in chatgpt_api_endpoints:
    print(f" - {terminal_link(chatgpt_api_endpoint)}")

# Convert node-id-filter and interface-type-filter to lists if provided
# 如果提供了节点ID过滤器和接口类型过滤器，将其转换为列表
allowed_node_ids = args.node_id_filter.split(',') if args.node_id_filter else None
allowed_interface_types = args.interface_type_filter.split(',') if args.interface_type_filter else None

# 根据发现模块类型创建发现实例
if args.discovery_module == "udp":
  discovery = UDPDiscovery(
    args.node_id,
    args.node_port,
    args.listen_port,
    args.broadcast_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    allowed_node_ids=allowed_node_ids,
    allowed_interface_types=allowed_interface_types
  )
elif args.discovery_module == "tailscale":
  discovery = TailscaleDiscovery(
    args.node_id,
    args.node_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    tailscale_api_key=args.tailscale_api_key,
    tailnet=args.tailnet_name,
    allowed_node_ids=allowed_node_ids
  )
elif args.discovery_module == "manual":
  discovery = ManualDiscovery(args.discovery_config_path, args.node_id, create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))
elif args.discovery_module == "frp":
  if not args.frp_server_addr:
    raise ValueError("FRP server address is required when using frp discovery module. Please use --frp-server-addr to specify.")
  
  # 使用默认设备能力（稍后可以在节点启动后更新）
  from exo.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
  
  discovery = FRPDiscovery(
    frp_server_addr=args.frp_server_addr,
    frp_server_port=args.frp_server_port,
    node_id=args.node_id,
    local_port=args.node_port,
    create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    frp_token=args.frp_token,
    frp_remote_port=args.frp_remote_port,
    seed_peers=args.seed_peers,
    discovery_timeout=args.discovery_timeout,
    device_capabilities=UNKNOWN_DEVICE_CAPABILITIES,
    enable_p2p=not args.disable_p2p  # ✅ 默认启用 P2P，使用 --disable-p2p 禁用
  )
# 根据是否禁用TUI创建拓扑可视化实例
topology_viz = TopologyViz(chatgpt_api_endpoints=chatgpt_api_endpoints, web_chat_urls=web_chat_urls) if not args.disable_tui else None
# 创建节点实例
node = Node(
  args.node_id,
  None,  # server will be set later
  inference_engine,
  discovery,
  shard_downloader,
  partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
  max_generate_tokens=args.max_generate_tokens,
  topology_viz=topology_viz,
  default_sample_temperature=args.default_temp,
  manager_url=args.manager,
  chatgpt_api_port=args.chatgpt_api_port,
  auto_connect=args.auto_connect
)
# 创建GRPC服务器实例
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server
# 创建ChatGPT API实例
api = ChatGPTAPI(
  node,
  node.inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
  default_model=args.default_model,
  system_prompt=args.system_prompt
)
# 用于缓冲token输出的字典
buffered_token_output = {}
# 更新拓扑可视化的token输出
def update_topology_viz(req_id, tokens, __):
  if not topology_viz: return
  # 获取当前分片信息
  current_shard = node.inference_engine.shard
  if not current_shard: return
  if current_shard.model_id == 'stable-diffusion-2-1-base': return
  if req_id in buffered_token_output: buffered_token_output[req_id].extend(tokens)
  else: buffered_token_output[req_id] = tokens
  # 使用 get_tokenizer 获取正确模型的 tokenizer
  tokenizer = node.inference_engine.get_tokenizer(current_shard.model_id)
  if tokenizer:
    topology_viz.update_prompt_output(req_id, tokenizer.decode(buffered_token_output[req_id]))
# 注册token事件回调
node.on_token.register("update_topology_viz").on_next(update_topology_viz)
# 更新拓扑可视化的提示信息
def update_prompt_viz(request_id, opaque_status: str):
  if not topology_viz: return
  try:
    status = json.loads(opaque_status)
    if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
    topology_viz.update_prompt(request_id, status.get("prompt", "corrupted prompt (this should never happen)"))
  except Exception as e:
    if DEBUG >= 2:
      print(f"Failed to update prompt viz: {e}")
      traceback.print_exc()
# 注册不透明状态事件回调
node.on_opaque_status.register("update_prompt_viz").on_next(update_prompt_viz)

# 预先加载分片
def preemptively_load_shard(request_id: str, opaque_status: str):
  async def _async_load():
    try:
      status = json.loads(opaque_status)
      if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
      current_shard = await node.get_current_shard(Shard.from_dict(status.get("shard")))
      if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
      asyncio.create_task(node.inference_engine.ensure_shard(current_shard))
    except Exception as e:
      if DEBUG >= 2:
        print(f"Failed to preemptively start download: {e}")
        traceback.print_exc()
  # 创建异步任务来执行加载
  asyncio.create_task(_async_load())
# 注册不透明状态事件回调
node.on_opaque_status.register("preemptively_load_shard").on_next(preemptively_load_shard)

# 存储最后一次下载事件的字典
last_events: dict[str, tuple[float, RepoProgressEvent]] = {}
# 节流广播下载进度事件
def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
  global last_events
  current_time = time.time()
  if event.status == "not_started": return
  last_event = last_events.get(shard.model_id)
  if last_event and last_event[1].status == "complete" and event.status == "complete": return
  if last_event and last_event[0] == event.status and current_time - last_event[0] < 0.2: return
  last_events[shard.model_id] = (current_time, event)
  asyncio.create_task(node.broadcast_opaque_status("", json.dumps({"type": "download_progress", "node_id": node.id, "progress": event.to_dict()})))
# 注册分片下载进度事件回调
shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)

# 命令行运行模型
async def run_model_cli(node: Node, model_name: str, prompt: str):
  inference_class = node.inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  request_id = str(uuid.uuid4())
  callback_id = f"cli-wait-response-{request_id}"
  callback = node.on_token.register(callback_id)
  if topology_viz:
    topology_viz.update_prompt(request_id, prompt)
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

  try:
    print(f"Processing prompt: {prompt}")
    await node.process_prompt(shard, prompt, request_id=request_id)

    tokens = []
    def on_token(_request_id, _tokens, _is_finished):
      tokens.extend(_tokens)
      return _request_id == request_id and _is_finished
    await callback.wait(on_token, timeout=300)

    print("\nGenerated response:")
    print(tokenizer.decode(tokens))
  except Exception as e:
    print(f"Error processing prompt: {str(e)}")
    traceback.print_exc()
  finally:
    node.on_token.deregister(callback_id)

# 清理和解析路径
def clean_path(path):
    """Clean and resolve path"""
    if path.startswith("Optional("):
        path = path.strip('Optional("').rstrip('")')
    return os.path.expanduser(path)

# 等待所有未完成的请求完成
async def hold_outstanding(node: Node):
  while node.outstanding_requests:
    await asyncio.sleep(.5)
  return

# 运行一次迭代，用于训练或评估
async def run_iter(node: Node, shard: Shard, train: bool, data, batch_size=1):
  losses = []
  tokens = []
  for batch in tqdm(iterate_batches(data, batch_size), total=len(data) // batch_size):
    _, _, lengths = batch
    losses.append(np.sum(lengths * await node.enqueue_example(shard, *batch, train=train)))
    tokens.append(np.sum(lengths))
  total_tokens = np.sum(tokens)
  total_loss = np.sum(losses) / total_tokens

  return total_loss, total_tokens

# 命令行评估模型
async def eval_model_cli(node: Node, model_name, dataloader, batch_size, num_batches=-1):
  inference_class = node.inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Evaluating {len(test)} examples with batch_size {batch_size}")
  loss, tokens = await run_iter(node, shard, False, test, batch_size)
  print(f"total | {loss=}, {tokens=}")
  print("Waiting for outstanding tasks")
  await hold_outstanding(node)

# 命令行训练模型
async def train_model_cli(node: Node, model_name, dataloader, batch_size, iters, save_interval=0, checkpoint_dir=None):
  inference_class = node.inference_engine.__class__.__name__
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Training on {len(train)} examples with batch_size {batch_size} for {iters} epochs")
  for i in tqdm(range(3)):
    await asyncio.sleep(1)
  for epoch in range(iters):
    loss, tokens = await run_iter(node, shard, True, train, batch_size)
    print(f"epoch {epoch + 1}/{iters}\t| loss: {loss}, tokens: {tokens}")
    if save_interval > 0 and epoch > 0 and (epoch % save_interval) == 0 and checkpoint_dir is not None:
      await node.coordinate_save(shard, epoch, checkpoint_dir)
      await hold_outstanding(node)
  await hold_outstanding(node)

# 检查exo主目录的权限
async def check_exo_home():
  home, has_read, has_write = await ensure_exo_home(), await has_exo_home_read_access(), await has_exo_home_write_access()
  if DEBUG >= 1: print(f"exo home directory: {home}")
  print(f"{has_read=}, {has_write=}")
  if not has_read or not has_write:
    print(f"""
          WARNING: Limited permissions for exo home directory: {home}.
          This may prevent model downloads from working correctly.
          {"❌ No read access" if not has_read else ""}
          {"❌ No write access" if not has_write else ""}
          """)

# 主异步函数
async def main():
  loop = asyncio.get_running_loop()

  try: await check_exo_home()
  except Exception as e: print(f"Error checking exo home directory: {e}")

  if not args.models_seed_dir is None:
    try:
      models_seed_dir = clean_path(args.models_seed_dir)
      await seed_models(models_seed_dir)
    except Exception as e:
      print(f"Error seeding models: {e}")

  # 恢复光标显示的函数
  def restore_cursor():
    if platform.system() != "Windows":
        os.system("tput cnorm")  # Show cursor

  # Restore the cursor when the program exits
  # 程序退出时恢复光标显示
  atexit.register(restore_cursor)

  # Use a more direct approach to handle signals
  # 使用更直接的方式处理信号
  def handle_exit():
    asyncio.ensure_future(shutdown(signal.SIGTERM, loop, node.server))

  if platform.system() != "Windows":
    for s in [signal.SIGINT, signal.SIGTERM]:
      loop.add_signal_handler(s, handle_exit)

  # 启动节点
  await node.start(wait_for_peers=args.wait_for_peers)

  # 如果使用 manual discovery 且配置了分片，预加载模型
  if args.discovery_module == "manual" and isinstance(discovery, ManualDiscovery):
    # 先主动加载节点配置
    await discovery.load_node_config()
    # 获取所有配置的分片（支持多模型分片配置）
    configured_shards = discovery.get_current_node_shards()
    if configured_shards:
      print(f"[Main] Pre-loading {len(configured_shards)} configured shard(s)...")
      for configured_shard in configured_shards:
        print(f"[Main] Pre-loading shard: {configured_shard}")
        try:
          # 使用推理引擎的类名来获取正确的仓库ID
          engine_class_name = node.inference_engine.__class__.__name__
          model_path = await shard_downloader.ensure_shard(configured_shard, engine_class_name)
          print(f"[Main] Model path resolved: {model_path}")
          await node.inference_engine.load_checkpoint(configured_shard, model_path)
          print(f"[Main] Successfully pre-loaded shard for model {configured_shard.model_id}")
        except Exception as e:
          print(f"[Main] Failed to pre-load shard for model {configured_shard.model_id}: {e}")
          traceback.print_exc()
    else:
      print(f"[Main] No shard configured for node {args.node_id}, skipping pre-load")

  if args.command == "run" or args.run_model:
    model_name = args.model_name or args.run_model
    if not model_name:
      print("Error: Model name is required when using 'run' command or --run-model")
      return
    await run_model_cli(node, model_name, args.prompt)
  elif args.command == "eval" or args.command == 'train':
    model_name = args.model_name
    dataloader = lambda tok: load_dataset(args.data, preprocess=lambda item: tok(item)
                                                   , loadline=lambda line: json.loads(line).get("text",""))
    if args.command == 'eval':
      if not model_name:
        print("Error: Much like a human, I can't evaluate anything without a model")
        return
      await eval_model_cli(node, model_name, dataloader, args.batch_size)
    else:
      if not model_name:
        print("Error: This train ain't leaving the station without a model")
        return
      await train_model_cli(node, model_name, dataloader, args.batch_size, args.iters, save_interval=args.save_every, checkpoint_dir=args.save_checkpoint_dir)

  else:
    asyncio.create_task(api.run(port=args.chatgpt_api_port))  # Start the API server as a non-blocking task
    await asyncio.Event().wait()

  if args.wait_for_peers > 0:
    print("Cooldown to allow peers to exit gracefully")
    for i in tqdm(range(50)):
      await asyncio.sleep(.1)

# 运行程序的函数
def run():
    loop = None
    try:
        loop = configure_uvloop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
    finally:
        if loop:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except:
                pass
            loop.close()


if __name__ == "__main__":
  run()
