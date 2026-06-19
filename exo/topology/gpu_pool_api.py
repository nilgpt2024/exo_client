"""
统一GPU显存池 - 高级API接口
===========================

提供简单易用的接口来管理分布式GPU显存池中的模型权重

使用方式:
    from exo.topology.gpu_pool_api import GPUPoolAPI
    
    api = GPUPoolAPI(node)
    
    # 一键加载模型（自动分片到所有节点）
    result = await api.load("Qwen/Qwen3-4B", "./models/qwen3-4b")
    
    # 查看池状态
    api.status()
    
    # 卸载模型
    await api.unload("Qwen/Qwen3-4B")
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from .gpu_pool_manager import GPUPoolManager, ModelState, AllocationPlan
from exo.inference.shard import Shard
from exo import DEBUG

logger = logging.getLogger(__name__)


class GPUPoolAPI:
    """
    GPU显存池的高级API
    
    提供简洁的接口，封装了GPUPoolManager的所有功能
    """
    
    def __init__(self, node):
        """
        初始化API
        
        Args:
            node: Node实例
        """
        self.node = node
        self.manager = GPUPoolManager(node)
        self._initialized = False
    
    async def initialize(self):
        """初始化池管理器"""
        if not self._initialized:
            await self.manager.initialize()
            self._initialized = True
            logger.info("[GPUPoolAPI] 初始化完成")
        return self
    
    async def load(
        self,
        model_id: str,
        model_path: str,
        *,
        nodes: Optional[List[str]] = None,
        strategy: str = "memory_weighted",
        n_layers: Optional[int] = None,
        instance_id: Optional[str] = None,  # [OK] 新增实例ID参数
        force: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        加载模型到GPU池
        
        这是主要的使用方法，会自动完成：
        1. 检测模型层数
        2. 根据策略计算最优分配方案
        3. 将模型的各层分配到不同节点
        4. 在各节点上加载对应的权重
        
        Args:
            model_id: 模型标识符 (如 "Qwen/Qwen3-4B" 或 "qwen3-0.6b::worker-1")
            model_path: 模型文件路径
            nodes: 指定使用的节点列表 (None=所有可用节点)
            strategy: 分配策略:
                - "memory_weighted": 基于内存大小分配 (默认)
                - "uniform": 均匀分配
                - "performance_weighted": 基于性能分配
            n_layers: 手动指定总层数 (None=自动检测)
            instance_id: 实例ID (支持多实例，如 "worker-1", "worker-2")
            force: 强制重新加载
            **kwargs: 额外参数传递给推理引擎
            
        Returns:
            包含操作结果的字典
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"[API] 加载模型: {model_id}" + (f" (实例: {instance_id})" if instance_id else ""))
        
        try:
            plan = await self.manager.load_model(
                model_id=model_id,
                model_path=model_path,
                n_layers=n_layers,
                target_nodes=nodes,
                strategy=strategy,
                instance_id=instance_id,  # ✅ 传递实例ID
                force_reload=force,
                **kwargs
            )
            
            result = {
                "success": plan.success,
                "model_id": model_id,
                "message": plan.reason,
                "allocations": {
                    node_id: {
                        "layers": f"{shard.start_layer}-{shard.end_layer}",
                        "layer_count": shard.get_layer_count()
                    }
                    for node_id, shard in plan.allocations.items()
                },
                "warnings": plan.warnings if hasattr(plan, 'warnings') else []
            }

            if not plan.success:
                result["error"] = plan.reason or f"分配计划创建失败 (success={plan.success})"
            elif plan.warnings:
                result["error"] = "; ".join(plan.warnings)
                if result["success"]:
                    logger.warning(f"[API] 模型加载完成但有警告: {plan.warnings}")

            return result
            
        except Exception as e:
            logger.error(f"[API] 加载失败: {e}")
            return {
                "success": False,
                "model_id": model_id,
                "error": str(e),
                "message": f"加载模型失败: {e}"
            }
    
    async def load_single_shard(
        self,
        model_id: str,
        model_path: str,
        shard,
        base_model_id: str = None
    ) -> Dict[str, Any]:
        """
        加载单个分片到当前节点（供 Manager 远程调用）
        
        这是 EXO Manager 通过 gRPC 发送命令时调用的方法，
        只在当前节点加载指定的层范围，不进行自动分配。
        
        Args:
            model_id: 完整模型ID（可能包含 ::instance_id）
            model_path: 模型路径
            shard: Shard 对象，包含 start_layer, end_layer, n_layers
            base_model_id: 基础模型ID（不含实例信息，用于文件系统操作）
            
        Returns:
            操作结果字典
        """
        if not self._initialized:
            await self.initialize()
        
        # 确定用于下载的基础模型ID
        effective_base_id = base_model_id or model_id
        if "::" in effective_base_id:
            effective_base_id = effective_base_id.split("::")[0]
        
        logger.info(f"[API] [Manager] 加载单分片: {model_id} (基础ID: {effective_base_id}) L{shard.start_layer}-L{shard.end_layer}")
        
        try:
            from exo.inference.shard import Shard as ShardType
            
            if not isinstance(shard, ShardType):
                # 创建 Shard 对象时使用 base_model_id，避免路径污染
                shard = ShardType(
                    model_id=effective_base_id,
                    start_layer=shard.get("start_layer", 0) if isinstance(shard, dict) else getattr(shard, "start_layer", 0),
                    end_layer=shard.get("end_layer", 0) if isinstance(shard, dict) else getattr(shard, "end_layer", 0),
                    n_layers=shard.get("n_layers", 0) if isinstance(shard, dict) else getattr(shard, "n_layers", 0),
                    instance_id=getattr(shard, "instance_id", "default") if hasattr(shard, "instance_id") else "default"
                )
            
            plan = await self.manager.load_model(
                model_id=model_id,
                base_model_id=effective_base_id,
                model_path=model_path,
                n_layers=shard.n_layers,
                target_nodes=None,
                strategy="custom",
                force_reload=True,
                custom_shards={self.manager.node.id: shard}
            )
            
            return {
                "success": plan.success,
                "model_id": model_id,
                "base_model_id": effective_base_id,
                "shard": {
                    "start_layer": shard.start_layer,
                    "end_layer": shard.end_layer,
                    "n_layers": shard.n_layers
                },
                "message": plan.reason
            }
            
        except Exception as e:
            logger.error(f"[API] [Manager] 单分片加载失败: {e}")
            return {
                "success": False,
                "model_id": model_id,
                "error": str(e),
                "message": f"单分片加载失败: {e}"
            }
    
    async def unload(self, model_id: str, *, nodes: Optional[List[str]] = None, unload_all_instances: bool = False) -> Dict[str, Any]:
        """
        卸载模型（支持多实例）

        Args:
            model_id: 模型ID
            nodes: 从哪些节点卸载 (None=所有节点)
            unload_all_instances: 是否卸载该模型的所有实例

        Returns:
            操作结果
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"[API] 卸载模型: {model_id} (卸载所有实例={unload_all_instances})")

        try:
            if "::" in model_id:
                parts = model_id.split("::", 1)
                parsed_model_id = parts[0]
                instance_id = parts[1]
            else:
                parsed_model_id = model_id
                instance_id = None

            success = await self.manager.unload_model(
                parsed_model_id,
                instance_id=instance_id,
                target_nodes=nodes,
                unload_all_instances=unload_all_instances
            )

            return {
                "success": success,
                "model_id": model_id,
                "message": "卸载成功" if success else "卸载失败"
            }

        except Exception as e:
            logger.error(f"[API] 卸载失败: {e}")
            return {
                "success": False,
                "model_id": model_id,
                "error": str(e)
            }
    
    async def rebalance(self, model_id: str) -> Dict[str, Any]:
        """
        重新平衡模型分布
        
        当添加新节点或节点资源变化时使用，
        会自动重新计算最优分配方案并迁移权重
        
        Args:
            model_id: 模型ID
            
        Returns:
            新的分配信息
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"[API] 重平衡: {model_id}")
        
        try:
            plan = await self.manager.rebalance(model_id)
            
            return {
                "success": True,
                "model_id": model_id,
                "message": "重平衡完成",
                "new_allocations": {
                    node_id: {
                        "layers": f"{shard.start_layer}-{shard.end_layer}",
                        "layer_count": shard.get_layer_count()
                    }
                    for node_id, shard in plan.allocations.items()
                }
            }
            
        except Exception as e:
            logger.error(f"[API] 重平衡失败: {e}")
            return {
                "success": False,
                "model_id": model_id,
                "error": str(e)
            }
    
    def status(self, *, model_id: Optional[str] = None, verbose: bool = True) -> Union[Dict, str]:
        """
        获取池状态
        
        Args:
            model_id: 特定模型ID (None=全部)
            verbose: 是否打印详细信息
            
        Returns:
            状态字典或格式化的字符串
        """
        if not self._initialized:
            return {"error": "未初始化"}
        
        if model_id:
            details = self.manager.get_model_details(model_id)
            if verbose and details:
                self._print_model_details(details)
            return details or {"error": f"模型 {model_id} 不存在"}
        
        pool_status = self.manager.get_pool_status()
        
        if verbose:
            self.manager.print_pool_status()
        
        return pool_status
    
    def list_models(self) -> List[Dict]:
        """
        列出池中所有模型
        
        Returns:
            模型列表及其基本信息
        """
        if not self._initialized:
            return []
        
        models = []
        for model_id in self.manager.list_available_models():
            details = self.manager.get_model_details(model_id)
            if details:
                models.append({
                    "id": model_id,
                    "state": details["state"],
                    "layers": f"{details['coverage']['covered']}/{details['n_layers']}",
                    "nodes": len(details["shards"]),
                    "path": details["model_path"]
                })
        
        return models
    
    def list_nodes(self) -> List[Dict]:
        """
        列出池中所有节点及其负载情况
        
        Returns:
            节点列表
        """
        if not self._initialized:
            return []
        
        nodes = []
        for node_id, res in self.manager.node_resources.items():
            nodes.append({
                "id": node_id,
                "device": res.device_caps.chip,
                "memory_total_gb": round(res.device_caps.memory / 1024, 1),
                "memory_available_gb": round(res.available_memory_mb / 1024, 1),
                "loaded_models": res.current_models.keys(),
                "flops_tflops": round(res.device_caps.flops.fp16 / 1000, 1)
            })
        
        return nodes
    
    def preview_allocation(
        self,
        model_id: str,
        n_layers: int,
        *,
        nodes: Optional[List[str]] = None,
        strategy: str = "memory_weighted"
    ) -> Dict:
        """
        预览分配计划（不实际执行）
        
        在实际加载前查看模型会如何被分配到各个节点
        
        Args:
            model_id: 模型ID
            n_layers: 总层数
            nodes: 目标节点
            strategy: 分配策略
            
        Returns:
            分配预览信息
        """
        if not self._initialized:
            return {"error": "未初始化"}
        
        preview = self.manager.get_allocation_plan_preview(
            model_id=model_id,
            n_layers=n_layers,
            target_nodes=nodes,
            strategy=strategy
        )
        
        if "error" not in preview:
            print("\n[LIST] 分配预览:")
            print(f"   模型: {model_id} ({n_layers} 层)")
            print(f"   策略: {strategy}")
            print(f"   使用节点: {preview['nodes_used']}\n")

            for alloc in preview["allocations"]:
                print(f"   [PIN] {alloc['node_id']}")
                print(f"      设备: {alloc['device']}")
                print(f"      层数: {alloc['layers']} ({alloc['layer_count']} 层)")
                print(f"      占比: {alloc['ratio']}")
                print(f"      可用内存: {alloc['available_memory_mb']/1024:.1f} GB\n")
        
        return preview
    
    def _print_model_details(self, details: Dict):
        """打印模型详情"""
        print(f"\n{'='*60}")
        print(f"📦 模型: {details['model_id']}")
        print(f"{'='*60}")
        print(f"\n状态: {details['state']}")
        print(f"路径: {details['model_path']}")
        print(f"总层数: {details['n_layers']}")
        print(f"覆盖: {details['coverage']['percent']}% "
              f"({len(details['coverage']['covered'])}/{details['n_layers']} 层)")
        
        if details['coverage']['missing']:
            print(f"⚠️ 缺失层: {details['coverage']['missing']}")
        
        print(f"\n分片分布:")
        for i, shard in enumerate(details["shards"], 1):
            first_last = ""
            if shard["is_first"]:
                first_last = " [首层]"
            if shard["is_last"]:
                first_last += " [尾层]"
            
            print(f"  {i}. 节点 {shard['node_id']}: "
                  f"层 {shard['start_layer']}-{shard['end_layer']}"
                  f" ({shard['layer_count']}层){first_last}")
        
        print(f"\n创建时间: {details['created_at']}")
        print(f"最后访问: {details['last_accessed']}")
        print(f"{'='*60}\n")


class GPUBalancerCLI:
    """
    GPU显存池管理器的命令行界面
    
    提供交互式命令行操作
    """
    
    def __init__(self, api: GPUPoolAPI):
        self.api = api
    
    async def run_command(self, cmd: str, **kwargs) -> Any:
        """
        执行命令
        
        支持的命令:
            - load <model_id> <path> [--nodes ...] [--strategy ...]
            - unload <model_id>
            - rebalance <model_id>
            - status [--model ...]
            - list-models
            - list-nodes
            - preview <model_id> <n_layers>
            - help
        """
        parts = cmd.strip().split()
        if not parts:
            return {"error": "空命令"}
        
        command = parts[0].lower()
        args = parts[1:]
        
        if command == "load":
            if len(args) < 2:
                return {"error": "用法: load <model_id> <model_path>"}
            
            model_id = args[0]
            model_path = args[1]
            
            extra_kwargs = {}
            if "--nodes" in args:
                idx = args.index("--nodes")
                extra_kwargs["nodes"] = args[idx+1:].split(",")
            if "--strategy" in args:
                idx = args.index("--strategy")
                extra_kwargs["strategy"] = args[idx+1]
            if "--force" in args:
                extra_kwargs["force"] = True
            
            return await self.api.load(model_id, model_path, **extra_kwargs)
        
        elif command == "unload":
            if not args:
                return {"error": "用法: unload <model_id>"}
            return await self.api.unload(args[0])
        
        elif command == "rebalance":
            if not args:
                return {"error": "用法: rebalance <model_id>"}
            return await self.api.rebalance(args[0])
        
        elif command == "status":
            model_id = kwargs.get("model") or (args[0] if args else None)
            return self.api.status(model_id=model_id)
        
        elif command in ("list-models", "ls"):
            return self.api.list_models()
        
        elif command in ("list-nodes", "ln"):
            return self.api.list_nodes()
        
        elif command == "preview":
            if len(args) < 2:
                return {"error": "用法: preview <model_id> <n_layers>"}
            
            model_id = args[0]
            try:
                n_layers = int(args[1])
            except ValueError:
                return {"error": "n_layers 必须是整数"}
            
            extra = {}
            if "--strategy" in args:
                idx = args.index("--strategy")
                extra["strategy"] = args[idx+1]
            
            return self.api.preview_allocation(model_id, n_layers, **extra)
        
        elif command == "help":
            return self._help_text()
        
        else:
            return {"error": f"未知命令: {command}. 输入 'help' 查看帮助"}
    
    def _help_text(self) -> str:
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║              EXO GPU 显存池管理器 - 命令帮助                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📦 模型管理                                                 ║
║    load <model_id> <path>                                    ║
║         [--nodes node1,node2]  指定节点                      ║
║         [--strategy memory|uniform|perf]  分配策略           ║
║         [--force]             强制重载                       ║
║                                                              ║
║    unload <model_id>          卸载模型                       ║
║    rebalance <model_id>       重新平衡分布                   ║
║                                                              ║
║  [SEARCH] 状态查询                                                 ║
║    status [--model id]        查看池/模型状态                ║
║    list-models (ls)           列出所有模型                   ║
║    list-nodes (ln)            列出所有节点                   ║
║                                                              ║
║  📋 预览                                                     ║
║    preview <model_id> <n_layers>                             ║
║         [--strategy ...]     预览分配方案                    ║
║                                                              ║
║  [BULB] 示例                                                     ║
║    load qwen3-4b ./models/qwen3-4b                          ║
║    load llama3 ./models/llama3 --nodes node1,node2          ║
║    status                                                   ║
║    status --model qwen3-4b                                  ║
║    preview qwen3-4b 36                                      ║
║    rebalance qwen3-4b                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)
        return help_text


__all__ = ['GPUPoolAPI', 'GPUBalancerCLI']
