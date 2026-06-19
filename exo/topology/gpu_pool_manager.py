"""
统一GPU显存池权重调配系统
========================

核心功能：
1. 统一管理所有节点的GPU显存资源
2. 自动将模型权重分片到合适的节点
3. 支持动态加载/卸载/迁移模型权重
4. 提供统一的API接口供用户调用

使用示例:
    manager = GPUPoolManager(node)
    
    # 加载模型（自动分片到所有可用节点）
    await manager.load_model("Qwen/Qwen3-4B", model_path="./models/qwen3-4b")
    
    # 查看当前池子状态
    status = manager.get_pool_status()
    
    # 卸载模型
    await manager.unload_model("Qwen/Qwen3-4B")
    
    # 重新分配（比如添加了新节点）
    await manager.rebalance("Qwen/Qwen3-4B")
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
from pathlib import Path

from exo.inference.shard import Shard, ModelLoadState
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceMemory
from exo.topology.partitioning_strategy import Partition, map_partitions_to_shards
from exo import DEBUG

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """模型在池中的状态"""
    LOADING = "loading"        # 正在加载
    LOADED = "loaded"          # 已完全加载（所有层都已分配）
    PARTIAL = "partial"        # 部分加载（只有部分层）
    UNLOADING = "unloading"    # 正在卸载
    ERROR = "error"            # 加载出错
    MIGRATING = "migrating"    # 正在迁移


@dataclass
class NodeResourceInfo:
    """节点资源信息"""
    node_id: str
    address: str = ""
    port: int = 0
    device_caps: DeviceCapabilities = field(default_factory=lambda: DeviceCapabilities())
    current_models: Dict[str, ModelLoadState] = field(default_factory=dict)
    available_memory_mb: int = 0
    used_memory_mb: int = 0
    
    def update_memory(self):
        """更新内存信息"""
        if self.device_caps.memory_detail:
            self.available_memory_mb = self.device_caps.memory_detail.free
            self.used_memory_mb = self.device_caps.memory_detail.used
        else:
            self.available_memory_mb = self.device_caps.memory
            self.used_memory_mb = 0


@dataclass 
class PoolModelInfo:
    """池中模型的信息 - 支持多实例"""
    model_id: str
    model_path: str
    instance_id: str = "default"  # 实例ID，支持同一模型加载多个实例
    n_layers: int = 0
    state: ModelState = ModelState.LOADING
    shards: List[Shard] = field(default_factory=list)
    total_memory_required_mb: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_model_id(self) -> str:
        """获取完整的模型标识 (model_id::instance_id)"""
        if self.instance_id == "default":
            return self.model_id
        return f"{self.model_id}::{self.instance_id}"
    
    def get_coverage(self) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        获取层覆盖情况
        返回: (已覆盖的层集合, 缺失的层集合, 所有层集合)
        """
        covered = set()
        for shard in self.shards:
            for layer in range(shard.start_layer, shard.end_layer + 1):
                covered.add(layer)
        
        all_layers = set(range(self.n_layers))
        missing = all_layers - covered
        
        return covered, missing, all_layers
    
    def is_fully_covered(self) -> bool:
        """检查是否所有层都被覆盖"""
        _, missing, _ = self.get_coverage()
        return len(missing) == 0


@dataclass
class AllocationPlan:
    """分配计划 - 支持多实例"""
    model_id: str
    base_model_id: str = None  # 基础模型ID（不含实例信息）
    instance_id: str = "default"  # 实例ID
    allocations: Dict[str, Shard] = field(default_factory=dict)  # node_id -> Shard
    estimated_memory_per_node: Dict[str, int] = field(default_factory=dict)  # node_id -> memory_mb
    reason: str = ""
    success: bool = True
    warnings: List[str] = field(default_factory=list)
    
    @property
    def full_model_id(self) -> str:
        """获取完整的模型标识"""
        if self.instance_id == "default":
            return self.model_id
        return f"{self.model_id}::{self.instance_id}"
    
    @property
    def effective_base_model_id(self) -> str:
        """获取有效的基础模型ID（用于文件系统操作）"""
        if self.base_model_id:
            return self.base_model_id
        if "::" in self.model_id:
            return self.model_id.split("::")[0]
        return self.model_id


class GPUPoolManager:
    """
    统一GPU显存池管理器
    
    将所有连接的节点视为一个统一的GPU显存池，
    提供模型权重的自动分配、迁移、查询等功能。
    """
    
    def __init__(self, node):
        """
        初始化池管理器
        
        Args:
            node: Node实例，用于访问拓扑、发现、推理引擎等
        """
        self.node = node
        self.pool_models: Dict[str, PoolModelInfo] = {}  # full_model_id -> PoolModelInfo
        self.node_resources: Dict[str, NodeResourceInfo] = {}  # node_id -> NodeResourceInfo
        self.allocation_history: List[Dict] = []  # 分配历史记录
        self._lock = asyncio.Lock()
        self._model_config_cache: Dict[str, dict] = {}  # 模型配置缓存
        self._instance_counter: Dict[str, int] = {}  # model_id -> 实例计数器
        
        logger.info("[GPUPool] 初始化完成（支持多实例模式）")
    
    def _make_key(self, model_id: str, instance_id: str = "default") -> str:
        """生成池模型的键"""
        if instance_id == "default":
            return model_id
        return f"{model_id}::{instance_id}"
    
    def _parse_key(self, key: str) -> Tuple[str, str]:
        """解析键为 (model_id, instance_id)"""
        if "::" in key:
            parts = key.split("::", 1)
            return parts[0], parts[1]
        return key, "default"
    
    def _generate_instance_id(self, model_id: str) -> str:
        """
        为模型自动生成唯一的实例ID
        
        规则：worker-1, worker-2, ... 或 auto-1, auto-2, ...
        
        [STAR] 增强版：包含重复检测机制，确保生成的ID不会与现有实例冲突
        """
        if model_id not in self._instance_counter:
            self._instance_counter[model_id] = 0
        
        # 先尝试基于计数器生成
        self._instance_counter[model_id] += 1
        count = self._instance_counter[model_id]
        candidate_id = f"worker-{count}"
        
        # [OK] 重复检测：检查该ID是否已被使用
        full_key = self._make_key(model_id, candidate_id)
        max_retries = 100  # 防止无限循环
        retry_count = 0
        
        while full_key in self.pool_models and retry_count < max_retries:
            logger.warning(f"[GPUPool] ⚠️ 检测到实例ID冲突: {candidate_id} 已存在，尝试下一个...")
            self._instance_counter[model_id] += 1
            count = self._instance_counter[model_id]
            candidate_id = f"worker-{count}"
            full_key = self._make_key(model_id, candidate_id)
            retry_count += 1
        
        if retry_count > 0:
            logger.info(f"[GPUPool] [OK] 经过 {retry_count} 次重试，生成唯一实例ID: {candidate_id}")
        
        return candidate_id
    
    async def initialize(self):
        """初始化：从现有节点信息构建资源视图"""
        logger.info("[GPUPool] 正在初始化资源视图...")
        
        # 从topology获取节点信息
        for node_id, caps in self.node.topology.all_nodes():
            self._update_node_resource(node_id, caps)
        
        # 从已加载的模型信息同步
        self._sync_existing_models()
        
        logger.info(f"[GPUPool] 初始化完成: {len(self.node_resources)} 个节点, {len(self.pool_models)} 个模型")
    
    def _update_node_resource(self, node_id: str, caps: DeviceCapabilities):
        """更新或添加节点资源信息"""
        if node_id in self.node_resources:
            resource = self.node_resources[node_id]
            resource.device_caps = caps
            resource.update_memory()
        else:
            resource = NodeResourceInfo(
                node_id=node_id,
                device_caps=caps
            )
            resource.update_memory()
            self.node_resources[node_id] = resource
            
        logger.debug(f"[GPUPool] 更新节点资源: {node_id}, 可用内存: {resource.available_memory_mb}MB")

    def _sync_resources_from_topology(self):
        """从当前拓扑同步节点资源信息

        每次 load/unload 前调用，确保 node_resources 与 topology 同步
        这样新发现的节点能立即参与模型分配
        """
        try:
            current_topology_nodes = set(self.node.topology.nodes.keys())
            current_resource_nodes = set(self.node_resources.keys())

            # 添加拓扑中新出现的节点
            new_nodes = current_topology_nodes - current_resource_nodes
            for node_id in new_nodes:
                caps = self.node.topology.nodes.get(node_id)
                if caps:
                    self._update_node_resource(node_id, caps)
                    logger.info(f"[GPUPool] [SYNC] 新增节点到资源池: {node_id}")

            # 移除已从拓扑中消失的节点
            removed_nodes = current_resource_nodes - current_topology_nodes
            for node_id in removed_nodes:
                if node_id in self.node_resources:
                    del self.node_resources[node_id]
                    logger.info(f"[GPUPool] [SYNC] 从资源池移除节点: {node_id}")

            # 更新已有节点的资源信息（内存等会变化）
            for node_id in current_topology_nodes & current_resource_nodes:
                caps = self.node.topology.nodes.get(node_id)
                if caps:
                    self._update_node_resource(node_id, caps)

            if new_nodes or removed_nodes:
                logger.info(f"[GPUPool] [SYNC] 资源同步完成: "
                           f"+{len(new_nodes)} -{len(removed_nodes)} = {len(self.node_resources)} 节点")
            else:
                logger.debug(f"[GPUPool] [SYNC] 资源已同步: {len(self.node_resources)} 节点 (无变化)")

        except Exception as e:
            logger.error(f"[GPUPool] [ERROR] 资源同步失败: {e}")
            import traceback
            traceback.print_exc()

    def _sync_existing_models(self):
        """从node中同步已存在的模型信息"""
        # 同步本节点加载的模型
        for model_id, load_state in self.node.my_loaded_models.items():
            if model_id not in self.pool_models:
                pool_info = PoolModelInfo(
                    model_id=model_id,
                    model_path="",
                    state=ModelState.LOADED,
                    shards=[load_state.shard],
                    n_layers=load_state.shard.n_layers
                )
                self.pool_models[model_id] = pool_info
            else:
                pool_info = self.pool_models[model_id]
                if load_state.shard not in pool_info.shards:
                    pool_info.shards.append(load_state.shard)
            
            # [OK] 同步实例计数器（修复ID冲突问题）
            self._sync_instance_counter_from_model_id(model_id)
        
        # 同步其他节点加载的模型
        for node_id, models in self.node.node_loaded_models.items():
            for model_id, load_state in models.items():
                if model_id not in self.pool_models:
                    pool_info = PoolModelInfo(
                        model_id=model_id,
                        model_path="",
                        state=ModelState.PARTIAL if not self._check_full_coverage(model_id) else ModelState.LOADED,
                        shards=[load_state.shard],
                        n_layers=load_state.shard.n_layers
                    )
                    self.pool_models[model_id] = pool_info
                else:
                    pool_info = self.pool_models[model_id]
                    if load_state.shard not in pool_info.shards:
                        pool_info.shards.append(load_state.shard)
                
                # [OK] 同步实例计数器（修复ID冲突问题）
                self._sync_instance_counter_from_model_id(model_id)
                        
                # 更新节点资源中的模型列表
                if node_id in self.node_resources:
                    self.node_resources[node_id].current_models[model_id] = load_state
    
    def _sync_instance_counter_from_model_id(self, full_model_id: str):
        """
        从完整的模型ID中提取并同步实例计数器
        
        Args:
            full_model_id: 完整模型ID (如 "qwen-3-vl-2b::worker-1" 或 "qwen3-0.6b")
        """
        base_model_id, instance_id = self._parse_key(full_model_id)
        
        # 只统计非默认实例
        if instance_id and instance_id != "default":
            # 从实例ID中提取数字 (如 "worker-1" -> 1)
            try:
                if instance_id.startswith("worker-"):
                    worker_num = int(instance_id.split("-")[1])
                    # 更新计数器为最大值
                    if base_model_id not in self._instance_counter:
                        self._instance_counter[base_model_id] = 0
                    
                    if worker_num > self._instance_counter[base_model_id]:
                        self._instance_counter[base_model_id] = worker_num
                        
                    logger.debug(f"[GPUPool] 同步实例计数器: {base_model_id} -> {self._instance_counter[base_model_id]} (来自 {full_model_id})")
            except (ValueError, IndexError):
                logger.warning(f"[GPUPool] 无法解析实例ID: {instance_id}")
    
    def _check_full_coverage(self, model_id: str) -> bool:
        """检查模型是否被完整覆盖"""
        if model_id not in self.pool_models:
            return False
        return self.pool_models[model_id].is_fully_covered()
    
    async def load_model(
        self,
        model_id: str,
        model_path: str,
        n_layers: Optional[int] = None,
        target_nodes: Optional[List[str]] = None,
        strategy: str = "memory_weighted",
        force_reload: bool = False,
        custom_shards: Optional[Dict] = None,
        instance_id: Optional[str] = None,
        auto_instance: bool = False,
        base_model_id: str = None,
        **kwargs
    ) -> AllocationPlan:
        """
        加载模型到GPU池（自动分片）- 支持多实例
        
        Args:
            model_id: 模型ID（如 "Qwen/Qwen3-4B" 或 "qwen3-0.6b::worker-1"）
            model_path: 模型文件路径
            n_layers: 模型总层数（如果为None则自动检测）
            target_nodes: 目标节点列表（None表示使用所有可用节点）
            strategy: 分配策略 ("memory_weighted", "uniform", "custom")
            force_reload: 是否强制重新加载（即使已存在）
            custom_shards: 自定义分片配置
            instance_id: 实例ID（None表示使用默认，支持多实例时指定不同ID）
            auto_instance: 是否自动生成实例ID（用于快速创建多个实例）
            base_model_id: 基础模型ID（不含 ::instance_id，用于文件系统操作）
            **kwargs: 其他参数传递给推理引擎
            
        Returns:
            AllocationPlan: 分配结果
        """
        async with self._lock:
            # 处理基础模型ID（用于文件系统操作）
            if base_model_id is None:
                if "::" in model_id:
                    base_model_id = model_id.split("::")[0]
                else:
                    base_model_id = model_id
            
            # 处理实例ID
            if auto_instance and instance_id is None:
                instance_id = self._generate_instance_id(base_model_id)
                logger.info(f"[GPUPool] 自动生成实例ID: {instance_id}")
            
            if instance_id is None:
                instance_id = "default"
            
            full_key = self._make_key(base_model_id, instance_id)
            
            # 确定显示用的完整模型ID
            display_model_id = f"{base_model_id}::{instance_id}" if instance_id != "default" else base_model_id
            
            logger.info(f"[GPUPool] 开始加载模型: {base_model_id} (实例: {instance_id}, 键: {full_key})")

            # [STAR] 同步节点资源：从当前拓扑刷新 node_resources
            # 这确保新发现的节点能参与模型分配
            self._sync_resources_from_topology()

            # 检查是否已存在（基于完整键）
            if full_key in self.pool_models and not force_reload:
                existing = self.pool_models[full_key]
                if existing.is_fully_covered():
                    logger.info(f"[GPUPool] 模型 {full_key} 已完整加载，跳过")
                    return AllocationPlan(
                        model_id=display_model_id,
                        base_model_id=base_model_id,
                        instance_id=instance_id,
                        allocations={s.model_id: s for s in existing.shards},
                        reason=f"模型实例 {instance_id} 已完整加载"
                    )
            
            # 获取或检测模型层数（使用 base_model_id）
            if n_layers is None:
                n_layers = await self._detect_model_layers(model_path, base_model_id)
                if n_layers is None:
                    raise ValueError(f"无法检测模型层数，请手动指定 n_layers 参数")
            
            logger.info(f"[GPUPool] 模型 {base_model_id} 总层数: {n_layers}")
            
            # 创建池模型信息（包含 instance_id，但 model_id 使用 base_model_id）
            pool_info = PoolModelInfo(
                model_id=base_model_id,
                model_path=model_path,
                instance_id=instance_id,
                n_layers=n_layers,
                state=ModelState.LOADING
            )
            self.pool_models[full_key] = pool_info  # 使用完整键存储
            
            try:
                # 计算分配方案
                if custom_shards and strategy == "custom":
                    logger.info(f"[GPUPool] 使用自定义分片配置: {list(custom_shards.keys())}")
                    
                    # 确保自定义分片中的 Shard 使用 base_model_id
                    sanitized_shards = {}
                    for node_id, shard in custom_shards.items():
                        if hasattr(shard, 'model_id') and "::" in shard.model_id:
                            from exo.inference.shard import Shard as ShardType
                            shard = ShardType(
                                model_id=base_model_id,
                                start_layer=shard.start_layer,
                                end_layer=shard.end_layer,
                                n_layers=shard.n_layers,
                                repo_id=getattr(shard, 'repo_id', ''),
                                tie_word_embeddings=getattr(shard, 'tie_word_embeddings', True),
                                instance_id=instance_id
                            )
                        sanitized_shards[node_id] = shard
                    
                    plan = AllocationPlan(
                        model_id=display_model_id,
                        base_model_id=base_model_id,
                        instance_id=instance_id,
                        allocations=sanitized_shards,
                        success=True,
                        reason="使用 Manager 指定的分片配置"
                    )
                else:
                    plan = await self._create_allocation_plan(
                        model_id=base_model_id,
                        n_layers=n_layers,
                        target_nodes=target_nodes,
                        strategy=strategy,
                        instance_id=instance_id
                    )
                
                if not plan.success:
                    pool_info.state = ModelState.ERROR
                    return plan
                
                # 执行分配（在各节点上加载对应的分片）
                await self._execute_allocation(plan, model_path, base_model_id=base_model_id, **kwargs)
                
                # 更新状态
                pool_info.state = ModelState.LOADED if pool_info.is_fully_covered() else ModelState.PARTIAL
                pool_info.last_accessed = time.time()
                
                # 记录历史
                self.allocation_history.append({
                    "action": "load",
                    "model_id": base_model_id,
                    "instance_id": instance_id,
                    "full_key": full_key,
                    "timestamp": time.time(),
                    "plan": {
                        "allocations": {nid: s.to_dict() for nid, s in plan.allocations.items()},
                        "success": plan.success
                    }
                })
                
                logger.info(f"[GPUPool] 模型 {full_key} 加载完成，状态: {pool_info.state.value}")
                return plan
                
            except Exception as e:
                logger.error(f"[GPUPool] 加载模型失败: {e}")
                pool_info.state = ModelState.ERROR
                raise
    
    async def _detect_model_layers(self, model_path: str, model_id: str) -> Optional[int]:
        """检测模型的层数"""
        try:
            from exo.inference.pytorch.qwen3.sharded_utils import load_config
            config = load_config(Path(model_path))
            n_layers = config.get('num_hidden_layers', 
                       config.get('n_layer',
                       config.get('layers', None)))
            if n_layers:
                logger.info(f"[GPUPool] 检测到模型层数: {n_layers}")
                return int(n_layers)
        except Exception as e:
            logger.warning(f"[GPUPool] 无法从配置文件检测层数: {e}")
        
        # 尝试从缓存或其他方式获取
        return None
    
    async def _create_allocation_plan(
        self,
        model_id: str,
        n_layers: int,
        target_nodes: Optional[List[str]] = None,
        strategy: str = "memory_weighted",
        instance_id: str = "default"
    ) -> AllocationPlan:
        """
        创建分配计划
        
        根据策略和节点资源情况，决定如何将模型的各层分配到不同节点
        
        Args:
            model_id: 基础模型ID（不含 ::instance_id）
            n_layers: 模型总层数
            target_nodes: 目标节点列表
            strategy: 分配策略
            instance_id: 实例ID
        """
        logger.info(f"[GPUPool] 创建分配计划: {model_id}, 策略: {strategy}, 实例: {instance_id}")

        # 🔍 诊断日志：打印当前 node_resources 状态
        logger.info(f"[GPUPool] [DIAG] 当前 node_resources: {list(self.node_resources.keys())}")
        logger.info(f"[GPUPool] [DIAG] target_nodes 参数: {target_nodes}")
        if hasattr(self, 'node') and self.node and hasattr(self.node, 'topology'):
            logger.info(f"[GPUPool] [DIAG] topology.nodes: {list(self.node.topology.nodes.keys())}")

        # 确定可用节点
        # 🔧 修复：支持两种 target_nodes 格式
        # 1. 字符串列表: ["node1", "node2"]
        # 2. 字典列表: [{"node_id": "node1", ...}, {"node_id": "node2", ...}]
        if target_nodes:
            # 统一提取 node_id
            if target_nodes and isinstance(target_nodes[0], dict):
                # 字典格式 → 提取 node_id
                target_node_ids = [t.get("node_id") for t in target_nodes if t.get("node_id")]
                logger.info(f"[GPUPool] [FIX] 检测到字典格式 target_nodes，已转换为 ID 列表: {target_node_ids}")
            else:
                # 字符串格式 → 直接使用
                target_node_ids = list(target_nodes)

            available_nodes = [
                (nid, res) for nid, res in self.node_resources.items()
                if nid in target_node_ids
            ]
            logger.warning(f"[GPUPool] [DIAG] 使用 target_nodes 过滤: {len(available_nodes)}/{len(self.node_resources)} 匹配")
            if not available_nodes:
                missing = set(target_node_ids) - set(self.node_resources.keys())
                logger.error(f"[GPUPool] [ERROR] target_nodes 中有 {len(missing)} 个节点不在 node_resources 中: {missing}")
        else:
            available_nodes = list(self.node_resources.items())
        
        if not available_nodes:
            return AllocationPlan(
                model_id=model_id,
                instance_id=instance_id,
                success=False,
                reason="没有可用的节点"
            )
        
        # 更新节点内存信息
        for nid, res in available_nodes:
            res.update_memory()
        
        # 根据策略生成分区
        partitions = self._generate_partitions(available_nodes, n_layers, strategy)
        
        # 转换为Shard对象（使用 base_model_id，并附加 instance_id）
        shards = map_partitions_to_shards(partitions, n_layers, model_id, instance_id)
        
        # 构建分配计划
        allocations = {}
        for shard in shards:
            for partition in partitions:
                shard_start_ratio = shard.start_layer / n_layers
                shard_end_ratio = (shard.end_layer + 1) / n_layers
                
                if (abs(partition.start - shard_start_ratio) < 0.01 and 
                    abs(partition.end - shard_end_ratio) < 0.01):
                    allocations[partition.node_id] = shard
                    break
        
        plan = AllocationPlan(
            model_id=model_id,
            instance_id=instance_id,
            allocations=allocations,
            reason=f"使用 {strategy} 策略分配"
        )
        
        logger.info(f"[GPUPool] 分配计划创建完成: {len(allocations)} 个节点参与")
        for nid, shard in allocations.items():
            logger.info(f"  - 节点 {nid}: 层 {shard.start_layer}-{shard.end_layer} ({shard.get_layer_count()} 层)")
        
        return plan
    
    def _generate_partitions(
        self,
        available_nodes: List[Tuple[str, NodeResourceInfo]],
        n_layers: int,
        strategy: str
    ) -> List[Partition]:
        """根据策略生成分区"""
        if strategy == "memory_weighted":
            return self._strategy_memory_weighted(available_nodes, n_layers)
        elif strategy == "uniform":
            return self._strategy_uniform(available_nodes, n_layers)
        elif strategy == "performance_weighted":
            return self._strategy_performance_weighted(available_nodes, n_layers)
        else:
            logger.warning(f"[GPUPool] 未知策略 {strategy}，使用默认的 memory_weighted")
            return self._strategy_memory_weighted(available_nodes, n_layers)
    
    def _strategy_memory_weighted(
        self,
        nodes: List[Tuple[str, NodeResourceInfo]],
        n_layers: int
    ) -> List[Partition]:
        """
        基于内存权重的分配策略
        
        内存越大的节点分配越多层
        """
        # 按可用内存排序
        sorted_nodes = sorted(nodes, key=lambda x: x[1].available_memory_mb, reverse=True)
        
        total_memory = sum(res.available_memory_mb for _, res in sorted_nodes)
        if total_memory == 0:
            total_memory = sum(res.device_caps.memory for _, res in sorted_nodes)
        
        partitions = []
        start = 0.0
        
        for node_id, res in sorted_nodes:
            weight = res.available_memory_mb / total_memory if total_memory > 0 else 1.0 / len(sorted_nodes)
            end = start + weight
            
            # 确保最后一个分区到达1.0
            if node_id == sorted_nodes[-1][0]:
                end = 1.0
            
            partitions.append(Partition(node_id=node_id, start=start, end=end))
            start = end
        
        return partitions
    
    def _strategy_uniform(
        self,
        nodes: List[Tuple[str, NodeResourceInfo]],
        n_layers: int
    ) -> List[Partition]:
        """
        均匀分配策略
        
        每个节点分配相同数量的层
        """
        n_nodes = len(nodes)
        partitions = []
        
        for i, (node_id, _) in enumerate(nodes):
            start = i / n_nodes
            end = (i + 1) / n_nodes
            
            if i == n_nodes - 1:
                end = 1.0
            
            partitions.append(Partition(node_id=node_id, start=start, end=end))
        
        return partitions
    
    def _strategy_performance_weighted(
        self,
        nodes: List[Tuple[str, NodeResourceInfo]],
        n_layers: int
    ) -> List[Partition]:
        """
        基于性能（FLOPS）的分配策略
        
        性能越强的节点分配越多层
        """
        sorted_nodes = sorted(
            nodes, 
            key=lambda x: x[1].device_caps.flops.fp16, 
            reverse=True
        )
        
        total_flops = sum(res.device_caps.flops.fp16 for _, res in sorted_nodes)
        
        partitions = []
        start = 0.0
        
        for node_id, res in sorted_nodes:
            weight = res.device_caps.flops.fp16 / total_flops if total_flops > 0 else 1.0 / len(sorted_nodes)
            end = start + weight
            
            if node_id == sorted_nodes[-1][0]:
                end = 1.0
            
            partitions.append(Partition(node_id=node_id, start=start, end=end))
            start = end
        
        return partitions
    
    async def _execute_allocation(
        self,
        plan: AllocationPlan,
        model_path: str,
        base_model_id: str = None,
        **kwargs
    ):
        """执行分配计划：在各节点上加载对应的分片"""
        
        effective_base_id = base_model_id or plan.effective_base_model_id
        full_key = self._make_key(effective_base_id, plan.instance_id)
        
        logger.info(f"[GPUPool] 执行分配计划: {full_key}...")
        
        tasks = []
        
        for node_id, shard in plan.allocations.items():
            if node_id == self.node.id:
                task = self._load_shard_local(shard, model_path, **kwargs)
                tasks.append(("local", node_id, task))
            else:
                task = self._load_shard_remote(node_id, shard, model_path, **kwargs)
                tasks.append(("remote", node_id, task))
        
        results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)

        has_failures = False
        for (location, node_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                has_failures = True
                logger.error(f"[GPUPool] 在 {location} 节点 {node_id} 上加载分片失败: {result}")
                plan.warnings.append(f"节点 {node_id} 加载失败: {result}")
            else:
                logger.info(f"[GPUPool] 成功在 {location} 节点 {node_id} 上加载分片")
                
                if node_id in self.node_resources:
                    # 使用完整键存储（包含实例信息）
                    self.node_resources[node_id].current_models[full_key] = ModelLoadState(
                        model_id=full_key,
                        shard=shard
                    )
                
                if full_key in self.pool_models:
                    self.pool_models[full_key].shards.append(shard)

        if has_failures:
            plan.success = False
            logger.warning(f"[GPUPool] 分配计划执行完成但有 {len([r for r in results if isinstance(r, Exception)])} 个节点失败")
        
        await self._notify_peers_topology(plan)
    
    async def _load_shard_local(self, shard: Shard, model_path: str, **kwargs):
        """在本节点加载分片 - 支持多实例（每个实例使用独立引擎）"""
        # [STAR] 获取对应实例的推理引擎
        instance_id = getattr(shard, 'instance_id', None) or "default"
        target_engine = self.node.get_engine(instance_id)

        logger.info(f"[GPUPool] 在本节点加载分片: {shard} (引擎实例: {instance_id})")

        try:
            await target_engine.load_checkpoint(shard, model_path)

            # 触发回调以广播模型加载事件
            self.node.on_model_loaded(shard)

            return True
        except Exception as e:
            logger.error(f"[GPUPool] 本节点加载分片失败 (实例: {instance_id}): {e}")
            raise
    
    async def _notify_peers_topology(self, plan: 'AllocationPlan'):
        """通知所有参与节点建立 P2P 连接"""
        import json
        
        peer_list = []
        for node_id in plan.allocations.keys():
            if node_id == self.node.id:
                continue
            
            if node_id not in self.node_resources:
                logger.warning(f"[GPUPool] 节点 {node_id} 不在资源列表中，跳过")
                continue
            
            res = self.node_resources[node_id]
            caps = res.device_caps
            peer_list.append({
                "node_id": node_id,
                "address": res.address,
                "port": res.port,
                "device_capabilities": {
                    "model": getattr(caps, "model", "unknown"),
                    "chip": getattr(caps, "chip", "unknown"),
                    "memory": getattr(caps, "memory", 0),
                    "flops": {
                        "fp32": getattr(caps.flops, "fp32", 0) if hasattr(caps, "flops") else 0,
                        "fp16": getattr(caps.flops, "fp16", 0) if hasattr(caps, "flops") else 0,
                        "int8": getattr(caps.flops, "int8", 0) if hasattr(caps, "flops") else 0
                    },
                    "memory_detail": {
                        "total": getattr(caps.memory_detail, "total", 0) if hasattr(caps, "memory_detail") and caps.memory_detail else 0,
                        "free": getattr(caps.memory_detail, "free", 0) if hasattr(caps, "memory_detail") and caps.memory_detail else 0,
                        "used": getattr(caps.memory_detail, "used", 0) if hasattr(caps, "memory_detail") and caps.memory_detail else 0
                    } if hasattr(caps, "memory_detail") and caps.memory_detail else None
                }
            })
        
        if not peer_list:
            logger.info(f"[GPUPool] 无需通知其他节点（单节点模式）")
            return
        
        logger.info(f"[GPUPool] 通知 {len(peer_list)} 个节点建立 P2P 连接: {[p['node_id'] for p in peer_list]}")
        
        topology_msg = json.dumps({
            "type": "manager_peer_topology",
            "model_id": plan.model_id,
            "peer_list": peer_list
        })
        
        for peer in self.node.peers:
            try:
                if peer.id() in plan.allocations or peer.id() == self.node.id:
                    await asyncio.wait_for(peer.send_opaque_status("", topology_msg), timeout=5.0)
                    logger.info(f"[GPUPool] 已发送 P2P 拓扑信息给 {peer.id()}")
            except Exception as e:
                logger.warning(f"[GPUPool] 发送 P2P 拓扑信息给 {peer.id()} 失败: {e}")

    async def _load_shard_remote(self, node_id: str, shard: Shard, model_path: str, **kwargs):
        """在远程节点加载分片（通过gRPC）"""
        logger.info(f"[GPUPool] 请求远程节点 {node_id} 加载分片: {shard}")
        
        try:
            # 查找目标peer
            target_peer = None
            for peer in self.node.peers:
                if peer.id() == node_id:
                    target_peer = peer
                    break
            
            if not target_peer:
                raise Exception(f"未找到节点 {node_id} 的连接")
            
            # 发送加载请求（这里需要实现具体的RPC调用）
            # 目前先记录日志，实际实现需要扩展peer_handle接口
            logger.info(f"[GPUPool] 向节点 {node_id} 发送加载请求")
            
            # TODO: 实现远程加载的具体逻辑
            # 可能需要通过 send_opaque_status 或专门的RPC方法
            
            return True
            
        except Exception as e:
            logger.error(f"[GPUPool] 远程节点 {node_id} 加载分片失败: {e}")
            raise
    
    async def unload_model(
        self, 
        model_id: str, 
        instance_id: Optional[str] = None,
        target_nodes: Optional[List[str]] = None,
        unload_all_instances: bool = False
    ) -> bool:
        """
        卸载模型（支持多实例）
        
        Args:
            model_id: 模型ID
            instance_id: 实例ID（None表示默认实例或所有实例）
            target_nodes: 目标节点列表（None表示所有节点）
            unload_all_instances: 是否卸载该模型的所有实例
            
        Returns:
            bool: 是否成功
            
        示例:
            # 卸载默认实例
            await pool.unload_model("qwen3-0.6b")
            
            # 卸载指定实例
            await pool.unload_model("qwen3-0.6b", instance_id="worker-1")
            
            # 卸载所有实例
            await pool.unload_model("qwen3-0.6b", unload_all_instances=True)
        """
        async with self._lock:
            # 确定要卸载的键列表
            if unload_all_instances:
                keys_to_unload = [k for k in self.pool_models if self._parse_key(k)[0] == model_id]
                logger.info(f"[GPUPool] 卸载模型 {model_id} 的所有实例: {keys_to_unload}")
            else:
                full_key = self._make_key(model_id, instance_id or "default")
                keys_to_unload = [full_key] if full_key in self.pool_models else []
                logger.info(f"[GPUPool] 卸载模型: {full_key}")
            
            if not keys_to_unload:
                logger.warning(f"[GPUPool] 模型 {model_id} 不在池中（instance_id={instance_id}）")
                return False
            
            total_success = True
            for full_key in keys_to_unload:
                success = await self._unload_single_instance(full_key, target_nodes)
                if not success:
                    total_success = False
            
            return total_success
    
    async def _unload_single_instance(
        self, 
        full_key: str, 
        target_nodes: Optional[List[str]] = None
    ) -> bool:
        """卸载单个模型实例"""
        model_id, instance_id = self._parse_key(full_key)
        
        if full_key not in self.pool_models:
            logger.warning(f"[GPUPool] 模型 {full_key} 不在池中")
            return False
        
        pool_info = self.pool_models[full_key]
        pool_info.state = ModelState.UNLOADING
        
        try:
            shards_to_unload = []
            if target_nodes:
                for shard in pool_info.shards[:]:
                    for node_id in target_nodes:
                        if (node_id in self.node_resources and 
                            full_key in self.node_resources[node_id].current_models):
                            load_state = self.node_resources[node_id].current_models[full_key]
                            if load_state.shard == shard:
                                shards_to_unload.append((node_id, shard))
                                break
            else:
                for shard in pool_info.shards:
                    for node_id, res in self.node_resources.items():
                        if full_key in res.current_models and res.current_models[full_key].shard == shard:
                            shards_to_unload.append((node_id, shard))
                            break
            
            unload_tasks = []
            for node_id, shard in shards_to_unload:
                if node_id == self.node.id:
                    unload_tasks.append(self._unload_shard_local(full_key, shard))
                else:
                    unload_tasks.append(self._unload_shard_remote(node_id, full_key, shard))
            
            results = await asyncio.gather(*unload_tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if r is True or not isinstance(r, Exception))
            
            for node_id, shard in shards_to_unload:
                if node_id in self.node_resources and full_key in self.node_resources[node_id].current_models:
                    del self.node_resources[node_id].current_models[full_key]
                
                if shard in pool_info.shards:
                    pool_info.shards.remove(shard)
            
            if not pool_info.shards:
                del self.pool_models[full_key]
                logger.info(f"[GPUPool] 模型 {full_key} 已完全卸载并从池中移除")
            else:
                pool_info.state = ModelState.PARTIAL if not pool_info.is_fully_covered() else ModelState.LOADED
                logger.info(f"[GPUPool] 模型 {full_key} 部分卸载，剩余 {len(pool_info.shards)} 个分片")
            
            self.allocation_history.append({
                "action": "unload",
                "model_id": model_id,
                "instance_id": instance_id,
                "full_key": full_key,
                "timestamp": time.time(),
                "success": success_count == len(shards_to_unload)
            })
            
            return success_count == len(shards_to_unload)
            
        except Exception as e:
            logger.error(f"[GPUPool] 卸载模型失败: {e}")
            pool_info.state = ModelState.ERROR
            return False
    
    async def _unload_shard_local(self, model_id: str, shard: Shard) -> bool:
        """在本节点卸载分片"""
        logger.info(f"[GPUPool] 在本节点卸载分片: {shard}")
        
        try:
            import gc
            import torch
            
            full_model_id = f"{model_id}::{shard.instance_id}" if hasattr(shard, 'instance_id') and shard.instance_id else model_id
            
            if "::" in full_model_id:
                _, instance_id = full_model_id.split("::", 1)
            else:
                instance_id = "default"
            
            if hasattr(self.node, 'inference_engines') and instance_id in self.node.inference_engines:
                engine = self.node.inference_engines[instance_id]
                
                if hasattr(engine, 'model') and engine.model is not None:
                    logger.info(f"[GPUPool] 释放引擎实例 {instance_id} 的模型对象")
                    del engine.model
                    engine.model = None
                
                if hasattr(engine, 'tokenizer') and engine.tokenizer is not None:
                    del engine.tokenizer
                    engine.tokenizer = None
                
                if hasattr(engine, 'processor') and engine.processor is not None:
                    del engine.processor
                    engine.processor = None
                
                if hasattr(engine, 'shard') and engine.shard is not None:
                    del engine.shard
                    engine.shard = None
                
                del self.node.inference_engines[instance_id]
                logger.info(f"[GPUPool] [OK] 已删除引擎实例: {instance_id}")
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"[GPUPool] GPU内存清理完成 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")
            
            if model_id in self.node.my_loaded_models:
                del self.node.my_loaded_models[model_id]
            
            if model_id in self.node.node_shards_multi.get(self.node.id, []):
                self.node.node_shards_multi[self.node.id] = [
                    s for s in self.node.node_shards_multi[self.node.id]
                    if s.model_id != model_id
                ]
            
            logger.info(f"[GPUPool] [OK] 模型 {full_model_id} 已真正卸载并释放资源")
            return True
            
        except Exception as e:
            logger.error(f"[GPUPool] 本节点卸载分片失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _unload_shard_remote(self, node_id: str, model_id: str, shard: Shard) -> bool:
        """在远程节点卸载分片"""
        logger.info(f"[GPUPool] 请求远程节点 {node_id} 卸载分片: {shard}")
        
        try:
            # TODO: 实现远程卸载的逻辑
            # 类似于远程加载，需要通过RPC调用
            
            return True
            
        except Exception as e:
            logger.error(f"[GPUPool] 远程节点 {node_id} 卸载分片失败: {e}")
            raise
    
    async def rebalance(self, model_id: str, new_nodes: Optional[List[str]] = None) -> AllocationPlan:
        """
        重新平衡模型分布（例如添加了新节点后）
        
        Args:
            model_id: 模型ID
            new_nodes: 新增的节点列表（可选）
            
        Returns:
            新的分配计划
        """
        async with self._lock:
            logger.info(f"[GPUPool] 重新平衡模型: {model_id}")
            
            if model_id not in self.pool_models:
                raise ValueError(f"模型 {model_id} 不在池中")
            
            pool_info = self.pool_models[model_id]
            old_shards = pool_info.shards.copy()
            
            try:
                pool_info.state = ModelState.MIGRATING
                
                # 先卸载旧分配
                await self.unload_model(model_id)
                
                # 如果有新节点，更新资源视图
                if new_nodes:
                    for node_id in new_nodes:
                        if node_id in self.node.topology.nodes:
                            self._update_node_resource(node_id, self.node.topology.nodes[node_id])
                
                # 重新加载（会触发新的分配计算）
                plan = await self.load_model(
                    model_id=model_id,
                    model_path=pool_info.model_path,
                    n_layers=pool_info.n_layers
                )
                
                logger.info(f"[GPUPool] 模型 {model_id} 重平衡完成")
                return plan
                
            except Exception as e:
                logger.error(f"[GPUBalancer] 重平衡失败: {e}")
                pool_info.state = ModelState.ERROR
                
                # 尝试恢复旧的分配
                logger.warning(f"[GPUPool] 尝试恢复旧分配...")
                pool_info.shards = old_shards
                pool_info.state = ModelState.LOADED if pool_info.is_fully_covered() else ModelState.PARTIAL
                
                raise
    
    def get_model_instances(self, model_id: str) -> List[Dict[str, Any]]:
        """
        获取模型的所有实例信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            实例信息列表
        """
        instances = []
        for key, info in self.pool_models.items():
            mid, iid = self._parse_key(key)
            if mid == model_id:
                instances.append({
                    "instance_id": iid,
                    "full_key": key,
                    "state": info.state.value,
                    "n_layers": info.n_layers,
                    "shards_count": len(info.shards),
                    "is_fully_covered": info.is_fully_covered(),
                    "created_at": info.created_at,
                    "last_accessed": info.last_accessed
                })
        
        return sorted(instances, key=lambda x: x["created_at"])
    
    def list_all_instances(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        列出所有模型的所有实例
        
        Returns:
            {model_id: [instances]}
        """
        result = {}
        for key, info in self.pool_models.items():
            mid, iid = self._parse_key(key)
            if mid not in result:
                result[mid] = []
            
            result[mid].append({
                "instance_id": iid,
                "state": info.state.value,
                "shards_count": len(info.shards),
                "is_fully_covered": info.is_fully_covered()
            })
        
        return result
    
    def get_instance_count(self, model_id: str) -> int:
        """获取模型的实例数量"""
        return sum(1 for k in self.pool_models if self._parse_key(k)[0] == model_id)
    
    def get_pool_status(self) -> Dict:
        """
        获取池子的详细状态
        
        Returns:
            包含节点、模型、内存使用等信息的字典
        """
        total_memory = sum(res.device_caps.memory for res in self.node_resources.values())
        total_available = sum(res.available_memory_mb for res in self.node_resources.values())
        total_used = sum(res.used_memory_mb for res in self.node_resources.values())
        
        models_summary = {}
        for model_id, info in self.pool_models.items():
            covered, missing, all_layers = info.get_coverage()
            models_summary[model_id] = {
                "state": info.state.value,
                "total_layers": info.n_layers,
                "covered_layers": len(covered),
                "missing_layers": len(missing),
                "shard_count": len(info.shards),
                "nodes": list(set(
                    nid for nid, res in self.node_resources.items()
                    if model_id in res.current_models
                )),
                "last_accessed": info.last_accessed
            }
        
        nodes_summary = {}
        for node_id, res in self.node_resources.items():
            nodes_summary[node_id] = {
                "device": res.device_caps.chip,
                "total_memory_mb": res.device_caps.memory,
                "available_memory_mb": res.available_memory_mb,
                "used_memory_mb": res.used_memory_mb,
                "loaded_models": list(res.current_models.keys()),
                "flops_fp16": res.device_caps.flops.fp16
            }
        
        return {
            "pool_name": "exo_gpu_pool",
            "total_nodes": len(self.node_resources),
            "total_models": len(self.pool_models),
            "memory": {
                "total_mb": total_memory,
                "available_mb": total_available,
                "used_mb": total_used,
                "utilization_percent": round(total_used / total_memory * 100, 2) if total_memory > 0 else 0
            },
            "models": models_summary,
            "nodes": nodes_summary,
            "allocation_history_count": len(self.allocation_history)
        }
    
    def get_model_details(self, model_id: str) -> Optional[Dict]:
        """获取特定模型的详细信息"""
        if model_id not in self.pool_models:
            return None
        
        info = self.pool_models[model_id]
        covered, missing, all_layers = info.get_coverage()
        
        shard_details = []
        for shard in info.shards:
            # 找到承载这个分片的节点
            node_id = None
            for nid, res in self.node_resources.items():
                if model_id in res.current_models and res.current_models[model_id].shard == shard:
                    node_id = nid
                    break
            
            shard_details.append({
                "start_layer": shard.start_layer,
                "end_layer": shard.end_layer,
                "layer_count": shard.get_layer_count(),
                "is_first": shard.is_first_layer(),
                "is_last": shard.is_last_layer(),
                "node_id": node_id
            })
        
        return {
            "model_id": model_id,
            "state": info.state.value,
            "model_path": info.model_path,
            "n_layers": info.n_layers,
            "coverage": {
                "covered": sorted(list(covered)),
                "missing": sorted(list(missing)),
                "percent": round(len(covered) / len(all_layers) * 100, 2) if all_layers else 0
            },
            "shards": shard_details,
            "created_at": info.created_at,
            "last_accessed": info.last_accessed
        }
    
    def list_available_models(self) -> List[str]:
        """列出池中所有可用的模型"""
        return list(self.pool_models.keys())
    
    def list_node_models(self, node_id: str) -> List[str]:
        """列出指定节点上加载的所有模型"""
        if node_id not in self.node_resources:
            return []
        return list(self.node_resources[node_id].current_models.keys())
    
    async def add_node(self, node_id: str, caps: DeviceCapabilities):
        """添加新节点到池中"""
        logger.info(f"[GPUPool] 添加新节点: {node_id}")
        self._update_node_resource(node_id, caps)
    
    async def remove_node(self, node_id: str):
        """从池中移除节点"""
        logger.info(f"[GPUPool] 移除节点: {node_id}")
        
        if node_id not in self.node_resources:
            return
        
        res = self.node_resources[node_id]
        
        # 迁移该节点上的模型（如果有）
        if res.current_models:
            logger.warning(f"[GPUPool] 节点 {node_id} 上有 {len(res.current_models)} 个模型，需要迁移")
            # TODO: 实现自动迁移逻辑
        
        del self.node_resources[node_id]
    
    def get_allocation_plan_preview(
        self,
        model_id: str,
        n_layers: int,
        target_nodes: Optional[List[str]] = None,
        strategy: str = "memory_weighted"
    ) -> Dict:
        """
        预览分配计划（不实际执行）
        
        用于在实际加载前查看分配方案
        """
        if target_nodes:
            available_nodes = [
                (nid, res) for nid, res in self.node_resources.items()
                if nid in target_nodes
            ]
        else:
            available_nodes = list(self.node_resources.items())
        
        if not available_nodes:
            return {"error": "没有可用的节点"}
        
        # 更新内存信息
        for nid, res in available_nodes:
            res.update_memory()
        
        # 生成分区
        partitions = self._generate_partitions(available_nodes, n_layers, strategy)
        shards = map_partitions_to_shards(partitions, n_layers, model_id)
        
        preview = {
            "model_id": model_id,
            "n_layers": n_layers,
            "strategy": strategy,
            "nodes_used": len(partitions),
            "allocations": []
        }
        
        for shard in shards:
            for partition in partitions:
                if (abs(partition.start - shard.start_layer / n_layers) < 0.01 and
                    abs(partition.end - (shard.end_layer + 1) / n_layers) < 0.01):
                    
                    node_res = self.node_resources.get(partition.node_id)
                    preview["allocations"].append({
                        "node_id": partition.node_id,
                        "device": node_res.device_caps.chip if node_res else "unknown",
                        "layers": f"{shard.start_layer}-{shard.end_layer}",
                        "layer_count": shard.get_layer_count(),
                        "ratio": f"{partition.start:.2%}-{partition.end:.2%}",
                        "available_memory_mb": node_res.available_memory_mb if node_res else 0
                    })
                    break
        
        return preview
    
    def print_pool_status(self):
        """打印池状态的易读格式"""
        status = self.get_pool_status()
        
        print("\n" + "="*60)
        print("🖥️  EXO GPU 显存池状态")
        print("="*60)
        print(f"\n📊 池概览:")
        print(f"   总节点数: {status['total_nodes']}")
        print(f"   已加载模型数: {status['total_models']}")
        print(f"\n💾 内存使用:")
        print(f"   总计: {status['memory']['total_mb']/1024:.1f} GB")
        print(f"   可用: {status['memory']['available_mb']/1024:.1f} GB ({status['memory']['utilization_percent']}% 使用)")
        
        if status['models']:
            print(f"\n[BOX] 模型详情:")
            for model_id, info in status['models'].items():
                coverage_icon = "[OK]" if info['covered_layers'] == info['total_layers'] else "[WARN]"
                print(f"   {coverage_icon} {model_id}")
                print(f"      状态: {info['state']}")
                print(f"      层覆盖: {info['covered_layers']}/{info['total_layers']} ({info['missing_layers']} 缺失)")
                print(f"      分布在 {len(info['nodes'])} 个节点: {info['nodes']}")
        
        if status['nodes']:
            print(f"\n[TOOL] 节点详情:")
            for node_id, info in status['nodes'].items():
                print(f"   [PIN] {node_id}")
                print(f"      设备: {info['device']}")
                print(f"      内存: {info['available_memory_mb']/1024:.1f} GB 可用 / {info['total_memory_mb']/1024:.1f} GB 总计")
                print(f"      已加载: {info['loaded_models'] or '无'}")
        
        print("\n" + "="*60 + "\n")


__all__ = ['GPUPoolManager', 'PoolModelInfo', 'NodeResourceInfo', 'AllocationPlan', 'ModelState']
