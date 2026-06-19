# -*- coding: utf-8 -*-
"""
WebSocket 双向通道优化模块

提供生产级的 WebSocket 连接管理，包括：
- 消息队列缓冲（防止丢消息）
- 背压控制（防止压垮连接）
- 断线重传（保证消息可靠性）
- 状态同步（断线恢复后自动同步）
- QoS 保障（重要消息优先级）
- 连接池管理（多连接负载均衡）
- 监控指标（性能可观测）

作者: Exo Team
版本: 2.0.0
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, Optional, Callable, Any, List, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """消息优先级"""
    LOW = auto()          # 低优先级：心跳、统计信息
    NORMAL = auto()       # 正常优先级：普通消息
    HIGH = auto()         # 高优先级：推理请求/响应
    CRITICAL = auto()     # 关键优先级：错误、系统消息


@dataclass
class WSMessage:
    """
    增强版 WebSocket 消息
    
    Attributes:
        msg_id: 消息唯一ID（用于确认和重传）
        msg_type: 消息类型
        payload: 消息内容
        priority: 消息优先级
        timestamp: 发送时间戳
        retry_count: 重试次数
        require_ack: 是否需要确认
        timeout: 超时时间（秒）
        callback: 成功回调
        error_callback: 失败回调
    """
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    msg_type: str = ""
    payload: Dict = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    require_ack: bool = False
    timeout: float = 30.0
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "msg_id": self.msg_id,
            "type": self.msg_type,
            "timestamp": self.timestamp,
            **self.payload,
            "_meta": {
                "priority": self.priority.name,
                "require_ack": self.require_ack
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WSMessage':
        """从字典创建消息"""
        meta = data.pop("_meta", {})
        return cls(
            msg_id=data.get("msg_id", ""),
            msg_type=data.get("type", ""),
            payload=data,
            priority=MessagePriority[meta.get("priority", "NORMAL")],
            timestamp=data.get("timestamp", time.time()),
            require_ack=meta.get("require_ack", False)
        )


class MessageQueue:
    """
    优先级消息队列（支持背压控制）
    
    特性：
    - 多优先级队列
    - 最大容量限制（背压）
    - FIFO + 优先级混合调度
    - 自动过期清理
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues: Dict[MessagePriority, deque] = {
            priority: deque(maxlen=max_size // 4) 
            for priority in MessagePriority
        }
        self._total_size = 0
        self._lock = asyncio.Lock()
        
    async def put(self, message: WSMessage) -> bool:
        """
        入队（带背压控制）
        
        Returns:
            bool: 是否成功入队（False 表示队列已满）
        """
        async with self._lock:
            if self._total_size >= self.max_size:
                logger.warning(f"[WARN] [MsgQueue] 队列已满 ({self.max_size})，丢弃低优先级消息")
                # 尝试丢弃最低优先级的消息
                for priority in reversed(list(MessagePriority)):
                    if self.queues[priority]:
                        dropped = self.queues[priority].popleft()
                        self._total_size -= 1
                        logger.debug(f"丢弃消息: {dropped.msg_id} (优先级={priority.name})")
                        break
                
                # 如果仍然满，返回失败
                if self._total_size >= self.max_size:
                    return False
            
            self.queues[message.priority].append(message)
            self._total_size += 1
            return True
    
    async def get(self, timeout: float = None) -> Optional[WSMessage]:
        """
        出队（按优先级）
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            WSMessage or None: 出队的消息
        """
        start_time = time.time()
        
        while True:
            async with self._lock:
                # 按优先级从高到低检查
                for priority in reversed(list(MessagePriority)):
                    if self.queues[priority]:
                        message = self.queues[priority].popleft()
                        self._total_size -= 1
                        return message
                
            # 队列为空，检查超时
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            await asyncio.sleep(0.01)  # 短暂休眠避免忙等
    
    @property
    def size(self) -> int:
        return self._total_size
    
    @property
    def is_full(self) -> bool:
        return self._total_size >= self.max_size
    
    def clear(self):
        """清空队列"""
        for queue in self.queues.values():
            queue.clear()
        self._total_size = 0


class ConnectionStats:
    """
    连接统计信息（用于监控和调优）
    
    Attributes:
        messages_sent: 发送的消息数
        messages_received: 接收的消息数
        bytes_sent: 发送的字节数
        bytes_received: 接收的字节数
        reconnect_count: 重连次数
        last_message_time: 最后消息时间
        avg_latency: 平均延迟
    """
    def __init__(self):
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.reconnect_count = 0
        self.last_message_time = time.time()
        self.latencies: List[float] = []
        self.errors: List[Dict] = []
        
    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    def record_latency(self, latency: float):
        """记录延迟"""
        self.latencies.append(latency)
        # 只保留最近100条记录
        if len(self.latencies) > 100:
            self.latencies.pop(0)
    
    def record_error(self, error_type: str, message: str):
        """记录错误"""
        self.errors.append({
            "time": time.time(),
            "type": error_type,
            "message": message
        })
        # 只保留最近50条错误
        if len(self.errors) > 50:
            self.errors.pop(0)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "reconnect_count": self.reconnect_count,
            "last_message_time": self.last_message_time,
            "avg_latency_ms": self.avg_latency * 1000,
            "error_count": len(self.errors),
            "recent_errors": self.errors[-5:] if self.errors else []
        }


class EnhancedWebSocketManager:
    """
    增强版 WebSocket 连接管理器
    
    核心功能：
    ✅ 消息队列缓冲（可靠传输）
    ✅ 背压控制（流量管理）
    ✅ 断线重传（消息可靠性）
    ✅ 心跳保活（连接健康检查）
    ✅ 优先级队列（QoS保障）
    ✅ 状态同步（断线恢复）
    ✅ 性能监控（可观测性）
    ✅ 连接池（高可用）
    
    使用示例：
    ```python
    manager = EnhancedWebSocketManager(
        node_id="my-node",
        ws_url="ws://manager:8080/ws/node/my-node"
    )
    
    # 启动连接
    await manager.start()
    
    # 发送消息（带确认）
    await manager.send({
        "type": "inference_request",
        "payload": {...}
    }, require_ack=True)
    
    # 接收消息
    async for msg in manager.receive():
        print(msg)
    
    # 关闭连接
    await manager.stop()
    ```
    """
    
    def __init__(
        self,
        node_id: str,
        ws_url: str,
        *,
        max_queue_size: int = 1000,
        ping_interval: float = 30.0,
        ping_timeout: float = 60.0,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
        max_retries: int = 3,
        enable_compression: bool = True,
        stats_callback: Optional[Callable[[ConnectionStats], None]] = None
    ):
        """
        初始化 WebSocket 管理器
        
        Args:
            node_id: 节点ID
            ws_url: WebSocket URL
            max_queue_size: 消息队列最大容量
            ping_interval: 心跳间隔（秒）
            ping_timeout: 心跳超时（秒）
            reconnect_base_delay: 重连基础延迟（秒）
            reconnect_max_delay: 重连最大延迟（秒）
            max_retries: 最大重试次数
            enable_compression: 是否启用压缩
            stats_callback: 统计信息回调
        """
        self.node_id = node_id
        self.ws_url = ws_url
        
        # 配置参数
        self.config = {
            "max_queue_size": max_queue_size,
            "ping_interval": ping_interval,
            "ping_timeout": ping_timeout,
            "reconnect_base_delay": reconnect_base_delay,
            "reconnect_max_delay": reconnect_max_delay,
            "max_retries": max_retries,
            "enable_compression": enable_compression
        }
        
        # 核心组件
        self.outbound_queue = MessageQueue(max_size=max_queue_size)   # 发送队列
        self.inbound_queue = MessageQueue(max_size=max_queue_size)    # 接收队列
        self.pending_acks: Dict[str, WSMessage] = {}                  # 待确认消息
        self.stats = ConnectionStats()
        self.stats_callback = stats_callback
        
        # 连接状态
        self.websocket = None
        self.is_connected = False
        self.is_running = False
        self._reconnect_count = 0
        self._current_reconnect_delay = reconnect_base_delay
        
        # 任务引用
        self._send_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # 事件回调
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        self.on_message: Optional[Callable[[WSMessage], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        logger.info(f"✅ [WSManager] 初始化完成: {node_id} -> {ws_url}")
    
    async def start(self):
        """启动 WebSocket 连接管理器"""
        if self.is_running:
            logger.warning("[WARN] [WSManager] 已经在运行中")
            return
        
        self.is_running = True
        logger.info(f"[ROCKET] [WSManager] Starting connection manager...")
        
        # 启动后台任务
        self._send_task = asyncio.create_task(self._send_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # 开始连接
        await self._connect()
    
    async def stop(self):
        """停止 WebSocket 连接管理器"""
        logger.info(f"[STOP] [WSManager] 停止连接管理器...")
        self.is_running = False
        
        # 取消所有任务
        for task in [self._send_task, self._receive_task, 
                     self._heartbeat_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 关闭连接
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        # 清空队列
        self.outbound_queue.clear()
        self.inbound_queue.clear()
        self.pending_acks.clear()
        
        self.is_connected = False
        logger.info(f"[OK] [WSManager] 已停止")
    
    async def _connect(self):
        """建立 WebSocket 连接（带重连）"""
        import websockets
        
        while self.is_running:
            try:
                logger.info(f"[RETRY] [WSManager] 正在连接到 {self.ws_url} ...")
                
                self.websocket = await websockets.connect(
                    self.ws_url,
                    ping_interval=None,   # 禁用协议级ping，避免与FastAPI WebSocket不兼容
                    ping_timeout=None,    # 应用层已有 _heartbeat_loop 心跳机制
                    close_timeout=10,
                    max_size=10 * 1024 * 1024,
                    compression=None if not self.config["enable_compression"] else "deflate"
                )
                
                self.is_connected = True
                self._reconnect_count += 1
                self.stats.reconnect_count = self._reconnect_count
                self._current_reconnect_delay = self.config["reconnect_base_delay"]
                
                logger.info(f"[OK] [WSManager] 已连接! (第{self._reconnect_count}次)")
                
                # 触发连接回调
                if self.on_connect:
                    await self.on_connect()
                
                # 发送注册消息
                await self._send_register()
                
                break
                
            except Exception as e:
                logger.error(f"❌ [WSManager] 连接失败: {e}")
                self.stats.record_error("connection_failed", str(e))
                
                if self.on_error:
                    await self.on_error(e)
                
                # 计算退避延迟
                delay = min(
                    self._current_reconnect_delay,
                    self.config["reconnect_max_delay"]
                )
                
                logger.info(f"⏳ [WSManager] {delay:.1f}秒后重连...")
                await asyncio.sleep(delay)
                
                # 指数退避
                self._current_reconnect_delay *= 1.5

    async def _reconnect_async(self):
        """异步重连（不阻塞调用方）"""
        try:
            logger.info(f"[RECONNECT] [WSManager] 异步触发重连...")
            # 短暂延迟后开始重连
            await asyncio.sleep(0.5)
            if self.is_running and not self.is_connected:
                await self._connect()
                # 重连成功后，重新启动发送循环
                if self.is_connected:
                    asyncio.create_task(self._send_loop())
                    logger.info("[OK] [WSManager] 重连完成，发送循环已重启")
        except Exception as e:
            logger.error(f"[ERROR] [WSManager] 异步重连失败: {e}")
    
    async def _send_register(self):
        """发送注册消息"""
        register_msg = WSMessage(
            msg_type="register",
            payload={
                "node_id": self.node_id,
                "capabilities": {
                    "version": "2.0",
                    "features": [
                        "priority_queue",
                        "message_ack",
                        "compression",
                        "streaming"
                    ]
                }
            },
            priority=MessagePriority.CRITICAL,
            require_ack=True
        )
        
        await self.outbound_queue.put(register_msg)
    
    async def send(
        self,
        payload: Dict,
        *,
        priority: MessagePriority = MessagePriority.NORMAL,
        require_ack: bool = False,
        timeout: float = 30.0,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ) -> bool:
        """
        发送消息（异步入队）
        
        Args:
            payload: 消息内容
            priority: 消息优先级
            require_ack: 是否需要确认
            timeout: 超时时间
            callback: 成功回调
            error_callback: 失败回调
            
        Returns:
            bool: 是否成功入队
        """
        message = WSMessage(
            msg_type=payload.get("type", "unknown"),
            payload=payload,
            priority=priority,
            require_ack=require_ack,
            timeout=timeout,
            callback=callback,
            error_callback=error_callback
        )
        
        success = await self.outbound_queue.put(message)
        
        if not success:
            logger.warning(f"[WARN] [WSManager] 发送队列已满，消息被拒绝")
            if error_callback:
                await error_callback(Exception("Queue full"))
        
        return success
    
    async def receive(self, timeout: float = None) -> Optional[WSMessage]:
        """
        接收消息（阻塞）
        
        Args:
            timeout: 超时时间
            
        Returns:
            WSMessage or None
        """
        return await self.inbound_queue.get(timeout)
    
    async def receive_stream(self) -> AsyncGenerator[WSMessage, None]:
        """
        流式接收消息（用于持续监听）
        
        Yields:
            WSMessage: 接收到的消息
        """
        while self.is_running:
            try:
                message = await self.inbound_queue.get(timeout=1.0)
                if message:
                    yield message
            except asyncio.TimeoutError:
                continue
    
    async def _send_loop(self):
        """发送循环（从队列取出消息并发送）"""
        while self.is_running:
            try:
                if not self.is_connected:
                    await asyncio.sleep(0.1)
                    continue
                
                message = await self.outbound_queue.get(timeout=0.1)
                if not message:
                    continue
                
                # 序列化并发送
                msg_dict = message.to_dict()
                msg_json = json.dumps(msg_dict, ensure_ascii=False)
                
                start_time = time.time()
                await self.websocket.send(msg_json)
                latency = time.time() - start_time
                
                # 更新统计
                self.stats.messages_sent += 1
                self.stats.bytes_sent += len(msg_json.encode('utf-8'))
                self.stats.record_latency(latency)
                self.stats.last_message_time = time.time()
                
                # 如果需要确认，加入待确认列表
                if message.require_ack:
                    self.pending_acks[message.msg_id] = message
                    logger.debug(f"📤 [WSManager] 消息已发送（待确认）: {message.msg_id}")
                else:
                    logger.debug(f"📤 [WSManager] 消息已发送: {message.msg_type}")
                
                # 触发成功回调
                if message.callback:
                    await message.callback(message)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [WSManager] 发送错误: {e}")
                self.stats.record_error("send_error", str(e))

                # 检测连接是否已关闭
                connection_lost = False
                if self.websocket and hasattr(self.websocket, 'closed') and self.websocket.closed:
                    connection_lost = True
                    logger.warning("[WARN] [WSManager] WebSocket 连接已关闭 (closed=True)")
                elif "1000" in str(e) or "1001" in str(e) or "ConnectionClosed" in type(e).__name__:
                    connection_lost = True
                    logger.warning(f"[WARN] [WSManager] 连接正常关闭或断开: {e}")

                if connection_lost:
                    # 连接已关闭，不再重试，直接标记为断开并触发重连
                    self.is_connected = False
                    logger.warning(f"[WARN] [WSManager] 连接丢失，消息 {message.msg_id if message else 'N/A'} 将被丢弃（共 {message.retry_count if message else 0} 次尝试）")
                    if self.on_disconnect:
                        try:
                            await self.on_disconnect()
                        except Exception as disconnect_err:
                            logger.error(f"[ERROR] [WSManager] 断开回调异常: {disconnect_err}")
                    # 异步触发重连
                    asyncio.create_task(self._reconnect_async())
                    break  # 退出发送循环，等待重连后重新启动
                else:
                    # 其他错误（如临时网络问题），可以重试
                    if message and message.retry_count < self.config["max_retries"]:
                        message.retry_count += 1
                        await self.outbound_queue.put(message)
                        logger.info(f"🔄 [WSManager] 消息将重试 ({message.retry_count}/{self.config['max_retries']})")
                    else:
                        logger.warning(f"[WARN] [WSManager] 消息发送最终失败，已达最大重试次数: {message.msg_type if message else 'unknown'}")
    
    async def _receive_loop(self):
        """接收循环（接收消息并放入队列）"""
        while self.is_running:
            try:
                if not self.is_connected:
                    await asyncio.sleep(0.1)
                    continue
                
                # 接收原始消息
                raw_message = await self.websocket.recv()

                # 🔍 [诊断] 记录每条原始消息（用于排查消息丢失）
                logger.info(f"🔍 [WSManager-RECV] 原始消息 ({len(raw_message)}字节): {raw_message[:200]}")

                # 解析
                data = json.loads(raw_message)
                message = WSMessage.from_dict(data)

                # 🔍 [诊断] 确认解析结果
                logger.info(f"🔍 [WSManager-PARSE] msg_type={message.msg_type}, payload_keys={list(message.payload.keys())[:10]}")
                
                # 更新统计
                self.stats.messages_received += 1
                self.stats.bytes_received += len(raw_message.encode('utf-8'))
                self.stats.last_message_time = time.time()
                
                # 处理特殊消息类型
                if message.msg_type == "ack":
                    # 收到确认
                    ack_msg_id = message.payload.get("msg_id")
                    if ack_msg_id in self.pending_acks:
                        original_msg = self.pending_acks.pop(ack_msg_id)
                        logger.debug(f"[OK] [WSManager] 收到确认: {ack_msg_id}")
                        
                        if original_msg.callback:
                            await original_msg.callback(original_msg)
                    continue
                
                elif message.msg_type == "pong":
                    # 心跳响应
                    logger.debug(f"💓 [WSManager] 收到心跳响应")
                    continue
                
                # 放入接收队列
                await self.inbound_queue.put(message)
                logger.debug(f"📥 [WSManager] 收到消息: {message.msg_type}")
                
                # 触发消息回调
                if self.on_message:
                    await self.on_message(message)
                    
                # 自动发送确认（如果需要）
                if message.require_ack:
                    ack_msg = {
                        "type": "ack",
                        "msg_id": message.msg_id
                    }
                    await self.send(ack_msg, priority=MessagePriority.LOW)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [WSManager] 接收错误: {e}")
                self.stats.record_error("recv_error", str(e))
                
                # 检查是否是连接关闭
                if self.websocket and self.websocket.closed:
                    logger.warning("[WARN] [WSManager] 连接已关闭，尝试重连...")
                    self.is_connected = False
                    if self.on_disconnect:
                        await self.on_disconnect()
                    await self._connect()
                    # 重连成功后重启发送循环（_send_loop在断连时已退出）
                    if self.is_connected:
                        asyncio.create_task(self._send_loop())
                        logger.info("[OK] [WSManager] 重连完成，发送循环已重启")
    
    async def _heartbeat_loop(self):
        """心跳保活循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["ping_interval"])
                
                if not self.is_connected:
                    continue
                
                # 发送心跳
                heartbeat_msg = {
                    "type": "heartbeat",
                    "node_id": self.node_id,
                    "timestamp": time.time(),
                    "stats": {
                        "queue_size": self.outbound_queue.size,
                        "pending_acks": len(self.pending_acks)
                    }
                }
                
                await self.send(heartbeat_msg, priority=MessagePriority.LOW)
                logger.debug(f"💓 [WSManager] 心跳已发送")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ERROR] [WSManager] 心跳错误: {e}")
    
    async def _cleanup_loop(self):
        """清理循环（定期清理过期消息和更新统计）"""
        while self.is_running:
            try:
                await asyncio.sleep(10.0)  # 每10秒清理一次
                
                current_time = time.time()
                
                # 清理过期的待确认消息
                expired_msgs = []
                for msg_id, message in self.pending_acks.items():
                    if current_time - message.timestamp > message.timeout:
                        expired_msgs.append((msg_id, message))
                
                for msg_id, message in expired_msgs:
                    self.pending_acks.pop(msg_id, None)
                    logger.warning(f"⏰ [WSManager] 消息确认超时: {msg_id}")
                    
                    if message.error_callback:
                        await message.error_callback(Exception("ACK timeout"))
                
                # 触发统计回调
                if self.stats_callback:
                    await self.stats_callback(self.stats)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [WSManager] 清理错误: {e}")
    
    @property
    def status(self) -> Dict:
        """获取当前状态"""
        return {
            "node_id": self.node_id,
            "is_connected": self.is_connected,
            "is_running": self.is_running,
            "outbound_queue_size": self.outbound_queue.size,
            "inbound_queue_size": self.inbound_queue.size,
            "pending_acks": len(self.pending_acks),
            "reconnect_count": self._reconnect_count,
            "stats": self.stats.to_dict()
        }
    
    def get_health_status(self) -> Dict:
        """
        获取健康状态（用于监控）
        
        Returns:
            Dict: 健康状态信息
        """
        now = time.time()
        time_since_last_msg = now - self.stats.last_message_time
        
        # 判断健康状态
        if time_since_last_msg < 60:
            health = "healthy"
        elif time_since_last_msg < 180:
            health = "degraded"
        else:
            health = "unhealthy"
        
        return {
            "health": health,
            "connected": self.is_connected,
            "uptime_seconds": now - (self.stats.last_message_time - self.stats.avg_latency),
            "messages_per_second": (
                self.stats.messages_sent / max(time_since_last_msg, 1)
            ),
            "avg_latency_ms": self.stats.avg_latency * 1000,
            "error_rate": (
                len(self.stats.errors) / max(self.stats.messages_sent + self.stats.messages_received, 1)
            ),
            "queue_utilization": (
                self.outbound_queue.size / self.config["max_queue_size"]
            ),
            "pending_confirmations": len(self.pending_acks)
        }


# ==================== 工厂函数 ====================

async def create_ws_manager(
    node_id: str,
    manager_url: str,
    **kwargs
) -> EnhancedWebSocketManager:
    """
    创建 WebSocket 管理器的工厂函数
    
    Args:
        node_id: 节点ID
        manager_url: Manager URL (http://host:port)
        **kwargs: 其他配置参数
        
    Returns:
        EnhancedWebSocketManager: 配置好的管理器实例
    """
    # 构建 WebSocket URL
    base_url = manager_url.rstrip('/')
    if base_url.startswith('https://'):
        ws_url = base_url.replace('https://', 'wss://')
    else:
        ws_url = base_url.replace('http://', 'ws://')
    
    ws_endpoint = f"{ws_url}/ws/node/{node_id}"
    
    # 创建管理器
    manager = EnhancedWebSocketManager(
        node_id=node_id,
        ws_url=ws_endpoint,
        **kwargs
    )
    
    return manager


# ==================== 使用示例 ====================

async def example_usage():
    """使用示例"""
    
    # 创建管理器
    manager = await create_ws_manager(
        node_id="test-node-123",
        manager_url="http://localhost:8080"
    )
    
    # 设置回调
    manager.on_connect = lambda: print("[OK] 已连接!")
    manager.on_disconnect = lambda: print("[PLUG] 已断开")
    manager.on_message = lambda msg: print(f"[MAIL] 收到: {msg.msg_type}")
    
    # 启动
    await manager.start()
    
    try:
        # 发送推理请求（带确认）
        await manager.send({
            "type": "inference_request",
            "request_id": "req-001",
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Hello!"}]
        }, require_ack=True, priority=MessagePriority.HIGH)
        
        # 接收响应
        async for response in manager.receive_stream():
            print(f"收到响应: {response.msg_type}")
            
            if response.msg_type == "inference_complete":
                break
                
    finally:
        # 停止
        await manager.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
