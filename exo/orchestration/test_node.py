import unittest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import pytest
import os

from .node import Node
from exo.networking.peer_handle import PeerHandle
from exo.download.shard_download import NoopShardDownloader, ShardDownloader
from exo.inference.shard import Shard
from exo.inference.inference_engine import get_inference_engine

class TestNode(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.mock_inference_engine = AsyncMock()
    self.mock_server = AsyncMock()
    self.mock_server.start = AsyncMock()
    self.mock_server.stop = AsyncMock()
    self.mock_discovery = AsyncMock()
    self.mock_discovery.start = AsyncMock()
    self.mock_discovery.stop = AsyncMock()
    mock_peer1 = Mock(spec=PeerHandle)
    mock_peer1.id.return_value = "peer1"
    mock_peer2 = Mock(spec=PeerHandle)
    mock_peer2.id.return_value = "peer2"
    self.mock_discovery.discover_peers = AsyncMock(return_value=[mock_peer1, mock_peer2])

    self.node = Node("test_node", self.mock_server, self.mock_inference_engine, "localhost", 50051, self.mock_discovery, NoopShardDownloader())

  async def asyncSetUp(self):
    await self.node.start()

  async def asyncTearDown(self):
    await self.node.stop()

  async def test_node_initialization(self):
    self.assertEqual(self.node.node_id, "test_node")
    self.assertEqual(self.node.host, "localhost")
    self.assertEqual(self.node.port, 50051)

  async def test_node_start(self):
    self.mock_server.start.assert_called_once_with("localhost", 50051)

  async def test_node_stop(self):
    await self.node.stop()
    self.mock_server.stop.assert_called_once()

  async def test_discover_and_connect_to_peers(self):
    await self.node.discover_and_connect_to_peers()
    self.assertEqual(len(self.node.peers), 2)
    self.assertIn("peer1", map(lambda p: p.id(), self.node.peers))
    self.assertIn("peer2", map(lambda p: p.id(), self.node.peers))

  async def test_process_tensor_calls_inference_engine(self):
    mock_peer = Mock()
    self.node.peers = [mock_peer]

    input_tensor = np.array([69, 1, 2])
    await self.node.process_tensor(input_tensor, None)

    self.node.inference_engine.process_shard.assert_called_once_with(input_tensor)

  @pytest.mark.asyncio
  async def test_node_capabilities(self):
    node = Node("capability_test_node", self.mock_server, self.mock_inference_engine,
                "localhost", 50053, self.mock_discovery, NoopShardDownloader())
    await node.start()
    caps = await node.get_device_capabilities()
    self.assertIsNotNone(caps)
    self.assertNotEqual(caps.model, "")
    await node.stop()



class TestNodeWithPyTorchEngine(unittest.IsolatedAsyncioTestCase):
  """测试使用PyTorch引擎的Node功能"""
  
  def setUp(self):
    # 使用环境变量控制是否运行PyTorch测试
    self.run_pytorch_tests = os.getenv("RUN_PYTORCH", "0") == "1"
    
    if self.run_pytorch_tests:
      # 创建实际的PyTorch引擎实例
      from exo.inference.pytorch.pytorch_inference_engine import PyTorchDynamicShardInferenceEngine
      self.shard_downloader = NoopShardDownloader()
      # 使用模拟的ShardDownloader确保不会实际下载模型
      with patch('exo.download.shard_download.NoopShardDownloader.ensure_shard', 
                 return_value="/mock/model/path"):
        self.inference_engine = PyTorchDynamicShardInferenceEngine(self.shard_downloader)
      
      # Mock模型加载逻辑
      self.inference_engine.model = Mock()
      self.inference_engine.tokenizer = Mock()
      self.inference_engine.tokenizer.encode = Mock(return_value=[1, 2, 3])
      self.inference_engine.tokenizer.decode = Mock(return_value="test output")
      self.inference_engine.infer_tensor = AsyncMock(return_value=(np.array([[1, 2, 3]]), None))
      self.inference_engine.sample = AsyncMock(return_value=np.array([[42]]))
    else:
      # 如果不运行PyTorch测试，使用mock
      self.inference_engine = AsyncMock()
      self.shard_downloader = NoopShardDownloader()
    
    # 创建其他mock对象
    self.mock_server = AsyncMock()
    self.mock_server.start = AsyncMock()
    self.mock_server.stop = AsyncMock()
    self.mock_discovery = AsyncMock()
    self.mock_discovery.start = AsyncMock()
    self.mock_discovery.stop = AsyncMock()
    
    # 创建Node实例
    self.node = Node("test_node_pytorch", self.mock_server, self.inference_engine, "localhost", 50052, 
                    self.mock_discovery, self.shard_downloader)

  async def asyncSetUp(self):
    if self.run_pytorch_tests:
      await self.node.start()

  async def asyncTearDown(self):
    if self.run_pytorch_tests:
      await self.node.stop()
  
  @pytest.mark.skipif(os.getenv("RUN_PYTORCH", "0") != "1", 
                     reason="PyTorch tests are disabled unless RUN_PYTORCH=1")
  async def test_node_with_pytorch_engine_initialization(self):
    """测试使用PyTorch引擎初始化Node"""
    # 验证Node正确初始化并使用了PyTorch引擎
    self.assertEqual(self.node.node_id, "test_node_pytorch")
    self.assertEqual(self.node.host, "localhost")
    self.assertEqual(self.node.port, 50052)
    
    # 验证get_supported_inference_engines返回PyTorch相关的引擎名称
    from exo.inference.pytorch.pytorch_inference_engine import PyTorchDynamicShardInferenceEngine
    if isinstance(self.inference_engine, PyTorchDynamicShardInferenceEngine):
      supported_engines = self.node.get_supported_inference_engines()
      self.assertIn("pytorch", supported_engines)
  
  @pytest.mark.skipif(os.getenv("RUN_PYTORCH", "0") != "1", 
                     reason="PyTorch tests are disabled unless RUN_PYTORCH=1")
  async def test_select_best_inference_engine(self):
    """测试Node选择最佳推理引擎功能"""
    # 模拟拓扑中没有其他引擎
    with patch.object(self.node, 'get_topology_inference_engines', return_value=[]):
      # 调用select_best_inference_engine方法
      await self.node.select_best_inference_engine()
      # 验证选择了PyTorch相关引擎
      self.assertIsNotNone(self.node.inference_engine)
  
  @pytest.mark.skipif(os.getenv("RUN_PYTORCH", "0") != "1", 
                     reason="PyTorch tests are disabled unless RUN_PYTORCH=1")
  async def test_process_prompt_with_pytorch(self):
    """测试使用PyTorch引擎处理prompt"""
    # 模拟基础分片
    mock_shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    test_prompt = "This is a test prompt"
    request_id = "test_request"
    
    # 模拟节点是第一层分片
    with patch.object(self.node, 'get_current_shard', return_value=mock_shard), \
         patch.object(self.node, '_process_prompt', AsyncMock()):
      await self.node.process_prompt(mock_shard, test_prompt, request_id)
      
      # 验证调用了_process_prompt
      self.node._process_prompt.assert_called_once()
  
  @pytest.mark.skipif(os.getenv("RUN_PYTORCH", "0") != "1", 
                     reason="PyTorch tests are disabled unless RUN_PYTORCH=1")
  async def test_process_tensor_with_pytorch(self):
    """测试使用PyTorch引擎处理tensor"""
    # 准备测试数据
    input_tensor = np.array([[1, 2, 3]])
    request_id = "tensor_request"
    mock_shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    
    # 模拟节点是第一层分片
    with patch.object(self.node, 'get_current_shard', return_value=mock_shard), \
         patch.object(self.node, '_process_tensor', AsyncMock()):
      await self.node.process_tensor(input_tensor, None, request_id=request_id, shard=mock_shard)
      
      # 验证调用了_process_tensor
      self.node._process_tensor.assert_called_once()


# 当直接运行此脚本时执行测试
if __name__ == '__main__':
  unittest.main()
