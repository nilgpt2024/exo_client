from exo.inference.inference_engine import InferenceEngine
from exo.download.new_shard_download import NewShardDownloader
from exo.inference.shard import Shard
from exo.helpers import DEBUG
import os
import asyncio
import numpy as np


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def main(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str, n_layers: int):
  prompt = "中国的首都在哪?"


  pp = n_layers // 2
  resp1, _ = await inference_engine_1.infer_prompt("B", shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers), prompt=prompt)
  resp2, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=n_layers - 1, n_layers=n_layers),
    input_data=resp1,
  )
  # 使用与不分片推理相同的采样参数
  temperature = 0.7
  top_p = 0.95
  
  tokens2 = await inference_engine_1.sample(resp2, temp=temperature, top_p=top_p)
  tokens2 = tokens2.reshape(1, -1)
  max_tokens = 500  # 减少最大token数以避免过长生成
  shard_tokens = []
  current_input = tokens2
  for i in range(max_tokens - 1):  # 已经生成了第一个token
      resp3, _ = await inference_engine_1.infer_tensor(
          "B",  # 使用相同的请求ID保持状态连续
          shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers),
          input_data=current_input,
      )
      resp4, _ = await inference_engine_2.infer_tensor(
          "B",  # 使用相同的请求ID保持状态连续
          shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=n_layers - 1, n_layers=n_layers),
          input_data=resp3,
      )
      next_token = await inference_engine_1.sample(resp4, temp=temperature, top_p=top_p)
      next_token = next_token.reshape(1, -1)
      shard_tokens.append(next_token[0])

      # 检查是否生成了结束标记
      if int(next_token.item()) == 151645:
          print(f"\n检测到EOS标记，停止生成")
          break

      current_input = next_token

  all_shard_tokens = np.concatenate([tokens2[0]] + shard_tokens)
  shard_answer = await inference_engine_1.decode(
      Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers),
      all_shard_tokens
  )
  print(f"分片模型回答: '{shard_answer}'")


from exo.inference.pytorch.qwen3.pytorch_inference_engine import PyTorchQwen3InferenceEngine
from exo.download.new_shard_download import NewShardDownloader
asyncio.run(main(
PyTorchQwen3InferenceEngine(NewShardDownloader()),
PyTorchQwen3InferenceEngine(NewShardDownloader()),
"qwen-3-0.6b", 28
))
