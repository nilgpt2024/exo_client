#!/usr/bin/env python3
"""
Qwen3 推理引擎测试
"""
import asyncio
import numpy as np
import torch
import sys
import time
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from transformers import AutoTokenizer
from exo.inference.pytorch.qwen3.pytorch_inference_engine import PyTorchQwen3InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.new_shard_download import new_shard_downloader
from pathlib import Path


class MockShardDownloader(ShardDownloader):
    """模拟的分片下载器"""
    def __init__(self, model_path: str):
        self.model_path = model_path

    async def ensure_shard(self, shard: Shard, inference_engine_name: str = None) -> Path:
        return Path(self.model_path)

    @property
    def on_progress(self):
        from exo.helpers import AsyncCallbackSystem
        return AsyncCallbackSystem()

    async def get_shard_download_status(self, inference_engine_name: str):
        if False:
            yield

model_path = r"C:\Users\nil\.cache\exo\downloads\Qwen--Qwen3-4B"
model_id = "qwen-3-4b"
async def test_single_shard():
    print("="*60)
    print("    模型推理测试 - EXO引擎版")
    print("="*60)

    # 使用EXO的下载器获取模型路径
    
    shard_downloader = new_shard_downloader()

    # 创建分片对象 - 使用完整模型范围
    shard = Shard(model_id=model_id, start_layer=0, end_layer=35, n_layers=36)

    print(f"模型ID: {model_id}")
    print(f"分片配置: start_layer={shard.start_layer}, end_layer={shard.end_layer}, n_layers={shard.n_layers}")

    # 获取模型路径（EXO下载的缓存路径）
    try:
        
        print(f"使用模型路径: {model_path}")

        # 检查路径是否存在
        import os
        if not os.path.exists(model_path):
            print(f"错误: 模型路径不存在: {model_path}")
            return

        print("开始加载tokenizer...")
        start_time = time.time()

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        load_time = time.time() - start_time
        print(f"Tokenizer加载完成，耗时: {load_time:.2f}秒")

        print("开始加载推理引擎...")
        start_time = time.time()

        # 创建推理引擎 - 使用完整模型范围
        print("创建PyTorchQwen3InferenceEngine实例...")
        engine = PyTorchQwen3InferenceEngine(
            shard_downloader=shard_downloader
        )
        print(f"推理引擎创建完成，类型: {type(engine)}")

        # 注意：不需要显式调用ensure_shard，infer_prompt内部会调用
        print(f"开始加载分片: {shard}")
        print("分片将在推理时自动加载")
        
        # 检查模型是否成功加载 - 注意：此时模型尚未加载，将在推理时加载
        print("注意：模型分片将在推理时自动加载")

        load_time = time.time() - start_time
        print(f"推理引擎加载完成，耗时: {load_time:.2f}秒")

        # 测试生成 - 使用中文提示
        prompt = "天空为什么是蓝色？" 

        
        # 生成参数 - 使用与Transformers测试相同的参数
        max_new_tokens = 200  # 减少生成的token数量，避免内存不足
        temperature = 0.8
        top_p = 0.9

        print(f"\n生成参数: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")

        # 生成回复
        try:
            print("开始生成...")

            start_gen_time = time.time()

            # 使用类似测试文件中的方法 - 简单的token生成循环
            request_id = f"test_1"

            # 首先使用infer_prompt获取初始状态
            prompt_shard = Shard(model_id=model_id, start_layer=0, end_layer=shard.n_layers-1, n_layers=shard.n_layers)
            print(f"调用infer_prompt，request_id: {request_id}")
            print(f"prompt_shard: {prompt_shard}")

            # 准备inference_state，包含图片信息
            inference_state = {
                "enable_thinking": False,
                "original_prompt": prompt  # 原始提示词
            }
            logits, inference_state = await engine.infer_prompt(request_id, prompt_shard, prompt, inference_state)

            print(f"infer_prompt完成，logits形状: {logits.shape if hasattr(logits, 'shape') else type(logits)}")

            # 获取生成的token（从inference_state中获取input_ids）
            input_ids = inference_state.get('input_ids', None)
            if input_ids is None:
                # 如果没有保存input_ids，使用空列表开始
                generated_tokens = []
            else:
                # 确保input_ids是一维数组
                if len(input_ids.shape) == 2 and input_ids.shape[0] == 1:
                    input_ids_flat = input_ids[0]
                else:
                    input_ids_flat = input_ids.flatten()
                # 使用列表存储生成的tokens
                generated_tokens = input_ids_flat.tolist()

            # 生成循环
            print(f"开始生成循环，最大token数: {max_new_tokens}")
            for i in range(max_new_tokens):
                print(f"\n--- 生成第 {i+1} 个token ---")

                # 采样下一个token
                print(f"采样参数: temperature={temperature}, top_p={top_p}")
                next_token = await engine.sample(logits, temp=temperature, top_p=top_p)
                print(f"采样结果: {next_token} (类型: {type(next_token)})")

                # 处理next_token格式
                if isinstance(next_token, np.ndarray):
                    next_token_value = int(next_token.item())
                else:
                    next_token_value = int(next_token)

                # 添加到生成序列
                generated_tokens.append(next_token_value)

                # 检查是否生成了结束token
                if next_token_value == tokenizer.eos_token_id:
                    print(f"生成结束token (eos_token_id={tokenizer.eos_token_id})")
                    break

                # 准备下一次推理的输入 - 使用刚生成的token
                token_input = np.array([[next_token_value]])  # 形状 [1, 1]
                print(f"调用infer_tensor，输入形状: {token_input.shape}")
                logits, inference_state = await engine.infer_tensor(
                    request_id,
                    prompt_shard,
                    token_input,
                    inference_state
                )
                print(f"infer_tensor完成，logits形状: {logits.shape if hasattr(logits, 'shape') else type(logits)}")

            gen_time = time.time() - start_gen_time

        except Exception as e:
            print(f"生成过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return

        # 解码所有生成的tokens
        all_generated_ids = np.array(generated_tokens).flatten()
        generated_text = tokenizer.decode(all_generated_ids, skip_special_tokens=True)

        print("="*50)
        print("生成结果:")
        print(generated_text)
        print("="*50)

        # 性能指标
        print(f"\n性能指标:")
        print(f"  生成的总token数: {len(generated_tokens)}")
        print(f"  总生成时间: {gen_time:.2f}秒")
        print(f"  平均每个token耗时: {gen_time*1000/len(generated_tokens):.2f}ms")
        print(f"  每秒生成token数(TPS): {len(generated_tokens)/gen_time:.2f} tokens/sec")
        print(f"  生成文本长度: {len(generated_text)}字符")


    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

async def test_sharded_inference():
    """测试 hidden_states 传递"""
    print("=" * 60)
    print("测试 hidden_states 传递")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    n_layers = 36
    pp = n_layers // 2  # 14

    print(f"\n模型配置:")
    print(f"  总层数: {n_layers}")
    print(f"  分片1: 层 0-{pp-1} ")
    print(f"  分片2: 层 {pp}-{n_layers-1} ")

    # 创建两个引擎
    shard_downloader = MockShardDownloader(model_path)
    engine_1 = PyTorchQwen2_5VlInferenceEngine(shard_downloader=shard_downloader, model_path=model_path)
    engine_2 = PyTorchQwen2_5VlInferenceEngine(shard_downloader=shard_downloader, model_path=model_path)

    # 加载分片
    shard_1 = Shard(model_id=model_id, start_layer=0, end_layer=pp-1, n_layers=n_layers)
    shard_2 = Shard(model_id=model_id, start_layer=pp, end_layer=n_layers-1, n_layers=n_layers)

    print(f"\n加载分片1: {shard_1}")
    await engine_1.load_checkpoint(shard_1, model_path)
    print(f"shard_1.is_first_layer(): {shard_1.is_first_layer()}")
    print(f"shard_1.is_last_layer(): {shard_1.is_last_layer()}")

    print(f"\n加载分片2: {shard_2}")
    await engine_2.load_checkpoint(shard_2, model_path)
    print(f"shard_2.is_first_layer(): {shard_2.is_first_layer()}")
    print(f"shard_2.is_last_layer(): {shard_2.is_last_layer()}")

    # 准备简单的输入
    print("\n准备输入")
    prompt = "天气为什么是蓝色？"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = engine_1.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = engine_1.processor(text=text, return_tensors="pt")
    input_ids = inputs['input_ids']

    print(f"  输入token形状: {input_ids.shape}")
    print(f"  输入token: {input_ids[0, :10]}")

    # 第一步：分片1处理
    print("\n步骤1: 分片1处理 input_ids")
    inference_state_1 = {}
    result_1, state_1 = await engine_1.infer_tensor(
        "test",
        shard=shard_1,
        input_data=input_ids.cpu().numpy(),
        inference_state=inference_state_1
    )
    print(f"  输出形状: {result_1.shape}")
    print(f"  输出类型: {type(result_1)}")
    print(f"  输出前5个值: {result_1[0, 0, :5]}")

    # 第二步：分片2处理 hidden_states
    print("\n步骤2: 分片2处理 hidden_states")
    inference_state_2 = {}
    result_2, state_2 = await engine_2.infer_tensor(
        "test",
        shard=shard_2,
        input_data=result_1,  # 传递 hidden_states
        inference_state=inference_state_2
    )
    print(f"  输出形状: {result_2.shape}")
    print(f"  输出类型: {type(result_2)}")
    print(f"  输出前5个值: {result_2[0, :5]}")

    # 采样 - 生成10个token
    print("\n步骤3: 采样生成10个token")
    generated_tokens = []
    max_tokens = 100

    # 准备下一次生成的输入
    current_input_ids = input_ids.cpu().numpy()

    for i in range(max_tokens):
        # 分片1处理 - 每个分片独立管理自己的KV缓存
        inference_state_1 = {'past_key_values': state_1.get('past_key_values') if i > 0 else None}
        result_1, state_1 = await engine_1.infer_tensor(
            "test",
            shard=shard_1,
            input_data=current_input_ids,
            inference_state=inference_state_1
        )

        # 分片2处理 - 每个分片独立管理自己的KV缓存
        inference_state_2 = {'past_key_values': state_2.get('past_key_values') if i > 0 else None}
        result_2, state_2 = await engine_2.infer_tensor(
            "test",
            shard=shard_2,
            input_data=result_1,
            inference_state=inference_state_2
        )

        # 采样
        next_token = await engine_1.sample(result_2, temp=0.7, repetition_penalty=1.1, generated_tokens=generated_tokens if generated_tokens else None)
        token_id = int(next_token.item())
        generated_tokens.append(token_id)

        print(f"  Token {i+1}: {token_id} -> '{engine_1.tokenizer.decode([token_id])}'")

        # 准备下一次输入
        current_input_ids = np.array([[token_id]])

        # 检查是否生成了结束符
        if token_id == engine_1.tokenizer.eos_token_id:
            break
    
    print(f"\n  生成的完整内容: '{engine_1.tokenizer.decode(generated_tokens)}'")
    print(f"  Token列表: {generated_tokens}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)



async def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("Fara-7B 推理引擎测试套件")
    print("=" * 70)

    try:
        # 测试1: 单分片推理
        await test_single_shard()

        # 测试2: 分片推理
        #await test_sharded_inference()

        # 测试3: 图像推理
        # await test_image_inference()

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
