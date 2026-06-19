#!/usr/bin/env python3
"""
Llama3-1B 模型推理测试 - 类似qwen3_test_inference.py的实现
使用EXO的模型加载机制，但采用类似Transformers的推理方式
"""

import asyncio
import numpy as np
import time
import torch
from transformers import AutoTokenizer
from exo.download.new_shard_download import new_shard_downloader
from exo.inference.pytorch.llama3.pytorch_inference_engine import PyTorchLlama3InferenceEngine
from exo.inference.shard import Shard

async def main():
    print("="*60)
    print("    Llama3-1B 模型推理测试 - EXO引擎版")
    print("="*60)

    # 使用EXO的下载器获取模型路径
    model_id = "llama-3.2-1b"
    shard_downloader = new_shard_downloader()

    # 创建分片对象 - 使用完整模型范围
    shard = Shard(model_id=model_id, start_layer=0, end_layer=15, n_layers=16)

    print(f"模型ID: {model_id}")
    print(f"分片配置: start_layer={shard.start_layer}, end_layer={shard.end_layer}, n_layers={shard.n_layers}")

    # 获取模型路径（EXO下载的缓存路径）
    try:
        model_path = r"C:\Users\nil\.cache\exo\downloads\unsloth--Llama-3.2-1B-Instruct"
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
        print("创建PyTorchLlama3InferenceEngine实例...")
        engine = PyTorchLlama3InferenceEngine(
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
        prompt = "why the sky is blue?"
        messages = [{"role": "user", "content": prompt}]

        print(f"\n测试提示: {prompt}")

        # 应用聊天模板 - 使用与Transformers测试相同的配置
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 禁用思考模式
        )

        print(f"格式化后的提示: {text[:100]}...")

        # 编码输入
        input_ids = tokenizer.encode(text, return_tensors="np")
        print(f"输入token数量: {input_ids.shape[1]}")

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
            print(f"输入文本长度: {len(text)} 字符")
            
            logits, _ = await engine.infer_prompt(request_id, prompt_shard, text, {"enable_thinking": False})
            
            print(f"infer_prompt完成，logits形状: {logits.shape if hasattr(logits, 'shape') else type(logits)}")

            # 确保input_ids是一维数组
            if len(input_ids.shape) == 2 and input_ids.shape[0] == 1:
                input_ids_flat = input_ids[0]
            else:
                input_ids_flat = input_ids.flatten()
            
            # 使用列表存储生成的tokens，避免numpy数组操作
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
                logits, _ = await engine.infer_tensor(
                    request_id,
                    prompt_shard,
                    token_input,
                    {"enable_thinking": False}
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

if __name__ == "__main__":
    asyncio.run(main())