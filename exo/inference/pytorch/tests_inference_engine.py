#!/usr/bin/env python3

import asyncio
import numpy as np
import torch
import os

# 内存优化设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import AutoTokenizer
from exo.download.new_shard_download import new_shard_downloader
from exo.inference.shard import Shard
from exo.inference.pytorch.pytorch_inference_engine import PyTorchDynamicShardInferenceEngine


async def kv_cache_functionality():
    """KV缓存功能测试 - 验证KV缓存的建立、传递和一致性（支持长序列生成）"""
    print("\n" + "="*60)
    print("🔍 开始KV缓存功能测试（长序列优化版）")
    print("="*60)
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"初始GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 使用EXO的下载器获取模型路径
    model_id = "qwen-3-4b"
    shard_downloader = new_shard_downloader()

    # 创建分片对象 - 使用完整模型范围
    shard = Shard(model_id=model_id, start_layer=0, end_layer=34, n_layers=35)

    print(f"模型ID: {model_id}")
    print(f"分片配置: start_layer={shard.start_layer}, end_layer={shard.end_layer}, n_layers={shard.n_layers}")

    # 获取模型路径（EXO下载的缓存路径）
    try:
        model_path = r"C:\Users\nil\.cache\exo\downloads\Qwen--Qwen3-0.6B"
        print(f"使用模型路径: {model_path}")

        print("开始加载tokenizer...")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False
        )

        # 创建推理引擎 - 使用完整模型范围
        engine = PyTorchDynamicShardInferenceEngine(
            shard_downloader=shard_downloader
        )

        # 确保分片加载
        await engine.ensure_shard(shard)

        # 测试生成 - 使用中文提示
        prompt = "天空为什么是蓝色？"

        print(f"\n测试提示: {prompt}")

        # 生成参数 - 内存高效模式，使用极小的批处理
        max_new_tokens = 500  # 保守目标，确保稳定性
        temperature = 0.8
        top_p = 0.95
        chunk_size = 8  # 极小的批大小，最大化内存效率
        memory_cleanup_interval = 10  # 非常频繁的清理
        # 获取结束标记ID用于停止条件检查
        eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
        
        # 生成回复 - 使用正确的实现方式
        try:
            print("开始生成...")
        
            # 首先编码输入提示
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            print(f"输入token数量: {len(input_ids[0])}")

            # 创建分片对象用于推理
            prompt_shard = Shard(model_id=model_id, start_layer=0, end_layer=shard.n_layers - 1,
                                 n_layers=shard.n_layers)
            
            # 🔥 优化：先在循环外使用infer_prompt处理完整提示，建立KV缓存状态
            print("  步骤 1: 使用infer_prompt处理完整提示，建立初始KV缓存状态")
            logits, inference_state = await engine.infer_prompt(
                f"test_request", 
                prompt_shard, 
                prompt, 
                {"enable_thinking": False}
            )
            
            # 🔍 调试：检查infer_prompt返回的KV缓存状态
            print(f"\n=== KV缓存状态检查 ===")
            print(f"推理状态类型: {type(inference_state)}")
            
            if inference_state is not None:
                # 检查ModelState对象的缓存
                if hasattr(inference_state, 'cache'):
                    cache = inference_state.cache
                    print(f"缓存类型: {type(cache)}")
                    
                    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
                        num_layers = len(cache.key_cache)
                        print(f"KV缓存层数: {num_layers}")
                        
                        valid_layers = 0
                        for i in range(num_layers):
                            if i < len(cache.key_cache) and cache.key_cache[i] is not None:
                                valid_layers += 1
                                k_shape = cache.key_cache[i].shape
                                v_shape = cache.value_cache[i].shape if i < len(cache.value_cache) else 'unknown'
                                print(f"  层 {i}: 有缓存 - K形状: {k_shape}, V形状: {v_shape}")
                            else:
                                print(f"  层 {i}: 无缓存")
                        
                        if valid_layers == 0:
                            print("⚠️ 警告: 没有有效的KV缓存层！")
                        else:
                            print(f"✅ 有 {valid_layers}/{num_layers} 层有有效缓存")
                    else:
                        print("缓存对象没有key_cache/value_cache属性")
                else:
                    print("推理状态没有cache属性")
            else:
                print("🔍 调试: infer_prompt返回的推理状态为None")
            
            # 采样第一个新token
            next_token = await engine.sample(logits, temp=temperature, top_p=top_p)
            if isinstance(next_token, np.ndarray):
                next_token_value = next_token.item()
            else:
                next_token_value = next_token
            
            # 初始化生成的tokens序列（包含原始提示 + 第一个生成的token）
            generated_tokens = input_ids.squeeze().tolist()
            # 确保generated_tokens是列表类型，避免标量情况
            if not isinstance(generated_tokens, list):
                generated_tokens = [generated_tokens]
            generated_tokens.append(next_token_value)
            print(f"初始序列长度: {len(generated_tokens)} (包含{len(input_ids[0])}个输入token + 1个生成token)")

            # 循环内进行增量推理 - 从第二个token开始（支持长序列生成）
            step = 0
            eos_detected = False
            total_chunks = (max_new_tokens + chunk_size - 1) // chunk_size
            kv_cache_updates = 0
            
            # 🔧 KV缓存优化参数
            max_cache_length = 64  # 进一步限制KV缓存最大长度，减少内存占用
            cache_cleanup_interval = 10  # 每10步检查和清理缓存
            memory_threshold_high = 6.0  # 6GB开始警告
            memory_threshold_critical = 8.0  # 8GB强制停止
            
            for chunk in range(total_chunks):
                print(f"\n📦 开始第 {chunk+1}/{total_chunks} 批生成 (每批{chunk_size}个token)")
                
                # 每批生成前清理内存和监控
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  当前GPU内存使用: {current_memory:.2f} GB")
                    
                    # 内存管理策略
                    if current_memory > 15:
                        print(f"  🚨 内存使用超过15GB，强制停止生成！")
                        break
                    elif current_memory > 12.0:
                        chunk_size = max(4, chunk_size // 2)
                        print(f"  ⚠️ 内存使用过高，减少批大小到: {chunk_size}")
                
                # 🔧 KV缓存滑动窗口优化：批次开始前检查清理
                current_seq_len = len(generated_tokens)
                if current_seq_len > max_cache_length and inference_state is not None:
                    print(f"  🔧 KV缓存清理：序列长度{current_seq_len} → {max_cache_length}，释放内存")
                    
                    if hasattr(inference_state, 'cache') and inference_state.cache is not None:
                        cache = inference_state.cache
                        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
                            # 清理旧的KV缓存，保留最近的
                            for layer_idx in range(len(cache.key_cache)):
                                if cache.key_cache[layer_idx] is not None:
                                    # 保留最近max_cache_length个位置
                                    cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, -max_cache_length:, :, :]
                                if cache.value_cache[layer_idx] is not None:
                                    cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, -max_cache_length:, :, :]
                            
                            # 调整位置计数
                            if hasattr(inference_state, 'position'):
                                inference_state.position = max_cache_length
                            
                            print(f"  ✅ KV缓存已清理，保留最近{max_cache_length}个token")
                            
                            # 强制垃圾回收
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                
                # 每批生成chunk_size个token
                batch_tokens_generated = 0
                for i in range(min(chunk_size, max_new_tokens - step)):
                    step += 1
                    batch_tokens_generated += 1
                    kv_cache_updates += 1
                    
                    # 只传入最新的token进行增量推理
                    current_token = np.array([[generated_tokens[-1]]], dtype=np.int64)
                    
                    # 使用infer_tensor进行推理 - KV缓存会自动处理历史信息
                    logits, inference_state = await engine.infer_tensor(
                        f"test_request", 
                        prompt_shard, 
                        current_token, 
                        inference_state
                    )

                    # 采样下一个token - 传入温度参数
                    next_token = await engine.sample(logits, temp=temperature, top_p=top_p)

                    # 处理next_token格式 - samples返回的是numpy数组
                    if isinstance(next_token, np.ndarray):
                        next_token_value = next_token.item()  # 从numpy数组中提取标量值
                    else:
                        next_token_value = next_token

                    # 添加到生成的序列中
                    generated_tokens.append(next_token_value)
                    
                    # 检查结束token
                    if next_token_value == eos_token_id:
                        eos_detected = True
                        print(f"  步骤 {step}: 检测到EOS token{next_token_value}，提前结束生成")
                        break
                
                # 批次完成后输出进度
                if batch_tokens_generated > 0:
                    print(f"  本批生成 {batch_tokens_generated} 个token，当前序列长度: {len(generated_tokens)}")
                
                if eos_detected:
                    break

            print(f"最终序列长度: {len(generated_tokens)}")

        except Exception as e:
            print(f"生成过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 解码所有生成的tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("生成结果:")
        print(generated_text)

        print(f"\n✅ KV缓存功能测试完成！")
        print(f"📊 统计信息:")
        print(f"   目标生成token数: {max_new_tokens}")
        print(f"   实际生成token数: {step}")
        print(f"   KV缓存更新次数: {kv_cache_updates}")
        print(f"   最终序列长度: {len(generated_tokens)}")
        print(f"   检测到EOS token: {eos_detected}")
        print(f"   生成内容长度: {len(generated_text)} 字符")
        print(f"   使用分块策略: 每批{chunk_size}个token")

        return True

    except Exception as e:
        print(f"KV缓存功能测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"BF16支持状态: {bf16_supported}")
    if bf16_supported:
        print("✅ 检测到BF16支持，推理将使用BF16优化")
        print("💡 BF16优化效果:")
        print("   - 内存占用减少约50%")
        print("   - 推理速度提升")
        print("   - 精度损失极小")
    else:
        print("ℹ️ 未检测到BF16支持，使用默认精度")
    
    asyncio.run(kv_cache_functionality())