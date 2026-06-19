import asyncio
import asyncio
import json
import os
from pathlib import Path
import tempfile

import aiohttp

from exo.download.new_shard_download import NewShardDownloader
from exo.download.new_shard_download import (
    _fetch_file_list_modelscope,
    check_modelscope_sdk,
    download_file_with_retry,
    file_meta_modelscope,
    get_modelscope_auth_headers,
    get_modelscope_endpoint,
    get_modelscope_model_list,
    install_modelscope_sdk,
    new_shard_downloader,
    test_modelscope_model_exists,
)
from exo.inference.shard import Shard
from exo.inference.shard import Shard

async def test_new_shard_download():
  shard_downloader = NewShardDownloader()
  shard_downloader.on_progress.register("test").on_next(lambda shard, event: print(shard, event))
  await shard_downloader.ensure_shard(Shard(model_id="llama-3.2-1b", start_layer=0, end_layer=0, n_layers=16), "MLXDynamicShardInferenceEngine")
  async for path, shard_status in shard_downloader.get_shard_download_status("MLXDynamicShardInferenceEngine"):
    print("Shard download status:", path, shard_status)

#if __name__ == "__main__":
  #asyncio.run(test_new_shard_download())
# 更新后的测试脚本


# 更新测试脚本以更好地处理错误情况
# 更新测试脚本
# 修复测试脚本中的错误




async def debug_modelscope_api():
    """调试ModelScope API调用"""
    print("=== 调试ModelScope API ===")

    # 获取一些已知的模型
    known_models = await get_modelscope_model_list()
    print(f"已知的ModelScope模型: {known_models[:3]}...")  # 只显示前几个

    # 测试SDK可用性
    print(f"\nModelScope SDK 可用性: {await check_modelscope_sdk()}")

    # 测试模型是否存在
    print("\n测试模型是否存在:")
    test_model = None
    for model_id in known_models[:3]:  # 测试前几个模型
        exists = await test_modelscope_model_exists(model_id)
        print(f"  {model_id}: {'存在' if exists else '不存在'}")
        if exists:
            test_model = model_id
            break
    else:
        print("  没有找到可用的测试模型")
        return

    # 测试API端点
    endpoint = get_modelscope_endpoint()
    print(f"\nModelScope endpoint: {endpoint}")

    # 测试认证头
    headers = await get_modelscope_auth_headers()
    print(f"Headers: {headers}")

    # 测试文件列表API
    print(f"\n测试文件列表API for {test_model}:")
    try:
        file_list = await _fetch_file_list_modelscope(test_model, "master", "")
        print(f"  成功获取文件列表，共{len(file_list)}个文件")
        for file in file_list[:3]:  # 显示前3个文件
            print(f"    - {file['path']} ({file['size']} bytes)")
    except Exception as e:
        print(f"  获取文件列表失败: {e}")


async def test_modelscope_model_discovery():
    """测试ModelScope模型发现"""
    print("\n测试ModelScope模型发现...")
    try:
        known_models = await get_modelscope_model_list()
        print(f"已知模型列表: {known_models}")

        for model_id in known_models[:5]:  # 测试前5个模型
            exists = await test_modelscope_model_exists(model_id)
            if exists:
                print(f"找到可用模型: {model_id}")
                return True  # 返回布尔值

        print("没有找到可用的测试模型")
        return False  # 返回布尔值
    except Exception as e:
        print(f"模型发现失败: {e}")
        import traceback
        traceback.print_exc()
        return False  # 返回布尔值


async def test_modelscope_file_list():
    """测试获取ModelScope文件列表功能"""
    print("\n测试获取ModelScope文件列表...")
    try:
        # 首先找到一个可用的模型
        known_models = await get_modelscope_model_list()
        model_id = None
        for test_model in known_models[:3]:
            if await test_modelscope_model_exists(test_model):
                model_id = test_model
                break

        if not model_id:
            print("无法找到可用的测试模型")
            return False

        revision = "master"
        file_list = await _fetch_file_list_modelscope(model_id, revision, "")
        print(f"成功获取文件列表，共{len(file_list)}个文件:")
        for file in file_list[:5]:  # 只显示前5个文件
            print(f"  - {file['path']} ({file['size']} bytes)")
        return True
    except Exception as e:
        print(f"获取文件列表失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_modelscope_file_metadata():
    """测试获取ModelScope文件元数据功能"""
    print("\n测试获取ModelScope文件元数据...")
    try:
        # 首先找到一个可用的模型
        known_models = await get_modelscope_model_list()
        model_id = None
        for test_model in known_models[:3]:
            if await test_modelscope_model_exists(test_model):
                model_id = test_model
                break

        if not model_id:
            print("无法找到可用的测试模型")
            return False

        revision = "master"
        # 尝试获取文件列表中的文件
        try:
            file_list = await _fetch_file_list_modelscope(model_id, revision, "")
            if not file_list:
                print("模型中没有文件")
                return False

            # 尝试获取第一个文件的元数据
            file_path = file_list[0]['path']
            size, etag = await file_meta_modelscope(model_id, revision, file_path)
            print(f"文件 {file_path} 的元数据:")
            print(f"  大小: {size} bytes")
            print(f"  标识符: {etag}")
            return True
        except Exception as e:
            print(f"获取文件元数据失败: {e}")
            return False

    except Exception as e:
        print(f"获取文件元数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_modelscope_file_download():
    """测试从ModelScope下载单个文件"""
    print("\n测试从ModelScope下载文件...")
    try:
        # 首先找到一个可用的模型
        known_models = await get_modelscope_model_list()
        model_id = None
        for test_model in known_models[:3]:
            if await test_modelscope_model_exists(test_model):
                model_id = test_model
                break

        if not model_id:
            print("无法找到可用的测试模型")
            return False

        revision = "master"
        # 获取文件列表
        try:
            file_list = await _fetch_file_list_modelscope(model_id, revision, "")
            if not file_list:
                print("模型中没有文件")
                return False

            # 尝试下载较小的文件
            files_to_try = [f for f in file_list if f['size'] > 0 and f['size'] < 1000000][:3]  # 选择小于1MB的文件
            if not files_to_try:
                # 如果没有小文件，尝试第一个文件
                files_to_try = file_list[:1]

            for file_info in files_to_try:
                file_path = file_info['path']
                try:
                    print(f"尝试下载文件: {file_path} ({file_info['size']} bytes)")
                    # 创建临时目录用于下载
                    with tempfile.TemporaryDirectory() as temp_dir:
                        target_dir = Path(temp_dir)
                        downloaded_path = await download_file_with_retry(
                            model_id, revision, file_path, target_dir,
                            source="modelscope"
                        )
                        print(f"文件下载成功: {downloaded_path}")
                        print(f"文件大小: {os.path.getsize(downloaded_path)} bytes")
                        return True
                except FileNotFoundError as e:
                    print(f"  文件 {file_path} 未找到: {e}")
                    continue
                except Exception as e:
                    print(f"  文件 {file_path} 下载失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print("所有尝试的文件都下载失败")
            return False
        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return False

    except Exception as e:
        print(f"文件下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_modelscope_sdk_functionality():
    """测试ModelScope SDK功能"""
    print("\n测试ModelScope SDK功能...")
    try:
        is_installed = await check_modelscope_sdk()
        if not is_installed:
            print("ModelScope SDK 未安装")
            return False

        print("ModelScope SDK 已正确安装")
        return True
    except Exception as e:
        print(f"ModelScope SDK 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_modelscope_integration():
    """综合测试ModelScope集成"""
    print("=== ModelScope 下载功能测试 ===")

    # 首先调试API
    await debug_modelscope_api()

    tests = [
        test_modelscope_model_discovery,
        test_modelscope_file_list,
        test_modelscope_file_metadata,
        test_modelscope_file_download,
        test_modelscope_sdk_functionality
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            # 确保返回的是布尔值
            if isinstance(result, bool):
                results.append(result)
            else:
                print(f"测试 {test.__name__} 返回了非布尔值: {result}")
                results.append(False)
        except Exception as e:
            print(f"测试 {test.__name__} 发生未预期错误: {e}")
            results.append(False)

    print(f"\n=== 测试结果 ===")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("所有测试通过！ModelScope下载功能正常工作。")
    else:
        print("部分测试失败，请检查ModelScope集成。")

    return passed == total


# 添加一个简单的使用示例
async def example_modelscope_download(model_id=None):
    """ModelScope下载使用示例"""
    print("\n=== ModelScope 下载使用示例 ===")

    try:
        from pathlib import Path
        # 导入标准下载目录函数
        from exo.download.new_shard_download import ensure_downloads_dir
        
        # 检查SDK
        if not await check_modelscope_sdk():
            print("正在安装ModelScope SDK...")
            if not await install_modelscope_sdk():
                print("ModelScope SDK 安装失败")
                return
            print("ModelScope SDK 安装成功")

        # 如果没有提供model_id，查找一个可用模型
        if not model_id:
            known_models = await get_modelscope_model_list()
            for test_model in known_models:
                if await test_modelscope_model_exists(test_model):
                    model_id = test_model
                    break

        if not model_id:
            print("未找到可用模型")
            return

        print(f"找到可用模型: {model_id}")

        # 使用标准下载目录替代临时目录
        downloads_dir = await ensure_downloads_dir()
        # 构建具体模型下载路径 (将"/"替换为"--")
        model_cache_dir = downloads_dir / model_id.replace("/", "--")
        
        # 使用SDK下载，直接下载到目标目录，然后处理可能的嵌套结构
        from modelscope.hub.snapshot_download import snapshot_download
        import shutil
        
        print(f"正在下载模型到: {model_cache_dir}")
        
        # 直接下载到目标目录的父目录作为cache位置
        cache_parent_dir = model_cache_dir.parent
        
        # 如果目标目录已存在，先删除它
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
        
        # 直接下载，使用目标目录的父目录作为cache位置
        downloaded_dir = snapshot_download(
            model_id, 
            cache_dir=str(cache_parent_dir),
            revision="master"
        )
        
        # 检查是否创建了嵌套目录结构
        downloaded_path = Path(downloaded_dir)
        
        # ModelScope SDK 可能创建嵌套目录结构，例如:
        # 下载到 Qwen--Qwen3-4B/Qwen/Qwen3-4B 但我们需要的是 Qwen--Qwen3-4B
        repo_parts = model_id.split('/')
        if len(repo_parts) >= 2:
            # 检查是否存在嵌套结构: downloaded_path/组织名/模型名
            nested_dir = downloaded_path / repo_parts[0] / repo_parts[1]
            if nested_dir.exists():
                print(f"发现嵌套目录结构，正在移动文件从 {nested_dir} 到 {model_cache_dir}")
                # 移动嵌套目录中的文件到目标位置
                shutil.move(str(nested_dir), str(model_cache_dir))
                
                # 清理剩余的嵌套目录结构
                remaining_org_dir = downloaded_path / repo_parts[0]
                if remaining_org_dir.exists() and not any(remaining_org_dir.iterdir()):
                    remaining_org_dir.rmdir()
                
                # 如果下载的目录现在为空，也删除它（避免留下空目录）
                if downloaded_path.exists() and downloaded_path != model_cache_dir and not any(downloaded_path.iterdir()):
                    downloaded_path.rmdir()
            else:
                # 如果没有嵌套结构，但路径不同，直接重命名
                if downloaded_path != model_cache_dir:
                    shutil.move(str(downloaded_path), str(model_cache_dir))
        else:
            # 单部分模型ID（没有/）
            if downloaded_path != model_cache_dir:
                shutil.move(str(downloaded_path), str(model_cache_dir))
        
        print(f"模型下载完成: {model_cache_dir}")

        # 列出下载的文件
        downloaded_files = list(model_cache_dir.rglob("*"))
        print(f"下载了 {len(downloaded_files)} 个文件/目录:")
        for f in downloaded_files[:10]:  # 显示前10个
            if f.is_file():
                print(f"  文件: {f.relative_to(model_cache_dir)} ({f.stat().st_size} bytes)")
        if len(downloaded_files) > 10:
            print(f"  ... 还有 {len(downloaded_files) - 10} 个文件")

    except Exception as e:
        print(f"下载示例失败: {e}")
        import traceback
        traceback.print_exc()


# 新增一个函数专门用于测试下载指定模型
async def test_specific_model_download(model_id: str):
    """测试下载指定的模型"""
    print(f"\n=== 测试下载指定模型: {model_id} ===")
    
    # 首先检查模型是否存在
    if not await test_modelscope_model_exists(model_id):
        print(f"错误: 模型 {model_id} 不存在")
        return False
    
    # 测试获取文件列表
    try:
        revision = "master"
        file_list = await _fetch_file_list_modelscope(model_id, revision, "")
        print(f"成功获取模型文件列表，共{len(file_list)}个文件")
        
        # 尝试下载模型
        await example_modelscope_download(model_id)
        return True
    except Exception as e:
        print(f"下载指定模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        # 运行集成测试
        #success = await test_modelscope_integration()
        
        # 如果测试成功，运行示例
        #if success:
            #await example_modelscope_download()
        await test_specific_model_download("Qwen/Qwen2.5-VL-3B-Instruct")

    asyncio.run(main())



