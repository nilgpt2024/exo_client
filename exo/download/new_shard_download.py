from exo.inference.shard import Shard
from exo.models import get_repo
from pathlib import Path
from exo.download.hf.hf_helpers import get_hf_endpoint, get_auth_headers, filter_repo_objects, get_allow_patterns
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent, RepoFileProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.models import get_supported_models, build_full_shard
import os
import aiofiles.os as aios
import aiohttp
import aiofiles
from urllib.parse import urljoin
from typing import Callable, Union, Tuple, Dict, List, Optional, Literal, AsyncIterator
import time
from datetime import timedelta
import asyncio
import json
import traceback
import shutil
import tempfile
import hashlib
import subprocess
import sys
from pathlib import Path
# 以下函数用于获取和管理Exo相关的目录路径

# 获取Exo主目录路径。优先从环境变量 EXO_HOME 获取，若未设置则默认使用 ~/.cache/exo
def exo_home() -> Path:
    return Path(os.environ.get("EXO_HOME", Path.home()/".cache"/"exo"))

# 获取Exo临时目录路径，默认在系统临时目录下创建 exo 子目录
def exo_tmp() -> Path:
    return Path(tempfile.gettempdir())/"exo"

# 确保Exo主目录存在，若不存在则创建该目录，最后返回主目录路径
async def ensure_exo_home() -> Path:
    await aios.makedirs(exo_home(), exist_ok=True)
    return exo_home()

# 确保Exo临时目录存在，若不存在则创建该目录，最后返回临时目录路径
async def ensure_exo_tmp() -> Path:
    await aios.makedirs(exo_tmp(), exist_ok=True)
    return exo_tmp()

# 检查Exo主目录是否有读取权限
async def has_exo_home_read_access() -> bool:
    try: return await aios.access(exo_home(), os.R_OK)
    except OSError: return False

# 检查Exo主目录是否有写入权限
async def has_exo_home_write_access() -> bool:
    try: return await aios.access(exo_home(), os.W_OK)
    except OSError: return False

# 确保Exo的下载目录存在，若不存在则创建该目录，最后返回下载目录路径
async def ensure_downloads_dir() -> Path:
    downloads_dir = exo_home()/"downloads"
    await aios.makedirs(downloads_dir, exist_ok=True)
    return downloads_dir

# 删除指定模型的下载目录，返回删除是否成功
async def delete_model(model_id: str, inference_engine_name: str) -> bool:
    repo_id = get_repo(model_id, inference_engine_name)
    model_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
    if not await aios.path.exists(model_dir): return False
    await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
    return True

# 将应用资源文件夹中的模型移动到 .cache/huggingface/hub 目录
async def seed_models(seed_dir: Union[str, Path]):
    """Move model in resources folder of app to .cache/huggingface/hub"""
    source_dir = Path(seed_dir)
    dest_dir = await ensure_downloads_dir()
    for path in source_dir.iterdir():
        if path.is_dir() and path.name.startswith("models--"):
            dest_path = dest_dir/path.name
            if await aios.path.exists(dest_path): print('Skipping moving model to .cache directory')
            else:
                try: await aios.rename(str(path), str(dest_path))
                except:
                    print(f"Error seeding model {path} to {dest_path}")
                    traceback.print_exc()

# 检查是否安装了ModelScope SDK
async def check_modelscope_sdk():
    """检查是否安装了ModelScope SDK"""
    try:
        import modelscope
        return True
    except ImportError:
        return False

# 安装ModelScope SDK
async def install_modelscope_sdk():
    """安装ModelScope SDK"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        return True
    except subprocess.CalledProcessError:
        return False

# 获取一些可用的ModelScope模型作为测试
async def get_modelscope_model_list():
    """获取一些可用的ModelScope模型作为测试"""
    # 一些已知可用的ModelScope模型
    known_models = [
        "damo/nlp_structbert_sentence-similarity_chinese-base",
        "AI-ModelScope/bert-base-uncased",
        "qwen/Qwen2-0.5B",
        "Qwen/Qwen2-0.5B-Instruct",
        "bert-base-chinese"
    ]
    return known_models

# 测试ModelScope模型是否存在
async def test_modelscope_model_exists(model_id: str, revision: str = "master") -> bool:
    """测试ModelScope模型是否存在"""
    try:
        # 首先尝试使用SDK方式检查
        if await check_modelscope_sdk():
            from modelscope.hub.api import HubApi
            api = HubApi()
            # 尝试获取模型信息
            model_info = api.get_model(model_id)
            return model_info is not None
    except:
        pass

    # 如果SDK方式失败，尝试API方式
    try:
        endpoint = get_modelscope_endpoint()
        api_url = f"{endpoint}/models/{model_id}"
        headers = await get_modelscope_auth_headers()

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, headers=headers) as response:
                return response.status == 200
    except:
        pass

    return False

# 获取ModelScope API端点
def get_modelscope_endpoint() -> str:
    """获取ModelScope API端点"""
    return "https://www.modelscope.cn/api/v1"

# 获取ModelScope认证头（不需要认证）
async def get_modelscope_auth_headers() -> Dict[str, str]:
    """获取ModelScope认证头（不需要认证）"""
    return {"User-Agent": "exo-model-downloader"}

# 使用ModelScope SDK下载模型
async def download_with_modelscope_sdk(repo_id: str, revision: str = "master",
                                       allow_patterns: Optional[List[str]] = None,
                                       ignore_patterns: Optional[List[str]] = None,
                                       local_dir: Optional[str] = None) -> str:
    """使用ModelScope SDK下载模型"""
    # 检查是否安装了ModelScope SDK
    if not await check_modelscope_sdk():
        print("ModelScope SDK未安装，正在安装...")
        if not await install_modelscope_sdk():
            raise Exception("无法安装ModelScope SDK")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
        import shutil

        # 如果指定了local_dir，直接使用它作为目标目录
        if local_dir:
            target_dir = Path(local_dir)
            await aios.makedirs(target_dir, exist_ok=True)
            
            # 检查目标目录是否已存在模型文件
            if await aios.path.exists(target_dir) and len(list(target_dir.iterdir())) > 0:
                # 检查是否存在关键模型文件
                has_model_files = False
                for file_pattern in ["*.safetensors", "*.bin", "config.json", "tokenizer.json"]:
                    if any(target_dir.glob(file_pattern)):
                        has_model_files = True
                        break
                
                if has_model_files:
                    print(f"模型文件已存在于 {target_dir}，跳过下载")
                    return str(target_dir)
            
            # 直接下载到目标目录的父目录作为cache位置
            import shutil
            cache_parent_dir = target_dir.parent
            
            # 如果目标目录已存在，先删除它
            if target_dir.exists():
                await asyncio.to_thread(shutil.rmtree, target_dir)
            
            # 准备下载参数
            download_kwargs = {
                "model_id": repo_id,
                "revision": revision,
                "cache_dir": str(cache_parent_dir)  # 直接下载到父目录
            }

            if allow_patterns:
                download_kwargs["allow_patterns"] = allow_patterns
            if ignore_patterns:
                download_kwargs["ignore_patterns"] = ignore_patterns

            # 执行下载（放到线程池，不阻塞事件循环）
            downloaded_dir = await asyncio.to_thread(snapshot_download, **download_kwargs)
            
            # 检查是否创建了嵌套目录结构
            downloaded_path = Path(downloaded_dir)

            def _has_model_files(path: Path) -> bool:
                if not path.exists() or not path.is_dir():
                    return False
                return any(
                    path.glob(pattern) for pattern in
                    ["model.safetensors.index.json", "pytorch_model.bin.index.json", "*.safetensors", "*.bin"]
                )

            def _find_actual_model_dir(path: Path) -> Optional[Path]:
                # 直接命中
                if _has_model_files(path):
                    return path
                # HF 风格缓存：models/<repo_id--name>/snapshots/<revision>/
                local_name = repo_id.replace("/", "--")
                hf_cache = path / "models" / local_name / "snapshots" / revision
                if _has_model_files(hf_cache):
                    return hf_cache
                # 原始 repo 结构：<org>/<model>/
                repo_parts = repo_id.split('/')
                if len(repo_parts) >= 2:
                    nested = path / repo_parts[0] / repo_parts[1]
                    if _has_model_files(nested):
                        return nested
                return None

            actual_dir = _find_actual_model_dir(downloaded_path)
            if actual_dir is None:
                raise Exception(f"无法在 {downloaded_path} 中找到模型文件（repo_id={repo_id}, revision={revision}）")

            if actual_dir != target_dir:
                print(f"发现嵌套目录结构，正在移动文件从 {actual_dir} 到 {target_dir}")
                await asyncio.to_thread(shutil.move, str(actual_dir), str(target_dir))

            # 清理下载过程中可能留下的空目录
            try:
                if downloaded_path.exists() and downloaded_path != target_dir and not any(downloaded_path.iterdir()):
                    await asyncio.to_thread(downloaded_path.rmdir)
            except Exception:
                pass
            
            # 验证下载是否成功
            if not target_dir.exists() or not any(target_dir.iterdir()):
                raise Exception(f"模型下载失败，目录为空或不存在: {target_dir}")
            
            print(f"模型成功下载到: {target_dir}")
            return str(target_dir)
        else:
            # 没有指定local_dir时使用默认行为
            download_kwargs = {
                "model_id": repo_id,
                "revision": revision
            }

            if allow_patterns:
                download_kwargs["allow_patterns"] = allow_patterns
            if ignore_patterns:
                download_kwargs["ignore_patterns"] = ignore_patterns

            # 执行下载（放到线程池，不阻塞事件循环）
            model_dir = await asyncio.to_thread(snapshot_download, **download_kwargs)
            return model_dir

    except Exception as e:
        raise Exception(f"使用ModelScope SDK下载失败: {e}")

# 使用ModelScope SDK下载单个文件
async def download_file_with_modelscope_sdk(repo_id: str, file_path: str,
                                            revision: str = "master",
                                            local_dir: Optional[str] = None) -> str:
    """使用ModelScope SDK下载单个文件"""
    # 检查是否安装了ModelScope SDK
    if not await check_modelscope_sdk():
        print("ModelScope SDK未安装，正在安装...")
        if not await install_modelscope_sdk():
            raise Exception("无法安装ModelScope SDK")

    try:
        from modelscope.hub.file_download import model_file_download

        # 准备下载参数
        download_kwargs = {
            "model_id": repo_id,
            "file_path": file_path,
            "revision": revision
        }

        if local_dir:
            download_kwargs["cache_dir"] = local_dir

        # 执行下载（放到线程池，不阻塞事件循环）
        file_path = await asyncio.to_thread(model_file_download, **download_kwargs)
        return file_path

    except Exception as e:
        raise Exception(f"使用ModelScope SDK下载文件失败: {e}")

# 获取文件列表并进行缓存，如果缓存存在则直接返回缓存数据，否则获取并缓存文件列表
async def fetch_file_list_with_cache(repo_id: str, revision: str = "main", source: str = "modelscope") -> List[Dict[str, Union[str, int]]]:
    cache_file = (await ensure_exo_tmp())/f"{repo_id.replace('/', '--')}--{revision}--{source}--file_list.json"
    if await aios.path.exists(cache_file):
        async with aiofiles.open(cache_file, 'r') as f: return json.loads(await f.read())
    file_list = await fetch_file_list_with_retry(repo_id, revision, source=source)
    await aios.makedirs(cache_file.parent, exist_ok=True)
    async with aiofiles.open(cache_file, 'w') as f: await f.write(json.dumps(file_list))
    return file_list

# 带重试机制获取文件列表，最多尝试30次，每次失败后等待一段时间再重试
async def fetch_file_list_with_retry(repo_id: str, revision: str = "main", path: str = "", source: str = "modelscope") -> List[Dict[str, Union[str, int]]]:
    n_attempts = 30
    for attempt in range(n_attempts):
        try:
            if source == "modelscope":
                return await _fetch_file_list_modelscope(repo_id, revision, path)
            else:
                return await _fetch_file_list(repo_id, revision, path)
        except Exception as e:
            if attempt == n_attempts - 1: raise e
            await asyncio.sleep(min(8, 0.1 * (2 ** attempt)))

# 从Hugging Face获取文件列表
async def _fetch_file_list(repo_id: str, revision: str = "main", path: str = "") -> List[Dict[str, Union[str, int]]]:
    api_url = f"{get_hf_endpoint()}/api/models/{repo_id}/tree/{revision}"
    url = f"{api_url}/{path}" if path else api_url

    headers = await get_auth_headers()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30, connect=10, sock_read=30, sock_connect=10)) as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                files = []
                for item in data:
                    if item["type"] == "file":
                        files.append({"path": item["path"], "size": item["size"]})
                    elif item["type"] == "directory":
                        subfiles = await _fetch_file_list(repo_id, revision, item["path"])
                        files.extend(subfiles)
                return files
            else:
                raise Exception(f"Failed to fetch file list: {response.status}")

# 在 new_shard_download_ms.py 中需要修改的函数
# 从ModelScope获取文件列表
async def _fetch_file_list_modelscope(repo_id: str, revision: str = "master", path: str = "") -> List[
    Dict[str, Union[str, int]]]:
    """从ModelScope获取文件列表"""
    # 首先检查模型是否存在
    if not await test_modelscope_model_exists(repo_id, revision):
        raise Exception(f"Model {repo_id} not found on ModelScope")

    # 使用ModelScope SDK获取文件列表（如果可用）
    if await check_modelscope_sdk():
        try:
            from modelscope.hub.api import HubApi
            api = HubApi()
            # 获取文件树
            files = api.get_model_files(model_id=repo_id, revision=revision)
            result = []
            if isinstance(files, list):
                for file_info in files:
                    if isinstance(file_info, dict):
                        file_type = file_info.get('Type', '')
                        if file_type == 'blob':  # 文件
                            result.append({
                                "path": file_info.get('Path', ''),
                                "size": file_info.get('Size', 0)
                            })
            if result:
                return result
        except Exception as e:
            if DEBUG >= 2:
                print(f"SDK方式获取文件列表失败: {e}")

    # 回退到API方式
    endpoints_to_try = [
        f"{get_modelscope_endpoint()}/models/{repo_id}/repo/files",
        f"{get_modelscope_endpoint()}/models/{repo_id}/tree",
        f"{get_modelscope_endpoint()}/models/{repo_id}/repo"
    ]

    params = {"Revision": revision}
    if path:
        params["Path"] = path

    headers = await get_modelscope_auth_headers()

    for api_url in endpoints_to_try:
        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30, connect=10, sock_read=30, sock_connect=10)) as session:
                async with session.get(api_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        files = []

                        # 解析文件列表
                        file_items = []
                        if isinstance(data, dict):
                            if "Data" in data and isinstance(data["Data"], dict) and "Files" in data["Data"]:
                                file_items = data["Data"]["Files"]
                            elif "Data" in data and isinstance(data["Data"], list):
                                file_items = data["Data"]
                            elif "files" in data:
                                file_items = data["files"]
                            elif "tree" in data:
                                file_items = data["tree"]

                        for item in file_items:
                            if isinstance(item, dict):
                                item_type = item.get("Type", "") or item.get("type", "")
                                item_path = item.get("Path", "") or item.get("path", "")
                                item_size = item.get("Size", 0) or item.get("size", 0)

                                if item_type.lower() in ["file", "blob"]:
                                    files.append({"path": item_path, "size": int(item_size)})
                                elif item_type.lower() in ["directory", "tree"] and item_path != path:
                                    try:
                                        subfiles = await _fetch_file_list_modelscope(repo_id, revision, item_path)
                                        files.extend(subfiles)
                                    except Exception:
                                        pass
                        return files
        except Exception as e:
            if DEBUG >= 2:
                print(f"API端点 {api_url} 失败: {e}")

    # 如果所有方法都失败，返回空列表而不是抛出异常
    if DEBUG >= 2:
        print(f"无法从ModelScope获取 {repo_id} 的文件列表")
    return []

# 获取ModelScope文件元数据，包含文件大小和ETag
async def file_meta_modelscope(repo_id: str, revision: str, path: str) -> Tuple[int, str]:
    """获取ModelScope文件元数据"""
    # 首先检查模型是否存在
    if not await test_modelscope_model_exists(repo_id, revision):
        raise Exception(f"Model {repo_id} not found on ModelScope")

    # 尝试使用不同的API端点
    endpoints_to_try = [
        {
            "url": f"{get_modelscope_endpoint()}/models/{repo_id}/repo",
            "params": {"Revision": revision, "FilePath": path}
        }
    ]

    headers = await get_modelscope_auth_headers()

    for endpoint_config in endpoints_to_try:
        api_url = endpoint_config["url"]
        params = endpoint_config["params"]

        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30, connect=10, sock_read=30, sock_connect=10)) as session:
                async with session.head(api_url, headers=headers, params=params) as r:
                    if r.status == 200:
                        content_length = int(r.headers.get('content-length') or 0)
                        etag = (r.headers.get('ETag') or
                                r.headers.get('Digest') or
                                r.headers.get('Content-MD5') or
                                r.headers.get('x-oss-hash-crc64ecma'))

                        if content_length > 0:
                            if etag is None:
                                last_modified = r.headers.get('Last-Modified', '')
                                etag = f"{repo_id}/{path}-{revision}-{content_length}-{last_modified}"

                            if etag and ((etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'")):
                                etag = etag[1:-1]
                            return content_length, etag
        except Exception as e:
            if DEBUG >= 2:
                print(f"HEAD请求到 {api_url} 失败: {e}")

    # 如果HEAD失败，尝试GET
    for endpoint_config in endpoints_to_try:
        api_url = endpoint_config["url"]
        params = endpoint_config["params"]

        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30, connect=10, sock_read=30, sock_connect=10)) as session:
                async with session.get(api_url, headers=headers, params=params) as r:
                    if r.status == 200:
                        content_length = int(r.headers.get('content-length') or 0)
                        etag = (r.headers.get('ETag') or
                                r.headers.get('Digest') or
                                r.headers.get('Content-MD5') or
                                r.headers.get('x-oss-hash-crc64ecma'))

                        if content_length > 0:
                            if etag is None:
                                last_modified = r.headers.get('Last-Modified', '')
                                etag = f"{repo_id}/{path}-{revision}-{content_length}-{last_modified}"

                            if etag and ((etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'")):
                                etag = etag[1:-1]
                            return content_length, etag
        except Exception as e:
            if DEBUG >= 2:
                print(f"GET请求到 {api_url} 失败: {e}")

    raise Exception(f"Failed to fetch metadata for {path} from ModelScope repo {repo_id}")

# 从ModelScope下载单个文件
async def _download_file_modelscope(repo_id: str, revision: str, path: str, target_dir: Path,
                                    on_progress: Callable[[int, int], None] = lambda _, __: None) -> Path:
    """从ModelScope下载文件"""
    # 首先确保目标目录存在
    if await aios.path.exists(target_dir / path):
        return target_dir / path
    await aios.makedirs((target_dir / path).parent, exist_ok=True)

    # 获取文件元数据
    try:
        length, etag = await file_meta_modelscope(repo_id, revision, path)
    except Exception as e:
        raise FileNotFoundError(f"Cannot get metadata for file {path} in repo {repo_id}: {e}")

    remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
    partial_path = target_dir / f"{path}.partial"
    resume_byte_pos = (await aios.stat(partial_path)).st_size if (await aios.path.exists(partial_path)) else None

    # 尝试下载文件
    endpoints_to_try = [
        {
            "url": f"{get_modelscope_endpoint()}/models/{repo_id}/repo",
            "params": {"Revision": revision, "FilePath": path}
        }
    ]

    download_successful = False
    for endpoint_config in endpoints_to_try:
        url = endpoint_config["url"]
        params = endpoint_config["params"]

        try:
            headers = await get_modelscope_auth_headers()
            if resume_byte_pos:
                headers['Range'] = f'bytes={resume_byte_pos}-'
            n_read = resume_byte_pos or 0

            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as session:
                async with session.get(url, headers=headers, params=params,
                                       timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800,
                                                                     sock_connect=60)) as r:
                    if r.status == 404:
                        continue
                    elif r.status not in [200, 206]:
                        continue

                    async with aiofiles.open(partial_path, 'ab' if resume_byte_pos else 'wb') as f:
                        while True:
                            chunk = await r.content.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            n_read += await f.write(chunk)
                            on_progress(n_read, length)

                    download_successful = True
                    break
        except Exception as e:
            if DEBUG >= 2:
                print(f"从 {url} 下载失败: {e}")
            continue

    if not download_successful:
        raise FileNotFoundError(f"File not found or cannot be downloaded: {repo_id}/{path}")

    await aios.rename(partial_path, target_dir / path)
    return target_dir / path

# 添加基于ModelScope SDK的完整下载支持
# 使用ModelScope SDK下载分片
async def download_shard_with_modelscope_sdk(shard: Shard, inference_engine_classname: str,
                                             on_progress: AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]],
                                             repo_id: str, revision: str = "master") -> tuple[Path, RepoProgressEvent]:
    """使用ModelScope SDK下载分片"""
    try:
        # 直接使用更新后的download_with_modelscope_sdk函数
        target_dir = await ensure_downloads_dir() / repo_id.replace("/", "--")
        await aios.makedirs(target_dir, exist_ok=True)

        # 获取允许的文件模式
        allow_patterns = await resolve_allow_patterns(shard, inference_engine_classname, source=source, revision=revision)

        # 使用更新后的下载函数，传递local_dir参数指向目标目录
        model_dir = await download_with_modelscope_sdk(
            repo_id, revision,
            allow_patterns=allow_patterns if allow_patterns != ["*"] else None,
            local_dir=str(target_dir)  # 传递local_dir参数，直接下载到目标目录
        )
        
        # 由于使用cache_dir，我们需要确保返回的是目标目录
        # 将下载的模型复制到目标目录
        if model_dir and Path(model_dir).exists():
            # 确保目标目录存在
            await aios.makedirs(target_dir, exist_ok=True)
            
            # 如果模型目录与目标目录不同，则复制文件
            if Path(model_dir) != target_dir:
                for item in Path(model_dir).iterdir():
                    dest_path = target_dir / item.name
                    if item.is_dir():
                        if await aios.path.exists(dest_path):
                            await asyncio.to_thread(shutil.rmtree, str(dest_path), ignore_errors=True)
                        await asyncio.to_thread(shutil.copytree, str(item), str(dest_path))
                    else:
                        await asyncio.to_thread(shutil.copy2, str(item), str(dest_path))

        # 创建RepoProgressEvent
        progress_event = RepoProgressEvent(
            shard, repo_id, revision, 1, 1, 0, 0, 0, 0,
            timedelta(0), {}, "complete"
        )

        return Path(target_dir), progress_event

    except Exception as e:
        raise Exception(f"使用ModelScope SDK下载失败: {e}")

# 添加基于ModelScope SDK的下载方法作为备选
# 使用ModelScope SDK作为备选下载方式
async def download_with_modelscope_sdk_fallback(repo_id: str, revision: str, target_dir: Path,
                                                allow_patterns: Optional[List[str]] = None) -> bool:
    """使用ModelScope SDK作为备选下载方式"""
    try:
        # 检查是否安装了ModelScope SDK
        if not await check_modelscope_sdk():
            if DEBUG >= 2:
                print("ModelScope SDK not installed, trying to install...")
            if not await install_modelscope_sdk():
                if DEBUG >= 2:
                    print("Failed to install ModelScope SDK")
                return False
            if DEBUG >= 2:
                print("ModelScope SDK installed successfully")

        from modelscope.hub.snapshot_download import snapshot_download

        # 准备下载参数
        download_kwargs = {
            "model_id": repo_id,
            "revision": revision,
            "cache_dir": str(target_dir)
        }

        if allow_patterns and allow_patterns != ["*"]:
            download_kwargs["allow_patterns"] = allow_patterns

        # 执行下载（放到线程池，不阻塞事件循环）
        model_dir = await asyncio.to_thread(snapshot_download, **download_kwargs)
        if DEBUG >= 2:
            print(f"ModelScope SDK download completed: {model_dir}")
        return True

    except Exception as e:
        if DEBUG >= 2:
            print(f"ModelScope SDK download failed: {e}")
            import traceback
            traceback.print_exc()
        return False

# 计算指定文件的哈希值，支持 sha1 和 sha256 两种算法
async def calc_hash(path: Path, type: Literal["sha1", "sha256"] = "sha1") -> str:
    hash = hashlib.sha1() if type == "sha1" else hashlib.sha256()
    if type == "sha1":
        header = f"blob {(await aios.stat(path)).st_size}\0".encode()
        hash.update(header)
    async with aiofiles.open(path, 'rb') as f:
        while chunk := await f.read(8 * 1024 * 1024):
            hash.update(chunk)
    return hash.hexdigest()

# 获取Hugging Face文件的元数据，包含文件大小和ETag
async def file_meta(repo_id: str, revision: str, path: str) -> Tuple[int, str]:
    url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
    headers = await get_auth_headers()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as session:
        async with session.head(url, headers=headers) as r:
            content_length = int(r.headers.get('x-linked-size') or r.headers.get('content-length') or 0)
            etag = r.headers.get('X-Linked-ETag') or r.headers.get('ETag') or r.headers.get('Etag')
            assert content_length > 0, f"No content length for {url}"
            assert etag is not None, f"No remote hash for {url}"
            if (etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'"): etag = etag[1:-1]
            return content_length, etag

# 带重试机制下载文件，最多尝试30次，每次失败后等待一段时间再重试
async def download_file_with_retry(repo_id: str, revision: str, path: str, target_dir: Path,
                                  on_progress: Callable[[int, int], None] = lambda _, __: None,
                                  source: str = "modelscope") -> Path:
    n_attempts = 30
    for attempt in range(n_attempts):
        try:
            if source == "modelscope":
                return await _download_file_modelscope(repo_id, revision, path, target_dir, on_progress)
            else:
                return await _download_file(repo_id, revision, path, target_dir, on_progress)
        except Exception as e:
            if isinstance(e, FileNotFoundError) or attempt == n_attempts - 1: raise e
            print(f"Download error on attempt {attempt}/{n_attempts} for {repo_id=} {revision=} {path=} {target_dir=}")
            traceback.print_exc()
            await asyncio.sleep(min(8, 0.1 * (2 ** attempt)))

# 从Hugging Face下载单个文件
async def _download_file(repo_id: str, revision: str, path: str, target_dir: Path, on_progress: Callable[[int, int], None] = lambda _, __: None) -> Path:
    if await aios.path.exists(target_dir/path): return target_dir/path
    await aios.makedirs((target_dir/path).parent, exist_ok=True)
    length, etag = await file_meta(repo_id, revision, path)
    remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
    partial_path = target_dir/f"{path}.partial"
    resume_byte_pos = (await aios.stat(partial_path)).st_size if (await aios.path.exists(partial_path)) else None
    if resume_byte_pos != length:
        url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
        headers = await get_auth_headers()
        if resume_byte_pos: headers['Range'] = f'bytes={resume_byte_pos}-'
        n_read = resume_byte_pos or 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as r:
                if r.status == 404: raise FileNotFoundError(f"File not found: {url}")
                assert r.status in [200, 206], f"Failed to download {path} from {url}: {r.status}"
                async with aiofiles.open(partial_path, 'ab' if resume_byte_pos else 'wb') as f:
                    while chunk := await r.content.read(8 * 1024 * 1024): on_progress(n_read := n_read + await f.write(chunk), length)

    final_hash = await calc_hash(partial_path, type="sha256" if len(remote_hash) == 64 else "sha1")
    integrity = final_hash == remote_hash
    if not integrity:
        try: await aios.remove(partial_path)
        except Exception as e: print(f"Error removing partial file {partial_path}: {e}")
        raise Exception(f"Downloaded file {target_dir/path} has hash {final_hash} but remote hash is {remote_hash}")
    await aios.rename(partial_path, target_dir/path)
    return target_dir/path

# 计算仓库的下载进度
async def calculate_repo_progress(shard: Shard, repo_id: str, repo_revision: str, file_progress: Dict[str, RepoFileProgressEvent], all_start_time: float) -> RepoProgressEvent:
    all_total_bytes = sum([p.total for p in file_progress.values()])
    all_downloaded_bytes = sum([p.downloaded for p in file_progress.values()])
    all_downloaded_bytes_this_session = sum([p.downloaded_this_session for p in file_progress.values()])
    elapsed_time = time.time() - all_start_time
    all_speed = all_downloaded_bytes_this_session / elapsed_time if elapsed_time > 0 else 0
    all_eta = timedelta(seconds=(all_total_bytes - all_downloaded_bytes) / all_speed) if all_speed > 0 else timedelta(seconds=0)
    
    # 新增：直接检查关键文件是否存在，作为额外的完成状态判断条件
    key_files_exist = False
    
    try:
        # 使用项目中已有的函数获取正确的下载目录
        from exo.download.new_shard_download import exo_home
        
        # 获取下载根目录
        downloads_dir = exo_home() / "downloads"
        
        # 构建可能的模型目录路径列表
        model_dir_candidates = [
            # 标准路径格式
            downloads_dir / repo_id.replace("/", "--"),
            # ModelScope SDK可能创建的嵌套目录格式
            downloads_dir / repo_id.replace("/", "--") / repo_id.replace("/", "/"),
            # Qwen模型特殊格式
            downloads_dir / repo_id.replace("/", "--") / repo_id.split("/")[-1],
            # 最内层目录格式
            downloads_dir / repo_id.replace("/", "--") / repo_id,
        ]
        
        # 检查常见的模型关键文件
        key_files = ["model.safetensors", "model.bin", "pytorch_model.bin", "tokenizer.json", "config.json"]
        
        # 遍历所有可能的目录和文件组合
        for model_dir in model_dir_candidates:
            if model_dir.exists():
                # 直接检查目录下的关键文件
                for key_file in key_files:
                    if (model_dir / key_file).exists():
                        key_files_exist = True
                        break
                
                # 如果直接目录下没有找到，检查所有子目录
                if not key_files_exist:
                    import shutil
                    # 使用asyncio.to_thread执行同步的shutil.walk
                    walk_result = await asyncio.to_thread(shutil.walk, model_dir)
                    for root, dirs, files in walk_result:

                        for key_file in key_files:
                            if key_file in files:
                                key_files_exist = True
                                break
                        if key_files_exist:
                            break
            
            if key_files_exist:
                break
        
    except Exception as e:
        if DEBUG >= 1:
            print(f"检查模型文件存在性时出错: {e}")
    
    # 原始状态判断逻辑
    status = "complete" if all(p.status == "complete" for p in file_progress.values()) else "in_progress" if any(p.status == "in_progress" for p in file_progress.values()) else "not_started"
    
    # 如果关键文件存在，强制将状态设为complete
    if key_files_exist:
        status = "complete"
        all_downloaded_bytes = all_total_bytes
        all_total_bytes = max(all_total_bytes, 1)  # 确保total_bytes不为0
    
    return RepoProgressEvent(shard, repo_id, repo_revision, len([p for p in file_progress.values() if p.downloaded == p.total]), len(file_progress), all_downloaded_bytes, all_downloaded_bytes_this_session, all_total_bytes, all_speed, all_eta, file_progress, status)


# 枚举 repo_id 可能存在的本地目录位置。
# ModelScope SDK 使用 cache_dir 时可能创建 HuggingFace 风格的缓存结构：
#   downloads/models/<repo_id--name>/snapshots/<revision>/
# 同时兼容 exo 内部命名的平级目录：
#   downloads/<repo_id--name>/
#   downloads/<repo_id>/
async def _candidate_local_model_dirs(repo_id: str, revision: Optional[str] = None) -> List[Path]:
    downloads_dir = await ensure_downloads_dir()
    local_dir_name = repo_id.replace("/", "--")
    revisions = [revision] if revision else ["master", "main"]

    candidates: List[Path] = [
        downloads_dir / local_dir_name,
        downloads_dir / repo_id,
    ]
    for rev in revisions:
        candidates.extend([
            downloads_dir / "models" / local_dir_name / "snapshots" / rev,
            downloads_dir / "models" / repo_id / "snapshots" / rev,
        ])
    return candidates


# 获取模型的权重映射文件内容
# 优先读取本地缓存中的 index 文件；本地不存在时，按指定 source/revision 下载。
# HTTP 下载失败时会回退到 ModelScope SDK 单文件下载。
async def get_weight_map(repo_id: str, revision: str = "master", source: str = "modelscope") -> Dict[str, str]:
    cache_dirs = await _candidate_local_model_dirs(repo_id, revision)
    tmp_dir = (await ensure_exo_tmp()) / repo_id.replace("/", "--")

    # 1) 优先检查本地 downloads 目录中的 index 文件（兼容多种目录格式）
    index_candidates = ["model.safetensors.index.json", "pytorch_model.bin.index.json"]
    for cache_dir in cache_dirs:
        for index_name in index_candidates:
            local_index = cache_dir / index_name
            if await aios.path.exists(local_index):
                try:
                    async with aiofiles.open(local_index, 'r') as f:
                        index_data = json.loads(await f.read())
                    weight_map = index_data.get("weight_map")
                    if weight_map:
                        if DEBUG >= 2:
                            print(f"[get_weight_map] 从本地缓存读取 weight_map: {local_index} ({len(weight_map)} 个张量)")
                        return weight_map
                except Exception as e:
                    if DEBUG >= 1:
                        print(f"[get_weight_map] 读取本地 index 失败 {local_index}: {e}")

    # 2) 本地没有，尝试用 HTTP API 下载 index 文件
    last_error = None
    for index_name in index_candidates:
        try:
            index_file = await download_file_with_retry(repo_id, revision, index_name, tmp_dir, source=source)
            async with aiofiles.open(index_file, 'r') as f:
                index_data = json.loads(await f.read())
            weight_map = index_data.get("weight_map")
            if weight_map:
                if DEBUG >= 2:
                    print(f"[get_weight_map] 从 {source} 下载 {index_name} 成功 ({len(weight_map)} 个张量)")
                return weight_map
        except Exception as e:
            last_error = e
            if DEBUG >= 2:
                print(f"[get_weight_map] HTTP 下载 {index_name} 失败: {e}")

    # 3) HTTP 失败时回退到 ModelScope SDK 单文件下载
    if source == "modelscope":
        for index_name in index_candidates:
            try:
                print(f"[get_weight_map] 尝试使用 ModelScope SDK 下载 {index_name}...")
                index_file = await download_file_with_modelscope_sdk(repo_id, index_name, revision=revision, local_dir=str(tmp_dir))
                async with aiofiles.open(index_file, 'r') as f:
                    index_data = json.loads(await f.read())
                weight_map = index_data.get("weight_map")
                if weight_map:
                    if DEBUG >= 2:
                        print(f"[get_weight_map] 从 ModelScope SDK 下载 {index_name} 成功 ({len(weight_map)} 个张量)")
                    return weight_map
            except Exception as e:
                last_error = e
                if DEBUG >= 2:
                    print(f"[get_weight_map] ModelScope SDK 下载 {index_name} 失败: {e}")

    raise Exception(f"无法获取 {repo_id} 的 weight_map (revision={revision}, source={source}): {last_error}")

# 解析允许下载的文件模式
async def resolve_allow_patterns(shard: Shard, inference_engine_classname: str, source: str = "modelscope", revision: str = "master") -> List[str]:
    try:
        repo_id = get_repo(shard.model_id, inference_engine_classname)
        if repo_id is None:
            # 兼容：从 model_cards 中查找 repo 配置
            from exo.models import model_cards
            fallback_info = model_cards.get(shard.model_id, {}).get("repo", {})
            repo_id = next(iter(fallback_info.values()), shard.model_id)

        weight_map = await get_weight_map(repo_id, revision=revision, source=source)
        allow_patterns = get_allow_patterns(weight_map, shard)
        if DEBUG >= 2:
            print(f"[resolve_allow_patterns] {shard.model_id=} {shard.start_layer=}-{shard.end_layer=} -> {allow_patterns=}")
        return allow_patterns
    except Exception as e:
        if DEBUG >= 1:
            print(f"[resolve_allow_patterns] 获取 weight_map 失败 {shard.model_id=} {inference_engine_classname=}: {e}")
        if DEBUG >= 1:
            traceback.print_exc()
        print(f"⚠️ 无法解析分片权重映射，将下载全部文件: {shard.model_id=} ({e})")
        return ["*"]

# 获取已下载文件的大小，优先检查完整文件，若不存在则检查部分下载文件
async def get_downloaded_size(path: Path) -> int:
    partial_path = path.with_suffix(path.suffix + ".partial")
    if await aios.path.exists(path): return (await aios.stat(path)).st_size
    if await aios.path.exists(partial_path): return (await aios.stat(partial_path)).st_size
    return 0

# 下载分片，支持使用ModelScope SDK
async def check_local_model_exists(repo_id: str, target_dir: Path) -> bool:
    """检查本地是否已存在完整的模型文件（必须包含权重文件）"""
    try:
        if not await aios.path.exists(target_dir):
            return False
        
        async def _check_dir_for_weights(dir_path: Path) -> bool:
            """检查目录是否包含权重文件"""
            has_weights = False
            
            safetensors_files = list(dir_path.glob("*.safetensors"))
            if safetensors_files:
                has_weights = True
            
            bin_files = list(dir_path.glob("*.bin"))
            if bin_files:
                has_weights = True
            
            gguf_files = list(dir_path.glob("*.gguf"))
            if gguf_files:
                has_weights = True
            
            index_file = dir_path / "model.safetensors.index.json"
            if await aios.path.exists(index_file):
                has_weights = True
            
            return has_weights
        
        if await _check_dir_for_weights(target_dir):
            return True
        
        try:
            entries = await aios.listdir(target_dir)
            for entry in entries:
                entry_path = target_dir / entry
                try:
                    is_dir = await aios.path.isdir(entry_path)
                    if is_dir:
                        if await _check_dir_for_weights(entry_path):
                            return True
                except:
                    continue
        except:
            pass
        
        return False
    except Exception as e:
        if DEBUG >= 1:
            print(f"检查本地模型存在性时出错: {e}")
        return False


async def download_shard(shard: Shard, inference_engine_classname: str,
                         on_progress: AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]],
                         max_parallel_downloads: int = 8,
                         skip_download: bool = False,
                         source: str = "modelscope",
                         use_sdk: bool = True) -> tuple[Path, RepoProgressEvent]:
    """下载分片，支持使用ModelScope SDK"""

    if DEBUG >= 2 and not skip_download:
        print(f"Downloading {shard.model_id=} for {inference_engine_classname} using {source}")
    
    safe_model_id = shard.model_id.split("::")[0] if "::" in shard.model_id else shard.model_id
    
    repo_id = get_repo(safe_model_id, inference_engine_classname)

    if repo_id is None:
        from exo.models import model_cards
        fallback_info = model_cards.get(safe_model_id, {}).get("repo", {})
        if fallback_info:
            repo_id = next(iter(fallback_info.values()), safe_model_id)
        else:
            repo_id = safe_model_id
    revision = "master"
    target_dir = await ensure_downloads_dir() / repo_id.replace("/", "--")
    file_progress: Dict[str, RepoFileProgressEvent] = {}
    
    # 检查本地是否已存在模型，如果存在则跳过下载
    if not skip_download:
        local_exists = await check_local_model_exists(repo_id, target_dir)
        if local_exists:
            if DEBUG >= 1:
                print(f"本地模型已存在，跳过下载: {target_dir}")
            skip_download = True

    # 初始化进度事件并触发
    initial_progress = RepoProgressEvent(
        shard=shard,
        repo_id=repo_id,
        repo_revision=revision,
        completed_files=0,
        total_files=1,  # 假设至少有一个文件
        downloaded_bytes=0,
        downloaded_bytes_this_session=0,
        total_bytes=0,
        overall_speed=0,
        overall_eta=timedelta(seconds=0),
        file_progress={},
        status="in_progress"
    )
    # 修复：移除await，因为trigger是同步方法
    on_progress.trigger("progress", (shard, initial_progress))

    if source == "modelscope" and use_sdk:
        # 使用ModelScope SDK下载
        if not skip_download:
            try:
                # 获取允许的文件模式
                allow_patterns = await resolve_allow_patterns(shard, inference_engine_classname, source=source, revision=revision)

                print(f"使用ModelScope SDK下载模型: {repo_id}")
                model_dir = await download_with_modelscope_sdk(
                    repo_id, revision,
                    allow_patterns=allow_patterns if allow_patterns != ["*"] else None,
                    local_dir=str(target_dir)
                )
                print(f"ModelScope SDK下载完成: {model_dir}")

                # 继续获取文件列表
                file_list = await fetch_file_list_with_cache(repo_id, revision, source)
                filtered_file_list = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, key=lambda x: x["path"]))

                # 计算实际的下载进度
                all_start_time = time.time()
                for file in filtered_file_list:
                    downloaded_bytes = await get_downloaded_size(target_dir / file["path"])
                    file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], downloaded_bytes, 0,
                                                                        file["size"], 0, timedelta(0),
                                                                        "complete" if downloaded_bytes == file["size"] else "not_started", time.time())

                # 计算最终进度并触发
                final_repo_progress = await calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time)
                on_progress.trigger("progress", (shard, final_repo_progress))
                
                if gguf := next((f for f in filtered_file_list if f["path"].endswith(".gguf")), None):
                    return target_dir / gguf["path"], final_repo_progress
                else:
                    return target_dir, final_repo_progress

            except Exception as e:
                print(f"ModelScope SDK下载失败，回退到API方式: {e}")
                # 回退到原来的API方式
                pass

    # 如果 skip_download=True（本地模型已存在），直接返回本地路径
    if skip_download:
        if DEBUG >= 1:
            print(f"本地模型已存在，直接返回: {target_dir}")
        
        # 计算本地模型文件的真实大小
        total_bytes = 0
        completed_files = 0
        file_progress = {}
        all_start_time = time.time()
        
        if await aios.path.exists(target_dir):
            try:
                # 遍历目录中的所有文件
                for root, dirs, files in os.walk(target_dir):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            file_size = (await aios.stat(file_path)).st_size
                            relative_path = str(file_path.relative_to(target_dir))
                            total_bytes += file_size
                            completed_files += 1
                            file_progress[relative_path] = RepoFileProgressEvent(
                                repo_id, revision, relative_path, 
                                file_size, 0, file_size, 0, timedelta(0), "complete", all_start_time
                            )
                        except Exception as e:
                            if DEBUG >= 2:
                                print(f"获取文件大小失败 {file_path}: {e}")
            except Exception as e:
                if DEBUG >= 1:
                    print(f"遍历模型目录失败 {target_dir}: {e}")
        
        # 如果没有找到任何文件，返回0表示未下载
        if total_bytes == 0:
            final_repo_progress = RepoProgressEvent(
                shard=shard,
                repo_id=repo_id,
                repo_revision=revision,
                completed_files=0,
                total_files=0,
                downloaded_bytes=0,
                downloaded_bytes_this_session=0,
                total_bytes=0,
                overall_speed=0,
                overall_eta=timedelta(seconds=0),
                file_progress={},
                status="not_started"
            )
            on_progress.trigger("progress", (shard, final_repo_progress))
            return target_dir, final_repo_progress
        
        # 创建进度事件，使用真实的文件大小
        final_repo_progress = RepoProgressEvent(
            shard=shard,
            repo_id=repo_id,
            repo_revision=revision,
            completed_files=completed_files,
            total_files=completed_files,
            downloaded_bytes=total_bytes,
            downloaded_bytes_this_session=0,
            total_bytes=total_bytes,
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            file_progress=file_progress,
            status="complete"
        )
        on_progress.trigger("progress", (shard, final_repo_progress))
        return target_dir, final_repo_progress

    # 原有的下载逻辑
    if not skip_download:
        await aios.makedirs(target_dir, exist_ok=True)

    if repo_id is None:
        raise ValueError(f"No repo found for {shard.model_id=} and inference engine {inference_engine_classname}")

    allow_patterns = await resolve_allow_patterns(shard, inference_engine_classname, source=source, revision=revision)
    if DEBUG >= 2:
        print(f"Downloading {shard.model_id=} with {allow_patterns=}")

    all_start_time = time.time()
    file_list = await fetch_file_list_with_cache(repo_id, revision, source)
    filtered_file_list = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, key=lambda x: x["path"]))

    if skip_download and not filtered_file_list:
        # 如果过滤后的文件列表为空，尝试使用不同的方式获取文件列表
        if DEBUG >= 1: print(f"尝试使用备用方式获取文件列表 for {repo_id}")
        # 尝试不使用过滤模式重新获取文件列表
        try:
            unfiltered_file_list = await fetch_file_list_with_cache(repo_id, revision, source)
            # 至少需要有一些文件来计算大小
            if unfiltered_file_list:
                # 为状态报告创建一个虚拟文件进度
                dummy_file = unfiltered_file_list[0] if unfiltered_file_list else {"path": "dummy", "size": 1}
                downloaded_bytes = await get_downloaded_size(target_dir / dummy_file["path"])
                file_progress[dummy_file["path"]] = RepoFileProgressEvent(repo_id, revision, dummy_file["path"], 
                                                                         downloaded_bytes, 0, dummy_file["size"], 
                                                                         0, timedelta(0),
                                                                         "complete" if downloaded_bytes == dummy_file["size"] else "not_started", 
                                                                         time.time())
        except Exception as e:
            if DEBUG >= 1: print(f"备用方式获取文件列表失败: {e}")
            # 如果都失败了，至少创建一个虚拟文件进度来避免total_size=0
            file_progress["dummy"] = RepoFileProgressEvent(repo_id, revision, "dummy", 0, 0, 1,
                                                          0, timedelta(0), "not_started", time.time())

    def on_progress_wrapper(file: dict, curr_bytes: int, total_bytes: int):
        start_time = file_progress[file["path"]].start_time if file["path"] in file_progress else time.time()
        downloaded_this_session = file_progress[file["path"]].downloaded_this_session + (
                    curr_bytes - file_progress[file["path"]].downloaded) if file[
                                                                                "path"] in file_progress else curr_bytes
        speed = downloaded_this_session / (time.time() - start_time) if time.time() - start_time > 0 else 0
        eta = timedelta(seconds=(total_bytes - curr_bytes) / speed) if speed > 0 else timedelta(seconds=0)
        file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], curr_bytes,
                                                            downloaded_this_session, total_bytes, speed, eta,
                                                            "complete" if curr_bytes == total_bytes else "in_progress",
                                                            start_time)
        # 修复：使用正确的trigger_all方法，移除await
        on_progress.trigger("progress", (shard, calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time)))
        if DEBUG >= 6:
            print(f"Downloading {file['path']} {curr_bytes}/{total_bytes} {speed} {eta}")

    for file in filtered_file_list:
        downloaded_bytes = await get_downloaded_size(target_dir / file["path"])
        file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], downloaded_bytes, 0,
                                                            file["size"], 0, timedelta(0),
                                                            "complete" if downloaded_bytes == file[
                                                                "size"] else "not_started", time.time())

    semaphore = asyncio.Semaphore(max_parallel_downloads)

    async def download_with_semaphore(file):
        async with semaphore:
            await download_file_with_retry(repo_id, revision, file["path"], target_dir,
                                           lambda curr_bytes, total_bytes: on_progress_wrapper(file, curr_bytes,
                                                                                               total_bytes),
                                           source)

    if not skip_download:
        await asyncio.gather(*[download_with_semaphore(file) for file in filtered_file_list])
    
    # 计算最终进度并触发
    final_repo_progress = await calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time)
    on_progress.trigger("progress", (shard, final_repo_progress))
    
    if gguf := next((f for f in filtered_file_list if f["path"].endswith(".gguf")), None):
        return target_dir / gguf["path"], final_repo_progress
    else:
        return target_dir, final_repo_progress

# 创建一个新的分片下载器实例
# 修复：添加use_sdk参数并默认设为True
def new_shard_downloader(max_parallel_downloads: int = 8, source: str = "modelscope", use_sdk: bool = True) -> ShardDownloader:
    return SingletonShardDownloader(CachedShardDownloader(NewShardDownloader(max_parallel_downloads, source, use_sdk))) 

# 单例分片下载器类，确保同一时间一个分片只有一个下载任务
class SingletonShardDownloader(ShardDownloader):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.active_downloads: Dict[Shard, asyncio.Task] = {}

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self.shard_downloader.on_progress

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        if shard not in self.active_downloads: self.active_downloads[shard] = asyncio.create_task(self.shard_downloader.ensure_shard(shard, inference_engine_name))
        try: return await self.active_downloads[shard]
        finally:
            if shard in self.active_downloads and self.active_downloads[shard].done(): del self.active_downloads[shard]

    async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
        async for path, status in self.shard_downloader.get_shard_download_status(inference_engine_name):
            yield path, status

# 带缓存的分片下载器类，缓存已下载的分片路径，避免重复下载
class CachedShardDownloader(ShardDownloader):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.cache: Dict[tuple[str, Shard], Path] = {}

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self.shard_downloader.on_progress

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        if (inference_engine_name, shard) in self.cache:
            if DEBUG >= 2: print(f"ensure_shard cache hit {shard=} for {inference_engine_name}")
            return self.cache[(inference_engine_name, shard)]
        if DEBUG >= 2: print(f"ensure_shard cache miss {shard=} for {inference_engine_name}")
        target_dir = await self.shard_downloader.ensure_shard(shard, inference_engine_name)
        self.cache[(inference_engine_name, shard)] = target_dir
        return target_dir

    async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
        async for path, status in self.shard_downloader.get_shard_download_status(inference_engine_name):
            yield path, status

# 新的分片下载器类，实现分片下载的核心逻辑
class NewShardDownloader(ShardDownloader):
    def __init__(self, max_parallel_downloads: int = 8, source: str = "modelscope", use_sdk: bool = True):
        self.max_parallel_downloads = max_parallel_downloads
        self.source = source
        self.use_sdk = use_sdk  # 添加SDK使用选项
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self._on_progress

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        target_dir, _ = await download_shard(shard, inference_engine_name, self.on_progress,
                                           max_parallel_downloads=self.max_parallel_downloads,
                                           source=self.source,
                                           use_sdk=self.use_sdk)  # 传递SDK选项
        return target_dir

    async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
        if DEBUG >= 2:
            print("Getting shard download status for", inference_engine_name)
        # 过滤掉不支持的模型（build_full_shard 返回 None）
        shards = []
        for model_id in get_supported_models([[inference_engine_name]]):
            shard = build_full_shard(model_id, inference_engine_name)
            if shard is not None:
                shards.append(shard)
            else:
                if DEBUG >= 2:
                    print(f"Skipping unsupported model: {model_id}")
        tasks = [download_shard(shard, inference_engine_name,
                              self.on_progress, skip_download=True, source=self.source)
                for shard in shards]
        for task in asyncio.as_completed(tasks):
            try:
                path, progress = await task
                yield (path, progress)
            except Exception as e:
                print("Error downloading shard:", e)