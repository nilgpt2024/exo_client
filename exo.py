#!/usr/bin/env python3
import argparse
import sys
import subprocess
import asyncio
import aiohttp
import json
from pathlib import Path
import os


def print_banner():
    banner = """
 ███████╗██╗  ██╗ ██████╗ 
 ██╔════╝╚██╗██╔╝██╔═══██╗
 █████╗   ╚███╔╝ ██║   ██║
 ██╔══╝   ██╔██╗ ██║   ██║
 ███████╗██╔╝ ██╗╚██████╔╝
 ╚══════╝╚═╝  ╚═╝ ╚═════╝ 
    """
    print(banner)


def get_config_dir():
    if sys.platform == "win32":
        return Path.home() / ".exo"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "exo"
    else:
        return Path.home() / ".config" / "exo"


def get_api_url():
    return os.environ.get("EXO_API_URL", "http://localhost:52415")


async def check_server_running():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{get_api_url()}/healthcheck", timeout=2) as resp:
                return resp.status == 200
    except:
        return False


def start_server():
    print("🚀 Starting exo server...")
    print_banner()
    try:
        subprocess.run([sys.executable, "-m", "exo.main"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


async def list_models():
    if not await check_server_running():
        print("❌ exo server is not running. Start it with: exo serve")
        return
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{get_api_url()}/v1/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("\n📦 Available models:\n")
                    for model in data.get("data", []):
                        print(f"  • {model['id']}")
                    print()
                else:
                    print(f"❌ Failed to list models: {resp.status}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def chat_with_model(model_name, prompt=None):
    if not await check_server_running():
        print("❌ exo server is not running. Start it with: exo serve")
        return
    
    if prompt is None:
        print(f"💬 Chatting with {model_name} (type 'exit' to quit)\n")
        messages = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                if not user_input:
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                print(f"{model_name}: ", end="", flush=True)
                full_response = ""
                
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "stream": True
                    }
                    async with session.post(f"{get_api_url()}/v1/chat/completions", json=payload) as resp:
                        if resp.status == 200:
                            async for line in resp.content:
                                line = line.decode("utf-8").strip()
                                if line.startswith("data: "):
                                    data_str = line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        delta = data.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            print(content, end="", flush=True)
                                            full_response += content
                                    except:
                                        pass
                        else:
                            print(f"\n❌ Error: {resp.status}")
                            error_text = await resp.text()
                            print(error_text)
                
                print("\n")
                messages.append({"role": "assistant", "content": full_response})
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    else:
        print(f"🤖 {model_name}: ", end="", flush=True)
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
            async with session.post(f"{get_api_url()}/v1/chat/completions", json=payload) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                            except:
                                pass
                else:
                    print(f"\n❌ Error: {resp.status}")
                    error_text = await resp.text()
                    print(error_text)
        print()


async def show_topology():
    if not await check_server_running():
        print("❌ exo server is not running. Start it with: exo serve")
        return
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{get_api_url()}/v1/topology") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("\n🌐 Network Topology:\n")
                    print(f"  Nodes: {len(data.get('nodes', {}))}")
                    print(f"  Connected peers: {len(data.get('peers', []))}\n")
                    
                    for node_id, node_info in data.get('nodes', {}).items():
                        print(f"  🖥️  {node_id}:")
                        print(f"      Device: {node_info.get('device_capabilities', {}).get('model', 'Unknown')}")
                        print(f"      Chip: {node_info.get('device_capabilities', {}).get('chip', 'Unknown')}")
                        memory = node_info.get('device_capabilities', {}).get('memory', 0)
                        print(f"      Memory: {memory / (1024**3):.1f} GB")
                        
                        loaded_models = data.get('node_loaded_models', {}).get(node_id, {})
                        if loaded_models:
                            print(f"      Loaded models: {', '.join(loaded_models.keys())}")
                        print()
                else:
                    print(f"❌ Failed to get topology: {resp.status}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="exo - Distributed inference framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  exo serve                    Start the exo server
  exo run qwen-3-0.6b         Chat with a model
  exo run qwen-3-0.6b "Hello" Run a single prompt
  exo list                     List available models
  exo status                   Show network topology
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the exo server")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run and chat with a model")
    run_parser.add_argument("model", help="Model name to run")
    run_parser.add_argument("prompt", nargs="?", help="Optional prompt to run (interactive mode if not provided)")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show network topology status")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        start_server()
    elif args.command == "run":
        asyncio.run(chat_with_model(args.model, args.prompt))
    elif args.command == "list":
        asyncio.run(list_models())
    elif args.command == "status":
        asyncio.run(show_topology())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
