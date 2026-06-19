import os
import sys
import subprocess
import asyncio
import time
from pathlib import Path
from typing import Optional
from threading import Thread


class FRPProcessManager:
    """frp 进程管理器"""
    
    def __init__(self, frpc_path: Path, config_path: Path, auto_restart: bool = True):
        self.frpc_path = frpc_path
        self.config_path = config_path
        self.auto_restart = auto_restart
        self.process: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[Thread] = None
        self._restart_thread: Optional[Thread] = None
        self._stopped = False
    
    def start(self) -> bool:
        """启动 frpc 进程"""
        if self.process and self.process.poll() is None:
            print("[FRP] frpc 已经在运行中")
            return True
        
        if not self.frpc_path.exists():
            print(f"[FRP] 错误: frpc 可执行文件不存在: {self.frpc_path}")
            return False
        
        if not self.config_path.exists():
            print(f"[FRP] 错误: 配置文件不存在: {self.config_path}")
            return False
        
        try:
            print(f"[FRP] 正在启动 frpc...")
            print(f"[FRP] 配置文件: {self.config_path}")
            
            self.process = subprocess.Popen(
                [str(self.frpc_path), "-c", str(self.config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self._stopped = False
            self._monitor_thread = Thread(target=self._monitor_output, daemon=True)
            self._monitor_thread.start()
            
            if self.auto_restart:
                self._restart_thread = Thread(target=self._restart_loop, daemon=True)
                self._restart_thread.start()
            
            time.sleep(2)
            
            if self.process.poll() is None:
                print("[FRP] frpc 启动成功")
            else:
                print(f"[FRP] frpc 启动后退出，退出码: {self.process.returncode}")
                if self.auto_restart:
                    print("[FRP] 自动重连已启用，将在 5 秒后尝试重连...")
            
            return True
                
        except Exception as e:
            print(f"[FRP] 启动 frpc 时出错: {e}")
            if self.auto_restart:
                print("[FRP] 自动重连已启用，将在 5 秒后尝试重连...")
            return True
    
    def _monitor_output(self):
        """监控 frpc 输出"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if self._stopped:
                    break
                line = line.strip()
                if line:
                    print(f"[FRP] {line}")
        except Exception as e:
            if not self._stopped:
                print(f"[FRP] 监控输出出错: {e}")
    
    def _restart_loop(self):
        """自动重连循环"""
        while not self._stopped:
            time.sleep(5)
            
            if self._stopped:
                break
            
            if self.process and self.process.poll() is not None:
                print("[FRP] frpc 进程已退出，尝试重新连接...")
                self._do_restart()
    
    def _do_restart(self):
        """执行重启"""
        try:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            self.process = subprocess.Popen(
                [str(self.frpc_path), "-c", str(self.config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self._monitor_thread = Thread(target=self._monitor_output, daemon=True)
            self._monitor_thread.start()
            
            time.sleep(2)
            
            if self.process.poll() is None:
                print("[FRP] frpc 重连成功")
            else:
                print(f"[FRP] frpc 重连失败，退出码: {self.process.returncode}")
                
        except Exception as e:
            print(f"[FRP] 重连时出错: {e}")
    
    def stop(self) -> bool:
        """停止 frpc 进程"""
        self._stopped = True
        
        if not self.process:
            return True
        
        try:
            print("[FRP] 正在停止 frpc...")
            self.process.terminate()
            
            # 等待进程结束
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[FRP] 强制终止 frpc")
                self.process.kill()
                self.process.wait()
            
            print("[FRP] frpc 已停止")
            return True
        except Exception as e:
            print(f"[FRP] 停止 frpc 时出错: {e}")
            return False
    
    def is_running(self) -> bool:
        """检查 frpc 是否正在运行"""
        return self.process is not None and self.process.poll() is None
    
    def restart(self) -> bool:
        """重启 frpc"""
        self.stop()
        return self.start()
    
    def __del__(self):
        """析构函数，确保进程被清理"""
        self.stop()
