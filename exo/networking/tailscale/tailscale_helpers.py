import json
import asyncio
import aiohttp
import shutil
import os
import platform
from typing import Dict, Any, List, Optional, Tuple
from exo.helpers import DEBUG_DISCOVERY
from datetime import datetime, timezone


def _find_tailscale_executable() -> str:
  """Find tailscale executable on the system (cross-platform support)

  Supports:
    - Windows: C:\\Program Files\\Tailscale\\tailscale.exe
    - Linux: /usr/bin/tailscale, /usr/local/bin/tailscale, etc.
    - macOS: /usr/local/bin/tailscale, /opt/homebrew/bin/tailscale
  """
  system = platform.system()
  possible_paths = []

  if system == "Windows":
    possible_paths = [
      r"C:\Program Files\Tailscale\tailscale.exe",
      r"C:\Program Files (x86)\Tailscale\tailscale.exe",
      os.path.expanduser(r"~\AppData\Local\Tailscale\tailscale.exe"),
    ]
  elif system == "Linux":
    possible_paths = [
      "/usr/bin/tailscale",
      "/usr/local/bin/tailscale",
      "/snap/bin/tailscale",
      "/usr/sbin/tailscale",
      "/opt/bin/tailscale",
      os.path.expanduser("~/.local/bin/tailscale"),
      os.path.expanduser("~/bin/tailscale"),
    ]
  elif system == "Darwin":  # macOS
    possible_paths = [
      "/usr/local/bin/tailscale",
      "/opt/homebrew/bin/tailscale",
      "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    ]

  for path in possible_paths:
    if os.path.exists(path) and os.access(path, os.X_OK):
      if DEBUG_DISCOVERY >= 3:
        print(f"[Tailscale] Found executable at: {path}")
      return path

  tailscale_path = shutil.which("tailscale")
  if tailscale_path:
    if DEBUG_DISCOVERY >= 3:
      print(f"[Tailscale] Found via PATH: {tailscale_path}")
    return tailscale_path

  raise FileNotFoundError(
    f"Tailscale executable not found on {system}. "
    f"Please install Tailscale from https://tailscale.com/download "
    f"or ensure it's in your PATH."
  )


class Device:
  def __init__(self, device_id: str, name: str, addresses: List[str], last_seen: Optional[datetime] = None):
    self.device_id = device_id
    self.name = name
    self.addresses = addresses
    self.last_seen = last_seen

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'Device':
    return cls(device_id=data.get('id', ''), name=data.get('name', ''), addresses=data.get('addresses', []), last_seen=cls.parse_datetime(data.get('lastSeen')))

  @classmethod
  def from_tailscale_status(cls, peer_data: Dict[str, Any]) -> 'Device':
    """Create Device from 'tailscale status' JSON output (Peer field)"""
    addresses = []
    if 'TailscaleIPs' in peer_data:
      addresses = [str(ip) for ip in peer_data['TailscaleIPs']]

    dns_name = peer_data.get('DNSName', '')
    hostname = dns_name.split('.')[0] if '.' in dns_name else dns_name

    return cls(
      device_id=peer_data.get('ID', ''),
      name=hostname,
      addresses=addresses,
      last_seen=datetime.now(timezone.utc)  # Active peers are online now
    )

  @staticmethod
  def parse_datetime(date_string: Optional[str]) -> Optional[datetime]:
    if not date_string:
      return None
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


async def get_tailscale_devices_local() -> Tuple[Dict[str, Device], str]:
  """Get Tailscale devices using local CLI (no API key required!)

  Uses 'tailscale status --json' to list all peers in the tailnet.
  This works automatically once both nodes join the same Tailscale network.

  Returns:
    Tuple of (devices_dict, self_node_id)
  """
  try:
    tailscale_cmd = _find_tailscale_executable()
    print(f"[Tailscale] 🔍 Using CLI: {tailscale_cmd}")

    process = await asyncio.create_subprocess_exec(
      tailscale_cmd, 'status', '--json',
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
      print(f"[Tailscale] ❌ Command failed (exit code {process.returncode}): {stderr.decode().strip()}")
      raise Exception(f"Tailscale command failed: {stderr.decode().strip()}")

    raw_output = stdout.decode()
    print(f"[Tailscale] 📄 Raw output length: {len(raw_output)} bytes")

    data = json.loads(raw_output)

    # Debug: print structure
    self_data = data.get('Self', {})
    peer_data = data.get('Peer', {})
    print(f"[Tailscale] 📊 Self: {self_data.get('HostName', 'N/A')}")
    print(f"[Tailscale] 📊 Peer count: {len(peer_data)}")

    devices = {}
    self_node_id = self_data.get('HostName', '')

    # Parse Peer nodes (other machines in the tailnet)
    for peer_id, peer_info in peer_data.items():
      is_online = peer_info.get('Online', False)
      peer_name = peer_info.get('DNSName', peer_id)
      print(f"[Tailscale] 🔎 Peer: {peer_name} | Online: {is_online}")

      # Only include active/online peers
      if is_online:
        device = Device.from_tailscale_status(peer_info)
        devices[device.name] = device

        print(f"[Tailscale] ✅ Active peer: {device.name} at {device.addresses}")

    print(f"[Tailscale] 🎯 Local discovery complete: {len(devices)} active peers (Self: {self_node_id})")

    return devices, self_node_id

  except FileNotFoundError as e:
    print(f"[Tailscale] ❌ CLI not found: {e}")
    raise Exception(f"{str(e)}")
  except json.JSONDecodeError as e:
    print(f"[Tailscale] ❌ JSON parse error: {e}")
    raise Exception(f"Failed to parse tailscale status output: {e}")
  except Exception as e:
    print(f"[Tailscale] ❌ Unexpected error: {type(e).__name__}: {e}")
    raise Exception(f"Failed to get local Tailscale status: {e}")


async def get_tailscale_devices(api_key: str, tailnet: str) -> Dict[str, Device]:
  """Get Tailscale devices using API (fallback method)

  Requires API key with read access. Use get_tailscale_devices_local() for automatic discovery.
  """
  async with aiohttp.ClientSession() as session:
    url = f"https://api.tailscale.com/api/v2/tailnet/{tailnet}/devices"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with session.get(url, headers=headers) as response:
      response.raise_for_status()
      data = await response.json()

      devices = {}
      for device_data in data.get("devices", []):
        if DEBUG_DISCOVERY >= 4:
          print(f"[Tailscale] API Device data: {device_data}")
        device = Device.from_dict(device_data)
        devices[device.name] = device

      return devices
