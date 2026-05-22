from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path

from app.schemas import MemoryStatus


def get_memory_status() -> MemoryStatus:
    dedicated = _nvidia_smi_memory()
    if dedicated:
        total_mb, used_mb = dedicated
        return MemoryStatus(kind="dedicated", label="显存", total_mb=total_mb, used_mb=used_mb)

    total_mb, used_mb = _system_memory()
    label = "共享内存" if _is_jetson() else "内存"
    return MemoryStatus(kind="shared" if _is_jetson() else "system", label=label, total_mb=total_mb, used_mb=used_mb)


def _nvidia_smi_memory() -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None

    total = 0
    used = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            total += int(float(parts[0]))
            used += int(float(parts[1]))
        except ValueError:
            continue
    return (total, used) if total > 0 else None


def _system_memory() -> tuple[int, int]:
    if os.name == "nt":
        return _windows_memory()
    values = _linux_meminfo()
    if values:
        return values
    return 0, 0


def _linux_meminfo() -> tuple[int, int] | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, _, rest = line.partition(":")
        amount = rest.strip().split()[0] if rest.strip() else "0"
        try:
            values[key] = int(amount)
        except ValueError:
            continue
    total_kb = values.get("MemTotal", 0)
    available_kb = values.get("MemAvailable", values.get("MemFree", 0))
    used_kb = max(0, total_kb - available_kb)
    return round(total_kb / 1024), round(used_kb / 1024)


def _windows_memory() -> tuple[int, int]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0, 0
    total_mb = round(status.ullTotalPhys / 1024 / 1024)
    used_mb = round((status.ullTotalPhys - status.ullAvailPhys) / 1024 / 1024)
    return total_mb, used_mb


def _is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return False
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        return False
    model = model_path.read_text(encoding="utf-8", errors="ignore").lower()
    return "nvidia" in model or "jetson" in model
