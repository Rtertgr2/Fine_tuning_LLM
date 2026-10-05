"""Hardware Environment Inspector — single source of truth ของสภาพแวดล้อม (สเปก plan.md §4.1)

รัน: from core.hardware import inspect
status == "ready" เชื่อถือได้ 100% — เช็คตามลำดับ XPU → RAM → disk → ready
"""

from __future__ import annotations

import psutil
import torch

from configs.safe_defaults import DISK_MIN_GB, RAM_MIN_GB

GB = 1024**3


def _xpu_available() -> bool:
    """เช็ค capability จริงของ XPU (P1 D3) — hasattr(torch,'xpu') ไม่ใช่ capability check

    torch ไม่มี xpu module (AttributeError) หรือ driver ไม่พร้อม (RuntimeError) → False
    """
    try:
        return bool(torch.xpu.is_available())
    except (AttributeError, RuntimeError):
        return False


def inspect(output_dir: str = ".") -> dict:
    """ตรวจทรัพยากรระบบ คืน dict ตาม contract ของสเปก §4.1"""
    ram_available_gb = psutil.virtual_memory().available / GB
    disk_free_gb = psutil.disk_usage(output_dir).free / GB

    # (1) XPU ต้องมาก่อน — ไม่มี XPU คือ no_xpu ไม่ว่า RAM/disk จะเป็นอย่างไร
    if not _xpu_available():
        return {
            "status": "no_xpu",
            "device_name": "",
            "total_vram_gb": 0.0,
            "free_vram_gb": 0.0,
            "ram_available_gb": ram_available_gb,
            "disk_free_gb": disk_free_gb,
        }

    # (2) VRAM จาก mem_get_info — คืน (free, total) เป็น bytes
    free_b, total_b = torch.xpu.mem_get_info(0)
    result = {
        "status": "ready",
        "device_name": torch.xpu.get_device_name(0),
        "total_vram_gb": total_b / GB,
        "free_vram_gb": free_b / GB,
        "ram_available_gb": ram_available_gb,
        "disk_free_gb": disk_free_gb,
    }

    # (3) RAM → (4) disk → ready
    if ram_available_gb < RAM_MIN_GB:
        result["status"] = "insufficient_ram"
    elif disk_free_gb < DISK_MIN_GB:
        result["status"] = "insufficient_disk"
    return result
