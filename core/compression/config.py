"""paths/variants/device ของ compression pipeline — จุดเดียวของค่าคงที่

- llama.cpp tools หาจาก `LLAMA_CPP_DIR` (รองรับ system install) → default `tools/llama.cpp`
- artifact อยู่ `<source_dir>/gguf/` เสมอ; report อยู่ `benchmarks/<source>/<variant>.json`
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from core import estimator
from core.compression import CompressionError

DEFAULT_VARIANTS: tuple[str, ...] = ("fp16", "q8_0", "q4_k_m")
QUANT_TYPE: dict[str, str] = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}
VARIANT_BITS: dict[str, int] = {"fp16": 16, "q8_0": 8, "q4_k_m": 4}


def tools_dir() -> Path:
    """โฟลเดอร์รากของ llama.cpp checkout/build (env override → default)"""
    env = os.environ.get("LLAMA_CPP_DIR")
    return Path(env) if env else Path("tools/llama.cpp")


def tool_path(name: str) -> Path:
    """binary ใน build/bin (llama-server / llama-quantize / llama-bench)"""
    return tools_dir() / "build" / "bin" / name


def convert_script() -> Path:
    return tools_dir() / "convert_hf_to_gguf.py"


def require_tools() -> None:
    """ยังไม่ได้ build → CompressionError (English, ชี้ทาง setup)"""
    missing = [n for n in ("llama-server", "llama-quantize", "llama-bench") if not tool_path(n).is_file()]
    if missing or not convert_script().is_file():
        raise CompressionError("llama.cpp not found. Run scripts/setup_llamacpp.sh first.")


def check_vulkan() -> bool:
    """vulkaninfo ตอบสนองจริงไหม (ไม่มี → False — ไม่ใช่ raise)"""
    if shutil.which("vulkaninfo") is None:
        return False
    try:
        proc = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def resolve_source(model_id: str) -> Path:
    """ค่า `--model` → HF dir ที่มี `config.json` (path ตรง ๆ / ชื่อใต้ `MODELS_DIR`)

    ไม่พบในเครื่อง → CompressionError + คำสั่ง `hf download` (English เสมอ)
    """
    source = estimator.resolve_local_model(model_id)
    if source is None:
        name = Path(model_id).name
        raise CompressionError(
            f"model not found locally: {model_id}. "
            f"Download it first: hf download {model_id} --local-dir models/{name}"
        )
    if not (source / "config.json").is_file():
        raise CompressionError(
            f"{source} is not a HuggingFace model directory (config.json missing)."
        )
    return source


def device_args(device: str) -> list[str]:
    """flag ของ llama-server/llama-bench (ยืนยันจาก --help ของ b11371):
    vulkan = default (server ngl=auto, bench auto) → ไม่ต้องส่งอะไร;
    cpu = `-ngl 0` ปิด offload (llama.cpp ไม่มี flag `--no-gpu`)
    """
    if device == "vulkan":
        return []
    if device == "cpu":
        return ["-ngl", "0"]
    raise CompressionError(f"unknown device: {device} (expected vulkan or cpu)")


def gguf_dir(source: Path) -> Path:
    return source / "gguf"


def artifact_path(source: Path, variant: str) -> Path:
    return gguf_dir(source) / f"{source.name}-{variant}.gguf"


def report_path(source: Path, variant: str) -> Path:
    return Path("benchmarks") / source.name / f"{variant}.json"
