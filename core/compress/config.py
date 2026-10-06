"""paths/variants/device ของ compression pipeline — จุดเดียวของค่าคงที่

- llama.cpp tools หาจาก `LLAMA_CPP_DIR` (รองรับ system install) → default `tools/llama.cpp`
- artifact อยู่ `<source_dir>/gguf/` เสมอ; report อยู่ `benchmarks/<model>-<hash>/<variant>.json`
  (ตรงรูป README "Reports" ซึ่งเป็นแหล่งจริง — `<model>` = `source.name`,
  `<hash>` = sha256[:8] ของ resolved path ของ source (`report_id`);
  layout `benchmarks/compression/...` ใน phase6a design = historical/frozen ห้ามอ้าง)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from core.infra import estimator
from core.compress import CompressionError

DEFAULT_VARIANTS: tuple[str, ...] = ("fp16", "q8_0", "q4_k_m")
QUANT_TYPE: dict[str, str] = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}
VARIANT_BITS: dict[str, int] = {"fp16": 16, "q8_0": 8, "q4_k_m": 4}

# repo root จากตำแหน่งไฟล์นี้ (core/compress/config.py → ขึ้น 2 ชั้น) — ตรงกับที่
# setup_llamacpp.sh ติดตั้ง (`$0/..`) เรียก CLI จาก directory อื่นก็หา tools เจอ (Review #5)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def tools_dir() -> Path:
    """โฟลเดอร์รากของ llama.cpp checkout/build (env override → default = repo root)"""
    env = os.environ.get("LLAMA_CPP_DIR")
    return Path(env) if env else _REPO_ROOT / "tools" / "llama.cpp"


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
    นอก sandbox → CompressionError บอกเรื่อง sandbox (คืน None ถูก caller ตีความ
    ว่า "โหลดจาก Hub" → bug: ส่ง local path เข้า hf_hub_download);
    input ที่เป็น path-like แม้ไม่มีจริง → ห้ามเสนอ `hf download` (คำสั่งใช้ไม่ได้)
    """
    try:
        source = estimator.resolve_local_model(model_id)
    except ValueError as exc:
        raise CompressionError(str(exc)) from exc
    if source is None:
        name = Path(model_id).name
        if Path(model_id).expanduser().is_absolute() or model_id.startswith((".", "~")):
            raise CompressionError(
                f"model not found locally: {model_id} (this looks like a local path — "
                "check the path, or move it under models/)."
            )
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


def report_id(source: Path) -> str:
    """identity ของ report: basename + hash8 ของ resolved path — source คนละที่ชื่อเดียวกัน

    ต้องไม่เขียนทับกัน (Review #3)
    """
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{source.name}-{digest}"


def report_path(source: Path, variant: str) -> Path:
    return Path("benchmarks") / report_id(source) / f"{variant}.json"
