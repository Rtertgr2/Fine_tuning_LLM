"""HF dir → fp16 GGUF → quantize — subprocess ทั้งคู่ (mock ได้ผ่าน `run_cmd`)"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from core.compression import CompressionError
from core.compression.config import (
    DEFAULT_VARIANTS,
    QUANT_TYPE,
    artifact_path,
    convert_script,
    tool_path,
)


def run_cmd(cmd: list[str]) -> str:
    """รันคำสั่ง — จุดเดียวที่แตะ subprocess (unit test mock ที่นี่)"""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def _tail(exc: subprocess.CalledProcessError, n: int = 15) -> str:
    text = (exc.stderr or "") + (exc.stdout or "")
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:]) or "(no output)"


def convert_to_fp16(source: Path) -> Path:
    """`convert_hf_to_gguf.py` (venv python) → `<source>/gguf/<name>-fp16.gguf`"""
    out = artifact_path(source, "fp16")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(convert_script()),
        str(source),
        "--outfile",
        str(out),
        "--outtype",
        "f16",
    ]
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        raise CompressionError(
            f"convert_hf_to_gguf failed for {source}: {_tail(exc)}"
        ) from exc
    return out


def quantize_fp16(fp16: Path, variant: str) -> Path:
    """`llama-quantize <fp16> <out> <TYPE>` → `<source>/gguf/<name>-<variant>.gguf`"""
    qtype = QUANT_TYPE.get(variant)
    if qtype is None:
        raise CompressionError(
            f"unknown variant: {variant} (expected one of {', '.join(DEFAULT_VARIANTS)})"
        )
    source = fp16.parent.parent  # layout: <source>/gguf/<name>-fp16.gguf
    out = artifact_path(source, variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(tool_path("llama-quantize")), str(fp16), str(out), qtype]
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        raise CompressionError(
            f"llama-quantize failed for {fp16} ({variant}): {_tail(exc)}"
        ) from exc
    return out


def build_artifacts(
    source: Path, variants: Sequence[str] = DEFAULT_VARIANTS
) -> list[Path]:
    """สร้าง artifact ทุก variant ตามลำดับ (fp16 ก่อน — convert ครั้งเดียว)

    ไฟล์มีอยู่แล้ว → reuse ข้าม; 未知 variant / source ไม่ใช่ HF dir → CompressionError
    """
    if not (source / "config.json").is_file():
        raise CompressionError(
            f"{source} is not a HuggingFace model directory (config.json missing)."
        )
    unknown = [v for v in variants if v != "fp16" and v not in QUANT_TYPE]
    if unknown:
        raise CompressionError(
            f"unknown variant: {', '.join(unknown)} "
            f"(expected one of {', '.join(DEFAULT_VARIANTS)})"
        )

    fp16_path = artifact_path(source, "fp16")
    fp16_ready = fp16_path.is_file()
    results: list[Path] = []
    for variant in variants:
        out = artifact_path(source, variant)
        if out.is_file():
            results.append(out)
            continue
        if variant == "fp16":
            results.append(convert_to_fp16(source))
            fp16_ready = True
            continue
        if not fp16_ready:
            fp16_path = convert_to_fp16(source)
            fp16_ready = True
        results.append(quantize_fp16(fp16_path, variant))
    return results
