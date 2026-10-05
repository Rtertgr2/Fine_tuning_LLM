"""HF dir → fp16 GGUF → quantize — subprocess ทั้งคู่ (mock ได้ผ่าน `run_cmd`)

ความสดของ artifact: แต่ตัวมี sidecar `<name>.srcmeta.json` = fingerprint ของ source
ณ ตอนสร้าง (`size + mtime_ns` ของไฟล์ระดับบน) — source เปลี่ยน / sidecar หาย → rebuild;
เขียนไฟล์แบบ atomic (`.partial` → `os.replace`) — fail/Ctrl+C ห้ามทิ้งเศษให้ reuse รอบหน้า
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from core.compress import CompressionError
from core.compress.config import (
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


def source_fingerprint(source: Path) -> dict:
    """`{ชื่อไฟล์: [size, mtime_ns]}` ของไฟล์ระดับบนใน source (ไม่รวมไดเรกทอรี เช่น `gguf/`)"""
    meta = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            st = path.stat()
            meta[path.name] = [st.st_size, st.st_mtime_ns]
    return meta


def _meta_path(out: Path) -> Path:
    return out.with_name(out.name + ".srcmeta.json")


def write_meta(out: Path, source: Path) -> None:
    """เขียน sidecar ระบุ fingerprint ของ source ณ ตอนสร้าง `out` (เรียกหลังสร้างสำเร็จเท่านั้น)"""
    _meta_path(out).write_text(
        json.dumps(source_fingerprint(source), sort_keys=True), encoding="utf-8"
    )


def is_fresh(out: Path, source: Path) -> bool:
    """artifact ใช้ต่อได้ = ไฟล์อยู่ + sidecar ตรงกับ source ปัจจุบัน

    ไม่มี sidecar (legacy artifact) หรือ JSON เสีย → ถือ stale → rebuild (ปลอดภัยไว้ก่อน)
    """
    if not out.is_file():
        return False
    meta = _meta_path(out)
    if not meta.is_file():
        return False
    try:
        recorded = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return recorded == source_fingerprint(source)


def _finalize(tmp: Path, out: Path, source: Path) -> None:
    """ย้าย tmp → out แบบ atomic แล้วเขียน sidecar (tool ไม่เขียนอะไรเลย = ข้าม — ไม่มีอะไรให้ finalize)"""
    if tmp.is_file():
        os.replace(tmp, out)
        write_meta(out, source)


def convert_to_fp16(source: Path) -> Path:
    """`convert_hf_to_gguf.py` (venv python) → `<source>/gguf/<name>-fp16.gguf` (atomic)"""
    out = artifact_path(source, "fp16")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    cmd = [
        sys.executable,
        str(convert_script()),
        str(source),
        "--outfile",
        str(tmp),
        "--outtype",
        "f16",
    ]
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)  # เศษไฟล์จาก fail → ห้ามค้าง (Review #9)
        raise CompressionError(
            f"convert_hf_to_gguf failed for {source}: {_tail(exc)}"
        ) from exc
    except BaseException:
        tmp.unlink(missing_ok=True)  # Ctrl+C ระหว่างเขียน
        raise
    _finalize(tmp, out, source)
    return out


def quantize_fp16(fp16: Path, variant: str) -> Path:
    """`llama-quantize <fp16> <out> <TYPE>` → `<source>/gguf/<name>-<variant>.gguf` (atomic)"""
    qtype = QUANT_TYPE.get(variant)
    if qtype is None:
        raise CompressionError(
            f"unknown variant: {variant} (expected one of {', '.join(DEFAULT_VARIANTS)})"
        )
    source = fp16.parent.parent  # layout: <source>/gguf/<name>-fp16.gguf
    out = artifact_path(source, variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    cmd = [str(tool_path("llama-quantize")), str(fp16), str(tmp), qtype]
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)  # Review #9
        raise CompressionError(
            f"llama-quantize failed for {fp16} ({variant}): {_tail(exc)}"
        ) from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _finalize(tmp, out, source)
    return out


def build_artifacts(
    source: Path, variants: Sequence[str] = DEFAULT_VARIANTS
) -> list[Path]:
    """สร้าง artifact ทุก variant ตามลำดับ (fp16 ก่อน — convert ครั้งเดียว)

    reuse เฉพาะตอน sidecar ตรงกับ source ปัจจุบัน (Review #1) — source เปลี่ยน → rebuild;
    未知 variant / source ไม่ใช่ HF dir → CompressionError
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
    fp16_ready = is_fresh(fp16_path, source)  # เช็คครั้งเดียวตอนเข้า loop (กัน convert ซ้ำใน run เดียวกัน)
    results: list[Path] = []
    for variant in variants:
        out = artifact_path(source, variant)
        if is_fresh(out, source):
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
