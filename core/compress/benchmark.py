"""llama-bench metrics + memory sampling (VmHWM /proc, xpu-smi ถ้ามี — ไม่มี → None)"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from core.compress import CompressionError
from core.compress.config import device_args, tool_path

_MB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(MiB|MB|GiB|GB)\b")
_TO_MB = {"MiB": 1.0, "MB": 1.0, "GiB": 1024.0, "GB": 1024.0}


def parse_bench_output(stdout: str) -> dict:
    """`llama-bench -o json` (array ของ row) → metrics — field หาย → None (ห้าม crash)

    row n_prompt>0 & n_gen==0 = prompt test (avg_ts), n_gen>0 = generation test;
    llama-bench ไม่ emit load time → load_ms คงเป็น None เว้นแต่จะมีใน output
    """
    result: dict = {"prompt_tps": None, "gen_tps": None, "load_ms": None}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return result
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if not isinstance(row, dict):
            continue
        load_ms = row.get("load_ms")
        if isinstance(load_ms, (int, float)):
            result["load_ms"] = float(load_ms)
        ts = row.get("avg_ts")
        if not isinstance(ts, (int, float)):
            continue
        n_prompt, n_gen = row.get("n_prompt"), row.get("n_gen")
        if n_prompt and not n_gen:
            result["prompt_tps"] = float(ts)
        elif n_gen and not n_prompt:
            result["gen_tps"] = float(ts)
    return result


def _vmhwm_mb(pid: int) -> float | None:
    """peak RSS ของ process จาก /proc/<pid>/status (ไม่มี proc → None)"""
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024.0  # kB → MB
                except ValueError:
                    return None
    return None


def _xpu_smi_output() -> str | None:
    exe = shutil.which("xpu-smi")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-t"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout


def _parse_vram(output: str | None) -> float | None:
    """ค่าหน่วยความจำสูงสุดใน output (ไม่เจอเลขหน่วย → None — ห้ามเดา)"""
    if not output:
        return None
    values = [
        float(num) * _TO_MB[unit] for num, unit in _MB_RE.findall(output)
    ]
    return max(values) if values else None


def sample_memory(pid: int) -> dict:
    """ตัวอย่าง memory ณ ขณะนั้น — peak_rss จาก /proc, peak_vram จาก xpu-smi (ถ้ามี)"""
    return {
        "peak_rss_mb": _vmhwm_mb(pid),
        "peak_vram_mb": _parse_vram(_xpu_smi_output()),
    }


def _drain_text(stream, buf: list[str]) -> None:
    """อ่านบรรทัดจน EOF ใส่ buf (reader thread — เรียกจาก thread เท่านั้น)"""
    buf.extend(stream)


def run_bench(gguf: Path, *, device: str = "vulkan") -> dict:
    """รัน `llama-bench -m <gguf> -o json` + poll memory ทุก 100ms (เก็บ max)

    stdout/stderr ถูก drain ตลอดเวลาด้วย reader thread — ห้ามรออ่านหลังจบ
    (pipe 64KB เต็ม → child block เอง = deadlock — Review #13);
    rc != 0 → CompressionError + log tail (English เสมอ)
    """
    cmd = [str(tool_path("llama-bench")), "-m", str(gguf), "-o", "json"]
    cmd += device_args(device)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )

    out_lines: list[str] = []
    err_lines: list[str] = []
    readers: list[threading.Thread] = []
    for stream, buf in ((proc.stdout, out_lines), (proc.stderr, err_lines)):
        if stream is not None:
            thread = threading.Thread(
                target=_drain_text, args=(stream, buf), daemon=True
            )
            thread.start()
            readers.append(thread)

    peak: dict = {"peak_rss_mb": None, "peak_vram_mb": None}
    while (rc := proc.poll()) is None:
        for key, value in sample_memory(proc.pid).items():
            if value is not None and (peak[key] is None or value > peak[key]):
                peak[key] = value
        time.sleep(0.1)

    # child ตายแล้ว → EOF → thread จบ (timeout กันค้างแบบไม่ปกติ)
    for thread in readers:
        thread.join(timeout=10)
    stdout, stderr = "".join(out_lines), "".join(err_lines)
    if rc != 0:
        text = ((stderr or "") + (stdout or "")).splitlines()
        tail = "\n".join(text[-15:]) or "(no output)"
        raise CompressionError(f"llama-bench failed for {gguf}: {tail}")

    return {**parse_bench_output(stdout), **peak}
