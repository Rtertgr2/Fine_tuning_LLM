"""จุดเดียวของ llama-server: start → (health + alias check) / stop / HTTP completion

ใช้ร่วมโดย eval (ขึ้น-ปิด) และ serve.py (ค้างจน user สั่งหยุด) — Ctrl+C = SIGINT
→ รอ process ปิดจริง (ไม่ leak VRAM)
"""

from __future__ import annotations

import json
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from core.compression import CompressionError
from core.compression.config import device_args, require_tools, tool_path

_KV_RE = re.compile(r"llama_kv_cache:\s+size\s*=\s*([\d.]+)\s*MiB")


def find_free_port() -> int:
    """port ว่างบน 127.0.0.1 (ให้ OS เลือก — กันชน server เดิม)"""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_get(url: str, *, timeout: float = 5.0) -> tuple[int, str]:
    """→ (status_code, body) — ติดต่อไม่ได้ = (0, ""); non-2xx = (code, "")"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), ""
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, ""


def _http_post(url: str, body: dict, *, timeout: float = 300.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _log_tail(text: str, n: int = 15) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:]) or "(no output)"


def _drain(stream, log: list[str]) -> None:
    """อ่าน stdout ต่อเนื่องกัน (กัน pipe buffer เต็ม = server แขวน)"""
    log.extend(stream)


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    url: str
    alias: str
    _log: list[str] = field(default_factory=list, repr=False)
    _reader: threading.Thread | None = field(default=None, repr=False)

    @property
    def log_text(self) -> str:
        """stdout ทั้งรัน — CLI ใช้ parse kv_cache_mb"""
        return "".join(self._log)

    def complete(self, prompt: str, *, n_predict: int, temperature: float = 0.0) -> str:
        """POST /completion — prompt ส่งออกไป byte เดียวกับ input (ไม่ strip)"""
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": False,
        }
        resp = _http_post(f"{self.url}/completion", body)
        return resp["content"]


def start_server(
    gguf: Path,
    *,
    port: int | None = None,
    device: str = "vulkan",
    timeout: float = 60.0,
) -> ServerHandle:
    """เปิด llama-server → รอ /health 200 → ยืนยันตัวตนผ่าน /props.model_alias

    ใช้ `-v` เสมอ (ruling) — บรรทัด `llama_kv_cache: size = X MiB` ที่ spec §4 ต้องการ
    มาแค่ verbose log; error ทุกกรณี = English + ชี้ทาง (--port / --device cpu)
    """
    require_tools()
    if port is None:
        port = find_free_port()
    alias = gguf.stem
    cmd = [
        str(tool_path("llama-server")),
        "-m", str(gguf),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--alias", alias,
        "-v",
    ]
    cmd += device_args(device)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    handle = ServerHandle(proc=proc, url=f"http://127.0.0.1:{port}", alias=alias)
    if proc.stdout is not None:
        handle._reader = threading.Thread(
            target=_drain, args=(proc.stdout, handle._log), daemon=True
        )
        handle._reader.start()

    deadline = time.monotonic() + timeout
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise CompressionError(
                f"llama-server failed to start: {_log_tail(handle.log_text)}. "
                "Try --port or --device cpu."
            )
        status, _ = _http_get(f"{handle.url}/health", timeout=2.0)
        if status == 200:
            ready = True
            break
        time.sleep(0.15)
    if not ready:
        stop_server(handle)
        raise CompressionError(
            f"llama-server did not become ready in {timeout}s: "
            f"{_log_tail(handle.log_text)}"
        )

    status, body = _http_get(f"{handle.url}/props", timeout=5.0)
    served_alias = None
    if status == 200 and body:
        try:
            served_alias = json.loads(body).get("model_alias")
        except json.JSONDecodeError:
            served_alias = None
    if served_alias != alias:
        stop_server(handle)
        raise CompressionError(
            f"port {port} is serving a different model (expected {alias}). "
            "Try another --port."
        )
    return handle


def stop_server(handle: ServerHandle, *, timeout: float = 10.0) -> None:
    """SIGINT → รอปิดจริง → ไม่ทันค่อย kill (llama-server ปิด graceful เอง)"""
    proc = handle.proc
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def kv_cache_mb(log_text: str) -> float | None:
    """ขนาด KV cache จาก log (บรรทัดสุดท้ายที่เจอ — ไม่มี → None ห้ามเดา)"""
    matches = _KV_RE.findall(log_text)
    return float(matches[-1]) if matches else None
