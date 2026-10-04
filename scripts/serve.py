#!/usr/bin/env python3
"""Serve โมเดล GGUF บน local URL — ต่อ llama-server ให้เครื่องมือข้างนอกเรียกได้

    .venv/bin/python scripts/serve.py --model Qwen/Qwen2.5-Coder-0.5B --variant q4_k_m

Ctrl+C / SIGTERM = หยุด server จริง (รอ process ปิด — ไม่ leak VRAM)
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# รันเป็น `python scripts/serve.py` → sys.path[0] = scripts/ ต้องเพิ่ม root ก่อน
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compression import CompressionError
from core.compression.config import artifact_path, resolve_source
from core.compression.llama_runner import _log_tail, start_server, stop_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve.py",
        description="Serve a GGUF model on a local URL (llama-server).",
    )
    parser.add_argument("--gguf", default=None, help="path to a .gguf file")
    parser.add_argument(
        "--model",
        default=None,
        help="model under models/ (e.g. Qwen/Qwen2.5-Coder-0.5B)",
    )
    parser.add_argument(
        "--variant",
        default="q4_k_m",
        help="artifact variant used with --model (default: q4_k_m)",
    )
    parser.add_argument("--port", type=int, default=8080, help="default: 8080")
    parser.add_argument(
        "--device", choices=("vulkan", "cpu"), default="vulkan", help="default: vulkan"
    )
    return parser


def _resolve_gguf(args: argparse.Namespace) -> Path:
    if args.gguf is None and args.model is None:
        raise CompressionError("pass either --gguf <path> or --model <name>.")
    if args.gguf is not None:
        path = Path(args.gguf)
        if path.suffix != ".gguf":
            raise CompressionError(
                f"no .gguf extension: {path} "
                "(pass a .gguf file, or use --model with --variant)."
            )
        return path
    return artifact_path(resolve_source(args.model), args.variant)


def serve(
    gguf: Path,
    *,
    port: int,
    device: str,
    start=start_server,
    stop=stop_server,
) -> int:
    """เปิด server → พิมพ์ URL → รอจนจบ → stop จริง

    - ถูกขัดจังหวะ (Ctrl+C/SIGTERM) → "Server stopped." + คืน 0 (ปกติ)
    - server ตายเอง → rc + log tail ไป stderr + คืน 1 (Review #12)
    """
    try:
        handle = start(gguf, port=port, device=device)
    except CompressionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Server ready: {handle.url} (model: {handle.alias})", flush=True)
    interrupted = False
    rc = None
    try:
        rc = handle.proc.wait()
    except KeyboardInterrupt:
        interrupted = True
    stop(handle)
    if not interrupted:
        # Review #12: ตายเอง (crash/device loss) ≠ user stop → ต้องบอกผู้ใช้
        # พร้อม rc + log tail แล้วคืน 1 (ห้ามรายงานว่าหยุดปกติ)
        print(f"llama-server exited unexpectedly (rc={rc}).", file=sys.stderr)
        print(_log_tail(handle.log_text), file=sys.stderr)
        return 1
    print("Server stopped.", flush=True)
    return 0


def _sigterm_interrupt(_signum: int, _frame) -> None:
    raise KeyboardInterrupt


def _ensure_sigint() -> None:
    """ติด handler SIGINT คืนเมื่อ inherited เป็น ignore (job background/nohup)

    contract ของ serve = 'รอ KeyboardInterrupt' — ถ้า SIGINT ถูก ignore ตั้งแต่
    เริ่ม process (bash `cmd &`) Python จะไม่ติด handler เอง → interrupt ใช้ไม่ได้
    """
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        gguf = _resolve_gguf(args)
    except CompressionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _ensure_sigint()
    signal.signal(signal.SIGTERM, _sigterm_interrupt)
    return serve(
        gguf,
        port=args.port,
        device=args.device,
        start=start_server,
        stop=stop_server,
    )


if __name__ == "__main__":
    raise SystemExit(main())
