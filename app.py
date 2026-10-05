"""Gradio UI entry point — Phase 4 (สเปก plan.md §4.7 + §5 guardrails)

- `mp.set_start_method("spawn")` ที่นี่ที่เดียว (§5 Driver Deadlock)
- bind 127.0.0.1:7860 เท่านั้น (§5 Network Exposure) — การกัน Process Stacking จริงอยู่ที่
  `concurrency_id="model_load"` ของ predict/merge/eval/start (dashboard) + controller
  re-entrancy guard + tick ล็อกปุ่ม — **ไม่ใช่** queue default (P1 C2: เดิมกล่าวอ้างเท็จ)
- exit handler → `controller.exit()` ฆ่า child กัน orphan/XPU context leak (§5 Orphan Process)

รัน: python app.py
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import signal

import gradio as gr

from ui.controller import TrainingController
from ui.dashboard import build_dashboard

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860  # M10: constants นี้อยู่ตรงนี้ (ไม่ใช่ safe_defaults) เพราะเป็น runtime binding ของ entry point เดียว — spec §4 กำหนดไว้แล้ว


def main() -> None:
    mp.set_start_method("spawn", force=True)

    controller = TrainingController()
    demo: gr.Blocks = build_dashboard(controller)

    # exit handler ทุกทาง: normal exit, SIGTERM, SIGINT (Ctrl-C)
    atexit.register(controller.exit)

    def _handle_signal(signum, _frame):
        controller.exit()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # P1 C2: default_concurrency_limit เป็นค่า default อยู่แล้ว (ไม่มีผล) + limit จริงเป็นราย
    # concurrency-id — ปล่อยค่า default ไม่แสร้งว่าเป็นเกราะป้องกัน
    demo.queue()
    demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT)


if __name__ == "__main__":
    main()
