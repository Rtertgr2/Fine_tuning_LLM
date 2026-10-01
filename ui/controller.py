"""TrainingController — แกนกลางประสานงานระหว่าง Gradio UI กับ training subprocess (สเปก plan.md §4.4)

ข้อกำหนดสำคัญ (spec §3.2):
- **ห้าม import gradio** — เป็น pure Python ให้ unit test ด้วย fake Process/Queue ได้ (duck-typed)
- ทุกการ mutate สถานะครอบด้วย `threading.Lock` (Timer thread vs button handlers)
- Message จาก child ทุกใบผ่าน `ipc_bridge.validate_message` ก่อนใช้ — ไม่ผ่าน → เก็บเป็น error
- Status: state machine `idle → starting → training → saving → finished|aborted`
  หลัง terminal **ห้ามมี status ใหม่มาทับ** (Phase 3 ส่ง `finished` ซ้ำ 2 ครั้ง → dedup โดย ignore)
- Abort: เรียก `abort_process` (SIGTERM→SIGKILL) แล้วตั้ง `aborted` ทันที — child ถูก kill
  ไม่ทันส่ง status เอง (กัน watchdog false alarm, ส่งไม้ต่อจาก Phase 3)
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from queue import Empty
from typing import NamedTuple

from configs.safe_defaults import MAX_SEQ_LENGTH_CAP
from core import hardware
from core.estimator import EstimateResult, estimate
from core.ipc_bridge import (
    TERMINAL_STATUSES,
    TRAINING_STATUSES,
    abort_process,
    validate_message,
    watchdog_error,
)
from core.trainer_worker import run_training, validate_config


class TickSnapshot(NamedTuple):
    """ผลลัพธ์ 1 รอบของ `tick()` — ให้ dashboard นำไปอัปเดต widget (ห้ามตีความใน component)"""

    status: str
    metrics: list[dict]
    logs: list[str]
    error: str | None
    watchdog: str | None
    training_active: bool


class TrainingController:
    def __init__(self, *, queue_factory=mp.Queue, process_factory=mp.Process):
        self._queue_factory = queue_factory
        self._process_factory = process_factory
        self._lock = threading.Lock()
        self._queue = queue_factory()
        self._process = None
        self._status: str = "idle"
        self._metrics: list[dict] = []
        self._logs: list[str] = []
        self._error: str | None = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> str:
        return self._status

    @property
    def training_active(self) -> bool:
        """True เมื่อมี process ที่ยังไม่จบ (ใช้ล็อกปุ่ม Predict/Merge/Start — §5 VRAM Contention)"""
        return self._process is not None and self._status not in TERMINAL_STATUSES

    # ------------------------------------------------------------------ #
    # Pre-flight (Tab 1)
    # ------------------------------------------------------------------ #

    def preflight(self, config: dict, user_params_b: float | None = None) -> EstimateResult:
        """inspect(output_dir) → estimate(...) — คืน verdict + reason + breakdown ให้ dashboard แสดง

        verdict ที่ไม่ใช่ safe/warning (blocked/no_xpu/insufficient_*) → dashboard ปิดปุ่ม Start
        """
        hw = hardware.inspect(output_dir=config.get("output_dir", "."))
        return estimate(
            hw,
            model_id=config["model_id"],
            user_params_b=user_params_b,
            seq_length=config.get("max_seq_length", MAX_SEQ_LENGTH_CAP),
        )

    # ------------------------------------------------------------------ #
    # Start / Abort / Exit (Tab 2)
    # ------------------------------------------------------------------ #

    def start(self, config: dict) -> str:
        """validate → spawn `run_training` — คืน "started" หรือข้อความ error (ไม่ raise)"""
        with self._lock:
            try:
                validate_config(config)
            except ValueError as exc:
                return str(exc)
            self._process = self._process_factory(
                target=run_training, args=(config, self._queue)
            )
            self._process.start()
            self._status = "starting"
            return "started"

    def abort(self) -> bool:
        """SIGTERM→SIGKILL แล้วตั้ง aborted ทันที — คืน True ถ้า process ตายสนิท

        เรียกตอนยังไม่ start → คืน False (ไม่ raise)
        """
        with self._lock:
            if self._process is None:
                return False
            dead = abort_process(self._process)
            self._status = "aborted"
            return dead

    def exit(self) -> None:
        """ปิด app → ฆ่า child (ถ้ามี) กัน orphan/XPU context leak (§5 Orphan Process) — เรียกซ้ำได้"""
        with self._lock:
            if self._process is not None and self._process.is_alive():
                abort_process(self._process)
                self._status = "aborted"

    # ------------------------------------------------------------------ #
    # Tick (gr.Timer เรียกทุก 1 วินาที)
    # ------------------------------------------------------------------ #

    def tick(self) -> TickSnapshot:
        """drain queue จนหมดแบบ non-blocking → อัปเดตสถานะ → ตรวจ zombie — คืน snapshot ให้ UI"""
        with self._lock:
            self._drain_queue()
            watchdog = (
                watchdog_error(self._process, self._status)
                if self._process is not None
                else None
            )
            watchdog_text = None
            if watchdog is not None:
                watchdog_text = f"{watchdog['message']}\n{watchdog['traceback']}"
            return TickSnapshot(
                status=self._status,
                metrics=list(self._metrics),
                logs=list(self._logs),
                error=self._error,
                watchdog=watchdog_text,
                training_active=self.training_active,
            )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _drain_queue(self) -> None:
        while True:
            try:
                msg = self._queue.get_nowait()
            except Empty:
                return
            if not validate_message(msg):
                self._error = f"message ไม่ผ่าน validate: {msg!r}"
                continue
            self._handle_message(msg)

    def _handle_message(self, msg: dict) -> None:
        mtype = msg["type"]
        if mtype == "metric":
            self._metrics.append(msg)
        elif mtype == "log":
            self._append_log(msg["level"], msg["text"])
        elif mtype == "status":
            state = msg["state"]
            if self._status in TERMINAL_STATUSES:
                return  # terminal แล้ว — ห้าม status ใหม่มาทับ (dedup กันซ้ำ)
            if state in TRAINING_STATUSES:
                self._status = state
        elif mtype == "error":
            self._error = msg["message"]
            self._append_log("ERROR", msg["message"])
            if msg["traceback"]:
                self._logs.append(msg["traceback"])

    def _append_log(self, level: str, text: str) -> None:
        self._logs.append(f"[{level}] {text}")
