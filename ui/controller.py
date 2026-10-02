"""TrainingController — แกนกลางประสานงานระหว่าง Gradio UI กับ training subprocess (สเปก plan.md §4.4)

ข้อกำหนดสำคัญ (spec §3.2):
- **ห้าม import gradio** — เป็น pure Python ให้ unit test ด้วย fake Process/Queue ได้ (duck-typed)
- ทุกการ mutate สถานะครอบด้วย `threading.Lock` (Timer thread vs button handlers)
- Message จาก child ทุกใบผ่าน `ipc_bridge.validate_message` ก่อนใช้ — ไม่ผ่าน → เก็บเป็น error
- Status: state machine `idle → starting → training → saving → finished|aborted`
  หลัง terminal **ห้ามมี status ใหม่มาทับ** (Phase 3 ส่ง `finished` ซ้ำ 2 ครั้ง → dedup โดย ignore)
- Abort: ตั้ง `aborted` ก่อน escalate แล้วเรียก `abort_process` (SIGTERM→SIGKILL) **นอก lock**
  — child ถูก kill ไม่ทันส่ง status เอง (กัน watchdog false alarm, ส่งไม้ต่อจาก Phase 3)
  และการ escalate นอก lock ทำให้ tick ไม่ถูกบล็อก 10 วินาที
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import threading
from pathlib import Path
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
from core.evaluator import EVAL_DIR
from core.trainer_worker import (
    EVAL_DONE,
    predict_middle,
    run_eval_worker,
    run_training,
    validate_config,
)

# `get_nowait()` ของ mp.Queue อาจคาย error อื่นนอกจาก Empty เมื่อ pipe ถูก kill กลางเขียน
# (truncated pickle → EOFError/OSError/UnpicklingError) — ห้ามให้หลุดออกจาก tick() ไม่งั้น UI ค้างถาวร
_DRAIN_ERRORS = (EOFError, OSError, pickle.UnpicklingError)


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
        # output_dir อาจยังไม่ถูกสร้าง (fresh start) — ใช้ ancestor ที่มีอยู่จริงแทน (disk อยู่ volume เดียวกัน)
        out_dir = Path(config.get("output_dir", "."))
        while not out_dir.exists() and out_dir != out_dir.parent:
            out_dir = out_dir.parent
        hw = hardware.inspect(output_dir=str(out_dir))
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
        """validate → spawn `run_training` — คืน "started" หรือข้อความ error (ไม่ raise)

        มี re-entrancy guard: ระหว่าง training อยู่ห้าม spawn ซ้อน (§5 Process Stacking)
        และ reset metrics/logs/error ของรอบก่อน — plot ต้องไม่ปนรันเก่า (spec-silent → reasonable expectation)
        """
        with self._lock:
            if self.training_active:
                return "Training is already running — cannot start another run"
            try:
                validate_config(config)
            except (ValueError, TypeError) as exc:
                return str(exc)
            self._reset_run_state()
            self._process = self._process_factory(
                target=run_training, args=(config, self._queue)
            )
            self._process.start()
            self._status = "starting"
            return "started"

    def abort(self) -> bool:
        """ตั้ง aborted ทันที → SIGTERM→SIGKILL นอก lock — คืน True ถ้า process ตายสนิท

        เรียกตอนยังไม่ start → คืน False (ไม่ raise)
        ถ้า escalate ล้มเหลว (เดดไลน์ kill ไม่ตาย) → status ถอยกลับ ไม่ปลดล็อกปุ่มทั้งที่ process ยังอยู่
        """
        with self._lock:
            if self._process is None:
                return False
            prev = self._status
            if prev not in TERMINAL_STATUSES:
                # ตั้ง terminal ก่อน — กัน watchdog false alarm ระหว่าง kill (ส่งไม้ต่อจาก Phase 3)
                self._status = "aborted"
            process = self._process
        dead = abort_process(process)  # นอก lock — tick ไม่ถูกบล็อก 10 วิ (M9)
        if not dead:
            with self._lock:
                if self._status == "aborted" and prev not in TERMINAL_STATUSES:
                    self._status = prev  # ยังไม่ตายจริง → คง lock ไว้ (M8)
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
        """drain queue จนหมดแบบ non-blocking → อัปเดตสถานะ → ตรวจ zombie — คืน snapshot ให้ UI

        ห้าม raise เด็ดขาด — ถ้า queue พัง (pipe โดน kill กลางเขียน) เก็บเป็น error แล้วหยุด drain
        """
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

    def _reset_run_state(self) -> None:
        """ล้างประวัติรอบก่อน (I4) — plot/log/error ของรันใหม่ต้องเริ่มว่างเปล่า"""
        self._metrics.clear()
        self._logs.clear()
        self._error = None
        # msg ค้างใน queue จากรอบก่อน (ถ้ามี) ต้องทิ้ง ไม่งั้นถูกนับเป็นของรันใหม่
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return
            except _DRAIN_ERRORS:
                return

    def _drain_queue(self) -> None:
        while True:
            try:
                msg = self._queue.get_nowait()
            except Empty:
                return
            except _DRAIN_ERRORS as exc:
                # pipe พัง = ข้อมูลหาย — เก็บเป็น error แล้วหยุด drain (ห้ามให้ tick raise → UI ค้าง)
                self._error = f"queue read failed (IPC pipe broken): {exc}"
                return
            if not validate_message(msg):
                self._error = f"message failed validation (ignored): {msg!r}"
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

    def run_eval(
        self,
        config: dict,
        mode: str,
        *,
        eval_dir: str | Path = EVAL_DIR,
        timeout: float = 900.0,
    ) -> dict:
        """spawn `run_eval_worker` ใน process แยก → drain log (เว้น EVAL_DONE) → คืน JSON result

        - ระหว่างเทรนอยู่ห้าม eval (§5 VRAM Contention — Review Focus #3)
        - child ตายเงียบ/หมด timeout → RuntimeError ทันที
        - ทุก path ผ่าน finally เก็บ child เสมอ — ห้ามปล่อย orphan (§5 I3)
        - progress จาก child เข้า `self._logs` ให้ Timer thread โชว์ระหว่างรอ
        """
        import time

        if self.training_active:
            raise RuntimeError(
                "Training in progress — evaluation is locked (VRAM contention)"
            )
        q = self._queue_factory()
        process = self._process_factory(
            target=run_eval_worker, args=(config, mode, q, eval_dir)
        )
        process.start()
        deadline = time.monotonic() + timeout
        try:
            while True:
                msg = None
                try:
                    msg = q.get_nowait()
                except Empty:
                    pass
                except _DRAIN_ERRORS as exc:
                    raise RuntimeError(f"IPC pipe broken during evaluation: {exc}")

                if msg is None:
                    if not process.is_alive():
                        # child เพิ่งตาย — รอ pipe flush สั้น ๆ แล้วอ่านครั้งสุดท้าย
                        time.sleep(0.15)
                        try:
                            msg = q.get_nowait()
                        except (Empty,) + _DRAIN_ERRORS:
                            msg = None
                        if msg is None:
                            raise RuntimeError(
                                "Evaluation process died without a result"
                            )
                    elif time.monotonic() >= deadline:
                        raise RuntimeError(f"Evaluation timeout after {timeout}s")
                    else:
                        time.sleep(0.05)
                        continue

                if not validate_message(msg):
                    continue
                if msg["type"] == "error":
                    raise RuntimeError(msg["message"])
                if msg["type"] == "log":
                    if msg["text"] == EVAL_DONE:
                        break
                    with self._lock:
                        self._append_log(msg["level"], msg["text"])

            result_path = Path(eval_dir) / f"{mode}.json"
            return json.loads(result_path.read_text(encoding="utf-8"))
        finally:
            process.join(1.0)
            if process.is_alive():
                abort_process(process)


def run_predict(
    config: dict,
    prefix: str,
    suffix: str,
    *,
    timeout: float = 180.0,
    process_factory=mp.Process,
    queue_factory=mp.Queue,
) -> str:
    """Spawn predict_middle ใน process แยก แล้วรอผล — คืนข้อความที่ model generate

    - child ตายเงียบ → RuntimeError ทันที (ไม่รอจน timeout — M1)
    - หมด timeout → RuntimeError
    - ทุก path ผ่าน finally เก็บ child เสมอ — ห้ามปล่อย orphan (§5 Orphan Process, I3)
    """
    import time

    q = queue_factory()
    process = process_factory(
        target=predict_middle, args=(config, prefix, suffix, q)
    )
    process.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            msg = None
            try:
                msg = q.get_nowait()
            except Empty:
                pass
            except _DRAIN_ERRORS as exc:
                raise RuntimeError(f"IPC pipe broken during predict: {exc}")

            if msg is None:
                if not process.is_alive():
                    # child เพิ่งตาย — รอ pipe flush สั้น ๆ แล้วอ่านครั้งสุดท้าย (msg อาจส่งทัน)
                    time.sleep(0.15)
                    try:
                        msg = q.get_nowait()
                    except (Empty,) + _DRAIN_ERRORS:
                        msg = None
                    if msg is None:
                        raise RuntimeError(
                            "Predict process died without sending a result"
                        )
                elif time.monotonic() >= deadline:
                    raise RuntimeError(f"Predict timeout after {timeout}s")
                else:
                    time.sleep(0.05)
                    continue

            if not validate_message(msg):
                continue
            if msg["type"] == "log":
                return msg["text"]
            if msg["type"] == "error":
                raise RuntimeError(msg["message"])
    finally:
        process.join(1.0)
        if process.is_alive():
            abort_process(process)
