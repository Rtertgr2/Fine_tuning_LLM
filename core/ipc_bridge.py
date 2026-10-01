"""Message Protocol, Watchdog & Abort — สะพานสื่อสารระหว่าง training subprocess กับ UI (สเปก plan.md §4.5)

ส่งข้อความผ่าน `multiprocessing.Queue` 4 ประเภทตามสเปก:
  metric → กราฟ loss/LR | log → Live Log | status → สลับสถานะปุ่ม | error → Log Viewer (full traceback)

Watchdog: `watchdog_error(process, last_status)` เรียกทุกวินาทีจาก UI (Phase 4)
Abort: `abort_process(process)` = SIGTERM → รอ 10 วิ → SIGKILL (คืน XPU VRAM)
"""

from __future__ import annotations

from configs.safe_defaults import ABORT_SIGTERM_TIMEOUT_S

TRAINING_STATUSES: tuple[str, ...] = (
    "starting",
    "training",
    "saving",
    "finished",
    "aborted",
)
TERMINAL_STATUSES: frozenset[str] = frozenset({"finished", "aborted"})

_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "metric": ("step", "loss", "lr", "epoch"),
    "log": ("level", "text"),
    "status": ("state",),
    "error": ("message", "traceback"),
}


def metric_msg(step: int, loss: float, lr: float, epoch: float) -> dict:
    return {"type": "metric", "step": step, "loss": loss, "lr": lr, "epoch": epoch}


def log_msg(level: str, text: str) -> dict:
    return {"type": "log", "level": level, "text": text}


def status_msg(state: str) -> dict:
    if state not in TRAINING_STATUSES:
        raise ValueError(
            f"สถานะไม่รู้จัก: {state!r} (ต้องเป็นหนึ่งใน {TRAINING_STATUSES})"
        )
    return {"type": "status", "state": state}


def error_msg(message: str, traceback_text: str) -> dict:
    return {"type": "error", "message": message, "traceback": traceback_text}


def validate_message(msg: object) -> bool:
    """ตรวจว่า msg เป็นหนึ่งใน 4 type ที่รู้จักและ payload keys ครบ"""
    if not isinstance(msg, dict):
        return False
    required = _REQUIRED_KEYS.get(msg.get("type"))
    if required is None:
        return False
    return all(key in msg for key in required)


def watchdog_error(process, last_status: str | None) -> dict | None:
    """Zombie detection: process ตายโดยไม่จบเอง (finished/aborted) → คืน error message

    เรียกจาก UI ทุก 1 วินาที — ถ้าปล่อยไว้ UI จะค้างรอ Queue ตลอดกาล
    """
    if process.is_alive():
        return None
    if last_status in TERMINAL_STATUSES:
        return None
    return error_msg(
        "Training process ตายโดยไม่ได้ส่ง finished/aborted (zombie)",
        f"last_status={last_status!r}",
    )


def abort_process(process, *, timeout: float = ABORT_SIGTERM_TIMEOUT_S) -> bool:
    """SIGTERM → รอ `timeout` วิ → ยังไม่ตาย → SIGKILL — คืน True ถ้า process ตายสนิท

    ทำงานกับ `multiprocessing.Process` หรือ object ที่มี
    `is_alive()/terminate()/kill()/join(timeout)` (duck-typed)
    """
    process.terminate()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(1.0)
    return not process.is_alive()
