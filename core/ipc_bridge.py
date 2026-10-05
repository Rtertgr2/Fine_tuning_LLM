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
    "val_metric": ("step", "epoch", "val_loss"),
    "log": ("level", "text"),
    "status": ("state",),
    "error": ("message", "traceback"),
}


def metric_msg(step: int, loss: float, lr: float, epoch: float) -> dict:
    return {"type": "metric", "step": step, "loss": loss, "lr": lr, "epoch": epoch}


def val_metric_msg(step: int, epoch: float, val_loss: float) -> dict:
    """val loss ระหว่างเทรน — schema ของตัวเอง (validate เข้ม: metric ต้องครบทั้ง 4 key)"""
    return {"type": "val_metric", "step": step, "epoch": epoch, "val_loss": val_loss}


def log_msg(level: str, text: str) -> dict:
    return {"type": "log", "level": level, "text": text}


def status_msg(state: str) -> dict:
    if state not in TRAINING_STATUSES:
        raise ValueError(
            f"unknown status: {state!r} (must be one of {TRAINING_STATUSES})"
        )
    return {"type": "status", "state": state}


def error_msg(message: str, traceback_text: str) -> dict:
    return {"type": "error", "message": message, "traceback": traceback_text}


def validate_message(msg: object) -> bool:
    """ตรวจว่า msg เป็นหนึ่งใน 4 type ที่รู้จัก, payload keys ครบ และชนิดถูก"""
    if not isinstance(msg, dict):
        return False
    msg_type = msg.get("type")
    required = _REQUIRED_KEYS.get(msg_type)
    if required is None or not all(key in msg for key in required):
        return False
    if msg_type == "metric":
        return all(
            isinstance(msg[key], (int, float)) for key in ("step", "loss", "lr", "epoch")
        )
    if msg_type == "val_metric":
        return all(
            isinstance(msg[key], (int, float)) for key in ("step", "epoch", "val_loss")
        )
    if msg_type == "log":
        return isinstance(msg["level"], str) and isinstance(msg["text"], str)
    if msg_type == "status":
        return msg["state"] in TRAINING_STATUSES
    return isinstance(msg["message"], str) and isinstance(msg["traceback"], str)


def watchdog_error(process, last_status: str | None) -> dict | None:
    """Zombie detection: process ตายโดยไม่จบเอง (finished/aborted) → คืน error message

    เรียกจาก UI ทุก 1 วินาที — ถ้าปล่อยไว้ UI จะค้างรอ Queue ตลอดกาล
    """
    if process.is_alive():
        return None
    if last_status in TERMINAL_STATUSES or last_status == "aborting":
        return None  # "aborting" = kill ที่ตั้งใจอยู่ (M4) — ตายตรงนี้ไม่ใช่ zombie
    return error_msg(
        "Training process died without sending finished/aborted (zombie)",
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
