"""UI component builders — รับ raw data แล้วคืน figure/style อย่างเดียว (data contract §4.3/§4.6)

ห้ามตีความ/แปลงค่า — รับ output ดิบจาก hardware/ipc เท่านั้น
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # ปลอดภัยทั้งตอน test และตอน Gradio ใช้คนละ thread

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# สีตาม verdict — label คงเป็นค่า verdict เดิม (ห้ามแปลงเป็นข้อความอื่น)
_VERDICT_COLORS = {
    "safe": "#16a34a",
    "warning": "#ca8a04",
    "blocked": "#dc2626",
}
_PASSTHROUGH_FALLBACK_COLOR = "#dc2626"  # no_xpu / insufficient_ram / insufficient_disk / อื่น ๆ


def verdict_style(verdict: str) -> tuple[str, str]:
    """คืน (label, color) — label = verdict เดิม, color ตามชนิด (เขียว/เหลือง/แดง)"""
    return verdict, _VERDICT_COLORS.get(verdict, _PASSTHROUGH_FALLBACK_COLOR)


def build_metric_plot(metrics: list[dict]) -> Figure:
    """วาด Loss (แกนซ้าย) + LR (แกนขวา) จาก list ของ metric_msg ดิบ — ว่างก็คืน figure เปล่า"""
    fig, ax_loss = plt.subplots(figsize=(7, 3))
    ax_lr = ax_loss.twinx()

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    lrs = [m["lr"] for m in metrics]

    ax_loss.plot(steps, losses, color="#2563eb", label="loss")
    ax_lr.plot(steps, lrs, color="#9333ea", label="lr")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss", color="#2563eb")
    ax_lr.set_ylabel("lr", color="#9333ea")
    ax_loss.set_title("Training metrics")
    fig.tight_layout()
    return fig
