"""UI component builders — รับ raw data แล้วคืน figure/style อย่างเดียว (data contract §4.3/§4.6)

ห้ามตีความ/แปลงค่า — รับ output ดิบจาก hardware/ipc เท่านั้น
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # ปลอดภัยทั้งตอน test และตอน Gradio ใช้คนละ thread

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
    """วาด Loss (แกนซ้าย) + LR (แกนขวา) จาก metric/val_metric ดิบ — ว่างก็คืน figure เปล่า

    P1 C5: ใช้ Figure ตรง ๆ ไม่ผ่าน pyplot — plt.subplots() = global figure registry
    ที่ Gradio threads หลายตัวแย่งกันเขียน (race); P1 C4: legend รวมทั้งสองแกน
    """
    fig = Figure(figsize=(7, 3))
    ax_loss = fig.add_subplot(111)
    ax_lr = ax_loss.twinx()

    train = [m for m in metrics if "loss" in m]  # val_metric ไม่มี loss/lr
    steps = [m["step"] for m in train]
    losses = [m["loss"] for m in train]
    lrs = [m.get("lr", 0.0) for m in train]

    ax_loss.plot(steps, losses, color="#2563eb", label="loss")
    ax_lr.plot(steps, lrs, color="#9333ea", label="lr")
    val_points = [(m["step"], m["val_loss"]) for m in metrics if "val_loss" in m]
    if val_points:
        val_steps, val_losses = zip(*val_points)
        ax_loss.plot(
            val_steps,
            val_losses,
            linestyle="None",
            marker="o",
            color="#f97316",
            label="val_loss",
        )
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss", color="#2563eb")
    ax_lr.set_ylabel("lr", color="#9333ea")
    ax_loss.set_title("Training metrics")
    # C4: รวม handles จากทั้งสองแกน (twinx แยก collection — เอาเฉพาะแกนซ้ายจะหาย lr)
    handles, labels = [], []
    for ax in (ax_loss, ax_lr):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles += ax_handles
        labels += ax_labels
    ax_loss.legend(handles, labels, loc="best")
    fig.tight_layout()
    return fig
