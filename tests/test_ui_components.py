"""Tests สำหรับ ui/components.py — plot รับ raw data, สี gauge ตาม verdict (test-once: ห้ามรันจนถึง Task 7)"""

from __future__ import annotations

import matplotlib

from core.ipc_bridge import metric_msg
from ui.components import build_metric_plot, verdict_style


def test_build_metric_plot_two_axes():
    metrics = [metric_msg(1, 2.0, 2e-4, 0.1), metric_msg(2, 1.5, 1.8e-4, 0.2)]
    fig = build_metric_plot(metrics)
    assert len(fig.axes) == 2  # loss (left) + lr (right)
    ax_loss = fig.axes[0]
    line = ax_loss.lines[0]
    assert list(line.get_xdata()) == [1, 2]
    assert list(line.get_ydata()) == [2.0, 1.5]


def test_build_metric_plot_empty():
    fig = build_metric_plot([])
    assert len(fig.axes) == 2  # ไม่ raise, axes ครบ
    assert fig.axes[0].lines[0].get_xdata().size == 0


def test_verdict_style_colors():
    assert verdict_style("safe") == ("safe", "#16a34a")
    assert verdict_style("warning") == ("warning", "#ca8a04")
    # blocked + passthrough (no_xpu/insufficient_*) = แดงเท่ากัน
    assert verdict_style("blocked") == ("blocked", "#dc2626")
    assert verdict_style("no_xpu") == ("no_xpu", "#dc2626")
    assert verdict_style("insufficient_disk") == ("insufficient_disk", "#dc2626")
