"""Tests สำหรับ ui/dashboard.py — โครงสร้าง 3 แท็บ + cap/defaults (test-once: ห้ามรันจนถึง Task 7)"""

from __future__ import annotations

import gradio as gr

from configs.safe_defaults import MAX_SEQ_LENGTH_CAP, MAX_SEQ_LENGTH_DEFAULT
from ui.dashboard import build_dashboard


class FakeController:
    """duck-typed พอสำหรับ build (ไม่มีการเรียก method ตอนสร้าง Blocks)"""

    training_active = False

    def preflight(self, config, user_params_b=None):
        raise NotImplementedError

    def start(self, config):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


def _iter_blocks(block):
    yield block
    for child in getattr(block, "children", []) or []:
        yield from _iter_blocks(child)


def test_build_dashboard_structure():
    demo = build_dashboard(FakeController())
    assert isinstance(demo, gr.Blocks)

    blocks = list(_iter_blocks(demo))
    tab_labels = [b.label for b in blocks if isinstance(b, gr.Tab)]
    assert tab_labels == [
        "Configuration & Pre-flight",
        "Training Mission Control",
        "Playground & Export",
    ]

    timers = [b for b in blocks if isinstance(b, gr.Timer)]
    assert len(timers) >= 1
    assert timers[0].value == 1.0  # อัปเดตทุก 1 วินาที (§4.3)

    seq_sliders = [
        b for b in blocks
        if isinstance(b, gr.Slider) and b.maximum == MAX_SEQ_LENGTH_CAP
    ]
    assert len(seq_sliders) == 1  # max_seq_length: cap 2048 ตาม §5 — ช่องเดียวที่ชนเพดาน
    assert seq_sliders[0].value == MAX_SEQ_LENGTH_DEFAULT


def test_build_dashboard_locks_exist():
    """ปุ่ม Predict/Merge ต้องถูกสร้าง (interactivity อัปเดตโดย tick — Task 1/2)"""
    demo = build_dashboard(FakeController())
    buttons = [b for b in _iter_blocks(demo) if isinstance(b, gr.Button)]
    labels = {b.label for b in buttons}
    assert "Predict Middle" in labels
    assert "Merge & Export Full Weights" in labels
    assert "Start Fine-Tuning" in labels
    assert "Abort Process" in labels
    start = next(b for b in buttons if b.label == "Start Fine-Tuning")
    assert start.interactive is False  # ยังไม่ผ่าน preflight → ปิดอยู่
