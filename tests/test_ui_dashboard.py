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
    # gradio 6: ข้อความปุ่มอยู่ที่ .value (label ถูกใช้กับ event metadata)
    labels = {b.value for b in buttons}
    assert "Predict Middle" in labels
    assert "Merge & Export Full Weights" in labels
    assert "Start Fine-Tuning" in labels
    assert "Abort Process" in labels
    start = next(b for b in buttons if b.value == "Start Fine-Tuning")
    assert start.interactive is False  # ยังไม่ผ่าน preflight → ปิดอยู่


def test_collect_config_tolerates_cleared_number_fields():
    """M7: ช่อง Number ถูกล้าง (None) → ใช้ default จาก safe_defaults ไม่ใช่ crash int(None)"""
    from configs.safe_defaults import (
        LORA_RANK_DEFAULT,
        MAX_SEQ_LENGTH_DEFAULT,
        MAX_STEPS,
        TRAIN_CODE_LIMIT,
    )
    from ui.dashboard import _collect_config

    config, params = _collect_config(
        "m/x", None, "ds", "col", "qwen",
        None, None, None, None, "out",
    )
    assert params is None
    assert config["lora_rank"] == LORA_RANK_DEFAULT
    assert config["max_seq_length"] == MAX_SEQ_LENGTH_DEFAULT
    assert config["max_steps"] == MAX_STEPS
    assert config["code_limit"] == TRAIN_CODE_LIMIT


# ---------------------------------------------------------------------------
# Task 6: Run Evaluation (handler + widgets + tick lock)
# ---------------------------------------------------------------------------


class EvalController:
    """controller ปลอมสำหรับ on_eval — บันทึกการเรียก + คืนค่าที่กำหนด"""

    training_active = False

    def __init__(self, *, fail_on: str | None = None, n_base: int = 2, n_fine: int = 3):
        self.calls: list[str] = []
        self.fail_on = fail_on
        self._n_base = n_base
        self._n_fine = n_fine

    def run_eval(self, config, mode, **kw):
        self.calls.append(mode)
        if self.fail_on == mode:
            raise RuntimeError(f"{mode} exploded")
        return {
            "mode": mode,
            "n": self._n_base if mode == "base" else self._n_fine,
            "exact_match_pct": 10.0 if mode == "base" else 40.0,
            "token_f1_mean": 0.2 if mode == "base" else 0.5,
            "per_case": [
                {
                    "i": 0,
                    "exact": mode == "finetuned",
                    "f1": 1.0 if mode == "finetuned" else 0.0,
                    "pred": "right" if mode == "finetuned" else "wrong",
                    "gt": "right",
                }
            ],
            "samples": [],
        }

    # methods ที่ build_dashboard ไม่เรียกตอนสร้าง แต่ใส่ครบตาม duck-type เดิม
    def preflight(self, config, user_params_b=None):
        raise NotImplementedError

    def start(self, config):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


def _cfg_args():
    """ค่า cfg ครบ 10 ช่องตาม cfg_inputs ของ build_dashboard"""
    return ("m/x", None, "ds", "col", "qwen", 8, 1024, 6, 64, "out")


def test_on_eval_runs_base_then_finetuned_sequentially():
    """รัน base → finetuned เรียงกัน (คนละ process, ไม่ stacking §5) → compare_results"""
    from ui.dashboard import _make_handlers

    ctl = EvalController()
    h = _make_handlers(ctl)
    rows, err = h["on_eval"](*_cfg_args())
    assert ctl.calls == ["base", "finetuned"]
    assert err == ""
    # rows = [EM, F1, n] — fine ดีกว่า base ทุกแถว
    assert rows[0][0] == "Exact Match %"
    assert float(rows[0][1]) < float(rows[0][2])
    assert rows[2][1] == "2" and rows[2][2] == "3"


def test_on_eval_error_returns_english_message():
    """exception → (None, English markdown) — ห้าม raise ออกจาก handler (ภาษาอังกฤษเสมอ)"""
    from ui.dashboard import _make_handlers

    ctl = EvalController(fail_on="base")
    h = _make_handlers(ctl)
    rows, err = h["on_eval"](*_cfg_args())
    assert rows is None
    assert err.startswith("**Evaluation failed:**")
    assert "base exploded" in err


def test_build_dashboard_has_eval_widgets():
    """Tab 3 มี eval_btn/eval_table/eval_error_md (headers ตรง spec §3.4)"""
    demo = build_dashboard(FakeController())
    blocks = list(_iter_blocks(demo))
    buttons = [b for b in blocks if isinstance(b, gr.Button)]
    assert "Run Evaluation" in {b.value for b in buttons}
    tables = [b for b in blocks if isinstance(b, gr.Dataframe)]
    assert len(tables) == 1
    assert list(tables[0].headers) == ["Metric", "Base", "Fine-tuned", "Δ"]


def test_on_tick_returns_nine_outputs():
    """on_tick คืน 9 แถว (ตัวที่ 9 = eval_btn lock ระหว่างเทรน §5)"""

    class TickController:
        training_active = True
        def preflight(self, *a, **k): raise NotImplementedError
        def start(self, *a): raise NotImplementedError
        def abort(self): raise NotImplementedError
        def tick(self):
            from types import SimpleNamespace
            return SimpleNamespace(
                status="training", metrics=[], logs=[], error=None,
                watchdog=None, training_active=True,
            )

    from ui.dashboard import _make_handlers

    h = _make_handlers(TickController())
    out = h["on_tick"](None)
    assert len(out) == 9
    eval_lock = out[8]
    assert isinstance(eval_lock, gr.Button)
    assert eval_lock.interactive is False  # training_active → ปุ่ม eval ถูกล็อก
