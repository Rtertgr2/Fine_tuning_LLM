"""Tests สำหรับ core/eval/worker.py — run_eval_worker spawn target (progress + EVAL_DONE)"""

import pytest

from conftest import FakeQueue
from core.eval import worker as wa


def test_run_eval_worker_sends_progress_and_done(monkeypatch):
    """spawn target: progress → log_msg ทุก step, จบ → EVAL_DONE (UI ใช้จับจบ)"""
    from core.eval import evaluator as ev_mod

    def fake_run_eval(config, *, mode, eval_dir, progress=None):
        progress(1, 3)
        progress(3, 3)
        return {"mode": mode, "n": 3}

    monkeypatch.setattr(ev_mod, "run_eval", fake_run_eval)
    q = FakeQueue()
    wa.run_eval_worker({"any": 1}, "base", q, eval_dir="data_cache/eval")
    texts = [m["text"] for m in q.messages if m.get("type") == "log"]
    assert "Evaluating base: 1/3" in texts
    assert "Evaluating base: 3/3" in texts
    assert texts[-1] == "EVAL_DONE"
    assert not [m for m in q.messages if m.get("type") == "error"]


def test_run_eval_worker_error_sends_error_and_raises(monkeypatch):
    """ผิด → error_msg (message + traceback) ก่อน raise — pattern predict_middle"""
    from core.eval import evaluator as ev_mod

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ev_mod, "run_eval", boom)
    q = FakeQueue()
    with pytest.raises(RuntimeError, match="boom"):
        wa.run_eval_worker({"any": 1}, "base", q, eval_dir="data_cache/eval")
    errs = [m for m in q.messages if m.get("type") == "error"]
    assert len(errs) == 1
    assert errs[0]["message"] == "boom"
    assert "RuntimeError" in errs[0]["traceback"]
