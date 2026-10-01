"""Tests สำหรับ ui/controller.py — fake Process/Queue duck-typed (test-once: ห้ามรันจนถึง Task 7)

ดู plan: docs/superpowers/plans/2026-10-01-phase4-gradio-ui.md (Task 1–2)
"""

from __future__ import annotations

import queue as stdlib_queue

from core.ipc_bridge import error_msg, log_msg, metric_msg, status_msg
from ui.controller import TrainingController


class FakeProcess:
    """duck-typed เหมือน `multiprocessing.Process` ที่ controller/abort_process ใช้"""

    def __init__(self, *, alive: bool = False, dies_on_terminate: bool = True):
        self.alive = alive
        self.dies_on_terminate = dies_on_terminate
        self.started = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.target = None
        self.args = None

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.dies_on_terminate:
            self.alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        return None


class FakeQueue:
    """duck-typed เหมือน `multiprocessing.Queue` (non-blocking get_nowait เท่านั้น)"""

    def __init__(self, messages: list | None = None):
        self.messages = list(messages or [])
        self.sent: list = []

    def put(self, msg) -> None:
        self.sent.append(msg)

    def get_nowait(self):
        if not self.messages:
            raise stdlib_queue.Empty
        return self.messages.pop(0)


def make_controller(messages: list | None = None, *, process: FakeProcess | None = None):
    """คืน (controller, fake_process, process_call_state) — queue ถูก pre-load ด้วย messages"""
    fq = FakeQueue(messages)
    fp = process if process is not None else FakeProcess()
    state = {"process_calls": 0}

    def process_factory(target, args):
        state["process_calls"] += 1
        fp.target = target
        fp.args = args
        return fp

    ctl = TrainingController(
        queue_factory=lambda: fq,
        process_factory=process_factory,
    )
    return ctl, fp, state


def valid_config(**overrides) -> dict:
    cfg = {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": "data_cache/finetune_run",
        "max_seq_length": 1024,
        "max_steps": 500,
        "code_limit": 8192,
        "lora_rank": 8,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Task 1: start + tick state machine
# ---------------------------------------------------------------------------


def test_start_rejects_invalid_config():
    ctl, _fp, state = make_controller()
    result = ctl.start({"model_id": "x"})
    assert "config ขาด key" in result
    assert "dataset_id" in result
    assert state["process_calls"] == 0  # ไม่มี spawn
    assert ctl.training_active is False


def test_start_rejects_seq_over_cap():
    ctl, _fp, state = make_controller()
    result = ctl.start(valid_config(max_seq_length=999999))
    assert "max_seq_length" in result
    assert state["process_calls"] == 0


def test_start_spawns_run_training():
    from core.trainer_worker import run_training

    ctl, fp, state = make_controller()
    cfg = valid_config()
    assert ctl.start(cfg) == "started"
    assert state["process_calls"] == 1
    assert fp.started is True
    assert fp.target is run_training
    assert fp.args[0] is cfg
    assert ctl.training_active is True


def test_tick_drains_metric_and_log():
    ctl, _fp, _state = make_controller(
        messages=[metric_msg(3, 1.5, 2e-4, 0.1), log_msg("INFO", "x")]
    )
    snap = ctl.tick()
    assert snap.metrics == [metric_msg(3, 1.5, 2e-4, 0.1)]
    assert snap.logs == ["[INFO] x"]
    # tick ซ้ำ (queue ว่าง) → ข้อมูลเดิมยังอยู่ ไม่หาย ไม่ raise
    snap2 = ctl.tick()
    assert snap2.metrics == snap.metrics
    assert snap2.logs == snap.logs


def test_status_machine_dedup_and_terminal_ignored():
    """Phase 3 ส่ง finished ซ้ำ 2 ครั้ง + ต้องไม่มี status ใหม่ทับหลัง terminal (Review Focus 1)"""
    ctl, _fp, _state = make_controller(
        messages=[
            status_msg("starting"),
            status_msg("training"),
            status_msg("finished"),
            status_msg("finished"),
            status_msg("training"),
        ]
    )
    snap = ctl.tick()
    assert snap.status == "finished"
    assert snap.training_active is False


def test_invalid_msg_becomes_error():
    ctl, _fp, _state = make_controller(messages=[{"type": "alien", "foo": 1}])
    snap = ctl.tick()
    assert snap.error is not None
    assert "alien" in snap.error
    assert snap.metrics == []
    assert snap.logs == []


def test_error_msg_surfaces_error_and_traceback_in_logs():
    ctl, _fp, _state = make_controller(
        messages=[error_msg("boom", "Traceback: ...")]
    )
    snap = ctl.tick()
    assert snap.error == "boom"
    assert "[ERROR] boom" in snap.logs
    assert "Traceback: ..." in snap.logs


def test_tick_without_start_is_safe():
    ctl, _fp, _state = make_controller()
    snap = ctl.tick()
    assert snap.status == "idle"
    assert snap.training_active is False
    assert snap.error is None and snap.watchdog is None
