"""Tests สำหรับ ui/controller.py — fake Process/Queue duck-typed (test-once: ห้ามรันจนถึง Task 7)

ดู plan: docs/superpowers/plans/2026-10-01-phase4-gradio-ui.md (Task 1–2)
"""

from __future__ import annotations

import queue as stdlib_queue

import pytest

from core.ipc_bridge import error_msg, log_msg, metric_msg, status_msg
from ui import controller as ui_controller
from ui.controller import TrainingController


class FakeProcess:
    """duck-typed เหมือน `multiprocessing.Process` ที่ controller/abort_process ใช้"""

    def __init__(
        self,
        *,
        alive: bool = False,
        dies_on_terminate: bool = True,
        dies_on_kill: bool = True,
    ):
        self.alive = alive
        self.dies_on_terminate = dies_on_terminate
        self.dies_on_kill = dies_on_kill
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
        if self.dies_on_kill:
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
    state = {"process_calls": 0, "queue": fq}

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
    assert "missing key" in result
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


# ---------------------------------------------------------------------------
# Task 2: preflight + abort + watchdog + exit
# ---------------------------------------------------------------------------


def _fake_ready_hw(**overrides) -> dict:
    hw = {
        "status": "ready",
        "device_name": "Intel(R) Arc(TM) GPU",
        "total_vram_gb": 32.0,
        "free_vram_gb": 30.0,
        "ram_available_gb": 64.0,
        "disk_free_gb": 500.0,
    }
    hw.update(overrides)
    return hw


def test_preflight_passthrough_blocked(monkeypatch):
    """hardware ไม่พร้อม → passthrough ทันที (Review Focus 5: dashboard ใช้ปิด Start)"""
    from ui import controller as controller_mod

    monkeypatch.setattr(
        controller_mod.hardware,
        "inspect",
        lambda output_dir=".": {
            "status": "insufficient_disk",
            "free_vram_gb": 5.0,
        },
    )
    ctl, _fp, _state = make_controller()
    result = ctl.preflight(valid_config())
    assert result.verdict == "insufficient_disk"
    assert result.reason  # มีเหตุผลให้แสดง


def test_preflight_safe_estimate(monkeypatch):
    from ui import controller as controller_mod

    monkeypatch.setattr(
        controller_mod.hardware,
        "inspect",
        lambda output_dir=".": _fake_ready_hw(),
    )
    ctl, _fp, _state = make_controller()
    result = ctl.preflight(valid_config(), user_params_b=None)
    assert result.verdict in ("safe", "warning")
    assert result.total_required_gb > 0


def test_abort_without_process_returns_false():
    """Review Focus 3: abort ก่อน start → False ไม่ raise"""
    ctl, _fp, _state = make_controller()
    assert ctl.abort() is False


def test_abort_kills_and_marks_terminal():
    """Review Focus 3: abort ซ้ำ/หลัง abort → terminal ทันที, training_active False"""
    fp = FakeProcess(alive=True, dies_on_terminate=False)
    ctl, _fp, _state = make_controller(process=fp)
    assert ctl.start(valid_config()) == "started"
    assert ctl.abort() is True
    assert fp.terminate_calls == 1
    assert fp.kill_calls == 1
    assert fp.is_alive() is False
    assert ctl.training_active is False
    # zombie alarm ต้องไม่ขึ้นหลัง abort (last_status terminal แล้ว)
    assert ctl.tick().watchdog is None
    # abort อีกรอบ (process เดิมตายแล้ว) ไม่ raise
    assert ctl.abort() is True


def test_zombie_watchdog_after_drain():
    """Review Focus 2: process ตายแต่ queue มี error ค้าง → error ถูกส่งก่อน + watchdog ขึ้น"""
    fp = FakeProcess(alive=False)  # ตายแล้วตั้งแต่ก่อน start tick
    ctl, _fp, state = make_controller(process=fp)
    ctl.start(valid_config())
    # start() drain queue เก่าทิ้ง (I4) — msg ของรันปัจจุบันต้อง inject หลัง start
    state["queue"].messages.extend(
        [status_msg("training"), error_msg("boom", "tb-line")]
    )
    snap = ctl.tick()
    assert snap.error == "boom"  # drain ก่อนตัดสิน
    assert "tb-line" in snap.logs
    assert snap.watchdog is not None
    assert "zombie" in snap.watchdog


def test_exit_terminates_child():
    fp = FakeProcess(alive=True, dies_on_terminate=False)
    ctl, _fp, _state = make_controller(process=fp)
    ctl.start(valid_config())
    assert fp.is_alive() is True
    ctl.exit()
    assert fp.is_alive() is False
    ctl.exit()  # เรียกซ้ำไม่ raise


# ---------------------------------------------------------------------------
# Task 3: run_predict (spawn + drain + timeout — Review Focus 4)
# ---------------------------------------------------------------------------


def _predict_factory(queue_msgs, *, alive=True):
    """คืน (process_factory, fake_predict_queue, fake_process) สำหรับ run_predict"""
    fq = FakeQueue(queue_msgs)
    holder = {}

    def process_factory(target, args):
        fp = FakeProcess(alive=alive, dies_on_terminate=True)
        fp.target = target
        fp.args = args
        holder["process"] = fp
        return fp

    return process_factory, fq, holder


def test_run_predict_success_returns_text():
    factory, fq, _h = _predict_factory([log_msg("INFO", "def f(): pass")])
    result = ui_controller.run_predict(
        valid_config(), "def f():", "return 1",
        timeout=1.0, process_factory=factory, queue_factory=lambda: fq,
    )
    assert result == "def f(): pass"
    # success ก็ต้องไม่ปล่อย child ลอย (finally เก็บเสมอ — I3)
    assert _h["process"].is_alive() is False


def test_run_predict_timeout_raises_and_kills_child():
    """Review Focus 4 + I3: queue ว่าง + child ยังอยู่ → RuntimeError ทันที + ถูก kill"""
    factory, fq, holder = _predict_factory([], alive=True)
    with pytest.raises(RuntimeError, match="timeout"):
        ui_controller.run_predict(
            valid_config(), "a", "b",
            timeout=0.1, process_factory=factory, queue_factory=lambda: fq,
        )
    assert holder["process"].terminate_calls == 1  # ไม่มี orphan (I3)


def test_run_predict_dead_child_raises_fast():
    """M1: child ตายเงียบ → RuntimeError บอกชัด ไม่ต้องรอจนหมด timeout"""
    factory, fq, _h = _predict_factory([], alive=False)
    with pytest.raises(RuntimeError, match="died"):
        ui_controller.run_predict(
            valid_config(), "a", "b",
            timeout=30.0, process_factory=factory, queue_factory=lambda: fq,
        )


def test_run_predict_error_raises():
    factory, fq, _h = _predict_factory([error_msg("no adapter", "")])
    with pytest.raises(RuntimeError, match="no adapter"):
        ui_controller.run_predict(
            valid_config(), "a", "b",
            timeout=1.0, process_factory=factory, queue_factory=lambda: fq,
        )


def test_preflight_output_dir_not_created_yet(tmp_path):
    """Regression: output_dir ยังไม่ถูกสร้าง (fresh start) → preflight ต้องไม่ crash FileNotFoundError"""
    ctl, _fp, _state = make_controller()
    cfg = valid_config(output_dir=str(tmp_path / "not_yet_created" / "run"))
    result = ctl.preflight(cfg)  # inspect ต้องใช้ ancestor ที่มีอยู่จริง
    assert result.verdict in ("safe", "warning", "blocked", "no_xpu", "insufficient_ram", "insufficient_disk")


# ---------------------------------------------------------------------------
# Review fixes: I2 (re-entrancy), I4 (run state reset), I5 (broken queue),
#               M8 (abort escalation failure)
# ---------------------------------------------------------------------------


def test_start_refused_while_training():
    """I2: start ซ้ำระหว่าง training → ไม่ spawn process ที่สอง (§5 Process Stacking)"""
    ctl, _fp, state = make_controller()
    assert ctl.start(valid_config()) == "started"
    result = ctl.start(valid_config())
    assert "already running" in result
    assert state["process_calls"] == 1  # spawn ครั้งเดียว — ไม่ orphan ตัวแรก


def test_start_resets_previous_run_state():
    """I4: เริ่มรันใหม่ → metrics/logs/error ของรันก่อนถูกล้าง (plot ห้ามปนกัน)"""
    ctl, _fp, _state = make_controller(
        messages=[metric_msg(1, 9.9, 2e-4, 0.1), status_msg("finished"),
                  error_msg("old run error", "tb")]
    )
    snap = ctl.tick()
    assert snap.metrics and snap.error == "old run error"
    assert ctl.training_active is False  # finished → start ได้อีก
    assert ctl.start(valid_config()) == "started"
    snap2 = ctl.tick()
    assert snap2.metrics == []
    assert snap2.logs == []
    assert snap2.error is None


class BrokenQueue:
    """get_nowait คาย EOFError เหมือน pipe ถูก kill กลางเขียน (I5)"""

    def put(self, msg):
        pass

    def get_nowait(self):
        raise EOFError("broken pipe")


def test_tick_survives_broken_queue():
    """I5: queue พัง → tick ต้องคืน error ไม่ใช่ raise (ไม่งั้น UI ค้างถาวร)"""
    ctl = TrainingController(queue_factory=lambda: BrokenQueue())
    snap = ctl.tick()  # ห้าม raise
    assert snap.error is not None
    assert "broken" in snap.error


def test_abort_escalation_failure_keeps_lock():
    """M8: SIGTERM+SIGKILL ไม่ตาย → คืน False + status ถอยกลับ (ปุ่มยังล็อก — process ยังอยู่)"""
    fp = FakeProcess(alive=True, dies_on_terminate=False, dies_on_kill=False)
    ctl, _fp, _state = make_controller(process=fp)
    ctl.start(valid_config())
    assert ctl.abort() is False
    assert ctl.training_active is True  # ยังปลดล็อกไม่ได้ (I3/M8)
    assert fp.is_alive() is True


# ---------------------------------------------------------------------------
# Task 5: run_eval (spawn + drain + EVAL_DONE + JSON — VRAM contention I3)
# ---------------------------------------------------------------------------


def _eval_factory(queue_msgs, *, alive=True):
    """คืน (process_factory, fake_queue, holder) สำหรับ run_eval — เหมือน _predict_factory"""
    fq = FakeQueue(queue_msgs)
    holder = {}

    def process_factory(target, args):
        fp = FakeProcess(alive=alive, dies_on_terminate=True)
        fp.target = target
        fp.args = args
        holder["process"] = fp
        return fp

    return process_factory, fq, holder


def test_run_eval_happy_path(tmp_path):
    """log ถูก append เข้า _logs (เว้น EVAL_DONE), คืน dict จาก JSON, child ถูกเก็บ"""
    import json as _json

    (tmp_path / "base.json").write_text(
        _json.dumps({"mode": "base", "n": 2, "exact_match_pct": 50.0}),
        encoding="utf-8",
    )
    factory, fq, holder = _eval_factory(
        [log_msg("INFO", "Evaluating base: 1/2"), log_msg("INFO", "EVAL_DONE")]
    )
    ctl = TrainingController(
        queue_factory=lambda: fq, process_factory=factory
    )
    result = ctl.run_eval(valid_config(), "base", eval_dir=tmp_path, timeout=1.0)
    assert result["mode"] == "base"
    assert result["n"] == 2
    assert any("Evaluating base: 1/2" in line for line in ctl._logs)
    assert not any("EVAL_DONE" in line for line in ctl._logs)  # marker ไม่โชว์ UI
    assert holder["process"].is_alive() is False  # finally เก็บ child เสมอ (I3)


def test_run_eval_rejects_while_training():
    """Review Focus #3: eval ชน training → VRAM stacking → ต้อง raise ก่อน spawn"""
    factory, fq, _h = _eval_factory([])
    ctl = TrainingController(queue_factory=lambda: fq, process_factory=factory)
    ctl.start(valid_config())  # spawn → training_active = True
    with pytest.raises(RuntimeError, match="VRAM contention"):
        ctl.run_eval(valid_config(), "base", eval_dir="/tmp/x", timeout=1.0)
    from core.trainer_worker import run_training

    assert _h["process"].target is run_training  # ยังเป็น process ของ training — ไม่ spawn ซ้อน


def test_run_eval_error_raises_and_cleans_up(tmp_path):
    """error_msg จาก child → RuntimeError + finally เก็บ child (I3)"""
    factory, fq, holder = _eval_factory(
        [error_msg("eval exploded", "tb here")], alive=True
    )
    ctl = TrainingController(queue_factory=lambda: fq, process_factory=factory)
    with pytest.raises(RuntimeError, match="eval exploded"):
        ctl.run_eval(valid_config(), "base", eval_dir=tmp_path, timeout=1.0)
    assert holder["process"].is_alive() is False


def test_run_eval_dead_child_raises(tmp_path):
    """child ตายเงียบ → RuntimeError ทันที ไม่รอ timeout (M1)"""
    factory, fq, holder = _eval_factory([], alive=False)
    ctl = TrainingController(queue_factory=lambda: fq, process_factory=factory)
    with pytest.raises(RuntimeError, match="died without"):
        ctl.run_eval(valid_config(), "base", eval_dir=tmp_path, timeout=5.0)
    assert holder["process"] is not None


# ---------------------------------------------------------------------------
# M4: สถานะ "aborting" ระหว่าง kill (Fix.md) — ไม่ใช่ terminal ปลอม ๆ อีกต่อไป
# ---------------------------------------------------------------------------


def test_abort_shows_aborting_while_killing():
    """M4: อ่าน status ระหว่าง abort_process → ต้องเป็น "aborting" (เดิม: "aborted" ตั้งก่อน kill)"""
    fp = FakeProcess(alive=True, dies_on_terminate=False, dies_on_kill=True)
    ctl, _fp, _state = make_controller(process=fp)
    assert ctl.start(valid_config()) == "started"

    seen = {}
    orig_terminate = fp.terminate

    def terminate_hook():
        seen["status"] = ctl.tick().status  # tick ไม่ถูกบล็อก (lock ปล่อยก่อน kill — M9)
        orig_terminate()

    fp.terminate = terminate_hook
    assert ctl.abort() is True
    assert seen["status"] == "aborting"
    assert ctl.tick().status == "aborted"  # ตายจริงแล้วค่อยเป็น terminal


def test_status_message_cannot_overwrite_aborting():
    """M4: child ยังส่ง status ระหว่าง kill ได้ — ห้ามทับ "aborting" (เดิม: ไม่ guard)"""
    fp = FakeProcess(alive=True)
    ctl, _fp, state = make_controller(process=fp)
    ctl.start(valid_config())
    ctl._status = "aborting"  # จำลองระหว่าง kill
    state["queue"].messages.append(status_msg("training"))
    snap = ctl.tick()
    assert snap.status == "aborting"


def test_watchdog_silent_while_aborting():
    """M4: process ตายระหว่าง kill ที่ตั้งใจ → ห้ามกรี๊ด zombie ปลอม (เดิม: เห็น non-terminal → alarm)"""
    fp = FakeProcess(alive=False)
    ctl, _fp, _state = make_controller(process=fp)
    ctl.start(valid_config())
    ctl._status = "aborting"
    snap = ctl.tick()
    assert snap.watchdog is None


def test_training_active_locked_while_aborting():
    """M4: ระหว่าง "aborting" ปุ่มต้องล็อกต่อ (process ยังอาจอยู่) — ต่างจากเดิมที่เปิดด้วย "aborted" ก่อน kill"""
    fp = FakeProcess(alive=True)
    ctl, _fp, _state = make_controller(process=fp)
    ctl.start(valid_config())
    ctl._status = "aborting"
    assert ctl.training_active is True
