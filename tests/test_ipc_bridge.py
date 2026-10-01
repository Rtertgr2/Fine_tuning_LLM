"""Tests สำหรับ core/ipc_bridge.py — message protocol 4 ประเภท, watchdog, abort escalation (สเปก plan.md §4.5)"""

import pytest

from core import ipc_bridge as ipc


class FakeProcess:
    """จำลอง multiprocessing.Process — นับจำนวน call + กำหนดว่า terminate แล้วตายไหม"""

    def __init__(self, dies_on_terminate: bool = True, alive: bool = True):
        self._alive = alive
        self.dies_on_terminate = dies_on_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_timeouts: list = []

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.dies_on_terminate:
            self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)


def test_metric_msg_shape():
    msg = ipc.metric_msg(step=10, loss=1.5, lr=2e-4, epoch=0.2)
    assert msg == {"type": "metric", "step": 10, "loss": 1.5, "lr": 2e-4, "epoch": 0.2}
    assert ipc.validate_message(msg) is True


def test_log_and_error_msg():
    log = ipc.log_msg("ERROR", "boom happened")
    assert log == {"type": "log", "level": "ERROR", "text": "boom happened"}
    assert ipc.validate_message(log) is True

    tb = "Traceback (most recent call last):\n  ValueError: x"
    err = ipc.error_msg("training ล้มเหลว", tb)
    assert err == {"type": "error", "message": "training ล้มเหลว", "traceback": tb}
    assert "ValueError: x" in err["traceback"]  # full traceback ต้องไปถึง UI ครบทุกบรรทัด
    assert ipc.validate_message(err) is True


def test_status_msg_states():
    assert len(ipc.TRAINING_STATUSES) == 5
    for state in ipc.TRAINING_STATUSES:
        assert ipc.status_msg(state) == {"type": "status", "state": state}
        assert ipc.validate_message(ipc.status_msg(state)) is True
    with pytest.raises(ValueError):
        ipc.status_msg("done")  # state นอกชุด → fail fast ห้ามส่ง msg รูปผิดให้ UI เดางาน


def test_validate_message_rejects_bad():
    assert ipc.validate_message("not a dict") is False
    assert ipc.validate_message({"payload": 1}) is False  # ไม่มี type
    assert ipc.validate_message({"type": "metric", "step": 1}) is False  # payload key หาย
    assert ipc.validate_message({"type": "alien"}) is False  # type ไม่รู้จัก


def test_watchdog_detects_zombie():
    dead = FakeProcess(alive=False)
    err = ipc.watchdog_error(dead, "training")
    assert err is not None
    assert err["type"] == "error"
    assert ipc.validate_message(err) is True
    # ตายก่อนส่ง status ใด ๆ → ก็เป็น zombie ต้องจับให้ได้
    assert ipc.watchdog_error(FakeProcess(alive=False), None) is not None


def test_watchdog_silent_when_terminal():
    # ยังมีชีวิต → เงียบ
    assert ipc.watchdog_error(FakeProcess(alive=True), "training") is None
    # ตายแต่จบเองแล้ว (finished/aborted) → เงียบ
    dead = FakeProcess(alive=False)
    assert ipc.watchdog_error(dead, "finished") is None
    assert ipc.watchdog_error(dead, "aborted") is None


def test_abort_escalates_to_kill():
    p = FakeProcess(dies_on_terminate=False)  # terminate แล้วยังไม่ตาย
    dead = ipc.abort_process(p, timeout=10.0)
    assert p.terminate_calls == 1
    assert p.join_timeouts and p.join_timeouts[0] == 10.0  # รอ 10 วิ ตามสเปก
    assert p.kill_calls == 1  # ยังไม่ตาย → SIGKILL (คืน VRAM)
    assert dead is True


def test_abort_no_kill_when_terminate_works():
    p = FakeProcess(dies_on_terminate=True)  # SIGTERM ก็ตายแล้ว
    dead = ipc.abort_process(p, timeout=10.0)
    assert p.terminate_calls == 1
    assert p.kill_calls == 0  # ตายแล้วห้าม kill ซ้ำ
    assert dead is True
