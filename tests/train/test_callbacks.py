"""Tests สำหรับ core/train/callbacks.py — NanGuard + StreamToQueueCallback + AtomicSaveTrainer (สเปก plan.md §4.5)"""

from types import SimpleNamespace

from conftest import FakeQueue
from core.infra import ipc_bridge as ipc
from core.train import callbacks as wa


def test_nan_guard_aborts_on_three_consecutive():
    g = wa.NanGuard()
    assert g.register(1.0) is False
    assert g.register(float("nan")) is False  # 1
    assert g.register(float("inf")) is False  # 2
    assert g.register(float("-inf")) is True  # ครบ 3 ติดต → สั่ง abort


def test_nan_guard_resets_on_finite():
    g = wa.NanGuard()
    g.register(float("nan"))
    g.register(float("nan"))
    assert g.register(0.5) is False  # finite คั่น → reset streak
    assert g.register(float("nan")) is False  # นับใหม่ = 1
    assert g.register(float("nan")) is False  # = 2 ยังไม่ครบ 3 (false positive กันไว้)


def _cb_with_states():
    q = FakeQueue()
    cb = wa.StreamToQueueCallback(q)
    state = SimpleNamespace(global_step=10)
    control = SimpleNamespace(should_training_stop=False)
    return q, cb, state, control


def test_on_log_emits_metric():
    q, cb, state, control = _cb_with_states()
    cb.on_log(
        None,
        state,
        control,
        {"loss": 1.5, "learning_rate": 2e-4, "step": 10, "epoch": 0.2},
    )
    assert len(q.messages) == 1  # loss key → metric แท่งเดียว ไม่ปน log
    msg = q.messages[0]
    assert msg["type"] == "metric"
    assert ipc.validate_message(msg) is True
    assert msg["lr"] == 2e-4
    assert msg["step"] == 10
    assert msg["loss"] == 1.5
    assert msg["epoch"] == 0.2


def test_on_log_without_lr_uses_cache_not_zero():
    """P1 A3: lr หาย = อย่าโชว์ 0.0 เหมือนจริง — ใช้ค่าล่าสุดที่เห็นจริง"""
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"loss": 1.5, "learning_rate": 2e-4, "step": 1})
    cb.on_log(None, state, control, {"loss": 1.4, "step": 2})
    metrics = [m for m in q.messages if m["type"] == "metric"]
    assert len(metrics) == 2
    assert metrics[1]["lr"] == 2e-4  # ใช้ cache — ไม่ใช่ 0.0


def test_on_log_lr_never_seen_sends_warning_not_metric():
    """P1 A3: ยังไม่เคยเห็น lr เลย → ห้าม emit metric ปลอม ต้องมี WARNING แทน"""
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"loss": 1.5, "step": 1})
    assert all(m["type"] != "metric" for m in q.messages)
    warn = next(m for m in q.messages if m["type"] == "log")
    assert warn["level"] == "WARNING" and "learning_rate" in warn["text"]
    assert ipc.validate_message(warn) is True


def test_status_flow():
    q, cb, state, control = _cb_with_states()
    cb.on_train_begin(None, state, control)
    cb.on_save(None, state, control)
    cb.on_train_end(None, state, control)
    assert all(m["type"] == "status" for m in q.messages)
    assert [m["state"] for m in q.messages] == ["training", "saving", "finished"]


def test_nan_trip_sends_error_and_stops():
    q, cb, state, control = _cb_with_states()
    for _ in range(3):
        cb.on_log(None, state, control, {"loss": float("nan"), "step": 10})
    # trip ครั้งที่ 3: metric + error + status aborted
    assert cb.aborted is True
    assert control.should_training_stop is True
    assert len(q.messages) == 5  # 3 metric + error + status
    assert q.messages[-2]["type"] == "error"
    assert q.messages[-1] == {"type": "status", "state": "aborted"}
    assert ipc.validate_message(q.messages[-2]) is True
    # aborted แล้ว → on_train_end ห้ามส่ง finished
    cb.on_train_end(None, state, control)
    assert len(q.messages) == 5
    assert all(m.get("state") != "finished" for m in q.messages)


def test_on_log_returns_control_explicitly():
    """A4: on_log ทุกทางออกคืน control ชัดเจน — เดิม mutation อย่างเดียว อ่านไม่ออกว่าจบ method (รายงาน A4)"""
    q, cb, state, control = _cb_with_states()          # fixture เดิมจาก test_nan_trip :196
    ret = cb.on_log(None, state, control, {"loss": 0.5, "step": 1})
    assert ret is control                              # path ปกติ
    for _ in range(3):
        ret = cb.on_log(None, state, control, {"loss": float("nan"), "step": 10})
    assert ret is control                              # trip path
    assert control.should_training_stop is True
    ret = cb.on_log(None, state, control, {"loss": 0.4, "step": 11})
    assert ret is control                              # path หลัง trip


def test_non_metric_logs_forwarded():
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"train_runtime": 100.0})
    assert len(q.messages) == 1
    msg = q.messages[0]
    assert msg["type"] == "log"
    assert ipc.validate_message(msg) is True


def test_on_log_eval_loss_sends_val_metric():
    # eval log มาในรูป {"eval_loss": ...} (ไม่มี "loss") → ต้องเป็น val_metric ไม่ใช่ log ดิบ
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"eval_loss": 0.9, "step": 10, "epoch": 0.5})
    assert len(q.messages) == 1
    msg = q.messages[0]
    assert msg == {"type": "val_metric", "step": 10, "epoch": 0.5, "val_loss": 0.9}
    assert ipc.validate_message(msg) is True


def test_on_save_suppressed_after_abort():
    q, cb, state, control = _cb_with_states()
    for _ in range(3):
        cb.on_log(None, state, control, {"loss": float("nan"), "step": 10})
    n = len(q.messages)
    cb.on_save(None, state, control)
    # aborted แล้วห้าม status ถอยหลัง — ไม่งั้นค่าสุดท้ายไม่ใช่ terminal
    # → watchdog คาย "zombie" false alarm ใน Phase 4
    assert len(q.messages) == n
