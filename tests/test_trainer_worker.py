"""Tests สำหรับ core/trainer_worker.py — build_training_args pin hyperparameters จาก safe_defaults (สเปก plan.md §4.4)"""

from types import SimpleNamespace

import pytest

from core import ipc_bridge as ipc
from core import trainer_worker as wa


def test_args_pin_hyperparams():
    args = wa.build_training_args("data_cache/x")
    assert args.learning_rate == 2e-4
    assert args.warmup_steps == 15  # round(0.03 × 500) — transformers 5.x ไม่มี warmup_ratio
    assert args.gradient_accumulation_steps == 8
    assert args.per_device_train_batch_size == 1
    assert args.seed == 42
    assert args.save_total_limit == 2
    assert args.save_steps == 100
    assert args.max_steps == 500
    assert args.gradient_checkpointing is True
    assert args.optim == "adamw_torch"
    assert args.bf16 is True
    assert args.max_length == 1024
    assert args.max_length <= 2048  # hard cap §5 — context length ห้ามเกิน 2048
    assert args.packing is True
    assert args.dataset_text_field == "text"


def test_args_overrides_for_smoke():
    args = wa.build_training_args("out", max_steps=6, save_steps=2)
    assert args.max_steps == 6
    assert args.save_steps == 2
    assert args.warmup_steps == 0  # round(0.03 × 6) = 0


class FakeTokenizer:
    """จำลอง tokenizer สำหรับ FIM guard — จงใจไม่มี method resize (ถ้าโค้ดเรียก = AttributeError ทันที)"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = dict(vocab)
        self.unk_token_id = 0  # id 0 = UNK เหมือน HF convention

    def __len__(self) -> int:
        return 30500

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, tid: int) -> str:
        for tok, i in self.vocab.items():
            if i == tid:
                return tok
        return "<unk>"

    def decode(self, tid: int) -> str:
        return self.convert_ids_to_tokens(tid)


FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


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


def test_ensure_fim_tokens_ok():
    tok = FakeTokenizer({"<|fim_prefix|>": 100, "<|fim_suffix|>": 101, "<|fim_middle|>": 102})
    ids = wa.ensure_fim_tokens(tok, FIM_TOKENS)
    assert ids == {"<|fim_prefix|>": 100, "<|fim_suffix|>": 101, "<|fim_middle|>": 102}


def test_ensure_fim_tokens_missing_raises():
    with pytest.raises(ValueError) as exc:
        wa.ensure_fim_tokens(FakeTokenizer({}), FIM_TOKENS)  # ว่าง = ทุกตัวคืน UNK
    assert "<|fim_prefix|>" in str(exc.value)  # ชื่อ token ที่ขาดต้องอยู่ในข้อความ


def test_commit_checkpoint_moves_and_cleans(tmp_path):
    tmp = tmp_path / "checkpoint-100.saving"
    tmp.mkdir()
    (tmp / "trainer_state.json").write_text("{}", encoding="utf-8")
    final = tmp_path / "checkpoint-100"
    wa.commit_checkpoint(tmp, final)
    assert (final / "trainer_state.json").exists()
    assert not tmp.exists()  # ไม่มีโฟลเดอร์ชั่วคราวค้าง


def test_commit_checkpoint_overwrites(tmp_path):
    # final มี checkpoint เดิม → commit ใหม่ต้องแทนที่หมด ไม่มี .saving/.old หลงเหลือ
    old = tmp_path / "checkpoint-100"
    old.mkdir()
    (old / "stale.txt").write_text("old", encoding="utf-8")
    tmp = tmp_path / "checkpoint-100.saving"
    tmp.mkdir()
    (tmp / "trainer_state.json").write_text("{}", encoding="utf-8")
    wa.commit_checkpoint(tmp, old)
    assert (old / "trainer_state.json").exists()
    assert not (old / "stale.txt").exists()  # ไฟล์เก่าถูกแทนที่หมด
    assert not list(tmp_path.glob("*.saving"))
    assert not list(tmp_path.glob("*.old"))


class FakeQueue:
    """จำลอง multiprocessing.Queue — เก็บ message ที่ put ไว้ให้ assert ทีหลัง"""

    def __init__(self):
        self.messages: list[dict] = []

    def put(self, msg: dict) -> None:
        self.messages.append(msg)


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


def test_non_metric_logs_forwarded():
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"train_runtime": 100.0})
    assert len(q.messages) == 1
    msg = q.messages[0]
    assert msg["type"] == "log"
    assert ipc.validate_message(msg) is True
