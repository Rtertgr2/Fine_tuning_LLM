"""Tests สำหรับ core/trainer_worker.py — build_training_args pin hyperparameters จาก safe_defaults (สเปก plan.md §4.4)"""

from pathlib import Path
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
    assert args.save_strategy == "steps"
    assert args.logging_steps == 1
    assert args.report_to == []
    assert args.logging_nan_inf_filter is False  # P0 A1: filter ปิด → NanGuard เห็น loss จริง


def test_args_overrides_for_smoke():
    args = wa.build_training_args("out", max_steps=6, save_steps=2)
    assert args.max_steps == 6
    assert args.save_steps == 2
    assert args.warmup_steps == 0  # round(0.03 × 6) = 0
    # max_seq_length จาก config ต้องถึง SFTConfig.max_length (packing ใช้ค่านี้;
    # ถ้าไม่ส่ง → hardcode 1024 → UI ตั้ง 2048 ใน Phase 4 ถูกเพิกเฉย)
    assert wa.build_training_args("out", max_seq_length=2048).max_length == 2048
    assert wa.build_training_args("out").max_length == 1024  # default = MAX_SEQ_LENGTH_DEFAULT


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


FULL_CONFIG = {
    "model_id": "Qwen/Qwen2.5-Coder-0.5B",
    "dataset_id": "smangrul/hf-stack-v1",
    "dataset_column": "content",
    "fim_registry_key": "qwen",
    "output_dir": "data_cache/phase3_test",
    "max_seq_length": 1024,
    "max_steps": 6,
    "code_limit": 64,
    "lora_rank": 8,
}


def test_bf16_follows_xpu_runtime_support(monkeypatch):
    # 🟡 เดิม hardcode bf16=True ไม่เคยเช็คว่า autocast รันบน XPU ได้จริง
    monkeypatch.setattr(wa.torch.xpu, "is_bf16_supported", lambda: False)
    assert wa.build_training_args("out").bf16 is False


def test_bf16_fallback_true_when_api_unavailable(monkeypatch):
    # API หาย/raise → คงพฤติกรรมเดิม (True) + warning ไม่ใช่ crash
    def _boom():
        raise RuntimeError("xpu api unavailable")

    monkeypatch.setattr(wa.torch.xpu, "is_bf16_supported", _boom)
    with pytest.warns(UserWarning, match="bf16"):
        args = wa.build_training_args("out")
    assert args.bf16 is True


def test_args_enable_validation_loop():
    # 🟡 เดิมไม่มี eval loop เลย → overfitting มองไม่เห็นจน eval สุดท้าย
    args = wa.build_training_args("out", max_steps=500)
    assert args.eval_strategy == "steps"
    assert args.eval_steps == 50  # max(VAL_EVAL_MIN_STEPS=50, 500//10)
    tiny = wa.build_training_args("out", max_steps=6)
    assert tiny.eval_steps == 50  # ห้าม < 50 — smoke run 6 steps ไม่ต้องชน eval


def test_args_disable_eval_when_no_heldout():
    args = wa.build_training_args("out", has_eval_dataset=False)
    assert args.eval_strategy == "no"  # ไม่มี heldout → ปิด val loop (ห้าม crash)


def test_on_log_eval_loss_sends_val_metric():
    # eval log มาในรูป {"eval_loss": ...} (ไม่มี "loss") → ต้องเป็น val_metric ไม่ใช่ log ดิบ
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"eval_loss": 0.9, "step": 10, "epoch": 0.5})
    assert len(q.messages) == 1
    msg = q.messages[0]
    assert msg == {"type": "val_metric", "step": 10, "epoch": 0.5, "val_loss": 0.9}
    assert ipc.validate_message(msg) is True


def test_validate_config():
    wa.validate_config(dict(FULL_CONFIG))  # ครบ → ไม่ raise
    for key in FULL_CONFIG:
        broken = dict(FULL_CONFIG)
        del broken[key]
        with pytest.raises(ValueError) as exc:
            wa.validate_config(broken)
        assert key in str(exc.value)  # ValueError ต้องบอกชื่อ key ที่ขาด
    # hard cap §5: max_seq_length ต้องอยู่ใน (0, 2048]
    too_long = dict(FULL_CONFIG)
    too_long["max_seq_length"] = 4096
    with pytest.raises(ValueError):
        wa.validate_config(too_long)
    zero = dict(FULL_CONFIG)
    zero["max_seq_length"] = 0
    with pytest.raises(ValueError):
        wa.validate_config(zero)


def test_is_checkpoint_dir_only_matching_names():
    # atomic path เฉพาะ checkpoint-\d+ เท่านั้น — ถ้าชื่ออื่นผ่าน (เช่น run root ที่มี
    # checkpoint ซ้อนอยู่) commit_checkpoint จะ rename+rmtree ทิ้ง checkpoint ทั้งหมด
    assert wa.is_checkpoint_dir("data_cache/run/checkpoint-100") is True
    assert wa.is_checkpoint_dir("checkpoint-1") is True
    assert wa.is_checkpoint_dir("checkpoint-abc") is False
    assert wa.is_checkpoint_dir("checkpoint-100.saving") is False
    assert wa.is_checkpoint_dir("data_cache/phase3_full") is False


def test_on_save_suppressed_after_abort():
    q, cb, state, control = _cb_with_states()
    for _ in range(3):
        cb.on_log(None, state, control, {"loss": float("nan"), "step": 10})
    n = len(q.messages)
    cb.on_save(None, state, control)
    # aborted แล้วห้าม status ถอยหลัง — ไม่งั้นค่าสุดท้ายไม่ใช่ terminal
    # → watchdog คาย "zombie" false alarm ใน Phase 4
    assert len(q.messages) == n


# ---------------------------------------------------------------------------
# Task 3: predict pipeline helpers (latest_checkpoint + build_fim_prompt)
# ---------------------------------------------------------------------------


def test_latest_checkpoint_numeric_order(tmp_path):
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-abc").mkdir()  # ไม่ใช่ checkpoint จริง — ต้องเพิกเฉย
    (tmp_path / "logs").mkdir()
    result = wa.latest_checkpoint(tmp_path)
    assert result == tmp_path / "checkpoint-100"  # numeric ไม่ใช่ lexical ("checkpoint-2" > "checkpoint-100" ถ้าผิด)


def test_latest_checkpoint_none_raises(tmp_path):
    with pytest.raises(ValueError):
        wa.latest_checkpoint(tmp_path)
    (tmp_path / "checkpoint-abc").mkdir()
    with pytest.raises(ValueError):
        wa.latest_checkpoint(tmp_path)  # มีแต่ชื่อปลอม → ยัง raise


def test_build_fim_prompt_exact():
    fim_tokens = {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"}
    prompt = wa.build_fim_prompt("def f():", "return 1", fim_tokens=fim_tokens)
    # PSM แบบไม่มี middle (model generate ต่อจาก middle_tok เอง) — ไม่มี eos เสริม
    assert prompt == "<|fim_prefix|>def f():<|fim_suffix|>return 1<|fim_middle|>"


def test_save_adapter_only_copies(tmp_path):
    out = tmp_path / "run1"
    ckpt = out / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")
    exports = tmp_path / "exports"
    dest = wa.save_adapter_only(out, exports_dir=exports)
    assert dest == exports / "run1"
    assert (dest / "adapter_config.json").exists()
    assert (dest / "adapter_model.safetensors").read_bytes() == b"fake"


def test_save_adapter_only_no_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError):
        wa.save_adapter_only(tmp_path / "empty_missing")


def test_merge_export_importable():
    assert callable(wa.merge_export)


def test_save_adapter_only_config_without_weights_raises(tmp_path):
    """M5: มีแค่ adapter_config.json (น้ำหนักหาย) → ต้อง raise ไม่ใช่รายงานสำเร็จ"""
    ckpt = tmp_path / "run" / "checkpoint-5"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError):
        wa.save_adapter_only(tmp_path / "run")


def test_save_adapter_only_copies_shards(tmp_path):
    """M5: sharded weights ต้องถูก copy ครบ (ไม่ใช่แค่ index)"""
    ckpt = tmp_path / "run" / "checkpoint-7"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model-00001-of-00002.safetensors").write_bytes(b"a")
    (ckpt / "adapter_model-00002-of-00002.safetensors").write_bytes(b"b")
    (ckpt / "adapter_model.safetensors.index.json").write_text("{}")
    dest = wa.save_adapter_only(tmp_path / "run", exports_dir=tmp_path / "exports")
    names = sorted(p.name for p in dest.iterdir())
    assert names == [
        "adapter_config.json",
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
        "adapter_model.safetensors.index.json",
    ]


def test_run_eval_worker_sends_progress_and_done(monkeypatch):
    """spawn target: progress → log_msg ทุก step, จบ → EVAL_DONE (UI ใช้จับจบ)"""
    from core import evaluator as ev_mod

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
    from core import evaluator as ev_mod

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


def test_peak_xpu_memory_zero_when_unavailable(monkeypatch):
    """XPU ไม่มี → คืน 0.0 (CPU-only machine ห้าม crash)"""
    monkeypatch.setattr(wa.torch.xpu, "is_available", lambda: False)
    assert wa.peak_xpu_memory_gb() == 0.0


def test_peak_xpu_memory_invalid_kind():
    with pytest.raises(ValueError, match="reserved"):
        wa.peak_xpu_memory_gb("nope")


def test_peak_xpu_memory_uses_torch(monkeypatch):
    monkeypatch.setattr(wa.torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(
        wa.torch.xpu, "max_memory_reserved", lambda: 2 * 1024**3, raising=False
    )
    assert wa.peak_xpu_memory_gb("reserved") == 2.0
    monkeypatch.setattr(
        wa.torch.xpu, "max_memory_allocated", lambda: 1 * 1024**3, raising=False
    )
    assert wa.peak_xpu_memory_gb("allocated") == 1.0


def test_run_training_filters_heldout_from_iter_codes(monkeypatch):
    """spec Phase 5: `run_training` ต้องกรอง heldout ก่อนสร้างชุดเทรน (mock iter_codes)"""
    from core.dataset_builder import is_heldout

    heldout_code = next(c for c in (f"heldout_{i}" for i in range(5000)) if is_heldout(c))
    train_code = next(c for c in (f"train_{i}" for i in range(5000)) if not is_heldout(c))
    assert heldout_code != train_code

    recorded: list[list[str]] = []

    class _Tok:
        eos_token = "</s>"

        def encode(self, text, add_special_tokens=False):
            # double ต้องพอให้ build_eval_texts (ของจริง) คำนวณ budget ได้ —
            # code เดียวบรรทัดโดน build_samples skip (min_lines) อยู่ดี
            return [0] * max(1, len(text))

    class _Trainer:
        def __init__(self, **_kw):
            pass

        def train(self):
            pass

    monkeypatch.setattr(wa.AutoTokenizer, "from_pretrained", lambda mid: _Tok())
    monkeypatch.setattr(wa, "ensure_fim_tokens", lambda tok, toks: None)
    monkeypatch.setattr(wa, "iter_codes", lambda *a, **k: [heldout_code, train_code])

    def _fake_build_samples(codes, **_kw):
        recorded.append(list(codes))
        return ["dummy text"]

    monkeypatch.setattr(wa, "build_samples", _fake_build_samples)
    # โมเดล fake มีเฉพาะ q_proj → LoraConfig ต้องได้เฉพาะตัวที่มีจริง (ไม่ใช่ทั้ง 7)
    fake_model = SimpleNamespace(
        named_modules=lambda: iter([("model.layers.0.attn.q_proj", None)])
    )
    monkeypatch.setattr(
        wa.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: fake_model
    )
    lora_targets: list = []
    _real_lora_config = wa.LoraConfig

    def _recording_lora_config(**kw):
        lora_targets.append(kw.get("target_modules"))
        return _real_lora_config(**kw)

    monkeypatch.setattr(wa, "LoraConfig", _recording_lora_config)
    monkeypatch.setattr(wa, "AtomicSaveTrainer", _Trainer)

    cfg = {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": "data_cache/x",
        "max_seq_length": 1024,
        "max_steps": 1,
        "code_limit": 100,
        "lora_rank": 8,
    }
    q = FakeQueue()
    wa.run_training(cfg, q)

    # heldout ห้ามหลุดเข้าชุดเทรน — filter_train_codes ถูก apply กับ output ของ iter_codes
    assert recorded == [[train_code]]
    assert q.messages[-1] == {"type": "status", "state": "finished"}
    assert all(m.get("type") != "error" for m in q.messages)
    # wiring: LoraConfig ได้เฉพาะ target ที่มีในโมเดลจริง (q_proj เท่านั้น ไม่ใช่ทั้ง 7)
    assert lora_targets == [["q_proj"]]


# ---------------------------------------------------------------------------
# LoRA targets: filter ตามโมเดลจริง (ข้าม family — design 2026-10-02 picker)
# ---------------------------------------------------------------------------


def test_available_lora_targets_filters_to_model_modules():
    model = SimpleNamespace(
        named_modules=lambda: iter(
            [
                ("model", None),
                ("model.layers.0.attn.q_proj", None),
                ("model.layers.0.attn.o_proj", None),
                ("model.layers.0.mlp.gate_up_proj", None),
            ]
        )
    )
    # ได้เฉพาะตัวที่มีจริง + เรียงตามลำดับ LORA_TARGET_MODULES (gate_up ไม่ตรงชื่อ)
    assert wa.available_lora_targets(model) == ["q_proj", "o_proj"]


def test_available_lora_targets_none_match_gives_english_error():
    model = SimpleNamespace(named_modules=lambda: iter([("model.layers.0.qkv", None)]))
    with pytest.raises(ValueError) as exc:
        wa.available_lora_targets(model)
    assert "LoRA" in str(exc.value)
    assert "not found" in str(exc.value)


# ---------------------------------------------------------------------------
# H1 + M1: validate_config hardening (Fix.md)
# ---------------------------------------------------------------------------


def test_validate_config_rejects_output_outside_sandbox():
    bad = dict(FULL_CONFIG)
    bad["output_dir"] = "/home/someone/important"
    with pytest.raises(ValueError, match="output_dir"):
        wa.validate_config(bad)  # เดิม: ผ่านหมด → commit_checkpoint rmtree ได้ทุกที่


def test_validate_config_rejects_tilde_output_expanding_outside(tmp_path):
    bad = dict(FULL_CONFIG)
    bad["output_dir"] = "~/somewhere"
    with pytest.raises(ValueError, match="output_dir"):
        wa.validate_config(bad)


def test_validate_config_allows_repo_and_temp_output(tmp_path):
    repo_ok = dict(FULL_CONFIG)
    repo_ok["output_dir"] = "data_cache/run"
    wa.validate_config(repo_ok)

    tmp_ok = dict(FULL_CONFIG)
    tmp_ok["output_dir"] = str(tmp_path / "run")
    wa.validate_config(tmp_ok)


def test_validate_config_rejects_nonpositive_bounds():
    for key, val in (("max_steps", 0), ("max_steps", -3), ("lora_rank", -5), ("code_limit", 0)):
        cfg = dict(FULL_CONFIG)
        cfg[key] = val
        with pytest.raises(ValueError, match=key):
            wa.validate_config(cfg)  # M1: เดิม fail ลึกใน SFTConfig/LoraConfig (error ไม่ชัด)

    cfg = dict(FULL_CONFIG)
    cfg["save_steps"] = 0
    with pytest.raises(ValueError, match="save_steps"):
        wa.validate_config(cfg)


# ---------------------------------------------------------------------------
# L4: use_safetensors=True ทุกจุดโหลดโมเดล (Fix.md) — ปฏิเสธ .bin (pickle) เสมอ
# ---------------------------------------------------------------------------


def test_all_auto_model_loads_pin_safetensors():
    """AST จับครบทุก call site (trainer×3 + evaluator×1) — เพิ่มจุดใหม่โดยไม่ใส่ flag = เทสต์ตก"""
    import ast

    root = Path(__file__).resolve().parents[1]
    found = 0
    for rel in ("core/trainer_worker.py", "core/evaluator.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "from_pretrained":
                continue
            if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "AutoModelForCausalLM"):
                continue
            found += 1
            kws = {kw.arg: kw.value for kw in node.keywords}
            flag = kws.get("use_safetensors")
            assert isinstance(flag, ast.Constant) and flag.value is True, (
                f"{rel}:{node.lineno} ต้องตั้ง use_safetensors=True (L4: ไม่รับ pickle .bin)"
            )
    assert found >= 4  # trainer×3 + evaluator×1
