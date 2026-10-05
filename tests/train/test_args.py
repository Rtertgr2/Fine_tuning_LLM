"""Tests สำหรับ core/train/args.py — build_training_args pin hyperparameters จาก safe_defaults (สเปก plan.md §4.4)"""

from types import SimpleNamespace

import pytest

from core.train import args as wa


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


def test_args_eval_batch_equals_train():
    args = wa.build_training_args("data_cache/x")
    assert args.per_device_eval_batch_size == 1  # P1 A2: default 8 + packing = OOM ตอน eval


def test_args_overrides_for_smoke():
    args = wa.build_training_args("out", max_steps=6, save_steps=2)
    assert args.max_steps == 6
    assert args.save_steps == 2
    assert args.warmup_steps == 0  # round(0.03 × 6) = 0
    # max_seq_length จาก config ต้องถึง SFTConfig.max_length (packing ใช้ค่านี้;
    # ถ้าไม่ส่ง → hardcode 1024 → UI ตั้ง 2048 ใน Phase 4 ถูกเพิกเฉย)
    assert wa.build_training_args("out", max_seq_length=2048).max_length == 2048
    assert wa.build_training_args("out").max_length == 1024  # default = MAX_SEQ_LENGTH_DEFAULT


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
    assert args.eval_steps == 0  # A6: strategy=no ห้ามตั้ง eval_steps ที่ไม่ถูกใช้ (RED เดิม = 50)


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
