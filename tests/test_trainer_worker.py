"""Tests สำหรับ core/trainer_worker.py — build_training_args pin hyperparameters จาก safe_defaults (สเปก plan.md §4.4)"""

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
