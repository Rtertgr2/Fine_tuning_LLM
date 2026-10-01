"""Training Subprocess — SFTTrainer + LoRA บน XPU ใน process แยก (สเปก plan.md §4.4)

ส่วนที่ 1 (Task 2): `build_training_args` — config factory ทุกค่ามาจาก
`configs/safe_defaults.py` (ห้าม hardcode ซ้ำ)
ส่วนที่ต่อมาในไฟล์เดียวกัน: guardrails (NaN/FIM/atomic) + StreamToQueueCallback + run_training

รัน: from core.trainer_worker import build_training_args
"""

from __future__ import annotations

from trl import SFTConfig

from configs.safe_defaults import (
    DEFAULT_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH_DEFAULT,
    MAX_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    WARMUP_RATIO,
)


def build_training_args(
    output_dir: str,
    *,
    max_steps: int = MAX_STEPS,
    save_steps: int = SAVE_STEPS,
) -> SFTConfig:
    """สร้าง SFTConfig จาก safe_defaults — ทุกค่าถูก pin ด้วย test แล้ว

    `warmup_steps = round(WARMUP_RATIO × max_steps)`: transformers 5.x ถอด
    `warmup_ratio` ออกแล้ว — คงสัดส่วน 3% ไว้แบบ dynamic (500 steps → 15)
    """
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        save_steps=save_steps,
        save_strategy="steps",
        per_device_train_batch_size=DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=round(WARMUP_RATIO * max_steps),
        seed=SEED,
        save_total_limit=SAVE_TOTAL_LIMIT,
        gradient_checkpointing=True,
        optim="adamw_torch",
        bf16=True,
        logging_steps=1,
        report_to=[],
        max_length=MAX_SEQ_LENGTH_DEFAULT,
        dataset_text_field="text",
        packing=True,
    )
