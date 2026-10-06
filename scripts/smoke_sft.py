"""SFTTrainer + LoRA บน XPU — Compat Spike (plan.md §6 Phase 1).

รันด้วย: .venv/bin/python scripts/smoke_sft.py
exit 0 = trl/peft/transformers เข้าคู่กันและเทรนบน XPU ได้จริง 2 steps
"""

from __future__ import annotations

import math
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")  # มีใน tokenizer อยู่แล้ว — ห้าม add_special_tokens
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

SYNTHETIC_CODE = [
    'def add(a: int, b: int) -> int:\n    return a + b\n',
    'def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n',
    'class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n',
    'import json\n\ndef load_config(path: str) -> dict:\n    with open(path) as f:\n        return json.load(f)\n',
    'def dedupe(items: list) -> list:\n    return list(dict.fromkeys(items))\n',
    'for i in range(10):\n    if i % 2 == 0:\n        print(i)\n',
    'def celsius_to_fahrenheit(c: float) -> float:\n    return c * 9 / 5 + 32\n',
    'with open("out.txt", "w") as f:\n    f.write("hello")\n',
]


def check_fim_tokens(tokenizer: AutoTokenizer) -> None:
    """FIM tokens ต้องเป็น token จริงใน vocab (ไม่ split/UNK) และ decode กลับได้"""
    for tok in FIM_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        assert tid is not None and tid < len(tokenizer), f"{tok} not in vocab: {tid}"
        assert tokenizer.convert_ids_to_tokens(tid) == tok, f"{tok} split or UNK"
        if tokenizer.unk_token_id is not None:
            assert tid != tokenizer.unk_token_id, f"{tok} returns UNK"
        assert tokenizer.decode(tid) == tok, f"{tok} decode mismatch"
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in FIM_TOKENS}
    print(f"FIM tokens OK   : {ids}")


def main() -> int:
    # (1) tokenizer + FIM guard
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    check_fim_tokens(tokenizer)

    # (2) model: BF16 + SDPA บน XPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # (3) LoRA r=8, attention + MLP
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    # (4) dataset สังเคราะห์ ~8 ตัวอย่าง
    train_dataset = Dataset.from_dict({"text": SYNTHETIC_CODE})

    # (5) SFTConfig + SFTTrainer ด้วย processing_class= (API รุ่นใหม่ trl >= 0.12)
    args = SFTConfig(
        output_dir="data_cache/smoke_out",
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    # (6) train 2 steps — transformers 5.x ไม่รับ max_steps ใน train() แล้ว
    #     (ใช้ค่าจาก SFTConfig(max_steps=2) แทน)
    trainer.train()
    assert trainer.state.global_step == 2, f"global_step={trainer.state.global_step} != 2"
    device = next(trainer.model.parameters()).device
    print(f"model device    : {device}")
    assert str(device).startswith("xpu"), f"training on {device}, not XPU"

    # (7) loss ต้อง finite
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    assert losses, "no loss in log_history"
    assert all(math.isfinite(x) for x in losses), f"loss not finite: {losses}"
    print(f"losses (2 steps): {losses}")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params: {trainable:,} / {total:,}")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
