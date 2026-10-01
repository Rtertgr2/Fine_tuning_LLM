"""Training Subprocess — SFTTrainer + LoRA บน XPU ใน process แยก (สเปก plan.md §4.4)

ส่วนที่ 1 (Task 2): `build_training_args` — config factory ทุกค่ามาจาก
`configs/safe_defaults.py` (ห้าม hardcode ซ้ำ)
ส่วนที่ต่อมาในไฟล์เดียวกัน: guardrails (NaN/FIM/atomic) + StreamToQueueCallback + run_training

รัน: from core.trainer_worker import build_training_args
"""

from __future__ import annotations

import json
import math
import os
import shutil
import traceback
from collections.abc import Iterable
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from configs.safe_defaults import (
    DEFAULT_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH_CAP,
    MAX_SEQ_LENGTH_DEFAULT,
    MAX_STEPS,
    NONFINITE_ABORT_THRESHOLD,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    WARMUP_RATIO,
)
from core.dataset_builder import build_samples, iter_codes
from core.ipc_bridge import error_msg, log_msg, metric_msg, status_msg


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


class NanGuard:
    """นับ loss non-finite ติดต่อกัน — ครบ threshold → คืน True (Task 4 สั่ง abort, สเปก §4.4: 3 ครั้งติด)"""

    def __init__(self, threshold: int = NONFINITE_ABORT_THRESHOLD):
        self.threshold = threshold
        self._streak = 0

    def register(self, loss: float) -> bool:
        if math.isfinite(loss):
            self._streak = 0
            return False
        self._streak += 1
        return self._streak >= self.threshold


def ensure_fim_tokens(tokenizer, fim_tokens: Iterable[str]) -> dict[str, int]:
    """ตรวจว่า FIM tokens อยู่ใน vocab จริง — ขาด/ผิด → ValueError พร้อมชื่อ token

    ผ่านเมื่อ: id ไม่ None, id < len(tokenizer), ไม่ใช่ unk, convert_ids_to_tokens roundtrip ตรง
    ห้ามเรียก method resize ใด ๆ — แก้ที่ token ก่อนเทรน ไม่ใช่ขยาย embedding ระหว่างทาง
    """
    ids: dict[str, int] = {}
    problems: list[str] = []
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in fim_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid >= len(tokenizer):
            problems.append(f"{tok} (id={tid} — อยู่นอก vocab)")
            continue
        if unk is not None and tid == unk:
            problems.append(f"{tok} (คืนค่า unk_token_id — ไม่ได้อยู่ใน vocab)")
            continue
        if tokenizer.convert_ids_to_tokens(tid) != tok:
            problems.append(f"{tok} (roundtrip ไม่ตรง — ถูก split หรือ mapping ผิด)")
            continue
        ids[tok] = int(tid)
    if problems:
        raise ValueError(
            "FIM token ขาด/ผิด: " + ", ".join(problems) + " — ต้องแก้ token ก่อนเริ่มเทรน"
        )
    return ids


def commit_checkpoint(tmp_dir: Path, final_dir: Path) -> None:
    """ย้าย checkpoint จากโฟลเดอร์ชั่วคราว → ที่อยู่จริงแบบ atomic — ไฟล์ไม่มีวันเสียกลางทาง

    ถ้า final มีของเดิมอยู่ → เก็บออกไปเป็น *.old ก่อน แล้ว rename เข้าที่ แล้วล้าง *.old
    (ถ้าโดน abort กลาง save → เหลือแค่ *.saving รอบถัดไปไม่แตะ checkpoint จริง)
    """
    tmp = Path(tmp_dir)
    final = Path(final_dir)
    trash = final.with_name(final.name + ".old")
    if final.exists():
        if trash.exists():
            shutil.rmtree(trash)
        os.rename(final, trash)
    os.rename(tmp, final)
    if trash.exists():
        shutil.rmtree(trash)


class AtomicSaveTrainer(SFTTrainer):
    """SFTTrainer ที่เขียน checkpoint ลงโฟลเดอร์ `.saving` ก่อนค่อย rename เข้าที่ (สเปก §4.4)

    ไม่มี unit test แยก — ถูก exercise โดย commit_checkpoint test + integration run
    """

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        if output_dir is None:
            super()._save(output_dir, state_dict)
            return
        tmp = f"{output_dir}.saving"
        super()._save(tmp, state_dict)
        commit_checkpoint(Path(tmp), Path(output_dir))


class StreamToQueueCallback(TrainerCallback):
    """ส่ง metric/log/status/error จาก trainer ลง Queue ให้ UI อ่าน (สเปก §4.5)

    NaN guard: loss non-finite ติดต่อกันครบ threshold → error + status aborted + หยุดเทรน
    (เมื่อ aborted แล้ว: on_train_begin/on_train_end จะไม่ส่ง status ทับ)
    """

    def __init__(self, queue):
        self.queue = queue
        self.guard = NanGuard()
        self.aborted = False

    def on_log(self, args, state, control, logs, **kwargs):
        if "loss" not in logs:
            self.queue.put(log_msg("INFO", str(logs)))
            return
        loss = logs["loss"]
        self.queue.put(
            metric_msg(
                step=logs.get("step", state.global_step),
                loss=loss,
                lr=logs.get("learning_rate", 0.0),
                epoch=logs.get("epoch", 0.0),
            )
        )
        if self.guard.register(loss):
            self.queue.put(
                error_msg(
                    f"loss non-finite ติดต่อกัน {self.guard.threshold} ครั้ง — abort",
                    "",
                )
            )
            self.queue.put(status_msg("aborted"))
            self.aborted = True
            control.should_training_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.aborted:
            self.queue.put(status_msg("training"))

    def on_save(self, args, state, control, **kwargs):
        self.queue.put(status_msg("saving"))

    def on_train_end(self, args, state, control, **kwargs):
        if not self.aborted:
            self.queue.put(status_msg("finished"))


REQUIRED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "model_id",
        "dataset_id",
        "dataset_column",
        "fim_registry_key",
        "output_dir",
        "max_seq_length",
        "max_steps",
        "code_limit",
        "lora_rank",
    }
)


def validate_config(config: dict) -> None:
    """ตรวจ config ก่อนเทรน — ขาด key → ValueError บอกชื่อที่ขาด; seq length ผิด hard cap → ValueError"""
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"config ขาด key: {', '.join(missing)}")
    seq = config["max_seq_length"]
    if not 0 < seq <= MAX_SEQ_LENGTH_CAP:
        raise ValueError(
            f"max_seq_length={seq} ต้องอยู่ใน (0, {MAX_SEQ_LENGTH_CAP}] (hard cap §5)"
        )


def run_training(config: dict, queue_) -> None:
    """เทรนใน process ปัจจุบัน — ส่ง status/metric/log/error ลง `queue_` ให้ UI (สเปก §4.4/§4.5)

    flow: validate → starting → โหลด registry/tokenizer/dataset/model → LoRA SFT → finished
    ผิดพลาดระหว่างกลาง → error (full traceback) + aborted + raise ต่อ (exit code ของ process ≠ 0)
    """
    validate_config(config)
    queue_.put(status_msg("starting"))
    try:
        registry_path = (
            Path(__file__).resolve().parents[1] / "configs" / "fim_registry.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        fim_tokens = registry[config["fim_registry_key"]]

        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        ensure_fim_tokens(tokenizer, fim_tokens.values())

        codes = iter_codes(
            config["dataset_id"],
            config["dataset_column"],
            limit=config["code_limit"],
        )
        texts = list(
            build_samples(
                codes,
                fim_tokens=fim_tokens,
                eos=tokenizer.eos_token,
                tokenizer=tokenizer,
                max_seq_length=config["max_seq_length"],
            )
        )
        train_dataset = Dataset.from_dict({"text": texts})

        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        lora = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_rank"] * 2,
            lora_dropout=0.0,
            target_modules=list(LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        )

        callback = StreamToQueueCallback(queue_)
        trainer = AtomicSaveTrainer(
            model=model,
            args=build_training_args(
                config["output_dir"],
                max_steps=config["max_steps"],
                save_steps=config.get("save_steps", SAVE_STEPS),
            ),
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=lora,
            callbacks=[callback],
        )
        trainer.train()
        if not callback.aborted:
            queue_.put(status_msg("finished"))
    except Exception as exc:
        queue_.put(error_msg(str(exc), traceback.format_exc()))
        queue_.put(status_msg("aborted"))
        raise
