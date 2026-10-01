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
    max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT,
) -> SFTConfig:
    """สร้าง SFTConfig จาก safe_defaults — ทุกค่าถูก pin ด้วย test แล้ว

    `warmup_steps = round(WARMUP_RATIO × max_steps)`: transformers 5.x ถอด
    `warmup_ratio` ออกแล้ว — คงสัดส่วน 3% ไว้แบบ dynamic (500 steps → 15)
    `max_seq_length` ต้องมาจาก config เพราะ packing ตัด/รวม token ที่ `max_length`
    (hardcode = UI ตั้งค่าใน Phase 4 ถูกเพิกเฉยเงียบ ๆ)
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
        max_length=max_seq_length,
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
            problems.append(f"{tok} (id={tid} — outside vocab)")
            continue
        if unk is not None and tid == unk:
            problems.append(f"{tok} (maps to unk_token_id — not in vocab)")
            continue
        if tokenizer.convert_ids_to_tokens(tid) != tok:
            problems.append(f"{tok} (roundtrip mismatch — split or mapped wrongly)")
            continue
        ids[tok] = int(tid)
    if problems:
        raise ValueError(
            "FIM token missing/invalid: " + ", ".join(problems) + " — fix tokens before training"
        )
    return ids


def is_checkpoint_dir(path) -> bool:
    """True เมื่อ basename เป็น `checkpoint-<digits>` — atomic path มีเฉพาะชื่อนี้เท่านั้น

    กันกับดักข้อมูลหาย: ถ้า `_save` ถูกเรียกด้วย run root ที่มี checkpoint-* ซ้อนอยู่
    (เช่น `trainer.save_model(run_dir)` ใน Phase 5) `commit_checkpoint` จะ rename+rmtree
    ทิ้ง checkpoint ทั้งหมด — ของชื่ออื่นต้อง save ตรง ๆ ผ่าน super
    """
    name = Path(path).name
    return name.startswith("checkpoint-") and name[len("checkpoint-") :].isdigit()


def commit_checkpoint(tmp_dir: Path, final_dir: Path) -> None:
    """ย้าย checkpoint จากโฟลเดอร์ชั่วคราว → ที่อยู่จริงแบบ atomic — model weights ไม่มีวันเสียกลางทาง

    ถ้า final มีของเดิมอยู่ → เก็บออกไปเป็น *.old ก่อน แล้ว rename เข้าที่ แล้วล้าง *.old
    (ถ้าโดน abort กลาง save → เหลือแค่ *.saving รอบถัดไปไม่แตะ checkpoint จริง)
    หมายเหตุ (ตรงความจริงตาม source transformers): atomic ครอบเฉพาะ weights ที่เขียนผ่าน
    `_save` — optimizer/scheduler/trainer_state.json เขียนลง checkpoint dir ตรง ๆ ทีหลัง
    → adapter weights ปลอดภัยเสมอ แต่ resume อาจไม่ครบถ้าโดน kill กลางเขียน checkpoint
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

    atomic เฉพาะ output_dir ที่ชื่อเป็น `checkpoint-<digits>` — ที่อยู่อื่น (เช่น run root)
    ให้ super เขียนตรง ๆ เพื่อไม่ให้ checkpoint ที่ซ้อนอยู่ถูกล้าง (ดู `is_checkpoint_dir`)
    """

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        if output_dir is None or not is_checkpoint_dir(output_dir):
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
                    f"loss non-finite for {self.guard.threshold} consecutive steps — aborting",
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
        if not self.aborted:
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
        raise ValueError(f"config missing key: {', '.join(missing)}")
    seq = config["max_seq_length"]
    if not 0 < seq <= MAX_SEQ_LENGTH_CAP:
        raise ValueError(
            f"max_seq_length={seq} must be in (0, {MAX_SEQ_LENGTH_CAP}] (hard cap plan §5)"
        )


def run_training(config: dict, queue_) -> None:
    """เทรนใน process ปัจจุบัน — ส่ง status/metric/log/error ลง `queue_` ให้ UI (สเปก §4.4/§4.5)

    flow: validate → starting → โหลด registry/tokenizer/dataset/model → LoRA SFT → finished
    ผิดพลาด (รวม config ผิด) → error (full traceback) + aborted + raise ต่อ (exit code ≠ 0)
    note: "finished" อาจถูกส่ง 2 ครั้ง (on_train_end ของ callback + ท้าย flow — ตรงสเปก Task 4/5
    ทั้งคู่) → Phase 4 ต้องทนรับ status ซ้ำได้
    """
    try:
        validate_config(config)
        queue_.put(status_msg("starting"))
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
                max_seq_length=config["max_seq_length"],
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


# --------------------------------------------------------------------------- #
# Phase 4: Playground + Export workers (spawn targets ใน process แยก — §5 VRAM)
# --------------------------------------------------------------------------- #


def latest_checkpoint(output_dir: str | Path) -> Path:
    """หา `checkpoint-<n>` ที่เลขมากสุดใน output_dir (numeric เทียบ ไม่ใช่ lexical)

    ไม่มี checkpoint ที่ถูกต้อง → raise ValueError (ผู้เรียกตัดสินใจเอง)
    """
    root = Path(output_dir)
    best: tuple[int, Path] | None = None
    if root.is_dir():
        for child in root.iterdir():
            if not (child.is_dir() and is_checkpoint_dir(child.name)):
                continue
            step_text = child.name.removeprefix("checkpoint-")
            if not step_text.isdigit():
                continue
            step = int(step_text)
            if best is None or step > best[0]:
                best = (step, child)
    if best is None:
        raise ValueError(f"no checkpoint found in {root}")
    return best[1]


def build_fim_prompt(prefix: str, suffix: str, *, fim_tokens: dict) -> str:
    """PSM prompt (ไม่มี middle) — model generate ต่อจาก middle_tok เอง"""
    return (
        f"{fim_tokens['prefix']}{prefix}"
        f"{fim_tokens['suffix']}{suffix}"
        f"{fim_tokens['middle']}"
    )


def predict_middle(config: dict, prefix: str, suffix: str, queue_) -> None:
    """Playground inference: โหลด base + LoRA จาก checkpoint ล่าสุด → FIM predict middle

    spawn target (process แยก) — ส่งผลลัพธ์เป็น log_msg ลง queue_, ผิด → error_msg + raise
    """
    try:
        validate_config(config)
        registry_path = Path(__file__).resolve().parents[1] / "configs" / "fim_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        fim_tokens = registry[config["fim_registry_key"]]

        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        ensure_fim_tokens(tokenizer, fim_tokens.values())

        from peft import PeftModel

        adapter_dir = latest_checkpoint(config["output_dir"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        device = "xpu" if torch.xpu.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        prompt = build_fim_prompt(prefix, suffix, fim_tokens=fim_tokens)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        continuation = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        queue_.put(log_msg("INFO", text))
    except Exception as exc:
        queue_.put(error_msg(str(exc), traceback.format_exc()))
        raise


def save_adapter_only(
    output_dir: str | Path, *, exports_dir: str | Path = "exports"
) -> Path:
    """คัดลอกเฉพาะไฟล์ LoRA จาก checkpoint ล่าสุด → exports/<output_dir.name>/ (file op ไม่กิน VRAM)"""
    ckpt = latest_checkpoint(output_dir)
    config_src = ckpt / "adapter_config.json"
    # sharded weights (adapter_model-00001-of-00002.safetensors) + index ต้องโดนด้วย
    weight_srcs = sorted(ckpt.glob("adapter_model*.safetensors"))
    index_srcs = sorted(ckpt.glob("adapter_model*.safetensors.index.json"))
    if not config_src.exists() or (not weight_srcs and not index_srcs):
        raise ValueError(f"checkpoint {ckpt} contains no adapter files")
    dest = Path(exports_dir) / Path(output_dir).name
    dest.mkdir(parents=True, exist_ok=True)
    for src in (config_src, *weight_srcs, *index_srcs):
        shutil.copy2(src, dest / src.name)
    return dest


def merge_export(config: dict) -> Path:
    """Merge LoRA เข้า base แล้วบันทึก full weights → exports/<name>-merged/ (spawn target, กิน RAM ~2×)"""
    validate_config(config)
    registry_path = Path(__file__).resolve().parents[1] / "configs" / "fim_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    fim_tokens = registry[config["fim_registry_key"]]

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    ensure_fim_tokens(tokenizer, fim_tokens.values())

    from peft import PeftModel

    adapter_dir = latest_checkpoint(config["output_dir"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()

    dest = Path("exports") / f"{Path(config['output_dir']).name}-merged"
    dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    return dest
