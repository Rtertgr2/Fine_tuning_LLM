"""Training runner — run_training + atomic checkpoint helpers (แยกจาก core/trainer_worker.py)

flow: validate → starting → โหลด registry/tokenizer/dataset/model → LoRA SFT → finished
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.safe_defaults import SAVE_STEPS
from core.data.dataset_builder import (
    build_eval_texts,
    build_samples,
    filter_train_codes,
    iter_codes,
)
from core.data.fim import ensure_fim_tokens
from core.infra.ipc_bridge import error_msg, log_msg, status_msg
from core.train.args import (
    available_lora_targets,
    build_training_args,
    peak_xpu_memory_gb,
    validate_config,
)
from core.train.callbacks import AtomicSaveTrainer, StreamToQueueCallback


def is_checkpoint_dir(path) -> bool:
    """True เมื่อ basename เป็น `checkpoint-<digits>` — atomic path มีเฉพาะชื่อนี้เท่านั้น

    กันกับดักข้อมูลหาย: ถ้า `_save` ถูกเรียกด้วย run root ที่มี checkpoint-* ซ้อนอยู่
    (เช่น `trainer.save_model(run_dir)` ใน Phase 5) `commit_checkpoint` จะ rename+rmtree
    ทิ้ง checkpoint ทั้งหมด — ของชื่ออื่นต้อง save ตรง ๆ ผ่าน super
    """
    name = Path(path).name
    return name.startswith("checkpoint-") and name[len("checkpoint-") :].isdigit()


def _has_adapter_files(ckpt: Path) -> bool:
    return (ckpt / "adapter_config.json").is_file() and bool(
        list(ckpt.glob("adapter_model*.safetensors"))
    )


def _recover_interrupted_commit(root: Path) -> None:
    """Sec-12: กู้ swap ที่ค้างกลางทาง (SIGKILL หลัง final→*.old) — เรียกก่อน scan ใน latest_checkpoint

    - base หาย + มี .old และ .saving → เอา .saving (new — `super()._save` เสร็จแล้วตอน commit เริ่ม)
    - base หาย + มีแค่ .old → คืน .old (checkpoint จริงที่ใช้ได้)
    - base หาย + มีแค่ .saving โดยไม่มี adapter files → ไม่แตะ (อาจเขียนไม่ครบ)
    """
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        name = child.name
        if name.endswith(".old"):
            base = child.with_name(name[: -len(".old")])
            if not is_checkpoint_dir(base) or base.exists():
                continue
            saving = root / f"{base.name}.saving"
            if saving.is_dir() and _has_adapter_files(saving):
                os.rename(saving, base)
                shutil.rmtree(child, ignore_errors=True)
            else:
                os.rename(child, base)
        elif name.endswith(".saving"):
            base = child.with_name(name[: -len(".saving")])
            if not is_checkpoint_dir(base) or base.exists():
                continue
            if _has_adapter_files(child):
                os.rename(child, base)


def commit_checkpoint(tmp_dir: Path, final_dir: Path) -> None:
    """ย้าย checkpoint จากโฟลเดอร์ชั่วคราว → ที่อยู่จริงแบบ atomic — model weights ไม่มีวันเสียกลางทาง

    ถ้า final มีของเดิมอยู่ → เก็บออกไปเป็น *.old ก่อน แล้ว rename เข้าที่ แล้วล้าง *.old
    (ถ้าโดน abort กลาง save → เหลือแค่ *.saving รอบถัดไปไม่แตะ checkpoint จริง)
    หมายเหตุ Sec-12: ถ้าโดน SIGKILL ระหว่าง 2 rename นี้ → resume path คืนผ่าน
    `_recover_interrupted_commit` (เรียกใน latest_checkpoint) — sequence นี้ไม่แก้
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
            Path(__file__).resolve().parents[2] / "configs" / "fim_registry.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        fim_tokens = registry[config["fim_registry_key"]]

        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        ensure_fim_tokens(tokenizer, fim_tokens.values())

        codes = filter_train_codes(
            iter_codes(
                config["dataset_id"],
                config["dataset_column"],
                limit=config["code_limit"],
            )
        )
        # M4: จำนวนก่อนสร้างชุด — ดูว่า heldout + near-dup ทิ้งไปเท่าไหร่ (ห้ามเงียบ)
        queue_.put(log_msg("INFO", f"train codes after filters: {len(codes)}"))
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
        # val loop: heldout (ไม่เคยเห็นตอนเทรน) → eval_dataset — ไม่มี → None (ปิด eval)
        eval_texts = build_eval_texts(
            iter_codes(
                config["dataset_id"],
                config["dataset_column"],
                limit=config["code_limit"],
            ),
            fim_tokens=fim_tokens,
            eos=tokenizer.eos_token,
            tokenizer=tokenizer,
            max_seq_length=config["max_seq_length"],
        )
        eval_dataset = Dataset.from_dict({"text": eval_texts}) if eval_texts else None

        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,  # L4: ปฏิเสธ .bin (pickle) เสมอ
        )
        lora = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_rank"] * 2,
            lora_dropout=0.0,
            target_modules=available_lora_targets(model),
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
                has_eval_dataset=eval_dataset is not None,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=lora,
            callbacks=[callback],
        )
        # M8: resume flag → โหลด checkpoint ล่าสุดต่อ (ไม่มี checkpoint = fresh + log บอกชัด)
        resume: str | None = None
        if config.get("resume"):
            resume = resume_checkpoint(config["output_dir"])
            if resume is None:
                queue_.put(log_msg("INFO", "resume requested but no checkpoint found — starting fresh"))
            else:
                queue_.put(log_msg("INFO", f"resuming from {resume}"))
        trainer.train(resume_from_checkpoint=resume)
        if not callback.aborted:
            # calibration §8.4 — peak VRAM ตอนจบ (test_once: manual gate Task 10 เก็บค่า)
            queue_.put(
                log_msg(
                    "INFO",
                    f"xpu_peak_reserved_gb={peak_xpu_memory_gb('reserved'):.2f} "
                    f"xpu_peak_allocated_gb={peak_xpu_memory_gb('allocated'):.2f}",
                )
            )
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
    _recover_interrupted_commit(root)  # Sec-12: กู้ swap ค้างกลางทางก่อน scan (resume path)
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


def resume_checkpoint(output_dir: str | Path) -> str | None:
    """M8: checkpoint ล่าสุดเป็น str สำหรับ `trainer.train(resume_from_checkpoint=...)` — ไม่มี → None"""
    try:
        return str(latest_checkpoint(output_dir))
    except ValueError:
        return None
