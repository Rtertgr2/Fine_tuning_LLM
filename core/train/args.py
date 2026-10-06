"""Training args factory + config guards (สเปก plan.md §4.4 — แยกจาก trainer_worker.py เดิม)

ส่วน `build_training_args` — config factory ทุกค่ามาจาก
`configs/safe_defaults.py` (ห้าม hardcode ซ้ำ)

รัน: from core.train.args import build_training_args
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import torch
from trl import SFTConfig

from configs.safe_defaults import (
    DEFAULT_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH_CAP,
    MAX_SEQ_LENGTH_DEFAULT,
    MAX_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    VAL_EVAL_MIN_STEPS,
    WARMUP_RATIO,
)
from core.infra.sandbox import project_roots


def available_lora_targets(model) -> list[str]:
    """LORA_TARGET_MODULES ที่มีอยู่จริงในโมเดล — รองรับ family ใหม่ที่ชื่อเลเยอร์ต่างกัน

    ไม่ตรงสักตัว → ValueError (English) บอกให้ตรวจความเข้ากันได้ของ family
    """
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    found = [target for target in LORA_TARGET_MODULES if target in module_names]
    if not found:
        raise ValueError(
            f"LoRA target modules {list(LORA_TARGET_MODULES)} not found in model — "
            "this model family is incompatible; check its layer names."
        )
    return found


def _xpu_bf16_supported() -> bool:
    """ถาม runtime จริงว่า XPU รัน bf16 ได้ (🟡 เดิม hardcode=True ไม่เคยเช็ค)

    API หาย/raise (torch เปลี่ยน mid-version, driver ไม่พร้อม) → True + warning
    = คงพฤติกรรมเดิม ไม่ใช่ crash
    """
    try:
        return bool(torch.xpu.is_bf16_supported())
    # จับเฉพาะ error ที่คาด (แนว M3): API หาย/RuntimeError ของ driver → fallback
    # bug อื่นที่ไม่คาด → ปล่อย raise ให้ run_training จับ (ห้าม swallow เงียบ)
    except (AttributeError, RuntimeError, NotImplementedError) as exc:
        warnings.warn(
            f"bf16 support check unavailable ({exc}) — assuming supported",
            stacklevel=2,
        )
        return True


def build_training_args(
    output_dir: str,
    *,
    max_steps: int = MAX_STEPS,
    save_steps: int = SAVE_STEPS,
    max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT,
    has_eval_dataset: bool = True,
) -> SFTConfig:
    """สร้าง SFTConfig จาก safe_defaults — ทุกค่าถูก pin ด้วย test แล้ว

    `warmup_steps = round(WARMUP_RATIO × max_steps)`: transformers 5.x ถอด
    `warmup_ratio` ออกแล้ว — คงสัดส่วน 3% ไว้แบบ dynamic (500 steps → 15)
    `max_seq_length` ต้องมาจาก config เพราะ packing ตัด/รวม token ที่ `max_length`
    (hardcode = UI ตั้งค่าใน Phase 4 ถูกเพิกเฉยเงียบ ๆ)
    `has_eval_dataset=False` (ไม่มี heldout) → eval_strategy="no" + eval_steps=0 —
    ห้ามตั้ง steps/ค่าที่ไม่ถูกใช้ แล้วไม่ส่ง eval_dataset (transformers raise)
    """
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        save_steps=save_steps,
        save_strategy="steps",
        eval_strategy="steps" if has_eval_dataset else "no",
        eval_steps=max(VAL_EVAL_MIN_STEPS, max_steps // 10) if has_eval_dataset else 0,
        per_device_train_batch_size=DEFAULT_BATCH_SIZE,
        # P1 A2: eval batch = train — default 8 + packing แถวยาวเต็ม → OOM ตอน eval
        per_device_eval_batch_size=DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=round(WARMUP_RATIO * max_steps),
        seed=SEED,
        save_total_limit=SAVE_TOTAL_LIMIT,
        gradient_checkpointing=True,
        optim="adamw_torch",
        bf16=_xpu_bf16_supported(),
        logging_steps=1,
        # P0 A1: transformers default กรอง non-finite loss ก่อนถึง on_log → NanGuard ไม่เคยทำงาน
        logging_nan_inf_filter=False,
        report_to=[],
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
    )


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


REPO_ROOT = Path(__file__).resolve().parents[2]


def validate_output_dir(raw_output: str | Path) -> Path:
    """H1: output_dir ต้องอยู่ใต้ repo หรือ system temp — ใช้ร่วม validate_config + save_adapter_only
    (commit_checkpoint ทำ rename/rmtree ใน output_dir → กัน data loss นอก sandbox, Sec-10)
    """
    output = Path(raw_output).expanduser().resolve()  # expanduser ก่อน: "~/..." จะไม่หนี sandbox
    # worktree: data_cache เป็น symlink ชี้ main repo → project_roots resolve ให้แล้ว
    allowed = project_roots(REPO_ROOT) + (Path(tempfile.gettempdir()).resolve(),)
    if not any(output.is_relative_to(root) for root in allowed):
        raise ValueError(
            f"output_dir must be under the repo or the system temp dir (got: {raw_output!r})"
        )
    return output


def validate_config(config: dict) -> None:
    """ตรวจ config ก่อนเทรน — ขาด key → ValueError บอกชื่อที่ขาด; seq length ผิด hard cap → ValueError

    H1: output_dir ต้องอยู่ใต้ repo หรือ system temp เท่านั้น — `commit_checkpoint`
    ทำ rename/rmtree ใน output_dir ได้ตรง ๆ (กัน unintended data loss นอก sandbox);
    M1: ค่าเชิงตัวเลขต้อง > 0 — fail ที่นี่ดีกว่า crash ลึกใน SFTConfig/LoraConfig
    """
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"config missing key: {', '.join(missing)}")
    seq = config["max_seq_length"]
    if not 0 < seq <= MAX_SEQ_LENGTH_CAP:
        raise ValueError(
            f"max_seq_length={seq} must be in (0, {MAX_SEQ_LENGTH_CAP}] (hard cap plan §5)"
        )
    raw_output = config["output_dir"]
    validate_output_dir(raw_output)  # H1 — message คงเดิมทุกตัวอักษร (แยกมาเป็น function Sec-10)
    for key in ("max_steps", "lora_rank", "code_limit"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be > 0 (got {config[key]})")
    if "save_steps" in config and config["save_steps"] <= 0:
        raise ValueError(f"save_steps must be > 0 (got {config['save_steps']})")


def peak_xpu_memory_gb(kind: str = "reserved") -> float:
    """คืนค่า peak XPU memory (GB) จาก `torch.xpu.max_memory_<kind>` — ไม่มี XPU → 0.0

    ใช้ตอนจบการเทรน (calibration §8.4) — `kind` รับ "reserved" หรือ "allocated" เท่านั้น
    """
    if kind not in ("reserved", "allocated"):
        raise ValueError(f'kind must be "reserved" or "allocated", got {kind!r}')
    if not torch.xpu.is_available():
        return 0.0
    return getattr(torch.xpu, f"max_memory_{kind}")() / 1024**3
