"""สัญญาที่ 1: โครงสร้าง `run_manifest.json` (schema 1.0) — null-over-zero

เขียนโดย `core/train/export.py` ตอน export สำเร็จเท่านั้น (invariant #3)
import ได้เฉพาะ stdlib + `configs.safe_defaults` — ห้ามดึง subsystem ใดเข้ามา (§2.3)
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

from configs.safe_defaults import (
    DEFAULT_BATCH_SIZE,
    FIM_RATE,
    GRADIENT_ACCUMULATION_STEPS,
    HELDOUT_RATIO,
    HF_HUB_REVISION,
    LEARNING_RATE,
    LORA_TARGET_MODULES,
    SAVE_STEPS,
    SEED,
)

SCHEMA_VERSION: str = "1.0"
ARTIFACT_TYPE: str = "run_manifest"

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoraInfo:
    rank: int
    target_modules: list[str]  # list (ไม่ใช่ tuple) — JSON round-trip เท่ากันเป๊ะ


@dataclass(frozen=True)
class BaseModelInfo:
    model_id: str
    revision: str
    num_params: int | None
    license: str | None


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    column: str
    code_limit: int
    fim_rate: float
    heldout_ratio: float
    train_samples: int | None   # v1: เสมอ null — runner ยังไม่เขียนกลับ (nullable → เติมทีหลังไม่ต้อง bump)
    heldout_samples: int | None


@dataclass(frozen=True)
class TrainingInfo:
    seed: int
    learning_rate: float
    max_steps: int
    save_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    lora: LoraInfo


@dataclass(frozen=True)
class ExportInfo:
    mode: str        # "adapter_only" | "merged"
    path: str        # relative จาก repo root เมื่อทำได้
    dtype: str | None


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    artifact_type: str
    created_at: str
    base_model: BaseModelInfo
    dataset: DatasetInfo
    training: TrainingInfo
    export: ExportInfo
    git_commit: str | None
    tool_versions: dict[str, str | None]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        d = dict(data)
        d["base_model"] = BaseModelInfo(**d["base_model"])
        d["dataset"] = DatasetInfo(**d["dataset"])
        training = dict(d["training"])
        training["lora"] = LoraInfo(**training["lora"])
        d["training"] = TrainingInfo(**training)
        d["export"] = ExportInfo(**d["export"])
        return cls(**d)


def _git_commit() -> str | None:
    """`git rev-parse HEAD` — ดึงตอน build (round-trip ต้องเท่ากัน); ล้ม = None ห้ามเดา"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _tool_versions() -> dict[str, str | None]:
    """version ต่อ package — เจอไม่ได้ = null ต่อตัว (null-over-zero)"""
    versions: dict[str, str | None] = {}
    for pkg in ("torch", "transformers", "peft"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except Exception:
            versions[pkg] = None
    return versions


def _export_path_str(raw: str | Path) -> str:
    """relative จาก repo root เมื่อทำได้ — นอก repo (เช่น tmp_path) เก็บ as-is"""
    try:
        return str(Path(raw).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(raw)


def build_run_manifest(
    config: dict,
    *,
    export_mode: str,
    export_path: str | Path,
    num_params: int | None = None,
    dtype: str | None = None,
    license: str | None = None,
) -> RunManifest:
    """组装 manifest จาก config + ค่าที่ดึงได้จริง (ตาราง spec §3.2) — ดึงไม่ได้ = null

    required keys ใช้ `config["..."]` — ขาด = caller ผิดสัญญา (fail loud)
    `license` ให้ caller (export) เป็นคน fetch แล้วส่งเข้ามา — contracts ห้ามแตะเน็ต
    """
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        artifact_type=ARTIFACT_TYPE,
        created_at=datetime.now(UTC).isoformat(),
        base_model=BaseModelInfo(
            model_id=config["model_id"],
            revision=HF_HUB_REVISION,
            num_params=num_params,
            license=license,
        ),
        dataset=DatasetInfo(
            dataset_id=config["dataset_id"],
            column=config["dataset_column"],
            code_limit=config["code_limit"],
            fim_rate=FIM_RATE,
            heldout_ratio=HELDOUT_RATIO,
            train_samples=None,
            heldout_samples=None,
        ),
        training=TrainingInfo(
            seed=SEED,
            learning_rate=LEARNING_RATE,
            max_steps=config["max_steps"],
            save_steps=config.get("save_steps", SAVE_STEPS),
            batch_size=DEFAULT_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_seq_length=config["max_seq_length"],
            lora=LoraInfo(
                rank=config["lora_rank"],
                target_modules=list(LORA_TARGET_MODULES),
            ),
        ),
        export=ExportInfo(
            mode=export_mode,
            path=_export_path_str(export_path),
            dtype=dtype,
        ),
        git_commit=_git_commit(),
        tool_versions=_tool_versions(),
    )


def write_run_manifest(manifest: RunManifest, dest_dir: str | Path) -> Path:
    """เขียน `dest_dir/run_manifest.json` แบบ atomic — ไฟล์เดียว จึง tmp + os.replace (invariant #6)"""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "run_manifest.json"
    tmp = dest / "run_manifest.json.tmp"
    tmp.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, final)
    return final
