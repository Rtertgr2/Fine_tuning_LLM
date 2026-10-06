"""Phase A สัญญาที่ 1: RunManifest — round-trip, null-over-zero, safe_defaults, atomic write"""

from __future__ import annotations

import dataclasses
import json

import pytest

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
from core.contracts import manifest as bm


def _config(**over) -> dict:
    """config fixture = REQUIRED_CONFIG_KEYS ครบ (core/train/args.py:113)"""
    base = {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": "data_cache/finetune_run",
        "max_seq_length": 1024,
        "max_steps": 6,
        "code_limit": 64,
        "lora_rank": 8,
    }
    base.update(over)
    return base


def test_manifest_roundtrip_and_version():
    m = bm.build_run_manifest(_config(), export_mode="adapter_only",
                              export_path="exports/run1")
    d = m.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["artifact_type"] == "run_manifest"
    assert dataclasses.asdict(m) == d          # to_dict = asdict เป๊ะ
    assert bm.RunManifest.from_dict(d) == m    # round-trip เท่ากันทุกคีย์ (acceptance #1)
    assert d["training"]["lora"]["target_modules"] == list(LORA_TARGET_MODULES)


def test_null_over_zero_and_safe_default_values(monkeypatch):
    """git ดึงไม่ได้ + license ไม่ส่ง → null ทั้งคู่ (acceptance #2); ค่า constant จาก safe_defaults เท่านั้น"""
    def _boom(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(bm.subprocess, "run", _boom)
    m = bm.build_run_manifest(_config(max_steps=500), export_mode="adapter_only",
                              export_path="exports/run1")
    d = m.to_dict()
    # --- null-over-zero ---
    assert d["git_commit"] is None
    assert d["base_model"]["license"] is None
    assert d["base_model"]["num_params"] is None
    assert d["export"]["dtype"] is None
    assert d["dataset"]["train_samples"] is None      # v1 เสมอ null (spec §3.2)
    assert d["dataset"]["heldout_samples"] is None
    # --- safe_defaults ไม่ hardcode ซ้ำ (invariant #4) ---
    assert d["training"]["seed"] == SEED
    assert d["training"]["learning_rate"] == LEARNING_RATE
    assert d["training"]["batch_size"] == DEFAULT_BATCH_SIZE
    assert d["training"]["gradient_accumulation_steps"] == GRADIENT_ACCUMULATION_STEPS
    assert d["training"]["save_steps"] == SAVE_STEPS          # config ไม่มี key → default
    assert d["training"]["max_steps"] == 500                  # จาก config
    assert d["training"]["max_seq_length"] == 1024
    assert d["training"]["lora"]["rank"] == 8
    assert d["base_model"]["model_id"] == "Qwen/Qwen2.5-Coder-0.5B"
    assert d["base_model"]["revision"] == HF_HUB_REVISION
    assert d["dataset"]["dataset_id"] == "smangrul/hf-stack-v1"
    assert d["dataset"]["column"] == "content"
    assert d["dataset"]["code_limit"] == 64
    assert d["dataset"]["fim_rate"] == FIM_RATE
    assert d["dataset"]["heldout_ratio"] == HELDOUT_RATIO
    assert set(d["tool_versions"]) == {"torch", "transformers", "peft"}


def test_build_manifest_merged_fields():
    """merge mode: num_params/dtype มาจริงจาก caller (acceptance #3 ฝั่ง merge — unit เท่านั้น)"""
    m = bm.build_run_manifest(_config(), export_mode="merged",
                              export_path="exports/run1-merged",
                              num_params=498_431_872, dtype="bfloat16")
    d = m.to_dict()
    assert d["export"] == {"mode": "merged", "path": d["export"]["path"], "dtype": "bfloat16"}
    assert d["base_model"]["num_params"] == 498_431_872


def test_write_run_manifest_atomic_and_parseable(tmp_path):
    m = bm.build_run_manifest(_config(), export_mode="adapter_only",
                              export_path=tmp_path / "dest")
    out = bm.write_run_manifest(m, tmp_path / "dest")
    assert out == tmp_path / "dest" / "run_manifest.json"
    assert json.loads(out.read_text(encoding="utf-8")) == m.to_dict()
    leftovers = [p.name for p in (tmp_path / "dest").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []          # os.replace ทิ้ง tmp ไว้ไม่ได้
