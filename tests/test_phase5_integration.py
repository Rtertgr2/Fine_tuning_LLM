"""Integration tests Phase 5 — Abort กัน orphan/VRAM leak, Disk gate, Cache growth (spec §5, §8)

ทั้งไฟล์เป็น `@pytest.mark.integration` — ค่าเริ่มต้น (`addopts`) ไม่รัน,
รันแยกด้วย: `.venv/bin/python -m pytest tests/ -m integration -v`
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from configs.safe_defaults import (
    DEFAULT_DATASET_COLUMN,
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    LORA_RANK_DEFAULT,
    MAX_SEQ_LENGTH_DEFAULT,
    MAX_STEPS,
    TRAIN_CODE_LIMIT,
)
from core import evaluator as ev
from core import hardware as hw
from ui.controller import TrainingController

pytestmark = pytest.mark.integration

GB = 1024**3
_ITEST_OUTPUT = Path("data_cache/itest_abort")


def _valid_config(**overrides) -> dict:
    cfg = {
        "model_id": DEFAULT_MODEL_ID,
        "dataset_id": DEFAULT_DATASET_ID,
        "dataset_column": DEFAULT_DATASET_COLUMN,
        "fim_registry_key": "qwen",
        "output_dir": str(_ITEST_OUTPUT),
        "max_seq_length": MAX_SEQ_LENGTH_DEFAULT,
        "max_steps": MAX_STEPS,
        "code_limit": TRAIN_CODE_LIMIT,
        "lora_rank": LORA_RANK_DEFAULT,
    }
    cfg.update(overrides)
    return cfg


def test_abort_mid_training_no_orphan_no_vram_leak():
    """abort → child ตายจริง, ไม่มี orphan, VRAM คืน (§5 Orphan Process + RAM/VRAM)"""
    shutil.rmtree(_ITEST_OUTPUT, ignore_errors=True)
    ctl = TrainingController()
    vram_before = hw.inspect("data_cache")["free_vram_gb"]
    assert hw.inspect("data_cache")["status"] == "ready", "ต้องมี XPU ก่อนรัน test นี้"

    try:
        cfg = _valid_config(max_steps=60)
        assert ctl.start(cfg) == "started"
        child_pid = ctl._process.pid  # type: ignore[union-attr]

        # รอ metric ตัวแรก (มี metric = model โหลด + เทรนเริ่มแล้ว) — timeout 180s
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            snap = ctl.tick()
            if snap.metrics:
                break
            time.sleep(0.5)
        else:
            pytest.fail("no training metrics within 180s")

        assert ctl.abort() is True, "abort ต้อง kill child สำเร็จ"
        ctl.exit()

        # child ไม่ alive + ไม่มี orphan (PID หายจากรายชื่อ children)
        assert ctl._process is not None and ctl._process.is_alive() is False  # type: ignore[union-attr]
        child_pids = {p.pid for p in psutil.Process().children(recursive=True)}
        assert child_pid not in child_pids

        # VRAM คืนมา (รอ context release สั้น ๆ แล้ววัด)
        for _ in range(10):
            time.sleep(1.0)
            vram_after = hw.inspect("data_cache")["free_vram_gb"]
            if vram_after >= vram_before - 0.5:
                break
        assert vram_after >= vram_before - 0.5, (
            f"VRAM leak: before={vram_before:.2f}GB after={vram_after:.2f}GB"
        )
    finally:
        ctl.exit()
        shutil.rmtree(_ITEST_OUTPUT, ignore_errors=True)


def test_disk_gate_blocks_preflight(monkeypatch):
    """disk < DISK_MIN_GB → insufficient_disk → ปุ่ม Start ถูกบล็อก (spec §5 Review Focus)"""
    monkeypatch.setattr(hw.torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(hw.torch.xpu, "get_device_name", lambda i: "Intel Arc B580")
    monkeypatch.setattr(hw.torch.xpu, "mem_get_info", lambda i: (11 * GB, 12 * GB))
    monkeypatch.setattr(
        hw.psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * GB)
    )
    monkeypatch.setattr(
        hw.psutil, "disk_usage", lambda p: SimpleNamespace(free=19 * GB)  # < DISK_MIN_GB
    )

    assert hw.inspect("data_cache")["status"] == "insufficient_disk"

    from ui.dashboard import _START_VERDICTS

    ctl = TrainingController()
    result = ctl.preflight(_valid_config(), None)
    assert result.verdict == "insufficient_disk"
    assert "insufficient_disk" not in _START_VERDICTS  # start ถูกบล็อกแน่นอน


def test_datacache_growth_bounded(capsys):
    """stream eval cases 512 ตัวอย่าง → cache ห้ามโตเกิน 5GB (informational)"""
    from transformers import AutoTokenizer

    def dir_gb(path: str) -> float:
        total = 0
        for p in Path(path).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / GB

    before = dir_gb("data_cache")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    cases = ev.build_eval_cases(
        dataset_id=DEFAULT_DATASET_ID,
        dataset_column=DEFAULT_DATASET_COLUMN,
        limit=512,
        n_cases=10,
        tokenizer=tokenizer,
    )
    after = dir_gb("data_cache")
    delta = after - before
    with capsys.disabled():
        print(f"\ndata_cache growth: {delta:.3f} GB ({len(cases)} eval cases built)")
    assert delta < 5.0, f"data_cache grew {delta:.3f} GB"
