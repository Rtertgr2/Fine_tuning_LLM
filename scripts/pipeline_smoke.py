"""Integration smoke ของ training pipeline — รันจริงบน Arc B580 (สเปก plan.md §6 Phase 3)

รัน: ../../.venv/bin/python scripts/pipeline_smoke.py --mode full|abort|all (default all)

- full: เทรน 6 steps จริง → assert status/metric/checkpoint ครบ + ไม่มี *.saving ค้าง
- abort: เริ่มเทรน 500 steps แล้วหยุดกลางทาง → process ตายจริง + XPU VRAM คืน
- ทุก mode: วัด VRAM ก่อน spawn / เปรียบเทียบหลัง process exit สนิท
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue as queue_mod
import shutil
import sys
import time
from pathlib import Path

# รันเป็น `python scripts/pipeline_smoke.py` → sys.path[0] = scripts/ ต้องเพิ่ม root ก่อน
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.safe_defaults import (
    DEFAULT_DATASET_COLUMN,
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    LORA_RANK_DEFAULT,
    MAX_SEQ_LENGTH_DEFAULT,
)
from core.infra.ipc_bridge import abort_process, validate_message
from core.train.runner import run_training

DEADLINE_S = 600.0
VRAM_TOLERANCE_GB = 0.5


def _free_vram_gb() -> float:
    free_bytes, _total = torch.xpu.mem_get_info(torch.xpu.current_device())  # D5
    return free_bytes / 1024**3


def _base_config(output_dir: str, *, max_steps: int, save_steps: int) -> dict:
    return {
        "model_id": DEFAULT_MODEL_ID,
        "dataset_id": DEFAULT_DATASET_ID,
        "dataset_column": DEFAULT_DATASET_COLUMN,
        "fim_registry_key": "qwen",
        "output_dir": output_dir,
        "max_seq_length": MAX_SEQ_LENGTH_DEFAULT,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "code_limit": 64,
        "lora_rank": LORA_RANK_DEFAULT,
    }


def _spawn(config: dict) -> tuple[mp.Process, mp.Queue]:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=run_training, args=(config, q))
    proc.start()
    return proc, q


def _collect(proc: mp.Process, q: mp.Queue, *, stop, deadline_s: float) -> list[dict]:
    """อ่าน queue จนเจอ msg ที่ `stop` คืน True / process ตาย / หมดเวลา — ทุก msg ต้องผ่าน validate"""
    msgs: list[dict] = []
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            msg = q.get(timeout=1.0)
        except queue_mod.Empty:
            if not proc.is_alive():
                break
            continue
        assert validate_message(msg), f"message รูปผิด: {msg!r}"
        msgs.append(msg)
        if stop(msg):
            break
    return msgs


def mode_full() -> None:
    free_before = _free_vram_gb()
    print(f"[full] VRAM free ก่อน spawn: {free_before:.2f} GB")
    out_dir = Path("data_cache/phase3_full")
    shutil.rmtree(out_dir, ignore_errors=True)  # ล้างของเก่า — ไม่งั้น assert ถูก satisfy โดย leftover
    proc, q = _spawn(_base_config(str(out_dir), max_steps=6, save_steps=2))
    msgs = _collect(
        proc,
        q,
        stop=lambda m: m.get("state") in ("finished", "aborted"),
        deadline_s=DEADLINE_S,
    )
    proc.join(120)
    assert proc.exitcode == 0, f"process exitcode={proc.exitcode}"

    states = [m["state"] for m in msgs if m["type"] == "status"]
    for required in ("starting", "training", "finished"):
        assert required in states, f"ขาด status {required}: {states}"
    assert any(m["type"] == "metric" for m in msgs), "ไม่มี metric เลย"
    checkpoints = sorted(out_dir.glob("checkpoint-*"))
    assert checkpoints, f"ไม่มี checkpoint ใน {out_dir}"
    assert not list(out_dir.rglob("*.saving")), "*.saving ค้าง — atomic save ไม่ผ่าน"
    time.sleep(2)  # ให้ driver reclaim VRAM ก่อนวัด (เทียบทุก mode ตาม plan)
    free_after = _free_vram_gb()
    assert free_after >= free_before - VRAM_TOLERANCE_GB, (
        f"VRAM ไม่คืน: ก่อน {free_before:.2f} GB → หลัง {free_after:.2f} GB "
        f"(เกิน tolerance {VRAM_TOLERANCE_GB} GB)"
    )
    print(f"[full] VRAM free หลัง exit: {free_after:.2f} GB (คืนแล้ว)")
    print("[full] OK")


def mode_abort() -> None:
    free_before = _free_vram_gb()
    print(f"[abort] VRAM free ก่อน spawn: {free_before:.2f} GB")
    out_dir = Path("data_cache/phase3_abort")
    shutil.rmtree(out_dir, ignore_errors=True)  # ล้างของเก่า เหมือน mode_full
    proc, q = _spawn(_base_config(str(out_dir), max_steps=500, save_steps=100))
    msgs = _collect(
        proc, q, stop=lambda m: m["type"] == "metric", deadline_s=DEADLINE_S
    )
    assert any(m["type"] == "metric" for m in msgs), "ไม่ถึง metric ตัวแรกก่อน abort"
    assert abort_process(proc) is True, "abort_process ไม่คืน True"
    proc.join()
    assert not proc.is_alive(), "process ยังไม่ตายสนิท"
    time.sleep(2)  # ให้ driver reclaim VRAM ก่อนวัด
    free_after = _free_vram_gb()
    assert free_after >= free_before - VRAM_TOLERANCE_GB, (
        f"VRAM ไม่คืน: ก่อน {free_before:.2f} GB → หลัง {free_after:.2f} GB "
        f"(เกิน tolerance {VRAM_TOLERANCE_GB} GB)"
    )
    print(f"[abort] VRAM free หลัง exit: {free_after:.2f} GB (คืนแล้ว)")
    print("[abort] OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 pipeline smoke")
    parser.add_argument("--mode", choices=("full", "abort", "all"), default="all")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    modes = ("full", "abort") if args.mode == "all" else (args.mode,)
    if "full" in modes:
        mode_full()
    if "abort" in modes:
        mode_abort()
    print("PIPELINE SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
