"""Eval worker — spawn target สำหรับ eval (process แยก) — แยกจาก core/trainer_worker.py

progress → log_msg, จบ → EVAL_DONE
"""

from __future__ import annotations

import traceback
from pathlib import Path

from core.infra.ipc_bridge import error_msg, log_msg

EVAL_DONE = "EVAL_DONE"  # marker: eval เสร็จสมบูรณ์ — controller ใช้จบลูป drain (ไม่โชว์ใน UI)


def run_eval_worker(
    config: dict, mode: str, queue_, eval_dir: str | Path | None = None
) -> None:
    """spawn target สำหรับ eval (process แยก) — progress → log_msg, จบ → EVAL_DONE

    ผิด → error_msg (message + traceback) + raise — pattern `predict_middle`
    note: โหลด evaluator ข้างใน (คงพฤติกรรมเดิม — ไม่ดึง torch/transformers ตอน import)
    """
    from core.eval import evaluator as ev

    if eval_dir is None:
        eval_dir = ev.EVAL_DIR
    try:
        ev.run_eval(
            config,
            mode=mode,
            eval_dir=eval_dir,
            progress=lambda done, total: queue_.put(
                log_msg("INFO", f"Evaluating {mode}: {done}/{total}")
            ),
        )
        queue_.put(log_msg("INFO", EVAL_DONE))
    except Exception as exc:
        queue_.put(error_msg(str(exc), traceback.format_exc()))
        raise
