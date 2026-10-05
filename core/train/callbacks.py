"""Guardrails + UI queue callback (สเปก plan.md §4.4/§4.5 — แยกจาก trainer_worker.py เดิม)

NanGuard (NaN streak) + AtomicSaveTrainer (checkpoint atomic) + StreamToQueueCallback
"""

from __future__ import annotations

import math
from pathlib import Path

from transformers import TrainerCallback
from trl import SFTTrainer

from configs.safe_defaults import NONFINITE_ABORT_THRESHOLD
from core.infra.ipc_bridge import error_msg, log_msg, metric_msg, status_msg, val_metric_msg


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


class AtomicSaveTrainer(SFTTrainer):
    """SFTTrainer ที่เขียน checkpoint ลงโฟลเดอร์ `.saving` ก่อนค่อย rename เข้าที่ (สเปก §4.4)

    atomic เฉพาะ output_dir ที่ชื่อเป็น `checkpoint-<digits>` — ที่อยู่อื่น (เช่น run root)
    ให้ super เขียนตรง ๆ เพื่อไม่ให้ checkpoint ที่ซ้อนอยู่ถูกล้าง (ดู `is_checkpoint_dir`)
    """

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        # function-level: runner ใช้ AtomicSaveTrainer ที่ top-level → หลบ circular import
        from core.train.runner import commit_checkpoint, is_checkpoint_dir

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
        self.last_lr: float | None = None  # P1 A3: ค่า lr ล่าสุดที่เห็นจริง — ห้ามเดาเป็น 0.0

    def on_log(self, args, state, control, logs, **kwargs):
        if "eval_loss" in logs:
            # val loop: eval log (ไม่มี "loss") → val_metric (schema แยก — validate เข้ม)
            self.queue.put(
                val_metric_msg(
                    step=logs.get("step", state.global_step),
                    epoch=logs.get("epoch", 0.0),
                    val_loss=logs["eval_loss"],
                )
            )
            return control
        if "loss" not in logs:
            self.queue.put(log_msg("INFO", str(logs)))
            return control
        loss = logs["loss"]
        if "learning_rate" in logs:
            self.last_lr = logs["learning_rate"]
        if self.last_lr is None:
            # P1 A3: ยังไม่เคยเห็น lr เลย → WARNING + raw logs แทน metric ปลอม (กราฟห้ามโกหก)
            self.queue.put(log_msg("WARNING", f"loss log without learning_rate — raw: {logs}"))
        else:
            self.queue.put(
                metric_msg(
                    step=logs.get("step", state.global_step),
                    loss=loss,
                    lr=self.last_lr,
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
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.aborted:
            self.queue.put(status_msg("training"))

    def on_save(self, args, state, control, **kwargs):
        if not self.aborted:
            self.queue.put(status_msg("saving"))

    def on_train_end(self, args, state, control, **kwargs):
        if not self.aborted:
            self.queue.put(status_msg("finished"))
