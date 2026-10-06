"""สัญญาที่ 3: LocalRuntime — interface สำหรับเสิร์ฟ GGUF แบบ backend-agnostic

วันนี้ backend เดียวคือ llama-server binary (core/compress/llama_runner.py ยังเป็น
implementation ที่ใช้จริงต่อไปจน Phase D เขียน adapter ครอบมัน, Phase E เพิ่ม
llama-cpp-python) — สัญญาห้ามผูกกับ backend: ห้าม import llama_runner/llama_cpp ที่นี่
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class LocalRuntime(ABC):
    """สัญญา runtime สำหรับเสิร์ฟ GGUF — backend-agnostic

    ทุก method ห้าม raise ข้อความกำกวม: start ล้ม = ต้อง raise พร้อมเหตุผล;
    is_ready = poll จน timeout แล้วคืน bool
    """

    @abstractmethod
    def start(self, model_path: Path, *, device: str = "vulkan",
              context_tokens: int | None = None) -> None:
        """โหลด GGUF แล้วเปิด endpoint — context_tokens=None = ปล่อยให้ backend ใช้ค่า native
        (ณ วันนี้ serving ไม่ pin context — RM-008; เมื่อไหร่ pin ค่า ให้ส่งจาก caller)"""

    @abstractmethod
    def is_ready(self, timeout_s: float) -> bool:
        """poll health จน ready หรือหมดเวลา — ต้องรวม identity check (โมเดลที่เปิด = โมเดลที่ขอ)"""

    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int,
                 temperature: float | None = None, stop: list[str] | None = None) -> str:
        """FIM completion ดิบ — คง raw prompt semantics (BM-008) ห้าม strip/insert เงียบ ๆ
        max_tokens maps ไป n_predict ของ backend เอง (หน้าที่ adapter)"""

    @abstractmethod
    def stop(self) -> None:
        """graceful stop แล้ว escalate kill ตาม §4.2 (SIGINT → รอ grace → SIGKILL) — เรียกซ้ำได้ (idempotent)"""
