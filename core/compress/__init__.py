"""Core compression (GGUF / llama.cpp) — standalone pipeline ไม่แตะ Gradio app"""

from __future__ import annotations


class CompressionError(RuntimeError):
    """ข้อความ error เป็น English เสมอ (กติกาเดียวกับ UI/CLI copy)"""
