"""Feasibility & VRAM Calculator — ประเมินทรัพยากรล่วงหน้าก่อนเทรน (สเปก plan.md §4.2)

Total Required VRAM = weights + trainable + activations + overhead
- weights    = P × 2 bytes (BF16)
- trainable  = P_lora × (2 grads + 8 optimizer) bytes
- activations = b × s × d × N × 2 bytes (gradient checkpointing เปิด)
- overhead   = ESTIMATOR_OVERHEAD_GB (XPU context + driver buffer)

ที่มาของ P: default hardcode → config.json จาก HF Hub (แคชครั้งแรก) → user กรอก (B)
resolve ไม่ได้และไม่กรอก = blocked (ห้ามเดา)
"""

from __future__ import annotations

import json
import math
from typing import NamedTuple

from huggingface_hub import hf_hub_download

from configs.safe_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LORA_NUM_PARAMS,
    DEFAULT_MODEL_HIDDEN_SIZE,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NUM_LAYERS,
    DEFAULT_MODEL_NUM_PARAMS,
    DISK_MIN_GB,
    ESTIMATOR_FALLBACK_HIDDEN_SIZE,
    ESTIMATOR_FALLBACK_NUM_LAYERS,
    ESTIMATOR_OVERHEAD_GB,
    MAX_SEQ_LENGTH_DEFAULT,
    RAM_MIN_GB,
    VRAM_SAFE_RATIO,
    VRAM_WARNING_RATIO,
)

GB = 1024**3


class ModelSpec(NamedTuple):
    num_params: int
    hidden_size: int
    num_layers: int
    source: str  # "default" | "hf_config" | "user_fallback"


class ModelSpecUnavailable(Exception):
    """โหลด config ไม่ได้และไม่มี P ที่ user กรอก — ห้ามเดา"""


class EstimateResult(NamedTuple):
    verdict: str
    reason: str
    total_required_gb: float
    weights_gb: float
    trainable_gb: float
    activations_gb: float
    overhead_gb: float
    free_vram_gb: float
    spec_source: str


_PASSTHROUGH_REASONS = {
    "no_xpu": "ไม่พบอุปกรณ์ XPU",
    "insufficient_ram": f"RAM ว่างไม่พอ (ต่ำกว่า {RAM_MIN_GB} GB)",
    "insufficient_disk": f"ดิสก์ว่างไม่พอ (ต่ำกว่า {DISK_MIN_GB} GB)",
}

_VERDICT_REASONS = {
    "safe": f"ปลอดภัย — ใช้ ≤ {VRAM_SAFE_RATIO:.0%} ของ VRAM ว่าง",
    "warning": f"ระวัง — ใช้ {VRAM_SAFE_RATIO:.0%}–{VRAM_WARNING_RATIO:.0%} ของ VRAM ว่าง",
    "blocked": f"บล็อก — ต้องการทรัพยากรเกิน {VRAM_WARNING_RATIO:.0%} ของ VRAM ว่าง",
}


def resolve_model_spec(model_id: str, user_params_b: float | None) -> ModelSpec:
    """หา P/d/N ของโมเดล: default hardcode → config.json จาก Hub → user fallback"""
    if model_id == DEFAULT_MODEL_ID:
        return ModelSpec(
            DEFAULT_MODEL_NUM_PARAMS,
            DEFAULT_MODEL_HIDDEN_SIZE,
            DEFAULT_MODEL_NUM_LAYERS,
            "default",
        )
    try:
        path = hf_hub_download(model_id, "config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        d = int(cfg["hidden_size"])
        n = int(cfg["num_hidden_layers"])
        inter = int(cfg["intermediate_size"])
        vocab = int(cfg["vocab_size"])
        # สมมติ MHA (4d²) — กว่าจริงเล็กน้อยสำหรับ GQA = ทิศทางปลอดภัย
        num_params = n * (4 * d * d + 3 * d * inter) + vocab * d
        return ModelSpec(num_params, d, n, "hf_config")
    except Exception:  # เงื่อนไขสเปก "ออฟไลน์ / โหลดไม่ได้" = อะไรก็ตามที่ขวาง
        # P ที่ใช้ไม่ได้ (ไม่กรอก / 0 / ลบ / NaN / inf) = ยังไม่ทราบที่เชื่อถือได้ → ห้ามเดา
        if (
            user_params_b is None
            or not math.isfinite(user_params_b)
            or user_params_b <= 0
        ):
            raise ModelSpecUnavailable(model_id) from None
        return ModelSpec(
            int(user_params_b * 1e9),
            ESTIMATOR_FALLBACK_HIDDEN_SIZE,
            ESTIMATOR_FALLBACK_NUM_LAYERS,
            "user_fallback",
        )


def classify(total_required_gb: float, free_vram_gb: float) -> str:
    """เกณฑ์ตัดสิน: safe ≤ 0.75×free < warning ≤ 0.90×free < blocked"""
    if total_required_gb <= VRAM_SAFE_RATIO * free_vram_gb:
        return "safe"
    if total_required_gb <= VRAM_WARNING_RATIO * free_vram_gb:
        return "warning"
    return "blocked"


def estimate(
    hardware: dict,
    *,
    model_id: str,
    user_params_b: float | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_length: int = MAX_SEQ_LENGTH_DEFAULT,
) -> EstimateResult:
    """คำนวณความต้องการ VRAM แล้วคืน verdict + เหตุผล (รับ dict จาก core.hardware.inspect)"""
    free = hardware.get("free_vram_gb", 0.0)

    # (1) hardware ไม่พร้อม = passthrough ทันที ไม่ต้องคำนวณ
    status = hardware.get("status")
    if status != "ready":
        return EstimateResult(status, _PASSTHROUGH_REASONS.get(status, str(status)),
                              0.0, 0.0, 0.0, 0.0, 0.0, free, "")

    # (2) resolve P ไม่ได้ + ไม่กรอก → blocked (ห้ามเดา)
    try:
        spec = resolve_model_spec(model_id, user_params_b)
    except ModelSpecUnavailable:
        return EstimateResult(
            "blocked",
            "ไม่ทราบจำนวนพารามิเตอร์ — ต้องกรอก P (B) ใน UI (ห้ามเดา)",
            0.0, 0.0, 0.0, 0.0, 0.0, free, "",
        )

    # (3) สูตรคำนวณ (bytes)
    weights_b = spec.num_params * 2
    p_lora = spec.num_params * (DEFAULT_LORA_NUM_PARAMS / DEFAULT_MODEL_NUM_PARAMS)
    trainable_b = p_lora * 10
    activations_b = batch_size * seq_length * spec.hidden_size * spec.num_layers * 2
    overhead_b = ESTIMATOR_OVERHEAD_GB * GB
    total_b = weights_b + trainable_b + activations_b + overhead_b

    total_gb = total_b / GB
    verdict = classify(total_gb, free)
    return EstimateResult(
        verdict,
        _VERDICT_REASONS[verdict],
        total_gb,
        weights_b / GB,
        trainable_b / GB,
        activations_b / GB,
        overhead_b / GB,
        free,
        spec.source,
    )
