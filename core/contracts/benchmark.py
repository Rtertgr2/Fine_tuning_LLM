"""สัญญาที่ 2: benchmark report schema v1.0 — 23 คีย์ flat + validator

`build_report()` (core/compress/report.py) สร้างครบตาม REPORT_KEYS เสมอ,
`write_report()` validate ก่อนเขียนไฟล์ทุกครั้ง — report พัง = ไม่เขียน (gate ที่ขอบระบบ)
null-over-zero เป็นหน้าที่ผู้สร้างรายงาน — validator ไม่ตรวจชนิด/ช่วงค่า metrics
"""

from __future__ import annotations

SCHEMA_VERSION: str = "1.0"

REPORT_KEYS: frozenset[str] = frozenset({
    # --- มีอยู่แล้วใน build_report (core/compress/report.py) ---
    "model", "variant", "optimization", "backend", "weight_bits",
    "context_tokens", "parameter_count", "model_disk_mb",
    "peak_vram_mb", "kv_cache_mb",
    "tokens_per_sec", "prompt_tokens_per_sec", "latency_ms", "load_time_ms",
    "exact_match_pct", "token_f1", "syntax_pass_rate", "execution_pass_rate",
    "eval", "timestamp",
    # --- เพิ่มใหม่ใน v1.0 ---
    "schema_version",   # "1.0" เสมอ
    "peak_rss_mb",      # nullable — วัดแล้วแต่ CLI ยังไม่เขียน (BM-006); null ก่อน
    "environment",      # nullable dict {hardware, os, llama_cpp_commit} — ยังไม่เก็บ (BM-010); null ก่อน
})


def validate_report(data: dict) -> None:
    """key set ต้องตรง REPORT_KEYS เป๊ะ + schema_version ต้องตรง — ไม่ผ่าน = ValueError

    รายงานเก่า (ไม่มี schema_version) จะ fail เสมอ — ตั้งใจ: ทุกไฟล์หลังสัญญาตรึงต้องระบุเวอร์ชันได้
    """
    missing = sorted(REPORT_KEYS - set(data))
    extra = sorted(set(data) - REPORT_KEYS)
    if missing or extra:
        raise ValueError(
            f"report keys mismatch — missing: {missing}, extra: {extra}"
        )
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {data['schema_version']!r} != {SCHEMA_VERSION!r}"
        )
