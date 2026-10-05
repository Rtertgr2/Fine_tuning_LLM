"""report.json ตาม §4 schema + ตารางสรุป terminal + baseline pick (report-only, ห้ามเดา)"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from configs.safe_defaults import DEFAULT_MODEL_ID, EVAL_DIR

# baseline ของ Phase 5 — single source: configs.safe_defaults.EVAL_DIR (root/data_cache/eval)


def build_report(
    *,
    model: str,
    variant: str,
    backend: str,
    weight_bits: int,
    parameter_count: int | None,
    context_tokens: int,
    model_disk_mb: float,
    tokens_per_sec: float | None,
    prompt_tokens_per_sec: float | None = None,
    latency_ms: float | None,
    load_time_ms: float | None,
    peak_vram_mb: float | None,
    kv_cache_mb: float | None,
    exact_match_pct: float | None,
    token_f1: float | None,
    eval_identity: dict | None = None,
) -> dict:
    """รวม metrics → §4 schema dict (วัดไม่ได้ = None — syntax/execution = เสมอ None)

    `eval_identity` = identity ของ evaluation ที่ใช้ (dataset/column/limit/cases/fim) —
    เก็บไว้เทียบกับ baseline ก่อนแสดง delta (Review #2); None เมื่อ `--no-eval`
    """
    return {
        "model": model,
        "variant": variant,
        "optimization": "quantization",
        "backend": backend,
        "weight_bits": weight_bits,
        "context_tokens": context_tokens,
        "parameter_count": parameter_count,
        "model_disk_mb": model_disk_mb,
        "peak_vram_mb": peak_vram_mb,
        "kv_cache_mb": kv_cache_mb,
        "tokens_per_sec": tokens_per_sec,  # gen rate (Review #8: ไม่ใช่ pp — อย่า overstate decode)
        "prompt_tokens_per_sec": prompt_tokens_per_sec,  # pp512 เก็บแยก (None = วัดไม่ได้)
        "latency_ms": latency_ms,
        "load_time_ms": load_time_ms,
        "exact_match_pct": exact_match_pct,
        "token_f1": token_f1,
        "syntax_pass_rate": None,  # ยังไม่มี sandbox phase — ห้ามเดา
        "execution_pass_rate": None,
        "eval": eval_identity,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def write_report(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def pick_baseline(model_id: str, source: Path) -> Path | None:
    """baseline Phase 5 ที่ควรเทียบ — เจอไฟล์จริงเท่านั้น (ไม่ก็ None)"""
    if "exports" in source.parts:
        candidate = EVAL_DIR / "finetuned.json"  # merged LoRA ที่ eval เป็น finetuned
    elif model_id == DEFAULT_MODEL_ID:
        candidate = EVAL_DIR / "base.json"
    else:
        return None
    return candidate if candidate.is_file() else None


def _fmt(value: float | None, decimals: int, base: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.{decimals}f}"
    if base is None:
        return text
    return f"{text} ({value - base:+.{decimals}f})"


def format_table(rows: list[dict], baseline: dict | None = None) -> str:
    """ตาราง `Variant / Size / TPS / EM / F1` — delta ใต้ EM/F1 เมื่อมี baseline"""
    base_em = baseline.get("exact_match_pct") if baseline else None
    base_f1 = baseline.get("token_f1_mean") if baseline else None

    matrix: list[list[str]] = [["Variant", "Size", "TPS", "EM", "F1"]]
    for row in rows:
        matrix.append(
            [
                row["variant"].upper(),
                f"{row['model_disk_mb']:.1f} MB",
                "-" if row["tokens_per_sec"] is None else f"{row['tokens_per_sec']:.1f}",
                _fmt(row["exact_match_pct"], 1, base_em),
                _fmt(row["token_f1"], 3, base_f1),
            ]
        )

    widths = [max(len(line[i]) for line in matrix) for i in range(len(matrix[0]))]
    rendered = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)) for line in matrix]
    rendered.insert(1, "-" * (sum(widths) + 2 * (len(widths) - 1)))
    return "\n".join(rendered)
