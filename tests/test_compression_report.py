"""Unit tests สำหรับ core.compression.report — §4 schema, table, baseline pick"""

from __future__ import annotations

import json
from datetime import datetime

from configs.safe_defaults import DEFAULT_MODEL_ID
from core.compression import report


def _kwargs(**over) -> dict:
    base = {
        "model": "Qwen/Qwen2.5-Coder-0.5B",
        "variant": "q4_k_m",
        "backend": "llama.cpp-vulkan",
        "weight_bits": 4,
        "parameter_count": 494_000_000,
        "context_tokens": 2048,
        "model_disk_mb": 379.4,
        "tokens_per_sec": 120.5,
        "latency_ms": 45.2,
        "load_time_ms": 850.0,
        "peak_vram_mb": None,
        "kv_cache_mb": 512.0,
        "exact_match_pct": 2.0,
        "token_f1": 0.338,
    }
    base.update(over)
    return base


def test_build_report_honest_nulls():
    r = report.build_report(**_kwargs())
    assert r["syntax_pass_rate"] is None
    assert r["execution_pass_rate"] is None
    assert r["peak_vram_mb"] is None  # วัดไม่ได้ = None ห้ามเป็น 0


def test_build_report_schema_keys():
    r = report.build_report(**_kwargs())
    expected = {
        "model",
        "variant",
        "optimization",
        "backend",
        "weight_bits",
        "context_tokens",
        "parameter_count",
        "model_disk_mb",
        "peak_vram_mb",
        "kv_cache_mb",
        "tokens_per_sec",
        "latency_ms",
        "load_time_ms",
        "exact_match_pct",
        "token_f1",
        "syntax_pass_rate",
        "execution_pass_rate",
        "timestamp",
    }
    assert expected <= set(r)
    assert r["optimization"] == "quantization"
    datetime.fromisoformat(r["timestamp"])  # UTC ISO parse ได้


def test_write_report_roundtrip(tmp_path):
    r = report.build_report(**_kwargs())
    out = report.write_report(r, tmp_path / "sub" / "q4_k_m.json")
    assert out == tmp_path / "sub" / "q4_k_m.json"
    assert json.loads(out.read_text(encoding="utf-8")) == r


def test_pick_baseline_exports(monkeypatch, tmp_path):
    ev = tmp_path / "eval"
    ev.mkdir()
    (ev / "finetuned.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(report, "EVAL_DIR", ev)
    src = tmp_path / "exports" / "x-merged"
    src.mkdir(parents=True)
    assert report.pick_baseline("whatever/model", src) == ev / "finetuned.json"


def test_pick_baseline_default_model(monkeypatch, tmp_path):
    ev = tmp_path / "eval"
    ev.mkdir()
    (ev / "base.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(report, "EVAL_DIR", ev)
    src = tmp_path / "somewhere"
    src.mkdir()
    assert report.pick_baseline(DEFAULT_MODEL_ID, src) == ev / "base.json"


def test_pick_baseline_unknown_returns_none(monkeypatch, tmp_path):
    ev = tmp_path / "eval"
    ev.mkdir()
    (ev / "base.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(report, "EVAL_DIR", ev)
    other = tmp_path / "somewhere"
    other.mkdir()

    # model_id ไม่ใช่ default และ source ไม่อยู่ใต้ exports → None
    assert report.pick_baseline("acme/other", other) is None
    # default model แต่ไฟล์ baseline ไม่มี → None
    (ev / "base.json").unlink()
    assert report.pick_baseline(DEFAULT_MODEL_ID, other) is None


def test_format_table_lists_variants_and_delta():
    rows = [
        report.build_report(**_kwargs(variant="fp16", weight_bits=16, model_disk_mb=948.1, exact_match_pct=1.0, token_f1=0.248)),
        report.build_report(**_kwargs(variant="q4_k_m", weight_bits=4, model_disk_mb=379.4, exact_match_pct=2.0, token_f1=0.338)),
    ]
    baseline = {"exact_match_pct": 10.0, "token_f1_mean": 0.5}
    table = report.format_table(rows, baseline)
    assert "Q4_K_M" in table
    assert "FP16" in table
    assert "-8.0" in table  # delta EM ของ q4 (2.0 - 10.0) — แสดงเครื่องหมายลบจริง
    assert "379.4 MB" in table


def test_format_table_without_baseline_has_no_delta():
    rows = [report.build_report(**_kwargs(exact_match_pct=2.0, token_f1=0.338))]
    table = report.format_table(rows, None)
    assert "2.0" in table
    assert "(" not in table  # ไม่มี delta cell
