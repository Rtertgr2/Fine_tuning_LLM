"""Unit tests สำหรับ core.compress.report — §4 schema, table, baseline pick"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from configs.safe_defaults import DEFAULT_MODEL_ID
from core.contracts.benchmark import REPORT_KEYS, SCHEMA_VERSION
from core.compress import report


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
    assert set(r) == REPORT_KEYS      # ล็อกเป๊ะ 23 คีย์ (แทน subset เดิม = แข็งขึ้น ไม่ใช่ลดเกณฑ์)
    assert r["schema_version"] == SCHEMA_VERSION == "1.0"
    assert r["peak_rss_mb"] is None   # BM-006 ยังไม่เขียน — null ก่อน (null-over-zero)
    assert r["environment"] is None   # BM-010 ยังไม่เก็บ
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


def test_write_report_rejects_invalid_and_stale_schema(tmp_path):
    """gate ที่ขอบระบบ — report พัง = ไม่เขียนไฟล์เลย"""
    bad = report.build_report(**_kwargs())
    bad.pop("kv_cache_mb")
    with pytest.raises(ValueError, match="kv_cache_mb"):
        report.write_report(bad, tmp_path / "a.json")
    assert not (tmp_path / "a.json").exists()

    stale = report.build_report(**_kwargs())
    stale["schema_version"] = "0.9"
    with pytest.raises(ValueError, match="schema_version"):
        report.write_report(stale, tmp_path / "b.json")
    assert not (tmp_path / "b.json").exists()
