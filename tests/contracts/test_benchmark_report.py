"""Phase A สัญญาที่ 2: Benchmark Report Schema v1.0 — 23 คีย์ flat + validator"""

from __future__ import annotations

import pytest

from core.contracts import benchmark as cb


def _full_report() -> dict:
    return {k: None for k in cb.REPORT_KEYS} | {"schema_version": cb.SCHEMA_VERSION}


def test_report_keys_pinned():
    """23 คีย์ flat — แก้ = แก้ test+spec (schema v2 ค่อยมาจับกลุ่ม Sprint 3)"""
    assert len(cb.REPORT_KEYS) == 23
    assert {"schema_version", "peak_rss_mb", "environment"} <= cb.REPORT_KEYS
    assert cb.SCHEMA_VERSION == "1.0"


def test_validate_accepts_full_report():
    cb.validate_report(_full_report())          # ไม่ raise


def test_validate_rejects_missing_key():
    data = _full_report()
    data.pop("token_f1")
    with pytest.raises(ValueError, match="token_f1"):
        cb.validate_report(data)


def test_validate_rejects_extra_key():
    data = _full_report()
    data["legacy_field"] = 1
    with pytest.raises(ValueError, match="legacy_field"):
        cb.validate_report(data)


def test_validate_rejects_stale_schema_version():
    data = _full_report()
    data["schema_version"] = "0.9"              # report เก่าไหลเข้าระบบใหม่ = ต้องบล็อก
    with pytest.raises(ValueError, match="schema_version"):
        cb.validate_report(data)
