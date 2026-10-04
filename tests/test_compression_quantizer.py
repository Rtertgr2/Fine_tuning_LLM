"""Unit tests สำหรับ core.compression.quantizer — convert/quantize subprocess"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.compression import config, quantizer


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "m"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    return src


def test_build_artifacts_command_order_and_mapping(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd):
        calls.append(cmd)
        return "ok"

    monkeypatch.setattr(quantizer, "run_cmd", fake_run)
    src = _src(tmp_path)

    out = quantizer.build_artifacts(src, ("fp16", "q8_0", "q4_k_m"))

    assert len(calls) == 3  # convert 1 + quantize 2
    assert "--outtype" in calls[0] and "f16" in calls[0]
    assert calls[1][-1] == "Q8_0" and calls[2][-1] == "Q4_K_M"
    assert out[0].name == "m-fp16.gguf"
    assert out[1].name == "m-q8_0.gguf"
    assert out[2].name == "m-q4_k_m.gguf"


def test_build_artifacts_reuses_existing_fp16(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd):
        calls.append(cmd)
        return "ok"

    monkeypatch.setattr(quantizer, "run_cmd", fake_run)
    src = _src(tmp_path)
    (src / "gguf" / "m-fp16.gguf").parent.mkdir(parents=True)
    (src / "gguf" / "m-fp16.gguf").write_bytes(b"x")  # fp16 มีแล้ว → ห้าม convert ซ้ำ

    quantizer.build_artifacts(src, ("fp16", "q8_0", "q4_k_m"))

    assert not any("--outtype" in c for c in calls)  # convert ถูกข้าม
    assert len(calls) == 2  # เหลือ quantize 2 ตัว


def test_convert_failure_raises_english_with_log_tail(monkeypatch, tmp_path):
    def fake_run(cmd):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="boom")

    monkeypatch.setattr(quantizer, "run_cmd", fake_run)
    src = _src(tmp_path)

    with pytest.raises(config.CompressionError) as exc:
        quantizer.build_artifacts(src, ("fp16",))

    msg = str(exc.value)
    assert "convert_hf_to_gguf" in msg
    assert "boom" in msg


def test_unknown_variant_error(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(quantizer, "run_cmd", lambda cmd: calls.append(cmd) or "ok")
    src = _src(tmp_path)

    with pytest.raises(config.CompressionError) as exc:
        quantizer.build_artifacts(src, ("q3",))

    assert "unknown variant" in str(exc.value)
    assert calls == []  # validate ก่อน — ไม่ convert อะไรเลย


def test_source_without_config_json_error(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(config.CompressionError) as exc:
        quantizer.build_artifacts(bare, ("fp16",))
    assert "config.json" in str(exc.value)


def test_quantize_reuses_existing_artifact(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(quantizer, "run_cmd", lambda cmd: calls.append(cmd) or "ok")
    src = _src(tmp_path)
    gguf = src / "gguf"
    gguf.mkdir()
    (gguf / "m-fp16.gguf").write_bytes(b"x")
    (gguf / "m-q4_k_m.gguf").write_bytes(b"x")  # ตัวนี้มีแล้ว

    out = quantizer.build_artifacts(src, ("fp16", "q4_k_m", "q8_0"))

    assert len(calls) == 1  # เหลือ q8_0 ตัวเดียว
    assert out[2].name == "m-q8_0.gguf"


@pytest.mark.integration
def test_build_artifacts_real_qwen_smoke():
    try:
        config.require_tools()
    except config.CompressionError:
        pytest.skip("llama.cpp not built")
    try:
        src = config.resolve_source("Qwen/Qwen2.5-Coder-0.5B")
    except config.CompressionError:
        pytest.skip("Qwen model not downloaded to models/")

    out = quantizer.build_artifacts(src, ("fp16", "q4_k_m"))

    fp16, q4 = out
    assert fp16.is_file() and q4.is_file()
    assert q4.stat().st_size < fp16.stat().st_size
