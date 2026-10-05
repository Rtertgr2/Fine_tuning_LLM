"""Unit tests สำหรับ core.compress.quantizer — convert/quantize subprocess"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.compress import config, quantizer


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
    quantizer.write_meta(src / "gguf" / "m-fp16.gguf", src)  # sidecar = fresh

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
    quantizer.write_meta(gguf / "m-fp16.gguf", src)  # sidecar = fresh ทั้งคู่
    quantizer.write_meta(gguf / "m-q4_k_m.gguf", src)

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


# --- Review #1 + #9: freshness fingerprint (sidecar) + atomic write ---


def _fake_writing_run(monkeypatch, calls: list) -> None:
    """run_cmd จำลอง tool เขียนไฟล์จริงตาม path ใน cmd (แบบ atomic tmp) แล้วสำเร็จ"""

    def fake_run(cmd):
        calls.append(cmd)
        _outfile_of(cmd).write_bytes(b"GGUF-DATA")
        return "ok"

    monkeypatch.setattr(quantizer, "run_cmd", fake_run)


def _outfile_of(cmd: list[str]) -> Path:
    if "--outfile" in cmd:  # convert: ... --outfile <path> --outtype f16
        return Path(cmd[cmd.index("--outfile") + 1])
    return Path(cmd[2])  # quantize: [tool, fp16, out, TYPE]


def test_reuse_requires_matching_fingerprint(monkeypatch, tmp_path):
    calls: list = []
    _fake_writing_run(monkeypatch, calls)
    src = _src(tmp_path)

    quantizer.build_artifacts(src, ("fp16", "q4_k_m"))
    first = len(calls)
    quantizer.build_artifacts(src, ("fp16", "q4_k_m"))  # source ไม่เปลี่ยน → reuse

    assert len(calls) == first  # ไม่มี subprocess ใหม่ (sidecar ตรง)


def test_rebuild_when_source_changed(monkeypatch, tmp_path):
    calls: list = []
    _fake_writing_run(monkeypatch, calls)
    src = _src(tmp_path)
    quantizer.build_artifacts(src, ("fp16", "q4_k_m"))
    first = len(calls)

    (src / "config.json").write_text('{"changed": true}', encoding="utf-8")  # size ต่าง

    quantizer.build_artifacts(src, ("fp16", "q4_k_m"))

    assert len(calls) > first  # ต้อง rebuild ทั้ง fp16 และ q4 (artifact เก่าใช้ไม่ได้)


def test_rebuild_when_sidecar_missing(monkeypatch, tmp_path):
    calls: list = []
    _fake_writing_run(monkeypatch, calls)
    src = _src(tmp_path)
    quantizer.build_artifacts(src, ("fp16",))
    first = len(calls)

    quantizer.artifact_path(src, "fp16").with_name(
        quantizer.artifact_path(src, "fp16").name + ".srcmeta.json"
    ).unlink()  # legacy artifact ไม่มี sidecar → ถือว่า stale

    quantizer.build_artifacts(src, ("fp16",))

    assert len(calls) > first


def test_convert_failure_leaves_no_partial_or_artifact(monkeypatch, tmp_path):
    def failing_run(cmd):
        _outfile_of(cmd).write_bytes(b"PARTIAL")  # tool เขียนครึ่ง ๆ กลาง ๆ แล้ว fail
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="boom")

    monkeypatch.setattr(quantizer, "run_cmd", failing_run)
    src = _src(tmp_path)

    with pytest.raises(config.CompressionError):
        quantizer.build_artifacts(src, ("fp16",))

    assert not quantizer.artifact_path(src, "fp16").is_file()  # เศษไฟล์ห้ามค้าง
    assert list((src / "gguf").glob("*.partial")) == []


def test_quantize_failure_leaves_no_partial_or_artifact(monkeypatch, tmp_path):
    calls: list = []
    _fake_writing_run(monkeypatch, calls)
    src = _src(tmp_path)
    quantizer.build_artifacts(src, ("fp16",))  # fp16 พร้อม (มี sidecar)

    def failing_run(cmd):
        _outfile_of(cmd).write_bytes(b"PARTIAL")
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="q boom")

    monkeypatch.setattr(quantizer, "run_cmd", failing_run)

    with pytest.raises(config.CompressionError):
        quantizer.build_artifacts(src, ("q8_0",))

    assert not quantizer.artifact_path(src, "q8_0").is_file()
    assert list((src / "gguf").glob("*.partial")) == []
