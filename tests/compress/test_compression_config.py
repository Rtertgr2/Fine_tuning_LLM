"""Unit tests สำหรับ core.compress.config — paths/variants/device/resolve_source"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.infra import estimator
from core.compress import config


def test_tools_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert config.tools_dir() == tmp_path


def test_tools_dir_default(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    default = config.tools_dir()
    # Review #5: default ต้องอิง repo root (ตรง setup script) ไม่ใช่ CWD
    assert default.is_absolute()
    assert default == Path(config.__file__).resolve().parents[2] / "tools" / "llama.cpp"


def test_require_tools_missing_raises_english(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "tools_dir", lambda: tmp_path)  # ว่างเปล่า
    with pytest.raises(config.CompressionError) as exc:
        config.require_tools()
    assert "setup_llamacpp.sh" in str(exc.value)


def test_resolve_source_hub_id_error():
    with pytest.raises(config.CompressionError) as exc:
        config.resolve_source("acme/some-model")
    msg = str(exc.value)
    assert "hf download acme/some-model" in msg
    assert "--local-dir models/" in msg


def test_resolve_source_dir_without_config_json_error(tmp_path, monkeypatch):
    monkeypatch.setattr(estimator, "MODELS_DIR", str(tmp_path))  # H1: อยู่ใต้ sandbox root
    (tmp_path / "bare").mkdir()
    with pytest.raises(config.CompressionError) as exc:
        config.resolve_source(str(tmp_path / "bare"))
    assert "config.json" in str(exc.value)


def test_resolve_source_models_name(monkeypatch, tmp_path):
    root = tmp_path / "models"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(estimator, "MODELS_DIR", str(root))
    assert config.resolve_source("alpha") == root / "alpha"


def test_artifact_paths_layout(tmp_path):
    src = tmp_path / "m"
    assert config.gguf_dir(src) == src / "gguf"
    assert config.artifact_path(src, "q4_k_m") == src / "gguf" / "m-q4_k_m.gguf"
    p = config.report_path(src, "q8_0")
    assert p.parent.parent == Path("benchmarks")
    assert p.name == "q8_0.json"
    assert p.parent.name.startswith("m")  # ยังอ่านออกว่ามาจากชื่อไหน


def test_report_path_disambiguates_same_basename(tmp_path):
    """Review #3: basename เดียวกันคนละที่ → report คนละไฟล์ (ไม่เขียนทับกัน)"""
    a = tmp_path / "a" / "m"
    b = tmp_path / "b" / "m"
    a.mkdir(parents=True)
    b.mkdir(parents=True)

    pa = config.report_path(a, "q8_0")
    pb = config.report_path(b, "q8_0")

    assert pa != pb
    assert config.report_path(a, "q8_0") == pa  # deterministic
    assert pa != config.report_path(a, "q4_k_m")  # variant แยกกัน


def test_device_args_vulkan_is_empty_and_cpu_is_not():
    assert config.device_args("vulkan") == []
    assert config.device_args("cpu") != []


@pytest.mark.skipif(shutil.which("vulkaninfo") is None, reason="vulkaninfo not installed")
def test_check_vulkan_real():
    assert isinstance(config.check_vulkan(), bool)


# ---------------------------------------------------------------------------
# Bug: resolve_source — path นอก sandbox / path ที่ไม่มีจริง ต้องไม่แนะนำ "hf download"
# (None ถูกตีความว่า = โหลดจาก Hub ทั้งที่ input เป็น local path)
# ---------------------------------------------------------------------------


def test_resolve_source_outside_path_gives_sandbox_error(monkeypatch, tmp_path):
    outside = tmp_path / "outside-model"
    outside.mkdir()
    (outside / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(estimator, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(estimator, "REPO_ROOT", str(tmp_path / "repo"))

    with pytest.raises(config.CompressionError) as exc:
        config.resolve_source(str(outside))
    msg = str(exc.value)
    assert "outside" in msg  # error บอกเรื่อง sandbox — ไม่ใช่ "ไปโหลดจาก Hub"
    assert "hf download" not in msg


def test_resolve_source_nonexistent_absolute_path_no_hf_suggestion():
    # path .absolut eที่ไม่มีจริง — แนะนำ "hf download /abs/path" คือคำสั่งที่ใช้ไม่ได้เลย
    with pytest.raises(config.CompressionError) as exc:
        config.resolve_source("/no/such/model-dir")
    assert "hf download /no/such" not in str(exc.value)
