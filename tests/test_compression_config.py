"""Unit tests สำหรับ core.compression.config — paths/variants/device/resolve_source"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core import estimator
from core.compression import config


def test_tools_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert config.tools_dir() == tmp_path


def test_tools_dir_default(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    assert config.tools_dir() == Path("tools/llama.cpp")


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


def test_resolve_source_dir_without_config_json_error(tmp_path):
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
    assert config.report_path(src, "q8_0") == Path("benchmarks/m/q8_0.json")


def test_device_args_vulkan_is_empty_and_cpu_is_not():
    assert config.device_args("vulkan") == []
    assert config.device_args("cpu") != []


@pytest.mark.skipif(shutil.which("vulkaninfo") is None, reason="vulkaninfo not installed")
def test_check_vulkan_real():
    assert isinstance(config.check_vulkan(), bool)
