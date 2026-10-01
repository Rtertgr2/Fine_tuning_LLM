"""Tests สำหรับ core/estimator.py — verdict thresholds, ที่มาของ P, passthrough (สเปก plan.md §4.2)"""

import json

import pytest

import core.estimator as est
from configs.safe_defaults import DEFAULT_MODEL_ID, DEFAULT_MODEL_NUM_PARAMS

GB = 1024**3

FIXTURE_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "vocab_size": 1000,
}
# 2×(4×64² + 3×64×128) + 1000×64 = 81_920 + 64_000
FIXTURE_NUM_PARAMS = 145_920


def _hw_ready(free: float = 20.0) -> dict:
    return {
        "status": "ready",
        "device_name": "Intel(R) Arc(TM) B580 Graphics",
        "total_vram_gb": 12.0,
        "free_vram_gb": free,
        "ram_available_gb": 20.0,
        "disk_free_gb": 50.0,
    }


def _patch_fetch(monkeypatch, tmp_path, content=None, error=None):
    """แทนที่ hf_hub_download — คืน path config fixture หรือ raise"""
    if error is not None:
        def _raise(*args, **kwargs):
            raise error

        monkeypatch.setattr(est, "hf_hub_download", _raise)
    else:
        p = tmp_path / "config.json"
        p.write_text(json.dumps(content), encoding="utf-8")
        monkeypatch.setattr(est, "hf_hub_download", lambda *args, **kwargs: str(p))


def test_resolve_default_model():
    spec = est.resolve_model_spec(DEFAULT_MODEL_ID, None)
    assert spec.num_params == 498_431_872
    assert spec.hidden_size == 896
    assert spec.num_layers == 24
    assert spec.source == "default"


def test_resolve_from_config(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, content=FIXTURE_CONFIG)
    spec = est.resolve_model_spec("acme/fixture-model", None)
    assert spec.num_params == FIXTURE_NUM_PARAMS
    assert spec.hidden_size == 64
    assert spec.num_layers == 2
    assert spec.source == "hf_config"


def test_blocked_when_params_unknown(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, error=OSError("offline"))
    result = est.estimate(_hw_ready(), model_id="acme/offline-model", user_params_b=None)
    assert result.verdict == "blocked"
    assert "พารามิเตอร์" in result.reason


def test_user_fallback_params(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, error=OSError("offline"))
    result = est.estimate(_hw_ready(), model_id="acme/offline-model", user_params_b=1.5)
    assert result.spec_source == "user_fallback"
    assert result.weights_gb == pytest.approx(1.5e9 * 2 / GB)
    assert result.verdict == "safe"


def test_classify_boundaries():
    # free=20.0 → 0.75×20=15.0 และ 0.90×20=18.0 (exact ใน float)
    assert est.classify(15.0, 20.0) == "safe"
    assert est.classify(15.001, 20.0) == "warning"
    assert est.classify(18.0, 20.0) == "warning"
    assert est.classify(18.001, 20.0) == "blocked"


def test_insufficient_passthrough():
    for status in ("insufficient_ram", "insufficient_disk", "no_xpu"):
        hw = _hw_ready()
        hw["status"] = status
        result = est.estimate(hw, model_id=DEFAULT_MODEL_ID)
        assert result.verdict == status


def test_estimate_components_sum():
    result = est.estimate(_hw_ready(), model_id=DEFAULT_MODEL_ID)
    assert result.verdict == "safe"
    assert result.spec_source == "default"
    assert result.weights_gb == pytest.approx(DEFAULT_MODEL_NUM_PARAMS * 2 / GB)
    parts = (
        result.weights_gb + result.trainable_gb + result.activations_gb + result.overhead_gb
    )
    assert parts == pytest.approx(result.total_required_gb)
