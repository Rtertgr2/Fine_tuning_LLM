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
    assert "parameter count" in result.reason


def test_invalid_user_params_blocked(monkeypatch, tmp_path):
    # P ที่ใช้ไม่ได้ (0/ลบ/NaN/inf) ต้องไม่ได้ "safe" ปลอม และห้าม crash
    _patch_fetch(monkeypatch, tmp_path, error=OSError("offline"))
    for bad in (0, -1.5, float("nan"), float("inf")):
        result = est.estimate(
            _hw_ready(), model_id="acme/offline-model", user_params_b=bad
        )
        assert result.verdict == "blocked", bad
        assert "parameter count" in result.reason


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


# ---------------------------------------------------------------------------
# Folder detection: dropdown models/ (design 2026-10-02 model-dataset-picker)
# ---------------------------------------------------------------------------


def test_list_models_returns_only_config_folders(tmp_path, monkeypatch):
    from core import estimator as est

    root = tmp_path / "models"
    (root / "good-model").mkdir(parents=True)
    (root / "good-model" / "config.json").write_text("{}", encoding="utf-8")
    (root / "half-model").mkdir()
    (root / "half-model" / "weights.safetensors").write_bytes(b"")

    monkeypatch.setattr(est, "MODELS_DIR", str(root))
    assert est.list_models() == ["good-model"]


def test_list_models_missing_dir_is_empty(tmp_path, monkeypatch):
    from core import estimator as est

    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path / "not-there"))
    assert est.list_models() == []


# ---------------------------------------------------------------------------
# resolve_model_spec: โมเดลท้องถิ่น (design 2026-10-02 model-dataset-picker)
# ---------------------------------------------------------------------------


def _write_local_config(model_dir, *, hidden=1024, layers=12, inter=4096, vocab=32000):
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "intermediate_size": inter,
                "vocab_size": vocab,
            }
        ),
        encoding="utf-8",
    )


def test_resolve_model_spec_reads_local_config(tmp_path, monkeypatch):
    model_dir = tmp_path / "my-model"
    _write_local_config(model_dir)
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path))  # H1: ต้องอยู่ใต้ sandbox root

    spec = est.resolve_model_spec(str(model_dir), None)

    assert spec.source == "local_config"
    assert spec.hidden_size == 1024
    assert spec.num_layers == 12
    expected = 12 * (4 * 1024 * 1024 + 3 * 1024 * 4096) + 32000 * 1024
    assert spec.num_params == expected


def test_resolve_model_spec_bare_name_under_models_dir(tmp_path, monkeypatch):
    root = tmp_path / "models"
    _write_local_config(root / "alpha", hidden=768, layers=6, inter=3072, vocab=20000)

    monkeypatch.setattr(est, "MODELS_DIR", str(root))
    spec = est.resolve_model_spec("alpha", None)

    assert spec.source == "local_config"
    assert spec.hidden_size == 768
    assert spec.num_layers == 6


def test_resolve_model_spec_broken_local_config_uses_user_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path))  # H1: อยู่ใต้ sandbox root
    model_dir = tmp_path / "broken-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{ not json", encoding="utf-8")

    with pytest.warns(UserWarning, match="broken-model"):  # M3: JSON เสีย → warning + fallback
        spec = est.resolve_model_spec(str(model_dir), 1.5)

    assert spec.source == "user_fallback"
    assert spec.num_params == 1_500_000_000


# ---------------------------------------------------------------------------
# H1: path sandbox (Fix.md security review) — path นอก root ต้องถูกบล็อก
# ---------------------------------------------------------------------------


def test_resolve_local_model_rejects_path_outside_roots(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside-model"
    outside.mkdir()
    monkeypatch.setattr(est, "MODELS_DIR", str(root))
    monkeypatch.setattr(est, "REPO_ROOT", str(tmp_path / "no-repo"))

    # มีอยู่จริงนอก sandbox → ValueError (ห้าม None = caller จะส่งเข้า Hub)
    with pytest.raises(ValueError, match="outside"):
        est.resolve_local_model(str(outside))


def test_resolve_local_model_rejects_dotdot_escape(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    (tmp_path / "secret").mkdir()
    monkeypatch.setattr(est, "MODELS_DIR", str(root))
    monkeypatch.setattr(est, "REPO_ROOT", str(tmp_path / "no-repo"))

    # เดิม: Path(root) / "../secret" resolve หลุด root โดยไม่มีใครตรวจ
    with pytest.raises(ValueError, match="outside"):
        est.resolve_local_model(str(root / ".." / "secret"))


def test_resolve_local_model_allows_repo_internal_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "exports" / "merged").mkdir(parents=True)
    monkeypatch.setattr(est, "REPO_ROOT", str(repo))
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path / "models"))

    got = est.resolve_local_model(str(repo / "exports" / "merged"))
    assert got == (repo / "exports" / "merged").resolve()  # Run B workflow ต้องใช้ได้ต่อ


def test_resolve_local_model_name_under_models_dir_still_works(tmp_path, monkeypatch):
    root = tmp_path / "models"
    (root / "alpha").mkdir(parents=True)
    monkeypatch.setattr(est, "MODELS_DIR", str(root))
    monkeypatch.setattr(est, "REPO_ROOT", str(tmp_path / "no-repo"))

    assert est.resolve_local_model("alpha") == (root / "alpha").resolve()


# ---------------------------------------------------------------------------
# M3: except กว้างเกินไป (Fix.md) — bug จริงต้องโผล่ ไม่ใช่ fallback เงียบ
# ---------------------------------------------------------------------------


def test_resolve_model_spec_unexpected_error_propagates(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(est, "hf_hub_download", _boom)
    with pytest.raises(RuntimeError, match="unexpected bug"):  # เดิม: กลืนเงียบ → fallback
        est.resolve_model_spec("acme/boom-model", None)


def test_resolve_model_spec_known_errors_warn_and_fall_back(monkeypatch, tmp_path):
    # OSError/ValueError/KeyError/JSONDecodeError = ที่คาดไว้ (offline/JSON เสีย/config ไม่ครบ)
    # → warning + fallback คงพฤติกรรมเดิม — hub 1.x: HfHubHTTPError/LocalEntryNotFound ⊂ OSError
    _patch_fetch(monkeypatch, tmp_path, error=OSError("offline"))
    with pytest.warns(UserWarning, match="acme/offline-model"):
        spec = est.resolve_model_spec("acme/offline-model", 1.5)
    assert spec.source == "user_fallback"


# ---------------------------------------------------------------------------
# Bug: path ที่มีอยู่จริงนอก sandbox คืน None → caller ตีความว่า "โหลดจาก Hub"
# (confused deputy: hf_hub_download ได้ path จริงไป → อาจโหลด repo ผิดเงียบ ๆ)
# ---------------------------------------------------------------------------


def test_resolve_model_spec_outside_existing_path_never_hits_hub(monkeypatch, tmp_path):
    outside = tmp_path / "outside-model"
    outside.mkdir()
    (outside / "config.json").write_text(
        json.dumps({"hidden_size": 64, "num_hidden_layers": 2,
                    "intermediate_size": 128, "vocab_size": 256}),
        encoding="utf-8",
    )
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(est, "REPO_ROOT", str(tmp_path / "repo"))

    def _hub_called(*args, **kwargs):
        raise AssertionError(f"HUB CALLED WITH LOCAL PATH: {args!r}")

    monkeypatch.setattr(est, "hf_hub_download", _hub_called)

    # อยู่นอก sandbox → ต้อง reject ด้วย error ชัดเจน ห้าม fall through ไป Hub
    with pytest.raises(ValueError, match="outside"):
        est.resolve_model_spec(str(outside), None)
