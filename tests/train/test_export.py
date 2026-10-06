"""Tests สำหรับ core/train/export.py — save_adapter_only + merge_export (spawn target, §5 VRAM)"""

import json

import pytest

from configs.safe_defaults import SEED
from core.train import export as wa


def _config(**over) -> dict:
    """config fixture = REQUIRED_CONFIG_KEYS ครบ (core/train/args.py:113) — ตามสัญญา save_adapter_only(config=...)"""
    base = {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": "data_cache/finetune_run",
        "max_seq_length": 1024,
        "max_steps": 6,
        "code_limit": 64,
        "lora_rank": 8,
    }
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _offline_hub(monkeypatch):
    """hermetic ทั้งไฟล์ (review Important #2): แทน binding model_info ก่อนทุก test
    — ไม่งั้น 3 tests เดิมเรียก hub จริงทุกรัน (ค้าง 90s+ บนเครือข่ายที่ drop packet)"""
    def _offline(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(wa, "model_info", _offline)


def test_save_adapter_only_copies(tmp_path):
    out = tmp_path / "run1"
    ckpt = out / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")
    exports = tmp_path / "exports"
    dest = wa.save_adapter_only(out, config=_config(), exports_dir=exports)
    assert dest == exports / "run1"
    assert (dest / "adapter_config.json").exists()
    assert (dest / "adapter_model.safetensors").read_bytes() == b"fake"


def test_save_adapter_only_no_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError):
        wa.save_adapter_only(tmp_path / "empty_missing", config=_config())


def test_save_adapter_only_from_interrupted_swap_artifact(tmp_path):
    """Sec-12: final หาย เหลือ `checkpoint-<n>.old` กลางทาง swap → export ต้องใช้ได้
    (latest_checkpoint เลือก artifact ให้) และห้าม rename artifact"""
    out = tmp_path / "run1"
    ckpt = out / "checkpoint-10.old"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")
    exports = tmp_path / "exports"
    dest = wa.save_adapter_only(out, config=_config(), exports_dir=exports)
    assert dest == exports / "run1"
    assert (dest / "adapter_config.json").exists()
    assert (dest / "adapter_model.safetensors").read_bytes() == b"fake"
    assert (out / "checkpoint-10.old").is_dir()  # read-only — ไม่ rename/rmtree
    assert not (out / "checkpoint-10").exists()


def test_merge_export_importable():
    assert callable(wa.merge_export)


def test_save_adapter_only_config_without_weights_raises(tmp_path):
    """M5: มีแค่ adapter_config.json (น้ำหนักหาย) → ต้อง raise ไม่ใช่รายงานสำเร็จ"""
    ckpt = tmp_path / "run" / "checkpoint-5"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError):
        # exports_dir ชี้ tmp_path — dest จริงของ test นี้ ไม่งั้น assert ล่าง check path ผิด (review Critical #1)
        wa.save_adapter_only(tmp_path / "run", config=_config(), exports_dir=tmp_path / "exports")
    assert not (tmp_path / "exports" / "run" / "run_manifest.json").exists()  # fail ไม่ค้าง (acceptance #4)


def test_save_adapter_only_copies_shards(tmp_path):
    """M5: sharded weights ต้องถูก copy ครบ (ไม่ใช่แค่ index)"""
    ckpt = tmp_path / "run" / "checkpoint-7"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model-00001-of-00002.safetensors").write_bytes(b"a")
    (ckpt / "adapter_model-00002-of-00002.safetensors").write_bytes(b"b")
    (ckpt / "adapter_model.safetensors.index.json").write_text("{}")
    dest = wa.save_adapter_only(tmp_path / "run", config=_config(), exports_dir=tmp_path / "exports")
    names = sorted(p.name for p in dest.iterdir())
    # exact list — มีแค่ไฟล์ adapter ครบชุด + run_manifest.json เท่านั้น (ไม่มีไฟล์อื่นหลุดเข้า)
    assert names == [
        "adapter_config.json",
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
        "adapter_model.safetensors.index.json",
        "run_manifest.json",
    ]


def test_save_adapter_only_rejects_output_outside_sandbox(tmp_path, monkeypatch):
    """Sec-10: on_save_adapter ไม่ผ่าน validate_config → คัดลอกจากนอก sandbox ได้ — H1 rule ต้อง applied ที่ save (คุ้มครองทุก caller)"""
    run = tmp_path / "run"
    ckpt = run / "checkpoint-3"
    ckpt.mkdir(parents=True)                    # fixture ลอกจาก test_save_adapter_only_copies (:347-350)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")
    # หัน temp root ออกจาก tmp_path → path นี้ "นอก sandbox" ทั้งที่อยู่ใน tmp (hermetic)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="under the repo or the system temp dir"):
        wa.save_adapter_only(run, config=_config(), exports_dir=tmp_path / "exports")


def test_save_adapter_only_writes_manifest(tmp_path):
    """acceptance #3: manifest เขียนที่ dest สำเร็จ + ออฟไลน์ → license null, ไม่แตะเน็ตจริง
    (hub ออฟไลน์มาจาก autouse fixture)"""
    out = tmp_path / "run1"
    ckpt = out / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")

    dest = wa.save_adapter_only(out, config=_config(), exports_dir=tmp_path / "exports")
    m = json.loads((dest / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["schema_version"] == "1.0"
    assert m["artifact_type"] == "run_manifest"
    assert m["export"]["mode"] == "adapter_only"
    assert m["export"]["path"].endswith("run1")
    assert m["base_model"]["license"] is None         # null ไม่ใช่ 0/เดา (acceptance #2)
    assert m["base_model"]["num_params"] is None
    assert m["export"]["dtype"] is None
    assert m["training"]["seed"] == SEED


def test_export_suite_is_offline():
    """hermeticity pin (review Important #2): wa.model_info ต้องถูกแทนด้วย fixture ตลอดเวลา
    — ถ้าเป็น function ตัวเดียวกับ huggingface_hub จริง = test นี้เรียก hub จริงทุกครั้ง (ค้างได้บนเครือข่ายที่ drop packet)"""
    import huggingface_hub

    assert wa.model_info is not huggingface_hub.model_info
