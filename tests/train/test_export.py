"""Tests สำหรับ core/train/export.py — save_adapter_only + merge_export (spawn target, §5 VRAM)"""

import pytest

from core.train import export as wa


def test_save_adapter_only_copies(tmp_path):
    out = tmp_path / "run1"
    ckpt = out / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")
    exports = tmp_path / "exports"
    dest = wa.save_adapter_only(out, exports_dir=exports)
    assert dest == exports / "run1"
    assert (dest / "adapter_config.json").exists()
    assert (dest / "adapter_model.safetensors").read_bytes() == b"fake"


def test_save_adapter_only_no_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError):
        wa.save_adapter_only(tmp_path / "empty_missing")


def test_merge_export_importable():
    assert callable(wa.merge_export)


def test_save_adapter_only_config_without_weights_raises(tmp_path):
    """M5: มีแค่ adapter_config.json (น้ำหนักหาย) → ต้อง raise ไม่ใช่รายงานสำเร็จ"""
    ckpt = tmp_path / "run" / "checkpoint-5"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError):
        wa.save_adapter_only(tmp_path / "run")


def test_save_adapter_only_copies_shards(tmp_path):
    """M5: sharded weights ต้องถูก copy ครบ (ไม่ใช่แค่ index)"""
    ckpt = tmp_path / "run" / "checkpoint-7"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model-00001-of-00002.safetensors").write_bytes(b"a")
    (ckpt / "adapter_model-00002-of-00002.safetensors").write_bytes(b"b")
    (ckpt / "adapter_model.safetensors.index.json").write_text("{}")
    dest = wa.save_adapter_only(tmp_path / "run", exports_dir=tmp_path / "exports")
    names = sorted(p.name for p in dest.iterdir())
    assert names == [
        "adapter_config.json",
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
        "adapter_model.safetensors.index.json",
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
        wa.save_adapter_only(run, exports_dir=tmp_path / "exports")
