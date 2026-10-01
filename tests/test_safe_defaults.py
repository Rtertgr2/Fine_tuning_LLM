"""Pin ค่าคงที่ของ safe_defaults ตามที่ plan.md กำหนด"""

from configs import safe_defaults as sd


def test_resource_thresholds():
    assert sd.RAM_MIN_GB == 16
    assert sd.DISK_MIN_GB == 20
    assert sd.VRAM_SAFE_RATIO == 0.75
    assert sd.VRAM_WARNING_RATIO == 0.90


def test_training_defaults():
    assert sd.FIM_RATE == 0.5
    assert sd.MAX_SEQ_LENGTH_DEFAULT == 1024
    assert sd.MAX_SEQ_LENGTH_CAP == 2048
    assert sd.SEED == 42
    assert 0.05 <= sd.HELDOUT_RATIO <= 0.10


def test_default_ids():
    assert sd.DEFAULT_MODEL_ID == "Qwen/Qwen2.5-Coder-0.5B"
    assert sd.DEFAULT_DATASET_ID == "smangrul/hf-stack-v1"
    assert sd.DEFAULT_DATASET_COLUMN == "content"
