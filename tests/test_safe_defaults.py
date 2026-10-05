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


def test_ipc_and_training_blocks():
    # Subprocess / IPC
    assert sd.ABORT_SIGTERM_TIMEOUT_S == 10.0
    # Training hyperparameters (plan.md §4.4)
    assert sd.GRADIENT_ACCUMULATION_STEPS == 8
    assert sd.LEARNING_RATE == 2e-4
    assert sd.WARMUP_RATIO == 0.03
    assert sd.SAVE_TOTAL_LIMIT == 2
    assert sd.SAVE_STEPS == 100
    assert sd.MAX_STEPS == 500
    assert sd.NONFINITE_ABORT_THRESHOLD == 3
    assert sd.LORA_RANK_DEFAULT == 8
    assert sd.LORA_TARGET_MODULES == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert sd.TRAIN_CODE_LIMIT == 8192
    # P1 B1: hub fetch pin — revision คงที่ + cache อยู่ใต้ data_cache
    assert sd.HF_HUB_REVISION == "main"
    assert sd.HF_HUB_CACHE_DIR == "data_cache/hf_hub"
