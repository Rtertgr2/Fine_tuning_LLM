"""ค่าคงที่ของระบบ (safe defaults) — pin ด้วย tests/test_safe_defaults.py

แก้ค่าที่นี่เท่านั้น ห้าม hardcode ค่าเหล่านี้ในโค้ดส่วนอื่น
"""

from __future__ import annotations

# --- ทรัพยากร (hardware gate) ---
RAM_MIN_GB: int = 16
DISK_MIN_GB: int = 20
VRAM_SAFE_RATIO: float = 0.75
VRAM_WARNING_RATIO: float = 0.90

# --- การเทรน ---
FIM_RATE: float = 0.5
MAX_SEQ_LENGTH_DEFAULT: int = 1024
MAX_SEQ_LENGTH_CAP: int = 2048
SEED: int = 42
MIN_SAMPLE_LINES: int = 3

# --- Model / Dataset defaults ---
DEFAULT_MODEL_ID: str = "Qwen/Qwen2.5-Coder-0.5B"
DEFAULT_DATASET_ID: str = "smangrul/hf-stack-v1"
DEFAULT_DATASET_COLUMN: str = "content"
HELDOUT_RATIO: float = 0.10

# --- Estimator (core/estimator.py) ---
# ค่า hardcode ของโมเดล default (plan.md §4.2) — วัดจริงจาก Phase 1:
# 498,431,872 = ผลรวมพารามิเตอร์จาก smoke test, 4,399,104 = LoRA r=8 trainable จริง
DEFAULT_MODEL_NUM_PARAMS: int = 498_431_872
DEFAULT_MODEL_HIDDEN_SIZE: int = 896
DEFAULT_MODEL_NUM_LAYERS: int = 24
DEFAULT_LORA_NUM_PARAMS: int = 4_399_104
DEFAULT_BATCH_SIZE: int = 1
ESTIMATOR_OVERHEAD_GB: float = 1.0
# dims สำรองตอนโหลด config ไม่ได้ (upper bound อนุรักษ์นิยมสำหรับ activation)
ESTIMATOR_FALLBACK_HIDDEN_SIZE: int = 4096
ESTIMATOR_FALLBACK_NUM_LAYERS: int = 40
