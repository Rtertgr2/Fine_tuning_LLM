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

# --- Model / Dataset defaults ---
DEFAULT_MODEL_ID: str = "Qwen/Qwen2.5-Coder-0.5B"
DEFAULT_DATASET_ID: str = "smangrul/hf-stack-v1"
DEFAULT_DATASET_COLUMN: str = "content"
HELDOUT_RATIO: float = 0.10
