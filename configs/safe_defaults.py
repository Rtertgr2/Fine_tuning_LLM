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
# โฟลเดอร์วาง asset สำหรับ dropdown (design model-dataset-picker: detect → dropdown)
DATASETS_DIR: str = "datasets"
MODELS_DIR: str = "models"
HELDOUT_RATIO: float = 0.10

# --- Estimator (core/estimator.py) ---
# ค่า hardcode ของโมเดล default (plan.md §4.2) — วัดจริงจาก Phase 1:
# 498,431,872 = ผลรวมพารามิเตอร์จาก smoke test, 4,399,104 = LoRA r=8 trainable จริง
DEFAULT_MODEL_NUM_PARAMS: int = 498_431_872
DEFAULT_MODEL_HIDDEN_SIZE: int = 896
DEFAULT_MODEL_NUM_LAYERS: int = 24
DEFAULT_MODEL_VOCAB_SIZE: int = 151_936  # config.json จริงของ Qwen2.5-Coder-0.5B (logits buffer)
DEFAULT_LORA_NUM_PARAMS: int = 4_399_104
DEFAULT_BATCH_SIZE: int = 1
ESTIMATOR_OVERHEAD_GB: float = 0.8  # calibrate Phase 5 §8.4: เดิม 1.0 → err 11.1% (peak 1.81GB) → 0.80
# dims สำรองตอนโหลด config ไม่ได้ (upper bound อนุรักษ์นิยมสำหรับ activation)
ESTIMATOR_FALLBACK_HIDDEN_SIZE: int = 4096
ESTIMATOR_FALLBACK_NUM_LAYERS: int = 40

# --- Subprocess / IPC ---
ABORT_SIGTERM_TIMEOUT_S: float = 10.0

# --- Training hyperparameters (plan.md §4.4) — ห้าม hardcode ซ้ำในโค้ดส่วนอื่น ---
GRADIENT_ACCUMULATION_STEPS: int = 8
LEARNING_RATE: float = 2e-4
WARMUP_RATIO: float = 0.03
SAVE_TOTAL_LIMIT: int = 2
SAVE_STEPS: int = 100
MAX_STEPS: int = 500
# Validation loop ระหว่างเทรน (plan3 §4.4 เพิ่มเติม — overfit ต้องเห็นกลางทาง ไม่ใช่ตอนจบ)
VAL_EVAL_SAMPLES: int = 32  # heldout samples ต่อรอบ eval (val loss ไม่ต้องแม่นเท่า final eval)
VAL_EVAL_MIN_STEPS: int = 50  # eval อย่างน้อยทุก 50 steps (ถี่กว่านี้ overhead ไม่คุ้ม)
NONFINITE_ABORT_THRESHOLD: int = 3
LORA_RANK_DEFAULT: int = 8
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
TRAIN_CODE_LIMIT: int = 8192
