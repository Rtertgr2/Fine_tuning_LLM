# Phase 2: โมดูลคำนวณและประมวลผลข้อมูล Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** สร้าง `core/hardware.py` (สถานะฮาร์ดแวร์ 4 ค่า), `core/estimator.py` (ประเมิน VRAM ก่อนเทรน), `core/dataset_builder.py` (แปลงโค้ดเป็น FIM ตามกติกา) พร้อมชุด test ครอบคลุมทั้ง 3 โมดูล

**Architecture:** โมดูลล้วนเป็น pure function / dataclass ไม่มี state — `hardware.inspect()` คืน dict ตาม contract ของสเปก, `estimator.estimate()` รับ dict นั้น + model spec แล้วคืน verdict, `dataset_builder` แปลงข้อความโค้ดเป็น string รูปแบบ PSM (tokenization ให้ SFTTrainer ทำทีหลังเหมือน Phase 1) ทุกค่าคงที่อ่านจาก `configs/safe_defaults.py`

**Tech Stack:** Python 3.11, torch (XPU), psutil, huggingface_hub (hf_hub_download แคช config.json), datasets (ไม่ต้องใช้ใน unit test), pytest

**Spec:** `plan.md` §4 โมดูลที่ 1–3 (บรรทัด 93–170) + §6 Phase 2 (บรรทัด 261–266)

## ⚠️ Execution Rules (override ตามคำสั่ง human partner 2026-10-01)

- **ห้ามรัน test ระหว่าง Task 1–3** — เขียนโค้ด + test ให้ครบทุก task ก่อน แล้วรัน pytest **ครั้งเดียว** ที่ Task 4
- ถ้า Task 4 FAIL จึงค่อย debug + rerun (ช่วง debug รันซ้ำได้ตามปกติ)
- Worktree นี้ไม่มี `.venv` — คำสั่ง test ทุกจุดใช้: `../../.venv/bin/python -m pytest tests/ -v` (cwd = worktree root, `python -m` ใส่ cwd ลง sys.path → import `core`/`configs` ได้)

## Global Constraints

- Python **3.11 เท่านั้น** ผ่าน `../../.venv/bin/python` (ห้าม python ระบบ 3.14)
- ห้ามแตะการติดตั้ง torch (มาจาก XPU index, `torch==2.14.1+xpu` ใน constraints.txt)
- ค่า threshold: `RAM_MIN_GB = 16`, `DISK_MIN_GB = 20`, `VRAM_SAFE_RATIO = 0.75`, `VRAM_WARNING_RATIO = 0.90` (มีใน `configs/safe_defaults.py` แล้ว — import มาใช้ ห้าม hardcode ซ้ำ)
- หน่วย GB ทุกจุด = `bytes / 1024**3` (ตรง basis เดียวกับ `check_runtime.py` และ `mem_get_info`)
- FIM: `fim_rate = 0.5`, รูปแบบ **PSM เท่านั้น**, ตัดที่ **boundary บรรทัด**, EOS ท้ายทุก sample, `seed = 42`, `max_seq_length` default 1024 / cap 2048
- ห้าม `add_special_tokens` / ห้าม `resize_token_embeddings` (FIM tokens มีใน tokenizer แล้ว)
- Dataset default: `smangrul/hf-stack-v1` / คอลัมน์ `content` / `HELDOUT_RATIO = 0.10`
- `$P$ ของโมเดล default ต้อง hardcode ใน `safe_defaults.py` — **ห้ามเดาค่าเมื่อ resolve ไม่ได้** (บล็อกแทน)
- ทุก task commit หลัง Task 4 เขียวแล้ว (commit แยกตาม task)

## Review Focus

1. **Boundary ของ verdict ที่ 0.75/0.90** — ค่าเท่ากับ threshold ต้องตก Safe/Warning (≤) ไม่ใช่ Warning/Blocked → `tests/test_estimator.py::test_classify_boundaries`
2. **ลำดับ status ใน hardware** — `no_xpu` ต้องชนะ `insufficient_ram`/`insufficient_disk` (เช็ค XPU ก่อน) → `tests/test_hardware.py::test_no_xpu_takes_precedence`
3. **Offline fallback ห้ามเดา P** — config.json อ่านไม่ได้ + ไม่กรอก P → `verdict == "blocked"` พร้อม reason → `tests/test_estimator.py::test_blocked_when_params_unknown`
4. **ตัดกลางบรรทัด (ห้าม)** — prefix/middle/suffix ต้องประกอบด้วยบรรทัดเต็ม ๆ และ reconstruct ได้เป๊ะ → `tests/test_dataset_builder.py::test_split_line_boundary_and_reconstruct`
5. **EOS หายหลัง truncate** — sample ยาวต้องถูกตัดไม่เกิน 1024 tokens **แต่ยังลงท้ายด้วย EOS** → `tests/test_dataset_builder.py::test_truncation_keeps_eos`

---

## File Structure

- Create: `core/__init__.py` (ว่าง), `core/hardware.py`, `core/estimator.py`, `core/dataset_builder.py`
- Modify: `configs/safe_defaults.py` (เพิ่มค่าคงที่ estimator + min sample lines)
- Create: `tests/test_hardware.py`, `tests/test_estimator.py`, `tests/test_dataset_builder.py`

---

### Task 1: `core/hardware.py` — Hardware Environment Inspector

**Files:**
- Create: `core/__init__.py`, `core/hardware.py`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: `configs.safe_defaults.RAM_MIN_GB`, `DISK_MIN_GB`
- Produces: `inspect(output_dir: str = ".") -> dict` คืน contract เป๊ะตามสเปก §4.1:
  `{"status": "ready"|"no_xpu"|"insufficient_ram"|"insufficient_disk", "device_name": str, "total_vram_gb": float, "free_vram_gb": float, "ram_available_gb": float, "disk_free_gb": float}`
  - `status == "ready"` ต้องเชื่อถือได้ 100% — เช็คตามลำดับ: XPU → RAM → disk → ready
  - `no_xpu`: `device_name = ""`, VRAM = `0.0` แต่ยังวัด RAM/disk ได้ตามจริง

- [ ] **Step 1: เขียน `tests/test_hardware.py`** — monkeypatch `torch.xpu.is_available` / `torch.xpu.mem_get_info` / `torch.xpu.get_device_name` และ `psutil.virtual_memory` / `psutil.disk_usage` (คืน `SimpleNamespace`) แล้ว assert:
  - `test_contract_keys_and_types`: คืน dict มี key ครบ 6 ตัว + type ถูกต้อง
  - `test_ready_when_all_ok`: XPU มี, RAM 20GB, disk 50GB → `status == "ready"` + ค่า GB ตรงที่ stub คืน (หาร 1024³)
  - `test_no_xpu_takes_precedence`: XPU ไม่มี + RAM 15GB + disk 10GB → `status == "no_xpu"` (ไม่ใช่ insufficient_ram)
  - `test_insufficient_ram`: XPU มี + RAM 15.9GB → `status == "insufficient_ram"`
  - `test_insufficient_disk`: XPU มี + RAM 20GB + disk 19.9GB → `status == "insufficient_disk"`

- [ ] **Step 2: สร้าง `core/__init__.py` (ว่าง) + implement `core/hardware.py`**
  - `inspect(output_dir: str = ".") -> dict` — ลำดับ: (1) `torch.xpu.is_available()` ถ้า False → status `no_xpu` (VRAM 0.0, device_name "") (2) `mem_get_info(0)` คืน `(free_b, total_b)` → GB (3) `psutil.virtual_memory().available` < `RAM_MIN_GB×1024³` → `insufficient_ram` (4) `psutil.disk_usage(output_dir).free` < `DISK_MIN_GB×1024³` → `insufficient_disk` (5) else `ready`
  - ทุกค่า GB = `bytes / 1024**3`

- [ ] **Step 3: Commit (ยังไม่รัน test — ตาม Execution Rules)**

```bash
git add core/ tests/test_hardware.py && git commit -m "feat: core/hardware.py ตรวจสถานะ XPU/RAM/disk ตาม contract สเปก"
```

---

### Task 2: `core/estimator.py` — Feasibility & VRAM Calculator

**Files:**
- Modify: `configs/safe_defaults.py` (เพิ่ม block "Estimator")
- Create: `core/estimator.py`
- Test: `tests/test_estimator.py`

**Interfaces:**
- Consumes: `hardware.inspect()` dict (Task 1), `safe_defaults.VRAM_SAFE_RATIO/VRAM_WARNING_RATIO/MAX_SEQ_LENGTH_DEFAULT/DEFAULT_MODEL_ID`
- Produces:
  - `safe_defaults` เพิ่ม: `DEFAULT_MODEL_NUM_PARAMS: int = 498_431_872`, `DEFAULT_MODEL_HIDDEN_SIZE: int = 896`, `DEFAULT_MODEL_NUM_LAYERS: int = 24`, `DEFAULT_LORA_NUM_PARAMS: int = 4_399_104`, `DEFAULT_BATCH_SIZE: int = 1`, `ESTIMATOR_OVERHEAD_GB: float = 1.0`, `ESTIMATOR_FALLBACK_HIDDEN_SIZE: int = 4096`, `ESTIMATOR_FALLBACK_NUM_LAYERS: int = 40`
  - `class ModelSpec(NamedTuple)`: `num_params: int`, `hidden_size: int`, `num_layers: int`, `source: str` ("default"|"hf_config"|"user_fallback")
  - `class ModelSpecUnavailable(Exception)`
  - `class EstimateResult(NamedTuple)`: `verdict: str`, `reason: str`, `total_required_gb: float`, `weights_gb: float`, `trainable_gb: float`, `activations_gb: float`, `overhead_gb: float`, `free_vram_gb: float`, `spec_source: str`
  - `resolve_model_spec(model_id: str, user_params_b: float | None) -> ModelSpec`
  - `classify(total_required_gb: float, free_vram_gb: float) -> str` → `"safe"|"warning"|"blocked"`
  - `estimate(hardware: dict, *, model_id: str, user_params_b: float | None = None, batch_size: int = DEFAULT_BATCH_SIZE, seq_length: int = MAX_SEQ_LENGTH_DEFAULT) -> EstimateResult`

- [ ] **Step 1: เขียน `tests/test_estimator.py`** (monkeypatch `core.estimator.hf_hub_download` เพื่อไม่ใช้ network):
  - `test_resolve_default_model`: `resolve_model_spec(DEFAULT_MODEL_ID, None)` → `num_params == 498_431_872`, `hidden_size == 896`, `num_layers == 24`, `source == "default"`
  - `test_resolve_from_config`: fetch คืน fixture `{"hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "vocab_size": 1000}` → `num_params == 145_920` (สูตร: `layers×(4d²+3·d·i) + vocab×d`), `source == "hf_config"`
  - `test_blocked_when_params_unknown`: fetch raise + `user_params_b=None` → `verdict == "blocked"`, `"พารามิเตอร์" in reason`
  - `test_user_fallback_params`: fetch raise + `user_params_b=1.5` → `spec_source == "user_fallback"`, `weights_gb ≈ 1.5e9×2/1024³`, ไม่ blocked
  - `test_classify_boundaries`: `free=20.0` (0.75×20=15.0, 0.90×20=18.0 exact) → `classify(15.0,20)=="safe"`, `classify(15.001,20)=="warning"`, `classify(18.0,20)=="warning"`, `classify(18.001,20)=="blocked"`
  - `test_insufficient_passthrough`: hw dict `status` = `insufficient_ram`/`insufficient_disk`/`no_xpu` → `verdict == status` ทันที (ไม่ compute)
  - `test_estimate_components_sum`: default model + `free_vram_gb=20.0` → `weights+trainable+activations+overhead ≈ total` (isclose) + `verdict=="safe"` + `weights_gb ≈ 498_431_872×2/1024³`

- [ ] **Step 2: เพิ่ม constant ใน `configs/safe_defaults.py`** (ตาม block "Estimator" ด้านบน)

- [ ] **Step 3: Implement `core/estimator.py`**
  - `resolve_model_spec`: (1) `model_id == DEFAULT_MODEL_ID` → ค่า hardcode ไม่แตะ network (2) else `from huggingface_hub import hf_hub_download` (import ระดับ module เพื่อให้ test monkeypatch ที่ `core.estimator.hf_hub_download` ได้) → `hf_hub_download(model_id, "config.json")` (ค่า default cache = แคชครั้งแรกตามสเปก) → parse → `num_params = layers×(4·d² + 3·d·i) + vocab×d` (สมมติ MHA — มากกว่าจริงเล็กน้อย = ทิศทางปลอดภัย), `source="hf_config"` (3) จับ exception → `user_params_b is None` → raise `ModelSpecUnavailable`; มีค่า → `ModelSpec(int(user_params_b×1e9), ESTIMATOR_FALLBACK_HIDDEN_SIZE, ESTIMATOR_FALLBACK_NUM_LAYERS, "user_fallback")` (dims สำรอง = upper bound อนุรักษ์นิยมสำหรับ activation)
  - `classify`: `total <= VRAM_SAFE_RATIO×free` → safe; `total <= VRAM_WARNING_RATIO×free` → warning; else blocked
  - `estimate`: (1) `hardware["status"] != "ready"` → คืน verdict=status, reason จาก mapping, component = 0.0 (2) `ModelSpecUnavailable` → blocked + reason "ไม่ทราบจำนวนพารามิเตอร์ — ต้องกรอก P (B) ใน UI (ห้ามเดา)" (3) สูตร: `weights = P×2`, `trainable = P_lora×10` bytes โดย `P_lora = P × (DEFAULT_LORA_NUM_PARAMS/DEFAULT_MODEL_NUM_PARAMS)`, `activations = batch_size×seq_length×d×N×2` bytes (gradient checkpointing เปิด), `overhead = ESTIMATOR_OVERHEAD_GB×1024³` (4) GB = `/1024³`, `classify` → verdict
  - reason ภาษาไทยสั้น ๆ ต่อ verdict (เช่น "ปลอดภัย — ใช้ ≤75% ของ VRAM ว่าง")

- [ ] **Step 4: Commit (ยังไม่รัน test)**

```bash
git add configs/safe_defaults.py core/estimator.py tests/test_estimator.py && git commit -m "feat: core/estimator.py ประเมิน VRAM + verdict thresholds + P fallback"
```

---

### Task 3: `core/dataset_builder.py` — FIM Processing

**Files:**
- Modify: `configs/safe_defaults.py` (เพิ่ม `MIN_SAMPLE_LINES: int = 3`)
- Create: `core/dataset_builder.py`
- Test: `tests/test_dataset_builder.py`

**Interfaces:**
- Consumes: `configs.fim_registry.json` (ค่า), `safe_defaults.FIM_RATE/SEED/MAX_SEQ_LENGTH_DEFAULT/MIN_SAMPLE_LINES/HELDOUT_RATIO`
- Produces:
  - `split_fim(code: str, rng: random.Random) -> tuple[str, str, str]` คืน `(prefix, suffix, middle)` — ตัดที่ line boundary (index หลัง `\n` เท่านั้น), ทั้ง 3 ส่วนมี ≥1 บรรทัด, `prefix + middle + suffix == code`
  - `format_psm(prefix: str, suffix: str, middle: str, *, fim_tokens: dict, eos: str) -> str` = `prefix_tok + prefix + suffix_tok + suffix + middle_tok + middle + eos`
  - `truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str`
  - `build_samples(codes: Iterable[str], *, fim_tokens: dict, eos: str, fim_rate: float = FIM_RATE, seed: int = SEED, min_lines: int = MIN_SAMPLE_LINES, tokenizer: Any | None = None, max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT) -> Iterator[str]`
    - `rng = random.Random(seed)` ตัวเดียวทั้ง generator (ผลซ้ำได้)
    - sample ที่มีบรรทัด < `min_lines` → ข้าม
    - `rng.random() < fim_rate` → FIM path (split+format), else plain path (code ดิบ)
    - มี `tokenizer` → truncate body เหลือ `max_seq_length − len(tokenizer.encode(eos))` tokens แล้วค่อยต่อ `eos` (รับประกัน EOS อยู่ท้าย + ไม่เกิน max)
  - `is_heldout(code: str, heldout_ratio: float = HELDOUT_RATIO) -> bool` — `md5(code)` stable hash < ratio (code เดียวกันตกข้างเดียวกันเสมอ = กัน leakage)

- [ ] **Step 1: เขียน `tests/test_dataset_builder.py`** (ไม่แตะ network; tokenizer จริงจาก cache ของ Qwen):
  - `test_format_psm_exact`: `format_psm("PRE\n", "SUF\n", "MID\n", fim_tokens=qwen, eos="</s>")` == `"<|fim_prefix|>" + "PRE\n" + "<|fim_suffix|>" + "SUF\n" + "<|fim_middle|>" + "MID\n" + "</s>"` (assert ทั้ง string + ลำดับ index prefix_tok < suffix_tok < middle_tok)
  - `test_split_line_boundary_and_reconstruct`: code ≥6 บรรทัด, rng seeds หลายค่า → ทุกครั้ง `prefix+middle+suffix == code`, ทุกส่วนไม่ว่าง, จุดตัดอยู่หลัง `\n` (`code[i-1] == "\n"` สำหรับ i = จบ prefix, จบ prefix+middle)
  - `test_eos_at_end`: `build_samples` ทุก output (`fim` และ `plain`) `endswith(eos)`
  - `test_seed_reproducible`: เรียกซ้ำด้วย seed เดิม → list _identical; `seed=99` → list ต่างอย่างน้อย 1 ชิ้น
  - `test_fim_rate_half`: 400 samples หลายบรรทัด (seed 42) → สัดส่วนที่มี `<|fim_prefix|>` อยู่ใน 0.5 ± 0.08
  - `test_short_sample_skipped`: code 2 บรรทัด → ไม่มีในผลลัพธ์เลย
  - `test_truncation_keeps_eos`: code ~200 บรรทัด + tokenizer Qwen จริง + `max_seq_length=1024` → `len(tokenizer.encode(out, add_special_tokens=False)) <= 1024` และ `out.endswith(eos)` (Review Focus #5)
  - `test_heldout_stable_and_ratio`: `is_heldout` คงที่ต่อ code เดิม; 1000 codes ที่สร้างเอง → สัดส่วน heldout อยู่ 0.05–0.15

- [ ] **Step 2: เพิ่ม `MIN_SAMPLE_LINES: int = 3` ใน `configs/safe_defaults.py`** + implement `core/dataset_builder.py` ตาม Interfaces
  - `split_fim`: หาร index ที่เป็นเส้นแบ่งบรรทัด `cuts = [i for i,ch in enumerate(code) if ch == "\n"]` ที่ i+1 < len → สุ่ม `i < j` จาก cuts (ต้องเหลือ ≥1 บรรทัดทุกส่วน) → `prefix=code[:i+1]`, `middle=code[i+1:j+1]`, `suffix=code[j+1:]`
  - plain path: `code + eos` (code ดิบ ไม่ตัด)

- [ ] **Step 3: Commit (ยังไม่รัน test)**

```bash
git add configs/safe_defaults.py core/dataset_builder.py tests/test_dataset_builder.py && git commit -m "feat: core/dataset_builder.py FIM PSM + line-boundary + seed 42 + heldout"
```

---

### Task 4: Final Verification (รัน test ครั้งเดียว) + รายงาน

**Files:** ไม่มีไฟล์ใหม่ — รันชุด test ทั้งหมดครั้งแรกและครั้งเดียวของเฟส

- [ ] **Step 1: รัน pytest ทั้ง suite ครั้งเดียว**

Run: `../../.venv/bin/python -m pytest tests/ -v`
Expected: PASS ทุก test (ชุดเดิม 7 ตัวของ Phase 1 + ชุดใหม่ ~20 ตัว)

- [ ] **Step 2: ถ้า FAIL → debug ตาม root cause (systematic-debugging) แล้ว rerun จนเขียว** (อนุญาตให้รันซ้ำเฉพาะช่วงนี้)

- [ ] **Step 3: Commit ที่ค้างอยู่ทั้ง 3 task** (ถ้า Step 1–2 ผ่าน)

Run: `git log --oneline` ยืนยัน commit ครบ + `git status` ว่าง

- [ ] **Step 4: รายงาน Phase 2 Gate ให้ human partner** — ผ่านทั้งหมด = พร้อม merge/ไปต่อ Phase 3
