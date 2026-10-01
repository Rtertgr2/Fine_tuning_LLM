# Phase 1: Runtime Baseline & Compat Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** สร้าง runtime baseline ที่ใช้การได้จริงบน Intel Arc B580 — venv Python 3.11 + torch XPU + ชุดไลบรารีที่ล็อกเวอร์ชันแล้ว + smoke test `SFTTrainer` ตัวจริง + configs ของโปรเจกต์

**Architecture:** ทำงานใน worktree `phase1-runtime-baseline` ทั้งหมด ติดตั้ง torch จาก XPU index ก่อนเป็นตัวแรก (กัน pip ดึง torch จาก PyPI) แล้วค่อยติดตั้งไลบรารีที่เหลือเพื่อสร้าง `constraints.txt` ทดสอบจริงด้วยสคริปต์ sanity 2 ตัว (`check_runtime.py` = XPU fwd/bwd, `smoke_sft.py` = SFTTrainer + LoRA + Qwen บน XPU) ส่วน configs เขียนแบบ test-first

**Tech Stack:** Python 3.11 venv, PyTorch XPU (download.pytorch.org/whl/xpu), transformers/peft/trl/datasets/accelerate, pytest

**Spec:** `plan.md` (§1 Runtime Baseline, §6 Phase 1, §7 requirements) — โต้แย้งทุกข้อจากสเปกนี้

## Global Constraints

- **Python: 3.11 เท่านั้น** (สเปก: 3.11 หรือ 3.12; เครื่องมี 3.11.16 — ห้ามใช้ 3.14 ของระบบ) — ทุกคำสั่งใช้ `.venv/bin/python` เท่านั้น
- **torch ติดตั้งจาก `--index-url https://download.pytorch.org/whl/xpu` เท่านั้น** — ห้ามมาจาก PyPI (สเปก: ห้าม `torch` ตัวมาตรฐาน)
- `torch ไม่อยู่ใน requirements.txt` — ติดตั้งแยกก่อนไลบรารีอื่นเสมอ (กัน pip resolve ดึง torch จาก PyPI)
- `trl >= 0.12.0` — เรียก SFTTrainer ด้วย `processing_class=` เท่านั้น (ห้าม `tokenizer=`)
- `numpy >= 2.0.0` — ต้องพิสูจน์ใน spike แล้วค่อยล็อกค่าจริงลง `constraints.txt`
- **ห้าม resize embedding / ห้าม add_special_tokens กับ FIM tokens** — Qwen2.5-Coder มีอยู่แล้ว
- ค่า configs ที่สเปก pin ไว้แล้ว (ห้ามเดา): `RAM_MIN_GB=16`, `DISK_MIN_GB=20`, `FIM_RATE=0.5`, รูปแบบ PSM, `SEED=42`, `max_seq_length` default 1024 / cap 2048, `DEFAULT_MODEL_ID="Qwen/Qwen2.5-Coder-0.5B"`, `DEFAULT_DATASET_ID="smangrul/hf-stack-v1"` คอลัมน์ `content`, held-out **10%**
- FIM tokens ของ qwen ใน `fim_registry.json` ต้องตรงตามสเปกเป๊ะ: `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`

## Review Focus

เงื่อนไขที่สเปกไม่ได้เขียน test กำกับโดยตรงแต่ทำให้คนใช้เจ็บได้ — ทุกบรรทัดมี test ใน task ที่รับผิดชอบ:

1. **รันด้วย Python 3.14 ของระบบแทน venv** → เทียบกับ torch XPU ไม่ได้ตั้งแต่ต้น — *เป็นเจ้าของโดย Task 1: assert `sys.version_info[:2] == (3,11)` จาก `.venv/bin/python`*
2. **torch ถูก pip ดึงจาก PyPI (ไม่มี XPU)** → ฟังก์ชัน xpu หาย — *เป็นเจ้าของโดย Task 1: assert `torch.xpu.is_available() is True`*
3. **API drift ของ trl (`tokenizer=` ถูกแทนที่)** → smoke test พังด้วย `TypeError` — *เป็นเจ้าของโดย Task 4: `smoke_sft.py` เรียกด้วย `processing_class=` และรันผ่าน 2 steps จริง*
4. **numpy รุ่นผิด (1.x หรือ 2.x ที่ชน dependency)** → runtime error ตอน import torch — *เป็นเจ้าของโดย Task 2: assert major version == 2 หลังติดตั้ง*
5. **FIM tokens หาย/UNK** → เทรนแล้วโมเดลเรียนรู้ token ผิด ๆ — *เป็นเจ้าของโดย Task 4: assert `convert_tokens_to_ids` ไม่คืนค่า UNK และ decode กลับได้ตรง*

---

### Task 1: Python 3.11 venv + torch XPU + โครงสร้างโปรเจกต์

**Files:**
- Create: `.venv/` (gitignored, จาก `python3.11`)
- Modify: `.gitignore` (เพิ่มถ้ายังไม่มี)
- Create: `requirements.txt`

**Interfaces:**
- Consumes: —
- Produces: `.venv/bin/python` (3.11) ที่ task ถัดไปใช้ทั้งหมด, `requirements.txt` (list deps นอก torch)

- [ ] **Step 1: เขียน plan นี้ไว้ใน repo และ commit**

Run: `git add docs/ && git commit -m "docs: phase 1 implementation plan"`
Expected: commit สำเร็จ

- [ ] **Step 2: สร้าง venv และยืนยันว่าเป็น Python 3.11**

Run: `python3.11 -m venv .venv && .venv/bin/python -c "import sys; assert sys.version_info[:2]==(3,11), sys.version; print(sys.version)"`
Expected: พิมพ์ `3.11.x` (ไม่ใช่ 3.14)

- [ ] **Step 3: ติดตั้ง torch จาก XPU index (ตัวแรกเสมอ)**

Run: `.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/xpu`
Expected: ติดตั้งสำเร็จ (ไฟล์ wheel ลงท้ายด้วย xpu หรือมาจาก whl/xpu index)

- [ ] **Step 4: ยืนยัน XPU ทำงานจริงบน Arc B580**

Run: `.venv/bin/python -c "import torch; print(torch.__version__); assert torch.xpu.is_available(), 'NO XPU'; print(torch.xpu.get_device_name(0))"`
Expected: พิมพ์เวอร์ชัน + ชื่อการ์ด (มี `Arc` หรือ `B580`) — FAIL = หยุด รายงาน ห้ามไปต่อ

- [ ] **Step 5: เขียน `requirements.txt` ตามสเปก §7 (ยังไม่ต้อง install)**

เนื้อหา: คัดลอกบล็อก code จาก `plan.md` §7 ทั้งหมด (transformers/peft/trl/datasets/accelerate/gradio/psutil/numpy/pandas/pytest พร้อม comment) — ไม่ใส่ torch

- [ ] **Step 6: Commit**

```bash
git add requirements.txt docs/ .gitignore
git commit -m "chore: python3.11 venv + torch XPU baseline + requirements"
```

---

### Task 2: ติดตั้งไลบรารีที่เหลือ + สร้าง constraints.txt

**Files:**
- Create: `constraints.txt`

**Interfaces:**
- Consumes: `.venv/bin/python` + torch (Task 1), `requirements.txt` (Task 1)
- Produces: `constraints.txt` — ไฟล์ล็อกเวอร์ชีนที่ task 4+ อ้างอิง (`pip install -c constraints.txt`)

- [ ] **Step 1: ติดตั้ง requirements ทั้งหมด (torch มีอยู่แล้ว)**

Run: `.venv/bin/pip install -r requirements.txt`
Expected: สำเร็จโดยไม่มีการแก้ torch (ตรวจ: `.venv/bin/pip show torch` ยังเป็นรุ่นจาก XPU index เหมือนเดิม)

- [ ] **Step 2: พิสูจน์ numpy major version == 2**

Run: `.venv/bin/python -c "import numpy; assert numpy.__version__.split('.')[0]=='2', numpy.__version__; print(numpy.__version__)"`
Expected: พิมพ์ `2.x.y` — ถ้า pip resolve ได้ 1.x ให้แก้ constraint ใน `requirements.txt` แล้ว re-run Step 1

- [ ] **Step 3: สร้าง constraints.txt จากของจริง**

Run: `.venv/bin/pip freeze | grep -E '^(torch|transformers|peft|trl|datasets|accelerate|gradio|psutil|numpy|pandas|pytest)==' > constraints.txt && cat constraints.txt`
Expected: มีครบทุกตัว (อย่างน้อย 11 บรรทัด) — ตรวจด้วยตาว่า `trl` ≥ 0.12 และ `torch` ขึ้นต้นด้วยเวอร์ชันจาก xpu index

- [ ] **Step 4: Commit**

```bash
git add constraints.txt requirements.txt
git commit -m "chore: lock verified dependency versions in constraints.txt"
```

---

### Task 3: `scripts/check_runtime.py` — สุขภาพ XPU แบบ end-to-end

**Files:**
- Create: `scripts/check_runtime.py`

**Interfaces:**
- Consumes: `.venv/bin/python` + torch XPU (Task 1)
- Produces: `main() -> int` (exit 0 = ผ่าน) — ใช้ด้วยมือและเป็นเกณฑ์ผ่าน Phase 1

- [ ] **Step 1: เขียนสคริปต์**

`main()` ทำตามลำดับ: (1) พิมพ์ `torch.__version__` + `torch.xpu.is_available()` + `get_device_name(0)` (2) พิมพ์ total/free VRAM จาก `torch.xpu.mem_get_info(0)` (3) sanity ดิสก์และ RAM ผ่าน `psutil` (ค่า threshold จาก `configs.safe_defaults` — ถ้ายังไม่มีไฟล์ ให้ใช้ค่า 16/20 inline ชั่วคราว) (4) ** fwd/bwd บน XPU**: สร้าง `torch.randn(64, 64, device="xpu", requires_grad=True)`, ทำ matmul กับ weight แล้วคำนวณ MSE กับ target, `backward()` แล้ว assert ว่า grad มีค่า finite และมีอย่างน้อย 1 ค่าที่ไม่เป็นศูนย์ (5) return 0 ถ้าผ่านทุกข้อ, raise/return 1 ถ้าข้อใดข้อหนึ่งตก

- [ ] **Step 2: รันและตรวจสอบ**

Run: `.venv/bin/python scripts/check_runtime.py`
Expected: พิมพ์ทุกบรรทัด + exit 0

- [ ] **Step 3: ตรวจ clinfo (Intel Compute Runtime)**

Run: `clinfo 2>/dev/null | grep -iE 'device name|intel' | head -5 || echo "clinfo ไม่พบอุปกรณ์"`
Expected: เห็นอุปกรณ์ Intel อย่างน้อย 1 ตัว (ถ้า clinfo ไม่ผ่านแต่ torch XPU ผ่าน = ผ่านต่อ แต่ให้จดบันทึกลง commit message)

- [ ] **Step 4: Commit**

```bash
git add scripts/check_runtime.py
git commit -m "feat: XPU runtime sanity check (fwd/bwd, VRAM, RAM, disk)"
```

---

### Task 4: `scripts/smoke_sft.py` — SFTTrainer + LoRA บน XPU ตัวจริง

**Files:**
- Create: `scripts/smoke_sft.py`

**Interfaces:**
- Consumes: dependencies + `constraints.txt` (Task 2), FIM token rules จากสเปก (Global Constraints)
- Produces: exit 0 เมื่อ trainer รันผ่าน 2 steps บน XPU โดย loss finite — เป็นเกณฑ์ผ่าน Compat Spike

- [ ] **Step 1: เขียนสคริปต์**

เนื้อหา: (1) โหลด tokenizer ของ `Qwen/Qwen2.5-Coder-0.5B` แล้ว **assert FIM tokens ไม่ใช่ UNK** (`convert_tokens_to_ids("<|fim_prefix|>") != unk_token_id` และ decode กลับได้) — ห้ามเรียก `add_special_tokens` เด็ดขาด (2) โหลด model ด้วย `torch_dtype=torch.bfloat16`, `attn_implementation="sdpa"` บน XPU (3) สร้าง LoRA ด้วย `peft` (`r=8`, target modules ของ qwen2 ตามที่ `get_peft_model` แนะนำ) (4) สร้าง dataset เล็ก ๆ จากข้อความโค้ดสังเคราะห์ ~8 ตัวอย่าง (ไม่ต้องพึ่ง HF dataset ใน spike นี้) (5) เรียก `SFTTrainer` + `SFTConfig` **ด้วย `processing_class=tokenizer`** (6) `trainer.train(max_steps=2)` (7) assert loss ของ 2 steps เป็น finite แล้ว return 0

- [ ] **Step 2: รัน (ครั้งแรก — คาดว่าต้องแก้)**

Run: `.venv/bin/python scripts/smoke_sft.py`
Expected: ถ้า `TypeError ... tokenizer` → แปลว่าติดตั้ง trl < 0.12 = แก้ที่ `constraints.txt`/`requirements.txt` แล้ว re-install ถ้าเป็นอย่างอื่น ให้ debug จนรันผ่าน — **ห้าม install torch ใหม่ระหว่างทาง**

- [ ] **Step 3: รันซ้ำจน clean + ตรวจว่า loss finite**

Run: `.venv/bin/python scripts/smoke_sft.py && echo PASS`
Expected: `PASS`, ไม่มี warning ประเภท `torch.xpu` ไม่รู้จัก / CPU fallback

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_sft.py constraints.txt requirements.txt
git commit -m "feat: SFTTrainer+LoRA XPU smoke test (processing_class API, FIM token guard)"
```

---

### Task 5: `configs/` แบบ Test-First (fim_registry + safe_defaults)

**Files:**
- Test: `tests/test_fim_registry.py`, `tests/test_safe_defaults.py`
- Create: `configs/fim_registry.json`, `configs/safe_defaults.py`, `configs/__init__.py` ถ้าจำเป็น, `tests/__init__.py` ถ้าจำเป็น

**Interfaces:**
- Consumes: —
- Produces (task ถัดไปในเฟสถัดไปจะ import):
  - `configs.safe_defaults`: `RAM_MIN_GB: int = 16`, `DISK_MIN_GB: int = 20`, `VRAM_SAFE_RATIO: float = 0.75`, `VRAM_WARNING_RATIO: float = 0.90`, `FIM_RATE: float = 0.5`, `MAX_SEQ_LENGTH_DEFAULT: int = 1024`, `MAX_SEQ_LENGTH_CAP: int = 2048`, `SEED: int = 42`, `DEFAULT_MODEL_ID: str`, `DEFAULT_DATASET_ID: str`, `DEFAULT_DATASET_COLUMN: str = "content"`, `HELDOUT_RATIO: float = 0.10`
  - `fim_registry.json`: keys `qwen` / `starcoder` / `deepseek` — แต่ละตัวมี `prefix`/`suffix`/`middle` ตรงตามสเปกเป๊ะ

- [ ] **Step 1: เขียน test ที่ fail**

`tests/test_fim_registry.py`: assert โหลด JSON ได้, มี key ครบ 3 ค่าย, ค่าของ qwen == `{"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"}`, และ `starcoder`/`deepseek` มีทั้ง 3 key (ค่าตรงตามสเปก §4.3)
`tests/test_safe_defaults.py`: assert ค่าคงที่ทุกตัวใน Interfaces = ค่าข้างบน และ `RAM_MIN_GB == 16`, `DISK_MIN_GB == 20`, `HELDOUT_RATIO` อยู่ในช่วง 0.05–0.10

- [ ] **Step 2: รันให้เห็น FAIL**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: FAIL ด้วย `FileNotFoundError` / `ImportError` (ไฟล์ยังไม่มี)

- [ ] **Step 3: สร้างไฟล์ configs**

สร้าง `configs/fim_registry.json` (ค่าคัดลอกจากสเปก §4.3 ทั้งบล็อก) และ `configs/safe_defaults.py` (ค่าคงที่ตาม Interfaces)

- [ ] **Step 4: รันให้เห็น PASS**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS ทุก test

- [ ] **Step 5: Commit**

```bash
git add tests/ configs/
git commit -m "feat: FIM registry + safe_defaults with pinning tests"
```

---

### Task 6: ตรวจจบเฟส (Phase 1 Gate)

**Files:** ไม่มีไฟล์ใหม่ — เป็น checklist ยืนยัน

- [ ] **Step 1: รันสคริปต์ทั้งหมดซ้ำอีกรอบ**

Run: `.venv/bin/python scripts/check_runtime.py && .venv/bin/python scripts/smoke_sft.py && .venv/bin/python -m pytest tests/ -q`
Expected: ทั้งสามผ่าน (exit 0)

- [ ] **Step 2: ยืนยันว่า constraints ตรงของจริง**

Run: `.venv/bin/pip check && diff <(.venv/bin/pip freeze | grep -E '^(torch|transformers|peft|trl|datasets|accelerate|gradio|psutil|numpy|pandas|pytest)==') constraints.txt && echo SYNCED`
Expected: `SYNCED` (ไม่มี dependency conflict, ไม่มี version drift)

- [ ] **Step 3: Commit/Report**

```bash
git log --oneline   # ทุก commit ของเฟสอยู่ครบบน phase1-runtime-baseline
```
Expected: รายงานผล Phase 1 Gate ให้ human partner — ถ้าผ่านทั้งหมด = Phase 1 เสร็จ พร้อม merge/ไปต่อ Phase 2
