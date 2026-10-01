# Fine-tuning-LLM — Fine-tune โมเดลโค้ดบน Intel GPU

โปรเจค fine-tune **Qwen/Qwen2.5-Coder-0.5B** ด้วย **LoRA + Fill-in-the-Middle (FIM)** บนกราฟิกการ์ด
**Intel Arc (XPU)** ผ่าน `torch 2.14+xpu` — ทั้ง training pipeline รันใน subprocess แยก
พร้อม protocol สื่อสาร UI (`metric`/`log`/`status`/`error`), watchdog จับ zombie process,
และ abort ที่การันตีคืน VRAM จริง

## สถานะ

| Phase | เนื้อหา | สถานะ |
|---|---|---|
| 1 | Runtime baseline — XPU + transformers/trl/peft compat | ✅ |
| 2 | Hardware gate + Estimator + Dataset builder (FIM) | ✅ |
| 3 | Subprocess training pipeline + IPC | ✅ |
| 4 | GUI (Gradio) | ⬜ ตามแผน |
| 5 | Save Adapter / Export | ⬜ ตามแผน |

## ความต้องการของระบบ

- Linux (ทดสอบบน EndeavourOS + Arc B580)
- Intel GPU ที่ซัพพอร์ต level-zero — ติดตั้ง `level-zero-loader` + `intel-compute-runtime`
- Python **3.11** (3.12+ ยังไม่รับรอง)
- RAM ≥ 16 GB, พื้นที่ว่าง ≥ 20 GB

## ติดตั้ง

```bash
python3.11 -m venv .venv

# torch ลงผ่าน XPU index แยก (ไม่อยู่ใน requirements.txt)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/xpu

# ที่เหลือล็อกด้วย constraints (เวอร์ชันที่ทดสอบแล้วทั้งหมด)
.venv/bin/pip install -r requirements.txt -c constraints.txt
```

## คำสั่งรัน

```bash
# 1. ตรวจ hardware + runtime ก่อนเสมอ (ผ่าน = ทุก check สีเขียว)
.venv/bin/python scripts/check_runtime.py

# 2. compat spike — SFT + LoRA บน XPU จริง 2 steps
.venv/bin/python scripts/smoke_sft.py

# 3. integration ของ training pipeline ทั้งวงจร
#    full (เทรน 6 steps จริง) + abort (SIGTERM→SIGKILL) + ตรวจ VRAM คืน
.venv/bin/python scripts/pipeline_smoke.py --mode all

# test ทั้งหมด (53 tests)
.venv/bin/python -m pytest tests/ -v
```

## โครงสร้างโปรเจค

```
core/
  hardware.py        # ตรวจ RAM/VRAM/Disk/Driver ก่อนอนุญาตให้เทรน
  estimator.py       # ประมาณการ VRAM ก่อนเริ่ม (static + fallback จาก HF config)
  dataset_builder.py # แปลงโค้ดดิบเป็น FIM (PSM, line boundary, seed 42)
  ipc_bridge.py      # message protocol 4 ประเภท + watchdog + abort escalation
  trainer_worker.py  # build_training_args, guardrails, callback, run_training
configs/
  safe_defaults.py   # ค่าคงที่ของระบบทั้งหมด (pin ด้วย test)
  fim_registry.json  # FIM tokens ต่อ family (qwen/starcoder/deepseek)
scripts/             # check_runtime, smoke_sft, pipeline_smoke
tests/               # pytest 53 ตัว
docs/                # แผน implementation ราย phase
```

## กติกาสำคัญ

- **ค่าคงที่ทุกอย่างอยู่ที่ `configs/safe_defaults.py`** — ห้าม hardcode ซ้ำในโค้ดส่วนอื่น
- Hard cap: batch 1, seq ≤ 2048, lr 2e-4, grad_accum 8, seed 42, max_steps 500
- FIM: rate 0.5, รูปแบบ **PSM เท่านั้น**, ตัดที่ line boundary เท่านั้น, EOS ท้ายทุก sample
- สื่อสาร UI ผ่าน `multiprocessing.Queue` — 4 message types: `metric`/`log`/`status`/`error`,
  status เป็น state machine: `starting → training → saving → finished | aborted`
- Abort: SIGTERM → รอ 10 วิ → SIGKILL (คืน VRAM); loss non-finite 3 ครั้งติด → abort เอง
- Checkpoint เขียนลง `*.saving` ก่อน rename เข้าที่ (atomic) — กันไฟล์เสียถ้าโดน kill กลาง save
