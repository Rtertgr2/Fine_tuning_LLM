# Fine-tuning-LLM — Fine-tune โมเดลโค้ดบน Intel GPU

โปรเจค fine-tune **Qwen/Qwen2.5-Coder-0.5B** ด้วย **LoRA + Fill-in-the-Middle (FIM)** บนกราฟิกการ์ด
**Intel Arc (XPU)** ผ่าน `torch 2.14+xpu` — training pipeline รันใน subprocess แยก
พร้อม protocol สื่อสาร UI (`metric`/`log`/`status`/`error`), watchdog จับ zombie process,
abort ที่การันตีคืน VRAM จริง, **evaluator เทียบ base vs fine-tuned** และ Gradio dashboard 3 แท็บ

## สถานะ

| Phase | เนื้อหา | สถานะ |
|---|---|---|
| 1 | Runtime baseline — XPU + transformers/trl/peft compat | ✅ |
| 2 | Hardware gate + Estimator + Dataset builder (FIM) | ✅ |
| 3 | Subprocess training pipeline + IPC | ✅ |
| 4 | GUI (Gradio) — Configuration, Mission Control, Playground & Export | ✅ |
| 5 | Evaluator (Exact Match + Token F1) + `eval.py` + guardrail tests + docs | ✅ |

> **ผล eval จริง** (500 steps, heldout-clean, 100 ตัวอย่าง, §8.2 PASS):
>
> | Metric | Base | Fine-tuned | Δ |
> |---|---|---|---|
> | Exact Match % | 1.0 | 2.0 | **+1.0** |
> | Token F1 | 0.248 | 0.338 | **+0.09** |
>
> ตัวอย่างที่ base ผิด → fine-tuned ถูก: header ใบอนุญาต (F1 0.28→1.00, exact fix),
> `_import_structure` ของ transformers (F1 0.30→1.00) และเคสที่ base เพี้ยนเป็น `!!!...`
> แต่ fine-tuned เขียน `from ...modeling_tf_outputs import ...` ได้จริง (F1 0→0.68) — ไม่มี regression เลย (0 case)

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

# ที่เหลือล็อกด้วย constraints (เวอร์ชันที่ทดสอบแล้วทั้งหมด — requirements pin == ตรง constraints แล้ว)
.venv/bin/pip install -r requirements.txt -c constraints.txt
```

## เริ่มใช้งาน

```bash
# ตรวจ hardware + runtime ก่อนเสมอ (ผ่าน = ทุก check สีเขียว)
.venv/bin/python scripts/check_runtime.py

# เปิด dashboard (bind 127.0.0.1:7860 เท่านั้น)
.venv/bin/python app.py
```

Flow บน UI:

1. **Configuration & Pre-flight** — ตั้ง model/dataset/LoRA/seq/steps แล้วกด **Run Environment Check**
   (คำนวณ VRAM ก่อน — ผ่าน = verdict `safe`/`warning`)
2. **Training Mission Control** — กด **Start Fine-Tuning** → ดู loss/LR + live log; 途中กด **Abort** ได้
   (SIGTERM → 10s → SIGKILL, การันตีคืน VRAM)
3. **Playground & Export** — ทดสอบ **Predict Middle** ด้วย checkpoint ล่าสุด,
   กด **Run Evaluation** เทียบ base vs fine-tuned บน held-out set,
   แล้ว **Save Adapter Only** หรือ **Merge & Export Full Weights**

## โมเดล/ดาต้าเซ็ตในเครื่อง (dropdown detect)

ช่อง **Model** และ **Dataset** เป็น dropdown ที่ detect โฟลเดอร์ในโปรเจกต์ให้อัตโนมัติ
(refresh เองทุกครั้งที่สลับมาแท็บ Configuration) — พิมพ์ค่าใหม่เองก็ได้:

| โฟลเดอร์ | เนื้อหา | ตัวอย่างค่าใน dropdown |
|---|---|---|
| `models/` | โฟลเดอร์โมเดล (มี `config.json` + น้ำหนัก) | วาง `models/My-Model` → เลือก `My-Model` |
| `datasets/` | โฟลเดอร์ดาต้าเซ็ต (ไฟล์ `.parquet`) | วาง `datasets/my-ds` → เลือก `my-ds` |

- **ดาต้าเซ็ตท้องถิ่น**: อ่าน `.parquet` ตรง ๆ แบบ streaming ทีละ batch — **รองรับ dataset ใหญ่ ~20GB**
  (ไม่ convert เป็น arrow cache ซ้ำ กินดิสก์แค่ไฟล์ต้นฉบับ, ไม่โหลดทั้งไฟล์ลง RAM,
  เทรนใช้ ~8,000 แถวแรกเหมือนเดิม — ขนาด dataset ไม่กระทบเวลาเทรน/VRAM)
- **โมเดล Hub** (เช่น `Qwen/Qwen2.5-Coder-0.5B`): คงใช้ HF cache เดิม ไม่ copy เข้าโปรเจกต์ —
  เลือกแล้วใช้ได้ทันทีถ้าเคยโหลด; โมเดลรุ่นใหม่พิมพ์ model id ลง dropdown แล้วระบบดึง config จาก Hub เอง
- **Estimator** อ่าน `config.json` จากโฟลเดอร์ท้องถิ่นได้โดยตรง (นอกเหนือจาก Hub)
- **LoRA** target modules ถูก filter ตามชื่อเลเยอร์จริงของโมเดล (รองรับ family อื่นที่ชื่อเลเยอร์ต่างกัน)
  — ไม่ตรงสักตัว → error อังกฤษบอกให้ตรวจ family ของโมเดล
- ไฟล์ผิดรูปแบบ (โฟลเดอร์ไม่มี `.parquet` / คอลัมน์ไม่ตรง / config ไม่ครบ) → error อังกฤษบอกวิธีแก้

> เหตุผลที่ขัด plan.md §4.4 (สั่ง streaming): dataset ตัวอย่างเล็กแค่ ~30MB และการอ่าน parquet
> ตรงทำให้ใช้ออฟไลน์ได้ทันที + รองรับชุดใหญ่โดยไม่กินดิสก์ซ้ำ — semantics "แถวแรก `code_limit`
> ตัวอย่าง" และ order/holdout เดิมทั้งหมดยังคงเดิม

## Eval (spec §3.3)

Evaluator วัด **FIM Exact Match** + **Token F1** บนชุด held-out ที่แยกด้วย md5 คงที่
(`is_heldout` — code เดียวกันตกข้างเดียวกันเสมอ, train ไม่มีทางปนชุด eval):

```bash
# รัน eval บนโมเดล base → data_cache/eval/base.json
.venv/bin/python eval.py --mode base

# รัน eval บน base+LoRA (checkpoint ล่าสุด) → data_cache/eval/finetuned.json
.venv/bin/python eval.py --mode finetuned

# เทียบผล — fine-tuned ต้องไม่แย่กว่า base ทุก metric (exit 0 = PASS, 1 = FAIL, 2 = ข้อมูลไม่ครบ)
.venv/bin/python eval.py --compare
```

บน UI: ปุ่ม **Run Evaluation** ใน Tab 3 รันทั้ง 2 mode แล้วเทียบให้ในตาราง
(ล็อกปุ่มระหว่างเทรน — กัน stack 2 process โหลดโมเดลพร้อมกัน, §5 VRAM Contention)

## Test

```bash
# unit suite (เร็ว — integration ถูก deselect ด้วย addopts)
.venv/bin/python -m pytest tests/ -v

# integration (abort + disk gate + cache growth — ใช้ XPU จริง ~1 นาที)
.venv/bin/python -m pytest tests/ -m integration -v
```

## Compression (GGUF / llama.cpp)

ย่อโมเดลเป็น GGUF รันบน Arc B580 ผ่าน llama.cpp (Vulkan) — standalone CLI ไม่แตะ Gradio app
(spec: `docs/superpowers/specs/2026-10-03-phase6a-gguf-compression-design.md`,
roadmap: `docs/roadmaps/2026-10-03-phase6-llm-optimization-roadmap.md` §6)

```bash
# 1) build llama.cpp (pin commit + Vulkan) — ครั้งแรก ~5-10 นาที
bash scripts/setup_llamacpp.sh

# 2) สร้าง artifacts + benchmark + eval ในคำสั่งเดียว (fp16 → q8_0 → q4_k_m)
#    (โมเดลยังไม่มีในเครื่อง → error จะบอกคำสั่ง `hf download ... --local-dir models/<name>` ให้)
.venv/bin/python scripts/benchmark_compression.py \
  --model Qwen/Qwen2.5-Coder-0.5B --variants fp16,q8_0,q4_k_m

# ตัวเลือกที่ใช้บ่อย
--no-eval                     # ข้าม llama-server eval (metrics = null ซื่อสัตย์)
--device cpu                  # บังคับ CPU (ค่าเริ่ม: vulkan — มีปัญหาเฉพาะ B580 ใช้ fallback นี้)
--eval-cases 10               # จำนวนเคส eval (ค่าเริ่ม: 100)
--baseline <path>             # baseline เทียบ delta ในตาราง (ค่าเริ่ม: auto-pick data_cache/eval/)

# 3) เสิร์ฟโมเดลบน local URL (ต่อเครื่องมือข้างนอกได้; Ctrl+C/SIGTERM = หยุดจริง ไม่ leak VRAM)
.venv/bin/python scripts/serve.py --model Qwen/Qwen2.5-Coder-0.5B --variant q4_k_m
# → Server ready: http://127.0.0.1:8080 (model: Qwen2.5-Coder-0.5B-q4_k_m)
```

- **Artifacts**: `<source>/gguf/<name>-<variant>.gguf` — มีอยู่แล้ว reuse ข้ามรอบ (ไม่แปลงซ้ำ)
- **Reports**: `benchmarks/<model>/<variant>.json` — schema §4 ของ roadmap + ตารางสรุปพิมพ์ใน terminal
- **Baseline**: โมเดล base → `data_cache/eval/base.json`, source ใต้ `exports/` (merge LoRA) → `finetuned.json`
  — ตารางพิมพ์ delta (`+x.xx`) ใต้ตัวเลข EM/F1 เทียบ Phase 5
- **Honest null**: วัดไม่ได้เป็น `null` ไม่ใช่ 0 — `peak_vram_mb` (เครื่องนี้ไม่มี `xpu-smi`),
  `load_time_ms` (llama-bench รุ่น pin ไม่ emit ค่านี้), `syntax_pass_rate`/`execution_pass_rate`
  (ยังไม่มี sandbox phase — เป็น `null` เสมอ)
- **Scope / กติกา**: report-only — ยังไม่ตั้ง quality threshold; runtime เป้า = llama.cpp + GGUF เท่านั้น —
  **GPTQ / AWQ / bitsandbytes / OpenVINO / HQQ ยังไม่อยู่ใน scope** (roadmap §6.1 main track)

## โครงสร้างโปรเจค

```
core/
  hardware.py        # ตรวจ RAM/VRAM/Disk/Driver ก่อนอนุญาตให้เทรน
  estimator.py       # ประมาณการ VRAM ก่อนเริ่ม (static + fallback จาก HF config)
  dataset_builder.py # แปลงโค้ดดิบเป็น FIM (PSM, line boundary, seed 42) + heldout split
  ipc_bridge.py      # message protocol 4 ประเภท + watchdog + abort escalation
  trainer_worker.py  # build_training_args, guardrails, callback, run_training, eval worker
  evaluator.py       # build_eval_cases, Exact Match + Token F1, run_eval, compare_results
configs/
  safe_defaults.py   # ค่าคงที่ของระบบทั้งหมด (pin ด้วย test)
  fim_registry.json  # FIM tokens ต่อ family (qwen/starcoder/deepseek)
ui/
  controller.py      # TrainingController — start/abort/tick/run_predict/run_eval (ไม่ import gradio)
  dashboard.py       # Gradio Blocks 3 แท็บ + event wiring
scripts/             # check_runtime, smoke_sft, pipeline_smoke
tests/               # pytest (unit + integration marker)
eval.py              # CLI eval — --mode base|finetuned, --compare
app.py               # entry point (spawn method + bind 127.0.0.1:7860)
docs/                # spec + implementation plan ราย phase
```

## กติกาสำคัญ

- **ค่าคงที่ทุกอย่างอยู่ที่ `configs/safe_defaults.py`** — ห้าม hardcode ซ้ำในโค้ดส่วนอื่น
- Hard cap: batch 1, seq ≤ 2048, lr 2e-4, grad_accum 8, seed 42, max_steps 500
- FIM: rate 0.5, รูปแบบ **PSM เท่านั้น**, ตัดที่ line boundary เท่านั้น, EOS ท้ายทุก sample
- **Train ต้องกรอง held-out เสมอ** (`filter_train_codes` ใน `run_training`) — กัน leakage ใส่ชุด eval
- สื่อสาร UI ผ่าน `multiprocessing.Queue` — 4 message types: `metric`/`log`/`status`/`error`,
  status เป็น state machine: `starting → training → saving → finished | aborted`
- Abort: SIGTERM → รอ 10 วิ → SIGKILL (คืน VRAM); loss non-finite 3 ครั้งติด → abort เอง
- Checkpoint เขียนลง `*.saving` ก่อน rename เข้าที่ (atomic) — กันไฟล์เสียถ้าโดน kill กลาง save
- **VRAM จำกัด 1 การ์ด** — ห้าม spawn predict/eval/merge/training พร้อมกัน
  (ปุ่มถูกล็อกโดย tick ระหว่างเทรน + `concurrency_id="model_load"` ที่ Gradio)

## Troubleshooting

| อาการ | สาเหตุ/วิธีแก้ |
|---|---|
| `no_xpu` ตอน Environment Check | ติดตั้ง level-zero-loader + intel-compute-runtime, login ใหม่ |
| `insufficient_disk` / `insufficient_ram` | คืนพื้นที่ ≥ 20 GB / RAM ≥ 16 GB (ค่ากำหนดใน `safe_defaults`) |
| `Cannot re-initialize XPU in forked subprocess` | ห้าม fork หลังแตะ XPU — ใช้ `spawn` เท่านั้น (`app.py` ตั้งให้ที่เดียว) |
| eval บอก `No checkpoint found ... train first` | ยังไม่เคยเทรนใน `output_dir` นี้ — เทรนก่อนแล้วรัน `--mode finetuned` |
| `Cannot compare: eval sets differ` | รันทั้ง 2 mode ด้วย `--n-cases` เดียวกัน |
