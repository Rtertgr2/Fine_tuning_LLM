# Fine-tuning-LLM — ระบบ Fine-tune โมเดลโค้ดบน Intel GPU (Arc / XPU)

โปรเจกต์พัฒนาและปรับแต่งโมเดลภาษาขนาดเล็กสำหรับงานเติมโค้ดเฉพาะทาง (**Code Completion** และ **Fill-in-the-Middle: FIM**) โดยใช้โมเดลตั้งต้น **`Qwen/Qwen2.5-Coder-0.5B`** ปรับจูนด้วยเทคนิค **LoRA (Low-Rank Adaptation)** บนความละเอียด **Native Bfloat16 (BF16)** ออกแบบมาโดยเฉพาะสำหรับทำงานบนกราฟิกการ์ด **Intel Arc (XPU)** ผ่าน `torch 2.14+xpu` ร่วมกับ Intel Compute Runtime (Level Zero)

ระบบถูกออกแบบด้วยสถาปัตยกรรม **Single-Language Stack (Python ทั้งระบบ)** ที่เน้นความเสถียรและความปลอดภัยสูงสุด: แยกกระบวนการฝึกสอน (Training), การประเมินผล (Evaluation) และการส่งออกโมเดล (Export) ออกจาก Web UI ด้วย **Subprocess** สื่อสารผ่าน Inter-Process Communication (IPC Queue) มีระบบตรวจจับและประมาณการ VRAM ก่อนเทรนจริง (Pre-flight Estimator), Watchdog ควบคุมสถานะ, ระบบบันทึก Checkpoint แบบ Atomic, และปุ่ม Abort ที่การันตีการคืนหน่วยความจำ VRAM 100%

---

## สารบัญ

1. [สถานะการพัฒนาและผลการประเมิน](#-สถานะการพัฒนาและผลการประเมิน)
2. [สถาปัตยกรรมและจุดเด่นของระบบ](#-สถาปัตยกรรมและจุดเด่นของระบบ)
3. [ความต้องการของระบบ](#-ความต้องการของระบบ)
4. [ขั้นตอนการติดตั้ง](#-ขั้นตอนการติดตั้ง)
5. [คู่มือการเริ่มใช้งาน](#-คู่มือการเริ่มใช้งาน)
6. [การจัดการโมเดลและชุดข้อมูลในเครื่อง](#-การจัดการโมเดลและชุดข้อมูลในเครื่อง)
7. [การประเมินผลโมเดล (Evaluation CLI)](#-การประเมินผลโมเดล-evaluation-cli)
8. [การบีบอัดและให้บริการโมเดล (GGUF / llama.cpp)](#-การบีบอัดและให้บริการโมเดล-gguf--llamacpp)
9. [โครงสร้างไดเรกทอรีและสถาปัตยกรรมโค้ด](#-โครงสร้างไดเรกทอรีและสถาปัตยกรรมโค้ด)
10. [กติกาและหลักการออกแบบความปลอดภัย](#-กติกาและหลักการออกแบบความปลอดภัย)
11. [การทดสอบระบบ (Testing)](#-การทดสอบระบบ-testing)
12. [การแก้ไขปัญหาที่พบบ่อย (Troubleshooting)](#-การแก้ไขปัญหาที่พบบ่อย-troubleshooting)

---

## สถานะการพัฒนาและผลการประเมิน

| เฟส (Phase) | รายละเอียดงาน | สถานะ |
|---|---|:---:|
| **Phase 1** | Runtime baseline — ตรวจสอบความเข้ากันได้ของ XPU กับ PyTorch, Transformers, TRL และ PEFT | ✅ สำเร็จ |
| **Phase 2** | Hardware gate + VRAM Estimator + Dataset builder (FIM Processing) | ✅ สำเร็จ |
| **Phase 3** | Subprocess training pipeline + IPC Protocol + Watchdog & Abort escalation | ✅ สำเร็จ |
| **Phase 4** | Web GUI (Gradio) — หน้า Configuration, Mission Control, Playground & Model Export | ✅ สำเร็จ |
| **Phase 5** | Evaluator (Exact Match + Token F1) + เครื่องมือ CLI `eval.py` + Guardrail tests | ✅ สำเร็จ |
| **Phase 6a** | Model Compression — แปลงเป็น GGUF, วัด Benchmark และเสิร์ฟผ่าน `llama.cpp` (Vulkan) | ✅ สำเร็จ |
| **Reorg & Clean** | ปรับโครงสร้างแยก Domain Modules (Approach B) + รวมค่าคงที่ปลอดภัย + ทำความสะอาดโค้ด | ✅ สำเร็จ (337 tests passed) |

### ผลการวัดผลจริงบนชุด Held-out (500 steps, n=100 samples)

ระบบใช้วิธีวัดผลแบบ **FIM Exact Match (EM)** และ **Sequence-Aware Token F1 (คำนวณผ่าน LCS)** บนชุดข้อมูลทดสอบ held-out ที่แยกไว้เฉพาะ (ไม่เคยผ่านการเทรน):

| ตัวชี้วัด (Metric) | Base Model | Fine-tuned Model | การเปลี่ยนแปลง (Δ) | สถานะเกณฑ์ (Gate) |
|---|:---:|:---:|:---:|:---:|
| **Exact Match %** | 1.0% | 2.0% | **+1.0%** | ผ่าน (เกณฑ์ ≥ +0.1pt) |
| **Token F1 (LCS)** | 0.248 | 0.338 | **+0.090** | ผ่าน (เกณฑ์ ≥ +0.01) |

> **ตัวอย่างพฤติกรรมโมเดลที่ปรับปรุงขึ้นจริง:**
> - การเติม Header ลิขสิทธิ์ซอฟต์แวร์: ค่า Token F1 เพิ่มจาก 0.28 เป็น 1.00 (เติมได้ถูกต้องสมบูรณ์)
> - การสร้างโครงสร้าง `_import_structure` ของ Transformers: ค่า Token F1 เพิ่มจาก 0.30 เป็น 1.00
> - กรณีที่ Base model สับสนและพิมพ์อักขระวนซ้ำ `!!!...` โมเดลที่ Fine-tune แล้วสามารถเขียน Import Statement `from ...modeling_tf_outputs import ...` ได้อย่างถูกต้อง (F1 เพิ่มจาก 0.00 เป็น 0.68) โดยไม่มีกรณี Regression (แย่ลง) เกิดขึ้นเลยแม้แต่เคสเดียว

---

## สถาปัตยกรรมและจุดเด่นของระบบ

1. **Fill-in-the-Middle (FIM) Engine เฉพาะทาง:**
   - รองรับรูปแบบ **PSM (Prefix-Suffix-Middle)** ตามมาตรฐานของ Qwen Coder
   - ตัดแบ่งขอบเขตเฉพาะที่ **Line Boundary** เพื่อให้โค้ดที่โมเดลเรียนรู้เป็นหน่วยตรรกะที่สมบูรณ์
   - ระบบป้องกัน Data Leakage สองชั้น: แยกชุดทดสอบ (Held-out 10%) ด้วย **Deterministic MD5 Hashing** พร้อมระบบ **Near-duplicate Leakage Guard (M4)** ดักจับโค้ดที่มีความคล้ายคลึงด้วย Line-shingle Jaccard Similarity (≥ 0.8) ทิ้งทันทีก่อนเริ่มเทรน
2. **การทำงานบน Intel Arc GPU (XPU) อย่างปลอดภัย:**
   - ใช้ Native Bfloat16 สำหรับประหยัด VRAM และเพิ่มความเร็วในการคำนวณ
   - ระบบ **Pre-flight VRAM Estimator** คำนวณความต้องการหน่วยความจำทั้งจากขนาดพารามิเตอร์, Activation, และ Buffers ล่วงหน้า หากพบความเสี่ยง VRAM ไม่พอจะล็อกการเทรนทันที
3. **การแยก Process เด็ดขาด (Subprocess Isolation & IPC):**
   - หน้าจอ Gradio ทำหน้าที่เป็นตัวแสดงผลเท่านั้น การเทรนและการประเมินผลจะรันบน Subprocess ด้วยโหมด `spawn`
   - สื่อสารผ่าน `multiprocessing.Queue` ด้วย 4 รูปแบบข้อความมาตรฐาน: `metric`, `log`, `status`, และ `error`
   - ระบบ **Watchdog** คอยตรวจสอบกระบวนการ หาก Subprocess เงียบผิดปกติหรือค้าง จะทำการแจ้งเตือน
   - ระบบ **Abort Escalation**: ส่งสัญญาณ `SIGTERM` ให้โมเดลหยุดอย่างสุภาพ หากไม่หยุดภายใน 10 วินาที จะยกระดับเป็น `SIGKILL` ทันที มั่นใจได้ว่าหน่วยความจำ VRAM จะถูกคืนให้ระบบ 100%
4. **ความสมบูรณ์ของ Checkpoint (Atomic Checkpoint Swap):**
   - ขณะเทรน Checkpoint จะถูกบันทึกในชื่อชั่วคราว `.saving` จนกระทั่งเขียนเสร็จสมบูรณ์จึงสลับชื่อ (Atomic Rename) ป้องกันไฟล์เสียหายเมื่อไฟดับหรือมีการสั่ง Abort กลางคัน
   - หากตรวจพบ Checkpoint ที่บันทึกค้างไว้ จะมีระบบกู้คืนอัตโนมัติก่อนเริ่มเทรนต่อ
5. **การอ่านชุดข้อมูลแบบ Batch Streaming (รองรับไฟล์ >20GB):**
   - สามารถอ่านไฟล์ `.parquet` จากโฟลเดอร์ `datasets/` ในเครื่องได้โดยตรงทีละ batch
   - ไม่ต้องแปลงเป็น HuggingFace Arrow cache ซ้ำซ้อน ประหยัดเนื้อที่ดิสก์ และไม่โหลดไฟล์ทั้งหมดขึ้น RAM

---

## ความต้องการของระบบ

- **ระบบปฏิบัติการ:** Linux 64-bit (ทดสอบและรับรองบน EndeavourOS / Arch Linux / Ubuntu 22.04+)
- **กราฟิกการ์ด (GPU):** การ์ดจอแยก Intel Arc (ทดสอบบน **Intel Arc B580 12GB Battlemage / Xe2** และรองรับตระกูล Alchemist A770/A750)
- **ไดรเวอร์และรันไทม์:**
  - `intel-compute-runtime` (OpenCL / Level Zero driver)
  - `level-zero-loader`
- **เวอร์ชัน Python:** **Python 3.11** เท่านั้น *(ไม่อนุญาตให้ใช้ Python 3.12+ เนื่องจาก PyTorch XPU wheel และไลบรารี C-extension ยังไม่รองรับสมบูรณ์บนเวอร์ชันใหม่กว่า)*
- **หน่วยความจำระบบ (RAM):** อย่างน้อย 16 GB
- **พื้นที่ว่างบนดิสก์:** อย่างน้อย 20 GB (สำหรับการเก็บ Cache, ชุดข้อมูล และ Checkpoints)

---

## ขั้นตอนการติดตั้ง

### 1. ติดตั้งไดรเวอร์ของ Intel (สำหรับ Arch / EndeavourOS)

```bash
sudo pacman -S intel-compute-runtime level-zero-loader
```
*(หากเป็น Ubuntu ให้ติดตั้งแพ็กเกจ `intel-opencl-icd` และ `libze-loader1` จากคลัง Intel oneAPI)*

### 2. สร้าง Virtual Environment ด้วย Python 3.11

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. ติดตั้ง PyTorch สำหรับ Intel XPU

```bash
# ต้องติดตั้งผ่าน index-url ของ PyTorch XPU โดยเฉพาะ (ห้ามติดตั้งจาก PyPI ปกติ)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/xpu
```

### 4. ติดตั้ง Dependencies และล็อกเวอร์ชัน

```bash
# ติดตั้งแพ็กเกจตามรายการที่ผ่านการทดสอบความเข้ากันได้
.venv/bin/pip install -r requirements.txt -c constraints.txt
```

> **หมายเหตุความปลอดภัย (Security Advisory - PEFT CVE-2026-71281):**
> ช่องโหว่ CVE-2026-71281 มีผลกระทบกับแพ็กเกจ `peft <= 0.19.1` (เกิดจากการเรียก `torch.load` ใน LoRA-GA/CorDA โดยไม่ได้ตั้ง `weights_only=True`) โปรเจกต์นี้ได้รับการตรึงเวอร์ชันไว้ที่ **`peft==0.21.1`** ใน `requirements.txt` และ `constraints.txt` แล้ว จึง**ปลอดภัยจากช่องโหว่ดังกล่าว 100%**

---

## คู่มือการเริ่มใช้งาน

### 1. ตรวจสอบความพร้อมของระบบ (Pre-flight Runtime Check)

ก่อนเริ่มใช้งาน ควรเรียกใช้สคริปต์ตรวจสอบสภาพแวดล้อมเพื่อยืนยันว่าไดรเวอร์ GPU และไลบรารีพร้อมทำงาน:

```bash
.venv/bin/python scripts/check_runtime.py
```
*หากระบบพร้อม ข้อความตรวจสอบทุกข้อ (Intel XPU, Memory, Bfloat16 Support, Disk) จะแสดงผลเป็นสีเขียวทั้งหมด*

### 2. เปิดใช้งาน Web Dashboard

เปิดใช้งานส่วนติดต่อผู้ใช้ Gradio Web Application:

```bash
.venv/bin/python app.py
```
เปิดเบราว์เซอร์ไปที่: **`http://127.0.0.1:7860`** *(ระบบผูกการทำงานกับ Localhost เท่านั้น เพื่อความปลอดภัย)*

---

### ขั้นตอนการทำงานบน Web Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GRADIO WEB DASHBOARD                             │
├──────────────────────────┬─────────────────────────┬────────────────────────┤
│ 1. Configuration         │ 2. Mission Control      │ 3. Playground & Export │
│ - เลือก Model / Dataset  │ - ปุ่ม Start / Abort    │ - ทดสอบ Predict Middle │
│ - ตั้งค่า LoRA / Steps   │ - กราฟ Loss & LR สด     │ - รัน Eval เปรียบเทียบ │
│ - Run Pre-flight Check   │ - กระดาน Live Log 200 แถว│ - Export LoRA / Merged │
└──────────────────────────┴─────────────────────────┴────────────────────────┘
```

#### แท็บ 1: Configuration & Pre-flight
- **เลือก Model:** เลือกโมเดลเริ่มต้น `Qwen/Qwen2.5-Coder-0.5B` หรือเลือกโฟลเดอร์โมเดลในเครื่องจากโฟลเดอร์ `models/`
- **เลือก Dataset:** เลือก `smangrul/hf-stack-v1` (Hub streaming) หรือเลือกไฟล์ `.parquet` ในเครื่องจากโฟลเดอร์ `datasets/`
- **ตั้งค่าการเทรน:** ปรับ LoRA Rank (8, 16, 32), Max Sequence Length (ค่าเริ่มต้น 1024, เพดานสูงสุด 2048), Training Steps (ค่าเริ่มต้น 500)
- **กดปุ่ม "Run Environment Check":** ระบบจะคำนวณขนาด VRAM ที่ต้องใช้ล่วงหน้า หากผ่านเกณฑ์จะแสดงสถานะสีเขียว (**Safe**) หรือสีส้ม (**Warning**) พร้อมปลดล็อกให้เริ่มเทรนได้

#### แท็บ 2: Training Mission Control
- **กดปุ่ม "Start Fine-Tuning":** ระบบจะเริ่มเปิด Subprocess สำหรับการฝึกสอน
- **ดูกราฟและสถานะแบบ Real-time:** แสดงเส้นกราฟ Training Loss, Validation Loss (ประเมินบนชุด Held-out ทุก 50 ก้าว) และอัตราการเรียนรู้ (Learning Rate)
- **กระดานข้อความ (Live Log):** แสดงบันทึกการทำงานสดจาก Worker Process
- **ปุ่ม "Abort":** หากต้องการยกเลิก สามารถกดปุ่ม Abort ได้ตลอดเวลา ระบบจะส่งสัญญาณ `SIGTERM` และตามด้วย `SIGKILL` ภายใน 10 วินาที เพื่อคืน VRAM ทั้งหมดให้ระบบทันที

#### แท็บ 3: Playground & Export
- **Predict Middle (ทดสอบเดาโค้ด):** ป้อนโค้ดส่วน Prefix และ Suffix เพื่อให้โมเดลที่เพิ่งเทรนเสร็จทดลองเติมโค้ดส่วน Middle แบบสด ๆ
- **Run Evaluation:** กดปุ่มเพื่อรันการวัดผลความแม่นยำ FIM Exact Match และ Token F1 เปรียบเทียบระหว่างโมเดลก่อนเทรนและหลังเทรน
- **Save Adapter Only:** บันทึกเฉพาะน้ำหนัก LoRA Adapter ขนาดเล็ก (~17 MB) สำหรับนำไปใช้งานต่อ
- **Merge & Export Full Weights:** รวมน้ำหนัก LoRA เข้ากับโมเดลหลัก แล้วบันทึกเป็นโมเดลฉบับเต็มลงในโฟลเดอร์ `exports/`

---

## การจัดการโมเดลและชุดข้อมูลในเครื่อง

Dropdown ในหน้าจอ Configuration สามารถตรวจจับไฟล์ที่วางไว้ในเครื่องได้โดยอัตโนมัติ:

| โฟลเดอร์ | ชนิดของไฟล์ที่รองรับ | ตัวอย่างการใช้งาน |
|---|---|---|
| `models/` | โฟลเดอร์โมเดลที่มีไฟล์ `config.json` และน้ำหนักโมเดล (`.safetensors`) | วางไว้ที่ `models/My-Custom-Model` → เลือก `My-Custom-Model` จาก Dropdown |
| `datasets/` | โฟลเดอร์ที่เก็บไฟล์ชุดข้อมูลนามสกุล `.parquet` | วางไว้ที่ `datasets/my_code_dataset` → เลือก `my_code_dataset` จาก Dropdown |

- **การอ่านข้อมูลท้องถิ่น:** ระบบจะอ่านไฟล์ `.parquet` แบบ Streaming ทีละ 64 แถว ทำให้รองรับไฟล์ขนาดใหญ่ได้ถึงระดับ **~20 GB** โดยไม่กินพื้นที่ RAM และไม่ต้องแปลงแคชซ้ำ
- **โมเดลจาก Hugging Face Hub:** เมื่อพิมพ์ Model ID เช่น `Qwen/Qwen2.5-Coder-0.5B` ระบบจะใช้แคชร่วมของ Hugging Face โดยไม่คัดลอกไฟล์ซ้ำซ้อนในโปรเจกต์

---

## การประเมินผลโมเดล (Evaluation CLI)

นอกจากการกดปุ่มประเมินผลบนหน้าเว็บ คุณสามารถใช้คำสั่ง CLI ผ่านสคริปต์ `eval.py` เพื่อวัดผลโมเดลได้อย่างละเอียด:

```bash
# 1. รันการวัดผลบนโมเดลตั้งต้น (Base Model) → บันทึกผลที่ data_cache/eval/base.json
.venv/bin/python eval.py --mode base

# 2. รันการวัดผลบนโมเดลที่ผ่านการ Fine-tuned (ดึง Checkpoint ล่าสุด) → บันทึกผลที่ data_cache/eval/finetuned.json
.venv/bin/python eval.py --mode finetuned

# 3. เปรียบเทียบผลลัพธ์ระหว่าง Base และ Fine-tuned
.venv/bin/python eval.py --compare
```

**เกณฑ์การตัดสินผล (Min-Delta Gate):**
เมื่อรัน `--compare` โมเดล Fine-tuned จะต้องมีผลการประเมินดีกว่าโมเดล Base เกินค่าเกณฑ์ขั้นต่ำ:
- **Exact Match (EM):** เพิ่มขึ้นอย่างน้อย **+0.1%**
- **Token F1:** เพิ่มขึ้นอย่างน้อย **+0.01**
- *หากค่าเท่าเดิมหรือแย่ลง ระบบจะส่ง Exit Code = 1 (FAIL)*

---

## การบีบอัดและให้บริการโมเดล (GGUF / llama.cpp)

ระบบรองรับการแปลงโมเดลที่ Fine-tune แล้วไปเป็นฟอร์แมต **GGUF** เพื่อนำไปรันบนฮาร์ดแวร์ Intel Arc ได้อย่างมีประสิทธิภาพสูงสุดผ่าน **llama.cpp** โดยใช้ตัวเร่งความเร็ว **Vulkan Backend**:

### 1. คอมไพล์เครื่องมือ llama.cpp

เรียกใช้สคริปต์ติดตั้งและคอมไพล์ `llama.cpp` พร้อมเปิดใช้งาน Vulkan:

```bash
bash scripts/setup_llamacpp.sh
```

### 2. ทำ Benchmark และแปลงโมเดล (Quantization)

ทำการแปลงโมเดลเป็นระดับความละเอียดต่าง ๆ (เช่น `fp16`, `q8_0`, `q4_k_m`) พร้อมทดสอบความเร็วและวัดผลความแม่นยำในคำสั่งเดียว:

```bash
.venv/bin/python scripts/benchmark_compression.py \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --variants fp16,q8_0,q4_k_m
```

### 3. เปิดให้บริการ Local API Server

รัน `llama-server` เพื่อเปิดให้บริการโมเดลผ่าน REST API แบบ Local (เข้ากันได้กับ OpenAI API):

```bash
.venv/bin/python scripts/serve.py --model Qwen/Qwen2.5-Coder-0.5B --variant q4_k_m
```
*ระบบจะเปิดเซิร์ฟเวอร์ที่: `http://127.0.0.1:8080` (สามารถกด `Ctrl+C` เพื่อหยุดการทำงานได้อย่างปลอดภัยโดยไม่ค้าง VRAM)*

---

## โครงสร้างไดเรกทอรีและสถาปัตยกรรมโค้ด

โค้ดทั้งหมดถูกจัดวางตามแนวทาง **Domain-Driven Layout (Approach B)** เพื่อแยกความรับผิดชอบอย่างชัดเจน:

```text
Fine-tuning-LLM/
├── configs/
│   ├── safe_defaults.py          # นิยามค่าคงที่และขีดจำกัดความปลอดภัยของระบบทั้งหมด (Single Source of Truth)
│   └── fim_registry.json         # ตารางจับคู่ Special Tokens ของ FIM ตามแต่ละตระกูลโมเดล
│
├── core/
│   ├── infra/                    # ระบบโครงสร้างพื้นฐานและการจัดการฮาร์ดแวร์
│   │   ├── hardware.py           # ตรวจจับ Intel XPU, หน่วยความจำ RAM และพื้นที่ดิสก์
│   │   ├── estimator.py          # คำนวณประมาณการ VRAM ล่วงหน้าทั้งแบบ Static และ Dynamic
│   │   ├── ipc_bridge.py         # โปรโตคอลสื่อสาร IPC ระหว่าง Process พร้อมระบบ Watchdog และ Abort
│   │   └── sandbox.py            # ควบคุมความปลอดภัยของ Path ป้องกัน Path Traversal
│   │
│   ├── data/                     # ระบบเตรียมและประมวลผลข้อมูล
│   │   ├── dataset_builder.py    # อ่าน Parquet แบบสตรีมมิ่ง, ตัด FIM (PSM), จัดการ Held-out Split
│   │   └── fim.py                # ระบบ FIM Guardrail และการประกอบ Prompt รูปแบบ PSM
│   │
│   ├── train/                    # ระบบและกระบวนการฝึกสอนโมเดล
│   │   ├── args.py               # สร้าง TrainingArguments และตรวจสอบความถูกต้องของการตั้งค่า
│   │   ├── callbacks.py          # NanGuard, AtomicSaveTrainer (เขียน Checkpoint ปลอดภัย), StreamToQueue
│   │   ├── runner.py             # ฟังก์ชันรันการเทรนหลัก (run_training) และจัดการ Checkpoint
│   │   ├── export.py             # ฟังก์ชัน Export LoRA Adapter และ Merge Full Weights
│   │   └── predict.py            # ฟังก์ชันเดาโค้ด (Predict Middle) สำหรับ Playground
│   │
│   ├── eval/                     # ระบบประเมินผลความแม่นยำ
│   │   ├── evaluator.py          # ตัวคำนวณ Exact Match, LCS-based Token F1 และเปรียบเทียบผล
│   │   └── worker.py             # Subprocess Worker สำหรับการประเมินผลพร้อมส่งความคืบหน้าเข้าคิว
│   │
│   └── compress/                 # ไปป์ไลน์การบีบอัดโมเดลด้วย GGUF และ llama.cpp (Vulkan)
│       ├── config.py             # การตั้งค่าเครื่องมือ llama.cpp และ Path ไบนารี
│       ├── quantizer.py          # สคริปต์เรียกคำสั่งแปลงโมเดลเป็น GGUF
│       ├── llama_runner.py       # ควบคุม Subprocess ของ llama-bench และ llama-server
│       ├── llama_eval.py         # วัดผลความแม่นยำผ่าน llama-server
│       └── report.py             # สรุปผลการทดสอบ Benchmark เป็น JSON และตาราง
│
├── ui/                           # ระบบหน้าจอผู้ใช้ (Gradio Web UI)
│   ├── controller.py             # TrainingController จัดการ State Machine และ IPC โดยไม่อิงกับ Gradio
│   ├── dashboard.py              # จัดวางบล็อก UI 3 แท็บและเชื่อมโยง Event Listener
│   └── components.py             # คอมโพเนนต์ UI พื้นฐาน (กราฟ Matplotlib ป้องกัน Thread-lock, Gauge)
│
├── scripts/                      # สคริปต์ยูทิลิตี้สำหรับการใช้งานผ่านเทอร์มินัล
│   ├── check_runtime.py          # ตรวจสอบสภาพแวดล้อมฮาร์ดแวร์และไดรเวอร์
│   ├── pipeline_smoke.py         # ทดสอบการทำงานตลอดทั้งไปป์ไลน์แบบครบวงจร
│   ├── smoke_sft.py              # ทดสอบการรัน SFT สั้น ๆ สำหรับการตรวจสอบความเข้ากันได้
│   ├── setup_llamacpp.sh         # สคริปต์ช่วยติดตั้งและคอมไพล์ llama.cpp บน Vulkan
│   ├── benchmark_compression.py  # รัน Benchmark การบีบอัด GGUF
│   └── serve.py                  # สคริปต์เสิร์ฟโมเดล GGUF ผ่าน HTTP API
│
├── tests/                        # ชุดการทดสอบอัตโนมัติ (pytest) จัดโครงสร้างสะท้อน core/
│   ├── infra/                    # เทสต์ฮาร์ดแวร์, estimator, sandbox
│   ├── data/                     # เทสต์การตัด FIM, leakage guard, dataset builder
│   ├── train/                    # เทสต์การสร้าง args, callbacks, runner, export
│   ├── eval/                     # เทสต์ evaluator, worker, CLI
│   ├── compress/                 # เทสต์ quantization, runner, benchmark
│   ├── ui/                       # เทสต์ UI controller, dashboard, components
│   └── conftest.py               # ตัวช่วยทดสอบส่วนกลาง (FakeQueue ฯลฯ)
│
├── app.py                        # Entry Point หลักสำหรับเริ่มรัน Web Dashboard
├── eval.py                       # เครื่องมือ CLI สำหรับการประเมินผลโมเดล
├── requirements.txt              # รายการแพ็กเกจหลักของระบบ
└── constraints.txt               # ไฟล์ล็อกเวอร์ชันที่ปลอดภัยและผ่านการทดสอบ
```

---

## กติกาและหลักการออกแบบความปลอดภัย

เพื่อให้ระบบทำงานได้อย่างมั่นคงบนฮาร์ดแวร์จำกัด โปรเจกต์นี้มีกฎเหล็กในการออกแบบ (Invariants) ดังนี้:

1. **ค่าคงที่ทั้งหมดต้องมาจาก [`configs/safe_defaults.py`](file:///run/media/teerametr/3b44566d-03e0-4d76-8469-1bf9cf418a63/Project/Fine-tuning-LLM/configs/safe_defaults.py):** ห้ามทำการ Hardcode ตัวเลขการตั้งค่าหรือพารามิเตอร์ซ้ำซ้อนในโมดูลอื่นเด็ดขาด
2. **ขีดจำกัดความปลอดภัยของฮาร์ดแวร์ (Safety Limits):**
   - Per-device Train/Eval Batch Size ต้องเป็น **1** เสมอ (ป้องกัน OOM บนการ์ดระดับ 12GB)
   - Max Sequence Length ถูกจำกัดเพดานไว้ที่ **2048** (ค่าเริ่มต้น 1024)
   - Learning Rate ถูกจำกัดไว้ที่ **2e-4** พร้อม Gradient Accumulation Steps = **8**
3. **การป้องกัน Data Leakage ระหว่าง Train และ Eval:**
   - การแบ่งชุดข้อมูลใน [`is_heldout`](file:///run/media/teerametr/3b44566d-03e0-4d76-8469-1bf9cf418a63/Project/Fine-tuning-LLM/core/data/dataset_builder.py#L104) ต้องใช้อัลกอริทึม MD5 ที่คงที่ตลอดไป โค้ดเดิมจะต้องตกอยู่ฝั่งเดิมเสมอ
   - กระบวนการเทรนต้องเรียกใช้ `filter_train_codes` เพื่อตัดข้อมูลที่อาจรั่วไหลออกทุกครั้ง
4. **การจัดการ VRAM แบบ Single-Device Lock:**
   - ไม่อนุญาตให้รันฟังก์ชันที่โหลดโมเดลขึ้น GPU ซ้อนกันเด็ดขาด (เช่น ไม่สามารถกด Run Eval หรือ Predict ขณะที่ระบบกำลังเทรนอยู่ได้)
   - ป้องกันด้วยสองชั้น: ปิดการทำงานของปุ่มบน UI ตาม State Machine และล็อกคิวด้วย `concurrency_id="model_load"` ใน Gradio
5. **ความปลอดภัยของระบบไฟล์ (Sandbox Path Traversal Guard):**
   - ทุก Path ที่รับจากผู้ใช้หรือ Dropdown จะต้องผ่านการตรวจสอบด้วย `project_roots` ใน [`core/infra/sandbox.py`](file:///run/media/teerametr/3b44566d-03e0-4d76-8469-1bf9cf418a63/Project/Fine-tuning-LLM/core/infra/sandbox.py) เพื่อป้องกันการเข้าถึงหรือเขียนไฟล์ออกนอกขอบเขตโปรเจกต์
6. **Lazy Loading สำหรับไลบรารีขนาดใหญ่:**
   - ในโมดูลประเมินผล [`evaluator.py`](file:///run/media/teerametr/3b44566d-03e0-4d76-8469-1bf9cf418a63/Project/Fine-tuning-LLM/core/eval/evaluator.py) จะต้องไม่ทำการ `import torch` ที่ระดับ Top-level เพื่อให้คำสั่ง CLI และส่วนอื่นเรียกดูข้อมูลความช่วยเหลือหรือทำงานร่วมกับ subprocess ได้อย่างรวดเร็วโดยไม่ต้องรอโหลดไลบรารีขนาดใหญ่

---

## การทดสอบระบบ (Testing)

โปรเจกต์นี้ใช้ **Test-Driven Development (TDD)** และมีชุดทดสอบครอบคลุมทุกโมดูล:

```bash
# รันชุดทดสอบ Unit Tests ทั้งหมด (ทำงานรวดเร็ว ประมาณ 10-15 วินาที)
.venv/bin/python -m pytest tests/ -q
# ผลลัพธ์: 337 passed, 8 deselected

# รันชุดทดสอบ Integration Tests (ทดสอบการติดต่อ Hardware XPU จริง การ Abort และการบันทึก Checkpoint)
.venv/bin/python -m pytest tests/ -m integration -v

# รันการทดสอบกระบวนการทำงานครบวงจรตั้งแต่ต้นจนจบ (End-to-End Pipeline Smoke Test)
.venv/bin/python scripts/pipeline_smoke.py --mode all
```

---

## การแก้ไขปัญหาที่พบบ่อย (Troubleshooting)

| อาการที่พบ | สาเหตุที่เป็นไปได้ | วิธีการแก้ไข |
|---|---|---|
| **ขึ้นข้อความ `no_xpu` ตอนตรวจระบบ** | ระบบไม่พบการ์ดจอ Intel หรือยังไม่ได้ลง Driver | ตรวจสอบว่าติดตั้ง `intel-compute-runtime` และ `level-zero-loader` แล้ว จากนั้นทำการ Log out หรือ Restart เครื่อง |
| **ขึ้นข้อความ `insufficient_disk` หรือ `insufficient_ram`** | ทรัพยากรระบบไม่ถึงเกณฑ์ขั้นต่ำ | ตรวจสอบให้แน่ใจว่ามีพื้นที่ดิสก์ว่าง ≥ 20 GB และ RAM ว่าง ≥ 16 GB ก่อนเริ่มการเทรน |
| **Error: `Cannot re-initialize XPU in forked subprocess`** | มีการเรียกใช้งาน PyTorch XPU ก่อนคำสั่ง `fork` | ห้ามใช้โหมด `fork` กับ PyTorch XPU ระบบถูกตั้งค่าให้ใช้ `multiprocessing.set_start_method("spawn")` ใน `app.py` เท่านั้น |
| **Eval ฟ้องว่า `No checkpoint found in ...`** | ยังไม่เคยผ่านการฝึกสอนในโฟลเดอร์ปลายทางนั้น | ให้ทำการกดเริ่มเทรนบนหน้าต่าง Mission Control หรือระบุโฟลเดอร์ Checkpoint ที่มีอยู่จริงก่อนรัน `--mode finetuned` |
| **Error: `Cannot compare: eval sets differ`** | จำนวนตัวอย่างของ Base และ Fine-tuned ไม่เท่ากัน | การเปรียบเทียบผลลัพธ์จะต้องรันด้วยจำนวนเคส (`--n-cases`) ที่เท่ากันทั้งสองโหมด (ค่าเริ่มต้นคือ 100 เคส) |
| **เกิดปัญหา Out of Memory (OOM) ระหว่างเทรน** | มีกระบวนการอื่นแย่งใช้งานหน่วยความจำ VRAM | ตรวจสอบว่าไม่มีโปรเซสอื่นกำลังใช้งาน GPU หรือลองปรับลด `max_seq_length` ลงมาที่ 512 หรือ 1024 ในหน้า Configuration |
| **Loss กลายเป็นค่า NaN หรือ Inf ติดต่อกัน** | อัตราการเรียนรู้สูงเกินไปหรือข้อมูลผิดปกติ | ระบบมี `NanGuardCallback` ที่จะหยุดการเทรนอัตโนมัติหากพบค่าผิดปกติ 3 ครั้งติด ให้ลองปรับลดค่า Learning Rate ลง |
