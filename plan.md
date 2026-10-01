# Master Project Blueprint: Specialized Code Model Tuner (Intel XPU Edition)

เอกสารแผนแม่บทสำหรับการพัฒนาโปรเจกต์ Fine-tuning โมเดลเพื่องานเดาโค้ดเฉพาะทาง (Fill-in-the-Middle & Code Completion) โดยใช้ภาษา **Python ทั้งระบบ (Single-Language Stack)** รองรับฮาร์ดแวร์ **Intel GPU (XPU)** พร้อมระบบประเมินทรัพยากร ป้องกันระบบค้าง และส่วนติดต่อผู้ใช้ด้วย Gradio

---

## 1. ข้อมูลภาพรวมโครงการ (System Specifications)

* **เป้าหมายหลัก:** ปรับจูนโมเดลโค้ดขนาด 0.5B – 3B ให้เข้าใจ Syntax, ไลบรารีเฉพาะทาง หรือโครงสร้างโค้ดภายใน — **โมเดลเป้าหมายเดียว: `Qwen/Qwen2.5-Coder-0.5B`** (โฟกัสค่ายเดียวเพื่อความเสถียร หากจะเพิ่ม DeepSeek-Coder / StarCoder2 ค่อยเพิ่มผ่าน `fim_registry.json` + LoRA target config ภายหลัง)
* **กลยุทธ์การเทรน:** Parameter-Efficient Fine-Tuning (PEFT) ด้วย LoRA บนความละเอียด Native Bfloat16 (BF16)
* **Backend ฮาร์ดแวร์:** PyTorch Upstream XPU Backend (`torch.xpu`) ร่วมกับ Intel Compute Runtime / Level Zero
* **สถาปัตยกรรม UI:** Event-Driven Web Application ด้วย Gradio Blocks ทำงานแบบ Asynchronous แยก Thread/Process อิสระ
* **Runtime Baseline (สภาพแวดล้อมอ้างอิง):**
  * OS: EndeavourOS (Arch-based), การ์ด **Intel Arc B580 (Battlemage / Xe2)**
  * **Python: สร้าง venv รุ่น 3.11 หรือ 3.12 เท่านั้น** — เครื่องมี Python 3.14 แต่ build ของ torch XPU ผลิต wheel สำหรับ Python 3.9–3.13 เท่านั้น (ยังไม่มี cp314) และ `numpy` รุ่น 1.x ไม่รองรับ Python 3.14 → **ห้ามใช้ Python ของระบบ (3.14)**
  * ติดตั้ง PyTorch ผ่าน `pip install torch --index-url https://download.pytorch.org/whl/xpu` (ห้ามใช้ wheel `torch` จาก PyPI ปกติ)
* **Dataset อ้างอิง:** `smangrul/hf-stack-v1` บน HF Hub (คอลัมน์ `content` — raw code สำหรับทำ FIM เอง, ไม่ gated, ใช้ streaming ได้) แยก held-out 5–10% — เปลี่ยนผ่าน config ได้โดยไม่แตะโค้ด

---

## 2. โครงสร้างไฟล์และไดเรกทอรี (Directory Structure)

```text
code_tuner/
│
├── configs/
│   ├── fim_registry.json         # แฟ้มแมป Special Tokens แยกตามค่ายโมเดล
│   └── safe_defaults.py          # ค่าขีดจำกัดความปลอดภัย (Safety Boundaries)
│
├── core/
│   ├── __init__.py
│   ├── hardware.py               # ระบบตรวจจับ Intel XPU, System RAM, Free Disk
│   ├── estimator.py              # เครื่องยนต์คำนวณและประเมิน VRAM / Feasibility
│   ├── dataset_builder.py        # ตัวประมวลผล FIM Splitting & Tokenizer Packaging
│   ├── trainer_worker.py         # Subprocess Pipeline: SFTTrainer + PEFT Engine
│   └── ipc_bridge.py             # Message Protocol, Watchdog & Status Synchronization
│
├── ui/
│   ├── __init__.py
│   ├── components.py             # Reusable UI Blocks (Gauge, Log Viewer, Input Box)
│   └── dashboard.py              # หน้าต่างควบคุมหลัก (Tabs: Pre-flight, Train, Playground)
│
├── exports/                      # ไดเรกทอรีจัดเก็บ LoRA Adapter และ Merged Weights
├── data_cache/                   # โฟลเดอร์สำรองแคช Dataset ท้องถิ่น
├── app.py                        # Entry Point หลักของระบบ (Process Initializer)
├── eval.py                       # เกณฑ์วัดผล base vs fine-tuned (FIM Exact Match + Token F1)
├── constraints.txt               # ไฟล์ล็อกเวอร์ชันที่เข้าคู่กัน (ได้จาก Compat Spike)
├── requirements.txt              # รายการแพ็กเกจ Python
└── README.md                     # คู่มือการติดตั้งไดรเวอร์และเริ่มใช้งาน (ภาษาไทย)
```

---

## 3. ผังการทำงานและการแยก Process (System Architecture)

```text
[ app.py (Main Process: spawn mode) ]
  │
  ├── 1. Initial Launch: Gradio UI Server (WebSocket Enabled)
  │      └── Gradio Queue (concurrency_limit=1)
  │      └── Bind localhost:7860 เท่านั้น (ไม่เปิดสู่ LAN)
  │
  ├── 2. Pre-flight Check:
  │      └── core/hardware.py ──> core/estimator.py ──> Return Verdict (Safe/Block)
  │
  └── 3. Start Training Trigger:
         │
         ├── Spawns Subprocess: multiprocessing.Process(target=run_training)
         │   │
         │   ├── [Subprocess Engine]
         │   │   ├── Load Tokenizer & Apply FIM Registry
         │   │   ├── Prepare HF Dataset via ConstantLengthDataset
         │   │   ├── Load Base Model in BF16 (SDPA Attention)
         │   │   ├── Apply LoRA Config (Targeting Attention & MLP)
         │   │   └── SFTTrainer Run Loop
         │   │       └── StreamToQueueCallback ──> metric / log / status / error
         │   │
         │   └── IPC Channel (multiprocessing.Queue) ผ่าน core/ipc_bridge.py
         │       └── [Main UI] gr.Timer(1s) reads Queue ──> Update Loss Plot / Logs
         │       └── Watchdog: เช็ค process.is_alive() ทุกวินาที ──> zombie detection
         │
         ├── User Abort Trigger:
         │      └── SIGTERM ──> รอ 10 วิ ──> SIGKILL (คืน Level Zero / XPU VRAM ทันที)
         │
         └── App Exit Trigger:
                └── terminate child process ทันที (ป้องกัน orphan + XPU context leak)
```

---

## 4. รายละเอียดข้อกำหนดแต่ละโมดูล (Detailed Module Specifications)

### โมดูลที่ 1: `core/hardware.py` (Hardware Environment Inspector)
* **หน้าที่:** ดึงข้อมูลสภาพแวดล้อมฮาร์ดแวร์ก่อนทำกิจกรรมใด ๆ — เป็นแหล่งความจริงเดียว (single source of truth) ของ "สภาพแวดล้อมพร้อมหรือไม่"
* **ตัวแปรที่ต้องดึง:**
  * XPU Device Availability (`hasattr(torch, "xpu") and torch.xpu.is_available()`)
  * Device Name (`torch.xpu.get_device_name(0)`)
  * Total & Free VRAM (`torch.xpu.mem_get_info(0)`)
  * System RAM Capacity & Available (`psutil.virtual_memory()`)
  * Free Disk Space บริเวณที่กำหนด Output Directory (`psutil.disk_usage('.')`)
* **เกณฑ์ตัดสินของ `status`** (ค่า threshold เก็บใน `configs/safe_defaults.py`):
  * XPU ไม่มี → `no_xpu`
  * RAM ว่าง < **16 GB** → `insufficient_ram` (สำหรับโมเดล 3B: weights ~6GB + OS/runtime + buffer สำหรับ datasets cache ที่กิน RAM เฉพาะตัว)
  * ดิสก์ว่าง < **20 GB** → `insufficient_disk`
  * ผ่านทุกข้อ → `ready`
* **Output Data Contract:**
  ```python
  {
      "status": "ready" | "no_xpu" | "insufficient_ram" | "insufficient_disk",
      "device_name": str,
      "total_vram_gb": float,
      "free_vram_gb": float,
      "ram_available_gb": float,
      "disk_free_gb": float
  }
  ```
  > เงื่อนไข: `status == "ready"` ต้องเชื่อถือได้ 100% — fail-fast ตั้งแต่ชั้นนี้ โค้ดส่วนอื่นจึงเช็คบรรทัดเดียวได้โดยไม่ต้องรู้กฎของ estimator

### โมดูลที่ 2: `core/estimator.py` (Feasibility & VRAM Calculator)
* **หน้าที่:** คำนวณความต้องการทรัพยากรล่วงหน้า ป้องกันปัญหา Out of Memory (OOM)
* **โมเดลคณิตศาสตร์:**
  $$\text{Total Required VRAM} = M_{\text{weights}} + M_{\text{trainable}} + M_{\text{activations}} + M_{\text{overhead}}$$
  1. $M_{\text{weights}} = P \times 2 \text{ bytes}$ (สำหรับ BF16 โดย $P$ คือจำนวน Parameters เช่น 1.5B)
  2. $M_{\text{trainable}} = P_{\text{lora}} \times (2_{\text{grads}} + 8_{\text{optimizer}}) \approx 0.05 \text{ ถึง } 0.15 \text{ GB}$
  3. $M_{\text{activations}} = b \times s \times d_{\text{model}} \times N_{\text{layers}} \times 2 \text{ bytes}$ (เมื่อเปิด Gradient Checkpointing)
  4. $M_{\text{overhead}} \approx 1.0 \text{ GB}$ (สำรองสำหรับ XPU Context และ Driver Buffer)
* **ที่มาของ $P$ (จำนวน Parameters):**
  1. โมเดล default (`Qwen/Qwen2.5-Coder-0.5B`): hardcode ค่าไว้ใน `safe_defaults.py`
  2. Custom HF Repo ID: ดึง `config.json` จาก HF Hub มา **แคชครั้งแรก** แล้วคำนวณจาก field ของ config
  3. ออฟไลน์ / โหลดไม่ได้: ให้ user **กรอกจำนวนพารามิเตอร์ (B) ใน UI เป็น fallback** (ถ้าไม่กรอก = บล็อก ห้ามเดา)
* **การ Calibrate:** หลังรันเทรนรอบแรก เทียบค่า `Total Required` กับ VRAM จริงที่ใช้ (`mem_get_info` ก่อน/หลังเริ่มเทรน) แล้วปรับค่าสัมประสิทธิ์ในสูตรให้ความคลาดเคลื่อน ≤ 10% ก่อนถือว่า estimator ใช้งานจริงได้
* **เกณฑ์การตัดสิน (Verdict Thresholds):**
  * **Safe (เขียว):** $\text{Total Required} \le 0.75 \times \text{Free VRAM}$
  * **Warning (เหลือง):** $0.75 \times \text{Free VRAM} < \text{Total Required} \le 0.90 \times \text{Free VRAM}$
  * **Blocked (แดง):** $\text{Total Required} > 0.90 \times \text{Free VRAM}$ หรือ $\text{Free Disk} < 20 \text{ GB}$

### โมดูลที่ 3: `core/dataset_builder.py` (FIM Token Registry & Processing)
* **หน้าที่:** แปลงโค้ดดิบ (Raw Code) ให้กลายเป็นโครงสร้าง Fill-in-the-Middle
* **Token Registry (`configs/fim_registry.json`):**
  ```json
  {
    "qwen": {
      "prefix": "<|fim_prefix|>",
      "suffix": "<|fim_suffix|>",
      "middle": "<|fim_middle|>"
    },
    "starcoder": {
      "prefix": "<fim_prefix>",
      "suffix": "<fim_suffix>",
      "middle": "<fim_middle>"
    },
    "deepseek": {
      "prefix": "<｜fim begin｜>",
      "suffix": "<｜fim hole｜>",
      "middle": "<｜fim end｜>"
    }
  }
  ```
* **การตรวจสอบ FIM Tokens (บังคับก่อนเทรน):**
  * Qwen2.5-Coder **มี token อยู่ใน tokenizer แล้ว** (`<|fim_prefix|>` ID 151659, `<|fim_middle|>` ID 151660) → ตรวจสอบด้วยโค้ด (`tokenizer.convert_tokens_to_ids`) แล้ว **ห้าม resize embedding เด็ดขาด**
  * ถ้าโมเดลที่เลือกไม่มี FIM token → ขึ้นสถานะ error ใน UI และ **ห้ามเริ่มเทรน**
* **กติกา FIM (Hard Rules):**
  * **`fim_rate = 0.5`** — อีก 50% เป็น plain LM (รักษา ability เดิม)
  * **รูปแบบ PSM เท่านั้น** (prefix → suffix → middle) — ตรงกับรูปแบบ inference ของ Playground
  * สุ่มจุดตัด 3 ท่อน **ที่ boundary บรรทัด** — ห้ามตัดกลางบรรทัด
  * วาง **EOS ท้ายทุก sample**, ข้าม/ตัด sample ที่สั้นเกินค่าขั้นต่ำ, truncate ไม่ให้เกิน `max_seq_length` (Default: 1024)
  * การสุ่มทั้งหมดใช้ `seed = 42` (ผลซ้ำได้)
  * รูปแบบฟอร์แมต:
    $$\text{Formatted Text} = \text{Prefix Token} + \text{Prefix Text} + \text{Suffix Token} + \text{Suffix Text} + \text{Middle Token} + \text{Middle Text} + \text{EOS}$$
* **Dataset Config:** HF Repo ID + คอลัมน์โค้ด + split ratio เก็บใน config (เปลี่ยนได้ไม่แตะโค้ด) — default: `smangrul/hf-stack-v1` / คอลัมน์ `content` / held-out 5–10%

### โมดูลที่ 4: `core/trainer_worker.py` (Training Subprocess)
* **หน้าที่:** รันลูปการเทรนใน Process แยก พร้อมส่ง Metric ผ่าน Queue
* **กฎการคอนฟิกฮาร์ดแวร์สำหรับ XPU:**
  * Model Loading: โหลดตรงด้วย `torch_dtype=torch.bfloat16` และ `attn_implementation="sdpa"`
  * Optimizer: ใช้ `adamw_torch` (ห้ามใช้ `paged_adamw_8bit` ที่ขึ้นกับ CUDA)
  * Gradient Checkpointing: `True` เสมอ
  * API ของ `trl`: ใช้รุ่น ≥ 0.12 — เรียกด้วย `processing_class=` (**ห้ามใช้ `tokenizer=`** ซึ่งถูกแทนที่แล้ว)
* **Safe Hyperparameters Defaults:**
  * `per_device_train_batch_size = 1`
  * `gradient_accumulation_steps = 8`
  * `learning_rate = 2e-4`
  * `warmup_ratio = 0.03`
  * `seed = 42`
  * `save_total_limit = 2` (ลบ Checkpoint เก่าทิ้งอัตโนมัติ)
  * `save_steps = 100`
  * `max_steps = 500`
* **Guardrails ภายใน Worker:**
  * **NaN/Inf Guard:** ถ้า loss เป็น non-finite 3 ครั้งติด → สั่ง abort อัตโนมัติ + ส่ง `error` message ขึ้น Log Viewer (กันเทรนเป็นขยะแบบไม่รู้ตัว)
  * **Checkpoint Atomic:** เขียน checkpoint ลงโฟลเดอร์ temp แล้ว `rename` เมื่อเสร็จ (กันไฟล์เสียถ้าโดน abort กลาง `save_steps`)
  * ส่งค่า learning rate ปัจจุบันลง Queue พร้อม loss (แสดงในกราฟ)

### โมดูลที่ 5: `core/ipc_bridge.py` (Message Protocol, Watchdog & Synchronization)
* **หน้าที่:** กำหนดรูปแบบข้อความระหว่าง subprocess กับ UI และเฝ้าระวังสุขภาพของ process
* **Message Types (ฟิลด์ `type`):**
  | type | payload | ใช้ทำอะไร |
  |---|---|---|
  | `metric` | step, loss, lr, epoch | อัปเดตกราฟ loss/LR |
  | `log` | level, text | แสดงใน Live Log Terminal |
  | `status` | `starting` \| `training` \| `saving` \| `finished` \| `aborted` | สลับสถานะปุ่มใน UI |
  | `error` | message + **full traceback** | แสดงใน Log Viewer เมื่อ process ล้มเหลว |
* **Watchdog:** `gr.Timer(1s)` เช็ค `process.is_alive()` ทุกครั้ง — ถ้า process ตายโดยไม่ได้ส่ง `finished`/`aborted` มาเอง → สร้าง `error` message ขึ้น UI ทันที (zombie detection, ป้องกัน UI ค้างรอ Queue ตลอดกาล)
* **Abort Escalation:** สั่ง `SIGTERM` → รอ 10 วินาที → ถ้ายังไม่ตาย `SIGKILL` (คืน XPU VRAM ให้ระบบ)
* **App Exit:** ลงทะเบียน exit handler ที่ `app.py` → terminate child process ทันทีเมื่อ Gradio ปิด (ป้องกัน orphan + XPU context leak)

### โมดูลที่ 6: `ui/components.py` (Reusable UI Blocks)
* **หน้าที่:** สร้าง Blocks ที่ใช้ซ้ำได้ ให้ `dashboard.py` เรียกใช้แทนการเขียนซ้ำ
* **ชิ้นส่วน:** `VRAM Gauge` (ไฟ Safe/Warning/Blocked), `Log Viewer` (append-only, รองรับ error traceback), `Metric Plot` (loss + lr), `Input Box`/`Dropdown` สำหรับ model/dataset config
* **Data Contract:** รับ output ดิบจาก `hardware.py` / `ipc_bridge.py` เท่านั้น — ห้ามแปลงค่า/ตีความใน component

### โมดูลที่ 7: `ui/dashboard.py` (Gradio Interface)
* **Tab 1: Configuration & Pre-flight Inspection**
  * เลือกรุ่นโมเดล (Dropdown รองรับ Custom HF Repo ID — ถ้าดึง config ไม่ได้ ให้กรอกจำนวนพารามิเตอร์เอง)
  * ระบุชื่อ Dataset และคอลัมน์โค้ด
  * ปรับค่า LoRA Rank ($r=8, 16, 32$) และ Max Sequence Length
  * ปุ่ม "Run Environment Check" แสดงสถานะ VRAM, พื้นที่ Disk และไฟสัญญาณเตือน (Safe / Warning / Blocked พร้อมบอกสาเหตุ เช่น `insufficient_disk`)
* **Tab 2: Training Mission Control**
  * ปุ่ม "Start Fine-Tuning" (Disable ทันทีเมื่อเริ่มงาน)
  * ปุ่ม "Abort Process" (ผ่าน ipc_bridge: SIGTERM → 10 วิ → SIGKILL)
  * Real-time Loss + LR Plot (ดึงค่าจาก Queue ทุก 1 วินาที)
  * Live Log Terminal (แสดง Step/Loss/error traceback)
* **Tab 3: Playground & Model Export**
  * Code Editor ช่องใส่ Prefix และ Suffix
  * ปุ่ม "Predict Middle" เรียกโมเดล Base + Adapter มา Predict โค้ดตรงกลาง
  * ปุ่ม "Save Adapter Only" (บันทึกเฉพาะไฟล์ LoRA น้ำหนัก < 50 MB)
  * ปุ่ม "Merge & Export Full Weights"
* **กฎล็อกขณะเทรน:** ปุ่ม **"Predict Middle" และ "Merge & Export" ถูก disable อัตโนมัติตลอดเวลาที่ training process วิ่ง** (กันแย่ง VRAM กลางเกม; การ merge ยังกิน RAM ~2× น้ำหนักโมเดล) — ปลดล็อกเมื่อเทรนเสร็จหรือ abort แล้วเท่านั้น

---

## 5. กฎเหล็กการทำงาน (Hard Guardrails & Safety Policy)

| ด้าน | ขีดจำกัดความปลอดภัย (Safety Hard Cap) | กลไกการบังคับใช้ในระบบ |
|---|---|---|
| **VRAM Overload** | ห้ามใช้ Batch Size เกิน 1 ต่ออุปกรณ์ | ล็อกค่าใน UI หรือบังคับ Override ใน Trainer Script |
| **Context Length** | ล็อกสูงสุดไม่เกิน 2048 tokens | Slider ใน UI กำหนดค่าเพดานสูงสุดไว้ที่ 2048 (Default: 1024) |
| **Disk Exhaustion** | พื้นที่ว่างในดิสก์ต้อง $\ge 20 \text{ GB}$ | `hardware.py` คืน `insufficient_disk` ตั้งแต่ pre-flight → บล็อกปุ่ม Run ทันที |
| **Checkpoint Bloat** | เก็บไฟล์ Checkpoint ไม่เกิน 2 ชุด | ตั้งค่า `save_total_limit=2` ใน `TrainingArguments` |
| **Driver Deadlock** | ป้องกันค้างบนระดับ Level Zero Runtime | บังคับใช้ `mp.set_start_method("spawn", force=True)` ที่ `app.py` |
| **Process Stacking** | ห้ามรันงานเทรนซ้อนกันเด็ดขาด | เปิดใช้ Gradio Queue: `app.queue(default_concurrency_limit=1)` |
| **Loss Corruption** | loss เป็น NaN/Inf 3 ครั้งติด → หยุดเทรนทันที | `StreamToQueueCallback` ตรวจค่าแล้วสั่ง abort อัตโนมัติ |
| **Orphan Process** | ห้ามปล่อย worker เป็น zombie หลังปิด app | Exit handler ที่ `app.py` → terminate child + Escalation SIGTERM→SIGKILL |
| **VRAM Contention** | ห้าม inference/merge ชนกับ training | ปุ่ม Playground/Merge ถูกล็อกอัตโนมัติตลอดเวลาเทรนวิ่ง |
| **Network Exposure** | ไม่เปิดพอร์ตสู่ LAN | Gradio bind `localhost:7860` เท่านั้น (ไม่มี auth ในขอบเขต) |

---

## 6. ลำดับขั้นการพัฒนาโครงการ (Implementation Roadmap)

> รวมประมาณ **14 วัน** (เดิม 12 — ขยายด้วย Compat Spike + pytest + Evaluation)

### Phase 1: Runtime Baseline & Compat Spike (วันที่ 1–3)
- [ ] ตรวจสอบ Intel Compute Runtime / Level Zero บน EndeavourOS (เครื่องมี `/dev/dri` แล้ว — ยืนยันด้วย `clinfo` หรือ `sycl-ls`)
- [ ] **สร้าง venv Python 3.11 หรือ 3.12** (ห้ามใช้ Python 3.14 ของระบบ)
- [ ] ติดตั้ง torch XPU wheel ผ่าน `--index-url https://download.pytorch.org/whl/xpu` + ทดสอบ `torch.xpu.is_available() == True` บน Arc B580
- [ ] **Compat Spike:** ติดตั้ง transformers / peft / trl / datasets / accelerate รุ่นที่เข้าคู่กัน → **บันทึกลง `constraints.txt`**
- [ ] **Smoke test เรียก `SFTTrainer` ตัวจริง** (ยืนยัน API รุ่นใหม่: `processing_class=`, trl ≥ 0.12)
- [ ] เขียนสคริปต์ทดสอบสร้าง Tensor เล็ก ๆ และรัน Forward/Backward pass บน XPU
- [ ] สร้างไฟล์ `configs/fim_registry.json` และ `configs/safe_defaults.py`

### Phase 2: โมดูลคำนวณและประมวลผลข้อมูล (วันที่ 4–5)
- [ ] พัฒนา `core/hardware.py` (status ครบ 4 ค่า: ready / no_xpu / insufficient_ram / insufficient_disk)
- [ ] พัฒนา `core/estimator.py` (ที่มาของ $P$ + fallback + สูตรคำนวณ)
- [ ] พัฒนา `core/dataset_builder.py` (กติกา FIM: rate 0.5, PSM, boundary บรรทัด, EOS, seed 42)
- [ ] เขียน **pytest** ทดสอบการสับคำแบบ Prefix/Suffix/Middle (PSM format, boundary, EOS, ความยาว)
- [ ] เขียน **pytest** ทดสอบ estimator thresholds (Safe/Warning/Blocked/insufficient_disk/RAM)

### Phase 3: Subprocess Training Pipeline & IPC (วันที่ 6–8)
- [ ] สร้าง `core/trainer_worker.py` ประกอบ LoRA + SFTTrainer (+ seed, warmup, NaN guard, atomic checkpoint)
- [ ] สร้าง `core/ipc_bridge.py` (message protocol 4 type + watchdog + abort escalation)
- [ ] ติดตั้ง `StreamToQueueCallback` เพื่อส่ง `metric/log/status/error` ผ่าน `multiprocessing.Queue`
- [ ] เขียน **pytest** ทดสอบ IPC protocol, watchdog (จำลอง subprocess ตายเงียบ), abort escalation
- [ ] ทดสอบสั่งรัน Subprocess ผ่านคำสั่ง Python ปกติโดยยังไม่ต่อ UI
- [ ] ทดสอบระบบส่งคำสั่ง Abort และยืนยันว่า VRAM คืนกลับสู่ระบบสมบูรณ์

### Phase 4: Gradio UI & บูรณาการระบบ (วันที่ 9–11)
- [ ] พัฒนา Layout 3 แท็บใน `ui/dashboard.py` + `ui/components.py`
- [ ] เชื่อมต่อปุ่ม Run Pre-flight เข้ากับ `estimator.py` (แสดงสาเหตุที่ block ชัดเจน)
- [ ] เชื่อมต่อปุ่ม Start/Stop เข้ากับ Subprocess Controller ผ่าน `ipc_bridge.py`
- [ ] ทำระบบ Real-time Loss + LR Plot ผ่าน `gr.Timer`
- [ ] ล็อกปุ่ม Playground/Merge อัตโนมัติระหว่างเทรน + exit handler ตอนปิด app
- [ ] bind `localhost:7860` + เพิ่มหน้าต่าง Playground ทดสอบ Inference

### Phase 5: End-to-End, Evaluation & Docs (วันที่ 12–14)
- [ ] รันการเทรนจริงด้วยโมเดล `Qwen/Qwen2.5-Coder-0.5B` บน Dataset สั้น ๆ (500 Steps)
- [ ] **รัน `eval.py`: base vs fine-tuned (FIM Exact Match + Token F1 บน held-out)** — ต้องดีขึ้นจึงถือว่าผ่าน
- [ ] ทดสอบกดปุ่ม Abort กลางคันเพื่อยืนยันว่าไม่มี Memory Leak หรือ Zombie Process
- [ ] ทดสอบส่ง Dataset ขนาดใหญ่เพื่อตรวจสอบระบบ Disk Limiter
- [ ] Calibrate สูตร estimator เทียบ VRAM จริง (คลาดเคลื่อน ≤ 10%)
- [ ] เขียน `README.md` (ภาษาไทย: ติดตั้ง driver, ตั้งค่า venv, เริ่มใช้งาน)
- [ ] Freeze `requirements.txt` ให้ตรงกับ `constraints.txt`

---

## 7. ชุดคำสั่งและ Dependencies ตั้งต้น (`requirements.txt`)

```text
# Base Deep Learning Core
# NOTE: torch ไม่อยู่ในไฟล์นี้ — ติดตั้งแยกผ่าน XPU index (ดูคำเตือนด้านล่าง)
transformers>=4.44.0
peft>=0.12.0
trl>=0.12.0            # สำคัญ: API รุ่นใหม่ใช้ processing_class= — ห้ามต่ำกว่า 0.12 (tokenizer= ถูกแทนที่แล้ว)
datasets>=2.20.0
accelerate>=0.33.0

# UI & Monitoring
gradio>=4.40.0
psutil>=5.9.0

# Numerical & Data Processing
numpy>=2.0.0           # เดิมเขียน <2.0.0 ใช้กับ Python 3.12+ ไม่ได้ — ทดสอบจริงใน Compat Spike แล้วค่อยล็อกค่า
pandas>=2.2.0

# Testing
pytest>=8.0.0
```

> ค่าที่ pin จริงทั้งหมดอยู่ใน **`constraints.txt`** ซึ่งได้มาจาก Compat Spike ใน Phase 1 — ห้ามเดาเวอร์ชันเอง

> **คำเตือนการติดตั้ง PyTorch XPU:**
> ต้องติดตั้ง PyTorch จากช่องทางของ Intel/PyTorch Upstream โดยตรงตามเวอร์ชันของไดรเวอร์ในระบบ (เช่น สั่งติดตั้งผ่าน Wheel ของ Intel หรือ nightly/stable build ที่ระบุรองรับ XPU) ห้ามติดตั้ง `torch` ตัวมาตรฐานจาก PyPI ปกติเพราะจะมองไม่เห็นการ์ดจอ Intel — **และต้องใช้ Python 3.11/3.12 เท่านั้น** (wheel XPU รองรับ Python 3.9–3.13)

---

## 8. เกณฑ์ความสำเร็จ (Success Criteria & Evaluation)

เครื่องมือวัดผล: **`eval.py`** รัน 2 ครั้ง — ครั้งที่ 1 *ก่อนเทรน* (base model) และครั้งที่ 2 *หลังเทรนเสร็จ* (base + LoRA adapter) บนชุด held-out (5–10% ที่แยกไว้ตั้งแต่ต้น)

1. **Loss Curve:** ค่า loss ลดลงต่อเนื่องภายใน 500 steps และค่าสุดท้ายต่ำกว่าจุดเริ่มต้นอย่างมีนัยสำคัญ
2. **FIM Accuracy (เกณฑ์หลัก):**
   * **Exact Match %** — ทำนาย middle ตรงกับ ground truth เป๊ะ
   * **Token-level F1** (ตัวประกอบ) — โค้ดไม่ตรงเป๊ะแต่ถูกส่วนใหญ่
   * ผลลัพธ์ของ fine-tuned **ต้องดีกว่า base model** จึงถือว่าผ่าน
3. **Qualitative Playground:** ตัวอย่างจริง 3–5 ตัวอย่างที่ base model เดาไม่ได้ → fine-tuned ต้องเดาได้ดีขึ้นอย่างเห็นได้ชัด

**ไม่ผ่านเกณฑ์ข้อใดข้อหนึ่ง = ยังไม่ถือว่าโปรเจกต์เสร็จ** — กลับไปตรวจ pipeline (dataset / กติกา FIM / hyperparameter) ก่อน
