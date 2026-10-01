# Phase 4 Design: Gradio UI & System Integration

วันที่: 2026-10-01
สถานะ: approved by user (Section 1 + Section 2 ผ่านใน brainstorming session)
ที่มา: `plan.md` §4 โมดูล 6–7, §6 Phase 4, §5 Hard Guardrails

## 1. Context & Intent

สร้าง Gradio UI เป็น subsystem ใหม่ (`ui/` + `app.py`) เป็นส่วนต่อจาก Phase 3 — ผู้ใช้_single-user บนเครื่องเดียวกัน` ตั้งค่า fine-tune → pre-flight → เริ่ม/หยุด training → ดู loss สด → ทดสอบ inference → export adapter/weights ได้โดยไม่ต้องแตะ command line

**การตัดสินใจจาก user (บันทึกไว้ใน brainstorm):**
1. Tab 1 show **ครบทั้ง 9 ฟิลด์** ของ `run_training` config (ไม่ใช่แค่ 4 ตาม plan)
2. ข้อความ UI ทั้งหมดเป็น **English**
3. Approach B: เพิ่ม `ui/controller.py` แยกจาก plan §2 (4 ไฟล์แทน 3) — เหตุผล: state logic ต้อง test ได้ด้วย fake ตาม pattern Phase 1–3

**Success criteria:** plan.md §6 Phase 4 ครบ 6 ข้อ + guardrails §5 ทุกข้อมีกลไกบังคับใช้จริง

## 2. Scope

**อยู่ในขอบเขต:** `app.py`, `ui/__init__.py`, `ui/components.py`, `ui/dashboard.py`, `ui/controller.py`, `tests/test_ui_controller.py`, `predict_middle()` ใน `core/trainer_worker.py`, pre-flight ต่อ `hardware`+`estimator`, export flow (`exports/`)

**ไม่อยู่:** `eval.py` (Phase 5), `save_steps` field (ใช้ `SAVE_STEPS` จาก safe_defaults), auth/LAN (bind localhost เท่านั้นตาม §5), การปรับแต่ง theme/branding

## 3. Architecture

### 3.1 ไฟล์

```
app.py                      # entry: mp.set_start_method("spawn", force=True),
                            # สร้าง TrainingController + dashboard, exit handler → controller.exit(),
                            # launch(server_name="127.0.0.1", server_port=7860)
                            # + demo.queue(default_concurrency_limit=1)
ui/__init__.py
ui/controller.py            # TrainingController — pure Python, ห้าม import gradio
ui/components.py            # builder functions ล้วน — รับ raw data ตาม data contract (§4.3)
ui/dashboard.py             # 3 tabs + wire events → controller — ไม่มีสถานะฝังอยู่
core/trainer_worker.py      # เพิ่ม predict_middle(config, prefix, suffix) (spawn target ตัวเดิม)
tests/test_ui_controller.py # controller tests ด้วย fake Process/Queue
```

### 3.2 TrainingController (pure Python)

```python
class TrainingController:
    def __init__(self, *, queue_factory=mp.Queue, process_factory=mp.Process)
    def preflight(config: dict, user_params_b: float | None) -> EstimateResult
    def start(config: dict) -> str          # คืน "started" หรือข้อความ error (ยังไม่ raise)
    def abort() -> bool                     # abort_process + ตั้ง last_status="aborted" ทันที
    def tick() -> TickSnapshot              # เรียกทุก 1 วิ โดย gr.Timer
    def exit() -> None                      # app ปิด → terminate child (SIGTERM→SIGKILL)
    @property
    def training_active(self) -> bool
```

- `TickSnapshot` (NamedTuple): `status: str`, `metrics: list[dict]` (raw metric_msg), `logs: list[str]`, `error: str | None`, `watchdog: str | None`, `training_active: bool`
- ทุกการ mutate ครอบด้วย `threading.Lock` (Timer thread vs button handlers)
- **ไม่ import gradio** — test ได้ด้วย fake process/queue (duck-typed เหมือน Phase 3)

### 3.3 Data flow

1. **Start:** `validate_config(config)` → สร้าง `mp.Queue` + `mp.Process(target=run_training, args=(config, q))` → spawn
2. **Tick ทุก 1 วิ:** drain queue แบบ non-blocking (`get_nowait` จน Empty):
   - `metric` → append history (ข้อ 4)
   - `log` → append บรรทัด `[LEVEL] text`
   - `status` → state machine `starting→training→saving→finished|aborted` — **dedup ค่าซ้ำ** (Phase 3 ส่ง `finished` ซ้ำ 2 ครั้ง: `on_train_end` + `run_training`)
   - `error` → `error` field + log พร้อม traceback เต็ม
   - ทุก message ผ่าน `ipc_bridge.validate_message` ก่อนใช้ (ค้าง msg ที่ไม่ผ่านเป็น error)
   - หลัง drain: `watchdog_error(process, last_status)` — คืน error → `watchdog` field (zombie banner)
3. **Abort:** `ipc_bridge.abort_process(process)` → **ตั้ง `last_status = "aborted"` ทันที** เพราะ process ถูก kill จะไม่ทันส่ง status เอง (กัน false alarm จาก watchdog — ส่งไม้ต่อจาก Phase 3)
4. **Terminal status** (`finished`/`aborted`) + process จบ → `training_active=False` → ปลดล็อก Playground/Merge
5. **Exit handler:** `atexit` + signal handler → `controller.exit()` → `abort_process` → กัน orphan/XPU context leak (§5 Orphan Process)

## 4. UI Design (English labels)

### Tab 1: Configuration & Pre-flight
| Field | Widget | Default |
|---|---|---|
| Model | Dropdown (allow_custom_value) | `Qwen/Qwen2.5-Coder-0.5B` |
| Parameters (B) | Number | ว่าง (บังคับเมื่อ resolve ไม่ได้ — ไม่กรอก = blocked) |
| Dataset ID | Textbox | `smangrul/hf-stack-v1` |
| Code column | Textbox | `content` |
| FIM registry | Dropdown | `qwen` (qwen/starcoder/deepseek) |
| LoRA rank | Slider | 8 (8/16/32) |
| Max seq length | Slider 64–**2048** | 1024 (cap §5) |
| Max steps | Number | 500 |
| Code limit | Number | 8192 |
| Output dir | Textbox | `data_cache/finetune_run` |

- ปุ่ม **Run Environment Check** → `controller.preflight` = `hardware.inspect(output_dir)` → `estimator.estimate(hardware, model_id=..., user_params_b=..., seq_length=...)`
- ผลลัพธ์: **VRAM Gauge** (Safe/Warning/Blocked ตาม `verdict`) + `reason` + ตาราง breakdown (`weights/trainable/activations/overhead/total/free`) + `spec_source`
- `verdict != safe/warning` (เช่น `blocked`/`no_xpu`/`insufficient_disk`) → **Start disabled**

### Tab 2: Training Mission Control
- **Start Fine-Tuning** — disabled เมื่อ `training_active` หรือ preflight ยังไม่ผ่าน; ผล `start()` เป็น error → ข้อความแดงใต้ปุ่ม
- **Abort Process** — enabled เฉพาะขณะ training
- **Status badge**: Idle / Training / Saving / Finished / Aborted (สีเขียว/เหลือง/แดง)
- **Loss + LR plot** (`gr.Plot`): อัปเดตจาก `tick().metrics` ทุก 1 วิ — component รับ raw list แล้ววาดเอง
- **Live Log** (append-only Textbox/Code): log + error traceback โชว์ครบทุกบรรทัด

### Tab 3: Playground & Export
- Prefix / Suffix code inputs + **Predict Middle** → spawn subprocess `predict_middle(config, prefix, suffix)` (โหลด base + adapter จาก checkpoint ล่าสุด → FIM predict → พิมพ์ผล → จบ process)
- **Save Adapter Only** → คัดลอกไฟล์ adapter จาก `checkpoint-*` ล่าสุด (sorted by step) → `exports/<run-name>/` — เป็น file op อย่างเดียว ไม่ spawn, ไม่กิน VRAM
- **Merge & Export Full Weights** → spawn subprocess (load model + adapter → `merge_and_unload` → save `exports/`)
- **ล็อก Predict Middle + Merge & Export** ตลอดเวลา `training_active` (§5 VRAM Contention — merge กิน RAM ~2×)

### Thread/Concurrency
- `demo.queue(default_concurrency_limit=1)` (§5 Process Stacking)
- `gr.Timer(value=1000, active=True)` → `tick()` ทุก 1 วินาที
- ปุ่ม handlers + Timer ต่าง thread → `threading.Lock` ใน controller

## 5. Interfaces ที่ใช้ (ของจริงใน repo)

- `core/hardware.inspect(output_dir=".") -> dict` (contract: status/device_name/total_vram_gb/free_vram_gb/ram_available_gb/disk_free_gb)
- `core/estimator.estimate(hardware, *, model_id, user_params_b, batch_size, seq_length) -> EstimateResult(verdict, reason, total_required_gb, weights_gb, trainable_gb, activations_gb, overhead_gb, free_vram_gb, spec_source)`
- `core/ipc_bridge`: `validate_message`, `watchdog_error(process, last_status)`, `abort_process(process)`, `TRAINING_STATUSES`, `TERMINAL_STATUSES`
- `core/trainer_worker.run_training(config, queue)` (spawn target เดิม) + เพิ่ม `predict_middle(...)`
- `configs/safe_defaults`: ค่า default ทุกช่อง (ห้าม hardcode ซ้ำ — เรียกจาก constants)
- gradio 6.29 API ตรวจสอบแล้ว: `gr.Timer` ✓, `queue(default_concurrency_limit=)` ✓, `gr.Plot` ✓

## 6. Error Handling

| กรณี | พฤติกรรม |
|---|---|
| config ผิด (validate_config raise) | ข้อความแดงใต้ปุ่ม Start — ไม่ spawn |
| preflight blocked | Start disabled พร้อมแสดง `reason` |
| child raise | message `error` จาก queue → banner + traceback ใน log |
| process ตายเงียบ (zombie) | `watchdog_error` → red banner |
| abort กลางทาง | status → `aborted`, watchdog เงียบ (last_status terminal แล้ว) |
| msg รูปผิด (validate fail) | แสดงเป็น error ใน log — ไม่ใช้ค่า |
| ปิด app ระหว่างเทรน | exit handler → abort_process → ไม่มี orphan |

## 7. Testing

- `tests/test_ui_controller.py` (~8–10 tests, fake Process/Queue duck-typed): start validate+spawn, tick drain+dedup `finished`, status state machine, watchdog detect, abort → `last_status="aborted"`+คืน bool, error path, training_active lock rule, tick หลัง terminal ไม่ crash
- **ไม่ test** components/dashboard โดยตรง (thin glue) — ตรวจจริงด้วย `python app.py` (manual smoke: launch ได้, preflight แสดงผล, start/abort ทำงาน, plot เดิน)
- ของเดิม 53 tests ต้องเขียวตลอด

## 8. Deferred / เปิดไว้

- `save_steps` field ใน UI (ตอนนี้ใช้ safe_defaults)
- ปรับปรุง prompt หน้า README ส่วน driver install (Phase 5)
- warning `packing + sdpa` (พิจารณา `attn_implementation` ตอน inference/eval)
