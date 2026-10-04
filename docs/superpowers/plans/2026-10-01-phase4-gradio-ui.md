# Phase 4: Gradio UI & Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** สร้าง Gradio UI 3 แท็บ (Configuration/Pre-flight, Mission Control, Playground & Export) ที่คุม training subprocess ผ่าน `TrainingController` พร้อม pre-flight gate, real-time plot/log, abort, predict และ export

**Architecture:** Approach B (user เลือก): `app.py` (entry + exit handler + launch localhost:7860) + `ui/controller.py` (pure Python ไม่แตะ gradio — test ด้วย fake Process/Queue) + `ui/components.py` (builder ล้วน รับ raw data) + `ui/dashboard.py` (layout + wiring เท่านั้น) + `predict_middle`/export workers ใน `core/trainer_worker.py` (spawn target เดิม)

**Tech Stack:** gradio 6.29 (ตรวจแล้ว: `gr.Timer` ✓, `queue(default_concurrency_limit=)` ✓, `gr.Plot` ✓), matplotlib, torch XPU, multiprocessing spawn

**Spec:** `docs/superpowers/specs/2026-10-01-phase4-gradio-ui-design.md`

## Global Constraints
- Python 3.11 (ห้าม 3.14), รันจาก worktree: `../../.venv/bin/python -m pytest tests/ -v`
- **ค่าคงที่ทุกค่ามาจาก `configs/safe_defaults.py`** — ห้าม hardcode ซ้ำ (widget defaults, caps, ports ที่มีใน safe_defaults)
- UI label ทั้งหมด **English** (user ตัดสินใจ)
- `mp.set_start_method("spawn", force=True)` ที่ `app.py` เท่านั้น (§5 Driver Deadlock)
- bind `127.0.0.1:7860` เท่านั้น; `demo.queue(default_concurrency_limit=1)` (§5 Network Exposure / Process Stacking)
- Max seq length slider: default 1024, maximum **2048** (`MAX_SEQ_LENGTH_CAP` — §5 Context Length); batch คงที่ 1 (ไม่มีช่องให้กรอก)
- `ui/controller.py` **ห้าม import gradio**; `ui/components.py` รับ raw data ห้ามตีความ (data contract §4.6)
- ล็อก Predict Middle + Merge & Export ตลอดเวลา `controller.training_active` (§5 VRAM Contention)
- ของเดิม 53 tests ต้องเขียวตลอด
- **Test-once (user ตัดสินใจ Phase 4):** เขียน tests + implement ครบทุก task (1–7) ก่อน แล้วรัน `pytest` ทีเดียวตอน Task 7 — **ห้ามรันระหว่างทาง**; หลัง run แรกถ้า fail ให้ debug + รันซ้ำได้ตามปกติ

## Review Focus
(ข้อ 1–5 = failure modes ที่ spec บอกไว้แต่เผลอหลุดง่าย — task เจ้าของต้องมี test คุม)

1. **Status ข้าม/ซ้ำหลัง terminal** — หลัง `finished`/`aborted` ห้ามมี status ใหม่มาทับ (expect: state machine ignore) → test ใน Task 1
2. **Process ตายแต่ queue ยังมี msg ค้าง** — ต้อง drain ให้หมดก่อนตัดสิน watchdog (expect: error จริงถูกส่งก่อน banner zombie) → test ใน Task 2
3. **Abort ก่อน start / ซ้ำสองครั้ง** — ต้องคืน `False`/ไม่ crash (expect: ไม่มี exception) → test ใน Task 2
4. **Predict ค้าง / child ตาย** — `run_predict` ต้อง timeout เป็น exception ไม่ใช่ hang (expect: RuntimeError ใน deadline) → test ใน Task 3
5. **Preflight blocked passthrough** — `no_xpu`/`insufficient_disk` ต้องเป็น verdict ที่ dashboard ใช้ปิด Start ได้ (expect: verdict == status เดิม + มี reason) → test ใน Task 2

---

### Task 1: `ui/controller.py` — Controller core (start + tick state machine)

**Files:**
- Create: `ui/__init__.py` (ว่าง)
- Create: `ui/controller.py`
- Test: `tests/test_ui_controller.py`

**Interfaces:**
- Consumes: `core.trainer_worker.run_training(config, queue)` (spawn target), `core.ipc_bridge.{metric_msg, log_msg, status_msg, error_msg, validate_message, TRAINING_STATUSES, TERMINAL_STATUSES}`, `configs.safe_defaults` (ไม่มีค่าใหม่)
- Produces (Task 2/5/6 ใช้ต่อ):
  - `TickSnapshot(NamedTuple)`: `status: str`, `metrics: list[dict]`, `logs: list[str]`, `error: str | None`, `watchdog: str | None`, `training_active: bool`
  - `class TrainingController`: `__init__(self, *, queue_factory=mp.Queue, process_factory=mp.Process)`; `start(config: dict) -> str` คืน `"started"` หรือข้อความ error; `tick() -> TickSnapshot`; property `training_active -> bool`
  - `_status` state machine: `idle → starting → training → saving → finished|aborted` — หลัง terminal **ห้ามรับ status ใหม่**

- [ ] **Step 1: เขียน `tests/test_ui_controller.py`** — helpers: `FakeProcess` (duck-typed: `start/is_alive/terminate/kill/join` + บันทึก target/args), `FakeQueue` (`put` เก็บ list, `get_nowait` คืน msg แล้วลบ — ว่าง → `queue.Empty`), `make_controller(messages=None)`:
  - `test_start_rejects_invalid_config`: config ขาด key → return str ที่ mention ชื่อ key; `factory` ไม่ถูกเรียก (ไม่มี spawn); `training_active is False`
  - `test_start_spawns_run_training`: config ครบ → `== "started"`; process factory ถูกเรียกด้วย `target=run_training` และ `args[0] is config`; `training_active is True` (status="starting")
  - `test_tick_drains_metric_and_log`: queue มี `metric_msg(...)` + `log_msg("INFO","x")` → snapshot.metrics == [metric dict ดิบ], logs == `["[INFO] x"]`; tick อีกรอบ (queue ว่าง) → ข้อมูลเดิมยังอยู่ (accumulated ไม่หาย)
  - `test_status_machine_and_dedup`: drain `starting, training, finished, finished, training` → `status == "finished"` (dedup + terminal ignore), `training_active is False`
  - `test_invalid_msg_becomes_error`: queue มี `{"type": "alien", ...}` → `error` ไม่ None + mention msg, ไม่เข้า metrics/logs, ไม่ raise

- [ ] **Step 2: implement ใน `ui/controller.py`** (ยังไม่รัน test — test-once):
  - `class TickSnapshot(NamedTuple)` ตาม fields ด้านบน
  - `TrainingController.__init__`: lock = `threading.Lock()`, `self._queue=None`, `self._process=None`, `self._status="idle"`, `self._metrics=[]`, `self._logs=[]`, `self._error=None`
  - `start(config)`: `validate_config(config)` ใน try/except → คืน f"config ขาด key: ..." (ไม่ spawn); else สร้าง queue/process ผ่าน factory (`process_factory(target=run_training, args=(config, q))`), `.start()`, `_status="starting"`, คืน `"started"`
  - `tick()`: ภายใต้ lock — drain `get_nowait` จน `queue.Empty`: `validate_message` ไม่ผ่าน → `_error`; `metric` → append `_metrics`; `log` → `_logs.append(f"[{level}] {text}")`; `error` → `_error = message` + log traceback บรรทัด; `status` → state machine (terminal แล้ว = ignore); คืน `TickSnapshot(_status, list, list, _error, None, training_active)` — **`watchdog` คืน `None` (Task 2 เติม)**
  - `training_active`: `self._process is not None and _status not in TERMINAL_STATUSES`
- [ ] **Step 3: Commit** — `git add ui/ tests/test_ui_controller.py && git commit -m "feat: TrainingController core (start + tick state machine)"`

---

### Task 2: Controller safety — `preflight` + `abort` + watchdog + `exit`

**Files:**
- Modify: `ui/controller.py`
- Test: ต่อใน `tests/test_ui_controller.py`

**Interfaces:**
- Consumes: Task 1 class; `core.hardware.inspect(output_dir) -> dict`, `core.estimator.estimate(hardware, *, model_id, user_params_b, batch_size, seq_length) -> EstimateResult(verdict, reason, total_required_gb, ...)`, `core.ipc_bridge.{watchdog_error, abort_process, TERMINAL_STATUSES}`
- Produces (Task 5/6 ใช้):
  - `preflight(config: dict, user_params_b: float | None) -> EstimateResult` — `inspect(output_dir=config.get("output_dir","."))` → `estimate(hardware, model_id=config["model_id"], user_params_b=user_params_b, seq_length=config["max_seq_length"])`
  - `abort() -> bool` — เรียก `abort_process(process)` แล้ว **ตั้ง `_status="aborted"` ทันที** (ส่งไม้ต่อ Phase 3: child ถูก kill ไม่ทันส่ง status → กัน watchdog false alarm)
  - `exit() -> None` — ถ้า process ยังอยู่ → `abort_process` (ถ้ามันยังไม่ตั้ง status ก็ตั้งให้) ; เรียกซ้ำได้ไม่ raise

- [ ] **Step 1: เขียน tests ต่อ**:
  - `test_preflight_passthrough_blocked`: monkeypatch `ui.controller` ให้ `hardware.inspect` คืน `{"status": "insufficient_disk", "free_vram_gb": 5.0}` → `preflight(...).verdict == "insufficient_disk"` และ `.reason` มี `20` (GB) — **Review Focus 5**
  - `test_preflight_safe_estimate`: inspect ready (VRAM/RAM/disk พอ) → `verdict in ("safe","warning")` + `total_required_gb > 0`
  - `test_abort_without_process_returns_false`: controller ใหม่ → `abort() is False` ไม่ raise — **Review Focus 3**
  - `test_abort_kills_and_marks_terminal`: start ด้วย FakeProcess(dies_on_terminate=False) → `abort() is True`; `terminate`+`kill` ถูกเรียก; `tick().watchdog is None`; `training_active is False` — **Review Focus 3**
  - `test_zombie_watchdog_after_drain`: start แล้ว make process dead + status ยัง `"training"` + queue มี `error_msg("boom","tb")` ค้าง → `tick().error` มี "boom" (drain ก่อน) **และ** `.watchdog` ไม่ None — **Review Focus 2**
  - `test_exit_terminates_child`: start แล้ว `exit()` → process `is_alive() is False`; เรียก `exit()` ซ้ำไม่ raise
- [ ] **Step 2: implement** — `preflight` (composition สองบรรทัด), `abort` (lock + `abort_process` + ตั้ง terminal + คืน bool), `exit`, และ **เติม watchdog ใน `tick()`**: หลัง drain แล้วเรียก `watchdog_error(self._process, self._status)` คืน msg → ใส่ field `watchdog`
- [ ] **Step 3: Commit** — `git commit -m "feat: controller preflight + abort escalation + watchdog"`

---

### Task 3: Predict pipeline — `build_fim_prompt` + `predict_middle` + `run_predict`

**Files:**
- Modify: `core/trainer_worker.py` (เพิ่ม 2 functions)
- Modify: `ui/controller.py` (เพิ่ม module-level `run_predict`)
- Test: ต่อใน `tests/test_ui_controller.py` (run_predict) + `tests/test_trainer_worker.py` (prompt/checkpoint)

**Interfaces:**
- Consumes: `configs/fim_registry.json` (key "qwen"), `latest checkpoint` ใน `output_dir` (`checkpoint-<step>` ของ AtomicSaveTrainer), `LORA_TARGET_MODULES`, `ipc_bridge.{log_msg, error_msg}`
- Produces (Task 6 ใช้):
  - `core.trainer_worker.latest_checkpoint(output_dir: str | Path) -> Path` — เลือก `checkpoint-\d+` ที่ **เลขมากสุด** (numeric ไม่ใช่ lexical); ไม่มี → `raise ValueError`
  - `core.trainer_worker.build_fim_prompt(prefix: str, suffix: str, *, fim_tokens: dict, eos: str) -> str` = `f"{fim_tokens['prefix']}{prefix}{fim_tokens['suffix']}{suffix}{fim_tokens['middle']}"` (ไม่ใส่ eos — model generate ต่อจาก middle_tok เอง)
  - `core.trainer_worker.predict_middle(config: dict, prefix: str, suffix: str, queue) -> None` — spawn target: โหลด tokenizer + `AutoModelForCausalLM` (dtype=torch.bfloat16, sdpa) + `PeftModel.from_pretrained(base, str(latest_checkpoint(config["output_dir"])))` → prompt → `generate(max_new_tokens=256, do_sample=False)` → decode เฉพาะส่วนที่ต่อจาก prompt → `queue.put(log_msg("INFO", result))`; ผิด → `queue.put(error_msg(...))` + raise
  - `ui.controller.run_predict(config, prefix, suffix, *, timeout: float = 180.0, process_factory=mp.Process) -> str` — spawn `predict_middle(args=(config,prefix,suffix,q))`, drain `get_nowait`+sleep 0.05 จน `log_msg` (คืน text) / `error_msg` (raise RuntimeError) / เกิน timeout (`raise RuntimeError("predict timeout ...")`) — **Review Focus 4**

- [ ] **Step 1: เขียน tests**:
  - `tests/test_trainer_worker.py`: `test_latest_checkpoint_numeric_order` (tmp: `checkpoint-2`, `checkpoint-100` → คืน `-100`; ไม่มี → `pytest.raises(ValueError)`), `test_build_fim_prompt_exact` (qwen tokens + prefix/suffix → string เป๊ะ ไม่มี middle/eos เสริม)
  - `tests/test_ui_controller.py`: `test_run_predict_success` (fake process + FakeQueue ใส่ `log_msg("INFO","def f(): pass")` → คืน `"def f(): pass"`), `test_run_predict_timeout_raises` (queue ว่าง + timeout=0.05 → `pytest.raises(RuntimeError)`), `test_run_predict_error_raises` (ใส่ `error_msg("no adapter","")` → `RuntimeError` message มี "no adapter")
- [ ] **Step 2: implement** ทั้ง 3 functions (run_predict ใช้ sleep 0.05 loop + deadline; `predict_middle` ตาม flow ใน Interfaces — generation greedy)
- [ ] **Step 3: Commit** — `git commit -m "feat: FIM predict pipeline (predict_middle + run_predict)"`

---

### Task 4: Export workers — `save_adapter_only` + `merge_export`

**Files:**
- Modify: `core/trainer_worker.py`
- Test: ต่อใน `tests/test_trainer_worker.py`

**Interfaces:**
- Consumes: `latest_checkpoint` (Task 3)
- Produces (Task 6 ใช้):
  - `save_adapter_only(output_dir: str | Path, *, exports_dir: str | Path = "exports") -> Path` — copy `checkpoint-<n>/adapter_config.json` + `adapter_model.safetensors` (+ `*.safetensors.index.json` ถ้ามี) จาก checkpoint ล่าสุด → `exports/<output_dir.name>/`; ไม่มี checkpoint → `raise ValueError`; คืน path ที่ copy ไป
  - `merge_export(config: dict) -> Path` — spawn target: โหลด base + adapter → `merge_and_unload()` → `save_pretrained(exports/<name>-merged/)`; **ไม่มี unit test** (body = โหลดโมเดลจริง — ตรวจจริงตอน Phase 5 หลังเทรนจบ; task นี้ขอแค่ test ของ `save_adapter_only` + import check ของ `merge_export`)

- [ ] **Step 1: เขียน tests**: `test_save_adapter_only_copies` (tmp output_dir มี `checkpoint-10/adapter_config.json` + `adapter_model.safetensors` → คืน `exports/<name>/` ที่มีไฟล์ครบ), `test_save_adapter_only_no_checkpoint_raises`, `test_merge_export_importable` (`callable(wa.merge_export)`)
- [ ] **Step 2: implement** — `save_adapter_only` ด้วย `shutil.copy2`; `merge_export` ตาม flow ข้างบน (imports PeftModel เพิ่ม)
- [ ] **Step 3: Commit** — `git commit -m "feat: export workers (adapter copy + merge target)"`

---

### Task 5: `ui/components.py` — metric plot + verdict style

**Files:**
- Create: `ui/components.py`
- Test: `tests/test_ui_components.py`

**Interfaces:**
- Consumes: raw dict จาก `ipc_bridge.metric_msg`, verdict str จาก `EstimateResult.verdict` (+ passthrough statuses)
- Produces (Task 6 ใช้):
  - `build_metric_plot(metrics: list[dict]) -> matplotlib.figure.Figure` — 2 axes: loss (ซ้าย) กับ lr (ขวา) เทียบ `step`; ว่าง → figure พร้อม axes เปล่า (ไม่ raise)
  - `verdict_style(verdict: str) -> tuple[str, str]` — `("safe", "#22c55e")`-style: safe→เขียว `#16a34a`, warning→เหลือง `#ca8a04`, อย่างอื่น (blocked/no_xpu/insufficient_*) →แดง `#dc2626`; คืน `(verdict, color)` (label = verdict เดิม ห้ามแปลง — data contract)
  - (Log Viewer = `gr.Textbox` ธรรมดาใน dashboard — ไม่มี function เดี่ยว)

- [ ] **Step 1: เขียน tests**: `test_build_metric_plot_two_axes` (fig.axes == 2; line axes[0] มี x=steps,y=loss ตรง input), `test_build_metric_plot_empty` (ว่าง → ไม่ raise, axes ครบ), `test_verdict_style_colors` (safe→`#16a34a`, warning→`#ca8a04`, `"blocked"`/`"no_xpu"`→`#dc2626`; label คงเป็น verdict เดิม) — ใช้ `matplotlib.use("Agg")` ใน test
- [ ] **Step 2: implement** (`matplotlib.pyplot` — สร้าง fig แล้ว `plt.close` ไม่ต้องใน component ปล่อย Gradio จัดการ)
- [ ] **Step 3: Commit** — `git commit -m "feat: ui/components metric plot + verdict style"`

---

### Task 6: `ui/dashboard.py` — 3 tabs + wiring

**Files:**
- Create: `ui/dashboard.py`
- Test: `tests/test_ui_dashboard.py`

**Interfaces:**
- Consumes: ทุกอย่าง — `TrainingController.start/preflight/abort/tick/run_predict/training_active` (Task 1–3), `save_adapter_only`, `merge_export` (Task 4), `build_metric_plot/verdict_style` (Task 5), `safe_defaults` ทุก default
- Produces (Task 7 ใช้): `build_dashboard(controller: TrainingController) -> gr.Blocks`

- [ ] **Step 1: เขียน tests**: `test_build_dashboard_structure` — `demo = build_dashboard(FakeController())` → `isinstance(demo, gr.Blocks)`; ป้าย tab ทั้ง 3 = `["Configuration & Pre-flight", "Training Mission Control", "Playground & Export"]`; มี `gr.Timer` อย่างน้อย 1 ตัว; slider `max_seq_length` มี `maximum == MAX_SEQ_LENGTH_CAP` และ `value == MAX_SEQ_LENGTH_DEFAULT`
- [ ] **Step 2: implement** — `build_dashboard(controller)`:
  - **Tab 1**: widgets 9 ฟิลด์ + `Parameters (B)` (ค่า default จาก safe_defaults ทั้งหมด: model `DEFAULT_MODEL_ID`, dataset `DEFAULT_DATASET_ID`, column `DEFAULT_DATASET_COLUMN`, fim key `"qwen"`, lora rank `LORA_RANK_DEFAULT` slider choices [8,16,32], max_seq slider cap `MAX_SEQ_LENGTH_CAP`, max_steps `MAX_STEPS`, code_limit `TRAIN_CODE_LIMIT`, output_dir `"data_cache/finetune_run"`) + ปุ่ม `Run Environment Check` → `controller.preflight(...)` → gauge (`verdict_style`) + reason + ตาราง breakdown + **บันทึก verdict ใน `gr.State`**; Start ถูก disable เมื่อ verdict ไม่ใช่ `safe`/`warning`
  - **Tab 2**: `Start Fine-Tuning` (gather config → `validate` ผ่าน `start()` → ข้อความแดงใต้ปุ่มถ้า error), `Abort Process`, status badge, `gr.Plot`, live log textbox; `gr.Timer(1.0)` → `tick()` → อัปเดต plot (จาก `snapshot.metrics`), log, badge, error banner, interactivity ปุ่มทุกปุ่ม (locks ตาม `training_active`)
  - **Tab 3**: Prefix/Suffix `gr.Code`, `Predict Middle` (เรียก `run_predict` sync — บล็อกได้เพราะ predict ถูกล็อกไม่ให้วิ่งพร้อม training), `Save Adapter Only` (`save_adapter_only` → ข้อความ path), `Merge & Export` (เรียก `merge_export` sync + `gr.Button loading`); **Predict/Merge `interactive=not training_active`** ทุก tick
  - ฟังก์ชัน helper ในไฟล์เดียวกัน: `_collect_config(...) -> tuple[dict, float | None]` (9 keys + user_params_b)
- [ ] **Step 4: Commit** — `git commit -m "feat: ui/dashboard 3 tabs + controller wiring"`

---

### Task 7: `app.py` entry + Integration verification + full suite

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py` (เล็กมาก) + verification จริง

**Interfaces:**
- Consumes: `TrainingController`, `build_dashboard`
- Produces: `main() -> None`

- [ ] **Step 1: เขียน test**: `test_app_import_does_not_launch` — `import app` (ใน subprocess หรือโดยตรง) แล้ว `callable(app.main)` — **ห้าม** เปิด server ตอน import
- [ ] **Step 2: implement `app.py`**: `main()` = `mp.set_start_method("spawn", force=True)` → `TrainingController()` → `demo = build_dashboard(c)` → `atexit.register(c.exit)` + `signal.SIGTERM/SIGINT` handler → `c.exit()` → `demo.queue(default_concurrency_limit=1)` → `demo.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=False)`; `if __name__ == "__main__": main()`
- [ ] **Step 3: Integration จริง (รัน app ได้ ไม่ใช่ pytest)**:
  - `timeout 25 ../../.venv/bin/python app.py &` แล้ว `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860` → `200` → kill
  - Manual checklist (report ผล): เปิด UI ได้ / Run Environment Check โชว์ gauge+reason / Start training 6 steps จาก UI เห็น plot เดิน + log / Abort กลางทาง → badge Aborted + VRAM คืน / Predict+Merge ถูกล็อกขณะเทรน / ปิด app กลางเทรน → ไม่มี orphan (`pgrep -af trainer_worker` ว่าง)
- [ ] **Step 4: Full suite — จุดรัน pytest ทีเดียวของทั้ง phase (test-once):** `../../.venv/bin/python -m pytest tests/ -v` → 53 เดิม + ~19 ใหม่ = **72** ทั้งหมดเขียว — fail ตรงไหน debug + รันซ้ำได้ตามปกติจนกว่าจะเขียวหมด
- [ ] **Step 5: Commit** — `git add app.py tests/ && git commit -m "feat: app.py entry + integration verification"`

## อัปเดตสเปค 2026-10-04 — ML logic review

- `ui/components.build_metric_plot`: รับ list ผสม `metric` + `val_metric` — train series ใช้เฉพาะ dict ที่มี `"loss"` (ลำดับ loss → lr เดิมคงไว้), `val_loss` วาดเป็นจุด marker สีส้ม `#f97316` label `"val_loss"` บนแกน loss (มีเมื่อไหร่วาดเมื่อนั้น)
- `ui/controller._handle_message`: `mtype in ("metric", "val_metric")` → append เข้า `metrics` (ผ่าน validate ก่อนเหมือนเดิม)
