# Phase 3: Subprocess Training Pipeline & IPC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** สร้าง training subprocess ที่รัน SFTTrainer+LoRA บน XPU ใน process แยก ส่ง metric/status/error ผ่าน Queue ครบวงจร พร้อม watchdog และ abort escalation — ผ่านการทดสอบจริงโดยยังไม่ต่อ UI

**Architecture:** `run_training(config, queue)` เป็น target ของ `multiprocessing.Process` (spawn) — โหลด tokenizer (+FIM guard) → dataset streaming → FIM samples → โมเดล BF16/SDPA + LoRA → `AtomicSaveTrainer.train()` แล้วส่งข้อความ 4 ประเภท (`metric/log/status/error` จาก `core/ipc_bridge.py`) ลง `multiprocessing.Queue` ทุกจุดสำคัญ UI (Phase 4) จะ drain Queue + เรียก watchdog/abort ทุกวินาที

**Tech Stack:** Python 3.11, torch 2.14.1+xpu, trl 1.14.1 (`SFTConfig`/`SFTTrainer`/`TrainerCallback`), peft 0.21.1, transformers 5.18.0, datasets 5.0.1, multiprocessing (spawn)

**Spec:** `plan.md` — §6 Phase 3 (บรรทัด 268–274), §4 โมดูล 4–5 (172–204), §3 (54–87), §5 (231–244)

**Deviation ที่อนุมัติแล้ว (API จริง vs spec):**
- spec §3 สั่ง `ConstantLengthDataset` → trl 1.14.1 ถอดออกแล้ว → ใช้ `SFTConfig(packing=True, max_length=...)` (พฤติกรรมเดียวกัน: ต่อ samples แล้วตัดเป็นบล็อกคงที่)
- spec §4.4 สั่ง `warmup_ratio = 0.03` → transformers 5.18 เหลือแค่ `warmup_steps` → ตั้ง `warmup_steps = round(0.03 × max_steps)` (=15 เมื่อ max_steps=500) คงสัดส่วน 3% ไว้

## Global Constraints

- Python **3.11/3.12 เท่านั้น** — ห้าม Python ระบบ 3.14; torch ต้องรุ่น XPU (`--index-url https://download.pytorch.org/whl/xpu`)
- trl ≥ 0.12: เรียกด้วย `processing_class=` เท่านั้น — **ห้าม `tokenizer=`**; `SFTConfig.max_steps` เท่านั้น (transformers 5.x `train()` ไม่รับ `max_steps`)
- Optimizer `adamw_torch` (ห้าม `paged_adamw_8bit`); model = `dtype=torch.bfloat16` + `attn_implementation="sdpa"`; `gradient_checkpointing=True` เสมอ
- `per_device_train_batch_size = 1` (hard cap §5), `max_seq_length ≤ MAX_SEQ_LENGTH_CAP` (default 1024), `save_total_limit = 2`
- Hyperparams: grad_accum 8, lr 2e-4, warmup 3%, seed 42, save_steps 100, max_steps 500 — **ค่าทั้งหมดมาจาก `configs/safe_defaults.py` ห้าม hardcode ซ้ำ**
- Abort: `SIGTERM` → รอ 10 วิ → `SIGKILL`; NaN/Inf ติดต่อกัน 3 ครั้ง → abort + ส่ง `error`
- FIM: ห้าม `resize_token_embeddings` / `add_special_tokens` (token มีใน vocab แล้ว); fim_rate 0.5, PSM, seed 42 (คงเดิมจาก Phase 2)
- `multiprocessing.set_start_method("spawn", force=True)` ทุกจุดที่เปิด process (กัน Level Zero deadlock §5)
- เวอร์ชันแพ็กเกจตาม `constraints.txt` — ห้ามเปลี่ยน/เดาเอง
- ไม่เปิดพอร์ต/ไม่ bind host (Gradio = Phase 4)

## Review Focus

1. **Zombie detection** — process ตายเงียบระหว่าง `training` (ไม่ส่ง `finished`/`aborted` เอง) → watchdog ต้องคืน `error` message ทันที ไม่งั้น UI ค้างรอ Queue ตลอดกาล → `tests/test_ipc_bridge.py::test_watchdog_detects_zombie` (+ `last_status=None` ก็ต้องจับ)
2. **NaN streak semantics** — 3 ครั้งติดต → abort แต่ถ้ามี finite คั่น ต้อง reset นับใหม่ (ไม่งั้น loss กระเพื่อมปกติ = false positive หยุดเทรน) → `tests/test_trainer_worker.py::test_nan_guard_aborts_on_three_consecutive` + `::test_nan_guard_resets_on_finite`
3. **Abort escalation จริง** — `terminate()` แล้วยังไม่ตายใน 10 วิ ต้อง `kill()` จริง (คืน VRAM); ตายแล้วห้าม kill ซ้ำ → `tests/test_ipc_bridge.py::test_abort_escalates_to_kill` + `::test_abort_no_kill_when_terminate_works`
4. **status enum กันพัง** — state นอกชุด 5 ค่า (เช่น worker พิมพ์ `"done"`) ต้อง `ValueError` fail-fast ทันที แทนส่ง message รูปผิดให้ UI เดางาน → `tests/test_ipc_bridge.py::test_status_msg_rejects_unknown_state`
5. **Checkpoint atomic** — ทับ checkpoint เดิมแล้วไฟล์เก่าต้องถูกแทนที่หมด และไม่มี `*.saving` ค้างหลัง commit (กันไฟล์เสียถ้าโดน abort กลาง `save_steps`) → `tests/test_trainer_worker.py::test_commit_checkpoint_overwrites` + `::test_commit_checkpoint_moves_and_cleans`

---

### Task 1: `core/ipc_bridge.py` — Message Protocol + Watchdog + Abort

**Files:**
- Modify: `configs/safe_defaults.py` (เพิ่ม block "Subprocess / IPC")
- Create: `core/ipc_bridge.py`
- Test: `tests/test_ipc_bridge.py`

**Interfaces:**
- Consumes: `safe_defaults.ABORT_SIGTERM_TIMEOUT_S` (เพิ่มใน task นี้)
- Produces (task 4/5 และ Phase 4 ใช้ต่อ):
  - `TRAINING_STATUSES: tuple[str, ...] = ("starting", "training", "saving", "finished", "aborted")`
  - `metric_msg(step: int, loss: float, lr: float, epoch: float) -> dict` → `{"type": "metric", "step", "loss", "lr", "epoch"}`
  - `log_msg(level: str, text: str) -> dict` → `{"type": "log", "level", "text"}`
  - `status_msg(state: str) -> dict` → `{"type": "status", "state"}`; `state` นอก `TRAINING_STATUSES` → `raise ValueError`
  - `error_msg(message: str, traceback_text: str) -> dict` → `{"type": "error", "message", "traceback"}`
  - `validate_message(msg: object) -> bool` — ตรวจ type field + payload keys ครบ/ชนิดถูก
  - `watchdog_error(process, last_status: str | None) -> dict | None` — `process.is_alive()` เป็น False และ `last_status` ไม่ใช่ `"finished"`/`"aborted"` → คืน `error_msg` (zombie); ไม่งั้น `None`
  - `abort_process(process, *, timeout: float = ABORT_SIGTERM_TIMEOUT_S) -> bool` — `terminate()` → `join(timeout)` → ยัง alive → `kill()` → คืน `not process.is_alive()`
  - `process` = object ที่มี `is_alive()/terminate()/kill()/join(timeout)` (duck-typed — รับ `mp.Process` หรือ fake ใน test ได้)

- [ ] **Step 1: เขียน `tests/test_ipc_bridge.py`** — helper `FakeProcess(dies_on_terminate: bool)` นับจำนวน `terminate`/`kill`/`join(timeout)` ที่ถูกเรียก แล้ว assert:
  - `test_metric_msg_shape`: dict เป๊ะ + `validate_message` True
  - `test_log_and_error_msg`: shape ถูก; `error["traceback"]` มีข้อความ traceback เต็ม (assert substring)
  - `test_status_msg_states`: ครบทั้ง 5 ค่า; `status_msg("done")` → `pytest.raises(ValueError)`
  - `test_validate_message_rejects_bad`: ไม่ใช่ dict / ไม่มี `type` / payload key หาย → False
  - `test_watchdog_detects_zombie`: dead + `last_status="training"` → error message validate ผ่าน; dead + `last_status=None` → ยังได้ error (ตายก่อนส่ง status ใด ๆ ก็เป็น zombie)
  - `test_watchdog_silent_when_terminal`: alive → None; dead + `"finished"`/`"aborted"` → None
  - `test_abort_escalates_to_kill`: `dies_on_terminate=False` → terminate 1 ครั้ง, `join` ถูกเรียกด้วย `timeout=10.0`, kill 1 ครั้ง, คืน True
  - `test_abort_no_kill_when_terminate_works`: `dies_on_terminate=True` → kill 0 ครั้ง, คืน True

- [ ] **Step 2: รัน test — ต้อง FAIL (import ไม่เจอ)**

Run: `../../.venv/bin/python -m pytest tests/test_ipc_bridge.py -v`

- [ ] **Step 3: เพิ่ม `safe_defaults` + implement `core/ipc_bridge.py`**

```python
# configs/safe_defaults.py
# --- Subprocess / IPC ---
ABORT_SIGTERM_TIMEOUT_S: float = 10.0
```

โค้ด `ipc_bridge.py`: builders คืน dict รูปข้างบน, `status_msg` validate กับ `TRAINING_STATUSES`, `watchdog_error` ใช้ `{"finished", "aborted"}` เป็น terminal set, `abort_process` = terminate → join(timeout) → ถ้ายัง `is_alive()` → kill → join สั้น (1.0) → คืนผล

- [ ] **Step 4: รัน test — PASS ทั้ง 8**

- [ ] **Step 5: Commit**

```bash
git add configs/safe_defaults.py core/ipc_bridge.py tests/test_ipc_bridge.py
git commit -m "feat: core/ipc_bridge.py message protocol + watchdog + abort escalation"
```

---

### Task 2: `build_training_args()` — Hyperparameters จาก safe_defaults

**Files:**
- Modify: `configs/safe_defaults.py` (เพิ่ม block "Training hyperparameters")
- Create: `core/trainer_worker.py`
- Test: `tests/test_trainer_worker.py`

**Interfaces:**
- Consumes: `safe_defaults.MAX_STEPS/SAVE_STEPS/SEED/MAX_SEQ_LENGTH_DEFAULT/DEFAULT_BATCH_SIZE`
- Produces:
  - `safe_defaults` เพิ่ม: `GRADIENT_ACCUMULATION_STEPS: int = 8`, `LEARNING_RATE: float = 2e-4`, `WARMUP_RATIO: float = 0.03`, `SAVE_TOTAL_LIMIT: int = 2`, `SAVE_STEPS: int = 100`, `MAX_STEPS: int = 500`, `NONFINITE_ABORT_THRESHOLD: int = 3`, `LORA_RANK_DEFAULT: int = 8`, `LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")`, `TRAIN_CODE_LIMIT: int = 8192`
  - `build_training_args(output_dir: str, *, max_steps: int = MAX_STEPS, save_steps: int = SAVE_STEPS, max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT) -> SFTConfig` ใน `core/trainer_worker.py` — `max_seq_length` ต้องส่งถึง `SFTConfig.max_length` เพราะ packing ตัดที่ค่านี้ (ถ้า hardcode → UI ตั้งค่าใน Phase 4 ถูกเพิกเฉยเงียบ ๆ; review finding #1)

- [ ] **Step 1: เขียน `tests/test_trainer_worker.py`** — 2 tests:
  - `test_args_pin_hyperparams`: `args = build_training_args("data_cache/x")` แล้ว assert `learning_rate == 2e-4`, `warmup_steps == 15` (=round(0.03×500)), `gradient_accumulation_steps == 8`, `per_device_train_batch_size == 1`, `seed == 42`, `save_total_limit == 2`, `save_steps == 100`, `max_steps == 500`, `gradient_checkpointing is True`, `optim == "adamw_torch"`, `bf16 is True`, `max_length == 1024`, `max_length <= 2048`, `packing is True`, `dataset_text_field == "text"`
  - `test_args_overrides_for_smoke`: `build_training_args("out", max_steps=6, save_steps=2)` → `max_steps == 6`, `save_steps == 2`, `warmup_steps == 0`
- [ ] **Step 2: รัน test — FAIL (module ยังไม่มี)**
- [ ] **Step 3: implement** — เพิ่ม block safe_defaults ตาม Interfaces ข้างบน แล้วสร้าง `core/trainer_worker.py`: import `SFTConfig` จาก trl, คืน `SFTConfig(...)` ครบตาม assertion (ตั้ง `warmup_steps=round(WARMUP_RATIO * max_steps)` — transformers 5.x ไม่มี `warmup_ratio`; `save_strategy="steps"`, `logging_steps=1`, `report_to=[]`)
- [ ] **Step 4: รัน test — PASS ทั้ง 2**
- [ ] **Step 5: Commit** — `git add configs/safe_defaults.py core/trainer_worker.py tests/test_trainer_worker.py && git commit -m "feat: build_training_args จาก safe_defaults hyperparameters"`

---

### Task 3: Guardrails — `NanGuard`, `ensure_fim_tokens`, `commit_checkpoint`

**Files:**
- Modify: `core/trainer_worker.py` (ต่อจาก Task 2)
- Test: ต่อใน `tests/test_trainer_worker.py`

**Interfaces:**
- Consumes: `safe_defaults.NONFINITE_ABORT_THRESHOLD` (Task 2)
- Produces (Task 4/5 ใช้ต่อ):
  - `class NanGuard`: `__init__(threshold: int = NONFINITE_ABORT_THRESHOLD)`; `register(loss: float) -> bool` — คืน True เมื่อ non-finite ติดต่อกันครบ threshold (finite ตัวไหนก็ reset เป็น 0); `math.isfinite` ใช้ตัดสิน
  - `ensure_fim_tokens(tokenizer, fim_tokens: Iterable[str]) -> dict[str, int]` — ตรวจว่าทุก token อยู่ใน vocab จริง (`convert_tokens_to_ids` → id ไม่ None, `< len(tokenizer)`, ≠ unk, `convert_ids_to_tokens` roundtrip ตรง) — ขาด/ผิด → `raise ValueError` พร้อมชื่อ token (ห้าม train); คืน `{token: id}`; **ห้ามเรียก method resize ใด ๆ** (fake tokenizer ใน test ไม่มี method นี้ — เรียก = AttributeError)
  - `commit_checkpoint(tmp_dir: Path, final_dir: Path) -> None` — ย้าย `tmp_dir` → `final_dir` แบบ atomic (`os.replace`/`os.rename`); ถ้า `final_dir` มีอยู่แล้ว → เก็บของเดิมออกก่อนแล้วแทนที่; จบแล้วต้องไม่มี `tmp_dir` หรือ `*.saving` เหลือ
  - `class AtomicSaveTrainer(SFTTrainer)`: override `_save(self, output_dir=None, state_dict=None)` — `output_dir is None` → super; ไม่งั้น `super()._save(f"{output_dir}.saving", state_dict)` แล้ว `commit_checkpoint(Path(tmp), Path(output_dir))` (ไม่ต้องมี unit test แยก — ถูก exercise โดย commit_checkpoint test + integration run)

- [ ] **Step 1: เขียน tests ต่อใน `tests/test_trainer_worker.py`** — helper `FakeTokenizer(vocab: dict)` (methods: `convert_tokens_to_ids`, `convert_ids_to_tokens`, `decode`, `__len__`, `unk_token_id` — **ไม่มี** method resize):
  - `test_nan_guard_aborts_on_three_consecutive`: register(1.0)→False, nan→False, inf→False, `-inf`→True
  - `test_nan_guard_resets_on_finite`: nan, nan, register(0.5)→False, nan→False, nan→False (reset แล้วต้องนับใหม่ — ยังไม่ครบ 3)
  - `test_ensure_fim_tokens_ok`: vocab ครบ 3 ตัว → คืน dict ครบ int id
  - `test_ensure_fim_tokens_missing_raises`: vocab ว่าง → `pytest.raises(ValueError)` + ชื่อ token อยู่ใน str(exc)
  - `test_commit_checkpoint_moves_and_cleans(tmp_path)`: tmp มีไฟล์ → commit → final มีไฟล์, tmp หาย
  - `test_commit_checkpoint_overwrites(tmp_path)`: final มีไฟล์เดิม → commit → เนื้อหาใหม่แทนที่หมด, ไม่มี `*.saving` ค้าง (glob ตรวจ)
- [ ] **Step 2: รัน test — FAIL (ยังไม่มี ใน module)**
- [ ] **Step 3: implement** ทั้ง 3 ส่วนใน `core/trainer_worker.py` (import `os`, `math`, `Path`, `SFTTrainer`)
- [ ] **Step 4: รัน test — PASS ทั้ง 6 (รวมเดิม = 8)**
- [ ] **Step 5: Commit** — `git add core/trainer_worker.py tests/test_trainer_worker.py && git commit -m "feat: NaN guard + FIM token guard + atomic checkpoint"`

---

### Task 4: `StreamToQueueCallback` — ส่ง metric/log/status/error ลง Queue

**Files:**
- Modify: `core/trainer_worker.py`
- Test: ต่อใน `tests/test_trainer_worker.py`

**Interfaces:**
- Consumes: `ipc_bridge.metric_msg/log_msg/status_msg/error_msg/validate_message` (Task 1), `NanGuard` (Task 3)
- Produces (Task 5 ใช้ต่อ):
  - `class StreamToQueueCallback(TrainerCallback)`:
    - `__init__(self, queue)` — `queue` มีแค่ `.put(msg)` (duck-type: mp.Queue หรือ fake ใน test); เก็บ `self.guard = NanGuard()`, `self.aborted: bool = False`
    - `on_log(args, state, control, logs, **kwargs)`:
      - มี key `"loss"` → `queue.put(metric_msg(step=logs.get("step", state.global_step), loss=logs["loss"], lr=logs.get("learning_rate", 0.0), epoch=logs.get("epoch", 0.0)))` แล้ว `guard.register(loss)` — คืน True (ครบ 3 ครั้ง) → `queue.put(error_msg(f"loss non-finite ติดต่อกัน {threshold} ครั้ง — abort", ""))` + `queue.put(status_msg("aborted"))` + `self.aborted = True` + `control.should_training_stop = True`
      - ไม่มี key `"loss"` → `queue.put(log_msg("INFO", str(logs)))`
    - `on_train_begin(...)` → `status_msg("training")` (ข้ามถ้า `self.aborted`)
    - `on_save(...)` → `status_msg("saving")`
    - `on_train_end(...)` → `status_msg("finished")` (ข้ามถ้า `self.aborted`)

- [ ] **Step 1: เขียน tests ต่อใน `tests/test_trainer_worker.py`** — helper `FakeQueue` (เก็บ list ของ `.put`), `state=SimpleNamespace(global_step=10)`, `control=SimpleNamespace(should_training_stop=False)`:
  - `test_on_log_emits_metric`: logs `{"loss": 1.5, "learning_rate": 2e-4, "step": 10, "epoch": 0.2}` → msg เดียว `type=="metric"` + keys ครบ + `ipc.validate_message(msg)` True + `lr == 2e-4`
  - `test_status_flow`: `on_train_begin` → `training`; `on_save` → `saving`; `on_train_end` → `finished` (เรียงตามที่เรียก)
  - `test_nan_trip_sends_error_and_stops`: logs loss=nan สามครั้ง → ครั้งที่ 3 มี `error` + `aborted` messages, `control.should_training_stop is True`, `cb.aborted is True`; เรียก `on_train_end` ต่อ → **ไม่มี** `finished` เพิ่ม
  - `test_non_metric_logs_forwarded`: logs `{"train_runtime": 100.0}` (ไม่มี `"loss"`) → msg `type=="log"` + `validate_message` True
  - ทุก test ใช้ `from core import ipc_bridge as ipc` เพื่อ validate
- [ ] **Step 2: รัน test — FAIL**
- [ ] **Step 3: implement** `StreamToQueueCallback` ใน `core/trainer_worker.py` (import `TrainerCallback` จาก transformers)
- [ ] **Step 4: รัน test — PASS ทั้ง 4 (รวมเดิม = 12)**
- [ ] **Step 5: Commit** — `git commit -m "feat: StreamToQueueCallback metric/log/status/error + NaN abort"`

---

### Task 5: `run_training()` + dataset wiring + Integration Smoke (ของจริง)

**Files:**
- Modify: `core/dataset_builder.py` (เพิ่ม `iter_codes`)
- Modify: `core/trainer_worker.py` (เพิ่ม `validate_config` + `run_training`)
- Create: `scripts/pipeline_smoke.py`
- Test: ต่อใน `tests/test_dataset_builder.py` + `tests/test_trainer_worker.py`

**Interfaces:**
- Consumes: ทุกอย่างจาก Task 1–4 + `dataset_builder.build_samples` + `safe_defaults.LORA_TARGET_MODULES/TRAIN_CODE_LIMIT/MAX_SEQ_LENGTH_CAP/fim_registry.json`
- Produces:
  - `iter_codes(dataset_id: str, column: str, *, limit: int, split: str = "train", cache_dir: str = "data_cache") -> list[str]` ใน `dataset_builder.py` — `load_dataset(dataset_id, split=split, streaming=True, cache_dir=cache_dir)` แล้ว `islice` `limit` ตัวอย่างแรก คืน `row[column]` (monkeyspatch จุด: `core.dataset_builder.load_dataset`)
  - `REQUIRED_CONFIG_KEYS: frozenset` + `validate_config(config: dict) -> None` — ขาด key → `ValueError` บอกชื่อที่ขาด; `max_seq_length` ไม่อยู่ใน `(0, MAX_SEQ_LENGTH_CAP]` → `ValueError`
  - config keys ที่ `run_training` รับ: `model_id, dataset_id, dataset_column, fim_registry_key, output_dir, max_seq_length, max_steps, code_limit, lora_rank` (+ optional `save_steps` ถ้าไม่ใส่ใช้ `SAVE_STEPS`)
  - `run_training(config: dict, queue) -> None` — flow:
    1. `validate_config(config)`; `queue.put(status_msg("starting"))`
    2. try: โหลด `configs/fim_registry.json` (`Path(__file__).resolve().parents[1]/"configs"/"fim_registry.json"`) → `fim_tokens = registry[config["fim_registry_key"]]`
    3. `AutoTokenizer.from_pretrained(model_id)` → `ensure_fim_tokens(tokenizer, fim_tokens.values())`
    4. `codes = iter_codes(dataset_id, dataset_column, limit=code_limit)` → `texts = list(build_samples(codes, fim_tokens=fim_tokens, eos=tokenizer.eos_token, tokenizer=tokenizer, max_seq_length=config["max_seq_length"]))` → `Dataset.from_dict({"text": texts})`
    5. model `AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")`; `LoraConfig(r=lora_rank, lora_alpha=lora_rank*2, lora_dropout=0.0, target_modules=list(LORA_TARGET_MODULES), task_type="CAUSAL_LM")`
    6. `AtomicSaveTrainer(model=..., args=build_training_args(output_dir, max_steps=config["max_steps"], save_steps=config.get("save_steps", SAVE_STEPS), max_seq_length=config["max_seq_length"]), train_dataset=..., processing_class=tokenizer, peft_config=lora, callbacks=[StreamToQueueCallback(queue)])` → `trainer.train()`
    7. จบ: ถ้า `callback.aborted` เป็น False → `queue.put(status_msg("finished"))`
    8. `except Exception as exc:` → `queue.put(error_msg(str(exc), traceback.format_exc()))` + `queue.put(status_msg("aborted"))` + `raise`
  - `scripts/pipeline_smoke.py` — รัน `mp.set_start_method("spawn", force=True)`; รับ `--mode full|abort|all` (default `all`):
    - **full**: spawn `run_training` ด้วย config เล็ก (`max_steps=6, code_limit=64, save_steps=2, output_dir="data_cache/phase3_full", max_seq_length=1024, lora_rank=8, fim_registry_key="qwen"`, dataset/model จริง) → drain queue (get timeout มี deadline รวม 600 วิ) → assert: มี status `starting`+`training`+`finished`, มี `metric` ≥1, ทุก msg ผ่าน `validate_message`, มี `checkpoint-*` อย่างน้อย 1 ไดรฟ์ และไม่มี `*.saving`
    - **abort**: spawn config เดียวกับ full แต่ `max_steps=500, output_dir="data_cache/phase3_abort"` → อ่าน queue จนได้ `metric` ตัวแรก (deadline 600 วิ) → `free_before = torch.xpu.mem_get_info(0)` วัดก่อน spawn (เก็บไว้) → `abort_process(proc)` คืน True → `proc.join()` → `free_after` ≥ `free_before - 0.5` GB → พิมพ์ `PIPELINE SMOKE PASSED`
    - ตัววัด VRAM: `free_before` ต้องวัด **ก่อน** spawn ทุก mode, เปรียบเทียบหลัง process exit สนิท

- [ ] **Step 1: เขียน tests** —
  - `tests/test_dataset_builder.py::test_iter_codes_limits_and_column` — monkeypatch `db.load_dataset` คืน generator ของ dict 100 ตัว → `iter_codes("fake/ds", "content", limit=5)` คืน 5 ตัวแรก + assert kwargs ที่ส่ง (`streaming=True`, `split="train"`, `cache_dir="data_cache"`)
  - `tests/test_trainer_worker.py::test_validate_config` — config ครบ → ไม่ raise; ลบ key ใด key หนึ่ง → `ValueError` + ชื่อ key; `max_seq_length=4096` → `ValueError`
- [ ] **Step 2: รัน test 2 ตัว — FAIL**
- [ ] **Step 3: implement `iter_codes` + `validate_config`** → รัน PASS
- [ ] **Step 4: เขียน `scripts/pipeline_smoke.py`** (ใช้ `run_training` ที่ยังไม่มี = red ตอนรัน)
- [ ] **Step 5: implement `run_training`** ใน `core/trainer_worker.py` ตาม flow ข้างบน
- [ ] **Step 6: รัน integration** — `../../.venv/bin/python scripts/pipeline_smoke.py --mode all`
  Expected: `PIPELINE SMOKE PASSED`, exit 0 (รันบน Arc B580 จริง; โหลดโมเดล+stream dataset ครั้งแรกใช้เวลาหลายนาที)
- [ ] **Step 7: รัน unit suite ทั้งหมด** — `../../.venv/bin/python -m pytest tests/ -v` → PASS ทั้งหมด (ของเดิม 28 + ใหม่ 22 = 50)
- [ ] **Step 8: Commit** — `git add core/ scripts/pipeline_smoke.py tests/ && git commit -m "feat: run_training subprocess pipeline + pipeline smoke (full/abort/VRAM)"`
