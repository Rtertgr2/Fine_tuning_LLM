# Audit Backlog Wave P0+P1+ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ปิด audit waves P0 (3 ข้อ), P1 (10 ข้อ + 2 dispositions), ML logic (4 ข้อ) บน branch `fix/audit-backlog` — ทุกข้อ TDD, wave ละ 1 commit

**Architecture:** แก้ใน place ตาม refs ของ spec (ไม่มีการย้ายไฟล์ — reorg เป็น plan ถัดไป) · behavior ที่ตั้งใจเปลี่ยน 2 จุดต้องแก้ test เดิมด้วย: eval gate (tie→FAIL) และ token-F1 (order ต้อง mattered) · ค่าคงที่ใหม่ทั้งหมดลง `configs/safe_defaults.py` + pin ใน `tests/test_safe_defaults.py`

**Tech Stack:** Python 3.11, pytest, transformers 5.x (SFTConfig), peft 0.21.1, Gradio 6.29.0, matplotlib (Agg)

**Scope note:** M9 มี 2 ส่วน — min-delta gate ของ `eval.py` อยู่ใน plan นี้ (Task 12) · **ส่วน quant-regression gate อัตโนมัติ** ของ `scripts/benchmark_compression.py` ซ้อนกับ Wave-1 #11 (delta gate ไม่เทียบ dataset/column) → ทำรวมกันเป็น task เดียวใน **Plan 2 (P2+Wave1)** เพื่อไม่ให้ 2 plan แก้ gate เดียวกัน

**Spec:** `docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md` (§3 dispositions, §4 inventory, §7 policies)

## Global Constraints

- ทุกคำสั่ง: รันจาก repo root ด้วย `.venv/bin/python -m pytest ...` (system python ไม่มี deps) · `pytest.ini` กรอง `-m "not integration"` อยู่แล้ว — ห้ามรัน integration
- ลำดับห้ามข้าม: Task 1→14 ตามเลข · commit แค่ตอนจบ wave (Task 3, 10, 14) เท่านั้น
- ค่าคงที่ใหม่ → `configs/safe_defaults.py` เท่านั้น + เพิ่ม pin ใน `tests/test_safe_defaults.py` (README:186)
- error string ที่โผล่ UI/CLI = ภาษาอังกฤษเสมอ
- `is_heldout` algorithm ห้ามแตะ (md5 split = baseline ทุกชุด) — spec §5 caveats
- ห้ามแตะ: `tools/`, `data_cache/`, `datasets/`, `models/`, `exports/`, `benchmarks/`, `diagnosis-issue-draft.md`, `docs/superpowers/{specs,plans}` (ยกเว้น Task 10 append §3 เท่านั้น)
- FIM = PSM เท่านั้น, seed 42, hard caps เดิม (seq ≤ 2048, batch 1, lr 2e-4)
- test ที่ pin พฤติกรรมเก่าซึ่งตั้งใจเปลี่ยน → แก้ใน task ที่ระบุ (ห้ามลบ)

## Review Focus

1. **transformers 5.x อาจไม่มี/เปลี่ยน field `logging_nan_inf_filter`** — Task 1 test-first จะเปิดเผยเป็น TypeError/งด assert ไม่ได้ → ถ้า field หายจริง หยุดถาม (ratchet spec §9)
2. **Gradio 6 instance-update อาจ reset prop ที่ไม่ระบุ** — Task 8 pin `value=` explicit ทุก prop + test assert value คงเดิม
3. **D1 id-filter ต้องไม่ตัด token จริง** — ตัดเฉพาะ id ที่มาจาก `fim_tokens` (validate แล้วโดย `ensure_fim_tokens`) — test ครอบคู่ marker-id ↔ คำจริงด้วย FakeTok pre-seed
4. **M4 ห้ามแตะ eval baseline** — `find_near_dup_leakage` ทำงานฝั่ง train เท่านั้น, heldout ห้ามหาย/ขยับ — test ตรง fixture is_heldout + determinism
5. **M9 เป็น breaking CLI contract ที่ตั้งใจ** (tie เคย PASS → FAIL) — แก้ test + README:104 + docstring พร้อมกันใน task เดียว ห้ามแยก

---

### Task 1: P0-A1 — NaN guard ต้องเห็น loss จริง

**Files:**
- Modify: `core/trainer_worker.py:103-124` (`build_training_args`)
- Test: `tests/test_trainer_worker.py` (เพิ่มใน `test_args_pin_hyperparams`)

**Interfaces:**
- Produces: `SFTConfig.logging_nan_inf_filter = False` — `StreamToQueueCallback.on_log` (`:253`) จะได้เห็น loss non-finite จริง (NanGuard เดิมถูก filter ก่อนถึง)

- [ ] **Step 1: เพิ่ม assertion ใน `test_args_pin_hyperparams`**

```python
assert args.logging_nan_inf_filter is False  # P0 A1: filter ปิด → NanGuard เห็น loss จริง
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_trainer_worker.py::test_args_pin_hyperparams -v`
Expected: FAIL (`assert None/True is False` หรือ attribute error — ทั้งคู่แปลว่า test ยังไม่ผ่าน)

- [ ] **Step 3: เพิ่ม `logging_nan_inf_filter=False` ใน `SFTConfig(...)` call ที่ `core/trainer_worker.py:103`**

พร้อม comment สั้น ๆ อ้าง A1: transformers default True กรอง non-finite ก่อน on_log → NanGuard ไม่เคยทำงาน

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/test_trainer_worker.py::test_args_pin_hyperparams -v`
Expected: PASS — ถ้า TypeError (unexpected keyword) = Review Focus #1 → **หยุด รายงาน ก่อนทำต่อ**

---

### Task 2: P0-C1 — Start ต้องอยู่ใน concurrency group เดียวกับ eval/predict

**Files:**
- Modify: `ui/dashboard.py:374`
- Test: `tests/test_ui_dashboard.py` (test ใหม่ ต่อจาก `test_dashboard_wires_refresh_choices_on_tab1`)

**Interfaces:**
- Produces: dependency `on_start` มี `concurrency_id="model_load"` (group เดียวกับ predict/merge/eval ที่ `:386/:391/:396`)

- [ ] **Step 1: เขียน test ล้มเหลวก่อน**

```python
def test_dashboard_wires_start_in_model_load_group():
    """P0 C1: Start ต้อง serialize กับ eval/predict ที่ค้าง — ไม่งั้น 2 process โหลดโมเดล = OOM"""
    demo = build_dashboard(FakeController())
    deps = demo.get_config_file()["dependencies"]
    on_start = next(d for d in deps if d.get("api_name") == "on_start")
    assert on_start.get("concurrency_id") == "model_load"
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_ui_dashboard.py::test_dashboard_wires_start_in_model_load_group -v`
Expected: FAIL (`None != 'model_load'` หรือ `api_name` ไม่พบ)

- [ ] **Step 3: เพิ่ม `concurrency_id="model_load"` ใน `start_btn.click(...)` ที่ `ui/dashboard.py:374`**

(ตรงกับ pattern M2 ที่ predict/merge/eval ใช้อยู่ — comment เดิมที่ `:383` ใช้ได้)

- [ ] **Step 4: Run → PASS + test กลุ่ม dashboard**

Run: `.venv/bin/python -m pytest tests/test_ui_dashboard.py -v`
Expected: ทั้งไฟล์ PASS

---

### Task 3: P0-D1 — decode ต้องไม่ปล่อย FIM marker ใน prediction

**Files:**
- Modify: `core/evaluator.py:125-134` (`evaluate_cases`)
- Test: `tests/test_evaluator.py` (FakeTok + test ใหม่)

**Interfaces:**
- Consumes: `fim_tokens: dict` (param ของ `evaluate_cases` — values เป็น token strings, validated แล้วโดย `ensure_fim_tokens` ใน `run_eval`)
- Produces (test double): `FakeTok.convert_tokens_to_ids(tok) -> int` — Task 6 จะใช้ต่อ
- Produces: `pred` ไม่มี FIM marker (id 151659-61) → EM/F1 เทียบ baseline เดิมได้

- [ ] **Step 1: เพิ่ม `convert_tokens_to_ids` ใน `FakeTok` (tests/test_evaluator.py)**

```python
def convert_tokens_to_ids(self, tok: str) -> int:
    return self.encode(tok)[0]
```

และ test ใหม่:

```python
def test_evaluate_cases_strips_fim_markers_from_pred():
    """P0 D1: marker id ไม่ใช่ special → decode เดิมปล่อยออกมา → EM บวม/เพี้ยน"""
    tok = FakeTok()
    tok._vocab["<|fim_middle|>"] = 151660          # ตรง id จริงของ registry
    case = ev.EvalCase("def f():", "pass", "return 1")
    model = FakeModel("return 1 <|fim_prefix|>", tok)   # model echo marker ออกมา
    result = ev.evaluate_cases(model, tok, [case], fim_tokens=FIM, device="cpu")
    assert result["per_case"][0]["pred"] == "return 1"
    assert result["per_case"][0]["exact"] is True
```

หมายเหตุ: `FIM` dict ในไฟล์นี้เป็น token strings อยู่แล้ว (`{"prefix": "<|fim_prefix|>", ...}`)

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::test_evaluate_cases_strips_fim_markers_from_pred -v`
Expected: FAIL (`exact is False` — pred มี `<![CDATA[EGIN]]>` หลงเหลือ / `<unk>`)

- [ ] **Step 3: Implement id-filter ใน `evaluate_cases`**

```python
def evaluate_cases(model, tokenizer, cases, *, fim_tokens, device, progress=None) -> dict
```
ก่อน loop: `fim_ids = {tokenizer.convert_tokens_to_ids(t) for t in fim_tokens.values() if ...}`
หลัง generate: กรอง `continuation` ให้เหลือ id ∉ fim_ids **ก่อน** `tokenizer.decode(...)` (decode เดิมที่ `:134` คงไว้ รวม `skip_special_tokens=True`) — filter แบบ list-of-int แล้วส่งเข้า decode (FakeTok/ของจริงรับทั้งคู่)

- [ ] **Step 4: Run test ใหม่ + ทั้งไฟล์**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py -v`
Expected: PASS ทั้งไฟล์ (`test_evaluate_cases_prompt_equals_build_fim_prompt` ห้ามพัง)

- [ ] **Step 5: Commit wave P0**

```bash
git add core/trainer_worker.py ui/dashboard.py core/evaluator.py \
        tests/test_trainer_worker.py tests/test_ui_dashboard.py tests/test_evaluator.py
git commit -m "fix(p0): expose raw loss to NaN guard, lock Start behind model_load, strip FIM markers (A1/C1/D1)"
```

---

### Task 4: P1-A2+A3 — eval batch size = train + LR ที่โชว์ต้องไม่ใช่ 0.0 ปลอม

**Files:**
- Modify: `core/trainer_worker.py:103-124` (args), `:218-262` (`StreamToQueueCallback`)
- Test: `tests/test_trainer_worker.py` (`test_args_pin_hyperparams` + test ใหม่ 2 ตัว)

**Interfaces:**
- Produces: `SFTConfig.per_device_eval_batch_size == 1` · `StreamToQueueCallback.last_lr: float | None` (cache) — log ไม่มี `learning_rate` ครั้งแรก → WARNING log แทน metric ปลอม

- [ ] **Step 1: เขียน test ล้มเหลวก่อน**

```python
def test_args_eval_batch_equals_train():
    args = wa.build_training_args("data_cache/x")
    assert args.per_device_eval_batch_size == 1  # P1 A2: default 8 + packing = OOM ตอน eval

def test_on_log_without_lr_uses_cache_not_zero(monkeypatch):
    """P1 A3: lr หาย = อย่าโชว์ 0.0 เหมือนจริง — ใช้ค่าล่าสุดที่เห็นจริง"""
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"loss": 1.5, "learning_rate": 2e-4, "step": 1})
    cb.on_log(None, state, control, {"loss": 1.4, "step": 2})
    metrics = [m for m in q.messages if m["type"] == "metric"]
    assert len(metrics) == 2
    assert metrics[1]["lr"] == 2e-4            # ใช้ cache — ไม่ใช่ 0.0

def test_on_log_lr_never_seen_sends_warning_not_metric():
    q, cb, state, control = _cb_with_states()
    cb.on_log(None, state, control, {"loss": 1.5, "step": 1})
    assert all(m["type"] != "metric" for m in q.messages)
    warn = next(m for m in q.messages if m["type"] == "log")
    assert warn["level"] == "WARNING" and "learning_rate" in warn["text"]
```

หมายเหตุ: `_cb_with_states()` มีอยู่แล้วในไฟล์ (ใช้โดย `test_on_log_emits_metric`)

- [ ] **Step 2: Run → FAIL ทั้ง 3**

Run: `.venv/bin/python -m pytest tests/test_trainer_worker.py -k "eval_batch or lr_ or never_seen" -v`
Expected: FAIL 3 ตัว

- [ ] **Step 3: Implement**

- `per_device_eval_batch_size=DEFAULT_BATCH_SIZE` ใน `SFTConfig(...)` (`:110 附近`)
- `StreamToQueueCallback.__init__`: `self.last_lr: float | None = None`
- `on_log` branch ที่มี `"loss"`: ถ้า `"learning_rate" in logs` → update `self.last_lr` แล้ว emit `metric_msg(lr=self.last_lr)`; ถ้า `self.last_lr is None` → `log_msg("WARNING", ...)` (raw logs ในข้อความ) **แล้วข้าม metric**; `self.guard.register(loss)` ต้องถูกเรียกทุกทางเสมอ (ห้าม skip NaN guard)

- [ ] **Step 4: Run → PASS ทั้งไฟล์**

Run: `.venv/bin/python -m pytest tests/test_trainer_worker.py -v`
Expected: PASS ทั้งไฟล์ (รวม `test_on_log_emits_metric`, `test_nan_trip_sends_error_and_stops`)

---

### Task 5: P1-B1 — hf_hub_download pin revision + cache ใน data_cache

**Files:**
- Modify: `core/estimator.py:157`, `configs/safe_defaults.py`
- Test: `tests/test_estimator.py` (test ใหม่), `tests/test_safe_defaults.py` (pin ใหม่)

**Interfaces:**
- Produces: `safe_defaults.HF_HUB_REVISION: str = "main"` · `safe_defaults.HF_HUB_CACHE_DIR: str = "data_cache/hf_hub"` · hub call ใช้ `revision=` + `cache_dir=str(REPO_ROOT / HF_HUB_CACHE_DIR)` (REPO_ROOT มีอยู่แล้วใน estimator)

- [ ] **Step 1: เขียน test ล้มเหลวก่อน**

```python
def test_hub_config_fetch_pins_revision_and_cache(monkeypatch, tmp_path):
    """P1 B1: ไม่ตรึง revision = config ขยับเงียบ ๆ; cache ต้องอยู่ใต้ data_cache"""
    import configs.safe_defaults as sd
    cfg = {"hidden_size": 64, "num_hidden_layers": 2,
           "intermediate_size": 128, "vocab_size": 256}
    target = tmp_path / "config.json"
    target.write_text(json.dumps(cfg), encoding="utf-8")
    seen: dict = {}
    def _fake(model_id, filename, **kw):
        seen.update(kw); return str(target)
    monkeypatch.setattr(est, "hf_hub_download", _fake)
    spec = est.resolve_model_spec("acme/pinned-model", None)
    assert spec.source == "hf_config"
    assert seen["revision"] == sd.HF_HUB_REVISION
    assert seen["cache_dir"].endswith(str(Path(sd.HF_HUB_CACHE_DIR)))
    assert str(est.REPO_ROOT) in seen["cache_dir"]          # absolute จาก repo root — ไม่ใช่ cwd
```

(เพิ่ม pin ใน `test_safe_defaults.py`: `assert sd.HF_HUB_REVISION == "main"`, `assert sd.HF_HUB_CACHE_DIR == "data_cache/hf_hub"`)

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_estimator.py::test_hub_config_fetch_pins_revision_and_cache tests/test_safe_defaults.py -v`
Expected: FAIL (AttributeError `HF_HUB_REVISION` / `cache_dir` ไม่ถูกส่ง)

- [ ] **Step 3: Implement**

- คงที่ 2 ตัวใน `safe_defaults.py` หมวด Estimator + comment: revision เป็นจุดเปลี่ยนเดียวเมื่อต้องการ pin commit sha
- เรียก `hf_hub_download(model_id, "config.json", revision=HF_HUB_REVISION, cache_dir=str(REPO_ROOT / HF_HUB_CACHE_DIR))`

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/test_estimator.py tests/test_safe_defaults.py -v`
Expected: PASS ทั้ง 2 ไฟล์ (test เดิมที่ monkeypatch `hf_hub_download` ด้วย `*a, **k` ต้องรอด — ถ้าตัวไหนรับ positional ไม่ครบ ให้แก้ test นั้นรับ `**kw`)

---

### Task 6: P1-D2 — eval tokenize ตรงกับ train (add_special_tokens=False)

**Files:**
- Modify: `core/evaluator.py:127`
- Test: `tests/test_evaluator.py` (FakeTok `__call__` + test ใหม่)

**Interfaces:**
- Consumes: FakeTok จาก Task 3
- Produces: ทุก call ใน `evaluate_cases` ใช้ `add_special_tokens=False` — model ที่เติม BOS ไม่ mismatch กับ packing ฝั่งเทรน

- [ ] **Step 1: FakeTok `__call__` บันทึก kwargs + test ใหม่**

```python
# FakeTok.__call__ — เพิ่ม param + เก็บ record
def __call__(self, prompt, *, return_tensors="pt", add_special_tokens=True):
    self.captured_kwargs.append(
        {"return_tensors": return_tensors, "add_special_tokens": add_special_tokens}
    )
    ...
# __init__: self.captured_kwargs: list[dict] = []

def test_evaluate_cases_tokenizes_without_special_tokens():
    tok = FakeTok()
    case = ev.EvalCase("def f():", "pass", "return 1")
    model = FakeModel("return 1", tok)
    ev.evaluate_cases(model, tok, [case], fim_tokens=FIM, device="cpu")
    assert tok.captured_kwargs[0]["add_special_tokens"] is False  # P1 D2: ตรงฝั่งเทรน
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py::test_evaluate_cases_tokenizes_without_special_tokens -v`
Expected: FAIL (`True is False` หรือ FakeTok `TypeError` ถ้ายังไม่เพิ่ม param)

- [ ] **Step 3: เพิ่ม `add_special_tokens=False` ใน `tokenizer(prompt, return_tensors="pt")` ที่ `:127`**

- [ ] **Step 4: Run → PASS ทั้งไฟล์**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py -v`
Expected: PASS ทั้งไฟล์

---

### Task 7: P1-D3+D4 — XPU capability check + abort guard process ที่ยังไม่ start

**Files:**
- Modify: `core/hardware.py:22-23`, `core/ipc_bridge.py:95-106`
- Test: `tests/test_hardware.py`, `tests/test_ipc_bridge.py`

**Interfaces:**
- Produces: `hardware._xpu_available() -> bool` (try/except `AttributeError, RuntimeError` → False) · `abort_process` ไม่เรียก `terminate()` เมื่อ process ตาย/ยังไม่ start (คืน True = ตายสนิทแล้ว)

- [ ] **Step 1: เขียน test ล้มเหลวก่อน**

```python
# tests/test_hardware.py
def test_missing_torch_xpu_attr_is_no_xpu(monkeypatch):
    """P1 D3: hasattr(torch,'xpu') ไม่ใช่ capability check — เอาออกแล้ว behavior ต้องอยู่"""
    monkeypatch.delattr(hw.torch, "xpu")
    result = hw.inspect()
    assert result["status"] == "no_xpu"

# tests/test_ipc_bridge.py
class NeverStartedProcess(FakeProcess):
    def terminate(self):
        raise AssertionError("can only terminate a started process")  # mp จริง throw แบบนี้

def test_abort_never_started_process_is_safe():
    """P1 D4: terminate() ก่อน start = AssertionError → abort ต้องนิ่ง"""
    p = NeverStartedProcess(alive=False)   # is_alive False ทั้งที่ terminate จะพัง
    assert ipc.abort_process(p) is True
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_hardware.py::test_missing_torch_xpu_attr_is_no_xpu tests/test_ipc_bridge.py::test_abort_never_started_process_is_safe -v`
Expected: FAIL 2 ตัว (ตัวแรก fail เมื่อถึงขั้นเอา hasattr ออก — เขียน implement ให้เสร็จก่อนค่อยเช็ค; ตัวที่ 2 fail `AssertionError` leak ตอนนี้เลย)

- [ ] **Step 3: Implement**

- `hardware.py`: helper `_xpu_available()` = `try: return bool(torch.xpu.is_available()) except (AttributeError, RuntimeError): return False`; `inspect` เรียก helper แทน `hasattr(...) and ...` (`:23`)
- `ipc_bridge.py`: ครอบ `terminate/join/kill` ด้วย `if process.is_alive():` — process ไม่มีชีวิต → ข้าม แล้ว `return not process.is_alive()`

- [ ] **Step 4: Run → PASS ทั้ง 2 ไฟล์**

Run: `.venv/bin/python -m pytest tests/test_hardware.py tests/test_ipc_bridge.py -v`
Expected: PASS ทั้งคู่ (รวม `test_abort_escalates_to_kill`, `test_abort_no_kill_when_terminate_works` — ทั้งคู่เริ่ม alive=True → behavior คงเดิม)

---

### Task 8: P1-C2+C3 — docstring app.py ต้องไม่กล่าวอ้างเท็จ + gr.update → instance pattern

**Files:**
- Modify: `app.py:4,41`, `ui/dashboard.py:232-235,368`
- Test: `tests/test_ui_dashboard.py` (แก้ `test_on_refresh_choices_lists_detected_assets`)

**Interfaces:**
- Produces: `on_refresh_choices(model_value, dataset_value)` คืน `gr.Dropdown` 2 ตัวพร้อม `value=` คงค่าเดิม · wiring เพิ่ม `inputs=[model_in, dataset_in]` · app.py docstring อธิบายกลไกกัน stack จริง (concurrency_id + controller guard + tick locks) แทนการอ้าง `default_concurrency_limit`

- [ ] **Step 1: แก้ test `test_on_refresh_choices_lists_detected_assets` ให้เรียกแบบมี args + assert instance**

```python
    model_upd, dataset_upd = h["on_refresh_choices"]("custom-model", "custom-ds")
    import gradio as gr
    assert isinstance(model_upd, gr.Dropdown)      # P1 C3: instance pattern ไม่ใช่ gr.update
    assert isinstance(dataset_upd, gr.Dropdown)
    assert model_upd.value == "custom-model"       # value คงเดิม (tab switch ห้ามล้าง selection)
    assert dataset_upd.value == "custom-ds"
    # choices assertions เดิมทั้งหมดคงไว้
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_ui_dashboard.py::test_on_refresh_choices_lists_detected_assets -v`
Expected: FAIL (`gr.update` dict ไม่ใช่ `gr.Dropdown` / ไม่มี `.value`)

- [ ] **Step 3: Implement**

- `on_refresh_choices(model_value, dataset_value)` → คืน `(gr.Dropdown(choices=[DEFAULT_MODEL_ID, *list_models()], value=model_value, allow_custom_value=True), gr.Dropdown(choices=[DEFAULT_DATASET_ID, *list_datasets()], value=dataset_value, allow_custom_value=True))`
- wiring `:368`: `tab1.select(h["on_refresh_choices"], inputs=[model_in, dataset_in], outputs=[model_in, dataset_in])`
- `app.py` docstring `:4` + comment `:41`: **ลบ `default_concurrency_limit=1` ออกจาก `demo.queue(...)`** (ค่า default อยู่แล้ว = no-op ตาม audit C2) และเขียน docstring ตามจริง: ตัวกัน Process Stacking จริง = `concurrency_id="model_load"` (dashboard) + re-entrancy guard ใน controller + tick ล็อกปุ่ม

- [ ] **Step 4: Run → PASS + test app**

Run: `.venv/bin/python -m pytest tests/test_ui_dashboard.py tests/test_app.py -v`
Expected: PASS ทั้งคู่

---

### Task 9: P1-C4+C5 — legend ต้องมี + plot ห้ามใช้ pyplot global state

**Files:**
- Modify: `ui/components.py:29-57`
- Test: `tests/test_ui_components.py` (test ใหม่ 1 ตัว + assertions เพิ่ม)

**Interfaces:**
- Produces: `build_metric_plot` คืน `Figure` ที่มี legend (loss/val_loss/lr) และไม่แตะ `pyplot` global registry (ปลอดภัยบน Gradio threads)

- [ ] **Step 1: เขียน test ล้มเหลวก่อน**

```python
def test_build_metric_plot_has_legend_labels():
    import matplotlib.pyplot as plt
    metrics = [metric_msg(1, 2.0, 2e-4, 0.1), val_metric_msg(step=5, epoch=0.4, val_loss=0.9)]
    before = set(plt.get_fignums())
    fig = build_metric_plot(metrics)
    after = set(plt.get_fignums())
    assert after == before                      # P1 C5: ห้ามสร้าง figure ใน pyplot global state
    ax_loss = fig.axes[0]
    assert ax_loss.get_legend() is not None     # P1 C4: แยก loss/val_loss/lr ได้
    labels = {line.get_label() for line in ax_loss.get_legend().get_texts()}
    assert {"loss", "val_loss", "lr"} <= labels
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_ui_components.py::test_build_metric_plot_has_legend_labels -v`
Expected: FAIL (`after != before` — plt.subplots สร้าง fignum / legend เป็น None)

- [ ] **Step 3: Implement**

- แทน `plt.subplots(figsize=(7,3))`: `fig = Figure(figsize=(7, 3))` + `ax_loss = fig.add_subplot(111)` (import `Figure` มีอยู่แล้ว `:13`) — ลบ `import matplotlib.pyplot as plt` ถ้าไม่ใช้ที่อื่น
- หลังวาด series ทั้งหมด: เก็บ handles/labels จาก `ax_loss` + `ax_lr` (twinx) แล้ว `ax_loss.legend(handles, labels, loc="best")`
- test เดิม 3 ตัว (`two_axes`, `empty`, `val_points`) ต้องรอด — axes/lines API ไม่เปลี่ยน

- [ ] **Step 4: Run → PASS ทั้งไฟล์**

Run: `.venv/bin/python -m pytest tests/test_ui_components.py -v`
Expected: PASS ทั้งไฟล์

---

### Task 10: P1 dispositions — A5/B4 บันทึกเป็น dropped + commit wave P1

**Files:**
- Modify: `docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md` §3 (append เท่านั้น)

**Interfaces:**
- Produces: spec §3 มี 2 แถวเพิ่ม (disposition record — spec §7 บังคับ ไม่ปิดเงียบ)

- [ ] **Step 1: Append 2 แถวในตาราง §3 (verify แล้ว 2026-10-05)**

| A5 | refs `including_emulation=True` ไม่มีจริง — โค้ดเป็น `bool(torch.xpu.is_bf16_supported())` (`trainer_worker.py:75`); fallback True+warning มีครบแล้ว | **dropped** — refs ผิด, native-only bf16 = ทางเลือกปลอดภัยแล้ว |
| B4 | refs `iter_batches(batch_size=65536, row_groups=None)` ไม่มีจริง — โค้ดคือ `batch_size=64, columns=[column]` (`dataset_builder.py:236`) อ่าน bounded อยู่แล้ว | **dropped** — refs ผิด, ไม่มีอะไรต้องแก้ |

- [ ] **Step 2: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS ทั้งชุด (ไม่มี integration ตาม pytest.ini)

- [ ] **Step 3: Commit wave P1**

```bash
git add core/ ui/ configs/safe_defaults.py app.py \
        tests/ docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md
git commit -m "fix(p1): eval batch=1, honest LR logging, hub revision/cache pin, xpu+abort guards, UI instance pattern, plot legend (A2,B1,C2-C5,D2-D4; A5/B4 dropped)"
```

---

### Task 11: M10 — token-F1 ต้อง sequence-aware (LCS)

**Files:**
- Modify: `core/evaluator.py:58-71` (`token_f1`)
- Test: `tests/test_evaluator.py` (แก้ 1 test + เพิ่ม 1 test)

**Interfaces:**
- Produces: `token_f1(pred, gt, tokenizer) -> float` — คง signature เดิม (ผู้ใช้: `evaluate_cases`, `core/compression/llama_eval`) · semantics ใหม่: overlap = **longest common subsequence** ของ token ids (order mattered) · empty คู่ = 1.0, ฝั่งเดียวว่าง = 0.0 (คงเดิม)

- [ ] **Step 1: แก้ test พฤติกรรมเก่า + เพิ่ม test ใหม่**

```python
def test_token_f1_order_matters():          # RENAMED จาก test_token_f1_order_insensitive
    tok = FakeTok()
    # LCS("b a c","a b c") = 2 → P=R=2/3 → F1=2/3 — สลับลำดับห้ามได้ 1.0 อีก (M10)
    assert abs(ev.token_f1("b a c", "a b c", tok) - (2 / 3)) < 1e-9

def test_token_f1_shuffled_middle_not_perfect():
    tok = FakeTok()
    assert ev.token_f1("c b a", "a b c", tok) < 1.0
    assert ev.token_f1("a b c", "a b c", tok) == 1.0
```

`test_token_f1_partial_multiset` (คาด 2/3) — LCS(a a b, a b c)=2 → P=R=2/3 = ค่าเดิม **คงไว้ ไม่ต้องแก้**

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py -k "token_f1" -v`
Expected: FAIL `test_token_f1_order_matters` (ได้ 1.0 จาก Counter multiset เดิม)

- [ ] **Step 3: Implement LCS-based `token_f1`**

```python
def token_f1(pred: str, gt: str, tokenizer) -> float
```
- encode ทั้งคู่ด้วย `add_special_tokens=False` (คงเดิม) · helper `_lcs_length(a: list[int], b: list[int]) -> int` (DP มาตรฐาน O(n·m) — eval bounded ที่ `EVAL_MAX_NEW_TOKENS=256` ทั้งสองฝั่ง → ≤ 65k cells)
- `lcs = _lcs_length(pred_ids, gt_ids)`; empty cases คงเดิม; `p = lcs/len(pred)`, `r = lcs/len(gt)`, คืน harmonic mean
- ลบ `Counter` import ถ้าไม่ใช้ที่อื่น

- [ ] **Step 4: Run ทั้ง suite (รวม compression ที่ import `token_f1`)**

Run: `.venv/bin/python -m pytest tests/test_evaluator.py tests/test_compression_llama_eval.py -v`
Expected: PASS — compression fixtures เป็น identical/empty → LCS คง 1.0/0.0

---

### Task 12: M9 (ส่วน eval CLI) — gate ต้องมี min-delta

**Files:**
- Modify: `eval.py:80-94`, `README.md:104`
- Test: `tests/test_eval_cli.py` (แก้ 1 + เพิ่ม 2)

**Interfaces:**
- Produces: `eval.MIN_DELTA_EM_PCT: float = 0.1` · `eval.MIN_DELTA_F1: float = 0.01` · PASS ต้องมี delta ≥ min ทุก metric (tie = FAIL) · FAIL line ใหม่: `"FAIL — " + "; ".join(parts)` ซึ่ง parts เป็น `base better on: X` (regression) และ/หรือ `improvement below min-delta on: Y` (stall) — ชื่อ metric ทุกตัวต้องโผล่ในบรรทัด FAIL (spec §3.3 คงไว้)

- [ ] **Step 1: แก้/เพิ่ม test**

```python
def test_compare_fail_on_tie(tmp_path, capsys):      # RENAMED จาก test_compare_pass_on_tie
    r = _result("base", em=5.0, f1=0.2)
    eval_dir = _write_two(tmp_path, r, {**r, "mode": "finetuned"})
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 1                                    # M9: +0.0 ห้าม PASS อีกต่อไป
    fail_line = next(line for line in out.splitlines() if "FAIL" in line)
    assert "Exact Match" in fail_line and "Token F1" in fail_line

def test_compare_pass_at_min_delta(tmp_path, capsys):
    base = _result("base", em=10.0, f1=0.5)
    fine = _result("finetuned", em=10.1, f1=0.51)     # ตรง min delta ทั้งคู่
    eval_dir = _write_two(tmp_path, base, fine)
    assert cli_main(["--compare", "--eval-dir", eval_dir]) == 0

def test_compare_fail_below_min_delta(tmp_path, capsys):
    base = _result("base", em=10.0, f1=0.5)
    fine = _result("finetuned", em=10.05, f1=0.5)     # เกิน 0 แต่ต่ำกว่า min
    eval_dir = _write_two(tmp_path, base, fine)
    assert cli_main(["--compare", "--eval-dir", eval_dir]) == 1
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_eval_cli.py -k "compare" -v`
Expected: FAIL `test_compare_fail_on_tie` (เดิม assert rc==0) + 2 test ใหม่

- [ ] **Step 3: Implement gate ใน `_run_compare` (eval.py:80-94)**

- `passed = (fine_em - base_em >= MIN_DELTA_EM_PCT) and (fine_f1 - base_f1 >= MIN_DELTA_F1)`
- สร้าง `parts`: delta < 0 → `base better on: ...`; `0 <= delta < min` → `improvement below min-delta on: ...` — join ด้วย `"; "` ขึ้นต้น `FAIL — `
- `README.md:104`: เปลี่ยน `"ต้องไม่แย่กว่า base ทุก metric"` → `"ต้องดีขึ้นเกิน min-delta (EM ≥ +0.1pt, F1 ≥ +0.01) ทุก metric"`

- [ ] **Step 4: Run → PASS**

Run: `.venv/bin/python -m pytest tests/test_eval_cli.py -v`
Expected: PASS ทั้งไฟล์ (test qualitative/n-mismatch เดิมไม่กระทบ)

---

### Task 13: M8 — resume จาก checkpoint ล่าสุด

**Files:**
- Modify: `core/trainer_worker.py:337-432` (`run_training`), `ui/dashboard.py:51-78` (`_collect_config`), widget block Tab1
- Test: `tests/test_trainer_worker.py`, `tests/test_ui_dashboard.py`

**Interfaces:**
- Consumes: `latest_checkpoint(output_dir)` (มีอยู่แล้ว `:440`)
- Produces: `trainer_worker.resume_checkpoint(output_dir: str | Path) -> str | None` · config key `"resume": bool` (optional — `validate_config` ไม่ต้องแก้) · widget `resume_in = gr.Checkbox(value=False, label="Resume from latest checkpoint")` ต่อท้าย `cfg_inputs` · `trainer.train(resume_from_checkpoint=...)` เสมอ (None = fresh)

- [ ] **Step 1: เขียน test ล้มเหลวก่อน**

```python
# tests/test_trainer_worker.py
def test_resume_checkpoint_returns_latest_or_none(tmp_path):
    assert wa.resume_checkpoint(tmp_path) is None          # ยังไม่มี checkpoint
    (tmp_path / "checkpoint-5").mkdir()
    assert wa.resume_checkpoint(tmp_path) == str(tmp_path / "checkpoint-5")

# run_training wiring — ใช้ pattern ของ test_run_training_filters_heldout_from_iter_codes
# (extract monkeypatch block เป็น helper _stub_training_env(monkeypatch) ใช้ร่วม):
def test_run_training_resumes_when_flag_set(monkeypatch, tmp_path):
    calls: list = []
    _stub_training_env(monkeypatch, calls)      # train recorder: train(resume_from_checkpoint=...)
    (tmp_path / "checkpoint-5").mkdir()
    cfg = _minimal_config(output_dir=str(tmp_path)) | {"resume": True}
    q = FakeQueue()
    wa.run_training(cfg, q)
    assert calls[-1] == str(tmp_path / "checkpoint-5")
    assert any(m.get("text", "").startswith("resuming from") for m in q.messages if m["type"] == "log")

def test_run_training_resume_without_checkpoint_starts_fresh(monkeypatch, tmp_path):
    calls: list = []
    _stub_training_env(monkeypatch, calls)
    cfg = _minimal_config(output_dir=str(tmp_path)) | {"resume": True}
    q = FakeQueue()
    wa.run_training(cfg, q)
    assert calls[-1] is None
    assert any("starting fresh" in m.get("text", "") for m in q.messages if m["type"] == "log")
```

(`_minimal_config` = dict config 9 keys เดิมจาก test `filters_heldout` เปลี่ยน output_dir เป็น tmp_path — validate_config อนุญาต system temp)

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_trainer_worker.py -k "resume" -v`
Expected: FAIL (AttributeError `resume_checkpoint` / train ไม่ได้รับ kwargs)

- [ ] **Step 3: Implement**

- `resume_checkpoint`: ห่อ `latest_checkpoint` ด้วย try/ValueError → `None`
- `run_training` ก่อน `trainer.train()` (`:418`):
  ```python
  resume = resume_checkpoint(config["output_dir"]) if config.get("resume") else None
  # log INFO: resume path / "resume requested but no checkpoint found — starting fresh"
  trainer.train(resume_from_checkpoint=resume)
  ```
- Dashboard: `_collect_config(..., output_dir, resume: bool = False)` → เพิ่ม `"resume": bool(resume)` ใน config dict · เพิ่ม `resume_in` checkbox ต่อท้าย Tab1 หลัง `output_in` · เพิ่มใน `cfg_inputs` (handlers ทุกตัวรับผ่าน `*cfg_values` อัตโนมัติ)
- test dashboard: `test_collect_config_tolerates_cleared_number_fields` (เรียก 10 positional → resume default False) ต้องรอด + เพิ่ม assert `config["resume"] is False` และ `_collect_config(...11 args...True)` → `is True`

- [ ] **Step 4: Run → PASS ทั้ง 2 ไฟล์**

Run: `.venv/bin/python -m pytest tests/test_trainer_worker.py tests/test_ui_dashboard.py -v`
Expected: PASS ทั้งคู่

---

### Task 14: M4 — near-dup leakage guard + commit wave ML

**Files:**
- Modify: `core/dataset_builder.py:137-140` (`filter_train_codes` + functions ใหม่), `configs/safe_defaults.py`, `core/trainer_worker.py:357-363` (log count)
- Test: `tests/test_dataset_builder.py`, `tests/test_safe_defaults.py`

**Interfaces:**
- Produces:
  - `safe_defaults.NEAR_DUP_JACCARD: float = 0.8` · `safe_defaults.NEAR_DUP_MAX_POSTING: int = 50`
  - `dataset_builder._line_shingles(code: str) -> frozenset[bytes]` — md5 ของบรรทัด non-blank หลัง strip (`usedforsecurity=False` ตามแนว L1)
  - `dataset_builder.find_near_dup_leakage(train: list[str], heldout: list[str]) -> list[str]` — คืน train codes ที่ต้องทิ้ง
  - `filter_train_codes(codes)` — behavior เดิม (กรอง heldout) **บวก** ทิ้ง train ที่ near-dup กับ heldout · **ไม่แตะ `is_heldout`, ไม่แตะ heldout set (baseline คงเดิม)**
- ข้อจำกัดที่ code ต้อง comment: inverted index จาก heldout shingles → ตัด shingle ที่ posting > `NEAR_DUP_MAX_POSTING` (บรรทัดสามัญ เช่น `}` — tradeoff: คู่ near-dup ที่มีแต่บรรทัดสามัญอาจหลุด) → candidate ต้องมี shared ≥ `ceil(t/(1+t)·(|A|+|B|))` → คำนวณ Jaccard จริง

- [ ] **Step 1: เขียน test ล้มเหลวก่อน**

```python
# tests/test_safe_defaults.py — เพิ่มใน block training
assert sd.NEAR_DUP_JACCARD == 0.8
assert sd.NEAR_DUP_MAX_POSTING == 50

# tests/test_dataset_builder.py
def test_near_dup_threshold_boundary():
    """M4: Jaccard ≥ 0.8 = ทิ้ง — 9/11=0.818 ทิ้ง, 8/12=0.667 เก็บ"""
    held = "\n".join(f"line_{i} = {i}" for i in range(10))
    one_changed = "\n".join([f"line_{i} = {i}" for i in range(9)] + ["line_9 = 999"])
    two_changed = "\n".join([f"line_{i} = {i}" for i in range(8)] + ["line_8 = 888", "line_9 = 999"])
    assert db.find_near_dup_leakage([one_changed], [held]) == [one_changed]
    assert db.find_near_dup_leakage([two_changed], [held]) == []

def test_filter_train_codes_drops_reindented_twin_of_heldout():
    """M4: ฝั่ง train เท่านั้น — twin ของ heldout (คนละ indent = md5 คนละตัว) ห้ามอยู่ในชุดเทรน"""
    heldout = next(
        c for c in (f"def f_{i}():\n    return {i}" for i in range(5000)) if db.is_heldout(c)
    )
    twin = "\n".join("    " + line for line in heldout.splitlines())
    kept = "def unrelated():\n    return 0"
    result = db.filter_train_codes([heldout, twin, kept])
    assert heldout not in result and twin not in result
    assert kept in result
    assert db.filter_train_codes([heldout, twin, kept]) == result   # deterministic
```

- [ ] **Step 2: Run → FAIL**

Run: `.venv/bin/python -m pytest tests/test_dataset_builder.py -k "near_dup or reindented" tests/test_safe_defaults.py -v`
Expected: FAIL (AttributeError `find_near_dup_leakage` / pin ขาด)

- [ ] **Step 3: Implement**

- คงที่ 2 ตัวใน `safe_defaults.py` (หมวด Dataset) — `_line_shingles` + `find_near_dup_leakage` ตาม Interfaces (algorithm: inverted index + posting cap + shared-count prune + exact Jaccard)
- `filter_train_codes`: materialize → partition ด้วย `is_heldout` (เรียกครั้งเดียวต่อ code เก็บ tuple) → `train = [c for ...]`, `heldout = [c for ...]` → ทิ้ง `set(find_near_dup_leakage(train, heldout))` → คืน train คงลำดับเดิม
- `run_training` (`trainer_worker.py:363` หลัง filter): `queue_.put(log_msg("INFO", f"train codes after filters: {len(codes)}"))` — test `filters_heldout` assert `messages[-1]` เป็น finished ยังรอด (log มาก่อน finished)

- [ ] **Step 4: Run → PASS ทั้ง suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS ทั้งชุด — โดยเฉพาะ `test_filter_train_codes_removes_all_heldout` (fixture lines disjoint → ไม่โดนทิ้ง), `test_run_training_filters_heldout_from_iter_codes`, compression suite

- [ ] **Step 5: Commit wave ML**

```bash
git add core/ configs/safe_defaults.py eval.py README.md tests/
git commit -m "fix(ml): sequence-aware token-F1 (LCS), min-delta eval gate, resume-from-checkpoint, near-dup train/heldout leakage guard (M10/M9/M8/M4)"
```

---

## Wave completion checks

- [ ] ทุก wave commit แล้ว 3 commits (P0/P1/ML) · `git log --oneline -4` ตรง
- [ ] `.venv/bin/python -m pytest -q` PASS ทั้งชุด
- [ ] spec §3 มี disposition A5/B4/D11 ครบ (Task 10)
- [ ] ไม่มีไฟล์นอก scope แก้ไข: `git status --short` สะอาด (ยกเว้น untracked เดิม)
