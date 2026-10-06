# Implementation Plan: Audit Backlog Waves P2 + Wave 1

**Plan type:** Implementation (single repo, sequential tasks, one commit per wave)
**Execution method:** Native (assistant inline; one fresh whole-branch review at the end — user-approved)
**Date:** 2026-10-05
**Spec:** `docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md` (§4 = inventory, §3 = dispositions, §7 = policies, §9 = ratchet)
**Depends on:** Plan 1 (`2026-10-05-audit-backlog-p0-p1-ml.md`) — complete at `384367c`, suite = **303 passed, 8 deselected**
**Parent goal:** close the audit backlog (P0/P1/ML done in Plan 1; this plan = P2 + Wave 1; Plan 3 = Reorg + clean)

---

## Goal

ปิด audit items 2 กลุ่มสุดท้ายก่อน reorg:

- **P2 (11 ข้อ):** A4, A6, B2, B3, B5, B6, C6, D5, D6, D7, D10 — *C7/D8 ถูก §6 มอบหมายให้ commit clean (Plan 3) แล้ว จึงไม่อยู่ใน plan นี้*
- **Wave 1 (Standards 3 / Spec 5 / Security 5 / Doc 6):** #1, #2, #3, #4, #5, #6, #7, #8, #10, #11+M9-quant, #12, #13, #14 + Doc points — *#9 ซ้ำกับ P0 C1 ทำแล้ว; §9-doc = D11 dropped (§3)*

## Architecture / Stack

- โครงสร้างเดิมทั้งหมด: `core/` (trainer_worker, dataset_builder, evaluator, estimator, hardware, ipc_bridge, compression/*), `ui/` (dashboard, controller), `scripts/` (check_runtime, benchmark_compression, pipeline_smoke), `eval.py`, `configs/safe_defaults.py`, root `plan.md` + `README.md`
- Testing: pytest, TDD red-green ทุก task, `.venv/bin/python -m pytest tests/ -q` (integration auto-deselect)
- ไม่มี library ใหม่; ไม่มี packaging/formatter/linter

## Non-goals (ห้ามแตะ)

- `tools/`, `diagnosis-issue-draft.md`, `.worktrees/`, data dirs, historical `docs/superpowers/**` — **ยกเว้น** appending disposition rows ใน spec §3 (อนุญาตไว้แล้ว)
- `is_heldout` / heldout baselines — ห้ามแตะ
- formatter/linter/pre-commit/packaging ใหม่ — ห้าม
- README แก้ได้เฉพาะ: peft known-issue (Task 9) + จุดเล็ก ๆ ของ Doc (Task 17) เท่านั้น
- **Plan 3 scope (ห้ามทำที่นี่):** C7 (`active=True`), D8 (`no_grad` ซ้ำ) — spec §6 มอบหมายให้ commit clean

## Design decisions (pin ไว้แล้ว — ห้ามเปลี่ยนระหว่าง wave)

| # | Decision |
|---|---|
| D-1 | **M9-quant gate (อนุมัติโดย user 2026-10-05):** hard gate เทียบ **fp16 ในรันเดียวกัน** — quantized variant ต้อง `EM ≥ fp16_EM − 1.0pt` และ `F1 ≥ fp16_F1 − 0.02` ไม่งั้น exit 1 บอกชื่อ variant+metric; ไม่มี fp16 ในรัน → gate ข้าม + note |
| D-2 | Threshold = `QUANT_MAX_EM_DROP_PCT = 1.0`, `QUANT_MAX_F1_DROP = 0.02` — วางข้างโค้ด gate (แบบ `MIN_DELTA_*` ใน eval.py) + value-pin ใน test file ของ script |
| D-3 | `F1_KIND = "lcs"` เป็น single source ใน `configs/safe_defaults.py` (ผู้ใช้ร่วม 3 ที่: evaluator เขียน / eval.py เทียบ / benchmark เทียบ) |
| D-4 | `DEFAULT_OUTPUT_DIR`, `EVAL_DIR`, `EVAL_N_CASES` → `configs/safe_defaults.py` (กติกา README:186) |
| D-5 | Sec-10 แก้ที่ **core** (`save_adapter_only` เรียก H1 check ร่วมกับ `validate_config`) ไม่ใช่ที่ dashboard — คุ้มครองทุก caller; deviation จาก audit ref (dashboard:212) จดไว้ด้านบน |
| D-6 | D10 แก้ด้วยการย้าย heavy import ใน `core/evaluator.py` ไป function-level (docstring เดิมสัญญาไว้แล้ว: "logic ล้วน ไม่โหลด model เองนอกจาก run_eval") — **ไม่สร้าง module ใหม่** (กันชน reorg) |
| D-7 | B6 fix = `detail` ที่บอกชื่อไฟล์เสมอ ส่งต่อถึง warning **และ** `ModelSpecUnavailable` message + `estimate().reason` (UI เห็น) |
| D-8 | zombie reset = `status → "aborted"` + `_process → None` ใน `tick()` (TERMINAL_STATUSES มี "aborted" อยู่แล้ว) |

## Global constraints (เหมือน Plan 1)

1. TDD: test RED ก่อนแก้ทุก task ที่มี test; script-only (check_runtime) = manual run
2. ค่าคงที่ใหม่ → `configs/safe_defaults.py` + pin ใน `tests/test_safe_defaults.py` **ยกเว้น** CLI gate constants (D-2, ตามแบบแผน MIN_DELTA ที่ review อนุมัติ)
3. error string ใหม่ทุกตัว = อังกฤษ
4. Full suite ต้องเขียวที่จุดจบของแต่ละ wave commit (Task 6 จบ → `fix(p2)`; Task 17 จบ → `fix(wave1)`)
5. Audit ref lines มาจากต้น branch `82452d5` (ข้าม Plan 1 มาแล้ว — บางบรรทัดเลื่อน) — ถ้า grep แล้วเจอตำแหน่งใหม่ ให้ตาม grep ไม่ใช่ตามตัวเลข

## Review Focus (จุดที่ reviewer เคยจับใน Plan 1 — เตรียมรับมือ)

- RF-1: `eval_steps=0` — transformers 5.18 อนุญาตเมื่อ `eval_strategy="no"` (validated: fallback เกิดเฉพาะ strategy=steps) แต่ถ้า trl/SFTConfig reject → ใช้ conditional-kwarg fallback (มี escape hatch ใน Task 1)
- RF-2: `on_log` return value — ต้อง return object เดิม (identity) เพื่อไม่ให้เปลี่ยน semantics ของ `call_event`
- RF-3: `EVAL_DIR` relative → absolute — ห้ามมี test ที่ pin ค่า relative (`grep -rn "data_cache/eval" tests/`)
- RF-4: Sec-10 test ต้องแยก message ออกจาก "no checkpoint exists" ให้ได้ (match ข้อความ sandbox)
- RF-5: M9 gate flip ครั้งนี้ = benchmark exit code ใหม่ (เดิมไม่มีทาง fail เลย) — test + note ต้องชัดว่าเป็น intentional behavior change
- RF-6: baseline guard ทำให้ไฟล์ eval เก่า (ไม่มี identity) ถูก skip — นั่นคือเจตนา (#11) ห้าม "แก้ให้เทียบได้เหมือนเดิม"

---

## Task 1: A4+A6 — trainer_worker: explicit `return control` + conditional `eval_steps`

**Audit:** A4 (`trainer_worker.py` `on_log` :235-273), A6 (`build_training_args` :109)
**Files:** `core/trainer_worker.py`, `tests/test_trainer_worker.py`

**Test (RED):**

```python
def test_on_log_returns_control_explicitly():
    """A4: on_log ทุกทางออกคืน control ชัดเจน — เดิม mutation อย่างเดียว อ่านไม่ออกว่าจบ method (รายงาน A4)"""
    q, cb, state, control = _cb_with_states()          # fixture เดิมจาก test_nan_trip :196
    ret = cb.on_log(None, state, control, {"loss": 0.5, "step": 1})
    assert ret is control                              # path ปกติ
    for _ in range(3):
        ret = cb.on_log(None, state, control, {"loss": float("nan"), "step": 10})
    assert ret is control                              # trip path
    assert control.should_training_stop is True
    ret = cb.on_log(None, state, control, {"loss": 0.4, "step": 11})
    assert ret is control                              # path หลัง trip
```

```python
# ต่อจาก test_args_disable_eval_when_no_heldout (:261-263) — เพิ่ม assertion:
def test_args_disable_eval_when_no_heldout():
    args = build_training_args("out", has_eval_dataset=False)
    assert args.eval_strategy == "no"
    assert args.eval_steps == 0        # A6: strategy=no ห้ามตั้ง eval_steps ที่ไม่ถูกใช้ (RED เดิม = 50)
```

**Steps:**

1. RED: เพิ่ม 2 test ข้างบน → `pytest tests/test_trainer_worker.py -q` → `test_on_log_returns_control_explicitly` fail (คืน `None`) + `test_args_disable_eval_when_no_heldout` fail (`eval_steps == 50`)
2. GREEN A4: `core/trainer_worker.py` `on_log` (`:235-273`) มีทางออก 3 จุด — เปลี่ยนเป็น `return control` ทุกจุด (คง mutation logic เดิม):
   - `:244` bare `return` (eval_loss branch) → `return control`
   - `:247` bare `return` (`"loss" not in logs` branch) → `return control`
   - trip block จบที่ `:273` (`control.should_training_stop = True`) + fall-through ของ path ปกติ → เพิ่ม `return control` เป็นบรรทัดสุดท้ายของ method
3. GREEN A6: `:109` → `eval_steps=max(VAL_EVAL_MIN_STEPS, max_steps // 10) if has_eval_dataset else 0,` + แก้ docstring `:100-101` ให้ครอบคลุม eval_steps ด้วย
   - **Escape hatch (RF-1):** ถ้า suite แดงเพราะ transformers/trl reject `eval_steps=0` → ใช้ `**({"eval_steps": ...} if has_eval_dataset else {})` แทน แล้วปรับ test ให้ assert ว่าไม่ได้ตั้งค่า (ค่า default ของ transformers)
4. VERIFY: `.venv/bin/python -m pytest tests/test_trainer_worker.py -q` — ผ่านทั้งไฟล์ โดยเฉพาะ `test_nan_trip_sends_error_and_stops` (ห้ามพลาด) และ `test_args_enable_validation_loop` (True-path ยัง `== 50`)

** checkboxes **
- [ ] RED 2 test
- [ ] `return control` ทุกทางออก + `eval_steps` conditional
- [ ] `pytest tests/test_trainer_worker.py -q` เขียว

**What NOT to do:** ห้ามแตะ logic guard/register/abort; ห้ามเปลี่ยน eval_strategy; ห้ามแก้ `save_steps`/`save_strategy`

---

## Task 2: B2+B3 — `iter_codes` hub branch: ลบ `cache_dir` + English error

**Audit:** B2 (`dataset_builder.py:321`), B3 (`:322`)
**Files:** `core/dataset_builder.py`, `tests/test_dataset_builder.py`

**Test (RED):**

```python
# 1) UPDATE test_iter_codes_limits_and_column (:273-292):
#    - fake signature: def fake_load_dataset(dataset_id, *, split, streaming):   # ลบ cache_dir
#    - ลบ lines calls["cache_dir"] (:280)
#    - expected kwargs (:287-292) → {"dataset_id": "fake/ds", "split": "train", "streaming": True}
#    RED: โค้ดจริงยังส่ง cache_dir → fake ได้ unexpected kwarg → TypeError

def test_iter_codes_hub_missing_column_gives_english_error(monkeypatch):
    """B3: streaming column ผิด → ValueError อังกฤษพร้อมรายชื่อ column ใช้ได้ (ตรง convention ฝั่ง local :285-289)"""
    monkeypatch.setattr(db, "load_dataset", lambda *a, **k: iter([{"alpha": "x = 1"}, {"beta": "y = 2"}]))
    with pytest.raises(ValueError) as excinfo:
        db.iter_codes("fake/ds", "content", limit=2)
    msg = str(excinfo.value)
    assert "Column 'content' not found in dataset 'fake/ds'" in msg
    assert "available: alpha, beta" in msg
```

**Steps:**

1. RED: อัปเดต test 1 + เพิ่ม test 2 → fail ทั้งคู่
2. GREEN B2: `iter_codes` signature `:305-311` — ลบ `cache_dir: str = "data_cache"`; `load_dataset(...)` `:321` — ลบ `cache_dir=cache_dir`
3. GREEN B3: แทน `:322`:

```python
    ds = load_dataset(dataset_id, split=split, streaming=True)
    rows = islice(iter(ds), limit)
    first = next(rows, None)
    if first is None:
        return []
    if column not in first:
        raise ValueError(
            f"Column '{column}' not found in dataset '{dataset_id}' "
            f"(available: {', '.join(first)}). Fix the Dataset column."
        )
    return [first[column], *(row[column] for row in rows)]
```

4. `grep -n "cache_dir" tests/test_dataset_builder.py core/ ui/ scripts/` — เอาออกให้หมด (callers ปัจจุบันไม่มีใครส่ง cache_dir แล้ว ยืนยันด้วย grep)
5. VERIFY: `.venv/bin/python -m pytest tests/test_dataset_builder.py -q`

** checkboxes **
- [ ] RED (kwargs pin + English error test)
- [ ] cache_dir ออก + hub ValueError
- [ ] suite file เขียว + grep cache_dir เหลือ 0 (ยกเว้น evaluator/trainer ไม่เกี่ยว)

**What NOT to do:** ห้ามแตะ local branch (`_read_local_parquet`, `_resolve_local_dataset_dir`); ห้ามเปลี่ยนข้อความ local error (test :358 pin ไว้)

---

## Task 3: B5+B6 — estimator: แคบ catch + บอกชื่อไฟล์เสมอ

**Audit:** B5 (`estimator.py:168-173`), B6 (`:155-158` + `:179` + `:220`)
**Files:** `core/estimator.py`, `tests/test_estimator.py`

**Test (RED):**

```python
def test_resolve_model_spec_missing_key_propagates(tmp_path, monkeypatch):
    """B5: config.json ไม่มี key (KeyError จาก _spec_from_config) ห้ามถูกกลืนเป็น warning+fallback"""
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path))   # convention: str (:207,:232)
    d = tmp_path / "m1"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError):
        est.resolve_model_spec("m1", None)


def test_resolve_model_spec_none_value_propagates(tmp_path, monkeypatch):
    """B5: int(None) = TypeError จาก _spec_from_config — ห้ามกลืน"""
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path))
    d = tmp_path / "m2"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"hidden_size": None}), encoding="utf-8")
    with pytest.raises(TypeError):
        est.resolve_model_spec("m2", None)


def test_resolve_model_spec_missing_config_json_names_the_file(tmp_path, monkeypatch):
    """B6: local dir ไม่มี config.json → exception ต้องบอกชื่อไฟล์ (เดิม = bare model_id)"""
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path))
    (tmp_path / "m3").mkdir()
    with pytest.raises(est.ModelSpecUnavailable, match="config.json"):
        est.resolve_model_spec("m3", None)


def test_estimate_blocked_reason_names_missing_config(tmp_path, monkeypatch):
    """B6: UI เห็นผ่าน estimate().reason — ต้องบอกไฟล์ที่หาย"""
    monkeypatch.setattr(est, "MODELS_DIR", str(tmp_path))
    (tmp_path / "m4").mkdir()
    result = est.estimate(_hw_ready(), model_id="m4", user_params_b=None)
    assert "config.json" in result.reason
```

**Steps:**

1. RED: เพิ่ม 4 test → fail ทั้งหมด (KeyError/TypeError โดนกลืน; message = bare id; reason = fixed string)
2. GREEN B5: except tuple `:168-173` → เหลือ `(OSError, ValueError)`; comment ใหม่:

```python
    except (
        OSError,   # offline/HTTP: hub 1.x — HfHubHTTPError, LocalEntryNotFoundError ⊂ OSError
        ValueError,  # JSONDecodeError, HFValidationError, int("x")
    ) as exc:
        # M3: จับเฉพาะ error ของ library — KeyError/TypeError ที่มาจาก _spec_from_config
        # (ข้อมูล config พัง/บั๊กจริง) ต้อง propagate ไม่ใช่ถูกกลืนเป็น warning + ค่าผิด (B5)
        ...
```

   - อัปเดต comment ใน `test_resolve_model_spec_known_errors_warn_and_fall_back` (`tests/test_estimator.py:335`) — เดิมเขียน "OSError/ValueError/KeyError/JSONDecodeError = ที่คาดไว้" → ตัด `KeyError` ออก (ตัว test ใช้แค่ OSError → ยังเขียว แต่ comment ห้ามโกหก)
3. GREEN B6: ใน `resolve_model_spec` — เพิ่ม `detail` ที่บอกที่อยู่ไฟล์เสมอ:

```python
    local = resolve_local_model(model_id)
    detail = f"hub config.json for {model_id!r}"   # B6: บอกไฟล์เสมอ ทุก branch
    lookup_reason: str
    try:
        if local is not None:
            cfg_path = local / "config.json"
            detail = str(cfg_path)
            with open(cfg_path, encoding="utf-8") as f:
                return _spec_from_config(json.load(f), "local_config")
        # P1 B1: pin revision ... (คงเดิม)
        path = hf_hub_download(...)
        detail = str(path)
        with open(path, encoding="utf-8") as f:
            return _spec_from_config(json.load(f), "hf_config")
    except (OSError, ValueError) as exc:
        warnings.warn(f"spec lookup failed for {model_id!r}: {detail}: {exc}", stacklevel=2)
        lookup_reason = f"{detail}: {exc}"
    # (คง P-fallback block เดิม)
    if user_params_b is None or not math.isfinite(user_params_b) or user_params_b <= 0:
        raise ModelSpecUnavailable(f"{model_id}: {lookup_reason}") from None
```

4. `core/estimator.py:220` → `f"Unknown parameter count: {exc} — enter Parameters (B) in the UI (guessing is not allowed)"` (ยืนยันว่าไม่มี test pin ข้อความเก่า — grep แล้วไม่เจอ ณ HEAD)
5. VERIFY: `.venv/bin/python -m pytest tests/test_estimator.py -q` — ไฟล์เดิมทั้งหมดต้องเขียว โดยเฉพาะ `test_resolve_model_spec_broken_local_config_uses_user_fallback` (match "broken-model" ยังอยู่ใน message ใหม่ ✓) และ `test_resolve_model_spec_known_errors_warn_and_fall_back` — ถ้า test ไหน pin format warning เก่า ให้แก้ expectation ตาม fix นี้ (มันคือ contract ที่ตั้งใจเปลี่ยน)

** checkboxes **
- [ ] RED 4 test
- [ ] except เหลือ (OSError, ValueError) + detail chain + estimate reason
- [ ] `pytest tests/test_estimator.py -q` เขียว

**What NOT to do:** ห้ามลบ fallback `user_fallback` (ยังเป็น path ถูกต้องเมื่อ user กรอก P); ห้ามแก้ `_spec_from_config` สูตรคำนวณ; `from None` คงไว้

---

## Task 4: C6 — disposition row สำหรับ Timer doc (frozen)

**Audit:** C6 (`phase4-gradio-ui-design.md:108` — `value=1000` สื่อเป็น ms แต่ gr.Timer หน่วย = วินาที; โค้ดจริง `value=1.0` ถูกแล้ว)
**Files:** `docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md` (§3 เท่านั้น — allowed)

**Steps (no test — doc row):**

1. เพิ่ม row ต่อท้ายตาราง §3 (รูปแบบเดียวกับ M9-doc/M10-doc):

```
| C6-doc | `2026-10-01-phase4-gradio-ui-design.md:108` เขียน `gr.Timer(value=1000, ...)` (สื่อหน่วย ms) แต่ gr.Timer ใช้ **วินาที** และโค้ดจริง `ui/dashboard.py` = `gr.Timer(value=1.0)` (ถูกแล้ว) | **code ถูก, doc ผิด** — phase4 เป็น historical (non-goal §7) → จดไว้ที่นี่: หน่วย Timer = วินาที; ห้ามอ้าง :108 ตีความหน่วย |
```

2. VERIFY: `grep -n "C6-doc" docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md`

** checkboxes **
- [ ] Row C6-doc อยู่ใน §3

**What NOT to do:** ห้ามแก้ไฟล์ phase4 (frozen); ห้ามแก้โค้ด (โค้ดถูกอยู่แล้ว)

---

## Task 5: D5+D6+D7 — XPU current device, disk check บน output dir, bf16 probe

**Audit:** D5 (`hardware.py:45,48` + `check_runtime.py:33,37` + `pipeline_smoke.py:40`), D6 (`check_runtime.py:44`), D7 (check_runtime ไม่ probe bf16)
**Files:** `configs/safe_defaults.py`, `core/hardware.py`, `scripts/check_runtime.py`, `scripts/pipeline_smoke.py`, `eval.py`, `ui/dashboard.py`, `tests/test_safe_defaults.py`, `tests/test_hardware.py`

**Test (RED):**

```python
# tests/test_safe_defaults.py — เพิ่มใน test รวมของ constants หรือ test ใหม่:
def test_default_output_dir_pinned():
    """D6: output dir default ต้องเป็นที่เดียวใน safe_defaults (เดิมซ้ำใน eval.py + dashboard)"""
    assert sd.DEFAULT_OUTPUT_DIR == "data_cache/finetune_run"


# tests/test_hardware.py:
def test_existing_ancestor_walks_up(tmp_path):
    """D6: output_dir อาจยังไม่ถูกสร้าง — disk check ต้องใช้ ancestor ที่มีอยู่จริง"""
    deep = tmp_path / "a" / "b" / "c"
    assert hw.existing_ancestor(deep) == tmp_path
    assert hw.existing_ancestor(tmp_path) == tmp_path


def test_inspect_uses_current_device_not_hardcoded_zero(monkeypatch):
    """D5: device index ต้องมาจาก xpu.current_device() — hardcode 0 ผิดบน multi-XPU"""
    _patch(monkeypatch)                      # fixture เดิม :18 (is_available/psutil พร้อม)
    calls: dict[str, list[int]] = {"mem": [], "name": []}
    monkeypatch.setattr(hw.torch.xpu, "current_device", lambda: 3)
    monkeypatch.setattr(
        hw.torch.xpu, "mem_get_info",
        lambda i: calls["mem"].append(i) or (11 * GB, 12 * GB),
    )
    monkeypatch.setattr(
        hw.torch.xpu, "get_device_name",
        lambda i: calls["name"].append(i) or "Intel(R) Arc(TM) B580 Graphics",
    )
    hw.inspect()
    assert calls["mem"] == [3] and calls["name"] == [3]   # RED เดิม = [0], [0]
```

**Steps:**

1. RED: เพิ่ม 3 test → `AttributeError` (DEFAULT_OUTPUT_DIR / existing_ancestor ยังไม่มี) + test current_device fail (mem/name ถูกเรียกด้วย 0)
2. GREEN constants: `configs/safe_defaults.py` += `DEFAULT_OUTPUT_DIR: str = "data_cache/finetune_run"  # spec §4 table default — เดิมซ้ำใน eval.py:29 + ui/dashboard.py:32`
3. GREEN consumers: `eval.py:29` → `from configs.safe_defaults import DEFAULT_OUTPUT_DIR` (ลบ literal, คงชื่อใช้ต่อที่ `:125`); `ui/dashboard.py:32` → import เดียวกัน (ลบ comment "ไม่อยู่ใน safe_defaults — มีที่เดียว" ที่ไม่จริง), อัปเดต `:317` — `grep -rn "_DEFAULT_OUTPUT_DIR" tests/` ถ้ามี test อ้าง ให้แก้เป็นชื่อใหม่
4. GREEN D5: `core/hardware.py:45` → `torch.xpu.mem_get_info(torch.xpu.current_device())`; `:48` → `torch.xpu.get_device_name(torch.xpu.current_device())`; `scripts/pipeline_smoke.py:40` → `mem_get_info(torch.xpu.current_device())` (+ grep `xpu:0|device=\"cuda|, 0)` ในไฟล์เดียวกัน เอาที่เหลือ)
5. GREEN D6: `core/hardware.py` เพิ่ม:

```python
def existing_ancestor(path: str | Path) -> Path:
    """เดินขึ้นจนเจอ path ที่มีอยู่จริง (output_dir อาจยังไม่ถูกสร้าง — disk อยู่ volume เดียวกัน) (D6)"""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p
```

   `ui/controller.preflight` `:94-98` → ใช้ helper นี้แทน inline walk (พฤติกรรมเท่าเดิม); `scripts/check_runtime.py:44` → `disk_free = psutil.disk_usage(existing_ancestor(DEFAULT_OUTPUT_DIR)).free / GB` + เพิ่มบรรทัด `print(f"disk path      : {existing_ancestor(DEFAULT_OUTPUT_DIR)}")`
6. GREEN Standards-3 (deviation อนุมัติ — ลบเร็วกว่ากำหนด): **ลบ try/except `:14-18` ทั้งก้อน** ใน `scripts/check_runtime.py` → เปลี่ยนเป็น plain import บรรทัดเดียว `from configs.safe_defaults import DEFAULT_OUTPUT_DIR, DISK_MIN_GB, RAM_MIN_GB` (เหตุผล: D6 ต้องใช้ `DEFAULT_OUTPUT_DIR` — ถ้าต้องรอ Task 8 จะต้องเพิ่ม fallback literal ชั่วคราวใน `except` = literal ซ้ำจะหลุดเข้า commit `fix(p2)`; comment "configs อาจยังไม่ถูกสร้าง" เป็นซากประวัติ — configs มีจริงตั้งแต่ Plan 1)
7. GREEN D7: ใน `check_runtime.main` หลัง block (4) `:65` เพิ่ม:

```python
    # (5) bf16 — trainer ใช้ bf16 จริง (fp32 probe ผ่าน ไม่ได้แปลว่า bf16 kernel ใช้ได้) (D7)
    bf16_ok = torch.xpu.is_bf16_supported()
    print(f"bf16 supported : {bf16_ok}")
    if bf16_ok:
        x16 = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16, requires_grad=True)
        w16 = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
        t16 = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
        loss16 = ((x16 @ w16) - t16).pow(2).mean()
        loss16.backward()
        g16 = x16.grad
        print(f"bf16 fwd/bwd   : loss {loss16.item():.4f}")
        if g16 is None or not torch.isfinite(g16).all():
            failures.append("bf16 forward/backward produced non-finite gradients")
        elif g16.abs().sum() == 0:
            failures.append("bf16 forward/backward produced all-zero gradients")
        else:
            print("bf16 fwd/bwd   : finite + non-zero ✓")
    else:
        print("bf16 unsupported — training would fall back to fp32")
```

8. VERIFY: `.venv/bin/python -m pytest tests/test_hardware.py tests/test_safe_defaults.py tests/test_ui_dashboard.py tests/test_ui_controller.py -q` + `.venv/bin/python scripts/check_runtime.py` (รันจริง — ขึ้นกับเครื่อง ให้สังเกตว่า section (5) โผล่ + ข้อความอังกฤษ/แสดง disk path) + `.venv/bin/python eval.py --help` (import ยังใช้ได้)

** checkboxes **
- [ ] RED 3 test
- [ ] constants + existing_ancestor + current_device ×3 file + bf16 probe
- [ ] suite targeted เขียว + manual script run

**What NOT to do:** ห้ามแก้ threshold logic RAM/disk; ห้ามแตะ `_xpu_bf16_supported` ใน trainer_worker (คนละตัว); ห้ามแก้ข้อความไทยใน check_runtime (Task 7 ทำ)

---

## Task 6: D10 — `import eval` / `core.evaluator` ห้ามดึง torch

**Audit:** D10 (`eval.py:27`)
**Files:** `core/evaluator.py`, `tests/test_eval_cli.py`, `tests/test_evaluator.py`

**Test (RED):**

```python
def test_import_eval_and_evaluator_do_not_load_torch():
    """D10: --compare เป็น pure stdlib — `import eval` ห้ามดึง torch+transformers+peft+trl ทั้งชุด"""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import core.evaluator, eval; "
         "assert 'torch' not in sys.modules, 'torch was pulled in'"],
        capture_output=True, text=True, cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr
```

**Steps:**

1. RED: เพิ่ม test → fail (`torch loaded by import eval`)
2. GREEN: `core/evaluator.py` — ย้าย heavy imports ไป function ตามแผนที่ใช้งานจริง (grep ยืนยันก่อนแก้):
   - `import torch` (`:16`) → ใช้ที่ `evaluate_cases` (`:147` `torch.no_grad`) และ `run_eval` (`:223` `torch.bfloat16`, `:231` `torch.xpu.is_available`) → local `import torch` ใน 2 function นี้
   - `from core.trainer_worker import (...)` (`:30-37`) → `evaluate_cases` ใช้แค่ `build_fim_prompt` (`:143`) · `run_eval` ใช้ `validate_config` (:190), `latest_checkpoint` (:199), `AutoTokenizer` (:205), `ensure_fim_tokens` (:206), `AutoModelForCausalLM` (:221) → ย้ายทั้งหมดเข้า 2 function นี้
   - ทุกจุดที่ย้ายใส่ comment: `# lazy import: --compare ห้ามดึง torch/transformers (D10)`
   - top-level คงเหลือ: stdlib + `configs.safe_defaults` + `core.dataset_builder` (datasets lib ไม่ดึง torch — verified)
3. **GREEN tests — ย้าย patch site (จำเป็น!):** ชื่อที่ย้ายออกไม่ใช่ module attr ของ evaluator อีก → `monkeypatch.setattr(ev, ...)` จะ raise AttributeError. อัปเดต 5 จุดใน `tests/test_evaluator.py` ให้ patch ที่ `core.trainer_worker` แทน (local import อ่านค่าตอน call time → patch นี้มีผลจริง):
   - `:208` `setattr(ev, "AutoTokenizer", ...)` → `setattr("core.trainer_worker.AutoTokenizer", ...)`
   - `:209` `setattr(ev, "ensure_fim_tokens", ...)` → `setattr("core.trainer_worker.ensure_fim_tokens", ...)`
   - `:218-219` `setattr(ev, "AutoModelForCausalLM", ...)` → `setattr("core.trainer_worker.AutoModelForCausalLM", ...)`
   - `:339`, `:340` (test_run_eval_no_cases_raises: `AutoTokenizer`/`ensure_fim_tokens`) → แบบเดียวกัน
   - **ห้ามแตะ** patch ของ `ev.iter_codes` / `ev.evaluate_cases` (ยังอยู่ระดับโมดูล — ไม่ได้ย้าย)
4. ยืนยัน docstring โมดูล (`:1-6`) ยังตรงจริง ("logic ล้วน ไม่โหลด model เองนอกจาก run_eval") ✓
5. VERIFY: `.venv/bin/python -m pytest tests/test_eval_cli.py tests/test_evaluator.py tests/test_compression_cli.py -q` — benchmark ยัง import `build_eval_cases`/`EVAL_MAX_NEW_TOKENS` จาก evaluator ได้ (ชื่อยังอยู่ระดับโมดูล; `evaluate_with_llama` อยู่ `core/compression/llama_eval.py` อยู่แล้ว — ไม่เกี่ยว), `run_eval` ยังทำงาน (import ใน function)
6. `grep -rn "from core.trainer_worker import\|import torch" core/evaluator.py` → เหลือ 0 ที่ top-level

** checkboxes **
- [ ] RED test (subprocess)
- [ ] heavy imports → function-level
- [ ] patch sites ใน test_evaluator ย้ายครบ 5 จุด
- [ ] suite targeted เขียว + grep top-level สะอาด

**What NOT to do:** ห้ามสร้าง module ใหม่/ย้าย `compare_results` (กันชน reorg Plan 3); ห้ามแก้ eval.py; ห้ามแก้ logic ของ run_eval/evaluate; ห้าม lazy import ที่ทำให้เกิด circular import (trainer_worker ไม่ได้ import evaluator ที่ top — ห้ามเพิ่ม)

---

### ▶ Wave P2 commit (หลัง Task 6)

```bash
.venv/bin/python -m pytest tests/ -q          # ต้องเขียวทั้งชุด
git add core/ configs/safe_defaults.py ui/dashboard.py ui/controller.py eval.py \
        scripts/check_runtime.py scripts/pipeline_smoke.py \
        docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md tests/
git commit -m "fix(p2): explicit on_log control, conditional eval_steps, hub cache/error surface, narrow estimator catch + file-naming, XPU current-device + output-dir disk + bf16 probe, torch-free eval imports (A4,A6,B2,B3,B5,B6,C6,D5,D6,D7,D10)"
```

- [ ] Full suite เขียวก่อน commit
- [ ] commit `fix(p2)` สร้างแล้ว

---

## Task 7: Standards #1+#2 — Thai → English

**Audit:** Wave1-1 (`dataset_builder.py:45` split_fim), Wave1-2 (`check_runtime.py:31,40,48,50,61,63`)
**Files:** `core/dataset_builder.py`, `scripts/check_runtime.py`, `tests/test_dataset_builder.py`

**Test (RED):**

```python
def test_split_fim_error_message_is_english():
    """Wave1-1: error string ที่ leak สู่ UI ต้องอังกฤษ (ข้อ 1 ใน Standards)"""
    with pytest.raises(ValueError, match="cannot split FIM"):
        db.split_fim("x = 1\n", random.Random(0))
```

**Steps:**

1. RED: เพิ่ม test → fail (ข้อความไทยไม่ match "cannot split FIM")
2. GREEN: `split_fim` raise (`core/dataset_builder.py:45` — เดิม `ตัด FIM ไม่ได้: ...`) → `raise ValueError(f"cannot split FIM: fewer than 2 line-boundary cut points (found {len(cuts)})")` (ไม่มี test/consumer pin ข้อความไทย — grep แล้ว ✓)
3. GREEN Standards-2: `scripts/check_runtime.py` — **ใช้ anchor ข้อความ ไม่ใช่เลขบรรทัด** (Task 5 ลบ try-import → เลข lines หลัง `:18` เลื่อน ~-4): `grep -nP '[\x{0E00}-\x{0E7F}]' scripts/check_runtime.py` (คำสั่งนี้ใช้ได้จริงในเครื่องนี้ — `[ก-๙]` range พังใน grep บาง locale) แล้วแปลเฉพราะข้อความ runtime (docstring/คอมเมนต์ภาษาไทย = ตาม convention โค้ด — ไม่ใช่ user-facing):
   - `FAIL: torch.xpu ไม่เห็นอุปกรณ์ (ตรวจ level-zero + intel-compute-runtime)` → `FAIL: no XPU device visible to torch.xpu (check level-zero + intel-compute-runtime)`
   - `VRAM ค่าผิดปกติ` → `VRAM values are invalid (total <= 0 or free < 0)`
   - `RAM ว่างน้อยกว่า {RAM_MIN_GB} GB` → `available RAM below {RAM_MIN_GB} GB`
   - `ดิสก์ว่างน้อยกว่า {DISK_MIN_GB} GB` → `free disk below {DISK_MIN_GB} GB`
   - `grad ไม่ finite` → `forward/backward gradient is not finite`
   - `grad ทั้งหมดเป็นศูนย์` → `forward/backward gradient is all zeros`
   - `fwd/bwd grad    : finite + มีค่า != 0 ✓` → `fwd/bwd grad    : finite + non-zero ✓`
   - (ข้อความ bf16 ที่ Task 5 เพิ่มมาเป็นอังกฤษอยู่แล้ว ✓)
4. VERIFY: `.venv/bin/python -m pytest tests/test_dataset_builder.py -q` + `grep -nP '[\x{0E00}-\x{0E7F}]' scripts/check_runtime.py` → เหลือ **แค่ docstring บนสุด + คอมเมนต์** (`# (1) identity ของ runtime` ฯลฯ — อนุญาต) ไม่มี string ข้อความ runtime ไทยเหลือ / `.venv/bin/python scripts/check_runtime.py` รันได้

** checkboxes **
- [ ] RED test
- [ ] split_fim + check_runtime เป็นอังกฤษ
- [ ] suite file เขียว

**What NOT to do:** ห้ามแก้ local parquet messages (อังกฤษอยู่แล้ว); ห้ามแก้ test ที่ pin ข้อความ local

---

## Task 8: Standards #3 — dedupe: try-import, `EVAL_DIR`, `n_cases=100`

**Audit:** Wave1-3 (`check_runtime.py:14-18`; `EVAL_DIR` 2 นิยาม evaluator:39 vs report.py:12; `n_cases=100` 4 ที่)
**Files:** `configs/safe_defaults.py`, `scripts/check_runtime.py`, `core/evaluator.py`, `core/compression/report.py`, `eval.py`, `scripts/benchmark_compression.py`, `tests/test_safe_defaults.py`

**Test (RED):**

```python
def test_eval_dir_and_n_cases_constants_pinned():
    """Standards-3: EVAL_DIR (repo-rooted) + EVAL_N_CASES = single source ใน safe_defaults"""
    assert sd.EVAL_DIR.is_absolute()
    assert str(sd.EVAL_DIR).endswith("data_cache/eval")
    assert sd.EVAL_N_CASES == 100
```

**Steps:**

1. RED: เพิ่ม test → `AttributeError`
2. GREEN constants: `configs/safe_defaults.py` เพิ่ม (ต้อง import `Path` ถ้ายังไม่มี):

```python
EVAL_DIR: Path = Path(__file__).resolve().parents[1] / "data_cache" / "eval"
# ↑ absolute จาก repo root — แทนที่ 2 นิยาม (evaluator: relative, report: absolute) (Wave1-3)
EVAL_N_CASES: int = 100  # eval set size — เดิม hardcode 4 ที่ (build_eval_cases, run_eval, eval.py, benchmark)
```

3. GREEN consumers:
   - `core/evaluator.py`: ลบ `EVAL_DIR = Path("data_cache/eval")` (`:39`) → เพิ่ม `EVAL_DIR` ใน import จาก safe_defaults (ชื่อ `ev.EVAL_DIR` คงเดิม — consumers controller/dashboard/eval.py ไม่ต้องแก้); `n_cases: int = 100` (`:95`, `:179`) → `EVAL_N_CASES`
   - `core/compression/report.py:12`: แทน def → `from configs.safe_defaults import EVAL_DIR` (แก้ comment `:11` เป็น "single source: configs.safe_defaults.EVAL_DIR"); ชื่อ `report.EVAL_DIR` คงเดิม (tests monkeypatch ✓)
   - `eval.py:123`: `default=100` → `default=EVAL_N_CASES` (เพิ่ม import)
   - `scripts/benchmark_compression.py`: `--eval-cases` default 100 → `EVAL_N_CASES` (เพิ่ม import จาก safe_defaults)
   - `scripts/check_runtime.py`: try/except `:14-18` — **ลบไปแล้วใน Task 5** (Standards-3 early removal — deviation table) → ยืนยันด้วย `grep -n "try:\|except ImportError" scripts/check_runtime.py` → ต้องว่าง + import มาจาก safe_defaults ครบ 3 ตัว (`DEFAULT_OUTPUT_DIR, DISK_MIN_GB, RAM_MIN_GB`)
4. grep guard: `grep -rn "EVAL_DIR = \|n_cases: int = 100\|default=100" --include=*.py core/ scripts/ eval.py` → ต้องไม่เหลือ definition นอก safe_defaults
5. RF-3: `grep -rn "data_cache/eval" tests/` — ถ้ามี test pin ค่า relative ให้ดูว่า pin ผ่าน `ev.EVAL_DIR` (module attr — monkeypatch pattern ใช้ได้ ✓) หรือ pin literal (ถ้า literal → แก้ expectation เป็น absolute)
6. VERIFY: `.venv/bin/python -m pytest tests/test_safe_defaults.py tests/test_evaluator.py tests/test_eval_cli.py tests/test_compression_report.py tests/test_compression_benchmark.py tests/test_ui_controller.py -q`

** checkboxes **
- [ ] RED pin test
- [ ] constants + 4 consumers + try/except ลบ
- [ ] grep ไม่มี definition ซ้ำ + suite targeted เขียว

**What NOT to do:** ห้ามเปลี่ยนค่า (100/ที่อยู่เดิม); ห้ามแตะ `VAL_EVAL_SAMPLES` (คนละตัว); ห้ามสร้าง eval dir เพิ่ม

---

## Task 9: Spec #4 + Security #13 — pin deps + peft CVE policy

**Audit:** Wave1-4 (`requirements.txt:21-22`), Wave1-13 (peft CVE-2026-71281 → §7 policy)
**Files:** `requirements.txt`, `constraints.txt`, `README.md` (branch-dependent), ไม่มี test (packaging = non-goal)

**Steps:**

1. `.venv/bin/pip show sentencepiece protobuf | grep -E "^(Name|Version)"` → ยืนยัน `0.2.2` / `7.36.2`
2. `requirements.txt:21-22` → `sentencepiece==0.2.2`, `protobuf==7.36.2`; เพิ่ม 2 บรรทัดเดียวกันใน `constraints.txt` (parity ตาม README:3)
3. **peft:** เช็คเวอร์ชันล่าสุด — `.venv/bin/pip index versions peft` หรือ `webfetch https://pypi.org/pypi/peft/json`:
   - **มีแพตช์ (เวอร์ชีนใหม่กว่า 0.21.1 ระบุแก้ CVE):** bump `peft==<ver>` ใน requirements + constraints → `.venv/bin/python -m pytest tests/ -q` → เขียว = เก็บ; แดง = revert pin แล้วใช้ branch ถัดไป + จดใน ledger
   - **ยังไม่มี:** README เพิ่ม known-issue block (ใต้ส่วน requirements/install):

```
> ⚠️ **Known issue (security):** peft 0.21.1 มี CVE-2026-71281 — ยังไม่มีแพตช์ (เช็กล่าสุด 2026-10-05) ·
> **mitigation:** pin คงไว้ใน `requirements.txt`/`constraints.txt`; ห้ามโหลด adapter/LoRA จากแหล่งที่ไม่ไว้ใจ
> (`adapter_config.json` จากแหล่งภายนอก)
```

4. จดผลลัพธ์ (branch ไหน ถูกเวอร์ชันไหน) ลง ledger — §7: ห้ามปิดเงียบ
5. VERIFY: full suite `.venv/bin/python -m pytest tests/ -q` (ถ้า bump = regression evidence; ถ้าไม่ bump = dep files ไม่กระทบ tests แต่ยังรัน)

** checkboxes **
- [ ] deps pin == installed
- [ ] peft branch ตัดสิน + (bump ผ่าน suite | README note)
- [ ] ledger จดผล
- [ ] full suite เขียว

**What NOT to do:** ห้าม bump แพ็กเกจอื่น; ห้ามเพิ่ม dependency ใหม่; ห้ามแตะ `.venv`

---

## Task 10: Spec #5+#7+#8 — live log ครบ, LoRA ranks, preflight fallback

**Audit:** Wave1-5 (`dashboard.py:43,146-151`), Wave1-7 (`:300-302`), Wave1-8 (`controller.py:103`)
**Files:** `ui/dashboard.py`, `ui/controller.py`, `tests/test_ui_dashboard.py`, `tests/test_ui_controller.py`

**Test (RED):**

```python
# tests/test_ui_dashboard.py — pattern จาก test_on_tick_returns_nine_outputs (:200)
def test_on_tick_shows_all_log_lines():
    """Spec-5: live log ครบทุกบรรทัดตาม spec phase4 — เดิมตัดที่ 200 + นับบรรทัดที่หาย"""
    from types import SimpleNamespace
    from ui.dashboard import _make_handlers

    class TickController:
        training_active = False
        def preflight(self, *a, **k): raise NotImplementedError
        def start(self, *a): raise NotImplementedError
        def abort(self): raise NotImplementedError
        def tick(self):
            return SimpleNamespace(
                status="finished", metrics=[],
                logs=[f"[INFO] line {i}" for i in range(250)],
                error=None, watchdog=None, training_active=False,
            )

    h = _make_handlers(TickController())
    out = h["on_tick"](None)
    logs_text = out[2]                      # return tuple: (status, fig, logs, banners, ...)
    assert "[INFO] line 0" in logs_text     # ครบตั้งแต่บรรทัดแรก
    assert "older lines not shown" not in logs_text


def test_lora_rank_widget_only_offers_spec_ranks():
    """Spec-7: LoRA rank ได้แค่ 8/16/32 (spec §4) — Slider step=8 เลือก 24 ได้"""
    demo = build_dashboard(FakeController())
    widgets = [b for b in _iter_blocks(demo)
               if isinstance(b, gr.Dropdown) and b.label == "LoRA rank"]
    assert len(widgets) == 1
    # gradio 6 เก็บ choices เป็น tuple pairs [("8", 8), ...] — value ต้องเป็น int
    assert [v for _label, v in widgets[0].choices] == [8, 16, 32]
```

```python
# tests/test_ui_controller.py — pattern จาก test_preflight_safe_estimate (:245) + make_controller (:72)
def test_preflight_fallback_seq_length_is_default(monkeypatch):
    """Spec-8: config ไม่มี max_seq_length → fallback = MAX_SEQ_LENGTH_DEFAULT (1024) ไม่ใช่ CAP (2048) — estimate VRAM ผิดเงียบ ๆ"""
    from configs.safe_defaults import MAX_SEQ_LENGTH_DEFAULT
    from ui import controller as controller_mod

    monkeypatch.setattr(
        controller_mod.hardware, "inspect", lambda output_dir=".": _fake_ready_hw()
    )
    seen: dict = {}

    def fake_estimate(hw, *, model_id, user_params_b=None, seq_length=0):
        seen["seq_length"] = seq_length
        return "SENTINEL"

    monkeypatch.setattr(controller_mod, "estimate", fake_estimate)
    cfg = valid_config()
    del cfg["max_seq_length"]
    ctl, _fp, _state = make_controller()
    assert ctl.preflight(cfg) == "SENTINEL"
    assert seen["seq_length"] == MAX_SEQ_LENGTH_DEFAULT   # RED เดิม = 2048
```

**Steps:**

1. RED: เพิ่ม 3 test → fail ทั้งหมด (log ถูกตัด / widget ยังเป็น Slider / fallback = 2048)
2. GREEN Spec-5: `ui/dashboard.py` ลบ `_LOG_TAIL_LINES` (`:43`) และแทน `:146-151` เป็น `logs = "\n".join(snap.logs)` (ลบ branch "+N older")
3. GREEN Spec-7: `:300-302` → `lora_in = gr.Dropdown(choices=[8, 16, 32], value=LORA_RANK_DEFAULT, label="LoRA rank")` — `_collect_config` มี `_to_int(lora_rank, ...)` ดูแลค่าแล้ว (`:76`); `test_...lora_rank == LORA_RANK_DEFAULT` (`:88`) ยังผ่าน
4. GREEN Spec-8: `ui/controller.py:25` import เปลี่ยน `MAX_SEQ_LENGTH_CAP` → `MAX_SEQ_LENGTH_DEFAULT` (ยืนยัน grep ว่า CAP ไม่ถูกใช้ที่อื่นในไฟล์นี้ — ณ HEAD มีแค่ `:103`); `:103` default → `MAX_SEQ_LENGTH_DEFAULT`
5. VERIFY: `.venv/bin/python -m pytest tests/test_ui_dashboard.py tests/test_ui_controller.py -q`

** checkboxes **
- [ ] RED 3 test
- [ ] log unbounded + Dropdown + DEFAULT fallback
- [ ] suite UI เขียว

**What NOT to do:** ห้ามแก้ seq slider (ยังใช้ CAP เป็นเพดาน); ห้ามแก้ `_to_int`/`_collect_config` logic; ห้ามแตะ concurrency wiring (ทำแล้วใน Plan 1)

---

## Task 11: Spec #6 — report path docstring ตรงโค้ด

**Audit:** Wave1-6 (`compression/config.py:4` docstring ไม่ตรง `:121-122`)
**Files:** `core/compression/config.py`

**Steps (no test — comment/docstring):**

1. อ่าน `:1-10` — docstring ปัจจุบันอ้าง `benchmarks/<source>/<variant>.json` (ไม่มี hash)
2. แก้ให้ตรงโค้ดจริง (`report_id` `:112-118` = basename + sha256[:8] ของ resolved path) + README:150 (`benchmarks/<model>-<hash>/<variant>.json` — เขียนให้ตรงรูป README ทั้งอัน) — ถ้ามี comment อ้าง phase6a design layout (`benchmarks/compression/...`) ให้ชี้ว่า README (living) เป็นแหล่งจริง + phase6a = historical (frozen)
3. VERIFY: `.venv/bin/python -m pytest tests/test_compression_config.py -q` (test pin `p.parent.name.startswith("m")` ไม่กระทบ)

** checkboxes **
- [ ] docstring ตรง code+README
- [ ] suite config เขียว

**What NOT to do:** ห้ามเปลี่ยนคืน `report_path()` (โค้ดถูกแล้ว — README/tests เห็นตรงกัน); ห้ามแก้ phase6a docs (frozen)

---

## Task 12: Security #10 — `save_adapter_only` ต้องผ่าน H1 sandbox

**Audit:** Wave1-10 (`dashboard.py:212-217` → `trainer_worker.py` save)
**Files:** `core/trainer_worker.py`, `tests/test_trainer_worker.py`

**Test (RED):**

```python
def test_save_adapter_only_rejects_output_outside_sandbox(tmp_path, monkeypatch):
    """Sec-10: on_save_adapter ไม่ผ่าน validate_config → คัดลอกจากนอก sandbox ได้ — H1 rule ต้อง applied ที่ save (คุ้มครองทุก caller)"""
    run = tmp_path / "run"
    ckpt = run / "checkpoint-3"
    ckpt.mkdir(parents=True)                    # fixture ลอกจาก test_save_adapter_only_copies (:347-350)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")
    # หัน temp root ออกจาก tmp_path → path นี้ "นอก sandbox" ทั้งที่อยู่ใน tmp (hermetic)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="under the repo or the system temp dir"):
        wa.save_adapter_only(run, exports_dir=tmp_path / "exports")
```

**Steps:**

1. RED: เพิ่ม test → fail (ปัจจุบันคัดลอกสำเร็จ ไม่ raise)
2. GREEN: ใน `core/trainer_worker.py` แยก H1 check จาก `validate_config` (`:320-327`) เป็น function ใหม่ (วางก่อน `validate_config`):

```python
def validate_output_dir(raw_output: str | Path) -> Path:
    """H1: output_dir ต้องอยู่ใต้ repo หรือ system temp — ใช้ร่วม validate_config + save_adapter_only
    (commit_checkpoint ทำ rename/rmtree ใน output_dir → กัน data loss นอก sandbox, Sec-10)
    """
    output = Path(raw_output).expanduser().resolve()  # expanduser ก่อน: "~/..." จะไม่หนี sandbox
    allowed = project_roots(REPO_ROOT) + (Path(tempfile.gettempdir()).resolve(),)
    if not any(output.is_relative_to(root) for root in allowed):
        raise ValueError(
            f"output_dir must be under the repo or the system temp dir (got: {raw_output!r})"
        )
    return output
```

   - `validate_config`: แทน inline block ด้วย `validate_output_dir(config["output_dir"])` (message คงเดิมทุกตัวอักษร)
   - `save_adapter_only`: บรรทัดแรกของ function → `validate_output_dir(output_dir)` ก่อน `latest_checkpoint`
3. VERIFY: `.venv/bin/python -m pytest tests/test_trainer_worker.py -q` — validate_config tests เดิมเขียว (message เท่าเดิม), save tests ใช้ tmp_path (allowed ✓) เขียว
4. RF-4: `grep -n "save_adapter_only" tests/test_trainer_worker.py` — test ไหนใช้ path นอก tmp/repo ต้องปรับ (คาดว่าไม่มี)

** checkboxes **
- [ ] RED test
- [ ] `validate_output_dir` แยก + validate_config/save ใช้ร่วม
- [ ] suite file เขียว

**What NOT to do:** ห้ามแก้ dashboard handler (core gate คุ้มครองกว่า — deviation จดแล้ว D-5); ห้ามเปลี่ยนข้อความ H1; ห้ามแตะ `exports/` dest logic

---

## Task 13: Security #12 — `latest_checkpoint` กู้ interrupted swap

**Audit:** Wave1-12 (`commit_checkpoint` rename 2 ขั้น → SIGKILL กลางทาง)
**Files:** `core/trainer_worker.py`, `tests/test_trainer_worker.py`

**Facts:** `commit_checkpoint` (`:185-205`) self-heal ที่ save ถัดไปได้อยู่แล้ว (rename tmp→final ทำงานเมื่อ final หาย) — **ช่องว่างจริง = resume path**: kill หลัง `rename(final→old)` ก่อน `rename(tmp→final)` → `latest_checkpoint` มองไม่เห็นทั้ง old (.old ไม่ผ่าน `is_checkpoint_dir`) และ tmp → resume เริ่มใหม่เงียบ ๆ

**Test (RED):**

```python
def _mk_adapter_ckpt(path: Path, marker: str | None = None) -> None:
    """fixture: checkpoint dir ที่ผ่าน is_checkpoint_dir + มี adapter files (ลอกจาก save tests :347-350)"""
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"fake")
    if marker is not None:
        (path / "marker").write_text(marker, encoding="utf-8")


def test_latest_checkpoint_restores_old_after_interrupted_swap(tmp_path):
    """Sec-12: SIGKILL หลัง final→*.old → resume ต้องกู้ .old กลับ (เดิม: checkpoint หายจากรัน เงียบ ๆ)"""
    root = tmp_path / "out"
    _mk_adapter_ckpt(root / "checkpoint-5.old")
    assert wa.latest_checkpoint(root) == root / "checkpoint-5"
    assert not (root / "checkpoint-5.old").exists()


def test_latest_checkpoint_prefers_saving_over_old(tmp_path):
    """Sec-12: มีทั้ง .old (เก่า) และ .saving (ใหม่ — super()._save เสร็จแล้วตอน commit เริ่ม) → ต้องเอา .saving"""
    root = tmp_path / "out"
    _mk_adapter_ckpt(root / "checkpoint-5.old", marker="old")
    _mk_adapter_ckpt(root / "checkpoint-5.saving", marker="new")
    restored = wa.latest_checkpoint(root)
    assert restored == root / "checkpoint-5"
    assert (restored / "marker").read_text(encoding="utf-8") == "new"
    assert not (root / "checkpoint-5.old").exists()
    assert not (root / "checkpoint-5.saving").exists()


def test_latest_checkpoint_does_not_trust_partial_saving(tmp_path):
    """Sec-12: .saving ที่ไม่มี adapter files (first save โดน kill กลางเขียน) — ห้ามกู้ (อาจ partial)"""
    root = tmp_path / "out"
    saving = root / "checkpoint-7.saving"
    saving.mkdir(parents=True)
    (saving / "trainer_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no checkpoint found"):   # ข้อความ raise เดิม (:479)
        wa.latest_checkpoint(root)
```

(หมายเหตุ: module alias ใน `tests/test_trainer_worker.py` = `wa` ไม่ใช่ `tw` — :9)

**Steps:**

1. RED: เพิ่ม 3 test → **test 1-2 fail** (ไม่กู้ .old / เอา .old เก่าทั้งที่ .saving ใหม่กว่า); **test 3 = guard เขียวตั้งแต่ต้น** (ยังไม่มี recovery → raise ตามเดิม) — คอยกันไม่ให้ recovery ไปกู้ .saving ที่เขียนไม่ครบ
2. GREEN: ใน `core/trainer_worker.py` หลัง `is_checkpoint_dir` เพิ่ม:

```python
def _has_adapter_files(ckpt: Path) -> bool:
    return (ckpt / "adapter_config.json").is_file() and bool(
        list(ckpt.glob("adapter_model*.safetensors"))
    )


def _recover_interrupted_commit(root: Path) -> None:
    """Sec-12: กู้ swap ที่ค้างกลางทาง (SIGKILL หลัง final→*.old) — เรียกก่อน scan ใน latest_checkpoint

    - base หาย + มี .old และ .saving → เอา .saving (new — `super()._save` เสร็จแล้วตอน commit เริ่ม)
    - base หาย + มีแค่ .old → คืน .old (checkpoint จริงที่ใช้ได้)
    - base หาย + มีแค่ .saving โดยไม่มี adapter files → ไม่แตะ (อาจเขียนไม่ครบ)
    """
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        name = child.name
        if name.endswith(".old"):
            base = child.with_name(name[: -len(".old")])
            if not is_checkpoint_dir(base) or base.exists():
                continue
            saving = root / f"{base.name}.saving"
            if saving.is_dir() and _has_adapter_files(saving):
                os.rename(saving, base)
                shutil.rmtree(child, ignore_errors=True)
            else:
                os.rename(child, base)
        elif name.endswith(".saving"):
            base = child.with_name(name[: -len(".saving")])
            if not is_checkpoint_dir(base) or base.exists():
                continue
            if _has_adapter_files(child):
                os.rename(child, base)
```

   - `latest_checkpoint`: บรรทัดแรก → `_recover_interrupted_commit(Path(output_dir))` (ก่อน scan)
   - ยืนยัน `os`/`shutil` import มีอยู่แล้วในไฟล์ (commit_checkpoint ใช้อยู่ ✓)
3. `commit_checkpoint` **ไม่แก้** (self-heal อยู่แล้ว — เขียนโน้ตใน docstring ว่า resume path คืนผ่าน `_recover_interrupted_commit`)
4. VERIFY: `.venv/bin/python -m pytest tests/test_trainer_worker.py -q` — `is_checkpoint_dir`/commit tests เดิมเขียว
5. ตรวจด้วยตา: `_recover` แตะเฉพาะชื่อ `checkpoint-*` (`is_checkpoint_dir` gate) — ไฟล์อื่นใน run dir ไม่โดน

** checkboxes **
- [ ] RED 3 test
- [ ] `_has_adapter_files` + `_recover_interrupted_commit` + call ใน latest_checkpoint
- [ ] suite file เขียว

**What NOT to do:** ห้ามแก้ `commit_checkpoint` sequence; ห้ามแตะ `AtomicSaveTrainer._save`; ห้ามลบ .saving/.old ที่ base ยังอยู่ (ขอบเขตแค่ recovery เมื่อ base หาย)

---

## Task 14: Security #14 — zombie tick คืนสถานะ + ปลดล็อก Start

**Audit:** Wave1-14 (`controller.tick` — watchdog ไม่ reset → `training_active` ค้าง True → Start ล็อกถาวร)
**Files:** `ui/controller.py`, `tests/test_ui_controller.py`

**Test (RED):**

```python
def test_zombie_tick_resets_status_and_unlocks_start():
    """Sec-14: zombie ไม่ reset status → ปุ่ม Start ล็อกถาวร — tick ต้องคืนสถานะ terminal + หยุดกรี๊ดซ้ำ"""
    fp = FakeProcess(alive=False)  # ตายแล้วตั้งแต่ก่อน start tick
    ctl, _fp, state = make_controller(process=fp)
    ctl.start(valid_config())
    # start() drain queue เก่าทิ้ง (I4) — inject หลัง start แบบ test_zombie_watchdog_after_drain :281-294
    state["queue"].messages.extend(
        [status_msg("training"), error_msg("boom", "tb-line")]
    )
    assert ctl.training_active is True  # BUG ก่อนแก้: process ตายแล้วแต่ status non-terminal → Start ล็อก
    snap = ctl.tick()
    assert snap.watchdog is not None and "zombie" in snap.watchdog   # ยังรายงานครั้งแรก
    assert snap.status == "aborted"
    assert snap.training_active is False
    snap2 = ctl.tick()
    assert snap2.watchdog is None          # ไม่ spam
    assert ctl.training_active is False    # Start ปลดล็อก (training_active = process and not terminal)
```

**Steps:**

1. RED: เพิ่ม test → fail (`status` ยัง "training", `training_active` True)
2. GREEN: `ui/controller.py` `tick()` `:169-186` — หลังคำนวณ `watchdog` และก่อน build snapshot:

```python
            if watchdog is not None:
                # Sec-14: zombie = process ตายแล้ว — คืนสถานะ terminal ทันที
                # (ไม่งั้น training_active ค้าง True = Start ล็อกถาวร)
                self._status = "aborted"
                self._process = None
```

   คง `watchdog_text` build จากค่าก่อน reset (ยังรายงาน "zombie" ใน tick แรก ✓)
3. VERIFY: `.venv/bin/python -m pytest tests/test_ui_controller.py -q` — `test_zombie_watchdog_after_drain` (`:281`) และ `test_watchdog_silent_while_aborting` (`:544`) ต้องเขียว
4. ตรวจอ้อม: `_drain_queue` ยังถูกเรียกทุก tick (process None ไม่ block drain) ✓ — ข้อความค้างใน queue จากรอบ zombie ยังเข้า

** checkboxes **
- [ ] RED test
- [ ] reset ใน tick + คง watchdog_text
- [ ] suite controller เขียว

**What NOT to do:** ห้ามแก้ `watchdog_error` ใน ipc_bridge; ห้ามแก้ `abort()`/`_handle_message`; ห้ามเพิ่ม status ใหม่ (ใช้ "aborted" ที่มี)

---

## Task 15: eval markers — `f1_kind` + dataset identity ในผล eval + guard ใน `--compare`

**Audit:** review#5 (metric-version marker) + prerequisite ของ #11
**Files:** `configs/safe_defaults.py`, `core/evaluator.py`, `eval.py`, `tests/test_safe_defaults.py`, `tests/test_evaluator.py`, `tests/test_eval_cli.py`

**Test (RED):**

```python
# tests/test_safe_defaults.py:
def test_f1_kind_pinned():
    assert sd.F1_KIND == "lcs"


# tests/test_evaluator.py — pattern ครบจาก test_run_eval_json_reports_skipped_long_middle (:193):
# args = (monkeypatch, tmp_path) · setup = iter_codes + fake tok/model + evaluate_cases fake
# ⚠ ใช้ string-form patch "core.trainer_worker.*" (แบบหลัง Task 6 ย้าย import — ห้ามใช้ setattr(ev, ...))
def test_run_eval_stamps_metric_kind_and_identity(monkeypatch, tmp_path):
    """review#5: ผล eval ต้องระบุ f1_kind + dataset identity — กันเทียบข้าม metric version/ชุดข้อมูล"""
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: _codes_where(True, 3))

    class _FakeAutoTok:
        @staticmethod
        def from_pretrained(model_id):
            return FakeTok()

    monkeypatch.setattr("core.trainer_worker.AutoTokenizer", _FakeAutoTok)
    monkeypatch.setattr("core.trainer_worker.ensure_fim_tokens", lambda tok, vals: {})

    class _FakeModel:
        def to(self, *_args):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        "core.trainer_worker.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel()),
    )
    monkeypatch.setattr(
        ev,
        "evaluate_cases",
        lambda *a, **k: {
            "exact_match_pct": 0.0,
            "token_f1_mean": 0.0,
            "per_case": [],
        },
    )
    cfg = _valid_config()
    result = ev.run_eval(cfg, mode="base", eval_dir=tmp_path)
    assert result["f1_kind"] == "lcs"
    assert result["dataset_id"] == cfg["dataset_id"]
    assert result["dataset_column"] == cfg["dataset_column"]
    on_disk = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert on_disk["f1_kind"] == "lcs"                     # JSON ที่เขียนก็มี key ครบ
    assert on_disk["dataset_id"] == cfg["dataset_id"]


# tests/test_eval_cli.py — fixture helpers จริง: _result (:10) + _write_two (:21)
def test_compare_rejects_f1_kind_mismatch(tmp_path, capsys):
    """review#5: ต่าง f1_kind → ห้ามเทียบ (คนละความหมายของ metric)"""
    base = {**_result("base", em=5.0, f1=0.2), "f1_kind": "lcs"}
    fine = _result("finetuned", em=6.0, f1=0.25)          # legacy = ไม่มี key
    eval_dir = _write_two(tmp_path, base, fine)
    assert cli_main(["--compare", "--eval-dir", eval_dir]) == 2
    assert "metric semantics differ" in capsys.readouterr().err


def test_compare_rejects_dataset_mismatch(tmp_path, capsys):
    base = {**_result("base", em=5.0, f1=0.2),
            "f1_kind": "lcs", "dataset_id": "a/ds", "dataset_column": "content"}
    fine = {**_result("finetuned", em=6.0, f1=0.25),
            "f1_kind": "lcs", "dataset_id": "b/ds", "dataset_column": "content"}
    eval_dir = _write_two(tmp_path, base, fine)
    assert cli_main(["--compare", "--eval-dir", eval_dir]) == 2
    assert "different datasets" in capsys.readouterr().err
```

(หมายเหตุ: legacy = existing compare tests ที่ fixture ทั้งคู่ไม่มี keys → `None==None` → ยัง compare ผ่าน = backward-compat guard; `_result`/`_write_two` มีจริงที่ `tests/test_eval_cli.py:10/:21` — ลอก signature จาก `test_compare_fail_on_tie` (`:36`))

**Steps:**

1. RED: เพิ่ม 3-4 test → fail
2. GREEN constants: `configs/safe_defaults.py` += `F1_KIND: str = "lcs"  # ความหมายของ token_f1 — เทียบได้เฉพาะ kind เดียวกัน (M10)`; pin ใน test_safe_defaults
3. GREEN evaluator: `run_eval` result dict (`:238-246`) เพิ่ม:

```python
        "f1_kind": F1_KIND,
        "dataset_id": config["dataset_id"],
        "dataset_column": config["dataset_column"],
```

   (+ import F1_KIND จาก safe_defaults)
4. GREEN eval.py `_run_compare` — หลัง n-guard (`:64-70`) เพิ่ม 2 guard (ก่อน `ev.compare_results`):

```python
    if base.get("f1_kind") != fine.get("f1_kind"):
        print(
            f"Cannot compare: metric semantics differ (base f1_kind={base.get('f1_kind')!r} "
            f"vs finetuned f1_kind={fine.get('f1_kind')!r}) — rerun both modes with the current build",
            file=sys.stderr,
        )
        return 2
    if (base.get("dataset_id"), base.get("dataset_column")) != (
        fine.get("dataset_id"),
        fine.get("dataset_column"),
    ):
        print(
            f"Cannot compare: eval sets come from different datasets "
            f"(base {base.get('dataset_id')!r}/{base.get('dataset_column')!r} vs "
            f"finetuned {fine.get('dataset_id')!r}/{fine.get('dataset_column')!r}) — "
            "rerun both modes on the same dataset",
            file=sys.stderr,
        )
        return 2
```

5. VERIFY: `.venv/bin/python -m pytest tests/test_eval_cli.py tests/test_evaluator.py tests/test_safe_defaults.py -q` — compare tests เดิม (fixtures ไม่มี keys ทั้งคู่ → None==None → ผ่าน = legacy compat) เขียว, n-mismatch test ยัง exit 2

** checkboxes **
- [ ] RED tests
- [ ] F1_KIND + writer fields + 2 guards
- [ ] suite targeted เขียว (legacy compare ยังผ่าน)

**What NOT to do:** ห้ามแก้ metric math (LCS token_f1 — Plan 1 ทำแล้ว); ห้ามแก้ samples/per_case schema; ห้ามลบ guard n

---

## Task 16: Security #11 + M9-quant — baseline identity guard + quant-regression gate

**Audit:** Wave1-11 (`benchmark_compression.py` delta เทียบเฉพาะ n) + M9 quant half (deferred จาก Plan 1; design = D-1/D-2)
**Files:** `scripts/benchmark_compression.py`, `tests/test_compression_cli.py`

**Test (RED):** (ทุก test ในไฟล์นี้ใช้ loader `cli = _load_cli()` — `tests/test_compression_cli.py:18`; **ไม่**ใช่ `import scripts.benchmark_compression` — `scripts/` ไม่มี `__init__.py`)

```python
def test_quant_gate_constants_pinned():
    """M9-quant (D-2): threshold ตรึง — เปลี่ยน = แก้ test+spec ด้วย"""
    cli = _load_cli()
    assert cli.QUANT_MAX_EM_DROP_PCT == 1.0
    assert cli.QUANT_MAX_F1_DROP == 0.02


def _row(variant, em, f1):
    # key ตรง build_report (core/compression/report.py:41,:54,:55) — verified
    return {"variant": variant, "exact_match_pct": em, "token_f1": f1}


def test_check_quant_regression_flags_em_drop_beyond_tolerance():
    cli = _load_cli()
    rows = [_row("fp16", 50.0, 0.60), _row("q8_0", 49.5, 0.595), _row("q4_k_m", 48.4, 0.58)]
    out = cli.check_quant_regression(rows)
    # q4_k_m: EM drop 1.6 > 1.0 → flag; q8_0 drop 0.5 ≤ 1.0 → ผ่าน; F1 drops 0.005/0.02 ไม่เกิน → ไม่ flag
    assert len(out) == 1
    assert "q4_k_m" in out[0] and "EM" in out[0]


def test_check_quant_regression_passes_at_boundary():
    cli = _load_cli()
    rows = [_row("fp16", 50.0, 0.60), _row("q4_k_m", 49.0, 0.58)]
    assert cli.check_quant_regression(rows) == []   # drop == threshold ไม่ใช่ violation (>) + f1 0.02 == ไม่เกิน


def test_check_quant_regression_skips_without_fp16():
    cli = _load_cli()
    assert cli.check_quant_regression([_row("q8_0", 40.0, 0.4)]) == []


def test_main_fails_on_quant_regression(monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "run_benchmark", lambda args: [_row("fp16", 50.0, 0.6), _row("q4_k_m", 40.0, 0.3)])
    assert cli.main(["--model", "dummy"]) == 1        # RF-5: intentional exit-code change เดิมคืน 0 เสมอ


def test_main_passes_when_within_tolerance(monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "run_benchmark", lambda args: [_row("fp16", 50.0, 0.6), _row("q4_k_m", 49.6, 0.595)])
    assert cli.main(["--model", "dummy"]) == 0


def _baseline_args(cli, baseline_file):
    """args ผ่าน parser จริง → dataset/column = DEFAULT_DATASET_ID/COLUMN (ค่า default เดียวกับรันจริง)"""
    return cli.build_parser().parse_args(
        ["--model", "m", "--baseline", str(baseline_file)]
    )


def _write_baseline(tmp_path, **extra):
    bl = tmp_path / "baseline.json"
    bl.write_text(
        json.dumps({"n": 2, "exact_match_pct": 1.0, "token_f1_mean": 0.1, **extra}),
        encoding="utf-8",
    )
    return bl


def test_baseline_dict_skips_identity_mismatch(tmp_path, capsys):
    """#11: baseline ต้องมี dataset/column ตรงกับรัน — ไม่งั้น delta = คนละชุด"""
    cli = _load_cli()
    bl = _write_baseline(tmp_path, dataset_id="other/ds", dataset_column="content",
                         f1_kind="lcs")
    args = _baseline_args(cli, bl)
    assert cli._baseline_dict(args, tmp_path / "src", 2) is None
    err = capsys.readouterr().err
    assert "baseline skipped" in err and "identity" in err   # note เป็นอังกฤษ


def test_baseline_dict_skips_legacy_without_identity(tmp_path, capsys):
    """ไฟล์ eval ยุคก่อน markers — เทียบไม่ได้ → note ให้รัน eval ใหม่ (RF-6: เจตนา)"""
    cli = _load_cli()
    bl = _write_baseline(tmp_path)                      # ไม่มี identity keys
    args = _baseline_args(cli, bl)
    assert cli._baseline_dict(args, tmp_path / "src", 2) is None
    assert "identity" in capsys.readouterr().err


def test_baseline_dict_returns_matching(tmp_path):
    cli = _load_cli()
    bl = _write_baseline(tmp_path, dataset_id=cli.DEFAULT_DATASET_ID,
                         dataset_column=cli.DEFAULT_DATASET_COLUMN, f1_kind="lcs")
    args = _baseline_args(cli, bl)
    out = cli._baseline_dict(args, tmp_path / "src", 2)
    assert out is not None and out["n"] == 2
```

**Steps:**

1. RED: เพิ่ม test ทั้งหมด → fail (`QUANT_*` ไม่มี, `check_quant_regression` ไม่มี, `_baseline_dict` คืน baseline แม้ identity ไม่ตรง) · **เพิ่มด้วย** RED จาก guard ใหม่: `test_delta_shown_when_case_count_matches` (`tests/test_compression_cli.py:346`) เดิม assert `captured["baseline"][...]` — fixture ไม่มี identity → หลัง GREEN baseline จะเป็น `None` → test นี้แดงตอน RED-ใหม่ → อัปเดต fixture ในขั้น GREEN (ดู step 4)
2. ยืนยัน key names: อ่าน `build_report` (**`core/compression/report.py:15`** — `variant` :41, `exact_match_pct` :54, `token_f1` :55 — verified แล้ว; ถ้าไม่ตรง ปรับ test helper)
3. GREEN: `scripts/benchmark_compression.py` เพิ่ม:

```python
# M9-quant (D-1/D-2): quantized ต้องไม่แย่กว่า fp16 ในรันเดียวกันเกิน threshold — hard gate
QUANT_MAX_EM_DROP_PCT: float = 1.0
QUANT_MAX_F1_DROP: float = 0.02


def check_quant_regression(rows: list[dict]) -> list[str]:
    """คืน violation list — variant ที่ไม่ใช่ fp16 แย่กว่า fp16 (รันเดียวกัน = identity ตรง by construction)

    ไม่มี fp16 หรือไม่มี metrics (--no-eval) → [] (gate ข้าม — main พิมพ์ note)
    """
    fp16 = next(
        (r for r in rows
         if r.get("variant") == "fp16" and r.get("exact_match_pct") is not None),
        None,
    )
    if fp16 is None:
        return []
    out = []
    for r in rows:
        if r is fp16 or r.get("exact_match_pct") is None:
            continue
        em_drop = fp16["exact_match_pct"] - r["exact_match_pct"]
        if em_drop > QUANT_MAX_EM_DROP_PCT:
            out.append(
                f"{r['variant']}: EM drop {em_drop:.1f}pt vs fp16 exceeds {QUANT_MAX_EM_DROP_PCT}pt"
            )
        if fp16.get("token_f1") is None or r.get("token_f1") is None:
            continue                      # build_report ค่า float | None — ข้ามถ้า eval ฝั่งไหนไม่มี
        f1_drop = fp16["token_f1"] - r["token_f1"]
        if f1_drop > QUANT_MAX_F1_DROP:
            out.append(
                f"{r['variant']}: token F1 drop {f1_drop:.2f} vs fp16 exceeds {QUANT_MAX_F1_DROP}"
            )
    return out
```

   - `_baseline_dict` (`scripts/benchmark_compression.py:116-141`) — **กลับ n-check เป็น early-return** แล้วค่อยใส่ 2 guard ก่อน `return baseline` (ถ้าแทรกแบบ "ต่อท้าย" โดยไม่กลับ n-check = หลัง `return baseline` = dead code!):

```python
    if baseline.get("n") != built_cases:
        # Review #2: จำนวนเคสไม่ตรง = คนละการทดสอบ → ห้ามเทียบ (ตรง convention eval.py --compare)
        print(
            f"note: baseline skipped (case count differs: baseline n={baseline.get('n')!r}, "
            f"run n={built_cases}) — deltas require the same evaluation",
            file=sys.stderr,
        )
        return None
    # #11 identity guard: dataset/column/f1_kind ไม่ตรง = คนละ eval → delta โกหก
    if baseline.get("dataset_id") != args.dataset or baseline.get("dataset_column") != args.column:
        print(
            f"note: baseline skipped (eval identity differs or missing: baseline="
            f"{baseline.get('dataset_id')!r}/{baseline.get('dataset_column')!r} vs run="
            f"{args.dataset!r}/{args.column!r}) — rerun eval with this build",
            file=sys.stderr,
        )
        return None
    if baseline.get("f1_kind") != F1_KIND:
        print(
            f"note: baseline skipped (f1_kind={baseline.get('f1_kind')!r} != {F1_KIND!r}) — "
            "rerun eval with the current build",
            file=sys.stderr,
        )
        return None
    return baseline
```

   (+ import `F1_KIND` จาก safe_defaults — เพิ่มในบล็อก import `:19-23`)

   **ผลกระทบทดสอบเดิม (ต้องเข้าใจก่อนแก้):**
   - `test_delta_omitted_when_case_count_differs` (`:317`) — n ไม่ตรง → early-return แรก fire **ก่อน** identity → ข้อความ "case count" เหมือนเดิม ✓ ไม่ต้องแก้
   - `test_table_printed_with_baseline_delta` (`:162`, `--no-eval`) — `built_cases is None` early-return อยู่ **ก่อน** guards → pass-through เหมือนเดิม ✓ ไม่ต้องแก้
   - `test_delta_shown_when_case_count_matches` (`:346`) — RED หลัง guard → **แก้ fixture** เพิ่ม identity:

```python
    baseline_path.write_text(
        json.dumps({"exact_match_pct": 10.0, "token_f1_mean": 0.5, "n": 100,
                    "dataset_id": cli.DEFAULT_DATASET_ID, "dataset_column": cli.DEFAULT_DATASET_COLUMN,
                    "f1_kind": "lcs"}),
        encoding="utf-8",
    )
```

   - `main()` — รับ rows + gate:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = run_benchmark(args)
    except CompressionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # M9-quant hard gate (D-1): exit 1 เมื่อ quantized แย่กว่า fp16 เกิน threshold
    violations = check_quant_regression(rows)
    if violations:
        print("FAIL — quant regression vs fp16 in the same run:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    if not args.no_eval and not any(r.get("variant") == "fp16" for r in rows):
        print("note: quant-regression gate skipped (fp16 not in this run)", file=sys.stderr)
    return 0
```

4. อัปเดต fixture `test_delta_shown_when_case_count_matches` (บล็อกด้านบน) → GREEN; VERIFY: `.venv/bin/python -m pytest tests/test_compression_cli.py tests/test_compression_benchmark.py -q` + `.venv/bin/python scripts/benchmark_compression.py --help` (ไม่ต้องรัน benchmark จริง)
5. ถ้ามี test เดิมที่ assert `main == 0` ทุกกรณี → แก้ตาม behavior ใหม่ (RF-5 intentional) — ณ HEAD: `:194` (usage exit 2) / `:289` (error rc) ไม่กระทบ, ไม่มี test อื่น assert rc ตรง ๆ

** checkboxes **
- [ ] RED tests
- [ ] QUANT_* + check_quant_regression + baseline guards + main gate
- [ ] suite benchmark เขียว + --help ใช้ได้

**What NOT to do:** ห้ามเปลี่ยน threshold (D-1/D-2 pin แล้ว); ห้ามแก้ format_table/report schema; ห้าม gate กับ baseline JSON (อนุมัติแล้วว่า gate = fp16 same-run); ห้ามลบ delta table (ยังแสดงตามเดิม แค่เพิ่ม identity guard)

---

## Task 17: Doc points — plan.md/README ตรงพฤติกรรมจริง

**Audit:** Wave1 Doc 6 ข้อ (ข้าม §9 = D11 dropped): Timer(C6 → Task 4 แล้ว), logs/Step (A3-doc), mem_get_info/get_device_name (D5-doc), §4 pyarrow (B4-doc), label→legend (C4-doc → row ใน task นี้), + watchdog note (พฤติกรรมใหม่ Sec-14)
**Files:** `plan.md` (root — living doc ✓), `README.md`, spec §3 (rows: C4-doc)

**Steps (no tests — docs; verify ด้วย grep):**

1. **§4/แผนภาพ vs pyarrow:** อ่าน `plan.md:70-76` — แทน `Prepare HF Dataset via ConstantLengthDataset` (`:73`) ให้ตรงจริง รักษาอักขระ tree: `Prepare samples (hub: load_dataset streaming; local: pyarrow parquet batch)` — แล้ว grep `ConstantLengthDataset|datasets API` ทั้งไฟล์ ถ้ามีที่อื่นใน §4 ที่สัญญาว่าใช้ HF `datasets` กับ local parquet ให้แก้ตามจริง (โค้ดจริง = `pyarrow.iter_batches` สำหรับ local, `load_dataset(streaming=True)` สำหรับ hub — ดู `iter_codes`)
2. **Live log/Step (A3-doc):** `plan.md:222` `Live Log Terminal (แสดง Step/Loss/error traceback)` → `Live Log Terminal (ระดับ + ข้อความ ครบทุกบรรทัด; Step/Loss อยู่ใน Plot/History)`
3. **Metric Plot legend (C4-doc):** `plan.md:209` `Metric Plot (loss + lr)` → `Metric Plot (loss + lr — legend แสดงทั้งสองแกน)`
4. **Watchdog reset:** `plan.md:203` บรรทัด watchdog — ต่อท้าย `...ตลอดกาล` → `...ตลอดกาล; zombie ที่เจอ → รีเซ็ตสถานะเป็น \`aborted\` + ปลดล็อกปุ่ม Start (Sec-14)`
5. **mem_get_info/get_device_name (D5-doc):** `plan.md` Phase 1 checklist (หลัง item "เขียนสคริปต์ทดสอบสร้าง Tensor...") เพิ่ม:
   `- [ ] preflight อ่าน VRAM ผ่าน \`torch.xpu.mem_get_info\` + รายงานชื่ออุปกรณ์ด้วย \`torch.xpu.get_device_name\` (hardware.inspect / check_runtime)`
   และ `README:60` `Run Environment Check` → `Run Environment Check` + ต่อท้าย `(VRAM จาก \`mem_get_info\`, ชื่อเครื่องจาก \`get_device_name\`, disk เช็คบน output dir)`
   และ **แก้ stale index:** `plan.md:98-99` `get_device_name(0)` / `mem_get_info(0)` → `get_device_name(current_device())` / `mem_get_info(current_device())` (ตรง D5 ที่แก้แล้ว)
6. **C4-doc disposition row:** เพิ่ม row ใน spec §3 (รูปแบบเดียวกับ C6-doc):

```
| C4-doc | (audit Docs: "doc เชื่อม label→legend") | grep `legend`/`label→legend` ทั้ง plan.md/README/phase4 docs → **ไม่พบข้อความผูก label=legend** ใน doc ใด ๆ; โค้ด C4 แก้แล้ว (`db0f811` legend จริง + `test_build_metric_plot_has_legend_and_no_pyplot_state`) → ไม่มี doc ต้องแก้ — จดปิดข้อ |
```

7. VERIFY: `grep -n "ConstantLengthDataset" plan.md` → ว่าง; `grep -n "ครบทุกบรรทัด" plan.md` → เจอ; `grep -n "get_device_name(0)\|mem_get_info(0)" plan.md` → ว่าง; `grep -n "C4-doc\|C6-doc" docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md` → เจอทั้งคู่; full suite `.venv/bin/python -m pytest tests/ -q`

** checkboxes **
- [ ] plan.md 6 จุด แก้แล้ว (§4 · live log · legend · watchdog · Phase1 checklist · stale device index)
- [ ] README env-check note
- [ ] C4-doc row อยู่ใน §3
- [ ] full suite เขียว

**What NOT to do:** ห้ามแก้ `docs/superpowers/plans/2026-10-01-*` / `specs/2026-10-01-*` (frozen — ใช้ §3 disposition เท่านั้น); ห้ามแก้ historical ไฟล์อื่น; ห้ามแตะ `diagnosis-issue-draft.md`

---

### ▶ Wave 1 commit (หลัง Task 17)

```bash
.venv/bin/python -m pytest tests/ -q          # ต้องเขียวทั้งชุด
git add core/ configs/safe_defaults.py ui/ eval.py scripts/ \
        requirements.txt constraints.txt README.md plan.md \
        docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md tests/
git commit -m "fix(wave1): English errors + single-source constants, pinned deps + peft CVE note, full live log + spec-rank dropdown + preflight default, report docstring, save sandbox gate, checkpoint swap recovery, zombie status reset, eval identity markers + quant-regression gate, doc alignment (Wave1 #1-8,#10-14 + Doc)"
```

- [ ] Full suite เขียวก่อน commit
- [ ] commit `fix(wave1)` สร้างแล้ว

---

## Definition of Done (ทั้ง plan)

- [ ] Tasks 1–17 ครบ (checkbox ทุกอัน)
- [ ] 2 wave commits: `fix(p2)` + `fix(wave1)` — full suite เขียวทั้งคู่ (expect ≈ 330+ tests)
- [ ] `git status` สะอาด (นอกจาก untracked `diagnosis-issue-draft.md` ที่ห้ามแตะ)
- [ ] Ledger `.superpowers/sdd/.../progress.md` มี task-done line ครบทุก task + deviations ทั้งหมดจด
- [ ] Spec §3 มี disposition rows ครบ (C6-doc + C4-doc เพิ่มรอบนี้; M9-doc/M10-doc/A5/B4/D9/D11 มีอยู่แล้ว)

## Deviations from audit mapping (บันทึกลง ledger ทุกข้อ)

| ข้อ | Deviation | เหตุผล |
|---|---|---|
| C7, D8 | ไม่ได้ทำใน plan นี้ | spec §6 มอบหมายให้ commit clean (Plan 3) |
| Standards-3 (check_runtime try-import) | ถูกลบใน Task 5 (P2) ไม่ใช่ Task 8 | D6 ต้องใช้ `DEFAULT_OUTPUT_DIR` ก่อน — ถ้าไม่ลบ try/except จะเกิด literal ซ้ำชั่วคราว |
| Standards-3 (DEFAULT_OUTPUT_DIR) | เพิ่ม constant ใหม่เข้ารายการ dedupe | D6 บังคับให้ check_runtime อ้าง default เดียวกับ eval.py/UI — เดิมซ้ำ 2 ที่อยู่แล้ว |
| Sec-10 | แก้ที่ core (`save_adapter_only`) ไม่ใช่ dashboard | คุ้มครองทุก caller ลึกกว่า — H1 rule เดียวกับ validate_config |
| Standards-2 | แปลเพิ่ม `check_runtime` success print `fwd/bwd grad ... มีค่า != 0` (audit ไล่แค่ `:31,40,48,50,61,63`) | บรรทัดเดียวกับกลุ่มที่ audit ชี้ — ปล่อยไว้ = output ไทยหลุด CLI เหมือนเดิม |
| #11 baseline | ไฟล์ eval ยุคก่อน markers ถูก skip (note ให้รันใหม่) | เจตนาตาม audit — identity ไม่ทราบที่เชื่อถือได้ = เทียบไม่ได้ |
| A4 | ทำครบทุกทางออกของ on_log ไม่ใช่แค่ trip block | audit "return ชัดกว่า" — ครึ่งๆ กลางๆ ทำให้อ่านยากกว่าเดิม |

## คำแนะนำสำหรับ executing-plans (task-start / task-done)

- Task numbering 1–17 ตามหัวข้อ `## Task N`
- BASE สำหรับ wave P2 = `384367c` (HEAD ก่อนเริ่ม plan); wave 1 = commit ของ `fix(p2)`
- Test command ราย task: ดูบรรทัด VERIFY ในแต่ละ task; จุดจบ wave = `.venv/bin/python -m pytest tests/ -q`
- ถ้าเจอสิ่งที่ขยายเกินขอบเขต (ratchet §9): หยุด → รายงาน → ถามก่อนทำ — M9-quant design อนุมัติแล้ว (D-1/D-2) จึงไม่ต้องถามซ้ำ