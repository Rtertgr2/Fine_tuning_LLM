# Phase 5: End-to-End, Evaluation & Docs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ปิด project — heldout-clean training, `eval.py` (FIM Exact Match + Token F1, base vs fine-tuned), ปุ่ม eval ใน UI, guardrail tests, estimator calibration ≤10%, README + freeze

**Architecture:** `core/evaluator.py` เป็น pure logic (สร้างชุด eval จาก heldout, generate, metrics) ใช้ร่วม CLI (`eval.py`) และ UI (spawn worker ผ่าน queue protocol เดิมของ `predict_middle`); heldout filter เพิ่มใน `dataset_builder` แล้วเรียกจาก `run_training`; integration tests แยก marker `integration`

**Tech Stack:** Python 3.11, torch 2.14+xpu, transformers 5.18, peft/trl, gradio 6, pytest 9

**Spec:** `docs/superpowers/specs/2026-10-02-phase5-eval-docs-design.md`

**Worktree:** สร้าง `.worktrees/phase5` จาก master ก่อนเริ่ม Task 1 (superpowers:using-git-worktrees) — ทุก task รันใน worktree; tests ใช้ `../../.venv/bin/python -m pytest tests/ -v`

## Global Constraints

- **Test-once protocol:** เขียน tests + implement ครบใน Task 1–8 **ห้ามรัน pytest** — รันครั้งแรกเดียวที่ Task 9; หลังรันแรก debug + รันซ้ำได้ตามปกติ
- **UI copy = English ทุกข้อความที่โผล่ UI** รวมข้อความจาก `core/` ที่เด้งเข้า UI (ข้อความ error/progress เป็นต้น)
- **ห้ามแตะ guardrails** §5: batch=1, `MAX_SEQ_LENGTH_CAP=2048`, `DISK_MIN_GB=20`, `SAVE_TOTAL_LIMIT=2`, spawn, localhost:7860
- **`configs/safe_defaults.py` = ที่เดียวของค่าคงที่** — ห้าม hardcode ซ้ำ (tests pin ค่าเหล่านี้)
- **torch ไม่อยู่ใน `requirements.txt`** (ติดตั้งผ่าน XPU index)
- **Ledger:** `.superpowers/sdd/2026-10-02-phase5-eval-docs/progress.md` — บันทึกผล manual gates (loss, peak VRAM, eval numbers) ทุก task
- Docs/README = ภาษาไทย; โค้ด/ข้อความ UI = อังกฤษ

## Review Focus

1. **Leakage: eval cases ต้องมาจาก heldout เท่านั้น และชุดเทรนต้องไม่มี heldout** — ถ้าหลุดฝั่งใดฝั่งหนึ่ง ตัวเลข eval จะ bias สูง → test: `test_filter_train_codes_removes_all_heldout` (Task 1) + `test_build_eval_cases_only_heldout` (Task 3)
2. **finetuned mode ไม่มี checkpoint** → ข้อความ error สื่อสารไม่ได้ (user งง ว่าต้องทำอะไรก่อน) → test: `test_run_eval_missing_checkpoint_message` (Task 3) — `FileNotFoundError` ข้อความ `No checkpoint found in <dir> — train first`
3. **โหลดโมเดล 2 ตัวพร้อมกัน / eval ชน training** (VRAM Contention §5) → test: `test_on_tick_locks_eval_button` (Task 6) + `run_eval` raise ถ้า `training_active` (Task 5, `test_run_eval_rejects_while_training`)
4. **Prompt ไม่มี FIM special tokens** → generate ได้ผลขยะโดยไม่รู้ตัว → test: `test_evaluate_cases_prompt_equals_build_fim_prompt` (Task 3) — prompt ที่ส่งเข้า model ต้อง `== build_fim_prompt(prefix, suffix, fim_tokens=...)`
5. **`--compare` ทิศทาง metric/ tie** — base ดีกว่าต้อง FAIL, เท่ากันต้อง PASS → test: `test_compare_fail_when_base_better` + `test_compare_pass_on_tie` (Task 4)
6. **Orphan ตอน eval timeout/kill** → test: `test_run_eval_cleans_up_on_error` (Task 5) — ทุก path ผ่าน finally เก็บ child (pattern `run_predict`)

---

### Task 1: Heldout filter ใน train path

**Files:**
- Modify: `core/dataset_builder.py` (เพิ่มฟังก์ชันต่อจาก `is_heldout`, ~บรรทัด 99)
- Modify: `core/trainer_worker.py:268` (`run_training` — แทนที่ `codes = iter_codes(...)`)
- Test: `tests/test_dataset_builder.py`

**Interfaces:**
- Consumes: `is_heldout(code: str, heldout_ratio: float = HELDOUT_RATIO) -> bool` (มีอยู่แล้ว)
- Produces: `filter_train_codes(codes: Iterable[str]) -> list[str]` — คืนเฉพาะ codes ที่ `not is_heldout(code)` (Task: evaluator ไม่ได้ใช้ แต่ run_training เรียก; test pin การมีอยู่)

- [ ] **Step 1: เขียน failing test**

```python
def test_filter_train_codes_removes_all_heldout():
    codes = [f"line_a_{i}\nline_b_{i}\nline_c_{i}" for i in range(200)]
    held = {c for c in codes if db.is_heldout(c)}
    assert held, "fixture ต้องมีทั้งสองฝั่ง (md5 split ~10%)"
    result = db.filter_train_codes(codes)
    assert all(not db.is_heldout(c) for c in result)
    assert set(result) == set(codes) - held  # ไม่หาย/ไม่เพิ่ม
    assert db.filter_train_codes(codes) == result  # deterministic
```

- [ ] **Step 2: Implement `filter_train_codes(codes: Iterable[str]) -> list[str]` ใน `core/dataset_builder.py`**

Body: list comprehension กรอง `not is_heldout(code)` — วางต่อจาก `is_heldout`

- [ ] **Step 3: เปลี่ยน `run_training` ใน `core/trainer_worker.py`**

แทน `codes = iter_codes(config["dataset_id"], config["dataset_column"], limit=config["code_limit"])` → ครอบด้วย `filter_train_codes(...)` (import จาก `core.dataset_builder` — ตรวจสอบ import ที่มีอยู่แล้วบรรทัดบน)

- [ ] **Step 4: Commit**

```bash
git add core/dataset_builder.py core/trainer_worker.py tests/test_dataset_builder.py
git commit -m "feat: filter heldout codes out of training set"
```

---

### Task 2: Evaluator — metrics บริสุทธิ์ (Exact Match + Token F1)

**Files:**
- Create: `core/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: (none — pure)
- Produces:
  - `EvalCase` (dataclass frozen): `prefix: str`, `suffix: str`, `middle: str`
  - `exact_match(pred: str, gt: str) -> bool`
  - `token_f1(pred: str, gt: str, tokenizer) -> float` — `tokenizer.encode(text, add_special_tokens=False)` → multiset F1; ทั้งคู่ว่าง = 1.0; ฝั่งใดฝั่งหนึ่งว่าง = 0.0

- [ ] **Step 1: เขียน failing tests**

```python
def test_exact_match_normalizes():
    assert ev.exact_match("x = 1\n", "x = 1")
    assert ev.exact_match("a\r\nb", "a\nb")
    assert ev.exact_match("  x = 1  ", "x = 1")
    assert not ev.exact_match("x = 2", "x = 1")

def test_token_f1_perfect_and_empty():
    tok = FakeTok()  # encode = text.split(), add_special_tokens ignored
    assert ev.token_f1("a b c", "a b c", tok) == 1.0
    assert ev.token_f1("", "", tok) == 1.0
    assert ev.token_f1("", "a b", tok) == 0.0

def test_token_f1_partial_multiset():
    tok = FakeTok()
    # pred {a,a,b} vs gt {a,b,c}: P=2/3 R=2/3 F1=2/3
    assert abs(ev.token_f1("a a b", "a b c", tok) - (2 / 3)) < 1e-9

def test_evalcase_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.EvalCase("a", "b", "c").prefix = "x"
```

- [ ] **Step 2: Implement ใน `core/evaluator.py`**

`exact_match`: `pred.replace("\r\n", "\n").strip() == gt.replace("\r\n", "\n").strip()`
`token_f1`: `collections.Counter` intersect → `p, r, f1`; guard empty cases ก่อน

- [ ] **Step 3: Commit**

```bash
git add core/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluator metrics (FIM exact match + token F1)"
```

---

### Task 3: Evaluator — build cases, generate, run_eval

**Files:**
- Modify: `core/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `iter_codes`, `is_heldout`, `split_fim`, `truncate_to_tokens` (จาก `core/dataset_builder`); `build_fim_prompt`, `latest_checkpoint`, `validate_config`, `ensure_fim_tokens` (จาก `core/trainer_worker` — import ระดับ top ได้ ห้าม `core.trainer_worker` import `core.evaluator` ระดับ top เด็ดขาด)
- Produces:
  - `EVAL_DIR = Path("data_cache/eval")`
  - `EVAL_MAX_NEW_TOKENS: int = 256`
  - `build_eval_cases(*, dataset_id: str, dataset_column: str, limit: int, tokenizer, n_cases: int = 100, max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT, seed: int = SEED) -> list[EvalCase]`
  - `evaluate_cases(model, tokenizer, cases: list[EvalCase], *, fim_tokens: dict, device: str, progress: Callable[[int, int], None] | None = None) -> dict` → `{"per_case": [{"i", "exact", "f1", "pred", "gt"}], "exact_match_pct": float, "token_f1_mean": float}`
  - `run_eval(config: dict, *, mode: Literal["base", "finetuned"], n_cases: int = 100, eval_dir: Path | str = EVAL_DIR, progress=None) -> dict` — เขียน `eval_dir/<mode>.json`, คืน result dict ที่มี keys: `"mode", "n", "exact_match_pct", "token_f1_mean", "per_case", "samples"` (+ `"warning"` ถ้า `n < n_cases`)

**Algorithms ที่ test ไม่กำหนด (pin ไว้):**
- `build_eval_cases`: stream `iter_codes(dataset_id, dataset_column, limit=limit)` → เฉพาะ `is_heldout(code)` → เงื่อนไขเดียวกับ `build_samples` (`len(splitlines()) >= MIN_SAMPLE_LINES` และ `len(_cut_positions(code)) >= 2` — import `_cut_positions` จาก dataset_builder) → `split_fim(code, random.Random(seed))` (rng ตัวเดียว seed ครั้งเดียวก่อน loop) → **ข้ามถ้า `middle` มี token เกิน `EVAL_MAX_NEW_TOKENS`** → ตัด suffix `truncate_to_tokens(..., max_seq_length // 2)` และ prefix `truncate_to_tokens(..., max_seq_length - max_seq_length // 2)` → เก็บจนครบ `n_cases` (หมด stream ก็คืนเท่าที่มี ไม่ raise)
- `run_eval` **โหลดตามลำดับนี้** (fail fast ก่อนแตะ network): `validate_config(config)` → อ่าน `configs/fim_registry.json` → ถ้า `mode == "finetuned"`: ถ้า `latest_checkpoint(config["output_dir"])` raise → `raise FileNotFoundError(f"No checkpoint found in {config['output_dir']} — train first")` → `AutoTokenizer.from_pretrained` + `ensure_fim_tokens` → `build_eval_cases` (ใช้ `config["code_limit"]` เป็น limit, `config["max_seq_length"]`) → โหลด model (`dtype=torch.bfloat16, attn_implementation="sdpa"`; finetuned = `PeftModel.from_pretrained(model, str(ckpt))`) → `evaluate_cases` → `json.dump` → คืน
- `evaluate_cases`: ต่อ case: `prompt = build_fim_prompt(case.prefix, case.suffix, fim_tokens=fim_tokens)` → `tokenizer(prompt, return_tensors="pt")` → `.to(device)` → `model.generate(**inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False)` → decode continuation (`skip_special_tokens=True`) → metrics → `progress(i + 1, len(cases))`; `"samples"` = `per_case[:5]`

- [ ] **Step 1: เขียน failing tests** (FakeModel/FakeTok pattern — ดู `tests/test_trainer_worker.py` `test_build_fim_prompt_exact` สำหรับ key names ของ fim_tokens)

```python
def test_build_eval_cases_only_heldout(monkeypatch):
    # codes 2 กลุ่ม: กลุ่ม A ถูกยิงจน is_heldout=True, กลุ่ม B จน False (นับ md5 จนกว่าจะได้)
    held = _codes_where(is_heldout=True, n=40)   # helper: ลอง nonce จน db.is_heldout ตรง
    train = _codes_where(is_heldout=False, n=40)
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: held + train)
    cases = ev.build_eval_cases(dataset_id="x", dataset_column="content",
                                limit=80, tokenizer=FakeTok(), n_cases=10)
    assert 0 < len(cases) <= 10
    # ทุก middle ต้องมาจาก held เท่านั้น (codes แต่ละตัว unique — middle ไม่ซ้ำข้ามกลุ่ม)
    held_middles = {db.split_fim(c, random.Random(42))[1] for c in held}
    for case in cases:
        assert case.middle in held_middles

def test_build_eval_cases_skips_long_middle(monkeypatch):
    long_mid = "word " * 300  # FakeTok = word tokens → 300 > 256
    code = "def a():\n    pass\n" + long_mid + "def b():\n    pass\n    z = 1\n"
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: [code])
    cases = ev.build_eval_cases(dataset_id="x", dataset_column="content",
                                limit=1, tokenizer=FakeTok(), n_cases=10)
    assert cases == []  # middle เกิน 256 ถูกข้าม ไม่ raise

def test_build_eval_cases_deterministic(monkeypatch):
    # stream ชุดเดิม → เรียก 2 ครั้ง ได้ list เท่ากันทุกตัวอักษร

def test_build_eval_cases_fewer_than_requested(monkeypatch):
    # stream มี heldout ผ่าน criteria แค่ 3 → คืน 3 ไม่ raise (spec §6)

def test_evaluate_cases_prompt_equals_build_fim_prompt():
    # FakeModel/FakeTok capture prompt → assert == build_fim_prompt(prefix, suffix, fim_tokens=...)
    # ยืนยัน fim_tokens ไหลถึง prompt (Review Focus #4)

def test_evaluate_cases_aggregates_and_progress():
    # FakeModel คืน continuation คงที่ → exact_match_pct/token_f1_mean ถูกต้อง
    # progress ถูกเรียก (1..n) ครบ

def test_run_eval_missing_checkpoint_message(tmp_path):
    cfg = _valid_config(output_dir=str(tmp_path))  # tmp ว่าง, mode="finetuned"
    with pytest.raises(FileNotFoundError, match="train first"):
        ev.run_eval(cfg, mode="finetuned")
```

หมายเหตุ: `_codes_where` = helper ใน test ไฟล์: loop nonce ต่อท้าย code จน `db.is_heldout(candidate)` ได้ค่าที่ต้องการ (pattern เดียวกับ test heldout ที่มีอยู่ใน `test_dataset_builder.py`)

- [ ] **Step 2: Implement ตาม Algorithms block ข้างบน**

- [ ] **Step 3: Commit**

```bash
git add core/evaluator.py tests/test_evaluator.py
git commit -m "feat: eval case building, generation, run_eval"
```

---

### Task 4: `eval.py` CLI

**Files:**
- Create: `eval.py` (repo root, ระดับเดียวกับ `app.py`)
- Test: `tests/test_eval_cli.py`

**Interfaces:**
- Consumes: `ev.run_eval`, `ev.EVAL_DIR` (Task 3); constants จาก `configs/safe_defaults` (`DEFAULT_MODEL_ID`, `DEFAULT_DATASET_ID`, `DEFAULT_DATASET_COLUMN`, `MAX_SEQ_LENGTH_DEFAULT`, `MAX_STEPS`, `TRAIN_CODE_LIMIT`, `LORA_RANK_DEFAULT`); `REQUIRED_CONFIG_KEYS` จาก `core.trainer_worker`
- Produces: `main(argv: list[str] | None = None) -> int` — parse args แล้ว dispatch; `if __name__ == "__main__": raise SystemExit(main())`
  - args: `--mode {base,finetuned}` (บังคับ เว้นมี `--compare`), `--compare`, `--n-cases N` (default 100), `--eval-dir PATH` (default `EVAL_DIR`), `--output-dir PATH` (default `data_cache/finetune_run`), `--fim-key` (default `"qwen"`)
  - `--mode`: build config dict ครบทุก key ใน `REQUIRED_CONFIG_KEYS` จาก safe_defaults → `run_eval` → พิมพ์สรุป `mode=... n=... exact_match=..% token_f1=..` → คืน 0; exception → พิมพ์ stderr → คืน 1
  - `--compare`: อ่าน `<eval_dir>/base.json` + `finetuned.json` (ขาดไฟล์ → stderr `Missing <path> — run --mode <mode> first` → คืน 2) → `compare_results(base, finetuned)` → พิมพ์ตาราง + ตัวอย่าง qualitative → `PASS`/`FAIL` → คืน 0/1
- Produces (ใน `core/evaluator.py`, เพิ่มที่นี่ด้วย): `compare_results(base: dict, finetuned: dict) -> tuple[list[list[str]], list[dict]]` — คืน `(rows, qualitative)`; rows = `[["Exact Match %", b, f, delta], ["Token F1", ...]]` (str formatting ทศนิยม 1 / 2); qualitative = up to 5 entries `{"i", "gt", "base_pred", "finetuned_pred"}` ที่ `base.per_case[i].exact == False` และ `finetuned.per_case[i].exact == True` (match by `"i"`)

- [ ] **Step 1: เขียน failing tests**

```python
def test_compare_pass_on_tie(capsys):
    r = {"exact_match_pct": 5.0, "token_f1_mean": 0.2, "per_case": [], "mode": "base", "n": 3}
    rows, qual = ev.compare_results(r, {**r, "mode": "finetuned"})
    rc = cli_main(["--compare", "--eval-dir", str(tmp_write_two(r, r))])  # helper เขียน 2 ไฟล์
    assert rc == 0 and "PASS" in capsys.readouterr().out

def test_compare_fail_when_base_better(capsys):
    base = {"exact_match_pct": 10.0, "token_f1_mean": 0.5, "per_case": [], "mode": "base", "n": 3}
    fine = {"exact_match_pct": 4.0, "token_f1_mean": 0.3, "per_case": [], "mode": "finetuned", "n": 3}
    rc = cli_main(["--compare", "--eval-dir", str(tmp_write_two(base, fine))])
    assert rc == 1 and "FAIL" in capsys.readouterr().out

def test_compare_missing_file(tmp_path, capsys):
    rc = cli_main(["--compare", "--eval-dir", str(tmp_path)])
    assert rc == 2 and "run --mode" in capsys.readouterr().err

def test_compare_qualitative_pairs():
    base = {"mode": "base", "n": 2, "exact_match_pct": 0.0, "token_f1_mean": 0.0,
            "per_case": [{"i": 0, "exact": False, "f1": 0.0, "pred": "wrong", "gt": "right"},
                         {"i": 1, "exact": False, "f1": 0.0, "pred": "w2", "gt": "g2"}]}
    fine = {**base, "mode": "finetuned",
            "per_case": [{**base["per_case"][0], "exact": True, "pred": "right"},
                         base["per_case"][1]]}
    rows, qual = ev.compare_results(base, fine)
    assert len(qual) == 1 and qual[0]["i"] == 0 and qual[0]["base_pred"] == "wrong"

def test_mode_writes_json(monkeypatch, tmp_path, capsys):
    def fake_run_eval(cfg, *, mode, eval_dir, **kw):
        result = {"mode": mode, "n": 2, "exact_match_pct": 50.0,
                  "token_f1_mean": 0.5, "per_case": [], "samples": []}
        Path(eval_dir, f"{mode}.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(ev, "run_eval", fake_run_eval)
    rc = cli_main(["--mode", "base", "--eval-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "base.json").exists()
    assert "exact_match" in capsys.readouterr().out
```

- [ ] **Step 2: Implement `compare_results` ใน `core/evaluator.py` + `eval.py` CLI**

Argparse: `--compare` กับ `--mode` กัน `required` (ใช้ group mutually exclusive required=True) — `main` คืน int เสมอ (ไม่ raise จาก argparse เพราะ `SystemExit` → wrap ด้วย try/except SystemExit คืน `e.code` เพื่อให้ test เรียก `main` ได้)

- [ ] **Step 3: Commit**

```bash
git add eval.py core/evaluator.py tests/test_eval_cli.py
git commit -m "feat: eval.py CLI with base vs finetuned compare"
```

---

### Task 5: `run_eval_worker` + `TrainingController.run_eval`

**Files:**
- Modify: `core/trainer_worker.py` (เพิ่ม spawn target ท้ายไฟล์ — section เดียวกับ `predict_middle`)
- Modify: `ui/controller.py` (method บน `TrainingController` ต่อจาก `run_predict` module function)
- Test: `tests/test_trainer_worker.py`, `tests/test_ui_controller.py`

**Interfaces:**
- Consumes: `ev.run_eval`, `ev.EVAL_DIR` (Task 3 — **import ภายในฟังก์ชันเท่านั้น** เพื่อหลบ circular import)
- Produces:
  - `run_eval_worker(config: dict, mode: str, queue_, eval_dir: str | Path = EVAL_DIR) -> None` — ทำ `ev.run_eval(config, mode=mode, eval_dir=eval_dir, progress=lambda d, t: queue_.put(log_msg("INFO", f"Evaluating {mode}: {d}/{t}")))` แล้ว `queue_.put(log_msg("INFO", "EVAL_DONE"))`; ผิด → `queue_.put(error_msg(str(exc), traceback.format_exc()))` + raise (pattern `predict_middle`)
  - `TrainingController.run_eval(config: dict, mode: str, *, eval_dir=..., timeout: float = 900.0, process_factory=mp.Process, queue_factory=mp.Queue) -> dict` — re-entrancy: `if self.training_active: raise RuntimeError("Training in progress — evaluation is locked (VRAM contention)")`; drain loop แบบ `run_predict` (log → `self._append_log("INFO", text)` **เว้น `EVAL_DONE`**; error → `RuntimeError`; child ตายเงียบ → `RuntimeError("Evaluation process died without a result")`; timeout → `RuntimeError`; ไม่เจอ `EVAL_DONE` → `RuntimeError`); สำเร็จ → อ่าน `<eval_dir>/<mode>.json` (`json.load`) คืน dict; **finally: `process.join(1.0)` + `abort_process` ถ้ายัง alive** (pattern `run_predict` เป๊ะ)

- [ ] **Step 1: เขียน failing tests**

```python
# tests/test_trainer_worker.py
def test_run_eval_worker_done_and_error():
    # done: monkeypatch tw.ev (module attr) stub → fake queue list → call → มี log "EVAL_DONE"
    # error: stub raise RuntimeError("boom") → มี error_msg message="boom" + raise

# tests/test_ui_controller.py — reuse fake process/queue helper (ดูบรรทัด ~301 ของไฟล์นี้)
def test_run_eval_happy_path(tmp_path):
    # เขียน finetuned.json ลง eval_dir ก่อน; fq  preload [log "x", log "EVAL_DONE"]
    # → คืน dict จากไฟล์; log "x" เข้า controller._logs

def test_run_eval_rejects_while_training():
    ctl._status = "training"  # (หรือตาม pattern ที่ test อื่นตั้ง)
    with pytest.raises(RuntimeError, match="VRAM contention"):
        ctl.run_eval({"any": 1}, "base")

def test_run_eval_cleans_up_on_error():
    # error_msg ใน fq → RuntimeError; fake process ต้อง join/abort ถูกเรียก (holder flag)
```

- [ ] **Step 2: Implement** — `run_eval_worker` ใน `core/trainer_worker.py`; `run_eval` method ใน `ui/controller.py` (copy drain loop จาก `run_predict` แล้วปรับ: log ทุกตัว append เข้า `self._append_log` ยกเว้น `EVAL_DONE`, คืนจากการอ่านไฟล์)

- [ ] **Step 3: Commit**

```bash
git add core/trainer_worker.py ui/controller.py tests/test_trainer_worker.py tests/test_ui_controller.py
git commit -m "feat: eval worker + controller run_eval with cleanup"
```

---

### Task 6: ปุ่ม Run Evaluation ใน UI

**Files:**
- Modify: `ui/dashboard.py` (Tab 3 ~บรรทัด 276; handlers `_make_handlers` ~88; wiring ~295; `on_tick` ~129)
- Test: `tests/test_ui_dashboard.py`

**Interfaces:**
- Consumes: `controller.run_eval` (Task 5); `ev.compare_results` (Task 4)
- Produces: handler `on_eval(*cfg_values) -> tuple[list[list[str]] | None, str]` ใน handlers dict — `_collect_config` → รัน `controller.run_eval(config, "base")` แล้ว `controller.run_eval(config, "finetuned")` (sequential, คนละ process, ไม่ stacking) → `compare_results` → คืน `(rows, "")`; exception → `(None, f"**Evaluation failed:** {exc}")` (English เสมอ)
  - Widgets: `eval_btn = gr.Button("Run Evaluation", variant="secondary")`, `eval_table = gr.Dataframe(headers=["Metric", "Base", "Fine-tuned", "Δ"], interactive=False)`, `eval_error_md = gr.Markdown("")` ใน Tab 3
  - `on_tick` เพิ่ม output ตัวที่ 9 = `gr.Button(interactive=not ta)` สำหรับ `eval_btn` (lock ระหว่างเทรน, §5 VRAM Contention)
  - Wiring: `eval_btn.click(h["on_eval"], inputs=cfg_inputs, outputs=[eval_table, eval_error_md], concurrency_id="model_load")` (กัน predict/merge/eval ชนกัน)

- [ ] **Step 1: เขียน failing tests**

```python
def test_build_dashboard_structure():  # เพิ่ม assertions ใน test เดิม
    # มี Dataframe ชื่อหัว ["Metric", "Base", "Fine-tuned", "Δ"] + Button "Run Evaluation"

def test_on_tick_locks_eval_button():  # ต่อยอด test_build_dashboard_locks_exist เดิม
    # on_tick คืน tuple ยาว 9; ตัวท้ายสุดเป็น gr.Button — interactive=True เมื่อไม่เทรน, False เมื่อเทรน

def test_on_eval_renders_table(monkeypatch):
    # monkeypatch controller.run_eval → คืน result dict 2 ครั้ง (base/fine) → rows จาก compare_results
    # on_eval คืน (rows, "")
def test_on_eval_error_is_english(monkeypatch):
    # run_eval raise RuntimeError("Training in progress ...") → (None, str) ขึ้นต้น "**Evaluation failed:**"
```

- [ ] **Step 2: Implement** — widgets ใน Tab 3, handler ใน `_make_handlers`, `on_tick` return + outputs list ขยายเป็น 9, wiring `concurrency_id="model_load"`

- [ ] **Step 3: Commit**

```bash
git add ui/dashboard.py tests/test_ui_dashboard.py
git commit -m "feat: Run Evaluation button with lock + result table"
```

---

### Task 7: Peak VRAM log ตอนจบการเทรน (สำหรับ calibration)

**Files:**
- Modify: `core/trainer_worker.py` (`run_training` ~บรรทัด 311)
- Test: `tests/test_trainer_worker.py`

**Interfaces:**
- Consumes: `torch`, `log_msg`
- Produces: `peak_xpu_memory_gb(kind: str = "reserved") -> float` — คืน `torch.xpu.max_memory_<kind>() / 1024**3` ถ้า `torch.xpu.is_available()` มิฉะนั้น `0.0` (`kind` รับ `"reserved"`/`"allocated"`; invalid kind → `ValueError`); ใน `run_training` หลัง `trainer.train()` และ `not callback.aborted`:
  `queue_.put(log_msg("INFO", f"xpu_peak_reserved_gb={peak_xpu_memory_gb('reserved'):.2f} xpu_peak_allocated_gb={peak_xpu_memory_gb('allocated'):.2f}"))` แล้วค่อย `status_msg("finished")`

- [ ] **Step 1: เขียน failing tests**

```python
def test_peak_xpu_memory_zero_when_unavailable(monkeypatch):
    monkeypatch.setattr(tw.torch.xpu, "is_available", lambda: False)
    assert tw.peak_xpu_memory_gb() == 0.0

def test_peak_xpu_memory_invalid_kind():
    with pytest.raises(ValueError):
        tw.peak_xpu_memory_gb("nope")

def test_peak_xpu_memory_uses_torch(monkeypatch):
    monkeypatch.setattr(tw.torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(tw.torch.xpu, "max_memory_reserved", lambda: 2 * 1024**3, raising=False)
    assert tw.peak_xpu_memory_gb("reserved") == 2.0
```

- [ ] **Step 2: Implement** — function + log line ใน `run_training` (ตาม Interfaces)

- [ ] **Step 3: Commit**

```bash
git add core/trainer_worker.py tests/test_trainer_worker.py
git commit -m "feat: log peak XPU memory after training for calibration"
```

---

### Task 8: pytest marker + Integration tests (Abort + Disk gate)

**Files:**
- Create: `pytest.ini`, `tests/test_phase5_integration.py`
- Test: ตัวมันเอง

**Interfaces:**
- Consumes: `TrainingController` (ui/controller), `core.hardware.inspect`, `psutil`, `_START_VERDICTS` (ui/dashboard), `ev.build_eval_cases`
- Produces: marker `integration` + `addopts = -m "not integration"` ใน `pytest.ini` (suite เร็วเหมือนเดิม; รันแยกด้วย `-m integration` ซึ่ง override จาก CLI)

**Test ที่ต้องเขียน (ทั้งไฟล์ `@pytest.mark.integration`):**

1. `test_abort_mid_training_no_orphan_no_vram_leak`:
   - `inspect("data_cache")` → `vram_before = free_vram_gb`
   - config smoke: defaults ทั้งหมด + `max_steps=60`, `output_dir="data_cache/itest_abort"` (ลบก่อน/หลัง)
   - `ctl.start(cfg)` → poll `ctl.tick()` จนมี `snap.metrics` (timeout 180s) → `ctl.abort()` → `ctl.exit()`
   - assert: process ไม่ alive; `psutil.Process().children(recursive=True)` ไม่มี PID ของ child (เปรียบเทียบกับ PID ที่บันทึกไว้); `inspect(...)` `free_vram_gb >= vram_before - 0.5`
2. `test_disk_gate_blocks_preflight` (monkeypatch `psutil.disk_usage` → free 19GB):
   - `inspect("data_cache")["status"] == "insufficient_disk"`
   - `ctl.preflight(cfg, None).verdict == "insufficient_disk"` และ `"insufficient_disk" not in _START_VERDICTS` (=start ถูกบล็อก — Review Focus ของ spec §5)
3. `test_datacache_growth_bounded` (informational):
   - `du` `data_cache` ก่อน/หลัง `ev.build_eval_cases(..., limit=512, n_cases=10)` (stream จริง) → print delta → assert growth < 5GB

- [ ] **Step 1: เขียน `pytest.ini`**

```ini
[pytest]
markers =
    integration: full-stack tests needing XPU/network/disk (run with -m integration)
addopts = -m "not integration"
```

- [ ] **Step 2: เขียน `tests/test_phase5_integration.py`** ตาม 3 tests ข้างบน (poll pattern ดูจาก integration ของ Phase 3/4 ถ้ามีใน ledger หรือเขียนใหม่จาก `ctl.tick()`)

- [ ] **Step 3: Commit**

```bash
git add pytest.ini tests/test_phase5_integration.py
git commit -m "test: integration suite (abort, disk gate, cache growth)"
```

---

### Task 9: รัน Test Suite ครั้งแรก (test-once gate)

- [ ] **Step 1: รัน unit suite ครั้งแรก**

Run: `../../.venv/bin/python -m pytest tests/ -v`
Expected: ทุก test PASS (91 เดิม + ใหม่); `integration` tests ถูก deselect (นับ `passed` แล้วบันทึกตัวเลขลง ledger)
- [ ] **Step 2: Debug + รันซ้ำ** จนเขียว (cycle นี้เท่านั้นที่ซ้ำได้)
- [ ] **Step 3: รัน integration suite**

Run: `../../.venv/bin/python -m pytest tests/ -m integration -v`
Expected: abort + disk + cache growth PASS (~2–3 นาที, ใช้ XPU จริง)
- [ ] **Step 4: Commit ถ้ามีการแก้ + บันทึกตัวเลขทั้ง 2 ชุดลง ledger**

---

### Task 10: เทรนจริง 500 steps ใหม่ (heldout-clean) — Manual gate §8.1

**เงื่อนไข:** Task 1 + 9 ผ่านแล้ว

- [ ] **Step 1: ล้าง checkpoint เก่า**

Run: `rm -rf data_cache/finetune_run` (checkpoint ปน heldout — ห้ามใช้ต่อ)
- [ ] **Step 2: เทรน 500 steps** — heredoc script ขับ `TrainingController` (pattern เดียวกับที่เคยใช้ — spawn + tick loop + drain log ลง `/tmp/opencode/train500.log`), config = safe defaults ทั้งหมด (`max_steps=500, code_limit=8192, seq=1024, output_dir=data_cache/finetune_run`)
  - Expected: `status=finished` (~17 นาที), log มีบรรทัด `xpu_peak_reserved_gb=... xpu_peak_allocated_gb=...`
- [ ] **Step 3: บันทึกผลลง ledger** — loss เริ่ม/จบจาก `data_cache/finetune_run/trainer_state.json` (`log_history[0]["loss"]` vs `train_loss`) + peak VRAM ทั้ง 2 ค่า + ยืนยัน §8.1 (loss จุดสุดท้าย < จุดเริ่ม อย่างมีนัยสำคัญ)

---

### Task 11: Calibrate estimator ≤10% — Manual gate

**เงื่อนไข:** Task 10 (มี peak จริง)

- [ ] **Step 1: คำนวณ error** — `estimate(inspect("data_cache"), model_id=DEFAULT_MODEL_ID, seq_length=1024)` → `total_required_gb` เทียบ `xpu_peak_reserved_gb` จาก Task 10: `err% = abs(total - peak) / peak * 100`
- [ ] **Step 2: ถ้า err > 10%** → ปรับ `ESTIMATOR_OVERHEAD_GB` ใน `configs/safe_defaults.py`: `new = old + (peak - total)` (ปัด 1 ตำแหน่ง; ถ้า new ≤ 0 → หยุดแล้วรายงาน user เลือก knob อื่น) → อัปเดต test ที่ pin ค่าใน `tests/test_safe_defaults.py` / `tests/test_estimator.py` → รัน `../../.venv/bin/python -m pytest tests/test_safe_defaults.py tests/test_estimator.py -v` (รันซ้ำหลัง gate แรก = อนุญาต)
- [ ] **Step 3: ยืนยัน err ≤ 10% ใหม่** (คำนวณซ้ำ) + บันทึก equation/ค่าเดิม/ค่าใหม่ลง ledger

---

### Task 12: รัน eval จริง + เกณฑ์ §8.2/§8.3 — Manual gate

**เงื่อนไข:** Task 9 + 10

- [ ] **Step 1: วัด data_cache growth ตอน eval (informational)** — `du -sb data_cache` ก่อน/หลัง → ledger
- [ ] **Step 2: รันทั้ง 2 modes + compare**

Run: `../../.venv/bin/python eval.py --mode base` → `--mode finetuned` → `--compare`
Expected: mode ละ ~100 ตัวอย่าง (~5–8 นาที), `--compare` exit 0 `PASS`
- [ ] **Step 3: ตัดสิน §8.2** — fine-tuned ต้องดีกว่า base ทั้ง Exact Match % และ Token F1
  - **ถ้า FAIL: หยุด แล้วรายงาน user** (plan §8 สั่งกลับไปตรวจ pipeline — อย่า iterate เอง)
- [ ] **Step 4: §8.3 qualitative** — ดึง 3–5 ตัวอย่างจาก output `--compare` (base ผิด → fine ถูก) → ledger
- [ ] **Step 5: ทดสอบปุ่มใน UI** (manual, 5 นาที) — `.venv/bin/python app.py` → กด Run Evaluation → ตารางขึ้น, ระหว่างเทรนปุ่มล็อก → บันทึก ledger

---

### Task 13: README + freeze requirements

**Files:**
- Modify: `README.md` (เขียนใหม่ — คงโครงมีอยู่)
- Modify: `requirements.txt`

- [ ] **Step 1: README ภาษาไทย** — sections: ภาพรวม + สถานะ (อัปเดต table: Phase 4 ✅, Phase 5 ✅ พร้อมสรุปผล eval จริงจาก Task 12), ความต้องการระบบ (driver Intel XPU + EndeavourOS), ติดตั้ง (venv, torch XPU wheel, `pip install -r requirements.txt -c constraints.txt`), เริ่มใช้งาน (`.venv/bin/python app.py` → localhost:7860), eval (`eval.py` 3 คำสั่ง), export (Save Adapter/Merge), troubleshooting สั้นๆ
- [ ] **Step 2: Freeze requirements** — เทียบ `pip freeze` กับ `constraints.txt` → pin `requirements.txt` ทุกบรรทัดเป็น `pkg==version` ที่ตรงกับ constraints (คง comment + บรรทัด torch แยก); verify: `../../.venv/bin/python -m pip install -r requirements.txt -c constraints.txt --dry-run` (ไม่มี conflict) + `pip check`
- [ ] **Step 3: Commit + push-ready**

```bash
git add README.md requirements.txt
git commit -m "docs: README walkthrough + frozen requirements"
```

---

## Self-Review (เขียน plan แล้ว)

1. **Spec coverage:** §3.1 ไฟล์ 11 ตัว → Task 1–8, 13 ✓; §3.2 evaluator → Task 2–3 ✓; §3.3 CLI → Task 4 ✓; §3.4 UI → Task 5–6 ✓; §5 unit/integration/manual → Task 7–12 ✓; §6 error messages → Task 3 (`train first`), Task 4 (missing file), Task 5–6 (English) ✓; heldout fix + retrain → Task 1 + 10 ✓
2. **Step scan:** ทุก step มี test name/assertion หรือ signature หรือ command — ไม่มี "TBD/handle edge cases" ✓
3. **Type consistency:** `run_eval(config, mode=..., eval_dir=...)` ตรงกันทุก task (worker/controller/CLI); `compare_results → (rows, qualitative)` ตรง Task 4/6 ✓; `EVAL_DIR` ใช้ชื่อเดียวกัน ✓
4. **Review Focus 6 ข้อ** มี test กำกับทุกข้อ (Task 1/3/5/6/4 ตามลำดับ) ✓
5. **สัดส่วน:** plan ยาวกว่า spec ~2 เท่า ส่วนใหญ่เป็น test code + command — ไม่มี body ที่เขียนแทนคน ✓
