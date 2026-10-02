# Phase 5 Design: End-to-End, Evaluation & Docs

- **วันที่:** 2026-10-02
- **สถานะ:** Approved (user เห็นด้วยกับ scope + 4 sections ในแชท)
- **Plan แม่บท:** `plan.md` §6 Phase 5, §8 Success Criteria, §5 Guardrails

## 1. Context & Intent

Phase 1–4 สร้าง pipeline ครบทั้ง hardware gate → estimator → dataset builder (FIM) →
training subprocess + IPC → Gradio UI แล้ว (91 tests) เหลืองานปิด project: สร้าง
**เกณฑ์วัดผลจริง** (base vs fine-tuned ต้องดีขึ้น), ทดสอบ guardrail ที่เหลือ,
calibrate estimator กับ VRAM จริง และ docs/freeze

**ข้อเท็จจริงที่กำหนด design:**
- `run_training` ไม่ได้กรอง heldout ออกจากชุดเทรน → checkpoint-500 ที่มีอยู่
  (user run + soak) **ปน heldout ~10%** = leakage ถ้าใช้วัดผล
- `is_heldout()` (md5 คงที่, `HELDOUT_RATIO=0.10`) มีใน `dataset_builder.py` แล้วแต่ไม่ถูกเรียกตอนเทรน
- `predict_middle` (Phase 4) มี pattern spawn-subprocess + queue อยู่แล้ว → eval ควร reuse

**User decisions (จาก Q&A):**
1. แก้ code กรอง heldout + **เทรนใหม่ 500 steps**
2. eval.py = **CLI + ปุ่มใน UI**
3. ชุด eval = **100 ตัวอย่าง**
4. Disk test = **จำลอง disk เต็ม + วัด cache** (ไม่ดึง dataset ใหญ่จริง)

## 2. Scope

**ทำ:**
1. กรอง heldout ใน `run_training` + test
2. ล้าง `data_cache/finetune_run` + เทรนใหม่ 500 steps (checkpoint สะอาด)
3. `core/evaluator.py` (logic + metrics) + `eval.py` (CLI) + ปุ่ม UI
4. รัน eval 100 × 2 → **fine ต้องดีกว่า base** (§8 gate)
5. Abort กลางคัน → ไม่มี zombie/VRAM leak (integration test)
6. จำลอง disk <20GB → gate บล็อก + วัดการโต data_cache
7. Calibrate estimator ≤10% + README (ไทย) + freeze requirements

**ไม่ทำ (YAGNI):**
- ไม่แก้ metrics/สูตรเทรนเพิ่มนอกเหนือจากที่ calibrate กำหนด
- ไม่ทำ eval เป็น scheduled/CI job — รันมือ + ปุ่ม UI เท่านั้น
- ไม่เพิ่ม model ใหม่/dataset ใหม่

## 3. Architecture

### 3.1 ไฟล์ใหม่/แก้

| ไฟล์ | สถานะ | เนื้อหา |
|---|---|---|
| `core/evaluator.py` | ใหม่ | build heldout cases, generate, metrics (pure functions) |
| `eval.py` | ใหม่ | CLI: `--mode base\|finetuned` → JSON + ตารางเทียบ |
| `core/trainer_worker.py` | แก้ | กรอง heldout ใน `run_training`; log `max_memory_allocated` ตอนจบ; spawn target `run_eval_worker` |
| `ui/dashboard.py` | แก้ | ปุ่ม Run Evaluation + ตารางผล ใน Playground tab (ล็อกตอนเทรน) |
| `tests/test_evaluator.py` | ใหม่ | metrics + case building |
| `tests/test_eval_cli.py` | ใหม่ | CLI บันทึก/เทียบผล (mock model) |
| `tests/test_trainer_worker.py` | แก้ | เพิ่ม test heldout filter |
| `tests/test_ui_dashboard.py` | แก้ | เพิ่ม test ปุ่ม eval + lock |
| `tests/test_phase5_integration.py` | ใหม่ | `@pytest.mark.integration` — abort + disk e2e |
| `README.md` | แก้ | ไทย: driver → venv → ใช้งาน → eval/export |
| `requirements.txt` | แก้ | freeze ตรง constraints.txt |

### 3.2 `core/evaluator.py`

```python
@dataclass(frozen=True)
class EvalCase:
    prefix: str; suffix: str; middle: str   # ground truth

def build_eval_cases(*, limit, n_cases=100, seed=SEED, dataset_id, dataset_column,
                     tokenizer, max_seq_length) -> list[EvalCase]
    # stream เดียวกับเทรน (iter_codes limit) → is_heldout เท่านั้น
    # → ผ่าน criteria เดียวกับ build_samples (min_lines, _cut_positions ≥ 2)
    # → split_fim(seed) → ตัด prefix/suffix ด้วย truncate_to_tokens
    # → **ข้าม middle > 256 tokens** (เกิน max_new_tokens → exact match เป็นไปไม่ได้)
    # → เอา n_cases แรก (deterministic ซ้ำได้)

def generate_middle(model, tokenizer, case, *, device, max_new_tokens=256) -> str
    # build_fim_prompt (reuse จาก trainer_worker) → greedy generate
    # → decode skip_special_tokens=True

def exact_match(pred: str, gt: str) -> bool
    # normalize: CRLF→LF + strip 两端 เทียบเป๊ะ

def token_f1(pred: str, gt: str, tokenizer) -> float
    # multiset F1 บน token IDs (add_special_tokens=False)
    # both empty → 1.0

def run_eval(config, *, mode: Literal["base","finetuned"]) -> dict
    # โหลด base (dtype=bfloat16, sdpa) → mode finetuned คือ wrap PeftModel จาก latest_checkpoint
    # วน 100 cases → {exact_match, token_f1, samples[5]} + aggregates
    # คืน {"mode", "n", "exact_match_pct", "token_f1_mean", "per_case", "samples"}
```

- **VRAM:** โหลดโมเดลทีละ 1 ตัวตลอด (ไม่ stacking) — `mode` แยกการรัน
- **samples[5]:** เก็บ 5 ตัวอย่าง (case ที่ base ผิด → fine ถูก prioritized) สำหรับ §8.3 qualitative

### 3.3 `eval.py` CLI

```
.venv/bin/python eval.py --mode base        # วัด base → data_cache/eval/base.json
.venv/bin/python eval.py --mode finetuned   # วัด base+LoRA → data_cache/eval/finetuned.json
.venv/bin/python eval.py --compare          # อ่าน 2 ไฟล์ → พิมพ์ตารางเทียบ
```

- ไม่มี `output_dir`/checkpoint → error ข้อความ English ชัดเจน
- พิมพ์ progress ทุก 10 cases
- `--compare` สรุป: exact_match %, token F1 mean, ตัวอย่าง 3–5 ที่ fine ดีกว่า (§8.3)
- exit code: `--mode` = 0 เสมอ (เว้น error); `--compare` = 0 ถ้า fine ≥ base ทุก metric (พิมพ์ `PASS`), 1 ถ้า metric ใด base ดีกว่า (พิมพ์ `FAIL` + ชื่อ metric)

### 3.4 UI (Playground tab)

- ปุ่ม **Run Evaluation** (disabled ระหว่างเทรน — pattern เดียวกับ Playground/Merge lock)
- worker = spawn target ใหม่ `run_eval_worker(config, mode, queue_)` ใน `trainer_worker.py`
  (reuse validate_config + queue protocol) — ส่ง `log_msg` progress ทุก 10 cases,
  เสร็จแล้วเขียน JSON → UI อ่านแล้ว render `gr.Dataframe` (คอลัมน์: Metric | Base | Fine-tuned)
- ปุ่ม **Compare** (กดหลังรันครบ 2 modes) — หรือ auto-render เมื่อ JSON ครบ (เลือก: auto-render)
- ข้อความ UI ทั้งหมด English

## 4. Guardrail ที่เกี่ยวข้อง (§5)

| Guardrail | การปฏิบัติใน phase นี้ |
|---|---|
| VRAM Contention | eval = subprocess แยก + ปุ่มล็อกตอนเทรน |
| Process Stacking | ปุ่ม eval ผ่าน queue concurrency เดิม |
| Orphan Process | eval worker มี exit handler + watchdog เดิม |
| Driver Deadlock | `mp.set_start_method("spawn")` ที่ app.py เดิม |
| Network Exposure | ไม่แตะ — localhost เดิม |

## 5. Testing (test-once: เขียนครบก่อน แล้วรันทีเดียวตอนท้าย)

- **Unit (suite เดิม 91 + ใหม่):**
  - `exact_match` / `token_f1`: normalize, empty-equals-empty, multiset ordering ไม่สน
  - `build_eval_cases`: deterministic (เรียกซ้ำได้ชุดเดิม), ทุกตัวอย่างเป็น heldout, middle ≤ 256 tokens, ไม่วิ่งทะลุ limit
  - heldout filter ใน `run_training`: codes ผ่าน `is_heldout` ไม่หลุดเข้า `train_dataset` (mock iter_codes)
  - CLI: `--mode` เขียน JSON ถูก shape; `--compare` พิมพ์ตาราง + exit code
  - dashboard: ปุ่มมี event wiring + disabled ระหว่างเทรน (pattern เดิม)
- **Integration (`@pytest.mark.integration`, skip default):**
  - Abort: เริ่มเทรน → abort ตอน ~step 10 → assert process จบ, ไม่มี orphan, XPU free กลับ baseline
  - Disk e2e: mock `psutil.disk_usage` < 20GB → `inspect` คืน `insufficient_disk` → preflight block + วัด data_cache growth (informational)
- **Manual gates (บันทึกผลใน ledger):**
  - เทรน 500 ใหม่ → loss curve (§8.1)
  - eval 100 × 2 → **fine > base** (§8.2)
  - estimator: log `torch.xpu.max_memory_allocated()` ตอนเทรนจบ → เทียบ `estimate()` → ปรับ constant จนคลาด ≤10%
  - README walkthrough + `pip check` หลัง freeze

## 6. Error Handling

- ไม่มี checkpoint (ยังไม่เคยเทรน) → CLI/UI ข้อความ English: `No checkpoint found in <dir> — train first`
- heldout ไม่พอ 100 cases (dataset เล็ก) → ใช้เท่าที่มีแล้ว warn (ไม่ fail)
- generate timeout/OOM ใน eval worker → `error_msg` + process exit — ไม่ค้าง UI
- JSON อ่านไม่ได้/ไฟล์หาย → `--compare` บอกชัดว่าขาดไฟล์ไหน

## 7. Workflow

- worktree `.worktrees/phase5` → ledger `.superpowers/sdd/2026-10-02-phase5-*/progress.md`
- execute (native, test-once) → code review → fix → merge master → ล้าง worktree

## 8. Deferred / เปิดไว้

- Streaming eval (ไม่โหลดทั้งชุดเข้า memory) — 100 cases ยังเล็ก ไม่ต้อง
- เปรียบเทียบหลาย checkpoint (100 vs 500) — นอก scope
- BLEU/CodeBLEU — plan กำหนดแค่ Exact Match + Token F1
