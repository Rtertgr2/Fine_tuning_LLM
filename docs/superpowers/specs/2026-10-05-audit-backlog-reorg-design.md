# Audit Backlog Execution + Domain Restructure — Design Spec

**วันที่:** 2026-10-05 · **สถานะ:** design approved (conversation) → รอ written review
**ฐาน code:** `fix/ml-logic` @ 82452d5 — audit ทั้งหมดตรวจสอบที่ commit นี้
**Branch ทำงาน:** `fix/audit-backlog` (สร้างจาก 82452d5 แล้ว)

## 1. Context

Audit 3 ชุด (P0/P1/P2 + ML logic + Wave 1) ระบุปัญหา ~60+ ข้อในโค้ด ณ 82452d5 บางข้อ
มี refs ผิด/เกินจริง (⚠️) — สเปคฉบับนี้บันทึก disposition ทุกข้อ ตามด้วยการย้ายโครงสร้าง
directory ตาม domain (Approach B ที่เลือก) และ clean code เป็นงานสุดท้าย

ลำดับงานที่ตกลง: **P0 → P1 → ML → P2 → Wave 1 → Reorg → Clean code**
(ทุกอย่างใน branch เดียว, commit แยกตาม wave)

## 2. Goals / Non-goals

**Goals**
- ปิด audit ทุก wave ตามลำดับ — **wave ละ 1 commit** (แยกเพิ่มเฉพาะเมื่อ wave นั้นใหญ่จริง)
- ทุกข้อมี test (ข้อ P0 ระบุ "+ test" ชัดเจน) — suite ผ่านทุก commit
- ทุกข้อมี disposition บันทึก: fixed / mitigated / dropped / re-derive — **ไม่ปิดเงียบ**
- Reorg: `core/` ถังรวม → domain packages; `trainer_worker.py` 589 บรรทัด → 7 modules; tests mirror
- Clean code: dedupe ค่าคงที่เข้า `configs/safe_defaults.py`, ลบของซ้ำ/ไม่ทำงาน, จัดเนื้อในไฟล์

**Non-goals**
- ไม่เพิ่ม formatter/linter/pre-commit (user เลือกแบบไม่มี)
- ไม่ทำ packaging (pyproject / src layout / console scripts)
- ไม่แตะ `tools/llama.cpp`, `data_cache/`, `datasets/`, `models/`, `exports/`, `benchmarks/`, `diagnosis-issue-draft.md`
- ไม่แก้ historical specs/plans ใน `docs/superpowers/**` (เอกสารย้อนหลังมีวันที่) — แก้เฉพาะ README
- ไม่ merge กลับ master ในงานนี้ (การ integrate เป็นงานหลังเสร็จ)

## 3. Verified dispositions (ข้อ ⚠️ — verify จากโค้ดจริงแล้ว 2026-10-05)

| ข้อ | ผล verify ที่ 82452d5 | Disposition |
|---|---|---|
| A5 | `trainer_worker.py:75` = `bool(torch.xpu.is_bf16_supported())` เปล่า ๆ — ไม่มี `including_emulation` ตามที่รายงานอ้าง | **dropped** (re-derive + verify แล้ว 2026-10-05) — refs ผิด, ไม่มี substance ให้แก้; fallback True+warning (`:78-83`) มีครบ, native-only bf16 = ทางเลือกปลอดภัยแล้ว |
| B4 | `dataset_builder.py:236` = `iter_batches(batch_size=64, columns=[column])` — ไม่มี `batch_size=65536` / `row_groups` | **dropped** (re-derive + verify แล้ว 2026-10-05) — refs ผิด, โค้ดอ่าน bounded อยู่แล้ว ไม่มีอะไรต้องแก้ |
| D9 | peft 0.21.1: `utils/other.py` = 1,838 บรรทัด ✅ / ค่าจริงอยู่ `tuners/tuners_utils.py` (มีบรรทัด 2705 จริง) | audit แก้ refs ถูก → ใช้ refs ใหม่, substance ยังอยู่ใน P1/P2 |
| D11 | doc §9 ชี้ `core/ipc_bridge.py::abort_process` ถูกต้องแล้ว | **dropped** (item ปลอม) — ทั้งข้อ D11 และบรรทัด "§9 ชี้ผิดไฟล์" ใน Doc list |
| D2-doc | docs มี `add_special_tokens` จริง 5 ไฟล์ (รายงานบอก grep→0 ผิด) — แต่ **code mismatch train=False/eval=True จริง** | แก้ code (D2); ตรวจ claim ใน docs ตอน wave |
| A1 | ไม่มี `logging_nan_inf_filter` ใน `trainer_worker.py` เลย → default True ตามที่รายงาน | **P0 ยังอยู่** ✅ |

## 4. Wave inventory & sequence

### P0 — ทำก่อน (3 ข้อ, ผลกระทบจริง)

| # | ที่ | ปัญหา / งาน |
|---|---|---|
| A1 | trainer_worker.py:103-124 | `logging_nan_inf_filter` default True → transformers กรอง non-finite ก่อนถึง on_log → NaN guard ไม่ทำงาน loss ถูก smooth ตอน diverging → ตั้ง `False` ชัดเจนใน `build_training_args` + test |
| C1 | dashboard.py:374 | `start_btn.click` ไม่มี `concurrency_id` → Start ซ้อน eval/predict ค้าง = โหลดโมเดล 2 process = OOM → เพิ่ม `concurrency_id` + test (ข้อ 9 ใน Security ซ้ำกัน → ทำที่นี่ที่เดียว) |
| D1 | evaluator.py:134 | `decode(skip_special_tokens=True)` ตัด FIM marker ไม่ได้ (id 151659-61 ไม่ใช่ special) → EM/F1 ทุกชุดเพี้ยน → logic กรอง FIM ids หลัง decode + test |

### P1 — รอบถัดไป

| # | ที่ | ปัญหา |
|---|---|---|
| A2 | trainer_worker.py:110 | ไม่ตั้ง `per_device_eval_batch_size` → default 8 (train=1) + packing แถวยาวเต็ม → OOM ตอน eval |
| A3 | trainer_worker.py:249 | `logs.get("learning_rate", 0.0)` → key หาย = โชว์ LR 0.0 เหมือนจริง = กราฟโกหก |
| B1 | estimator.py:157 | `hf_hub_download` ไม่ pin `revision`/`cache_dir` → ดึง main ไม่ตรึงเวอร์ชัน + cache คนละที่กับ data_cache |
| C2 ⭐ | app.py:41 | `default_concurrency_limit=1` ไม่มีผล (default อยู่แล้วเป็น 1 + limit เป็นราย concurrency-id) · docstring :4 อ้างว่ากัน process stacking = ไม่จริง — **ข้อนี้หายไปจากรายการต้นทาง** |
| C3 | dashboard.py:233 | ผสม `gr.update()` กับ instance pattern (installed source ทำเครื่องหมาย deprecated path) |
| C4 | components.py:39 | มี `label=` 3 เส้น แต่ไม่เคยเรียก `ax.legend()` → แยก loss/val_loss/lr ไม่ได้ |
| C5 | components.py:31 | `plt.subplots()` ใช้ global state บน Gradio threads ไม่มี lock → รูปเพี้ยนตอนเทรนจริง |
| D2 | evaluator.py:127 | tokenize ด้วย `add_special_tokens=True` แต่ฝั่งเทรนใช้ `False` → model ที่เติม BOS mismatch |
| D3 | hardware.py:23 | `hasattr(torch,"xpu")` ไม่ใช่ capability check → คืน True ตลอด |
| D4 | ipc_bridge.py:101 | `abort_process` เรียก `terminate()` บน process ที่ยังไม่ start → error |
| A5/B4/D9 | ดู §3 | refs ผิด — re-derive ตอน wave, หา substance ไม่เจอ = dropped |

> นับจำนวน: รายการต้นทางบอก "11 ข้อ" แต่ rows ที่ระบุชัด = 10 + A5/B4/D9 (refs ผิด) — inventory ข้างบนเป็น authoritative

### ML Logic (4 ข้อ — เลื่อนไว้)

| # | ที่ | ปัญหา |
|---|---|---|
| M4 | dataset_builder.py:99-112 | dedup ใช้ md5 exact อย่างเดียว → code ใกล้เคียงรั่ว train→heldout = EM/F1 บวกเทียม |
| M8 | — | ไม่มี `resume_from_checkpoint` → abort ทุกครั้ง state หาย เริ่มใหม่จาก adapter |
| M9 | eval.py:80-83 | gate ไม่มี min-delta (+0.0 ก็ PASS) + ไม่มี quant-regression gate อัตโนมัติ |
| M10 | evaluator.py:58-71 | token-F1 = bag-of-tokens → middle สลับลำดับได้ F1 = 1.0 |

Design ของ M4/M8/M9/M10 ตัดสินตอน wave (ถ้าขยายเกินขอบเขต → step up ratchet, หยุดถามก่อน)

### P2 — ความสะอาด (17 refs)

A4 return control · A6 `eval_steps` ตั้งทั้งที่ไม่มี eval dataset · A5⚠ (§3) ·
B2 `cache_dir` เป็น no-op ตอน streaming · B3 streaming เปลี่ยน error surface ·
B4⚠ (§3) · B5 จับ error กว้าง · B6 error ไม่บอกว่าไฟล์ไหนหาย ·
C6 doc ผิด (Timer = วินาที ไม่ใช่ ms) · C7 `active=True` ซ้ำ default ·
D5 hardcode device 0 · D6 เช็คดิสก์คนละ path · D7 smoke test ไม่ probe bf16 ·
D8 `no_grad` ซ้ำ · D9⚠ (§3) · D10 `--compare` ดัน torch ทั้งชุด · D11⚠ → **dropped**

### Wave 1 (Standards 11 / Spec 8 / Security 10 / Doc 6)

**Standards (hard violations)**
1. error ภาษาไทยหลุดเข้า UI — `dataset_builder.py:43`
2. CLI copy ไทย — `check_runtime.py:31,40,48,50,61,63`
3. hardcode ซ้ำ ขัด README:186 — `check_runtime.py:14-18`; `EVAL_DIR` 2 นิยาม (`evaluator.py:40` relative vs `compression/report.py:12` absolute); `n_cases=100` 4 จุด

**Spec**
4. requirements ไม่ pin — `requirements.txt:21-22`
5. Live log ไม่ครบ — `dashboard.py:43,144` ตัด 200 บรรทัด
6. ที่อยู่ report ไม่ตรง layout — `compression/config.py:122`
7. LoRA slider `step=8` → ได้ 8/16/24/32 (spec บอก 8,16,32) — `dashboard.py:291-293`
8. preflight fallback = `MAX_SEQ_LENGTH_CAP` 2048 ไม่ใช่ DEFAULT 1024 → ประเมิน VRAM ผิดเงียบ ๆ — `controller.py:103`

**Security (2 High)**
9. 🔴 Start ไม่ถูกกันระหว่าง eval/predict → OOM — **ซ้ำกับ C1 (P0) → ทำที่ P0**
10. 🔴 `on_save_adapter` ไม่ผ่าน `validate_config` → copy checkpoint ออกนอก sandbox ได้ — `dashboard.py:212-217` → `trainer_worker.py:490-505`
11. delta gate เช็คแค่ n ไม่เทียบ dataset/column — `benchmark_compression.py:116-140`
12. `commit_checkpoint` rename 2 ขั้น → SIGKILL กลางทาง = checkpoint หาย — `trainer_worker.py:160-169`
13. peft CVE-2026-71281 ยังไม่มี patch → **policy §7**
14. zombie ไม่ reset status → ปุ่ม Start ล็อก

**Doc (6 จุด)**
- Timer หน่วยวินาที · logs มี step เฉพาะ history entry · ขาด `mem_get_info`/`get_device_name` ·
  §4 เป็นสัญญา `datasets` แต่โค้ดใช้ pyarrow ดิบ · label→legend · ~~§9 ชี้ผิดไฟล์~~ → **dropped (D11)**

> หมายเหตุนับจำนวน: รายการต้นทางระบุ "29 ข้อ" แต่ rows ที่ transcribe ได้ 35 (มี 1 ซ้ำ = #9/C1,
> 1 dropped = §9/D11) — inventory ข้างบนเป็น authoritative แทนตัวเลข 29

## 5. Target structure (Approach B — approved)

**หลักการ:** ชื่อไฟล์คงเดิมเท่าที่ยังถูกต้อง · `ui/`, `scripts/`, root entries ไม่ย้าย ·
ย้ายเฉพาะ `core/` + แตก `trainer_worker.py` + tests mirror

```
BEFORE                         AFTER
core/hardware.py            → core/infra/hardware.py
core/estimator.py           → core/infra/estimator.py
core/ipc_bridge.py          → core/infra/ipc_bridge.py
core/sandbox.py             → core/infra/sandbox.py
core/dataset_builder.py     → core/data/dataset_builder.py
core/evaluator.py           → core/eval/evaluator.py
core/compression/*  (7 ไฟล์) → core/compress/*  (คงชื่อไฟล์)
core/trainer_worker.py      → แตก 7 modules:
    core/train/args.py        available_lora_targets, _xpu_bf16_supported,
                              build_training_args (A1 ลงที่นี่), peak_xpu_memory_gb,
                              validate_config
    core/train/callbacks.py   NanGuard, AtomicSaveTrainer, StreamToQueueCallback
    core/train/runner.py      run_training, is_checkpoint_dir, commit_checkpoint,
                              latest_checkpoint
    core/train/export.py      save_adapter_only, merge_export
    core/train/predict.py     predict_middle
    core/data/fim.py          ensure_fim_tokens, build_fim_prompt
    core/eval/worker.py       run_eval_worker, EVAL_DONE
+ ลบ re-export แปลกๆ: evaluator/benchmark_compression ที่ import
  AutoTokenizer/AutoModelForCausalLM จาก trainer_worker → import จาก transformers ตรงๆ

app.py, eval.py, ui/*, scripts/*  → คงที่เดิม
tests/  → mirror: tests/{infra,data,train,eval,compress,ui}/
          ( basename 23 ไฟล์ไม่ซ้ำ → pytest ไม่ชน )
          test_phase5_integration.py, test_safe_defaults.py อยู่ root ต่อ
```

**Caveats ที่บังคับตอน reorg:**
- `Path(__file__).parents[N]` depth-sensitive (เช่น `compression/report.py:12`) →
  grep ทุกจุดแล้ว recompute (`EVAL_DIR` รวมเป็น single source ไปแล้วตอน Wave 1 — reorg แค่ย้ายที่อยู่)
- `mp spawn` pickles function โดย module path → ทุก `target=` ใน `ui/controller.py` แก้พร้อมกัน
- README §โครงสร้างโปรเจค + ข้ออ้าง path ใน docs ที่ยังใช้งาน → อัปเดตใน reorg commit
- คอมเมนต์ที่อ้าง path/line เดิม (เช่น "Fix.md L1") ที่ยัง valid → ไม่ต้องแก้; อ้างไฟล์ที่ย้าย → แก้ตาม

## 6. Clean code scope (commit สุดท้าย)

- **Dedupe ค่าคงที่ → `configs/safe_defaults.py`** (กติกา README:186): `EVAL_DIR` (2→1),
  `n_cases=100` (4→1), hardcode ใน `check_runtime.py:14-18` — *หลักทำใน Wave 1; commit นี้เก็บตก*
- **ลบของซ้ำ/ไม่ทำงาน:** `no_grad` ซ้ำ (D8), `active=True` ซ้ำ default (C7), re-export แปลกๆ
- **จัดเนื้อในไฟล์ที่แตกใหม่:** module docstring ใหม่ชี้ที่อยู่จริง; imports/constants order
- **Thai strings ใน UI/CLI → error อังกฤษ** (Wave1 #1,2 ทำหลัก; commit นี้เก็บตก)

## 7. Policies

- **Docs:** README อัปเดตใน reorg commit · historical specs/plans ไม่แตะ ·
  `diagnosis-issue-draft.md` + `tools/` (untracked ของ user) ไม่แตะ
- **peft CVE-2026-71281:** ตอน implement เช็ค peft > 0.21.1; ถ้ายังไม่มี patch →
  จด known-issue + mitigation (pin ปัจจุบันใน requirements/constraints) ใน README — **ไม่ปิดเงียบ**
- **Dropped:** D11/§9-doc (verify แล้วผิด) · A5/B4 ถ้า re-derive ไม่เจอ substance →
  บันทึก "dropped — report refs wrong" ใน spec ตอน wave
- **Verification:** `pytest -m "not integration"` (`pytest.ini` ตั้งอยู่แล้ว) ผ่านทุก commit ·
  integration tests รัน manual ตอนท้าย · `scripts/pipeline_smoke.py` เฉพาะ wave ที่แตะ training path

## 8. Branch & commits

Branch `fix/audit-backlog` จาก 82452d5 — commit ลำดับ:
`docs(spec)` → `fix(p0)` → `fix(p1)` → `fix(ml)` → `fix(p2)` → `fix(wave1)` → `refactor(reorg)` → `refactor(clean)`

## 9. Risks / open decisions

- **A5/B4 substance** อาจหาไม่เจอ → dropped ตาม §7
- **M4/M8/M9/M10 design** ตัดสินตอน wave — ถ้าเรื่องใหญ่กว่าคาด → ratchet: หยุด รายงาน ถามก่อนทำ
- **Reorg churn** ~ทุก import + tests 23 ไฟล์ → mechanical; pytest จับ regression — ถ้าไฟล์แตกใหม่
  ทำ test import พัง ให้แก้ test ให้ mirror structure ใหม่
- **Environment:** รัน tests ด้วย `.venv/bin/python -m pytest` จาก repo root (Python 3.11, torch XPU,
  peft 0.21.1) — system python ไม่มี deps
