# Spec: Phase A — Freeze Contracts (Sprint 1 / PR-01)

> **วันที่:** 2026-10-06  
> **ฐานโค้ด:** commit `fdc3469` (371 tests green)  
> **สถานะ:** design อนุมัติแล้วในการสนทนา → เขียนเป็นเอกสารรอบตรวจ (user review gate)  
> **ขอบเขต:** Phase A (Freeze Contracts) ครบทั้ง 3 สัญญา — RunManifest, Benchmark Report Schema v1.0, LocalRuntime interface  
> **อ้างอิง:** `WORK_PLAN.md` §2.2 (Artifact Contract), §2.3 (Unidirectional Rule), §3.1 FT-010, §3.2 BM-001/BM-006/BM-010, §6.1 สัปดาห์ที่ 1, §6.2 Sprint 1

---

## 1. บริบทและเป้าหมาย

ระบบปัจจุบัน (domain layout @ `fdc3469`) ยังไม่มีสัญญาข้อมูลที่เสถียรระหว่าง 3 ระบบย่อย:
- Fine-tuning ส่งต่อ run config ผ่าน IPC dict ไม่มีไฟล์ `run_manifest.json` (FT-010 🔄)
- Benchmark report ไม่มี `schema_version` และฟิลด์ memory/environment ไม่ครบ (BM-001 🔄, BM-006/010)
- Runtime ยังไม่มี interface กลาง — serving code ผูกกับ `llama-server` binary โดยตรง (Phase D/E ยังไม่เริ่ม)

**เป้าหมายของ Phase A:** ตรึงสัญญาทั้ง 3 ให้เป็น invariant ก่อนการ refactor ใหญ่ (Phase B–G) — สัญญาต้องนิ่ง ทดสอบล็อกได้ และทิศทาง dependency ไหลลงสู่ shared core เท่านั้น

**หลักเกณฑ์กลางที่ใช้ทั้งเอกสาร:** null-over-zero — ค่าวัด/ดึงไม่ได้จริงต้องเป็น `null` ห้ามเดา ห้ามใส่ 0 (§4.3)

---

## 2. การตัดสินใจเชิงสถาปัตยกรรม

**กลไกสคีมา: Python dataclasses ใน `core/contracts/` (approach A)**

| Approach | ผลที่ได้ | เหตุผลที่ไม่เลือก |
|---|---|---|
| **A. dataclass + `to_dict()`/`from_dict()` (เลือก)** | single source of truth ใน Python, typed, ทดสอบได้, ไม่เพิ่ม dependency, ไฟล์บนดิสก์เป็น plain JSON | — |
| B. JSON Schema files + `jsonschema` dep | language-neutral | เพิ่ม dependency + แหล่งความจริงที่สองต้อง sync กับ Python callers เอง — ตอนนี้ผู้บริโภคทุกตัวเป็น Python |
| C. dict constants อย่างเดียว | เร็วสุด | ไม่มี validation, สัญญาหลวม ทดสอบล็อกไม่อยู่ |

**ที่ตั้ง: `core/contracts/` (แพ็กเกจใหม่ 3 โมดูล)** — สัญญาเป็นของที่ subsystem ทั้งสามนำเข้าร่วมกัน จึงต้องอยู่ชั้น shared (เดียวกับ `core/infra/`) ไม่ใช่ในโมดูลของ subsystem ใด subsystem หนึ่ง — กฎ §2.3: `core/contracts/` import เฉพาะ stdlib เท่านั้น ห้าม import `core.train`/`core.compress`/`core.runtime`

---

## 3. สัญญาที่ 1: RunManifest — `core/contracts/manifest.py`

### 3.1 โครงสร้างไฟล์ `run_manifest.json` (schema_version 1.0)

```json
{
  "schema_version": "1.0",
  "artifact_type": "run_manifest",
  "created_at": "2026-10-06T12:00:00+00:00",
  "base_model": {
    "model_id": "Qwen/Qwen2.5-Coder-0.5B",
    "revision": "main",
    "num_params": null,
    "license": null
  },
  "dataset": {
    "dataset_id": "smangrul/hf-stack-v1",
    "column": "content",
    "code_limit": 8192,
    "fim_rate": 0.5,
    "heldout_ratio": 0.1,
    "train_samples": null,
    "heldout_samples": null
  },
  "training": {
    "seed": 42,
    "learning_rate": 0.0002,
    "max_steps": 500,
    "save_steps": 100,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "lora": { "rank": 8, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] }
  },
  "export": {
    "mode": "adapter_only",
    "path": "exports/finetune_run",
    "dtype": null
  },
  "git_commit": "fdc3469...",
  "tool_versions": { "torch": "2.14.1+xpu", "transformers": "...", "peft": "0.21.1" }
}
```

### 3.2 ที่มาของค่าแต่ละฟิลด์ (ห้ามเดา — ดึงไม่ได้ = `null`)

| ฟิลด์ | ที่มา | Null rule |
|---|---|---|
| `schema_version`, `artifact_type` | ค่าคงที่ `"1.0"` / `"run_manifest"` ในโมดูล | ไม่มี |
| `created_at` | `datetime.now(UTC).isoformat()` ตอนเขียนไฟล์ | ไม่มี |
| `base_model.model_id` | `config["model_id"]` (REQUIRED_CONFIG_KEYS — `core/train/args.py:113`) | ไม่มี (บังคับโดย validate_config อยู่แล้ว) |
| `base_model.revision` | `configs/safe_defaults.HF_HUB_REVISION` (`"main"`) | ไม่มี |
| `base_model.num_params` | **merged mode:** `model.num_parameters()` บนโมเดลที่เพิ่ง merge (มีอยู่จริงใน `merge_export`); **adapter_only:** ไม่มีโมเดลโหลด → `null` | null เมื่อไม่ได้วัดจริง |
| `base_model.license` | best-effort ครั้งเดียวตอน export: `huggingface_hub.model_info(model_id).cardData.get("license")` | try/except → `null` ทุกกรณี fail/ออฟไลน์ |
| `dataset.dataset_id`, `dataset.column` | `config["dataset_id"]`, `config["dataset_column"]` | ไม่มี (บังคับ) |
| `dataset.code_limit` | `config["code_limit"]` | ไม่มี (บังคับ) |
| `dataset.fim_rate`, `dataset.heldout_ratio` | `safe_defaults.FIM_RATE`, `HELDOUT_RATIO` | ไม่มี |
| `dataset.train_samples`, `heldout_samples` | ยังไม่มีใครส่งตัวเลขมา (runner คำนวณตอนเทรน ไม่ได้เขียนกลับ) | **`null` เสมอใน v1** — เติมจริงภายหลังโดยไม่ bump schema (nullable อยู่แล้ว) |
| `training.seed` | `safe_defaults.SEED` (42) — ค่าเดียวกับที่ `build_training_args` ใช้จริง | ไม่มี |
| `training.learning_rate` | `safe_defaults.LEARNING_RATE` (2e-4) | ไม่มี |
| `training.max_steps`, `training.max_seq_length`, `training.lora.rank` | `config["max_steps"]`, `config["max_seq_length"]`, `config["lora_rank"]` | ไม่มี (บังคับ) |
| `training.save_steps` | `config.get("save_steps", safe_defaults.SAVE_STEPS)` | ไม่มี |
| `training.batch_size` | `safe_defaults.DEFAULT_BATCH_SIZE` (1) | ไม่มี |
| `training.gradient_accumulation_steps` | `safe_defaults.GRADIENT_ACCUMULATION_STEPS` (8) | ไม่มี |
| `training.lora.target_modules` | `safe_defaults.LORA_TARGET_MODULES` — ป้ายกำกับ "configured targets" (โมเดลจริงอาจใช้ subset ตาม `available_lora_targets`) | ไม่มี |
| `export.mode` | `"adapter_only"` หรือ `"merged"` (โหมดที่กำลัง export) | ไม่มี |
| `export.path` | path ปลายทางที่ export คืนมา (relative จาก repo root เมื่อทำได้) | ไม่มี |
| `export.dtype` | merged: `"bfloat16"` (ค่าจริงใน `merge_export`); adapter_only: `null` | null เมื่อไม่มี |
| `git_commit` | `git rev-parse HEAD` ผ่าน subprocess ตอนเขียนไฟล์ | try/except → `null` |
| `tool_versions` | `importlib.metadata.version()` ต่อ package (`torch`, `transformers`, `peft`) | ต่อตัว → `null` เมื่อ package ไม่เจอ |

### 3.3 API ของโมดูล

```python
SCHEMA_VERSION: str = "1.0"

@dataclass(frozen=True)
class RunManifest:
    # nested dataclasses: BaseModelInfo, DatasetInfo, TrainingInfo, LoraInfo, ExportInfo
    def to_dict(self) -> dict: ...      # dataclasses.asdict
    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest": ...  # round-trip สำหรับผู้อ่าน (Benchmark/Archive)

def build_run_manifest(config: dict, *, export_mode: str, export_path: str,
                       num_params: int | None = None, dtype: str | None = None) -> RunManifest: ...
def write_run_manifest(manifest: RunManifest, dest_dir: str | Path) -> Path:
    # เขียน dest_dir / "run_manifest.json" — atomic: เขียน .tmp แล้ว os.replace (ไฟล์เดียว ต่างจาก checkpoint ที่เป็น dir)
```

### 3.4 การเชื่อมกับ `core/train/export.py`

- **`merge_export(config)`** — เรียก `write_run_manifest()` ก่อน `return dest` โดยส่ง `num_params=model.num_parameters()`, `dtype="bfloat16"`, `export_mode="merged"`
- **`save_adapter_only(output_dir, *, config, exports_dir="exports")`** — **เพิ่ม parameter `config` ที่จำเป็น** (เดิมรับเฉพาะ output_dir) เพื่อให้ manifest เขียนได้ครบทุกครั้ง; เรียก `write_run_manifest()` ก่อน `return dest` (`export_mode="adapter_only"`, `num_params=None`, `dtype=None`)
  - ผู้เรียกปัจจุบันที่ไม่ส่ง config ต้องถูกอัปเดตให้ส่ง (มีใน UI export flow ซึ่งถือ config dict อยู่แล้ว) — implementation plan จะ grep ผู้เรียกทุกตัว
  - เทสต์เดิมใน `tests/train/test_export.py` ที่เรียกโดยไม่มี config ต้องอัปเดตให้ส่ง config fixture
- **ข้อห้าม:** manifest เขียนหลัง export สำเร็จเท่านั้น — ถ้า copy/merge พัง ห้ามมี `run_manifest.json` ค้างหลอกใน dest (เขียนเป็น step สุดท้าย)

**ตำแหน่งไฟล์ผลลัพธ์:** `exports/<name>/run_manifest.json` (adapter) และ `exports/<name>-merged/run_manifest.json` (merged) — ข้าง artifact ที่ manifest อธิบายเสมอ

---

## 4. สัญญาที่ 2: Benchmark Report Schema v1.0 — `core/contracts/benchmark.py`

### 4.1 ค่าคงที่และ validator

```python
SCHEMA_VERSION: str = "1.0"

REPORT_KEYS: frozenset[str] = frozenset({
    # --- มีอยู่แล้วใน build_report (core/compress/report.py:38-59) ---
    "model", "variant", "optimization", "backend", "weight_bits",
    "context_tokens", "parameter_count", "model_disk_mb",
    "peak_vram_mb", "kv_cache_mb",
    "tokens_per_sec", "prompt_tokens_per_sec", "latency_ms", "load_time_ms",
    "exact_match_pct", "token_f1", "syntax_pass_rate", "execution_pass_rate",
    "eval", "timestamp",
    # --- เพิ่มใหม่ใน v1.0 ---
    "schema_version",   # "1.0" เสมอ
    "peak_rss_mb",      # nullable — วัดแล้วแต่ CLI ยังไม่เขียน (BM-006); null ก่อน
    "environment",      # nullable dict {hardware, os, llama_cpp_commit} — ยังไม่เก็บ (BM-010); null ก่อน
})  # = 23 คีย์

def validate_report(data: dict) -> None:
    # 1) key set ต้องตรง REPORT_KEYS เป๊ะ (ขาด/เกิน = ValueError)
    # 2) data["schema_version"] == SCHEMA_VERSION (ไม่ตรง = ValueError — กัน report เก่าไหลเข้าระบบใหม่)
    # ไม่ตรวจชนิด/ช่วงค่า metrics — null-over-zero ดูแลโดยผู้สร้างรายงาน
```

### 4.2 การเชื่อมกับ `core/compress/report.py`

- `build_report()` import `SCHEMA_VERSION` จาก `core.contracts.benchmark` แล้ว:
  - เพิ่ม key `"schema_version": SCHEMA_VERSION`
  - เพิ่ม key `"peak_rss_mb": None` (ค่าวัดจริงจะถูก CLI เติมใน Sprint 3 — รอบนี้ null ตาม null-over-zero)
  - เพิ่ม key `"environment": None` (โครง `{hardware, os, llama_cpp_commit}` — เก็บจริง Sprint 3/BM-010)
- **คงโครงสร้าง flat เดิมทั้งหมด** ไม่จับกลุ่มเป็น Identity/Artifact/Performance/Memory/Quality — ผู้บริโภคปัจจุบัน (`scripts/benchmark_compression.py`, `format_table`) อ่านคีย์ flat; การจับกลุ่มเป็น schema v2 (รอ Sprint 3 พร้อม variant-vs-variant rework — นอกขอบเขตรอบนี้)
- `write_report()` เรียก `validate_report(data)` ก่อนเขียนไฟล์ทุกครั้ง (gate ที่ขอบระบบ — report พัง = ไม่เขียน)

### 4.3 ความเข้ากันได้

รายงานเก่า (ไม่มี `schema_version`) จะ fail `validate_report` — ตั้งใจ: รายงานทุกไฟล์หลังสัญญาตรึงต้องระบุเวอร์ชันได้ รายงานเก่าใน `benchmarks/` ยังไม่มีอยู่จริง (dir ยังไม่ถูกสร้าง — ตรวจแล้ว) จึงไม่มีของที่ต้อง migrate

---

## 5. สัญญาที่ 3: LocalRuntime ABC — `core/contracts/runtime.py`

### 5.1 Interface

```python
class LocalRuntime(ABC):
    """สัญญา runtime สำหรับเสิร์ฟ GGUF — backend-agnostic (llama-server binary วันนี้,
    llama-cpp-python ในอนาคต Phase E) ทุก method ห้าม raise ข้อความกำกวม:
    start ล้ม = ต้อง raise พร้อมเหตุผล; is_ready = poll จน timeout แล้วคืน bool"""

    @abstractmethod
    def start(self, model_path: Path, *, device: str = "vulkan",
              context_tokens: int | None = None) -> None:
        """โหลด GGUF แล้วเปิด endpoint — context_tokens=None = ปล่อยให้ backend ใช้ค่า native
        (ณ วันนี้ serving ไม่ pin context — RM-008 🔄; เมื่อไหร่ pin ค่า ให้ส่งจาก caller)"""

    @abstractmethod
    def is_ready(self, timeout_s: float) -> bool:
        """poll health จน ready หรือหมดเวลา — ต้องรวม identity check (โมเดลที่เปิด = โมเดลที่ขอ)"""

    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int,
                 temperature: float | None = None, stop: list[str] | None = None) -> str:
        """FIM completion ดิบ — คง raw prompt semantics (BM-008) ห้าม strip/insert เงียบ ๆ
        max_tokens maps ไป n_predict ของ backend เอง (หน้าที่ adapter)"""

    @abstractmethod
    def stop(self) -> None:
        """graceful stop แล้ว escalate kill ตาม §4.2 (SIGINT → รอ grace → SIGKILL) — เรียกซ้ำได้ (idempotent)"""
```

### 5.2 ขอบเขตและ non-goals รอบนี้

- **มี:** ABC + docstring contract + เทสต์ล็อก (abstract class instantiate ไม่ได้; signature คงที่ผ่าน `inspect.signature`)
- **ไม่มี:** adapter ตัวไหนทั้งสิ้น — โค้ด `core/compress/llama_runner.py` ยังคงเป็น implementation เดียวที่ใช้งานจริงต่อไปจน Phase D เขียน adapter ครอบมัน และ Phase E เพิ่ม llama-cpp-python backend
- ห้าม import `llama_runner` (หรือ `llama_cpp`) จาก `core/contracts/` เด็ดขาด — สัญญาห้ามผูกกับ backend

---

## 6. ทิศทาง Dependency

```text
core/contracts/  (stdlib เท่านั้น)
  ▲
  ├── core/train/       (เขียน run_manifest.json ตอน export)
  ├── core/compress/    (build_report ใช้ SCHEMA_VERSION + validate ก่อนเขียนไฟล์)
  └── core/runtime/     (อนาคต — adapter ของ LocalRuntime; ยังไม่มีในรอบนี้)
```

ห้ามมีการ import ย้อนกลับจาก `core/contracts/` ไปยังทั้งสามข้าง — ทดสอบด้วยการ assert ในเทสต์ contracts (import `core.contracts.*` แล้ว inspect `sys.modules` ไม่มี `core.train`/`core.compress`/`core.runtime`) หรือ grep-based guard อย่างง่าย

---

## 7. รายการไฟล์ที่แตะ (10 ไฟล์)

| ไฟล์ | การกระทำ |
|---|---|
| `core/contracts/__init__.py` | ใหม่ — export ชื่อสาธารณะของทั้ง 3 สัญญา |
| `core/contracts/manifest.py` | ใหม่ — RunManifest dataclass + build/write helpers |
| `core/contracts/benchmark.py` | ใหม่ — SCHEMA_VERSION + REPORT_KEYS + validate_report |
| `core/contracts/runtime.py` | ใหม่ — LocalRuntime ABC |
| `core/train/export.py` | แก้ — `save_adapter_only` เพิ่ม param `config`; ทั้ง 2 โหมดเรียก `write_run_manifest()` เป็น step สุดท้าย |
| `core/compress/report.py` | แก้ — import จาก contracts, เพิ่ม 3 คีย์, `write_report` validate ก่อนเขียน |
| `tests/contracts/test_manifest.py` | ใหม่ |
| `tests/contracts/test_benchmark_report.py` | ใหม่ |
| `tests/contracts/test_runtime.py` | ใหม่ |
| `tests/train/test_export.py` | แก้ — fixture config + assert `run_manifest.json` ถูกเขียน |

(ผู้เรียก `save_adapter_only` นอก `tests/` ที่ต้องส่ง config เพิ่ม — จะไล่ grep ในขั้น implementation plan)

---

## 8. เกณฑ์ยอมรับ (Acceptance Criteria)

1. **Manifest round-trip:** `build_run_manifest(config_fixture) → to_dict → write → from_dict` ได้โครงสร้างเท่ากันทุกคีย์; `schema_version == "1.0"`
2. **Null-over-zero:** config fixture ที่ออฟไลน์ → `license == null`, `git_commit == null` (ไม่มีค่าเดา ไม่มี 0 แทน null)
3. **Export integration:** `save_adapter_only` (tmp_path fixture + monkeypatch `model_info`/`subprocess.run`) เขียน `run_manifest.json` ลง dest สำเร็จ — ไม่แตะเน็ตจริง; ฝั่ง `merge_export` ทดสอบที่ระดับ unit (`build_run_manifest(export_mode="merged", num_params=..., dtype=...)` + write) เท่านั้น — integration จริงของ merge ต้องใช้โมเดลจริง ~2× RAM ซึ่งเทสต์เดิมก็ไม่เคยรัน (assert `callable` เท่านั้น) คงมาตรฐานเดิม
4. **Fail ไม่ค้าง:** export พังกลางทาง (เช่น checkpoint ไม่มีน้ำหนัก — เทสต์เดิม) → ไม่มี `run_manifest.json` ใน dest
5. **Report schema:** `build_report()` คืนคีย์ครบ 23 ตัว รวม `schema_version="1.0"`, `peak_rss_mb=None`, `environment=None`; `write_report` ปฏิเสธ dict ที่ key ไม่ครบ/เวอร์ชันผิด
6. **LocalRuntime:** instantiate ไม่ได้; ทั้ง 4 method มี signature ตรงตาม spec (ล็อกด้วย inspect)
7. **Dependency guard:** `import core.contracts.manifest/benchmark/runtime` แล้วไม่มี `core.train`/`core.compress`/`core.runtime` หลุดเข้า `sys.modules`
8. **Regression:** `pytest tests/` = ชุดเดิม 371 + เทสต์ใหม่ทั้งหมดเขียว 100% (§6.3) — ห้าม skip/deselect เพิ่ม

---

## 9. Non-goals / Deferred (ทำภายหลัง ไม่บั่นทอนสัญญา)

| เรื่อง | ไปอยู่ที่ | เหตุผลที่เลื่อนได้ |
|---|---|---|
| เติม `train_samples`/`heldout_samples` จริง | Sprint 2+ (runner เขียนกลับ config) | ฟิลด์ nullable อยู่แล้ว — ไม่ต้อง bump schema |
| เติม `peak_rss_mb`/`environment` ค่าจริง | Sprint 3 (BM-006/BM-010) | ฟิลด์พร้อมใน schema แล้ว — CLI เติมทีหลัง |
| LocalRuntime adapter ตัวแรก | Phase D | สัญญานิ่งก่อน implementation (contract-first ตาม §6.1) |
| llama-cpp-python backend | Phase E (Sprint 4) | — |
| Benchmark schema จับกลุ่ม Identity/Artifact/... เป็น v2 | Sprint 3 พร้อม variant-vs-variant (BM-009) | v1 คง flat เพื่อไม่ทำลายผู้บริโภคปัจจุบัน |
| Pin context ฝั่ง serving | RM-008 fix (Sprint 4) | interface เปิดรับ `context_tokens` ไว้แล้ว |
| Rename `scripts/benchmark_compression.py` → `benchmark.py` | Sprint 3 (BM-011) | ไม่เกี่ยวกับสัญญา |
| ย้าย `core/compress` → `core/benchmark` | Phase C (Sprint 3) | สัญญาตั้งใน `core/contracts/` อยู่แล้ว ย้ายภายหลังไม่กระทบ |

---

## 10. Invariants ที่ต้องคงไว้ระหว่าง implementation

1. **Null-over-zero ทุกฟิลด์** — ดึงไม่ได้ = `null` ห้าม 0 ห้ามเดา (§4.3, §6.3)
2. **Unidirectional dependency** — `core/contracts/` ห้าม import ข้างบน (§2.3); สัญญาไม่รู้จัก backend
3. **Manifest เขียนเฉพาะตอน export สำเร็จ** — ห้ามมีไฟล์หลอกเมื่อ export พัง
4. **ค่าคงที่มาจาก `configs/safe_defaults.py` เท่านั้น** — manifest ห้าม hardcode seed/lr/batch ซ้ำ (FT-001)
5. **371+ tests ต้องเขียว 100%** — ห้ามแก้ expectations เก่าเพื่อให้เขียว นอกจากการเพิ่ม param `config` ที่เป็นส่วนหนึ่งของสัญญา (เทสต์เดิมที่เรียก `save_adapter_only` โดยไม่มี config ต้องอัปเดตให้ส่ง config fixture ซึ่งเป็นการ follow สัญญาใหม่ ไม่ใช่การลดเกณฑ์)
6. **Atomic write** — `run_manifest.json` เขียนแบบ tmp + `os.replace` (ไฟล์เดียว) เหมือนแนวทาง checkpoint ที่มีอยู่
