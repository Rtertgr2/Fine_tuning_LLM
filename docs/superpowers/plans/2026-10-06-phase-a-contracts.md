# Phase A — Freeze Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ตรึงสัญญาทั้ง 3 ของระบบ — RunManifest, Benchmark Report Schema v1.0, LocalRuntime ABC — เป็น dataclass/validator ที่เทสต์ล็อกได้ใน `core/contracts/` แล้วเชื่อมเข้ากับ export + report path โดยคง invariant ทั้งหมดของ spec

**Architecture:** แพ็กเกจใหม่ `core/contracts/` (3 โมดูล + `__init__`) import ได้เฉพาะ stdlib + `configs/safe_defaults` — ฝั่ง `core/train/export.py` เขียน `run_manifest.json` เป็น step สุดท้ายตอน export สำเร็จ, `core/compress/report.py` ดึง `SCHEMA_VERSION` จาก contracts และ validate ก่อนเขียนไฟล์, LocalRuntime เป็น ABC เปล่า (ไม่มี adapter)

**Tech Stack:** Python 3 dataclasses, pytest (รันด้วย `.venv/bin/python -m pytest tests/` — system python ไม่มี pytest)

**Spec:** `docs/superpowers/specs/2026-10-06-phase-a-contracts-design.md` (อนุมัติแล้ว — ตาราง §3.2 คือ source-of-truth ของค่า manifest ทุกฟิลด์)

## Global Constraints

1. **Null-over-zero ทุกฟิลด์** — ดึงค่าไม่ได้จริง = `null` ห้าม `0` ห้ามเดา (spec §4.3, invariant #1)
2. **Unidirectional dependency** — `core/contracts/` ห้าม import `core.train` / `core.compress` / `core.runtime` / `llama_runner` / `llama_cpp`; อนุญาตเฉพาะ stdlib + `configs.safe_defaults` (spec §2, §6, §5.2, invariant #2) — guard test อยู่ใน Task 6
3. **ค่าคงที่มาจาก `configs/safe_defaults.py` เท่านั้น** — manifest ห้าม hardcode seed/lr/batch/ga/fim_rate ซ้ำ (invariant #4)
4. **Manifest เขียนเฉพาะตอน export สำเร็จ** — export พัง = ไม่ต้องมี `run_manifest.json` ค้าง (invariant #3)
5. **Atomic write** — `run_manifest.json` เขียน tmp + `os.replace` (invariant #6)
6. **371 tests เดิมเขียว 100% ห้าม skip/deselect เพิ่ม** — แก้ expectations เดิมได้เฉพาะที่ follow สัญญาใหม่ (เช่น เพิ่ม param `config`, เพิ่มคีย์ใหม่) ห้ามลดเกณฑ์ (invariant #5)
7. **Report คงโครงสร้าง flat 23 คีย์** — ห้ามจับกลุ่มเป็น Identity/Artifact/... (v2 deferred, spec §4.2)
8. **Atomicity scope = ไฟล์เดียว** — ต่างจาก checkpoint (dir) — ใช้ `os.replace` ตรง ๆ

## Deviation ที่จงใจ (resolve ความขัดแย้งใน spec — ทวนอีกครั้งตอน review)

1. **`build_run_manifest` ได้ param เพิ่ม `license: str | None = None`** — spec §3.3 ล็อก signature ไม่มี field นี้ แต่ §3.2 สั่งให้ดึงจาก `huggingface_hub.model_info` ซึ่งขัดกับ invariant "contracts = stdlib เท่านั้น" — ตัดสินโดยให้ **`core/train/export.py` เป็นคน fetch** (ลอง/พลาด → `None`) แล้วส่งผ่าน param เข้ามา → contracts สะอาด ไม่มี network I/O ใน builder, guard test ผ่านจริง
2. **`git_commit` ดึงตอน build (ไม่ใช่ตอน write)** — spec §3.2 บอก "ตอนเขียนไฟล์" แต่ถ้า fill ทีหลังจะทำ round-trip `build → to_dict → write → from_dict` ไม่เท่ากัน (acceptance #1) — จึงดึงใน `build_run_manifest` ผ่าน `subprocess` (stdlib) try/except → `null`

## Review Focus

(5 failure modes ที่ spec บอกนัยไว้แต่ไม่มี test ตรง ๆ — แต่ละบรรทัดชี้ test ที่ pin มัน)

1. report ที่ hand-built/ขาดคีย์/เวอร์ชันเก่าไหลเขียนไฟล์ → `write_report` ต้อง `ValueError` และ **ไม่เขียนไฟล์เลย** — `tests/compress/test_compression_report.py::test_write_report_rejects_invalid_and_stale_schema` (Task 4)
2. export พังกลางทางแล้ว manifest ค้างหลอก → ต้องไม่มี `run_manifest.json` ใน dest — `tests/train/test_export.py::test_save_adapter_only_config_without_weights_raises` (ขยาย assert, Task 2)
3. `model_info` ออฟไลน์/โยน exception → export ต้องสำเร็จต่อด้วย `license = null` (ไม่ใช่ raise, ไม่ใช่ 0) — `tests/train/test_export.py::test_save_adapter_only_writes_manifest` (monkeypatch โยน, Task 2)
4. ค่า constant hardcode รั่วเข้า manifest (FT-001) → เทียบกับ `safe_defaults` ตรง ๆ — `tests/contracts/test_manifest.py::test_null_over_zero_and_safe_default_values` (Task 1)
5. contracts ค่อย ๆ import subsystem ตามมา / backend รั่วเข้าสัญญา → subprocess guard ในสภาพแวดล้อมสะอาด — `tests/contracts/test_runtime.py::test_imports_clean_subprocess` (Task 5+6)

---

## File Structure

| ไฟล์ | ความรับผิดชอบ |
|---|---|
| `core/contracts/__init__.py` | ใหม่ — re-export ชื่อสาธารณะของ 3 สัญญา |
| `core/contracts/manifest.py` | ใหม่ — `RunManifest` + nested dataclasses + `build_run_manifest` + `write_run_manifest` |
| `core/contracts/benchmark.py` | ใหม่ — `SCHEMA_VERSION` + `REPORT_KEYS` (23 คีย์) + `validate_report` |
| `core/contracts/runtime.py` | ใหม่ — `LocalRuntime` ABC (4 abstract methods) |
| `core/train/export.py` | แก้ — `save_adapter_only` รับ `config` + ทั้ง 2 โหมดเขียน manifest ตอนจบ + `_model_license` helper |
| `ui/dashboard.py` | แก้ — `on_save_adapter` collect config แล้วส่งต่อ (grep ได้ caller จริง 2 จุด: handler :232-237, wiring :435) |
| `core/compress/report.py` | แก้ — 3 คีย์ใหม่ + validate ก่อนเขียน |
| `tests/contracts/test_manifest.py` | ใหม่ |
| `tests/contracts/test_benchmark_report.py` | ใหม่ |
| `tests/contracts/test_runtime.py` | ใหม่ — signature lock + subprocess import guards |
| `tests/train/test_export.py` | แก้ — config fixture + manifest asserts |
| `tests/ui/test_ui_dashboard.py` | แก้ — lambda mock รับ `config` + handler เรียกแบบ `*_cfg_args()` |
| `tests/compress/test_compression_report.py` | แก้ — ล็อก 23 คีย์ + reject tests |

(13 ไฟล์ = spec §7 "10 ไฟล์" + caller 2 ไฟล์จาก grep ใน §7 บรรทัดสุดท้าย + test report 1 ไฟล์ที่ spec ไม่ได้ระบุแต่ต้องตามสัญญา)

---

### Task 1: RunManifest contract — `core/contracts/manifest.py`

**Files:**
- Create: `core/contracts/__init__.py`, `core/contracts/manifest.py`
- Test: `tests/contracts/test_manifest.py`

**Interfaces:**
- Consumes: `configs.safe_defaults` (`SEED`, `LEARNING_RATE`, `DEFAULT_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `SAVE_STEPS`, `FIM_RATE`, `HELDOUT_RATIO`, `HF_HUB_REVISION`, `LORA_TARGET_MODULES`), stdlib (`dataclasses`, `json`, `os`, `subprocess`, `importlib.metadata`, `datetime`, `pathlib`)
- Produces (Task 2 ใช้):
  - `SCHEMA_VERSION: str = "1.0"`, `ARTIFACT_TYPE: str = "run_manifest"`
  - `@dataclass(frozen=True) class RunManifest` — fields: `schema_version: str`, `artifact_type: str`, `created_at: str`, `base_model: BaseModelInfo`, `dataset: DatasetInfo`, `training: TrainingInfo`, `export: ExportInfo`, `git_commit: str | None`, `tool_versions: dict[str, str | None]`
  - nested: `BaseModelInfo(model_id, revision, num_params: int|None, license: str|None)`, `DatasetInfo(dataset_id, column, code_limit: int, fim_rate: float, heldout_ratio: float, train_samples: int|None, heldout_samples: int|None)`, `TrainingInfo(seed, learning_rate, max_steps, save_steps, batch_size, gradient_accumulation_steps, max_seq_length, lora: LoraInfo)`, `LoraInfo(rank: int, target_modules: list[str])`, `ExportInfo(mode: str, path: str, dtype: str|None)`
  - `RunManifest.to_dict() -> dict` (= `dataclasses.asdict`), `RunManifest.from_dict(data: dict) -> RunManifest` (classmethod, unwrap nested dicts ออกเป็น dataclass)
  - `build_run_manifest(config: dict, *, export_mode: str, export_path: str | Path, num_params: int | None = None, dtype: str | None = None, license: str | None = None) -> RunManifest`
  - `write_run_manifest(manifest: RunManifest, dest_dir: str | Path) -> Path` → คืน `dest_dir / "run_manifest.json"`

- [ ] **Step 1: เขียน failing tests — `tests/contracts/test_manifest.py`**

```python
"""Phase A สัญญาที่ 1: RunManifest — round-trip, null-over-zero, safe_defaults, atomic write"""

from __future__ import annotations

import dataclasses
import json

import pytest

from configs.safe_defaults import (
    DEFAULT_BATCH_SIZE,
    FIM_RATE,
    GRADIENT_ACCUMULATION_STEPS,
    HELDOUT_RATIO,
    HF_HUB_REVISION,
    LEARNING_RATE,
    LORA_TARGET_MODULES,
    SAVE_STEPS,
    SEED,
)
from core.contracts import manifest as bm


def _config(**over) -> dict:
    """config fixture = REQUIRED_CONFIG_KEYS ครบ (core/train/args.py:113)"""
    base = {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": "data_cache/finetune_run",
        "max_seq_length": 1024,
        "max_steps": 6,
        "code_limit": 64,
        "lora_rank": 8,
    }
    base.update(over)
    return base


def test_manifest_roundtrip_and_version():
    m = bm.build_run_manifest(_config(), export_mode="adapter_only",
                              export_path="exports/run1")
    d = m.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["artifact_type"] == "run_manifest"
    assert dataclasses.asdict(m) == d          # to_dict = asdict เป๊ะ
    assert bm.RunManifest.from_dict(d) == m    # round-trip เท่ากันทุกคีย์ (acceptance #1)
    assert d["training"]["lora"]["target_modules"] == list(LORA_TARGET_MODULES)


def test_null_over_zero_and_safe_default_values(monkeypatch):
    """git ดึงไม่ได้ + license ไม่ส่ง → null ทั้งคู่ (acceptance #2); ค่า constant จาก safe_defaults เท่านั้น"""
    def _boom(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(bm.subprocess, "run", _boom)
    m = bm.build_run_manifest(_config(max_steps=500), export_mode="adapter_only",
                              export_path="exports/run1")
    d = m.to_dict()
    # --- null-over-zero ---
    assert d["git_commit"] is None
    assert d["base_model"]["license"] is None
    assert d["base_model"]["num_params"] is None
    assert d["export"]["dtype"] is None
    assert d["dataset"]["train_samples"] is None      # v1 เสมอ null (spec §3.2)
    assert d["dataset"]["heldout_samples"] is None
    # --- safe_defaults  bezpoบ hardcode (invariant #4) ---
    assert d["training"]["seed"] == SEED
    assert d["training"]["learning_rate"] == LEARNING_RATE
    assert d["training"]["batch_size"] == DEFAULT_BATCH_SIZE
    assert d["training"]["gradient_accumulation_steps"] == GRADIENT_ACCUMULATION_STEPS
    assert d["training"]["save_steps"] == SAVE_STEPS          # config ไม่มี key → default
    assert d["training"]["max_steps"] == 500                  # จาก config
    assert d["training"]["max_seq_length"] == 1024
    assert d["training"]["lora"]["rank"] == 8
    assert d["base_model"]["model_id"] == "Qwen/Qwen2.5-Coder-0.5B"
    assert d["base_model"]["revision"] == HF_HUB_REVISION
    assert d["dataset"]["dataset_id"] == "smangrul/hf-stack-v1"
    assert d["dataset"]["column"] == "content"
    assert d["dataset"]["code_limit"] == 64
    assert d["dataset"]["fim_rate"] == FIM_RATE
    assert d["dataset"]["heldout_ratio"] == HELDOUT_RATIO
    assert set(d["tool_versions"]) == {"torch", "transformers", "peft"}


def test_build_manifest_merged_fields():
    """merge mode: num_params/dtype มาจริงจาก caller (acceptance #3 ฝั่ง merge — unit เท่านั้น)"""
    m = bm.build_run_manifest(_config(), export_mode="merged",
                              export_path="exports/run1-merged",
                              num_params=498_431_872, dtype="bfloat16")
    d = m.to_dict()
    assert d["export"] == {"mode": "merged", "path": d["export"]["path"], "dtype": "bfloat16"}
    assert d["base_model"]["num_params"] == 498_431_872


def test_write_run_manifest_atomic_and_parseable(tmp_path):
    m = bm.build_run_manifest(_config(), export_mode="adapter_only",
                              export_path=tmp_path / "dest")
    out = bm.write_run_manifest(m, tmp_path / "dest")
    assert out == tmp_path / "dest" / "run_manifest.json"
    assert json.loads(out.read_text(encoding="utf-8")) == m.to_dict()
    leftovers = [p.name for p in (tmp_path / "dest").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []          # os.replace ทิ้ง tmp ไว้ไม่ได้
```

- [ ] **Step 2: รันให้ FAIL**

Run: `.venv/bin/python -m pytest tests/contracts/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.contracts'`

- [ ] **Step 3: Implement `core/contracts/manifest.py` + `core/contracts/__init__.py`**

`core/contracts/__init__.py`: docstring สั้น + `from core.contracts.manifest import RunManifest, build_run_manifest, write_run_manifest` + `__all__` (Task 3/5 เพิ่มชื่อของตัวเองต่อท้าย)

`manifest.py` — `from __future__ import annotations` + stdlib imports + `from configs.safe_defaults import (...)` แล้ว implement:

- dataclasses ทั้ง 6 ตัวตาม Interfaces block (frozen, `LoraInfo.target_modules: list[str]` — **list ไม่ใช่ tuple** เพื่อให้ JSON round-trip เท่ากันเป๊ะ)
- `to_dict()` = `dataclasses.asdict(self)`; `from_dict()` unwrap nested: `data["base_model"] = BaseModelInfo(**data["base_model"])` ทำเหมือนกันกับ `dataset`, `training` (และ `training["lora"] = LoraInfo(**...)` ก่อน), แล้ว `return cls(**data)`
- `build_run_manifest` ดึงค่าตาม **ตาราง spec §3.2 ทั้งหมด** นอกเหนือจากที่ test ข้างบนระบุ:
  - `created_at = datetime.now(UTC).isoformat()`
  - `git_commit`: `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()` ห่อ `try/except Exception → None`
  - `tool_versions`: loop `("torch", "transformers", "peft")` ผ่าน `importlib.metadata.version(pkg)` ห่อ `try/except Exception → None` ต่อตัว
  - `export.path`: ถ้า `Path(export_path).resolve()` อยู่ใต้ repo root (`Path(__file__).resolve().parents[2]`) → เก็บแบบ relative, ไม่ก็ `str(export_path)` (ValueError → fallback)
  - `export.mode = export_mode`, `num_params`, `dtype`, `license` = argument ที่รับมาตรง ๆ
  - required keys จาก config ใช้ `config["..."]` (KeyError = caller ผิดสัญญา — fail loud)
- `write_run_manifest`: `dest.mkdir(parents=True, exist_ok=True)` → เขียน `dest / "run_manifest.json.tmp"` → `os.replace(tmp, final)` → คืน final

- [ ] **Step 4: รันให้ PASS**

Run: `.venv/bin/python -m pytest tests/contracts/test_manifest.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add core/contracts/ tests/contracts/test_manifest.py
git commit -m "feat(contracts): RunManifest schema 1.0 — dataclass + build/write helpers (Phase A)"
```

---

### Task 2: Export integration — `core/train/export.py` + caller ทั้งหมด

**Files:**
- Modify: `core/train/export.py` (`save_adapter_only` :19-35, `merge_export` :38-64), `ui/dashboard.py` (:232-237 handler, :435 wiring)
- Test: `tests/train/test_export.py`, `tests/ui/test_ui_dashboard.py`

**Interfaces:**
- Consumes (Task 1): `build_run_manifest(config, *, export_mode, export_path, num_params=None, dtype=None, license=None)`, `write_run_manifest(manifest, dest_dir)`
- Produces:
  - `save_adapter_only(output_dir: str | Path, *, config: dict, exports_dir: str | Path = "exports") -> Path` — **`config` เป็น required keyword** (ไม่ default)
  - `merge_export(config: dict) -> Path` — signature ไม่เปลี่ยน, เพิ่มเขียน manifest ก่อน return
  - `core/train/export.py::_model_license(model_id: str) -> str | None` — best-effort `huggingface_hub.model_info` try/except → `None`
  - `ui/dashboard.py::_make_handlers::on_save_adapter(*cfg_values) -> str` — รับ `cfg_inputs` ครบ 11 ช่อง (เหมือน `on_merge`)

- [ ] **Step 1: อัปเดต `tests/train/test_export.py` ให้ fail (ตามสัญญาใหม่) + เพิ่ม test ใหม่**

- เพิ่ม helper `_config()` (คัดจาก `tests/contracts/test_manifest.py::_config` มาวางเต็ม — ไฟล์ test มี helper แยกตามแบบ repo เดิม)
- เรียก `wa.save_adapter_only(...)` ทั้ง 6 จุด (บรรทัด :15, :23, :35, :53, :64, :84) เพิ่ม `config=_config()`
- ขยาย `test_save_adapter_only_config_without_weights_raises` — เพิ่ม assert หลัง `pytest.raises`:

```python
    assert not (tmp_path / "exports" / "run" / "run_manifest.json").exists()  # fail ไม่ค้าง (acceptance #4)
```

- เพิ่ม test ใหม่ (import `json` + `from configs.safe_defaults import SEED` บนหัวไฟล์):

```python
def test_save_adapter_only_writes_manifest(tmp_path, monkeypatch):
    """acceptance #3: manifest เขียนที่ dest สำเร็จ + ออฟไลน์ → license null, ไม่แตะเน็ตจริง"""
    out = tmp_path / "run1"
    ckpt = out / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"fake")

    def _offline(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(wa, "model_info", _offline)   # hub ออฟไลน์
    dest = wa.save_adapter_only(out, config=_config(), exports_dir=tmp_path / "exports")
    m = json.loads((dest / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["schema_version"] == "1.0"
    assert m["artifact_type"] == "run_manifest"
    assert m["export"]["mode"] == "adapter_only"
    assert m["export"]["path"].endswith("run1")
    assert m["base_model"]["license"] is None         # null ไม่ใช่ 0/เดา (acceptance #2)
    assert m["base_model"]["num_params"] is None
    assert m["export"]["dtype"] is None
    assert m["training"]["seed"] == SEED
```

- **รันให้ FAIL:** `.venv/bin/python -m pytest tests/train/test_export.py -v`
  Expected: FAIL — `TypeError: save_adapter_only() got an unexpected keyword argument 'config'` (ทั้ง 6 เดิม + test ใหม่)

- [ ] **Step 2: Implement ใน `core/train/export.py`**

- เพิ่ม imports: `from huggingface_hub import model_info`, `from core.contracts.manifest import build_run_manifest, write_run_manifest`
- เพิ่ม helper:

```python
def _model_license(model_id: str) -> str | None:
    """best-effort ครั้งเดียวตอน export — ออฟไลน์/hub ล่ม/cardData ไม่มี = None (null-over-zero)"""
```

  body: `try: return model_info(model_id).cardData.get("license") except Exception: return None` (cardData เป็น `None` → `.get` โยน AttributeError → โดน catch → `None` ✓)
- `save_adapter_only`: เปลี่ยน signature เป็น `(output_dir, *, config: dict, exports_dir="exports")` — หลัง loop copy สำเร็จ (**ก่อน `return dest`**, เป็น step สุดท้าย):

```python
    manifest = build_run_manifest(
        config, export_mode="adapter_only", export_path=dest,
        license=_model_license(config["model_id"]),
    )
    write_run_manifest(manifest, dest)
    return dest
```

- `merge_export`: ก่อน `return dest` (หลัง `tokenizer.save_pretrained(dest)`) — เรียกเหมือนกันแต่ `export_mode="merged"`, `num_params=merged.num_parameters()`, `dtype="bfloat16"` (ค่าจริงที่โหลดด้วย line :53)

- [ ] **Step 3: รันฝั่ง train ให้ PASS**

Run: `.venv/bin/python -m pytest tests/train/test_export.py -v`
Expected: 8 passed (7 เดิม + 1 ใหม่)

- [ ] **Step 4: อัปเดต caller UI — ให้ `tests/ui/test_ui_dashboard.py` fail ก่อน**

- `test_on_save_adapter_escapes_dest_and_error` (:555-568): เปลี่ยน lambda ทั้งสองเป็น `lambda _p, config: ...` และเรียก `h["on_save_adapter"](*_cfg_args())` แทน `("out")`
- **รันให้ FAIL:** `.venv/bin/python -m pytest tests/ui/test_ui_dashboard.py::test_on_save_adapter_escapes_dest_and_error -v`
  Expected: FAIL — `TypeError: <lambda>() missing 1 required positional argument: 'config'`

- [ ] **Step 5: แก้ `ui/dashboard.py` 2 จุด**

- handler `on_save_adapter(output_dir)` → `on_save_adapter(*cfg_values)`:

```python
    def on_save_adapter(*cfg_values):
        try:
            config, _user_params = _collect_config(*cfg_values)
        except ValueError as exc:
            return f'<span style="color:#dc2626">**Save failed:** {html.escape(str(exc))}</span>'
        try:
            dest = save_adapter_only(config["output_dir"], config=config)
            return f'<span style="color:#16a34a">✅ Adapter saved → `{html.escape(str(dest))}`</span>'
        except Exception as exc:
            return f'<span style="color:#dc2626">**Save failed:** {html.escape(str(exc))}</span>'
```

  (pattern เดียวกับ `on_merge` :239-251 — error ต้อง escape H2 เหมือนเดิม)
- wiring :435: `save_btn.click(h["on_save_adapter"], inputs=cfg_inputs, outputs=[export_msg])`

- [ ] **Step 6: รันทั้งสอง test file ให้ PASS**

Run: `.venv/bin/python -m pytest tests/train/test_export.py tests/ui/test_ui_dashboard.py -v`
Expected: ทั้งคู่ PASS ทั้งหมด

- [ ] **Step 7: Commit**

```bash
git add core/train/export.py ui/dashboard.py tests/train/test_export.py tests/ui/test_ui_dashboard.py
git commit -m "feat(train): เขียน run_manifest.json ตอน export ทั้ง 2 โหมด + save_adapter_only รับ config (FT-010)"
```

---

### Task 3: Benchmark Report Schema v1.0 — `core/contracts/benchmark.py`

**Files:**
- Create: `core/contracts/benchmark.py`
- Modify: `core/contracts/__init__.py` (เพิ่ม export)
- Test: `tests/contracts/test_benchmark_report.py`

**Interfaces:**
- Consumes: stdlib เท่านั้น
- Produces (Task 4 ใช้):
  - `SCHEMA_VERSION: str = "1.0"`
  - `REPORT_KEYS: frozenset[str]` = **23 คีย์** (20 เดิม + 3 ใหม่):

```python
REPORT_KEYS: frozenset[str] = frozenset({
    "model", "variant", "optimization", "backend", "weight_bits",
    "context_tokens", "parameter_count", "model_disk_mb",
    "peak_vram_mb", "kv_cache_mb",
    "tokens_per_sec", "prompt_tokens_per_sec", "latency_ms", "load_time_ms",
    "exact_match_pct", "token_f1", "syntax_pass_rate", "execution_pass_rate",
    "eval", "timestamp",
    "schema_version", "peak_rss_mb", "environment",
})
```

  - `validate_report(data: dict) -> None` — ขาด/เกินคีย์ → `ValueError` (ข้อความบอกทั้ง missing และ extra); `data["schema_version"] != SCHEMA_VERSION` → `ValueError`; ไม่ตรวจชนิด/ช่วงค่า metrics (null-over-zero เป็นหน้าที่ผู้สร้าง)

- [ ] **Step 1: เขียน failing test — `tests/contracts/test_benchmark_report.py`**

```python
"""Phase A สัญญาที่ 2: Benchmark Report Schema v1.0 — 23 คีย์ flat + validator"""

from __future__ import annotations

import pytest

from core.contracts import benchmark as cb


def _full_report() -> dict:
    return {k: None for k in cb.REPORT_KEYS} | {"schema_version": cb.SCHEMA_VERSION}


def test_report_keys_pinned():
    """23 คีย์ flat — แก้ = แก้ test+spec (schema v2 ค่อยมาจับกลุ่ม Sprint 3)"""
    assert len(cb.REPORT_KEYS) == 23
    assert {"schema_version", "peak_rss_mb", "environment"} <= cb.REPORT_KEYS
    assert cb.SCHEMA_VERSION == "1.0"


def test_validate_accepts_full_report():
    cb.validate_report(_full_report())          # ไม่ raise


def test_validate_rejects_missing_key():
    data = _full_report()
    data.pop("token_f1")
    with pytest.raises(ValueError, match="token_f1"):
        cb.validate_report(data)


def test_validate_rejects_extra_key():
    data = _full_report()
    data["legacy_field"] = 1
    with pytest.raises(ValueError, match="legacy_field"):
        cb.validate_report(data)


def test_validate_rejects_stale_schema_version():
    data = _full_report()
    data["schema_version"] = "0.9"              # report เก่าไหลเข้าระบบใหม่ = ต้องบล็อก
    with pytest.raises(ValueError, match="schema_version"):
        cb.validate_report(data)
```

- `core/contracts/__init__.py` เพิ่ม: `from core.contracts.benchmark import REPORT_KEYS, validate_report` + ใส่ `__all__`

- [ ] **Step 2: รันให้ FAIL**

Run: `.venv/bin/python -m pytest tests/contracts/test_benchmark_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.contracts.benchmark'`

- [ ] **Step 3: Implement `core/contracts/benchmark.py`**

ตาม Interfaces — `validate_report` สองขั้นตอน (key set เป๊ะก่อน แล้วค่อยเวอร์ชัน) เก็บ message เป็น str ที่มีชื่อคีย์/เวอร์ชันจริง (test match ด้วย)

- [ ] **Step 4: รันให้ PASS**

Run: `.venv/bin/python -m pytest tests/contracts/test_benchmark_report.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add core/contracts/benchmark.py core/contracts/__init__.py tests/contracts/test_benchmark_report.py
git commit -m "feat(contracts): Benchmark Report Schema v1.0 — 23 flat keys + validate_report (BM-001)"
```

---

### Task 4: Report integration — `core/compress/report.py`

**Files:**
- Modify: `core/compress/report.py` (`build_report` :38-59, `write_report` :62-67)
- Test: `tests/compress/test_compression_report.py`

**Interfaces:**
- Consumes (Task 3): `SCHEMA_VERSION`, `validate_report`
- Produces: `build_report()` คืน dict ครบ `REPORT_KEYS`; `write_report(data, path)` validate ก่อน mkdir/write

- [ ] **Step 1: แก้ test ให้ fail ก่อน — `tests/compress/test_compression_report.py`**

- เพิ่ม imports: `import pytest`, `from core.contracts.benchmark import REPORT_KEYS, SCHEMA_VERSION`
- `test_build_report_schema_keys` — เปลี่ยน `assert expected <= set(r)` เป็น `assert set(r) == REPORT_KEYS` (ล็อกเป๊ะ 23 คีย์ แทน subset) + เพิ่ม:

```python
    assert r["schema_version"] == "1.0"
    assert r["peak_rss_mb"] is None   # BM-006 ยังไม่เขียน — null ก่อน (null-over-zero)
    assert r["environment"] is None   # BM-010 ยังไม่เก็บ
```

  (ชุด `expected` ใน test เดิมลบออกได้เลยเมื่อเทียบกับ `REPORT_KEYS` — การแทน subset ด้วย equality = แข็งขึ้น ไม่ใช่ลดเกณฑ์)
- เพิ่ม test ใหม่:

```python
def test_write_report_rejects_invalid_and_stale_schema(tmp_path):
    """gate ที่ขอบระบบ — report พัง = ไม่เขียนไฟล์เลย"""
    bad = report.build_report(**_kwargs())
    bad.pop("kv_cache_mb")
    with pytest.raises(ValueError, match="kv_cache_mb"):
        report.write_report(bad, tmp_path / "a.json")
    assert not (tmp_path / "a.json").exists()

    stale = report.build_report(**_kwargs())
    stale["schema_version"] = "0.9"
    with pytest.raises(ValueError, match="schema_version"):
        report.write_report(stale, tmp_path / "b.json")
    assert not (tmp_path / "b.json").exists()
```

- **รันให้ FAIL:** `.venv/bin/python -m pytest tests/compress/test_compression_report.py -v`
  Expected: FAIL — `assert {'model', ...} (20 keys) == REPORT_KEYS (23 keys)` + `KeyError: 'schema_version'`

- [ ] **Step 2: Implement ใน `core/compress/report.py`**

- เพิ่ม `from core.contracts.benchmark import SCHEMA_VERSION, validate_report` (ข้างใต้ import `configs.safe_defaults` เดิม)
- `build_report` return dict เพิ่ม 3 คีย์ (คง flat structure เดิมทุกอย่าง):

```python
        "schema_version": SCHEMA_VERSION,
        "peak_rss_mb": None,   # BM-006: CLI เติมค่าจริง Sprint 3 — วัดไม่ได้ = null
        "environment": None,   # BM-010: {hardware, os, llama_cpp_commit} — เก็บจริง Sprint 3
```

- `write_report`: เรียก `validate_report(data)` **บรรทัดแรก** ก่อน mkdir/path.write_text

- [ ] **Step 3: รัน test ของ report + regression ของการ consume report**

Run: `.venv/bin/python -m pytest tests/compress/ -v`
Expected: PASS ทั้งโฟลเดอร์ — โดยเฉพาะ `test_build_report_schema_keys`, `test_write_report_roundtrip` (validate ผ่านเพราะ build_report คืนครบ), `test_compression_cli.py` ทั้งไฟล์ (`recording_write` เรียก `write_report` จริงด้วย output ของ `build_report` → ผ่าน; `_row()` hand-built dict ไม่ได้ผ่าน `write_report` → ไม่กระทบ)

- [ ] **Step 4: Commit**

```bash
git add core/compress/report.py tests/compress/test_compression_report.py
git commit -m "feat(compress): report v1.0 — schema_version/peak_rss_mb/environment + validate ก่อนเขียน (BM-001)"
```

---

### Task 5: LocalRuntime ABC — `core/contracts/runtime.py`

**Files:**
- Create: `core/contracts/runtime.py`
- Modify: `core/contracts/__init__.py` (เพิ่ม export)
- Test: `tests/contracts/test_runtime.py`

**Interfaces:**
- Consumes: stdlib (`abc`, `inspect`, `pathlib`, `subprocess`, `sys`)
- Produces: `LocalRuntime(ABC)` — 4 abstract methods `start`, `is_ready`, `complete`, `stop` (ไม่มี adapter จนถึง Phase D)

- [ ] **Step 1: เขียน failing test — `tests/contracts/test_runtime.py`**

```python
"""Phase A สัญญาที่ 3: LocalRuntime ABC — signature lock + ห้ามผูก backend"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from core.contracts import runtime as rt

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cannot_instantiate():
    with pytest.raises(TypeError):
        rt.LocalRuntime()          # abstract — instantiate ไม่ได้


def test_start_signature_locked():
    sig = inspect.signature(rt.LocalRuntime.start)
    assert list(sig.parameters) == ["self", "model_path", "device", "context_tokens"]
    assert sig.parameters["device"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["device"].default == "vulkan"
    assert sig.parameters["context_tokens"].default is None   # None = ปล่อย backend ใช้ค่า native (RM-008)


def test_is_ready_signature_locked():
    sig = inspect.signature(rt.LocalRuntime.is_ready)
    assert list(sig.parameters) == ["self", "timeout_s"]
    assert sig.parameters["timeout_s"].default is inspect.Parameter.empty  # required positional


def test_complete_signature_locked():
    sig = inspect.signature(rt.LocalRuntime.complete)
    assert list(sig.parameters) == ["self", "prompt", "max_tokens", "temperature", "stop"]
    assert sig.parameters["max_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["max_tokens"].default is inspect.Parameter.empty
    assert sig.parameters["temperature"].default is None
    assert sig.parameters["stop"].default is None


def test_stop_signature_locked():
    assert list(inspect.signature(rt.LocalRuntime.stop).parameters) == ["self"]


def test_imports_clean_subprocess():
    """สภาพแวดล้อมสะอาด — sys.modules ใน pytest process มี test ตัวอื่นปนอยู่แล้ว จึงต้องแยก subprocess
    (guard รวม: ทั้ง 3 สัญญา ห้ามดึง subsystem/backend เข้ามา — Global constraint 2)"""
    code = (
        "import sys; "
        "import core.contracts.manifest, core.contracts.benchmark, core.contracts.runtime; "
        "banned = ('core.train', 'core.compress', 'core.runtime', "
        "'core.compress.llama_runner', 'llama_cpp'); "
        "bad = [m for m in banned if any(x == m or x.startswith(m + '.') for x in sys.modules)]; "
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, f"contracts pulled in: {res.stdout} {res.stderr}"
```

- `core/contracts/__init__.py` เพิ่ม `from core.contracts.runtime import LocalRuntime` + `__all__`

- [ ] **Step 2: รันให้ FAIL**

Run: `.venv/bin/python -m pytest tests/contracts/test_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.contracts.runtime'`

- [ ] **Step 3: Implement `core/contracts/runtime.py`**

`from __future__ import annotations` + `from abc import ABC, abstractmethod` + `from pathlib import Path` — class `LocalRuntime(ABC)` 4 methods ตรง signature test ทุกตัว พร้อม docstring contract จาก spec §5.1 (คัดข้อความสำคัญ: start = โหลด GGUF เปิด endpoint, `context_tokens=None` = backend native; is_ready = poll จน timeout ต้องรวม identity check; complete = raw prompt semantics BM-008 ห้าม strip/insert เงียบ, max_tokens → n_predict หน้าที่ adapter; stop = SIGINT → grace → SIGKILL ตาม §4.2, idempotent) — ห้าม import อะไรนอกจาก stdlib

- [ ] **Step 4: รันให้ PASS**

Run: `.venv/bin/python -m pytest tests/contracts/test_runtime.py -v`
Expected: 6 passed (รวม subprocess guard — ตรวจด้วยว่า `res.stdout` ของ assert ไม่โชว์ banned modules)

- [ ] **Step 5: Commit**

```bash
git add core/contracts/runtime.py core/contracts/__init__.py tests/contracts/test_runtime.py
git commit -m "feat(contracts): LocalRuntime ABC — 4 methods + signature lock ไม่มี adapter (Phase D)"
```

---

### Task 6: ตรวจครบ + full regression + package exports ครบ

**Files:**
- Modify: `core/contracts/__init__.py` (ตรวจว่า export ครบ 3 สัญญาแล้ว)
- Test: `tests/contracts/test_manifest.py` (เพิ่ม guard test สุดท้าย)

**Interfaces:**
- Consumes: ทุก Task ก่อนหน้า
- Produces: หลักฐาน acceptance #7 + #8

- [ ] **Step 1: เพิ่ม test สุดท้ายใน `tests/contracts/test_manifest.py` — top-level package export**

```python
def test_package_exports_all_contracts():
    """core.contracts (จุดเดียว) ต้อง expose สัญญาครบทั้ง 3"""
    import core.contracts as c

    for name in ("RunManifest", "build_run_manifest", "write_run_manifest",
                 "REPORT_KEYS", "validate_report", "LocalRuntime"):
        assert hasattr(c, name), name
```

- ตรวจ `core/contracts/__init__.py` ด้วยตา — ถ้าชื่อไหนใน list ขาด เติม import + `__all__`

- [ ] **Step 2: รัน test ใหม่ทั้งหมด**

Run: `.venv/bin/python -m pytest tests/contracts/ -v`
Expected: ทั้ง 3 ไฟล์ PASS (16 tests: manifest 5 + benchmark 5 + runtime 6)

- [ ] **Step 3: Full regression — acceptance #8**

Run: `.venv/bin/python -m pytest tests/`
Expected: **ชุดเดิม 371 + ใหม่ทั้งหมดเขียว 100% — failed 0, skipped 0, deselected คงเดิม (8)** — ห้ามแก้ expectations เก่าเพิ่ม นอกจากส่วนที่ follow สัญญาใหม่ในแต่ละ task

- [ ] **Step 4: ตรวจ dependency guard ด้วย grep (belt + suspenders)**

Run: `grep -rn "core\.train\|core\.compress\|core\.runtime" core/contracts/`
Expected: ว่างเปล่า — สัญญาไม่อ้าง subsystem ตัวไหน (stdlib + `configs.safe_defaults` เท่านั้น)

- [ ] **Step 5: Commit**

```bash
git add core/contracts/__init__.py tests/contracts/test_manifest.py
git commit -m "test(contracts): package export guard + full Phase A regression 371+ green"
```
