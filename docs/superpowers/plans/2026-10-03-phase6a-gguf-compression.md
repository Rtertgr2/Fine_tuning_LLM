# Phase 6a — GGUF Compression Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** สร้าง pipeline แปลงโมเดล HF → GGUF (fp16/q8_0/q4_k_m) + bench + eval เทียบ Phase 5 + serve URL บน llama.cpp ผ่าน CLI

**Architecture:** standalone ใต้ `core/compression/` (config → quantizer → benchmark → llama_runner → llama_eval → report) + 3 scripts (`setup_llamacpp.sh`, `benchmark_compression.py`, `serve.py`) — ไม่แตะ Gradio app; subprocess ทุกตัว mock ได้ใน unit test; `llama_runner` มีวงจรชีวิตเดียวทั้ง eval และ serve

**Tech Stack:** llama.cpp (Vulkan, pin commit), Python stdlib (urllib/socket/signal), transformers tokenizer (eval cases เดิม), pytest + `@pytest.mark.integration`

**Spec:** `docs/superpowers/specs/2026-10-03-phase6a-gguf-compression-design.md`

## Global Constraints

- Error/UI copy = **English เสมอ**; README/docs = **ภาษาไทย**
- Python = venv 3.11/3.12 เท่านั้น — รันผ่าน `.venv/bin/python` (main) หรือ `../../.venv/bin/python` (worktree)
- ruff: `uvx ruff check core/ ui/ configs/ tests/` ห้ามแย่กว่า master (baseline = 5 pre-existing errors); `ruff format` ไม่บังคับ
- ห้าม commit: `tools/`, `models/`, `datasets/`, `exports/`, `data_cache/`, `*.gguf` — เพิ่ม `/tools/` ใน `.gitignore` (Task 1)
- Report honesty: วัดไม่ได้ = `null` **ห้ามเดา**; ห้ามตั้ง quality threshold (report-only)
- Exception ใหม่ใช้ `CompressionError` (message English เสมอ)
- Integration test = `@pytest.mark.integration` (pytest.ini deselect ให้อัตโนมัติ)
- Eval ต้อง reuse ตัวเดิม: `build_eval_cases` (seed=`SEED`, n_cases=100), `exact_match`, `token_f1`, `build_fim_prompt` (`core/trainer_worker.py:381`), `EVAL_MAX_NEW_TOKENS=256`
- llama.cpp: pin commit ใน setup script (ไม่ใช้ branch); unit test ห้ามเรียก binary จริง

## Review Focus

5 อย่างที่ spec บอกว่าเสี่ยงแต่ไม่มี task ไหนครอบอัตโนมัติ — มี test ของตัวเองใน task เจ้าของ:

1. **FIM prompt ต้องถูกส่งออกไป verbatim** (ไม่ re-encode เอง) — ถ้าเพี้ยน eval ทั้งชุดเสีย → Task 6: `test_eval_prompt_verbatim_equals_build_fim_prompt`
2. **Port ถูกใช้ / มี server อื่นค้างอยู่** — ต้อง error อังกฤษชี้ `--port` ไม่ใช่คิดว่าสำเร็จ → Task 5: `test_start_reports_port_busy` + `test_start_rejects_foreign_model`
3. **Ctrl+C ห้าม leak child process** (VRAM ค้าง) → Task 5: `test_stop_server_sends_sigint_then_wait`, Task 8: `test_serve_stops_child_on_keyboard_interrupt`
4. **xpu-smi ไม่มี / parse ไม่ได้ → `null`** (ห้าม 0 ปลอม) → Task 4: `test_sample_memory_vram_none_when_xpu_smi_missing`
5. **โมเดลเป็น Hub id หรือโฟลเดอร์ไม่ใช่ HF layout** → English + คำสั่ง `hf download` → Task 1: `test_resolve_source_hub_id_error`, `test_resolve_source_dir_without_config_json_error`

---

## File Structure

| ไฟล์ | ความรับผิดชอบ |
|---|---|
| Create `core/compression/__init__.py` | `CompressionError` |
| Create `core/compression/config.py` | paths/variants/device + `resolve_source` |
| Create `core/compression/quantizer.py` | convert → quantize (subprocess) |
| Create `core/compression/benchmark.py` | llama-bench parse + memory sample |
| Create `core/compression/llama_runner.py` | llama-server lifecycle + HTTP |
| Create `core/compression/llama_eval.py` | eval loop ผ่าน server |
| Create `core/compression/report.py` | §4 schema + table + baseline pick |
| Create `scripts/setup_llamacpp.sh` | clone pin + cmake Vulkan build |
| Create `scripts/benchmark_compression.py` | orchestration CLI |
| Create `scripts/serve.py` | serve wrapper |
| Create `tests/test_compression_{config,quantizer,benchmark,runner,llama_eval,report,cli,serve}.py` | 1:1 กับ module |
| Modify `.gitignore` | เพิ่ม `/tools/` |
| Modify `README.md` | ภาค compression (ภาษาไทย) |

---

### Task 1: config + setup script

**Files:**
- Create: `core/compression/__init__.py`, `core/compression/config.py`, `scripts/setup_llamacpp.sh`, `tests/test_compression_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `core.estimator.resolve_local_model(model_id) -> Path | None`
- Produces (task อื่นใช้เหล่านี้):
  - `CompressionError(RuntimeError)` — นิยามใน `core/compression/__init__.py`, **re-export ใน `config`** (test เข้าถึงผ่าน `config.CompressionError`)
  - `DEFAULT_VARIANTS: tuple[str, ...] = ("fp16", "q8_0", "q4_k_m")`
  - `QUANT_TYPE: dict[str, str] = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}`
  - `VARIANT_BITS: dict[str, int] = {"fp16": 16, "q8_0": 8, "q4_k_m": 4}`
  - `tools_dir() -> Path` (env `LLAMA_CPP_DIR` → default `tools/llama.cpp`)
  - `tool_path(name: str) -> Path` (= `tools_dir()/"build"/"bin"/name`)
  - `convert_script() -> Path` (= `tools_dir()/"convert_hf_to_gguf.py"`)
  - `require_tools() -> None` — ขาด → `CompressionError("llama.cpp not found. Run scripts/setup_llamacpp.sh first.")`
  - `check_vulkan() -> bool`
  - `resolve_source(model_id: str) -> Path` — local dir/models name ต้องมี `config.json`; Hub id → error + `hf download ... --local-dir models/<name>`
  - `device_args(device: str) -> list[str]` — `"vulkan"` → `[]`, `"cpu"` → non-empty
  - `gguf_dir(source: Path) -> Path` / `artifact_path(source: Path, variant: str) -> Path` / `report_path(source: Path, variant: str) -> Path`

- [ ] **Step 1: เขียน failing tests**

`tests/test_compression_config.py` — อย่างน้อย:
```python
def test_tools_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert config.tools_dir() == tmp_path

def test_tools_dir_default(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    assert config.tools_dir() == Path("tools/llama.cpp")

def test_require_tools_missing_raises_english(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "tools_dir", lambda: tmp_path)  # ว่างเปล่า
    with pytest.raises(config.CompressionError) as exc:
        config.require_tools()
    assert "setup_llamacpp.sh" in str(exc.value)

def test_resolve_source_hub_id_error():
    with pytest.raises(config.CompressionError) as exc:
        config.resolve_source("acme/some-model")
    msg = str(exc.value)
    assert "hf download acme/some-model" in msg and "--local-dir models/" in msg

def test_resolve_source_dir_without_config_json_error(tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(config.CompressionError) as exc:
        config.resolve_source(str(tmp_path / "bare"))
    assert "config.json" in str(exc.value)

def test_resolve_source_models_name(monkeypatch, tmp_path):
    root = tmp_path / "models"; (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "MODELS_DIR_STR", str(root))  # หรือ patch estimator.MODELS_DIR
    assert config.resolve_source("alpha") == root / "alpha"

def test_artifact_paths_layout(tmp_path):
    src = tmp_path / "m"
    assert config.gguf_dir(src) == src / "gguf"
    assert config.artifact_path(src, "q4_k_m") == src / "gguf" / "m-q4_k_m.gguf"
    assert config.report_path(src, "q8_0") == Path("benchmarks/m/q8_0.json")

def test_device_args_vulkan_is_empty_and_cpu_is_not():
    assert config.device_args("vulkan") == []
    assert config.device_args("cpu") != []

def test_check_vulkan_real():  # hardware — skipif ไม่มี binary
    assert isinstance(config.check_vulkan(), bool)
```

- [ ] **Step 2: รันให้ fail** — `../../.venv/bin/python -m pytest tests/test_compression_config.py -q` → FAIL (module ยังไม่มี)

- [ ] **Step 3: implement** — `core/compression/__init__.py` (CompressionError), `config.py` (ค่าคงที่ + ฟังก์ชันด้านบน; `resolve_source` ใช้ `estimator.resolve_local_model` แล้วตรวจ `config.json`; `check_vulkan` = `shutil.which("vulkaninfo")` + `subprocess.run(..., timeout=10)` rc==0; `device_args` flag จริงยืนยันด้วย `--help` ของ bin หลัง build), `.gitignore` เพิ่ม `/tools/`

- [ ] **Step 4: เขียน `scripts/setup_llamacpp.sh`** — `set -euo pipefail`; เช็ค `vulkaninfo` (ไม่มี → error อังกฤษบอกติดตั้ง `vulkan-intel`); `git clone https://github.com/ggml-org/llama.cpp tools/llama.cpp` + `git -C ... checkout <commit>` (pin = HEAD ที่ stable ณ ตอนรัน — จด rev จริงลงใน script); `cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`; จบด้วย echo รายการ binary ที่คาด

- [ ] **Step 5: รัน tests ให้เขียว** — `...pytest tests/test_compression_config.py -q` → PASS ทั้งหมด

- [ ] **Step 6: Build จริง (DoD ข้อ 1)** — `bash scripts/setup_llamacpp.sh` (ใช้เวลาหลายนาที) แล้วตรวจ: `test -x tools/llama.cpp/build/bin/llama-server && test -x tools/llama.cpp/build/bin/llama-quantize && test -f tools/llama.cpp/convert_hf_to_gguf.py` → ผ่านทั้ง 3

- [ ] **Step 7: Commit** — `git add core/compression scripts/setup_llamacpp.sh tests/test_compression_config.py .gitignore && git commit -m "feat(compression): config, tool paths, setup script for llama.cpp (Vulkan)"`

---

### Task 2: quantizer (HF → fp16 → quantize)

**Files:**
- Create: `core/compression/quantizer.py`, `tests/test_compression_quantizer.py`

**Interfaces:**
- Consumes: Task 1 (`CompressionError`, `QUANT_TYPE`, `convert_script`, `tool_path`, `artifact_path`, `require_tools`)
- Produces:
  - `convert_to_fp16(source: Path) -> Path` — subprocess `[sys.executable, convert_script, source, --outfile, <gguf_dir>/<name>-fp16.gguf, --outtype, f16]`; fail → `CompressionError(f"convert_hf_to_gguf failed for {source}: {log_tail}")`
  - `quantize_fp16(fp16: Path, variant: str) -> Path` — subprocess `[tool_path("llama-quantize"), fp16, out, QUANT_TYPE[variant]]`
  - `build_artifacts(source: Path, variants: Sequence[str] = DEFAULT_VARIANTS) -> list[Path]` — fp16 ก่อน (convert ครั้งเดียว) แล้ว quantize ตามลำดับ; **ไฟล์มีอยู่แล้ว → reuse ข้าม**; 未知 variant → `CompressionError`

- [ ] **Step 1: เขียน failing tests** (`tests/test_compression_quantizer.py`) — mock `subprocess.run` ผ่าน `monkeypatch.setattr(quantizer, "run_cmd", ...)` (ห่อ subprocess ไว้ใน `run_cmd(cmd: list[str]) -> str` เพื่อ mock):
```python
def test_build_artifacts_command_order_and_mapping(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    def fake_run(cmd): calls.append(cmd); return str(tmp_path / "out.gguf")
    monkeypatch.setattr(quantizer, "run_cmd", fake_run)
    src = tmp_path / "m"; src.mkdir(); (src / "config.json").write_text("{}")
    out = quantizer.build_artifacts(src, ("fp16", "q8_0", "q4_k_m"))
    assert len(calls) == 3                      # convert 1 + quantize 2
    assert "--outtype" in calls[0] and "f16" in calls[0]
    assert calls[1][-1] == "Q8_0" and calls[2][-1] == "Q4_K_M"
    assert out[0].name == "m-fp16.gguf"

def test_build_artifacts_reuses_existing_fp16(monkeypatch, tmp_path):  # ไม่ convert ซ้ำ
def test_convert_failure_raises_english_with_log_tail(monkeypatch, tmp_path):  # run_cmd raise CalledProcessError(stderr="boom") → msg มี "convert_hf_to_gguf" และ "boom"
def test_unknown_variant_error():                # variant "q3" → CompressionError "unknown variant"
def test_source_without_config_json_error(tmp_path):  # โฟลเดอร์เปล่า → "config.json"
```

- [ ] **Step 2: รันให้ fail** → FAIL (module ยังไม่มี)
- [ ] **Step 3: implement** — `run_cmd(cmd) -> str` = `subprocess.run(cmd, capture_output=True, text=True, check=True)`; log tail = 15 บรรทัดสุดท้ายของ stderr/stdout; สร้าง `gguf_dir` ด้วย `mkdir(parents=True, exist_ok=True)`
- [ ] **Step 4: รันให้เขียว**
- [ ] **Step 5: Integration test** — `@pytest.mark.integration`:
```python
def test_build_artifacts_real_qwen_smoke():
    # skipif: ไม่มี tools หรือไม่มี models/Qwen.../config.json → pytest.skip
    # build_artifacts(qwen_dir, ("fp16", "q4_k_m")) → fp16 exists, q4 exists, q4 size < fp16 size
```
รันจริง: `...pytest tests/test_compression_quantizer.py -q -m integration` → PASS
- [ ] **Step 6: Commit** — `feat(compression): HF→GGUF convert + quantize pipeline`

---

### Task 3: report (§4 schema + table + baseline)

**Files:**
- Create: `core/compression/report.py`, `tests/test_compression_report.py`

**Interfaces:**
- Consumes: `configs.safe_defaults.DEFAULT_MODEL_ID`
- Produces:
  - `build_report(*, model: str, variant: str, backend: str, weight_bits: int, parameter_count: int | None, context_tokens: int, model_disk_mb: float, tokens_per_sec: float | None, latency_ms: float | None, load_time_ms: float | None, peak_vram_mb: float | None, kv_cache_mb: float | None, exact_match_pct: float | None, token_f1: float | None) -> dict` — คง field `optimization="quantization"`, **`syntax_pass_rate=None`, `execution_pass_rate=None`**, `timestamp` = UTC ISO
  - `write_report(data: dict, path: Path) -> Path` (mkdir parents, json ensure_ascii=False, indent=2)
  - `pick_baseline(model_id: str, source: Path) -> Path | None` — source อยู่ใต้ `exports/` → `data_cache/eval/finetuned.json`; `model_id == DEFAULT_MODEL_ID` → `data_cache/eval/base.json`; ไม่มีไฟล์/อื่น → `None`
  - `format_table(rows: list[dict], baseline: dict | None = None) -> str` — คอลัมน์ `Variant / Size / TPS / EM / F1` + แถว Δ ใต้ EM/F1 เมื่อ baseline มี

- [ ] **Step 1: failing tests** (`tests/test_compression_report.py`):
```python
def test_build_report_honest_nulls():            # syntax/execution/peak_vram(ไม่ส่ง) = None — ไม่ใช่ 0
def test_build_report_schema_keys():             # keys ครบตาม §4: model, variant, optimization, backend, weight_bits, context_tokens, parameter_count, model_disk_mb, peak_vram_mb, kv_cache_mb, tokens_per_sec, latency_ms, exact_match_pct, token_f1, syntax_pass_rate, execution_pass_rate, timestamp
def test_write_report_roundtrip(tmp_path):        # เขียนแล้ว json.load ได้ ครบ
def test_pick_baseline_exports(tmp_path):         # src=tmp/exports/x-merged แต่ง srcใต้ exports → finetuned.json (สร้างไฟล์ stub)
def test_pick_baseline_default_model(tmp_path):   # model_id=DEFAULT → base.json (stub)
def test_pick_baseline_unknown_returns_none(tmp_path)
def test_format_table_lists_variants_and_delta(): # rows 2 ตัว + baseline → มี "Q4_K_M" และ "-"
```
- [ ] **Step 2: fail** → **Step 3: implement** → **Step 4: pass**
- [ ] **Step 5: Commit** — `feat(compression): benchmark report schema + terminal table`

---

### Task 4: benchmark (llama-bench parse + memory)

**Files:**
- Create: `core/compression/benchmark.py`, `tests/test_compression_benchmark.py`

**Interfaces:**
- Consumes: Task 1 (`tool_path`, `device_args`, `require_tools`, `CompressionError`)
- Produces:
  - `parse_bench_output(stdout: str) -> dict` → `{"prompt_tps": float | None, "gen_tps": float | None, "load_ms": float | None}` — ทนต่อ key หาย (→ `None`)
  - `sample_memory(pid: int) -> dict` → `{"peak_rss_mb": float | None, "peak_vram_mb": float | None}`
  - `run_bench(gguf: Path, *, device: str = "vulkan") -> dict` — Popen `llama-bench -m <gguf> -o json` + device args; poll `sample_memory(proc.pid)` ทุก 100ms จนจบ (เก็บ max) → merge เป็น dict เดียว; rc!=0 → `CompressionError` + log tail

- [ ] **Step 1: capture fixture จริง** — รันครั้งเดียวด้วย artifact จาก Task 2: `tools/llama.cpp/build/bin/llama-bench -m models/Qwen/Qwen2.5-Coder-0.5B/gguf/Qwen-Q4_K_M.gguf -o json` (ดู `--help` ประกอบ) → เก็บ stdout จริงเป็นค่าคงที่ `BENCH_FIXTURE` ใน test file + จดค่า `prompt_tps` จริงที่เห็น
- [ ] **Step 2: failing tests** (`tests/test_compression_benchmark.py`):
```python
def test_parse_real_fixture():        # parse(BENCH_FIXTURE)["prompt_tps"] == <ค่าจริงที่ capture> (อย่างน้อย > 0)
def test_parse_tolerates_missing_keys():  # json ไม่มี field load_ms → None (ไม่ crash)
def test_sample_memory_reads_vmhwm_for_this_process():
    mem = benchmark.sample_memory(os.getpid())
    assert mem["peak_rss_mb"] is not None and mem["peak_rss_mb"] > 0   # /proc จริง
def test_sample_memory_vram_none_when_xpu_smi_missing(monkeypatch):
    monkeypatch.setattr(benchmark.shutil, "which", lambda *_: None)
    assert benchmark.sample_memory(os.getpid())["peak_vram_mb"] is None
def test_sample_memory_vram_none_when_parse_fails(monkeypatch):        # xpu-smi คืน output ขยะ → None
def test_run_bench_nonzero_exit_raises(monkeypatch):                    # mock Popen rc=1 → CompressionError + "llama-bench"
```
- [ ] **Step 3: fail** → **Step 4: implement** (json.loads แบบลอง list/dict; VmHWM จาก `/proc/<pid>/status`; xpu-smi: `which` → None, output parse ไม่ได้ → None)
- [ ] **Step 5: pass** → **Step 6: Integration** — `@pytest.mark.integration` `test_run_bench_real_artifact`: `run_bench(q4_path)` → `prompt_tps > 0` → รัน `-m integration` PASS
- [ ] **Step 7: Commit** — `feat(compression): llama-bench metrics + memory sampling`

---

### Task 5: llama_runner (server lifecycle)

**Files:**
- Create: `core/compression/llama_runner.py`, `tests/test_compression_runner.py`

**Interfaces:**
- Consumes: Task 1 (`tool_path`, `device_args`, `require_tools`, `CompressionError`)
- Produces:
  - `find_free_port() -> int`
  - `class ServerHandle` — `proc`, `url`, `alias`, **`log_text`** (accumulate stdout ทั้งรัน — ให้ CLI ใช้ parse `kv_cache_mb`); method `complete(prompt: str, *, n_predict: int, temperature: float = 0.0) -> str` (POST `{url}/completion` ด้วย `urllib.request`, body `{"prompt": prompt, "n_predict": n_predict, "temperature": temperature, "cache_prompt": False}` → `resp["content"]`)
  - `start_server(gguf: Path, *, port: int | None = None, device: str = "vulkan", timeout: float = 60.0) -> ServerHandle` — Popen `llama-server -m gguf --host 127.0.0.1 --port <p> --alias <gguf.stem> + device_args`, stdout→pipe เก็บ log; poll `/health` จน ready; **check `GET /props` → `model == alias`**; error คงที่:
    - proc ตายก่อน ready → `f"llama-server failed to start: {tail}. Try --port or --device cpu."`
    - props ไม่ตรง → `f"port {port} is serving a different model (expected {alias}). Try another --port."`
    - timeout → `f"llama-server did not become ready in {timeout}s: {tail}"`
  - `stop_server(handle, *, timeout: float = 10.0) -> None` — SIGINT → `wait(timeout)` → ไม่จบ → `kill()`
  - `kv_cache_mb(log_text: str) -> float | None`
  - HTTP ผ่าน helper `_http_get(url)` / `_http_post(url, body)` module-level (monkeypatch ได้)

- [ ] **Step 1: capture fixture จริง 1 ครั้ง** — start server manual กับ gguf ที่มี → copy บรรทัด log ที่บอก KV cache size + response `GET /props` → เป็นค่าคงที่ใน test
- [ ] **Step 2: failing tests** (`tests/test_compression_runner.py`):
```python
def test_find_free_port_is_unused():              # bind ได้จริง
def test_start_builds_alias_and_waits_for_health(monkeypatch):  # mock Popen ไม่ตาย + mock _http_get health=200, props={"model": alias} → handle.url มี port
def test_start_reports_port_busy(monkeypatch):     # mock proc ตาย rc=1 → CompressionError, msg มี "failed to start" และ "Try --port"
def test_start_rejects_foreign_model(monkeypatch): # props={"model": "someone-else"} → msg มี "different model"; และ stop ถูกเรียก
def test_start_timeout_error(monkeypatch):         # health ไม่มา → msg มี "did not become ready"
def test_complete_passes_prompt_and_params_verbatim(monkeypatch):  # mock _http_post จับ body — prompt ต้องเป็น byte เดียวกับ input
def test_stop_server_sends_sigint_then_wait(monkeypatch):          # mock proc: send_signal(SIGINT) → wait(timeout) — ไม่ใช่ kill ก่อน
def test_stop_server_kills_after_timeout(monkeypatch):             # wait raise TimeoutExpired → kill()
def test_kv_cache_mb_parses_fixture_and_garbage():                 # บรรทัดจริง → float, ขยะ → None
```
- [ ] **Step 3: fail** → **Step 4: implement** → **Step 5: pass**
- [ ] **Step 6: Integration** — `@pytest.mark.integration` `test_server_lifecycle_real`: start(q4) → `complete("def f():", n_predict=8)` คืน string ไม่ว่าง → stop → `proc.poll() is not None`
- [ ] **Step 7: Commit** — `feat(compression): llama-server runner with health/alias checks`

---

### Task 6: llama_eval (eval ผ่าน server — reuse metrics เดิม)

**Files:**
- Create: `core/compression/llama_eval.py`, `tests/test_compression_llama_eval.py`

**Interfaces:**
- Consumes: `core.evaluator.{EvalCase, build_eval_cases, exact_match, token_f1, EVAL_MAX_NEW_TOKENS}`, `core.trainer_worker.build_fim_prompt`, `ServerHandle.complete`
- Produces:
  - `evaluate_with_llama(cases: list[EvalCase], server, *, tokenizer, fim_tokens: dict, n_predict: int = EVAL_MAX_NEW_TOKENS, progress: Callable[[int, int], None] | None = None) -> dict` — คืน `{"per_case": [...], "exact_match_pct": ..., "token_f1_mean": ...}` รูปเดียวกับ `evaluate_cases`; แต่ละ `per_case` เพิ่ม key `"latency_ms"`; prompt = `build_fim_prompt(case.prefix, case.suffix, fim_tokens=fim_tokens)` **ส่งออกไป byte เดียวกัน**; progress ทุก 10 + ครั้งสุดท้าย (กฎเดิม)

- [ ] **Step 1: failing tests** (`tests/test_compression_llama_eval.py`):
```python
def test_eval_prompt_verbatim_equals_build_fim_prompt():   # fake server จับ prompt — assert == build_fim_prompt(...) ทั้ง string (Review Focus #1)
def test_perfect_predictions_score_full():                # fake server คืน case.middle → exact_match_pct == 100.0, token_f1_mean == 1.0
def test_empty_predictions_score_zero():                  # คืน "" → 0.0 / 0.0
def test_result_shape_matches_evaluate_cases():           # keys == {"per_case","exact_match_pct","token_f1_mean"}; per_case มี i/exact/f1/pred/gt/latency_ms
def test_progress_every_ten_and_last():                   # cases=12 → calls == [(10,12),(12,12)]
```
(Fake case = `EvalCase(prefix=..., suffix=..., middle=...)` ตรง ๆ; fake server มี attribute `complete`)
- [ ] **Step 2: fail** → **Step 3: implement** (ลูปเดียวกับ `evaluate_cases` แต่แทน `model.generate` ด้วย `server.complete` + จับเวลา `time.perf_counter`) → **Step 4: pass**
- [ ] **Step 5: Commit** — `feat(compression): FIM eval via llama-server reusing Phase 5 metrics`

---

### Task 7: CLI `benchmark_compression.py`

**Files:**
- Create: `scripts/benchmark_compression.py`, `tests/test_compression_cli.py`

**Interfaces:**
- Consumes: ทุก module ก่อนหน้า + `core.estimator.{resolve_model_spec, ModelSpecUnavailable}` + `core.evaluator.{build_eval_cases, EVAL_MAX_NEW_TOKENS}` + `AutoTokenizer.from_pretrained(source)`
- Produces:
  - `main(argv: list[str] | None = None) -> int`
  - `run_benchmark(args) -> list[dict]` (คืน rows ของ report ทุก variant)
  - flags: `--model` (required), `--variants` (comma, default `fp16,q8_0,q4_k_m`), `--device` (`vulkan`|`cpu`, default `vulkan`), `--context` (default `MAX_SEQ_LENGTH_DEFAULT`), `--eval-cases` (default `100`), `--no-eval`, `--dataset` (default `DEFAULT_DATASET_ID`), `--column` (default `DEFAULT_DATASET_COLUMN`), `--limit` (default `TRAIN_CODE_LIMIT`), `--fim-key` (default `"qwen"`), `--baseline` (optional path)
  - flow: `resolve_source` → `require_tools` → `parameter_count` = `resolve_model_spec(model_id, None)` (ยก `ModelSpecUnavailable` → `None` + warning อังกฤษ) → ต่อ variant: `build_artifacts` → `run_bench` → (unless `--no-eval`: สร้าง cases ด้วย `build_eval_cases(dataset_id=..., dataset_column=..., limit=..., tokenizer=..., n_cases=args.eval_cases, max_seq_length=args.context)` → `start_server` → `evaluate_with_llama` → `stop_server`; `latency_ms` = ค่าเฉลี่ย per_case) → `build_report` + `write_report` → ปิดท้าย `print(format_table(rows, baseline))` (`--baseline` หรือ `pick_baseline`)
  - mapping เข้า `build_report`: `tokens_per_sec = bench["prompt_tps"]`, `load_time_ms = bench["load_ms"]`, `peak_vram_mb = bench["peak_vram_mb"]`, `kv_cache_mb = kv_cache_mb(handle.log_text)`, `model_disk_mb` = ขนาดไฟล์ gguf, `context_tokens = args.context`
  - `latency_ms` = mean ของ `per_case["latency_ms"]`; `tokens_per_sec` = `prompt_tps` ของ bench; `model_disk_mb` = ขนาด gguf

- [ ] **Step 1: failing tests** (`tests/test_compression_cli.py`) — monkeypatch ทุก module function (fake record):
```python
def test_variants_flag_parses_comma():          # "fp16,q4_k_m" → tuple 2 ตัว; default = DEFAULT_VARIANTS
def test_run_benchmark_orchestration_order():   # record order ต่อ variant == [artifact, bench, server.start, eval, server.stop, report]; --no-eval → ไม่มี server call
def test_run_benchmark_eval_disabled():         # --no-eval → exact_match_pct is None ใน report
def test_table_printed_with_baseline_delta():   # baseline stub → format_table ถูกเรียกพร้อม baseline
def test_missing_model_flag_exits_english():    # main([]) → SystemExit 2 + "--model" ใน usage
def test_unavailable_parameter_count_is_null(monkeypatch):  # resolve_model_spec raise → report["parameter_count"] is None
```
- [ ] **Step 2: fail** → **Step 3: implement** (argparse + orchestration; import module functions ตรง ๆ เพื่อ monkeypatch ได้) → **Step 4: pass**
- [ ] **Step 5: Integration e2e (หัวใจ DoD)** — `@pytest.mark.integration` `test_compression_pipeline_e2e`:
  - `run_benchmark` ด้วย Qwen + variants `fp16,q4_k_m` + `eval_cases=2` (monkeypatch นิดหน่อยให้ report dir = tmp_path)
  - assert: ทั้ง 4 gguf/ reports มีอยู่จริง (fp16/q4 × 2), `q4 model_disk_mb < fp16`, report keys ครบ §4, `exact_match_pct` ไม่ใช่ None
  - รัน `... -m integration` → PASS (ใช้เวลาหลายนาที)
- [ ] **Step 6: Commit** — `feat(compression): compression benchmark CLI with eval + table`

---

### Task 8: `serve.py` (local URL)

**Files:**
- Create: `scripts/serve.py`, `tests/test_compression_serve.py`

**Interfaces:**
- Consumes: `config.{resolve_source, artifact_path, CompressionError}`, `llama_runner.{start_server, stop_server, ServerHandle}`
- Produces: `serve(gguf: Path, *, port: int, device: str, start=start_server, stop=stop_server) -> int` และ `main(argv=None) -> int`
  - flags: `--gguf <path>` **หรือ** `--model <name>` (+ `--variant` default `q4_k_m`), `--port` default `8080`, `--device` default `vulkan`
  - flow: resolve gguf → `start` → print `Server ready: {handle.url} (model: {handle.alias})` (flush) → รอ KeyboardInterrupt/SIGTERM → `stop` → print `Server stopped.` → return 0
  - `CompressionError` → print message (English) → return 1

- [ ] **Step 1: failing tests** (`tests/test_compression_serve.py`):
```python
def test_serve_prints_url_then_stops_child_on_keyboard_interrupt():
    # fake start คืน fake handle; fake wait raise KeyboardInterrupt → assert stop(fake_handle) ถูกเรียก,
    # output มี "Server ready: " และ "Server stopped.", return 0   (Review Focus #3)
def test_serve_start_error_returns_1():        # fake start raise CompressionError("...") → return 1 + message ถูก print
def test_serve_gguf_missing_error():           # path ไม่มี .gguf → CompressionError "no .gguf" + return 1
def test_serve_model_variant_resolves_artifact_path(monkeypatch, tmp_path):  # --model alpha --variant q8_0 → ได้ path ที่ artifact_path กำหนด
```
- [ ] **Step 2: fail** → **Step 3: implement** (แยก `serve()` รับ start/stop injectable; `main` wire argparse) → **Step 4: pass**
- [ ] **Step 5: Verification จริง (DoD)** — `python scripts/serve.py --model Qwen/Qwen2.5-Coder-0.5B --variant q4_k_m --port 8099` ใน background → `curl -sf http://127.0.0.1:8099/health` → HTTP 200 → ส่ง SIGINT → บรรทัด `Server stopped.` + port ว่าง (`curl` fail)
- [ ] **Step 6: Commit** — `feat(compression): serve.py local URL wrapper for llama-server`

---

### Task 9: README + final gates

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: ทุกอย่างจาก Task 1–8
- Produces: เอกสารใช้งาน (ภาษาไทย) +  chứng ว่าทั้ง branch ผ่าน

- [ ] **Step 1: เพิ่ม section README** — หัวข้อ `## Compression (GGUF / llama.cpp)` ใต้ section Test: วิธี `setup_llamacpp.sh`, ตัวอย่าง `benchmark_compression.py` (variants, --no-eval, --device cpu), ตัวอย่าง `serve.py` + URL, ที่อยู่ artifacts (`<source>/gguf/`) กับ reports (`benchmarks/<model>/<variant>.json`), หมายเหตุ honest-null + report-only (อ้าง spec/roadmap), ข้อความว่า GPTQ/AWQ/OpenVINO ยังไม่อยู่ใน scope
- [ ] **Step 2: Full suite** — `../../.venv/bin/python -m pytest tests/ -q` → ต้อง PASS ทั้งหมด (ของเดิม 153 + ใหม่), `3 deselected` คงเดิม
- [ ] **Step 3: Lint** — `uvx ruff check core/ ui/ configs/ tests/` → `Found N errors` ต้อง `N <= 5` (ห้ามแย่กว่า master)
- [ ] **Step 4: ทวน DoD ทุกข้อ** จาก spec §9 — รายการไหนยังไม่ครบ กลับไปทำให้ครบ
- [ ] **Step 5: Commit** — `docs: compression (GGUF) usage guide (Thai)`
