# Phase 6a — GGUF Compression Pipeline (Design)

- **วันที่:** 2026-10-03
- **Status:** Approved (design sections §1–§3 อนุมัติ in-chat แล้ว)
- **อ้างอิง:** `docs/roadmaps/2026-10-03-phase6-llm-optimization-roadmap.md` (§4 benchmark schema, §6.1–6.5, §6.9 CLI, §7 test strategy, §8 safety)
- **Scope:** Sub-project แรกสุดของ LLM Optimization Lab (roadmap "Main Track" ข้อ 1–3)

---

## 1. Context & การตัดสินใจ (จากบทสนทนา)

| # | การตัดสินใจ | ผลกับแผนเดิม |
|---|---|---|
| 1 | ผลลัพธ์แรกของ 6a = **artifact โมเดล compress ที่ใช้ได้จริง** — benchmark เป็นผลพลอยได้ | แกนงาน = ได้ไฟล์ที่รันได้ ไม่ใช่ตารางตัวเลข |
| 2 | Runtime เป้าหมาย = **llama.cpp + GGUF** บนเครื่องนี้ (Intel Arc B580 ผ่าน Vulkan / CPU fallback) | **แทน** backend ที่ roadmap §6.1 เขียนไว้ทั้งหมด — bitsandbytes / Quanto / GPTQ / AWQ / OpenVINO / HQQ ออกจาก main track |
| 3 | **Approach A: standalone pipeline + ไม่แตะ Gradio app** | UI Compression tab + integration ใน app = sub-project ถัดไป |
| 4 | เพิ่ม **serve** = มี local URL ให้ต่อเครื่องมือข้างนอก | ราคาถูกเพราะใช้ `llama-server` ตัวเดียวกับที่ eval ใช้ |

**Assets เดิมที่ reuse (ไม่สร้างซ้ำ):** `build_eval_cases` / `exact_match` / `token_f1` / `build_fim_prompt` (`core/evaluator.py`), heldout predicate + local dataset (`core/dataset_builder.py`), `fim_registry.json`, `merge_export` → `exports/<name>-merged/` (`core/trainer_worker.py`), model path resolution (`models/`)

## 2. Goals / Non-goals

**Goals**
1. `scripts/setup_llamacpp.sh` build llama.cpp (Vulkan) จากศูนย์บนเครื่องนี้ (pin commit)
2. คำสั่งเดียวสร้าง artifact: HF dir → `fp16` → `q8_0` + `q4_k_m` GGUF
3. Measurements + `report.json` (§4 schema) + ตารางสรุป terminal
4. Eval เทียบ Phase 5 baseline ผ่าน llama-server — **ชุดเคส/เมตริกเดิม**
5. `scripts/serve.py` → local URL ที่ health check ผ่านจริง
6. Tests 4 ระดับ + README ไทย + error ทุกจุด English

**Non-goals (ชัดเจนว่าไม่ทำ)**
- ไม่แตะ Gradio app / ไม่เพิ่ม tab ใด ๆ
- ไม่ทำ REST API design ครบ (เป็น Phase 11)
- ไม่ทำ GPTQ/AWQ/bnb/OpenVINO (ตามข้อ 2 ข้างบน)
- ไม่ตั้ง quality threshold (report-only — §6.6 "ห้ามตั้ง threshold จากตัวเลขสมมุติ")
- ไม่ทำ syntax/execution evaluation (schema = `null` จนกว่าจะมี sandbox phase)
- Research tracks (BitNet / TurboQuant / upcycling) ไม่เกี่ยวกับ 6a

## 3. Architecture

```
core/compression/
├── __init__.py
├── config.py        # variants ที่เปิดใช้ (fp16, q8_0, q4_k_m), path ของ llama.cpp tools
│                    #   (env LLAMA_CPP_DIR → default tools/llama.cpp), device (vulkan|cpu)
├── quantizer.py     # HF dir → convert_hf_to_gguf.py → FP16 → llama-quantize
│                    #   (subprocess ทั้งคู่ — คืน artifact path)
├── benchmark.py     # llama-bench -o json + poll VmHWM (/proc) + xpu-smi (ถ้ามี) → metrics dict
├── llama_runner.py  # จุดเดียวของ llama-server: start → (proc, url) / stop / health
│                    #   ใช้ร่วมโดย: eval (HTTP แล้วปิด) และ serve (ค้างจน user สั่งหยุด)
└── report.py        # เขียน report.json ตาม §4 schema (fields ที่วัดได้จริง — วัดไม่ได้ = null)

scripts/
├── setup_llamacpp.sh          # clone (pin commit) + cmake -DGGML_VULKAN=ON → tools/
├── benchmark_compression.py   # quantize + bench + eval ในคำสั่งเดียว (roadmap §6.9)
└── serve.py                   # เลือก gguf + port → llama_runner.start → แสดง URL ค้างไว้

tools/llama.cpp/                # build output (gitignored — ไม่ commit binary)
<source_dir>/gguf/              # artifacts อยู่ข้าง source เสมอ —
                                #   models/<m>/gguf/ หรือ exports/<merged>/gguf/
                                #   ไฟล์: <source-name>-fp16.gguf / -q8_0.gguf / -q4_k_m.gguf
benchmarks/compression/<source-name>/<variant>.json   # §4 schema
```

**สัญญา interface (test แยกได้):** `quantizer` คืน artifact path → `benchmark` / `llama_runner` รับ path เท่านั้น → `report` รับ metrics dict → subprocess ทั้งหมด mock ได้ใน unit test

**ปรับจาก roadmap §6.1:** `registry.py` ตัด (มี backend เดียว), `calibrator.py` ตัด (k-quant ไม่ต้อง calibration), `memory.py` ยุบเข้า `benchmark.py`

## 4. Data flow, Measurements, Eval

```
โมเดล HF ต้นทาง (models/<name> หรือ exports/<merged> หลัง merge LoRA)
  ① convert_hf_to_gguf.py (venv python) → fp16.gguf        (~1.0GB)
  ② llama-quantize → q8_0.gguf (~0.53GB), q4_k_m.gguf (~0.40GB)
  ③ llama-bench -o json (+ memory poll) → metrics
  ④ --eval: llama-server ขึ้นครั้เดียว → FIM generation เซ็ตเดิม Phase 5 → EM/F1
  ⑤ report.json ต่อ variant + ตารางสรุป terminal:

  Variant     Size      TPS      EM      F1
  --------------------------------------------
  FP16        1.0 GB    ...     ...     ...
  Q8_0        0.53 GB   ...     ...     ...
  Q4_K_M      0.40 GB   ...     ...     ...
```

| Field (§4 schema) | Source |
|---|---|
| `model_disk_mb` | ขนาดไฟล์ `.gguf` |
| `tokens_per_sec`, `latency_ms`, `load_time_ms` | `llama-bench -o json` + log parse (unit test ด้วย fixture) |
| `peak_vram_mb` | `xpu-smi` ถ้ามี — ไม่มี → **`null` (ห้ามเดา)** |
| `kv_cache_mb` | log ของ llama-server |
| `exact_match_pct`, `token_f1` | `exact_match` / `token_f1` จาก `core/evaluator.py` ตัวเดิม |
| `syntax_pass_rate`, `execution_pass_rate` | `null` |
| `backend` | `"llama.cpp-vulkan"` หรือ `"llama.cpp-cpu"` |
| `optimization` | `"quantization"`, `weight_bits` = 16/8/4 |

**Eval (คุณภาพเป็นผลพลอยได้ — แต่ต้องซื่อสัตย์):**
- สร้างเคสด้วย `build_eval_cases(...)` **tokenizer ตัวเดิม** + dataset เดิม + seed เดิม → **เคสเดียวกับ Phase 5 ทุกตัวอักษร**
- generate: `build_fim_prompt(prefix, suffix, fim_tokens)` (ของเดิม) → `POST /completion` ต่อ llama-server (`n_predict = EVAL_MAX_NEW_TOKENS`, `temperature = 0`) — server ค้างทั้ง session ไม่ reload ทุกเคส
- metric: `exact_match` + `token_f1(pred, gt, tokenizer)` → คืน dict รูปเดียวกับ `evaluate_cases` → `compare_results` ใช้ได้เลย
- `n_cases`: คง default เดิมกับ `run_eval` (หรือกำหนดเองผ่าน `--eval-cases`) → เคสถึงจะเดียวกัน
- baseline ที่เทียบ: `data_cache/eval/*.json` (Phase 5) — แสดง Δ ในตาราง **เฉพาะเมื่อเจอไฟล์**; **report ไม่ใช่ hard gate**

**รับโมเดล:** `--model` = ชื่อใต้ `models/` หรือ path ตรง ๆ; ชื่อ Hub ที่ยังไม่มี → error อังกฤษ + บอกคำสั่ง `hf download ... --local-dir models/X`

## 5. Setup

```bash
scripts/setup_llamacpp.sh
# git clone llama.cpp → tools/llama.cpp (pin commit ใน script)
# cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# ผลลัพธ์: tools/llama.cpp/build/bin/llama-{quantize,server,bench,cli}
#          + tools/llama.cpp/convert_hf_to_gguf.py
```
- เช็ค `vulkaninfo` ก่อน → ไม่มี → error อังกฤษ (บอกติดตั้ง `vulkan-intel`)
- `config.py` หา tools จาก `LLAMA_CPP_DIR` (รองรับ system install) → default `tools/llama.cpp`
- `tools/` gitignore ทั้งโฟลเดอร์; convert รันด้วย venv python (มี transformers/numpy แล้ว)

## 6. Error handling (English เสมอ)

| กรณี | ข้อความ |
|---|---|
| ยังไม่ได้ build | `llama.cpp not found. Run scripts/setup_llamacpp.sh first.` |
| convert/quantize fail | error + log tail ของคำสั่งนั้น |
| server ไม่ขึ้น / port ถูกใช้ | `llama-server failed to start: <log tail>. Try --port or --device cpu.` |
| โมเดลไม่มีใน `models/` | บอกคำสั่ง `hf download ... --local-dir models/X` |
| Ctrl+C ตอน serve | SIGINT → รอ process ปิดจริง → `Server stopped.` (ไม่ leak) |

## 7. Testing (4 ระดับ — roadmap §7)

- **Unit (รันปกติ ไม่ต้องมี llama.cpp):** path/ชื่อ artifact, parse `llama-bench` JSON + server log จาก **fixture (output จริง)**, report schema (ความซื่อสัตย์ `null`), การสร้าง command line (mock subprocess), FIM prompt ของ eval + ยืนยัน metric ได้ผลเท่า evaluator ตัวเดิม, `serve.py` arg → URL (mock runner)
- **Integration** (marker `integration` — deselected default ตาม pattern เดิม): `skipif` ไม่มี tools → quantize Qwen-0.5B จริง, bench ได้ metrics จริง, eval 2 เคสผ่าน server, `serve.py` → health 200 → stop
- **Hardware:** `vulkaninfo` presence + CPU fallback
- **Regression:** report-only — พิมพ์ Δ เทียบ Phase 5 ในตาราง ยังไม่ตั้ง threshold

## 8. Known risks

1. **FIM special tokens ใน GGUF tokenizer** — ถ้า map ไม่ถูก eval จะเพี้ยน → เห็นตรงจากตัวเลข; แก้ตอน implement ถ้าเจอ
2. **llama.cpp เปลี่ยนชื่อ binary/flag บ่อย** → pin commit ใน setup script
3. **Vulkan บน B580** อาจมีปัญหาเฉพาะ → `--device cpu` fallback + error message ชี้ทาง
4. `xpu-smi` อาจไม่มีในเครื่อง → `peak_vram_mb = null` (ไม่เดา)

## 9. Definition of Done

- [ ] `setup_llamacpp.sh` build สำเร็จจากศูนย์บนเครื่องนี้ (Vulkan)
- [ ] คำสั่งเดียวได้ `fp16 + q8_0 + q4_k_m` จาก base model **และ** merged LoRA (`exports/`)
- [ ] `report.json` ครบ §4 schema ทุก variant + ตารางสรุป terminal
- [ ] Eval เทียบ Phase 5 baseline ผ่าน llama-server (เคสเดิม)
- [ ] `serve.py` คืน URL ที่ health check ผ่านจริง
- [ ] Unit ผ่าน + integration มี (deselected default) + ruff ไม่แย่กว่า master
- [ ] README ไทย: setup + ใช้งานทุกคำสั่ง; error ทุกจุด English

## 10. Deferred (ต่อจาก 6a)

- Compression tab ใน Gradio app (sub-project ถัดไป — roadmap §6 UI)
- ปุ่ม Start/Stop Server + REST `/v1/generate` `/v1/fim` = Phase 11
- INT3/INT2 + quality degradation curve (6b), QLoRA (6c)
- Syntax/execution metrics + sandbox (roadmap §8)
- Quality threshold เมื่อมีข้อมูลจริงพอ
