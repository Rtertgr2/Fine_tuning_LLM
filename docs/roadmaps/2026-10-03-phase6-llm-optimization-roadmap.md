# LLM Optimization Lab — Roadmap ต่อจาก Phase 5

> Repository: `Rtertgr2/Fine_tuning_LLM`  
> เป้าหมาย: ขยายจาก Fine-tuning Tool ให้เป็น **LLM Optimization Lab** ที่รองรับ Fine-tuning, Compression, Context Optimization, MoE, Distillation และ Deployment  
> วันที่แผน: 2026-10-03

---

## 1. ภาพรวม

โปรเจกต์ปัจจุบันทำ **Fine-tuning pipeline** ได้ค่อนข้างครบแล้ว โดยแกนหลักที่มีอยู่ประกอบด้วย:

- Intel XPU runtime / hardware gate
- VRAM estimator
- FIM dataset builder
- held-out split และ leakage guard
- LoRA fine-tuning
- subprocess training + IPC
- watchdog / abort / atomic checkpoint save
- Gradio dashboard
- Base vs Fine-tuned evaluation
- Exact Match + Token F1
- CLI สำหรับ evaluation
- unit/integration tests
- documentation และ phase-based implementation plan

ดังนั้น phase ถัดไป **ไม่ควรรื้อ Phase 1–5** แต่ควรต่อยอดให้ระบบครอบคลุมคำถามที่ใหญ่ขึ้น:

> ทำอย่างไรให้โมเดลเล็กลง ใช้ memory น้อยลง เร็วขึ้น รองรับ context ได้มากขึ้น และรักษาคุณภาพไว้ให้มากที่สุด

---

# 2. Architecture เป้าหมาย

```text
                         LLM Optimization Lab
                                  |
        +-------------------------+--------------------------+
        |                         |                          |
        v                         v                          v
   Fine-tuning              Compression                 Context
        |                         |                          |
     LoRA/FIM                 INT8/INT4                 Long Context
     QLoRA                    GPTQ/AWQ                  Chunking
     DoRA                     INT3/INT2                 KV Cache
     DPO                      1.58-bit                  KV Compression
        |                         |
        +-------------------------+
                                  |
                                  v
                              MoE / Distillation
                                  |
                                  v
                              Benchmark
                                  |
                                  v
                              Deployment
```

---

# 3. หลักการสำคัญของแผน

ทุก optimization ต้องถูกวัดด้วยมาตรฐานเดียวกัน

```text
Optimization
     |
     +--> Memory
     +--> Disk Size
     +--> Latency
     +--> Tokens/sec
     +--> Context Capacity
     +--> Quality
     +--> Reliability
```

ห้ามสรุปว่า optimization ดีเพียงเพราะ model เล็กลง

ต้องตอบได้ว่า:

- ลด memory ได้เท่าไร
- ลด disk ได้เท่าไร
- inference เร็วขึ้นหรือช้าลง
- context ที่รองรับเปลี่ยนอย่างไร
- quality ลดลงเท่าไร
- syntax/execution correctness เปลี่ยนอย่างไร
- hardware/backend ไหนรองรับจริง

---

# 4. Benchmark Schema กลาง

ควรมี schema เดียวสำหรับทุก experiment

```json
{
  "model": "Qwen/Qwen2.5-Coder-0.5B",
  "variant": "int4-gptq",
  "optimization": "quantization",
  "backend": "openvino",
  "weight_bits": 4,
  "context_tokens": 8192,
  "parameter_count": 0,
  "model_disk_mb": 0,
  "peak_vram_mb": 0,
  "kv_cache_mb": 0,
  "tokens_per_sec": 0,
  "latency_ms": 0,
  "exact_match_pct": 0,
  "token_f1": 0,
  "syntax_pass_rate": 0,
  "execution_pass_rate": 0,
  "timestamp": "..."
}
```

ทุก benchmark phase ควร export schema ที่ compatible กัน เพื่อให้สร้าง comparison dashboard ได้ในภายหลัง

---

# 5. Main Roadmap

```text
Phase 5
  Fine-tuning + Evaluation
          |
          v
Phase 6
  Model Compression Lab
          |
          v
Phase 7
  Context Engineering
          |
          v
Phase 8
  MoE Lab
          |
          v
Phase 9
  Knowledge Distillation
          |
          v
Phase 10
  Unified Optimization Benchmark
          |
          v
Phase 11
  Inference / Deployment
```

---

# PHASE 6 — Model Compression Lab

## Goal

เพิ่มระบบสำหรับลดขนาด model / ลด VRAM / เพิ่ม inference efficiency โดยยังสามารถเปรียบเทียบ quality กับ baseline ได้

## 6.1 Compression Architecture

### Files

```text
core/
└── compression/
    ├── __init__.py
    ├── registry.py
    ├── config.py
    ├── quantizer.py
    ├── calibrator.py
    ├── memory.py
    ├── benchmark.py
    └── report.py
```

### Responsibilities

#### `registry.py`

ลงทะเบียน backend:

```text
bitsandbytes
Quanto
GPTQ
AWQ
OpenVINO
HQQ
future: BitNet
```

#### `config.py`

เก็บ configuration:

```yaml
precision: int4
backend: openvino
group_size: 128
calibration_samples: 128
```

#### `quantizer.py`

interface กลาง:

```python
class Quantizer:
    def prepare(self, model):
        ...

    def calibrate(self, dataset):
        ...

    def quantize(self, model):
        ...

    def export(self, output_dir):
        ...
```

#### `memory.py`

วัด:

- parameter memory
- model load memory
- peak inference memory
- KV cache memory
- total process memory

#### `benchmark.py`

วัด:

- latency
- throughput
- tokens/sec
- quality
- memory

---

## 6.2 Baseline Snapshot

ต้องสร้าง BF16 baseline ก่อนเริ่ม quantization

### Files

```text
benchmarks/
└── baseline/
    └── bf16.json
```

### Metrics

```text
parameter_count
disk_size_mb
peak_vram_mb
load_time_ms
latency_ms
tokens_per_sec
Exact Match
Token F1
Syntax Pass
Execution Pass
```

### Definition of Done

- baseline รันซ้ำได้
- schema ตายตัว
- result เก็บใน JSON
- benchmark ทุก compression variant ต้องใช้ชุด evaluation เดียวกัน

---

## 6.3 INT8

### Flow

```text
BF16
  |
  v
INT8
  |
  v
Benchmark
```

### Tasks

- เพิ่ม INT8 backend
- quantized model loading
- memory measurement
- latency measurement
- quality evaluation
- report generator

### Test

- quantized model load ได้
- output shape ถูกต้อง
- inference ไม่ crash
- metrics ถูกบันทึก
- baseline เทียบได้

---

## 6.4 INT4

INT4 เป็น compression target หลักของ project

### Variants

```text
INT4
├── generic / RTN baseline
├── GPTQ
└── AWQ
```

### Benchmark

```text
BF16
vs
INT8
vs
INT4-RTN
vs
INT4-GPTQ
vs
INT4-AWQ
```

### Metrics

```text
disk size
VRAM
load time
latency
tokens/sec
Token F1
Exact Match
Syntax pass
Execution pass
```

---

## 6.5 Hardware Backend Matrix

อย่า assume ว่า algorithm เดียวรองรับ hardware ทุกแบบ

ต้องมี matrix:

```text
              Intel Arc     CUDA     CPU
BF16             OK           OK       OK
INT8             ?            ?        ?
INT4-GPTQ         ?            ?        ?
INT4-AWQ          ?            ?        ?
OpenVINO INT4     OK           -        ?
```

สถานะ:

```text
SUPPORTED
EXPERIMENTAL
UNSUPPORTED
```

ต้องตรวจจาก backend จริงก่อนเปิดใช้งาน UI

---

## 6.6 INT3 / INT2

หลัง INT4 เสถียรแล้วค่อยทำ low-bit benchmark

### Flow

```text
INT4
 |
 +--> INT3
 |
 +--> INT2
```

ต้องเน้น **quality degradation curve**

ตัวอย่าง report:

```text
bits/weight -> quality -> memory -> throughput
```

ห้ามตั้ง threshold จากตัวเลขสมมุติ

---

## 6.7 QLoRA

QLoRA เป็น bridge ระหว่าง:

```text
Fine-tuning
+
Quantization
```

### Architecture

```text
Quantized Base Model
       |
       | frozen
       v
     LoRA
       |
       v
   Training
```

### Tasks

- เพิ่ม training mode `qlora`
- estimator สำหรับ QLoRA
- configuration
- compatibility gate
- training test
- compare LoRA vs QLoRA

### UI

```text
Fine-tuning Method

[ LoRA ]
[ QLoRA ]
```

### Benchmark

```text
LoRA
vs
QLoRA
```

วัด:

- training VRAM
- training time
- final quality
- adapter size

---

## 6.8 1.58-bit / Ternary Research Track

**แยกออกจาก production pipeline**

แนวคิดนี้สัมพันธ์กับ ternary weights:

```text
{-1, 0, +1}
```

และ representation ระดับประมาณ 1.58 bits/weight ในงาน BitNet b1.58

### Files

```text
research/
└── bitnet/
    ├── ternary.py
    ├── quant_math.py
    ├── toy_model.py
    ├── train.py
    └── benchmark.py
```

### Scope

ยังไม่พยายามแปลง production Qwen เป็น ternary แบบตรง ๆ

ให้ทำ:

```text
FP weight
   |
   v
ternary approximation
   |
   v
error measurement
   |
   v
QAT / retraining experiment
   |
   v
quality recovery
```

### Definition of Done

- toy model train ได้
- ternary quantizer deterministic
- memory reduction วัดได้
- quality degradation วัดได้
- documentation แยกจาก production

---

## 6.9 Compression Benchmark CLI

สร้าง CLI กลาง:

```bash
python benchmark_compression.py \
  --model ./models/base \
  --variants bf16,int8,int4-gptq,int4-awq
```

Output:

```text
Compression Benchmark

Variant       Size     VRAM     TPS      F1       Syntax
-----------------------------------------------------------
BF16          ...      ...      ...      ...      ...
INT8          ...      ...      ...      ...      ...
INT4-GPTQ     ...      ...      ...      ...      ...
INT4-AWQ      ...      ...      ...      ...      ...
```

---

# PHASE 7 — Context Engineering

## Goal

แยก “max sequence length” ออกจาก “context optimization”

ตอนนี้ repo มี `MAX_SEQ_LENGTH_CAP=2048`

เป้าหมายใหม่คือ:

```text
2048
4096
8192
16384
32768
```

พร้อม hardware-aware safety check

---

## 7.1 Context Profiler

### Files

```text
core/context/
├── profiler.py
├── estimator.py
├── token_budget.py
├── chunker.py
├── compressor.py
├── sliding_window.py
├── kv_cache.py
└── benchmark.py
```

### Tasks

วัด:

- VRAM
- latency
- tokens/sec
- OOM boundary
- KV cache growth

---

## 7.2 Context Capacity Estimator

Flow:

```text
requested context
        |
        v
model native context
        |
        v
hardware memory estimation
        |
        v
safe context
```

ตัวอย่าง:

```text
Model supports 32768
Hardware safe limit = 8192

UI maximum = 8192
```

ห้ามยก hard cap โดยไม่มี estimator รองรับ

---

## 7.3 Context Budget Manager

แบ่ง budget:

```text
32K total

System       1K
Prefix       8K
Retrieved    12K
Suffix       5K
Output       6K
```

### API

```python
allocate_context_budget(
    total_tokens,
    system_tokens,
    input_tokens,
    output_tokens,
)
```

---

## 7.4 Chunking

รองรับ:

```text
fixed-token
line-based
function-based
class-based
AST-aware
```

ลำดับการพัฒนา:

```text
fixed
  |
  v
line
  |
  v
function
  |
  v
AST-aware
```

สำหรับ Code LLM ควรให้ AST-aware เป็น target ระยะยาว

---

## 7.5 Context Compression

สร้าง:

```text
core/context/compressor.py
```

Strategies:

```text
truncate
extract
summary
semantic
code-aware
```

API:

```python
compress_context(
    context,
    max_tokens,
    strategy="code-aware",
)
```

---

## 7.6 Sliding Window

ใช้สำหรับ context ที่ยาวเกิน budget

```text
[window 1]
[window 2]
[window 3]
...
```

ต้องเก็บ:

- overlap
- token count
- chunk identity
- source offsets

เพื่อป้องกันข้อมูลหลุดและสามารถ reconstruct context ได้

---

## 7.7 KV Cache Profiler

ทดสอบ:

```text
2K
4K
8K
16K
32K
```

บันทึก:

```text
context_tokens
kv_cache_mb
peak_vram_mb
latency
tokens_per_sec
```

---

## 7.8 KV Cache Compression

แยก:

```text
Weight Compression
       +
KV Cache Compression
```

ตัวอย่าง matrix:

```text
Weights    KV Cache

BF16       FP16
INT4       FP16
BF16       INT8
INT4       INT8
INT4       INT4
```

### Intel Track

ศึกษาการใช้งาน OpenVINO INT4 KV cache กับ Intel GPU

---

## 7.9 TurboQuant Research Track

แยกเป็น:

```text
research/
└── turboquant/
```

Scope:

- reproduce algorithm concept
- compare QJL-based compression
- memory measurement
- quality evaluation
- long-context stress test

ห้ามผูกกับ production backend ใน phase แรก

---

# PHASE 8 — MoE Lab

## Goal

ศึกษาและ benchmark Mixture-of-Experts โดยไม่เริ่มจากการสร้าง large-scale MoE เอง

---

## 8.1 MoE Inspector

### Files

```text
core/moe/
├── inspector.py
├── router.py
├── metrics.py
└── benchmark.py
```

ตรวจ:

```text
num_experts
active_experts
expert dimensions
router
shared experts
routing policy
```

---

## 8.2 Existing MoE Benchmark

เปรียบเทียบ Dense กับ MoE

```text
Dense
vs
MoE
```

Metrics:

```text
total parameters
active parameters
VRAM
latency
tokens/sec
quality
```

ต้องอธิบายว่า:

```text
Total Parameters != Active Parameters != Compute per token
```

---

## 8.3 Router Analysis

วัด:

```text
expert token distribution
load imbalance
routing entropy
expert utilization
expert collapse
```

ตัวอย่าง visualization:

```text
Expert 01 ███████████
Expert 02 ███
Expert 03 ██████████████
Expert 04 █████
```

---

## 8.4 Dense → MoE Upcycling Research Track

ไม่ train production-scale MoE

เริ่มจาก toy model:

```text
Dense 100M-ish
      |
      v
4 experts
      |
      v
Top-1 / Top-2 routing
      |
      v
MoE research model
```

เป้าหมาย:

- routing correctness
- capacity factor
- load balancing
- training stability
- quality vs compute

---

# PHASE 9 — Knowledge Distillation

## Goal

ถ่ายทอดความสามารถของ model ใหญ่ไปยัง model เล็ก

```text
Teacher
   |
   v
Teacher Outputs
   |
   v
Distillation Dataset
   |
   v
Student
```

---

## 9.1 Teacher Generator

### Files

```text
core/distillation/
├── teacher.py
├── generator.py
├── dataset.py
├── loss.py
└── trainer.py
```

---

## 9.2 Dataset Types

สำหรับ Code LLM:

```text
FIM completion
bug fixing
refactoring
code explanation
```

ตัวอย่าง:

```json
{
  "prompt": "...",
  "teacher_output": "..."
}
```

---

## 9.3 Distillation Loss

เริ่มจาก:

```text
Student CE loss
+
KL divergence
```

จากนั้นค่อยเพิ่ม:

- token-level soft targets
- response quality
- code correctness
- selective distillation

---

## 9.4 Distillation Benchmark

เปรียบเทียบ:

```text
Teacher
Base Student
Fine-tuned Student
Distilled Student
```

วัด:

```text
quality retention
disk size
VRAM
latency
tokens/sec
syntax pass
execution pass
```

---

# PHASE 10 — Unified Optimization Benchmark

## Goal

ทำให้ทุกวิธีวัดบนสนามเดียวกัน

### Candidate Matrix

```text
BF16
INT8
INT4
INT4-GPTQ
INT4-AWQ
QLoRA
Distilled
MoE
Long Context
KV INT4
```

อาจมี combination เช่น:

```text
Distilled + INT4
Fine-tuned + INT4
Fine-tuned + Long Context
INT4 + KV INT4
```

---

## Metrics

### Model

```text
parameter_count
disk_size
```

### Runtime

```text
load_time
peak_vram
kv_cache
latency
tokens/sec
```

### Quality

```text
Exact Match
Token F1
Syntax Pass
Execution Pass
```

### Context

```text
max_safe_context
```

---

# PHASE 11 — Inference / Deployment

## Goal

แยก Training System ออกจาก Inference System

```text
Training
    |
    v
Model Artifact
    |
    v
Inference Engine
    |
    +--> Transformers
    +--> OpenVINO
    +--> vLLM
    +--> GGUF backend
```

---

## 11.1 Inference abstraction

### Files

```text
inference/
├── engine.py
├── transformers_backend.py
├── openvino_backend.py
├── vllm_backend.py
└── benchmark.py
```

API:

```python
class InferenceEngine:
    def load(self, model_path):
        ...

    def generate(self, prompt):
        ...

    def benchmark(self):
        ...
```

---

## 11.2 REST API

Target:

```text
POST /v1/generate
POST /v1/fim
POST /v1/benchmark
```

FIM example:

```json
{
  "prefix": "def add(a, b):\n    ",
  "suffix": "\n    return result"
}
```

Response:

```json
{
  "middle": "result = a + b",
  "model": "..."
}
```

---

# 6. UI Roadmap

จาก dashboard 3 tabs ปัจจุบัน ค่อยขยายเป็น:

```text
LLM Optimization Lab

├── Training
├── Compression
├── Context
├── MoE
├── Distillation
├── Benchmark
├── Models
└── Settings
```

---

## Compression UI

```text
Model
[ Qwen2.5-Coder-0.5B ]

Precision
[ BF16 ]

Target
[ INT4 ]

Backend
[ OpenVINO ]

Calibration Samples
[ 128 ]

[ Quantize ]

Result
---------------------------------
Disk:
VRAM:
Latency:
Tokens/sec:
Token F1:
Syntax:
Execution:
```

---

## Context UI

```text
Context Length
[ 2048 ]
[ 4096 ]
[ 8192 ]
[ 16384 ]
[ 32768 ]

KV Cache
[ FP16 ]
[ INT8 ]
[ INT4 ]

[ Benchmark Context ]
```

---

## MoE UI

```text
Model: Qwen3-30B-A3B

Experts: 128
Active Experts: 8

Routing Distribution
Expert 01 █████████
Expert 02 ███
Expert 03 ███████████
...
```

---

# 7. Test Strategy

ทุก phase ต้องเพิ่ม test อย่างน้อย 4 ระดับ

## Unit

```text
quantizer
context budget
router
distillation loss
benchmark schema
```

## Integration

```text
load model
quantize
benchmark
export
load artifact
```

## Hardware

```text
XPU availability
VRAM
OpenVINO backend
XPU memory
```

## Regression

```text
quality regression
memory regression
latency regression
context regression
```

---

# 8. Safety / Reliability

โดยเฉพาะเมื่อเพิ่ม execution evaluation และ code generation:

ห้าม execute generated code ตรง ๆ บน host

ต้องมี sandbox สำหรับ:

```text
CPU limit
memory limit
timeout
no network
read-only filesystem
process limit
```

ผลลัพธ์ต้องแยก:

```text
PASS
SYNTAX_ERROR
RUNTIME_ERROR
TIMEOUT
MEMORY_LIMIT
SANDBOX_ERROR
```

---

# 9. Definition of Done ต่อ Phase

## Phase 6

- [ ] BF16 baseline
- [ ] INT8
- [ ] INT4
- [ ] GPTQ
- [ ] AWQ
- [ ] QLoRA
- [ ] INT3/INT2 benchmark
- [ ] compression report
- [ ] hardware/backend matrix
- [ ] tests

## Phase 7

- [ ] context profiler
- [ ] 4K
- [ ] 8K
- [ ] 16K
- [ ] 32K
- [ ] context estimator
- [ ] token budget
- [ ] chunking
- [ ] sliding window
- [ ] KV cache profiler
- [ ] KV compression

## Phase 8

- [ ] MoE inspector
- [ ] existing MoE benchmark
- [ ] routing analysis
- [ ] toy MoE
- [ ] upcycling research

## Phase 9

- [ ] teacher generator
- [ ] distillation dataset
- [ ] KL/CE loss
- [ ] student training
- [ ] student benchmark

## Phase 10

- [ ] unified schema
- [ ] unified benchmark CLI
- [ ] comparison dashboard
- [ ] regression rules

## Phase 11

- [ ] inference abstraction
- [ ] OpenVINO backend
- [ ] Transformers backend
- [ ] optional vLLM backend
- [ ] REST API
- [ ] inference benchmark

---

# 10. Main Track vs Research Track

## Main Track

สิ่งที่ควรทำก่อนเพราะใช้งานจริงและ debug ได้ง่ายกว่า:

```text
1. BF16 baseline
2. INT8
3. INT4
4. GPTQ/AWQ
5. QLoRA
6. Context profiling
7. KV cache profiling
8. KV INT4
9. Existing MoE benchmark
10. Distillation
11. Unified benchmark
12. OpenVINO inference
```

## Research Track

แยกไม่ให้กระทบระบบหลัก:

```text
BitNet 1.58-bit
TurboQuant
2-bit / ternary experiments
Dense → MoE upcycling
Advanced routing
```

---

# 11. ลำดับ Task ที่แนะนำ

```text
Phase 6.1
Compression architecture
        |
        v
Phase 6.2
BF16/INT8 baseline
        |
        v
Phase 6.3
INT4
        |
        v
Phase 6.4
GPTQ/AWQ
        |
        v
Phase 6.5
QLoRA
        |
        v
Phase 6.6
INT3/INT2
        |
        v
Phase 6.7
Compression benchmark
        |
        v
Phase 6.8
BitNet research

        |
        v

Phase 7.1
Context profiler
        |
        v
Phase 7.2
Context estimator
        |
        v
Phase 7.3
4K/8K/16K/32K
        |
        v
Phase 7.4
Chunking
        |
        v
Phase 7.5
Context compression
        |
        v
Phase 7.6
KV cache
        |
        v
Phase 7.7
KV INT4
        |
        v
Phase 7.8
TurboQuant research

        |
        v

Phase 8
MoE Lab

        |
        v
Phase 9
Distillation

        |
        v
Phase 10
Unified Benchmark

        |
        v
Phase 11
Deployment
```

---

# 12. Expected Final Project

สุดท้าย repo ควรสามารถทำ flow นี้ได้:

```text
                User chooses model
                        |
                        v
                Hardware preflight
                        |
          +-------------+-------------+
          |                           |
          v                           v
     Fine-tuning                 Compression
          |                           |
       LoRA/QLoRA                 INT8/INT4
          |                       GPTQ/AWQ
          |                           |
          +-------------+-------------+
                        |
                        v
                Context Optimization
                        |
             +----------+----------+
             |                     |
             v                     v
        Long Context            KV Cache
             |                     |
             +----------+----------+
                        |
                        v
                   Evaluation
                        |
        +---------------+----------------+
        |               |                |
        v               v                v
     Quality          Memory           Speed
        |               |                |
        +---------------+----------------+
                        |
                        v
                  Model Registry
                        |
                        v
                  Inference API
```

---

# 13. Final Project Question

หลังจาก roadmap นี้เสร็จ โปรเจกต์จะไม่ตอบแค่:

> “Fine-tune model ได้ไหม?”

แต่สามารถตอบได้ว่า:

> “Model แบบไหนเหมาะกับ hardware นี้ที่สุด เมื่อพิจารณา quality, memory, latency, context และ deployment cost พร้อมกัน?”

นั่นคือเป้าหมายของ **LLM Optimization Lab**

---

# 14. References

- Repository: https://github.com/Rtertgr2/Fine_tuning_LLM
- Qwen2.5-Coder-0.5B: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B
- Hugging Face Quantization Docs: https://huggingface.co/docs/transformers/main_classes/quantization
- QLoRA paper: https://arxiv.org/abs/2305.14314
- GPTQ paper: https://arxiv.org/abs/2210.17323
- AWQ paper: https://arxiv.org/abs/2306.00978
- BitNet b1.58: https://arxiv.org/abs/2402.17764
- Google TurboQuant: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Qwen3 / MoE overview: https://qwenlm.github.io/blog/qwen3/
- Qwen MoE / upcycling: https://qwenlm.github.io/blog/qwen-moe/
- Intel Extension for Transformers: https://github.com/intel/intel-extension-for-transformers
- OpenVINO: https://github.com/openvinotoolkit/openvino
