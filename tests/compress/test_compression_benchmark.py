"""Unit tests สำหรับ core.compress.benchmark — parse fixture จริง + memory sample"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.compress import benchmark, config

# stdout จริงของ `llama-bench -m <q4_k_m> -o json` (b11371, B580/Vulkan) — 2026-10-03
BENCH_FIXTURE = """\
[
  {
    "build_commit": "99b9548",
    "build_number": 1,
    "cpu_info": "AMD Ryzen 5 7500F 6-Core Processor",
    "gpu_info": "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
    "backends": "Vulkan",
    "model_filename": "models/Qwen/Qwen2.5-Coder-0.5B/gguf/Qwen2.5-Coder-0.5B-q4_k_m.gguf",
    "model_type": "qwen2 1B Q4_K - Medium",
    "model_size": 391859712,
    "model_n_params": 494032768,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 6,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": -1,
    "n_cpu_moe": 0,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": -1,
    "devices": "auto",
    "tensor_split": "0.00",
    "tensor_buft_overrides": "none",
    "load_mode": "auto",
    "lazy_mode": "auto",
    "embeddings": false,
    "no_op_offload": 0,
    "no_host": false,
    "repack": true,
    "fit_target": 0,
    "fit_min_ctx": 0,
    "n_prompt": 512,
    "n_gen": 0,
    "n_depth": 0,
    "test_time": "2026-10-03T09:43:39Z",
    "avg_ns": 38035085,
    "stddev_ns": 402294,
    "avg_ts": 13462.443199,
    "stddev_ts": 140.422336,
    "samples_ns": [ 38750537, 37892080, 37904861, 37813450, 37814501 ],
    "samples_ts": [ 13212.7, 13512.1, 13507.5, 13540.2, 13539.8 ]
  },
  {
    "build_commit": "99b9548",
    "build_number": 1,
    "cpu_info": "AMD Ryzen 5 7500F 6-Core Processor",
    "gpu_info": "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
    "backends": "Vulkan",
    "model_filename": "models/Qwen/Qwen2.5-Coder-0.5B/gguf/Qwen2.5-Coder-0.5B-q4_k_m.gguf",
    "model_type": "qwen2 1B Q4_K - Medium",
    "model_size": 391859712,
    "model_n_params": 494032768,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 6,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": -1,
    "n_cpu_moe": 0,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": -1,
    "devices": "auto",
    "tensor_split": "0.00",
    "tensor_buft_overrides": "none",
    "load_mode": "auto",
    "lazy_mode": "auto",
    "embeddings": false,
    "no_op_offload": 0,
    "no_host": false,
    "repack": true,
    "fit_target": 0,
    "fit_min_ctx": 0,
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 0,
    "test_time": "2026-10-03T09:43:39Z",
    "avg_ns": 345380369,
    "stddev_ns": 4678980,
    "avg_ts": 370.659695,
    "stddev_ts": 4.964620,
    "samples_ns": [ 341794994, 341911661, 343584447, 346619793, 352990953 ],
    "samples_ts": [ 374.493, 374.366, 372.543, 369.281, 362.616 ]
  }
]
"""


def test_parse_real_fixture():
    metrics = benchmark.parse_bench_output(BENCH_FIXTURE)
    assert metrics["prompt_tps"] == 13462.443199  # row n_prompt=512, n_gen=0
    assert metrics["gen_tps"] == 370.659695  # row n_gen=128
    assert metrics["load_ms"] is None  # llama-bench ไม่มี field นี้ → None ซื่อสัตย์


def test_parse_tolerates_missing_keys():
    metrics = benchmark.parse_bench_output('[{"n_prompt": 512, "n_gen": 0}]')
    assert metrics == {"prompt_tps": None, "gen_tps": None, "load_ms": None}


def test_sample_memory_reads_vmhwm_for_this_process():
    mem = benchmark.sample_memory(os.getpid())
    assert mem["peak_rss_mb"] is not None
    assert mem["peak_rss_mb"] > 0  # /proc จริง


def test_sample_memory_vram_none_when_xpu_smi_missing(monkeypatch):
    monkeypatch.setattr(benchmark.shutil, "which", lambda *_: None)
    assert benchmark.sample_memory(os.getpid())["peak_vram_mb"] is None


def test_sample_memory_vram_none_when_parse_fails(monkeypatch):
    monkeypatch.setattr(benchmark, "_xpu_smi_output", lambda: "no numbers here at all")
    assert benchmark.sample_memory(os.getpid())["peak_vram_mb"] is None


def test_run_bench_nonzero_exit_raises(monkeypatch):
    class FakeProc:
        pid = 99999999  # /proc ไม่มี → sample_memory คืน None ทั้งคู่ (ไม่ crash)
        stdout = iter(())  # drain contract: stream ไม่ใช่ communicate()
        stderr = iter(("bench exploded\n",))

        def poll(self):
            return 1  # rc=1 ทันที

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **k: FakeProc())

    with pytest.raises(config.CompressionError) as exc:
        benchmark.run_bench(Path("/tmp/x.gguf"))

    msg = str(exc.value)
    assert "llama-bench" in msg
    assert "bench exploded" in msg


@pytest.mark.integration
def test_run_bench_real_artifact():
    try:
        config.require_tools()
    except config.CompressionError:
        pytest.skip("llama.cpp not built")
    try:
        src = config.resolve_source("Qwen/Qwen2.5-Coder-0.5B")
    except config.CompressionError:
        pytest.skip("Qwen model not downloaded")
    gguf = config.artifact_path(src, "q4_k_m")
    if not gguf.is_file():
        pytest.skip("q4_k_m artifact not built yet")

    metrics = benchmark.run_bench(gguf)

    assert metrics["prompt_tps"] is not None and metrics["prompt_tps"] > 0
    assert metrics["gen_tps"] is not None and metrics["gen_tps"] > 0
    assert set(metrics) >= {"prompt_tps", "gen_tps", "load_ms", "peak_rss_mb", "peak_vram_mb"}


def test_run_bench_drains_stderr_while_running(monkeypatch, tmp_path):
    """Review #13: child เขียน stderr เต็ม pipe (64KB) → ถ้าไม่ drain ระหว่างรอ = deadlock"""
    import threading

    tools = tmp_path / "tools"
    bin_dir = tools / "build" / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "llama-bench"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "head -c 200000 /dev/zero >&2\n"  # เกิน 64KB pipe buffer → เต็มถ้าไม่อ่าน
        "echo '[{\"n_prompt\": 512, \"n_gen\": 0, \"avg_ts\": 100.0}, "
        "{\"n_prompt\": 0, \"n_gen\": 128, \"avg_ts\": 50.0}]'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tools))

    result: dict = {}

    def run():
        try:
            result["out"] = benchmark.run_bench(Path("/tmp/x.gguf"), device="cpu")
        except Exception as exc:  # noqa: BLE001 — จับให้ test เห็น error ชัด ๆ
            result["err"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=20)

    assert not t.is_alive(), "run_bench hung on full stderr pipe (Review #13)"
    assert "err" not in result, result.get("err")
    assert result["out"]["prompt_tps"] == 100.0
    assert result["out"]["gen_tps"] == 50.0
