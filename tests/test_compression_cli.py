"""Unit tests สำหรับ scripts/benchmark_compression.py — CLI orchestration (fake ทุกจุด)"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.compression.config import DEFAULT_VARIANTS

CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compression.py"


def _load_cli():
    """โหลด script เป็น module ใหม่ทุกครั้ง (monkeypatch ต่อ instance — ไม่รั่วข้าม test)"""
    spec = importlib.util.spec_from_file_location("benchmark_compression", CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeHandle:
    log_text = ""
    proc = None


def _wire(cli, monkeypatch, tmp_path, calls: list[str], reports: list[dict]):
    """ทำทุก module function ให้เป็น fake recorder — คืน (source_dir, fake_gguf)"""
    src = tmp_path / "m"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    gguf = src / "gguf" / "m-fp16.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"0" * 2048)

    monkeypatch.setattr(cli, "resolve_source", lambda m: src)
    monkeypatch.setattr(cli, "require_tools", lambda: None)
    monkeypatch.setattr(
        cli, "resolve_model_spec", lambda m, u: SimpleNamespace(num_params=123_000_000)
    )
    monkeypatch.setattr(
        cli, "build_artifacts", lambda s, v: (calls.append("artifact") or [gguf])
    )
    monkeypatch.setattr(
        cli,
        "run_bench",
        lambda g, device="vulkan": (
            calls.append("bench")
            or {
                "prompt_tps": 100.0,
                "gen_tps": 50.0,
                "load_ms": 800.0,
                "peak_rss_mb": 900.0,
                "peak_vram_mb": None,
            }
        ),
    )
    monkeypatch.setattr(
        cli, "start_server", lambda g, **k: (calls.append("server.start") or FakeHandle())
    )
    monkeypatch.setattr(
        cli,
        "evaluate_with_llama",
        lambda cases, handle, **k: (
            calls.append("eval")
            or {
                "per_case": [{"latency_ms": 40.0}, {"latency_ms": 60.0}],
                "exact_match_pct": 50.0,
                "token_f1_mean": 0.5,
            }
        ),
    )
    monkeypatch.setattr(
        cli, "stop_server", lambda h, **k: calls.append("server.stop")
    )
    monkeypatch.setattr(
        cli, "kv_cache_mb", lambda text: calls.append("kv") or None
    )  # Review #6: ต้องถูกเรียกหลัง stop (log ครบแล้ว)

    real_write = cli.write_report

    def recording_write(data, path):
        calls.append("report")
        reports.append(data)
        return real_write(data, path)

    monkeypatch.setattr(cli, "write_report", recording_write)
    # eval setup — ไม่ต้อง record (เกิดก่อน loop): ให้ผ่านเฉย ๆ
    monkeypatch.setattr(
        cli, "build_eval_cases", lambda **k: ["case"] * int(k["n_cases"])
    )
    monkeypatch.setattr(
        cli, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: object())
    )
    monkeypatch.setattr(
        cli, "report_path", lambda s, v: tmp_path / "benchmarks" / f"{v}.json"
    )
    return src, gguf


def test_variants_flag_parses_comma():
    cli = _load_cli()
    parser = cli.build_parser()

    args = parser.parse_args(["--model", "x", "--variants", "fp16,q4_k_m"])
    assert args.variants == ("fp16", "q4_k_m")

    defaults = parser.parse_args(["--model", "x"])
    assert defaults.variants == DEFAULT_VARIANTS
    assert defaults.device == "vulkan"
    assert defaults.eval_cases == 100


def test_run_benchmark_orchestration_order(monkeypatch, tmp_path):
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)

    args = cli.build_parser().parse_args(
        ["--model", "m", "--variants", "fp16,q4_k_m"]
    )
    rows = cli.run_benchmark(args)

    per_variant = [
        "artifact",
        "bench",
        "server.start",
        "eval",
        "server.stop",
        "kv",  # Review #6: parse หลัง stop+join เท่านั้น
        "report",
    ]
    assert calls == per_variant * 2
    assert len(rows) == 2


def test_run_benchmark_eval_disabled(monkeypatch, tmp_path, capsys):
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)

    args = cli.build_parser().parse_args(
        ["--model", "m", "--variants", "fp16", "--no-eval"]
    )
    rows = cli.run_benchmark(args)

    assert not any(c in {"eval", "server.start", "server.stop"} for c in calls)
    assert rows[0]["exact_match_pct"] is None
    assert rows[0]["token_f1"] is None
    assert rows[0]["latency_ms"] is None
    assert rows[0]["kv_cache_mb"] is None
    assert "FP16" in capsys.readouterr().out  # table ยังพิมพ์เสมอ


def test_table_printed_with_baseline_delta(monkeypatch, tmp_path, capsys):
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"exact_match_pct": 10.0, "token_f1_mean": 0.5, "n": 100}),
        encoding="utf-8",
    )
    captured: dict = {}
    monkeypatch.setattr(
        cli,
        "format_table",
        lambda rows, baseline=None: captured.update(rows=rows, baseline=baseline)
        or "TABLE-SENTINEL",
    )

    args = cli.build_parser().parse_args(
        ["--model", "m", "--variants", "fp16", "--no-eval", "--baseline", str(baseline_path)]
    )
    cli.run_benchmark(args)

    assert captured["baseline"] == {"exact_match_pct": 10.0, "token_f1_mean": 0.5, "n": 100}
    assert len(captured["rows"]) == 1
    assert "TABLE-SENTINEL" in capsys.readouterr().out  # print เกิดจริง


def test_missing_model_flag_exits_english(capsys):
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--model" in err
    assert "usage:" in err.lower()


def test_unavailable_parameter_count_is_null(monkeypatch, tmp_path, capsys):
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)

    def raise_unavailable(m, u):
        raise cli.ModelSpecUnavailable("model spec unavailable")

    monkeypatch.setattr(cli, "resolve_model_spec", raise_unavailable)

    args = cli.build_parser().parse_args(["--model", "m", "--variants", "fp16", "--no-eval"])
    rows = cli.run_benchmark(args)

    assert rows[0]["parameter_count"] is None
    warn = capsys.readouterr().err
    assert "parameter count unavailable" in warn  # warning เป็นอังกฤษ


@pytest.mark.integration
def test_compression_pipeline_e2e(monkeypatch, tmp_path):
    """หัวใจ DoD: คำสั่งเดียว → gguf ครบ + report §4 + eval รันได้จริง"""
    from core.compression import config as comp_config

    cli = _load_cli()
    try:
        comp_config.require_tools()
    except comp_config.CompressionError:
        pytest.skip("llama.cpp not built")
    try:
        src = comp_config.resolve_source("Qwen/Qwen2.5-Coder-0.5B")
    except comp_config.CompressionError:
        pytest.skip("Qwen model not downloaded")

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "report_path", lambda s, v: reports_dir / f"{v}.json")

    args = cli.build_parser().parse_args(
        [
            "--model",
            "Qwen/Qwen2.5-Coder-0.5B",
            "--variants",
            "fp16,q4_k_m",
            "--eval-cases",
            "2",
        ]
    )
    rows = cli.run_benchmark(args)

    assert len(rows) == 2
    fp16 = comp_config.artifact_path(src, "fp16")
    q4 = comp_config.artifact_path(src, "q4_k_m")
    assert fp16.is_file() and q4.is_file()
    assert q4.stat().st_size < fp16.stat().st_size

    assert sorted(p.name for p in reports_dir.glob("*.json")) == [
        "fp16.json",
        "q4_k_m.json",
    ]

    expected_keys = {
        "model", "variant", "optimization", "backend", "weight_bits",
        "context_tokens", "parameter_count", "model_disk_mb", "peak_vram_mb",
        "kv_cache_mb", "tokens_per_sec", "latency_ms", "load_time_ms",
        "exact_match_pct", "token_f1", "syntax_pass_rate", "execution_pass_rate",
        "timestamp",
    }
    for row in rows:
        assert expected_keys <= set(row)
        assert row["exact_match_pct"] is not None  # eval รันจริง
        assert row["syntax_pass_rate"] is None and row["execution_pass_rate"] is None
    assert rows[1]["model_disk_mb"] < rows[0]["model_disk_mb"]


# --- Review #2 + #10: eval identity + empty cases guard ---


def test_empty_eval_cases_errors_before_server(monkeypatch, tmp_path, capsys):
    """Review #10: build_eval_cases ว่าง (dataset หมด / --eval-cases 0) → error ก่อน start server

    ห้ามรายงาน 0.0 ดูเหมือน score ที่วัดได้จริง
    """
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)
    monkeypatch.setattr(cli, "build_eval_cases", lambda **k: [])

    rc = cli.main(["--model", "m", "--variants", "fp16"])

    assert rc == 1
    assert "no eval cases" in capsys.readouterr().err
    assert calls == []  # ไม่ build/ไม่ start server อะไรเลย (fail fast)


def test_report_persists_eval_identity(monkeypatch, tmp_path):
    """Review #2: report ต้องเก็บ identity ของ eval (dataset/column/limit/cases/fim)"""
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)

    args = cli.build_parser().parse_args(
        ["--model", "m", "--variants", "fp16", "--eval-cases", "7"]
    )
    cli.run_benchmark(args)

    identity = reports[0]["eval"]
    assert identity["built_cases"] == 7
    assert identity["requested_cases"] == 7
    assert identity["dataset_id"] == args.dataset
    assert identity["dataset_column"] == args.column
    assert identity["limit"] == args.limit
    assert identity["fim_key"] == args.fim_key


def test_delta_omitted_when_case_count_differs(monkeypatch, tmp_path, capsys):
    """Review #2: baseline คนละจำนวนเคส → ห้ามแสดง delta (คนละการทดสอบ)"""
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"exact_match_pct": 10.0, "token_f1_mean": 0.5, "n": 50}),
        encoding="utf-8",
    )
    captured: dict = {}
    monkeypatch.setattr(
        cli,
        "format_table",
        lambda rows, baseline=None: captured.update(baseline=baseline) or "T",
    )

    args = cli.build_parser().parse_args(
        ["--model", "m", "--variants", "fp16", "--baseline", str(baseline_path)]
    )  # eval เปิด → build_eval_cases fake คืน 100 เคส (n_cases default)
    cli.run_benchmark(args)

    assert captured["baseline"] is None
    err = capsys.readouterr().err
    assert "baseline skipped" in err and "case count" in err  # note เป็นอังกฤษ


def test_delta_shown_when_case_count_matches(monkeypatch, tmp_path):
    """equivalent (n ตรงกัน) → delta ผ่านเข้าไปใน format_table"""
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"exact_match_pct": 10.0, "token_f1_mean": 0.5, "n": 100}),
        encoding="utf-8",
    )
    captured: dict = {}
    monkeypatch.setattr(
        cli,
        "format_table",
        lambda rows, baseline=None: captured.update(baseline=baseline) or "T",
    )

    args = cli.build_parser().parse_args(
        ["--model", "m", "--variants", "fp16", "--baseline", str(baseline_path)]
    )
    cli.run_benchmark(args)

    assert captured["baseline"]["exact_match_pct"] == 10.0


def test_start_server_receives_context_arg(monkeypatch, tmp_path):
    """Review #7: CLI ต้องส่ง args.context เป็น ctx_size ให้ server (reported = applied)"""
    cli = _load_cli()
    calls: list[str] = []
    reports: list[dict] = []
    _wire(cli, monkeypatch, tmp_path, calls, reports)
    captured_kwargs: dict = {}
    fake_handle = SimpleNamespace(
        url="http://127.0.0.1:18080", alias="x", proc=None, log_text="", _reader=None
    )
    monkeypatch.setattr(
        cli,
        "start_server",
        lambda g, **k: captured_kwargs.update(k) or fake_handle,
    )

    args = cli.build_parser().parse_args(["--model", "m", "--variants", "fp16"])
    cli.run_benchmark(args)

    assert captured_kwargs.get("ctx_size") == args.context
