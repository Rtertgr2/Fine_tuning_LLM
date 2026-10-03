#!/usr/bin/env python3
"""Compression benchmark CLI — คำสั่งเดียว: build artifacts → bench → eval → report

ตัวอย่าง:
    .venv/bin/python scripts/benchmark_compression.py \
        --model Qwen/Qwen2.5-Coder-0.5B --variants fp16,q8_0,q4_k_m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from configs.safe_defaults import (
    DEFAULT_DATASET_COLUMN,
    DEFAULT_DATASET_ID,
    MAX_SEQ_LENGTH_DEFAULT,
    TRAIN_CODE_LIMIT,
)
from core.compression import CompressionError
from core.compression.benchmark import run_bench
from core.compression.config import (
    DEFAULT_VARIANTS,
    VARIANT_BITS,
    report_path,
    require_tools,
    resolve_source,
)
from core.compression.llama_eval import evaluate_with_llama
from core.compression.llama_runner import kv_cache_mb, start_server, stop_server
from core.compression.quantizer import build_artifacts
from core.compression.report import (
    build_report,
    format_table,
    pick_baseline,
    write_report,
)
from core.estimator import ModelSpecUnavailable, resolve_model_spec
from core.evaluator import build_eval_cases
from core.trainer_worker import AutoTokenizer


def parse_variants(raw: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in raw.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_compression.py",
        description="Build GGUF variants and benchmark them with llama.cpp (report-only).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model under models/ or a path (e.g. Qwen/Qwen2.5-Coder-0.5B)",
    )
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=DEFAULT_VARIANTS,
        help="comma list (default: fp16,q8_0,q4_k_m)",
    )
    parser.add_argument(
        "--device", choices=("vulkan", "cpu"), default="vulkan", help="default: vulkan"
    )
    parser.add_argument(
        "--context", type=int, default=MAX_SEQ_LENGTH_DEFAULT, help="context tokens"
    )
    parser.add_argument(
        "--eval-cases", type=int, default=100, help="number of eval cases"
    )
    parser.add_argument(
        "--no-eval", action="store_true", help="skip llama-server eval (metrics = null)"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="eval dataset id")
    parser.add_argument("--column", default=DEFAULT_DATASET_COLUMN, help="dataset column")
    parser.add_argument(
        "--limit", type=int, default=TRAIN_CODE_LIMIT, help="code length limit"
    )
    parser.add_argument("--fim-key", default="qwen", help="FIM token registry key")
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline eval json path (default: data_cache/eval pick by model)",
    )
    return parser


def _parameter_count(model_id: str) -> int | None:
    try:
        return resolve_model_spec(model_id, None).num_params
    except ModelSpecUnavailable as exc:
        print(
            f"warning: parameter count unavailable for {model_id} ({exc}) "
            "— report will use null",
            file=sys.stderr,
        )
        return None


def _load_fim(fim_key: str) -> dict:
    registry_path = Path(__file__).resolve().parents[1] / "configs" / "fim_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return registry[fim_key]


def _baseline_dict(args: argparse.Namespace, source: Path) -> dict | None:
    path = Path(args.baseline) if args.baseline else pick_baseline(args.model, source)
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(
            f"warning: baseline is not valid JSON: {path} — continuing without baseline",
            file=sys.stderr,
        )
        return None


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    """ทุก variant: build → bench → (eval ผ่าน server) → report — จบด้วยตารางสรุป"""
    source = resolve_source(args.model)
    require_tools()
    parameter_count = _parameter_count(args.model)

    tokenizer = None
    fim_tokens = None
    cases: list = []
    if not args.no_eval:
        fim_tokens = _load_fim(args.fim_key)
        tokenizer = AutoTokenizer.from_pretrained(str(source))
        cases = build_eval_cases(
            dataset_id=args.dataset,
            dataset_column=args.column,
            limit=args.limit,
            tokenizer=tokenizer,
            n_cases=args.eval_cases,
            max_seq_length=args.context,
        )

    rows: list[dict] = []
    for variant in args.variants:
        gguf = build_artifacts(source, (variant,))[0]
        bench = run_bench(gguf, device=args.device)

        kv = None
        eval_result = None
        if not args.no_eval:
            handle = start_server(gguf, device=args.device)
            try:
                eval_result = evaluate_with_llama(
                    cases, handle, tokenizer=tokenizer, fim_tokens=fim_tokens
                )
            finally:
                kv = kv_cache_mb(handle.log_text)
                stop_server(handle)

        latency = None
        if eval_result is not None:
            latencies = [c["latency_ms"] for c in eval_result["per_case"]]
            latency = sum(latencies) / len(latencies) if latencies else None

        report = build_report(
            model=args.model,
            variant=variant,
            backend=f"llama.cpp-{args.device}",
            weight_bits=VARIANT_BITS[variant],
            parameter_count=parameter_count,
            context_tokens=args.context,
            model_disk_mb=gguf.stat().st_size / (1024 * 1024),
            tokens_per_sec=bench["prompt_tps"],
            latency_ms=latency,
            load_time_ms=bench["load_ms"],
            peak_vram_mb=bench["peak_vram_mb"],
            kv_cache_mb=kv,
            exact_match_pct=eval_result["exact_match_pct"] if eval_result else None,
            token_f1=eval_result["token_f1_mean"] if eval_result else None,
        )
        rows.append(report)
        write_report(report, report_path(source, variant))

    baseline = _baseline_dict(args, source)
    print(format_table(rows, baseline))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_benchmark(args)
    except CompressionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
