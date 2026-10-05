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

# รันเป็น `python scripts/benchmark_compression.py` → sys.path[0] = scripts/ ต้องเพิ่ม root ก่อน
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.safe_defaults import (
    DEFAULT_DATASET_COLUMN,
    DEFAULT_DATASET_ID,
    EVAL_N_CASES,
    F1_KIND,
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
from core.evaluator import EVAL_MAX_NEW_TOKENS, build_eval_cases
from core.trainer_worker import AutoTokenizer

# กันชน token พิเศษของ FIM prompt ที่อยู่เหนือ input budget (วัดจริง ≤6 โทเคน) —
# server ctx ต้องครอบคลุม theoretical worst (budget + specials + EVAL_MAX_NEW_TOKENS) เสมอ
FIM_PROMPT_MARGIN: int = 16

# M9-quant (D-1/D-2): quantized ต้องไม่แย่กว่า fp16 ในรันเดียวกันเกิน threshold — hard gate
QUANT_MAX_EM_DROP_PCT: float = 1.0
QUANT_MAX_F1_DROP: float = 0.02


def check_quant_regression(rows: list[dict]) -> list[str]:
    """คืน violation list — variant ที่ไม่ใช่ fp16 แย่กว่า fp16 (รันเดียวกัน = identity ตรง by construction)

    ไม่มี fp16 หรือไม่มี metrics (--no-eval) → [] (gate ข้าม — main พิมพ์ note)
    """
    fp16 = next(
        (r for r in rows
         if r.get("variant") == "fp16" and r.get("exact_match_pct") is not None),
        None,
    )
    if fp16 is None:
        return []
    out = []
    for r in rows:
        if r is fp16 or r.get("exact_match_pct") is None:
            continue
        em_drop = round(fp16["exact_match_pct"] - r["exact_match_pct"], 6)
        if em_drop > QUANT_MAX_EM_DROP_PCT:
            out.append(
                f"{r['variant']}: EM drop {em_drop:.1f}pt vs fp16 exceeds {QUANT_MAX_EM_DROP_PCT}pt"
            )
        if fp16.get("token_f1") is None or r.get("token_f1") is None:
            continue                      # build_report ค่า float | None — ข้ามถ้า eval ฝั่งไหนไม่มี
        # round 6dp: float (0.60-0.58 = 0.020000000000000018) ห้ามหลุด boundary "เท่ากับ threshold = ผ่าน"
        # — precedent เดียวกับ M9 min-delta ใน eval.py
        f1_drop = round(fp16["token_f1"] - r["token_f1"], 6)
        if f1_drop > QUANT_MAX_F1_DROP:
            out.append(
                f"{r['variant']}: token F1 drop {f1_drop:.2f} vs fp16 exceeds {QUANT_MAX_F1_DROP}"
            )
    return out


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
        "--eval-cases", type=int, default=EVAL_N_CASES, help="number of eval cases"
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


def _baseline_dict(
    args: argparse.Namespace, source: Path, built_cases: int | None
) -> dict | None:
    path = Path(args.baseline) if args.baseline else pick_baseline(args.model, source)
    if path is None or not path.is_file():
        return None
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(
            f"warning: baseline is not valid JSON: {path} — continuing without baseline",
            file=sys.stderr,
        )
        return None
    if built_cases is None:
        return baseline  # --no-eval: ไม่มี metrics ปัจจุบัน → delta ไม่ถูกแสดงอยู่ดี
    if baseline.get("n") != built_cases:
        # Review #2: จำนวนเคสไม่ตรง = คนละการทดสอบ → ห้ามเทียบ (ตรง convention eval.py --compare)
        print(
            f"note: baseline skipped (case count differs: baseline n={baseline.get('n')!r}, "
            f"run n={built_cases}) — deltas require the same evaluation",
            file=sys.stderr,
        )
        return None
    # #11 identity guard: dataset/column/f1_kind ไม่ตรง = คนละ eval → delta โกหก
    if baseline.get("dataset_id") != args.dataset or baseline.get("dataset_column") != args.column:
        print(
            f"note: baseline skipped (eval identity differs or missing: baseline="
            f"{baseline.get('dataset_id')!r}/{baseline.get('dataset_column')!r} vs run="
            f"{args.dataset!r}/{args.column!r}) — rerun eval with this build",
            file=sys.stderr,
        )
        return None
    if baseline.get("f1_kind") != F1_KIND:
        print(
            f"note: baseline skipped (f1_kind={baseline.get('f1_kind')!r} != {F1_KIND!r}) — "
            "rerun eval with the current build",
            file=sys.stderr,
        )
        return None
    return baseline


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    """ทุก variant: build → bench → (eval ผ่าน server) → report — จบด้วยตารางสรุป"""
    source = resolve_source(args.model)
    require_tools()
    parameter_count = _parameter_count(args.model)
    # Review #7: server ต้องรันด้วย context ที่รายงาน = input budget (args.context =
    # MAX_SEQ_LENGTH_DEFAULT ที่ build_eval_cases ใช้ truncate) + gen headroom เท่า HF eval
    # (EVAL_MAX_NEW_TOKENS) + margin กันชน FIM specials (prompt เหนือ budget วัดจริง ≤6 โทเคน;
    # theoretical worst 1024+6+256 = 1286 ต้อง ≤ ค่านี้) → ทุกเคส fit เท่า baseline
    server_ctx = args.context + EVAL_MAX_NEW_TOKENS + FIM_PROMPT_MARGIN

    tokenizer = None
    fim_tokens = None
    cases: list = []
    eval_identity = None
    if not args.no_eval:
        fim_tokens = _load_fim(args.fim_key)
        tokenizer = AutoTokenizer.from_pretrained(str(source))
        cases, _skipped = build_eval_cases(
            dataset_id=args.dataset,
            dataset_column=args.column,
            limit=args.limit,
            tokenizer=tokenizer,
            n_cases=args.eval_cases,
            max_seq_length=args.context,
        )
        if not cases:
            # Review #10: ห้ามรายงาน 0.0 ดูเหมือน score ที่วัดได้จริง → fail fast ก่อน start server
            raise CompressionError(
                "no eval cases built (dataset exhausted or --eval-cases 0) — "
                "use --no-eval to skip evaluation."
            )
        eval_identity = {
            "dataset_id": args.dataset,
            "dataset_column": args.column,
            "limit": args.limit,
            "requested_cases": args.eval_cases,
            "built_cases": len(cases),
            "fim_key": args.fim_key,
        }

    rows: list[dict] = []
    for variant in args.variants:
        gguf = build_artifacts(source, (variant,))[0]
        bench = run_bench(gguf, device=args.device)

        kv = None
        eval_result = None
        if not args.no_eval:
            handle = start_server(gguf, device=args.device, ctx_size=server_ctx)
            # eager parse ก่อน eval — วัดจริง eval 100 เคส ≈ 319k บรรทัด (3,156/เคส)
            # ≫ LOG_MAX_LINES → รอจนหลัง stop = startup kv line ถูก deque evict → None
            kv = kv_cache_mb(handle.log_text)
            try:
                eval_result = evaluate_with_llama(
                    cases, handle, tokenizer=tokenizer, fim_tokens=fim_tokens
                )
            finally:
                stop_server(handle)  # stop = join reader ในตัว (log ครบ — Review #6)

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
            context_tokens=server_ctx,  # reported = applied (Review #7)
            model_disk_mb=gguf.stat().st_size / (1024 * 1024),
            tokens_per_sec=bench["gen_tps"],  # Review #8: headline = decode rate
            prompt_tokens_per_sec=bench["prompt_tps"],  # pp เก็บแยก (อย่าเอา pp มาเป็น TPS)
            latency_ms=latency,
            load_time_ms=bench["load_ms"],
            peak_vram_mb=bench["peak_vram_mb"],
            kv_cache_mb=kv,
            exact_match_pct=eval_result["exact_match_pct"] if eval_result else None,
            token_f1=eval_result["token_f1_mean"] if eval_result else None,
            eval_identity=eval_identity,
        )
        rows.append(report)
        write_report(report, report_path(source, variant))

    baseline = _baseline_dict(
        args, source, None if args.no_eval else len(cases)
    )  # Review #2: เทียบก็ต่อเมื่อจำนวนเคสตรงกัน
    print(format_table(rows, baseline))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = run_benchmark(args)
    except CompressionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # M9-quant hard gate (D-1): exit 1 เมื่อ quantized แย่กว่า fp16 เกิน threshold
    violations = check_quant_regression(rows)
    if violations:
        print("FAIL — quant regression vs fp16 in the same run:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    if not args.no_eval and not any(r.get("variant") == "fp16" for r in rows):
        print("note: quant-regression gate skipped (fp16 not in this run)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
