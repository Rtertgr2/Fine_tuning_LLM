"""eval.py — CLI ประเมิน FIM (Exact Match + Token F1) เทียบ base vs finetuned (spec §3.3)

ใช้:
  .venv/bin/python eval.py --mode base        # รัน eval บนโมเดล base → data_cache/eval/base.json
  .venv/bin/python eval.py --mode finetuned   # รัน eval บน base+LoRA → data_cache/eval/finetuned.json
  .venv/bin/python eval.py --compare          # เทียบผล 2 ไฟล์ → PASS/FAIL (exit 0/1)

exit codes: --mode 0=สำเร็จ 1=ผิดพลาด; --compare 0=PASS 1=FAIL 2=ข้อมูลไม่ครบ/เทียบไม่ได้
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from configs.safe_defaults import (
    DEFAULT_DATASET_COLUMN,
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    LORA_RANK_DEFAULT,
    MAX_SEQ_LENGTH_DEFAULT,
    MAX_STEPS,
    TRAIN_CODE_LIMIT,
)
from core import evaluator as ev

_DEFAULT_OUTPUT_DIR = "data_cache/finetune_run"  # spec §4 table default (ตรงกับ UI)


def _build_config(args: argparse.Namespace) -> dict:
    """config ครบทุก key ใน REQUIRED_CONFIG_KEYS — ค่าอื่นมาจาก safe_defaults"""
    return {
        "model_id": DEFAULT_MODEL_ID,
        "dataset_id": DEFAULT_DATASET_ID,
        "dataset_column": DEFAULT_DATASET_COLUMN,
        "fim_registry_key": args.fim_key,
        "output_dir": args.output_dir,
        "max_seq_length": MAX_SEQ_LENGTH_DEFAULT,
        "max_steps": MAX_STEPS,
        "code_limit": TRAIN_CODE_LIMIT,
        "lora_rank": LORA_RANK_DEFAULT,
    }


def _run_compare(eval_dir: Path) -> int:
    results: dict[str, dict] = {}
    for mode in ("base", "finetuned"):
        path = eval_dir / f"{mode}.json"
        if not path.exists():
            print(
                f"Missing {path} — run --mode {mode} first",
                file=sys.stderr,
            )
            return 2
        results[mode] = json.loads(path.read_text(encoding="utf-8"))

    base, fine = results["base"], results["finetuned"]
    if base.get("n") != fine.get("n"):
        print(
            f"Cannot compare: eval sets differ (base n={base.get('n')} "
            f"vs finetuned n={fine.get('n')}) — rerun both modes with same --n-cases",
            file=sys.stderr,
        )
        return 2

    rows, qualitative = ev.compare_results(base, fine)
    print(f"{'Metric':<14} {'Base':>8} {'Fine-tuned':>11} {'Δ':>8}")
    for label, b, f, delta in rows:
        print(f"{label:<14} {b:>8} {f:>11} {delta:>8}")

    if qualitative:
        print("\nExamples base could not guess → fine-tuned got right:")
        for q in qualitative:
            print(f"  [{q['i']}] gt:      {q['gt']!r}")
            print(f"       base:    {q['base_pred']!r}")
            print(f"       tuned:   {q['finetuned_pred']!r}")

    passed = (
        fine["exact_match_pct"] >= base["exact_match_pct"]
        and fine["token_f1_mean"] >= base["token_f1_mean"]
    )
    print("\nPASS — fine-tuned ≥ base on all metrics" if passed else "\nFAIL — base better on some metric")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="FIM evaluation: Exact Match + Token F1 (base vs fine-tuned)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=["base", "finetuned"], help="run one eval mode")
    group.add_argument("--compare", action="store_true", help="compare saved results")
    parser.add_argument("--n-cases", type=int, default=100, help="eval set size (default 100)")
    parser.add_argument("--eval-dir", type=Path, default=ev.EVAL_DIR, help="where JSON results live")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR, help="training output dir (finetuned mode)")
    parser.add_argument("--fim-key", default="qwen", help="FIM registry key (default qwen)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse error/help → main คืน int เสมอ (test เรียก main ได้)
        return int(exc.code or 0)

    if args.compare:
        return _run_compare(Path(args.eval_dir))

    try:
        result = ev.run_eval(
            _build_config(args),
            mode=args.mode,
            n_cases=args.n_cases,
            eval_dir=args.eval_dir,
        )
    except Exception as exc:  # noqa: BLE001 — CLI แปลงทุก exception เป็น exit 1
        print(f"eval failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"mode={result['mode']} n={result['n']} "
        f"exact_match={result['exact_match_pct']:.1f}% "
        f"token_f1={result['token_f1_mean']:.3f}"
    )
    if "warning" in result:
        print(f"warning: {result['warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
