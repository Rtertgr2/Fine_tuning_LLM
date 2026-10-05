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
    DEFAULT_OUTPUT_DIR as _DEFAULT_OUTPUT_DIR,
    EVAL_N_CASES,
    LORA_RANK_DEFAULT,
    MAX_SEQ_LENGTH_DEFAULT,
    MAX_STEPS,
    TRAIN_CODE_LIMIT,
)
from core.eval import evaluator as ev

# M9: gate ต้องมี min-delta — +0.0 (tie) ห้าม PASS: ต้องดีขึ้นอย่างมีนัยทุก metric
# EM: n=100 → 1 case = 1.0pt — 0.1pt = ต่ำกว่า 1 case ก็ยังยอมรับได้ แต่ 0.0 ไม่ผ่าน
# F1: ตรง display precision ของรายงาน (ทศนิยม 2 ตำแหน่ง)
MIN_DELTA_EM_PCT: float = 0.1
MIN_DELTA_F1: float = 0.01


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

    if base.get("f1_kind") != fine.get("f1_kind"):
        print(
            f"Cannot compare: metric semantics differ (base f1_kind={base.get('f1_kind')!r} "
            f"vs finetuned f1_kind={fine.get('f1_kind')!r}) — rerun both modes with the current build",
            file=sys.stderr,
        )
        return 2
    if (base.get("dataset_id"), base.get("dataset_column")) != (
        fine.get("dataset_id"),
        fine.get("dataset_column"),
    ):
        print(
            f"Cannot compare: eval sets come from different datasets "
            f"(base {base.get('dataset_id')!r}/{base.get('dataset_column')!r} vs "
            f"finetuned {fine.get('dataset_id')!r}/{fine.get('dataset_column')!r}) — "
            "rerun both modes on the same dataset",
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

    # M9: PASS ต้องดีขึ้นเกิน min-delta ทุก metric (เดิม >= ทำให้ tie/+0.0 ผ่าน = gate ไร้ความหมาย)
    # round 6dp: ค่า float (10.1-10.0 = 0.0999…) ห้ามหลุด boundary "เท่ากับ min = ผ่าน"
    em_delta = round(fine["exact_match_pct"] - base["exact_match_pct"], 6)
    f1_delta = round(fine["token_f1_mean"] - base["token_f1_mean"], 6)
    passed = em_delta >= MIN_DELTA_EM_PCT and f1_delta >= MIN_DELTA_F1
    if passed:
        print(
            f"\nPASS — fine-tuned beats base by ≥ min-delta "
            f"(EM +{MIN_DELTA_EM_PCT}pt, F1 +{MIN_DELTA_F1}) on all metrics"
        )
        return 0
    # spec §3.3: FAIL ต้องบอกชื่อ metric — แยก regression (base ดีกว่า) กับ stall (ต่ำกว่า min-delta)
    parts = []
    regressed, stalled = [], []
    if em_delta < 0:
        regressed.append("Exact Match %")
    elif em_delta < MIN_DELTA_EM_PCT:
        stalled.append("Exact Match %")
    if f1_delta < 0:
        regressed.append("Token F1")
    elif f1_delta < MIN_DELTA_F1:
        stalled.append("Token F1")
    if regressed:
        parts.append(f"base better on: {', '.join(regressed)}")
    if stalled:
        parts.append(f"improvement below min-delta on: {', '.join(stalled)}")
    print(f"\nFAIL — {'; '.join(parts)}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="FIM evaluation: Exact Match + Token F1 (base vs fine-tuned)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=["base", "finetuned"], help="run one eval mode")
    group.add_argument("--compare", action="store_true", help="compare saved results")
    parser.add_argument("--n-cases", type=int, default=EVAL_N_CASES, help="eval set size (default 100)")
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
            # spec §3.3: พิมพ์ progress ทุก 10 cases (gating อยู่ใน evaluator — ทุก consumer ได้เหมือนกัน)
            progress=lambda done, total: print(
                f"Evaluating {args.mode}: {done}/{total}", flush=True
            ),
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
