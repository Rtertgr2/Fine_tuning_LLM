"""Evaluator — FIM Exact Match + Token F1 (spec §3.2)

logic ล้วน ไม่โหลด model เองนอกจาก `run_eval` — ใช้ร่วมโดย
`eval.py` (CLI) และ `ui/controller` (ผ่าน `run_eval_worker`)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from configs.safe_defaults import (
    MAX_SEQ_LENGTH_DEFAULT,
    MIN_SAMPLE_LINES,
    SEED,
)
from core.dataset_builder import (
    _cut_positions,
    is_heldout,
    iter_codes,
    split_fim,
    truncate_to_tokens,
)
from core.trainer_worker import (
    AutoModelForCausalLM,
    AutoTokenizer,
    build_fim_prompt,
    ensure_fim_tokens,
    latest_checkpoint,
    validate_config,
)

EVAL_DIR = Path("data_cache/eval")
EVAL_MAX_NEW_TOKENS: int = 256


@dataclass(frozen=True)
class EvalCase:
    """FIM case 1 ชุด: prefix/suffix เข้า prompt, middle คือคำตอบอ้างอิง"""

    prefix: str
    suffix: str
    middle: str


def exact_match(pred: str, gt: str) -> bool:
    """เทียบเป๊ะหลัง normalize CRLF→LF + strip หัวท้าย"""
    return pred.replace("\r\n", "\n").strip() == gt.replace("\r\n", "\n").strip()


def token_f1(pred: str, gt: str, tokenizer) -> float:
    """F1 บน multiset ของ token ids — ทั้งคู่ว่าง = 1.0, ฝั่งเดียวว่าง = 0.0"""
    pred_ids = Counter(tokenizer.encode(pred, add_special_tokens=False))
    gt_ids = Counter(tokenizer.encode(gt, add_special_tokens=False))
    if not pred_ids and not gt_ids:
        return 1.0
    if not pred_ids or not gt_ids:
        return 0.0
    overlap = sum((pred_ids & gt_ids).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_ids.values())
    recall = overlap / sum(gt_ids.values())
    return 2 * precision * recall / (precision + recall)


def build_eval_cases(
    *,
    dataset_id: str,
    dataset_column: str,
    limit: int,
    tokenizer,
    n_cases: int = 100,
    max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT,
    seed: int = SEED,
) -> list[EvalCase]:
    """สร้างชุด eval จาก heldout เท่านั้น — criteria เดียวกับ `build_samples`

    - rng ตัวเดียว seed ด้วย `seed` → ผลซ้ำได้
    - middle เกิน `EVAL_MAX_NEW_TOKENS` → ข้าม (คำตอบยาวกว่าที่ model จะ generate)
    - เลิกก่อนครบ `n_cases` ได้ (heldout หมด stream) — คืนเท่าที่มี ไม่ raise
    """
    codes = iter_codes(dataset_id, dataset_column, limit=limit)
    rng = random.Random(seed)
    cases: list[EvalCase] = []
    for code in codes:
        if not is_heldout(code):
            continue
        if len(code.splitlines()) < MIN_SAMPLE_LINES or len(_cut_positions(code)) < 2:
            continue
        prefix, suffix, middle = split_fim(code, rng)
        if len(tokenizer.encode(middle, add_special_tokens=False)) > EVAL_MAX_NEW_TOKENS:
            continue
        suffix = truncate_to_tokens(suffix, tokenizer, max_seq_length // 2)
        prefix = truncate_to_tokens(prefix, tokenizer, max_seq_length - max_seq_length // 2)
        cases.append(EvalCase(prefix=prefix, suffix=suffix, middle=middle))
        if len(cases) >= n_cases:
            break
    return cases


def evaluate_cases(
    model,
    tokenizer,
    cases: list[EvalCase],
    *,
    fim_tokens: dict,
    device: str,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """generate middle ต่อ case แบบ greedy → คำนวณ Exact Match + Token F1"""
    per_case: list[dict] = []
    total = len(cases)
    for i, case in enumerate(cases):
        prompt = build_fim_prompt(case.prefix, case.suffix, fim_tokens=fim_tokens)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False
            )
        continuation = output_ids[0][inputs["input_ids"].shape[1] :]
        pred = tokenizer.decode(continuation, skip_special_tokens=True)
        per_case.append(
            {
                "i": i,
                "exact": exact_match(pred, case.middle),
                "f1": token_f1(pred, case.middle, tokenizer),
                "pred": pred,
                "gt": case.middle,
            }
        )
        if progress is not None:
            progress(i + 1, total)
    return {
        "per_case": per_case,
        "exact_match_pct": 100.0 * sum(c["exact"] for c in per_case) / total
        if total
        else 0.0,
        "token_f1_mean": sum(c["f1"] for c in per_case) / total if total else 0.0,
    }


def run_eval(
    config: dict,
    *,
    mode: Literal["base", "finetuned"],
    n_cases: int = 100,
    eval_dir: Path | str = EVAL_DIR,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """รัน eval 1 mode → เขียน `eval_dir/<mode>.json` → คืน result dict

    fail-fast: validate → checkpoint (finetuned) → tokenizer → cases → model
    โหลดโมเดลทีละ 1 ตัวเท่านั้น (caller ต้องไม่รัน 2 mode ขนาน — §5 VRAM)
    """
    if mode not in ("base", "finetuned"):
        raise ValueError(f'mode must be "base" or "finetuned", got {mode!r}')
    validate_config(config)
    registry_path = Path(__file__).resolve().parents[1] / "configs" / "fim_registry.json"
    fim_tokens = json.loads(registry_path.read_text(encoding="utf-8"))[
        config["fim_registry_key"]
    ]

    adapter_dir: Path | None = None
    if mode == "finetuned":
        try:
            adapter_dir = latest_checkpoint(config["output_dir"])
        except ValueError:
            raise FileNotFoundError(
                f"No checkpoint found in {config['output_dir']} — train first"
            ) from None

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    ensure_fim_tokens(tokenizer, fim_tokens.values())
    cases = build_eval_cases(
        dataset_id=config["dataset_id"],
        dataset_column=config["dataset_column"],
        limit=config["code_limit"],
        tokenizer=tokenizer,
        n_cases=n_cases,
        max_seq_length=config["max_seq_length"],
    )
    if not cases:
        raise ValueError(
            "no eval cases built — heldout stream empty after filters "
            f"(dataset_id={config['dataset_id']}, limit={config['code_limit']})"
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    device = "xpu" if torch.xpu.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    metrics = evaluate_cases(
        model, tokenizer, cases, fim_tokens=fim_tokens, device=device, progress=progress
    )
    result: dict = {
        "mode": mode,
        "n": len(cases),
        "exact_match_pct": metrics["exact_match_pct"],
        "token_f1_mean": metrics["token_f1_mean"],
        "per_case": metrics["per_case"],
        "samples": metrics["per_case"][:5],
    }
    if len(cases) < n_cases:
        result["warning"] = (
            f"only {len(cases)} eval cases available (requested {n_cases})"
        )
    out_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mode}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def compare_results(base: dict, finetuned: dict) -> tuple[list[list[str]], list[dict]]:
    """เปรียบเทียบ 2 JSON → (table rows, ตัวอย่าง base ผิด → fine ถูก up to 5)

    rows: [label, base, finetuned, delta] — EM ทศนิยม 1, F1 ทศนิยม 2
    qualitative: จับคู่ per_case ด้วย "i" — base exact=False และ finetuned exact=True
    """
    b_em, f_em = base["exact_match_pct"], finetuned["exact_match_pct"]
    b_f1, f_f1 = base["token_f1_mean"], finetuned["token_f1_mean"]
    rows = [
        ["Exact Match %", f"{b_em:.1f}", f"{f_em:.1f}", f"{f_em - b_em:+.1f}"],
        ["Token F1", f"{b_f1:.2f}", f"{f_f1:.2f}", f"{f_f1 - b_f1:+.2f}"],
        ["Cases", str(base["n"]), str(finetuned["n"]), ""],
    ]
    fine_by_i = {c["i"]: c for c in finetuned.get("per_case", [])}
    qualitative: list[dict] = []
    for b_case in base.get("per_case", []):
        if len(qualitative) >= 5:
            break
        f_case = fine_by_i.get(b_case["i"])
        if f_case is not None and not b_case["exact"] and f_case["exact"]:
            qualitative.append(
                {
                    "i": b_case["i"],
                    "gt": b_case["gt"],
                    "base_pred": b_case["pred"],
                    "finetuned_pred": f_case["pred"],
                }
            )
    return rows, qualitative
