"""Evaluator — FIM Exact Match + Token F1 (spec §3.2)

logic ล้วน ไม่โหลด model เองนอกจาก `run_eval` — ใช้ร่วมโดย
`eval.py` (CLI) และ `ui/controller` (ผ่าน `run_eval_worker`)
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from configs.safe_defaults import (
    EVAL_DIR,
    EVAL_N_CASES,
    F1_KIND,
    MAX_SEQ_LENGTH_DEFAULT,
    MIN_SAMPLE_LINES,
    SEED,
)
from core.data.dataset_builder import (
    _cut_positions,
    is_heldout,
    iter_codes,
    split_fim,
    truncate_to_tokens,
)

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
    """F1 บน longest common subsequence ของ token ids (M10: order mattered จริง ๆ)

    ทั้งคู่ว่าง = 1.0, ฝั่งเดียวว่าง = 0.0, overlap 0 = 0.0 (คง contract เดิม)
    """
    pred_ids = tokenizer.encode(pred, add_special_tokens=False)
    gt_ids = tokenizer.encode(gt, add_special_tokens=False)
    if not pred_ids and not gt_ids:
        return 1.0
    if not pred_ids or not gt_ids:
        return 0.0
    lcs = _lcs_length(pred_ids, gt_ids)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_ids)
    recall = lcs / len(gt_ids)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[int], b: list[int]) -> int:
    """LCS length (standard DP) — ทั้งสองฝั่ง bounded ที่ EVAL_MAX_NEW_TOKENS (256) → ≤ 65k cells"""
    if len(a) > len(b):
        a, b = b, a  # inner loop สั้นฝั่ง — LCS สมมาตร
    prev = [0] * (len(a) + 1)
    for bj in b:
        cur = [0]
        for i, ai in enumerate(a, 1):
            cur.append(prev[i - 1] + 1 if ai == bj else max(prev[i], cur[-1]))
        prev = cur
    return prev[-1]


def build_eval_cases(
    *,
    dataset_id: str,
    dataset_column: str,
    limit: int,
    tokenizer,
    n_cases: int = EVAL_N_CASES,
    max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT,
    seed: int = SEED,
) -> tuple[list[EvalCase], int]:
    """สร้างชุด eval จาก heldout เท่านั้น — criteria เดียวกับ `build_samples`

    - rng ตัวเดียว seed ด้วย `seed` → ผลซ้ำได้
    - middle เกิน `EVAL_MAX_NEW_TOKENS` → ข้าม (คำตอบยาวกว่าที่ model จะ generate)
    - เลิกก่อนครบ `n_cases` ได้ (heldout หมด stream) — คืนเท่าที่มี ไม่ raise
    - คืน `(cases, skipped_long_middle)` — skipped ต้องรายงาน (เงียบ = EM ถูกอ่าน
      ว่า representative ทั้งที่ตัดเคส span ยาวทิ้ง)
    """
    codes = iter_codes(dataset_id, dataset_column, limit=limit)
    rng = random.Random(seed)
    cases: list[EvalCase] = []
    skipped = 0
    for code in codes:
        if not is_heldout(code):
            continue
        if len(code.splitlines()) < MIN_SAMPLE_LINES or len(_cut_positions(code)) < 2:
            continue
        prefix, suffix, middle = split_fim(code, rng)
        if len(tokenizer.encode(middle, add_special_tokens=False)) > EVAL_MAX_NEW_TOKENS:
            skipped += 1
            continue
        # suffix ตัด head (คงเดิม), prefix ตัด tail — เก็บบริบทถัดจาก middle
        # ตรง build_samples (train/eval ต้องหด prefix แบบเดียวกัน)
        suffix = truncate_to_tokens(suffix, tokenizer, max_seq_length // 2)
        prefix = truncate_to_tokens(
            prefix, tokenizer, max_seq_length - max_seq_length // 2, keep="tail"
        )
        cases.append(EvalCase(prefix=prefix, suffix=suffix, middle=middle))
        if len(cases) >= n_cases:
            break
    return cases, skipped


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
    # lazy import: --compare ห้ามดึง torch/transformers (D10)
    from core.data.fim import build_fim_prompt, decode_continuation, encode_prompt

    per_case: list[dict] = []
    total = len(cases)
    for i, case in enumerate(cases):
        prompt = build_fim_prompt(case.prefix, case.suffix, fim_tokens=fim_tokens)
        inputs = encode_prompt(tokenizer, prompt)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False
        )
        continuation = output_ids[0][inputs["input_ids"].shape[1] :]
        pred = decode_continuation(tokenizer, continuation, fim_tokens=fim_tokens)
        per_case.append(
            {
                "i": i,
                "exact": exact_match(pred, case.middle),
                "f1": token_f1(pred, case.middle, tokenizer),
                "pred": pred,
                "gt": case.middle,
            }
        )
        # spec §3.3: progress ทุก 10 cases + ครั้งสุดท้ายเสมอ (กัน flood log ทั้ง CLI และ UI worker)
        if progress is not None and ((i + 1) % 10 == 0 or i + 1 == total):
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
    n_cases: int = EVAL_N_CASES,
    eval_dir: Path | str = EVAL_DIR,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """รัน eval 1 mode → เขียน `eval_dir/<mode>.json` → คืน result dict

    fail-fast: validate → checkpoint (finetuned) → tokenizer → cases → model
    โหลดโมเดลทีละ 1 ตัวเท่านั้น (caller ต้องไม่รัน 2 mode ขนาน — §5 VRAM)
    """
    # lazy import: --compare ห้ามดึง torch/transformers (D10)
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from core.data.fim import ensure_fim_tokens
    from core.train.args import validate_config
    from core.train.runner import latest_checkpoint

    if mode not in ("base", "finetuned"):
        raise ValueError(f'mode must be "base" or "finetuned", got {mode!r}')
    validate_config(config)
    registry_path = Path(__file__).resolve().parents[2] / "configs" / "fim_registry.json"
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
    cases, skipped_long_middle = build_eval_cases(
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
        use_safetensors=True,  # L4: ปฏิเสธ .bin (pickle) เสมอ
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
        "skipped_long_middle": skipped_long_middle,
        "exact_match_pct": metrics["exact_match_pct"],
        "token_f1_mean": metrics["token_f1_mean"],
        "per_case": metrics["per_case"],
        "samples": metrics["per_case"][:5],
        # review#5: identity markers — กันเทียบข้าม metric version/ชุดข้อมูล (Task 15)
        "f1_kind": F1_KIND,
        "dataset_id": config["dataset_id"],
        "dataset_column": config["dataset_column"],
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


def check_comparable(base: dict, finetuned: dict) -> None:
    """identity gate ร่วมของ CLI (`eval.py --compare`) และ UI (dashboard compare) — ไม่ตรง → raise

    - `f1_kind` ของ **ทั้งสองรายงาน** ต้องเท่ากับ `F1_KIND` ปัจจุบัน — legacy ไม่มี key
      (None == None) เดิมผ่าน check แบบเทียบสองรายงานด้วยกัน → ต้อง reject (คนละ build = คนละความหมาย)
    - `(dataset_id, dataset_column)` ต้องตรงกัน — คนละชุดข้อมูล = delta โกหก
    ข้อความคงแบบเดิมที่ CLI เคยพิมพ์ (test แชร์ assertion ไว้)
    """
    if base.get("f1_kind") != F1_KIND or finetuned.get("f1_kind") != F1_KIND:
        raise ValueError(
            f"Cannot compare: metric semantics differ (base f1_kind={base.get('f1_kind')!r} "
            f"vs finetuned f1_kind={finetuned.get('f1_kind')!r}) — "
            "rerun both modes with the current build"
        )
    if (base.get("dataset_id"), base.get("dataset_column")) != (
        finetuned.get("dataset_id"),
        finetuned.get("dataset_column"),
    ):
        raise ValueError(
            "Cannot compare: eval sets come from different datasets "
            f"(base {base.get('dataset_id')!r}/{base.get('dataset_column')!r} vs "
            f"finetuned {finetuned.get('dataset_id')!r}/{finetuned.get('dataset_column')!r}) — "
            "rerun both modes on the same dataset"
        )


def compare_results(base: dict, finetuned: dict) -> tuple[list[list[str]], list[dict]]:
    """เปรียบเทียบ 2 JSON → (table rows, ตัวอย่าง base ผิด → fine ถูก up to 5)

    rows: [label, base, finetuned, delta] — EM ทศนิยม 1, F1 ทศนิยม 2
    qualitative: จับคู่ per_case ด้วย "i" — base exact=False และ finetuned exact=True
    identity gate: f1_kind/dataset ไม่ผ่าน `check_comparable` → raise (CLI จับเป็น rc 2,
    dashboard จับเป็น error banner — UI ห้าม render ผลที่เทียบไม่ได้เหมือน CLI)
    """
    check_comparable(base, finetuned)
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
