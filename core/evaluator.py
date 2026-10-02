"""Evaluator — FIM Exact Match + Token F1 (spec §3.2)

logic ล้วน ไม่โหลด model เอง (โหลดใน `run_eval` เท่านั้น) — ใช้ร่วมโดย
`eval.py` (CLI) และ `ui/controller` (ผ่าน `run_eval_worker`)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
