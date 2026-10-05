"""FIM eval ผ่าน llama-server — reuse metrics ของ Phase 5 (EM + token F1) เป๊ะ

loop เดียวกับ `core.eval.evaluator.evaluate_cases` แต่แทน `model.generate` ด้วย
`server.complete` (จับเวลา `time.perf_counter` → `latency_ms` ต่อ case)
"""

from __future__ import annotations

import time
from collections.abc import Callable

from core.compress.llama_runner import ServerHandle
from core.eval.evaluator import (
    EVAL_MAX_NEW_TOKENS,
    EvalCase,
    exact_match,
    token_f1,
)
from core.data.fim import build_fim_prompt


def evaluate_with_llama(
    cases: list[EvalCase],
    server: ServerHandle,
    *,
    tokenizer,
    fim_tokens: dict,
    n_predict: int = EVAL_MAX_NEW_TOKENS,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """generate middle ต่อ case (greedy, temperature=0) → EM + Token F1 + latency

    prompt = `build_fim_prompt(...)` ส่งออกไป byte เดียวกัน (ไม่ strip/แต่งใหม่);
    progress ทุก 10 cases + ครั้งสุดท้ายเสมอ (กฎเดิมจาก spec §3.3)
    """
    per_case: list[dict] = []
    total = len(cases)
    for i, case in enumerate(cases):
        prompt = build_fim_prompt(case.prefix, case.suffix, fim_tokens=fim_tokens)
        start = time.perf_counter()
        pred = server.complete(prompt, n_predict=n_predict, temperature=0.0)
        latency_ms = (time.perf_counter() - start) * 1000.0
        per_case.append(
            {
                "i": i,
                "exact": exact_match(pred, case.middle),
                "f1": token_f1(pred, case.middle, tokenizer),
                "pred": pred,
                "gt": case.middle,
                "latency_ms": latency_ms,
            }
        )
        if progress is not None and ((i + 1) % 10 == 0 or i + 1 == total):
            progress(i + 1, total)
    return {
        "per_case": per_case,
        "exact_match_pct": 100.0 * sum(c["exact"] for c in per_case) / total
        if total
        else 0.0,
        "token_f1_mean": sum(c["f1"] for c in per_case) / total if total else 0.0,
    }
