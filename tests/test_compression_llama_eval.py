"""Unit tests สำหรับ core.compression.llama_eval — eval ผ่าน ServerHandle.complete"""

from __future__ import annotations

import pytest

from core.compression import llama_eval
from core.evaluator import EVAL_MAX_NEW_TOKENS, EvalCase
from core.trainer_worker import build_fim_prompt

FIM = {"prefix": "<PRE>", "suffix": "<SUF>", "middle": "<MID>"}


class FakeTok:
    """encode หยาบ ๆ พอให้ token_f1 ทำงาน (multiset ของ bytes)"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


class FakeServer:
    """คืน response ตามลำดับ + จับ prompt ทุกตัว (Review Focus #1)"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def complete(self, prompt: str, *, n_predict: int, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        self.kwargs.append({"n_predict": n_predict, "temperature": temperature})
        return self._responses.pop(0)


def _cases(n: int) -> list[EvalCase]:
    return [
        EvalCase(prefix=f"def f{i}():\n    ", suffix=f"\n# end {i}", middle=f"return {i}")
        for i in range(n)
    ]


def test_eval_prompt_verbatim_equals_build_fim_prompt():
    case = EvalCase(prefix="  spaced\nline\twith  ", suffix=" tail \n", middle="body")
    server = FakeServer([case.middle])

    llama_eval.evaluate_with_llama(
        [case], server, tokenizer=FakeTok(), fim_tokens=FIM
    )

    expected = build_fim_prompt(case.prefix, case.suffix, fim_tokens=FIM)
    assert len(server.prompts) == 1
    assert server.prompts[0] == expected  # byte เดียวกันทั้ง string — ไม่ strip/แทรก


def test_perfect_predictions_score_full():
    cases = _cases(3)
    server = FakeServer([c.middle for c in cases])

    result = llama_eval.evaluate_with_llama(
        cases, server, tokenizer=FakeTok(), fim_tokens=FIM
    )

    assert result["exact_match_pct"] == 100.0
    assert result["token_f1_mean"] == 1.0
    assert server.kwargs[0]["n_predict"] == EVAL_MAX_NEW_TOKENS  # default คงที่


def test_empty_predictions_score_zero():
    cases = _cases(4)
    server = FakeServer(["", "", "", ""])

    result = llama_eval.evaluate_with_llama(
        cases, server, tokenizer=FakeTok(), fim_tokens=FIM
    )

    assert result["exact_match_pct"] == 0.0
    assert result["token_f1_mean"] == 0.0


def test_result_shape_matches_evaluate_cases():
    cases = _cases(2)
    server = FakeServer([c.middle for c in cases])

    result = llama_eval.evaluate_with_llama(
        cases, server, tokenizer=FakeTok(), fim_tokens=FIM
    )

    assert set(result) == {"per_case", "exact_match_pct", "token_f1_mean"}
    first = result["per_case"][0]
    assert set(first) == {"i", "exact", "f1", "pred", "gt", "latency_ms"}
    assert isinstance(first["latency_ms"], float) and first["latency_ms"] >= 0


def test_progress_every_ten_and_last():
    cases = _cases(12)
    server = FakeServer([c.middle for c in cases])
    calls: list[tuple[int, int]] = []

    llama_eval.evaluate_with_llama(
        cases,
        server,
        tokenizer=FakeTok(),
        fim_tokens=FIM,
        progress=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(10, 12), (12, 12)]


@pytest.mark.integration
def test_evaluate_with_llama_real_server():
    from core.compression import config, llama_runner

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

    cases = [
        EvalCase(prefix="def add(a, b):\n    ", suffix="\nprint(add(1, 2))", middle="return a + b"),
        EvalCase(prefix="x = [1, 2, 3]\ny = ", suffix="\nprint(y)", middle="x"),
    ]
    handle = llama_runner.start_server(gguf, timeout=120.0)
    try:
        result = llama_eval.evaluate_with_llama(
            cases, handle, tokenizer=FakeTok(), fim_tokens=FIM, n_predict=32
        )
        assert set(result) == {"per_case", "exact_match_pct", "token_f1_mean"}
        assert all(c["pred"] is not None for c in result["per_case"])
    finally:
        llama_runner.stop_server(handle)
    assert handle.proc.poll() is not None
