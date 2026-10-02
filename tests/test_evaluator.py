"""Tests สำหรับ core/evaluator.py — FIM Exact Match, Token F1, EvalCase (spec §3.2)"""

import dataclasses

import pytest

from core import evaluator as ev


class FakeTok:
    """Tokenizer ปลอมสำหรับ token_f1 — word-level, vocab ร่วมต่อ instance (นับ multiset ได้จริง)"""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)
            out.append(self._vocab[word])
        return out


def test_exact_match_normalizes():
    assert ev.exact_match("x = 1\n", "x = 1")
    assert ev.exact_match("a\r\nb", "a\nb")
    assert ev.exact_match("  x = 1  ", "x = 1")
    assert not ev.exact_match("x = 2", "x = 1")


def test_token_f1_perfect_and_empty():
    tok = FakeTok()
    assert ev.token_f1("a b c", "a b c", tok) == 1.0
    assert ev.token_f1("", "", tok) == 1.0
    assert ev.token_f1("", "a b", tok) == 0.0
    assert ev.token_f1("a b", "", tok) == 0.0


def test_token_f1_partial_multiset():
    tok = FakeTok()
    # pred {a,a,b} vs gt {a,b,c}: intersection = {a,b} → P=2/3 R=2/3 F1=2/3
    assert abs(ev.token_f1("a a b", "a b c", tok) - (2 / 3)) < 1e-9


def test_token_f1_order_insensitive():
    tok = FakeTok()
    assert ev.token_f1("b a c", "a b c", tok) == 1.0


def test_evalcase_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.EvalCase("a", "b", "c").prefix = "x"
