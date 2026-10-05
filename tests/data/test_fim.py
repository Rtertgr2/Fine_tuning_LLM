"""Tests สำหรับ core/data/fim.py — ensure_fim_tokens guard + build_fim_prompt PSM"""

import pytest

from core.data import fim as wa


class FakeTokenizer:
    """จำลอง tokenizer สำหรับ FIM guard — จงใจไม่มี method resize (ถ้าโค้ดเรียก = AttributeError ทันที)"""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = dict(vocab)
        self.unk_token_id = 0  # id 0 = UNK เหมือน HF convention

    def __len__(self) -> int:
        return 30500

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, tid: int) -> str:
        for tok, i in self.vocab.items():
            if i == tid:
                return tok
        return "<unk>"

    def decode(self, tid: int) -> str:
        return self.convert_ids_to_tokens(tid)


FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


def test_ensure_fim_tokens_ok():
    tok = FakeTokenizer({"<|fim_prefix|>": 100, "<|fim_suffix|>": 101, "<|fim_middle|>": 102})
    ids = wa.ensure_fim_tokens(tok, FIM_TOKENS)
    assert ids == {"<|fim_prefix|>": 100, "<|fim_suffix|>": 101, "<|fim_middle|>": 102}


def test_ensure_fim_tokens_missing_raises():
    with pytest.raises(ValueError) as exc:
        wa.ensure_fim_tokens(FakeTokenizer({}), FIM_TOKENS)  # ว่าง = ทุกตัวคืน UNK
    assert "<|fim_prefix|>" in str(exc.value)  # ชื่อ token ที่ขาดต้องอยู่ในข้อความ


def test_build_fim_prompt_exact():
    fim_tokens = {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"}
    prompt = wa.build_fim_prompt("def f():", "return 1", fim_tokens=fim_tokens)
    # PSM แบบไม่มี middle (model generate ต่อจาก middle_tok เอง) — ไม่มี eos เสริม
    assert prompt == "<|fim_prefix|>def f():<|fim_suffix|>return 1<|fim_middle|>"
