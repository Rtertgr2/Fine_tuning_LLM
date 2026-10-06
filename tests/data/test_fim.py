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


# ---------------------------------------------------------------------------
# encode_prompt + decode_continuation — path เดียวกับ evaluate_cases/predict_middle
# (fake ล้วน ไม่พึ่ง torch/transformers — D10: evaluator ต้อง import ได้โดยไม่ดึง torch)
# ---------------------------------------------------------------------------


class PromptTok:
    """tokenizer ปลอมสำหรับ encode/decode prompt-cont path — บันทึก kwargs ที่ถูกเรียก"""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.call_kwargs: list[dict] = []
        self.decode_kwargs: list[dict] = []

    def __call__(self, prompt: str, **kwargs) -> dict:
        self.call_kwargs.append(kwargs)
        return {"prompt_len": len(prompt)}

    def convert_tokens_to_ids(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = 100 + len(self.vocab)
        return self.vocab[tok]

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        self.decode_kwargs.append({"skip_special_tokens": skip_special_tokens})
        inv = {v: k for k, v in self.vocab.items()}
        return ",".join(inv.get(int(i), f"id{i}") for i in ids)


FIM_ROLE_TOKENS = {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"}


def test_encode_prompt_disables_special_tokens():
    # P1 D2: ตรงฝั่งเทรน (packing ไม่เติม special) — BOS ฝั่งเดียว = prompt บวม ≠ ฝั่ง eval
    tok = PromptTok()
    out = wa.encode_prompt(tok, "<|fim_prefix|>def f():")
    assert tok.call_kwargs == [{"return_tensors": "pt", "add_special_tokens": False}]
    assert out == {"prompt_len": len("<|fim_prefix|>def f():")}


def test_decode_continuation_drops_fim_marker_ids():
    # P0 D1: marker id ไม่ใช่ special token → decode ปล่อยออกมา → กรองเองก่อน decode
    tok = PromptTok()
    marker_id = tok.convert_tokens_to_ids("<|fim_middle|>")
    content_ids = [tok.convert_tokens_to_ids(w) for w in ("return", "1")]
    text = wa.decode_continuation(
        tok, [*content_ids, marker_id, marker_id], fim_tokens=FIM_ROLE_TOKENS
    )
    assert text == "return,1"  # marker ทั้ง 2 ตัวหลุดออก เนื้อหาคงเดิม/ลำดับเดิม
    assert tok.decode_kwargs == [{"skip_special_tokens": True}]


def test_decode_continuation_all_markers_is_empty():
    tok = PromptTok()
    marker_ids = [tok.convert_tokens_to_ids(t) for t in FIM_ROLE_TOKENS.values()]
    text = wa.decode_continuation(tok, marker_ids, fim_tokens=FIM_ROLE_TOKENS)
    assert text == ""  # เฉย marker ล้วน → pred ว่าง (ไม่ใช่ string ของ marker)
