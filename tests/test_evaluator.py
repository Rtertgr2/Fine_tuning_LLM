"""Tests สำหรับ core/evaluator.py — FIM Exact Match, Token F1, EvalCase (spec §3.2)"""

import dataclasses

import pytest
import torch

from core import dataset_builder as db
from core import evaluator as ev
from core.trainer_worker import build_fim_prompt

FIM = {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"}


class FakeTok:
    """Tokenizer ปลอม word-level — vocab ร่วมต่อ instance (encode/decode/call ครบทั้ง 3)"""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.captured_prompts: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab)
            out.append(self._vocab[word])
        return out

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(int(i), "<unk>") for i in ids)

    def __call__(self, prompt: str, *, return_tensors: str = "pt") -> dict:
        self.captured_prompts.append(prompt)
        ids = torch.tensor([self.encode(prompt)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel:
    """model ปลอม — generate คืน continuation คงที่ (ตัดต่อ input_ids)"""

    def __init__(self, continuation_text: str, tok: FakeTok) -> None:
        self._cont = continuation_text
        self._tok = tok
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        cont = torch.tensor([self._tok.encode(self._cont)], dtype=input_ids.dtype)
        return torch.cat([input_ids, cont], dim=1)


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


# --- helpers สำหรับ build_eval_cases ---


def _codes_where(want_heldout: bool, n: int) -> list[str]:
    """ลอง nonce จน `is_heldout` ได้ค่าที่ต้องการ — ทุกตัวผ่าน criteria (4 บรรทัด, 3 จุดตัด)"""
    out: list[str] = []
    i = 0
    while len(out) < n:
        code = f"def f{i}():\n    x = {i}\n    y = {i} # nonce_{i}\n    return x\n"
        if db.is_heldout(code) == want_heldout:
            out.append(code)
        i += 1
    return out


def _heldout_variant(make) -> str:
    """make(i) → code ที่ผ่าน criteria ทุกเงื่อนไข; วนจน is_heldout = True"""
    i = 0
    while True:
        code = make(i)
        if db.is_heldout(code):
            return code
        i += 1


def test_build_eval_cases_only_heldout(monkeypatch):
    held = _codes_where(True, 40)
    train = _codes_where(False, 40)
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: held + train)
    cases = ev.build_eval_cases(
        dataset_id="x",
        dataset_column="content",
        limit=80,
        tokenizer=FakeTok(),
        n_cases=10,
    )
    assert len(cases) == 10
    # ทุก middle มาจาก heldout เท่านั้น — middle มีเลข nonce ของ code ต้นทางเสมอ
    # (middle ครอบคลุมอย่างน้อย 1 บรรทัดเต็มของบรรทัด 2 หรือ 3 ซึ่งมี `{i}` กำกับ)
    for case in cases:
        sources = [c for c in held + train if case.middle in c]
        assert len(sources) == 1, f"middle ต้องระบุ code เดียว: {case.middle!r}"
        assert db.is_heldout(sources[0])


def test_build_eval_cases_skips_long_middle(monkeypatch):
    # ทุกบรรทัด 300 คำ → middle ทุกกรณี (จุดตัดมีค่าเดียว) > 256 tokens → ข้าม
    def make(i: int) -> str:
        line = " ".join(f"w{j}" for j in range(300))
        return f"{line} # a{i}\n{line}\n{line}\n"

    code = _heldout_variant(make)
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: [code])
    cases = ev.build_eval_cases(
        dataset_id="x",
        dataset_column="content",
        limit=1,
        tokenizer=FakeTok(),
        n_cases=10,
    )
    assert cases == []  # middle เกิน 256 ถูกข้าม ไม่ raise


def test_build_eval_cases_deterministic(monkeypatch):
    held = _codes_where(True, 30)
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: held)
    kwargs = dict(dataset_id="x", dataset_column="content", limit=30, n_cases=10)
    run_a = ev.build_eval_cases(tokenizer=FakeTok(), **kwargs)
    run_b = ev.build_eval_cases(tokenizer=FakeTok(), **kwargs)
    assert run_a == run_b


def test_build_eval_cases_fewer_than_requested(monkeypatch):
    held = _codes_where(True, 3)
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: held)
    cases = ev.build_eval_cases(
        dataset_id="x",
        dataset_column="content",
        limit=3,
        tokenizer=FakeTok(),
        n_cases=10,
    )
    assert len(cases) == 3  # คืนเท่าที่มี ไม่ raise (spec §6)


def test_evaluate_cases_prompt_equals_build_fim_prompt():
    # Review Focus #4: fim_tokens ต้องไหลถึง prompt เป๊ะ
    tok = FakeTok()
    case = ev.EvalCase("def f():", "pass", "return 1")
    model = FakeModel("return 1", tok)
    result = ev.evaluate_cases(
        model, tok, [case], fim_tokens=FIM, device="cpu"
    )
    assert tok.captured_prompts == [build_fim_prompt("def f():", "pass", fim_tokens=FIM)]
    assert model.calls[0]["do_sample"] is False  # greedy — ผล eval ต้องซ้ำได้
    assert result["per_case"][0]["exact"] is True


def test_evaluate_cases_aggregates_and_progress():
    tok = FakeTok()
    cases = [
        ev.EvalCase("a", "b", "return 1"),
        ev.EvalCase("a", "b", "xyz"),
    ]
    model = FakeModel("return 1", tok)  # case0 ตรง → exact, case1 ผิด
    seen: list[tuple[int, int]] = []
    result = ev.evaluate_cases(
        model, tok, cases, fim_tokens=FIM, device="cpu",
        progress=lambda done, total: seen.append((done, total)),
    )
    # spec §3.3: progress ทุก 10 cases + ครั้งสุดท้ายเสมอ (total=2 < 10 → มีแค่ครั้งสุดท้าย)
    assert seen == [(2, 2)]
    assert result["exact_match_pct"] == 50.0
    assert result["token_f1_mean"] == 0.5  # (1.0 + 0.0) / 2
    assert len(result["per_case"]) == 2
    assert result["per_case"][1]["exact"] is False
    assert result["per_case"][0]["gt"] == "return 1"


def test_progress_fires_every_ten_and_final():
    """spec §3.3: progress ทุก 10 cases (ไม่ใช่ทุก case — กัน flood log)"""
    tok = FakeTok()
    cases = [ev.EvalCase("a", "b", "return 1")] * 12
    model = FakeModel("return 1", tok)
    seen: list[tuple[int, int]] = []
    ev.evaluate_cases(
        model, tok, cases, fim_tokens=FIM, device="cpu",
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(10, 12), (12, 12)]


def _valid_config(**overrides) -> dict:
    cfg = {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": "data_cache/unused",
        "max_seq_length": 1024,
        "max_steps": 6,
        "code_limit": 64,
        "lora_rank": 8,
    }
    cfg.update(overrides)
    return cfg


def test_run_eval_missing_checkpoint_message(tmp_path):
    cfg = _valid_config(output_dir=str(tmp_path))  # tmp ว่าง
    with pytest.raises(FileNotFoundError, match="train first"):
        ev.run_eval(cfg, mode="finetuned")


def test_run_eval_rejects_bad_mode(tmp_path):
    with pytest.raises(ValueError, match="base"):
        ev.run_eval(_valid_config(), mode="train")


def test_run_eval_no_cases_raises(monkeypatch, tmp_path):
    # stream ว่าง → ห้ามเขียน JSON metrics ปลอม (false-green §8.2)
    class _FakeAutoTok:
        @staticmethod
        def from_pretrained(model_id):
            return FakeTok()

    monkeypatch.setattr(ev, "AutoTokenizer", _FakeAutoTok)
    monkeypatch.setattr(ev, "ensure_fim_tokens", lambda tok, vals: {})
    monkeypatch.setattr(ev, "iter_codes", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no eval cases"):
        ev.run_eval(_valid_config(), mode="base", eval_dir=tmp_path)
