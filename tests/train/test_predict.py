"""Tests สำหรับ core/train/predict.py — predict_middle token path (P1 D2 BOS + P0 D1 marker)

fake tokenizer/model ล้วน — ไม่โหลด model/tokenizer จริง, ไม่ spawn process
"""

import sys
from types import SimpleNamespace

import torch

from conftest import FakeQueue
from core.train import predict as pr

# ตรง configs/fim_registry.json key "qwen" (predict_middle อ่านไฟล์จริง)
FIM_TOKENS = {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"}


class FakeTok:
    """tokenizer ปลอม word-level — บันทึก kwargs ของ __call__ + roundtrip FIM marker"""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = dict(
            zip(FIM_TOKENS.values(), (100, 101, 102), strict=True)
        )
        self.captured_prompts: list[str] = []
        self.call_kwargs: list[dict] = []

    def __len__(self) -> int:
        return 30500

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = 1000 + len(self.vocab)
            out.append(self.vocab[word])
        return out

    def __call__(self, prompt: str, *, return_tensors: str = "pt",
                 add_special_tokens: bool = True) -> dict:
        self.captured_prompts.append(prompt)
        self.call_kwargs.append(
            {"return_tensors": return_tensors, "add_special_tokens": add_special_tokens}
        )
        ids = torch.tensor([self.encode(prompt)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        return self.vocab.get(tok)

    def convert_ids_to_tokens(self, tid: int) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return inv.get(tid, "<unk>")

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(int(i), "<unk>") for i in ids)


class FakeModel:
    """model ปลอม — generate คืน continuation คงที่ (ต่อท้าย input_ids เหมือนของจริง)"""

    def __init__(self, continuation_text: str, tok: FakeTok) -> None:
        self._cont = continuation_text
        self._tok = tok
        self.calls: list[dict] = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        # device ตาม input_ids (เครื่องนี้ predict เลือก xpu — cont ต้องอยู่เครื่องเดียวกัน)
        cont = torch.tensor(
            [self._tok.encode(self._cont)],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, cont], dim=1)


def _config(tmp_path) -> dict:
    # ครบ REQUIRED_CONFIG_KEYS — validate_config ตัวจริงต้องผ่าน
    return {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": str(tmp_path),
        "max_seq_length": 1024,
        "max_steps": 6,
        "code_limit": 64,
        "lora_rank": 8,
    }


def _run(monkeypatch, tmp_path, *, model: FakeModel, tok: FakeTok) -> FakeQueue:
    """predict_middle ครบ loop ด้วยของปลอมทุกจุด (tokenizer/model/adapter/peft import)"""
    monkeypatch.setattr(
        pr, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: tok)
    )
    monkeypatch.setattr(
        pr, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *a, **k: model)
    )
    monkeypatch.setattr(pr, "latest_checkpoint", lambda out: tmp_path / "checkpoint-6")
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=lambda *a, **k: model)
        ),
    )
    queue = FakeQueue()
    pr.predict_middle(_config(tmp_path), "def f():", "pass", queue)
    return queue


def test_predict_middle_tokenizes_without_special_tokens(monkeypatch, tmp_path):
    # P1 D2: เดิมเรียก tokenizer(prompt, return_tensors="pt") = default add_special_tokens=True
    # → stray BOS บวม prompt ≠ eval (evaluate_cases ใช้ add_special_tokens=False)
    tok = FakeTok()
    model = FakeModel("return 1", tok)
    _run(monkeypatch, tmp_path, model=model, tok=tok)
    assert tok.call_kwargs[0] == {"return_tensors": "pt", "add_special_tokens": False}


def test_predict_middle_strips_fim_markers_from_output(monkeypatch, tmp_path):
    # P0 D1: เดิม decode ต่อท้ายตรง ๆ ไม่กรอง marker id → model echo ออกมาเป็น pred
    tok = FakeTok()
    model = FakeModel("return 1 <|fim_middle|>", tok)
    queue = _run(monkeypatch, tmp_path, model=model, tok=tok)
    assert queue.messages == [{"type": "log", "level": "INFO", "text": "return 1"}]


def test_predict_middle_greedy_and_prompt_is_psm(monkeypatch, tmp_path):
    # guard: generate ต้อง greedy (ผลซ้ำได้) และ prompt ต้องเป็น PSM ของ case จริง
    from core.data.fim import build_fim_prompt

    tok = FakeTok()
    model = FakeModel("x", tok)
    _run(monkeypatch, tmp_path, model=model, tok=tok)
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 256
    assert tok.captured_prompts == [
        build_fim_prompt("def f():", "pass", fim_tokens=FIM_TOKENS)
    ]
