"""Pin fim_registry.json ให้ตรง plan.md §4.3 — ห้ามแก้นอกเหนือจากสเปก"""

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "fim_registry.json"

EXPECTED = {
    "qwen": {"prefix": "<|fim_prefix|>", "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>"},
    "starcoder": {
        "prefix": "<fim_prefix>",
        "suffix": "<fim_suffix>",
        "middle": "<fim_middle>",
    },
    "deepseek": {
        "prefix": "<｜fim begin｜>",
        "suffix": "<｜fim hole｜>",
        "middle": "<｜fim end｜>",
    },
}


def _load() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_loads_with_three_families():
    data = _load()
    assert {"qwen", "starcoder", "deepseek"} <= set(data)


def test_qwen_values_match_spec():
    assert _load()["qwen"] == EXPECTED["qwen"]


def test_starcoder_values_match_spec():
    assert _load()["starcoder"] == EXPECTED["starcoder"]


def test_deepseek_values_match_spec():
    assert _load()["deepseek"] == EXPECTED["deepseek"]
