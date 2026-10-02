"""Tests สำหรับ core/dataset_builder.py — FIM PSM, line boundary, EOS, seed 42 (สเปก plan.md §4.3)"""

import json
import random
from pathlib import Path

from configs.safe_defaults import HELDOUT_RATIO, MAX_SEQ_LENGTH_DEFAULT
from core import dataset_builder as db

REGISTRY = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "fim_registry.json").read_text(
        encoding="utf-8"
    )
)
QWEN = REGISTRY["qwen"]
EOS = "</s>"

MULTI_LINE_CODE = "\n".join(f"line_{i} = {i}" for i in range(8)) + "\n"


def _codes(n: int) -> list[str]:
    return [
        "\n".join([f"def f{i}(x):", f"    y = x + {i}", "    if y > 0:", "        return y", "    return 0", ""]) 
        for i in range(n)
    ]


def test_format_psm_exact():
    out = db.format_psm("PRE\n", "SUF\n", "MID\n", fim_tokens=QWEN, eos=EOS)
    expected = "<|fim_prefix|>" + "PRE\n" + "<|fim_suffix|>" + "SUF\n" + "<|fim_middle|>" + "MID\n" + EOS
    assert out == expected
    assert out.index("<|fim_prefix|>") < out.index("<|fim_suffix|>") < out.index("<|fim_middle|>")


def test_split_line_boundary_and_reconstruct():
    for seed in (0, 1, 2, 42, 99):
        rng = random.Random(seed)
        prefix, suffix, middle = db.split_fim(MULTI_LINE_CODE, rng)
        # reconstruct เป๊ะ
        assert prefix + middle + suffix == MULTI_LINE_CODE
        # ทุกส่วนไม่ว่าง
        assert prefix and middle and suffix
        # จุดตัดอยู่หลัง \n เสมอ (ไม่ตัดกลางบรรทัด)
        assert MULTI_LINE_CODE[len(prefix) - 1] == "\n"
        assert MULTI_LINE_CODE[len(prefix) + len(middle) - 1] == "\n"


def test_eos_at_end():
    outputs = list(db.build_samples(_codes(20), fim_tokens=QWEN, eos=EOS))
    assert outputs, "ไม่มีผลลัพธ์"
    for out in outputs:
        assert out.endswith(EOS)


def test_seed_reproducible():
    codes = _codes(20)
    run_a = list(db.build_samples(codes, fim_tokens=QWEN, eos=EOS, seed=42))
    run_b = list(db.build_samples(codes, fim_tokens=QWEN, eos=EOS, seed=42))
    assert run_a == run_b
    run_c = list(db.build_samples(codes, fim_tokens=QWEN, eos=EOS, seed=99))
    assert run_c != run_a


def test_fim_rate_half():
    codes = _codes(400)
    outputs = list(db.build_samples(codes, fim_tokens=QWEN, eos=EOS, seed=42))
    fim_count = sum(1 for out in outputs if "<|fim_prefix|>" in out)
    ratio = fim_count / len(outputs)
    assert abs(ratio - 0.5) <= 0.08, f"fim_rate = {ratio}"


def test_short_sample_skipped():
    short = "line_a = 1\nline_b = 2\n"  # 2 บรรทัด < MIN_SAMPLE_LINES=3
    outputs = list(db.build_samples([short, MULTI_LINE_CODE], fim_tokens=QWEN, eos=EOS))
    assert len(outputs) == 1
    assert not any(short.splitlines()[0] in out for out in outputs)


def test_truncation_keeps_eos():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", local_files_only=True
    )
    long_code = "\n".join(f"value_{i} = compute({i})" for i in range(200)) + "\n"
    outputs = list(
        db.build_samples(
            [long_code],
            fim_tokens=QWEN,
            eos=EOS,
            seed=42,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH_DEFAULT,
        )
    )
    assert len(outputs) == 1
    out = outputs[0]
    assert out.endswith(EOS)
    n_tokens = len(tokenizer.encode(out, add_special_tokens=False))
    assert n_tokens <= MAX_SEQ_LENGTH_DEFAULT, f"{n_tokens} tokens"


def test_heldout_stable_and_ratio():
    # คงที่ต่อ code เดิม
    code = _codes(1)[0]
    assert db.is_heldout(code) == db.is_heldout(code)
    # สัดส่วนใกล้ HELDOUT_RATIO
    codes = _codes(1000)
    held = sum(1 for c in codes if db.is_heldout(c))
    ratio = held / len(codes)
    assert 0.05 <= ratio <= 0.15, f"heldout ratio = {ratio}"
    assert HELDOUT_RATIO == 0.10


def test_filter_train_codes_removes_all_heldout():
    codes = [f"line_a_{i}\nline_b_{i}\nline_c_{i}" for i in range(200)]
    held = {c for c in codes if db.is_heldout(c)}
    assert held, "fixture ต้องมีทั้งสองฝั่ง (md5 split ~10%)"
    result = db.filter_train_codes(codes)
    assert all(not db.is_heldout(c) for c in result)
    assert set(result) == set(codes) - held  # ไม่หาย/ไม่เพิ่ม
    assert db.filter_train_codes(codes) == result  # deterministic


def test_iter_codes_limits_and_column(monkeypatch):
    calls: dict = {}

    def fake_load_dataset(dataset_id, *, split, streaming, cache_dir):
        calls["dataset_id"] = dataset_id
        calls["split"] = split
        calls["streaming"] = streaming
        calls["cache_dir"] = cache_dir
        return ({"content": f"code_{i}", "other": i} for i in range(100))

    monkeypatch.setattr(db, "load_dataset", fake_load_dataset)
    got = db.iter_codes("fake/ds", "content", limit=5)
    assert got == [f"code_{i}" for i in range(5)]  # 5 ตัวแรกตามลำดับ
    # kwargs ที่ส่งไปต้องตรง: streaming + split + cache_dir
    assert calls == {
        "dataset_id": "fake/ds",
        "split": "train",
        "streaming": True,
        "cache_dir": "data_cache",
    }
