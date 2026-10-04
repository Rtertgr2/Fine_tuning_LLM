"""Tests สำหรับ core/dataset_builder.py — FIM PSM, line boundary, EOS, seed 42 (สเปก plan.md §4.3)"""

import json
import random
from pathlib import Path

import pytest

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


def _qwen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", local_files_only=True
    )


def _expected_fim_parts(code: str, seed: int, fim_rate: float):
    """จำลอง rng stream ของ build_samples เพื่อรู้ prefix/suffix/middle จริง"""
    rng = random.Random(seed)
    is_fim = rng.random() < fim_rate
    assert is_fim, "test นี้ต้องเดินเข้า FIM branch"
    return db.split_fim(code, rng)


def test_fim_budget_keeps_middle_or_drops_sample():
    # 🔴 regression: middle อยู่ท้าย PSM → tail-truncate เดิมตัด middle แล้วประกบ EOS
    # กลาง span (สอนโมเดลผิด) — ต้องมาครบทั้งก้อนหรือ drop เท่านั้น
    tokenizer = _qwen_tokenizer()
    long_code = "\n".join(f"value_{i} = compute({i})" for i in range(200)) + "\n"
    outputs = list(
        db.build_samples(
            [long_code],
            fim_tokens=QWEN,
            eos=EOS,
            fim_rate=1.0,
            seed=42,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH_DEFAULT,
        )
    )
    _prefix, _suffix, middle = _expected_fim_parts(long_code, 42, 1.0)
    eos_len = len(tokenizer.encode(EOS, add_special_tokens=False))
    overhead = sum(
        len(tokenizer.encode(t, add_special_tokens=False)) for t in QWEN.values()
    )
    middle_len = len(tokenizer.encode(middle, add_special_tokens=False))
    budget = MAX_SEQ_LENGTH_DEFAULT - eos_len
    if middle_len + overhead > budget:
        assert outputs == [], f"middle {middle_len}+overhead เกิน budget → ต้อง drop"
        return
    assert len(outputs) == 1, "middle อยู่ใน budget → ต้อง yield (แค่หด prefix/suffix)"
    out = outputs[0]
    assert out.endswith(EOS)
    got_middle = out.split(QWEN["middle"], 1)[1][: -len(EOS)]
    assert got_middle == middle, "middle ต้องมาครบทุกอักขระ (ห้าม tail-truncate)"
    n_tokens = len(tokenizer.encode(out, add_special_tokens=False))
    assert n_tokens <= MAX_SEQ_LENGTH_DEFAULT, f"{n_tokens} tokens"


def test_fim_dropped_when_middle_exceeds_budget():
    tokenizer = _qwen_tokenizer()
    outputs = list(
        db.build_samples(
            [MULTI_LINE_CODE],
            fim_tokens=QWEN,
            eos=EOS,
            fim_rate=1.0,
            seed=42,
            tokenizer=tokenizer,
            max_seq_length=5,
        )
    )
    assert outputs == [], "budget 5 tokens ใส่ middle ไม่ได้ → drop ทั้งก้อน"


def test_plain_lm_truncate_unchanged():
    # plain LM (ไม่ใช่ FIM) ยัง tail-truncate + EOS เหมือนเดิมเป๊ะ
    tokenizer = _qwen_tokenizer()
    long_code = "\n".join(f"value_{i} = compute({i})" for i in range(200)) + "\n"
    outputs = list(
        db.build_samples(
            [long_code],
            fim_tokens=QWEN,
            eos=EOS,
            fim_rate=0.0,
            seed=42,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH_DEFAULT,
        )
    )
    assert len(outputs) == 1
    budget = MAX_SEQ_LENGTH_DEFAULT - len(tokenizer.encode(EOS, add_special_tokens=False))
    expected = db.truncate_to_tokens(long_code, tokenizer, budget) + EOS
    assert outputs[0] == expected


def test_build_eval_texts_heldout_only_and_limited():
    # 🟡 val loop: eval ต้องมาจาก heldout (ไม่เคยเห็นตอนเทรน) จำกัด limit
    codes = _codes(40)
    heldout = [c for c in codes if db.is_heldout(c)]
    assert heldout and len(heldout) < len(codes), "fixture ต้องมีทั้งสองฝั่ง (md5 bucket)"
    texts = db.build_eval_texts(
        codes,
        fim_tokens=QWEN,
        eos=EOS,
        tokenizer=None,
        max_seq_length=MAX_SEQ_LENGTH_DEFAULT,
        limit=3,
    )
    assert len(texts) == min(len(heldout), 3)
    expected = list(db.build_samples(heldout, fim_tokens=QWEN, eos=EOS))[:3]
    assert texts == expected  # ตรงกับ build_samples บน heldout เฉย ๆ (seed เดิม)


def test_build_eval_texts_empty_when_no_heldout():
    # dataset ที่ไม่มี code ตก heldout → [] (run_training ต้องปิด eval loop แทน crash)
    codes = [c for c in _codes(60) if not db.is_heldout(c)]
    assert codes, "ต้องหา non-heldout ได้บ้าง"
    texts = db.build_eval_texts(
        codes,
        fim_tokens=QWEN,
        eos=EOS,
        tokenizer=None,
        max_seq_length=MAX_SEQ_LENGTH_DEFAULT,
        limit=3,
    )
    assert texts == []


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


# ---------------------------------------------------------------------------
# Folder detection: dropdown datasets/ (design 2026-10-02 model-dataset-picker)
# ---------------------------------------------------------------------------


def test_list_datasets_returns_only_parquet_folders(tmp_path, monkeypatch):
    from core import dataset_builder as db

    root = tmp_path / "datasets"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "part.parquet").write_bytes(b"")
    (root / "beta").mkdir()
    (root / "beta" / "readme.txt").write_text("x", encoding="utf-8")
    (root / "loose.parquet").write_bytes(b"")  # ไฟล์ลอย ๆ ไม่ใช่โฟลเดอร์ → ไม่นับ

    monkeypatch.setattr(db, "DATASETS_DIR", str(root))
    assert db.list_datasets() == ["alpha"]


def test_list_datasets_missing_dir_is_empty(tmp_path, monkeypatch):
    from core import dataset_builder as db

    monkeypatch.setattr(db, "DATASETS_DIR", str(tmp_path / "not-there"))
    assert db.list_datasets() == []


# ---------------------------------------------------------------------------
# iter_codes: อ่านโฟลเดอร์ .parquet ตรง ๆ — 20GB-safe ไม่ convert arrow ซ้ำ
# ---------------------------------------------------------------------------


def _write_parquet(path, values: list[str], column: str = "content"):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({column: values}), path)


def test_iter_codes_reads_local_parquet_folder_sorted_multifile(tmp_path, monkeypatch):
    from core import dataset_builder as db

    monkeypatch.setattr(db, "DATASETS_DIR", str(tmp_path))  # H1: ต้องอยู่ใต้ sandbox root
    ds = tmp_path / "local-ds"
    ds.mkdir()
    _write_parquet(ds / "part-00001.parquet", ["b1", "b2"])
    _write_parquet(ds / "part-00000.parquet", ["a1", "a2", "a3"])

    codes = db.iter_codes(str(ds), "content", limit=4)
    # อ่านเรียงตามชื่อไฟล์ (sorted) ภายในไฟล์เรียงตามแถว + เคารพ limit
    assert codes == ["a1", "a2", "a3", "b1"]


def test_iter_codes_resolves_bare_name_under_datasets_dir(tmp_path, monkeypatch):
    from core import dataset_builder as db

    root = tmp_path / "datasets"
    (root / "alpha").mkdir(parents=True)
    _write_parquet(root / "alpha" / "part.parquet", ["x", "y"])

    monkeypatch.setattr(db, "DATASETS_DIR", str(root))
    assert db.iter_codes("alpha", "content", limit=5) == ["x", "y"]


def test_iter_codes_local_missing_column_gives_english_error(tmp_path, monkeypatch):
    from core import dataset_builder as db

    monkeypatch.setattr(db, "DATASETS_DIR", str(tmp_path))  # H1: อยู่ใต้ sandbox root
    ds = tmp_path / "local-ds"
    ds.mkdir()
    _write_parquet(ds / "part.parquet", ["x"], column="other_col")

    try:
        db.iter_codes(str(ds), "content", limit=5)
    except ValueError as exc:
        msg = str(exc)
        assert "content" in msg and "other_col" in msg  # บอกทั้งคอลัมน์ที่ขอและที่มี
        assert "local" in msg or "parquet" in msg
    else:
        raise AssertionError("โฟลเดอร์ไม่มีคอลัมน์ที่ขอ → ต้อง ValueError")


def test_iter_codes_local_folder_without_parquet_gives_english_error(tmp_path, monkeypatch):
    from core import dataset_builder as db

    monkeypatch.setattr(db, "DATASETS_DIR", str(tmp_path))  # H1: อยู่ใต้ sandbox root
    ds = tmp_path / "empty-ds"
    ds.mkdir()
    (ds / "notes.txt").write_text("no data", encoding="utf-8")

    try:
        db.iter_codes(str(ds), "content", limit=5)
    except ValueError as exc:
        assert ".parquet" in str(exc)
    else:
        raise AssertionError("โฟลเดอร์ไม่มี parquet → ต้อง ValueError")


def test_iter_codes_and_list_support_nested_parquet_layout(tmp_path, monkeypatch):
    """layout `data/*.parquet` (คัดลอกมาจาก Hub repo ทั้งก้อน) ต้องใช้ได้"""
    from core import dataset_builder as db

    root = tmp_path / "datasets"
    nested = root / "copied-ds" / "data"
    nested.mkdir(parents=True)
    _write_parquet(nested / "train-0.parquet", ["n1", "n2"])

    monkeypatch.setattr(db, "DATASETS_DIR", str(root))
    assert db.list_datasets() == ["copied-ds"]
    assert db.iter_codes("copied-ds", "content", limit=5) == ["n1", "n2"]


# ---------------------------------------------------------------------------
# H1: path sandbox (Fix.md) — dataset ก็ต้องไม่หลุด root เหมือน model
# ---------------------------------------------------------------------------


def test_resolve_local_dataset_rejects_path_outside_roots(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(db, "DATASETS_DIR", str(root))
    monkeypatch.setattr(db, "REPO_ROOT", str(tmp_path / "no-repo"))

    # มีอยู่จริงนอก sandbox → ValueError (ห้าม None = caller จะส่งเข้า load_dataset)
    with pytest.raises(ValueError, match="outside"):
        db._resolve_local_dataset_dir(str(outside))


def test_resolve_local_dataset_allows_repo_internal_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    ds = repo / "data_cache" / "local-ds"
    ds.mkdir(parents=True)
    monkeypatch.setattr(db, "REPO_ROOT", str(repo))
    monkeypatch.setattr(db, "DATASETS_DIR", str(tmp_path / "datasets"))

    got = db._resolve_local_dataset_dir(str(ds))
    assert got == ds.resolve()


def test_read_local_parquet_rejects_non_text_values(tmp_path):
    """M2: column เป็น int → ValueError ภาษาอังกฤษบอกชื่อ column/ชนิดค่า/แถว (ไม่ใช่ AttributeError ลึก)"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = tmp_path / "ds"
    ds.mkdir()
    pq.write_table(pa.table({"nums": [1, 2, 3]}), ds / "part.parquet")

    with pytest.raises(ValueError, match="nums"):
        db._read_local_parquet(ds, "nums", limit=5)  # เดิม: คืน [1, 2, 3] แล้ว crash ลึก


# ---------------------------------------------------------------------------
# L1: md5 = bucket ล้วน ๆ (Fix.md) — ห้ามเปลี่ยน algorithm, ต้องไม่ถูก security scanner flag
# ---------------------------------------------------------------------------


def test_is_heldout_pins_md5_split_membership():
    """ค่าเหล่านี้มาจาก md5 จริง — เปลี่ยน hash = ขยับ split = base.json/finetuned.json (n=100)
    เทียบกันไม่ได้อีก + checkpoint-500 เปลี่ยนชุดเทรน (ทุก evidence บน baselines เดิม)"""
    assert db.is_heldout("def foo(): pass") is True
    assert db.is_heldout("print(1)") is True
    assert db.is_heldout("x = [i*i for i in range(10)]") is False
    assert db.is_heldout("") is False
    assert db.is_heldout("a") is True


def test_is_heldout_declares_usedforsecurity_false():
    """md5 ที่นี่ไม่ได้ใช้เพื่อ security → usedforsecurity=False ให้ bandit/CodeQL ไม่ flag (B324)
    โดย digest/ผลลัพธ์เปลี่ยนแปลงตรงไหนไม่ได้เลย"""
    import inspect

    src = inspect.getsource(db.is_heldout)
    assert "usedforsecurity=False" in src  # เดิม: ไม่มี → scanner ฟ้อง md5


def test_iter_codes_outside_existing_path_never_hits_hub(monkeypatch, tmp_path):
    """Bug: path parquet จริงนอก sandbox → None → load_dataset(path) = ส่ง local path เข้า Hub"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    outside = tmp_path / "outside-ds"
    outside.mkdir()
    pq.write_table(pa.table({"content": ["a", "b"]}), outside / "part.parquet")
    monkeypatch.setattr(db, "DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(db, "REPO_ROOT", str(tmp_path / "repo"))

    def _hub_called(*args, **kwargs):
        raise AssertionError(f"HUB CALLED WITH LOCAL PATH: {args!r}")

    monkeypatch.setattr(db, "load_dataset", _hub_called)

    with pytest.raises(ValueError, match="outside"):
        db.iter_codes(str(outside), "content", limit=5)
