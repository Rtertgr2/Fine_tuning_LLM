"""dataset_builder — แปลงโค้ดดิบเป็น Fill-in-the-Middle (สเปก plan.md §4.3)

กติกา hard: fim_rate = 0.5, รูปแบบ PSM เท่านั้น, ตัดที่ line boundary เท่านั้น,
EOS ท้ายทุก sample, seed = 42 (ผลซ้ำได้), truncate ไม่เกิน max_seq_length
รูปแบบ: prefix_tok + prefix + suffix_tok + suffix + middle_tok + middle + EOS
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

from datasets import load_dataset

from configs.safe_defaults import (
    FIM_RATE,
    HELDOUT_RATIO,
    MAX_SEQ_LENGTH_DEFAULT,
    MIN_SAMPLE_LINES,
    SEED,
)

_HEX_MAX = 16**32


def _cut_positions(code: str) -> list[int]:
    """index ของทุก \\n ที่ยังมีข้อความเหลือหลังมัน (ใช้เป็นจุดตัดได้)"""
    return [i for i, ch in enumerate(code) if ch == "\n" and i + 1 < len(code)]


def split_fim(code: str, rng: random.Random) -> tuple[str, str, str]:
    """สุ่มจุดตัด 2 จุดที่ line boundary → (prefix, suffix, middle) ทุกส่วน ≥1 บรรทัด"""
    cuts = _cut_positions(code)
    if len(cuts) < 2:
        raise ValueError("ตัด FIM ไม่ได้: จุดตัดที่ line boundary น้อยกว่า 2")
    i, j = sorted(rng.sample(cuts, 2))
    return code[: i + 1], code[j + 1 :], code[i + 1 : j + 1]


def format_psm(
    prefix: str, suffix: str, middle: str, *, fim_tokens: dict, eos: str
) -> str:
    """ประกอบเป็น PSM: prefix_tok + prefix + suffix_tok + suffix + middle_tok + middle + EOS"""
    return (
        f"{fim_tokens['prefix']}{prefix}"
        f"{fim_tokens['suffix']}{suffix}"
        f"{fim_tokens['middle']}{middle}"
        f"{eos}"
    )


def truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    """ตัดข้อความไม่ให้เกิน max_tokens (นับด้วย tokenizer จริง)"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def build_samples(
    codes: Iterable[str],
    *,
    fim_tokens: dict,
    eos: str,
    fim_rate: float = FIM_RATE,
    seed: int = SEED,
    min_lines: int = MIN_SAMPLE_LINES,
    tokenizer: Any | None = None,
    max_seq_length: int = MAX_SEQ_LENGTH_DEFAULT,
) -> Iterator[str]:
    """สร้าง sample สำหรับเทรน: FIM (PSM) ตาม fim_rate ที่เหลือเป็น plain LM

    - rng ตัวเดียว seed ด้วย `seed` ทั้ง generator → ผลซ้ำได้
    - sample สั้นกว่า min_lines หรือมีจุดตัดไม่พอ → ข้าม (ไม่ consume rng)
    - truncate (เมื่อมี tokenizer) จะเผื่อที่ให้ EOS ก่อนเสมอ → EOS อยู่ท้ายเสมอ
    """
    rng = random.Random(seed)
    for code in codes:
        if len(code.splitlines()) < min_lines or len(_cut_positions(code)) < 2:
            continue
        if rng.random() < fim_rate:
            prefix, suffix, middle = split_fim(code, rng)
            body = format_psm(prefix, suffix, middle, fim_tokens=fim_tokens, eos="")
        else:
            body = code
        if tokenizer is not None:
            eos_len = len(tokenizer.encode(eos, add_special_tokens=False))
            body = truncate_to_tokens(body, tokenizer, max_seq_length - eos_len)
        yield body + eos


def is_heldout(code: str, heldout_ratio: float = HELDOUT_RATIO) -> bool:
    """แยก train/held-out ด้วย md5 คงที่ — code เดียวกันตกข้างเดียวกันเสมอ (กัน leakage)"""
    digest = hashlib.md5(code.encode("utf-8")).hexdigest()
    return int(digest, 16) / _HEX_MAX < heldout_ratio


def filter_train_codes(codes: Iterable[str]) -> list[str]:
    """กรอง heldout ออกจากชุดเทรน — ห้าม train ปนชุดประเมิน (plan.md §8)"""
    return [code for code in codes if not is_heldout(code)]


def iter_codes(
    dataset_id: str,
    column: str,
    *,
    limit: int,
    split: str = "train",
    cache_dir: str = "data_cache",
) -> list[str]:
    """อ่าน `limit` ตัวอย่างแรกจาก dataset แบบ streaming — ชุดเทรนย่อย (smoke, plan.md §4.4)"""
    ds = load_dataset(dataset_id, split=split, streaming=True, cache_dir=cache_dir)
    return [row[column] for row in islice(ds, limit)]
