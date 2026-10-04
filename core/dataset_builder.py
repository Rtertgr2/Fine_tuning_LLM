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
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from configs.safe_defaults import (
    DATASETS_DIR,
    FIM_RATE,
    HELDOUT_RATIO,
    MAX_SEQ_LENGTH_DEFAULT,
    MIN_SAMPLE_LINES,
    SEED,
)
from core.sandbox import project_roots
from datasets import load_dataset

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
    """แยก train/held-out ด้วย md5 คงที่ — code เดียวกันตกข้างเดียวกันเสมอ (กัน leakage)

    L1 (Fix.md): md5 ที่นี่เป็นการ bucket ล้วน ๆ ไม่ใช่ security → usedforsecurity=False
    (bandit/CodeQL ไม่ flag B324 โดย digest เปลี่ยนตรงไหนไม่ได้) — ห้ามเปลี่ยน algorithm:
    digest ขยับ = split ขยับ = ทุก baseline eval (n=100) + checkpoint-500 เทียบกันไม่ได้อีก
    """
    digest = hashlib.md5(code.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) / _HEX_MAX < heldout_ratio


def filter_train_codes(codes: Iterable[str]) -> list[str]:
    """กรอง heldout ออกจากชุดเทรน — ห้าม train ปนชุดประเมิน (plan.md §8)"""
    return [code for code in codes if not is_heldout(code)]


def list_datasets() -> list[str]:
    """ชื่อโฟลเดอร์ใน `DATASETS_DIR` ที่มีไฟล์ `.parquet` อย่างน้อย 1 ไฟล์ — ตัวเลือก dropdown

    โฟลเดอร์ที่ไม่มี parquet / ไฟล์ลอย ๆ ไม่นับ — path ที่ `iter_codes` รับคือ
    ชื่อนี้ (resolve ใต้ `DATASETS_DIR`) หรือ path ตรง ๆ ก็ได้
    """
    root = Path(DATASETS_DIR)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and any(p.rglob("*.parquet"))
    )


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_local_dataset_dir(dataset_id: str) -> Path | None:
    """หาโฟลเดอร์ dataset ท้องถิ่นจากค่า dropdown/path — ไม่พบ/อยู่นอก sandbox → None (ใช้ Hub)

    H1 (Fix.md): เหมือน `resolve_local_model` — resolve แล้วต้องอยู่ใต้
    `DATASETS_DIR` หรือ repo เสมอ (กัน arbitrary file read ผ่าน path traversal)
    """
    roots = project_roots(REPO_ROOT, DATASETS_DIR)
    direct = Path(dataset_id)
    if direct.is_dir():
        resolved = direct.resolve()
        if any(resolved.is_relative_to(root) for root in roots):
            return resolved
    under_root = Path(DATASETS_DIR) / dataset_id
    if under_root.is_dir():
        resolved = under_root.resolve()
        if any(resolved.is_relative_to(root) for root in roots):
            return resolved
    return None


def _read_local_parquet(ds_dir: Path, column: str, limit: int) -> list[str]:
    """อ่านคอลัมน์ `column` จาก `.parquet` ในโฟลเดอร์ ตรง ๆ ทีละ batch (row-group)

    20GB-safe: ไม่ convert เป็น arrow cache ซ้ำ ไม่โหลดทั้งไฟล์ลง RAM
    """
    files = sorted(ds_dir.rglob("*.parquet"))
    if not files:
        raise ValueError(
            f"Local dataset folder '{ds_dir}' has no .parquet file. "
            "Put dataset .parquet file(s) in the folder, or pick a HF Hub dataset."
        )
    out: list[str] = []
    for file in files:
        pf = pq.ParquetFile(file)
        available = pf.schema_arrow.names
        if column not in available:
            raise ValueError(
                f"Column '{column}' not found in local parquet '{file.name}' "
                f"(available: {', '.join(available)}). Fix the Dataset column."
            )
        for batch in pf.iter_batches(batch_size=64, columns=[column]):
            for value in batch.column(0).to_pylist():
                # M2: ค่าไม่ใช่ text (int/None/...) → fail ตรงนี้ดีกว่า crash ลึกที่ .splitlines()
                if not isinstance(value, str):
                    raise ValueError(  # noqa: TRY004 — ข้อมูลใน column ผิด ไม่ใช่ type ของ arg (ตรง convention ฟังก์ชันนี้)
                        f"Column '{column}' in local parquet '{file.name}' contains "
                        f"non-text values ({type(value).__name__} at row {len(out)}) — "
                        "clean the nulls or pick a text column."
                    )
                out.append(value)
                if len(out) >= limit:
                    return out
    return out


def iter_codes(
    dataset_id: str,
    column: str,
    *,
    limit: int,
    split: str = "train",
    cache_dir: str = "data_cache",
) -> list[str]:
    """อ่าน `limit` ตัวอย่างแรก — โฟลเดอร์ `.parquet` ท้องถิ่นอ่านตรง ๆ, Hub ใช้ streaming

    `dataset_id` = path โฟลเดอร์ หรือชื่อใต้ `DATASETS_DIR` → อ่าน local;
    ไม่พบโฟลเดอร์ → ถือเป็น Hub dataset id (พฤติกรรมเดิม, plan.md §4.4)
    """
    local = _resolve_local_dataset_dir(dataset_id)
    if local is not None:
        return _read_local_parquet(local, column, limit)
    ds = load_dataset(dataset_id, split=split, streaming=True, cache_dir=cache_dir)
    return [row[column] for row in islice(ds, limit)]
