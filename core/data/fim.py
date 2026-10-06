"""FIM helpers — ensure_fim_tokens (vocab guard) + build_fim_prompt (PSM)
+ encode_prompt/decode_continuation (predict/eval ใช้ร่วมกัน — P1 D2 + P0 D1)
แยกจาก trainer_worker.py เดิม (Approach B split)
"""

from __future__ import annotations

from collections.abc import Iterable


def ensure_fim_tokens(tokenizer, fim_tokens: Iterable[str]) -> dict[str, int]:
    """ตรวจว่า FIM tokens อยู่ใน vocab จริง — ขาด/ผิด → ValueError พร้อมชื่อ token

    ผ่านเมื่อ: id ไม่ None, id < len(tokenizer), ไม่ใช่ unk, convert_ids_to_tokens roundtrip ตรง
    ห้ามเรียก method resize ใด ๆ — แก้ที่ token ก่อนเทรน ไม่ใช่ขยาย embedding ระหว่างทาง
    """
    ids: dict[str, int] = {}
    problems: list[str] = []
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in fim_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid >= len(tokenizer):
            problems.append(f"{tok} (id={tid} — outside vocab)")
            continue
        if unk is not None and tid == unk:
            problems.append(f"{tok} (maps to unk_token_id — not in vocab)")
            continue
        if tokenizer.convert_ids_to_tokens(tid) != tok:
            problems.append(f"{tok} (roundtrip mismatch — split or mapped wrongly)")
            continue
        ids[tok] = int(tid)
    if problems:
        raise ValueError(
            "FIM token missing/invalid: " + ", ".join(problems) + " — fix tokens before training"
        )
    return ids


def build_fim_prompt(prefix: str, suffix: str, *, fim_tokens: dict) -> str:
    """PSM prompt (ไม่มี middle) — model generate ต่อจาก middle_tok เอง"""
    return (
        f"{fim_tokens['prefix']}{prefix}"
        f"{fim_tokens['suffix']}{suffix}"
        f"{fim_tokens['middle']}"
    )


def encode_prompt(tokenizer, prompt: str) -> dict:
    """encode prompt แบบไม่เติม special token — ใช้ร่วม evaluate_cases + predict_middle

    P1 D2: ตรงฝั่งเทรน (packing ไม่เติม special) — BOS ฝั่งเดียว = prompt บวม ≠ ฝั่ง eval
    """
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)


def decode_continuation(tokenizer, continuation, *, fim_tokens: dict) -> str:
    """กรอง FIM marker id ออกจาก continuation แล้ว decode — ใช้ร่วม evaluate_cases + predict_middle

    P0 D1: marker (id ของ fim_tokens) ไม่ใช่ special token → decode ปล่อยออกมา → กรองเอง
    `continuation` = ids ต่อท้าย input (iterable ของ int หรือ tensor 1 มิติ)
    """
    fim_ids = {tokenizer.convert_tokens_to_ids(t) for t in fim_tokens.values()}
    kept = [int(t) for t in continuation if int(t) not in fim_ids]
    return tokenizer.decode(kept, skip_special_tokens=True)
