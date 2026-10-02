"""Tests สำหรับ eval.py CLI — --mode, --compare, exit codes (spec §3.3)"""

import json
from pathlib import Path

from core import evaluator as ev
from eval import main as cli_main


def _result(mode: str, *, em: float, f1: float, n: int = 3, per_case=None) -> dict:
    return {
        "mode": mode,
        "n": n,
        "exact_match_pct": em,
        "token_f1_mean": f1,
        "per_case": per_case if per_case is not None else [],
        "samples": [],
    }


def _write_two(tmp_path: Path, base: dict, finetuned: dict) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "finetuned.json").write_text(json.dumps(finetuned), encoding="utf-8")
    return str(tmp_path)


def test_compare_pass_on_tie(tmp_path, capsys):
    r = _result("base", em=5.0, f1=0.2)
    eval_dir = _write_two(tmp_path, r, {**r, "mode": "finetuned"})
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out
    assert "Exact Match" in out


def test_compare_fail_when_base_better(tmp_path, capsys):
    base = _result("base", em=10.0, f1=0.5)
    fine = _result("finetuned", em=4.0, f1=0.3)
    eval_dir = _write_two(tmp_path, base, fine)
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out


def test_compare_missing_file(tmp_path, capsys):
    rc = cli_main(["--compare", "--eval-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "run --mode" in err


def test_compare_qualitative_pairs():
    base = _result(
        "base",
        em=0.0,
        f1=0.0,
        per_case=[
            {"i": 0, "exact": False, "f1": 0.0, "pred": "wrong", "gt": "right"},
            {"i": 1, "exact": False, "f1": 0.0, "pred": "w2", "gt": "g2"},
        ],
    )
    fine = _result(
        "finetuned",
        em=50.0,
        f1=0.5,
        per_case=[
            {"i": 0, "exact": True, "f1": 1.0, "pred": "right", "gt": "right"},
            {"i": 1, "exact": False, "f1": 0.0, "pred": "w2", "gt": "g2"},
        ],
    )
    rows, qual = ev.compare_results(base, fine)
    assert len(qual) == 1
    assert qual[0]["i"] == 0
    assert qual[0]["base_pred"] == "wrong"
    assert qual[0]["finetuned_pred"] == "right"
    # rows = [metric, base, finetuned, delta] 3 แถว (EM, F1, n)
    assert len(rows) == 3
    assert rows[0][0] == "Exact Match %"
    assert rows[0][1] == "0.0" and rows[0][2] == "50.0"
    assert rows[0][3].startswith("+")  # delta แสดงทิศทางขึ้น/ลง


def test_compare_n_mismatch_is_invalid(tmp_path, capsys):
    # eval set ไม่เท่ากัน = เทียบกันไม่ได้ (Review Focus #4 spec)
    base = _result("base", em=5.0, f1=0.2, n=100)
    fine = _result("finetuned", em=9.0, f1=0.4, n=50)
    eval_dir = _write_two(tmp_path, base, fine)
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    err = capsys.readouterr().err
    assert rc == 2
    assert "n=" in err  # บอกทั้งสองค่า


def test_mode_writes_json(monkeypatch, tmp_path, capsys):
    def fake_run_eval(cfg, *, mode, eval_dir, **kw):
        result = _result(mode, em=50.0, f1=0.5, n=2)
        out = Path(eval_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{mode}.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(ev, "run_eval", fake_run_eval)
    rc = cli_main(["--mode", "base", "--eval-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert (tmp_path / "base.json").exists()
    assert "exact_match" in out


def test_mode_requires_mode_or_compare():
    # ไม่ใส่ flag เลย → argparse error → exit 2 (ไม่ใช่ traceback)
    try:
        rc = cli_main([])
    except SystemExit as exc:  # argparse raise SystemExit — main ต้อง wrap เป็น int
        rc = exc.code
    assert rc == 2


def test_compare_renders_qualitative_to_stdout(tmp_path, capsys):
    base = _result(
        "base",
        em=0.0,
        f1=0.0,
        per_case=[
            {"i": 0, "exact": False, "f1": 0.0, "pred": "wrong", "gt": "right"},
        ],
    )
    fine = _result(
        "finetuned",
        em=100.0,
        f1=1.0,
        per_case=[
            {"i": 0, "exact": True, "f1": 1.0, "pred": "right", "gt": "right"},
        ],
    )
    eval_dir = _write_two(tmp_path, base, fine)
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 0
    assert "right" in out  # ตัวอย่าง qualitative ถูกพิมพ์ (spec §8.3)
