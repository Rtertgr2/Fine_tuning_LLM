"""Tests สำหรับ eval.py CLI — --mode, --compare, exit codes (spec §3.3)"""

import json
from pathlib import Path

from core.eval import evaluator as ev
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


def test_min_delta_constants_pinned():
    """M9: gate thresholds ถูกตรึง — เปลี่ยน = breaking change ต้องแก้ test+README+docstring ด้วย"""
    import eval as eval_module

    assert eval_module.MIN_DELTA_EM_PCT == 0.1
    assert eval_module.MIN_DELTA_F1 == 0.01


def test_import_eval_and_evaluator_do_not_load_torch():
    """D10: --compare เป็น pure stdlib — `import eval` ห้ามดึง torch+transformers+peft+trl ทั้งชุด"""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import core.eval.evaluator, eval; "
         "assert 'torch' not in sys.modules, 'torch was pulled in'"],
        capture_output=True, text=True, cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr


def test_compare_rejects_f1_kind_mismatch(tmp_path, capsys):
    """review#5: ต่าง f1_kind → ห้ามเทียบ (คนละความหมายของ metric)"""
    base = {**_result("base", em=5.0, f1=0.2), "f1_kind": "lcs"}
    fine = _result("finetuned", em=6.0, f1=0.25)          # legacy = ไม่มี key
    eval_dir = _write_two(tmp_path, base, fine)
    assert cli_main(["--compare", "--eval-dir", eval_dir]) == 2
    assert "metric semantics differ" in capsys.readouterr().err


def test_compare_rejects_dataset_mismatch(tmp_path, capsys):
    base = {**_result("base", em=5.0, f1=0.2),
            "f1_kind": "lcs", "dataset_id": "a/ds", "dataset_column": "content"}
    fine = {**_result("finetuned", em=6.0, f1=0.25),
            "f1_kind": "lcs", "dataset_id": "b/ds", "dataset_column": "content"}
    eval_dir = _write_two(tmp_path, base, fine)
    assert cli_main(["--compare", "--eval-dir", eval_dir]) == 2
    assert "different datasets" in capsys.readouterr().err


def test_compare_fail_on_tie(tmp_path, capsys):
    # M9: เดิม tie → PASS (+0.0 ผ่าน gate) — ตอนนี้ต้องดีขึ้นเกิน min-delta ถึงจะ PASS
    r = _result("base", em=5.0, f1=0.2)
    eval_dir = _write_two(tmp_path, r, {**r, "mode": "finetuned"})
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 1
    fail_line = next(line for line in out.splitlines() if "FAIL" in line)
    assert "Exact Match" in fail_line and "Token F1" in fail_line  # ชื่อ metric ครบ (spec §3.3)


def test_compare_pass_at_min_delta(tmp_path, capsys):
    base = _result("base", em=10.0, f1=0.5)
    fine = _result("finetuned", em=10.1, f1=0.51)  # ตรง min delta ทั้งคู่ → ผ่าน
    eval_dir = _write_two(tmp_path, base, fine)
    assert cli_main(["--compare", "--eval-dir", eval_dir]) == 0


def test_compare_fail_below_min_delta(tmp_path, capsys):
    base = _result("base", em=10.0, f1=0.5)
    fine = _result("finetuned", em=10.05, f1=0.5)  # EM เกิน 0 แต่ต่ำกว่า min → ยัง FAIL
    eval_dir = _write_two(tmp_path, base, fine)
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 1
    assert "min-delta" in out


def test_compare_fail_when_base_better(tmp_path, capsys):
    base = _result("base", em=10.0, f1=0.5)
    fine = _result("finetuned", em=4.0, f1=0.3)
    eval_dir = _write_two(tmp_path, base, fine)
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 1
    # spec §3.3: FAIL ต้องพิมพ์ชื่อ metric ที่ base ดีกว่า
    fail_line = next(line for line in out.splitlines() if "FAIL" in line)
    assert "Exact Match" in fail_line
    assert "Token F1" in fail_line


def test_compare_fail_names_only_failing_metric(tmp_path, capsys):
    # fine ดีกว่าเรื่อง EM แต่แย่กว่าเรื่อง F1 → FAIL ต้องชี้แค่ Token F1
    base = _result("base", em=4.0, f1=0.5)
    fine = _result("finetuned", em=8.0, f1=0.3)
    eval_dir = _write_two(tmp_path, base, fine)
    rc = cli_main(["--compare", "--eval-dir", eval_dir])
    out = capsys.readouterr().out
    assert rc == 1
    fail_line = next(line for line in out.splitlines() if "FAIL" in line)
    assert "Token F1" in fail_line
    assert "Exact Match" not in fail_line


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


def test_mode_prints_progress_via_callback(monkeypatch, capsys):
    """spec §3.3: --mode ต้องส่ง progress callback ให้ run_eval (พิมพ์ทุก 10 cases)"""
    import eval as eval_module

    def fake_run_eval(config, *, mode, n_cases=100, eval_dir=None, progress=None, **kw):
        assert callable(progress), "CLI ต้องส่ง progress callback"
        progress(10, 100)   # evaluator gating จะเรียกเฉพาะทุก 10 + ครั้งสุดท้าย
        progress(100, 100)
        return {
            "mode": mode,
            "n": 100,
            "exact_match_pct": 1.0,
            "token_f1_mean": 0.1,
        }

    monkeypatch.setattr(eval_module.ev, "run_eval", fake_run_eval)
    rc = cli_main(["--mode", "base", "--eval-dir", "/tmp/x"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "10/100" in out
    assert "100/100" in out
