"""Tests สำหรับ ui/dashboard.py — โครงสร้าง 3 แท็บ + cap/defaults (test-once: ห้ามรันจนถึง Task 7)"""

from __future__ import annotations

import gradio as gr

from configs.safe_defaults import MAX_SEQ_LENGTH_CAP, MAX_SEQ_LENGTH_DEFAULT
from core.eval.evaluator import PROMPT_VERSION
from ui.dashboard import build_dashboard


class FakeController:
    """duck-typed พอสำหรับ build (ไม่มีการเรียก method ตอนสร้าง Blocks)"""

    training_active = False

    def preflight(self, config, user_params_b=None):
        raise NotImplementedError

    def start(self, config):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


def _iter_blocks(block):
    yield block
    for child in getattr(block, "children", []) or []:
        yield from _iter_blocks(child)


def test_build_dashboard_structure():
    demo = build_dashboard(FakeController())
    assert isinstance(demo, gr.Blocks)

    blocks = list(_iter_blocks(demo))
    tab_labels = [b.label for b in blocks if isinstance(b, gr.Tab)]
    assert tab_labels == [
        "Configuration & Pre-flight",
        "Training Mission Control",
        "Playground & Export",
    ]

    timers = [b for b in blocks if isinstance(b, gr.Timer)]
    assert len(timers) >= 1
    assert timers[0].value == 1.0  # อัปเดตทุก 1 วินาที (§4.3)

    seq_sliders = [
        b for b in blocks
        if isinstance(b, gr.Slider) and b.maximum == MAX_SEQ_LENGTH_CAP
    ]
    assert len(seq_sliders) == 1  # max_seq_length: cap 2048 ตาม §5 — ช่องเดียวที่ชนเพดาน
    assert seq_sliders[0].value == MAX_SEQ_LENGTH_DEFAULT


def test_build_dashboard_locks_exist():
    """ปุ่ม Predict/Merge ต้องถูกสร้าง (interactivity อัปเดตโดย tick — Task 1/2)"""
    demo = build_dashboard(FakeController())
    buttons = [b for b in _iter_blocks(demo) if isinstance(b, gr.Button)]
    # gradio 6: ข้อความปุ่มอยู่ที่ .value (label ถูกใช้กับ event metadata)
    labels = {b.value for b in buttons}
    assert "Predict Middle" in labels
    assert "Merge & Export Full Weights" in labels
    assert "Start Fine-Tuning" in labels
    assert "Abort Process" in labels
    start = next(b for b in buttons if b.value == "Start Fine-Tuning")
    assert start.interactive is False  # ยังไม่ผ่าน preflight → ปิดอยู่


def test_collect_config_tolerates_cleared_number_fields():
    """M7: ช่อง Number ถูกล้าง (None) → ใช้ default จาก safe_defaults ไม่ใช่ crash int(None)"""
    from configs.safe_defaults import (
        LORA_RANK_DEFAULT,
        MAX_SEQ_LENGTH_DEFAULT,
        MAX_STEPS,
        TRAIN_CODE_LIMIT,
    )
    from ui.dashboard import _collect_config

    config, params = _collect_config(
        "m/x", None, "ds", "col", "qwen",
        None, None, None, None, "out",
    )
    assert params is None
    assert config["lora_rank"] == LORA_RANK_DEFAULT
    assert config["max_seq_length"] == MAX_SEQ_LENGTH_DEFAULT
    assert config["max_steps"] == MAX_STEPS
    assert config["code_limit"] == TRAIN_CODE_LIMIT


def test_collect_config_resume_flag():
    """M8: resume checkbox → config['resume'] — default False (10 args เดิมต้องรอด)"""
    from ui.dashboard import _collect_config

    config, _ = _collect_config(*_cfg_args())
    assert config["resume"] is False
    config2, _ = _collect_config(*_cfg_args(), True)
    assert config2["resume"] is True


# ---------------------------------------------------------------------------
# Task 6: Run Evaluation (handler + widgets + tick lock)
# ---------------------------------------------------------------------------


class EvalController:
    """controller ปลอมสำหรับ on_eval — บันทึกการเรียก + คืนค่าที่กำหนด"""

    training_active = False

    def __init__(self, *, fail_on: str | None = None, n_base: int = 2, n_fine: int = 3):
        self.calls: list[str] = []
        self.fail_on = fail_on
        self._n_base = n_base
        self._n_fine = n_fine

    def run_eval(self, config, mode, **kw):
        self.calls.append(mode)
        if self.fail_on == mode:
            raise RuntimeError(f"{mode} exploded")
        return {
            "mode": mode,
            "n": self._n_base if mode == "base" else self._n_fine,
            "exact_match_pct": 10.0 if mode == "base" else 40.0,
            "token_f1_mean": 0.2 if mode == "base" else 0.5,
            "per_case": [
                {
                    "i": 0,
                    "exact": mode == "finetuned",
                    "f1": 1.0 if mode == "finetuned" else 0.0,
                    "pred": "right" if mode == "finetuned" else "wrong",
                    "gt": "right",
                }
            ],
            "samples": [],
            # identity markers ตรงกับ run_eval จริง — compare_results (check_comparable) ใช้
            "f1_kind": "lcs",
            "dataset_id": "a/ds",
            "dataset_column": "content",
            "prompt_version": PROMPT_VERSION,
        }

    # methods ที่ build_dashboard ไม่เรียกตอนสร้าง แต่ใส่ครบตาม duck-type เดิม
    def preflight(self, config, user_params_b=None):
        raise NotImplementedError

    def start(self, config):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


def _cfg_args():
    """ค่า cfg ครบ 11 ช่องตาม cfg_inputs ของ build_dashboard (รวม resume checkbox)"""
    return ("m/x", None, "ds", "col", "qwen", 8, 1024, 6, 64, "out")


def test_on_eval_runs_base_then_finetuned_sequentially():
    """รัน base → finetuned เรียงกัน (คนละ process, ไม่ stacking §5) → compare_results"""
    from ui.dashboard import _make_handlers

    ctl = EvalController()
    h = _make_handlers(ctl)
    rows, err = h["on_eval"](*_cfg_args())
    assert ctl.calls == ["base", "finetuned"]
    assert err == ""
    # rows = [EM, F1, n] — fine ดีกว่า base ทุกแถว
    assert rows[0][0] == "Exact Match %"
    assert float(rows[0][1]) < float(rows[0][2])
    assert rows[2][1] == "2" and rows[2][2] == "3"


def test_on_eval_error_returns_english_message():
    """exception → (None, English markdown) — ห้าม raise ออกจาก handler (ภาษาอังกฤษเสมอ)"""
    from ui.dashboard import _make_handlers

    ctl = EvalController(fail_on="base")
    h = _make_handlers(ctl)
    rows, err = h["on_eval"](*_cfg_args())
    assert rows is None
    # span wrapper สีแดงตาม pattern handler อื่น — ข้อความ English ต้องอยู่ข้างในครบ
    assert "**Evaluation failed:**" in err
    assert "base exploded" in err
    assert err.endswith("</span>")


def test_on_eval_rejects_reports_missing_f1_kind():
    """identity checks (f1_kind/dataset) เดิมมีแค่ใน CLI — UI compare ต้อง reject ด้วย

    legacy รายงานไม่มี f1_kind → compare_results ต้อง raise → handler โชว์ error banner
    แทน render ผลที่เทียบไม่ได้
    """

    class LegacyEvalController(EvalController):
        def run_eval(self, config, mode, **kw):
            result = super().run_eval(config, mode, **kw)
            result.pop("f1_kind", None)  # build เก่า = ไม่มี identity marker
            return result

    from ui.dashboard import _make_handlers

    rows, err = _make_handlers(LegacyEvalController())["on_eval"](*_cfg_args())
    assert rows is None
    assert "**Evaluation failed:**" in err
    assert "metric semantics differ" in err


def test_build_dashboard_has_eval_widgets():
    """Tab 3 มี eval_btn/eval_table/eval_error_md (headers ตรง spec §3.4)"""
    demo = build_dashboard(FakeController())
    blocks = list(_iter_blocks(demo))
    buttons = [b for b in blocks if isinstance(b, gr.Button)]
    assert "Run Evaluation" in {b.value for b in buttons}
    tables = [b for b in blocks if isinstance(b, gr.Dataframe)]
    assert len(tables) == 1
    assert list(tables[0].headers) == ["Metric", "Base", "Fine-tuned", "Δ"]


def test_on_tick_returns_nine_outputs():
    """on_tick คืน 9 แถว (ตัวที่ 9 = eval_btn lock ระหว่างเทรน §5)"""

    class TickController:
        training_active = True
        def preflight(self, *a, **k): raise NotImplementedError
        def start(self, *a): raise NotImplementedError
        def abort(self): raise NotImplementedError
        def tick(self):
            from types import SimpleNamespace
            return SimpleNamespace(
                status="training", metrics=[], logs=[], error=None,
                watchdog=None, training_active=True,
            )

    from ui.dashboard import _make_handlers

    h = _make_handlers(TickController())
    out = h["on_tick"](None)
    assert len(out) == 9
    eval_lock = out[8]
    assert isinstance(eval_lock, gr.Button)
    assert eval_lock.interactive is False  # training_active → ปุ่ม eval ถูกล็อก


def test_on_tick_shows_all_log_lines():
    """Spec-5: live log ครบทุกบรรทัดตาม spec phase4 — เดิมตัดที่ 200 + นับบรรทัดที่หาย"""
    from types import SimpleNamespace
    from ui.dashboard import _make_handlers

    class TickController:
        training_active = False
        def preflight(self, *a, **k): raise NotImplementedError
        def start(self, *a): raise NotImplementedError
        def abort(self, *a): raise NotImplementedError
        def tick(self):
            return SimpleNamespace(
                status="finished", metrics=[],
                logs=[f"[INFO] line {i}" for i in range(250)],
                error=None, watchdog=None, training_active=False,
            )

    h = _make_handlers(TickController())
    out = h["on_tick"](None)
    logs_text = out[2]                      # return tuple: (status, fig, logs, banners, ...)
    assert "[INFO] line 0" in logs_text     # ครบตั้งแต่บรรทัดแรก
    assert "older lines not shown" not in logs_text


def test_lora_rank_widget_only_offers_spec_ranks():
    """Spec-7: LoRA rank ได้แค่ 8/16/32 (spec §4) — Slider step=8 เลือก 24 ได้"""
    demo = build_dashboard(FakeController())
    widgets = [b for b in _iter_blocks(demo)
               if isinstance(b, gr.Dropdown) and b.label == "LoRA rank"]
    assert len(widgets) == 1
    # gradio 6 เก็บ choices เป็น tuple pairs [("8", 8), ...] — value ต้องเป็น int
    assert [v for _label, v in widgets[0].choices] == [8, 16, 32]


# ---------------------------------------------------------------------------
# F5: auto-render เมื่อ eval JSON ครบ (spec §3.4 "เลือก: auto-render")
# ---------------------------------------------------------------------------


def _write_eval_json(dirpath, mode, em, f1, n=3):
    import json as _json

    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"{mode}.json").write_text(
        _json.dumps(
            {
                "mode": mode,
                "n": n,
                "exact_match_pct": em,
                "token_f1_mean": f1,
                "per_case": [],
                "samples": [],
                # identity markers ตรงกับไฟล์ที่ run_eval เขียนจริง
                "f1_kind": "lcs",
                "dataset_id": "a/ds",
                "dataset_column": "content",
                "prompt_version": PROMPT_VERSION,
            }
        ),
        encoding="utf-8",
    )


def test_on_eval_render_missing_files_returns_empty(monkeypatch, tmp_path):
    from ui.dashboard import _make_handlers

    monkeypatch.setattr("ui.dashboard.ev.EVAL_DIR", tmp_path)
    h = _make_handlers(EvalController())
    rows, err = h["on_eval_render"]()
    assert rows is None
    assert err == ""  # ยังไม่เคยรัน eval → เงียบ ไม่ใช่ error


def test_on_eval_render_renders_when_both_jsons_exist(monkeypatch, tmp_path):
    from ui.dashboard import _make_handlers

    monkeypatch.setattr("ui.dashboard.ev.EVAL_DIR", tmp_path)
    _write_eval_json(tmp_path, "base", 1.0, 0.257)
    _write_eval_json(tmp_path, "finetuned", 3.0, 0.328)

    h = _make_handlers(EvalController())
    rows, err = h["on_eval_render"]()
    assert err == ""
    assert rows[0][0] == "Exact Match %"
    assert float(rows[0][2]) > float(rows[0][1])  # fine ดีกว่า base


def test_on_eval_render_corrupt_json_returns_english_error(monkeypatch, tmp_path):
    from ui.dashboard import _make_handlers

    monkeypatch.setattr("ui.dashboard.ev.EVAL_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "base.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "finetuned.json").write_text("{}", encoding="utf-8")

    h = _make_handlers(EvalController())
    rows, err = h["on_eval_render"]()
    assert rows is None
    assert "**Failed to load eval results:**" in err


def test_dashboard_wires_auto_render_event():
    """tab3.select → on_eval_render ต้องถูก wire จริง (api_name ปรากฏใน config)"""
    demo = build_dashboard(FakeController())
    cfg = demo.get_config_file()
    names = [d.get("api_name") for d in cfg.get("dependencies", [])]
    assert "on_eval_render" in names  # config เก็บชื่อไม่มี leading slash (view_api แสดง /)


# ---------------------------------------------------------------------------
# dropdown detect โฟลเดอร์ models/ + datasets/ (design 2026-10-02 picker)
# ---------------------------------------------------------------------------


def test_build_dashboard_model_and_dataset_are_dropdowns():
    """ทั้ง Model และ Dataset ต้องเป็น Dropdown + allow_custom_value (พิมพ์เพิ่มได้)"""
    from configs.safe_defaults import DEFAULT_DATASET_ID, DEFAULT_MODEL_ID

    demo = build_dashboard(FakeController())
    dropdowns = {
        b.label: b for b in _iter_blocks(demo) if isinstance(b, gr.Dropdown)
    }

    model_dd = dropdowns["Model"]
    dataset_dd = dropdowns["Dataset"]

    def _labels(dd):  # gradio 6 normalize choices → (label, value) tuples
        return [c[0] if isinstance(c, tuple) else c for c in dd.choices]

    assert DEFAULT_MODEL_ID in _labels(model_dd)
    assert DEFAULT_DATASET_ID in _labels(dataset_dd)
    assert model_dd.allow_custom_value is True
    assert dataset_dd.allow_custom_value is True


def test_on_refresh_choices_lists_detected_assets(monkeypatch, tmp_path):
    """สลับแท็บ → handler คืน choices ใหม่ที่มีโฟลเดอร์ models/ + datasets/ ที่ detect ได้"""
    from configs.safe_defaults import DEFAULT_DATASET_ID, DEFAULT_MODEL_ID
    from core.data import dataset_builder as db
    from core.infra import estimator as est
    from ui.dashboard import _make_handlers

    models_root = tmp_path / "models"
    (models_root / "local-model").mkdir(parents=True)
    (models_root / "local-model" / "config.json").write_text("{}", encoding="utf-8")
    data_root = tmp_path / "datasets"
    (data_root / "local-ds").mkdir(parents=True)
    (data_root / "local-ds" / "part.parquet").write_bytes(b"")

    monkeypatch.setattr(est, "MODELS_DIR", str(models_root))
    monkeypatch.setattr(db, "DATASETS_DIR", str(data_root))

    h = _make_handlers(FakeController())
    model_upd, dataset_upd = h["on_refresh_choices"]("custom-model", "custom-ds")

    # P1 C3: instance pattern — gr.update() เป็น deprecated path ของ gradio 6
    assert isinstance(model_upd, gr.Dropdown)
    assert isinstance(dataset_upd, gr.Dropdown)
    assert model_upd.value == "custom-model"  # tab switch ห้ามล้าง selection
    assert dataset_upd.value == "custom-ds"

    model_choices = [c[0] for c in model_upd.choices]
    dataset_choices = [c[0] for c in dataset_upd.choices]
    assert DEFAULT_MODEL_ID in model_choices
    assert "local-model" in model_choices
    assert DEFAULT_DATASET_ID in dataset_choices
    assert "local-ds" in dataset_choices


def test_dashboard_wires_refresh_choices_on_tab1():
    """tab1.select → on_refresh_choices ต้องถูก wire จริง (api_name ปรากฏใน config)"""
    demo = build_dashboard(FakeController())
    cfg = demo.get_config_file()
    names = [d.get("api_name") for d in cfg.get("dependencies", [])]
    assert "on_refresh_choices" in names


def test_dashboard_wires_start_in_model_load_group():
    """P0 C1: Start ต้อง serialize กับ eval/predict ที่ค้าง — ไม่งั้น 2 process โหลดโมเดล = OOM"""
    demo = build_dashboard(FakeController())
    by_api = {d.api_name: d for d in demo.fns.values() if getattr(d, "api_name", None)}
    # anchor: predict/merge/eval อยู่ใน group เดิม (M2) — ยืนยันว่าอ่านถูกแหล่ง
    assert by_api["on_predict"].concurrency_id == "model_load"
    assert by_api["on_start"].concurrency_id == "model_load"


def test_dashboard_wires_resume_checkbox_into_cfg_inputs():
    """M8: resume checkbox ต้องอยู่ใน cfg_inputs ทุก handler — ถ้าหลุด → _collect_config
    default False เงียบ ๆ (resume ไม่ทำงานแต่ suite เขียว)"""
    demo = build_dashboard(FakeController())
    cfg = demo.get_config_file()
    assert any(c.get("type") == "checkbox" for c in cfg["components"])  # widget ถูกสร้าง
    by_api = {d.api_name: d for d in demo.fns.values() if getattr(d, "api_name", None)}
    for api in ("on_check", "on_start", "on_merge", "on_eval", "on_predict"):
        assert any(
            isinstance(w, gr.Checkbox) for w in by_api[api].inputs
        ), f"{api} ไม่ได้รับ resume checkbox"


def test_collect_config_resolves_local_model_name(monkeypatch, tmp_path):
    """ชื่อโมเดลจาก dropdown (ใต้ models/) → config.model_id ต้องเป็น path ที่โหลดได้จริง

    จุดเดียวที่ทุก flow (check/start/eval/merge/predict) ผ่าน → resolve ที่นี่ทีเดียว
    """
    from core.infra import estimator as est
    from ui.dashboard import _collect_config

    models_root = tmp_path / "models"
    (models_root / "alpha").mkdir(parents=True)
    (models_root / "alpha" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(est, "MODELS_DIR", str(models_root))

    args = list(_cfg_args())
    args[0] = "alpha"
    config, _params = _collect_config(*args)
    assert config["model_id"] == str(models_root / "alpha")

    # Hub id (ไม่ใช่โฟลเดอร์) → ไม่แตะ คงพฤติกรรมเดิม
    config2, _ = _collect_config(*_cfg_args())
    assert config2["model_id"] == "m/x"


# ---------------------------------------------------------------------------
# H2: HTML escape (Fix.md) — ค่าจาก user/exception ห้ามโผล่เป็น HTML ตรง ๆ
# ---------------------------------------------------------------------------


class _StartCtl:
    """controller ปลอม: start คืน error string ที่ฝัง payload แปลกปลอม"""

    training_active = False

    def __init__(self, error: str):
        self._error = error

    def start(self, config):
        return self._error

    def preflight(self, config, user_params_b=None):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


class _CheckCtl:
    """controller ปลอม: preflight คืน reason ที่ฝัง payload แปลกปลอม"""

    training_active = False

    def preflight(self, config, user_params_b=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            verdict="OK",
            reason='model <img src=x onerror="alert(1)"> fits',
            spec_source="default",
            weights_gb=1.0,
            trainable_gb=0.1,
            activations_gb=0.5,
            overhead_gb=0.2,
            total_required_gb=1.8,
            free_vram_gb=8.0,
        )

    def start(self, config):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


def test_on_start_error_escapes_html():
    """H2: error จาก controller (อาจฝัง model_id ของ user) ต้อง escape ก่อนโผล่ Markdown"""
    from ui.dashboard import _make_handlers

    h = _make_handlers(_StartCtl("bad model <img src=x onerror=alert(1)>"))
    out = h["on_start"](*_cfg_args())
    assert "<img" not in out
    assert "&lt;img" in out
    assert "**Error:**" in out  # ข้อความ English เดิมยังอยู่ครบ


def test_on_eval_error_escapes_html():
    """H2: exception message (อาจฝัง path/เนื้อหา dataset) ต้อง escape"""
    from ui.dashboard import _make_handlers

    class Evil(EvalController):
        def run_eval(self, config, mode, **kw):
            raise RuntimeError(f"{mode} <script>alert(1)</script> exploded")

    rows, err = _make_handlers(Evil())["on_eval"](*_cfg_args())
    assert rows is None
    assert "<script>" not in err
    assert "&lt;script&gt;" in err


def test_on_save_adapter_escapes_dest_and_error(monkeypatch):
    """H2: dest มาจาก output_dir ของ user, exc มาจาก exception — คู่ทั้งสองต้อง escape"""
    from ui import dashboard as dash

    h = dash._make_handlers(_StartCtl("unused"))
    monkeypatch.setattr(
        dash, "save_adapter_only", lambda _p: (_ for _ in ()).throw(ValueError("boom <b>x</b>"))
    )
    err_out = h["on_save_adapter"]("out")
    assert "<b>" not in err_out and "&lt;b&gt;" in err_out

    monkeypatch.setattr(dash, "save_adapter_only", lambda _p: 'dir<i>"quoted"</i>')
    ok_out = h["on_save_adapter"]("out")
    assert "<i>" not in ok_out and "&lt;i&gt;" in ok_out


def test_on_merge_error_escapes_html(monkeypatch):
    """H2: merge exception (อาจฝังชื่อไฟล์/พาธ ของ user) ต้อง escape"""
    from ui import dashboard as dash

    h = dash._make_handlers(_StartCtl("unused"))
    monkeypatch.setattr(
        dash, "merge_export", lambda _c: (_ for _ in ()).throw(ValueError("nope <u>x</u>"))
    )
    out = h["on_merge"](*_cfg_args())
    assert "<u>" not in out and "&lt;u&gt;" in out


def test_on_check_reason_escapes_html():
    """H2: preflight reason อาจฝัง model_id/path ของ user — ต้อง escape ใน Markdown"""
    from ui.dashboard import _make_handlers

    md, verdict, _btn = _make_handlers(_CheckCtl())["on_check"](*_cfg_args())
    assert "<img" not in md
    assert "&lt;img" in md
    assert verdict == "OK"


# ---------------------------------------------------------------------------
# model path จริงแต่อยู่นอก sandbox → _collect_config raise ValueError
# ก่อนที่ handler จะถึง controller — ต้องโชว์ error แทนที่จะ raise ออกจาก handler
# ---------------------------------------------------------------------------


def _outside_model_args(tmp_path):
    """ค่า cfg ที่ชี้โมเดลเป็น path จริงอยู่นอก repo/models/ → resolve_local_model raise

    ชื่อโฟลเดอร์ฝัง `<b>` เพื่อพิสูจน์ว่า error message ถูก escape (H2)
    """
    outside = tmp_path / "outside<b>model"
    outside.mkdir()
    args = list(_cfg_args())
    args[0] = str(outside)
    return args


def test_on_check_outside_model_path_error_keeps_start_disabled(tmp_path):
    """ค่า outside sandbox → on_check คืน error md + verdict ไม่ผ่าน gate + Start ปิด"""
    from ui.dashboard import _make_handlers

    md, verdict, btn = _make_handlers(_CheckCtl())["on_check"](
        *_outside_model_args(tmp_path)
    )
    assert "**Error:**" in md
    assert "<b>" not in md and "&lt;b&gt;" in md  # escape (H2)
    assert verdict not in ("safe", "warning")  # on_tick ไม่ปลดล็อก Start
    assert isinstance(btn, gr.Button) and btn.interactive is False


def test_on_start_outside_model_path_returns_escaped_error(tmp_path):
    """on_start: _collect_config raise → คืน error span เดิม ไม่ใช่ raise ออกจาก handler"""
    from ui.dashboard import _make_handlers

    out = _make_handlers(_StartCtl("unused"))["on_start"](
        *_outside_model_args(tmp_path)
    )
    assert "**Error:**" in out
    assert "<b>" not in out and "&lt;b&gt;" in out


def test_on_predict_outside_model_path_returns_escaped_error(tmp_path):
    from ui.dashboard import _make_handlers

    out, err = _make_handlers(_StartCtl("unused"))["on_predict"](
        *_outside_model_args(tmp_path), "pre", "suf"
    )
    assert out == ""
    assert "**Predict failed:**" in err
    assert "<b>" not in err and "&lt;b&gt;" in err


def test_on_eval_outside_model_path_returns_escaped_error(tmp_path):
    from ui.dashboard import _make_handlers

    rows, err = _make_handlers(EvalController())["on_eval"](
        *_outside_model_args(tmp_path)
    )
    assert rows is None
    assert "**Evaluation failed:**" in err
    assert "<b>" not in err and "&lt;b&gt;" in err


def test_on_merge_outside_model_path_returns_escaped_error(tmp_path):
    from ui.dashboard import _make_handlers

    out = _make_handlers(_StartCtl("unused"))["on_merge"](
        *_outside_model_args(tmp_path)
    )
    assert "**Merge failed:**" in out
    assert "<b>" not in out and "&lt;b&gt;" in out
