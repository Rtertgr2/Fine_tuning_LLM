"""Dashboard — 3 แท็บ (Configuration/Pre-flight, Mission Control, Playground & Export) + wiring (สเปก §4.7)

หน้าที่: layout + event wiring เท่านั้น — ไม่มีสถานะฝังอยู่ (state ทั้งหมดอยู่ใน TrainingController
และ gr.State); ปุ่ม Predict/Merge ถูกล็อกทุกรอบ tick ขณะ `training_active` (§5 VRAM Contention)
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr

from configs.safe_defaults import (
    DEFAULT_DATASET_COLUMN,
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    LORA_RANK_DEFAULT,
    MAX_SEQ_LENGTH_CAP,
    MAX_SEQ_LENGTH_DEFAULT,
    MAX_STEPS,
    TRAIN_CODE_LIMIT,
)
from core import evaluator as ev
from core.dataset_builder import list_datasets
from core.estimator import list_models, resolve_local_model
from core.trainer_worker import merge_export, save_adapter_only
from ui.components import build_metric_plot, verdict_style
from ui.controller import run_predict

_DEFAULT_OUTPUT_DIR = "data_cache/finetune_run"  # spec §4 table default (ไม่อยู่ใน safe_defaults — มีที่เดียว)
_STATUS_COLORS = {
    "idle": "#6b7280",
    "starting": "#ca8a04",
    "training": "#ca8a04",
    "saving": "#ca8a04",
    "aborting": "#ca8a04",  # M4: ระหว่าง kill — สีเดียวกับ banner "Aborting…"
    "finished": "#16a34a",
    "aborted": "#dc2626",
}
_START_VERDICTS = ("safe", "warning")
_LOG_TAIL_LINES = 200


def _fim_choices() -> list[str]:
    registry_path = Path(__file__).resolve().parents[1] / "configs" / "fim_registry.json"
    return sorted(json.loads(registry_path.read_text(encoding="utf-8")))


def _collect_config(
    model_id,
    params_b,
    dataset_id,
    dataset_column,
    fim_key,
    lora_rank,
    max_seq_length,
    max_steps,
    code_limit,
    output_dir,
) -> tuple[dict, float | None]:
    """Widget values → run_training config (9 keys) + user_params_b (estimator fallback ไม่ใช่ config)"""
    # dropdown คืนชื่อใต้ models/ → resolve เป็น path ที่ from_pretrained โหลดได้ (Hub id ไม่แตะ)
    local_model = resolve_local_model(model_id)
    config = {
        "model_id": str(local_model) if local_model else model_id,
        "dataset_id": dataset_id,
        "dataset_column": dataset_column,
        "fim_registry_key": fim_key,
        "output_dir": output_dir,
        "max_seq_length": _to_int(max_seq_length, MAX_SEQ_LENGTH_DEFAULT),
        "max_steps": _to_int(max_steps, MAX_STEPS),
        "code_limit": _to_int(code_limit, TRAIN_CODE_LIMIT),
        "lora_rank": _to_int(lora_rank, LORA_RANK_DEFAULT),
    }
    user_params = float(params_b) if params_b else None
    return config, user_params


def _to_int(value, fallback: int) -> int:
    """M7: ช่อง Number ถูกล้าง = None → ต้องใช้ default ไม่ใช่ crash int(None)"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


# --------------------------------------------------------------------------- #
# Handlers — closure รอบ controller, ไม่อ้าง widget (รับค่าผ่าน *args)
# สร้างนอก `with gr.Blocks()` แล้ว wire ด้วย .click/.tick ข้างใน context
# --------------------------------------------------------------------------- #


def _make_handlers(controller) -> dict:
    def on_check(*cfg_values):
        config, user_params = _collect_config(*cfg_values)
        result = controller.preflight(config, user_params)
        _label, color = verdict_style(result.verdict)
        breakdown = "\n".join(
            f"| {name} | {value:.2f} |"
            for name, value in (
                ("weights", result.weights_gb),
                ("trainable", result.trainable_gb),
                ("activations", result.activations_gb),
                ("overhead", result.overhead_gb),
                ("**total required**", result.total_required_gb),
                ("**free VRAM**", result.free_vram_gb),
            )
        )
        md = (
            f"### Environment Check\n"
            f'<span style="color:{color};font-weight:bold">● {result.verdict}</span>'
            f" — {html.escape(str(result.reason))}  \n"  # H2: escape user-derived text
            f"spec source: `{result.spec_source}`\n\n"
            f"| Component | GB |\n|---|---|\n{breakdown}"
        )
        allowed = result.verdict in _START_VERDICTS
        # I2: ระหว่างเทรนห้ามปลดล็อก Start ถึง verdict จะดีแค่ไหน (§5 Process Stacking)
        return md, result.verdict, gr.Button(
            interactive=allowed and not controller.training_active
        )

    def on_start(*cfg_values):
        config, _user_params = _collect_config(*cfg_values)
        result = controller.start(config)
        if result == "started":
            return ""
        # H2: error string อาจฝังค่าที่ user กรอก — escape ก่อนโผล่เป็น HTML
        return f'<span style="color:#dc2626">**Error:** {html.escape(str(result))}</span>'

    def on_abort():
        if controller.abort():
            return '<span style="color:#ca8a04">Aborting… (SIGTERM → SIGKILL if still alive)</span>'
        return '<span style="color:#6b7280">No process to abort</span>'

    def on_tick(verdict):
        snap = controller.tick()
        color = _STATUS_COLORS.get(snap.status, "#6b7280")
        status_md = (
            f'**Status:** <span style="color:{color}">{snap.status.upper()}</span>'
        )
        fig = build_metric_plot(snap.metrics)
        shown = snap.logs[-_LOG_TAIL_LINES:]
        logs = "\n".join(shown)
        if len(snap.logs) > _LOG_TAIL_LINES:
            # M3: บอกให้รู้ว่าถูกตัด — spec ขอ "ครบทุกบรรทัด" แต่ textbox ยาวไม่ได้ (performance)
            hidden = len(snap.logs) - len(shown)
            logs = f"… (+{hidden} older lines not shown)\n" + logs
        banners = []
        if snap.error:
            banners.append(
                f'<span style="color:#dc2626">**Error:** {snap.error}</span>'
            )
        if snap.watchdog:
            banners.append(
                f'<span style="color:#dc2626">**Watchdog:** {snap.watchdog}</span>'
            )
        ta = snap.training_active
        allow_start = (not ta) and verdict in _START_VERDICTS
        return (
            status_md,
            fig,
            logs,
            "\n\n".join(banners),
            gr.Button(interactive=allow_start),
            gr.Button(interactive=ta),
            gr.Button(interactive=not ta),
            gr.Button(interactive=not ta),
            gr.Button(interactive=not ta),  # eval_btn — ล็อกระหว่างเทรน (§5 VRAM Contention)
        )

    def on_predict(*values):
        *cfg_values, prefix, suffix = values
        config, _user_params = _collect_config(*cfg_values)
        try:
            return run_predict(config, prefix, suffix), ""
        except Exception as exc:  # noqa: BLE001 — แสดง error ทุกชนิดใน UI
            return "", f'<span style="color:#dc2626">**Predict failed:** {exc}</span>'

    def on_eval(*cfg_values):
        config, _user_params = _collect_config(*cfg_values)
        try:
            # sequential — base เสร็จค่อย finetuned (คนละ process, ห้าม stacking §5)
            base = controller.run_eval(config, "base")
            fine = controller.run_eval(config, "finetuned")
            rows, _qualitative = ev.compare_results(base, fine)
            return rows, ""
        except Exception as exc:  # noqa: BLE001 — แสดง error ทุกชนิดใน UI (English เสมอ)
            return None, (
                '<span style="color:#dc2626">'
                f"**Evaluation failed:** {html.escape(str(exc))}</span>"
            )

    def on_eval_render():
        """auto-render เมื่อ eval JSON ครบ (spec §3.4 "เลือก: auto-render") — ไม่บังคับรันใหม่"""
        base_path = ev.EVAL_DIR / "base.json"
        fine_path = ev.EVAL_DIR / "finetuned.json"
        if not (base_path.exists() and fine_path.exists()):
            return None, ""  # ยังไม่เคยรัน eval → เงียบ
        try:
            base = json.loads(base_path.read_text(encoding="utf-8"))
            fine = json.loads(fine_path.read_text(encoding="utf-8"))
            rows, _qualitative = ev.compare_results(base, fine)
            return rows, ""
        except Exception as exc:  # noqa: BLE001 — JSON เสีย → แสดงใน UI (English เสมอ)
            return None, (
                '<span style="color:#dc2626">'
                f"**Failed to load eval results:** {html.escape(str(exc))}</span>"
            )

    def on_save_adapter(output_dir):
        try:
            dest = save_adapter_only(output_dir)
            return f'<span style="color:#16a34a">✅ Adapter saved → `{html.escape(str(dest))}`</span>'
        except Exception as exc:  # noqa: BLE001
            return f'<span style="color:#dc2626">**Save failed:** {html.escape(str(exc))}</span>'

    def on_merge(*cfg_values):
        config, _user_params = _collect_config(*cfg_values)
        try:
            dest = merge_export(config)
            return f'<span style="color:#16a34a">✅ Merged weights saved → `{html.escape(str(dest))}`</span>'
        except Exception as exc:  # noqa: BLE001
            return f'<span style="color:#dc2626">**Merge failed:** {html.escape(str(exc))}</span>'

    def on_refresh_choices():
        """สลับมาแท็บ Configuration → detect โฟลเดอร์ models/ + datasets/ ใหม่

        dropdown ทั้งคู่: hub default + โฟลเดอร์ท้องถิ่นที่มีอยู่ ณ ตอนนั้น
        """
        return (
            gr.update(choices=[DEFAULT_MODEL_ID, *list_models()]),
            gr.update(choices=[DEFAULT_DATASET_ID, *list_datasets()]),
        )

    return {
        "on_check": on_check,
        "on_start": on_start,
        "on_abort": on_abort,
        "on_tick": on_tick,
        "on_predict": on_predict,
        "on_eval": on_eval,
        "on_eval_render": on_eval_render,
        "on_save_adapter": on_save_adapter,
        "on_merge": on_merge,
        "on_refresh_choices": on_refresh_choices,
    }


def build_dashboard(controller) -> gr.Blocks:
    """สร้าง Blocks ทั้ง 3 แท็บแล้ว wire events เข้า `controller` — ไม่มีการเรียก controller ตอนสร้าง"""
    fim_choices = _fim_choices()
    h = _make_handlers(controller)

    with gr.Blocks(title="Code Fine-tuning Control Center") as demo:
        verdict_state = gr.State(None)  # ผล preflight ล่าสุด — ใช้ gate ปุ่ม Start

        with gr.Tabs():
            # ---------------------------------------------------------- #
            # Tab 1: Configuration & Pre-flight (9 ฟิลด์ + params fallback)
            # ---------------------------------------------------------- #
            with gr.Tab("Configuration & Pre-flight") as tab1:
                with gr.Row():
                    model_in = gr.Dropdown(
                        choices=[DEFAULT_MODEL_ID, *list_models()],
                        value=DEFAULT_MODEL_ID,
                        allow_custom_value=True,
                        label="Model",
                    )
                    params_in = gr.Number(
                        value=None,
                        label="Parameters (B) — fallback when config fetch fails",
                    )
                with gr.Row():
                    dataset_in = gr.Dropdown(
                        choices=[DEFAULT_DATASET_ID, *list_datasets()],
                        value=DEFAULT_DATASET_ID,
                        allow_custom_value=True,
                        label="Dataset",
                    )
                    column_in = gr.Textbox(
                        value=DEFAULT_DATASET_COLUMN, label="Code column"
                    )
                    fim_in = gr.Dropdown(
                        choices=fim_choices,
                        value="qwen" if "qwen" in fim_choices else fim_choices[0],
                        label="FIM registry",
                    )
                with gr.Row():
                    lora_in = gr.Slider(
                        minimum=8, maximum=32, step=8, value=LORA_RANK_DEFAULT,
                        label="LoRA rank",
                    )
                    seq_in = gr.Slider(
                        minimum=64, maximum=MAX_SEQ_LENGTH_CAP, step=64,
                        value=MAX_SEQ_LENGTH_DEFAULT, label="Max seq length",
                        elem_id="max-seq-length",
                    )
                    steps_in = gr.Number(
                        value=MAX_STEPS, precision=0, label="Max steps"
                    )
                    code_limit_in = gr.Number(
                        value=TRAIN_CODE_LIMIT, precision=0, label="Code limit"
                    )
                    output_in = gr.Textbox(value=_DEFAULT_OUTPUT_DIR, label="Output dir")
                check_btn = gr.Button("Run Environment Check", variant="secondary")
                gauge_md = gr.Markdown(
                    "_Not checked yet — run Environment Check before starting_"
                )

            # ---------------------------------------------------------- #
            # Tab 2: Training Mission Control
            # ---------------------------------------------------------- #
            with gr.Tab("Training Mission Control"):
                with gr.Row():
                    start_btn = gr.Button(
                        "Start Fine-Tuning", variant="primary", interactive=False
                    )
                    abort_btn = gr.Button(
                        "Abort Process", variant="stop", interactive=False
                    )
                status_md = gr.Markdown("**Status:** Idle")
                start_error_md = gr.Markdown("")  # error จากการกด Start/Abort — ห้ามให้ tick ทับ (I1)
                banner_md = gr.Markdown("")  # error/watchdog จาก tick รีเฟรชทุกวินาที
                plot_out = gr.Plot(label="Loss + LR")
                log_out = gr.Textbox(
                    label="Live Log", lines=16, max_lines=16, interactive=False
                )

            # ---------------------------------------------------------- #
            # Tab 3: Playground & Export
            # ---------------------------------------------------------- #
            with gr.Tab("Playground & Export") as tab3:
                prefix_in = gr.Code(label="Prefix", language="python", lines=6)
                suffix_in = gr.Code(label="Suffix", language="python", lines=6)
                predict_btn = gr.Button("Predict Middle", variant="secondary")
                predict_out = gr.Code(
                    label="Predicted middle", language="python",
                    interactive=False, lines=8,
                )
                predict_error_md = gr.Markdown("")
                with gr.Row():
                    save_btn = gr.Button("Save Adapter Only", variant="secondary")
                    merge_btn = gr.Button(
                        "Merge & Export Full Weights", variant="primary"
                    )
                export_msg = gr.Markdown("")

                # Eval (Phase 5, spec §3.4) — ล็อกโดย tick ระหว่างเทรน (§5 VRAM Contention)
                eval_btn = gr.Button("Run Evaluation", variant="secondary")
                eval_table = gr.Dataframe(
                    headers=["Metric", "Base", "Fine-tuned", "Δ"],
                    interactive=False,
                )
                eval_error_md = gr.Markdown("")

        timer = gr.Timer(value=1.0, active=True)

        # -------------------------------------------------------------- #
        # Wiring (ต้องอยู่ใน Blocks context — gradio 6 บังคับ)
        # -------------------------------------------------------------- #
        cfg_inputs = [
            model_in, params_in, dataset_in, column_in, fim_in,
            lora_in, seq_in, steps_in, code_limit_in, output_in,
        ]
        # detect โฟลเดอร์ใหม่ทุกครั้งที่สลับมาแท็บนี้ (dropdown ทั้งคู่)
        tab1.select(h["on_refresh_choices"], outputs=[model_in, dataset_in])

        check_btn.click(
            h["on_check"], inputs=cfg_inputs,
            outputs=[gauge_md, verdict_state, start_btn],
        )
        # P0 C1: Start อยู่ใน group เดียวกับ predict/merge/eval — กัน 2 process โหลดโมเดลพร้อมกัน (OOM)
        start_btn.click(
            h["on_start"], inputs=cfg_inputs, outputs=[start_error_md],
            concurrency_id="model_load",
        )
        abort_btn.click(h["on_abort"], inputs=[], outputs=[start_error_md])
        timer.tick(
            h["on_tick"], inputs=[verdict_state],
            outputs=[
                status_md, plot_out, log_out, banner_md,
                start_btn, abort_btn, predict_btn, merge_btn, eval_btn,
            ],
        )
        # M2: predict + merge ใช้ concurrency_id ร่วม — ห้ามสอง process โหลดโมเดลพร้อมกัน
        predict_btn.click(
            h["on_predict"], inputs=cfg_inputs + [prefix_in, suffix_in],
            outputs=[predict_out, predict_error_md], concurrency_id="model_load",
        )
        save_btn.click(h["on_save_adapter"], inputs=[output_in], outputs=[export_msg])
        merge_btn.click(
            h["on_merge"], inputs=cfg_inputs, outputs=[export_msg],
            concurrency_id="model_load",
        )
        # M2 ร่วมกับ predict/merge — ห้าม eval โหลดโมเดลพร้อมกัน
        eval_btn.click(
            h["on_eval"], inputs=cfg_inputs,
            outputs=[eval_table, eval_error_md], concurrency_id="model_load",
        )
        # auto-render เมื่อ JSON ครบ (spec §3.4) — สลับมาแท็บนี้แล้วตารางเดิมขึ้นทันที
        tab3.select(h["on_eval_render"], outputs=[eval_table, eval_error_md])

    return demo
