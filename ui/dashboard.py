"""Dashboard — 3 แท็บ (Configuration/Pre-flight, Mission Control, Playground & Export) + wiring (สเปก §4.7)

หน้าที่: layout + event wiring เท่านั้น — ไม่มีสถานะฝังอยู่ (state ทั้งหมดอยู่ใน TrainingController
และ gr.State); ปุ่ม Predict/Merge ถูกล็อกทุกรอบ tick ขณะ `training_active` (§5 VRAM Contention)
"""

from __future__ import annotations

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
from core.trainer_worker import merge_export, save_adapter_only
from ui.components import build_metric_plot, verdict_style
from ui.controller import run_predict

_DEFAULT_OUTPUT_DIR = "data_cache/finetune_run"
_STATUS_COLORS = {
    "idle": "#6b7280",
    "starting": "#ca8a04",
    "training": "#ca8a04",
    "saving": "#ca8a04",
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
    config = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_column": dataset_column,
        "fim_registry_key": fim_key,
        "output_dir": output_dir,
        "max_seq_length": int(max_seq_length),
        "max_steps": int(max_steps),
        "code_limit": int(code_limit),
        "lora_rank": int(lora_rank),
    }
    user_params = float(params_b) if params_b else None
    return config, user_params


def build_dashboard(controller) -> gr.Blocks:
    """สร้าง Blocks ทั้ง 3 แท็บแล้ว wire events เข้า `controller` — ไม่มีการเรียก controller ตอนสร้าง"""
    fim_choices = _fim_choices()

    with gr.Blocks(title="Code Fine-tuning Control Center") as demo:
        verdict_state = gr.State(None)  # ผล preflight ล่าสุด — ใช้ gate ปุ่ม Start

        with gr.Tabs():
            # ---------------------------------------------------------- #
            # Tab 1: Configuration & Pre-flight (9 ฟิลด์ + params fallback)
            # ---------------------------------------------------------- #
            with gr.Tab("Configuration & Pre-flight"):
                with gr.Row():
                    model_in = gr.Dropdown(
                        choices=[DEFAULT_MODEL_ID],
                        value=DEFAULT_MODEL_ID,
                        allow_custom_value=True,
                        label="Model",
                    )
                    params_in = gr.Number(
                        value=None, label="Parameters (B) — fallback เมื่อดึง config ไม่ได้"
                    )
                with gr.Row():
                    dataset_in = gr.Textbox(value=DEFAULT_DATASET_ID, label="Dataset ID")
                    column_in = gr.Textbox(value=DEFAULT_DATASET_COLUMN, label="Code column")
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
                    steps_in = gr.Number(value=MAX_STEPS, precision=0, label="Max steps")
                    code_limit_in = gr.Number(
                        value=TRAIN_CODE_LIMIT, precision=0, label="Code limit"
                    )
                    output_in = gr.Textbox(value=_DEFAULT_OUTPUT_DIR, label="Output dir")
                with gr.Row():
                    check_btn = gr.Button("Run Environment Check", variant="secondary")
                gauge_md = gr.Markdown(
                    "_ยังไม่ได้ตรวจ — กด Run Environment Check ก่อนเริ่มเทรน_"
                )

            # ---------------------------------------------------------- #
            # Tab 2: Training Mission Control
            # ---------------------------------------------------------- #
            with gr.Tab("Training Mission Control"):
                with gr.Row():
                    start_btn = gr.Button(
                        "Start Fine-Tuning", variant="primary", interactive=False
                    )
                    abort_btn = gr.Button("Abort Process", variant="stop", interactive=False)
                status_md = gr.Markdown("**Status:** Idle")
                start_error_md = gr.Markdown("")
                plot_out = gr.Plot(label="Loss + LR")
                log_out = gr.Textbox(
                    label="Live Log", lines=16, max_lines=16, interactive=False,
                    show_copy_button=True,
                )

            # ---------------------------------------------------------- #
            # Tab 3: Playground & Export
            # ---------------------------------------------------------- #
            with gr.Tab("Playground & Export"):
                prefix_in = gr.Code(label="Prefix", language="python", lines=6)
                suffix_in = gr.Code(label="Suffix", language="python", lines=6)
                predict_btn = gr.Button("Predict Middle", variant="secondary")
                predict_out = gr.Code(
                    label="Predicted middle", language="python", interactive=False, lines=8
                )
                predict_error_md = gr.Markdown("")
                with gr.Row():
                    save_btn = gr.Button("Save Adapter Only", variant="secondary")
                    merge_btn = gr.Button("Merge & Export Full Weights", variant="primary")
                export_msg = gr.Markdown("")

        timer = gr.Timer(value=1.0, active=True)

    # ------------------------------------------------------------------ #
    # Wiring — handlers เป็น closure รอบ controller (state ไม่อยู่ในไฟล์นี้)
    # ------------------------------------------------------------------ #
    cfg_inputs = [
        model_in, params_in, dataset_in, column_in, fim_in,
        lora_in, seq_in, steps_in, code_limit_in, output_in,
    ]

    def _on_check(*cfg_values):
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
            f'### Environment Check\n'
            f'<span style="color:{color};font-weight:bold">● {result.verdict}</span>'
            f" — {result.reason}  \n"
            f"spec source: `{result.spec_source}`\n\n"
            f"| Component | GB |\n|---|---|\n{breakdown}"
        )
        allowed = result.verdict in _START_VERDICTS
        return md, result.verdict, gr.Button(interactive=allowed)

    check_btn.click(
        _on_check, inputs=cfg_inputs, outputs=[gauge_md, verdict_state, start_btn]
    )

    def _on_start(*cfg_values):
        config, _user_params = _collect_config(*cfg_values)
        result = controller.start(config)
        if result == "started":
            return ""
        return f'<span style="color:#dc2626">**Error:** {result}</span>'

    start_btn.click(_on_start, inputs=cfg_inputs, outputs=[start_error_md])

    def _on_abort():
        if controller.abort():
            return '<span style="color:#ca8a04">Aborting… (SIGTERM → SIGKILL ถ้ายังไม่ตาย)</span>'
        return '<span style="color:#6b7280">ไม่มี process ที่ต้องหยุด</span>'

    abort_btn.click(_on_abort, inputs=[], outputs=[start_error_md])

    def _on_tick(verdict):
        snap = controller.tick()
        color = _STATUS_COLORS.get(snap.status, "#6b7280")
        status_md_html = f'**Status:** <span style="color:{color}">{snap.status.upper()}</span>'
        fig = build_metric_plot(snap.metrics)
        logs = "\n".join(snap.logs[-_LOG_TAIL_LINES:])
        banners = []
        if snap.error:
            banners.append(f'<span style="color:#dc2626">**Error:** {snap.error}</span>')
        if snap.watchdog:
            banners.append(
                f'<span style="color:#dc2626">**Watchdog:** {snap.watchdog}</span>'
            )
        ta = snap.training_active
        allow_start = (not ta) and verdict in _START_VERDICTS
        return (
            status_md_html,
            fig,
            logs,
            "\n\n".join(banners),
            gr.Button(interactive=allow_start),
            gr.Button(interactive=ta),
            gr.Button(interactive=not ta),
            gr.Button(interactive=not ta),
        )

    timer.tick(
        _on_tick,
        inputs=[verdict_state],
        outputs=[
            status_md, plot_out, log_out, start_error_md,
            start_btn, abort_btn, predict_btn, merge_btn,
        ],
    )

    def _on_predict(*values):
        *cfg_values, prefix, suffix = values
        config, _user_params = _collect_config(*cfg_values)
        try:
            return run_predict(config, prefix, suffix), ""
        except Exception as exc:  # noqa: BLE001 — แสดง error ทุกชนิดใน UI
            return "", f'<span style="color:#dc2626">**Predict failed:** {exc}</span>'

    predict_btn.click(
        _on_predict,
        inputs=cfg_inputs + [prefix_in, suffix_in],
        outputs=[predict_out, predict_error_md],
    )

    def _on_save_adapter(output_dir):
        try:
            dest = save_adapter_only(output_dir)
            return f'<span style="color:#16a34a">✅ Adapter saved → `{dest}`</span>'
        except Exception as exc:  # noqa: BLE001
            return f'<span style="color:#dc2626">**Save failed:** {exc}</span>'

    save_btn.click(_on_save_adapter, inputs=[output_in], outputs=[export_msg])

    def _on_merge(*cfg_values):
        config, _user_params = _collect_config(*cfg_values)
        try:
            dest = merge_export(config)
            return f'<span style="color:#16a34a">✅ Merged weights saved → `{dest}`</span>'
        except Exception as exc:  # noqa: BLE001
            return f'<span style="color:#dc2626">**Merge failed:** {exc}</span>'

    merge_btn.click(_on_merge, inputs=cfg_inputs, outputs=[export_msg])

    return demo
