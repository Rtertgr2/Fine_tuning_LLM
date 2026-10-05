"""Playground inference — predict_middle (spawn target, §5 VRAM)
แยกจาก trainer_worker.py เดิม (Approach B split)
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.data.fim import build_fim_prompt, ensure_fim_tokens
from core.infra.ipc_bridge import error_msg, log_msg
from core.train.args import validate_config
from core.train.runner import latest_checkpoint


def predict_middle(config: dict, prefix: str, suffix: str, queue_) -> None:
    """Playground inference: โหลด base + LoRA จาก checkpoint ล่าสุด → FIM predict middle

    spawn target (process แยก) — ส่งผลลัพธ์เป็น log_msg ลง queue_, ผิด → error_msg + raise
    """
    try:
        validate_config(config)
        registry_path = Path(__file__).resolve().parents[2] / "configs" / "fim_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        fim_tokens = registry[config["fim_registry_key"]]

        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        ensure_fim_tokens(tokenizer, fim_tokens.values())

        from peft import PeftModel

        adapter_dir = latest_checkpoint(config["output_dir"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,  # L4: ปฏิเสธ .bin (pickle) เสมอ
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        device = "xpu" if torch.xpu.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        prompt = build_fim_prompt(prefix, suffix, fim_tokens=fim_tokens)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs, max_new_tokens=256, do_sample=False
        )
        continuation = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        queue_.put(log_msg("INFO", text))
    except Exception as exc:
        queue_.put(error_msg(str(exc), traceback.format_exc()))
        raise
