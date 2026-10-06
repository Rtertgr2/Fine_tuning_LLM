"""Export workers — save_adapter_only (LoRA files) + merge_export (full weights)
แยกจาก trainer_worker.py เดิม (spawn targets ใน process แยก — §5 VRAM)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.contracts.manifest import build_run_manifest, write_run_manifest
from core.data.fim import ensure_fim_tokens
from core.train.args import validate_config, validate_output_dir
from core.train.runner import latest_checkpoint


def _model_license(model_id: str) -> str | None:
    """best-effort ครั้งเดียวตอน export — ออฟไลน์/hub ล่ม/cardData ไม่มี = None (null-over-zero)"""
    try:
        return model_info(model_id).cardData.get("license")
    except Exception:
        return None


def save_adapter_only(
    output_dir: str | Path, *, config: dict, exports_dir: str | Path = "exports"
) -> Path:
    """คัดลอกเฉพาะไฟล์ LoRA จาก checkpoint ล่าสุด → exports/<output_dir.name>/ (file op ไม่กิน VRAM)

    `config` = REQUIRED_CONFIG_KEYS ครบ — ใช้เขียน run_manifest.json (FT-010, เขียนตอน export สำเร็จเท่านั้น)
    """
    validate_output_dir(output_dir)  # Sec-10: H1 gate ที่ save — คุ้มครอง caller ที่ไม่ผ่าน validate_config
    ckpt = latest_checkpoint(output_dir)
    config_src = ckpt / "adapter_config.json"
    # sharded weights (adapter_model-00001-of-00002.safetensors) + index ต้องโดนด้วย
    weight_srcs = sorted(ckpt.glob("adapter_model*.safetensors"))
    index_srcs = sorted(ckpt.glob("adapter_model*.safetensors.index.json"))
    if not config_src.exists() or (not weight_srcs and not index_srcs):
        raise ValueError(f"checkpoint {ckpt} contains no adapter files")
    dest = Path(exports_dir) / Path(output_dir).name
    dest.mkdir(parents=True, exist_ok=True)
    for src in (config_src, *weight_srcs, *index_srcs):
        shutil.copy2(src, dest / src.name)
    # step สุดท้าย — export พังก่อนหน้า = ห้ามมี manifest ค้างหลอก (invariant #3)
    manifest = build_run_manifest(
        config, export_mode="adapter_only", export_path=dest,
        license=_model_license(config["model_id"]),
    )
    write_run_manifest(manifest, dest)
    return dest


def merge_export(config: dict) -> Path:
    """Merge LoRA เข้า base แล้วบันทึก full weights → exports/<name>-merged/ (spawn target, กิน RAM ~2×)"""
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
    merged = model.merge_and_unload()

    dest = Path("exports") / f"{Path(config['output_dir']).name}-merged"
    dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    # step สุดท้าย — เหมือน save_adapter_only: export สำเร็จเท่านั้น (invariant #3)
    manifest = build_run_manifest(
        config, export_mode="merged", export_path=dest,
        num_params=merged.num_parameters(), dtype="bfloat16",
        license=_model_license(config["model_id"]),
    )
    write_run_manifest(manifest, dest)
    return dest
