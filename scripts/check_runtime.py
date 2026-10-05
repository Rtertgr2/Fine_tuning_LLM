"""XPU runtime sanity check — เกณฑ์ผ่าน Phase 1 (plan.md §6 Phase 1).

รันด้วย: .venv/bin/python scripts/check_runtime.py
exit 0 = ผ่านทุกข้อ, exit 1 = มีข้อตก (พิมพ์สาเหตุทุกบรรทัด)
"""

from __future__ import annotations

import sys
from pathlib import Path

# รันเป็น `python scripts/check_runtime.py` → sys.path[0] = scripts/ ต้องเพิ่ม root ก่อน
# (try/except ImportError เดิมเคยกลืนปัญหานี้เงียบ ๆ — Standards-3 เอาออกแล้วต้องแก้ที่ต้นเหตุ)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import psutil
import torch

from configs.safe_defaults import DEFAULT_OUTPUT_DIR, DISK_MIN_GB, RAM_MIN_GB
from core.infra.hardware import existing_ancestor

GB = 1024**3


def main() -> int:
    failures: list[str] = []

    # (1) identity ของ runtime
    print(f"torch           : {torch.__version__}")
    available = torch.xpu.is_available()
    print(f"xpu available   : {available}")
    if not available:
        print("FAIL: no XPU device visible to torch.xpu (check level-zero + intel-compute-runtime)")
        return 1
    device_name = torch.xpu.get_device_name(torch.xpu.current_device())  # D5: ไม่ hardcode 0
    print(f"device name     : {device_name}")

    # (2) VRAM
    free_b, total_b = torch.xpu.mem_get_info(torch.xpu.current_device())  # D5
    print(f"vram total/free : {total_b / GB:.2f} / {free_b / GB:.2f} GB")
    if total_b <= 0 or free_b < 0:
        failures.append("VRAM values are invalid (total <= 0 or free < 0)")

    # (3) RAM + disk ตาม threshold — disk วัดบน ancestor ที่มีอยู่จริง (output dir อาจยังไม่ถูกสร้าง — D6)
    # path ผูกครั้งเดียว + anchor ที่ repo root — รันจาก CWD อื่นก็วัด disk ถูกที่
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    disk_ancestor = existing_ancestor(output_dir)
    ram_avail = psutil.virtual_memory().available / GB
    disk_free = psutil.disk_usage(disk_ancestor).free / GB
    print(f"disk path       : {disk_ancestor}")
    print(f"ram available   : {ram_avail:.2f} GB (min {RAM_MIN_GB})")
    print(f"disk free       : {disk_free:.2f} GB (min {DISK_MIN_GB})")
    if ram_avail < RAM_MIN_GB:
        failures.append(f"available RAM below {RAM_MIN_GB} GB")
    if disk_free < DISK_MIN_GB:
        failures.append(f"free disk below {DISK_MIN_GB} GB")

    # (4) forward/backward จริงบน XPU
    x = torch.randn(64, 64, device="xpu", requires_grad=True)
    w = torch.randn(64, 64, device="xpu")
    target = torch.randn(64, 64, device="xpu")
    loss = ((x @ w) - target).pow(2).mean()
    loss.backward()
    grad = x.grad
    print(f"fwd/bwd loss    : {loss.item():.4f}")
    if grad is None or not torch.isfinite(grad).all():
        failures.append("forward/backward gradient is not finite")
    elif grad.abs().sum() == 0:
        failures.append("forward/backward gradient is all zeros")
    else:
        print("fwd/bwd grad    : finite + non-zero ✓")

    # (5) bf16 — trainer ใช้ bf16 จริง (fp32 probe ผ่าน ไม่ได้แปลว่า bf16 kernel ใช้ได้) (D7)
    bf16_ok = torch.xpu.is_bf16_supported()
    print(f"bf16 supported  : {bf16_ok}")
    if bf16_ok:
        x16 = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16, requires_grad=True)
        w16 = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
        t16 = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
        loss16 = ((x16 @ w16) - t16).pow(2).mean()
        loss16.backward()
        g16 = x16.grad
        print(f"bf16 fwd/bwd    : loss {loss16.item():.4f}")
        if g16 is None or not torch.isfinite(g16).all():
            failures.append("bf16 forward/backward produced non-finite gradients")
        elif g16.abs().sum() == 0:
            failures.append("bf16 forward/backward produced all-zero gradients")
        else:
            print("bf16 fwd/bwd    : finite + non-zero ✓")
    else:
        print("bf16 unsupported — training would fall back to fp32")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
