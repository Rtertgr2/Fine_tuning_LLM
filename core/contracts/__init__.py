"""สัญญาข้อมูลร่วมของทั้ง 3 subsystem (Phase A — Freeze Contracts)

import ได้เฉพาะ stdlib + `configs.safe_defaults` เท่านั้น — ห้ามดึง
`core.train` / `core.compress` / `core.runtime` เข้ามา (dependency ไหลลงด้านเดียว)
"""

from core.contracts.benchmark import REPORT_KEYS, validate_report
from core.contracts.manifest import RunManifest, build_run_manifest, write_run_manifest
from core.contracts.runtime import LocalRuntime

__all__ = [
    "LocalRuntime",
    "REPORT_KEYS",
    "RunManifest",
    "build_run_manifest",
    "validate_report",
    "write_run_manifest",
]
