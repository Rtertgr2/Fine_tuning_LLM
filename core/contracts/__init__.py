"""สัญญาข้อมูลร่วมของทั้ง 3 subsystem (Phase A — Freeze Contracts)

import ได้เฉพาะ stdlib + `configs.safe_defaults` เท่านั้น — ห้ามดึง
`core.train` / `core.compress` / `core.runtime` เข้ามา (dependency ไหลลงด้านเดียว)
"""

from core.contracts.manifest import RunManifest, build_run_manifest, write_run_manifest

__all__ = ["RunManifest", "build_run_manifest", "write_run_manifest"]
