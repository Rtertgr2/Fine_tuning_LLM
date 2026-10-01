"""Tests สำหรับ app.py — ห้ามเปิด server ตอน import (test-once: รันพร้อม suite ที่ Task 7)"""

from __future__ import annotations


def test_app_import_does_not_launch():
    """import app ต้องไม่ launch server (มีเฉพาะ __main__ guard) — ตรวจด้วย subprocess กัน state รั่ว"""
    import subprocess
    import sys

    code = (
        "import app; "
        "assert callable(app.main); "
        "assert app.SERVER_HOST == '127.0.0.1'; "
        "assert app.SERVER_PORT == 7860"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
