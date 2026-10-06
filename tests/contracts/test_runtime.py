"""Phase A สัญญาที่ 3: LocalRuntime ABC — signature lock + ห้ามผูก backend"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from core.contracts import runtime as rt

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cannot_instantiate():
    with pytest.raises(TypeError):
        rt.LocalRuntime()          # abstract — instantiate ไม่ได้


def test_start_signature_locked():
    sig = inspect.signature(rt.LocalRuntime.start)
    assert list(sig.parameters) == ["self", "model_path", "device", "context_tokens"]
    assert sig.parameters["device"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["device"].default == "vulkan"
    assert sig.parameters["context_tokens"].default is None   # None = ปล่อย backend ใช้ค่า native (RM-008)


def test_is_ready_signature_locked():
    sig = inspect.signature(rt.LocalRuntime.is_ready)
    assert list(sig.parameters) == ["self", "timeout_s"]
    assert sig.parameters["timeout_s"].default is inspect.Parameter.empty  # required positional


def test_complete_signature_locked():
    sig = inspect.signature(rt.LocalRuntime.complete)
    assert list(sig.parameters) == ["self", "prompt", "max_tokens", "temperature", "stop"]
    assert sig.parameters["max_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["max_tokens"].default is inspect.Parameter.empty
    assert sig.parameters["temperature"].default is None
    assert sig.parameters["stop"].default is None


def test_stop_signature_locked():
    assert list(inspect.signature(rt.LocalRuntime.stop).parameters) == ["self"]


def test_imports_clean_subprocess():
    """สภาพแวดล้อมสะอาด — sys.modules ใน pytest process มี test ตัวอื่นปนอยู่แล้ว จึงต้องแยก subprocess
    (guard รวม: ทั้ง 3 สัญญา ห้ามดึง subsystem/backend เข้ามา — Global constraint 2)"""
    code = (
        "import sys; "
        "import core.contracts.manifest, core.contracts.benchmark, core.contracts.runtime; "
        "banned = ('core.train', 'core.compress', 'core.runtime', "
        "'core.compress.llama_runner', 'llama_cpp'); "
        "bad = [m for m in banned if any(x == m or x.startswith(m + '.') for x in sys.modules)]; "
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, f"contracts pulled in: {res.stdout} {res.stderr}"
