"""Tests สำหรับ core/hardware.py — contract dict + ลำดับ status ตามสเปก plan.md §4.1"""

from types import SimpleNamespace

import core.hardware as hw

GB = 1024**3
KEYS = {
    "status",
    "device_name",
    "total_vram_gb",
    "free_vram_gb",
    "ram_available_gb",
    "disk_free_gb",
}


def _patch(
    monkeypatch,
    *,
    has_xpu=True,
    ram_avail=20 * GB,
    disk_free=50 * GB,
    total=12 * GB,
    free=11 * GB,
):
    monkeypatch.setattr(hw.torch.xpu, "is_available", lambda: has_xpu)
    monkeypatch.setattr(
        hw.torch.xpu, "get_device_name", lambda i: "Intel(R) Arc(TM) B580 Graphics"
    )
    monkeypatch.setattr(hw.torch.xpu, "mem_get_info", lambda i: (free, total))
    monkeypatch.setattr(hw.psutil, "virtual_memory", lambda: SimpleNamespace(available=ram_avail))
    monkeypatch.setattr(hw.psutil, "disk_usage", lambda p: SimpleNamespace(free=disk_free))


def test_missing_torch_xpu_attr_is_no_xpu(monkeypatch):
    """P1 D3: hasattr(torch,'xpu') ไม่ใช่ capability check — เอาออกแล้ว behavior ต้องอยู่"""
    monkeypatch.delattr(hw.torch, "xpu")
    result = hw.inspect()
    assert result["status"] == "no_xpu"


def test_xpu_is_available_runtime_error_is_no_xpu(monkeypatch):
    """P1 D3: is_available() raise (driver ไม่พร้อม/API หาย) → no_xpu ไม่ใช่ crash"""

    def _boom():
        raise RuntimeError("XPU driver not ready")

    monkeypatch.setattr(hw.torch.xpu, "is_available", _boom)
    result = hw.inspect()
    assert result["status"] == "no_xpu"


def test_contract_keys_and_types(monkeypatch):
    _patch(monkeypatch)
    result = hw.inspect()
    assert set(result) == KEYS
    assert result["status"] in ("ready", "no_xpu", "insufficient_ram", "insufficient_disk")
    assert isinstance(result["device_name"], str)
    for key in ("total_vram_gb", "free_vram_gb", "ram_available_gb", "disk_free_gb"):
        assert isinstance(result[key], float), key


def test_ready_when_all_ok(monkeypatch):
    _patch(monkeypatch, has_xpu=True, ram_avail=20 * GB, disk_free=50 * GB, total=12 * GB, free=11 * GB)
    result = hw.inspect()
    assert result["status"] == "ready"
    assert result["device_name"] == "Intel(R) Arc(TM) B580 Graphics"
    assert result["total_vram_gb"] == 12.0
    assert result["free_vram_gb"] == 11.0
    assert result["ram_available_gb"] == 20.0
    assert result["disk_free_gb"] == 50.0


def test_no_xpu_takes_precedence(monkeypatch):
    # XPU ไม่มี + RAM/disk ต่ำ → ต้องได้ no_xpu (เช็ค XPU ก่อนเสมอ)
    _patch(monkeypatch, has_xpu=False, ram_avail=15 * GB, disk_free=10 * GB)
    result = hw.inspect()
    assert result["status"] == "no_xpu"
    assert result["device_name"] == ""
    assert result["total_vram_gb"] == 0.0
    assert result["free_vram_gb"] == 0.0
    # RAM/disk ยังวัดได้ตามจริง
    assert result["ram_available_gb"] == 15.0
    assert result["disk_free_gb"] == 10.0


def test_insufficient_ram(monkeypatch):
    _patch(monkeypatch, has_xpu=True, ram_avail=15.9 * GB, disk_free=50 * GB)
    result = hw.inspect()
    assert result["status"] == "insufficient_ram"


def test_insufficient_disk(monkeypatch):
    _patch(monkeypatch, has_xpu=True, ram_avail=20 * GB, disk_free=19.9 * GB)
    result = hw.inspect()
    assert result["status"] == "insufficient_disk"
