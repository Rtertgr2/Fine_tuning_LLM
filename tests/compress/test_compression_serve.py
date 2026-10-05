"""Unit tests สำหรับ scripts/serve.py — start/stop lifecycle + flag resolve"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

CLI_PATH = Path(__file__).resolve().parents[2] / "scripts" / "serve.py"


def _load():
    spec = importlib.util.spec_from_file_location("serve", CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _handle(url="http://127.0.0.1:8099", alias="m-q4_k_m", wait=None):
    proc = SimpleNamespace(wait=wait or (lambda: None))
    return SimpleNamespace(url=url, alias=alias, proc=proc)


def test_serve_prints_url_then_stops_child_on_keyboard_interrupt(capsys):
    mod = _load()
    handle = _handle(wait=lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    stopped: list = []

    rc = mod.serve(
        Path("/tmp/m.gguf"),
        port=8099,
        device="vulkan",
        start=lambda gguf, *, port, device: handle,
        stop=lambda h: stopped.append(h),
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Server ready: http://127.0.0.1:8099" in out
    assert "(model: m-q4_k_m)" in out
    assert "Server stopped." in out
    assert out.index("Server ready:") < out.index("Server stopped.")
    assert stopped == [handle]  # ปิด child ก่อนพิมพ์ stopped / คืนค่า (ไม่ leak)


def test_serve_start_error_returns_1(capsys):
    mod = _load()
    stopped: list = []

    def failing_start(gguf, *, port, device):
        raise mod.CompressionError("llama-server failed to start: boom.")

    rc = mod.serve(
        Path("/tmp/m.gguf"),
        port=8099,
        device="vulkan",
        start=failing_start,
        stop=lambda h: stopped.append(h),
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "llama-server failed to start" in captured.out + captured.err
    assert stopped == []  # ไม่มี handle → ไม่ต้อง stop


def test_serve_gguf_missing_error(tmp_path, capsys):
    mod = _load()
    bad = tmp_path / "x.bin"
    bad.write_text("?", encoding="utf-8")

    rc = mod.main(["--gguf", str(bad)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "no .gguf" in captured.out + captured.err


def test_serve_model_variant_resolves_artifact_path(monkeypatch, tmp_path):
    mod = _load()
    src = tmp_path / "m"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    seen: dict = {}

    monkeypatch.setattr(mod, "resolve_source", lambda m: src)

    def fake_artifact(source, variant):
        seen["variant"] = variant
        return tmp_path / f"{source.name}-{variant}.gguf"

    monkeypatch.setattr(mod, "artifact_path", fake_artifact)

    def fake_start(gguf, *, port, device):
        seen["start_gguf"] = gguf
        seen["port"] = port
        raise mod.CompressionError("stop after resolve")

    monkeypatch.setattr(mod, "start_server", fake_start)

    rc = mod.main(["--model", "alpha", "--variant", "q8_0", "--port", "8123"])

    assert rc == 1
    assert seen == {
        "variant": "q8_0",
        "start_gguf": tmp_path / "m-q8_0.gguf",
        "port": 8123,
    }


def test_ensure_sigint_reinstalls_when_inherited_ignored():
    """background/nohup job inherit SIGINT=ignore — serve ต้องติด handler คืน
    (contract 'รอ KeyboardInterrupt' เป็นจริงแม้ inherited-ignored; interactive เดิมไม่เปลี่ยน)"""
    import signal as signal_mod

    mod = _load()
    previous = signal_mod.getsignal(signal_mod.SIGINT)
    try:
        signal_mod.signal(signal_mod.SIGINT, signal_mod.SIG_IGN)
        mod._ensure_sigint()
        assert signal_mod.getsignal(signal_mod.SIGINT) is signal_mod.default_int_handler

        # สถานะปกติ (interactive) → ไม่แตะอะไร
        signal_mod.signal(signal_mod.SIGINT, signal_mod.default_int_handler)
        mod._ensure_sigint()
        assert signal_mod.getsignal(signal_mod.SIGINT) is signal_mod.default_int_handler
    finally:
        signal_mod.signal(signal_mod.SIGINT, previous)


def test_serve_spontaneous_exit_reports_failure(capsys):
    """Review #12: server ตายเอง (ไม่ใช่ user interrupt) → rc 1 + rc/log tail

    ห้ามพิมพ์ "Server stopped." แล้วคืน 0 เหมือนเป็นการหยุดปกติ
    """
    mod = _load()
    handle = _handle(wait=lambda: 1)
    handle.log_text = "srv  starting\nllama_server: device lost\n"
    stopped: list = []

    rc = mod.serve(
        Path("/tmp/m.gguf"),
        port=8099,
        device="vulkan",
        start=lambda gguf, *, port, device: handle,
        stop=lambda h: stopped.append(h),
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "exited unexpectedly (rc=1)" in captured.err
    assert "device lost" in captured.err  # log tail โชว์ให้เห็นสาเหตุ
    assert "Server stopped." not in captured.out
    assert stopped == [handle]  # ยัง stop (noop) ก่อนคืนค่า
