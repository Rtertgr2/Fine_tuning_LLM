"""Unit tests สำหรับ core.compression.llama_runner — server lifecycle + HTTP"""

from __future__ import annotations

import json
import signal
import socket
import subprocess
from pathlib import Path

import pytest

from core.compression import config, llama_runner
from core.compression.llama_runner import ServerHandle

# fixture จริงจาก llama-server b11371 (-v) — 2026-10-03
KV_LOG_LINE = (
    "0.00.196.856 I llama_kv_cache: size =  384.00 MiB ( 32768 cells,  "
    "24 layers,  4/1 seqs), K (f16):  192.00 MiB, V (f16):  192.00 MiB"
)
KV_LOG_LINE_EARLIER = (
    "0.00.196.852 I llama_kv_cache:    Vulkan0 KV buffer size =     0.00 MiB"
)
# /props ของ b11371 — identity อยู่ที่ key "model_alias" (ไม่มี key "model")
PROPS_FIXTURE = json.dumps(
    {
        "default_generation_settings": {"params": {"temperature": 0.8}, "n_ctx": 32768},
        "total_slots": 4,
        "model_alias": "Qwen2.5-Coder-0.5B-q4_k_m",
        "model_ftype": "Q4_K - Medium",
        "model_path": "models/Qwen/Qwen2.5-Coder-0.5B/gguf/Qwen2.5-Coder-0.5B-q4_k_m.gguf",
        "build_info": "b1-99b9548",
        "is_sleeping": False,
    }
)


class FakeProc:
    """Popen เฉย ๆ สำหรับ unit: poll กำหนดเอง, stdout=None → ข้าม reader thread"""

    def __init__(self, rc=None):
        self._rc = rc
        self.stdout = None
        self.events: list = []

    def poll(self):
        return self._rc

    def send_signal(self, sig):
        self.events.append(("signal", sig))

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        return 0

    def kill(self):
        self.events.append(("kill",))


def test_find_free_port_is_unused():
    port = llama_runner.find_free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # bind ได้จริง = ยังว่าง


def test_start_builds_alias_and_waits_for_health(monkeypatch):
    captured_cmd: list = []
    fake = FakeProc(rc=None)

    def fake_popen(cmd, **kwargs):
        captured_cmd.append(cmd)
        return fake

    def fake_http_get(url, **kwargs):
        if url.endswith("/health"):
            return (200, "")
        return (200, json.dumps({"model_alias": "m-q4_k_m"}))  # /props

    monkeypatch.setattr(llama_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_runner, "_http_get", fake_http_get)

    handle = llama_runner.start_server(Path("/tmp/m-q4_k_m.gguf"), port=18080)

    assert handle.url == "http://127.0.0.1:18080"
    assert handle.alias == "m-q4_k_m"
    cmd = captured_cmd[0]
    assert "--alias" in cmd and "m-q4_k_m" in cmd
    assert "-v" in cmd  # ruling: ต้องมี -v เพื่อ parse kv_cache_mb จาก log
    assert "-m" in cmd and "/tmp/m-q4_k_m.gguf" in cmd


def test_start_reports_port_busy(monkeypatch):
    def fake_popen(cmd, **kwargs):
        return FakeProc(rc=1)  # bind fail → ตายทันที

    monkeypatch.setattr(llama_runner.subprocess, "Popen", fake_popen)

    with pytest.raises(config.CompressionError) as exc:
        llama_runner.start_server(Path("/tmp/m.gguf"), port=18081)

    msg = str(exc.value)
    assert "failed to start" in msg
    assert "Try --port" in msg


def test_start_rejects_foreign_model(monkeypatch):
    def fake_popen(cmd, **kwargs):
        return FakeProc(rc=None)

    def fake_http_get(url, **kwargs):
        if url.endswith("/health"):
            return (200, "")
        return (200, json.dumps({"model_alias": "someone-else"}))

    stopped: list = []
    monkeypatch.setattr(llama_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_runner, "_http_get", fake_http_get)
    monkeypatch.setattr(llama_runner, "stop_server", lambda h, **k: stopped.append(h))

    with pytest.raises(config.CompressionError) as exc:
        llama_runner.start_server(Path("/tmp/m-q4_k_m.gguf"), port=18082)

    msg = str(exc.value)
    assert "different model" in msg
    assert "Try another --port" in msg
    assert len(stopped) == 1  # ไม่ leak server ที่เพิ่งเปิด


def test_start_timeout_error(monkeypatch):
    def fake_popen(cmd, **kwargs):
        return FakeProc(rc=None)

    stopped: list = []
    monkeypatch.setattr(llama_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_runner, "_http_get", lambda url, **k: (0, ""))
    monkeypatch.setattr(llama_runner, "stop_server", lambda h, **k: stopped.append(h))

    with pytest.raises(config.CompressionError) as exc:
        llama_runner.start_server(Path("/tmp/m.gguf"), port=18083, timeout=0)

    assert "did not become ready" in str(exc.value)
    assert len(stopped) == 1  # timeout ต้อง cleanup


def test_complete_passes_prompt_and_params_verbatim(monkeypatch):
    captured: dict = {}

    def fake_post(url, body, **kwargs):
        captured["url"] = url
        captured["body"] = body
        return {"content": "GENERATED"}

    monkeypatch.setattr(llama_runner, "_http_post", fake_post)
    handle = ServerHandle(proc=FakeProc(), url="http://127.0.0.1:9", alias="x")
    prompt = "  indented\nline\twith spaces  "

    out = handle.complete(prompt, n_predict=8, temperature=0.0)

    assert out == "GENERATED"
    assert captured["url"] == "http://127.0.0.1:9/completion"
    assert captured["body"] == {
        "prompt": prompt,  # byte เดียวกัน — ไม่ strip/re-encode
        "n_predict": 8,
        "temperature": 0.0,
        "cache_prompt": False,
    }


def test_stop_server_sends_sigint_then_wait():
    fake = FakeProc(rc=None)
    handle = ServerHandle(proc=fake, url="http://127.0.0.1:9", alias="x")

    llama_runner.stop_server(handle, timeout=7)

    assert fake.events == [("signal", signal.SIGINT), ("wait", 7)]  # ไม่ kill ก่อน


def test_stop_server_kills_after_timeout():
    class ImpatientProc(FakeProc):
        def wait(self, timeout=None):
            self.events.append(("wait", timeout))
            if self.events.count(("wait", 10)) == 1 and "kill" not in self.events:
                raise subprocess.TimeoutExpired(cmd="llama-server", timeout=10)
            return 0

    fake = ImpatientProc(rc=None)
    handle = ServerHandle(proc=fake, url="http://127.0.0.1:9", alias="x")

    llama_runner.stop_server(handle, timeout=10)

    kinds = [e[0] for e in fake.events]
    assert kinds.index("kill") > kinds.index("signal")  # SIGINT ก่อน kill เสมอ
    assert "kill" in kinds  # รอไม่ทัน → kill


def test_stop_server_noop_when_already_dead():
    fake = FakeProc(rc=0)
    handle = ServerHandle(proc=fake, url="http://127.0.0.1:9", alias="x")
    llama_runner.stop_server(handle)
    assert fake.events == []  # ไม่แตะ process ที่ตายแล้ว


def test_kv_cache_mb_parses_fixture_and_garbage():
    log = KV_LOG_LINE_EARLIER + "\n" + KV_LOG_LINE + "\nsomething else\n"
    assert llama_runner.kv_cache_mb(log) == 384.0
    assert llama_runner.kv_cache_mb(KV_LOG_LINE) == 384.0
    assert llama_runner.kv_cache_mb("no kv here") is None
    assert llama_runner.kv_cache_mb("") is None


def test_props_fixture_identity_key():
    # fixture ต้องมี key ที่ start_server ใช้ — ถ้า upstream เปลี่ยน key ให้ test นี้แดง
    assert json.loads(PROPS_FIXTURE)["model_alias"] == "Qwen2.5-Coder-0.5B-q4_k_m"


@pytest.mark.integration
def test_server_lifecycle_real():
    try:
        config.require_tools()
    except config.CompressionError:
        pytest.skip("llama.cpp not built")
    try:
        src = config.resolve_source("Qwen/Qwen2.5-Coder-0.5B")
    except config.CompressionError:
        pytest.skip("Qwen model not downloaded")
    gguf = config.artifact_path(src, "q4_k_m")
    if not gguf.is_file():
        pytest.skip("q4_k_m artifact not built yet")

    handle = llama_runner.start_server(gguf, port=None, timeout=120.0)
    try:
        out = handle.complete("def f():", n_predict=8, temperature=0.0)
        assert isinstance(out, str) and out != ""
        assert llama_runner.kv_cache_mb(handle.log_text) is not None  # log จริง parse ได้
    finally:
        llama_runner.stop_server(handle)

    assert handle.proc.poll() is not None  # ปิดจริง ไม่ leak


def test_start_server_keyboard_interrupt_stops_child(monkeypatch):
    """Review Focus #3: Ctrl+C ระหว่างรอ health — ห้าม leak child (ต้อง stop ก่อน propagate)"""

    def fake_popen(cmd, **kwargs):
        return FakeProc(rc=None)

    def interrupting_get(url, **kwargs):
        raise KeyboardInterrupt

    stopped: list = []
    monkeypatch.setattr(llama_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_runner, "_http_get", interrupting_get)
    monkeypatch.setattr(llama_runner, "stop_server", lambda h, **k: stopped.append(h))

    with pytest.raises(KeyboardInterrupt):
        llama_runner.start_server(Path("/tmp/m.gguf"), port=18085)

    assert len(stopped) == 1  # cleanup ก่อน propagate


def test_drain_decodes_bytes_leniently_without_dying():
    """Vulkan driver log มี byte ไม่ใช่ UTF-8 — drain ต้องแทนด้วย � ไม่ใช่ตายกลางทาง

    thread ตาย = ไม่มีคนอ่าน pipe → buffer เต็ม → server แขวน (บั๊กเจอจาก run จริง)
    """
    import io

    stream = io.BytesIO(b"ok line\n\xff\xfe garbage\nlast line\n")
    log: list[str] = []

    llama_runner._drain(stream, log)

    assert log[0] == "ok line\n"
    assert log[-1] == "last line\n"  # อ่านจนจบ ไม่หลุดกลางทาง
    assert len(log) == 3
    assert "garbage" in log[1]


# --- Review #4/#6/#11: cleanup ทุก path, join reader ก่อน parse, log bounded ---


def test_start_server_non_object_props_stops_child(monkeypatch):
    """Review #4: /props คืน JSON ที่ไม่ใช่ object → ต้อง stop child (ไม่ใช่ AttributeError แล้ว leak)"""

    def fake_popen(cmd, **kwargs):
        return FakeProc(rc=None)

    def fake_http_get(url, **kwargs):
        if url.endswith("/health"):
            return (200, "")
        return (200, "[1, 2]")  # valid JSON แต่ไม่ใช่ object

    stopped: list = []
    monkeypatch.setattr(llama_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_runner, "_http_get", fake_http_get)
    monkeypatch.setattr(llama_runner, "stop_server", lambda h, **k: stopped.append(h))

    with pytest.raises(config.CompressionError) as exc:
        llama_runner.start_server(Path("/tmp/m.gguf"), port=18087)

    assert "different model" in str(exc.value)  # fall เข้า alias-mismatch path (English)
    assert len(stopped) == 1


def test_start_server_unexpected_error_after_ready_stops_child(monkeypatch):
    """Review #4: path ไหนหลัง spawn ที่ raise นอกเหนือจาก CompressionError ก็ต้อง cleanup"""

    def fake_popen(cmd, **kwargs):
        return FakeProc(rc=None)

    def fake_http_get(url, **kwargs):
        if url.endswith("/health"):
            return (200, "")
        raise RuntimeError("unexpected boom")  # /props ระเบิดแบบไม่คาดคิด

    stopped: list = []
    monkeypatch.setattr(llama_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_runner, "_http_get", fake_http_get)
    monkeypatch.setattr(llama_runner, "stop_server", lambda h, **k: stopped.append(h))

    with pytest.raises(RuntimeError):
        llama_runner.start_server(Path("/tmp/m.gguf"), port=18088)

    assert len(stopped) == 1  # ไม่ leak ไม่ว่า exception อะไร


def test_stop_server_joins_log_reader():
    """Review #6: stop ต้องรอ reader จบก่อนคืน — caller จะได้ parse log ครบทุกบรรทัด"""
    joined: dict = {}

    class FakeReader:
        def join(self, timeout=None):
            joined["timeout"] = timeout

    handle = ServerHandle(
        proc=FakeProc(rc=0), url="http://127.0.0.1:9", alias="x"
    )
    handle._reader = FakeReader()

    llama_runner.stop_server(handle)  # child ตายแล้ว — เดิม return ก่อน join

    assert joined.get("timeout") is not None


def test_server_log_is_bounded():
    """Review #11: serve เปิดค้าง — log ต้อง bounded (deque) ไม่โตไม่จำกัด"""
    handle = ServerHandle(
        proc=FakeProc(rc=0), url="http://127.0.0.1:9", alias="x"
    )
    total = llama_runner.LOG_MAX_LINES + 50
    for i in range(total):
        handle._log.append(f"L{i}\n")

    assert len(handle._log) == llama_runner.LOG_MAX_LINES
    assert handle.log_text.endswith(f"L{total - 1}\n")  # ท้ายอยู่
    assert "L0\n" not in handle.log_text  # หัวหลุดออก
