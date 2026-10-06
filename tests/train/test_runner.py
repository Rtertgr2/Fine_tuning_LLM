"""Tests สำหรับ core/train/runner.py — run_training + atomic checkpoint helpers (สเปก plan.md §4.4)"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeQueue
from core.train import runner as wa


def test_commit_checkpoint_moves_and_cleans(tmp_path):
    tmp = tmp_path / "checkpoint-100.saving"
    tmp.mkdir()
    (tmp / "trainer_state.json").write_text("{}", encoding="utf-8")
    final = tmp_path / "checkpoint-100"
    wa.commit_checkpoint(tmp, final)
    assert (final / "trainer_state.json").exists()
    assert not tmp.exists()  # ไม่มีโฟลเดอร์ชั่วคราวค้าง


def test_commit_checkpoint_overwrites(tmp_path):
    # final มี checkpoint เดิม → commit ใหม่ต้องแทนที่หมด ไม่มี .saving/.old หลงเหลือ
    old = tmp_path / "checkpoint-100"
    old.mkdir()
    (old / "stale.txt").write_text("old", encoding="utf-8")
    tmp = tmp_path / "checkpoint-100.saving"
    tmp.mkdir()
    (tmp / "trainer_state.json").write_text("{}", encoding="utf-8")
    wa.commit_checkpoint(tmp, old)
    assert (old / "trainer_state.json").exists()
    assert not (old / "stale.txt").exists()  # ไฟล์เก่าถูกแทนที่หมด
    assert not list(tmp_path.glob("*.saving"))
    assert not list(tmp_path.glob("*.old"))


def test_is_checkpoint_dir_only_matching_names():
    # atomic path เฉพาะ checkpoint-\d+ เท่านั้น — ถ้าชื่ออื่นผ่าน (เช่น run root ที่มี
    # checkpoint ซ้อนอยู่) commit_checkpoint จะ rename+rmtree ทิ้ง checkpoint ทั้งหมด
    assert wa.is_checkpoint_dir("data_cache/run/checkpoint-100") is True
    assert wa.is_checkpoint_dir("checkpoint-1") is True
    assert wa.is_checkpoint_dir("checkpoint-abc") is False
    assert wa.is_checkpoint_dir("checkpoint-100.saving") is False
    assert wa.is_checkpoint_dir("data_cache/phase3_full") is False


# ---------------------------------------------------------------------------
# Task 3: predict pipeline helpers (latest_checkpoint)
# ---------------------------------------------------------------------------


def test_latest_checkpoint_numeric_order(tmp_path):
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-abc").mkdir()  # ไม่ใช่ checkpoint จริง — ต้องเพิกเฉย
    (tmp_path / "logs").mkdir()
    result = wa.latest_checkpoint(tmp_path)
    assert result == tmp_path / "checkpoint-100"  # numeric ไม่ใช่ lexical ("checkpoint-2" > "checkpoint-100" ถ้าผิด)


def test_latest_checkpoint_none_raises(tmp_path):
    with pytest.raises(ValueError):
        wa.latest_checkpoint(tmp_path)
    (tmp_path / "checkpoint-abc").mkdir()
    with pytest.raises(ValueError):
        wa.latest_checkpoint(tmp_path)  # มีแต่ชื่อปลอม → ยัง raise


def _mk_adapter_ckpt(path: Path, marker: str | None = None) -> None:
    """fixture: checkpoint dir ที่ผ่าน is_checkpoint_dir + มี adapter files (ลอกจาก save tests :347-350)"""
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"fake")
    if marker is not None:
        (path / "marker").write_text(marker, encoding="utf-8")


def test_latest_checkpoint_restores_old_after_interrupted_swap(tmp_path):
    """Sec-12: SIGKILL หลัง final→*.old → resume ต้องกู้ .old กลับ (เดิม: checkpoint หายจากรัน เงียบ ๆ)"""
    root = tmp_path / "out"
    _mk_adapter_ckpt(root / "checkpoint-5.old")
    assert wa.latest_checkpoint(root) == root / "checkpoint-5"
    assert not (root / "checkpoint-5.old").exists()


def test_latest_checkpoint_prefers_saving_over_old(tmp_path):
    """Sec-12: มีทั้ง .old (เก่า) และ .saving (ใหม่ — super()._save เสร็จแล้วตอน commit เริ่ม) → ต้องเอา .saving"""
    root = tmp_path / "out"
    _mk_adapter_ckpt(root / "checkpoint-5.old", marker="old")
    _mk_adapter_ckpt(root / "checkpoint-5.saving", marker="new")
    restored = wa.latest_checkpoint(root)
    assert restored == root / "checkpoint-5"
    assert (restored / "marker").read_text(encoding="utf-8") == "new"
    assert not (root / "checkpoint-5.old").exists()
    assert not (root / "checkpoint-5.saving").exists()


def test_latest_checkpoint_does_not_trust_partial_saving(tmp_path):
    """Sec-12: .saving ที่ไม่มี adapter files (first save โดน kill กลางเขียน) — ห้ามกู้ (อาจ partial)"""
    root = tmp_path / "out"
    saving = root / "checkpoint-7.saving"
    saving.mkdir(parents=True)
    (saving / "trainer_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no checkpoint found"):   # ข้อความ raise เดิม (:479)
        wa.latest_checkpoint(root)


def _stub_training_env(
    monkeypatch,
    *,
    codes: list[str],
    train_calls: list | None = None,
    recorded_codes: list | None = None,
    lora_targets: list | None = None,
) -> None:
    """ตัด dependency หนักของ run_training ทั้งหมด (tokenizer/model/dataset/trainer)

    - train_calls: บันทึกค่า resume_from_checkpoint ทุกครั้งที่ trainer.train ถูกเรียก
    - recorded_codes: เก็บ codes ที่ถูกส่งเข้า build_samples (หลัง filter)
    - lora_targets: เก็บ target_modules ที่ LoraConfig ได้รับ
    """

    class _Tok:
        eos_token = "</s>"

        def encode(self, text, add_special_tokens=False):
            # double ต้องพอให้ build_eval_texts (ของจริง) คำนวณ budget ได้
            return [0] * max(1, len(text))

    class _Trainer:
        def __init__(self, **_kw):
            pass

        def train(self, resume_from_checkpoint=None):
            if train_calls is not None:
                train_calls.append(resume_from_checkpoint)

    monkeypatch.setattr(wa.AutoTokenizer, "from_pretrained", lambda mid: _Tok())
    monkeypatch.setattr(wa, "ensure_fim_tokens", lambda tok, toks: None)
    monkeypatch.setattr(wa, "iter_codes", lambda *a, **k: list(codes))

    def _fake_build_samples(cs, **_kw):
        if recorded_codes is not None:
            recorded_codes.append(list(cs))
        return ["dummy text"]

    monkeypatch.setattr(wa, "build_samples", _fake_build_samples)
    # โมเดล fake มีเฉพาะ q_proj → LoraConfig ต้องได้เฉพาะตัวที่มีจริง (ไม่ใช่ทั้ง 7)
    fake_model = SimpleNamespace(
        named_modules=lambda: iter([("model.layers.0.attn.q_proj", None)])
    )
    monkeypatch.setattr(
        wa.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: fake_model
    )
    if lora_targets is not None:
        _real_lora_config = wa.LoraConfig

        def _recording_lora_config(**kw):
            lora_targets.append(kw.get("target_modules"))
            return _real_lora_config(**kw)

        monkeypatch.setattr(wa, "LoraConfig", _recording_lora_config)
    monkeypatch.setattr(wa, "AtomicSaveTrainer", _Trainer)


def test_run_training_filters_heldout_from_iter_codes(monkeypatch):
    """spec Phase 5: `run_training` ต้องกรอง heldout ก่อนสร้างชุดเทรน (mock iter_codes)"""
    from core.data.dataset_builder import is_heldout

    heldout_code = next(c for c in (f"heldout_{i}" for i in range(5000)) if is_heldout(c))
    train_code = next(c for c in (f"train_{i}" for i in range(5000)) if not is_heldout(c))
    assert heldout_code != train_code

    recorded: list[list[str]] = []
    lora_targets: list = []
    _stub_training_env(
        monkeypatch,
        codes=[heldout_code, train_code],
        recorded_codes=recorded,
        lora_targets=lora_targets,
    )

    cfg = {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": "data_cache/x",
        "max_seq_length": 1024,
        "max_steps": 1,
        "code_limit": 100,
        "lora_rank": 8,
    }
    q = FakeQueue()
    wa.run_training(cfg, q)

    # heldout ห้ามหลุดเข้าชุดเทรน — filter_train_codes ถูก apply กับ output ของ iter_codes
    assert recorded == [[train_code]]
    assert q.messages[-1] == {"type": "status", "state": "finished"}
    assert all(m.get("type") != "error" for m in q.messages)
    # wiring: LoraConfig ได้เฉพาะ target ที่มีในโมเดลจริง (q_proj เท่านั้น ไม่ใช่ทั้ง 7)
    assert lora_targets == [["q_proj"]]


def test_resume_checkpoint_returns_latest_or_none(tmp_path):
    assert wa.resume_checkpoint(tmp_path) is None  # ยังไม่มี checkpoint
    (tmp_path / "checkpoint-5").mkdir()
    assert wa.resume_checkpoint(tmp_path) == str(tmp_path / "checkpoint-5")


def _resume_cfg(output_dir: str, resume: bool = True) -> dict:
    return {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "dataset_id": "smangrul/hf-stack-v1",
        "dataset_column": "content",
        "fim_registry_key": "qwen",
        "output_dir": output_dir,
        "max_seq_length": 1024,
        "max_steps": 1,
        "code_limit": 100,
        "lora_rank": 8,
        "resume": resume,
    }


def test_run_training_resumes_when_flag_set(monkeypatch, tmp_path):
    """M8: resume=True + มี checkpoint → trainer.train ต้องได้ path จริง (ไม่ใช่เริ่มใหม่เงียบ ๆ)"""
    calls: list = []
    _stub_training_env(monkeypatch, codes=["train_dummy"], train_calls=calls)
    (tmp_path / "checkpoint-5").mkdir()
    q = FakeQueue()
    wa.run_training(_resume_cfg(str(tmp_path)), q)
    assert calls == [str(tmp_path / "checkpoint-5")]
    logs = [m["text"] for m in q.messages if m["type"] == "log"]
    assert any(t.startswith("resuming from") for t in logs)
    assert q.messages[-1] == {"type": "status", "state": "finished"}


def test_run_training_resume_without_checkpoint_starts_fresh(monkeypatch, tmp_path):
    """M8: resume=True แต่ไม่มี checkpoint → fresh start + ต้อง log บอก (ห้ามเงียบ)"""
    calls: list = []
    _stub_training_env(monkeypatch, codes=["train_dummy"], train_calls=calls)
    q = FakeQueue()
    wa.run_training(_resume_cfg(str(tmp_path)), q)
    assert calls == [None]
    logs = [m["text"] for m in q.messages if m["type"] == "log"]
    assert any("starting fresh" in t for t in logs)
    assert q.messages[-1] == {"type": "status", "state": "finished"}


def test_run_training_without_resume_flag_passes_none(monkeypatch, tmp_path):
    """M8 back-compat: ไม่มี key resume → resume_from_checkpoint=None (คงพฤติกรรมเดิม)"""
    calls: list = []
    _stub_training_env(monkeypatch, codes=["train_dummy"], train_calls=calls)
    q = FakeQueue()
    wa.run_training(_resume_cfg(str(tmp_path), resume=False), q)
    assert calls == [None]
    logs = [m["text"] for m in q.messages if m["type"] == "log"]
    assert not any("resuming" in t for t in logs)


# ---------------------------------------------------------------------------
# L4: use_safetensors=True ทุกจุดโหลดโมเดล (Fix.md) — ปฏิเสธ .bin (pickle) เสมอ
# ---------------------------------------------------------------------------


def test_all_auto_model_loads_pin_safetensors():
    """AST จับครบทุก call site (runner/predict/export อย่างละ 1 + evaluator×1) — เพิ่มจุดใหม่โดยไม่ใส่ flag = เทสต์ตก"""
    import ast

    root = Path(__file__).resolve().parents[2]
    found = 0
    for rel in (
        "core/train/runner.py",
        "core/train/predict.py",
        "core/train/export.py",
        "core/eval/evaluator.py",
    ):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "from_pretrained":
                continue
            if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "AutoModelForCausalLM"):
                continue
            found += 1
            kws = {kw.arg: kw.value for kw in node.keywords}
            flag = kws.get("use_safetensors")
            assert isinstance(flag, ast.Constant) and flag.value is True, (
                f"{rel}:{node.lineno} ต้องตั้ง use_safetensors=True (L4: ไม่รับ pickle .bin)"
            )
    assert found >= 4  # runner×1 + predict×1 + export×1 + evaluator×1
