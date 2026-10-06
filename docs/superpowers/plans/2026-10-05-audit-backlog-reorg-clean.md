# Plan 3: Reorg (Approach B) + Clean Pass + Hermeticity

**Date:** 2026-10-05
**Spec:** `docs/superpowers/specs/2026-10-05-audit-backlog-reorg-design.md` (§5 structure, §6 clean scope, §7 policies, §8 commits, §9 risks)
**Base:** `fc128e9` (`fix(wave1)` — suite: 336 passed, 8 deselected)
**Execution:** Native (inline in this session; fresh whole-branch review at end)
**Commits:** `refactor(reorg)` (after Task 7) + `refactor(clean)` (after Task 11) — plan updates between tasks, commits only at wave boundaries per spec §8.

---

## Goal

Restructure `core/` to domain layout (Approach B: `data/`, `eval/`, `infra/`, `compress/`, `train/`), split the 666-line `trainer_worker.py` monolith, move test files to mirror directories, update README structure docs — then execute the clean-code pass (C7, D8, Thai strings, review minors, docstring/import hygiene) as one final commit.

## Architecture / Target Structure (spec §5, verified against code)

```
core/
├── data/         dataset_builder.py, fim.py (from trainer_worker), fim_registry.py
├── eval/         evaluator.py, eval_cli.py, worker.py (from trainer_worker)
├── infra/        hardware.py, estimator.py, ipc_bridge.py
├── compress/     config.py, alias.py, loader.py, llama_runner.py, metrics.py, prompt_builder.py, report.py
├── train/        args.py, callbacks.py, runner.py, export.py, predict.py (all from trainer_worker)
├── app.py  runtime_check.py  ui/  cli/  indexer/  attention/  reward/  health/  evaluation/  graph/  report/
tests/ (mirror: tests/{infra,data,eval,compress,ui}/ + tests/train/ + root stays)
scripts/ eval.py main.py safe_sweep.py (+ unchanged files at root)
```

trainer_worker split (spec §5 exact):

| New module | Contents (from `core/trainer_worker.py`) |
|---|---|
| `core/train/args.py` | `available_lora_targets`, `_xpu_bf16_supported`, `build_training_args`, `peak_xpu_memory_gb`, `validate_config`, `validate_output_dir`, `REPO_ROOT` |
| `core/train/callbacks.py` | `NanGuardCallback`, `AtomicSaveTrainer`, `StreamToQueueCallback` |
| `core/train/runner.py` | `run_training`, `is_checkpoint_dir`, `commit_checkpoint`, `latest_checkpoint`, `_has_adapter_files`, `_recover_interrupted_commit`, `resume_checkpoint` |
| `core/train/export.py` | `save_adapter_only`, `merge_export` |
| `core/train/predict.py` | `predict_middle` |
| `core/data/fim.py` | `ensure_fim_tokens`, `build_fim_prompt` |
| `core/eval/worker.py` | `run_eval_worker`, `EVAL_DONE` |

Circular import runner↔callbacks broken: runner imports callbacks top-level; callbacks imports runner **function-level** inside `AtomicSaveTrainer._save` (D10 precedent: function-level imports break cycles safely).

## Tech Stack

Python 3.10+, pytest (`./.venv/bin/python -m pytest`), safetensors/transformers/peft/torch/trl, gradio, llama-cpp-python (integration only).

## Global Constraints

- ✅ Every behavior change TDD (red→green); pure moves verified by grep + full suite green.
- ✅ Full suite green at every task boundary: `.venv/bin/python -m pytest tests/ -q` → **336 passed, 8 deselected** (until clean commit: **337 passed, 8 deselected** after Task 10 adds one test).
- ✅ Suite command always `.venv/bin/python -m pytest tests/ -q` (workspace `.venv` — set up per Progress-Ledger "Environment").
- ✅ Wave commits only at Task 7 and Task 11; intermediate tasks verify green but do not commit.
- ✅ `is_heldout` algorithm NEVER changes (grep-verify untouched).
- ✅ UI/CLI **string literals** English (comments/docstrings may stay Thai).
- ✅ Exclude `.worktrees/gguf-compression/**` from all greps.
- ✅ Constants pinned in `tests/test_safe_defaults.py` (exception: CLI gate constants co-located + value-pinned).
- ✅ No formatter/linter/pre-commit; no packaging; don't touch `tools/`, data dirs, `diagnosis-issue-draft.md`.
- ⚠ Ratchet: expanding beyond spec §6 / approved additions → stop and ask.

## Review Focus (self-review findings)

1. **Pickle target= sites** (`ui/controller.py` mp targets :123/:267/:336) — `target=name` picks up the module-level name of the function in the controller module after imports are updated; controller's **imports** drive correctness, no string edits needed (verified: targets are bare identifiers `target=...` referencing controller-namespace names, which the updated imports rebind).
2. **`target=` string forms?** Probed: controller uses identifier form — but grep for string-form `target="core..."` during Task 5 to be safe (step included).
3. **`parents[]` recompute** — every `Path(__file__).parents[N]` in moved files changes depth: core files moved one level deeper → `parents[1]`→`parents[2]`; tests moved into subdir → same; documented per file below with probe-verified line numbers.
4. **test_evaluator string patches** — 8 sites patch `core.trainer_worker.*`; after split these resolve to `core.train.*` / `core.data.fim.*`; run_eval's lazy imports make `monkeypatch.setattr` bind to the same function objects.
5. **Hermeticity RED reproduction** — must see exactly `9 failed, 11 passed, 1 deselected` before GREEN.
6. **FakeQueue import** — `tests/conftest.py` at `tests/` root; moved tests keep `from conftest import FakeQueue` working (rootdir on sys.path via pytest; probe already passed 1/1).

---

## Task 1: Hermeticity — compression-runner tests independent of `LLAMA_CPP_DIR`

**Files:** `tests/test_compression_runner.py`
**Interfaces:** file-local autouse fixture; no production code change.
**Probes already run (record):**
- RED command: `LLAMA_CPP_DIR=/nonexistent .venv/bin/python -m pytest tests/test_compression_runner.py -q` → **9 failed, 11 passed, 1 deselected** (failing: `test_start_builds_alias_and_waits_for_health`, `test_reports_port_busy`, `test_rejects_foreign_model`, `test_timeout_error`, `test_keyboard_interrupt_stops_child`, `test_non_object_props_stops_child`, `test_unexpected_error_after_ready_stops_child`, `test_passes_context_size`, `test_omits_context_by_default`).
- `require_tools` is imported into `llama_runner` namespace at `core/compression/llama_runner.py:23` (`from .config import require_tools`), called at `:118`; bound-method probe: `monkeypatch.setattr(llama_runner, "require_tools", lambda: None)` → **PASSED** (module has the attribute).
- Integration test `:219` (`test_build_*`) uses `config.require_tools()` directly → unaffected by the fixture.

**Steps:**
- [ ] 1.1 RED re-verify: run exact RED command above; confirm `9 failed, 11 passed, 1 deselected`.
- [ ] 1.2 GREEN: add file-local fixture to `tests/test_compression_runner.py` (module has `llama_runner` imported):
  ```python
  @pytest.fixture(autouse=True)
  def _no_tool_gate(monkeypatch):
      monkeypatch.setattr(llama_runner, "require_tools", lambda: None)
  ```
- [ ] 1.3 Verify: `LLAMA_CPP_DIR=/nonexistent .venv/bin/python -m pytest tests/test_compression_runner.py -q` → **20 passed, 1 deselected, 0 failed**; full suite → 336 passed, 8 deselected.
- [ ] 1.4 `git add tests/test_compression_runner.py` (staged for wave commit; no commit).

## Task 2: Move `core/{estimator,hardware,ipc_bridge}.py` → `core/infra/`

**Files:** `core/infra/__init__.py` (new), `core/estimator.py`→`core/infra/estimator.py`, `core/hardware.py`→`core/infra/hardware.py`, `core/ipc_bridge.py`→`core/infra/ipc_bridge.py`; importers; moved file `parents[]`.
**Interfaces:** public API unchanged (module paths move).

**Importers to update** (research batch — exhaustive grep re-run in 2.4 to catch stragglers):
- `core/evaluator.py:188` (function-level: `from core.hardware import ...` → `core.infra.hardware`)
- `core/trainer_worker.py:17-18` (`from core import estimator, hardware` → `from core.infra import estimator, hardware`)
- `core/runtime_check.py` (estimator/hardware imports — grep)
- `ui/controller.py:36-41` block (ipc_bridge — grep)
- tests: `tests/test_estimator.py`, `tests/test_hardware.py`, `tests/test_ipc_bridge.py`, `tests/test_evaluator.py`, `tests/test_trainer_worker.py`, `tests/test_runtime_check.py`, `tests/phase5_integration.py` — grep `core.estimator|core.hardware|core.ipc_bridge` and update.
- `main.py`, `eval.py`, `scripts/*` — grep.

**`parents[]`:** `core/estimator.py:99` → `parents[2]` (was `parents[1]` for repo root; now depth +1).

**Steps:**
- [ ] 2.1 `mkdir core/infra`; write `core/infra/__init__.py`; `git mv` the three files.
- [ ] 2.2 Update `core/estimator.py:99` `parents[1]` → `parents[2]`.
- [ ] 2.3 Update all importers listed above (exhaustive grep: `grep -rn --include='*.py' -e 'core\.estimator' -e 'core\.hardware' -e 'core\.ipc_bridge' -e 'from core import.*estimator' . | grep -v .worktrees`).
- [ ] 2.4 Grep-verify zero old-path references (outside `.worktrees`); full suite → **336 passed, 8 deselected**.

## Task 3: Move `{dataset_builder,evaluator,eval_cli}.py` → `core/data|eval/`

**Files:** `core/data/dataset_builder.py` (from `core/dataset_builder.py`), `core/eval/evaluator.py` (from `core/evaluator.py`), `core/eval/eval_cli.py` (from `core/eval_cli.py`); `core/data/__init__.py` may exist (dataset_builder-era? — check; `core/eval/__init__.py` new).
**Interfaces:** module paths move; public API unchanged.

**Importers to update** (research batch + exhaustive grep in 3.4):
- `core/evaluator.py:133` → after move lives at new path; its `from core.dataset_builder import ...` → `from core.data.dataset_builder`
- `eval.py:29` (`from core.evaluator import run_eval` → `core.eval.evaluator`), eval.py may import eval_cli — grep
- `core/eval_cli.py` self refs + `core/trainer_worker.py` (eval_cli/evaluator/dataset_builder imports — grep)
- `ui/controller.py:36-41`, `ui/dashboard.py:29`, `scripts/pipeline_smoke.py:33`, `scripts/*` — grep
- tests: `tests/test_evaluator.py:12` (`from core import evaluator as ev` → `from core.eval import evaluator as ev`), `tests/test_eval_cli.py:6` (`from core import evaluator as ev` → `from core.eval import evaluator as ev`), `tests/test_eval_cli.py:41` **D10 subprocess string**: `import core.evaluator, eval` → `import core.eval.evaluator, eval`, `tests/test_dataset_builder.py`, `tests/phase5_integration.py` — grep.

**`parents[]`:** `core/evaluator.py:199` → `parents[2]`; `core/dataset_builder.py:242` → `parents[2]` (probe-verified lines).

**Steps:**
- [ ] 3.1 `git mv` into `core/data/` and `core/eval/` (write `__init__.py` if missing).
- [ ] 3.2 Update `parents[]` in both moved files (:199, :242 → `parents[2]`).
- [ ] 3.3 Update all importers incl. D10 subprocess string in `tests/test_eval_cli.py:41`.
- [ ] 3.4 Exhaustive grep `core.evaluator|core.eval_cli|core.dataset_builder` → zero (outside `.worktrees`); full suite → **336 passed, 8 deselected**.

## Task 4: Rename `core/compression/` → `core/compress/`

**Files:** package dir rename; all importers (`core/compression/*` internal + external).
**Interfaces:** package path only; submodule names unchanged (`config`, `alias`, `loader`, `llama_runner`, `metrics`, `prompt_builder`, `report`).

**Importers:** `core/compression/*.py` relative imports unaffected (intra-package). External: `core/trainer_worker.py` (if it imports compression), `core/runtime_check.py`, `ui/controller.py`, `ui/dashboard.py`, `scripts/*`, tests `tests/test_compression_*.py` (7 files), `tests/test_compression_config.py` (uses `config.__file__` `parents[2]` — module-based, **unchanged**), `tests/phase5_integration.py` — exhaustive grep in 4.3.

**`parents[]`:** `core/compression/config.py:27` — `Path(__file__).resolve().parents[2]` already yields repo root after rename? **Probe before editing**: old path depth `core/compression/config.py` → same depth after rename (both 2 levels) → `parents[2]` **unchanged**. Verify: old `parents[2]` from `core/compression/config.py` = root; new same depth → no edit. But `core/compression/report.py` etc. — grep all `parents[` in package and reason per depth.

**Steps:**
- [ ] 4.1 `git mv core/compression core/compress`.
- [ ] 4.2 Grep `parents\[` in `core/compress/` → verify each still resolves to repo root (depth unchanged for `config.py:27`; adjust only if probe shows drift).
- [ ] 4.3 Exhaustive grep `core\.compression` → zero (outside `.worktrees`); update all importers.
- [ ] 4.4 Full suite → **336 passed, 8 deselected** (hermeticity fixture from Task 1 now patching `core.compress.llama_runner` — import path updated in test file's import line).

## Task 5: Split `core/trainer_worker.py` → 7 modules

**Files:** `core/train/{args,callbacks,runner,export,predict}.py`, `core/data/fim.py`, `core/eval/worker.py`; delete `core/trainer_worker.py`; update importers.
**Interfaces:** all public names re-exported at new paths; `EVAL_DONE`, `run_training`, `predict_middle`, `save_adapter_only`, `merge_export`, `validate_config`, `ensure_fim_tokens`, `build_fim_prompt`, `run_eval_worker`, `latest_checkpoint` — callers unchanged.

**Split map with probe-verified anchors (old line numbers from research):**

| New file | Functions/classes | Key line anchors (old file) |
|---|---|---|
| `core/train/args.py` | `available_lora_targets`, `_xpu_bf16_supported`, `build_training_args`, `peak_xpu_memory_gb`, `validate_config`, `validate_output_dir`, `REPO_ROOT` | `REPO_ROOT` :341 (`parents[1]`→`parents[2]`) |
| `core/train/callbacks.py` | `NanGuardCallback`, `AtomicSaveTrainer`, `StreamToQueueCallback` | — |
| `core/train/runner.py` | `run_training`, `is_checkpoint_dir`, `commit_checkpoint`, `latest_checkpoint`, `_has_adapter_files`, `_recover_interrupted_commit`, `resume_checkpoint` | registry :406 `parents[1]`→`parents[2]` |
| `core/train/export.py` | `save_adapter_only`, `merge_export` | registry → `parents[2]` |
| `core/train/predict.py` | `predict_middle` | no_grad site :578; registry :554 → `parents[2]` |
| `core/data/fim.py` | `ensure_fim_tokens`, `build_fim_prompt` | — |
| `core/eval/worker.py` | `run_eval_worker`, `EVAL_DONE` | :590 (Thai comment stays); docstring path-comment `core.evaluator` → `core.eval.evaluator` |

**Circular import:** `runner.py` imports `callbacks` top-level; `callbacks.py` imports `from core.train import runner` **function-level** inside `AtomicSaveTrainer._save`.

**Importer updates** (research batch — exhaustive grep in 5.6):
- `ui/controller.py:36-41`: `EVAL_DONE`, `predict_middle`, `run_eval_worker`, `run_training`, `validate_config` → new paths (`core.eval.worker`, `core.train.predict`, `core.train.runner`, `core.train.args`)
- `ui/dashboard.py:210,218`: `save_adapter_only`, `merge_export` → `core.train.export`
- `core/evaluator.py` function-level imports :188-194: `ensure_fim_tokens` → `core.data.fim`; `validate_config` → `core.train.args`; `latest_checkpoint` → `core.train.runner`
- `core/evaluator.py:133`: `build_fim_prompt` → `core.data.fim`
- `core/compression`→`core/compress` callers: `core/compress/llama_eval.py:19` (benchmark imports), `core/benchmark_compression.py:47` (`AutoTokenizer` from transformers direct), `core/runtime_check.py`, `scripts/pipeline_smoke.py:33`
- `core/eval/worker.py` self: imports from `core.train.*`
- tests: `tests/test_trainer_worker.py` (split in Task 6 — its imports updated there), `tests/test_evaluator.py` (8 string patch sites), `tests/test_ui_controller.py:129,525`, `tests/test_compression_llama_eval.py:9`

**`tests/test_evaluator.py` 8 patch sites** (grep `core.trainer_worker` in file): each `core.trainer_worker.X` → resolved new path (`core.train.args.validate_config`, `core.train.runner.latest_checkpoint`, `core.data.fim.ensure_fim_tokens`, `core.data.fim.build_fim_prompt`, ...). run_eval imports these lazily at call time → `monkeypatch.setattr(new_module, name, ...)` binds (probe: `bound method` pattern verified in Task 1).

**`target=` string check:** grep `target="core\.` in `ui/` — probe said identifier form; confirm zero string forms referencing `core.trainer_worker`.

**Steps:**
- [ ] 5.1 Grep `target="core\.` in `ui/` → confirm no string targets reference trainer_worker; note controller mp target sites :123/:267/:336 are identifier-form (imports drive them).
- [ ] 5.2 Create `core/train/__init__.py`; split functions into the 7 files per map, moving code verbatim (update intra-imports and `parents[N]` per file: args :341→[2], runner :406→[2], export→[2], predict :554→[2], worker :590 no path edit except docstring comment).
- [ ] 5.3 Wire circular import: `runner.py` top-level `from core.train.callbacks import ...`; `callbacks.py` function-level `from core.train import runner` inside `_save` only.
- [ ] 5.4 Update importers per list (exhaustive grep `core\.trainer_worker|from core import trainer_worker` → zero outside `.worktrees`).
- [ ] 5.5 Update `tests/test_evaluator.py` 8 patch sites; grep `core.trainer_worker` in `tests/` → zero.
- [ ] 5.6 Delete `core/trainer_worker.py` (`git rm`); full suite → **336 passed, 8 deselected** (test_trainer_worker.py still imports new paths — update its import block in this task's 5.4 so suite runs; file split happens in Task 6).
- [ ] 5.7 `grep -rn 'is_heldout' core/ tests/` → confirm algorithm untouched (only location: `core/data/dataset_builder.py` + tests).

## Task 6: Move 22 test files to mirror dirs + split `test_trainer_worker.py` → 6 files

**Files:** test files per mapping; `tests/conftest.py` (new, holds `FakeQueue`); `tests/train/__init__.py` etc. if needed (pytest rootdir-based — check existing pattern: tests currently flat with no `__init__.py`? probe: `ls tests/` shows flat; mirror dirs need no `__init__.py` if rootdir conftest — but `from conftest import FakeQueue` requires rootdir `tests/` on sys.path: pytest adds rootdir of conftest — `tests/conftest.py` → `tests/` auto-added → subdir tests import `conftest` fine).

**Mapping (no duplicate basenames):**

| Move to | Files |
|---|---|
| `tests/infra/` | `test_estimator.py`, `test_hardware.py`, `test_ipc_bridge.py` |
| `tests/data/` | `test_dataset_builder.py`, `test_fim_registry.py`, **new** `test_fim.py` (from split) |
| `tests/eval/` | `test_evaluator.py`, `test_eval_cli.py`, **new** `test_worker.py` (from split) |
| `tests/compress/` | all `tests/test_compression_*.py` — research counted **8** (`config`, `alias`, `loader`, `llama_runner`, `metrics`, `prompt_builder`, `report`, `llama_eval`) but the `parents[]` probe also hit `test_compression_cli.py:15` and `test_compression_serve.py:9` → **grep `tests/test_compression_*` in 6.1 and move every hit** |
| `tests/ui/` | `test_app.py`, `test_ui_components.py`, `test_ui_controller.py`, `test_ui_dashboard.py` |
| `tests/train/` (new) | `test_args.py`, `test_callbacks.py`, `test_runner.py`, `test_export.py` (from split) |
| **stays root** | `phase5_integration.py`, `test_safe_defaults.py`, `test_runtime_check.py` + any `tests/test_*.py` not matched by the mappings above — **authoritative check in 6.1: `ls tests/*.py` after moves must contain only root-stays files + `conftest.py`** |

**`parents[]` in moved tests:** recompute to `parents[2]` — probes: `test_compression_cli:15`, `test_compression_serve:9`, `test_dataset_builder:13`, `test_eval_cli:42`, `test_fim_registry:6`; grep all `parents[` under tests/ in 6.1 and recompute per moved depth.

**`test_trainer_worker.py` → 6 files** (all 53 tests, counts preserved; `wa` alias = module under test per file):

| New file | `wa` = | Tests (old line anchors) | Shared helpers |
|---|---|---|---|
| `tests/train/test_args.py` | `core.train.args` | :12,:35,:40,:248,:254,:265,:274,:290,:498,:504,:509,:680,:695,:708,:715,:722,:732 | `FULL_CONFIG` :235 → test_args |
| `tests/train/test_callbacks.py` | `core.train.callbacks` | :77,:85,:149,:167,:177,:187,:196,:213,:226,:280,:319 | `_cb_with_states` :141 |
| `tests/train/test_runner.py` | `core.train.runner` | :106,:116,:309,:335,:344,:432,:440,:452,:579,:618,:639,:652,:664 | `_mk_adapter_ckpt` :423, `_stub_training_env` :521, `_resume_cfg` :624, `setattr(wa,…)` patch sites :552-576, **AST test :750** (file list → `core/train/{runner,predict,export}.py`, `core/eval/evaluator.py`; `found >= 4`; `parents[1]`→`parents[2]` in that test's expectation) |
| `tests/train/test_export.py` | `core.train.export` | :359,:372,:377,:381,:390,:408 | — |
| `tests/data/test_fim.py` | `core.data.fim` | :94,:100,:352 | `FakeTokenizer` :51, `FIM_TOKENS` :74 |
| `tests/eval/test_worker.py` | `core.eval.worker` | :462,:481 | `from core.eval import evaluator as ev_mod` :464,:483 |

**`FakeQueue` :131** → `tests/conftest.py`; all 6 files `from conftest import FakeQueue` (probe PASSED: 1/1).

**Steps:**
- [ ] 6.1 Grep `parents\[` across `tests/` → list every hit with file; recompute for each moved file (moved depth +1 → `parents[2]`; `config.__file__`-based :24 unchanged).
- [ ] 6.2 `git mv` the flat test files per mapping; create `tests/{infra,data,eval,compress,ui,train}/` dirs.
- [ ] 6.3 Create `tests/conftest.py` with `FakeQueue` class (copied from `test_trainer_worker.py:131`); remove from original.
- [ ] 6.4 Split `tests/test_trainer_worker.py` → 6 files per table (verbatim move of test functions; `wa` alias per file; shared helpers to matching file; `from conftest import FakeQueue`); sanity: `grep -c 'def test_' tests/train/*.py tests/data/test_fim.py tests/eval/test_worker.py` sums to **53** (old file's count — verify old count first with same grep before deleting).
- [ ] 6.5 Update `parents[]` in moved tests per 6.1 list.
- [ ] 6.6 Full suite → **336 passed, 8 deselected**; `grep -rn 'test_trainer_worker\|from conftest' tests/` → no stale refs to old module name.

## Task 7: README structure block + plan.md paths + pipeline_smoke → **commit `refactor(reorg)`**

**Files:** `README.md:166-188`, `plan.md` (gitignored — on-disk only), manual pipeline run.
**Interfaces:** docs only.

**Steps:**
- [ ] 7.1 Rewrite `README.md:166-188` structure block to target structure (Task list tree above; keep prose tone/formatting).
- [ ] 7.2 Update `plan.md` on-disk `core/` path references (13 refs; `trainer_worker` at :36,:173,:271) to new paths — **not committed** (gitignored).
- [ ] 7.3 Manual integration (spec §7): `.venv/bin/python scripts/pipeline_smoke.py --mode all` (default `all`) → must complete OK (training path touched by split).
- [ ] 7.4 Full suite → **336 passed, 8 deselected**; grep `core\.trainer_worker|core\.compression\b|from core import estimator` → zero.
- [ ] 7.5 **Commit `refactor(reorg)`**: all moved/new files + importers + README; **do NOT add** `diagnosis-issue-draft.md`, `plan.md`. Commit message follows spec §8.
- [ ] 7.6 task-done (ledger: reorg complete, suite green).

## Task 8: C7 + D8 (timer kwarg + dead no_grad wrappers) — suite + grep only

**Files:** `ui/dashboard.py:366`, `core/eval/evaluator.py:144,131`, `core/train/predict.py` (old :578 → new path).
**Interfaces:** none (behavior-preserving).

**Probes already run:**
- C7: `gr.Timer` signature `{'value': 1, 'active': True}` → `active=True` is default; only `active=True` hit in `ui/` is `dashboard.py:366`; `test_ui_dashboard.py:212` uses `training_active=True` (unrelated).
- D8: `evaluate_cases` :144 `with torch.no_grad():` wrapping a call to `@torch.no_grad()`-decorated `generate` → dead; lazy `import torch` at :131 becomes unused after removal (grep-verify no other torch use in function). `predict_middle` :578 same pattern (keep module-level torch import — `bfloat16`/xpu use).

**Steps:**
- [ ] 8.1 C7: `ui/dashboard.py:366` → `gr.Timer(value=1.0)` (drop `active=True`).
- [ ] 8.2 D8: remove both `with torch.no_grad():` wrappers (evaluator `evaluate_cases` + `predict_middle`); de-indent bodies; drop `evaluate_cases`'s function-level `import torch` if grep shows no remaining use in that function (predict keeps torch).
- [ ] 8.3 Verify: full suite → 336 passed, 8 deselected; `grep -n 'active=True' ui/` → zero; `grep -c 'no_grad' core/eval/evaluator.py core/train/predict.py` → only decorator sites remain.

## Task 9: Thai CLI strings → English (scripts only)

**Files:** `scripts/pipeline_smoke.py` (:77,:86,:101,:102,:104,:105,:112,:118,:125,:126,:128,:135), `scripts/smoke_sft.py` (:41,:42,:44,:45,:100,:104,:105).
**Interfaces:** printed/logged string literals only; **comments/docstrings stay Thai**. No tests pin these strings (research-verified).

**Steps:**
- [ ] 9.1 Translate the 19 string literals to English (keep format placeholders `{}`/`%s` intact).
- [ ] 9.2 `grep -nP '[\x{0E00}-\x{0E7F}]' scripts/*.py` → only comments/docstrings remain (manual eyeball of each hit).
- [ ] 9.3 Full suite → 336 passed, 8 deselected.

## Task 10: Review minors (eval --help TDD + check_runtime) + dedupe verification

**Files:** `eval.py:144`, `tests/test_eval_cli.py` (new test), `scripts/check_runtime.py:46-47,:14`.
**Interfaces:** `eval --help` text change (behavior: help string only); `check_runtime` output path correctness (CWD-independent).

**Minor (a) — TDD:**
- RED: add test to `tests/test_eval_cli.py`: monkeypatch `eval.EVAL_N_CASES` (imported into eval.py :23), call `eval.main(["--help"])` catching `SystemExit`, assert `"default 42"` in output → fails (current text `default 100` at :144).
  ```python
  def test_help_reflects_eval_n_cases(monkeypatch, capsys):
      monkeypatch.setattr(ev_cli, "EVAL_N_CASES", 42)  # module = eval.py imported as in file's existing import
      with pytest.raises(SystemExit):
          ev_cli.main(["--help"])
      assert "default 42" in capsys.readouterr().out
  ```
  (alias `ev_cli` = whatever test_eval_cli.py already imports eval.py as — reuse existing import; probe :136 `def main`, :23 `EVAL_N_CASES`.)
- GREEN: `eval.py:144` → `help=f"eval set size (default {EVAL_N_CASES})"`.

**Minor (b) — check_runtime:**
- Edit `scripts/check_runtime.py`: add `REPO_ROOT = Path(__file__).resolve().parents[1]` at :14 (next to sys.path insert); bind `existing_ancestor` once with `REPO_ROOT` at :46-47 (single call, both branches share).

**Dedupe verification (research done — confirm, no edits expected):**
- `EVAL_DIR` single definition `core/../configs/safe_defaults.py:52` ✓ (grep `EVAL_DIR =` → 1 hit).
- Remaining `n_cases=100`: `eval.py:144` (fixed above) + `tests/test_eval_cli.py:227` (fake-signature default — leave).
- `check_runtime` hardcoded `Path("outputs/evals/holdout.jsonl")` — replaced in minor (b).
- `GB = 1024**3` ×4 → **out of scope** (not in spec §6; ratchet).
- README peft/CVE note present at :48-50 → verify-only.

**Steps:**
- [ ] 10.1 RED test for eval --help → run → fail.
- [ ] 10.2 GREEN `eval.py:144` → re-run → pass.
- [ ] 10.3 check_runtime edit; manual run: `cd /tmp && /path/to/.venv/bin/python /path/to/scripts/check_runtime.py` → `disk path` line shows absolute repo path (or dry-read code to confirm same).
- [ ] 10.4 Dedupe greps: `grep -rn 'EVAL_DIR =' core/ configs/` → 1; `grep -rn 'n_cases=100' eval.py tests/` → only :227 fake; README :48-50 present.
- [ ] 10.5 Full suite → **337 passed, 8 deselected** (+1 new test).

## Task 11: Docstring/import audit + final greps → **commit `refactor(clean)`**

**Files:** moved-file docstrings mentioning old module paths (`core.trainer_worker`, `core.evaluator`, `core.compression`) → update path references inside docstrings/comments; import order sanity (stdlib/third/local blocks per file touched in Tasks 8-10).

**Steps:**
- [ ] 11.1 Grep old module names in docstrings/comments: `grep -rn 'core\.trainer_worker\|core\.evaluator\|core\.compression' core/ tests/ scripts/ README.md` → update legitimate path references (docs pointing at moved files); leave historical prose in `docs/` untouched.
- [ ] 11.2 Import-order sanity: no duplicate imports (`grep -c 'import torch' core/eval/evaluator.py` etc.), no unused obvious leftovers from moves (manual scan of the 7 new modules' headers).
- [ ] 11.3 Final greps: `grep -rn 'core\.trainer_worker' . --include='*.py' | grep -v .worktrees` → zero; `grep -nP '[\x{0E00}-\x{0E7F}]' core/**/*.py` hits all comments/docstrings (spot-check); `git status` clean except untracked `diagnosis-issue-draft.md`.
- [ ] 11.4 Full suite → **337 passed, 8 deselected**.
- [ ] 11.5 **Commit `refactor(clean)`**: Tasks 8-11 changes only; exclude `diagnosis-issue-draft.md`, `plan.md`.
- [ ] 11.6 task-done (ledger: clean pass complete, suite green).
