# Graph Report - Fine-tuning-LLM  (2026-10-06)

## Corpus Check
- 87 files · ~85,956 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2 file(s) not represented in the graph (top: (none) 1, .ini 1)

## Summary
- 1408 nodes · 2996 edges · 78 communities (63 shown, 15 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 117 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Evaluation Metrics & Cases
- Benchmark CLI Tests
- VRAM Estimator & Model Resolve
- Eval Compare & Identity Gate
- GGUF Quality Evaluation
- Benchmark Report Builder
- Performance Benchmark Runner
- Compression Config Paths
- Server Lifecycle Management
- Hardware Inspector
- Dashboard Test Fakes
- Safe Defaults & Estimator
- FIM Decode Helpers
- Training Args Factory
- Trainer Runner Tests
- Quantizer Artifact Tests
- Process Abort & IPC
- UI Controller Tests
- UI Dashboard Tests
- Eval Controller Flow
- Training Controller Core
- Training Worker Pipeline
- Gradio App Entry
- GGUF Convert & Quantize
- Eval Worker Process
- Val Metric UI Components
- LoRA Export Workers
- Optimization Roadmap & Papers
- FIM Helpers & Export
- Training Callbacks & Messages
- Eval Test Factories
- Process Test Fakes
- Serve CLI Script
- llama-runner HTTP Core
- Dataset Loader Tests
- Shared Test Helpers
- Phase 5 Integration Tests
- Train Log Streaming
- Merge Export & Config Collect
- Pipeline Smoke Script
- Training Sample Builder
- Dataset FIM Cutting
- Historical Phase Plans
- Dataset Builder Tests
- Runner Unit Tests
- Server Stop & Log Reader
- Token Truncation Tests
- Dependency Pins & Guides
- Local Parquet Reader
- Serve Process Fakes
- Safe Defaults Tests
- Predict Worker Fakes
- Phase A Contracts Spec
- Abort & Preflight Tests
- SFT Compat Spike
- Leakage Filter Tests
- Checkpoint Recovery Tests
- Phase 5 Eval Docs Plan
- Approach B Reorg Docs
- Migration Phases A-G
- Audit Wave & Supply Chain
- FIM Registry Tests
- Check Control Escape Tests
- 3-Part Architecture Plan
- NaN Guard Tests
- Near-dup Leakage Guard
- Phase 6a Compression Docs
- Broken Queue Tests
- On-Tick Output Tests
- Live Log Lines Test
- Resume Checkpoint Tests
- Vulkan Check Test
- Log Drain Decode Test
- App Import Test
- PSM Format Test
- XPU Compat Spike Docs
- llama.cpp Setup Script

## God Nodes (most connected - your core abstractions)
1. `_make_handlers()` - 39 edges
2. `make_controller()` - 33 edges
3. `start_server()` - 32 edges
4. `run_training()` - 31 edges
5. `TrainingController` - 28 edges
6. `resolve_model_spec()` - 27 edges
7. `valid_config()` - 27 edges
8. `FakeTok` - 25 edges
9. `_load_cli()` - 24 edges
10. `FakeProc` - 23 edges

## Surprising Connections (you probably didn't know these)
- `Quantization Variants (FP16 / Q8_0 / Q4_K_M)` --shares_data_with--> `Benchmark Subsystem (BM)`  [INFERRED]
  docs/superpowers/specs/2026-10-03-phase6a-gguf-compression-design.md → WORK_PLAN.md
- `Interrupted Checkpoint Swap Recovery (Security #12)` --implements--> `FT-001..FT-010 Fine-tuning Task Group`  [INFERRED]
  docs/superpowers/plans/2026-10-05-audit-backlog-p2-wave1.md → WORK_PLAN.md
- `Guardrail Mapping Section (§4)` --references--> `6 Work Safety Dimensions`  [INFERRED]
  docs/superpowers/specs/2026-10-02-phase5-eval-docs-design.md → WORK_PLAN.md
- `main()` --indirect_call--> `start_server()`  [INFERRED]
  scripts/serve.py → core/compress/llama_runner.py
- `main()` --indirect_call--> `stop_server()`  [INFERRED]
  scripts/serve.py → core/compress/llama_runner.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Phase A Frozen Contracts (RunManifest / Benchmark Report / LocalRuntime)** — docs_superpowers_specs_2026_10_06_phase_a_contracts_design_run_manifest_contract, docs_superpowers_specs_2026_10_06_phase_a_contracts_design_benchmark_report_schema_v1, docs_superpowers_specs_2026_10_06_phase_a_contracts_design_local_runtime_abc [EXTRACTED 1.00]
- **FT / BM / RM Subsystem Trio** — work_plan_finetuning_subsystem, work_plan_benchmark_subsystem, work_plan_run_local_model_subsystem [EXTRACTED 1.00]
- **Migration Phases A-G Participate in System Migration** — work_plan_phase_a_freeze_contracts, work_plan_phase_b_finetuning_isolation, work_plan_phase_c_benchmark_extraction, work_plan_phase_d_runtime_abstraction, work_plan_phase_e_llama_cpp_python, work_plan_phase_f_remove_old_server, work_plan_phase_g_cleanup_docs [EXTRACTED 1.00]

## Communities (78 total, 15 thin omitted)

### Community 0 - "Evaluation Metrics & Cases"
Cohesion: 0.06
Nodes (37): build_eval_cases(), EvalCase, evaluate_cases(), exact_match(), _lcs_length(), run_eval(), token_f1(), _codes_where() (+29 more)

### Community 1 - "Benchmark CLI Tests"
Cohesion: 0.06
Nodes (36): _baseline_args(), FakeHandle, _load_cli(), _row(), test_baseline_dict_returns_matching(), test_baseline_dict_skips_identity_mismatch(), test_baseline_dict_skips_legacy_without_identity(), test_check_quant_regression_flags_em_drop_beyond_tolerance() (+28 more)

### Community 2 - "VRAM Estimator & Model Resolve"
Cohesion: 0.07
Nodes (35): classify(), estimate(), resolve_local_model(), resolve_model_spec(), _hw_ready(), _patch_fetch(), test_blocked_when_params_unknown(), test_classify_boundaries() (+27 more)

### Community 3 - "Eval Compare & Identity Gate"
Cohesion: 0.09
Nodes (28): check_comparable(), compare_results(), _build_config(), main(), _run_compare(), _legacy(), _result(), test_compare_fail_below_min_delta() (+20 more)

### Community 4 - "GGUF Quality Evaluation"
Cohesion: 0.08
Nodes (18): evaluate_with_llama(), build_fim_prompt(), _cases(), FakeServer, FakeTok, test_empty_predictions_score_zero(), test_eval_prompt_verbatim_equals_build_fim_prompt(), test_evaluate_with_llama_real_server() (+10 more)

### Community 5 - "Benchmark Report Builder"
Cohesion: 0.09
Nodes (23): build_report(), _fmt(), format_table(), pick_baseline(), write_report(), ModelSpecUnavailable, _baseline_dict(), build_parser() (+15 more)

### Community 6 - "Performance Benchmark Runner"
Cohesion: 0.08
Nodes (17): _drain_text(), parse_bench_output(), _parse_vram(), run_bench(), sample_memory(), _vmhwm_mb(), _xpu_smi_output(), device_args() (+9 more)

### Community 7 - "Compression Config Paths"
Cohesion: 0.11
Nodes (20): artifact_path(), convert_script(), gguf_dir(), report_id(), report_path(), require_tools(), resolve_source(), tool_path() (+12 more)

### Community 8 - "Server Lifecycle Management"
Cohesion: 0.09
Nodes (18): start_server(), test_start_builds_alias_and_waits_for_health(), fake_http_get(), fake_popen(), test_start_rejects_foreign_model(), fake_popen(), test_start_reports_port_busy(), fake_popen() (+10 more)

### Community 9 - "Hardware Inspector"
Cohesion: 0.11
Nodes (14): existing_ancestor(), inspect(), _xpu_available(), main(), _patch(), test_contract_keys_and_types(), test_existing_ancestor_walks_up(), test_inspect_uses_current_device_not_hardcoded_zero() (+6 more)

### Community 10 - "Dashboard Test Fakes"
Cohesion: 0.09
Nodes (13): FakeController, _iter_blocks(), test_build_dashboard_has_eval_widgets(), test_build_dashboard_locks_exist(), test_build_dashboard_model_and_dataset_are_dropdowns(), test_build_dashboard_structure(), test_dashboard_wires_auto_render_event(), test_dashboard_wires_refresh_choices_on_tab1() (+5 more)

### Community 11 - "Safe Defaults & Estimator"
Cohesion: 0.11
Nodes (6): EstimateResult, ModelSpec, _spec_from_config(), project_roots(), validate_output_dir(), _xpu_bf16_supported()

### Community 12 - "FIM Decode Helpers"
Cohesion: 0.09
Nodes (10): decode_continuation(), FakeTokenizer, PromptTok, test_build_fim_prompt_exact(), test_decode_continuation_all_markers_is_empty(), test_decode_continuation_converts_tensor_to_list_once(), test_decode_continuation_drops_fim_marker_ids(), test_encode_prompt_disables_special_tokens() (+2 more)

### Community 13 - "Training Args Factory"
Cohesion: 0.10
Nodes (21): available_lora_targets(), build_training_args(), peak_xpu_memory_gb(), validate_config(), test_args_disable_eval_when_no_heldout(), test_args_enable_validation_loop(), test_args_eval_batch_equals_train(), test_args_overrides_for_smoke() (+13 more)

### Community 14 - "Trainer Runner Tests"
Cohesion: 0.10
Nodes (15): latest_checkpoint(), _mk_adapter_ckpt(), test_all_auto_model_loads_pin_safetensors(), test_commit_checkpoint_moves_and_cleans(), test_commit_checkpoint_overwrites(), test_latest_checkpoint_does_not_trust_partial_saving(), test_latest_checkpoint_final_wins_over_artifact_same_step(), test_latest_checkpoint_has_no_side_effects() (+7 more)

### Community 15 - "Quantizer Artifact Tests"
Cohesion: 0.16
Nodes (20): build_artifacts(), _fake_writing_run(), fake_run(), _outfile_of(), _src(), test_build_artifacts_command_order_and_mapping(), fake_run(), test_build_artifacts_real_qwen_smoke() (+12 more)

### Community 16 - "Process Abort & IPC"
Cohesion: 0.11
Nodes (12): abort_process(), watchdog_error(), FakeProcess, NeverStartedProcess, test_abort_escalates_to_kill(), test_abort_never_started_process_is_safe(), test_abort_no_kill_when_terminate_works(), test_metric_msg_shape() (+4 more)

### Community 17 - "UI Controller Tests"
Cohesion: 0.12
Nodes (14): _fake_ready_hw(), make_controller(), test_abort_without_process_returns_false(), test_error_msg_surfaces_error_and_traceback_in_logs(), test_invalid_msg_becomes_error(), test_preflight_fallback_seq_length_is_default(), test_preflight_passthrough_blocked(), test_preflight_safe_estimate() (+6 more)

### Community 18 - "UI Dashboard Tests"
Cohesion: 0.11
Nodes (12): _outside_model_args(), _StartCtl, test_on_check_outside_model_path_error_keeps_start_disabled(), test_on_eval_outside_model_path_returns_escaped_error(), test_on_eval_render_renders_when_both_jsons_exist(), test_on_merge_error_escapes_html(), test_on_merge_outside_model_path_returns_escaped_error(), test_on_predict_outside_model_path_returns_escaped_error() (+4 more)

### Community 19 - "Eval Controller Flow"
Cohesion: 0.08
Nodes (11): _cfg_args(), EvalController, test_collect_config_resolves_local_model_name(), test_collect_config_resume_flag(), test_on_eval_error_escapes_html(), test_on_eval_error_returns_english_message(), test_on_eval_rejects_reports_missing_f1_kind(), run_eval() (+3 more)

### Community 21 - "Training Worker Pipeline"
Cohesion: 0.13
Nodes (9): run_training(), _resume_cfg(), _stub_training_env(), test_run_training_filters_heldout_from_iter_codes(), test_run_training_recovers_interrupted_swap_without_resume_flag(), test_run_training_repairs_swap_before_resume_scan(), test_run_training_resume_without_checkpoint_starts_fresh(), test_run_training_resumes_when_flag_set() (+1 more)

### Community 22 - "Gradio App Entry"
Cohesion: 0.11
Nodes (10): main(), list_datasets(), list_models(), test_list_datasets_missing_dir_is_empty(), test_list_datasets_returns_only_parquet_folders(), test_list_models_missing_dir_is_empty(), test_list_models_returns_only_config_folders(), _fim_choices() (+2 more)

### Community 23 - "GGUF Convert & Quantize"
Cohesion: 0.18
Nodes (9): convert_to_fp16(), _finalize(), is_fresh(), _meta_path(), quantize_fp16(), run_cmd(), source_fingerprint(), _tail() (+1 more)

### Community 24 - "Eval Worker Process"
Cohesion: 0.15
Nodes (8): run_eval_worker(), error_msg(), log_msg(), metric_msg(), AtomicSaveTrainer, commit_checkpoint(), test_log_and_error_msg(), test_tick_drains_metric_and_log()

### Community 25 - "Val Metric UI Components"
Cohesion: 0.14
Nodes (10): val_metric_msg(), test_val_metric_msg_shape(), test_build_metric_plot_empty(), test_build_metric_plot_has_legend_and_no_pyplot_state(), test_build_metric_plot_two_axes(), test_build_metric_plot_val_points(), test_verdict_style_colors(), build_metric_plot() (+2 more)

### Community 26 - "LoRA Export Workers"
Cohesion: 0.14
Nodes (8): save_adapter_only(), test_save_adapter_only_config_without_weights_raises(), test_save_adapter_only_copies(), test_save_adapter_only_copies_shards(), test_save_adapter_only_from_interrupted_swap_artifact(), test_save_adapter_only_no_checkpoint_raises(), test_save_adapter_only_rejects_output_outside_sandbox(), on_save_adapter()

### Community 27 - "Optimization Roadmap & Papers"
Cohesion: 0.12
Nodes (18): AWQ Paper (arXiv 2306.00978), BitNet b1.58 Paper (arXiv 2402.17764), Central Benchmark Schema, Compression Architecture (6.1), GPTQ Paper (arXiv 2210.17323), LLM Optimization Lab Roadmap (Phases 6-11), Phase 10 — Unified Optimization Benchmark, Phase 11 — Inference / Deployment (+10 more)

### Community 28 - "FIM Helpers & Export"
Cohesion: 0.21
Nodes (3): encode_prompt(), ensure_fim_tokens(), predict_middle()

### Community 29 - "Training Callbacks & Messages"
Cohesion: 0.21
Nodes (11): validate_message(), _cb_with_states(), test_nan_trip_sends_error_and_stops(), test_non_metric_logs_forwarded(), test_on_log_emits_metric(), test_on_log_eval_loss_sends_val_metric(), test_on_log_lr_never_seen_sends_warning_not_metric(), test_on_log_returns_control_explicitly() (+3 more)

### Community 30 - "Eval Test Factories"
Cohesion: 0.12
Nodes (8): _eval_factory(), process_factory(), FakeQueue, process_factory(), test_run_eval_dead_child_raises(), test_run_eval_error_raises_and_cleans_up(), test_run_eval_happy_path(), test_run_eval_rejects_while_training()

### Community 31 - "Process Test Fakes"
Cohesion: 0.12
Nodes (6): FakeProcess, test_abort_escalation_failure_keeps_lock(), test_abort_kills_and_marks_terminal(), test_exit_terminates_child(), test_training_active_locked_while_aborting(), test_watchdog_silent_while_aborting()

### Community 32 - "Serve CLI Script"
Cohesion: 0.21
Nodes (7): _log_tail(), build_parser(), _ensure_sigint(), main(), _resolve_gguf(), serve(), _sigterm_interrupt()

### Community 33 - "llama-runner HTTP Core"
Cohesion: 0.13
Nodes (4): find_free_port(), _http_get(), _http_post(), test_find_free_port_is_unused()

### Community 34 - "Dataset Loader Tests"
Cohesion: 0.15
Nodes (10): iter_codes(), test_iter_codes_and_list_support_nested_parquet_layout(), test_iter_codes_hub_missing_column_gives_english_error(), test_iter_codes_limits_and_column(), test_iter_codes_local_folder_without_parquet_gives_english_error(), test_iter_codes_local_missing_column_gives_english_error(), test_iter_codes_outside_existing_path_never_hits_hub(), test_iter_codes_reads_local_parquet_folder_sorted_multifile() (+2 more)

### Community 35 - "Shared Test Helpers"
Cohesion: 0.16
Nodes (3): FakeQueue, test_run_eval_worker_error_sends_error_and_raises(), test_run_eval_worker_sends_progress_and_done()

### Community 36 - "Phase 5 Integration Tests"
Cohesion: 0.15
Nodes (5): _data_cache_dir(), test_abort_mid_training_no_orphan_no_vram_leak(), test_datacache_growth_bounded(), test_disk_gate_blocks_preflight(), _valid_config()

### Community 37 - "Train Log Streaming"
Cohesion: 0.15
Nodes (6): status_msg(), StreamToQueueCallback, test_start_resets_previous_run_state(), test_zombie_tick_resets_status_and_unlocks_start(), test_zombie_watchdog_after_drain(), test_zombie_watchdog_failure_persists_in_later_ticks()

### Community 38 - "Merge Export & Config Collect"
Cohesion: 0.18
Nodes (9): merge_export(), test_collect_config_tolerates_cleared_number_fields(), _collect_config(), _make_handlers(), on_check(), on_eval(), on_merge(), on_predict() (+1 more)

### Community 39 - "Pipeline Smoke Script"
Cohesion: 0.32
Nodes (7): _base_config(), _collect(), _free_vram_gb(), main(), mode_abort(), mode_full(), _spawn()

### Community 40 - "Training Sample Builder"
Cohesion: 0.21
Nodes (11): build_eval_texts(), build_samples(), _codes(), test_build_eval_texts_empty_when_no_heldout(), test_build_eval_texts_heldout_only_and_limited(), test_eos_at_end(), test_fim_rate_half(), test_heldout_stable_and_ratio() (+3 more)

### Community 42 - "Historical Phase Plans"
Cohesion: 0.18
Nodes (7): Estimator + Dataset Builder Modules, Phase 2 Estimator & Dataset Implementation Plan, Subprocess Training + IPC Bridge Design, Phase 3 Subprocess Training Pipeline & IPC Plan, Audit Backlog P0+P1+ML Implementation Plan, FT-001..FT-010 Fine-tuning Task Group, 6 Work Safety Dimensions

### Community 43 - "Dataset Builder Tests"
Cohesion: 0.21
Nodes (9): _expected_fim_parts(), _qwen_tokenizer(), test_build_samples_prefix_keeps_tail_next_to_middle(), test_fim_budget_keeps_middle_or_drops_sample(), test_fim_dropped_when_middle_exceeds_budget(), test_is_heldout_declares_usedforsecurity_false(), test_plain_lm_truncate_unchanged(), test_split_fim_error_message_is_english() (+1 more)

### Community 44 - "Runner Unit Tests"
Cohesion: 0.18
Nodes (5): kv_cache_mb(), _no_tool_gate(), test_kv_cache_mb_parses_fixture_and_garbage(), test_server_lifecycle_real(), test_startup_kv_line_survives_eval_log_volume()

### Community 45 - "Server Stop & Log Reader"
Cohesion: 0.23
Nodes (6): ServerHandle, stop_server(), test_stop_server_joins_log_reader(), test_stop_server_kills_after_timeout(), test_stop_server_noop_when_already_dead(), test_stop_server_sends_sigint_then_wait()

### Community 46 - "Token Truncation Tests"
Cohesion: 0.23
Nodes (7): truncate_to_tokens(), test_truncate_to_tokens_fits_returns_original(), test_truncate_to_tokens_head_default_unchanged(), test_truncate_to_tokens_rejects_bad_keep(), test_truncate_to_tokens_tail_keeps_end(), test_truncate_to_tokens_zero_budget_is_empty(), _WordTok

### Community 47 - "Dependency Pins & Guides"
Cohesion: 0.18
Nodes (11): Pinned Tested Dependency Set, torch==2.14.1+xpu Pin, Phase 4 Gradio UI Implementation Plan, Gradio UI Integration Tasks, Phase 4 Gradio UI Design Spec, English UI Labels & Error Handling, GGUF Compression & llama.cpp Serving Guide, Intel XPU Installation Guide (+3 more)

### Community 48 - "Local Parquet Reader"
Cohesion: 0.18
Nodes (5): _read_local_parquet(), _resolve_local_dataset_dir(), test_read_local_parquet_rejects_non_text_values(), test_resolve_local_dataset_allows_repo_internal_path(), test_resolve_local_dataset_rejects_path_outside_roots()

### Community 49 - "Serve Process Fakes"
Cohesion: 0.18
Nodes (3): FakeProc, test_complete_passes_prompt_and_params_verbatim(), test_server_log_is_bounded()

### Community 51 - "Predict Worker Fakes"
Cohesion: 0.24
Nodes (7): _predict_factory(), process_factory(), test_run_predict_dead_child_raises_fast(), test_run_predict_error_raises(), test_run_predict_success_returns_text(), test_run_predict_timeout_raises_and_kills_child(), run_predict()

### Community 52 - "Phase A Contracts Spec"
Cohesion: 0.29
Nodes (8): Benchmark Report Schema v1.0, core/contracts Package (stdlib-only, unidirectional), LocalRuntime ABC (core/contracts/runtime.py), Spec: Phase A — Freeze Contracts (Sprint 1 / PR-01), RunManifest Contract (core/contracts/manifest.py), Artifact Specification Matrix (9 artifacts), Benchmark Report JSON Artifact, Run Manifest JSON Artifact

### Community 53 - "Abort & Preflight Tests"
Cohesion: 0.20
Nodes (6): test_abort_shows_aborting_while_killing(), test_preflight_output_dir_not_created_yet(), test_start_rejects_seq_over_cap(), test_start_spawns_run_training(), test_status_message_cannot_overwrite_aborting(), valid_config()

### Community 55 - "Leakage Filter Tests"
Cohesion: 0.28
Nodes (5): filter_train_codes(), is_heldout(), test_filter_train_codes_drops_reindented_twin_of_heldout(), test_filter_train_codes_removes_all_heldout(), test_is_heldout_pins_md5_split_membership()

### Community 56 - "Checkpoint Recovery Tests"
Cohesion: 0.25
Nodes (5): _has_adapter_files(), is_checkpoint_dir(), _recover_interrupted_commit(), test_is_checkpoint_dir_only_matching_names(), test_recover_interrupted_commit_ignores_partial_saving()

### Community 57 - "Phase 5 Eval Docs Plan"
Cohesion: 0.25
Nodes (7): Test-Once E2E Testing Strategy, Phase 5 End-to-End, Evaluation & Docs Plan, Base vs Fine-tuned Evaluation Workflow, Phase 5 Evaluation & Docs Design Spec, Guardrail Mapping Section (§4), Evaluation CLI (base vs fine-tuned compare), Held-out Evaluation Results (500 steps, n=100)

### Community 58 - "Approach B Reorg Docs"
Cohesion: 0.28
Nodes (7): Approach B Domain Restructure, Reorg (Approach B) + Clean + Hermeticity Plan, trainer_worker.py Split into 7 Modules, Approach B Target Structure (§5), Audit Backlog Execution + Domain Restructure Design Spec, Audit Wave Inventory & Sequence, Target 3-Part Directory Layout Mapping

### Community 59 - "Migration Phases A-G"
Cohesion: 0.22
Nodes (9): Migration Phases A-G, Migration Phase A — Freeze Contracts, Migration Phase B — Fine-tuning Isolation, Migration Phase C — Benchmark Extraction, Migration Phase D — Runtime Abstraction (LocalRuntime), Migration Phase E — llama-cpp-python Runtime, Migration Phase F — Remove llama-server Binary, Migration Phase G — Cleanup & Docs (+1 more)

### Community 60 - "Audit Wave & Supply Chain"
Cohesion: 0.25
Nodes (3): Audit Backlog P2 + Wave 1 Plan (Tasks 1-17), H1 Sandbox for save_adapter_only (Security #10), requirements.txt Pinned Dependencies

### Community 61 - "FIM Registry Tests"
Cohesion: 0.43
Nodes (5): _load(), test_deepseek_values_match_spec(), test_qwen_values_match_spec(), test_registry_loads_with_three_families(), test_starcoder_values_match_spec()

### Community 63 - "3-Part Architecture Plan"
Cohesion: 0.43
Nodes (7): Benchmark Subsystem (BM), BM-001..BM-012 Benchmark Task Group, Fine-tuning Subsystem (FT), RM-001..RM-012 Run Local Model Task Group, Run Local Model Subsystem (RM), System Readiness Matrix, 3-Part Subsystem Architecture

### Community 64 - "NaN Guard Tests"
Cohesion: 0.29
Nodes (3): NanGuard, test_nan_guard_aborts_on_three_consecutive(), test_nan_guard_resets_on_finite()

### Community 65 - "Near-dup Leakage Guard"
Cohesion: 0.33
Nodes (3): find_near_dup_leakage(), _line_shingles(), test_near_dup_threshold_boundary()

### Community 66 - "Phase 6a Compression Docs"
Cohesion: 0.33
Nodes (6): GGUF Compression Pipeline Tasks, Phase 6a GGUF Compression Pipeline Plan, llama.cpp Vulkan Build Recipe, Phase 6a GGUF Compression Design Spec, llama.cpp Toolchain (convert/quantize/bench/server), Quantization Variants (FP16 / Q8_0 / Q4_K_M)

### Community 70 - "Resume Checkpoint Tests"
Cohesion: 0.40
Nodes (3): resume_checkpoint(), test_resume_checkpoint_is_pure_scan(), test_resume_checkpoint_returns_latest_or_none()

## Knowledge Gaps
- **25 isolated node(s):** `setup_llamacpp.sh script`, `Migration Phase B — Fine-tuning Isolation`, `Migration Phase C — Benchmark Extraction`, `Migration Phase E — llama-cpp-python Runtime`, `Migration Phase F — Remove llama-server Binary` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 570 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrainingController` connect `Training Controller Core` to `VRAM Estimator & Model Resolve`, `Broken Queue Tests`, `Phase 5 Integration Tests`, `Safe Defaults & Estimator`, `UI Controller Tests`, `Gradio App Entry`, `Eval Test Factories`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `_make_handlers()` (e.g. with `on_abort()` and `on_check()`) actually correct?**
  _`_make_handlers()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `setup_llamacpp.sh script`, `Migration Phase B — Fine-tuning Isolation`, `Migration Phase C — Benchmark Extraction` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Evaluation Metrics & Cases` be split into smaller, more focused modules?**
  _Cohesion score 0.05711849957374254 - nodes in this community are weakly interconnected._
- **Why does `run_training()` connect `Training Worker Pipeline` to `Dataset Loader Tests`, `Train Log Streaming`, `Resume Checkpoint Tests`, `Pipeline Smoke Script`, `Training Sample Builder`, `Safe Defaults & Estimator`, `Training Args Factory`, `UI Controller Tests`, `Training Controller Core`, `Leakage Filter Tests`, `Eval Worker Process`, `Checkpoint Recovery Tests`, `FIM Helpers & Export`, `FIM Registry Tests`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `start_server()` (e.g. with `_drain()` and `main()`) actually correct?**
  _`start_server()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Benchmark CLI Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05888376856118792 - nodes in this community are weakly interconnected._