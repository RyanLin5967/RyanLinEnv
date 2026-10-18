# Signal-limit Stage 0 smoke on a UW host (not a study stage, not a look)

**What ran.** `scripts/freshpool_launch/signal_limit_production/smoke_stage0.sh` at commit
`46d10374c601ea790f5e069d32205b2515aaa07b` (`ecetesla1/ARCHIVED_HEAD`), through
`~/.claude/bin/uwrun run --host ecetesla1 --threads 16 --timeout 5400` (run
`slsmoke-ecetesla1-1`, rc 0). The real Stage 0 entry (`stage0_gate.py`) through the real
`phase_worker.py --remote-cpu`, every trainer part in its own guarded process, CPU FP32, the
host's provisioned env (`reports/2026-10-18-signal-limit-uw-cost/provision/ecetesla1.json`).

**Scope.** Smoke parameters, recorded by the entry as `study_parameters: false`: G8 only,
E0 over 2 steps (2 uninterrupted vs 1+1 vs 1+1 resumed), a 1-accepted-update cost probe.
The launcher's `stage0_validator` refuses such a fragment (it needs `e0_groups [8, 16]`,
`e0_steps 4`, `probe_accepted >= 8`), so this can never stand in for the study's Stage 0.
Data: 64 native-TRAIN rows of train400 (every 6th row), init W (`f05840be...`); no H split,
no native VAL, no sealed TEST image was read.

**Result** (`ecetesla1/fragment.json`, `ecetesla1/smoke.json`):

| Field | Value |
|---|---|
| phase worker exit code | 0 |
| wall | 3,361 s (host AMD Ryzen 9 5900X, 16 threads; shared host, label the number with it) |
| checks at G8 (13) | all true: resume_2_plus_2_identical, resume_1_plus_3_identical, finite_nonzero_task_gradient, no_reference_forward_at_beta_0, loss_type_dr_grpo_recorded, scan_follows_train_manifest, initial_adapter_is_w, useful_mixed_group, lora_parameter_changed, scorer_uses_sampler_transform, active_time_ledger_cumulative, scorer_seconds_within_active_time, group_size_is_g |
| sampler-teacher max abs log-prob difference | 4.53e-06 (bound 0.001) |
| c(G8), 1 accepted update | 247.68 s per accepted update, scorer included |

The 2-step resume is byte-identical on CPU FP32 (adapter bytes, screen journal, reward groups
with completion token IDs, exposure), the beta-0 path leaves no reference trace, and the
`--remote-cpu` child flag reached every trainer part. `ecetesla1/guards/` holds each process's
receipt (peak RSS, wall).
