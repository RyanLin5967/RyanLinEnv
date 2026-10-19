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

## Re-run under uwrun v5 (the package's frozen copy)

`smoke_stage0.sh` at commit `d0688e5ac569facbfa4a4aa9568d67f029df9906`
(`ecetesla4-uwrun-v5/ARCHIVED_HEAD`), launched with that commit's
`signal_limit_production/uwrun.frozen` (uwrun v5) on ecetesla4, 16 threads, `--timeout 14400`
(run `slsmoke-v5-ecetesla4-5`, rc 0, wall 5,123 s). Same scope and data as above.

| Field | Value (`ecetesla4-uwrun-v5/fragment.json`) |
|---|---|
| checks at G8 (13) | all true |
| sampler-teacher max abs log-prob difference | 4.53e-06 |
| c(G8), 1 accepted update | 443.83 s (ecetesla4 at load 15-20 of 24 threads during the run; 247.68 s on ecetesla1 earlier) |

A first v5 attempt on ecetesla1 (`slsmoke-v5-ecetesla1-4`) ended `DONE rc=124 timedout` at its
5,400 s bound with 2 of 4 trajectories done: the host ran at load 38-43 on 24 threads (another
user plus other sessions of this account). The same contention inside the study would stop an arm
at its 1.25x cap; `min_idle_threads` (16) refuses such a host only when a stage starts.

## AVX-512 host: ecetesla0 (Xeon Gold 5120), E0 byte identity FAILED

`smoke_stage0.sh` at `1f90f0dcd627c040c20a1702ee8151d0709abd53` (`ecetesla0-avx512/ARCHIVED_HEAD`),
frozen uwrun v5, ecetesla0, **8 threads** (the host had about 7 idle; it then ran at load 36-42
of 28 threads), host env provisioned with `MODEL_SRC` (`reports/.../provision/ecetesla0.json`).
Run `slsmoke-avx512-ecetesla0-7`, phase worker rc 0, wall 13,050 s.

| Field | Value (`ecetesla0-avx512/fragment.json`) |
|---|---|
| checks at G8 | 12 of 13 true; **`resume_1_plus_3_identical` false** |
| sampler-teacher max abs log-prob difference | 3.81e-06 |
| c(G8), 1 accepted update | 891.58 s |

With `--e0-steps 2`, both resumed trajectories are 1+1, the same procedure. Adapter bytes, reward
groups (with completion token IDs) and exposure match in all three runs. One run's screen journal
differs in a rejected candidate's p: 0.9964093471659141 vs 0.9964060268643068 (u 0.02837 vs 0.02839;
both rejected). Raw: `ecetesla0-avx512/*-checkpoint-2-screen.json`. So the no-grad p forward is not
reproducible across processes on this host at 8 threads under that load, while the same commit's
smoke on the Ryzen 5900X hosts was byte-identical twice (16 threads). The study's Stage 0
validator refuses a fragment with any check false, so Stage 0 would stop on such a host: the
guard works. **This supports the plan's single Ryzen device class and argues against moving any
stage to the AVX-512 hosts** without first finding the source (oneDNN/MKL thread-dependent
kernels are the unverified suspect) and proving E0 there at the study's parameters.
