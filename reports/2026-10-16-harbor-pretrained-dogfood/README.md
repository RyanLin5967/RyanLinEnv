# Pretrained Docker + Harbor dogfood: first passing end-to-end trial

Date: 2026-10-16. Scope: **integration only**. Native FGVC-Aircraft TRAIN images, not held-out,
not a benchmark, not an RL-vs-SFT result, no frontier comparison.

## What ran
A Harbor 0.24 trial on the local Colima VM `vision-harbor` (aarch64 Linux, 4 vCPU / 16 GiB). The
agent container got 2 CPUs and a 13 GiB cgroup with zero swap. It trained the pretrained
**Qwen2-VL-2B-Instruct** (FP32, CPU) from the common warm adapter (`f05840be…`, rank-8 Q/V LoRA, 112
tensors). A **separate, fresh verifier environment** then scored the submitted numeric adapter.
By design the verifier executes no submitted code, config or pickle; its report records `submitted_code_executed: false` (that code path was not independently audited here).

Plan `plan/plan.json` (sha256 `62fe9ff1…`) is the reviewed local variant of the frozen CPU7946
science (TRL 1.16 GRPO, G4, answer-only, LR 1e-5, beta 0.04, seed 42). The image was built from the
37 hash-verified frozen source files.

| Stage (guarded child) | Result | Wall s | Peak tree RSS |
|---|---|---|---|
| SFT resource probe (1 update) | completed | 1012 | 11,906,678,784 B |
| RL exposure: 1 GRPO update | completed, 530.31 s active | 1307 | 12,235,571,200 B |
| SFT exposure: 1 update | completed, 68.08 s active | 748 | 12,164,743,168 B |
| SFT time: to RL active time | completed; resumed from the exposure checkpoint (1 update, 68.08 s) and ran 6 more updates (477.29 s), cumulative 7 updates / 545.38 s ≥ 530.31 s | 963 | 12,188,053,504 B |
| Separate verifier | completed | 104 | 10,776,674,304 B |

All guards: cgroup memory.max 13 GiB, swap.max = swap.current = 0, RSS ≤ 12 GiB. Caveat: cgroup `memory.peak` reached the 13 GiB limit in every stage, including the verifier. That counts page cache; all stages exited 0 and no OOM kill occurred, but memory-event counters were not saved, so reclaim pressure is not ruled out.

**RL signal is real, but n=1.** The single GRPO group (image `0143341`, label 737-500) sampled
`[737-300, 737-300, 737-500, 737-500]`, giving rewards `[0, 0, 1, 1]`. The group was mixed, so the
gradient carried task advantage, not only KL. Its pre-clip norm was clipped to 1. 112 tensors
changed, ΔL2 0.01042 (Adam's first step moves each coordinate by about the LR, so ΔL2 ≈
1e-5·√1.09M). Final adapter `ee0e7d0f…`.

## Pass criteria (pre-registered in the design; all four PASS)
`evidence/dogfood-local-result.json`:
- **C1 pretrained model loaded:** the pinned pread loader opened both safetensors shards on CPU.
  Model source manifest `2c20d30e…`.
- **C2 real optimizer update:** global_step 1, a nonzero gradient step, 112 changed tensors, and an
  adapter hash different from the init.
- **C3 separate verifier scored the submission:** the verifier environment was separate, and its
  report's adapter hash equals the RL receipt's `ee0e7d0f…`.
- **C4 resource guards held:** 5 guards and 7,930 resource rows, all within limits.

The verifier recorded accuracy 3/3. **It is not a criterion and means nothing about learning:** all
7 fixture images are in the warm adapter's own train400 SFT set.

## Failed attempts (preserved, not rerun)
| Attempt | Plan sha | What happened |
|---|---|---|
| v1 | 277bb9b2 | SFT probe trained (87.3 s active) but the wrapper ran the trainer via `runpy` as `__main__`. Its `SystemExit(0)` escaped the scoped loader, so the receipt read *failed*. Fixed by `plan/runtime-systemexit.patch`. |
| v2 | a2cd71df | GRPO update committed (752.8 s active). The 1800 s stage wall, sized from a GitHub x86 projection, killed the post-train reload/audit at 1801 s. Fixed by `plan/runtime-stage-walls.patch` and `plan/prepare-task-agent-timeout.patch`, using measured timing (`plan/timing.json`). |
| v3 | 6f1d0d6c | GRPO update committed (634.0 s active), then the host disk guard ended the trial when the shared host disk fell to ~0.3 GiB free (attribution to other sessions comes from the decision log, D8, not from a saved measurement). |
| v4 | 62fe9ff1 | **PASS** (this report). The plan content equals v3's, with new single-use output paths. |

Also fixed during this work: a real race in the draft host output scan (FileNotFoundError during
checkpoint rotation, reproduced with a fixture, patched in `local_run.py`).

## What this does and does not show
It shows that the full Docker + Harbor path works with the real pretrained model on Linux CPU:
packaging, agent-side guarded RL and SFT controls, artifact publication, and independent numeric
verification. It does **not** show learning, an RL advantage, or any held-out or benchmark result.
Throughput is about 530–750 s per GRPO update on 2 container CPUs, so CPU Harbor suits
dogfooding and CI, not the study itself.

Reproduce: `plan/README.md` (steps 3–9). Decision log: `docs/decision-log-2026-10-16-claude.md` D4–D9. Independent claim check: `runs/harbor-dogfood-report-verification/verification.json` (local).
