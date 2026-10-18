# Signal-limit on UW hosts: FP32 unit costs and the device choice

Every number below names its receipt (this directory) and its host. These are shared hosts:
timings include other users' load at the time and are not dedicated-hardware numbers.
Data: native-TRAIN rows only (train400 subsets); no H split, no native VAL, no sealed TEST.

## Measurements

**RTX 2080 Ti, eceubuntu2, CUDA FP32** (`eceubuntu2-2080ti/run.log`, run
`fp32cost-gpu-2080ti-1`; another user's process held 222 MiB of the GPU):

| Unit | Result |
|---|---|
| SFT, per update (updates 2-6) | 1.54-1.62 s; peak torch allocation 9,689.8 MiB |
| screened dr_grpo beta 0, G8, per accepted update | 18.02-20.12 s; peak allocation 10,035.0 MiB of 10.57 GiB |
| same at G16 | **CUDA out of memory** on the first update (9.80 GiB allocated, 312 MiB free) |

**Ryzen 9 5900X, ecetesla4, CPU FP32, 16 threads** (`ecetesla4-ryzen5900x/units.jsonl`, run
`slunits-ecetesla4-1`, `scripts/freshpool_launch/signal_limit_production/measure_units.sh` at
`ARCHIVED_HEAD`; per-update numbers are committed active seconds from the trainer's
`optimizer_audit.jsonl`, updates 2..N):

| Unit | median | max (used for forecasts) |
|---|---|---|
| A1 SFT update | 23.4 s | 23.5 s |
| A2 EG-SFT (no-std, B_8) update | 34.5 s | 38.0 s |
| A3 HN-SFT update | 109.4 s | 113.4 s |
| A4 HLS-SFT update | 570.4 s | 668.8 s |
| A5 constrained-CE update | 22.4 s | 30.9 s |
| screened dr_grpo beta 0 G8, per accepted update | 319.1 s | 338.5 s |
| same at G16 | 639.8 s | 673.7 s |
| p scorer, one candidate (8 measured) | 8.1 s | 16.9 s |
| greedy decode, one image ((wall 24 - wall 8) / 16) | 9.7 s | |
| one G4 sampled group, one image (same instrument) | 42.6 s | |
| per trainer process outside active time (wall - active) | 331-431 s | forecast 450 s |

**Xeon Gold 6136, iccad9, CPU FP32, 16 threads, under the exploratory sweep's load (46/48)**
(`iccad9-gold6136-contended/`): SFT 35 s, EG-SFT 28 s, HN-SFT 111 s, HLS-SFT 450 s (medians),
GRPO G8 275-631 s; the run hit its 7,200 s bound before G16 and inference. The sweep's own
idle-host bench (PR #29 `cpu-bench.json`) has SFT 11.3 s and G8 134 s on these hosts.

**Host provisioning** (`provision/*.json`): `provision_host.sh` installed the hashed lock plus
torch 2.14.1+cpu into `/var/tmp/r226lin-slenv-v1` and the pinned model into
`/var/tmp/r226lin-slmodel-v1`; the trainer's own preflight and model-directory check passed
on iccad9, iccad11, ecetesla1 and ecetesla4 (iccad9 and iccad11 were removed again: their
/var/tmp is what the sweep needs).

## Choice: CPU FP32 on the three Ryzen 9 5900X hosts (ecetesla1, ecetesla2, ecetesla4)

- The 2080 Ti is about 15x faster per SFT update and 17x per G8 update, but **cannot run the
  pre-registered study**: Stage 0 must prove E0 and measure c(G) at G16 (GB0 may choose G16)
  and G16 FP32 does not fit; G8 already uses 10.0 of 10.6 GiB on a GPU another user shares.
  The 3070s (8 GiB) cannot hold the FP32 weights. bf16/fp16 would change the study's FP32
  profile.
- Among CPUs, one hardware class for every stage keeps budget matching in active seconds
  (T_RL for SFT-T and LIM-T) and every H comparison on one device class. The Gold 61xx hosts
  are faster when idle but are full with the exploratory sweep (and measured 3x slower under
  its load); the three 5900X hosts are idle or lightly used and have 300-600 GB of /var/tmp.
- Guards against contention: a stage starts only on a host with 16 idle threads
  (`min_idle_threads`), at most two uwrun sessions of anyone on it, and every cap is on
  launcher wall time, so later contention can only stop the study (pre-registered), never
  buy extra compute unseen.

## What it implies (estimates from the units above, not measurements)

- **A4 (HLS-SFT) dominates Stage A**: 600 x 668.8 s = 111 h for one arm, and the launcher runs
  A4 (at A1*'s LR) before Gate A. The study's compute plan assumed A4 at the SFT rate
  (3.29 s/update on MPS); on this device HLS-SFT costs 28x an SFT update.
- Stage A to Gate A on three hosts: about 35 h for the 12 grid arms, about 13 h of H-select
  assays, then A4 about 111 h: about 7 days.
- Stage B at G8 (if Gate A passes): RL to 100 and to 300 about 35 h, SFT-T and LIM-T about
  30 h each at T_RL, assays and P0': about 4-5 days more. About twice that at G16.
