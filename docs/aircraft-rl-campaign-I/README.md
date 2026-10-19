# RL campaign from I: launcher

Goal: an RL recipe that starts FROM I (sft-1e5-time, adapter `8f27d0ca...`) and beats I continued with SFT
for the same measured compute. Exploratory. Every number comes from dev800, a TRAIN-derived holdout. Native
VAL and the sealed TEST are never read.

Driver: `scripts/rl_campaign/campaign.py`. It runs on the Mac and only does light, stdlib-only work. All
heavy work runs on UW hosts through uwrun.

It reuses existing pieces rather than duplicating them:
- the GPU lane's queue and worker (`gpu_lane.push`/`fetch`/`continue_spec`, `gpu_worker.sh`). The worker
  now also has a CPU free-core wait, slots, and a per-arm RSS trace.
- the sweep runner's state lock, receipts, host table, control template and Binomial.
- `payload.py` + `bootstrap_host.sh` + `host_verify.py`. Their root is `r226lin-rlcampaign`.
- `arm_exec.py` on each host.

## Device classes
| class | hosts | notes |
|---|---|---|
| cpu-fp32 | AVX-512F CPU hosts only (AVX2-only hosts die in MKL with SIGILL) | `--threads 16`; the worker waits for `--need-cores` free physical cores |
| gtx1070-fp32 | ecetesla3, eceubuntu4 | torch cu126 wheels: cu130 has no sm_61 kernels (measured) |
| gtx1070-fp16 | same | SFT and eval only. GRPO is refused outside fp32 (C2) |

The GPU lane owns ecetesla2/4 and eceubuntu2. The campaign refuses to use them.

## Usage
```
P=runs/aircraft-rl-campaign-I/payload-<head>
python3 -I scripts/exploratory_sweep/payload.py --out $P       # from a clean, committed tree
campaign.py bootstrap --payload $P --host H [--host H ...]
campaign.py receipts
campaign.py worker --host H [--slot N] [--mem GB] [--need-cores N]   # --mem = measured peak RSS + 20%
campaign.py enqueue --table T [--ids A,B] [--host H]
campaign.py fetch                 # status, s/update, peak RSS, GPU memory, co-tenancy, E0 result
campaign.py controls              # time-matched SFT-from-I control per finished RL / placebo arm
campaign.py report                # per checkpoint: vs I and vs control; best-of-k per group
campaign.py promote --table T --rung R
```

## Arm table
Format: `{"study_id": "aircraft-rl-campaign-I", "rung_stops": {"1": [...], "2": [...]}, "control_lrs":
[5e-6, 1e-5, 2e-5], "defaults": {...}, "arms": [...]}`.

Per-row fields:
- `id`, `device_class`, `mode`, `lr`, `stops`, `max_steps_total` (the last rung's last stop), `group`
- `track`: `rl`, `placebo`, `smoke`, `eval` or `e0`
- optional: `host`, `extra_trainer_args`, `pool_scoring_seconds`
- placebo rows only: `placebo_control_seed`

## Root corrections enforced
The rules come from `runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md`.
- **C2:** GRPO runs only in fp32. A GRPO stage whose `sampler_teacher_logprob_parity.within_tolerance` is
  not True refuses (`arm_exec.check_parity`).
- **M6:** each trajectory keeps one device class and one host. Continuations and controls run where the
  RL arm ran.
- **C4/C10 controls.**
  - A group has k RL LRs. They pair by rank with k control LRs, which default to {5e-6, 1e-5, 2e-5}.
  - Each control trains on every row the RL arm read, in scan order:
    - screened GRPO: every scored candidate;
    - dynamic sampling: every drawn group;
    - otherwise: the committed prompts.
  - A control is time-matched at EVERY RL checkpoint.
  - `pool_scoring_seconds` is added to the RL side.
- **C5:** `--dynamic-sampling-max-draws` must be at least 8. Dropped draws are generated inside the update
  span, so their time already counts in the RL arm's active seconds.
- **C10/M5/C11 promotion.**
  - Each group is scored best-of-k against best-of-k, with the same k on both sides.
  - The bar is the maximum placebo delta: SFT-from-I against SFT-from-I at another seed, scored by the
    same code. Promotion refuses if no placebo result exists.
  - Promotion keeps the top third of groups, and only those strictly above the bar.
  - The rung schedule is fixed by `rung_stops`, with no early stop.
- **C16:** co-tenancy is recorded per stage. GPU co-tenants come from nvidia-smi; a CPU sample counts as shared when load1 exceeds the stage's threads + 1; load1 is also
  recorded. A time comparison refuses when the RL arm and its control ran under different co-tenancy.
- **C19:** a continuation is reported only after its device class's E0 has passed. E0 is a `kind: e0`
  row that runs 4 updates uninterrupted twice and 2+2 resumed. It passes when |B−A| ≤ 2·|A2−A| + 1e-7.
- **M3:** multi-image updates (`--prompts-per-update`) are a trainer option, passed through
  `extra_trainer_args`.
- **Trainer extras (PR #38 review):**
  - `extra_trainer_args` is an allowlist: `arm_exec.EXTRA_FLAGS`, each flag with a value validator (the same design as the CPU sweep's 4a4dc5f).
  - Refused: any flag arm_exec sets itself, `--train`/`--validation`, `--flag=value` forms, argparse abbreviations, and repeats. The argv that actually runs is checked again for repeated flags.
  - `check_parity` reads the dtype the trainer recorded (`control_identity.profile.dtype`) and refuses anything but float32 for GRPO.
- **Controls and promotion (same review):**
  - An RL arm's control refuses if there is no `scanned/` file; it never falls back to the committed prompts.
  - The report compares co-tenancy cumulatively: every train stage up to the RL checkpoint against every control stage up to its time match, because active time accumulates across stages. A missing stage record refuses. It also refuses a pair where the control saw fewer rows than the RL arm read.
  - Extra values must be finite numbers, and `--label-hierarchy` must stay inside payload/code. GRPO and RFT stages refuse a `local_training.json` that is older than the stage, carries another stage_id, or is not float32.
  - Promotion's placebo bar uses only placebo groups from the same rung with the same k.
- **Stage identity (review 3):**
  - arm_exec chooses `--stage-id <work dir>-<stage label>` for every non-RFT stage. Both `local_training.json` and `endpoint_receipt.json` must record exactly that id, and the record must be newer than the stage start.
  - RFT's CLI refuses `--stage-id`. Its fresh record must instead carry this stage's `--stop-after-steps` boundary (none for the final stage), and must be float32.
  - `@payload/code/...` values are confined to payload/code after resolving symlinks.
  - C16 needs every stage of the trajectory up to a checkpoint (`trajectory_stops`, which promote extends), so a continuation whose earlier rung was not fetched refuses.
- **Merge of main (#36) and review 4:**
  - The extras allowlist is main's `EXTRA_FLAGS`, which has the tighter value ranges, plus `--prompts-per-update` (1..32) and `--skip-zero-advantage-steps`. File flags are checked twice: as raw `@payload/code/...` text, and as the resolved file after `payload_arg`.
  - Scanned rows come from main's `Arm.scanned_ids` (RFT rounds, and dynamic-sampling blocks) plus screened-GRPO scored candidates.
  - An RFT stage reads its stop from `--rft-stop-after-steps`. An early end by rounds, max_steps or active seconds finishes as `total_budget_exhausted`. A reused completed RFT endpoint must still be float32.
  - `endpoint_receipt.json` must be newer than the stage start.
