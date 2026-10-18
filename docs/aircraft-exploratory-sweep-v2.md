# Aircraft exploratory sweep v2: plan, written before any result

Study id `aircraft-exploratory-sweep-v2`. Machine-readable twin:
`docs/aircraft-exploratory-sweep-v2.json`. Generated design files (byte-reproducible with
`python3 -I scripts/exploratory_sweep/design.py --check`): `docs/aircraft-exploratory-sweep-v2/`
(`splits.json`, `arms-rung1.json`, `manifests/*.jsonl`).

## Status: EXPLORATORY

This sweep looks for an RL recipe that substantially beats the best fair SFT for Qwen2-VL-2B
(rank-8 Q/V LoRA) on FGVC-Aircraft 100-way. Every number it produces comes from holdouts carved
out of **native TRAIN**. It never reads, lists, copies or scores native VAL or the sealed TEST.
A recipe that looks good here is a *candidate* only. Confirmation on native VAL and sealed TEST
happens later, under a separate pre-registration that fixes the recipe before that run starts.
Nothing measured here is a held-out benchmark claim.

This file and its JSON twin were committed before any arm trained and before anything was
assayed. CPU seconds per update (timing only, no accuracy) are measured afterwards and recorded
separately in `docs/aircraft-exploratory-sweep-v2/cpu-bench.json`.

## Initial adapters

| name | source | weights sha256 | native VAL (known before this plan) |
|---|---|---|---|
| W | `runs/aircraft-warmstart-v1/final_adapter` | `f05840be…2c1c` | 0.6784 |
| I | `runs/aircraft-controlled400-v1/setup/sft-1e5-time/published/final_adapter` (controlled400 `sft-1e5-time`, the strongest SFT) | `8f27d0ca…f7` | 0.7539 |

These VAL numbers were already on record (RL400 got 0.6982). They are context only. Nothing
in this sweep recomputes them or selects by them.

## Data: holdouts and training pools (seeded, class-stratified)

The source is the freshpool G0 manifests (`runs/aircraft-freshpool-g0-v1/manifests`): S
(2,027 rows), H-select (200) and H-gate (200), all native TRAIN. Every file hash listed in their
`manifest.json` is re-checked. Native TRAIN is `data/fgvc-aircraft-2013b/development/train.jsonl`
(3,334 rows, sha256 `909ea0c9…3eec`).

- **dev600**: built from S. For each of the 100 classes, sort that class's S rows by
  `sha256(json[study_id, "dev600", 20261018, label, id])` and take the first 6. That gives
  6 × 100 = 600 rows. Every class has at least 10 S rows, so the split is exactly stratified.
- **dev800** (the selection holdout) = H-select 200 + dev600. Every arm is assayed on dev800 at
  every kept endpoint.
- **h_gate_final** = freshpool H-gate (200). Only rung-3 finalists touch it, and only once
  (see the decision rule). Nothing is selected after it is read.
- **Near-duplicate guard**: the G0 pHash audit lists within-pool pairs (Hamming ≤ 6). Any
  image paired with a holdout image never trains. That removes 5 images: 0062131, 0894311,
  1114210, 1237633, 1239225.
- **s_train** = S − dev600 − partners (1,424 rows). This is the RL and SFT pool for the main
  matrix.
- **all_train** = native TRAIN − H-select − H-gate − dev600 − partners (2,329 rows). This is
  the data-scale pool. It keeps 477 pool images that the G0 audit flagged as near-duplicates
  of native VAL/TEST images. They are legitimate native TRAIN, and native training would use
  them. The later confirmatory study must account for them.
- **Order**: `random.Random(seed).shuffle(sorted(ids))` for seeds 42, 314 and 2718. This gives
  `s_train_scan_seed*.jsonl` and `all_train_scan_seed*.jsonl`. The seed-42 order serves rung 1.

Every run trains and is assayed under the task data spec that W and I were trained under,
`candidates/fourshot-seed791/data_spec.json` (sha256 `afc69edd…477a`). It has the same 100
labels, question and preprocessing as `development/data_spec.json`. The trainer's assay of a
saved adapter refuses any other spec.

The trainer receives only TRAIN manifests. `arm_exec.py` refuses any training manifest named
`dev*` or `h_*`, and drops holdout ids from control sequences. `payload.py` ships an image only
if its id is native TRAIN, its source is `development/images/train/<id>.jpg`, and its bytes
match `audit/train-images.json`. This is an allowlist, so no VAL or TEST image can enter a
host payload.

## Metrics

On dev800, with greedy decoding through the trainer's own `predict_train_subset` (chunks of
32), the same canonical exact-EOS scoring as `scripts/freshpool_launch/h_assay.py`. A
completion counts only if it ended at EOS and `parse_label` maps it to one of the 100 labels;
otherwise it is invalid, and invalid counts as wrong.

- **Primary**: accuracy on dev800.
- **Secondary**: valid fraction (share of rows with a parsed in-vocabulary label), and
  equal-class accuracy (mean per-class accuracy over the 100 classes).
- **Comparison**: paired bootstrap of the accuracy difference against the matched SFT
  control. The 800 paired items are resampled jointly with 20,000 resamples, drawn exactly as
  a multinomial over (win, tie, loss) counts, seed 20261018. The report gives the point delta,
  the 95% percentile CI, P(delta ≤ 0) and wins/losses.

## The matrix

Wave 1 uses current trainer features only. All runs are CPU FP32, 16 torch threads, rank-8
Q/V LoRA (`legacy-qv-v1`), answer-only format, 50,176–200,704 pixels and 256 max completion
tokens, unless stated otherwise. Every run declares the rung-3 total (1,500 updates) as its
budget and stops at the rung boundary. Promoted runs therefore continue the same trajectory.
The trainer's schedule does not depend on where a run is stopped.

| family | init | arms (rung 1) | settings |
|---|---|---|---|
| sft-W | W | 4 | LR {5e-6, 1e-5, 3e-5, 1e-4} |
| sft-I (SFT continuation) | I | 3 | LR {1e-6, 5e-6, 1e-5} |
| egsft-W (addition) | W | 2 | entropy-gated SFT, LR {1e-5, 3e-5} |
| grpo-W | W | 30 | fractional design (below) |
| grpo-I | I | 6 | LR {1e-6, 5e-6, 1e-5} × G {4, 8}; β 0.04, T 1.0, no screening |
| think-grpo-W (addition) | W | 2 | think-then-answer format, LR {5e-6, 1e-5}, G 8 |
| sftall | W, W, I | 3 | all_train, two epochs (4,658 updates), endpoints at 1 and 2 epochs; W LR {1e-5, 3e-5}, I LR 5e-6 |
| noise (addition) | W | 4 | SFT-W 1e-5 and GRPO-W 1e-5 G4, seeds 314 and 2718: the seed-noise floor for rung 1 |
| baseline | W, I | 2 | assay only (no training) |
| control | per RL arm | 40 | see "Matched SFT control" |

**GRPO-from-W fractional design.** The full LR × G grid (5 × 3 = 15 cells) runs twice,
which gives 30 runs. In each cell the two runs take two different β values out of
{0, 0.04, 0.1}, opposite temperatures in {0.7, 1.0}, and opposite screening (off, or on at
threshold 0.5). The assignment is chosen among 4,000 seeded candidates (seed 20261018) with
β balanced 10/10/10. The kept assignment minimises the largest absolute correlation between
any two coded factor columns; that correlation is 0.0816. Every main effect is estimable
(each level appears at least 10 times), and so is the LR×G interaction (every cell is
replicated). Higher-order interactions are confounded by design.

**LoRA capacity.** Rung 2 compares rank-8 Q/V with the merged expanded-LoRA v2 profile
(`qwen2vl2b-language-seven-r8-v1`, approved INIT converted from W) on two units: the best
SFT unit from W and the best RL unit from W, 3 seeds each, at the rung-2 budget. The expanded
contract allows only GRPO with β 0.04 and SFT, from the approved INIT. These arms run only
on a host whose receipt lists the `expanded_lora` trainer feature. Until then the runner
refuses them, and does not substitute anything.

**Data scale.** The sftall arms above give the strongest-possible SFT on all native TRAIN
minus holdouts. After sftall finishes rung 1, the best sftall endpoint (by dev800, either
epoch) seeds GRPO: LR {1e-6, 5e-6, 1e-5}, G 8, β 0.04, T 1.0 on s_train. Each of these
arms has a matched SFT control from that same endpoint, at LR 5e-6. They run on the host
that holds the endpoint (`init: {"arm", "label"}`). **An RL recipe matters only if it also
beats the best sftall arm and SFT(all) → its matched control.**

**Wave 2** (gated). Each arm names the trainer feature it needs, and the runner launches it
only on a host whose bootstrap receipt lists that feature, found in the trainer's `--help`.
Some features come from PR #26 work by another workflow: Dr. GRPO / `scale_rewards none`,
trie-constrained decoding, confuser unlikelihood and constrained CE. The others are built in
parallel with this sweep: hierarchical partial-credit reward (manufacturer/family/variant
from TRAIN annotations, shipped as `data/train_hierarchy.json`), rejection-sampling FT /
expert iteration, closed-set prompt listing the 100 labels, think-then-answer with answer
reward, DAPO (clip-higher, token-level loss, dynamic sampling dropping all-same groups), and
CUDA bf16/fp16. Additions also wait on features: prompts per update 4, μ = 2 inner
iterations, cosine LR, auxiliary NLL in RL, invalid-answer penalty, SFT augmentation, and a
higher-resolution transfer (401,408 pixels). For each wave-2 feature, rung 1 runs GRPO-W G 8
at LR {5e-6, 1e-5}, plus RFT and augmented SFT at LR {1e-5, 3e-5}. All are in
`arms-rung1.json` with `wave: 2`. A wave-2 arm that reaches its rung after that rung's
promotion was computed is promoted if it scores at least as high as the lowest unit promoted
at that rung in its track.

## Matched SFT control (every RL arm)

Each control is SFT from the RL arm's own init, at LR 1e-5 from W, 5e-6 from I, and 5e-6
from a kept endpoint. It trains on exactly the sequence of prompt ids the RL arm committed
(read from that arm's `endpoint_receipt.json`), in order, one update per prompt, cycling.
It has two endpoints:

- **E (examples-matched)**: the same number of examples as the RL arm.
- **T (time-matched)**: the same run continued until its committed active seconds equal the
  RL arm's.

The control runs on the RL arm's host after the RL arm finishes, so its time comparison is
on the same hardware. Both endpoints are assayed on dev800, and the report gives a paired
bootstrap against each.

## Rungs and promotion (successive halving)

| rung | updates | seeds | who |
|---|---|---|---|
| 1 | 150 | 42 | every runnable arm |
| 2 | 600 (4×) | 42 (continues its rung-1 run in place), 314, 2718 | top third of each track |
| 3 | 1,500 (10×) | the same 3 seeds, each continuing its rung-2 run | top 3 of each track |

- **Tracks** are ranked separately: `sft` (sft-W, sft-I, egsft-W, wave-2 SFT) and `rl`
  (grpo-W, grpo-I, think-grpo-W, wave-2 RL and RFT). Controls, noise, baselines and sftall
  are not halved.
- **Unit score** is dev800 accuracy at the unit's last step endpoint, averaged over its seeds
  at rungs 2 and 3. Ties go to equal-class accuracy, then valid fraction.
- **Promotion** waits until every arm of the rung has finished. A failed arm ranks last and
  is never promoted. Rung 1 keeps ⌈n/3⌉ of each track; rung 2 keeps 3.
- The sftall arms also run seeds 314 and 2718 at their full budget once rung 1 is done.
- `runner.py promote` implements this rule and writes `runs/aircraft-exploratory-sweep-v2/arms-rung{2,3}.json`
  together with the kept list in `state.json`.

## Decision rule (exploratory)

A rung-3 RL unit is a **candidate** when all of the following hold on dev800:

1. Its mean accuracy over 3 seeds beats the best SFT unit of the sweep by at least
   **2.0 points**. The best SFT unit is the best of: the rung-3 SFT track, the sftall arms
   (3 seeds), and the SFT controls.
2. The paired bootstrap of the pooled-seed delta against its own time-matched control has a
   95% CI lower bound above 0.
3. The same comparison against its examples-matched control also has a CI lower bound above 0.
4. Its valid fraction is not lower than the compared SFT's by more than 1 point.

Then each candidate, and the best SFT unit, is assayed once on `h_gate_final`. Those numbers
are reported as they come out. Nothing is re-selected or re-tuned after reading them. The
candidate and its exact recipe go to the separate confirmatory pre-registration. If no unit
qualifies, the result is "no RL recipe found that beats fair SFT under this budget", and
that result is reported as it stands.

## Compute and execution

- **Where it runs.** UW hosts through `~/.claude/bin/uwrun`. The Mac runs only the light
  driver. Each arm uses 16 threads. The runner keeps 25% of every host's cores and memory
  free, counting about 16 GB resident per arm (the measured max RSS was about 13.6 GB).
- **Host environment.** `scripts/exploratory_sweep/bootstrap_host.sh` sets up each host,
  idempotently, under `/var/tmp/r226lin-rlesweep` (or `/dev/shm/...` where /var/tmp is
  small), never on the NFS home. It builds:
  - a uv venv with Python 3.12.13;
  - torch 2.14.1 and torchvision 0.29.1 from the CPU or CUDA index, and everything else
    from `requirements-vlm-local.lock` with `--require-hashes`;
  - Qwen2-VL-2B at the pinned revision, verified by the trainer's own snapshot verifier;
  - the payload (code, TRAIN-only data, W and I).

  Every file is sha256-checked against `PAYLOAD.json`, and an extra file refuses. Package
  versions must equal `.venv-detection` for every shared package, with one declared
  exception: fsspec 2026.7.0. `.venv-detection` has no trl or datasets, datasets 5.1.0
  requires fsspec ≤ 2026.7.0, and W and I were both trained under 2026.7.0. The resulting
  `RECEIPT.json` is copied to `~/rlesweep/receipts/HOST.json`.
- **Receipt checks.** `runner.py` refuses a host without a receipt, a receipt for another
  payload, or a receipt that does not list the arm's features.

## Blind spots, stated on purpose

- **Small holdout.** dev800 and the per-rung comparisons are small. At about 0.70 accuracy, a
  single 800-item accuracy has a standard error near 1.6 points. The noise arms measure the
  seed-to-seed spread at rung 1. Promotions at rung 1 are therefore coarse by design.
- **Shared timing.** The hosts are shared with other users. Time-matched controls run on the
  same host as their RL arm, but host load still adds noise to active seconds.
- **Wave-2 detection.** A wave-2 feature counts as available if its flag appears in the
  trainer's `--help`. A flag that exists but is broken is caught only when its rung-1 arm
  fails.
- **Selection bias.** The h_gate_final check guards against selection on dev800, but it has
  only 200 images.

## Measured CPU seconds per update (appended after measurement; not a result)

Instrument: `runner.py bench` (6 updates, init W, 16 threads, CPU fp32), per-update seconds
from the trainer's `optimizer_audit.jsonl` active seconds, median of updates 2–6. Shared UW
hosts, so each number is labelled with its host. Raw rows: `aircraft-exploratory-sweep-v2/cpu-bench.json`.

| kind | iccad13 (72 threads, Gold 6154) | iccad9 (48 threads, Gold 6136) |
|---|---|---|
| GRPO G4 | 69.5 s | 70.2 s |
| GRPO G8 | 134.0 s | 134.3 s |
| SFT | 11.3 s | 11.5 s |

At 16 threads the two host types measured the same. GRPO cost scales about linearly in G
(G8 / G4 ≈ 1.93 on both hosts).

## Amendment 2 (2026-10-18, before any accuracy was read): screened arms declare 600 updates

The first screened GRPO-W arm (`grpo-W-04…-s`, uwrun `rlesweep-grpo-W-04-lr1e-6-g16-b0_1-t1_0-s-326254823`,
iccad13) refused at start: a screened trainer run needs `max_steps == target_accepted` and at
least that many **unique** scan rows, and s_train has 1,424. So 1,500 accepted updates cannot
exist. The 15 screened GRPO-W arms, and their matched controls, now declare `max_steps_total`
600 (the rung-2 total). The trainer's LR schedule is constant with no warmup, so the declared
total changes no update before a stop. Consequences, fixed now: a screened unit cannot run
rung 3 (if promoted there, it is recorded as infeasible and not replaced); at rung 2, a
screened run whose scan yields fewer than 600 accepted prompts refuses and is recorded as
failed. `design_files_sha256` in the JSON twin is updated; the old hashes are kept in its
`amendments` list.

## Engineering fix (2026-10-18, plan-neutral): constant save_steps across stages

`arm_exec.py` passed `--save-steps` equal to the current stage's stop. The trainer stores
save_steps in its settings and refuses a resume whose settings differ, so every multi-stop arm
refused at its second stage (found by the GPU lane: 3 of 3 multi-stop GPU arms). It now passes
the gcd of the declared stops and any `resume_stop`, which is the same in every stage and still
saves at every boundary. Nothing about updates, data or schedules changes. The sftall arms
launched before the fix will refuse their epoch-2 stage; each is continued from its kept
checkpoint-2329 under the fixed payload, on the same host.

## Engineering fix 2 (2026-10-18, plan-neutral): save_steps across rungs, host table, continuations

- **save_steps across rungs.** The gcd of an arm's own stops and `resume_stop` drifts across
  rungs (150 → 600 → 1500 gives 150, 150, 300), so every rung-3 resume would have refused.
  A promoted arm now carries `save_steps`, the value its trajectory started with: 150 for seed
  42 and 600 for a seed entering at rung 2.
- **Host table.** uwrun v5 added `cores` and `ourThr` columns to `uwrun hosts`. The positional
  parser then read physical cores as free GB, with no error. The runner now reads columns by
  header and refuses when one is missing. Capacity counts physical cores, which is how uwrun v5
  places runs.
- **Payload rollout.** A host is re-bootstrapped with the pinned payload (`runner.py refresh`)
  only once none of our arms is LAUNCHED or RUNNING on it, because bootstrap rewrites
  `ROOT/payload`. Until then its matched controls wait (skipped, not launched) instead of
  aborting the whole launch pass.
- **sftall continuation.** `runner.py continue ARM --reason R` retires the FAILED record as
  `ARM#failN` and relaunches the remaining stops in place on the same host: same id and work
  dir, `resume_stop` set to the last completed stop, and the earlier endpoints carried into the
  new summary.
- **Memory guard (uwrun v5.4).** Each launch passes `--mem`: 1.2 × the larger of its kind's
  measured training peak and the assay peak. The training peaks come from 22 live arms
  (VmHWM per process, summed): SFT 12.97 GiB, GRPO G8 13.56, GRPO G16 15.22. The assay peak
  is still an extrapolation: it grew to 14.36 GiB over 6 of 25 chunks, giving an estimated
  20.2 GiB until the first full assay under this payload records its own peak. A kind with no measurement (egsft) gets no
  per-run cap. `poll` records `failure_reason`:
  - `oom-guard:host-ram` (infrastructure): the record is retired and the arm relaunches from
    scratch in a fresh work dir, at most twice. If it already finished a stop, it resumes via
    `continue` instead.
  - `oom-guard:mem-cap`: our sizing was wrong; the arm is never relaunched automatically.
  - Other reasons: `timedout`, `overquota`, `stopped`, `code`.
  `summary.json` now records `resources` (peak RSS) for every arm.

## Amendment 3 (2026-10-18, exploratory, before any VAL use): stop W-init, hold for GATE 0

Recorded before any VAL use. The only accuracies read so far are the init baselines (eval-W
and eval-I on dev800); no trained arm of this sweep had finished. The reasons come from the
RL lessons audit (`runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md`, C8 and C11):
GRPO-from-W loses to I by 5–9 pp on dev800, and the GATE 0 SFT runs are starved on our own
oversubscribed hosts (21–54 s per SFT update against the benched 11.3).

- The 22 running rung-1 arms finish. Their results are recorded as exploratory.
- New launches are an allowlist in `runner.py` (`LAUNCH_FAMILIES`). Only grpo-I arms and
  their matched controls may launch, and only once GATE 0 has passed
  (`state.json` `gate0.passed` is `true`). Every other family is skipped as held: W-init
  controls, grpo-W, sft-W, egsft-W, noise, sft-I and sftall-rl.
- `runner.py continue` is allowed only for sftall-I and sftall-W.
- No W-init unit is promoted to rung 2. `promote` leaves them out of the ranking and lists
  them in `held_w_init`.
- Freed cores stay idle for the GATE 0 SFT lane (iccad5/6/7/8/10/13, ecetesla0/1). While
  GATE 0 is open, `refresh` bootstraps only hosts that hold a continuable sftall arm.

## Amendment 4 (2026-10-18, before any wave-2 arm ran): the concrete wave-2 arms

The wave-2 rows of `arms-rung1.json` name a feature but no trainer arguments, so no runner
can launch them. Three of the features have now been built and merged (branch
`sweep-wave2-integration.noindex`): RFT / expert iteration, the format and DAPO-style GRPO
knobs, and an invalid-answer penalty (the hierarchical-reward mechanism with zero credits).
CUDA training has also been merged. The concrete arms are in a separate table,
`aircraft-exploratory-sweep-v2/arms-wave2-rung1.json`, which
`scripts/exploratory_sweep/wave2_arms.py` (`--check`) generates byte-reproducibly. The
pre-registered `arms-rung1.json` is unchanged, and its abstract `w2-*` rows are never
launched. The new ids start with `wave2-`, so they cannot collide with those rows.

**Root RL corrections (binding, 2026-10-18 ~22:30Z).** The table follows
`runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md` (root checkout; evidence in its
`audit.md` C2–C7, C12, C17 and `critic.md` M6, M9).

- **Init.** Every family starts from I, because GRPO-W loses to I by 5–9 pp on dev800.
- **β and LR.** GRPO uses β 0 and LR {1e-5, 2e-5, 5e-5}. Dr-GRPO also gets 3e-5. One
  3e-6 plain GRPO-I arm guards against forgetting.
- **Precision.** GRPO runs only in fp32, on CPU or an RTX 2080 Ti: bf16 failed
  sampler–teacher parity (0.028 against a 0.02 tolerance), and fp16 has no parity
  evidence. RFT has no importance ratio and is SFT on the policy's own samples, so it keeps
  bf16 on the RTX 3070. Each trajectory stays on one device class.
- **Dropped families.** Hierarchical partial credit is refuted: it gave contrast in 1 of
  30–50 groups. Both CoT formats are refuted: valid answers fell from 94.9% to 82.8%, and
  accuracy from 70.7% to 55.5%.
- **Dynamic sampling.** At least 8 draws.

| family | device | trainer arguments | LR |
|---|---|---|---|
| wave2-fmtpen | CPU fp32 | exact reward, `--invalid-reward -0.5` (hierarchy file with both credits 0) | {1e-5, 2e-5, 5e-5} |
| wave2-closedset | CPU fp32 | response format `closed_set_answer_only` | {1e-5, 2e-5, 5e-5} |
| wave2-drgrpo | CPU fp32 | `--loss-type dr_grpo --scale-rewards none` | {1e-5, 2e-5, 3e-5, 5e-5} |
| wave2-mu2 | CPU fp32 | `--num-iterations 2` | {1e-5, 2e-5, 5e-5} |
| wave2-dapo | RTX 2080 Ti fp32 | `--loss-type dapo --epsilon-high 0.28 --dynamic-sampling-max-draws 8` | {1e-5, 2e-5, 5e-5} |
| wave2-guard | CPU fp32 | plain GRPO-I | 3e-6 |
| wave2-rft | RTX 3070 bf16 | RFT: K 8 at T 1.0, 32 prompts per round, keep 1 correct, 1 epoch, 400 rounds, cap 1,500 updates | {1e-5, 2e-5, 5e-5} |

All arms use G 8 (RFT: K 8) and the plan's rung-1 budget: 150 updates, seed 42, declared
total 1,500. Each arm declares its device (`device`, `dtype`, `gpu`, `device_class`).
The table has 46 arms: 20 RL arms, 20 matched controls and 6 device-control evals.

- **Matched controls.** Each RL arm has a matched SFT-from-I control on its own host and
  device. Each family's controls use as many distinct LRs as its RL arms (C4). The family's
  k-th smallest RL LR gets the k-th control LR:
  - three LRs: {5e-6, 1e-5, 2e-5};
  - Dr-GRPO's four: {5e-6, 1e-5, 2e-5, 3e-5};
  - the guard's single 3e-6: 5e-6.

  A family's best-of-k is therefore compared with its controls' best-of-k at the same k.
  Each closed-set control trains and is assayed on the closed-set prompt, so the pair
  isolates the RL effect. Promotion keeps each pairing (`matched_control_lr`,
  `control_response_format`, `control_replays`).
- **Every scored row is replayed (C4).** Two families score more rows than they commit
  updates. A DAPO update draws up to 8 groups and keeps one; an RFT round scores all 32 of
  its prompts against gold and trains only on the correct samples. Their controls
  (`control_replays: "scanned"`) replay every scored row, in scan order. The rows are
  taken from the trainer's own logs:
  - DAPO: `dynamic_sampling.jsonl`, every draw. The last block per update counts, and its
    kept draw must equal the committed id.
  - RFT: `rounds[].prompt_ids` in `local_training.json`.

  `arm_exec` writes them to `scanned_ids.json`. A missing or mismatched file refuses the
  control and never falls back. The step caps of these controls are 12,000 (DAPO) and
  12,800 (RFT).
- **Device controls.** W and I are each assayed on every device class
  (`wave2-dev-eval-{W,I}-<class>`). A GPU arm is read against I measured under its own
  numerics, and each GPU eval against the CPU fp32 eval.
- **RFT stops.** An RFT rung stop is round-granular (`--rft-stop-after-steps`). The endpoint
  is published at the first round boundary at or past 150 updates and records its true
  update count. The trajectory does not depend on where it was stopped.
- **Launch gate.** The runner launches an arm only on a host whose receipt reports the
  arm's device kind and GPU model, with at most one GPU arm per host at a time.
  `host_verify.FEATURE_FLAGS` now uses the built features' real flags.
- **Gate 0 is enforced, not only stated.** Every wave-2 family (`WAVE2_FAMILIES`: the
  seven RL families and the device-control evals) joins Amendment 3's launch allowlist
  behind the same check. `runner.py launch` holds the family's arms, its controls and the
  device-control evals until `state.json` `gate0.passed` is exactly `true`. That needs all
  four Gate-0 measurements: P0-I headroom, the SFT-from-I curve, dev800 calibration and the
  placebo.
- **Promotion.** Wave-2 arms rank at rung 1 in the `rl` track together with wave 1, under
  the plan's late-entrant rule. That ranking mixes device classes, so it is read against
  the device-control offsets. A paired comparison (arm against its matched control, or
  against I) never crosses device classes.
- **Not in this table.** The multi-image-update family (M3: P images per optimizer step)
  is not in the wave-2 table. Its trainer feature is now built: the GRPO knobs
  `--prompts-per-update P` (1..32, default 1) and `--skip-zero-advantage-steps true`
  (beta 0 only), plus the arm keys `prompts_per_update` and `skip_zero_advantage_steps` in
  `arm_exec`. A host receipt lists `prompts_per_update` once its payload carries this
  trainer. Every GRPO `optimizer_audit.jsonl` row now logs `param_delta_l2`.
- **M3 budget accounting (decided 2026-10-18).** An update whose whole batch has zero
  advantage takes no optimizer step under the skip, so parameters, AdamW moments and the
  AdamW step count do not move. It still advances `global_step`. So it counts toward
  `max_steps`, `stops` and `updates` as P consumed images, not as an applied optimizer
  update. Applied updates = `optimizer_steps_observed - zero_advantage_skipped_steps`
  (audit). An arm's stop at N updates consumes N*P images. Its matched control replays all
  N*P committed prompt ids (`runner.control_spec`), so the control stays matched on images.
- **Known limits.** First, with `--num-iterations 1` DAPO's clip-higher term never binds,
  because every update is on-policy. The dapo family is therefore effectively token-level
  loss plus dynamic sampling. wave2-mu2 is the arm where clipping is active. Second, C5
  requires discarded generation seconds to be charged to RL time. This is verified for
  dynamic sampling: every draw runs inside the update's active-time span.
  `tests/test_grpo_multi_image.py::test_dynamic_sampling_dropped_draws_are_charged_to_committed_active_seconds`
  passes. It fails when the draws are excluded from the span (forced fire, uwrun
  `m3-tests-5477c6a`).
