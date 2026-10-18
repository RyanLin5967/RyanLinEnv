# Aircraft RL contingency design v1: what to run if freshpool fails G1, G2 or G3

**Status: design only. Nothing here is pre-registered or activated, and nothing here came from
running a model.** This file plans the next study for each way `aircraft-freshpool-screened-grpo-v1`
(`docs/aircraft-freshpool-study-v1.md`) can stop before VAL. Each contingency has to be
pre-registered as its own protocol revision before any output it depends on is seen. Every
probability here is a judgement and is marked **unverified**. Every cost is extrapolated from the
measured rates in §1, and the extrapolation step is marked **unverified**.

## 0. The bar and the structural problem

- **Bar.** The `substantial_rl` gates are unchanged: VAL ≥ 0.80, ≥ +0.05 over init, ≥ +0.05 over the
  strongest SFT, relative error reduction ≥ 0.20, a paired 95% LB > 0, and three-seed replication.
  The strongest SFT today is 0.7539 (`reports/2026-10-17-controlled400-native-val/README.md`, SFT
  1e-5 equal time), so RL needs **≥ 0.8039**. Any fresh-pool SFT endpoint also joins the pool, so
  the bar can only rise. At a\* = 0.79 it would be 0.84.
- **Why answer-only RL is capped (algebra, freshpool §0).** Take one label, a 0/1 exact-match
  reward and G samples. The expected GRPO update is then a per-image scalar w_G(p) times the gold
  SFT gradient (`expected_gradient_weight` in `src/vision_rl/vlm_local_training.py`). The algebra
  ignores KL and token-length normalization (that function's docstring says so), so this is an
  approximation, and it holds **in expectation only**. Under it, answer-only GRPO is reweighted SFT
  that wastes the compute spent on zero-advantage groups. Under an equal-time comparator the cost
  makes this worse: an unscreened RL update costs 20.29 s and a screened one 30.61 s, against
  SFT's 3.29 s (§1), so 6.2× and 9.3× as much. The screened ratio is the one that applies to
  freshpool and to S, L, G and C. The **Visual-RFT suitability** paper (`docs/rl-strategy-review-2026-10-16.md`
  row 1) reports the same pattern: on FGVC-Aircraft, RFT *without* thinking scores 63.57, below SFT's
  75.84, while RFT *with* thinking scores 83.32. That paper used Qwen2.5-VL-7B with 8 rollouts, and
  how tuned its SFT was is unverified here. Our RL400 at −5.57 pp against tuned SFT has the same
  sign as its no-thinking row, which is −12.27 pp (63.57 − 75.84). Only the sign agrees; the
  magnitudes differ by more than a factor of two.
- **Consequence for ranking.** A contingency that keeps answer-only exact-match RL can win only
  through the weighting, through variance or on-policy effects, or through the start point.
  Freshpool's EG-SFT arm is the direct test of the weighting. Only a change that puts unsupervised
  tokens in the output (reasoning) has a mechanism that SFT on gold labels cannot copy. That is why
  §3 ranks it first, even though it is the most expensive option and the least tested on this
  model.

## 1. Measured per-unit costs (instrument named; nothing re-measured here)

`runs/` is gitignored (`.gitignore`), so rows marked **(local)** cite files that exist only on the
study machine, and `runs/aircraft-freshpool-v1` is a live run whose files can still change. Their
values were read, read only, on that machine and snapshotted into the committed
`docs/aircraft-rl-contingency-v1-instruments.json`: each source path, the source file's SHA-256 at
snapshot time, and the fields copied verbatim (the four `update_seconds`, mean 30.61 s; the `e0`
and `W-h_select` `stage_finished` events, 2165.83 s and 1815.56 s; the mechanism32 profile counts).
An independent review re-read the live files and got the same values. Rows without the mark cite
committed files.

| Unit | Value | Instrument |
|---|---:|---|
| Unscreened G4 GRPO update, FP32 MPS | 20.29 s | `reports/2026-10-17-controlled400-native-val/plan.json` `training_receipts.rl-exposure.committed_active_seconds` = 8115.88 s, over 400 updates |
| **Screened** G4 accepted update, including scorer time for rejected candidates | **30.61 s** mean (29.0, 30.0, 32.1, 31.4; n = 4) | **(local)** live freshpool E0 fragment `runs/aircraft-freshpool-v1/stages/e0/instance-1/fragment.json` `measurements.update_seconds`, read only. The design forecast was 22.23 s, so this is 1.38× the forecast. n = 4 is small |
| SFT update at equal active time | 2.83–3.29 s (1e-5: 8116.34 s / 2464) | `reports/2026-10-17-controlled400-native-val/plan.json` `training_receipts` |
| Full native VAL endpoint (3,333 images) | 0.98 h wall | supervision journals in the same report: `supervision-envelope-2-journal.jsonl` plus `-3-` first-to-last event, 153.5 s + 28,100.6 s = 28,254 s / 8 endpoints |
| H-select assay (200 images, 5 completions each) | 0.504 h wall (1815.6 s) | **(local)** live freshpool `runs/aircraft-freshpool-v1/journal.jsonl` `W-h_select` `stage_finished`. The design forecast was 0.265 h, so this is 1.90× |
| H-gate assay (200 images, 9 completions each) | ≈ 0.91 h | **unverified**: the design's 0.477 h scaled by the measured 1.90× H-select overrun |
| E0 resume gate | 0.60 h wall (2165.8 s) | **(local)** live freshpool journal, `e0` `stage_finished` |
| Teacher-forced scorer | 0.97 s / candidate | **unverified** (design §10). It is included in the 30.61 s figure above |
| Answer-only G4 sampling over 32 images | ≈ 122 s (3.81 s / image) | **(local)** `runs/aircraft-mechanism32-warm-v1/gate-result.json` |
| `think_then_answer` G4 sampling over 32 images | ≈ 122 s; 1,838 tokens over 256 completions (≈ 7.2 / completion); 4 / 256 complete think blocks | same file. The model barely reasoned, so **the cost of real reasoning is unmeasured** |

**Freshpool's own forecast, re-scaled.** Re-scaled with the measured overruns (RL ×1.38, H-select
×1.90, H-gate ×1.90, and T_RL rising from 13,338 s to 18,366 s for SFT-T), the 31.74 h to G4
becomes about **42 h**. This is an **unverified** extrapolation. The same rates are used for every
cost below.

## 2. Which contingency each failure points to

The freshpool launcher journals most of what tells the sub-branches apart: per-candidate p, u and
the accept decision; per-arm KL, adapter L2 and pre-clip norms; and the H-gate scores of RL\* and
SFT-E\* (and SFT-T\* after G3a). It does **not** hold an EG-SFT endpoint at a G3 stop: the launcher
runs EG-SFT and its H-gate assay only after `self.gate("G3b")` passes
(`scripts/freshpool_launch/launch.py`, `sequence`), and a gate that does not pass stops the study
(same file, module docstring). So the G3 rows below key on the **G3 tie-break** (§2.1), which the
contingency runs itself. Each row's trigger must be frozen in the contingency's pre-registration
**before** that freshpool evidence is read for the purpose of choosing.

**One sequence for every RL-study row.** The rows below do not choose what runs first. On any stop
that leads to an RL study, the sequence is the same, and it matches §3's recommended order:

1. Run **R0** (§3.1).
2. If R0 passes and R's forecast at the measured k is within R's pre-registered GPU-hour ceiling,
   run **R**. After R, whatever its result, RL on FGVC-Aircraft stops and the result is published.
3. If R0 fails, or R's forecast exceeds the ceiling, run the row's **answer-only fallback**. If the
   row has none, or the fallback also fails, stop RL on FGVC-Aircraft and publish the negative
   result. On a G3a or G3b stop, step 3 first runs the **G3 tie-break** (§2.1), and only then
   matches the table. Step 2 never needs it, so it runs only when step 3 is reached.

The table only fixes the fallback in step 3, plus the two engineering rows, which are freshpool
fixes and not RL studies. **Rows are matched top to bottom and the first match wins**, so every stop
maps to exactly one row. The last row is the default.

| Freshpool stop | Sub-cause, read from the freshpool journals (G3 rows: from the §2.1 tie-break) | Answer-only fallback (step 3) |
|---|---|---|
| **G1**, valid < 0.90 or EOS < 0.95 | interface | not an RL study: engineering or prompt fix and a P0 rerun under an amendment. If the rerun stops again, its stop is re-matched against this table |
| **G1**, observed useful-mixed fraction off its p-prediction by more than ±0.10, or scorer > 2.0 s | scorer calibration or speed | not an RL study: engineering (the tokenisation-sum switch, or batching), then as the row above |
| **G2**, reference changed, or adapter unchanged | engineering defect in the arm | not an RL study: fix, and rerun the arm from scratch under an amendment |
| **G1**, a < 0.30, with ≥ 2/3 of rejections at p < 0.159 | too hard: the warm policy has little gold mass on fresh images | **S′** (§3.2, SFT first on half of S) |
| **G1**, a < 0.30, with ≥ 2/3 of rejections at p > 0.841 | too easy: the mid-p band is thin | **G** (§3.4, G8 widens the band to [0.083, 0.917]), with **C** as an add-on |
| **G1**, a < 0.30, with both tails each holding ≥ 1/4 of rejections | bimodal | **G** |
| **G2** (futility at 150) | median adapter L2 ≤ 2 × RL400's 0.207, KL ≤ 0.0035, and clip fraction ≥ 0.9: the step is too small | **L** (§3.3) |
| **G3a / G3b** (RL ≤ SFT + 0.02 on H-gate) | the G3 tie-break (§2.1) completed, and its EG-SFT H-gate score ≥ RL\*'s freshpool H-gate score − 0.01 | none. The RL gain is explained by reweighting, so S, L, G, C, H and M have the same ceiling |
| **G3a / G3b** | the G3 tie-break completed, and its EG-SFT H-gate score < RL\*'s H-gate score − 0.01 | **S** (§3.2) |
| G4 / G5 non-pass (outside this file's remit) | — | as the two G3 rows, reading freshpool's own EG-SFT endpoint (it ran, because G3b passed), plus a stated note on the H-gate-to-VAL optimism |
| **Otherwise (default)** | any stop not matched above. Examples: G1 predicted mean u among accepted < 0.60; G1 with a < 0.30 whose rejections fit none of the three tail rows; G2 on the invalid count or a rolling useful-mixed fraction < 0.6; G2 with the adapter moved beyond the small-step row; a G3 stop whose tie-break was refused, capped or failed its validator | none: steps 1 and 2 only, then stop |

The thresholds in this table (2/3, 1/4, 2 × L2, 0.9 clip fraction, EG-SFT margin 0.01) are
**unverified judgements**. They are the values to freeze, and a contingency protocol may replace
them only before any freshpool field they read is opened for choosing.

### 2.1 The G3 tie-break (runs only on a G3a / G3b stop that reaches step 3)

- **What.** The EG-SFT arm freshpool would have run after G3b, run exactly as freshpool defines it:
  the `egsft` stage template, at SFT-E\*'s LR, on RL\*'s accepted manifest, 600 updates
  (`FINAL_UPDATES`), validated by the same `train_validator(..., "EG-SFT", lr, 600)`, then one
  H-gate assay. RL\*, SFT-E\* and RL\*'s accepted manifest all exist at a G3a stop (the launcher
  records them before `self.gate("G3a")`), so the arm is fully defined at either G3 stop.
- **Holdout cost.** It adds **one H-gate look** (the EG-SFT endpoint). It goes in the cross-study
  look ledger (§4) and is disclosed with the verdict.
- **Compute.** 600 × 3.29 s = 0.55 h of SFT plus one 0.91 h H-gate assay ≈ **1.5 h** (**unverified**
  extrapolation from §1).
- **Code change.** `launch.py` can only reach the `egsft` stage after G3b passes. The tie-break needs
  a separate entry, a new `scripts/freshpool_launch/g3_tiebreak.py`, that loads the stopped run's
  state read only, refuses unless the last gate is a non-passing G3a or G3b and `rl_star` and
  `sft_e_star` are recorded, and writes its endpoint and assay under the contingency's own run
  directory, never under `runs/aircraft-freshpool-v1`. Tests: refusal on a passing or missing G3,
  and on missing selections.

## 3. Ranked contingency studies

The probabilities are p(passing every `substantial_rl` gate including replication). They are
**unverified judgements**, anchored on the design panel's 0.05–0.08 prior for freshpool itself (D11).
They are not independent, so do not add them.

| Rank | Study | Triggered by | Dev-to-G4 GPU-h (unverified) | p(substantial) (unverified) | Trainer code change? |
|---:|---|---|---:|---:|---|
| 1 | **R**: reasoning GRPO with a STaR-SFT control | first on every RL-study row (§2 step 1) | 109 (k = 3) to 275 (k = 8) | 0.05 (range 0.02–0.10) | **yes** |
| 2 | **S**: GRPO initialised from the strongest fresh-pool SFT, on the unscanned remainder | fallback: G3 with tie-break EG-SFT < RL\* − 0.01; G1-hard as S′ | 27.2, of which 1.5 is the G3 tie-break (S′: ≈ 31) | 0.03 | no trainer change; launcher, evaluator and preparer changes |
| 3 | **L**: higher LR and a symmetric clip relaxation | fallback: G2-small-step | 43 | 0.02 | **yes** (small; plus launcher and evaluator `GRID`) |
| 4 | **G**: screened G8 | fallback: G1-easy or bimodal | 61 | 0.015 | no (launcher scripts only) |
| 5 | **C**: confusable-pair curriculum | add-on to G or S | ≈ 42 | ≤ 0.02 | no (preparer only) |
| 6 | **M**: sample reuse (`num_iterations` 2) | add-on to L under equal time | ≈ 40 | ≤ 0.02 | **yes** |
| 7 | **H**: near-miss (hierarchical) reward shaping | not recommended | ≈ 42 | ≤ 0.01 | **yes** |

Every point estimate sits at or below freshpool's own prior of 0.05–0.08 (D11; the judges gave
0.08 / 0.06 / 0.05). R's stated range reaches 0.10, above that prior; its upper end is the one
place this file allows a contingency to beat freshpool's odds. The honest reading is that if
freshpool fails, the expected value of more answer-only RL on this task is low. **R** is the one bet
with a different mechanism, and its first step (R0, about 0.4–1.1 h) is cheap and decides on its
own whether R can run at all. R is also by far the most expensive option: 109–275 h to G4, 4.0–10×
S, because every comparator that is matched to RL's active time scales with k. **Recommended order:** the §2
sequence. Run R0 on every RL-study row. If R0 passes and R fits its ceiling, run R and then stop. If
not, run the row's answer-only fallback (S, S′, L or G, or none). If that also fails, stop RL on
FGVC-Aircraft and publish the negative result.

### 3.1 R: reasoning GRPO (think-then-answer) with a STaR-SFT control (rank 1)

- **Mechanism.** The policy writes a short rationale and then the label. The reward is the existing
  exact canonical-label reward (`make_reward` with `allow_reasoning=True`). The gradient now flows
  through rationale tokens that have no gold target, so the expected update is no longer a
  reweighting of the answer-only SFT gradient. This is the mechanism the suitability paper credits
  for its +7.5 pp RFT-over-SFT on FGVC-Aircraft, and for the −12 pp without thinking.
- **Known obstacle (measured).** At W, the reasoning profile failed validity: 212/256 valid, 142/256
  correct, below the 0.90 gate (`docs/local-vlm-rl-design.md`, `runs/aircraft-mechanism32-warm-v1/`).
  The `think_then_answer` prompt does instruct the model to explain inside `<think>` tags
  (`prompt_text`); only the parse grammar makes the think block optional. Even so, 4 of 256
  completions had a complete think block, and completions averaged about 7 tokens, so the 2B model
  hardly reasoned (`gate-result.json`, snapshotted in the instruments file). R may have no mechanism to exercise at this scale. R0 settles that before anything
  expensive runs.
- **Design.**
  - **R0 (gate, no update).** A new reasoning prompt (`reasoning_v3`, an explicit think-first
    instruction that names visible features: engines, winglets, fuselage length, windows), run at W
    on 64 train400 images with G4 at seeds 42 and 314. Pass needs: valid ≥ 0.90, EOS ≥ 0.95, a
    complete think block in ≥ 0.80 of completions, a median rationale of ≥ 20 tokens,
    recognition-mixed groups ≥ 0.30, and a measured seconds-per-token figure that fixes k (cost
    relative to the answer-only G4 update). A failure stops R. Prompt iterations count toward a
    pre-registered maximum of 2.
  - **RL arm.** GRPO G4 with `response_format=reasoning_v3` and `max_completion_length` 512 (inside
    the profile's bound). Unscreened, because the teacher-forced p is a marginal over rationales and
    cannot be computed. LR grid 1e-5 / 5e-5 / 1e-4 to 150 updates, top two continue to 600, RL\*
    chosen on H-select, as in freshpool. Data: S in freshpool's seed-42 scan order. Init: W.
  - **Comparators.**
    - The protocol strongest-SFT pool takes every controlled400 and freshpool SFT endpoint, plus
      answer-only SFT-E (three LRs, 600 updates, exposure-matched on RL\*'s images) and SFT-T
      (continued to T_RL = RL\*'s committed active seconds, for every LR within 2 pp), as freshpool
      defines them, on R's exact images.
    - **STaR-SFT** (robust pool): for each image R's RL arms were presented, sample one G4 group
      under W with the R prompt, keep the correct valid traces, and SFT on them. The RL arms are
      unscreened and read S in the same seed-42 scan order, one image per update (R's plan fixes one
      prompt per update, as RL400 did), so that set is
      the first 600 images of the scan order: RL\*'s presented images, and also the union over
      every RL arm. STaR-SFT therefore reads exactly the labels RL read, about 600, not all 2,027
      of S, and uses no external rationales. Its tuning budget mirrors RL's grid: three LRs to the
      active time of RL's 150-update checkpoint, and the H-select top two continued to T_RL. It
      controls for the rationale *format* without the RL objective.
    - Two verdicts are published, protocol and robust, as freshpool does with EG-SFT.
  - **Evaluation and the profile asymmetry.** RL\* and STaR-SFT are scored on native VAL with the
    reasoning profile: greedy, 512 tokens, think-aware strict parse. This needs a new evaluator
    activation, and its VAL cost scales by k. Every answer-only endpoint (W, SFT-E, SFT-T, the
    controlled400 and freshpool pool) keeps its existing answer-only score. Scoring an answer-only
    model under a think-first prompt it was never trained on would handicap the comparator, and
    re-scoring the whole pool would cost about 0.98·k h per endpoint. The comparison stays valid
    across profiles for three reasons. First, the metric is the same: exact canonical-label
    mean-class accuracy on the same 3,333 images, so the paired unit is the image, not the profile.
    Second, each model is scored in the output format it was trained to produce, which is the
    comparator's best case. Third, STaR-SFT gives a same-profile comparator. To measure the profile
    effect directly, **W is also scored under the reasoning profile**: one extra endpoint. The init
    gate (≥ +0.05 over init) uses the higher of W's two scores, so the asymmetry cannot lower the
    init bar.
- **Cost** (scratch arithmetic from §1; k unmeasured until R0, all **unverified**). Items marked ·k
  run under the reasoning profile, or are matched to RL's active time, and scale with k:

  | Item | k = 3 | k = 8 |
  |---|---:|---:|
  | R0 | 0.4 | 1.1 |
  | E0 ·k (0.6·k h, an upper bound: most of E0's 0.6 h is not update time) | 1.8 | 4.8 |
  | RL grid: 1,350 updates × 20.29·k s | 22.8 | 60.9 |
  | STaR trace generation: 600 images (RL\*'s presented set) × 3.81·k s | 1.9 | 5.1 |
  | STaR-SFT grid at RL's tuning time (1,350 × 20.29·k s) | 22.8 | 60.9 |
  | H-select ·k: RL 3 + 2, STaR-SFT 3 + 2 = 10 × 0.504·k h | 15.1 | 40.3 |
  | H-gate ·k: RL\*, STaR-SFT\* = 2 × 0.91·k h | 5.5 | 14.6 |
  | VAL ·k: RL\*, STaR-SFT\*, W-reasoning = 3 × 0.98·k h | 8.8 | 23.5 |
  | SFT-E ×3: 1,800 × 3.29 s, plus H-select ×3 | 3.2 | 3.2 |
  | SFT-T ×2 to T_RL (600 × 20.29·k s = 10.1 h or 27.1 h each, less SFT-E's 1,974 s), plus H-select ×2 | 20.2 | 54.0 |
  | H-gate SFT-E\*, SFT-T\*: 2 × 0.91 h | 1.8 | 1.8 |
  | VAL SFT-E ×3, SFT-T ×2: 5 × 0.98 h | 4.9 | 4.9 |
  | **Total to G4** | **109** | **275** |

  An earlier draft of this doc gave 54–136 h. That figure billed STaR-SFT at a flat 4.9 h,
  although it is matched to RL's active time, left out SFT-E and SFT-T, and sampled STaR traces on
  all 2,027 S images. SFT-T assumes two LRs within 2 pp of SFT-E\*; a third adds another SFT-T
  row (10.1 h or 27.1 h). Replicates re-run RL, STaR-SFT, SFT-E and SFT-T per seed, so each one adds roughly 70–75% of the total
  (**unverified**).
- **Pre-register.** The prompt text and its SHA. The R0 thresholds and the prompt-iteration cap. k
  computed by a stated formula from R0. A GPU-hour ceiling for R, stated before R0 runs. If R0's k
  puts the forecast in the table above over that ceiling, R does not launch (§2 step 3). The
  STaR trace minimum (code change 3). The two-verdict comparator pools. A new protocol id with
  `substantial_gain_gates` byte-identical. H-select / H-gate reuse, with the freshpool look count
  carried into a cross-study look ledger. The evaluator profile (512 tokens, think-aware strict
  parse requiring EOS). Caps at 1.25× forecast, as in freshpool. No length or format reward.
- **Probability: 0.05** (range 0.02–0.10), **unverified**. Points for: the literature mechanism, and
  our no-think result having the same sign as the paper's no-think row (−5.57 vs −12.27 pp). Points against: the 2B model, a frozen-vision
  rank-8 LoRA, failed reasoning validity at W, an SFT comparator that is tuned (the paper's was not
  shown to be), and a bar of ≥ 0.80 absolute.
- **Code changes (exact).**
  1. `src/vision_rl/vlm_local_training.py`:
     - `prompt_text`: add a `reasoning_v3` branch.
     - `Profile.validate`: allow `reasoning_v3`.
     - `parse_label` / `OUTPUT_GRAMMAR`: confirm a think block of 512 tokens or fewer is accepted
       under the existing optional-think grammar, or version the grammar.
  2. `training_records` / `_dataset` (same file): allow an SFT row to carry an explicit `completion`
     text (a STaR trace) instead of the gold label. It must refuse a trace whose parsed label is not
     the row's gold label.
  3. New `scripts/build_star_traces.py`. It chunks rows ≤ 32 per call through `sample_train_groups`
     (the same pattern as `scripts/freshpool_launch/h_assay.py`), keeps correct valid traces, and
     writes a hashed manifest. It **refuses and exits non-zero**, naming the shortfall, when the
     kept traces number fewer than 600 (RL\*'s update count, so STaR-SFT never trains on fewer
     examples than RL saw) or cover fewer than 300 distinct images. Both minimums are
     **unverified judgements** and are pre-registered. At W's measured 142/256 correct, 600
     images × 4 samples would give about 1,330 correct traces
     (**unverified** extrapolation). An empty manifest is never written.
  4. `scripts/freshpool_launch/h_assay.py` and `p0_screen.py`: pass the reasoning profile through.
     The G4 checks at `h_assay.py:132` and `p0_screen.py:164` stay as they are.
  5. A new native-VAL activation package cloned from `docs/aircraft-native-val-activation-v1/`,
     with the reasoning profile and 512 tokens.
  6. Tests:
     - a CPU tiny-model test that a STaR row's loss mask covers the trace and the label;
     - refusal of a trace whose parsed label does not match gold;
     - `build_star_traces.py` refusal on an empty manifest and on one just below each minimum,
       and acceptance exactly at the minimums;
     - parse fixtures for think-block outputs, with expected labels hand-written.

### 3.2 S: GRPO from the strongest fresh-pool SFT, on never-trained images (rank 2)

- **Mechanism.** After tuned SFT, most training images sit at p ≈ 1, and the remaining errors
  concentrate in confusable pairs: Gulfstream IV/V, E-190/E-195 and 737-500/600 (RL400 diagnosis,
  `runs/aircraft-controlled400-diagnosis-v1/report.md`). Screened GRPO from that point spends every
  update on mid-p images, and its sampled wrong answers are the specific confusions. This is the
  standard SFT-then-RL recipe. In expectation it is still reweighted SFT, so EG-SFT from the same
  init is the decisive control.
- **Design.**
  - **Init:** SFT\*, the H-select-best freshpool SFT-T, or SFT-E if SFT-T never ran.
  - **Data:** S minus the **union of every freshpool arm's `scanned_manifest`**: the three
    150-update arms, both 600-update continuers, and any voided instance. This is the **unscanned
    remainder** that no arm has scanned or trained on. RL\*'s scanned set alone is not enough: the
    screen scores candidates under each arm's current policy (`_ScreenedScan` in
    `vlm_local_training.py`), so the other continuer, and possibly a dropped 150-update arm, can
    have scanned past RL\*'s position. S = 2,027 after the G0 pHash audit drops (D17; the freshpool
    study doc's "about 2,519" is the pre-audit figure). With a = 0.5, RL\* alone scans about 1,200,
    so **at most** about 827 images remain, fewer if another arm scanned further (unverified until
    the scans finish).
  - **N = 300 accepted.** P0 is run at SFT\* on the first 400 remainder images. The study refuses
    unless remainder size × P0 acceptance ≥ 1.25 × 300 = 375. At 827 images that means acceptance
    ≥ 0.45. Below that, use **S′** instead.
  - **RL grid:** 3 LRs to 100 updates, top two to 300.
  - **Comparators:**
    - SFT\* continued, with no reset, over the same remainder in scan order to RL's active time, at
      every LR within 2 pp (this is the SFT-T machinery, `continue_from_checkpoint`);
    - EG-SFT from SFT\* (robust verdict);
    - the full existing strongest-SFT pool.
- **S′ (when freshpool stopped at G1-hard, so no SFT\* exists).** Split S by hash into S_a and S_b
  (half each). SFT on S_a at the three LRs, select on H-select, then run screened RL from it on S_b.
  The comparator is SFT on S_a ∪ S_b to equal time. That adds about 2.7 h of SFT and 1.5 h of
  H-select.
- **Cost** (**unverified** extrapolation from §1):
  - E0 0.6 h and P0 at the init 1.0 h;
  - RL 700 accepted × 30.61 s = 5.95 h;
  - 5 H-select assays 2.52 h;
  - 3 SFT continuations to T_RL = 9,183 s, 7.65 h;
  - EG-SFT 0.35 h;
  - 3 H-gate assays 2.72 h;
  - 5 VAL endpoints 4.9 h.

  Total **25.7 h**. When S follows a G3 stop, the §2.1 tie-break has already run, which brings the
  branch to **27.2 h**. S′ follows G1-hard, needs no tie-break, and adds the S_a SFT and H-select
  above: about 31 h.
- **Pre-register.**
  - The init identity, pinned by SHA before S is drafted.
  - The remainder manifest, derived from every arm's committed screen journal
    (`committed_screen_entries`) and `scanned_manifest`, as their union.
  - The minimum-remainder refusal, the H-gate threshold (+0.02, or +0.05 under B2 mechanism mode),
    and the gates unchanged.
  - The claim: "RL after SFT beats SFT continued on the same images". RL − init is reported, but the
    claim rests on RL − strongest SFT.
- **Probability: 0.03**, **unverified**. The algebra says the gain has to come from the weighting.
  The remainder is small, and SFT\* will already be near the bar.
- **Code changes.**
  - None in the trainer: `init_adapter` takes any adapter, the reference is the saved init, and
    `continue_from_checkpoint` covers the comparator.
  - New `scripts/prepare_aircraft_contingency_split.py`, which builds the remainder and S_a / S_b
    manifests with SHA256s, plus tests on hand-built fixtures.
  - A new launcher plan whose `w_endpoint` is SFT\*. `launch.py` already reads W from the plan (lines
    341 and 907), and `GRID` (line 74) is unchanged.
  - **Launcher and evaluator update counts.** `launch.py:76` hard-codes
    `RL_SCREEN_UPDATES, FINAL_UPDATES = 150, 600`, and `FINAL_UPDATES` drives the SFT-E template
    check (`sft_e_template_stop_after_steps_not_600`, line 398), the SFT-E and RL\* accepted-manifest
    runs (line 1209) and the EG-SFT validator (line 1246). `scripts/evaluate_freshpool_gates.py`
    hard-codes the same counts (`G2["updates"] = 150`, line 69; `FINAL_UPDATES = 600`, line 78),
    and the SFT-E / EG-SFT budget check refuses anything but 600. S's 100 / 300 needs both files to
    take the counts from the plan, with the launcher cross-checking them against the evaluator as it
    already does for `GRID` (line 335), plus tests that the freshpool defaults are byte-identical.

### 3.3 L: higher LR with a symmetric clip relaxation (rank 3)

- **Mechanism.** In RL400 every useful update was clipped (median pre-clip norm 3.75, mean scale
  0.27), KL never bound, and the adapter moved L2 0.207 against the warm SFT's 2.948. If G2 fails with
  the same signature, the step size may be the binding constraint, not the signal. **Caveat:** the
  diagnosis this cites (`runs/aircraft-controlled400-diagnosis-v1/report.md`, item 6) itself notes
  that Adam largely normalises gradient scale, so the clip's real effect on the step is
  **unverified**. Under Adam the LR is the lever that is known to set the step; the clip arms test
  whether the clip matters at all, and a null on them is the expected outcome.
- **Design.** RL at LR {2e-4, 5e-4} × max_grad_norm {1.0, 4.0}: four arms to 150, top two to 600.
  SFT-E and SFT-T get the **same** 2 × 2 grid, so the clip change is symmetric. EG-SFT runs as in
  freshpool.
- **Cost.**
  - RL 1,500 accepted × 30.61 s = 12.75 h;
  - SFT-E ×4, 4.2 h;
  - SFT-T ×2 to T_RL, plus assays, 12.1 h;
  - 6 H-select assays 3.0 h;
  - H-gate 1.8 h;
  - EG-SFT 1.6 h;
  - 8 VAL endpoints 7.8 h.

  Total **43 h** (**unverified**).
- **Pre-register.** The grid. A divergence stop: NaN, or KL > 0.5 for 20 consecutive updates
  (threshold **unverified**). That the higher SFT LRs also enter the strongest pool. The G2 signature
  (L2 and clip fraction) that triggered the study, frozen before the freshpool fragments are re-read.
- **Probability: 0.02**, **unverified**. A larger step fixes futility, not the reweighting ceiling,
  and SFT gets the same larger step.
- **Code changes (exact).**
  - `trainer_kwargs`: `max_grad_norm` becomes a parameter (it is hard-coded to 1.0 today).
  - `train_local`: takes it through and refuses values outside (0, 100]. It is recorded in
    `settings` only when it differs from 1.0, so existing run identities stay byte-identical (the
    file's "New options enter settings only when used" pattern).
  - `main`: adds `--max-grad-norm`.
  - `scripts/freshpool_launch/launch.py` `GRID` becomes plan-driven, and so does
    `scripts/evaluate_freshpool_gates.py` `GRID` (line 61): `launch.py:335` refuses
    (`grid_differs_from_evaluator`) unless the plan grid equals the evaluator's. The evaluator also
    has to key arms by (LR, max_grad_norm), not LR alone.
  - A test that the default identity is unchanged and that the clip value reaches `GRPOConfig` and
    `SFTConfig`.

### 3.4 G: screened G8 (rank 4)

- **Mechanism.** u_G(p) ≥ 0.5 holds on p ∈ [0.159, 0.841] at G4, [0.083, 0.917] at G8 and [0.042,
  0.958] at G16 (computed from `group_useful_probability`). That rescues acceptance when the warm p
  distribution is too extreme for G4. As G grows, w_G(p) flattens toward the SFT weighting. G
  therefore buys signal but moves RL *toward* SFT, not past it.
- **Design.** Freshpool unchanged except `num_generations` 8 and the screening threshold kept at
  u ≥ 0.5 under G8. The H assays and the evaluation stay G4 (the evaluation profile is separate from
  the training G).
- **Cost.** Assumes the update cost is linear in G (**unverified**; generation and the per-completion
  backward both scale with G), on top of the measured 30.61 s: about 58–59 s per accepted update
  (a = 0.5–0.3). RL 1,350 updates 21.7–22.2 h, SFT-T to the doubled T_RL 21.2–21.6 h, everything else
  as in freshpool. Total **61–62 h**.
- **Probability: 0.015**, **unverified**.
- **Code changes.** None in the trainer: `Profile.num_generations` accepts 2–16, `_ScreenedScan`
  takes `group_size`, and `egsft_weight` takes `group_size`. In the scripts:
  - `scripts/freshpool_launch/p0_screen.py:164` (`GROUP`, set at line 50) and `h_assay.py:132` hard-code G = 4 for
    sampling. They must take the training G separately from the assay G.
  - `scripts/evaluate_freshpool_gates.py:93` hard-codes G4 in the G1 prediction of useful-mixed
    groups and must take G.

### 3.5 C: confusable-pair curriculum (rank 5; add-on)

- **Mechanism.** Order the scan so that images of confusable variant pairs arrive together, after a
  first pass, so each sampled wrong answer is the sibling class. SFT replays the same order
  (`order="sequential"` exists), which keeps it symmetric. Ordering does not change the expected
  update.
- **Cost.** About 42 h as a freshpool rerun, **unverified**, or 0 extra as an add-on to G or S.
  **p ≤ 0.02**, **unverified**.
- **Code changes.**
  - `scripts/prepare_aircraft_freshpool.py`: an ordering mode keyed by a family map.
  - Commit the train-only taxonomy, which today exists only in gitignored
    `runs/aircraft-hierarchical-opportunity-v1/taxonomy-train-only.json`.
  - A fixture test of the order.

### 3.6 M: sample reuse, `num_iterations` = 2 (rank 6; add-on to L)

- **Mechanism.** Each generated group drives two gradient steps, so the PPO ratio clip (inert today)
  binds and generation cost is shared across steps. It changes only RL's cost per step under the
  equal-time comparator. It does not change the ceiling.
- **Cost.** About 40 h. **p ≤ 0.02**, **unverified**.
- **Code changes.**
  - `trainer_kwargs`: `num_iterations` is hard-coded to 1 and must become a parameter.
  - `ScreenedTrainer._get_train_sampler` (`vlm_local_training.py:2293`) refuses outright today:
    it raises `screening requires one prompt per generation and one iteration` whenever
    `num_iterations != 1`. That guard has to be redesigned, not just re-proved: the screen must
    decide acceptance once per generated group and the sampler must then replay the accepted index
    for the second iteration without re-scanning or re-scoring, with a forced-fire test that a
    second screen on the same group is refused.
  - `_sampler_parity` and the resume sampler must then be re-proved under reuse, because
    `_ScreenedScan` yields each accepted index G times, per step.
  - E0 must pass again.
  - The protocol's `clip_epsilon_status` text changes.

### 3.7 H: near-miss hierarchical reward (rank 7; not recommended)

- **Mechanism.** Partial credit for the right family or manufacturer would create contrast in
  all-wrong groups. **Measured against it.** Hierarchical credit added contrast in 1 of 50 groups,
  and 9 of 11 all-wrong groups had a single distinct valid prediction
  (`runs/aircraft-hierarchical-opportunity-v1/report.md`). The SFT analogue, hierarchical label
  smoothing, is just as available.
- **p ≤ 0.01**, **unverified**.
- **Code changes.**
  - A new reward in `make_reward`, whose docstring forbids synthetic variance, so it needs a
    separately named reward.
  - `gold_probability`, `group_useful_probability` and `expected_gradient_weight` all assume a 0/1
    reward. Screening and EG-SFT would be invalid, so the arm runs unscreened.
  - The taxonomy commit from §3.5.

## 4. Pre-registration items every contingency shares

- A new protocol id that revises `docs/aircraft-rl-protocol-freshpool-v1.json`. `substantial_gain_gates`
  and `strong_baseline` must be byte-identical, asserted by a test like
  `tests/test_aircraft_protocol_freshpool.py`.
- **Comparator pool.** Every controlled400 and freshpool SFT endpoint already evaluated on VAL, plus
  the new study's own SFT arms. The pool only grows.
- **Holdout reuse.** H-select and H-gate have already been looked at by freshpool. A contingency may
  reuse H-select for selection, because the risk is symmetric across families. Its H-gate looks are
  added to a cross-study ledger and disclosed, including the G3 tie-break's one look (§2.1). No
  fresh holdout exists: freshpool's scan consumes
  most of S. TEST stays sealed until a G4 / G5 pass.
- **The trigger.** The freshpool fields that select the branch, and their thresholds (§2), are frozen
  before those fields are read for that purpose.
- **Budget matching.** Exposure-matched and active-time-matched SFT, the 1.25× per-arm caps, and a
  capped or interrupted arm re-run from scratch.
- **Label access.** train400 ∪ the S images the study reads. Not a 4-shot claim. Gains from the
  extra labels are credited to data.
- **Order.** Raw journals and endpoints are committed before interpretation.

## 5. Not proposed

- **BF16.** Rejected: sampler-teacher error 0.072.
- **Expanded LoRA.** It would help SFT equally, and the RL400 diagnosis found no evidence that
  capacity binds. It is integrated (D19) but unactivated.
- **Hosted or paid compute.** Ruled out by repo policy.
- **GitHub CPU runners for fits.** D3 measured 525 s per GRPO update on 2 vCPU, and D7's Harbor
  local dogfood measured 752.76 s for one pretrained update.
- **Changing any gate, or the comparator definition, after freshpool output.**

## 6. Decision record text (proposed D22)

This section carries the decision so that it lives in the committed tree. The numbered entry in
`docs/decision-log-2026-10-16-claude.md` is appended by the root session, which owns that file's
numbering; this branch does not edit it.

**D22 — Contingency design if freshpool fails G1/G2/G3 (design only, unactivated)**
- `docs/aircraft-rl-contingency-v1.md`, with live-run instrument values snapshotted in
  `docs/aircraft-rl-contingency-v1-instruments.json`. Answer-only exact-match GRPO equals
  reweighted SFT in expectation (ignoring KL and length normalisation), so the answer-only
  contingencies (S, L, G, C, M, H) share the reweighting ceiling that EG-SFT tests. Only reasoning
  RL (R) has a mechanism that gold-label SFT cannot copy.
- Order on any RL-study stop: R0 first (reasoning validity and throughput gate at W). If R0 passes
  and R's forecast at the measured k fits R's pre-registered GPU-hour ceiling, run R, then stop.
  Otherwise run the §2 row's answer-only fallback; if there is none, or it fails, stop RL on
  FGVC-Aircraft and publish the negative result. §2 is matched top to bottom with an explicit
  default row, so no outcome is left to choose after reading the data.
- A G3a or G3b stop leaves no EG-SFT endpoint (the launcher runs EG-SFT only after G3b passes), so
  the contingency runs a G3 tie-break: freshpool's EG-SFT arm plus one H-gate look (≈ 1.5 h,
  counted in the cross-study ledger). The G3 rows key on it.
- R costs 109 h (k = 3) to 275 h (k = 8) to G4, 4–10× S (27.2 h with the tie-break). STaR-SFT is
  time-matched to RL's grid and samples traces only on RL\*'s 600 presented images, so it reads
  the same labels as R.
- Every probability is a judgement; every point estimate is at or below freshpool's 0.05–0.08 prior,
  and only R's range (to 0.10) exceeds it. Costs extrapolate from measured rates: screened update
  30.61 s (live E0, n = 4), H-select 1815.6 s, unscreened update 20.29 s, SFT 3.29 s, VAL 0.98 h.
- Each contingency needs its own protocol revision. `substantial_gain_gates` stay byte-identical,
  the comparator pool only grows, H-gate looks go in a cross-study ledger, and branch triggers are
  frozen before freshpool fields are read for choosing.
