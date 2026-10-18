# What is still missing to beat SFT (completeness critic, items not covered by C1–C19)

The ranking is by how much each item affects whether a RL-from-I win would be real and would survive confirmation. All paths are under `/Users/idide/RyanLinEnv` unless stated, and branches are on origin. INF marks my inference. Nothing was edited and no model was run; I only read committed files with `git show` and `python3 -I`.

## M1. Nobody has measured the real bar: SFT that uses the same extra labels
**Why**
- I was trained on train400, which is 4 images per class.
- RL-from-I trains on s_train (1,424 rows) or all_train (2,329 rows), about 3.6–5.8× more labelled images (`docs/aircraft-exploratory-sweep-v2/splits.json` `sizes`).
- So the "strongest fair SFT" for any such recipe is a tuned SFT on those labels, not I at 0.7539.
- The campaign goal compares RL only against "the paired control (+0.03) and I (+0.03)". It calls sftall-I "a global reference; not the paired control" (`rl-campaign-design.noindex:docs/aircraft-rl-campaign-I/arms.json` `.goal`, `.facts_relied_on.competing_sft_in_flight`).
- sftall-I runs at a single LR, 5e-6, and has no result yet (local `runs/aircraft-exploratory-sweep-v2/state.json`: RUNNING on iccad9).
- The GPU lane has no SFT-control result at all. Only evals and GRPO-W summaries are committed on `exploratory-sweep-v2-gpu.noindex`.
- So the size of the target is unmeasured. INF: it is very likely well above 0.7539, and the ≥0.80 absolute gate may be met by SFT alone.

**Action**
- Before any RL fan-out, run SFT-from-I learning curves on s_train and on all_train at LR {5e-6, 1e-5, 2e-5}.
- Assay checkpoints on dev800 at the RL time stops.
- In the plan, set the comparator as max(I, best tuned SFT on the same labels).

## M2. Wrong starting point for the scientific question: RL is never tested from the best SFT
**Why**
- From I, RL competes with SFT for labels SFT has not yet used, and SFT extracts more per label.
- Measured: RL400 trailed SFT at the same LR and the same exposure by −1.27 pp (`reports/2026-10-17-controlled400-native-val/README.md`).
- Argued: a binary reward gives about zero gradient on images the policy already answers correctly, while cross-entropy keeps pushing p(gold) up.
- The sweep plan already has the clean design, "the best sftall endpoint … seeds GRPO" (`docs/aircraft-exploratory-sweep-v2.md` lines 120–125). The 22-arm campaign has no such arm.

**Action**
- Add an arm pair: RL from sftall\* against continued SFT from sftall\*, time-matched.
- If sftall\* beats I on dev800, make this the primary family. It is the only design where "RL adds something SFT cannot" is not confounded by unused labels.

## M3. Every recipe uses 1 image per update, constant LR, and steps the optimizer even at zero gradient; none of this was ever varied
**Why**
- The trainer config (`sweep-wave2-integration.noindex:src/vision_rl/vlm_local_training.py` lines 966–996) sets:
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=num_generations`
  - `generation_batch_size=num_generations`
  - `lr_scheduler_type="constant"`, `warmup_steps=0`
- So each optimizer step is one image's group.
- The `GRPO_KNOBS` allowlist (lines 46–56) cannot express more prompts per step.
- Measured: steps with zero gradient still take an optimizer step. GPU arm 03 (β0) logs `optimizer_steps_observed 2`, `latest_gradient_norm 0.0` (`optimizer_audit.jsonl` steps 2–3).
- INF, not measured: AdamW momentum therefore moves the parameters on the 53–66% of steps with zero advantage, which is the B4 phantom-step mechanism.
- The audit logs only the first nonzero update's delta, so the per-step parameter delta on zero-gradient steps has never been recorded.
- Single-image updates are also why every useful step is clipped (pre-clip norm above 1.0 on mixed steps; C3).

**Action**
- Add a family with P images per update, for example 8 images × G8 via `generation_batch_size=P·G`, and expose it through the allowlist.
- Skip the optimizer step when a batch has no advantage at all.
- Log the per-step parameter L2 delta.

## M4. dev800 has never been calibrated against native VAL
**Why**
- dev800 against VAL:
  - I: 0.775 on dev800 against 0.7539 on VAL (+2.1 pp).
  - W: 0.685 (bf16) or 0.69 (fp32) against 0.6784 (+0.7 to +1.2 pp).
  - Sources: GPU `raw/*/eval-*/summary.json` and the native-val README.
- Whether dev800 differences rank arms the way VAL does is unknown.
- Seven endpoints already have native VAL scores: RL400 and six SFT controls.

**Action**
- Assay those 7 endpoints on dev800 on the campaign's device. That is about 7 × 315 s on a 3070, or about 7 × 68 min on CPU.
- Require rank agreement, and check that dev800 reproduces the −1.27 pp RL-versus-same-LR-SFT sign.

## M5. No A/A placebo runs through the promotion rule
**Why**
- The noise at 150 steps is already about 0.75 pp. The same config gives 0.70375 (3070 bf16 β0), 0.69875 (3070 β0.04) and 0.69625 (2080 Ti fp32).
- Checkpoints are non-monotone: the lr5e-5 arm went 0.69625 → 0.6825 → 0.71625 (GPU summaries).
- Rung A promotes the top 6 of 22 by (RL best − control best) over 2 stops each (`arms.json` `.rungs.A`).
- C11 asks for noise arms, but nothing measures what this max-of-22 rule promotes when the true effect is zero.

**Action**
- Run 2–3 placebo pairs (SFT-from-I against SFT-from-I at another seed) through the identical best-of-k and promotion code.
- Set the promotion threshold above the placebo maximum.

## M6. The device changes mid-trajectory, and the final evaluation device is unspecified
**Why**
- Rung A runs on CPU fp32, but rungs B and C "should move to rtx3070-bfloat16" (`arms.json` `.device_classes.note_rtx3070`).
- bf16 changes predictions: W scores 0.685 in bf16 against 0.69 in fp32 on dev800. The audit's C2 cites `same_prediction_fraction 0.9575`.
- a\* was scored with the frozen evaluator 85f0865 on MPS fp32.

**Action**
- Keep one device class per trajectory from start to end.
- In the confirmatory plan, re-score every contender (RL\*, its control, I, sftall\*) on native VAL with the frozen MPS fp32 evaluator.

## M7. No confirmatory pre-registration exists for RL-from-I, and the gates are undefined for init I
**Why**
- D27 withdrew the only confirmatory plan. D31 says VAL stays untouched "until a recipe is pre-registered" (origin/main `docs/decision-log-2026-10-16-claude.md`).
- `substantial_rl` (`docs/aircraft-rl-protocol-freshpool-v1.json`) requires:
  - +0.05 over the common initialization (with I as init, that means ≥0.8039);
  - +0.05 over the strongest SFT;
  - a relative error reduction of ≥0.20;
  - replicates at seeds 42/314/2718 against the "corresponding strongest SFT".
- I and W exist only at seed 42, so "corresponding SFT at seed 314" is undefined.
- The campaign's +0.03 dev800 success criterion matches none of these gates.

**Action**
- Write the pre-registration now, before any dev800 winner exists. It needs to fix:
  - the comparator pool: I, every SFT-from-I control LR, and tuned sftall\*;
  - replicate semantics: RL and its SFT controls at seeds 314 and 2718 from the same I, or retraining I at those seeds;
  - the VAL-look ledger (VAL already selected I);
  - the TEST role allowlist;
  - the evaluator identity;
  - the disclosure of the 477 near-duplicates;
  - an explicit relabel of the setting from 4-shot to about 23 images per class.

## M8. The "SFT overfits few-shot" premise is refuted here; train400 has plateaued, and all_train contains train400
**Why**
- Equal time beat equal exposure at all three LRs: 0.7109 → 0.7539, 0.7399 → 0.7534, 0.7301 → 0.7520 (native-val README).
- Equal time is about 6.2 epochs: checkpoint 2464 / 400 (`addendum-5.json`).
- All three converge to 0.752–0.754. That means a data plateau, not overfitting. There is no evidence that RL "generalizes where SFT memorizes".
- all_train = native TRAIN − H-select − H-gate − dev600 − partners = 3,334 − 1,005 = 2,329 rows. It therefore includes train400, which I has fit for about 6 epochs (`splits.json` `.rules.all_train`).

**Action**
- Drop the overfitting rationale from every plan.
- Exclude train400 from all_train RL pools started from I, or count those rows as known all-correct groups.

## M9. The bf16-versus-fp32 choice can be settled by measuring instead of applying C2's rule
**Why**
- C2 refuses bf16 GRPO on the parity number 0.02814 > 0.02. That costs 2.7× throughput: 5.85 s/update (3070 bf16) against 15.7 s/update (2080 Ti fp32), per `arms.json` facts.
- The only fp32 GPU is a shared 11 GB 2080 Ti. fp32 weights take 8.84 GB, so the 8 GB 3070 cannot run fp32.
- With `num_iterations=1` the ratio is 1, so sampler–teacher mismatch only adds off-policy bias. That point is argued and its relevance is unmeasured.
- The one matched pair (lr1e-5, G8, β0, 150 steps) scored 0.70375 on bf16 and 0.69625 on fp32, a 0.75 pp gap inside the noise.

**Action**
- Run a 2-seed bf16-versus-fp32 replicate of one GRPO-I config.
- Keep bf16 only if its outcome difference is within the placebo spread from M5. Otherwise apply C2's refusal.

## M10. The resolution and label-ambiguity ceiling has never been checked
**Why**
- The dominant errors are sibling variants: Gulfstream IV/V, E-195/E-190, 737-600/500, A318/A319 (controlled400 diagnosis).
- The input resolution is fixed: `max_pixels` 200704 (about 448²) after removing the bottom 20 px.
- No label-noise audit exists, so it is unknown whether the ≥0.80 absolute gate is resolution-bound for both arms.

**Action**
- Do a quick audit of about 50 dev800 images where I is confidently wrong (p_gold < 0.01) in the top confusion pairs. Use a human or the 27B model.
- Report the fraction that is ambiguous at the trained resolution.

## M11. Compute ordering
**Why**
- The campaign estimates about 16 lane-hours per pair at rung A (`arms.json` `.compute_cost_lane_hours_rung_A_per_pair`, unverified). Over 22 arms that is about 352 CPU lane-hours (INF arithmetic).
- M1, M4, M5 and P0-I (C1) each cost a few GPU-hours and decide whether any arm can clear the bar.

**Action**
- Gate rung A on four measurements: P0-I, the SFT-from-I curve, the dev800 calibration and the placebo spread.
- Write the go/no-go thresholds into the plan beforehand.

## M12. No full-VAL named baseline is finished
**Why**
- `named_frontier_comparison` needs a full-VAL named baseline.
- The 27B MLX run 2 was paused permanently. Its replacement is the v2 plan on UW CPU hosts (D31). The Qwen2.5-VL-3B full VAL was dispatched with N = 159 (D34).
- Discovery scores are 34/100 and 25/100, so the gate will be easy to pass, but it still needs a number.

**Action**
- Make one completed full-VAL named baseline a prerequisite of the confirmatory look.

## Resolved
Early-aircraft lesson 24 (grpo against dr_grpo on Docker CPU) has no outcome to look up. The CPU and Linux routes never produced a training step (controlled400-17).