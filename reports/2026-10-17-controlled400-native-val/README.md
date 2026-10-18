# Controlled400 on native FGVC-Aircraft VAL: RL generalizes slightly, tuned SFT wins

Date: 2026-10-17. **Held-out development result** on all 3,333 native VAL images, 100 variants.
TEST is still sealed. Raw artifacts in this folder (`comparison.json`, `execution.json`, `plan.json`,
supervision journals, `SHA256SUMS`) were committed before this interpretation (f23266f).

## Setup (pre-registered)
- Setup values in this section (LR, beta, active seconds) come from the controlled400 run records
  (`runs/aircraft-controlled400-v1`, local and ignored); this folder holds only the VAL artifacts.
- Qwen2-VL-2B, FP32 MPS, rank-8 Q/V LoRA. Every arm starts from the same warm adapter (`f05840be…`,
  SFT on train400, i.e. 4 images per class).
- **RL400:** TRL GRPO, G4, LR 1e-5, beta 0.04, one pass over the same 400 TRAIN images, 8,116 s active.
- **Six SFT controls**, LR 1e-5 / 5e-5 / 1e-4, each at **equal exposure** (400 updates) and at **equal
  committed active time** (same trajectory continued to 8,116–8,118 s).
- **Evaluator:** frozen at 85f0865, activation `c9557c70…` committed before any VAL inference (D14).
  Greedy, answer-only, strict canonical-label parse requiring EOS. Metric: equal-class mean of the
  row-normalised confusion diagonal. Paired bootstrap, 20,000 draws, seed 2718.
- **Execution:** 216 evaluator invocations (1 in envelope 2, 215 in envelope 3; envelope 1 refused
  with 0), zero unknown outcomes, zero generation errors in every arm. Envelope history: decision log
  D15 and D16.

## Results (macro accuracy; Δ vs warm; 95% stratified-image CI shown for RL400 only)
| Arm | VAL macro acc | Valid | Δ vs warm |
|---|---|---|---|
| warm (W) | **0.6784** | 0.973 | — |
| RL400 | **0.6982** | 0.977 | **+1.98 pp** [1.34, 2.60] |
| SFT 1e-5 exposure | 0.7109 | 0.977 | +3.25 pp |
| SFT 5e-5 exposure | 0.7399 | 0.983 | +6.15 pp |
| SFT 1e-4 exposure | 0.7301 | 0.990 | +5.17 pp |
| **SFT 1e-5 equal time** (strongest declared) | **0.7539** | 0.988 | **+7.55 pp** |
| SFT 5e-5 equal time | 0.7534 | 0.983 | +7.50 pp |
| SFT 1e-4 equal time | 0.7520 | 0.987 | +7.36 pp |

**RL400 vs strongest SFT: −5.57 pp.** The 95% CI is [−6.63, −4.52] by stratified image bootstrap and
[−10.14, −1.27] by class-cluster bootstrap. RL400 also trails the SFT run at the **same LR and the same
exposure** (−1.27 pp). Not every pairwise gap is robust to class-cluster resampling: RL vs SFT 1e-4
exposure [−6.99, +0.37] and vs SFT 1e-4 equal time [−11.23, +0.30] include zero.

## What this means
- RL does transfer to held-out images. The +1.98 pp over warm has a class-cluster CI of
  [0.74, 3.33]. (The in-sample TRAIN32 greedy gain was +3.1 pp, so the in-sample assay did not hide it.)
- On this setup, budget-matched, LR-tuned SFT beats this GRPO recipe: the strongest SFT is ahead by
  5.57 pp, with both CIs excluding zero. RL400 fails all five point conditions of `substantial_rl`: accuracy
  ≥ 0.80 (0.698); ≥ +5 pp over W (+1.98); ≥ +5 pp over strongest SFT (−5.57); relative error reduction
  ≥ 0.20 vs strongest SFT (−0.226); paired 95% lower bound > 0 (−6.63 pp). The sixth condition,
  `validation_replicates` (seeds 42/314/2718), cannot be tested with one training seed. The protocol
  is not activated for this evaluation. The evaluator's `primary_requirement_satisfied` is always false
  by construction and carries no information.
- The diagnosis (`runs/aircraft-controlled400-diagnosis-v1`) points to why: 65.75% of RL updates carried
  zero task advantage, every useful update was gradient-clipped (its effect is unverified), and KL
  never bound. That RL received less task signal per unit of compute than SFT fits this, but it was
  not measured. The diagnosis lives in an ignored local run directory, not on this branch.
- Caveats: one training seed; VAL is the selection set for the strongest-SFT choice; capture dependence
  and pretraining contamination are not excluded (`uncertainty.caveat`).

## Consequence for the next study (pre-registered branch, `docs/aircraft-freshpool-study-v1.json`)
- B0 passes (W valid 0.973 ≥ 0.95). B1 passes (W 0.678 ∈ [0.15, 0.85]).
- B2: a* = 0.7539, so D_req = max(0.05, 0.2(1−a*), 0.80−a*) = 0.05 ≤ 0.10, so **full mode**.
- B3: all six SFT endpoints enter the strongest-SFT pool. B4: the best SFT LR is 1e-5, so no 2e-4 arm.
- B5: W VAL is reused. The `common` arm's weights hash in
  `docs/aircraft-native-val-activation-v1/activated-declaration.json` is `f05840be…`, the warm init the
  study declares.
- B6: RL400's VAL is reported only as a stale-pool reference, never as a pool effect.
- To pass the frozen gates, a fresh-pool RL arm would need VAL ≥ 0.8039 (+12.55 pp over W). That is a
  lower bound: fresh-pool SFT arms also join the strongest-SFT pool and can raise a*. The design
  panel's prior for this was ~5–8%.
