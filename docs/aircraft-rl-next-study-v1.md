# aircraft-rl-next-study-v1 (design: aircraft-signal-limit-v1): pre-registration draft

**Status:** this draft is unactivated. **Amendment 1 (2026-10-18, before any output)** applies the PR #26 design review; §13 lists every finding with its verdict, and the sections below already read as amended. Before activation it needs root review, a new protocol (`docs/aircraft-rl-protocol-signal-limit-v1.json`, see §2), the code changes in §12, and a frozen copy that root activates. Amendments are append-only. Nothing in this document came from running a model. Every number below either names its source file or instrument, or is marked **estimate** or **unverified**. The machine-readable twin is `docs/aircraft-rl-next-study-v1.json`. Where the two disagree, the JSON wins.

**Why this study exists.** The freshpool screened-GRPO study (`docs/aircraft-freshpool-study-v1.*`) stopped at G1 on two checks: acceptance was 0.2975 against a 0.30 threshold, and validity was 0.8926 against 0.90 (`runs/aircraft-freshpool-v1/gates/G1-report-1.json`). That stop was pre-registered and is final, so the study is not rerun. The forensics found two problems in the rollouts:
- 18 of the 47 mixed G4 groups were format-only. In those groups, every wrong answer was an invalid string such as "King Air 350" or "EMB-145".
- W is overconfident: 187 of 400 candidates have p > 0.841.

**Lineage.** Three designs were judged, and all three judges ranked `aircraft-signal-limit-v1` first (scores 4/10, 9/10 and 7/10). Five ideas were grafted onto it:
1. From the constrained-G8 design: the label trie as a CPU-only fairness layer, a constrained-CE SFT arm (A5) tuned on the full LR grid, and the free invalid-to-correct upper bound.
2. From sft-then-rl: the exposure-matched SFT-E arm at all three LRs in Stage B.
3. From sft-then-rl: a zero-training pre-stage, P0-I, that scores candidates at the existing strongest SFT adapter I.
4. From sft-then-rl: the refusal arithmetic a_floor = N × 1.25 / |S_b|.
5. From judge 1: a Bonferroni-adjusted paired lower bound on Gate A.

Two parts of the constrained design were deliberately left out:
- The trie mask inside GRPO sampling and the GRPO loss. Judge 3 identified this as the largest engineering surface and the most likely place for a bias bug.
- A constrained primary evaluator. Judge 2 noted it would invite a "moved the goalposts" objection.

---

## 0. Facts relied on

| Fact | Value | Source / instrument |
|---|---|---|
| Parent G1 | acceptance 0.2975, valid 0.8926, EOS 1.0, mean u 0.715, mixed 0.367 observed vs 0.311 predicted, scorer 0.742 s/candidate, non-canonical share 0.030 | `runs/aircraft-freshpool-v1/gates/G1-report-1.json` |
| p histogram at W (n = 400) | 93 below 0.159, 119 in the G4 band, 187 above 0.841 | the fragment journal at `runs/aircraft-freshpool-v1/stages/p0/instance-1/fragment.json` |
| Predicted acceptance at W for u_G ≥ 0.5 | G4 0.2975 (band [0.159, 0.841]); G8 0.420 ([0.083, 0.917]), mean u 0.825; G16 0.5525 ([0.042, 0.958]), mean u 0.871 | scratch `next/calc.py` on a copy of the journal, run with `python3 -I` |
| G8 calibration at W | pooling both seeds into one 8-sample group gives 32/64 observed mixed vs 0.448 predicted | signal-limit design grounding (scratch `an.py`, `an2.py`) |
| Format-only signal | 18/47 mixed groups contrast a correct answer only with invalid strings; 36/55 invalid answers have Beechcraft or Embraer-regional gold labels | G1 forensics (`cat.py`) |
| SFT cost | 3.294 s/update (8,116.34 committed active s over 2,464 updates) | `runs/aircraft-controlled400-v1/work/sft-1e5/endpoint_receipt.json` |
| Screened G4 accepted update | 30.61 s (n = 4) | live freshpool E0 as cited by PR #19, not re-derived |
| Stage times | P0 797.4 s; E0 2,165.8 s; H-select 1,815.6 s; H-gate 2,015.6 s | `runs/aircraft-freshpool-v1/journal.jsonl` `stage_finished` events |
| VAL | 0.98 h per endpoint | native-VAL supervision journals as cited by PR #19 |
| Native VAL | W 0.6784, RL400 0.6982, strongest SFT sft-1e5-time 0.7539, so D_req = 0.05 | `reports/2026-10-17-controlled400-native-val/README.md` |
| I | sft-1e5-time, weights `8f27d0ca170284a2…` | `docs/aircraft-native-val-activation-v1/activated-declaration.json` |
| G8 β=0 accepted update | **44 s, unmeasured estimate**. Stage 0 replaces it with c(G8) before any RL forecast or cap is computed | assumes cost linear in G, minus the reference forward |

## 1. Algebra, and one correction to the base design

- **Exact-match GRPO with TRL's group-std advantage.** E[update] = w_G(p) × the gold-SFT gradient. Here w_G is `expected_gradient_weight()` (`src/vision_rl/vlm_local_training.py:1013`), which weights mid-p images most.
- **Dr-GRPO (`scale_rewards="none"`).** E[Σ_i (r_i − r̄) ∇log π(y_i)] = (G−1)·p·∇log π(gold) = (G−1)·∇p, for every G.
  - **Instrument.** Exact enumeration equals (G−1)p for G ∈ {4, 8, 16} and p ∈ {0.1, 0.5, 0.9} (asserted to 1e-12). A 200k-draw Monte Carlo gives 2.1003 against 2.1 (G8, p = 0.3) and 2.1001 against 2.1 (G4, p = 0.7). Script: scratch `next/calc.py`.
  - **Consequence.** The limit of β=0 Dr-GRPO is **CE weighted by p**, normalised to 2p. It does not depend on G, and it upweights *easy* images, the opposite of the std version.
  - **Correction.** The base design called this limit "w8(p)/w8(0.5)", which is wrong for no-std scaling.
  - **With screening.** The limit is CE × 2p × 1[p ∈ B_G]. A larger G only lowers variance and widens the band B_G.
- **Hard-negative reward** (r(gold) = 1, r(c) = −0.5 for every frozen-map confuser c). **Corrected in amendment 1.** For any reward the Dr-GRPO expected update is (G−1)·∇E_π[r]. For `exact_hn` that is (G−1)[p·∇log π(gold) − 0.5·Σ_{c∈M(gold)} π(c)·∇log π(c)], times the band: the gold term is p-weighted, and the confuser term sums over every map confuser. A3 (unweighted CE + 0.5·∇[−log(1−π(c))] on the single top live confuser, no band) is **not** this limit. So `exact_hn` is not used, and A3 becomes a supervised comparator.
- **Partial (hierarchical) credit.** Its limit is hierarchical label smoothing.
- **What the algebra ignores.** KL (β = 0 here), token-length normalisation, and the max_grad_norm = 1.0 clip. The clip bound on every useful RL400 step (`runs/aircraft-controlled400-diagnosis-v1/report.md`), so per-image weights only change steps whose norm is below 1.
  - **Measured for CE (amendment 1).** The pre-clip `grad_norm` in `trainer_state.json` exceeds 1.0 on 47.3% (sft-1e5), 34.4% (sft-5e5) and 32.6% (sft-1e4) of controlled400 SFT steps, and on 137/400 = 34.3% of RL400 steps. Instrument: `python3 -I` over `runs/aircraft-controlled400-v1/work/*/trainer_state.json`. (`optimizer_audit.jsonl` logs the norm *after* clipping, maximum 1.0000000107 on RL400.) So on a third to a half of CE steps the clip erases A2's 2p factor, and A2 is then band-masked CE on those steps.
  - **Consequence.** The limit is of the unclipped expected gradient. RL's expected step is E[clip(ĝ)], not clip(E[ĝ]). The clip stays, because RL, every supervised arm and the whole strongest-SFT pool lineage all train with it. Gate A's claims are narrowed instead (§6).

The study first tests the A2 limit at SFT prices (Stage A). It spends RL compute (Stage B) only if that limit beats plain CE, HN-SFT and constrained CE.

## 2. Hypotheses, scope, protocol revision

- **HA** (amended). A2\*, the supervised limit of screened exact-reward Dr-GRPO, beats C\* = max(A1\* CE, A3\* HN-SFT, A5\* constrained CE) on H-select by at least +0.03 sampled macro. Its paired lower bound must be above 0 after the small-n correction and a Bonferroni adjustment over 3 candidates. All of this is under the fixed profile: max_grad_norm 1.0, one prompt per update, constant AdamW.
- **HB** (tested only if HA holds). Screened Dr-GRPO with β = 0 and the exact reward, started from I\*, beats the strongest fair SFT on native VAL by the unchanged `substantial_rl` margins.
- **HB-robust.** The HB gain also survives A2's objective continued from I\* for the same active time (LIM-T), at seed 42 and in each replicate seed's pool. The claim rests on this robust verdict.
- **What a Gate A pass means** (amendment 1). If RL behaves like its own limit, it cannot beat LIM-T by +0.05: at G8, LIM-T gets about 13× more updates. So passing Gate A raises the bar for Stage B rather than predicting success. Stage B passes only if the clipped, finite-G, single-sample RL update does something its expected-gradient limit does not. HB tests exactly that.
- **Scope.** Label access is train400 ∪ S_a for Stage A and train400 ∪ the scanned part of S_b for Stage B. Gains that come from the extra labels are credited to the data, not to RL.
- **Protocol.** The new protocol is `docs/aircraft-rl-protocol-signal-limit-v1.json`, revising `aircraft-rl-protocol-freshpool-v1-activated.json`.
  - **Changed:** train membership, the S_a/S_b split, init (W for Stage A, I\* for Stage B), the RL profile (G ∈ {8, 16}, dr_grpo, `scale_rewards` none, β = 0) and the control set.
  - **Byte-identical, asserted by a test:** `substantial_gain_gates` and `strong_baseline`.
  - **Unchanged:** the frozen native-VAL evaluator (unconstrained greedy), and H-select and H-gate.

## 3. Data

- **S.** 2,027 images in `runs/aircraft-freshpool-g0-v1/manifests/s.jsonl` (the G0 audit survivors). S is unspent: nothing was trained on it, and only the first 400 had their labels read, by the W scorer.
- **S_a / S_b.** Split within each class. Images are ranked by sha256 of `[protocol_id,"signal-limit-split",791,label,id]`; odd ranks go to S_a and even ranks to S_b.
  - **Preview:** S_a has 1,040 images and S_b has 987, with 11–26 images per class. Instrument: scratch `next/split.py`, which reads only ids and labels. The exact sizes are re-emitted and hashed at G0.
  - **Stage A order:** `random.Random(42).shuffle(sorted(ids))`, the same for every arm.
- **H-select and H-gate.** The existing freshpool manifests, with their pinned salt. H-select is used only for selection and H-gate only for Stage B's G3.
- **Look ledger.** A cross-study ledger of every H-select, H-gate and VAL look, starting from freshpool's W looks.
- **VAL** is used for the final gates only. **TEST** stays sealed until G6.

## 4. Fixed profile and label trie

- **Model and training.** Qwen2-VL-2B at the pinned revision, FP32 MPS, Q/V LoRA r8/α16. Answer-only prompt with the full 100-label list, maximum 256 tokens. Constant AdamW, weight decay 0, max_grad_norm 1.0, one prompt per update.
- **Assays.** H-select uses greedy plus one G4 seed (5 completions per image). H-gate uses greedy plus G4 seeds 42 and 314 (9 per image). The assay G stays at 4 whatever G is used for training.
- **Label trie** (`scripts/build_label_trie.py`, CPU, tokenizer only).
  - **Contents:** every tokenisation of the 100 labels plus EOS, pruned to at most canonical length + 2 tokens. The prune list and the sha are frozen at G0.
  - **Coverage check:** every valid greedy `completion_ids` in the 8 controlled400 VAL journals must lie in T. The constrained design counted 26,191 such paths; the script recounts them and refuses on any mismatch.
  - **Forced fire:** a planted off-trie row must be masked.
  - **Upper-bound table:** the macro each endpoint would score if every invalid answer became correct, computed from existing predictions. Constrained design values: W 0.7057, sft-1e5-time 0.7662, sft-5e5-time 0.7702. These are recomputed at G0. **So validity alone is worth at most +2.7 pp to W and +1.2 pp to the strongest SFT (sft-1e5-time; sft-5e5-time +1.7 pp).** Instrument: `headroom_pp` in `reports/2026-10-18-signal-limit-label-trie/g0-preview.json`. (Amendment 1 corrects the draft's "+1.6 pp", which was sft-5e5-time's figure.)
  - **Where the trie is used:** constrained greedy is reported on every H assay, and the trie mask defines the cCE loss of A5. It is **never** used in GRPO sampling or loss, and never in the primary evaluator.
  - **Disclosures (amendment 1).**
    - The coverage check reads the 8 controlled400 VAL journals, so VAL greedy outputs decided which tokenisations the build had to cover. The pre-declared prune rule defines T, not the journals, and no VAL label or score enters T. Its influence on A5's training mask is at the tokeniser level only.
    - The P0 confusion map came from freshpool P0 rollouts on the first 64 images of `s_scan_seed42`. Those are S images, now split between S_a and S_b. Only the wrong labels W produced were used, never a gold score.

## 5. Stages and arms

### P0-I: zero-training pre-stage (graft from judge 3), about 0.25 h (estimate)
- **Work.** Run the existing `p0_screen` at G4 with init **I** (`8f27d0ca…`) on the first 400 S_b candidates (scan seed 42), with 64 × 2 G4 rollouts. Predicted acceptance and mean u for G ∈ {4, 8, 16} come from p alone.
- **New code.** None beyond the `init_endpoint` plan field.
- **Advisory since amendment 1.** PI publishes a forecast of GB0 and never stops the study (§6).

### Stage 0: cost probe and E0, about 3.0 h (estimate)
- **Cost probe** (amended). At W, on train400 images, run screened β = 0 dr_grpo at G8 and at G16, each until it has at least 8 accepted updates. This gives c(G8) and c(G16), scorer time included. Both are **measured**, not extrapolated from one step. They are written to the state file before any RL forecast reads them.
- **E0** (amended). 4 uninterrupted updates must be byte-identical to 2+2 and 1+3 resumed, for β = 0 / dr_grpo / `scale_rewards` none, **at G8 and at G16**.
- **Launcher check.** The launcher refuses a Stage 0 fragment unless it has `e0_groups` [8, 16] and at least 8 `c_accepted_steps` per G.

### Stage A: limit screen
All arms start from W and run 600 updates in the same S_a order. LRs come from {1e-5, 5e-5, 1e-4}. Selection is by H-select sampled macro, using the unconstrained evaluator.

| Arm | Objective | LRs | Role |
|---|---|---|---|
| A1 | CE-SFT (gold+EOS) | 3 | baseline |
| A2 | CE × stopgrad[2p] × 1[p ∈ B_8], with p from the same forward pass. This is the screened Dr-GRPO limit. Amended: an image outside B_8 takes **no optimizer step**, so AdamW's parameters, moments and step count are untouched. Its presentation still counts, so A2 sees the same 600 S_a images as every other arm. The in-band step count is reported | 3 | **the only Gate A candidate** |
| A3 | CE + 0.5 × −log(1 − π(c)), where c is the model's top valid wrong label within the frozen confusion map | 3 | supervised comparator in C\*, and an I\* candidate. Not an RL limit (§1) |
| A4 | hierarchical label smoothing, ε = 0.1 (taxonomy-train-only) | A1\*'s LR | reported control; **cannot pass Gate A** |
| A5 | constrained CE: log-softmax over trie-allowed tokens (graft) | 3 | baseline |

That is 13 arms in total.

### Stage B: RL, only if GA passes (PI is advisory)
- **Init and P0'.** I\* is the H-select-best of A1\*–A3\*. P0' scores the same 400 S_b candidates at I\* and runs rollouts at the chosen G.
- **RL(lr).** Screened GRPO at G ∈ {8, 16}, dr_grpo, `scale_rewards` none, β = 0, T 1.0 / top-p 1 / top-k 0.
  - **Reward:** exact-EOS 0/1, the reward whose screened Dr-GRPO limit is A2. Amendment 1 drops `exact_hn`.
  - **Screen:** accept when u_G(p) ≥ 0.5.
  - **Schedule:** all three LRs run to 100 accepted updates. The top two continue without reset to **N = 300**, and RL\* is the H-select best at 300.
- **Controls:**

| Arm | Definition | What it isolates |
|---|---|---|
| SFT-E(lr) ×3 (graft from judge 2) | I\* + CE on exactly RL\*'s 300 accepted IDs, in the same order | data selection by the screen |
| SFT-T(lr) | SFT-E continued without reset over RL\*'s full scanned S_b sequence (cycling) up to T_RL, for every LR within 2 pp of the best SFT-E. There is no intermediate checkpoint grid; SFT-E(lr), its own start, is gated in G3a and pooled at VAL (§13, M5) | compute and label access |
| LIM-T | A2's objective, with the chosen G's band and no step outside it, continued from I\* over the same sequence up to T_RL at A2\*'s LR. Rerun for each replicate seed | the objective in expectation (robust verdict and the G3b gate) |

### Strongest SFT
- **Protocol pool:**
  - the 6 controlled400 SFT endpoints, with VAL reused on an identity match (a\* = 0.7539);
  - the Stage-A A1, A3 and A5 endpoints at every LR within 2 pp of their family's best on H-select;
  - I\*;
  - SFT-E ×3;
  - every qualifying SFT-T.
- **Robust pool:** the protocol pool plus LIM-T plus A2\*. A3 is already in the protocol pool.
- The pool only grows. D_req is recomputed from the realised a\*.

### Budget matching and caps
- **Stage A.** Every arm gets the same 600 presentations, in the same order, on the same grid. A2's out-of-band presentations take no optimizer step, so A2 is matched on data, not on optimizer steps.
- **SFT-E.** Equal exposure.
- **SFT-T and LIM-T.** Equal active seconds T_RL, which includes scorer forwards for every scanned candidate, plus equal label access.
- **Caps.** Each arm is capped at **1.25 × its forecast active seconds**, where forecast = measured unit cost × planned units, using c(G) from Stage 0 for RL. The cap does not reset on resume. A capped or interrupted arm is re-run from scratch and never quoted.

## 6. Gates (in order; every failure is kept and published)

| Gate | When | Pass (all must hold) | On fail |
|---|---|---|---|
| G0 | before any model run | S_a/S_b hashed and disjoint from H, VAL and TEST; trie coverage 26,191/26,191, forced fire passes, sha frozen; upper-bound table published; protocol committed with byte-identical gates (test); **a_floor = 375/\|S_b\|** published (preview 375/987 = 0.380); confusion map and taxonomy pinned; ledger initialised | do not launch |
| PI (**advisory**) | P0-I at I | forecast only: max over G ∈ {8, 16} of predicted acceptance ≥ a_floor; scorer ≤ 2.0 s/candidate; EOS ≥ 0.95; sampled validity ≥ 0.95. Report mixed calibration and the p histogram | publish the failed forecast and continue. I only proxies I\*, and Stage A's result does not depend on it |
| E0 | Stage 0 | resume runs byte-identical **at G8 and G16**; no reference forward at β = 0, with the reference identity recorded; finite nonzero gradient; sampler–teacher agreement ≤ 0.001; c(G8) and c(G16) written, each over ≥ 8 accepted steps | engineering fix, charged to engineering |
| GA | Stage A done | C\* = max(A1\*, A3\*, A5\*). (i) A2\* minus C\* ≥ **+0.03** H-select sampled macro; (ii) the paired one-sided bootstrap LB, **small-n corrected**, at quantile **0.05/3** (Bonferroni over A2 ×3) is > 0; (iii) under constrained greedy, A2\* minus C\* is > 0; (iv) A2\*'s validity ≥ C\*'s validity | **STOP and publish the negative.** Narrowed claim: under the fixed profile, the data-matched screened Dr-GRPO limit does not beat CE, HN-SFT or cCE. This says nothing about RL itself, about other rewards, or about an unclipped limit. A pass is necessary for Stage B, not a prediction of it. A2 is screened at B_8 even if GB0 later picks G16, and A2's token-mean CE differs from dr_grpo's per-token sum by a 1/len(gold) weight; both are disclosed |
| GB0 | P0' at I\* | G is the smallest value in {8, 16} with predicted acceptance ≥ a_floor; valid ≥ 0.95; EOS ≥ 0.95; observed mixed within ±0.10 of prediction; mean u among accepted ≥ 0.60; scorer ≤ 2.0 s | stop |
| G2 (futility) | RL at 100, three LRs | best RL minus I\* > 0 on H-select; invalid ≤ I\*'s + max(2, 2·√(invalid + I\*'s invalid)), which is two Poisson SDs of the difference; adapter changed; rolling useful-mixed ≥ 0.6 | stop |
| G3a | RL\* and SFT-E done | on H-gate: RL\* − SFT-E\* ≥ **+0.05** (mechanism mode); RL\* − I\* one-sided 90% paired LB, small-n corrected, > 0; validity and EOS ≥ 0.95 | stop before SFT-T |
| G3b | SFT-T and LIM-T done | on H-gate: RL\* − SFT-T\* ≥ **+0.05** and RL\* − LIM-T ≥ **+0.05** (LIM-T gates, amendment 1) | stop before VAL |
| G4 | seed-42 VAL | protocol `substantial_rl`, **unchanged**: accuracy ≥ 0.80; ≥ +0.05 vs W (the protocol's "init" is W, even though Stage B starts from I\*, which sits in both pools); ≥ +0.05 vs strongest SFT; relative error reduction ≥ 0.20; paired 95% class-stratified LB > 0. Both the protocol verdict and the robust verdict are reported, and **the claim rests on the robust verdict** | FAILED; publish every endpoint |
| G5 | seeds 314 and 2718, only if G4 passes | protocol replication rule, unchanged. Each seed's strongest SFT comes from its **robust** pool: every seed-42 robust member that no replicate reruns (including I\* and A2\*), plus that seed's SFT-E and SFT-T at every LR within 2 pp, plus its own LIM-T | FAILED; publish |
| G6 | once, only if G4 and G5 pass | TEST for RL\*, the **robust-verdict** strongest SFT, and W; 27B baseline per `strong_baseline` | — |

## 7. Bootstrap and endpoints

- **H splits.** Paired, class-stratified, image-cluster bootstrap: images are resampled within each class, all of an image's completions move together, and both arms use the same indices. 20,000 resamples with seed 2718, the same settings as the activated native-VAL declaration.
  - **Why cluster by image:** on the parent P0, the iid validity CI was [0.8625, 0.9180] while the image-cluster CI was [0.842, 0.9375] (G1 forensics).
  - **Quantiles:** Gate A uses 0.05/3 = 0.0167; G3 uses 0.10.
  - **Small-n correction (amendment 1).** Both H splits hold exactly 2 images per class (counted from the manifests). The within-class bootstrap of n images has (n−1)/n of the unbiased variance: half at n = 2, so the SE comes out about 0.71× too small. Each gating H bound is therefore rescaled to mean + √(n/(n−1)) × (plain bound − mean). With equal n in every class, that equals rescaling every resampled class mean about its observed value. The code is `small_n_lower_bound` in the evaluator; unequal class sizes are refused. VAL keeps the protocol's unchanged bound.
- **VAL.** The protocol's `compare_classification` `stratified_image_ci95`, unchanged.
- **Primary endpoint.** Native VAL mean-class accuracy under the unconstrained frozen greedy evaluator.
- **Selection and gating endpoints.** H-select macro (5 completions) for selection; H-gate macro (9 completions) for gating.
- **Secondary endpoints.**
  - constrained-greedy macro;
  - validity and EOS;
  - the decomposition RL−SFT-E (selection), SFT-E−SFT-T (compute) and RL−LIM-T (objective);
  - per-class Δ for 737-500/600, DR-400/SR-20, A318/A319/A320, Gulfstream IV/V, Beechcraft 1900/B200, ERJ 135/145 and EMB-120;
  - acceptance drift;
  - pre-clip norms and clip fraction per arm, from `trainer_state.json` `grad_norm`. For A2 and LIM-T, also the in-band optimizer-step count and the clip fraction among in-band steps.

## 8. Refusal conditions (any one stops the study)

- GA fails. GB0 fails, including the case where no G qualifies at I\*. (PI is advisory.)
- The scan exhausts S_b before N = 300.
- The rolling accepted useful-mixed fraction falls below 0.40 over any 100 accepted updates.
- An arm hits its 1.25× cap.
- NaN, or an adapter or reference identity change. β = 0 without a recorded `loss_type`.
- Trie coverage below 26,191/26,191, or a count mismatch. c(G) missing when an RL forecast is computed, measured over fewer than 8 accepted steps, or E0 not proven at both G8 and G16.
- An H or VAL look that is not in the ledger.
- The disk guard (5 GiB reserve), RSS guard or MPS-driver guard fires, or a second MPS owner appears. **This includes the 27B MLX baseline while it runs: nothing in this study runs while it holds the GPU.**
- Any sealed TEST access before G6.

Nothing changes after any output is seen: no threshold, margin, seed, N, set of G values, grid, split salt or confusion map.

## 9. Compute plan (MPS hours, **estimates** extrapolated from §0; c(G8) = 44 s and c(G16) = 88 s are unmeasured)

| Step | h | Cumulative |
|---|---|---|
| P0-I | 0.25 | 0.25 |
| Stage 0: probe (≥ 8 accepted at G8 and at G16) + E0 at G8 and G16 | 3.0 | 3.25 |
| Stage A: 13 arms × 600 updates (A1/A4/A5 3.29 s; A2 ~3.5 s and A3 ~6 s are estimates) | 8.6 | 11.85 |
| Stage A: 13 H-select assays + constrained greedy | 7.25 | **19.1 (GA)** |
| P0' at I\* | 0.4 | 19.5 |
| RL: 700 accepted updates × 44 s | 8.6 | 28.1 |
| RL H-select ×5 | 2.5 | 30.6 |
| SFT-E ×3 + H-select ×3 | 2.35 | 32.95 |
| H-gate for RL\*, SFT-E\* and I\* (G3a) | 1.7 | 34.65 |
| SFT-T ≤2 and LIM-T to T_RL (13,200 s each) + H-select ×3 | 12.5 | 47.15 |
| H-gate for SFT-T\* and LIM-T (G3b) | 1.15 | 48.3 |
| VAL, about 12 new endpoints (A3 pool members added) | 11.8 | **60.1 (G4)** |

- **Early stops (cumulative h):** GA 19.1, GB0 19.5, G2 28.1, G3a 34.65, G3b 48.3. PI no longer stops.
- **If GB0 chooses G16** (amendment 1; c(G16) ≈ 88 s is an estimate, linear in G). P0' becomes 0.8 h, RL 17.1 h, and SFT-T/LIM-T to T_RL ≈ 26,400 s comes to 23.5 h. That puts G4 at about **80 h**.
- **Beyond G4:** each replicate seed adds about 29–34 h at G8, including its own LIM-T, and about 50–60 h at G16. TEST adds about 2.9 h. The full conditional total is about 125 h at G8 and 190 h at G16.
- **Expected spend: about 29 h** = 19.1 + 0.25 × 41. P(GA) ≈ 0.25 is judgement.

## 10. Prior

**p_substantial ≈ 0.02** (range 0.015–0.02; judgement). All three designs and all three judges put the base rate here. The §1 algebra gives no reason to raise it. The β = 0 Dr-GRPO limit is p-weighted CE, which upweights easy images, and the clip erases that weighting on many steps, so P(GA) may well be below 0.25. The study is still worth running because its most likely stop, GA at about 19 h, publishes a clean negative at SFT prices. That negative is narrowed (amendment 1): under the fixed profile, the data-matched screened Dr-GRPO limit does not beat CE, HN-SFT or constrained CE here.

## 11. Reporting

- Raw outputs are committed before any interpretation.
- The full Stage-A table: every arm × LR, unconstrained and constrained macro, and validity.
- Both G4 verdicts, plus the three-way decomposition.
- The look ledger.
- H results are native-TRAIN development evidence only.

## 12. Implementation work list

The items below were implemented on branch `signal-limit-impl-v1.noindex`, together with amendment 1's code (the last bullet).

- **`src/vision_rl/vlm_local_training.py`**
  - `trainer_kwargs` (l.444): add the parameters `loss_type` ('grpo' | 'dr_grpo') and `scale_rewards` ('group' | 'none'), replacing the literals at l.507–508 with the defaults unchanged. Refuse β = 0 unless `loss_type` is passed explicitly.
  - `train_local` (l.2624): forward β, `loss_type` and `scale_rewards`, and record them in settings only when they differ from the defaults, so legacy identities stay byte-identical. Extend the mode set at l.2680 with `hn_sft`, `hls_sft` and `cce_sft`. Give `egsft` the options `egsft_scale` ('group' | 'none') and `egsft_band_group`.
  - `expected_gradient_weight` (l.1013) and `egsft_weight` (l.1042): add a no-std branch that enumerates exactly and must equal (G−1)p, plus the band mask. They are consumed at l.1702.
  - `make_reward` (l.330): add a separately named `exact_hn` reward; `exact` is unchanged. (Implemented, but unused by this study since amendment 1.)
  - `_completion_logprob` (l.1047): an optional per-position allowed-token mask for cCE and for constrained scoring; the unmasked path stays byte-identical.
  - New `label_trie_logits_processor`, for assay generation only.
  - New hn and hls loss helpers. The hn helper refuses when c == gold.
- **`scripts/build_label_trie.py`** (new): builds the trie, runs the coverage check and produces the upper-bound table.
- **`scripts/prepare_aircraft_freshpool.py`**: new `split_signal_limit`, placed next to `split_holdout` (l.209) and `scan_order` (l.223).
- **`scripts/freshpool_launch/p0_screen.py`**: `GROUP` (l.53) and `main` (l.170) take G from the profile, refusing anything outside {4, 8, 16}. `g1_document` (l.126) emits predictions for G ∈ {4, 8, 16}.
- **`scripts/freshpool_launch/h_assay.py`**: `main` (l.132) separates the assay G (fixed at 4) from the training G. `assay` (l.71) adds a constrained-greedy pass.
- **`scripts/freshpool_launch/launch.py`**:
  - `GRID`, `RL_SCREEN_UPDATES` and `FINAL_UPDATES` (l.81, 83) become plan fields.
  - `w_endpoint` (l.512, 1165, 1307, 1404, 1492, 1584) becomes `init_endpoint`, pinned per stage.
  - `verify_stage_templates` (l.549) is driven by plan counts.
  - A new stage table: P0-I, Stage 0, Stage A and Stage B.
- **`scripts/evaluate_signal_limit_gates.py`** (new): reuses the `evaluate_freshpool_gates.py` helpers (`at_least`, `best`, `ranked`, `within`, `d_req`), generalises u to G, and adds `gate_pi`, `gate_a`, `gate_b0`, G2–G6 and both G4 verdicts.
- **`data/fgvc-aircraft-2013b/audit/`**: `taxonomy-train-only.json` and the frozen confusion map, both sha-pinned.
- **`docs/aircraft-rl-protocol-signal-limit-v1.json`** (new).
- **Tests.** Expected values come from fixtures, never from the subject:
  - no-std weights against hand values and a Monte Carlo check;
  - β = 0 refusal, and legacy identity unchanged;
  - the hn term lowers π(c) on a tiny CPU model;
  - trie fixtures written by hand, plus the forced fire;
  - gate fixtures, including the a_floor boundary (0.379 refuses, 0.380 passes at \|S_b\| = 987), the Bonferroni quantile and the G choice;
  - a split fixture with expected IDs;
  - the protocol's byte-identical gates.
- **Amendment 1 code:**
  - **`vlm_local_training.py`.** The EG-SFT band skips the optimizer step outside B_G. The trie, confusion-map and taxonomy loaders read the builder's committed formats, and the trie loader checks the trie's content sha256.
  - **The evaluator.** It adds the small-n bound, makes A2 the only Gate A candidate, adds A3 to C\* and to the protocol pool, makes PI advisory, adds the G2 slack, makes LIM-T gate G3b, and gives G5 a robust per-seed pool.
  - **The launcher.** A2 is the only limit arm. The reward is always exact. The Stage 0 validator now checks c(G) and E0 at both G.
  - **Tests.** Each change has hand-computed fixtures and a forced fire.
- **Decision record:** at activation. Not in this PR.

## 13. Amendment 1: PR #26 design review (5 blocking, 10 minor)

The reviewer confirmed these as sound: the Dr-GRPO (G−1)p claim, no VAL or TEST in RL selection, choosing the strongest SFT by max VAL, C\* as a maximum, a_floor and its 375/987 boundary, the compute sums, and D_req = 0.05. The JSON's `design_review_v1` holds the full record.

| # | Finding | Verdict | Resolution |
|---|---|---|---|
| B1 | H splits have n = 2 per class, so the bootstrap has half the variance | **accepted** (2 × 100 counted in both manifests) | small-n-corrected bounds on Gate A and G3a, with a forced fire |
| B2 | the max_grad_norm 1.0 clip undoes "RL ≈ limit" | **partially accepted** | Measured: the clip binds on 33–47% of CE steps and 34% of RL400 steps. Claims are narrowed and clip fractions reported. **Rejected:** removing the clip. RL, every supervised arm and the whole pool lineage share it, and removing it would need a new LR grid |
| B3 | A3 is not the limit of `exact_hn` | **accepted** | A3 becomes a comparator, Stage B always uses the exact reward, and LIM-T always runs A2's objective. **Rejected:** redefining A3 as the exact limit. That would be a new, untested loss mode for a minority signal (29/128 groups) |
| B4 | A2's out-of-band images take phantom AdamW steps and mix three effects | **accepted** (momentum) | No optimizer step outside the band; A2 stays data-matched. **Rejected:** matching informative updates. 600 in-band steps would need about 1,430 presentations, more than S_a's 1,040 images |
| B5 | G3b, G5 and G6 ignore the robust pool | **accepted** | LIM-T gates G3b; G5 uses each seed's robust pool, with LIM-T run per seed; G6 names the robust strongest SFT |
| M1 | Gate A screens at B_8 but GB0 may choose G16; token-mean vs sequence-sum loss | disclosed | Rerunning Stage A after GB0, or changing the EG-SFT loss, was rejected |
| M2 | G2's +2 invalid slack fires under the null | **accepted** | slack max(2, 2√(a + b)) |
| M3 | the cost cap rests on one probe step; E0 only at G8 | **accepted** | c(G) measured over ≥ 8 steps at each G; E0 at both G; checked by the launcher |
| M4 | PI can stop the study on a proxy; PI ignores validity | **accepted** | PI is advisory and includes sampled validity |
| M5 | SFT-T has no checkpoint selection | partially accepted (disclosed) | SFT-E is SFT-T's start, gated in G3a and pooled at VAL. **Rejected:** a checkpoint grid. Each checkpoint costs about 0.5 h of H-select plus a VAL endpoint, and the robust verdict carries the claim |
| M6 | "+1.6 pp" is wrong | **accepted** | now +1.2 pp (sft-1e5-time); +1.7 pp is sft-5e5-time's figure |
| M7 | the compute plan has only G8 | **accepted** | G16 branch costed (§9) |
| M8 | G4 "vs init W" while Stage B starts from I\* | **accepted** (wording) | §6 G4 |
| M9 | VAL journals feed the trie; the P0 map saw S_b images | disclosed | §4 |
| M10 | Stage B's success depends on the algebra being wrong | **accepted** | §2, "What a Gate A pass means" |

