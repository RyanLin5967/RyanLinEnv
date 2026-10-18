# aircraft-rl-next-study-v1 (design: aircraft-signal-limit-v1): pre-registration draft

**Status:** this draft is unactivated. Before activation it needs root review, a new protocol (`docs/aircraft-rl-protocol-signal-limit-v1.json`, see §2), the code changes in §12, and a frozen copy that root activates. Amendments are append-only. Nothing in this document came from running a model. Every number below either names its source file or instrument, or is marked **estimate** or **unverified**. The machine-readable twin is `docs/aircraft-rl-next-study-v1.json`. Where the two disagree, the JSON wins.

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
- **Hard-negative reward** (r(gold) = 1, r(confuser) = −0.5). Its limit is CE plus unlikelihood pressure on the confuser.
- **Partial (hierarchical) credit.** Its limit is hierarchical label smoothing.
- **What the algebra ignores.** KL (β = 0 here), token-length normalisation, and the max_grad_norm = 1.0 clip. The clip bound on every useful RL400 step (`runs/aircraft-controlled400-diagnosis-v1/report.md`), so per-image weights only change steps whose norm is below 1.

The study first tests these limits at SFT prices (Stage A). It spends RL compute (Stage B) only if a limit beats plain CE and constrained CE.

## 2. Hypotheses, scope, protocol revision

- **HA.** The better of A2\* (the screened Dr-GRPO limit) and A3\* (CE plus confuser unlikelihood) beats C\* = max(A1\* CE, A5\* constrained CE) on H-select by at least +0.03 sampled macro, with a Bonferroni-adjusted paired lower bound above 0.
- **HB** (tested only if HA holds). Screened Dr-GRPO with β = 0, started from I\*, beats the strongest fair SFT on native VAL by the unchanged `substantial_rl` margins.
- **HB-robust.** The HB gain also survives the winning limit objective, continued from I\* for the same active time (LIM-T). The claim rests on this robust verdict.
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
  - **Upper-bound table:** the macro each endpoint would score if every invalid answer became correct, computed from existing predictions. Constrained design values: W 0.7057, sft-1e5-time 0.7662, sft-5e5-time 0.7702. These are recomputed at G0. **So validity alone is worth at most +2.7 pp to W and +1.6 pp to the strongest SFT.**
  - **Where the trie is used:** constrained greedy is reported on every H assay, and the trie mask defines the cCE loss of A5. It is **never** used in GRPO sampling or loss, and never in the primary evaluator.

## 5. Stages and arms

### P0-I: zero-training pre-stage (graft from judge 3), about 0.25 h (estimate)
- **Work.** Run the existing `p0_screen` at G4 with init **I** (`8f27d0ca…`) on the first 400 S_b candidates (scan seed 42), with 64 × 2 G4 rollouts. Predicted acceptance and mean u for G ∈ {4, 8, 16} come from p alone.
- **New code.** None beyond the `init_endpoint` plan field.

### Stage 0: cost probe and E0, about 1.4 h (estimate)
- **Cost probe.** At W, on 32 train400 images, time one screened G8 step (β = 0, dr_grpo) against one G4 β = 0.04 step. This gives c(G8) and c(G16), including scorer time, and they are written to the state file before any RL forecast reads them.
- **E0.** 4 uninterrupted updates against 2+2 and 1+3 resumed must be byte-identical for G8 / β = 0 / dr_grpo / `scale_rewards` none.

### Stage A: limit screen
All arms start from W and run 600 updates in the same S_a order. LRs come from {1e-5, 5e-5, 1e-4}. Selection is by H-select sampled macro, using the unconstrained evaluator.

| Arm | Objective | LRs | Role |
|---|---|---|---|
| A1 | CE-SFT (gold+EOS) | 3 | baseline |
| A2 | CE × stopgrad[2p] × 1[p ∈ B_8], with p from the same forward pass. This is the screened Dr-GRPO limit | 3 | candidate |
| A3 | CE + 0.5 × −log(1 − π(c)), where c is the model's top valid wrong label within the frozen confusion map | 3 | candidate |
| A4 | hierarchical label smoothing, ε = 0.1 (taxonomy-train-only) | A1\*'s LR | reported control; **cannot pass Gate A** |
| A5 | constrained CE: log-softmax over trie-allowed tokens (graft) | 3 | baseline |

That is 13 arms in total.

### Stage B: RL, only if PI and GA pass
- **Init and P0'.** I\* is the H-select-best of A1\*–A3\*. P0' scores the same 400 S_b candidates at I\* and runs rollouts at the chosen G.
- **RL(lr).** Screened GRPO at G ∈ {8, 16}, dr_grpo, `scale_rewards` none, β = 0, T 1.0 / top-p 1 / top-k 0.
  - **Reward:** exact-EOS if A2 won Gate A, or `exact_hn` (−0.5 for a frozen-map confuser) if A3 won.
  - **Screen:** accept when u_G(p) ≥ 0.5.
  - **Schedule:** all three LRs run to 100 accepted updates. The top two continue without reset to **N = 300**, and RL\* is the H-select best at 300.
- **Controls:**

| Arm | Definition | What it isolates |
|---|---|---|
| SFT-E(lr) ×3 (graft from judge 2) | I\* + CE on exactly RL\*'s 300 accepted IDs, in the same order | data selection by the screen |
| SFT-T(lr) | SFT-E continued without reset over RL\*'s full scanned S_b sequence (cycling) up to T_RL, for every LR within 2 pp of the best SFT-E | compute and label access |
| LIM-T | the winning limit objective continued from I\* over the same sequence up to T_RL | the objective in expectation (robust verdict) |

### Strongest SFT
- **Protocol pool:**
  - the 6 controlled400 SFT endpoints, with VAL reused on an identity match (a\* = 0.7539);
  - the Stage-A A1 and A5 endpoints at every LR within 2 pp of their family's best on H-select;
  - I\*;
  - SFT-E ×3;
  - every qualifying SFT-T.
- **Robust pool:** the protocol pool plus LIM-T plus A2\* and A3\*.
- The pool only grows. D_req is recomputed from the realised a\*.

### Budget matching and caps
- **Stage A.** Equal updates, equal order and the same grid for every arm.
- **SFT-E.** Equal exposure.
- **SFT-T and LIM-T.** Equal active seconds T_RL, which includes scorer forwards for every scanned candidate, plus equal label access.
- **Caps.** Each arm is capped at **1.25 × its forecast active seconds**, where forecast = measured unit cost × planned units, using c(G) from Stage 0 for RL. The cap does not reset on resume. A capped or interrupted arm is re-run from scratch and never quoted.

## 6. Gates (in order; every failure is kept and published)

| Gate | When | Pass (all must hold) | On fail |
|---|---|---|---|
| G0 | before any model run | S_a/S_b hashed and disjoint from H, VAL and TEST; trie coverage 26,191/26,191, forced fire passes, sha frozen; upper-bound table published; protocol committed with byte-identical gates (test); **a_floor = 375/\|S_b\|** published (preview 375/987 = 0.380); confusion map and taxonomy pinned; ledger initialised | do not launch |
| PI | P0-I at I | max over G ∈ {8, 16} of predicted acceptance ≥ a_floor; scorer ≤ 2.0 s/candidate; EOS ≥ 0.95. Report validity, mixed calibration and the p histogram | **STOP the study.** Stage A is not run under this id. Passing PI is necessary, not sufficient |
| E0 | Stage 0 | resume runs byte-identical; no reference forward at β = 0, with the reference identity recorded; finite nonzero gradient; sampler–teacher agreement ≤ 0.001; c(G) written | engineering fix, charged to engineering |
| GA | Stage A done | (i) the best of A2\*/A3\* minus C\* ≥ **+0.03** H-select sampled macro; (ii) the paired one-sided bootstrap LB at quantile **0.05/6** (Bonferroni over A2 ×3 and A3 ×3) is > 0; (iii) under constrained greedy, the winner minus C\* is > 0; (iv) the winner's validity ≥ C\*'s validity | **STOP and publish the negative**: no answer-only signal-density limit beats CE or cCE here |
| GB0 | P0' at I\* | G is the smallest value in {8, 16} with predicted acceptance ≥ a_floor; valid ≥ 0.95; EOS ≥ 0.95; observed mixed within ±0.10 of prediction; mean u among accepted ≥ 0.60; scorer ≤ 2.0 s | stop |
| G2 (futility) | RL at 100, three LRs | best RL minus I\* > 0 on H-select; invalid ≤ I\*'s + 2; adapter changed; rolling useful-mixed ≥ 0.6 | stop |
| G3a | RL\* and SFT-E done | on H-gate: RL\* − SFT-E\* ≥ **+0.05** (mechanism mode); RL\* − I\* one-sided 90% paired LB > 0; validity and EOS ≥ 0.95 | stop before SFT-T |
| G3b | SFT-T and LIM-T done | on H-gate: RL\* − SFT-T\* ≥ **+0.05**. RL\* − LIM-T is reported but does not gate | stop before VAL |
| G4 | seed-42 VAL | protocol `substantial_rl`, **unchanged**: accuracy ≥ 0.80; ≥ +0.05 vs W; ≥ +0.05 vs strongest SFT; relative error reduction ≥ 0.20; paired 95% class-stratified LB > 0. Both the protocol verdict and the robust verdict are reported, and **the claim rests on the robust verdict** | FAILED; publish every endpoint |
| G5 | seeds 314 and 2718, only if G4 passes | protocol replication rule, unchanged. Each seed's strongest SFT is SFT-E and SFT-T at every LR within 2 pp, plus I\* | FAILED; publish |
| G6 | once, only if G4 and G5 pass | TEST for RL\*, the strongest SFT and W; 27B baseline per `strong_baseline` | — |

## 7. Bootstrap and endpoints

- **H splits.** Paired, class-stratified, image-cluster bootstrap: images are resampled within each class, all of an image's completions move together, and both arms use the same indices. 20,000 resamples with seed 2718, the same settings as the activated native-VAL declaration.
  - **Why cluster by image:** on the parent P0, the iid validity CI was [0.8625, 0.9180] while the image-cluster CI was [0.842, 0.9375] (G1 forensics).
  - **Quantiles:** Gate A uses 0.05/6 = 0.00833; G3 uses 0.10.
- **VAL.** The protocol's `compare_classification` `stratified_image_ci95`, unchanged.
- **Primary endpoint.** Native VAL mean-class accuracy under the unconstrained frozen greedy evaluator.
- **Selection and gating endpoints.** H-select macro (5 completions) for selection; H-gate macro (9 completions) for gating.
- **Secondary endpoints.**
  - constrained-greedy macro;
  - validity and EOS;
  - the decomposition RL−SFT-E (selection), SFT-E−SFT-T (compute) and RL−LIM-T (objective);
  - per-class Δ for 737-500/600, DR-400/SR-20, A318/A319/A320, Gulfstream IV/V, Beechcraft 1900/B200, ERJ 135/145 and EMB-120;
  - acceptance drift;
  - pre-clip norms and clip fraction.

## 8. Refusal conditions (any one stops the study)

- PI fails. GA fails. GB0 fails, including the case where no G qualifies at I\*.
- The scan exhausts S_b before N = 300.
- The rolling accepted useful-mixed fraction falls below 0.40 over any 100 accepted updates.
- An arm hits its 1.25× cap.
- NaN, or an adapter or reference identity change. β = 0 without a recorded `loss_type`.
- Trie coverage below 26,191/26,191, or a count mismatch. c(G) missing when an RL forecast is computed.
- An H or VAL look that is not in the ledger.
- The disk guard (5 GiB reserve), RSS guard or MPS-driver guard fires, or a second MPS owner appears. **This includes the 27B MLX baseline while it runs: nothing in this study runs while it holds the GPU.**
- Any sealed TEST access before G6.

Nothing changes after any output is seen: no threshold, margin, seed, N, set of G values, grid, split salt or confusion map.

## 9. Compute plan (MPS hours, **estimates** extrapolated from §0; c(G8) = 44 s is unmeasured)

| Step | h | Cumulative |
|---|---|---|
| P0-I | 0.25 | 0.25 |
| Stage 0 probe + E0 | 1.4 | 1.65 |
| Stage A: 13 arms × 600 updates (A1/A4/A5 3.29 s; A2 ~3.5 s and A3 ~6 s are estimates) | 8.6 | 10.25 |
| Stage A: 13 H-select assays + constrained greedy | 7.25 | **17.5 (GA)** |
| P0' at I\* | 0.4 | 17.9 |
| RL: 700 accepted updates × 44 s | 8.6 | 26.5 |
| RL H-select ×5 | 2.5 | 29.0 |
| SFT-E ×3 + H-select ×3 | 2.35 | 31.35 |
| H-gate for RL\*, SFT-E\* and I\* (G3a) | 1.7 | 33.05 |
| SFT-T ≤2 and LIM-T to T_RL (13,200 s each) + H-select ×3 | 12.5 | 45.55 |
| H-gate for SFT-T\* and LIM-T (G3b) | 1.15 | 46.7 |
| VAL, about 11 new endpoints | 10.8 | **57.5 (G4)** |

- **Early stops (cumulative h):** PI 0.25, GA 17.5, GB0 17.9, G2 26.5, G3a 33.05, G3b 46.7.
- **Beyond G4:** each replicate seed adds about 25–30 h, TEST adds about 2.9 h, and the full conditional total is about 115 h.
- **Expected spend: about 25 h** = 0.25 + 0.9 × (17.25 + 0.25 × 40). P(PI) ≈ 0.9 and P(GA) ≈ 0.25 are both judgement.

## 10. Prior

**p_substantial ≈ 0.02** (range 0.015–0.02; judgement). All three designs and all three judges put the base rate here. The §1 algebra gives no reason to raise it: the β = 0 Dr-GRPO limit is p-weighted CE, which upweights easy images, so P(GA) may well be below 0.25. The study is worth running because its most likely stop, GA at about 17.5 h, publishes a clean negative at SFT prices. That negative would say that no answer-only signal-density RL variant beats CE or constrained CE here in expectation.

## 11. Reporting

- Raw outputs are committed before any interpretation.
- The full Stage-A table: every arm × LR, unconstrained and constrained macro, and validity.
- Both G4 verdicts, plus the three-way decomposition.
- The look ledger.
- H results are native-TRAIN development evidence only.

## 12. Implementation work list (no edits in this PR)

- **`src/vision_rl/vlm_local_training.py`**
  - `trainer_kwargs` (l.444): add the parameters `loss_type` ('grpo' | 'dr_grpo') and `scale_rewards` ('group' | 'none'), replacing the literals at l.507–508 with the defaults unchanged. Refuse β = 0 unless `loss_type` is passed explicitly.
  - `train_local` (l.2624): forward β, `loss_type` and `scale_rewards`, and record them in settings only when they differ from the defaults, so legacy identities stay byte-identical. Extend the mode set at l.2680 with `hn_sft`, `hls_sft` and `cce_sft`. Give `egsft` the options `egsft_scale` ('group' | 'none') and `egsft_band_group`.
  - `expected_gradient_weight` (l.1013) and `egsft_weight` (l.1042): add a no-std branch that enumerates exactly and must equal (G−1)p, plus the band mask. They are consumed at l.1702.
  - `make_reward` (l.330): add a separately named `exact_hn` reward; `exact` is unchanged.
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
- **Decision record:** at activation. Not in this PR.
