# aircraft-rl-from-I-confirmatory-v1: pre-registration (unactivated)

**Status:** UNACTIVATED. Written 2026-10-18, before any dev800 winner of the RL-from-I program exists (critic M7). No RL-from-I endpoint has been named, frozen or scored on native VAL. The machine-readable twin is `docs/aircraft-rl-from-I-confirmatory-v1.json`; where they disagree the JSON wins. The inherited `substantial_gain_gates` block of `docs/aircraft-rl-protocol-freshpool-v1.json` is a **floor**: every inherited threshold must hold, and every stricter rule here (the Bonferroni 98.33% interval, replicate and intake rules) applies on top of it. No inherited value relaxes a stricter rule. Amendments are append-only. The gate-bearing sections (§2–§7, §9, §10) freeze at the first dev800 or h_gate result of any post-GATE-0 RL-from-I arm, or at activation if that comes first. After that freeze, only disclosure or pin additions, or a device change after a failed canary, are allowed. Nothing may be amended after activation. Nothing here came from running a model; every cost is marked **unverified**.

**What it is for.** RL started from I (controlled400 `sft-1e5-time`, weights `8f27d0ca…`, native VAL 0.7539) is being screened on dev800 under CORRECTIONS.md (adopted in D35). This plan fixes, before any screen result, what may be confirmed on native VAL and then sealed TEST, against which comparators, with which gates. D27 withdrew the only previous confirmatory plan, and D31 keeps VAL untouched until a recipe is pre-registered; this is that pre-registration.

## 1. Pins

| Pin | Path | SHA256 |
|---|---|---|
| Protocol revision (gates inherited) | `docs/aircraft-rl-protocol-freshpool-v1.json` | `3d735a32…68b9` |
| CORRECTIONS (GATE 0 and arm rules) | `docs/aircraft-rl-lessons-audit-2026-10-18/CORRECTIONS.md` | `cc392fa0…1dcf` |
| Audit C1–C19 | `docs/aircraft-rl-lessons-audit-2026-10-18/audit.md` | `bfc3dd1b…ab29` |
| Critic M1–M12 | `docs/aircraft-rl-lessons-audit-2026-10-18/critic.md` | `ed9c3845…bd3` |
| Evaluator bundle at 85f0865 | `reports/2026-10-17-controlled400-native-val/plan.json` | `cf871cfb…5896` |
| Endpoint declaration (I weights) | `docs/aircraft-native-val-activation-v1/activated-declaration.json` | `6a1faaaf…faefe` |
| VAL look that selected I | `reports/2026-10-17-controlled400-native-val/comparison.json` | `a49482bc…f2f5` |
| Splits (dev800, s_train, all_train, scans) | `docs/aircraft-exploratory-sweep-v2/splits.json` (PR #29, not yet on main) | `761d79f1…f4ed` |

Inherited unchanged (canonical-JSON SHA256): `substantial_gain_gates` `f58da53c…`, `strong_baseline` `818a3c1f…`. The three audit files are byte copies of `runs/rl-lessons-audit-2026-10-18/` (gitignored), committed so the adopted rules are in history.

## 2. Candidate intake: at most K = 3

**Who may enter.** Only an RL arm from the RL-from-I dev800 screen whose whole trajectory ran after GATE 0 was committed, and which has all of:

1. **GATE 0 go.** A committed GATE 0 receipt with all four measurements (P0-I, SFT-from-I curve on s_train and all_train at LR {5e-6, 1e-5, 2e-5} excluding train400, dev800 calibration on the 7 endpoints that have VAL scores, placebo pairs plus I's seed noise). P0-I pass@8 headroom on I-greedy-wrong images ≥ 0.05 of images (or GATE 0's own committed number). dev800 calibration passes, including the negative sign of RL400 minus same-LR SFT.
2. **Arm rules.** Every CORRECTIONS arm rule that applies is evidenced in the arm's record: fp32 GRPO with parity `within_tolerance` true (or a committed bf16 waiver), one device class start to end, controls at the same LR count with best-of-k vs best-of-k, full label access and per-checkpoint time matching, pool-scoring seconds charged to RL, the 477 near-duplicates and train400 out of the RL pool and its controls, the rebuilt confusion map only, equal co-tenancy, no refuted families, and a passed CUDA fp32 E0 (C19) before any CUDA resume. Missing evidence makes the arm inadmissible; there are no waivers at intake.
3. **Placebo-adjusted promotion bar.** P_max is the largest absolute placebo statistic (SFT-from-I vs SFT-from-I at another seed, same best-of-k code, same k, same stops). B = max(0.03, P_max). The arm's dev800 delta, RL best-of-k minus max(paired control best-of-k, I, best tuned same-label SFT), must be **strictly greater than B**.
4. **h_gate sign.** Exactly one ledgered h_gate_final assay; RL minus the best comparator is > 0 (sign only, C18).

**Why K = 3.**
- Bonferroni at K = 3 moves the interval from 95% to 98.33% two-sided. At the measured paired SE of about 0.54 pp on full VAL (RL400 vs strongest SFT CI [−6.63, −4.52]), the lower bound drops by about 0.23 pp, which is negligible next to the 5 pp gate that binds.
- K = 1 would bet the single look on a dev800 rank whose SE is about 1.6 pp and whose agreement with VAL is checked on only 7 endpoints.
- K > 3 is unaffordable: each candidate brings 21 full-VAL endpoints (3 RL + 18 controls over three seeds) and two seeds of replicate training. Three candidates already make about 77 roster endpoints (**unverified** cost).

**Ranking and freeze.** If more than 3 are admissible, take the top 3 by (dev800 delta − B), then lower active seconds, then arm id. The intake receipt (exclusive create, committed) records each candidate's `adapter_model.safetensors` SHA256, config and endpoint-receipt SHA256, step, init, LR, recipe hash, label set, device class, active seconds, dev800 and h_gate numbers. It is committed before any replicate training and before any VAL prediction. Intake then closes for good. If nothing is admissible after the fixed rung schedule, this plan is WITHDRAWN_NOT_RUN and VAL stays untouched.

## 3. Comparator pool

The strongest SFT is the **maximum native-VAL accuracy over every endpoint below, all scored in the same look on the same evaluator and device class**. A max on VAL inflates the comparator; that bias is against RL and is accepted.

**Seed 42 (S₄₂):**
- I, re-scored in this look (0.7539 is reported but not used);
- every candidate's paired SFT controls at LR {5e-6, 1e-5, 2e-5}, both E and T endpoints at the candidate's frozen checkpoint;
- sftall\* and siblings: the GATE 0 SFT-from-I curve endpoints on every label set any candidate used, the dev800-best stop at each LR. sftall\* is the dev800-best on all_train;
- for a candidate started from sftall\* (M2 family): sftall\* itself and its continued-SFT controls at all three LRs, E and T;
- the five other controlled400 SFT endpoints, **re-predicted in this look** like every member. Their old scores are not used, because sft-5e5-time (0.7534) is within 0.0005 of I and could bind on another device.

**Seed s ∈ {314, 2718} (S_s):** I (same prediction file); every candidate's controls retrained at seed s; the sftall\* recipe retrained from I at seed s; for the sftall\* family, its continued-SFT controls at seed s from the same frozen seed-42 sftall\*.

S₃₁₄ and S₂₇₁₈ omit the sftall siblings and the controlled400 endpoints, so they can be weaker than S₄₂ (disclosed). They still contain I.

The pool is global: each candidate faces the maximum over all candidates' controls, so every candidate has the same bar, and the bar is at least as strong as a per-candidate one.

## 4. VAL gates (per candidate k, all must hold)

I is always in the pool. If I re-predicts at 0.7539 (MPS canary), then S ≥ 0.7539 and the binding point gate is **acc(RL_k) ≥ S₄₂ + 0.05** (≥ 0.8039). On a validated equivalent, I may move by up to 0.005; S can then fall below 0.75 and V1 binds instead. Every gate is always evaluated.

| Gate | Rule |
|---|---|
| V1 | acc(RL_k) ≥ 0.80 |
| V2 | acc(RL_k) − acc(I) ≥ 0.05 (sftall\* family: also − acc(sftall\*) ≥ 0.05) |
| V3 | acc(RL_k) − acc(S₄₂) ≥ 0.05 |
| V4 | (acc(RL_k) − S₄₂) / (1 − S₄₂) ≥ 0.20 |
| V5 | lower bound of the two-sided **(1 − 0.05/K) = 98.333%** paired class-stratified bootstrap interval (20,000 draws, seed 2718; 0.8333th percentile) of RL_k − S₄₂ > 0. K = 3 even if fewer were admitted. |
| V6 | replicates: every d_k,s = acc(RL_k,s) − acc(S_s) > 0 for s ∈ {42, 314, 2718}, and mean ≥ 0.05 |

**Multiplicity.** Within a candidate, V1–V6 is an intersection-union test (no adjustment needed). Across candidates the claim is "at least one passes", so V5 is Bonferroni-adjusted over K, which holds under any dependence (candidates share I and the comparator). S₄₂ is treated as fixed in the bootstrap although it is VAL-selected (disclosed).

**Which candidate goes to TEST.** The passing candidate with the largest mean replicate delta; ties by lower seed-42 active seconds, then id. Only its seed-42 endpoint goes to TEST.

## 5. Replicates at seeds 314 and 2718

- **Same init.** Replicates start from the same frozen I (or, for the sftall\* family, the same frozen seed-42 sftall\*). I is never retrained: it exists only at seed 42, so the claim is conditional on this I, and its seed variance is disclosed as unreplicated.
- **RL at seed s.** The candidate's frozen recipe (init, LR, G, β, T, pool and screen settings, checkpoint step, device class), with trainer seed s and the seed-s scan manifest of the same label set (`s_train_scan_seed{s}` or `all_train_scan_seed{s}`).
- **SFT controls at seed s.** From the same init, on that seed's RL label access (every scanned or scored row, scan order), time-matched to that seed's RL checkpoint, at all three control LRs, E and T. Plus the sftall\* recipe retrained from I at seed s.
- **Nothing is re-selected per seed.** No LR, stop or checkpoint search.
- **Before the look.** All replicates are trained for every admitted candidate and frozen in the roster before the single VAL look, so no seed can be hidden or chosen after VAL.
- **Caps.** 1.25 × the seed-42 active seconds. A failed replicate is re-run once from scratch and its first attempt published; a second failure counts as d ≤ 0. A replicate resumed on CUDA needs a passed CUDA fp32 E0 (C19) on that GPU model.

## 6. One native-VAL look, with a ledger

- **The look** is the single scoring event that joins roster predictions to VAL labels. Prediction runs may take many invocations, but none computes or prints any accuracy. The scorer runs once, after every roster prediction file is complete and committed, and its result is committed before interpretation.
- **Roster.** K seed-42 candidates, 2K replicate RL endpoints, and all of S₄₂ (including I and the five other controlled400 SFT), S₃₁₄ and S₂₇₁₈. All are frozen by SHA256 in a committed roster receipt. W is not in the roster. **Hard cap: 80 endpoints.** A roster above 80 is refused, not trimmed.
- **No second look.** Nothing is scored twice, nothing is added after scoring, and no gate is recomputed with a changed roster, threshold, seed or device. An incomplete look passes nothing.
- **Ledger.** `reports/aircraft-rl-from-I-confirmatory-v1/val-look-ledger.jsonl`, append-only, one line per VAL-touching event. It shares the h_gate ledger by reference (C18).
- **Prior VAL looks, disclosed.**
  - **2026-10-17 controlled400 native VAL** (W, RL400, six SFT): this look **selected I** (a\* = 0.7539) and fixed D_req.
  - 27B MLX discovery runs 1 and 2 (100 images each) and run-2 full invocations 1–3 before the pause (D22, D23, D31).
  - 3B CI discovery (100 images, D34) and its full VAL, run 39144098450 (queued at 16:14Z when this was written).
  - Any 27B v2 run (PR #31).
  - No RL-from-I, GATE 0 or sftall endpoint has touched VAL.
- **What VAL's selection of I means.** I's VAL score carries upward selection noise. That makes RL − I harder, but RL inherits I's weights, so the absolute 0.80 gate may be helped by the same noise. Only sealed TEST is free of it.

## 7. Evaluator and device

- **Identity.** The frozen evaluator at 85f0865 as bundled by `plan.json` `cf871cfb…`: 41 source files re-hashed before import, prompt `859dd366…`, greedy, 256 tokens, EOS-required canonical parse, equal-class metric with failures in the denominator.
- **Default device class.** MPS fp32, as on 2026-10-17.
- **One device class.** Every roster member, the canary and every TEST actor uses the same class. Mixed classes are refused.
- **Validated equivalent.** One other fp32 class (one CPU model on one UW host type, or one CUDA GPU model) may replace MPS for the whole look. It must first re-predict all VAL for W, RL400 and I (already looked at, so no new information). It passes only if each endpoint's per-image parsed prediction agrees with its 2026-10-17 journal on ≥ 0.99 of images, |Δ macro accuracy| ≤ 0.005, and |Δ(RL400 − I)| ≤ 0.005. bf16 and fp16 are not eligible (bf16 same-prediction 0.9575, C2).
- **Canary (activation prerequisite).** Before activation, I is re-predicted on the chosen class. This spends no new VAL information, because I was already looked at. On MPS fp32 it must match the I journal on all 3,333 images. On a failure, the plan is not activated on that class: another class is chosen by appended amendment and the canary is re-run. The passing I prediction file becomes the I roster member.

## 8. Disclosures

- **477 near-duplicates.** all_train keeps 477 pHash near-duplicates of VAL/TEST images. RL pools and their paired controls exclude them; sftall\* may include them, which only strengthens the comparator. D17 found the pHash threshold has a high false-positive rate. A near-duplicate-free VAL sensitivity is reported, descriptive only.
- **Not 4-shot.** I saw 4 images per class. Candidates and sftall\* read s_train (1,424 images, about 14 per class) or all_train (2,329 images, **about 23 per class**, containing train400). Every claim is labelled with the largest label access in the comparison, plus the 1,005 TRAIN holdout images whose labels were read for selection only.
- **Confusion map / dev600.** The old `p0-confusion-map.json` drew 15 of its 64 source images from dev600 ⊂ dev800 (C13). Only the map rebuilt from s_train-only P0-I rollouts is admissible.
- **VAL selected I** (§6). **I is unreplicated** (§5). **Candidates are dev800-selected**, so their VAL scores are clean of that selection, but the comparator max on VAL is not.
- **No overfitting rationale** (M8): SFT on train400 plateaued at 0.752–0.754; it did not overfit.

## 9. Prerequisite: a full-VAL named baseline

Before activation, at least one full-VAL named baseline is complete and merged on main:
- the **Qwen2.5-VL-3B** CI run 39144098450 (plan `8ae93e99…`), or
- the **Qwen3.8-27B v2** HF-CPU plan (PR #31) after its activation.

The activated copy names which ones completed. Only those enter TEST.

## 10. Sealed TEST: once, only after VAL passes

- **Opened only if** at least one candidate passes V1–V6 and the VAL result is committed on main. One use, all 3,333 images.
- **Role allowlist.** `rl_star_seed42` (RL\*\*), `strongest_sft` (argmax of S₄₂), `init_I`, and `named_baseline_3b_ci` and/or `named_baseline_27b_v2` if listed in the activated copy. If strongest SFT is I, one run serves both roles.
- **Never on TEST.** W, replicate-seed endpoints, other candidates, other SFT endpoints, RL400, GATE 0 and placebo endpoints, and anything chosen after a TEST output exists.
- **T1 (primary).** RL\*\* ≥ 0.80; − I ≥ 0.05; − strongest SFT ≥ 0.05; RER ≥ 0.20; 95% paired lower bound > 0. No K adjustment: one candidate reaches TEST, selected on VAL.
- **T2 (only if T1 passes).** Fixed order: 3B, then 27B v2. Each requires baseline valid EOS ≥ 0.95 (else interface failure, no claim), Δ ≥ 0.10, RER ≥ 0.25, and lower bound > 0.05. A failure stops the sequence.
- **Order of families.** VAL → T1 → T2(3B) → T2(27B). Each is tested only if every earlier one passed; nothing is re-tested.
- **Seal mechanics.**
  - The envelope `reports/aircraft-rl-from-I-confirmatory-v1/test-envelope.json` is created by exclusive create, used once, and committed by the authorizer. One look record per role goes in `test-looks/`.
  - Identities are derived from the committed receipts; nobody types them.
  - The guard binds cooperating runners only. Git-history checks need one full clone. FGVC-Aircraft TEST is public, so the seal is procedure, not secrecy.

## 11. Outcomes

| Outcome | VAL | TEST | Action |
|---|---|---|---|
| GATE 0 no-go, or nothing admissible | no | no | WITHDRAWN_NOT_RUN |
| Prerequisite unmet | no | no | not activated |
| Canary or equivalence fails (before activation) | no | no | not activated on that class; choose another by amendment and re-run the canary |
| Look incomplete | partial | no | nothing passes; publish what completed |
| No candidate passes | yes | no | claim FAILED; publish every gate, the pool and the replicates |
| A candidate passes | yes | once | envelope for the allowlisted roles; score once |
| T1 fails | | no further | final claim FAILED; no second look |
| T1 passes, T2 fails | | | RL-from-I beats strongest SFT at the stated label access; no frontier claim |
| T1 and T2 pass | | | claim scoped to baseline, quantization, profile, dataset, label access and split |

## 12. Refusals (each needs a fire test in the validator)

`plan_not_activated`, `pin_mismatch`, `inherited_block_changed`, `intake_not_frozen`, `roster_not_frozen`, `device_class_mixed`, `evaluator_not_reproduced`, `second_val_look`, `prerequisite_missing`, `test_before_val_pass`, `test_role_not_allowlisted`, `test_envelope_reused`, `roster_too_large`, `gate_block_amended_late`. Conditions are in the JSON.

## 13. Compute (**unverified**)

About 77 full-VAL roster endpoints at K = 3, which is an upper bound; the hard cap is 80. Replicate arms are capped at 1.25 × their seed-42 active seconds, with one re-run each. No other compute is authorized. Each is about 0.94 MPS-h (freshpool study §10) or about 4.7 CPU-lane-h (68 min per 800 images on iccad7). Heavy prediction runs on UW hosts on the one validated class (D31).

## 14. Activation prerequisites

1. A GATE 0 receipt with a go is committed on main.
2. `splits.json` is on main with SHA256 `761d79f1…`.
3. At least one full-VAL named baseline is committed on main.
4. A validator implements every refusal id, with fire tests, and has been reviewed and merged. Its SHA256 is added to the pins by amendment.
5. The intake receipt is committed, with at most 3 candidates frozen by SHA256.
6. The device class is chosen. Its equivalence validation (if it is not MPS fp32) and the I canary on that class are committed and passed.
7. Root commits `docs/aircraft-rl-from-I-confirmatory-v1-activated.json`. Its `activation.activates` pins this file's SHA256, and only `status`, `activated` and `activation` may differ.
