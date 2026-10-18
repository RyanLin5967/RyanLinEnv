# aircraft-freshpool-confirmatory-v1: pre-registration draft (unactivated)

**Status:** WITHDRAWN, NOT RUN (2026-10-18, D27). The freshpool study stopped at G1 (D25), before G4, so under this plan's own `development_stop_before_G4` outcome it is never activated and TEST stays sealed. Original status: unactivated draft. It needs root review and a separately identified activated copy (`docs/aircraft-freshpool-confirmatory-v1-activated.json`) before anything in it runs. The machine-readable twin is `docs/aircraft-freshpool-confirmatory-v1.json`; where the two disagree, the JSON wins, and the activated protocol wins over both. Amendments are append-only. Nothing here came from running a model. Every cost is cited or marked **unverified**.

**What this plan is for.** The freshpool study (`docs/aircraft-freshpool-study-v1.md`) is running its seed-42 development stages under the activated protocol `fgvc-aircraft-freshpool-screened-grpo-v1-activated-2026-10-17`. That activation covers E0 through G3b and EG-SFT only. This plan pre-registers what happens after seed 42 passes G4 on native VAL: the G5 replication seeds and the once-only sealed TEST (G6), so the confirmatory phase can start as soon as G4 passes, with no decision left to make after outputs exist.

**What it does not change.** No margin, seed, N, learning-rate grid, pool rule or TEST rule of the activated protocol. The validator recomputes the canonical SHA256 of `substantial_gain_gates`, `strong_baseline`, `evaluation_and_seal.uncertainty` and `evaluation_and_seal.final_test` and refuses any difference.

## 1. Pinned inputs

| Pin | Path | SHA256 |
|---|---|---|
| Activated protocol | `docs/aircraft-rl-protocol-freshpool-v1-activated.json` | `df8e0867…cddf14` |
| Protocol revision | `docs/aircraft-rl-protocol-freshpool-v1.json` | `3d735a32…68b9` |
| Study | `docs/aircraft-freshpool-study-v1.json` | `c21a7920…514d` |
| Branch receipt | `docs/aircraft-freshpool-branch-receipt-v1.json` | `846716f5…b630` |
| Named-baseline plan | `docs/aircraft-named-baseline-plan-v1-activated.json` | `4c9845f0…132b` |
| Gate evaluator | `scripts/evaluate_freshpool_gates.py` | `9a575a94…354d` |
| Bootstrap module | `src/vision_rl/classification_comparison.py` | `269b2d78…6861` |

Freezing the gate and bootstrap code here is the protocol's `test_freeze_before_access` item ("freeze bootstrap code, random seed and all numerical acceptance margins"). Editing either file makes this plan refuse until a reviewed PR re-pins it, which is the intended effect. `authorize` executes the exact bytes it hashed: it loads the pinned bootstrap module into `sys.modules` before the evaluator's `from vision_rl.classification_comparison import …`, so an editable install pointing at another checkout is never used, and it then refuses (`pin_mismatch`) unless every comparison function the evaluator holds was compiled from the pinned path.

Branch, from the receipt: a\* = 0.7539 (`sft-1e5-time`), D_req = 0.05, **full mode** (G3 margin +0.02), no B4 edge extension, `test_consumed: false`.

## 2. Entry condition

The pinned evaluator, run on the seed-42 evidence document, reports G1, G2, G3a, G3b and G4 evaluated and passed, with `failed_at` null. G4 here means the **protocol** verdict (protocol pool). The robust verdict (pool + EG-SFT) is reported but does not gate entry, because the study makes EG-SFT a mechanism control. Activating this plan is what authorizes replicate training; the current protocol activation does not.

## 3. Replication (G5)

- **Seeds:** 314 then 2718, serially, one MPS owner. Both always run to completion even if 314 already fails, so no seed is hidden. G5 is evaluated once, after both.
- **Per seed:**
  - **RL** at RL\*'s seed-42 learning rate, frozen, for N = 600 accepted updates over that seed's own scan (`s_scan_seed<seed>.jsonl` from the G0 manifests). There is no 150-update screen and no LR search.
  - **SFT-E** at every LR in **Q**, for 600 updates on that seed's RL accepted sequence in the same order.
  - **SFT-T** at every LR in Q, continuing SFT-E without reset to that seed's T_RL over its full scanned sequence.
- **Q** is the seed-42 gate report's `G3a.sft_t_lrs`: every SFT-E LR within 2.0 pp (H-select sampled macro) of the best seed-42 SFT-E. It is frozen from seed 42 and never re-chosen per seed.
- **W** is seedless and shared, and its VAL is reused on exact identity (B5).
- **Not run at replicates:** EG-SFT, other RL learning rates, and H-gate gates.
- **Per-seed strongest SFT:** the maximum full-VAL mean-class accuracy over W, that seed's SFT-E and SFT-T for every LR in Q, using the protocol tie rule.
- **Pass:** protocol `validation_replicates`, unchanged. Each of the three deltas (seed 42 is its G4 protocol delta) is > 0, and their mean is ≥ 0.05.
- **Caps and stops:** the study's standing stops and the 1.25× caps apply, frozen from seed-42 measured forecasts before the first replicate stage. A capped or interrupted arm is re-run from scratch and its first attempt is published. Each endpoint gets at most one VAL look, which is logged in the VAL-look ledger.

## 4. TEST: exactly which models, exactly once

TEST is the full native split of 3,333 images. It is opened at most once, and only on outcome `G4_pass_G5_pass`.

| Role | Identity comes from |
|---|---|
| `rl_star_seed42` | gate report `G4.rl_star` (must be the seed-42, 600-update RL endpoint) |
| `strongest_sft_protocol_pool` | gate report `G4.protocol.strongest_sft` (VAL-selected; it can be W or a controlled400 endpoint) |
| `initial_policy_w` | the evidence's W endpoint, whose adapter must equal the frozen W `f05840be…` |
| `named_baseline_27b` | `mlx-community/Qwen3.8-27B-8bit` @ `815b83c0…`, the activated named-baseline plan's native-instruct direct-answer profile (D23 run 2): `qwen3.8-native-instruct-v1`, prompt `859dd366…`, parser source `f75d20ee…`. The validator checks these against the pinned named-baseline plan, the receipt carries them, and `claim` refuses a baseline look under any other profile. |

**Never on TEST:**
- replicate-seed endpoints;
- EG-SFT;
- any RL endpoint other than RL\*;
- any SFT endpoint other than the protocol-pool strongest;
- RL400;
- anything chosen after a TEST output exists.

If W is itself the strongest SFT, that one identity gets one TEST run and both roles read its prediction file.

Decoding is the protocol's: one greedy completion per image, at most 256 tokens, the frozen pixel profile and parser. The baseline uses its frozen native profile. Every contender's prediction file is complete and committed before any TEST score is computed. The pinned `gate_g6` then scores them once.

## 5. Primary endpoint and test

- **H1-confirmatory.** On native TEST, seed-42 RL\* beats the VAL-selected strongest SFT under the unchanged `substantial_rl` gates.
- **Metric.** Mean of the per-class exact-label accuracies over the 100 classes, on all 3,333 images. Malformed, missing and truncated outputs count as errors.
- **Test.** A paired, class-stratified image bootstrap: within each class, images are resampled with replacement and the same indices are used for both models. It uses 20,000 draws and seed 2718, and reports a two-sided 95% linear-percentile interval. These settings match the protocol's `uncertainty` block, and the validator checks them.
- **Decision.** All five conditions must hold:
  - RL\* ≥ 0.80;
  - RL\* − W ≥ 0.05;
  - RL\* − strongest SFT ≥ 0.05;
  - relative error reduction ≥ 0.20;
  - lower 95% bound > 0.
- **H2-confirmatory (secondary).** This is tested only if H1 passes. RL\* is compared with the named 27B baseline:
  - the baseline's valid EOS-terminated fraction must be ≥ 0.95, or the result is an interface failure and no frontier claim is made;
  - delta ≥ 0.10;
  - relative error reduction ≥ 0.25;
  - lower 95% bound > 0.05.

## 6. Multiplicity

The procedure is a **fixed sequence of conjunctive families**: G4 (VAL, seed 42) → G5 (VAL replication) → G6a (TEST, RL vs strongest SFT) → G6b (TEST, RL vs the 27B baseline).

- Each family requires every one of its conditions to pass. That makes it an intersection-union test, which is level-α without adjustment.
- Each family is tested only if every earlier family passed, so the familywise error over the confirmatory claims stays at the per-test level: one-sided 2.5% for each lower bound.
- Nothing is re-tested, and the order is fixed.

**Descriptive only** (no claim, no α spent, and reported whatever they show):
- RL\* − W on TEST as a standalone claim;
- the robust G4 verdict and the EG-SFT statement;
- micro accuracy;
- per-class results, including the confusable pairs;
- the class-cluster bootstrap;
- EOS, invalid and truncation rates, plus tokens, latency and memory;
- the seed spread.

## 7. What happens on each outcome

| Outcome | TEST | Action |
|---|---|---|
| Stop before G4 | no | Study stops at its gate. This plan is never activated, and a commit marks it WITHDRAWN_NOT_RUN. |
| G4 protocol fail | no | Substantial claim FAILED. No replicates. Publish all VAL endpoints and both verdicts. |
| G4 protocol pass, robust fail | no (decided by G5) | Proceed to G5. Report the robust failure, and if EG-SFT ≥ RL\*, state "no evidence RL adds anything beyond reweighting". |
| G4 pass, G5 fail | no | Replication FAILED. Publish both replicate seeds next to seed 42. |
| **G4 pass, G5 pass** | **yes, once** | Authorize the envelope for the four roles. Run each one once, commit the predictions, then score once. |
| G6a fail | no further | Final claim FAILED. Report all TEST numbers. No second look. |
| G6a pass, baseline interface failure | no further | The protocol claim (RL > tuned SFT on TEST) stands. No frontier claim, no baseline re-run and no prompt change. |
| G6a pass, G6b fail | no further | The protocol claim stands. No frontier claim. |
| G6a pass, G6b pass | no further | The final claim holds, scoped to this baseline, quantization, profile, dataset, label access (train400 + scanned stream) and split. |
| TEST run interrupted | no re-issue | The run resumes only from its append-only journal under the same identity. No image is predicted twice and nothing is read before completion. If it cannot complete, that result and every claim that needs it is INCOMPLETE. |

## 8. How TEST stays sealed

- **Allowlist guard.** `scripts/validate_freshpool_confirmatory.py authorize` refuses unless the activated copy, the pre-TEST evidence and the seed-42 G4 report are committed and unmodified and no tracked file differs from HEAD. It accepts only allowlisted evidence keys (top level `labels`, `rows`, `a_star`, `b4_edge_extension`, `g1`, `endpoints`; rows and endpoint splits `h_select`, `h_gate`, `val`), so any TEST-like key is refused whatever its spelling. It then runs the **pinned** gate evaluator itself and derives the four identities from that report, so nobody types them. `claim` refuses any role, identity or baseline profile that is not in the receipt.
- **Single-use envelope.** The receipt `reports/aircraft-freshpool-confirmatory-v1/test-envelope.json` is created by exclusive create, so a second authorize fails even in a race, and `authorize` commits it before returning (removing it if the commit fails), so an issued envelope is always in history. `claim` accepts only that path, tracked and identical to HEAD, with `plan_sha256` equal to the committed activated copy. Each role's look is `test-looks/<role>.json`, created exclusively and committed by `claim`. A second claim for a role is refused if the look exists or ever existed in history, including commits only the reflog still reaches.
- **Already used.** `validate` refuses if the receipt exists, or if it **ever** existed in git history (all refs and reflogs, so a deleted receipt still counts). It also refuses if the plan's state is not `unused` or its maximum uses is not 1. If git history cannot be read, it refuses rather than passing.
- **Runner obligation.** A TEST runner calls `claim` (the baseline also passes its profile) and stops unless it gets a look record, before it opens anything under `data/fgvc-aircraft-2013b/sealed/`.
- **Blind spots (stated in the script too).**
  - The guard binds cooperating runners only. A process that opens the sealed directory directly is not seen.
  - A shallow or fresh clone's git history is incomplete, and a history rewrite followed by reflog expiry is not seen, so validate, authorize and claim run in one full clone.
  - A receipt hand-written after activation and committed at the receipt path with the activated plan's SHA256 is trusted by `claim`, which does not re-run the evaluator.
  - The script cannot verify which adapter produced a prediction file, and it trusts the file's recorded hash, as the gate evaluator does.
  - The G6-awaiting check in `authorize` is implied by the no-TEST-evidence check. A mutation that removes it survives the tests, which is expected for an equivalent condition.

## 9. Refusals

Every refusal exits 2 and names one of these ids:

| When | Refusal ids |
|---|---|
| Validate | `malformed`, `pin_mismatch`, `inherited_block_changed`, `seeds_or_n_changed`, `branch_mismatch`, `test_not_sealed`, `contender_not_allowlisted`, `envelope_already_used` |
| Authorize | `plan_not_activated`, `evidence_not_committed`, `evidence_contains_test`, `gates_not_passed`, `evidence_branch_mismatch`, `w_identity_mismatch`, `pin_mismatch`, `record_not_committed` |
| Claim | `receipt_not_authentic`, `look_not_allowlisted`, `look_already_taken`, `record_not_committed` |

Input the script cannot parse (evidence that is not an object, a field of the wrong type) is refused as `malformed` with exit 2, and any exception from the evaluator is `gates_not_passed`.

`tests/test_freshpool_confirmatory.py` forces each one to fire against the committed plan, or against a git fixture repository holding the pinned files. It uses the hand-computed gate fixture of `tests/test_evaluate_freshpool_gates.py` with its synthetic TEST rows removed. No real TEST data is read.

## 10. Compute (MPS hours, **unverified**)

- **Each replicate seed:** 17.68 h if two SFT LRs qualify, 12.10 h if one. This is from study §10.
- **Actor TEST (three identities):** 2.81 h, from study §10.
- **27B TEST:** about 5.35 h (19,253 s). This is D22's projection for 3,333 images under the named-baseline plan's `stages.full.gate.projection` formula (3,333 × discovery mean latency, 5.61 s/image, plus per-invocation load for each chunk); TEST has the same image count. It is unverified at full scale.
- **Scheduling:** one MPS owner at a time, and the 27B and the actor never overlap.

## 11. Activation prerequisites

1. The seed-42 G4 report is committed, from the pinned evaluator on pre-TEST evidence, with a G4 protocol pass. `authorize` enforces this (`evidence_not_committed`, `gates_not_passed`).
2. Root commits `docs/aircraft-freshpool-confirmatory-v1-activated.json`. Its `activation.activates` pins this draft's SHA256, and only `status`, `activated` and `activation` may differ from the draft. The validator refuses anything else.
3. `python scripts/validate_freshpool_confirmatory.py validate --plan docs/aircraft-freshpool-confirmatory-v1-activated.json` passes from a full clone.
4. Replicate caps are frozen from the seed-42 measured forecasts.
5. The named-baseline profile used on VAL (D23 run 2) is the TEST profile, with no new choice.
