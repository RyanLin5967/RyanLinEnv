# aircraft-freshpool-screened-grpo-v1: pre-registration draft

**Status:** this is an unactivated draft. It needs root review, a protocol revision (§2) and an activated copy that root freezes after the controlled400 native-VAL branch (§9) resolves. Amendments are append-only. Nothing in this document came from running a model. Every effect size and cost given here is either cited or marked **unverified**. The machine-readable twin is `study.json`, which carries the same content, and where the two disagree `study.json` wins.

**Lineage.** The base design is `signal: fresh-pool-screened-grpo-v1`, which the three judges scored highest on p(substantial): 0.08, 0.06 and 0.05. Every fatal flaw the judges named is fixed in §13. Ideas grafted from `rigor` are the replication Monte Carlo, the required-delta branch, the SFT-T rule, the active-time caps and the VAL-look ledger. From `pivot`, the EG-SFT mechanism control is grafted as a separate verdict and not as a redefinition of the comparator.

---

## 0. Measured facts relied on (cited, not re-measured)

| Fact | Value | Source |
|---|---|---|
| Cost per GRPO G4 update / SFT update / greedy image | 20.29 s / 5.64 s / 0.966 s | `runs/rl-strategy-v2-2026-10-16/options.json` (measured_inputs), `verification/` |
| Cost of a G4 seed over 32 images | ≈122 s (3.81 s/image) | `runs/aircraft-mechanism32-warm-v1/gate-result.json` via `runs/aircraft-train-screen-review-v1/review.md` |
| RL400 groups | 400 total: 193 all-correct, 70 all-wrong, 137 mixed (65.75% had zero advantage) | `runs/rl-strategy-v2-2026-10-16/measurements.json` |
| RL400 optimiser | every useful step was clipped (median pre-clip norm 3.75); KL window mean ≤0.0035; adapter L2 0.207; TRAIN32 went 26→27 greedy, in-sample (sign test p≈0.75) | `runs/aircraft-controlled400-diagnosis-v1/report.md` |
| Equal-exposure SFT 1e-5 | 27/32 greedy, 181/256 sampled | the task facts; `runs/aircraft-controlled400-diagnosis-v1/report.md` |
| All-wrong groups and hierarchical credit | 9 of 11 all-wrong groups had a single distinct valid prediction; hierarchical credit added contrast in 1 of 50 groups | `runs/aircraft-hierarchical-opportunity-v1/report.md` |
| Fresh pool | 2,919 images, 28–30 per class, outside train400 and probe16 | `runs/aircraft-train-screen-review-v1/analysis.json` |
| Overlap with the 16-shot plan | 1,390 of the 2,919 pool images are in the unactivated 16-shot plan (seed 2718) | `runs/aircraft-train-screen-review-v1/verification/verification.json` |
| PPO ratio clip | `epsilon=0.2` with `num_iterations=1`. The ratio is 1, so the clip is **inert**. Only `max_grad_norm=1.0` binds | `src/vision_rl/vlm_local_training.py` `trainer_kwargs` |
| BF16 | rejected (sampler-teacher error 0.072) | task facts |
| Literature | Visual-RFT and No-Thinking-RL report RL>SFT by 6.9–14.3 pp, but with full fine-tuning, untuned SFT and (for No-Thinking-RL) about 40% of labels offered as options. **The gap does not transfer.** | `runs/rl-strategy-v2-2026-10-16/options.json` |
| Expected GRPO update with exact-match reward | in expectation, a per-image scalar w_G(p) times the SFT gradient, ignoring KL, token-length normalisation and std details | algebra (`pivot` design; judges 2 and 3 confirmed it) |

## 1. Hypothesis and claim scope

- **H1 (primary).** On native VAL, GRPO with just-in-time gold-probability screening on never-trained native-TRAIN images beats the strongest budget-matched, LR-tuned SFT by the unchanged `substantial_rl` margins.
- **H2 (mechanism, reported separately).** Any such gain survives EG-SFT, the deterministic expected-gradient reweighting of SFT. If it does, the gain is attributable to something other than per-image reweighting.
- **Scope.** The arm's label-access set is train400 ∪ S_scanned(seed), up to about 29 per class. This is **not** a 4-shot claim and is not comparable to the papers. Any rise in absolute accuracy that comes from the extra labels is credited to the data, not to RL. W and every SFT arm get the same label access.

## 2. Protocol revision (required before activation)

Create `docs/aircraft-rl-protocol-freshpool-v1.json` as a new protocol id that revises `docs/aircraft-rl-protocol-candidate-v1.json`. Only the following fields change:

- `data.train_membership` becomes "train400 (warm) ∪ the S candidates scanned by that seed's RL arm. Gold labels are read by the scorer for every scanned candidate, so the full scanned set counts as label access."
- `data.shots_per_class` becomes "4 (warm) + screened stream; ≤29".
- `supervised_controls.strong_answer_only_controls.selection` changes from 400/1,200 presentations to N = 600 accepted updates, with selection on H-select, and every SFT endpoint still enters the full-VAL strongest pool.
- Add `data.train_holdout`, which is H-select and H-gate (§3).
- **The `substantial_gain_gates` block is byte-identical** (≥0.80; +0.05 vs init; +0.05 vs strongest SFT; ≥0.20 relative error reduction; paired 95% LB > 0; replication rule). `strong_baseline` (27B) is also unchanged.
- In the same commit, retire the unactivated 16-shot plan in writing with a decision-record entry. It shares 1,390 pool IDs, and the holdout is clean only while no study trains on that plan.

## 3. Data (manifests emitted and hashed before any output)

- **Pool.** Native TRAIN minus train400 minus the 15 non-train400 probe16 images, which is 2,919 images.
- **pHash audit (G0), run first.** It uses a 64-bit DCT pHash and flags pairs at Hamming ≤ 6. The threshold is pre-registered and is **unverified** as optimal. Any pool image that is a near-duplicate of a VAL, TEST or train400 image is dropped from the pool. Within-pool near-duplicate pairs that straddle H and S are dropped from S. The drops are counted and published. Forced fire: a planted resized and JPEG-recompressed copy must be flagged, and 50 random distinct-class pairs must not be.
- **H-select.** 200 images, 2 per class. Within each class, rank by lowest SHA256 of `[protocol_id,"train-holdout-screen",791,label,id]` and take ranks 1 and 3. It is used **only** to choose LRs, for both RL and SFT.
- **H-gate.** 200 images, 2 per class: ranks 2 and 4 by the same hash. It is used **only** for G3. Choosing on one half and gating on the other removes the winner's-curse risk flagged at `runs/aircraft-train-screen-review-v1/review.md` §3.2.
- **S.** The pool minus H, about 2,519 images before audit drops. The scan order per seed is `random.Random(seed).shuffle(sorted(ids))`. No arm ever trains on H.
- **VAL.** All 3,333 images. It is used only for the final gates and for strongest-SFT selection, as the protocol defines.
- **TEST.** Stays sealed.

## 4. Fixed profile (identical for every arm)

- **Model.** Qwen2-VL-2B at the pinned revision, FP32 MPS, Q/V LoRA r8 / alpha 16, initialised from warm `f05840be…`.
- **Prompt and decoding.** Answer-only prompt with the full 100-label list. Maximum 256 tokens.
- **Optimiser.** Constant AdamW, weight decay 0, max_grad_norm 1.0 (shared with SFT through `trainer_kwargs`). One prompt per update.
- **GRPO.** G4 at T 1.0 / top-p 1 / top-k 0, beta 0.04 with ref = W, exact-EOS canonical reward, `scale_rewards="group"`. The PPO ratio clip is inert and is not presented as a control.
- **Screening (RL only).**
  - Before each update, the next S candidate is scored under the **current** adapter. p is P(exact gold+EOS) from one teacher-forced forward pass, using the sampler's exact logit transforms, including TRL's image-token suppression normalisation.
  - The candidate is accepted if u(p) = 1 − p⁴ − (1−p)⁴ ≥ 0.5, which corresponds to p ∈ [0.159, 0.841].
  - Every candidate is journalled with id, p, u, the accept decision, scorer seconds and the adapter hash.
  - If P0 finds that more than 2% of correct samples used a non-canonical tokenisation, p is summed over the tokenisations observed in P0. This switch is pre-registered and frozen before G2.
- **N.** 600 **accepted** updates, fixed. If the scan exhausts S before N, the arm is incomplete. It is not quoted and the study stops (§8). N therefore does not depend on the data.

## 5. Arms (per seed)

| Arm | Definition | Updates | Data |
|---|---|---|---|
| W | warm adapter. It is the initial policy and the KL reference. Its controlled400 VAL is reused **only** on an exact adapter identity match | 0 | train400 |
| RL(lr) | screened GRPO with lr ∈ {1e-5, 5e-5, 1e-4}, the same grid as SFT. At seed 42 all three run to 150 accepted updates, and the top two by H-select continue to 600 without reset. RL\* is the H-select best **at 600**, which re-checks the LR at the final exposure. Replicates freeze RL\*'s LR | 600 | accepted IDs from S_scanned |
| SFT-E(lr) | gold+EOS SFT with lr ∈ {1e-5, 5e-5, 1e-4}. **Exposure-matched:** the same N = 600 updates on RL\*'s accepted IDs in the same order | 600 | RL\*'s accepted sequence |
| SFT-T(lr) | SFT-E(lr) continued without reset over RL\*'s **full scanned sequence**, accepted and rejected, in scan order and cycling. It stops at RL\*'s committed active seconds T_RL, which include scorer time for rejected and accepted candidates. It runs for every lr within 2.0 pp (H-select sampled) of the best SFT-E. It gives equal time and equal label access, including the p≈0 images RL cannot learn from | to T_RL | S_scanned |
| EG-SFT | SFT loss × stopgrad[w₄(p)/w₄(0.5)], where p comes from the same forward pass and w₄ is enumerated exactly under TRL's group-std advantage. It uses the H-select-best SFT-E lr and RL\*'s accepted sequence. **This is a mechanism control.** It is not in the protocol's strongest-SFT pool (see the G4 verdicts) | 600 | RL\*'s accepted sequence |

**Strongest SFT (protocol verdict).** The maximum full-VAL mean-class accuracy over W, SFT-E×3, SFT-T (all qualifying), every controlled400 SFT endpoint already VAL-evaluated, and the edge-extension LR if branch B4 adds it. Protocol tie rules apply. This set can only make the comparator stronger.

**Tuning symmetry.** RL and SFT share one LR grid, and both are selected on H-select. RL screening costs 450 + 900 accepted updates, about 8.3 h (unverified), against SFT-E's 1,800 SFT updates, about 2.8 h. Each family's tuning cost is reported separately. Every RL step that did not win, and every SFT LR, is published.

## 6. Budget matching

- **Exposure.** SFT-E and EG-SFT see exactly RL\*'s 600 accepted prompt presentations and optimizer updates, in the same order.
- **Active time.** SFT-T continues SFT-E's optimizer trajectory, with no reset, to T_RL. T_RL is checkpoint-bound cumulative committed active time across resumes. It covers scorer forwards for every candidate, generation, reward, reference and backward. Setup, evaluation and I/O are logged separately.
- **Label access.** SFT-T sees every image whose label RL's scorer read. It therefore has at least as much label information as RL.
- **Caps.** Each arm is capped at 1.25 × its forecast active seconds, with no reset on resume. A capped or interrupted arm leaves its comparison incomplete. It is re-run from scratch and never quoted.

## 7. Gates (in order; every failure is kept and published)

| Gate | When | Pass condition (all must hold) | On fail |
|---|---|---|---|
| G0 | before any model run | manifests hashed, pHash audit done with its forced-fire test passing, protocol revision committed, 16-shot plan retired, branch (§9) resolved and published, D_req published | do not launch |
| E0 (engineering) | before any fit | protocol `real_learning_and_resume_gate` on MPS **with the screened sampler**: 4 uninterrupted updates vs 2+2 and 1+3 resumed give identical scan position, journal, sampled tokens, rewards and adapter bytes; reference unchanged; finite nonzero task gradient; sampler-teacher log-prob agreement ≤ 0.001. The scorer's p uses the same transform | engineering fix, then rerun (charged to engineering) |
| G1 / P0 | seed 42, at W | Score the first 400 S candidates in scan order and run G4 rollouts at seeds 42 and 314 on 64 of them. Requires acceptance a ≥ 0.30; predicted mean u among accepted ≥ 0.60; observed useful-mixed fraction among the 64 within ±0.10 of the prediction from p; valid ≥ 0.90; EOS ≥ 0.95; scorer ≤ 2.0 s/candidate; the non-canonical tokenisation share of correct samples is reported | stop: no-go for this recipe |
| G2 (futility) | RL at 150 updates, three LRs | the best RL(lr) on H-select has sampled macro Δ vs W > 0, invalid count ≤ W's + 2, adapter changed, reference unchanged, and rolling useful-mixed fraction ≥ 0.6. **This is labelled a futility stop, not evidence** | stop |
| G3a | RL\* and SFT-E done | on **H-gate** (greedy plus G4 at seeds 42 and 314, 9 completions per image): RL\* − SFT-E\* (the H-select-best SFT-E) ≥ +0.02 sampled macro; RL\* − W one-sided 90% paired LB > 0; validity and EOS held | stop before SFT-T and VAL |
| G3b | SFT-T done | on H-gate: RL\* − SFT-T\* (the H-select-best SFT-T) ≥ +0.02 | stop before VAL |
| G4 (VAL) | seed 42 | protocol `substantial_rl`, **unchanged**: RL\* ≥ 0.80; ≥ +0.05 vs W; ≥ +0.05 vs strongest SFT; relative error reduction ≥ 0.20; paired 95% class-stratified LB > 0. A second, **robust** verdict applies the same gates with EG-SFT added to the pool | substantial claim FAILED; report all endpoints |
| G5 (replication) | seeds 314 and 2718, only if G4 passes | protocol rule, unchanged. Each seed's RL (frozen LR) minus that seed's strongest SFT (SFT-E and SFT-T for **every** LR within 2 pp of the seed-42 best on H-select, plus W) is > 0, and the three-seed mean is ≥ 0.05 | FAILED, and every seed is published |
| G6 (TEST) | once, only if G4 and G5 pass | seed-42 RL\*, strongest SFT and W, evaluated on TEST once. The 27B named-baseline gates follow the protocol's `strong_baseline`, unchanged | — |

**Standing stops.** Any one of the following stops the run:
- the rolling accepted useful-mixed fraction falls below 0.40 over any 100 accepted updates;
- the scan exhausts S before N;
- NaN, or a reference hash change;
- an arm hits its 1.25× cap;
- the disk guard (addendum-5 5 GiB reserve, addendum-6 semantics), the RSS guard or the MPS-driver guard fires;
- a second MPS owner appears.

No threshold changes after any output is seen.

## 8. Power (inputs unverified unless cited)

- **VAL.** The paired SE is about 0.55–0.95 pp for discordance ψ 0.10–0.30 (`rigor` design, derived from the train-screen review's ψ, **unverified** for trained endpoints). The LB > 0 condition therefore does not bind. The binding condition is the required delta **D_req = max(0.05, 0.2(1−a_SFT), 0.80−a_SFT)**. That is 0.05 at a_SFT ≥ 0.75, 0.10 at 0.70 and 0.15 at 0.65.
- **Replication.** The `rigor` design's Monte Carlo of the protocol rule gives a pass probability of 0.35 at a true 5 pp, and 0.82 / 0.69 at a true 7 pp for seed σ 2 / 3 pp. σ is **unverified**. Seeds share the warm init, so σ leaves out warm-start variance, and this is disclosed.
- **H-gate.** With 200 images and 9 completions, the SE of the RL−SFT difference is about 2.0–2.7 pp. That is the `rigor` design's 1.4–1.9 pp at n = 400 scaled by √2, and it is **unverified**. Against the +0.02 threshold, a true null passes with probability ≈ 0.20 and a true 6.5 pp passes with probability ≈ 0.97. G3 is a cost filter, and VAL catches a null that gets through. Because selection happens on H-select, these rates are not inflated by the winner's curse.
- **Expected effect** (unverified, `signal` prior): RL − W +2 to +8 pp; RL − strongest SFT −4 to +4 pp. The probability of passing every gate is **≤ 0.08**, the judges' maximum. The study is worth running because a VAL-gated null against tuned SFT, with EG-SFT alongside, is informative. A pass is not expected.

## 9. Branches on the controlled400 native-VAL result (resolved at G0; no branch alters margins, seeds, TEST rules or the strongest-SFT definition)

Let a\* be the maximum VAL accuracy over controlled400 W and SFT endpoints.

- **B0.** W interface failure (valid EOS < 0.95, or the evaluator refuses) → do not launch.
- **B1.** W VAL outside [0.15, 0.85] → do not launch. The expected group signal is too small.
- **B2.** Publish D_req(a\*). It is a lower bound, because fresh-pool SFT will probably beat a\*.
  - If D_req > 0.10 (a\* < 0.70), use **mechanism mode**: G3a and G3b both require ≥ +0.05 instead of +0.02, which tightens them and does not weaken them. Replicates run only if seed 42 passes every G4 point gate.
  - Otherwise use full mode.
- **B3.** Every controlled400 SFT endpoint enters the strongest-SFT pool. This takes no new VAL looks.
- **B4.** If the best controlled400 answer-SFT LR is 1e-4 (a grid edge), add 2e-4 to **both** the RL and SFT grids, symmetrically (+3.33 h, unverified).
- **B5.** If W's identity matches, reuse its VAL. Otherwise W costs one more VAL endpoint (+0.94 h).
- **B6.** The RL400 VAL result is reported as the stale-pool reference. RL\* − RL400 is **never** reported as a pool effect, because the update count and LR also change.

## 10. Compute plan (MPS hours; all extrapolated from §0 rates, a = 0.5, scorer 0.97 s; **unverified**)

These rates give 22.23 s per accepted RL update and T_RL = 13,338 s. One VAL endpoint costs 0.94 h (7.5 h / 8, task facts). One H-select eval (200 images, greedy plus 1 G4 seed) costs 0.265 h, and one H-gate eval (greedy plus 2 seeds) costs 0.477 h.

| Step | Hours | Cumulative |
|---|---:|---:|
| E0 resume and real-update gate, screened sampler (engineering ledger) | 2.00 | 2.00 |
| P0: score 400, G4 on 64×2, W on H-select and H-gate → G1 | 0.99 | 2.99 |
| RL screen 3 × 150 + H-select ×3 → G2 | 3.58 | 6.56 |
| RL top-2 continued to 600 + H-select ×2 | 6.09 | 12.65 |
| SFT-E ×3 (600) + H-select ×3 | 3.62 | 16.27 |
| H-gate RL\*, SFT-E\* → G3a | 0.96 | 17.22 |
| SFT-T ×2 to T_RL + H-select ×2 + H-gate SFT-T\* → G3b | 6.54 | 23.76 |
| EG-SFT + H-gate | 1.42 | 25.18 |
| VAL ×7 (RL\*, SFT-E×3, SFT-T×2, EG-SFT; W reused) → G4 | 6.56 | **31.74** |
| Each replicate seed (2 SFT LRs qualify / 1 qualifies) | 17.68 / 12.10 | |
| TEST ×3 | 2.81 | |
| **Full conditional total** | **69.9** (54.8 if 1 SFT LR qualifies; +3.33 under B4) | |

**Early stops.** G1 ≈ 3.0 h, G2 ≈ 6.6 h, G3a ≈ 17.2 h, G3b ≈ 23.8 h.

**Resources.**
- One MPS owner, run serially, with no parallel arms.
- Rank-8 LoRA checkpoints are small, but `save_total_limit 3` per arm is kept and pruned under the 5 GiB disk reserve.
- Resume state (branch, done, next action) is written to `runs/aircraft-freshpool-v1/state.json` before each stage.
- No GitHub runner is used for fits, at about 525 s/update. Runners are for CPU contract tests only.

## 11. Reporting

- Commit the raw outputs (journal, predictions, ledgers) before interpreting them.
- Report RL − W and RL − strongest SFT separately. Any gain that SFT shares is an SFT or data gain.
- Report both G4 verdicts, protocol and robust. If EG-SFT ≥ RL\* on VAL, state that there is no evidence RL adds anything beyond reweighting.
- Publish the full candidate journal: p, u, accept decision, seconds and adapter hash. Report the acceptance-rate drift over training.
- Report per-arm KL, adapter L2 and the pre-clip norm distribution, and the clip fraction for RL and SFT.
- Break out per-class Δ for the confusable pairs: Gulfstream IV/V, E-190/E-195 and 737-500/600.
- Report classes with p ≈ 0 as unreachable by RL.
- Keep a VAL-look ledger per contender, across studies.
- A pass on H is native-TRAIN development evidence and nothing more.

## 12. Implementation work list (no edits made by this design; files to change)

1. `scripts/prepare_aircraft_freshpool.py` (new) creates the pool, H-select, H-gate and per-seed S scan manifests with their SHA256s. It also runs the DCT pHash audit with no new dependency (PIL plus numpy) and emits the drop list. Tests go in `tests/test_aircraft_freshpool_manifests.py`, using hand-built fixtures with expected IDs and the planted near-duplicate forced fire.
2. `src/vision_rl/vlm_local_training.py`:
   - (a) `gold_probability(policy, processor, row, spec, profile, device)`, a sequence log-prob of gold+EOS under the sampler's exact transforms, built on the `_gold_nll` collator path but summed and not averaged;
   - (b) `_ScreenedRepeatSampler`, hooked through the existing `_get_train_sampler` override in `_trainer_with_grpo_resume`. It scores lazily with the live policy, yields each accepted index G times, writes the journal, and restores scan position and journal on resume;
   - (c) scorer seconds charged inside `_ActiveStepTimer` / `_ExposureLedger`;
   - (d) an `order="sequential"` option so that SFT replays a given JSONL order without data_seed shuffling, plus a `cycle` option for SFT-T;
   - (e) `mode="egsft"`: per-sample loss weight stopgrad[w₄(p)/w₄(0.5)], with w₄ enumerated exactly over k = 0..4 under TRL's group-std advantage formula and eps;
   - (f) new `train_local` parameters `screen_journal`, `screen_threshold=0.5` and `target_accepted=600`, and a refusal if the scan exhausts S.
3. `tests/test_vlm_local_training.py` gets CPU tests on the tiny probe model:
   - scorer p matches the empirical exact-match frequency over 2,000 samples within 3 SE;
   - a planted all-correct image (p > 0.841) and a planted all-wrong image (p < 0.159) are rejected and a mixed image is accepted;
   - the screened resume is byte-identical;
   - w₄ enumeration agrees with Monte Carlo;
   - the EG-SFT weight carries no gradient;
   - the alternate-tokenisation sum works.

   Expected values come from fixtures, never from the subject.
4. `runs/aircraft-freshpool-launch-v1/phase_worker.py` and `launch.py` are modelled on `runs/aircraft-controlled400-launch-v1/`. They add stage sequencing, the G1–G3b gate checks, the 1.25× caps and resume state, and reuse its `guards/`.
5. `scripts/evaluate_freshpool_gates.py` (new) computes H-select selection, G2–G6 and both G4 verdicts, plus the required delta. `tests/test_evaluate_freshpool_gates.py` checks it against hand-computed fixtures.
6. The native-VAL activation package `runs/aircraft-freshpool-native-val-v1/` is cloned from `runs/aircraft-native-val-activation-v1/`. It refreezes the source at the new trainer commit, registers the endpoint identities and adds a `compare` step that includes controlled400 SFT endpoints.
7. `docs/aircraft-rl-protocol-freshpool-v1.json` (the revision in §2) and a decision-record entry that retires the 16-shot plan.
8. **Not used:** expanded LoRA (open P1 defects D1, D3 and D4 in `runs/aircraft-expanded-lora-v2-integration/claude-review-v1/review.md`), G8, dynamic generate-then-reject, BF16, reasoning prompts, hierarchical credit, and confusable-pair curricula.

## 13. Fatal flaws raised by judges and their fixes

| Flaw | Fix |
|---|---|
| The LR was chosen on H and G3 reused the same H, a winner's curse (all three judges) | H is split 200/200. H-select is used for every LR choice in both families, and H-gate is used only for G3 |
| The RL LR was chosen at 150 updates and never re-checked at the endpoint | the top two RL LRs continue to 600, and RL\* is selected at 600 |
| The replicates' frozen-LR SFT was weaker than the seed-42 SFT | replicate SFT = every LR within 2 pp at both E and T endpoints, per seed |
| N and T_RL depended on acceptance a | N is fixed at 600 accepted. Exhausting the scan is a stop. G1 requires a ≥ 0.30, so the expected scan is ≈ 2,000 of about 2,519 |
| SFT-acc was handicapped by excluding the p≈0 images | SFT-T trains on the full scanned set, and G3b compares RL with SFT-T before VAL |
| The scope change broke `train_membership` | a new protocol id with the gates byte-identical, and extra-label gains credited to the data |
| p was a lower bound because of alternate tokenisations | P0 measures the non-canonical share, with a pre-registered switch to a sum over tokenisations, and a test |
| The ratio clip of 0.2 was presented as a control | stated to be inert; max_grad_norm 1.0 is shared by RL and SFT, and the pre-clip norms are reported |
| RL was under-tuned against SFT (pivot) | the same 3-LR grid for both families, and the B4 edge extension is applied symmetrically |
| EG-SFT redefined the comparator | it appears as a separate robust verdict; the protocol verdict uses the protocol pool |
| An early gate had no causal content | G2 is labelled futility; G3a and G3b compare against SFT |
| The 0.80 gate may be arithmetically out of reach | D_req is published at G0, and the B2 mechanism mode tightens G3 |
| There was no real_learning_and_resume_gate for the new sampler | E0, charged to engineering at 2 h |
