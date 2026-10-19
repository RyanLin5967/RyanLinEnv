# Lessons audit: are the live RL-from-I plans repeating past mistakes?

Yes. All three plans repeat several measured mistakes. The most serious is that none of them measures I's signal before launch. Below are the corrections, ranked by how much they affect the chance of beating SFT, then the lessons the plans already follow, then what I checked and how.

Plans audited:
- **(a) GPU lane:** `~/wt/RyanLinEnv-exploratory-sweep-v2-gpu.noindex` at 48fce3a.
- **(b) RL-from-I campaign:** the workflow script, plus design branch a5ec3d3 (`docs/aircraft-rl-campaign-I/arms.json`, 22 arms). The pools worktree only has uncommitted `scripts/rl_campaign_pools/`. The launcher branch only has merges.
- **(c) CPU sweep v2 and wave 2:** `docs/aircraft-exploratory-sweep-v2.md` and Amendment 3 on origin/sweep-wave2-integration.noindex.

I ran nothing heavy. Every number I computed myself came from `python3 -I` over committed or fetched raw files, using `/private/tmp/claude-501/-Users-idide-RyanLinEnv/e49d0521-f3cb-4650-ad80-61ea854d38c0/scratchpad/audit/rg.py`. Inferences are marked INF.

## Ranked corrections

### C1. No plan measures I's signal before launching GRPO from I (all plans)

**Problem.** The GPU lane queued 24 GRPO-I arms at 20:24Z (`state.json` queued keys 20–25). The campaign launches "ALL rung-1 arms". The CPU sweep runs grpo-I at rung 1. No plan first measures any of these at I:
- I's p(gold) histogram;
- the G8 acceptance rate at u≥0.5;
- the realised mixed fraction;
- pass@8 headroom: images where I's greedy answer is wrong but at least one of 8 samples is right.

**Evidence.**
- Lessons controlled400-1, freshpool-2/3 and other-domains-12/15.
- At W, which is weaker than I, measured mixed groups in the first 150 G8 steps were 65, 68, 67 and 71 of 150. So 53–57% of updates had zero advantage. Source: `reward_groups.jsonl` of GPU arms 03/10 on ecetesla2, 03 on ecetesla4 and 03 on eceubuntu2.
- In the 600-step arms, the all-correct count per 150-step window rose from 57 to 76 (lr2e-5) and from 63 to 79 (lr5e-5).
- INF: I at 0.775 on dev800 will be worse on all of these.

**Correction.** Run a P0-I first. The pool scorer in `rl_campaign_pools/score_worker.py` already computes p_gold and greedy correctness on all_train; add G8 rollouts at T 1.0 on about 200 images. Pre-register these, then go or no-go:
- **(i)** acceptance a(G8, u≥0.5) on s_train and on all_train;
- **(ii)** the realised mixed fraction, split into recognition-mixed and format-only;
- **(iii)** pass@8 among I-greedy-wrong images.

Hold every GRPO-I arm until P0-I is in. A +5 pp gain needs about 40 of 800 dev images to flip. If pass@8 headroom on greedy-wrong images is below about 5% of all images, no GRPO arm can reach the bar (INF).

### C2. The bf16 sampler–teacher parity gate fails on every 3070 GRPO arm, and the runs continue anyway (a, c)

**Problem.** `local_training.json` for every bf16 3070 GRPO arm records `sampler_teacher_logprob_parity.max_abs_raw_teacher_vs_sampler_logprob_error = 0.02814`, `tolerance 0.02`, `within_tolerance: False`. That covers 03, 10 and 11 on ecetesla2, and 03 and 10 on ecetesla4. The fp32 2080 Ti arm measures 9.6e-7 (True). The runs went ahead anyway. This is lesson early-aircraft-13 repeating: bf16 was rejected at 0.0725 > 0.02.

There is a second problem. I in bf16 matches CPU fp32 on only `same_prediction_fraction 0.9575` (GPU `report.json` `device_control`). So the bf16 starting point is not exactly I.

The exposure is large:
- 12 of the 24 GRPO-I GPU arms are bf16 3070.
- wave2-dapo and wave2-rft are planned on 3070 bf16.
- wave2-cot is planned on 2080 Ti fp16, which has no parity evidence.

**Correction.**
- Make `within_tolerance == False` a refusal in `arm_exec`.
- Run GRPO-I only on fp32. The 2080 Ti does 15.7 s/update.
- If 3070 throughput is still wanted, keep it for SFT controls and dev800 assays, which have no sampler–teacher term. Or keep the bf16 GRPO arms only as labelled exploratory runs whose winners must replicate in fp32.

### C3. The GPU lane's LR grid from I goes down when the evidence says the policy needs to move more (a)

**Problem.** The GRPO-I grid is LR {1e-6, 3e-6, 1e-5} (`gpu_lane.py:128`). The evidence points the other way:
- RL400 at 1e-5 barely moved: KL per 50-step window was ≤0.0035 (controlled400-2).
- On the GPU lane, mean KL per 50-step window rose to 0.06 at lr2e-5 with β 0.04 (arm 11 `trainer_state.json`). The best GRPO-W result is lr2e-5 at 600 steps: 0.7225, +3.75 pp over W's bf16 0.685 (`report.json`).
- 1e-6 is 5× below any LR that has moved this policy (GPU W lr5e-6 at 150 steps: 0.69625).
- Every useful update is clipped. In the GPU arms, the count of steps with pre-clip norm > 1 equals the mixed-group count exactly: 65/65, 68/68, 215/215, 67/67, 185/185, 71/71.

**Correction.**
- Replace the eight 1e-6 arms with LR 2e-5 and 5e-5.
- Keep 1e-5.
- Keep one 3e-6 arm as the forgetting guard.

The design branch's LR set {5e-6, 1e-5, 2e-5, 3e-6} is closer to right.

### C4. SFT controls are weaker than the RL arms in every plan

**(a) GPU lane, `cmd_control`:**
- **Single LR.** Every control runs at LR 5e-6 only (`gpu_lane.py:310`). Meanwhile RL gets 3 LRs × 2 β × 2 screen settings × 3 checkpoints. Controlled400-4 measured that SFT LR matters: 5e-5 beat 1e-5 by 2.9 pp at equal exposure.
- **Label access.** Controls train on `committed_prompt_ids`, the accepted IDs only, even for screened arms. This violates freshpool-22: the control must get every label the screen read.
- **Checkpoint mismatch.** The control has one E endpoint and one T endpoint, matched to RL's final step. The report then compares every RL checkpoint (150/300/600) against that final control (`cmd_report`). This mismatches the checkpoints and gives RL best-of-3.

**(b) Campaign design.** It fixes label access (`@scanned`, prerequisite P2) and time-matches each checkpoint, which is good. But rung-A controls still run at 5e-6 only, and the control LR grid starts only at rung B.

**Correction.**
- Give the SFT-from-I control the same LR count as the RL family at every rung: {5e-6, 1e-5, 2e-5}. Compare RL best-of-k against control best-of-k with the same k.
- Train screened-arm controls on all scanned IDs, in scan order.
- Time-match the control at each RL checkpoint (`--stop-after-active-seconds` set to RL `committed_active_seconds` at that checkpoint).
- Keep `sftall-I-5e-6` (CPU sweep, running on iccad9) as the global bar, as the CPU plan already says ("must also beat the best sftall arm").

### C5. Unscreened GRPO-I runs on the GPU lane have no dynamic sampling (a)

**Problem.** The 12 "-n" GRPO-I arms run up to 1,200 steps on s_train with nothing filtering out groups whose rewards are all equal. At W, 53–69% of updates already had zero advantage (C1). INF: at I this will be worse, while the time-matched SFT spends its seconds on images that all carry signal.

**Correction.** Add `--dynamic-sampling-max-draws ≥ 8`. The flag exists on wave2, and freshpool-14 says to cap rejections at 16 or more. Count the discarded generation seconds in RL's T.

### C6. Screened GPU-I arms will very likely run out of images before 600 accepted (a)

**Problem.** These arms need 600 accepted from 1,424 rows, which is an acceptance floor of 600/1424 = 0.421 with no margin. The predicted G8 acceptance at W is 0.42 (next-study JSON `predicted_acceptance_at_W_by_G.G8`), and INF it is lower at I. When the screen runs out of rows, the trainer raises `LocalVLMError` (`next_accepted`), and Amendment 2 records such an arm as failed (freshpool-5). The 150 and 300 checkpoints survive.

**Correction.** Choose from P0-I. Either set `target_accepted` = floor(0.8 × a_I × |S|), or scan all_train, which gives a floor of 0.32 at 600/2329 before margin.

### C7. Refuted reward and format families are still queued (b, c)

| Plan | Family | Refuting evidence |
|---|---|---|
| Campaign | A02 hierarchical (rank 2), A10 and A13 | controlled400-12: contrast in 1/50 groups; 9/11 exact-zero groups had one distinct prediction. freshpool-15: 1/30 groups. A constant prediction earns 22% manufacturer credit. |
| Wave 2 | wave2-hier | Same as above. |
| Wave 2 | wave2-cot, closedset-cot | early-aircraft-6 and controlled400-19: canonical answers fell from 94.92% to 82.81% and exact answers from 70.70% to 55.47%; only 4 think blocks. |

A02's stated reason ("signal on 737-500/600, E-190/195…") is exactly what was measured not to happen.

**Correction.**
- Drop them, or demote them below every exact-reward arm.
- If any is kept, add a stop rule on the first 50 groups: halt if fewer than 5% of groups gain contrast from partial credit.
- Drop both CoT families.

### C8. Wave 2 still trains from W (c)

**Problem.** Amendment 3 fixes rung 1 as "GRPO-W G 8 at LR {5e-6, 1e-5} per family". The decisions-6 lesson and the GPU report both show that GRPO from W loses to I: −5.25 to −9.25 pp on dev800.

**Correction.** Re-point the wave-2 families that survive C7 (drgrpo, mu2, dapo, rft, closedset) to init I, each with paired SFT-from-I controls. Alternatively freeze wave 2 and let the campaign's design cover these axes from I, which it already does in A03, A06, A09, A16, A17 and A19.

### C9. Pool and screen p(gold) is canonical-only, and the hard pool includes unreachable images (b, a)

**Problems.**
- `score_worker.p_gold` calls `gold_probability` without alternative tokenisations. `arm_exec` never passes `--screen-tokenizations`. Freshpool-8 measured the non-canonical share at 0.0301, above the pre-registered 2% switch.
- The pool `hard` = "I greedy wrong" includes p≈0 images. In RL400, 70/400 groups were all wrong and 4 classes were never solved (controlled400-10).

**Correction.**
- Score p summed over the observed tokenisations, using `data/fgvc-aircraft-2013b/audit/label-trie-qwen2vl2b.json` as the main lineage does for egsft.
- Define hard = greedy-wrong ∩ p ≥ 0.02, where u₈ is about 0.15.
- List the p≈0 classes as unreachable, and report them separately.

### C10. Pool construction reads labels and costs time, but neither is charged to RL (b)

**Problem.** P4 scores all 2,329 all_train rows using gold labels. Then A10 and A18 controls train only on `pool_I_hard`. That breaks freshpool-22 (control label access equals every label read) and controlled400-5 (charge every scorer second to RL).

**Correction.**
- Give A10 and A18 controls all 2,329 scored rows, in the same order.
- Add the pool-scoring seconds, pro rata, to each pool arm's T before time-matching.

### C11. Success thresholds and stopping invite a winner's curse (b, c)

**Problems.**
- The campaign's goal is +3 pp on dev800, and the CPU sweep's candidate bar is +2.0 pp. Both sit below D_req of 5 pp (controlled400-7).
- The Run phase stops "until some arm beats … ≥3 pp". That is optional stopping on a selection set whose SE is about 1.6 pp (CPU plan blind-spot note).
- Only one seed runs until rung C, and the seed-noise floor from I is unmeasured: the CPU noise arms are from W only (other-domains-9/10).

**Correction.**
- Keep +3 pp only as a screening bar, and state in the plan that the confirmatory bar stays +5 pp on native VAL.
- Remove the early stop on success: run the fixed rung schedule.
- Add I-noise arms at rung 1: SFT-I and GRPO-I at seeds 314 and 2718.
- Have the adversary report a best-of-k-adjusted bound.
- Write a hard compute and candidate cap into the workflow (other-domains-22/23). It currently says "all free UW capacity" with no cap.

### C12. The arm budget goes to axes that don't bind (a, c)

**Problem.**
- Half the GRPO-I arms, on both the GPU lane and the CPU sweep, vary β {0, 0.04}. β·KL never bound: 5.88e-5 in RL400 (decisions-3).
- The GPU lane runs an identical 12-configuration grid on two device classes. Device duplication is not seed replication.

**Correction.**
- Fix β=0, or keep a single β=0.04 arm.
- Spend the freed arms on LR ≥ 2e-5 (C3), dynamic sampling (C5), T > 1.0 (C14) and seed replicates.

### C13. Holdout labels leak into the confusion map (b, A05)

**Problem.** `p0-confusion-map.json` lists its observed confusers from freshpool P0 rollouts. I measured that 15 of its 64 source images are in dev600, which is inside dev800. (No overlap with h_select or h_gate_final.)

**Correction.** Rebuild the map from P0-I rollouts on s_train only, or drop those 15 images' contributions. Record the change as a disclosure (freshpool-25).

### C14. Sampling temperature is never explored from I (a, b)

**Problem.** Every GRPO-I arm runs at T 1.0, and the CPU sweep's only other value is 0.7, which lowers exploration. Initial per-token entropy at W was 0.13–0.18 (GPU `trainer_state.json`). INF: I is sharper. Other-domains-14 says temperature must be swept in the pilot.

**Correction.** Include T ∈ {1.0, 1.3} in P0-I's pass@8 and mixed-fraction measurement. Add a T 1.3 arm only if it raises the recognition-mixed fraction without collapsing validity.

### C15. Near-duplicates of VAL and TEST are allowed into the RL pools (b)

**Problem.** all_train keeps 477 pHash near-duplicates of native VAL and TEST (`splits.json` `val_test_near_duplicates_kept_in_all_train`). The pool builder annotates them rather than dropping them. A hard pool can concentrate them, and the eventual native-VAL confirmation would then be inflated (freshpool-6; INF).

**Correction.** Exclude the 477 from RL pools and from their matched controls, or carry them as an explicit disclosure into the confirmatory pre-registration.

### C16. Co-tenancy on the shared 2080 Ti contaminates time matching (a)

**Problem.** Another user's roughly 7 GB job shared the eceubuntu2 2080 Ti, and arms 03@c150 and 04–06 hit out-of-memory errors (RESUME log). An RL arm and a control that runs later under different co-tenancy are not time-matched (controlled400-23).

**Correction.** Record `nvidia-smi` co-tenant processes per stage (`gpu-trace.csv` exists). Refuse a T comparison when RL and control ran under different occupancy, or fall back to matching by updates × the clean s/update.

### C17. Dr-GRPO runs at an LR tuned for classic GRPO (b, A06)

**Problem.** A06 runs Dr-GRPO at LR 1e-5 untuned. Dividing by 256 rescales each sequence by 0.0117–0.0547 compared with classic GRPO, which interacts with the clip (controlled400-16).

**Correction.** Pair A06 at LR {1e-5, 3e-5}.

### C18. h_gate is double-booked and underpowered (b, c)

**Problem.** The CPU sweep reserves h_gate_final for "rung-3 finalists only, once". The campaign also plans to assay its top 3 there. At 200 images the paired SE is about 2.0–2.7 pp (freshpool-21), so h_gate cannot confirm a 3 pp gap.

**Correction.**
- Keep one shared look ledger across studies.
- Use h_gate for sign only.
- Move confirmation to a pre-registered native-VAL study with a single look.

### C19. Exact resume is unproven on CUDA (a)

**Problem.** Early-aircraft-14 proved exact resume on MPS only. The GPU lane has already hit a resume-identity bug (save_steps, fixed in acdab57). In the fetched GPU results I found no CUDA E0 run comparing an uninterrupted 4-update run against a 2+2 resumed one tensor for tensor. I only searched fetched results, so this is unverified.

**Correction.** Run a CUDA fp32 E0 at G8 from I (4 uninterrupted updates against 2+2) before any GRPO-I continuation is quoted.

## Lessons the plans already respect

- RL starts from I, verified by hash in `init_adapter`.
- Every RL arm has a time-matched SFT control on the same host. The campaign design also checks per-checkpoint T and scanned-row label access.
- Native VAL and TEST are never touched. dev800, h_select and h_gate come from TRAIN, pHash partners of holdouts are excluded, and `payload.py` ships images from an allowlist.
- s_train is drawn from the audited freshpool S, so it is disjoint from train400.
- Screen scorer time runs inside the committed active step, so it is charged to RL.
- `reward_groups.jsonl` and the pre-clip `grad_norm` in `trainer_state.json` are logged every step.
- The answer-only format is used for the main arms.
- Device-control W and I assays are run per device class.
- Raw results are committed before interpretation.
- The campaign includes the EG-SFT limit diagnostic (D20) and rung-C seed replication. A11's trie arm requires sampler–teacher parity ≤1e-3 and is scored with the unconstrained evaluator.
- The CPU plan states its own blind spots: a small holdout and shared timing.

## Verified facts behind this report

- **Parity flags:** the five bf16 3070 GRPO-W arms (C2 lists them) all record 0.02814 against a 0.02 tolerance, `within_tolerance: False`. The fp32 2080 Ti arm records 9.6e-7, True.
- **Mixed and clipped counts per GPU arm:**

  | Arm | Steps | Mixed | All-correct | All-wrong | Clipped (= mixed) |
  |---|---|---|---|---|---|
  | lr1e-5, β0, bf16 | 150 | 65 | 64 | 21 | 65 |
  | lr5e-6, bf16 | 150 | 68 | 58 | 24 | 68 |
  | lr1e-5, β0.04, bf16 | 150 | 67 | 61 | 22 | 67 |
  | lr1e-5, fp32 2080 Ti | 150 | 71 | 57 | 22 | 71 |
  | lr2e-5, β0.04 | 600 | 215 | 274 | 111 | 215 |
  | lr5e-5, β0 | 600 | 185 | 294 | 121 | 185 |

- **Confusion-map overlap:** 15 of the 64 P0 source IDs are in dev600, and none are in h_select or h_gate_final.

Nothing was edited in any repo or worktree. One stray empty file I had created in `/private/tmp/claude-501/` was removed.