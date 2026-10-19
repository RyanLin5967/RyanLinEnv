# RL-from-I corrections (root, 2026-10-18 ~22:30Z) — binding for every RL lane
Source: audit.md (C1–C19) and critic.md (M1–M12) in this directory, mined from all prior RL work (lessons.md). Each item cites its evidence there.

## GATE 0: no new GRPO-I arm launches until these four are measured (they decide whether any arm can clear the bar)
- **P0-I (C1, C14).** Run at I on about 200 s_train images and about 200 all_train images, at T 1.0 and T 1.3, G8, fp32. Measure p(gold) summed over tokenisations (label trie), the G8 acceptance a(u≥0.5), the realised mixed fraction (recognition-mixed vs format-only), and pass@8 among I-greedy-wrong images. No-go: pass@8 headroom on greedy-wrong images below about 5% of images.
- **SFT-from-I curve on the same labels (M1, M8).** LR {5e-6, 1e-5, 2e-5} on s_train and on all_train, excluding train400, assayed on dev800 at the RL time stops. The comparator for any RL arm is max(I, the best tuned SFT on the same labels).
- **dev800 calibration (M4).** Assay the 7 endpoints that already have native VAL scores (RL400 + 6 SFT controls) on dev800, same device. Require rank agreement, and require dev800 to reproduce the −1.27 pp sign of RL vs same-LR SFT.
- **Placebo (M5, C11).** 2–3 pairs of SFT-from-I against SFT-from-I at another seed, run through the identical best-of-k/promotion code. The promotion threshold must exceed the placebo maximum. Measure I's seed noise as well.

## Arm design rules (all lanes)
- **Precision (C2, M9).** GRPO only in fp32 (2080 Ti or CPU) unless a 2-seed bf16-vs-fp32 replicate of one GRPO-I config falls within the placebo spread. Make `within_tolerance == False` a refusal. bf16 is fine for SFT controls and assays. Keep one device class per trajectory, start to end (M6).
- **LR (C3, C17).** GRPO-I uses LR {1e-5, 2e-5, 5e-5} plus one 3e-6 forgetting guard. No 1e-6. Dr-GRPO also gets 3e-5.
- **β (C12).** Fix β=0, or keep a single β=0.04 arm. Spend freed arms on LR, dynamic sampling, T, and seeds.
- **Dynamic sampling (C5).** `--dynamic-sampling-max-draws >= 8`, with discarded generation seconds charged to RL time.
- **Screened target (C6).** target_accepted = floor(0.8 × a_I × |S|) from P0-I, or scan all_train.
- **Multi-image updates (M3).** Add a family with P images per optimizer step (e.g. 8×G8). Skip the optimizer step when the whole batch has zero advantage. Log the per-step parameter L2 delta.
- **Controls (C4, C10).** Same LR count as RL at every rung ({5e-6, 1e-5, 2e-5}). Best-of-k vs best-of-k with the same k. Train on ALL scanned or scored rows in scan order, which is the label access the screen or pool read. Time-match at EACH RL checkpoint. Charge pool-scoring seconds to the RL arm.
- **Dropped families (C7).** Hierarchical partial-credit reward and CoT/think formats are refuted. Drop them or put them last with a 50-group contrast stop rule.
- **Pools (C9, C15, M8).** p(gold) summed over tokenisations. hard = greedy-wrong ∩ p ≥ 0.02. Report p≈0 classes as unreachable. Exclude the 477 VAL/TEST pHash near-duplicates and train400 from RL pools and their controls.
- **Confusion map (C13).** Rebuild from P0-I rollouts on s_train only; 15 of the old map's source images are in dev600.
- **Start point (M2).** Add RL from sftall\* (best SFT on the larger labels) against continued SFT from sftall\*. If sftall\* beats I on dev800, this is the primary family.
- **Co-tenancy (C16).** Record co-tenant GPU processes per stage. Refuse a time comparison when RL and its control ran under different co-tenancy.
- **CUDA resume (C19).** Run a CUDA fp32 E0 (4 uninterrupted updates vs 2+2 resumed) before quoting any continuation.
- **Stopping (C11).** Fixed rung schedule, no early stop on success. dev800 +3 pp is only a screening bar; the confirmatory bar stays +5 pp on native VAL. Hard compute/candidate cap written into the plan.
- **h_gate (C18).** One shared look ledger. h_gate gives sign only. Confirmation is a pre-registered native-VAL study with a single look (M7), after one full-VAL named baseline exists (M12).
