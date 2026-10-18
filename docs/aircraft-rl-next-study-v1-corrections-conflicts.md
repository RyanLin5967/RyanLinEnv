# Signal-limit v1 vs the binding RL-from-I CORRECTIONS (D35)

`docs/aircraft-rl-lessons-audit-2026-10-18/CORRECTIONS.md` (adopted in D35, "binding for every RL
lane") postdates the signal-limit pre-registration (`docs/aircraft-rl-next-study-v1.json`, amendment
1) and its production package (PR #34). This file lists, item by item, where the pre-registered
design agrees with CORRECTIONS and where it conflicts. The design is unchanged: amendments are
root's. The activated copies name this file, so **root's activation line must say how each
conflict is resolved** (amend before any output, or accept as a disclosed deviation).

Status words: **CONFLICT** (the design does what CORRECTIONS forbids, or omits what it requires),
**OK** (consistent, with the check named), **N/A**.

| # | CORRECTIONS item | Signal-limit design | Status |
|---|---|---|---|
| 1 | GATE 0: no GRPO-I arm before P0-I at I on ~200 s_train + ~200 all_train images, T 1.0 and 1.3, G8, fp32, p(gold) summed over tokenisations, pass@8 headroom (C1, C14) | P0-I scores 400 S_b candidates at I, G4 rollouts at T 1.0 only, canonical p, predictions for G in {4, 8, 16}; advisory, never a stop | **CONFLICT** |
| 2 | GATE 0: SFT-from-I curve on the same labels, LR {5e-6, 1e-5, 2e-5}, assayed on dev800; comparator max(I, best tuned SFT) (M1, M8) | No SFT-from-I curve. Stage B's comparators are SFT-E/SFT-T/LIM-T from I\* on H-select/H-gate | **CONFLICT** |
| 3 | GATE 0: dev800 calibration of the 7 VAL-scored endpoints (M4) | Selection and gating use H-select / H-gate (200 each); no dev800 | **CONFLICT** |
| 4 | GATE 0: placebo pairs (SFT vs SFT at another seed) through the promotion code (M5, C11) | None | **CONFLICT** |
| 5 | Precision: GRPO only in fp32; one device class per trajectory (C2, M6, M9) | FP32 everywhere; the plan runs every stage on one CPU class (Ryzen 9 5900X). bf16 never used, so no bf16 parity replicate is needed | **OK** |
| 6 | LR: GRPO-I {1e-5, 2e-5, 5e-5} + one 3e-6 guard; no 1e-6; Dr-GRPO also 3e-5 (C3, C17) | One grid {1e-5, 5e-5, 1e-4} for every Stage A arm, RL (Dr-GRPO), SFT-E and LIM-T | **CONFLICT** (1e-4 present; 2e-5, 3e-5 and the 3e-6 guard absent) |
| 7 | β fixed at 0 (C12) | β = 0 everywhere | **OK** |
| 8 | Dynamic sampling, max draws ≥ 8, discarded generation charged (C5) | Screened sampler only, no dynamic sampling | **CONFLICT** |
| 9 | Screened target = floor(0.8 × a_I × \|S\|) or scan all_train (C6) | Fixed N = 300 accepted, with refusal a_floor = 375/\|S_b\| = 0.380 | **CONFLICT** |
| 10 | Multi-image update family, P images per step, skip all-zero-advantage steps, log L2 delta (M3) | One prompt per update only | **CONFLICT** |
| 11 | Controls: same LR count and values as RL ({5e-6, 1e-5, 2e-5}) | Three LRs, but {1e-5, 5e-5, 1e-4}; SFT-T only for LRs within 2 pp | **CONFLICT** (values) |
| 12 | Controls: best-of-k vs best-of-k with the same k | RL\* is best of 2 continuers at 300; SFT-E\* best of 3; SFT-T\* best of ≤ 3 | **CONFLICT** (k differs) |
| 13 | Controls train on ALL scanned rows in scan order (the label access the screen read) | SFT-T and LIM-T: RL\*'s full scanned S_b sequence from row 0; SFT-E: accepted only (an extra control) | **OK** |
| 14 | Controls time-matched at EACH RL checkpoint; pool-scoring seconds charged to RL | Time-matched once, at RL\*'s final T_RL; scorer seconds are inside RL's committed active time | **CONFLICT** (each checkpoint) / OK (charging) |
| 15 | Dropped families: hierarchical partial-credit reward, CoT (C7) | No partial-credit reward and no CoT. A4 is hierarchical label SMOOTHING (supervised), a reported control that cannot pass Gate A, and it is the most expensive arm (669 s/update, ~111 h) | **OK** for the rule; A4 flagged |
| 16 | Pools: p(gold) summed over tokenisations; hard = greedy-wrong ∩ p ≥ 0.02; report p≈0 classes (C9, C15) | Screen and A2's p are canonical-only (the trie is used only in assays and A5) | **CONFLICT** |
| 17 | Exclude the 477 VAL/TEST pHash near-duplicates and train400 from RL pools and their controls (C15) | S_a, S_b, H-select and H-gate hold **0** of the 477 (prepare.py refuses otherwise; the 477 are the freshpool G0 audit's VAL/TEST partners, which S never contained). Stage A and RL never train on train400 rows | **OK** (checked) |
| 18 | Confusion map rebuilt from P0-I rollouts on s_train only; 15 of the old map's 64 source images are in dev600 (C13) | A3 uses the old freshpool P0 map (64 S images, disclosed in study §4). dev600 is not a holdout of this study (H-select/H-gate have no overlap) | **CONFLICT** with the rule; no leak into this study's holdouts |
| 19 | Start point: RL from sftall\* vs continued SFT from sftall\* (M2) | Stage B starts from I\* (best of A1\*–A3\* on H-select) | **CONFLICT** (family absent) |
| 20 | Co-tenancy recorded per stage; refuse a time comparison under different co-tenancy (C16) | Host CPU load is checked at launch (`min_idle_threads`), the hardware is journalled, later contention is not recorded or compared | **CONFLICT** |
| 21 | CUDA fp32 E0 before quoting a continuation (C19) | CPU, not CUDA. Stage 0 proves E0 on the chosen device (smoke: G8, 2 steps, byte-identical on CPU) | **OK** (N/A for CUDA) |
| 22 | Stopping: fixed rung schedule, no early stop on success; dev800 +3 pp screening only; confirmatory bar +5 pp on native VAL; hard compute cap in the plan (C11) | Fixed schedule (100 → 300); stops on failure only; native-VAL margins unchanged (+0.05); per-arm 1.25× caps; chain bound of 10 days | **OK** |
| 23 | h_gate gives sign only; one shared look ledger; confirmation by a separate single-look native-VAL study after a full-VAL named baseline (C18, M7, M12) | G3a/G3b require margins of +0.05 on H-gate; a look ledger exists but is this study's own; G4 is native VAL inside this study | **CONFLICT** |

Checked on the committed files: rows 17 (`prepare.py` g0 check against
`runs/aircraft-freshpool-g0-v1/manifests/phash_audit.json`, 477 = its VAL+TEST partners) and 5
(`plan.json` `uw.device` cpu, `--dtype float32` in every template). Every other row is read from the
study JSON and CORRECTIONS.md text.
