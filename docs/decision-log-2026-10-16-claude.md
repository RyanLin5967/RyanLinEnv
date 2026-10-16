# Decision log — Claude session adopting the 2026-10-16 Codex handoff

Append-only. Each entry: decision, evidence (file), status. TRAIN numbers are in-sample
diagnostics, never held-out results.

## D1 — Continue the interrupted controlled400 SFT controls under a new envelope
- Pre-registered in `docs/aircraft-controlled400-continuation-v1.json` (4dd645f) and
  `...-addendum-1.json` (7fb7bbb, zero-call failed attempt 0). Original 12 h deadline stays expired.
- Running; journal `runs/aircraft-controlled400-continuation-v1/guards/journal.jsonl`.
- First endpoint (sft-1e5, 400 equal-exposure updates), TRAIN32 assay: greedy 27/32, sampled
  181/256. For comparison warm 26/32 and 181/256, RL400 27/32 and 187/256. Scored by exact
  `raw_response == label<|im_end|>`, which reproduces the recorded warm/RL numbers.

## D2 — RL400's TRAIN32 gain is in-sample noise
- `runs/aircraft-controlled400-diagnosis-v1/{report.md,evidence.json}`, independently verified
  (`verification/verification.json`, sound with minor corrections).
- All 32 assay images were RL training prompts. Greedy +1 is a single image (CRJ-700). Sampled per
  image: 6 up, 4 down, sign test p = 0.75.
- HANDOFF's "87 recognition-mixed / 100 mixed" is the step-258 cumulative count; the final counts
  are 121 / 137 of 400 (65.75% of updates had zero task advantage).
- Every useful update was clipped (median pre-clip norm 3.75 vs max 1.0); KL never bound
  (beta·KL ≈ 6e-5). Conclusion: the policy barely moved, and TRAIN32 cannot measure generalization.

## D3 — Stop dispatching the free-CPU GitHub first4 path
- `runs/aircraft-ci-first4-38889151068/audit-v1/` (verified): 525 s per GRPO update on 2 vCPU,
  projection 5.84× over the 5760 s rule; zero updates committed.
- The candidate speedups (shared-prompt group compute, vision-embedding cache, 4 threads)
  change pinned code or caps, so they need a re-registered plan, not an operational patch.
  Combined they are 3.9–6.8× (unmeasured), so even the best case barely passes. Not pursued now;
  free runners remain useful for CI and non-model work.

## D4 — Pretrained Harbor dogfood runs locally in Colima `vision-harbor`, 1 RL update
- Audit `runs/aircraft-harbor-dogfood-v1-plan/claude-audit-v1/` (verified): the suspected
  output-size race in `run.py` is real (patch written). A 4-update task cannot fit the 1800 s RL
  wall at the measured CPU rate. The fixture images are in train400, so verifier accuracy is
  contaminated and is never a pass criterion.
- VM grown to 4 CPU / 16 GiB / 20 GiB. Kernel preflight measured memory.max 13958643712, swap
  0/0. Frozen-source image `vision-rl:aircraft-cpu7946-continuation` (sha256:26957de2…) was
  built from the 37 hash-verified 7946 source files; offline preflight adapter source f0af7e2b
  matches the plan.
- Running alongside the MPS continuation adds 2 container CPUs of host contention during the SFT
  active-time controls. The host was already at load average ~33 from other sessions, so wall
  and active-time comparisons carry an uncontrolled host-load caveat either way.

## D5 — Next learning experiment is chosen after native VAL, not on TRAIN32
- `runs/rl-strategy-v2-2026-10-16/recommendation.md` (verified with corrections) proposes RL on
  fresh native-TRAIN images the warm adapter never saw. Its first draft confounded the pool
  change with a doubled budget; any version must match budget to its comparator.
- `runs/aircraft-train-screen-review-v1/` (verified): the 32-image exposed screen is biased toward
  SFT and underpowered. A future screen uses a 400-image hash-selected native-TRAIN holdout
  (2,919-image pool outside train400/probe16); the old assay is reported only as a fit diagnostic.
- Order: (1) finish the controlled400 continuation; (2) run the published eight-endpoint native
  VAL contract (`docs/aircraft-native-validation-plan-v1.json`); (3) pre-register the next RL study
  from the measured held-out gaps.

## D6 — Harbor local dogfood attempt v1 failed on a wrapper defect (preserved)
- Plan `runs/aircraft-harbor-dogfood-local-v1-plan` sha 277bb9b2…, image sha256:402dd239…,
  evidence `runs/aircraft-harbor-dogfood-local-v1/` (execute.log, output/dogfood-local-result.json).
- Measured: the Harbor SFT resource probe loaded pretrained Qwen2-VL-2B in the container (pread
  loader, both shards), committed 1 SFT update in 87.3 s active (112 tensors changed, ΔL2 0.0104),
  peak tree RSS 11,990,003,712 B under the 12 GiB cap, cgroup swap 0, guard exit 0.
- Failure: runtime.py runs the trainer via runpy as `__main__`; its `raise SystemExit(0)` escapes the
  scoped-loader context, so the loader receipt reads failed/SystemExit and assert_probe refuses. The
  main RL trial and verifier never ran: C1–C4 all false. Fix goes into a new v2 plan, never a rerun of v1.

## D7 — Harbor local dogfood v2: probe passed, RL stage killed by its 1800 s wall (preserved)
- Plan sha a2cd71df… (v1 plus the reviewed SystemExit wrapper fix), evidence
  `runs/aircraft-harbor-dogfood-local-v2/`.
- Measured: the SFT probe passed assert_probe (83.1 s active). In the main trial's rl-exposure stage,
  pretrained GRPO update 1 committed (752.76 s active, nonzero gradient), and the after-train
  diagnostics were written. The 1800 s stage wall then stopped it at 1801.29 s during the reload/audit
  replay (peak tree RSS 12,138,778,624 B, under the 12 GiB cap). C1–C4 false; the verifier never ran.
- The wall came from a GitHub projection (525 s/update). Locally one update took 752.76 s active under
  host contention (MPS continuation plus other sessions). v3 raises only walls, from measured timing.
