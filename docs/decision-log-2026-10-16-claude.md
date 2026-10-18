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

## D8 — Resource incident; continuation attempt 3 and Harbor v3 stopped by disk guards (preserved)
- Host disk fell to ~0.3 GiB free and RAM to 28% free. The RAM came from another session's leaking
  jobs (8–18 GB each), which were killed and their launcher paused. The disk came mostly from other
  sessions; RyanLinEnv's VM disk and outputs added about 1.3 GB.
- Continuation attempt 3: sft-1e5-time stopped on disk_reserve at 5263.8 s. Addendum-4 (1027304) resumes
  it from its newest durable checkpoint as that stage's second and final attempt.
- Harbor v3 (plan 6f1d0d6c…): SFT probe completed (77.78 s active). GRPO update 1 committed
  (633.98 s active), then the host disk guard ended the trial before the verifier. v4 (plan 62fe9ff1…)
  is the identical plan with new single-use output paths.

## D9 — Pretrained Docker + Harbor dogfood PASSES (local v4)
- Plan sha 62fe9ff1…, report `reports/2026-10-16-harbor-pretrained-dogfood/`. C1–C4 all pass: the
  pretrained model loaded via pread; GRPO update 1 committed with a mixed group [0,0,1,1] (530.31 s
  active, adapter ee0e7d0f…); SFT exposure plus time-matched SFT (545.38 s cumulative); a separate
  verifier scored the same adapter hash; all guards completed.
- An independent fresh-context check reproduced every number and hash. Its corrections (cumulative
  SFT-time wording, memory.peak at the cgroup limit, v3 attribution source) went into the README.
- Integration evidence only: no learning, RL-advantage, or held-out claim.

## D10 — Continuation addenda 5 and 6 (disk)
- Addendum-5 (f8b20ac, 99b2e86): continuation disk reserve 10 → 5 GiB. The study state dir measured
  476 MB, and 3 of 4 disk stops were caused by other sessions. Addendum-6 (ee6761a): at most 2 extra
  attempts, only for stages whose every attempt ended on disk_reserve, each resuming from the
  newest checkpoint.
- Host cleanup: 4.1 GB of dead-process .tmp files deleted and 8.4 GB of logs gzipped (lossless) in
  another session's /private/tmp scratchpad.

## D11 — Fresh-pool screened GRPO design adopted as an unactivated protocol revision; 16-shot plan retired
- Superseded by D12 (field list widened; study now committed at `docs/aircraft-freshpool-study-v1.{json,md}`).
- Design: `aircraft-freshpool-screened-grpo-v1` (study draft `runs/rl-next-study-design-v1/study.{md,json}`,
  local and gitignored; study.json wins on conflicts). GRPO with just-in-time gold-probability
  screening on never-trained native-TRAIN images, against the strongest budget-matched, LR-tuned SFT,
  with EG-SFT reported as a separate mechanism verdict. Label access is train400 ∪ the scanned S
  candidates (≤29/class): **not a 4-shot claim**; gains from the extra labels are credited to data.
- Protocol: `docs/aircraft-rl-protocol-freshpool-v1.json` (protocol id
  `fgvc-aircraft-freshpool-screened-grpo-v1`) revises `docs/aircraft-rl-protocol-candidate-v1.json`
  only in `data.train_membership`, `data.shots_per_class`, the SFT `selection` rule (N = 600 accepted
  updates, LR chosen on H-select, every SFT endpoint still in the full-VAL strongest pool) and the
  added `data.train_holdout` (H-select / H-gate, 200 each). `substantial_gain_gates` and
  `strong_baseline` are byte-identical to the candidate, asserted by
  `tests/test_aircraft_protocol_freshpool.py`. UNACTIVATED: root review, the controlled400 native-VAL
  branch and D_req publication are still G0 prerequisites.
- **Retired: the unactivated 16-shot prospective TRAIN plan** (seed 2718, 1,600 images,
  `docs/fgvc-aircraft-audit-2026-10-16.json` `prospective_train_fewshot`). It shares 1,390 of the 2,919
  fresh-pool IDs (`runs/aircraft-train-screen-review-v1/verification/verification.json`,
  `additional_risk`), so the H-select/H-gate holdout is clean only while no study trains on it. No
  study may activate that plan while this protocol's holdout is in use; it was never activated.
- Honest prior: the three design judges put p(substantial) at 0.08 / 0.06 / 0.05, i.e. **~5–8%**
  (study lineage; judged, not measured). The study's own pass-all estimate is ≤0.08 (unverified).
  Most likely outcome is a stop at G2/G3 or a VAL non-pass; that is a valid, reportable result.

## D12 — Root decisions on the freshpool protocol revision (2026-10-17; widens D11's field list)
- Source: root review of PR #4 (reviewer defects: budget mismatch, VAL-vs-H-select selection,
  stale scope fields, holdout salt fragility, uncommitted study authority, empty-dict test blind spot).
- (a) LR, checkpoint and arm selection for RL and SFT uses **H-select only**. Full native VAL is used
  only for the strongest-SFT pool and the final gates (G4/G5). `evaluation_and_seal.validation` and
  `test_freeze_before_access` are revised to match. This overrides the study's
  `protocol_revision.unchanged` listing of `evaluation_and_seal`; study.json is otherwise unchanged.
- (b) `exposure_and_staging` and `supervised_controls` drop the 400/1,200-presentation budgets for the
  study's N = 600 accepted updates, the SFT-E / SFT-T / EG-SFT definitions and the 1.25x per-arm caps
  (capped or interrupted arm re-run from scratch, never quoted). `actor_and_objective`: LR grid
  1e-5/5e-5/1e-4 selected on H-select, answer-only prompt, one prompt per update.
- (c) `data.claim_scope`, `data.train_images` and `supervised_controls.scope` now state the study's
  label access (train400 ∪ S_scanned, ≤29/class; not a 4-shot claim).
- (d) `data.train_holdout.hash_protocol_id` = `fgvc-aircraft-freshpool-screened-grpo-v1` is a pinned
  literal salt. `scripts/prepare_aircraft_freshpool.py` reads it (refusing when absent; no fallback to
  `protocol_id`), so an activated copy under a new id cannot reshuffle H.
- (e) The study design is committed as `docs/aircraft-freshpool-study-v1.{json,md}` (byte copies of
  `runs/rl-next-study-design-v1/study.{json,md}`); the protocol pins both by sha256.
- `substantial_gain_gates` and `strong_baseline` stay byte-identical, asserted by
  `tests/test_aircraft_protocol_freshpool.py`, whose leaf walk now also sees empty-dict keys.

## D13 — Low review fixes on the freshpool protocol revision (2026-10-17)
- Source: PR #4 re-review (four LOW defects). `substantial_gain_gates` and `strong_baseline` untouched.
- Baseline-discovery salt: `strong_baseline.validation_discovery.selection` hashes
  `[fgvc-aircraft-2013b, protocol_id, baseline-discovery, 791, label, id]`. Its `protocol_id` element is
  pinned to the literal `fgvc-aircraft-freshpool-screened-grpo-v1` (`revises.baseline_discovery_salt`),
  not the then-current top-level id, so an activated copy cannot reshuffle the 100-image VAL discovery
  sample. No discovery run under either id was found (grep over `runs/`, not exhaustive).
- `supervised_controls.protocol_strongest_pool` narrows `strongest_definition` (kept verbatim): EG-SFT is
  excluded from the protocol strongest-SFT pool; the pool governs (`revises.strongest_definition_scope`).
- `resources.storage_proposal.minimum_free_bytes` 10 GiB → 5 GiB (5368709120), the study's standing-stop
  reserve (addendum-5/6). The stale "V2 reasoning tokens/second" item is dropped from
  `resources.not_measured`.

## D14 — controlled400 complete; native VAL activated before any VAL call
- Continuation envelope (addendum-6) completed: all 8 endpoints published, equal active time 8116-8118 s per arm.
- TRAIN32 (in-sample) greedy/sampled: warm 26/0.707, RL400 27/0.730, SFT-time 1e-5 29/0.879, 5e-5 30/0.938, 1e-4 30/0.934. SFT memorizes TRAIN far better than RL at equal time; this is not a held-out result.
- Native VAL eight-endpoint evaluation activated (activation sha c9557c70..., bundle plan cf871cfb...; copies in docs/aircraft-native-val-activation-v1/). Next: supervise_chunks envelope-1, no TEST.

## D15 — Native VAL envelope-2 operating choices (before any VAL call)
- envelope-1 halted with 0 invocations: ~/.claude/QUIET exists (turned on 23:49 when Ryan asked for quieter fans; later "a BIT loud is fine").
- envelope-2 runs with --quiet-mode-rationale (journalled): one moderate GPU job is within the fan request; chunk python processes measured ~24% CPU in the continuation, below the fan-guard 40% demotion line, and the evaluator 120 s/image cap is ~100x the measured ~1 s/image.
- envelope-2 disk reserve 5 GiB via --minimum-free-bytes (shared host disk sits at 11-12 GiB; evaluation outputs capped at 2 GiB), matching continuation addendum-5. Evaluator, plan, arms, budgets and the 24 h/512-invocation caps are unchanged. Supervisor copy used for envelope-1 is kept at supervision/supervise_chunks.envelope-1.py.

## D16 — native VAL envelope-2 halted by my own G0 run (between chunks; nothing lost)
- envelope-2 completed chunk 1 (arm common, 128 images, exit 0), then halted before chunk 2 with accelerator_owner_live: the guard saw my concurrent CPU-only G0 manifest run (scripts/prepare_aircraft_freshpool.py) as a possible second accelerator owner. Halt was between chunks: no reservation left unknown. Root error, not a guard defect.
- envelope-3 resumes with identical flags; evaluator invocation count and 24 h wall budget carry over (never reset). Rule while VAL runs: no other RyanLinEnv python process.
- G0 (runs/aircraft-freshpool-g0-v1, TEST pHashes 3333/3333 digest-verified, 0 model calls): pool 2919 -> 2429 after audit (490 near-duplicate drops), S=2027. The drop count is far above the design assumption and is under review before any launch.

## D17 — G0 pHash audit: high false-positive rate, threshold kept (pre-registered)
- 2,455 external pairs flagged at Hamming <= 6 (test 1,200 / validation 1,100 / train400 155), Hamming {0:3, 2:94, 4:504, 6:1,854}; 490 pool drops (2,919 -> 2,429), S = 2,027.
- Where reference labels may be read (VAL, train400), only 6-14% of flagged pairs share the pool image label (random-pair baseline ~1% with 100 classes); even Hamming-0 pairs are cross-label. So 64-bit DCT pHash on aircraft-against-sky images mostly flags look-alike compositions, not duplicates.
- Consequence: drops are conservative (cost data, cannot leak). The detector is weak evidence about true near-duplicates in either direction. Threshold 6 was pre-registered, so it is NOT re-tuned after seeing data. S = 2,027 still supports N = 600 accepted at acceptance >= 0.30.

## D19 — Expanded-LoRA v2 integrated into tracked code; review defects fixed (unactivated)
- Source: `runs/aircraft-expanded-lora-v2-integration` and its review `claude-review-v1` (D1-D7). Branch `expanded-lora-v2`.
- Kept: `expanded_converter.py` byte-identical (sha 32fb2927..., pinned by `approval.json`; ruff E501/I001 ignored for that one file). The 112-tensor legacy grader (`vlm_local_task_builder.py`) is untouched. The expanded profile is a separate class `ExpandedProfile` (`adapter_profile = qwen2vl2b-language-seven-r8-v1`), so every legacy `asdict(Profile())` identity is byte-identical. v2 instead added the field to `Profile` itself, which would have changed every legacy control identity and broken legacy resumes.
- The transform hooks (v2 `build_native.py`) are applied by hand to the current tracked trainer. Tracked `vlm_local_training.py` (78bb91dc...) has moved past v2's base 5d608416, so v2's hash-gated transform and its AST-equality test cannot apply to it. Expanded training also refuses EG-SFT and SFT-T continuation.
- D1/D1b: an endpoint loads only from `<run>/final_adapter`, and its bytes must match the pinned endpoint. `attach_numeric` takes the validated weights SHA and refuses if the deserialized bytes differ from it.
- D3: the endpoint records the SHA of every `vision_rl/*.py` (`environment.expanded_source_closure`, written in `train_local`), and verification needs the verifier's closure to be identical. `base-source.json` is superseded: tracked siblings differ from the controlled400 snapshot by design.
- D5: INIT is compared by hashes, not by path. The verifier passes `initial_path`.
- D6: `validate_endpoint` needs a 64-hex pin. Implicit use (TRAIN assay, attach) needs the pin listed in the data file `expanded_assets/endpoint-pins.json`, which is empty, so nothing is accepted until a reviewed change lists a pin. The registry is not a .py file, so registering a pin leaves the bound source closure unchanged. The first draft kept it in the module, and the review showed that registering a pin would then have invalidated the endpoint itself.
- Endpoints verify only against the exact `vision_rl` source that produced them: a verifier checks out that commit. `publish_endpoint` runs after `train_local`'s try block, so a refusal there cannot rewrite a completed native record as failed.
- D2: `check_matched_controls` now needs: pins; distinct endpoints; declared exposure or time arms with distinct (arm, LR); exhausted budgets; and an explicit `accepted_subset` whose prompts RL committed.
- D4: `tests/test_expanded_lora.py` trains a tiny random Llama with released TRL (SFT and GRPO, 2 steps) on a tiny trust root, then publishes and verifies the endpoint and runs tamper negatives. It runs in the CI local-vlm job, which also lists it. `train_local` itself is still not exercised end to end, because it requires the pinned Qwen directory.
- Not committed: the 4.3 MB warm `adapter_model.safetensors` (gitignored). Production expanded paths refuse until an operator places it.
- Legacy source identity does change. `asdict(Profile())` is unchanged, but tracked `vlm_local_training.py` moves from 78bb91dc... to this branch's hash, and a native run binds that file's SHA (`environment.adapter_source_sha256`). So after merge: any legacy run started from the tracked tree at 78bb91dc refuses to resume (`resume requires the original environment and adapter source`), and any plan or recipe context that pins the tracked file (aircraft_recipe `source_hashes`, a freshpool plan's `file_sha256`, a task_builder `SOURCE_FILES` snapshot) stops matching and must be re-cut. Measured 2026-10-17 by grep of `runs/`, `recipes/`, `tasks/`, `scripts/`, `jobs/` and every local and origin branch: nothing pins 78bb91dc (the same grep finds the older trainer hashes 5d608416, 61abbc79, 8d77d31a and others, so it reads those records). Runs that cannot resume: none recorded. Studies that run from a frozen copy (native-VAL `runs/aircraft-native-val-activation-v1/frozen/src`) hash their own copy and are unaffected; `test_frozen_copy_studies_keep_their_own_source_identity` checks that.
- Review fixes (review of b79218c): an endpoint whose final weights equal INIT (by file SHA or tensor fingerprint) is refused even with `global_step >= 1`; the tiny GRPO test forces within-group reward variance and asserts changed tensors, a final SHA unlike INIT and changed logits; `predict_train_subset(expanded_initial_path=...)` passes this machine's INIT path to both the identity check and `attach_numeric`; lineage and GRPO-reference expected values are computed from the files and the native audit's convention, not from the contract.
