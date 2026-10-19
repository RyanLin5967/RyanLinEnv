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

## D18 — Freshpool CPU dry run added; three launch-path defects fixed (PR #12)
- `scripts/freshpool_dryrun/run.py` (workflow `freshpool-dryrun.yml`, GitHub CPU runner) runs the real preparer, launcher, phase worker, wrappers, trainer and gate evaluator end to end on a tiny random Qwen2-VL with synthetic 4-class images. Counts scaled 150 -> 3 and 600 -> 6, cap factor 1.25 -> 100 (wall-time cap vs process start-up), pool 442; P0 (400 / 64 x 2) and H (2 per class) unscaled. Integration evidence only; no learning claim.
- CI result (PR #12, second run): drive launch completed all 31 stages; G2 failed for real (H-select delta vs W not > 0, as the near-frozen tiny model predicts), G3a/G3b were forced by the dry run, and the final check stopped with failed_at G2. Strict launch stopped at G2 with gate_failed matching the evaluator report. G0, E0 (4 vs 2+2 vs 1+3 identical on CPU) and G1 passed for real.
- Fixed: (1) every train-only SFT-T continuation raised KeyError on `before_train_gold_nll` after training, so SFT-T could never finish (found by the first CI run); (2) the launcher refuses an `sft_e` template without `--order sequential --cycle`, which the trainer needs for max_steps above the 600-row manifest; (3) `phase_worker.py` takes the device from `ACCELERATE_TORCH_DEVICE` (mps arms the MPS guard; cpu only for the dry run, guarded as below).
- Review fixes (same PR): cpu is unrepresentable in a production plan. `launch.py` refuses at plan load unless `ACCELERATE_TORCH_DEVICE` is `mps`, or `cpu` under the literal plan marker `"dry_run": true` (the dry run sets it); the launcher then passes `--dry-run` to `phase_worker.py`, which requires mps without it and cpu with it, and refuses a model stage whose forwarded `--device` (default `mps` when absent, as in h_assay, p0_screen and the trainer) differs from that device, so an inference entry cannot load on an unguarded MPS.
- Also changed in `launch.py`, undisclosed in the first report: the fragment's rolling useful-mixed check now uses the existing `MIXED_WINDOW` / `MIXED_MIN_IN_WINDOW` (100 / 40) constants instead of literals. Same behaviour, checked by hand by the reviewer: `range(0, len-99)` equals `range(0, len-100+1)`, and for integer sums `sum/100 < 0.40` holds exactly when `sum < 40`.
- SFT-T's probe `gold_nll_improved` is None (not measured): a continuation takes no pre-score, and the parent's `after_train_gold_nll` is not comparable because the 8 diagnostic rows come from each run's own TRAIN (accepted vs scanned manifest). The docstring and comment that said the parent endpoint "is scored there" were corrected; nothing compares the two.
- `run.py` `judge_strict` reports a gate_failed stop with no `gate_finished` event, or an unreadable report, as a problem instead of raising TypeError.
- Not handled, stated in the launch README: SFT-T refuses if RL* accepted every scanned candidate (scanned == accepted manifest); the dry-run E0 entry is not the production E0 entry.

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
- After merging origin/main (PR #12, D18): main's tracked `vlm_local_training.py` is already 6497cf3a, not 78bb91dc (PR #12's SFT-T KeyError fix moved it), so the legacy-identity break above applies from 6497cf3a; measured by `git show origin/main:src/vision_rl/vlm_local_training.py | shasum -a 256`. Added `test_endpoint_with_init_tensors_but_other_file_bytes_is_refused`: INIT tensors re-saved with other safetensors metadata (file SHA differs) are refused by `endpoint_payload` via the tensor fingerprint; fire-checked by disabling that branch, which the existing equal-INIT test does not catch.

## D20 — Named open-weight baseline plan pre-registered, unactivated (waits for the GPU)
- `docs/aircraft-named-baseline-plan-v1.{json,md}` + `scripts/aircraft_named_baseline.py`: Qwen3.8-27B-8bit native-instruct on native VAL, discovery (100 IDs pinned, sha c8d73a4c…, D13 salt) then full 3,333 behind a gate recomputed from the discovery journal (no unknown/hard-stop, valid >= 0.90, projected active <= 172,800 s; full cap = min(172,800, 2 x projection), a hard ceiling: an invocation starts only if charged + its wall cap fits). Projection overhead per invocation = max(mean load, charged seconds not spent generating / invocations); full refuses a discovery ledger from another plan sha or source set.
- Same prompt (sha 859dd366…, equal to the evaluator bundle's), same parser (parse_label source sha f75d20ee…, equal in the evaluator frozen copy and main) and same equal-class scorer as the eight-endpoint evaluator. Decoding differs (one seeded native sample vs greedy) and is disclosed.
- Only the direct-answer profile is planned; xhigh stays unmeasured, so the claim label is "native-instruct", not "strongest". Any unknown/hard-stop outcome halts the stage for review. Runner refuses the committed plan; activation needs a root-approved copy and its sha on every invocation. Not run: the native-VAL evaluation owns the GPU.
- Gate made consistent with the hard ceiling (review fix): full VAL proceeds only if projection + one full 18,060 s invocation wall cap <= the full cap. Without it, an accurate projection within 172,800 s (e.g. 160,000 s) passed the gate yet the ceiling would refuse the last chunk; with the 2x term binding, a projection below 18,060 s (e.g. a zero-latency discovery) passed and then every full invocation was refused. Chosen over sizing the last wall cap by its chunk because the shortfall is not confined to the last chunk when 2x binds.

## D21 — Named Qwen3.8-27B-8bit baseline activated (discovery first)
- Activated copy docs/aircraft-named-baseline-plan-v1-activated.json, sha256 4c9845f0faac78a5e526f1ac3aa81e6fb764894510af69ddf76738299180132b, committed before any baseline call. Discovery (100 protocol-salted VAL images) runs first under supervise_local_vlm; its gate result is committed before full VAL. Single accelerator owner; TEST untouched.

## D22 — 27B discovery complete: 34/100 (0 invalid), gate proceed; root git-pull incident
- Discovery (plan 4c9845f0...) completed 100/100 over 2 invocations (inv 1: 50, inv 2: 50), 0 invalid, 0 unknown, mean 5.61 s/image. Equal-class accuracy 0.34 (Hoeffding 95% [0.204, 0.476]). Gate: proceed; projected full VAL 19,253 s, cap 38,506 s.
- Incident (root error): during invocation 2 my background merge script ran `git pull` in the checkout the runner reads; the end-of-invocation re-verification saw a mid-write tree and exited 2 after the ledger had recorded a normal finish. Re-running the same check minutes later passes; parse_label/prompt hashes unchanged (f75d20ee..., 859dd366...). Generation used sources verified at invocation start; results kept. Also: chain v1 ran only one invocation and treated it as done; chain v2 first failed before any model work on a fixed telemetry filename.
- RULE: no git checkout/pull/merge in the run tree while any run is in flight; do git work in a separate worktree.

## D23 — 27B baseline run 2: fresh discovery + full VAL under one code identity (pre-registered)
- Full VAL refused (correctly) because discovery ran under different source files: PR #11 merged mid-stage. Run 1 (runs/aircraft-named-baseline-v1, discovery 34/100) is preserved and will be reported.
- Run 2 uses a NEW output root runs/aircraft-named-baseline-v1-run2 at main da4e401: discovery (same 100 IDs, same activated plan 4c9845f0...) then full VAL, chained in one detached script. Both discovery results are reported side by side; full VAL proceeds if the run-2 gate says proceed. The rerun is not outcome-selected: run 1 already passed the gate.
- No git checkout/pull/merge in /Users/idide/RyanLinEnv until run 2 ends.
## D24 — Freshpool production launch package; protocol activated; branch and D_req published (2026-10-17)
- Numbered D24, not D22: open PR #16 (`named-baseline-discovery-gate`) adds D22 and D23 to this file.
- Activation: `docs/aircraft-rl-protocol-freshpool-v1-activated.json`, protocol id
  `fgvc-aircraft-freshpool-screened-grpo-v1-activated-2026-10-17`, is a separately identified copy of
  the unactivated revision `docs/aircraft-rl-protocol-freshpool-v1.json` (pinned by sha256 in its
  `activation.activates`). Only `protocol_id`, `status`, `activated` and `activation` differ; both salts
  (`data.train_holdout.hash_protocol_id`, `revises.baseline_discovery_salt.literal`) stay the literal
  `fgvc-aircraft-freshpool-screened-grpo-v1`, equal to the G0 manifests' `hash_protocol_id`, so H and the
  VAL discovery sample cannot reshuffle. `launch.py` now requires exactly this form. Frozen in the copy:
  actor model revision and manifest, W (`f05840be…`), data spec, answer-only prompt text sha256, parser
  grammar, trainer source sha256, processor file hashes, 27B baseline revision and manifest, and the
  activated named-baseline plan (D21, sha256 pinned; native-instruct profile, D20), which only G6 uses.
  Scope: seed-42 development through G3b and EG-SFT; VAL/TEST only via the native-VAL package.
- GPU ownership: the named-baseline runner (D21) does not detect other accelerator processes, so it must
  not be started while this study runs; this launcher refuses to start or continue a stage while it runs
  (owner patterns include `aircraft_named_baseline.py` and `supervise_local_vlm.py`).
- Branch (`docs/aircraft-freshpool-branch-receipt-v1.json`, from `reports/2026-10-17-controlled400-native-val/comparison.json`):
  B0 pass (W valid+EOS 0.9727 ≥ 0.95, 0 evaluator failures); B1 pass (W 0.6784 in [0.15, 0.85]);
  B2 a* = 0.7539 (sft-1e5-time), **D_req = 0.05, full mode** (G3a/G3b margin +0.02); an RL arm needs VAL
  ≥ 0.8039 against an unchanged a*; B3 all six controlled400 SFT endpoints enter the pool; B4 best LR 1e-5
  (best exposure-matched 5e-5), no 2e-4 arm; B5 W weights and config hashes equal controlled400's common
  adapter, W VAL reused; B6 RL400 VAL 0.6982 is a stale-pool reference only.
- Launch package: `scripts/freshpool_launch/production/` (`prepare.py` writes the receipts, evidence base
  and `plan.json`; `start.py` is the single command; `chain.py` runs the launcher detached; `status.py`
  monitors). The launch runs from a detached worktree at the merge commit
  (`/Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex`), so later edits to main cannot change a
  running study's pinned files or receipts.
- Production E0 entry `scripts/freshpool_launch/e0_gate.py` replaces the dry-run-only `dryrun_e0.py`:
  4 vs 2+2 vs 1+3 updates, each trainer part in its own guarded process, plus an altered-reference resume
  that must be refused; 11 checks, all required by the launcher. E0 learning rate 1e-4 (grid maximum).
- Defects found while building it, fixed in the same PR: (1) the G0 manifests' image paths are relative to
  `data/fgvc-aircraft-2013b/development/`, not to the manifest directory, so E0 would have died on its
  first image; G0 now checks every H and scan image resolves, and `start.py` links
  `manifests/images` to the development images; (2) RL passed `{active_cap}`, which shrinks after a voided
  instance, so a 600 continuation after any interruption would have been refused by the trainer's resume
  identity check; RL now passes the constant `{arm_cap}` and plan load requires rl600 = rl150 minus its
  150 boundary; (3) the study's alt-tokenization switch (>2% non-canonical correct P0 samples) was not
  implemented; the launcher now freezes the tokenization file after G1 and passes it to RL and EG-SFT;
  (4) the study's EG-SFT estimate omitted EG-SFT's extra no-grad forward per update, so its 1.25x cap would
  likely have tripped; the forecast is 600 x (5.64 + 0.97) s; (5) the owner scan matched any command line,
  so a `grep` naming a pattern would stop the study; it now reads only Python programs.
- Forecasts (caps are 1.25x): E0 7,200 s; P0 3,564 s; RL 13,338 s (study T_RL); SFT-E 3,384 s; EG-SFT
  3,966 s; H-select/H-gate assays 2,513/4,608.8 s from the slowest measured 32-image chunk (controlled400
  RL400 assay), since an assay cap only guards a hang.
- Review fixes (three adversarial reviews of PR #17 at 5db8a61; each has a test in
  `tests/test_freshpool_{launch,production,e0,launch_wrappers}.py` or `tests/test_evaluate_freshpool_gates.py`):
  (1) chain.py's SIGKILL after its bound killed only the launcher; every stage runs in its own
  session, so the stage kept the GPU unguarded and the next resume would have stopped on it as a
  second owner. The launcher now records `running_stage` (pgid + leader start time); chain.py kills
  the launcher's group and reaps that stage (`launch.reap_orphan_stage`, identity-checked), and a
  resume reaps it before its owner scan. A second signal during cleanup is ignored (`interrupt_once`).
  (2) Ordinary test runs stopped the study: owner patterns matched substrings (`test_local_vision.py`
  named `local_vision.py`) and every launcher test spawns `phase_worker.py`. Patterns now match whole
  command-line words; a `phase_worker.py --dry-run` (CPU by its own refusal) and its descendants are
  not owners; `freshpool_launch/launch.py` is no longer a pattern (it never touches the GPU);
  `tests/conftest.py` and the dry run refuse while the production state is `running` for any test
  module naming an owner pattern; an owner already present at launch or resume is a refusal (state
  untouched), not a stop. (3) The interpreter: the plan pins the binary the venv link resolves to
  and launch.py re-resolves the link before every stage (`interpreter_link_changed`). (4) Disk:
  start.py's output allowance is measured (18 x 107,084 KiB, the largest controlled400 trainer
  output, + 128 MiB = 1.96 GiB; required 6.96 GiB; `df` showed 9.7 GiB free on 2026-10-17), a resume
  needs what the launcher needs, and before every stage the launcher PAUSES (exit 3, resumable,
  nothing voided) unless free >= reserve + 4 x 107,084 KiB, so a slow squeeze stops between stages
  instead of breaching the 5 GiB standing stop mid-stage. Swap shares the volume: still a risk.
  (5) D22 collision: this entry is D24; start.py runs from a detached worktree of the merged commit
  (no checkout, pull or merge in the main checkout, which PR #16's D23 forbids while its run 2 is live),
  so a later decision-log edit on main does not break the plan's pinned receipt.
  (6, 3-reviewer) Caps from measured pilot timing: after G1 the launcher freezes RL = N(20.29 + s k/a)
  and EG-SFT = N(5.64 + s k) from P0's measured acceptance a, scorer seconds s and the tokenization
  count k (one teacher-forced forward per scored sequence), journalled as
  `forecasts_frozen_from_measured_pilot`; the plan's 13,338 / 3,966 s are no longer caps. At G1's
  worst pass (a 0.30, s 2.0) the cap is 1.25 x 16,176 s, not 16,672.5 s.
  (7) Quiet mode is waited out before every stage and a mid-stage priority drop is journalled
  (`stage_demoted`); it cannot be undone without root. (8) start.py: verify-only timeout is a
  refusal; "launched" only after the launcher journals its start; lock = pid + start time, so a pid
  reused after a reboot is stale, not "still running".
  (9) The alt-tokenization share counted think-block/tagged correct answers as non-canonical
  tokenizations; numerator is now bare-label (`label_only`) and non-canonical only.
  (10) SFT-E and EG-SFT exposure: the fragment carries the receipt's `committed_prompt_ids`; the
  launcher requires exactly RL*'s 600 accepted IDs in order, and for SFT-T SFT-E's then the scanned
  sequence from row 0, cycling. (11) RL vs SFT interrupt charging: SFT-T is charged its SFT-E
  parent's voided seconds (`lineage`), as RL 600 is charged RL 150's under one arm label. Parents
  are charged committed active seconds and voids wall seconds in both lineages (wall >= active).
  (12) E0 now also checks `scorer_seconds_within_active_time`. (13) Verify-only and every stage
  check the import tree: `import_root` (FROZEN/src) must be first on PYTHONPATH and every importable
  file in it pinned, so an untracked `src/sitecustomize.py` stops it. (14) A latent crash: macOS
  answers `killpg` with EPERM for an all-zombie group, which escaped `stop_group` as an
  `unexpected_exception` study stop when a stage exited between polls; EPERM is now settled by the
  member list. (15) G0 checked that each image exists, not its bytes: the plan now pins
  `g0.images_sha256` over every H and S_scan_seed42 image file (`launch.images_digest`).
- Interpretations in `scripts/evaluate_freshpool_gates.py`, ratified at this activation: sampled
  macro over every completion per image, greedy included (one reading, used identically for every
  arm, so it cannot favour one); G2's rolling useful-mixed fraction over the last 100 accepted
  updates (the study's standing-stop window); G3a validity/EOS reuse G1's 0.90/0.95 (the only
  thresholds the study gives); G1's 64 rollout images are the first 64 scanned candidates, accepted
  or not ("64 of them"); the alt-tokenization share counts bare-label non-canonical completions only.
- Activated protocol: `root_review_status` now describes the activated copy (allowed to differ from
  the revision, as `launch.py` ACTIVATION_KEYS states), and `activation.prerequisites` records how
  each of the protocol's six prerequisites is met or enforced (E0 and the measured forecasts are
  enforced by the launcher, not yet met).

## D25 — Freshpool study stopped at G1 (pre-registered; final, not rerun) (2026-10-18)
Launched 2026-10-18 00:35 from frozen worktree @6be5d02 (plan 7facaf2c…). E0 passed 12/12 checks
(2,166 s); W H-select and H-gate assays completed; P0 completed. G1 (`runs/aircraft-freshpool-v1/gates/G1-report-1.json`,
sha 4cbcbe18…) FAILED two of six checks: acceptance 0.2975 < 0.30 and valid 0.8926 < 0.90
(eos 1.0, mean_u_accepted 0.715, observed useful-mixed 0.367 vs predicted 0.311, scorer 0.74 s/candidate all passed).
The launcher stopped the study by design. Per the protocol this is final: no rerun, no threshold change.
No RL arm was trained, so there is no RL-vs-SFT result from this study. Next: forensics and a new unactivated
pre-registration (workflow wf_b00c5f55-971), informed by the contingency draft (PR #19).

## D26 — 27B baseline run 2 full stage resumed (2026-10-18)
The GPU became free when the freshpool study stopped. `runs/aircraft-named-baseline-v1-run2/chain-full-resume.sh`
(copy of chain.sh: discovery skipped, full invocations 2..32) relaunched against the unchanged root checkout
(main da4e401, clean), so run 2 stays under one code identity (D23). The pause after full-inv-1 was for Ryan's
machine lag, not a run defect. Root checkout must not move until chain_end.

## D27 — Freshpool confirmatory phase pre-registered, then withdrawn not run (PR #21; 2026-10-18)
- docs/aircraft-freshpool-confirmatory-v1.json/.md pre-register G5 (seeds 314 and 2718; RL at RL*'s frozen LR; SFT-E and SFT-T at every LR in Q) and a once-only sealed native TEST (G6) for exactly four allowlisted roles: RL* seed 42, the protocol-pool strongest SFT, W, and Qwen3.8-27B-8bit @815b83c0 (native-instruct profile).
- There is one primary endpoint: RL* minus the strongest SFT under the unchanged substantial_rl gates, with the class-stratified bootstrap (20,000 draws, seed 2718). Multiplicity is a fixed sequence of conjunctive (intersection-union) families, G4 -> G5 -> G6a -> G6b. Secondary measures are descriptive only. The plan names 17 refusal conditions.
- scripts/validate_freshpool_confirmatory.py works as follows. validate refuses a malformed plan, drift in any pin or inherited block, and an envelope receipt that exists now or ever existed in git history. authorize derives the contenders from the pinned evaluator on committed pre-TEST evidence, then writes and commits the receipt by exclusive create. claim allows one committed look per role. The 55 tests use hand-coded expected values.
- Status: the parent study stopped at G1 (D25), so under the plan's own development_stop_before_G4 outcome it is never activated and is to be marked WITHDRAWN_NOT_RUN.
Withdrawn in the same breath: the freshpool study stopped at G1 (D25), so per the plan's own development_stop_before_G4 rule it is never activated; status set to UNACTIVATED_WITHDRAWN_NOT_RUN. TEST stays sealed.

## D28 — Freshpool results report tool (PR #22)
gate results for aircraft-freshpool-screened-grpo-v1 are reported with scripts/freshpool_report.py. Every gate and every journal selection is recomputed by the pinned evaluator (scripts/evaluate_freshpool_gates.py, run via runpy after its bytes, launch.py's and classification_comparison.py's are hashed against the plan's pins) from journal-admitted fragments only, and must agree with the launcher's own evidence, report and selection records; any disagreement, missing fragment, or unreproducible selection is a refusal (exit 2, one JSON line), never a warning. Only H-select / H-gate assays are admitted, so VAL and TEST are never scored; H uncertainty is the protocol's paired within-class bootstrap (20,000 draws, seed 2718). The VAL target 0.8039 = a* 0.7539 + D_req 0.05 is a lower bound (G4 also needs +0.05 over the strongest SFT on VAL). Live run aircraft-freshpool-v1 stopped at G1 gate_failed (acceptance 0.2975 < 0.30, valid 0.8926 < 0.90, per gates/G1-report-1.json). Known launcher window (state.json saved before stage_completed is appended) fails closed with gate_evidence_differs_from_fragments; launch.py is unchanged because the live plan pins it.
## D29 — Pretrained Harbor dogfood reproduction on a GitHub-hosted runner (PR #25)
We add a manual workflow, `aircraft-harbor-dogfood.yml`, that re-runs the passing local plan `62fe9ff1…` byte-for-byte on a free `ubuntu-24.04` runner. Plan files come from the tracked archive checked against `plan-SHA256SUMS`. Host and build files come from source commit `1323eb1`. Inputs come from `setup_runtime`. The trial is `local_run.execute`, with only its Mac/Colima launch gate swapped for a first-attempt GitHub-runner gate. Pass means C1–C4 from `local_run` hold **and** the driver's `dogfood-local-result.json` exists with `pass: true` and `driver_error: null`. There are three deviations from the plan, all disclosed in `docs/aircraft-harbor-dogfood-ci.md`:
- The outer timeout is `timeout -s INT -k 120 15300` instead of the plan's 14600 s with TERM.
- The README step 3 cgroup read runs in the runtime image.
- The runner recreates `/Users/idide/RyanLinEnv` rather than editing pinned plan bytes.

The scope stays integration-only (TRAIN fixtures, not held-out, not RL-vs-SFT). Known risks, all unmeasured on x86:
- About 0.2 GiB of headroom over the 14 GiB preflight.
- SFT tree RSS has never been measured on x86.
- C2 can fail through a uniform-reward GRPO group.

A failure is reported as a result and is never retried. The status is not yet dispatched.

Review check at head 9e31db3 (read-only, nothing run locally): both blocking findings are resolved and every CI check passes.
1. The PR body now describes this PR. It lists the same four files as the diff (`.github/workflows/aircraft-harbor-dogfood.yml`, `scripts/harbor_dogfood_ci.py`, `docs/aircraft-harbor-dogfood-ci.md`, `tests/test_harbor_dogfood_ci.py`). The named-baseline plan text is gone.
2. `criteria()` now calls `driver_record_error(root/output/local_run.RESULT_NAME)`, and `RESULT_NAME` is `dogfood-local-result.json` in the v4 `local_run.py`. That function returns an error if the file is missing or unreadable, if `driver_error` is non-null, or if `pass` is not true. The error is passed into `local_run.evaluate`, which sets pass to `driver_error is None and all(criteria)`. A new parametrized planted test, `test_driver_failure_fails_although_evidence_passes`, covers a CalledProcessError, the wall guard, `pass: false` and a missing file. In each case C1–C4 still pass, but the receipt fails and the exit code is 1.

CI (`gh pr checks 25`): cpu, detection, harbor and local-vlm passed in both runs.
## D30 — Named open-weight baseline Qwen2.5-VL-3B on free CI, pre-registered unactivated (PR #27)
named open-weight baseline on free CI, pre-registered and UNACTIVATED. Adds plan aircraft-named-baseline-ci-qwen2.5-vl-3b-v1 (docs/aircraft-named-baseline-ci-plan-v1.json): Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3, all 13 files pinned by SHA256. It runs zero-shot in bf16 on free ubuntu-24.04 CPU runners as a sharded workflow_dispatch matrix. Prompt, greedy predict, the EOS-terminated parse_label(allow_reasoning=True) and the metrics/bootstrap (20,000 draws, seed 2718) come from the frozen evaluator at 85f0865. All 41 sources are materialised and hash-checked against reports/2026-10-17-controlled400-native-val/plan.json (cf871cfb...).

Disclosed deviations: bf16 on CPU, repetition_penalty 1.0, 4 threads, and ATen AVX2 with oneDNN off on every host. The pinned kernel path was added because an AVX-512 bf16 host changed one greedy TRAIN answer. The PR plumbing canary requires the pre-registered answers.

Stages: plumbing (3 TRAIN images, never scored), discovery (the D13 100-ID sample, exactly 4 shards) and full VAL (3,333 images). Full VAL is gated on a committed discovery receipt from one first-attempt run on main, made by the same runner and workflow SHA256, with at least 90% valid answers. Its shard count must satisfy N >= ceil(3333 / floor((19800-900-max fixed setup) / max(330, 3.0 x (slowest discovery shard mean latency + mean fetch)))).

VAL is dispatched only from refs/heads/main. The merge refuses any missing, duplicate or misplaced image, any non-completed outcome, any re-run shard, any foreign run and any identity mismatch. No TEST data is touched.

Activation requires root to commit an activated copy, which may differ only in activated, activation and status, and to pass its SHA256.

Review (2026-10-18), against head fca62ef with read-only checks: all three blocking findings are resolved. (1) Merge gaps and duplicates: verify_shards and read_shard_records refuse every case. (2) Prompt and parse drift: the frozen Backend.predict, _deadline, the run_chunk output checks and the terminated-only parse were reused unchanged, and the merge re-parses every response. (3) Pinning: the model revision is the full hash and equals the frozen MODEL_REVISIONS, with per-file SHA256 and hashed torch locks. CI is green on fca62ef: runs 39124910292, 39124907507 and 39124910205.
## D31 — Housekeeping and compute (2026-10-18)
- Closed unmerged (branches kept): #20 and #23 (duplicates from agents that died; superseded by reviewed #25 and #27); #19 contingency draft (fresh verification found unresolved blocking issues; superseded by PR #26 and the exploratory sweep PR #29).
- 27B MLX baseline run 2 paused permanently on the Mac after full invocations 1-3 (37.5 GB wired memory lagged Ryan's Mac). Replacement: v2 plan on UWaterloo big-RAM CPU hosts (PR #28), a different engine identity; v1 run-2 partial results are not merged with v2.
- Heavy compute moved to UWaterloo Engineering hosts via uwrun (Ryan's order). Exploratory sweeps there use TRAIN-derived holdouts only; native VAL and sealed TEST are untouched until a recipe is pre-registered.

## D32 — Harbor dogfood reproduced on a free GitHub runner (run 39126983510, 2026-10-18)
- Workflow from PR #25, dispatched on main against plan sha256 62fe9ff1d71ba6c50a183ed1ffecdc2318be2fe5b31c521b3079e6aedc494901 (the local v4 PASS plan). Every step succeeded; ci-criteria.json reports C1-C4 all pass, missing_criteria [], driver_error null, native VAL and TEST unused.
- C2 delta L2 0.0104217 matches local v4 (0.01042); 112 changed tensors. Final adapter sha 2a4964bd... differs from local ee0e7d0f... (x86 runner vs aarch64 Colima; floating-point order differs). C3 verifier hash equals the RL adapter hash in the same run, which is the criterion.
- Evidence downloaded to runs/harbor-ci-dogfood-39126983510/ (local, ignored). Integration only: not learning, not held-out, not an RL-vs-SFT result. The real-pretrained Harbor dogfood requirement now holds locally (v4) and on CI.

## D33 — Qwen2.5-VL-3B free-CI named baseline activated (discovery first)
- Activated copy docs/aircraft-named-baseline-ci-plan-v1-activated.json, sha256 8ae93e9969f636084329a875801e50c73f6cb149b1603dcf78bd0addd71e0901, from source plan 6914b1ad... (PR #27). Differs only in activated, status and activation. Plumbing passed on PR #27 head fca62ef, and every pipeline file is byte-identical on main.
- Next: dispatch discovery (exactly 4 shards) from main with this copy. Commit the merged discovery-receipt.json before full VAL. Native VAL only; sealed TEST untouched.

## D34 — Qwen2.5-VL-3B CI discovery receipt (run 39133341093)
- Discovery completed 100/100 on 4 first-attempt shards from main (sha e8c4dd0), plan sha 8ae93e99. Accuracy 25/100 (Hoeffding 95% [0.114, 0.386]), valid 0.98, unterminated 0. Not a benchmark score. Slowest shard mean latency 298.19 s/image, fetch 0.585 s, max fixed setup 64.33 s (all AMD EPYC 7763).
- Shard rule: per-image budget max(330, 3.0 x (298.19 + 0.585)) = 896.3 s; floor((19800 - 900 - 64.33) / 896.3) = 21 images per shard; N >= ceil(3333 / 21) = 159. Full VAL dispatched with N = 159 after this receipt is on main. TEST untouched.

## D35 — RL-from-I corrections adopted; confirmatory pre-registration written, unactivated (2026-10-18)
- Adopted as binding for every RL-from-I lane: `docs/aircraft-rl-lessons-audit-2026-10-18/CORRECTIONS.md` (sha256 cc392fa0…1dcf), a byte copy of the gitignored `runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md`. Its evidence, `audit.md` bfc3dd1b… (C1–C19) and `critic.md` ed9c3845… (M1–M12), is copied beside it. It has two parts:
  - **GATE 0.** No new GRPO-I arm launches until four things are measured: P0-I, the SFT-from-I curve on the same labels, the dev800 calibration against native VAL, and the placebo pairs. P0-I is a no-go below about 5% pass@8 headroom.
  - **Arm design rules.** These cover:
    - fp32 GRPO unless a bf16 waiver is earned, and one device class per trajectory;
    - the LR grid, β, dynamic sampling, the screened target and multi-image updates;
    - controls with the same LR count and best-of-k, full label access, and time matching at each checkpoint;
    - dropped families;
    - pool rules that exclude the 477 near-duplicates and train400, and a rebuilt confusion map;
    - the RL-from-sftall\* family;
    - co-tenancy, the CUDA resume E0, a fixed rung schedule, and one shared look ledger.
- Pre-registered before any dev800 winner exists, status UNACTIVATED: `docs/aircraft-rl-from-I-confirmatory-v1.{json,md}`. It fixes:
  - **Intake.** At most K = 3 dev800-screened arms. Each must have passed GATE 0 and the arm rules, beaten a placebo-adjusted bar B = max(0.03, placebo max), and passed an h_gate sign check. Each is frozen by adapter sha256 before replicates or VAL.
  - **Comparator pool.** One global pool: I re-scored, every SFT-from-I control LR with E and T endpoints, and sftall\* with its LR siblings, all scored in one look.
  - **VAL gates.** Inherited from `substantial_rl`: ≥ 0.80, +0.05 over I, +0.05 over the strongest SFT, RER ≥ 0.20, and a lower bound > 0 on a Bonferroni 98.33% interval across K.
  - **Replicates.** Seeds 314 and 2718, run from the same frozen I. I is never retrained.
  - **One native-VAL scoring event**, with an append-only look ledger that discloses that VAL already selected I.
  - **Evaluator.** The frozen 85f0865 evaluator on MPS fp32, or one validated fp32 equivalent.
  - **Disclosures.** The 477 near-duplicates; the relabel from 4-shot to about 23 labelled images per class; the confusion-map/dev600 overlap.
  - **Prerequisite.** A full-VAL named baseline: 3B CI run 39144098450 or 27B v2.
  - **TEST.** A once-only sealed TEST for an allowlist of roles, opened only after a VAL pass.
- Not activated. Activation needs, among other things, a committed GATE 0 go, `splits.json` on main, a completed full-VAL named baseline, and a reviewed validator with fire tests for every refusal id.
