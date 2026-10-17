# freshpool launch package

Launcher for the seed-42 development stages of `aircraft-freshpool-screened-grpo-v1`
(`runs/rl-next-study-design-v1/study.json`, which wins over `study.md`). It is modelled on
`runs/aircraft-controlled400-launch-v1/{launch.py,phase_worker.py}` and
`runs/aircraft-controlled400-continuation-v1/continue.py`. Those stay frozen records and are
not imported.

Nothing here has run a model. `tests/test_freshpool_launch.py` drives the launcher with a
fake trainer and a fake `torch` module, and uses the real `scripts/evaluate_freshpool_gates.py`
for every gate and selection. `tests/test_freshpool_launch_wrappers.py` drives the three stage
wrappers with fake trainer and generation functions that write the real file layouts, read
back through the trainer's real verifiers.

## Files

| File | Role |
|---|---|
| `launch.py` | Sequences the stages, runs the gate evaluator at each gate, and owns the guards, state and journal |
| `phase_worker.py` | Runs one stage's `module:function` entry in its own process group, only if that module's source SHA256 matches the plan. Applies the 24 GiB MPS driver guard to `train` and `assay` kinds |
| `trainer_fragment.py` | Entry `trainer_fragment:main` for the `rl150`, `rl600`, `sft_e`, `sft_t` and `egsft` templates. Calls `vision_rl.vlm_local_training.main`, then writes the training fragment from the verified endpoint receipt and committed screen entries only |
| `h_assay.py` | Entry `h_assay:main` for the `assay` template. Greedy plus G4 groups (H-select: seed 42; H-gate: seeds 42 and 314) on all 200 images, through the trainer's own `predict_train_subset` and `sample_train_groups` in chunks of at most 32 |
| `p0_screen.py` | Entry `p0_screen:main` for the `p0` template. Scores the first 400 `S_scan_seed42` candidates under W with the trainer's `_batch_gold_probability` and samples G4 groups at seeds 42 and 314 on the first 64, writing the evaluator's `g1` document |

## Stage order (seed 42)

1. **G0.** Checked by the launcher before anything starts, by content where the content is checkable:
   - `manifest.json` hash, `g0_audit_complete`, `forced_fire_passed`, and every manifest file hash;
   - H-select and H-gate: unique IDs, exactly 2 per class of the evidence labels, disjoint, counts equal to `manifest.json`, and `(id, label)` pairs equal to the evidence base's rows;
   - `s_scan_seed42.jsonl`: unique IDs, disjoint from H, in `random.Random(42).shuffle(sorted(ids))` order;
   - `phash_audit.json`: forced fire passed, the planted copy at Hamming ≤ 6 and flagged, exactly 50 distinct-class pairs, each recorded at Hamming > 6, none flagged;
   - protocol revision: id `fgvc-aircraft-freshpool-screened-grpo-v1`, `activated: true`, its `revises` candidate present at the stated SHA256, `substantial_gain_gates` and `strong_baseline` equal to the candidate's, and both files committed (bytes equal to the index and `HEAD`);
   - decision record: committed, with a line recording the 16-shot plan as retired;
   - branch receipt: committed; B0 (W valid EOS ≥ 0.95 and `w_evaluator_refused: false`), B1 (W VAL in [0.15, 0.85]), D_req and the B2 mode equal to the evaluator's `d_req(a*)`.
2. **E0**, the resume proof. The fragment must report 2+2 and 1+3 identity, an unchanged reference, a finite nonzero gradient, the scorer using the sampler transform, and sampler-teacher agreement ≤ 0.001.
3. **W on H-select and H-gate, then P0**, which writes the G1 document. Then **G1**.
4. **RL** at every grid LR to 150 accepted updates, each followed by an H-select assay. Then **G2**.
5. **The two G2 `continue_to_600` arms** continue from their own validated 150 endpoint to 600, then an H-select assay each. The continuation's per-step seconds must pass through the 150 endpoint's own committed seconds at step 150 and its accepted sequence must extend the 150 endpoint's, so T_RL is cumulative. **RL\*** is the evaluator's `select_rl_star()`.
6. **SFT-E** at every grid LR on RL\*'s accepted manifest, each with an H-select assay. SFT-E\* is the evaluator's `select_sft_e()`.
7. **H-gate assays** for RL\* and SFT-E\*. Then **G3a**, which supplies the SFT-T LRs (`sft_t_lrs`).
8. **SFT-T** for each of those LRs:
   - continues its SFT-E checkpoint on RL\*'s scanned manifest;
   - receives `{t_rl}`, RL\*'s cumulative committed active seconds;
   - must start its step series at SFT-E's step 601 at or above SFT-E's committed seconds (no reset);
   - must stop at the first step at or above T_RL: its final seconds are ≥ T_RL and the step before is < T_RL;
   - is followed by an H-select assay.
9. **H-gate assay for SFT-T\***, then **G3b**.
10. **EG-SFT** at SFT-E\*'s LR on RL\*'s accepted manifest, then an H-gate assay.
11. **Final evaluator run.** It must report no failure, G1 through G3b passed, and `awaiting == "G4"`.

VAL (G4), the replicate seeds (G5) and TEST (G6) belong to the native-VAL package (study work item 6). They are not run here.

## Guards

All of them refuse; none of them warns.

| Guard | Rule |
|---|---|
| Missing gate result | The gate stops the study if the evaluator exits non-zero, writes no report, reports the gate as anything other than `evaluated`, reports `pass` as anything other than `true`, or reports any `failed_at` |
| Disk | A 5 GiB reserve (`DISK_RESERVE_BYTES`). A plan may raise it, but a plan below it is refused. It is checked before every stage and polled during each one |
| 1.25× caps | Applied per arm (`CAP_FACTOR`). The cap is 1.25 × `forecast_seconds[...]`, or 1.25 × T_RL for SFT-T. It is passed to the trainer as `{active_cap}` (cumulative), enforced by the launcher on wall time that starts, for a continuation (RL 150 → 600, SFT-E → SFT-T), at the validated parent's committed seconds (never reset), and checked again against the fragment's cumulative `committed_active_seconds`. A capped arm is a standing stop |
| Second owner | Checked before every stage and every `OWNER_POLL_SECONDS` (2 s) while one runs. The launcher, its ancestors, and its running stage's process group and descendants are exempt; any other process whose command line contains a plan `owner_patterns` entry stops the stage and the study |
| Unexpected exceptions | Any exception that is not a `Stop` is journalled as `study_stopped` with reason `unexpected_exception`, the error and the traceback tail, and the state becomes `stopped`. A malformed fragment is a stop, never a pass. Before any state exists, the launcher refuses with `verification_error` |
| Survivor grace | After a stage exits, its process group gets `survivor_grace_seconds` to drain. Anything still alive after that is journalled, killed, and fails the stage |
| Other guards | Process-group RSS limit; every pinned file hash checked before every stage and gate; frozen environment; rolling useful-mixed ≥ 0.40 over every 100 accepted RL updates; RL reference and every SFT arm's initial adapter equal to W |
| Assays | Each assay's image IDs must equal its split manifest's 200 IDs exactly, with exactly 5 (H-select) or 9 (H-gate) completions per image, each a label string or null. A partial assay stops the study: the evaluator would otherwise score a missing image as all-invalid, which lowers W or a comparator and loosens G2/G3a/G3b |

## State and journal

The state directory is `plan.state_directory`.

- **Single use.** A start creates the directory and refuses if it already exists. `--resume` continues only a `running` state of the same plan. A `stopped` or `completed` state never runs again.
- **Interrupted stages are never resumed or quoted** (study.json `budget_matching.caps`: "capped/interrupted arm re-run from scratch, never quoted"). Every stage attempt runs in a fresh `stages/<stage>/instance-<k>` and `arms/<stage>/instance-<k>`. On `--resume`, a stage that was started but not completed is journalled as `arm_instance_void` (and listed in `state.json` `void_instances`), then re-run from scratch in instance k + 1, from the stage's own declared start: W, or the validated parent endpoint for a continuation. Nothing of the void instance is passed to the trainer. A plan template that declares a `resume_argv` is refused. A capped arm is not re-run: the cap is a standing stop.
- **Append-only journal.** `journal.jsonl` is appended to and fsynced. `state.json` records the journal's byte length and SHA256, and a resume refuses if that prefix changed.
- **Lock.** A `lock` file holds the launcher's pid. A second launcher refuses without touching the state.

## Plan (`schema: freshpool-launch-plan-v1`)

| Key | Contents |
|---|---|
| `python` | Interpreter path |
| `state_directory` | Where state, journal and outputs go |
| `evaluator` | Path to the gate evaluator |
| `environment` | The frozen environment |
| `file_sha256` | Pinned file hashes. Must include the evaluator and `phase_worker.py` |
| `limits` | `minimum_free_bytes`, `rss_limit_bytes`, `survivor_grace_seconds`, `poll_seconds` |
| `forecast_seconds` | `e0`, `p0`, `assay_h_select`, `assay_h_gate`, `rl`, `sft_e`, `egsft` |
| `templates` | One per kind: `e0`, `assay`, `p0`, `rl150`, `rl600`, `sft_e`, `sft_t`, `egsft`. Each has `kind`, `entry`, `entry_sha256` and `argv`; a `resume_argv` is refused |
| `b4_edge_extension` | Must match the evaluator's grid |
| `w_endpoint` | `adapter_sha256` and `adapter_path` |
| `owner_patterns` | Command-line patterns that mark a second accelerator owner |
| `g0` | `manifest_dir`, `manifest_sha256`, and `receipts` for `protocol_revision`, `decision_record` and `branch` |
| `evidence_base` | Path and SHA256 of a file holding `labels`, `rows`, `a_star` and `b4_edge_extension` |

Template placeholders:

- `{python}`, `{stage_id}`, `{arm}`, `{stage_dir}`, `{arm_dir}`, `{fragment}`, `{seed}`, `{active_cap}`, `{w_adapter}`
- `{manifest_<name>}` for each G0 manifest file
- `{run_id}`, `{lr}`
- `{split}`, `{adapter}` (the endpoint's `adapter_path`), `{split_manifest}` for assays
- `{resume_checkpoint}` for the RL 150→600 continuation (pass it to `trainer_fragment.py --continue-output-from`)
- `{accepted_manifest}`, `{scanned_manifest}`, `{parent_checkpoint}` and `{t_rl}` for the SFT arms (SFT-T passes `{t_rl}` as the trainer's `--stop-after-active-seconds`)

An unknown placeholder stops the study.

## Fragments

Each stage entry writes `{fragment}` as JSON.

| Stage | Fragment contents |
|---|---|
| Training (`trainer_fragment.py`) | `status: completed`, `run_id`, `arm` (RL, SFT-E, SFT-T or EG-SFT), `lr`, `seed`, `updates`, cumulative `committed_active_seconds`, `step_committed_active_seconds` (contiguous, nondecreasing, ending at `updates` with the committed seconds), `adapter_sha256` (final adapter weights file), `adapter_path`, `reference_sha256` (W's weights SHA256: RL's KL reference, SFT arms' initial adapter), `useful_mixed`, `checkpoint`. RL also writes `accepted_manifest` and `scanned_manifest` (scanned IDs a prefix of `S_scan_seed42`) |
| Assay (`h_assay.py`) | `run_id`, `split`, `adapter_sha256` (must equal the endpoint's), `predictions` (every split-manifest ID, 5 or 9 completions each) and `eos_fraction`, in the evaluator's format |
| P0 (`p0_screen.py`) | The evaluator's `g1` document, seed 42 under W, whose `scan_order` must equal the `S_scan_seed42` manifest and which holds 400 journal entries and 128 rollout groups |
| E0 | `checks` and `sampler_teacher_max_abs_logprob_difference` |

The launcher assembles the evaluator's evidence document from these fragments plus `evidence_base`.

## Open dependencies

Each of these is needed before a real launch. None is a defect in this package.

- **G0 receipts.** G0 refuses until they exist: the activated protocol revision (PR #4 adds an unactivated one), the decision record (also PR #4), and the controlled400 branch receipt (study work item 6).
- **E0 entry point.** The `e0` template must name a pinned entry that writes the E0 checks. None is tracked yet.
- **The plan itself.** It must pin `vision_rl/vlm_local_training.py` and the three wrapper files in `file_sha256`, because the phase worker hashes only each stage's entry module.

## Blind spots

- **Process groups.** A descendant that calls `setsid()` escapes the RSS sum and the survivor check.
- **Sampling.** RSS, disk and MPS driver use are sampled. The launcher polls every `poll_seconds`, and the phase worker samples MPS every 0.5 s.
- **Owner detection.** It matches command-line substrings only.
- **Cap measurement.** The cap is measured on wall time, which is an upper bound on active time.
- **Trust.** Fragment contents are trusted as written, apart from the identity, assay-shape and lineage checks above. The adapter hash on every assay must match its endpoint's.
- **Assay chunking.** The trainer refuses TRAIN inference on more than 32 images per call, so `h_assay.py` and `p0_screen.py` work in chunks of 32 in manifest order and seed each sampled chunk at its start. The draws are deterministic for a fixed manifest, but they are not one continuous RNG stream over the split.
- **Physical journals.** The wrapper reads per-step seconds and useful-mixed flags from the trainer's append-only `optimizer_audit.jsonl` and `reward_groups.jsonl`. These are exact only because no stage is ever resumed after an interruption. A mismatch with the receipt refuses.
- **Not run on a model.** `p0_screen.score_candidates` loads the model the way `predict_train_subset` does and calls the trainer's `_batch_gold_probability`. The tests replace it, so its model path has not been executed.
- **Lock.** It trusts that a dead lock pid was not reused.
