# freshpool launch package

Launcher for the seed-42 development stages of `aircraft-freshpool-screened-grpo-v1`
(`runs/rl-next-study-design-v1/study.json`, which wins over `study.md`). It is modelled on
`runs/aircraft-controlled400-launch-v1/{launch.py,phase_worker.py}` and
`runs/aircraft-controlled400-continuation-v1/continue.py`. Those stay frozen records and are
not imported.

Nothing here has run a model. The tests (`tests/test_freshpool_launch.py`) drive the
launcher with a fake trainer and a fake `torch` module. They use the real
`scripts/evaluate_freshpool_gates.py` for every gate and selection.

## Files

| File | Role |
|---|---|
| `launch.py` | Sequences the stages, runs the gate evaluator at each gate, and owns the guards, state and journal |
| `phase_worker.py` | Runs one stage's `module:function` entry in its own process group, only if that module's source SHA256 matches the plan. Applies the 24 GiB MPS driver guard to `train` and `assay` kinds |

## Stage order (seed 42)

1. **G0.** Checked by the launcher before anything starts:
   - `manifest.json` hash, `g0_audit_complete`, `forced_fire_passed`, and every manifest file hash;
   - the protocol revision, decision-record and branch receipts, each pinned by hash;
   - the branch receipt: B0 (W valid EOS ≥ 0.95), B1 (W VAL in [0.15, 0.85]), and D_req equal to the evaluator's `d_req(a*)`.
2. **E0**, the resume proof. The fragment must report 2+2 and 1+3 identity, an unchanged reference, a finite nonzero gradient, the scorer using the sampler transform, and sampler-teacher agreement ≤ 0.001.
3. **W on H-select and H-gate, then P0**, which writes the G1 document. Then **G1**.
4. **RL** at every grid LR to 150 accepted updates, each followed by an H-select assay. Then **G2**.
5. **The two G2 `continue_to_600` arms** resume from their own 150 checkpoint to 600, then an H-select assay each. **RL\*** is the evaluator's `select_rl_star()`.
6. **SFT-E** at every grid LR on RL\*'s accepted manifest, each with an H-select assay. SFT-E\* is the evaluator's `select_sft_e()`.
7. **H-gate assays** for RL\* and SFT-E\*. Then **G3a**, which supplies the SFT-T LRs (`sft_t_lrs`).
8. **SFT-T** for each of those LRs:
   - continues its SFT-E checkpoint on RL\*'s scanned manifest;
   - receives `{t_rl}`, RL\*'s committed active seconds;
   - must reach T_RL;
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
| 1.25× caps | Applied per arm (`CAP_FACTOR`). The cap is 1.25 × `forecast_seconds[...]`, or 1.25 × T_RL for SFT-T. It is passed to the trainer as `{active_cap}`, enforced by the launcher on wall time accumulated in `state.json` across resumes (never reset), and checked again against the fragment's `committed_active_seconds` |
| Survivor grace | After a stage exits, its process group gets `survivor_grace_seconds` to drain. Anything still alive after that is journalled, killed, and fails the stage |
| Other guards | Process-group RSS limit; a second accelerator owner (plan `owner_patterns`); every pinned file hash checked before every stage and gate; frozen environment; rolling useful-mixed ≥ 0.40 over every 100 accepted RL updates; RL reference equal to W |

## State and journal

The state directory is `plan.state_directory`.

- **Single use.** A start creates the directory and refuses if it already exists. `--resume` continues only a `running` state of the same plan. A `stopped` or `completed` state never runs again: a re-run needs a new launch, as the study requires for a capped or interrupted arm.
- **Interrupted training stages.** On resume, such a stage continues through its template's `resume_argv` with `{latest_checkpoint}`. If it has no checkpoint, the study stops.
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
| `templates` | One per kind: `e0`, `assay`, `p0`, `rl150`, `rl600`, `sft_e`, `sft_t`, `egsft`. Each has `kind`, `entry`, `entry_sha256`, `argv`, and optionally `resume_argv` |
| `b4_edge_extension` | Must match the evaluator's grid |
| `w_endpoint` | `adapter_sha256` and `adapter_path` |
| `owner_patterns` | Command-line patterns that mark a second accelerator owner |
| `g0` | `manifest_dir`, `manifest_sha256`, and `receipts` for `protocol_revision`, `decision_record` and `branch` |
| `evidence_base` | Path and SHA256 of a file holding `labels`, `rows`, `a_star` and `b4_edge_extension` |

Template placeholders:

- `{python}`, `{stage_id}`, `{arm}`, `{stage_dir}`, `{arm_dir}`, `{fragment}`, `{seed}`, `{active_cap}`, `{w_adapter}`
- `{manifest_<name>}` for each G0 manifest file
- `{run_id}`, `{lr}`
- `{split}`, `{adapter}`, `{split_manifest}` for assays
- `{resume_checkpoint}` for the RL 150→600 continuation
- `{accepted_manifest}`, `{scanned_manifest}`, `{parent_checkpoint}` and `{t_rl}` for the SFT arms
- `{latest_checkpoint}` in `resume_argv`

An unknown placeholder stops the study.

## Fragments

Each stage entry writes `{fragment}` as JSON.

| Stage | Fragment contents |
|---|---|
| Training | `status: completed`, `run_id`, `arm` (RL, SFT-E, SFT-T or EG-SFT), `lr`, `seed`, `updates`, `committed_active_seconds`, `adapter_sha256`, `reference_sha256`, `useful_mixed`, `checkpoint`. RL also writes `accepted_manifest` and `scanned_manifest` |
| Assay | `run_id`, `split`, `adapter_sha256` (must equal the endpoint's), `predictions` and `eos_fraction`, in the evaluator's format |
| P0 | The evaluator's `g1` document |
| E0 | `checks` and `sampler_teacher_max_abs_logprob_difference` |

The launcher assembles the evaluator's evidence document from these fragments plus `evidence_base`.

## Open dependencies

Each of these is needed before a real launch. None is a defect in this package.

- **No assay or P0 entry point.** Tracked code has none that writes these fragments. `vision_rl.vlm_local_training` caps `--predict-train` and `--sample-train` at 32 images. The plan's `assay`/`p0` templates must name a pinned entry once one exists.
- **The trainer's CLI does not write fragments.** A thin pinned wrapper must read the trainer's `local_training.json` and endpoint receipt, and write the training fragment, including RL's accepted and scanned manifests from the screen journal.
- **G0 receipts.** The protocol revision, decision record and branch receipt are not yet committed (study work items 6 and 7).

## Blind spots

- **Process groups.** A descendant that calls `setsid()` escapes the RSS sum and the survivor check.
- **Sampling.** RSS, disk and MPS driver use are sampled. The launcher polls every `poll_seconds`, and the phase worker samples MPS every 0.5 s.
- **Owner detection.** It matches command-line substrings only.
- **Cap measurement.** The cap is measured on wall time, which is an upper bound on active time.
- **Trust.** Fragment contents are trusted as written. The adapter hash on every assay must match its endpoint's.
- **Lock.** It trusts that a dead lock pid was not reused.
