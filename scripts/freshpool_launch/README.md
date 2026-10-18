# freshpool launch package

Launcher for the seed-42 development stages of `aircraft-freshpool-screened-grpo-v1`
(`docs/aircraft-freshpool-study-v1.json`, which wins over `study.md`), under the activated
protocol `docs/aircraft-rl-protocol-freshpool-v1-activated.json`. It is modelled on
`runs/aircraft-controlled400-launch-v1/{launch.py,phase_worker.py}` and
`runs/aircraft-controlled400-continuation-v1/continue.py`. Those stay frozen records and are
not imported.

Nothing here has run the pinned model yet: the first real run is the launch below.
`tests/test_freshpool_launch.py` drives the launcher with a
fake trainer and a fake `torch` module, and uses the real `scripts/evaluate_freshpool_gates.py`
for every gate and selection. `tests/test_freshpool_launch_wrappers.py` drives the three stage
wrappers with fake trainer and generation functions that write the real file layouts, read
back through the trainer's real verifiers.

`scripts/freshpool_dryrun/run.py` (workflow `.github/workflows/freshpool-dryrun.yml`, CPU) runs
this package end to end for real on a tiny random Qwen2-VL with synthetic 4-class images: the
preparer's G0 manifests, G0, E0 (the production `e0_gate.py`), every stage through `phase_worker.py` and
the real wrappers and trainer, and every gate through the real evaluator, with the update
counts scaled (150 -> 3, 600 -> 6) and the cap factor widened (`run.py` lists every deviation).
It checks integration only, not learning. Its first CI run found the SFT-T `KeyError` fixed in
`vlm_local_training.gold_nll_improved`.

## Launch (production, MPS)

Nothing is checked out, pulled or merged in the main checkout (PR #16's D23 forbids that while
its run 2 is live, and the main checkout's tree is not what runs). Root freezes the merged launch
commit in its own detached worktree and runs `start.py` from there:

```
git -C /Users/idide/RyanLinEnv fetch origin
git -C /Users/idide/RyanLinEnv worktree add --detach \
    /Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex <merge commit of the launch PR>
python3 /Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex/scripts/freshpool_launch/production/start.py
```

It returns once the launcher has journalled `launch_started` (G0 is verified twice, with no model
call), and prints the chain pid, the frozen commit, the plan SHA256 and where to look. The study then
runs detached for about a day; nothing waits on the session that started it. `start.py` (its
docstring lists every step) refuses, with one JSON line naming the reason, unless all of these hold:
it runs from that worktree, which is clean (untracked files included) at a commit on `origin/main`;
`~/.claude/QUIET` is absent; free disk is at least 5 GiB (the standing reserve) + 1.96 GiB (study
outputs: 18 x the largest controlled400 trainer output, 107,084 KiB by `du -sk`, + 128 MiB); the
state directory `runs/aircraft-freshpool-v1` does not exist yet; `plan.json` hashes to
`plan.sha256`; the G0 manifests' `images` link points at `data/fgvc-aircraft-2013b/development/images`
(created if absent); `launch.py --verify-only` passes under the plan's frozen environment (every pin,
the interpreter link, the import tree, G0 with all three receipts, every template); and the launcher
then journals its start (a second accelerator owner already present comes back as the launcher's
refusal `second_accelerator_owner_present`, with nothing started). It starts `production/chain.py` in
a new session, which runs `launch.py` with `caffeinate -i -m -s -w <launcher pid>` (no idle, disk or
AC system sleep; a closed lid on battery still sleeps) and records
`runs/aircraft-freshpool-v1-chain/chain.json`.

Why a frozen worktree: `launch.py` resolves every repository path in the plan against its own
checkout, checks every pin before every stage, and requires its own `src` to be first on the stages'
`PYTHONPATH` with every importable file in it pinned. Running from a detached worktree at the merge
commit means later commits to main (another decision-log entry, say) cannot change a running study's
code, receipts or decision record, and do not break the launch either: the plan and its receipts are
read from the frozen commit. If the plan is stale at that commit, verify-only refuses
(`pinned_file_changed`): re-cut it with `python scripts/freshpool_launch/production/prepare.py --write`
in a PR (`--check` reports whether the committed receipts and plan are current).

Measured before this PR (2026-10-17, against the real files, no model): `--verify-only` on the
real plan passes all G0 checks; without the `images` link it refuses with
`g0_manifest_image_missing` for all 2,427 H and scan rows; every training template's arguments
(E0, RL 150, SFT-E and EG-SFT on a stand-in 600-row manifest) pass the trainer's own validation up
to its device preflight, where a sentinel stopped them.

### Expected wall time per stage (estimate)

All numbers are estimates from measured costs, not measurements of this study: GRPO 20.29 s per
update and SFT 5.64 s per update (controlled400 on this Mac, FP32 MPS), scorer 0.97 s per
candidate and acceptance 0.5 (the study's assumptions; P0 measures both), greedy about 35 s and
one G4 seed about 130 s per 32-image chunk (controlled400 TRAIN32 assays; the RL400 endpoint's G4
chunks took 262-299 s). k is the number of SFT-T learning rates G3a qualifies (1-3).

| Stage | Estimate | Cap (1.25 x forecast) |
|---|---|---|
| E0 (4 vs 2+2 vs 1+3 + altered-reference resume, 6 trainer processes) | 10-20 min | 9,000 s |
| W on H-select, W on H-gate | 17 min, 31 min | 3,141 s, 5,761 s |
| P0 (400 scored, 128 groups) then G1 | 15 min | 4,455 s |
| RL x3 to 150 + H-select each, then G2 | 3 x (56 + 17) min = 3.7 h | 1.25 x the forecast frozen after G1 from P0's measured timing (16,672.5 s at the assumed values), cumulative to 600 |
| Top-2 RL 150 -> 600 + H-select each | 2 x (2.8 h + 17 min) = 6.1 h | same arm cap |
| SFT-E x3 + H-select each | 3 x (56 + 17) min = 3.7 h | 4,230 s per arm |
| H-gate RL*, SFT-E*, then G3a | 1.0 h | 5,761 s each |
| SFT-T x k + H-select each, H-gate SFT-T*, then G3b | k x 3.05 h + 31 min (k = 2: 6.6 h) | 1.25 x T_RL, cumulative over SFT-E |
| EG-SFT + H-gate, final check | 66 + 31 min = 1.6 h | 1.25 x the measured forecast (4,957.5 s at the assumed values) |
| **Total through the final check** | **about 24 h (k = 2); 21-27 h for k = 1-3** | |

Early stops end sooner: G1 at about 1.3 h, G2 at about 5 h, G3a at about 16 h (estimates).
A cap is a standing stop, not a slowdown: if a stage runs past its cap the study stops there.
After G1 the launcher journals `forecasts_frozen_from_measured_pilot` with the RL and EG-SFT
forecasts it uses: N x (20.29 + s k / a) and N x (5.64 + s k), from P0's measured acceptance a and
scorer seconds s and the number k of sequences scored per row (1 unless the alt-tokenization switch
fired). At G1's worst pass (a = 0.30, s = 2.0 s) the RL cap is 1.25 x 16,176 s.

### Monitoring

```
python3 /Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex/scripts/freshpool_launch/production/status.py
tail -f /Users/idide/RyanLinEnv/runs/aircraft-freshpool-v1/journal.jsonl
cat /Users/idide/RyanLinEnv/runs/aircraft-freshpool-v1-chain/chain.json
```

`status.py` is read-only. It prints the study status and next action, whether the launcher (lock pid) and
the chain are alive, how long ago the journal last advanced, the running stage's elapsed wall
seconds against its cap and its log tail, gates passed, selections, voided instances and free disk.
A live pid is not progress: judge by `journal_last_event_age_seconds` and `stage_log_age_seconds`
advancing. Each stage's output is in `runs/aircraft-freshpool-v1/stages/<stage>/instance-<k>/`
(`stage.log`, `fragment.json`) and `arms/<stage>/instance-<k>/`; MPS telemetry per stage is in
`runs/aircraft-freshpool-v1/guards/`; gate evidence and reports are in `gates/`.

To stop: `kill -TERM <chain_pid>` (from `chain.json`). The launcher journals `launch_interrupted`
and exits 3; the state stays `running`. `start.py --resume` continues it from the frozen worktree
as it is: the interrupted stage is voided and re-run from scratch, its wall time still charged
to the arm. A reboot leaves the same resumable state (stale lock; the lock holds the pid and its
start time, so a pid reused after the reboot is recognised as stale). If the launcher died while a
stage ran, chain.py stops that stage's process group at once (`chain.json` `orphan_stage`), and a
resume stops any that is left (`orphan_stage_reaped`) before its owner scan. A second accelerator
owner present at resume refuses the resume (`resume_refused_second_owner`, state still running).
Before every stage the launcher waits while `~/.claude/QUIET` exists (`quiet_mode_wait`) and pauses
resumably (`launch_paused`, exit 3, nothing voided) if free disk is below the reserve plus one
stage's output (4 x 107,084 KiB): free disk, then `start.py --resume`. A `stopped` or `completed`
state never runs again.

When the development stages finish, `state.json` is `completed` with next action
`native-VAL package (G4)`: VAL (G4), the replicate seeds (G5) and TEST (G6) belong to the
native-VAL package (study work item 6), which this launch does not run.

Operating constraints, stated:
- Disk: `df` showed 9.7 GiB free on 2026-10-17 (reviews saw 8.6 and 9.3 GiB earlier that day).
  Breaching the 5 GiB reserve mid-stage is a protocol standing stop. Swap lives on the same volume
  (`sysctl vm.swapusage`), so memory pressure from other work eats the same margin. The pre-stage
  pause catches a slow squeeze between stages, not one inside a stage.
- Tests: while the state is `running`, `tests/conftest.py` refuses (exit 2, naming them) any pytest
  session that collected a test module naming an owner pattern, `phase_worker` or `launch.py`
  (`production/live.py`), and `scripts/freshpool_dryrun/run.py` refuses outright. Other test modules
  run normally. `pytest tests/test_x.py` itself is never an owner: patterns match whole words.
- The named-baseline runner (D21) does not detect other accelerator processes, so it must not be
  started while this study runs (this launcher stops if it sees it).
- Quiet mode turned on mid-stage demotes the trainer to efficiency cores at nice 19, which cannot be
  undone without root; the launcher journals `stage_demoted`, and the stage may then reach its cap.

## Files

| File | Role |
|---|---|
| `launch.py` | Sequences the stages, runs the gate evaluator at each gate, and owns the guards, state and journal |
| `phase_worker.py` | Runs one stage's `module:function` entry in its own process group, only if that module's source SHA256 matches the plan. The device is the frozen `ACCELERATE_TORCH_DEVICE`, which must be `mps` (the 24 GiB MPS driver guard on `train` and `assay` kinds). `cpu` is accepted only with `--dry-run`, which the launcher passes only for a plan carrying the literal marker `"dry_run": true` (and which then requires `cpu`); `launch.py` refuses any other plan device at load (`plan_device_not_allowed`, `dry_run_marker_not_bool`). For model kinds every forwarded `--device`, or the entries' default `mps` when none is given, must equal it |
| `trainer_fragment.py` | Entry `trainer_fragment:main` for the `rl150`, `rl600`, `sft_e`, `sft_t` and `egsft` templates. Calls `vision_rl.vlm_local_training.main`, then writes the training fragment from the verified endpoint receipt and committed screen entries only |
| `h_assay.py` | Entry `h_assay:main` for the `assay` template. Greedy plus G4 groups (H-select: seed 42; H-gate: seeds 42 and 314) on all 200 images, through the trainer's own `predict_train_subset` and `sample_train_groups` in chunks of at most 32 |
| `p0_screen.py` | Entry `p0_screen:main` for the `p0` template. Scores the first 400 `S_scan_seed42` candidates under W with the trainer's `_batch_gold_probability` and samples G4 groups at seeds 42 and 314 on the first 64, writing the evaluator's `g1` document (each completion with its token IDs) |
| `e0_gate.py` | Entry `e0_gate:main` for the `e0` template: the protocol's `real_learning_and_resume_gate`. Four screened GRPO updates from W uninterrupted, 2+2 and 1+3, each trainer part in its own process through `phase_worker.py` (24 GiB MPS guard each), plus a resume from the 2+2 split checkpoint with one reference-adapter bit flipped, which the trainer must refuse. Twelve checks, all required by the launcher; the docstring lists them |
| `production/` | The real launch: `prepare.py` (writes the receipts, evidence base, `plan.json`, `plan.sha256`), `start.py` (the single command), `chain.py` (detached runner), `status.py` (monitoring), `live.py` (whether the study is running, for the test and dry-run refusals) |

## Results report

`scripts/freshpool_report.py --state-directory <state> [--json out.json] [--markdown out.md] [--watch 60]`
is read-only over a state directory. It admits only the fragments the journal completed (by
path and SHA256), rebuilds the evaluator's evidence with `Launcher.evidence` for every
`gate_started`, re-decides each gate with `evaluate`, and refuses (exit 2) when a fragment is
missing or changed, a rebuilt evidence file or a recomputed gate differs from what the
launcher wrote, or a selection disagrees with the evaluator's. It reports per-stage status, E0,
every gate, H-select / H-gate accuracies with the protocol's paired bootstrap, RL* vs W,
SFT-E*, SFT-T* and EG-SFT, and the branch receipt's VAL target a* + D_req. It never scores
TEST. Its synthetic tests are `tests/test_freshpool_report.py` (safe beside the live study) and
`tests/test_freshpool_report_launcher_e2e.py` (real launcher runs; CI only, refused by
`tests/conftest.py` while the study runs).

## Stage order (seed 42)

1. **G0.** Checked by the launcher before anything starts, by content where the content is checkable:
   - `manifest.json` hash, `g0_audit_complete`, `forced_fire_passed`, and every manifest file hash;
   - H-select and H-gate: unique IDs, exactly 2 per class of the evidence labels, disjoint, counts equal to `manifest.json`, and `(id, label)` pairs equal to the evidence base's rows;
   - `s_scan_seed42.jsonl`: unique IDs, disjoint from H, in `random.Random(42).shuffle(sorted(ids))` order;
   - `pool.jsonl`: unique IDs, exactly 2,919 (study.json `pool.expected_images`, the pool before the audit) minus the audit's `pool_drops`, none of which it contains, with `manifest.json` `counts.pool_after_audit` and `counts.pool_drops` equal to those sizes; H-select and H-gate are subsets of the pool and the scan is a subset of pool − H;
   - `phash_audit.json`: forced fire passed, the planted copy at Hamming ≤ 6 and flagged, exactly 50 distinct-class pairs, each recorded at Hamming > 6, none flagged;
   - protocol revision: the activated copy `fgvc-aircraft-freshpool-screened-grpo-v1-activated-2026-10-17` (`activated: true`) of the committed unactivated revision `fgvc-aircraft-freshpool-screened-grpo-v1`, as under G0 receipts below: every other top-level field equal, both salts the revision's literal id, the `revises` candidate present at the stated SHA256 with equal `substantial_gain_gates` and `strong_baseline`, and all three files committed (bytes equal to the index and `HEAD`);
   - decision record: committed, with an entry that starts with an explicit `Retired:` status (list and bold markup allowed), names the 16-shot plan, `seed 2718` and `1,600 images`, and contains no hedge (`not`, `pending`, `proposed`, `reverted` and the like). The entry is the whole bullet: its first line plus every following line indented deeper (wrapped continuations and nested sub-bullets), so a hedge on a continuation line refuses. An entry that merely mentions "16-shot" and "retire" does not pass;
   - images: every H and `S_scan_seed42` row's image exists where the trainer resolves it, and the bytes hash to `g0.images_sha256` (`g0_manifest_image_missing`, `g0_image_content_differs`);
   - branch receipt: committed; B0 (W valid EOS ≥ 0.95 and `w_evaluator_refused: false`), B1 (W VAL in [0.15, 0.85]), D_req and the B2 mode equal to the evaluator's `d_req(a*)`.
2. **E0**, the resume proof (`e0_gate.py`). The fragment must hold exactly the entry's twelve checks, every one true (2+2 and 1+3 identical to the uninterrupted run, scan in S_scan_seed42 order, every run from W, reference unchanged, the altered-reference resume refused, a useful mixed group, a finite nonzero task gradient on a step with unequal rewards, a LoRA change, the scorer on the sampler transform, a cumulative active-time ledger, the trainer's scorer seconds inside committed active time equal to the committed screen journal's), and sampler-teacher agreement ≤ 0.001.
3. **W on H-select and H-gate, then P0**, which writes the G1 document. Then **G1**. Right after G1 the launcher freezes `screen_tokenizations.json` (study.json `alt_tokenization_switch`): if more than 2% of P0's correct samples are non-canonical tokenizations (bare-label answers whose token IDs differ from gold+EOS; a think-block or tagged answer is not a tokenization) it lists every correct tokenization P0 observed per label, otherwise it is empty (canonical only). Every RL and EG-SFT stage receives it as `{screen_tokenizations}`, and its SHA256 is checked before every later stage.
4. **RL** at every grid LR to 150 accepted updates, each followed by an H-select assay. Then **G2**.
5. **The two G2 `continue_to_600` arms** continue from their own validated 150 endpoint to 600, then an H-select assay each. The continuation's per-step seconds must pass through the 150 endpoint's own committed seconds at step 150 and its accepted sequence must extend the 150 endpoint's, so T_RL is cumulative. **RL\*** is the evaluator's `select_rl_star()`.
6. **SFT-E** at every grid LR on RL\*'s accepted manifest, each with an H-select assay. SFT-E\* is the evaluator's `select_sft_e()`.
   - The `sft_e` template must pass the trainer (after the wrapper's `--`) exactly one `--stop-after-steps 600` **and** one literal integer `--max-steps` above 600, checked at plan load (`sft_e_template_stop_after_steps_not_600`, `sft_e_template_max_steps_not_above_600`) (the SFT-T step bound, since SFT-T's `max_steps` is cumulative and must exceed the parent's anyway). The trainer continues SFT-T only from a parent whose receipt is `boundary_reached` at its declared `stop_after_steps`; a run to `max_steps = 600` is `total_budget_exhausted` and would be refused only at SFT-T, after G3a. It must also pass `--order sequential --cycle` (`sft_e_template_not_sequential_cycle`): the trainer refuses a sequential run whose `max_steps` exceeds its 600-row manifest unless it cycles.
   - The launcher therefore requires each SFT-E fragment to report `receipt_status: boundary_reached` and `requested_boundary.stop_after_steps == 600` (`sft_e_not_a_continuable_boundary` otherwise).
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
| Disk | A 5 GiB reserve (`DISK_RESERVE_BYTES`). A plan may raise it, but a plan below it is refused. It is polled during every stage (a breach is the standing stop). Before a stage starts, free disk below the reserve plus `stage_headroom_bytes` pauses resumably instead (`disk_headroom_before_stage`) |
| 1.25× caps | Applied per arm (`CAP_FACTOR`). The cap is 1.25 × the forecast: `forecast_seconds[...]` for E0, P0, assays and SFT-E; for RL and EG-SFT the forecast frozen after G1 from P0's measured timing (`state.json` `measured_forecasts`); 1.25 × T_RL for SFT-T. The arm is charged, cumulatively and never reset: the validated parent's committed seconds for a continuation (RL 150 → 600, SFT-E → SFT-T), plus the wall seconds of every voided (interrupted) instance of the same arm label (`state.json` `void_seconds_by_arm`; RL 150 and its 600 continuation share one label, and SFT-T is also charged its SFT-E parent's voids, `lineage`), plus the running instance's wall time. The trainer receives `{active_cap}` = cap − voided seconds as its cumulative limit, and the fragment's cumulative `committed_active_seconds` is checked against the same reduced cap. A capped arm is a standing stop |
| Second owner | Checked before a launch or resume (a refusal, state untouched), before every stage and every `OWNER_POLL_SECONDS` (2 s) while one runs (a stop). The launcher, its ancestors, and its running stage's process group and descendants are exempt, and so are a `phase_worker.py --dry-run` (CPU by its own refusal) and its descendants; any other process whose program (argv[0]) is a Python interpreter and one of whose command-line words IS a plan `owner_patterns` entry (`x.py` or `.../x.py`; a module, a submodule of it or a path through it) is an owner. A shell, `grep` or editor naming a pattern, or `pytest tests/test_local_vision.py`, is not |
| Unexpected exceptions | Any exception that is not a `Stop` is journalled as `study_stopped` with reason `unexpected_exception`, the error and the traceback tail, and the state becomes `stopped`. A malformed fragment is a stop, never a pass. Before any state exists, the launcher refuses with `verification_error` |
| Survivor grace | After a stage exits, its process group gets `survivor_grace_seconds` to drain. Anything still alive after that is journalled, killed, and fails the stage |
| Rolling useful-mixed | Standing stop, enforced live on every RL stage (150 and 600): the launcher tails the one `reward_groups.jsonl` under the stage's `{arm_dir}` each poll and stops the arm as soon as any 100 consecutive rows hold fewer than 40 `mixed: true` (`rolling_useful_mixed_below_0_40`). A second `reward_groups.jsonl` under the arm directory, at any time (the directory is re-scanned every poll, also after the first was found), an unparseable row, or an RL stage that never wrote one there (`live_useful_mixed_unmonitored`) stops the study. The `rl150` and `rl600` templates must therefore give the wrapper (before its `--`) exactly one `--output` equal to `{arm_dir}` or under `{arm_dir}/` with no `..`; plan load refuses otherwise (`template_output_not_under_arm_dir`, or `template_missing_wrapper_separator` with no `--`). The fragment's `useful_mixed` is checked again after the stage |
| Other guards | Process-group RSS limit; every pinned file hash checked before every stage and gate, with the venv interpreter link re-resolved to the pinned binary (`interpreter_link_changed`) and `import_root` (`src`) first on `PYTHONPATH` with every importable file in it pinned (`unpinned_importable_file`); frozen environment; RL reference and every SFT arm's initial adapter equal to W; SFT-E and EG-SFT committed exposure exactly RL\*'s 600 accepted IDs in order, SFT-T's SFT-E's then the scanned sequence from row 0 (`exposure_differs_from_rl_star_accepted_sequence`, `sft_t_exposure_not_sft_e_then_scanned_from_row_0`); quiet mode waited out before every stage, a mid-stage priority drop journalled (`stage_demoted`) |
| RL continuation identity | Plan load: rl600's trainer arguments must equal rl150's minus its single `--stop-after-steps 150` (`rl600_template_differs_from_rl150`, `rl150_template_stop_after_steps_not_150`, `rl600_template_declares_stop_after_steps`), and RL must not pass `{active_cap}` (`rl_template_uses_active_cap_not_arm_cap`): rl600 resumes the 150 checkpoint and the trainer refuses a resume whose settings differ, `max_active_seconds` included. `{active_cap}` shrinks after a voided instance; `{arm_cap}` does not. The launcher's wall cap still charges voided seconds |
| Tokenizations | Plan load: rl150, rl600 and egsft must pass `--screen-tokenizations {screen_tokenizations}` (`template_screen_tokenizations_missing`); a changed file after G1 stops the next stage (`screen_tokenizations_changed`) |
| Assays | Each assay's image IDs must equal its split manifest's 200 IDs exactly, with exactly 5 (H-select) or 9 (H-gate) completions per image, each a label string or null. A partial assay stops the study: the evaluator would otherwise score a missing image as all-invalid, which lowers W or a comparator and loosens G2/G3a/G3b |

## State and journal

The state directory is `plan.state_directory`.

- **Single use.** A start creates the directory and refuses if it already exists. `--resume` continues only a `running` state of the same plan. A `stopped` or `completed` state never runs again.
- **Interrupted stages are never resumed or quoted** (study.json `budget_matching.caps`: "capped/interrupted arm re-run from scratch, never quoted"). Every stage attempt runs in a fresh `stages/<stage>/instance-<k>` and `arms/<stage>/instance-<k>`. On `--resume`, a stage that was started but not completed is journalled as `arm_instance_void` (and listed in `state.json` `void_instances`), then re-run from scratch in instance k + 1, from the stage's own declared start: W, or the validated parent endpoint for a continuation. Nothing of the void instance is passed to the trainer. A plan template that declares a `resume_argv` is refused. A capped arm is not re-run: the cap is a standing stop.
- **Append-only journal.** `journal.jsonl` is appended to and fsynced. `state.json` records the journal's byte length and SHA256, and a resume refuses if that prefix changed.
- **Lock.** A `lock` file holds the launcher's pid and its `ps` start time. A second launcher refuses without touching the state; a lock whose pid is gone or now names another process (start time differs) is stale and removed (`stale_lock_removed`).
- **Running stage.** While a stage runs, `state.json` `running_stage` holds its process group and the group leader's start time, so a stage left behind by a dead launcher can be identified and stopped (`reap_orphan_stage`).

## Plan (`schema: freshpool-launch-plan-v1`)

| Key | Contents |
|---|---|
| `python` | Interpreter path (the venv link) |
| `python_resolved` | The binary it resolves to; pinned in `file_sha256` and re-resolved before every stage |
| `import_root` | `src` (must be first on `PYTHONPATH`, every importable file in it pinned), or null when no repository import tree is on the path |
| `quiet_file` | The quiet-mode file waited out before every stage |
| `state_directory` | Where state, journal and outputs go |
| `evaluator` | Path to the gate evaluator |
| `environment` | The frozen environment |
| `file_sha256` | Pinned file hashes. Must include the evaluator and `phase_worker.py` |
| `limits` | `minimum_free_bytes`, `rss_limit_bytes`, `survivor_grace_seconds`, `poll_seconds`, `stage_headroom_bytes` |
| `forecast_seconds` | `e0`, `p0`, `assay_h_select`, `assay_h_gate`, `rl`, `sft_e`, `egsft` (`rl` and `egsft` are the study's assumptions, for the record; the caps use the measured ones) |
| `forecast_sources` | `grpo_seconds_per_update`, `sft_seconds_per_update` (measured controlled400 values, used in the measured RL / EG-SFT forecasts) and the assumptions behind the plan's own numbers |
| `templates` | One per kind: `e0`, `assay`, `p0`, `rl150`, `rl600`, `sft_e`, `sft_t`, `egsft`. Each has `kind`, `entry`, `entry_sha256` and `argv`; a `resume_argv` is refused |
| `b4_edge_extension` | Must match the evaluator's grid |
| `w_endpoint` | `adapter_sha256` and `adapter_path` |
| `owner_patterns` | Command-line patterns that mark a second accelerator owner |
| `g0` | `manifest_dir`, `manifest_sha256`, and `receipts` for `protocol_revision`, `decision_record` and `branch` |
| `evidence_base` | Path and SHA256 of a file holding `labels`, `rows`, `a_star` and `b4_edge_extension` |

Template placeholders:

- `{python}`, `{stage_id}`, `{arm}`, `{stage_dir}`, `{arm_dir}`, `{fragment}`, `{seed}`, `{active_cap}` (cap minus this arm's voided seconds), `{arm_cap}` (the arm's full 1.25x cap, constant across instances), `{w_adapter}`
- `{screen_tokenizations}` after G1
- `{manifest_<name>}` for each G0 manifest file
- `{run_id}`, `{lr}`
- `{split}`, `{adapter}` (the endpoint's `adapter_path`), `{split_manifest}` for assays
- `{resume_checkpoint}` for the RL 150→600 continuation (pass it to `trainer_fragment.py --continue-output-from`)
- `{accepted_manifest}`, `{scanned_manifest}`, `{parent_checkpoint}` and `{t_rl}` for the SFT arms (SFT-T passes `{parent_checkpoint}` as the trainer's `--continue-from-checkpoint`, after the wrapper's `--`, and `{t_rl}` as its `--stop-after-active-seconds`)

An unknown placeholder stops the study.

## Fragments

Each stage entry writes `{fragment}` as JSON.

| Stage | Fragment contents |
|---|---|
| Training (`trainer_fragment.py`) | `status: completed`, `run_id`, `arm` (RL, SFT-E, SFT-T or EG-SFT), `lr`, `seed`, `updates`, cumulative `committed_active_seconds`, `step_committed_active_seconds` (contiguous, nondecreasing, ending at `updates` with the committed seconds), `adapter_sha256` (final adapter weights file), `adapter_path`, `reference_sha256` (W's weights SHA256: RL's KL reference, SFT arms' initial adapter), `useful_mixed`, `checkpoint`, `committed_prompt_ids` (the receipt's exposure, one ID per update). RL also writes `accepted_manifest` and `scanned_manifest` (scanned IDs a prefix of `S_scan_seed42`) |
| Assay (`h_assay.py`) | `run_id`, `split`, `adapter_sha256` (must equal the endpoint's), `predictions` (every split-manifest ID, 5 or 9 completions each) and `eos_fraction`, in the evaluator's format |
| P0 (`p0_screen.py`) | The evaluator's `g1` document, seed 42 under W, whose `scan_order` must equal the `S_scan_seed42` manifest and which holds 400 journal entries and 128 rollout groups |
| E0 (`e0_gate.py`) | `checks` (exactly the twelve), `sampler_teacher_max_abs_logprob_difference`, the altered-reference result, and measurements: per invocation the child's peak RSS, peak MPS driver and tensor bytes and wall seconds; per update committed seconds; generated tokens; checkpoint bytes |

The launcher assembles the evaluator's evidence document from these fragments plus `evidence_base`.

## G0 receipts (resolved)

- Protocol: `docs/aircraft-rl-protocol-freshpool-v1-activated.json`, a separately identified
  activated copy of the unactivated revision. `verify_protocol` requires: `activated: true`; an
  `activation.activates` naming the committed revision by path and SHA256, whose own id is
  `fgvc-aircraft-freshpool-screened-grpo-v1` and `activated: false`; a different `protocol_id`;
  every other top-level field equal to the revision's; both salts equal to the revision's literal
  id; frozen gates equal to the committed candidate's; all three files committed. G0 also requires
  the manifest's `hash_protocol_id` to be that literal.
- Decision record: D11's `Retired:` entry; D24 records this launch package (numbered D24 because open PR #16 adds D22 and D23).
- Branch: `docs/aircraft-freshpool-branch-receipt-v1.json` (B0-B6 with numbers from
  `reports/2026-10-17-controlled400-native-val/comparison.json`; D_req = 0.05, full mode).

## Blind spots

- **Process groups.** A descendant that calls `setsid()` escapes the RSS sum and the survivor check.
- **Owner programs.** Only Python programs are read as owners; a non-Python accelerator owner (an `ollama` server, a native app) is not seen.
- **Image check.** G0 checks that every H and scan image file exists and that their bytes hash to the plan's `g0.images_sha256` (`launch.images_digest`), at launch and resume only; a file replaced mid-run is the trainer's row identity to catch.
- **Sampling.** RSS, disk and MPS driver use are sampled. The launcher polls every `poll_seconds`, and the phase worker samples MPS every 0.5 s.
- **Owner detection.** It matches whole command-line words only, so an owner whose command line names none of the patterns as a word (`python -c "from vision_rl import vlm_local_training"`) is not seen. The test and dry-run refusals judge a test module by its own source text.
- **Cap measurement.** The cap is measured on wall time, which is an upper bound on active time. Parents are charged their committed active seconds and voided instances their wall seconds, in both lineages (RL 150 -> 600; SFT-E -> SFT-T). The measured RL forecast uses P0's acceptance under W; acceptance can fall as the policy improves, which the 0.25x margin has to absorb.
- **Live useful-mixed.** The tail sees rows at the poll interval and only under `{arm_dir}`; rows the trainer has not yet flushed are seen at the next poll or after the stage exits.
- **Trust.** Fragment contents are trusted as written, apart from the identity, assay-shape and lineage checks above. The adapter hash on every assay must match its endpoint's.
- **Assay chunking.** The trainer refuses TRAIN inference on more than 32 images per call, so `h_assay.py` and `p0_screen.py` work in chunks of 32 in manifest order and seed each sampled chunk at its start. The draws are deterministic for a fixed manifest, but they are not one continuous RNG stream over the split.
- **Physical journals.** The wrapper reads per-step seconds and useful-mixed flags from the trainer's append-only `optimizer_audit.jsonl` and `reward_groups.jsonl`. These are exact only because no stage is ever resumed after an interruption. A mismatch with the receipt refuses.
- **Not run on the pinned model.** `p0_screen.score_candidates` loads the model the way `predict_train_subset` does and calls the trainer's `_batch_gold_probability`. The unit tests replace it; the CPU dry run executes it on the tiny random model only, never on Qwen2-VL-2B or MPS.
- **Cap and start-up.** The 1.25x cap is launcher wall time, process start-up included. At study scale (T_RL about 13,338 s, unverified) start-up is small against the 0.25x margin; on the dry run's seconds-long arms it is not, which is why the dry run widens the factor.
- **SFT-T on an all-accepted scan.** The trainer refuses a continuation whose TRAIN identity equals the parent's. If RL\* accepted every candidate it scanned, the scanned and accepted manifests are byte-identical and SFT-T would refuse. G1 requires acceptance of at least 0.30, not 1.0, so this needs 600 accepted with no rejection; it is not handled.
- **Lock and orphans.** A pid plus `ps` start time identifies a process; the orphan reap acts only on the recorded group whose leader has the recorded start time (or, with the leader gone, whose every member names the state directory).
- **Import tree.** Only `import_root` is checked for unpinned importable files; `runs/vlm-local-libs/site-packages` on `PYTHONPATH` is covered by the controlled400 runtime pins, not by a full scan.

## Signal-limit v1 (`aircraft-rl-next-study-v1`, pre-registered, not activated)

`signal_limit_launch.py` sequences `docs/aircraft-rl-next-study-v1.json` (P0-I, Stage 0,
Stage A, Stage B) on the same `Launcher`, so every guard above applies unchanged. Nothing of it
has run on the model. What is new:

| File | Role |
|---|---|
| `stage_table.py` | The stage table (P0-I, Stage 0, Stage A's 13 arms, Stage B), the init endpoint role of every stage (I, W, I\*), the update counts (Stage A 600; RL screen 100, final 300) and `a_floor = N x 1.25 / |S_b|`. `verify_against_study` cross-checks it against the study JSON field by field; any disagreement refuses the launch |
| `signal_limit_launch.py` | The launcher (`schema: signal-limit-launch-plan-v1`). Init endpoints are pinned per stage (`init_endpoints.I` / `.W` in the plan, checked against the study's I hash and W prefix; I\* pinned in the state from Gate A's `i_star`, by adapter SHA256); a plan-wide `init_endpoint`/`w_endpoint` refuses. Update counts come from the plan's `updates`, which must equal the study's unless the plan is a dry run. c(G) is written to the state from Stage 0 before any RL forecast reads it (RL cap = 1.25 x c(G) x 300). Every H assay carries the constrained-greedy pass and enters the look ledger. Gates and selections come from `scripts/evaluate_signal_limit_gates.py` (contract in the module docstring) |
| `uw_stage_a.py` | Optional UWaterloo executor for Stage A (`"stage_a_execution": "uwaterloo"` and a `uw` block): runs the independent Stage A arms and their H-select assays concurrently through `~/.claude/bin/uwrun`, one per host slot, in uwrun's host-local `/var/tmp` run directories, through the same `prepare` / `finish` as a local stage (pins, cap with voids, fragment validation), and journals every stage with its uwrun name, host and hardware record. Host choice is an allowlist with per-host run, free-RAM, free-disk (>= 5 GB) and provisioning checks; a lost host interrupts resumably and the instance is re-run from scratch; a remote timeout is the arm cap; a fetched adapter whose bytes differ stops. What ships is an allowlist (pinned files, `src`'s .py/.json, the stage's inputs, all hash-pinned in `pins.json`, itself pinned on the command line); the arm clock reaches `state.json` every 5 s; every run it stops or finishes is `uwrun rm`ed; `uwrun` must be absolute and plan-pinned; a dry-run plan cannot use it. The docstring lists its blind spots |
| `p0_screen.py` | G from the profile (refuses G outside {4, 8, 16}); `--predict-groups` adds predicted acceptance and mean u per G from p only (P0-I); `--choose-groups 8,16 --a-floor F` picks the smallest qualifying G and samples there (P0') |
| `h_assay.py` | Assay G stays 4 whatever the training G (`--training-group` only records it); `--label-trie T --label-trie-sha256 H` adds the constrained greedy pass, refused unless the trie hashes to H and the trainer's `predict_train_subset` takes `label_trie` |
| `trainer_fragment.py` | Arms A1-A5 and LIM-T (either limit objective's mode); their fragments, and an RL fragment whose trainer recorded a `loss_type`, carry `mode`, `init_sha256` and the RL profile (loss_type, scale_rewards, beta, group, reward). Freshpool fragments are unchanged |
| `phase_worker.py` | `--remote-cpu`: the UW path's real cpu run, with `UWRUN_THREADS` torch threads (`VISION_RL_TORCH_THREADS` must equal it; that variable refuses a stage anywhere else, a deliberate legacy refusal) |

The Stage 0 entry (cost probe and the G8 / beta 0 / dr_grpo resume proof) is the plan's
`stage0` template; the launcher requires its fragment to report the checks named in
`signal_limit_launch.STAGE0_CHECKS` under each G's own key (`checks["8"]`, `checks["16"]`),
sampler-teacher agreement <= 0.001 and `c_seconds_per_accepted_update` for G8 and G16.

Tests: `tests/test_signal_limit_launch_units.py` (stage table, counts, P0 and assay options,
profile fields, `--remote-cpu`, the UW helpers) and `tests/test_signal_limit_launch.py` (the
whole sequence with a fake stage entry, a contract evaluator or the real one when present, and a
fake `uwrun`: concurrency per host, a lost host and its resume, a remote timeout, a transfer
mismatch, launch failure and every host refusal).

### Every stage on UWaterloo (`"execution": "uwaterloo"`) and the production package

`"execution": "uwaterloo"` runs **every** signal-limit stage (P0-I, Stage 0, Stage A, P0', RL,
SFT-E, SFT-T, LIM-T and every H assay) through `uw_stage_a.UWRunner`, on the `uw` block's one
device class (`device: cpu`, or `cuda` with `min_free_vram_mib` and `per_host: 1`). The Mac
only sequences, gates (the evaluator's numpy bootstrap) and journals; quiet mode is not waited
out. What the all-stages path adds over the Stage-A-only one:

| Concern | How |
|---|---|
| Forecasts | the plan's `forecast_seconds` are the UW ones; `uw.forecast_seconds` must be absent |
| Continuations | RL 100 -> 300 and SFT-E -> SFT-T ship the fetched parent checkpoint; any adapter or checkpoint whose parent holds `local_training.json` ships with its whole run directory (the trainer reads that record) |
| RL manifests | the fetched accepted/scanned manifests name remote image paths; on fetch they are rewritten to the local bundle's copies (`*.local.jsonl`), refusing any path outside the run's inputs |
| Live useful-mixed | every `uw.mixed_poll_seconds` the remote `reward_groups.jsonl` is mirrored under the local arm directory (`uwrun ssh ... tail -c +N`) and read by the same `MixedMonitor`; a window below 0.40 stops the remote run; after the fetch the mirror must be a prefix of the fetched file and is completed for the late poll |
| Threads | the plan's Mac thread pins (`OMP/MKL/OPENBLAS_NUM_THREADS=2`) are not exported remotely; uwrun's `--threads` stand |
| Busy hosts | `min_idle_threads`: a host qualifies only with `nproc - load1` at least that many |
| CUDA | `phase_worker.py --remote-cuda`: `ACCELERATE_TORCH_DEVICE=cuda`, `CUDA_VISIBLE_DEVICES=0`, one device, peak CUDA allocation journalled; `p0_screen.py` and `h_assay.py` accept `--device cuda` |
| Host setup | there is no `uw.setup` hook (it would run before the pin check or rewrite the bundle unseen after it); hosts are provisioned beforehand with `signal_limit_production/provision_host.sh` and the probe checks the result |

`signal_limit_production/`: `prepare.py --write|--check` (activated study and protocol copies
differing only in `status`/`activated`/`activation`, the G0 manifests, the evidence base, the
plan and its sha256); `start.py [--dry-run|--resume]` (frozen worktree, root's decision-log
line naming both activated copies' sha256, `prepare.py --check`, disk, single-use state, the
host probe, `--verify-only`; then `production/chain.py`); `provision_host.sh` (venv + model on
a host's `/var/tmp`, preflight receipt); `measure_units.sh` (unit costs on TRAIN rows);
`smoke_stage0.sh` (Stage 0's entry at G8, two steps, on a host).
