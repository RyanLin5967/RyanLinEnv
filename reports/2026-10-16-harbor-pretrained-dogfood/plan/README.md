# Pretrained Harbor dogfood: local variant v3 (not executed)

This is the third local variant of `runs/aircraft-harbor-dogfood-v1-plan`. It implements step 5 of
`runs/aircraft-harbor-dogfood-v1-plan/claude-audit-v1/dogfood-design.json`. Nothing in v3
has started Colima, run Docker, built an image, loaded a model or trained. The draft is untouched,
and `test_draft_untouched` checks that against the draft's own `preparation-evidence.json`.
`local_launch_reviewed` is false.

v3 is v2 with operational wall changes only. Training, data, model, loader and the
`SystemExit` fix are unchanged.

## Why v3: the failed v2 attempt

`runs/aircraft-harbor-dogfood-local-v2-plan` (plan sha `a2cd71df…`) was executed once and failed
with `driver_error: "ValueError: A native model phase failed its actual resource gate"`. Its plan
and its output in `runs/aircraft-harbor-dogfood-local-v2/` are preserved evidence and are never
edited or re-run. `derive_plan.py` re-reads them and pins them in `plan.json`
`previous_attempt.v2`.

The v2 SFT probe passed. It committed 1 update with 83.13 s active, its loader receipt was
`completed` (so the v2 `SystemExit` fix worked), and the stage took 921.26 s. In `rl-exposure`, GRPO
update 1 committed with 752.76 s active and a nonzero gradient, and
`after_train_diagnostic_predictions.json` was written. Then the draft's flat 1800 s stage wall in
`runtime.py` killed the stage at 1801.29 s, during the post-train reload/audit replay. The guard
recorded exit -2, `stop_reason: "Disk/output/wall resource guard"`, and a peak tree RSS of
12,138,778,624 B.

`timing.json` holds the per-phase breakdown. `derive_plan.py` `measure_v2_timing` re-measures it
from the preserved receipts and from file mtimes relative to the guard start, and refuses to write
the plan if it differs. The `rl-exposure` timeline, in seconds since the guard started:

| phase | seconds |
|---|---|
| start, model load, before-train diagnostic (written at 223.01) | 223.01 |
| before-train gold NLL, sampler parity, GRPO step 1 (checkpoint at 1258.89; train_runtime 747.7, active 752.76) | 1035.88 |
| after-train diagnostic (written at 1420.09) | 161.20 |
| save the final adapter | 0.35 |
| post-train audit (gold NLL, reload, reload NLL, replay) until the kill at 1801.29 | 380.85 |

The actor.log traceback puts the kill inside the replay `_evaluate`. The projected need is
1962.49 s: the elapsed time at the kill plus one more after-train-diagnostic duration, because the
replay re-runs that diagnostic. That is an unverified assumption, and 1801.29 s is the hard lower
bound. For the probe, the fixed non-step cost is 837.98 s (921.26 minus train_runtime 83.28), and
its post-train audit took 429.67 s.

The rule is that every wall must be at least 2x its measured or projected need, and every outer
wall must equal its inner walls plus 300 s:

| wall | v2 | v3 | need (s) | v3/need |
|---|---|---|---|---|
| `rl-exposure` stage | 1800 | **4000** | 1962.49 projected | 2.04 |
| `sft-exposure` stage (same argv as the probe) | 1800 | **2000** | 921.26 | 2.17 |
| `sft-resource-probe` stage | 1800 | **2000** | 921.26 | 2.17 |
| `sft-time` stage | 3600 | 3600 | 1592.07 | 2.26 |
| verify stage | 1800 | 1800 | at most 138.69 (unmeasured; bound) | 12.98 |
| Harbor `[agent] timeout_sec` (runs the 3 training stages in series) | 7200 | **9900** | 4475.82 | 2.21 |
| `local_run.py` probe cap (`local.probe_wall_seconds`) | 2100 | **2300** | stage wall + 300 | |
| `local_run.py` main cap (`local.main_wall_seconds`) | 9300 | **12000** | agent + verify + 300 | |
| README outer `timeout` | 11700 | **14600** | probe + main + 300 | |

The handoff proposed 3600 s for `rl-exposure`. That is below 2x even the 1801.29 s lower bound, so
the rule gives 4000 s. The `sft-time` need is the probe's fixed cost plus the RL active time.
Because it resumes `sft-exposure`'s step, its new steps' active time is below 752.76 s, and it
reaches that target in 10 of its 32 steps at 83.13 s per step. The Harbor agent timeout was a wall
that the handoff did not name. At 7200 s it was 1.61x the three stages' combined need, and it would
have killed the agent before the new stage walls could fire.

The three wall changes are declared patches over the draft:
- `runtime-stage-walls.patch` is applied after `runtime-systemexit.patch`, which is kept. It
  replaces `3600 if stage == "sft-time" else 1800` with `STAGE_WALL_SECONDS[stage]`.
- `prepare-task-agent-timeout.patch` changes `prepare_task.py`'s `timeout_sec=7200` to `9900`.
- The driver caps live in `plan.json` `local`. `local_run.py` `require_wall_nesting` refuses a plan
  whose outer walls do not enclose the inner ones plus 300 s. `verify_bundle` refuses a bundle
  whose packaged `task.toml` agent timeout differs from the plan.

## Why v2: the failed v1 attempt

`runs/aircraft-harbor-dogfood-local-v1-plan` (plan sha `277bb9b2…`) was executed once and failed
with `driver_error: "ValueError: Scoped loader identity/completion differs"`. Its plan and its
output in `runs/aircraft-harbor-dogfood-local-v1/` are preserved evidence and are never edited or
re-run. `derive_plan.py` re-reads them and pins them in `plan.json` `previous_attempt.v1`.

The SFT probe itself trained correctly: 1 committed update, guard `status: completed`,
`exit_code: 0`. The failure was in `runtime.py` `child()`. It runs the frozen trainer with
`runpy.run_module` inside `scoped_loader`, and the trainer ends with `raise SystemExit(main())`.
A clean run therefore raised `SystemExit(0)` through the context manager. The pinned loader
recorded `status: failed, error_type: SystemExit`, and `assert_probe` rejected the receipt.

The v2 fix is `runtime-systemexit.patch`, which is applied to `runtime.py` only. Inside the
`with` block it catches `SystemExit` around `run_module`. Exit code 0 or None continues, so the
loader records `completed`. Any other code is re-raised, so the loader records `failed`.
`aircraft_cpu_loader.py` is pinned and shared with CI, and it is unchanged. `verify-child` does not
have this defect: `builder.verify_submission()` returns a report and never raises `SystemExit`.
A test checks this against both the draft and the patched runtime.

All v3 outputs go to `runs/aircraft-harbor-dogfood-local-v4` (jobs
`aircraft-harbor-dogfood-local-sft-probe-v4` and `aircraft-harbor-dogfood-local-v4`).

Scope: an integration-only plumbing dogfood on native TRAIN images. It is not held-out, not a
benchmark and not an RL-vs-SFT result. All 7 fixture images are in the warm adapter's `train400` SFT
set, so the verifier's accuracy is recorded but is never a pass criterion.

## Files

| file | what | change from draft |
|---|---|---|
| `aircraft_cpu_loader.py`, `profile.json` | copies | none: byte-identical |
| `prepare_task.py` | draft plus `prepare-task-agent-timeout.patch` | Harbor `[agent] timeout_sec` 7200→9900. `max_steps` still reaches the task via `plan.json` → `make_task` → `contract.json` → `runtime.checked_native_argv` |
| `runtime.py` | draft plus `runtime-systemexit.patch`, then `runtime-stage-walls.patch` | catches the trainer's `SystemExit` inside `scoped_loader` (v2); per-stage walls from `STAGE_WALL_SECONDS` (v3). `derive_plan.py` and a test apply the patches in order to the draft and compare bytes |
| `runtime-systemexit.patch`, `runtime-stage-walls.patch`, `prepare-task-agent-timeout.patch` | the only changes to draft files | pinned in `support_sha256` and `copies[...].patch_sha256` |
| `timing.json` | v2 per-phase timing and wall needs | re-measured by `derive_plan.py` from the preserved v2 evidence; pinned in `support_sha256` |
| `plan.json` | derived by `derive_plan.py` | `training.max_steps` 4→1; local scope and status; `local_launch_reviewed=false`; `launch_reviewed` removed; first4-parent gates off; `workflow_minutes` removed; added `local` (with the wall fields), `build_files`, `model_identity`, `success_criteria`, `derived_from`, `changes`, `copies`, `pin_verification`, `previous_attempt` (`v1` and `v2`); every output path is `-v3`; `support_sha256` pins the 2 copies, the 2 patched files, the 3 patches, `timing.json` and `local_run.py` |
| `derive_plan.py` | regenerates `timing.json` and `plan.json` | recomputes every pin it can from live files, checks the patched files as draft plus patches, re-reads the v1 and v2 failure evidence, refuses any wall below 2x its need, and refuses to write on any mismatch |
| `local_run.py` | local driver | see below |
| `test_local_plan.py` | fixture tests | v2 tests carried over, plus the v3 wall tests (see Tests) |

Pins recomputed by `derive_plan.py`: the 37 frozen sources, 6 host sources, `uv.lock`, the first4
descriptor/plan/helper, `data_spec`, the 7 JPEGs plus their model-visible pixel hashes, both local
release tarballs, the warm adapter/config/record, the 3 build files, and the model manifest, marker
and revision. One pin cannot be recomputed on the host: `loader.source_sha256` names installed files
inside the image. It is checked at run time by `assert_runtime_receipts`.

`local_run.py`:
- copies `sha`, `stop_owned_containers`, `artifact_file`, `assert_runtime_receipts` and
  `assert_probe` verbatim from the draft `run.py`;
- takes `tree_bytes` and `run_harbor` from `run-output-size-race.patch`. A test re-applies the
  patch and compares the result with `fixed_run.py`;
- has `assert_complete` require `global_step == 1`;
- refuses to launch unless `local_launch_reviewed` and `cgroup_policy_reviewed` are both true,
  `sys.platform == "darwin"`, `DOCKER_HOST == unix:///Users/idide/.colima/vision-harbor/docker.sock`,
  `DOCKER_CONTEXT` is unset and the working directory is the repo root. There is no GitHub or parent
  requirement;
- before any Docker call, checks every pin and checks that the prepared bundle was packaged from
  this exact `plan.json` sha (compose label, runtime pins, `max_steps=1`, model identity,
  `FROM vision-rl:aircraft-cpu7946-continuation`);
- refuses (`require_wall_nesting`) unless the agent timeout, probe cap and main cap each enclose
  the walls inside them plus 300 s, and checks the packaged `task.toml` agent timeout;
- runs the Harbor SFT probe itself (2300 s wall), then `assert_probe`, then the main trial (12000 s
  wall). Each runs under `run_harbor`: 2 GiB host output cap, 1 GiB free-disk floor, and in its
  `finally`, `docker stop` only for containers labelled `vision-rl.harbor-plan=<plan sha>`.
  `output/` must be new, so it never retries;
- writes C1–C4 to `runs/aircraft-harbor-dogfood-local-v4/output/dogfood-local-result.json` and
  exits non-zero unless every criterion passes and the driver raised nothing. An empty tree fails
  every criterion.

There are two deviations from the design's wording, both forced by the frozen schemas.
- C1: `training-evidence.json` `model_source` has no revision field. The revision `895c3a49…` is
  checked through the packaged `contract.json`, whose manifest and marker the runtime re-hashes.
- C2: `parameter_delta_l2` is written at train end into
  `arms['rl-exposure'].optimizer_audit`, not into the step-1 jsonl row. The row is checked for
  `optimizer_steps_observed`, `nonzero_gradient_steps` and `first_nonzero_update.changed_tensors…`.

## Command sequence (steps 3–9)

Preconditions (design steps 1–2) come first. Ryan must give an explicit go-ahead, because the image
build and the CPU training fall under his no-builds-on-Mac order. Host free memory must be at least
22 GiB, with at least 15 GiB free on disk. The controlled400 continuation must still be advancing;
do not touch it. Then start the VM with
`colima start --profile vision-harbor --cpu 4 --memory 16 --disk 20`.

```sh
cd /Users/idide/RyanLinEnv
export DOCKER_HOST=unix:///Users/idide/.colima/vision-harbor/docker.sock
unset DOCKER_CONTEXT
L=runs/aircraft-harbor-dogfood-local-v4-plan
R=runs/aircraft-harbor-dogfood-local-v4

# 3. Kernel guard preflight. Pass: 13958643712, 0, 0; MemAvailable >= 14 GiB; root free >= 4 GiB.
timeout 120 docker run --rm --network none --memory 13312m --memory-swap 13312m python:3.12.13-slim \
  sh -c 'cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max /sys/fs/cgroup/memory.swap.current; grep MemAvailable /proc/meminfo; df -B1 /'

# 4. Build the frozen-source runtime image (context = 3 pinned build files + 37 frozen sources).
#    runtime.py is not in this image; prepare_task.py copies it into the task image, so the v1
#    build (image sha256:402dd239…) has the same inputs. Rebuilding is still the reviewed path.
mkdir -p $R
timeout 60 .venv/bin/python -I -B $L/local_run.py build-context --context $R/build-context
timeout 1200 docker build -f $R/build-context/Dockerfile.vlm -t vision-rl:aircraft-cpu7946-continuation $R/build-context
timeout 120 docker run --rm --network none vision-rl:aircraft-cpu7946-continuation   # offline preflight

# 5. Review this directory, then the reviewer flips the gate (this changes the plan sha, so it
#    MUST happen before packaging), and re-runs the tests.
timeout 120 .venv/bin/python -I -B $L/derive_plan.py --local-launch-reviewed
timeout 300 .venv/bin/python -m pytest -q $L        # must stay green after the flip
.venv/bin/python -I -B $L/local_run.py check        # launchable: true, plan_sha256 = container label

# 6. Package the task inside the runtime image (network none, repo read-only).
mkdir -p $R/prepared
timeout 300 docker run --rm --network none --cpus 1 --memory 1g --memory-swap 1g --user $(id -u):$(id -g) \
  -v /Users/idide/RyanLinEnv:/Users/idide/RyanLinEnv:ro \
  -v /Users/idide/RyanLinEnv/$L:/prepared-source:ro \
  -v /Users/idide/RyanLinEnv/$R/prepared:/Users/idide/RyanLinEnv/$R/prepared:rw \
  vision-rl:aircraft-cpu7946-continuation python -I /prepared-source/prepare_task.py \
  --root /Users/idide/RyanLinEnv --output /Users/idide/RyanLinEnv/$R/prepared/bundle \
  --base-image vision-rl:aircraft-cpu7946-continuation

# 7 + 8. SFT probe, then the main trial and separate verifier, both under the driver's caps.
#        Harbor builds the thin task image FROM the step-4 image. Exit 0 only if C1-C4 all pass.
timeout 14600 .venv/bin/python -I -B $L/local_run.py execute
cat $R/output/dogfood-local-result.json
# Progress: watch .../runtime/<stage>/resources.jsonl advance, not pid presence.
# Read-only re-evaluation of an existing run: .venv/bin/python -I -B $L/local_run.py criteria

# 9. Return memory to the host.
colima stop --profile vision-harbor
```

The outer `timeout 14600` covers the probe (2300 s), the main trial (12000 s) and 300 s of slack.
The driver's own wall caps fire first. Inside them, the Harbor agent timeout (9900 s) and then the
per-stage walls fire first.

## Open, unmeasured risks (carried from the design and verification)

- The `rl-exposure` need of 1962.49 s is a projection. It assumes the reload replay costs no more
  than the after-train diagnostic did (161.20 s). Only the 1801.29 s lower bound is measured.
- The same diagnostic work ran slower in v2's `rl-exposure` than in the probe: the after-train
  diagnostic took 161.20 s against 128.77 s, a factor of 1.25. The cause is unmeasured (GRPO's
  extra reference adapter, or host load). If the SFT arms in the main trial slow down by that factor,
  the `sft-exposure` and `sft-time` walls would cover 1.74x and 1.81x of their need rather than 2x.
- `sft-exposure`, `sft-time` and the verifier have never run locally. Their needs come from the
  probe.
- Peak tree RSS reached 12,165,832,704 B in the probe and 12,138,778,624 B in `rl-exposure`, against
  the 12 GiB (12,884,901,888 B) RSS guard: about 0.7 GB of headroom. Host free memory fell to
  5,989,818,368 B during `rl-exposure`.
- Linux arm64 `randperm` parity is unverified (step 1 is derived as image 0143341).
- The step-6 mount exposes the whole repo, sealed data included, read-only to a no-network
  packaging container (per the design). A narrower mount set is untested.
- `pids_limit 128` was exercised for GRPO on GitHub but not for SFT under Harbor.

## Tests

`timeout 300 .venv/bin/python -m pytest -q runs/aircraft-harbor-dogfood-local-v4-plan`

The v3 wall tests do the following:
- check the literal wall values and that each change is declared;
- check that `runtime.STAGE_WALL_SECONDS` equals the plan, that the verifier wall stays 1800 s,
  and that the draft still has the flat walls;
- check `timing.json` spot values against the v2 receipts, read independently of `derive_plan.py`;
- check that every wall is at least 2x its need, and that the outer walls enclose the inner ones;
- check that `require_wall_nesting` and `execute` refuse the v2 values before any launch;
- check that the packaged `task.toml` agent timeout is 9900 and that `verify_bundle` rejects a
  mismatch;
- check that re-derivation refuses a stale `timing.json` or a wall below 2x its need.

The `SystemExit` regression tests run `runtime.child()` for real. They put a stub
`vision_rl.vlm_local_training` on `sys.path` that opens both stub shards and then raises
`SystemExit(0)`, `SystemExit(None)` or `SystemExit()` (receipt `completed`), or `SystemExit(3)`,
`SystemExit(1)` or `SystemExit("fatal")` (re-raised, receipt `failed`, `error_type: SystemExit`).
Each case runs under two loaders. One is a stub loader with the pinned receipt contract. The other is the real
pinned `aircraft_cpu_loader.py` with only its Linux and version probes stubbed. The same scenario
against the draft `runtime.py` reproduces the v1 failure, which shows the test detects the defect.

These tests use fixtures only and do not run Docker, Colima or a model. Packaging is checked by
running `prepare_task.py` into a temp directory with `.venv-detection/bin/python`, because `.venv`
lacks `safetensors`. `-X importtime` showed that packaging imports `safetensors.numpy` and no torch.
