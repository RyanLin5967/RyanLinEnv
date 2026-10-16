# Reviewed Linux CPU continuation

This manual workflow continues the **same** two CPU7946 trajectories after both
remote first-four hardware gates pass. It does not restart those updates or make
a held-out claim. No continuation has been dispatched by this implementation.
The first failed run, `38881659770`, stopped before either model worker started;
it is ineligible. The second run, `38882469176`, hit the unchanged memory guard
while loading weights, before generation or updates; it is also ineligible.
A successful replacement using the prospective CPU loader amendment must be
reviewed and pinned.

The workflow has four finite jobs: shared warm assay, parallel `grpo` and
`dr_grpo` continuation, and offline comparison. Every job uses a standard public
`ubuntu-24.04` runner with a six-hour limit. Only output artifacts are uploaded,
with one-day retention. Inputs, images and base weights are never uploaded.
Expired parent artifacts fail closed; the workflow does not regenerate them.

Before dispatch, provide a reviewed JSON envelope containing:

```json
{
  "schema_version": 1,
  "repository": "RyanLin5967/RyanLinEnv",
  "parent_run_id": 123,
  "parent_commit": "EXACT_40_HEX_PARENT_COMMIT",
  "first4_files": {
    "scripts/aircraft_ci_experiment.py": "EXACT_SHA256",
    "scripts/aircraft_cpu_loader.py": "EXACT_SHA256",
    ".github/workflows/aircraft-cpu-experiment.yml": "EXACT_SHA256",
    "docs/aircraft-cpu-experiment-plan.json": "EXACT_SHA256"
  },
  "artifacts": {
    "grpo": {"id": 1, "sha256": "EXACT_ZIP_SHA256", "bytes": 1},
    "dr_grpo": {"id": 2, "sha256": "EXACT_ZIP_SHA256", "bytes": 1}
  }
}
```

The placeholders deliberately cannot run. `validate --plan FILE` checks only the
schema and performs no network or model work. The parent commit may precede the
continuation workflow commit, but all four first-four files must match that
parent and the current checkout exactly. The original source archive, every
member, native worker, 37-module frozen package, task metadata and descriptor
remain unchanged between that successful parent and its continuation. Current
project sources are not substituted into the image.

Before any API download or checkpoint restoration, the actual loader bootstrap
must match both its direct source pin and the complete policy in the pinned
descriptor. There is no fallback for older envelopes without the loader pin.
The parent CI state, every phase guard and each `loader.json` must agree on this
identity. Receipts must show completed CPU `pread` calls for both fixed base
shards, the reviewed package versions and implementation hashes, and restored
process-local alias/environment state. Missing, partial or changed receipts fail
closed even when an artifact's file inventory has been recalculated. The same
checks apply to warm and continued-arm handoffs; the descriptor and loader are
mounted read-only for every container stage.

The controller verifies the parent repository, workflow path, manual first
attempt, successful conclusion, commit, artifact names/IDs, archive hashes and
sizes through GitHub's API. It bounds ZIP extraction and rejects traversal,
links, devices, duplicates and oversized payloads before restoring training
state. Each arm's complete file inventory, hardware result and original
checkpoint-four receipt must agree. This is trusted checkpoint restoration from
a specifically reviewed owned workflow, never a loader for arbitrary uploads.
[GitHub artifact metadata and download API](https://docs.github.com/en/rest/actions/artifacts).

The restored directory retains optimizer, scheduler, RNG, reference adapter,
exposure lineage and checkpoint-bound active time at their original virtual
paths. The unchanged native receipt verifier checks them again before training.
Original totals stay **64 updates and 7,200 committed active seconds per arm**.
The controller exposes no first-four restart, changed learning rate, precision,
examples, loss or native budget. Any failure is terminal and preserved.

Before either arm resumes, the common warm model must finish fresh CPU greedy
predictions and G4 samples with seeds 42 and 314 on all 32 fixed TRAIN assay
images. The exact shared output archive is passed to both arms by its artifact
ID and SHA256. Each arm then runs the original `*-full` worker followed by its
three endpoint assays. The frozen comparator runs only after both finish.
These are TRAIN measurements; no validation or test images are downloaded.

The original 43,200-second aggregate physical-phase budget includes both first
four phases and the shared warm assays. The remainder is divided equally into
two disjoint reservations before parallel continuation. Each arm's supervisor
subtracts its own later phases from its reservation. Final merging counts shared
phases once, verifies identical common evidence and rejects aggregate excess.
Unused time is not transferred or silently reset. Build/transfer/job wall time
is reported separately and remains bounded by the six-hour job timeout.

Docker retains CPU2, 13-GiB memory/no extra swap, no network, read-only inputs,
dropped capabilities and the original process-tree/disk/output guards. It runs
as the host output owner's UID/GID and exercises the actual mounted directory
before model work. No token enters the container. Artifact redirects are fetched
without forwarding the GitHub credential.

The workflow accepts only attempt one. Its fixed parent-specific run name and
history audit reject another invocation for an already consumed parent,
including a failed continuation. There is no automatic dispatch, retry or
replacement trajectory. These controls and light tests are preparation evidence;
actual Linux continuation and resume parity still require the prospective run.
