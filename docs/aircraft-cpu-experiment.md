# Manual Linux CPU reproduction

The `Aircraft CPU first-four reproduction` workflow runs the two frozen CPU7946
arms, `grpo` and `dr_grpo`, on separate standard public `ubuntu-24.04` runners.
It is a TRAIN-only mechanism and hardware check. It makes no held-out, frontier,
or improvement claim. Neither job is dispatched automatically on a push or PR.

Both jobs use the exact 64 TRAIN images, disjoint 32-image TRAIN assay selection,
100-label vocabulary, common SFT adapter `f05840be…`, seed 42, FP32, G4, 256-token
ceiling, and two CPU threads from the frozen Mac plan `7946c228…`. Released TRL's
`loss_type` is the sole scientific intervention. Linux results are independent
of Mac results; wall time and stochastic samples must not be pooled across them.

The first release permits exactly four optimizer updates per arm. Each native
checkpoint retains its original 64-update and 7,200-active-second budget. The
frozen worker performs its existing numerical, reference, exact warm-weight,
saved/reloaded output, and projected-throughput checks. Reward and accuracy are
not first-four promotion criteria. A failed arm does not cancel the other arm.

GitHub documents standard public Linux runners as 4 CPU, 16 GB RAM and 14 GB
storage, with free standard-runner usage for public repositories.
[Runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The original Mac 18-GiB available-memory and 20-GiB free-disk requirements cannot
apply to that runner. This separately recorded Linux operational profile uses:

- Docker CPU limit 2, memory limit 13 GiB and no additional swap allowance.
- Process-tree RSS limit 12 GiB; host available memory at least 13 GiB before
  launch and 1 GiB while running; no observed host swap growth.
- One GiB free-disk reserve, two GiB maximum outputs, 30-minute first-four wall
  limit, and a 100-minute whole-job timeout including dependencies and transfer.
- Networking disabled, read-only source/base/input mounts, no credentials in
  the container, and a dedicated writable checkpoint/evidence mount. The
  container runs with the host output owner's UID/GID and checks a real
  directory/file create, write, fsync, read and deletion before starting a worker.

These limits are prospective. Linux pretrained loading and four-update training
have not yet been measured. The earlier Linux image build and no-model offline
preflight do not prove that FP32 training fits this runner. The first-four job
must stop on resource failure; it must not switch precision, model, examples,
group size, or objective to obtain a passing result.

The first actual run, `38881659770`, built both images and verified the inputs,
then both containers failed creating `output/phases` before worker creation.
With all capabilities dropped, UID 0 could not write the runner-owned directory.
The prospective infrastructure amendment sets `--user` to the host UID/GID and
records a same-container filesystem receipt before any model work. No permission
broadening, capability addition, frozen-source change, or scientific adjustment
is part of this correction. The failed run is retained; a new manual dispatch is
an explicitly recorded infrastructure attempt, not continuation of a trajectory.

The committed JSON descriptor pins a 1,658,880-byte source/metadata archive and
a separately reviewed small warm-adapter archive. The source archive contains
all 37 frozen package files, the unmodified original worker/comparator/plan,
and task metadata; it contains no image, base model, optimizer or credentials.
The source archive is prepared locally with:

```sh
python3 scripts/aircraft_ci_experiment.py pack
```

The repository owner publishes both reviewed assets under the research release
named in the descriptor before dispatch. This command does not publish or fetch
anything. The warm archive includes its exact three original identity files;
the descriptor validates every archive member and the native common weight SHA.

Each runner downloads one 4,429,621,382-byte public Qwen snapshot at revision
`895c3a49bc3fa70a340399125c650a463535e71c` and only 27,237,237 bytes of the selected
96 TRAIN JPEGs from the pinned public mirror. Every file is size/SHA256 checked;
native model-visible cropped RGB hashes are checked before loading weights.
The original manifests' absolute paths are virtual paths inside the container,
so their bytes and the trainer's continuation identity remain unchanged.
No validation or sealed test images are downloaded.

Each job builds `Dockerfile.vlm` from the hash-pinned lock and frozen source,
then calls the original `run.worker` under a Linux resource supervisor. The Mac
plan, watcher and local training remain untouched. The job records CPU hardware,
image ID, source/input identities, resource samples, native receipts, and actual
outputs, including failures. Artifacts retain only the output directory for one
day; no base weights or input JPEGs are uploaded.

Continuation is intentionally not exposed in this first workflow release.
After both first-four gates pass, a later reviewed manual stage must restore both
native checkpoints from these exact repository/workflow runs, verify every
artifact and source hash, collect the fresh shared Linux warm assay, and resume
the original trajectories to 64 without resetting optimizer, reference, RNG,
exposure or accumulated active time. Native pickle checkpoint files are trusted
training state, not public numeric-model submissions, and must never be restored
from arbitrary runs or unreviewed uploads. Endpoint assays and the unchanged
frozen comparator are subsequent finite stages, not automatic retries or a grid.
