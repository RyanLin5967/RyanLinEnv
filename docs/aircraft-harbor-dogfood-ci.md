# Pretrained Harbor dogfood on a GitHub-hosted runner

Workflow: `.github/workflows/aircraft-harbor-dogfood.yml` (manual `workflow_dispatch` only,
`ubuntu-24.04` standard public runner, `timeout-minutes: 360`). Driver:
`scripts/harbor_dogfood_ci.py`. Test: `tests/test_harbor_dogfood_ci.py`.

It reproduces the local PASS in `reports/2026-10-16-harbor-pretrained-dogfood/` (plan
`62fe9ff1…`) off the Mac. Scope is unchanged: integration only, native TRAIN fixtures, not
held-out, not a benchmark, not an RL-vs-SFT result. Status: **not yet dispatched**. A
`workflow_dispatch` workflow can only be run once the file is on the default branch.

## What is reused (nothing is reimplemented)

| Step | Reused code |
|---|---|
| Plan directory | tracked `plan-source.tar.gz`, every file checked against `plan-SHA256SUMS` (macOS AppleDouble `._*` sidecars are skipped, never extracted) |
| Host helpers, `pyproject.toml`, `uv.lock`, `Dockerfile.vlm`, VLM lock | `git show 1323eb1:<path>` (the plan's `source_commit`), each checked against the plan's pins. `pyproject.toml` on main differs (ruff config only, since `2a54bb6`), so main's copy is never used |
| Frozen 37-file source, data spec, warm adapter `f05840be…`, model, 7 fixture JPEGs | `scripts/aircraft_ci_continue.setup_runtime` with the CPU7946 descriptor `docs/aircraft-cpu-experiment-plan.json` |
| Runtime image | same `setup_runtime` build (`vision-rl:aircraft-cpu7946-continuation`, 1200 s cap) |
| Harbor host | `uv sync --extra harbor --locked --no-install-project` of the pinned project; Harbor version must equal the plan's `0.24.0` |
| Kernel preflight, packaging | plan README steps 3 and 6, the same pass lines (cgroup `13958643712 0 0`, host MemAvailable ≥ 14 GiB, root free ≥ 4 GiB) and the same packaging command. Deviation: the step 3 cgroup read runs in the runtime image instead of `python:3.12.13-slim` |
| Probe, main trial, separate verifier, C1–C4 | `local_run.execute`, with only `require_local_launch` (darwin + Colima socket) replaced by a GitHub-runner gate that keeps the review, `max_steps == 1` and single-attempt checks |
| Criteria check | `local_run.assert_probe`, `assert_complete`, `evaluate`, plus the plan's own `local_run.py criteria`; also reads the driver's `output/dogfood-local-result.json` and exits 1 unless that file exists with `pass: true` and `driver_error: null` and all of C1–C4 pass. A missing file (outer SIGINT, refusal before launch) or a driver error (non-zero Harbor exit, wall or disk guard) fails the receipt even when the evidence on disk would pass |

The plan hard-codes `/Users/idide/RyanLinEnv`: the model bind source in the packaged
`docker-compose.yaml`, the fixture paths, and the driver paths. The runner therefore creates
that directory (`sudo install -d /Users/idide`) and stages the inputs there. Editing the pinned
plan bytes would have changed the plan sha and the container ownership label.

## Where the inputs come from (legal and reproducible)

- Warm adapter `f05840be…`: this repository's public release `aircraft-cpu7946-inputs-v1`, asset
  `aircraft-common-warm-f05840be.tar.gz`, size and SHA-256 pinned in the descriptor, with the
  Apache-2.0 `LICENSE`, `NOTICE` and `MODEL_CARD.md` it ships.
- Base model: `Qwen/Qwen2-VL-2B-Instruct` at revision `895c3a49…` from Hugging Face (Apache-2.0),
  each file pinned by size and SHA-256.
- Fixture images: the 7 are among the 96 CPU7946 TRAIN JPEGs. They come from the public HF mirror
  `datasets/Shiym/ViT-FineTune` at revision `16c63682…`, pinned per file. FGVC-Aircraft is for
  non-commercial research use. Images are downloaded on the runner and never uploaded.
- No secrets, paid runners or hosted model APIs are used.

## Evidence artifact

`aircraft-harbor-dogfood-evidence` (14-day retention) is a basename allowlist
(`EVIDENCE_NAMES` in the driver: named receipts such as `guard.json`, `loader.json`,
`report.json`, `resources.jsonl`, plus `*.log` and `invocation-*.json`) applied to the run
output, plus `ci-stage.json`, `ci-criteria.json` and `preparation.json`. Anything under a
`final_adapter/`, `checkpoint-*` or `ref/` directory (adapter, reference-model and tokenizer
copies) is skipped, as is every name not on the allowlist: safetensors, `.bin` pickles,
generated `.py`, images, and unlisted JSON such as tokenizer configs. Files over 64 MiB, or past
512 MiB in total, are skipped too. Every skipped path is listed in `evidence-manifest.json`.
`collect` exits 1 if it copied nothing, so a manifest alone never reads as evidence. Applied
read-only to the local v4 output, the collector kept 81 files (4,597,080 B) and skipped 123
(116 under model directories, 7 not allowlisted).

## Memory sizing for a 16 GB runner, and the risk

Measured on GitHub `ubuntu-24.04` in run 38889151068 (first-four `ci-state.json`, before the
model launched): MemTotal 15.61 GiB, MemAvailable 14.19 and 14.26 GiB, SwapTotal 3.0 GiB.
The same run's GRPO phase (`audit-v1/measured.json`) peaked at 11,795,185,664 B tree RSS. Its
minimum host available memory was 3.90 GB.

The plan keeps every guard. The container cgroup is 13 GiB (`--override-memory-mb 13312`,
`memswap_limit` equal, so cgroup swap is 0). The tree RSS guard is 12 GiB. Each child needs host
MemAvailable of at least 13 GiB before it launches, and at least 1 GiB while it runs. The stage
step refuses before any model work unless the cgroup reads exactly `13958643712 0 0`, host
MemAvailable is at least 14 GiB (plan README step 3's pass line, stricter than the runtime's
13 GiB per-child gate) and root has at least 4 GiB free.

Risks, all unmeasured on this path:
- **Headroom is thin.** Locally the SFT stages peaked at 12.16–12.19 GB tree RSS, about 0.7 GB
  under the 12 GiB guard. The runner adds dockerd, the Harbor host process and the runner agent
  outside the cgroup. First-four measured 14.19–14.26 GiB available, so the 14 GiB stage gate
  passes with about 0.2 GiB to spare and each child's 13 GiB gate with about 1.2 GiB.
  x86 RSS for the SFT arms has never been measured. A guard trip ends the run as a FAIL with
  evidence, never as a retry.
- **C2 can fail by sampling.** The local GRPO group was mixed (`[0, 0, 1, 1]`). x86 sampling
  differs from the aarch64 VM, and a uniform-reward group gives zero task gradient. C2 then fails
  by design. That is a reported result, not a bug to retry around.
- **Time.** The driver caps are 2300 s for the probe and 12000 s for the main trial, inside
  `timeout -s INT -k 120 15300`, with staging under `timeout 3000`. Deviation: the plan's
  `local.outer_timeout_seconds` is 14600 with the default TERM. CI uses SIGINT so `run_harbor`'s
  `finally` still stops Harbor and its containers, and 15300 s adds room for the pre- and
  post-trial work outside the two driver caps. An outer timeout leaves no
  `dogfood-local-result.json`, so the criteria check fails. That fits the 360-minute job. GitHub x86
  measured about 525 s per GRPO update (first-four), against 530 s locally.
- **Single attempt.** `GITHUB_RUN_ATTEMPT` must be 1, so a re-run is refused.
