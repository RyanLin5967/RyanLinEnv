# Named open-weight baseline, plan v2: Qwen3.8-27B bf16 on CPU (UNACTIVATED)

The machine-readable plan is [aircraft-named-baseline-plan-v2.json](aircraft-named-baseline-plan-v2.json).
It is **pre-registered and unactivated**. The runner refuses it until root review commits
an activated copy (`activated: true`, `activation.activated_at`, `activation.approved_by`)
and passes that copy's SHA256 on every invocation.

## Why a v2, and what changes

v1 ([plan](aircraft-named-baseline-plan-v1.md)) ran `mlx-community/Qwen3.8-27B-8bit` with MLX on
the owner's Mac. Run 2 finished discovery twice at 34/100 and full-VAL invocations 1-3. It was
then paused because MLX wired about 37.5 GB and made the Mac lag. MLX runs only on Apple
hardware, so the UWaterloo hosts cannot run it.

v2 runs the **same base model with its original weights**: `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the same upstream revision the 8-bit conversion names.
It loads the 18 bfloat16 safetensors shards (55.6 GB) with transformers on CPU, on the
ram-pool hosts (iccad1, iccad3 and iccad5, each with about 490 GB free).

**v2 does not use the v1 model artifact.** These things change:

| | v1 | v2 |
|---|---|---|
| Weights | 8-bit affine conversion (group 64) | the author's bf16 safetensors |
| Kernels | MLX Metal | PyTorch 2.14.1 CPU |
| Sampler random numbers | MLX RNG | `torch.Generator`, seeded with the same per-image seed |

Per-image outputs are therefore not expected to match v1. A v1-to-v2 difference cannot be
attributed to any one of these three changes.

**No v1 run-2 result is merged into v2.** That covers the discovery runs, the full-VAL
invocations 1-3 and the ledgers. v2 regenerates every image in its own journals. v1 numbers can
be cited only as v1 numbers.

This change edits `scripts/aircraft_named_baseline.py`, and v1 run 2 recorded that file's hash
in its ledger. A v1 resume from this tree is therefore refused (it fails closed). Resuming v1
needs a checkout of `da4e401`.

## What is held identical

v2 reuses the same runner, `scripts/aircraft_named_baseline.py`. The plan's `model.engine`
(`transformers-cpu`) selects the new engine. The following are unchanged from v1:

- Stage manifests, discovery selection and IDs, and the full-VAL `ids_sha256`.
- The prompt (`prompt_sha256`) and the parser (`parser_source_sha256`).
- The terminated-only rule.
- The scorer: equal-class `mean(diag(row_normalized_confusion))`, with micro (image) accuracy
  reported beside it.
- The Hoeffding interval and the discovery gate.
- Ledger budget semantics.

The new engine, `scripts/aircraft_named_baseline_hf.py`, runs the **same journaled core**
(`local_vlm._run_locked`). That core reserves each image durably before generation. An
interrupted reservation becomes an unknown outcome and is never regenerated, and completed
answers are never repeated. The engine also reuses `final_answer`, `prepare_local` pixel
identity, the native-instruct profile and the shared bottom-20 transform.
`local_vlm.py` and `local_vlm_execution.py` are byte-identical to v1's pins.

The engine refuses an image if the chat-template output differs from the formatted prompt the
MLX engine recorded for every v1 image (`c4409cac…`). The text the model sees is therefore the
v1 text.

The native-instruct sampler is re-stated in torch from mlx-vlm 0.6.8, in this order:

1. Presence penalty of 1.5, applied once per distinct token among the last 20 context tokens.
2. Nucleus filtering at 0.8 on the untempered distribution.
3. Top-k of 20.
4. Temperature of 0.7.

transformers runs its own decode loop in greedy mode over a one-hot of that sample, so no
transformers warper changes the logits. A test checks the distribution against an independent
numpy re-statement.

## Execution

- **Weights:** the files are downloaded to each host's `/dev/shm` and rehashed before every load
  against the committed integrity record. That record is
  [aircraft-named-baseline-v2-model-integrity.json](aircraft-named-baseline-v2-model-integrity.json),
  pinned by sha256. Each LFS file in it equals the Hub's `lfs.sha256`, and each small file equals
  the Hub's git blob id from [the revision listing](aircraft-named-baseline-v2-hub-metadata.json).
  The weights are not kept on the NFS home because it has a per-user quota (about 25 GB,
  inferred from a "Disk quota exceeded" error at that usage) that cannot hold 55.6 GB.
- **Shards:** there are 3 shards, assigned round-robin in author order: shard k takes
  `ids[k::3]`, which gives 34/33/33 discovery images and 1111 full-VAL images per shard. Each
  shard has its own journal and ledger and runs on one host through `uwrun --timeout <wall cap>`.
- **Merge:** `merge` and every report refuse the following cases:
  - an image recorded in two shards, or in a shard that does not own it;
  - a shard whose run identity binds other rows;
  - a shard directory that is not in the plan.

  `merge`, `compare` and `compare-endpoints` also refuse while any image is unattempted or
  unknown.
- **Gate:** discovery runs first, then full VAL (3,333 images). The full-VAL projection is
  calculated per shard, because shards run in parallel. It uses the worst discovery shard's
  mean latency and overhead.
- **Comparisons:** paired image bootstrap within each class, 20,000 draws, seed 2718. The
  comparisons are against W (`common`), RL400 (`rl-exposure`) and the strongest declared SFT
  (`sft-1e5-time`, fixed by the bundle's pre-declared rule). Each is read from its hash-pinned
  native-VAL journal.

## Smoke and cost

SMOKE_SECTION

## Never touched

The plan never touches the sealed TEST split. The host dataset copy excludes `sealed/` and
every TEST audit file. The smoke reads only four pinned TRAIN images.
