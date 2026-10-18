# Named open-weight baselines on native VAL, on free CI runners (pre-registered, unactivated)

The machine-readable plan is [aircraft-ci-named-baselines-plan-v1.json](aircraft-ci-named-baselines-plan-v1.json).
Its status is **UNACTIVATED. It must not run.** Every job of the workflow
[`aircraft-named-baseline-ci.yml`](../.github/workflows/aircraft-named-baseline-ci.yml) refuses it
before downloading anything. A dispatch needs a root-approved copy with `activated: true` and an
`activation` record, committed first, and that copy's SHA256 as a workflow input.

This plan is separate from the [27B MLX plan](aircraft-named-baseline-plan-v1.md), which stays
unchanged and still waits for the Mac GPU. It reuses that plan's inputs block, discovery sample and
stage manifests verbatim, and it reuses its `stage_inputs` function in code.

## What is measured

| key | model | revision | dtype | why |
|---|---|---|---|---|
| `qwen2-vl-2b-instruct` | Qwen/Qwen2-VL-2B-Instruct | `895c3a49…` | FP32 | The untrained base of every controlled400 arm, with no adapter. It shows what the warm start and each later arm added. |
| `qwen2.5-vl-3b-instruct` | Qwen/Qwen2.5-VL-3B-Instruct | `66285546…` | BF16 | A newer, larger open-weight model that still fits a 16 GB runner. |

Each model file is pinned by size and SHA256. For Qwen2-VL-2B the hashes are the frozen evaluator
plan's `base.file_sha256`, and the sizes are from the Hugging Face API; the total of 4,429,621,382
bytes equals the evaluator's own figure. For Qwen2.5-VL-3B, the small files were downloaded and hashed
on 2026-10-18. The two safetensors shards are pinned by their Hugging Face LFS `oid`, which was not
hashed locally; every CI download is hashed and refused on any mismatch. The Qwen2.5-VL-3B licence is
the Qwen Research License (non-commercial research use).

- **Prompt, parser, decoding and metric are the frozen evaluator's.** The prompt is
  `prompt_text(spec, "answer_only")` over all 100 labels (SHA256 `859dd366…`), and the parser is
  `parse_label(…, allow_reasoning=True)`, applied only to a terminated generation. Decoding is greedy
  through `vlm_local_evaluation._generation_kwargs` (256 new tokens). Each prediction is
  `vlm_local_evaluation.Backend.predict`, unchanged, bound to a base model with no adapter. The metric
  is `classification_comparison`'s equal-class accuracy, and invalid or missing answers stay in
  their class denominators.
- **Profile:** the evaluator's profile (50,176–200,704 pixels, 2,048 prompt tokens, answer only).
  The dtype is set per model.
- **Images:** native-VAL JPEGs from the same pinned mirror that the fourshot candidate used
  (`Shiym/ViT-FineTune@16c63682…`). Only the stage's VAL IDs are requested, and each one is checked
  against `encoded_sha256` and `encoded_bytes`. The shared input builder then rehashes every JPEG,
  and each image's model-visible pixels are rehashed again before it is generated. No TEST ID is
  ever formed.
- **Comparisons:** `compare_classification(gold, endpoint, baseline)`, with 20,000 draws and seed
  2718, against W (`common`), RL400 (`rl-exposure`) and the strongest SFT (`sft-1e5-time`).
  Delta is the endpoint minus the named baseline. The endpoints' per-image canonical predictions are
  in
  [`reports/2026-10-18-ci-named-baseline-endpoints/endpoint-predictions.json`](../reports/2026-10-18-ci-named-baseline-endpoints/endpoint-predictions.json),
  pinned by SHA256. Each source journal's SHA256 equals the committed `comparison.json`, and the
  recomputed macro accuracy matches it for all three arms.

## Sharding

- **Discovery** is the same 100-image, one-per-class sample as the 27B plan, in 4 shards of 25.
  It is a cost and validity diagnostic and never a benchmark score.
- **Full** covers all 3,333 images in author order. Its shard count is fixed by the discovery gate:
  `ci_shards.full_shard_count(3333, mean latency, overhead, 19,800 s, 600 s, safety 1.5)`. The gate
  stops on any refusal, on a largest image latency above 300 s, or when more than 256 shards would
  be needed. The workflow input must equal this count. Three places recompute it from the committed
  discovery report: the matrix job, every shard and the merge.
- Shards are contiguous slices of the stage order. A shard starts an image only while its elapsed
  time plus 600 s fits within 19,800 s. The job timeout is 345 minutes, under GitHub's 6 h job limit.
  At most 20 shards run in parallel. The model and the images are downloaded once per dispatch and
  passed to the shards through a run-scoped cache.
- **The merge refuses on any gap.** It refuses when a shard is missing or duplicated, when an image
  is missing, foreign or has no completed outcome, when a shard is partial, or when a shard carries
  a different plan, model, stage, shard count, run ID or run attempt. It also re-parses every
  response and refuses any stored label that differs. The run-attempt rule means that re-running
  only the failed shards cannot be merged. A dispatch is all of its shards or nothing.
- Every dispatch is reported. Each (model, stage) gets at most two complete-stage dispatches. Because
  decoding is greedy, a re-dispatch cannot sample a more favourable answer.

## Deviations, disclosed

- The runs use a CPU with 4 threads, not MPS with 2. The last bits of the numerics can differ from
  the endpoints' runs.
- Qwen2.5-VL-3B runs in BF16, its checkpoint dtype, because FP32 weights (about 15 GB) do not fit a
  16 GB runner.
- The per-image deadline is 600 s instead of 120 s, because CPU generation is slower. This is a wall
  guard and does not change decoding.
- The gate has no validity floor. Invalid answers are part of a zero-shot baseline and count as
  wrong. The 27B plan's 0.9 floor protected GPU time, and CI time is free here.

## Activation checklist (all required, none done)

1. Root review approves this plan. An activated copy is committed, and its SHA256 is recorded in the
   decision log.
2. Dispatch `stage=discovery`, `shards=4` for one model, then commit its merged report at
   `reports/aircraft-ci-named-baselines-v1/<model>/discovery-merged.json`.
3. Dispatch `stage=full` with the shard count from `gate`. Commit the merged report and its
   comparisons before interpreting them.

```bash
gh workflow run aircraft-named-baseline-ci.yml -f plan=docs/ACTIVATED.json -f plan_sha256=SHA \
  -f model=qwen2-vl-2b-instruct -f stage=discovery -f shards=4
python3 -I scripts/aircraft_ci_named_baseline.py gate \
  --plan docs/aircraft-ci-named-baselines-plan-v1.json --model qwen2-vl-2b-instruct
```
