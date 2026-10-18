# Named open-weight baseline on native VAL, sharded on free CI: Qwen2.5-VL-3B-Instruct (pre-registered, unactivated)

The machine-readable plan is [aircraft-named-baseline-ci-plan-v1.json](aircraft-named-baseline-ci-plan-v1.json).
Its status is **UNACTIVATED**. The workflow
[`aircraft-named-baseline-ci.yml`](../.github/workflows/aircraft-named-baseline-ci.yml) and the runner
[`scripts/aircraft_named_baseline_ci.py`](../scripts/aircraft_named_baseline_ci.py) refuse the committed
plan for every VAL stage. A VAL run needs a root-approved copy with `activated: true` and an
`activation` record, committed before dispatch; its path and SHA256 are dispatch inputs. The plan is
append-only: corrections go to `amendments[]` before any inference they could affect, and any
scientific change is a new `plan_id`.

## Why this exists

The Mac GPU is held by a live study, and the 27B MLX baseline
([aircraft-named-baseline-plan-v1](aircraft-named-baseline-plan-v1.md)) needs it. Free GitHub-hosted
runners for a public repository cost nothing. A 3B model in bfloat16 (7.51 GB of weights) fits a
16 GB CPU runner. This adds a second named comparator without touching the Mac.

## What is measured

- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct` at revision `66285546…`, official bfloat16 safetensors,
  zero-shot, no adapter, zero trainable parameters. The frozen evaluator's own `MODEL_REVISIONS`
  already pins this revision, so its `_verified_model_directory` accepts exactly this snapshot. All 13
  files are pinned by SHA256 in
  [the model manifest](aircraft-named-baseline-ci-v1/inputs/qwen2.5-vl-3b-instruct-model-source.json).
  License: Qwen Research License (non-commercial research and evaluation).
- **Prompt, decoding, parser and metric are the eight-endpoint evaluator's code, not a copy of it.**
  Each job materialises the 41 evaluator sources from commit `85f0865` and refuses any byte that
  differs from the SHA256 list in
  [the evaluator plan](../reports/2026-10-17-controlled400-native-val/plan.json). It imports only those
  files and rehashes every `vision_rl` module that ran. Prompt: `prompt_text(spec, "answer_only")`
  (SHA256 `859dd366…`) through `_processor`/`_inputs`. The chat template file is byte-identical to
  Qwen2-VL-2B's. Inference: `vlm_local_evaluation.Backend.predict` (greedy, 256 new tokens; an answer
  counts as terminated only if it ends in EOS). Parser: `parse_label(…, allow_reasoning=True)`, applied
  to terminated answers only. Metric: `classification_comparison` equal-class accuracy; invalid answers
  stay in their class denominators.
- **Disclosed differences from the endpoints, all fixed before any inference:**
  - bfloat16 on CPU instead of float32 on MPS, because float32 does not fit 16 GB;
  - `repetition_penalty` set to 1.0. The model's own config sets 1.05, which would penalise copying
    label tokens from the prompt. The endpoints' base config uses 1.0, so the override makes decoding
    pure greedy, the same as the endpoints;
  - 4 torch threads instead of 2.
- **Images:** the original native-VAL JPEGs, fetched one by one from the pinned Hugging Face mirror
  (`Shiym/ViT-FineTune@16c63682…`), the same mirror the four-shot candidate used. Each file must match
  the author manifest's bytes and SHA256, and its model-visible pixels must match `model_rgb_sha256`.
  Only IDs in the committed native-VAL author manifest are ever fetched. No TEST image or TEST metadata
  is downloaded.

## Stages

0. **Plumbing (pull requests only).** The same shard code runs on three pinned TRAIN images. It is
   never scored and needs no activation. It proves that downloads, the frozen import, model loading
   and `Backend.predict` work on a free runner, and it records setup and latency. The same job checks
   that the frozen `compare_classification` reproduces the committed endpoint report from the
   committed endpoint export.
1. **Discovery.** The same 100 protocol-defined, one-per-class IDs as the 27B plan (`c8d73a4c…`),
   read from that plan and checked against it. It uses 4 declared shards (1–20 allowed). The report
   gives the accuracy, the Hoeffding 95% interval (±0.1358), the valid fraction, latency and setup
   time. Its `discovery-receipt.json` is committed and feeds the full gate. It is not a benchmark
   score.
2. **Full VAL.** All 3,333 images in author order (`c33a8794…`). The `matrix` command refuses it
   unless the committed discovery receipt belongs to this activated plan, completed all 100 images and
   has at least 90% valid answers. It also refuses unless the shard count N satisfies the
   pre-registered rule
   `N ≥ ceil(3333 / floor((19800 − 900 − max discovery setup s) / (1.5 × discovery mean latency s)))`.

## Sharding and the merge

Image *i* in author order goes to shard *i* mod N. Each shard is one GitHub job, kept under the 6 h
limit:

- the job timeout is 355 min;
- a shard starts an image only if its elapsed time plus the 900 s per-image deadline fits within
  19,800 s;
- a shard stops at its first generation error, interrupted image or budget stop, and fails.

Nothing is retried or regenerated. The merge job runs even when shards fail. Before it scores
anything, it refuses on any of the following:

- a shard index without a completed record, or any unexpected artifact;
- any image that is missing, duplicated, outside its partition or outside the manifest;
- any outcome that did not complete;
- any shard from a re-run attempt, another run or another commit;
- any shard whose identity differs from the others: model, runtime, profile, generation config,
  prompt or frozen sources;
- any canonical label that differs from the frozen parser applied to the raw response.

The unit tests (`tests/test_aircraft_named_baseline_ci.py`) cover these refusals, including a missing
shard at both the function and the command level. A mutation that drops the missing-shard check fails
three of them.

**Runs policy:** every dispatched VAL run is reported, failed or not. A failed run is never patched by
re-running shards. A new dispatch is a new run, and the report lists it next to the failed one.

## Comparison (full VAL only)

`compare_classification(gold, Qwen2.5-VL-3B, endpoint)` runs with 20,000 paired bootstrap draws and
seed 2718 against three endpoints:

- **W** (`common`, 0.6784);
- **RL400** (`rl-exposure`, 0.6982);
- the **strongest SFT** (`sft-1e5-time`, 0.7539).

The endpoint predictions come from
[endpoint-predictions.json](aircraft-named-baseline-ci-v1/inputs/endpoint-predictions.json).
[`scripts/export_native_val_endpoint_predictions.py`](../scripts/export_native_val_endpoint_predictions.py)
exported them from the evaluator journals whose SHA256 the committed report records. The export is
accepted only if two checks pass:

- **At export time:** for RL400 and the strongest SFT against W, the recomputed `input_sha256`
  equalled the report's. This hash covers gold labels and both arms' outcomes for all 3,333 images.
  A planted single-label flip changes it.
- **At merge time, before any baseline comparison:** the frozen `compare_classification` must
  reproduce the report's full comparison for those arms (scores, per-class differences and bootstrap
  intervals).

## Limits

- This is a development comparison on VAL. VAL was also the selection set for the strongest SFT.
  Nothing here touches the frozen native TEST.
- A named baseline claim requires all 3,333 predictions, with zero non-completed outcomes, from one
  merged run.
- bfloat16 CPU numerics are not shown to equal float32 or GPU numerics. The claim label says
  "bfloat16 on CPU".
- Latency on shared runners is not measured yet; plumbing and discovery measure it. A shard count
  that is too small fails closed as a gap.

## Activation checklist (all required, none done)

1. The PR plumbing job has passed on the commit to be activated.
2. Root review approves. An activated copy is committed, and its SHA256 goes in the decision log.
3. Discovery is dispatched with that copy. Its merged `discovery-receipt.json` is committed before
   full VAL is dispatched.

## Dispatch (prospective; refused until activation)

```bash
gh workflow run aircraft-named-baseline-ci.yml -f stage=discovery -f shards=4 \
  -f plan=docs/ACTIVATED-COPY.json -f plan_sha256=SHA
gh workflow run aircraft-named-baseline-ci.yml -f stage=full -f shards=N \
  -f plan=docs/ACTIVATED-COPY.json -f plan_sha256=SHA \
  -f discovery_receipt=docs/aircraft-named-baseline-ci-v1/discovery-receipt.json
```
