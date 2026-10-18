# Named baseline v2: author Qwen3.8-27B on CPU, sharded (PRE-REGISTERED, UNACTIVATED)

Machine-readable plan: `docs/aircraft-named-baseline-plan-v2.json`. Runner:
`scripts/aircraft_named_baseline_v2.py`. Engine: `scripts/aircraft_named_baseline_hf.py` (kept out of `src/vision_rl`, whose modules other pre-registered plans hash). The committed plan
is unactivated and the runner refuses it. A run needs an activated copy whose SHA256 is passed on
every invocation, and whose `sources_pinned` has no null entry.

## Why a v2

v1 ran `mlx-community/Qwen3.8-27B-8bit` with MLX on Ryan's Mac. Discovery gave 34/100 in both
runs. Full VAL was paused after invocations 1-3 because MLX wired about 37.5 GB and made the Mac
lag (`runs/aircraft-named-baseline-v1-run2`). MLX only runs on Apple hardware, so the UWaterloo
hosts cannot run v1.

## Identity: what changed relative to v1

| | v1 | v2 |
| --- | --- | --- |
| weights | `mlx-community/Qwen3.8-27B-8bit@815b83c0`, 8-bit affine, group 64 | `Qwen/Qwen3.8-27B@1d4bf0f2`, the author's bf16 safetensors (55.6 GB) |
| relation | lossy conversion of upstream `1d4bf0f2` | that upstream revision itself |
| engine | mlx-vlm 0.6.8 on Metal | transformers 5.20.0 + torch 2.14.1+cpu, float32 compute over the exactly upcast bf16 weights |
| random stream | MLX | `torch.Generator`, with the same per-image seed |
| sampling definition | mlx-vlm chain | the same chain, reimplemented in `local_hf.mlx_filter` |
| prompt, parser, scorer | eight-endpoint evaluator | the same; v1's `stage_inputs`, `canonical` and the evaluator scorer are called directly |
| templated prompt | sha `c4409cac…` | must equal `c4409cac…`, or the image is refused |
| hosts | Mac | iccad1/3/5, with one shard per host |

So v2 runs the same base model at different numerics. A v2 number is never labelled as the 8-bit
model. **No v1 run-2 result is merged with v2.** v2 never reads, resumes, copies or pools the v1
discovery outputs (34/100) or full-VAL invocations 1-3. It regenerates all 100 discovery images and
all 3,333 VAL images. v1 numbers may appear next to v2 numbers only under their own label.

v1's files (`local_vlm.py`, `local_vlm_execution.py`, `scripts/aircraft_named_baseline.py`) are
left byte-identical. The paused v1 run hashes them into its identity and ledger, so any edit would
make it refuse to resume. A test asserts that they still match v1's pins.

## Engine

- **Snapshot.** The engine verifies the snapshot before every load against
  `docs/qwen3.8-27b-hf-integrity-1d4bf0f2.json`. That record was taken from the Hub API: sha256
  for the 19 LFS files, git blob SHA-1 for the other 13, 32 files in total.
- **Sampling.** The engine reproduces the mlx-vlm 0.6.8 chain:
  1. Presence penalty of 1.5 on the unique tokens among the last 20 generated tokens.
  2. log-softmax.
  3. Top-p 0.8 on the untempered distribution.
  4. Top-k 20.
  5. A categorical draw at temperature 0.7.

  The engine draws the token inside a logits processor and hands `generate()` a one-hot row.
  `generate()` runs greedy, with the author's `generation_config` replaced by a neutral one, so
  no Hugging Face warper or default can change the token. After each image, the emitted tokens
  are compared with the drawn tokens, and any difference is refused.
- **Stop rule.** `finish_reason` is `stop` only when the last token is an author eos token, and
  that eos token is not part of the text. Otherwise it is `length`, at the native 16,384-token
  ceiling.
- **Dtype.** float32. The author bf16 weights are upcast exactly at load, and only the
  activations are computed at higher precision. The ram-pool CPUs (Xeon E5-2667 v2: AVX, but no
  AVX2, FMA or AVX512-BF16) have no native bf16, so torch emulates it. In the bf16 smoke, TRAIN
  image 0447760 was still unfinished after about 3,120 s; in float32 the same image took 505 s on
  iccad3 and 860 s on iccad1. Dtype is part of the run identity, so changing it needs a new plan
  version.
- **Threads.** Threads affect speed only, so they are not part of the identity. The thread count
  is recorded in each ledger start and in `load_info`.

## Shards and merge

- **Assignment.** Author-order position i goes to shard i mod 3. Each shard has its own journal,
  ledger and budget under `stage-<stage>/shard-<k>-of-3/`.
- **Merge refusals.** `merge` unions the shard journals. It refuses any of the following:
  - an image journaled outside its shard, or twice;
  - a shard directory that is not `shard-k-of-3`;
  - shards whose run identities differ in anything other than their rows and shard index;
  - any missing or unknown image.

  It scores only a complete stage.
- **Gate from discovery to full VAL.** The gate is projected per shard:
  `ceil(3333/3) × discovery mean latency + chunks × per-invocation overhead`. The shard cap is
  `min(max_projected_shard_active_seconds, 2 × projection)`.

## Stages and analysis

1. **Discovery.** The same 100 IDs as v1 (D13 salt), sharded 34/33/33. The report gives the
   Hoeffding 95% interval. Discovery is a validity and sizing diagnostic, not a score.
2. **Full native VAL.** All 3,333 images in author order, run only when the discovery gate says
   `proceed`.
3. **Compare.** The baseline is compared with three arms from
   `runs/aircraft-native-val-activation-v1/bundle`, with `comparison.json` sha `a49482bc…`:
   - **W**: the `common` arm.
   - **RL400**: the `rl-exposure` arm.
   - **strongest SFT**: `sft-1e5-time`, the `strongest_declared_sft` chosen by the evaluator's
     pre-declared rule.

   `export-arm` reads each arm through the evaluator's own hash-chained `Journal`, re-parses the
   answers and refuses any disagreement. Two metrics are reported:
   - equal-class accuracy, using `compare_classification`'s paired bootstrap;
   - image accuracy, using a paired image bootstrap.

   Both use 20,000 draws and seed 2718.

## Measured cost

Source: `reports/2026-10-18-named-baseline-v2-smoke`, float32 with 32 threads, 4 TRAIN images
per host. Both hosts were shared with other jobs, so these are per-host figures.

| host | s/image | mean |
| --- | --- | --- |
| iccad1 | 859.7, 831.2, 177.1, 266.6 | 533.7 |
| iccad3 | 504.6, 596.6, 207.9, 283.2 | 398.1 |

- **Where the time goes.** Prefill takes 91-96% of each image on iccad3. Decode runs at 2.9-6.1 s per
  token, and answers are 5-7 tokens. Peak RSS was 164.7 GB.
- **Full-VAL projection with 3 shards (1,111 images each):**

  | s/image assumed | hours per shard | wall clock |
  | --- | --- | --- |
  | 398 | 123 | 5.1 days |
  | 466 (pooled mean) | 144 | 6.0 days |
  | 534 | 165 | 6.9 days |

  One host alone would need 369-494 h. Discovery takes about 3.8-5.0 h per shard.
- **Budgets per shard.**
  - Discovery: wall cap 162,400 s, 3 invocations.
  - Full: wall cap 216,400 s, 24 invocations, 28 calendar days.
  - Gate ceiling: 1,209,600 s per shard.
  - Each wall cap includes 100,000 s, enough for one native 16,384-token answer at 6.1 s per
    token, so a long answer is not truncated.
- **Not a forecast.** VAL images reach 1.9 MP, and the largest smoke image was 1.67 MP. The full
  cap comes from the discovery measurement.

## Not claimed

- Bit-identical outputs to v1 or to any GPU or MLX run.
- The strongest Qwen3.8 profile: xhigh thinking remains unmeasured.
- Anything about TEST.
