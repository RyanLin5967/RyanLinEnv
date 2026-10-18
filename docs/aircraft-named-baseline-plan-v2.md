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
| hosts | Mac | iccad1/3/5; 6 host-independent shards that the hosts pull in turn |

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
  1. Presence penalty of 1.5 on the unique tokens among the last 20 tokens of mlx-vlm's
     history. That history is the whole prompt plus the generated tokens. For a prompt over
     2,048 tokens, which mlx-vlm prefills in chunks, it is the last prompt token plus the
     generated tokens. So the first answer tokens are also penalised for appearing in the
     prompt tail.
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
- **CPU kernel path.** The engine pins one kernel path on every host. It sets
  `ATEN_CPU_CAPABILITY=default` and `MKL_CBWR=AVX` before torch is imported, and turns oneDNN
  off. It refuses any host where torch reports a different path, and records the observed path
  in each shard's run identity. This is needed because the pool mixes Ivy Bridge (iccad1/3, AVX
  only) with Broadwell (iccad5, AVX2 and FMA). A 4-image TRAIN canary on iccad3 and iccad5
  checks whether the pin gives the same answers on both; the result is under `measured_cost`.
- **Threads.** Threads affect speed only, so they are not part of the identity. The thread count
  is recorded in each ledger start and in `load_info`.

## Shards and merge

- **Assignment.** Author-order position i goes to shard i mod 6. Each shard has its own journal,
  ledger and budget under `stage-<stage>/shard-<k>-of-6/`.
- **Hosts pull shards.** The kernel path is pinned, so a shard does not depend on the host it
  runs on, and each host takes the next unstarted shard. At the measured rates, iccad3 runs 3
  shards, iccad1 runs 2 and iccad5 runs 1. With one fixed shard per host, the slow iccad5 shard
  would set the finish time.
- **Merge refusals.** `merge` unions the shard journals. It refuses any of the following:
  - an image journaled outside its shard, or reserved or completed twice within one;
  - a shard directory that is not `shard-k-of-6`;
  - shards whose run identities differ in anything other than their rows and shard index
    (the identity includes dtype, the observed CPU kernel path, runtime versions and the
    templated-prompt hash);
  - shard ledgers that started under a different plan SHA256 or different runner sources;
  - any missing or unknown image.

  It scores only a complete stage. It writes `<output>/merged-<stage>-predictions.json`, outside
  the stage directory.
- **Pinned files.** `sources_pinned` lists all 9 files the runner hashes, and the runner refuses
  an activated plan whose list differs. The plan also pins the snapshot record's SHA256, which
  `run` and `smoke` check before they rehash the weights.
- **Gate from discovery to full VAL.** The gate is projected per shard:
  `ceil(3333/6) × discovery mean latency + chunks × per-invocation overhead`. The gate stops if
  the projection exceeds 14 days. The shard cap is
  `min(max_shard_active_seconds, 2 × projection + one invocation wall cap)`.

  v1's cap was `2 × projection` alone. v2 adds one wall cap because, on these CPUs, the allowance
  for a single 16,384-token answer is larger than a whole shard's projection. Without it, v1's
  consistency rule would stop every run.

## Stages and analysis

1. **Discovery.** The same 100 IDs as v1 (D13 salt), sharded 17/17/17/17/16/16. The report gives the
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
per run. All hosts were shared with other jobs, so these are per-host figures and not
dedicated-hardware numbers.

| run | s/image | mean |
| --- | --- | --- |
| **pinned** iccad3 (Ivy Bridge) | 333.1, 550.9, 257.5, 330.1 | **367.9** |
| **pinned** iccad5 (Broadwell) | 912.0, 1249.4, 1004.9, 915.2 | **1020.4** |
| unpinned iccad1 (oneDNN on) | 859.7, 831.2, 177.1, 266.6 | 533.7 |
| unpinned iccad3 (oneDNN on) | 504.6, 596.6, 207.9, 283.2 | 398.1 |

- **Where the time goes.** Pinned prefill took 177-381 s on iccad3 and 477-766 s on iccad5.
  Pinned decode ran at 13-28 s per token on iccad3 and 73-88 s per token on iccad5. Answers were
  5-7 tokens, and peak RSS was 164.7 GB.
- **Canary.** Pinned iccad3 and pinned iccad5 gave the same four answers: Fokker 50, Fokker 70,
  DC-9-30 and ATR-42. The unpinned runs gave them too. This is weak evidence, not proof, that
  the numerics are equal.
- **Full-VAL projection.**

  | layout | hours | wall clock |
  | --- | --- | --- |
  | 6 shards pulled by 3 hosts | about 170 | 7.1 days |
  | 3 fixed shards, one per host (bounded by iccad5) | 314.9 | 13.1 days |
  | one host alone (iccad3) | 340.6 | 14.2 days |

  With 6 shards, iccad3 runs 1,667 images in 170.4 h, iccad1 runs 1,111 in 164.7 h, and iccad5
  runs 555 in 157.3 h. Discovery takes 1.7-4.8 h per 17-image shard.
- **Budgets per shard.**
  - Discovery: wall cap 1,486,000 s, 3 invocations.
  - Full: wall cap 1,608,200 s, 12 invocations, 60 calendar days.
  - Gate: affordability limit 1,209,600 s; hard ceiling 5,184,000 s.
  - Each wall cap includes 1,440,000 s, enough for one native 16,384-token answer at the slowest
    measured decode rate (87.8 s per token on iccad5), so a long answer is never truncated.
  - These are ceilings, not expectations.
- **Not a forecast.** VAL images reach 1.9 MP, and the largest smoke image was 1.67 MP. The full
  cap comes from the discovery measurement.

## Not claimed

- Bit-identical outputs to v1 or to any GPU or MLX run.
- The strongest Qwen3.8 profile: xhigh thinking remains unmeasured.
- Anything about TEST.
