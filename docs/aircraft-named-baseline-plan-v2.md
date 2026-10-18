# Named baseline v2: author Qwen3.8-27B on CPU, sharded (PRE-REGISTERED, UNACTIVATED)

Machine-readable plan: `docs/aircraft-named-baseline-plan-v2.json`. Runner:
`scripts/aircraft_named_baseline_v2.py`. Engine: `scripts/aircraft_named_baseline_hf.py` (kept out
of `src/vision_rl`, whose modules other pre-registered plans hash). The committed plan is
unactivated and the runner refuses it. A run needs an activated copy whose SHA256 is passed on
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
| hosts | Mac | 7 allowlisted UW hosts that pull 10 discovery and 60 full-VAL shards |

So v2 runs the same base model at different numerics. A v2 number is never labelled as the 8-bit
model. **No v1 run-2 result is merged with v2.** v2 never reads, resumes, copies or pools the v1
discovery outputs (34/100) or full-VAL invocations 1-3. It regenerates all 100 discovery images and
all 3,333 VAL images. v1 numbers may appear next to v2 numbers only under their own label.

v1's files (`local_vlm.py`, `local_vlm_execution.py`, `scripts/aircraft_named_baseline.py`) are
left byte-identical. The paused v1 run hashes them into its identity and ledger, so any edit would
make it refuse to resume. A test asserts that they still match v1's pins.

## Engine

- **Snapshot.** Before every load the engine verifies the snapshot against
  `docs/qwen3.8-27b-hf-integrity-1d4bf0f2.json` (from the Hub API: sha256 for the 19 LFS files, git
  blob SHA-1 for the other 13). It also refuses any file in the directory that the record does not
  list, because transformers loads optional files it finds and the directory is in shared
  `/dev/shm`. The only exception is `hf download`'s own bookkeeping under `.cache/huggingface`,
  and only for listed files and the pinned revision.
- **Runtime lock.** `docs/qwen3.8-27b-hf-cpu-freeze.txt` is the `uv pip freeze` of the venv that ran
  the smoke (39 distributions, `huggingface-hub==2.2.0` included), on Python 3.12.15. `run` and
  `smoke` refuse unless every installed distribution equals the lock exactly and the Python version
  matches. The lock's SHA256 and the Python version are in each shard's run identity.
- **Sampling.** The engine reproduces the mlx-vlm 0.6.8 chain:
  1. A presence penalty of 1.5 on the unique tokens among the last 20 entries of mlx-vlm's token
     history. That history is the whole prompt plus the generated tokens. For a prompt over 2,048
     tokens, which mlx-vlm prefills in chunks, it is the last prompt token plus the generated
     tokens.
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
- **Dtype.** float32: the author bf16 weights are upcast exactly at load. The pool CPUs have no
  native bf16; in the bf16 smoke, TRAIN image 0447760 was unfinished after about 3,120 s.
- **CPU kernel path.** Before torch is imported the engine sets `ATEN_CPU_CAPABILITY=default`,
  `MKL_CBWR=AVX` and `ONEDNN_MAX_CPU_ISA=AVX` (with `DNNL_MAX_CPU_ISA` unset), keeps oneDNN on, and
  refuses if torch reports another path or if torch or MKL was already loaded under another
  environment. Each shard's run identity holds `kernel_fingerprint`: the SHA256 of fixed float32
  computations (matrix product, one-row linear, softmax, layer norm, attention, the vision
  tower's strided 3-D convolution) on integer-valued inputs. The merge refuses shards whose
  fingerprints differ.
  - **Why this path.** The previous pin turned oneDNN off. Measured back to back on iccad3, that
    was what made decode slow, not host load:

    | configuration (iccad3, 32 threads) | s/image | decode s/token | cores busy |
    | --- | --- | --- | --- |
    | previous pin: oneDNN off + `MKL_CBWR=AVX` | 462.1 | 61.74 | 25.4 |
    | unpinned | 255.0 | 2.12 | 31.5 |
    | oneDNN off only | 360.7 | 13.72 | 31.5 |
    | `MKL_CBWR=AVX` only, oneDNN on | 243.3 | 2.09 | 31.6 |
    | this plan (dnnavx), 4 images | 265.0 | 2.08 | 31.6 |

  - **Same function on every host.** Under this path the first-step score rows (SHA256 of the
    float32 bytes) and the answers of all four TRAIN canary images are identical on iccad1, iccad2,
    iccad3 (Ivy Bridge), iccad5 (Broadwell) and iccad10/12/13 (Skylake), and at 16 threads on
    iccad3. Unpinned, iccad5's score rows differ from iccad3's (no unpinned model run finished on
    a Skylake host; there only the unpinned fingerprint was measured). This is strong evidence, not a
    proof for every VAL image; the fingerprint is the runtime check.
  - **Cost.** The pin is free on Ivy Bridge but not on newer CPUs: iccad5 ran 503.5 s/image
    under this path against 230.1 s/image fully unpinned. It is the price of a single numerics
    path. Which of the three settings costs what on iccad5 was not measured.
- **Threads.** Fixed at 32 and part of the run identity. Under this path, 16 and 32 threads gave
  identical score rows and fingerprints, but the canary ran at 32.
- **Sampling settings.** The engine refuses any setting that mlx-vlm would apply but this engine
  does not implement: a repetition penalty other than 1.0, a frequency penalty, or `min_p`.

## Shards, hosts and merge

- **Assignment.** Author-order position i goes to shard i mod N: 10 discovery shards of 10 images,
  60 full shards of 55 or 56. Each shard has its own journal, ledger and budget under
  `stage-<stage>/shard-<k>-of-<N>/`.
- **Hosts.** `sharding.hosts` is an allowlist of hostname and CPU model, and `run_shard` refuses any
  other machine before writing anything. It lists the hosts whose canary reproduced the score rows
  and that have the RAM for the process: iccad1, iccad2, iccad3, iccad5, iccad10, iccad12 and
  iccad13. The 164.7 GB peak RSS includes the mapped `/dev/shm` snapshot pages (iccad2, 188 GB,
  ran with about 27 GB free), so the footprint is about 165 GB. ecetesla0/1, iccad4 and iccad6
  have too little RAM. iccad0/7/8 fit, but beside the exploratory sweep they had only 180-222 GB
  free. iccad9/11 fit, but the sweep saturated them and their canary did not finish.
- **A shard never changes host.** The host is in the run identity. A resume on another host, or
  under another kernel fingerprint, runtime or snapshot, is refused before the ledger charges an
  invocation. A shard whose host is lost needs a new plan version.
- **Paths, memory and sync.** Each host keeps the staged tree, the venv and `--output` at fixed
  paths under `/var/tmp/r226lin-nbv2/`, outside uwrun's per-run directory, because the ledger and
  budget clock live in `--output`. The staging path is not in the run identity. Launch with
  `--threads 32 --mem 180` (and `--need none` on iccad1/2/3). A run stopped by uwrun's
  `oom-guard` or `--mem` is a killed invocation: charged its full wall cap, its reserved image
  unknown and never quoted, and the shard halted for review. Fetch each whole finished shard
  directory (never only `run/`) to one collector. Before a host's first full-VAL invocation, copy
  the collector's whole `stage-discovery/` into that host's `--output`. Until then the full stage
  refuses ("discovery shard k never ran").
- **Merge refusals.** `merge` unions the shard journals. It refuses:
  - an image journaled outside its shard, or reserved or completed twice within one;
  - a shard directory that is not `shard-k-of-N`;
  - a shard with journaled images but no budget ledger or `run.json`;
  - shards whose run identities differ in anything but rows, shard index and host (kernel
    fingerprint, dtype, threads, runtime lock, versions and prompt hash included);
  - shard ledgers that started under a different plan SHA256 or different runner sources;
  - any missing or unknown image, or any image that ended in a hard stop.

  It reports each shard's host and each host's mean latency. `merge` and `compare` need the
  activated plan and its SHA256. `report` is diagnostic and never scores.
- **Pinned files.** `sources_pinned` lists all 9 files the runner hashes. Three are pinned now (v1's
  two engine files and v1's runner); the other six are filled from the activated commit, and an
  activated plan with any null pin is refused.
- **Gate from discovery to full VAL.** A full shard runs on one host, so the projection uses the
  slowest discovery host's mean latency:
  `ceil(3333/60) × slowest host mean + chunks × per-invocation overhead`. A full shard is refused
  on any host that ran no discovery image, so the projection covers every host that can run one.
  The gate stops if the projection exceeds 14 days. The shard cap is
  `min(max_shard_active_seconds, 2 × projection + one invocation wall cap)`. A static rule refuses
  any plan where `max_projected + one wall cap > max_shard_active_seconds`, which is what keeps that
  cap from ever falling below `projection + one wall cap`.

## Stages and analysis

1. **Discovery.** The same 100 IDs as v1 (D13 salt), in 10 shards. Every pool host should take at
   least one. The report gives the Hoeffding 95% interval. Discovery is a validity and sizing
   diagnostic, not a score.
2. **Full native VAL.** All 3,333 images, run only when the discovery gate says `proceed`.
3. **Compare.** The baseline is compared with three arms from
   `runs/aircraft-native-val-activation-v1/bundle`, whose `comparison.json` has SHA256 `a49482bc…`:
   - **W**: the `common` arm.
   - **RL400**: the `rl-exposure` arm.
   - **strongest SFT**: `sft-1e5-time`, the `strongest_declared_sft` the evaluator chose.

   The arms are fixed in code. `export-arm --plan … --arm W|RL400|strongest_SFT` refuses unless the
   bundle's `comparison.json` has the pinned SHA256, its `plan.json` is the one that comparison ran
   under, the arm's journal has the SHA256 that comparison recorded, and, for strongest SFT, the
   evaluator selected `sft-1e5-time`. It then re-parses every answer through the evaluator's own
   `Journal`. Its output carries that provenance. `compare` refuses any arm file whose provenance
   does not match, or that has a label outside the vocabulary. Two metrics are reported:
   equal-class accuracy with `compare_classification`'s paired bootstrap, and image accuracy with
   a paired image bootstrap. Both use 20,000 draws and seed 2718.

## Measured cost

Source: `reports/2026-10-18-named-baseline-v2-kernel-probe` (this plan's kernel path), float32 with
32 threads, 4 TRAIN images per host. Every host was shared; the Skylake hosts carried the
exploratory sweep (load 45-97 on 72 threads) and our probes ran at nice 10-19 there. These are
per-host figures, not dedicated-hardware numbers.

| host | CPU | s/image (mean) | decode s/token |
| --- | --- | --- | --- |
| iccad1 | E5-2667 v2 | 258.0 | 2.20 |
| iccad2 | E5-2667 v2 | 281.9 | 1.96 |
| iccad3 | E5-2667 v2 | 265.0 | 2.08 |
| iccad5 | E5-2640 v4 | 503.5 | 6.99 |
| iccad10 | Gold 6154 | 558.1 | 12.14 |
| iccad12 | Gold 6154 | 973.0 | 18.64 |
| iccad13 | Gold 6154 | 546.6 | 7.96 |

- **Full-VAL projection.** With the 7 hosts pulling 60 shards: 0.01783 images/s, so about
  186,900 s = 51.9 h of generation. With per-invocation loads and the end of the run, that is about
  52-60 h (2.2-2.5 days). The three Ivy Bridge hosts alone would take 82.7 h. The previous plan's
  figure was about 170 h. Discovery takes 0.7-2.7 h per 10-image shard.
- **Budgets per shard.**
  - Discovery: wall cap 1,338,520 s, 3 invocations.
  - Full: wall cap 1,478,920 s, 4 invocations, 60 calendar days.
  - Gate: affordability limit 1,209,600 s; hard ceiling 5,184,000 s.
  - Each wall cap allows 1,310,720 s of long generation: two native 16,384-token answers at
    40 s/token, about 2.1× the slowest decode measured (18.64 s/token on iccad12).
  - That is not unlimited. A third maximum-length answer in one invocation would reach the wall
    cap. The invocation is then killed and charged its full cap, and the reserved image becomes
    unknown, which halts the shard for human review.
  - These are ceilings, not expectations: canary answers were 5-7 tokens.
- **Not a forecast.** VAL images reach 1.9 MP, and the largest canary image was 1.67 MP. The
  Skylake rates move with the sweep's load. The full cap comes from the discovery measurement.

## Not claimed

- Bit-identical outputs to v1 or to any GPU or MLX run.
- Identical outputs on every VAL image across hosts: the evidence is 4 TRAIN score rows per host
  plus the kernel fingerprint.
- The strongest Qwen3.8 profile: xhigh thinking remains unmeasured.
- Anything about TEST.
