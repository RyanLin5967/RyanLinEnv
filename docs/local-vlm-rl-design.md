# Local token-GRPO design and measured execution

This records a local pretrained VLM experiment, dated October 16,
2026. The native Gaussian wheat branch failed its validation gate. Aircraft
token GRPO is now the primary candidate, contingent on real task-reward
variation and baseline headroom before any substantial fit. The adapter is
[`vision_rl.vlm_local_training`](../src/vision_rl/vlm_local_training.py).
It delegates the actual token policy-gradient loss to released TRL 1.16.0.
There is no second GRPO implementation in this adapter.

**Measured so far:** after a common 400-image label-SFT warm start, real
pretrained MPS GRPO made three updates with correct-versus-valid-wrong Aircraft
reward variation in a four-update TRAIN diagnostic. Gold-answer NLL fell, while
greedy accuracy remained 3/4. Its four-update SFT control had slightly lower
final NLL. Uninterrupted GRPO and a fresh-process two-plus-two continuation
matched exactly in samples, rewards and every final adapter tensor. These are
TRAIN mechanism measurements, not held-out improvement or RL superiority.
Earlier unsuccessful base-policy probes remain preserved below.

The full TRAIN400 controlled study is now running from that same common SFT
warm start, with a frozen FP32 profile and no reuse of diagnostic weights.
Its original RL budget is 400 updates / 10,800 active seconds. Each of three
SFT learning rates (1e-5, 5e-5, 1e-4) has an original 6,000-update / 10,800-second
budget, with endpoints at 400 presentations and the RL endpoint's cumulative
active time. The complete isolated package, inputs and runtime are pinned in
`runs/aircraft-controlled400-v1-plan` and
`runs/aircraft-controlled400-launch-v1/launch-plan.json` (SHA-256
`63cf08acdd1ee80fccfca29bb81a84c39569fbc78396b42ac3750d23068ed7ba`).
Each phase runs in a fresh process. Independent guards bound aggregate RSS and
Metal driver allocation to 24 GiB, preserve 10 GiB free disk, and stop at
12 controller hours / 4 hours per phase without automatic retries. This live
TRAIN study has no completed control comparison or held-out result yet.

Validation: **94 tests passed** (62 local-adapter tests and 32 existing VLM
tests), including actual released configurations, dataset lazy image decoding,
and the pinned Qwen processor plus released TRL completion-only collator.
The current bare-label target supervises `blue<|im_end|>\n` without prompt loss.
Ruff check/format passed after the sampler, exposure and reference-anchor changes.
Independent review also passed the 85-test suite and verified the sampled
generation configuration against the pinned native TRL source.
The historical strict-markup processor evidence remains separately preserved in
`runs/vlm-local-readiness/cpu-readiness.json`.

| Real pretrained CPU diagnostic, four updates | SFT | GRPO |
| --- | --- | --- |
| Nonzero finite gradient steps | 4/4 | 0/4 |
| Changed LoRA tensors | 112 | 0 |
| Train gold-answer NLL | 0.808378 → 0.571192 | 0.808378 → 0.808378 |
| Saved/reloaded NLL | Exact agreement | Exact agreement |
| Strict-markup validation accuracy | 0/4 before and after | 0/4 before and after |
| Mixed / all-zero reward groups | Not applicable | 0 / 4 |
| Trainer active runtime | 10.9186 seconds | 49.9208 seconds |
| Sampled process peak RSS | 12.88 GiB | 9.21 GiB |

Both arms started from the same frozen base, used float32, two CPU threads,
50,176 pixels, rank-4 language LoRA, 32 completion tokens, and LR 0.0001.
Combined supervised wall time was 140.34 seconds and output size 135,972,206
bytes. The real GRPO sampler/teacher-forcing maximum log-probability difference
was 1.34e-6 on the sampled two-token completion. Evidence is
`runs/vlm-local-readiness/pretrained-cpu-probe-summary.json` and
`runs/vlm-local-cpu-probe-v2/{sft,grpo}`. An earlier dataset serialization failure
is retained in `runs/vlm-local-cpu-probe`. There was no RL weight update or
accuracy gain; successful execution is not learning.

The strict-markup diagnostic rejected bare color labels. Its results are kept
unchanged. Before any Aircraft output, the coordinator prospectively amended
the shared grammar to `canonical-label-v2/optional-tags`: exactly one canonical
bare label or exactly one `<answer>label</answer>`, optionally preceded by one
bounded `<think>` block. This avoids making XML formatting the benchmark skill.
It is not a retroactive rescore, answer repair, or evidence of learning.

## Evidence and candidate task

[Visual-RFT v1, Table 2](https://arxiv.org/html/2503.01785v1) reports Qwen2-VL-2B
few-shot classification improvements, including Aircraft 4-shot accuracy
74.8 versus SFT 67.9 and base 45.9; its 16-shot values are 79.4 versus 76.1.
This motivates testing token RL on a
classification task. The paper does not establish a local MPS result, a win over
today's frontier models, or our prospective protocol. Some other reported SFT
controls fall below their base model, so their large differences cannot define
our acceptance threshold. Its published classification reward/evaluation uses
more permissive matching than our exact canonical verifier, so numerical parity
with the paper is not established. See the independent
[strategy review](rl-strategy-review-2026-10-16.md).

The prospective Aircraft task is the **actual 2013b, 100-variant archive**, not
the website's 102-variant prose. See the independent
[author metadata audit](fgvc-aircraft-audit-2026-10-16.md). Its annotation SHA-256
is `50e4419bd330b68a2e85dbcb32339d31120190efe6944682e87c9b431f5f202a`;
native train/validation/test counts are 3,334/3,333/3,333. The audit contains an
optional prospectively selected 16-per-class training subset; it is not an
activated experiment. A four-per-class, 400-image scarce-data experiment is a
candidate after actual base/frontier feasibility checks, with strongest matched
SFT controls on the same 400 images. It would not support claims against
full-data SFT. Remove the bottom
20 pixels from every image before any model sees it, and use the complete sorted
variant vocabulary for every arm. The data owner supplied a fixed 16-TRAIN-image
mirror subset with immutable byte/pixel hashes; equivalence to original author
image bytes and the full cross-split pixel audit remain pending. The unchanged
author evaluator passed nine synthetic cases under Octave 9.4.0; see
[native evaluator evidence](fgvc-aircraft-evaluator-parity-2026-10-16.json).
Mean-class accuracy is the intended primary metric. Full benchmark comparisons
remain pending.

The reasoning hypothesis must survive the experiment. The prospective paired
warm-policy assay selected `answer_only` because the reasoning prompt failed
its unchanged canonical-validity gate. The selected method is task-only token
RL and makes no reasoning claim.
`think_then_answer` permits one short `<think>...</think>` block followed by
exactly one canonical bare or tagged answer. The optional reasoning block is
bounded to 32,768 characters and the generation budget remains the tighter
256-token default. A direct answer is also valid.
Unknown labels, multiple answers, nested markup, extra prose, and truncated
generations fail. Reasoning is journaled as policy output, never supplied as a
gold annotation. This is a prospective design, not a reproduction claim.

## Frozen model and software identity

The sole downloaded actor is the official
[Qwen2-VL-2B-Instruct snapshot](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/895c3a49bc3fa70a340399125c650a463535e71c):

| Identity | Value |
| --- | --- |
| Repository | `Qwen/Qwen2-VL-2B-Instruct` |
| Full revision | `895c3a49bc3fa70a340399125c650a463535e71c` |
| Local single copy | `data/models/Qwen2-VL-2B-Instruct-895c3a49` |
| Verified selected files | 13 files, **4,429,621,382 bytes** |
| Safetensors weights | **4,418,050,768 bytes** |
| Download manifest SHA-256 | `2c20d30e3b0c27b2da711fa1b977d7dbb47049f22bef9d5b2af2d47e75e6bfcc` |
| Weight shard 1 SHA-256 | `994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9` |
| Weight shard 2 SHA-256 | `92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833` |
| Processor configuration SHA-256 | `b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc` |
| Tokenizer JSON SHA-256 | `cb63a0a23eef3d5b01063a9880a1925a65aaf4d1591d519910ee3527852950a0` |

`scripts/download_pinned_model.py` performed the download with one worker, Xet
cache disabled, no credentials, and a 10 GiB free-space reserve. It finished with
30,471,925,760 free bytes. Source manifest and full file hashes are retained in
`runs/vlm-local-readiness/qwen2-vl-2b-source.json` and
`runs/vlm-local-readiness/model-integrity.json`; `model-source.json` beside the
weights is an exact copy of that source manifest. The adapter verifies every
listed file and safetensors index again before loading. `local_files_only=True`
and `trust_remote_code=False` apply to both model and processor. The processor
always comes from the pinned base, never an adapter-provided template.

[`requirements-vlm-local.in`](../requirements-vlm-local.in) specifies torch
2.14.1, torchvision 0.29.1, TRL 1.16.0, Transformers 5.20.0, PEFT 0.21.2,
Accelerate 1.15.0, datasets 5.1.0, and fsspec 2026.7.0. Exact TRL trainer/config
file SHA-256 values are enforced by preflight against the downloaded release
source. The official sources inspected were
[GRPOTrainer](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py),
[GRPOConfig](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_config.py),
[SFTTrainer](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/sft_trainer.py),
and [SFTConfig](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/sft_config.py).

The existing `.venv-detection` was not modified. The missing small dependencies
are isolated in `runs/vlm-local-libs/site-packages` with no duplicate torch. datasets 5.1.0
requires fsspec <=2026.7.0, so this target also shadows the shared 2026.9.0 copy.
The exact isolated install performed was:

```sh
uv pip install --python .venv-detection/bin/python \
  --target runs/vlm-local-libs/site-packages --no-deps \
  trl==1.16.0 datasets==5.1.0 pandas==3.0.6 pyarrow==26.0.0 \
  dill==0.4.1 multiprocess==0.70.19 xxhash==4.0.1 fsspec==2026.7.0
```

The target must end in `site-packages`. The first actual actor attempt exposed
dill's module-by-value serialization behavior for a bare PYTHONPATH target:
Dataset construction raised `PicklingError: MonthDayNano`. The exact installed
files were moved into the `site-packages` child directory, without changing
package versions or copying torch. Dataset construction/decoding is now covered
by the actual dependency test and dependency preflight, before actor allocation.
The original failed attempt remains in `runs/vlm-local-cpu-probe`; the guarded
retry uses `runs/vlm-local-cpu-probe-v2`.

## Profiles and algorithm contract

| Setting | Format probe / ablation | Prospective reasoning experiment |
| --- | --- | --- |
| Response format | `answer_only` | `think_then_answer` |
| Completion limit | 32 tokens | 256 tokens; 512 is a separately frozen alternative |
| Image pixels | 50,176 minimum and maximum | 50,176 minimum, 200,704 maximum |
| Approximate Qwen merged image tokens | 64 | 64–256 |
| Untruncated prompt limit | 2,048 tokens, fail if exceeded | Same |
| LoRA | Language q/v projections, rank 4, alpha 8 | Rank 8, alpha 16 |
| Base precision | Explicit float32 | Explicit float32 initially |
| Sampling group | Four completions, one distinct prompt per update | Same |
| Sampling | Temperature 1, top-p 1, top-k 0 | Same |
| Attention | SDPA | SDPA |

The table's left column describes the historical cheap format probe. The
selected answer-only Aircraft method uses the right column's rank-8, image and
256-token limits, with the exact answer-only instruction. Its actual four-step
MPS GRPO runtime is reported below. Reasoning-enabled MPS training remains
unmeasured.
Explicit bfloat16 is available as a separately recorded experiment; fp16,
quantization, FlashAttention, vLLM, continuous batching, and hidden CPU fallback
are excluded from this adapter. Image limits are checked through the actual
processor. Prompts are never truncated through image tokens. Every class's gold
answer must fit the completion limit.

The current preparation uses released `loss_type="grpo"`, group-normalized rewards,
token-level importance ratios, clipping epsilon 0.2, and one iteration per sampled
group. The prospective research candidate now defaults to **`beta=0.04`**, with
`--beta 0` explicitly selecting the no-KL ablation used in the historical CPU
probe. This coefficient is motivated by Suitability, not tuned or validated on
our task. Passing a prewrapped PEFT model makes released TRL copy its initial
adapter to `ref`, sharing the frozen base without allocating a second 2B model.
Fresh zero-B LoRA therefore references the pinned base; a common saved warm
start references that exact initial adapter instead. Metadata names this
difference and hashes both the initial saved weights and reference tensors.
The adapter verifies the copied tensors match initially, are excluded from
training, and remain unchanged. Resume compares the reference with the original
run identity, then rejects altered reference weights immediately after checkpoint
loading and again before training. A reference forward pass still costs time.
Per-group clipping at one iteration does not bound cumulative drift from the
initial policy. A frozen base plus LoRA
is used in both arms, with dropout disabled and no vision-tower training.
Reward is exactly 0/1 task correctness with EOS termination required. There is
no format bonus or injected reward variance. Every group journals its actual
completion tokens, text, rewards, and whether both correct/incorrect outcomes
were present. All-wrong or all-correct groups have no task-reward advantage.
Journals also record actual prompt presentations, unique image IDs, generated
tokens, EOS termination, complete think-block counts and task correctness with
and without those blocks. Think-block syntax is not evidence of reasoning
quality, and format-only variation never counts as useful task-reward variation.

TRL's GRPO generation batch is four and gradient accumulation is four;
microbatch size is one. SFT uses one example per update, completion-only
cross-entropy, no packing, and no truncation. Both receive the same images,
class list, prompt, data, frozen base or shared warm start, LoRA structure,
optimizer, and prospective seed set. SFT labels contain the true final answer
only; it receives no invented chain of thought. In the reasoning profile the
shared prompt permits reasoning, while supervised targets remain answer-only;
that distinction must be reported.
Both algorithms expose one distinct image per optimizer update. Their released
samplers differ after the first epoch, so identical seed and exposure counts do
not establish identical image order. No identical-trace control claim is made.

At minimum, select the strongest validation SFT among a predeclared learning-rate
and update-budget search. Include both equal-update and equal-active-wall-time
controls; the latter counts RL generation/reward time. `--max-active-seconds`
stops at an optimizer boundary, so its overshoot must be reported. It is now a
cumulative limit on complete optimizer-update time along the checkpoint
continuation. The original total stays fixed on resume; the adapter subtracts
time committed with that exact checkpoint. Each span starts after data prefetch
and ends after gradient accumulation, clipping, optimizer and scheduler; MPS
synchronizes both boundaries. Generation, rewards and reference inference are
inside the span. Data decoding/prefetch, evaluation, logging and checkpoint IO
are outside it and must be reported separately for end-to-end comparisons.
Interrupted or rolled-back work is recorded separately when observed; this is
not a hard cap on total physical compute. Never compare against a collapsed SFT when the base or another
prospective SFT is better. A common format warmup may use different train data;
both experimental arms must use the same saved initial adapter hash. Parent
data fingerprints and structural model/processor/LoRA settings remain pinned.

Training input is strict JSONL with only `id`, `split`, `image`, and `label`.
`split` is `train` or `validation`; every split represents every declared class.
A separate `data_spec.json` has schema version 1, benchmark name, common
question, complete ordered `labels`, split policy, preprocessing description,
and source URLs/SHA-256 hashes. Aircraft specs additionally declare
`preprocessing_id="fgvc-aircraft-rgb-remove-bottom20-v1"`. Image paths point to
original JPEGs. Audit, dataset decode, generation, and SFT all call the shared
`aircraft_data.load_aircraft_rgb` exactly once: RGB decode, reject unexpected EXIF
orientation, remove bottom 20 rows, no resize/re-encode or transformed cache.
Native model processors resize after this shared transform. Example-specific
prompts are rejected. Dataset identities include manifest, image-byte, derived
pixel, and transform-source hashes; shared decoded-pixel checks
reject cross-split copies/re-encodings. Validation is only greedy pass@1
inference before and after training, never a trainer/reward input. Test inputs
are not accepted. Full native validation selection and a sealed final test are
required later; this file does not authorize opening test images.

## Fixed TRAIN-only Aircraft inference

`--predict-train` performs frozen-base greedy pass@1 with no optimizer. It
accepts at most 32 TRAIN images, permits a subset of classes, and retains the
complete vocabulary. Training still requires every declared class in both its
train and validation manifests. Test rows are rejected before image access.
Partial predictions, EOS status, token IDs/counts, latency, and run errors are
preserved; existing outputs cannot be overwritten.

`--sample-train` uses the same strict TRAIN-only inputs and cap, but draws a
group of four completions per image from the unchanged base. It uses native
Transformers sampling with the pinned released-TRL generation settings:
temperature 1, top-p 1, top-k 0, image-token suppression, and first-EOS-inclusive
completion masks. The diagnostic draws samples serially at batch size one;
it does not establish the memory cost of TRL's batched training rollouts. The
shared exact reward records mixed/all-zero/all-correct groups, token counts,
termination and think-block syntax. Partial raw samples and missing-sample
counts survive errors. No optimizer, policy-gradient loss or update is created.
The historical API preparation was CPU-contract tested; the subsequent actual
Aircraft sampling and training results appear below.

The first run used the fixed 16-image selection in
`runs/aircraft-shared-probe16/metadata-selection.json`, seed 791, all 100 labels,
and `prepared/{train.jsonl,data_spec.json,provenance.json}`. Input identities:

- Manifest SHA-256: `1921e0de9a30d31e3405195470355a67dc40156d43a18ec8afcd515d67c9d2e0`.
- Spec SHA-256: `a1253f4ea9012480b3328e10103462b99f5fc3887190ef3081a70a22dc3aff6a`.
- Shared prompt SHA-256: `c8cc2849d641c5a0a9c2f73d9a1dba7a7fdfe4f64fc3977dc70c2c0c30503182`.
- Prompt/parser SHA-256: `fa5d19bbb87eae1486ea51a3991ee1425f92dc138a658faee0908f80f062b7d6`.

The processor observed at most 1,014 prompt tokens and 17 tagged-answer tokens
including EOS. All 16 model-visible pixel hashes matched the shared audit.
Evidence: `runs/vlm-local-readiness/aircraft16-processor-readiness.json`.

The actual MPS run completed in **23.72 seconds**, with **0/16 correct, 13
invalid answers**, 16 EOS-terminated outputs, and no `<think>` blocks. Nine
outputs were literal `EXACT_LABEL`, three prefixed a label with `EXACT_LABEL:`,
and one used a noncanonical label. The three canonical outputs were wrong.
No responses were repaired. Completion lengths were 4–13 tokens, so this does
not measure 256-token reasoning throughput. Sampled peak RSS was 9,467,379,712
bytes; MPS driver allocation peaked at 9,835,479,040 bytes and tensor allocation
at 9,008,212,736 bytes. Separate RSS/driver guards were 16 GiB, with 2 GiB output
and 30-minute wall limits; no guard fired and hidden CPU fallback was disabled.

Artifacts are `runs/aircraft-probe16-qwen2vl2b` and its adjacent `-guard`
directory. The exact executing adapter source was copied after verifying its
hash against recorded metadata. This is a TRAIN feasibility failure, not a
held-out accuracy estimate or learning result. The scheduled 27B comparison
uses the same task text, raw cropped pixels and grammar, but its native xhigh
sampling and image/output budget differ; that difference must be reported.

The completed run's inference arguments were:

```sh
PYTHONPATH=runs/vlm-local-libs/site-packages:src \
  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-detection/bin/python -B -m vision_rl.vlm_local_training \
  --predict-train runs/aircraft-shared-probe16/prepared/train.jsonl \
  --data-spec runs/aircraft-shared-probe16/prepared/data_spec.json \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --output runs/aircraft-probe16-qwen2vl2b \
  --device mps --dtype float32 --response-format think_then_answer \
  --min-pixels 50176 --max-pixels 200704 --max-completion-length 256
```

The actual launch used `runs/vlm-local-readiness/run_aircraft16_probe.py` to
supervise these arguments and collect in-process MPS telemetry. It refuses
existing outputs. No additional fit is authorized by this command example.

The coordinator approved a prospective second prompt after the v1 comparison
finishes naturally. It keeps the question, all 100 labels, image selection and
canonical parser, and replaces the instruction with:

> Inspect the aircraft’s visible distinguishing features. First give a brief explanation of the visible details, enclosed by <think> and </think>. Then write one label copied exactly from the allowed list on its own line. Do not add a prefix, quotes, or any text after that label.

This makes the requested reasoning explicit and removes the literal label
placeholder. It must be frozen before any v2 output. A proposed next diagnostic
is v2 greedy inference followed by four sampled completions per same TRAIN
image, reporting task-reward diversity and truncation without any update. It
does not rescore v1. SFT would still receive only gold labels, without fabricated
rationales; that target/prompt asymmetry and strong answer-SFT controls must be
explicit in any subsequent experiment.

The prospective sampled diagnostic command, after freezing that v2 prompt, is:

```sh
PYTHONPATH=runs/vlm-local-libs/site-packages:src \
  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-detection/bin/python -B -m vision_rl.vlm_local_training \
  --sample-train runs/aircraft-shared-probe16/prepared/train.jsonl \
  --data-spec runs/aircraft-shared-probe16/prepared/data_spec.json \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --output runs/aircraft-probe16-qwen2vl2b-v2-groups \
  --device mps --dtype float32 --response-format think_then_answer \
  --min-pixels 50176 --max-pixels 200704 --max-completion-length 256 \
  --num-generations 4 --temperature 1 --seed 791
```

The scheduled launch must apply the same explicit memory/disk/wall supervisor
guards as the first inference probe. The command alone is not the supervisor.

The concrete prelaunch artifact is now frozen at
`runs/aircraft-probe16-v2-plan/plan.json` (SHA-256
`70e1e6498eae94b7959e50ab8c3c03438e023a83b6eb59f99ecde87471594eaf`).
It contains an isolated eight-module source snapshot, so the live v1 prompt
and parser remain unchanged. Its v2 adapter SHA-256 is
`5526f41bef4eec4c688b334b46e9b61a24924a1f5c737aa9b4bd64da59a2c54f`;
the exact prompt SHA-256 is
`5a37649dd8f5e33a521e91f64e0a97a38034e9c850bed6dce564b07c66c6700a`.
The CPU native processor observed at most 1,022 prompt tokens and reproduced
every frozen image hash. `launch.py --verify-only` verified source/input hashes
and detected both prior-owner processes still alive; no model call was made.
The actual launch command is in `launch-command.txt`. At this historical freeze,
execution required the frontier owner's explicit natural-exit and release.
The later cancellation-condition amendment and actual execution are recorded below.

## Small gradient diagnostic interface

The following creates tiny synthetic grids. It is deliberately labeled
diagnostic-only and is not public-benchmark evidence. The strict-markup CPU
version already ran, with the results above. These commands describe the
current prospective grammar, not a request to repeat color-format training.
At that historical stage no MPS backward had run; the subsequent real Aircraft
backward and continuation measurements appear below. The useful learning signal must come from real
public TRAIN data, not engineered synthetic ambiguity.

```sh
export PYTHONPATH="$PWD/runs/vlm-local-libs/site-packages:$PWD/src"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0

.venv-detection/bin/python -B -m vision_rl.vlm_local_training --preflight --device mps
.venv-detection/bin/python -B -m vision_rl.vlm_local_training \
  --make-probe-data runs/vlm-local-probe/data

# A real pretrained SFT backward/update and saved-adapter reload probe.
.venv-detection/bin/python -B -m vision_rl.vlm_local_training \
  --mode sft --device mps --probe \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --train runs/vlm-local-probe/data/train.jsonl \
  --validation runs/vlm-local-probe/data/validation.jsonl \
  --data-spec runs/vlm-local-probe/data/data_spec.json \
  --output runs/vlm-local-probe/sft \
  --response-format answer_only --dtype float32 \
  --min-pixels 50176 --max-pixels 50176 --max-completion-length 32 \
  --lora-rank 4 --num-generations 4 --learning-rate 0.0001 \
  --max-steps 4 --save-steps 1 --seed 42

# A distinct released-TRL token-GRPO probe from the same frozen base.
.venv-detection/bin/python -B -m vision_rl.vlm_local_training \
  --mode grpo --device mps --probe \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --train runs/vlm-local-probe/data/train.jsonl \
  --validation runs/vlm-local-probe/data/validation.jsonl \
  --data-spec runs/vlm-local-probe/data/data_spec.json \
  --output runs/vlm-local-probe/grpo \
  --response-format answer_only --dtype float32 \
  --min-pixels 50176 --max-pixels 50176 --max-completion-length 32 \
  --lora-rank 4 --num-generations 4 --learning-rate 0.0001 \
  --max-steps 4 --save-steps 1 --seed 42
```

Inspect `local_training.json`, `optimizer_audit.jsonl`, `reward_groups.jsonl`,
and before/after validation predictions. A useful GRPO probe needs finite,
nonzero gradients, actual LoRA changes, mixed reward groups, saved/reloaded
agreement, and a measurable learning signal. The code reports gold-answer NLL
before/after as a diagnostic, not as benchmark reward or training feedback.
The GRPO probe also compares sampled-token log probabilities with teacher
forcing over the same tokens, reporting the image-token suppression
normalization separately. Any failed metric is visible; there is no automatic
retry, manufactured reward diversity, or claim that a finished run learned.
If groups are all zero, a task/initial-policy decision is required before a
long fit. A successful diagnostic still does not establish reasoning-enabled
GRPO learning on Aircraft.

## Subsequent real TRAIN execution and staged controls

The 27B v1 diagnostic was explicitly cancelled by the coordinator after three
completed predictions; its partial evidence is preserved. The separate launch
condition amendment at `runs/aircraft-probe16-v2-plan/launch-condition-amendment.json`
records that cancellation. It did not alter the frozen v2 scientific plan.

The actual 2B v2 MPS probe completed naturally in **151.87 seconds**. Greedy
scored 0/16 with 16 invalid responses. The 64 sampled completions contained
one exact correct answer, six canonical valid answers, 63 EOS terminations and
11 complete think blocks. One of 16 groups contained both a canonical correct
and canonical incorrect answer; the other 15 groups had zero reward. This
useful group belongs to an image outside the frozen four-shot 400, and is not
used for that study's training. Peak RSS was 12,718,768,128 bytes and MPS driver
allocation was 9,835,511,808 bytes. No optimizer ran. Evidence is
`runs/aircraft-probe16-qwen2vl2b-v2/{phase-summary,diagnostic-summary}.json`.

The coordinator then authorized one common label-SFT warm start, completed
under the isolated plan `runs/aircraft-warmstart-v1-plan/plan.json`
(SHA-256 `cfa109186c3e29263e747524fdcb7634ff35dd9d17131500653d009b032c01bd`).
It is exactly one shuffled pass over the frozen 400 TRAIN images: 400 updates,
one prompt per update, LR 0.0001, seed 42, unchanged rank-8/pixel/precision
profile, and a placeholder-free answer-only instruction. Targets contain only
the canonical label and native EOS. All 400 updates had finite nonzero gradients;
112 LoRA tensors changed with L2 parameter delta 2.9481. On the fixed eight TRAIN
diagnostic images, greedy accuracy rose from 2/8 to 4/8 and gold-answer NLL fell
from 0.2884249165 to 0.1724672064. The reloaded adapter reproduced the latter
exactly. Total supervised wall time was 1,157.24 seconds, with 1,104.81 seconds
of committed synchronized optimizer time. These are SFT training measurements,
never RL or held-out improvement. The final adapter SHA-256 is
`f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c`.
This historical warmup's exposure count is
configuration-derived; it predates the actual prompt-ID ledger.

The separately frozen coordinator amendment
[`aircraft-warmstart-assay-amendment-v1.json`](aircraft-warmstart-assay-amendment-v1.json)
requires both prompts on the same prospectively selected 32 images from the
400, seeds 42/314, G4: 512 completions in total. Each prompt independently
needs 16 useful groups over at least eight IDs, 90% valid canonical responses
and 95% EOS termination. Choose the reasoning prompt if eligible, otherwise
the answer-only prompt if eligible; never pool their results. Answer-only RL
is task-only token RL and carries no reasoning claim. The actual assay completed
under `runs/aircraft-mechanism32-warm-v1-plan/plan.json`, SHA-256
`9db91c2c04b1eed4571befdaa17c5d04e9c00390b07e12fb26c0dcc16bd04e02`.
Before execution, both prompt profiles were prospectively corrected to use the
same optional-think verifier; the differing prompts no longer also alter the
acceptance grammar. No historical score or warmup source was changed. The assay
binds the completed warm weights and every source/input hash; it performs no
optimizer update. The answer-only profile passed: 181/256 correct, 243/256 valid,
256 EOS, 22 recognition-mixed groups over 14 IDs. The reasoning profile had
142/256 correct, 212/256 valid, 256 EOS and 22 useful groups over 16 IDs; it
failed the 90% validity threshold. The predeclared rule therefore selected
answer-only task-only token RL. These sampled TRAIN counts are not greedy
accuracy or a held-out result. Full evidence is
`runs/aircraft-mechanism32-warm-v1/gate-result.json`.

A four-update GRPO/SFT mechanism diagnostic completed from the same warm
adapter under `runs/aircraft-four-update-v1-plan/plan.json` (SHA-256
`192729be4466a0d8c5f279b01deb150878bc10f04c49cf92f4a5d21c73e48495`).
Its four IDs were adaptively selected by a frozen hash rule from useful TRAIN
groups within the original 400. It includes uninterrupted four-step GRPO,
four-step SFT and a separate two-plus-two GRPO resume check. The extra four
GRPO updates are explicitly charged as diagnostic compute. These weights never
become the final study initialization. Three groups contained both a correct and
valid wrong canonical label, all 16 completions were valid and EOS-terminated,
and native pre-clipping gradient norms on those updates were 6.7104, 3.1141 and
2.2282. The initial all-correct group had zero task advantages and a tiny
4.81e-8 numerical gradient; it is excluded from task-learning evidence. Task and
KL gradient norms were not separately measured. The initial reference check was
exactly zero, and the first mixed update had logged KL zero. The scalar
on-policy task loss may be zero while its policy gradient is nonzero.

GRPO changed 112 LoRA tensors (L2 delta 0.016218) and reduced four-TRAIN gold NLL
from 0.0924919 to 0.0821459; SFT reached 0.0819206. Both retained 3/4 greedy
accuracy and reproduced final NLL exactly after adapter reload. Actual prompt
order was identical across arms. GRPO took 69.7556 active seconds, SFT 11.1262;
this is not an equal-time SFT comparison. The fresh-process two-plus-two GRPO
continuation reproduced all prompt IDs, sampled tokens/rewards, reference and
final adapter tensors exactly, with maximum tensor difference zero. Evidence:
`runs/aircraft-four-update-v1/mechanism-summary.json`.

The original multi-phase supervisor stopped after the successful first GRPO
endpoint: Accelerate 1.15's default-device selection changed
`PYTORCH_ENABLE_MPS_FALLBACK` to `1`, so the next preflight rejected the polluted
environment before SFT loading. The generic fallback dispatch is registered
statically at torch import in
[PyTorch 2.14.1](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/mps/MPSFallback.mm);
that worker imported torch with the flag `0`. This is not an operation-level
CPU trace. An explicit continuation amendment preserved the failed supervisor
and completed only the remaining phases in fresh processes. No scientific
settings or original source snapshot changed.

One separately frozen BF16 engineering trial completed under
`runs/aircraft-bf16-four-update-v1-plan/plan.json`. It keeps the same four images,
warm adapter, prompt, rank, reward and optimizer, changes frozen-base precision
to BF16 and verifies every initial FP32 LoRA tensor remains bit-identical to the
saved warm adapter. The explicit `initial_adapter_precision_transfer` field
records the parent and target precision; ordinary identity checks still reject
undeclared changes. Promotion requires at least 1.3x measured active throughput,
finite useful task gradients, initial reference discrepancy at most 0.001,
sampler/teacher discrepancy at most 0.02, exact initial canonical predictions
and the existing reload check. The original FP32 tolerance remains 0.001.
The trial took 34.1230 active seconds, a measured 2.044x speedup over the four-step
FP32 run. It passed exact initial FP32 adapter values, initial canonical
predictions, reference and reload checks, with three useful groups. However,
its sampler/teacher maximum log-probability discrepancy was **0.0724754**, above
the frozen 0.02 limit. Promotion therefore failed; retain FP32 without an
optimization retry loop. Evidence is
`runs/aircraft-bf16-four-update-v1/promotion-result.json`. The 112 trainable
adapter tensors remained FP32 and 729 frozen parameter tensors used BF16.
This speed result does not establish adequate numerical fidelity or accuracy.

`train_local(..., validation=None, train_only=True)` now provides a bounded
real TRAIN diagnostic, rejecting validation/test inputs and limiting the
manifest to 400 images and GRPO to 400 total updates. SFT may use a separately
frozen finite positive step horizon above 400 with a finite original active-time
budget, allowing honest equal-time controls. The controller must fix both caps
before the first fit; neither can change on resume. The capability constant
`TRAIN_ONLY_STEP_CAPS` exposes GRPO's 400-step cap and `None` for SFT's additional
internal step ceiling; `None` never removes the required finite original run
budgets. It retains the full vocabulary and
uses a hash-selected set of at most eight TRAIN images for before/after greedy,
gold-NLL and saved-adapter reload diagnostics. These are training measurements,
not held-out evaluation. A four-update run presents four distinct-image draws,
not sixteen: each GRPO draw yields four completions.

`predict_train_subset` and `sample_train_groups` accept `init_adapter` for a
frozen saved endpoint. They verify its completed parent run, model/processor,
LoRA/pixel/precision profile, task spec and exact final weights, and load one
adapter into one pinned base. No inference optimizer is created. The numerical
sampler/teacher check is capped at 16 sampled tokens; it reports both the raw
likelihood difference and the difference after matching image-token suppression,
plus initial-reference discrepancy. It does not prove parity for every token
of every 256-token rollout.

Prospective staged training accepts `stop_after_steps`,
`stop_after_active_seconds` and `stage_id`. The first two are cumulative
operational targets below unchanged original total budgets; already-reached
resume targets fail before model allocation. Native SFT batches retain IDs
through a collator wrapper and remove them at actual `training_step`; GRPO
records the actual reward-group ID. Each optimizer boundary commits exactly
one ID. Physical attempts remain in `physical_prompt_events.jsonl`, while each
checkpoint's `exposure.json` contains only its committed ancestry and is hashed
by `active_time.json`. Rollback therefore does not turn discarded work into
committed exposure.

`read_endpoint_receipt(output, expected_stage_id=...)` checks the saved endpoint,
complete checkpoint files, original budgets, reached boundary, reference/source
identity and committed prompt sequence. `endpoint_receipt.json` exposes those
fields for a controller. An actual tiny random CPU fixture with released SFT
and GRPO matched uninterrupted four steps against two-plus-two exactly in final
adapter tensors and actual committed IDs; evidence is
`runs/vlm-local-readiness/tiny-stage-receipt-v1/result.json`. This proves a CPU
contract, not pretrained MPS resume or public-task RL learning.
An explicit `finalize_only=True` with the existing checkpoint restores an
already-reached boundary without calling `train` or performing an optimizer
step, then completes pending diagnostics, adapter export and the receipt. The
normal resume path still rejects an already-reached boundary. The actual tiny
CPU native-helper/export proof in
`runs/vlm-local-readiness/tiny-stage-receipt-v2-finalize/result.json` restored
exact adapter tensors with zero training calls and unchanged checkpoint bytes.
Focused independent review found no blocker. This is not a full pretrained MPS
end-to-end crash-recovery measurement.

Every checkpoint retains released Trainer optimizer, scheduler, CPU/Python/NumPy
RNG, adapter, and trainer state. Transformers 5.20 omits MPS RNG in its default
RNG checkpoint code; a narrow subclass adds `mps_rng_state.pth` and restores it
at the same point as Trainer's other RNGs. It does not override generation or
the objective. Two further CPU reproductions found resume bugs in the pinned
stack: Trainer loads the `ref/` subdirectory while skipping the default adapter
at the checkpoint root, and RepeatSampler's private generator repeats epoch
zero after restart. Narrow extensions restore root adapter weights and
reconstruct the released sampler's exact native epoch permutations. Neither
extension changes GRPO's objective or group construction.

An actual tiny random text model with released TRL, nonzero gradients and
beta 0.04 matched uninterrupted four-step training against both two-plus-two
steps (epoch boundary) and one-plus-three steps (mid-epoch) exactly in sampled
prompts/completions/rewards, final LoRA bytes and frozen reference hash. A later
mid-epoch three-plus-one continuation also matched exactly.
Failures and successful evidence remain separately preserved in
`runs/vlm-local-readiness/tiny-reference-resume-v{1,2,3-order,4-fixed,5-mid-epoch,6-later-mid-epoch}`. This is
a CPU checkpoint contract test, not pretrained VLM or public-task learning.
Resume requires that complete checkpoint inside the same output
directory and identical model/data/profile/settings/environment/source hashes.
Positive-beta checkpoints must include reference-adapter safetensors. New runs
also archive the exact executing adapter source.

Every new optimizer checkpoint also needs `active_time.json`, written only after
the native save succeeds and bound to the SHA-256 of optimizer, scheduler,
default adapter and trainer state plus its global step. The previous marker is
invalidated before overwriting a checkpoint. Missing, altered or already-exhausted
timer state rejects resume before model allocation. Atomic per-invocation files
separate current complete-update time, last durable checkpoint time and observed
discarded/uncheckpointed work; a hard kill's unfinished duration remains unknown.
The independent review's 10-second checkpoint + 5 unsaved + 3 interrupted-second
case reports 10 durable, 15 completed and 18 observed, with 8 outside the durable
checkpoint.

An actual tiny random CPU TRL run resumed a fixed **0.25-second** active budget
after two updates: 0.071621 seconds was retained and 0.178379 seconds remained.
It stopped at update 16 with 0.252941 seconds committed, a 0.002941-second
optimizer-boundary overshoot; restarting that exhausted checkpoint was rejected.
Evidence is `runs/vlm-local-readiness/tiny-active-budget-v3/result.json`, with the
exact source archived beside it. This validates the CPU accounting mechanism,
not VLM learning, a reproducible update count across machines, or MPS timing.
Owned reward journaling is now isolated from reward computation and excluded
from the timed span, with its duration reported separately. This exclusion has
an explicit mocked synchronization/boundary test; the tiny real budget fixture
uses a no-IO reward and does not measure MPS synchronization.
Example: rerun the original command with
`--resume-from-checkpoint runs/vlm-local-probe/grpo/checkpoint-2`.
The later real pretrained FP32 VLM/MPS four-versus-two-plus-two continuation
matched exactly, as reported above. That bounded measurement does not establish
every later checkpoint or arbitrary-length trajectory; CPU evidence and the
existence of RNG files alone would not have proved it.

Before a substantial claim: demonstrate useful real-task reward
diversity, freeze the public task/protocol, fit repeated-seed strong SFT and
matched GRPO, keep all runs visible, evaluate frozen native test once under the
same image/output budget as named baselines, and require a substantial paired
improvement over the strongest SFT **and** the chosen frontier model. Nothing
measured in this preparation fulfills that outcome.
