# Local vision baseline development evidence

The local baseline is **Qwen3.8-27B, 8-bit affine quantization, group size 64**,
from `mlx-community/Qwen3.8-27B-8bit` at revision
`815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9`. It is a distinct local configuration;
these observations do not establish equivalence to a hosted configuration.

The native setup uses xhigh thinking, a 16,384-token completion limit,
temperature 1.0, top-p 0.95 and top-k 20. Aircraft inputs use the shared lazy
bottom 20-pixel crop, identical original RGB pixels, question and 100-class
vocabulary. Each model retains its native processor; the 27B budget is
65,536–16,777,216 pixels. MLX's 32 GiB setting is advisory. The supervisor samples
the owned child's RSS and Metal active-plus-cache allocations against separate
36 GiB limits; these are sampled safety stops, not hard allocation caps.

## Measured observations and limits

| Scope | Observed outcome | Interpretation |
| --- | --- | --- |
| Two pills TRAIN images | Completed naturally. Native inference 423.257 s and 52.228 s; 11,997 + 2,340 and 921 + 440 prompt+generation tokens. | Resource and strict-geometry feasibility only; no accuracy claim. |
| Aircraft v1, fixed 16 TRAIN images | **Explicitly cancelled**, supervisor exit−2. Three completed responses, one interrupted reservation and 12 unattempted images. | Partial diagnostic; no complete accuracy, headroom or benchmark claim. |
| Aircraft v2, the same fixed 16 TRAIN images | Preparation only. All 16 cropped-pixel hashes, 100 labels, prompt and parser match the frozen shared plan. | **No local 27B v2 inference or latency measured.** |
| Wheat, two fixed TRAIN images | Prepared, zero local 27B model calls. | Wheat runtime and accuracy remain unmeasured. |

Aircraft v1 completed inference times were 92.635 s, 1,354.874 s and 937.090 s.
The completed calls report 5,112 prompt tokens and 17,286 generation tokens
(22,398 total). The interrupted request's usage is unknown. Two completed
responses failed the shared output grammar; the remaining response was wrong.
One invalid response contained the correct label amid extra prose; it remains
invalid. The stored `normal_finish=3` means native EOS completion, not three
valid classifications. Cancellation released the accelerator for the small-model
learning experiment. Its raw reservations, responses, source identities and
usage remain unchanged, and the runner refuses to resume its cancellation marker.

The second completed Aircraft input was audited on CPU. Its cropped RGB hash,
native image grid, tensor shape, image-token count and formatted prompt were
reproduced. The historical journal did not retain tensor bytes, so this is not
a retrospective byte-for-byte proof of the historical tensor. No model inference
was performed during that audit.

The mirror probe's author-original image-byte equivalence and full cross-split
audit were pending when this evidence was frozen. No native test images were
evaluated. The proposed 216-image wheat paired subset is still a proposal,
not an active or completed benchmark.

## Durable execution controls

`scripts/aircraft_local_probe.py run` accepts `--stop-after-completions N` and
`--stream-progress`. A completion boundary keeps the complete fixed scope and
stops only after a new answer is durable. `--resume` reuses completed answers,
including malformed ones, and never repeats an ambiguous reservation. Model,
prompt, cropped pixels, seed, source and streaming settings remain bound to
the run identity. The operational chunk size may change between invocations.

`attempts.jsonl` and `executions.jsonl` are append-only. Each invocation has an
immutable `summary-invocation-NNNN.json`; partial scopes have `metric: null` and
explicit completed, unknown and unattempted coverage. Progress reports native
cumulative token counters rather than summing token chunks.

CPU synthetic streams reproduced all released mlx-vlm 0.6.8 aggregate result
fields across six cases, including final detokenizer flush and token cutoff.
Real-model streaming/sampling parity has **not** been measured. The controls
have CPU tests; their existence is not a trained-model result.

An additional `instruct` profile is now implemented for prospective evaluation.
It uses the author's native `enable_thinking=False` template and recommended
sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5,
repetition penalty 1.0, with the same 16,384-token limit. The author explicitly
documents this mode for image inputs in the
[pinned model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md#instruct-or-non-thinking-mode).
It has no measured inference speed or accuracy here. Xhigh remains the default
and a separate candidate; an incomplete xhigh scope cannot be treated as defeated.

`aircraft_local_probe.py plan-instruct` derives a new immutable prompt plan from
the fixed shared v2 scope. `prepare-mirror --profile instruct` requires that
separate plan; it rejects the v2 prompt that requires a thinking explanation.
The new plan retains the same pixels, question, full vocabulary and canonical
parser. Generation passes the entire direct answer to that parser, preserving
extra prose as invalid instead of repairing it. This compares complete candidate
configurations, not an isolated causal effect of thinking mode. No local27B
instruct inference has run.

The Aircraft wrapper currently imports the shared classification contract from
`src/vision_rl/vlm_local_training.py`: `OUTPUT_GRAMMAR`, `load_spec`, `prompt_text`,
`parse_label` and `classification_metrics`. Publish it together with a reviewed
version of that dependency. The core runner, supervisor, generic wheat preparer
and usage summary can be tested with that trainer source physically absent.

Local review evidence is indexed in
`runs/local-qwen8bit/publication-review.json`. The cancelled diagnostic,
usage report, input audit, CPU parity record and v2 preparation check remain
under `runs/local-qwen8bit/`; local run files are not automatically published.
