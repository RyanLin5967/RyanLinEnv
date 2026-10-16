# Aircraft: real GRPO updates, exact continuation, rejected precision change

The pretrained policy made task-reward updates and resumed exactly on the local
Apple MPS device. **This experiment did not establish an accuracy improvement or
an RL advantage over supervised training.** Both four-update arms remained at
3/4 greedy accuracy on these four training images; SFT had slightly lower final
gold-answer loss.

| Measured four-update run | GRPO | SFT |
| --- | ---: | ---: |
| Initial TRAIN gold-answer NLL | 0.0924918884 | 0.0924918884 |
| Final TRAIN gold-answer NLL | 0.0821458549 | 0.0819206293 |
| Greedy accuracy, before → after | 3/4 → 3/4 | 3/4 → 3/4 |
| Committed active training time | 69.7556 s | 11.1262 s |
| Saved/reloaded NLL | Exact agreement | Exact agreement |

The fixed model is Qwen2-VL-2B-Instruct at revision
`895c3a49bc3fa70a340399125c650a463535e71c`, with FP32 frozen weights and rank-8
language Q/V LoRA. Both arms start from the same
[400-update supervised warm start](../2026-10-16-aircraft-warmstart/README.md),
use the same image order, direct-answer prompt, full 100-class vocabulary and
learning rate `1e-5`. GRPO uses four samples per prompt, exact canonical-label
correctness, and KL coefficient `0.04` against the frozen common initialization.
The four images were selected deterministically from the preceding TRAIN assay's
useful reward groups. This adaptive mechanism diagnostic is not a random sample
of the benchmark, a held-out evaluation, or the strongest supervised comparison.

Three of the four GRPO groups contained both correct and valid incorrect labels.
The first group was uniformly correct and had zero task advantage: its tiny
numerical gradient is explicitly excluded from task-learning evidence. The
second group's rewards were `[1, 0, 0, 0]`, its logged KL was zero, and its
pre-clipping gradient norm was **6.7104**. All 16 completions terminated with EOS
and parsed as canonical labels. Across the four updates, 112 adapter tensors
changed. Initial policy/reference log probabilities agreed exactly; the measured
sampler-versus-teacher maximum log-probability error was **6.56e-6**, below the
frozen `0.001` tolerance. That check covers the recorded eight-token completion,
not every possible rollout.

An uninterrupted four-update trajectory and a fresh-process two-plus-two
continuation produced identical prompt IDs, sampled token IDs, answers, rewards,
EOS masks, immutable reference weights and every final adapter tensor
(`maximum_absolute_difference = 0`). The final adapter file hash for both is
`bd5018ecc0e365e607e847cde451ec352eacc11db40790c0bed775da45729b2d`.
The measured continuation scope is these four pretrained MPS updates; it does
not prove determinism for arbitrary hardware or training lengths.

The original multi-phase launcher failed after completing GRPO because
Accelerate changed an MPS environment flag before the next phase's preflight.
The failure log is retained. A separate execution amendment ran each remaining
phase in a fresh process with the original frozen scientific source and settings.
No completed GRPO work was silently repeated. The extra four GRPO updates used
for the continuation check are separate physical work, and neither diagnostic
endpoint becomes the initialization of the larger study.

A separately frozen BF16 trial completed four GRPO updates in **34.1230 seconds**,
a **2.044×** active-time speedup. It preserved all 112 initial FP32 LoRA tensors,
initial greedy predictions, the frozen reference and saved/reloaded loss.
However, its sampler-versus-teacher log-probability error was **0.0724754**,
exceeding the prospectively declared **0.02** precision tolerance. **Promotion was
rejected; the larger controlled experiment retains FP32.** Faster execution was
not allowed to override the recorded numerical gate.

[Full measurements](mechanism-summary.json),
[BF16 decision](bf16-promotion-result.json), and the
[independent verification](independent-review.json) accompany raw reward groups,
optimizer observations, predictions, prompt traces, endpoint receipts, timing
records, frozen plans, and both successful and failed supervision logs.
[The artifact manifest](artifact-manifest.json) maps each byte-exact copy to its
original local path. Archived Python sources have a `.py.txt` suffix and are
evidence snapshots. Model weights and dataset images are not included.
`SHA256SUMS` covers every other file in this report.

The substantial controlled RL gain, full native held-out evaluation and named
frontier comparison remain unfinished. Original-author image equivalence and
the full cross-split audit also remain pending at this report's publication.
