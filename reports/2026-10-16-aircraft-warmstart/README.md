# Aircraft warm start and reward support

The 2B policy now produces enough correct and incorrect aircraft labels to test
reinforcement learning. **This report contains supervised training and training-data
diagnostics; it does not establish an RL improvement or a frontier-model win.**

The experiment uses FGVC-Aircraft 2013b's native 100 variants and a prospectively
selected four-shot training set: four images per class, 400 images total. The
original author archive and full cross-split image audit are still pending.
Validation and test images have not been used by these experiments.

## Common initialization

Qwen2-VL-2B-Instruct received one supervised pass over those 400 images, with
rank-8 language Q/V LoRA, float32, LR `0.0001`, and seed 42. All subsequent RL
and supervised arms must start from the exact same saved adapter.

| Fixed eight-image TRAIN diagnostic | Before | After |
| --- | ---: | ---: |
| Greedy correct answers | 2/8 | 4/8 |
| Gold-answer negative log likelihood | 0.288425 | 0.172467 |
| Invalid answers | 0/8 | 0/8 |

All 400 optimizer updates had finite nonzero gradients. The saved adapter
reproduced the final diagnostic loss exactly after reloading. Committed active
training time was **1,104.81 seconds**; total supervised wall time was
**1,157.24 seconds** on Apple M5 Pro. The 400-presentation count is derived from
the configuration; this historical run predates the actual prompt-ID ledger.

Adapter SHA-256:
`f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c`.

## Prospective prompt comparison

Before seeing the warmed model's outcomes, the
[amendment](assay-amendment.json) fixed 32 training images, two seeds, and four
samples per image for each prompt. Both prompts use all 100 labels, the same
image processing and adapter, and the same exact-label parser. An answer must
terminate normally; invalid answers count as wrong. No format reward is added.

| TRAIN samples from the warmed policy | Reasoning request | Direct-answer request |
| --- | ---: | ---: |
| Correct | 142/256 (55.47%) | 181/256 (70.70%) |
| Canonical valid answers | 212/256 (82.81%) | 243/256 (94.92%) |
| EOS terminated | 256/256 | 256/256 |
| Groups with a valid correct and valid wrong answer | 22/64 | 22/64 |
| Distinct images with such groups | 16/32 | 14/32 |
| Preset support gate | Fail: validity below 90% | Pass |

These are individual sampled-answer accuracies, not best-of-four scores or
held-out estimates. The two seeds produce repeated samples of the same images.
All 512 raw parses and rewards were independently replayed exactly.
The full assay took **492.65 seconds** and performed no optimizer updates.

The preset rule selects the direct-answer prompt because the reasoning request
failed its validity gate. The next experiment is therefore **task-reward token
RL**, with no reasoning-mechanism claim. Its gains must be measured against
continued supervised training from this shared initialization.

## Inspect the evidence

- [Machine-readable summary](summary.json) and [independent score review](independent-score-review.json).
- [Warm-start outcomes](warmstart/phase-summary.json), [training metadata](warmstart/local_training.json), and [all optimizer observations](warmstart/optimizer_audit.jsonl).
- [Per-phase support outcomes](support/phase-summary.json) and [gate decision](support/gate-result.json).
- All sampled answers and token IDs under `support/{answer_only,think_then_answer}/seed-{42,314}/reward_groups.jsonl`.
- Frozen launch plans, exact source archives, and memory/time supervision records under each phase. Source archives end in `.py.txt` and retain the original bytes.
- [Artifact manifest](artifact-manifest.json) binds every copied evidence file to its SHA-256 and original local path. `SHA256SUMS` also covers this report and summary.

No source images, credentials, or model weights are included. Input preparation
is documented in the [Aircraft data guide](../../docs/fgvc-aircraft-image-preparation-2026-10-16.md).
The [strategy review](../../docs/rl-strategy-review-2026-10-16.md) explains the
differences from published Visual-RFT results, including their different scorer.
