# Wheat native-detector RL: no-go

The native Gaussian policy experiment **did not meet the RL requirement**. After 20 committed updates, full validation accuracy declined under the same calibration search. The intended 64-update study stopped at its trust-region guard; neither matched supervised control was run. There is no completed-study or frontier-superiority claim.

The benchmark is `gwhd2021-hf-domain-ada-v1`: the authors' Hugging Face split evaluated with unchanged WILDS 2.0.0 equal-domain detection accuracy. It is distinct from the WILDS-native and competition leaderboard protocols. Development used 3,655 training and all 1,476 validation images across 11 validation domains. The 1,381 test images remain sealed.

The starting model was the fixed epoch-3 EMA of a fresh COCO-pretrained RF-DETR Nano512, within a planned 30-epoch supervised schedule. The reward policy trained 4,288,438 native decoder/query/class/box parameters, retained all 300 queries, and used the official image-level reward, importance-weighted counterfactual credit assignment, and analytic reference KL. It was not a frozen-feature postprocessor.

## Measured validation results

Both models were evaluated on CPU with the same complete validation set and predeclared seven-offset confidence calibration. Both selected a foreground-logit offset of `+0.5`.

| Metric | Initial detector | Partial 20-update detector | Difference |
|---|---:|---:|---:|
| Calibrated equal-domain accuracy | 65.2140% | 64.8293% | **−0.3847 points** |
| Raw accuracy, zero offset | 62.8271% | 63.6945% | +0.8674 points |
| Supplemental COCO AP | 48.8619% | 48.7610% | −0.1009 points |

The paired acquisition-domain bootstrap used all 11 domains, 20,000 draws, and seed 2718. Its 95% interval for the calibrated difference was **[−1.5314, +0.3315] percentage points**. The interval includes zero and does not correct validation selection bias. Neither the required five-point gain nor a positive lower confidence bound was achieved.

## What actually ran

The unchanged exploration gate passed: 128 domain-balanced training draws, containing 121 unique images, found a mean best-of-64 gain of 5.3134 points at the selected two-pixel scale. This was training-only sampling headroom, not a learned-model result. The one- and four-pixel alternatives failed the same five-point gate.

The first MPS fit failed when the frozen-reference output validity check rejected geometry or logits. Its durable checkpoint contains zero updates; the number of uncommitted updates is unknown. Forward-only probes did not reproduce the failure, so its exact cause remains unresolved. No invalid boxes were clipped or replaced.

A prospective CPU fallback kept the same scientific settings for all intended arms. It committed **20 updates**, accounting for **24.8127 seconds of committed active optimization**. This excludes setup, validation, artifact I/O, diagnostics, and discarded uncommitted work; it is not a total-compute measurement. Replaying that state accepted diagnostic updates 21 and 22, then rejected proposal 23: none of the 12 halved Adam proposals satisfied every per-image reference-KL limit. The evaluated checkpoint is the durable step 20, independently exported and reloaded as safetensors.

An intermediate status message incorrectly inferred that the CPU checkpoint also contained zero updates. That led to a mistaken initialization-mismatch diagnosis. Direct inspection corrected the count to 20; fresh identical models produced zero KL, and all 466 initial tensors matched the reference. The correction and diagnostic artifacts are retained.

## Training audit and decision

The independently fixed 32-image training sentinel covered all 18 training domains. With 16 samples per image and common random numbers, expected reward increased by **0.007822**, with conditional Monte Carlo noise standard error **0.001295**. That standard error measures action-sampling noise on these fixed training images. It does not measure dataset-sampling uncertainty or generalization.

Deterministic sentinel accuracy increased from 0.783051 to 0.786303. Substituting only the updated confidence outputs reproduced the entire increase; substituting only updated geometry reproduced the initial score. These are output-component diagnostics, not separately trained ablations. The drift alarm also failed: maximum per-image mean-query KL was **1.7813**, above its fixed limit of 1.0; the overall mean was 0.1440.

The experiment learned a confidence change that improved raw scores, but it did not improve the fairly calibrated validation result. This branch is stopped. A proposal for explicitly recording rejected attempts remains a design document only; no second fit or relaxed guard was introduced. The equal-update and equal-walltime supervised controls are **incomplete**, and no claim against the eventual strongest supervised checkpoint is made. The broader project continues with the aircraft approach.

The [machine-readable report](wheat-direct-rl-outcome-2026-10-16.json) includes exact metrics, configurations, source and artifact hashes, and evidence paths. The [prospective protocol](wheat-direct-rl-protocol-v1.json), frozen trainer, failed runs, and numeric checkpoints are preserved.
