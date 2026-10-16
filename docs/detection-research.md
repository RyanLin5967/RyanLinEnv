# Research behind the detection environment

Primary sources checked on October 16, 2026 UTC. This is an independent Roboflow Labs onboarding project. The architecture and experiments here are our implementation; the public announcements do not establish that Roboflow uses this training stack internally.

## What Roboflow announced

Roboflow announced Labs on September 30, 2026 with three directions: open models, work with frontier labs on models and training data/environments, and enterprise evaluations for capabilities that remain weak in practice. That supports building a project around measuring a concrete visual gap, adapting a model, and verifying whether the adaptation closes it. It does not establish a publicly released, general-purpose Harbor RL trainer. [Roboflow Labs announcement](https://roboflow.com/blog/roboflow-labs-launch)

Roboflow Agent, announced October 14, is a closer product analogue for the outer workflow: it recommends models, creates vision workflows, and trains efficient models on user data. Our training agent runs explicit, reproducible commands and submits learned weights to Harbor. It is a learning environment rather than a reproduction of Agent. [Roboflow Agent announcement](https://roboflow.com/blog/roboflow-agent)

The useful onboarding questions are concrete: Which visual errors matter? Do the annotations express the intended task? Does a specialist improve native task performance? What did supervised learning contribute, and what did RL add? Can a clean environment reproduce the result from a numeric checkpoint?

## Why these frontier baselines

Roboflow Playground's object-detection evaluation, updated October 13, 2026, reports the following high-effort results. These are external benchmark measurements used to choose relevant current baselines, not pills scores or measurements from this repository. [Official object-detection evaluation](https://playground.roboflow.com/evals/object-detection)

| Model | Published high-effort AP50:95 |
| --- | --- |
| Gemini 4 Argon | 68.2 |
| GPT-6 Astra | 61.9 |
| Claude Opus 5.5 | 58.3 |

The external page also reports AP50, AP75, token use, estimated cost, and latency. Those values depend on its task mix and provider settings. Our experiment must call the named models on the actual matched images and record its own errors, usage, and timing. In particular, the public Claude “high” row is not equivalent to the proxy's adaptive/provider-default control.

## Why native detection

RF100-VL evaluates object detection across 100 datasets and seven domains. Its paper uses dataset-specific experiments and `pycocotools` with `maxDets=500`; it also studies rich annotation instructions and visual examples. Detection provides a measurable localization task where a plausible textual answer is insufficient. [RF100-VL paper](https://arxiv.org/html/2505.20612v1), [official implementation](https://github.com/roboflow/rf100-vl)

The primary experiment uses **countingpills-exf0r-gfkt, version 2**, and predicts a box for each pill. It uses the native COCO annotations and split membership. This is one RF100-VL dataset, not the 100-dataset aggregate, an HSS score, or a claim about general intelligence. The data version, downloaded archive, decoded images, labels, and preparation decisions are recorded by [`prepare_pills.py`](../scripts/prepare_pills.py).

Humanity's Sixth Sense remains a separate research target. Its public card supplies a test-only split with open-ended image/video questions and task rubrics; it does not supply a training set for this loop. A faithful HSS extension needs independent training data, the native media and grading protocol, and a separate evaluator. The detector score here cannot be converted into HSS accuracy. [Official HSS dataset card](https://huggingface.co/datasets/ScaleAI/HSS)

The downloaded export contains 1,449 training images, 423 validation images, and 200 test images, with 6,154, 1,820, and 987 boxes respectively. Its README declares 2,186 images while the actual COCO/archive membership is 2,072. The audit retains the actual membership and reports that disagreement; it does not invent missing examples or treat absent annotations as negatives.

Exact file, decoded-pixel, and original-name checks found no overlap across the native splits. Filename-derived capture-hour groups do overlap: 34 groups span splits. Consequently, a native-split result measures performance within this data collection, not generalization to unseen capture sessions, cameras, or sites. The final protocol resamples capture-hour groups and puts all images without a recoverable group into one conservative shared cluster. Earlier development reports may use singleton unknown groups; their grouping is recorded separately. Clustering handles some dependence in uncertainty estimates but does not remove training/test session overlap.

The [timestamped grouping amendment](detection-cluster-preregistration.json) records that change before final evaluation. The frozen final `capture-groups.json` and comparison reports retain the exact grouping and resampling-unit counts, so the uncertainty calculation can be reproduced.

The fixed visual review sampled 12 training and 12 validation images with annotations. No obvious label-scope error was found in those contact sheets. A sample review is limited evidence, not a guarantee that every annotation is correct. Native test images are excluded from the review sheets.

## A second dataset exposed a different problem

The exploratory stomata dataset, **stomata-cells-upfae-ngum version 1**, contains 1,473/421/211 native train/validation/test images. All 117 recovered source-frame groups cross the native split boundaries. In particular, 128 of 199 native test crops have their corresponding full frame in training, and all 12 native test full frames have training crops. Exact image deduplication alone would miss this relationship.

[`prepare_stomata.py`](../scripts/prepare_stomata.py) therefore also creates a separately named experiment, `rf100vl-stomata-group-heldout-v1`. A fixed seed assigns entire source frames to 82/18/17 train/validation/test groups, giving 1,464/332/309 images. This changes the benchmark split and must never be reported as the native RF100-VL result. Three images assigned to its test set had appeared in an earlier, native-split annotation review; that exposure is disclosed rather than hidden. Only five capture hours are represented overall, so held-out source frames still do not establish cross-site generalization.

This audit is a reason to broaden the research process: the most valuable improvement may be a better evaluation boundary before it is a larger model.

## What is trained

The main detector starts from public RF-DETR Nano weights and fine-tunes the complete transformer: the measured configuration has **30,147,076 trainable parameters**. It uses supervised bounding-box and classification losses, an exponential moving average, validation selection, and a resumable optimizer/scheduler checkpoint. This follows RF-DETR's supported training stack, with our own data isolation, telemetry, and artifact export. [RF-DETR training documentation](https://rfdetr.roboflow.com/latest/learn/train/)

The RL stage freezes that detector and caches its top 128 foreground queries, including boxes, logits, and 256-dimensional decoder embeddings. A contextual policy chooses among 81 bounded box-geometry actions and learns query confidence. Its defaults use eight sampled groups, a leave-one-out reward baseline, a KL penalty, entropy regularization, and gradient clipping. The policy is small enough to iterate on CPU while full detector training uses an accelerator.

Perception-R1 provides relevant evidence for visual rewards: it applies GRPO to multimodal models, uses geometric correctness signals, and emphasizes matching multiple predictions to ground truth. It also shows why localization alone is insufficient when object count and missed detections matter. Our implementation uses a detector-query policy and COCO-style unique matching with group-relative REINFORCE. It is not a reproduction of that paper's autoregressive VLM training or its reported results. [Perception-R1 paper](https://arxiv.org/html/2504.07954v1), [authors' code](https://github.com/linkangheng/PR1)

The implemented reward combines multi-IoU detection utility, image-level interpolated AP, and a small IoU shaping term. Matching follows score-ordered COCO-style greedy assignment; each target can be matched only once, so duplicate proposals do not earn repeated credit. The score-function gradient uses the full sampled image reward; treating dependent matches as independent per-query rewards would bias it. Tests check this gradient against exact enumeration on a small policy. The training reward remains a proxy: image-level AP is not the pooled dataset AP used for validation selection and final reporting.

An optional supervised oracle warmup teaches useful geometry actions from training labels. That stage is supervised learning, not RL. Reports retain the raw detector, warmup, best RL checkpoint, and validation-selected checkpoint separately. If RL lowers AP, the selected model may remain the supervised model; a frontier comparison cannot by itself establish an RL benefit.

## How the comparison earns its claim

The study chooses prompts and model settings on a fixed 48-image validation discovery sample. The predeclared prompt family includes minimal instructions, richer annotation instructions, and rich instructions with two training examples. Each baseline uses its highest observed validation AP configuration before test settings are frozen; discovery failures remain in the reports. The specialist may use all labeled training data; the comparison therefore measures task adaptation versus the declared frontier prompting budget, not equal training-data access.

The final native score is `AP@[0.50:0.95]` with 101 recall points and at most 500 detections per image. AP50 and AP75 are diagnostic companions. Provider-specific prompts declare the box convention before responses are converted to native COCO pixels: Gemini/Claude use normalized `[ymin, xmin, ymax, xmax]`; Astra uses absolute `[xmin, ymin, xmax, ymax]`. Missing confidence is assigned 1.0, consistent with the RF100-VL evaluation description. The implementation preserves low-confidence detector proposals for AP rather than selecting a favorable score threshold after observing test results. [RF100-VL evaluation details](https://arxiv.org/html/2505.20612v1#A2)

The evaluator independently tests several easy-to-miss details: annotation ID zero is remapped to avoid COCO's unmatched sentinel; empty predictions retain their images; arbitrary valid native detector coordinates are preserved; and the 500-detection AP is read from the correct precision tensor rather than a summary field hardcoded for 100 detections. Cluster bootstrap recomputes pooled AP under resampling, rather than averaging per-image AP.

The fixed objective is specialist AP of at least 0.80, a margin of at least 0.20 against each named frontier baseline, and a paired 95% lower confidence bound of at least 0.10. Zero API/parse failures are required for the capability-win gate. These are acceptance criteria, not promised outcomes. Validation measurements guide development; the final claim requires the frozen all-200-image test evaluation.

The baseline adapters identify Gemini 4 Argon, GPT-6 Astra, and Claude Opus 5.5 through Roboflow's native provider responses. The Claude proxy exposes adaptive thinking without an effort override, so its setting is reported as **adaptive/provider default**, never relabeled “high.” Request journals preserve provider identity, available usage, transport failures, and retries. A response with missing cost or reasoning-token fields remains unknown; those values are not estimated as zero. The adapter is pinned to [Roboflow Inference commit `1505779`](https://github.com/roboflow/inference/tree/150577969c6dbe5c0da854942c5961fe3f293088).

## Why Harbor and an experiment loop are separate

Harbor provides the task, container, agent, and verifier lifecycle. The experiment loop schedules candidates, journals attempts, enforces budgets, selects on validation, and freezes artifacts. PyTorch performs the weight updates. These responsibilities are independently inspectable. [Harbor core concepts](https://docs.harborframework.com/core-concepts)

The detector task builds an offline training image containing only train and validation data. Harbor grades numeric-only submissions in a separate pristine verifier with test images. The verifier reconstructs a fixed model architecture from installed code; it does not execute submitted Python, configurations, or pickle checkpoints. That boundary is stronger than the local training directory convention, where a trusted process still has the user's ordinary filesystem access.

The persistent loop can resume development after interruption, but it does not keep testing new candidates against a consumed final set. A failed frozen final evaluation remains evidence. Further optimization belongs to a new experiment with a fresh held-out evaluation. See the [walkthrough](detection-onboarding.md) for executable commands and the limits of each path.
