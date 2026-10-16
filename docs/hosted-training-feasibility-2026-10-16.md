# Roboflow hosted training feasibility

Checked October 16, 2026 using official documentation and public source only. No account requests, project changes, uploads, jobs, purchases, inference, or sealed-test access were performed for this research.

**Hosted RF-DETR is a feasible additional validation experiment with existing credits, but custom hosted resolution, remaining account credits, and a capped recipe submission still require verification.** It does not yet establish an improvement over the local detector and native-pixel refinement work.

## Confirmed product controls

Core, including Free Tier, supports RF-DETR Nano, Small, Medium, and Base training. The service accepts a saved dataset version and a public or previous checkpoint. It exposes a customizable training recipe, including hyperparameters, preprocessing, and online augmentation. Saved-version preprocessing and augmentation are already applied and locked. An optional per-job credit cap stops training while retaining weights; each model has a minimum cap, shown in the UI. No fixed numerical minimum is published. A below-minimum request is rejected with `invalid_credit_cap`. The API key needs `trainingJob:create`. [Training documentation](https://docs.roboflow.com/models/train/train-a-model)

The pricing page includes private workspaces, training, and weight download on every plan; Free Tier includes ten credits per month. This is a plan entitlement, **not this account's current balance**. Priority access to faster GPUs is an Enterprise feature; the reviewed API does not offer a GPU-type selector. [Current pricing](https://roboflow.com/pricing)

## Exact API and SDK routes

The Python SDK source is pinned to commit `72e8c01e0b8332da6c0c8514803345e3a0df46be`. All platform paths below use `https://api.roboflow.com`; authentication is the SDK's existing API-key mechanism. Never place credentials in committed manifests or logs.

| Operation | Confirmed route / SDK method |
| --- | --- |
| Describe tunable recipe | `GET /{workspace}/{project}/{version}/v2/trainings/recipe?modelType=rfdetr-medium`; `version.describe_train_recipe("rfdetr-medium")` |
| Create asynchronous training | `POST /{workspace}/{project}/{version}/v2/trainings`; `version.create_training(...)` |
| Retrieve resulting models | `training.refresh()` and `training.models` |
| Download selected model | `model.download(format="pt", location=...)`; resolves `GET /{workspace}/{project}/{model-address}/ptFile` to a signed `weightsUrl` |

The v2 creation body uses `modelType`, `checkpoint`, `epochs`, `speed`, and `trainRecipe`. The recipe endpoint returns its tunable schema, allowed preprocessing/augmentation, and an editable `template`. Put epochs in `trainRecipe.hyperparameters.epochs` when supplying a recipe: recipe defaults otherwise take precedence over top-level epochs. The current SDK folds epochs into a recipe for this reason. [Pinned API adapter](https://github.com/roboflow/roboflow-python/blob/72e8c01e0b8332da6c0c8514803345e3a0df46be/roboflow/adapters/rfapi.py#L260), [version interface](https://github.com/roboflow/roboflow-python/blob/72e8c01e0b8332da6c0c8514803345e3a0df46be/roboflow/core/version.py), [epoch precedence](https://github.com/roboflow/roboflow-python/blob/72e8c01e0b8332da6c0c8514803345e3a0df46be/roboflow/util/train_recipe.py)

There is a compatibility gap: public HTTP documentation exposes `trainingCreditCap` on legacy `POST /{workspace}/{project}/{version}/train`, while the inspected v2 SDK signature does **not** forward credit-cap arguments. The legacy documentation spells the model field `model_type`, while the SDK adapter sends `modelType`. Do not silently assume that a hand-added cap or resolution works on v2. Confirm accepted fields and persist the returned job configuration before launching a run. Also prefer `create_training` for asynchronous orchestration: the older `Version.train` implementation polls until completion despite the website describing queue-only return semantics.

## Resolution and backbone limits

The server-provided recipe is the authoritative next check for **hosted** `384`, `512`, or `768` training. Public source does not establish those arbitrary hosted overrides. Resizing a dataset version to 768 pixels alone does not prove the model consumes 768-pixel tensors.

The local RF-DETR package explicitly supports custom square resolutions divisible by `patch_size * num_windows`. Its current source shows the following defaults:

| Variant | Resolution | Decoder layers | Encoder | Divisibility |
| --- | ---: | ---: | --- | ---: |
| Nano | 384 | 2 | `dinov2_windowed_small` | 32 |
| Small | 512 | 3 | `dinov2_windowed_small` | 32 |
| Medium | 576 | 4 | `dinov2_windowed_small` | 32 |
| Base | 560 | 3 | `dinov2_windowed_small` | 56 |

Thus 768 is valid for local Nano/Small/Medium, but not local Base with its default architecture. These four variants do **not** provide progressively larger encoders. The local Large configuration also uses the small encoder at 704 pixels; its hosted availability was not established. [Local training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/), [pinned architecture configuration](https://github.com/roboflow/rf-detr/blob/a0065d0acaa3663ce5ff09fe91d08d1b2a4046c3/src/rfdetr/config.py#L996)

## Export and replay

Standard RF-DETR supports raw PyTorch weight download on Core/Free through the SDK. The product also lists ONNX/CoreML exports, but this SDK download method currently accepts only `pt`. RF-DETR NAS and SAM3 raw weights cannot be downloaded; avoid those for this independent-replay objective. Download permission does not override commercial-license terms. [Weight-export documentation](https://docs.roboflow.com/models/model-weights/download-roboflow-model-weights), [pinned download implementation](https://github.com/roboflow/roboflow-python/blob/72e8c01e0b8332da6c0c8514803345e3a0df46be/roboflow/core/training.py#L162)

A reproducible run should preserve the returned recipe, class map, dataset split/file checksums, model identity, raw weights and hash, package versions, and preprocessing. Recompute predictions and native COCO metrics locally on the existing validation IDs. Hosted dashboard metrics alone cannot establish parity with this project's evaluation protocol.

## Credits and the read-only preflight

Published GPU training costs **one credit per 30 minutes**, or two per GPU hour; no separate rental is required. Uploads, version generation, and storage also consume credits. Actual job duration and credit estimate depend on dataset and model, so there is no defensible pills-specific cost estimate yet. [Credit rates](https://roboflow.com/credits)

Official read/report routes are `GET /usage/plan` (`workspace.get_plan()`) and `POST /{workspace}/billing-usage-report` (`workspace.get_usage()`). The latter is a usage retrieval despite its POST verb. Documentation describes included credits, feature limits, and consumed training/inference usage, but does not specify a stable, single `remaining_credits` field or an atomic reservation endpoint. Inspect those responses and current pending usage before choosing a cap. [Usage API documentation](https://docs.roboflow.com/platform/billing-and-plans/credits/view-credit-usage)

Included credits are consumed before prepaid credits; enabled Flex can charge beyond both on paid plans. Free Tier has no Flex. The user's authorization covers existing credits only, so any eventual run must fit the verified available allocation, accounting for ongoing baseline inference and import costs. A per-run cap alone does not reserve the same credits against other jobs. [Credit billing rules](https://docs.roboflow.com/platform/billing-and-plans/credits)

## Experiment judgment

Given the lead's current validation findings—high recall with remaining box-localization error—a hosted Medium run is a reasonable test of higher input resolution and a deeper decoder. It is **not evidence of a larger visual backbone** and cannot guarantee the required AP gain. Native-pixel crop refinement directly targets localization at the existing detections; its validation ablation is therefore at least as relevant. Compare both on unchanged development data, preserve latency/credit measurements, and keep the final 200-image pills test sealed until the full selection protocol is frozen.
