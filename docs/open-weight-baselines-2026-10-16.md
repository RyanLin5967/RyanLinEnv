# Open-weight baseline and public benchmark research

Research checked October 16, 2026. This document records source inspection, not model evaluation. No benchmark images or model weights were downloaded and no inference calls were made for this research. The native 200-image pills test set remains outside this work.

## Recommended comparison

Use both **Qwen3.8-27B** and **Kimi K3** on the same validation discovery images before freezing final settings. Qwen has the better published detection result among the released vision-weight candidates checked here; Kimi supplies a current large flagship MoE comparison. Neither published result is a pills score or an RF100-VL aggregate.

| Model | Actual released artifact | License and modality | Practical route |
| --- | --- | --- | --- |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Apache-2.0; native image/text model | Managed Roboflow OpenRouter, native Roboflow model, or local MLX quantization |
| Kimi K3 | `moonshotai/Kimi-K3`, revision `f831ab66814297da540d832a5235f8e904f29d06` | Custom Kimi K3 License; native image/text model | Managed Roboflow OpenRouter; unsuitable for this Mac locally |

Qwen's official card provides multimodal Transformers, vLLM and SGLang examples. Its original safetensors total **55,563,006,776 bytes**, measured through Hugging Face metadata, so the full precision weights exceed a 48 GB machine before runtime overhead. [Official model and license](https://huggingface.co/Qwen/Qwen3.8-27B)

Kimi K3 has 2.8 trillion total parameters, 104 billion active parameters, and a released MoonViT-V2 vision encoder. Its 96 safetensors total **1,560,936,091,448 bytes**, measured through metadata. It uses native MXFP4 weights and MXFP8 activations. The license includes commercial service conditions; describe this model as **open-weight**, without implying an OSI license. [Official model card](https://huggingface.co/moonshotai/Kimi-K3), [actual license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)

The official Roboflow object-detection table, updated October 13, reports Qwen3.8-27B high effort at **49.1 AP50:95**, Kimi K3 low effort at **40.4**, Qwen3.8-Max high at **62.3**, and Gemini 4 Argon high at **68.2**. Kimi's published detection row is only low effort, so it does not justify restricting our comparison to low effort. Its official API supports low/high/max, defaults to max, and always enables thinking. [Roboflow detection evaluation](https://playground.roboflow.com/evals/object-detection), [Kimi usage](https://huggingface.co/moonshotai/Kimi-K3#6-model-usage)

## Exact managed route and inference contracts

Source pinned to Roboflow Inference commit `150577969c6dbe5c0da854942c5961fe3f293088`:

- [Managed OpenRouter request implementation](https://github.com/roboflow/inference/blob/150577969c6dbe5c0da854942c5961fe3f293088/workflows/roboflow_workflows/core_steps/common/openrouter.py).
- [Qwen model mappings and prompt contract](https://github.com/roboflow/inference/blob/150577969c6dbe5c0da854942c5961fe3f293088/workflows/roboflow_workflows/core_steps/models/foundation/qwen_vlm/v4.py).
- [Kimi block](https://github.com/roboflow/inference/blob/150577969c6dbe5c0da854942c5961fe3f293088/workflows/roboflow_workflows/core_steps/models/foundation/kimi_openrouter/v3.py).

The platform-client endpoint is `apiproxy/openrouter`, authenticated with the existing Roboflow platform API key. The source passes this payload:

```json
{
  "openrouter_api_key": "rf_key:account",
  "model": "qwen/qwen3.8-27b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "EXPLICIT DETECTION INSTRUCTIONS"}
      ]
    }
  ],
  "max_tokens": 16384,
  "privacy_level": "deny",
  "reasoning": {"effort": "xhigh"},
  "quantizations": ["fp8", "bf16", "fp32"]
}
```

`rf_key:account` is a managed-key selector, not a credential. The raw API key must remain in the existing authentication mechanism. The source says the managed route bills Roboflow credits. It caps `max_tokens` at 16,384; omission can lead to a 500-token proxy default and truncated detection JSON. Temperature is optional and omitted when unset. Responses expose `choices[0].message.content`, optional reasoning, and usage. Preserve the native response and actual model identity. The platform may reject unsupported fields or strip reasoning on older versions; record any fallback instead of silently labeling it high effort.

For Qwen, use a single user message with **image before text** and no system message. Its benchmarked detection convention is a JSON array of `{ "label": "pill", "box_2d": [xmin, ymin, xmax, ymax] }`, with integer coordinates normalized to **0–1000**. This is XYXY and differs from the existing Gemini YXYX convention. Convert explicitly, without coordinate autodetection. The source also maps native Roboflow model ID `qwen3_8-27b`; the hosted OpenRouter slug is `qwen/qwen3.8-27b`.

**Parser amendment before final evaluation:** Qwen's own grounding cookbook uses `bbox_2d`, not the `box_2d` spelling requested by our Roboflow-derived prompt. It specifies XYXY coordinates normalized to 0–1000; its plotting arithmetic confirms that order. The Qwen3.8 card itself does not specify a separate grounding JSON schema. The shared parser now accepts exactly one of `box`, `box_2d`, or `bbox_2d` for every array coordinate format. Spelling never selects the scale or axis order, mixed aliases fail, and reversed coordinates still fail. The amendment addresses a formatting incompatibility observed on development responses; old raw responses/reports remain intact and offline rescoring must produce separate artifacts. Native COCO scoring and the final-test split are unchanged. [Pinned author grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/cookbooks/2d_grounding.ipynb)

**Current effort metadata supersedes the stale block enum.** The read-only OpenRouter model listing checked at `2026-10-16T04:16:11Z` lists Qwen efforts `xhigh`, `medium`, `low`, `none`, default `xhigh`; Kimi K3 efforts `max`, `high`, `low`, `none`, default `max`. The pinned Qwen block offers `high`, which is absent from the current Qwen listing. Use explicit `reasoning: {"effort": "xhigh"}` for the strongest Qwen request, or `{"effort": "max"}` for Kimi through the generic proxy; verify acceptance before a run. The router metadata calls Kimi reasoning optional while its author card says always enabled, so use high/max and preserve that discrepancy. [Snapshot](open-weight-routing-metadata-2026-10-16.json), [official reasoning schema](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)

For Kimi K3, use model `moonshotai/kimi-k3` and a separately validated high/max reasoning request. Do not reuse Qwen's precision allowlist: Kimi's native quantization is different. The pinned Kimi-specific block only enumerates K2.5/K2.6 and uses named normalized 0–1 box fields. The current Playground explicitly offers K3 workflows using Roboflow credits, and OpenRouter advertises the K3 image-capable endpoint, but the generic managed route still needs an account-specific smoke test by the baseline runner. [K3 Roboflow deployment](https://playground.roboflow.com/models/moonshot-ai/kimi-k3), [OpenRouter endpoint](https://openrouter.ai/moonshotai/kimi-k3)

The current product deployment routes, after deploying their workflows into the workspace, are:

```text
POST https://serverless.roboflow.com/<workspace>/workflows/qwen3-8-27b-open-prompt
POST https://serverless.roboflow.com/<workspace>/workflows/kimi-k3-open-prompt
```

Both accept `api_key` in the request body and `inputs.image` as `{type: "url" | "base64", value: ...}` plus `inputs.prompt`. Object-detection workflow counterparts end in `-object-detection` and take `inputs.classes`. These documented routes do not prove the workflows already exist in this account. [Qwen deployment](https://playground.roboflow.com/models/qwen/qwen3-8-27b), [Kimi deployment](https://playground.roboflow.com/models/moonshot-ai/kimi-k3)

## Local fallback and runtime limits

`mlx-community/Qwen3.8-27B-4bit`, revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, contains **16,054,541,349 weight bytes** across three safetensors. The quantizer's card identifies the upstream Apache-2.0 model and `mlx-vlm` 0.6.8 conversion. This is a plausible 48 GB Mac fallback with bounded image/context sizes; runtime memory and latency still require measurement. Schedule it after releasing training allocations if memory contention occurs. Quantized and hosted runs must be named separately. [Quantization card and invocation](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)

No local throughput has been measured. Roboflow's published Qwen high-effort detection latency is 39.37 seconds per sample; this describes that external service workload and does not predict Mac throughput or our dense-image output length.

### Preferred local candidate after hosted credits were exhausted

The lead selected **`mlx-community/Qwen3.8-27B-8bit`**, revision **`815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9`**, to reduce the precision compromise. Its six weight shards total **29,501,218,479 bytes** (27.48 GiB); the full repository totals **29,531,521,322 bytes**. Quantization is affine 8-bit with group size 64. This is a community quantization of the author's released vision model, not a claim of identical hosted or full-precision performance. [Pinned quantization card](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit/tree/815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9), [expected file hashes and template verification](local-qwen3.8-27b-8bit-provenance.json)

The 36 GiB free-disk observation makes one copy plausible, with a required 6 GiB reserve. Download into a single local directory, account for the isolated environment and temporary files, and avoid a second cached model copy. Do not retain both 4-bit and 8-bit checkpoints under that disk budget. These are planning constraints, not measured runtime memory. The 48 GB machine must first release the detector/RL accelerator slot; start with two training images and record peak memory, latency, termination reason, and tokens before any validation run. If 8-bit cannot fit, retain the actual failure evidence and name a 4-bit fallback separately.

Use a separate Python 3.12 environment with **`mlx-vlm==0.6.8`**, then preserve its actual dependency lock. Its published wheel requires Python >=3.10, `mlx>=0.32.0`, `transformers>=5.14.0`, `mlx-lm>=0.31.3`, and `mlx-audio>=0.4.3`, among other dependencies. The wheel contains the `qwen3_5` multimodal implementation matching this checkpoint's configuration; `language_model_only` is false. This is source-level compatibility, still subject to the measured runtime probe. [Package metadata](https://pypi.org/project/mlx-vlm/0.6.8/), [runtime implementation](https://github.com/Blaizzy/mlx-vlm/tree/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/models/qwen3_5)

The quantized and author checkpoints have byte-identical `chat_template.jinja` and `preprocessor_config.json`, verified by SHA256. Use the shipped Qwen3VL processor on each original image, with `resize_shape=None`: patch size 16, spatial merge 2, temporal patch 2, RGB means/stds 0.5, pixel-area bounds 65,536–16,777,216. The processor performs its own resizing; record its resulting image grid and the original dimensions. The upper bound is an area, not a 16-million-pixel edge length. [Pinned image processor](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit/blob/815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9/preprocessor_config.json)

**Enable thinking explicitly.** MLX-VLM's template helper otherwise defaults it off. The model template consumes `reasoning_effort="xhigh"`, inserts the corresponding instruction, and opens the assistant's thinking segment. A sandboxed template-only check verified those effects, image-before-text ordering, and rejection of unsupported `high`; it did not load weights or run the model. The author recommends thinking-mode sampling at temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, and repetition penalty 1. [Author settings](https://huggingface.co/Qwen/Qwen3.8-27B#api-usage), [pinned template](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit/blob/815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9/chat_template.jinja), [MLX template helper](https://github.com/Blaizzy/mlx-vlm/blob/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/prompt_utils.py)

The exact 0.6.8 Python interface supports this configuration:

```python
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

# local_snapshot contains the single verified pinned download.
model, processor = load(local_snapshot, strict=True, trust_remote_code=False)
formatted = apply_chat_template(
    processor, model.config, detection_instructions, num_images=1,
    enable_thinking=True, reasoning_effort="xhigh", preserve_thinking=True,
)
assert "Reasoning effort is set to xhigh." in formatted
assert formatted.endswith("<|im_start|>assistant\n<think>\n")
mx.random.seed(image_seed)
result = generate(
    model, processor, formatted, image=image_path, resize_shape=None,
    max_tokens=16384, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
    presence_penalty=0.0, repetition_penalty=1.0, enable_thinking=True,
    seed=image_seed, verbose=False,
)
```

The 16,384 generated-token limit is a declared local experiment budget, not the author's maximum. Do not force an earlier thinking cutoff or silently truncate an answer. Preserve the complete raw output, explicitly separate the final segment after `</think>`, and send only that segment to the detection parser. Missing thinking closure, malformed JSON, and `finish_reason="length"` are explicit failures, even if a partial detection list can be recovered. Record `prompt_tokens`, `generation_tokens`, `prompt_tps`, `generation_tps`, `peak_memory` (decimal GB), wall time, and exact package/model hashes. The API returns these generation statistics; this research supplies no observed throughput or AP.

## Misleading open-model labels to avoid

- **Qwen3.8-Max:** the released `Qwen/Qwen3.8-2.4T-A95B` card explicitly describes a text-only checkpoint and says the hosted Max version adds vision. It also has a custom license. A hosted Max vision result is useful, but cannot establish performance of released vision weights. [Author's checkpoint card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- **Mistral Large 4:** the official checkpoint page currently says upcoming release, with October 31, 2026 ETA. Its Playground categorization does not establish that weights have already shipped. [Official pending release](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
- **Qwen3.8-Flash-Next:** actual native vision weights exist, but use the custom Qwen Community License. Its published detection result is 40.8 AP50:95 at low effort, so model size alone does not make it a stronger detection baseline than 27B. [Official checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [detection table](https://playground.roboflow.com/evals/object-detection)

## Public benchmark pivots

**First choice: Global Wheat Head Dataset 2021.** The author's release has 3,655 training, 1,476 validation, and 1,381 test images. Its declared download size is **1,260,938,177 bytes**, license CC-BY-4.0. Images are 1024×1024, with native wheat-head boxes and acquisition-domain metadata. The competition uses different geographic regions for training and evaluation and describes 44 acquisition sessions. This is a better documented independence boundary than randomly splitting crops from the same microscopy frame. Before training, verify that the current author's HF split membership and domain metadata match the intended competition protocol; preserve original image IDs and whole-domain groups. [Author release](https://huggingface.co/datasets/Etienne-David/GlobalWheatHeadDataset2021), [official challenge](https://www.aicrowd.com/challenges/global-wheat-challenge-2021)

The native metric is **Average Domain Accuracy**, using matches above IoU 0.5, per-image `TP / (TP + FP + FN)`, then image averaging within domains and equal averaging across domains. COCO AP can be a supplemental measure but is not the official challenge score. Existing detector/crop prediction code fits this task; add the native metric, validate threshold selection, and check dense-image query limits. Its 3,655 training images imply about 2.5 times the images per epoch of the present pills training split, before resolution/crop effects. Actual training time remains unmeasured. The challenge page lists a winning native score of 0.700, so this research does not promise that the original specialist-AP target transfers to this task.

**Second, more substantial extension: LIVECell.** The authors publish native COCO annotations and a well-held-out test protocol: one complete well per cell type was selected for testing before annotation. Filenames preserve cell type, well, location, timestamp, and crop. This gives usable groups for auditing microscopy crops. Images require about 1.3 GB; all experiment annotations require another 2.1 GB. The repository is MIT licensed. However, the principal benchmark is instance segmentation. A boxes-only experiment would need an explicitly named detection protocol and cannot be presented as the original segmentation score. Some images are much denser than 128 or 500 cells, so existing query/output caps must be audited. This makes LIVECell less direct than wheat or the already prepared, separately named stomata group-held-out experiment. [Author repository](https://github.com/sartorius-research/LIVECell), [original paper and held-out-well protocol](https://www.nature.com/articles/s41592-021-01249-6.pdf)

No current Argon/Astra/Kimi/Qwen per-wheat or per-LIVECell comparison was found in the checked primary sources. The old RF100-VL paper's medical-domain failures are evidence about its named 2025 models, not the current frontier. Neither alternative currently has a measured strong frontier gap in this project. A pivot should follow a failed, documented pills development hypothesis plus matched validation discovery on the proposed replacement; it must not consume the sealed pills test to select the replacement.
