# RF-DETR Nano512 pill detector — development checkpoint

This supervised specialist reaches **92.75% COCO AP50:95 on 423 validation images** from one RF100-VL pill-detection dataset. The numeric weights are distributed as a GitHub release asset so the result can be reproduced without retraining.

This is a development checkpoint, selected using validation. It does not establish a substantial RL improvement or a held-out frontier-model win. The 200-image native test remains unevaluated. See the [complete measured report](../../reports/2026-10-16-progressive512-development/README.md) for predictions, uncertainty, negative comparisons and training evidence.

| Property | Value |
| --- | --- |
| Architecture | RF-DETR Nano, DINOv2 backbone |
| Parameters | 30,319,108 |
| Input / checkpoint positional grid | 512 × 512 |
| Foreground mapping | Model channel 0 → COCO category 1, `pill` |
| Export | 466 tensors, float32 safetensors, 121,333,608 bytes |
| Training data | RF100-VL/countingpills-exf0r-gfkt, version 2; 1,449 images |
| Selection data | All 423 native validation images |
| Training | Supervised 384-pixel fit, then 12 epochs / 1,092 updates at 512 pixels |
| Selected checkpoint | EMA weights from progressive epoch 4 |
| License | Apache-2.0; upstream attribution in [NOTICE](NOTICE) |

The original dataset has a parent category with ID 0 but no foreground annotations in that category. This model has one foreground class; do not load it as a two-class detector.

## Download and verify

From the repository root, with GitHub CLI installed:

```bash
gh release download pills-nano512-development-20261016 \
  --repo RyanLin5967/RyanLinEnv \
  --dir runs/published-pills-nano512
cd runs/published-pills-nano512
shasum -a 256 -c SHA256SUMS
cd ../..
```

The model SHA-256 is `66b9b34df0bcc4751022c40e31dbd17c0ee7658447ca679b61b8cf49a51629c9`. The checkpoint contains numeric tensors only. The repository's loader constructs the trusted architecture locally and strictly checks every tensor; model loading requires no base-weight download.

## Run on an image

Install the separate detector environment described in the [main README](../../README.md). Put an image named `example.jpg` in `runs/my-images`, then create an image-only manifest:

```bash
.venv-detection/bin/python - <<'PY'
import json
from pathlib import Path
from PIL import Image

root = Path("runs/my-images")
with Image.open(root / "example.jpg") as image:
    width, height = image.size
document = {"images": [{"id": 1, "file_name": "example.jpg",
                        "width": width, "height": height}]}
(root / "images.json").write_text(json.dumps(document))
PY

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-detection/bin/python -m vision_rl.detection_inference predict \
  --checkpoint runs/published-pills-nano512/nano512-final-model.safetensors \
  --manifest runs/my-images/images.json --image-root runs/my-images \
  --output runs/my-pill-predictions --category-ids 1 \
  --architecture nano --checkpoint-resolution 512 --resolutions 512 \
  --device cpu --cpu-threads 2 --batch-size 1
```

The output includes `predictions.coco.json`, timing and input/checkpoint hashes. Boxes use original-image pixel coordinates in `[x, y, width, height]` format. All 300 proposals are retained for AP evaluation; this is not an operating-point count of pills. Any confidence threshold used in an application needs separate validation.

## Measured scope

| Comparison | Matched images | AP50:95 |
| --- | ---: | ---: |
| This checkpoint | 423 | 92.7456% |
| Supervised 384-pixel checkpoint | 423 | 90.0879% |
| This checkpoint | 48 | 90.6369% |
| Strongest measured Gemini configuration | Same 48 | 73.1910% |

The matched 48-image advantage is **17.45 percentage points**, with a paired capture-group bootstrap 95% interval of **[13.56, 21.60]** points. It misses the project's 20-point target. The full 423-image score cannot be compared directly with the 48-image baseline score.

Capture-hour groups overlap across the native splits, and validation guided checkpoint and prompt selection. Results describe this dataset's development split, not unseen capture sessions, clinical use, or the aggregate RF100-VL benchmark. This model locates the dataset's pill class; it does not identify medication or dosage.

The progressive fit took 2,196.23 seconds on an Apple M5 Pro with 48 GiB unified memory, excluding setup. Recorded combined 384/512 fit time was 8,424.32 seconds. These are training measurements, not inference-speed claims.

## Provenance

The weights derive from Roboflow's COCO-pretrained RF-DETR Nano and were modified by supervised fine-tuning for this project. Roboflow identifies its Nano–Large models as [Apache-2.0 licensed](https://rfdetr.roboflow.com/latest/learn/run/detection/). The original RF-DETR license is supplied unchanged in [LICENSE](LICENSE). Dataset images are not included in the release; the dataset's separate MIT terms and exact version are recorded in the project data audit.

Machine-readable configuration, source hashes and exact scores are in [model.json](model.json). This independent project is not an official Roboflow model release.
