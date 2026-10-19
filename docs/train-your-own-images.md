# Train a specialist on your own images

`vision-rl train-folder` turns a folder of labeled photos into a tested image classifier. It
uses the same SFT-then-RL recipe as the chess pawn experiment: a frozen, pretrained ResNet18
encoder, plus a small learned head.

## 1. Install

```bash
uv venv .venv-vision --python 3.12
uv pip install --python .venv-vision/bin/python -e . -r requirements-vision.txt
```

The first run downloads the public ResNet18 ImageNet weights (about 45 MB) through torchvision.

## 2. Arrange your images

```text
my-photos/
  ripe/      img001.jpg img002.jpg ...
  unripe/    ...
  rotten/    ...
```

- You need 2 to 26 labels, and at least 3 distinct images per label. Tens per label work much
  better.
- Accepted formats are jpg, png, webp, bmp and tif.
- Exact copies (identical pixels) are dropped, and the same image filed under two labels is
  refused. Near-copies such as a resized or re-saved photo, or burst shots, are not detected.
  Keep those together by hand if you want an honest test score.
- Hidden files and macOS zip leftovers (`__MACOSX`, `._name.jpg`) are skipped.
- If your folder already has `train/`, `test/` and `valid/` (or `validation/` or `val/`),
  for example a Roboflow classification export, those splits are used exactly as given.
- Every training image is held in memory once, at about 2.4 MB per image (4 views). A few
  thousand photos therefore needs several GB of memory.

## 3. Train and test

```bash
.venv-vision/bin/vision-rl train-folder --images my-photos --output runs/my-model
```

Optional flags:
- `--quick` runs a smaller search, for a fast first try.
- `--device mps` (Apple) or `--device cuda` (NVIDIA) uses a GPU. The default is `auto`.
- `--seed` changes the train/validation/test shuffle.

What happens:
1. Each label's images are split about 70% train, 15% validation and 15% test, and the split
   is written to `runs/my-model/data/`.
2. **SFT.** A head is trained on frozen image features with cross-entropy.
3. **RL.** The head samples answers and is rewarded only for the correct label. A KL penalty
   keeps it near the SFT model.
4. Whichever stage scores higher on validation is saved as `model/best.pt`; a tie keeps RL.
   Test images are not read until this choice is made.
5. The SFT model, the RL model and a guess-the-most-common-label baseline are each scored
   once on the test images.

`runs/my-model/summary.json` reports the results:

```json
{
  "labels": ["circle", "square", "triangle"],
  "images": {"train": 48, "validation": 12, "test": 12},
  "test_accuracy": {"majority_label_baseline": 0.333, "sft": 1.0, "rl": 1.0},
  "deployed_model": "rl"
}
```

These numbers come from the built-in example, which is deliberately easy. A small test set
gives a rough estimate, so collect more images before trusting small differences.

## 4. Use the model

```bash
.venv-vision/bin/vision-rl predict --model runs/my-model photo1.jpg more-photos/
```

Each image gets its predicted label and the model's confidence.

## Try it without your own photos

```bash
.venv-vision/bin/vision-rl make-example --output runs/shapes
.venv-vision/bin/vision-rl train-folder --images runs/shapes --output runs/shapes-model --quick
```

This writes 72 generated images of circles, squares and triangles, then trains on them. It
takes about 10 seconds on a laptop CPU.
