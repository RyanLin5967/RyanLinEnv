"""Train a specialist image classifier on your own labeled photos: SFT, then RL.

Give it a folder with one subfolder per label (or an exported train/valid/test folder):

    my-photos/
      cat/  img1.jpg img2.jpg ...
      dog/  ...

`train_folder` splits the images by label into train, validation and test, extracts frozen
ResNet18 features, trains a supervised head, refines it with sampled-reward RL, selects on
validation only, and scores both models once on the held-out test images. `predict` labels
new images with the saved model.
"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

from .data import image_hash, load_manifest, write_json
from .roboflow import IMAGE_SUFFIXES, _image_path, _row, _write_import, import_classification

PROMPT = "Which category best describes this image?"
MIN_PER_LABEL = 3  # one image each for train, validation and test


def _label_dirs(root: Path) -> list[Path]:
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if any(p.is_symlink() for p in dirs):
        raise ValueError("label folders cannot be symlinks")
    if not 2 <= len(dirs) <= 26:
        raise ValueError(f"need 2 to 26 label folders, found {len(dirs)} in {root}")
    return dirs


def _is_split_export(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "test").is_dir()


def build_dataset(
    images: str | Path,
    output: str | Path,
    *,
    seed: int = 0,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict:
    """Write train/validation/test manifests from a labeled image folder.

    Each label is split separately, so every split sees every label. Pixel-identical copies
    are dropped before splitting, so no test image is also a training image."""
    root = Path(images).resolve()
    if not root.is_dir():
        raise ValueError(f"image folder does not exist: {root}")
    if _is_split_export(root):
        result = import_classification(root, output)
        result["source"] = "existing train/validation/test split"
        return result
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("split fractions must be between 0 and 1")
    labels = [p.name for p in _label_dirs(root)]
    seen, duplicates, per_label = set(), 0, {}
    for index, name in enumerate(labels):
        label_dir = root / name
        paths = sorted(
            p for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        unique = []
        for path in paths:
            digest = image_hash(_image_path(path, label_dir))
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            unique.append(path.resolve())
        if len(unique) < MIN_PER_LABEL:
            raise ValueError(
                f"label {name!r} has {len(unique)} distinct images; need at least {MIN_PER_LABEL}"
            )
        per_label[index] = unique
    rng = random.Random(seed)
    rows = {"train": [], "validation": [], "test": []}
    for index, paths in per_label.items():
        paths = paths[:]
        rng.shuffle(paths)
        n_test = max(1, round(len(paths) * test_fraction))
        n_val = max(1, round(len(paths) * validation_fraction))
        if len(paths) - n_test - n_val < 1:
            n_test, n_val = 1, 1
        parts = {
            "test": paths[:n_test],
            "validation": paths[n_test : n_test + n_val],
            "train": paths[n_test + n_val :],
        }
        for split, items in parts.items():
            rows[split] += [_row(p, split, labels, index, PROMPT, "classification") for p in items]
    result = _write_import(rows, output)
    result.update(
        classes=labels,
        source="split by label",
        seed=seed,
        duplicates_dropped=duplicates,
    )
    return result


def _test_accuracy(manifest: Path, checkpoint: Path, device: str) -> dict:
    from .vision_policy import evaluate_count_policy

    out = checkpoint.with_name(checkpoint.stem + "-test.json")
    result = evaluate_count_policy(manifest, checkpoint, out, device=device)
    return {"accuracy": result["accuracy"], "correct": result["correct"], "n": result["n"]}


def train_folder(
    images: str | Path,
    output: str | Path,
    *,
    device: str = "auto",
    quick: bool = False,
    seed: int = 0,
    encoder_weights: Path | None = None,
) -> dict:
    """Prepare data, train (SFT then RL), select on validation, then score once on test."""
    from .vision_policy import train_count_policy

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output folder must be new or empty: {output}")
    data = build_dataset(images, output / "data", seed=seed)
    manifests = {k: Path(v) for k, v in data["manifests"].items()}
    report = train_count_policy(
        manifests["train"],
        manifests["validation"],
        output / "model",
        device=device,
        seeds=(7,) if quick else (7, 19),
        hidden_sizes=(0,) if quick else (0, 128),
        sft_epochs=40 if quick else 160,
        rl_steps=200 if quick else 800,
        encoder_weights=encoder_weights,
    )
    model = output / "model"
    sft = report["sft_candidate"]
    rl = report["selected_candidate"]
    sft_val = next(e for e in report["experiments"] if e["candidate"] == sft)["sft_validation"]
    rl_val = next(e for e in report["experiments"] if e["candidate"] == rl)["rl_validation"]
    # Deploy whichever stage validation prefers; test is read only after this choice.
    chosen = "rl" if rl_val["accuracy"] >= sft_val["accuracy"] else "sft"
    shutil.copyfile(model / ("policy.pt" if chosen == "rl" else "sft.pt"), model / "best.pt")
    test = manifests["test"]
    scores = {
        "sft": _test_accuracy(test, model / "sft.pt", device),
        "rl": _test_accuracy(test, model / "policy.pt", device),
    }
    train_labels = Counter(r["answer"] for r in load_manifest(manifests["train"]))
    test_rows = load_manifest(test)
    majority = train_labels.most_common(1)[0][0]
    baseline = sum(r["answer"] == majority for r in test_rows) / len(test_rows)
    summary = {
        "labels": data["classes"],
        "images": data["counts"],
        "duplicates_dropped": data.get("duplicates_dropped", 0),
        "validation_accuracy": {"sft": sft_val["accuracy"], "rl": rl_val["accuracy"]},
        "test_accuracy": {
            "majority_label_baseline": baseline,
            "sft": scores["sft"]["accuracy"],
            "rl": scores["rl"]["accuracy"],
        },
        "deployed_model": chosen,
        "model": str(model / "best.pt"),
        "quick": quick,
        "device": report["encoder_device"],
    }
    write_json(output / "summary.json", summary)
    return summary


def predict(model: str | Path, images: list[str | Path], *, device: str = "auto") -> list[dict]:
    """Label each image with the saved model: top label and its probability."""
    from .vision_policy import (
        _encoder,
        _features,
        _head,
        _load_head_checkpoint,
        _stack,
        select_device,
    )

    torch, _ = _stack()
    checkpoint = Path(model)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model" / "best.pt"
    saved = _load_head_checkpoint(checkpoint)
    meta = saved["metadata"]
    if meta.get("task") != "classification":
        raise ValueError("predict needs a model trained by train-folder (a classification task)")
    encoder = _encoder(checkpoint.parent / meta["encoder_file"])
    paths = []
    for item in images:
        path = Path(item)
        found = (
            sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            if path.is_dir()
            else [path]
        )
        paths += [p.resolve() for p in found]
    if not paths:
        raise ValueError("no images to label")
    rows = [{"image": str(p)} for p in paths]
    x = _features(rows, encoder, select_device(device))
    x = (x - saved["feature_mean"]) / saved["feature_scale"]
    head = _head(meta["feature_dim"], len(meta["choices"]), meta["hidden"])
    head.load_state_dict(saved["state_dict"])
    head.eval()
    with torch.inference_mode():
        probs = head(x).softmax(-1)
    best = probs.argmax(-1).tolist()
    return [
        {"image": str(p), "label": meta["choices"][k], "confidence": round(float(probs[i, k]), 4)}
        for i, (p, k) in enumerate(zip(paths, best, strict=True))
    ]


def make_example(output: str | Path, per_label: int = 24, seed: int = 0) -> Path:
    """A tiny synthetic dataset (circles, squares, triangles) to try the pipeline offline."""
    from PIL import Image, ImageDraw

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"example folder must be new or empty: {output}")
    rng = random.Random(seed)
    for shape in ("circle", "square", "triangle"):
        folder = output / shape
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_label):
            image = Image.new("RGB", (128, 128), tuple(rng.randrange(150, 256) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            size = rng.randrange(30, 60)
            cx, cy = rng.randrange(size, 128 - size), rng.randrange(size, 128 - size)
            color = tuple(rng.randrange(0, 120) for _ in range(3))
            box = (cx - size, cy - size, cx + size, cy + size)
            if shape == "circle":
                draw.ellipse(box, fill=color)
            elif shape == "square":
                draw.rectangle(box, fill=color)
            else:
                draw.polygon(
                    [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)], fill=color
                )
            image.save(folder / f"{shape}-{i:03d}.png")
    return output
