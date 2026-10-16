"""Procedural onboarding task, not Humanity's Sixth Sense or a frontier benchmark."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CHOICES = ["left", "right", "above", "below"]
PROMPT = "Where is the red square relative to the blue circle? Choose the dominant direction."


def generate_split(output: Path, split: str, count: int, seed: int) -> Path:
    if count < 4:
        raise ValueError("each split needs at least four examples")
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images" / split
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        label = i % 4
        while True:
            red, blue = rng.integers(9, 55, size=(2, 2))
            dx, dy = red - blue
            actual = (0 if dx < 0 else 1) if abs(dx) > abs(dy) else (2 if dy < 0 else 3)
            if actual == label and max(abs(dx), abs(dy)) > 17 and abs(abs(dx) - abs(dy)) > 3:
                break
        # Small background and size variations make images unique without encoding the label.
        bg = tuple(int(x) for x in rng.integers(225, 256, size=3))
        im = Image.new("RGB", (64, 64), bg)
        draw = ImageDraw.Draw(im)
        r = int(rng.integers(4, 8))
        x, y = map(int, red)
        draw.rectangle((x - r, y - r, x + r, y + r), fill=(220, 35, 40))
        x, y = map(int, blue)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(35, 65, 225))
        filename = f"{seed}-{i:05d}.png"
        im.save(images / filename)
        rows.append(
            {
                "id": f"spatial-{split}-{seed}-{i}",
                "split": split,
                "image": f"images/{split}/{filename}",
                "prompt": PROMPT,
                "choices": CHOICES,
                "answer": chr(65 + label),
                "category": "spatial-direction",
            }
        )
    manifest = output / f"{split}.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def generate(
    output: Path, seed: int = 42, train: int = 1024, validation: int = 256, test: int = 512
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    paths = {}
    for offset, (split, count) in enumerate(
        (("train", train), ("validation", validation), ("test", test))
    ):
        paths[split] = str(generate_split(output, split, count, seed + offset * 100003))
    return paths


def features(rows: list[dict]) -> np.ndarray:
    """Fixed color-centroid encoder; no answers or scene metadata enter the policy."""
    result = []
    for row in rows:
        if row["choices"] != CHOICES or row["prompt"] != PROMPT:
            raise ValueError(
                "CPU policy only supports the synthetic spatial task; use a VLM otherwise"
            )
        with Image.open(row["image"]) as im:
            pixels = np.asarray(im.convert("RGB"), dtype=np.float64)
        red = (pixels[..., 0] > 160) & (pixels[..., 1] < 100) & (pixels[..., 2] < 100)
        blue = (pixels[..., 2] > 160) & (pixels[..., 0] < 100) & (pixels[..., 1] < 120)
        if not red.any() or not blue.any():
            raise ValueError("CPU spatial policy requires a red square and blue circle")
        ry, rx = np.argwhere(red).mean(axis=0)
        by, bx = np.argwhere(blue).mean(axis=0)
        height, width = pixels.shape[:2]
        result.append([(rx - bx) / width, (ry - by) / height, 1.0])
    return np.asarray(result, dtype=np.float64)
