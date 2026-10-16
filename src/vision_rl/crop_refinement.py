"""A native-pixel, side-aware second detection stage.

The global detector sees resized images. This stage revisits original-resolution
RGB crops and learns four boundary distributions with a spatial feature pyramid.
The design is informed by Cascade R-CNN (arxiv:1712.00726), Side-Aware Boundary
Localization (arxiv:1912.04260), and Generalized Focal Loss (arxiv:2006.04388).
It is a separately measured experiment, not an implementation of those papers.

Training reads train/valid annotations. Public prediction accepts an image-only
manifest, proposals, and a numeric safetensors checkpoint; it cannot read labels.
Low-confidence proposals outside the fixed crop budget remain exactly unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import stat
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropConfig:
    crop_size: int = 96
    width: int = 24
    padding: float = 0.30
    min_score: float = 0.03
    max_per_image: int = 128
    jitter_per_gt: int = 2
    jitter_std: float = 0.06
    match_iou: float = 0.50
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 419
    cpu_threads: int = 4
    device: str = "cpu"
    evaluate_every: int = 2
    blend_candidates: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    quality_candidates: tuple[float, ...] = (0.0, 0.25)

    def validate(self) -> None:
        limits = {"crop_size": 128, "width": 32, "max_per_image": 500, "epochs": 100,
                  "batch_size": 256, "cpu_threads": 16, "evaluate_every": 100}
        for name, limit in limits.items():
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= limit:
                raise ValueError(f"{name} must be a positive bounded integer")
        if type(self.jitter_per_gt) is not int or not 0 <= self.jitter_per_gt <= 8:
            raise ValueError("jitter_per_gt must be a bounded integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.crop_size not in (64, 96, 128) or self.width not in (16, 24, 32):
            raise ValueError("Trusted crop sizes are 64/96/128 and widths are 16/24/32")
        if not 0.15 <= self.padding <= 0.5 or not 0 <= self.min_score <= 1:
            raise ValueError("Invalid crop padding or score threshold")
        if not 1 <= self.max_per_image <= 500 or not 0 <= self.jitter_per_gt <= 8:
            raise ValueError("Crop budget or jitter count exceeds bounded limits")
        if not 0 < self.jitter_std <= 0.2 or not 0.3 <= self.match_iou <= 0.9:
            raise ValueError("Invalid jitter or matching IoU")
        if not 1 <= self.cpu_threads <= 16 or self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("Invalid compute configuration")
        if not 0 < self.learning_rate <= 0.1 or not 0 <= self.weight_decay <= 0.1:
            raise ValueError("Invalid optimizer configuration")
        if type(self.evaluate_every) is not int or self.evaluate_every < 1:
            raise ValueError("evaluate_every must be positive")
        if not self.blend_candidates or not self.quality_candidates:
            raise ValueError("At least one validation candidate is required")
        for values in (self.blend_candidates, self.quality_candidates):
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Blend candidates must be finite in 0..1")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _xyxy(box: list[float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def pairwise_iou(boxes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    boxes, targets = np.asarray(boxes), np.asarray(targets)
    if len(boxes) == 0 or len(targets) == 0:
        return np.zeros((len(boxes), len(targets)), dtype=np.float32)
    overlap = np.maximum(
        np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
        - np.maximum(boxes[:, None, :2], targets[None, :, :2]),
        0,
    ).prod(-1)
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
    target_areas = np.maximum(targets[:, 2:] - targets[:, :2], 0).prod(-1)
    return overlap / np.maximum(areas[:, None] + target_areas[None] - overlap, 1e-12)


def crop_extent(box: np.ndarray, padding: float) -> np.ndarray:
    size = box[2:] - box[:2]
    return np.r_[box[:2] - padding * size, box[2:] + padding * size].astype(np.float32)


def crop_pixels(image: Image.Image, extent: np.ndarray, size: int) -> np.ndarray:
    """Sample original image pixels; the extent and inverse mapping stay fractional."""
    return np.asarray(
        image.transform(
            (size, size), Image.Transform.EXTENT, tuple(float(v) for v in extent),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    ).copy()


def relative_box(box: np.ndarray, extent: np.ndarray) -> np.ndarray:
    return (box - np.tile(extent[:2], 2)) / np.tile(extent[2:] - extent[:2], 2)


def absolute_box(box: np.ndarray, extent: np.ndarray) -> np.ndarray:
    return box * np.tile(extent[2:] - extent[:2], 2) + np.tile(extent[:2], 2)


def _load_predictions(path: Path, images: list[dict]) -> list[dict]:
    predictions = json.loads(path.read_text())
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO list")
    known = {r["id"] for r in images}
    for p in predictions:
        if not isinstance(p, dict) or p.get("image_id") not in known:
            raise ValueError("Prediction references an unknown image")
        if type(p.get("category_id")) is not int:
            raise ValueError("Prediction category must be an integer")
        if not isinstance(p.get("bbox"), list) or len(p["bbox"]) != 4:
            raise ValueError("Predictions require xywh boxes")
        values = p["bbox"] + [p.get("score")]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Prediction coordinates and scores must be finite")
        if p["bbox"][2] <= 0 or p["bbox"][3] <= 0 or not 0 <= p["score"] <= 1:
            raise ValueError("Invalid proposal extent or score")
    return predictions


def selected_proposals(predictions: list[dict], config: CropConfig) -> dict[int, list[int]]:
    """Budget only the refinement compute; never delete the remaining proposals."""
    selected = defaultdict(list)
    for index, row in enumerate(predictions):
        if row["score"] >= config.min_score:
            selected[row["image_id"]].append(index)
    for image_id, indices in selected.items():
        selected[image_id] = sorted(indices, key=lambda i: -predictions[i]["score"])[
            : config.max_per_image
        ]
    return dict(selected)


def build_bank(
    image_root: Path,
    images: list[dict],
    predictions: list[dict],
    config: CropConfig,
    *,
    annotations: list[dict] | None = None,
    training: bool = False,
    expected_image_hashes: dict[int, str] | None = None,
) -> dict[str, np.ndarray]:
    """Build pixel crops; labels are optional and forbidden at prediction call sites."""
    config.validate()
    if training and annotations is None:
        raise ValueError("Training requires annotations")
    by_image = defaultdict(list)
    for row in annotations or []:
        by_image[row["image_id"]].append(row)
    selected = selected_proposals(predictions, config)
    generator = np.random.default_rng(config.seed)
    values = defaultdict(list)
    root = image_root.resolve()
    for image_info in images:
        iid = image_info["id"]
        indices = selected.get(iid, [])
        gt = by_image.get(iid, [])
        if not indices and not (training and gt):
            continue
        path = (root / image_info["file_name"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image paths must stay within the image root")
        image_bytes = path.read_bytes()
        if expected_image_hashes is not None:
            if hashlib.sha256(image_bytes).hexdigest() != expected_image_hashes[iid]:
                raise ValueError("Image bytes differ from the frozen data fingerprint")
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
        if image.size != (image_info["width"], image_info["height"]):
            raise ValueError("Decoded image dimensions differ from the manifest")
        targets = np.array([_xyxy(a["bbox"]) for a in gt], dtype=np.float32).reshape(-1, 4)
        labels = np.array([a["category_id"] for a in gt], dtype=np.int64)
        matched: set[int] = set()

        def add(box: np.ndarray, index: int, target: np.ndarray, positive: bool,
                unique: bool, score: float) -> None:
            extent = crop_extent(box, config.padding)
            values["pixels"].append(crop_pixels(image, extent, config.crop_size))
            values["proposal"].append(box)
            values["extent"].append(extent)
            values["target"].append(relative_box(target, extent))
            values["positive"].append(positive)
            values["unique"].append(unique)
            values["index"].append(index)
            values["image_id"].append(iid)
            values["score"].append(score)

        for index in indices:
            p = predictions[index]
            box = _xyxy(p["bbox"])
            target, positive, unique = box, False, False
            if len(targets):
                overlaps = pairwise_iou(box[None], targets)[0]
                overlaps[labels != p["category_id"]] = 0
                best = int(overlaps.argmax())
                positive = bool(overlaps[best] >= config.match_iou)
                if positive:
                    target, unique = targets[best], best not in matched
                    matched.add(best)
            add(box, index, target, positive, unique, p["score"])
        if training:
            for target in targets:
                size = target[2:] - target[:2]
                for _ in range(config.jitter_per_gt):
                    noise = generator.normal(0, config.jitter_std, 4).clip(-0.2, 0.2)
                    proposal = target + np.tile(size, 2) * noise
                    add(proposal, -1, target, True, True, 0.9)
    count = len(values["index"])
    return {
        "pixels": np.asarray(values["pixels"], dtype=np.uint8).reshape(
            count, config.crop_size, config.crop_size, 3
        ),
        **{key: np.asarray(values[key], dtype=np.float32).reshape(count, 4)
           for key in ("proposal", "extent", "target")},
        **{key: np.asarray(values[key], dtype=np.bool_)
           for key in ("positive", "unique")},
        **{key: np.asarray(values[key], dtype=np.int64) for key in ("index", "image_id")},
        "score": np.asarray(values["score"], dtype=np.float32),
    }


def make_model(config: CropConfig):
    """Compact feature pyramid, spatial side heads, and separate quality head."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    config.validate()
    width = config.width

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(4, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(4, cout), nn.SiLU(),
        )

    class SpatialBoundaryNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = block(5, width, 2)
            self.middle = block(width, width * 2, 2)
            self.deep = block(width * 2, width * 4, 2)
            self.lateral2 = nn.Conv2d(width * 2, width, 1)
            self.lateral3 = nn.Conv2d(width * 4, width, 1)
            self.fusion = block(width, width)
            self.xhead = nn.Sequential(
                nn.Conv1d(width * 2, width, 5, padding=2), nn.SiLU(),
                nn.Conv1d(width, 2, 5, padding=2),
            )
            self.yhead = copy.deepcopy(self.xhead)
            self.quality = nn.Sequential(nn.Linear(width * 4 + 8, 64), nn.SiLU(), nn.Linear(64, 1))
            # Identity prior keeps the initial cascade close to the trained detector.
            for head in (self.xhead, self.yhead):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
            nn.init.zeros_(self.quality[-1].weight)
            nn.init.constant_(self.quality[-1].bias, 3.0)
            axis = torch.linspace(0, 1, config.crop_size)
            first = config.padding / (1 + 2 * config.padding)
            centers = torch.tensor([first, first, 1 - first, 1 - first])
            self.register_buffer("axis", axis)
            self.register_buffer("prior", -0.5 * ((axis[None] - centers[:, None]) / 0.055) ** 2)

        def forward(self, pixels):
            batch, _, height, size = pixels.shape
            axis = torch.linspace(-1, 1, size, device=pixels.device, dtype=pixels.dtype)
            x = axis.view(1, 1, 1, size).expand(batch, 1, height, size)
            y = axis.view(1, 1, height, 1).expand(batch, 1, height, size)
            a = self.stem(torch.cat((pixels, x, y), 1))
            b = self.middle(a)
            c = self.deep(b)
            fused = self.fusion(
                a + F.interpolate(self.lateral2(b), a.shape[-2:], mode="bilinear",
                                  align_corners=False)
                + F.interpolate(self.lateral3(c), a.shape[-2:], mode="bilinear",
                                align_corners=False)
            )
            xfeatures = torch.cat((fused.mean(2), fused.amax(2)), 1)
            yfeatures = torch.cat((fused.mean(3), fused.amax(3)), 1)
            xlogits = F.interpolate(self.xhead(xfeatures), config.crop_size,
                                   mode="linear", align_corners=True)
            ylogits = F.interpolate(self.yhead(yfeatures), config.crop_size,
                                   mode="linear", align_corners=True)
            logits = torch.stack((xlogits[:, 0], ylogits[:, 0],
                                  xlogits[:, 1], ylogits[:, 1]), 1) + self.prior
            probabilities = logits.softmax(-1)
            edges = (probabilities * self.axis).sum(-1)
            stats = torch.cat((probabilities.amax(-1),
                               -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)), 1)
            quality = self.quality(torch.cat((c.mean((2, 3)), stats), 1)).squeeze(-1)
            return {"logits": logits, "edges": edges, "quality": quality}

    return SpatialBoundaryNetwork()


def distribution_loss(logits, targets):
    """Linear interpolation of adjacent bins represents fractional edge targets."""
    import torch
    from torch.nn import functional as F

    bins = logits.shape[-1]
    positions = targets.clamp(0, 1) * (bins - 1)
    low = positions.floor().long().clamp_max(bins - 2)
    high = low + 1
    fraction = positions - low
    log_probabilities = F.log_softmax(logits, -1)
    return -((1 - fraction) * torch.gather(log_probabilities, -1, low[..., None]).squeeze(-1)
             + fraction * torch.gather(log_probabilities, -1, high[..., None]).squeeze(-1)).mean()


def _tensor_iou(boxes, targets):
    import torch

    overlap = (torch.minimum(boxes[..., 2:], targets[..., 2:])
               - torch.maximum(boxes[..., :2], targets[..., :2])).clamp_min(0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
    target_area = (targets[..., 2:] - targets[..., :2]).clamp_min(0).prod(-1)
    return overlap / (area + target_area - overlap).clamp_min(1e-8)


def supervised_loss(outputs, targets, positive, unique):
    import torch
    from torch.nn import functional as F

    zero = outputs["edges"].sum() * 0
    if positive.any():
        edge = outputs["edges"][positive]
        target = targets[positive]
        loc = (distribution_loss(outputs["logits"][positive], target)
               + 5 * F.smooth_l1_loss(edge, target, beta=0.02)
               + 2 * (1 - _tensor_iou(edge, target)).mean())
    else:
        loc = zero
    # One-to-one quality targets suppress duplicate proposals rather than teaching
    # every overlapping crop to become an equally confident detection.
    quality_targets = unique.float() * _tensor_iou(outputs["edges"].detach(), targets)
    weights = torch.where(unique, 1.0, 4.0)
    quality = (F.binary_cross_entropy_with_logits(
        outputs["quality"], quality_targets, reduction="none"
    ) * weights).mean()
    return loc + 0.25 * quality, {"localization": float(loc.detach()),
                                 "quality": float(quality.detach())}


def _batch(bank, indices, device, *, generator=None):
    import torch

    pixels = torch.from_numpy(bank["pixels"][indices].copy()).permute(0, 3, 1, 2).float() / 255
    targets = torch.from_numpy(bank["target"][indices].copy())
    if generator is not None:
        flipx = torch.rand(len(indices), generator=generator) < 0.5
        flipy = torch.rand(len(indices), generator=generator) < 0.5
        pixels[flipx] = pixels[flipx].flip(-1)
        targets[flipx, 0], targets[flipx, 2] = (
            1 - targets[flipx, 2].clone(), 1 - targets[flipx, 0].clone()
        )
        pixels[flipy] = pixels[flipy].flip(-2)
        targets[flipy, 1], targets[flipy, 3] = (
            1 - targets[flipy, 3].clone(), 1 - targets[flipy, 1].clone()
        )
        brightness = 0.85 + 0.3 * torch.rand(len(indices), 1, 1, 1, generator=generator)
        pixels = (pixels * brightness).clamp(0, 1)
    return (
        ((pixels - 0.5) / 0.25).to(device), targets.to(device),
        torch.from_numpy(bank["positive"][indices]).to(device),
        torch.from_numpy(bank["unique"][indices]).to(device),
    )


def predict_bank(model, bank, config: CropConfig) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    edges, quality = [], []
    with torch.inference_mode():
        for start in range(0, len(bank["pixels"]), config.batch_size):
            indices = np.arange(start, min(start + config.batch_size, len(bank["pixels"])))
            pixels, _, _, _ = _batch(bank, indices, config.device)
            outputs = model(pixels)
            edges.append(outputs["edges"].cpu().numpy())
            quality.append(outputs["quality"].sigmoid().cpu().numpy())
    return (
        np.concatenate(edges) if edges else np.empty((0, 4), dtype=np.float32),
        np.concatenate(quality) if quality else np.empty(0, dtype=np.float32),
    )


def apply_refinement(predictions, bank, edges, quality, *, blend=1.0, quality_blend=0.0):
    """Pure, label-free coordinate transform; no NMS or proposal truncation."""
    if not 0 <= blend <= 1 or not 0 <= quality_blend <= 1:
        raise ValueError("Blends must be in 0..1")
    if edges.shape != bank["proposal"].shape or quality.shape != bank["score"].shape:
        raise ValueError("Refiner output shape differs from crop bank")
    if not np.isfinite(edges).all() or not np.isfinite(quality).all():
        raise ValueError("Refiner outputs must be finite")
    result = copy.deepcopy(predictions)
    for i, index in enumerate(bank["index"]):
        if index < 0 or index >= len(result):
            raise ValueError("Prediction banks cannot contain training-only jitter crops")
        refined = absolute_box(edges[i], bank["extent"][i])
        refined = bank["proposal"][i] + blend * (refined - bank["proposal"][i])
        if (refined[2:] <= refined[:2]).any():
            raise ValueError("Refiner generated a reversed box")
        if blend:
            result[index]["bbox"] = [float(refined[0]), float(refined[1]),
                                      float(refined[2] - refined[0]),
                                      float(refined[3] - refined[1])]
        if quality_blend:
            result[index]["score"] *= float(np.clip(quality[i], 0, 1)) ** quality_blend
    return result


def save_model(model, path: Path, config: CropConfig, metadata: dict) -> dict:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"format": "native-pixel-cascade-v1", "config": asdict(config), **metadata}
    temporary = path.with_suffix(".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              temporary, metadata={"vision_rl": json.dumps(info, sort_keys=True)})
    os.replace(temporary, path)
    _write(path.with_suffix(".json"), {**info, "sha256": _sha(path)})
    return info


def _safe_checkpoint_bytes(path: Path) -> bytes:
    """Read one bounded regular file without following any path-component symlink."""
    path = path.absolute()
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 8 <= info.st_size <= 32 * 1024 * 1024:
                raise ValueError("Crop checkpoint must be a regular numeric file below 32 MiB")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(32 * 1024 * 1024 + 1)
        finally:
            os.close(fd)
        if len(data) != info.st_size:
            raise ValueError("Crop checkpoint changed while reading")
        return data
    finally:
        os.close(directory)


def load_model(path: Path, *, device="cpu", expected_detector_sha256=None,
               expected_sha256=None):
    """Load the historical numeric artifact without rewriting its training metadata."""
    from safetensors.torch import load

    data = _safe_checkpoint_bytes(path)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Crop checkpoint SHA differs from the frozen composition")
    header_size = int.from_bytes(data[:8], "little")
    if not 2 <= header_size <= min(1024 * 1024, len(data) - 8):
        raise ValueError("Invalid crop safetensors header size")
    try:
        header = json.loads(data[8:8 + header_size])
        metadata = json.loads(header["__metadata__"]["vision_rl"])
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Invalid crop checkpoint metadata") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        raise ValueError("Invalid crop checkpoint configuration")
    if metadata.get("format") != "native-pixel-cascade-v1":
        raise ValueError("Unknown crop checkpoint format")
    if expected_detector_sha256 and metadata.get("detector_sha256") != expected_detector_sha256:
        raise ValueError("Crop model was trained for a different detector")
    try:
        config = CropConfig(**{**metadata["config"], "device": device})
    except TypeError as error:
        raise ValueError("Unknown crop checkpoint configuration") from error
    config.validate()
    model = make_model(config)
    state = load(data)
    expected = model.state_dict()
    if state.keys() != expected.keys() or any(
        value.shape != expected[key].shape or value.dtype != expected[key].dtype
        for key, value in state.items()
    ):
        raise ValueError("Crop tensors differ from the fixed architecture")
    if any(not value.isfinite().all() for value in state.values()):
        raise ValueError("Crop checkpoint contains nonfinite parameters")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config, {**metadata, "checkpoint_sha256": digest}


def refine_manifest(checkpoint: Path, manifest: Path, image_root: Path, proposals: Path,
                    output: Path, *, device="cpu", blend=None, quality_blend=None,
                    cpu_threads=None, composition: dict | None = None) -> dict:
    """Image-only prediction with separate training and deployment provenance.

    A frozen composition may transfer this unchanged crop model to another global
    detector. It pins the submitted artifact and actual inference inputs; the
    checkpoint's historical detector/source hashes are never relabeled.
    """
    import torch

    from vision_rl.detection_inference import load_image_manifest

    started = time.monotonic()
    composition = composition or {}
    allowed = {"checkpoint_sha256", "proposals_sha256", "manifest_sha256", "blend",
               "quality_blend", "deployment_detector_sha256", "image_tree_sha256"}
    if set(composition) - allowed:
        raise ValueError("Unknown frozen composition keys")
    for key, value in composition.items():
        if key.endswith("sha256") and (not isinstance(value, str) or len(value) != 64
                                       or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError(f"Invalid frozen {key}")
    for key, override in (("blend", blend), ("quality_blend", quality_blend)):
        if override is not None and key in composition and override != composition[key]:
            raise ValueError(f"Conflicting frozen {key} override")
    hashes = {"proposals_sha256": _sha(proposals), "manifest_sha256": _sha(manifest)}
    for key, actual in hashes.items():
        if key in composition and composition[key] != actual:
            raise ValueError(f"Frozen {key} differs")
    model, config, metadata = load_model(
        checkpoint, device=device, expected_sha256=composition.get("checkpoint_sha256")
    )
    if cpu_threads is not None:
        config = replace(config, cpu_threads=cpu_threads)
        config.validate()
    torch.set_num_threads(config.cpu_threads)
    load_seconds = time.monotonic() - started
    images = load_image_manifest(manifest, image_root)
    image_hashes = {row["id"]: _sha(row["path"]) for row in images}
    image_tree_sha256 = hashlib.sha256(json.dumps([
        [row["id"], row["file_name"], image_hashes[row["id"]]] for row in images
    ], separators=(",", ":")).encode()).hexdigest()
    if composition.get("image_tree_sha256", image_tree_sha256) != image_tree_sha256:
        raise ValueError("Frozen image tree differs")
    predictions = _load_predictions(proposals, images)
    crop_started = time.monotonic()
    bank = build_bank(image_root, images, predictions, config,
                      expected_image_hashes=image_hashes)
    crop_seconds = time.monotonic() - crop_started
    prediction_started = time.monotonic()
    edges, quality = predict_bank(model, bank, config)
    prediction_seconds = time.monotonic() - prediction_started
    blend = composition.get("blend", metadata.get("blend", 1.0)) if blend is None else blend
    quality_blend = (composition.get("quality_blend", metadata.get("quality_blend", 0.0))
                     if quality_blend is None else quality_blend)
    refined = apply_refinement(predictions, bank, edges, quality,
                               blend=blend, quality_blend=quality_blend)
    if hashes != {"proposals_sha256": _sha(proposals), "manifest_sha256": _sha(manifest)}:
        raise ValueError("Inference inputs changed during prediction")
    if any(_sha(row["path"]) != image_hashes[row["id"]] for row in images):
        raise ValueError("Image bytes changed during prediction")
    _write(output, refined)
    report = {"format": "native-pixel-cascade-predictions-v1", "images": len(images),
              "proposals": len(predictions), "refined_proposals": len(edges),
              "blend": blend, "quality_blend": quality_blend,
              "checkpoint_sha256": metadata["checkpoint_sha256"], **hashes,
              "image_tree_sha256": image_tree_sha256,
              "training_detector_sha256": metadata.get("detector_sha256"),
              "training_proposals_sha256": metadata.get("train_proposals_sha256"),
              "training_source_sha256": metadata.get("source_code_sha256"),
              "deployment_detector_sha256": composition.get("deployment_detector_sha256"),
              "inference_source_sha256": _sha(Path(__file__)),
              "device": config.device, "cpu_threads": config.cpu_threads,
              "batch_size": config.batch_size, "load_seconds": load_seconds,
              "crop_seconds": crop_seconds, "prediction_seconds": prediction_seconds,
              "wall_seconds": time.monotonic() - started,
              "labels_available_to_predictor": False}
    _write(output.with_suffix(".report.json"), report)
    return report


def train(source: Path, train_proposals: Path, validation_proposals: Path,
          output: Path, detector_sha256: str, config: CropConfig, *, resume=False) -> dict:
    import torch

    from vision_rl.detection_eval import evaluate_coco, load_coco

    config.validate()
    if len(detector_sha256) != 64 or any(c not in "0123456789abcdef" for c in detector_sha256):
        raise ValueError("An exact detector SHA256 is required")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Use a fresh experiment directory; prior results are immutable")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    rng = torch.Generator().manual_seed(config.seed + 1)
    started = time.monotonic()
    train_path = source / "train" / "_annotations.coco.json"
    validation_path = source / "valid" / "_annotations.coco.json"
    documents = {"train": load_coco(train_path), "valid": load_coco(validation_path)}
    image_names = [{row["file_name"] for row in documents[s]["images"]} for s in ("train", "valid")]
    if image_names[0] & image_names[1]:
        raise ValueError("Training and validation filenames overlap")
    base_predictions = {
        "train": _load_predictions(train_proposals, documents["train"]["images"]),
        "valid": _load_predictions(validation_proposals, documents["valid"]["images"]),
    }
    provenance = {"detector_sha256": detector_sha256,
                  "train_proposals_sha256": _sha(train_proposals),
                  "validation_proposals_sha256": _sha(validation_proposals),
                  "train_annotations_sha256": _sha(train_path),
                  "validation_annotations_sha256": _sha(validation_path),
                  "source_code_sha256": _sha(Path(__file__)),
                  "test_split_accessed": False}
    for split in ("train", "valid"):
        fingerprints = [(row["file_name"], _sha(source / split / row["file_name"]))
                        for row in documents[split]["images"]]
        provenance[f"{split}_image_tree_sha256"] = hashlib.sha256(
            json.dumps(fingerprints, separators=(",", ":")).encode()
        ).hexdigest()
    plan = {"config": asdict(config), **provenance,
            "sources": ["https://arxiv.org/abs/1712.00726",
                        "https://arxiv.org/abs/1912.04260",
                        "https://arxiv.org/abs/2006.04388"]}
    if resume:
        if json.loads((output / "plan.json").read_text()) != json.loads(json.dumps(plan)):
            raise ValueError("Resume inputs, configuration, or implementation changed")
    else:
        _write(output / "plan.json", plan)
    banks = {
        s: build_bank(source / s, documents[s]["images"], base_predictions[s], config,
                      annotations=documents[s]["annotations"] if s == "train" else None,
                      training=s == "train")
        for s in ("train", "valid")
    }
    _write(output / "data.json", {"train_crops": len(banks["train"]["pixels"]),
                                 "validation_crops": len(banks["valid"]["pixels"]),
                                 "positive_train_crops": int(banks["train"]["positive"].sum()),
                                 "unique_train_crops": int(banks["train"]["unique"].sum()),
                                 "cache_seconds": time.monotonic() - started})
    print(json.dumps({"event": "crop_cache_ready", "seconds": time.monotonic() - started,
                      "train": len(banks["train"]["pixels"]),
                      "validation": len(banks["valid"]["pixels"])}), flush=True)
    model = make_model(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs,
                                                          eta_min=config.learning_rate / 20)
    baseline = evaluate_coco(validation_path, base_predictions["valid"])
    _write(output / "baseline-evaluation.json", baseline)
    best = {"AP": baseline["AP"], "epoch": -1, "blend": 0.0, "quality_blend": 0.0}
    best_refiner = {"AP": -1.0}
    history = []
    first_epoch, previous_elapsed = 0, 0.0
    if resume:
        # This local trainer state is never accepted by the public verifier.
        state = torch.load(output / "trainer-state.pt", map_location="cpu", weights_only=True)
        if state.get("version") != 1:
            raise ValueError("Unknown trainer state version")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        rng.set_state(state["shuffle_rng"])
        first_epoch, history = state["epoch"], state["history"]
        best, best_refiner = state["best"], state["best_refiner"]
        previous_elapsed = state["elapsed_seconds"]
    for epoch in range(first_epoch, config.epochs):
        epoch_started = time.monotonic()
        model.train()
        order = torch.randperm(len(banks["train"]["pixels"]), generator=rng).numpy()
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch = _batch(banks["train"], order[start:start + config.batch_size],
                           config.device, generator=rng)
            pixels, targets, positive, unique = batch
            outputs = model(pixels)
            loss, _ = supervised_loss(outputs, targets, positive, unique)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                  "train_seconds": time.monotonic() - epoch_started,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                  "learning_rate": scheduler.get_last_lr()[0]}
        if (epoch + 1) % config.evaluate_every == 0 or epoch + 1 == config.epochs:
            edges, quality = predict_bank(model, banks["valid"], config)
            records = []
            for blend in config.blend_candidates:
                for quality_blend in config.quality_candidates:
                    predictions = apply_refinement(base_predictions["valid"], banks["valid"],
                                                   edges, quality, blend=blend,
                                                   quality_blend=quality_blend)
                    result = evaluate_coco(validation_path, predictions)
                    candidate = {"AP": result["AP"], "AP50": result["AP50"],
                                 "AP75": result["AP75"], "epoch": epoch + 1,
                                 "blend": blend, "quality_blend": quality_blend}
                    records.append(candidate)
                    if candidate["AP"] > best_refiner["AP"]:
                        best_refiner = candidate
                        save_model(model, output / "best-refiner.safetensors", config,
                                   {**provenance, **candidate})
                        _write(output / "best-refiner-evaluation.json", result)
                        _write(output / "best-refiner-predictions.coco.json", predictions)
                    if candidate["AP"] > best["AP"]:
                        best = candidate
            record["validation_candidates"] = records
            record["best"] = best
        save_model(model, output / "last.safetensors", config,
                   {**provenance, "epoch": epoch + 1})
        history.append(record)
        _write(output / "history.json", history)
        temporary = output / "trainer-state.pt.tmp"
        torch.save({"version": 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "shuffle_rng": rng.get_state(),
                    "epoch": epoch + 1, "history": history, "best": best,
                    "best_refiner": best_refiner, "elapsed_seconds": record["elapsed_seconds"]},
                   temporary)
        os.replace(temporary, output / "trainer-state.pt")
        print(json.dumps(record, sort_keys=True), flush=True)
    summary = {"format": "native-pixel-cascade-experiment-v1", "baseline_AP": baseline["AP"],
               "best": best, "best_trained_refiner": best_refiner,
               "AP_gain": best["AP"] - baseline["AP"], "epochs": len(history),
               "wall_seconds": previous_elapsed + time.monotonic() - started,
               "parameters": sum(p.numel() for p in model.parameters()), **provenance}
    _write(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--source", type=Path, required=True)
    training.add_argument("--train-proposals", type=Path, required=True)
    training.add_argument("--validation-proposals", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--detector-sha256", required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--resume", action="store_true")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--image-root", type=Path, required=True)
    prediction.add_argument("--proposals", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--blend", type=float)
    prediction.add_argument("--quality-blend", type=float)
    prediction.add_argument("--cpu-threads", type=int)
    prediction.add_argument("--composition", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "train":
        path = args.pop("config")
        config = CropConfig(**json.loads(path.read_text())) if path else CropConfig()
        result = train(config=config, **args)
    else:
        path = args.pop("composition")
        args["composition"] = json.loads(path.read_text()) if path else None
        result = refine_manifest(**args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
