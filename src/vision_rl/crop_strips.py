"""Native-resolution side-strip cascade with a high-IoU training curriculum.

Each proposal has four RGB strips, sampled at 192 positions normal to the side
and 64 tangentially. A feature pyramid preserves the normal axis; learned
tangential attention identifies the correct boundary. Fresh training-only jitter
emphasizes the quality of actual detector proposals. This is a distinct model
from crop_refinement.py; historical experiment source and checkpoints stay valid.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import stat
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from vision_rl.crop_refinement import (
    _load_predictions,
    _sha,
    _tensor_iou,
    _write,
    _xyxy,
    distribution_loss,
    pairwise_iou,
)


@dataclass(frozen=True)
class StripConfig:
    normal_samples: int = 192
    tangent_samples: int = 64
    normal_span: float = 0.20
    tangent_padding: float = 0.10
    width: int = 24
    min_score: float = 0.03
    max_per_image: int = 128
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 661
    cpu_threads: int = 4
    image_workers: int = 2
    device: str = "cpu"
    mps_memory_gib: float = 2.0
    evaluate_every: int = 2
    blend_candidates: tuple[float, ...] = (0.5, 0.75, 1.0)
    quality_candidates: tuple[float, ...] = (0.0, 0.25)

    def validate(self):
        if self.normal_samples not in (128, 192, 256) or self.tangent_samples not in (32, 64):
            raise ValueError("Trusted strip dimensions are 128/192/256 by 32/64")
        if self.width not in (16, 24, 32):
            raise ValueError("Trusted strip width is 16/24/32")
        if not 0.1 <= self.normal_span <= 0.25 or not 0 <= self.tangent_padding <= 0.25:
            raise ValueError("Invalid strip geometry")
        if not 0 <= self.min_score <= 1 or not 1 <= self.max_per_image <= 500:
            raise ValueError("Invalid refinement budget")
        for name, limit in (("epochs", 100), ("batch_size", 128), ("cpu_threads", 16),
                            ("image_workers", 4), ("evaluate_every", 100)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be a positive bounded integer")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("Device must be cpu, mps, or cuda")
        if not math.isfinite(self.mps_memory_gib) or not 0.5 <= self.mps_memory_gib <= 8:
            raise ValueError("MPS memory cap must be finite between 0.5 and 8 GiB")
        if not 0 < self.learning_rate <= 0.01 or not 0 <= self.weight_decay <= 0.1:
            raise ValueError("Invalid optimizer configuration")
        for values in (self.blend_candidates, self.quality_candidates):
            if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Blend candidates must be finite in 0..1")


def side_strips(image: Image.Image, proposal: np.ndarray, config: StripConfig) -> np.ndarray:
    """Return [left, top, right, bottom], all with normals pointing inward.

    Output shape is [4, tangent_samples, normal_samples, 3]. Affine maps retain
    fractional proposal boundaries; no integer crop rounding is introduced.
    """
    x1, y1, x2, y2 = (float(v) for v in proposal)
    width, height = x2 - x1, y2 - y1
    n, t = config.normal_samples, config.tangent_samples
    span, padding = config.normal_span, config.tangent_padding
    normal_x, normal_y = 2 * span * width / n, 2 * span * height / n
    tangent_x, tangent_y = (1 + 2 * padding) * width / t, (1 + 2 * padding) * height / t
    maps = [
        (normal_x, 0, x1 - span * width, 0, tangent_y, y1 - padding * height),
        (0, tangent_x, x1 - padding * width, normal_y, 0, y1 - span * height),
        (-normal_x, 0, x2 + span * width, 0, tangent_y, y1 - padding * height),
        (0, tangent_x, x1 - padding * width, -normal_y, 0, y2 + span * height),
    ]
    return np.stack([
        np.asarray(image.transform((n, t), Image.Transform.AFFINE, mapping,
                                   resample=Image.Resampling.BILINEAR), dtype=np.uint8)
        for mapping in maps
    ])


def edge_displacements(proposals: np.ndarray, targets: np.ndarray) -> np.ndarray:
    size = np.tile(proposals[..., 2:] - proposals[..., :2], (1, 2))
    return (targets - proposals) / size * np.array([1, 1, -1, -1], np.float32)


def decode_edges(proposals: np.ndarray, displacements: np.ndarray) -> np.ndarray:
    size = np.tile(proposals[..., 2:] - proposals[..., :2], (1, 2))
    return proposals + displacements * size * np.array([1, 1, -1, -1], np.float32)


def _selected(predictions: list[dict], config: StripConfig) -> dict[int, list[int]]:
    grouped = defaultdict(list)
    for index, row in enumerate(predictions):
        if row["score"] >= config.min_score:
            grouped[row["image_id"]].append(index)
    return {iid: sorted(indices, key=lambda i: -predictions[i]["score"])[:config.max_per_image]
            for iid, indices in grouped.items()}


def jitter_scales(epoch: int, epochs: int) -> tuple[list[float], list[float]]:
    # Keep some coarse recovery examples while moving the majority of updates
    # into the .90-.99 IoU regime actually seen at deployment.
    coarse = 0.20 if epoch < epochs // 2 else 0.05
    return [0.05, 0.012, 0.004], [coarse, 0.50, 0.50 - coarse]


def frame_bank(image_root: Path, image_info: dict, predictions: list[dict], indices: list[int],
               config: StripConfig, *, annotations=None, epoch=0,
               expected_file_sha256=None) -> dict[str, np.ndarray]:
    """Load each image once; optional training labels never enter public inference."""
    root = image_root.resolve()
    path = (root / image_info["file_name"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image must stay inside its root")
    image_bytes = path.read_bytes()
    if expected_file_sha256 is not None:
        if hashlib.sha256(image_bytes).hexdigest() != expected_file_sha256:
            raise ValueError("Image bytes differ from the frozen data fingerprint")
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    if image.size != (image_info["width"], image_info["height"]):
        raise ValueError("Image dimensions differ from the manifest")
    generator = np.random.default_rng(np.random.SeedSequence(
        [config.seed, epoch, image_info["id"]]
    ))
    gt = annotations or []
    targets = np.array([_xyxy(row["bbox"]) for row in gt], np.float32).reshape(-1, 4)
    labels = np.array([row["category_id"] for row in gt], np.int64)
    proposal_boxes = np.array([_xyxy(predictions[i]["bbox"]) for i in indices],
                              np.float32).reshape(-1, 4)
    overlap = pairwise_iou(proposal_boxes, proposal_boxes)
    matched = set()
    result = defaultdict(list)

    def add(box, target, index, positive, unique, context):
        delta = edge_displacements(box[None], target[None])[0]
        within_strip = bool((np.abs(delta) < config.normal_span * 0.98).all())
        result["pixels"].append(side_strips(image, box, config))
        result["proposal"].append(box)
        result["target"].append(target)
        result["displacements"].append(delta)
        result["positive"].append(positive and within_strip)
        result["unique"].append(unique)
        result["context"].append(context)
        result["index"].append(index)
        result["image_id"].append(image_info["id"])

    for rank, index in enumerate(indices):
        row = predictions[index]
        box, target = proposal_boxes[rank], proposal_boxes[rank]
        positive, unique = False, False
        if len(targets):
            ious = pairwise_iou(box[None], targets)[0]
            ious[labels != row["category_id"]] = 0
            best = int(ious.argmax())
            positive = bool(ious[best] >= 0.5)
            if positive:
                target, unique = targets[best], best not in matched
                matched.add(best)
        context = [row["score"], float(overlap[rank, :rank].max()) if rank else 0,
                   rank / max(len(indices) - 1, 1),
                   (box[2] - box[0]) / image_info["width"],
                   (box[3] - box[1]) / image_info["height"],
                   float((overlap[rank] > 0.5).sum() - 1) / max(len(indices) - 1, 1)]
        add(box, target, index, positive, unique, context)
    if annotations is not None:
        scales, probabilities = jitter_scales(epoch, config.epochs)
        for target in targets:
            scale = generator.choice(scales, p=probabilities)
            size = np.tile(target[2:] - target[:2], 2)
            noise = generator.normal(0, scale, 4).clip(-0.15, 0.15)
            box = target + noise * size
            context = [0.9, 0.0, 0.0, (box[2] - box[0]) / image_info["width"],
                       (box[3] - box[1]) / image_info["height"], 0.0]
            add(box, target, -1, True, True, context)
    count = len(result["index"])
    return {
        "pixels": np.asarray(result["pixels"], np.uint8).reshape(
            count, 4, config.tangent_samples, config.normal_samples, 3
        ),
        **{k: np.asarray(result[k], np.float32).reshape(count, 4)
           for k in ("proposal", "target", "displacements")},
        "context": np.asarray(result["context"], np.float32).reshape(count, 6),
        **{k: np.asarray(result[k], np.bool_) for k in ("positive", "unique")},
        **{k: np.asarray(result[k], np.int64) for k in ("index", "image_id")},
    }


def _concat(banks):
    return {key: np.concatenate([bank[key] for bank in banks]) for key in banks[0]}


def iterate_frames(image_root: Path, images: list[dict], predictions: list[dict],
                   config: StripConfig, *, annotations=None, epoch=0, expected_hashes=None):
    selected = _selected(predictions, config)
    grouped_gt = defaultdict(list)
    for row in annotations or []:
        grouped_gt[row["image_id"]].append(row)
    order = np.arange(len(images))
    if annotations is not None:
        np.random.default_rng(config.seed + epoch).shuffle(order)

    def build(index):
        row = images[index]
        return frame_bank(image_root, row, predictions, selected.get(row["id"], []), config,
                          annotations=grouped_gt[row["id"]] if annotations is not None else None,
                          epoch=epoch,
                          expected_file_sha256=(expected_hashes[row["id"]]
                                                if expected_hashes is not None else None))

    # Only a bounded number of decoded frames/crops can exist at once. Futures
    # are consumed in submitted order, preserving deterministic training order.
    with ThreadPoolExecutor(max_workers=config.image_workers) as pool:
        pending = deque()
        iterator = iter(order)
        for _ in range(config.image_workers * 2):
            index = next(iterator, None)
            if index is not None:
                pending.append(pool.submit(build, index))
        while pending:
            yield pending.popleft().result()
            index = next(iterator, None)
            if index is not None:
                pending.append(pool.submit(build, index))


def iterate_batches(frames, batch_size):
    pending, count = [], 0
    for bank in frames:
        if not len(bank["index"]):
            continue
        pending.append(bank)
        count += len(bank["index"])
        if count >= batch_size:
            combined = _concat(pending)
            while len(combined["index"]) >= batch_size:
                yield {key: value[:batch_size] for key, value in combined.items()}
                combined = {key: value[batch_size:] for key, value in combined.items()}
            count = len(combined["index"])
            pending = [combined] if count else []
    if pending:
        yield _concat(pending)


def make_model(config: StripConfig):
    import torch
    from torch import nn
    from torch.nn import functional as F

    config.validate()
    width = config.width

    def separable(cin, cout, stride=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(4, cout), nn.SiLU(),
        )

    class NativeSideNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(5, width, 3, stride=(4, 1), padding=1,
                                                bias=False), nn.GroupNorm(4, width), nn.SiLU())
            self.side_embedding = nn.Embedding(4, width)
            self.fine = separable(width, width)
            self.middle = separable(width, width * 2, (2, 2))
            self.deep = separable(width * 2, width * 4, (2, 2))
            self.lateral2 = nn.Conv2d(width * 2, width, 1)
            self.lateral3 = nn.Conv2d(width * 4, width, 1)
            self.fusion = separable(width, width)
            self.attention = nn.Conv2d(width, 1, 1)
            self.boundary = nn.Sequential(nn.Conv1d(width * 2, width * 2, 5, padding=2),
                                          nn.SiLU(), nn.Conv1d(width * 2, 1, 3, padding=1))
            self.quality = nn.Sequential(nn.Linear(width * 4 + 14, 64), nn.SiLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.boundary[-1].weight)
            nn.init.zeros_(self.boundary[-1].bias)
            nn.init.zeros_(self.quality[-1].weight)
            nn.init.constant_(self.quality[-1].bias, 3)
            axis = ((torch.arange(config.normal_samples, dtype=torch.float32) + 0.5)
                    * (2 * config.normal_span / config.normal_samples) - config.normal_span)
            self.register_buffer("axis", axis)
            self.register_buffer("prior", -0.5 * (axis / 0.04) ** 2)

        def forward(self, pixels, context):
            batch, sides, _, height, normal = pixels.shape
            flat = pixels.reshape(batch * sides, 3, height, normal)
            x = torch.linspace(-1, 1, normal, device=flat.device).reshape(1, 1, 1, normal)
            y = torch.linspace(-1, 1, height, device=flat.device).reshape(1, 1, height, 1)
            features = torch.cat((flat, x.expand(batch * sides, 1, height, normal),
                                  y.expand(batch * sides, 1, height, normal)), 1)
            side = self.side_embedding(torch.arange(4, device=flat.device)).repeat(batch, 1)
            a = self.fine(self.stem(features) + side[:, :, None, None])
            b = self.middle(a)
            c = self.deep(b)
            fused = self.fusion(a + F.interpolate(self.lateral2(b), a.shape[-2:], mode="bilinear",
                                                  align_corners=False)
                                + F.interpolate(self.lateral3(c), a.shape[-2:], mode="bilinear",
                                                align_corners=False))
            attention = self.attention(fused).softmax(2)
            profile = torch.cat(((fused * attention).sum(2), fused.amax(2)), 1)
            logits = self.boundary(profile).reshape(batch, 4, normal) + self.prior
            probabilities = logits.softmax(-1)
            displacements = (probabilities * self.axis).sum(-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            statistics = torch.cat((probabilities.amax(-1), entropy), 1)
            global_features = c.mean((2, 3)).reshape(batch, 4, -1).mean(1)
            quality = self.quality(torch.cat((global_features, statistics, context), 1)).squeeze(-1)
            return {"logits": logits, "displacements": displacements, "quality": quality,
                    "side_features": profile.mean(-1).reshape(batch, 4, width * 2)}

    return NativeSideNetwork()


def distribution_targets(displacements, config: StripConfig):
    """Map the actual strip pixel centers to adjacent DFL bins, including subpixels."""
    position = ((displacements / config.normal_span + 1) * config.normal_samples / 2 - 0.5)
    return position / (config.normal_samples - 1)


def _batch(bank, device, *, generator=None):
    import torch

    pixels = torch.from_numpy(bank["pixels"].copy()).permute(0, 1, 4, 2, 3).float() / 255
    if generator is not None:
        reverse = torch.rand(len(pixels), generator=generator) < 0.5
        pixels[reverse] = pixels[reverse].flip(-2)
        brightness = 0.9 + 0.2 * torch.rand(len(pixels), 1, 1, 1, 1, generator=generator)
        contrast = 0.9 + 0.2 * torch.rand(len(pixels), 1, 1, 1, 1, generator=generator)
        pixels = ((pixels - 0.5) * contrast + 0.5).mul(brightness).clamp(0, 1)
    return ((pixels - 0.5) / 0.25).to(device), torch.from_numpy(bank["context"]).to(device)


def supervised_loss(outputs, bank, config: StripConfig):
    import torch
    from torch.nn import functional as F

    device = outputs["displacements"].device
    proposal = torch.from_numpy(bank["proposal"]).to(device)
    target = torch.from_numpy(bank["target"]).to(device)
    displacement = torch.from_numpy(bank["displacements"]).to(device)
    positive = torch.from_numpy(bank["positive"]).to(device)
    unique = torch.from_numpy(bank["unique"]).to(device)
    size = (proposal[:, 2:] - proposal[:, :2]).repeat(1, 2)
    sign = proposal.new_tensor([1, 1, -1, -1])
    decoded = proposal + outputs["displacements"] * size * sign
    if positive.any():
        loc = distribution_loss(outputs["logits"][positive],
                                distribution_targets(displacement[positive], config))
        loc = (loc + 10 * F.smooth_l1_loss(outputs["displacements"][positive],
                                          displacement[positive], beta=0.01)
               + 3 * (1 - _tensor_iou(decoded[positive], target[positive])).mean())
    else:
        loc = outputs["displacements"].sum() * 0
    quality_targets = unique.float() * _tensor_iou(decoded.detach(), target)
    quality = (F.binary_cross_entropy_with_logits(outputs["quality"], quality_targets,
                                                  reduction="none")
               * torch.where(unique, 1.0, 4.0)).mean()
    return loc + 0.25 * quality


def _safe_bytes(path: Path) -> bytes:
    path = path.absolute()
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 8 <= info.st_size <= 32 * 1024 * 1024:
                raise ValueError("Checkpoint must be a regular numeric file below 32 MiB")
            data = stream.read(32 * 1024 * 1024 + 1)
        if len(data) != info.st_size:
            raise ValueError("Checkpoint changed while reading")
        return data
    finally:
        os.close(directory)


def save_model(model, path: Path, config: StripConfig, metadata: dict):
    from safetensors.torch import save_file

    info = {"format": "native-side-strip-v1", "config": asdict(config), **metadata}
    temporary = path.with_suffix(".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              temporary, metadata={"vision_rl": json.dumps(info, sort_keys=True)})
    os.replace(temporary, path)
    _write(path.with_suffix(".json"), {**info, "sha256": _sha(path)})


def load_model(path: Path, *, device="cpu", expected_sha256=None):
    import torch
    from safetensors.torch import load

    data = _safe_bytes(path)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Crop checkpoint SHA differs from the frozen composition")
    header_size = int.from_bytes(data[:8], "little")
    if not 2 <= header_size <= min(1024 * 1024, len(data) - 8):
        raise ValueError("Invalid safetensors header size")
    header = json.loads(data[8:8 + header_size])
    metadata = json.loads(header["__metadata__"]["vision_rl"])
    if metadata.get("format") != "native-side-strip-v1":
        raise ValueError("Unknown strip model format")
    config = StripConfig(**{**metadata["config"], "device": device})
    config.validate()
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(min(
            1.0, config.mps_memory_gib * 1024**3 / torch.mps.recommended_max_memory()
        ))
    model = make_model(config)
    state = load(data)
    if any(not value.isfinite().all() for value in state.values()):
        raise ValueError("Nonfinite crop model tensors")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config, {**metadata, "checkpoint_sha256": digest}


def predict_batches(model, frames, config: StripConfig):
    import torch

    model.eval()
    with torch.inference_mode():
        for bank in iterate_batches(frames, config.batch_size):
            pixels, context = _batch(bank, config.device)
            output = model(pixels, context)
            yield (bank, output["displacements"].cpu().numpy(),
                   output["quality"].sigmoid().cpu().numpy())


def apply_refinement(predictions, indices, proposals, displacements, qualities,
                     *, blend=1.0, quality_blend=0.0):
    if not 0 <= blend <= 1 or not 0 <= quality_blend <= 1:
        raise ValueError("Blends must be in 0..1")
    result = copy.deepcopy(predictions)
    if displacements.shape != proposals.shape or qualities.shape != (len(proposals),):
        raise ValueError("Crop result shape mismatch")
    if not np.isfinite(displacements).all() or not np.isfinite(qualities).all():
        raise ValueError("Crop predictions must be finite")
    boxes = decode_edges(proposals, displacements * blend)
    for index, box, quality in zip(indices, boxes, qualities, strict=True):
        if index < 0 or index >= len(result):
            raise ValueError("Training-only crop cannot be used for inference")
        if (box[2:] <= box[:2]).any():
            raise ValueError("Crop prediction reversed a box")
        if blend:
            result[index]["bbox"] = [float(box[0]), float(box[1]),
                                      float(box[2] - box[0]), float(box[3] - box[1])]
        if quality_blend:
            result[index]["score"] *= float(np.clip(quality, 0, 1)) ** quality_blend
    return result


def _collect(model, frames, config):
    banks, displacements, qualities = [], [], []
    for bank, delta, quality in predict_batches(model, frames, config):
        banks.append({"index": bank["index"], "proposal": bank["proposal"]})
        displacements.append(delta)
        qualities.append(quality)
    if not banks:
        return (np.empty(0, np.int64), np.empty((0, 4), np.float32),
                np.empty((0, 4), np.float32), np.empty(0, np.float32))
    combined = _concat(banks)
    return (combined["index"], combined["proposal"],
            np.concatenate(displacements), np.concatenate(qualities))


def refine_manifest(checkpoint: Path, manifest: Path, image_root: Path, proposals: Path,
                    output: Path, *, device="cpu", composition: dict | None = None):
    """Image-only predictor; deployment provenance is distinct from training source."""
    import torch

    from vision_rl.detection_inference import load_image_manifest

    started = time.monotonic()
    composition = composition or {}
    allowed = {"checkpoint_sha256", "proposals_sha256", "manifest_sha256", "blend", "quality_blend",
               "deployment_detector_sha256", "image_tree_sha256"}
    if set(composition) - allowed:
        raise ValueError("Unknown frozen composition keys")
    hashes = {"proposals_sha256": _sha(proposals), "manifest_sha256": _sha(manifest)}
    for key, expected in composition.items():
        if key in hashes and expected != hashes[key]:
            raise ValueError(f"Frozen {key} differs")
    model, config, metadata = load_model(checkpoint, device=device,
                                         expected_sha256=composition.get("checkpoint_sha256"))
    torch.set_num_threads(config.cpu_threads)
    images = load_image_manifest(manifest, image_root)
    image_hashes = {row["id"]: _sha(row["path"]) for row in images}
    image_tree_sha256 = hashlib.sha256(json.dumps([
        [row["id"], row["file_name"], image_hashes[row["id"]]] for row in images
    ], separators=(",", ":")).encode()).hexdigest()
    if composition.get("image_tree_sha256", image_tree_sha256) != image_tree_sha256:
        raise ValueError("Frozen image tree differs")
    predictions = _load_predictions(proposals, images)
    frames = iterate_frames(image_root, images, predictions, config, expected_hashes=image_hashes)
    indices, boxes, displacements, qualities = _collect(model, frames, config)
    blend = composition.get("blend", metadata.get("blend", 1.0))
    quality_blend = composition.get("quality_blend", metadata.get("quality_blend", 0.0))
    refined = apply_refinement(predictions, indices, boxes, displacements, qualities,
                               blend=blend, quality_blend=quality_blend)
    if hashes != {"proposals_sha256": _sha(proposals), "manifest_sha256": _sha(manifest)}:
        raise ValueError("Inference inputs changed during prediction")
    if any(_sha(row["path"]) != image_hashes[row["id"]] for row in images):
        raise ValueError("Image bytes changed during prediction")
    _write(output, refined)
    report = {"format": "native-side-strip-predictions-v1", "images": len(images),
              "proposals": len(predictions), "refined_proposals": len(indices),
              "blend": blend, "quality_blend": quality_blend,
              "checkpoint_sha256": metadata["checkpoint_sha256"],
              "training_detector_sha256": metadata.get("training_detector_sha256"),
              "training_proposals_sha256": metadata.get("training_proposals_sha256"),
              "deployment_detector_sha256": composition.get("deployment_detector_sha256"),
              "image_tree_sha256": image_tree_sha256,
              **hashes, "wall_seconds": time.monotonic() - started,
              "labels_available_to_predictor": False}
    _write(output.with_suffix(".report.json"), report)
    return report


def _image_hashes(root: Path, images: list[dict]) -> tuple[list[str], list[str]]:
    root = root.resolve()
    files, pixels = [], []
    for row in images:
        path = (root / row["file_name"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Image path escapes its split")
        data = path.read_bytes()
        files.append(hashlib.sha256(data).hexdigest())
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
        digest = hashlib.sha256(f"{image.width}x{image.height}/RGB/".encode())
        digest.update(image.tobytes())
        pixels.append(digest.hexdigest())
    return files, pixels


def ap_at_iou(evaluation: dict, threshold=0.95) -> float:
    """Use the evaluator's exact cached matches, never a different localization metric."""
    from vision_rl.detection_eval import _cached_ap, _prepared_cache

    cache = _prepared_cache(evaluation["match_cache"])
    positions = np.flatnonzero(np.isclose(cache["iou_thresholds"], threshold))
    if len(positions) != 1:
        raise ValueError("Requested IoU is outside the native COCO protocol")
    index = int(positions[0])
    selected = {**cache, "iou_thresholds": cache["iou_thresholds"][index:index + 1],
                "items": {iid: {cid: {**row, "matches": row["matches"][index:index + 1],
                                      "ignored_detections":
                                          row["ignored_detections"][index:index + 1]}
                                for cid, row in classes.items()}
                          for iid, classes in cache["items"].items()}}
    return float(_cached_ap(selected, evaluation["image_ids"]))


def _publish_best(output: Path, best: dict) -> None:
    if "artifact" not in best:
        return
    source = output / best["artifact"]
    if _sha(source) != best["artifact_sha256"]:
        raise ValueError("Committed best artifact differs from the epoch checkpoint")
    temporary = output / "best.safetensors.tmp"
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, output / "best.safetensors")
    _write(output / "best.json", json.loads(source.with_suffix(".json").read_text()))


def train(source: Path, train_proposals: Path, validation_proposals: Path, output: Path,
          detector_sha256: str, config: StripConfig, *, resume=False):
    """Single-writer training; an epoch state commits the selected model atomically."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".train.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another trainer owns this output directory") from None
        return _train(source, train_proposals, validation_proposals, output,
                      detector_sha256, config, resume=resume)


def _train(source: Path, train_proposals: Path, validation_proposals: Path, output: Path,
           detector_sha256: str, config: StripConfig, *, resume=False):
    import torch

    from vision_rl.detection_eval import evaluate_coco, load_coco

    config.validate()
    if len(detector_sha256) != 64 or any(c not in "0123456789abcdef" for c in detector_sha256):
        raise ValueError("An exact detector SHA256 is required")
    if any(p.name != ".train.lock" for p in output.iterdir()) and not resume:
        raise FileExistsError("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    if config.device == "mps":
        torch.mps.set_per_process_memory_fraction(min(
            1.0, config.mps_memory_gib * 1024**3 / torch.mps.recommended_max_memory()
        ))
    generator = torch.Generator().manual_seed(config.seed + 1)
    started = time.monotonic()
    paths = {s: source / s / "_annotations.coco.json" for s in ("train", "valid")}
    documents = {s: load_coco(path) for s, path in paths.items()}
    predictions = {"train": _load_predictions(train_proposals, documents["train"]["images"]),
                   "valid": _load_predictions(validation_proposals, documents["valid"]["images"])}
    hashes = {s: _image_hashes(source / s, documents[s]["images"]) for s in paths}
    image_hashes, pixel_hashes = ({s: hashes[s][i] for s in paths} for i in (0, 1))
    if set(image_hashes["train"]) & set(image_hashes["valid"]):
        raise ValueError("Exact image content overlaps training and validation")
    if set(pixel_hashes["train"]) & set(pixel_hashes["valid"]):
        raise ValueError("Decoded image pixels overlap training and validation")
    metadata = {"training_detector_sha256": detector_sha256,
                "training_proposals_sha256": _sha(train_proposals),
                "validation_proposals_sha256": _sha(validation_proposals),
                "annotation_sha256": {s: _sha(p) for s, p in paths.items()},
                "image_tree_sha256": {s: hashlib.sha256(json.dumps(h).encode()).hexdigest()
                                      for s, h in image_hashes.items()},
                "pixel_tree_sha256": {s: hashlib.sha256(json.dumps(h).encode()).hexdigest()
                                      for s, h in pixel_hashes.items()},
                "source_sha256": _sha(Path(__file__)), "test_split_accessed": False}
    metadata["dependency_sha256"] = {
        name: _sha(Path(__file__).with_name(name)) for name in
        ("crop_refinement.py", "detection_eval.py", "detection_inference.py")
    }
    metadata["packages"] = {
        name: version(name) for name in ("torch", "numpy", "Pillow", "safetensors")
    }
    plan = {"config": asdict(config), **metadata,
            "promotion_criteria": {"AP_gain_over_best_v1_composition": 0.01,
                                   "AP95_gain_over_same_v1_composition": 0.10,
                                   "max_epochs": config.epochs},
            "curriculum": "fresh jitter/GT each epoch; .05/.012/.004 sigma, coarse20% then5%"}
    if resume:
        if json.loads((output / "plan.json").read_text()) != json.loads(json.dumps(plan)):
            raise ValueError("Resume inputs/configuration/implementation changed")
    else:
        _write(output / "plan.json", plan)
    # Validation has pixels/proposals only. Labels are read exclusively by COCO scoring.
    validation_frames = list(iterate_frames(source / "valid", documents["valid"]["images"],
                                            predictions["valid"], config, expected_hashes=dict(zip(
                                                [r["id"] for r in documents["valid"]["images"]],
                                                image_hashes["valid"], strict=True))))
    print(json.dumps({"event": "validation_strips_ready", "crops": sum(
        len(bank["index"]) for bank in validation_frames), "seconds": time.monotonic() - started}),
          flush=True)
    model = make_model(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs,
                                                          eta_min=config.learning_rate / 20)
    baseline = evaluate_coco(paths["valid"], predictions["valid"])
    _write(output / "baseline.json", {**{k: baseline[k] for k in ("AP", "AP50", "AP75", "n")},
                                      "AP95": ap_at_iou(baseline)})
    best = {"AP": -1.0}
    first_epoch, prior_elapsed, history = 0, 0.0, []
    if resume:
        state = torch.load(output / "trainer-state.pt", map_location="cpu", weights_only=True)
        if state.get("version") != 1:
            raise ValueError("Unknown strip trainer state version")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        first_epoch, history, best = state["epoch"], state["history"], state["best"]
        prior_elapsed = state["elapsed_seconds"]
        _publish_best(output, best)
    for epoch in range(first_epoch, config.epochs):
        epoch_started = time.monotonic()
        model.train()
        frames = iterate_frames(source / "train", documents["train"]["images"],
                                predictions["train"], config,
                                annotations=documents["train"]["annotations"], epoch=epoch,
                                expected_hashes=dict(zip(
                                    [r["id"] for r in documents["train"]["images"]],
                                    image_hashes["train"], strict=True)))
        losses, total_crops, high_iou = [], 0, 0
        for bank in iterate_batches(frames, config.batch_size):
            pixels, context = _batch(bank, config.device, generator=generator)
            outputs = model(pixels, context)
            loss = supervised_loss(outputs, bank, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
            total_crops += len(bank["index"])
            valid = bank["positive"]
            overlaps = pairwise_iou(bank["proposal"][valid], bank["target"][valid])
            high_iou += int((np.diag(overlaps) >= 0.9).sum())
        scheduler.step()
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                  "train_crops": total_crops, "positive_crops_with_input_IoU_ge90": high_iou,
                  "train_seconds": time.monotonic() - epoch_started}
        if config.device == "mps":
            record["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        if (epoch + 1) % config.evaluate_every == 0 or epoch + 1 == config.epochs:
            if {s: _sha(p) for s, p in paths.items()} != metadata["annotation_sha256"]:
                raise ValueError("Annotation inputs changed before validation")
            indices, boxes, deltas, qualities = _collect(model, iter(validation_frames), config)
            candidates = []
            for blend in config.blend_candidates:
                for quality_blend in config.quality_candidates:
                    refined = apply_refinement(predictions["valid"], indices, boxes, deltas,
                                                qualities, blend=blend,
                                                quality_blend=quality_blend)
                    evaluation = evaluate_coco(paths["valid"], refined)
                    candidate = {k: evaluation[k] for k in ("AP", "AP50", "AP75")}
                    candidate["AP95"] = ap_at_iou(evaluation)
                    candidate.update(epoch=epoch + 1, blend=blend, quality_blend=quality_blend)
                    candidates.append(candidate)
                    if candidate["AP"] > best["AP"]:
                        artifact = output / f"epoch-{epoch + 1:03d}-best.safetensors"
                        save_model(model, artifact, config,
                                   {**metadata, **candidate})
                        best = {**candidate, "artifact": artifact.name,
                                "artifact_sha256": _sha(artifact)}
            if {s: _sha(p) for s, p in paths.items()} != metadata["annotation_sha256"]:
                raise ValueError("Annotation inputs changed during validation")
            record["validation_candidates"] = candidates
            record["best"] = best
        record["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
        history.append(record)
        save_model(model, output / "last.safetensors", config, {**metadata, "epoch": epoch + 1})
        _write(output / "history.json", history)
        temporary = output / "trainer-state.pt.tmp"
        torch.save({"version": 1, "epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                    "history": history, "best": best, "elapsed_seconds": record["elapsed_seconds"]},
                   temporary)
        os.replace(temporary, output / "trainer-state.pt")
        _publish_best(output, best)
        print(json.dumps(record, sort_keys=True), flush=True)
    best_model, best_config, best_metadata = load_model(output / "best.safetensors",
                                                       device=config.device)
    indices, boxes, deltas, qualities = _collect(best_model, iter(validation_frames), best_config)
    refined = apply_refinement(predictions["valid"], indices, boxes, deltas, qualities,
                                blend=best_metadata["blend"],
                                quality_blend=best_metadata["quality_blend"])
    if {s: _sha(p) for s, p in paths.items()} != metadata["annotation_sha256"]:
        raise ValueError("Annotation inputs changed before final validation replay")
    evaluation = evaluate_coco(paths["valid"], refined)
    if abs(evaluation["AP"] - best["AP"]) > 1e-12:
        raise ValueError("Selected artifact does not replay its validation score")
    _write(output / "best-evaluation.json", evaluation)
    _write(output / "best-predictions.coco.json", refined)
    summary = {"baseline_AP": baseline["AP"], "best": best,
               "AP_gain": best["AP"] - baseline["AP"], "epochs": len(history),
               "parameters": sum(p.numel() for p in model.parameters()),
               "wall_seconds": prior_elapsed + time.monotonic() - started, **metadata}
    _write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    for name in ("source", "train-proposals", "validation-proposals", "output"):
        training.add_argument(f"--{name}", type=Path, required=True)
    training.add_argument("--detector-sha256", required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--resume", action="store_true")
    prediction = commands.add_parser("predict")
    for name in ("checkpoint", "manifest", "image-root", "proposals", "output"):
        prediction.add_argument(f"--{name}", type=Path, required=True)
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--composition", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "train":
        config_path = args.pop("config")
        config = (StripConfig(**json.loads(config_path.read_text()))
                  if config_path else StripConfig())
        report = train(config=config, **args)
    else:
        composition_path = args.pop("composition")
        composition = json.loads(composition_path.read_text()) if composition_path else None
        report = refine_manifest(composition=composition, **args)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
