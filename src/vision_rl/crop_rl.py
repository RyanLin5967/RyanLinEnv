"""Pixel-conditioned, image-reward refinement of a frozen supervised crop model.

The observation is a native-pixel boundary encoder's per-side feature and boundary
probability distribution. Five small native-pixel actions independently move each
edge. Detector proposals, scores, categories, and order remain frozen. The policy
therefore learns localization, rather than obtaining apparent gains by filtering
proposals or changing the supervised quality head.

Training uses exact score-ordered, unique matching at ten COCO IoU thresholds and
101-point *per-image* AP. This is an image-level surrogate for dataset-level AP;
checkpoint selection always uses the full native dataset evaluator. A group
leave-one-out score-function estimator credits all sampled edge actions with the
team reward, regularized against an explicit symmetric zero-mean SFT prior.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vision_rl.detection_rewards import interpolated_ap
from vision_rl.detection_rl import box_iou, leave_one_out_advantage

CACHE_VERSION = 1
SIDE_ACTIONS = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
JOINT_ACTIONS = torch.tensor(list(itertools.product(range(5), repeat=4)), dtype=torch.int64)
JOINT_WEIGHTS = torch.tensor([125, 25, 5, 1], dtype=torch.int64)
IDENTITY_JOINT_ACTION = 312


@dataclass(frozen=True)
class PixelRLConfig:
    hidden: int = 64
    groups: int = 16
    epochs: int = 12
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    kl_weight: float = 0.01
    entropy_weight: float = 0.00001
    edge_identity_probability: float = 0.8
    max_offset_pixels: float = 4.0
    max_offset_fraction: float = 0.02
    recall_bonus: float = 0.2
    batch_images: int = 8
    cpu_threads: int = 1
    seed: int = 731
    pilot_images: int = 128
    pilot_groups: int = 64
    minimum_pilot_gain: float = 0.0001
    minimum_validation_gain: float = 0.001

    def validate(self) -> None:
        for key, low, high in (
            ("hidden", 8, 256),
            ("groups", 2, 128),
            ("epochs", 1, 100),
            ("batch_images", 1, 128),
            ("cpu_threads", 1, 16),
            ("pilot_images", 1, 10000),
            ("pilot_groups", 2, 256),
        ):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer between {low} and {high}")
        for key, low, high in (
            ("learning_rate", 1e-7, 0.01),
            ("weight_decay", 0, 0.1),
            ("kl_weight", 0, 1),
            ("entropy_weight", 0, 0.1),
            ("edge_identity_probability", 0.2, 0.99),
            ("max_offset_pixels", 0.1, 16),
            ("max_offset_fraction", 0.001, 0.1),
            ("recall_bonus", 0, 1),
            ("minimum_pilot_gain", 0, 0.1),
            ("minimum_validation_gain", 0, 0.1),
        ):
            value = getattr(self, key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{key} must be finite between {low} and {high}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def _write(path: Path, document: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    os.replace(temporary, path)


def validate_cache(cache: dict, expected_split: str | None = None) -> None:
    """Check observations without inspecting targets at inference call sites."""
    if cache.get("version") != CACHE_VERSION:
        raise ValueError("unknown pixel-side cache version")
    if expected_split is not None and cache.get("split") != expected_split:
        raise ValueError(f"expected {expected_split} pixel-side cache")
    required = ("boxes", "scores", "selected_indices", "side_features", "side_logits")
    if any(not isinstance(cache.get(key), torch.Tensor) for key in required):
        raise ValueError("pixel-side cache requires numeric observation tensors")
    boxes, scores, selected, features, logits = (cache[key] for key in required)
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.shape != boxes.shape[:1]:
        raise ValueError("boxes/scores must be aligned [proposals,4]/[proposals]")
    if selected.ndim != 1 or selected.dtype != torch.int64 or len(selected) != len(features):
        raise ValueError("selected_indices must align with side features")
    if len(selected) and (selected.min() < 0 or selected.max() >= len(boxes)):
        raise ValueError("selected proposal index out of bounds")
    if len(selected.unique()) != len(selected):
        raise ValueError("selected proposal indices must be unique")
    if features.ndim != 3 or features.shape[1] != 4 or not 1 <= features.shape[2] <= 1024:
        raise ValueError("side_features must have shape [selected,4,D], D<=1024")
    if (
        logits.ndim != 3
        or logits.shape[:2] != features.shape[:2]
        or not 3 <= logits.shape[2] <= 512
    ):
        raise ValueError("side_logits must align with features and have 3..512 bins")
    for name in ("boxes", "scores", "side_features", "side_logits"):
        value = cache[name]
        allowed = (
            (torch.float32, torch.float64) if name in {"boxes", "scores"} else (torch.float32,)
        )
        if value.dtype not in allowed or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating point values")
    if (boxes[:, 2:] <= boxes[:, :2]).any() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("invalid frozen proposal boxes or scores")
    images = cache.get("images", [])
    ids = [row["id"] for row in images]
    if len(ids) != len(set(ids)) or not images:
        raise ValueError("image IDs must be unique and nonempty")
    if any(set(row) & {"annotations", "targets", "labels"} for row in images):
        raise ValueError("image metadata must not contain target fields")
    if len(cache.get("image_ids", [])) != len(boxes) or len(cache.get("category_ids", [])) != len(
        boxes
    ):
        raise ValueError("proposal image/category IDs must align with boxes")
    if set(cache["image_ids"].tolist()) - set(ids):
        raise ValueError("proposal references an unknown image")
    if len(set(cache["category_ids"].tolist())) > 1:
        raise ValueError("pixel-side RL currently supports exactly one object category")
    if any(int((cache["image_ids"] == image_id).sum()) > 500 for image_id in ids):
        raise ValueError("native reward supports at most 500 proposals per image")
    if "base_predictions" in cache:
        predictions = cache["base_predictions"]
        if len(predictions) != len(boxes):
            raise ValueError("frozen prediction records must align with proposals")
        coordinates = boxes.new_tensor(
            [[x, y, x + w, y + h] for row in predictions for x, y, w, h in [row["bbox"]]]
        ).reshape(-1, 4)
        if (
            not torch.equal(coordinates, boxes)
            or scores.tolist() != [row["score"] for row in predictions]
            or cache["image_ids"].tolist() != [row["image_id"] for row in predictions]
            or cache["category_ids"].tolist() != [row["category_id"] for row in predictions]
        ):
            raise ValueError("frozen prediction records differ from numeric observations")
    for key in ("detector_sha256", "supervised_sha256", "proposal_pipeline_sha256"):
        value = cache.get("metadata", {}).get(key, "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"cache must bind a valid {key}")
    if expected_split is not None:
        targets = cache.get("targets")
        if not isinstance(targets, list) or len(targets) != len(images):
            raise ValueError("train/validation targets must align with image metadata")
        for target in targets:
            if (
                not isinstance(target, torch.Tensor)
                or target.ndim != 2
                or target.shape[-1] != 4
                or not torch.isfinite(target).all()
                or (target[:, 2:] <= target[:, :2]).any()
            ):
                raise ValueError("targets must be finite native-pixel xyxy tensors")


def load_cache(path: Path, expected_split: str | None = None) -> dict:
    if path.is_symlink() or path.stat().st_size > 1024**3:
        raise ValueError("pixel-side cache must be a regular file under 1 GiB")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    validate_cache(cache, expected_split)
    return cache


def cache_fingerprint(cache: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(cache):
        value = cache[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.cpu().contiguous().numpy().tobytes())
        elif key == "targets":
            for item in value:
                digest.update(str(tuple(item.shape)).encode())
                digest.update(item.cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _training_invariants(train: dict, validation: dict) -> None:
    validate_cache(train, "train")
    validate_cache(validation, "validation")
    for key in (
        "detector_sha256",
        "supervised_sha256",
        "proposal_pipeline_sha256",
        "supervised_composition_sha256",
        "observation_source_sha256",
    ):
        if train["metadata"].get(key) != validation["metadata"].get(key):
            raise ValueError(f"train/validation {key} mismatch")
    if train["side_features"].shape[-1] != validation["side_features"].shape[-1]:
        raise ValueError("side feature dimensions differ across splits")
    train_pixels = {row["pixel_sha256"] for row in train["images"]}
    val_pixels = {row["pixel_sha256"] for row in validation["images"]}
    if train_pixels & val_pixels:
        raise ValueError("training and validation decoded pixels overlap")
    if not len(train["side_features"]):
        raise ValueError("training requires at least one native pixel region")


@torch.no_grad()
def export_pixel_cache(
    checkpoint: Path,
    manifest: Path,
    image_root: Path,
    proposals: Path,
    output: Path | None = None,
    *,
    split: str = "inference",
    detector_sha256: str,
    proposal_pipeline_sha256: str,
    annotation_path: Path | None = None,
    composition: dict | None = None,
    cpu_threads: int = 1,
) -> dict:
    """Export frozen SFT observations; labels attach only after image-only inference.

    ``proposals`` are the exact inputs to the supervised side-strip stage, not
    its refined outputs. Geometry, features, quality scores, and selected indices
    are collected in one forward pass and retained in that same order.
    """
    from vision_rl import crop_strips as strips
    from vision_rl.crop_refinement import _load_predictions
    from vision_rl.detection_inference import load_image_manifest

    if split not in {"train", "validation", "inference"}:
        raise ValueError("pixel cache split must be train, validation, or inference")
    if (annotation_path is not None) != (split in {"train", "validation"}):
        raise ValueError("only train/validation cache export accepts and requires annotations")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 16:
        raise ValueError("CPU threads must be an integer in 1..16")
    torch.set_num_threads(cpu_threads)
    composition = composition or {}
    allowed = {
        "checkpoint_sha256",
        "proposals_sha256",
        "manifest_sha256",
        "blend",
        "quality_blend",
        "deployment_detector_sha256",
    }
    if set(composition) - allowed:
        raise ValueError("unknown frozen supervised composition keys")
    hashes = {"proposals_sha256": _sha(proposals), "manifest_sha256": _sha(manifest)}
    for key, expected in composition.items():
        if key in hashes and expected != hashes[key]:
            raise ValueError(f"frozen {key} differs")
    if composition.get("deployment_detector_sha256", detector_sha256) != detector_sha256:
        raise ValueError("frozen deployment detector differs")
    annotation_sha = _sha(annotation_path) if annotation_path is not None else None
    model, config, metadata = strips.load_model(
        checkpoint, device="cpu", expected_sha256=composition.get("checkpoint_sha256")
    )
    images = load_image_manifest(manifest, image_root)
    predictions = _load_predictions(proposals, images)
    files, pixels = strips._image_hashes(image_root, images)
    collected: dict[str, list] = {
        name: []
        for name in ("indices", "proposals", "displacements", "qualities", "features", "logits")
    }
    frames = strips.iterate_frames(image_root, images, predictions, config, annotations=None)
    for bank in strips.iterate_batches(frames, config.batch_size):
        rgb, context = strips._batch(bank, "cpu")
        values = model(rgb, context)
        collected["indices"].append(torch.from_numpy(bank["index"]).clone())
        collected["proposals"].append(torch.from_numpy(bank["proposal"]).clone())
        for key, value in (
            ("displacements", values["displacements"]),
            ("qualities", values["quality"].sigmoid()),
            ("features", values["side_features"]),
            ("logits", values["logits"]),
        ):
            collected[key].append(value.detach().cpu().clone())
    shapes = {
        "indices": (0,),
        "proposals": (0, 4),
        "displacements": (0, 4),
        "qualities": (0,),
        "features": (0, 4, config.width * 2),
        "logits": (0, 4, config.normal_samples),
    }
    arrays = {
        key: torch.cat(parts)
        if parts
        else torch.empty(shapes[key], dtype=torch.int64 if key == "indices" else torch.float32)
        for key, parts in collected.items()
    }
    blend = composition.get("blend", metadata.get("blend", 1.0))
    quality_blend = composition.get("quality_blend", metadata.get("quality_blend", 0.0))
    frozen = strips.apply_refinement(
        predictions,
        arrays["indices"].numpy(),
        arrays["proposals"].numpy(),
        arrays["displacements"].numpy(),
        arrays["qualities"].numpy(),
        blend=blend,
        quality_blend=quality_blend,
    )
    supervised_composition = {
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "blend": blend,
        "quality_blend": quality_blend,
    }
    cache = {
        "version": CACHE_VERSION,
        "split": split,
        "images": [
            {key: row[key] for key in ("id", "file_name", "width", "height")}
            | {"pixel_sha256": pixels[i], "file_sha256": files[i]}
            for i, row in enumerate(images)
        ],
        "boxes": torch.tensor(
            [[x, y, x + w, y + h] for row in frozen for x, y, w, h in [row["bbox"]]],
            dtype=torch.float64,
        ).reshape(-1, 4),
        "scores": torch.tensor([row["score"] for row in frozen], dtype=torch.float64),
        "image_ids": torch.tensor([row["image_id"] for row in frozen], dtype=torch.int64),
        "category_ids": torch.tensor([row["category_id"] for row in frozen], dtype=torch.int64),
        "base_predictions": frozen,
        "selected_indices": arrays["indices"],
        "side_features": arrays["features"],
        "side_logits": arrays["logits"],
        "metadata": {
            "detector_sha256": detector_sha256,
            "supervised_sha256": metadata["checkpoint_sha256"],
            "proposal_pipeline_sha256": proposal_pipeline_sha256,
            "supervised_composition_sha256": hashlib.sha256(
                json.dumps(supervised_composition, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "supervised_composition": supervised_composition,
            "observation_source_sha256": _sha(Path(strips.__file__)),
            "quality_head_frozen": True,
            **hashes,
        },
    }
    # Image features above never see labels, even when exporting a training cache.
    if annotation_path is not None:
        from vision_rl.detection_eval import load_coco

        document = load_coco(annotation_path)
        annotations = {row["id"]: [] for row in images}
        native_images = {row["id"]: row for row in document["images"]}
        categories = set(cache["category_ids"].tolist())
        for row in images:
            native = native_images.get(row["id"], {})
            if any(row[key] != native.get(key) for key in ("file_name", "width", "height")):
                raise ValueError("native annotation images differ from image-only manifest")
        for row in document["annotations"]:
            if row["image_id"] in annotations:
                if row["category_id"] not in categories or row.get("iscrowd", 0):
                    raise ValueError("pixel reward requires one category and no crowd labels")
                x, y, w, h = row["bbox"]
                annotations[row["image_id"]].append([x, y, x + w, y + h])
        cache["targets"] = [
            torch.tensor(annotations[row["id"]], dtype=torch.float64).reshape(-1, 4)
            for row in images
        ]
        cache["metadata"].update(
            annotation_path=str(annotation_path.resolve()), annotation_sha256=annotation_sha
        )
        if _sha(annotation_path) != annotation_sha:
            raise ValueError("annotations changed during observation export")
    if hashes != {"proposals_sha256": _sha(proposals), "manifest_sha256": _sha(manifest)}:
        raise ValueError("inference inputs changed during observation export")
    # Images are cheap to hash relative to neural extraction and are pinned on both sides.
    if files != [strips._sha((image_root / row["file_name"]).resolve()) for row in images]:
        raise ValueError("image bytes changed during observation export")
    validate_cache(cache, split if annotation_path is not None else None)
    if output is not None:
        _save(output, cache)
        _write(
            output.with_suffix(".json"),
            {
                "split": split,
                "images": len(images),
                "proposals": len(frozen),
                "selected_pixel_regions": len(arrays["indices"]),
                "cache_sha256": cache_fingerprint(cache),
                "metadata": cache["metadata"],
            },
        )
    return cache


def offset_scales(boxes: torch.Tensor, config: PixelRLConfig) -> torch.Tensor:
    """Edgewise pixel bounds; even the extreme action cannot reverse a box."""
    sizes = boxes[..., 2:] - boxes[..., :2]
    return (torch.cat((sizes, sizes), dim=-1) * config.max_offset_fraction).clamp_max(
        config.max_offset_pixels
    )


def action_boxes(boxes: torch.Tensor, actions: torch.Tensor, config: PixelRLConfig) -> torch.Tensor:
    return boxes + SIDE_ACTIONS.to(boxes.device)[actions] * offset_scales(boxes, config)


def observation_features(cache: dict, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    encoded = ((cache["side_features"] - mean) / scale).clamp(-8, 8)
    probabilities = cache["side_logits"].softmax(-1)
    bins = probabilities.shape[-1]
    axis = torch.linspace(-1, 1, bins)
    expected = (probabilities * axis).sum(-1)
    uncertainty = (probabilities * (axis - expected[..., None]).square()).sum(-1).sqrt()
    entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum(-1) / math.log(bins)
    profile = F.interpolate(
        probabilities.reshape(-1, 1, bins), 32, mode="linear", align_corners=True
    ).reshape(len(encoded), 4, 32)
    profile = profile / profile.sum(-1, keepdim=True).clamp_min(1e-12)
    boxes = cache["boxes"][cache["selected_indices"]]
    sizes = boxes[:, 2:] - boxes[:, :2]
    scores = cache["scores"][cache["selected_indices"]]
    image_sizes = {row["id"]: (row["width"], row["height"]) for row in cache["images"]}
    dimensions = torch.tensor(
        [image_sizes[int(i)] for i in cache["image_ids"][cache["selected_indices"]]]
    )
    geometry = torch.cat(
        (boxes / dimensions.repeat(1, 2), sizes.log() / 10, scores[:, None]), -1
    ).float()
    side_id = torch.eye(4)[None].expand(len(encoded), -1, -1)
    context = encoded.mean(1, keepdim=True).expand_as(encoded)
    stats = torch.stack((expected, uncertainty, entropy, probabilities.amax(-1)), -1)
    return torch.cat(
        (encoded, context, profile, stats, side_id, geometry[:, None].expand(-1, 4, -1)), -1
    )


class PixelSidePolicy(nn.Module):
    """One shared contextual policy; side identity is part of the observation."""

    def __init__(self, feature_dim: int, config: PixelRLConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, config.hidden),
            nn.SiLU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, 5),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        prior = torch.full((5,), (1 - config.edge_identity_probability) / 4)
        prior[2] = config.edge_identity_probability
        self.register_buffer("reference_logprob", prior.log())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features) + self.reference_logprob


@torch.no_grad()
def match_iou_reward(ious: torch.Tensor, scores: torch.Tensor, *, recall_bonus: float = 0.2):
    """Exact native ranking/matching, retaining every frozen proposal as FP or TP.

    Only proposals incapable of any match are skipped in the *matching loop*.
    Their slots remain in precision denominators and score order. This is exact,
    unlike dropping low-confidence detections from the reward computation.
    """
    groups, queries, targets = ious.shape
    order = scores.argsort(descending=True, stable=True)
    ordered = ious[:, order]
    hits = torch.zeros(groups, 10, queries, dtype=torch.bool)
    if targets:
        claimed = torch.zeros(groups, 10, targets, dtype=torch.bool)
        thresholds = torch.tensor(np.linspace(0.5, 0.95, 10), dtype=ious.dtype)
        eligible = torch.nonzero((ordered >= 0.5).any((0, 2))).flatten()
        for rank in eligible:
            available = ordered[:, rank, None].expand(-1, 10, -1).masked_fill(claimed, -1)
            # COCO breaks exact target-IoU ties in favor of the later target.
            best, reverse_target = available.flip(-1).max(-1)
            target = targets - 1 - reverse_target
            hit = best >= thresholds
            hits[:, :, rank] = hit
            claimed.scatter_(
                -1, target[..., None], claimed.gather(-1, target[..., None]) | hit[..., None]
            )
    ap = interpolated_ap(hits, torch.ones(groups, queries, dtype=torch.bool), targets)
    recall = hits.float().sum(-1).mean(-1) / max(targets, 1)
    return ap + recall_bonus * recall, {
        "AP": ap,
        "AP95": interpolated_ap(
            hits[:, -1:], torch.ones(groups, queries, dtype=torch.bool), targets
        ),
        "recall": recall,
    }


@dataclass
class ImageRewardTable:
    image_index: int
    selected_rows: torch.Tensor
    selected_local_indices: torch.Tensor
    baseline_ious: torch.Tensor
    candidate_ious: torch.Tensor
    scores: torch.Tensor
    baseline_reward: float
    baseline_ap: float
    baseline_ap95: float

    def rewards(self, actions: torch.Tensor, recall_bonus: float):
        combined = (actions * JOINT_WEIGHTS).sum(-1)
        groups = len(actions)
        ious = self.baseline_ious[None].expand(groups, -1, -1).clone()
        if len(self.selected_rows):
            rows = torch.arange(len(self.selected_rows))[None].expand(groups, -1)
            ious[:, self.selected_local_indices] = self.candidate_ious[rows, combined]
        return match_iou_reward(ious, self.scores, recall_bonus=recall_bonus)


def prepare_rewards(cache: dict, config: PixelRLConfig) -> list[ImageRewardTable]:
    """Precompute only train-ground-truth IoUs for 625 bounded actions per crop."""
    if cache.get("split") != "train":
        raise ValueError("reward preparation requires training split")
    selected = cache["selected_indices"]
    selected_images = cache["image_ids"][selected]
    tables = []
    for image_index, image in enumerate(cache["images"]):
        indices = torch.nonzero(cache["image_ids"] == image["id"]).flatten()
        rows = torch.nonzero(selected_images == image["id"]).flatten()
        local = torch.searchsorted(indices, selected[rows])
        boxes, scores = cache["boxes"][indices], cache["scores"][indices]
        targets = cache["targets"][image_index]
        base_iou = box_iou(boxes, targets)
        if len(rows):
            candidates = action_boxes(boxes[local, None], JOINT_ACTIONS[None], config)
            candidate_iou = box_iou(candidates, targets)
        else:
            candidate_iou = torch.empty(0, 625, len(targets))
        utility, stats = match_iou_reward(base_iou[None], scores, recall_bonus=config.recall_bonus)
        tables.append(
            ImageRewardTable(
                image_index,
                rows,
                local,
                base_iou,
                candidate_iou,
                scores,
                float(utility[0]),
                float(stats["AP"][0]),
                float(stats["AP95"][0]),
            )
        )
    return tables


def sample_actions(logits: torch.Tensor, groups: int, generator: torch.Generator) -> torch.Tensor:
    queries, sides, actions = logits.shape
    return (
        torch.multinomial(
            logits.detach().softmax(-1).reshape(-1, actions),
            groups,
            replacement=True,
            generator=generator,
        )
        .reshape(queries, sides, groups)
        .permute(2, 0, 1)
    )


def policy_gradient(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    reference_logprob: torch.Tensor,
    config: PixelRLConfig,
):
    """Unbiased group leave-one-out score-function gradient of the image utility."""
    logprob = logits.log_softmax(-1)
    chosen = (
        logprob[None].expand(len(actions), -1, -1, -1).gather(-1, actions[..., None]).squeeze(-1)
    )
    advantage = leave_one_out_advantage(rewards[:, None]).squeeze(-1)
    policy_loss = -(advantage * chosen.sum((-1, -2))).mean()
    probability = logprob.exp()
    kl = (probability * (logprob - reference_logprob)).sum()
    entropy = -(probability * logprob).sum()
    loss = policy_loss + config.kl_weight * kl - config.entropy_weight * entropy
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "joint_kl": float(kl.detach()),
        "joint_entropy": float(entropy.detach()),
        "reward_std": float(rewards.std()),
    }


def exploration_pilot(cache: dict, tables: list[ImageRewardTable], config: PixelRLConfig) -> dict:
    """Training-only attainable/sample headroom: never reported as model accuracy."""
    generator = torch.Generator().manual_seed(config.seed + 991)
    eligible = [table for table in tables if len(table.selected_rows)]
    chosen = torch.randperm(len(eligible), generator=generator)[: config.pilot_images].tolist()
    prior = torch.full((5,), (1 - config.edge_identity_probability) / 4)
    prior[2] = config.edge_identity_probability
    records = []
    for index in chosen:
        table = eligible[index]
        logits = prior.log().expand(len(table.selected_rows), 4, 5)
        actions = sample_actions(logits, config.pilot_groups, generator)
        _, stats = table.rewards(actions, config.recall_bonus)
        # Diagnostic optimistic per-proposal geometry, not a deployable detector.
        if table.candidate_ious.shape[-1]:
            best_joint = table.candidate_ious.amax(-1).argmax(-1)
        else:
            best_joint = torch.full((len(table.selected_rows),), IDENTITY_JOINT_ACTION)
        _, oracle = table.rewards(JOINT_ACTIONS[best_joint][None], config.recall_bonus)
        records.append(
            {
                "image_id": cache["images"][table.image_index]["id"],
                "baseline_AP": table.baseline_ap,
                "baseline_AP95": table.baseline_ap95,
                "best_of_samples_AP": float(stats["AP"].max()),
                "sampled_AP_std": float(stats["AP"].std()),
                "oracle_AP": float(oracle["AP"][0]),
                "oracle_AP95": float(oracle["AP95"][0]),
            }
        )
    sample_gain = sum(row["best_of_samples_AP"] - row["baseline_AP"] for row in records) / max(
        len(records), 1
    )
    oracle_gain = sum(row["oracle_AP"] - row["baseline_AP"] for row in records) / max(
        len(records), 1
    )
    return {
        "split": "train",
        "images": len(records),
        "samples_per_image": config.pilot_groups,
        "mean_best_of_samples_AP_gain": sample_gain,
        "mean_oracle_AP_gain": oracle_gain,
        "go": bool(records and sample_gain >= config.minimum_pilot_gain),
        "not_a_model_score": True,
        "items": records,
    }


@torch.no_grad()
def predictions_from_policy(
    cache: dict,
    policy: PixelSidePolicy,
    mean: torch.Tensor,
    scale: torch.Tensor,
    config: PixelRLConfig,
) -> list[dict]:
    """No label fields inspected; frozen proposal scores/order are preserved exactly."""
    validate_cache(cache)
    result = copy.deepcopy(cache.get("base_predictions", []))
    boxes = cache["boxes"].clone()
    moved = torch.zeros(len(boxes), dtype=torch.bool)
    if len(cache["selected_indices"]):
        logits = policy(observation_features(cache, mean, scale))
        delta = (logits.softmax(-1) * SIDE_ACTIONS).sum(-1)
        delta = torch.where(delta.abs() < 1e-8, torch.zeros_like(delta), delta)
        selected = cache["selected_indices"]
        boxes[selected] += delta * offset_scales(boxes[selected], config)
        moved[selected] = (delta != 0).any(-1)
    frozen_records = bool(result)
    for i, (box, score) in enumerate(zip(boxes, cache["scores"], strict=True)):
        if frozen_records and not moved[i]:
            continue
        x1, y1, x2, y2 = map(float, box)
        row = {
            "image_id": int(cache["image_ids"][i]),
            "category_id": int(cache["category_ids"][i]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        }
        if frozen_records:
            # No confidence arithmetic, even when the score differs below float32 precision.
            result[i]["bbox"] = row["bbox"]
        else:
            result.append(row)
    return result


def _annotation_identity(cache: dict) -> str:
    path = Path(cache["metadata"]["annotation_path"])
    digest = _sha(path)
    if cache["metadata"].get("annotation_sha256") != digest:
        raise ValueError("native annotations differ from the frozen pixel cache")
    doc = json.loads(path.read_text())
    category = int(cache["category_ids"][0])
    by_image = {row["id"]: [] for row in cache["images"]}
    for row in doc["annotations"]:
        if row["image_id"] in by_image and row["category_id"] == category:
            if row.get("iscrowd", 0):
                raise ValueError("pixel-side reward requires non-crowd annotations")
            x, y, w, h = row["bbox"]
            by_image[row["image_id"]].append([x, y, x + w, y + h])
    for image, target in zip(cache["images"], cache["targets"], strict=True):
        native = torch.tensor(by_image[image["id"]], dtype=torch.float64).reshape(-1, 4)
        cached = target.double()
        if native.shape != cached.shape or not torch.allclose(native, cached, atol=1e-4, rtol=1e-6):
            raise ValueError("native labels differ from cached target coordinates")
    return digest


def _evaluate(cache: dict, predictions: list[dict], evaluator: Callable | None):
    if evaluator is not None:
        return evaluator(predictions)
    from vision_rl.detection_eval import _cached_ap, _prepared_cache, evaluate_coco

    path = Path(cache["metadata"]["annotation_path"])
    if _sha(path) != cache["metadata"]["annotation_sha256"]:
        raise ValueError("native validation annotation file changed")
    result = evaluate_coco(
        path,
        predictions,
        image_ids=[row["id"] for row in cache["images"]],
        model={"name": "native-pixel-side-RL"},
    )
    if result["source_annotation_sha256"] != cache["metadata"]["annotation_sha256"]:
        raise ValueError("native annotations changed during validation")
    # Reuse native COCO matches for exact dataset AP95, without another IoU pass.
    prepared = _prepared_cache(result["match_cache"])
    prepared["iou_thresholds"] = prepared["iou_thresholds"][-1:]
    for image in prepared["items"].values():
        for category in image.values():
            category["matches"] = category["matches"][-1:]
            category["ignored_detections"] = category["ignored_detections"][-1:]
    result["AP95"] = _cached_ap(prepared, result["image_ids"])
    return result


def _payload(policy, mean, scale, config, provenance) -> dict:
    return {
        "version": 1,
        "state": {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()},
        "mean": mean,
        "scale": scale,
        "feature_dim": policy.layers[0].in_features,
        "config": asdict(config),
        "provenance": provenance,
    }


def export_policy(payload: dict, path: Path) -> None:
    """Numeric-only, bounded architecture; source artifacts are hash-bound."""
    from safetensors.torch import save_file

    tensors = {"policy." + key: value.contiguous() for key, value in payload["state"].items()}
    tensors.update(
        {
            "feature_mean": payload["mean"].contiguous(),
            "feature_scale": payload["scale"].contiguous(),
        }
    )
    metadata = {
        "format": "native-pixel-side-RL-v1",
        "config": payload["config"],
        "feature_dim": payload["feature_dim"],
        "provenance": payload["provenance"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"vision_rl": json.dumps(metadata, sort_keys=True)})
    os.replace(temporary, path)


def load_policy(path: Path, *, expected_sha256: str | None = None):
    from safetensors.torch import load

    from vision_rl.crop_strips import _safe_bytes

    data = _safe_bytes(path)
    if len(data) > 8 * 1024**2:
        raise ValueError("pixel policy must be a regular safetensors file below 8 MiB")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("pixel policy SHA differs from frozen composition")
    header_size = int.from_bytes(data[:8], "little")
    if not 2 <= header_size <= min(1024**2, len(data) - 8):
        raise ValueError("invalid pixel policy header size")
    header = json.loads(data[8 : 8 + header_size])
    metadata = json.loads(header["__metadata__"]["vision_rl"])
    if metadata.get("format") != "native-pixel-side-RL-v1":
        raise ValueError("unknown pixel-side policy format")
    config = PixelRLConfig(**metadata["config"])
    config.validate()
    dim = metadata["feature_dim"]
    if type(dim) is not int or not 49 <= dim <= 2095:
        raise ValueError("invalid pixel-side feature dimension")
    state = load(data)
    for value in state.values():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("pixel-side policy tensors must be finite float32")
    feature_width = (dim - 47) // 2
    if dim != feature_width * 2 + 47:
        raise ValueError("pixel-side feature dimension is inconsistent")
    if state.get("feature_mean", torch.empty(0)).shape != (feature_width,):
        raise ValueError("pixel feature mean has an invalid shape")
    if (
        state.get("feature_scale", torch.empty(0)).shape != (feature_width,)
        or (state["feature_scale"] < 0.05).any()
    ):
        raise ValueError("pixel feature scale has an invalid shape or range")
    policy = PixelSidePolicy(dim, config)
    model_state = {
        key.removeprefix("policy."): value
        for key, value in state.items()
        if key.startswith("policy.")
    }
    if set(state) != {"policy." + key for key in policy.state_dict()} | {
        "feature_mean",
        "feature_scale",
    }:
        raise ValueError("unexpected pixel-side policy tensors")
    if not torch.equal(model_state["reference_logprob"], policy.reference_logprob):
        raise ValueError("pixel-side reference prior differs from trusted construction")
    policy.load_state_dict(model_state, strict=True)
    metadata["checkpoint_sha256"] = digest
    return policy.eval(), state["feature_mean"], state["feature_scale"], config, metadata


def predict_cache(cache: dict | Path, checkpoint: Path) -> list[dict]:
    cache = load_cache(cache) if isinstance(cache, Path) else cache
    validate_cache(cache)
    policy, mean, scale, config, metadata = load_policy(checkpoint)
    for key in (
        "detector_sha256",
        "supervised_sha256",
        "proposal_pipeline_sha256",
        "supervised_composition_sha256",
        "observation_source_sha256",
    ):
        if cache["metadata"].get(key) != metadata["provenance"].get(key):
            raise ValueError(f"pixel-side policy/cache {key} differs")
    return predictions_from_policy(cache, policy, mean, scale, config)


def refine_manifest(
    checkpoint: Path,
    supervised_checkpoint: Path,
    manifest: Path,
    image_root: Path,
    proposals: Path,
    output: Path,
    *,
    detector_sha256: str,
    proposal_pipeline_sha256: str,
    composition: dict | None = None,
    expected_sha256: str | None = None,
    cpu_threads: int = 1,
) -> dict:
    """Strict image-only SFT + pixel RL replay; no annotation path is accepted.

    ``composition`` freezes the supervised stage exactly as crop_strips does.
    The deployment descriptor supplies all model hashes and proposal recipe SHA.
    """
    started = time.perf_counter()
    _, _, _, _, policy_info = load_policy(checkpoint, expected_sha256=expected_sha256)
    if policy_info["provenance"].get("supervised_sha256") != _sha(supervised_checkpoint):
        raise ValueError("supervised checkpoint differs from policy training source")
    cache = export_pixel_cache(
        supervised_checkpoint,
        manifest,
        image_root,
        proposals,
        detector_sha256=detector_sha256,
        proposal_pipeline_sha256=proposal_pipeline_sha256,
        composition=composition,
        cpu_threads=cpu_threads,
    )
    predictions = predict_cache(cache, checkpoint)
    _write(output, predictions)
    report = {
        "format": "native-pixel-side-RL-predictions-v1",
        "images": len(cache["images"]),
        "proposals": len(predictions),
        "selected_pixel_regions": len(cache["selected_indices"]),
        "checkpoint_sha256": policy_info["checkpoint_sha256"],
        **cache["metadata"],
        "quality_head_frozen": True,
        "proposal_set_and_scores_frozen": True,
        "labels_available_to_predictor": False,
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output.with_suffix(".report.json"), report)
    return report


def train(
    train_cache: dict | Path,
    validation_cache: dict | Path,
    output: Path,
    config: PixelRLConfig | None = None,
    *,
    resume: bool = False,
    evaluator: Callable | None = None,
) -> dict:
    """Finite image-reward training with a diagnostic go/no-go gate and exact resume."""
    train_cache = load_cache(train_cache, "train") if isinstance(train_cache, Path) else train_cache
    validation_cache = (
        load_cache(validation_cache, "validation")
        if isinstance(validation_cache, Path)
        else validation_cache
    )
    _training_invariants(train_cache, validation_cache)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".train.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another pixel RL trainer owns this output") from None
        return _train(
            train_cache, validation_cache, output, config, resume=resume, evaluator=evaluator
        )


def _train(train_cache, validation_cache, output, config=None, *, resume=False, evaluator=None):
    config = config or PixelRLConfig()
    config.validate()
    train_cache = load_cache(train_cache, "train") if isinstance(train_cache, Path) else train_cache
    validation_cache = (
        load_cache(validation_cache, "validation")
        if isinstance(validation_cache, Path)
        else validation_cache
    )
    _training_invariants(train_cache, validation_cache)
    if evaluator is None:
        _annotation_identity(train_cache)
        _annotation_identity(validation_cache)
    if output.exists() and any(p.name != ".train.lock" for p in output.iterdir()) and not resume:
        raise ValueError("pixel RL output must be fresh or explicitly resumed")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    mean = train_cache["side_features"].mean((0, 1))
    scale = train_cache["side_features"].std((0, 1), unbiased=False).clamp_min(0.05)
    features = observation_features(train_cache, mean, scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        policy = PixelSidePolicy(features.shape[-1], config)
    generator = torch.Generator().manual_seed(config.seed + 1)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    provenance = {
        "train_cache_sha256": cache_fingerprint(train_cache),
        "validation_cache_sha256": cache_fingerprint(validation_cache),
        "detector_sha256": train_cache["metadata"]["detector_sha256"],
        "supervised_sha256": train_cache["metadata"]["supervised_sha256"],
        "proposal_pipeline_sha256": train_cache["metadata"].get("proposal_pipeline_sha256"),
        "supervised_composition_sha256": train_cache["metadata"].get(
            "supervised_composition_sha256"
        ),
        "observation_source_sha256": train_cache["metadata"].get("observation_source_sha256"),
        "source_sha256": _sha(Path(__file__)),
        "reward_source_sha256": {
            name: _sha(Path(__file__).with_name(name))
            for name in ("detection_rl.py", "detection_rewards.py")
        },
        "train_images": len(train_cache["images"]),
        "validation_images": len(validation_cache["images"]),
        "selected_pixel_regions": len(features),
        "native_test_accessed": False,
        "quality_head_frozen": True,
        "proposal_set_and_scores_frozen": True,
    }
    tables = prepare_rewards(train_cache, config)
    pilot = exploration_pilot(train_cache, tables, config)
    if not resume:
        _write(output / "plan.json", {"config": asdict(config), "provenance": provenance})
        _write(output / "train-headroom.json", pilot)
    if not pilot["go"]:
        result = {
            "status": "no_go_action_space",
            "pilot": pilot,
            "epochs": 0,
            "provenance": provenance,
        }
        _write(output / "summary.json", result)
        return result
    started = time.perf_counter()
    history = []
    epoch_start, best_ap, best_rl_ap = 0, -1.0, -1.0
    best_payload = best_rl_payload = None
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if state["config"] != asdict(config) or state["provenance"] != provenance:
            raise ValueError("pixel RL resume requires identical data, source, and configuration")
        policy.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["rng"])
        history, epoch_start = state["history"], state["epoch"]
        best_ap, best_rl_ap = state["best_ap"], state["best_rl_ap"]
        best_payload, best_rl_payload = state["best_payload"], state["best_rl_payload"]
        for name, payload in (("best", best_payload), ("best-rl", best_rl_payload)):
            if payload is not None:
                export_policy(payload, output / f"{name}.safetensors")
            else:
                (output / f"{name}.safetensors").unlink(missing_ok=True)
    else:
        baseline_predictions = predictions_from_policy(
            validation_cache, policy, mean, scale, config
        )
        baseline = _evaluate(validation_cache, baseline_predictions, evaluator)
        best_ap = float(baseline["AP"])
        _write(output / "sft-evaluation.json", baseline)
        _write(output / "sft-predictions.coco.json", baseline_predictions)
        best_payload = _payload(policy, mean, scale, config, provenance)
        export_policy(best_payload, output / "sft.safetensors")
        export_policy(best_payload, output / "best.safetensors")
        _save(
            output / "last.pt",
            {
                **best_payload,
                "optimizer": optimizer.state_dict(),
                "rng": generator.get_state(),
                "epoch": 0,
                "history": [],
                "best_ap": best_ap,
                "best_rl_ap": best_rl_ap,
                "best_payload": best_payload,
                "best_rl_payload": None,
            },
        )
    eligible = [i for i, table in enumerate(tables) if len(table.selected_rows)]
    for epoch in range(epoch_start, config.epochs):
        ordering = torch.randperm(len(eligible), generator=generator).tolist()
        totals: dict[str, float] = {}
        count = 0
        for start in range(0, len(ordering), config.batch_images):
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for position in ordering[start : start + config.batch_images]:
                table = tables[eligible[position]]
                logits = policy(features[table.selected_rows])
                actions = sample_actions(logits, config.groups, generator)
                rewards, stats = table.rewards(actions, config.recall_bonus)
                loss, metrics = policy_gradient(
                    logits, actions, rewards, policy.reference_logprob, config
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite pixel-side RL objective")
                losses.append(loss)
                metrics.update(
                    {
                        "sampled_AP": float(stats["AP"].mean()),
                        "sampled_AP95": float(stats["AP95"].mean()),
                    }
                )
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0) + value
                count += 1
            torch.stack(losses).mean().backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite pixel-side RL gradients")
            optimizer.step()
        predictions = predictions_from_policy(validation_cache, policy, mean, scale, config)
        evaluation = _evaluate(validation_cache, predictions, evaluator)
        ap = float(evaluation["AP"])
        if not math.isfinite(ap) or not 0 <= ap <= 1:
            raise ValueError("native AP must be a finite fraction in [0,1]")
        payload = _payload(policy, mean, scale, config, provenance)
        improved = ap > best_ap
        if improved:
            best_ap, best_payload = ap, payload
        if ap > best_rl_ap:
            best_rl_ap, best_rl_payload = ap, payload
        record = {
            "epoch": epoch + 1,
            "validation_AP": ap,
            "best_validation_AP": best_ap,
            "selected": improved,
            "validation_AP95": evaluation.get("AP95"),
            **{key: value / count for key, value in totals.items()},
        }
        history.append(record)
        # Commit all state first. Artifact writes are idempotent materializations
        # of this single durable checkpoint, including after interrupted writes.
        _save(
            output / "last.pt",
            {
                **payload,
                "optimizer": optimizer.state_dict(),
                "rng": generator.get_state(),
                "epoch": epoch + 1,
                "history": history,
                "best_ap": best_ap,
                "best_rl_ap": best_rl_ap,
                "best_payload": best_payload,
                "best_rl_payload": best_rl_payload,
            },
        )
        export_policy(best_payload, output / "best.safetensors")
        export_policy(best_rl_payload, output / "best-rl.safetensors")
        _write(output / "history.json", history)
        print(json.dumps(record), flush=True)
    baseline_report = json.loads((output / "sft-evaluation.json").read_text())
    baseline_ap = baseline_report["AP"]
    replays = {}
    for name in ("best", "best-rl"):
        predictions = predict_cache(validation_cache, output / f"{name}.safetensors")
        report = _evaluate(validation_cache, predictions, evaluator)
        if abs(report["AP"] - (best_ap if name == "best" else best_rl_ap)) > 1e-12:
            raise ValueError("selected safe pixel policy failed independent validation replay")
        replays[name] = report
        _write(output / f"{name}-evaluation.json", report)
        _write(output / f"{name}-predictions.coco.json", predictions)
    summary = {
        "status": "completed",
        "epochs": config.epochs,
        "sft_validation_AP": baseline_ap,
        "best_validation_AP": best_ap,
        "best_rl_validation_AP": best_rl_ap,
        "RL_AP_gain": best_rl_ap - baseline_ap,
        "sft_validation_AP95": baseline_report.get("AP95"),
        "best_rl_validation_AP95": replays["best-rl"].get("AP95"),
        "meets_predeclared_validation_gain": best_rl_ap - baseline_ap
        >= config.minimum_validation_gain,
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in policy.parameters()),
        "provenance": provenance,
    }
    _write(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--train-cache", required=True, type=Path)
    training.add_argument("--validation-cache", required=True, type=Path)
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--config", type=Path)
    training.add_argument("--resume", action="store_true")
    prediction = commands.add_parser("predict-cache")
    prediction.add_argument("--cache", required=True, type=Path)
    prediction.add_argument("--checkpoint", required=True, type=Path)
    prediction.add_argument("--output", required=True, type=Path)
    export = commands.add_parser("export-cache")
    for name in ("checkpoint", "manifest", "image-root", "proposals", "output"):
        export.add_argument(f"--{name}", required=True, type=Path)
    export.add_argument(
        "--split", choices=("train", "validation", "inference"), default="inference"
    )
    export.add_argument("--annotation-path", type=Path)
    export.add_argument("--detector-sha256", required=True)
    export.add_argument("--proposal-pipeline-sha256", required=True)
    export.add_argument("--composition", type=Path)
    export.add_argument("--cpu-threads", type=int, default=1)
    image_prediction = commands.add_parser("predict")
    for name in (
        "checkpoint",
        "supervised-checkpoint",
        "manifest",
        "image-root",
        "proposals",
        "output",
    ):
        image_prediction.add_argument(f"--{name}", required=True, type=Path)
    image_prediction.add_argument("--detector-sha256", required=True)
    image_prediction.add_argument("--proposal-pipeline-sha256", required=True)
    image_prediction.add_argument("--composition", type=Path)
    image_prediction.add_argument("--expected-sha256")
    image_prediction.add_argument("--cpu-threads", type=int, default=1)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    if command == "train":
        path = arguments.pop("config")
        config = PixelRLConfig(**json.loads(path.read_text())) if path else PixelRLConfig()
        print(json.dumps(train(config=config, **arguments), indent=2))
    elif command == "predict-cache":
        _write(arguments["output"], predict_cache(arguments["cache"], arguments["checkpoint"]))
    else:
        path = arguments.pop("composition")
        composition = json.loads(path.read_text()) if path else None
        if command == "export-cache":
            cache = export_pixel_cache(composition=composition, **arguments)
            print(json.dumps({"images": len(cache["images"]), "metadata": cache["metadata"]}))
        else:
            print(json.dumps(refine_manifest(composition=composition, **arguments)))


if __name__ == "__main__":
    main()
