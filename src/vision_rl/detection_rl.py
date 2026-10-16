"""Group-relative reinforcement learning of a frozen detector's query refinements.

The detector is trained separately. This module consumes frozen decoder features,
never images or a model API. The 81-way geometry action changes a box's center and
log width/height by bounded amounts. A contextual policy also learns each query's
keep probability, used as its detection confidence at inference. There is no
score threshold at evaluation: the native evaluator determines the ranked AP.

Rewards use actual unique, score-ordered matching at all ten COCO IoU thresholds.
A matched retained query earns +3, an unmatched retained query -1, and a dropped
query 0; the resulting team utility is (3*TP-FP)/(3*number_of_targets).
An image AP bonus provides ranking/recall credit. All queries receive this team
utility so the estimator includes the effects of one query displacing another.
A small separable IoU shaping term improves local box credit without pretending
that independently credited greedy matches are an unbiased AP gradient. The
baseline is the other independent samples' mean, not
the sample's own reward. Box actions for dropped queries are marginalized out.
Supervised oracle-action warmup is recorded separately from RL, including its AP.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from vision_rl.detection_rewards import interpolated_ap

SCHEMA_VERSION = 1
IDENTITY_ACTION = 40
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


@dataclass(frozen=True)
class RefinementConfig:
    hidden: int = 128
    groups: int = 8
    keep_sampling_temperature: float = 1.0
    warmup_epochs: int = 8
    rl_epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 3e-4
    rl_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    entropy_weight: float = 0.0001
    kl_weight: float = 0.02
    ap_bonus: float = 1.0
    iou_shaping: float = 0.05
    geometry_step: float = 0.03
    geometry_inference: str = "argmax"
    identity_probability: float = 0.75
    seed: int = 17
    device: str = "cpu"
    cpu_threads: int = 1
    gradient_clip: float = 1.0

    def validate(self) -> None:
        if self.geometry_inference not in {"argmax", "mean"}:
            raise ValueError("geometry_inference must be argmax or mean")
        if not 8 <= self.hidden <= 1024:
            raise ValueError("hidden must be between 8 and 1024")
        if not 0.1 <= self.keep_sampling_temperature <= 2.0:
            raise ValueError("keep_sampling_temperature must be between .1 and 2")
        if not 2 <= self.groups <= 128:
            raise ValueError("groups must be between 2 and 128")
        if not 1 <= self.cpu_threads <= 32:
            raise ValueError("cpu_threads must be between 1 and 32")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        if self.warmup_epochs < 0 or self.rl_epochs < 1:
            raise ValueError("nonnegative warmup and positive RL epochs required")
        if not 0 < self.geometry_step <= 0.5 or not 0 < self.identity_probability < 1:
            raise ValueError("invalid bounded action/reference distribution")
        if (
            not 0 < self.learning_rate <= 0.1
            or not 0 < self.rl_learning_rate <= 0.1
            or self.gradient_clip <= 0
        ):
            raise ValueError("invalid learning rate or gradient clip")
        for name in ("weight_decay", "entropy_weight", "kl_weight", "ap_bonus", "iou_shaping"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")


def box_iou(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Broadcasted IoUs, with leading dimensions preserved."""
    lt = torch.maximum(boxes[..., :, None, :2], targets[..., None, :, :2])
    rb = torch.minimum(boxes[..., :, None, 2:], targets[..., None, :, 2:])
    intersection = (rb - lt).clamp_min(0).prod(-1)
    area_a = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
    area_b = (targets[..., 2:] - targets[..., :2]).clamp_min(0).prod(-1)
    union = area_a[..., :, None] + area_b[..., None, :] - intersection
    return intersection / union.clamp_min(1e-12)


def action_table(step: float = 0.1) -> torch.Tensor:
    """dx/width, dy/height, log(width multiplier), log(height multiplier)."""
    return torch.tensor(list(itertools.product((-step, 0.0, step), repeat=4)))


def refine_boxes(boxes: torch.Tensor, actions: torch.Tensor, step: float = 0.1) -> torch.Tensor:
    deltas = action_table(step).to(boxes.device)[actions]
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-6)
    center = center + deltas[..., :2] * size
    size = size * deltas[..., 2:].exp()
    refined = torch.cat((center - size / 2, center + size / 2), -1)
    return torch.where((actions == IDENTITY_ACTION)[..., None], boxes, refined)


def mean_refine_boxes(
    boxes: torch.Tensor, geometry_logits: torch.Tensor, step: float = 0.03
) -> torch.Tensor:
    """Deterministic mean center/log-scale action, still within the action bounds."""
    delta = geometry_logits.softmax(-1) @ action_table(step).to(geometry_logits.device)
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-6)
    center = center + delta[..., :2] * size
    size = size * delta[..., 2:].exp()
    refined = torch.cat((center - size / 2, center + size / 2), -1)
    # Symmetric reference probabilities have exactly zero mathematical mean;
    # suppress floating-point cancellation noise to preserve the identity model.
    return torch.where((delta.abs().amax(-1) <= 1e-8)[..., None], boxes, refined)


def _finite_float(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_cache(cache: dict, *, expected_split: str | None = None) -> None:
    """Inference validation intentionally does not read the targets field."""
    if cache.get("version", cache.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError("unsupported detector cache version")
    if expected_split is not None and cache.get("split") != expected_split:
        raise ValueError(f"expected {expected_split} cache; got {cache.get('split')}")
    boxes, logits, embeddings = (cache[key] for key in ("boxes", "logits", "embeddings"))
    for name, tensor in (("boxes", boxes), ("logits", logits), ("embeddings", embeddings)):
        _finite_float(tensor, name)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [images, queries, 4]")
    n, q, _ = boxes.shape
    if not 1 <= n <= 100000 or not 1 <= q <= 1000:
        raise ValueError("cache image/query dimensions exceed bounds")
    if logits.shape != (n, q) or embeddings.ndim != 3 or embeddings.shape[:2] != (n, q):
        raise ValueError("query features and logits must align with boxes")
    if not 1 <= embeddings.shape[-1] <= 4096:
        raise ValueError("decoder embedding dimension exceeds bounds")
    if (boxes < -1).any() or (boxes > 2).any() or (boxes[..., 2:] < boxes[..., :2]).any():
        raise ValueError("boxes must be valid xyxy in normalized image coordinates")
    ids = cache.get("ids", [])
    if len(ids) != n or len(set(map(str, ids))) != n:
        raise ValueError("cache IDs must be unique and align with images")
    sizes = torch.as_tensor(cache.get("image_sizes", []))
    if sizes.shape != (n, 2) or not torch.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("image_sizes must be positive [width, height]")
    if expected_split is not None:
        targets = cache.get("targets", [])
        if len(targets) != n:
            raise ValueError("training/validation targets must align with images")
        for target in targets:
            _finite_float(target, "targets")
            if target.ndim != 2 or target.shape[-1] != 4:
                raise ValueError("targets must each have shape [objects, 4]")
            if (target < 0).any() or (target > 1).any() or (target[:, 2:] <= target[:, :2]).any():
                raise ValueError("targets must be nonempty normalized xyxy boxes")


def load_cache(path: Path, *, expected_split: str | None = None) -> dict:
    if path.stat().st_size > 2 * 1024**3:
        raise ValueError("detector cache exceeds 2 GiB limit")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError("cache must be a tensor dictionary")
    validate_cache(cache, expected_split=expected_split)
    return cache


def _digest(cache: dict) -> str:
    """Content identity includes observations and labels, independent of cache paths."""
    digest = hashlib.sha256()
    for key in ("split", "ids", "coco_image_ids", "metadata"):
        value = cache.get(key)
        if key == "metadata":
            value = {k: v for k, v in (value or {}).items() if k != "annotation_path"}
        digest.update(json.dumps(value, sort_keys=True).encode())
    for name in ("boxes", "logits", "embeddings", "image_sizes"):
        value = torch.as_tensor(cache[name]).cpu().contiguous()
        digest.update(str((name, value.shape, value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    for target in cache.get("targets", []):
        digest.update(target.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_isolated(train: dict, validation: dict) -> None:
    validate_cache(train, expected_split="train")
    validate_cache(validation, expected_split="validation")
    if set(map(str, train["ids"])) & set(map(str, validation["ids"])):
        raise ValueError("training and validation IDs overlap")
    for key in ("pixel_sha256", "image_sha256"):
        if set(train.get(key, [])) & set(validation.get(key, [])):
            raise ValueError(f"training and validation {key} overlap")
    train_model = train.get("metadata", {}).get("model_sha256")
    val_model = validation.get("metadata", {}).get("model_sha256")
    if not train_model or train_model != val_model:
        raise ValueError("caches must identify the same frozen detector model_sha256")
    if train["embeddings"].shape[-1] != validation["embeddings"].shape[-1]:
        raise ValueError("cache embedding dimensions differ")


def make_features(cache: dict, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    embeddings = ((cache["embeddings"].float() - mean) / scale).clamp(-10, 10)
    boxes, logits = cache["boxes"].float(), cache["logits"].float()
    scores = logits.sigmoid()
    context = (embeddings * scores[..., None]).sum(1) / scores.sum(1).clamp_min(1e-6)[:, None]
    context = context[:, None].expand_as(embeddings)
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    sizes = boxes[..., 2:] - boxes[..., :2]
    q = boxes.shape[1]
    overlaps = box_iou(boxes, boxes)
    eye = torch.eye(q, dtype=torch.bool)[None]
    overlaps = overlaps.masked_fill(eye, 0)
    # A strictly higher scored neighbour exposes duplicate proposals without targets.
    higher = logits[:, None, :] > logits[:, :, None]
    max_higher = overlaps.masked_fill(~higher, 0).amax(-1)
    density = (overlaps > 0.5).float().sum(-1) / max(q - 1, 1)
    ranks = logits.argsort(dim=-1, descending=True, stable=True).argsort(-1).float() / max(q - 1, 1)
    geometry = torch.cat(
        (
            boxes,
            centers,
            sizes,
            logits.clamp(-10, 10)[..., None] / 10,
            ranks[..., None],
            max_higher[..., None],
            density[..., None],
        ),
        -1,
    )
    return torch.cat((embeddings, context, geometry), -1)


class QueryRefinementPolicy(nn.Module):
    def __init__(self, feature_dim: int, config: RefinementConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, 82),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        prior = torch.full((81,), (1 - config.identity_probability) / 80)
        prior[IDENTITY_ACTION] = config.identity_probability
        self.register_buffer("reference_geometry_logprob", prior.log())

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor):
        residual = self.network(features)
        geometry_logits = residual[..., :81] + self.reference_geometry_logprob
        keep_logits = base_logits + 5 * residual[..., 81].tanh()
        return geometry_logits, keep_logits


@torch.no_grad()
def matching_rewards(
    boxes: torch.Tensor,
    keep: torch.Tensor,
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    ap_bonus: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Exact greedy unique matching for [groups, queries] sampled detections.

    Matching uses CPU tensors deliberately: short sequential greedy steps are much
    faster there than hundreds of small MPS dispatches. All groups and all ten IoU
    thresholds are vectorized inside each step. Stable ranking breaks score ties.
    """
    if boxes.ndim != 3 or keep.shape != boxes.shape[:2] or scores.shape != boxes.shape[1:2]:
        raise ValueError("reward expects boxes[G,Q,4], keep[G,Q], scores[Q]")
    groups, queries = keep.shape
    order = scores.argsort(descending=True, stable=True)
    sorted_keep = keep[:, order].bool()
    ious = box_iou(boxes[:, order], targets)
    thresholds = IOU_THRESHOLDS.to(boxes.device)
    hits = torch.zeros(groups, 10, queries, dtype=torch.bool, device=boxes.device)
    if len(targets):
        claimed = torch.zeros(groups, 10, len(targets), dtype=torch.bool, device=boxes.device)
        for query in range(queries):
            available = ious[:, query, None, :].expand(-1, 10, -1).masked_fill(claimed, -1)
            best_iou, best_target = available.max(-1)
            hit = (best_iou >= thresholds) & sorted_keep[:, query, None]
            hits[:, :, query] = hit
            claimed.scatter_(
                -1,
                best_target[..., None],
                claimed.gather(-1, best_target[..., None]) | hit[..., None],
            )
    tp = hits.float()
    ap = interpolated_ap(hits, sorted_keep, len(targets))
    local = (4 * tp - sorted_keep[:, None].float()).mean(1)
    team_utility = local.sum(-1) / (3 * max(len(targets), 1)) + ap_bonus * ap
    inverse = order.argsort()
    return team_utility[:, None].expand(-1, queries), {
        "AP": ap,
        "tp_fraction": tp.mean(1)[:, inverse],
        "kept": keep.float().sum(-1),
    }


def leave_one_out_advantage(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.shape[0] < 2:
        raise ValueError("at least two independent samples required")
    return (rewards * rewards.shape[0] - rewards.sum(0)) / (rewards.shape[0] - 1)


def reinforce_loss(
    geometry_logits: torch.Tensor,
    keep_logits: torch.Tensor,
    boxes: torch.Tensor,
    targets: list[torch.Tensor],
    generator: torch.Generator,
    config: RefinementConfig,
    reference_geometry_logits: torch.Tensor,
    reference_keep_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """An unbiased score-function estimator for the sampled matching utility.

    The reward and matching order are detached, as required for REINFORCE. Ranking
    is a fixed context within an update; the confidence head learns via sampled
    retain/drop decisions. The separately reported KL/entropy terms are regularizers.
    """
    batch, queries, actions_count = geometry_logits.shape
    probabilities = geometry_logits.detach().softmax(-1).cpu()
    sampled = (
        torch.multinomial(
            probabilities.reshape(-1, actions_count),
            config.groups,
            replacement=True,
            generator=generator,
        )
        .reshape(batch, queries, config.groups)
        .permute(0, 2, 1)
    )
    behavior_keep_logits = keep_logits / config.keep_sampling_temperature
    reference_behavior_logits = reference_keep_logits / config.keep_sampling_temperature
    keep_probability = behavior_keep_logits.detach().sigmoid().cpu()
    ranking_scores = keep_logits.detach().sigmoid().cpu()
    retained = (
        torch.rand((batch, config.groups, queries), generator=generator)
        < (keep_probability[:, None])
    )
    advantages, aps = [], []
    for index in range(batch):
        selected_boxes = refine_boxes(
            boxes[index].cpu()[None], sampled[index], config.geometry_step
        )
        reward, stats = matching_rewards(
            selected_boxes,
            retained[index],
            ranking_scores[index],
            targets[index].cpu(),
            ap_bonus=config.ap_bonus,
        )
        # IoU shaping is separable by query: cross-query score-function terms
        # have expectation zero, so local credit is valid for this term only.
        if len(targets[index]):
            local_iou = box_iou(selected_boxes, targets[index].cpu()).amax(-1)
            reward = reward + config.iou_shaping * retained[index] * local_iou
        advantages.append(leave_one_out_advantage(reward))
        aps.append(stats["AP"].mean())
    advantage = torch.stack(advantages).to(geometry_logits.device)
    chosen_actions = sampled.to(geometry_logits.device)
    retained_float = retained.to(geometry_logits.device, dtype=keep_logits.dtype)
    action_logprob = (
        geometry_logits.log_softmax(-1)[:, None]
        .expand(-1, config.groups, -1, -1)
        .gather(-1, chosen_actions[..., None])
        .squeeze(-1)
    )
    keep_logprob = F.logsigmoid(behavior_keep_logits)[:, None] * retained_float + F.logsigmoid(
        -behavior_keep_logits
    )[:, None] * (1 - retained_float)
    policy_loss = -(advantage * (action_logprob * retained_float + keep_logprob)).sum(-1).mean()
    logprob = geometry_logits.log_softmax(-1)
    prob = logprob.exp()
    keep_prob = behavior_keep_logits.sigmoid()
    entropy = (
        -(prob * logprob).sum(-1).mean()
        + (
            -keep_prob * F.logsigmoid(behavior_keep_logits)
            - (1 - keep_prob) * F.logsigmoid(-behavior_keep_logits)
        ).mean()
    )
    geometry_kl = (prob * (logprob - reference_geometry_logits.log_softmax(-1))).sum(-1)
    keep_kl = keep_prob * (
        F.logsigmoid(behavior_keep_logits) - F.logsigmoid(reference_behavior_logits)
    ) + (1 - keep_prob) * (
        F.logsigmoid(-behavior_keep_logits) - F.logsigmoid(-reference_behavior_logits)
    )
    kl = (geometry_kl + keep_kl).mean()
    # The sampled trajectory contains Q query actions. Its log probability is a
    # SUM, so its KL and entropy must also be SUMS. Averaging only the penalties
    # accidentally weakens the trust region by Q (128 in the actual detector).
    loss = policy_loss + queries * (-config.entropy_weight * entropy + config.kl_weight * kl)
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "entropy": float(entropy.detach()),
        "kl": float(kl.detach()),
        "sampled_image_AP": float(torch.stack(aps).mean()),
        "advantage_std": float(advantage.std().detach()),
        "mean_retention_probability": float(keep_probability.mean()),
    }


@torch.no_grad()
def oracle_actions(cache: dict, config: RefinementConfig) -> tuple[torch.Tensor, ...]:
    """Train-only action labels for a separately measured supervised warmup."""
    action_labels, keep_labels, weights = [], [], []
    for boxes, logits, targets in zip(
        cache["boxes"], cache["logits"], cache["targets"], strict=True
    ):
        q = len(boxes)
        candidates = refine_boxes(boxes[:, None], torch.arange(81)[None], config.geometry_step)
        if len(targets):
            quality = box_iou(candidates, targets).amax(-1)
            best_quality, best_action = quality.max(-1)
            # Prefer identity on exact ties, reducing gratuitous box motion.
            identity_best = quality[:, IDENTITY_ACTION] >= best_quality - 1e-7
            best_action[identity_best] = IDENTITY_ACTION
            oracle_boxes = candidates[torch.arange(q), best_action]
            _, stats = matching_rewards(
                oracle_boxes[None],
                torch.ones(1, q, dtype=torch.bool),
                logits.sigmoid(),
                targets,
                ap_bonus=0,
            )
            soft_keep = stats["tp_fraction"][0]
        else:
            best_action = torch.full((q,), IDENTITY_ACTION)
            best_quality, soft_keep = torch.zeros(q), torch.zeros(q)
        action_labels.append(best_action)
        keep_labels.append(soft_keep)
        weights.append(best_quality * (best_quality > 0.25))
    return torch.stack(action_labels), torch.stack(keep_labels), torch.stack(weights)


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _policy_payload(policy, mean, scale, config, provenance) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "feature_dim": int(policy.network[0].in_features),
        "state_dict": {
            key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
        },
        "feature_mean": mean,
        "feature_scale": scale,
        "provenance": provenance,
    }


def load_policy(path: Path, device: str = "cpu") -> tuple[QueryRefinementPolicy, dict]:
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("refinement policy exceeds 64 MiB")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported refinement policy")
    config = RefinementConfig(**payload["config"])
    config.validate()
    feature_dim = payload.get("feature_dim")
    if not isinstance(feature_dim, int) or not 14 <= feature_dim <= 8204:
        raise ValueError("invalid policy feature dimension")
    for value in payload["state_dict"].values():
        _finite_float(value, "policy weights")
    policy = QueryRefinementPolicy(feature_dim, config)
    policy.load_state_dict(payload["state_dict"], strict=True)
    return policy.to(device).eval(), payload


@torch.no_grad()
def _predict(policy, cache, mean, scale, config, device) -> list[dict]:
    features = make_features(cache, mean, scale)
    results = []
    for start in range(0, len(cache["ids"]), config.batch_size):
        stop = start + config.batch_size
        geometry, keep_logits = policy(
            features[start:stop].to(device), cache["logits"][start:stop].to(device)
        )
        actions = geometry.argmax(-1).cpu()
        if config.geometry_inference == "mean":
            boxes = mean_refine_boxes(
                cache["boxes"][start:stop], geometry.cpu(), config.geometry_step
            )
        else:
            boxes = refine_boxes(cache["boxes"][start:stop], actions, config.geometry_step)
        for offset, (image_boxes, scores) in enumerate(zip(boxes, keep_logits.sigmoid().cpu())):
            results.append(
                {
                    "image_id": cache["ids"][start + offset],
                    "boxes": image_boxes,
                    "scores": scores,
                    "modal_actions": actions[offset],
                    "geometry_inference": config.geometry_inference,
                }
            )
    return results


def predict_refinement(cache: Path | dict, checkpoint: Path, *, device: str = "cpu") -> list[dict]:
    """Predict observations only; heldout targets are never inspected."""
    cache = load_cache(cache) if isinstance(cache, Path) else cache
    validate_cache(cache)
    policy, payload = load_policy(checkpoint, device)
    model_sha = cache.get("metadata", {}).get("model_sha256")
    if model_sha != payload["provenance"]["model_sha256"]:
        raise ValueError("refinement checkpoint belongs to a different frozen detector")
    return _predict(
        policy,
        cache,
        payload["feature_mean"],
        payload["feature_scale"],
        RefinementConfig(**payload["config"]),
        device,
    )


def export_refinement_safetensors(checkpoint: Path, output: Path) -> None:
    """Export the fixed Harbor architecture as bounded numeric tensors only."""
    from safetensors.torch import save_file

    policy, payload = load_policy(checkpoint)
    config = RefinementConfig(**payload["config"])
    if (config.hidden, config.geometry_step, config.identity_probability) != (128, 0.03, 0.75):
        raise ValueError(
            "Harbor export requires hidden=128, geometry_step=.03, identity_probability=.75"
        )
    if payload["feature_dim"] != 524:
        raise ValueError("Harbor export requires 256-dimensional detector query embeddings")
    model_sha = payload["provenance"]["model_sha256"]
    try:
        model_hash_bytes = bytes.fromhex(model_sha)
    except (ValueError, TypeError) as error:
        raise ValueError("model_sha256 must be 32 hexadecimal bytes") from error
    if len(model_hash_bytes) != 32:
        raise ValueError("model_sha256 must be 32 hexadecimal bytes")
    tensors = {
        "policy." + key: value.contiguous()
        for key, value in policy.state_dict().items()
        if key != "reference_geometry_logprob"
    }
    tensors.update(
        {
            "feature_mean": payload["feature_mean"].contiguous(),
            "feature_scale": payload["feature_scale"].contiguous(),
            "model_sha256": torch.tensor(list(model_hash_bytes), dtype=torch.uint8),
            "geometry_inference": torch.tensor(
                [int(config.geometry_inference == "mean")], dtype=torch.uint8
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, output)


def predict_refinement_safetensors(
    cache: Path | dict, checkpoint: Path, *, device: str = "cpu"
) -> list[dict]:
    """Verifier entry point: fixed architecture, no submitted config or pickle."""
    from safetensors.torch import load_file

    if checkpoint.stat().st_size > 8 * 1024**2:
        raise ValueError("safe refinement artifact exceeds 8 MiB")
    cache = load_cache(cache) if isinstance(cache, Path) else cache
    validate_cache(cache)
    if cache["embeddings"].shape[-1] != 256:
        raise ValueError("trusted query embeddings must have dimension 256")
    state = load_file(str(checkpoint), device="cpu")
    # Numeric mode extends the original fixed artifact without accepting code
    # or arbitrary configuration. Legacy artifacts have argmax semantics.
    state.setdefault("geometry_inference", torch.tensor([0], dtype=torch.uint8))
    mode = state["geometry_inference"]
    if mode.dtype != torch.uint8 or mode.shape != (1,) or int(mode[0]) not in (0, 1):
        raise ValueError("safe geometry inference mode must be uint8[1] containing 0 or 1")
    config = RefinementConfig(geometry_inference="mean" if int(mode[0]) else "argmax")
    policy = QueryRefinementPolicy(524, config)
    expected = {
        "policy." + key: value.shape
        for key, value in policy.state_dict().items()
        if key != "reference_geometry_logprob"
    }
    expected.update(
        {
            "feature_mean": (256,),
            "feature_scale": (256,),
            "model_sha256": (32,),
            "geometry_inference": (1,),
        }
    )
    if state.keys() != expected.keys():
        raise ValueError("safe refinement artifact contains unexpected or missing tensors")
    for key, shape in expected.items():
        value = state[key]
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"safe refinement tensor shape mismatch: {key}")
        if key in {"model_sha256", "geometry_inference"}:
            if value.dtype != torch.uint8:
                raise ValueError("model_sha256 tensor must contain uint8 bytes")
        else:
            _finite_float(value, key)
            if value.dtype != torch.float32:
                raise ValueError(f"safe refinement tensor must be float32: {key}")
    model_sha = bytes(state["model_sha256"].tolist()).hex()
    if model_sha != cache.get("metadata", {}).get("model_sha256"):
        raise ValueError("safe refinement artifact belongs to a different frozen detector")
    if (state["feature_scale"] < 0.05).any():
        raise ValueError("safe feature scale must be at least .05")
    model_state = {
        key.removeprefix("policy."): value
        for key, value in state.items()
        if key.startswith("policy.")
    }
    model_state["reference_geometry_logprob"] = policy.reference_geometry_logprob
    policy.load_state_dict(model_state, strict=True)
    policy.to(device).eval()
    return _predict(policy, cache, state["feature_mean"], state["feature_scale"], config, device)


def coco_predictions(cache: dict, predictions: list[dict]) -> list[dict]:
    ids = cache.get("coco_image_ids")
    if ids is None:
        raise ValueError("native evaluation requires coco_image_ids in cache")
    if len(predictions) != len(ids):
        raise ValueError("predictions do not cover every cached image")
    category_id = cache.get("metadata", {}).get("category_id", 1)
    records = []
    for index, prediction in enumerate(predictions):
        if str(prediction["image_id"]) != str(cache["ids"][index]):
            raise ValueError("prediction image order differs from cache")
        width, height = map(float, cache["image_sizes"][index])
        for box, score in zip(prediction["boxes"], prediction["scores"], strict=True):
            x1, y1, x2, y2 = map(float, box)
            records.append(
                {
                    "image_id": int(ids[index]),
                    "category_id": int(category_id),
                    "bbox": [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height],
                    "score": float(score),
                }
            )
    return records


def pin_native_annotations(cache: dict) -> dict:
    """Bind native validation labels to the immutable target tensors in the cache."""
    path = cache.get("metadata", {}).get("annotation_path")
    if not path:
        raise ValueError("native evaluation requires metadata.annotation_path")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    declared = cache.get("metadata", {}).get("annotation_sha256")
    if declared and declared != digest:
        raise ValueError("native annotation file changed since query export")
    document = json.loads(raw)
    images = {int(image["id"]): image for image in document["images"]}
    selected = set(cache["coco_image_ids"])
    category = cache.get("metadata", {}).get("category_id", 1)
    boxes_by_image = {image_id: [] for image_id in selected}
    for annotation in document["annotations"]:
        if annotation["image_id"] not in selected or annotation["category_id"] != category:
            continue
        if annotation.get("iscrowd", 0):
            raise ValueError("refinement training currently requires non-crowd annotations")
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        boxes_by_image[annotation["image_id"]].append(
            [
                x / image["width"],
                y / image["height"],
                (x + width) / image["width"],
                (y + height) / image["height"],
            ]
        )
    for index, image_id in enumerate(cache["coco_image_ids"]):
        image = images[image_id]
        if list(map(int, cache["image_sizes"][index])) != [image["width"], image["height"]]:
            raise ValueError("native image dimensions differ from cached observations")
        current = torch.tensor(sorted(boxes_by_image[image_id]), dtype=torch.float32).reshape(-1, 4)
        cached = torch.tensor(sorted(cache["targets"][index].tolist())).reshape(-1, 4)
        if current.shape != cached.shape or not torch.allclose(
            current, cached, atol=1e-6, rtol=1e-6
        ):
            raise ValueError("native annotation targets differ from frozen query cache")
    return {**cache, "_annotation_sha256": digest}


def _evaluate(cache: dict, predictions: list[dict], evaluator: Callable | None) -> dict:
    if evaluator is not None:
        return evaluator(predictions)
    from vision_rl.detection_eval import evaluate_coco

    annotation = cache.get("metadata", {}).get("annotation_path")
    if not annotation:
        raise ValueError("native evaluation requires metadata.annotation_path")
    expected_hash = cache.get("_annotation_sha256") or cache.get("metadata", {}).get(
        "annotation_sha256"
    )
    if expected_hash and hashlib.sha256(Path(annotation).read_bytes()).hexdigest() != expected_hash:
        raise ValueError("native annotations changed during refinement training")
    report = evaluate_coco(
        Path(annotation),
        coco_predictions(cache, predictions),
        image_ids=cache["coco_image_ids"],
        model={"name": "query-refinement-policy"},
    )
    if expected_hash and report["source_annotation_sha256"] != expected_hash:
        raise ValueError("native annotations changed during evaluation")
    return report


def train_refinement(
    train_cache: Path | dict,
    validation_cache: Path | dict,
    output: Path,
    config: RefinementConfig | None = None,
    *,
    resume: bool = False,
    evaluator: Callable | None = None,
) -> dict:
    """Train to a fixed epoch budget; select using validation AP, never test labels."""
    config = config or RefinementConfig()
    config.validate()
    train = (
        load_cache(train_cache, expected_split="train")
        if isinstance(train_cache, Path)
        else (train_cache)
    )
    validation = (
        load_cache(validation_cache, expected_split="validation")
        if isinstance(validation_cache, Path)
        else validation_cache
    )
    assert_isolated(train, validation)
    if evaluator is None:
        validation = pin_native_annotations(validation)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("output directory is nonempty; use resume or a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    device = config.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if device == "cpu":
        # Small cached-query MLPs suffer badly from oversubscription on laptop CPUs.
        torch.set_num_threads(config.cpu_threads)
    mean = train["embeddings"].float().mean((0, 1))
    scale = train["embeddings"].float().std((0, 1), unbiased=False).clamp_min(0.05)
    features = make_features(train, mean, scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        policy = QueryRefinementPolicy(features.shape[-1], config).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed + 1)
    provenance = {
        "train_cache_sha256": _digest(train),
        "validation_cache_sha256": _digest(validation),
        "validation_annotation_sha256": validation.get("_annotation_sha256"),
        "model_sha256": train["metadata"]["model_sha256"],
        "train_images": len(train["ids"]),
        "validation_images": len(validation["ids"]),
        "policy_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "selection_metric": "native_validation_AP",
        "test_used_for_training": False,
    }
    start_epoch = 0
    history = []
    best_ap, best_rl_ap = -1.0, -1.0
    reference_geometry = reference_keep = None
    best_payload = best_rl_payload = None
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if state["provenance"] != provenance or state["config"] != asdict(config):
            raise ValueError("resume requires identical caches and configuration")
        policy.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        generator.set_state(state["generator_state"])
        start_epoch = state["completed_epochs"]
        history = state["history"]
        best_ap, best_rl_ap = state["best_ap"], state["best_rl_ap"]
        reference_geometry, reference_keep = state["reference_geometry"], state["reference_keep"]
        best_payload, best_rl_payload = state["best_payload"], state["best_rl_payload"]
        # last.pt is the commit record. A crash may leave best.pt from an epoch
        # that never committed, so restore artifacts from the durable snapshots.
        for name, selected_payload in (("best", best_payload), ("best-rl", best_rl_payload)):
            if selected_payload is None:
                (output / f"{name}.pt").unlink(missing_ok=True)
                (output / f"{name}-evaluation.json").unlink(missing_ok=True)
                continue
            _atomic_save(selected_payload, output / f"{name}.pt")
            with torch.random.fork_rng(devices=[]):
                selected_policy = QueryRefinementPolicy(features.shape[-1], config).to(device)
            selected_policy.load_state_dict(selected_payload["state_dict"])
            selected_evaluation = _evaluate(
                validation,
                _predict(selected_policy, validation, mean, scale, config, device),
                evaluator,
            )
            (output / f"{name}-evaluation.json").write_text(
                json.dumps(selected_evaluation, indent=2) + "\n"
            )
    else:
        baseline_predictions = [
            {"image_id": image_id, "boxes": boxes, "scores": logits.sigmoid()}
            for image_id, boxes, logits in zip(
                validation["ids"], validation["boxes"], validation["logits"], strict=True
            )
        ]
        baseline = _evaluate(validation, baseline_predictions, evaluator)
        (output / "sft-evaluation.json").write_text(json.dumps(baseline, indent=2) + "\n")
        best_ap = float(baseline["AP"])
        (output / "best-evaluation.json").write_text(json.dumps(baseline, indent=2) + "\n")
        best_payload = _policy_payload(policy, mean, scale, config, provenance)
        _atomic_save(best_payload, output / "best.pt")
        (output / "plan.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "provenance": provenance,
                    "reward": (
                        "mean_iou_thresholds(3*TP-FP)/(3*targets)+image_AP+separable_IoU_shaping"
                    ),
                    "inference": (
                        f"{config.geometry_inference} geometry parameters; "
                        "keep probability as native AP confidence"
                    ),
                    "selection": (
                        "validation AP only, strict improvement; ties retain earlier checkpoint"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    if not resume:
        # Commit epoch zero as well, permitting recovery from first-epoch failure.
        _atomic_save(
            {
                **best_payload,
                "optimizer": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "completed_epochs": 0,
                "history": [],
                "best_ap": best_ap,
                "best_rl_ap": best_rl_ap,
                "reference_geometry": None,
                "reference_keep": None,
                "best_payload": best_payload,
                "best_rl_payload": None,
            },
            output / "last.pt",
        )
    if config.warmup_epochs:
        labels, soft_keep, action_weights = oracle_actions(train, config)
    total_epochs = config.warmup_epochs + config.rl_epochs
    started = time.perf_counter()
    for epoch in range(start_epoch, total_epochs):
        phase = "supervised_refinement" if epoch < config.warmup_epochs else "rl"
        if phase == "rl" and reference_geometry is None:
            # Start RL from the validation-selected supervised candidate, including
            # the unchanged detector if warmup never improves it. Reinitialize
            # Adam moments: supervised and policy-gradient objectives differ.
            selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            policy.load_state_dict(selected["state_dict"])
            optimizer = torch.optim.AdamW(
                policy.parameters(), lr=config.rl_learning_rate, weight_decay=config.weight_decay
            )
            with torch.no_grad():
                reference_geometry_parts, reference_keep_parts = [], []
                for start in range(0, len(features), config.batch_size):
                    geometry, keep = policy(
                        features[start : start + config.batch_size].to(device),
                        train["logits"][start : start + config.batch_size].to(device),
                    )
                    reference_geometry_parts.append(geometry.cpu())
                    reference_keep_parts.append(keep.cpu())
                reference_geometry = torch.cat(reference_geometry_parts)
                reference_keep = torch.cat(reference_keep_parts)
            _atomic_save(
                _policy_payload(policy, mean, scale, config, provenance), output / "warmup.pt"
            )
            warmup_metrics = _evaluate(
                validation, _predict(policy, validation, mean, scale, config, device), evaluator
            )
            (output / "warmup-evaluation.json").write_text(
                json.dumps(warmup_metrics, indent=2) + "\n"
            )
        permutation = torch.randperm(len(features), generator=generator)
        metrics_sum: dict[str, float] = {}
        batches = 0
        policy.train()
        for indices in permutation.split(config.batch_size):
            geometry, keep = policy(
                features[indices].to(device), train["logits"][indices].to(device)
            )
            if phase == "supervised_refinement":
                per_action = F.cross_entropy(
                    geometry.flatten(0, 1), labels[indices].flatten().to(device), reduction="none"
                ).reshape(len(indices), -1)
                weights = action_weights[indices].to(device)
                box_loss = (per_action * weights).sum() / weights.sum().clamp_min(1)
                keep_loss = F.binary_cross_entropy_with_logits(keep, soft_keep[indices].to(device))
                loss = box_loss + keep_loss
                metrics = {
                    "box_loss": float(box_loss.detach()),
                    "keep_loss": float(keep_loss.detach()),
                }
            else:
                loss, metrics = reinforce_loss(
                    geometry,
                    keep,
                    train["boxes"][indices],
                    [train["targets"][int(index)] for index in indices],
                    generator,
                    config,
                    reference_geometry[indices].to(device),
                    reference_keep[indices].to(device),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite refinement loss; checkpoint not advanced")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite refinement gradients")
            optimizer.step()
            metrics["loss"] = float(loss.detach())
            metrics["gradient_norm"] = float(gradient_norm)
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
            batches += 1
        policy.eval()
        validation_metrics = _evaluate(
            validation, _predict(policy, validation, mean, scale, config, device), evaluator
        )
        ap = float(validation_metrics["AP"])
        if not math.isfinite(ap) or not 0 <= ap <= 1:
            raise ValueError("native evaluator AP must be a finite fraction in [0,1]")
        improved = ap > best_ap
        payload = _policy_payload(policy, mean, scale, config, provenance)
        if improved:
            best_ap = ap
            best_payload = payload
            _atomic_save(payload, output / "best.pt")
            (output / "best-evaluation.json").write_text(
                json.dumps(validation_metrics, indent=2) + "\n"
            )
        if phase == "rl" and ap > best_rl_ap:
            best_rl_ap = ap
            best_rl_payload = payload
            _atomic_save(payload, output / "best-rl.pt")
            (output / "best-rl-evaluation.json").write_text(
                json.dumps(validation_metrics, indent=2) + "\n"
            )
        record = {
            "epoch": epoch + 1,
            "phase": phase,
            **{key: value / batches for key, value in metrics_sum.items()},
            "validation_AP": ap,
            "best_validation_AP": best_ap,
            "selected": improved,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        _atomic_save(
            {
                **payload,
                "optimizer": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "completed_epochs": epoch + 1,
                "history": history,
                "best_ap": best_ap,
                "best_rl_ap": best_rl_ap,
                "reference_geometry": reference_geometry,
                "reference_keep": reference_keep,
                "best_payload": best_payload,
                "best_rl_payload": best_rl_payload,
            },
            output / "last.pt",
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    summary = {
        "completed_epochs": total_epochs,
        "best_validation_AP": best_ap,
        "best_rl_validation_AP": best_rl_ap,
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "config": asdict(config),
        "provenance": provenance,
        "checkpoint": str(output / "best.pt"),
        "rl_checkpoint": str(output / "best-rl.pt"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--train-cache", required=True, type=Path)
    train.add_argument("--validation-cache", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--config", type=Path)
    train.add_argument("--resume", action="store_true")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--cache", required=True, type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--safetensors", action="store_true")
    export = subparsers.add_parser("export")
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        config = RefinementConfig(**json.loads(args.config.read_text())) if args.config else None
        result = train_refinement(
            args.train_cache, args.validation_cache, args.output, config, resume=args.resume
        )
        print(json.dumps(result, indent=2))
    elif args.command == "export":
        export_refinement_safetensors(args.checkpoint, args.output)
    else:
        cache = load_cache(args.cache)
        predictor = predict_refinement_safetensors if args.safetensors else predict_refinement
        predictions = predictor(cache, args.checkpoint, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(coco_predictions(cache, predictions)) + "\n")


if __name__ == "__main__":
    main()
