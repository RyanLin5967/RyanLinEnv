"""Direct reward fine-tuning of RF-DETR's native image-to-prediction network.

The actor is the detector itself, with its decoder and native prediction heads
trainable. There is no learned postprocessor or frozen-feature residual policy.
All 300 native queries remain in every reward and deterministic deployment.

A factorized Gaussian over query (cx, cy, log w, log h, foreground logit) defines
the stochastic policy. Standard deviations depend only on the frozen reference.
Detached samples are scored under live means: AP is not differentiated through.
Per-image COCO AP is a training surrogate; model selection uses dataset COCO AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

VERSION = 1
TRAINABLE_PREFIXES = (
    "transformer.decoder.",
    "bbox_embed.",
    "class_embed.",
    "query_feat.",
    "refpoint_embed.",
)
CONTROL_CONFIG_FIELDS = (
    "resolution",
    "seed",
    "epochs",
    "batch_images",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "train_encoder_last_blocks",
    "horizontal_flip_probability",
    "reward_backend",
    "domain_balanced_sampling",
    "device",
    "cpu_threads",
)
ACCEPTANCE = {
    "validation_AP_gain_over_frozen_and_each_matched_SFT": 0.02,
    "validation_AP95_gain_over_frozen_and_each_matched_SFT": 0.08,
    "paired_capture_hour_cluster_bootstrap_95pct_AP_lower_bound": 0.0,
    "unknown_capture_times_pooled_as_one_cluster": True,
    "all_conditions_required": True,
    "validation_selection_bias_caveat": True,
    "native_test_accessed": False,
}


@dataclass(frozen=True)
class PolicyTrainingConfig:
    resolution: int = 384
    device: str = "cpu"
    cpu_threads: int = 2
    seed: int = 791
    epochs: int = 8
    batch_images: int = 4
    groups: int = 16
    query_samples: int = 16
    estimator: str = "counterfactual"
    pixel_std: float = 2.0
    max_relative_center_std: float = 0.05
    max_log_size_std: float = 0.10
    score_std: float = 0.25
    active_query_probability: float = 1.0
    query_uniform_mixture: float = 0.10
    ap95_weight: float = 0.20
    learning_rate: float = 0.00001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    kl_coefficient: float = 0.05
    kl_target_per_query: float = 0.05
    train_encoder_last_blocks: int = 0
    horizontal_flip_probability: float = 0.5
    pilot_images: int = 128
    pilot_groups: int = 64
    pilot_min_AP_gain: float = 0.02
    pilot_min_useful_fraction: float = 0.25
    pilot_snr_images: int = 8
    pilot_snr_repeats: int = 8
    max_steps: int | None = None
    wall_seconds: float | None = None
    reward_backend: str = "coco_ap"
    domain_balanced_sampling: bool = False
    pilot_min_reward_gain: float | None = None
    trust_region_max_kl: float = 0.10
    trust_region_backtracks: int = 12
    drift_sentinel_images: int = 32
    drift_max_mean_kl: float = 0.50
    drift_max_image_kl: float = 1.0
    audit_groups: int = 16

    def validate(self):
        if self.resolution not in (384, 512) or self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("trusted resolution384/512 and cpu/mps/cuda device required")
        if self.estimator not in ("counterfactual", "loo"):
            raise ValueError("estimator must be counterfactual or loo")
        if self.reward_backend not in ("coco_ap", "wheat_ada"):
            raise ValueError("unknown native reward backend")
        if self.reward_backend == "wheat_ada" and not self.domain_balanced_sampling:
            raise ValueError("wheat ADA training requires equal-domain sampling")
        if self.pilot_min_reward_gain is not None and not 0 <= self.pilot_min_reward_gain <= 1:
            raise ValueError("pilot reward gain must be in [0,1]")
        for key, lo, hi in (
            ("cpu_threads", 1, 16),
            ("epochs", 1, 100),
            ("batch_images", 1, 32),
            ("groups", 2, 128),
            ("query_samples", 1, 300),
            ("train_encoder_last_blocks", 0, 12),
            ("pilot_images", 1, 100000),
            ("pilot_groups", 2, 256),
            ("pilot_snr_images", 1, 128),
            ("pilot_snr_repeats", 2, 64),
            ("trust_region_backtracks", 0, 20),
            ("drift_sentinel_images", 1, 512),
            ("audit_groups", 2, 128),
        ):
            if type(getattr(self, key)) is not int or not lo <= getattr(self, key) <= hi:
                raise ValueError(f"invalid bounded integer {key}")
        for key, lo, hi in (
            ("pixel_std", 0.01, 16),
            ("max_relative_center_std", 0.001, 0.25),
            ("max_log_size_std", 0.001, 0.3),
            ("score_std", 0.001, 2),
            ("query_uniform_mixture", 0.01, 1),
            ("active_query_probability", 0.01, 1),
            ("ap95_weight", 0, 1),
            ("learning_rate", 1e-8, 0.001),
            ("weight_decay", 0, 0.1),
            ("gradient_clip", 0.01, 100),
            ("kl_coefficient", 0, 10),
            ("kl_target_per_query", 0.001, 10),
            ("horizontal_flip_probability", 0, 1),
            ("pilot_min_AP_gain", 0, 1),
            ("pilot_min_useful_fraction", 0, 1),
            ("trust_region_max_kl", 0.001, 10),
            ("drift_max_mean_kl", 0.001, 100),
            ("drift_max_image_kl", 0.001, 100),
        ):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"invalid finite bounded {key}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative signed64 integer")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError("max_steps must be positive")
        if self.wall_seconds is not None:
            raise ValueError("wall_seconds is derived from the matched RL run")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def digest(document) -> str:
    return hashlib.sha256(
        (
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    ).hexdigest()


def write(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_state(path: Path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def latent_mean(raw: dict) -> torch.Tensor:
    """CPU float64 likelihood arithmetic keeps Gaussian math off MPS float64.

    The device-to-CPU conversion remains differentiable. No live output is
    detached here. Native boxes use positive widths, including bbox_reparam.
    """
    boxes = raw["pred_boxes"].to(device="cpu", dtype=torch.float64)
    logits = raw["pred_logits"][..., :1].to(device="cpu", dtype=torch.float64)
    if (
        not torch.isfinite(boxes).all()
        or not torch.isfinite(logits).all()
        or (boxes[..., 2:] <= 0).any()
    ):
        raise ValueError("native actor emitted invalid geometry/logits")
    return torch.cat((boxes[..., :2], boxes[..., 2:].log(), logits), -1)


def policy_scales(reference_mean: torch.Tensor, image_size, config: PolicyTrainingConfig):
    """Fixed reference-only SDs: center in native pixels, size in log units."""
    reference_mean = reference_mean.detach()
    dimensions = reference_mean.new_tensor(image_size)
    size = reference_mean[..., 2:4].exp()
    center_std = torch.minimum(
        torch.full_like(size, config.pixel_std) / dimensions, size * config.max_relative_center_std
    ).clamp_min(1e-8)
    size_std = (2 * config.pixel_std / (size * dimensions)).clamp(0.0001, config.max_log_size_std)
    return torch.cat((center_std, size_std, torch.full_like(size[..., :1], config.score_std)), -1)


def _quantiles(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not len(values):
        return {"count": 0, "quantiles": None}
    probabilities = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    return {
        "count": len(values),
        "quantiles": {
            str(probability): float(value)
            for probability, value in zip(
                probabilities,
                torch.quantile(values, torch.tensor(probabilities, dtype=torch.float64)),
                strict=True,
            )
        },
    }


@torch.no_grad()
def scale_observations(reference, image_size, targets, config):
    """Actual exploration scales and train-only residuals, including relative caps.

    Error diagnostics use score>.5 greedy one-to-one IoU>=.5 matches. They exclude
    misses, which are counted separately; they are not a detector performance score.
    """
    from vision_rl.detection_rl import box_iou

    dimensions = reference.new_tensor(image_size)
    scale = policy_scales(reference, image_size, config)
    size = reference[:, 2:4].exp()
    foreground = reference[:, 4].sigmoid() > 0.5
    capped = size * config.max_relative_center_std <= config.pixel_std / dimensions
    size_capped = 2 * config.pixel_std / (size * dimensions) >= config.max_log_size_std
    observations = {
        "all_query_center_SD_pixels": (scale[:, :2] * dimensions).flatten(),
        "foreground_center_SD_pixels": (scale[foreground, :2] * dimensions).flatten(),
        "all_query_log_size_SD": scale[:, 2:4].flatten(),
        "foreground_log_size_SD": scale[foreground, 2:4].flatten(),
        "all_query_center_relative_cap_indicator": capped.double().flatten(),
        "foreground_center_relative_cap_indicator": capped[foreground].double().flatten(),
        "all_query_log_size_upper_cap_indicator": size_capped.double().flatten(),
        "foreground_log_size_upper_cap_indicator": size_capped[foreground].double().flatten(),
    }
    boxes, scores = decode_latents(reference)
    ious = box_iou(boxes, targets)
    pairs, used = [], set()
    for query in torch.argsort(scores, descending=True, stable=True).tolist():
        if not foreground[query] or not len(targets):
            continue
        values = ious[query].clone()
        if used:
            values[list(used)] = -1
        target = len(targets) - 1 - int(values.flip(0).argmax())
        if values[target] >= 0.5:
            pairs.append((query, target))
            used.add(target)
    if pairs:
        query, target = torch.tensor(pairs).T
        truth_size = targets[target, 2:] - targets[target, :2]
        truth_center = (targets[target, 2:] + targets[target, :2]) / 2
        center_error = (reference[query, :2] - truth_center).abs() * dimensions
        size_error = (reference[query, 2:4] - truth_size.log()).abs()
        observations.update(
            {
                "matched_center_absolute_error_pixels": center_error.flatten(),
                "matched_log_size_absolute_error": size_error.flatten(),
                "matched_center_error_in_SD": (
                    center_error / (scale[query, :2] * dimensions)
                ).flatten(),
                "matched_log_size_error_in_SD": (size_error / scale[query, 2:4]).flatten(),
            }
        )
    counts = {
        "queries": len(reference),
        "foreground_queries": int(foreground.sum()),
        "targets": len(targets),
        "matched_targets_at_IoU50": len(pairs),
        "unmatched_targets_at_IoU50": len(targets) - len(pairs),
    }
    return observations, counts


def summarize_scales(observations):
    return {
        name: (
            {
                "count": len(values),
                "fraction": float(values.double().mean()) if len(values) else None,
            }
            if name.endswith("indicator")
            else _quantiles(values)
        )
        for name, values in observations.items()
    }


def sample_latents(mean, scale, groups, generator):
    noise = torch.randn((groups, *mean.shape), generator=generator, dtype=mean.dtype, device="cpu")
    return mean.detach()[None] + scale.detach()[None] * noise


def sample_policy(mean, reference_mean, scale, groups, generator, active_probability=1.0):
    """Mixed measure: a Gaussian action or an atom at the FROZEN reference.

    Anchoring inactive factors at the live mean would invalidate the plain
    score-function estimator. The reference atom is parameter independent.
    """
    samples = sample_latents(mean, scale, groups, generator)
    if active_probability == 1:
        return samples, torch.ones(samples.shape[:-1], dtype=torch.bool)
    active = torch.rand(samples.shape[:-1], generator=generator) < active_probability
    samples = torch.where(active[..., None], samples, reference_mean.detach()[None])
    return samples, active


def factor_log_probability(mean, scale, samples, active=None, active_probability=1.0):
    if samples.requires_grad or scale.requires_grad:
        raise ValueError("policy samples and reference-derived scales must be detached")
    # One factor is an entire query's geometry and confidence action.
    normal = (
        -0.5 * ((samples - mean[None]) / scale[None]).square()
        - scale.log()[None]
        - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    if active_probability < 1 and active is None:
        raise ValueError("mixture likelihood requires explicit sampled query gates")
    if active_probability == 1:
        return normal
    if active.shape != normal.shape or active.dtype != torch.bool:
        raise ValueError("query gates must align with sampled factors")
    return torch.where(
        active,
        normal + math.log(active_probability),
        normal.new_full((), math.log1p(-active_probability)),
    )


def gaussian_kl(mean, reference_mean, scale):
    """Exact joint Normal KL with identical fixed variances, one value per query."""
    if reference_mean.requires_grad or scale.requires_grad:
        raise ValueError("KL reference and scale must be frozen")
    return 0.5 * ((mean - reference_mean) / scale).square().sum(-1)


def decode_latents(samples):
    sizes = samples[..., 2:4].exp()
    boxes = torch.cat((samples[..., :2] - sizes / 2, samples[..., :2] + sizes / 2), -1)
    scores = samples[..., 4].sigmoid()
    return boxes, scores


@torch.no_grad()
def image_reward(
    boxes: torch.Tensor, scores: torch.Tensor, targets: torch.Tensor, *, ap95_weight=0.2
):
    """Exact COCO image AP (single category, noncrowd, maxDets500), float64.

    All queries retain their ranks, including unmatched negatives and duplicates.
    Unlike the older fixed-score refiner, each sampled image has its own ranking.
    Empty-GT images have undefined COCO AP; use negative mean confidence for
    their explicitly separate background suppression reward convention.
    """
    b = boxes.detach().cpu().double().numpy()
    s = scores.detach().cpu().double().numpy()
    t = targets.detach().cpu().double().numpy()
    if b.ndim != 3 or s.shape != b.shape[:2] or b.shape[-1] != 4 or t.ndim != 2 or t.shape[-1] != 4:
        raise ValueError("reward expects boxes[G,Q,4], scores[G,Q], targets[T,4]")
    if b.shape[1] > 500 or not np.isfinite(b).all() or not np.isfinite(s).all():
        raise ValueError("reward requires <=500 finite proposals")
    groups, queries = s.shape
    if not len(t):
        zero = torch.zeros(groups, dtype=torch.float64)
        return -torch.from_numpy(s.mean(-1)), {"AP": zero, "AP95": zero, "empty_ground_truth": True}
    order = np.argsort(-s, axis=-1, kind="stable")
    ordered = np.take_along_axis(b, order[..., None], axis=1)
    intersection = np.maximum(
        0,
        np.minimum(ordered[:, :, None, 2:], t[None, None, :, 2:])
        - np.maximum(ordered[:, :, None, :2], t[None, None, :, :2]),
    ).prod(-1)
    areas = np.maximum(0, ordered[..., 2:] - ordered[..., :2]).prod(-1)
    target_areas = (t[:, 2:] - t[:, :2]).prod(-1)
    ious = intersection / np.maximum(areas[:, :, None] + target_areas - intersection, 1e-20)
    thresholds = np.linspace(0.5, 0.95, 10)
    claimed = np.zeros((groups, 10, len(t)), bool)
    hits = np.zeros((groups, 10, queries), bool)
    group_index, threshold_index = np.arange(groups)[:, None], np.arange(10)[None]
    for rank in np.flatnonzero((ious >= 0.5).any((0, 2))):
        available = np.where(claimed, -1, ious[:, rank, None, :])
        target = len(t) - 1 - available[..., ::-1].argmax(-1)  # native later-GT tie
        best = np.take_along_axis(available, target[..., None], -1)[..., 0]
        hit = best >= thresholds
        hits[:, :, rank] = hit
        claimed[group_index, threshold_index, target] |= hit
    tp = hits.cumsum(-1, dtype=float)
    precision = tp / (np.arange(queries)[None, None, :] + 1)
    recall = tp / len(t)
    precision = np.maximum.accumulate(precision[..., ::-1], axis=-1)[..., ::-1]
    precision = np.concatenate((precision, np.zeros((*precision.shape[:-1], 1))), -1)
    grid = np.linspace(0, 1, 101)
    interpolated = np.zeros((groups, 10, 101))
    for g in range(groups):
        for threshold in range(10):
            indices = np.searchsorted(recall[g, threshold], grid, side="left")
            interpolated[g, threshold] = precision[g, threshold, indices]
    ap = torch.from_numpy(interpolated.mean((1, 2)))
    ap95 = torch.from_numpy(interpolated[:, -1].mean(-1))
    return (1 - ap95_weight) * ap + ap95_weight * ap95, {
        "AP": ap,
        "AP95": ap95,
        "empty_ground_truth": False,
    }


def query_probabilities(reference_mean, uniform_mixture):
    confidence = reference_mean.detach()[:, 4].sigmoid()
    importance = confidence / confidence.sum().clamp_min(1e-20)
    return (1 - uniform_mixture) * importance + uniform_mixture / len(confidence)


def native_reward(boxes, scores, targets, config, image_size=None):
    if config.reward_backend == "wheat_ada":
        from vision_rl.wheat_eval import official_image_rewards

        if image_size is not None:
            width, height = image_size
            dimensions = boxes.new_tensor([width, height, width, height])
            boxes, targets = boxes * dimensions, targets * dimensions
        reward = official_image_rewards(boxes, scores, targets)
        return reward, {"ADA": reward}
    return image_reward(boxes, scores, targets, ap95_weight=config.ap95_weight)


def _metric_name(config):
    return "ADA" if config.reward_backend == "wheat_ada" else "AP"


def _pilot_threshold(config):
    return (
        config.pilot_min_AP_gain
        if config.pilot_min_reward_gain is None
        else config.pilot_min_reward_gain
    )


def select_pilot_scale(summaries):
    passing = [key for key, row in summaries.items() if row["go"]]
    pool = passing or list(summaries)
    return min(pool, key=lambda key: (-summaries[key]["mean_best_of_samples_gain"], float(key)))


def validate_study_budget(
    config, protocol, train_images=None, *, integration_only=False, mode="rl", pilot=False
):
    """Bind standalone and workflow calls to the same prospective wheat budget.

    Returns the absolute update safety cap when image count is available. A
    walltime control may exceed the RL update count, never this epoch cap; if
    it cannot consume the measured time, training fails without a matched claim.
    Integration fixtures bypass first-study settings, never the global cap.
    """
    config.validate()
    if config.reward_backend != "wheat_ada":
        return None
    if mode not in ("rl", "sft_updates", "sft_walltime"):
        raise ValueError("unknown controlled-training mode")
    if protocol.get("benchmark_id") != "gwhd2021-hf-domain-ada-v1":
        raise ValueError("wheat policy requires its frozen prospective experiment protocol")
    maximum = protocol.get("budget", {}).get("maximum_epochs_equivalent")
    if type(maximum) is not int or maximum < 1 or config.epochs > maximum:
        raise ValueError("wheat configuration exceeds the prospective epoch budget")
    cap = None
    if train_images is not None:
        if type(train_images) is not int or train_images < 1:
            raise ValueError("study budget requires a positive training image count")
        cap = maximum * math.ceil(train_images / config.batch_images)
        if not pilot and config.max_steps is not None and config.max_steps > cap:
            raise ValueError("wheat update budget exceeds native epoch-equivalent cap")
    if integration_only:
        if config.max_steps is None or config.max_steps > 4:
            raise ValueError("integration study budget requires <=4 input updates")
        return cap
    study = protocol.get("first_learning_test", {})
    fixed = study.get("configuration")
    if not isinstance(fixed, dict) or not fixed:
        raise ValueError("prospective study must bind its scientific configuration")
    for name, expected in fixed.items():
        if not hasattr(config, name) or getattr(config, name) != expected:
            raise ValueError(f"wheat configuration differs from the prospective study: {name}")
    for name in ("batch_images", "learning_rate"):
        if getattr(config, name) != study.get(name):
            raise ValueError(f"wheat configuration differs from the first learning test: {name}")
    maximum_updates = study.get("maximum_optimizer_updates")
    if type(maximum_updates) is not int or maximum_updates < 1:
        raise ValueError("prospective study requires an explicit positive update budget")
    if not pilot and (config.max_steps is None or config.max_steps > maximum_updates):
        raise ValueError("wheat configuration exceeds the first learning test update budget")
    return cap


def _validate_protocol(config, protocol, *, pixel_scales=None, integration_only=False):
    if config.reward_backend != "wheat_ada":
        return
    if protocol.get("benchmark_id") != "gwhd2021-hf-domain-ada-v1":
        raise ValueError("wheat policy requires its frozen prospective experiment protocol")
    gate = protocol["exploration_gate"]
    required = {
        "pilot_images": gate["images"],
        "pilot_groups": gate["groups"],
        "active_query_probability": gate["active_query_probability"],
        "pilot_min_useful_fraction": gate[
            "minimum_fraction_images_with_useful_variation_and_positive_gain"
        ],
    }
    if any(getattr(config, key) != value for key, value in required.items()):
        raise ValueError("wheat policy configuration differs from its prospective exploration gate")
    if _pilot_threshold(config) != gate["minimum_mean_best_of_64_official_image_accuracy_gain"]:
        raise ValueError("wheat pilot threshold differs from its prospective protocol")
    if pixel_scales is not None and list(pixel_scales) != gate["native_pixel_stds"]:
        raise ValueError("wheat exploration action scales differ from its prospective protocol")
    validate_study_budget(
        config, protocol, integration_only=integration_only, pilot=pixel_scales is not None
    )


def _sampling_probabilities(data, config):
    if not config.domain_balanced_sampling:
        return None
    from collections import Counter

    domains = [image.get("domain") for image in data.images]
    if any(not isinstance(domain, str) or not domain for domain in domains):
        raise ValueError("equal-domain sampling requires an explicit domain on every image")
    counts = Counter(domains)
    probabilities = torch.tensor([1 / counts[domain] for domain in domains], dtype=torch.float64)
    return probabilities / probabilities.sum()


def _draw_order(data, config, generator, count=None):
    probabilities = _sampling_probabilities(data, config)
    count = len(data) if count is None else count
    if probabilities is None:
        return torch.randperm(len(data), generator=generator)[:count].tolist()
    return torch.multinomial(probabilities, count, replacement=True, generator=generator).tolist()


def query_indices(probabilities, count, generator):
    return torch.multinomial(probabilities.detach(), count, replacement=True, generator=generator)


def counterfactual_loss(log_probability, rewards, counterfactual_rewards, indices, probabilities):
    """Unbiased with-replacement query-factor sampling, including duplicates.

    Baseline[g,k] resets ONLY query indices[k], keeping every other sampled
    action fixed. It multiplies ONLY that factor's log probability. Baseline,
    rewards and selection probabilities are detached from the gradient.
    """
    advantage = (rewards[:, None] - counterfactual_rewards).detach()
    weights = probabilities[indices].detach().reciprocal()
    return -(advantage * log_probability[:, indices] * weights[None]).mean()


def leave_one_out_loss(log_probability, rewards):
    if len(rewards) < 2:
        raise ValueError("leave-one-out needs at least two independent rollouts")
    other = (rewards.sum() - rewards) / (len(rewards) - 1)
    return -((rewards - other).detach() * log_probability.sum(-1)).mean()


@torch.no_grad()
def counterfactual_rewards(
    samples, reference_mean, targets, indices, ap95_weight, config=None, image_size=None
):
    """Vectorized replacement; reuse duplicate selected factors without changing weights."""
    unique, inverse = indices.unique(sorted=True, return_inverse=True)
    groups, queries, _ = samples.shape
    result = []
    # Bound memory independently of the sampled-query budget.
    for start in range(0, len(unique), 8):
        selected = unique[start : start + 8]
        modified = samples[:, None].expand(-1, len(selected), -1, -1).clone()
        modified[:, torch.arange(len(selected)), selected] = reference_mean[selected]
        boxes, scores = decode_latents(modified.reshape(-1, queries, 5))
        rewards, _ = (
            native_reward(boxes, scores, targets, config, image_size)
            if config is not None
            else image_reward(boxes, scores, targets, ap95_weight=ap95_weight)
        )
        result.append(rewards.reshape(groups, len(selected)))
    return torch.cat(result, 1)[:, inverse]


def policy_objective(
    mean, reference_mean, targets, image_size, config, generator, *, kl_coefficient=None
):
    scale = policy_scales(reference_mean, image_size, config)
    samples, active = sample_policy(
        mean, reference_mean, scale, config.groups, generator, config.active_query_probability
    )
    boxes, scores = decode_latents(samples)
    rewards, stats = native_reward(boxes, scores, targets, config, image_size)
    log_probability = factor_log_probability(
        mean, scale, samples, active, config.active_query_probability
    )
    if config.estimator == "counterfactual":
        probabilities = query_probabilities(reference_mean, config.query_uniform_mixture)
        indices = query_indices(probabilities, config.query_samples, generator)
        baseline = counterfactual_rewards(
            samples, reference_mean, targets, indices, config.ap95_weight, config, image_size
        )
        actor_loss = counterfactual_loss(log_probability, rewards, baseline, indices, probabilities)
        useful_fraction = float((rewards[:, None] != baseline).double().mean())
    else:
        actor_loss = leave_one_out_loss(log_probability, rewards)
        useful_fraction = float((rewards != rewards.mean()).double().mean())
    kl = config.active_query_probability * gaussian_kl(mean, reference_mean, scale)
    coefficient = config.kl_coefficient if kl_coefficient is None else kl_coefficient
    loss = actor_loss + coefficient * kl.mean()
    return loss, {
        "actor_loss": float(actor_loss.detach()),
        "joint_KL": float(kl.sum().detach()),
        "KL_per_query": float(kl.mean().detach()),
        "reward": float(rewards.mean()),
        **{
            "sampled_" + name: float(value.mean())
            for name, value in stats.items()
            if isinstance(value, torch.Tensor)
        },
        "reward_std": float(rewards.std()),
        "useful_factor_fraction": useful_fraction,
    }


def configure_trainable(model, last_encoder_blocks=0):
    """Native eval-mode forward with gradients: never activate 13 Group-DETR groups."""
    import re

    layer_numbers = sorted(
        {
            int(match.group(1))
            for name, _ in model.named_parameters()
            if (match := re.search(r"backbone\..*encoder\.layer\.(\d+)\.", name))
        }
    )
    selected_layers = set(layer_numbers[-last_encoder_blocks:]) if last_encoder_blocks else set()
    if last_encoder_blocks and len(selected_layers) != last_encoder_blocks:
        raise ValueError("requested encoder blocks do not match the trusted architecture")
    names = []
    for name, parameter in model.named_parameters():
        match = re.search(r"backbone\..*encoder\.layer\.(\d+)\.", name)
        trainable = name.startswith(TRAINABLE_PREFIXES) or bool(
            match and int(match.group(1)) in selected_layers
        )
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    model.eval()
    if not names:
        raise ValueError("no native detector parameters selected")
    return {
        "names": names,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "native_model_parameters": sum(p.numel() for p in model.parameters()),
        "eval_mode_with_gradients": True,
        "last_encoder_blocks": sorted(selected_layers),
    }


class NativeData:
    """Only train/valid, native full-RGB validation resize; deterministic optional flips."""

    def __init__(self, source: Path, split: str, resolution: int):
        from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64

        from vision_rl.detection_training import inspect_split

        if split not in ("train", "valid"):
            raise ValueError("direct detector training accepts only train/valid")
        self.directory = source / split
        self.annotation_path = self.directory / "_annotations.coco.json"
        self.identity = inspect_split(self.directory)
        if len(self.identity["categories"]) != 1:
            raise ValueError("current reward policy requires a single foreground category")
        self.category_id = int(next(iter(self.identity["categories"])))
        self.dataset = CocoDetection(
            self.directory,
            self.annotation_path,
            transforms=make_coco_transforms_square_div_64("val", resolution),
            remap_category_ids=True,
            cat2label={self.category_id: 0},
            draft_size=None,
        )
        self.images = [self.dataset.coco.imgs[i] for i in self.dataset.ids]
        from vision_rl.data import image_hash

        self.identity["pixel_sha256"] = [
            image_hash(self.directory / image["file_name"])
            for image in sorted(self.images, key=lambda row: str(row["id"]))
        ]
        self.image_hashes = dict(
            zip(
                [image["id"] for image in sorted(self.images, key=lambda row: str(row["id"]))],
                self.identity["image_sha256"],
                strict=True,
            )
        )
        self.targets = []
        for info in self.images:
            rows = self.dataset.coco.imgToAnns[info["id"]]
            if any(row.get("iscrowd", 0) or row.get("ignore", 0) for row in rows):
                raise ValueError(
                    "exact image reward currently requires noncrowd, nonignored targets"
                )
            boxes = [[x, y, x + w, y + h] for row in rows for x, y, w, h in [row["bbox"]]]
            self.targets.append(
                torch.tensor(boxes, dtype=torch.float64).reshape(-1, 4)
                / torch.tensor([info["width"], info["height"], info["width"], info["height"]])
            )

    def __len__(self):
        return len(self.images)

    def get(self, index, flip=False):
        info = self.images[index]
        if sha(self.directory / info["file_name"]) != self.image_hashes[info["id"]]:
            raise ValueError("native image bytes changed")
        image, target = self.dataset[index]
        native = self.targets[index].clone()
        if flip:
            image = image.flip(-1)
            target["boxes"][:, 0] = 1 - target["boxes"][:, 0]
            native[:, [0, 2]] = 1 - native[:, [2, 0]]
        return image, target, native

    def verify(self):
        if sha(self.annotation_path) != self.identity["annotation_sha256"]:
            raise ValueError("native annotation bytes changed")


def _model(checkpoint, config):
    from vision_rl.detection_training import load_detector_safetensors

    return load_detector_safetensors(checkpoint, device=config.device, resolution=config.resolution)


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def pilot(
    source: Path,
    checkpoint: Path,
    output: Path,
    config: PolicyTrainingConfig | None = None,
    *,
    pixel_scales=(1.0, 2.0, 4.0),
    protocol: Path | None = None,
):
    """Prospective train-only exploration and output-gradient SNR; no model fitting."""
    config = config or PolicyTrainingConfig()
    config.validate()
    protocol_document = json.loads(protocol.read_text()) if protocol else {}
    _validate_protocol(config, protocol_document, pixel_scales=pixel_scales)
    if output.exists() and any(output.iterdir()):
        raise ValueError("pilot output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    data = NativeData(source, "train", config.resolution)
    generator = torch.Generator().manual_seed(config.seed)
    chosen = _draw_order(data, config, generator, config.pilot_images)
    metric = _metric_name(config)
    plan = {
        "version": VERSION,
        "configuration": asdict(config),
        "pixel_scales": list(pixel_scales),
        "image_ids": [data.images[i]["id"] for i in chosen],
        "reference_sha256": sha(checkpoint),
        "train_identity": data.identity,
        "source_sha256": sha(Path(__file__)),
        "acceptance": protocol_document.get("RL_acceptance", ACCEPTANCE),
        "protocol_sha256": sha(protocol) if protocol else None,
        "pilot_gate": {
            "mean_best_of_samples_gain": _pilot_threshold(config),
            "metric": metric,
            "useful_images_fraction": config.pilot_min_useful_fraction,
            "both_required": True,
        },
        "validation_accessed": False,
        "native_test_accessed": False,
    }
    write(output / "plan.json", plan)
    (output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    detector = _model(checkpoint, config)
    model = detector.model.model
    scope = configure_trainable(model, config.train_encoder_last_blocks)
    results = {str(value): [] for value in pixel_scales}
    scale_values = {str(value): {} for value in pixel_scales}
    cached = []
    started = time.perf_counter()
    for position, index in enumerate(chosen):
        info = data.images[index]
        size = (info["width"], info["height"])
        image, _, targets = data.get(index)
        _sync(config.device)
        forward_started = time.perf_counter()
        with torch.no_grad():
            raw = model(image[None].to(config.device))
            mean = latent_mean(raw)[0]
        _sync(config.device)
        forward_seconds = time.perf_counter() - forward_started
        boxes, scores = decode_latents(mean[None])
        _, baseline = native_reward(boxes, scores, targets, config, size)
        if position < config.pilot_snr_images:
            cached.append((mean, targets, (info["width"], info["height"])))
        for pixels in pixel_scales:
            cfg = replace(config, pixel_std=pixels)
            scale = policy_scales(mean, (info["width"], info["height"]), cfg)
            observed, match_counts = scale_observations(mean, size, targets, cfg)
            for name, values in observed.items():
                scale_values[str(pixels)].setdefault(name, []).append(values)
            samples, _ = sample_policy(
                mean, mean, scale, config.pilot_groups, generator, config.active_query_probability
            )
            sampled_boxes, sampled_scores = decode_latents(samples)
            reward_started = time.perf_counter()
            _, statistics = native_reward(sampled_boxes, sampled_scores, targets, cfg, size)
            _, geometry_only = native_reward(
                sampled_boxes, scores.expand(config.pilot_groups, -1), targets, cfg, size
            )
            _, score_only = native_reward(
                boxes.expand(config.pilot_groups, -1, -1), sampled_scores, targets, cfg, size
            )
            # Label-derived per-query best sampled geometry is only a diagnostic.
            from vision_rl.detection_rl import box_iou

            ious = box_iou(sampled_boxes, targets)
            if len(targets):
                best = ious.amax(-1).argmax(0)
                oracle_boxes = sampled_boxes[best, torch.arange(len(mean))]
                _, oracle = native_reward(oracle_boxes[None], scores, targets, cfg, size)
                oracle_ap = float(oracle[metric][0])
            else:
                oracle_ap = 0.0
            row = {
                "image_id": info["id"],
                "metric": metric,
                "baseline_" + metric: float(baseline[metric][0]),
                "best_sample_" + metric: float(statistics[metric].max()),
                "mean_sample_" + metric: float(statistics[metric].mean()),
                "best_geometry_only_" + metric: float(geometry_only[metric].max()),
                "best_score_only_" + metric: float(score_only[metric].max()),
                "sample_std": float(statistics[metric].std()),
                "oracle_geometry_" + metric: oracle_ap,
                "forward_seconds": forward_seconds,
                "reward_seconds": time.perf_counter() - reward_started,
                "actual_policy_scales": summarize_scales(observed),
                "residual_diagnostic_matching": match_counts,
            }
            results[str(pixels)].append(row)
        if (position + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "event": "pilot_images",
                        "completed": position + 1,
                        "total": len(chosen),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    summaries = {}
    for key, rows in results.items():
        gains = np.array([row["best_sample_" + metric] - row["baseline_" + metric] for row in rows])
        useful = np.array(
            [
                row["sample_std"] > 1e-6 and gain > 1e-6
                for row, gain in zip(rows, gains, strict=True)
            ]
        )
        summaries[key] = {
            "mean_best_of_samples_gain": float(gains.mean()),
            "metric": metric,
            "useful_images_fraction": float(useful.mean()),
            "mean_oracle_geometry_gain": float(
                np.mean([r["oracle_geometry_" + metric] - r["baseline_" + metric] for r in rows])
            ),
            "go": bool(
                gains.mean() >= _pilot_threshold(config)
                and useful.mean() >= config.pilot_min_useful_fraction
            ),
            "actual_policy_scales": summarize_scales(
                {name: torch.cat(values) for name, values in scale_values[key].items()}
            ),
        }
    selected_scale = select_pilot_scale(summaries)
    snr = {}
    for estimator in ("loo", "counterfactual"):
        cfg = replace(config, pixel_std=float(selected_scale), estimator=estimator)
        gradients = []
        snr_started = time.perf_counter()
        for repetition in range(config.pilot_snr_repeats):
            current = []
            for mean, targets, size in cached:
                live = mean.detach().clone().requires_grad_()
                loss, _ = policy_objective(live, mean, targets, size, cfg, generator)
                current.append(torch.autograd.grad(loss, live)[0].flatten())
            gradients.append(torch.cat(current))
        gradients = torch.stack(gradients)
        signal = gradients.mean(0).square().sum()
        noise = gradients.var(0, unbiased=True).sum()
        snr[estimator] = {
            "mean_gradient_norm": float(signal.sqrt()),
            "noise_norm": float(noise.sqrt()),
            "output_gradient_SNR": float((signal / noise.clamp_min(1e-30)).sqrt()),
            "repetitions": config.pilot_snr_repeats,
            "images": len(cached),
            "wall_seconds": time.perf_counter() - snr_started,
            "scope": "gradient at native actor outputs, not full parameter gradient",
        }
    # Real image-forward/backward probe confirms reward gradients reach native heads.
    image, _, targets = data.get(chosen[0])
    model.zero_grad(set_to_none=True)
    gradient_started = time.perf_counter()
    raw = model(image[None].to(config.device))
    live = latent_mean(raw)[0]
    cfg = replace(config, pixel_std=float(selected_scale))
    loss, metrics = policy_objective(live, cached[0][0], targets, cached[0][2], cfg, generator)
    loss.backward()
    _sync(config.device)
    gradient_scopes = {}
    for prefix in ("transformer.decoder.", "class_embed.", "bbox_embed."):
        values = [
            p.grad
            for name, p in model.named_parameters()
            if name.startswith(prefix) and p.grad is not None
        ]
        gradient_scopes[prefix] = {
            "tensors": len(values),
            "finite": all(bool(v.isfinite().all()) for v in values),
            "absolute_sum": sum(float(v.abs().sum()) for v in values),
        }
    data.verify()
    summary = {
        "status": "pilot_completed",
        "scales": summaries,
        "selected_pixel_std": float(selected_scale),
        "go": summaries[selected_scale]["go"],
        "signal_to_noise": snr,
        "native_gradient_probe": gradient_scopes,
        "gradient_probe_seconds": time.perf_counter() - gradient_started,
        "gradient_probe_metrics": metrics,
        "trainable_scope": scope,
        "total_wall_seconds": time.perf_counter() - started,
        "forward_images_per_second": len(chosen)
        / sum(r["forward_seconds"] for r in results[selected_scale]),
        "reference_sha256": plan["reference_sha256"],
        "not_a_model_score": True,
        "native_test_accessed": False,
        "validation_accessed": False,
    }
    write(output / "items.json", results)
    write(output / "summary.json", summary)
    return summary


def _native_predictions(raw, images, category_id):
    from vision_rl.detection_inference import decode_queries

    predictions = []
    boxes = raw["pred_boxes"].detach().cpu().numpy()
    logits = raw["pred_logits"].detach().cpu().numpy()
    for i, image in enumerate(images):
        for row in decode_queries(boxes[i], logits[i], (category_id,)):
            x1, y1, x2, y2 = row["box"]
            predictions.append(
                {
                    "image_id": image["id"],
                    "category_id": category_id,
                    "bbox": [
                        x1 * image["width"],
                        y1 * image["height"],
                        (x2 - x1) * image["width"],
                        (y2 - y1) * image["height"],
                    ],
                    "score": row["score"],
                }
            )
    return predictions


def calibrated_wheat_evaluation(
    annotation_path, predictions, image_ids=None, *, model=None, output=None
):
    """Compatibility wrapper for the one shared calibration implementation."""
    from vision_rl.wheat_eval import calibrated_wheat_evaluation as evaluate

    return evaluate(annotation_path, predictions, image_ids=image_ids, model=model, output=output)


@torch.no_grad()
def evaluate_model(model, data, config):
    from vision_rl.detection_eval import _cached_ap, _prepared_cache, evaluate_coco

    data.verify()
    model.eval()
    predictions = []
    for start in range(0, len(data), config.batch_images):
        indices = list(range(start, min(start + config.batch_images, len(data))))
        batch = torch.stack([data.get(i)[0] for i in indices]).to(config.device)
        raw = model(batch)
        predictions.extend(
            _native_predictions(raw, [data.images[i] for i in indices], data.category_id)
        )
    report = evaluate_coco(
        data.annotation_path,
        predictions,
        image_ids=[image["id"] for image in data.images],
        model={"name": "native-detector-policy"},
    )
    prepared = _prepared_cache(report["match_cache"])
    prepared["iou_thresholds"] = prepared["iou_thresholds"][-1:]
    for image in prepared["items"].values():
        for category in image.values():
            category["matches"] = category["matches"][-1:]
            category["ignored_detections"] = category["ignored_detections"][-1:]
    report["AP95"] = _cached_ap(prepared, report["image_ids"])
    if config.reward_backend == "wheat_ada":
        wheat, predictions = calibrated_wheat_evaluation(
            data.annotation_path, predictions, report["image_ids"]
        )
        report["wheat"] = wheat
        report["ADA"] = wheat["metrics"]["detection_acc_avg_dom"]
        report["foreground_logit_offset"] = wheat["calibration"]["foreground_logit_offset"]
    data.verify()
    return report, predictions


def _criterion(detector, source, output, device):
    from rfdetr.config import TrainConfig
    from rfdetr.models.lwdetr import build_criterion_from_config

    criterion, _ = build_criterion_from_config(
        detector.model_config, TrainConfig(dataset_dir=str(source), output_dir=str(output))
    )
    # Native300 queries in both arms, not GroupDETR13x training-only queries.
    return criterion.to(device).eval()


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _rng_state(generators, device):
    import random

    numpy_state = np.random.get_state()
    state = {
        "generators": {key: value.get_state() for key, value in generators.items()},
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }
    if device == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    elif device == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state, generators, device):
    import random

    for key, generator in generators.items():
        generator.set_state(state["generators"][key])
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    if device == "mps":
        torch.mps.set_rng_state(state["accelerator"])
    elif device == "cuda":
        torch.cuda.set_rng_state_all(state["accelerator"])


def _trainable_state(model):
    # Frozen parameters are recovered from the immutable reference checkpoint;
    # avoid writing120MB of unchanged weights at every committed training step.
    return {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }


def _restore_trainable(model, state):
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if current.keys() != state.keys():
        raise ValueError("checkpoint trainable parameter scope differs")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if (
                value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not value.isfinite().all()
            ):
                raise ValueError("checkpoint contains an invalid native parameter tensor")
            parameter.copy_(value.to(parameter.device))


def _restore_buffers(model, state):
    current = dict(model.named_buffers())
    if current.keys() != state.keys():
        raise ValueError("checkpoint buffer scope differs")
    with torch.no_grad():
        for name, buffer in current.items():
            value = state[name]
            if (
                value.shape != buffer.shape
                or value.dtype != buffer.dtype
                or not value.isfinite().all()
            ):
                raise ValueError("checkpoint contains an invalid native buffer")
            buffer.copy_(value.to(buffer.device))


@torch.no_grad()
def _batch_reference_kl(means, reference_means, image_sizes, config):
    return torch.stack(
        [
            config.active_query_probability
            * gaussian_kl(mean, reference, policy_scales(reference, size, config)).mean()
            for mean, reference, size in zip(means, reference_means, image_sizes, strict=True)
        ]
    )


def _sentinel_indices(data, config):
    """Independent fixed train-only monitor; cover every domain when requested."""
    generator = torch.Generator().manual_seed(config.seed + 1000003)
    count = min(config.drift_sentinel_images, len(data))
    selected = []
    if config.domain_balanced_sampling:
        domains = sorted({image["domain"] for image in data.images})
        if len(domains) > count:
            raise ValueError("drift sentinel must cover every training domain")
        for domain in domains:
            members = [i for i, image in enumerate(data.images) if image["domain"] == domain]
            selected.append(members[int(torch.randint(len(members), (), generator=generator))])
    for index in torch.randperm(len(data), generator=generator).tolist():
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    return selected


@torch.no_grad()
def paired_training_reward_audit(mean, reference, targets, image_size, config, seed):
    """Common-random-number audit of the same fixed TRAIN image across checkpoints.

    This estimates expected noisy reward; it neither changes the objective nor
    replaces full deterministic validation. Audit RNG never touches optimizer RNG.
    """
    scale = policy_scales(reference, image_size, config)
    sampled, _ = sample_policy(
        mean,
        reference,
        scale,
        config.audit_groups,
        torch.Generator().manual_seed(seed),
        config.active_query_probability,
    )
    original, _ = sample_policy(
        reference,
        reference,
        scale,
        config.audit_groups,
        torch.Generator().manual_seed(seed),
        config.active_query_probability,
    )
    current_reward, _ = native_reward(*decode_latents(sampled), targets, config, image_size)
    reference_reward, _ = native_reward(*decode_latents(original), targets, config, image_size)
    difference = current_reward - reference_reward
    deterministic = {}
    geometry = torch.cat((mean[:, :4], reference[:, 4:]), -1)
    scores = torch.cat((reference[:, :4], mean[:, 4:]), -1)
    for name, values in (
        ("reference", reference),
        ("current", mean),
        ("geometry_only", geometry),
        ("confidence_only", scores),
    ):
        reward, _ = native_reward(*decode_latents(values[None]), targets, config, image_size)
        deterministic[name] = float(reward[0])
    return {
        "reference_expected_reward": float(reference_reward.mean()),
        "current_expected_reward": float(current_reward.mean()),
        "paired_expected_reward_gain": float(difference.mean()),
        "paired_noise_standard_error": float(
            difference.std(unbiased=True) / math.sqrt(config.audit_groups)
        ),
        "deterministic_reward": deterministic,
        "groups": config.audit_groups,
        "confidence_calibration": "raw foreground logits, fixed zero offset in every component",
        "scope": "TRAIN diagnostic; output component substitutions, not independent models",
    }


@torch.no_grad()
def monitor_reference_drift(model, reference_model, data, indices, config):
    values, audits = [], []
    for start in range(0, len(indices), config.batch_images):
        chosen = indices[start : start + config.batch_images]
        batch = torch.stack([data.get(i)[0] for i in chosen]).to(config.device)
        means = latent_mean(model(batch))
        references = latent_mean(reference_model(batch))
        sizes = [(data.images[i]["width"], data.images[i]["height"]) for i in chosen]
        values.extend(_batch_reference_kl(means, references, sizes, config).tolist())
        for position, index in enumerate(chosen):
            audits.append(
                {
                    "image_id": data.images[index]["id"],
                    "domain": data.images[index].get("domain"),
                    **paired_training_reward_audit(
                        means[position],
                        references[position],
                        data.targets[index],
                        sizes[position],
                        config,
                        config.seed + 2000003 + index,
                    ),
                }
            )
    report = {
        "mean_KL_per_query": float(np.mean(values)),
        "max_image_KL_per_query": max(values),
        "per_image": [
            {"image_id": data.images[i]["id"], "KL_per_query": value}
            for i, value in zip(indices, values, strict=True)
        ],
        "scope": "Fixed train sentinel, a heuristic drift alarm, not a global bound",
    }
    report["passed"] = (
        report["mean_KL_per_query"] <= config.drift_max_mean_kl
        and report["max_image_KL_per_query"] <= config.drift_max_image_kl
    )
    report["KL_quantiles"] = _quantiles(values)
    report["paired_reward_audit"] = audits
    if config.domain_balanced_sampling:
        domains = sorted({row["domain"] for row in audits})
        report["paired_expected_reward_gain"] = float(
            np.mean(
                [
                    np.mean(
                        [
                            row["paired_expected_reward_gain"]
                            for row in audits
                            if row["domain"] == domain
                        ]
                    )
                    for domain in domains
                ]
            )
        )
        report["reward_aggregation"] = "equal-domain mean of fixed TRAIN sentinel image diagnostics"
    else:
        report["paired_expected_reward_gain"] = float(
            np.mean([row["paired_expected_reward_gain"] for row in audits])
        )
        report["reward_aggregation"] = "mean of fixed TRAIN sentinel image diagnostics"
    return report


def backtracked_adam_step(
    model, optimizer, batch, live_means, reference_means, image_sizes, config
):
    """Accept an Adam parameter proposal only after measured reference-KL checks.

    Each image's mean-query KL must stay below max(the fixed bound, its pre-step
    KL). Backtracking scales parameter displacement, preserving Adam's gradient
    moments for an accepted step. Failure restores parameters, buffers, optimizer
    and raises before a checkpoint/update is committed. This is a batch-local
    trust check, not a guarantee about unseen training or validation images.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    buffers = _cpu(dict(model.named_buffers()))
    optimizer_before = _cpu(optimizer.state_dict())
    pre = _batch_reference_kl(live_means.detach(), reference_means, image_sizes, config)
    limits = pre.clamp_min(config.trust_region_max_kl)
    optimizer.step()
    delta = [p.detach() - old for p, old in zip(parameters, before, strict=True)]
    accepted = None
    try:
        for attempt in range(config.trust_region_backtracks + 1):
            fraction = 0.5**attempt
            with torch.no_grad():
                for parameter, old, change in zip(parameters, before, delta, strict=True):
                    parameter.copy_(old + fraction * change)
                proposal = latent_mean(model(batch))
                post = _batch_reference_kl(proposal, reference_means, image_sizes, config)
            if torch.isfinite(post).all() and (post <= limits + 1e-10).all():
                accepted = {
                    "trust_region_step_fraction": fraction,
                    "trust_region_backtracks": attempt,
                    "trust_region_post_KL_per_query": float(post.mean()),
                    "trust_region_max_image_KL_per_query": float(post.max()),
                    "trust_region_preexisting_bound_violations": int(
                        (pre > config.trust_region_max_kl).sum()
                    ),
                }
                break
    finally:
        if accepted is None:
            with torch.no_grad():
                for parameter, old in zip(parameters, before, strict=True):
                    parameter.copy_(old)
            _restore_buffers(model, buffers)
            optimizer.load_state_dict(optimizer_before)
    if accepted is None:
        raise FloatingPointError(
            "No native Adam proposal passed the bounded reference-KL backtracking check"
        )
    return accepted


def _export_native(model, output):
    from safetensors.torch import save_file

    state = {
        key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()
    }
    if any(not value.isfinite().all() for value in state.values()):
        raise FloatingPointError("native checkpoint contains nonfinite weights")
    temporary = output.with_suffix(".tmp")
    save_file(state, str(temporary))
    os.replace(temporary, output)
    return sha(output)


def _publish_selected(output, selections):
    import shutil

    for name, record in selections.items():
        if record is None:
            (output / (name + ".safetensors")).unlink(missing_ok=True)
            continue
        source = output / record["artifact"]
        if sha(source) != record["sha256"]:
            raise ValueError("committed selected native checkpoint changed")
        temporary = output / (name + ".safetensors.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, output / (name + ".safetensors"))


def _gate(pilot_path, checkpoint_sha, train_identity, config, protocol=None):
    if pilot_path is None:
        raise ValueError("a completed train-only exploration pilot is required before RL fitting")
    summary = json.loads((pilot_path / "summary.json").read_text())
    plan = json.loads((pilot_path / "plan.json").read_text())
    implementation = pilot_path / "implementation.py"
    if (
        plan.get("source_sha256") != sha(Path(__file__))
        or not implementation.is_file()
        or sha(implementation) != plan["source_sha256"]
    ):
        raise ValueError("pilot implementation differs from the current native trainer")
    if plan.get("protocol_sha256") != (sha(protocol) if protocol else None):
        raise ValueError("pilot experiment protocol differs")
    if summary["reference_sha256"] != checkpoint_sha or plan["train_identity"] != train_identity:
        raise ValueError("pilot reference checkpoint or training data differs")
    if summary.get("validation_accessed") or summary.get("native_test_accessed"):
        raise ValueError("exploration gate must use training images only")
    for name in (
        "active_query_probability",
        "score_std",
        "max_relative_center_std",
        "max_log_size_std",
        "ap95_weight",
        "pilot_min_AP_gain",
        "pilot_min_useful_fraction",
        "pilot_min_reward_gain",
        "reward_backend",
        "domain_balanced_sampling",
        "pilot_images",
        "pilot_groups",
    ):
        if plan["configuration"].get(name, getattr(PolicyTrainingConfig(), name)) != getattr(
            config, name
        ):
            raise ValueError(f"pilot policy configuration differs: {name}")
    scale = summary["scales"].get(str(float(config.pixel_std)), {})
    return {
        "go": bool(scale.get("go")),
        "pilot_summary_sha256": sha(pilot_path / "summary.json"),
        "pilot_plan_sha256": sha(pilot_path / "plan.json"),
        "scale": scale,
    }


def train(
    source: Path,
    checkpoint: Path,
    output: Path,
    config: PolicyTrainingConfig | None = None,
    *,
    mode="rl",
    pilot_path: Path | None = None,
    control_run: Path | None = None,
    protocol: Path | None = None,
    resume=False,
    integration_only=False,
):
    """Native image-forward optimization and explicit matched continued-SFT arms.

    Modes are rl, sft_updates, and sft_walltime. SFT controls require a completed
    RL run and identical core training settings, inputs, initialization, and
    native inference. Equal-walltime budgets count preprocessing/optimization
    only; checkpoint/evaluation overhead is separately recorded. CPU resume is
    tested bit-exact; accelerator replay remains subject to backend determinism.
    """
    import fcntl

    config = config or PolicyTrainingConfig()
    config.validate()
    if mode not in ("rl", "sft_updates", "sft_walltime"):
        raise ValueError("unknown controlled-training mode")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another trainer owns this output") from None
        return _train(
            source,
            checkpoint,
            output,
            config,
            mode=mode,
            pilot_path=pilot_path,
            control_run=control_run,
            protocol=protocol,
            resume=resume,
            integration_only=integration_only,
        )


def _train(
    source,
    checkpoint,
    output,
    config,
    *,
    mode,
    pilot_path,
    control_run,
    protocol,
    resume,
    integration_only=False,
):
    import random
    from importlib.metadata import version

    if any(p.name != ".training.lock" for p in output.iterdir()) and not resume:
        raise ValueError("training output must be fresh or explicitly resumed")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    train_data = NativeData(source, "train", config.resolution)
    val_data = NativeData(source, "valid", config.resolution)
    if train_data.category_id != val_data.category_id:
        raise ValueError("train/validation class semantics differ")
    if set(train_data.identity["image_sha256"]) & set(val_data.identity["image_sha256"]):
        raise ValueError("training and validation image bytes overlap")
    if set(train_data.identity.get("pixel_sha256", [])) & set(
        val_data.identity.get("pixel_sha256", [])
    ):
        raise ValueError("training and validation decoded pixels overlap")
    reference_sha = sha(checkpoint)
    primary = _metric_name(config)
    protocol_document = json.loads(protocol.read_text()) if protocol else {}
    _validate_protocol(config, protocol_document, integration_only=integration_only)
    study_step_cap = validate_study_budget(
        config, protocol_document, len(train_data), integration_only=integration_only, mode=mode
    )
    acceptance = protocol_document.get("RL_acceptance", ACCEPTANCE)
    if integration_only and (
        len(train_data) > 4 or len(val_data) > 4 or config.max_steps is None or config.max_steps > 4
    ):
        raise ValueError(
            "integration checks require <=4 train/validation images and <=4 RL updates"
        )
    gate = (
        (
            {"go": True, "integration_only": True, "not_a_benchmark_experiment": True}
            if integration_only
            else _gate(pilot_path, reference_sha, train_data.identity, config, protocol)
        )
        if mode == "rl"
        else None
    )
    if gate is not None and not gate["go"]:
        report = {
            "status": "no_go_action_space",
            "mode": mode,
            "gate": gate,
            "optimizer_steps": 0,
            "reference_sha256": reference_sha,
            "native_test_accessed": False,
        }
        write(output / "summary.json", report)
        return report
    batches_per_epoch = math.ceil(len(train_data) / config.batch_images)
    schedule_steps = config.epochs * batches_per_epoch
    target_steps = config.max_steps or schedule_steps
    wall_budget = None
    control_identity = None
    if mode != "rl":
        if control_run is None:
            raise ValueError("matched SFT controls require a completed RL run")
        rl_summary = json.loads((control_run / "summary.json").read_text())
        rl_plan = json.loads((control_run / "plan.json").read_text())
        if rl_summary["status"] != "completed" or rl_plan["mode"] != "rl":
            raise ValueError("control reference must be completed actual RL training")
        if rl_plan.get("protocol_sha256") != (sha(protocol) if protocol else None):
            raise ValueError("matched control protocol differs")
        if rl_plan.get("integration_only", False) != integration_only:
            raise ValueError("matched control experiment purpose differs")
        for name in CONTROL_CONFIG_FIELDS:
            if rl_plan["configuration"][name] != getattr(config, name):
                raise ValueError(f"matched control configuration differs: {name}")
        if (
            rl_plan["reference_sha256"] != reference_sha
            or rl_plan["train_identity"] != train_data.identity
            or rl_plan["validation_identity"] != val_data.identity
        ):
            raise ValueError("matched control initialization/data differs")
        target_steps = rl_summary["optimizer_steps"]
        if mode == "sft_walltime":
            wall_budget = rl_summary["optimization_seconds"]
            target_steps = (
                study_step_cap
                if study_step_cap is not None
                else max(target_steps, 100 * batches_per_epoch)
            )
        if study_step_cap is not None and target_steps > study_step_cap:
            raise ValueError("matched control updates exceed the prospective epoch budget")
        control_identity = {
            "run_plan_sha256": sha(control_run / "plan.json"),
            "run_summary_sha256": sha(control_run / "summary.json"),
            "RL_updates": rl_summary["optimizer_steps"],
            "RL_optimization_seconds": rl_summary["optimization_seconds"],
        }
    detector = _model(checkpoint, config)
    model = detector.model.model
    scope = configure_trainable(model, config.train_encoder_last_blocks)
    reference_model = _model(checkpoint, config).model.model if mode == "rl" else None
    if reference_model is not None:
        reference_model.requires_grad_(False).eval()
    criterion = _criterion(detector, source, output, config.device) if mode != "rl" else None
    sentinel = _sentinel_indices(train_data, config)
    plan = {
        "version": VERSION,
        "integration_only": integration_only,
        "mode": mode,
        "configuration": asdict(config),
        "reference_sha256": reference_sha,
        "train_identity": train_data.identity,
        "validation_identity": val_data.identity,
        "trainable_scope": scope,
        "target_optimizer_steps": target_steps,
        "wall_budget_seconds": wall_budget,
        "learning_rate_schedule_steps": schedule_steps,
        "gate": gate,
        "matched_control": control_identity,
        "acceptance": acceptance,
        "primary_metric": primary,
        "sampling": "inverse-domain-size IID draws with replacement"
        if config.domain_balanced_sampling
        else "uniform permutation",
        "drift_sentinel_image_ids": [train_data.images[i]["id"] for i in sentinel],
        "protocol_sha256": sha(protocol) if protocol else None,
        "source_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                "detector_policy_training.py",
                "detection_training.py",
                "detection_inference.py",
                "detection_eval.py",
            )
            + (("wheat_eval.py",) if config.reward_backend == "wheat_ada" else ())
        },
        "software": {
            name: version(name) for name in ("torch", "rfdetr", "torchvision", "pycocotools")
        },
        "native_test_accessed": False,
        "augmentation": "native fullRGB fixed-square normalization + seededhorizontalflip",
        "resume_semantics": (
            "Exact CPU fixed-update replay; accelerator numerics and walltime stopping "
            "depend on runtime conditions."
        ),
    }
    if mode != "rl":
        for key in ("source_sha256", "software", "trainable_scope", "drift_sentinel_image_ids"):
            if plan[key] != rl_plan[key]:
                raise ValueError(f"matched control {key} differs from the RL implementation")
    if sha(checkpoint) != reference_sha:
        raise ValueError("initial checkpoint changed while loading")
    if resume:
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError(
                "resume requires identical source/configuration/data/reference/control"
            )
    else:
        write(output / "plan.json", plan)
        (output / "implementation.py").write_bytes(Path(__file__).read_bytes())
        if protocol is not None:
            (output / "protocol.json").write_bytes(protocol.read_bytes())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # Same schedule in both control arms, held at its final rate after the RL update budget.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / schedule_steps, 1.0))),
    )
    generators = {
        name: torch.Generator().manual_seed(config.seed + i)
        for i, name in enumerate(("order", "actions", "augmentation"), start=1)
    }
    state = {
        "version": VERSION,
        "plan_sha256": digest(plan),
        "epoch": 0,
        "cursor": 0,
        "order": [],
        "optimizer_steps": 0,
        "optimization_seconds": 0.0,
        "kl_coefficient": config.kl_coefficient,
        "history": [],
        "selections": {"best": None, "best-trained": None},
        "epoch_metrics": {},
        "epoch_images": 0,
        "draw_chain_sha256": digest([]),
        "matched_prefix_draw_sha256": None,
        "draw_chain_history": [],
    }
    started = time.perf_counter()

    def commit():
        saved = {
            **state,
            "model": _trainable_state(model),
            "buffers": _cpu(dict(model.named_buffers())),
            "optimizer": _cpu(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(generators, config.device),
        }
        save_state(output / "last.pt", saved)

    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved["plan_sha256"] != digest(plan):
            raise ValueError("resume state belongs to different frozen inputs")
        _restore_trainable(model, saved["model"])
        _restore_buffers(model, saved["buffers"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = {
            k: v
            for k, v in saved.items()
            if k not in ("model", "buffers", "optimizer", "scheduler", "rng")
        }
        _restore_rng(saved["rng"], generators, config.device)
        _publish_selected(output, state["selections"])
    else:
        baseline, predictions = evaluate_model(model, val_data, config)
        write(output / "initial-evaluation.json", baseline)
        write(output / "initial-predictions.coco.json", predictions)
        path = output / "initial.safetensors"
        model_sha = _export_native(model, path)
        state["selections"]["best"] = {
            "artifact": path.name,
            "sha256": model_sha,
            "AP": baseline["AP"],
            "AP95": baseline["AP95"],
            "primary_score": baseline[primary],
            "primary_metric": primary,
            "foreground_logit_offset": baseline.get("foreground_logit_offset", 0.0),
            "optimizer_step": 0,
        }
        # Lazy third-party imports/model construction can consume or reseed
        # global RNGs on their first process invocation. Establish the durable
        # training boundary only after all initial evaluation/export setup.
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed % (2**32))
        commit()
        _publish_selected(output, state["selections"])

    while state["optimizer_steps"] < target_steps and (
        wall_budget is None or state["optimization_seconds"] < wall_budget
    ):
        if not state["order"]:
            state["order"] = _draw_order(train_data, config, generators["order"])
            state["cursor"] = 0
        indices = state["order"][state["cursor"] : state["cursor"] + config.batch_images]
        active_started = time.perf_counter()
        flips = (
            torch.rand(len(indices), generator=generators["augmentation"])
            < config.horizontal_flip_probability
        ).tolist()
        loaded = [train_data.get(index, flip) for index, flip in zip(indices, flips, strict=True)]
        batch = torch.stack([row[0] for row in loaded]).to(config.device)
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        raw = model(batch)
        if raw["pred_boxes"].shape[1] != 300:
            raise ValueError("native actor changed the300-query deployment policy")
        metrics = {}
        if mode == "rl":
            with torch.no_grad():
                reference_raw = reference_model(batch)
                reference_means = latent_mean(reference_raw)
            means = latent_mean(raw)
            losses = []
            for position, index in enumerate(indices):
                image = train_data.images[index]
                loss, one = policy_objective(
                    means[position],
                    reference_means[position],
                    loaded[position][2],
                    (image["width"], image["height"]),
                    config,
                    generators["actions"],
                    kl_coefficient=state["kl_coefficient"],
                )
                # Monitor the deployed all-live deterministic policy, including for sparse mixtures.
                native_boxes = raw["pred_boxes"][position].detach().cpu().double()
                xyxy = torch.cat(
                    (
                        native_boxes[:, :2] - native_boxes[:, 2:] / 2,
                        native_boxes[:, :2] + native_boxes[:, 2:] / 2,
                    ),
                    -1,
                )
                _, deterministic = native_reward(
                    xyxy[None],
                    raw["pred_logits"][position, :, 0].detach().cpu().double().sigmoid()[None],
                    loaded[position][2],
                    config,
                    (image["width"], image["height"]),
                )
                one.update(
                    {
                        "deterministic_train_" + key: float(value[0])
                        for key, value in deterministic.items()
                        if isinstance(value, torch.Tensor)
                    }
                )
                losses.append(loss)
                for key, value in one.items():
                    metrics[key] = metrics.get(key, 0) + value / len(indices)
            objective = torch.stack(losses).mean()
        else:
            targets = [
                {
                    key: value.to(config.device) if isinstance(value, torch.Tensor) else value
                    for key, value in row[1].items()
                }
                for row in loaded
            ]
            losses = criterion(raw, targets)
            objective = sum(
                losses[key] * criterion.weight_dict[key]
                for key in losses
                if key in criterion.weight_dict
            )
            metrics["supervised_loss"] = float(objective.detach())
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite native optimization objective")
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.gradient_clip
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite native policy gradient")
        if mode == "rl":
            metrics.update(
                backtracked_adam_step(
                    model,
                    optimizer,
                    batch,
                    means,
                    reference_means,
                    [
                        (train_data.images[i]["width"], train_data.images[i]["height"])
                        for i in indices
                    ],
                    config,
                )
            )
        else:
            optimizer.step()
        scheduler.step()
        _sync(config.device)
        state["optimization_seconds"] += time.perf_counter() - active_started
        state["optimizer_steps"] += 1
        state["draw_chain_sha256"] = digest(
            {
                "previous": state["draw_chain_sha256"],
                "batch": [
                    {"image_id": train_data.images[i]["id"], "flip": flip}
                    for i, flip in zip(indices, flips, strict=True)
                ],
            }
        )
        if control_identity and state["optimizer_steps"] == control_identity["RL_updates"]:
            state["matched_prefix_draw_sha256"] = state["draw_chain_sha256"]
        state["draw_chain_history"].append(state["draw_chain_sha256"])
        state["cursor"] += len(indices)
        metrics["unclipped_gradient_norm"] = float(norm)
        if mode == "rl" and config.kl_coefficient:
            ratio = metrics["KL_per_query"] / config.kl_target_per_query
            multiplier = 1.02 if ratio > 2 else 1 / 1.02 if ratio < 0.5 else 1
            state["kl_coefficient"] = min(10.0, max(0.0001, state["kl_coefficient"] * multiplier))
        for key, value in metrics.items():
            state["epoch_metrics"][key] = state["epoch_metrics"].get(key, 0) + value * len(indices)
        state["epoch_images"] += len(indices)
        terminal = (
            state["optimizer_steps"] >= target_steps
            or wall_budget is not None
            and state["optimization_seconds"] >= wall_budget
        )
        epoch_end = state["cursor"] >= len(state["order"])
        if epoch_end or terminal:
            if mode == "rl":
                drift_started = time.perf_counter()
                drift = monitor_reference_drift(
                    model, reference_model, train_data, sentinel, config
                )
                _sync(config.device)
                state["optimization_seconds"] += time.perf_counter() - drift_started
                write(output / f"drift-step-{state['optimizer_steps']:06d}.json", drift)
                if not drift["passed"]:
                    raise FloatingPointError(
                        "Native policy exceeded the preregistered train sentinel drift alarm"
                    )
            evaluation, predictions = evaluate_model(model, val_data, config)
            record = {
                "epoch": state["epoch"] + 1,
                "optimizer_step": state["optimizer_steps"],
                "validation_AP": evaluation["AP"],
                "validation_AP95": evaluation["AP95"],
                "validation_primary_score": evaluation[primary],
                "primary_metric": primary,
                **{
                    key: value / state["epoch_images"]
                    for key, value in state["epoch_metrics"].items()
                },
            }
            step = state["optimizer_steps"]
            trained = state["selections"]["best-trained"]
            if trained is None or evaluation[primary] > trained["primary_score"]:
                path = output / f"selected-step-{step:06d}.safetensors"
                selected = {
                    "artifact": path.name,
                    "sha256": _export_native(model, path),
                    "AP": evaluation["AP"],
                    "AP95": evaluation["AP95"],
                    "primary_score": evaluation[primary],
                    "primary_metric": primary,
                    "foreground_logit_offset": evaluation.get("foreground_logit_offset", 0.0),
                    "optimizer_step": step,
                }
                state["selections"]["best-trained"] = selected
                if evaluation[primary] > state["selections"]["best"]["primary_score"]:
                    state["selections"]["best"] = selected
                write(output / f"selected-step-{step:06d}-evaluation.json", evaluation)
                write(output / f"selected-step-{step:06d}-predictions.coco.json", predictions)
            state["history"].append(record)
            state["epoch"] += 1
            state["order"] = []
            state["cursor"] = 0
            state["epoch_metrics"] = {}
            state["epoch_images"] = 0
            train_data.verify()
            val_data.verify()
            commit()
            _publish_selected(output, state["selections"])
            write(output / "history.json", state["history"])
            print(json.dumps(record), flush=True)
        elif state["optimizer_steps"] % 20 == 0:
            commit()
            print(
                json.dumps(
                    {
                        "event": "native_optimization",
                        "mode": mode,
                        "step": state["optimizer_steps"],
                        "optimization_seconds": state["optimization_seconds"],
                        **metrics,
                    }
                ),
                flush=True,
            )
    if wall_budget is not None and state["optimization_seconds"] < wall_budget:
        raise RuntimeError("Matched SFT walltime budget not reached before the safety step cap")
    initial = json.loads((output / "initial-evaluation.json").read_text())
    # Reload portable full-native selected weights independently before reporting.
    selected_detector = _model(output / "best-trained.safetensors", config)
    replay, predictions = evaluate_model(selected_detector.model.model, val_data, config)
    if abs(replay[primary] - state["selections"]["best-trained"]["primary_score"]) > 1e-10:
        raise ValueError("selected native safetensors failed independent validation replay")
    write(output / "trained-evaluation.json", replay)
    write(output / "trained-predictions.coco.json", predictions)
    summary = {
        "status": "completed",
        "integration_only": integration_only,
        "mode": mode,
        "optimizer_steps": state["optimizer_steps"],
        "optimization_seconds": state["optimization_seconds"],
        "optimization_timing_scope": (
            "Committed active preprocessing/forward/reward/backward/optimizer/drift-monitor time; "
            "excludes setup, validation, artifact IO and discarded uncommitted work after a crash."
        ),
        "wall_seconds_this_process": time.perf_counter() - started,
        "wall_timing_scope": "After dataset/model setup; includes validation and checkpoint I/O.",
        "initial_AP": initial["AP"],
        "initial_AP95": initial["AP95"],
        "trained_AP": replay["AP"],
        "trained_AP95": replay["AP95"],
        "AP_gain_over_initial": replay["AP"] - initial["AP"],
        "AP95_gain_over_initial": replay["AP95"] - initial["AP95"],
        "primary_metric": primary,
        "initial_primary_score": initial[primary],
        "trained_primary_score": replay[primary],
        "primary_gain_over_initial": replay[primary] - initial[primary],
        "selections": state["selections"],
        "reference_sha256": reference_sha,
        "initial_evaluation_sha256": sha(output / "initial-evaluation.json"),
        "trained_evaluation_sha256": sha(output / "trained-evaluation.json"),
        "trained_predictions_sha256": sha(output / "trained-predictions.coco.json"),
        "draw_chain_sha256": state["draw_chain_sha256"],
        "matched_prefix_draw_sha256": state["matched_prefix_draw_sha256"],
        "draw_chain_history": state["draw_chain_history"],
        "acceptance": acceptance,
        "matched_control": control_identity,
        "native_test_accessed": False,
        "primary_requirement_satisfied": False,
        "requirement_note": "Requires both matched SFT controls and the declared clustered CI.",
    }
    write(output / "summary.json", summary)
    return summary


def compare_wheat_controls(
    rl_run: Path,
    equal_updates_run: Path,
    equal_walltime_run: Path,
    strongest_sft_report: Path,
    output: Path,
):
    """Separate, fail-closed substantial-gain decision after all actual fit arms.

    Inputs are complete native validation reports. This never grants a test-set
    claim, and it does not correct validation checkpoint/calibration selection.
    """
    from vision_rl.wheat_eval import WHEAT_LOGIT_OFFSETS, paired_domain_comparison

    paths = {"rl": rl_run, "sft_updates": equal_updates_run, "sft_walltime": equal_walltime_run}
    plans, summaries, reports = {}, {}, {}
    for mode, path in paths.items():
        plan = json.loads((path / "plan.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        if (
            plan["mode"] != mode
            or summary["mode"] != mode
            or summary["status"] != "completed"
            or plan.get("integration_only")
            or summary.get("integration_only")
            or plan["configuration"]["reward_backend"] != "wheat_ada"
        ):
            raise ValueError(
                "comparison requires completed actual wheat RL and both matched SFT arms"
            )
        for key, filename in (
            ("trained_evaluation_sha256", "trained-evaluation.json"),
            ("initial_evaluation_sha256", "initial-evaluation.json"),
            ("trained_predictions_sha256", "trained-predictions.coco.json"),
        ):
            if summary[key] != sha(path / filename):
                raise ValueError("controlled-run evaluation evidence changed")
        if (
            sha(path / "best-trained.safetensors")
            != summary["selections"]["best-trained"]["sha256"]
        ):
            raise ValueError("controlled native model changed")
        evaluation = json.loads((path / "trained-evaluation.json").read_text())
        reports[mode] = evaluation["wheat"]
        plans[mode], summaries[mode] = plan, summary
    reference_plan, reference = plans["rl"], summaries["rl"]
    protocol_path = rl_run / "protocol.json"
    if sha(protocol_path) != reference_plan["protocol_sha256"]:
        raise ValueError("frozen experiment protocol changed")
    protocol = json.loads(protocol_path.read_text())
    for mode in ("sft_updates", "sft_walltime"):
        plan, summary = plans[mode], summaries[mode]
        if any(
            plan["configuration"][key] != reference_plan["configuration"][key]
            for key in CONTROL_CONFIG_FIELDS
        ):
            raise ValueError("matched SFT optimizer/data configuration differs")
        for key in (
            "reference_sha256",
            "train_identity",
            "validation_identity",
            "trainable_scope",
            "source_sha256",
            "software",
            "protocol_sha256",
            "drift_sentinel_image_ids",
        ):
            if plan[key] != reference_plan[key]:
                raise ValueError(f"matched SFT {key} differs")
        expected = {
            "run_plan_sha256": sha(rl_run / "plan.json"),
            "run_summary_sha256": sha(rl_run / "summary.json"),
            "RL_updates": reference["optimizer_steps"],
            "RL_optimization_seconds": reference["optimization_seconds"],
        }
        if summary["matched_control"] != expected or plan["matched_control"] != expected:
            raise ValueError("matched SFT is bound to a different RL run")
        prefix = min(summary["optimizer_steps"], reference["optimizer_steps"])
        if (
            len(summary["draw_chain_history"]) != summary["optimizer_steps"]
            or len(reference["draw_chain_history"]) != reference["optimizer_steps"]
        ):
            raise ValueError("controlled image/augmentation trace is incomplete")
        if summary["draw_chain_history"][:prefix] != reference["draw_chain_history"][:prefix]:
            raise ValueError("matched SFT image/augmentation sequence differs")
        if mode == "sft_updates" and summary["optimizer_steps"] != reference["optimizer_steps"]:
            raise ValueError("equal-update SFT did not receive the exact RL update count")
        if (
            mode == "sft_walltime"
            and summary["optimization_seconds"] < reference["optimization_seconds"]
        ):
            raise ValueError("matched SFT did not receive the measured committed RL active time")
    reports["initial"] = json.loads((rl_run / "initial-evaluation.json").read_text())["wheat"]
    strongest = json.loads(strongest_sft_report.read_text())
    reports["strongest_available_sft"] = strongest.get("wheat", strongest)
    count = protocol["data"]["validation_images"]
    for report in reports.values():
        calibration = report.get("calibration", {})
        if (
            report["scope"] != "complete_annotation_split"
            or report["n"] != count
            or set(row["foreground_logit_offset"] for row in calibration.get("grid", []))
            != set(WHEAT_LOGIT_OFFSETS)
            or calibration.get("selection_scope") != "complete_annotation_split"
        ):
            raise ValueError(
                "comparison requires full native validation and identical calibration search"
            )
    minimum = protocol["RL_acceptance"][
        "minimum_absolute_ADA_gain_over_initial_and_each_matched_SFT"
    ]
    comparisons = {
        name: paired_domain_comparison(reports["rl"], report, minimum_gain=minimum)
        for name, report in reports.items()
        if name != "rl"
    }
    result = {
        "benchmark_id": protocol["benchmark_id"],
        "primary_metric": "detection_acc_avg_dom",
        "comparisons": comparisons,
        "primary_requirement_satisfied": all(
            row["substantial_gain_and_positive_ci"] for row in comparisons.values()
        ),
        "native_test_accessed": False,
        "validation_selection_bias_corrected": False,
        "protocol_sha256": sha(protocol_path),
        "run_summary_sha256": {name: sha(path / "summary.json") for name, path in paths.items()},
        "strongest_sft_report_sha256": sha(strongest_sft_report),
        "timing_scope": reference["optimization_timing_scope"],
    }
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exploration = commands.add_parser("pilot")
    for name in ("source", "checkpoint", "output"):
        exploration.add_argument("--" + name, required=True, type=Path)
    exploration.add_argument("--config", type=Path)
    exploration.add_argument("--pixel-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    exploration.add_argument("--protocol", type=Path)
    fitting = commands.add_parser("train")
    for name in ("source", "checkpoint", "output"):
        fitting.add_argument("--" + name, required=True, type=Path)
    fitting.add_argument("--config", type=Path)
    fitting.add_argument("--mode", choices=("rl", "sft_updates", "sft_walltime"), default="rl")
    fitting.add_argument("--pilot-path", type=Path)
    fitting.add_argument("--control-run", type=Path)
    fitting.add_argument("--protocol", type=Path)
    fitting.add_argument("--resume", action="store_true")
    comparison = commands.add_parser("compare-wheat")
    for name in (
        "rl-run",
        "equal-updates-run",
        "equal-walltime-run",
        "strongest-sft-report",
        "output",
    ):
        comparison.add_argument("--" + name, required=True, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "compare-wheat":
        print(json.dumps(compare_wheat_controls(**args), indent=2))
        return
    config_path = args.pop("config")
    config = (
        PolicyTrainingConfig(**json.loads(config_path.read_text()))
        if config_path
        else PolicyTrainingConfig()
    )
    print(json.dumps((pilot if command == "pilot" else train)(config=config, **args), indent=2))


if __name__ == "__main__":
    main()
