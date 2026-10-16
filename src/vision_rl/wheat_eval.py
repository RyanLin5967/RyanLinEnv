"""Pinned WILDS ADA scoring for the separately named author-HF wheat protocol.

This module calls the unchanged official matcher and equal-domain evaluator.
It does not assert competition-leaderboard or WILDS-native dataset equivalence.
Optional WILDS dependencies are loaded only when scoring is requested.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

BENCHMARK_ID = "gwhd2021-hf-domain-ada-v1"
PROTOCOL = "wilds2.0/ADA/score>0.5/IoU>=0.5/maxDets500/v1"
WHEAT_LOGIT_OFFSETS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
SOURCE_HASHES = {
    "DetectionAccuracy": "4fe677510f6e7e11005e0e73f16189c5a5c8a5664fe126055426dcbe5c873d72",
    "GlobalWheatDataset": "25948859e68e6cb4759709bc0158db060c5ab0d15aa1a467ac1f41383e7fcc1e",
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@lru_cache(maxsize=1)
def _backend():
    try:
        import torch
        import torchvision
        import wilds
        from wilds.common.grouper import CombinatorialGrouper
        from wilds.common.metrics.all_metrics import DetectionAccuracy
        from wilds.datasets.globalwheat_dataset import SESSIONS, GlobalWheatDataset
    except ImportError as error:
        raise RuntimeError(
            "Wheat ADA requires wilds==2.0.0, torch-scatter==2.1.2, torch and torchvision; "
            "see docs/wheat-data-audit-2026-10-16.md for the isolated scorer environment"
        ) from error
    hashes = {
        cls.__name__: _digest(Path(inspect.getfile(cls)).read_bytes())
        for cls in (DetectionAccuracy, GlobalWheatDataset)
    }
    if wilds.__version__ != "2.0.0" or hashes != SOURCE_HASHES:
        raise ValueError("Wheat metric source differs from the pinned official WILDS 2.0.0")
    # The complete domain catalogue defines group IDs; no images are synthesized or loaded.
    catalogue = SimpleNamespace(
        metadata_fields=["session"],
        metadata_map={"session": SESSIONS},
        metadata_array=torch.arange(len(SESSIONS), dtype=torch.long).reshape(-1, 1),
    )
    adapter = GlobalWheatDataset.__new__(GlobalWheatDataset)
    adapter._metric = DetectionAccuracy()
    adapter._eval_grouper = CombinatorialGrouper(catalogue, ["session"])
    provenance = {
        "library": "wilds",
        "version": wilds.__version__,
        "source_file_sha256": hashes,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_scatter": version("torch-scatter"),
        "geometry_dtype": "float32",
        "device": "cpu",
        "entry_point": "wilds.datasets.globalwheat_dataset.GlobalWheatDataset.eval",
        "adapter_sha256": _digest(Path(__file__).read_bytes()),
    }
    return torch, adapter, tuple(SESSIONS), provenance


def official_sessions() -> tuple[str, ...]:
    """Pinned domain order used by the unchanged official library."""
    return _backend()[2]


def _tensor_boxes(boxes, *, xywh=False):
    torch = _backend()[0]
    tensor = torch.as_tensor(boxes).detach().to(device="cpu", dtype=torch.float32)
    if tensor.numel() == 0:
        tensor = tensor.reshape(0, 4)
    if tensor.ndim != 2 or tensor.shape[1] != 4:
        raise ValueError("Boxes must have shape [N,4]")
    tensor = tensor.clone()
    if xywh:
        tensor[:, 2:] += tensor[:, :2]
    if not torch.isfinite(tensor).all() or (tensor[:, 2:] <= tensor[:, :2]).any():
        raise ValueError("Box coordinates must be finite with positive area in float32")
    return tensor


def official_image_rewards(boxes, scores, targets):
    """Return CPU float64 [G] rewards from XYXY [G,Q,4], scores [G,Q], GT [T,4].

    Geometry may be normalized or in pixels, but must share a coordinate space.
    Scores are already calibrated by the caller; the native >0.5 gate is fixed.
    This is a nondifferentiable reward; inputs are detached and never modified.
    """
    torch, adapter, _, _ = _backend()
    boxes = torch.as_tensor(boxes).detach().cpu()
    scores = torch.as_tensor(scores).detach().to(device="cpu", dtype=torch.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or scores.shape != boxes.shape[:2]:
        raise ValueError("Expected boxes [G,Q,4] and scores [G,Q]")
    if boxes.shape[1] > 500:
        raise ValueError("Wheat protocol permits at most 500 predictions per image")
    if not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities")
    truth = {"boxes": _tensor_boxes(targets)}
    if not len(boxes):
        return torch.empty(0, dtype=torch.float64)
    predictions = [
        {"boxes": _tensor_boxes(item), "scores": score}
        for item, score in zip(boxes, scores, strict=True)
    ]
    return adapter._metric._compute_element_wise(predictions, [truth] * len(boxes)).double()


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain four XYWH coordinates")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError("bbox must contain finite numbers")
    if value[2] <= 0 or value[3] <= 0:
        raise ValueError("bbox must have positive area")
    # Full extents of truncated objects are retained, including out-of-frame GT.
    return list(value)


def validate_wheat_coco(document: dict, predictions: list[dict], image_ids=None):
    """Validate without dropping empty images, clipping boxes or deduplicating GT."""
    if [item.get("id") for item in document.get("categories", [])] != [1]:
        raise ValueError("Wheat requires exactly COCO category 1")
    images, truth, seen_annotations = {}, defaultdict(list), set()
    for image in document["images"]:
        image_id = _integer(image["id"], "image ID")
        if image_id in images:
            raise ValueError("Duplicate image ID")
        if any(_integer(image[key], key) <= 0 for key in ("width", "height")):
            raise ValueError("Image dimensions must be positive")
        if not isinstance(image.get("domain"), str) or not image["domain"]:
            raise ValueError("Every image requires a domain string")
        images[image_id] = image
    selected = list(images) if image_ids is None else list(image_ids)
    for image_id in selected:
        _integer(image_id, "selected image ID")
    if not selected or len(selected) != len(set(selected)) or set(selected) - images.keys():
        raise ValueError("Image selection must contain unique known IDs and cannot be empty")
    selected = sorted(selected)
    for annotation in document["annotations"]:
        annotation_id = _integer(annotation["id"], "annotation ID")
        image_id = _integer(annotation["image_id"], "annotation image ID")
        if annotation_id in seen_annotations or image_id not in images:
            raise ValueError("Duplicate annotation ID or unknown annotation image")
        if _integer(annotation["category_id"], "category ID") != 1:
            raise ValueError("Unknown annotation category")
        if annotation.get("iscrowd", 0) != 0 or annotation.get("ignore", 0) != 0:
            raise ValueError("Wheat ADA has no crowd/ignore annotation semantics")
        seen_annotations.add(annotation_id)
        truth[image_id].append(_box(annotation["bbox"]))
    output = defaultdict(list)
    selected_set = set(selected)
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list of COCO detections")
    for prediction in predictions:
        image_id = _integer(prediction["image_id"], "prediction image ID")
        if image_id not in selected_set:
            raise ValueError("Prediction refers to an unknown or unevaluated image")
        if _integer(prediction["category_id"], "prediction category ID") != 1:
            raise ValueError("Unknown prediction category")
        score = prediction.get("score", 1.0)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction score must be a finite probability")
        output[image_id].append({"bbox": _box(prediction["bbox"]), "score": score})
        if len(output[image_id]) > 500:
            raise ValueError("Wheat protocol permits at most 500 predictions per image")
    return images, truth, output, selected


def evaluate_wheat(
    annotation_path,
    predictions,
    *,
    output=None,
    image_ids=None,
    model=None,
    errors=0,
    request_or_parse_failures=None,
):
    """Score COCO XYWH detections using native WILDS matching and domain averaging.

    Every selected image contributes, including empty images and empty predictions.
    Parsing/request failures must be supplied explicitly; they invalidate comparisons.
    Missing confidence values are assigned 1.0, as frozen in the HF protocol.
    """
    path = Path(annotation_path)
    raw = path.read_bytes()
    document = json.loads(raw)
    images, truth, predictions_by_id, selected = validate_wheat_coco(
        document, predictions, image_ids
    )
    _integer(errors, "errors")
    if errors < 0:
        raise ValueError("errors cannot be negative")
    failures = list(request_or_parse_failures or [])
    errors = max(errors, len(failures))
    torch, adapter, sessions, provenance = _backend()
    domains = {image_id: images[image_id]["domain"] for image_id in selected}
    if any(domain not in sessions for domain in domains.values()):
        raise ValueError("Unknown domain outside the pinned official session catalogue")
    for image_id in selected:
        declared_session = images[image_id].get("session_id")
        if declared_session is not None and declared_session != sessions.index(domains[image_id]):
            raise ValueError("Domain/session_id mismatch")
    y_true = [{"boxes": _tensor_boxes(truth[image_id], xywh=True)} for image_id in selected]
    y_pred = [
        {
            "boxes": _tensor_boxes(
                [item["bbox"] for item in predictions_by_id[image_id]], xywh=True
            ),
            "scores": torch.tensor(
                [item["score"] for item in predictions_by_id[image_id]], dtype=torch.float32
            ),
        }
        for image_id in selected
    ]
    metadata = torch.tensor([[sessions.index(domains[image_id])] for image_id in selected])
    native, _ = adapter.eval(y_pred, y_true, metadata)
    per_image_scores = adapter._metric._compute_element_wise(y_pred, y_true).tolist()
    per_domain = {}
    for domain in sorted(set(domains.values())):
        field = adapter._eval_grouper.group_field_str(sessions.index(domain))
        members = [image_id for image_id in selected if domains[image_id] == domain]
        per_domain[domain] = {
            "score": float(native[f"detection_acc_{field}"]),
            "n": int(native[f"count_{field}"]),
            "image_ids": members,
        }
    report = {
        "protocol": PROTOCOL,
        "benchmark_id": document.get("info", {}).get("benchmark_id"),
        "primary_key": "detection_acc_avg_dom",
        "metrics": {
            key: float(native[key])
            for key in ("detection_acc_avg_dom", "detection_acc_avg", "detection_acc_wg")
        },
        "n": len(selected),
        "image_ids": selected,
        "domains": len(per_domain),
        "scope": "complete_annotation_split" if len(selected) == len(images) else "subset",
        "per_domain": per_domain,
        "per_image": [
            {
                "image_id": image_id,
                "domain": domains[image_id],
                "score": score,
                "gt_count": len(truth[image_id]),
                "prediction_count": len(predictions_by_id[image_id]),
            }
            for image_id, score in zip(selected, per_image_scores, strict=True)
        ],
        "annotation_sha256": _digest(raw),
        "dataset_fingerprint": _digest(_canonical({"document": document, "image_ids": selected})),
        "predictions_sha256": _digest(_canonical(predictions)),
        "scorer": provenance,
        "model": model or {},
        "errors": errors,
        "request_or_parse_failures": failures,
        "evaluation_valid": errors == 0,
        "competition_leaderboard_equivalence_claimed": False,
        "wilds_native_dataset_equivalence_claimed": False,
    }
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def calibrate_wheat_predictions(predictions, offset):
    """Apply a frozen foreground logit offset without reordering or dropping boxes."""
    if type(offset) not in (int, float) or not math.isfinite(offset) or abs(offset) > 20:
        raise ValueError("Foreground logit offset must be finite and within [-20,20]")
    odds = math.exp(offset)
    output = []
    for row in predictions:
        score = row.get("score", 1.0)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction score must be a finite probability")
        output.append({**row, "score": score * odds / (1 - score + score * odds)})
    return output


def calibrated_wheat_evaluation(
    annotation_path, predictions, image_ids=None, *, model=None, output=None
):
    """Development-only fixed calibration grid shared by initialization, SFT and RL.

    Ties prefer the smallest absolute offset, then the numerically smaller offset.
    The chosen offset must be frozen and applied with calibrate_wheat_predictions
    for final testing; this search refuses explicitly marked native test data.
    """
    document = json.loads(Path(annotation_path).read_text())
    if document.get("info", {}).get("native_split") == "test":
        raise ValueError("Confidence selection is forbidden on sealed test annotations")
    records, best = [], None
    for offset in sorted(WHEAT_LOGIT_OFFSETS, key=lambda value: (abs(value), value)):
        calibrated = calibrate_wheat_predictions(predictions, offset)
        report = evaluate_wheat(annotation_path, calibrated, image_ids=image_ids, model=model)
        score = report["metrics"]["detection_acc_avg_dom"]
        records.append({"foreground_logit_offset": offset, "ADA": score})
        if best is None or score > best[0]:
            best = (score, report, calibrated, offset)
    _, report, calibrated, offset = best
    report["calibration"] = {
        "foreground_logit_offset": offset,
        "grid": records,
        "selection_metric": "validation detection_acc_avg_dom",
        "selection_scope": report["scope"],
        "selection_image_ids": report["image_ids"],
        "validation_selected": True,
        "must_freeze_before_test": True,
        "tie_break": "smallest absolute offset, then numerically smaller offset",
        "raw_zero_offset_ADA": next(
            row["ADA"] for row in records if row["foreground_logit_offset"] == 0
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report, calibrated


def paired_domain_comparison(candidate, baseline, *, resamples=20000, seed=2718, minimum_gain=0.05):
    """Paired domain-cluster percentile CI; each held-out acquisition domain has equal weight.

    Resampling is over complete paired domains, not individual images. This estimates
    uncertainty across observed domains; it does not cure validation selection bias.
    """
    import numpy as np

    for report in (candidate, baseline):
        if (
            not report.get("evaluation_valid")
            or report.get("errors", 0)
            or report.get("request_or_parse_failures")
        ):
            raise ValueError("Incomplete evaluations cannot support a comparison")
        if (
            report.get("model", {}).get("oracle")
            or report.get("model", {}).get("synthetic")
            or report.get("synthetic_oracle_bound")
            or report.get("evaluation_kind") == "annotation_oracle_upper_bound"
        ):
            raise ValueError("Oracle/synthetic model outputs cannot support a model win")
    for key in ("protocol", "dataset_fingerprint", "image_ids", "scope"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"Comparison requires identical {key}")
    if candidate["protocol"] != PROTOCOL:
        raise ValueError("Comparison requires the wheat ADA protocol")
    if candidate["scorer"] != baseline["scorer"]:
        raise ValueError("Comparison requires identical scorer provenance")
    domains = sorted(candidate["per_domain"])
    if domains != sorted(baseline["per_domain"]) or len(domains) < 2:
        raise ValueError("Comparison requires the same two or more domains")
    differences = []
    for domain in domains:
        first, second = candidate["per_domain"][domain], baseline["per_domain"][domain]
        if first["image_ids"] != second["image_ids"] or first["n"] != second["n"]:
            raise ValueError("Domain membership mismatch")
        if first["n"] != len(first["image_ids"]) or first["n"] < 1:
            raise ValueError("Invalid domain count")
        if any(
            not math.isfinite(item["score"]) or not 0 <= item["score"] <= 1
            for item in (first, second)
        ):
            raise ValueError("Invalid domain score")
        differences.append(first["score"] - second["score"])
    if type(resamples) is not int or resamples < 1000:
        raise ValueError("At least 1000 domain resamples are required")
    if not math.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("Minimum gain must be finite and nonnegative")
    values = np.asarray(differences, dtype=np.float64)
    draws = np.random.default_rng(seed).integers(0, len(domains), (resamples, len(domains)))
    distribution = values[draws].mean(axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    gain = float(values.mean())
    return {
        "metric": "detection_acc_avg_dom",
        "delta": gain,
        "ci95": [float(lower), float(upper)],
        "n_domains": len(domains),
        "per_domain_delta": dict(zip(domains, differences, strict=True)),
        "method": "paired acquisition-domain cluster bootstrap percentile interval",
        "resamples": resamples,
        "seed": seed,
        "minimum_gain": minimum_gain,
        "substantial_gain_and_positive_ci": bool(gain >= minimum_gain and lower > 0),
        "selection_bias_corrected": False,
    }
