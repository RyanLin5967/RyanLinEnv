"""Native COCO detection scoring and paired bootstrap without repeating IoU work.

RF100-VL uses COCO AP@[.50:.05:.95], 101 recall points and maxDets=500.
COCOeval.summarize() hardcodes 100 for its first metric, so metrics here are
extracted from the precision tensor explicitly. Bootstrap units may be capture
sessions; duplicated units reuse independent per-image matching results.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "coco-bbox/AP50:95/101-recall/maxDets500/v1"
MAX_DETECTIONS = 500


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _bbox(value: Any, image: dict, name: str, *, require_in_bounds: bool = True) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain four xywh coordinates")
    x, y, w, h = [_number(v, name) for v in value]
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} must have positive extent")
    if require_in_bounds and (
        x < 0 or y < 0 or x + w > image["width"] + 1e-5 or y + h > image["height"] + 1e-5
    ):
        raise ValueError(f"{name} extends outside the image")
    return [x, y, w, h]


def load_coco(annotation_path: Path | str) -> dict:
    """Validate bbox annotations and omit categories never annotated in the file."""
    document = json.loads(Path(annotation_path).read_text())
    if not isinstance(document, dict):
        raise ValueError("COCO document must be an object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(document.get(key), list):
            raise ValueError(f"COCO {key} must be a list")
    images, categories, annotation_ids = {}, {}, set()
    for row in document["images"]:
        image_id = _integer(row.get("id"), "image id")
        if image_id in images:
            raise ValueError("duplicate image id")
        for axis in ("width", "height"):
            if _integer(row.get(axis), axis) <= 0:
                raise ValueError("image dimensions must be positive")
        images[image_id] = row
    if not images:
        raise ValueError("COCO dataset must contain images")
    for category in document["categories"]:
        cid = _integer(category.get("id"), "category id")
        if cid in categories or not isinstance(category.get("name"), str):
            raise ValueError("duplicate or invalid category")
        categories[cid] = category
    active = set()
    for row in document["annotations"]:
        aid = _integer(row.get("id"), "annotation id")
        if aid in annotation_ids:
            raise ValueError("duplicate annotation id")
        annotation_ids.add(aid)
        iid = _integer(row.get("image_id"), "annotation image id")
        cid = _integer(row.get("category_id"), "annotation category id")
        if iid not in images or cid not in categories:
            raise ValueError("annotation references unknown image or category")
        row["bbox"] = _bbox(row.get("bbox"), images[iid], "annotation bbox")
        row["area"] = _number(row.get("area", row["bbox"][2] * row["bbox"][3]), "area")
        if row["area"] <= 0:
            raise ValueError("annotation area must be positive")
        if row.get("iscrowd", 0) not in (0, 1):
            raise ValueError("iscrowd must be 0 or 1")
        row["iscrowd"] = int(row.get("iscrowd", 0))
        active.add(cid)
    if not active:
        raise ValueError("COCO dataset must contain at least one annotated category")
    document["categories"] = [categories[c] for c in sorted(active)]
    document.setdefault("info", {})
    return document


def _label_map(categories: list[dict] | dict[str, int]) -> dict[str, int]:
    pairs = (
        categories.items()
        if isinstance(categories, dict)
        else ((c["name"], c["id"]) for c in categories)
    )
    result = {}
    for label, cid in pairs:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("category names must be nonempty strings")
        key = label.strip().casefold()
        if key in result:
            raise ValueError("category names must be unambiguous ignoring case")
        result[key] = _integer(cid, "category id")
    if not result:
        raise ValueError("at least one category is required")
    return result


def _json_detections(text: str) -> list:
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise ValueError("detection response must be text smaller than 2 MB")
    decoder, candidates, position = json.JSONDecoder(), [], 0
    while position < len(text):
        if text[position] not in "[{":
            position += 1
            continue
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            tail = text[position + 1 :].lstrip()
            if tail.startswith(('"', "{", "[", "]")):
                raise ValueError("response contains malformed or truncated JSON") from None
            position += 1
            continue
        position = end
        if isinstance(value, dict):
            keys = [k for k in ("detections", "objects", "predictions") if k in value]
            if len(keys) != 1:
                continue
            value = value[keys[0]]
        if isinstance(value, list):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("response must contain exactly one JSON detection array")
    return candidates[0]


def parse_detection_response(
    text: str,
    image: dict,
    categories: list[dict] | dict[str, int],
    *,
    score: float = 1.0,
    coordinate_format: str = "yxyx_1000",
) -> list[dict]:
    """Parse explicitly specified boxes; malformed responses raise, never become empty.

    Accept fences, surrounding prose, and one objects/detections/predictions
    envelope. Default yxyx_1000 uses normalized coordinates in 0..1000;
    xyxy_1000 uses normalized x1,y1,x2,y2 coordinates in 0..1000, while
    xyxy_pixels uses absolute x1,y1,x2,y2 pixel coordinates. named_normalized
    requires class_name and x_min/y_min/x_max/y_max fields in 0..1. The format is never
    inferred from values, and reversed coordinates are never repaired.
    Array formats accept exactly one of box, box_2d, or bbox_2d; field spelling
    never changes the caller-selected coordinate format.
    Empty JSON arrays are valid.
    """
    if coordinate_format not in ("yxyx_1000", "xyxy_1000", "xyxy_pixels", "named_normalized"):
        raise ValueError(
            "coordinate_format must be yxyx_1000, xyxy_1000, xyxy_pixels, or named_normalized"
        )
    labels = _label_map(categories)
    _integer(image.get("id"), "image id")
    for axis in ("width", "height"):
        if _number(image.get(axis), axis) <= 0:
            raise ValueError("image dimensions must be positive")
    default_score = _number(score, "score")
    if not 0 <= default_score <= 1:
        raise ValueError("score must be in 0..1")
    objects = _json_detections(text)
    if len(objects) > 10_000:
        raise ValueError("response contains too many detections")
    predictions = []
    for obj in objects:
        label_key = "class_name" if coordinate_format == "named_normalized" else "label"
        if not isinstance(obj, dict) or not isinstance(obj.get(label_key), str):
            raise ValueError(f"each detection requires a {label_key}")
        label = obj[label_key].strip().casefold()
        if label not in labels:
            raise ValueError("response contains an unknown class label")
        named_keys = ("x_min", "y_min", "x_max", "y_max")
        if coordinate_format == "named_normalized":
            if any(key in obj for key in ("box", "box_2d", "bbox_2d", "label")) or not all(
                key in obj for key in named_keys
            ):
                raise ValueError("named_normalized requires only named coordinates and class_name")
            coords = [obj[key] for key in named_keys]
        else:
            if any(key in obj for key in (*named_keys, "class_name")):
                raise ValueError("array box format cannot contain named coordinates or class_name")
            keys = [k for k in ("box", "box_2d", "bbox_2d") if k in obj]
            if len(keys) != 1:
                raise ValueError("each detection requires exactly one box, box_2d, or bbox_2d")
            coords = obj[keys[0]]
            if not isinstance(coords, list) or len(coords) != 4:
                raise ValueError("box must contain four coordinates")
        numbers = [_number(v, "coordinate") for v in coords]
        if coordinate_format in ("yxyx_1000", "xyxy_1000", "named_normalized"):
            if coordinate_format == "yxyx_1000":
                ymin, xmin, ymax, xmax = numbers
            else:
                xmin, ymin, xmax, ymax = numbers
            scale = 1 if coordinate_format == "named_normalized" else 1000
            if not (0 <= ymin < ymax <= scale and 0 <= xmin < xmax <= scale):
                order = "yxyx" if coordinate_format == "yxyx_1000" else "xyxy"
                raise ValueError(f"box must be ordered {order} with coordinates in 0..{scale}")
            bbox = [
                xmin * image["width"] / scale,
                ymin * image["height"] / scale,
                (xmax - xmin) * image["width"] / scale,
                (ymax - ymin) * image["height"] / scale,
            ]
        else:
            xmin, ymin, xmax, ymax = numbers
            if not (0 <= xmin < xmax <= image["width"] and 0 <= ymin < ymax <= image["height"]):
                raise ValueError("pixel box must be ordered xyxy within the image")
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        confidence = _number(obj.get("score", obj.get("confidence", default_score)), "score")
        if not 0 <= confidence <= 1:
            raise ValueError("score must be in 0..1")
        predictions.append(
            {
                "image_id": image["id"],
                "category_id": labels[label],
                "score": confidence,
                "bbox": bbox,
            }
        )
    return predictions


parse_detections = parse_detection_response


def detection_prompt(categories: list[dict] | dict[str, int], rich: bool = True) -> str:
    labels = list(_label_map(categories))
    detail = ""
    if rich and labels == ["pill"]:
        detail = (
            " Detect every visible individual pill/tablet. Include its full visible extent; "
            "for partly obscured pills estimate the full shape from visible portions. "
            "Exclude shadows, reflections, and objects too indistinct to identify as pills."
        )
    return (
        f"Locate every object belonging to these classes: {json.dumps(labels)}.{detail} "
        "Return one JSON array, with one object per detection: "
        '{"label":"class name","box":[ymin,xmin,ymax,xmax],"score":0.95}. '
        "Coordinates must be normalized to 0..1000 relative to the entire image, "
        "in ymin,xmin,ymax,xmax order. Score is your confidence from 0 to 1. "
        "Use only the listed labels. Return [] if no objects are present."
    )


def _write(report: dict, output: Path | str | None) -> dict:
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _mean_valid(values: np.ndarray) -> float | None:
    values = values[values >= 0]
    return float(values.mean()) if len(values) else None


def _cache(evaluator: Any, image_ids: list[int]) -> dict:
    lookup = {
        (e["image_id"], e["category_id"]): e
        for e in evaluator.evalImgs
        if e is not None and list(e["aRng"]) == list(evaluator.params.areaRng[0])
    }
    items = []
    for image_id in image_ids:
        records = []
        for cid in evaluator.params.catIds:
            e = lookup.get((image_id, cid))
            if e is None:
                continue
            records.append(
                {
                    "category_id": int(cid),
                    "scores": [float(s) for s in e["dtScores"]],
                    "matches": (np.asarray(e["dtMatches"]) > 0).tolist(),
                    "ignored_detections": np.asarray(e["dtIgnore"], dtype=bool).tolist(),
                    "ignored_ground_truth": np.asarray(e["gtIgnore"], dtype=bool).tolist(),
                }
            )
        items.append({"image_id": image_id, "categories": records})
    return {
        "iou_thresholds": evaluator.params.iouThrs.tolist(),
        "recall_thresholds": evaluator.params.recThrs.tolist(),
        "category_ids": [int(c) for c in evaluator.params.catIds],
        "items": items,
    }


def _counts(records: list[dict]) -> dict:
    tp = fp = ground_truth = 0
    for record in records:
        matched = np.asarray(record["matches"], dtype=bool)[0]
        ignored = np.asarray(record["ignored_detections"], dtype=bool)[0]
        tp += int(np.count_nonzero(matched & ~ignored))
        fp += int(np.count_nonzero(~matched & ~ignored))
        ground_truth += sum(not v for v in record["ignored_ground_truth"])
    return {
        "tp": tp,
        "fp": fp,
        "fn": ground_truth - tp,
        "ground_truth": ground_truth,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / ground_truth if ground_truth else None,
    }


def evaluate_coco(
    annotation_path: Path | str,
    predictions: list[dict],
    output: Path | str | None = None,
    image_ids: list[int] | None = None,
    model: dict | None = None,
) -> dict:
    """Evaluate all declared/selected images, including ones with zero detections.

    An absent prediction list scores as zero detections, never removes an image.
    Record transport/parse failures separately in caller provenance. Invalid COCO
    prediction objects raise ValueError rather than being silently dropped. COCO
    permits predictions extending beyond image edges; their original finite
    geometry is preserved without clipping.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    document = load_coco(annotation_path)
    images = {im["id"]: im for im in document["images"]}
    categories = {c["id"] for c in document["categories"]}
    selected = sorted(images) if image_ids is None else list(image_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("image_ids must be nonempty and unique")
    for iid in selected:
        if _integer(iid, "selected image id") not in images:
            raise ValueError("selected image id is absent from annotations")
    selected = sorted(selected)
    selected_set = set(selected)
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    normalized = []
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError("each prediction must be an object")
        iid = _integer(pred.get("image_id"), "prediction image id")
        cid = _integer(pred.get("category_id"), "prediction category id")
        if iid not in selected_set or cid not in categories:
            raise ValueError("prediction references unknown/unselected image or inactive category")
        score = _number(pred.get("score"), "score")
        if not 0 <= score <= 1:
            raise ValueError("prediction score must be in 0..1")
        normalized.append(
            {
                "image_id": iid,
                "category_id": cid,
                "score": score,
                "bbox": _bbox(
                    pred.get("bbox"), images[iid], "prediction bbox", require_in_bounds=False
                ),
            }
        )
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = copy.deepcopy(document)
        # pycocotools reserves match ID zero as the unmatched sentinel. Some
        # valid Roboflow exports number annotations from zero; reindex safely.
        for index, annotation in enumerate(truth.dataset["annotations"], 1):
            annotation["id"] = index
        truth.createIndex()
        # loadRes([]) indexes its first element and fails; build an empty COCO instead.
        if normalized:
            detected = truth.loadRes(copy.deepcopy(normalized))
        else:
            detected = COCO()
            detected.dataset = {
                "images": copy.deepcopy(document["images"]),
                "categories": copy.deepcopy(document["categories"]),
                "annotations": [],
            }
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = selected
        evaluator.params.catIds = sorted(categories)
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    cache = _cache(evaluator, selected)
    items = []
    by_image: dict[int, list[dict]] = {iid: [] for iid in selected}
    for pred in normalized:
        by_image[pred["image_id"]].append(pred)
    for item in cache["items"]:
        iid = item["image_id"]
        items.append(
            {
                "image_id": iid,
                "file_name": images[iid].get("file_name"),
                "counts_at_iou50": _counts(item["categories"]),
                "predictions": by_image[iid],
            }
        )
    source_hash = hashlib.sha256(Path(annotation_path).read_bytes()).hexdigest()
    # Content-derived identity includes declared images/labels, selected IDs and active
    # classes; it deliberately does not imply that image pixels were independently hashed.
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256((canonical + json.dumps(selected)).encode()).hexdigest()
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model or {},
        "dataset_fingerprint": identity,
        "fingerprint_scope": "COCO annotations and selected IDs",
        "source_annotation_sha256": source_hash,
        "image_ids": selected,
        "n": len(selected),
        "categories": document["categories"],
        "max_dets": MAX_DETECTIONS,
        "AP": _mean_valid(precision),
        "AP50": _mean_valid(precision[0]),
        "AP75": _mean_valid(precision[5]),
        "AR": _mean_valid(evaluator.eval["recall"][:, :, 0, -1]),
        "counts_at_iou50": _counts([r for i in cache["items"] for r in i["categories"]]),
        "count_protocol": (
            "IoU=0.50, all scores, maxDets=500 per image/category; COCO crowd ignores"
        ),
        "prediction_count": len(normalized),
        "images_without_predictions": sum(not by_image[iid] for iid in selected),
        "items": items,
        "match_cache": cache,
    }
    return _write(report, output)


def _prepared_cache(cache: dict) -> dict:
    """Convert serialized matches once before the bootstrap loop."""
    result = {
        "category_ids": cache["category_ids"],
        "recall_thresholds": np.asarray(cache["recall_thresholds"]),
        "iou_thresholds": np.asarray(cache["iou_thresholds"]),
        "items": {},
    }
    for item in cache["items"]:
        result["items"][item["image_id"]] = {
            c["category_id"]: {
                "scores": np.asarray(c["scores"], dtype=float),
                "matches": np.asarray(c["matches"], dtype=bool),
                "ignored_detections": np.asarray(c["ignored_detections"], dtype=bool),
                "ground_truth": sum(not v for v in c["ignored_ground_truth"]),
            }
            for c in item["categories"]
        }
    return result


def _cached_ap(prepared: dict, image_ids: list[int]) -> float | None:
    """COCO precision interpolation from cached independent-image match results."""
    category_means = []
    for cid in prepared["category_ids"]:
        entries = [
            prepared["items"][iid][cid] for iid in image_ids if cid in prepared["items"][iid]
        ]
        gt = sum(e["ground_truth"] for e in entries)
        if not gt:
            continue
        scores = np.concatenate([e["scores"] for e in entries])
        order = np.argsort(-scores, kind="mergesort")
        matches = np.concatenate([e["matches"] for e in entries], axis=1)[:, order]
        ignored = np.concatenate([e["ignored_detections"] for e in entries], axis=1)[:, order]
        tp = np.cumsum(matches & ~ignored, axis=1, dtype=float)
        fp = np.cumsum(~matches & ~ignored, axis=1, dtype=float)
        recall = tp / gt
        precision = tp / (tp + fp + np.spacing(1))
        if precision.shape[1]:
            precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
        interpolated = np.zeros(
            (len(prepared["iou_thresholds"]), len(prepared["recall_thresholds"]))
        )
        for t in range(len(interpolated)):
            indices = np.searchsorted(recall[t], prepared["recall_thresholds"], side="left")
            valid = indices < precision.shape[1]
            interpolated[t, valid] = precision[t, indices[valid]]
        category_means.append(float(interpolated.mean()))
    return float(np.mean(category_means)) if category_means else None


def _evaluation_error_count(report: dict) -> int:
    """Read wrapper failure provenance; absent fields are valid for local detectors.

    The count and failure list can describe the same requests, so take the maximum
    rather than double-counting. A malformed supplied field fails closed.
    """
    errors = report.get("errors", 0)
    if isinstance(errors, bool) or not isinstance(errors, int) or errors < 0:
        raise ValueError("evaluation errors must be a nonnegative integer")
    failures = report.get("request_or_parse_failures", [])
    if not isinstance(failures, list):
        raise ValueError("request_or_parse_failures must be a list")
    return max(errors, len(failures))


def compare_detections(
    a: dict | Path | str,
    b: dict | Path | str,
    groups: dict | None = None,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    margin: float = 0.20,
    min_ci_lower: float = 0.10,
    output: Path | str | None = None,
) -> dict:
    """Paired AP difference (a - b), resampling capture groups or independent images.

    Group bootstrap samples groups uniformly with replacement and includes every
    image in each selected group. All models receive the same resampled image list.
    Intervals are percentile 95% CIs, not repeated-test or selection corrections.
    Wrapper transport/parse errors invalidate comparisons and cannot count as a
    frontier win. comparison_valid checks evaluation provenance, not label audits.
    """
    a = json.loads(Path(a).read_text()) if isinstance(a, (Path, str)) else a
    b = json.loads(Path(b).read_text()) if isinstance(b, (Path, str)) else b
    for key in ("protocol", "dataset_fingerprint", "image_ids", "categories", "max_dets"):
        if a.get(key) != b.get(key):
            raise ValueError(f"detection comparisons require matching {key}")
    if a.get("protocol") != PROTOCOL or a.get("max_dets") != MAX_DETECTIONS:
        raise ValueError("unsupported detection evaluation protocol")
    if bootstrap_samples < 1 or isinstance(bootstrap_samples, bool):
        raise ValueError("bootstrap_samples must be a positive integer")
    _integer(bootstrap_samples, "bootstrap_samples")
    margin, min_ci_lower = _number(margin, "margin"), _number(min_ci_lower, "min_ci_lower")
    if margin < 0 or min_ci_lower < 0:
        raise ValueError("required advantage margins must be nonnegative")
    evaluation_errors = {"a": _evaluation_error_count(a), "b": _evaluation_error_count(b)}
    ids = a["image_ids"]
    if groups is None:
        members = [[iid] for iid in ids]
        method = "paired independent-image percentile bootstrap"
    else:
        normalized_groups = {}
        for iid in ids:
            if iid not in groups and str(iid) not in groups:
                raise ValueError("groups must cover every evaluated image")
            group = groups.get(iid, groups.get(str(iid)))
            if not isinstance(group, (str, int)) or isinstance(group, bool):
                raise ValueError("group identifiers must be strings or integers")
            normalized_groups.setdefault((type(group).__name__, group), []).append(iid)
        members = list(normalized_groups.values())
        method = "paired capture-group percentile bootstrap"
    if len(members) < 2:
        raise ValueError("at least two independent resampling units are required")
    pa, pb = _prepared_cache(a["match_cache"]), _prepared_cache(b["match_cache"])
    point_a, point_b = _cached_ap(pa, ids), _cached_ap(pb, ids)
    if point_a is None or point_b is None:
        raise ValueError("AP is undefined without nonignored ground truth")
    # Reject stale cache/report pairs before publishing uncertainty for wrong scores.
    if not np.isclose(point_a, a["AP"], atol=1e-12) or not np.isclose(point_b, b["AP"], atol=1e-12):
        raise ValueError("reported AP does not match cached matching data")
    rng, differences, undefined = np.random.default_rng(seed), [], 0
    for _ in range(bootstrap_samples):
        sampled = [
            iid for unit in rng.integers(0, len(members), len(members)) for iid in members[unit]
        ]
        first, second = _cached_ap(pa, sampled), _cached_ap(pb, sampled)
        if first is None or second is None:
            undefined += 1
        else:
            differences.append(first - second)
    if not differences:
        raise ValueError("every bootstrap sample has undefined AP")
    ci = np.quantile(differences, [0.025, 0.975]).tolist()
    difference = point_a - point_b
    synthetic_oracle_evidence = {
        name: bool(
            report.get("synthetic_oracle_bound")
            or report.get("evaluation_kind") == "annotation_oracle_upper_bound"
        )
        for name, report in (("a", a), ("b", b))
    }
    comparison_valid = (
        not any(evaluation_errors.values())
        and undefined == 0
        and not any(synthetic_oracle_evidence.values())
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset_fingerprint": a["dataset_fingerprint"],
        "image_ids": ids,
        "n": len(ids),
        "model_a": a.get("model"),
        "model_b": b.get("model"),
        "AP_a": point_a,
        "AP_b": point_b,
        "AP_difference": difference,
        "AP_difference_ci95": ci,
        "bootstrap_method": method,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_defined_samples": len(differences),
        "bootstrap_undefined_samples": undefined,
        "bootstrap_seed": seed,
        "resampling_units": len(members),
        "group_sizes": [len(m) for m in members],
        "required_margin": margin,
        "required_ci_lower": min_ci_lower,
        "evaluation_errors": evaluation_errors,
        "synthetic_oracle_evidence": synthetic_oracle_evidence,
        "comparison_valid": comparison_valid,
        "margin_target_met": bool(
            difference >= margin and ci[0] >= min_ci_lower and comparison_valid
        ),
        "limitation": (
            "Conditional on this dataset and model selection; annotation quality "
            "and capture independence require separate audit."
        ),
    }
    return _write(report, output)
