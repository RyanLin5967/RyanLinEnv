"""Optimistic missing-response bounds, never a replacement frontier measurement.

For single-class COCO AP without ignored/crowd annotations, a failed image can
contribute at most one true positive per ground-truth object at each IoU level.
Perfect ground-truth boxes, no false positives, and ranking these detections
before every completed-image detection therefore upper-bound any possible
completion of those failures. Moving a true positive earlier cannot decrease
the interpolated precision envelope. This construction uses labels deliberately.

The primary report and its zero-error gate remain unchanged. These reports retain
the original errors and carry an explicit synthetic-oracle marker. Bootstrap
intervals describe the optimistic counterfactual conditional on the observed
completed responses; they do not estimate the failed provider's actual answers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .detection_eval import MAX_DETECTIONS, PROTOCOL, compare_detections, evaluate_coco, load_coco

KIND = "annotation_oracle_upper_bound"


def _read_report(report: dict | Path | str) -> dict:
    if isinstance(report, (Path, str)):
        report = json.loads(Path(report).read_text())
    if not isinstance(report, dict):
        raise ValueError("Evaluation report must be an object")
    return report


def _write(report: dict, output: Path | str | None) -> dict:
    if output is not None:
        path = Path(output)
        if path.exists():
            raise ValueError("Sensitivity evidence already exists; choose a fresh path")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def frontier_failure_upper_bound(
    annotation_path: Path | str,
    frontier_report: dict | Path | str,
    *,
    output: Path | str | None = None,
) -> dict:
    """Replace failed images with impossibly favorable, annotation-derived answers.

    Require native, single-class, nonignored annotations with at most 500 objects
    per failed image. Completed-image predictions and score ordering are fixed.
    Their scores are mapped monotonically below .5, preserving every tie exactly;
    oracle detections receive score 1. Rank encoding avoids floating underflow
    from multiplying arbitrarily small confidence values by a scale factor.

    Failure IDs must be the complete, unique failure list in the original wrapper
    report. The original score is independently replayed before constructing the
    bound. The original report, detections, and primary failure gate are untouched.
    """
    observed = _read_report(frontier_report)
    if observed.get("synthetic_oracle_bound") or observed.get("evaluation_kind") == KIND:
        raise ValueError("An oracle bound cannot be used as an observed frontier report")
    if observed.get("protocol") != PROTOCOL or observed.get("max_dets") != MAX_DETECTIONS:
        raise ValueError("Sensitivity analysis requires the native maxDets500 protocol")
    document = load_coco(annotation_path)
    if len(document["categories"]) != 1:
        raise ValueError("This conservative bound supports exactly one foreground class")
    if any(row.get("iscrowd", 0) or row.get("ignore", 0) for row in document["annotations"]):
        raise ValueError("Crowd or ignored annotations require a different bound proof")
    selected = observed.get("image_ids")
    if (
        not isinstance(selected, list)
        or not selected
        or any(type(iid) is not int for iid in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Observed evaluation must declare unique integer image IDs")
    selected_set = set(selected)
    failures, error_count = observed.get("request_or_parse_failures"), observed.get("errors")
    if (
        not isinstance(failures, list)
        or type(error_count) is not int
        or error_count != len(failures)
    ):
        raise ValueError("Errors must exactly match the complete per-image failure list")
    failed = set()
    for row in failures:
        if not isinstance(row, dict) or type(row.get("id")) not in (str, int):
            raise ValueError("Every failure must identify its native image ID")
        try:
            iid = int(row["id"])
        except ValueError as error:
            raise ValueError("Failure image IDs must be native integers") from error
        if str(iid) != str(row["id"]) or iid not in selected_set or iid in failed:
            raise ValueError("Failure IDs must be unique evaluated native image IDs")
        failed.add(iid)
    items = observed.get("items")
    if not isinstance(items, list) or len(items) != len(selected):
        raise ValueError("Observed per-image predictions must cover every evaluated image")
    predictions, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or type(item.get("image_id")) is not int:
            raise ValueError("Invalid observed per-image prediction record")
        iid = item["image_id"]
        rows = item.get("predictions")
        if iid not in selected_set or iid in seen or not isinstance(rows, list):
            raise ValueError("Observed prediction records must have unique evaluated image IDs")
        if any(not isinstance(row, dict) or row.get("image_id") != iid for row in rows):
            raise ValueError("Observed predictions disagree with their per-image identity")
        seen.add(iid)
        predictions.extend(copy.deepcopy(rows))
    replay = evaluate_coco(annotation_path, predictions, image_ids=selected)
    if replay["dataset_fingerprint"] != observed.get("dataset_fingerprint"):
        raise ValueError("Observed annotations or evaluated image IDs changed")
    reported_ap = observed.get("AP")
    if (
        isinstance(reported_ap, bool)
        or not isinstance(reported_ap, (int, float))
        or not math.isclose(replay["AP"], reported_ap, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("Observed score does not replay from its recorded predictions")
    complete = [row for row in predictions if row["image_id"] not in failed]
    scores = sorted({row["score"] for row in complete})
    ranks = {score: (index + 1) / (2 * (len(scores) + 1)) for index, score in enumerate(scores)}
    for row in complete:
        row["score"] = ranks[row["score"]]
    counts = dict.fromkeys(failed, 0)
    for row in document["annotations"]:
        iid = row["image_id"]
        if iid in failed:
            counts[iid] += 1
            complete.append(
                {
                    "image_id": iid,
                    "category_id": row["category_id"],
                    "bbox": list(row["bbox"]),
                    "score": 1.0,
                }
            )
    if any(count > MAX_DETECTIONS for count in counts.values()):
        raise ValueError("Failed images with more than 500 objects need a separate bound proof")
    result = evaluate_coco(
        annotation_path,
        complete,
        image_ids=selected,
        model={
            "kind": KIND,
            "name": "Perfect failed-image oracle plus fixed completed frontier responses",
            "observed_frontier": copy.deepcopy(observed.get("model", {})),
        },
    )
    if result["AP"] + 1e-12 < replay["AP"]:
        raise ValueError("Oracle construction unexpectedly reduced AP; bound aborted")
    result.update(
        evaluation_kind=KIND,
        synthetic_oracle_bound=True,
        errors=error_count,
        request_or_parse_failures=copy.deepcopy(failures),
        failure_sensitivity={
            "schema_version": 1,
            "observed_AP": replay["AP"],
            "upper_bound_AP": result["AP"],
            "failed_image_ids": sorted(failed),
            "oracle_detections": sum(counts.values()),
            "ground_truth_used_for_failed_answers": True,
            "completed_score_transform": "strict ordinal rank / (2 * (unique scores + 1))",
            "oracle_score": 1.0,
            "observed_report_sha256": hashlib.sha256(
                json.dumps(
                    observed, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest(),
            "assumptions": [
                "one foreground class",
                "no crowd or ignored ground truth",
                "at most 500 ground-truth objects per failed image",
                "completed-image detections, confidence ordering and ties remain fixed",
            ],
            "interpretation": (
                "Supplemental optimistic bound using labels, not a measured frontier score. "
                "Original request failures and the primary zero-error claim gate remain unchanged."
            ),
        },
    )
    return _write(result, output)


def compare_to_failure_upper_bound(
    specialist: dict | Path | str,
    upper_bound: dict | Path | str,
    groups: dict | None = None,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    margin: float = 0.20,
    min_ci_lower: float = 0.10,
    output: Path | str | None = None,
) -> dict:
    """Describe whether an advantage survives perfect answers for all failures.

    The numerical interval uses the ordinary paired bootstrap on counterfactual
    predictions. Primary ``comparison_valid`` and ``margin_target_met`` remain
    false: these are synthetic answers, never a substitute passing primary run.
    """
    specialist, bound = _read_report(specialist), _read_report(upper_bound)
    if bound.get("evaluation_kind") != KIND or bound.get("synthetic_oracle_bound") is not True:
        raise ValueError("Expected an explicitly marked annotation-oracle upper bound")
    if specialist.get("synthetic_oracle_bound") or specialist.get("evaluation_kind") == KIND:
        raise ValueError("Specialist must be an actual model evaluation")
    # Remove errors only in a private bootstrap input. Public evidence retains the
    # original failures; the public primary claim flags are always false below.
    numerical_bound = {**bound, "errors": 0, "request_or_parse_failures": []}
    result = compare_detections(
        specialist,
        numerical_bound,
        groups,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        margin=margin,
        min_ci_lower=min_ci_lower,
    )
    # The shared evaluator deliberately invalidates all synthetic-oracle primary
    # comparisons. The separate sensitivity flag asks only whether this numerical
    # counterfactual is defined and the real specialist has no evaluation errors.
    sensitivity_valid = (
        not any(result["evaluation_errors"].values())
        and result["bootstrap_undefined_samples"] == 0
    )
    survives = (
        sensitivity_valid
        and result["AP_difference"] >= result["required_margin"]
        and result["AP_difference_ci95"][0] >= result["required_ci_lower"]
    )
    result.update(
        evaluation_kind="counterfactual_failure_sensitivity",
        synthetic_oracle_bound=True,
        sensitivity_comparison_valid=sensitivity_valid,
        advantage_survives_perfect_failures=survives,
        comparison_valid=False,
        margin_target_met=False,
        observed_frontier_errors=bound["errors"],
        failed_image_ids=bound["failure_sensitivity"]["failed_image_ids"],
        limitation=(
            "Supplemental comparison to annotation-derived, perfect failed-image answers. "
            "The interval is conditional on completed responses and the declared resampling "
            "units. It does not estimate unobserved provider answers or satisfy the primary "
            "zero-error claim gate."
        ),
    )
    return _write(result, output)
