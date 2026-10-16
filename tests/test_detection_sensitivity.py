"""Counterfactual bounds cannot silently become measured frontier victories."""

import copy
import itertools
import json
import math

import pytest

from vision_rl.detection_eval import compare_detections, evaluate_coco
from vision_rl.detection_sensitivity import (
    compare_to_failure_upper_bound,
    frontier_failure_upper_bound,
)


def dataset(tmp_path, images=4):
    document = {
        "images": [
            {"id": index, "width": 100, "height": 100, "file_name": f"{index}.jpg"}
            for index in range(1, images + 1)
        ],
        "categories": [{"id": 1, "name": "pill"}],
        "annotations": [
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [10, 10, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
            for index in range(1, images + 1)
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document))
    return path, document


def prediction(iid, score, box=(10, 10, 10, 10)):
    return {"image_id": iid, "category_id": 1, "bbox": list(box), "score": score}


def observed(path, predictions, failures=(2,)):
    report = evaluate_coco(path, predictions, model={"name": "observed-frontier"})
    report["errors"] = len(failures)
    report["request_or_parse_failures"] = [
        {"id": str(iid), "error": "HTTP 502"} for iid in failures
    ]
    return report


def test_bound_dominates_enumerated_possible_answers_at_all_iou_thresholds(tmp_path):
    path, _ = dataset(tmp_path, images=3)
    completed = [
        prediction(1, 0.9, (60, 60, 10, 10)),
        prediction(1, 0.6),
        prediction(3, 0.2, (11, 10, 10, 10)),
    ]
    source = observed(path, completed)
    pristine = copy.deepcopy(source)
    bound = frontier_failure_upper_bound(path, source)
    # Include exact, imperfect and duplicate answers, score ties, every possible
    # order relative to completed predictions, and a missed image (empty answer).
    possible = [
        prediction(2, score, box)
        for score, box in itertools.product(
            (0.0, 0.2, 0.6, 0.9, 1.0),
            ((10, 10, 10, 10), (11, 10, 10, 10), (60, 60, 10, 10)),
        )
    ]
    for count in range(3):
        for answer in itertools.combinations_with_replacement(possible, count):
            candidate = evaluate_coco(path, completed + list(answer))
            for metric in ("AP", "AP50", "AP75"):
                assert bound[metric] + 1e-12 >= candidate[metric], (metric, answer)
    assert source == pristine
    assert bound["errors"] == 1
    assert bound["evaluation_kind"] == "annotation_oracle_upper_bound"
    assert bound["failure_sensitivity"]["ground_truth_used_for_failed_answers"]


def test_completed_scores_preserve_extreme_order_and_ties_exactly(tmp_path):
    path, _ = dataset(tmp_path)
    tiny = math.nextafter(0.0, 1.0)
    source = observed(
        path,
        [prediction(1, tiny), prediction(3, tiny * 2), prediction(4, tiny * 2)],
    )
    bound = frontier_failure_upper_bound(path, source)
    scores = {
        item["image_id"]: item["predictions"][0]["score"]
        for item in bound["items"]
    }
    assert 0 < scores[1] < scores[3] == scores[4] < 0.5 < scores[2]


def test_perfect_missing_answers_do_not_pass_primary_claim_gate(tmp_path):
    path, _ = dataset(tmp_path)
    source = observed(path, [], failures=(2,))
    bound = frontier_failure_upper_bound(path, source)
    specialist = evaluate_coco(path, [prediction(iid, 0.9) for iid in range(1, 5)])
    ordinary = compare_detections(specialist, bound, bootstrap_samples=200)
    assert ordinary["comparison_valid"] is False
    assert ordinary["margin_target_met"] is False
    supplemental = compare_to_failure_upper_bound(
        specialist, bound, bootstrap_samples=400, min_ci_lower=0.0
    )
    assert supplemental["sensitivity_comparison_valid"] is True
    assert supplemental["advantage_survives_perfect_failures"] is True
    assert supplemental["comparison_valid"] is False
    assert supplemental["margin_target_met"] is False
    assert supplemental["observed_frontier_errors"] == 1
    assert supplemental["failed_image_ids"] == [2]
    assert source["AP"] == 0
    assert source["errors"] == bound["errors"] == 1


def test_failed_empty_image_gets_no_hallucinated_objects(tmp_path):
    path, document = dataset(tmp_path)
    document["annotations"] = [row for row in document["annotations"] if row["image_id"] != 2]
    path.write_text(json.dumps(document))
    bound = frontier_failure_upper_bound(path, observed(path, []))
    assert bound["failure_sensitivity"]["oracle_detections"] == 0
    assert bound["AP"] == 0
    assert next(item for item in bound["items"] if item["image_id"] == 2)["predictions"] == []


@pytest.mark.parametrize("kind", ["crowd", "ignore", "multiclass", "over500"])
def test_unsupported_assumptions_fail_closed(tmp_path, kind):
    path, document = dataset(tmp_path)
    if kind == "crowd":
        document["annotations"][0]["iscrowd"] = 1
    elif kind == "ignore":
        document["annotations"][0]["ignore"] = 1
    elif kind == "multiclass":
        document["categories"].append({"id": 2, "name": "other"})
        document["annotations"][0]["category_id"] = 2
    else:
        document["annotations"].extend(
            {**document["annotations"][1], "id": index}
            for index in range(10, 510)
        )
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        frontier_failure_upper_bound(path, observed(path, []))


@pytest.mark.parametrize("kind", ["count", "duplicate", "unknown", "bool", "missing"])
def test_incomplete_or_ambiguous_failure_provenance_rejected(tmp_path, kind):
    path, _ = dataset(tmp_path)
    source = observed(path, [])
    if kind == "count":
        source["errors"] = 2
    elif kind == "duplicate":
        source["request_or_parse_failures"] *= 2
        source["errors"] = 2
    elif kind == "unknown":
        source["request_or_parse_failures"][0]["id"] = "999"
    elif kind == "bool":
        source["request_or_parse_failures"][0]["id"] = True
    else:
        del source["request_or_parse_failures"]
    with pytest.raises(ValueError):
        frontier_failure_upper_bound(path, source)


def test_source_score_and_labels_must_replay(tmp_path):
    path, document = dataset(tmp_path)
    source = observed(path, [])
    source["AP"] = 0.5
    with pytest.raises(ValueError, match="score does not replay"):
        frontier_failure_upper_bound(path, source)
    source["AP"] = 0.0
    document["annotations"][0]["bbox"] = [20, 20, 10, 10]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="annotations.*changed"):
        frontier_failure_upper_bound(path, source)


def test_no_failures_preserves_ap_without_enabling_primary_oracle_claim(tmp_path):
    path, _ = dataset(tmp_path)
    source = observed(path, [prediction(1, 0.7), prediction(3, 0.7)], failures=())
    bound = frontier_failure_upper_bound(path, source)
    assert bound["AP"] == pytest.approx(source["AP"], abs=1e-12)
    assert bound["failure_sensitivity"]["oracle_detections"] == 0
    comparison = compare_to_failure_upper_bound(source, bound, bootstrap_samples=20)
    assert comparison["margin_target_met"] is False
    assert comparison["comparison_valid"] is False


def test_outputs_cannot_overwrite_existing_evidence(tmp_path):
    path, _ = dataset(tmp_path)
    source = observed(path, [])
    destination = tmp_path / "bound.json"
    bound = frontier_failure_upper_bound(path, source, output=destination)
    assert json.loads(destination.read_text()) == bound
    with pytest.raises(ValueError, match="already exists"):
        frontier_failure_upper_bound(path, source, output=destination)
    with pytest.raises(ValueError, match="cannot be used"):
        frontier_failure_upper_bound(path, bound)
    with pytest.raises(ValueError, match="explicitly marked"):
        compare_to_failure_upper_bound(source, source)


def test_specialist_errors_invalidate_supplemental_advantage(tmp_path):
    path, _ = dataset(tmp_path)
    source = observed(path, [], failures=(2,))
    bound = frontier_failure_upper_bound(path, source)
    specialist = observed(path, [prediction(iid, 1) for iid in range(1, 5)], failures=(1,))
    comparison = compare_to_failure_upper_bound(
        specialist, bound, bootstrap_samples=20, min_ci_lower=0.0
    )
    assert comparison["sensitivity_comparison_valid"] is False
    assert comparison["advantage_survives_perfect_failures"] is False
