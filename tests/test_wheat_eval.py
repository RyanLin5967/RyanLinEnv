import copy
import json

import pytest

from vision_rl.wheat_eval import (
    calibrate_wheat_predictions,
    calibrated_wheat_evaluation,
    evaluate_wheat,
    official_image_rewards,
    paired_domain_comparison,
    validate_wheat_coco,
)


def fixture_document():
    return {
        "images": [
            {
                "id": i,
                "width": 1024,
                "height": 1024,
                "domain": "Arvalis_1" if i < 4 else "Arvalis_2",
            }
            for i in range(1, 5)
        ],
        "categories": [{"id": 1, "name": "wheat_head"}],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 3, 3]} for i in range(1, 5)
        ],
    }


def test_validation_preserves_repeated_and_out_of_frame_gt():
    doc = fixture_document()
    doc["annotations"][0]["bbox"] = [-2, 0, 1040, 40]
    repeated = dict(doc["annotations"][0], id=5)
    doc["annotations"].append(repeated)
    before = copy.deepcopy(doc)
    _, truth, _, selected = validate_wheat_coco(doc, [])
    assert doc == before
    assert truth[1] == [[-2, 0, 1040, 40], [-2, 0, 1040, 40]]
    assert selected == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "bad",
    [
        {"image_id": 9, "category_id": 1, "bbox": [0, 0, 3, 3]},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 3, 3]},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 3]},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 3, 3], "score": float("nan")},
        {"image_id": True, "category_id": 1, "bbox": [0, 0, 3, 3]},
    ],
)
def test_malformed_predictions_fail_instead_of_becoming_empty(bad):
    with pytest.raises(ValueError):
        validate_wheat_coco(fixture_document(), [bad])


def test_cap_and_subset_cannot_silently_discard_predictions():
    detection = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 3, 3]}
    with pytest.raises(ValueError, match="500"):
        validate_wheat_coco(fixture_document(), [detection] * 501)
    with pytest.raises(ValueError, match="unevaluated"):
        validate_wheat_coco(fixture_document(), [detection], [2, 3, 4])


@pytest.fixture
def native():
    pytest.importorskip("wilds")
    pytest.importorskip("torch_scatter")


@pytest.mark.parametrize(
    "truth,predictions,scores,expected",
    [
        ([[0, 0, 10, 10], [3, 0, 13, 10]], [[1, 0, 11, 10], [0, 0, 10, 10]], [1, 1], 1 / 3),
        ([[0, 0, 3, 3]], [[1, 0, 4, 3]], [1], 1),
        ([[0, 0, 3, 3]], [[0, 0, 3, 3]], [0.5], 0),
        ([[0, 0, 3, 3]], [[0, 0, 3, 3]], [0.5001], 1),
        ([[0, 0, 3, 3]], [[0, 0, 3, 3], [0, 0, 3, 3]], [1, 1], 0.5),
        ([[0, 0, 3, 3], [0, 0, 3, 3]], [[0, 0, 3, 3], [0, 0, 3, 3]], [1, 1], 1 / 3),
        ([[-2, 0, 1040, 40]], [[-2, 0, 1040, 40]], [1], 1),
        ([], [], [], 1),
        ([], [[0, 0, 3, 3]], [1], 0),
        ([[0, 0, 3, 3]], [], [], 0),
    ],
)
def test_official_reward_edge_cases(native, truth, predictions, scores, expected):
    import torch

    boxes = torch.tensor(predictions, dtype=torch.float32).reshape(1, -1, 4)
    reward = official_image_rewards(boxes, torch.tensor([scores]), truth)
    assert reward.dtype == torch.float64
    assert reward.tolist() == pytest.approx([expected], abs=1e-7)


def test_native_group_average_complete_ids_and_failure_provenance(native, tmp_path):
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(fixture_document()))
    predictions = [{"image_id": i, "category_id": 1, "bbox": [0, 0, 3, 3]} for i in (1, 2, 3)]
    report = evaluate_wheat(path, predictions, request_or_parse_failures=[{"image_id": 4}])
    assert report["metrics"]["detection_acc_avg_dom"] == 0.5
    assert report["metrics"]["detection_acc_avg"] == 0.75
    assert report["image_ids"] == [1, 2, 3, 4]
    assert report["per_domain"]["Arvalis_1"]["n"] == 3
    assert report["per_domain"]["Arvalis_2"]["image_ids"] == [4]
    assert not report["evaluation_valid"]
    assert report["errors"] == 1
    assert report["benchmark_id"] is None  # Synthetic COCO is not the wheat benchmark.


def test_all_empty_images_still_contribute(native, tmp_path):
    doc = fixture_document()
    doc["annotations"] = []
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(doc))
    report = evaluate_wheat(path, [])
    assert report["metrics"]["detection_acc_avg_dom"] == 1
    assert report["n"] == 4


def test_unknown_domains_are_not_silently_grouped(native, tmp_path):
    doc = fixture_document()
    doc["images"][0]["domain"] = "unknown"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Unknown domain"):
        evaluate_wheat(path, [])


def test_calibration_preserves_predictions_and_fixed_grid_ties(native, tmp_path):
    path = tmp_path / "validation.json"
    document = fixture_document()
    document["info"] = {"native_split": "validation"}
    path.write_text(json.dumps(document))
    predictions = [
        {"image_id": i, "category_id": 1, "bbox": [0, 0, 3, 3], "score": 0.4} for i in range(1, 5)
    ]
    before = copy.deepcopy(predictions)
    report, calibrated = calibrated_wheat_evaluation(path, predictions)
    assert predictions == before
    assert report["metrics"]["detection_acc_avg_dom"] == 1
    assert report["calibration"]["foreground_logit_offset"] == 0.5
    assert report["calibration"]["raw_zero_offset_ADA"] == 0
    assert len(report["calibration"]["grid"]) == 7
    assert [row["image_id"] for row in calibrated] == [1, 2, 3, 4]
    assert [row["bbox"] for row in calibrated] == [row["bbox"] for row in predictions]
    assert calibrate_wheat_predictions(predictions, 0.5) == calibrated
    report, _ = calibrated_wheat_evaluation(path, calibrated)
    assert report["calibration"]["foreground_logit_offset"] == 0


def test_calibration_cannot_search_native_test(tmp_path):
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps({"info": {"native_split": "test"}}))
    with pytest.raises(ValueError, match="forbidden"):
        calibrated_wheat_evaluation(path, [])


def test_domain_bootstrap_uses_equal_domains_and_paired_membership(native, tmp_path):
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(fixture_document()))
    predictions = [{"image_id": i, "category_id": 1, "bbox": [0, 0, 3, 3]} for i in range(1, 5)]
    baseline = evaluate_wheat(path, predictions[:3])
    candidate = evaluate_wheat(path, predictions)
    comparison = paired_domain_comparison(candidate, baseline, resamples=1000)
    assert comparison["delta"] == 0.5  # Image-weighted difference would incorrectly give0.25.
    assert comparison["ci95"] == [0.0, 1.0]
    assert not comparison["substantial_gain_and_positive_ci"]
    assert json.loads(json.dumps(comparison, allow_nan=False)) == comparison
    failed = copy.deepcopy(baseline)
    failed["evaluation_valid"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        paired_domain_comparison(candidate, failed, resamples=1000)
    mismatched = copy.deepcopy(baseline)
    mismatched["per_domain"]["Arvalis_1"]["image_ids"] = [9, 10, 11]
    with pytest.raises(ValueError, match="membership"):
        paired_domain_comparison(candidate, mismatched, resamples=1000)


def test_uniform_domain_gain_has_positive_cluster_interval(native, tmp_path):
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(fixture_document()))
    predictions = [{"image_id": i, "category_id": 1, "bbox": [0, 0, 3, 3]} for i in range(1, 5)]
    baseline = evaluate_wheat(path, [])
    candidate = evaluate_wheat(path, predictions)
    comparison = paired_domain_comparison(candidate, baseline, resamples=1000)
    assert comparison["delta"] == 1
    assert comparison["ci95"] == [1, 1]
    assert comparison["substantial_gain_and_positive_ci"] is True
    assert json.loads(json.dumps(comparison, allow_nan=False)) == comparison
