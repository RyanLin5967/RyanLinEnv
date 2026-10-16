"""Native AP and cached bootstrap are checked against actual pycocotools evaluation."""

import copy
import json

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from vision_rl.detection_eval import (  # noqa: E402
    _cached_ap,
    _prepared_cache,
    compare_detections,
    detection_prompt,
    evaluate_coco,
    load_coco,
    parse_detection_response,
)


def write_coco(tmp_path, document, name="annotations.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return path


@pytest.fixture
def dataset():
    return {
        "images": [
            {"id": i, "width": 100, "height": 80, "file_name": f"{i}.png"} for i in range(4)
        ],
        "categories": [
            {"id": 0, "name": "parent"},
            {"id": 1, "name": "pill"},
            {"id": 2, "name": "other"},
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [60, 30, 10, 10]},
        ],
    }


def detections(document):
    return [
        {
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": a["bbox"],
            "score": 0.9,
        }
        for a in document["annotations"]
    ]


def test_perfect_includes_zero_annotation_id_and_negative_images(tmp_path, dataset):
    path = write_coco(tmp_path, dataset)
    output = tmp_path / "report.json"
    report = evaluate_coco(path, detections(dataset), output=output)
    assert report["AP"] == pytest.approx(1)
    assert report["AP50"] == pytest.approx(1)
    assert report["AP75"] == pytest.approx(1)
    assert report["n"] == 4
    assert report["images_without_predictions"] == 1
    assert report["counts_at_iou50"] == {
        "tp": 4,
        "fp": 0,
        "fn": 0,
        "ground_truth": 4,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert [c["id"] for c in report["categories"]] == [1, 2]
    assert json.loads(output.read_text())["AP"] == report["AP"]


def test_missing_predictions_never_drop_images(tmp_path, dataset):
    path = write_coco(tmp_path, dataset)
    report = evaluate_coco(path, [])
    assert report["AP"] == report["AP50"] == report["AP75"] == 0
    assert report["n"] == report["images_without_predictions"] == 4
    assert report["counts_at_iou50"]["fn"] == 4
    one = evaluate_coco(path, detections(dataset)[:1], image_ids=[0, 1])
    assert one["AP"] == pytest.approx(51 / 101)
    assert one["counts_at_iou50"]["recall"] == 0.5


def test_max_dets_500_not_coco_summarize_default100(tmp_path):
    document = {
        "images": [{"id": 0, "width": 1000, "height": 1000}],
        "categories": [{"id": 1, "name": "pill"}],
        "annotations": [
            {
                "id": i,
                "image_id": 0,
                "category_id": 1,
                "bbox": [(i % 20) * 40, (i // 20) * 40, 20, 20],
            }
            for i in range(150)
        ],
    }
    report = evaluate_coco(write_coco(tmp_path, document), detections(document))
    assert report["AP"] == pytest.approx(1)
    assert report["AR"] == pytest.approx(1)
    assert report["counts_at_iou50"]["tp"] == 150


@pytest.mark.parametrize(
    "text",
    [
        '[{"label":"PILL","box":[125,100,375,300]}]',
        'Here are detections:\n```json\n[{"label":"pill","box_2d":[125,100,375,300]}]\n```',
        '{"objects":[{"label":"pill","box":[125,100,375,300],"score":0.8}]}',
    ],
)
def test_vlm_parser_scale_and_envelopes(text):
    result = parse_detection_response(text, {"id": 7, "width": 200, "height": 160}, {"pill": 3})
    assert result[0]["bbox"] == [20.0, 20.0, 40.0, 40.0]
    assert result[0]["category_id"] == 3
    assert result[0]["image_id"] == 7


@pytest.mark.parametrize(
    "text",
    [
        "no pills",
        "[] []",
        '{"objects": []',
        '[{"label":"pill","box":[300,100,125,300]}]',
        '[{"label":"pill","box":[0,0,1001,1000]}]',
        '[{"label":"pill","box":[0,0,0,100]}]',
        '[{"label":"other","box":[0,0,100,100]}]',
        '[{"label":"pill","box":[true,0,100,100]}]',
        '[{"label":"pill","box":[0,0,100,100],"score":2}]',
        '[{"label":"pill","box":[0,0,100,100],"box_2d":[0,0,100,100]}]',
        '[{"label":"pill","box":[0,0,100,100]},false]',
    ],
)
def test_vlm_parser_rejects_invalid_without_silently_scoring_empty(text):
    with pytest.raises(ValueError):
        parse_detection_response(text, {"id": 1, "width": 200, "height": 100}, {"pill": 1})


def test_empty_is_valid_and_prompt_defines_coordinate_order():
    assert (
        parse_detection_response(
            "```json\n[]\n```", {"id": 1, "width": 200, "height": 100}, {"pill": 1}
        )
        == []
    )
    assert "ymin,xmin,ymax,xmax" in detection_prompt({"pill": 1})
    assert "shadows" in detection_prompt({"pill": 1})


@pytest.mark.parametrize(
    "change",
    [
        {"category_id": 0},
        {"image_id": 999},
        {"score": float("nan")},
        {"score": -1},
        {"bbox": [0, 0, -1, 20]},
    ],
)
def test_invalid_predictions_fail(tmp_path, dataset, change):
    pred = {**detections(dataset)[0], **change}
    with pytest.raises(ValueError):
        evaluate_coco(write_coco(tmp_path, dataset), [pred])


def test_coco_structural_validation(tmp_path, dataset):
    duplicate = copy.deepcopy(dataset)
    duplicate["images"].append(duplicate["images"][0])
    with pytest.raises(ValueError, match="duplicate image"):
        load_coco(write_coco(tmp_path, duplicate))
    duplicate = copy.deepcopy(dataset)
    duplicate["annotations"][1]["id"] = 0
    with pytest.raises(ValueError, match="duplicate annotation"):
        load_coco(write_coco(tmp_path, duplicate))


def clone_sample(tmp_path, document, predictions, sample):
    """Slow reference: physically duplicate images, labels and predictions, rerun IoUs."""
    cloned = {"images": [], "annotations": [], "categories": document["categories"]}
    cloned_predictions = []
    image_map = {image["id"]: image for image in document["images"]}
    for index, image_id in enumerate(sample):
        cloned["images"].append({**image_map[image_id], "id": index})
        for ann in document["annotations"]:
            if ann["image_id"] == image_id:
                cloned["annotations"].append(
                    {**ann, "id": len(cloned["annotations"]) + 1, "image_id": index}
                )
        cloned_predictions.extend(
            {**pred, "image_id": index} for pred in predictions if pred["image_id"] == image_id
        )
    return evaluate_coco(write_coco(tmp_path, cloned, "clone.json"), cloned_predictions)["AP"]


def test_cached_bootstrap_matches_bruteforce_iou_recomputation(tmp_path, dataset):
    predictions = detections(dataset)
    # Include score ties, a localization error, a false positive on a negative image,
    # and an entirely absent class in some resamples to test COCO averaging semantics.
    predictions[1]["bbox"] = [14, 10, 20, 20]
    predictions.pop(2)
    predictions.append({"image_id": 3, "category_id": 1, "bbox": [1, 1, 10, 10], "score": 0.9})
    report = evaluate_coco(write_coco(tmp_path, dataset), predictions)
    prepared = _prepared_cache(report["match_cache"])
    rng = np.random.default_rng(123)
    for _ in range(12):
        sample = rng.integers(0, 4, 4).tolist()
        if all(i == 3 for i in sample):
            continue
        assert _cached_ap(prepared, sample) == pytest.approx(
            clone_sample(tmp_path, dataset, predictions, sample), abs=1e-14
        )


def test_group_bootstrap_matches_bruteforce_draws(tmp_path, dataset):
    path = write_coco(tmp_path, dataset)
    better_predictions, worse_predictions = detections(dataset), detections(dataset)[:1]
    better = evaluate_coco(path, better_predictions)
    worse = evaluate_coco(path, worse_predictions)
    groups = {0: "capture-a", 1: "capture-a", 2: "capture-b", 3: "capture-b"}
    result = compare_detections(better, worse, groups, bootstrap_samples=20, seed=8)
    rng, differences = np.random.default_rng(8), []
    members = [[0, 1], [2, 3]]
    for _ in range(20):
        sample = [iid for unit in rng.integers(0, 2, 2) for iid in members[unit]]
        differences.append(
            clone_sample(tmp_path, dataset, better_predictions, sample)
            - clone_sample(tmp_path, dataset, worse_predictions, sample)
        )
    assert result["AP_difference_ci95"] == pytest.approx(np.quantile(differences, [0.025, 0.975]))
    assert result["resampling_units"] == 2
    assert result["group_sizes"] == [2, 2]
    assert result["margin_target_met"]


def test_comparison_rejects_mismatched_data_and_stale_cache(tmp_path, dataset):
    path = write_coco(tmp_path, dataset)
    first = evaluate_coco(path, detections(dataset))
    second = evaluate_coco(path, [])
    wrong = copy.deepcopy(second)
    wrong["AP"] = 0.9
    with pytest.raises(ValueError, match="cached"):
        compare_detections(first, wrong, bootstrap_samples=10)
    wrong = copy.deepcopy(second)
    wrong["dataset_fingerprint"] = "other"
    with pytest.raises(ValueError, match="fingerprint"):
        compare_detections(first, wrong, bootstrap_samples=10)
    with pytest.raises(ValueError, match="cover"):
        compare_detections(first, second, groups={0: "a"}, bootstrap_samples=10)
    with pytest.raises(ValueError, match="two independent"):
        compare_detections(first, second, groups={i: "a" for i in range(4)}, bootstrap_samples=10)


def test_crowd_ignored_and_false_positive_at_negative_image(tmp_path, dataset):
    dataset["annotations"][0]["iscrowd"] = 1
    predictions = detections(dataset)
    predictions.append({"image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 1})
    report = evaluate_coco(write_coco(tmp_path, dataset), predictions)
    assert report["counts_at_iou50"]["tp"] == 3
    assert report["counts_at_iou50"]["fp"] == 1
    assert report["counts_at_iou50"]["fn"] == 0
    assert _cached_ap(_prepared_cache(report["match_cache"]), [0, 1, 2, 3]) == pytest.approx(
        report["AP"]
    )


def test_out_of_frame_predictions_retain_native_coco_geometry(tmp_path, dataset):
    prediction = {"image_id": 0, "category_id": 1, "bbox": [-10, -20, 130, 120], "score": 0.9}
    report = evaluate_coco(write_coco(tmp_path, dataset), [prediction])
    assert report["items"][0]["predictions"][0]["bbox"] == [-10.0, -20.0, 130.0, 120.0]
    assert report["counts_at_iou50"]["fp"] == 1
    bad_truth = copy.deepcopy(dataset)
    bad_truth["annotations"][0]["bbox"] = prediction["bbox"]
    with pytest.raises(ValueError, match="outside"):
        load_coco(write_coco(tmp_path, bad_truth))


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize(
    "failure",
    [
        {"errors": 1},
        {"request_or_parse_failures": [{"image_id": 0, "error": "HTTP 402 quota exhausted"}]},
        {"errors": 0, "request_or_parse_failures": [{"image_id": 0, "error": "invalid JSON"}]},
        {"errors": 1, "request_or_parse_failures": [{"image_id": 0, "error": "invalid JSON"}]},
    ],
)
def test_failed_evaluations_cannot_trigger_margin_win(tmp_path, dataset, side, failure):
    path = write_coco(tmp_path, dataset)
    a = evaluate_coco(path, detections(dataset))
    b = evaluate_coco(path, [])
    valid = compare_detections(a, b, bootstrap_samples=20)
    assert valid["comparison_valid"]
    assert valid["margin_target_met"]
    assert valid["evaluation_errors"] == {"a": 0, "b": 0}
    (a if side == "a" else b).update(failure)
    failed = compare_detections(a, b, bootstrap_samples=20)
    assert failed["AP_difference"] == pytest.approx(1)
    assert not failed["comparison_valid"]
    assert not failed["margin_target_met"]
    assert failed["evaluation_errors"][side] == 1


@pytest.mark.parametrize(
    "failure",
    [
        {"errors": "unknown"},
        {"errors": -1},
        {"errors": True},
        {"request_or_parse_failures": "HTTP 402"},
    ],
)
def test_malformed_error_provenance_fails_closed(tmp_path, dataset, failure):
    path = write_coco(tmp_path, dataset)
    a = evaluate_coco(path, detections(dataset))
    b = evaluate_coco(path, [])
    b.update(failure)
    with pytest.raises(ValueError):
        compare_detections(a, b, bootstrap_samples=20)


def test_explicit_pixel_xyxy_parser_does_not_autodetect_scale_or_order():
    image = {"id": 1, "width": 200, "height": 100}
    text = '{"detections":[{"label":"pill","box_2d":[20,10,60,30]}]}'
    pixel = parse_detection_response(text, image, {"pill": 1}, coordinate_format="xyxy_pixels")
    assert pixel[0]["bbox"] == [20.0, 10.0, 40.0, 20.0]
    default = parse_detection_response(text, image, {"pill": 1})
    assert default[0]["bbox"] == [2.0, 2.0, 4.0, 4.0]
    for box in ([60, 10, 20, 30], [20, 10, 201, 30], [20, 10, 60, 101]):
        with pytest.raises(ValueError, match="pixel box"):
            parse_detection_response(
                json.dumps([{"label": "pill", "box_2d": box}]),
                image,
                {"pill": 1},
                coordinate_format="xyxy_pixels",
            )
    with pytest.raises(ValueError, match="coordinate_format"):
        parse_detection_response("[]", image, {"pill": 1}, coordinate_format="auto")


@pytest.mark.parametrize(
    "marker",
    [{"synthetic_oracle_bound": True}, {"evaluation_kind": "annotation_oracle_upper_bound"}],
)
def test_oracle_sensitivity_is_never_a_primary_comparison(tmp_path, dataset, marker):
    path = write_coco(tmp_path, dataset)
    a = evaluate_coco(path, detections(dataset))
    b = evaluate_coco(path, [])
    b.update(marker)
    comparison = compare_detections(a, b, bootstrap_samples=20)
    assert comparison["AP_difference"] == pytest.approx(1)
    assert comparison["evaluation_errors"] == {"a": 0, "b": 0}
    assert comparison["synthetic_oracle_evidence"] == {"a": False, "b": True}
    assert not comparison["comparison_valid"]
    assert not comparison["margin_target_met"]
