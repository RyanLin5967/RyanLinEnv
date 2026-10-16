import json

import numpy as np
import pytest
from PIL import Image

from vision_rl.detection_inference import (
    InferenceConfig,
    benchmark_detector,
    decode_queries,
    load_image_manifest,
    prediction_difference,
    weighted_box_fusion,
)


def proposal(box=(0.1, 0.2, 0.4, 0.5), score=0.9, view=0, query=0, category=7):
    return {
        "box": list(box),
        "score": score,
        "view_id": view,
        "query_id": query,
        "category_id": category,
    }


def test_flip_inverts_normalized_coordinates_and_preserves_sparse_categories():
    decoded = decode_queries(
        np.array([[0.2, 0.5, 0.2, 0.4]]), np.array([[0.0, 2.0, 100.0]]), (4, 91), flipped=True
    )
    assert [r["category_id"] for r in decoded] == [91, 4]
    assert decoded[0]["box"] == pytest.approx([0.7, 0.3, 0.9, 0.7])
    assert decoded[1]["score"] == 0.5
    assert decoded[0]["score"] == pytest.approx(1 / (1 + np.exp(-2)))


def test_scores_are_not_thresholded_and_extreme_logits_remain_finite():
    result = decode_queries(np.array([[0.5, 0.5, 1.2, 0.3]]), np.array([[-1000.0, 1000.0]]), (8,))
    assert result[0]["score"] == 0
    assert result[0]["box"][0] == pytest.approx(-0.1)


def test_raw_predictions_are_bounded_to_500_with_stable_ties():
    result = decode_queries(np.tile([0.5, 0.5, 0.1, 0.1], (300, 1)), np.zeros((300, 3)), (3, 8))
    assert len(result) == 500
    assert [(r["query_id"], r["category_id"]) for r in result[:4]] == [
        (0, 3),
        (0, 8),
        (1, 3),
        (1, 8),
    ]


@pytest.mark.parametrize(
    "boxes,logits,categories",
    [
        ([[0.5, 0.5, 0.0, 0.1]], [[0]], (1,)),
        ([[0.5, float("nan"), 0.1, 0.1]], [[0]], (1,)),
        ([[0.5, 0.5, 0.1, 0.1]], [[float("inf")]], (1,)),
        ([[0.5, 0.5, 0.1, 0.1]], [[0]], (1, 2)),
        ([[0.5, 0.5, 0.1, 0.1]], [[0, 0]], (1, 1)),
    ],
)
def test_invalid_decoder_outputs_fail_closed(boxes, logits, categories):
    with pytest.raises(ValueError):
        decode_queries(np.asarray(boxes), np.asarray(logits), categories)


def test_same_view_adjacent_and_identical_objects_are_not_merged():
    views = [[proposal(query=0), proposal(query=1), proposal(box=(0.15, 0.2, 0.45, 0.5), query=2)]]
    result = weighted_box_fusion(views)
    assert len(result) == 3
    assert all(len(r["contributors"]) == 1 for r in result)
    assert [r["score"] for r in result] == [0.9, 0.9, 0.9]


def test_cross_view_box_fusion_uses_confidence_and_records_provenance():
    result = weighted_box_fusion(
        [
            [proposal(score=0.9)],
            [proposal(box=(0.12, 0.2, 0.42, 0.5), score=0.3, view=1, query=3)],
        ]
    )
    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(0.6)
    assert result[0]["box"] == pytest.approx([0.105, 0.2, 0.405, 0.5])
    assert result[0]["contributors"] == [[0, 0], [1, 3]]


def test_adjacent_objects_keep_one_member_per_view():
    left, right = (0.1, 0.1, 0.4, 0.4), (0.13, 0.1, 0.43, 0.4)
    result = weighted_box_fusion(
        [
            [proposal(left, query=0), proposal(right, score=0.8, query=1)],
            [
                proposal(left, score=0.95, view=1, query=0),
                proposal(right, score=0.85, view=1, query=1),
            ],
        ]
    )
    assert len(result) == 2
    assert all(len(r["contributors"]) == 2 for r in result)
    assert all(len({v for v, _ in r["contributors"]}) == 2 for r in result)


def test_missing_view_contributes_zero_to_confidence_and_categories_do_not_merge():
    result = weighted_box_fusion([[proposal()], [proposal(view=1, category=99)], []])
    assert len(result) == 2
    assert all(r["score"] == pytest.approx(0.3) for r in result)


def test_fusion_zero_scores_are_finite():
    result = weighted_box_fusion([[proposal(score=0)], [proposal(score=0, view=1)]])
    assert len(result) == 1
    assert result[0]["score"] == 0
    assert np.isfinite(result[0]["box"]).all()


def test_fusion_rejects_duplicate_query_provenance():
    with pytest.raises(ValueError, match="provenance"):
        weighted_box_fusion([[proposal(), proposal()]])
    with pytest.raises(ValueError, match="provenance"):
        weighted_box_fusion([[proposal(view=9)]])


def test_image_manifest_forbids_labels_and_path_escape(tmp_path):
    path = tmp_path / "images.json"
    Image.new("RGB", (12, 8)).save(tmp_path / "sample.png")
    row = {"id": 2, "file_name": "sample.png", "width": 12, "height": 8}
    path.write_text(json.dumps({"images": [row]}))
    result = load_image_manifest(path, tmp_path)
    assert result[0]["path"] == tmp_path / "sample.png"
    path.write_text(json.dumps({"images": [row], "annotations": []}))
    with pytest.raises(ValueError, match="without labels"):
        load_image_manifest(path, tmp_path)
    row["file_name"] = "../outside.png"
    path.write_text(json.dumps({"images": [row]}))
    with pytest.raises(ValueError, match="within the image root"):
        load_image_manifest(path, tmp_path)


@pytest.mark.parametrize(
    "configuration",
    [
        {"resolutions": (384, 384)},
        {"resolutions": (385,)},
        {"resolutions": ()},
        {"resolutions": (2048,)},
        {"fusion_iou": float("nan")},
        {"max_detections": 501},
        {"batch_size": 0},
        {"device": "magic"},
        {"architecture": "large"},
    ],
)
def test_invalid_configuration_fails_before_model_load(configuration):
    with pytest.raises(ValueError):
        InferenceConfig(**configuration).validate()


def test_views_have_unambiguous_order():
    config = InferenceConfig(resolutions=(384, 512), horizontal_flip=True)
    assert config.views == [(384, False), (384, True), (512, False), (512, True)]


def test_prediction_difference_reports_actual_numeric_drift():
    row = {
        "image_id": 1,
        "category_id": 7,
        "contributors": [[0, 2]],
        "bbox": [10, 10, 5, 5],
        "score": 0.9,
    }
    other = {**row, "score": 0.89, "bbox": [10, 10.1, 5, 5]}
    diff = prediction_difference([row], [other])
    assert not diff["exact_prediction_match"]
    assert diff["maximum_score_difference"] == pytest.approx(0.01)
    assert diff["maximum_bbox_pixel_difference"] == pytest.approx(0.1)
    assert diff["missing_detection_identities"] == 0


def test_refinement_uses_top128_without_labels_and_undoes_flip_after_correction(monkeypatch):
    torch = pytest.importorskip("torch")
    from pathlib import Path

    import vision_rl.detection_rl as rl
    from vision_rl.detection_inference import _refine_view

    def fake_predict(cache, checkpoint, *, device):
        assert "targets" not in cache
        assert cache["boxes"].shape == (1, 128, 4)
        assert cache["metadata"]["model_sha256"] == "a" * 64
        assert torch.equal(cache["logits"][0], torch.arange(299, 171, -1).float())
        return [
            {
                "image_id": "5",
                "boxes": cache["boxes"][0] + torch.tensor([0.02, 0, 0.02, 0]),
                "scores": torch.full((128,), 0.4),
            }
        ]

    monkeypatch.setattr(rl, "predict_refinement_safetensors", fake_predict)
    raw = {
        "pred_boxes": torch.tensor([0.2, 0.5, 0.2, 0.2]).expand(1, 300, 4),
        "pred_logits": torch.stack((torch.arange(300), torch.zeros(300)), -1)[None],
    }
    result = _refine_view(
        raw,
        torch.zeros(1, 300, 256),
        [{"id": 5, "width": 100, "height": 80}],
        Path("refiner.safetensors"),
        "a" * 64,
        7,
        flipped=True,
        view_id=1,
    )
    assert len(result[0]) == 128
    assert result[0][0]["query_id"] == 299
    assert result[0][0]["box"] == pytest.approx([0.68, 0.4, 0.88, 0.6])
    assert result[0][0]["category_id"] == 7


def test_forward_embedding_hook_is_removed_after_failure():
    torch = pytest.importorskip("torch")
    from vision_rl.detection_inference import _forward

    class BrokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.class_embed = torch.nn.Linear(2, 1)

        def forward(self, batch):
            self.class_embed(torch.zeros(1, 1, 1, 2))
            raise RuntimeError("intentional forward failure")

    model = BrokenModel()
    with pytest.raises(RuntimeError, match="intentional forward"):
        _forward(model, None, capture_embeddings=True)
    assert not model.class_embed._forward_hooks


def test_benchmark_includes_warmup_repetitions_and_differences(monkeypatch):
    import vision_rl.detection_inference as module

    calls = []

    def fake_infer(detector, images, category_ids, config):
        calls.append((len(images), config.batch_size))
        return [], {"images": len(images), "elapsed_seconds": 2.0, "batch_seconds": [0.5, 0.5]}

    monkeypatch.setattr(module, "infer_images", fake_infer)
    report = benchmark_detector(
        None, [{}, {}], (1,), InferenceConfig(), batch_sizes=(1, 2), warmup=1, repeats=2
    )
    assert calls == [(1, 1), (2, 1), (2, 1), (2, 2), (2, 2), (2, 2)]
    assert report["measurements"][0]["pooled_images_per_second"] == 1.0
    assert report["measurements"][0]["median_batch_latency_ms"] == 500


@pytest.mark.parametrize("dimensions", [(110, 71), (2400, 1600)])
def test_label_free_preprocessing_matches_native_validation_exactly(tmp_path, dimensions):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from types import SimpleNamespace

    from rfdetr.datasets.coco import (
        CocoDetection,
        draft_size_for_transforms,
        make_coco_transforms_square_div_64,
    )

    from vision_rl.detection_inference import infer_images

    rng = np.random.default_rng(42)
    image_path = tmp_path / "image.jpg"
    width, height = dimensions
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(image_path)
    manifest_row = {"id": 8, "file_name": "image.jpg", "width": width, "height": height}
    annotation = tmp_path / "_annotations.coco.json"
    annotation.write_text(
        json.dumps(
            {"images": [manifest_row], "categories": [{"id": 3, "name": "pill"}], "annotations": []}
        )
    )
    draft_size = draft_size_for_transforms("val", 128)
    assert draft_size is None  # Native validation preserves full source dimensions.
    native = CocoDetection(
        tmp_path,
        annotation,
        transforms=make_coco_transforms_square_div_64("val", 128),
        draft_size=draft_size,
    )
    captured = []

    class Model:
        def eval(self):
            return self

        def __call__(self, batch):
            captured.append(batch.clone())
            return {
                "pred_boxes": torch.tensor([[[0.5, 0.5, 0.3, 0.3]]]),
                "pred_logits": torch.zeros(1, 1, 2),
            }

    detector = SimpleNamespace(
        model=SimpleNamespace(model=Model()),
        model_config=SimpleNamespace(patch_size=16, num_windows=2),
    )
    predictions, timing = infer_images(
        detector, [{**manifest_row, "path": image_path}], (3,), InferenceConfig(resolutions=(128,))
    )
    assert torch.equal(captured[0][0], native[0][0])
    assert len(predictions) == 1 and predictions[0]["category_id"] == 3
    assert timing["model_forward_seconds"] > 0
