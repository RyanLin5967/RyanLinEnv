import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from vision_rl.detection_training import (
    DetectorTrainingConfig,
    _validate_resolution,
    export_detector_safetensors,
    inspect_split,
    stage_training_data,
    summarize_training_run,
    train_detector,
)


def make_split(path: Path, content: bytes = b"image") -> None:
    path.mkdir(parents=True)
    color = tuple(hashlib.sha256(content).digest()[:3])
    Image.new("RGB", (100, 80), color).save(path / "sample.jpg")
    (path / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "sample.jpg", "width": 100, "height": 80}],
                "categories": [{"id": 1, "name": "pill"}],
                "annotations": [
                    {
                        "id": 2,
                        "image_id": 7,
                        "category_id": 1,
                        "bbox": [10, 10, 30, 20],
                        "area": 600,
                    }
                ],
            }
        )
    )


def test_stage_contains_development_only(tmp_path):
    source = tmp_path / "source"
    for split in ("train", "valid", "test"):
        make_split(source / split, split.encode())
    destination = tmp_path / "stage"
    result = stage_training_data(source, destination)
    assert set(p.name for p in destination.iterdir()) == {"train", "valid"}
    assert set(result) == {"train", "validation"}
    assert result["train"]["categories"] == {"1": "pill"}
    assert not (destination / "test").exists()


def test_exact_leakage_rejected(tmp_path):
    make_split(tmp_path / "source/train")
    make_split(tmp_path / "source/valid")
    with pytest.raises(ValueError, match="Identical image bytes"):
        stage_training_data(tmp_path / "source", tmp_path / "stage")


def test_fingerprint_detects_annotation_change(tmp_path):
    make_split(tmp_path / "train")
    first = inspect_split(tmp_path / "train")
    annotation = tmp_path / "train/_annotations.coco.json"
    coco = json.loads(annotation.read_text())
    coco["annotations"][0]["bbox"][0] = 11
    annotation.write_text(json.dumps(coco))
    assert inspect_split(tmp_path / "train")["fingerprint"] != first["fingerprint"]


def test_fingerprint_detects_image_change(tmp_path):
    make_split(tmp_path / "train")
    first = inspect_split(tmp_path / "train")
    Image.new("RGB", (100, 80), (0, 0, 0)).save(tmp_path / "train/sample.jpg")
    assert inspect_split(tmp_path / "train")["fingerprint"] != first["fingerprint"]


def test_reencoded_pixel_leakage_rejected(tmp_path):
    source = tmp_path / "source"
    make_split(source / "train", b"train")
    make_split(source / "valid", b"valid")
    with Image.open(source / "train/sample.jpg") as image:
        image.save(source / "valid/sample.jpg", format="PNG")
    with pytest.raises(ValueError, match="Identical decoded pixels"):
        stage_training_data(source, tmp_path / "stage")


def test_path_escape_rejected(tmp_path):
    make_split(tmp_path / "train")
    annotation = tmp_path / "train/_annotations.coco.json"
    coco = json.loads(annotation.read_text())
    coco["images"][0]["file_name"] = "../sample.jpg"
    annotation.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="escapes"):
        inspect_split(tmp_path / "train")


def test_staging_with_test_directory_rejected(tmp_path):
    make_split(tmp_path / "source/train", b"train")
    make_split(tmp_path / "source/valid", b"valid")
    (tmp_path / "stage/test").mkdir(parents=True)
    with pytest.raises(ValueError, match="only train and valid"):
        stage_training_data(tmp_path / "source", tmp_path / "stage")


def test_duplicate_annotation_ids_rejected(tmp_path):
    make_split(tmp_path / "train")
    annotation = tmp_path / "train/_annotations.coco.json"
    coco = json.loads(annotation.read_text())
    coco["annotations"].append(coco["annotations"][0].copy())
    annotation.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="Duplicate COCO annotation"):
        inspect_split(tmp_path / "train")


def test_unannotated_leaf_classes_preserve_rfdetr_label_mapping(tmp_path):
    source = tmp_path / "source"
    for split in ("train", "valid"):
        make_split(source / split, split.encode())
        path = source / split / "_annotations.coco.json"
        coco = json.loads(path.read_text())
        coco["categories"] = [
            {"id": 1, "name": "unused"},
            {"id": 2, "name": "rare"},
            {"id": 3, "name": "pill"},
        ]
        coco["annotations"][0]["category_id"] = 3
        if split == "valid":
            coco["annotations"] = []
        path.write_text(json.dumps(coco))
    metadata = stage_training_data(source, tmp_path / "stage")
    expected = {"1": "unused", "2": "rare", "3": "pill"}
    assert metadata["train"]["categories"] == expected
    assert metadata["validation"]["categories"] == expected


def test_concurrent_trainers_cannot_overwrite_checkpoint(tmp_path):
    import fcntl

    output = tmp_path / "run"
    output.mkdir()
    with (output / ".training.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another trainer"):
            train_detector(tmp_path / "source", output, DetectorTrainingConfig())


def test_resumed_evidence_includes_prior_time_and_best_epoch(tmp_path):
    (tmp_path / "failure.json").write_text(json.dumps({"elapsed_seconds": 20.0}))
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "elapsed_seconds": 80.0,
                "metrics": {"val/mAP_50_95": 0.7},
            }
        )
    )
    rows = [
        {"event": "validation_end", "epoch": 0, "metrics": {"val/mAP_50_95": 0.8}},
        {"event": "validation_end", "epoch": 1, "metrics": {"val/mAP_50_95": 0.7}},
    ]
    (tmp_path / "telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    evidence = summarize_training_run(tmp_path)
    assert evidence["total_recorded_active_fit_seconds"] == 100
    assert evidence["selection"]["epoch_1based"] == 1
    assert evidence["selection"]["AP"] == 0.8
    assert evidence["final_epoch_metrics"]["val/mAP_50_95"] == 0.7


def test_safetensors_loader_rejects_wrong_keys_and_nonfinite_values(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    rfdetr = pytest.importorskip("rfdetr")
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from vision_rl.detection_training import load_detector_safetensors

    def fake_detector(**kwargs):
        return SimpleNamespace(model=SimpleNamespace(model=torch.nn.Linear(2, 1)))

    monkeypatch.setattr(rfdetr, "RFDETRNano", fake_detector)
    path = tmp_path / "model.safetensors"
    save_file({"unknown": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="keys"):
        load_detector_safetensors(path)
    save_file({"weight": torch.tensor([[float("nan"), 0.0]]), "bias": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="nonfinite"):
        load_detector_safetensors(path)


def test_export_preserves_custom_resolution_without_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    # RF-DETR strips model_config from checkpoint_best_total.pth. Reconstructing
    # that checkpoint would resize this trained 32x32 grid to the default24x24.
    key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
    state = {key: torch.arange(1025 * 4, dtype=torch.float32).reshape(1, 1025, 4)}
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    torch.save({"model": state, "model_name": "RFDETRNano"}, checkpoint)
    output = tmp_path / "model.safetensors"
    report = export_detector_safetensors(checkpoint, output)
    actual = safetensors.load_file(output)
    assert torch.equal(actual[key], state[key])
    assert report["positional_embedding_shapes"][key] == [1, 1025, 4]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"lr": float("nan")},
        {"architecture": "giant"},
        {"preflight_batches": 1, "grad_accum_steps": 4},
        {"workers": -1},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        DetectorTrainingConfig(**kwargs).validate()


@pytest.mark.parametrize("resolution", [True, 384.0, 127, 129, 1056])
def test_untrusted_or_incompatible_resolution_rejected(resolution):
    with pytest.raises(ValueError, match="multiple of 32"):
        _validate_resolution(resolution)


@pytest.mark.parametrize("resolution", [None, 128, 384, 512, 1024])
def test_supported_resolution_accepted(resolution):
    _validate_resolution(resolution)
