import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from vision_rl.detection_training import (
    DetectorTrainingConfig,
    _configure_wheat_selection,
    _foreground_results,
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


def test_wheat_foreground_postprocess_keeps_all_queries_and_native_extents():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    boxes = torch.full((2, 300, 4), 0.5)
    boxes[:, 0] = torch.tensor([0.0, 0.0, 0.5, 0.5])
    logits = torch.zeros(2, 300, 2)
    logits[..., 1] = 100  # Background must never evict a foreground query.
    logits[0, :, 0] = torch.linspace(-2, 2, 300)
    result = _foreground_results(
        {"pred_boxes": boxes, "pred_logits": logits}, torch.tensor([[100, 200], [50, 100]])
    )
    assert result[0]["boxes"].shape == (300, 4)
    assert result[0]["boxes"][-1].tolist() == [-50, -25, 50, 25]
    assert result[0]["scores"].dtype == torch.float64
    torch.testing.assert_close(result[0]["scores"], logits[0, :, 0].double().sigmoid().flip(0))
    assert result[1]["boxes"][0].tolist() == [-25, -12.5, 25, 12.5]
    assert result[0]["labels"].tolist() == [0] * 300


def test_wheat_summary_selects_ada_not_supplemental_ap(tmp_path):
    (tmp_path / "training-plan.json").write_text(
        json.dumps(
            {
                "selection_protocol": {"primary_key": "detection_acc_avg_dom"},
                "selection_metric": "validation calibrated WILDS equal-domain detection accuracy",
            }
        )
    )
    rows = []
    for epoch, ap, ada, offset in [(0, 0.9, 0.6, 0.0), (1, 0.8, 0.7, -0.5)]:
        rows.append(
            {
                "event": "validation_end",
                "epoch": epoch,
                "metrics": {
                    "val/mAP_50_95": ap,
                    "val/detection_acc_avg_dom": ada,
                    "val/foreground_logit_offset": offset,
                    "val/raw_zero_offset_ADA": 0.5,
                },
            }
        )
    (tmp_path / "telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "elapsed_seconds": 12,
                "metrics": rows[-1]["metrics"],
            }
        )
    )
    (tmp_path / "validation-best.json").write_text(
        json.dumps(
            {
                "epoch_1based": 2,
                "metrics": {"detection_acc_avg_dom": 0.7},
                "calibration": {"foreground_logit_offset": -0.5, "raw_zero_offset_ADA": 0.5},
            }
        )
    )
    evidence = summarize_training_run(tmp_path)
    assert evidence["selection"]["epoch_1based"] == 2
    assert evidence["selection"]["ADA"] == 0.7
    assert evidence["selection"]["AP"] == 0.8
    assert evidence["selection"]["foreground_logit_offset"] == -0.5


def test_wheat_callback_uses_original_annotations_and_rejects_missing_images(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from types import SimpleNamespace

    from pytorch_lightning.callbacks import EarlyStopping
    from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping

    from vision_rl import wheat_eval

    annotations = tmp_path / "_annotations.coco.json"
    annotations.write_text(json.dumps({"images": [{"id": 1}, {"id": 2}]}))
    original = annotations.read_bytes()
    calls = []

    def score(path, predictions, image_ids):
        calls.append((path, predictions, image_ids))
        return {
            "metrics": {"detection_acc_avg_dom": 0.0},
            "calibration": {
                "foreground_logit_offset": 0.0,
                "raw_zero_offset_ADA": 0.0,
            },
        }, predictions

    monkeypatch.setattr(wheat_eval, "calibrated_wheat_evaluation", score)
    trainer = SimpleNamespace(
        callbacks=[BestModelCallback(str(tmp_path)), RFDETREarlyStopping()],
        world_size=1,
        sanity_checking=False,
        callback_metrics={},
        current_epoch=0,
        global_step=1,
    )
    module = SimpleNamespace(log=lambda *args, **kwargs: None)
    _configure_wheat_selection(trainer, module, tmp_path, tmp_path, DetectorTrainingConfig())
    best, early, callback = trainer.callbacks
    assert best._monitor_ema == "val/ema_detection_acc_avg_dom"
    assert best._best_ema == -1.0  # A genuine zero must select the first epoch.
    assert isinstance(early, EarlyStopping)
    assert early.monitor == "val/ema_detection_acc_avg_dom"
    callback.on_validation_epoch_start(trainer, module)
    outputs = {
        "results": [{"boxes": torch.tensor([[-1.0, 1, 3, 4]]), "scores": torch.tensor([0.9])}],
        "targets": [{"image_id": torch.tensor(1), "boxes": torch.zeros(0, 4)}],
    }
    callback.on_validation_batch_end(trainer, module, outputs, None, 0)
    with pytest.raises(ValueError, match="complete native validation"):
        callback.on_validation_epoch_end(trainer, module)
    outputs["targets"][0]["image_id"] = torch.tensor(2)
    callback.on_validation_batch_end(trainer, module, outputs, None, 1)
    callback.on_validation_epoch_end(trainer, module)
    assert annotations.read_bytes() == original
    assert calls[0][0] == annotations.resolve()
    assert calls[0][2] == [1, 2]
    assert calls[0][1][0]["bbox"] == [-1, 1, 4, 3]
    assert callback.state_dict() == {"best_score": 0, "best_epoch": 1, "best_offset": 0}
    callback.load_state_dict({"best_score": 0.75, "best_epoch": 4, "best_offset": -0.5})
    assert callback.best_epoch == 4
    assert callback.best_score == 0.75


def test_lightning_near_tied_ada_keeps_checkpoint_and_calibration_together(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    pytest.importorskip("rfdetr")
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.training.callbacks.best_model import BestModelCallback

    from vision_rl import wheat_eval

    values = [0.123456789, 0.123456790]
    assert torch.tensor(values[0]).item() == torch.tensor(values[1]).item()
    calls = []

    def score(path, predictions, image_ids):
        value = values[len(calls)]
        calls.append(value)
        return {
            "metrics": {"detection_acc_avg_dom": value},
            "calibration": {
                "foreground_logit_offset": 0.0,
                "raw_zero_offset_ADA": value,
            },
        }, predictions

    monkeypatch.setattr(wheat_eval, "calibrated_wheat_evaluation", score)
    (tmp_path / "_annotations.coco.json").write_text(json.dumps({"images": [{"id": 1}]}))

    class TinyModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1)
            self._should_compute_val_loss = False
            self.model_config = RFDETRNanoConfig(pretrain_weights=None)
            self.train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))

        def training_step(self, batch, batch_idx):
            return self.model(batch).square().mean()

        def validation_step(self, batch, batch_idx):
            return {
                "results": [
                    {"boxes": torch.tensor([[0.0, 0, 1, 1]]), "scores": torch.tensor([0.9])}
                ],
                "targets": [{"image_id": torch.tensor(1)}],
            }

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    module = TinyModule()

    class KnownEMA(pl.Callback):
        def get_ema_model_state_dict(self):
            return module.model.state_dict()

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        callbacks=[BestModelCallback(str(tmp_path)), KnownEMA()],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    _configure_wheat_selection(trainer, module, tmp_path, tmp_path, DetectorTrainingConfig())
    loader = torch.utils.data.DataLoader(torch.ones(1, 1), batch_size=1)
    # Native RF-DETR uses a list of target dictionaries for each image batch.
    class NativeBatch(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.ones(1, 1), [{"image_id": torch.tensor(1)}]

    validation_loader = torch.utils.data.DataLoader(NativeBatch(), batch_size=None)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=validation_loader)
    # Real Lightning rounds the logged scalar, but the native checkpoint writer
    # and calibration artifact must share the exact higher-precision decision.
    assert trainer.callback_metrics["val/detection_acc_avg_dom"].dtype == torch.float32
    selected = json.loads((tmp_path / "validation-best.json").read_text())
    checkpoint = torch.load(tmp_path / "checkpoint_best_ema.pth", weights_only=True)
    assert selected["epoch_1based"] == 2
    assert selected["metrics"]["detection_acc_avg_dom"] == values[1]
    assert checkpoint["epoch"] == 1
    assert all(torch.equal(v, checkpoint["model"][k]) for k, v in module.model.state_dict().items())
