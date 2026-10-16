"""Reproducible RF-DETR fine-tuning and raw-query export for reward optimization.

The detector is supervised first. Reinforcement learning is a separate, explicitly
measured policy stage in ``detection_rl``. Training sees a staged dataset containing
only train/valid; the final test split is never passed to RF-DETR's trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _category_schema(coco: dict) -> dict[str, str]:
    """Mirror RF-DETR's parent filtering, retaining unannotated leaf classes."""
    rows = sorted(coco["categories"], key=lambda row: row["id"])
    used = {row["category_id"] for row in coco["annotations"]}
    parents = {
        row.get("supercategory")
        for row in rows
        if row.get("supercategory") not in {None, "", "none", "null"}
        and row.get("supercategory") != row["name"]
    }
    kept = [row for row in rows if row["name"] not in parents or row["id"] in used]
    return {str(row["id"]): row["name"] for row in (kept or rows)}


def inspect_split(directory: Path) -> dict[str, Any]:
    """Validate a COCO split and fingerprint labels plus actual image contents."""
    from PIL import Image

    directory = directory.resolve()
    annotation = directory / "_annotations.coco.json"
    coco = json.loads(annotation.read_text())
    images = coco["images"]
    if not images:
        raise ValueError("Dataset split is empty")
    by_id = {image["id"]: image for image in images}
    if len(by_id) != len(images):
        raise ValueError("Duplicate COCO image IDs")
    categories = {row["id"]: row["name"] for row in coco["categories"]}
    if len(categories) != len(coco["categories"]):
        raise ValueError("Duplicate COCO category IDs")
    annotation_ids: set[int] = set()
    annotations: dict[int, list[dict]] = defaultdict(list)
    for row in coco["annotations"]:
        if row["id"] in annotation_ids:
            raise ValueError("Duplicate COCO annotation IDs")
        annotation_ids.add(row["id"])
        if row["image_id"] not in by_id or row["category_id"] not in categories:
            raise ValueError("Annotation references an unknown image/category")
        box = row["bbox"]
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
            raise ValueError("Invalid bounding box")
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError("Bounding boxes must have positive area")
        image = by_id[row["image_id"]]
        if (
            box[0] < 0
            or box[1] < 0
            or box[0] + box[2] > image["width"] + 1e-4
            or box[1] + box[3] > image["height"] + 1e-4
        ):
            raise ValueError("Ground-truth bounding box extends outside its image")
        annotations[row["image_id"]].append(row)
    digest = hashlib.sha256()
    hashes: list[str] = []
    for image in sorted(images, key=lambda row: str(row["id"])):
        path = (directory / image["file_name"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("COCO image path escapes split directory")
        if image["width"] <= 0 or image["height"] <= 0:
            raise ValueError("Image dimensions must be positive")
        with Image.open(path) as decoded:
            if decoded.size != (image["width"], image["height"]):
                raise ValueError("COCO dimensions do not match the image")
        image_hash = _sha256(path)
        hashes.append(image_hash)
        row = {
            "id": image["id"],
            "sha256": image_hash,
            "width": image["width"],
            "height": image["height"],
            "annotations": sorted(annotations[image["id"]], key=lambda item: item["id"]),
        }
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    digest.update(json.dumps(coco["categories"], sort_keys=True).encode())
    return {
        "images": len(images),
        "annotations": len(coco["annotations"]),
        "categories": _category_schema(coco),
        "fingerprint": digest.hexdigest(),
        "annotation_sha256": _sha256(annotation),
        "image_sha256": hashes,
    }


def stage_training_data(source: Path, destination: Path) -> dict[str, Any]:
    """Expose train/validation and reject exact file and decoded-pixel leakage.

    Near-duplicate/session auditing remains a separate scientific assessment.
    The local symlinks avoid copying images; they are not a security sandbox.
    """
    from vision_rl.data import image_hash

    source, destination = source.resolve(), destination.resolve()
    train = inspect_split(source / "train")
    validation = inspect_split(source / "valid")
    if set(train["image_sha256"]) & set(validation["image_sha256"]):
        raise ValueError("Identical image bytes occur in train and validation")
    documents = {
        split: json.loads((source / split / "_annotations.coco.json").read_text())
        for split in ("train", "valid")
    }
    schemas = [
        sorted(documents[split]["categories"], key=lambda row: row["id"])
        for split in ("train", "valid")
    ]
    if schemas[0] != schemas[1]:
        raise ValueError("Training and validation category schemas differ")
    if any(
        str(row["category_id"]) not in train["categories"]
        for row in documents["valid"]["annotations"]
    ):
        raise ValueError("Validation annotates a category excluded by the training class mapping")
    # RF-DETR derives the mapping once from train, including annotated parents.
    validation["categories"] = dict(train["categories"])
    pixel_sets = []
    for split in ("train", "valid"):
        directory = source / split
        rows = json.loads((directory / "_annotations.coco.json").read_text())["images"]
        pixel_sets.append({image_hash(directory / row["file_name"]) for row in rows})
    if pixel_sets[0] & pixel_sets[1]:
        raise ValueError("Identical decoded pixels occur in train and validation")
    destination.mkdir(parents=True, exist_ok=True)
    if set(p.name for p in destination.iterdir()) - {"train", "valid"}:
        raise ValueError("Staged training directory must contain only train and valid")
    for split in ("train", "valid"):
        link = destination / split
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != source / split:
                raise ValueError("Existing staged dataset points at a different source")
        else:
            link.symlink_to(source / split, target_is_directory=True)
    return {"train": train, "validation": validation}


@dataclass(frozen=True)
class DetectorTrainingConfig:
    architecture: str = "nano"
    device: str = "mps"
    epochs: int = 40
    batch_size: int = 4
    grad_accum_steps: int = 4
    eval_batch_size: int = 8
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    seed: int = 42
    workers: int = 2
    cpu_threads: int = 4
    mps_memory_gib: float = 16.0
    early_stopping_patience: int = 12
    multi_scale: bool = False
    preflight_batches: int = 0

    def validate(self) -> None:
        if self.architecture not in {"nano", "small"}:
            raise ValueError("Architecture must be nano or small")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Device must be cpu, mps, or cuda")
        for field in ("epochs", "batch_size", "grad_accum_steps", "eval_batch_size", "cpu_threads"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.workers < 0 or self.preflight_batches < 0 or self.mps_memory_gib <= 0:
            raise ValueError("Invalid resource limit")
        if self.preflight_batches and self.preflight_batches < self.grad_accum_steps:
            raise ValueError("Preflight must include at least one optimizer step")
        if not all(
            math.isfinite(v) and v > 0 for v in (self.lr, self.lr_encoder, self.weight_decay)
        ):
            raise ValueError("Optimizer settings must be finite and positive")


def train_detector(
    source: Path,
    output: Path,
    config: DetectorTrainingConfig,
    *,
    resume: bool = False,
    warmstart: Path | None = None,
    resolution: int | None = None,
    warmstart_resolution: int | None = None,
    selection_metric: str = "coco_ap",
    pause_after_epoch: int | None = None,
) -> dict:
    """Fine-tune the complete transformer with optimizer/scheduler/EMA resume.

    RF-DETR reseeds shuffled loaders on each fit invocation. Resume restores
    training state but does not reproduce an uninterrupted stochastic trajectory.
    """
    import fcntl

    config.validate()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another trainer owns this output directory") from exc
        try:
            return _train_detector_locked(
                source,
                output,
                config,
                resume=resume,
                warmstart=warmstart,
                resolution=resolution,
                warmstart_resolution=warmstart_resolution,
                selection_metric=selection_metric,
                pause_after_epoch=pause_after_epoch,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _train_detector_locked(
    source: Path,
    output: Path,
    config: DetectorTrainingConfig,
    *,
    resume: bool,
    warmstart: Path | None,
    resolution: int | None,
    warmstart_resolution: int | None,
    selection_metric: str,
    pause_after_epoch: int | None,
) -> dict:
    config.validate()
    if selection_metric not in {"coco_ap", "wheat_ada"}:
        raise ValueError("Selection metric must be coco_ap or wheat_ada")
    if pause_after_epoch is not None and (
        selection_metric != "wheat_ada"
        or type(pause_after_epoch) is not int
        or not 1 <= pause_after_epoch <= config.epochs
    ):
        raise ValueError("Planned epoch pause requires wheat_ada and an epoch within this fit")
    _validate_resolution(resolution)
    _validate_resolution(warmstart_resolution)
    if warmstart_resolution is not None and warmstart is None:
        raise ValueError("warmstart_resolution requires a warm-start checkpoint")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "training-plan.json"
    staged = output / "training-data"
    splits = stage_training_data(source, staged)
    plan = {
        "format_version": 1,
        "algorithm": "supervised_rfdetr_finetuning",
        "configuration": asdict(config),
        "splits": splits,
        "selection_metric": "validation COCO AP@[0.50:0.95], maxDets=500",
        "test_access": False,
        "software": {
            name: version(name)
            for name in (
                "rfdetr",
                "torch",
                "torchvision",
                "pytorch-lightning",
                "transformers",
                "pycocotools",
            )
        },
    }
    if resolution is not None:
        plan["model_resolution"] = resolution
    if selection_metric == "wheat_ada":
        plan["selection_metric"] = "validation calibrated WILDS equal-domain detection accuracy"
        plan["selection_protocol"] = _wheat_selection_plan(source / "valid")
        plan["selection_protocol"]["pause_after_epoch"] = pause_after_epoch
    if warmstart is not None:
        if warmstart.suffix != ".safetensors":
            raise ValueError("Warm start requires a numeric .safetensors checkpoint")
        plan["initialization"] = {
            "kind": "warm_start",
            "source_sha256": _sha256(warmstart),
            "source_resolution": warmstart_resolution,
            "optimizer": "fresh at phase start; full-state restoration only with --resume",
            "position_embeddings": "official RF-DETR bicubic grid interpolation",
        }
    if plan_path.exists():
        if not resume:
            raise ValueError("Run already exists; use --resume or choose a fresh output")
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError(
                "Resume requires the identical dataset, software, and training configuration"
            )
        if not (output / "last.ckpt").is_file():
            raise ValueError("No full last.ckpt exists; cannot resume optimizer state")
    elif resume:
        raise ValueError("Cannot resume a run without its original frozen plan")
    else:
        _write_json(plan_path, plan)

    import psutil
    import pytorch_lightning as pl
    import torch
    from rfdetr import RFDETRNano, RFDETRSmall
    from rfdetr.config import TrainConfig
    from rfdetr.models.weights import interpolate_position_embeddings
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    torch.set_num_threads(config.cpu_threads)
    pl.seed_everything(config.seed, workers=True)
    if config.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS is unavailable")
        recommended = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(
            min(1.0, config.mps_memory_gib * 1024**3 / recommended)
        )
    model_class = RFDETRNano if config.architecture == "nano" else RFDETRSmall
    num_classes = len(splits["train"]["categories"])
    initial_state = None
    if warmstart is not None:
        detector = load_detector_safetensors(
            warmstart,
            architecture=config.architecture,
            num_classes=num_classes,
            resolution=warmstart_resolution,
            device="cpu",
        )
        initial_state = detector.model.model.state_dict()
        initial_weights_sha = _sha256(warmstart)
    else:
        kwargs = (
            {}
            if resolution is None
            else {
                "resolution": resolution,
                "positional_encoding_size": resolution // 16,
            }
        )
        detector = model_class(device="cpu", num_classes=num_classes, amp=False, **kwargs)
        initial_weights_sha = _sha256(Path(detector.model_config.pretrain_weights))
    model_config = detector.model_config
    model_config.model_name = model_class.__name__
    if warmstart is not None and resolution is not None:
        model_config.resolution = resolution
        model_config.positional_encoding_size = resolution // model_config.patch_size
        interpolate_position_embeddings(initial_state, model_config.positional_encoding_size)
    del detector
    train_config = TrainConfig(
        dataset_dir=str(staged),
        output_dir=str(output),
        epochs=config.epochs,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accum_steps,
        eval_batch_size=config.eval_batch_size,
        lr=config.lr,
        lr_encoder=config.lr_encoder,
        weight_decay=config.weight_decay,
        seed=config.seed,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
        multi_scale=config.multi_scale,
        expanded_scales=False,
        use_ema=True,
        eval_base_model=False,
        run_test=False,
        eval_max_dets=500,
        early_stopping=True,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=True,
        checkpoint_interval=config.epochs,
        lr_scheduler="cosine",
        warmup_epochs=1,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        progress_bar=None,
        num_sanity_val_steps=0,
        compute_train_metrics=False,
        compute_val_loss=False,
        notes={"training_plan_sha256": _sha256(plan_path), "test_access": False},
        resume=str(output / "last.ckpt") if resume else None,
    )
    module = RFDETRModelModule(model_config, train_config)
    if initial_state is not None:
        module.model.load_state_dict(initial_state, strict=True)
        del initial_state
    data = RFDETRDataModule(model_config, train_config)
    trainer_options: dict[str, Any] = {"enable_model_summary": False, "log_every_n_steps": 10}
    if config.preflight_batches:
        trainer_options.update(
            max_epochs=1,
            limit_train_batches=config.preflight_batches,
            limit_val_batches=config.preflight_batches,
        )
    trainer = build_trainer(
        train_config,
        model_config,
        accelerator="gpu" if config.device == "cuda" else config.device,
        **trainer_options,
    )
    if selection_metric == "wheat_ada":
        _configure_wheat_selection(
            trainer, module, source / "valid", output, config, pause_after_epoch=pause_after_epoch
        )
    started = time.monotonic()
    process = psutil.Process()

    class Telemetry(pl.Callback):
        def __init__(self):
            self.last_log = -1

        def emit(self, trainer, event: str):
            row = {
                "event": event,
                "epoch": trainer.current_epoch,
                "optimizer_step": trainer.global_step,
                "elapsed_seconds": time.monotonic() - started,
                "rss_bytes": process.memory_info().rss,
                "metrics": {
                    k: float(v.detach().cpu())
                    for k, v in trainer.callback_metrics.items()
                    if hasattr(v, "numel") and v.numel() == 1
                },
            }
            if config.device == "mps":
                row.update(
                    mps_allocated_bytes=torch.mps.current_allocated_memory(),
                    mps_driver_bytes=torch.mps.driver_allocated_memory(),
                )
            with (output / "telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step != self.last_log and (
                trainer.global_step <= 3 or trainer.global_step % 20 == 0
            ):
                self.last_log = trainer.global_step
                self.emit(trainer, "train_progress")

        def on_validation_end(self, trainer, pl_module):
            self.emit(trainer, "validation_end")
            if (output / "STOP_AFTER_EPOCH").exists():
                trainer.should_stop = True

        def on_validation_start(self, trainer, pl_module):
            self.emit(trainer, "validation_start")

    trainer.callbacks.append(Telemetry())
    manifest = {
        "attempt_id": uuid.uuid4().hex,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "model_parameters": sum(p.numel() for p in module.parameters()),
        "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
        "initial_weights_sha256": initial_weights_sha,
        "initialization": "warm_start" if warmstart is not None else "COCO_pretrained",
        "model_configuration": model_config.model_dump(mode="json"),
        "training_configuration": train_config.model_dump(mode="json"),
        "resume_semantics": "optimizer/scheduler/EMA restored; shuffled-loader RNG restarts",
    }
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    source_snapshot = attempts / f"{manifest['attempt_id']}.trainer.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    manifest["trainer_source_sha256"] = _sha256(source_snapshot)
    manifest["trainer_source_snapshot"] = str(source_snapshot)
    attempt_path = attempts / f"{manifest['attempt_id']}.json"
    _write_json(attempt_path, manifest)
    previous_runtime = output / "runtime.json"
    if previous_runtime.exists():
        previous = json.loads(previous_runtime.read_text())
        previous_id = previous.get("attempt_id", "legacy-" + uuid.uuid4().hex)
        previous_path = attempts / f"{previous_id}.json"
        if not previous_path.exists():
            _write_json(previous_path, previous)
    _write_json(output / "runtime.json", manifest)
    try:
        trainer.fit(module, data, ckpt_path=train_config.resume)
    except BaseException as exc:
        elapsed = time.monotonic() - started
        failed = {
            **manifest,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(attempt_path, failed)
        _write_json(output / "runtime.json", failed)
        _write_json(
            output / "failure.json",
            {
                "attempt_id": manifest["attempt_id"],
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": elapsed,
            },
        )
        raise
    result = {
        "attempt_id": manifest["attempt_id"],
        "status": (
            "paused"
            if (output / "STOP_AFTER_EPOCH").exists() or trainer.interrupted
            else "preflight_completed"
            if config.preflight_batches
            else "completed"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "elapsed_seconds_scope": "this fit invocation; excludes model setup and prior attempts",
        "optimizer_steps": trainer.global_step,
        "epochs_completed": trainer.current_epoch,
        "metrics": {
            k: float(v.detach().cpu())
            for k, v in trainer.callback_metrics.items()
            if hasattr(v, "numel") and v.numel() == 1
        },
        "checkpoints": {p.name: _sha256(p) for p in output.glob("*.pth")},
        "test_access": False,
    }
    _write_json(attempt_path, {**manifest, **result})
    _write_json(output / "runtime.json", {**manifest, **result})
    _write_json(output / "result.json", result)
    summarize_training_run(output)
    return result


def summarize_training_run(output: Path) -> dict:
    """Audit total recorded fit time and best validation separately from final epoch.

    Also supports the original measured run, whose first interrupted attempt was
    recorded in failure.json before per-attempt manifests were introduced.
    """
    result = json.loads((output / "result.json").read_text())
    attempts = [json.loads(path.read_text()) for path in (output / "attempts").glob("*.json")]
    recorded = {row.get("attempt_id") for row in attempts if "elapsed_seconds" in row}
    durations = [
        {
            "attempt_id": row.get("attempt_id"),
            "status": row.get("status", "unknown"),
            "seconds": row["elapsed_seconds"],
        }
        for row in attempts
        if "elapsed_seconds" in row
    ]
    if result.get("attempt_id", "legacy-final") not in recorded:
        durations.append(
            {
                "attempt_id": result.get("attempt_id", "legacy-final"),
                "status": result["status"],
                "seconds": result["elapsed_seconds"],
            }
        )
    failure_path = output / "failure.json"
    if failure_path.exists():
        if failure_path.stat().st_mtime_ns > (output / "result.json").stat().st_mtime_ns:
            raise ValueError("Latest attempt failed; the completed result is stale")
        failure = json.loads(failure_path.read_text())
        if failure.get("attempt_id", "legacy-interrupted") not in recorded:
            durations.append(
                {
                    "attempt_id": failure.get("attempt_id", "legacy-interrupted"),
                    "status": "historical_failure",
                    "seconds": failure["elapsed_seconds"],
                }
            )
    events = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
    validation = [
        row
        for row in events
        if row["event"] == "validation_end" and "val/mAP_50_95" in row["metrics"]
    ]
    best = max(validation, key=lambda row: row["metrics"]["val/mAP_50_95"])
    plan_path = output / "training-plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    wheat = "selection_protocol" in plan
    if wheat:
        selected = json.loads((output / "validation-best.json").read_text())
        best = next(
            row for row in reversed(validation) if row["epoch"] + 1 == selected["epoch_1based"]
        )
    audit = {
        "status": result["status"],
        "attempts": durations,
        "total_recorded_active_fit_seconds": sum(row["seconds"] for row in durations),
        "timing_scope": "fit process wall time; excludes model setup and pauses between attempts",
        "selection": {
            "epoch_1based": best["epoch"] + 1,
            "metric": "validation EMA COCO AP@[0.50:0.95], maxDets=500",
            "AP": best["metrics"]["val/mAP_50_95"],
            "AP50": best["metrics"].get("val/mAP_50"),
            "AP75": best["metrics"].get("val/mAP_75"),
        },
        "final_epoch_metrics": result["metrics"],
        "resume_semantics": "stateful optimizer/scheduler/EMA resume; not exact stochastic replay",
        "test_access": False,
    }
    if wheat:
        audit["selection"].update(
            metric=plan["selection_metric"],
            ADA=selected["metrics"]["detection_acc_avg_dom"],
            foreground_logit_offset=selected["calibration"]["foreground_logit_offset"],
            raw_zero_offset_ADA=selected["calibration"]["raw_zero_offset_ADA"],
        )
    _write_json(output / "training-evidence.json", audit)
    return audit


def export_queries(
    checkpoint: Path,
    split_directory: Path,
    output: Path,
    *,
    split: str,
    device: str = "mps",
    queries: int = 128,
    batch_size: int = 8,
    max_images: int | None = None,
    architecture: str = "nano",
    cpu_threads: int = 4,
    resolution: int | None = None,
) -> dict:
    """Cache real decoder embeddings and proposals without thresholding.

    Uses RF-DETR's validation preprocessing (full RGB decode, no JPEG draft),
    avoiding the package's different public ``predict`` resize defaults. Outputs
    all foreground queries to COCO JSON and the top ``queries`` to the RL cache.
    The test cache is allowed only as inference input; RL training rejects it.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    if not 1 <= queries <= 300 or batch_size < 1:
        raise ValueError("Invalid query or batch size")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")
    if not 1 <= cpu_threads <= 64:
        raise ValueError("cpu_threads must be in 1..64")
    if output.exists():
        raise ValueError("Query cache already exists; outputs are immutable")
    import torch
    from rfdetr import RFDETR
    from rfdetr.datasets.coco import (
        CocoDetection,
        draft_size_for_transforms,
        make_coco_transforms_square_div_64,
    )
    from torchvision.ops import box_convert

    torch.set_num_threads(cpu_threads)
    split_directory = split_directory.resolve()
    annotation_path = split_directory / "_annotations.coco.json"
    inspected = inspect_split(split_directory)
    category_ids = sorted(int(key) for key in inspected["categories"])
    if len(category_ids) != 1:
        raise ValueError("The current RL query cache requires exactly one foreground class")
    category_id = category_ids[0]
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint.suffix == ".safetensors":
        detector = load_detector_safetensors(
            checkpoint,
            architecture=architecture,
            device=device,
            resolution=resolution,
        )
    else:
        if resolution is not None:
            raise ValueError("Explicit inference resolution requires a numeric checkpoint")
        detector = RFDETR.from_checkpoint(str(checkpoint.resolve()), device=device)
        source_tensors = _checkpoint_tensors(checkpoint)
        loaded_tensors = detector.model.model.state_dict()
        if set(source_tensors) != set(loaded_tensors) or any(
            source_tensors[key].shape != loaded_tensors[key].shape
            or not torch.equal(source_tensors[key], loaded_tensors[key].cpu())
            for key in source_tensors
        ):
            raise ValueError(
                "Native checkpoint reconstruction changed trained weights; export exact "
                "safetensors and provide the trusted training resolution"
            )
        del source_tensors, loaded_tensors
    # RF-DETR lazily moves a native checkpoint only inside its public predict()
    # path. Raw-query extraction bypasses predict(), so move the actual module.
    model = detector.model.model.to(device).eval()
    resolution = detector.model_config.resolution
    transforms = make_coco_transforms_square_div_64("val", resolution)
    draft_size = draft_size_for_transforms(
        "val",
        resolution,
        multi_scale=False,
        expanded_scales=False,
        patch_size=16,
        num_windows=2,
        scale_jitter=True,
        include_masks=False,
    )
    decoded_hashes: dict[int, str] = {}

    class HashedCocoDetection(CocoDetection):
        def _decode_image(self, image_id):
            image, scales = super()._decode_image(image_id)
            if scales != (1.0, 1.0):
                raise RuntimeError("Validation pixel fingerprints require a full RGB decode")
            # Reuse the native decoder's exact RGB pixels before transforms.
            # This matches image_hash() without decoding every JPEG twice.
            decoded_hashes[image_id] = hashlib.sha256(
                str(image.size).encode() + image.tobytes()
            ).hexdigest()
            return image, scales

    dataset = HashedCocoDetection(
        split_directory,
        annotation_path,
        transforms=transforms,
        remap_category_ids=True,
        cat2label={category_id: 0},
        draft_size=draft_size,
    )
    ids = dataset.ids[:max_images] if max_images else dataset.ids
    captured: list = []

    def capture_embeddings(_module, inputs, _outputs):
        captured.append(inputs[0][-1].detach())

    hook = model.class_embed.register_forward_hook(capture_embeddings)
    boxes, logits, embeddings, targets, image_sizes, pixel_hashes = [], [], [], [], [], []
    predictions: list[dict] = []
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start : start + batch_size]
                batch = [dataset[index][0] for index in range(start, start + len(batch_ids))]
                captured.clear()
                raw = model(torch.stack(batch).to(device))
                if len(captured) != 1:
                    raise RuntimeError("RF-DETR decoder embedding hook changed; cache aborted")
                features = captured[0]
                raw_logits = raw["pred_logits"][..., 0]
                raw_boxes = box_convert(raw["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
                # Stable ordering makes tied proposals reproducible across exports.
                ordering = torch.argsort(raw_logits, dim=-1, descending=True, stable=True)
                ordering = ordering[:, :queries]
                selected_boxes = torch.gather(raw_boxes, 1, ordering[..., None].expand(-1, -1, 4))
                selected_logits = torch.gather(raw_logits, 1, ordering)
                selected_features = torch.gather(
                    features, 1, ordering[..., None].expand(-1, -1, features.shape[-1])
                )
                boxes.extend(selected_boxes.cpu().unbind())
                logits.extend(selected_logits.cpu().unbind())
                embeddings.extend(selected_features.cpu().unbind())
                raw_boxes, raw_scores = raw_boxes.cpu(), raw_logits.sigmoid().cpu()
                for index, image_id in enumerate(batch_ids):
                    image = dataset.coco.imgs[image_id]
                    width, height = image["width"], image["height"]
                    image_sizes.append([width, height])
                    pixel_hashes.append(decoded_hashes.pop(image_id))
                    rows = [
                        row
                        for row in dataset.coco.imgToAnns[image_id]
                        if row["category_id"] == category_id and not row.get("iscrowd", 0)
                    ]
                    gt = torch.tensor([row["bbox"] for row in rows], dtype=torch.float32).reshape(
                        -1, 4
                    )
                    gt = box_convert(gt, in_fmt="xywh", out_fmt="xyxy")
                    targets.append(gt / torch.tensor([width, height, width, height]))
                    absolute = raw_boxes[index] * torch.tensor([width, height, width, height])
                    xywh = box_convert(absolute, in_fmt="xyxy", out_fmt="xywh")
                    for box, score in zip(xywh.tolist(), raw_scores[index].tolist(), strict=True):
                        predictions.append(
                            {
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": box,
                                "score": score,
                            }
                        )
                if start % (batch_size * 10) == 0:
                    print(
                        json.dumps(
                            {
                                "cached_images": start + len(batch_ids),
                                "total": len(ids),
                                "elapsed_seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
    finally:
        hook.remove()
    cache = {
        "version": 1,
        "split": split,
        "ids": pixel_hashes,
        "coco_image_ids": ids,
        "boxes": torch.stack(boxes),
        "logits": torch.stack(logits),
        "embeddings": torch.stack(embeddings),
        "targets": targets,
        "image_sizes": torch.tensor(image_sizes),
        "pixel_sha256": pixel_hashes,
        "metadata": {
            "model_sha256": checkpoint_hash,
            "dataset_fingerprint": inspected["fingerprint"],
            "annotation_path": str(annotation_path),
            "category_id": category_id,
            "image_sizes_order": "width,height",
            "box_format": "normalized_xyxy",
            "embedding_source": "last decoder layer, class_embed input",
            "preprocessing": "rfdetr validation transforms; full RGB decode (draft_size=None)",
            "source_queries": int(raw_logits.shape[1]),
            "retained_queries": queries,
            "truncated_images": max_images is not None,
            "device": device,
            "batch_size": batch_size,
            "cpu_threads": cpu_threads,
            "resolution": resolution,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(cache, temporary)
    temporary.replace(output)
    _write_json(output.with_suffix(".coco.json"), predictions)
    summary = {
        "images": len(ids),
        "queries": queries,
        "embedding_dimension": embeddings[0].shape[-1],
        "elapsed_seconds": time.monotonic() - started,
        "cache_sha256": _sha256(output),
        "metadata": cache["metadata"],
    }
    _write_json(output.with_suffix(".json"), summary)
    return summary


def _checkpoint_tensors(checkpoint: Path) -> dict:
    """Read our local trainer's numeric state without reconstructing its model."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("Local checkpoint must contain a nonempty model tensor dictionary")
    if any(
        not isinstance(key, str)
        or not isinstance(value, torch.Tensor)
        or not torch.isfinite(value).all()
        for key, value in state.items()
    ):
        raise ValueError("Local checkpoint contains invalid or nonfinite model tensors")
    return state


def export_detector_safetensors(checkpoint: Path, output: Path) -> dict:
    """Preserve exact locally trained tensors in a numeric-only submission.

    RF-DETR's stripped best-total checkpoint can omit custom resolution metadata.
    Reconstructing a detector here would silently resize its positional embeddings;
    direct numeric extraction preserves the trained weights exactly.
    """
    from safetensors.torch import save_file

    if output.exists():
        raise ValueError("Detector submission already exists")
    source_sha = _sha256(checkpoint)
    tensors = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in _checkpoint_tensors(checkpoint).items()
    }
    if source_sha != _sha256(checkpoint):
        raise ValueError("Training checkpoint changed during export; copy a stable snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_file(tensors, temporary)
    temporary.replace(output)
    report = {
        "source_sha256": source_sha,
        "safetensors_sha256": _sha256(output),
        "parameters": sum(tensor.numel() for tensor in tensors.values()),
        "tensor_count": len(tensors),
        "format": "safetensors",
        "export_semantics": "exact stored tensors, without model reconstruction or interpolation",
        "positional_embedding_shapes": {
            key: list(value.shape)
            for key, value in tensors.items()
            if key.endswith("position_embeddings")
        },
    }
    _write_json(output.with_suffix(".json"), report)
    return report


def load_detector_safetensors(
    path: Path,
    *,
    architecture: str = "nano",
    num_classes: int = 1,
    device: str = "cpu",
    resolution: int | None = None,
):
    """Load numeric tensors into a verifier-owned fixed architecture offline.

    The submitted file cannot supply Python code, architecture settings, class
    mappings, transforms, or checkpoint pickle objects. Shapes are checked before
    tensor materialization; nonfinite weights and unexpected dtypes are rejected.
    """
    import torch
    from rfdetr import RFDETRNano, RFDETRSmall
    from safetensors import safe_open

    if architecture not in {"nano", "small"} or not 1 <= num_classes <= 256:
        raise ValueError("Unsupported trusted detector architecture")
    _validate_resolution(resolution)
    if not path.is_file() or path.stat().st_size > 200 * 1024**2:
        raise ValueError("Detector tensor file missing or exceeds 200 MiB")
    model_class = RFDETRNano if architecture == "nano" else RFDETRSmall
    kwargs = (
        {}
        if resolution is None
        else {
            "resolution": resolution,
            "positional_encoding_size": resolution // 16,
        }
    )
    detector = model_class(
        pretrain_weights=None, num_classes=num_classes, device="cpu", amp=False, **kwargs
    )
    expected = detector.model.model.state_dict()
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as stream:
        if set(stream.keys()) != set(expected):
            raise ValueError("Detector tensor keys do not match the trusted architecture")
        for key, reference in expected.items():
            if list(stream.get_slice(key).get_shape()) != list(reference.shape):
                raise ValueError(f"Invalid detector tensor shape: {key}")
        for key, reference in expected.items():
            tensor = stream.get_tensor(key)
            if tensor.dtype != reference.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid detector tensor dtype or nonfinite value: {key}")
            tensors[key] = tensor
    detector.model.model.load_state_dict(tensors, strict=True)
    detector.model.model.to(device).eval()
    detector.model.device = torch.device(device)
    return detector


def _validate_resolution(resolution: int | None) -> None:
    if resolution is not None and (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or not 128 <= resolution <= 1024
        or resolution % 32
    ):
        raise ValueError("Resolution must be a multiple of 32 in 128..1024")


def _wheat_selection_plan(validation_directory: Path) -> dict:
    """Pin the shared calibration and original-label scorer before any optimizer step."""
    import inspect

    from vision_rl.detection_inference import decode_queries
    from vision_rl.wheat_eval import BENCHMARK_ID, WHEAT_LOGIT_OFFSETS, evaluate_wheat

    annotation = validation_directory / "_annotations.coco.json"
    document = json.loads(annotation.read_text())
    if document.get("info", {}).get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Wheat selection requires the audited author-HF benchmark contract")
    # Empty predictions still invoke the unchanged official implementation. This
    # preflight checks dependencies/provenance before starting a potentially long fit.
    reference = evaluate_wheat(annotation, [], image_ids=[document["images"][0]["id"]])
    return {
        "benchmark_id": BENCHMARK_ID,
        "primary_key": "detection_acc_avg_dom",
        "scorer": reference["scorer"],
        "trainer_source_sha256": _sha256(Path(__file__)),
        "portable_decoder_source_sha256": _sha256(Path(inspect.getfile(decode_queries))),
        "annotation_sha256": _sha256(annotation),
        "foreground_logit_offsets": list(WHEAT_LOGIT_OFFSETS),
        "calibration_ties": "smallest absolute offset, then numeric offset",
        "prediction_queries": 300,
        "prediction_order": "stable foreground-score descending; native query index breaks ties",
        "ground_truth": "original annotation boxes, including duplicate/out-of-frame extents",
        "checkpoint_ties": "earliest epoch",
        "validation_images": len(document["images"]),
        "test_access": False,
    }


def _foreground_results(raw, original_sizes):
    """Return all native foreground queries, without background top-k competition.

    RF-DETR's default postprocessor ranks foreground and background together.
    The wheat policy contract instead deploys all 300 foreground queries. This
    conversion reuses the existing validation forward. The shared portable
    decoder preserves float64 score/geometry arithmetic before official float32
    scoring, including close confidence-threshold decisions. MPS has no float64,
    so this small postprocessing step runs on CPU.
    """
    import torch

    from vision_rl.detection_inference import decode_queries

    boxes, logits = raw["pred_boxes"], raw["pred_logits"]
    if boxes.ndim != 3 or boxes.shape[1:] != (300, 4) or logits.shape != (*boxes.shape[:2], 2):
        raise ValueError("Wheat validation requires 300 queries and one foreground class")
    if original_sizes.shape != (boxes.shape[0], 2):
        raise ValueError("Original sizes must have shape [batch, height/width]")
    result = []
    for box, logit, size in zip(
        boxes.detach().cpu().numpy(),
        logits.detach().cpu().numpy(),
        original_sizes.detach().cpu().tolist(),
        strict=True,
    ):
        decoded = decode_queries(box, logit, (1,))
        height, width = size
        xyxy = torch.tensor([row["box"] for row in decoded], dtype=torch.float64)
        result.append(
            {
                "boxes": xyxy * torch.tensor([width, height, width, height]),
                "scores": torch.tensor([row["score"] for row in decoded], dtype=torch.float64),
                "labels": torch.zeros(300, dtype=torch.long),
            }
        )
    return result


def _configure_wheat_selection(
    trainer, module, validation_directory, output, config, *, pause_after_epoch=None
):
    """Use original-label calibrated ADA for EMA selection and early stopping.

    Callback state is included in the native full Lightning checkpoint. The
    annotation file supplies targets by image ID; RF-DETR's clipped augmentation
    targets are never substituted into the official wheat evaluator.
    """
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping
    from rfdetr.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping

    from vision_rl.wheat_eval import calibrated_wheat_evaluation

    annotation = (validation_directory / "_annotations.coco.json").resolve()
    annotation_sha = _sha256(annotation)
    expected_ids = {row["id"] for row in json.loads(annotation.read_text())["images"]}

    class ForegroundPostprocess(torch.nn.Module):
        def forward(self, raw, original_sizes):
            return _foreground_results(raw, original_sizes)

    module.postprocess = ForegroundPostprocess()
    selection_callback = None

    class WheatBestModel(BestModelCallback):
        def on_validation_end(self, trainer, pl_module):
            # Lightning logs Python scalars in float32 and may overwrite direct
            # callback_metrics assignments. Reuse the selector's exact official
            # decision so close scores cannot pair one epoch's calibration/report
            # with another epoch's weights. Native full checkpoint state and EMA
            # serialization are retained; this is pinned to RF-DETR 1.10.0.
            if trainer.sanity_checking or not trainer.is_global_zero:
                return
            report = selection_callback.current_report
            if report is None or report["epoch_1based"] != trainer.current_epoch + 1:
                raise RuntimeError("Missing current-epoch wheat selection decision")
            if report["selected"]:
                self._best_ema = report["metrics"]["detection_acc_avg_dom"]
                state = self._get_ema_model_state_dict(trainer, pl_module)
                self._write_ema_checkpoint(
                    trainer, pl_module, state, output / "checkpoint_best_ema.pth"
                )

    callbacks = []
    for callback in trainer.callbacks:
        if isinstance(callback, BestModelCallback):
            callback = WheatBestModel(
                output_dir=str(output),
                monitor_regular="val/detection_acc_avg_dom",
                monitor_ema="val/ema_detection_acc_avg_dom",
                evaluates_base_model=False,
                run_test=False,
            )
            # Upstream initializes this to zero, which would select the final
            # weights when every epoch scores zero. Start at -1 for earliest-tie
            # selection; this finite value survives upstream state restoration.
            callback._best_ema = -1.0
        elif isinstance(callback, RFDETREarlyStopping):
            callback = EarlyStopping(
                monitor="val/ema_detection_acc_avg_dom",
                mode="max",
                patience=config.early_stopping_patience,
                min_delta=0.001,
                check_on_train_epoch_end=False,
                check_finite=True,
                strict=True,
            )
        callbacks.append(callback)

    class WheatSelection(pl.Callback):
        def __init__(self):
            self.best_score = -1.0
            self.best_epoch = None
            self.best_offset = None
            self.predictions = []
            self.image_ids = []
            self.current_report = None

        def state_dict(self):
            return {
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "best_offset": self.best_offset,
            }

        def load_state_dict(self, state_dict):
            self.best_score = state_dict["best_score"]
            self.best_epoch = state_dict["best_epoch"]
            self.best_offset = state_dict["best_offset"]

        def on_validation_epoch_start(self, trainer, pl_module):
            if trainer.world_size != 1:
                raise ValueError("Wheat callback currently requires a single trainer process")
            self.predictions, self.image_ids = [], []
            self.current_report = None

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            # Portable decoding uses CPU float64 because MPS has no float64.
            # RF-DETR's supplemental F1 matcher requires predictions and targets
            # on the same device. Validation loss is explicitly disabled in this
            # trainer; only image samples need to remain on the accelerator.
            if pl_module._should_compute_val_loss:
                raise ValueError("Wheat portable scoring requires validation loss disabled")
            for target in batch[1]:
                for key, value in target.items():
                    if isinstance(value, torch.Tensor):
                        target[key] = value.detach().cpu()

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            for prediction, target in zip(outputs["results"], outputs["targets"], strict=True):
                image_id = int(target["image_id"].item())
                self.image_ids.append(image_id)
                boxes = prediction["boxes"].detach().cpu()
                scores = prediction["scores"].detach().cpu()
                for box, score in zip(boxes.tolist(), scores.tolist(), strict=True):
                    x1, y1, x2, y2 = box
                    self.predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            if len(self.image_ids) != len(set(self.image_ids)):
                raise ValueError("Validation repeated image IDs; refusing a biased ADA score")
            if not config.preflight_batches and set(self.image_ids) != expected_ids:
                raise ValueError("Wheat selection requires the complete native validation split")
            if _sha256(annotation) != annotation_sha:
                raise ValueError("Original wheat annotations changed during training")
            started = time.monotonic()
            report, calibrated = calibrated_wheat_evaluation(
                annotation, self.predictions, image_ids=self.image_ids
            )
            calibration = report["calibration"]
            score = report["metrics"]["detection_acc_avg_dom"]
            offset = calibration["foreground_logit_offset"]
            raw_score = calibration["raw_zero_offset_ADA"]
            values = {
                "val/detection_acc_avg_dom": score,
                "val/ema_detection_acc_avg_dom": score,
                "val/raw_zero_offset_ADA": raw_score,
                "val/foreground_logit_offset": offset,
            }
            for key, value in values.items():
                pl_module.log(key, value, on_step=False, on_epoch=True, logger=True)
                trainer.callback_metrics[key] = torch.tensor(value, dtype=torch.float64)
            selected = score > self.best_score
            report.update(
                epoch_1based=trainer.current_epoch + 1,
                optimizer_steps=trainer.global_step,
                selected=selected,
                calibration_seconds=time.monotonic() - started,
                test_access=False,
            )
            self.current_report = report
            validation_output = output / "validation"
            validation_output.mkdir(exist_ok=True)
            _write_json(validation_output / f"epoch-{trainer.current_epoch + 1:04d}.json", report)
            if selected:
                self.best_score = score
                self.best_epoch = trainer.current_epoch + 1
                self.best_offset = offset
                _write_json(output / "validation-best.json", report)
                _write_json(output / "validation-best.coco.json", calibrated)
                _write_json(output / "validation-best-raw.coco.json", self.predictions)
            if report["epoch_1based"] == pause_after_epoch:
                from safetensors.torch import save_file

                state = BestModelCallback._get_ema_model_state_dict(trainer, pl_module)
                tensors = {
                    key: value.detach().cpu().contiguous().clone() for key, value in state.items()
                }
                snapshots = output / "snapshots"
                snapshots.mkdir(exist_ok=True)
                snapshot = snapshots / f"epoch-{pause_after_epoch:04d}.safetensors"
                if snapshot.exists():
                    raise ValueError(
                        "Fixed-epoch snapshot exists; refusing to overwrite provenance"
                    )
                temporary = snapshot.with_suffix(".safetensors.tmp")
                save_file(tensors, temporary)
                temporary.replace(snapshot)
                snapshot_report = {
                    **report,
                    "checkpoint_selection": f"prospectively fixed epoch {pause_after_epoch}",
                    "model": {
                        "safetensors_sha256": _sha256(snapshot),
                        "resolution": pl_module.model_config.resolution,
                    },
                    "training_plan_sha256": _sha256(output / "training-plan.json"),
                }
                _write_json(snapshot.with_suffix(".json"), snapshot_report)
                _write_json(snapshot.with_suffix(".coco.json"), calibrated)
                _write_json(snapshot.with_suffix(".raw.coco.json"), self.predictions)
                # Telemetry's later on_validation_end hook requests a clean stop;
                # Lightning still writes the full optimizer/EMA checkpoint at
                # the epoch boundary. Resume uses the same frozen 30-epoch plan
                # after the caller removes this marker, never a shortened schedule.
                (output / "STOP_AFTER_EPOCH").write_text(
                    f"Planned pause after epoch {pause_after_epoch}; immutable snapshot ready.\n"
                )
            print(
                json.dumps(
                    {
                        "event": "wheat_validation",
                        **values,
                        "selected": selected,
                        "epoch": trainer.current_epoch + 1,
                        "calibration_seconds": report["calibration_seconds"],
                    }
                ),
                flush=True,
            )
            self.predictions, self.image_ids = [], []

    selection_callback = WheatSelection()
    callbacks.append(selection_callback)
    trainer.callbacks = callbacks


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "safetensors":
        parser = argparse.ArgumentParser(description="Export a numeric-only detector submission")
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        print(json.dumps(export_detector_safetensors(args.checkpoint, args.output), indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        parser = argparse.ArgumentParser(description="Export frozen RF-DETR query features")
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--split-directory", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
        parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
        parser.add_argument("--queries", type=int, default=128)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--max-images", type=int)
        parser.add_argument("--architecture", choices=("nano", "small"), default="nano")
        parser.add_argument("--cpu-threads", type=int, default=4)
        parser.add_argument("--resolution", type=int)
        args = parser.parse_args(sys.argv[2:])
        print(json.dumps(export_queries(**vars(args)), indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("nano", "small"), default="nano")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--mps-memory-gib", type=float, default=16)
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--preflight-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmstart", type=Path)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--warmstart-resolution", type=int)
    parser.add_argument("--selection-metric", choices=("coco_ap", "wheat_ada"), default="coco_ap")
    parser.add_argument("--pause-after-epoch", type=int)
    args = parser.parse_args()
    config = DetectorTrainingConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k
            not in {
                "source",
                "output",
                "resume",
                "warmstart",
                "resolution",
                "warmstart_resolution",
                "selection_metric",
                "pause_after_epoch",
            }
        }
    )
    print(
        json.dumps(
            train_detector(
                args.source,
                args.output,
                config,
                resume=args.resume,
                warmstart=args.warmstart,
                resolution=args.resolution,
                warmstart_resolution=args.warmstart_resolution,
                selection_metric=args.selection_metric,
                pause_after_epoch=args.pause_after_epoch,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
