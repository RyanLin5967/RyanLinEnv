"""Build offline Harbor tasks for full RF-DETR training and query-refinement RL.

The grading boundary accepts numeric safetensors, never a submitted pickle,
configuration, feature cache, preprocessing function, or Python module. The
pristine verifier constructs the fixed architectures from its installed package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import tempfile
from pathlib import Path

from .data import image_hash
from .detection_eval import PROTOCOL, load_coco
from .detection_training import inspect_split

MODEL_LIMIT = 200 * 1024**2
REFINER_LIMIT = 8 * 1024**2
PROFILE_LIMIT = 64 * 1024
CALIBRATION_LIMIT = 1024


def write_wheat_calibration(path: Path, offset: float) -> None:
    """Export only a validation-selected member of the fixed confidence grid."""
    import numpy as np
    from safetensors.numpy import save_file

    from .wheat_eval import WHEAT_LOGIT_OFFSETS

    if type(offset) not in (int, float) or offset not in WHEAT_LOGIT_OFFSETS:
        raise ValueError("Calibration must select the fixed seven-point wheat grid")
    save_file({"foreground_logit_offset": np.asarray([offset], dtype=np.float32)}, str(path))


def _read_wheat_calibration(path: Path) -> float:
    import numpy as np
    from safetensors.numpy import load

    from .wheat_eval import WHEAT_LOGIT_OFFSETS

    # Caller has already copied through the bounded no-follow staging boundary.
    values = load(path.read_bytes())
    if set(values) != {"foreground_logit_offset"}:
        raise ValueError("Calibration accepts only its one numeric offset tensor")
    value = values["foreground_logit_offset"]
    if value.dtype != np.float32 or value.shape != (1,):
        raise ValueError("Calibration requires exactly float32[1]")
    offset = float(value[0])
    if offset not in WHEAT_LOGIT_OFFSETS:
        raise ValueError("Calibration is outside the fixed seven-point wheat grid")
    return offset


def verify_wheat_training_submission(
    trusted_template: Path,
    artifacts: Path = Path("/logs/artifacts/training"),
    test: Path = Path("/tests/data/test"),
    logs: Path = Path("/logs/verifier"),
) -> dict:
    """Grade new numeric detector weights with author-owned architecture and code.

    Only the newly computed detector digest and bounded numeric calibration vary.
    No submitted profile, configuration, module, or optimizer pickle is consumed.
    """
    from .detection_recipe import read, validate_profile

    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text("0\n")
    try:
        with tempfile.TemporaryDirectory(prefix="trusted-wheat-training-") as temporary:
            staged = Path(temporary).resolve()
            template = staged / "pipeline.json"
            _stage_artifact(trusted_template, template, PROFILE_LIMIT)
            profile = read(template)
            validate_profile(profile)
            if (
                profile["schema_version"] != 3
                or profile["detector"]["file"] != "detector.safetensors"
            ):
                raise ValueError("Training verifier requires its fixed wheat schema3 template")
            model = staged / "detector.safetensors"
            calibration = staged / "calibration.safetensors"
            _stage_artifact(artifacts / model.name, model, MODEL_LIMIT)
            _stage_artifact(artifacts / calibration.name, calibration, CALIBRATION_LIMIT)
            profile["detector"]["sha256"] = _hash_file(model)
            profile["wheat"]["foreground_logit_offset"] = _read_wheat_calibration(calibration)
            validate_profile(profile)
            template.write_text(json.dumps(profile, sort_keys=True, indent=2) + "\n")
            result = _verify_pipeline_submission(staged, test, logs, template)
            (logs / "submission.json").write_text(
                json.dumps(
                    {
                        "detector_sha256": profile["detector"]["sha256"],
                        "calibration_sha256": _hash_file(calibration),
                        "foreground_logit_offset": profile["wheat"]["foreground_logit_offset"],
                        "trusted_template_sha256": _hash_file(trusted_template),
                        "submitted_code_executed": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            return result
    except Exception as exc:
        error = {"type": type(exc).__name__}
        (logs / "error.json").write_text(json.dumps(error) + "\n")
        (logs / "reward.txt").write_text("0\n")
        return {"error": error, "reward": 0.0}


def _inference_configuration(inference: str, configuration: dict | None) -> dict:
    """Validate verifier-owned settings without accepting paths or executable code."""
    if inference == "native128":
        if configuration is not None:
            raise ValueError("Native query replay does not accept inference overrides")
        return {"pipeline": "native128", "queries": 128}
    if inference != "label_free":
        raise ValueError("Inference must be native128 or label_free")
    if configuration is None:
        configuration = {}
    allowed = {"resolutions", "horizontal_flip", "fusion_iou", "checkpoint_resolution"}
    if not isinstance(configuration, dict) or set(configuration) - allowed:
        raise ValueError("Unexpected trusted inference configuration fields")
    resolutions = configuration.get("resolutions", [384])
    flip = configuration.get("horizontal_flip", False)
    fusion = configuration.get("fusion_iou", 0.55)
    checkpoint_resolution = configuration.get("checkpoint_resolution")
    if checkpoint_resolution is not None and (
        type(checkpoint_resolution) is not int or checkpoint_resolution not in {384, 512}
    ):
        raise ValueError("Trusted Nano checkpoint resolution must be 384 or 512")
    if not isinstance(resolutions, (tuple, list)) or not 1 <= len(resolutions) <= 4:
        raise ValueError("Use one to four bounded inference resolutions")
    if any(
        type(value) is not int or not 128 <= value <= 1024 or value % 32 for value in resolutions
    ):
        raise ValueError("Inference resolutions must be multiples of 32 in 128..1024")
    if len(set(resolutions)) != len(resolutions) or type(flip) is not bool:
        raise ValueError("Inference views must be unique and horizontal_flip must be boolean")
    if type(fusion) not in (int, float) or not math.isfinite(fusion) or not 0 < fusion <= 1:
        raise ValueError("Fusion IoU must be a finite fraction in (0, 1]")
    return {
        "pipeline": "label_free",
        "resolutions": list(resolutions),
        "horizontal_flip": flip,
        "fusion_iou": float(fusion),
        "max_detections": 500,
        "batch_size": 1,
        "cpu_threads": 2,
        "device": "cpu",
        "architecture": "nano",
        "checkpoint_resolution": checkpoint_resolution,
    }


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stage_artifact(source: Path, destination: Path, maximum: int) -> None:
    """Copy a bounded regular file, rejecting symlinks in every path component."""
    source = source.absolute()
    directory = os.open(source.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in source.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as stream:
            attributes = os.fstat(stream.fileno())
            if not stat.S_ISREG(attributes.st_mode) or not 0 < attributes.st_size <= maximum:
                raise ValueError("Missing, nonregular, empty, or oversized submitted artifact")
            with destination.open("xb") as output:
                remaining = maximum + 1
                while remaining:
                    block = stream.read(min(1024**2, remaining))
                    if not block:
                        break
                    output.write(block)
                    remaining -= len(block)
                if remaining == 0:
                    raise ValueError("Submitted artifact grew beyond its size limit")
    finally:
        os.close(directory)


def verify_detection_submission(
    artifacts: Path = Path("/logs/artifacts/training"),
    test: Path = Path("/tests/data/test"),
    logs: Path = Path("/logs/verifier"),
    *,
    inference: str = "native128",
    inference_config: dict | None = None,
    expected_model_sha256: str | None = None,
    expected_refiner_sha256: str | None = None,
    trusted_profile: Path | None = None,
) -> dict:
    """Evaluate declared numeric safetensors with pristine CPU inference code.

    Only trusted generated ``/tests/verify.py`` calls this in Harbor. Arguments
    permit local auditing; they are never obtained from an agent's configuration.
    All failures leave a zero reward and an error type, avoiding path/secret leaks.
    """
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text("0\n")
    try:
        if trusted_profile is not None:
            if (
                inference != "native128"
                or inference_config is not None
                or expected_model_sha256 is not None
                or expected_refiner_sha256 is not None
            ):
                raise ValueError("A frozen pipeline profile cannot be mixed with legacy overrides")
            return _verify_pipeline_submission(artifacts, test, logs, trusted_profile)
        configuration = _inference_configuration(inference, inference_config)
        from .detection_eval import evaluate_coco
        from .detection_rl import (
            coco_predictions,
            load_cache,
            predict_refinement_safetensors,
        )
        from .detection_training import export_queries

        with tempfile.TemporaryDirectory(prefix="trusted-detector-") as temporary:
            staged = Path(temporary).resolve()
            model, refiner = staged / "model.safetensors", staged / "refiner.safetensors"
            _stage_artifact(artifacts / model.name, model, MODEL_LIMIT)
            _stage_artifact(artifacts / refiner.name, refiner, REFINER_LIMIT)
            provenance = {
                "configuration": configuration,
                "model_sha256": _hash_file(model),
                "refiner_sha256": _hash_file(refiner),
                "artifact_format": "safetensors",
                "submitted_code_executed": False,
            }
            for name, expected in (
                ("model_sha256", expected_model_sha256),
                ("refiner_sha256", expected_refiner_sha256),
            ):
                if expected is not None and provenance[name] != expected:
                    raise ValueError("Submitted weights differ from the frozen replay artifact")
            if inference == "native128":
                cache_path = staged / "test-queries.pt"
                export_queries(
                    model, test, cache_path, split="test", device="cpu", queries=128, batch_size=1
                )
                cache = load_cache(cache_path)
                predictions = coco_predictions(
                    cache, predict_refinement_safetensors(cache, refiner, device="cpu")
                )
            else:
                from .detection_inference import InferenceConfig, infer_images, load_image_manifest
                from .detection_training import load_detector_safetensors

                document = load_coco(test / "_annotations.coco.json")
                if len(document["categories"]) != 1:
                    raise ValueError("The fixed verifier supports one foreground category")
                manifest = staged / "images.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "images": [
                                {key: row[key] for key in ("id", "file_name", "width", "height")}
                                for row in document["images"]
                            ]
                        }
                    )
                )
                images = load_image_manifest(manifest, test)
                detector = load_detector_safetensors(
                    model,
                    architecture="nano",
                    num_classes=1,
                    device="cpu",
                    resolution=configuration["checkpoint_resolution"],
                )
                settings = {key: value for key, value in configuration.items() if key != "pipeline"}
                settings["resolutions"] = tuple(settings["resolutions"])
                predictions, timing = infer_images(
                    detector,
                    images,
                    (document["categories"][0]["id"],),
                    InferenceConfig(**settings),
                    refinement=refiner,
                    detector_sha256=provenance["model_sha256"],
                )
                provenance["timing"] = timing
            (logs / "predictions.json").write_text(json.dumps(predictions, allow_nan=False) + "\n")
            (logs / "inference.json").write_text(json.dumps(provenance, indent=2) + "\n")
            result = evaluate_coco(
                test / "_annotations.coco.json",
                predictions,
                output=logs / "evaluation.json",
                model={
                    "architecture": "RF-DETR-Nano",
                    "refinement": "fixed-query-policy-v1",
                    **provenance,
                },
            )
            reward = float(result["AP"])
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError("Native AP must be a finite fraction between zero and one")
            (logs / "reward.txt").write_text(f"{reward}\n")
            return result
    except Exception as exc:
        error = {"type": type(exc).__name__}
        (logs / "error.json").write_text(json.dumps(error) + "\n")
        (logs / "reward.txt").write_text("0\n")
        return {"error": error, "reward" if trusted_profile is not None else "AP": 0.0}


def _pipeline_document(directory: Path, profile: dict) -> dict:
    if "wheat" not in profile:
        return load_coco(directory / "_annotations.coco.json")
    from .wheat_eval import BENCHMARK_ID, validate_wheat_coco

    document = json.loads((directory / "_annotations.coco.json").read_text())
    validate_wheat_coco(document, [])
    if document.get("info", {}).get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Wheat replay requires its explicit benchmark identity")
    return document


def _inspect_wheat_split(directory: Path, document: dict) -> dict:
    """Inspect image integrity without COCO's in-frame/nonempty annotation restrictions."""
    from PIL import Image

    rows = []
    for row in document["images"]:
        path = (directory / row["file_name"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Wheat image path escapes its split or is not a file")
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Wheat image dimensions differ from its metadata")
        rows.append({"id": row["id"], "sha256": _hash_file(path)})
    fingerprint = hashlib.sha256(
        json.dumps(
            {"document": document, "image_hashes": rows},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return {
        "images": len(rows),
        "fingerprint": fingerprint,
        "annotation_sha256": _hash_file(directory / "_annotations.coco.json"),
    }


def _verify_pipeline_submission(
    artifacts: Path, test: Path, logs: Path, profile_path: Path
) -> dict:
    """Stage only the trusted profile's declared weights and reuse its image-only runner."""
    from .detection_eval import evaluate_coco
    from .detection_recipe import load_profile, predict_profile, profile_artifacts, read

    with tempfile.TemporaryDirectory(prefix="trusted-pipeline-") as temporary:
        # The verifier creates this directory itself. Canonicalize platform
        # temp roots such as macOS /var before strict no-follow model loaders;
        # submitted artifact paths still pass the original no-follow checks.
        staged = Path(temporary).resolve()
        trusted = staged / "pipeline.json"
        _stage_artifact(profile_path, trusted, PROFILE_LIMIT)
        profile = read(trusted)
        specifications = profile_artifacts(profile)
        for specification in specifications.values():
            name = specification["file"]
            _stage_artifact(artifacts / name, staged / name, specification["max_bytes"])
        load_profile(trusted)
        # The trusted grader owns annotations. Only image metadata enters the
        # pipeline API, so optional crop or RL stages cannot access test labels.
        document = _pipeline_document(test, profile)
        if sorted(row["id"] for row in document["categories"]) != sorted(profile["category_ids"]):
            raise ValueError("Test categories differ from the frozen training mapping")
        manifest = staged / "images.json"
        manifest.write_text(
            json.dumps(
                {
                    "images": [
                        {key: row[key] for key in ("id", "file_name", "width", "height")}
                        for row in document["images"]
                    ]
                }
            )
            + "\n"
        )
        evidence = staged / "inference"
        predict_profile(trusted, manifest, test, evidence)
        predictions_path = evidence / "predictions.coco.json"
        if predictions_path.stat().st_size > 256 * 1024**2:
            raise ValueError("Pipeline predictions exceed 256 MiB")
        predictions = json.loads(predictions_path.read_text())
        provenance = {
            "pipeline": profile,
            "profile_sha256": _hash_file(trusted),
            "artifact_format": "safetensors",
            "submitted_code_executed": False,
            "timing": json.loads((evidence / "inference.json").read_text()),
            "prediction_provenance": json.loads((evidence / "provenance.json").read_text()),
        }
        if "wheat" in profile:
            from .wheat_eval import evaluate_wheat

            result = evaluate_wheat(
                test / "_annotations.coco.json",
                predictions,
                output=logs / "evaluation.json",
                model=provenance,
            )
            if not result["evaluation_valid"]:
                raise ValueError("Wheat scoring reported an incomplete evaluation")
            reward = float(result["metrics"]["detection_acc_avg_dom"])
        else:
            result = evaluate_coco(
                test / "_annotations.coco.json",
                predictions,
                output=logs / "evaluation.json",
                model=provenance,
            )
            reward = float(result["AP"])
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Native reward must be a finite fraction between zero and one")
        shutil.copyfile(predictions_path, logs / "predictions.json")
        (logs / "inference.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (logs / "reward.txt").write_text(f"{reward}\n")
        return result


def _bundle_coco(source: Path, destination: Path, document: dict, *, wheat: bool = False) -> None:
    """Copy only declared images; strip source paths and incidental metadata."""
    destination.mkdir(parents=True)
    images = []
    for index, original in enumerate(document["images"]):
        path = (source / original["file_name"]).resolve()
        suffix = path.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise ValueError("Unsupported COCO image extension")
        name = f"image-{index:06d}{suffix}"
        shutil.copyfile(path, destination / name)
        images.append({key: original[key] for key in ("id", "width", "height")})
        images[-1]["file_name"] = name
        if wheat:
            images[-1].update(
                {
                    key: original[key]
                    for key in (
                        "domain",
                        "session_id",
                        "original_image_id",
                        "native_split",
                    )
                    if key in original
                }
            )
    annotations = (
        [dict(row) for row in document["annotations"]]
        if wheat
        else [
            {key: row[key] for key in ("id", "image_id", "category_id", "bbox", "area", "iscrowd")}
            for row in document["annotations"]
        ]
    )
    categories = [{"id": row["id"], "name": row["name"]} for row in document["categories"]]
    bundled = {"images": images, "annotations": annotations, "categories": categories, "info": {}}
    if wheat:
        bundled["info"] = {
            key: document["info"][key]
            for key in (
                "benchmark_id",
                "native_split",
            )
            if key in document["info"]
        }
    (destination / "_annotations.coco.json").write_text(json.dumps(bundled) + "\n")


def make_pipeline_replay_task(
    profile: Path,
    test: Path,
    output: Path,
    *,
    base_image: str | None = None,
) -> dict:
    """Replay a frozen typed pipeline, including its explicitly optional stages.

    The profile is task-author input copied into the verifier's private tests,
    never a submitted configuration. Use only after the relevant model-selection
    boundary; calling this function reads the specified evaluation split.
    """
    from .detection_recipe import load_profile, profile_artifacts

    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Task output already exists; choose a new directory")
    profile = Path(profile).resolve()
    if profile.stat().st_size > PROFILE_LIMIT:
        raise ValueError("Trusted pipeline profile exceeds 64 KiB")
    definition, _, artifacts = load_profile(profile)
    specifications = profile_artifacts(definition)
    wheat = "wheat" in definition
    if base_image is None:
        base_image = "vision-rl:wheat" if wheat else "vision-rl:detection"
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", base_image):
        raise ValueError("Invalid base image reference")
    test = Path(test).resolve()
    document = _pipeline_document(test, definition)
    inspected = _inspect_wheat_split(test, document) if wheat else inspect_split(test)
    if sorted(row["id"] for row in document["categories"]) != sorted(definition["category_ids"]):
        raise ValueError("Test categories differ from the frozen training mapping")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pipeline-replay-", dir=output.parent) as temporary:
        staged = Path(temporary) / "task"
        environment, tests, solution = (
            staged / name for name in ("environment", "tests", "solution")
        )
        submission = environment / "submission"
        submission.mkdir(parents=True)
        solution.mkdir(parents=True)
        commands = ["#!/bin/bash", "set -euo pipefail", "mkdir -p /logs/artifacts/training"]
        for role, specification in specifications.items():
            name = specification["file"]
            _stage_artifact(
                artifacts[role].resolve(), submission / name, specification["max_bytes"]
            )
            if _hash_file(submission / name) != specification["sha256"]:
                raise ValueError("Artifact changed during pipeline task packaging")
            commands.append(
                shlex.join(
                    [
                        "cp",
                        "/app/submission/" + name,
                        "/logs/artifacts/training/" + name,
                    ]
                )
            )
        _bundle_coco(test, tests / "data/test", document, wheat=wheat)
        # Canonical trusted profile bytes exclude surrounding source-directory
        # artifacts; only the schema's declared numeric files enter the image.
        (tests / "pipeline.json").write_text(
            json.dumps(definition, sort_keys=True, indent=2) + "\n"
        )
        (environment / "Dockerfile").write_text(
            f'FROM {base_image}\nCOPY submission/ /app/submission/\nCMD ["sleep", "infinity"]\n'
        )
        (solution / "solve.sh").write_text("\n".join(commands) + "\n")
        (tests / "test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
            "echo 0 > /logs/verifier/reward.txt\npython -I /tests/verify.py\n"
        )
        (tests / "verify.py").write_text(
            "from pathlib import Path\n"
            "from vision_rl.detection_task_builder import verify_detection_submission\n"
            'verify_detection_submission(trusted_profile=Path("/tests/pipeline.json"))\n'
        )
        for script in (solution / "solve.sh", tests / "test.sh"):
            script.chmod(0o755)
        (staged / "task.toml").write_text(
            'schema_version = "1.4"\n\n[task]\nname = "vision-rl/frozen-pipeline-replay"\n'
            'version = "0.2.0"\ndescription = "Verify a frozen image-only numeric pipeline"\n\n'
            '[metadata]\ncategory = "machine-learning"\npurpose = "artifact-replay"\n\n'
            "[agent]\ntimeout_sec = 300.0\n\n"
            '[verifier]\ntimeout_sec = 14400.0\nenvironment_mode = "separate"\n\n'
            "[environment]\ncpus = 2\nmemory_mb = 3072\nstorage_mb = 8192\n"
            'network_mode = "no-network"\n'
        )
        (staged / "instruction.md").write_text(
            "# Replay a frozen specialist pipeline\n\n"
            "Copy every `.safetensors` file from `/app/submission/` into "
            "`/logs/artifacts/training/`. The separate verifier owns the frozen typed profile, "
            "checks each artifact and installed implementation hash, and runs its trusted "
            "image-only prediction pipeline. Optional stages run only when the frozen profile "
            "declares them; a detector-only submission needs no refinement head.\n\n"
            "This verifies already selected artifacts. It performs no training or "
            "test-based selection. Predictions, provenance, timing, and the native metric "
            "are recorded under `/logs/verifier`.\n"
        )
        bundle = {
            "schema_version": 2,
            "purpose": "frozen-pipeline-replay",
            "protocol": PROTOCOL,
            "pipeline": definition,
            "profile_sha256": _hash_file(tests / "pipeline.json"),
            "test_images": inspected["images"],
            "test_fingerprint": inspected["fingerprint"],
            "source_annotation_sha256": inspected["annotation_sha256"],
            "test_in_build_context": False,
            "submission_format": "safetensors",
        }
        if wheat:
            from .wheat_eval import BENCHMARK_ID
            from .wheat_eval import PROTOCOL as WHEAT_PROTOCOL

            bundle.update(
                protocol=WHEAT_PROTOCOL,
                benchmark_id=BENCHMARK_ID,
                reward_metric="detection_acc_avg_dom",
            )
        (staged / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
        staged.rename(output)
    return {**bundle, "task": str(output.resolve()), "agent": "oracle"}


def make_detection_replay_task(
    test: Path,
    model: Path,
    refiner: Path,
    output: Path,
    *,
    inference: str = "native128",
    inference_config: dict | None = None,
    base_image: str = "vision-rl:detection",
) -> dict:
    """Package frozen numeric weights for an independent Harbor artifact replay.

    This is a post-selection verification task. Call it on the final test only
    after model/configuration freeze and the primary evaluation; it neither
    trains nor selects weights. ``oracle`` copies the supplied numeric files and
    the separate pristine verifier checks their hashes before evaluating them.
    No model-originated metadata controls architecture or preprocessing.
    """
    configuration = _inference_configuration(inference, inference_config)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", base_image):
        raise ValueError("Invalid base image reference")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Task output already exists; choose a new directory")
    test = Path(test).resolve()
    inspected = inspect_split(test)
    document = load_coco(test / "_annotations.coco.json")
    if len(inspected["categories"]) != 1:
        raise ValueError("The fixed detector replay requires one foreground category")
    if Path(model).suffix != ".safetensors" or Path(refiner).suffix != ".safetensors":
        raise ValueError("Replay accepts only safetensors artifact paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".detection-replay-", dir=output.parent) as temporary:
        staged = Path(temporary) / "task"
        environment = staged / "environment"
        submission = environment / "submission"
        tests, solution = staged / "tests", staged / "solution"
        submission.mkdir(parents=True)
        solution.mkdir(parents=True)
        # Source paths belong to the trusted task author. The runtime verifier
        # additionally rejects symlinks in all submitted path components.
        _stage_artifact(Path(model).resolve(), submission / "model.safetensors", MODEL_LIMIT)
        _stage_artifact(Path(refiner).resolve(), submission / "refiner.safetensors", REFINER_LIMIT)
        model_hash = _hash_file(submission / "model.safetensors")
        refiner_hash = _hash_file(submission / "refiner.safetensors")
        _bundle_coco(test, tests / "data/test", document)
        (environment / "Dockerfile").write_text(
            f'FROM {base_image}\nCOPY submission/ /app/submission/\nCMD ["sleep", "infinity"]\n'
        )
        (solution / "solve.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/artifacts/training\n"
            "cp /app/submission/model.safetensors /logs/artifacts/training/model.safetensors\n"
            "cp /app/submission/refiner.safetensors /logs/artifacts/training/refiner.safetensors\n"
        )
        (tests / "test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
            "echo 0 > /logs/verifier/reward.txt\npython -I /tests/verify.py\n"
        )
        (tests / "verify.py").write_text(
            "from vision_rl.detection_task_builder import verify_detection_submission\n"
            "verify_detection_submission(\n"
            f"    inference={inference!r}, inference_config={inference_config!r},\n"
            f"    expected_model_sha256={model_hash!r},\n"
            f"    expected_refiner_sha256={refiner_hash!r},\n)\n"
        )
        for path in (solution / "solve.sh", tests / "test.sh"):
            path.chmod(0o755)
        (staged / "task.toml").write_text(
            'schema_version = "1.4"\n\n[task]\nname = "vision-rl/frozen-detector-replay"\n'
            'version = "0.1.0"\ndescription = "Verify frozen detector and refiner artifacts"\n\n'
            '[metadata]\ncategory = "machine-learning"\npurpose = "artifact-replay"\n\n'
            "[agent]\ntimeout_sec = 300.0\n\n"
            '[verifier]\ntimeout_sec = 14400.0\nenvironment_mode = "separate"\n\n'
            "[environment]\ncpus = 2\nmemory_mb = 3072\nstorage_mb = 8192\n"
            'network_mode = "no-network"\n'
        )
        (staged / "instruction.md").write_text(
            "# Replay frozen detector artifacts\n\n"
            "Copy `/app/submission/model.safetensors` and `refiner.safetensors` into "
            "`/logs/artifacts/training/`. The separate verifier requires their exact frozen "
            "hashes and runs its own installed inference code. This task verifies a previously "
            "selected artifact; it performs no training or test-based model selection.\n\n"
            "Only numeric weights are submitted. The verifier owns the inference settings, "
            "image manifest, annotations, and scoring code. Its predictions, hashes, settings, "
            "and native COCO evaluation are recorded under `/logs/verifier`. Compare replay "
            "predictions using an explicit numeric tolerance across hardware.\n"
        )
        bundle = {
            "schema_version": 1,
            "purpose": "frozen-artifact-replay",
            "architecture": "RF-DETR-Nano",
            "protocol": PROTOCOL,
            "configuration": configuration,
            "model_sha256": model_hash,
            "refiner_sha256": refiner_hash,
            "test_images": inspected["images"],
            "test_fingerprint": inspected["fingerprint"],
            "source_annotation_sha256": inspected["annotation_sha256"],
            "test_in_build_context": False,
            "submission_format": "safetensors",
        }
        (staged / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
        staged.rename(output)
    return {**bundle, "task": str(output.resolve()), "agent": "oracle"}


def _training_script(configuration: dict) -> str:
    return '''"""Supervised detector adaptation followed by measured query-policy RL."""
import json
import shutil
from pathlib import Path
from vision_rl.detection_training import (
    DetectorTrainingConfig, export_detector_safetensors, export_queries, train_detector,
)
from vision_rl.detection_rl import (
    RefinementConfig, export_refinement_safetensors, train_refinement,
)

configuration = CONFIGURATION
output = Path("/logs/artifacts/training")
output.mkdir(parents=True, exist_ok=True)
work = Path("/app/training-work")
work.mkdir(parents=True, exist_ok=True)
device = configuration["device"]
training = DetectorTrainingConfig(
    architecture="nano", device=device, epochs=configuration["detector_epochs"],
    batch_size=configuration["batch_size"],
    grad_accum_steps=configuration["grad_accum_steps"],
    eval_batch_size=1, workers=0, cpu_threads=2,
    preflight_batches=configuration["preflight_batches"],
)
train_detector(Path("/app/data"), work / "detector", training)
export_detector_safetensors(
    work / "detector/checkpoint_best_total.pth", output / "model.safetensors",
)
for directory, split in (("train", "train"), ("valid", "validation")):
    export_queries(
        output / "model.safetensors", Path("/app/data") / directory,
        work / (split + "-queries.pt"), split=split, device=device, queries=128, batch_size=1,
    )
summary = train_refinement(
    work / "train-queries.pt", work / "validation-queries.pt", work / "refinement",
    RefinementConfig(
        device=device, warmup_epochs=configuration["warmup_epochs"],
        rl_epochs=configuration["rl_epochs"], batch_size=configuration["batch_size"],
    ),
)
# Selection uses validation AP only and may retain the pre-RL model. Save the
# best RL checkpoint separately so users can measure RL rather than assume gains.
export_refinement_safetensors(work / "refinement/best.pt", output / "refiner.safetensors")
export_refinement_safetensors(
    work / "refinement/best-rl.pt", output / "refiner-rl-ablation.safetensors",
)
export_refinement_safetensors(
    work / "refinement/warmup.pt", output / "refiner-warmup-ablation.safetensors",
)
# Native optimizer checkpoints stay in the training container. Archive small
# experiment reports plus safe model/policy weights for reproducibility.
for phase in ("detector", "refinement"):
    destination = output / phase
    destination.mkdir()
    for report in (work / phase).iterdir():
        if report.is_file() and report.suffix in {".json", ".jsonl"}:
            shutil.copyfile(report, destination / report.name)
print(json.dumps(summary, indent=2))
'''.replace("CONFIGURATION", repr(configuration))


def make_detection_task(
    source: Path,
    output: Path,
    *,
    detector_epochs: int = 40,
    rl_epochs: int = 30,
    warmup_epochs: int = 8,
    device: str = "cpu",
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    preflight_batches: int = 0,
    queries: int = 128,
    base_image: str = "vision-rl:detection",
    timeout_sec: int = 86460,
) -> dict:
    """Package a one-class COCO dataset with strict training/test separation.

    ``source`` contains ``train``, ``valid``, and ``test`` directories, each with
    ``_annotations.coco.json``. The optional CUDA task requires a CUDA image and
    a Linux NVIDIA host; Docker on macOS cannot expose MPS to Linux containers.
    ``preflight_batches`` limits optimizer smoke tests and is recorded explicitly.
    """
    for name, value in (
        ("detector_epochs", detector_epochs),
        ("rl_epochs", rl_epochs),
        ("batch_size", batch_size),
        ("grad_accum_steps", grad_accum_steps),
        ("timeout_sec", timeout_sec),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("warmup_epochs", warmup_epochs), ("preflight_batches", preflight_batches)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if preflight_batches and preflight_batches < grad_accum_steps:
        raise ValueError("preflight_batches must include at least one optimizer step")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Docker training device must be cpu or cuda")
    if queries != 128 or isinstance(queries, bool) or not isinstance(queries, int):
        raise ValueError("The fixed verifier requires exactly 128 queries")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", base_image):
        raise ValueError("Invalid base image reference")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Task output already exists; choose a new directory")
    source = Path(source).resolve()
    documents, inspections, seen_pixels = {}, {}, set()
    categories = None
    for split in ("train", "valid", "test"):
        directory = source / split
        inspections[split] = inspect_split(directory)
        documents[split] = load_coco(directory / "_annotations.coco.json")
        active_categories = inspections[split]["categories"]
        if len(active_categories) != 1:
            raise ValueError("The fixed detector task requires exactly one foreground category")
        if categories is not None and categories != active_categories:
            raise ValueError("COCO categories must match across train, valid, and test")
        categories = active_categories
        pixels = {
            image_hash(directory / image["file_name"]) for image in documents[split]["images"]
        }
        if seen_pixels & pixels:
            raise ValueError("Image leakage: identical decoded pixels occur across splits")
        seen_pixels.update(pixels)
    configuration = {
        "detector_epochs": detector_epochs,
        "rl_epochs": rl_epochs,
        "warmup_epochs": warmup_epochs,
        "device": device,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "preflight_batches": preflight_batches,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".detection-task-", dir=output.parent) as temporary:
        staged = Path(temporary) / "task"
        environment, tests, solution = (
            staged / name for name in ("environment", "tests", "solution")
        )
        solution.mkdir(parents=True)
        for split in ("train", "valid"):
            _bundle_coco(source / split, environment / "data" / split, documents[split])
        _bundle_coco(source / "test", tests / "data/test", documents["test"])
        (environment / "Dockerfile").write_text(
            f"FROM {base_image}\n"
            "COPY data/ /app/data/\nCOPY train.py /app/train.py\nCOPY train.sh /app/train.sh\n"
            "RUN chmod +x /app/train.sh\n"
            'CMD ["sleep", "infinity"]\n'
        )
        (environment / "train.py").write_text(_training_script(configuration))
        (environment / "train.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\npython -I /app/train.py\n"
        )
        (solution / "solve.sh").write_text("#!/bin/bash\nset -euo pipefail\nbash /app/train.sh\n")
        (tests / "test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
            "echo 0 > /logs/verifier/reward.txt\npython -I /tests/verify.py\n"
        )
        (tests / "verify.py").write_text(
            "from vision_rl.detection_task_builder import verify_detection_submission\n"
            "verify_detection_submission()\n"
        )
        for path in (environment / "train.sh", solution / "solve.sh", tests / "test.sh"):
            path.chmod(0o755)
        gpu = "gpus = 1\n" if device == "cuda" else ""
        (staged / "task.toml").write_text(
            'schema_version = "1.4"\n\n[task]\nname = "vision-rl/rfdetr-query-refinement"\n'
            'version = "0.1.0"\ndescription = "Train RF-DETR and a bounded RL query policy"\n\n'
            '[metadata]\ncategory = "machine-learning"\npurpose = "specialist-training"\n\n'
            f"[agent]\ntimeout_sec = {float(timeout_sec)}\n\n"
            '[verifier]\ntimeout_sec = 14400.0\nenvironment_mode = "separate"\n\n'
            f"[environment]\ncpus = 2\nmemory_mb = {16384 if device == 'cuda' else 3072}\n"
            f"storage_mb = 12288\n{gpu}"
            'network_mode = "no-network"\n'
        )
        (staged / "instruction.md").write_text(
            "# Train an object detection specialist\n\n"
            "Fine-tune RF-DETR Nano on `/app/data/train`; select checkpoints only with "
            "`/app/data/valid`. Each directory has COCO images and `_annotations.coco.json`. "
            "Run `bash /app/train.sh` for full detector supervised learning, decoder-query "
            "caching, optional supervised refinement warmup, and group-relative REINFORCE. "
            "The full transformer is trained; the RL stage trains a bounded query policy.\n\n"
            "Submit `/logs/artifacts/training/model.safetensors` (at most 200 MiB) and "
            "`/logs/artifacts/training/refiner.safetensors` (at most 8 MiB). Use the installed "
            "`export_detector_safetensors` and `export_refinement_safetensors` functions. "
            "The fixed architecture is one-class RF-DETR Nano with 128 retained queries, "
            "256-dimensional embeddings, and a 128-unit refiner. The policy may adjust "
            "box center/size by bounded geometry actions and learn query confidence. "
            "Submitted code, model configuration, pickles, and feature caches are never graded.\n\n"
            "The verifier runs in a fresh offline container and sees test images only there. "
            f"Reward is native {PROTOCOL} AP in [0, 1]. Preserve supervised, warmup, "
            "and RL checkpoint metrics for ablations. Validation may retain an earlier "
            "checkpoint if RL reduces AP. A score does not establish a frontier win "
            "without comparable frontier runs on the identical held-out protocol.\n\n"
            "Safe model/policy weights and JSON experiment reports are archived under "
            "`/logs/artifacts/training`. Native optimizer checkpoints and query caches "
            "remain in `/app/training-work`; export them explicitly if needed for resume.\n\n"
            f"Training device: `{device}`. CPU containers support portable integration tests; "
            "full training needs adequate compute. CUDA requires a compatible image and "
            "Linux NVIDIA runtime. macOS MPS training runs outside Docker.\n"
        )
        bundle = {
            "schema_version": 1,
            "architecture": "RF-DETR-Nano",
            "categories": categories,
            "protocol": PROTOCOL,
            "configuration": configuration,
            "queries": 128,
            "counts": {key: value["images"] for key, value in inspections.items()},
            "fingerprints": {key: value["fingerprint"] for key, value in inspections.items()},
            "test_in_build_context": False,
            "submission_format": "safetensors",
            "smoke_test_only": bool(preflight_batches),
        }
        (staged / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
        staged.rename(output)
    return {
        **bundle,
        "task": str(output.resolve()),
        "agent": "vision_rl.harbor_agent:DetectionTrainingAgent",
    }


def _wheat_training_script(configuration: dict, integration_only: bool, resume_check: bool) -> str:
    return (
        '''"""Native detector RL and matched SFT; only development images are available."""
import json
import shutil
from dataclasses import replace
from pathlib import Path
import torch
from safetensors.torch import load_file
from vision_rl import detector_policy_training as dpt
from vision_rl.detection_task_builder import write_wheat_calibration

config = dpt.PolicyTrainingConfig(**CONFIGURATION)
integration_only = INTEGRATION_ONLY
output = Path("/logs/artifacts/training")
output.mkdir(parents=True, exist_ok=True)
work = Path("/app/training-work")
work.mkdir(parents=True, exist_ok=True)
checkpoint = Path("/app/initialization.safetensors")
common = dict(source=Path("/app/data"), checkpoint=checkpoint,
              protocol=Path("/app/protocol.json"), integration_only=integration_only)
if not integration_only:
    pilot = dpt.pilot(common["source"], checkpoint, work / "pilot", config,
                      protocol=common["protocol"])
    if not pilot["go"]:
        dpt.write(output / "no-go.json", pilot)
        raise RuntimeError("Train-only exploration gate failed; no RL fit is permitted")
    config = replace(config, pixel_std=pilot["selected_pixel_std"])
    common["pilot_path"] = work / "pilot"
common["config"] = config
actor_diagnostics = []
if integration_only:
    objective = dpt.policy_objective
    def measured_objective(mean, reference, targets, image_size, cfg, generator, **kwargs):
        loss, statistics = objective(mean, reference, targets, image_size, cfg, generator, **kwargs)
        coefficient = kwargs.get("kl_coefficient", cfg.kl_coefficient)
        scale = dpt.policy_scales(reference, image_size, cfg)
        kl = cfg.active_query_probability * dpt.gaussian_kl(mean, reference, scale).mean()
        actor = loss - coefficient * kl
        gradient = torch.autograd.grad(actor, mean, retain_graph=True)[0]
        actor_diagnostics.append({
            "actor_output_gradient_absolute_sum": float(gradient.abs().sum()),
            "actor_output_gradient_finite": bool(gradient.isfinite().all()),
            "reward_std": statistics["reward_std"],
            "useful_factor_fraction": statistics["useful_factor_fraction"],
            "scope": "Reward-only gradient at native actor outputs; excludes reference KL",
        })
        return loss, statistics
    dpt.policy_objective = measured_objective
full = dpt.train(output=work / "rl", **common)
primary_actor_diagnostics = list(actor_diagnostics)
state = torch.load(work / "rl/last.pt", map_location="cpu", weights_only=True)
initial = load_file(str(checkpoint))
changed = {name: float((value-initial[name]).abs().max())
           for name, value in state["model"].items() if not torch.equal(value, initial[name])}
del initial
checks = {}
if RESUME_CHECK:
    original = dpt.save_state
    def interrupt(path, saved):
        original(path, saved)
        if saved["optimizer_steps"] == 2:
            raise InterruptedError("Intentional integration-only durable interruption")
    dpt.save_state = interrupt
    try:
        dpt.train(output=work / "resume", **common)
    except InterruptedError:
        pass
    finally:
        dpt.save_state = original
    dpt.train(output=work / "resume", resume=True, **common)
    resumed = torch.load(work / "resume/last.pt", map_location="cpu", weights_only=True)
    def exact(left, right):
        if isinstance(left, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(exact(x, y) for x, y in zip(left, right))
        return left == right
    checks = {key: exact(state[key], resumed[key]) for key in (
        "model", "buffers", "optimizer", "scheduler", "rng", "history", "selections",
        "order", "cursor", "draw_chain_history", "draw_chain_sha256",
        "matched_prefix_draw_sha256", "kl_coefficient", "epoch", "epoch_metrics",
    )}
    if not all(checks.values()):
        raise RuntimeError("Exact CPU interrupted-resume comparison failed")
    del resumed
    # Only our disposable duplicate verification run is removed to bound disk use.
    shutil.rmtree(work / "resume")
del state
shutil.copyfile(work / "rl/best-trained.safetensors", output / "detector.safetensors")
write_wheat_calibration(output / "calibration.safetensors",
    full["selections"]["best-trained"]["foreground_logit_offset"])
reports = output / "rl"
reports.mkdir()
for report in (work / "rl").glob("*.json"):
    shutil.copyfile(report, reports / report.name)
# Preserve successful phases even if a later matched control fails closed.
dpt.write(output / "training-progress.json", {
    "status": "RL_and_resume_completed_controls_pending",
    "integration_only": integration_only,
    "RL_updates": full["optimizer_steps"],
    "RL_active_seconds": full["optimization_seconds"],
    "resume_exact": checks,
    "changed_native_tensors": len(changed),
    "maximum_parameter_change": max(changed.values(), default=0.0),
    "reward_only_output_gradients": primary_actor_diagnostics,
    "primary_requirement_satisfied": False,
})
summaries = {"rl": full}
for mode in ("sft_updates", "sft_walltime"):
    try:
        summaries[mode] = dpt.train(output=work / mode, mode=mode,
                                    control_run=work / "rl", **common)
    finally:
        destination = output / mode
        destination.mkdir()
        for report in (work / mode).glob("*.json"):
            shutil.copyfile(report, destination / report.name)
        if (work / mode / "best-trained.safetensors").is_file():
            shutil.copyfile(work / mode / "best-trained.safetensors",
                            output / (mode + ".safetensors"))
    shutil.rmtree(work / mode)
evidence = {
    "integration_only": integration_only,
    "scope": ("integration check only; not a benchmark result"
              if integration_only else "development training"),
    "native_test_accessed": False,
    "resume_exact": checks,
    "changed_native_tensors": len(changed),
    "maximum_parameter_change": max(changed.values(), default=0.0),
    "RL_updates": full["optimizer_steps"],
    "SFT_equal_updates": summaries["sft_updates"]["optimizer_steps"],
    "RL_active_seconds": full["optimization_seconds"],
    "SFT_equal_walltime_seconds": summaries["sft_walltime"]["optimization_seconds"],
    "SFT_equal_walltime_updates": summaries["sft_walltime"]["optimizer_steps"],
    "parameter_changes_alone_do_not_establish_reward_learning": True,
    "primary_requirement_satisfied": False,
    "reward_diagnostics": json.loads((work / "rl/history.json").read_text()),
    "reward_only_output_gradients": primary_actor_diagnostics,
    "selected_model_sha256": dpt.sha(output / "detector.safetensors"),
    "calibration_sha256": dpt.sha(output / "calibration.safetensors"),
}
dpt.write(output / "training-evidence.json", evidence)
print(json.dumps(evidence, indent=2))
'''.replace("CONFIGURATION", repr(configuration))
        .replace("INTEGRATION_ONLY", repr(integration_only))
        .replace("RESUME_CHECK", repr(resume_check))
    )


def make_wheat_training_task(
    source: Path,
    initialization: Path,
    profile: Path,
    protocol: Path,
    output: Path,
    *,
    configuration: dict | None = None,
    integration_only: bool = False,
    resume_check: bool = False,
    base_image: str = "vision-rl:wheat",
    timeout_sec: int = 86460,
) -> dict:
    """Build native detector RL and both matched SFT arms with official ADA grading.

    The task author supplies a schema3 inference template and immutable legitimate
    SFT initialization. A submitted detector may differ from that initialization;
    only its safe numeric tensors and one seven-grid calibration value are read.
    ``source/test`` is verifier-only. Integration fixtures must be explicitly
    marked and cannot represent the full exploration gate or benchmark result.
    """
    from dataclasses import asdict

    from .detection_recipe import load_profile
    from .detector_policy_training import (
        PolicyTrainingConfig,
        _validate_protocol,
        validate_study_budget,
    )
    from .wheat_eval import PROTOCOL as WHEAT_PROTOCOL

    if type(integration_only) is not bool or type(resume_check) is not bool:
        raise ValueError("Integration and resume flags require explicit booleans")
    if type(timeout_sec) is not int or not 1 <= timeout_sec <= 86460:
        raise ValueError("Task timeout must be an integer in 1..86460")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", base_image):
        raise ValueError("Invalid base image reference")
    config = PolicyTrainingConfig(**(configuration or {}))
    config.validate()
    if config.reward_backend != "wheat_ada" or config.device not in {"cpu", "cuda"}:
        raise ValueError("Wheat task requires wheat_ada and CPU or CUDA training")
    source, initialization, profile, protocol = (
        Path(path).resolve() for path in (source, initialization, profile, protocol)
    )
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Task output already exists; choose a new directory")
    definition, inference, _ = load_profile(profile)
    if definition["schema_version"] != 3 or inference.checkpoint_resolution != config.resolution:
        raise ValueError("Training and trusted wheat inference resolution must agree")
    if definition["detector"]["file"] != "detector.safetensors":
        raise ValueError("Wheat training uses the fixed detector.safetensors filename")
    if (
        initialization.suffix != ".safetensors"
        or _hash_file(initialization) != definition["detector"]["sha256"]
    ):
        raise ValueError("Initialization differs from the task author's frozen numeric model")
    if _hash_file(protocol) != definition["wheat"]["protocol_sha256"]:
        raise ValueError("Training protocol differs from the trusted wheat profile")
    protocol_document = json.loads(protocol.read_text())
    _validate_protocol(config, protocol_document, integration_only=integration_only)
    documents, inspections, seen_pixels = {}, {}, set()
    for split in ("train", "valid", "test"):
        directory = source / split
        documents[split] = _pipeline_document(directory, definition)
        inspections[split] = _inspect_wheat_split(directory, documents[split])
        pixels = {image_hash(directory / row["file_name"]) for row in documents[split]["images"]}
        if pixels & seen_pixels:
            raise ValueError("Image leakage: identical decoded pixels occur across splits")
        seen_pixels.update(pixels)
    validate_study_budget(
        config,
        protocol_document,
        inspections["train"]["images"],
        integration_only=integration_only,
    )
    if not integration_only:
        for split, native_split, count_key in (
            ("train", "train", "train_images"),
            ("valid", "validation", "validation_images"),
            ("test", "test", "test_images"),
        ):
            if (
                documents[split]["info"].get("native_split") != native_split
                or inspections[split]["images"] != protocol_document["data"][count_key]
                or any(
                    row.get("native_split") != native_split for row in documents[split]["images"]
                )
            ):
                raise ValueError("Benchmark training tasks require complete native protocol splits")
    if integration_only and (
        any(inspections[split]["images"] > 4 for split in ("train", "valid"))
        or config.max_steps is None
        or config.max_steps > 4
    ):
        raise ValueError("Integration checks require <=4 train/valid images and <=4 RL updates")
    if resume_check and not (
        integration_only
        and config.device == "cpu"
        and inspections["train"]["images"] == 2
        and config.batch_images == 1
        and config.max_steps == 4
    ):
        raise ValueError(
            "Exact resume smoke requires CPU, two train images, batch1 and four updates"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wheat-training-", dir=output.parent) as temporary:
        staged = Path(temporary) / "task"
        environment, tests, solution = (
            staged / key for key in ("environment", "tests", "solution")
        )
        solution.mkdir(parents=True)
        for split in ("train", "valid"):
            _bundle_coco(source / split, environment / "data" / split, documents[split], wheat=True)
        _bundle_coco(source / "test", tests / "data/test", documents["test"], wheat=True)
        _stage_artifact(initialization, environment / "initialization.safetensors", MODEL_LIMIT)
        shutil.copyfile(protocol, environment / "protocol.json")
        (tests / "template.json").write_text(json.dumps(definition, indent=2) + "\n")
        (environment / "train.py").write_text(
            _wheat_training_script(asdict(config), integration_only, resume_check)
        )
        (environment / "train.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\npython -I /app/train.py\n"
        )
        (environment / "Dockerfile").write_text(
            f"FROM {base_image}\nCOPY data/ /app/data/\n"
            "COPY initialization.safetensors protocol.json train.py train.sh /app/\n"
            'RUN chmod +x /app/train.sh\nCMD ["sleep", "infinity"]\n'
        )
        (solution / "solve.sh").write_text("#!/bin/bash\nset -euo pipefail\nbash /app/train.sh\n")
        (tests / "test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
            "echo 0 > /logs/verifier/reward.txt\npython -I /tests/verify.py\n"
        )
        (tests / "verify.py").write_text(
            "from pathlib import Path\n"
            "from vision_rl.detection_task_builder import verify_wheat_training_submission\n"
            'verify_wheat_training_submission(Path("/tests/template.json"))\n'
        )
        for path in (environment / "train.sh", solution / "solve.sh", tests / "test.sh"):
            path.chmod(0o755)
        gpu = "gpus = 1\n" if config.device == "cuda" else ""
        (staged / "task.toml").write_text(
            'schema_version = "1.4"\n\n[task]\nname = "vision-rl/wheat-native-policy"\n'
            'version = "0.1.0"\ndescription = "Native detector RL with official wheat ADA"\n\n'
            '[metadata]\ncategory = "machine-learning"\npurpose = "specialist-training"\n\n'
            f"[agent]\ntimeout_sec = {float(timeout_sec)}\n\n"
            '[verifier]\ntimeout_sec = 14400.0\nenvironment_mode = "separate"\n\n'
            f"[environment]\ncpus = 2\nmemory_mb = {16384 if config.device == 'cuda' else 3072}\n"
            f'storage_mb = 12288\n{gpu}network_mode = "no-network"\n'
        )
        (staged / "instruction.md").write_text(
            "# Train the native wheat detector with reward\n\n"
            "Run `bash /app/train.sh` to optimize RF-DETR's native decoder and prediction "
            "heads using official per-image wheat detection accuracy. The controller records "
            "both matched continued-SFT controls. Only `/app/data/train` and `/app/data/valid` "
            "are available; checkpoint and confidence selection use validation.\n\n"
            "Submit `detector.safetensors` (200 MiB maximum) and `calibration.safetensors` "
            "(1 KiB maximum) under `/logs/artifacts/training`. The latter contains exactly "
            "float32[1] `foreground_logit_offset`, selecting -1.5, -1, -.5, 0, .5, 1, or 1.5. "
            "Use the installed `write_wheat_calibration` exporter. No submitted code or "
            "configuration is executed. The fresh offline verifier checks the fixed Nano "
            "architecture and independently runs all 300 queries, calibrating once. "
            "Reward is official equal-domain `detection_acc_avg_dom`, not COCO AP.\n\n"
            f"Integration-only: `{integration_only}`. Tiny integration data bypass the "
            "exploration gate explicitly and cannot establish benchmark gains. Parameter "
            "changes, reward gradients, validation results, and controlled comparisons "
            "must be interpreted separately. CUDA needs a Linux NVIDIA runtime; macOS "
            "Docker runs CPU only.\n"
        )
        bundle = {
            "schema_version": 1,
            "benchmark_id": definition["wheat"]["benchmark_id"],
            "protocol": WHEAT_PROTOCOL,
            "reward_metric": "detection_acc_avg_dom",
            "configuration": asdict(config),
            "integration_only": integration_only,
            "resume_check": resume_check,
            "initialization_sha256": _hash_file(initialization),
            "protocol_sha256": _hash_file(protocol),
            "counts": {key: row["images"] for key, row in inspections.items()},
            "fingerprints": {key: row["fingerprint"] for key, row in inspections.items()},
            "test_in_build_context": False,
            "submission_format": "safetensors",
        }
        (staged / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
        staged.rename(output)
    return {
        **bundle,
        "task": str(output.resolve()),
        "agent": "vision_rl.harbor_agent:DetectionTrainingAgent",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-epochs", type=int, default=40)
    parser.add_argument("--rl-epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--preflight-batches", type=int, default=0)
    parser.add_argument("--base-image", default="vision-rl:detection")
    parser.add_argument("--timeout-sec", type=int, default=86460)
    print(json.dumps(make_detection_task(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
