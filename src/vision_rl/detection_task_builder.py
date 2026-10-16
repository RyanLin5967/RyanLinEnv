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
        return {"error": error, "AP": 0.0}


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
        document = load_coco(test / "_annotations.coco.json")
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
        result = evaluate_coco(
            test / "_annotations.coco.json",
            predictions,
            output=logs / "evaluation.json",
            model=provenance,
        )
        reward = float(result["AP"])
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Native AP must be a finite fraction between zero and one")
        shutil.copyfile(predictions_path, logs / "predictions.json")
        (logs / "inference.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (logs / "reward.txt").write_text(f"{reward}\n")
        return result


def _bundle_coco(source: Path, destination: Path, document: dict) -> None:
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
    annotations = [
        {key: row[key] for key in ("id", "image_id", "category_id", "bbox", "area", "iscrowd")}
        for row in document["annotations"]
    ]
    categories = [{"id": row["id"], "name": row["name"]} for row in document["categories"]]
    bundled = {"images": images, "annotations": annotations, "categories": categories, "info": {}}
    (destination / "_annotations.coco.json").write_text(json.dumps(bundled) + "\n")


def make_pipeline_replay_task(
    profile: Path,
    test: Path,
    output: Path,
    *,
    base_image: str = "vision-rl:detection",
) -> dict:
    """Replay a frozen typed pipeline, including its explicitly optional stages.

    The profile is task-author input copied into the verifier's private tests,
    never a submitted configuration. Use only after the relevant model-selection
    boundary; calling this function reads the specified evaluation split.
    """
    from .detection_recipe import load_profile, profile_artifacts

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", base_image):
        raise ValueError("Invalid base image reference")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Task output already exists; choose a new directory")
    profile = Path(profile).resolve()
    if profile.stat().st_size > PROFILE_LIMIT:
        raise ValueError("Trusted pipeline profile exceeds 64 KiB")
    definition, _, artifacts = load_profile(profile)
    specifications = profile_artifacts(definition)
    test = Path(test).resolve()
    inspected = inspect_split(test)
    document = load_coco(test / "_annotations.coco.json")
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
        _bundle_coco(test, tests / "data/test", document)
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
            "test-based selection. Predictions, provenance, timing, and native COCO AP "
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
