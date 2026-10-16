"""One durable development recipe: detector fitting, progressive training, and RL.

This module compiles a recipe into experiment_loop's existing setup/candidate
state machine. It does not own another scheduler. Only train/valid are read;
selection ends at awaiting_final_evaluation and never consumes a final test.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

from vision_rl import experiment_loop as loop

PACKAGE = Path(__file__).resolve().parent
PROFILE_KIND = "rfdetr_detection_pipeline"
ARTIFACT_LIMITS = {
    "detector": 200 * 1024**2,
    "query_refiner": 8 * 1024**2,
    "crop_refiner": 32 * 1024**2,
}
RUNTIME_PACKAGES = (
    "rfdetr",
    "torch",
    "torchvision",
    "safetensors",
    "numpy",
    "pillow",
    "pycocotools",
    "transformers",
)


def read(path):
    return loop._read_json(Path(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    loop._atomic_json(path, value)


def source_paths():
    return [
        PACKAGE / name
        for name in (
            "detection_recipe.py",
            "experiment_loop.py",
            "detection_training.py",
            "detection_rl.py",
            "detection_rewards.py",
            "detection_inference.py",
            "detection_eval.py",
            "crop_refinement.py",
        )
    ]


def source_hashes(*, wheat=False):
    paths = source_paths() + ([PACKAGE / "wheat_eval.py"] if wheat else [])
    return {path.name: loop._hash_path(path) for path in paths}


def check_sources(expected, *, wheat=False):
    if expected != source_hashes(wheat=wheat):
        raise ValueError("Recipe implementation changed; preserve this run and prepare a new plan")


def runtime_versions(*, public=True):
    """Pin release versions while retaining platform build suffixes in provenance."""
    from packaging.version import Version

    versions = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    return {name: Version(value).public for name, value in versions.items()} if public else versions


def upstream_descriptor(profile):
    """Semantic proposal generator identity, independent of images and crop weights."""
    descriptor = {
        "schema_version": 1,
        "detector_sha256": profile["detector"]["sha256"],
        "query_refiner_sha256": (
            profile["query_refiner"]["sha256"] if profile["query_refiner"] else None
        ),
        "inference": profile["inference"],
        "category_ids": profile["category_ids"],
    }
    if profile.get("schema_version") == 3:
        descriptor["wheat"] = profile["wheat"]
    return descriptor


def proposal_pipeline_sha256(profile):
    return loop._digest(upstream_descriptor(profile))


def _config(cls, values, name):
    if not isinstance(values, dict) or set(values) - {field.name for field in fields(cls)}:
        raise ValueError(f"Unknown {name} configuration fields")
    result = cls(**values)
    result.validate()
    return asdict(result)


def normalize_recipe(raw):
    from vision_rl.detection_inference import InferenceConfig
    from vision_rl.detection_rl import RefinementConfig
    from vision_rl.detection_training import DetectorTrainingConfig

    allowed = {"version", "detector", "progressive", "export", "candidates", "budget", "minimum_ap"}
    if not isinstance(raw, dict) or raw.get("version") != 1 or set(raw) - allowed:
        raise ValueError("Recipe version must be 1 with known fields")
    base = _config(DetectorTrainingConfig, {"device": "cpu", **raw.get("detector", {})}, "detector")
    if base["architecture"] != "nano":
        raise ValueError("This initial progressive recipe supports the trusted Nano architecture")
    progressive = raw.get("progressive")
    if progressive is not None:
        progressive = _config(
            DetectorTrainingConfig,
            {
                **base,
                "epochs": 12,
                "lr": 1e-5,
                "lr_encoder": 1e-5,
                **progressive,
            },
            "progressive detector",
        )
        if progressive["architecture"] != base["architecture"]:
            raise ValueError("Progressive training must retain the detector architecture")
    export = {"device": "cpu", "batch_size": 4, "cpu_threads": 2, **raw.get("export", {})}
    if set(export) != {"device", "batch_size", "cpu_threads"} or export["device"] != "cpu":
        raise ValueError("Export accepts only explicit CPU batch_size/cpu_threads")
    if any(
        type(export[key]) is not int or not 1 <= export[key] <= 64
        for key in ("batch_size", "cpu_threads")
    ):
        raise ValueError("Export batch_size/cpu_threads must be integers in 1..64")
    candidates, ids = [], set()
    for row in raw.get("candidates", []):
        if not isinstance(row, dict) or set(row) - {
            "id",
            "detector_stage",
            "query_rl",
            "inference",
        }:
            raise ValueError(
                "Unknown candidate fields; crop adapters are not enabled in this recipe"
            )
        name = row.get("id")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name)
            or name in ids
        ):
            raise ValueError("Candidate IDs must be unique safe filenames")
        ids.add(name)
        stage = row.get("detector_stage", "progressive" if progressive is not None else "base")
        if stage not in {"base", "progressive"} or (stage == "progressive" and progressive is None):
            raise ValueError("Candidate detector_stage is unavailable")
        resolution = 384 if stage == "base" else 512
        inference = _config(
            InferenceConfig,
            {
                "resolutions": [resolution],
                "checkpoint_resolution": resolution,
                "device": "cpu",
                **row.get("inference", {}),
            },
            "inference",
        )
        if inference["checkpoint_resolution"] != resolution or inference["architecture"] != "nano":
            raise ValueError(
                "Inference must retain the actual trained detector resolution/architecture"
            )
        if inference["device"] != "cpu":
            raise ValueError("Portable recipe validation uses CPU image-only inference")
        refinement = row.get("query_rl")
        if refinement is not None:
            refinement = _config(RefinementConfig, refinement, "query RL")
            if (
                refinement["device"],
                refinement["cpu_threads"],
                refinement["hidden"],
                refinement["geometry_step"],
                refinement["identity_probability"],
            ) != ("cpu", 1, 128, 0.03, 0.75):
                raise ValueError("Query RL requires CPU1 and the fixed safe architecture")
        candidates.append(
            {"id": name, "detector_stage": stage, "query_rl": refinement, "inference": inference}
        )
    if not candidates:
        raise ValueError("A recipe must declare at least one candidate")
    minimum = loop._number(raw.get("minimum_ap", 0), "minimum_ap")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum_ap must be in [0, 1]")
    budget = {
        "max_attempts": 2 * len(candidates),
        "max_wall_seconds": 86400,
        "command_timeout_seconds": 43200,
        **raw.get("budget", {}),
    }
    if set(budget) != {"max_attempts", "max_wall_seconds", "command_timeout_seconds"}:
        raise ValueError("Unknown recipe budget field")
    return {
        "version": 1,
        "detector": base,
        "progressive": progressive,
        "export": export,
        "candidates": candidates,
        "budget": budget,
        "minimum_ap": minimum,
    }


def prepare(source: Path, recipe_path: Path, output: Path):
    from vision_rl.detection_training import inspect_split

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Recipe bundle already exists; choose a new directory")
    recipe = normalize_recipe(read(recipe_path))
    train, valid = inspect_split(source / "train"), inspect_split(source / "valid")
    if train["categories"] != valid["categories"]:
        raise ValueError("Train/validation foreground mappings differ")
    if set(train["image_sha256"]) & set(valid["image_sha256"]):
        raise ValueError("Identical images occur in training and validation")
    if len(train["categories"]) != 1 and any(
        row["query_rl"] is not None for row in recipe["candidates"]
    ):
        raise ValueError(
            "Query refinement currently requires one foreground class; "
            "detector-only supports multiple"
        )
    recipe.update(
        source=str(source),
        implementation_hashes=source_hashes(),
        category_ids=sorted(int(key) for key in train["categories"]),
        splits={"train": train, "validation": valid},
        test_access=False,
    )
    output.mkdir(parents=True)
    recipe_file = output / "recipe.json"
    write(recipe_file, recipe)
    common = ["{python}", "-m", "vision_rl.detection_recipe", "stage", "--recipe", str(recipe_file)]
    setup = []
    stages = ["base"] + (["progressive"] if recipe["progressive"] is not None else [])
    for name in stages:
        command = [*common, "--kind", "fit", "--name", name, "--output", "{setup_dir}"]
        if name == "progressive":
            command.extend(["--detector", "{setup:base}/model.safetensors"])
        setup.append(
            {
                "id": name,
                "command": command,
                "resume": [*command, "--resume"],
                "artifacts": ["{setup_dir}/model.safetensors", "{setup_dir}/receipt.json"],
            }
        )
    for name in stages:
        if not any(
            row["detector_stage"] == name and row["query_rl"] is not None
            for row in recipe["candidates"]
        ):
            continue
        command = [
            *common,
            "--kind",
            "cache",
            "--name",
            name,
            "--output",
            "{setup_dir}",
            "--detector",
            "{setup:" + name + "}/model.safetensors",
        ]
        setup.append(
            {
                "id": name + "-cache",
                "command": command,
                "resume": [*command, "--resume"],
                "artifacts": [
                    "{setup_dir}/train.pt",
                    "{setup_dir}/validation.pt",
                    "{setup_dir}/receipt.json",
                ],
            }
        )
    candidates = []
    for row in recipe["candidates"]:
        command = [
            *common,
            "--name",
            row["id"],
            "--output",
            "{candidate_dir}",
            "--detector",
            "{setup:" + row["detector_stage"] + "}/model.safetensors",
        ]
        if row["query_rl"] is not None:
            command.extend(["--cache", "{setup:" + row["detector_stage"] + "-cache}"])
        candidates.append(
            {
                "id": row["id"],
                "train": [*command, "--kind", "candidate"],
                "validate": [*command, "--kind", "validate"],
                "artifacts": ["{candidate_dir}/artifacts"],
                "validation_metrics": "{candidate_dir}/validation.json",
            }
        )
    plan = {
        "version": 1,
        "workspace": str(PACKAGE.parents[1]),
        "evaluation_mode": "development",
        "inputs": [
            str(recipe_file),
            str(source / "train"),
            str(source / "valid"),
            *map(str, source_paths()),
        ],
        "setup": setup,
        "candidates": candidates,
        "budget": recipe["budget"],
        "selection_metric": "AP",
        "validation_gate": {"AP": {"min": recipe["minimum_ap"]}, "errors": {"max": 0}},
        "recipe": str(recipe_file),
    }
    loop.validate_plan(plan, output)
    write(output / "plan.json", plan)
    return {
        "plan": str(output / "plan.json"),
        "setup_stages": len(setup),
        "candidates": len(candidates),
        "train_images": train["images"],
        "validation_images": valid["images"],
        "test_access": False,
    }


def descriptor(path):
    path = Path(path)
    return {"file": path.name, "sha256": loop._hash_path(path)}


def package_profile(
    detector,
    configuration,
    output,
    *,
    category_ids,
    training_origin,
    query_refiner=None,
    crop_refiner=None,
    crop_configuration=None,
    wheat=None,
):
    """Import numeric trained artifacts explicitly; this never claims to train them."""
    from vision_rl.detection_inference import InferenceConfig

    if output.exists():
        raise ValueError("Pipeline packages are immutable; choose a new directory")
    config = InferenceConfig(**configuration)
    config.validate()
    profile = {
        "schema_version": 3 if wheat is not None else 2,
        "kind": PROFILE_KIND,
        "detector": {"file": "detector.safetensors", "sha256": loop._hash_path(detector)},
        "query_refiner": None,
        "crop_refiner": None,
        "crop_configuration": crop_configuration,
        "inference": asdict(config),
        "category_ids": category_ids,
        "training_origin": training_origin,
        "implementation_hashes": source_hashes(wheat=wheat is not None),
        "runtime_packages": runtime_versions(),
    }
    if wheat is not None:
        profile["wheat"] = wheat
    files = {"detector": detector}
    for key, path in (("query_refiner", query_refiner), ("crop_refiner", crop_refiner)):
        if path is not None:
            files[key] = path
            profile[key] = {
                "file": key.replace("_", "-") + ".safetensors",
                "sha256": loop._hash_path(path),
            }
    validate_profile(profile)
    for key, path in files.items():
        if path.stat().st_size > ARTIFACT_LIMITS[key]:
            raise ValueError("Numeric artifact exceeds its trusted size bound")
    output.mkdir(parents=True)
    for key, path in files.items():
        shutil.copyfile(path, output / profile[key]["file"])
    write(output / "pipeline.json", profile)
    load_profile(output / "pipeline.json")
    return {"profile": str(output / "pipeline.json"), "training_performed": False}


def prepare_profile(source, profile_path, output):
    """Compile existing-model validation into the same durable development journal."""
    from vision_rl.detection_training import inspect_split

    source, profile_path, output = source.resolve(), profile_path.resolve(), output.resolve()
    profile, _, artifacts = load_profile(profile_path)
    if profile["schema_version"] != 2:
        raise ValueError("Imported pipelines must use schema2 package and source provenance")
    if output.exists():
        raise ValueError("Plan bundles are immutable")
    train, valid = inspect_split(source / "train"), inspect_split(source / "valid")
    if train["categories"] != valid["categories"] or set(train["image_sha256"]) & set(
        valid["image_sha256"]
    ):
        raise ValueError("Development splits overlap or their foreground mappings differ")
    if sorted(int(key) for key in valid["categories"]) != profile["category_ids"]:
        raise ValueError("Pipeline category order differs from this dataset")
    context = {
        "schema_version": 1,
        "source": str(source),
        "profile": str(profile_path),
        "splits": {"train": train, "validation": valid},
        "training_performed": False,
        "test_access": False,
        "implementation_hashes": source_hashes(),
    }
    output.mkdir(parents=True)
    context_path = output / "context.json"
    write(context_path, context)
    common = [
        "{python}",
        "-m",
        "vision_rl.detection_recipe",
        "stage",
        "--recipe",
        str(context_path),
        "--name",
        "imported-pipeline",
        "--output",
        "{candidate_dir}",
    ]
    plan = {
        "version": 1,
        "workspace": str(PACKAGE.parents[1]),
        "evaluation_mode": "development",
        "inputs": [
            str(context_path),
            str(profile_path),
            *map(str, artifacts.values()),
            str(source / "train"),
            str(source / "valid"),
            *map(str, source_paths()),
        ],
        "budget": {"max_attempts": 1, "max_wall_seconds": 86400, "command_timeout_seconds": 43200},
        "selection_metric": "AP",
        "validation_gate": {"AP": {"min": 0}, "errors": {"max": 0}},
        "pipeline_context": str(context_path),
        "candidates": [
            {
                "id": "imported-pipeline",
                "train": [*common, "--kind", "import"],
                "validate": [*common, "--kind", "validate"],
                "artifacts": ["{candidate_dir}/artifacts"],
                "validation_metrics": "{candidate_dir}/validation.json",
            }
        ],
    }
    loop.validate_plan(plan, output)
    write(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"), "training_performed": False, "test_access": False}


def _import_profile(context, args):
    profile_path = Path(context["profile"])
    _, _, artifacts = load_profile(profile_path)
    destination = args.output / "artifacts"
    destination.mkdir()
    shutil.copyfile(profile_path, destination / "pipeline.json")
    for path in artifacts.values():
        shutil.copyfile(path, destination / path.name)
    load_profile(destination / "pipeline.json")
    return {"training_performed": False, "test_access": False}


def validate_profile(profile):
    """Validate trusted profile metadata before staging any submitted artifact."""
    from vision_rl.detection_inference import InferenceConfig, _categories

    keys = {
        "schema_version",
        "kind",
        "detector",
        "query_refiner",
        "crop_refiner",
        "inference",
        "category_ids",
        "implementation_hashes",
        "training_origin",
    }
    version = profile.get("schema_version") if isinstance(profile, dict) else None
    if version in (2, 3):
        keys |= {"runtime_packages", "crop_configuration"}
    if version == 3:
        keys.add("wheat")
    if (
        not isinstance(profile, dict)
        or set(profile) != keys
        or type(version) is not int
        or version not in (1, 2, 3)
        or profile["kind"] != PROFILE_KIND
    ):
        raise ValueError("Unknown pipeline profile schema or fields")
    check_sources(profile["implementation_hashes"], wheat=version == 3)
    if version in (2, 3):
        from packaging.version import Version

        versions = profile["runtime_packages"]
        if (
            not isinstance(versions, dict)
            or set(versions) != set(RUNTIME_PACKAGES)
            or any(
                not isinstance(value, str) or str(Version(value).public) != value
                for value in versions.values()
            )
        ):
            raise ValueError("Runtime packages must pin every critical public release version")
    config = InferenceConfig(**profile["inference"])
    config.validate()
    if type(config.horizontal_flip) is not bool or any(
        type(getattr(config, name)) is not int
        for name in ("max_detections", "batch_size", "cpu_threads")
    ):
        raise ValueError("Inference flags and resource limits must use explicit JSON types")
    if config.checkpoint_resolution not in (384, 512) or config.device != "cpu":
        raise ValueError("Pipeline must fix its trained checkpoint resolution and CPU inference")
    _categories(tuple(profile["category_ids"]))
    if version == 3:
        from vision_rl.wheat_eval import BENCHMARK_ID, WHEAT_LOGIT_OFFSETS

        wheat = profile["wheat"]
        if (
            not isinstance(wheat, dict)
            or set(wheat)
            != {"benchmark_id", "protocol_sha256", "contract_sha256", "foreground_logit_offset"}
            or wheat["benchmark_id"] != BENCHMARK_ID
            or type(wheat["foreground_logit_offset"]) not in (int, float)
            or wheat["foreground_logit_offset"] not in WHEAT_LOGIT_OFFSETS
            or any(
                not isinstance(wheat[key], str) or not re.fullmatch(r"[0-9a-f]{64}", wheat[key])
                for key in ("protocol_sha256", "contract_sha256")
            )
        ):
            raise ValueError(
                "Wheat pipeline requires its protocol, contract and frozen grid offset"
            )
        if (
            profile["query_refiner"] is not None
            or profile["crop_refiner"] is not None
            or profile["category_ids"] != [1]
            or config.architecture != "nano"
            or config.horizontal_flip
            or list(config.resolutions) != [config.checkpoint_resolution]
            or config.max_detections < 300
        ):
            raise ValueError(
                "Wheat primary deployment requires all300 native Nano queries, one view"
            )
    crop = profile.get("crop_configuration")
    if profile["crop_refiner"] is not None and version != 2:
        raise ValueError(
            "This profile version has no registered crop adapter; refusing to ignore weights"
        )
    if profile["crop_refiner"] is None and crop is not None:
        raise ValueError("Crop configuration requires its numeric artifact")
    if profile["crop_refiner"] is not None:
        if config.cpu_threads > 16:
            raise ValueError("Crop inference supports at most 16 CPU threads")
        if (
            not isinstance(crop, dict)
            or set(crop)
            != {
                "adapter",
                "blend",
                "quality_blend",
                "training_detector_sha256",
                "training_proposals_sha256",
            }
            or crop["adapter"] != "crop_refinement"
        ):
            raise ValueError("Unknown or incomplete trusted crop adapter configuration")
        for name in ("blend", "quality_blend"):
            if not 0 <= loop._number(crop[name], name) <= 1:
                raise ValueError("Crop blend must be finite in [0,1]")
        for name in ("training_detector_sha256", "training_proposals_sha256"):
            if not isinstance(crop[name], str) or not re.fullmatch(r"[0-9a-f]{64}", crop[name]):
                raise ValueError("Crop training provenance requires exact SHA-256 digests")
    if not isinstance(profile["training_origin"], str):
        raise ValueError("Profile training_origin must be explicit text")
    names = set()
    for key in ARTIFACT_LIMITS:
        value = profile[key]
        if value is None and key != "detector":
            continue
        if not isinstance(value, dict) or set(value) != {"file", "sha256"}:
            raise ValueError("Numeric artifact descriptor requires only file and sha256")
        name = value["file"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Numeric artifact must be a sibling .safetensors file")
        if (
            name in names
            or not isinstance(value["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        ):
            raise ValueError("Numeric artifacts require unique filenames and SHA-256 digests")
        names.add(name)
    if profile["query_refiner"] is not None and len(profile["category_ids"]) != 1:
        raise ValueError("Query refinement requires one foreground class")
    return config


def profile_artifacts(profile):
    """Return the allowlisted files and bounds for verifier-owned safe staging."""
    validate_profile(profile)
    return {
        key: {**profile[key], "max_bytes": maximum}
        for key, maximum in ARTIFACT_LIMITS.items()
        if profile[key] is not None
    }


def load_profile(path: Path):
    """Strict portable numeric profile; no arbitrary loader or submitted code."""
    profile = read(path)
    config = validate_profile(profile)
    if (
        profile.get("runtime_packages") is not None
        and profile["runtime_packages"] != runtime_versions()
    ):
        raise ValueError("Installed inference package versions differ from the frozen profile")
    artifacts = {}
    for key, value in profile_artifacts(profile).items():
        name, maximum = value["file"], value["max_bytes"]
        artifact = path.parent / name
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size > maximum:
            raise ValueError("Numeric artifact is missing, nonregular, or exceeds its size bound")
        if loop._hash_path(artifact) != value["sha256"]:
            raise ValueError("Numeric artifact hash differs from the frozen pipeline")
        artifacts[key] = artifact
    return profile, config, artifacts


def predict_profile(profile_path: Path, manifest: Path, image_root: Path, output: Path):
    """Trusted image-only prediction; neither annotations nor a scoring API is accepted."""
    import torch

    from vision_rl.detection_inference import infer_images, load_image_manifest
    from vision_rl.detection_training import load_detector_safetensors

    started = time.monotonic()
    if output.exists():
        raise ValueError("Prediction evidence already exists; choose a new output directory")
    profile, config, artifacts = load_profile(profile_path)
    images = load_image_manifest(manifest, image_root)
    hashes = {str(row["path"]): loop._hash_path(row["path"]) for row in images}
    profile_hash = loop._hash_path(profile_path)
    manifest_hash = loop._hash_path(manifest)
    torch.set_num_threads(config.cpu_threads)
    detector = load_detector_safetensors(
        artifacts["detector"],
        architecture=config.architecture,
        num_classes=len(profile["category_ids"]),
        device=config.device,
        resolution=config.checkpoint_resolution,
    )
    predictions, timing = infer_images(
        detector,
        images,
        tuple(profile["category_ids"]),
        config,
        refinement=artifacts.get("query_refiner"),
        detector_sha256=profile["detector"]["sha256"],
    )
    if profile["schema_version"] == 3:
        from collections import Counter

        from vision_rl.wheat_eval import calibrate_wheat_predictions

        if Counter(row["image_id"] for row in predictions) != {row["id"]: 300 for row in images}:
            raise ValueError("Wheat native detector must retain exactly300 predictions per image")
        predictions = calibrate_wheat_predictions(
            predictions, profile["wheat"]["foreground_logit_offset"]
        )
    output.mkdir(parents=True)
    if "crop_refiner" in artifacts:
        from vision_rl.crop_refinement import load_model, refine_manifest

        crop = profile["crop_configuration"]
        _, _, metadata = load_model(artifacts["crop_refiner"], device="cpu")
        if (
            metadata.get("detector_sha256") != crop["training_detector_sha256"]
            or metadata.get("train_proposals_sha256") != crop["training_proposals_sha256"]
        ):
            raise ValueError("Crop checkpoint training provenance differs from frozen profile")
        proposals = output / "upstream-proposals.coco.json"
        write(proposals, predictions)
        composition = {
            "checkpoint_sha256": profile["crop_refiner"]["sha256"],
            "proposals_sha256": loop._hash_path(proposals),
            "manifest_sha256": manifest_hash,
            "deployment_detector_sha256": profile["detector"]["sha256"],
            "blend": crop["blend"],
            "quality_blend": crop["quality_blend"],
        }
        crop_timing = refine_manifest(
            artifacts["crop_refiner"],
            manifest,
            image_root,
            proposals,
            output / "predictions.coco.json",
            device="cpu",
            composition=composition,
            cpu_threads=config.cpu_threads,
        )
        timing["crop_refinement"] = crop_timing
    else:
        write(output / "predictions.coco.json", predictions)
    load_profile(profile_path)
    if loop._hash_path(profile_path) != profile_hash or loop._hash_path(manifest) != manifest_hash:
        raise ValueError("Pipeline profile or image manifest changed during prediction")
    if any(loop._hash_path(Path(name)) != digest for name, digest in hashes.items()):
        raise ValueError("Image bytes changed during prediction")
    timing["end_to_end_seconds"] = time.monotonic() - started
    write(output / "inference.json", timing)
    write(
        output / "provenance.json",
        {
            "schema_version": 1,
            "profile_sha256": profile_hash,
            "manifest_sha256": manifest_hash,
            "image_hashes": hashes,
            "configuration": asdict(config),
            "artifacts": {key: descriptor(value) for key, value in artifacts.items()},
            "label_access": False,
            "implementation_hashes": profile["implementation_hashes"],
            "runtime_packages": runtime_versions(),
            "runtime_builds": runtime_versions(public=False),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "proposal_pipeline_sha256": proposal_pipeline_sha256(profile),
            "predictions_sha256": loop._hash_path(output / "predictions.coco.json"),
            **({"wheat": profile["wheat"]} if profile["schema_version"] == 3 else {}),
        },
    )
    return {"images": len(images), "seconds": timing["end_to_end_seconds"], "label_access": False}


def _fit(recipe, args):
    from vision_rl.detection_training import (
        DetectorTrainingConfig,
        export_detector_safetensors,
        train_detector,
    )

    directory = args.output / "training"
    config = DetectorTrainingConfig(**recipe["detector" if args.name == "base" else "progressive"])
    result_path = directory / "result.json"
    if not args.resume and directory.exists():
        raise ValueError("New detector stage cannot adopt an existing training directory")
    complete = result_path.exists() and read(result_path).get("status") in {
        "completed",
        "preflight_completed",
    }
    if complete:
        training_plan = read(directory / "training-plan.json")
        expected_splits = recipe["splits"]
        if (
            training_plan["configuration"] != asdict(config)
            or any(
                training_plan["splits"][name]["fingerprint"] != expected_splits[name]["fingerprint"]
                for name in ("train", "validation")
            )
            or training_plan.get("model_resolution") != (384 if args.name == "base" else 512)
        ):
            raise ValueError("Completed phase configuration or data differs from this recipe")
        initialization = training_plan.get("initialization")
        if (args.detector is None and initialization is not None) or (
            args.detector is not None
            and (
                not initialization
                or initialization.get("source_sha256") != loop._hash_path(args.detector)
                or initialization.get("source_resolution") != 384
            )
        ):
            raise ValueError("Completed progressive phase used another initialization")
    if not complete:
        result = train_detector(
            Path(recipe["source"]),
            directory,
            config,
            resume=(directory / "training-plan.json").exists(),
            warmstart=args.detector,
            resolution=384 if args.name == "base" else 512,
            warmstart_resolution=None if args.name == "base" else 384,
        )
        if result["status"] not in {"completed", "preflight_completed"}:
            raise ValueError("Detector phase did not complete; later stages remain blocked")
    checkpoint = directory / "checkpoint_best_ema.pth"
    if read(result_path)["checkpoints"].get(checkpoint.name) != loop._hash_path(checkpoint):
        raise ValueError("Completed selected detector checkpoint changed")
    safe = args.output / "model.safetensors"
    if safe.exists():
        exported = read(safe.with_suffix(".json"))
        if exported["source_sha256"] != loop._hash_path(checkpoint) or exported[
            "safetensors_sha256"
        ] != loop._hash_path(safe):
            raise ValueError("Recovered detector export differs from the completed phase")
    else:
        exported = export_detector_safetensors(checkpoint, safe)
    result = read(result_path)
    write(
        args.output / "receipt.json",
        {
            "stage": args.name,
            "model": descriptor(safe),
            "source_checkpoint_sha256": exported["source_sha256"],
            "training_result": result,
            "training_plan_sha256": loop._hash_path(directory / "training-plan.json"),
            "resolution": 384 if args.name == "base" else 512,
            "test_access": False,
        },
    )
    return result


def _cache(recipe, args):
    from vision_rl.detection_rl import load_cache, pin_native_annotations
    from vision_rl.detection_training import export_queries

    records = {}
    for split, dirname in (("train", "train"), ("validation", "valid")):
        output = args.output / (split + ".pt")
        if not output.exists():
            export_queries(
                args.detector,
                Path(recipe["source"]) / dirname,
                output,
                split=split,
                queries=128,
                architecture="nano",
                resolution=384 if args.name == "base" else 512,
                **recipe["export"],
            )
        elif not args.resume:
            raise ValueError("New cache stage cannot adopt existing outputs")
        cache = load_cache(output, expected_split=split)
        expected_split = recipe["splits"]["train" if split == "train" else "validation"]
        expected_resolution = 384 if args.name == "base" else 512
        if cache["metadata"]["model_sha256"] != loop._hash_path(args.detector) or (
            cache["metadata"]["dataset_fingerprint"] != expected_split["fingerprint"]
            or cache["metadata"]["resolution"] != expected_resolution
            or len(cache["ids"]) != expected_split["images"]
            or cache["metadata"].get("truncated_images")
            or cache["metadata"]["retained_queries"] != 128
            or cache["metadata"]["annotation_path"]
            != str(Path(recipe["source"]) / dirname / "_annotations.coco.json")
        ):
            raise ValueError("Recovered query cache differs from the frozen detector/data/config")
        pin_native_annotations(cache)
        records[split] = {"sha256": loop._hash_path(output), "images": len(cache["ids"])}
    write(
        args.output / "receipt.json",
        {"stage": args.name + "-cache", "outputs": records, "test_access": False},
    )
    return records


def _candidate(recipe, args):
    row = next(row for row in recipe["candidates"] if row["id"] == args.name)
    artifacts = args.output / "artifacts"
    artifacts.mkdir(exist_ok=False)
    model = artifacts / "detector.safetensors"
    shutil.copyfile(args.detector, model)
    origin = "supervised_detector"
    policy = None
    if row["query_rl"] is not None:
        from vision_rl.detection_rl import (
            RefinementConfig,
            export_refinement_safetensors,
            train_refinement,
        )

        training = args.output / "query-training"
        summary = train_refinement(
            args.cache / "train.pt",
            args.cache / "validation.pt",
            training,
            RefinementConfig(**row["query_rl"]),
        )
        history = read(training / "history.json")
        selected = [entry for entry in history if entry["selected"]]
        origin = selected[-1]["phase"] if selected else "detector_identity"
        for name, target in (
            ("best.pt", "query-refiner.safetensors"),
            ("best-rl.pt", "query-rl.safetensors"),
            ("warmup.pt", "query-warmup.safetensors"),
        ):
            export_refinement_safetensors(training / name, artifacts / target)
        policy = artifacts / "query-refiner.safetensors"
        write(artifacts / "query-training.json", {"summary": summary, "selected_origin": origin})
    profile = {
        "schema_version": 2,
        "kind": PROFILE_KIND,
        "detector": descriptor(model),
        "query_refiner": descriptor(policy) if policy else None,
        "crop_refiner": None,
        "crop_configuration": None,
        "runtime_packages": runtime_versions(),
        "inference": row["inference"],
        "category_ids": recipe["category_ids"],
        "implementation_hashes": source_hashes(),
        "training_origin": origin,
    }
    write(artifacts / "pipeline.json", profile)
    load_profile(artifacts / "pipeline.json")
    return {"candidate": args.name, "training_origin": origin, "test_access": False}


def _validate(recipe, args):
    from vision_rl.detection_eval import evaluate_coco, load_coco

    annotation = Path(recipe["source"]) / "valid/_annotations.coco.json"
    dataset = load_coco(annotation)
    manifest = args.output / "validation-images.json"
    write(
        manifest,
        {
            "images": [
                {key: row[key] for key in ("id", "file_name", "width", "height")}
                for row in dataset["images"]
            ]
        },
    )
    output = args.output / "image-validation"
    predict_profile(args.output / "artifacts/pipeline.json", manifest, annotation.parent, output)
    # Native prediction arrays exceed the small configuration limit on full splits.
    prediction_path = output / "predictions.coco.json"
    if prediction_path.stat().st_size > 256 * 1024**2:
        raise ValueError("Native predictions exceed 256 MiB")
    predictions = json.loads(prediction_path.read_text(), parse_constant=loop._bad_constant)
    report = evaluate_coco(
        annotation, predictions, image_ids=[row["id"] for row in dataset["images"]]
    )
    report.update(errors=0, split="validation", test_access=False)
    write(output / "evaluation.json", report)
    metrics = {key: report[key] for key in ("AP", "AP50", "AP75", "n", "errors")}
    metrics.update(
        split="validation",
        test_access=False,
        pipeline_sha256=loop._hash_path(args.output / "artifacts/pipeline.json"),
    )
    write(args.output / "validation.json", metrics)
    return metrics


def stage(args):
    recipe = read(args.recipe)
    check_sources(recipe["implementation_hashes"])
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "fit": _fit,
        "cache": _cache,
        "candidate": _candidate,
        "validate": _validate,
        "import": _import_profile,
    }[args.kind](recipe, args)
    check_sources(recipe["implementation_hashes"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = sub.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--recipe", type=Path, required=True)
        command.add_argument(
            "--output" if name == "prepare" else "--state-dir", type=Path, required=True
        )
        if name == "run":
            command.add_argument("--background", action="store_true")
    for name in ("resume", "status"):
        command = sub.add_parser(name)
        command.add_argument("--state-dir", type=Path, required=True)
        if name == "resume":
            command.add_argument("--background", action="store_true")
    worker = sub.add_parser("stage", help=argparse.SUPPRESS)
    worker.add_argument("--recipe", type=Path, required=True)
    worker.add_argument(
        "--kind", choices=("fit", "cache", "candidate", "validate", "import"), required=True
    )
    worker.add_argument("--name", required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--detector", type=Path)
    worker.add_argument("--cache", type=Path)
    worker.add_argument("--resume", action="store_true")
    prediction = sub.add_parser("predict", help="predict from a frozen portable numeric pipeline")
    for name in ("profile", "manifest", "image-root", "output"):
        prediction.add_argument("--" + name, type=Path, required=True)
    imported = sub.add_parser(
        "run-profile", help="validate an explicitly imported trained pipeline"
    )
    for name in ("profile", "source", "state-dir"):
        imported.add_argument("--" + name, type=Path, required=True)
    imported.add_argument("--background", action="store_true")
    package = sub.add_parser(
        "package-profile", help="package numeric trained weights without training"
    )
    for name in ("detector", "config", "output"):
        package.add_argument("--" + name, type=Path, required=True)
    for name in ("query-refiner", "crop-refiner", "crop-config", "wheat-config"):
        package.add_argument("--" + name, type=Path)
    package.add_argument("--category-ids", type=int, nargs="+", required=True)
    package.add_argument("--training-origin", required=True)
    args = parser.parse_args(argv)
    if args.command in ("resume", "status"):
        command = [args.command, "--state-dir", str(args.state_dir)]
        if getattr(args, "background", False):
            command.append("--background")
        return loop.main(command)
    if args.command in {"run", "run-profile"}:
        state_dir = args.state_dir.resolve()
        bundle = state_dir.with_name(state_dir.name + "-plan")
        result = (
            prepare(args.source, args.recipe, bundle)
            if args.command == "run"
            else prepare_profile(args.source, args.profile, bundle)
        )
        command = ["run", "--plan", result["plan"], "--state-dir", str(state_dir)]
        if args.background:
            command.append("--background")
        return loop.main(command)
    if args.command == "prepare":
        result = prepare(args.source, args.recipe, args.output)
    elif args.command == "package-profile":
        result = package_profile(
            args.detector,
            read(args.config),
            args.output,
            category_ids=args.category_ids,
            training_origin=args.training_origin,
            query_refiner=args.query_refiner,
            crop_refiner=args.crop_refiner,
            crop_configuration=read(args.crop_config) if args.crop_config else None,
            wheat=read(args.wheat_config) if args.wheat_config else None,
        )
    elif args.command == "predict":
        result = predict_profile(args.profile, args.manifest, args.image_root, args.output)
    else:
        result = stage(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
