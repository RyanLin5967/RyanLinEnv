"""The shipped recipe compiles into one durable loop and fixed numeric profiles."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from vision_rl import detection_recipe as recipe  # noqa: E402
from vision_rl import experiment_loop as loop  # noqa: E402
from vision_rl.detection_inference import InferenceConfig  # noqa: E402


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "coco"
    for split, color in (("train", 40), ("valid", 120)):
        directory = source / split
        directory.mkdir(parents=True)
        images = []
        for index in range(2):
            name = f"{index}.png"
            Image.new("RGB", (64, 64), (color, index * 50, 25)).save(directory / name)
            images.append({"id": index, "file_name": name, "width": 64, "height": 64})
        recipe.write(
            directory / "_annotations.coco.json",
            {
                "images": images,
                "categories": [
                    {"id": 0, "name": "parent", "supercategory": "none"},
                    {"id": 1, "name": "pill", "supercategory": "parent"},
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": i,
                        "category_id": 1,
                        "bbox": [16, 16, 32, 32],
                        "area": 1024,
                        "iscrowd": 0,
                    }
                    for i in range(2)
                ],
            },
        )
    # Any attempt to inspect final data must fail, while development remains usable.
    (source / "test").write_text("Not a directory; development must never open it")
    return source


@pytest.fixture
def specification(tmp_path):
    path = tmp_path / "recipe.json"
    recipe.write(
        path,
        {
            "version": 1,
            "detector": {
                "device": "cpu",
                "epochs": 1,
                "batch_size": 1,
                "grad_accum_steps": 1,
                "workers": 0,
                "cpu_threads": 2,
                "preflight_batches": 2,
            },
            "progressive": {"epochs": 1, "preflight_batches": 2},
            "candidates": [
                {"id": "base-control", "detector_stage": "base"},
                {
                    "id": "progressive-rl",
                    "detector_stage": "progressive",
                    "query_rl": {"warmup_epochs": 0, "rl_epochs": 1, "groups": 2},
                },
            ],
        },
    )
    return path


def test_recipe_compiles_full_sequence_and_never_pins_test(dataset, specification, tmp_path):
    result = recipe.prepare(dataset, specification, tmp_path / "bundle")
    plan = recipe.read(result["plan"])
    assert [row["id"] for row in plan["setup"]] == ["base", "progressive", "progressive-cache"]
    assert result["train_images"] == result["validation_images"] == 2
    assert not result["test_access"] and plan["evaluation_mode"] == "development"
    assert all(not Path(path).is_relative_to(dataset / "test") for path in plan["inputs"])
    assert "{setup:base}/model.safetensors" in plan["setup"][1]["command"]
    assert plan["setup"][1]["resume"][-1] == "--resume"
    assert all("evaluate" not in row for row in plan["candidates"])
    assert plan["candidates"][0]["validate"][-1] == "validate"
    loop.initialize(Path(result["plan"]), tmp_path / "state")


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda x: x.update(arbitrary_python="evil.py"), "known fields"),
        (lambda x: x["candidates"][0].update(crop_refiner="unregistered"), "crop adapters"),
        (
            lambda x: x["candidates"][0].update(inference={"checkpoint_resolution": 512}),
            "trained detector",
        ),
        (lambda x: x["candidates"][1].update(query_rl={"device": "mps"}), "CPU1"),
        (lambda x: x.update(export={"device": "mps"}), "CPU"),
    ],
)
def test_recipe_rejects_unsupported_or_changed_contract(specification, change, match):
    raw = recipe.read(specification)
    change(raw)
    with pytest.raises(ValueError, match=match):
        recipe.normalize_recipe(raw)


def test_completed_fit_resume_exports_without_another_epoch(
    dataset, specification, tmp_path, monkeypatch
):
    from vision_rl import detection_training

    recipe.prepare(dataset, specification, tmp_path / "bundle")
    document = recipe.read(tmp_path / "bundle/recipe.json")
    output = tmp_path / "stage"
    output.mkdir()
    calls = []

    def fit(source, destination, config, **kwargs):
        calls.append(kwargs)
        destination.mkdir()
        checkpoint = destination / "checkpoint_best_ema.pth"
        checkpoint.write_bytes(b"actual fixture checkpoint")
        recipe.write(
            destination / "training-plan.json",
            {
                "configuration": asdict(config),
                "splits": document["splits"],
                "model_resolution": 384,
            },
        )
        result = {
            "status": "preflight_completed",
            "epochs_completed": 1,
            "checkpoints": {checkpoint.name: loop._hash_path(checkpoint)},
        }
        recipe.write(destination / "result.json", result)
        return result

    def export(checkpoint, safe):
        safe.write_bytes(checkpoint.read_bytes())
        record = {
            "source_sha256": loop._hash_path(checkpoint),
            "safetensors_sha256": loop._hash_path(safe),
        }
        recipe.write(safe.with_suffix(".json"), record)
        return record

    monkeypatch.setattr(detection_training, "train_detector", fit)
    monkeypatch.setattr(detection_training, "export_detector_safetensors", export)
    args = SimpleNamespace(output=output, name="base", resume=False, detector=None)
    recipe._fit(document, args)
    args.resume = True
    recipe._fit(document, args)
    assert len(calls) == 1
    assert recipe.read(output / "receipt.json")["resolution"] == 384
    (output / "training/checkpoint_best_ema.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        recipe._fit(document, args)
    assert len(calls) == 1


@pytest.fixture
def profile(tmp_path):
    detector = tmp_path / "detector.safetensors"
    detector.write_bytes(
        b"numeric loader tested in detection_training; this is descriptor metadata"
    )
    document = {
        "schema_version": 1,
        "kind": recipe.PROFILE_KIND,
        "detector": recipe.descriptor(detector),
        "query_refiner": None,
        "crop_refiner": None,
        "inference": asdict(InferenceConfig(checkpoint_resolution=384)),
        "category_ids": [1],
        "implementation_hashes": recipe.source_hashes(),
        "training_origin": "supervised_detector",
    }
    path = tmp_path / "pipeline.json"
    recipe.write(path, document)
    return path


def test_profile_metadata_can_be_checked_before_artifact_staging(profile):
    document = recipe.read(profile)
    (profile.parent / "detector.safetensors").unlink()
    assert recipe.validate_profile(document).checkpoint_resolution == 384
    artifacts = recipe.profile_artifacts(document)
    assert artifacts["detector"]["max_bytes"] == 200 * 1024**2
    assert set(artifacts) == {"detector"}
    with pytest.raises(ValueError, match="missing"):
        recipe.load_profile(profile)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda x: x.update(loader="untrusted.py"), "schema or fields"),
        (lambda x: x["detector"].update(file="../weights.safetensors"), "sibling"),
        (lambda x: x["detector"].update(sha256="fake"), "SHA-256"),
        (lambda x: x["inference"].update(checkpoint_resolution=None), "trained checkpoint"),
        (lambda x: x["inference"].update(horizontal_flip="yes"), "JSON types"),
        (lambda x: x.update(crop_refiner={"file": "crop.safetensors"}), "no registered crop"),
        (
            lambda x: x["implementation_hashes"].update({"detection_recipe.py": "modified"}),
            "implementation changed",
        ),
    ],
)
def test_profile_rejects_untrusted_or_ambiguous_fields(profile, change, match):
    document = recipe.read(profile)
    change(document)
    with pytest.raises(ValueError, match=match):
        recipe.validate_profile(document)


def test_profile_rejects_changed_weights(profile):
    (profile.parent / "detector.safetensors").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="hash differs"):
        recipe.load_profile(profile)


def test_image_only_profile_prediction_and_separate_validation(
    profile, dataset, monkeypatch, tmp_path
):
    from vision_rl import detection_inference, detection_training

    annotation = recipe.read(dataset / "valid/_annotations.coco.json")
    manifest = tmp_path / "images.json"
    recipe.write(manifest, {"images": annotation["images"]})
    monkeypatch.setattr(detection_training, "load_detector_safetensors", lambda *a, **k: object())
    calls = []

    def infer(detector, images, categories, configuration, **kwargs):
        assert categories == (1,) and configuration.checkpoint_resolution == 384
        assert kwargs["refinement"] is None
        assert all(set(row) == {"id", "file_name", "width", "height", "path"} for row in images)
        calls.append(1)
        return [
            {"image_id": row["id"], "category_id": 1, "bbox": [16, 16, 32, 32], "score": 1}
            for row in images
        ], {"elapsed_seconds": 0.1}

    monkeypatch.setattr(detection_inference, "infer_images", infer)
    output = tmp_path / "predictions"
    result = recipe.predict_profile(profile, manifest, dataset / "valid", output)
    assert result["images"] == 2 and result["label_access"] is False and calls == [1]
    assert not (output / "evaluation.json").exists()
    recipe.write(manifest, annotation)
    with pytest.raises(ValueError, match="without labels"):
        recipe.predict_profile(profile, manifest, dataset / "valid", tmp_path / "forbidden")


def upgrade_profile(path):
    document = recipe.read(path)
    document.update(
        schema_version=2, runtime_packages=recipe.runtime_versions(), crop_configuration=None
    )
    recipe.write(path, document)
    return document


def test_runtime_pin_accepts_build_suffix_but_rejects_different_release(profile, monkeypatch):
    document = upgrade_profile(profile)
    original = recipe.importlib.metadata.version
    monkeypatch.setattr(
        recipe.importlib.metadata,
        "version",
        lambda name: original(name).split("+")[0] + "+cpu" if name == "torch" else original(name),
    )
    recipe.load_profile(profile)
    document["runtime_packages"]["torch"] = "0.1.0"
    recipe.write(profile, document)
    recipe.validate_profile(document)  # Metadata staging does not require an installed GPU stack.
    with pytest.raises(ValueError, match="package versions"):
        recipe.load_profile(profile)


def test_upstream_digest_excludes_crop_and_paths_but_binds_detector_geometry(profile):
    document = upgrade_profile(profile)
    before = recipe.proposal_pipeline_sha256(document)
    document["crop_refiner"] = {"file": "crop.safetensors", "sha256": "c" * 64}
    document["crop_configuration"] = {"blend": 0.75}
    document["detector"]["file"] = "same-numeric-weights.safetensors"
    assert recipe.proposal_pipeline_sha256(document) == before
    document["inference"]["horizontal_flip"] = True
    assert recipe.proposal_pipeline_sha256(document) != before


def test_imported_profile_uses_same_development_loop_without_training(profile, dataset, tmp_path):
    upgrade_profile(profile)
    result = recipe.prepare_profile(dataset, profile, tmp_path / "import-plan")
    plan = recipe.read(result["plan"])
    assert not result["training_performed"] and not result["test_access"]
    assert plan["evaluation_mode"] == "development"
    assert plan["candidates"][0]["train"][-1] == "import"
    assert "setup" not in plan and "evaluate" not in plan["candidates"][0]
    assert all(not Path(path).is_relative_to(dataset / "test") for path in plan["inputs"])


def test_crop_adapter_uses_original_training_identity_and_new_deployment_identity(
    profile, dataset, tmp_path, monkeypatch
):
    from vision_rl import crop_refinement as crop
    from vision_rl import detection_inference, detection_training

    document = upgrade_profile(profile)
    crop_path = profile.parent / "crop.safetensors"
    config = crop.CropConfig(width=16, crop_size=64, cpu_threads=1)
    crop.save_model(
        crop.make_model(config),
        crop_path,
        config,
        {
            "detector_sha256": "a" * 64,
            "train_proposals_sha256": "b" * 64,
            "blend": 0.75,
            "quality_blend": 0,
        },
    )
    document.update(
        crop_refiner=recipe.descriptor(crop_path),
        crop_configuration={
            "adapter": "crop_refinement",
            "blend": 0.75,
            "quality_blend": 0,
            "training_detector_sha256": "a" * 64,
            "training_proposals_sha256": "b" * 64,
        },
    )
    recipe.write(profile, document)
    manifest = tmp_path / "image-only.json"
    recipe.write(
        manifest, {"images": recipe.read(dataset / "valid/_annotations.coco.json")["images"]}
    )
    monkeypatch.setattr(detection_training, "load_detector_safetensors", lambda *a, **k: object())
    proposals = [
        {"image_id": i, "category_id": 1, "bbox": [16, 16, 32, 32], "score": 0.9} for i in range(2)
    ]
    monkeypatch.setattr(
        detection_inference, "infer_images", lambda *a, **k: (proposals, {"elapsed_seconds": 0.01})
    )
    output = tmp_path / "crop-inference"
    recipe.predict_profile(profile, manifest, dataset / "valid", output)
    provenance = recipe.read(output / "provenance.json")
    assert provenance["label_access"] is False
    assert provenance["proposal_pipeline_sha256"] == recipe.proposal_pipeline_sha256(document)
    result = recipe.read(output / "predictions.coco.report.json")
    assert result["deployment_detector_sha256"] == document["detector"]["sha256"]
    assert result["training_detector_sha256"] == "a" * 64
    assert [row["score"] for row in recipe.read(output / "predictions.coco.json")] == [0.9, 0.9]
    document["crop_configuration"]["training_proposals_sha256"] = "d" * 64
    recipe.write(profile, document)
    with pytest.raises(ValueError, match="training provenance"):
        recipe.predict_profile(profile, manifest, dataset / "valid", tmp_path / "bad-crop")
