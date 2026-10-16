import hashlib
import json
import os
import shlex
import tomllib
from dataclasses import asdict

import pytest
from PIL import Image

from vision_rl.detection_task_builder import (
    _bundle_coco,
    _inference_configuration,
    _inspect_wheat_split,
    _pipeline_document,
    _stage_artifact,
    make_detection_replay_task,
    make_detection_task,
    make_pipeline_replay_task,
    make_wheat_training_task,
    verify_detection_submission,
    verify_wheat_training_submission,
    write_wheat_calibration,
)


@pytest.fixture
def coco_dataset(tmp_path):
    source = tmp_path / "source"
    for split, color in (("train", "red"), ("valid", "green"), ("test", "blue")):
        directory = source / split
        directory.mkdir(parents=True)
        Image.new("RGB", (16, 16), color).save(directory / f"private-{split}-name.png")
        document = {
            "images": [
                {
                    "id": 7,
                    "file_name": f"private-{split}-name.png",
                    "width": 16,
                    "height": 16,
                    "original_source": "/a/private/path",
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 7,
                    "category_id": 1,
                    "bbox": [2, 3, 5, 4],
                }
            ],
            "categories": [
                {"id": 0, "name": "pill-parent", "supercategory": "none"},
                {"id": 1, "name": "pill", "supercategory": "pill-parent"},
            ],
        }
        (directory / "_annotations.coco.json").write_text(json.dumps(document))
    return source


def test_task_keeps_test_images_out_of_build_context(tmp_path, coco_dataset):
    output = tmp_path / "task"
    result = make_detection_task(
        coco_dataset,
        output,
        detector_epochs=1,
        rl_epochs=2,
        warmup_epochs=0,
        grad_accum_steps=1,
        preflight_batches=1,
    )
    assert result["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert result["test_in_build_context"] is False
    assert result["smoke_test_only"] is True
    assert result["agent"] == "vision_rl.harbor_agent:DetectionTrainingAgent"
    assert sorted(p.name for p in (output / "environment/data").iterdir()) == ["train", "valid"]
    assert (output / "tests/data/test/image-000000.png").read_bytes() == (
        coco_dataset / "test/private-test-name.png"
    ).read_bytes()
    public = "\n".join(
        p.read_text()
        for p in (output / "environment").rglob("*")
        if p.is_file() and p.suffix != ".png"
    )
    for forbidden in ("private-test-name", str(coco_dataset), "/a/private/path", "data/test"):
        assert forbidden not in public
    train = (output / "environment/train.py").read_text()
    compile(train, "generated-train.py", "exec")
    assert "export_detector_safetensors" in train
    assert 'output / "model.safetensors"' in train
    assert 'output / "refiner.safetensors"' in train
    assert 'output / "refiner-rl-ablation.safetensors"' in train
    assert 'split="test"' not in train
    assert "bash /app/train.sh" in (output / "solution/solve.sh").read_text()
    config = tomllib.loads((output / "task.toml").read_text())
    assert config["environment"]["network_mode"] == "no-network"
    assert config["verifier"]["environment_mode"] == "separate"
    assert "gpus" not in config["environment"]
    assert "COPY ." not in (output / "environment/Dockerfile").read_text()
    assert "python -I /tests/verify.py" in (output / "tests/test.sh").read_text()
    pytest.importorskip("harbor")
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    task = Task(output)
    verifier = resolve_verifier_environment_definition(task.config, task.paths)
    assert verifier.directory == output / "environment"
    assert verifier.bundled_tests is False


def test_cuda_task_requests_gpu_but_keeps_offline_verifier(tmp_path, coco_dataset):
    output = tmp_path / "task"
    make_detection_task(coco_dataset, output, device="cuda", base_image="registry/vision:cuda")
    config = tomllib.loads((output / "task.toml").read_text())
    assert config["environment"]["gpus"] == 1
    assert config["environment"]["memory_mb"] == 16384
    assert config["environment"]["network_mode"] == "no-network"
    assert (output / "environment/Dockerfile").read_text().startswith("FROM registry/vision:cuda\n")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"detector_epochs": 0},
        {"rl_epochs": True},
        {"warmup_epochs": -1},
        {"preflight_batches": 2},
        {"batch_size": 0},
        {"grad_accum_steps": 0},
        {"device": "mps"},
        {"queries": 64},
        {"base_image": "image\nRUN bad"},
        {"timeout_sec": float("inf")},
    ],
)
def test_invalid_training_configuration_is_rejected(tmp_path, coco_dataset, kwargs):
    output = tmp_path / "task"
    with pytest.raises(ValueError):
        make_detection_task(coco_dataset, output, **kwargs)
    assert not output.exists()


def test_task_rejects_pixel_leakage_even_with_different_file_bytes(tmp_path, coco_dataset):
    Image.open(coco_dataset / "train/private-train-name.png").save(
        coco_dataset / "test/private-test-name.png", compress_level=0
    )
    assert (coco_dataset / "train/private-train-name.png").read_bytes() != (
        coco_dataset / "test/private-test-name.png"
    ).read_bytes()
    with pytest.raises(ValueError, match="leakage"):
        make_detection_task(coco_dataset, tmp_path / "task")


def test_task_rejects_escape_and_preserves_existing_destination(tmp_path, coco_dataset):
    annotation = coco_dataset / "test/_annotations.coco.json"
    doc = json.loads(annotation.read_text())
    doc["images"][0]["file_name"] = "../train/private-train-name.png"
    annotation.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="escapes"):
        make_detection_task(coco_dataset, tmp_path / "task")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("untouched")
    with pytest.raises(ValueError, match="already exists"):
        make_detection_task(coco_dataset, output)
    assert (output / "keep").read_text() == "untouched"


def test_task_rejects_inconsistent_categories(tmp_path, coco_dataset):
    annotation = coco_dataset / "test/_annotations.coco.json"
    doc = json.loads(annotation.read_text())
    doc["categories"][1]["name"] = "different object"
    annotation.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="categories must match"):
        make_detection_task(coco_dataset, tmp_path / "task")


@pytest.mark.parametrize(
    "kind", ["valid", "missing", "empty", "oversized", "symlink", "parent-link", "fifo"]
)
def test_artifact_staging_rejects_unsafe_files(tmp_path, kind):
    # macOS's /var is itself a symlink. Tests use the real absolute directory,
    # matching Linux's canonical /logs/artifacts mount in production.
    directory = tmp_path.resolve()
    source = directory / "source"
    source.mkdir()
    file = source / "model.safetensors"
    if kind != "missing":
        file.write_bytes(
            b"safe"
            if kind not in {"empty", "oversized"}
            else (b"" if kind == "empty" else b"x" * 11)
        )
    if kind == "symlink":
        target = source / "target"
        file.rename(target)
        file.symlink_to(target)
    elif kind == "parent-link":
        link = directory / "link"
        link.symlink_to(source, target_is_directory=True)
        file = link / file.name
    elif kind == "fifo":
        file.unlink()
        os.mkfifo(file)
    destination = directory / "staged"
    if kind == "valid":
        _stage_artifact(file, destination, 10)
        assert destination.read_bytes() == b"safe"
    else:
        with pytest.raises((ValueError, OSError)):
            _stage_artifact(file, destination, 10)


@pytest.mark.parametrize("kind", ["valid", "bad-model", "missing-refiner", "nan-score"])
def test_verifier_uses_only_numeric_artifacts_and_pristine_inference(tmp_path, monkeypatch, kind):
    pytest.importorskip("torch")
    import vision_rl.detection_eval as evaluation
    import vision_rl.detection_rl as refinement
    import vision_rl.detection_training as training

    directory = tmp_path.resolve()
    artifacts, logs, test = (directory / name for name in ("artifacts", "logs", "test"))
    artifacts.mkdir()
    (artifacts / "model.safetensors").write_bytes(b"mocked-model")
    if kind != "missing-refiner":
        (artifacts / "refiner.safetensors").write_bytes(b"mocked-refiner")
    (artifacts / "evil.py").write_text("raise RuntimeError('must never run')")
    (artifacts / "config.json").write_text('{"architecture": "untrusted"}')
    (artifacts / "queries.pt").write_bytes(b"untrusted-cache")
    calls = []

    def export(model, split_directory, cache_path, **kwargs):
        calls.append("export")
        assert model.read_bytes() == b"mocked-model"
        assert sorted(p.name for p in model.parent.iterdir()) == [
            "model.safetensors",
            "refiner.safetensors",
        ]
        assert split_directory == test
        assert kwargs == {"split": "test", "device": "cpu", "queries": 128, "batch_size": 1}
        if kind == "bad-model":
            raise ValueError("model tensors invalid")
        cache_path.write_bytes(b"trusted-generated-queries")

    def load(cache_path):
        assert cache_path.read_bytes() == b"trusted-generated-queries"
        return {"generated": True}

    def predict(cache, refiner, *, device):
        calls.append("predict")
        assert cache == {"generated": True}
        assert refiner.read_bytes() == b"mocked-refiner"
        assert device == "cpu"
        return ["trusted-predictions"]

    def evaluate(annotation, predictions, **kwargs):
        calls.append("evaluate")
        assert annotation == test / "_annotations.coco.json"
        assert predictions == ["trusted-coco"]
        return {"AP": float("nan") if kind == "nan-score" else 0.625}

    monkeypatch.setattr(training, "export_queries", export)
    monkeypatch.setattr(refinement, "load_cache", load)
    monkeypatch.setattr(refinement, "predict_refinement_safetensors", predict, raising=False)
    monkeypatch.setattr(refinement, "coco_predictions", lambda *args: ["trusted-coco"])
    monkeypatch.setattr(evaluation, "evaluate_coco", evaluate)
    result = verify_detection_submission(artifacts, test, logs)
    if kind == "valid":
        assert result["AP"] == 0.625
        assert calls == ["export", "predict", "evaluate"]
        assert (logs / "reward.txt").read_text() == "0.625\n"
    else:
        assert result["AP"] == 0
        assert (logs / "reward.txt").read_text() == "0\n"
        assert set(json.loads((logs / "error.json").read_text())) == {"type"}


@pytest.mark.parametrize(
    "inference,configuration",
    [
        ("unknown", None),
        ("native128", {}),
        ("label_free", {"device": "cuda"}),
        ("label_free", {"architecture": "untrusted"}),
        ("label_free", {"resolutions": [384, 384]}),
        ("label_free", {"resolutions": [2048]}),
        ("label_free", {"resolutions": [384.0]}),
        ("label_free", {"horizontal_flip": "yes"}),
        ("label_free", {"fusion_iou": float("nan")}),
        ("label_free", {"checkpoint_resolution": 256}),
    ],
)
def test_verifier_configuration_is_bounded(inference, configuration):
    with pytest.raises(ValueError):
        _inference_configuration(inference, configuration)


def test_replay_task_pins_only_numeric_artifacts_and_seals_test_data(tmp_path, coco_dataset):
    model, refiner = tmp_path / "model.safetensors", tmp_path / "policy.safetensors"
    model.write_bytes(b"numeric model")
    refiner.write_bytes(b"numeric policy")
    (tmp_path / "evil.py").write_text("raise RuntimeError('untrusted code')")
    output = tmp_path / "replay"
    result = make_detection_replay_task(
        coco_dataset / "test",
        model,
        refiner,
        output,
        inference="label_free",
        inference_config={
            "resolutions": [384, 512],
            "horizontal_flip": True,
            "checkpoint_resolution": 512,
        },
    )
    assert result["agent"] == "oracle"
    assert result["test_images"] == 1
    assert result["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert result["refiner_sha256"] == hashlib.sha256(refiner.read_bytes()).hexdigest()
    assert result["test_in_build_context"] is False
    assert sorted(path.name for path in (output / "environment").iterdir()) == [
        "Dockerfile",
        "submission",
    ]
    assert sorted(path.name for path in (output / "environment/submission").iterdir()) == [
        "model.safetensors",
        "refiner.safetensors",
    ]
    verify = (output / "tests/verify.py").read_text()
    compile(verify, "frozen-replay-verifier.py", "exec")
    assert result["model_sha256"] in verify
    assert result["refiner_sha256"] in verify
    assert "checkpoint_resolution" in verify
    assert "evil.py" not in verify
    solve = (output / "solution/solve.sh").read_text()
    assert "cp /app/submission/model.safetensors" in solve
    assert "python" not in solve
    config = tomllib.loads((output / "task.toml").read_text())
    assert config["environment"]["network_mode"] == "no-network"
    assert config["verifier"]["environment_mode"] == "separate"
    with pytest.raises(ValueError, match="already exists"):
        make_detection_replay_task(coco_dataset / "test", model, refiner, output)


def test_replay_rejects_pickle_path_without_creating_task(tmp_path, coco_dataset):
    model, refiner = tmp_path / "model.pth", tmp_path / "refiner.safetensors"
    model.write_bytes(b"pickle")
    refiner.write_bytes(b"policy")
    with pytest.raises(ValueError, match="only safetensors"):
        make_detection_replay_task(coco_dataset / "test", model, refiner, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_replay_rejects_artifact_that_differs_from_frozen_hash(tmp_path):
    pytest.importorskip("torch")
    directory = tmp_path.resolve()
    artifacts = directory / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.safetensors").write_bytes(b"different model")
    (artifacts / "refiner.safetensors").write_bytes(b"policy")
    result = verify_detection_submission(
        artifacts,
        directory / "unused-test",
        directory / "logs",
        expected_model_sha256="0" * 64,
    )
    assert result == {"error": {"type": "ValueError"}, "AP": 0.0}
    assert not (directory / "logs/predictions.json").exists()


def test_label_free_replay_uses_trusted_config_and_image_only_manifest(
    tmp_path,
    coco_dataset,
    monkeypatch,
):
    pytest.importorskip("torch")
    import vision_rl.detection_eval as evaluation
    import vision_rl.detection_inference as inference
    import vision_rl.detection_training as training

    artifacts, logs = tmp_path.resolve() / "artifacts", tmp_path.resolve() / "logs"
    artifacts.mkdir()
    (artifacts / "model.safetensors").write_bytes(b"model")
    (artifacts / "refiner.safetensors").write_bytes(b"policy")
    calls = []

    def load(model, **kwargs):
        assert model.read_bytes() == b"model"
        assert kwargs == {
            "architecture": "nano",
            "num_classes": 1,
            "device": "cpu",
            "resolution": 512,
        }
        return "trusted-model"

    def infer(model, images, categories, config, **kwargs):
        calls.append("infer")
        assert model == "trusted-model" and categories == (1,)
        assert len(images) == 1
        assert set(images[0]) == {"id", "file_name", "width", "height", "path"}
        assert images[0]["path"].is_file()
        assert config.resolutions == (384, 512)
        assert config.checkpoint_resolution == 512
        assert config.horizontal_flip is True
        assert config.architecture == "nano" and config.device == "cpu"
        assert config.max_detections == 500 and config.batch_size == 1
        assert kwargs["refinement"].read_bytes() == b"policy"
        assert kwargs["detector_sha256"] == hashlib.sha256(b"model").hexdigest()
        return [{"image_id": 7, "category_id": 1, "bbox": [2, 3, 5, 4], "score": 0.9}], {
            "images": 1,
            "elapsed_seconds": 1,
        }

    monkeypatch.setattr(training, "load_detector_safetensors", load)
    monkeypatch.setattr(inference, "infer_images", infer)
    monkeypatch.setattr(evaluation, "evaluate_coco", lambda *a, **kw: {"AP": 0.75})
    result = verify_detection_submission(
        artifacts,
        coco_dataset / "test",
        logs,
        inference="label_free",
        inference_config={
            "resolutions": [384, 512],
            "horizontal_flip": True,
            "checkpoint_resolution": 512,
        },
    )
    assert result == {"AP": 0.75}
    assert calls == ["infer"]
    assert len(json.loads((logs / "predictions.json").read_text())) == 1
    provenance = json.loads((logs / "inference.json").read_text())
    assert provenance["configuration"]["checkpoint_resolution"] == 512
    assert provenance["configuration"]["resolutions"] == [384, 512]
    assert provenance["submitted_code_executed"] is False


def _numeric_pipeline(directory, *, query=False, crop=False, name="detector.safetensors"):
    from vision_rl.detection_inference import InferenceConfig
    from vision_rl.detection_recipe import descriptor, source_hashes

    directory.mkdir()
    detector = directory / name
    detector.write_bytes(b"numeric detector")
    policy = directory / "query-refiner.safetensors"
    if query:
        policy.write_bytes(b"numeric policy")
    profile = {
        "schema_version": 1,
        "kind": "rfdetr_detection_pipeline",
        "detector": descriptor(detector),
        "query_refiner": descriptor(policy) if query else None,
        "crop_refiner": None,
        "inference": asdict(InferenceConfig(checkpoint_resolution=384, batch_size=1)),
        "category_ids": [1],
        "implementation_hashes": source_hashes(),
        "training_origin": "supervised_detector",
    }
    if crop:
        from vision_rl.detection_recipe import RUNTIME_PACKAGES

        crop_model = directory / "crop-refiner.safetensors"
        crop_model.write_bytes(b"numeric crop model")
        profile.update(
            schema_version=2,
            crop_refiner=descriptor(crop_model),
            crop_configuration={
                "adapter": "crop_refinement",
                "blend": 0.5,
                "quality_blend": 0.0,
                "training_detector_sha256": "1" * 64,
                "training_proposals_sha256": "2" * 64,
            },
            runtime_packages={name: "1.0" for name in RUNTIME_PACKAGES},
        )
    path = directory / "pipeline.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.mark.parametrize("query,crop", [(False, False), (True, False), (False, True), (True, True)])
def test_pipeline_task_packages_only_declared_weights_with_private_profile(
    tmp_path,
    coco_dataset,
    monkeypatch,
    query,
    crop,
):
    import vision_rl.detection_recipe as recipe

    profile = _numeric_pipeline(tmp_path / "frozen", query=query, crop=crop)
    if crop:
        versions = json.loads(profile.read_text())["runtime_packages"]
        monkeypatch.setattr(recipe, "runtime_versions", lambda **kwargs: versions)
    (profile.parent / "untrusted.py").write_text("raise RuntimeError('not submitted')")
    output = tmp_path / "replay"
    result = make_pipeline_replay_task(profile, coco_dataset / "test", output)
    expected = sorted(
        [
            "detector.safetensors",
            *(["query-refiner.safetensors"] if query else []),
            *(["crop-refiner.safetensors"] if crop else []),
        ]
    )
    assert sorted(p.name for p in (output / "environment/submission").iterdir()) == expected
    assert not (output / "environment/pipeline.json").exists()
    assert not (output / "environment/data").exists()
    assert json.loads((output / "tests/pipeline.json").read_text()) == json.loads(
        profile.read_text()
    )
    assert result["agent"] == "oracle" and result["test_in_build_context"] is False
    assert result["test_images"] == 1
    verifier = (output / "tests/verify.py").read_text()
    compile(verifier, "generated-pipeline-verifier.py", "exec")
    assert 'trusted_profile=Path("/tests/pipeline.json")' in verifier
    config = tomllib.loads((output / "task.toml").read_text())
    assert config["environment"]["network_mode"] == "no-network"
    assert config["verifier"]["environment_mode"] == "separate"
    with pytest.raises(ValueError, match="already exists"):
        make_pipeline_replay_task(profile, coco_dataset / "test", output)


def test_pipeline_replay_quotes_numeric_filenames_as_shell_data(tmp_path, coco_dataset):
    name = "weights;$(touch dangerous).safetensors"
    profile = _numeric_pipeline(tmp_path / "frozen", name=name)
    output = tmp_path / "replay"
    make_pipeline_replay_task(profile, coco_dataset / "test", output)
    copy = (output / "solution/solve.sh").read_text().splitlines()[-1]
    assert shlex.split(copy) == [
        "cp",
        "/app/submission/" + name,
        "/logs/artifacts/training/" + name,
    ]
    assert "'/app/submission/" in copy


@pytest.mark.parametrize("query,crop", [(False, False), (True, False), (False, True), (True, True)])
def test_pipeline_verifier_stages_optional_weights_and_uses_only_image_manifest(
    tmp_path,
    coco_dataset,
    monkeypatch,
    query,
    crop,
):
    import vision_rl.detection_eval as evaluation
    import vision_rl.detection_recipe as recipe

    root = tmp_path.resolve()
    profile = _numeric_pipeline(root / "frozen", query=query, crop=crop)
    if crop:
        versions = json.loads(profile.read_text())["runtime_packages"]
        monkeypatch.setattr(recipe, "runtime_versions", lambda **kwargs: versions)
        # Model loaders reject symlinks in every component. The verifier must
        # canonicalize its own temporary root even on systems where it is an alias.
        temporary = root / "trusted-temporary"
        temporary.mkdir()
        alias = root / "temporary-alias"
        alias.symlink_to(temporary, target_is_directory=True)
        monkeypatch.setattr("tempfile.tempdir", str(alias))
    artifacts, logs = root / "submitted", root / "logs"
    artifacts.mkdir()
    for path in profile.parent.glob("*.safetensors"):
        (artifacts / path.name).write_bytes(path.read_bytes())
    (artifacts / "pipeline.json").write_text('{"implementation":"evil.py"}')
    (artifacts / "evil.py").write_text("raise RuntimeError('not trusted')")
    calls = []
    predictions = [{"image_id": 7, "category_id": 1, "bbox": [2, 3, 5, 4], "score": 0.9}]

    def predict(staged, manifest, image_root, output):
        calls.append("predict")
        assert staged == staged.resolve()
        assert json.loads(staged.read_text()) == json.loads(profile.read_text())
        expected_roles = {"detector"}
        if query:
            expected_roles.add("query_refiner")
        if crop:
            expected_roles.add("crop_refiner")
        assert recipe.load_profile(staged)[2].keys() == expected_roles
        assert sorted(path.name for path in staged.parent.iterdir()) == sorted(
            [
                "detector.safetensors",
                "images.json",
                "pipeline.json",
                *(["query-refiner.safetensors"] if query else []),
                *(["crop-refiner.safetensors"] if crop else []),
            ]
        )
        document = json.loads(manifest.read_text())
        assert set(document) == {"images"}
        assert set(document["images"][0]) == {"id", "file_name", "width", "height"}
        assert image_root == coco_dataset / "test"
        output.mkdir()
        (output / "predictions.coco.json").write_text(json.dumps(predictions))
        (output / "inference.json").write_text('{"elapsed_seconds": 1}')
        (output / "provenance.json").write_text('{"label_access": false}')

    def evaluate(annotation, values, **kwargs):
        calls.append("evaluate")
        assert annotation == coco_dataset / "test/_annotations.coco.json"
        assert values == predictions
        assert kwargs["model"]["submitted_code_executed"] is False
        assert kwargs["model"]["prediction_provenance"]["label_access"] is False
        return {"AP": 0.75}

    monkeypatch.setattr(recipe, "predict_profile", predict)
    monkeypatch.setattr(evaluation, "evaluate_coco", evaluate)
    result = verify_detection_submission(
        artifacts, coco_dataset / "test", logs, trusted_profile=profile
    )
    assert result == {"AP": 0.75} and calls == ["predict", "evaluate"]
    assert (logs / "reward.txt").read_text() == "0.75\n"
    assert json.loads((logs / "predictions.json").read_text()) == predictions


@pytest.mark.parametrize(
    "failure", ["hash", "missing", "symlink", "source", "category", "override"]
)
def test_pipeline_verifier_rejects_invalid_submission_before_inference(
    tmp_path,
    coco_dataset,
    monkeypatch,
    failure,
):
    import vision_rl.detection_recipe as recipe

    root = tmp_path.resolve()
    profile = _numeric_pipeline(root / "frozen")
    artifacts = root / "submitted"
    artifacts.mkdir()
    detector = artifacts / "detector.safetensors"
    detector.write_bytes((profile.parent / detector.name).read_bytes())
    settings = {}
    if failure == "hash":
        detector.write_bytes(b"changed weights")
    elif failure == "missing":
        detector.unlink()
    elif failure == "symlink":
        detector.unlink()
        detector.symlink_to(profile.parent / detector.name)
    elif failure in {"source", "category"}:
        document = json.loads(profile.read_text())
        if failure == "source":
            document["implementation_hashes"]["detection_recipe.py"] = "0" * 64
        else:
            document["category_ids"] = [9]
        profile.write_text(json.dumps(document))
    else:
        settings["inference"] = "label_free"
    calls = []
    monkeypatch.setattr(recipe, "predict_profile", lambda *args: calls.append("inference"))
    result = verify_detection_submission(
        artifacts,
        coco_dataset / "test",
        root / "logs",
        trusted_profile=profile,
        **settings,
    )
    assert result["reward"] == 0 and "error" in result
    assert calls == []
    assert (root / "logs/reward.txt").read_text() == "0\n"
    assert not (root / "logs/predictions.json").exists()


@pytest.mark.parametrize("failure", ["oversized-crop", "crop-symlink", "runtime-version"])
def test_pipeline_crop_replay_rejects_unsafe_artifact_or_runtime_before_inference(
    tmp_path,
    coco_dataset,
    monkeypatch,
    failure,
):
    import vision_rl.detection_recipe as recipe

    root = tmp_path.resolve()
    profile = _numeric_pipeline(root / "frozen", crop=True)
    definition = json.loads(profile.read_text())
    versions = dict(definition["runtime_packages"])
    monkeypatch.setattr(recipe, "runtime_versions", lambda **kwargs: versions)
    artifacts = root / "submission"
    artifacts.mkdir()
    for source in profile.parent.glob("*.safetensors"):
        (artifacts / source.name).write_bytes(source.read_bytes())
    crop = artifacts / "crop-refiner.safetensors"
    if failure == "oversized-crop":
        with crop.open("wb") as stream:
            stream.truncate(32 * 1024**2 + 1)
    elif failure == "crop-symlink":
        crop.unlink()
        crop.symlink_to(profile.parent / crop.name)
    else:
        versions["torch"] = "2.0"
    calls = []
    monkeypatch.setattr(recipe, "predict_profile", lambda *args: calls.append("predict"))
    result = verify_detection_submission(
        artifacts,
        coco_dataset / "test",
        root / "logs",
        trusted_profile=profile,
    )
    assert result["reward"] == 0 and "error" in result
    assert calls == []
    assert (root / "logs/reward.txt").read_text() == "0\n"


@pytest.mark.parametrize("empty", [False, True])
def test_wheat_bundle_preserves_native_domains_and_ground_truth(tmp_path, empty):
    from vision_rl.wheat_eval import BENCHMARK_ID

    source = tmp_path / "wheat"
    source.mkdir()
    Image.new("RGB", (16, 16), "green").save(source / "original.png")
    rows = (
        []
        if empty
        else [{"id": i, "image_id": 7, "category_id": 1, "bbox": [-2, 1, 20, 5]} for i in (0, 1)]
    )
    document = {
        "images": [
            {
                "id": 7,
                "file_name": "original.png",
                "width": 16,
                "height": 16,
                "domain": "Arvalis_1",
                "session_id": 0,
                "original_image_id": "native-id",
                "native_split": "validation",
                "source_shard": "/private/source.parquet",
            }
        ],
        "annotations": rows,
        "categories": [{"id": 1, "name": "wheat_head"}],
        "info": {"benchmark_id": BENCHMARK_ID, "native_split": "validation"},
    }
    (source / "_annotations.coco.json").write_text(json.dumps(document))
    loaded = _pipeline_document(source, {"wheat": {"benchmark_id": BENCHMARK_ID}})
    inspection = _inspect_wheat_split(source.resolve(), loaded)
    assert inspection["images"] == 1
    output = tmp_path / "bundled"
    _bundle_coco(source, output, loaded, wheat=True)
    bundled = json.loads((output / "_annotations.coco.json").read_text())
    assert bundled["annotations"] == rows
    assert bundled["info"] == document["info"]
    image = bundled["images"][0]
    assert image["domain"] == "Arvalis_1" and image["session_id"] == 0
    assert image["original_image_id"] == "native-id"
    assert image["native_split"] == "validation"
    assert "source_shard" not in image
    assert (output / image["file_name"]).read_bytes() == (source / "original.png").read_bytes()
    document["info"]["benchmark_id"] = "pills"
    (source / "_annotations.coco.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="benchmark identity"):
        _pipeline_document(source, {"wheat": {"benchmark_id": BENCHMARK_ID}})


def _wheat_pipeline(directory, monkeypatch):
    import vision_rl.detection_recipe as recipe
    from vision_rl.wheat_eval import BENCHMARK_ID

    path = _numeric_pipeline(directory)
    profile = json.loads(path.read_text())
    profile.update(
        schema_version=3,
        crop_configuration=None,
        runtime_packages={name: "1.0" for name in recipe.RUNTIME_PACKAGES},
        implementation_hashes=recipe.source_hashes(wheat=True),
        wheat={
            "benchmark_id": BENCHMARK_ID,
            "protocol_sha256": "1" * 64,
            "contract_sha256": "2" * 64,
            "foreground_logit_offset": 0.5,
        },
    )
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(recipe, "runtime_versions", lambda **kwargs: profile["runtime_packages"])
    return path


def _as_wheat(source):
    from vision_rl.wheat_eval import BENCHMARK_ID

    for path in source.glob("*/_annotations.coco.json"):
        document = json.loads(path.read_text())
        document["categories"] = [{"id": 1, "name": "wheat_head"}]
        document["info"] = {"benchmark_id": BENCHMARK_ID, "native_split": "validation"}
        for row in document["images"]:
            row.update(domain="Arvalis_1", session_id=0)
        path.write_text(json.dumps(document))


def test_wheat_replay_uses_own_image_and_metric(tmp_path, coco_dataset, monkeypatch):
    from vision_rl.wheat_eval import BENCHMARK_ID
    from vision_rl.wheat_eval import PROTOCOL as WHEAT_PROTOCOL

    _as_wheat(coco_dataset)
    profile = _wheat_pipeline(tmp_path / "frozen", monkeypatch)
    output = tmp_path / "task"
    bundle = make_pipeline_replay_task(profile, coco_dataset / "test", output)
    assert bundle["benchmark_id"] == BENCHMARK_ID
    assert bundle["protocol"] == WHEAT_PROTOCOL
    assert bundle["reward_metric"] == "detection_acc_avg_dom"
    assert (output / "environment/Dockerfile").read_text().startswith("FROM vision-rl:wheat\n")
    assert not (output / "environment/data").exists()
    assert (
        json.loads((output / "tests/data/test/_annotations.coco.json").read_text())["images"][0][
            "domain"
        ]
        == "Arvalis_1"
    )


@pytest.mark.parametrize("valid", [True, False])
def test_wheat_verifier_scores_once_without_exposing_domain_or_labels(
    tmp_path, coco_dataset, monkeypatch, valid
):
    import vision_rl.detection_recipe as recipe
    import vision_rl.wheat_eval as evaluation

    _as_wheat(coco_dataset)
    root = tmp_path.resolve()
    profile = _wheat_pipeline(root / "frozen", monkeypatch)
    calls = []
    predictions = [{"image_id": 7, "category_id": 1, "bbox": [2, 3, 5, 4], "score": 0.6}]

    def predict(staged, manifest, image_root, output):
        calls.append("predict")
        assert json.loads(staged.read_text())["wheat"]["foreground_logit_offset"] == 0.5
        assert set(json.loads(manifest.read_text())["images"][0]) == {
            "id",
            "file_name",
            "width",
            "height",
        }
        output.mkdir()
        (output / "predictions.coco.json").write_text(json.dumps(predictions))
        (output / "inference.json").write_text("{}")
        (output / "provenance.json").write_text('{"label_access": false}')

    def score(annotation, values, **kwargs):
        calls.append("official_ada")
        assert values == predictions  # Already calibrated exactly once by predict_profile.
        assert json.loads(annotation.read_text())["images"][0]["domain"] == "Arvalis_1"
        return {"evaluation_valid": valid, "metrics": {"detection_acc_avg_dom": 0.375}}

    monkeypatch.setattr(recipe, "predict_profile", predict)
    monkeypatch.setattr(evaluation, "evaluate_wheat", score)
    logs = root / "logs"
    result = verify_detection_submission(
        profile.parent, coco_dataset / "test", logs, trusted_profile=profile
    )
    assert calls == ["predict", "official_ada"]
    assert (logs / "reward.txt").read_text() == ("0.375\n" if valid else "0\n")
    assert "AP" not in result
    if not valid:
        assert result["reward"] == 0 and "error" in result


@pytest.mark.parametrize(
    "failure",
    [None, "extra", "dtype", "shape", "nan", "grid", "oversized", "symlink", "model-link"],
)
def test_wheat_new_weights_use_computed_hash_and_only_numeric_calibration(
    tmp_path, coco_dataset, monkeypatch, failure
):
    np = pytest.importorskip("numpy")
    safe = pytest.importorskip("safetensors.numpy")
    import vision_rl.detection_recipe as recipe

    root = tmp_path.resolve()
    profile = _wheat_pipeline(root / "frozen", monkeypatch)
    _as_wheat(coco_dataset)
    artifacts = root / "submission"
    artifacts.mkdir()
    model = artifacts / "detector.safetensors"
    model.write_bytes(b"newly trained numeric weights")
    calibration = artifacts / "calibration.safetensors"
    write_wheat_calibration(calibration, -0.5)
    # Untrusted configs and code are present but can never alter the trusted template.
    (artifacts / "pipeline.json").write_text('{"architecture": "evil"}')
    (artifacts / "evil.py").write_text("raise RuntimeError('untrusted')")
    if failure in {"extra", "dtype", "shape", "nan", "grid"}:
        values = {"foreground_logit_offset": np.asarray([-0.5], dtype=np.float32)}
        if failure == "extra":
            values["architecture"] = np.asarray([1], dtype=np.float32)
        elif failure == "dtype":
            values["foreground_logit_offset"] = np.asarray([-0.5], dtype=np.float64)
        elif failure == "shape":
            values["foreground_logit_offset"] = np.asarray([-0.5, 0.5], dtype=np.float32)
        else:
            values["foreground_logit_offset"][0] = np.nan if failure == "nan" else 0.25
        safe.save_file(values, str(calibration))
    elif failure == "oversized":
        calibration.write_bytes(b"x" * 1025)
    elif failure in {"symlink", "model-link"}:
        target = calibration if failure == "symlink" else model
        original = root / target.name
        target.rename(original)
        target.symlink_to(original)
    calls = []

    def predict(staged, manifest, image_root, output):
        calls.append("predict")
        definition, _, weights = recipe.load_profile(staged)
        assert definition["detector"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
        assert (
            definition["detector"]["sha256"]
            != json.loads(profile.read_text())["detector"]["sha256"]
        )
        assert definition["wheat"]["foreground_logit_offset"] == -0.5
        assert weights.keys() == {"detector"}
        output.mkdir()
        (output / "predictions.coco.json").write_text("[]")
        (output / "inference.json").write_text("{}")
        (output / "provenance.json").write_text("{}")

    monkeypatch.setattr(recipe, "predict_profile", predict)
    monkeypatch.setattr(
        "vision_rl.wheat_eval.evaluate_wheat",
        lambda *a, **kw: {"evaluation_valid": True, "metrics": {"detection_acc_avg_dom": 0.25}},
    )
    logs = root / "logs"
    result = verify_wheat_training_submission(profile, artifacts, coco_dataset / "test", logs)
    if failure:
        assert result["reward"] == 0 and "error" in result and calls == []
    else:
        assert result["metrics"]["detection_acc_avg_dom"] == 0.25 and calls == ["predict"]
        assert json.loads((logs / "submission.json").read_text())["foreground_logit_offset"] == -0.5


def test_wheat_training_task_separates_test_and_reuses_native_agent(
    tmp_path, coco_dataset, monkeypatch
):
    pytest.importorskip("torch")
    _as_wheat(coco_dataset)
    profile = _wheat_pipeline(tmp_path / "frozen", monkeypatch)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "benchmark_id": "gwhd2021-hf-domain-ada-v1",
                "exploration_gate": {
                    "images": 128,
                    "groups": 64,
                    "active_query_probability": 1.0,
                    "minimum_fraction_images_with_useful_variation_and_positive_gain": 0.25,
                    "minimum_mean_best_of_64_official_image_accuracy_gain": 0.05,
                },
                "budget": {"maximum_epochs_equivalent": 8},
                "first_learning_test": {
                    "configuration": {"epochs": 2, "batch_images": 1},
                    "batch_images": 1,
                    "learning_rate": 0.00001,
                    "maximum_optimizer_updates": 4,
                },
                "data": {"train_images": 3655, "validation_images": 1476, "test_images": 1381},
            }
        )
    )
    definition = json.loads(profile.read_text())
    definition["wheat"]["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    profile.write_text(json.dumps(definition))
    output = tmp_path / "task"
    config = {
        "reward_backend": "wheat_ada",
        "domain_balanced_sampling": True,
        "epochs": 2,
        "max_steps": 4,
        "batch_images": 1,
        "pilot_min_reward_gain": 0.05,
    }
    result = make_wheat_training_task(
        coco_dataset,
        profile.parent / "detector.safetensors",
        profile,
        protocol,
        output,
        configuration=config,
        integration_only=True,
    )
    assert result["agent"] == "vision_rl.harbor_agent:DetectionTrainingAgent"
    assert result["reward_metric"] == "detection_acc_avg_dom"
    assert result["integration_only"] is True
    assert sorted(p.name for p in (output / "environment/data").iterdir()) == ["train", "valid"]
    assert not (output / "environment/template.json").exists()
    assert (output / "tests/template.json").is_file()
    script = (output / "environment/train.py").read_text()
    compile(script, "wheat-training.py", "exec")
    assert "mode=mode" in script and '"sft_updates", "sft_walltime"' in script
    assert '"/tests' not in script and '"test"' not in script
    assert "integration_only = True" in script
    task = tomllib.loads((output / "task.toml").read_text())
    assert task["environment"]["network_mode"] == "no-network"
    assert task["verifier"]["environment_mode"] == "separate"
    with pytest.raises(ValueError, match="Exact resume smoke"):
        make_wheat_training_task(
            coco_dataset,
            profile.parent / "detector.safetensors",
            profile,
            protocol,
            tmp_path / "invalid",
            configuration=config,
            integration_only=True,
            resume_check=True,
        )
    with pytest.raises(ValueError, match="complete native protocol splits"):
        make_wheat_training_task(
            coco_dataset,
            profile.parent / "detector.safetensors",
            profile,
            protocol,
            tmp_path / "not-a-benchmark",
            configuration=config,
        )
