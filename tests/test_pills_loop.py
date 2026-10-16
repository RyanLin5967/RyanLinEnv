"""Exercise the concrete development loop and the explicit final-test boundary."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")
pytest.importorskip("safetensors")

from vision_rl import experiment_loop as loop  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "pills_loop_test", Path(__file__).parents[1] / "scripts/pills_loop.py"
)
PIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPE)


@pytest.fixture
def caches(tmp_path):
    detector = tmp_path / "model.safetensors"
    detector.write_bytes(b"fixture identifier; development never loads the detector")
    model_hash = loop._hash_path(detector)
    paths = {}
    for split, dirname in (("train", "train"), ("validation", "valid")):
        directory = tmp_path / dirname
        directory.mkdir()
        annotation = directory / "_annotations.coco.json"
        annotation.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": i, "file_name": f"fixture-{i}.png", "width": 64, "height": 64}
                        for i in range(2)
                    ],
                    "categories": [
                        {"id": 0, "name": "unused-roboflow-parent", "supercategory": "none"},
                        {"id": 1, "name": "pill", "supercategory": "unused-roboflow-parent"},
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
                }
            )
        )
        for index in range(2):
            Image.new("RGB", (64, 64), (index * 60, 25, 70)).save(
                directory / f"fixture-{index}.png"
            )
        boxes = torch.tensor([0.25, 0.25, 0.75, 0.75]).repeat(2, 128, 1)
        logits = torch.full((2, 128), -10.0)
        logits[:, 0] = 4
        cache = {
            "version": 1,
            "split": split,
            "ids": [f"{split}-{i}" for i in range(2)],
            "coco_image_ids": [0, 1],
            "boxes": boxes,
            "logits": logits,
            "embeddings": torch.randn(2, 128, 256, generator=torch.Generator().manual_seed(7)),
            "targets": [boxes[i, :1].clone() for i in range(2)],
            "image_sizes": torch.tensor([[64, 64], [64, 64]]),
            "pixel_sha256": [f"{split}-{i}" for i in range(2)],
            "metadata": {
                "model_sha256": model_hash,
                "annotation_path": str(annotation),
                "category_id": 1,
                "truncated_images": False,
            },
        }
        paths[split] = tmp_path / f"{split}.pt"
        torch.save(cache, paths[split])
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "id": "small-real-rl",
                    "config": {
                        "warmup_epochs": 0,
                        "rl_epochs": 1,
                        "groups": 2,
                        "geometry_inference": "mean",
                        "cpu_threads": 1,
                    },
                }
            ]
        )
    )
    args = SimpleNamespace(
        detector=detector,
        train_cache=paths["train"],
        validation_cache=paths["validation"],
        candidates=candidates,
        output=tmp_path / "bundle",
        minimum_validation_ap=0,
        max_attempts=None,
        max_wall_seconds=120,
        command_timeout_seconds=60,
    )
    return args


def test_plan_contains_only_train_validation_and_real_stage_commands(caches):
    result = PIPE.prepare_plan(caches)
    plan = PIPE.read(Path(result["plan"]))
    assert plan["evaluation_mode"] == "development"
    assert "test_gate" not in plan
    assert result["test_access"] is False
    assert len(plan["candidates"]) == 1
    candidate = plan["candidates"][0]
    assert "evaluate" not in candidate and "test_metrics" not in candidate
    assert "stage" in candidate["train"] and candidate["train"][-1] == "train"
    assert candidate["validate"][-1] == "validate"
    assert all("/test/" not in item for item in plan["inputs"])
    assert str(caches.detector.resolve()) in plan["inputs"]
    assert str(caches.train_cache.resolve()) in plan["inputs"]


@pytest.mark.parametrize(
    "config",
    [
        {"device": "mps"},
        {"device": "cuda"},
        {"cpu_threads": 4},
        {"hidden": 64},
        {"geometry_step": 0.1},
    ],
)
def test_candidate_contract_rejects_accelerator_or_unsafe_export_settings(caches, config):
    caches.candidates.write_text(json.dumps([{"id": "bad", "config": config}]))
    with pytest.raises(ValueError, match="CPU|fixed safe"):
        PIPE.prepare_plan(caches)
    assert not caches.output.exists()


def test_detector_hash_must_match_both_query_caches(caches):
    caches.detector.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="match the immutable query caches"):
        PIPE.prepare_plan(caches)


def test_real_refinement_subprocess_exports_and_replays_without_test(caches):
    result = PIPE.prepare_plan(caches)
    state_dir = caches.output.parent / "state"
    loop.initialize(Path(result["plan"]), state_dir)
    result = loop.run_loop(state_dir)
    assert result["status"] == "awaiting_final_evaluation" and not result["test_consumed"]
    _, _, directory = PIPE.development_selection(state_dir)
    metrics = PIPE.read(directory / "validation.json")
    assert metrics["split"] == "validation" and metrics["test_access"] is False
    assert metrics["n"] == 2 and metrics["AP"] > 0.99
    assert metrics["rl_AP"] >= 0 and metrics["warmup_AP"] >= 0
    for name in ("policy", "rl-policy", "warmup-policy"):
        assert (directory / "artifacts" / (name + ".safetensors")).stat().st_size > 1000
    assert (
        PIPE.read(directory / "artifacts/selection.json")["training_summary"]["completed_epochs"]
        == 1
    )
    assert loop.run_loop(state_dir)["attempts_started"] == 1
    (directory / "artifacts/policy.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Pinned experiment input changed"):
        PIPE.development_selection(state_dir)


def test_unknown_capture_names_share_one_conservative_cluster():
    groups = PIPE.capture_groups(
        [
            {"id": 0, "file_name": "WIN_20240621_10_26_38_Pro_jpg.png"},
            {"id": 1, "file_name": "IMG_1234.png"},
            {"id": 2, "file_name": "IMG_5678.png"},
        ]
    )
    assert groups["1"] == groups["2"] == "unknown_capture_shared_cluster"
    assert groups["0"] == "WIN:20240621:10"
    with pytest.raises(ValueError, match="at least two capture groups"):
        PIPE.capture_groups([{"id": 0, "file_name": "unknown.png"}])


def test_final_profile_requires_validation_evidence_even_for_default_config():
    with pytest.raises(ValueError, match="configuration and validation"):
        PIPE.checked_inference_profile(None, None, {}, Path("unused"))


def test_native_report_reader_handles_full_matching_arrays_with_a_separate_bound(tmp_path):
    category = {
        "category_id": 1,
        "scores": [0.123456789] * 500,
        "matches": [[False] * 500 for _ in range(10)],
        "ignored_detections": [[False] * 500 for _ in range(10)],
        "ignored_ground_truth": [False] * 30,
    }
    report = {
        "AP": 0.9,
        "image_ids": list(range(423)),
        "n": 423,
        "match_cache": {
            "items": [{"image_id": index, "categories": [category]} for index in range(423)]
        },
    }
    path = tmp_path / "full-native-evaluation.json"
    path.write_text(json.dumps(report))
    assert path.stat().st_size > 10 * 1024 * 1024
    with pytest.raises(loop.LoopError, match="10 MiB"):
        PIPE.read(path)
    replay = PIPE.read_result(path)
    assert replay["AP"] == 0.9 and len(replay["match_cache"]["items"]) == 423
    with path.open("wb") as handle:
        handle.truncate(256 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="256 MiB"):
        PIPE.read_result(path)


def test_recovery_plan_reuses_identical_training_without_replacing_history(caches):
    initial = PIPE.prepare_plan(caches)
    first_state = caches.output.parent / "original-state"
    loop.initialize(Path(initial["plan"]), first_state)
    loop.run_loop(first_state)
    _, _, original = PIPE.development_selection(first_state)
    original_hash = loop._hash_path(original / "training")
    candidate = PIPE.read(caches.candidates)[0]
    candidate.update(id="recovered-real-rl", resume_training=str(original / "training"))
    PIPE.write(caches.candidates, [candidate])
    caches.output = caches.output.parent / "recovery-bundle"
    recovery = PIPE.prepare_plan(caches)
    second_state = caches.output.parent / "recovered-state"
    loop.initialize(Path(recovery["plan"]), second_state)
    result = loop.run_loop(second_state)
    assert result["status"] == "awaiting_final_evaluation" and not result["test_consumed"]
    _, _, restored = PIPE.development_selection(second_state)
    selection = PIPE.read(restored / "artifacts/selection.json")
    assert selection["resume_training"] == str(original / "training")
    assert selection["training_summary"]["completed_epochs"] == 1
    assert len(PIPE.read(restored / "training/history.json")) == 1
    assert loop._hash_path(original / "training") == original_hash
    assert loop._hash_path(original / "artifacts/policy.safetensors") == loop._hash_path(
        restored / "artifacts/policy.safetensors"
    )


def test_final_profile_rejects_gpu_before_loading_any_model(tmp_path):
    configuration = tmp_path / "profile.json"
    configuration.write_text('{"device":"mps"}')
    with pytest.raises(ValueError, match="explicitly use CPU"):
        PIPE.checked_inference_profile(configuration, tmp_path, {}, tmp_path)


@pytest.fixture
def portable_evidence(caches, monkeypatch):
    from vision_rl import detection_inference, detection_recipe, detection_training
    from vision_rl.detection_eval import evaluate_coco, load_coco

    source = caches.output.parent
    selected = source / "typed-candidate"
    detection_recipe.package_profile(
        caches.detector,
        {"checkpoint_resolution": 384},
        selected / "artifacts",
        category_ids=[1],
        training_origin="imported fixture detector",
    )
    annotation = source / "valid/_annotations.coco.json"
    document = load_coco(annotation)
    PIPE.write(selected / "validation-images.json", {"images": document["images"]})
    predictions = [
        {"image_id": i, "category_id": 1, "bbox": [16, 16, 32, 32], "score": 1} for i in range(2)
    ]
    monkeypatch.setattr(detection_training, "load_detector_safetensors", lambda *a, **k: object())
    monkeypatch.setattr(
        detection_inference,
        "infer_images",
        lambda *a, **k: (predictions, {"elapsed_seconds": 0.01}),
    )
    profile = selected / "artifacts/pipeline.json"
    evidence = selected / "image-validation"
    detection_recipe.predict_profile(
        profile, selected / "validation-images.json", annotation.parent, evidence
    )
    report = evaluate_coco(annotation, predictions)
    report.update(errors=0, split="validation", test_access=False)
    PIPE.write(evidence / "evaluation.json", report)
    return profile, evidence, annotation


@pytest.mark.parametrize("tamper", [None, "image", "predictions", "packages", "score"])
def test_pipeline_bridge_requires_exact_full_validation_evidence(portable_evidence, tamper):
    profile, evidence, annotation = portable_evidence
    if tamper == "image":
        image = PIPE.read(annotation)["images"][0]
        Image.new("RGB", (64, 64), "red").save(annotation.parent / image["file_name"])
    elif tamper == "predictions":
        PIPE.write(evidence / "predictions.coco.json", [])
    elif tamper == "packages":
        provenance = PIPE.read(evidence / "provenance.json")
        provenance["runtime_packages"]["torch"] = "0.1.0"
        PIPE.write(evidence / "provenance.json", provenance)
    elif tamper == "score":
        report = PIPE.read(evidence / "evaluation.json")
        report["AP"] = 0.1
        PIPE.write(evidence / "evaluation.json", report)
    if tamper:
        with pytest.raises(ValueError, match="differs"):
            PIPE.checked_pipeline_validation(profile, evidence, annotation)
    else:
        _, artifacts, report, _ = PIPE.checked_pipeline_validation(profile, evidence, annotation)
        assert set(artifacts) == {"detector"} and report["AP"] > 0.99


def test_pipeline_smoke_cannot_authorize_native_final_before_any_test_read(caches, tmp_path):
    from vision_rl.detection_training import inspect_split

    source = caches.output.parent
    (source / "test").write_text("Not readable as a native test directory")
    context = tmp_path / "typed-context.json"
    PIPE.write(
        context,
        {
            "source": str(source),
            "splits": {
                "train": inspect_split(source / "train"),
                "validation": inspect_split(source / "valid"),
            },
        },
    )
    args = SimpleNamespace(source=source, output=tmp_path / "forbidden-final")
    with pytest.raises(ValueError, match="all 423 validation"):
        PIPE.freeze_pipeline_final(args, {}, {"pipeline_context": str(context)}, tmp_path, -1)
    assert not args.output.exists()


def test_profile_helper_replays_only_validation_and_binds_code(caches, monkeypatch):
    from vision_rl import detection_inference, detection_training

    prepared = PIPE.prepare_plan(caches)
    state_dir = caches.output.parent / "profile-state"
    loop.initialize(Path(prepared["plan"]), state_dir)
    loop.run_loop(state_dir)
    config = caches.output.parent / "inference-config.json"
    PIPE.write(config, {"device": "cpu", "horizontal_flip": True})
    monkeypatch.setattr(detection_training, "load_detector_safetensors", lambda *a, **k: object())
    calls = []

    def infer(detector, images, category_ids, configuration, **kwargs):
        calls.append(images)
        assert len(images) == 2 and category_ids == (1,)
        assert configuration.horizontal_flip
        assert all(
            "annotations" not in row and Path(row["path"]).parent.name == "valid" for row in images
        )
        return [
            {"image_id": row["id"], "category_id": 1, "bbox": [16, 16, 32, 32], "score": 1}
            for row in images
        ], {"images": 2}

    monkeypatch.setattr(detection_inference, "infer_images", infer)
    output = caches.output.parent / "profile-evidence"
    result = PIPE.validate_profile(
        SimpleNamespace(state_dir=state_dir, config=config, output=output)
    )
    assert result["AP"] > 0.99 and not result["test_access"] and calls
    _, plan, selected = PIPE.development_selection(state_dir)
    context = PIPE.read(plan["pills_context"])
    provenance = PIPE.read(output / "provenance.json")
    provenance["implementation_hashes"][str(PIPE.ROOT / "src/vision_rl/detection_rl.py")] = "bad"
    PIPE.write(output / "provenance.json", provenance)
    with pytest.raises(ValueError, match="Pinned experiment input changed"):
        PIPE.checked_inference_profile(output / "profile.json", output, context, selected)


@pytest.fixture
def saved_frontiers(caches):
    from vision_rl import frontier

    api = PIPE.wrapper()
    pairs = []
    for index, (provider, structured) in enumerate(
        [("gemini", False), ("gemini", True), ("anthropic", False), ("openai", False)]
    ):
        input_dir = caches.output.parent / f"inputs-{index}"
        output_dir = caches.output.parent / f"responses-{index}"
        input_dir.mkdir()
        output_dir.mkdir()
        protocol = api["build_protocol"](
            SimpleNamespace(
                source=caches.output.parent,
                split="validation",
                count=None,
                seed=1729,
                prompt="rich",
                examples=0,
                provider=provider,
                structured_output=structured,
            )
        )
        PIPE.write(input_dir / "inputs.json", protocol)
        config = {
            "provider": provider,
            "model": frontier.PROVIDERS[provider][0],
            "effort": "adaptive" if provider == "anthropic" else "high",
            "max_output_tokens": 16384,
            "max_requests": 2,
            "max_inflight": 2,
            "max_transient_retries": 0,
        }
        if provider == "openai":
            config.update(
                image_detail=protocol["image_detail"], response_format=protocol["response_format"]
            )
        if structured:
            config["response_schema"] = protocol["response_schema"]
        identity = {
            "schema_version": 2,
            "config": config,
            "rows": [frontier._public(frontier._prepare(row)) for row in protocol["rows"]],
            "examples": [],
        }
        run = {"identity": identity, "identity_sha256": frontier._digest(identity)}
        PIPE.write(output_dir / "run.json", run)
        predictions, events = {}, []
        for attempt, row in enumerate(identity["rows"], 1):
            if provider == "openai":
                response = '{"detections":[{"label":"pill","box_2d":[16,16,48,48]}]}'
            else:
                response = '[{"label":"pill","box_2d":[250,250,750,750]}]'
            if index == 0:
                response = "[]"
            prediction = {"response": response}
            if structured and attempt == 2:
                prediction = {"response": "", "error": "fixture HTTP502"}
            predictions[row["id"]] = prediction
            events.extend(
                [
                    {
                        "event": "reserved",
                        "id": row["id"],
                        "attempt": attempt,
                        "request": row,
                        "identity_sha256": run["identity_sha256"],
                    },
                    {
                        "event": "completed",
                        "id": row["id"],
                        "attempt": attempt,
                        "prediction": prediction,
                        "status_code": 502 if prediction.get("error") else 200,
                        "identity_sha256": run["identity_sha256"],
                    },
                ]
            )
        (output_dir / "attempts.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events)
        )
        (output_dir / "predictions.jsonl").write_text(
            "".join(
                json.dumps({"id": identifier, **row}) + "\n"
                for identifier, row in predictions.items()
            )
        )
        model = {
            "provider": provider,
            "name": config["model"],
            "effort": config["effort"],
            "max_output_tokens": 16384,
            "protocol_sha256": loop._hash_path(input_dir / "inputs.json"),
        }
        report = api["score_predictions"](protocol, predictions, model)
        PIPE.write(output_dir / "evaluation.json", report)
        pairs.append((input_dir, output_dir))
    return SimpleNamespace(
        candidate=pairs,
        output=caches.output.parent / "baselines.json",
        max_transient_retries=2,
        retry_delay_seconds=2,
    )


def test_select_strongest_three_providers_preserves_schema_and_errors(saved_frontiers):
    result = PIPE.select_baselines(saved_frontiers)
    assert len(result["baselines"]) == 3
    assert result["required_providers"] == ["anthropic", "gemini", "openai"]
    selected = {row["config"]["provider"]: row for row in result["selected"]}
    assert selected["gemini"]["config"]["structured_output"] is True
    assert selected["gemini"]["errors"] == 1 and selected["gemini"]["AP"] > 0
    assert selected["openai"]["config"]["coordinate_format"] == "xyxy_pixels"
    assert selected["openai"]["config"]["image_detail"] == "original"
    assert selected["openai"]["AP"] > 0.99 and selected["openai"]["errors"] == 0
    PIPE.check_baseline_selection(result)
    result["baselines"][0] = result["baselines"][-1]
    with pytest.raises(ValueError, match="strongest measured"):
        PIPE.check_baseline_selection(result)


def test_unavailable_required_model_is_recorded_but_cannot_count_as_win(saved_frontiers):
    from vision_rl.frontier import PROVIDERS

    unavailable = saved_frontiers.output.with_name("unavailable.json")
    PIPE.write(
        unavailable,
        [
            {
                "provider": "qwen",
                "model": PROVIDERS["qwen"][0],
                "reason": "Managed route unavailable in this fixture",
            }
        ],
    )
    saved_frontiers.required_provider = ["gemini", "anthropic", "openai", "qwen"]
    saved_frontiers.unavailable_models = unavailable
    result = PIPE.select_baselines(saved_frontiers)
    assert result["selection_complete"] is False
    assert result["missing_required_providers"] == ["qwen"]
    assert result["unavailable_models"][0]["reason"]
    assert len(result["baselines"]) == 3
    with pytest.raises(ValueError, match="cannot count as wins"):
        PIPE.check_baseline_selection(result)


def test_hard_stopped_scope_is_retained_but_cannot_be_selected(saved_frontiers):
    output = saved_frontiers.candidate[-1][1]
    path = output / "attempts.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1].update(hard_stop=True, stop_reason="rate_limit", status_code=429)
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    result = PIPE.select_baselines(saved_frontiers)
    assert len(result["considered"]) == 4
    assert result["considered"][-1]["availability"]["hard_stops"][0]["status_code"] == 429
    assert result["missing_required_providers"] == ["openai"]
    assert not result["selection_complete"]
    assert all(row["provider"] != "openai" for row in result["baselines"])
    assert result["unavailable_models"][0]["provider"] == "openai"
    with pytest.raises(ValueError, match="cannot count as wins"):
        PIPE.check_baseline_selection(result)


def test_baseline_selection_rejects_edited_report(saved_frontiers):
    result_dir = saved_frontiers.candidate[0][1]
    report = PIPE.read(result_dir / "evaluation.json")
    report["AP"] = 1
    PIPE.write(result_dir / "evaluation.json", report)
    with pytest.raises(ValueError, match="independently rescored"):
        PIPE.select_baselines(saved_frontiers)


def test_offline_scoring_rejects_completed_answer_replacement(saved_frontiers):
    input_dir, result_dir = saved_frontiers.candidate[0]
    path = result_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["response"] = "a replacement answer"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="final journaled attempt"):
        PIPE.checked_saved_responses(input_dir, result_dir)


def test_finalize_requires_development_selection_before_loading_detector(tmp_path, monkeypatch):
    from vision_rl import detection_training

    called = []
    monkeypatch.setattr(
        detection_training, "load_detector_safetensors", lambda *a, **k: called.append(1)
    )
    args = SimpleNamespace(state_dir=tmp_path / "missing", output=tmp_path / "final", baseline=[])
    with pytest.raises(FileNotFoundError):
        PIPE.finalize(args)
    assert called == []
    assert not args.output.exists()


@pytest.mark.parametrize("backend", ["legacy", "portable_pipeline"])
def test_final_consumption_precedes_detector_and_cannot_be_repeated(tmp_path, monkeypatch, backend):
    from vision_rl import detection_inference, detection_recipe, detection_training

    seed = tmp_path / "input.json"
    seed.write_text("{}")
    plan = {
        "version": 1,
        "workspace": str(tmp_path),
        "evaluation_mode": "development",
        "inputs": [str(seed)],
        "budget": {"max_attempts": 1, "max_wall_seconds": 60, "command_timeout_seconds": 30},
        "selection_metric": "AP",
        "validation_gate": {"AP": {"min": 0.8}},
        "candidates": [
            {
                "id": "candidate",
                "train": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "Path('{candidate_dir}/artifact').write_text('weights')",
                ],
                "validate": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "Path('{candidate_dir}/validation.json').write_text('{\"AP\":1}')",
                ],
                "artifacts": ["{candidate_dir}/artifact"],
                "validation_metrics": "{candidate_dir}/validation.json",
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    PIPE.write(plan_path, plan)
    state_dir = tmp_path / "state"
    loop.initialize(plan_path, state_dir)
    loop.run_loop(state_dir)
    bundle = tmp_path / "final-protocol"
    bundle.mkdir()
    PIPE.write(bundle / "freeze.json", {})
    PIPE.write(bundle / "capture-groups.json", {"groups": {"0": "a", "1": "b"}})
    source = tmp_path / "source"
    (source / "test").mkdir(parents=True)
    PIPE.write(
        source / "test/_annotations.coco.json",
        {
            "images": [{"id": 0, "file_name": "a.png", "width": 32, "height": 32}],
            "categories": [{"id": 1, "name": "pill"}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16}
            ],
        },
    )
    PIPE.write(
        bundle / "specialist.json",
        {
            "source": str(source),
            "detector": str(seed),
            "inference_backend": backend,
            "pipeline_profile": str(seed),
            "test_image_hashes": {"0": loop._hash_path(seed)},
            "implementation_hashes": {},
            "inference_configuration": {},
            "inference_contract": {"final_test_images": 1},
        },
    )
    state = PIPE.read(state_dir / "state.json")
    state["final_protocol"] = {
        "directory": str(bundle),
        "freeze_sha256": loop._hash_path(bundle / "freeze.json"),
        "specialist_sha256": loop._hash_path(bundle / "specialist.json"),
    }
    PIPE.write(state_dir / "state.json", state)
    monkeypatch.setattr(PIPE, "wrapper", lambda: {"check_freeze": lambda path: {}})
    monkeypatch.setattr(PIPE, "offline_baselines", lambda *args: ({}, {}))
    monkeypatch.setattr(
        detection_inference, "load_image_manifest", lambda *args: [{"id": 0, "path": seed}]
    )
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        assert PIPE.read(state_dir / "state.json")["test_consumed"] is True
        raise RuntimeError("simulated detector failure after durable reservation")

    monkeypatch.setattr(detection_training, "load_detector_safetensors", interrupted)
    monkeypatch.setattr(detection_recipe, "predict_profile", interrupted)
    args = SimpleNamespace(state_dir=state_dir, output=tmp_path / "final", baseline=[])
    with pytest.raises(RuntimeError, match="simulated detector"):
        PIPE.finalize(args)
    assert PIPE.read(state_dir / "state.json")["status"] == "needs_fresh_test"
    with pytest.raises(ValueError, match="awaiting_final_evaluation"):
        PIPE.finalize(args)
    assert calls == [1]
