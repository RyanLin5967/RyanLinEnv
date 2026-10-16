"""Native wheat pipeline calibration, stage integrity, and no-go boundaries."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from vision_rl import detection_recipe as pipeline  # noqa: E402
from vision_rl import experiment_loop as loop  # noqa: E402
from vision_rl import wheat_eval  # noqa: E402
from vision_rl import wheat_recipe as recipe
from vision_rl.detection_inference import InferenceConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "native"
    for split, role, color in (("train", "train", 40), ("valid", "validation", 120)):
        directory = source / split
        directory.mkdir(parents=True)
        images = []
        for index in range(2):
            name = f"{index}.png"
            Image.new("RGB", (64, 64), (color, 50 * index, 25)).save(directory / name)
            images.append(
                {
                    "id": index,
                    "file_name": name,
                    "width": 64,
                    "height": 64,
                    "domain": "Arvalis_1" if index == 0 else "Arvalis_2",
                }
            )
        pipeline.write(
            directory / "_annotations.coco.json",
            {
                "info": {"benchmark_id": wheat_eval.BENCHMARK_ID, "native_split": role},
                "images": images,
                "categories": [{"id": 1, "name": "wheat_head"}],
                # Repeated out-of-frame targets retained; second image is empty.
                "annotations": [
                    {
                        "id": i,
                        "image_id": 0,
                        "category_id": 1,
                        "bbox": [-2, 0, 10, 10],
                        "iscrowd": 0,
                    }
                    for i in range(2)
                ],
            },
        )
    (source / "test").write_text("Do not inspect sealed data")
    return source


@pytest.fixture
def prepared(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(wheat_eval, "_backend", lambda: (None, None, None, {"verified": "fixture"}))
    specification = tmp_path / "recipe.json"
    pipeline.write(
        specification,
        {
            "version": 1,
            "integration_only": True,
            "policy": {
                "reward_backend": "wheat_ada",
                "domain_balanced_sampling": True,
                "pilot_min_reward_gain": 0.05,
                "max_steps": 1,
            },
        },
    )
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"descriptor fixture; real numeric loader tested independently")
    result = recipe.prepare(
        dataset,
        specification,
        ROOT / "docs/wheat-direct-rl-protocol-v1.json",
        ROOT / "docs/gwhd2021-hf-domain-ada-v1.json",
        tmp_path / "plan",
        initial_checkpoint=model,
    )
    return result, pipeline.read(tmp_path / "plan/context.json")


def test_native_inspection_preserves_empty_repeated_and_outside_targets(dataset):
    before = loop._hash_path(dataset / "valid/_annotations.coco.json")
    result = recipe.inspect_split(dataset / "valid", expected_split="validation")
    assert result["images"] == result["annotations"] == 2
    assert result["annotation_sha256"] == before
    assert loop._hash_path(dataset / "valid/_annotations.coco.json") == before
    with pytest.raises(ValueError, match="split identity"):
        recipe.inspect_split(dataset / "valid", expected_split="test")


def test_one_journal_has_both_controls_and_no_sealed_test(prepared, dataset, tmp_path):
    result, context = prepared
    plan = pipeline.read(result["plan"])
    assert [row["id"] for row in plan["setup"]] == list(recipe.STAGES)
    assert plan["evaluation_mode"] == "development" and plan["selection_metric"] == "ADA"
    assert "{setup:rl}/outputs" in plan["setup"][4]["command"]
    assert all("evaluate" not in row for row in plan["candidates"])
    assert all(not Path(path).is_relative_to(dataset / "test") for path in plan["inputs"])
    assert context["integration_only"] and not context["test_access"]
    loop.initialize(Path(result["plan"]), tmp_path / "state")


def test_benchmark_scope_rejects_tiny_data_before_model_access(dataset, tmp_path):
    spec = tmp_path / "recipe.json"
    pipeline.write(
        spec,
        {
            "version": 1,
            "policy": {
                **pipeline.read(ROOT / "recipes/wheat-direct-policy-64.json"),
                "max_steps": 1,
            },
        },
    )
    with pytest.raises(ValueError, match="every native"):
        recipe.prepare(
            dataset,
            spec,
            ROOT / "docs/wheat-direct-rl-protocol-v1.json",
            ROOT / "docs/gwhd2021-hf-domain-ada-v1.json",
            tmp_path / "plan",
            initial_checkpoint=tmp_path / "must-not-load.safetensors",
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"epochs": 100}, "epoch budget"),
        ({"max_steps": 1_000_000_000}, "first learning test update budget"),
        ({"max_steps": 65}, "first learning test update budget"),
        ({"batch_images": 1}, "prospective study: batch_images"),
        ({"learning_rate": 2e-5}, "prospective study: learning_rate"),
    ],
)
def test_prospective_budget_rejected_before_data_or_workers(tmp_path, change, match):
    specification = tmp_path / "recipe.json"
    policy = pipeline.read(ROOT / "recipes/wheat-direct-policy-64.json")
    pipeline.write(specification, {"version": 1, "policy": {**policy, **change}})
    output = tmp_path / "uncreated-plan"
    with pytest.raises(ValueError, match=match):
        recipe.prepare(
            tmp_path / "must-not-read-data",
            specification,
            ROOT / "docs/wheat-direct-rl-protocol-v1.json",
            ROOT / "docs/gwhd2021-hf-domain-ada-v1.json",
            output,
        )
    assert not output.exists()


def test_native_image_count_limits_updates_before_scope_or_model_access(dataset, tmp_path):
    specification = tmp_path / "recipe.json"
    policy = pipeline.read(ROOT / "recipes/wheat-direct-policy-64.json")
    pipeline.write(specification, {"version": 1, "policy": policy})
    with pytest.raises(ValueError, match="native epoch-equivalent cap"):
        recipe.prepare(
            dataset,
            specification,
            ROOT / "docs/wheat-direct-rl-protocol-v1.json",
            ROOT / "docs/gwhd2021-hf-domain-ada-v1.json",
            tmp_path / "uncreated-plan",
        )


@pytest.mark.parametrize("epoch", [1, 2, 4])
def test_real_study_requires_fixed_epoch3_before_data_access(tmp_path, epoch):
    specification = tmp_path / "recipe.json"
    study = pipeline.read(ROOT / "recipes/wheat-controlled-study.json")
    pipeline.write(specification, {**study, "snapshot_epoch": epoch})
    output = tmp_path / "uncreated-plan"
    with pytest.raises(ValueError, match="fixed epoch3"):
        recipe.prepare(
            tmp_path / "must-not-read-data",
            specification,
            ROOT / "docs/wheat-direct-rl-protocol-v1.json",
            ROOT / "docs/gwhd2021-hf-domain-ada-v1.json",
            output,
        )
    assert not output.exists()


def test_integration_exempts_first_study_fields_but_not_global_epochs(dataset, tmp_path):
    specification = tmp_path / "recipe.json"
    pipeline.write(
        specification,
        {
            "version": 1,
            "integration_only": True,
            "policy": {
                "reward_backend": "wheat_ada",
                "domain_balanced_sampling": True,
                "pilot_min_reward_gain": 0.05,
                "max_steps": 1,
                "epochs": 9,
            },
        },
    )
    with pytest.raises(ValueError, match="epoch budget"):
        recipe.prepare(
            dataset,
            specification,
            ROOT / "docs/wheat-direct-rl-protocol-v1.json",
            ROOT / "docs/gwhd2021-hf-domain-ada-v1.json",
            tmp_path / "uncreated-plan",
        )


@pytest.mark.parametrize(
    "initial",
    [
        {"device": "mps"},
        {"device": "cpu", "cpu_threads": 4},
        {"device": "cpu", "cpu_threads": 2, "epochs": 2},
    ],
)
def test_integration_guard_applies_to_initial_fit_too(initial):
    with pytest.raises(ValueError, match="Integration initial SFT"):
        recipe.normalize_recipe(
            {
                "version": 1,
                "integration_only": True,
                "initial_sft": {"epochs": 1, **initial},
                "snapshot_epoch": 1,
                "policy": {
                    "reward_backend": "wheat_ada",
                    "domain_balanced_sampling": True,
                    "max_steps": 1,
                },
            }
        )


def test_failed_pilot_never_calls_any_trainer(prepared, tmp_path, monkeypatch):
    from vision_rl import detector_policy_training as trainer

    _, context = prepared
    context["integration_only"] = False
    pilot = tmp_path / "pilot"
    pipeline.write(pilot / "summary.json", {"go": False, "scales": {"1.0": {"go": False}}})
    monkeypatch.setattr(
        trainer, "train", lambda *a, **k: pytest.fail("No-go launched optimization")
    )
    for mode in ("rl", "sft_updates", "sft_walltime"):
        output = tmp_path / mode
        output.mkdir()
        result = recipe._arm(context, SimpleNamespace(pilot=pilot, kind=mode), output)
        assert result["status"] == "no_go_action_space" and result["optimizer_steps"] == 0
        assert not list(output.glob("*.safetensors"))


def test_integration_pilot_never_claims_pass(prepared, tmp_path):
    _, context = prepared
    output = tmp_path / "pilot"
    result = recipe._pilot(context, SimpleNamespace(), output)
    assert result["integration_only"] and not result["go"]
    assert result["status"] == "integration_only_pilot_not_run"


def test_completed_stage_recovery_performs_no_extra_work(prepared, tmp_path, monkeypatch):
    result, _ = prepared
    args = SimpleNamespace(
        context=Path(result["plan"]).with_name("context.json"),
        kind="snapshot",
        output=tmp_path / "stage",
        resume=False,
    )
    calls = []

    def snapshot(context, args, output):
        calls.append(1)
        (output / "model.safetensors").write_bytes(b"fixture numeric snapshot")
        return {"status": "snapshot_ready"}

    monkeypatch.setattr(recipe, "_snapshot", snapshot)
    recipe.stage(args)
    before = loop._hash_path(args.output)
    args.resume = True
    recipe.stage(args)
    assert calls == [1] and before == loop._hash_path(args.output)
    (args.output / "outputs/model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="intact immutable"):
        recipe.stage(args)
    assert calls == [1]


def test_changed_inputs_reject_before_resume(prepared, tmp_path, monkeypatch):
    result, context = prepared
    Path(context["initial_checkpoint"]).write_bytes(b"changed")
    monkeypatch.setattr(recipe, "_snapshot", lambda *a: pytest.fail("changed input ran"))
    args = SimpleNamespace(
        context=Path(result["plan"]).with_name("context.json"),
        kind="snapshot",
        output=tmp_path / "stage",
        resume=True,
    )
    with pytest.raises(ValueError, match="input changed"):
        recipe.stage(args)


@pytest.fixture
def profile(tmp_path):
    detector = tmp_path / "detector.safetensors"
    detector.write_bytes(b"numeric descriptor fixture")
    document = {
        "schema_version": 3,
        "kind": pipeline.PROFILE_KIND,
        "detector": pipeline.descriptor(detector),
        "query_refiner": None,
        "crop_refiner": None,
        "crop_configuration": None,
        "inference": asdict(InferenceConfig(checkpoint_resolution=384)),
        "category_ids": [1],
        "implementation_hashes": pipeline.source_hashes(wheat=True),
        "runtime_packages": pipeline.runtime_versions(),
        "training_origin": "direct_native_rl",
        "wheat": {
            "benchmark_id": wheat_eval.BENCHMARK_ID,
            "protocol_sha256": "a" * 64,
            "contract_sha256": "b" * 64,
            "foreground_logit_offset": 0.5,
        },
    }
    path = tmp_path / "pipeline.json"
    pipeline.write(path, document)
    return path


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x["wheat"].update(foreground_logit_offset=0.7),
        lambda x: x["wheat"].update(foreground_logit_offset=True),
        lambda x: x["wheat"].update(protocol_sha256="missing"),
        lambda x: x["inference"].update(horizontal_flip=True),
        lambda x: x["inference"].update(max_detections=128),
        lambda x: x["inference"].update(resolutions=[512]),
        lambda x: x.update(query_refiner={"file": "query.safetensors", "sha256": "c" * 64}),
    ],
)
def test_wheat_profile_rejects_protocol_drift(profile, change):
    document = pipeline.read(profile)
    change(document)
    with pytest.raises(ValueError, match="Wheat"):
        pipeline.validate_profile(document)


def test_calibration_applied_exactly_once_to_all300_image_only_predictions(
    profile, dataset, tmp_path, monkeypatch
):
    from vision_rl import detection_inference, detection_training

    manifest = tmp_path / "images.json"
    annotation = pipeline.read(dataset / "valid/_annotations.coco.json")
    pipeline.write(
        manifest,
        {
            "images": [
                {k: r[k] for k in ("id", "file_name", "width", "height")}
                for r in annotation["images"]
            ]
        },
    )
    monkeypatch.setattr(detection_training, "load_detector_safetensors", lambda *a, **k: object())
    rows = [
        {"image_id": i, "category_id": 1, "bbox": [-3, 1, 10, 10], "score": 0.5}
        for i in range(2)
        for _ in range(300)
    ]
    monkeypatch.setattr(detection_inference, "infer_images", lambda *a, **k: (rows, {}))
    output = tmp_path / "prediction"
    pipeline.predict_profile(profile, manifest, dataset / "valid", output)
    assert pipeline.read(
        output / "predictions.coco.json"
    ) == wheat_eval.calibrate_wheat_predictions(rows, 0.5)
    assert pipeline.read(output / "provenance.json")["label_access"] is False
    assert not (output / "evaluation.json").exists()
    rows.pop()
    with pytest.raises(ValueError, match="exactly300"):
        pipeline.predict_profile(profile, manifest, dataset / "valid", tmp_path / "truncated")


def test_upstream_identity_binds_calibration_and_protocol(profile):
    document = pipeline.read(profile)
    first = pipeline.proposal_pipeline_sha256(document)
    document["wheat"]["foreground_logit_offset"] = 0
    assert pipeline.proposal_pipeline_sha256(document) != first


def test_strict_wheat_source_pin(profile):
    document = pipeline.read(profile)
    document["implementation_hashes"].pop("wheat_eval.py")
    with pytest.raises(ValueError, match="implementation changed"):
        pipeline.validate_profile(document)


def test_completed_arm_without_stage_receipt_does_not_repeat_optimizer_updates(
    prepared, tmp_path, monkeypatch
):
    from vision_rl import detector_policy_training as trainer

    _, context = prepared
    output, snapshot, pilot = tmp_path / "arm", tmp_path / "snapshot", tmp_path / "pilot"
    output.mkdir()
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"initial native weights")
    pipeline.write(pilot / "summary.json", {"integration_only": True, "go": False})
    (output / "best-trained.safetensors").write_bytes(b"trained native weights")
    summary = {
        "status": "completed",
        "optimizer_steps": 1,
        "selections": {
            "best-trained": {"sha256": loop._hash_path(output / "best-trained.safetensors")}
        },
    }
    for key, name in (
        ("trained_evaluation_sha256", "trained-evaluation.json"),
        ("initial_evaluation_sha256", "initial-evaluation.json"),
        ("trained_predictions_sha256", "trained-predictions.coco.json"),
    ):
        pipeline.write(output / name, {"fixture": name})
        summary[key] = loop._hash_path(output / name)
    pipeline.write(output / "summary.json", summary)
    pipeline.write(
        output / "plan.json",
        {
            "configuration": context["policy"],
            "mode": "rl",
            "reference_sha256": loop._hash_path(snapshot / "model.safetensors"),
            "protocol_sha256": loop._hash_path(Path(context["protocol"])),
        },
    )
    monkeypatch.setattr(trainer, "train", lambda *a, **k: pytest.fail("Extra optimizer work"))
    args = SimpleNamespace(kind="rl", snapshot=snapshot, pilot=pilot)
    before = loop._hash_path(output)
    assert recipe._arm(context, args, output)["optimizer_steps"] == 1
    assert loop._hash_path(output) == before
    (output / "trained-evaluation.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        recipe._arm(context, args, output)


def test_no_go_comparison_preserves_supervised_weights_and_never_claims_rl(prepared, tmp_path):
    _, context = prepared
    context["integration_only"] = False
    snapshot, rl, output = tmp_path / "snapshot", tmp_path / "rl", tmp_path / "comparison"
    snapshot.mkdir()
    output.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"original supervised weights")
    pipeline.write(
        snapshot / "evaluation.json",
        {
            "metrics": {"detection_acc_avg_dom": 0.6},
            "calibration": {"foreground_logit_offset": 0.5},
        },
    )
    pipeline.write(rl / "summary.json", {"status": "no_go_action_space"})
    result = recipe._comparison(context, SimpleNamespace(snapshot=snapshot, rl=rl), output)
    assert result["origin"] == "initial_supervised"
    assert not result["primary_requirement_satisfied"] and not result["native_test_accessed"]
    assert (output / "model.safetensors").read_bytes() == (
        snapshot / "model.safetensors"
    ).read_bytes()
    assert pipeline.read(output / "comparison.json")["status"] == "no_go_action_space"


@pytest.mark.parametrize("interruption", ["package", "inference"])
def test_partial_snapshot_substage_recovers_without_repeating_sft(
    prepared, tmp_path, monkeypatch, interruption
):
    from vision_rl import detection_training

    _, context = prepared
    context.update(initial_checkpoint=None, initial_sft={"device": "cpu"})
    stage_root = tmp_path / "snapshot-stage"
    source = stage_root / "training/snapshots/epoch-0003.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"completed fixed-epoch numeric snapshot")
    pipeline.write(
        source.with_suffix(".json"), {"model": {"safetensors_sha256": loop._hash_path(source)}}
    )
    output = stage_root / "outputs"
    output.mkdir()
    calls = {"package": 0, "inference": 0}
    monkeypatch.setattr(
        detection_training,
        "train_detector",
        lambda *a, **k: pytest.fail("Recovery trained an extra epoch"),
    )

    def package(model, config, destination, **kwargs):
        calls["package"] += 1
        destination.mkdir()
        (destination / "detector.safetensors").write_bytes(model.read_bytes())
        if interruption == "package" and calls["package"] == 1:
            raise InterruptedError("Crash during package copy")
        pipeline.write(destination / "pipeline.json", {"complete": True})

    def predict(profile, manifest, image_root, destination):
        calls["inference"] += 1
        destination.mkdir()
        pipeline.write(destination / "predictions.coco.json", [])
        if interruption == "inference" and calls["inference"] == 1:
            raise InterruptedError("Crash before provenance publication")
        pipeline.write(destination / "provenance.json", {"complete": True})

    def checked(profile, manifest, image_root, destination):
        assert pipeline.read(destination / "provenance.json")["complete"]
        return []

    monkeypatch.setattr(pipeline, "package_profile", package)
    monkeypatch.setattr(pipeline, "predict_profile", predict)
    monkeypatch.setattr(recipe, "checked_predictions", checked)
    monkeypatch.setattr(
        wheat_eval,
        "calibrated_wheat_evaluation",
        lambda *a, **k: ({"metrics": {"detection_acc_avg_dom": 0}}, []),
    )
    args = SimpleNamespace(output=stage_root)
    with pytest.raises(InterruptedError):
        recipe._snapshot(context, args, output)
    destination = output / ("raw-pipeline" if interruption == "package" else "image-validation")
    assert not destination.exists()
    interrupted = list((stage_root / "interrupted-work").iterdir())
    assert interrupted
    result = recipe._snapshot(context, args, output)
    assert result["status"] == "snapshot_ready" and calls[interruption] == 2
    assert all(path.exists() for path in interrupted)
    assert (output / "model.safetensors").read_bytes() == source.read_bytes()


def _parity_selection(backend="mps"):
    return {
        "ADA": 0.6521448980678212,
        "evaluation_backend": backend,
        "backend_parity_policy": recipe.BACKEND_PARITY_POLICY,
        "primary_requirement_satisfied": True,
        "controlled_comparison_reported_requirement_satisfied": True,
        "selection_backend_requirement_satisfied": True,
        "requirement_scope": "Fixture comparison only",
        "strongest_sft_backend_provenance_verified": False,
    }


def test_observed_mps_cpu_delta_is_a_recorded_diagnostic_not_a_claim_tolerance():
    result = recipe.backend_parity(_parity_selection(), 0.6521398750218478, "cpu")
    assert result["selection_ADA"] == 0.6521448980678212
    assert result["deployment_ADA"] == 0.6521398750218478
    assert result["signed_ADA_delta"] == pytest.approx(-5.023045973384654e-6)
    assert result["absolute_ADA_tolerance"] == 1e-5
    assert result["replay_within_declared_tolerance"]
    assert not result["permits_transferring_primary_requirement"]
    larger = recipe.backend_parity(_parity_selection(), 0.65212, "cpu")
    assert not larger["replay_within_declared_tolerance"]
    same = recipe.backend_parity(_parity_selection("cpu"), 0.6521398750218478, "cpu")
    assert same["absolute_ADA_tolerance"] == 1e-10
    assert not same["replay_within_declared_tolerance"]


@pytest.mark.parametrize("backend", ["cuda", "unknown"])
def test_unmeasured_backend_pairs_fail_closed(backend):
    with pytest.raises(ValueError, match="backend"):
        recipe.backend_parity(_parity_selection(backend), 0.6521448980678212, "cpu")


@pytest.mark.parametrize("score,passes", [(0.6521398750218478, True), (0.65212, False)])
def test_validation_preserves_backend_scores_and_never_transfers_cross_backend_claim(
    prepared, tmp_path, monkeypatch, score, passes
):
    _, context = prepared
    output = tmp_path / "candidate"
    pipeline.write(output / "artifacts/selection.json", _parity_selection())
    (output / "artifacts/pipeline.json").write_text("{}")
    before = loop._hash_path(output / "artifacts")
    monkeypatch.setattr(pipeline, "predict_profile", lambda *a: None)
    monkeypatch.setattr(recipe, "checked_predictions", lambda *a: [])
    monkeypatch.setattr(
        wheat_eval,
        "calibrated_wheat_evaluation",
        lambda *a, **k: pytest.fail("Deployment replay reselected the fixed calibration"),
    )
    monkeypatch.setattr(
        wheat_eval,
        "evaluate_wheat",
        lambda *a, **k: {
            "metrics": {"detection_acc_avg_dom": score},
            "n": 2,
            "domains": 2,
            "errors": 0,
        },
    )
    if passes:
        result = recipe._validate(context, SimpleNamespace(output=output))
        assert result["ADA"] == score
        assert result["controlled_comparison_reported_requirement_satisfied"]
        assert not result["primary_requirement_satisfied"]
    else:
        with pytest.raises(ValueError, match="independent ADA replay"):
            recipe._validate(context, SimpleNamespace(output=output))
        assert not (output / "validation.json").exists()
    parity = pipeline.read(output / "image-validation/backend-parity.json")
    assert parity["deployment_ADA"] == score
    assert parity["replay_within_declared_tolerance"] is passes
    assert loop._hash_path(output / "artifacts") == before


@pytest.mark.parametrize("strongest_backend", [None, "cpu", "mps"])
def test_native_selection_uses_common_backend_and_unknown_strongest_is_provisional(
    prepared, tmp_path, monkeypatch, strongest_backend
):
    from vision_rl import detector_policy_training as trainer

    _, context = prepared
    context["integration_only"] = False
    context["policy"]["device"] = "mps"
    snapshot, output = tmp_path / "snapshot", tmp_path / "comparison"
    snapshot.mkdir()
    output.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"immutable initial weights")
    reference = loop._hash_path(snapshot / "model.safetensors")

    def report(score):
        return {
            "metrics": {"detection_acc_avg_dom": score},
            "calibration": {"foreground_logit_offset": 0.5},
        }

    pipeline.write(snapshot / "evaluation.json", report(0.6521398750218478))
    paths = {mode: tmp_path / mode for mode in ("rl", "sft_updates", "sft_walltime")}
    for mode, path in paths.items():
        pipeline.write(
            path / "plan.json", {"configuration": {"device": "mps"}, "reference_sha256": reference}
        )
        pipeline.write(path / "trained-evaluation.json", {"wheat": report(0.652142)})
        (path / "best-trained.safetensors").write_bytes(b"native arm weights")
    initial = paths["rl"] / "initial-evaluation.json"
    pipeline.write(initial, {"wheat": report(0.6521448980678212)})
    pipeline.write(
        paths["rl"] / "summary.json",
        {
            "status": "completed",
            "reference_sha256": reference,
            "initial_evaluation_sha256": loop._hash_path(initial),
        },
    )
    strongest = tmp_path / "strongest.json"
    pipeline.write(strongest, {"evaluation_backend": strongest_backend})
    context["strongest_sft_report"] = str(strongest)

    def raw_comparison(*args):
        result = {"primary_requirement_satisfied": True}
        pipeline.write(args[-1], result)
        return result

    monkeypatch.setattr(trainer, "compare_wheat_controls", raw_comparison)
    result = recipe._comparison(context, SimpleNamespace(snapshot=snapshot, **paths), output)
    assert result["origin"] == "initial_supervised"  # CPU snapshot would choose the arm.
    assert result["ADA"] == 0.6521448980678212
    assert result["evaluation_backend"] == "mps"
    assert result["foreground_logit_offset"] == 0.5
    assert result["strongest_sft_declared_backend_matches_selection"] is (
        strongest_backend == "mps"
    )
    assert not result["strongest_sft_backend_provenance_verified"]
    assert not result["selection_backend_requirement_satisfied"]
    assert not result["primary_requirement_satisfied"]  # Deployment remains CPU.
    standalone = pipeline.read(output / "comparison.json")
    assert standalone["controlled_comparison_reported_requirement_satisfied"]
    assert not standalone["primary_requirement_satisfied"]
    assert standalone["claim_status"] == "provisional_report_only_strongest_sft"
    package = tmp_path / "packaged"

    def fake_package(*args, **kwargs):
        args[2].mkdir(parents=True)

    monkeypatch.setattr(pipeline, "package_profile", fake_package)
    recipe._package(context, SimpleNamespace(comparison=output, output=package))
    assert pipeline.read(package / "artifacts/comparison.json") == standalone
    (snapshot / "model.safetensors").write_bytes(b"changed initialization")
    with pytest.raises(ValueError, match="initialization or backend"):
        recipe._comparison(context, SimpleNamespace(snapshot=snapshot, **paths), output)
