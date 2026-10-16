"""Compile a controlled native wheat RL study into the existing durable runner.

The same journal owns snapshot -> train-only pilot -> RL -> equal-update SFT ->
equal-active-time SFT -> checked comparison -> image-only validation. Development
ends at awaiting_final_evaluation. This module has no sealed-test training path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from vision_rl import detection_recipe as pipeline
from vision_rl import experiment_loop as loop
from vision_rl.wheat_eval import BENCHMARK_ID

STAGES = ("snapshot", "pilot", "rl", "sft_updates", "sft_walltime", "comparison")
MAX_REPORT_BYTES = 256 * 1024**2
BACKEND_PARITY_POLICY = {
    "version": 1,
    "same_backend_absolute_ADA_tolerance": 1e-10,
    "cross_backend_absolute_ADA_tolerances": {"mps->cpu": 1e-5},
    "cross_backend_allows_primary_claim": False,
}


def read_report(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Wheat evidence must be a regular JSON file of at most256 MiB")
    return json.loads(path.read_text(), parse_constant=loop._bad_constant)


def inspect_split(directory, *, expected_split=None):
    """Preserve native annotations, including empty images and out-of-frame boxes."""
    from PIL import Image

    from vision_rl.data import image_hash
    from vision_rl.wheat_eval import validate_wheat_coco

    directory = Path(directory).resolve()
    annotation = directory / "_annotations.coco.json"
    document = read_report(annotation)
    images, _, _, ids = validate_wheat_coco(document, [])
    info = document.get("info", {})
    if info.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Expected the separately named author-HF wheat benchmark")
    if expected_split is not None and info.get("native_split") != expected_split:
        raise ValueError("Native split identity differs from the requested development role")
    rows = []
    for image_id in ids:
        row = images[image_id]
        name = row.get("file_name")
        if not isinstance(name, str):
            raise ValueError("Native wheat image needs a filename")
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Wheat image path escaped its split")
        digest = loop._hash_path(path)
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Wheat image dimensions differ from annotations")
        rows.append(
            {
                "id": image_id,
                "file_name": name,
                "sha256": digest,
                "pixel_sha256": image_hash(path),
                "domain": row["domain"],
            }
        )
    return {
        "images": len(ids),
        "annotations": len(document["annotations"]),
        "annotation_sha256": loop._hash_path(annotation),
        "image_records": rows,
        "fingerprint": loop._digest({"document": document, "images": rows}),
    }


def source_paths():
    return [
        *pipeline.source_paths(),
        pipeline.PACKAGE / "data.py",
        pipeline.PACKAGE / "wheat_eval.py",
        pipeline.PACKAGE / "detector_policy_training.py",
        Path(__file__).resolve(),
    ]


def source_hashes():
    return {str(path): loop._hash_path(path) for path in source_paths()}


def check_context(context):
    if context["implementation_hashes"] != source_hashes():
        raise ValueError("Wheat study implementation changed; preserve run and prepare a new plan")
    for name, digest in context["input_hashes"].items():
        if loop._hash_path(Path(name)) != digest:
            raise ValueError("Wheat study input changed")
    from vision_rl.wheat_eval import _backend

    if context["scorer"] != _backend()[3]:
        raise ValueError("Wheat scorer runtime differs from the frozen study")


def normalize_recipe(raw):
    from vision_rl.detector_policy_training import PolicyTrainingConfig

    allowed = {
        "version",
        "integration_only",
        "policy",
        "initial_sft",
        "snapshot_epoch",
        "inference",
        "budget",
    }
    if not isinstance(raw, dict) or raw.get("version") != 1 or set(raw) - allowed:
        raise ValueError("Unknown wheat recipe version or fields")
    integration = raw.get("integration_only", False)
    if type(integration) is not bool:
        raise ValueError("integration_only must be an explicit boolean")
    policy = pipeline._config(PolicyTrainingConfig, raw.get("policy", {}), "native policy")
    if policy["reward_backend"] != "wheat_ada" or not policy["domain_balanced_sampling"]:
        raise ValueError("Wheat study requires official ADA and equal-domain training draws")
    if integration and (
        policy["device"] != "cpu"
        or policy["cpu_threads"] > 2
        or policy["max_steps"] is None
        or policy["max_steps"] > 4
    ):
        raise ValueError("Integration-only recipe requires CPU<=2 and <=4 RL updates")
    from vision_rl.detection_inference import InferenceConfig

    inference = pipeline._config(
        InferenceConfig,
        {
            "resolutions": [policy["resolution"]],
            "checkpoint_resolution": policy["resolution"],
            "device": "cpu",
            "cpu_threads": 2,
            "batch_size": policy["batch_images"],
            **raw.get("inference", {}),
        },
        "native inference",
    )
    if inference["checkpoint_resolution"] != policy["resolution"]:
        raise ValueError("Inference resolution must match the native trained model")
    initial = raw.get("initial_sft")
    if initial is not None:
        from vision_rl.detection_training import DetectorTrainingConfig

        initial = pipeline._config(DetectorTrainingConfig, initial, "initial SFT")
        if initial["architecture"] != "nano":
            raise ValueError("The native wheat recipe requires the Nano architecture")
        if integration and (
            initial["device"] != "cpu"
            or initial["cpu_threads"] > 2
            or initial["workers"] != 0
            or initial["epochs"] != 1
            or initial["preflight_batches"] > 4
        ):
            raise ValueError("Integration initial SFT requires CPU<=2 and one tiny epoch")
    epoch = raw.get("snapshot_epoch", 3)
    if type(epoch) is not int or epoch < 1 or initial is not None and epoch > initial["epochs"]:
        raise ValueError("Snapshot epoch must lie within the declared initial SFT schedule")
    if not integration and epoch != 3:
        raise ValueError("The prospective wheat study requires the fixed epoch3 SFT snapshot")
    budget = {
        "max_attempts": 2,
        "max_wall_seconds": 172800,
        "command_timeout_seconds": 86400,
        **raw.get("budget", {}),
    }
    if set(budget) != {"max_attempts", "max_wall_seconds", "command_timeout_seconds"}:
        raise ValueError("Unknown wheat budget field")
    return {
        "version": 1,
        "integration_only": integration,
        "policy": policy,
        "initial_sft": initial,
        "snapshot_epoch": epoch,
        "inference": inference,
        "budget": budget,
        "backend_parity_policy": json.loads(json.dumps(BACKEND_PARITY_POLICY)),
    }


def wheat_descriptor(context, offset):
    return {
        "benchmark_id": BENCHMARK_ID,
        "protocol_sha256": context["input_hashes"][context["protocol"]],
        "contract_sha256": context["input_hashes"][context["contract"]],
        "foreground_logit_offset": offset,
    }


def prepare(
    source,
    recipe,
    protocol,
    contract,
    output,
    *,
    initial_checkpoint=None,
    strongest_sft_report=None,
):
    from vision_rl.detector_policy_training import (
        PolicyTrainingConfig,
        _validate_protocol,
        validate_study_budget,
    )
    from vision_rl.wheat_eval import _backend

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Wheat plan bundles are immutable")
    spec = normalize_recipe(pipeline.read(recipe))
    protocol, contract = protocol.resolve(), contract.resolve()
    declaration, benchmark = pipeline.read(protocol), pipeline.read(contract)
    policy = PolicyTrainingConfig(**spec["policy"])
    _validate_protocol(policy, declaration, integration_only=spec["integration_only"])
    if benchmark["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("Wrong benchmark contract")
    splits = {
        name: inspect_split(source / dirname, expected_split=name)
        for name, dirname in (("train", "train"), ("validation", "valid"))
    }
    validate_study_budget(
        policy,
        declaration,
        train_images=splits["train"]["images"],
        integration_only=spec["integration_only"],
    )
    if spec["integration_only"]:
        if any(row["images"] > 4 for row in splits.values()):
            raise ValueError("Integration-only study permits at most4 images per development split")
    elif any(
        splits[name]["images"] != declaration["data"][name + "_images"]
        or splits[name]["images"] != benchmark["splits"][name]["images"]
        for name in splits
    ):
        raise ValueError("Benchmark study requires every native train/validation image")
    for key in ("sha256", "pixel_sha256"):
        if {r[key] for r in splits["train"]["image_records"]} & {
            r[key] for r in splits["validation"]["image_records"]
        }:
            raise ValueError("Training and validation image bytes/pixels overlap")
    if (initial_checkpoint is None) == (spec["initial_sft"] is None):
        raise ValueError(
            "Declare exactly one imported early snapshot or initial_sft training phase"
        )
    if strongest_sft_report is None and not spec["integration_only"]:
        raise ValueError("A frozen strongest-available SFT report is required for benchmark claims")
    paths = [protocol, contract, source / "train", source / "valid"]
    if initial_checkpoint is not None:
        initial_checkpoint = initial_checkpoint.resolve()
        paths.append(initial_checkpoint)
    if strongest_sft_report is not None:
        strongest_sft_report = strongest_sft_report.resolve()
        paths.append(strongest_sft_report)
    context = {
        **spec,
        "source": str(source),
        "protocol": str(protocol),
        "contract": str(contract),
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "strongest_sft_report": str(strongest_sft_report) if strongest_sft_report else None,
        "splits": splits,
        "input_hashes": {str(p): loop._hash_path(p) for p in paths},
        "implementation_hashes": source_hashes(),
        "scorer": _backend()[3],
        "test_access": False,
    }
    # Validate the exact native deployment contract before any worker can train.
    profile = {
        "schema_version": 3,
        "kind": pipeline.PROFILE_KIND,
        "detector": {"file": "detector.safetensors", "sha256": "0" * 64},
        "query_refiner": None,
        "crop_refiner": None,
        "crop_configuration": None,
        "inference": spec["inference"],
        "category_ids": [1],
        "training_origin": "check",
        "runtime_packages": pipeline.runtime_versions(),
        "implementation_hashes": pipeline.source_hashes(wheat=True),
        "wheat": wheat_descriptor(context, 0),
    }
    pipeline.validate_profile(profile)
    output.mkdir(parents=True)
    context_path = output / "context.json"
    pipeline.write(context_path, context)
    common = ["{python}", "-m", "vision_rl.wheat_recipe", "stage", "--context", str(context_path)]
    setup = []
    for index, name in enumerate(STAGES):
        command = [*common, "--kind", name, "--output", "{setup_dir}"]
        for previous in STAGES[:index]:
            command.extend(["--" + previous.replace("_", "-"), "{setup:" + previous + "}/outputs"])
        setup.append(
            {
                "id": name,
                "command": command,
                "resume": [*command, "--resume"],
                "artifacts": ["{setup_dir}/outputs", "{setup_dir}/receipt.json"],
            }
        )
    candidate = [
        *common,
        "--output",
        "{candidate_dir}",
        "--comparison",
        "{setup:comparison}/outputs",
    ]
    plan = {
        "version": 1,
        "workspace": str(pipeline.PACKAGE.parents[1]),
        "evaluation_mode": "development",
        "inputs": [str(context_path), *map(str, paths), *map(str, source_paths())],
        "setup": setup,
        "budget": spec["budget"],
        "wheat_context": str(context_path),
        "selection_metric": "ADA",
        "validation_gate": {"ADA": {"min": 0}, "errors": {"max": 0}},
        "candidates": [
            {
                "id": "controlled-selection",
                "train": [*candidate, "--kind", "package"],
                "validate": [*candidate, "--kind", "validate"],
                "artifacts": ["{candidate_dir}/artifacts"],
                "validation_metrics": "{candidate_dir}/validation.json",
            }
        ],
    }
    loop.validate_plan(plan, output)
    pipeline.write(output / "plan.json", plan)
    return {
        "plan": str(output / "plan.json"),
        "integration_only": spec["integration_only"],
        "test_access": False,
        "stages": list(STAGES),
    }


def _manifest(context, target):
    annotation = Path(context["source"]) / "valid/_annotations.coco.json"
    document = read_report(annotation)
    pipeline.write(
        target,
        {
            "images": [
                {key: row[key] for key in ("id", "file_name", "width", "height")}
                for row in document["images"]
            ]
        },
    )
    return annotation


def checked_predictions(profile_path, manifest, image_root, directory):
    """Recheck a prior image-only invocation before using its saved detections."""
    from vision_rl.detection_inference import load_image_manifest

    profile, config, artifacts = pipeline.load_profile(profile_path)
    provenance = pipeline.read(directory / "provenance.json")
    images = load_image_manifest(manifest, image_root)
    expected_images = {str(row["path"]): loop._hash_path(row["path"]) for row in images}
    expected = {
        "profile_sha256": loop._hash_path(profile_path),
        "manifest_sha256": loop._hash_path(manifest),
        "image_hashes": expected_images,
        "configuration": json.loads(json.dumps(asdict(config))),
        "artifacts": {key: pipeline.descriptor(path) for key, path in artifacts.items()},
        "implementation_hashes": profile["implementation_hashes"],
        "runtime_packages": profile["runtime_packages"],
        "proposal_pipeline_sha256": pipeline.proposal_pipeline_sha256(profile),
        "predictions_sha256": loop._hash_path(directory / "predictions.coco.json"),
        "wheat": profile["wheat"],
        "label_access": False,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved wheat inference evidence differs from its frozen inputs")
    return read_report(directory / "predictions.coco.json")


def _publish_directory(destination, work_root, produce):
    """Expose complete substage evidence atomically; retain interrupted generations."""
    if destination.exists():
        return
    work_root.mkdir(parents=True, exist_ok=True)
    pending = work_root / (destination.name + "-" + uuid.uuid4().hex)
    produce(pending)
    # Same filesystem rename publishes the complete package/provenance together.
    # Interrupted work remains outside the artifact tree for diagnosis.
    pending.replace(destination)


def _publish_model(source, destination, work_root):
    if destination.exists():
        if loop._hash_path(destination) != loop._hash_path(source):
            raise ValueError("Recovered snapshot differs from its immutable initialization")
        return
    work_root.mkdir(parents=True, exist_ok=True)
    pending = work_root / (destination.name + "-" + uuid.uuid4().hex)
    shutil.copyfile(source, pending)
    if loop._hash_path(pending) != loop._hash_path(source):
        raise ValueError("Snapshot changed during numeric artifact copy")
    pending.replace(destination)


def _snapshot(context, args, output):
    from vision_rl.wheat_eval import calibrated_wheat_evaluation

    model = output / "model.safetensors"
    if context["initial_checkpoint"]:
        source = Path(context["initial_checkpoint"])
    else:
        from vision_rl.detection_training import DetectorTrainingConfig, train_detector

        # Native RF-DETR stages symlinked train/valid inputs. Keep mutable
        # optimizer state outside the public artifact tree frozen by the runner.
        training = args.output / "training"
        source = training / "snapshots" / f"epoch-{context['snapshot_epoch']:04d}.safetensors"
        if not source.exists():
            train_detector(
                Path(context["source"]),
                training,
                DetectorTrainingConfig(**context["initial_sft"]),
                resolution=context["policy"]["resolution"],
                selection_metric="wheat_ada",
                pause_after_epoch=context["snapshot_epoch"],
                resume=(training / "training-plan.json").exists(),
            )
        record = pipeline.read(source.with_suffix(".json"))
        if record["model"]["safetensors_sha256"] != loop._hash_path(source):
            raise ValueError("Fixed-epoch SFT snapshot differs from its durable record")
    work = args.output / "interrupted-work"
    _publish_model(source, model, work)
    package = output / "raw-pipeline"
    _publish_directory(
        package,
        work,
        lambda pending: pipeline.package_profile(
            model,
            context["inference"],
            pending,
            category_ids=[1],
            training_origin="fixed_early_supervised_snapshot",
            wheat=wheat_descriptor(context, 0),
        ),
    )
    manifest = output / "validation-images.json"
    annotation = _manifest(context, manifest)
    prediction = output / "image-validation"
    _publish_directory(
        prediction,
        work,
        lambda pending: pipeline.predict_profile(
            package / "pipeline.json", manifest, annotation.parent, pending
        ),
    )
    # The raw package fixes offset0. Calibration search belongs only to development.
    report, _ = calibrated_wheat_evaluation(
        annotation,
        checked_predictions(package / "pipeline.json", manifest, annotation.parent, prediction),
        model={"safetensors_sha256": loop._hash_path(model)},
    )
    pipeline.write(output / "evaluation.json", report)
    return {
        "status": "snapshot_ready",
        "training_performed": context["initial_checkpoint"] is None,
        "model_sha256": loop._hash_path(model),
        "ADA": report["metrics"]["detection_acc_avg_dom"],
    }


def _pilot(context, args, output):
    if context["integration_only"]:
        result = {
            "status": "integration_only_pilot_not_run",
            "go": False,
            "integration_only": True,
            "not_a_benchmark_experiment": True,
            "reason": "Exercise native optimizer plumbing only; no exploration gate claimed",
            "native_test_accessed": False,
            "validation_accessed": False,
        }
        pipeline.write(output / "summary.json", result)
        return result
    from vision_rl.detector_policy_training import PolicyTrainingConfig, pilot

    if (output / "summary.json").exists():
        return pipeline.read(output / "summary.json")
    # A pilot has no optimizer state. Preserve interrupted attempts in-place and
    # require a new immutable generation instead of silently rerolling its draws.
    return pilot(
        Path(context["source"]),
        args.snapshot / "model.safetensors",
        output,
        PolicyTrainingConfig(**context["policy"]),
        protocol=Path(context["protocol"]),
    )


def _arm(context, args, output):
    from vision_rl.detector_policy_training import PolicyTrainingConfig, train

    gate = pipeline.read(args.pilot / "summary.json")
    passing = [
        (float(key), row["mean_best_of_samples_gain"])
        for key, row in gate.get("scales", {}).items()
        if row["go"]
    ]
    if not context["integration_only"] and not passing:
        result = {
            "status": "no_go_action_space",
            "mode": args.kind,
            "optimizer_steps": 0,
            "pilot_summary_sha256": loop._hash_path(args.pilot / "summary.json"),
            "primary_requirement_satisfied": False,
            "native_test_accessed": False,
        }
        pipeline.write(output / "summary.json", result)
        return result
    config = PolicyTrainingConfig(**context["policy"])
    if not context["integration_only"]:
        config = replace(config, pixel_std=min(passing, key=lambda item: (-item[1], item[0]))[0])
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = pipeline.read(summary_path)
        if summary.get("status") == "completed":
            plan = pipeline.read(output / "plan.json")
            if (
                plan["configuration"] != asdict(config)
                or plan["mode"] != args.kind
                or plan["reference_sha256"] != loop._hash_path(args.snapshot / "model.safetensors")
                or plan["protocol_sha256"] != loop._hash_path(Path(context["protocol"]))
            ):
                raise ValueError("Completed training arm has another immutable identity")
            for key, name in (
                ("trained_evaluation_sha256", "trained-evaluation.json"),
                ("initial_evaluation_sha256", "initial-evaluation.json"),
                ("trained_predictions_sha256", "trained-predictions.coco.json"),
            ):
                if summary[key] != loop._hash_path(output / name):
                    raise ValueError("Completed training evidence changed")
            if summary["selections"]["best-trained"]["sha256"] != loop._hash_path(
                output / "best-trained.safetensors"
            ):
                raise ValueError("Completed native weights changed")
            return summary
    return train(
        Path(context["source"]),
        args.snapshot / "model.safetensors",
        output,
        config,
        mode=args.kind,
        pilot_path=args.pilot,
        control_run=None if args.kind == "rl" else args.rl,
        protocol=Path(context["protocol"]),
        resume=(output / "plan.json").exists(),
        integration_only=context["integration_only"],
    )


def _comparison(context, args, output):
    from vision_rl.detector_policy_training import compare_wheat_controls

    completed = pipeline.read(args.rl / "summary.json").get("status") == "completed"
    backend = context["inference"]["device"]
    initial = read_report(args.snapshot / "evaluation.json")
    if completed:
        # The snapshot's portable replay is CPU. Compare completed arms with
        # their shared initialization evaluated on the same training backend.
        plan = pipeline.read(args.rl / "plan.json")
        summary = pipeline.read(args.rl / "summary.json")
        reference = loop._hash_path(args.snapshot / "model.safetensors")
        backend = plan["configuration"]["device"]
        if (
            backend != context["policy"]["device"]
            or plan["reference_sha256"] != reference
            or summary["reference_sha256"] != reference
            or summary["initial_evaluation_sha256"]
            != loop._hash_path(args.rl / "initial-evaluation.json")
        ):
            raise ValueError("Native selection initialization or backend differs from its plan")
        initial = read_report(args.rl / "initial-evaluation.json")["wheat"]
    options = [
        {
            "origin": "initial_supervised",
            "model": str(args.snapshot / "model.safetensors"),
            "report": initial,
        }
    ]
    if completed:
        for mode in ("rl", "sft_updates", "sft_walltime"):
            path = getattr(args, mode)
            if pipeline.read(path / "plan.json")["configuration"]["device"] != backend:
                raise ValueError("Native selection arms require the same evaluation backend")
            report = read_report(path / "trained-evaluation.json")["wheat"]
            options.append(
                {"origin": mode, "model": str(path / "best-trained.safetensors"), "report": report}
            )
    if completed and not context["integration_only"]:
        comparison = compare_wheat_controls(
            args.rl,
            args.sft_updates,
            args.sft_walltime,
            Path(context["strongest_sft_report"]),
            output / "comparison.json",
        )
    else:
        comparison = {
            "primary_requirement_satisfied": False,
            "native_test_accessed": False,
            "integration_only": context["integration_only"],
            "status": "integration_only_no_benchmark_claim" if completed else "no_go_action_space",
        }
        pipeline.write(output / "comparison.json", comparison)
    selected = max(options, key=lambda row: row["report"]["metrics"]["detection_acc_avg_dom"])
    shutil.copyfile(Path(selected["model"]), output / "model.safetensors")
    pipeline.write(output / "evaluation.json", selected["report"])
    strongest_backend = None
    if context["strongest_sft_report"]:
        strongest = read_report(context["strongest_sft_report"])
        strongest_backend = strongest.get("evaluation_backend")
    # A report-only input cannot attest its inference backend. A user-supplied
    # device label is diagnostic metadata, not a verified evaluation receipt.
    # Canonical promotion needs the strongest checkpoint and immutable
    # inference plan/report/prediction provenance, which this adapter does not
    # yet accept. Preserve the numerical comparator without asserting that
    # this missing provenance has been established.
    numerical_requirement = comparison["primary_requirement_satisfied"]
    requirement_scope = (
        "The controlled comparison is provisional: a report-only strongest-SFT input cannot "
        "attest its backend even when a device is declared. Promotion requires the strongest "
        "checkpoint and immutable canonical inference plan/report/prediction provenance. "
        "Cross-backend deployment also needs canonical matched-control evaluation. Numerical "
        "replay tolerance never grants the substantial-gain requirement."
    )
    comparison.update(
        controlled_comparison_reported_requirement_satisfied=numerical_requirement,
        primary_requirement_satisfied=False,
        strongest_sft_backend_provenance_verified=False,
        claim_status="provisional_report_only_strongest_sft",
        requirement_scope=requirement_scope,
    )
    # Publish the qualification in the standalone comparison too. Consumers
    # must not need to discover selection.json to interpret a raw True flag.
    pipeline.write(output / "comparison.json", comparison)
    selection = {
        "origin": selected["origin"],
        "ADA": selected["report"]["metrics"]["detection_acc_avg_dom"],
        "foreground_logit_offset": selected["report"]["calibration"]["foreground_logit_offset"],
        "model_sha256": loop._hash_path(output / "model.safetensors"),
        "evaluation_backend": backend,
        "strongest_sft_evaluation_backend": strongest_backend,
        "strongest_sft_declared_backend_matches_selection": strongest_backend == backend,
        "strongest_sft_backend_provenance_verified": False,
        "controlled_comparison_reported_requirement_satisfied": numerical_requirement,
        "selection_backend_requirement_satisfied": False,
        "primary_requirement_satisfied": False,
        "requirement_scope": requirement_scope,
        "backend_parity_policy": context["backend_parity_policy"],
        "integration_only": context["integration_only"],
        "native_test_accessed": False,
        "arms": [
            {"origin": row["origin"], "ADA": row["report"]["metrics"]["detection_acc_avg_dom"]}
            for row in options
        ],
    }
    pipeline.write(output / "selection.json", selection)
    return selection


def _package(context, args):
    selected = pipeline.read(args.comparison / "selection.json")
    pipeline.package_profile(
        args.comparison / "model.safetensors",
        context["inference"],
        args.output / "artifacts",
        category_ids=[1],
        training_origin=selected["origin"],
        wheat=wheat_descriptor(context, selected["foreground_logit_offset"]),
    )
    for name in ("selection.json", "comparison.json", "evaluation.json"):
        shutil.copyfile(args.comparison / name, args.output / "artifacts" / name)
    return selected


def backend_parity(selection, replay_score, deployment_backend):
    """A bounded numerical replay diagnostic, never a statistical-gain tolerance."""
    policy = selection["backend_parity_policy"]
    if policy != BACKEND_PARITY_POLICY:
        raise ValueError("Backend parity policy differs from this frozen implementation")
    backend = selection["evaluation_backend"]
    if backend not in {"cpu", "mps", "cuda"} or deployment_backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unknown evaluation backend")
    same_backend = backend == deployment_backend
    pair = backend + "->" + deployment_backend
    if not same_backend and pair not in policy["cross_backend_absolute_ADA_tolerances"]:
        raise ValueError("No declared numerical replay tolerance for this backend pair")
    tolerance = (
        policy["same_backend_absolute_ADA_tolerance"]
        if same_backend
        else policy["cross_backend_absolute_ADA_tolerances"][pair]
    )
    original = loop._number(selection["ADA"], "selection ADA")
    actual = loop._number(replay_score, "deployment ADA")
    if not 0 <= original <= 1 or not 0 <= actual <= 1:
        raise ValueError("ADA must lie in [0,1]")
    delta = actual - original
    return {
        "selection_backend": backend,
        "deployment_backend": deployment_backend,
        "selection_ADA": original,
        "deployment_ADA": actual,
        "signed_ADA_delta": delta,
        "absolute_ADA_delta": abs(delta),
        "absolute_ADA_tolerance": tolerance,
        "same_backend": same_backend,
        "replay_within_declared_tolerance": abs(delta) <= tolerance,
        "permits_transferring_primary_requirement": same_backend,
        "purpose": "Numerical artifact replay only; no relaxation of gain or confidence gates",
    }


def _validate(context, args):
    from vision_rl.wheat_eval import evaluate_wheat

    manifest = args.output / "validation-images.json"
    annotation = _manifest(context, manifest)
    output = args.output / "image-validation"
    profile_path = args.output / "artifacts/pipeline.json"
    pipeline.predict_profile(profile_path, manifest, annotation.parent, output)
    report = evaluate_wheat(
        annotation,
        checked_predictions(profile_path, manifest, annotation.parent, output),
        output=output / "evaluation.json",
    )
    selected = pipeline.read(args.output / "artifacts/selection.json")
    parity = backend_parity(
        selected, report["metrics"]["detection_acc_avg_dom"], context["inference"]["device"]
    )
    pipeline.write(output / "backend-parity.json", parity)
    if not parity["replay_within_declared_tolerance"]:
        raise ValueError("Portable calibrated native weights failed independent ADA replay")
    metrics = {
        "ADA": report["metrics"]["detection_acc_avg_dom"],
        "n": report["n"],
        "domains": report["domains"],
        "errors": report["errors"],
        "primary_requirement_satisfied": (
            selected["primary_requirement_satisfied"] and parity["same_backend"]
        ),
        "controlled_comparison_reported_requirement_satisfied": selected[
            "controlled_comparison_reported_requirement_satisfied"
        ],
        "selection_backend_requirement_satisfied": selected[
            "selection_backend_requirement_satisfied"
        ],
        "requirement_scope": selected["requirement_scope"],
        "strongest_sft_backend_provenance_verified": selected[
            "strongest_sft_backend_provenance_verified"
        ],
        "backend_parity": parity,
        "integration_only": context["integration_only"],
        "test_access": False,
        "pipeline_sha256": loop._hash_path(profile_path),
    }
    pipeline.write(args.output / "validation.json", metrics)
    return metrics


def stage(args):
    context = pipeline.read(args.context)
    check_context(context)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind in ("package", "validate"):
        result = (_package if args.kind == "package" else _validate)(context, args)
    else:
        output = args.output / "outputs"
        receipt = args.output / "receipt.json"
        if receipt.exists():
            record = pipeline.read(receipt)
            if not args.resume or record["outputs_sha256"] != loop._hash_path(output):
                raise ValueError("Completed stage recovery requires intact immutable outputs")
            return record
        output.mkdir(exist_ok=True)
        function = {"snapshot": _snapshot, "pilot": _pilot, "comparison": _comparison}.get(
            args.kind, _arm
        )
        result = function(context, args, output)
        pipeline.write(
            receipt,
            {
                "stage": args.kind,
                "outputs_sha256": loop._hash_path(output),
                "status": result.get("status", "completed"),
                "test_access": False,
            },
        )
    check_context(context)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = sub.add_parser(name)
        for field in ("source", "recipe", "protocol", "contract"):
            command.add_argument("--" + field, type=Path, required=True)
        command.add_argument(
            "--output" if name == "prepare" else "--state-dir", type=Path, required=True
        )
        command.add_argument("--initial-checkpoint", type=Path)
        command.add_argument("--strongest-sft-report", type=Path)
        if name == "run":
            command.add_argument("--background", action="store_true")
    for name in ("resume", "status"):
        command = sub.add_parser(name)
        command.add_argument("--state-dir", type=Path, required=True)
        if name == "resume":
            command.add_argument("--background", action="store_true")
    worker = sub.add_parser("stage", help=argparse.SUPPRESS)
    worker.add_argument("--context", type=Path, required=True)
    worker.add_argument("--kind", choices=(*STAGES, "package", "validate"), required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--resume", action="store_true")
    for name in STAGES:
        worker.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args(argv)
    if args.command in ("resume", "status"):
        return loop.main(
            [args.command, "--state-dir", str(args.state_dir)]
            + (["--background"] if getattr(args, "background", False) else [])
        )
    if args.command == "stage":
        result = stage(args)
    else:
        output = (
            args.output
            if args.command == "prepare"
            else args.state_dir.with_name(args.state_dir.name + "-plan")
        )
        result = prepare(
            args.source,
            args.recipe,
            args.protocol,
            args.contract,
            output,
            initial_checkpoint=args.initial_checkpoint,
            strongest_sft_report=args.strongest_sft_report,
        )
        if args.command == "run":
            return loop.main(
                ["run", "--plan", result["plan"], "--state-dir", str(args.state_dir)]
                + (["--background"] if args.background else [])
            )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
