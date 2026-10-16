"""A bounded TRAIN-only Aircraft controlled experiment on the durable runner.

This is the small learning-mechanism study, not the full validation/test study.
Import one completed common warm start, train declared RL/SFT trajectories, and
retain equal-exposure and equal-committed-time endpoints. Every endpoint is
immutable; optimizer trajectories live separately under ``state/work``.

The trainer owns checkpoint, actual exposure and active-time accounting. Missing
trainer capabilities reject preparation rather than falling back to inferred
exposures or resetting budgets. No command in this module accepts held-out data.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from . import experiment_loop as loop

SCOPE = "train_controlled_diagnostic"
MAX_JSON_BYTES = 64 * 1024**2
PACKAGE = Path(__file__).resolve().parent


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Aircraft evidence must be a regular JSON file of at most64 MiB")
    return json.loads(path.read_text(), parse_constant=loop._bad_constant)


def trainer_api():
    from . import vlm_local_training as trainer

    required = {"stop_after_steps", "stop_after_active_seconds", "stage_id", "train_only"}
    if not required <= set(inspect.signature(trainer.train_local).parameters) or not callable(
        getattr(trainer, "read_endpoint_receipt", None)
    ):
        raise ValueError(
            "Trainer lacks the agreed staged-stop and checked endpoint receipt API; "
            "no model or training has been started"
        )
    return trainer


def _positive(value, name):
    result = loop._number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", value):
        raise ValueError("IDs must be lowercase safe names of1..40 characters")
    return value


def normalize_study(raw):
    """Require a separately activated diagnostic; never activate a draft protocol."""
    from . import vlm_local_training as trainer

    Profile = trainer.Profile

    fields = {
        "schema_version",
        "study_id",
        "activated",
        "scope",
        "device",
        "profile",
        "seed",
        "max_steps",
        "max_active_seconds",
        "stop_after_steps",
        "save_steps",
        "rl_learning_rate",
        "beta",
        "sft_controls",
        "assay_seeds",
        "budget",
    }
    if isinstance(raw, dict) and raw.get("schema_version") == 2:
        fields |= {"sft_max_steps", "sft_max_active_seconds"}
    if isinstance(raw, dict) and "initial_adapter_precision_transfer" in raw:
        fields.add("initial_adapter_precision_transfer")
    if not isinstance(raw, dict) or set(raw) != fields or raw.get("schema_version") not in {1, 2}:
        raise ValueError("Aircraft diagnostic requires exact versioned study fields")
    if raw["activated"] is not True or raw["scope"] != SCOPE:
        raise ValueError("A separately activated TRAIN-controlled diagnostic is required")
    result = json.loads(loop._canonical(raw))
    if result.get("initial_adapter_precision_transfer") is not None:
        raise ValueError("Precision transfer is not authorized in this controlled recipe")
    _identifier(result["study_id"])
    if result["device"] not in {"cpu", "mps"}:
        raise ValueError("Choose one explicit CPU or MPS backend; no fallback")
    if not isinstance(result["profile"], dict) or set(result["profile"]) != set(asdict(Profile())):
        raise ValueError("Freeze every profile field explicitly")
    profile = Profile(**result["profile"])
    profile.validate()
    for name in ("seed", "max_steps", "stop_after_steps", "save_steps"):
        if type(result[name]) is not int or result[name] < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a valid integer")
    if not result["stop_after_steps"] <= result["max_steps"] <= 400:
        raise ValueError("TRAIN diagnostic supports at most400 RL steps, never1200")
    _positive(result["max_active_seconds"], "max_active_seconds")
    if result["schema_version"] == 2:
        limit = getattr(trainer, "TRAIN_ONLY_STEP_CAPS", {"grpo": 400, "sft": 400})["sft"]
        if (
            type(result["sft_max_steps"]) is not int
            or result["sft_max_steps"] < result["stop_after_steps"]
            or limit is not None
            and result["sft_max_steps"] > limit
        ):
            raise ValueError(f"Frozen SFT horizon exceeds the current native capability ({limit})")
        if (
            _positive(result["sft_max_active_seconds"], "sft_max_active_seconds")
            < result["max_active_seconds"]
        ):
            raise ValueError("SFT original active-time budget must cover the RL original budget")
    if not 0 < _positive(result["rl_learning_rate"], "rl_learning_rate") <= 0.01:
        raise ValueError("Invalid RL learning rate")
    if not 0 < loop._number(result["beta"], "beta") <= 1:
        raise ValueError("This controlled study requires the frozen common-initial reference")
    controls = result["sft_controls"]
    if not isinstance(controls, list) or not 1 <= len(controls) <= 8:
        raise ValueError("Declare1..8 SFT controls prospectively")
    names = {"rl"}
    for control in controls:
        if not isinstance(control, dict) or set(control) != {
            "id",
            "learning_rate",
            "response_format",
        }:
            raise ValueError("Each SFT control requires id,learning_rate,response_format")
        name = _identifier(control["id"])
        if name in names:
            raise ValueError("Duplicate arm ID")
        names.add(name)
        if not 0 < _positive(control["learning_rate"], "SFT learning_rate") <= 0.01:
            raise ValueError("Invalid SFT learning rate")
        replace(profile, response_format=control["response_format"]).validate()
    if not any(
        c["response_format"] == profile.response_format
        and c["learning_rate"] == result["rl_learning_rate"]
        for c in controls
    ):
        raise ValueError("Declare the SFT control with the same prompt and learning rate as RL")
    seeds = result["assay_seeds"]
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 4
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Declare1..4 unique diagnostic sampling seeds")
    budget = result["budget"]
    if not isinstance(budget, dict) or set(budget) != {
        "max_wall_seconds",
        "command_timeout_seconds",
    }:
        raise ValueError("Freeze total wall and per-command timeout budgets")
    for name, value in budget.items():
        _positive(value, name)
    return result


def source_paths():
    # Prepare and run from the same isolated package snapshot for long studies.
    # Pin the complete Python closure, including helpers imported through vlm,
    # rather than leaving evaluation/spatial/training transitive imports unbound.
    return sorted(PACKAGE.rglob("*.py"))


def _source_hashes():
    return {str(path): loop._hash_path(path) for path in source_paths()}


def _check_context(context):
    if context["source_hashes"] != _source_hashes():
        raise ValueError("Frozen Aircraft implementation changed; preserve this study")
    for path, expected in context["input_hashes"].items():
        if loop._hash_path(Path(path)) != expected:
            raise ValueError(f"Frozen Aircraft input changed: {path}")


def _arms(study):
    return [
        {
            "id": "rl",
            "mode": "grpo",
            "learning_rate": study["rl_learning_rate"],
            "response_format": study["profile"]["response_format"],
        },
        *[{**control, "mode": "sft"} for control in study["sft_controls"]],
    ]


def _arm_budget(study, mode):
    """Separate original optimizer horizons; operational endpoints never alter these."""
    prefix = "sft_" if mode == "sft" and study["schema_version"] == 2 else ""
    return {name: study[prefix + name] for name in ("max_steps", "max_active_seconds")}


def prepare(train, assay, data_spec, initial_adapter, model_directory, study, output):
    """Freeze real inputs and compile into the existing ordered controller."""
    declaration = normalize_study(read(study))
    trainer = trainer_api()  # Fail before image/model access if the actual API is unavailable.
    paths = {
        name: Path(path).resolve()
        for name, path in {
            "train": train,
            "assay": assay,
            "data_spec": data_spec,
            "initial_adapter": initial_adapter,
            "model_directory": model_directory,
            "study": study,
        }.items()
    }
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Plan bundle must be new; existing studies resume their original plan")
    spec = trainer.load_spec(paths["data_spec"])
    rows = trainer.load_classification(
        paths["train"], "train", spec["labels"], preprocessing_id=spec.get("preprocessing_id")
    )
    sample_rows = trainer.load_classification(
        paths["assay"],
        "train",
        spec["labels"],
        preprocessing_id=spec.get("preprocessing_id"),
        require_all_classes=False,
    )
    if len(rows) > 400 or len(sample_rows) > 32:
        raise ValueError("Diagnostic permits at most400 TRAIN and32 assay images")
    by_id = {row["id"]: row for row in rows}
    if any(
        row["id"] not in by_id
        or any(row[key] != by_id[row["id"]][key] for key in ("label", "image_sha256"))
        for row in sample_rows
    ):
        raise ValueError("Assay must be an unchanged subset of these TRAIN images")
    parent = paths["initial_adapter"].parent / "local_training.json"
    warmup = read(parent)
    weights = paths["initial_adapter"] / "adapter_model.safetensors"
    if warmup.get("status") != "completed" or warmup.get("final_adapter_sha256") != loop._hash_path(
        weights
    ):
        raise ValueError("Common initialization must be the exact completed warmup export")
    if warmup["control_identity"]["data_spec_sha256"] != loop._hash_path(
        paths["data_spec"]
    ) or warmup["control_identity"]["train"] != trainer._classification_identity(
        paths["train"], rows
    ):
        raise ValueError("Common warmup must use this exact TRAIN manifest, pixels and task")
    frozen = [paths[name] for name in ("train", "assay", "data_spec", "study", "initial_adapter")]
    frozen += [
        parent,
        paths["model_directory"] / "model-source.json",
        paths["model_directory"] / ".pinned-download.json",
    ]
    model_source = read(paths["model_directory"] / "model-source.json")
    for item in model_source["files"]:
        path = paths["model_directory"] / item["path"]
        if not path.resolve().is_relative_to(paths["model_directory"]):
            raise ValueError("Model source entry escapes its pinned directory")
        frozen.append(path)
    frozen += [Path(row["image"]) for row in rows + sample_rows]
    context = {
        "schema_version": 1,
        "scope": SCOPE,
        "study": declaration,
        "paths": {name: str(path) for name, path in paths.items()},
        "labels": spec["labels"],
        "train_ids": [row["id"] for row in rows],
        "train_identity": trainer._classification_identity(paths["train"], rows),
        "assay_identity": trainer._classification_identity(paths["assay"], sample_rows),
        "assay_rows": [{"id": row["id"], "label": row["label"]} for row in sample_rows],
        "input_hashes": {str(path): loop._hash_path(path) for path in dict.fromkeys(frozen)},
        "source_hashes": _source_hashes(),
    }
    context_path = output / "context.json"
    setup = []

    def append(identifier, kind, arm=None, endpoint=None):
        command = [
            "{python}",
            "-m",
            "vision_rl.aircraft_recipe",
            "stage",
            "--context",
            str(context_path),
            "--kind",
            kind,
            "--output",
            "{setup_dir}",
            "--state-dir",
            "{state_dir}",
        ]
        if arm:
            command += ["--arm", arm]
        if endpoint:
            command += ["--endpoint", endpoint]
        setup.append(
            {
                "id": identifier,
                "command": command,
                "resume": command,
                "artifacts": ["{setup_dir}/published"],
                "max_attempts": 2,
            }
        )

    append("common", "common")
    append("initial-assay", "assay", "initial", "initial")
    for arm in _arms(declaration):
        append(arm["id"] + "-exposure", "train", arm["id"], "exposure")
        append(arm["id"] + "-exposure-assay", "assay", arm["id"], "exposure")
    for arm in declaration["sft_controls"]:
        append(arm["id"] + "-time", "train", arm["id"], "time")
        append(arm["id"] + "-time-assay", "assay", arm["id"], "time")
    append("comparison", "compare")
    candidate_command = [
        "{python}",
        "-m",
        "vision_rl.aircraft_recipe",
        "stage",
        "--context",
        str(context_path),
        "--state-dir",
        "{state_dir}",
        "--output",
        "{candidate_dir}",
    ]
    plan = {
        "version": 1,
        "workspace": str(PACKAGE.parents[1]),
        "evaluation_mode": "development",
        "inputs": [str(context_path), *context["input_hashes"], *context["source_hashes"]],
        "budget": {"max_attempts": 1, **declaration["budget"]},
        "setup": setup,
        "selection_metric": "evidence_complete",
        "validation_gate": {"evidence_complete": {"min": 1}},
        "candidates": [
            {
                "id": "train-evidence",
                "train": candidate_command + ["--kind", "package"],
                "validate": candidate_command + ["--kind", "validate"],
                "artifacts": ["{candidate_dir}/published"],
                "validation_metrics": "{candidate_dir}/metrics.json",
            }
        ],
    }
    loop.validate_plan(plan, output)
    output.mkdir(parents=True)
    loop._atomic_json(context_path, context)
    loop._atomic_json(output / "plan.json", plan)
    return output / "plan.json"


def _publish(output, build):
    destination = output / "published"
    if destination.exists():
        return destination
    temporary = output / ("incomplete-" + uuid.uuid4().hex)
    temporary.mkdir()
    build(temporary)
    temporary.rename(destination)
    return destination


def _endpoint(state, arm, endpoint):
    return state / "setup" / (arm + "-" + endpoint) / "published"


def _common(context, output):
    source = Path(context["paths"]["initial_adapter"])
    record = read(source.parent / "local_training.json")

    def build(temporary):
        shutil.copytree(source, temporary / "final_adapter")
        if (
            loop._hash_path(temporary / "final_adapter" / "adapter_model.safetensors")
            != record["final_adapter_sha256"]
        ):
            raise ValueError("Common warmup changed while publishing its immutable snapshot")
        shutil.copy2(source.parent / "local_training.json", temporary / "local_training.json")
        archived_source = source.parent / "adapter-source.py"
        if archived_source.is_file():
            shutil.copy2(archived_source, temporary / "warmup-source.py")
        loop._atomic_json(
            temporary / "receipt.json",
            {
                "role": "common_initialization",
                "training_performed_by_recipe": False,
                "adapter_sha256": loop._hash_path(temporary / "final_adapter"),
                "weights_sha256": record["final_adapter_sha256"],
                "warmup_exposure_accounting": (
                    "configuration-derived; no retroactive recorded trace"
                ),
                "warmup_configured_steps": record["settings"]["max_steps"],
                "warmup_active_time": record.get("optimizer_audit", {}).get("active_time"),
                "source_record_sha256": loop._hash_path(source.parent / "local_training.json"),
            },
        )

    return _publish(output, build)


def _checked_receipt(trainer, work, context, arm, stage_id):
    receipt = trainer.read_endpoint_receipt(work, expected_stage_id=stage_id)
    study = context["study"]
    if receipt["status"] not in {"boundary_reached", "total_budget_exhausted"}:
        raise ValueError("Trainer did not publish a completed controlled endpoint")
    if receipt["original_budgets"] != _arm_budget(study, arm["mode"]):
        raise ValueError("Trainer receipt changed original budgets")
    ids = receipt["committed_prompt_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) != receipt["global_step"]
        or (
            receipt["committed_prompt_presentations"] != len(ids)
            or not set(ids) <= set(context["train_ids"])
        )
    ):
        raise ValueError("Receipt does not contain the actual one-prompt/update TRAIN lineage")
    if receipt["source_sha256"] != context["source_hashes"][str(PACKAGE / "vlm_local_training.py")]:
        raise ValueError("Trainer receipt belongs to another implementation")
    count = len(context["train_ids"])
    for start in range(0, len(ids) - count + 1, count):
        if set(ids[start : start + count]) != set(context["train_ids"]):
            raise ValueError("A completed TRAIN exposure pass omitted or repeated frozen IDs")
    record = read(work / "local_training.json")
    initial = read(Path(context["paths"]["initial_adapter"]).parent / "local_training.json")
    if record["settings"]["init_adapter"]["weights_sha256"] != initial["final_adapter_sha256"]:
        raise ValueError("Arm did not start from the common warmup")
    if (
        record["settings"]["mode"] != arm["mode"]
        or record["settings"]["learning_rate"] != arm["learning_rate"]
        or record["settings"]["device"] != study["device"]
        or record["settings"]["seed"] != study["seed"]
        or record["control_identity"]["profile"]
        != {**study["profile"], "response_format": arm["response_format"]}
        or record["control_identity"]["train"] != context["train_identity"]
        or record["control_identity"]["data_spec_sha256"]
        != context["input_hashes"][context["paths"]["data_spec"]]
        or record["settings"].get("beta") != (study["beta"] if arm["mode"] == "grpo" else None)
        or record["settings"].get("initial_adapter_precision_transfer") is not None
    ):
        raise ValueError("Arm settings do not match its prospective declaration")
    if arm["mode"] == "grpo" and (
        record["reference"]["initial_adapter_weights_sha256"] != initial["final_adapter_sha256"]
        or not record["optimizer_audit"]["reference_unchanged"]
    ):
        raise ValueError("RL reference must remain the exact common warmup")
    return receipt


def _resume_checkpoint(work):
    """Select a published checkpoint; the trainer verifies its entire state before load."""
    checkpoints = []
    for path in work.glob("checkpoint-*"):
        if re.fullmatch(r"checkpoint-[0-9]+", path.name) and path.is_dir():
            if (path / "active_time.json").is_file() and (path / "exposure.json").is_file():
                checkpoints.append((int(path.name.split("-")[-1]), path))
    if not checkpoints:
        raise ValueError("Interrupted arm has no durable time/exposure checkpoint; do not restart")
    return max(checkpoints)[1]


def _unmet_time_control(output, receipt, target):
    """Retain a capped control as incomplete, never publish it as time-matched."""
    loop._atomic_json(
        output / "unmet-time-control.json",
        {
            "status": "equal_time_target_not_reached",
            "equal_time_control_complete": False,
            "target_committed_active_seconds": target,
            "actual_committed_active_seconds": receipt["committed_active_seconds"],
            "shortfall_seconds": target - receipt["committed_active_seconds"],
            "global_step": receipt["global_step"],
            "original_budgets": receipt["original_budgets"],
            "trainer_receipt_sha256": loop._digest(receipt),
            "primary_requirement_satisfied": False,
            "scope": SCOPE,
        },
    )
    raise ValueError(
        "SFT exhausted its original budget before the RL committed-time target; "
        "the equal-time control is incomplete, never extend this trajectory's budget"
    )


def _train(context, state, output, arm_id, endpoint):
    trainer = trainer_api()
    study = context["study"]
    arm = next((a for a in _arms(study) if a["id"] == arm_id), None)
    if arm is None or endpoint not in {"exposure", "time"} or endpoint == "time" and arm_id == "rl":
        raise ValueError("Unknown controlled arm/endpoint")
    budget = _arm_budget(study, arm["mode"])
    stage_id = arm_id + "-" + endpoint
    work = state / "work" / arm_id
    work.parent.mkdir(exist_ok=True)
    steps = study["stop_after_steps"] if endpoint == "exposure" else None
    target = None
    prior = None
    if endpoint == "time":
        target = read(_endpoint(state, "rl", "exposure") / "trainer-receipt.json")[
            "committed_active_seconds"
        ]
        if not 0 < target <= budget["max_active_seconds"]:
            raise ValueError("RL matched-time target exceeds the original SFT active-time budget")
        prior = read(_endpoint(state, arm_id, "exposure") / "trainer-receipt.json")
        if (
            prior["committed_active_seconds"] < target
            and prior["global_step"] >= budget["max_steps"]
        ):
            _unmet_time_control(output, prior, target)
    training_performed = False
    finalization_performed = False
    if not (output / "published").exists():
        receipt_path = work / "endpoint_receipt.json"
        existing = read(receipt_path) if receipt_path.exists() else None
        already = existing is not None and existing.get("stage_id") == stage_id
        no_work = prior is not None and prior["committed_active_seconds"] >= target
        if no_work:
            receipt = _checked_receipt(trainer, work, context, arm, arm_id + "-exposure")
        else:
            if not already:
                resume = _resume_checkpoint(work) if work.exists() else None
                if resume is not None:
                    active = read(resume / "active_time.json")
                    reached = (steps is not None and active["global_step"] >= steps) or (
                        target is not None and active.get("committed_active_seconds", 0) >= target
                    )
                    exhausted = active["global_step"] >= budget["max_steps"] or (
                        active.get("committed_active_seconds", 0) >= budget["max_active_seconds"]
                    )
                    finalization_performed = reached or exhausted
                    if (
                        finalization_performed
                        and "finalize_only" not in inspect.signature(trainer.train_local).parameters
                    ):
                        raise ValueError(
                            "Boundary checkpoint exists without its completed endpoint receipt; "
                            "zero-update trainer endpoint finalization is required, never retrain"
                        )
                trainer.train_local(
                    Path(context["paths"]["train"]),
                    None,
                    Path(context["paths"]["data_spec"]),
                    work,
                    mode=arm["mode"],
                    model_directory=Path(context["paths"]["model_directory"]),
                    device=study["device"],
                    profile=trainer.Profile(
                        **{**study["profile"], "response_format": arm["response_format"]}
                    ),
                    **budget,
                    learning_rate=arm["learning_rate"],
                    beta=study["beta"],
                    seed=study["seed"],
                    save_steps=study["save_steps"],
                    train_only=True,
                    init_adapter=state / "setup" / "common" / "published" / "final_adapter",
                    resume_from_checkpoint=resume,
                    stop_after_steps=steps,
                    stop_after_active_seconds=target,
                    stage_id=stage_id,
                    **(
                        {"initial_adapter_precision_transfer": None}
                        if "initial_adapter_precision_transfer" in study
                        else {}
                    ),
                    **({"finalize_only": True} if finalization_performed else {}),
                )
                training_performed = not finalization_performed
            receipt = _checked_receipt(trainer, work, context, arm, stage_id)
        if not no_work and receipt["requested_boundary"] != {
            "stop_after_steps": steps,
            "stop_after_active_seconds": target,
        }:
            raise ValueError("Trainer receipt changed the frozen operational stop target")
        if steps is not None and receipt["global_step"] != steps:
            raise ValueError("Equal-exposure endpoint has an unexpected optimizer-step count")
        if target is not None and receipt["committed_active_seconds"] < target:
            _unmet_time_control(output, receipt, target)

        def build(temporary):
            shutil.copytree(work / "final_adapter", temporary / "final_adapter")
            for name, key in (
                ("adapter_model.safetensors", "weights_sha256"),
                ("adapter_config.json", "config_sha256"),
            ):
                if (
                    loop._hash_path(temporary / "final_adapter" / name)
                    != receipt["final_adapter"][key]
                ):
                    raise ValueError("Endpoint changed while copying its immutable artifact")
            shutil.copy2(work / "local_training.json", temporary / "local_training.json")
            loop._atomic_json(temporary / "trainer-receipt.json", receipt)
            loop._atomic_json(
                temporary / "receipt.json",
                {
                    "stage_id": stage_id,
                    "training_performed_in_publishing_invocation": training_performed,
                    "zero_update_finalization_performed": finalization_performed,
                    "trainer_receipt_sha256": loop._digest(receipt),
                    "equal_time_target_seconds": target,
                    "time_target_already_exceeded_at_equal_exposure": bool(no_work),
                    "active_time_overshoot_seconds": receipt["committed_active_seconds"] - target
                    if target
                    else None,
                    "adapter_sha256": loop._hash_path(temporary / "final_adapter"),
                    "scope": SCOPE,
                    "held_out_improvement_measured": False,
                },
            )

        return _publish(output, build)
    return output / "published"


def _release_inference_model(device):
    """Release native model cycles and cached Metal storage before another load."""
    gc.collect()
    if device == "mps":
        from importlib import import_module

        torch = import_module("torch")
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _assay(context, state, output, arm_id, endpoint):
    trainer = trainer_api()
    study = context["study"]
    arms = _arms(study)
    if arm_id == "initial":
        adapter = state / "setup" / "common" / "published" / "final_adapter"
        formats = sorted({arm["response_format"] for arm in arms})
    else:
        arm = next((arm for arm in arms if arm["id"] == arm_id), None)
        if arm is None:
            raise ValueError("Unknown assay arm")
        adapter = _endpoint(state, arm_id, endpoint) / "final_adapter"
        formats = [arm["response_format"]]
    work = output / "work"
    work.mkdir(exist_ok=True)
    records = {}
    for response_format in formats:
        for seed in [None, *study["assay_seeds"]]:
            name = response_format + ("-greedy" if seed is None else f"-sample-{seed}")
            destination = work / name
            metadata_name = (
                "train_subset_inference.json" if seed is None else "train_subset_groups.json"
            )
            if destination.exists():
                record = read(destination / metadata_name)
                if record.get("status") != "completed":
                    raise ValueError(
                        "Partial assay retained; never automatically resample its outcomes"
                    )
            else:
                function = (
                    trainer.predict_train_subset if seed is None else trainer.sample_train_groups
                )
                try:
                    record = function(
                        Path(context["paths"]["assay"]),
                        data_spec=Path(context["paths"]["data_spec"]),
                        output=destination,
                        model_directory=Path(context["paths"]["model_directory"]),
                        device=study["device"],
                        profile=trainer.Profile(
                            **{**study["profile"], "response_format": response_format}
                        ),
                        init_adapter=adapter,
                        **({"seed": seed} if seed is not None else {}),
                    )
                finally:
                    _release_inference_model(study["device"])
            if record["initial_adapter"]["weights_sha256"] != loop._hash_path(
                adapter / "adapter_model.safetensors"
            ):
                raise ValueError("Assay did not use the exact immutable endpoint adapter")
            if (
                record["status"] != "completed"
                or record["profile"] != {**study["profile"], "response_format": response_format}
                or record["data_spec_sha256"]
                != context["input_hashes"][context["paths"]["data_spec"]]
                or record["data_identity"] != context["assay_identity"]
                or record["generation"]["seed"] != seed
                or record["generation"]["do_sample"] != (seed is not None)
                or record["environment"]["adapter_source_sha256"]
                != context["source_hashes"][str(PACKAGE / "vlm_local_training.py")]
                or record["environment"]["device"] != study["device"]
            ):
                raise ValueError("Assay identity differs from the frozen endpoint protocol")
            raw_path = destination / (
                "predictions.json" if seed is None else "sampled_completions.json"
            )
            raw = read(raw_path)
            if set(raw) != {row["id"] for row in context["assay_rows"]} or (
                seed is not None
                and any(
                    len(samples) != study["profile"]["num_generations"] for samples in raw.values()
                )
            ):
                raise ValueError("Assay did not complete the exact frozen image/group denominator")
            records[name] = record

    def build(temporary):
        shutil.copytree(work, temporary / "assays")
        loop._atomic_json(
            temporary / "receipt.json",
            {
                "scope": SCOPE,
                "arm": arm_id,
                "endpoint": endpoint,
                "adapter_sha256": loop._hash_path(adapter),
                "records": records,
                "training_performed": False,
                "held_out_improvement_measured": False,
            },
        )

    return _publish(output, build)


def _canonical_predictions(context, path, response_format):
    from .vlm_local_training import parse_label

    # Both prompt styles use the same task grammar, as in the native reward and
    # live support assay. Think-block frequency is diagnostic, not validity.
    raw = read(path)
    expected = {row["id"] for row in context["assay_rows"]}
    if set(raw) != expected:
        raise ValueError("Greedy assay lacks the exact frozen TRAIN denominator")
    return {
        identifier: parse_label(
            value["response"],
            context["labels"],
            allow_reasoning=True,
        )
        if value["terminated"]
        else None
        for identifier, value in raw.items()
    }


def _sample_comparison(context, endpoint, initial, response_format):
    """Same-seed TRAIN assay, retaining invalid outcomes in the full denominator."""
    from .vlm_local_training import parse_label

    correct = initial_correct = valid_pairs = valid_pair_delta = total = 0
    useful_groups = 0
    for seed in context["study"]["assay_seeds"]:
        name = response_format + f"-sample-{seed}"
        a = read(endpoint / name / "sampled_completions.json")
        b = read(initial / name / "sampled_completions.json")
        for row in context["assay_rows"]:
            outcomes = []
            for current, previous in zip(a[row["id"]], b[row["id"]], strict=True):

                def label(value):
                    return (
                        parse_label(
                            value["response"],
                            context["labels"],
                            allow_reasoning=True,
                        )
                        if value["terminated"]
                        else None
                    )

                current_label, previous_label = label(current), label(previous)
                x, y = int(current_label == row["label"]), int(previous_label == row["label"])
                correct += x
                initial_correct += y
                total += 1
                if current_label is not None and previous_label is not None:
                    valid_pairs += 1
                    valid_pair_delta += x - y
                outcomes.append(current_label)
            useful_groups += int(
                row["label"] in outcomes
                and any(value is not None and value != row["label"] for value in outcomes)
            )
    return {
        "completions": total,
        "exact_accuracy": correct / total,
        "initial_exact_accuracy": initial_correct / total,
        "sampled_exact_accuracy_delta": (correct - initial_correct) / total,
        "both_valid_fraction": valid_pairs / total,
        "valid_pair_correctness_delta": valid_pair_delta / valid_pairs if valid_pairs else None,
        "useful_mixed_groups": useful_groups,
        "scope": "Fixed TRAIN sampled mechanism diagnostic; not greedy pass@1 or held-out evidence",
    }


def _compare(context, state, output):
    from .classification_comparison import compare_classification

    study = context["study"]
    results, endpoints, sampled, canonical = {}, {}, {}, {}
    initial_root = state / "setup" / "initial-assay" / "published" / "assays"
    for arm in _arms(study):
        for endpoint in ["exposure"] if arm["id"] == "rl" else ["exposure", "time"]:
            key = arm["id"] + "-" + endpoint
            assay_root = state / "setup" / (key + "-assay") / "published" / "assays"
            name = arm["response_format"] + "-greedy"
            prediction = _canonical_predictions(
                context, assay_root / name / "predictions.json", arm["response_format"]
            )
            initial = _canonical_predictions(
                context, initial_root / name / "predictions.json", arm["response_format"]
            )
            results[key] = compare_classification(
                context["assay_rows"],
                prediction,
                initial,
                context["labels"],
                require_all_classes=False,
            )
            canonical[key] = prediction
            sampled[key] = _sample_comparison(
                context, assay_root, initial_root, arm["response_format"]
            )
            endpoints[key] = read(_endpoint(state, arm["id"], endpoint) / "trainer-receipt.json")
    latest = {
        arm["id"]: endpoints[arm["id"] + ("-exposure" if arm["id"] == "rl" else "-time")]
        for arm in _arms(study)
    }
    assay_receipts = [
        read(path / "published" / "receipt.json")
        for path in sorted((state / "setup").glob("*-assay"))
    ]
    controller_state = read(state / "state.json") if (state / "state.json").exists() else None
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "split": "train",
        "assay_images": len(context["assay_rows"]),
        "comparisons_vs_common_initialization": results,
        "rl_vs_each_declared_control": {
            name: compare_classification(
                context["assay_rows"],
                canonical["rl-exposure"],
                predictions,
                context["labels"],
                require_all_classes=False,
            )
            for name, predictions in canonical.items()
            if name != "rl-exposure"
        },
        "sampled_train_comparisons_vs_common_initialization": sampled,
        "endpoint_receipts": endpoints,
        "per_arm_final_committed_active_seconds": {
            name: r["committed_active_seconds"] for name, r in latest.items()
        },
        "sft_search_total_committed_active_seconds": sum(
            r["committed_active_seconds"] for name, r in latest.items() if name != "rl"
        ),
        "per_arm_physical_work": {name: r["physical_work"] for name, r in latest.items()},
        "assay_reported_wall_seconds": sum(
            record["elapsed_seconds"]
            for receipt in assay_receipts
            for record in receipt["records"].values()
        ),
        "assay_time_scope": (
            "Trainer-reported inference elapsed time; full child setup/IO is also retained "
            "in the controller command journal"
        ),
        "controller_wall_seconds_at_comparison": time.time() - controller_state["created_at"]
        if controller_state
        else None,
        "warmup": read(state / "setup" / "common" / "published" / "receipt.json"),
        "wall_time_scope": (
            "Controller state.created_at and append-only command durations; never reset by resume"
        ),
        "primary_requirement_satisfied": False,
        "held_out_improvement_measured": False,
        "test_consumed": False,
        "claim": "TRAIN mechanism evidence only, not a benchmark or frontier win",
    }
    return _publish(
        output, lambda temporary: loop._atomic_json(temporary / "comparison.json", report)
    )


def stage(context_path, state, output, kind, arm=None, endpoint=None):
    context = read(context_path)
    _check_context(context)
    state, output = Path(state).resolve(), Path(output).resolve()
    if not output.is_relative_to(state):
        raise ValueError("Stage output must be inside its controller state")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if kind == "common":
        result = _common(context, output)
    elif kind == "train":
        result = _train(context, state, output, arm, endpoint)
    elif kind == "assay":
        result = _assay(context, state, output, arm, endpoint)
    elif kind == "compare":
        result = _compare(context, state, output)
    elif kind == "package":
        source = state / "setup" / "comparison" / "published" / "comparison.json"
        result = _publish(
            output, lambda temporary: shutil.copy2(source, temporary / "comparison.json")
        )
    elif kind == "validate":
        report = read(output / "published" / "comparison.json")
        if (
            report["primary_requirement_satisfied"]
            or report["test_consumed"]
            or report["split"] != "train"
        ):
            raise ValueError("TRAIN diagnostic cannot publish a held-out achievement")
        loop._atomic_json(
            output / "metrics.json",
            {
                "evidence_complete": 1,
                "primary_requirement_satisfied": 0,
                "controller_wall_seconds_at_validation": time.time()
                - read(state / "state.json")["created_at"],
            },
        )
        result = output / "metrics.json"
    else:
        raise ValueError("Unknown Aircraft stage")
    _check_context(context)
    return {"output": str(result), "seconds": time.monotonic() - started, "scope": SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in (
        "train",
        "assay",
        "data-spec",
        "initial-adapter",
        "model-directory",
        "study",
        "output",
    ):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    stage_parser = commands.add_parser("stage")
    for name in ("context", "state-dir", "output"):
        stage_parser.add_argument("--" + name, type=Path, required=True)
    stage_parser.add_argument(
        "--kind",
        choices=("common", "train", "assay", "compare", "package", "validate"),
        required=True,
    )
    stage_parser.add_argument("--arm")
    stage_parser.add_argument("--endpoint")
    for name in ("run", "resume", "status"):
        command = commands.add_parser(name)
        command.add_argument("--state-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--plan", type=Path, required=True)
        if name != "status":
            command.add_argument("--background", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command in {"run", "resume", "status"}:
            forwarded = [command, "--state-dir", str(args["state_dir"])]
            if command == "run":
                forwarded += ["--plan", str(args["plan"])]
            if args.get("background"):
                forwarded += ["--background"]
            return loop.main(forwarded)
        if command == "prepare":
            result = {"plan": str(prepare(**args)), "scope": SCOPE}
        else:
            args["context_path"] = args.pop("context")
            args["state"] = args.pop("state_dir")
            result = stage(**args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"aircraft recipe: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
