"""Run real cached-query RL experiments, then explicitly freeze one native test.

Development commands never open test data or make hosted requests. The generic
controller runs candidate subprocesses persistently and stops at
``awaiting_final_evaluation``. Final evaluation is a separate explicit command;
it consumes the selected model once, using a frozen native-test protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import runpy
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from vision_rl import experiment_loop as loop

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_REGISTRATION = ROOT / "docs/detection-cluster-preregistration.json"
DEFAULT_REQUIRED_PROVIDERS = ("gemini", "anthropic", "openai")
DEFAULT_CRITERIA = {
    "minimum_specialist_ap": 0.8,
    "minimum_margin_over_each_frontier": 0.2,
    "minimum_cluster_bootstrap_lower95_margin": 0.1,
    "bootstrap_samples": 2000,
    "bootstrap_seed": 42,
    "allowed_primary_errors": 0,
}


def wrapper():
    return runpy.run_path(str(ROOT / "scripts/pills_frontier.py"))


def read(path):
    return loop._read_json(Path(path))


def read_result(path):
    """Bound native COCO prediction/matching reports separately from small plans.

    Full 423-image, 128-query validation matching arrays exceed 40 MiB with
    indentation; the native 500-detection cap needs more headroom. Keep the
    generic plan/configuration limit at 10 MiB and cap these results at 256 MiB.
    """
    limit = 256 * 1024 * 1024
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError(f"Native result JSON exceeds 256 MiB: {path}")
    with path.open("rb") as handle:
        contents = handle.read(limit + 1)
    if len(contents) > limit:
        raise ValueError(f"Native result JSON exceeds 256 MiB: {path}")
    return json.loads(contents, parse_constant=loop._bad_constant)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    loop._atomic_json(path, value)


def pinned(paths):
    return {str(Path(path).resolve()): loop._hash_path(Path(path)) for path in paths}


def check_hashes(hashes):
    for name, expected in hashes.items():
        if loop._hash_path(Path(name)) != expected:
            raise ValueError(f"Pinned experiment input changed: {name}")


def implementation_paths():
    paths = [Path(__file__), ROOT / "src/vision_rl/experiment_loop.py"]
    paths.extend(
        ROOT / "src/vision_rl" / name
        for name in (
            "detection_rl.py",
            "detection_rewards.py",
            "detection_eval.py",
            "detection_training.py",
        )
    )
    return paths


def prepare_plan(args):
    from vision_rl.detection_rl import RefinementConfig, assert_isolated, load_cache

    output = args.output.resolve()
    if output.exists():
        raise ValueError("Plan bundles are immutable; choose a new output directory")
    train_path, val_path, detector = (
        path.resolve()
        for path in (
            args.train_cache,
            args.validation_cache,
            args.detector,
        )
    )
    train = load_cache(train_path, expected_split="train")
    validation = load_cache(val_path, expected_split="validation")
    assert_isolated(train, validation)
    model_hash = loop._hash_path(detector)
    if detector.suffix != ".safetensors" or model_hash != train["metadata"]["model_sha256"]:
        raise ValueError("Detector safetensors must match the immutable query caches")
    for cache in (train, validation):
        if cache["embeddings"].shape[-1] != 256 or cache["boxes"].shape[1] != 128:
            raise ValueError("This pipeline requires 256-dimensional embeddings and 128 queries")
    train_annotation = Path(train["metadata"]["annotation_path"]).resolve()
    val_annotation = Path(validation["metadata"]["annotation_path"]).resolve()
    if train_annotation.parent.name != "train" or val_annotation.parent.name != "valid":
        raise ValueError("Development cache annotations must be native train/valid paths")
    if not 0 <= args.minimum_validation_ap <= 1:
        raise ValueError("minimum validation AP must be in [0, 1]")
    candidates = read(args.candidates)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidates must be a nonempty JSON list of {id, config}")
    ids, configs = set(), []
    for candidate in candidates:
        identifier = candidate.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier) or identifier in ids:
            raise ValueError("Candidate IDs must be unique safe filenames")
        ids.add(identifier)
        config = RefinementConfig(**candidate.get("config", {}))
        config.validate()
        if config.device != "cpu" or config.cpu_threads != 1:
            raise ValueError("Development refinement uses CPU with exactly one thread")
        if (config.hidden, config.geometry_step, config.identity_probability) != (128, 0.03, 0.75):
            raise ValueError("Candidate must use the fixed safe refinement architecture")
        entry = {"id": identifier, "config": asdict(config)}
        if candidate.get("resume_training") is not None:
            previous = Path(candidate["resume_training"]).resolve()
            if not previous.is_dir() or not (previous / "last.pt").is_file():
                raise ValueError("resume_training must contain a durable last.pt checkpoint")
            loop._hash_path(previous)  # Reject symlinks before copying recovery evidence.
            entry["resume_training"] = str(previous)
        configs.append(entry)
    inputs = [
        detector,
        train_path,
        val_path,
        train_annotation,
        val_annotation,
        CLUSTER_REGISTRATION,
        *implementation_paths(),
        *[
            Path(candidate["resume_training"])
            for candidate in configs
            if "resume_training" in candidate
        ],
    ]
    context = {
        "schema_version": 1,
        "benchmark": "RF100-VL/countingpills-exf0r-gfkt/v2",
        "detector": str(detector),
        "train_cache": str(train_path),
        "validation_cache": str(val_path),
        "train_annotation": str(train_annotation),
        "validation_annotation": str(val_annotation),
        "input_hashes": pinned(inputs),
        "train_images": len(train["ids"]),
        "validation_images": len(validation["ids"]),
        "truncated_development": bool(
            train["metadata"].get("truncated_images")
            or validation["metadata"].get("truncated_images")
        ),
        "final_criteria": DEFAULT_CRITERIA,
        "selection": "Highest independently replayed validation AP across all candidates",
        "test_access": False,
    }
    output.mkdir(parents=True)
    write(output / "context.json", context)
    entries = []
    for candidate in configs:
        specification = output / f"{candidate['id']}.json"
        write(specification, candidate)
        common = [
            "{python}",
            str(Path(__file__).resolve()),
            "stage",
            "--context",
            str(output / "context.json"),
            "--candidate",
            str(specification),
            "--output",
            "{candidate_dir}",
        ]
        entries.append(
            {
                "id": candidate["id"],
                "train": [*common, "--phase", "train"],
                "validate": [*common, "--phase", "validate"],
                "artifacts": ["{candidate_dir}/artifacts"],
                "validation_metrics": "{candidate_dir}/validation.json",
            }
        )
        inputs.append(specification)
    inputs.append(output / "context.json")
    plan = {
        "version": 1,
        "workspace": str(ROOT),
        "evaluation_mode": "development",
        "inputs": [str(path) for path in inputs],
        "budget": {
            "max_attempts": args.max_attempts or len(entries) * 2,
            "max_wall_seconds": args.max_wall_seconds,
            "command_timeout_seconds": args.command_timeout_seconds,
        },
        "selection_metric": "AP",
        "validation_gate": {"AP": {"min": args.minimum_validation_ap}, "errors": {"max": 0}},
        "candidates": entries,
        "pills_context": str(output / "context.json"),
    }
    loop.validate_plan(plan, output)
    write(output / "plan.json", plan)
    return {
        "plan": str(output / "plan.json"),
        "candidates": len(entries),
        **{key: context[key] for key in ("train_images", "validation_images", "test_access")},
    }


def stage(args):
    import torch

    from vision_rl.detection_eval import evaluate_coco
    from vision_rl.detection_rl import (
        RefinementConfig,
        coco_predictions,
        export_refinement_safetensors,
        load_cache,
        predict_refinement_safetensors,
        train_refinement,
    )

    context, candidate = read(args.context), read(args.candidate)
    check_hashes(context["input_hashes"])
    torch.set_num_threads(1)
    config = RefinementConfig(**candidate["config"])
    if config.device != "cpu" or config.cpu_threads != 1:
        raise ValueError("Loop stages cannot launch MPS or GPU work")
    output = args.output.resolve()
    artifacts, training = output / "artifacts", output / "training"
    if args.phase == "train":
        if artifacts.exists():
            raise ValueError("Candidate artifact output already exists")
        previous = candidate.get("resume_training")
        if previous:
            shutil.copytree(previous, training)
        summary = train_refinement(
            Path(context["train_cache"]),
            Path(context["validation_cache"]),
            training,
            config,
            resume=bool(previous),
        )
        artifacts.mkdir()
        for source, target in (
            ("best.pt", "policy.safetensors"),
            ("best-rl.pt", "rl-policy.safetensors"),
            ("warmup.pt", "warmup-policy.safetensors"),
        ):
            export_refinement_safetensors(training / source, artifacts / target)
        history = read(training / "history.json")
        selected = [row for row in history if row["selected"]]
        origin = selected[-1]["phase"] if selected else "detector_identity"
        write(
            artifacts / "selection.json",
            {
                "candidate": candidate["id"],
                "selected_origin": origin,
                "selected_epoch": selected[-1]["epoch"] if selected else 0,
                "config": asdict(config),
                "training_summary": summary,
                "resume_training": previous,
                "artifact_hashes": pinned(sorted(artifacts.glob("*.safetensors"))),
                "test_used_for_selection": False,
            },
        )
        check_hashes(context["input_hashes"])
        return {
            "candidate": candidate["id"],
            "selected_origin": origin,
            "best_validation_AP": summary["best_validation_AP"],
        }
    validation = load_cache(Path(context["validation_cache"]), expected_split="validation")
    selection = read(artifacts / "selection.json")
    check_hashes(selection["artifact_hashes"])
    reports = {}
    for name, filename in (
        ("selected", "policy.safetensors"),
        ("rl", "rl-policy.safetensors"),
        ("warmup", "warmup-policy.safetensors"),
    ):
        predictions = predict_refinement_safetensors(validation, artifacts / filename, device="cpu")
        boxes = coco_predictions(validation, predictions)
        report = evaluate_coco(
            Path(context["validation_annotation"]),
            boxes,
            image_ids=validation["coco_image_ids"],
            model={"name": name, "artifact_sha256": loop._hash_path(artifacts / filename)},
        )
        report.update(errors=0, split="validation", test_access=False)
        write(output / f"{name}-validation.json", report)
        reports[name] = report
    if not math.isclose(
        reports["selected"]["AP"], selection["training_summary"]["best_validation_AP"], abs_tol=1e-8
    ):
        raise ValueError("Safe artifact replay disagrees with training validation score")
    sft = read_result(training / "sft-evaluation.json")
    metrics = {
        "AP": reports["selected"]["AP"],
        "sft_AP": sft["AP"],
        "warmup_AP": reports["warmup"]["AP"],
        "rl_AP": reports["rl"]["AP"],
        "rl_gain_over_warmup": reports["rl"]["AP"] - reports["warmup"]["AP"],
        "selected_origin": selection["selected_origin"],
        "n": len(validation["ids"]),
        "errors": 0,
        "split": "validation",
        "test_access": False,
        "ablation_scope": "All policies use the same frozen top-128 detector query cache",
    }
    check_hashes(context["input_hashes"])
    write(output / "validation.json", metrics)
    return metrics


def development_selection(state_dir):
    """Verify the immutable runner records, including every selected output hash."""
    state_dir = Path(state_dir).resolve()
    state, plan = read(state_dir / "state.json"), read(state_dir / "plan.json")
    if state["status"] != "awaiting_final_evaluation" or state["test_consumed"]:
        raise ValueError("Development must finish at awaiting_final_evaluation without test access")
    if (
        loop._digest(plan) != state["plan_sha256"]
        or loop._digest(plan) != (state_dir / "plan.sha256").read_text().strip()
    ):
        raise ValueError("Development plan hash changed")
    loop._check_events(state_dir / "events.jsonl")
    freeze = read(state_dir / "frozen-selection.json")
    if loop._digest(freeze) != state["frozen_selection_sha256"]:
        raise ValueError("Selected artifact freeze changed")
    check_hashes(freeze["input_hashes"])
    check_hashes(freeze["artifact_hashes"])
    candidate = state["candidates"][state["selected_candidate"]]
    directory = Path(candidate["attempt"]["directory"])
    if loop._hash_path(directory / "validation.json") != freeze["validation_metrics_sha256"]:
        raise ValueError("Selected validation report changed")
    return state, plan, directory


def checked_saved_responses(input_dir, output_dir):
    """Bind an offline score to its exact image/prompt/model request journal."""
    from vision_rl import frontier

    api = wrapper()
    protocol, run = read(input_dir / "inputs.json"), read(output_dir / "run.json")
    api["verify_protocol"](protocol)
    if run["identity_sha256"] != frontier._digest(run["identity"]):
        raise ValueError("Frontier request identity checksum changed")
    for name in ("rows", "examples"):
        prepared = [
            frontier._public(frontier._prepare(row, example=name == "examples"))
            for row in protocol[name]
        ]
        if run["identity"][name] != prepared:
            raise ValueError("Frontier request images, prompts, or examples differ from inputs")
    config = run["identity"]["config"]
    if config["max_requests"] != len(protocol["rows"]) * (
        1 + config.get("max_transient_retries", 0)
    ):
        raise ValueError("Frontier attempt budget differs from the declared policy")
    if config["provider"] == "openai" and (
        config.get("image_detail") != protocol["image_detail"]
        or config.get("response_format") != protocol["response_format"]
    ):
        raise ValueError("OpenAI detail/schema differs from the detection protocol")
    if config.get("response_schema") != protocol.get("response_schema"):
        raise ValueError("Gemini structured-output schema differs from the detection protocol")
    if config["provider"] in {"qwen", "kimi"} and any(
        config.get(key) != protocol.get(key) for key in ("privacy_level", "quantizations")
    ):
        raise ValueError("Managed model routing/privacy differs from the prepared protocol")
    predictions = {}
    for line in (output_dir / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] in predictions:
            raise ValueError("Duplicate frontier prediction ID")
        predictions[row["id"]] = row
    if set(predictions) != {row["id"] for row in protocol["rows"]}:
        raise ValueError("Frontier predictions must cover every selected image")
    check_response_journal(output_dir / "attempts.jsonl", run, predictions)
    model = {
        "provider": config["provider"],
        "name": config["model"],
        "effort": config["effort"],
        "max_output_tokens": config["max_output_tokens"],
        "protocol_sha256": loop._hash_path(input_dir / "inputs.json"),
        "frozen_experiment": protocol.get("frozen_experiment"),
    }
    report = api["score_predictions"](protocol, predictions, model)
    hashes = pinned(
        [
            input_dir / "inputs.json",
            *[output_dir / name for name in ("run.json", "attempts.jsonl", "predictions.jsonl")],
        ]
    )
    return protocol, config, report, hashes


def check_response_journal(path, run, predictions):
    """Read-only journal replay; no transport or resume function can run here."""
    from vision_rl import frontier

    config = run["identity"]["config"]
    rows = {row["id"]: row for row in run["identity"]["rows"]}
    reserved, completed, attempts = {}, {}, {}
    stopped = False
    for line in path.read_text().splitlines():
        event = json.loads(line)
        identifier, attempt = event["id"], event["attempt"]
        if (
            identifier not in rows
            or event.get("identity_sha256", run["identity_sha256"]) != run["identity_sha256"]
        ):
            raise ValueError("Frontier journal differs from its request identity")
        previous = attempts.get(identifier, [])
        if event["event"] == "reserved":
            if (
                attempt != len(reserved) + 1
                or event["request"] != rows[identifier]
                or event.get("item_attempt", 1) != len(previous) + 1
                or (
                    previous
                    and (
                        previous[-1] not in completed
                        or not frontier._retry_allowed(
                            completed[previous[-1]], len(previous), config
                        )
                    )
                )
            ):
                raise ValueError("Frontier journal reservation violates the fixed retry policy")
            reserved[attempt] = event
            attempts.setdefault(identifier, []).append(attempt)
        elif event["event"] == "completed":
            if (
                attempt not in reserved
                or attempt in completed
                or (reserved[attempt]["id"] != identifier)
            ):
                raise ValueError("Frontier journal completion lacks a unique reservation")
            completed[attempt] = event
            stopped |= event.get("hard_stop", False)
        else:
            raise ValueError("Unknown frontier journal event")
    if len(reserved) > config["max_requests"]:
        raise ValueError("Frontier journal exceeds its request budget")
    for identifier, prediction in predictions.items():
        history = attempts.get(identifier, [])
        if not history:
            expected = {
                "response": "",
                "error": "not_requested_hard_stop" if stopped else "not_requested_budget",
            }
        elif history[-1] not in completed:
            expected = {"response": "", "error": "unknown_after_interruption"}
        else:
            expected = completed[history[-1]]["prediction"]
        if any(prediction.get(key) != value for key, value in expected.items()) or (
            bool(prediction.get("error")) != bool(expected.get("error"))
        ):
            raise ValueError("Saved frontier answer differs from the final journaled attempt")


def selected_baseline_config(protocol, measured, retries, delay):
    return wrapper()["baseline_config"](
        {
            "provider": measured["provider"],
            "model": measured["model"],
            "effort": measured["effort"],
            "max_output_tokens": measured["max_output_tokens"],
            "prompt": protocol["prompt_mode"],
            "seed": protocol["sampling_seed"],
            "examples": len(protocol["examples"]),
            "concurrency": 2,
            "max_transient_retries": retries,
            "retry_delay_seconds": delay,
            "structured_output": protocol.get("structured_output", False),
            **(
                {
                    "image_detail": protocol["image_detail"],
                    "coordinate_format": protocol["coordinate_format"],
                }
                if measured["provider"] == "openai"
                else {}
            ),
            **(
                {
                    key: protocol[key]
                    for key in ("privacy_level", "quantizations", "coordinate_format")
                }
                if measured["provider"] in {"qwen", "kimi"}
                else {}
            ),
        }
    )


def response_availability(output_dir):
    """Preserve infrastructure stops without ranking missing answers as capability."""
    events = [json.loads(line) for line in (output_dir / "attempts.jsonl").read_text().splitlines()]
    predictions = [
        json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    missing = [
        row["id"]
        for row in predictions
        if str(row.get("error", "")).startswith("not_requested_")
        or row.get("error") == "unknown_after_interruption"
    ]
    stops = [
        {
            "id": row["id"],
            "status_code": row.get("status_code"),
            "stop_reason": row.get("stop_reason"),
            "attempt": row["attempt"],
        }
        for row in events
        if row.get("hard_stop")
    ]
    return {
        "scope_complete": not missing and not stops,
        "missing_answer_ids": missing,
        "hard_stops": stops,
        "transport_error_answers": sum(bool(row.get("error")) for row in predictions),
        "reason": "incomplete or infrastructure-stopped run" if missing or stops else None,
    }


def select_baselines(args):
    """Choose each provider's highest validation AP; never inspect native test scores."""
    from vision_rl.frontier import PROVIDERS

    records, ids, annotation = [], None, None
    for input_dir, result_dir in args.candidate:
        input_dir, result_dir = Path(input_dir).resolve(), Path(result_dir).resolve()
        protocol = read(input_dir / "inputs.json")
        if protocol["split"] != "validation":
            raise ValueError("Baseline selection may only use validation reports")
        protocol, request_config, report, hashes = checked_saved_responses(input_dir, result_dir)
        saved = read_result(result_dir / "evaluation.json")
        if (
            any(
                saved.get(key) != report.get(key)
                for key in ("AP", "AP50", "AP75", "errors", "image_ids", "source_annotation_sha256")
            )
            or saved["model"]["protocol_sha256"] != report["model"]["protocol_sha256"]
        ):
            raise ValueError("Validation report disagrees with independently rescored answers")
        if ids is not None and (
            report["image_ids"] != ids or report["source_annotation_sha256"] != annotation
        ):
            raise ValueError(
                "Baseline selection requires identical validation images and annotations"
            )
        ids, annotation = report["image_ids"], report["source_annotation_sha256"]
        config = selected_baseline_config(
            protocol, request_config, args.max_transient_retries, args.retry_delay_seconds
        )
        records.append(
            {
                "config": config,
                "AP": report["AP"],
                "errors": report.get("errors", 0),
                "availability": response_availability(result_dir),
                "inputs": str(input_dir),
                "results": str(result_dir),
                "measured_request_configuration": request_config,
                "hashes": {**hashes, **pinned([result_dir / "evaluation.json"])},
            }
        )
    eligible = [row for row in records if row["availability"]["scope_complete"]]
    present = {record["config"]["provider"] for record in eligible}
    required = getattr(args, "required_provider", None) or list(DEFAULT_REQUIRED_PROVIDERS)
    if (
        not isinstance(required, list)
        or not required
        or len(set(required)) != len(required)
        or set(required) - set(PROVIDERS)
    ):
        raise ValueError("Required providers must be a nonempty explicit unique supported set")
    unavailable_path = getattr(args, "unavailable_models", None)
    unavailable = read(unavailable_path) if unavailable_path else []
    if not isinstance(unavailable, list) or any(
        not isinstance(row, dict)
        or set(row) != {"provider", "model", "reason"}
        or row["provider"] not in PROVIDERS
        or row["model"] != PROVIDERS[row["provider"]][0]
        or not isinstance(row["reason"], str)
        or not row["reason"].strip()
        for row in unavailable
    ):
        raise ValueError("Unavailable models require exact provider/model and a nonempty reason")
    missing = set(required) - present
    for provider in sorted(missing):
        stopped = [row for row in records if row["config"]["provider"] == provider]
        if stopped and provider not in {row["provider"] for row in unavailable}:
            unavailable.append(
                {
                    "provider": provider,
                    "model": PROVIDERS[provider][0],
                    "reason": "All supplied protocols are incomplete or infrastructure-stopped",
                }
            )
    if missing - {row["provider"] for row in unavailable}:
        raise ValueError(
            "Missing required baseline; record unavailable model and reason explicitly"
        )
    chosen = [
        max(
            (row for row in eligible if row["config"]["provider"] == provider),
            key=lambda row: row["AP"],
        )
        for provider in sorted(present)
    ]
    result = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_access": False,
        "required_providers": sorted(required),
        "selection_complete": not missing,
        "missing_required_providers": sorted(missing),
        "unavailable_models": unavailable,
        "rule": (
            "Highest observed AP per provider among complete request scopes without hard stops; "
            "all incomplete protocols and completed-answer errors retained"
        ),
        "considered": records,
        "selected": chosen,
        "baselines": [row["config"] for row in chosen],
    }
    if args.output.exists():
        raise ValueError("Baseline selection records are immutable")
    write(args.output, result)
    return result


def check_baseline_selection(record):
    """Reject edited rankings, omitted providers, or swapped selected configurations."""
    from vision_rl.frontier import PROVIDERS

    if record.get("selection_split") != "validation" or record.get("test_access"):
        raise ValueError("Frontier protocols must have been selected on validation")
    considered = record["considered"]
    eligible = []
    for row in considered:
        availability = response_availability(Path(row["results"]))
        if row.get("availability", availability) != availability:
            raise ValueError("Baseline availability differs from its pinned request journal")
        if availability["scope_complete"]:
            eligible.append(row)
    present = {row["config"]["provider"] for row in eligible}
    required = record.get("required_providers", list(DEFAULT_REQUIRED_PROVIDERS))
    if (
        not isinstance(required, list)
        or not required
        or len(set(required)) != len(required)
        or set(required) - set(PROVIDERS)
    ):
        raise ValueError("Invalid explicit required provider set")
    if set(required) - present or record.get("selection_complete") is False:
        raise ValueError("Missing required baseline; unavailable models cannot count as wins")
    for row in considered:
        check_hashes(row["hashes"])
        if not math.isfinite(row["AP"]) or not 0 <= row["AP"] <= 1:
            raise ValueError("Baseline selection contains invalid AP")
        if wrapper()["baseline_config"](row["config"]) != row["config"]:
            raise ValueError("Baseline configuration is not canonical")
        measured = read(Path(row["results"]) / "run.json")["identity"]["config"]
        protocol = read(Path(row["inputs"]) / "inputs.json")
        expected_config = selected_baseline_config(
            protocol,
            measured,
            row["config"]["max_transient_retries"],
            row["config"]["retry_delay_seconds"],
        )
        if row["config"] != expected_config or row["measured_request_configuration"] != measured:
            raise ValueError("Selected model or prompt differs from measured validation settings")
        report = read_result(Path(row["results"]) / "evaluation.json")
        if row["AP"] != report["AP"] or row["errors"] != report.get("errors", 0):
            raise ValueError("Baseline ranking differs from its pinned measured report")
    expected = [
        max(
            (row for row in eligible if row["config"]["provider"] == provider),
            key=lambda row: row["AP"],
        )
        for provider in sorted(present)
    ]
    if record["selected"] != expected or record["baselines"] != [row["config"] for row in expected]:
        raise ValueError("Final baselines must be the strongest measured validation selections")


def capture_groups(images):
    """Unknown capture metadata forms one conservative shared cluster."""
    groups = {}
    for row in images:
        name = row.get("extra", {}).get("name") or row["file_name"].split(".rf.")[0]
        matched = re.match(r"^WIN_(\d{8})_(\d{2})_\d{2}_\d{2}_Pro(?:[._]|$)", name)
        groups[str(row["id"])] = (
            f"WIN:{matched[1]}:{matched[2]}" if matched else "unknown_capture_shared_cluster"
        )
    if len(set(groups.values())) < 2:
        raise ValueError("Final cluster bootstrap needs at least two capture groups")
    return groups


def validate_profile(args):
    """Measure the selected policy using image-only inference on all native validation."""
    import torch

    from vision_rl.detection_eval import evaluate_coco, load_coco
    from vision_rl.detection_inference import InferenceConfig, infer_images, load_image_manifest
    from vision_rl.detection_training import load_detector_safetensors

    state_dir, output = args.state_dir.resolve(), args.output.resolve()
    with loop._lock(state_dir):
        _, plan, selected = development_selection(state_dir)
        context = read(plan["pills_context"])
        if context["truncated_development"]:
            raise ValueError("Final-profile evidence requires full native development caches")
        config = InferenceConfig(**read(args.config))
        config.validate()
        if config.device != "cpu":
            raise ValueError("Reproducible profile validation must explicitly use CPU")
        if output.exists():
            raise ValueError("Profile evidence is immutable; choose a fresh output directory")
        annotation = Path(context["validation_annotation"])
        dataset = load_coco(annotation)
        category_ids = tuple(row["id"] for row in dataset["categories"])
        if len(category_ids) != 1:
            raise ValueError("Refinement inference requires exactly one foreground category")
        model = Path(context["detector"])
        policy = selected / "artifacts/policy.safetensors"
        sources = [*implementation_paths(), ROOT / "src/vision_rl/detection_inference.py"]
        inputs = pinned([model, policy, annotation, args.config, *sources])
        output.mkdir(parents=True)
        manifest = output / "validation-images.json"
        write(
            manifest,
            {
                "images": [
                    {key: row[key] for key in ("id", "file_name", "width", "height")}
                    for row in dataset["images"]
                ]
            },
        )
        images = load_image_manifest(manifest, annotation.parent)
        provenance = {
            "checkpoint_sha256": loop._hash_path(model),
            "refinement_sha256": loop._hash_path(policy),
            "manifest_sha256": loop._hash_path(manifest),
            "images": [{"id": row["id"], "sha256": loop._hash_path(row["path"])} for row in images],
            "category_ids_in_training_order": list(category_ids),
            "configuration": asdict(config),
            "implementation_hashes": pinned(sources),
            "label_access": False,
            "split": "validation",
            "test_access": False,
        }
        write(output / "profile.json", asdict(config))
        write(output / "provenance.json", provenance)
        torch.set_num_threads(config.cpu_threads)
        detector = load_detector_safetensors(
            model,
            architecture=config.architecture,
            num_classes=len(category_ids),
            device=config.device,
            resolution=config.checkpoint_resolution,
        )
        boxes, timing = infer_images(
            detector,
            images,
            category_ids,
            config,
            refinement=policy,
            detector_sha256=provenance["checkpoint_sha256"],
        )
        write(output / "predictions.coco.json", boxes)
        write(output / "inference.json", timing)
        report = evaluate_coco(annotation, boxes, image_ids=[row["id"] for row in images])
        report.update(errors=0, split="validation", test_access=False)
        check_hashes(inputs)
        for row in images:
            expected = next(
                item["sha256"] for item in provenance["images"] if item["id"] == row["id"]
            )
            if loop._hash_path(row["path"]) != expected:
                raise ValueError("Validation image changed during profile replay")
        write(output / "evaluation.json", report)
        checked_inference_profile(output / "profile.json", output, context, selected)
        return {
            "AP": report["AP"],
            "n": len(images),
            "split": "validation",
            "test_access": False,
            "inference_config": str(output / "profile.json"),
            "inference_validation": str(output),
        }


def checked_inference_profile(config_path, validation_dir, context, selected):
    """Accept optional TTA only with replayable validation evidence for these weights."""
    from vision_rl.detection_eval import evaluate_coco, load_coco
    from vision_rl.detection_inference import InferenceConfig

    if config_path is None or validation_dir is None:
        raise ValueError("An inference profile requires both configuration and validation evidence")
    config = InferenceConfig(**read(config_path))
    config.validate()
    if config.device != "cpu":
        raise ValueError("The final reproducible profile must explicitly use CPU")
    provenance_path = validation_dir / "provenance.json"
    predictions_path = validation_dir / "predictions.coco.json"
    report_path = validation_dir / "evaluation.json"
    provenance, report = read(provenance_path), read_result(report_path)
    if provenance.get("label_access") is not False:
        raise ValueError("Inference evidence must declare label-free prediction")
    implementations = provenance.get("implementation_hashes", {})
    required = {
        str(ROOT / "src/vision_rl" / name)
        for name in (
            "detection_inference.py",
            "detection_training.py",
            "detection_rl.py",
            "detection_rewards.py",
        )
    }
    if not required.issubset(implementations):
        raise ValueError("Inference validation must pin its actual implementation sources")
    check_hashes(implementations)
    if loop._canonical(asdict(config)) != loop._canonical(provenance["configuration"]):
        raise ValueError("Inference profile differs from its measured validation configuration")
    if provenance["checkpoint_sha256"] != loop._hash_path(Path(context["detector"])) or (
        provenance["refinement_sha256"]
        != loop._hash_path(selected / "artifacts/policy.safetensors")
    ):
        raise ValueError("Inference validation used different specialist weights")
    annotation = Path(context["validation_annotation"])
    # Roboflow COCO includes an unused category-0 parent label. Match the same
    # validated foreground mapping used for training, inference, and scoring.
    native = load_coco(annotation)
    expected_ids = sorted(row["id"] for row in native["images"])
    if (
        report["image_ids"] != expected_ids
        or report.get("errors", 0)
        or report["source_annotation_sha256"] != loop._hash_path(annotation)
        or sorted(row["id"] for row in provenance["images"]) != expected_ids
        or provenance["category_ids_in_training_order"]
        != [row["id"] for row in native["categories"]]
    ):
        raise ValueError("Inference validation must cover the complete native validation split")
    image_hashes = {row["id"]: row["sha256"] for row in provenance["images"]}
    for image in native["images"]:
        if image_hashes.get(image["id"]) != loop._hash_path(annotation.parent / image["file_name"]):
            raise ValueError("Inference validation images differ from native validation")
    replay = evaluate_coco(annotation, read_result(predictions_path), image_ids=expected_ids)
    if not math.isclose(replay["AP"], report["AP"], abs_tol=1e-12):
        raise ValueError("Inference validation AP does not match saved predictions")
    return asdict(config), [
        config_path,
        provenance_path,
        predictions_path,
        report_path,
        ROOT / "src/vision_rl/detection_inference.py",
    ]


def checked_pipeline_validation(profile_path, validation_dir, annotation):
    """Independently replay scoring for the exact portable image-only pipeline."""
    from vision_rl import detection_recipe as recipe
    from vision_rl.detection_eval import evaluate_coco, load_coco
    from vision_rl.detection_inference import load_image_manifest

    profile, _, artifacts = recipe.load_profile(profile_path)
    if profile["schema_version"] != 2:
        raise ValueError("Final pipelines require schema2 package and source provenance")
    provenance = read(validation_dir / "provenance.json")
    prediction_path = validation_dir / "predictions.coco.json"
    report = read_result(validation_dir / "evaluation.json")
    manifest = validation_dir.parent / "validation-images.json"
    native = load_coco(annotation)
    rows = load_image_manifest(manifest, annotation.parent)
    expected = [
        {key: image[key] for key in ("id", "file_name", "width", "height")}
        for image in native["images"]
    ]
    if read(manifest) != {"images": expected}:
        raise ValueError("Pipeline validation must cover every native validation image")
    if (
        provenance.get("label_access") is not False
        or provenance.get("profile_sha256") != loop._hash_path(profile_path)
        or provenance.get("manifest_sha256") != loop._hash_path(manifest)
        or provenance.get("predictions_sha256") != loop._hash_path(prediction_path)
        or provenance.get("implementation_hashes") != profile["implementation_hashes"]
        or provenance.get("runtime_packages") != profile["runtime_packages"]
        or provenance.get("proposal_pipeline_sha256") != recipe.proposal_pipeline_sha256(profile)
        or provenance.get("artifacts")
        != {key: recipe.descriptor(path) for key, path in artifacts.items()}
        or provenance.get("image_hashes")
        != {str(row["path"]): loop._hash_path(row["path"]) for row in rows}
    ):
        raise ValueError(
            "Pipeline evidence differs from frozen weights, code, packages, or image bytes"
        )
    replay = evaluate_coco(annotation, read_result(prediction_path))
    if (
        report.get("split") != "validation"
        or report.get("test_access") is not False
        or report.get("errors") != 0
        or any(
            report.get(key) != replay.get(key)
            for key in ("AP", "AP50", "AP75", "image_ids", "source_annotation_sha256")
        )
    ):
        raise ValueError("Pipeline validation report differs from its saved native predictions")
    return (
        profile,
        artifacts,
        report,
        [
            profile_path,
            manifest,
            prediction_path,
            validation_dir / "provenance.json",
            validation_dir / "evaluation.json",
            *artifacts.values(),
        ],
    )


def freeze_pipeline_final(args, state, plan, selected, lock_fd):
    """Bind a selected typed pipeline to the existing once-only final protocol."""
    from vision_rl import detection_recipe as recipe
    from vision_rl.detection_eval import load_coco
    from vision_rl.detection_training import inspect_split

    source, output = args.source.resolve(), args.output.resolve()
    context = read(plan.get("recipe") or plan["pipeline_context"])
    if Path(context["source"]) != source:
        raise ValueError("Final source differs from the development recipe")
    for name, dirname in (("train", "train"), ("validation", "valid")):
        current_split = inspect_split(source / dirname)
        if any(
            context["splits"][name][key] != current_split[key]
            for key in ("fingerprint", "images", "categories")
        ):
            raise ValueError("Development data changed before final freeze")
    if context["splits"]["validation"]["images"] != 423:
        raise ValueError(
            "Native final freeze requires all 423 validation images; smoke cannot qualify"
        )
    profile_path = selected / "artifacts/pipeline.json"
    profile, artifacts, report, evidence = checked_pipeline_validation(
        profile_path, selected / "image-validation", source / "valid/_annotations.coco.json"
    )
    if report["n"] != 423:
        raise ValueError("Native final freeze requires all 423 measured validation images")
    if report["AP"] != read(selected / "validation.json")["AP"]:
        raise ValueError("Selected candidate metrics differ from full pipeline validation")
    baselines = read(args.baseline_selection)
    check_baseline_selection(baselines)
    test = load_coco(source / "test/_annotations.coco.json")
    if len(test["images"]) != 200:
        raise ValueError("Native countingpills final evaluation requires all 200 test images")
    groups = capture_groups(test["images"])
    output.mkdir(parents=True)
    sources = [
        *implementation_paths(),
        *recipe.source_paths(),
        ROOT / "src/vision_rl/detection_sensitivity.py",
    ]
    specialist = {
        "schema_version": 2,
        "source": str(source),
        "detector": str(artifacts["detector"]),
        "detector_sha256": profile["detector"]["sha256"],
        "inference_backend": "portable_pipeline",
        "pipeline_profile": str(profile_path),
        "selected_origin": profile["training_origin"],
        "inference_contract": {"final_test_images": 200},
        "implementation_hashes": pinned(sources),
        "runtime_packages": profile["runtime_packages"],
        "proposal_pipeline_sha256": recipe.proposal_pipeline_sha256(profile),
        "development_plan_sha256": state["plan_sha256"],
        "development_selection_sha256": state["frozen_selection_sha256"],
        "required_providers": baselines.get("required_providers", list(DEFAULT_REQUIRED_PROVIDERS)),
        "unavailable_models": baselines.get("unavailable_models", []),
        "test_image_hashes": {
            str(row["id"]): loop._hash_path(source / "test" / row["file_name"])
            for row in test["images"]
        },
        "selection_rule": "Keep the complete validation-selected pipeline; never adapt to test",
    }
    write(output / "specialist.json", specialist)
    write(
        output / "capture-groups.json",
        {"rule": "WIN hour; all unknown captures share one cluster", "groups": groups},
    )
    pinned_files = list(
        dict.fromkeys(
            [
                *evidence,
                *sources,
                output / "specialist.json",
                output / "capture-groups.json",
                args.baseline_selection.resolve(),
                CLUSTER_REGISTRATION,
            ]
        )
    )
    frozen = wrapper()["create_freeze"](
        source,
        artifacts["detector"],
        baselines["baselines"],
        DEFAULT_CRITERIA,
        output / "freeze.json",
        [path for path in pinned_files if path != artifacts["detector"]],
    )
    runner = loop._Runner(args.state_dir.resolve(), lock_fd)
    runner.state["final_protocol"] = {
        "directory": str(output),
        "freeze_sha256": loop._hash_path(output / "freeze.json"),
        "specialist_sha256": loop._hash_path(output / "specialist.json"),
    }
    runner.save()
    runner.event("native_final_protocol_frozen", **runner.state["final_protocol"])
    return {
        "status": "awaiting_final_evaluation",
        "freeze": str(output / "freeze.json"),
        "baselines": len(frozen["baselines"]),
        "native_test_images": 200,
        "test_consumed": False,
        "selected_origin": profile["training_origin"],
    }


def freeze_final(args):
    from vision_rl.detection_eval import load_coco

    state_dir, output, source = (
        path.resolve() for path in (args.state_dir, args.output, args.source)
    )
    with loop._lock(state_dir) as lock_fd:
        state, plan, selected = development_selection(state_dir)
        if output.exists() or state.get("final_protocol"):
            raise ValueError("A final protocol is already frozen; never create a replacement")
        if "recipe" in plan or "pipeline_context" in plan:
            return freeze_pipeline_final(args, state, plan, selected, lock_fd)
        context = read(plan["pills_context"])
        if context["truncated_development"]:
            raise ValueError(
                "Smoke/subsample development runs cannot authorize the native final test"
            )
        baselines = read(args.baseline_selection)
        check_baseline_selection(baselines)
        annotation = source / "test/_annotations.coco.json"
        test = load_coco(annotation)
        if len(test["images"]) != 200:
            raise ValueError(
                "This native countingpills v2 final protocol requires all 200 test images"
            )
        groups = capture_groups(test["images"])
        selection = read(selected / "artifacts/selection.json")
        profile, profile_inputs = checked_inference_profile(
            getattr(args, "inference_config", None),
            getattr(args, "inference_validation", None),
            context,
            selected,
        )
        output.mkdir(parents=True)
        specialist = {
            "schema_version": 1,
            "source": str(source),
            "detector": context["detector"],
            "detector_sha256": loop._hash_path(Path(context["detector"])),
            "selected_policy": str(selected / "artifacts/policy.safetensors"),
            "selected_origin": selection["selected_origin"],
            "inference_backend": "image_views",
            "inference_configuration": profile,
            "refinement_configuration": selection["config"],
            "policies": {
                name: str(selected / "artifacts" / filename)
                for name, filename in (
                    ("selected", "policy.safetensors"),
                    ("rl", "rl-policy.safetensors"),
                    ("warmup", "warmup-policy.safetensors"),
                )
            },
            "inference_contract": {
                "resolutions": profile["resolutions"],
                "horizontal_flip": profile["horizontal_flip"],
                "fusion_iou": profile["fusion_iou"],
                "checkpoint_resolution": profile.get("checkpoint_resolution"),
                "preprocessing": "Label-free RF-DETR validation resize with full RGB decode",
                "refinement_hidden": 128,
                "geometry_step": 0.03,
                "identity_probability": 0.75,
                "policy_device": "cpu",
                "policy_cpu_threads": 1,
                "final_test_images": 200,
            },
            "implementation_hashes": pinned(
                [
                    *implementation_paths(),
                    ROOT / "src/vision_rl/detection_sensitivity.py",
                    ROOT / "src/vision_rl/detection_inference.py",
                ]
            ),
            "development_plan_sha256": state["plan_sha256"],
            "development_selection_sha256": state["frozen_selection_sha256"],
            "selection_rule": "Keep the validation-selected artifact regardless of test ablations",
            "capture_cluster_registration": {
                "path": str(CLUSTER_REGISTRATION),
                "sha256": loop._hash_path(CLUSTER_REGISTRATION),
            },
        }
        write(output / "specialist.json", specialist)
        write(
            output / "capture-groups.json",
            {
                "rule": (
                    "Explicit WIN capture hour; unknown captures share one conservative cluster"
                ),
                "groups": groups,
            },
        )
        artifacts = [
            *sorted((selected / "artifacts").iterdir()),
            output / "specialist.json",
            output / "capture-groups.json",
            args.baseline_selection.resolve(),
            CLUSTER_REGISTRATION,
            *implementation_paths(),
            ROOT / "src/vision_rl/detection_sensitivity.py",
            ROOT / "src/vision_rl/detection_inference.py",
            *profile_inputs,
        ]
        api = wrapper()
        frozen = api["create_freeze"](
            source,
            Path(context["detector"]),
            baselines["baselines"],
            context["final_criteria"],
            output / "freeze.json",
            list(dict.fromkeys(artifacts)),
        )
        runner = loop._Runner(state_dir, lock_fd)
        runner.state["final_protocol"] = {
            "directory": str(output),
            "freeze_sha256": loop._hash_path(output / "freeze.json"),
            "specialist_sha256": loop._hash_path(output / "specialist.json"),
        }
        runner.save()
        runner.event("native_final_protocol_frozen", **runner.state["final_protocol"])
    return {
        "status": "awaiting_final_evaluation",
        "freeze": str(output / "freeze.json"),
        "baselines": len(frozen["baselines"]),
        "native_test_images": 200,
        "test_consumed": False,
        "selected_origin": selection["selected_origin"],
    }


def offline_baselines(pairs, frozen, freeze_path):
    """Verify and independently rescore saved responses without issuing requests."""
    api, reports, hashes = wrapper(), {}, {}
    reservation = read(freeze_path.with_name(freeze_path.name + ".runs.json"))
    if reservation["freeze_sha256"] != loop._hash_path(freeze_path):
        raise ValueError("Frontier outputs belong to a different frozen experiment")
    for input_dir, output_dir in pairs:
        input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
        protocol, config, report, response_hashes = checked_saved_responses(input_dir, output_dir)
        reference = protocol.get("frozen_experiment", {})
        if reference.get("sha256") != loop._hash_path(freeze_path):
            raise ValueError("Frontier protocol is bound to a different final freeze")
        settings = SimpleNamespace(
            provider=config["provider"],
            model=config["model"],
            effort=config["effort"],
            max_tokens=config["max_output_tokens"],
            concurrency=config["max_inflight"],
            max_transient_retries=config.get("max_transient_retries", 0),
            retry_delay_seconds=config.get("retry_delay_seconds", 2),
        )
        baseline = api["frozen_baseline"](frozen, protocol, settings)
        if baseline["id"] in reports or reservation["baselines"].get(baseline["id"], {}).get(
            "output"
        ) != str(output_dir):
            raise ValueError("Each frozen frontier must use exactly its reserved output journal")
        reports[baseline["id"]] = report
        hashes.update(response_hashes)
    if set(reports) != {entry["id"] for entry in frozen["baselines"]}:
        raise ValueError("Final evaluation requires every preselected frozen frontier baseline")
    return reports, hashes


def finalize(args):
    """Explicit one-shot native inference; never called by development or resume."""
    from vision_rl.detection_eval import compare_detections, evaluate_coco, load_coco
    from vision_rl.detection_inference import InferenceConfig, infer_images, load_image_manifest
    from vision_rl.detection_sensitivity import (
        compare_to_failure_upper_bound,
        frontier_failure_upper_bound,
    )
    from vision_rl.detection_training import load_detector_safetensors

    state_dir, output = args.state_dir.resolve(), args.output.resolve()
    with loop._lock(state_dir) as lock_fd:
        state, _, selected = development_selection(state_dir)
        binding = state.get("final_protocol")
        if not binding:
            raise ValueError("Run freeze-final before explicit native final evaluation")
        bundle = Path(binding["directory"])
        freeze_path = bundle / "freeze.json"
        if loop._hash_path(freeze_path) != binding["freeze_sha256"]:
            raise ValueError("Final freeze changed")
        if loop._hash_path(bundle / "specialist.json") != binding["specialist_sha256"]:
            raise ValueError("Frozen specialist inference protocol changed")
        frozen = wrapper()["check_freeze"](freeze_path)
        specialist, groups = (
            read(bundle / "specialist.json"),
            read(bundle / "capture-groups.json")["groups"],
        )
        check_hashes(specialist["implementation_hashes"])
        frontiers, response_hashes = offline_baselines(args.baseline, frozen, freeze_path)
        if output.exists():
            raise ValueError("Final result directory must be new")
        runner = loop._Runner(state_dir, lock_fd)
        runner.state.update(
            status="evaluating", test_consumed=True, native_final_output=str(output)
        )
        runner.save()  # Durable reservation precedes every native detector/test operation.
        runner.event("native_final_test_consumed", output=str(output))
        output.mkdir(parents=True)
        try:
            source = Path(specialist["source"])
            native = load_coco(source / "test/_annotations.coco.json")
            manifest = {
                "images": [
                    {key: row[key] for key in ("id", "file_name", "width", "height")}
                    for row in native["images"]
                ]
            }
            write(output / "test-images.json", manifest)
            images = load_image_manifest(output / "test-images.json", source / "test")
            if len(images) != specialist["inference_contract"]["final_test_images"]:
                raise ValueError("Native final manifest does not contain all frozen test images")
            if specialist.get("inference_backend") == "portable_pipeline":
                from vision_rl import detection_recipe as recipe

                image_hashes = {str(row["id"]): loop._hash_path(row["path"]) for row in images}
                if image_hashes != specialist["test_image_hashes"]:
                    raise ValueError("Final image bytes differ from the pretest freeze")
                recipe.predict_profile(
                    Path(specialist["pipeline_profile"]),
                    output / "test-images.json",
                    source / "test",
                    output / "selected-inference",
                )
                boxes = read_result(output / "selected-inference/predictions.coco.json")
                report = evaluate_coco(
                    source / "test/_annotations.coco.json",
                    boxes,
                    image_ids=[image["id"] for image in images],
                    model={
                        "name": "selected",
                        "selected_origin": specialist["selected_origin"],
                        "profile_sha256": loop._hash_path(Path(specialist["pipeline_profile"])),
                    },
                )
                report.update(errors=0, split="test")
                write(output / "selected-evaluation.json", report)
                ablations = {"selected": report}
            else:
                config = InferenceConfig(**specialist["inference_configuration"])
                config.validate()
                detector = load_detector_safetensors(
                    Path(specialist["detector"]),
                    architecture=config.architecture,
                    device=config.device,
                    resolution=config.checkpoint_resolution,
                )
                ablations, inference_cache = {}, {}
                for name, path in specialist["policies"].items():
                    artifact_hash = loop._hash_path(Path(path))
                    if artifact_hash not in inference_cache:
                        inference_cache[artifact_hash] = infer_images(
                            detector,
                            images,
                            category_ids=tuple(category["id"] for category in native["categories"]),
                            config=config,
                            refinement=Path(path),
                            detector_sha256=specialist["detector_sha256"],
                        )
                    boxes, timing = inference_cache[artifact_hash]
                    write(output / f"{name}-predictions.json", boxes)
                    write(output / f"{name}-inference.json", timing)
                    report = evaluate_coco(
                        source / "test/_annotations.coco.json",
                        boxes,
                        image_ids=[image["id"] for image in images],
                        model={
                            "name": name,
                            "selected_origin": specialist["selected_origin"],
                            "artifact_sha256": artifact_hash,
                        },
                    )
                    report.update(errors=0, split="test")
                    write(output / f"{name}-evaluation.json", report)
                    ablations[name] = report
            criteria = frozen["success_criteria"]
            options = {
                "bootstrap_samples": criteria["bootstrap_samples"],
                "seed": criteria["bootstrap_seed"],
                "margin": criteria["minimum_margin_over_each_frontier"],
                "min_ci_lower": criteria["minimum_cluster_bootstrap_lower95_margin"],
            }
            comparisons, sensitivity = {}, {}
            for identifier, report in frontiers.items():
                name = report["model"]["provider"]
                write(output / f"{name}-evaluation.json", report)
                comparisons[name] = compare_detections(
                    ablations["selected"],
                    report,
                    groups,
                    output=output / f"{name}-comparison.json",
                    **options,
                )
                if report.get("errors", 0):
                    upper = frontier_failure_upper_bound(
                        source / "test/_annotations.coco.json",
                        report,
                        output=output / f"{name}-oracle-upper-bound.json",
                    )
                    sensitivity[name] = compare_to_failure_upper_bound(
                        ablations["selected"],
                        upper,
                        groups,
                        output=output / f"{name}-supplemental-sensitivity.json",
                        **options,
                    )
            passed = (
                ablations["selected"]["AP"] >= criteria["minimum_specialist_ap"]
                and all(item["margin_target_met"] for item in comparisons.values())
                and all(
                    item.get("errors", 0) <= criteria["allowed_primary_errors"]
                    for item in frontiers.values()
                )
            )
            check_hashes(response_hashes)
            wrapper()["check_freeze"](freeze_path)
            result = {
                "primary_goal_met": passed,
                "selected_origin": specialist["selected_origin"],
                "native_test_images": len(images),
                "selected_AP": ablations["selected"]["AP"],
                "warmup_AP": ablations.get("warmup", {}).get("AP"),
                "rl_AP": ablations.get("rl", {}).get("AP"),
                "required_providers": specialist.get(
                    "required_providers", list(DEFAULT_REQUIRED_PROVIDERS)
                ),
                "unavailable_models": specialist.get("unavailable_models", []),
                "criteria": criteria,
                "comparisons": comparisons,
                "supplemental_failure_sensitivity": sensitivity,
                "frontier_response_hashes": response_hashes,
                "capture_group_map": {
                    "path": str(bundle / "capture-groups.json"),
                    "sha256": loop._hash_path(bundle / "capture-groups.json"),
                    "resampling_units": len(set(groups.values())),
                    "group_sizes": {
                        group: list(groups.values()).count(group)
                        for group in sorted(set(groups.values()))
                    },
                },
                "test_used_for_selection": False,
                "no_new_hosted_requests": True,
            }
            write(output / "summary.json", result)
            runner.finish(
                "completed" if passed else "needs_fresh_test",
                test_gate_passed=passed,
                native_final_summary=str(output / "summary.json"),
            )
            return result
        except BaseException as exc:
            runner.finish(
                "needs_fresh_test",
                test_gate_passed=False,
                reason=f"native_final_interrupted:{type(exc).__name__}",
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="create an immutable development-only RL search plan")
    for name in ("detector", "train-cache", "validation-cache", "candidates", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--minimum-validation-ap", type=float, default=0.8)
    prepare.add_argument("--max-attempts", type=int)
    prepare.add_argument("--max-wall-seconds", type=float, default=14400)
    prepare.add_argument("--command-timeout-seconds", type=float, default=3600)
    worker = sub.add_parser("stage", help=argparse.SUPPRESS)
    for name in ("context", "candidate", "output"):
        worker.add_argument("--" + name, type=Path, required=True)
    worker.add_argument("--phase", choices=("train", "validate"), required=True)
    for name in ("run", "resume", "status"):
        command = sub.add_parser(name)
        command.add_argument("--state-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--plan", type=Path, required=True)
        if name != "status":
            command.add_argument("--background", action="store_true")
    baselines = sub.add_parser(
        "select-baselines", help="choose strongest measured validation protocols"
    )
    baselines.add_argument(
        "--candidate",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("INPUT_DIR", "RESULT_DIR"),
    )
    baselines.add_argument("--output", type=Path, required=True)
    baselines.add_argument("--max-transient-retries", type=int, choices=(0, 1, 2), default=0)
    baselines.add_argument("--retry-delay-seconds", type=float, default=2)
    baselines.add_argument(
        "--required-provider",
        action="append",
        help="Repeat to fix the required provider set; default gemini, anthropic, openai",
    )
    baselines.add_argument(
        "--unavailable-models",
        type=Path,
        help="JSON provider/model/reason records; missing required models block final freeze",
    )
    profile = sub.add_parser(
        "validate-profile",
        help="replay selected weights on full validation using image-only CPU inference",
    )
    for name in ("state-dir", "config", "output"):
        profile.add_argument("--" + name, type=Path, required=True)
    freeze = sub.add_parser("freeze-final", help="pin selected artifacts and native test protocol")
    for name in (
        "state-dir",
        "source",
        "baseline-selection",
        "output",
    ):
        freeze.add_argument("--" + name, type=Path, required=True)
    for name in ("inference-config", "inference-validation"):
        freeze.add_argument("--" + name, type=Path, help="Required for legacy cached-query plans")
    final = sub.add_parser("finalize", help="explicit one-shot native test; no new hosted calls")
    final.add_argument("--state-dir", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument(
        "--baseline",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("INPUT_DIR", "RESULT_DIR"),
    )
    args = parser.parse_args(argv)
    if args.command in {"run", "resume", "status"}:
        delegated = [args.command, "--state-dir", str(args.state_dir)]
        if args.command == "run":
            delegated.extend(("--plan", str(args.plan)))
        if getattr(args, "background", False):
            delegated.append("--background")
        return loop.main(delegated)
    handlers = {
        "prepare": prepare_plan,
        "stage": stage,
        "select-baselines": select_baselines,
        "validate-profile": validate_profile,
        "freeze-final": freeze_final,
        "finalize": finalize,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
