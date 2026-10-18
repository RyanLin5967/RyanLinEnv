"""Named baseline v2: author Qwen3.8-27B (bf16 Hugging Face weights) on CPU, sharded.

Pre-registered by docs/aircraft-named-baseline-plan-v2.json. The committed plan is
UNACTIVATED and this runner refuses it: a run needs a separately activated copy whose
SHA256 the operator passes explicitly, exactly as in v1.

This is the v1 runner (scripts/aircraft_named_baseline.py) with an engine option, not a
reimplementation. v1 is imported and its stage inputs (pinned-input rehash, discovery
selection, prompt/vocabulary/parser identity), journal and ledger replay, canonical answer
rule and Hoeffding interval are called as they are. What v2 adds:

  engine   scripts/aircraft_named_baseline_hf.py: transformers on CPU over the author
           weights, the same journal loop as v1 (local_vlm._run_locked) and the mlx-vlm
           sampler chain
  shards   author-order position i of a stage manifest belongs to shard i mod N; each
           shard has its own journal and budget ledger and runs on its own host
  merge    the union of shard journals; refuses a duplicate image, an image outside its
           shard, a shard directory outside the declared N, or differing run identities,
           and scores only when every image of the stage completed with no unknown outcome
  smoke    four TRAIN images through the engine without a journal, for latency sizing

v1 files are not edited: the paused v1 run hashes them into its identity and ledger, and
an edit would make it refuse to resume. v1 run-2 results are never read by v2.

Blind spots, stated on purpose: the ledgers are local state, not tamper-proof witnesses;
a shard killed with SIGKILL leaves no "finished" record and is charged its full wall cap;
shards on different hosts share no lock, so running the same shard on two hosts at once is
prevented only by the operator (a merge then refuses the duplicate images); latency on
shared hosts includes other users' load.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from vision_rl import local_vlm
from vision_rl.aircraft_data import PREPROCESSING_ID
from vision_rl.classification_comparison import _inputs, _score, compare_classification
from vision_rl.vlm_local_training import load_spec, parse_label, prompt_text

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("aircraft_named_baseline",
                                               HERE / "aircraft_named_baseline.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)
_spec = importlib.util.spec_from_file_location("aircraft_named_baseline_hf",
                                               HERE / "aircraft_named_baseline_hf.py")
local_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local_hf)

PLAN_ID = "aircraft-named-baseline-qwen3.8-27b-hf-cpu-v2"
STAGES = v1.STAGES
SOURCE_MODULES = v1.SOURCE_MODULES
SHARD_NAME = re.compile(r"shard-(\d+)-of-(\d+)")


def load_plan(path, expected_sha256=None, *, require_activated=True):
    path = Path(path)
    plan = v1.read_json(path)
    if plan.get("plan_id") != PLAN_ID or plan.get("schema_version") != 1:
        raise ValueError("Not the named-baseline v2 plan schema")
    if require_activated:
        activation = plan.get("activation")
        if (
            plan.get("activated") is not True
            or not isinstance(activation, dict)
            or not activation.get("activated_at")
            or not activation.get("approved_by")
        ):
            raise ValueError("Named-baseline v2 plan is unactivated; it must not run")
        if expected_sha256 is None or local_vlm.sha_file(path) != expected_sha256:
            raise ValueError("Pass the activated plan's exact SHA256 to run it")
        if any(digest is None for digest in plan["sources_pinned"].values()):
            raise ValueError("An activated plan must pin every runner source")
    return plan


def source_hashes():
    hashes = {
        name: local_vlm.sha_file(importlib.import_module(module).__file__)
        for name, module in SOURCE_MODULES.items()
    }
    hashes["scripts/aircraft_named_baseline.py"] = local_vlm.sha_file(v1.__file__)
    hashes["scripts/aircraft_named_baseline_hf.py"] = local_vlm.sha_file(local_hf.__file__)
    hashes["scripts/aircraft_named_baseline_v2.py"] = local_vlm.sha_file(__file__)
    return hashes


def shard_count(plan):
    shards = plan["sharding"]["shards"]
    if type(shards) is not int or shards < 1:
        raise ValueError("Plan must declare a positive integer shard count")
    return shards


def shard_ids(identifiers, shard, shards):
    """Author-order position i belongs to shard i mod N; metadata only."""
    if type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("Shard index outside the declared shard count")
    return list(identifiers)[shard::shards]


def stage_inputs(plan, stage, repository, shard=None):
    """v1's stage inputs (every pinned input rehashed), restricted to one shard."""
    inputs = v1.stage_inputs(plan, stage, repository)
    contract = inputs["task_contract"]
    contract["plan_id"] = PLAN_ID
    if shard is None:
        return inputs
    shards = shard_count(plan)
    chosen = set(shard_ids([row["id"] for row in inputs["rows"]], shard, shards))
    inputs["rows"] = [row for row in inputs["rows"] if row["id"] in chosen]
    inputs["gold"] = [row for row in inputs["gold"] if row["id"] in chosen]
    contract.update(stage_images=contract["images"], images=len(inputs["rows"]),
                    shard=shard, shards=shards,
                    shard_rule="author-order position i belongs to shard i mod shards")
    return inputs


def shard_directory(output, stage, shard, shards):
    return Path(output) / f"stage-{stage}" / f"shard-{shard}-of-{shards}"


def _shared_identity(identity):
    """Run identity minus what legitimately differs between shards of one stage."""
    shared = json.loads(json.dumps(identity))
    shared.pop("rows")
    contract = shared["config"]["task_contract"]
    for key in ("images", "shard"):
        contract.pop(key)
    return shared


def merged_report(plan, stage, repository, output):
    """Union of every shard journal of a stage. Metric is null unless all images completed."""
    inputs = stage_inputs(plan, stage, repository)
    identifiers = [row["id"] for row in inputs["rows"]]
    shards = shard_count(plan)
    stage_directory = Path(output) / f"stage-{stage}"
    if stage_directory.exists():
        for child in stage_directory.iterdir():
            match = SHARD_NAME.fullmatch(child.name)
            if not match or int(match[2]) != shards or int(match[1]) >= shards:
                raise ValueError(f"Unexpected shard directory {child.name}; plan declares "
                                 f"{shards} shards")
    owner, completed, reserved, unknown, hard_stop = {}, {}, {}, [], []
    loads, latencies, shared, per_shard = [], [], None, []
    charged = generation = 0.0
    invocations = 0
    for shard in range(shards):
        assigned = shard_ids(identifiers, shard, shards)
        for identifier in assigned:
            owner[identifier] = shard
        directory = shard_directory(output, stage, shard, shards)
        journal = v1.journal_state(directory / "run", assigned)
        for identifier in set(journal["reserved"]) | set(journal["completed"]):
            if owner.get(identifier) != shard:
                raise ValueError(f"Image {identifier} journaled outside its shard {shard}")
            if identifier in reserved:
                raise ValueError(f"Duplicate image across shards: {identifier}")
            reserved[identifier] = shard
        completed.update(journal["completed"])
        unknown += journal["unknown"]
        hard_stop += journal["hard_stop"]
        run_json = directory / "run" / "run.json"
        if run_json.exists():
            identity = _shared_identity(json.loads(run_json.read_text())["identity"])
            if shared is None:
                shared = identity
            elif identity != shared:
                raise ValueError(f"Shard {shard} ran under a different run identity")
        ledger = v1.ledger_state(directory / "ledger.jsonl")
        invocations += ledger["invocations"]
        charged += ledger["charged_active_seconds"]
        shard_generation = sum(e["prediction"]["latency_seconds"]
                               for e in journal["completed"].values())
        generation += shard_generation
        latencies += [e["prediction"]["latency_seconds"] for e in journal["completed"].values()]
        runtime = directory / "run" / "runtime.jsonl"
        if runtime.exists():
            loads += [event["load_seconds"]
                      for event in map(json.loads, runtime.read_text().splitlines())
                      if event.get("event") == "load_completed"]
        per_shard.append({
            "shard": shard, "assigned": len(assigned), "completed": len(journal["completed"]),
            "unknown": len(journal["unknown"]), "invocations": ledger["invocations"],
            "charged_active_seconds": ledger["charged_active_seconds"],
            "generation_seconds": shard_generation,
        })
    if set(owner) != set(identifiers) or len(owner) != len(identifiers):
        raise ValueError("Shard assignment does not cover the stage manifest exactly once")
    predictions, failures, finish = {}, 0, {}
    tokens = {"prompt_tokens": 0, "generation_tokens": 0}
    for identifier, event in completed.items():
        prediction = event["prediction"]
        predictions[identifier] = v1.canonical(prediction, inputs["labels"])
        failures += predictions[identifier] is None
        reason = str(prediction.get("finish_reason"))
        finish[reason] = finish.get(reason, 0) + 1
        for key in tokens:
            tokens[key] += (prediction.get("usage") or {}).get(key) or 0
    missing = [i for i in identifiers if i not in completed]
    complete = not missing and not unknown
    report = {
        "plan_id": PLAN_ID, "stage": stage, "shards": shards, "per_shard": per_shard,
        "denominator": len(identifiers), "completed": len(completed),
        "unknown_after_interruption": sorted(unknown), "hard_stop_errors": sorted(hard_stop),
        "unattempted": len(set(identifiers) - set(reserved)),
        "missing_images": len(missing),
        "invalid_or_unterminated_answers": failures, "finish_reasons": finish,
        "native_tokens": tokens, "generation_seconds_total": generation,
        "mean_image_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "max_image_latency_seconds": max(latencies) if latencies else None,
        "mean_load_seconds": sum(loads) / len(loads) if loads else None,
        "invocations": invocations, "charged_active_seconds": charged,
        "ids_sha256": v1.ids_sha256(identifiers),
        "status": "completed" if complete else "partial", "metric": None,
        "test_consumed": False,
    }
    if complete:
        rows = _inputs(inputs["gold"], (predictions,), inputs["labels"])
        score, _ = _score(rows, predictions, inputs["labels"])
        report["metric"] = {
            "name": "mean(diag(row_normalized_confusion))",
            "mean_class_accuracy": score["mean_class_accuracy"],
            "image_accuracy": score["micro_accuracy"],
            "correct": score["correct"], "n": score["n"],
            "invalid_answers_including_missing": score["invalid_answers_including_missing"],
        }
        if stage == "discovery":
            rule = plan["stages"]["discovery"]["uncertainty"]
            half = v1.hoeffding_half_width(score["n"], rule["confidence_level"])
            mean = score["micro_accuracy"]
            report["metric"]["hoeffding"] = {
                "confidence_level": rule["confidence_level"], "half_width": half,
                "interval": [max(0.0, mean - half), min(1.0, mean + half)],
            }
        report["predictions"] = predictions
    return report


def merge(plan, stage, repository, output):
    """The merged predictions, refusing any missing, unknown or duplicate image."""
    report = merged_report(plan, stage, repository, output)
    if report["status"] != "completed":
        raise ValueError(f"Merge refused: {report['missing_images']} missing and "
                         f"{len(report['unknown_after_interruption'])} unknown images")
    return report


def invocation_wall_seconds(budget):
    return v1.invocation_wall_seconds(budget)


def discovery_decision(plan, report):
    """Pre-registered gate from discovery to full VAL, projected per shard (wall clock)."""
    rule, full = plan["stages"]["full"]["gate"], plan["stages"]["full"]
    reasons = []
    if report["status"] != "completed":
        reasons.append("discovery incomplete")
    if report["unknown_after_interruption"] or report["hard_stop_errors"]:
        reasons.append("discovery has unknown or hard-stop outcomes")
    projection = cap = per_invocation = None
    if report["completed"]:
        valid = 1 - report["invalid_or_unterminated_answers"] / report["completed"]
        if valid < rule["minimum_valid_answer_fraction"]:
            reasons.append("valid-answer fraction below the pre-registered floor")
        load = report["mean_load_seconds"]
        if load is None:
            reasons.append("discovery recorded no completed model load")
        elif report["invocations"]:
            overhead = (report["charged_active_seconds"]
                        - report["generation_seconds_total"]) / report["invocations"]
            per_invocation = max(load, overhead)
            images = math.ceil(full["images"] / shard_count(plan))
            invocations = math.ceil(images / full["budget"]["chunk_images"])
            projection = (images * report["mean_image_latency_seconds"]
                          + invocations * per_invocation)
            if projection > rule["max_projected_shard_active_seconds"]:
                reasons.append("measured profile is unaffordable at the declared ceiling")
            cap = min(rule["max_projected_shard_active_seconds"],
                      rule["active_cap_multiplier"] * projection)
            wall = invocation_wall_seconds(full["budget"])
            if (projection <= rule["max_projected_shard_active_seconds"]
                    and projection + wall > cap):
                reasons.append("projection plus one invocation wall cap exceeds the shard "
                               "active-seconds cap; a shard would halt before its last chunk")
        else:
            reasons.append("discovery recorded no invocation in its ledgers")
    return {
        "decision": "stop" if reasons else "proceed", "reasons": reasons,
        "per_invocation_overhead_seconds": per_invocation,
        "projected_shard_active_seconds": projection,
        "shard_max_active_seconds": cap,
        "headroom_gate": "not applied here; recorded for the root decision",
    }


def stage_budget(plan, stage, repository, output):
    budget = dict(plan["stages"][stage]["budget"])
    if stage == "full":
        decision = discovery_decision(plan, merged_report(plan, "discovery", repository,
                                                          output))
        if decision["decision"] != "proceed":
            raise ValueError("Full VAL refused by the discovery gate: "
                             + "; ".join(decision["reasons"]))
        budget["max_active_seconds"] = decision["shard_max_active_seconds"]
    budget["invocation_wall_seconds"] = invocation_wall_seconds(budget)
    return budget


def run_shard(plan_path, expected_sha256, stage, shard, repository, output, weights, threads,
              *, backend_factory=None, clock=time.time):
    """One bounded invocation of one shard of one stage."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    plan = load_plan(plan_path, expected_sha256)
    plan_sha = local_vlm.sha_file(plan_path)
    sources = source_hashes()
    if any(sources[name] != digest for name, digest in plan["sources_pinned"].items()):
        raise ValueError("A pinned runner source changed")
    shards = shard_count(plan)
    if stage == "full":
        for number in range(shards):
            ledger = v1.ledger_state(shard_directory(output, "discovery", number, shards)
                                     / "ledger.jsonl")
            if not ledger["started"]:
                raise ValueError("Full VAL refused by the discovery gate: discovery shard "
                                 f"{number} never ran")
            if ledger["unfinished"]:
                raise ValueError("Discovery has an unfinished invocation; review first")
            if any(event["plan_sha256"] != plan_sha or event["sources"] != sources
                   for event in ledger["started"]):
                raise ValueError("Discovery ran under a different plan or source; "
                                 "full VAL needs discovery under this exact plan and code")
    inputs = stage_inputs(plan, stage, repository, shard)
    identifiers = [row["id"] for row in inputs["rows"]]
    budget = stage_budget(plan, stage, repository, output)
    directory = shard_directory(output, stage, shard, shards)
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path, run_directory = directory / "ledger.jsonl", directory / "run"
    ledger = v1.ledger_state(ledger_path)
    if ledger["first"] is not None and (
        ledger["first"]["plan_sha256"] != plan_sha or ledger["first"]["sources"] != sources
    ):
        raise ValueError("Plan or source changed since this shard's first invocation")
    journal = v1.journal_state(run_directory, identifiers)
    if journal["complete"]:
        return merged_report(plan, stage, repository, output)
    halts = []
    if ledger["unfinished"]:
        halts.append("previous invocation never recorded its finish")
    if journal["unknown"] or journal["hard_stop"]:
        halts.append("unknown or hard-stop outcome recorded; human review required")
    if ledger["invocations"] >= budget["max_invocations"]:
        halts.append("invocation budget exhausted")
    charged, wall = ledger["charged_active_seconds"], budget["invocation_wall_seconds"]
    if charged + wall > budget["max_active_seconds"]:
        halts.append(f"active-seconds budget exhausted: charged {charged:.1f} s + invocation "
                     f"wall cap {wall} s > {budget['max_active_seconds']} s")
    if ledger["first"] and clock() - ledger["first"]["time"] >= budget["max_calendar_seconds"]:
        halts.append("calendar budget exhausted")
    if run_directory.exists() and v1._output_bytes(run_directory) >= budget["max_output_bytes"]:
        halts.append("output-bytes budget exhausted")
    if halts:
        raise ValueError("Shard halted: " + "; ".join(halts))
    number = ledger["invocations"] + 1
    v1._append(ledger_path, {
        "event": "started", "invocation": number, "time": clock(), "stage": stage,
        "shard": shard, "shards": shards, "plan_sha256": plan_sha, "sources": sources,
        "wall_cap_seconds": budget["invocation_wall_seconds"],
        "completed_before": len(journal["completed"]), "threads": threads,
    })
    model, engine = plan["model"], plan["engine"]
    factory = backend_factory or local_hf.TransformersBackend
    started = time.monotonic()
    try:
        local_hf.run_local_hf(
            inputs["rows"], run_directory, model_directory=Path(weights),
            integrity=repository / model["integrity"],
            engine={"name": engine["name"], "dtype": engine["dtype"],
                    "formatted_prompt_sha256": plan["prompt_and_parser"][
                        "formatted_prompt_sha256"]},
            runtime_pins=model["runtime_pins"], phase="validation",
            seed=plan["generation"]["seed"], resume=run_directory.exists(),
            backend_factory=functools.partial(_with_threads, factory, threads),
            image_root=inputs["image_root"], task_contract=inputs["task_contract"],
            preprocessing_id=PREPROCESSING_ID,
            stop_after_completions=min(budget["chunk_images"], len(journal["remaining"])),
            stream_progress=True, profile=plan["generation"]["profile"],
        )
    finally:
        v1._append(ledger_path, {"event": "finished", "invocation": number,
                                 "elapsed_seconds": time.monotonic() - started})
    report = merged_report(plan, stage, repository, output)
    if stage == "discovery" and report["status"] == "completed":
        report["decision"] = discovery_decision(plan, report)
    report.pop("predictions", None)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _with_threads(factory, threads, directory, config):
    # Thread count is operational (it changes latency, not outputs), so it stays out of
    # the run identity and is recorded in each invocation's ledger start and load_info.
    return factory(directory, {**config, "engine": {**config["engine"], "threads": threads}})


def image_accuracy_bootstrap(rows, candidate, comparator, labels, resamples, seed):
    """Paired image bootstrap of the image-accuracy difference over the whole denominator."""
    rows = _inputs(rows, (candidate, comparator), labels)
    first, a = _score(rows, candidate, labels)
    second, b = _score(rows, comparator, labels)
    delta = (a - b).astype(np.int64)
    counts = np.bincount(delta + 1, minlength=3)
    draws = np.random.default_rng(seed).multinomial(len(delta), counts / len(delta),
                                                    size=resamples)
    resampled = (draws[:, 2] - draws[:, 0]) / len(delta)
    return {
        "candidate_image_accuracy": first["micro_accuracy"],
        "comparator_image_accuracy": second["micro_accuracy"],
        "image_accuracy_difference": float(delta.mean()),
        "draws": resamples, "seed": seed,
        "ci95": np.quantile(resampled, [0.025, 0.975]).tolist(),
        "method": "paired image bootstrap, exact ternary-outcome multinomial, "
                  "numpy default_rng(seed), linear percentile",
    }


def compare(plan, repository, output, against):
    """Baseline (candidate) versus each declared arm on the complete native VAL."""
    report = merge(plan, "full", repository, output)
    inputs = stage_inputs(plan, "full", repository)
    rule = plan["comparison"]
    declared = {arm["name"] for arm in rule["arms"]}
    if set(against) != declared:
        raise ValueError(f"Compare needs exactly the declared arms {sorted(declared)}")
    results = {}
    for name, path in sorted(against.items()):
        predictions = v1.read_json(path)
        if set(predictions) != {row["id"] for row in inputs["gold"]}:
            raise ValueError(f"Arm {name} does not cover the native VAL manifest exactly")
        results[name] = {
            "equal_class": compare_classification(
                inputs["gold"], report["predictions"], predictions, inputs["labels"],
                resamples=rule["resamples"], seed=rule["seed"]),
            "image": image_accuracy_bootstrap(
                inputs["gold"], report["predictions"], predictions, inputs["labels"],
                rule["resamples"], rule["seed"]),
        }
    return {"plan_id": PLAN_ID, "candidate": "named baseline v2 (merged shards)",
            "against": results, "test_consumed": False}


def export_arm(bundle, arm):
    """Canonical {id: label|None} of one evaluator arm, re-parsed by the frozen rule."""
    from vision_rl import vlm_local_evaluation as evaluation

    bundle = Path(bundle).resolve()
    plan = evaluation.read(bundle / "plan.json")
    images = evaluation.read(bundle / "images.json")
    declared = {entry["id"]: entry for entry in plan["study"]["arms"]}
    if arm not in declared:
        raise ValueError(f"Arm {arm} is not declared by the evaluator bundle")
    journal = evaluation._journal(bundle, plan, declared[arm], images)
    if journal.pending is not None or len(journal.results) != len(images):
        raise ValueError("Arm journal is incomplete")
    out = {}
    for identifier, result in journal.results.items():
        parsed = (parse_label(result["response"], plan["labels"], allow_reasoning=True)
                  if result["status"] == "completed" and result["terminated"] else None)
        if parsed != result["canonical_label"]:
            raise ValueError("Canonical prediction differs from the frozen raw-output parser")
        out[identifier] = parsed
    return out


def smoke(repository, weights, integrity, ids, data_spec, threads, dtype, runtime_pins,
          formatted_prompt_sha256, *, backend_factory=None, progress_path=None):
    """Latency smoke on TRAIN images only; no journal, never a score."""
    repository = Path(repository).resolve()
    train_root = (repository / "data/fgvc-aircraft-2013b/development/images/train").resolve()
    spec = load_spec(data_spec)
    prompt = prompt_text(spec, "answer_only")
    model = local_hf.verify_snapshot(weights, integrity)
    versions = local_hf.runtime_versions(runtime_pins)
    config = {
        **local_vlm.native_profile("instruct"), "seed": 1729,
        "engine": {"name": "transformers-cpu", "dtype": dtype, "threads": threads,
                   "formatted_prompt_sha256": formatted_prompt_sha256},
        "preprocessing_id": PREPROCESSING_ID, "stream_progress": progress_path is not None,
        "generation_progress_path": str(progress_path), "identity_sha256": None,
    }
    started = time.monotonic()
    backend = (backend_factory or local_hf.TransformersBackend)(weights, config)
    results = []
    for identifier in ids:
        image = (train_root / f"{identifier}.jpg").resolve(strict=True)
        if not image.is_relative_to(train_root):
            raise ValueError("Smoke images must be TRAIN images")
        seed = int(local_vlm._digest([1729, identifier])[:8], 16)
        start = time.monotonic()
        native = backend.infer({"id": identifier, "image_path": str(image), "prompt": prompt},
                               seed)
        latency = time.monotonic() - start
        response = native["native_response"]
        answer = local_vlm.final_answer(response["text"], response["finish_reason"],
                                        profile="instruct")
        results.append({
            "id": identifier, "latency_seconds": latency, "text": response["text"],
            "finish_reason": response["finish_reason"],
            "parsed": parse_label(response["text"], spec["labels"], allow_reasoning=True)
            if response["finish_reason"] == "stop" else None,
            "final_answer_error": answer.get("error"),
            "prompt_tokens": response["prompt_tokens"],
            "generation_tokens": response["generation_tokens"],
            "prefill_seconds": response.get("prefill_seconds"),
            "max_rss_gb": response["peak_memory"],
            **{key: native["preprocessing"][key] for key in (
                "formatted_prompt_sha256", "input_ids_shape", "pixel_values_shape",
                "image_grid_thw")},
        })
        print(json.dumps(results[-1]), flush=True)
    latencies = [r["latency_seconds"] for r in results]
    return {
        "kind": "named-baseline-v2-smoke", "split": "train", "images": len(results),
        "model": {"repository": model["repository"], "revision": model["revision"]},
        "runtime_versions": versions, "load_info": backend.load_info,
        "wall_seconds_including_load": time.monotonic() - started,
        "mean_image_seconds": sum(latencies) / len(latencies) if latencies else None,
        "results": results, "not_a_score": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report", "merge", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--stage", choices=STAGES, required=name != "compare")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--repository", type=Path, default=HERE.parent)
        if name == "run":
            command.add_argument("--expected-plan-sha256", required=True)
            command.add_argument("--shard", type=int, required=True)
            command.add_argument("--weights", type=Path, required=True)
            command.add_argument("--threads", type=int, required=True)
        if name == "compare":
            command.add_argument("--against", action="append", required=True,
                                 metavar="NAME=PATH",
                                 help="declared arm name = JSON {id: label or null}")
    export = commands.add_parser("export-arm")
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--arm", required=True)
    export.add_argument("--out", type=Path, required=True)
    smoke_command = commands.add_parser("smoke")
    smoke_command.add_argument("--plan", type=Path, required=True)
    smoke_command.add_argument("--weights", type=Path, required=True)
    smoke_command.add_argument("--ids", nargs="+", required=True)
    smoke_command.add_argument("--threads", type=int, required=True)
    smoke_command.add_argument("--dtype", choices=local_hf.DTYPES, required=True)
    smoke_command.add_argument("--out", type=Path, required=True)
    smoke_command.add_argument("--repository", type=Path, default=HERE.parent)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_shard(args.plan, args.expected_plan_sha256, args.stage, args.shard,
                               args.repository, args.output, args.weights, args.threads)
        elif args.command == "export-arm":
            args.out.write_text(json.dumps(export_arm(args.bundle, args.arm), indent=1) + "\n")
            return 0
        elif args.command == "smoke":
            plan = load_plan(args.plan, require_activated=False)
            inputs = plan["inputs"]
            result = smoke(
                args.repository, args.weights, args.repository / plan["model"]["integrity"],
                args.ids,
                args.repository / inputs["dataset_root"] / inputs["data_spec"]["path"],
                args.threads, args.dtype, plan["model"]["runtime_pins"],
                plan["prompt_and_parser"]["formatted_prompt_sha256"],
                progress_path=args.out.with_suffix(".progress.jsonl"))
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        else:
            plan = load_plan(args.plan, require_activated=False)
            if args.command == "compare":
                against = dict(item.split("=", 1) for item in args.against)
                result = compare(plan, args.repository, args.output, against)
            elif args.command == "merge":
                result = merge(plan, args.stage, args.repository, args.output)
                (args.output / f"stage-{args.stage}" / "merged-predictions.json").write_text(
                    json.dumps(result["predictions"], indent=1) + "\n")
            else:
                result = merged_report(plan, args.stage, args.repository, args.output)
                if args.stage == "discovery":
                    result["decision"] = discovery_decision(plan, result)
        result.pop("predictions", None)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"named baseline v2: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
