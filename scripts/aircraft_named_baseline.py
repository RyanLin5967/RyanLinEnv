"""Named open-weight baseline on FGVC-Aircraft native VAL: Qwen3.8-27B, native instruct.

Pre-registered by docs/aircraft-named-baseline-plan-v1.json (mlx-community 8-bit, MLX on
Apple silicon) and docs/aircraft-named-baseline-plan-v2.json (the original Hugging Face
bfloat16 weights, transformers on CPU, sharded across hosts). The plan's ``model.engine``
selects the engine; everything else below is one code path. Committed plans are
UNACTIVATED and this runner refuses them: a run needs a separately activated copy whose
SHA256 the operator passes explicitly. Two stages, in order:

  discovery  the protocol-defined 100-image, one-per-class native-VAL sample
             (strong_baseline.validation_discovery in aircraft-rl-protocol-freshpool-v1)
  full       all 3,333 native-VAL images in author order, only after the discovery
             decision computed from the discovery journal says "proceed"

Inference reuses the validated journaled runner ``vision_rl.local_vlm.run_local``
(durable reservation before generation, an interrupted reservation is an unknown outcome
and is never regenerated, completed answers are never repeated). The prompt is the
eight-endpoint evaluator's: ``prompt_text(spec, "answer_only")`` with the complete
100-label vocabulary. The parser is the same ``parse_label(..., allow_reasoning=True)``,
applied only to a normally terminated generation. The metric is the same
``classification_comparison`` equal-class accuracy, with invalid and missing answers
kept in their class denominators.

This runner owns only what run_local lacks: stage manifests, an append-only budget
ledger whose original clock and invocation count survive resume, and halt rules. A plan
with a ``sharding`` block splits each stage's fixed author-order manifest round-robin into
N shards (``ids[k::N]``), each with its own journal and ledger under
stage-<stage>/shard-<k>-of-<N>/; the report merges them and refuses any image recorded in
two shards, in the wrong shard, or (for a metric or ``merge``) missing.
Per-invocation wall and memory limits are enforced by the external supervisor
(scripts/supervise_local_vlm.py), whose limits the plan declares.

Blind spots, stated here on purpose: the ledger is local state, not a tamper-proof
witness; an invocation killed with SIGKILL leaves no "finished" record and is charged
its full declared wall cap; the runner does not detect other accelerator processes. The
active-seconds cap is a hard ceiling only because an invocation starts when charged + its wall
cap fits and the external supervisor kills it at that wall cap; run without the supervisor, a
finished invocation is charged its real elapsed seconds, which may exceed the wall cap. The
output-bytes limit is checked before each invocation, so one invocation can overshoot it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

from vision_rl import local_vlm, local_vlm_hf
from vision_rl.aircraft_data import PREPROCESSING_ID
from vision_rl.classification_comparison import _inputs, _score, compare_classification
from vision_rl.vlm_local_training import OUTPUT_GRAMMAR, load_spec, parse_label, prompt_text

PLAN_ID = "aircraft-named-baseline-qwen3.8-27b-8bit-v1"
PLAN_ID_V2 = "aircraft-named-baseline-qwen3.8-27b-bf16-cpu-v2"
# plan_id -> (schema_version, engine). The engine is fixed by the plan identity.
PLANS = {PLAN_ID: (1, "mlx"), PLAN_ID_V2: (2, "transformers-cpu")}
SHARD_ASSIGNMENT = "author-order-round-robin: shard k of N takes ids[k::N]"
STAGES = ("discovery", "full")
NATIVE_IMAGES, NATIVE_CLASSES = 3333, 100
# strong_baseline.validation_discovery.selection; salt pinned by decision D13.
DISCOVERY_SALT = (
    "fgvc-aircraft-2013b",
    "fgvc-aircraft-freshpool-screened-grpo-v1",
    "baseline-discovery",
    791,
)
SOURCE_MODULES = {
    "src/vision_rl/local_vlm.py": "vision_rl.local_vlm",
    "src/vision_rl/local_vlm_execution.py": "vision_rl.local_vlm_execution",
    "src/vision_rl/aircraft_data.py": "vision_rl.aircraft_data",
    "src/vision_rl/frontier.py": "vision_rl.frontier",
    "src/vision_rl/vlm_local_training.py": "vision_rl.vlm_local_training",
    "src/vision_rl/classification_comparison.py": "vision_rl.classification_comparison",
    "src/vision_rl/local_vlm_hf.py": "vision_rl.local_vlm_hf",
}


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ids_sha256(ids):
    return sha_text(compact(list(ids)))


def parser_source_sha256():
    return sha_text(inspect.getsource(parse_label))


def read_json(path):
    return json.loads(Path(path).read_text())


def select_discovery(author_rows, labels):
    """Lowest SHA256 of compact [salt..., label, id] per class; returned in author order.

    Uses metadata only: no image is decoded and no model output exists yet.
    """
    labels = list(labels)
    if len(labels) != NATIVE_CLASSES or len(set(labels)) != NATIVE_CLASSES:
        raise ValueError("Discovery needs the complete 100-label vocabulary")
    identifiers = [row["image_id"] for row in author_rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Native VAL image IDs must be unique")
    if any(row["split"] != "val" for row in author_rows):
        raise ValueError("Discovery selects native VAL images only")
    if {row["variant"] for row in author_rows} != set(labels):
        raise ValueError("Native VAL must cover exactly the declared classes")
    best = {}
    for row in author_rows:
        key = sha_text(compact([*DISCOVERY_SALT, row["variant"], row["image_id"]]))
        if row["variant"] not in best or key < best[row["variant"]][0]:
            best[row["variant"]] = (key, row["image_id"])
    chosen = {identifier for _, identifier in best.values()}
    return [identifier for identifier in identifiers if identifier in chosen]


def load_plan(path, expected_sha256=None, *, require_activated=True):
    path = Path(path)
    plan = read_json(path)
    schema, engine = PLANS.get(plan.get("plan_id"), (None, None))
    if schema is None or plan.get("schema_version") != schema:
        raise ValueError("Not the named-baseline plan schema")
    if plan["model"].get("engine", "mlx") != engine:
        raise ValueError("Plan engine differs from its plan identity")
    count = shard_count(plan)
    if engine == "mlx" and count is not None:
        raise ValueError("The MLX engine is single-machine; it takes no sharding")
    if require_activated:
        activation = plan.get("activation")
        if (
            plan.get("activated") is not True
            or not isinstance(activation, dict)
            or not activation.get("activated_at")
            or not activation.get("approved_by")
        ):
            raise ValueError("Named-baseline plan is unactivated; it must not run")
        if expected_sha256 is None or local_vlm.sha_file(path) != expected_sha256:
            raise ValueError("Pass the activated plan's exact SHA256 to run it")
    return plan


def engine_of(plan):
    return PLANS[plan["plan_id"]][1]


def shard_count(plan):
    """None for an unsharded plan; otherwise the pinned positive shard count."""
    sharding = plan.get("sharding")
    if sharding is None:
        return None
    count = sharding.get("shards")
    if sharding.get("assignment") != SHARD_ASSIGNMENT or type(count) is not int or count < 1:
        raise ValueError("Unknown shard assignment")
    return count


def parse_shard(text):
    index, _, count = str(text).partition("/")
    if not (index.isdigit() and count.isdigit()) or not 0 <= int(index) < int(count):
        raise ValueError("--shard must be k/N with 0 <= k < N")
    return int(index), int(count)


def stage_units(plan, stage, identifiers, output):
    """(name, directory, ids) per execution unit: one for an unsharded plan, else N."""
    base = Path(output) / f"stage-{stage}"
    count = shard_count(plan)
    if count is None:
        return [(None, base, list(identifiers))]
    return [(f"shard-{k}-of-{count}", base / f"shard-{k}-of-{count}", identifiers[k::count])
            for k in range(count)]


def _within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"{path} escapes {root}")
    return path


def checked_prompt(plan, data_spec):
    """The evaluator's answer-only prompt and vocabulary, checked against the plan pins."""
    spec = load_spec(data_spec)
    labels = spec["labels"]
    prompt = prompt_text(spec, "answer_only")
    contract = plan["prompt_and_parser"]
    if (
        len(labels) != NATIVE_CLASSES
        or sha_text(prompt) != contract["prompt_sha256"]
        or contract["response_format"] != "answer_only"
        or OUTPUT_GRAMMAR != contract["output_grammar"]
        or parser_source_sha256() != contract["parser_source_sha256"]
    ):
        raise ValueError("Prompt, vocabulary or parser differs from the eight-endpoint evaluator")
    return labels, prompt


def stage_inputs(plan, stage, repository):
    """Build the fixed request rows and gold labels; every pinned input is rehashed."""
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    repository = Path(repository).resolve()
    inputs = plan["inputs"]
    dataset = _within(repository / inputs["dataset_root"], repository)
    paths = {}
    for name in ("val_images", "validation", "data_spec"):
        paths[name] = _within(dataset / inputs[name]["path"], dataset)
        if local_vlm.sha_file(paths[name]) != inputs[name]["sha256"]:
            raise ValueError(f"Pinned input changed: {name}")
    labels, prompt = checked_prompt(plan, paths["data_spec"])
    contract = plan["prompt_and_parser"]
    author = read_json(paths["val_images"])
    published = [
        json.loads(line) for line in paths["validation"].read_text().splitlines() if line.strip()
    ]
    if len(author) != NATIVE_IMAGES or len(published) != NATIVE_IMAGES:
        raise ValueError("Require all 3333 native VAL images")
    image_root = _within(dataset / "development/images/validation", dataset)
    gold = {}
    for row, source in zip(published, author, strict=True):
        if (
            row.get("split") != "validation"
            or source.get("split") != "val"
            or row.get("id") != source.get("image_id")
            or row.get("label") != source.get("variant")
            or source.get("path") != f"development/images/validation/{row['id']}.jpg"
        ):
            raise ValueError("Published VAL rows differ from author membership, order or labels")
        gold[row["id"]] = row["label"]
    if set(gold.values()) != set(labels):
        raise ValueError("Native VAL labels differ from the vocabulary")
    if stage == "discovery":
        identifiers = select_discovery(author, labels)
        pinned = plan["stages"]["discovery"]
        if identifiers != pinned["ids"] or ids_sha256(identifiers) != pinned["ids_sha256"]:
            raise ValueError("Discovery manifest differs from the pre-registered IDs")
    else:
        identifiers = [row["image_id"] for row in author]
        if ids_sha256(identifiers) != plan["stages"]["full"]["ids_sha256"]:
            raise ValueError("Full native VAL order differs from the pre-registered manifest")
    by_id = {row["image_id"]: row for row in author}
    rows = []
    for identifier in identifiers:
        image = _within(image_root / f"{identifier}.jpg", image_root)
        if local_vlm.sha_file(image) != by_id[identifier]["encoded_sha256"]:
            raise ValueError(f"Native VAL JPEG bytes changed: {identifier}")
        rows.append({"id": identifier, "image": str(image), "split": "validation",
                     "prompt": prompt})
    task_contract = {
        "task": "classification",
        "benchmark": "FGVC-Aircraft 2013b native VAL",
        "plan_id": plan["plan_id"],
        "stage": stage,
        "images": len(rows),
        "labels": labels,
        "response_format": "answer_only",
        "prompt_sha256": contract["prompt_sha256"],
        "grammar": {"name": OUTPUT_GRAMMAR, "parser_source_sha256": parser_source_sha256()},
    }
    return {
        "rows": rows,
        "gold": [{"id": identifier, "label": gold[identifier]} for identifier in identifiers],
        "labels": labels,
        "image_root": image_root,
        "task_contract": task_contract,
    }


def source_hashes():
    """Hash the code that actually executes, wherever it was imported from."""
    hashes = {
        name: local_vlm.sha_file(importlib.import_module(module).__file__)
        for name, module in SOURCE_MODULES.items()
    }
    hashes["scripts/aircraft_named_baseline.py"] = local_vlm.sha_file(__file__)
    return hashes


def _append(path, value):
    with Path(path).open("a") as handle:
        handle.write(compact(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ledger_state(path):
    """Replay the runner ledger. Malformed history fails closed."""
    started, finished = {}, {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            event = json.loads(line)
            number = event["invocation"]
            if event["event"] == "started":
                if number in started or number != len(started) + 1:
                    raise ValueError("Budget ledger is reordered or repeated")
                started[number] = event
            elif event["event"] == "finished":
                if number not in started or number in finished:
                    raise ValueError("Budget ledger finish lacks its start")
                finished[number] = event
            else:
                raise ValueError("Unknown budget ledger event")
    charged = sum(
        finished[n]["elapsed_seconds"] if n in finished else started[n]["wall_cap_seconds"]
        for n in started
    )
    return {
        "invocations": len(started),
        "first": started.get(1),
        "started": [started[n] for n in sorted(started)],
        "charged_active_seconds": charged,
        "unfinished": sorted(set(started) - set(finished)),
    }


def journal_state(run_directory, identifiers):
    """Read run_local's attempts journal without modifying it."""
    reserved, completed = {}, {}
    path = Path(run_directory) / "attempts.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "reserved":
                reserved[event["id"]] = event
            elif event["event"] == "completed":
                completed[event["id"]] = event
    unknown = sorted(set(reserved) - set(completed))
    hard_stop = sorted(k for k, v in completed.items() if v.get("hard_stop"))
    return {
        "reserved": reserved,
        "completed": completed,
        "unknown": unknown,
        "hard_stop": hard_stop,
        "remaining": [i for i in identifiers if i not in reserved],
        "complete": set(completed) == set(identifiers) and not unknown,
    }


def canonical(prediction, labels):
    """The evaluator's rule: parse only a normally terminated generation."""
    if prediction.get("finish_reason") != "stop" or "classification_response" not in prediction:
        return None
    return parse_label(prediction["classification_response"], labels, allow_reasoning=True)


def hoeffding_half_width(n, confidence):
    return math.sqrt(math.log(2 / (1 - confidence)) / (2 * n))


def _output_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


def _unit_identity_ids(directory):
    run = Path(directory) / "run" / "run.json"
    if not run.exists():
        return None
    return [row["id"] for row in json.loads(run.read_text())["identity"]["rows"]]


def merge_units(plan, stage, identifiers, output):
    """Replay every unit's journal and ledger; refuse duplicates and misplaced images.

    An image reserved in two units, reserved in a unit that does not own it, or a unit
    whose run identity binds rows other than its assignment is refused outright. Missing
    images are reported (unattempted), never invented; a metric or a merge requires none.
    """
    units = stage_units(plan, stage, identifiers, output)
    base = Path(output) / f"stage-{stage}"
    if shard_count(plan) is not None and base.exists():
        expected = {name for name, _, _ in units}
        stray = sorted(p.name for p in base.iterdir()
                       if p.name.startswith("shard-") and p.name not in expected)
        if stray:
            raise ValueError(f"Merge refused: shard directories outside the plan: {stray}")
    reserved, completed, owner, per_unit = {}, {}, {}, []
    for name, directory, ids in units:
        bound = _unit_identity_ids(directory)
        if bound is not None and bound != ids:
            raise ValueError(f"Merge refused: {name or 'stage'} binds rows other than its "
                             "assignment")
        journal = journal_state(directory / "run", ids)
        foreign = sorted(set(journal["reserved"]) - set(ids))
        if foreign:
            raise ValueError(f"Merge refused: {name} holds images it does not own: "
                             f"{foreign[:5]}")
        for identifier, event in journal["reserved"].items():
            if identifier in owner:
                raise ValueError(f"Merge refused: image {identifier} recorded in both "
                                 f"{owner[identifier]} and {name}")
            owner[identifier] = name
            reserved[identifier] = event
        completed.update(journal["completed"])
        ledger = ledger_state(directory / "ledger.jsonl")
        loads = []
        runtime = directory / "run" / "runtime.jsonl"
        if runtime.exists():
            loads = [event["load_seconds"]
                     for event in map(json.loads, runtime.read_text().splitlines())
                     if event.get("event") == "load_completed"]
        latencies = [journal["completed"][i]["prediction"]["latency_seconds"]
                     for i in journal["completed"]]
        per_unit.append({
            "unit": name, "images": len(ids), "completed": len(journal["completed"]),
            "unknown_after_interruption": journal["unknown"],
            "hard_stop_errors": journal["hard_stop"],
            "unattempted": len(journal["remaining"]),
            "generation_seconds_total": sum(latencies),
            "mean_image_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
            "loads": loads,
            "mean_load_seconds": sum(loads) / len(loads) if loads else None,
            "invocations": ledger["invocations"],
            "charged_active_seconds": ledger["charged_active_seconds"],
            "unfinished_invocations": ledger["unfinished"],
            "hosts": sorted({event["host"] for event in ledger["started"] if "host" in event}),
        })
    unknown = sorted(set(reserved) - set(completed))
    hard_stop = sorted(k for k, v in completed.items() if v.get("hard_stop"))
    return {
        "reserved": reserved, "completed": completed, "unknown": unknown,
        "hard_stop": hard_stop, "remaining": [i for i in identifiers if i not in reserved],
        "complete": set(completed) == set(identifiers) and not unknown,
        "units": per_unit,
    }


def stage_report(plan, stage, repository, output):
    """Score a stage from its journal(s). Metric is null unless every image completed."""
    inputs = stage_inputs(plan, stage, repository)
    identifiers = [row["id"] for row in inputs["rows"]]
    journal = merge_units(plan, stage, identifiers, output)
    units = journal["units"]
    predictions, latencies, failures, finish = {}, [], 0, {}
    tokens = {"prompt_tokens": 0, "generation_tokens": 0}
    for identifier, event in journal["completed"].items():
        prediction = event["prediction"]
        predictions[identifier] = canonical(prediction, inputs["labels"])
        failures += predictions[identifier] is None
        reason = str(prediction.get("finish_reason"))
        finish[reason] = finish.get(reason, 0) + 1
        latencies.append(prediction["latency_seconds"])
        for key in tokens:
            tokens[key] += (prediction.get("usage") or {}).get(key) or 0
    loads = [load for unit in units for load in unit["loads"]]
    report = {
        "plan_id": plan["plan_id"],
        "stage": stage,
        "denominator": len(identifiers),
        "completed": len(journal["completed"]),
        "unknown_after_interruption": journal["unknown"],
        "hard_stop_errors": journal["hard_stop"],
        "unattempted": len(journal["remaining"]),
        "invalid_or_unterminated_answers": failures,
        "finish_reasons": finish,
        "native_tokens": tokens,
        "generation_seconds_total": sum(latencies),
        "mean_image_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "max_image_latency_seconds": max(latencies) if latencies else None,
        "mean_load_seconds": sum(loads) / len(loads) if loads else None,
        "invocations": sum(unit["invocations"] for unit in units),
        "charged_active_seconds": sum(unit["charged_active_seconds"] for unit in units),
        "ids_sha256": ids_sha256(identifiers),
        "status": "completed" if journal["complete"] else "partial",
        "metric": None,
        "test_consumed": False,
    }
    if shard_count(plan) is not None:
        report["shards"] = [{k: v for k, v in unit.items() if k != "loads"} for unit in units]
    if journal["complete"]:
        rows = _inputs(inputs["gold"], (predictions,), inputs["labels"])
        score, _ = _score(rows, predictions, inputs["labels"])
        report["metric"] = {
            "name": "mean(diag(row_normalized_confusion))",
            "mean_class_accuracy": score["mean_class_accuracy"],
            "micro_accuracy": score["micro_accuracy"],
            "correct": score["correct"],
            "n": score["n"],
            "invalid_answers_including_missing": score["invalid_answers_including_missing"],
        }
        if stage == "discovery":
            rule = plan["stages"]["discovery"]["uncertainty"]
            half = hoeffding_half_width(score["n"], rule["confidence_level"])
            mean = score["micro_accuracy"]
            report["metric"]["hoeffding"] = {
                "confidence_level": rule["confidence_level"],
                "half_width": half,
                "interval": [max(0.0, mean - half), min(1.0, mean + half)],
            }
        report["predictions"] = predictions
    return report


def discovery_decision(plan, report):
    """Pre-registered gate from discovery to full VAL; never weakens the baseline."""
    rule = plan["stages"]["full"]["gate"]
    full = plan["stages"]["full"]
    reasons = []
    if report["status"] != "completed":
        reasons.append("discovery incomplete")
    if report["unknown_after_interruption"] or report["hard_stop_errors"]:
        reasons.append("discovery has unknown or hard-stop outcomes")
    projection = cap = per_invocation = None
    count = shard_count(plan) or 1
    # One entry per execution unit. Shards run in parallel on different hosts, so the
    # full-stage projection is one shard's wall time at the WORST discovery unit's
    # latency and overhead; unsharded, this is exactly the single-unit rule.
    units = report.get("shards") or [report]
    if report["completed"]:
        valid = 1 - report["invalid_or_unterminated_answers"] / report["completed"]
        if valid < rule["minimum_valid_answer_fraction"]:
            reasons.append("valid-answer fraction below the pre-registered floor")
        images = math.ceil(full["images"] / count)
        invocations = math.ceil(images / full["budget"]["chunk_images"])
        # Per-invocation cost outside image latency: the larger of the measured model load
        # and the ledger's charged seconds not spent generating (load, integrity rehash,
        # input rehash, journal replay). A measured 0.0 is a measurement, not missing.
        loads = [unit["mean_load_seconds"] for unit in units]
        if any(load is None for load in loads):
            reasons.append("discovery recorded no completed model load")
        elif any(unit["mean_image_latency_seconds"] is None for unit in units):
            reasons.append("a discovery unit recorded no completed image")
        elif all(unit["invocations"] for unit in units):
            load = max(loads)
            overhead = max((unit["charged_active_seconds"] - unit["generation_seconds_total"])
                           / unit["invocations"] for unit in units)
            latency = max(unit["mean_image_latency_seconds"] for unit in units)
            per_invocation = max(load, overhead)
            projection = images * latency + invocations * per_invocation
            if projection > rule["max_projected_active_seconds"]:
                reasons.append("measured profile is unaffordable at the declared ceiling")
            cap = min(rule["max_projected_active_seconds"],
                      rule["active_cap_multiplier"] * projection)
            # Consistency with the hard ceiling in run_stage: an invocation starts only if
            # charged + its full wall cap fits, so a run that costs exactly the projection
            # reaches its last start with up to P charged. Proceed only if P + one full
            # invocation wall cap fits the cap, or the run would halt short of 3,333.
            wall = invocation_wall_seconds(full["budget"])
            if projection <= rule["max_projected_active_seconds"] and projection + wall > cap:
                reasons.append("projection plus one invocation wall cap exceeds the full "
                               "active-seconds cap; the hard ceiling would halt full VAL "
                               "before its last chunk")
        else:
            reasons.append("discovery recorded no invocation in its ledger")
    return {
        "decision": "stop" if reasons else "proceed",
        "reasons": reasons,
        "per_invocation_overhead_seconds": per_invocation,
        "projected_full_active_seconds": projection,
        "full_max_active_seconds": cap,
        "headroom_gate": "not applied here; recorded for the root decision",
        "projection_scope": "per shard (parallel units)" if count > 1 else "single unit",
    }


def stage_budget(plan, stage, repository, output):
    budget = dict(plan["stages"][stage]["budget"])
    if stage == "full":
        decision = discovery_decision(
            plan, stage_report(plan, "discovery", repository, output)
        )
        if decision["decision"] != "proceed":
            raise ValueError("Full VAL refused by the discovery gate: "
                             + "; ".join(decision["reasons"]))
        budget["max_active_seconds"] = decision["full_max_active_seconds"]
    budget["invocation_wall_seconds"] = invocation_wall_seconds(budget)
    return budget


def invocation_wall_seconds(budget):
    return (budget["load_allowance_seconds"]
            + budget["chunk_images"] * budget["image_seconds"]
            + budget["long_generation_allowance_seconds"])


def run_stage(plan_path, expected_sha256, stage, repository, output, *, backend_factory=None,
              clock=time.time, shard=None, model_directory=None):
    """One bounded invocation of one stage (or one shard of it).

    Refuses unactivated plans, spent budgets, and a shard argument that does not match
    the plan's sharding. ``model_directory`` overrides where the snapshot lives on this
    host; the files are still rehashed against the plan-pinned integrity record.
    """
    repository, output = Path(repository).resolve(), Path(output).resolve()
    plan = load_plan(plan_path, expected_sha256)
    count = shard_count(plan)
    if (count is None) != (shard is None) or (shard is not None and shard[1] != count):
        raise ValueError(f"This plan runs as {count} shards; pass --shard k/{count}"
                         if count else "This plan is unsharded; do not pass --shard")
    if model_directory is not None and engine_of(plan) == "mlx":
        raise ValueError("The MLX engine loads only the plan's model directory")
    plan_sha = local_vlm.sha_file(plan_path)
    sources = source_hashes()
    pinned = plan["sources_pinned"]
    if any(sources[name] != digest for name, digest in pinned.items()):
        raise ValueError("A pinned runner source changed")
    inputs = stage_inputs(plan, stage, repository)
    all_identifiers = [row["id"] for row in inputs["rows"]]
    if stage == "full":
        for name, unit, _ in stage_units(plan, "discovery", plan["stages"]["discovery"]["ids"],
                                         output):
            discovery_ledger = ledger_state(unit / "ledger.jsonl")
            where = f" ({name})" if name else ""
            if not discovery_ledger["started"]:
                raise ValueError("Full VAL refused by the discovery gate: discovery never "
                                 f"ran{where}")
            if discovery_ledger["unfinished"]:
                raise ValueError(f"Discovery has an unfinished invocation{where}; review "
                                 "before full VAL")
            if any(event["plan_sha256"] != plan_sha or event["sources"] != sources
                   for event in discovery_ledger["started"]):
                raise ValueError("Discovery ran under a different plan or source; "
                                 "full VAL needs discovery under this exact plan and code")
    budget = stage_budget(plan, stage, repository, output)
    unit_name, directory, identifiers = stage_units(plan, stage, all_identifiers, output)[
        0 if shard is None else shard[0]]
    rows = [row for row in inputs["rows"] if row["id"] in set(identifiers)]
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path, run_directory = directory / "ledger.jsonl", directory / "run"
    ledger = ledger_state(ledger_path)
    if ledger["first"] is not None and (
        ledger["first"]["plan_sha256"] != plan_sha or ledger["first"]["sources"] != sources
    ):
        raise ValueError("Plan or source changed since this stage's first invocation")
    journal = journal_state(run_directory, identifiers)
    if journal["complete"]:
        return stage_report(plan, stage, repository, output)
    halts = []
    if ledger["unfinished"]:
        halts.append("previous invocation never recorded its finish")
    if journal["unknown"] or journal["hard_stop"]:
        halts.append("unknown or hard-stop outcome recorded; human review required")
    if ledger["invocations"] >= budget["max_invocations"]:
        halts.append("invocation budget exhausted")
    # Hard ceiling: an invocation may run up to its wall cap (and an unfinished one is
    # charged exactly that), so it starts only if charged + wall cap still fits.
    charged, wall = ledger["charged_active_seconds"], budget["invocation_wall_seconds"]
    if charged + wall > budget["max_active_seconds"]:
        halts.append(f"active-seconds budget exhausted: charged {charged:.1f} s + invocation "
                     f"wall cap {wall} s > {budget['max_active_seconds']} s")
    if ledger["first"] and clock() - ledger["first"]["time"] >= budget["max_calendar_seconds"]:
        halts.append("calendar budget exhausted")
    if run_directory.exists() and _output_bytes(run_directory) >= budget["max_output_bytes"]:
        halts.append("output-bytes budget exhausted")
    if halts:
        raise ValueError("Stage halted: " + "; ".join(halts))
    number = ledger["invocations"] + 1
    started_event = {
        "event": "started", "invocation": number, "time": clock(), "stage": stage,
        "plan_sha256": plan_sha, "sources": sources,
        "wall_cap_seconds": budget["invocation_wall_seconds"],
        "completed_before": len(journal["completed"]),
    }
    if unit_name is not None:
        started_event.update(shard=unit_name, host=socket.gethostname())
    _append(ledger_path, started_event)
    model = plan["model"]
    task_contract = dict(inputs["task_contract"])
    if unit_name is not None:
        task_contract.update(shard=unit_name, shard_assignment=SHARD_ASSIGNMENT,
                             stage_images=len(all_identifiers), images=len(rows))
    common = {
        "phase": "validation", "seed": plan["generation"]["seed"],
        "resume": run_directory.exists(), "backend_factory": backend_factory,
        "image_root": inputs["image_root"], "task_contract": task_contract,
        "preprocessing_id": PREPROCESSING_ID,
        "stop_after_completions": min(budget["chunk_images"], len(journal["remaining"])),
        "stream_progress": True, "profile": plan["generation"]["profile"],
    }
    started = time.monotonic()
    try:
        if engine_of(plan) == "mlx":
            local_vlm.run_local(
                rows, run_directory,
                model_directory=repository / model["directory"],
                integrity=repository / model["integrity"],
                memory_limit_bytes=plan["generation"]["mlx_memory_limit_bytes"], **common,
            )
        else:
            local_vlm_hf.run_local_hf(
                rows, run_directory,
                model_directory=Path(model_directory or repository / model["directory"]),
                integrity=repository / model["integrity"],
                integrity_sha256=model["integrity_sha256"],
                runtime_pins=model["runtime_pins"], dtype=plan["generation"]["dtype"],
                expected_formatted_prompt_sha256=plan["prompt_and_parser"][
                    "formatted_prompt_sha256"],
                **common,
            )
    finally:
        _append(ledger_path, {"event": "finished", "invocation": number,
                              "elapsed_seconds": time.monotonic() - started})
    report = stage_report(plan, stage, repository, output)
    if stage == "discovery" and report["status"] == "completed":
        report["decision"] = discovery_decision(plan, report)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def smoke_train(plan, repository, output, *, model_directory=None, backend_factory=None):
    """Engine smoke on the plan's pinned TRAIN images: cost and plumbing, never a score.

    Opens no VAL or TEST image. Runs the same prompt, profile, preprocessing and engine as
    the stages, through the same journal, under phase train_probe. Allowed on an
    unactivated plan because it produces no VAL outcome.
    """
    if engine_of(plan) != "transformers-cpu":
        raise ValueError("The smoke is defined for the transformers engine only")
    repository = Path(repository).resolve()
    smoke, inputs = plan["smoke"], plan["inputs"]
    dataset = _within(repository / inputs["dataset_root"], repository)
    spec_path = _within(dataset / inputs["data_spec"]["path"], dataset)
    audit_path = _within(dataset / smoke["train_images"]["path"], dataset)
    for path, digest in ((spec_path, inputs["data_spec"]["sha256"]),
                         (audit_path, smoke["train_images"]["sha256"])):
        if local_vlm.sha_file(path) != digest:
            raise ValueError(f"Pinned input changed: {path.name}")
    labels, prompt = checked_prompt(plan, spec_path)
    by_id = {row["image_id"]: row for row in read_json(audit_path)}
    image_root = _within(dataset / "development/images/train", dataset)
    rows, gold = [], {}
    for identifier in smoke["train_ids"]:
        record = by_id[identifier]
        image = _within(image_root / f"{identifier}.jpg", image_root)
        if (record.get("split") != "train" or record.get("path")
                != f"development/images/train/{identifier}.jpg"
                or local_vlm.sha_file(image) != record["encoded_sha256"]):
            raise ValueError(f"TRAIN smoke image differs from the audit: {identifier}")
        rows.append({"id": identifier, "image": str(image), "split": "train", "prompt": prompt})
        gold[identifier] = record["variant"]
    contract = {
        "task": "classification", "benchmark": "FGVC-Aircraft 2013b TRAIN engine smoke",
        "plan_id": plan["plan_id"], "stage": "smoke-train", "probe_images": len(rows),
        "images": len(rows), "labels": labels, "response_format": "answer_only",
        "prompt_sha256": plan["prompt_and_parser"]["prompt_sha256"],
        "grammar": {"name": OUTPUT_GRAMMAR, "parser_source_sha256": parser_source_sha256()},
    }
    model = plan["model"]
    run_directory = Path(output).resolve() / "smoke-train" / "run"
    local_vlm_hf.run_local_hf(
        rows, run_directory,
        model_directory=Path(model_directory or repository / model["directory"]),
        integrity=repository / model["integrity"],
        integrity_sha256=model["integrity_sha256"], runtime_pins=model["runtime_pins"],
        phase="train_probe", seed=plan["generation"]["seed"],
        dtype=plan["generation"]["dtype"], resume=run_directory.exists(),
        backend_factory=backend_factory, image_root=image_root, task_contract=contract,
        preprocessing_id=PREPROCESSING_ID, stream_progress=True,
        profile=plan["generation"]["profile"],
        expected_formatted_prompt_sha256=plan["prompt_and_parser"]["formatted_prompt_sha256"],
    )
    journal = journal_state(run_directory, smoke["train_ids"])
    images = []
    for identifier in smoke["train_ids"]:
        event = journal["completed"].get(identifier)
        if event is None:
            images.append({"id": identifier, "status": "not completed"})
            continue
        prediction = event["prediction"]
        native = event.get("native_response") or {}
        images.append({
            "id": identifier, "latency_seconds": prediction["latency_seconds"],
            "finish_reason": prediction.get("finish_reason"),
            "response": prediction.get("response"),
            "canonical": canonical(prediction, labels),
            "correct_on_train": canonical(prediction, labels) == gold[identifier],
            "prompt_tokens": native.get("prompt_tokens"),
            "generation_tokens": native.get("generation_tokens"),
            "time_to_first_token_seconds": native.get("time_to_first_token_seconds"),
            "image_grid_thw": (event.get("preprocessing") or {}).get("image_grid_thw"),
            "formatted_prompt_sha256": (event.get("preprocessing") or {}).get(
                "formatted_prompt_sha256"),
            "error": prediction.get("error"),
        })
    latencies = [image["latency_seconds"] for image in images if "latency_seconds" in image]
    loads = [event["load_seconds"] for event in map(
        json.loads, (run_directory / "runtime.jsonl").read_text().splitlines())
        if event.get("event") == "load_completed"]
    return {
        "plan_id": plan["plan_id"], "kind": "TRAIN engine smoke; not a VAL result",
        "host": socket.gethostname(), "images": images,
        "completed": len(journal["completed"]), "unknown": journal["unknown"],
        "hard_stop": journal["hard_stop"],
        "mean_seconds_per_image": sum(latencies) / len(latencies) if latencies else None,
        "load_seconds": loads,
        "runtime": read_json(run_directory / "runtime.json"),
        "test_consumed": False, "val_consumed": False,
    }


def endpoint_predictions(plan, bundle, identifiers):
    """Canonical labels of each pinned comparator arm, from its hash-pinned journal."""
    result = {}
    for name, endpoint in plan["comparison"]["endpoints"].items():
        path = Path(bundle) / "arms" / endpoint["arm"] / "predictions.jsonl"
        if local_vlm.sha_file(path) != endpoint["prediction_journal_sha256"]:
            raise ValueError(f"Endpoint journal differs from the plan pin: {name}")
        labels = {}
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "outcome":
                continue
            if event["id"] in labels:
                raise ValueError(f"Endpoint {name} records {event['id']} twice")
            outcome = event["result"]
            labels[event["id"]] = (outcome.get("canonical_label")
                                   if outcome.get("status") == "completed" else None)
        if len(labels) != NATIVE_IMAGES:
            raise ValueError(f"Endpoint {name} is not a complete native-VAL arm")
        missing = sorted(set(identifiers) - set(labels))
        if missing:
            raise ValueError(f"Endpoint {name} lacks {len(missing)} stage images")
        result[name] = {identifier: labels[identifier] for identifier in identifiers}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report", "compare", "merge", "compare-endpoints"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--stage", choices=STAGES, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--repository", type=Path,
                             default=Path(__file__).resolve().parents[1])
        if name == "run":
            command.add_argument("--expected-plan-sha256", required=True)
            command.add_argument("--shard", type=parse_shard, default=None,
                                 help="k/N for a sharded plan")
            command.add_argument("--model-directory", type=Path, default=None,
                                 help="host-local snapshot (rehashed against the pin)")
        if name == "compare":
            command.add_argument("--against", type=Path, required=True,
                                 help="JSON {id: canonical label or null} from another arm")
        if name == "merge":
            command.add_argument("--write", type=Path, required=True,
                                 help="new JSON file for the merged {id: canonical label}")
        if name == "compare-endpoints":
            command.add_argument("--bundle", type=Path, required=True,
                                 help="completed native-VAL evaluator bundle directory")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--plan", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    smoke.add_argument("--model-directory", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_stage(args.plan, args.expected_plan_sha256, args.stage,
                               args.repository, args.output, shard=args.shard,
                               model_directory=args.model_directory)
        elif args.command == "smoke":
            plan = load_plan(args.plan, require_activated=False)
            result = smoke_train(plan, args.repository, args.output,
                                 model_directory=args.model_directory)
        else:
            plan = load_plan(args.plan, require_activated=False)
            result = stage_report(plan, args.stage, args.repository, args.output)
            if args.command == "report" and args.stage == "discovery":
                result["decision"] = discovery_decision(plan, result)
            if args.command in {"compare", "merge", "compare-endpoints"} and (
                result["status"] != "completed" or result["metric"] is None
            ):
                raise ValueError(
                    f"{args.command} refused: {result['unattempted']} unattempted and "
                    f"{len(result['unknown_after_interruption'])} unknown images")
            inputs = None
            if args.command in {"compare", "compare-endpoints"}:
                inputs = stage_inputs(plan, args.stage, args.repository)
                rule = plan["comparison"]
            if args.command == "compare":
                result = compare_classification(
                    inputs["gold"], result["predictions"], read_json(args.against),
                    inputs["labels"], resamples=rule["resamples"], seed=rule["seed"],
                )
            elif args.command == "compare-endpoints":
                identifiers = [row["id"] for row in inputs["rows"]]
                endpoints = endpoint_predictions(plan, args.bundle, identifiers)
                result = {
                    "baseline": {k: result[k] for k in ("plan_id", "stage", "metric")},
                    "comparisons": {
                        name: compare_classification(
                            inputs["gold"], result["predictions"], labels, inputs["labels"],
                            resamples=rule["resamples"], seed=rule["seed"],
                        ) for name, labels in endpoints.items()
                    },
                    "difference_sign": "baseline minus endpoint",
                }
            elif args.command == "merge":
                with args.write.open("x") as handle:
                    json.dump(result["predictions"], handle, indent=1, sort_keys=True)
                result = {"merged": len(result["predictions"]), "written": str(args.write),
                          "ids_sha256": result["ids_sha256"], "metric": result["metric"]}
        result.pop("predictions", None)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"named baseline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
