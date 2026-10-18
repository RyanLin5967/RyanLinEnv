"""Named open-weight baseline on FGVC-Aircraft native VAL: Qwen3.8-27B-8bit, native instruct.

Pre-registered by docs/aircraft-named-baseline-plan-v1.json. The committed plan is
UNACTIVATED and this runner refuses it: a run needs a separately activated copy whose
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
ledger whose original clock and invocation count survive resume, and halt rules.
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
import sys
import time
from pathlib import Path

from vision_rl import local_vlm
from vision_rl.aircraft_data import PREPROCESSING_ID
from vision_rl.classification_comparison import _inputs, _score, compare_classification
from vision_rl.vlm_local_training import OUTPUT_GRAMMAR, load_spec, parse_label, prompt_text

PLAN_ID = "aircraft-named-baseline-qwen3.8-27b-8bit-v1"
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
    if plan.get("plan_id") != PLAN_ID or plan.get("schema_version") != 1:
        raise ValueError("Not the named-baseline plan schema")
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


def _within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"{path} escapes {root}")
    return path


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
    spec = load_spec(paths["data_spec"])
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
        "plan_id": PLAN_ID,
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


def stage_report(plan, stage, repository, output):
    """Score a stage from its journal. Metric is null unless every image completed."""
    inputs = stage_inputs(plan, stage, repository)
    identifiers = [row["id"] for row in inputs["rows"]]
    directory = Path(output) / f"stage-{stage}"
    journal = journal_state(directory / "run", identifiers)
    ledger = ledger_state(directory / "ledger.jsonl")
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
    loads = []
    runtime = directory / "run" / "runtime.jsonl"
    if runtime.exists():
        loads = [
            event["load_seconds"]
            for event in map(json.loads, runtime.read_text().splitlines())
            if event.get("event") == "load_completed"
        ]
    report = {
        "plan_id": PLAN_ID,
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
        "invocations": ledger["invocations"],
        "charged_active_seconds": ledger["charged_active_seconds"],
        "ids_sha256": ids_sha256(identifiers),
        "status": "completed" if journal["complete"] else "partial",
        "metric": None,
        "test_consumed": False,
    }
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
    if report["completed"]:
        valid = 1 - report["invalid_or_unterminated_answers"] / report["completed"]
        if valid < rule["minimum_valid_answer_fraction"]:
            reasons.append("valid-answer fraction below the pre-registered floor")
        images = full["images"]
        invocations = math.ceil(images / full["budget"]["chunk_images"])
        # Per-invocation cost outside image latency: the larger of the measured model load
        # and the ledger's charged seconds not spent generating (load, integrity rehash,
        # input rehash, journal replay). A measured 0.0 is a measurement, not missing.
        load = report["mean_load_seconds"]
        if load is None:
            reasons.append("discovery recorded no completed model load")
        elif report["invocations"]:
            overhead = (report["charged_active_seconds"]
                        - report["generation_seconds_total"]) / report["invocations"]
            per_invocation = max(load, overhead)
            projection = (images * report["mean_image_latency_seconds"]
                          + invocations * per_invocation)
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
              clock=time.time):
    """One bounded invocation of one stage. Refuses unactivated plans and spent budgets."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    plan = load_plan(plan_path, expected_sha256)
    plan_sha = local_vlm.sha_file(plan_path)
    sources = source_hashes()
    pinned = plan["sources_pinned"]
    if any(sources[name] != digest for name, digest in pinned.items()):
        raise ValueError("A pinned runner source changed")
    if stage == "full":
        discovery_ledger = ledger_state(output / "stage-discovery" / "ledger.jsonl")
        if not discovery_ledger["started"]:
            raise ValueError("Full VAL refused by the discovery gate: discovery never ran")
        if discovery_ledger["unfinished"]:
            raise ValueError("Discovery has an unfinished invocation; review before full VAL")
        if any(event["plan_sha256"] != plan_sha or event["sources"] != sources
               for event in discovery_ledger["started"]):
            raise ValueError("Discovery ran under a different plan or source; "
                             "full VAL needs discovery under this exact plan and code")
    inputs = stage_inputs(plan, stage, repository)
    identifiers = [row["id"] for row in inputs["rows"]]
    budget = stage_budget(plan, stage, repository, output)
    directory = output / f"stage-{stage}"
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
    _append(ledger_path, {
        "event": "started", "invocation": number, "time": clock(), "stage": stage,
        "plan_sha256": plan_sha, "sources": sources,
        "wall_cap_seconds": budget["invocation_wall_seconds"],
        "completed_before": len(journal["completed"]),
    })
    model = plan["model"]
    started = time.monotonic()
    try:
        local_vlm.run_local(
            inputs["rows"], run_directory,
            model_directory=repository / model["directory"],
            integrity=repository / model["integrity"],
            phase="validation", seed=plan["generation"]["seed"],
            memory_limit_bytes=plan["generation"]["mlx_memory_limit_bytes"],
            resume=run_directory.exists(), backend_factory=backend_factory,
            image_root=inputs["image_root"], task_contract=inputs["task_contract"],
            preprocessing_id=PREPROCESSING_ID,
            stop_after_completions=min(budget["chunk_images"], len(journal["remaining"])),
            stream_progress=True, profile=plan["generation"]["profile"],
        )
    finally:
        _append(ledger_path, {"event": "finished", "invocation": number,
                              "elapsed_seconds": time.monotonic() - started})
    report = stage_report(plan, stage, repository, output)
    if stage == "discovery" and report["status"] == "completed":
        report["decision"] = discovery_decision(plan, report)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--stage", choices=STAGES, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--repository", type=Path,
                             default=Path(__file__).resolve().parents[1])
        if name == "run":
            command.add_argument("--expected-plan-sha256", required=True)
        if name == "compare":
            command.add_argument("--against", type=Path, required=True,
                                 help="JSON {id: canonical label or null} from another arm")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_stage(args.plan, args.expected_plan_sha256, args.stage,
                               args.repository, args.output)
        else:
            plan = load_plan(args.plan, require_activated=False)
            result = stage_report(plan, args.stage, args.repository, args.output)
            if args.command == "report" and args.stage == "discovery":
                result["decision"] = discovery_decision(plan, result)
            if args.command == "compare":
                if result["metric"] is None:
                    raise ValueError("Comparison requires a completed stage")
                inputs = stage_inputs(plan, args.stage, args.repository)
                rule = plan["comparison"]
                result = compare_classification(
                    inputs["gold"], result["predictions"], read_json(args.against),
                    inputs["labels"], resamples=rule["resamples"], seed=rule["seed"],
                )
        result.pop("predictions", None)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"named baseline: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
