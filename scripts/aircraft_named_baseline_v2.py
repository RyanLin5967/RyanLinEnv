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
  shards   author-order position i of a stage manifest belongs to shard i mod N (N per
           stage); each shard has its own journal and budget ledger, runs only on a host
           the plan's allowlist names, and never changes host (the host is in its identity)
  merge    the union of shard journals; refuses a duplicate image, an image outside its
           shard, a shard directory outside the declared N, a shard with journal events
           but no budget ledger, or run identities differing in anything but rows, shard
           index and host (the kernel fingerprint included), and scores only when every
           image of the stage completed with no unknown outcome
  gate     the full-stage cap is projected at the slowest discovery host's mean latency,
           and a full shard may run only on a host that ran discovery
  smoke    four TRAIN images through the engine without a journal, for latency sizing

v1 files are not edited: the paused v1 run hashes them into its identity and ledger, and
an edit would make it refuse to resume. v1 run-2 results are never read by v2.

Blind spots, stated on purpose: the ledgers are local state, not tamper-proof witnesses;
a shard killed with SIGKILL leaves no "finished" record and is charged its full wall cap;
shards on different hosts share no lock, so running the same shard on two hosts at once is
prevented only by the operator (a merge then refuses the duplicate images); latency on
shared hosts includes other users' load, so a host loaded more heavily during full VAL
than during discovery can still exhaust the cap (the multiplier is the only margin);
exported comparison arms carry their provenance in the file and compare checks it, but a
hand-edited arm file with matching provenance is not detected.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

# The plan runs `<venv>/bin/python -I scripts/aircraft_named_baseline_v2.py`: -I drops the
# script directory and PYTHONPATH, and the runtime lock forbids installing this project into
# the venv, so the staged tree's src/ is put first on the path here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_rl import local_vlm  # noqa: E402
from vision_rl.aircraft_data import PREPROCESSING_ID  # noqa: E402
from vision_rl.classification_comparison import (  # noqa: E402
    _inputs,
    _score,
    compare_classification,
)
from vision_rl.vlm_local_training import load_spec, parse_label, prompt_text  # noqa: E402

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
# The comparison arms are part of the claim, so they are fixed here, not only in the plan:
# plan arm name -> evaluator arm id. strongest_SFT must also be the arm the evaluator's own
# comparison.json selected (strongest_declared_sft).
ARMS = {"W": "common", "RL400": "rl-exposure", "strongest_SFT": "sft-1e5-time"}
ARM_SCHEMA = "named-baseline-v2-arm-predictions-v1"


def check_plan_consistency(plan):
    """Static rules every v2 plan must satisfy, activated or not."""
    full = plan["stages"]["full"]
    rule = full["gate"]
    wall = invocation_wall_seconds(full["budget"])
    # cap = min(max_shard_active, multiplier x P + wall) must leave room for P + wall at
    # every admissible projection P <= max_projected; only the first term can fall short.
    if rule["max_projected_shard_active_seconds"] + wall > rule["max_shard_active_seconds"]:
        raise ValueError("Plan gate is inconsistent: max_projected_shard_active_seconds plus "
                         "one full invocation wall cap exceeds max_shard_active_seconds")
    if rule["active_cap_multiplier"] < 1:
        raise ValueError("Plan gate multiplier must be at least 1")
    arms = {arm["name"]: arm["evaluator_arm"] for arm in plan["comparison"]["arms"]}
    if arms != ARMS:
        raise ValueError(f"Plan comparison arms must be exactly {ARMS}")
    hosts = plan["sharding"]["hosts"]
    if not isinstance(hosts, dict) or not hosts or not all(
            isinstance(model, str) and model for model in hosts.values()):
        raise ValueError("Plan must allowlist shard hosts as {hostname: cpu model}")
    for stage in STAGES:
        shard_count(plan, stage)


def load_plan(path, expected_sha256=None, *, require_activated=True):
    path = Path(path)
    plan = v1.read_json(path)
    if plan.get("plan_id") != PLAN_ID or plan.get("schema_version") != 1:
        raise ValueError("Not the named-baseline v2 plan schema")
    check_plan_consistency(plan)
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
        if set(plan["sources_pinned"]) != set(source_hashes()):
            raise ValueError("sources_pinned must name exactly the runner's hashed sources")
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


def integrity_path(plan, repository):
    """The committed snapshot record, refused unless it is the one the plan pins."""
    path = Path(repository) / plan["model"]["integrity"]
    if local_vlm.sha_file(path) != plan["model"]["integrity_sha256"]:
        raise ValueError("Snapshot integrity record differs from the plan's pinned SHA256")
    return path


def runtime_lock(plan, repository):
    """The committed ``uv pip freeze`` and Python version, refused unless the plan pins it."""
    lock = plan["model"]["runtime_lock"]
    path = Path(repository) / lock["freeze"]
    if local_vlm.sha_file(path) != lock["freeze_sha256"]:
        raise ValueError("Runtime lock differs from the plan's pinned SHA256")
    return {"freeze": path.read_text(), "python": lock["python"]}


def check_host(plan):
    """This machine must be one the plan allowlists, with the CPU model it was checked on."""
    host = local_hf.host_identity()
    allowed = plan["sharding"]["hosts"]
    if allowed.get(host["hostname"]) != host["cpu_model"]:
        raise ValueError(f"Host {host['hostname']} ({host['cpu_model']}) is not on the plan's "
                         "shard-host allowlist")
    return host


def engine_identity(plan):
    engine = plan["engine"]
    return {"name": engine["name"], "dtype": engine["dtype"], "threads": engine["threads"],
            "cpu_numerics": engine["cpu_numerics"],
            "formatted_prompt_sha256": plan["prompt_and_parser"]["formatted_prompt_sha256"]}


def strict_journal(run_directory, assigned):
    """v1's journal replay plus refusals it does not make: a repeated reservation or
    completion of one image (two runs of a shard concatenated) or a foreign image."""
    path = Path(run_directory) / "attempts.jsonl"
    seen = {"reserved": set(), "completed": set()}
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            kind, identifier = event["event"], event["id"]
            if identifier not in assigned:
                raise ValueError(f"Image {identifier} journaled outside its shard")
            if kind in seen:
                if identifier in seen[kind]:
                    raise ValueError(f"Duplicate {kind} event for image {identifier}")
                seen[kind].add(identifier)
    return v1.journal_state(run_directory, list(assigned))


def shard_count(plan, stage):
    shards = plan["sharding"]["shards"].get(stage)
    if type(shards) is not int or shards < 1:
        raise ValueError(f"Plan must declare a positive integer shard count for {stage}")
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
    shards = shard_count(plan, stage)
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
    shared["config"].pop("host")
    contract = shared["config"]["task_contract"]
    for key in ("images", "shard"):
        contract.pop(key)
    return shared


def merged_report(plan, stage, repository, output, *, plan_sha256=None, score=True):
    """Union of every shard journal of a stage. Metric is null unless all images completed
    and ``score`` is set (the diagnostic ``report`` command never scores)."""
    inputs = stage_inputs(plan, stage, repository)
    identifiers = [row["id"] for row in inputs["rows"]]
    shards = shard_count(plan, stage)
    stage_directory = Path(output) / f"stage-{stage}"
    if stage_directory.exists():
        for child in stage_directory.iterdir():
            if child.name.startswith("."):
                continue  # .DS_Store, rsync temporaries; never a shard
            match = SHARD_NAME.fullmatch(child.name)
            if not match or int(match[2]) != shards or int(match[1]) >= shards:
                raise ValueError(f"Unexpected shard directory {child.name}; plan declares "
                                 f"{shards} shards")
    completed, reserved, unknown, hard_stop = {}, {}, [], []
    loads, latencies, shared, per_shard, governing = [], [], None, [], None
    per_host = {}
    charged = generation = 0.0
    invocations = 0
    for shard in range(shards):
        assigned = shard_ids(identifiers, shard, shards)
        directory = shard_directory(output, stage, shard, shards)
        journal = strict_journal(directory / "run", set(assigned))
        # strict_journal refuses foreign and repeated images, so shards cannot overlap.
        reserved.update(dict.fromkeys(journal["reserved"], shard))
        completed.update(journal["completed"])
        unknown += journal["unknown"]
        hard_stop += journal["hard_stop"]
        ledger = v1.ledger_state(directory / "ledger.jsonl")
        run_json = directory / "run" / "run.json"
        if (journal["reserved"] or journal["completed"]) and not (
                ledger["started"] and run_json.exists()):
            # Without its ledger a shard's plan, sources and charged time are unknowable, so
            # its results cannot be merged (e.g. only run/ was fetched from its host).
            raise ValueError(f"Shard {shard} has journaled images but no budget ledger or "
                             "run identity; fetch the whole shard directory")
        host = None
        if run_json.exists():
            recorded = json.loads(run_json.read_text())["identity"]
            host = recorded["config"]["host"]
            identity = _shared_identity(recorded)
            if shared is None:
                shared = identity
            elif identity != shared:
                raise ValueError(f"Shard {shard} ran under a different run identity")
        for event in ledger["started"]:
            if (event.get("stage"), event.get("shard"), event.get("shards")) != (
                    stage, shard, shards) or (host is not None and event.get("host") != host):
                # e.g. a sibling shard's ledger copied in to satisfy the ledger requirement
                raise ValueError(f"Shard {shard} ledger belongs to another stage, shard or host")
            if plan_sha256 is not None and event["plan_sha256"] != plan_sha256:
                raise ValueError(f"Shard {shard} ran under a plan other than the one given")
            governed = (event["plan_sha256"], json.dumps(event["sources"], sort_keys=True))
            if governing is None:
                governing = governed
            elif governed != governing:
                raise ValueError(f"Shard {shard} ran under a different plan or runner source "
                                 "than another invocation of this stage")
        invocations += ledger["invocations"]
        charged += ledger["charged_active_seconds"]
        shard_latencies = [e["prediction"]["latency_seconds"]
                           for e in journal["completed"].values()]
        shard_generation = sum(shard_latencies)
        generation += shard_generation
        latencies += shard_latencies
        if host is not None and shard_latencies:
            entry = per_host.setdefault(host["hostname"], {
                "cpu_model": host["cpu_model"], "images": 0, "generation_seconds": 0.0})
            entry["images"] += len(shard_latencies)
            entry["generation_seconds"] += shard_generation
        runtime = directory / "run" / "runtime.jsonl"
        if runtime.exists():
            loads += [event["load_seconds"]
                      for event in map(json.loads, runtime.read_text().splitlines())
                      if event.get("event") == "load_completed"]
        per_shard.append({
            "shard": shard, "host": host, "assigned": len(assigned),
            "completed": len(journal["completed"]), "unknown": len(journal["unknown"]),
            "invocations": ledger["invocations"],
            "charged_active_seconds": ledger["charged_active_seconds"],
            "generation_seconds": shard_generation,
        })
    for entry in per_host.values():
        entry["mean_image_latency_seconds"] = entry["generation_seconds"] / entry["images"]
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
    complete = not missing and not unknown and not hard_stop
    report = {
        "plan_id": PLAN_ID, "stage": stage, "shards": shards, "per_shard": per_shard,
        "per_host": per_host,
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
    if complete and score:
        rows = _inputs(inputs["gold"], (predictions,), inputs["labels"])
        result, _ = _score(rows, predictions, inputs["labels"])
        report["metric"] = {
            "name": "mean(diag(row_normalized_confusion))",
            "mean_class_accuracy": result["mean_class_accuracy"],
            "image_accuracy": result["micro_accuracy"],
            "correct": result["correct"], "n": result["n"],
            "invalid_answers_including_missing": result["invalid_answers_including_missing"],
        }
        if stage == "discovery":
            rule = plan["stages"]["discovery"]["uncertainty"]
            half = v1.hoeffding_half_width(result["n"], rule["confidence_level"])
            mean = result["micro_accuracy"]
            report["metric"]["hoeffding"] = {
                "confidence_level": rule["confidence_level"], "half_width": half,
                "interval": [max(0.0, mean - half), min(1.0, mean + half)],
            }
        report["predictions"] = predictions
    return report


def merge(plan, stage, repository, output, plan_sha256):
    """The merged predictions, refusing any missing, unknown, hard-stop or repeated image,
    and any shard that ran under a plan other than the activated one given."""
    report = merged_report(plan, stage, repository, output, plan_sha256=plan_sha256)
    if report["status"] != "completed":
        raise ValueError(f"Merge refused: {report['missing_images']} missing, "
                         f"{len(report['unknown_after_interruption'])} unknown and "
                         f"{len(report['hard_stop_errors'])} hard-stop images")
    return report


def invocation_wall_seconds(budget):
    return v1.invocation_wall_seconds(budget)


def discovery_decision(plan, report):
    """Pre-registered gate from discovery to full VAL, projected per shard (wall clock).

    Each full shard runs on one host, so the projection uses the slowest discovery host's
    mean image latency, not the pooled mean, and the per-invocation overhead of the
    costliest discovery shard; run_shard refuses a full shard on a host that ran no
    discovery images, so every host that can run one is covered by the projection.
    """
    rule, full = plan["stages"]["full"]["gate"], plan["stages"]["full"]
    reasons = []
    if report["status"] != "completed":
        reasons.append("discovery incomplete")
    if report["unknown_after_interruption"] or report["hard_stop_errors"]:
        reasons.append("discovery has unknown or hard-stop outcomes")
    projection = cap = per_invocation = slowest = None
    if report["completed"]:
        valid = 1 - report["invalid_or_unterminated_answers"] / report["completed"]
        if valid < rule["minimum_valid_answer_fraction"]:
            reasons.append("valid-answer fraction below the pre-registered floor")
        load = report["mean_load_seconds"]
        overheads = [(s["charged_active_seconds"] - s["generation_seconds"]) / s["invocations"]
                     for s in report["per_shard"] if s["invocations"]]
        if load is None:
            reasons.append("discovery recorded no completed model load")
        elif not overheads:
            reasons.append("discovery recorded no invocation in its ledgers")
        elif not report["per_host"]:
            reasons.append("discovery recorded no host with completed images")
        else:
            per_invocation = max([load, *overheads])
            slowest = max(report["per_host"].items(),
                          key=lambda item: item[1]["mean_image_latency_seconds"])
            images = math.ceil(full["images"] / shard_count(plan, "full"))
            invocations = math.ceil(images / full["budget"]["chunk_images"])
            projection = (images * slowest[1]["mean_image_latency_seconds"]
                          + invocations * per_invocation)
            if projection > rule["max_projected_shard_active_seconds"]:
                reasons.append("measured profile is unaffordable at the declared ceiling")
            # v2 adds one invocation wall cap to the multiplied projection, so a shard that
            # costs the projection can always start its last chunk; check_plan_consistency
            # guarantees min(...) leaves that room whenever the projection is affordable.
            cap = min(rule["max_shard_active_seconds"],
                      rule["active_cap_multiplier"] * projection
                      + invocation_wall_seconds(full["budget"]))
    return {
        "decision": "stop" if reasons else "proceed", "reasons": reasons,
        "per_invocation_overhead_seconds": per_invocation,
        "slowest_host": None if slowest is None else {"hostname": slowest[0], **slowest[1]},
        "projected_shard_active_seconds": projection,
        "shard_max_active_seconds": cap,
        "headroom_gate": "not applied here; recorded for the root decision",
    }


def stage_budget(plan, stage, repository, output, plan_sha256=None, host=None):
    budget = dict(plan["stages"][stage]["budget"])
    if stage == "full":
        discovery = merged_report(plan, "discovery", repository, output,
                                  plan_sha256=plan_sha256, score=False)
        decision = discovery_decision(plan, discovery)
        if decision["decision"] != "proceed":
            raise ValueError("Full VAL refused by the discovery gate: "
                             + "; ".join(decision["reasons"]))
        measured = discovery["per_host"].get((host or {}).get("hostname"))
        if measured is None or measured["cpu_model"] != host["cpu_model"]:
            # The cap is projected from the discovery hosts only; a host none of them
            # measured could be slower than the slowest one.
            raise ValueError("Full VAL refused by the discovery gate: this host ran no "
                             "discovery images, so the shard cap does not cover it")
        budget["max_active_seconds"] = decision["shard_max_active_seconds"]
    budget["invocation_wall_seconds"] = invocation_wall_seconds(budget)
    return budget


def run_shard(plan_path, expected_sha256, stage, shard, repository, output, weights,
              *, backend_factory=None, clock=time.time):
    """One bounded invocation of one shard of one stage."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    plan = load_plan(plan_path, expected_sha256)
    plan_sha = local_vlm.sha_file(plan_path)
    sources = source_hashes()
    if any(sources[name] != digest for name, digest in plan["sources_pinned"].items()):
        raise ValueError("A pinned runner source changed")
    host = check_host(plan)
    shards = shard_count(plan, stage)
    if stage == "full":
        discovery_shards = shard_count(plan, "discovery")
        for number in range(discovery_shards):
            ledger = v1.ledger_state(shard_directory(output, "discovery", number,
                                                     discovery_shards) / "ledger.jsonl")
            if not ledger["started"]:
                raise ValueError("Full VAL refused by the discovery gate: discovery shard "
                                 f"{number} never ran (or its directory was not copied into "
                                 "this --output: sync every stage-discovery shard first)")
            if ledger["unfinished"]:
                raise ValueError("Discovery has an unfinished invocation; review first")
            if any(event["plan_sha256"] != plan_sha or event["sources"] != sources
                   for event in ledger["started"]):
                raise ValueError("Discovery ran under a different plan or source; "
                                 "full VAL needs discovery under this exact plan and code")
    inputs = stage_inputs(plan, stage, repository, shard)
    identifiers = [row["id"] for row in inputs["rows"]]
    budget = stage_budget(plan, stage, repository, output, plan_sha, host)
    lock = runtime_lock(plan, repository)
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
        return _shard_summary(plan, stage, repository, output, directory, plan_sha)
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
    model = plan["model"]
    integrity = integrity_path(plan, repository)
    # Every machine-side refusal (snapshot, runtime lock, kernel pin) and every identity
    # difference on resume (host, kernel fingerprint, versions) is decided here, before the
    # ledger charges an invocation; run_local_hf reuses this preflight.
    if any(event.get("host") != host for event in ledger["started"]):
        # An invocation can be charged and fail before run.json exists (a staging error,
        # a kill while hashing images); the ledger, not only run.json, binds the host.
        raise ValueError("This shard's ledger started on another host; a shard resumes only "
                         "where it started (refused before any invocation is charged)")
    checked = local_hf.machine_preflight(Path(weights), integrity, engine_identity(plan),
                                         model["runtime_pins"], lock)
    if (run_directory / "run.json").exists():
        recorded = json.loads((run_directory / "run.json").read_text())["identity"]
        differ = [key for key in local_hf.MACHINE_KEYS
                  if recorded["config"].get(key) != checked[key]]
        differ += ["model"] if recorded["model"] != checked["model"] else []
        if differ:
            raise ValueError(f"This machine differs from this shard's recorded run identity in "
                             f"{differ}; a shard resumes only where and as it started "
                             "(refused before any invocation is charged)")
    number = ledger["invocations"] + 1
    v1._append(ledger_path, {
        "event": "started", "invocation": number, "time": clock(), "stage": stage,
        "shard": shard, "shards": shards, "plan_sha256": plan_sha, "sources": sources,
        "host": host, "wall_cap_seconds": budget["invocation_wall_seconds"],
        "completed_before": len(journal["completed"]),
    })
    factory = backend_factory or local_hf.TransformersBackend
    started = time.monotonic()
    try:
        local_hf.run_local_hf(
            inputs["rows"], run_directory, model_directory=Path(weights),
            integrity=integrity, engine=engine_identity(plan), preflight=checked,
            runtime_pins=model["runtime_pins"], runtime_lock=lock, phase="validation",
            # run.json is written exclusively right after the directory is created, so it,
            # not the directory, marks a run that exists.
            seed=plan["generation"]["seed"], resume=(run_directory / "run.json").exists(),
            backend_factory=factory,
            image_root=inputs["image_root"], task_contract=inputs["task_contract"],
            preprocessing_id=PREPROCESSING_ID,
            stop_after_completions=min(budget["chunk_images"], len(journal["remaining"])),
            stream_progress=True, profile=plan["generation"]["profile"],
        )
    finally:
        v1._append(ledger_path, {"event": "finished", "invocation": number,
                                 "elapsed_seconds": time.monotonic() - started})
    return _shard_summary(plan, stage, repository, output, directory, plan_sha)


def _shard_summary(plan, stage, repository, output, directory, plan_sha):
    report = merged_report(plan, stage, repository, output, plan_sha256=plan_sha)
    if stage == "discovery" and report["status"] == "completed":
        report["decision"] = discovery_decision(plan, report)
    report.pop("predictions", None)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


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


def compare(plan, repository, output, against, plan_sha256):
    """Baseline (candidate) versus each declared arm on the complete native VAL.

    Each arm file must be export-arm output for that arm: its provenance must name the
    plan arm, the pinned evaluator arm (ARMS) and the plan's pinned comparison.json, and
    every prediction must be a vocabulary label or null.
    """
    report = merge(plan, "full", repository, output, plan_sha256)
    inputs = stage_inputs(plan, "full", repository)
    rule = plan["comparison"]
    pinned = rule["arm_source"]["comparison_json_sha256"]
    if set(against) != set(ARMS):
        raise ValueError(f"Compare needs exactly the declared arms {sorted(ARMS)}")
    vocabulary = set(inputs["labels"])
    results = {}
    for name, path in sorted(against.items()):
        exported = v1.read_json(path)
        provenance = {key: exported.get(key) for key in (
            "schema", "plan_arm", "evaluator_arm", "comparison_json_sha256")}
        if provenance != {"schema": ARM_SCHEMA, "plan_arm": name,
                          "evaluator_arm": ARMS[name], "comparison_json_sha256": pinned}:
            raise ValueError(f"Arm {name} file is not export-arm output for evaluator arm "
                             f"{ARMS[name]} of the pinned comparison.json: {provenance}")
        predictions = exported["predictions"]
        if set(predictions) != {row["id"] for row in inputs["gold"]}:
            raise ValueError(f"Arm {name} does not cover the native VAL manifest exactly")
        if any(label is not None and label not in vocabulary
               for label in predictions.values()):
            raise ValueError(f"Arm {name} has a prediction outside the label vocabulary")
        results[name] = {
            "evaluator_arm": ARMS[name],
            "prediction_journal_sha256": exported["prediction_journal_sha256"],
            "equal_class": compare_classification(
                inputs["gold"], report["predictions"], predictions, inputs["labels"],
                resamples=rule["resamples"], seed=rule["seed"]),
            "image": image_accuracy_bootstrap(
                inputs["gold"], report["predictions"], predictions, inputs["labels"],
                rule["resamples"], rule["seed"]),
        }
    return {"plan_id": PLAN_ID, "candidate": "named baseline v2 (merged shards)",
            "comparison_json_sha256": pinned, "against": results, "test_consumed": False}


def export_arm(plan, bundle, name):
    """Canonical {id: label|None} of one declared arm, re-parsed by the frozen rule.

    Refuses unless the bundle's comparison.json is the one the plan pins, its plan.json is
    the one that comparison ran under, the arm's journal is byte-identical to the one the
    comparison hashed, and, for strongest_SFT, the evaluator selected that very arm.
    """
    from vision_rl import vlm_local_evaluation as evaluation

    if name not in ARMS:
        raise ValueError(f"Arm {name} is not one of the declared arms {sorted(ARMS)}")
    evaluator_arm = ARMS[name]
    bundle = Path(bundle).resolve()
    comparison_path = bundle / "comparison.json"
    pinned = plan["comparison"]["arm_source"]["comparison_json_sha256"]
    if local_vlm.sha_file(comparison_path) != pinned:
        raise ValueError("Bundle comparison.json differs from the plan's pinned SHA256")
    comparison = v1.read_json(comparison_path)
    if comparison["plan_sha256"] != local_vlm.sha_file(bundle / "plan.json"):
        raise ValueError("Bundle plan.json is not the plan its comparison.json ran under")
    if name == "strongest_SFT" and comparison["strongest_declared_sft"] != evaluator_arm:
        raise ValueError("The evaluator's strongest declared SFT is "
                         f"{comparison['strongest_declared_sft']}, not {evaluator_arm}")
    bundle_plan = evaluation.read(bundle / "plan.json")
    declared = {entry["id"]: entry for entry in bundle_plan["study"]["arms"]}
    if evaluator_arm not in declared:
        raise ValueError(f"Arm {evaluator_arm} is not declared by the evaluator bundle")
    journal_sha = comparison["prediction_journal_sha256"][evaluator_arm]
    if local_vlm.sha_file(bundle / "arms" / evaluator_arm / "predictions.jsonl") != journal_sha:
        raise ValueError(f"Arm {evaluator_arm} journal differs from the one comparison.json "
                         "hashed")
    images = evaluation.read(bundle / "images.json")
    journal = evaluation._journal(bundle, bundle_plan, declared[evaluator_arm], images)
    if journal.pending is not None or len(journal.results) != len(images):
        raise ValueError("Arm journal is incomplete")
    out = {}
    for identifier, result in journal.results.items():
        parsed = (parse_label(result["response"], bundle_plan["labels"], allow_reasoning=True)
                  if result["status"] == "completed" and result["terminated"] else None)
        if parsed != result["canonical_label"]:
            raise ValueError("Canonical prediction differs from the frozen raw-output parser")
        out[identifier] = parsed
    return {"schema": ARM_SCHEMA, "plan_arm": name, "evaluator_arm": evaluator_arm,
            "comparison_json_sha256": pinned, "prediction_journal_sha256": journal_sha,
            "predictions": out}


def smoke(plan, repository, weights, ids, threads, dtype, *, backend_factory=None,
          progress_path=None):
    """Latency smoke on TRAIN images only; no journal, never a score."""
    repository = Path(repository).resolve()
    train_root = (repository / "data/fgvc-aircraft-2013b/development/images/train").resolve()
    inputs = plan["inputs"]
    spec = load_spec(repository / inputs["dataset_root"] / inputs["data_spec"]["path"])
    prompt = prompt_text(spec, "answer_only")
    model = local_hf.verify_snapshot(weights, integrity_path(plan, repository))
    versions = local_hf.runtime_versions(plan["model"]["runtime_pins"],
                                         runtime_lock(plan, repository))
    config = {
        **local_vlm.native_profile(plan["generation"]["profile"]),
        "seed": plan["generation"]["seed"], "generation": plan["generation"]["sampling"],
        "engine": {**engine_identity(plan), "dtype": dtype, "threads": threads},
        "preprocessing_id": PREPROCESSING_ID, "stream_progress": progress_path is not None,
        "generation_progress_path": str(progress_path), "identity_sha256": None,
    }
    started = time.monotonic()
    backend = (backend_factory or local_hf.TransformersBackend)(weights, config)
    # After the backend pinned the kernel path: the same fingerprint a shard's identity holds.
    fingerprint = local_hf.kernel_fingerprint(threads)
    results = []
    for identifier in ids:
        image = (train_root / f"{identifier}.jpg").resolve(strict=True)
        if not image.is_relative_to(train_root):
            raise ValueError("Smoke images must be TRAIN images")
        seed = int(local_vlm._digest([config["seed"], identifier])[:8], 16)
        start = time.monotonic()
        native = backend.infer({"id": identifier, "image_path": str(image), "prompt": prompt},
                               seed)
        latency = time.monotonic() - start
        response = native["native_response"]
        answer = local_vlm.final_answer(response["text"], response["finish_reason"],
                                        profile=plan["generation"]["profile"])
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
        "host": local_hf.host_identity(), "kernel_fingerprint": fingerprint,
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
        if name in ("merge", "compare"):
            command.add_argument("--expected-plan-sha256", required=True)
        if name == "compare":
            command.add_argument("--against", action="append", required=True,
                                 metavar="NAME=PATH",
                                 help="declared arm name = that arm's export-arm output")
    export = commands.add_parser("export-arm")
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--arm", choices=sorted(ARMS), required=True)
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
                               args.repository, args.output, args.weights)
        elif args.command == "export-arm":
            # Reads no v2 result, so the committed (unactivated) plan's pins suffice.
            plan = load_plan(args.plan, require_activated=False)
            args.out.write_text(
                json.dumps(export_arm(plan, args.bundle, args.arm), indent=1) + "\n")
            return 0
        elif args.command == "smoke":
            plan = load_plan(args.plan, require_activated=False)
            result = smoke(plan, args.repository, args.weights, args.ids, args.threads,
                           args.dtype, progress_path=args.out.with_suffix(".progress.jsonl"))
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        elif args.command in ("merge", "compare"):
            # Results are merged and compared only under the activated plan they ran under.
            plan = load_plan(args.plan, args.expected_plan_sha256)
            if args.command == "compare":
                against = dict(item.split("=", 1) for item in args.against)
                result = compare(plan, args.repository, args.output, against,
                                 args.expected_plan_sha256)
            else:
                result = merge(plan, args.stage, args.repository, args.output,
                               args.expected_plan_sha256)
                # Outside stage-<stage>/, whose children must all be shard directories.
                (args.output / f"merged-{args.stage}-predictions.json").write_text(
                    json.dumps(result["predictions"], indent=1) + "\n")
        else:
            # Diagnostic only: counts and latencies, never a score, a merge or a comparison;
            # a metric exists only through merge under the activated plan.
            plan = load_plan(args.plan, require_activated=False)
            result = merged_report(plan, args.stage, args.repository, args.output, score=False)
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
