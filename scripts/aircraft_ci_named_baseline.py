"""Named open-weight baselines on FGVC-Aircraft native VAL, sharded over free CI runners.

Pre-registered by docs/aircraft-ci-named-baselines-plan-v1.json. The committed plan is
UNACTIVATED and every model-touching command refuses it: a dispatch needs a separately
committed activated copy and that copy's SHA256.

Nothing here is new scoring. Rows, gold labels, prompt and parser checks come from
``scripts/aircraft_named_baseline.stage_inputs`` (the 27B plan's validated input builder);
each prediction is ``vlm_local_evaluation.Backend.predict`` (the frozen evaluator's code)
bound to a base model without an adapter, under the evaluator's greedy generation kwargs,
per-image ``_deadline`` and output checks; the metric is ``classification_comparison``.
This file owns only downloading pinned bytes, shard slicing (``vision_rl.ci_shards``), the
per-shard journal and the merge that refuses on any gap.

Commands: matrix, fetch-images, fetch-model, run-shard, merge, gate.

Blind spots, stated here on purpose: shard records and journals are CI-job files, not signed
witnesses; the workflow run ID and its artifacts are the external evidence. A shard killed
by the job timeout leaves its journal (uploaded with if: always()) but no record, so the
merge refuses. CPU numerics are not bit-identical to the endpoints' MPS runs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from vision_rl import ci_shards  # noqa: E402  (standard library only)

PLAN_ID = "aircraft-ci-named-baselines-v1"
STAGES = ("discovery", "full")
COMMITTED_INPUTS = Path("docs/aircraft-ci-named-baseline-v1/inputs/fgvc-aircraft-2013b")
INPUT_FILES = ("val_images", "validation", "data_spec")
ENDPOINTS = ("common", "rl-exposure", "sft-1e5-time")
NATIVE_IMAGES = 3333


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def named_baseline():
    """The 27B runner, reused for stage_inputs/select_discovery (imports the CPU stack)."""
    return _module("aircraft_named_baseline", "scripts/aircraft_named_baseline.py")


def load_plan(path, expected_sha256=None, *, require_activated=True):
    path = Path(path)
    plan = read_json(path)
    if plan.get("plan_id") != PLAN_ID or plan.get("schema_version") != 1:
        raise ValueError("Not the CI named-baselines plan schema")
    if require_activated:
        activation = plan.get("activation")
        if (
            plan.get("activated") is not True
            or not isinstance(activation, dict)
            or not activation.get("activated_at")
            or not activation.get("approved_by")
        ):
            raise ValueError("CI named-baselines plan is unactivated; it must not run")
        if expected_sha256 is None or sha_file(path) != expected_sha256:
            raise ValueError("Pass the activated plan's exact SHA256 to run it")
    return plan


def model_entry(plan, model):
    if model not in plan["models"]:
        raise ValueError(f"Model {model!r} is not pre-registered")
    return plan["models"][model]


def stage_order(plan, stage, repository):
    """Stage IDs from committed metadata only (no image is needed or opened)."""
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    if stage == "discovery":
        ids = list(plan["stages"]["discovery"]["ids"])
    else:
        author = read_json(
            Path(repository) / COMMITTED_INPUTS / plan["inputs"]["val_images"]["path"]
        )
        ids = [row["image_id"] for row in author]
    if ci_shards.ids_sha256(ids) != plan["stages"][stage]["ids_sha256"]:
        raise ValueError("Stage IDs differ from the pre-registered manifest")
    return ids


def materialize_inputs(plan, repository):
    """Copy the committed pinned inputs to the dataset root the shared input builder reads."""
    repository = Path(repository).resolve()
    source_root = repository / COMMITTED_INPUTS
    target_root = repository / plan["inputs"]["dataset_root"]
    for name in INPUT_FILES:
        entry = plan["inputs"][name]
        source, target = source_root / entry["path"], target_root / entry["path"]
        if sha_file(source) != entry["sha256"]:
            raise ValueError(f"Committed input differs from its pin: {name}")
        if target.exists():
            if sha_file(target) != entry["sha256"]:
                raise ValueError(f"Existing dataset input differs from its pin: {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target_root


def download(url, destination, *, size, sha256, attempts=3):
    """Bounded retries before any model output exists; content must match size and SHA256."""
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size == size and sha_file(destination) == sha256:
            return "present"
        raise ValueError(f"Existing file differs from its pin: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    import urllib.request

    last = None
    for attempt in range(1, attempts + 1):
        temporary.unlink(missing_ok=True)
        try:
            count, checksum = 0, hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as out:
                while chunk := response.read(1024**2):
                    count += len(chunk)
                    if count > size:
                        raise ValueError("Download exceeds its pinned size")
                    checksum.update(chunk)
                    out.write(chunk)
            if count != size or checksum.hexdigest() != sha256:
                raise ValueError("Downloaded content differs from its pinned size/SHA256")
            temporary.replace(destination)
            return f"downloaded (attempt {attempt})"
        except (OSError, ValueError) as exc:
            last = exc
            time.sleep(5 * attempt)
    temporary.unlink(missing_ok=True)
    raise ValueError(f"Download failed after {attempts} attempts: {destination.name}: {last}")


def fetch_images(plan, stage, repository):
    """Fetch exactly the stage's native-VAL JPEGs; TEST IDs are never formed or requested."""
    repository = Path(repository).resolve()
    dataset = materialize_inputs(plan, repository)
    author = {
        row["image_id"]: row for row in read_json(dataset / plan["inputs"]["val_images"]["path"])
    }
    ids = stage_order(plan, stage, repository)
    template = plan["image_source"]["url_template"]
    fetched = 0
    for identifier in ids:
        row = author[identifier]
        if (
            row["split"] != "val"
            or row["path"] != f"development/images/validation/{identifier}.jpg"
        ):
            raise ValueError("Only native VAL images may be fetched")
        download(
            template.format(id=identifier),
            dataset / row["path"],
            size=row["encoded_bytes"],
            sha256=row["encoded_sha256"],
        )
        fetched += 1
    if fetched != len(ids):
        raise ValueError("Fetched image count differs from the stage")
    return {"stage": stage, "images": fetched, "ids_sha256": ci_shards.ids_sha256(ids)}


def fetch_model(plan, model, directory):
    entry = model_entry(plan, model)
    directory = Path(directory)
    base = f"https://huggingface.co/{entry['repository']}/resolve/{entry['revision']}/"
    for name, pin in sorted(entry["files"].items()):
        download(base + name, directory / name, size=pin["bytes"], sha256=pin["sha256"])
    return verify_model_directory(plan, model, directory)


def verify_model_directory(plan, model, directory):
    """Rehash every pinned snapshot file; extra weight files are refused."""
    entry = model_entry(plan, model)
    directory = Path(directory)
    present = {p.name for p in directory.iterdir() if p.is_file()}
    if present != set(entry["files"]):
        raise ValueError(
            f"Model directory files differ from the pin: {sorted(present ^ set(entry['files']))}"
        )
    for name, pin in entry["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or path.stat().st_size != pin["bytes"]
            or sha_file(path) != pin["sha256"]
        ):
            raise ValueError(f"Model file differs from its pin: {name}")
    return {"model": model, "revision": entry["revision"], "files": len(entry["files"])}


def discovery_decision(plan, report):
    """Pre-registered sizing gate for the full stage, recomputed from a discovery report."""
    execution = plan["execution"]
    reasons = []
    if report.get("stage") != "discovery" or report.get("status") != "completed":
        reasons.append("discovery report is not a completed discovery merge")
    latencies = report.get("latencies_seconds") or []
    timings = report.get("shard_timings") or []
    if len(latencies) != plan["stages"]["discovery"]["images"]:
        reasons.append("discovery latencies do not cover all 100 images")
    if len(timings) != plan["stages"]["discovery"]["shards"]:
        reasons.append("discovery shard timings do not cover all shards")
    shards = mean = overhead = None
    if not reasons:
        mean = sum(latencies) / len(latencies)
        overhead = max(t["elapsed_seconds"] - t["generation_seconds"] for t in timings)
        if max(latencies) > execution["image_seconds"] / 2:
            reasons.append("largest discovery image latency exceeds image_seconds / 2")
        shards = ci_shards.full_shard_count(
            plan["stages"]["full"]["images"],
            mean_latency=mean,
            overhead=overhead,
            shard_wall_seconds=execution["shard_wall_seconds"],
            image_seconds=execution["image_seconds"],
            safety_factor=execution["full_safety_factor"],
        )
        if shards is None:
            reasons.append("no shard count of at most 256 fits the shard wall cap")
    return {
        "decision": "stop" if reasons else "proceed",
        "reasons": reasons,
        "mean_image_latency_seconds": mean,
        "per_shard_overhead_seconds": overhead,
        "full_shards": None if reasons else shards,
    }


def expected_shards(plan, model, stage, repository):
    if stage == "discovery":
        return plan["stages"]["discovery"]["shards"]
    path = Path(repository) / plan["stages"]["full"]["discovery_report"].format(model=model)
    if not path.exists():
        raise ValueError("Full stage refused: the discovery report is not committed")
    report = read_json(path)
    if report.get("model") != model or report.get("plan_id") != PLAN_ID:
        raise ValueError("Full stage refused: discovery report belongs to another model/plan")
    decision = discovery_decision(plan, report)
    if decision["decision"] != "proceed":
        raise ValueError(
            "Full stage refused by the discovery gate: " + "; ".join(decision["reasons"])
        )
    return decision["full_shards"]


def run_identity():
    values = {
        key: os.environ.get(key) for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA")
    }
    if not all(values.values()):
        raise ValueError("run-shard runs only inside a GitHub Actions job (run ID/attempt/SHA)")
    return {
        "run_id": values["GITHUB_RUN_ID"],
        "run_attempt": values["GITHUB_RUN_ATTEMPT"],
        "commit": values["GITHUB_SHA"],
    }


class BaselineBackend:
    """A frozen base model behind the evaluator's own predict(); no adapter, no trainer."""

    def __init__(self, plan, model, model_directory, spec_path):
        evaluation = importlib.import_module("vision_rl.vlm_local_evaluation")
        native = importlib.import_module("vision_rl.vlm_local_training")
        safe = importlib.import_module("vision_rl.vlm_local_task_builder")
        entry = model_entry(plan, model)
        self.predict_function = evaluation.Backend.predict
        self.profile = native.Profile(**{**plan["profile"], "dtype": entry["dtype"]})
        self.profile.validate()
        self.device = plan["execution"]["device"]
        self.spec = native.load_spec(Path(spec_path))
        self.environment = native.preflight(self.device)
        versions = safe._check_runtime()
        if {k: v.split("+")[0] for k, v in versions.items()} != plan["runtime"]:
            raise ValueError("Installed runtime differs from the evaluator's pinned runtime")
        torch, transformers = (importlib.import_module(n) for n in ("torch", "transformers"))
        self.torch = torch
        torch.set_num_threads(plan["execution"]["cpu_threads"])
        transformers.set_seed(42)
        self.processor = native._processor(Path(model_directory), self.profile)
        base = transformers.AutoModelForImageTextToText.from_pretrained(
            str(model_directory),
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            dtype=getattr(torch, entry["dtype"]),
            attn_implementation="sdpa",
        )
        if base.config.model_type != entry["model_type"]:
            raise ValueError("Unexpected pinned model architecture")
        self.policy = base.requires_grad_(False).to(self.device).eval()
        eos = self.policy.generation_config.eos_token_id
        self.eos = set(
            ([eos] if isinstance(eos, int) else list(eos or []))
            + [self.processor.tokenizer.eos_token_id]
        )
        from dataclasses import asdict

        self.provenance = {
            "environment": self.environment,
            "runtime": versions,
            "generation_kwargs": evaluation._generation_kwargs(asdict(self.profile)),
            "profile": asdict(self.profile),
            "prompt_sha256": hashlib.sha256(
                native.prompt_text(self.spec, self.profile.response_format).encode()
            ).hexdigest(),
            "model": model,
            "revision": entry["revision"],
            "model_type": base.config.model_type,
            "trainable_parameters": sum(
                p.numel() for p in self.policy.parameters() if p.requires_grad
            ),
            "prediction_fields": ["id", "image"],
        }

    def predict(self, row):
        return self.predict_function(self, row)

    def close(self):
        self.policy = self.processor = None


def _append(path, value):
    with Path(path).open("a") as handle:
        handle.write(ci_shards.compact(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_shard(
    plan_path,
    expected_sha256,
    model,
    stage,
    shard,
    shards,
    model_directory,
    output,
    repository=REPOSITORY,
    *,
    backend_factory=None,
    monotonic=time.monotonic,
):
    """Score one contiguous shard once. A partial shard writes no record."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    plan = load_plan(plan_path, expected_sha256)
    identity = run_identity()
    model_entry(plan, model)
    want = expected_shards(plan, model, stage, repository)
    if shards != want:
        raise ValueError(f"Shard count {shards} differs from the pre-registered {want}")
    evaluation = importlib.import_module("vision_rl.vlm_local_evaluation")
    native = importlib.import_module("vision_rl.vlm_local_training")
    aircraft = importlib.import_module("vision_rl.aircraft_data")
    inputs = named_baseline().stage_inputs(plan, stage, repository)
    order = [row["id"] for row in inputs["rows"]]
    if order != stage_order(plan, stage, repository):
        raise ValueError("Shared input builder order differs from the committed manifest")
    mine = ci_shards.shard_ids(order, shard, shards)
    author = {
        row["image_id"]: row
        for row in read_json(
            repository / plan["inputs"]["dataset_root"] / plan["inputs"]["val_images"]["path"]
        )
    }
    rows = {row["id"]: row for row in inputs["rows"]}
    output.mkdir(parents=True, exist_ok=True)
    journal = output / f"shard-{shard:03d}.journal.jsonl"
    record_path = output / f"shard-{shard:03d}.json"
    if journal.exists() or record_path.exists():
        raise ValueError("A shard runs once per job; prior output exists")
    execution = plan["execution"]
    started = monotonic()
    verify_model_directory(plan, model, model_directory)
    load_tick = monotonic()
    with evaluation._deadline(execution["load_deadline_seconds"]):
        backend = (backend_factory or BaselineBackend)(
            plan,
            model,
            model_directory,
            repository / plan["inputs"]["dataset_root"] / plan["inputs"]["data_spec"]["path"],
        )
    load_seconds = monotonic() - load_tick
    profile = plan["profile"]
    results, status = {}, "completed"
    _append(
        journal,
        {
            "event": "header",
            "shard": shard,
            "shards": shards,
            "model": model,
            "stage": stage,
            **identity,
            "provenance": backend.provenance,
        },
    )
    try:
        for identifier in mine:
            if monotonic() - started + execution["image_seconds"] > execution["shard_wall_seconds"]:
                status = "partial_wall_cap"
                break
            image = rows[identifier]["image"]
            if (
                sha_file(image) != author[identifier]["encoded_sha256"]
                or aircraft.aircraft_pixel_sha256(image) != author[identifier]["model_rgb_sha256"]
            ):
                raise ValueError("Model-visible native image changed")
            _append(journal, {"event": "reserved", "id": identifier})
            tick = monotonic()
            with evaluation._deadline(execution["image_seconds"]):
                result = backend.predict({"id": identifier, "image": image})
            if (
                type(result["terminated"]) is not bool
                or not isinstance(result["response"], str)
                or len(result["response"]) > 65536
                or len(result["completion_ids"]) > profile["max_completion_length"]
            ):
                raise ValueError("Malformed or oversized inference output")
            result.update(
                status="completed",
                canonical_label=native.parse_label(
                    result["response"], inputs["labels"], allow_reasoning=True
                )
                if result["terminated"]
                else None,
                latency_seconds=monotonic() - tick,
            )
            _append(journal, {"event": "outcome", "id": identifier, "result": result})
            results[identifier] = result
    finally:
        backend.close()
    elapsed = monotonic() - started
    if status != "completed" or set(results) != set(mine):
        _append(journal, {"event": "stopped", "status": status, "completed": len(results)})
        raise ValueError(
            f"Shard {shard} stopped {status} after {len(results)} of {len(mine)} images"
        )
    record = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "plan_sha256": sha_file(plan_path),
        "model": model,
        "stage": stage,
        "shard": shard,
        "shards": shards,
        **identity,
        "status": "completed",
        "ids": mine,
        "ids_sha256": ci_shards.ids_sha256(mine),
        "results": results,
        "provenance": backend.provenance,
        "elapsed_seconds": elapsed,
        "load_seconds": load_seconds,
        "generation_seconds": sum(r["latency_seconds"] for r in results.values()),
        "journal_sha256": sha_file(journal),
        "test_consumed": False,
    }
    write_json(record_path, record)
    return {
        k: record[k]
        for k in ("model", "stage", "shard", "shards", "elapsed_seconds", "generation_seconds")
    }


def merge(plan, plan_sha256, model, stage, shards, records_directory, inputs, repository):
    """Merge shard records into one scored report; refuses on any gap (ShardGapError)."""
    classification = importlib.import_module("vision_rl.classification_comparison")
    native = importlib.import_module("vision_rl.vlm_local_training")
    want = expected_shards(plan, model, stage, repository)
    if shards != want:
        raise ci_shards.ShardGapError(f"Refusing to score: {shards} shards, plan requires {want}")
    ids = [row["id"] for row in inputs["gold"]]
    if ids != stage_order(plan, stage, repository):
        raise ValueError("Gold order differs from the committed manifest")
    records = [read_json(p) for p in sorted(Path(records_directory).rglob("shard-*.json"))]
    if not records:
        raise ci_shards.ShardGapError("Refusing to score: no shard records found")
    first = records[0]
    expected = {
        "plan_id": PLAN_ID,
        "plan_sha256": plan_sha256,
        "model": model,
        "stage": stage,
        "shards": shards,
        "run_id": first.get("run_id"),
        "run_attempt": first.get("run_attempt"),
    }
    merged = ci_shards.merge_shard_records(records, ids, expected)
    labels = inputs["labels"]
    predictions = {}
    for identifier, result in merged.items():
        parsed = (
            native.parse_label(result["response"], labels, allow_reasoning=True)
            if result.get("terminated") is True
            else None
        )
        if parsed != result.get("canonical_label"):
            raise ValueError(f"Stored label differs from the frozen parser for {identifier}")
        predictions[identifier] = parsed
    provenance = {ci_shards.compact(r["provenance"]) for r in records}
    if len(provenance) != 1:
        raise ValueError("Shards ran under different backend provenance")
    rows = classification._inputs(inputs["gold"], (predictions,), labels)
    score, _ = classification._score(rows, predictions, labels)
    report = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "plan_sha256": plan_sha256,
        "model": model,
        "stage": stage,
        "status": "completed",
        "shards": shards,
        "run_id": expected["run_id"],
        "run_attempt": expected["run_attempt"],
        "commit": first.get("commit"),
        "n": len(ids),
        "ids_sha256": ci_shards.ids_sha256(ids),
        "provenance": records[0]["provenance"],
        "metric": {
            "name": "mean(diag(row_normalized_confusion))",
            "mean_class_accuracy": score["mean_class_accuracy"],
            "micro_accuracy": score["micro_accuracy"],
            "correct": score["correct"],
            "n": score["n"],
            "invalid_answers_including_missing": score["invalid_answers_including_missing"],
        },
        "unterminated": sum(r["terminated"] is not True for r in merged.values()),
        "latencies_seconds": [merged[i]["latency_seconds"] for i in ids],
        "shard_timings": [
            {k: r[k] for k in ("shard", "elapsed_seconds", "generation_seconds", "load_seconds")}
            for r in sorted(records, key=lambda r: r["shard"])
        ],
        "predictions": predictions,
        "test_consumed": False,
    }
    if stage == "discovery":
        rule = plan["stages"]["discovery"]["uncertainty"]
        half = named_baseline().hoeffding_half_width(score["n"], rule["confidence_level"])
        mean = score["micro_accuracy"]
        report["metric"]["hoeffding"] = {
            "confidence_level": rule["confidence_level"],
            "half_width": half,
            "interval": [max(0.0, mean - half), min(1.0, mean + half)],
        }
        report["decision"] = discovery_decision(plan, report)
    else:
        endpoints = plan["analysis"]["endpoint_predictions"]
        path = Path(repository) / endpoints["path"]
        if sha_file(path) != endpoints["sha256"]:
            raise ValueError("Committed endpoint predictions differ from their pin")
        arms = read_json(path)["arms"]
        rule = plan["comparison"]
        report["comparisons"] = {
            arm: classification.compare_classification(
                inputs["gold"],
                arms[arm]["predictions"],
                predictions,
                labels,
                resamples=rule["resamples"],
                seed=rule["seed"],
            )
            for arm in ENDPOINTS
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("matrix", "fetch-images", "fetch-model", "run-shard", "merge", "gate"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--repository", type=Path, default=REPOSITORY)
        if name in ("matrix", "fetch-model", "run-shard", "merge", "gate"):
            command.add_argument("--model", required=True)
        if name != "gate":
            command.add_argument("--expected-plan-sha256", required=True)
        if name in ("matrix", "fetch-images", "run-shard", "merge"):
            command.add_argument("--stage", choices=STAGES, required=True)
        if name in ("fetch-model", "run-shard"):
            command.add_argument("--model-directory", type=Path, required=True)
        if name == "run-shard":
            command.add_argument("--shard", type=int, required=True)
        if name in ("matrix", "run-shard", "merge"):
            command.add_argument("--shards", type=int, required=True)
        if name in ("run-shard", "merge"):
            command.add_argument("--output", type=Path, required=True)
        if name == "merge":
            command.add_argument("--records", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "matrix":
            # Standard library only: refuses before any runner downloads or loads anything.
            plan = load_plan(args.plan, args.expected_plan_sha256)
            model_entry(plan, args.model)
            want = expected_shards(plan, args.model, args.stage, args.repository)
            if args.shards != want:
                raise ValueError(
                    f"Shard count {args.shards} differs from the pre-registered {want}"
                )
            stage_order(plan, args.stage, args.repository)
            print(ci_shards.compact(list(range(want))))
            return 0
        if args.command == "gate":
            plan = load_plan(args.plan, require_activated=False)
            path = args.repository / plan["stages"]["full"]["discovery_report"].format(
                model=args.model
            )
            result = discovery_decision(plan, read_json(path))
        elif args.command == "fetch-images":
            plan = load_plan(args.plan, args.expected_plan_sha256)
            result = fetch_images(plan, args.stage, args.repository)
        elif args.command == "fetch-model":
            plan = load_plan(args.plan, args.expected_plan_sha256)
            result = fetch_model(plan, args.model, args.model_directory)
        elif args.command == "run-shard":
            result = run_shard(
                args.plan,
                args.expected_plan_sha256,
                args.model,
                args.stage,
                args.shard,
                args.shards,
                args.model_directory,
                args.output,
                args.repository,
            )
        else:
            plan = load_plan(args.plan, args.expected_plan_sha256)
            inputs = named_baseline().stage_inputs(plan, args.stage, args.repository)
            report = merge(
                plan,
                args.expected_plan_sha256,
                args.model,
                args.stage,
                args.shards,
                args.records,
                inputs,
                args.repository,
            )
            write_json(args.output, report)
            result = {k: report[k] for k in ("model", "stage", "status", "n", "metric")}
            result["decision"] = report.get("decision")
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ci named baseline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
