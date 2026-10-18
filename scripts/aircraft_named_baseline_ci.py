"""Named open-weight baseline on FGVC-Aircraft native VAL, sharded over free GitHub runners.

Pre-registered by docs/aircraft-named-baseline-ci-plan-v1.json. The committed plan is
UNACTIVATED and every VAL command refuses it: a VAL run needs a root-approved copy with
``activated: true`` and an ``activation`` record, committed before dispatch, whose SHA256 the
dispatcher passes. Importing this module performs no work and imports no model code.

Nothing here scores anything new. Inference, parsing and scoring are the eight-endpoint
native-VAL evaluator's own code, materialised byte for byte from commit 85f0865 and checked
against the source SHA256 list in reports/2026-10-17-controlled400-native-val/plan.json:

  prompt    vlm_local_training.prompt_text(spec, "answer_only"), via _processor/_inputs
  predict   vlm_local_evaluation.Backend.predict (greedy, _generation_kwargs, EOS rule)
  parse     vlm_local_training.parse_label(..., allow_reasoning=True), terminated only
  metric    classification_comparison._score / compare_classification (20,000 draws, 2718)

The only new code is plumbing: pinned downloads, a per-shard journal, the shard partition
and the merge, which refuses unless every image of the stage was scored exactly once.

Commands:
  matrix    validate a dispatch (activation, shard count, discovery gate) and print the matrix
  shard     run one shard (one GitHub job); writes journal.jsonl and shard.json
  merge     verify complete, exactly-once coverage from one run, then score
  plumbing  the PR check: same shard path on three pinned TRAIN images, never scored
  verify-endpoints  the PR check that the committed endpoint export reproduces the report

Blind spots, stated on purpose: a shard's journal and record are runner-local files carried
by a GitHub artifact, not a tamper-proof witness; the run-attempt check proves that no shard
of the merged run was re-run, not that no earlier run was dispatched (the plan requires every
dispatched run to be reported); latency on shared runners is noisy, so a too-small shard count
fails closed as a gap rather than being predicted exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/aircraft-named-baseline-ci-plan-v1.json"
PLAN_ID = "aircraft-named-baseline-ci-qwen2.5-vl-3b-v1"
STAGES = ("plumbing", "discovery", "full")
VAL_STAGES = ("discovery", "full")
NATIVE_IMAGES, NATIVE_CLASSES = 3333, 100
SHARD_RECORD = "shard.json"
FROZEN_PREFIX = "/Users/idide/RyanLinEnv/runs/aircraft-native-val-activation-v1/frozen/"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ATTEMPTS = 3
# The only fields an activated copy may differ in from the committed pre-registration.
ACTIVATION_FIELDS = ("activated", "activation", "status")


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ids_sha256(ids):
    return sha_bytes(compact(list(ids)).encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def repo_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{relative} escapes the repository")
    return path


def pinned(entry):
    """Read a committed {path, sha256} input and refuse any byte difference."""
    path = repo_path(entry["path"])
    if sha_file(path) != entry["sha256"]:
        raise ValueError(f"Pinned input changed: {entry['path']}")
    return path


# --------------------------------------------------------------------------- plan


def load_plan(path, expected_sha256=None, *, require_activated=True):
    path = Path(path)
    plan = read_json(path)
    if plan.get("plan_id") != PLAN_ID or plan.get("schema_version") != 1:
        raise ValueError("Not the named-baseline CI plan schema")
    if require_activated:
        activation = plan.get("activation")
        if (
            plan.get("activated") is not True
            or not isinstance(activation, dict)
            or not activation.get("activated_at")
            or not activation.get("approved_by")
        ):
            raise ValueError("Named-baseline CI plan is unactivated; VAL stages must not run")
        if expected_sha256 is None or sha_file(path) != expected_sha256:
            raise ValueError("Pass the activated plan's exact SHA256 to run it")
        committed = read_json(PLAN)
        if ({k: v for k, v in plan.items() if k not in ACTIVATION_FIELDS}
                != {k: v for k, v in committed.items() if k not in ACTIVATION_FIELDS}):
            raise ValueError("Activated copy changes pre-registered fields; only "
                             + ", ".join(ACTIVATION_FIELDS) + " may differ")
    return plan


def require_stage_plan(plan_path, plan_sha256, stage):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    if stage == "plumbing":
        # Plumbing never opens VAL and is never scored; it runs under the committed plan only.
        if Path(plan_path).resolve() != PLAN or plan_sha256 != sha_file(PLAN):
            raise ValueError("Plumbing runs only under the committed plan")
        return load_plan(PLAN, require_activated=False)
    return load_plan(plan_path, plan_sha256)


def val_rows(plan):
    rows = read_json(pinned(plan["inputs"]["val_images"]))
    if len(rows) != NATIVE_IMAGES or any(row["split"] != "val" for row in rows):
        raise ValueError("Require all 3333 native VAL author rows and nothing else")
    if len({row["image_id"] for row in rows}) != NATIVE_IMAGES:
        raise ValueError("Native VAL image IDs must be unique")
    return rows


def stage_images(plan, stage):
    """The stage's ordered image rows: id, encoded and model-visible hashes, bytes, URL."""
    if stage == "plumbing":
        rows = plan["stages"]["plumbing"]["images"]
        if ids_sha256([r["id"] for r in rows]) != plan["stages"]["plumbing"]["ids_sha256"]:
            raise ValueError("Plumbing manifest differs from the plan")
        return [dict(row) for row in rows]
    author = val_rows(plan)
    by_id = {row["image_id"]: row for row in author}
    declared = plan["stages"][stage]
    if stage == "discovery":
        source = declared["ids_source"]
        identifiers = read_json(pinned(source))["stages"]["discovery"]["ids"]
    else:
        identifiers = [row["image_id"] for row in author]
    if len(identifiers) != declared["images"] or ids_sha256(identifiers) != declared["ids_sha256"]:
        raise ValueError(f"{stage} manifest differs from the pre-registered IDs")
    mirror = plan["inputs"]["image_mirror"]
    images = []
    for identifier in identifiers:
        row = by_id[identifier]  # KeyError: an ID outside native VAL is never fetched
        if row["path"] != f"development/images/validation/{identifier}.jpg":
            raise ValueError("Only original native-VAL JPEG paths may be predicted")
        images.append({
            "id": identifier,
            "path": row["path"],
            "url": mirror["url_template"].format(revision=mirror["revision"], id=identifier),
            "bytes": row["encoded_bytes"],
            "encoded_sha256": row["encoded_sha256"],
            "model_rgb_sha256": row["model_rgb_sha256"],
        })
    return images


def partition(items, shards, index):
    """Author-order item i belongs to shard i mod N; each shard keeps author order."""
    if type(shards) is not int or type(index) is not int or not 0 <= index < shards:
        raise ValueError("Shard index must be an integer in [0, shards)")
    return list(items)[index::shards]


def shard_bounds(plan, stage, shards):
    declared = plan["sharding"]["shards"][stage]
    if type(shards) is not int or not declared["min"] <= shards <= declared["max"]:
        raise ValueError(
            f"{stage} shard count must be an integer in [{declared['min']}, {declared['max']}]"
        )
    if "declared" in declared and shards != declared["declared"]:
        raise ValueError(f"{stage} shard count is pre-registered as {declared['declared']}")


def required_full_shards(plan, receipt):
    """Pre-registered rule: the smallest N whose busiest shard fits the shard wall budget.

    Per image: 1.5 x (discovery mean generation latency + mean per-image fetch seconds), since
    a shard fetches all its images during setup. Fixed: the largest discovery setup excluding
    image fetches (install-free: model download, rehash, load), plus one per-image deadline.
    """
    rule = plan["stages"]["full"]["gate"]
    sharding, timing = plan["sharding"], receipt["timing"]
    usable = (sharding["shard_wall_seconds"] - sharding["image_seconds"]
              - timing["max_fixed_setup_seconds"])
    per_image = rule["latency_safety_factor"] * (timing["mean_image_latency_seconds"]
                                                 + timing["mean_image_fetch_seconds"])
    capacity = math.floor(usable / per_image) if usable > 0 and per_image > 0 else 0
    if capacity < 1:
        raise ValueError("Discovery latency leaves no image capacity per shard")
    return math.ceil(NATIVE_IMAGES / capacity)


def discovery_gate(plan, plan_sha256, receipt_path, shards):
    """Full VAL starts only from a committed, completed discovery receipt of this plan."""
    receipt = read_json(repo_path(receipt_path))
    rule = plan["stages"]["full"]["gate"]
    reasons = []
    if receipt.get("plan_id") != PLAN_ID or receipt.get("plan_sha256") != plan_sha256:
        reasons.append("discovery receipt belongs to another plan or plan copy")
    if receipt.get("stage") != "discovery" or receipt.get("status") != "completed":
        reasons.append("discovery receipt is not a completed discovery merge")
    counts = receipt.get("counts", {})
    if counts.get("completed") != plan["stages"]["discovery"]["images"]:
        reasons.append("discovery did not complete every image")
    if receipt.get("valid_fraction", 0) < rule["minimum_valid_answer_fraction"]:
        reasons.append("valid-answer fraction below the pre-registered floor")
    if not reasons:
        needed = required_full_shards(plan, receipt)
        if shards < needed:
            reasons.append(f"full VAL needs at least {needed} shards by the pre-registered rule")
    if reasons:
        raise ValueError("Full VAL refused by the discovery gate: " + "; ".join(reasons))


def matrix(plan_path, plan_sha256, stage, shards, discovery_receipt=None):
    if stage not in VAL_STAGES:
        raise ValueError("Only discovery and full VAL are dispatched as a matrix")
    plan = load_plan(plan_path, plan_sha256)
    shard_bounds(plan, stage, shards)
    if stage == "full":
        if not discovery_receipt:
            raise ValueError("Full VAL requires the committed discovery receipt")
        discovery_gate(plan, plan_sha256, discovery_receipt, shards)
    stage_images(plan, stage)
    return {"shard": list(range(shards))}


# ------------------------------------------------------------------ frozen evaluator


def frozen_sources(plan):
    evaluator = read_json(pinned(plan["evaluator_identity"]["plan"]))
    sources = {}
    for name, digest in evaluator["source_sha256"].items():
        if not name.startswith(FROZEN_PREFIX):
            raise ValueError("Unexpected frozen evaluator source path")
        sources[name.removeprefix(FROZEN_PREFIX)] = digest
    if len(sources) != plan["evaluator_identity"]["source_files"]:
        raise ValueError("Frozen evaluator source list changed")
    return sources


def materialize_evaluator(plan, destination):
    """Write the evaluator sources exactly as committed at the frozen commit."""
    commit = plan["evaluator_identity"]["commit"]
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Never reuse a frozen evaluator directory")
    for relative, digest in frozen_sources(plan).items():
        data = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
            check=True, capture_output=True, timeout=60,
        ).stdout
        if sha_bytes(data) != digest:
            raise ValueError(f"Frozen evaluator source differs at {commit}: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return destination / "src"


def load_frozen(plan, source_root):
    """Import the frozen package, never the working tree's, and prove which file ran."""
    source_root = Path(source_root).resolve()
    if any(name == "vision_rl" or name.startswith("vision_rl.") for name in sys.modules):
        raise ValueError("vision_rl was imported before the frozen evaluator")
    sys.path.insert(0, str(source_root))
    names = ("vlm_local_training", "vlm_local_evaluation", "vlm_local_task_builder",
             "classification_comparison", "aircraft_data")
    modules = {name: importlib.import_module(f"vision_rl.{name}") for name in names}
    expected = frozen_sources(plan)
    for module in [m for n, m in sys.modules.items() if n.startswith("vision_rl")]:
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f"Executed evaluator source is not the frozen copy: {path}")
        relative = str(path.relative_to(source_root.parent))
        if relative not in expected or sha_file(path) != expected[relative]:
            raise ValueError(f"Executed evaluator source is not the frozen copy: {path}")
    return modules


def evaluator_contract(plan, frozen):
    """Prompt, labels and parser must equal the eight-endpoint evaluator's."""
    native = frozen["vlm_local_training"]
    spec = native.load_spec(pinned(plan["inputs"]["data_spec"]))
    evaluator = read_json(pinned(plan["evaluator_identity"]["plan"]))
    prompt = native.prompt_text(spec, "answer_only")
    identity = plan["evaluator_identity"]
    if (
        spec["labels"] != evaluator["labels"]
        or len(spec["labels"]) != NATIVE_CLASSES
        or sha_bytes(prompt.encode()) != identity["prompt_sha256"]
        or evaluator["prompt_sha256"] != identity["prompt_sha256"]
    ):
        raise ValueError("Prompt or vocabulary differs from the eight-endpoint evaluator")
    return spec


# ------------------------------------------------------------------------ downloads


@functools.cache
def _download_module():
    path = ROOT / "scripts/aircraft_ci_experiment.py"
    spec = importlib.util.spec_from_file_location("aircraft_ci_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # importing performs no work (module docstring)
    return module


def fetch(url, destination, *, size, sha256):
    """The CPU7946 job's bounded, hash-checked download, with a bounded retry."""
    download = _download_module().download
    for attempt in range(1, ATTEMPTS + 1):
        Path(str(destination) + ".part").unlink(missing_ok=True)
        try:
            return download(url, Path(destination), size=size, sha256=sha256)
        except (OSError, ValueError):
            if attempt == ATTEMPTS:
                raise
            time.sleep(10 * attempt)


def fetch_model(plan, directory):
    manifest_path = pinned(plan["baseline_identity"]["manifest"])
    manifest = read_json(manifest_path)
    if (manifest["model_repository"] != MODEL_NAME
            or manifest["revision"] != plan["baseline_identity"]["revision"]):
        raise ValueError("Model manifest differs from the pre-registered baseline")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError("Model files must stay in the model directory")
        url = (f"https://huggingface.co/{MODEL_NAME}/resolve/{manifest['revision']}/"
               f"{entry['path']}")
        fetch(url, directory / relative, size=entry["size_bytes"], sha256=entry["sha256"])
    shutil.copyfile(manifest_path, directory / "model-source.json")
    write_json(directory / ".pinned-download.json", {
        "repository": MODEL_NAME, "revision": manifest["revision"],
        "manifest_sha256": sha_file(directory / "model-source.json"),
    })


def fetch_images(images, root):
    for row in images:
        target = Path(root) / row["path"]
        fetch(row["url"], target, size=row["bytes"], sha256=row["encoded_sha256"])
        row["image"] = str(target)


# ---------------------------------------------------------------------------- shard


def _append(path, value):
    with Path(path).open("a") as handle:
        handle.write(compact(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_identity():
    names = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REPOSITORY")
    values = {name: os.environ.get(name) for name in names}
    if not all(values.values()):
        raise ValueError("Shards run only inside a GitHub Actions run")
    return {"run_id": values["GITHUB_RUN_ID"], "run_attempt": values["GITHUB_RUN_ATTEMPT"],
            "sha": values["GITHUB_SHA"], "repository": values["GITHUB_REPOSITORY"]}


def host_facts():
    memory = None
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory = int(line.split()[1]) * 1024
    return {"cpu_count": os.cpu_count(), "mem_total_bytes": memory,
            "disk_free_bytes": shutil.disk_usage(Path.cwd()).free}


def load_backend(plan, frozen, model_directory):
    """A frozen evaluator Backend holding the named model: no adapter, no trainable weights."""
    native, evaluation = frozen["vlm_local_training"], frozen["vlm_local_evaluation"]
    safe = frozen["vlm_local_task_builder"]
    generation = plan["generation"]
    profile = native.Profile(**generation["profile"])
    profile.validate()
    environment = native.preflight("cpu")
    versions = safe._check_runtime()
    verified = native._verified_model_directory(Path(model_directory), MODEL_NAME)
    torch, transformers = (importlib.import_module(n) for n in ("torch", "transformers"))
    torch.set_num_threads(min(generation["torch_threads"], os.cpu_count() or 1))
    transformers.set_seed(42)
    spec = evaluator_contract(plan, frozen)
    backend = object.__new__(evaluation.Backend)
    backend.torch, backend.profile, backend.device, backend.spec = torch, profile, "cpu", spec
    backend.processor = native._processor(Path(model_directory), profile)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        str(model_directory), trust_remote_code=False, local_files_only=True,
        use_safetensors=True, dtype=getattr(torch, generation["profile"]["dtype"]),
        attn_implementation="sdpa",
    )
    if model.config.model_type != generation["model_type"]:
        raise ValueError("Unexpected pinned model architecture")
    for key, value in generation["generation_config_overrides"].items():
        setattr(model.generation_config, key, value)
    model.requires_grad_(False).to("cpu").eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The named baseline must have zero trainable parameters")
    backend.policy = model
    eos = model.generation_config.eos_token_id
    backend.eos = set(([eos] if isinstance(eos, int) else list(eos or []))
                      + [backend.processor.tokenizer.eos_token_id])
    identity = {
        "environment": {k: v for k, v in environment.items() if k != "adapter_source_sha256"},
        "runtime": versions,
        "model_manifest_sha256": verified["manifest_sha256"],
        "generation_config": model.generation_config.to_dict(),
        "generation_kwargs": evaluation._generation_kwargs(generation["profile"]),
        "profile": generation["profile"],
        "prompt_sha256": plan["evaluator_identity"]["prompt_sha256"],
        "trainable_parameters": 0,
        "torch_threads": torch.get_num_threads(),
    }
    return backend, identity


def run_shard(plan_path, plan_sha256, stage, shards, index, work, output, *, clock=time.monotonic):
    started = clock()
    plan = require_stage_plan(plan_path, plan_sha256, stage)
    if stage == "plumbing":
        if shards != 1 or index != 0:
            raise ValueError("Plumbing is a single shard")
    else:
        shard_bounds(plan, stage, shards)
    run = run_identity()
    images = partition(stage_images(plan, stage), shards, index)
    work, output = Path(work).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Never reuse a shard output directory")
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "journal.jsonl"
    record = {
        "schema_version": 1, "plan_id": PLAN_ID, "plan_sha256": plan_sha256, "stage": stage,
        "shards": shards, "index": index, "run": run, "host": host_facts(),
        "ids": [row["id"] for row in images], "results": {}, "status": "started",
        "timing": {},
    }
    write_json(output / SHARD_RECORD, record)
    frozen = load_frozen(plan, materialize_evaluator(plan, work / "frozen"))
    pixels = frozen["aircraft_data"].aircraft_pixel_sha256
    native, evaluation = frozen["vlm_local_training"], frozen["vlm_local_evaluation"]
    fetching = clock()
    fetch_images(images, work / "data")
    record["timing"]["image_fetch_seconds"] = clock() - fetching
    fetch_model(plan, work / "model")
    loading = clock()
    backend, identity = load_backend(plan, frozen, work / "model")
    record["identity"] = {
        **identity, "plan_sha256": plan_sha256, "stage": stage, "shards": shards,
        "run_id": run["run_id"], "sha": run["sha"],
        "evaluator_commit": plan["evaluator_identity"]["commit"],
        "frozen_sources_sha256": sha_bytes(compact(frozen_sources(plan)).encode()),
    }
    setup = clock() - started
    record["timing"].update(setup_seconds=setup, load_seconds=clock() - loading,
                            fixed_setup_seconds=setup - record["timing"]["image_fetch_seconds"])
    sharding, labels = plan["sharding"], backend.spec["labels"]
    limit = plan["generation"]["profile"]["max_completion_length"]
    status = "failed"  # anything that escapes the loop below leaves the shard failed
    try:
        for row in images:
            if clock() - started + sharding["image_seconds"] > sharding["shard_wall_seconds"]:
                status = "partial"
                break
            if (sha_file(row["image"]) != row["encoded_sha256"]
                    or pixels(row["image"]) != row["model_rgb_sha256"]):
                raise ValueError("Model-visible image changed")
            _append(journal, {"event": "reserved", "id": row["id"], "time": time.time()})
            tick = clock()
            try:
                with evaluation._deadline(sharding["image_seconds"]):
                    result = backend.predict({"id": row["id"], "image": row["image"]})
                # The frozen run_chunk's output checks and parse, inside its error handler.
                if (
                    type(result["terminated"]) is not bool
                    or not isinstance(result["response"], str)
                    or len(result["response"]) > 65536
                    or len(result["completion_ids"]) > limit
                ):
                    raise ValueError("Malformed or oversized inference output")
                result.update(
                    status="completed",
                    canonical_label=native.parse_label(result["response"], labels,
                                                       allow_reasoning=True)
                    if result["terminated"] else None,
                    latency_seconds=clock() - tick,
                )
            except KeyboardInterrupt:
                status = "unknown_after_interruption"
                _append(journal, {"event": "outcome", "id": row["id"], "result": {
                    "status": status, "canonical_label": None,
                    "latency_seconds": clock() - tick}})
                record["results"][row["id"]] = {"status": status, "canonical_label": None}
                break
            except Exception as exc:  # the evaluator records the error and stops
                status = "generation_error"
                failure = {"status": status, "error": f"{type(exc).__name__}: {exc}",
                           "canonical_label": None, "latency_seconds": clock() - tick}
                _append(journal, {"event": "outcome", "id": row["id"], "result": failure})
                record["results"][row["id"]] = failure
                break
            _append(journal, {"event": "outcome", "id": row["id"], "result": result})
            record["results"][row["id"]] = result
        else:
            status = "completed"
    finally:
        record["status"] = status
        record["timing"].update(
            elapsed_seconds=clock() - started,
            generation_seconds=sum(r.get("latency_seconds", 0)
                                   for r in record["results"].values()),
        )
        write_json(output / SHARD_RECORD, record)
    if status != "completed" or len(record["results"]) != len(images):
        raise ValueError(f"Shard {index} of {shards} ended {status}; the merge will refuse it")
    return record


# ---------------------------------------------------------------------------- merge


def verify_shards(records, expected_ids, shards, *, stage, plan_sha256, run=None):
    """Exactly-once coverage from one run, or refuse. Returns {id: result} in author order.

    ``records`` maps shard index to its parsed shard.json (None or absent when missing).
    """
    present = {index for index, record in records.items() if record is not None}
    missing = sorted(set(range(shards)) - present)
    extra = sorted(present - set(range(shards)))
    if missing or extra:
        raise ValueError(f"Shard coverage refused: missing {missing}, unexpected {extra}")
    identity = None
    merged, counts = {}, Counter()
    for index in range(shards):
        record = records[index]
        if (record.get("plan_id") != PLAN_ID or record.get("plan_sha256") != plan_sha256
                or record.get("stage") != stage or record.get("shards") != shards
                or record.get("index") != index):
            raise ValueError(f"Shard {index} belongs to another plan, stage or partition")
        if record.get("status") != "completed":
            raise ValueError(f"Shard {index} did not complete: {record.get('status')}")
        if record["run"]["run_attempt"] != "1":
            raise ValueError(f"Shard {index} was re-run; regenerated outcomes are refused")
        if run is not None and {k: record["run"][k] for k in ("run_id", "sha")} != run:
            raise ValueError(f"Shard {index} comes from another workflow run or commit")
        if identity is None:
            identity = record.get("identity")
        if not identity or record.get("identity") != identity:
            raise ValueError(f"Shard {index} ran under a different model/evaluator identity")
        if record["ids"] != partition(expected_ids, shards, index):
            raise ValueError(f"Shard {index} scored the wrong images")
        if list(record["results"]) != record["ids"]:
            raise ValueError(f"Shard {index} has a gap or extra outcome")
        for identifier, result in record["results"].items():
            if result.get("status") != "completed":
                raise ValueError(f"Shard {index} has a non-completed outcome: {identifier}")
            counts[identifier] += 1
            merged[identifier] = result
    if set(counts) != set(expected_ids) or any(value != 1 for value in counts.values()):
        raise ValueError("Images were not each scored exactly once")
    return {identifier: merged[identifier] for identifier in expected_ids}, identity


def read_shard_records(artifacts, stage, shards):
    """Shard records from downloaded artifacts named named-baseline-<stage>-shard-<i>."""
    artifacts, records = Path(artifacts), {}
    prefix = f"named-baseline-{stage}-shard-"
    for directory in sorted(artifacts.iterdir()) if artifacts.is_dir() else []:
        if not directory.name.startswith(prefix):
            raise ValueError(f"Unexpected artifact in the merge input: {directory.name}")
        suffix = directory.name.removeprefix(prefix)
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise ValueError(f"Malformed shard artifact name: {directory.name}")
        record = directory / SHARD_RECORD
        records[int(suffix)] = read_json(record) if record.is_file() else None
    return records


def hoeffding_half_width(n, confidence):
    return math.sqrt(math.log(2 / (1 - confidence)) / (2 * n))


def verify_endpoint_export(plan, frozen, gold, labels):
    """The committed endpoint export must reproduce the committed report exactly."""
    comparison = frozen["classification_comparison"]
    export = read_json(pinned(plan["analysis"]["endpoint_predictions"]))
    report = read_json(pinned(plan["analysis"]["endpoint_report"]))
    rule = plan["analysis"]["bootstrap"]
    for arm in plan["analysis"]["reproduce_against_report"]:
        observed = comparison.compare_classification(
            gold, export["arms"][arm], export["arms"]["common"], labels,
            resamples=rule["resamples"], seed=rule["seed"],
        )
        if json.loads(json.dumps(observed)) != report["versus_common_initialization"][arm]:
            raise ValueError(f"Endpoint export does not reproduce the committed report: {arm}")
    if report["strongest_declared_sft"] != plan["analysis"]["comparators"]["strongest_sft"]:
        raise ValueError("The committed strongest declared SFT differs from the plan")
    return export


def gold_rows(plan):
    validation = pinned(plan["inputs"]["validation"])
    return [{"id": row["id"], "label": row["label"]}
            for row in map(json.loads, validation.read_text().splitlines())]


def verify_endpoints(work):
    """Plumbing-time check, no inference: the frozen comparison reproduces the committed
    endpoint report from the committed export. Runs under the committed plan."""
    plan = load_plan(PLAN, require_activated=False)
    frozen = load_frozen(plan, materialize_evaluator(plan, Path(work) / "frozen"))
    labels = evaluator_contract(plan, frozen)["labels"]
    export = verify_endpoint_export(plan, frozen, gold_rows(plan), labels)
    return {"stage": "endpoint-export", "status": "reproduced",
            "arms": list(export["arms"]),
            "reproduced": plan["analysis"]["reproduce_against_report"]}


def merge(plan_path, plan_sha256, stage, shards, artifacts, work, output):
    if stage not in VAL_STAGES:
        raise ValueError("Only VAL stages are merged; plumbing is never scored")
    plan = load_plan(plan_path, plan_sha256)
    shard_bounds(plan, stage, shards)
    expected = [row["id"] for row in stage_images(plan, stage)]
    run = run_identity()
    records = read_shard_records(artifacts, stage, shards)
    results, identity = verify_shards(
        records, expected, shards,
        stage=stage, plan_sha256=plan_sha256,
        run={"run_id": run["run_id"], "sha": run["sha"]},
    )
    frozen = load_frozen(plan, materialize_evaluator(plan, Path(work) / "frozen"))
    native, comparison = frozen["vlm_local_training"], frozen["classification_comparison"]
    labels = evaluator_contract(plan, frozen)["labels"]
    predictions = {}
    for identifier, result in results.items():
        parsed = (native.parse_label(result["response"], labels, allow_reasoning=True)
                  if result["terminated"] else None)
        if parsed != result["canonical_label"]:
            raise ValueError(f"Canonical label differs from the frozen parser: {identifier}")
        predictions[identifier] = parsed
    gold_all = gold_rows(plan)
    gold = [row for row in gold_all if row["id"] in predictions]
    rows = comparison._inputs(gold, (predictions,), labels)
    score, _ = comparison._score(rows, predictions, labels)
    latencies = [r["latency_seconds"] for r in results.values()]
    receipt = {
        "schema_version": 1, "plan_id": PLAN_ID, "plan_sha256": plan_sha256, "stage": stage,
        "status": "completed", "shards": shards, "run": run, "identity": identity,
        "ids_sha256": ids_sha256(expected),
        "counts": {"completed": len(results), "denominator": len(expected)},
        "valid_fraction": sum(v is not None for v in predictions.values()) / len(expected),
        "unterminated": sum(not r["terminated"] for r in results.values()),
        "timing": {
            "mean_image_latency_seconds": sum(latencies) / len(latencies),
            "max_image_latency_seconds": max(latencies),
            "max_setup_seconds": max(r["timing"]["setup_seconds"] for r in records.values()),
            "max_fixed_setup_seconds": max(r["timing"]["fixed_setup_seconds"]
                                           for r in records.values()),
            "mean_image_fetch_seconds": sum(r["timing"]["image_fetch_seconds"]
                                            for r in records.values()) / len(expected),
            "max_shard_elapsed_seconds": max(r["timing"]["elapsed_seconds"]
                                             for r in records.values()),
        },
        "metric": score,
        "test_consumed": False,
    }
    if stage == "discovery":
        rule = plan["stages"]["discovery"]["uncertainty"]
        half = hoeffding_half_width(score["n"], rule["confidence_level"])
        mean = score["micro_accuracy"]
        receipt["hoeffding"] = {"confidence_level": rule["confidence_level"],
                                "half_width": half,
                                "interval": [max(0.0, mean - half), min(1.0, mean + half)]}
    else:
        export = verify_endpoint_export(plan, frozen, gold_all, labels)
        rule = plan["analysis"]["bootstrap"]
        receipt["comparisons"] = {
            role: comparison.compare_classification(
                gold_all, predictions, export["arms"][arm], labels,
                resamples=rule["resamples"], seed=rule["seed"],
            )
            for role, arm in plan["analysis"]["comparators"].items()
        }
    output = Path(output)
    write_json(output / f"{stage}-receipt.json", receipt)
    write_json(output / f"{stage}-predictions.json",
               {"plan_sha256": plan_sha256, "predictions": predictions,
                "raw": {k: {"response": r["response"], "raw_response": r["raw_response"],
                            "terminated": r["terminated"]} for k, r in results.items()}})
    return receipt


# ----------------------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("matrix", "shard", "merge", "plumbing", "verify-endpoints"):
        command = commands.add_parser(name)
        if name in ("matrix", "shard", "merge"):
            command.add_argument("--plan", type=Path, required=True)
            command.add_argument("--plan-sha256", required=True)
            command.add_argument("--stage", choices=VAL_STAGES, required=True)
            command.add_argument("--shards", type=int, required=True)
        if name == "matrix":
            command.add_argument("--discovery-receipt")
        if name == "shard":
            command.add_argument("--index", type=int, required=True)
        if name != "matrix":
            command.add_argument("--work", type=Path, required=True)
        if name in ("shard", "merge", "plumbing"):
            command.add_argument("--output", type=Path, required=True)
        if name == "merge":
            command.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "matrix":
            result = matrix(args.plan, args.plan_sha256, args.stage, args.shards,
                            args.discovery_receipt)
            print(compact(result))
            return 0
        if args.command == "shard":
            result = run_shard(args.plan, args.plan_sha256, args.stage, args.shards,
                               args.index, args.work, args.output)
        elif args.command == "plumbing":
            result = run_shard(PLAN, sha_file(PLAN), "plumbing", 1, 0, args.work, args.output)
        elif args.command == "verify-endpoints":
            result = verify_endpoints(args.work)
        else:
            result = merge(args.plan, args.plan_sha256, args.stage, args.shards,
                           args.artifacts, args.work, args.output)
        summary = {k: result[k] for k in ("stage", "status") if k in result}
        summary.update(timing=result.get("timing"), metric=result.get("metric"))
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"named baseline CI: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
