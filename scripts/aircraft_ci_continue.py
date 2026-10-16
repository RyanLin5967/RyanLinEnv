"""Single-use, provenance-pinned continuation of the two CPU7946 first-four jobs.

No training implementation lives here. Every model phase calls the unchanged
frozen worker through the existing CPU supervisor. Importing performs no work.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HELPER = Path(__file__).with_name("aircraft_ci_experiment.py")
if not HELPER.exists() and Path(__file__).parent == Path("/ci"):
    HELPER = Path("/ci/experiment.py")
SPEC = importlib.util.spec_from_file_location("aircraft_first4", HELPER)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)
ARMS = ("grpo", "dr_grpo")
FIRST4_WORKFLOW = ".github/workflows/aircraft-cpu-experiment.yml"
CONTINUATION_WORKFLOW = ".github/workflows/aircraft-cpu-continue.yml"
DESCRIPTOR = Path("docs/aircraft-cpu-experiment-plan.json")
LOADER_FILE = "scripts/aircraft_cpu_loader.py"
FIRST4_FILES = ("scripts/aircraft_ci_experiment.py", LOADER_FILE, FIRST4_WORKFLOW, str(DESCRIPTOR))
ASSAYS = ("greedy", "sample42", "sample314")
JOB_SECONDS = 6 * 3600


def object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hex_sha(value, size=64):
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{size}}}", value) is None:
        raise ValueError("Require an exact prospective SHA")
    return value


def envelope(raw):
    if (
        set(raw)
        != {
            "schema_version",
            "repository",
            "parent_run_id",
            "parent_commit",
            "first4_files",
            "artifacts",
        }
        or raw["schema_version"] != 1
    ):
        raise ValueError("Require the exact reviewed continuation envelope")
    if (
        raw["repository"] != ci.REPOSITORY
        or type(raw["parent_run_id"]) is not int
        or raw["parent_run_id"] <= 0
    ):
        raise ValueError("Continuation requires a unique owned parent run")
    hex_sha(raw["parent_commit"], 40)
    if set(raw["first4_files"]) != set(FIRST4_FILES) or set(raw["artifacts"]) != set(ARMS):
        raise ValueError(
            "Pin both first-four arms and all original workflow/descriptor/controller files"
        )
    for digest in raw["first4_files"].values():
        hex_sha(digest)
    for item in raw["artifacts"].values():
        if (
            set(item) != {"id", "sha256", "bytes"}
            or type(item["id"]) is not int
            or item["id"] <= 0
            or type(item["bytes"]) is not int
            or not 0 < item["bytes"] <= 2 * ci.GIB
        ):
            raise ValueError("Pin exact bounded parent artifact IDs, ZIP bytes and SHA256")
        hex_sha(item["sha256"])
    if raw["artifacts"]["grpo"]["id"] == raw["artifacts"]["dr_grpo"]["id"]:
        raise ValueError("The two arm artifacts must be distinct")
    return raw


def descriptor_path():
    # The container must receive the exact reviewed descriptor, never a fallback.
    return Path("/ci/experiment-plan.json") if Path(__file__).parent == Path("/ci") else DESCRIPTOR


def checked_loader(plan):
    contract = ci.read(descriptor_path())
    identity = ci.loader_identity()
    if (
        contract.get("loader") != identity
        or identity.get("file") != LOADER_FILE
        or identity.get("sha256") != plan["first4_files"].get(LOADER_FILE)
        or ci.digest(ci.regular(descriptor_path())) != plan["first4_files"][str(DESCRIPTOR)]
    ):
        raise ValueError("Frozen descriptor/envelope loader identity differs from actual bootstrap")
    return identity, contract


def loader_model_directory(identity, contract):
    shards = [
        Path(row["path"])
        for row in contract["downloads"]
        if Path(row["path"]).name in identity["shards"]
    ]
    if len(shards) != len(identity["shards"]) or len({p.parent for p in shards}) != 1:
        raise ValueError("Frozen descriptor does not identify the exact loader base shards")
    return str(ci.ORIGINAL / shards[0].parent)


def verify_loader_phases(output, identity, contract):
    for phase, guard in ci.completed_phases(output).items():
        if (
            guard.get("loader") != identity
            or guard.get("phase") != phase
            or guard.get("exit_code") != 0
        ):
            raise ValueError("Phase guard loader identity differs from the prospective descriptor")
        record = ci.read(output / "phases" / phase / "loader.json")
        if (
            any(record.get(key) != value for key, value in identity.items())
            or record.get("status") != "completed"
            or "error_type" in record
            or record.get("aliases_restored") is not True
            or record.get("environment_restored") is not True
            or record.get("installed_files_modified") is not False
            or record.get("global_safetensors_alias_modified") is not False
            or record.get("runtime_sources") != identity["source_sha256"]
            or record.get("model_directory") != loader_model_directory(identity, contract)
        ):
            raise ValueError("Loader receipt lacks matching completed/restored scoped execution")
        versions = record.get("runtime_versions", {})
        if set(versions) != set(identity["versions"]) or any(
            versions[name]
            not in (
                {version, identity["torch_cpu_wheel_version"]} if name == "torch" else {version}
            )
            for name, version in identity["versions"].items()
        ):
            raise ValueError("Loader receipt runtime differs from its pinned policy")
        opens = record.get("opens")
        if (
            not isinstance(opens, list)
            or not 1 <= len(opens) <= 64
            or any(
                not isinstance(item, dict)
                or item != {"file": item.get("file"), "device": "cpu", "backend": "pread"}
                or item.get("file") not in identity["shards"]
                for item in opens
            )
            or {item["file"] for item in opens} != set(identity["shards"])
        ):
            raise ValueError("Loader receipt does not show bounded CPU pread of both pinned shards")


def verify_parent(plan):
    """All API metadata is checked before downloading or restoring any pickle."""
    checked_loader(plan)  # Before API downloads, extraction, or trusted pickle restoration.
    repo = ci.api_json("")
    run = ci.api_json(f"actions/runs/{plan['parent_run_id']}")
    if (
        repo["private"]
        or repo["full_name"] != ci.REPOSITORY
        or (
            run["id"] != plan["parent_run_id"]
            or run["head_sha"] != plan["parent_commit"]
            or run["path"] != FIRST4_WORKFLOW
            or run["event"] != "workflow_dispatch"
            or run["run_attempt"] != 1
            or run["status"] != "completed"
            or run["conclusion"] != "success"
            or run["head_repository"]["full_name"] != ci.REPOSITORY
        )
    ):
        raise ValueError("Parent must be the reviewed successful public first-four workflow/commit")
    for path, expected in plan["first4_files"].items():
        if ci.digest(ci.regular(Path(path))) != expected:
            raise ValueError("Continuation checkout changed a frozen first-four helper/descriptor")
        remote = ci.api_json(f"contents/{path}?ref={plan['parent_commit']}")
        if (
            remote["encoding"] != "base64"
            or hashlib.sha256(base64.b64decode(remote["content"])).hexdigest() != expected
        ):
            raise ValueError("Parent commit differs from the reviewed first-four source identity")
    return run


def require_single_use(plan):
    title = f"aircraft-cpu-continue parent-{plan['parent_run_id']}"
    current = int(os.environ["GITHUB_RUN_ID"])
    for page in range(1, 101):
        records = ci.api_json(
            f"actions/workflows/aircraft-cpu-continue.yml/runs?event=workflow_dispatch&per_page=100&page={page}"
        )
        if any(
            r["id"] != current and r["display_title"] == title for r in records["workflow_runs"]
        ):
            raise ValueError(
                "A continuation invocation already consumed this parent; no retry/reset"
            )
        if len(records["workflow_runs"]) < 100:
            return
    raise ValueError("Cannot establish unique parent ownership within the bounded history audit")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def artifact_download(item, destination, *, run_id, commit, name):
    metadata = ci.api_json(f"actions/artifacts/{item['id']}")
    if (
        metadata["expired"]
        or metadata["name"] != name
        or metadata["workflow_run"]["id"] != run_id
        or metadata["workflow_run"]["head_sha"] != commit
        or metadata["digest"] != "sha256:" + item["sha256"]
        or metadata["size_in_bytes"] != item["bytes"]
    ):
        raise ValueError("Artifact ID/name/run/commit/digest/size differs from its frozen pin")
    url = f"https://api.github.com/repos/{ci.REPOSITORY}/actions/artifacts/{item['id']}/zip"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        urllib.request.build_opener(NoRedirect()).open(request, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code != 302:
            raise ValueError(f"Artifact redirect failed with HTTP {exc.code}") from None
        location = exc.headers["Location"]
    else:
        raise ValueError("Expected the GitHub artifact redirect")
    parsed = urllib.parse.urlsplit(location)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("Unsafe artifact redirect")
    # The signed download receives no GitHub credential; URL is never logged.
    ci.download(location, destination, size=item["bytes"], sha256=item["sha256"])
    return metadata


def extract_zip(path, destination):
    """Bound every entry before extraction; links/devices/path aliases are rejected."""
    with zipfile.ZipFile(ci.regular(path)) as archive:
        entries = archive.infolist()
        if len(entries) > 20000 or sum(e.file_size for e in entries) > ci.LIMITS["output_bytes"]:
            raise ValueError("Artifact exceeds its original output budget")
        names = set()
        for entry in entries:
            name = entry.filename[:-1] if entry.is_dir() else entry.filename
            ci.safe_name(name)
            mode = entry.external_attr >> 16
            if (
                name in names
                or entry.flag_bits & 1
                or stat.S_IFMT(mode) not in (0, stat.S_IFDIR if entry.is_dir() else stat.S_IFREG)
            ):
                raise ValueError("Duplicate, encrypted or nonregular ZIP entry")
            names.add(name)
        for entry in entries:
            if entry.is_dir():
                continue
            target = destination / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as out:
                shutil.copyfileobj(source, out, 1024**2)
            ci.regular(target)


def inventory(root, *, exclude=()):
    return {
        str(p.relative_to(root)): ci.digest(ci.regular(p))
        for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in exclude
    }


def verify_first4(root, arm, plan):
    loader, contract = checked_loader(plan)
    state = ci.read(root / "ci-state.json")
    if any(
        state.get(k) != v
        for k, v in {
            "status": "completed",
            "group": "first4",
            "arm": arm,
            "commit": plan["parent_commit"],
            "run_id": str(plan["parent_run_id"]),
            "descriptor_sha256": plan["first4_files"][str(DESCRIPTOR)],
            "original_plan_sha256": ci.PLAN_SHA,
            "loader": loader,
        }.items()
    ) or state["file_sha256"] != inventory(root, exclude=("ci-state.json",)):
        raise ValueError("First-four artifact attribution or complete file inventory differs")
    if set(ci.completed_phases(root)) != {f"{arm}-first4"}:
        raise ValueError("Only the single original first-four phase can be restored")
    verify_loader_phases(root, loader, contract)
    result = ci.read(root / f"phases/{arm}-first4/result.json")
    if (
        result.get("committed_steps") != 4
        or result["hardware"]["passed"] is not True
        or not all(result["hardware"]["gates"].values())
    ):
        raise ValueError("Both hardware gates must pass before any continuation")
    for prefix in (f"work/{arm}", f"endpoints/{arm}-first4"):
        receipt = ci.read(root / prefix / "endpoint_receipt.json")
        if (
            receipt["global_step"] != 4
            or receipt["original_budgets"] != {"max_steps": 64, "max_active_seconds": 7200}
            or receipt["checkpoint"]["path"] != "checkpoint-4"
        ):
            raise ValueError("Restore the original checkpoint-4 with its original total budgets")
        for name in (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "active_time.json",
            "exposure.json",
        ):
            ci.regular(root / prefix / "checkpoint-4" / name)
    # Native read_endpoint_receipt later checks all optimizer/RNG/reference hashes
    # again inside the original virtual paths, before model allocation/resume.
    return state


def merge_first4(root, sources, plan):
    root.mkdir(parents=True, exist_ok=False)
    for arm, source in sources.items():
        verify_first4(source, arm, plan)
        for prefix in ("work", "endpoints", "phases"):
            source_dir = source / prefix
            if source_dir.exists():
                for child in source_dir.iterdir():
                    expected = arm if prefix == "work" else f"{arm}-first4"
                    if child.name != expected:
                        raise ValueError("Unexpected first-four output subtree")
                    shutil.copytree(child, root / prefix / child.name)
        provenance = root / "parents" / arm
        provenance.mkdir(parents=True)
        shutil.copyfile(source / "ci-state.json", provenance / "ci-state.json")
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > ci.LIMITS["output_bytes"]:
        raise ValueError("Merged evidence exceeds original output cap")


def remaining_allocation(output):
    phases = ci.completed_phases(output)
    expected = {f"{arm}-first4" for arm in ARMS} | {f"warm-{stage}" for stage in ASSAYS}
    if set(phases) != expected:
        raise ValueError("Warm handoff requires both first-four gates and all three fresh assays")
    values = [r["elapsed_seconds"] for r in phases.values()]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid prior physical phase accounting")
    used = sum(values)
    each = math.floor((ci.LIMITS["all_phase_wall_seconds"] - used) / 2)
    if each <= 0:
        raise ValueError("Original aggregate physical phase budget is exhausted")
    return {
        "original_total_seconds": ci.LIMITS["all_phase_wall_seconds"],
        "prior_shared_seconds": used,
        "reserved_seconds": {arm: each for arm in ARMS},
    }


def write_handoff(output, plan, group, *, allocation=None):
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "group": group,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "commit": os.environ["GITHUB_SHA"],
        "parent_envelope": plan,
        "parent_envelope_sha256": object_sha(plan),
        "allocation": allocation,
        "file_sha256": inventory(output, exclude=("handoff.json",)),
        "test_consumed": False,
        "primary_requirement_satisfied": False,
    }
    ci.write(output / "handoff.json", receipt)
    return receipt


def verify_handoff(output, plan, group):
    loader, contract = checked_loader(plan)
    record = ci.read(output / "handoff.json")
    if (
        record["status"] != "completed"
        or record["group"] != group
        or record["parent_envelope"] != plan
        or record["parent_envelope_sha256"] != object_sha(plan)
        or record["run_id"] != os.environ["GITHUB_RUN_ID"]
        or record["commit"] != os.environ["GITHUB_SHA"]
        or record["file_sha256"] != inventory(output, exclude=("handoff.json",))
    ):
        raise ValueError("Handoff does not belong to this exact reviewed continuation invocation")
    verify_loader_phases(output, loader, contract)
    if group == "warm" and record["allocation"] != remaining_allocation(output):
        raise ValueError("Original aggregate budget allocation changed")
    return record


def _owned_subtrees(arm):
    return (
        f"work/{arm}",
        f"runtime-preflight/{arm}",
        f"endpoints/{arm}-full",
        f"assays/{arm}",
        f"phases/{arm}-full",
        *(f"phases/{arm}-{stage}" for stage in ASSAYS),
    )


def merge_completed(output, sources, plan):
    """Merge only each arm's owned trajectory; shared evidence stays byte-identical."""
    records = {name: verify_handoff(path, plan, name) for name, path in sources.items()}
    if (
        set(records) != set(ARMS)
        or records["grpo"]["allocation"] != records["dr_grpo"]["allocation"]
    ):
        raise ValueError("Both completed arms must share the original warm budget reservation")
    owned = tuple(prefix + "/" for arm in ARMS for prefix in _owned_subtrees(arm))
    shared = {
        name: {
            k: v
            for k, v in record["file_sha256"].items()
            if not k.startswith(owned) and k != "job-status.json"
        }
        for name, record in records.items()
    }
    if shared["grpo"] != shared["dr_grpo"]:
        raise ValueError("Shared first-four/warm evidence differs between parallel arms")
    shutil.copytree(sources["grpo"], output)
    for name, source in sources.items():
        for relative in _owned_subtrees(name):
            target = output / relative
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source / relative, target)
        ci.write(output / "continuations" / (name + ".json"), records[name])
    phases = ci.completed_phases(output)
    expected = {f"{arm}-{stage}" for arm in ARMS for stage in ("first4", "full", *ASSAYS)} | {
        f"warm-{a}" for a in ASSAYS
    }
    if (
        set(phases) != expected
        or sum(row["elapsed_seconds"] for row in phases.values())
        > ci.LIMITS["all_phase_wall_seconds"]
    ):
        raise ValueError("Merged physical phases exceed or differ from the original finite study")
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > ci.LIMITS["output_bytes"]:
        raise ValueError("Combined output exceeds the original study cap")


def container_stage(group, plan_path):
    if sys.platform != "linux" or os.environ.get("ACCELERATE_USE_CPU") != "true":
        raise ValueError("Only offline CPU container execution is permitted")
    plan = envelope(ci.read(plan_path))
    loader, contract = checked_loader(plan)
    output = ci.ORIGINAL / ci.OUTPUT_REL
    if group in ARMS:
        verify_handoff(output, plan, "warm")
    preflight = output / "runtime-preflight" / group
    preflight.mkdir(parents=True, exist_ok=False)
    ci.filesystem_preflight(preflight)
    runner, _ = ci.check_inputs()
    runner._require_hardware_pair()
    verify_loader_phases(output, loader, contract)
    if group == "warm":
        if set(ci.completed_phases(output)) != {f"{a}-first4" for a in ARMS}:
            raise ValueError("Warm assay is single-use; preserve all prior partial work")
        for suffix in ASSAYS:
            ci.supervise_phase(f"warm-{suffix}", ci.LIMITS)
        runner._require_warm_assay()
        allocation = remaining_allocation(output)
    elif group in ARMS:
        prior = ci.read(output / "handoff.json")  # Verified before this invocation's write proof.
        runner._require_warm_assay()
        allocation = prior["allocation"]
        # Each parallel arm gets a disjoint reservation from one original budget.
        limits = dict(
            ci.LIMITS,
            all_phase_wall_seconds=allocation["prior_shared_seconds"]
            + allocation["reserved_seconds"][group],
        )
        for phase in (f"{group}-full", *(f"{group}-{a}" for a in ASSAYS)):
            ci.supervise_phase(phase, limits)
    elif group == "compare":
        import compare

        if Path(compare.__file__).resolve() != ci.ORIGINAL / ci.PLAN_REL / "compare.py":
            raise ValueError("Unexpected frozen comparator")
        compare.main(ci.PLAN_SHA)
        allocation = ci.read(output / "continuations/grpo.json")["allocation"]
    else:
        raise ValueError("No first-four or arbitrary worker phase is exposed here")
    verify_loader_phases(output, loader, contract)
    write_handoff(output, plan, group, allocation=allocation)


def setup_runtime(root, contract):
    """Reuse exact first-four data/build contract; never use live package sources."""
    inputs = root / "inputs"
    for key, destination, expected in (
        ("bundle", inputs, contract["files"]),
        ("warm_bundle", root / "warm-release", contract["warm_bundle"]["files"]),
    ):
        item = contract[key]
        archive = root / item["file"]
        ci.download(item["url"], archive, size=item["bytes"], sha256=item["sha256"])
        ci.extract_tar(archive, destination, expected, limit=32 * 1024**2)
        archive.unlink()
    (root / "warm-release/aircraft-common-warm").rename(inputs / "runs/aircraft-warmstart-v1")
    (inputs / ci.OUTPUT_REL).mkdir(parents=True)
    context = root / "context"
    context.mkdir()
    for name, expected in contract["build_files"].items():
        if ci.digest(ci.regular(name)) != expected:
            raise ValueError("Frozen build source differs")
        shutil.copyfile(name, context / name)
    shutil.copytree(inputs / ci.PLAN_REL / "source", context / "src")
    image = "vision-rl:aircraft-cpu7946-continuation"
    subprocess.run(
        ["docker", "build", "-f", str(context / "Dockerfile.vlm"), "-t", image, str(context)],
        check=True,
        timeout=1200,
    )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", image], text=True
    ).strip()
    required = (
        sum(row["bytes"] for row in contract["downloads"]) + ci.LIMITS["output_bytes"] + ci.GIB
    )
    if shutil.disk_usage(root).free < required:
        raise ValueError("Insufficient disk for the unchanged base/data and output reserve")
    for item in contract["downloads"]:
        ci.safe_name(item["path"])
        if not item["url"].startswith("https://huggingface.co/"):
            raise ValueError("Only the original pinned public downloads are permitted")
        ci.download(item["url"], inputs / item["path"], size=item["bytes"], sha256=item["sha256"])
    return inputs, image_id


def docker_command(inputs, output, image_id, group, plan_path):
    command = ci.docker_command(inputs, output, image_id, "grpo")
    index = command.index(image_id)
    command[index:index] = [
        "-v",
        f"{Path(__file__).resolve()}:/ci/continue.py:ro",
        "-v",
        f"{plan_path.resolve()}:/ci/continuation-plan.json:ro",
        "-v",
        f"{DESCRIPTOR.resolve()}:/ci/experiment-plan.json:ro",
        "-e",
        "GITHUB_RUN_ID=" + os.environ["GITHUB_RUN_ID"],
        "-e",
        "GITHUB_SHA=" + os.environ["GITHUB_SHA"],
    ]
    command[command.index(image_id) + 1 :] = [
        "python",
        "/ci/continue.py",
        "container-stage",
        "--group",
        group,
        "--plan",
        "/ci/continuation-plan.json",
    ]
    return command


def host_run(root, plan, group, artifacts=None):
    if (
        sys.platform != "linux"
        or os.environ.get("GITHUB_REPOSITORY") != ci.REPOSITORY
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
        or os.environ.get("PARENT_RUN_ID") != str(plan["parent_run_id"])
    ):
        raise ValueError("Only one manual public continuation attempt is permitted")
    started = time.monotonic()
    verify_parent(plan)
    require_single_use(plan)
    contract = ci.read(DESCRIPTOR)
    if contract["limits"] != ci.LIMITS or contract["original_plan_sha256"] != ci.PLAN_SHA:
        raise ValueError("Original prospective CPU experiment contract changed")
    root.mkdir(parents=True, exist_ok=False)
    output = root / "output"
    status = {"status": "started", "group": group, "plan": plan, "model_calls_by_host": 0}
    ci.write(root / "status.json", status)
    plan_path = root / "continuation-plan.json"
    ci.write(plan_path, plan)
    try:
        if group == "warm":
            sources = {}
            for arm in ARMS:
                path = root / (arm + ".zip")
                artifact_download(
                    plan["artifacts"][arm],
                    path,
                    run_id=plan["parent_run_id"],
                    commit=plan["parent_commit"],
                    name=f"aircraft-cpu-first4-{arm}",
                )
                sources[arm] = root / "parents" / arm
                extract_zip(path, sources[arm])
                path.unlink()
            merge_first4(output, sources, plan)
            shutil.rmtree(root / "parents")  # Exact originals remain in first-four artifacts.
        elif group in ARMS:
            path = root / "warm.zip"
            artifact_download(
                artifacts["warm"],
                path,
                run_id=int(os.environ["GITHUB_RUN_ID"]),
                commit=os.environ["GITHUB_SHA"],
                name="aircraft-cpu-continuation-warm",
            )
            extract_zip(path, output)
            path.unlink()
            verify_handoff(output, plan, "warm")
        elif group == "compare":
            sources = {}
            for arm in ARMS:
                path = root / (arm + ".zip")
                artifact_download(
                    artifacts[arm],
                    path,
                    run_id=int(os.environ["GITHUB_RUN_ID"]),
                    commit=os.environ["GITHUB_SHA"],
                    name=f"aircraft-cpu-continuation-{arm}",
                )
                sources[arm] = root / "completed" / arm
                extract_zip(path, sources[arm])
                path.unlink()
            merge_completed(output, sources, plan)
            shutil.rmtree(root / "completed")
        else:
            raise ValueError("Unknown continuation group")
        inputs, image_id = setup_runtime(root, contract)
        memory = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if len(line.split()) == 3
        }
        if memory["MemAvailable"] < ci.LIMITS["prelaunch_available_bytes"]:
            raise ValueError("CPU prelaunch memory guard; no model call")
        status.update(image_id=image_id, prelaunch_memory=memory)
        command = docker_command(inputs, output, image_id, group, plan_path)
        remaining = JOB_SECONDS - (time.monotonic() - started) - 120
        if remaining <= 0:
            raise ValueError("Original job wall budget exhausted in preparation")
        subprocess.run(command, check=True, timeout=remaining)
        verify_handoff(output, plan, group)
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            subprocess.run(
                ["docker", "stop", "--time", "20", "aircraft-cpu-study"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            status["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            status["status"] = "failed"
        status["elapsed_seconds"] = time.monotonic() - started
        ci.write(root / "status.json", status)
        ci.write(output / "job-status.json", status)
        if status["status"] == "completed":
            allocation = ci.read(output / "handoff.json")["allocation"]
            write_handoff(output, plan, group, allocation=allocation)


def dynamic_artifact(name):
    number = int(os.environ[name.upper() + "_ARTIFACT_ID"])
    digest = hex_sha(os.environ[name.upper() + "_ARTIFACT_SHA"])
    metadata = ci.api_json(f"actions/artifacts/{number}")
    return {"id": number, "sha256": digest, "bytes": metadata["size_in_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "container-stage"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--root", type=Path, default=Path("runs/aircraft-ci-continuation"))
    parser.add_argument("--group", choices=("warm", *ARMS, "compare"))
    args = parser.parse_args()
    raw = args.plan.read_text() if args.plan else os.environ["CONTINUATION_PLAN"]
    if len(raw) > 16384:
        raise ValueError("Continuation envelope is oversized")
    plan = envelope(json.loads(raw))
    if args.command == "validate":
        print(
            json.dumps(
                {"valid_schema": True, "network_or_model_calls": 0, "plan_sha256": object_sha(plan)}
            )
        )
    elif args.command == "container-stage":
        container_stage(args.group, args.plan)
    else:
        artifacts = None
        if args.group in ARMS:
            artifacts = {"warm": dynamic_artifact("warm")}
        elif args.group == "compare":
            artifacts = {name: dynamic_artifact(name) for name in ARMS}
        host_run(args.root, plan, args.group, artifacts)


if __name__ == "__main__":
    main()
