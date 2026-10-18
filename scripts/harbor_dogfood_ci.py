"""GitHub-hosted reproduction of the passing pretrained Docker + Harbor dogfood (plan 62fe9ff1).

Scope: integration only, native FGVC-Aircraft TRAIN fixtures, never held-out, never a benchmark.
Importing performs no work. Only `run` downloads, builds, or calls Docker/Harbor; `criteria` and
`collect-evidence` are read-only over an existing run tree and need nothing but the stdlib.

Reuse, not reimplementation: the plan directory is extracted byte for byte from the committed
`reports/2026-10-16-harbor-pretrained-dogfood/plan-source.tar.gz` (checked against its
`plan-SHA256SUMS`). Pins, build context, bundle checks, the Harbor command, the host output guard,
the probe/complete assertions and C1-C4 are that plan's own `local_run.py` functions. Downloads and
archive extraction are `scripts/aircraft_ci_experiment.py` functions. The only new logic is the
GitHub launch gate (in place of `require_local_launch`, which requires the Colima Mac), assembling
the pinned input tree from public pinned URLs, and an allowlisted evidence copy.

Blind spots, stated: the GitHub gate trusts the GITHUB_* environment the runner sets; the repository
visibility check is one API call. The evidence allowlist is by file suffix and size, so a JSON file
that embeds image bytes would pass it (none of the plan's receipts do).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REPOSITORY = "RyanLin5967/RyanLinEnv"
REPORT = Path("reports/2026-10-16-harbor-pretrained-dogfood")
PLAN_TARBALL_SHA = "5526f60ac9468d9c1ac0c07e6c021d0e63ffbb6f2ae63368185a080ef8ae391e"
PLAN_SHA = "62fe9ff1d71ba6c50a183ed1ffecdc2318be2fe5b31c521b3079e6aedc494901"
DESCRIPTOR = Path("docs/aircraft-cpu-experiment-plan.json")
DEFAULT_ROOT = Path("runs/harbor-dogfood-ci")
RESULT_NAME = "dogfood-ci-result.json"
FROZEN_REL = Path("runs/aircraft-objective-ablation-v1-plan/source")
WARM_REL = Path("runs/aircraft-warmstart-v1")
IMAGE_DIR_REL = Path("data/fgvc-aircraft-2013b/development/images/train")
GIB = 1024**3
# Disk needed beyond the downloads: runtime image + two thin task images + build cache (8 GiB,
# the CPU torch/transformers image measured about 3 GiB locally; unverified on x86), Harbor's
# 2 GiB host output cap, and the 1 GiB floor run_harbor enforces.
DISK_HEADROOM_BYTES = 11 * GIB
# README step 3 preflight floors, checked on the runner before each Harbor run.
PRELAUNCH_AVAILABLE_BYTES = 14 * GIB
PRELAUNCH_ROOT_FREE_BYTES = 4 * GIB
SECRET_ENVIRONMENT = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
)
EVIDENCE_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".toml", ".md"}
# Base-model processor/tokenizer files that trainers copy next to adapters: base-model files,
# not evidence, so never shipped even outside checkpoint-* directories.
EVIDENCE_EXCLUDED_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
EVIDENCE_FILE_CAP = 16 * 1024**2
EVIDENCE_TOTAL_CAP = 256 * 1024**2


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The validated download/extract/digest/write helpers of the first-four CI reproduction.
# Importing that module performs no work.
CI = _load("aircraft_ci_experiment", HERE / "aircraft_ci_experiment.py")
sha = CI.digest
write_json = CI.write


class Terminated(Exception):
    """SIGTERM/SIGHUP from the step timeout: an Exception, so every finally and the result run."""


def _terminate(signum, frame):
    raise Terminated(f"Driver signal {signum} (outer step timeout)")


# ---- the plan directory, byte for byte from the committed report ---------------------------


def extract_plan(repository, destination):
    """Extract plan-source.tar.gz into destination/plan; refuse any byte that differs."""
    report = Path(repository) / REPORT
    tarball = report / "plan-source.tar.gz"
    if sha(tarball) != PLAN_TARBALL_SHA:
        raise ValueError("Committed plan-source.tar.gz differs from the reviewed tarball")
    expected = {}
    for line in (report / "plan-SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    destination = Path(destination)
    if (destination / "plan").exists():
        raise ValueError("Plan directory must be new")
    seen = set()
    with tarfile.open(tarball, "r:gz") as archive:
        for member in archive:
            if member.isdir() and member.name.rstrip("/") == "plan":
                continue
            path = PurePosixPath(member.name)
            if member.isfile() and path.name.startswith("._") and member.size <= 4096:
                continue  # macOS tar AppleDouble metadata; never extracted or used.
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name not in expected
                or member.name in seen
                or member.size > 4 * 1024**2
            ):
                raise ValueError(f"Unexpected plan archive member: {member.name}")
            seen.add(member.name)
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as out:
                shutil.copyfileobj(source, out)
            if sha(target) != expected[member.name]:
                raise ValueError(f"Extracted plan file differs: {member.name}")
    if seen != set(expected):
        raise ValueError("Plan archive is missing pinned files")
    plan_dir = destination / "plan"
    if sha(plan_dir / "plan.json") != PLAN_SHA:
        raise ValueError("Plan is not the passing v4 plan 62fe9ff1")
    return plan_dir


def plan_modules(plan_dir):
    """The plan's own local driver; importing it runs nothing."""
    return _load("dogfood_local_run", Path(plan_dir) / "local_run.py")


# ---- read-only criteria (the gate) ----------------------------------------------------------


def criteria(root, repository=REPO_ROOT):
    """Re-derive C1-C4 and the probe/complete assertions from an existing run tree.

    The plan code is extracted afresh from the committed tarball, never taken from the run tree
    being judged. This mirrors local_run.py's `criteria` subcommand, deliberately differing in
    two ways: a missing packaged contract is a recorded failure (so an empty tree still yields a
    full failing report), and failure reasons are kept for the summary.
    """
    with tempfile.TemporaryDirectory(prefix="dogfood-criteria-") as temporary:
        local = plan_modules(extract_plan(repository, temporary))
        return _criteria(local, Path(root).resolve())


def _criteria(local, root):
    plan = local.read_plan()
    jobs = root / "output" / "jobs"
    contract_path = root / "prepared/bundle/task/environment/contract.json"
    error = None
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, ValueError) as exc:
        contract, error = {}, f"{type(exc).__name__}: packaged contract unreadable: {exc}"
    if error is None:
        try:
            local.assert_probe(jobs / local.PROBE_JOB)
            local.assert_complete(jobs / local.MAIN_JOB, plan["training"]["max_steps"])
        except Exception as exc:  # Recorded; criteria still report what exists.
            error = f"{type(exc).__name__}: {exc}"
    return local.evaluate(root, plan, contract, plan_sha=PLAN_SHA, driver_error=error)


def summary(result):
    return {
        "pass": result["pass"],
        "driver_error": result["driver_error"],
        "criteria": {k: v["pass"] for k, v in result["criteria"].items()},
        "failures": {k: v["reason"] for k, v in result["criteria"].items() if not v["pass"]},
    }


# ---- allowlisted evidence copy ---------------------------------------------------------------


def evidence_candidates(root):
    root = Path(root)
    output = root / "output"
    if output.is_dir():
        for directory, names, files in os.walk(output):  # os.walk never follows symlinks.
            names[:] = sorted(n for n in names if not n.startswith("checkpoint-"))
            for name in sorted(files):
                yield Path(directory) / name
    bundle = root / "prepared/bundle"
    for name in (
        "preparation.json",
        "train.jsonl",
        "checker_train.jsonl",
        "task/task.toml",
        "task/bundle.json",
        "task/environment/contract.json",
        "task/environment/runtime-pins.json",
        "task/environment/docker-compose.yaml",
    ):
        yield bundle / name


def collect_evidence(root, destination):
    """Copy only small text receipts; never images, weights, checkpoints or symlinks."""
    root, destination = Path(root).resolve(), Path(destination)
    if destination.exists():
        raise ValueError("Evidence directory must be new")
    destination.mkdir(parents=True)
    copied, skipped, total = {}, [], 0
    for path in evidence_candidates(root):
        relative = path.relative_to(root)
        allowed = path.suffix in EVIDENCE_SUFFIXES or path.name == "docker-compose.yaml"
        if (
            not allowed
            or path.name in EVIDENCE_EXCLUDED_NAMES
            or path.is_symlink()
            or not path.is_file()
        ):
            if path.exists() or path.is_symlink():
                skipped.append(str(relative))
            continue
        size = path.lstat().st_size
        if size > EVIDENCE_FILE_CAP or total + size > EVIDENCE_TOTAL_CAP:
            skipped.append(str(relative))
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target, follow_symlinks=False)
        copied[str(relative)] = sha(target)
        total += size
    write_json(
        destination / "MANIFEST.json",
        {
            "copied_sha256": copied,
            "skipped_not_allowlisted": sorted(skipped),
            "allowlist_suffixes": sorted(EVIDENCE_SUFFIXES),
            "base_weights_or_images_included": False,
        },
    )
    return copied


# ---- the GitHub launch gate (replaces the Colima-only require_local_launch) ------------------


def require_ci_launch(plan, *, environ=None, platform=None):
    """Refuse unless this is a first manual attempt of the public repo on a Linux runner."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if (
        plan.get("local_launch_reviewed") is not True
        or plan.get("cgroup_policy_reviewed") is not True
    ):
        raise ValueError("Plan unreviewed; no launch")
    if plan["training"]["max_steps"] != 1:
        raise ValueError("The dogfood plan is defined only for max_steps == 1")
    if platform != "linux":
        raise ValueError("The CI driver runs only on a Linux GitHub-hosted runner")
    if (
        environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or environ.get("GITHUB_RUN_ATTEMPT") != "1"
        or environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
    ):
        raise ValueError("Only a first manual attempt on a GitHub-hosted runner is allowed")
    if environ.get("DOCKER_CONTEXT") or environ.get("DOCKER_HOST") not in (
        None,
        "",
        "unix:///var/run/docker.sock",
    ):
        raise ValueError("Docker must be the runner's local daemon")


def meminfo():
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if len(line.split()) == 3
    }


def require_memory(plan, path):
    """README step 3 floors (MemAvailable >= 14 GiB, root free >= 4 GiB) before each Harbor run."""
    available = meminfo()["MemAvailable"]
    floor = max(PRELAUNCH_AVAILABLE_BYTES, plan["resource"]["prelaunch_available_bytes"])
    if available < floor:
        raise RuntimeError(f"Prelaunch host memory guard: MemAvailable {available} B < {floor} B")
    free = {"root": shutil.disk_usage("/").free, "run": shutil.disk_usage(path).free}
    if min(free.values()) < PRELAUNCH_ROOT_FREE_BYTES:
        raise RuntimeError(f"Prelaunch disk guard: free {free} below 4 GiB")
    return {"mem_available_bytes": available, "disk_free_bytes": free}


def kernel_preflight(image, plan):
    """README step 3 with the runtime image: the kernel must enforce 13 GiB and zero swap."""
    mib = plan["resource"]["prospective_override_memory_mib"]
    out = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            f"{mib}m",
            "--memory-swap",
            f"{mib}m",
            "--entrypoint",
            "sh",
            image,
            "-c",
            "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max "
            "/sys/fs/cgroup/memory.swap.current",
        ],
        text=True,
        timeout=120,
    ).split()
    expected = [str(plan["resource"]["container_memory_bytes"]), "0", "0"]
    if out != expected:
        raise RuntimeError(f"Kernel cgroup preflight differs: {out} != {expected}")
    return out


def git_show(repository, commit, name):
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{commit}:{name}"], timeout=30
    )


def assemble_inputs(repository, inputs, plan, ci, read_at_commit=git_show):
    """Lay out the pinned tree the plan expects, from public pinned URLs and the source commit."""
    descriptor = json.loads((Path(repository) / DESCRIPTOR).read_text())
    if sha(Path(repository) / DESCRIPTOR) != plan["first4_descriptor_sha256"]:
        raise ValueError("First-four descriptor differs from the plan pin")
    bundle = plan["public_source_bundle"]
    if {k: descriptor["bundle"][k] for k in bundle} != bundle:
        raise ValueError("Public source bundle differs between plan and descriptor")
    inputs.mkdir(parents=True)
    staging = inputs.parent / "downloads"
    staging.mkdir()
    archive = staging / bundle["file"]
    ci.download(bundle["url"], archive, size=bundle["bytes"], sha256=bundle["sha256"])
    ci.extract_tar(archive, inputs, descriptor["files"], limit=32 * 1024**2)
    archive.unlink()
    warm = plan["public_warm_bundle"]
    warm_path = staging / warm["file"]
    ci.download(warm["url"], warm_path, size=warm["bytes"], sha256=warm["sha256"])
    ci.extract_tar(warm_path, staging / "warm", warm["files"], limit=32 * 1024**2)
    (staging / "warm/aircraft-common-warm").rename(inputs / WARM_REL)
    warm_path.unlink()
    adapter = inputs / WARM_REL / "final_adapter/adapter_model.safetensors"
    if sha(adapter) != plan["model_identity"]["initial_weights_sha256"]:
        raise ValueError("Warm adapter is not f05840be")
    # Host/build files exactly as of the plan's source commit (pyproject.toml changed later).
    for name, expected in {**plan["host_source_sha256"], **plan["build_files"]}.items():
        content = read_at_commit(repository, plan["source_commit"], name)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Pinned host file differs at the source commit: {name}")
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:  # Never overwrite a bundle member.
            stream.write(content)
    rows = {row["path"]: row for row in descriptor["downloads"]}
    wanted = []
    for item in plan["provenance"]:
        record = item["original_record"]
        if record["split"] != "train":
            raise ValueError("Every fixture must be native TRAIN")
        relative = str(IMAGE_DIR_REL / f"{record['id']}.jpg")
        row = rows.get(relative)
        if row is None:
            raise ValueError(f"No pinned mirror download for fixture {record['id']}")
        if row["sha256"] != item["encoded_sha256"]:
            raise ValueError("Pinned mirror JPEG differs from the plan provenance")
        wanted.append(row)
    model_prefix = "data/models/Qwen2-VL-2B-Instruct-895c3a49/"
    model_rows = [row for path, row in rows.items() if path.startswith(model_prefix)]
    identity = plan["model_identity"]
    if (
        sha(inputs / model_prefix / "model-source.json") != identity["manifest_sha256"]
        or sha(inputs / model_prefix / ".pinned-download.json") != identity["marker_sha256"]
    ):
        raise ValueError("Model manifest/marker differ from the plan's model identity")
    source = json.loads((inputs / model_prefix / "model-source.json").read_text())
    if source["revision"] != plan["model_identity"]["revision"] or {
        model_prefix + f["path"] for f in source["files"]
    } != {row["path"] for row in model_rows}:
        raise ValueError("Model download list differs from the pinned model manifest")
    needed = sum(row["bytes"] for row in wanted + model_rows) + DISK_HEADROOM_BYTES
    if shutil.disk_usage(inputs).free < needed:
        raise RuntimeError(f"Insufficient disk: need {needed} B free before downloads")
    for row in wanted + model_rows:
        ci.safe_name(row["path"])
        if not row["url"].startswith("https://huggingface.co/"):
            raise ValueError("Only the pinned public Hugging Face mirror is allowed")
        ci.download(row["url"], inputs / row["path"], size=row["bytes"], sha256=row["sha256"])
    shutil.rmtree(staging)  # Only the emptied warm extraction directory remains.
    return {"images": len(wanted), "model_files": len(model_rows)}


def run(repository, root):
    repository = Path(repository).resolve()
    root = Path(root)
    if root.exists():
        raise ValueError("Never reuse a CI directory; no retries")
    root.mkdir(parents=True)
    root = root.resolve()
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _terminate)
    plan_dir = extract_plan(repository, root)
    local = plan_modules(plan_dir)
    plan = local.read_plan()
    require_ci_launch(plan)  # Before any download, build or Docker call.
    local.require_wall_nesting(plan)
    ci = CI
    if ci.api_json("")["private"]:
        raise ValueError("Only the public repository on free standard runners is allowed")
    for name in SECRET_ENVIRONMENT:  # Needed only for that one call; never reaches Harbor.
        os.environ.pop(name, None)
    if sha(repository / "uv.lock") != plan["host_dependencies"]["uv_lock_sha256"]:
        raise ValueError("The installed Harbor host environment's uv.lock differs from the plan")
    harbor = repository / ".venv/bin/harbor"
    version = subprocess.check_output(
        [
            str(repository / ".venv/bin/python"),
            "-I",
            "-c",
            "import importlib.metadata as m; print(m.version('harbor'))",
        ],
        text=True,
        timeout=60,
    ).strip()
    if version != plan["host_dependencies"]["harbor"]:
        raise ValueError(
            f"Harbor {version} is not the pinned {plan['host_dependencies']['harbor']}"
        )
    output = root / "output"
    output.mkdir()
    state = {
        "schema_version": 1,
        "status": "running",
        "scope": "GitHub-hosted reproduction of " + plan["scope"],
        "reproduces_plan_sha256": PLAN_SHA,
        "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "harbor_version": version,
        "cpu_hardware": json.loads(
            subprocess.check_output(["lscpu", "--json"], text=True, timeout=30)
        ),
        "initial_memory": meminfo(),
        "initial_disk_free_bytes": shutil.disk_usage(root).free,
        "native_validation_used": False,
        "native_test_used": False,
    }
    write_json(output / "ci-state.json", state)
    error = None
    contract = {}
    inputs = root / "inputs"
    try:
        state["inputs"] = assemble_inputs(repository, inputs, plan, ci)
        local.verify_pins(plan, repository=inputs)
        context = local.build_context(root / "context", repository=inputs)
        started = time.monotonic()
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(context / "Dockerfile.vlm"),
                "-t",
                local.IMAGE,
                str(context),
            ],
            check=True,
            timeout=1800,
        )
        state["build_seconds"] = time.monotonic() - started
        subprocess.run(
            ["docker", "run", "--rm", "--network", "none", local.IMAGE], check=True, timeout=300
        )
        state["kernel_preflight"] = kernel_preflight(local.IMAGE, plan)
        prepared = root / "prepared"
        prepared.mkdir()
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "1g",
                "--memory-swap",
                "1g",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{inputs}:{inputs}:ro",
                "-v",
                f"{plan_dir}:/prepared-source:ro",
                "-v",
                f"{prepared}:{prepared}:rw",
                local.IMAGE,
                "python",
                "-I",
                "/prepared-source/prepare_task.py",
                "--root",
                str(inputs),
                "--output",
                str(prepared / "bundle"),
                "--base-image",
                local.IMAGE,
            ],
            check=True,
            timeout=600,
        )
        contract = local.verify_bundle(prepared / "bundle", PLAN_SHA, plan)
        state["runtime_image_id"] = local.image_id()
        env = {k: v for k, v in os.environ.items() if k not in SECRET_ENVIRONMENT}
        env["PYTHONPATH"] = str(inputs / FROZEN_REL)
        jobs = output / "jobs"
        caps = plan["local"]
        for task, job, wall, probe in (
            ("sft-resource-task", local.PROBE_JOB, caps["probe_wall_seconds"], True),
            ("task", local.MAIN_JOB, caps["main_wall_seconds"], False),
        ):
            state.setdefault("prelaunch", {})[job] = require_memory(plan, output)
            write_json(output / "ci-state.json", state)
            command = local.harbor_command(prepared / "bundle" / task, jobs, job, probe=probe)
            command[0] = str(harbor)  # The plan's exact flags; only the executable path differs.
            local.run_harbor(command, output, timeout=wall, env=env, plan_sha=PLAN_SHA)
            if probe:
                local.assert_probe(jobs / job)
            else:
                local.assert_complete(jobs / job, plan["training"]["max_steps"])
    except Exception as exc:  # Recorded, never retried; criteria still report what exists.
        error = f"{type(exc).__name__}: {exc}"
    result = local.evaluate(
        root,
        plan,
        contract,
        plan_sha=PLAN_SHA,
        image=state.get("runtime_image_id"),
        driver_error=error,
    )
    result["platform"] = "GitHub-hosted ubuntu-24.04 x86_64 runner, free public tier"
    state["status"] = "completed" if result["pass"] else "failed"
    state["final_memory"] = meminfo()
    write_json(output / "ci-state.json", state)
    write_json(output / RESULT_NAME, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "criteria", "collect-evidence"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if name == "collect-evidence":
            command.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    if args.command == "collect-evidence":
        copied = collect_evidence(args.root, args.destination or args.root / "evidence")
        print(json.dumps({"copied": len(copied)}))
        return 0
    if args.command == "criteria":
        result = criteria(args.root)
    else:
        result = run(Path.cwd(), args.root)
    print(json.dumps(summary(result), indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
