"""GitHub-hosted reproduction of the passing local pretrained Harbor dogfood (plan 62fe9ff1).

No training, packaging, guard or criteria logic lives here. Every step reuses code that
already ran in the local PASS (reports/2026-10-16-harbor-pretrained-dogfood):

- the plan directory is the tracked plan-source.tar.gz, verified against plan-SHA256SUMS;
- host helpers and build files are the plan's own pins, read from source commit 1323eb1;
- inputs, warm adapter f05840be, model and fixture JPEGs come from
  scripts/aircraft_ci_continue.setup_runtime (the CPU7946 public release + pinned HF URLs);
- packaging is plan README step 6; the trial is local_run.execute with only its darwin/Colima
  launch gate replaced by a GitHub-runner gate; C1-C4 are local_run's own criteria.

The plan hard-codes /Users/idide/RyanLinEnv (model bind source, fixture paths, driver paths),
so the runner recreates that absolute root instead of editing pinned plan bytes.
Importing performs no work. Only `criteria` and `collect` run off a GitHub runner.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

REPOSITORY = "RyanLin5967/RyanLinEnv"
ROOT = Path("/Users/idide/RyanLinEnv")
HOST = Path("/Users/idide/ci-host")
WORK = Path("/Users/idide/ci-work")
PLAN_DIR = ROOT / "runs/aircraft-harbor-dogfood-local-v4-plan"
RUN_ROOT = ROOT / "runs/aircraft-harbor-dogfood-local-v4"
REPORT = Path(__file__).resolve().parents[1] / "reports/2026-10-16-harbor-pretrained-dogfood"
PLAN_ARCHIVE = REPORT / "plan-source.tar.gz"
PLAN_SUMS = REPORT / "plan-SHA256SUMS"
PLAN_SHA = "62fe9ff1d71ba6c50a183ed1ffecdc2318be2fe5b31c521b3079e6aedc494901"
SOURCE_COMMIT = "1323eb10b376674c5d670995fc61246d2445c825"
IMAGE = "vision-rl:aircraft-cpu7946-continuation"
CRITERIA = (
    "C1_pretrained_model_loaded",
    "C2_real_optimizer_update",
    "C3_separate_verifier_scored_submission",
    "C4_resource_guards_held",
)
GIB = 1024**3
# Evidence allowlist: text receipts only. Base weights, adapters, tokenizer copies, pickles,
# generated code and images are never uploaded; anything not allowlisted here is skipped.
EVIDENCE_NAMES = (
    "*.log",
    "*_diagnostic_predictions.json",
    "ci-criteria.json",
    "ci-stage.json",
    "config.json",
    "dogfood-local-result.json",
    "endpoint_receipt.json",
    "guard.json",
    "invocation-*.json",
    "loader.json",
    "local_training.json",
    "lock.json",
    "manifest.json",
    "optimizer_audit.jsonl",
    "physical_prompt_events.jsonl",
    "predictions.json",
    "preparation.json",
    "receipt.json",
    "report.json",
    "resources.jsonl",
    "result.json",
    "reward.txt",
    "reward_groups.jsonl",
    "sft-resource-probe.json",
    "test-stdout.txt",
    "trainer_state.json",
    "training-evidence.json",
)
MODEL_DIRECTORIES = ("final_adapter", "checkpoint-*", "ref")
EVIDENCE_FILE_CAP = 64 * 1024**2
EVIDENCE_TOTAL_CAP = 512 * 1024**2


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_sums(path=PLAN_SUMS):
    sums = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if not name.startswith("plan/") or name in sums or len(digest) != 64:
            raise ValueError(f"Malformed plan-SHA256SUMS line: {line!r}")
        sums[name.removeprefix("plan/")] = digest
    if sums.get("plan.json") != PLAN_SHA:
        raise ValueError("plan-SHA256SUMS does not pin the passing plan 62fe9ff1")
    return sums


def extract_plan(destination, archive=PLAN_ARCHIVE, sums_path=PLAN_SUMS):
    """Extract exactly the pinned plan files; any extra, missing or changed byte refuses."""
    sums = read_sums(sums_path)
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Plan destination already exists: {destination}")
    destination.mkdir(parents=True)
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if member.isdir() and member.name.rstrip("/") == "plan":
                continue
            # macOS tar wrote an AppleDouble sidecar (163 B of xattrs) before each pinned
            # file; it is metadata only and is never extracted.
            sidecar = path.name.removeprefix("._")
            if (
                path.name.startswith("._")
                and member.isfile()
                and member.size <= 4096
                and str(path.parent / sidecar) in {"plan", *("plan/" + n for n in sums)}
            ):
                continue
            name = str(path.relative_to("plan")) if path.parts[:1] == ("plan",) else None
            if not member.isfile() or name not in sums or name in seen or "/" in name:
                raise ValueError(f"Unexpected plan archive member: {member.name}")
            data = stream.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != sums[name]:
                raise ValueError(f"Plan archive member differs from its pin: {name}")
            (destination / name).write_bytes(data)
            seen.add(name)
    if seen != set(sums):
        raise ValueError(f"Plan archive is missing pinned files: {sorted(set(sums) - seen)}")
    return destination


def load_local_run(plan_dir):
    """Import the plan's own driver after re-checking every plan file against the pins."""
    plan_dir = Path(plan_dir)
    for name, expected in read_sums().items():
        if sha(plan_dir / name) != expected:
            raise ValueError(f"Plan file differs from plan-SHA256SUMS: {name}")
    spec = importlib.util.spec_from_file_location("dogfood_local_run", plan_dir / "local_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_github_runner(environ=None, platform=None):
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if (
        platform != "linux"
        or environ.get("GITHUB_ACTIONS") != "true"
        or environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or environ.get("GITHUB_RUN_ATTEMPT") != "1"
    ):
        raise ValueError("Only a first manual attempt on a GitHub-hosted Linux runner is allowed")


def ci_launch_gate(plan, *, environ=None, platform=None, cwd=None):
    """Replaces local_run.require_local_launch: same review/scope gates, runner instead of Mac."""
    if (
        plan.get("local_launch_reviewed") is not True
        or plan.get("cgroup_policy_reviewed") is not True
    ):
        raise ValueError("Plan unreviewed; no launch")
    if plan["training"]["max_steps"] != 1:
        raise ValueError("The plan is defined only for max_steps == 1")
    require_github_runner(environ, platform)
    if (Path.cwd() if cwd is None else Path(cwd)).resolve() != ROOT:
        raise ValueError("Run from the recreated plan root")


def run(command, *, timeout, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run([str(c) for c in command], check=True, timeout=timeout, **kwargs)


def meminfo():
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if len(line.split()) == 3
    }


def materialize_host_files(plan, destination):
    """The plan's pinned host/build files, byte-exact from its source commit."""
    names = sorted(set(plan["host_source_sha256"]) | set(plan["build_files"]))
    for name in names:
        expected = plan["host_source_sha256"].get(name) or plan["build_files"][name]
        data = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{name}"], timeout=60)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Pinned host file differs at {SOURCE_COMMIT[:7]}: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return names


def require_plan_constants(plan):
    """The literals this script uses must be the plan's own pins, or nothing runs."""
    observed = (
        plan["source_commit"],
        plan["local"]["image"],
        plan["resource"]["prospective_override_memory_mib"] * 1024**2,
        plan["resource"]["container_memory_bytes"],
    )
    if observed != (SOURCE_COMMIT, IMAGE, 13 * GIB, 13 * GIB):
        raise ValueError(f"Plan pins differ from this driver's constants: {observed}")


def kernel_preflight(plan):
    """Plan README step 3 against the built image: exact cgroup bounds, then host memory/disk.

    Deviation: the README runs the cgroup read in python:3.12.13-slim; this runs it in the
    runtime image the trial uses. The bounds checked are the same.
    """
    memory_flag = f"{plan['resource']['prospective_override_memory_mib']}m"
    output = subprocess.check_output(
        [
            "docker", "run", "--rm", "--network", "none", "--memory", memory_flag,
            "--memory-swap", memory_flag, "--entrypoint", "sh", IMAGE, "-c",
            "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max "
            "/sys/fs/cgroup/memory.swap.current",
        ],
        text=True,
        timeout=300,
    ).split()
    resource = plan["resource"]
    observed = [int(x) for x in output]
    if observed != [resource["container_memory_bytes"], 0, 0]:
        raise RuntimeError(f"Kernel cgroup preflight failed: {observed}")
    memory = meminfo()
    free = shutil.disk_usage(ROOT).free
    require_host_headroom(memory["MemAvailable"], free)
    return {"cgroup": observed, "meminfo": memory, "disk_free_bytes": free}


def require_host_headroom(mem_available, disk_free):
    """Plan README step 3's own pass line: MemAvailable >= 14 GiB and root free >= 4 GiB.

    Deliberately the README's 14 GiB, not the runtime's 13 GiB `prelaunch_available_bytes`
    (which local_run still enforces per child): the stricter of the plan's two gates applies.
    """
    if mem_available < 14 * GIB:
        raise RuntimeError(f"Host MemAvailable {mem_available} B below README step 3's 14 GiB")
    if disk_free < 4 * GIB:
        raise RuntimeError(f"Root free disk {disk_free} B below README step 3's 4 GiB")


def stage():
    require_github_runner()
    for path in (ROOT, HOST, WORK):
        if path.exists():
            raise ValueError(f"Fresh runner required; {path} exists")
    repository = Path.cwd()
    plan_copy = extract_plan(HOST / "plan")
    plan = json.loads((plan_copy / "plan.json").read_text())
    require_plan_constants(plan)
    host_files = materialize_host_files(plan, HOST)
    os.chdir(HOST)  # setup_runtime checks the build files relative to the working directory.
    spec = importlib.util.spec_from_file_location(
        "aircraft_ci_continue", HOST / "scripts/aircraft_ci_continue.py"
    )
    continuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(continuation)
    descriptor = HOST / "docs/aircraft-cpu-experiment-plan.json"
    if sha(descriptor) != plan["first4_descriptor_sha256"]:
        raise ValueError("CPU7946 descriptor differs from the plan pin")
    contract = continuation.ci.read(descriptor)
    if contract["loader"] != plan["loader"] or contract["build_files"] != plan["build_files"]:
        raise ValueError("Descriptor loader/build contract differs from the plan")
    inputs, image_id = continuation.setup_runtime(WORK, contract)
    inputs.rename(ROOT)
    for name in host_files:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((HOST / name).read_bytes())
    extract_plan(PLAN_DIR)
    os.chdir(ROOT)
    environment = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv"))
    run(
        ["uv", "sync", "--project", ROOT, "--extra", "harbor", "--locked",
         "--no-install-project", "--python", "3.12"],
        timeout=900,
        env=environment,
    )
    harbor_version = subprocess.check_output(
        [ROOT / ".venv/bin/python", "-I", "-c",
         "import importlib.metadata as m; print(m.version('harbor'))"],
        text=True,
        timeout=60,
    ).strip()
    if harbor_version != plan["host_dependencies"]["harbor"]:
        raise ValueError(f"Harbor {harbor_version} differs from the pinned host dependency")
    run(["docker", "run", "--rm", "--network", "none", IMAGE], timeout=300)  # README step 4.
    preflight = kernel_preflight(plan)
    prepared = RUN_ROOT / "prepared"
    prepared.mkdir(parents=True)
    run(  # README step 6, same mounts and paths.
        [
            "docker", "run", "--rm", "--network", "none", "--cpus", "1",
            "--memory", "1g", "--memory-swap", "1g", "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{PLAN_DIR}:/prepared-source:ro",
            "-v", f"{prepared}:{prepared}:rw", IMAGE, "python", "-I",
            "/prepared-source/prepare_task.py", "--root", ROOT,
            "--output", prepared / "bundle", "--base-image", IMAGE,
        ],
        timeout=300,
    )
    local_run = load_local_run(PLAN_DIR)
    local_run.require_wall_nesting(plan)
    local_run.verify_pins(plan)
    local_run.verify_bundle(prepared / "bundle", PLAN_SHA, plan)
    write_json(
        RUN_ROOT / "ci-stage.json",
        {
            "plan_sha256": PLAN_SHA,
            "source_commit": SOURCE_COMMIT,
            "workflow_commit": os.environ.get("GITHUB_SHA"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "checkout": str(repository),
            "runtime_image_id": image_id,
            "harbor_version": harbor_version,
            "preflight": preflight,
            "cpu": subprocess.check_output(["lscpu", "--json"], text=True, timeout=30),
            "scope": plan["scope"],
            "platform": "GitHub-hosted ubuntu-24.04 x86_64, replacing the local Colima aarch64 VM",
        },
    )


def execute():
    require_github_runner()
    os.chdir(ROOT)
    local_run = load_local_run(PLAN_DIR)
    local_run.require_local_launch = ci_launch_gate
    result = local_run.execute(RUN_ROOT)
    print(json.dumps({k: v["pass"] for k, v in result["criteria"].items()}, indent=2))
    print("driver_error:", result["driver_error"], "pass:", result["pass"])
    return 0 if result["pass"] else 1


def driver_record_error(path):
    """None only if local_run.execute wrote its result with pass true and no driver_error.

    A missing file means execute never finished (refused before launch, or interrupted by the
    outer SIGINT timeout, which is a KeyboardInterrupt local_run does not catch).
    """
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return f"driver result unreadable ({Path(path).name}): {type(exc).__name__}: {exc}"
    if not isinstance(record, dict):
        return f"driver result is not an object ({Path(path).name})"
    if record.get("driver_error") is not None:
        return f"driver reported: {record['driver_error']}"
    if record.get("pass") is not True:
        return f"driver result pass is {record.get('pass')!r}, not true"
    return None


def criteria(root, plan_dir, result_path):
    """Re-derive C1-C4 from evidence on disk with the plan's own checks; non-zero on any miss."""
    local_run = load_local_run(plan_dir)
    plan = json.loads((Path(plan_dir) / "plan.json").read_text())
    root = Path(root)
    jobs, error = root / "output/jobs", None
    try:
        contract_path = root / "prepared/bundle/task/environment/contract.json"
        contract = json.loads(contract_path.read_text())
    except (OSError, ValueError) as exc:
        contract, error = {}, f"{type(exc).__name__}: {exc}"
    try:
        local_run.assert_probe(jobs / local_run.PROBE_JOB)
        local_run.assert_complete(jobs / local_run.MAIN_JOB, plan["training"]["max_steps"])
    except Exception as exc:  # Same as local_run's read-only `criteria` path.
        error = error or f"{type(exc).__name__}: {exc}"
    errors = [error] if error else []
    driver = driver_record_error(root / "output" / local_run.RESULT_NAME)
    if driver:  # The driver's own failures (Harbor exit, wall/disk guard, SIGINT) count too.
        errors.append(driver)
    error = "; ".join(errors) or None
    result = local_run.evaluate(root, plan, contract, plan_sha=PLAN_SHA, driver_error=error)
    try:  # The plan's own read-only checker must agree; a later fix to it reaches CI too.
        plan_exit = local_run.main(["criteria", "--root", str(root)])
    except Exception as exc:
        plan_exit = f"{type(exc).__name__}: {exc}"
    missing = [name for name in CRITERIA if name not in result["criteria"]]
    passed = (
        not missing
        and plan_exit == 0
        and result["pass"] is True
        and all(result["criteria"][name]["pass"] is True for name in CRITERIA)
    )
    result["ci_checker"] = {
        "missing_criteria": missing,
        "local_run_criteria_exit": plan_exit,
        "pass": passed,
    }
    write_json(result_path, result)
    for name in CRITERIA:
        row = result["criteria"].get(name, {"pass": False, "reason": "criterion missing"})
        print(f"{name}: {'PASS' if row['pass'] else 'FAIL ' + row['reason']}")
    print("driver_error:", error)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


def collect(root, destination):
    """Copy allowlisted text evidence only; every skipped file is listed, never uploaded."""
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise ValueError("Evidence destination must be new")
    destination.mkdir(parents=True)
    copied, skipped, total = {}, [], 0
    sources = [
        root / "ci-stage.json",
        root / "ci-criteria.json",
        root / "prepared/bundle/preparation.json",
    ]
    if (root / "output").is_dir():
        sources += sorted(p for p in (root / "output").rglob("*") if not p.is_dir())
    for path in sources:
        relative = path.relative_to(root)
        try:
            status = path.lstat()
        except FileNotFoundError:
            continue  # Optional top-level receipt, or rotated away after the listing.
        reason = None
        if not stat.S_ISREG(status.st_mode):
            reason = "not a regular file"
        elif any(fnmatch.fnmatch(part, d) for part in relative.parts for d in MODEL_DIRECTORIES):
            reason = "model artifact directory (adapter/tokenizer copies; hashes are in receipts)"
        elif not any(fnmatch.fnmatch(relative.name, name) for name in EVIDENCE_NAMES):
            reason = "name not allowlisted"
        elif status.st_size > EVIDENCE_FILE_CAP or total + status.st_size > EVIDENCE_TOTAL_CAP:
            reason = "size cap"
        if reason:
            skipped.append({"path": str(relative), "bytes": status.st_size, "reason": reason})
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(path, target)
        except FileNotFoundError:
            skipped.append({"path": str(relative), "bytes": status.st_size, "reason": "vanished"})
            continue
        copied[str(relative)] = sha(target)
        total += target.stat().st_size
    write_json(
        destination / "evidence-manifest.json",
        {"copied_sha256": copied, "skipped": skipped, "copied_bytes": total},
    )
    print(f"collected {len(copied)} files ({total} B), skipped {len(skipped)}")
    if not copied:  # A manifest alone is not evidence; zero copied files is a failed collect.
        print("FAIL: no evidence file was collected", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage")
    sub.add_parser("execute")
    check = sub.add_parser("criteria")
    check.add_argument("--root", type=Path, default=RUN_ROOT)
    check.add_argument("--plan-dir", type=Path, default=PLAN_DIR)
    check.add_argument("--result", type=Path)
    gather = sub.add_parser("collect")
    gather.add_argument("--root", type=Path, default=RUN_ROOT)
    gather.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage()
        return 0
    if args.command == "execute":
        return execute()
    if args.command == "criteria":
        result = args.result or args.root / "ci-criteria.json"
        return criteria(args.root, args.plan_dir, result)
    return collect(args.root, args.dest)


if __name__ == "__main__":
    sys.exit(main())
