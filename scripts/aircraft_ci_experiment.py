"""Bounded manual Linux reproduction of CPU7946; importing performs no work.

The frozen worker, trainer, comparison, inputs and scientific gates are reused
byte for byte. Only the operating-system resource supervisor is replaced. The
original absolute paths exist inside an isolated container, never on the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

REPOSITORY = "RyanLin5967/RyanLinEnv"
ORIGINAL = Path("/Users/idide/RyanLinEnv")
PLAN_REL = Path("runs/aircraft-objective-ablation-v1-plan")
OUTPUT_REL = Path("runs/aircraft-objective-ablation-v1")
PLAN_SHA = "7946c228ceede564600ac540706a0fc2020808b163c0543af84dcb6a97cc7b99"
ASSET = "aircraft-cpu7946-inputs-v1.tar"
RELEASE = "aircraft-cpu7946-inputs-v1"
GIB = 1024**3
LIMITS = {
    "docker_memory_bytes": 13 * GIB,
    "tree_rss_bytes": 12 * GIB,
    "prelaunch_available_bytes": 13 * GIB,
    "running_available_bytes": GIB,
    "disk_reserve_bytes": GIB,
    "output_bytes": 2 * GIB,
    "all_phase_wall_seconds": 43200,
    "first4_wall_seconds": 1800,
    "full_wall_seconds": 10800,
    "assay_wall_seconds": 3600,
    "swap_growth_bytes": 0,
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("Archive path is not a canonical relative file")
    return path


def regular(path):
    path = Path(path)
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("Symlinks are forbidden")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Expected a regular file")
    return path


def pack(root, destination, descriptor, warm_receipt):
    """Local preparation only: no downloads, publication, imports or model calls."""
    root = root.resolve()
    plan = read(root / PLAN_REL / "plan.json")
    if digest(root / PLAN_REL / "plan.json") != PLAN_SHA:
        raise ValueError("Expected the frozen CPU7946 plan")
    files = {}
    for source, expected in plan["input_sha256"].items():
        relative = Path(source).relative_to(ORIGINAL)
        if relative.suffix == ".jpg" or relative.is_relative_to("runs/aircraft-warmstart-v1"):
            continue
        files[str(relative)] = expected
    for name, expected in plan["source_sha256"].items():
        files[str(PLAN_REL / name)] = expected
    for name, key in (
        ("run.py", "runner_sha256"),
        ("prepare.py", "preparation_sha256"),
        ("compare.py", "comparison_sha256"),
    ):
        files[str(PLAN_REL / name)] = plan[key]
    files[str(PLAN_REL / "plan.json")] = PLAN_SHA
    for name, expected in files.items():
        if digest(regular(root / name)) != expected:
            raise ValueError(f"Frozen bundle input differs: {name}")
    provenance = read(root / "data/fgvc-aircraft-2013b/candidates/fourshot-seed791/provenance.json")
    selected = set(plan["selection"]["train_ids"] + plan["selection"]["assay_ids"])
    downloads = []
    for row in provenance["images"]:
        if row["id"] in selected:
            relative = "data/fgvc-aircraft-2013b/" + row["path"]
            if plan["input_sha256"][str(ORIGINAL / relative)] != row["encoded_sha256"]:
                raise ValueError("Pinned mirror JPEG differs from the original CPU plan")
            downloads.append(
                {
                    "path": relative,
                    "url": row["url"],
                    "sha256": row["encoded_sha256"],
                    "bytes": row["encoded_bytes"],
                }
            )
    if len(downloads) != 96 or len(selected) != 96:
        raise ValueError("Exactly 64 TRAIN plus 32 distinct TRAIN assay images are required")
    model_rel = Path(plan["paths"]["model"]).relative_to(ORIGINAL)
    model = read(root / model_rel / "model-source.json")
    for row in model["files"]:
        downloads.append(
            {
                "path": str(model_rel / row["path"]),
                "bytes": row["size_bytes"],
                "sha256": row.get("sha256") or digest(regular(root / model_rel / row["path"])),
                "url": f"https://huggingface.co/{model['model_repository']}/resolve/"
                f"{model['revision']}/{row['path']}",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream, tarfile.open(fileobj=stream, mode="w") as archive:
        for name in sorted(files):
            path = root / name
            entry = tarfile.TarInfo(name)
            entry.size = path.stat().st_size
            entry.mode = 0o644
            with path.open("rb") as content:
                archive.addfile(entry, content)
    warm = read(warm_receipt)
    warm_archive = regular(root / warm["archive"])
    if digest(warm_archive) != warm["sha256"] or warm_archive.stat().st_size != warm["bytes"]:
        raise ValueError("Warm release receipt differs from the actual archive")
    warm_files = {}
    with tarfile.open(warm_archive, "r:gz") as archive:
        for member in archive:
            name = str(safe_name(member.name))
            if not member.isfile() or member.size > 16 * 1024**2 or name in warm_files:
                raise ValueError("Unexpected warm bundle member")
            warm_files[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    for name in (
        "local_training.json",
        "final_adapter/adapter_model.safetensors",
        "final_adapter/adapter_config.json",
    ):
        original = str(ORIGINAL / "runs/aircraft-warmstart-v1" / name)
        if warm_files["aircraft-common-warm/" + name] != plan["input_sha256"][original]:
            raise ValueError("Warm release changes the original common initialization")
    write(
        descriptor,
        {
            "schema_version": 1,
            "repository": REPOSITORY,
            "original_plan_sha256": PLAN_SHA,
            "scope": "Linux CPU TRAIN mechanism reproduction; no held-out/frontier result",
            "bundle": {
                "file": ASSET,
                "bytes": destination.stat().st_size,
                "sha256": digest(destination),
                "url": f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}/{ASSET}",
            },
            "files": files,
            "downloads": sorted(downloads, key=lambda row: row["path"]),
            "warm_bundle": {
                "file": Path(warm["archive"]).name,
                "bytes": warm["bytes"],
                "sha256": warm["sha256"],
                "url": f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}/"
                + Path(warm["archive"]).name,
                "files": warm_files,
            },
            "build_files": {
                name: digest(root / name)
                for name in ("Dockerfile.vlm", "requirements-vlm-local.lock", "pyproject.toml")
            },
            "limits": LIMITS,
            "model_bytes": sum(row["size_bytes"] for row in model["files"]),
            "jpeg_bytes": sum(row["bytes"] for row in downloads if row["path"].endswith(".jpg")),
            "scientific_choices_changed": False,
            "linux_execution_verified": False,
            "publication_status": "prepared locally; not uploaded by this script",
        },
    )


def download(url, destination, *, size, sha256):
    """One bounded attempt, stream to one copy, never log credentials/redirect URLs."""
    if destination.exists():
        raise ValueError("Never overwrite prior downloaded evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url)
    count, checksum = 0, hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as out:
        while chunk := response.read(1024**2):
            count += len(chunk)
            if count > size or shutil.disk_usage(destination.parent).free < GIB:
                raise ValueError("Download byte/disk budget exceeded")
            checksum.update(chunk)
            out.write(chunk)
    if checksum.hexdigest() != sha256 or count != size:
        raise ValueError("Downloaded content differs from immutable size/SHA256")
    temporary.replace(destination)


def extract_tar(archive_path, destination, expected, *, limit):
    total, seen = 0, set()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            name = str(safe_name(member.name))
            total += member.size
            if not member.isfile() or name in seen or name not in expected or total > limit:
                raise ValueError("Unexpected, duplicate, nonregular or oversized archive member")
            seen.add(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as out:
                shutil.copyfileobj(source, out, 1024**2)
            if digest(regular(target)) != expected[name]:
                raise ValueError("Extracted member SHA256 differs")
    if seen != set(expected):
        raise ValueError("Archive is missing pinned members")


def api_json(path):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}" + ("/" + path if path else ""),
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read(4 * 1024**2))


def import_runner():
    directory = ORIGINAL / PLAN_REL
    sys.path.insert(0, str(directory))
    import run

    if Path(run.__file__).resolve() != directory / "run.py":
        raise ValueError("Wrong frozen worker module")
    return run


def check_inputs():
    runner = import_runner()
    plan = runner.verify(PLAN_SHA)
    local = runner._source_import()
    for identifier, expected in plan["model_visible_pixels"].items():
        path = ORIGINAL / "data/fgvc-aircraft-2013b/development/images/train" / f"{identifier}.jpg"
        if local.aircraft_pixel_sha256(path) != expected:
            raise ValueError("Decoded model-visible pixels differ")
    return runner, plan


def completed_phases(output):
    results = {}
    for path in (output / "phases").glob("*/guard.json"):
        row = read(path)
        if row["status"] != "completed" or row.get("stop_reason"):
            raise ValueError("Failed/partial phases are terminal; no automatic retry")
        results[path.parent.name] = row
    return results


def supervise_phase(phase, limits):
    """Container-local supervisor, no model imports; preserve every terminal failure."""
    import psutil

    runner, _ = check_inputs()
    output = ORIGINAL / OUTPUT_REL
    prior = completed_phases(output)
    prior_seconds = sum(row["elapsed_seconds"] for row in prior.values())
    wall = (
        limits["first4_wall_seconds"]
        if phase.endswith("first4")
        else limits["full_wall_seconds"]
        if phase.endswith("full")
        else limits["assay_wall_seconds"]
    )
    wall = min(wall, limits["all_phase_wall_seconds"] - prior_seconds)
    if wall <= 0:
        raise ValueError("Cumulative physical wall budget exhausted")
    phase_dir = output / "phases" / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OBJECTIVE_SUPERVISOR_PID=str(os.getpid()))
    start = time.monotonic()
    initial_swap = psutil.swap_memory().used
    receipt = {
        "phase": phase,
        "status": "running",
        "elapsed_seconds": 0.0,
        "limits": limits,
        "platform": "Linux CPU, separate from Mac study",
        "peak_tree_rss_bytes": 0,
        "prior_physical_phase_seconds": prior_seconds,
    }

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Supervisor signal {signum}")

    previous = {
        s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    with (phase_dir / "actor.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, __file__, "worker", "--phase", phase],
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        receipt["child_pid"] = child.pid
        try:
            process = psutil.Process(child.pid)
            while child.poll() is None:
                rss = runner._tree_rss(process)
                available = psutil.virtual_memory().available
                free = shutil.disk_usage(output).free
                elapsed = time.monotonic() - start
                size = runner.output_size()
                receipt.update(
                    elapsed_seconds=elapsed,
                    output_bytes=size,
                    peak_tree_rss_bytes=max(receipt["peak_tree_rss_bytes"], rss),
                )
                runner.append(
                    phase_dir / "resources.jsonl",
                    {
                        "elapsed_seconds": elapsed,
                        "tree_rss_bytes": rss,
                        "host_available_bytes": available,
                        "disk_free_bytes": free,
                    },
                )
                if rss > limits["tree_rss_bytes"] or available < limits["running_available_bytes"]:
                    raise RuntimeError("Memory resource guard")
                if psutil.swap_memory().used - initial_swap > limits["swap_growth_bytes"]:
                    raise RuntimeError("Swap resource guard")
                if free < limits["disk_reserve_bytes"] or size > limits["output_bytes"]:
                    raise RuntimeError("Disk/output resource guard")
                if elapsed > wall:
                    raise RuntimeError("Physical wall resource guard")
                write(phase_dir / "guard.json", receipt)
                time.sleep(0.5)
        except BaseException as exc:
            receipt["stop_reason"] = str(exc) or type(exc).__name__
            if child.poll() is None:
                for signum in previous:
                    signal.signal(signum, signal.SIG_IGN)
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
        finally:
            receipt.update(exit_code=child.wait(), elapsed_seconds=time.monotonic() - start)
            receipt["status"] = (
                "completed"
                if child.returncode == 0 and not receipt.get("stop_reason")
                else "failed"
            )
            write(phase_dir / "guard.json", receipt)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    if receipt["status"] != "completed":
        raise RuntimeError("Phase failed; preserve partial work without retry")
    if phase.endswith("first4") and not read(phase_dir / "result.json")["hardware"]["passed"]:
        raise RuntimeError("First4 numerical/throughput gate failed; no continuation")


def container_stage(arm):
    if sys.platform != "linux" or os.environ.get("ACCELERATE_USE_CPU") != "true":
        raise RuntimeError("Requires the explicit offline Linux CPU container")
    if arm not in {"grpo", "dr_grpo"}:
        raise ValueError("Expected one fixed first4 arm")
    filesystem_preflight(ORIGINAL / OUTPUT_REL)
    check_inputs()
    if completed_phases(ORIGINAL / OUTPUT_REL):
        raise ValueError("First4 requires a fresh output; no retries or continuation")
    supervise_phase(f"{arm}-first4", LIMITS)


def filesystem_preflight(output):
    """Exercise the actual mounted output with the training UID before model work."""
    expected_uid = int(os.environ["EXPERIMENT_UID"])
    expected_gid = int(os.environ["EXPERIMENT_GID"])
    if (os.geteuid(), os.getegid()) != (expected_uid, expected_gid):
        raise RuntimeError("Container identity differs from the output owner")
    probe_directory = output / ".write-preflight"
    probe_directory.mkdir()
    probe = probe_directory / "probe"
    content = b"bounded CPU output permission check\n"
    with probe.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if probe.read_bytes() != content:
        raise RuntimeError("Mounted output write/read check failed")
    probe.unlink()
    probe_directory.rmdir()
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "output_uid": output.stat().st_uid,
        "output_gid": output.stat().st_gid,
        "output_mode": stat.S_IMODE(output.stat().st_mode),
        "create_write_fsync_read_unlink_passed": True,
        "model_calls": 0,
        "worker_started": False,
    }
    write(output / "filesystem-preflight.json", receipt)
    return receipt


def host_run(root, descriptor, arm):
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
        or sys.platform != "linux"
    ):
        raise ValueError("Only a first manual attempt in the owned public repository is allowed")
    if arm not in {"grpo", "dr_grpo"}:
        raise ValueError(
            "This first release permits one arm's first4 only; continuation is not enabled"
        )
    contract = read(descriptor)
    if contract["limits"] != LIMITS or contract["original_plan_sha256"] != PLAN_SHA:
        raise ValueError("Prospective Linux contract differs")
    repository = api_json("")
    if repository["private"]:
        raise ValueError("This experiment permits only standard public-repository runners")
    descriptor_sha = digest(descriptor)
    output, inputs = root / "output", root / "inputs"
    if root.exists():
        raise ValueError("Never reuse a partially started CI directory")
    root.mkdir(parents=True)
    output.mkdir()
    bundle = root / ASSET
    download(
        contract["bundle"]["url"],
        bundle,
        size=contract["bundle"]["bytes"],
        sha256=contract["bundle"]["sha256"],
    )
    extract_tar(bundle, inputs, contract["files"], limit=32 * 1024**2)
    bundle.unlink()
    warm = contract["warm_bundle"]
    warm_path = root / warm["file"]
    download(warm["url"], warm_path, size=warm["bytes"], sha256=warm["sha256"])
    extract_tar(warm_path, root / "warm-release", warm["files"], limit=32 * 1024**2)
    (root / "warm-release/aircraft-common-warm").rename(inputs / "runs/aircraft-warmstart-v1")
    warm_path.unlink()
    (inputs / OUTPUT_REL).mkdir(parents=True)
    # Build once, with the exact frozen package rather than whatever current src contains.
    context = root / "context"
    context.mkdir()
    for name, expected in contract["build_files"].items():
        if digest(name) != expected:
            raise ValueError("Docker/lock/project metadata differs from the reviewed contract")
        shutil.copyfile(name, context / name)
    shutil.copytree(inputs / PLAN_REL / "source", context / "src")
    image = "vision-rl:aircraft-cpu7946"
    started = time.monotonic()
    subprocess.run(
        ["docker", "build", "-f", str(context / "Dockerfile.vlm"), "-t", image, str(context)],
        check=True,
        timeout=1200,
    )
    build_seconds = time.monotonic() - started
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", image], text=True
    ).strip()
    needed = sum(row["bytes"] for row in contract["downloads"]) + LIMITS["output_bytes"] + GIB
    if shutil.disk_usage(root).free < needed:
        raise RuntimeError("Insufficient disk for one base copy + bounded outputs + reserve")
    for row in contract["downloads"]:
        safe_name(row["path"])
        if not row["url"].startswith("https://huggingface.co/"):
            raise ValueError("Only the pinned public model/image mirror is allowed")
        download(row["url"], inputs / row["path"], size=row["bytes"], sha256=row["sha256"])
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if len(line.split()) == 3
    }
    if memory["MemAvailable"] < LIMITS["prelaunch_available_bytes"]:
        raise RuntimeError("Linux prelaunch memory guard; no model call")
    state = {
        "schema_version": 1,
        "status": "running",
        "group": "first4",
        "arm": arm,
        "commit": os.environ["GITHUB_SHA"],
        "run_id": os.environ["GITHUB_RUN_ID"],
        "descriptor_sha256": descriptor_sha,
        "original_plan_sha256": PLAN_SHA,
        "image_id": image_id,
        "build_seconds": build_seconds,
        "prelaunch_memory": memory,
        "container_uid": os.getuid(),
        "container_gid": os.getgid(),
        "cpu_hardware": json.loads(subprocess.check_output(["lscpu", "--json"], text=True)),
        "platform": "Linux x86_64 CPU, independent of Mac experiments",
    }
    write(output / "ci-state.json", state)
    command = docker_command(inputs, output, image_id, arm)
    try:
        subprocess.run(command, check=True, timeout=1950)
        state["status"] = "completed"
    finally:
        if state["status"] != "completed":
            state["status"] = "failed"
        inspected = subprocess.run(
            ["docker", "inspect", "aircraft-cpu-study"], capture_output=True, text=True
        )
        if inspected.returncode == 0:
            state["container_state"] = json.loads(inspected.stdout)[0]["State"]
            subprocess.run(["docker", "stop", "--time", "20", "aircraft-cpu-study"], check=False)
        state["file_sha256"] = {
            str(p.relative_to(output)): digest(regular(p))
            for p in output.rglob("*")
            if p.is_file() and p != output / "ci-state.json"
        }
        write(output / "ci-state.json", state)


def docker_command(inputs, output, image_id, arm):
    return [
        "docker",
        "run",
        "--init",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--name",
        "aircraft-cpu-study",
        "--network",
        "none",
        "--cpus",
        "2",
        "--memory",
        str(13 * GIB),
        "--memory-swap",
        str(13 * GIB),
        "--pids-limit",
        "128",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=268435456",
        "--shm-size",
        "64m",
        "-e",
        "ACCELERATE_USE_CPU=true",
        "-e",
        f"EXPERIMENT_UID={os.getuid()}",
        "-e",
        f"EXPERIMENT_GID={os.getgid()}",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "MKL_NUM_THREADS=2",
        "-e",
        "OPENBLAS_NUM_THREADS=2",
        "-e",
        "HF_HOME=/tmp/huggingface",
        "-e",
        "PYTORCH_ENABLE_MPS_FALLBACK=0",
        "-v",
        f"{inputs.resolve()}:{ORIGINAL}:ro",
        "-v",
        f"{output.resolve()}:{ORIGINAL / OUTPUT_REL}:rw",
        "-v",
        f"{Path(__file__).resolve()}:/ci/experiment.py:ro",
        image_id,
        "python",
        "/ci/experiment.py",
        "container-stage",
        "--arm",
        arm,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "run", "worker", "container-stage"))
    parser.add_argument("--root", type=Path, default=Path("runs/aircraft-ci"))
    parser.add_argument(
        "--descriptor", type=Path, default=Path("docs/aircraft-cpu-experiment-plan.json")
    )
    parser.add_argument("--bundle", type=Path, default=Path("runs/aircraft-ci-preparation") / ASSET)
    parser.add_argument(
        "--warm-receipt", type=Path, default=Path("runs/aircraft-warm-release-v1/preparation.json")
    )
    parser.add_argument("--phase")
    parser.add_argument("--arm", choices=("grpo", "dr_grpo"))
    args = parser.parse_args()
    if args.command == "pack":
        pack(Path.cwd(), args.bundle, args.descriptor, args.warm_receipt)
    elif args.command == "run":
        host_run(args.root, args.descriptor, args.arm)
    elif args.command == "container-stage":
        container_stage(args.arm)
    else:
        import_runner().worker(PLAN_SHA, args.phase)


if __name__ == "__main__":
    main()
