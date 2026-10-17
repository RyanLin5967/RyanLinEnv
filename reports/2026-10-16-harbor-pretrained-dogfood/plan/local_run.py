"""Local Colima driver for the pretrained Harbor dogfood; fail-closed until local review.

Derived from runs/aircraft-harbor-dogfood-v1-plan/run.py. The functions sha,
stop_owned_containers, run_harbor, artifact_file, assert_runtime_receipts and
assert_probe are verbatim copies; tree_bytes and the one-line run_harbor change
are claude-audit-v1/run-output-size-race.patch. test_local_plan.py proves both.
No subcommand except `execute` calls Docker, Harbor or any model.
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
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path("/Users/idide/RyanLinEnv")
DOCKER_HOST = "unix:///Users/idide/.colima/vision-harbor/docker.sock"
IMAGE = "vision-rl:aircraft-cpu7946-continuation"
FROZEN_SOURCE = REPOSITORY_ROOT / "runs/aircraft-objective-ablation-v1-plan/source"
DEFAULT_ROOT = REPOSITORY_ROOT / "runs/aircraft-harbor-dogfood-local-v4"
HARBOR = REPOSITORY_ROOT / ".venv/bin/harbor"
PROBE_JOB = "aircraft-harbor-dogfood-local-sft-probe-v4"
MAIN_JOB = "aircraft-harbor-dogfood-local-v4"
RESULT_NAME = "dogfood-local-result.json"
GIB = 1024**3


# ---- verbatim from the draft run.py ---------------------------------------------------


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stop_owned_containers(plan_sha):
    ids = subprocess.check_output(
        ["docker", "ps", "-q", "--filter", "label=vision-rl.harbor-plan=" + plan_sha],
        text=True,
        timeout=10,
    ).split()
    if ids:
        subprocess.run(["docker", "stop", "--time", "20", *ids], check=True, timeout=45)


def tree_bytes(root):
    """Sum regular-file bytes under root without following symlinks.

    The container rotates checkpoints (shutil.rmtree) inside the live bind mount,
    so an entry listed by the walk may vanish before lstat; that entry counts as
    zero. Any other error (missing root, permission) propagates: fail closed.
    """

    def vanished_only(error):
        if not isinstance(error, FileNotFoundError):
            raise error

    os.lstat(root)  # A missing root is an error, never a zero-byte pass.
    total = 0
    for directory, _, names in os.walk(root, onerror=vanished_only):
        for name in names:
            try:
                status = os.lstat(os.path.join(directory, name))
            except FileNotFoundError:
                continue
            if stat.S_ISREG(status.st_mode):
                total += status.st_size
    return total


def run_harbor(command, output, *, timeout, env, plan_sha):
    """Bound host-side copies and stop only this authored task's containers."""
    start = time.monotonic()
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            size = tree_bytes(output)
            if size > 2 * 1024**3 or shutil.disk_usage(output).free < 1024**3:
                raise RuntimeError("Harbor host output/disk guard")
            if time.monotonic() - start > timeout:
                raise RuntimeError("Harbor host physical wall guard")
            time.sleep(0.5)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stop_owned_containers(plan_sha)


def artifact_file(job, name):
    paths = [p for p in job.rglob(name) if p.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one retained {name}")
    return paths[0]


def assert_runtime_receipts(job):
    policy = json.loads((HERE / "plan.json").read_text())["loader"]
    guards = [json.loads(p.read_text()) for p in job.rglob("guard.json")]
    loaders = [json.loads(p.read_text()) for p in job.rglob("loader.json")]
    if not guards or len(guards) != len(loaders):
        raise ValueError("Every guarded model phase requires its own loader receipt")
    for row in guards:
        memory = row.get("cgroup_memory", {})
        if (
            row.get("status") != "completed"
            or row.get("exit_code") != 0
            or row.get("stop_reason")
            or row["peak_tree_rss_bytes"] > 12 * 1024**3
            or memory.get("memory.max") != 13 * 1024**3
            or memory.get("memory.swap.max") != 0
            or memory.get("memory.swap.current") != 0
        ):
            raise ValueError("A native model phase failed its actual resource gate")
    for row in loaders:
        if (
            any(row.get(k) != v for k, v in policy.items())
            or row.get("status") != "completed"
            or not row.get("aliases_restored")
            or not row.get("environment_restored")
            or row.get("model_directory") != "/opt/pinned-model"
            or row.get("runtime_sources") != policy["source_sha256"]
        ):
            raise ValueError("Scoped loader identity/completion differs")
        calls = row.get("opens", [])
        if not 1 <= len(calls) <= 64 or {c.get("file") for c in calls} != set(policy["shards"]):
            raise ValueError("Both pinned base shards must use the reviewed loader")
        if any(c != {"file": c["file"], "device": "cpu", "backend": "pread"} for c in calls):
            raise ValueError("Unexpected loader backend")


def assert_probe(job):
    assert_runtime_receipts(job)
    value = json.loads(artifact_file(job, "sft-resource-probe.json").read_text())
    if value["receipt"]["global_step"] != 1:
        raise ValueError("Native SFT resource probe must commit exactly one update")
    guards = [json.loads(p.read_text()) for p in job.rglob("guard.json")]
    if len(guards) != 1 or guards[0]["status"] != "completed" or guards[0].get("stop_reason"):
        raise ValueError("SFT resource probe guard failed")


# ---- local changes -------------------------------------------------------------------


def read_plan():
    return json.loads((HERE / "plan.json").read_text())


def harbor_command(task, output, name, *, probe=False):
    """The draft's exact Harbor flags; only the executable is the repo's pinned venv."""
    command = [
        str(HARBOR),
        "run",
        "-p",
        str(task),
        "--agent",
        "vision_rl.harbor_agent:DetectionTrainingAgent",
        "--job-name",
        name,
        "--jobs-dir",
        str(output),
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--override-cpus",
        "2",
        "--override-memory-mb",
        "13312",
    ]
    if probe:
        command.append("--disable-verification")
    return command


def assert_complete(job, expected_steps=1):
    """Draft assert_complete with global_step == 1 (dogfood-design changed.training.max_steps)."""
    if expected_steps != 1:
        raise ValueError("The local plan commits exactly one update per exposure arm")
    assert_runtime_receipts(job)
    evidence = json.loads(artifact_file(job, "training-evidence.json").read_text())
    if evidence["status"] != "completed":
        raise ValueError("All training/control stages must finish")
    arms = evidence["arms"]
    if any(
        arms[k]["receipt"]["global_step"] != expected_steps for k in ["rl-exposure", "sft-exposure"]
    ):
        raise ValueError("Both exposure controls must commit exactly one update")
    if arms["sft-time"]["receipt"]["committed_active_seconds"] < arms["sft-time"]["target_seconds"]:
        raise ValueError("SFT must reach the actual RL active-time boundary")
    reports = [p for p in job.rglob("report.json") if p.parent.name == "verifier"]
    if len(reports) != 1:
        raise ValueError("Fresh independent verifier report missing")
    report = json.loads(reports[0].read_text())
    if (
        "error" in report
        or report["scope"] != "integration_only"
        or report["submitted_code_executed"]
    ):
        raise ValueError("Independent numeric verifier rejected submission")
    if (
        report["adapter_sha256"]
        != arms["rl-exposure"]["receipt"]["final_adapter"]["weights_sha256"]
    ):
        raise ValueError("Verifier did not grade the exact submitted RL numeric adapter")
    # Accuracy may legitimately be zero; structural completion is the smoke gate.
    return {"scope": "native TRAIN integration only", "training": evidence, "verifier": report}


def require_local_launch(plan, *, environ=None, platform=None, cwd=None):
    """Checked before any hash, Docker, Harbor or filesystem write. Refuses, never warns."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    cwd = Path.cwd() if cwd is None else Path(cwd)
    if (
        plan.get("local_launch_reviewed") is not True
        or plan.get("cgroup_policy_reviewed") is not True
    ):
        raise ValueError("Local plan unreviewed (local_launch_reviewed is not true); no launch")
    if plan["training"]["max_steps"] != 1:
        raise ValueError("The local plan is defined only for max_steps == 1")
    if platform != "darwin":
        raise ValueError("The local driver runs only on the darwin host")
    if environ.get("DOCKER_HOST") != DOCKER_HOST:
        raise ValueError("DOCKER_HOST must be exactly the vision-harbor Colima socket")
    if environ.get("DOCKER_CONTEXT"):
        raise ValueError("DOCKER_CONTEXT must be unset so DOCKER_HOST alone selects the VM")
    if cwd.resolve() != REPOSITORY_ROOT:
        raise ValueError("Run from the repository root")


def verify_pins(plan, repository=REPOSITORY_ROOT):
    """Every pinned local, host, build and frozen-source file must match before launch."""
    for name, expected in plan["support_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError(f"Local plan file changed: {name}")
    for name, expected in plan["host_source_sha256"].items():
        if sha(repository / name) != expected:
            raise ValueError(f"Pinned host source changed: {name}")
    for name, expected in plan["build_files"].items():
        if sha(repository / name) != expected:
            raise ValueError(f"Pinned build file changed: {name}")
    frozen = repository / "runs/aircraft-objective-ablation-v1-plan/source/vision_rl"
    if {p.name: sha(p) for p in frozen.glob("*.py")} != plan["source_sha256"]:
        raise ValueError("Frozen 37-file source package changed")
    contract = json.loads((repository / "docs/aircraft-cpu-experiment-plan.json").read_text())
    if contract["loader"] != plan["loader"] or contract["build_files"] != plan["build_files"]:
        raise ValueError("Reviewed loader/build contract differs from the local plan")
    if sha(HERE / "aircraft_cpu_loader.py") != plan["loader"]["sha256"]:
        raise ValueError("Local loader copy differs from the pinned loader")


WALL_SLACK_SECONDS = 300


def require_wall_nesting(plan):
    """Every outer wall covers the walls it contains plus slack, so the inner guard fires first.

    Local v2 (plan a2cd71df) lost rl-exposure to a 1800 s stage wall; v3 raises the stage walls
    (timing.json), so the Harbor agent timeout and both driver caps must still enclose them.
    """
    local = plan["local"]
    stages = local["stage_wall_seconds"]
    trained = stages["rl-exposure"] + stages["sft-exposure"] + stages["sft-time"]
    if local["agent_timeout_seconds"] < trained + WALL_SLACK_SECONDS:
        raise ValueError("Harbor agent timeout does not cover the three training stage walls")
    if local["probe_wall_seconds"] < stages["sft-resource-probe"] + WALL_SLACK_SECONDS:
        raise ValueError("Probe driver wall does not cover the probe stage wall")
    if local["main_wall_seconds"] < (
        local["agent_timeout_seconds"] + local["verify_wall_seconds"] + WALL_SLACK_SECONDS
    ):
        raise ValueError("Main driver wall does not cover the agent timeout and verifier wall")


def verify_bundle(bundle, plan_sha, plan):
    """The prepared task must have been packaged from these exact local plan bytes."""
    preparation = json.loads((bundle / "preparation.json").read_text())
    if (
        preparation["plan_sha256"] != plan_sha
        or preparation["runtime_pins"]["plan_sha256"] != plan_sha
    ):
        raise ValueError("Prepared bundle was packaged from a different plan.json")
    for task in ("task", "sft-resource-task"):
        environment = bundle / task / "environment"
        compose = json.loads((environment / "docker-compose.yaml").read_text())["services"]["main"]
        if compose.get("labels") != {"vision-rl.harbor-plan": plan_sha}:
            raise ValueError(
                "Compose label is not this plan's sha; owned-container stop would miss"
            )
        if compose.get("memswap_limit") != 13 * GIB or compose.get("network_mode") != "none":
            raise ValueError("Compose memory/network contract differs")
        pins = json.loads((environment / "runtime-pins.json").read_text())
        if pins["plan_sha256"] != plan_sha or pins["runtime_sha256"] != sha(HERE / "runtime.py"):
            raise ValueError("Runtime pins differ from the local plan")
        if sha(environment / "runtime.py") != plan["support_sha256"]["runtime.py"]:
            raise ValueError("Packaged runtime differs")
        contract = json.loads((environment / "contract.json").read_text())
        if contract["max_steps"] != plan["training"]["max_steps"] or contract["max_steps"] != 1:
            raise ValueError("Packaged task does not commit exactly one update")
        identity = plan["model_identity"]
        if (
            contract["model_manifest_sha256"] != identity["manifest_sha256"]
            or contract["model_revision"] != identity["revision"]
        ):
            raise ValueError("Packaged base model identity differs")
        with (bundle / task / "task.toml").open("rb") as stream:
            agent_timeout = tomllib.load(stream)["agent"]["timeout_sec"]
        if agent_timeout != plan["local"]["agent_timeout_seconds"]:
            raise ValueError("Packaged Harbor agent timeout differs from the local plan")
        first = (environment / "Dockerfile").read_text().splitlines()[0].split()
        if first != ["FROM", IMAGE]:
            raise ValueError("Task image is not built from the pinned runtime image tag")
    return json.loads((bundle / "task/environment/contract.json").read_text())


def image_id():
    return subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE], text=True, timeout=30
    ).strip()


# ---- success criteria C1-C4 (dogfood-design.json success_criteria) -------------------


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exactly_one(paths, what):
    paths = [p for p in paths if p.is_file()]
    require(len(paths) == 1, f"Expected exactly one {what}, found {len(paths)}")
    return json.loads(paths[0].read_text())


def loader_ok(row, policy):
    calls = row.get("opens", [])
    require(row.get("status") == "completed", "loader not completed")
    require(row.get("model_directory") == "/opt/pinned-model", "loader model directory differs")
    require(row.get("aliases_restored") is True, "loader aliases not restored")
    require(row.get("environment_restored") is True, "loader environment not restored")
    require({c.get("file") for c in calls} == set(policy["shards"]), "not both pinned shards")
    require(
        all(c == {"file": c["file"], "device": "cpu", "backend": "pread"} for c in calls),
        "loader backend is not pread on cpu",
    )


def guard_ok(row):
    memory = row.get("cgroup_memory", {})
    require(row.get("status") == "completed" and row.get("exit_code") == 0, "guard not completed")
    require(not row.get("stop_reason"), "guard stop_reason present")
    require(row["peak_tree_rss_bytes"] <= 12 * GIB, "peak tree RSS above 12 GiB")
    require(memory.get("memory.max") == 13 * GIB, "cgroup memory.max is not 13 GiB")
    require(memory.get("memory.swap.max") == 0, "cgroup swap.max is not 0")
    require(memory.get("memory.swap.current") == 0, "cgroup swap.current is not 0")


def criterion_c1(job, plan, contract):
    identity = plan["model_identity"]
    loader = exactly_one(
        [p for p in job.rglob("loader.json") if p.parent.parts[-2:] == ("runtime", "rl-exposure")],
        "runtime/rl-exposure/loader.json",
    )
    loader_ok(loader, plan["loader"])
    evidence = json.loads(artifact_file(job, "training-evidence.json").read_text())
    require(
        evidence["model_source"]["manifest_sha256"] == identity["manifest_sha256"],
        "training-evidence model_source manifest differs from the pinned manifest",
    )
    # model_source carries no revision field; the revision is bound by the packaged
    # contract, whose manifest/marker hashes the runtime re-checks before loading.
    require(contract["model_revision"] == identity["revision"], "packaged revision differs")
    require(
        contract["model_manifest_sha256"] == identity["manifest_sha256"],
        "packaged manifest differs",
    )
    return {
        "loader_opens": loader["opens"],
        "model_source": evidence["model_source"]["manifest_sha256"],
    }


def criterion_c2(job, plan):
    evidence = json.loads(artifact_file(job, "training-evidence.json").read_text())
    arm = evidence["arms"]["rl-exposure"]
    receipt = arm["receipt"]
    require(receipt["global_step"] == 1, "rl-exposure global_step is not 1")
    rows = []
    for path in job.rglob("optimizer_audit.jsonl"):
        if path.is_file() and path.parent.parts[-2:] == ("training", "rl"):
            rows.append(path)
    require(
        len(rows) == 1, f"Expected exactly one training/rl/optimizer_audit.jsonl, found {len(rows)}"
    )
    step = [json.loads(line) for line in rows[0].read_text().splitlines() if line.strip()]
    step = [row for row in step if row.get("step") == 1]
    require(len(step) == 1, "Expected exactly one optimizer audit row for step 1")
    row = step[0]
    require(row["optimizer_steps_observed"] == 1, "optimizer_steps_observed is not 1")
    require(
        row["nonzero_gradient_steps"] >= 1,
        "zero gradient at step 1 (uniform-reward group): update committed, learning unproven",
    )
    first = row.get("first_nonzero_update") or {}
    require(first.get("changed_tensors_since_invocation_start", 0) > 0, "no tensor changed")
    # parameter_delta_l2 is written at train end into the arm's optimizer_audit summary,
    # not into the per-step jsonl row.
    summary = arm["optimizer_audit"]
    require(summary.get("parameter_delta_l2", 0) > 0, "final parameter_delta_l2 is not positive")
    require(
        receipt["final_adapter"]["weights_sha256"]
        != plan["model_identity"]["initial_weights_sha256"],
        "final RL adapter equals the warm initial adapter",
    )
    return {
        "nonzero_gradient_steps": row["nonzero_gradient_steps"],
        "changed_tensors": first["changed_tensors_since_invocation_start"],
        "parameter_delta_l2": summary["parameter_delta_l2"],
        "final_adapter_sha256": receipt["final_adapter"]["weights_sha256"],
    }


def criterion_c3(job, plan):
    evidence = json.loads(artifact_file(job, "training-evidence.json").read_text())
    submitted = evidence["arms"]["rl-exposure"]["receipt"]["final_adapter"]["weights_sha256"]
    report = exactly_one(
        [p for p in job.rglob("report.json") if p.parent.name == "verifier"], "verifier/report.json"
    )
    require("error" not in report, "verifier report carries an error")
    require(report.get("scope") == "integration_only", "verifier scope differs")
    require(report.get("submitted_code_executed") is False, "submitted code was executed")
    require(report.get("adapter_sha256") == submitted, "verifier graded a different adapter")
    loader = exactly_one(
        [p for p in job.rglob("loader.json") if p.parent.name == "verifier"], "verifier/loader.json"
    )
    loader_ok(loader, plan["loader"])
    guard = exactly_one(
        [p for p in job.rglob("guard.json") if p.parent.parts[-2:] == ("verifier", "runtime")],
        "verifier/runtime/guard.json",
    )
    guard_ok(guard)
    trials = []
    for path in job.rglob("result.json"):
        if path.is_file():
            value = json.loads(path.read_text())
            if isinstance(value, dict) and "trial_name" in value:
                trials.append(value)
    require(len(trials) == 1, f"Expected exactly one Harbor trial result.json, found {len(trials)}")
    require(trials[0].get("verifier_environment_mode") == "separate", "verifier env not separate")
    require(trials[0].get("exception_info") is None, "Harbor trial recorded an exception")
    return {"adapter_sha256": submitted, "verifier_environment_mode": "separate"}


def criterion_c4(jobs):
    guards = []
    for job in jobs:
        found = [p for p in job.rglob("guard.json") if p.is_file()]
        require(found, f"no guard.json under {job.name}: nothing ran, which is not a pass")
        guards += found
    rows = 0
    for path in guards:
        guard_ok(json.loads(path.read_text()))
        resources = path.parent / "resources.jsonl"
        lines = [json.loads(x) for x in resources.read_text().splitlines() if x.strip()]
        require(lines, f"{resources} has no rows")
        for line in lines:
            memory = line["cgroup_memory"]
            require(memory.get("memory.swap.current") == 0, "cgroup swap.current nonzero in a row")
            require(memory.get("memory.swap.max") == 0, "cgroup swap.max nonzero in a row")
            require(line["tree_rss_bytes"] <= 12 * GIB, "tree RSS above 12 GiB in a row")
        rows += len(lines)
    return {"guards": len(guards), "resource_rows": rows}


def success_criteria(jobs_dir, plan, contract):
    probe, main = jobs_dir / PROBE_JOB, jobs_dir / MAIN_JOB
    checks = {
        "C1_pretrained_model_loaded": lambda: criterion_c1(main, plan, contract),
        "C2_real_optimizer_update": lambda: criterion_c2(main, plan),
        "C3_separate_verifier_scored_submission": lambda: criterion_c3(main, plan),
        "C4_resource_guards_held": lambda: criterion_c4([probe, main]),
    }
    result = {}
    for name, check in checks.items():
        try:
            result[name] = {"pass": True, "detail": check()}
        except Exception as exc:  # Any missing receipt is incomplete, never pass.
            result[name] = {"pass": False, "reason": f"{type(exc).__name__}: {exc}"}
    return result


def evaluate(root, plan, contract, *, plan_sha, image=None, driver_error=None):
    jobs = root / "output" / "jobs"
    criteria = success_criteria(jobs, plan, contract)
    accuracy = None
    reports = [p for p in (jobs / MAIN_JOB).rglob("report.json") if p.parent.name == "verifier"]
    if len(reports) == 1:
        accuracy = json.loads(reports[0].read_text()).get("reward")
    return {
        "scope": plan["scope"],
        "plan_sha256": plan_sha,
        "runtime_image_id": image,
        "driver_error": driver_error,
        "criteria": criteria,
        "pass": driver_error is None and all(c["pass"] for c in criteria.values()),
        "verifier_reward_recorded_not_judged": accuracy,
        "accuracy_is_pass_criterion": False,
        "accuracy_note": (
            "All 7 fixture images are in the warm adapter's train400 SFT set; accuracy is "
            "memorization-contaminated TRAIN data, not a learning or held-out signal."
        ),
        "native_validation_used": False,
        "native_test_used": False,
    }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def execute(root):
    plan = read_plan()
    # Nothing (no hash, Docker, Harbor or directory creation) precedes this gate.
    require_local_launch(plan)
    require_wall_nesting(plan)
    plan_sha = sha(HERE / "plan.json")
    verify_pins(plan)
    contract = verify_bundle(root / "prepared/bundle", plan_sha, plan)
    image = image_id()
    output = root / "output"
    output.mkdir(exist_ok=False)  # Single use: an existing output means no retry.
    if shutil.disk_usage(output).free < GIB:
        raise RuntimeError("Less than 1 GiB free before launch")
    env = dict(os.environ, PYTHONPATH=str(FROZEN_SOURCE))
    jobs = output / "jobs"
    local = plan["local"]
    error = None
    try:
        run_harbor(
            harbor_command(root / "prepared/bundle/sft-resource-task", jobs, PROBE_JOB, probe=True),
            output,
            timeout=local["probe_wall_seconds"],
            env=env,
            plan_sha=plan_sha,
        )
        assert_probe(jobs / PROBE_JOB)
        run_harbor(
            harbor_command(root / "prepared/bundle/task", jobs, MAIN_JOB),
            output,
            timeout=local["main_wall_seconds"],
            env=env,
            plan_sha=plan_sha,
        )
        assert_complete(jobs / MAIN_JOB, plan["training"]["max_steps"])
    except Exception as exc:  # Recorded, never retried; criteria still report what exists.
        error = f"{type(exc).__name__}: {exc}"
    result = evaluate(root, plan, contract, plan_sha=plan_sha, image=image, driver_error=error)
    write_json(output / RESULT_NAME, result)
    return result


def build_context(context, repository=REPOSITORY_ROOT):
    """Copy the exact pinned build files and frozen 37 sources; no Docker call."""
    plan = read_plan()
    if context.exists():
        raise ValueError("Build context must be new")
    frozen = repository / "runs/aircraft-objective-ablation-v1-plan/source/vision_rl"
    sources = {p.name: p for p in frozen.glob("*.py")}
    if {name: sha(p) for name, p in sources.items()} != plan["source_sha256"]:
        raise ValueError("Frozen 37-file source package changed")
    for name, expected in plan["build_files"].items():
        if sha(repository / name) != expected:
            raise ValueError(f"Pinned build file changed: {name}")
    (context / "src/vision_rl").mkdir(parents=True)
    for name in plan["build_files"]:
        shutil.copyfile(repository / name, context / name)
    for name, path in sources.items():
        shutil.copyfile(path, context / "src/vision_rl" / name)
    copied = {str(p.relative_to(context)): sha(p) for p in context.rglob("*") if p.is_file()}
    expected = {
        **plan["build_files"],
        **{"src/vision_rl/" + k: v for k, v in plan["source_sha256"].items()},
    }
    if copied != expected:
        raise ValueError("Build context differs from the pinned file set")
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    context = sub.add_parser("build-context")
    context.add_argument("--context", type=Path, required=True)
    for name in ("execute", "criteria"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if args.command == "check":
        plan = read_plan()
        print(
            json.dumps(
                {
                    "launchable": plan.get("local_launch_reviewed") is True,
                    "plan_sha256": sha(HERE / "plan.json"),
                    "scope": plan["scope"],
                    "model_calls": 0,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "build-context":
        print(build_context(args.context.resolve()))
        return 0
    root = args.root.resolve()
    if args.command == "criteria":  # Read-only re-evaluation of an existing run.
        plan = read_plan()
        contract = json.loads((root / "prepared/bundle/task/environment/contract.json").read_text())
        jobs, error = root / "output/jobs", None
        try:
            assert_probe(jobs / PROBE_JOB)
            assert_complete(jobs / MAIN_JOB, plan["training"]["max_steps"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = evaluate(
            root, plan, contract, plan_sha=sha(HERE / "plan.json"), driver_error=error
        )
    else:
        result = execute(root)
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "driver_error": result["driver_error"],
                "criteria": {k: v["pass"] for k, v in result["criteria"].items()},
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
