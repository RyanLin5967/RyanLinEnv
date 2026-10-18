"""The single command that launches (or resumes) the signal-limit seed-42 study on UW hosts.

Run it FROM THE FROZEN WORKTREE, a detached checkout of the merged launch commit at
prepare.FROZEN (nothing in the main checkout is checked out, pulled or merged):

  git -C /Users/idide/RyanLinEnv fetch origin
  git -C /Users/idide/RyanLinEnv worktree add --detach \\
      /Users/idide/wt/RyanLinEnv-signal-limit-launch-v1.noindex <merged launch commit>
  cd /Users/idide/wt/RyanLinEnv-signal-limit-launch-v1.noindex
  python3 scripts/freshpool_launch/signal_limit_production/start.py --dry-run  # every check
  python3 scripts/freshpool_launch/signal_limit_production/start.py            # new launch
  python3 scripts/freshpool_launch/signal_limit_production/start.py --resume   # continue

Every stage runs on the UWaterloo hosts (plan ``execution: uwaterloo``); this machine only
sequences, gates and journals, so quiet mode does NOT block a launch here. Steps, each a
refusal (exit 2, one JSON line naming the reason) rather than a warning:

 1. this file runs from prepare.FROZEN, a clean worktree (untracked files included) whose
    HEAD is on origin/main;
 2. root's activation: the decision log names BOTH activated copies' SHA256 (the activation
    record's ``approved_by: root`` is a placeholder until root commits that line);
 3. the plan's bytes hash to plan.sha256, and ``prepare.py --check`` reproduces every
    committed output and the G0 files byte for byte (a pin mismatch anywhere refuses);
 4. free disk here: the 5 GiB reserve plus the fetched study output (new launch), or what
    the launcher needs before each stage (resume);
 5. a new launch only when the state directory does not exist (single use); --resume only
    with a ``running`` state whose lock holder is gone;
 6. every plan host is reachable, provisioned (the plan's uw.python and uw.model_directory
    exist), has AVX2 and, for cuda, the plan's free-VRAM floor (uw_stage_a's own probe);
    at least one must qualify now;
 7. ``signal_limit_launch.py --verify-only`` under the plan's frozen environment: every pin,
    the interpreter link, the import tree, G0, the uw block and every template, no model call;
 8. (not with --dry-run) chain.py started in a new session, and the launcher must journal its
    start within CONFIRM_SECONDS.
``--dry-run`` runs 1-7 and stops: nothing is started, no state is written.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare  # noqa: E402  (constants; nothing heavy at import)

GiB = 1024**3
RESERVE_BYTES = 5 * GiB
# Fetched outputs: 13 Stage A runs + 8 Stage B runs + 2 voided re-runs, each a trainer run
# with checkpoints (<= 107,084 KiB, controlled400 work/rl, du -sk 2026-10-17) plus its
# bundle (the shipped init run: <= 107,084 KiB again), + 1 GiB of assays and journals.
STUDY_OUTPUT_BYTES = 23 * 2 * 107084 * 1024 + GiB
CONFIRM_SECONDS = 1800
# chain.py's bound, then a resumable SIGTERM (UW runs stopped, their instances voided):
# 10 days, above the study's 60-80 h MPS estimate scaled by the measured UW unit costs.
MAX_CHAIN_SECONDS = 10 * 24 * 3600
LAUNCHER = HERE.parent / "signal_limit_launch.py"
CHAIN = HERE.parent / "production" / "chain.py"
DECISION_LOG = "docs/decision-log-2026-10-16-claude.md"


class Refused(Exception):
    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(directory, *args):
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True,
                          timeout=120, check=True).stdout.strip()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_worktree(here=HERE, frozen=prepare.FROZEN):
    tree = here.parents[2].resolve()
    if tree != Path(frozen).resolve():
        raise Refused("start_not_run_from_frozen_worktree", running_from=str(tree),
                      frozen=str(frozen))
    commit = git(tree, "rev-parse", "HEAD")
    dirty = git(tree, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise Refused("frozen_worktree_modified", changes=dirty.splitlines()[:10])
    merged = subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", commit,
                             "origin/main"], capture_output=True, text=True, timeout=120)
    if merged.returncode != 0:
        raise Refused("frozen_commit_not_on_origin_main", commit=commit)
    return commit


def root_activation(repo):
    """Root approves by committing a decision-log line that names both activated copies'
    SHA256. The activation record's approved_by 'root' is only a placeholder."""
    text = (repo / DECISION_LOG).read_text()
    found = {path: sha(repo / path) for path in (prepare.STUDY_ACTIVATED,
                                                prepare.PROTOCOL_ACTIVATED)}
    missing = [path for path, digest in found.items() if digest not in text]
    if missing:
        raise Refused("root_activation_line_missing", decision_log=DECISION_LOG,
                      need_sha256=found, missing=missing)
    return found


def plan_identity(repo):
    plan = repo / prepare.PLAN
    expected = (repo / prepare.PLAN_SHA).read_text().strip()
    if sha(plan) != expected:
        raise Refused("plan_differs_from_plan_sha256", plan=str(plan))
    return plan, expected


def prepare_check(repo, python):
    result = subprocess.run([python, "-B", str(repo / "scripts" / "freshpool_launch"
                                             / "signal_limit_production" / "prepare.py"),
                             "--check"], cwd=repo, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise Refused("prepare_check_failed", stdout=result.stdout[-2000:],
                      stderr=result.stderr[-2000:])


def check_disk(resume, plan):
    limits = json.loads(Path(plan).read_text())["limits"]
    required = (limits["minimum_free_bytes"] + limits["stage_headroom_bytes"] if resume
                else max(RESERVE_BYTES, limits["minimum_free_bytes"]) + STUDY_OUTPUT_BYTES)
    free = shutil.disk_usage(prepare.DATA_ROOT).free
    if free < required:
        raise Refused("disk_below_reserve_plus_study_output", free_bytes=free,
                      required_bytes=required)
    return free


def check_state(resume):
    directory = prepare.STATE
    if not resume:
        if directory.exists():
            raise Refused("state_directory_exists_single_use", state=str(directory))
        return None
    try:
        state = json.loads((directory / "state.json").read_text())
    except (OSError, ValueError):
        raise Refused("no_state_to_resume", state=str(directory)) from None
    if state.get("status") != "running":
        raise Refused("state_is_terminal", status=state.get("status"))
    lock = directory / "lock"
    if lock.exists():
        launch = load("signal_limit_launch_for_start", HERE.parent / "launch.py")
        try:
            holder, started = launch.read_lock(lock)
        except (OSError, ValueError):
            raise Refused("lock_unreadable", lock=str(lock)) from None
        if launch.lock_holder_alive(holder, started):
            raise Refused("launcher_still_running", pid=holder, started=started)
    return state


def probe_hosts(plan):
    """uw_stage_a's own host probe, on the plan's allowlist: {host: qualifies}."""
    sys.path.insert(0, str(HERE.parent))
    import uw_stage_a

    runner = object.__new__(uw_stage_a.UWRunner)
    runner.plan = json.loads(Path(plan).read_text())
    runner.uw = runner.plan["uw"]
    runner.device = runner.uw.get("device", "cpu")
    found = runner.hosts()
    report = {}
    for host in runner.uw["hosts"]:
        if host not in found:
            report[host] = {"reachable": False}
            continue
        load_, nproc, avail, uwruns, local, ready = found[host]
        report[host] = {"reachable": True, "provisioned_and_ready": ready, "load1": load_,
                        "nproc": nproc, "avail_gb": avail, "uw_runs": uwruns,
                        "local_gb": local,
                        "qualifies": ready and avail >= runner.uw["min_free_ram_gb"]
                        and local >= runner.uw["min_local_gb"]
                        and uwruns < runner.uw["per_host"]
                        and nproc - load_ >= runner.uw.get("min_idle_threads", 0)}
    if not any(entry.get("qualifies") for entry in report.values()):
        raise Refused("no_uw_host_qualifies", hosts=report)
    return report


def environment(plan):
    base = {"HOME": str(Path.home()), "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
    for name in ("USER", "LOGNAME", "TMPDIR", "LANG"):
        if name in os.environ:
            base[name] = os.environ[name]
    return {**base, **json.loads(Path(plan).read_text())["environment"]}


def verify(repo, plan, plan_sha, env, timeout=900):
    python = json.loads(Path(plan).read_text())["python"]
    try:
        result = subprocess.run(
            [python, "-B", str(repo / "scripts" / "freshpool_launch" / "signal_limit_launch.py"),
             "--plan", str(plan), "--expected-plan-sha256", plan_sha, "--verify-only"],
            cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Refused("verify_only_timed_out", seconds=timeout) from None
    lines = result.stdout.strip()
    try:
        report = json.loads(lines)
    except ValueError:
        report = json.loads(lines.splitlines()[-1]) if lines else {}
    if result.returncode != 0 or report.get("status") != "verified":
        raise Refused("verify_only_failed", launcher=report, stderr=result.stderr[-2000:])
    return report


def start_chain(repo, plan, plan_sha, env, resume):
    chain_dir = Path(str(prepare.STATE) + "-chain")
    chain_dir.mkdir(parents=True, exist_ok=True)
    log = chain_dir / f"chain-{time.strftime('%Y%m%dT%H%M%S')}.log"
    python = json.loads(Path(plan).read_text())["python"]
    command = [python, "-B", str(CHAIN), "--launcher", str(LAUNCHER), "--plan", str(plan),
               "--expected-plan-sha256", plan_sha, "--state-directory", str(prepare.STATE),
               "--chain-file", str(chain_dir / "chain.json"),
               "--max-seconds", str(MAX_CHAIN_SECONDS), *(["--resume"] if resume else [])]
    with log.open("x") as handle:
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=handle, start_new_session=True)
    return process, log


def confirm_started(process, log, since, resume, seconds=CONFIRM_SECONDS):
    wanted = "launch_resumed" if resume else "launch_started"
    journal = prepare.STATE / "journal.jsonl"
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            events = [json.loads(line) for line in journal.read_text().splitlines()
                      if line.strip()]
        except (OSError, ValueError):
            events = []
        if any(e.get("event") == wanted and e.get("time", 0) >= since for e in events):
            return wanted
        if process.poll() is not None:
            tail = log.read_text(errors="replace").splitlines()[-5:] if log.exists() else []
            raise Refused("launcher_exited_before_starting", chain_exit_code=process.returncode,
                          chain_log=str(log), log_tail=tail)
        time.sleep(1)
    raise Refused("launcher_start_not_confirmed", seconds=seconds, chain_pid=process.pid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo = HERE.parents[2]
    try:
        commit = frozen_worktree()
        approved = root_activation(repo)
        plan, plan_sha = plan_identity(repo)
        python = json.loads(plan.read_text())["python"]
        prepare_check(repo, python)
        free = check_disk(args.resume, plan)
        check_state(args.resume)
        hosts = probe_hosts(plan)
        env = environment(plan)
        report = verify(repo, plan, plan_sha, env)
        summary = {"commit": commit, "plan_sha256": plan_sha, "activated_sha256": approved,
                   "free_disk_bytes": free, "hosts": hosts, "a_floor": report.get("a_floor"),
                   "uw_device": report.get("uw_device"), "g0": report.get("g0")}
        if args.dry_run:
            print(json.dumps({"status": "dry_run_passed", **summary}, indent=2, default=str))
            return 0
        since = time.time()
        process, log = start_chain(repo, plan, plan_sha, env, args.resume)
        confirmed = confirm_started(process, log, since, args.resume)
    except Refused as refused:
        print(json.dumps({"status": "refused", "reason": refused.reason, **refused.detail},
                         default=str))
        return 2
    print(json.dumps({"status": "resumed" if args.resume else "launched",
                      "confirmed_by": confirmed, "chain_pid": process.pid, "chain_log": str(log),
                      "state_directory": str(prepare.STATE), **summary,
                      "monitor": f"python3 {repo}/scripts/freshpool_launch/production/status.py "
                                 f"--state-directory {prepare.STATE}"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
