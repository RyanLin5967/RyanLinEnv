"""The single command that launches (or resumes) the freshpool seed-42 study on MPS.

Run it FROM THE FROZEN WORKTREE, a detached checkout of the merged launch commit at
prepare.FROZEN; nothing in the main checkout is checked out, pulled or merged:

  git -C /Users/idide/RyanLinEnv fetch origin
  git -C /Users/idide/RyanLinEnv worktree add --detach \
      /Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex <merged launch commit>
  cd /Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex
  python3 scripts/freshpool_launch/production/start.py            # new launch
  python3 scripts/freshpool_launch/production/start.py --resume   # continue a running state

It returns once the launcher has journalled ``launch_started`` (or ``launch_resumed``), so
every refusal the launcher itself makes (a second accelerator owner, the lock) comes back
here. Steps, each a refusal (exit 2, one JSON line naming the reason) rather than a warning:

 1. this file runs from prepare.FROZEN, a clean worktree (untracked files included) whose
    HEAD is on origin/main: launch.py resolves every repository path against it and checks
    every pin before every stage, so later commits to main cannot change a running study;
 2. quiet mode off (``~/.claude/QUIET`` absent): the fan guard demotes busy Python to
    efficiency cores at nice 19 while it exists (the launcher also waits it out before
    every stage and journals a mid-stage demotion);
 3. free disk: a new launch needs the 5 GiB standing reserve plus STUDY_OUTPUT_BYTES; a
    resume needs what the launcher needs before every stage (reserve + stage headroom);
 4. a new launch only when the state directory does not exist (single use); --resume only
    when it holds a ``running`` state whose lock holder is gone (pid AND start time, so a
    pid reused after a reboot is not mistaken for the launcher);
 5. the plan's bytes hash to plan.sha256;
 6. the G0 manifests' ``images`` link: the manifest rows name ``images/train/<id>.jpg``
    relative to the development directory, and the trainer resolves them against the
    manifest directory, so ``manifests/images`` must point at the development images (made
    if absent, refused if it points anywhere else);
 7. ``launch.py --verify-only`` under the plan's frozen environment: every pin, the
    interpreter link, the import tree, G0 with all receipts, and every template, with no
    model call (its own timeout is a refusal too);
 8. chain.py started in a new session, its output in ``<state>-chain/chain-<time>.log`` and
    its record in ``<state>-chain/chain.json``; then the launcher must journal its start
    within CONFIRM_SECONDS, or the chain's exit is reported as the refusal it was.
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
import prepare  # noqa: E402  (constants only; nothing heavy at import)

GiB = 1024**3
RESERVE_BYTES = 5 * GiB  # launch.py DISK_RESERVE_BYTES
# What the development stages write. Measured (du -sk, 2026-10-17): the largest controlled400
# trainer output, runs/aircraft-controlled400-v1/work/rl with 3 LoRA checkpoints, is 107,084
# KiB. This study writes 16 such outputs (E0: 3 runs + the altered-reference copy; RL 150 x3;
# RL 600 x2, each a copy of its parent extended; SFT-E x3; SFT-T x<=3; EG-SFT) and allows 2
# voided re-runs: 18 x 107,084 KiB = 1.84 GiB, plus journals and assay outputs (KiB to MiB).
TRAINER_OUTPUT_BYTES = 107084 * 1024
STUDY_OUTPUT_BYTES = 18 * TRAINER_OUTPUT_BYTES + 128 * 1024**2
QUIET = Path.home() / ".claude" / "QUIET"
CONFIRM_SECONDS = 900  # G0 verification again inside the launcher, then launch_started
LAUNCH = HERE.parent / "launch.py"


def load_launch():
    """This checkout's launch.py (stdlib only at import), for its lock reader."""
    spec = importlib.util.spec_from_file_location("freshpool_launch_for_start", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
DEVELOPMENT_IMAGES = prepare.DATA_ROOT / "data" / "fgvc-aircraft-2013b" / "development" / "images"


class Refused(Exception):
    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(directory, *args, check=True):
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True,
                          timeout=120, check=check).stdout.strip()


def check_quiet(quiet=QUIET):
    if quiet.exists():
        raise Refused("quiet_mode_on", file=str(quiet),
                      why="the fan guard demotes busy Python while it exists; only Ryan "
                          "turns quiet mode off")


def required_disk(plan, resume):
    """A new launch: reserve + the whole study's output. A resume: exactly what the launcher
    requires before each stage (plan reserve + stage headroom), so the two never disagree."""
    limits = json.loads(Path(plan).read_text())["limits"]
    if resume:
        return limits["minimum_free_bytes"] + limits["stage_headroom_bytes"]
    return max(RESERVE_BYTES, limits["minimum_free_bytes"]) + STUDY_OUTPUT_BYTES


def check_disk(path, required):
    free = shutil.disk_usage(path).free
    if free < required:
        raise Refused("disk_below_reserve_plus_study_output", free_bytes=free,
                      required_bytes=required)
    return free


def check_state(state_directory, resume):
    state_path = state_directory / "state.json"
    if not resume:
        if state_directory.exists():
            raise Refused("state_directory_exists_single_use", state=str(state_directory))
        return None
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        raise Refused("no_state_to_resume", state=str(state_path)) from None
    if state.get("status") != "running":
        raise Refused("state_is_terminal", status=state.get("status"),
                      stop_reason=state.get("stop_reason"))
    lock = state_directory / "lock"
    if lock.exists():
        launch = load_launch()
        try:
            holder, started = launch.read_lock(lock)
        except (OSError, ValueError):
            # Empty (the launcher is between creating and writing it) or corrupt: ask.
            raise Refused("lock_unreadable", lock=str(lock)) from None
        if launch.lock_holder_alive(holder, started):
            raise Refused("launcher_still_running", pid=holder, started=started)
        # Dead, or the pid was reused (after a reboot): the launcher removes it and
        # journals stale_lock_removed.
    return state


def frozen_worktree(here=HERE, frozen=prepare.FROZEN):
    """start.py must run from the frozen worktree itself: a clean (untracked files
    included) detached checkout whose HEAD is on origin/main. Returns that commit."""
    tree = here.parents[2].resolve()
    if tree != Path(frozen).resolve():
        raise Refused("start_not_run_from_frozen_worktree", running_from=str(tree),
                      frozen=str(frozen),
                      fix=f"git -C {prepare.DATA_ROOT} worktree add --detach {frozen} "
                          "<merged launch commit>, then run its start.py")
    commit = git(tree, "rev-parse", "HEAD")
    dirty = git(tree, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise Refused("frozen_worktree_modified", path=str(tree), changes=dirty.splitlines()[:10])
    merged = subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", commit,
                             "origin/main"], capture_output=True, text=True, timeout=120)
    if merged.returncode != 0:
        raise Refused("frozen_commit_not_on_origin_main", commit=commit,
                      fix="fetch origin, and freeze a commit merged into main")
    return commit


def plan_identity(frozen):
    plan = frozen / prepare.PLAN
    expected = (frozen / prepare.PLAN_SHA).read_text().strip()
    if sha(plan) != expected:
        raise Refused("plan_differs_from_plan_sha256", plan=str(plan))
    return plan, expected


def images_link(manifests=prepare.MANIFESTS, target=DEVELOPMENT_IMAGES):
    link = manifests / "images"
    if link.is_symlink():
        if Path(os.readlink(link)) != target:
            raise Refused("manifest_images_link_elsewhere", link=str(link),
                          target=os.readlink(link))
        return "present"
    if link.exists():
        raise Refused("manifest_images_not_a_link", path=str(link))
    if not target.is_dir():
        raise Refused("development_images_missing", path=str(target))
    link.symlink_to(target, target_is_directory=True)
    return "created"


def environment(plan):
    base = {"HOME": str(Path.home()), "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
    for name in ("USER", "LOGNAME", "TMPDIR", "LANG"):
        if name in os.environ:
            base[name] = os.environ[name]
    return {**base, **json.loads(Path(plan).read_text())["environment"]}


def verify(frozen, plan, plan_sha, env, timeout=600):
    python = json.loads(Path(plan).read_text())["python"]
    try:
        result = subprocess.run(
            [python, "-B", str(frozen / "scripts" / "freshpool_launch" / "launch.py"),
             "--plan", str(plan), "--expected-plan-sha256", plan_sha, "--verify-only"],
            cwd=frozen, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise Refused("verify_only_timed_out", seconds=timeout,
                      stdout=str(exc.stdout or "")[-2000:]) from None
    try:
        report = json.loads(result.stdout)
    except ValueError:
        lines = result.stdout.strip().splitlines()
        report = json.loads(lines[-1]) if lines else {}
    if result.returncode != 0 or report.get("status") != "verified":
        raise Refused("verify_only_failed", launcher=report, stderr=result.stderr[-2000:])
    return report


def start_chain(frozen, plan, plan_sha, env, state_directory, resume):
    chain_dir = Path(str(state_directory) + "-chain")
    chain_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log = chain_dir / f"chain-{stamp}.log"
    python = json.loads(Path(plan).read_text())["python"]
    command = [python, "-B", str(frozen / "scripts" / "freshpool_launch" / "production"
                                 / "chain.py"),
               "--launcher", str(frozen / "scripts" / "freshpool_launch" / "launch.py"),
               "--plan", str(plan), "--expected-plan-sha256", plan_sha,
               "--state-directory", str(state_directory),
               "--chain-file", str(chain_dir / "chain.json"),
               *(["--resume"] if resume else [])]
    with log.open("x") as handle:
        process = subprocess.Popen(command, cwd=frozen, env=env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=handle, start_new_session=True)
    return process, log, chain_dir


def confirm_started(process, state_directory, log, since, resume, seconds=CONFIRM_SECONDS):
    """'launched' only once the launcher journalled its own start after ``since``; a chain
    that exits first is reported with its log's last lines (the launcher's refusal)."""
    wanted = "launch_resumed" if resume else "launch_started"
    journal = state_directory / "journal.jsonl"
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
    raise Refused("launcher_start_not_confirmed", seconds=seconds, chain_pid=process.pid,
                  chain_log=str(log), note="the chain is still running; read status.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        commit = frozen_worktree()
        check_quiet()
        plan, plan_sha = plan_identity(prepare.FROZEN)
        free = check_disk(prepare.DATA_ROOT, required_disk(plan, args.resume))
        check_state(prepare.STATE, args.resume)
        link = images_link()
        env = environment(plan)
        report = verify(prepare.FROZEN, plan, plan_sha, env)
        since = time.time()
        process, log, chain_dir = start_chain(prepare.FROZEN, plan, plan_sha, env,
                                              prepare.STATE, args.resume)
        confirmed = confirm_started(process, prepare.STATE, log, since, args.resume)
    except Refused as refused:
        print(json.dumps({"status": "refused", "reason": refused.reason, **refused.detail},
                         default=str))
        return 2
    print(json.dumps({
        "status": "resumed" if args.resume else "launched", "confirmed_by": confirmed,
        "chain_pid": process.pid,
        "commit": commit, "frozen_worktree": str(prepare.FROZEN), "plan_sha256": plan_sha,
        "free_disk_bytes": free, "images_link": link, "g0": report.get("g0"),
        "d_req": report.get("d_req"), "state_directory": str(prepare.STATE),
        "chain_log": str(log), "chain_record": str(chain_dir / "chain.json"),
        "monitor": f"python3 {prepare.FROZEN}/scripts/freshpool_launch/production/status.py",
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
