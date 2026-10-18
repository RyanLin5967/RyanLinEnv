"""The single command that launches (or resumes) the freshpool seed-42 study on MPS.

  python3 scripts/freshpool_launch/production/start.py            # new launch
  python3 scripts/freshpool_launch/production/start.py --resume   # continue a running state

Run it from the main checkout after the launch PR is merged and pulled. It returns within
about a minute; the study then runs detached (chain.py) for about a day. Steps, each a refusal
(exit 2, one JSON line naming the reason) rather than a warning:

 1. quiet mode off (``~/.claude/QUIET`` absent): the fan guard demotes busy Python to
    efficiency cores at nice 19 while it exists, which slows every stage against its fixed
    1.25x wall-time cap;
 2. free disk at least the 5 GiB standing reserve plus STUDY_OUTPUT_BYTES;
 3. a new launch only when the state directory does not exist (single use); --resume only
    when it holds a ``running`` state whose lock holder is gone;
 4. the frozen worktree: a new launch makes a detached worktree of this checkout's HEAD at
    prepare.FROZEN (or reuses one already at that commit and clean); --resume uses the
    existing one as it is. launch.py resolves every repository path in the plan against it,
    so later commits to main cannot change a running study's code or receipts;
 5. the plan's bytes hash to plan.sha256 (both from the frozen commit);
 6. the G0 manifests' ``images`` link: the manifest rows name ``images/train/<id>.jpg``
    relative to the development directory, and the trainer resolves them against the
    manifest directory, so ``manifests/images`` must point at the development images (made
    if absent, refused if it points anywhere else);
 7. ``launch.py --verify-only`` under the plan's frozen environment: every pin, G0 with all
    receipts, and every template, with no model call;
 8. chain.py started in a new session (start_new_session), its output in
    ``<state>-chain/chain-<time>.log`` and its record in ``<state>-chain/chain.json``.
"""

import argparse
import hashlib
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
# Upper estimate of what the development stages write (trainer outputs with at most 3
# LoRA checkpoints each, assay chunk outputs, journals), unmeasured on this study: ~0.5 GB
# by the controlled400 run directories' size, doubled.
STUDY_OUTPUT_BYTES = 2 * GiB
QUIET = Path.home() / ".claude" / "QUIET"
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


def check_disk(path, required=RESERVE_BYTES + STUDY_OUTPUT_BYTES):
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
        try:
            holder = int(lock.read_text().strip())
            os.kill(holder, 0)
        except (ValueError, ProcessLookupError):
            pass
        except PermissionError:
            raise Refused("launcher_still_running", lock=str(lock)) from None
        else:
            raise Refused("launcher_still_running", pid=holder)
    return state


def frozen_worktree(repository, frozen, resume):
    """The detached worktree the launch runs from: at this checkout's HEAD for a new launch,
    as it is for a resume; clean either way."""
    if not frozen.exists():
        if resume:
            raise Refused("frozen_worktree_missing", path=str(frozen))
        commit = git(repository, "rev-parse", "HEAD")
        git(repository, "worktree", "add", "--detach", str(frozen), commit)
    commit = git(frozen, "rev-parse", "HEAD")
    if not resume and commit != git(repository, "rev-parse", "HEAD"):
        raise Refused("frozen_worktree_at_another_commit", path=str(frozen), commit=commit,
                      fix=f"no study state exists, so remove it: git worktree remove {frozen}")
    if git(frozen, "status", "--porcelain", "--untracked-files=no"):
        raise Refused("frozen_worktree_modified", path=str(frozen))
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


def verify(frozen, plan, plan_sha, env):
    python = json.loads(Path(plan).read_text())["python"]
    result = subprocess.run(
        [python, "-B", str(frozen / "scripts" / "freshpool_launch" / "launch.py"), "--plan",
         str(plan), "--expected-plan-sha256", plan_sha, "--verify-only"],
        cwd=frozen, env=env, capture_output=True, text=True, timeout=600)
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
    return process.pid, log, chain_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    repository = prepare.REPO
    try:
        check_quiet()
        free = check_disk(prepare.DATA_ROOT)
        check_state(prepare.STATE, args.resume)
        commit = frozen_worktree(repository, prepare.FROZEN, args.resume)
        plan, plan_sha = plan_identity(prepare.FROZEN)
        link = images_link()
        env = environment(plan)
        report = verify(prepare.FROZEN, plan, plan_sha, env)
        pid, log, chain_dir = start_chain(prepare.FROZEN, plan, plan_sha, env, prepare.STATE,
                                          args.resume)
    except Refused as refused:
        print(json.dumps({"status": "refused", "reason": refused.reason, **refused.detail},
                         default=str))
        return 2
    print(json.dumps({
        "status": "resumed" if args.resume else "launched", "chain_pid": pid,
        "commit": commit, "frozen_worktree": str(prepare.FROZEN), "plan_sha256": plan_sha,
        "free_disk_bytes": free, "images_link": link, "g0": report.get("g0"),
        "d_req": report.get("d_req"), "state_directory": str(prepare.STATE),
        "chain_log": str(log), "chain_record": str(chain_dir / "chain.json"),
        "monitor": f"python3 {prepare.FROZEN}/scripts/freshpool_launch/production/status.py",
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
