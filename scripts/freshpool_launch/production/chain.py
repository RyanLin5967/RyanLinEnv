"""Run the freshpool launcher detached from any session, and record how it ended.

Started by start.py in a new session (no controlling terminal), so closing the terminal,
ending a Claude session or logging out does not stop the study; launch.py itself sequences
every stage and gate. This file only:
  * runs ``launch.py --plan ... --expected-plan-sha256 ... [--resume]`` from the frozen
    worktree, with the plan's frozen environment (start.py builds it);
  * keeps the Mac awake while it runs: ``caffeinate -i -m -s -w <launcher pid>`` (idle and
    disk sleep; system sleep while on AC power). A closed lid on battery still sleeps;
  * forwards SIGTERM/SIGINT/SIGHUP to the launcher, which journals ``launch_interrupted``
    and exits 3 (resumable: the interrupted stage re-runs from scratch on --resume);
  * bounds the whole run at MAX_SECONDS, then sends the launcher SIGTERM (the same
    resumable interrupt) and, if it is still alive KILL_AFTER_SECONDS later, SIGKILL to the
    launcher's own process group (it runs in its own session, so this also takes the gate
    evaluator it may be running). Every stage runs in ITS own session, so after the launcher
    is gone by any route (that SIGKILL, a crash) the stage group the state's
    ``running_stage`` names is stopped too, with launch.py's own reap_orphan_stage (identity
    checked: the recorded leader start time). The state stays running with a stale lock and
    no stage left behind: resumable, and the next owner scan does not see an orphan;
  * writes ``chain.json`` (running, then finished with the launcher's exit code and final
    state) next to its log.
It never restarts the launcher by itself: an interrupted or stopped study waits for a person
(``start.py --resume`` continues a running state; a stopped one never runs again).
"""

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_SECONDS = 72 * 3600  # about 3x the README's development estimate (an estimate, not a fact)
KILL_AFTER_SECONDS = 600  # after the bound's SIGTERM, SIGKILL the launcher's process group


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def load_launch(path):
    """This checkout's launch.py as a module (stdlib only at import; the same file start.py
    passes as --launcher): its reap_orphan_stage is the one implementation of 'stop the
    stage group a dead launcher left'."""
    spec = importlib.util.spec_from_file_location("freshpool_launch_for_chain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kill_launcher_group(launcher):
    try:
        os.killpg(launcher.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        launcher.kill()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--chain-file", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--kill-after-seconds", type=float, default=KILL_AFTER_SECONDS)
    args = parser.parse_args(argv)
    command = [sys.executable, "-B", str(args.launcher), "--plan", str(args.plan),
               "--expected-plan-sha256", args.expected_plan_sha256,
               *(["--resume"] if args.resume else [])]
    started = time.time()
    launch = load_launch(Path(__file__).resolve().parents[1] / "launch.py")
    launcher = subprocess.Popen(command, cwd=args.launcher.parents[2], stdin=subprocess.DEVNULL,
                                start_new_session=True)
    awake = shutil.which("caffeinate")
    keeper = (subprocess.Popen([awake, "-i", "-m", "-s", "-w", str(launcher.pid)],
                               stdin=subprocess.DEVNULL) if awake else None)
    record = {"status": "running", "chain_pid": os.getpid(), "launcher_pid": launcher.pid,
              "caffeinate_pid": keeper.pid if keeper else None, "command": command,
              "started": started, "max_seconds": args.max_seconds, "resume": args.resume}
    write(args.chain_file, record)

    def forward(signum, _frame):
        record.setdefault("signals_forwarded", []).append(signum)
        if launcher.poll() is None:
            launcher.send_signal(signal.SIGTERM)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    bounded = None
    while launcher.poll() is None:
        if bounded is None and time.time() - started > args.max_seconds:
            bounded = time.time()
            record["max_seconds_reached"] = True
            launcher.send_signal(signal.SIGTERM)
        elif bounded is not None and time.time() - bounded > args.kill_after_seconds:
            record["killed_after_bound"] = True
            kill_launcher_group(launcher)
        time.sleep(1)
    code = launcher.wait()
    state_path = args.state_directory / "state.json"
    try:
        left = json.loads(state_path.read_text())
    except (OSError, ValueError):
        left = {}
    if isinstance(left, dict) and left.get("running_stage"):
        # The launcher is gone but its stage may not be: stop it now rather than leave it
        # holding the accelerator with no cap, RSS or owner guard until someone resumes.
        try:
            record["orphan_stage"] = launch.reap_orphan_stage(left, args.state_directory)
        except Exception as exc:  # recorded, never silent
            record["orphan_stage"] = {"outcome": "reap_failed",
                                      "error": f"{type(exc).__name__}: {exc}"}
    if keeper is not None:
        try:
            keeper.wait(timeout=30)
        except subprocess.TimeoutExpired:
            keeper.kill()
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    record.update(status="finished", finished=time.time(), launcher_exit_code=code,
                  study_status=state.get("status"), stop_reason=state.get("stop_reason"),
                  next_action=state.get("next_action"))
    write(args.chain_file, record)
    print(json.dumps(record, default=str), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
