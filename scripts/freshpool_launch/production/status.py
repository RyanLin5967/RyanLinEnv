"""Read-only status of the running freshpool study: one screen, no side effects.

  python3 scripts/freshpool_launch/production/status.py [--events 15] [--log-lines 20]

Prints the study status and next action, whether the launcher (lock pid) and chain are alive,
how long ago the journal last advanced, the running stage with its elapsed wall seconds against
its 1.25x cap, gates and selections so far, the last journal events, the tail of the running
stage's log and the free disk. Liveness is judged by the journal advancing and the stage log
growing, not by a pid alone: a pid can be alive and stalled.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare  # noqa: E402


def alive(pid):
    try:
        os.kill(int(pid), 0)
    except (ValueError, TypeError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def summary(state_directory, events=15, log_lines=20):
    state = read_json(state_directory / "state.json")
    if state is None:
        return {"status": "no state", "state_directory": str(state_directory)}
    journal_path = state_directory / "journal.jsonl"
    lines = journal_path.read_text().splitlines() if journal_path.is_file() else []
    journal = [json.loads(line) for line in lines if line.strip()]
    lock = state_directory / "lock"
    pid = lock.read_text().strip() if lock.is_file() else None
    chain = read_json(Path(str(state_directory) + "-chain") / "chain.json") or {}
    running = None
    starts = [e for e in journal if e["event"] == "stage_started"]
    if starts:
        last = starts[-1]
        done = [e for e in journal if e["event"] == "stage_finished"
                and e["stage"] == last["stage"] and e["time"] >= last["time"]]
        if not done:
            log = (state_directory / "stages" / last["stage"] / f"instance-{last['instance']}"
                   / "stage.log")
            tail = log.read_text(errors="replace").splitlines()[-log_lines:] if log.is_file() \
                else []
            running = {"stage": last["stage"], "arm": last["arm"], "instance": last["instance"],
                       "elapsed_wall_seconds": round(time.time() - last["time"], 1),
                       "charged_before_start_seconds": last["inherited_seconds"]
                       + last["void_seconds"],
                       "cap_seconds": last["cap_seconds"],
                       "stage_log": str(log),
                       "stage_log_age_seconds": round(time.time() - log.stat().st_mtime, 1)
                       if log.is_file() else None,
                       "stage_log_tail": tail}
    return {
        "status": state.get("status"), "stop_reason": state.get("stop_reason"),
        "next_action": state.get("next_action"),
        "launcher_pid": pid, "launcher_alive": alive(pid) if pid else False,
        "chain": {k: chain.get(k) for k in ("status", "chain_pid", "launcher_exit_code",
                                            "started", "finished")},
        "chain_alive": alive(chain.get("chain_pid")) if chain.get("chain_pid") else False,
        "journal_last_event_age_seconds": round(time.time() - journal[-1]["time"], 1)
        if journal else None,
        "running_stage": running,
        "completed_stages": len(state.get("completed", {})),
        "gates": {name: entry["result"].get("pass") for name, entry in
                  state.get("gates", {}).items()},
        "selections": state.get("selections"),
        "screen_tokenizations": state.get("screen_tokenizations"),
        "void_instances": state.get("void_instances"),
        "free_disk_bytes": shutil.disk_usage(state_directory).free,
        "last_events": [{k: e.get(k) for k in ("event", "stage", "gate", "verdict", "reason",
                                              "time")} for e in journal[-events:]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-directory", type=Path, default=prepare.STATE)
    parser.add_argument("--events", type=int, default=15)
    parser.add_argument("--log-lines", type=int, default=20)
    args = parser.parse_args(argv)
    print(json.dumps(summary(args.state_directory, args.events, args.log_lines), indent=2,
                     default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
