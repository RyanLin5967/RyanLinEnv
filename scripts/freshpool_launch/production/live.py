"""Whether the production freshpool study is live, for code that must not run beside it.

The launcher stops the study for good when a second accelerator owner appears (a protocol
standing stop), and its owner scan reads command lines. Test modules that spawn owner-named
processes (phase_worker.py, launch.py, the named-baseline or local-vision runners) would
trip it, so tests/conftest.py and scripts/freshpool_dryrun/run.py ask here first and refuse
while the production state is ``running`` (including an interrupted run awaiting
``start.py --resume``: the resume's own scan would see them). An unreadable state file in an
existing state directory counts as running: ask, never allow.

Blind spot, stated: a test module is judged by its own source text naming a pattern (or
``phase_worker`` / ``launch.py``); one that spawns an owner through a helper module whose
name it does not mention is not seen here (the launcher's scan still is).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
EXTRA_PATTERNS = ("phase_worker", "launch.py")


def study_running(plan_path=PLAN):
    """The state.json path when the plan's study is running (or unreadable), else None."""
    try:
        plan = json.loads(Path(plan_path).read_text())
    except (OSError, ValueError):
        return None  # no production plan in this checkout: nothing to protect here
    directory = Path(plan["state_directory"])
    state_path = directory / "state.json"
    if not directory.exists():
        return None
    try:
        status = json.loads(state_path.read_text()).get("status")
    except (OSError, ValueError, AttributeError):
        return str(state_path)
    return str(state_path) if status == "running" else None


def patterns(plan_path=PLAN):
    plan = json.loads(Path(plan_path).read_text())
    return (*plan["owner_patterns"], *EXTRA_PATTERNS)


def owner_naming_modules(paths, names):
    """The paths whose source text names any of ``names`` (unreadable ones included)."""
    found = []
    for path in sorted({str(p) for p in paths}):
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            found.append(path)
            continue
        if any(name in text for name in names):
            found.append(path)
    return found
