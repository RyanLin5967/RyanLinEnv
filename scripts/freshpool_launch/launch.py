"""Sequence the aircraft-freshpool-screened-grpo-v1 seed-42 development stages, gate by gate.

Study: docs/aircraft-freshpool-study-v1.json (wins over study.md). Modelled on
runs/aircraft-controlled400-launch-v1/{launch.py,phase_worker.py} and
runs/aircraft-controlled400-continuation-v1/continue.py, which stay frozen records.

Order (study.json "gates" and "compute_plan_mps_hours.seed42_steps"):
  G0 manifests/receipts -> E0 resume proof -> W on H-select and H-gate -> P0 -> G1
  -> RL x grid to 150 + H-select -> G2 -> top two by G2 continue to 600 (no reset) + H-select
  -> RL* (H-select at 600) -> SFT-E x grid on RL*'s accepted sequence + H-select
  -> H-gate RL*, SFT-E* -> G3a -> SFT-T for G3a's qualifying LRs, continuing SFT-E to T_RL
  + H-select -> H-gate SFT-T* -> G3b -> EG-SFT at SFT-E*'s LR + H-gate -> evaluator must
  report no failure and await G4. VAL (G4), replicates (G5) and TEST (G6) belong to the
  native-VAL package, not to this launcher.

Every gate G1-G3b is decided by scripts/evaluate_freshpool_gates.py over the evidence the
stages wrote, and every selection (continuers, RL*, SFT-E*, SFT-T LRs, SFT-T*) is that
evaluator's own. A gate that is missing, not evaluated, unparseable or not passing stops the
study; there is no path from a missing result to a pass.

Guards (all refuse, none warn): plan identity and every pinned file hash before every stage;
frozen environment, with ACCELERATE_TORCH_DEVICE mps (cpu only under the plan's explicit
``"dry_run": true`` marker); a fixed 5 GiB disk reserve the plan may raise but not lower;
process-group RSS; a second owner of the accelerator, a refusal before a launch or resume and a
stop before every stage and every OWNER_POLL_SECONDS while one runs; the interpreter link and
the import tree re-checked with the pins; quiet mode waited out and too little disk headroom a
resumable pause, both before a stage starts; per-arm 1.25x forecast caps (RL and EG-SFT from
P0's measured timing, frozen after G1), counted from the validated
parent's committed seconds for a continuation plus the wall seconds of every voided instance of
the arm, cumulatively, so neither a continuation nor an interrupt resets them; the rolling
useful-mixed standing stop (< 40 of any 100 accepted updates), enforced live on RL stages by
tailing reward_groups.jsonl under the arm directory and again on the fragment; SFT-E must end
boundary_reached at stop_after_steps 600 so SFT-T can continue it (the sft_e and RL templates
are checked at plan load, before any stage runs); a survivor-grace check on
every stage's process group. State is single-use: a stopped or completed study never runs
again. An interrupted stage is NEVER resumed or quoted (study.json budget_matching.caps):
``--resume`` journals its instance as void and re-runs the stage from scratch in a fresh
instance directory, from the stage's own declared start (W, or the validated parent
endpoint). A capped arm is a standing stop. Any unexpected exception is converted to a
journalled study stop. The journal is append-only and fsynced.

Blind spots, stated: process-group membership is the ownership boundary, so a descendant that
calls setsid() escapes the RSS sum and the survivor check; RSS, disk and the owner scan are
sampled; the live useful-mixed tail reads only a reward_groups.jsonl under the arm
directory (a stage writing it elsewhere stops as unmonitored) and sees rows at the poll
interval; the owner scan matches whole command-line words, so an owner whose command line
names none of the plan's patterns as a word is invisible; the lock and the orphan reap identify
a process by pid plus ps start time; the per-arm cap is measured on launcher wall time (an
upper bound on active time) and
the trainer receives the same cap as its cumulative active-seconds limit; the owner scan reads
only processes whose program is a Python interpreter, so a non-Python accelerator owner is not
seen (a grep or editor whose command line names a pattern is not an owner); the fragment
contents (predictions, adapter hashes, active seconds) are trusted as the stage wrote them,
apart from the lineage checks below (cumulative seconds through the parent, accepted-prefix
continuation, SFT-T stopping at the first step at or above T_RL).
"""

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GiB = 1024**3
DISK_RESERVE_BYTES = 5 * GiB  # study.json standing_stops: addendum-5 5 GiB reserve
CAP_FACTOR = 1.25  # study.json budget_matching.caps
OWNER_POLL_SECONDS = 2.0  # second-owner re-check interval while a stage runs (standing stop)
QUIET_POLL_SECONDS = 30.0  # re-check interval while waiting out quiet mode between stages
SEED = 42
GRID = (1e-5, 5e-5, 1e-4)
B4_LR = 2e-4
RL_SCREEN_UPDATES, FINAL_UPDATES = 150, 600
SPLITS = ("h_select", "h_gate")
COMPLETIONS = {"h_select": 5, "h_gate": 9}  # greedy + 1 G4 seed; greedy + 2 G4 seeds
H_PER_CLASS = 2  # study.json data_manifests_to_create: H_select / H_gate per_class
G1_CANDIDATES, G1_ROLLOUTS = 400, 128  # 64 images x seeds 42, 314
HAMMING_MAX, FORCED_FIRE_PAIRS = 6, 50  # study.json phash_audit forced_fire
PROTOCOL_ID = "fgvc-aircraft-freshpool-screened-grpo-v1"
FROZEN_PROTOCOL_KEYS = ("substantial_gain_gates", "strong_baseline")
# The only top-level keys an activated copy may change relative to the revision it activates
# (protocol activation_prerequisites: "freezes a separately identified activated protocol").
# root_review_status: the revision's says "no study activation ... authorized by this
# document", which is true of the revision and false of the activated copy.
ACTIVATION_KEYS = ("protocol_id", "status", "activated", "activation", "root_review_status")
# study.json fixed_profile.screening.alt_tokenization_switch: "if >2% of correct P0 samples
# are non-canonical tokenizations, sum p over P0-observed tokenizations; frozen before G2".
ALT_TOKENIZATION_SHARE = 0.02
TOKENIZED_TEMPLATES = ("rl150", "rl600", "egsft")
TRAIN_TEMPLATES = ("e0", "rl150", "rl600", "sft_e", "sft_t", "egsft")
TEMPLATES = (*TRAIN_TEMPLATES, "assay", "p0")
FORECASTS = ("e0", "p0", "assay_h_select", "assay_h_gate", "rl", "sft_e", "egsft")
REQUIRED_MANIFESTS = ("pool.jsonl", "h_select.jsonl", "h_gate.jsonl", "s_scan_seed42.jsonl",
                      "phash_audit.json")
POOL_BEFORE_AUDIT = 2919  # study.json data_manifests_to_create pool.expected_images
MIXED_WINDOW, MIXED_MIN_IN_WINDOW = 100, 40  # standing stop: < 0.40 over any 100 accepted
# G0 '16-shot plan retired': study.json protocol_revision.same_commit names the plan
# "the unactivated 16-shot plan (seed 2718, 1,600 images)". A decision-record line passes
# only if it STARTS with an explicit "Retired:" status (list/bold markup allowed), names
# that plan by all three facts, and carries no hedge.
RETIRED_LINE = re.compile(r"^\s*(?:[-*]\s+)?(?:\*\*)?Retired:", re.I)
RETIRED_FACTS = (re.compile(r"16-shot", re.I), re.compile(r"\bseed 2718\b"),
                 re.compile(r"\b1,?600 images\b"))
RETIRED_HEDGE = re.compile(
    r"\b(?:not|no longer|pending|proposed|planned|reverted|undone|unretired|reinstated)\b",
    re.I)
RECEIPTS = ("protocol_revision", "decision_record", "branch")
HEX64 = re.compile(r"[0-9a-f]{64}")
GATE_ORDER = ("G1", "G2", "G3a", "G3b")
# scripts/freshpool_launch/e0_gate.py CHECKS, written out (the launcher does not import it).
E0_CHECKS = (
    "resume_2_plus_2_identical", "resume_1_plus_3_identical", "scan_follows_s_scan_seed42",
    "initial_adapter_is_w", "reference_unchanged", "altered_reference_checkpoint_refused",
    "useful_mixed_group", "finite_nonzero_task_gradient", "lora_parameter_changed",
    "scorer_uses_sampler_transform", "active_time_ledger_cumulative",
    "scorer_seconds_within_active_time",
)


class Stop(Exception):
    """A study stop: recorded, terminal, never retried."""

    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class Pause(Exception):
    """A resumable halt BEFORE a stage starts (nothing of the stage ran, nothing is voided):
    the state stays ``running`` and ``start.py --resume`` continues once a person fixed the
    cause. Only conditions that are not a protocol standing stop may pause."""

    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def jsonl_ids(path):
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError):
        return None
    if not all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows):
        return None
    return [row["id"] for row in rows]


def free_bytes(path):
    return shutil.disk_usage(path).free


def lr_tag(lr):
    return f"{lr:.0e}"


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def processes():
    """Rows of (pid, ppid, pgid, stat, rss_bytes, command, nice) from ps; no psutil needed.

    -ww: piped procps output is otherwise cut at 80 columns, which hid an owner pattern
    past that width (measured in CI on test_second_owner_refuses). ``nice`` is None when ps
    prints something other than an integer.
    """
    result = subprocess.run(
        ["ps", "-A", "-ww", "-o", "pid=,ppid=,pgid=,stat=,rss=,nice=,command="],
        capture_output=True, text=True, timeout=30, check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 6:
            continue
        try:
            pid, ppid, pgid, rss = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[4])
        except ValueError:
            continue
        try:
            nice = int(parts[5])
        except ValueError:
            nice = None
        rows.append((pid, ppid, pgid, parts[3], rss * 1024,
                     parts[6] if len(parts) > 6 else "", nice))
    return rows


def process_started(pid):
    """The process's start time as ps prints it (``lstart``), or None if it is gone. A pid
    plus its start time names one process; a pid alone can be reused (after a reboot, say)."""
    try:
        result = subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))],
                                capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    started = result.stdout.strip()
    return started or None


def kill_group(pgid, grace=15):
    """SIGTERM a process group, then SIGKILL whatever is left; True if it is gone."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        except PermissionError:
            if not group_members(pgid):
                return True
            return False
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not group_alive(pgid):
                return True
            time.sleep(0.1)
    return not group_alive(pgid)


def reap_orphan_stage(state, directory):
    """Stop the stage process group a dead launcher left running, if it is still ours.

    The launcher records ``running_stage`` (pgid, its leader's start time) while a stage
    runs, and every stage runs in its own session, so killing the launcher alone (chain.py's
    SIGKILL after its bound, a crash, kill -9) leaves the stage running unsupervised: no
    cap, RSS or owner check, still holding the accelerator, and seen by the next launch's
    owner scan as a second owner. Identity: the group's leader is alive with the recorded
    start time, or the leader is gone and every live member's command names this state
    directory (phase_worker.py's --guard-dir). A reused pgid matches neither and is left
    alone. Returns a journal record (None when no stage was recorded)."""
    running = state.get("running_stage")
    if not isinstance(running, dict):
        return None
    pgid = running.get("pgid")
    record = {"stage": running.get("stage"), "instance": running.get("instance"),
              "pgid": pgid}
    if type(pgid) is not int or pgid <= 1:
        return {**record, "outcome": "invalid_record"}
    members = group_members(pgid)
    if not members:
        return {**record, "outcome": "already_gone"}
    leader = process_started(pgid) if any(row[0] == pgid for row in members) else None
    ours = (leader is not None and leader == running.get("leader_started")) or (
        leader is None and all(str(directory) in row[5] for row in members))
    if not ours:
        return {**record, "outcome": "group_not_ours_left_alone",
                "processes": [{"pid": r[0], "command": r[5][:200]} for r in members]}
    stopped = kill_group(pgid)
    return {**record, "outcome": "stopped" if stopped else "still_alive",
            "processes": [{"pid": r[0], "command": r[5][:200]} for r in members]}


def group_members(pgid):
    """Live (non-zombie) members of a process group."""
    return [row for row in processes() if row[2] == pgid and not row[3].startswith("Z")]


def group_alive(pgid):
    """macOS answers killpg with EPERM (not ESRCH) for a group whose only members are
    zombies, so EPERM is settled by the member list, never assumed alive or dead."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return bool(group_members(pgid))


def stop_group(child, pgid, grace=15):
    """SIGTERM the stage's process group, then SIGKILL whatever is left."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            # EPERM for an all-zombie group (macOS): nothing is left to signal. A live
            # member that refuses the signal is not ours to ignore: re-raise.
            if not group_members(pgid):
                break
            raise
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if child.poll() is not None and not group_alive(pgid):
                break
            time.sleep(0.1)
    child.wait()


TOKEN_SPLIT = re.compile(r"[\s;,()'\"=]+")


def names_pattern(command, pattern):
    """True when one word of the command line IS the pattern, not merely contains it:
    a ``.py`` pattern matches a word equal to it or ending in ``/<pattern>``; a module
    pattern (``vision_rl.vlm_local_training``, ``mlx_vlm``) matches a word equal to it,
    a submodule of it (``mlx_vlm.generate``) or a path through it (``.../mlx_vlm/x.py``).
    So ``pytest tests/test_local_vision.py`` does not name ``local_vision.py``, and
    ``tests/test_vlm_local_training.py`` does not name ``vision_rl.vlm_local_training``."""
    for word in TOKEN_SPLIT.split(command):
        if not word:
            continue
        if pattern.endswith(".py"):
            if word == pattern or word.endswith("/" + pattern):
                return True
        elif (word == pattern or word.startswith(pattern + ".")
              or f"/{pattern}/" in word or word.startswith(pattern + "/")):
            return True
    return False


def dry_run_worker(command):
    """A phase_worker.py started with --dry-run (before its literal ``--``). phase_worker.py
    then refuses to start unless ACCELERATE_TORCH_DEVICE is cpu and every forwarded --device
    is cpu, before any import, so it and its descendants cannot own the accelerator."""
    words = command.split()
    for n, word in enumerate(words):
        if word == "phase_worker.py" or word.endswith("/phase_worker.py"):
            rest = words[n + 1:]
            head = rest[: rest.index("--")] if "--" in rest else rest
            return "--dry-run" in head
    return False


def descendants(rows, roots):
    children = {}
    for pid, ppid, *_ in rows:
        children.setdefault(ppid, []).append(pid)
    found, stack = set(roots), list(roots)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def second_owners(patterns, own_groups=()):
    """Python processes naming an owner pattern (as a whole word, names_pattern) other than
    this launcher, its ancestors, and its running stage: the members of ``own_groups`` and
    their descendants. Any other child of the launcher's own process is NOT exempt (a real
    launcher has none while a stage runs). A CPU dry-run phase worker (dry_run_worker) and
    its descendants are not owners: the freshpool dry run and its tests stay runnable."""
    rows = processes()
    parents = {pid: ppid for pid, ppid, *_ in rows}
    mine, pid = set(), os.getpid()
    while pid and pid not in mine:
        mine.add(pid)
        pid = parents.get(pid)
    mine |= descendants(rows, [pid for pid, _, pgid, *_ in rows if pgid in own_groups])
    mine |= descendants(rows, [row[0] for row in rows if dry_run_worker(row[5])])
    return [{"pid": pid, "command": command[:200]}
            for pid, _, _, stat, _, command, *_ in rows
            if pid not in mine and not stat.startswith("Z") and python_program(command)
            and any(names_pattern(command, p) for p in patterns)]


def python_program(command):
    """True when the command's program (argv[0]) is a Python interpreter: python, python3.12,
    or the framework binary Python. A shell, grep or editor whose arguments merely name an
    owner pattern is not an accelerator owner (a substring match on every command line would
    let `rg vision_rl.vlm_local_training` stop a 30-hour study)."""
    # argv[0] may contain spaces (".../Application Support/.../python3"), so try every
    # prefix that ends at whitespace and holds no option (" -"); prefer refusing a non-owner
    # whose arguments end in a python path over missing an owner.
    words = command.split(" ")
    for end in range(1, len(words) + 1):
        prefix = " ".join(words[:end])
        if " -" in prefix:
            break
        if re.fullmatch(r"pythonw?[0-9.]*(?:-[A-Za-z0-9_]+)?", Path(prefix).name, re.I):
            return True
    return False


def git_committed(path):
    """True only if ``path`` is tracked by its enclosing git repository with exactly these
    bytes in the index and in HEAD."""
    path = Path(path).resolve()
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(path.parent), *args], capture_output=True,
                                  text=True, timeout=60, check=True).stdout.strip()
        blob = git("hash-object", "--", path.name)
        head = git("rev-parse", f"HEAD:./{path.name}")
        staged = git("ls-files", "-s", "--", path.name).split()
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(blob) and blob == head and len(staged) >= 2 and staged[1] == blob


# ---------------------------------------------------------------- plan and G0


def load_evaluator(plan):
    return runpy.run_path(str(resolve(plan["evaluator"])))


def verify_device(plan):
    """A real launch runs on mps; cpu is representable only in a plan carrying the explicit
    dry-run marker ``"dry_run": true`` (scripts/freshpool_dryrun), and a dry run only on cpu.
    The marker must be a literal bool when present, so no truthy string turns a launch into
    a dry run. The phase worker repeats this check (``--dry-run``) and also refuses a stage
    whose forwarded ``--device`` differs from the device, since the inference entries do not
    go through Accelerate."""
    marker = plan.get("dry_run", False)
    if type(marker) is not bool:
        raise Stop("dry_run_marker_not_bool", dry_run=marker)
    device = (plan.get("environment") or {}).get("ACCELERATE_TORCH_DEVICE")
    if device != ("cpu" if marker else "mps"):
        raise Stop("plan_device_not_allowed", device=device, dry_run=marker)


def verify_plan(plan_path, expected):
    if sha(plan_path) != expected:
        raise Stop("plan_identity_differs")
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != "freshpool-launch-plan-v1":
        raise Stop("plan_schema_unknown")
    pinned = plan.get("file_sha256") or {}
    for required in (plan.get("evaluator"), str(HERE / "phase_worker.py")):
        if required is None or not any(resolve(p) == resolve(required) for p in pinned):
            raise Stop("plan_does_not_pin", file=required)
    verify_pins(plan)
    for name, value in (plan.get("environment") or {}).items():
        if os.environ.get(name) != value:
            raise Stop("frozen_environment_differs", variable=name)
    verify_device(plan)
    limits = plan.get("limits") or {}
    reserve = limits.get("minimum_free_bytes")
    if type(reserve) is not int or reserve < DISK_RESERVE_BYTES:
        raise Stop("disk_reserve_below_5_GiB_or_missing", minimum_free_bytes=reserve)
    for name in ("rss_limit_bytes", "survivor_grace_seconds", "poll_seconds"):
        value = limits.get(name)
        if not is_number(value) or value <= 0:
            raise Stop("limit_missing_or_invalid", limit=name)
    forecasts = plan.get("forecast_seconds") or {}
    for name in FORECASTS:
        value = forecasts.get(name)
        if not is_number(value) or value <= 0:
            raise Stop("forecast_missing_or_invalid", forecast=name)
    templates = plan.get("templates") or {}
    for name in TEMPLATES:
        template = templates.get(name)
        if (
            not isinstance(template, dict)
            or template.get("kind") not in ("train", "assay", "common")
            or not isinstance(template.get("argv"), list)
            or not HEX64.fullmatch(str(template.get("entry_sha256")))
            or ":" not in str(template.get("entry"))
        ):
            raise Stop("template_missing_or_invalid", template=name)
        if "resume_argv" in template:
            # study.json budget_matching.caps: an interrupted arm is re-run from scratch,
            # never resumed and quoted, so a plan cannot even declare a resume command.
            raise Stop("template_declares_resume_argv", template=name)
    verify_stage_templates(templates)
    if type(plan.get("b4_edge_extension")) is not bool:
        raise Stop("b4_edge_extension_not_stated")
    grid = sorted([*GRID, *([B4_LR] if plan["b4_edge_extension"] else [])])
    try:
        evaluator = load_evaluator(plan)
    except Exception as exc:
        raise Stop("evaluator_unloadable", error=f"{type(exc).__name__}: {exc}") from None
    expected_grid = sorted([*evaluator["GRID"], *(
        [evaluator["B4_LR"]] if plan["b4_edge_extension"] else [])])
    if grid != expected_grid:
        raise Stop("grid_differs_from_evaluator", grid=grid, evaluator=expected_grid)
    if evaluator["COMPLETIONS"] != COMPLETIONS:
        raise Stop("completions_differ_from_evaluator", evaluator=evaluator["COMPLETIONS"])
    w = plan.get("w_endpoint") or {}
    if not HEX64.fullmatch(str(w.get("adapter_sha256"))) or not w.get("adapter_path"):
        raise Stop("w_endpoint_identity_missing")
    if not isinstance(plan.get("owner_patterns"), list) or not plan["owner_patterns"]:
        raise Stop("owner_patterns_missing")
    sources = plan.get("forecast_sources") or {}
    for name in ("grpo_seconds_per_update", "sft_seconds_per_update"):
        if not is_number(sources.get(name)) or sources[name] <= 0:
            raise Stop("forecast_source_missing_or_invalid", source=name)
    if not isinstance(plan.get("quiet_file"), str) or not plan["quiet_file"]:
        raise Stop("quiet_file_not_stated")
    headroom = limits.get("stage_headroom_bytes")
    if type(headroom) is not int or headroom < 0:
        raise Stop("limit_missing_or_invalid", limit="stage_headroom_bytes")
    return plan, grid, evaluator


def flag_values(argv, flag):
    """Every value given for ``flag`` in ``argv``, as ``flag value`` or ``flag=value``."""
    values = []
    for n, part in enumerate(argv):
        if part == flag:
            values.append(argv[n + 1] if n + 1 < len(argv) else None)
        elif isinstance(part, str) and part.startswith(flag + "="):
            values.append(part[len(flag) + 1:])
    return values


def wrapper_and_trainer(argv):
    """Split a training template at the wrapper's literal ``--`` (trainer_fragment.py
    refuses a command without one). None when there is no separator."""
    if "--" not in argv:
        return None
    split = argv.index("--")
    return argv[:split], argv[split + 1:]


def verify_stage_templates(templates):
    """Checked at plan load, before any stage runs:
    - rl150/rl600 give the wrapper exactly one ``--output`` under ``{arm_dir}``, so the live
      useful-mixed tail can see the trainer's reward_groups.jsonl (otherwise the stage would
      run to completion and only then stop as live_useful_mixed_unmonitored);
    - rl600's trainer arguments equal rl150's without its single ``--stop-after-steps 150``,
      and neither names ``{active_cap}``: the 600 continuation resumes the 150 checkpoint,
      and the trainer refuses a resume whose settings differ (max_active_seconds included).
      ``{active_cap}`` shrinks after a voided instance, so RL passes the constant
      ``{arm_cap}``; the launcher's wall cap still charges the voided seconds;
    - rl150, rl600 and egsft pass ``--screen-tokenizations {screen_tokenizations}``, the
      alt-tokenization file the launcher freezes after G1 (empty unless the switch fires);
    - sft_e gives the trainer exactly ``--stop-after-steps 600`` and one integer
      ``--max-steps`` above 600, so SFT-E ends boundary_reached and SFT-T can continue it
      (otherwise every SFT-E arm would run and stop only at its fragment check), and
      ``--order sequential --cycle``, without which the trainer refuses that max_steps on
      a 600-row manifest."""
    for name in ("rl150", "rl600", "sft_e"):
        parts = wrapper_and_trainer(templates[name]["argv"])
        if parts is None or not all(isinstance(a, str) for a in templates[name]["argv"]):
            raise Stop("template_missing_wrapper_separator", template=name)
    for name in ("rl150", "rl600"):
        wrapper, _ = wrapper_and_trainer(templates[name]["argv"])
        outputs = flag_values(wrapper, "--output")
        target = outputs[0] if len(outputs) == 1 else None
        if (target is None
                or not (target == "{arm_dir}" or target.startswith("{arm_dir}/"))
                or ".." in target.split("/")):
            raise Stop("template_output_not_under_arm_dir", template=name, output=outputs)
    verify_rl_continuation_templates(templates)
    for name in TOKENIZED_TEMPLATES:
        if flag_values(templates[name]["argv"], "--screen-tokenizations") != [
                "{screen_tokenizations}"]:
            raise Stop("template_screen_tokenizations_missing", template=name)
    _, trainer = wrapper_and_trainer(templates["sft_e"]["argv"])
    stops, maxes = flag_values(trainer, "--stop-after-steps"), flag_values(trainer, "--max-steps")
    if stops != [str(FINAL_UPDATES)]:
        raise Stop("sft_e_template_stop_after_steps_not_600", stop_after_steps=stops)
    if (len(maxes) != 1 or not isinstance(maxes[0], str) or not maxes[0].isdigit()
            or int(maxes[0]) <= FINAL_UPDATES):
        raise Stop("sft_e_template_max_steps_not_above_600", max_steps=maxes)
    # RL*'s accepted manifest holds exactly FINAL_UPDATES rows, and the trainer refuses a
    # sequential run whose max_steps exceeds its manifest unless it cycles
    # (vlm_local_training.train_local), so SFT-E must replay it --order sequential --cycle.
    if flag_values(trainer, "--order") != ["sequential"] or trainer.count("--cycle") != 1:
        raise Stop("sft_e_template_not_sequential_cycle",
                   order=flag_values(trainer, "--order"), cycle=trainer.count("--cycle"))


def without_flag(argv, flag):
    """``argv`` without every ``flag value`` pair and ``flag=value`` item."""
    result, skip = [], False
    for part in argv:
        if skip:
            skip = False
        elif part == flag:
            skip = True
        elif not part.startswith(flag + "="):
            result.append(part)
    return result


def verify_rl_continuation_templates(templates):
    _, first = wrapper_and_trainer(templates["rl150"]["argv"])
    _, second = wrapper_and_trainer(templates["rl600"]["argv"])
    if flag_values(first, "--stop-after-steps") != [str(RL_SCREEN_UPDATES)]:
        raise Stop("rl150_template_stop_after_steps_not_150",
                   stop_after_steps=flag_values(first, "--stop-after-steps"))
    if flag_values(second, "--stop-after-steps"):
        raise Stop("rl600_template_declares_stop_after_steps")
    if without_flag(first, "--stop-after-steps") != second:
        raise Stop("rl600_template_differs_from_rl150")
    if any("{active_cap}" in part for part in first):
        raise Stop("rl_template_uses_active_cap_not_arm_cap")


IMPORTABLE = {".py", ".pyc", ".pyo", ".pth", ".so", ".pyd"}


def pythonpath(plan):
    value = (plan.get("environment") or {}).get("PYTHONPATH") or ""
    return [entry for entry in value.split(os.pathsep) if entry]


def verify_pins(plan):
    """Before every stage and gate: every pinned file's bytes; the interpreter link still
    resolving to the pinned binary; and the import tree being this checkout's, with nothing
    importable in it that is not pinned."""
    for name, expected in plan["file_sha256"].items():
        path = resolve(name)
        if not path.is_file() or sha(path) != expected:
            raise Stop("pinned_file_changed", file=name)
    # plan["python"] is a venv link (the venv's site-packages need it): a Homebrew upgrade
    # moves what it resolves to, so the resolved binary is pinned and re-resolved here.
    resolved = plan.get("python_resolved")
    if (not isinstance(resolved, str) or resolved not in plan["file_sha256"]
            or os.path.realpath(plan["python"]) != resolved):
        raise Stop("interpreter_link_changed", python=plan.get("python"),
                   resolves_to=os.path.realpath(plan["python"]), pinned=resolved)
    entries = [os.path.realpath(entry) for entry in pythonpath(plan)]
    if "import_root" not in plan:
        raise Stop("import_root_not_stated")
    root = plan["import_root"]
    if root is None:
        # No repository import tree: nothing on PYTHONPATH may carry vision_rl, or the
        # stages would import code no pin covers.
        carrying = [e for e in entries if Path(e, "vision_rl").exists()]
        if carrying:
            raise Stop("unpinned_import_tree_on_pythonpath", entries=carrying)
        return
    tree = os.path.realpath(resolve(root))
    if not entries or entries[0] != tree or not Path(tree).is_dir():
        # The stages import vision_rl from PYTHONPATH, the pins are checked against this
        # checkout: they must be the same tree, and it must come first.
        raise Stop("import_root_not_first_on_pythonpath", import_root=tree,
                   pythonpath=pythonpath(plan))
    pinned = {os.path.realpath(resolve(name)) for name in plan["file_sha256"]}
    unpinned = []
    for directory, folders, files in os.walk(tree):
        folders[:] = [f for f in folders if f != "__pycache__"]
        for name in files:
            path = os.path.join(directory, name)
            if Path(name).suffix in IMPORTABLE and os.path.realpath(path) not in pinned:
                unpinned.append(os.path.relpath(path, tree))
    if unpinned:
        raise Stop("unpinned_importable_file", files=sorted(unpinned)[:10])


def verify_manifest_content(directory, manifest, base, evaluator, images_sha256=None):
    """G0 'manifests hashed' and 'pHash audit with forced fire passing', by content."""
    labels = base.get("labels")
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        raise Stop("g0_evidence_labels_invalid")
    splits = {}
    for split in SPLITS:
        rows = read_jsonl(directory / f"{split}.jsonl")
        pairs = [(row.get("id"), row.get("label")) for row in rows]
        ids = [i for i, _ in pairs]
        per_class = {label: 0 for label in labels}
        for _, label in pairs:
            if label not in per_class:
                raise Stop("g0_manifest_unknown_label", split=split, label=label)
            per_class[label] += 1
        if (not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids)
                or any(n != H_PER_CLASS for n in per_class.values())):
            raise Stop("g0_holdout_not_2_per_class", split=split)
        if (manifest.get("counts") or {}).get(split) != len(rows):
            raise Stop("g0_manifest_count_differs", split=split)
        evidence_rows = (base.get("rows") or {}).get(split)
        if not isinstance(evidence_rows, list) or sorted(
                (r.get("id"), r.get("label")) for r in evidence_rows) != sorted(pairs):
            raise Stop("g0_evidence_rows_differ_from_manifest", split=split)
        splits[split] = ids
    if set(splits["h_select"]) & set(splits["h_gate"]):
        raise Stop("g0_h_select_and_h_gate_overlap")
    scan = jsonl_ids(directory / "s_scan_seed42.jsonl")
    holdout = set(splits["h_select"]) | set(splits["h_gate"])
    if not scan or len(set(scan)) != len(scan) or set(scan) & holdout:
        raise Stop("g0_scan_invalid_or_overlaps_holdout")
    if scan != evaluator["scan_order"](scan, SEED):
        raise Stop("g0_scan_not_seed42_order")
    verify_manifest_images(directory, images_sha256)
    audit = read_json(directory / "phash_audit.json")
    verify_pool(directory, manifest, audit, holdout, scan)
    fire = (audit or {}).get("forced_fire") if isinstance(audit, dict) else None
    pairs = (fire or {}).get("distinct_class_pairs") if isinstance(fire, dict) else None
    if not (
        isinstance(pairs, list) and len(pairs) == FORCED_FIRE_PAIRS
        and fire.get("passed") is True and fire.get("planted_flagged") is True
        and is_number(fire.get("planted_hamming")) and fire["planted_hamming"] <= HAMMING_MAX
        and all(isinstance(p, dict) and is_number(p.get("hamming"))
                and p["hamming"] > HAMMING_MAX for p in pairs)
        and fire.get("distinct_class_pairs_flagged") == 0
        and audit.get("hamming_max") == HAMMING_MAX
    ):
        raise Stop("g0_forced_fire_not_demonstrated")
    return {"split_ids": splits, "scan_ids": scan}


IMAGE_MANIFESTS = ("h_select.jsonl", "h_gate.jsonl", "s_scan_seed42.jsonl")


def images_digest(directory):
    """SHA256 over ``<manifest>\\t<id>\\t<image file sha256>\\n`` for every row a stage reads,
    in manifest order: the image CONTENT the G0 manifests name, as one pin."""
    digest = hashlib.sha256()
    for name in IMAGE_MANIFESTS:
        for row in read_jsonl(directory / name):
            digest.update(f"{name}\t{row['id']}\t{sha((directory / row['image']).resolve())}\n"
                          .encode())
    return digest.hexdigest()


def verify_manifest_images(directory, expected_sha256):
    """Every row a stage reads (H-select, H-gate, S_scan_seed42) names an image file that
    exists, resolved the way the trainer and the assays resolve it: against the manifest's
    own directory, and the images' bytes hash to the plan's ``g0.images_sha256``. The G0
    manifests carry paths relative to the development directory, so without the first check
    a launch would pass G0 and die at E0's first image; without the second a replaced image
    would be trained on unseen."""
    missing = []
    for name in IMAGE_MANIFESTS:
        for row in read_jsonl(directory / name):
            image = row.get("image") if isinstance(row, dict) else None
            if not isinstance(image, str) or not (directory / image).resolve().is_file():
                missing.append(f"{name}:{row.get('id') if isinstance(row, dict) else row}")
    if missing:
        raise Stop("g0_manifest_image_missing", count=len(missing), first=missing[:5])
    if not HEX64.fullmatch(str(expected_sha256)):
        raise Stop("g0_images_not_pinned")
    if images_digest(directory) != expected_sha256:
        raise Stop("g0_image_content_differs")


def verify_pool(directory, manifest, audit, holdout, scan):
    """G0 pool: pool.jsonl = native TRAIN - train400 - 15 probe16 - pHash drops, so its size is
    POOL_BEFORE_AUDIT minus the audit's pool_drops, none of which it contains; H-select,
    H-gate and S_scan_seed42 are subsets of it."""
    pool = jsonl_ids(directory / "pool.jsonl")
    if not pool or len(set(pool)) != len(pool):
        raise Stop("g0_pool_manifest_invalid")
    drops = audit.get("pool_drops") if isinstance(audit, dict) else None
    if (not isinstance(drops, list) or not all(isinstance(i, str) and i for i in drops)
            or len(set(drops)) != len(drops)):
        raise Stop("g0_pool_drop_list_invalid")
    if set(drops) & set(pool):
        raise Stop("g0_pool_contains_a_phash_drop")
    expected = POOL_BEFORE_AUDIT - len(drops)
    counts = manifest.get("counts") or {}
    if len(pool) != expected or counts.get("pool_after_audit") != len(pool) or counts.get(
            "pool_drops") != len(drops):
        raise Stop("g0_pool_count_differs", pool=len(pool), expected=expected,
                   pool_drops=len(drops))
    members = set(pool)
    if not holdout <= members:
        raise Stop("g0_holdout_outside_pool")
    if not set(scan) <= members - holdout:
        raise Stop("g0_scan_outside_pool_minus_h")


def verify_protocol(path):
    """G0 'protocol revision committed': a separately identified ACTIVATED copy of the
    committed, unactivated freshpool revision (protocol activation_prerequisites: root
    "freezes a separately identified activated protocol ...; this JSON remains historical
    candidate evidence"). The copy carries a new protocol_id and an ``activation`` block
    naming the revision by path and SHA256; every other top-level field equals the
    revision's, so the holdout salt (data.train_holdout.hash_protocol_id) and the
    baseline-discovery salt stay the revision's literal id. Its frozen gates equal those of
    the committed candidate the revision names, and all three files are committed."""
    protocol = read_json(path)
    if not isinstance(protocol, dict):
        raise Stop("g0_protocol_revision_wrong_id")
    if protocol.get("activated") is not True:
        raise Stop("g0_protocol_revision_not_activated")
    source = (protocol.get("activation") or {}).get("activates") or {}
    revision_path = resolve(source.get("path", "/nonexistent"))
    if not revision_path.is_file() or sha(revision_path) != source.get("sha256"):
        raise Stop("g0_protocol_revision_missing_or_changed")
    revision = read_json(revision_path)
    if (not isinstance(revision, dict) or revision.get("protocol_id") != PROTOCOL_ID
            or revision.get("activated") is not False):
        raise Stop("g0_protocol_revision_wrong_id")
    identifier = protocol.get("protocol_id")
    if not isinstance(identifier, str) or not identifier.strip() or identifier == PROTOCOL_ID:
        raise Stop("g0_activated_copy_not_separately_identified", protocol_id=identifier)
    changed = sorted(key for key in set(protocol) | set(revision)
                     if key not in ACTIVATION_KEYS and protocol.get(key) != revision.get(key))
    if changed:
        raise Stop("g0_activated_copy_differs_from_revision", fields=changed)
    holdout = ((protocol.get("data") or {}).get("train_holdout") or {}).get("hash_protocol_id")
    discovery = ((protocol.get("revises") or {}).get("baseline_discovery_salt") or {}).get(
        "literal")
    if holdout != PROTOCOL_ID or discovery != PROTOCOL_ID:
        raise Stop("g0_protocol_salt_changed", hash_protocol_id=holdout,
                   baseline_discovery_salt=discovery)
    revises = protocol.get("revises") or {}
    candidate_path = resolve(revises.get("path", "/nonexistent"))
    if not candidate_path.is_file() or sha(candidate_path) != revises.get("sha256"):
        raise Stop("g0_protocol_candidate_missing_or_changed")
    candidate = read_json(candidate_path)
    if not isinstance(candidate, dict) or any(
            key not in protocol or protocol[key] != candidate.get(key)
            for key in FROZEN_PROTOCOL_KEYS):
        raise Stop("g0_protocol_frozen_gates_changed")
    if not all(git_committed(p) for p in (path, revision_path, candidate_path)):
        raise Stop("g0_protocol_not_committed")


HEADING = re.compile(r"^\s*#")


def indent(line):
    return len(line) - len(line.lstrip())


def decision_bullets(lines):
    """Each entry with everything indented deeper than its first line joined onto it:
    wrapped continuation lines and nested sub-bullets alike, so a hedge anywhere under a
    bullet is read as part of it. A blank line or a heading ends the entry."""
    entries, base = [], None
    for line in lines:
        if (entries and base is not None and line.strip() and not HEADING.match(line)
                and indent(line) > base):
            entries[-1] += " " + line.strip()
            continue
        entries.append(line)
        base = indent(line) if line.strip() and not HEADING.match(line) else None
    return entries


def retirement_entry(entry):
    """True only for an explicit 'Retired:' entry naming the seed-2718, 1,600-image 16-shot
    plan, with no hedge ('not', 'pending', 'proposed', ...) anywhere in the whole entry,
    wrapped continuation lines included."""
    return bool(RETIRED_LINE.match(entry)) and all(f.search(entry) for f in RETIRED_FACTS) and (
        not RETIRED_HEDGE.search(entry))


def verify_decision_record(path):
    """G0 '16-shot plan retired': a committed decision record with an explicit retirement
    entry for the seed-2718, 1,600-image plan."""
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    if not any(retirement_entry(entry) for entry in decision_bullets(lines)):
        raise Stop("g0_16_shot_plan_not_retired")
    if not git_committed(path):
        raise Stop("g0_decision_record_not_committed")


def verify_g0(plan, evaluator):
    """G0: manifests hashed and well formed, forced fire passed, receipts committed with the
    stated content, branch resolved."""
    g0 = plan.get("g0") or {}
    directory = resolve(g0.get("manifest_dir", "/nonexistent"))
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or sha(manifest_path) != g0.get("manifest_sha256"):
        raise Stop("g0_manifest_identity_differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("g0_audit_complete") is not True or manifest.get(
            "forced_fire_passed") is not True:
        raise Stop("g0_audit_incomplete")
    if manifest.get("hash_protocol_id") != PROTOCOL_ID:
        raise Stop("g0_manifest_salt_differs", hash_protocol_id=manifest.get("hash_protocol_id"))
    files = manifest.get("files_sha256") or {}
    for name in REQUIRED_MANIFESTS:
        if name not in files:
            raise Stop("g0_manifest_missing", file=name)
    for name, expected in files.items():
        if not (directory / name).is_file() or sha(directory / name) != expected:
            raise Stop("g0_manifest_file_differs", file=name)
    receipts = g0.get("receipts") or {}
    for name in RECEIPTS:
        receipt = receipts.get(name) or {}
        path = resolve(receipt.get("path", "/nonexistent"))
        if not path.is_file() or sha(path) != receipt.get("sha256"):
            raise Stop("g0_receipt_missing_or_changed", receipt=name)
    evidence_base = plan.get("evidence_base") or {}
    base_path = resolve(evidence_base.get("path", "/nonexistent"))
    if not base_path.is_file() or sha(base_path) != evidence_base.get("sha256"):
        raise Stop("evidence_base_changed")
    base = read_json(base_path)
    branch_path = resolve(receipts["branch"]["path"])
    branch = read_json(branch_path)
    if not isinstance(base, dict) or not isinstance(branch, dict):
        raise Stop("g0_branch_or_evidence_unreadable")
    content = verify_manifest_content(directory, manifest, base, evaluator,
                                      g0.get("images_sha256"))
    verify_protocol(resolve(receipts["protocol_revision"]["path"]))
    verify_decision_record(resolve(receipts["decision_record"]["path"]))
    if not git_committed(branch_path):
        raise Stop("g0_branch_receipt_not_committed")
    try:
        d = evaluator["d_req"](branch["a_star"])
        checks = {
            "a_star_matches_evidence": branch["a_star"] == base.get("a_star"),
            "b4_matches_plan": branch["b4_edge_extension"] == plan["b4_edge_extension"]
            == base.get("b4_edge_extension"),
            "B0_w_valid_eos": branch["w_valid_eos"] >= 0.95,
            "B0_w_evaluator_not_refused": branch["w_evaluator_refused"] is False,
            "B1_w_val_in_range": 0.15 <= branch["w_val_accuracy"] <= 0.85,
            "d_req_published": abs(branch["d_req"] - d["d_req"]) <= 1e-9,
            "b2_mode_published": branch["mode"] == d["mode"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise Stop("g0_branch_receipt_invalid", error=str(exc)) from None
    if not all(checks.values()):
        raise Stop("g0_failed", checks=checks)
    paths = {f"manifest_{Path(name).stem}": str(directory / name) for name in files}
    return {"checks": checks, "d_req": d, "manifests": paths, **content}


class MixedMonitor:
    """Live standing stop: tails the RL stage's reward_groups.jsonl (one row per accepted
    update, ``mixed`` a bool) under its arm directory and reports the first 100-update window
    with fewer than 40 mixed groups. A second reward_groups.jsonl or an unparseable row is a
    stop, never a skip. The fragment's useful_mixed is checked again after the stage."""

    def __init__(self, arm_dir):
        self.arm_dir, self.path, self.offset, self.buffer = arm_dir, None, 0, b""
        self.flags = []

    def poll(self):
        # Re-scanned every poll: a second file appearing after the first was found is as
        # ambiguous as two at once, and is a stop.
        found = sorted(self.arm_dir.rglob("reward_groups.jsonl"))
        if len(found) > 1:
            return "reward_groups_ambiguous"
        if self.path is None:
            if not found:
                return None
            self.path = found[0]
        elif found != [self.path]:
            return "reward_groups_ambiguous"
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        self.offset += len(data)
        *lines, self.buffer = (self.buffer + data).split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                return "reward_groups_unparseable"
            if not isinstance(row, dict) or type(row.get("mixed")) is not bool:
                return "reward_groups_unparseable"
            self.flags.append(row["mixed"])
            if (len(self.flags) >= MIXED_WINDOW
                    and sum(self.flags[-MIXED_WINDOW:]) < MIXED_MIN_IN_WINDOW):
                return "rolling_useful_mixed_below_0_40"
        return None


# ---------------------------------------------------------------- state and journal


class Ledger:
    """state.json (resume pointer) plus an append-only, fsynced journal.jsonl."""

    def __init__(self, directory):
        self.directory = directory
        self.state_path = directory / "state.json"
        self.journal_path = directory / "journal.jsonl"
        self.state = None

    def append(self, event, **fields):
        line = json.dumps({"event": event, "time": time.time(), **fields}, allow_nan=False,
                          default=str)
        with self.journal_path.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save(self):
        data = self.journal_path.read_bytes() if self.journal_path.exists() else b""
        self.state["journal_bytes"] = len(data)
        self.state["journal_sha256"] = hashlib.sha256(data).hexdigest()
        write(self.state_path, self.state)

    def check_journal_prefix(self):
        data = self.journal_path.read_bytes() if self.journal_path.exists() else b""
        prefix = data[: self.state["journal_bytes"]]
        if len(prefix) != self.state["journal_bytes"] or hashlib.sha256(
                prefix).hexdigest() != self.state["journal_sha256"]:
            raise Stop("journal_not_append_only")


def read_lock(lock):
    """(pid, start time) from a lock file: the pid on line one, the holder's ps ``lstart``
    on line two. Raises ValueError when it is not that (empty while being written, corrupt)."""
    lines = Path(lock).read_text().splitlines()
    if len(lines) != 2 or not lines[1].strip():
        raise ValueError("lock is not 'pid' + 'start time'")
    return int(lines[0].strip()), lines[1].strip()


def lock_holder_alive(pid, started):
    """True only if the process that wrote the lock is still running: same pid AND same
    start time, so a pid reused after a reboot is not mistaken for the launcher."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's process holds the pid: cannot be ours, but ask
    return process_started(pid) == started


def acquire_lock(ledger):
    lock = ledger.directory / "lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder, started = read_lock(lock)
        except (OSError, ValueError):
            raise Stop("lock_unreadable") from None
        if lock_holder_alive(holder, started):
            raise Stop("another_launcher_holds_the_lock", pid=holder)
        ledger.append("stale_lock_removed", pid=holder, started=started)
        lock.unlink()
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    started = process_started(os.getpid())
    if not started:
        os.close(descriptor)
        lock.unlink()
        raise Stop("own_start_time_unreadable")
    os.write(descriptor, f"{os.getpid()}\n{started}\n".encode())
    os.close(descriptor)
    return lock


# ---------------------------------------------------------------- stages


class Launcher:
    def __init__(self, plan, plan_sha, grid, evaluator, ledger, g0):
        self.plan, self.plan_sha, self.grid = plan, plan_sha, grid
        self.evaluator, self.ledger, self.g0 = evaluator, ledger, g0
        self.limits = plan["limits"]
        self.state = ledger.state
        self.state.setdefault("void_instances", [])
        self.state.setdefault("void_seconds_by_arm", {})
        self.state.setdefault("instance_start_seconds", {})
        self.quiet = Path(plan["quiet_file"])

    def wait_while_quiet(self, stage_id):
        """Quiet mode (the fan guard demotes busy Python to efficiency cores at nice 19 while
        the file exists, and cannot undo the nice) is waited out BEFORE a stage starts, so
        no stage starts demoted against its fixed wall-time cap. Mid-stage demotion is seen
        by supervise() and journalled; it cannot be undone without root."""
        if not self.quiet.exists():
            return
        self.ledger.append("quiet_mode_wait", stage=stage_id, file=str(self.quiet))
        waited = time.monotonic()
        while self.quiet.exists():
            time.sleep(QUIET_POLL_SECONDS)
        self.ledger.append("quiet_mode_cleared", stage=stage_id,
                           waited_seconds=time.monotonic() - waited)

    # -- guards
    def check_disk(self):
        free = free_bytes(self.ledger.directory)
        if free < self.limits["minimum_free_bytes"]:
            raise Stop("disk_reserve", free_bytes=free,
                       minimum_free_bytes=self.limits["minimum_free_bytes"])
        return free

    def check_owners(self):
        found = second_owners(self.plan["owner_patterns"])
        if found:
            raise Stop("second_accelerator_owner", processes=found)

    # -- running one stage
    def run(self, stage_id, template_name, arm, forecast, values, validate, *,
            inherited_seconds=0.0, watch_mixed=False, lineage=()):
        """Run one stage in a fresh instance. ``inherited_seconds`` is the validated parent
        endpoint's committed active seconds for a continuation, so the arm cap counts the
        whole arm and is never reset. Wall seconds spent in this arm's voided (interrupted)
        instances are charged on top, cumulatively: the arm's cap is never refilled by an
        interrupt. ``lineage`` names the parent arms whose voided seconds are charged too
        (SFT-T continues SFT-E under its own label, as RL 600 continues RL 150 under one
        label), so an interrupt is charged the same way in every lineage. ``watch_mixed``
        enforces the rolling useful-mixed standing stop live."""
        if stage_id in self.state["completed"]:
            return self.state["completed"][stage_id]["fragment"]
        template = self.plan["templates"][template_name]
        self.wait_while_quiet(stage_id)
        verify_pins(self.plan)
        self.verify_screen_tokenizations()
        free = self.check_disk()
        headroom = self.limits["minimum_free_bytes"] + self.limits["stage_headroom_bytes"]
        if free < headroom:
            # Not the standing stop (free is still above the reserve), and nothing of this
            # stage has run: pause so a person can free disk, instead of starting a stage
            # that would breach the reserve mid-run and end the study.
            raise Pause("disk_headroom_before_stage", stage=stage_id, free_bytes=free,
                        required_bytes=headroom)
        self.check_owners()
        cap = CAP_FACTOR * forecast
        previous = self.state["attempts"].get(stage_id, 0)
        voids = self.state["void_seconds_by_arm"]
        if previous:
            # Interrupted earlier: never resumed and never quoted. Its instance is void and
            # the stage re-runs from scratch in a fresh directory. Its wall time stays
            # charged to the arm (study.json budget_matching.caps: no reset on resume).
            start = self.state["instance_start_seconds"].get(stage_id, 0.0)
            spent = max(0.0, self.state["arm_wall_seconds"].get(stage_id, start) - start)
            voids[arm] = voids.get(arm, 0.0) + spent
            void = {"stage": stage_id, "arm": arm, "instance": previous,
                    "instance_dir": str(self.ledger.directory / "arms" / stage_id
                                        / f"instance-{previous}"),
                    "wall_seconds": self.state["arm_wall_seconds"].get(stage_id),
                    "instance_wall_seconds": spent, "arm_void_seconds": voids[arm]}
            self.state["void_instances"].append(void)
            self.ledger.save()
            self.ledger.append("arm_instance_void", reason="interrupted; never resumed or quoted",
                               **void)
        voided = sum(voids.get(name, 0.0) for name in (arm, *lineage))
        used = inherited_seconds + voided
        if used >= cap:
            raise Stop("arm_cap", arm=arm, used_seconds=used, inherited_seconds=inherited_seconds,
                       void_seconds=voided, cap_seconds=cap)
        instance = previous + 1
        stage_dir = self.ledger.directory / "stages" / stage_id / f"instance-{instance}"
        arm_dir = self.ledger.directory / "arms" / stage_id / f"instance-{instance}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        arm_dir.mkdir(parents=True, exist_ok=False)
        fragment = stage_dir / "fragment.json"
        substitutions = {
            "python": self.plan["python"], "stage_id": stage_id, "arm": arm,
            "stage_dir": str(stage_dir), "arm_dir": str(arm_dir), "fragment": str(fragment),
            "seed": str(SEED), "active_cap": repr(cap - voided), "arm_cap": repr(cap),
            "w_adapter": self.plan["w_endpoint"][
                "adapter_path"], **self.g0["manifests"], **values,
        }
        if "screen_tokenizations" in self.state:
            substitutions["screen_tokenizations"] = self.state["screen_tokenizations"]["path"]
        try:
            forwarded = [part.format_map(substitutions) for part in template["argv"]]
        except (KeyError, ValueError, IndexError) as exc:
            raise Stop("template_placeholder_unresolved", stage=stage_id, error=str(exc)) from None
        command = [
            self.plan["python"], "-B", str(HERE / "phase_worker.py"),
            "--guard-dir", str(self.ledger.directory / "guards"), "--stage", stage_id,
            "--kind", template["kind"], "--entry", template["entry"],
            "--entry-sha256", template["entry_sha256"],
            *(["--dry-run"] if self.plan.get("dry_run") is True else []), "--", *forwarded,
        ]
        self.state["attempts"][stage_id] = instance
        self.state["arm_wall_seconds"][stage_id] = used
        self.state["instance_start_seconds"][stage_id] = used
        self.state["next_action"] = stage_id
        self.ledger.save()
        self.ledger.append("stage_started", stage=stage_id, arm=arm, instance=instance,
                           command=command, cap_seconds=cap, inherited_seconds=inherited_seconds,
                           void_seconds=voided, lineage=list(lineage), free_disk_bytes=free,
                           owner_poll_seconds=OWNER_POLL_SECONDS, live_useful_mixed=watch_mixed)
        monitor = MixedMonitor(arm_dir) if watch_mixed else None
        outcome = self.supervise(stage_id, command, stage_dir, used, cap, monitor,
                                 instance=instance)
        self.ledger.append("stage_finished", stage=stage_id, **outcome)
        if outcome["reason"]:
            raise Stop(outcome["reason"], stage=stage_id)
        if outcome["exit_code"] != 0:
            raise Stop("stage_failed", stage=stage_id, exit_code=outcome["exit_code"])
        if monitor is not None:
            # Read what the stage wrote after the last poll; a stage whose reward groups
            # were never visible under its arm directory was not monitored, so it stops.
            late = monitor.poll()
            if late:
                raise Stop(late, stage=stage_id)
            if monitor.path is None:
                raise Stop("live_useful_mixed_unmonitored", stage=stage_id)
            self.ledger.append("useful_mixed_monitored", stage=stage_id, path=str(monitor.path),
                               rows=len(monitor.flags))
        payload = read_json(fragment)
        if not isinstance(payload, dict):
            raise Stop("stage_fragment_missing", stage=stage_id)
        try:
            problem = validate(payload, cap - voided)
        except Stop:
            raise
        except Exception as exc:  # a malformed fragment is a stop, never a pass
            problem = f"fragment_unvalidatable: {type(exc).__name__}: {exc}"
        if problem:
            raise Stop(problem, stage=stage_id)
        self.state["completed"][stage_id] = {"fragment": payload, "path": str(fragment)}
        self.ledger.save()
        self.ledger.append("stage_completed", stage=stage_id, fragment=str(fragment),
                           fragment_sha256=sha(fragment))
        return payload

    def supervise(self, stage_id, command, stage_dir, used, cap, monitor=None, instance=None):
        started, peak, reason, demoted = time.monotonic(), 0, None, None
        last_save = last_owner = started
        with (stage_dir / "stage.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
            pgid = child.pid
            # Recorded before anything else can fail, so a launcher killed from outside
            # (chain.py's bound, kill -9) leaves a record reap_orphan_stage() can act on.
            self.state["running_stage"] = {"stage": stage_id, "instance": instance,
                                           "pgid": pgid,
                                           "leader_started": process_started(pgid)}
            self.ledger.save()
            try:
                while child.poll() is None:
                    members = group_members(pgid)
                    rss = sum(row[4] for row in members)
                    peak = max(peak, rss)
                    lowered = [{"pid": row[0], "nice": row[6]} for row in members
                               if row[6] is not None and row[6] > 0]
                    if lowered and demoted is None:
                        # Quiet mode's fan guard (or anything else) lowered the stage's
                        # priority mid-run. Not a stop; journalled so a cap hit is explained.
                        demoted = lowered
                        self.ledger.append("stage_demoted", stage=stage_id, processes=lowered)
                    elapsed = time.monotonic() - started
                    self.state["arm_wall_seconds"][stage_id] = used + elapsed
                    if time.monotonic() - last_save >= 5:
                        self.ledger.save()
                        last_save = time.monotonic()
                    owners = []
                    if time.monotonic() - last_owner >= OWNER_POLL_SECONDS:
                        owners = second_owners(self.plan["owner_patterns"], (pgid,))
                        last_owner = time.monotonic()
                        if owners:
                            self.ledger.append("second_owner_seen", stage=stage_id,
                                               processes=owners)
                    free = free_bytes(self.ledger.directory)
                    mixed = monitor.poll() if monitor is not None else None
                    reason = (
                        "process_group_rss_limit" if rss > self.limits["rss_limit_bytes"]
                        else "disk_reserve" if free < self.limits["minimum_free_bytes"]
                        else "arm_cap" if used + elapsed >= cap
                        else "second_accelerator_owner" if owners
                        else mixed
                    )
                    if reason:
                        stop_group(child, pgid)
                        break
                    time.sleep(self.limits["poll_seconds"])
                code = child.wait()
                # Survivor grace: helpers may still be exiting at the instant the stage
                # returns. Wait a bounded grace, then record and stop whatever remains.
                survivors, deadline = [], time.monotonic() + self.limits["survivor_grace_seconds"]
                while group_alive(pgid):
                    if time.monotonic() >= deadline:
                        survivors = [{"pid": row[0], "command": row[5][:200]}
                                     for row in group_members(pgid)]
                        break
                    time.sleep(0.1)
                if survivors:
                    self.ledger.append("owned_survivors", stage=stage_id, processes=survivors)
                    stop_group(child, pgid, grace=5)
                    reason = reason or "stage_exited_with_owned_descendants"
            except BaseException:
                stop_group(child, pgid, grace=5)
                self.state["arm_wall_seconds"][stage_id] = used + time.monotonic() - started
                if not group_alive(pgid):
                    self.state.pop("running_stage", None)
                self.ledger.save()
                raise
        elapsed = time.monotonic() - started
        self.state["arm_wall_seconds"][stage_id] = used + elapsed
        self.state.pop("running_stage", None)
        self.ledger.save()
        return {"exit_code": code, "reason": reason, "elapsed_seconds": elapsed,
                "arm_wall_seconds": used + elapsed, "peak_group_rss_bytes": peak,
                "demoted": demoted}

    # -- fragment validation
    def train_validator(self, run_id, arm_name, lr, updates=None, rl=False, check=None):
        w = self.plan["w_endpoint"]["adapter_sha256"]

        def validate(f, cap):
            seconds = f.get("committed_active_seconds")
            if f.get("status") != "completed":
                return "arm_not_completed"
            if (f.get("run_id"), f.get("arm"), f.get("lr"), f.get("seed")) != (
                    run_id, arm_name, lr, SEED):
                return "fragment_identity_differs"
            if not is_number(seconds) or seconds < 0:
                return "fragment_active_seconds_invalid"
            if seconds > cap:
                return "arm_cap"
            if updates is not None and f.get("updates") != updates:
                return "fragment_updates_differ"
            if type(f.get("updates")) is not int or f["updates"] < 1:
                return "fragment_updates_invalid"
            if not HEX64.fullmatch(str(f.get("adapter_sha256"))) or f["adapter_sha256"] == w:
                return "fragment_adapter_unchanged_or_invalid"
            # Every arm starts from W: RL keeps W as its KL reference; SFT arms start at W.
            if f.get("reference_sha256") != w:
                return "reference_changed" if rl else "fragment_initial_adapter_not_w"
            for key in ("checkpoint", "adapter_path"):
                if not f.get(key) or not Path(f[key]).exists():
                    return f"fragment_{key}_missing"
            steps = f.get("step_committed_active_seconds")
            if not isinstance(steps, dict) or not steps:
                return "fragment_step_seconds_missing"
            try:
                series = sorted((int(k), v) for k, v in steps.items())
            except (TypeError, ValueError):
                return "fragment_step_seconds_invalid"
            if (any(not is_number(v) for _, v in series)
                    or any(b[1] < a[1] for a, b in zip(series, series[1:]))
                    or [k for k, _ in series] != list(range(series[0][0], series[-1][0] + 1))
                    or series[-1][0] != f["updates"] or series[-1][1] != seconds):
                return "fragment_step_seconds_inconsistent"
            mixed = f.get("useful_mixed")
            if not isinstance(mixed, list) or any(type(m) is not bool for m in mixed) or (
                    rl and len(mixed) != f["updates"]):
                return "fragment_useful_mixed_invalid"
            if rl:
                for key in ("accepted_manifest", "scanned_manifest"):
                    if not f.get(key) or not Path(f[key]).is_file():
                        return f"fragment_{key}_missing"
                accepted = jsonl_ids(f["accepted_manifest"])
                scanned = jsonl_ids(f["scanned_manifest"])
                if (accepted is None or scanned is None or len(accepted) != f["updates"]
                        or not set(accepted) <= set(scanned)
                        or scanned != self.g0["scan_ids"][: len(scanned)]):
                    return "fragment_screen_manifests_inconsistent"
                for start in range(0, max(0, len(mixed) - MIXED_WINDOW + 1)):
                    if sum(mixed[start:start + MIXED_WINDOW]) < MIXED_MIN_IN_WINDOW:
                        return "rolling_useful_mixed_below_0_40"
            return check(f) if check else None

        return validate

    def assay_validator(self, run_id, split, adapter):
        expected_ids = set(self.g0["split_ids"][split])
        completions = COMPLETIONS[split]

        def validate(f, cap):
            if (f.get("run_id"), f.get("split"), f.get("adapter_sha256")) != (
                    run_id, split, adapter):
                return "assay_identity_differs"
            predictions = f.get("predictions")
            if not isinstance(predictions, dict) or set(predictions) != expected_ids:
                return "assay_image_ids_differ_from_manifest"
            for answers in predictions.values():
                if (not isinstance(answers, list) or len(answers) != completions
                        or any(a is not None and not isinstance(a, str) for a in answers)):
                    return "assay_completion_count_differs"
            eos = f.get("eos_fraction")
            if not is_number(eos) or not 0 <= eos <= 1:
                return "assay_eos_fraction_invalid"
            return None

        return validate

    @staticmethod
    def e0_validator(f, cap):
        """Exactly e0_gate.py's check set, every one true (protocol
        real_learning_and_resume_gate), and sampler-teacher agreement <= 0.001."""
        checks = f.get("checks")
        if (not isinstance(checks, dict) or set(checks) != set(E0_CHECKS)
                or any(checks[k] is not True for k in E0_CHECKS)):
            return "e0_failed"
        agreement = f.get("sampler_teacher_max_abs_logprob_difference")
        if not is_number(agreement) or not 0 <= agreement <= 0.001:
            return "e0_sampler_teacher_agreement_failed"
        return None

    def p0_validator(self, f, cap):
        if not all(k in f for k in ("seed", "adapter_sha256", "scan_order", "journal",
                                    "rollouts")):
            return "p0_g1_document_incomplete"
        if f["seed"] != SEED or f["adapter_sha256"] != self.plan["w_endpoint"]["adapter_sha256"]:
            return "p0_not_seed42_under_w"
        if f["scan_order"] != self.g0["scan_ids"]:
            return "p0_scan_order_differs_from_manifest"
        if len(f["journal"]) != G1_CANDIDATES or len(f["rollouts"]) != G1_ROLLOUTS:
            return "p0_g1_document_wrong_size"
        return None

    # -- lineage checks (study.json budget_matching: cumulative, checkpoint-bound seconds)
    @staticmethod
    def continuation_check(parent):
        """RL 150 -> 600: the continuation's step series passes through the parent's own
        committed seconds at the parent's step, and its accepted sequence extends the
        parent's, so T_RL is cumulative and the arm was not reset."""
        def check(f):
            steps = f["step_committed_active_seconds"]
            at_parent = steps.get(str(parent["updates"]))
            if at_parent != parent["committed_active_seconds"] or min(map(int, steps)) != 1:
                return "continuation_seconds_not_cumulative"
            if f["committed_active_seconds"] <= parent["committed_active_seconds"]:
                return "continuation_seconds_not_cumulative"
            parent_ids = jsonl_ids(parent["accepted_manifest"]) or []
            if (jsonl_ids(f["accepted_manifest"]) or [])[: len(parent_ids)] != parent_ids:
                return "continuation_accepted_sequence_not_extended"
            return None
        return check

    @staticmethod
    def exposure_check(manifest, then=None):
        """SFT-E and EG-SFT: exactly RL*'s accepted presentations in order (study.json arms
        SFT-E / EG-SFT), read from the trainer receipt's committed prompt IDs, not from the
        manifest the template handed it."""
        expected = jsonl_ids(manifest)

        def check(f):
            if expected is None or len(expected) != FINAL_UPDATES:
                return "rl_star_accepted_manifest_unreadable"
            if f.get("committed_prompt_ids") != expected:
                return "exposure_differs_from_rl_star_accepted_sequence"
            return then(f) if then else None
        return check

    @staticmethod
    def sft_e_boundary_check(steps):
        """SFT-E must end as a continuable endpoint: the trainer's SFT-T continuation accepts
        only a parent whose receipt is boundary_reached at its declared stop_after_steps
        (vlm_local_training._continuation_identity). An SFT-E run to max_steps is
        total_budget_exhausted and would be refused only at SFT-T, after G3a."""
        def check(f):
            boundary = f.get("requested_boundary")
            if (f.get("receipt_status") != "boundary_reached" or not isinstance(boundary, dict)
                    or boundary.get("stop_after_steps") != steps or f["updates"] != steps):
                return "sft_e_not_a_continuable_boundary"
            return None
        return check

    @staticmethod
    def sft_t_check(parent, t_rl, scanned_manifest):
        """SFT-T continues SFT-E without reset and stops at the first step at or above
        T_RL: the step before it is still below T_RL. Its exposure is SFT-E's, then RL*'s
        scanned sequence from row 0, cycling (vlm_local_training: "TRAIN row 0 at global
        step parent_global_step + 1, sequential, cycling")."""
        scanned = jsonl_ids(scanned_manifest)

        def check(f):
            ids = f.get("committed_prompt_ids")
            p = parent["updates"]
            if (not scanned or not isinstance(ids, list) or len(ids) != f["updates"]
                    or ids[:p] != parent.get("committed_prompt_ids")
                    or ids[p:] != [scanned[n % len(scanned)] for n in range(len(ids) - p)]):
                return "sft_t_exposure_not_sft_e_then_scanned_from_row_0"
            steps = f["step_committed_active_seconds"]
            first = min(map(int, steps))
            if (first != parent["updates"] + 1
                    or steps[str(first)] < parent["committed_active_seconds"]
                    or f["updates"] <= parent["updates"]):
                return "sft_t_not_cumulative_over_sft_e"
            if f["committed_active_seconds"] < t_rl - 1e-9:
                return "sft_t_short_of_t_rl"
            before = (steps[str(f["updates"] - 1)] if f["updates"] - 1 >= first
                      else parent["committed_active_seconds"])
            if before >= t_rl:
                return "sft_t_overshoots_first_boundary_at_t_rl"
            return None
        return check

    # -- evidence and gates
    def endpoints(self):
        w = self.plan["w_endpoint"]
        found = {"W": {"run_id": "W", "arm": "W", "lr": None, "seed": None, "updates": 0,
                       "committed_active_seconds": 0, "adapter_sha256": w["adapter_sha256"],
                       "reference_sha256": w["adapter_sha256"], "useful_mixed": [],
                       "splits": {}}}
        keys = ("run_id", "arm", "lr", "seed", "updates", "committed_active_seconds",
                "adapter_sha256", "reference_sha256", "useful_mixed")
        assays = []
        for entry in self.state["completed"].values():
            f = entry["fragment"]
            if "split" in f:
                assays.append(f)
            elif f.get("arm") in ("RL", "SFT-E", "SFT-T", "EG-SFT"):
                found[f["run_id"]] = {**{k: f[k] for k in keys}, "splits": {}}
        for f in assays:
            if f["run_id"] not in found:
                raise Stop("assay_for_unknown_endpoint", run_id=f["run_id"])
            found[f["run_id"]]["splits"][f["split"]] = {
                "predictions": f["predictions"], "eos_fraction": f["eos_fraction"]}
        return found

    def evidence(self):
        base = json.loads(resolve(self.plan["evidence_base"]["path"]).read_text())
        document = {k: base[k] for k in ("labels", "rows", "a_star", "b4_edge_extension")}
        document["endpoints"] = list(self.endpoints().values())
        if "p0" in self.state["completed"]:
            document["g1"] = self.state["completed"]["p0"]["fragment"]
        return document

    def study(self):
        try:
            return self.evaluator["Study"](self.evidence())
        except (KeyError, TypeError, ValueError) as exc:
            raise Stop("selection_failed", error=f"{type(exc).__name__}: {exc}") from None

    def run_evaluator(self, name):
        directory = self.ledger.directory / "gates"
        directory.mkdir(exist_ok=True)
        attempt = len(list(directory.glob(f"{name}-report-*.json"))) + 1
        evidence = directory / f"{name}-evidence-{attempt}.json"
        report_path = directory / f"{name}-report-{attempt}.json"
        write(evidence, self.evidence())
        self.ledger.append("gate_started", gate=name, evidence=str(evidence),
                           evidence_sha256=sha(evidence))
        try:
            completed = subprocess.run(
                [self.plan["python"], "-B", str(resolve(self.plan["evaluator"])),
                 "--evidence", str(evidence), "--output", str(report_path)],
                cwd=ROOT, capture_output=True, text=True, timeout=1800)
            code, stderr = completed.returncode, completed.stderr[-2000:]
        except subprocess.TimeoutExpired:
            code, stderr = None, "evaluator timed out"
        report = read_json(report_path)
        return code, stderr, report if isinstance(report, dict) else None, report_path

    def gate(self, name):
        if name in self.state["gates"]:
            return self.state["gates"][name]["result"]
        verify_pins(self.plan)
        code, stderr, report, report_path = self.run_evaluator(name)
        gates = (report or {}).get("gates")
        result = gates.get(name) if isinstance(gates, dict) else None
        verdict = (
            "gate_result_missing" if code != 0 or not isinstance(result, dict)
            else "gate_not_evaluated" if result.get("status") != "evaluated"
            else "gate_failed" if result.get("pass") is not True
            or report.get("failed_at") is not None
            else None
        )
        self.ledger.append("gate_finished", gate=name, exit_code=code, report=str(report_path),
                           report_sha256=sha(report_path) if report_path.is_file() else None,
                           verdict=verdict or "pass", stderr=stderr if verdict else "")
        if verdict:
            raise Stop(verdict, gate=name, report=str(report_path))
        self.state["gates"][name] = {"report": str(report_path), "result": result}
        self.ledger.save()
        return result

    # -- the study sequence
    def assay(self, endpoint, split, adapter_path, adapter_sha):
        stage = f"{endpoint}-{split}"
        return self.run(stage, "assay", stage, self.plan["forecast_seconds"][f"assay_{split}"],
                        {"run_id": endpoint, "split": split, "adapter": adapter_path,
                         "split_manifest": self.g0["manifests"][f"manifest_{split}"]},
                        self.assay_validator(endpoint, split, adapter_sha))

    def assay_fragment(self, fragment, split):
        return self.assay(fragment["run_id"], split, fragment["adapter_path"],
                          fragment["adapter_sha256"])

    def sequence(self):
        forecast = self.plan["forecast_seconds"]
        w = self.plan["w_endpoint"]
        self.run("e0", "e0", "e0", forecast["e0"], {}, self.e0_validator)
        for split in SPLITS:
            self.assay("W", split, w["adapter_path"], w["adapter_sha256"])
        self.run("p0", "p0", "p0", forecast["p0"], {}, self.p0_validator)
        g1 = self.gate("G1")
        self.freeze_screen_tokenizations(g1)
        measured = self.freeze_measured_forecasts(g1)

        rl = {}
        for lr in self.grid:
            arm = f"rl-{lr_tag(lr)}-s{SEED}"
            run_id = f"{arm}-u{RL_SCREEN_UPDATES}"
            rl[run_id] = self.run(run_id, "rl150", arm, measured["rl"],
                                  {"run_id": run_id, "lr": repr(lr)},
                                  self.train_validator(run_id, "RL", lr, RL_SCREEN_UPDATES,
                                                       rl=True),
                                  watch_mixed=True)
            self.assay_fragment(rl[run_id], "h_select")
        g2 = self.gate("G2")
        continuers = g2.get("continue_to_600")
        if not isinstance(continuers, list) or len(continuers) != 2 or any(
                c not in rl for c in continuers):
            raise Stop("g2_continuers_missing", continuers=continuers)
        self.record_selection("rl_continuers", continuers)

        for parent_id in continuers:
            parent = rl[parent_id]
            lr = parent["lr"]
            arm = f"rl-{lr_tag(lr)}-s{SEED}"
            run_id = f"{arm}-u{FINAL_UPDATES}"
            final = self.run(run_id, "rl600", arm, measured["rl"],
                             {"run_id": run_id, "lr": repr(lr),
                              "resume_checkpoint": parent["checkpoint"]},
                             self.train_validator(run_id, "RL", lr, FINAL_UPDATES, rl=True,
                                                  check=self.continuation_check(parent)),
                             inherited_seconds=parent["committed_active_seconds"],
                             watch_mixed=True)
            self.assay_fragment(final, "h_select")
        _, rl_star = self.study().select_rl_star()
        rl_star = self.fragment(rl_star["run_id"])
        self.record_selection("rl_star", rl_star["run_id"])

        sft_e = {}
        for lr in self.grid:
            run_id = f"sft-e-{lr_tag(lr)}-s{SEED}"
            sft_e[lr] = self.run(run_id, "sft_e", run_id, forecast["sft_e"],
                                 {"run_id": run_id, "lr": repr(lr),
                                  "accepted_manifest": rl_star["accepted_manifest"]},
                                 self.train_validator(
                                     run_id, "SFT-E", lr, FINAL_UPDATES,
                                     check=self.exposure_check(
                                         rl_star["accepted_manifest"],
                                         self.sft_e_boundary_check(FINAL_UPDATES))))
            self.assay_fragment(sft_e[lr], "h_select")
        _, sft_e_star, _ = self.study().select_sft_e()
        self.record_selection("sft_e_star", sft_e_star["run_id"])
        self.assay_fragment(rl_star, "h_gate")
        self.assay_fragment(self.fragment(sft_e_star["run_id"]), "h_gate")
        g3a = self.gate("G3a")
        if g3a.get("rl_star") != rl_star["run_id"] or g3a.get(
                "sft_e_star") != sft_e_star["run_id"]:
            raise Stop("g3a_selection_disagrees_with_launch")
        qualifying = g3a.get("sft_t_lrs")
        if not isinstance(qualifying, list) or not qualifying or any(
                lr not in self.grid for lr in qualifying):
            raise Stop("g3a_sft_t_lrs_missing", sft_t_lrs=qualifying)
        self.record_selection("sft_t_lrs", qualifying)

        # T_RL: RL*'s cumulative committed active seconds (checked through its 150 parent).
        t_rl = rl_star["committed_active_seconds"]
        for lr in qualifying:
            run_id = f"sft-t-{lr_tag(lr)}-s{SEED}"
            parent = sft_e[lr]
            fragment = self.run(run_id, "sft_t", run_id, t_rl,
                                {"run_id": run_id, "lr": repr(lr), "t_rl": repr(t_rl),
                                 "scanned_manifest": rl_star["scanned_manifest"],
                                 "parent_checkpoint": parent["checkpoint"]},
                                self.train_validator(
                                    run_id, "SFT-T", lr,
                                    check=self.sft_t_check(parent, t_rl,
                                                           rl_star["scanned_manifest"])),
                                inherited_seconds=parent["committed_active_seconds"],
                                lineage=(f"sft-e-{lr_tag(lr)}-s{SEED}",))
            self.assay_fragment(fragment, "h_select")
        _, sft_t_star = self.study().select_sft_t()
        self.record_selection("sft_t_star", sft_t_star["run_id"])
        self.assay_fragment(self.fragment(sft_t_star["run_id"]), "h_gate")
        self.gate("G3b")

        lr = sft_e_star["lr"]
        run_id = f"egsft-{lr_tag(lr)}-s{SEED}"
        eg = self.run(run_id, "egsft", run_id, measured["egsft"],
                      {"run_id": run_id, "lr": repr(lr),
                       "accepted_manifest": rl_star["accepted_manifest"]},
                      self.train_validator(run_id, "EG-SFT", lr, FINAL_UPDATES,
                                           check=self.exposure_check(
                                               rl_star["accepted_manifest"])))
        self.assay_fragment(eg, "h_gate")
        self.final_check()

    def freeze_screen_tokenizations(self, g1):
        """study.json alt_tokenization_switch, frozen before any RL stage (so before G2):
        when more than 2% of P0's correct samples are non-canonical tokenizations, the
        screen sums p over every tokenization P0 observed for a label; otherwise over the
        canonical one only. Either way one file is written and handed to every RL and
        EG-SFT stage as ``{screen_tokenizations}``: empty ``sequences`` is the trainer's
        canonical-only score."""
        if "screen_tokenizations" in self.state:
            return self.state["screen_tokenizations"]
        share = (g1.get("values") or {}).get("non_canonical_share_of_correct")
        if share is None:
            # No correct P0 sample: the switch's input is undefined, so it cannot be decided.
            raise Stop("g1_non_canonical_share_undefined")
        if not is_number(share) or not 0 <= share <= 1:
            raise Stop("g1_non_canonical_share_invalid", share=share)
        switch = share > ALT_TOKENIZATION_SHARE
        sequences = {}
        if switch:
            labels = {}
            for row in read_jsonl(self.g0["manifests"]["manifest_s_scan_seed42"]):
                labels[row["id"]] = row["label"]
            for group in self.state["completed"]["p0"]["fragment"]["rollouts"]:
                for completion in group["completions"]:
                    # Only a bare label is a tokenization of it; a correct completion that
                    # carried a think block or tags is not summed into p.
                    if completion.get("correct") is not True or completion.get(
                            "label_only") is not True:
                        continue
                    ids = completion.get("completion_ids")
                    if (not isinstance(ids, list) or not ids
                            or any(type(t) is not int or t < 0 for t in ids)):
                        raise Stop("p0_correct_completion_ids_missing", id=group.get("id"))
                    sequences.setdefault(labels[group["id"]], set()).add(tuple(ids))
            sequences = {label: [list(seq) for seq in sorted(found)]
                         for label, found in sorted(sequences.items())}
        path = self.ledger.directory / "screen_tokenizations.json"
        write(path, {"schema_version": 1, "sequences": sequences, "switch": switch,
                     "non_canonical_share_of_correct": share,
                     "threshold_exclusive": ALT_TOKENIZATION_SHARE,
                     "rule": "study.json fixed_profile.screening.alt_tokenization_switch"})
        entry = {"path": str(path), "sha256": sha(path), "switch": switch, "share": share,
                 "labels": len(sequences)}
        self.state["screen_tokenizations"] = entry
        self.ledger.save()
        self.ledger.append("screen_tokenizations_frozen", **entry)
        return entry

    def freeze_measured_forecasts(self, g1):
        """Protocol activation_prerequisites: "Freeze per-arm forecast active seconds (for
        the 1.25x caps) from measured pilot timing before the arm starts". Frozen once,
        after G1 and before any RL stage, from P0's measured acceptance a and scorer seconds
        per candidate s (G1's own values), and the frozen tokenization file's mean number
        of scored sequences per S_scan_seed42 row k (1 unless the alt-tokenization switch
        fired: the trainer runs one teacher-forced forward per sequence):
            RL     = N x (grpo_seconds_per_update + s k / a)
            EG-SFT = N x (sft_seconds_per_update + s k)
        with N the final update count and the per-update seconds the plan's measured
        controlled400 values (forecast_sources). The plan's RL and EG-SFT forecasts are the
        study's assumptions and are NOT used for a cap."""
        if "measured_forecasts" in self.state:
            return self.state["measured_forecasts"]
        values = g1.get("values") or {}
        a, s = values.get("acceptance"), values.get("scorer_seconds_per_candidate")
        if not is_number(a) or not 0 < a <= 1 or not is_number(s) or s <= 0:
            raise Stop("g1_timing_inputs_invalid", acceptance=a, scorer_seconds=s)
        tokenizations = json.loads(Path(self.state["screen_tokenizations"]["path"]).read_text())
        canonical = {}
        for group in self.state["completed"]["p0"]["fragment"]["rollouts"]:
            for completion in group["completions"]:
                if (completion.get("correct") is True
                        and completion.get("canonical_tokenization") is True):
                    canonical.setdefault(group["id"], tuple(completion["completion_ids"]))
        labels = {row["id"]: row["label"]
                  for row in read_jsonl(self.g0["manifests"]["manifest_s_scan_seed42"])}
        canonical_by_label = {labels[i]: ids for i, ids in canonical.items() if i in labels}
        extra = {label: sum(tuple(seq) != canonical_by_label.get(label) for seq in seqs)
                 for label, seqs in tokenizations["sequences"].items()}
        k = sum(1 + extra.get(label, 0) for label in labels.values()) / len(labels)
        sources = self.plan["forecast_sources"]
        frozen = {
            "rl": FINAL_UPDATES * (sources["grpo_seconds_per_update"] + s * k / a),
            "egsft": FINAL_UPDATES * (sources["sft_seconds_per_update"] + s * k),
            "inputs": {"acceptance": a, "scorer_seconds_per_candidate": s,
                       "mean_scored_sequences_per_row": k, "updates": FINAL_UPDATES,
                       "grpo_seconds_per_update": sources["grpo_seconds_per_update"],
                       "sft_seconds_per_update": sources["sft_seconds_per_update"]},
            "plan_assumed": {"rl": self.plan["forecast_seconds"]["rl"],
                             "egsft": self.plan["forecast_seconds"]["egsft"]},
            "cap_factor": CAP_FACTOR,
        }
        self.state["measured_forecasts"] = frozen
        self.ledger.save()
        self.ledger.append("forecasts_frozen_from_measured_pilot", **frozen)
        return frozen

    def verify_screen_tokenizations(self):
        entry = self.state.get("screen_tokenizations")
        if entry is not None and (not Path(entry["path"]).is_file()
                                  or sha(entry["path"]) != entry["sha256"]):
            raise Stop("screen_tokenizations_changed", path=entry["path"])

    def fragment(self, run_id):
        entry = self.state["completed"].get(run_id)
        if entry is None:
            raise Stop("selected_endpoint_has_no_fragment", run_id=run_id)
        return entry["fragment"]

    def record_selection(self, name, value):
        if self.state["selections"].get(name, value) != value:
            raise Stop("selection_changed_on_resume", selection=name)
        self.state["selections"][name] = value
        self.ledger.save()
        self.ledger.append("selection", name=name, value=value)

    def final_check(self):
        """Every development gate stays passed with all evidence in; VAL (G4) is next."""
        code, _, report, report_path = self.run_evaluator("final")
        gates = (report or {}).get("gates")
        ok = (code == 0 and isinstance(gates, dict)
              and report.get("failed_at") is None and report.get("awaiting") == "G4"
              and all(isinstance(gates.get(g), dict) and gates[g].get("pass") is True
                      for g in GATE_ORDER))
        self.ledger.append("final_check", report=str(report_path), passed=ok)
        if not ok:
            raise Stop("final_development_check_failed", report=str(report_path))


# ---------------------------------------------------------------- entry


def refused(reason, **detail):
    print(json.dumps({"status": "refused", "reason": reason, **detail}, default=str))
    return 2


INTERRUPTS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def interrupt_once(_signum, _frame):
    """The first SIGTERM/SIGINT/SIGHUP becomes one KeyboardInterrupt; later ones are ignored
    so a second signal cannot abort the stage group's own cleanup (stop_group) half way and
    leave the stage running unsupervised."""
    for signum in INTERRUPTS:
        signal.signal(signum, signal.SIG_IGN)
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan, grid, evaluator = verify_plan(args.plan, args.expected_plan_sha256)
        g0 = verify_g0(plan, evaluator)
    except Stop as stop:
        return refused(stop.reason, **stop.detail)
    except Exception as exc:  # no state exists yet: refuse, never fall through
        return refused("verification_error", error=f"{type(exc).__name__}: {exc}")
    if args.verify_only:
        print(json.dumps({"status": "verified", "g0": g0["checks"], "d_req": g0["d_req"],
                          "grid": grid, "model_calls": 0}, indent=2))
        return 0
    directory = resolve(plan["state_directory"])
    ledger = Ledger(directory)
    if not args.resume:
        # A second owner already present is a refusal, not a stop: nothing has started.
        try:
            owners = second_owners(plan["owner_patterns"])
        except (subprocess.SubprocessError, OSError) as exc:
            return refused("owner_scan_failed", error=f"{type(exc).__name__}: {exc}")
        if owners:
            return refused("second_accelerator_owner_present", processes=owners)
    if args.resume:
        ledger.state = read_json(ledger.state_path)
        if not isinstance(ledger.state, dict):
            return refused("no_state_to_resume")
        if ledger.state.get("plan_sha256") != args.expected_plan_sha256:
            return refused("state_belongs_to_another_plan")
        if ledger.state.get("status") != "running":
            return refused("state_is_terminal", state_status=ledger.state.get("status"))
    else:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return refused("state_directory_exists_single_use")
        ledger.state = {"plan_sha256": args.expected_plan_sha256, "status": "running",
                        "completed": {}, "attempts": {}, "arm_wall_seconds": {},
                        "void_instances": [], "gates": {}, "selections": {},
                        "next_action": "e0", "journal_bytes": 0,
                        "journal_sha256": hashlib.sha256(b"").hexdigest()}
        ledger.save()
    try:
        # Refusals here leave the state untouched: a second launcher must not stop the first.
        ledger.check_journal_prefix()
        lock = acquire_lock(ledger)
    except Stop as stop:
        return refused(stop.reason, **stop.detail)
    if args.resume:
        # A stage group a dead launcher left behind is ours: stop it before the owner scan
        # would count it as a second owner. Anything else present refuses the resume
        # (state untouched, still running) rather than stopping the study.
        reaped = reap_orphan_stage(ledger.state, directory)
        if reaped is not None:
            ledger.append("orphan_stage_reaped", **reaped)
            if reaped["outcome"] == "still_alive":
                lock.unlink(missing_ok=True)
                return refused("orphan_stage_still_alive", **reaped)
            ledger.state.pop("running_stage", None)
            ledger.save()
        try:
            owners = second_owners(plan["owner_patterns"])
        except (subprocess.SubprocessError, OSError) as exc:
            lock.unlink(missing_ok=True)
            return refused("owner_scan_failed", error=f"{type(exc).__name__}: {exc}")
        if owners:
            ledger.append("resume_refused_second_owner", processes=owners)
            ledger.save()
            lock.unlink(missing_ok=True)
            return refused("second_accelerator_owner_present", processes=owners)
    previous = {signum: signal.signal(signum, interrupt_once) for signum in INTERRUPTS}
    try:
        ledger.append("launch_resumed" if args.resume else "launch_started", pid=os.getpid(),
                      plan_sha256=args.expected_plan_sha256, g0=g0["checks"],
                      d_req=g0["d_req"], grid=grid)
        Launcher(plan, args.expected_plan_sha256, grid, evaluator, ledger, g0).sequence()
        ledger.state.update(status="completed", next_action="native-VAL package (G4)")
        ledger.save()
        ledger.append("study_development_complete", awaiting="G4")
        code = 0
    except KeyboardInterrupt:
        ledger.append("launch_interrupted", resumable=True,
                      note="the interrupted stage re-runs from scratch; never resumed")
        ledger.save()
        code = 3
    except Pause as pause:
        ledger.append("launch_paused", resumable=True, reason=pause.reason, **pause.detail)
        ledger.state["next_action"] = (f"paused before {pause.detail.get('stage')}: "
                                       f"{pause.reason}; fix it, then start.py --resume")
        ledger.save()
        code = 3
    except Exception as exc:
        stop = exc if isinstance(exc, Stop) else Stop(
            "unexpected_exception", error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-4000:])
        ledger.append("study_stopped", reason=stop.reason, **stop.detail)
        ledger.state.update(status="stopped", stop_reason=stop.reason,
                            next_action="none: stopped; a re-run needs a new launch")
        ledger.save()
        code = 2
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        lock.unlink(missing_ok=True)
    print(json.dumps({"status": ledger.state["status"],
                      "stop_reason": ledger.state.get("stop_reason"),
                      "next_action": ledger.state["next_action"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
