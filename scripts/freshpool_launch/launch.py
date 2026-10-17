"""Sequence the aircraft-freshpool-screened-grpo-v1 seed-42 development stages, gate by gate.

Study: runs/rl-next-study-design-v1/study.json (wins over study.md). Modelled on
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
frozen environment; a fixed 5 GiB disk reserve the plan may raise but not lower; process-group
RSS; a second owner of the accelerator, checked before every stage and every
OWNER_POLL_SECONDS while one runs; per-arm 1.25x forecast caps, counted from the validated
parent's committed seconds for a continuation plus the wall seconds of every voided instance of
the arm, cumulatively, so neither a continuation nor an interrupt resets them; the rolling
useful-mixed standing stop (< 40 of any 100 accepted updates), enforced live on RL stages by
tailing reward_groups.jsonl under the arm directory and again on the fragment; SFT-E must end
boundary_reached at stop_after_steps 600 so SFT-T can continue it; a survivor-grace check on
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
interval; the owner scan matches command-line substrings, so an owner whose command line names
none of the plan's patterns is invisible; the lock trusts that a dead lock pid was not
reused; the per-arm cap is measured on launcher wall time (an upper bound on active time) and
the trainer receives the same cap as its cumulative active-seconds limit; the fragment
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


class Stop(Exception):
    """A study stop: recorded, terminal, never retried."""

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
    """Rows of (pid, ppid, pgid, stat, rss_bytes, command) from ps; no psutil needed.

    -ww: piped procps output is otherwise cut at 80 columns, which hid an owner pattern
    past that width (measured in CI on test_second_owner_refuses).
    """
    result = subprocess.run(
        ["ps", "-A", "-ww", "-o", "pid=,ppid=,pgid=,stat=,rss=,command="],
        capture_output=True, text=True, timeout=30, check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 5:
            continue
        try:
            pid, ppid, pgid, rss = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[4])
        except ValueError:
            continue
        rows.append((pid, ppid, pgid, parts[3], rss * 1024, parts[5] if len(parts) > 5 else ""))
    return rows


def group_members(pgid):
    """Live (non-zombie) members of a process group."""
    return [row for row in processes() if row[2] == pgid and not row[3].startswith("Z")]


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return bool(group_members(pgid))


def stop_group(child, pgid, grace=15):
    """SIGTERM the stage's process group, then SIGKILL whatever is left."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if child.poll() is not None and not group_alive(pgid):
                break
            time.sleep(0.1)
    child.wait()


def second_owners(patterns, own_groups=()):
    """Processes matching an owner pattern other than this launcher, its ancestors, and its
    running stage: the members of ``own_groups`` and their descendants. Any other child of
    the launcher's own process is NOT exempt (a real launcher has none while a stage runs)."""
    rows = processes()
    parents = {pid: ppid for pid, ppid, *_ in rows}
    mine, pid = set(), os.getpid()
    while pid and pid not in mine:
        mine.add(pid)
        pid = parents.get(pid)
    children = {}
    for pid, ppid, *_ in rows:
        children.setdefault(ppid, []).append(pid)
    stack = [pid for pid, _, pgid, *_ in rows if pgid in own_groups]
    stage = set(stack)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in stage:
                stage.add(child)
                stack.append(child)
    mine |= stage
    return [{"pid": pid, "command": command[:200]}
            for pid, _, _, stat, _, command in rows
            if pid not in mine and not stat.startswith("Z")
            and any(p in command for p in patterns)]


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
    return plan, grid, evaluator


def verify_pins(plan):
    for name, expected in plan["file_sha256"].items():
        path = resolve(name)
        if not path.is_file() or sha(path) != expected:
            raise Stop("pinned_file_changed", file=name)


def verify_manifest_content(directory, manifest, base, evaluator):
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
    """G0 'protocol revision committed': the activated freshpool revision, committed, whose
    frozen gates equal those of the committed candidate it names."""
    protocol = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != PROTOCOL_ID:
        raise Stop("g0_protocol_revision_wrong_id")
    if protocol.get("activated") is not True:
        raise Stop("g0_protocol_revision_not_activated")
    revises = protocol.get("revises") or {}
    candidate_path = resolve(revises.get("path", "/nonexistent"))
    if not candidate_path.is_file() or sha(candidate_path) != revises.get("sha256"):
        raise Stop("g0_protocol_candidate_missing_or_changed")
    candidate = read_json(candidate_path)
    if not isinstance(candidate, dict) or any(
            key not in protocol or protocol[key] != candidate.get(key)
            for key in FROZEN_PROTOCOL_KEYS):
        raise Stop("g0_protocol_frozen_gates_changed")
    if not git_committed(path) or not git_committed(candidate_path):
        raise Stop("g0_protocol_not_committed")


def retirement_entry(line):
    """True only for an explicit 'Retired:' line naming the seed-2718, 1,600-image 16-shot
    plan, with no hedge ('not', 'pending', 'proposed', ...)."""
    return bool(RETIRED_LINE.match(line)) and all(f.search(line) for f in RETIRED_FACTS) and (
        not RETIRED_HEDGE.search(line))


def verify_decision_record(path):
    """G0 '16-shot plan retired': a committed decision record with an explicit retirement
    entry for the seed-2718, 1,600-image plan."""
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    if not any(retirement_entry(line) for line in lines):
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
    content = verify_manifest_content(directory, manifest, base, evaluator)
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
        if self.path is None:
            found = sorted(self.arm_dir.rglob("reward_groups.jsonl"))
            if len(found) > 1:
                return "reward_groups_ambiguous"
            if not found:
                return None
            self.path = found[0]
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


def acquire_lock(ledger):
    lock = ledger.directory / "lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = int(lock.read_text().strip())
        except ValueError:
            raise Stop("lock_unreadable") from None
        try:
            os.kill(holder, 0)
            raise Stop("another_launcher_holds_the_lock", pid=holder)
        except ProcessLookupError:
            ledger.append("stale_lock_removed", pid=holder)
            lock.unlink()
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except PermissionError:
            raise Stop("another_launcher_holds_the_lock", pid=holder) from None
    os.write(descriptor, str(os.getpid()).encode())
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
            inherited_seconds=0.0, watch_mixed=False):
        """Run one stage in a fresh instance. ``inherited_seconds`` is the validated parent
        endpoint's committed active seconds for a continuation, so the arm cap counts the
        whole arm and is never reset. Wall seconds spent in this arm's voided (interrupted)
        instances are charged on top, cumulatively: the arm's cap is never refilled by an
        interrupt. ``watch_mixed`` enforces the rolling useful-mixed standing stop live."""
        if stage_id in self.state["completed"]:
            return self.state["completed"][stage_id]["fragment"]
        template = self.plan["templates"][template_name]
        verify_pins(self.plan)
        free = self.check_disk()
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
        voided = voids.get(arm, 0.0)
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
            "seed": str(SEED), "active_cap": repr(cap - voided),
            "w_adapter": self.plan["w_endpoint"][
                "adapter_path"], **self.g0["manifests"], **values,
        }
        try:
            forwarded = [part.format_map(substitutions) for part in template["argv"]]
        except (KeyError, ValueError, IndexError) as exc:
            raise Stop("template_placeholder_unresolved", stage=stage_id, error=str(exc)) from None
        command = [
            self.plan["python"], "-B", str(HERE / "phase_worker.py"),
            "--guard-dir", str(self.ledger.directory / "guards"), "--stage", stage_id,
            "--kind", template["kind"], "--entry", template["entry"],
            "--entry-sha256", template["entry_sha256"], "--", *forwarded,
        ]
        self.state["attempts"][stage_id] = instance
        self.state["arm_wall_seconds"][stage_id] = used
        self.state["instance_start_seconds"][stage_id] = used
        self.state["next_action"] = stage_id
        self.ledger.save()
        self.ledger.append("stage_started", stage=stage_id, arm=arm, instance=instance,
                           command=command, cap_seconds=cap, inherited_seconds=inherited_seconds,
                           void_seconds=voided, free_disk_bytes=free,
                           owner_poll_seconds=OWNER_POLL_SECONDS, live_useful_mixed=watch_mixed)
        monitor = MixedMonitor(arm_dir) if watch_mixed else None
        outcome = self.supervise(stage_id, command, stage_dir, used, cap, monitor)
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

    def supervise(self, stage_id, command, stage_dir, used, cap, monitor=None):
        started, peak, reason = time.monotonic(), 0, None
        last_save = last_owner = started
        with (stage_dir / "stage.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
            pgid = child.pid
            try:
                while child.poll() is None:
                    rss = sum(row[4] for row in group_members(pgid))
                    peak = max(peak, rss)
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
                self.ledger.save()
                raise
        elapsed = time.monotonic() - started
        self.state["arm_wall_seconds"][stage_id] = used + elapsed
        self.ledger.save()
        return {"exit_code": code, "reason": reason, "elapsed_seconds": elapsed,
                "arm_wall_seconds": used + elapsed, "peak_group_rss_bytes": peak}

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
                for start in range(0, max(0, len(mixed) - 99)):
                    if sum(mixed[start:start + 100]) / 100 < 0.40:
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
        checks = f.get("checks")
        required = ("resume_2_plus_2_identical", "resume_1_plus_3_identical",
                    "reference_unchanged", "finite_nonzero_task_gradient",
                    "scorer_uses_sampler_transform")
        if not isinstance(checks, dict) or any(checks.get(k) is not True for k in required):
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
    def sft_t_check(parent, t_rl):
        """SFT-T continues SFT-E without reset and stops at the first step at or above
        T_RL: the step before it is still below T_RL."""
        def check(f):
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
        self.gate("G1")

        rl = {}
        for lr in self.grid:
            arm = f"rl-{lr_tag(lr)}-s{SEED}"
            run_id = f"{arm}-u{RL_SCREEN_UPDATES}"
            rl[run_id] = self.run(run_id, "rl150", arm, forecast["rl"],
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
            final = self.run(run_id, "rl600", arm, forecast["rl"],
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
                                     check=self.sft_e_boundary_check(FINAL_UPDATES)))
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
                                self.train_validator(run_id, "SFT-T", lr,
                                                     check=self.sft_t_check(parent, t_rl)),
                                inherited_seconds=parent["committed_active_seconds"])
            self.assay_fragment(fragment, "h_select")
        _, sft_t_star = self.study().select_sft_t()
        self.record_selection("sft_t_star", sft_t_star["run_id"])
        self.assay_fragment(self.fragment(sft_t_star["run_id"]), "h_gate")
        self.gate("G3b")

        lr = sft_e_star["lr"]
        run_id = f"egsft-{lr_tag(lr)}-s{SEED}"
        eg = self.run(run_id, "egsft", run_id, forecast["egsft"],
                      {"run_id": run_id, "lr": repr(lr),
                       "accepted_manifest": rl_star["accepted_manifest"]},
                      self.train_validator(run_id, "EG-SFT", lr, FINAL_UPDATES))
        self.assay_fragment(eg, "h_gate")
        self.final_check()

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
    previous = signal.signal(signal.SIGTERM,
                             lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
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
        signal.signal(signal.SIGTERM, previous)
        lock.unlink(missing_ok=True)
    print(json.dumps({"status": ledger.state["status"],
                      "stop_reason": ledger.state.get("stop_reason"),
                      "next_action": ledger.state["next_action"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
