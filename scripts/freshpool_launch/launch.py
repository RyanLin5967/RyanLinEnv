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
RSS; a second owner of the accelerator; per-arm 1.25x forecast caps that accumulate across
resumes and are never reset; a survivor-grace check on every stage's process group. State is
single-use: a stopped or completed study never runs again, and an interrupted one resumes only
with ``--resume`` against the same plan, with its journal prefix intact. The journal is
append-only and fsynced.

Blind spots, stated: process-group membership is the ownership boundary, so a descendant that
calls setsid() escapes the RSS sum and the survivor check; RSS and disk are sampled every
``poll_seconds``; the owner scan matches command-line substrings, so an owner whose command
line names none of the plan's patterns is invisible; the lock trusts that a dead lock pid was
not reused; the per-arm cap is measured on launcher wall time (an upper bound on active time)
and the trainer receives the same cap as its cumulative active-seconds limit; the fragment
contents (predictions, adapter hashes, active seconds) are trusted as the stage wrote them.
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GiB = 1024**3
DISK_RESERVE_BYTES = 5 * GiB  # study.json standing_stops: addendum-5 5 GiB reserve
CAP_FACTOR = 1.25  # study.json budget_matching.caps
SEED = 42
GRID = (1e-5, 5e-5, 1e-4)
B4_LR = 2e-4
RL_SCREEN_UPDATES, FINAL_UPDATES = 150, 600
SPLITS = ("h_select", "h_gate")
TRAIN_TEMPLATES = ("e0", "rl150", "rl600", "sft_e", "sft_t", "egsft")
TEMPLATES = (*TRAIN_TEMPLATES, "assay", "p0")
FORECASTS = ("e0", "p0", "assay_h_select", "assay_h_gate", "rl", "sft_e", "egsft")
REQUIRED_MANIFESTS = ("h_select.jsonl", "h_gate.jsonl", "s_scan_seed42.jsonl")
RECEIPTS = ("protocol_revision", "decision_record", "branch")
HEX64 = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER = re.compile(r"\{[^{}]*\}")
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


def free_bytes(path):
    return shutil.disk_usage(path).free


def lr_tag(lr):
    return f"{lr:.0e}"


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
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise Stop("limit_missing_or_invalid", limit=name)
    forecasts = plan.get("forecast_seconds") or {}
    for name in FORECASTS:
        value = forecasts.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
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


def verify_g0(plan, evaluator):
    """G0: manifests hashed, audit forced fire passed, receipts committed, branch resolved."""
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
    base = read_json(resolve(plan["evidence_base"]["path"]))
    branch = read_json(resolve(receipts["branch"]["path"]))
    if not isinstance(base, dict) or not isinstance(branch, dict):
        raise Stop("g0_branch_or_evidence_unreadable")
    if sha(resolve(plan["evidence_base"]["path"])) != plan["evidence_base"].get("sha256"):
        raise Stop("evidence_base_changed")
    try:
        d = evaluator["d_req"](branch["a_star"])
        checks = {
            "a_star_matches_evidence": branch["a_star"] == base.get("a_star"),
            "b4_matches_plan": branch["b4_edge_extension"] == plan["b4_edge_extension"]
            == base.get("b4_edge_extension"),
            "B0_w_valid_eos": branch["w_valid_eos"] >= 0.95,
            "B1_w_val_in_range": 0.15 <= branch["w_val_accuracy"] <= 0.85,
            "d_req_published": abs(branch["d_req"] - d["d_req"]) <= 1e-9,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise Stop("g0_branch_receipt_invalid", error=str(exc)) from None
    if not all(checks.values()):
        raise Stop("g0_failed", checks=checks)
    paths = {f"manifest_{Path(name).stem}": str(directory / name) for name in files}
    return {"checks": checks, "d_req": d, "manifests": paths}


# ---------------------------------------------------------------- state and journal


class Ledger:
    """state.json (resume pointer) plus an append-only, fsynced journal.jsonl."""

    def __init__(self, directory):
        self.directory = directory
        self.state_path = directory / "state.json"
        self.journal_path = directory / "journal.jsonl"
        self.state = None

    def append(self, event, **fields):
        line = json.dumps({"event": event, "time": time.time(), **fields}, allow_nan=False)
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

    # -- guards
    def check_disk(self):
        free = free_bytes(self.ledger.directory)
        if free < self.limits["minimum_free_bytes"]:
            raise Stop("disk_reserve", free_bytes=free,
                       minimum_free_bytes=self.limits["minimum_free_bytes"])
        return free

    def check_owners(self):
        mine, rows = set(), processes()
        parents = {pid: ppid for pid, ppid, *_ in rows}
        pid = os.getpid()
        while pid and pid not in mine:
            mine.add(pid)
            pid = parents.get(pid)
        found = [{"pid": pid, "command": command[:200]} for pid, _, _, stat, _, command in rows
                 if pid not in mine and not stat.startswith("Z")
                 and any(p in command for p in self.plan["owner_patterns"])]
        if found:
            raise Stop("second_accelerator_owner", processes=found)

    def cap(self, arm, forecast):
        return CAP_FACTOR * forecast

    # -- running one stage
    def run(self, stage_id, template_name, arm, forecast, values, validate):
        if stage_id in self.state["completed"]:
            return self.state["completed"][stage_id]["fragment"]
        template = self.plan["templates"][template_name]
        verify_pins(self.plan)
        free = self.check_disk()
        self.check_owners()
        cap = self.cap(arm, forecast)
        used = self.state["arm_wall_seconds"].get(arm, 0.0)
        if used >= cap:
            raise Stop("arm_cap", arm=arm, used_seconds=used, cap_seconds=cap)
        stage_dir = self.ledger.directory / "stages" / stage_id
        arm_dir = self.ledger.directory / "arms" / arm
        attempts = self.state["attempts"].get(stage_id, 0)
        argv = template["argv"]
        if attempts:
            if template["kind"] == "train":
                checkpoints = sorted(
                    arm_dir.glob("**/checkpoint-*"),
                    key=lambda p: int(p.name.rsplit("-", 1)[1])
                    if p.name.rsplit("-", 1)[1].isdigit() else -1)
                if "resume_argv" not in template or not checkpoints:
                    raise Stop("interrupted_arm_has_no_resume_checkpoint", stage=stage_id)
                argv = template["resume_argv"]
                values = {**values, "latest_checkpoint": str(checkpoints[-1])}
            self.ledger.append("stage_resumed", stage=stage_id, attempt=attempts + 1,
                               arm_wall_seconds=used)
        attempt_dir = stage_dir / f"attempt-{attempts + 1}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        arm_dir.mkdir(parents=True, exist_ok=True)
        fragment = attempt_dir / "fragment.json"
        substitutions = {
            "python": self.plan["python"], "stage_id": stage_id, "arm": arm,
            "stage_dir": str(attempt_dir), "arm_dir": str(arm_dir), "fragment": str(fragment),
            "seed": str(SEED), "active_cap": repr(cap), "w_adapter": self.plan["w_endpoint"][
                "adapter_path"], **self.g0["manifests"], **values,
        }
        try:
            forwarded = [part.format_map(substitutions) for part in argv]
        except (KeyError, ValueError, IndexError) as exc:
            raise Stop("template_placeholder_unresolved", stage=stage_id, error=str(exc)) from None
        command = [
            self.plan["python"], "-B", str(HERE / "phase_worker.py"),
            "--guard-dir", str(self.ledger.directory / "guards"), "--stage", stage_id,
            "--kind", template["kind"], "--entry", template["entry"],
            "--entry-sha256", template["entry_sha256"], "--", *forwarded,
        ]
        self.state["attempts"][stage_id] = attempts + 1
        self.state["next_action"] = stage_id
        self.ledger.save()
        self.ledger.append("stage_started", stage=stage_id, arm=arm, attempt=attempts + 1,
                           command=command, cap_seconds=cap, arm_wall_seconds=used,
                           free_disk_bytes=free)
        outcome = self.supervise(stage_id, arm, command, attempt_dir, used, cap)
        self.ledger.append("stage_finished", stage=stage_id, **outcome)
        if outcome["reason"]:
            raise Stop(outcome["reason"], stage=stage_id)
        if outcome["exit_code"] != 0:
            raise Stop("stage_failed", stage=stage_id, exit_code=outcome["exit_code"])
        payload = read_json(fragment)
        if not isinstance(payload, dict):
            raise Stop("stage_fragment_missing", stage=stage_id)
        problem = validate(payload, cap)
        if problem:
            raise Stop(problem, stage=stage_id)
        self.state["completed"][stage_id] = {"fragment": payload, "path": str(fragment)}
        self.ledger.save()
        self.ledger.append("stage_completed", stage=stage_id, fragment=str(fragment),
                           fragment_sha256=sha(fragment))
        return payload

    def supervise(self, stage_id, arm, command, attempt_dir, used, cap):
        started, peak, reason = time.monotonic(), 0, None
        last_save = started
        with (attempt_dir / "stage.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
            pgid = child.pid
            try:
                while child.poll() is None:
                    rss = sum(row[4] for row in group_members(pgid))
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - started
                    self.state["arm_wall_seconds"][arm] = used + elapsed
                    if time.monotonic() - last_save >= 5:
                        self.ledger.save()
                        last_save = time.monotonic()
                    free = free_bytes(self.ledger.directory)
                    reason = (
                        "process_group_rss_limit" if rss > self.limits["rss_limit_bytes"]
                        else "disk_reserve" if free < self.limits["minimum_free_bytes"]
                        else "arm_cap" if used + elapsed >= cap
                        else None
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
                self.state["arm_wall_seconds"][arm] = used + time.monotonic() - started
                self.ledger.save()
                raise
        elapsed = time.monotonic() - started
        self.state["arm_wall_seconds"][arm] = used + elapsed
        self.ledger.save()
        return {"exit_code": code, "reason": reason, "elapsed_seconds": elapsed,
                "arm_wall_seconds": used + elapsed, "peak_group_rss_bytes": peak}

    # -- fragment validation
    def train_validator(self, run_id, arm_name, lr, updates=None, minimum_seconds=None,
                        rl=False):
        w = self.plan["w_endpoint"]["adapter_sha256"]

        def validate(f, cap):
            seconds = f.get("committed_active_seconds")
            if f.get("status") != "completed":
                return "arm_not_completed"
            if (f.get("run_id"), f.get("arm"), f.get("lr"), f.get("seed")) != (
                    run_id, arm_name, lr, SEED):
                return "fragment_identity_differs"
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                return "fragment_active_seconds_invalid"
            if seconds > cap:
                return "arm_cap"
            if updates is not None and f.get("updates") != updates:
                return "fragment_updates_differ"
            if type(f.get("updates")) is not int or f["updates"] < 1:
                return "fragment_updates_invalid"
            if minimum_seconds is not None and seconds < minimum_seconds - 1e-9:
                return "sft_t_short_of_t_rl"
            if not HEX64.fullmatch(str(f.get("adapter_sha256"))) or f["adapter_sha256"] == w:
                return "fragment_adapter_unchanged_or_invalid"
            if not HEX64.fullmatch(str(f.get("reference_sha256"))):
                return "fragment_reference_invalid"
            if rl and f["reference_sha256"] != w:
                return "reference_changed"
            if not f.get("checkpoint") or not Path(f["checkpoint"]).exists():
                return "fragment_checkpoint_missing"
            mixed = f.get("useful_mixed")
            if not isinstance(mixed, list) or (rl and len(mixed) != f["updates"]):
                return "fragment_useful_mixed_invalid"
            if rl:
                for key in ("accepted_manifest", "scanned_manifest"):
                    if not f.get(key) or not Path(f[key]).is_file():
                        return f"fragment_{key}_missing"
                for start in range(0, max(0, len(mixed) - 99)):
                    if sum(mixed[start:start + 100]) / 100 < 0.40:
                        return "rolling_useful_mixed_below_0_40"
            return None

        return validate

    def assay_validator(self, run_id, split, adapter):
        def validate(f, cap):
            if (f.get("run_id"), f.get("split"), f.get("adapter_sha256")) != (
                    run_id, split, adapter):
                return "assay_identity_differs"
            if not isinstance(f.get("predictions"), dict) or not f["predictions"]:
                return "assay_predictions_missing"
            eos = f.get("eos_fraction")
            if type(eos) not in (int, float) or not 0 <= eos <= 1:
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
        if type(agreement) not in (int, float) or not 0 <= agreement <= 0.001:
            return "e0_sampler_teacher_agreement_failed"
        return None

    @staticmethod
    def p0_validator(f, cap):
        if not all(k in f for k in ("seed", "adapter_sha256", "scan_order", "journal",
                                    "rollouts")):
            return "p0_g1_document_incomplete"
        return None

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

    def gate(self, name):
        if name in self.state["gates"]:
            return self.state["gates"][name]["result"]
        verify_pins(self.plan)
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
        result = (report or {}).get("gates", {}).get(name) if isinstance(report, dict) else None
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
        return self.assay(fragment["run_id"], split, fragment["checkpoint"],
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
                                                       rl=True))
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
                             self.train_validator(run_id, "RL", lr, FINAL_UPDATES, rl=True))
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
                                 self.train_validator(run_id, "SFT-E", lr, FINAL_UPDATES))
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

        t_rl = rl_star["committed_active_seconds"]
        for lr in qualifying:
            run_id = f"sft-t-{lr_tag(lr)}-s{SEED}"
            fragment = self.run(run_id, "sft_t", run_id, t_rl,
                                {"run_id": run_id, "lr": repr(lr), "t_rl": repr(t_rl),
                                 "scanned_manifest": rl_star["scanned_manifest"],
                                 "parent_checkpoint": sft_e[lr]["checkpoint"]},
                                self.train_validator(run_id, "SFT-T", lr,
                                                     minimum_seconds=t_rl))
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
        directory = self.ledger.directory / "gates"
        attempt = len(list(directory.glob("final-report-*.json"))) + 1
        evidence = directory / f"final-evidence-{attempt}.json"
        report_path = directory / f"final-report-{attempt}.json"
        write(evidence, self.evidence())
        completed = subprocess.run(
            [self.plan["python"], "-B", str(resolve(self.plan["evaluator"])),
             "--evidence", str(evidence), "--output", str(report_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=1800)
        report = read_json(report_path)
        ok = (completed.returncode == 0 and isinstance(report, dict)
              and report.get("failed_at") is None and report.get("awaiting") == "G4"
              and all(report["gates"].get(g, {}).get("pass") is True for g in GATE_ORDER))
        self.ledger.append("final_check", report=str(report_path), passed=ok)
        if not ok:
            raise Stop("final_development_check_failed", report=str(report_path))


# ---------------------------------------------------------------- entry


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
        print(json.dumps({"status": "refused", "reason": stop.reason, **stop.detail},
                         default=str))
        return 2
    if args.verify_only:
        print(json.dumps({"status": "verified", "g0": g0["checks"], "d_req": g0["d_req"],
                          "grid": grid, "model_calls": 0}, indent=2))
        return 0
    directory = resolve(plan["state_directory"])
    ledger = Ledger(directory)
    if args.resume:
        ledger.state = read_json(ledger.state_path)
        if not isinstance(ledger.state, dict):
            print(json.dumps({"status": "refused", "reason": "no_state_to_resume"}))
            return 2
        if ledger.state.get("plan_sha256") != args.expected_plan_sha256:
            print(json.dumps({"status": "refused", "reason": "state_belongs_to_another_plan"}))
            return 2
        if ledger.state.get("status") != "running":
            print(json.dumps({"status": "refused", "reason": "state_is_terminal",
                              "state_status": ledger.state.get("status")}))
            return 2
    else:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(json.dumps({"status": "refused",
                              "reason": "state_directory_exists_single_use"}))
            return 2
        ledger.state = {"plan_sha256": args.expected_plan_sha256, "status": "running",
                        "completed": {}, "attempts": {}, "arm_wall_seconds": {},
                        "gates": {}, "selections": {}, "next_action": "e0",
                        "journal_bytes": 0, "journal_sha256": hashlib.sha256(b"").hexdigest()}
        ledger.save()
    try:
        # Refusals here leave the state untouched: a second launcher must not stop the first.
        ledger.check_journal_prefix()
        lock = acquire_lock(ledger)
    except Stop as stop:
        print(json.dumps({"status": "refused", "reason": stop.reason, **stop.detail}))
        return 2
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
    except Stop as stop:
        ledger.append("study_stopped", reason=stop.reason, **stop.detail)
        ledger.state.update(status="stopped", stop_reason=stop.reason,
                            next_action="none: stopped; a re-run needs a new launch")
        ledger.save()
        code = 2
    except KeyboardInterrupt:
        ledger.append("launch_interrupted", resumable=True)
        ledger.save()
        code = 3
    finally:
        signal.signal(signal.SIGTERM, previous)
        lock.unlink(missing_ok=True)
    print(json.dumps({"status": ledger.state["status"],
                      "stop_reason": ledger.state.get("stop_reason"),
                      "next_action": ledger.state["next_action"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
