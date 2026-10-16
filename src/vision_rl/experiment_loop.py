"""Resumable experiment search with immutable plans and a single final test.

This controller executes explicitly supplied argv; it is not a security sandbox.
Training commands must themselves enforce data separation and API budgets. Gates
are numerical requirements, not evidence that a benchmark or its labels are valid.

Plan format (all command output paths must be under ``{candidate_dir}``)::

    {
      "version": 1,
      "workspace": ".",
      "inputs": ["data/train.jsonl", "data/validation.jsonl", "data/test.jsonl"],
      "budget": {"max_attempts": 8, "max_wall_seconds": 7200,
                 "command_timeout_seconds": 1200},
      "selection_metric": "accuracy",
      "validation_gate": {"accuracy": {"min": 0.95}},
      "test_gate": {"accuracy": {"min": 0.95}, "margin_ci95.0": {"min": 0.1}},
      "candidates": [{
        "id": "seed-7",
        "train": ["{python}", "train.py", "--out", "{candidate_dir}"],
        "validate": ["{python}", "validate.py", "--out", "{candidate_dir}"],
        "evaluate": ["{python}", "evaluate.py", "--out", "{candidate_dir}"],
        "artifacts": ["{candidate_dir}/policy.pt"],
        "validation_metrics": "{candidate_dir}/validation.json",
        "test_metrics": "{candidate_dir}/test.json"
      }]
    }

Selection maximizes a validation metric after all budgeted candidates finish.
Declare baseline margins, confidence bounds, and sample-size gates in the plan.
An interrupted/failed final test consumes the test; resume never reruns it.
``evaluation_mode: "development"`` freezes the selected artifacts at
``awaiting_final_evaluation`` without executing or consuming a final test.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TERMINAL = {
    "completed",
    "needs_fresh_test",
    "test_interrupted",
    "validation_gate_not_met",
    "budget_exhausted",
    "integrity_error",
    "awaiting_final_evaluation",
}
_MAX_JSON_BYTES = 10 * 1024 * 1024


class LoopError(ValueError):
    """A plan, integrity, or concurrency condition prevents execution."""


class IntegrityError(LoopError):
    """A previously frozen input, checkpoint, or validation result changed."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> Any:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise LoopError(f"JSON exceeds 10 MiB: {path}")
    return json.loads(path.read_text(), parse_constant=lambda v: _bad_constant(v))


def _bad_constant(value: str) -> None:
    raise LoopError(f"non-finite JSON constant: {value}")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # Persist the rename, including the one-shot test-consumption marker, before
    # launching a worker. A process crash must never reopen the final holdout.
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _hash_path(path: Path) -> str:
    if path.is_symlink():
        raise LoopError(f"symlinks cannot be frozen: {path}")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        entries = {}
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise LoopError(f"symlinks cannot be frozen: {child}")
            if child.is_file():
                entries[child.relative_to(path).as_posix()] = _hash_path(child)
            elif not child.is_dir():
                raise LoopError(f"only regular files can be frozen: {child}")
        if not entries:
            raise LoopError(f"cannot freeze an empty directory: {path}")
        return _digest(entries)
    raise LoopError(f"missing regular file or directory: {path}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LoopError(f"{name} must be a finite number")
    return float(value)


def _metric(metrics: Any, key: str) -> float:
    value = metrics
    try:
        for component in key.split("."):
            value = value[int(component)] if isinstance(value, list) else value[component]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise LoopError(f"missing metric: {key}") from exc
    return _number(value, f"metric {key}")


def _gates(metrics: Any, gates: dict) -> tuple[bool, dict]:
    checks = {}
    for key, bounds in gates.items():
        value = _metric(metrics, key)
        passed = ("min" not in bounds or value >= bounds["min"]) and (
            "max" not in bounds or value <= bounds["max"]
        )
        checks[key] = {"value": value, **bounds, "passed": passed}
    return all(check["passed"] for check in checks.values()), checks


def validate_plan(plan: dict, base: Path) -> dict:
    """Normalize a plan without executing commands or reading dataset contents."""
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise LoopError("plan version must be 1")
    plan = json.loads(_canonical(plan))
    mode = plan.setdefault("evaluation_mode", "automatic")
    if mode not in {"automatic", "development"}:
        raise LoopError("evaluation_mode must be automatic or development")
    workspace = Path(plan.get("workspace", str(base))).expanduser()
    workspace = (base / workspace).resolve() if not workspace.is_absolute() else workspace.resolve()
    if not workspace.is_dir():
        raise LoopError("workspace must be an existing directory")
    plan["workspace"] = str(workspace)
    # Preserve the chosen virtual environment when another interpreter resumes
    # the controller. Resolving the executable symlink would discard the venv.
    plan["python_executable"] = str(Path(sys.executable).absolute())
    inputs = plan.get("inputs")
    if not isinstance(inputs, list) or not inputs or any(not isinstance(p, str) for p in inputs):
        raise LoopError("inputs must pin at least one dataset, protocol, or source path")
    plan["inputs"] = [str((workspace / p).absolute()) for p in inputs]
    if len(set(plan["inputs"])) != len(inputs):
        raise LoopError("duplicate input paths")
    budget = plan.get("budget", {})
    attempts = budget.get("max_attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 100000:
        raise LoopError("budget.max_attempts must be an integer in [1, 100000]")
    for key in ("max_wall_seconds", "command_timeout_seconds"):
        if _number(budget.get(key), f"budget.{key}") <= 0:
            raise LoopError(f"budget.{key} must be positive")
    if not isinstance(plan.get("selection_metric"), str) or not plan["selection_metric"]:
        raise LoopError("selection_metric must be a non-empty JSON metric path")
    for name in ("validation_gate", "test_gate"):
        if name == "test_gate" and mode == "development" and not plan.get(name):
            plan[name] = {}
            continue
        gate = plan.get(name)
        if not isinstance(gate, dict) or not gate:
            raise LoopError(f"{name} must contain predeclared metric bounds")
        for key, bounds in gate.items():
            if (
                not key
                or not isinstance(bounds, dict)
                or not bounds
                or set(bounds) - {"min", "max"}
            ):
                raise LoopError(f"invalid {name} metric bounds")
            for bound, value in bounds.items():
                _number(value, f"{name}.{key}.{bound}")
            if bounds.get("min", -math.inf) > bounds.get("max", math.inf):
                raise LoopError(f"inverted {name} bounds")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LoopError("candidates must be a non-empty list")
    identifiers = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise LoopError("each candidate must be an object")
        identifier = candidate.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier
        ):
            raise LoopError("candidate IDs must be 1-80 safe filename characters")
        if identifier in identifiers:
            raise LoopError("candidate IDs must be unique")
        identifiers.add(identifier)
        stages = ("train", "validate", "evaluate") if mode == "automatic" else ("train", "validate")
        for stage in stages:
            command = candidate.get(stage)
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(arg, str) or "\0" in arg for arg in command)
                or not command[0]
            ):
                raise LoopError(f"{identifier}.{stage} must be a non-empty argv list")
        outputs = [candidate.get("validation_metrics")]
        if mode == "automatic":
            outputs.append(candidate.get("test_metrics"))
        artifacts = candidate.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise LoopError(f"{identifier}.artifacts must contain checkpoint paths")
        outputs.extend(artifacts)
        if any(not isinstance(p, str) or not p.startswith("{candidate_dir}/") for p in outputs):
            raise LoopError("all output paths must start with {candidate_dir}/")
        if mode == "automatic" and candidate["validation_metrics"] == candidate["test_metrics"]:
            raise LoopError("validation and test metrics must be separate files")
    return plan


@contextlib.contextmanager
def _lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LoopError("another runner or its worker holds this experiment lock") from exc
        yield handle.fileno()


def _check_events(path: Path) -> tuple[int, str]:
    previous, count = "0" * 64, 0
    if path.exists():
        with path.open() as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                    claimed = event.pop("sha256")
                    if event["sequence"] != count or event["previous"] != previous:
                        raise LoopError("event log sequence was modified")
                    if claimed != _digest(event):
                        raise LoopError("event log checksum was modified")
                except (ValueError, KeyError, TypeError) as exc:
                    raise LoopError(
                        "event log is incomplete or modified; preserve it for audit"
                    ) from exc
                previous, count = claimed, count + 1
    return count, previous


class _Runner:
    def __init__(self, state_dir: Path, lock_fd: int):
        self.directory = state_dir
        self.lock_fd = lock_fd
        self.plan = _read_json(state_dir / "plan.json")
        expected = (state_dir / "plan.sha256").read_text().strip()
        if _digest(self.plan) != expected:
            raise LoopError("immutable plan checksum mismatch")
        self.state = _read_json(state_dir / "state.json")
        if self.state["plan_sha256"] != expected:
            raise LoopError("state belongs to a different plan")
        self.sequence, self.previous = _check_events(state_dir / "events.jsonl")

    def event(self, name: str, **details: Any) -> None:
        if set(details) & {"sequence", "previous", "time", "event", "sha256"}:
            raise LoopError("event details cannot replace hash-chain metadata")
        value = {
            "sequence": self.sequence,
            "previous": self.previous,
            "time": time.time(),
            "event": name,
            **details,
        }
        digest = _digest(value)
        with (self.directory / "events.jsonl").open("ab") as handle:
            handle.write(_canonical({**value, "sha256": digest}))
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence, self.previous = self.sequence + 1, digest

    def save(self) -> None:
        self.state["heartbeat_at"] = time.time()
        _atomic_json(self.directory / "state.json", self.state)

    def integrity(self, hashes: dict[str, str] | None = None) -> None:
        for name, expected in (
            hashes if hashes is not None else self.state["input_hashes"]
        ).items():
            try:
                current = _hash_path(Path(name))
            except (OSError, LoopError) as exc:
                raise IntegrityError(f"frozen input or artifact disappeared: {name}") from exc
            if current != expected:
                raise IntegrityError(f"frozen input or artifact changed: {name}")

    def remaining(self) -> float:
        return self.state["created_at"] + self.plan["budget"]["max_wall_seconds"] - time.time()

    def substitute(self, value: str, candidate: dict, attempt: dict) -> str:
        replacements = {
            "{python}": self.plan["python_executable"],
            "{workspace}": self.plan["workspace"],
            "{state_dir}": str(self.directory),
            "{candidate_id}": candidate["id"],
            "{candidate_dir}": attempt["directory"],
        }
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        return value

    def output(self, value: str, candidate: dict, attempt: dict) -> Path:
        path = Path(self.substitute(value, candidate, attempt)).absolute()
        root = Path(attempt["directory"]).resolve()
        if not path.resolve().is_relative_to(root) or path.resolve() == root:
            raise LoopError("output escaped its candidate directory")
        return path

    def artifact_hashes(self, candidate: dict, attempt: dict) -> dict[str, str]:
        return {
            str(path): _hash_path(path)
            for path in (self.output(name, candidate, attempt) for name in candidate["artifacts"])
        }

    def command(self, candidate: dict, attempt: dict, stage: str) -> dict:
        timeout = min(self.remaining(), self.plan["budget"]["command_timeout_seconds"])
        if timeout <= 0:
            return {"returncode": None, "reason": "wall_budget_exhausted", "seconds": 0.0}
        argv = [self.substitute(arg, candidate, attempt) for arg in candidate[stage]]
        prefix = self.directory / "logs" / f"{candidate['id']}-a{attempt['number']:04d}-{stage}"
        prefix.parent.mkdir(exist_ok=True)
        self.event(
            "command_started",
            candidate=candidate["id"],
            attempt=attempt["number"],
            stage=stage,
            argv=argv,
            stdout=str(prefix) + ".stdout.log",
            stderr=str(prefix) + ".stderr.log",
        )
        environment = {
            **os.environ,
            "LOOP_STATE_DIR": str(self.directory),
            "LOOP_CANDIDATE_ID": candidate["id"],
            "LOOP_STAGE": stage,
            "LOOP_CANDIDATE_DIR": attempt["directory"],
        }
        started = time.monotonic()
        process = None
        reason = "exited"
        try:
            with Path(str(prefix) + ".stdout.log").open("xb") as out:
                with Path(str(prefix) + ".stderr.log").open("xb") as err:
                    process = subprocess.Popen(
                        argv,
                        cwd=self.plan["workspace"],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                        pass_fds=(self.lock_fd,),
                    )
                    self.state["worker"] = {
                        "pid": process.pid,
                        "candidate": candidate["id"],
                        "stage": stage,
                        "started_at": time.time(),
                    }
                    self.save()
                    last_heartbeat = time.monotonic()
                    while process.poll() is None:
                        elapsed = time.monotonic() - started
                        if elapsed >= timeout:
                            reason = "timeout"
                            self._stop(process)
                            break
                        try:
                            process.wait(timeout=min(0.5, timeout - elapsed))
                        except subprocess.TimeoutExpired:
                            if time.monotonic() - last_heartbeat >= 1:
                                self.save()
                                last_heartbeat = time.monotonic()
        except OSError as exc:
            reason = f"launch_error:{type(exc).__name__}"
        except BaseException:
            if process is not None:
                self._stop(process)
            self.event("command_interrupted", candidate=candidate["id"], stage=stage)
            raise
        finally:
            self.state.pop("worker", None)
            self.save()
        result = {
            "returncode": process.returncode if process else None,
            "reason": reason,
            "seconds": time.monotonic() - started,
        }
        self.event("command_finished", candidate=candidate["id"], stage=stage, **result)
        return result

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def finish(self, status: str, **values: Any) -> dict:
        self.state.update(status=status, **values)
        self.save()
        self.event("run_finished", status=status, **values)
        return self.state

    def candidate(self, candidate: dict, record: dict) -> None:
        if record.get("status") == "training":
            # A crash may have left outputs. A new directory prevents stale artifacts
            # or metrics from being mistaken for a completed retry.
            self.event("training_retry", candidate=candidate["id"])
        if record.get("status") != "trained":
            if self.state["attempts_started"] >= self.plan["budget"]["max_attempts"]:
                return
            number = self.state["attempts_started"] + 1
            self.state["attempts_started"] = number
            directory = self.directory / "candidates" / candidate["id"] / f"attempt-{number:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            attempt = {"number": number, "directory": str(directory)}
            record.update(status="training", attempt=attempt)
            self.save()
            result = self.command(candidate, attempt, "train")
            record["train_result"] = result
            self.integrity()
            if result["returncode"] != 0 or result["reason"] != "exited":
                record.update(status="failed", reason="training_command_failed")
                self.save()
                return
            record["artifact_hashes"] = self.artifact_hashes(candidate, attempt)
            record["status"] = "trained"
            self.save()
        attempt = record["attempt"]
        self.integrity(record["artifact_hashes"])
        metrics_path = self.output(candidate["validation_metrics"], candidate, attempt)
        # Validation itself is retryable, but old results must not be accepted.
        if record.get("validation_started"):
            record["status"] = "retry_required"
            self.save()
            return
        if metrics_path.exists():
            raise LoopError("validation metrics existed before validation command")
        record["validation_started"] = True
        self.save()
        result = self.command(candidate, attempt, "validate")
        record["validation_result"] = result
        self.integrity()
        self.integrity(record["artifact_hashes"])
        if result["returncode"] != 0 or result["reason"] != "exited":
            record.update(status="failed", reason="validation_command_failed")
        else:
            metrics = _read_json(metrics_path)
            passed, checks = _gates(metrics, self.plan["validation_gate"])
            record.update(
                status="validated",
                selection_score=_metric(metrics, self.plan["selection_metric"]),
                validation_gate_passed=passed,
                validation_checks=checks,
                validation_metrics_sha256=_hash_path(metrics_path),
            )
            self.event(
                "candidate_validated",
                candidate=candidate["id"],
                selection_score=record["selection_score"],
                gate_passed=passed,
            )
        self.save()

    def execute(self) -> dict:
        if self.state["status"] in TERMINAL:
            return self.state
        if self.state.get("test_consumed"):
            return self.finish("test_interrupted", reason="final_test_already_started; never rerun")
        try:
            self.integrity()
            self.state["status"] = "searching"
            self.save()
            self.event("runner_started", pid=os.getpid())
            for candidate in self.plan["candidates"]:
                record = self.state["candidates"].setdefault(candidate["id"], {"status": "pending"})
                if record["status"] in {"validated", "failed"}:
                    continue
                if self.remaining() <= 0:
                    return self.finish("budget_exhausted", reason="wall_time")
                if (
                    self.state["attempts_started"] >= self.plan["budget"]["max_attempts"]
                    and record["status"] != "trained"
                ):
                    break
                try:
                    self.candidate(candidate, record)
                    if record["status"] == "retry_required":
                        record.pop("validation_started", None)
                        self.candidate(candidate, record)
                except IntegrityError:
                    raise
                except (OSError, ValueError) as exc:
                    record.update(
                        status="failed",
                        reason=f"missing_or_invalid_output:{type(exc).__name__}",
                        error=str(exc),
                    )
                    self.event(
                        "candidate_failed", candidate=candidate["id"], reason=record["reason"]
                    )
                    self.save()
            eligible = [
                candidate
                for candidate in self.plan["candidates"]
                if self.state["candidates"].get(candidate["id"], {}).get("validation_gate_passed")
            ]
            if self.remaining() <= 0:
                return self.finish("budget_exhausted", reason="wall_time")
            if not eligible:
                return self.finish("validation_gate_not_met", test_consumed=False)
            selected = max(
                eligible, key=lambda c: self.state["candidates"][c["id"]]["selection_score"]
            )
            return self.evaluate(selected)
        except KeyboardInterrupt:
            self.state["status"] = (
                "test_interrupted" if self.state.get("test_consumed") else "interrupted"
            )
            self.save()
            self.event("runner_interrupted", status=self.state["status"])
            raise
        except LoopError as exc:
            return self.finish("integrity_error", reason=str(exc))

    def evaluate(self, candidate: dict) -> dict:
        if self.remaining() <= 0:
            return self.finish("budget_exhausted", reason="wall_time_before_test")
        record = self.state["candidates"][candidate["id"]]
        attempt = record["attempt"]
        self.integrity()
        self.integrity(record["artifact_hashes"])
        validation = self.output(candidate["validation_metrics"], candidate, attempt)
        if _hash_path(validation) != record["validation_metrics_sha256"]:
            raise IntegrityError("selected candidate's validation metrics changed")
        development = self.plan["evaluation_mode"] == "development"
        if not development:
            metrics_path = self.output(candidate["test_metrics"], candidate, attempt)
            if metrics_path.exists():
                raise LoopError("test metrics existed before the frozen final evaluation")
        freeze = {
            "plan_sha256": self.state["plan_sha256"],
            "candidate": candidate["id"],
            "frozen_at": time.time(),
            "artifact_hashes": record["artifact_hashes"],
            "input_hashes": self.state["input_hashes"],
            "validation_metrics_sha256": record["validation_metrics_sha256"],
            "validation_checks": record["validation_checks"],
            "test_gate": self.plan["test_gate"],
            "evaluation_mode": self.plan["evaluation_mode"],
        }
        _atomic_json(self.directory / "frozen-selection.json", freeze)
        self.state.update(
            status="awaiting_final_evaluation" if development else "evaluating",
            selected_candidate=candidate["id"],
            frozen_selection_sha256=_digest(freeze),
            test_consumed=not development,
        )
        self.save()  # Durable test consumption precedes launching any held-out evaluation.
        self.event("selection_frozen", candidate=candidate["id"], frozen_sha256=_digest(freeze))
        if development:
            return self.finish(
                "awaiting_final_evaluation",
                test_consumed=False,
                reason="development selection frozen; no final test executed",
            )
        result = self.command(candidate, attempt, "evaluate")
        self.integrity()
        self.integrity(record["artifact_hashes"])
        if (
            _hash_path(self.directory / "frozen-selection.json")
            != self.state["frozen_selection_sha256"]
        ):
            raise IntegrityError("frozen selection record changed during final evaluation")
        if result["returncode"] != 0 or result["reason"] != "exited":
            return self.finish(
                "needs_fresh_test",
                test_gate_passed=False,
                reason="final_evaluation_failed",
                test_result=result,
            )
        try:
            metrics = _read_json(metrics_path)
            passed, checks = _gates(metrics, self.plan["test_gate"])
        except (OSError, ValueError) as exc:
            return self.finish(
                "needs_fresh_test",
                test_gate_passed=False,
                reason=f"invalid_final_metrics:{type(exc).__name__}",
                test_result=result,
            )
        return self.finish(
            "completed" if passed else "needs_fresh_test",
            test_gate_passed=passed,
            test_checks=checks,
            test_result=result,
            test_metrics_path=str(metrics_path),
            test_metrics_sha256=_hash_path(metrics_path),
        )


def initialize(plan_path: Path, state_dir: Path) -> None:
    """Freeze a canonical plan and inputs; identical initialization is idempotent."""
    plan_path, state_dir = plan_path.resolve(), state_dir.resolve()
    plan = validate_plan(_read_json(plan_path), plan_path.parent)
    with _lock(state_dir):
        if (state_dir / "state.json").exists():
            if _digest(plan) != (state_dir / "plan.sha256").read_text().strip():
                raise LoopError(
                    "existing run has a different immutable plan; use a new state directory"
                )
            return
        contents = [p for p in state_dir.iterdir() if p.name != ".lock"]
        if contents:
            raise LoopError("new state directory must be empty")
        hashes = {}
        for name in plan["inputs"]:
            path = Path(name)
            if state_dir.is_relative_to(path.resolve()):
                raise LoopError("frozen inputs cannot contain the mutable state directory")
            hashes[name] = _hash_path(path)
        _atomic_json(state_dir / "plan.json", plan)
        (state_dir / "plan.sha256").write_text(_digest(plan) + "\n")
        (state_dir / "plan.json").chmod(0o444)
        (state_dir / "plan.sha256").chmod(0o444)
        _atomic_json(
            state_dir / "state.json",
            {
                "version": 1,
                "status": "ready",
                "created_at": time.time(),
                "heartbeat_at": time.time(),
                "plan_sha256": _digest(plan),
                "input_hashes": hashes,
                "attempts_started": 0,
                "candidates": {},
                "test_consumed": False,
            },
        )


def run_loop(state_dir: Path) -> dict:
    state_dir = state_dir.resolve()
    with _lock(state_dir) as lock_fd:
        return _Runner(state_dir, lock_fd).execute()


def status(state_dir: Path) -> dict:
    result = _read_json(state_dir / "state.json")
    result["seconds_since_heartbeat"] = max(0.0, time.time() - result["heartbeat_at"])
    result["state_directory"] = str(state_dir.resolve())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume", "status"):
        command = sub.add_parser(name)
        command.add_argument("--state-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--plan", type=Path, required=True)
        if name != "status":
            command.add_argument(
                "--background",
                action="store_true",
                help="detach a resumable worker and return its PID",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            initialize(args.plan, args.state_dir)
        if args.command == "status":
            result = status(args.state_dir)
        elif args.background:
            # Validate snapshots before launching; the worker acquires its own lock.
            with _lock(args.state_dir.resolve()) as lock_fd:
                _Runner(args.state_dir.resolve(), lock_fd)
            with (args.state_dir / "runner.log").open("ab") as log:
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "vision_rl.experiment_loop",
                        "resume",
                        "--state-dir",
                        str(args.state_dir.resolve()),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            result = {
                "status": "worker_launched",
                "pid": worker.pid,
                "state_directory": str(args.state_dir.resolve()),
            }
        else:
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            result = run_loop(args.state_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "status":
            return 0
        return (
            0
            if result["status"]
            in {"completed", "worker_launched", "ready", "searching", "awaiting_final_evaluation"}
            else 2
        )
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as exc:
        print(f"experiment loop: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
