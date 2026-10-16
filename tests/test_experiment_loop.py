"""Exercise actual subprocesses, interrupted search, and one-shot test handling."""

import json
import sys
import time
from pathlib import Path

import pytest

from vision_rl import experiment_loop as loop

WORKER = """
import json
import os
import pathlib
import sys
import time

stage, score, behavior = sys.argv[1:4]
directory = pathlib.Path(os.environ["LOOP_CANDIDATE_DIR"])
state = pathlib.Path(os.environ["LOOP_STATE_DIR"])
assert os.environ["LOOP_STAGE"] == stage
with (state / "invocations.jsonl").open("a") as handle:
    handle.write(json.dumps({"stage": stage, "id": os.environ["LOOP_CANDIDATE_ID"]}) + "\\n")
print("worker", stage, flush=True)
print("diagnostic", stage, file=sys.stderr, flush=True)
if behavior == "sleep":
    time.sleep(30)
if behavior == "fail":
    sys.exit(7)
if stage == "train":
    if behavior != "missing":
        (directory / "policy.bin").write_text(score)
    if behavior == "stale":
        (directory / "validation.json").write_text('{"accuracy": 1.0}')
else:
    if behavior == "mutate":
        (directory / "policy.bin").write_text("changed during evaluation")
    filename = "validation.json" if stage == "validate" else "test.json"
    if behavior == "invalid":
        (directory / filename).write_text("not-json")
    elif behavior == "missing":
        pass
    else:
        (directory / filename).write_text(json.dumps({"accuracy": float(score),
            "margin_ci95": [float(score) - 0.5, float(score) - 0.3], "n": 100}))
"""


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "worker.py").write_text(WORKER)
    (tmp_path / "dataset.json").write_text('{"heldout_seed": 17}')

    def candidate(identifier, score, test_score=None):
        def command(stage, value):
            return ["{python}", "worker.py", stage, str(value), "normal"]

        return {
            "id": identifier,
            "train": command("train", score),
            "validate": command("validate", score),
            "evaluate": command("evaluate", score if test_score is None else test_score),
            "artifacts": ["{candidate_dir}/policy.bin"],
            "validation_metrics": "{candidate_dir}/validation.json",
            "test_metrics": "{candidate_dir}/test.json",
        }

    plan = {
        "version": 1,
        "workspace": ".",
        "inputs": ["dataset.json", "worker.py"],
        "budget": {"max_attempts": 5, "max_wall_seconds": 120, "command_timeout_seconds": 10},
        "selection_metric": "accuracy",
        "validation_gate": {"accuracy": {"min": 0.9}},
        "test_gate": {"accuracy": {"min": 0.9}, "margin_ci95.0": {"min": 0.3}, "n": {"min": 100}},
        "candidates": [candidate("small", 0.91), candidate("large", 0.98)],
    }
    plan_path = tmp_path / "plan.json"
    directory = tmp_path / "loop"

    def initialize():
        plan_path.write_text(json.dumps(plan))
        loop.initialize(plan_path, directory)
        return directory

    return plan, directory, initialize, candidate


def invocations(directory):
    path = directory / "invocations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_real_worker_search_selects_best_then_evaluates_once(setup):
    plan, directory, initialize, _ = setup
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "completed"
    assert result["selected_candidate"] == "large"
    assert result["test_consumed"] and result["test_gate_passed"]
    assert result["attempts_started"] == 2
    assert invocations(directory) == [
        {"id": "small", "stage": "train"},
        {"id": "small", "stage": "validate"},
        {"id": "large", "stage": "train"},
        {"id": "large", "stage": "validate"},
        {"id": "large", "stage": "evaluate"},
    ]
    frozen = json.loads((directory / "frozen-selection.json").read_text())
    assert frozen["candidate"] == "large"
    assert frozen["test_gate"] == plan["test_gate"]
    assert len(frozen["artifact_hashes"]) == 1
    assert len(frozen["input_hashes"]) == 2
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "run_finished"
    assert loop._check_events(directory / "events.jsonl")[0] == len(events)
    assert "diagnostic evaluate" in (directory / "logs/large-a0002-evaluate.stderr.log").read_text()
    assert loop.run_loop(directory) == result
    assert len(invocations(directory)) == 5


def test_failure_on_final_test_needs_fresh_test_and_never_retrains(setup):
    plan, directory, initialize, candidate = setup
    plan["candidates"] = [candidate("overfit", 0.99, 0.6)]
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "needs_fresh_test"
    assert result["test_consumed"] and not result["test_gate_passed"]
    assert not result["test_checks"]["accuracy"]["passed"]
    assert loop.run_loop(directory) == result
    assert len(invocations(directory)) == 3


def test_no_test_is_exposed_when_validation_gate_fails(setup):
    plan, directory, initialize, candidate = setup
    plan["candidates"] = [candidate("weak", 0.5)]
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "validation_gate_not_met"
    assert not result["test_consumed"]
    assert not (directory / "frozen-selection.json").exists()
    assert [v["stage"] for v in invocations(directory)] == ["train", "validate"]


def test_development_freezes_selection_without_executing_or_consuming_test(setup):
    plan, directory, initialize, _ = setup
    plan["evaluation_mode"] = "development"
    del plan["test_gate"]
    for candidate in plan["candidates"]:
        del candidate["evaluate"]
        del candidate["test_metrics"]
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "awaiting_final_evaluation"
    assert result["selected_candidate"] == "large"
    assert result["test_consumed"] is False
    assert "test_gate_passed" not in result
    assert all(item["stage"] != "evaluate" for item in invocations(directory))
    freeze = json.loads((directory / "frozen-selection.json").read_text())
    assert freeze["evaluation_mode"] == "development"
    assert freeze["artifact_hashes"] and freeze["test_gate"] == {}
    assert loop.run_loop(directory) == result


@pytest.mark.parametrize("stage", ["train", "validate"])
@pytest.mark.parametrize("behavior", ["fail", "missing", "invalid"])
def test_bad_candidate_outputs_do_not_prevent_other_candidates(setup, stage, behavior):
    plan, directory, initialize, _ = setup
    if stage == "train" and behavior == "invalid":
        behavior = "missing"
    plan["candidates"][0][stage][-1] = behavior
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "completed"
    assert result["selected_candidate"] == "large"
    assert result["candidates"]["small"]["status"] == "failed"


@pytest.mark.parametrize("behavior", ["fail", "missing", "invalid"])
def test_failed_final_evaluation_consumes_test(setup, behavior):
    plan, directory, initialize, _ = setup
    plan["candidates"][1]["evaluate"][-1] = behavior
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "needs_fresh_test"
    assert result["test_consumed"]
    assert loop.run_loop(directory) == result
    assert len([v for v in invocations(directory) if v["stage"] == "evaluate"]) == 1


@pytest.mark.parametrize("stage", ["validate", "evaluate"])
def test_frozen_checkpoint_mutation_fails_closed(setup, stage):
    plan, directory, initialize, _ = setup
    plan["candidates"] = plan["candidates"][:1]
    plan["candidates"][0][stage][-1] = "mutate"
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "integrity_error"
    assert "changed" in result["reason"]
    assert result["test_consumed"] == (stage == "evaluate")


def test_changed_input_before_resume_prevents_execution(setup):
    _, directory, initialize, _ = setup
    initialize()
    (directory.parent / "dataset.json").write_text("changed labels")
    result = loop.run_loop(directory)
    assert result["status"] == "integrity_error"
    assert invocations(directory) == []


def test_original_plan_changes_cannot_extend_existing_search(setup):
    plan, directory, initialize, _ = setup
    initialize()
    plan["budget"]["max_attempts"] = 500
    with pytest.raises(loop.LoopError, match="different immutable plan"):
        initialize()
    snapshot = json.loads((directory / "plan.json").read_text())
    assert snapshot["budget"]["max_attempts"] == 5


def test_modified_plan_snapshot_is_rejected(setup):
    _, directory, initialize, _ = setup
    initialize()
    path = directory / "plan.json"
    path.chmod(0o644)
    path.write_text(path.read_text().replace('"max_attempts":5', '"max_attempts":99'))
    with pytest.raises(loop.LoopError, match="plan checksum"):
        loop.run_loop(directory)


def test_event_log_tampering_is_detected(setup):
    _, directory, initialize, _ = setup
    initialize()
    loop.run_loop(directory)
    path = directory / "events.jsonl"
    path.write_text(path.read_text().replace('"event":"runner_started"', '"event":"forged"'))
    with pytest.raises(loop.LoopError, match="event log"):
        loop.run_loop(directory)


def test_lock_blocks_concurrent_runner(setup):
    _, directory, initialize, _ = setup
    initialize()
    with loop._lock(directory):
        with pytest.raises(loop.LoopError, match="holds this experiment lock"):
            loop.run_loop(directory)
    assert invocations(directory) == []


@pytest.mark.parametrize("stage", ["train", "validate"])
def test_interrupted_search_retries_in_new_directory(setup, monkeypatch, stage):
    plan, directory, initialize, _ = setup
    plan["candidates"] = plan["candidates"][:1]
    initialize()
    original = loop._Runner.command

    def interrupted(self, candidate, attempt, current_stage):
        if current_stage == stage:
            # An interrupted worker may have partially written any output.
            (Path(attempt["directory"]) / "partial.bin").write_text("preserve me")
            raise KeyboardInterrupt()
        return original(self, candidate, attempt, current_stage)

    monkeypatch.setattr(loop._Runner, "command", interrupted)
    with pytest.raises(KeyboardInterrupt):
        loop.run_loop(directory)
    assert loop.status(directory)["status"] == "interrupted"
    monkeypatch.setattr(loop._Runner, "command", original)
    result = loop.run_loop(directory)
    assert result["status"] == "completed"
    assert result["attempts_started"] == 2
    assert (directory / "candidates/small/attempt-0001/partial.bin").exists()
    assert result["candidates"]["small"]["attempt"]["directory"].endswith("attempt-0002")


def test_interrupted_test_is_not_repeated_even_if_no_metrics_written(setup, monkeypatch):
    plan, directory, initialize, _ = setup
    plan["candidates"] = plan["candidates"][:1]
    initialize()
    original = loop._Runner.command

    def interrupted(self, candidate, attempt, stage):
        if stage == "evaluate":
            raise KeyboardInterrupt()
        return original(self, candidate, attempt, stage)

    monkeypatch.setattr(loop._Runner, "command", interrupted)
    with pytest.raises(KeyboardInterrupt):
        loop.run_loop(directory)
    monkeypatch.setattr(loop._Runner, "command", original)
    result = loop.run_loop(directory)
    assert result["status"] == "test_interrupted" and result["test_consumed"]
    assert [v["stage"] for v in invocations(directory)] == ["train", "validate"]


def test_candidate_attempt_budget_applies_before_starting_next_worker(setup):
    plan, directory, initialize, _ = setup
    plan["budget"]["max_attempts"] = 1
    initialize()
    result = loop.run_loop(directory)
    assert result["attempts_started"] == 1
    assert result["selected_candidate"] == "small"
    assert len(invocations(directory)) == 3


def test_timeout_terminates_worker_and_never_opens_test(setup):
    plan, directory, initialize, _ = setup
    plan["candidates"] = plan["candidates"][:1]
    plan["budget"]["command_timeout_seconds"] = 0.2
    plan["candidates"][0]["train"][-1] = "sleep"
    initialize()
    started = time.monotonic()
    result = loop.run_loop(directory)
    assert time.monotonic() - started < 3
    assert result["candidates"]["small"]["train_result"]["reason"] == "timeout"
    assert not result["test_consumed"]
    with loop._lock(directory):
        pass


def test_wall_budget_counts_time_between_restarts(setup):
    plan, directory, initialize, _ = setup
    plan["budget"]["max_wall_seconds"] = 0.01
    initialize()
    time.sleep(0.02)
    assert loop.run_loop(directory)["status"] == "budget_exhausted"
    assert invocations(directory) == []


def test_stale_validation_output_cannot_be_accepted(setup):
    plan, directory, initialize, _ = setup
    plan["candidates"] = plan["candidates"][:1]
    plan["candidates"][0]["train"][-1] = "stale"
    initialize()
    result = loop.run_loop(directory)
    assert result["status"] == "validation_gate_not_met"
    assert result["candidates"]["small"]["status"] == "failed"
    assert not result["test_consumed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.9"])
def test_metric_gate_rejects_non_numeric_or_nonfinite(value):
    with pytest.raises(loop.LoopError, match="finite"):
        loop._gates({"accuracy": value}, {"accuracy": {"min": 0.8}})


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p.update(candidates=[]), "candidates"),
        (lambda p: p["candidates"][0].update(train="python train.py"), "argv"),
        (lambda p: p["candidates"][0].update(id="../escape"), "safe filename"),
        (lambda p: p["candidates"][0].update(artifacts=["/tmp/unchecked.pt"]), "candidate_dir"),
        (lambda p: p.update(test_gate={}), "predeclared"),
        (lambda p: p.update(inputs=[]), "pin at least"),
    ],
)
def test_plan_contract_rejects_unsafe_or_ungated_runs(setup, change, match):
    plan, _, initialize, _ = setup
    change(plan)
    with pytest.raises(loop.LoopError, match=match):
        initialize()


def test_cli_background_worker_completes_and_status_is_read_only(setup, capsys):
    _, directory, initialize, _ = setup
    initialize()
    assert loop.main(["resume", "--state-dir", str(directory), "--background"]) == 0
    launch = json.loads(capsys.readouterr().out)
    assert launch["pid"] > 0
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = loop.status(directory)
        if result["status"] in loop.TERMINAL:
            break
        time.sleep(0.05)
    assert result["status"] == "completed"
    before = len(invocations(directory))
    assert loop.main(["status", "--state-dir", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["test_gate_passed"]
    assert len(invocations(directory)) == before
    assert sys.executable in (directory / "events.jsonl").read_text()
