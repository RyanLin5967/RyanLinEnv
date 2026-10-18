"""The freshpool dry run's own verdicts must fire on a wrong stop, not only pass a right one.

scripts/freshpool_dryrun/run.py judges two launcher runs from their state and journal. These
fixtures are written out by hand (stage IDs from the study sequence at the dry run's scaled
counts, 3 and 6 updates); no launcher or model runs here.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_run():
    spec = importlib.util.spec_from_file_location(
        "freshpool_dryrun_under_test", ROOT / "scripts" / "freshpool_dryrun" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TAGS = ("1e-05", "5e-05", "1e-04")
BEFORE_G1 = ["e0", "W-h_select", "W-h_gate", "p0"]
BEFORE_G2 = BEFORE_G1 + [s for t in TAGS for s in (f"rl-{t}-s42-u3", f"rl-{t}-s42-u3-h_select")]
BEFORE_G3A = BEFORE_G2 + [
    "rl-5e-05-s42-u6", "rl-5e-05-s42-u6-h_select", "rl-1e-04-s42-u6", "rl-1e-04-s42-u6-h_select",
    *[s for t in TAGS for s in (f"sft-e-{t}-s42", f"sft-e-{t}-s42-h_select")],
    "rl-1e-04-s42-u6-h_gate", "sft-e-5e-05-s42-h_gate"]
BEFORE_G3B = BEFORE_G3A + ["sft-t-5e-05-s42", "sft-t-5e-05-s42-h_select",
                           "sft-t-5e-05-s42-h_gate"]
EVERY_STAGE = BEFORE_G3B + ["egsft-5e-05-s42", "egsft-5e-05-s42-h_gate"]


def test_structure_accepts_exactly_the_stages_before_each_stop():
    run = load_run()
    for stages, point in ((BEFORE_G1, "G1"), (BEFORE_G2, "G2"), (BEFORE_G3A, "G3a"),
                          (BEFORE_G3B, "G3b"), (EVERY_STAGE, "final")):
        assert run.structure_problems(stages, point)[0] == []


def test_structure_fires_on_a_missing_extra_or_unknown_stage():
    run = load_run()
    assert run.structure_problems(BEFORE_G2[:-1], "G2")[0] == [
        "rl150_h_select: 2 completed, expected 3"]
    assert run.structure_problems(BEFORE_G2, "G1")[0] == [
        "rl150: 3 completed, expected 0", "rl150_h_select: 3 completed, expected 0"]
    assert run.structure_problems([*BEFORE_G1, "rl-1e-05-s42-u150"], "G1")[0] == [
        "unclassified stage rl-1e-05-s42-u150"]
    no_sft_t = [s for s in BEFORE_G3B if not s.startswith("sft-t")] + ["sft-t-5e-05-s42-h_gate"]
    assert run.structure_problems(no_sft_t, "G3b")[0] == [
        "sft_t: 0 completed, expected k", "sft_t_h_select: 0 completed, expected k"]


def strict_run(tmp_path, stages, stop_reason, *, gate=None, failed_at=None, status="evaluated",
               passed_gates=()):
    journal = []
    if gate:
        report = tmp_path / "report.json"
        report.write_text(json.dumps({"failed_at": failed_at, "gates": {
            gate: {"status": status, "pass": False}}}))
        journal.append({"event": "gate_finished", "gate": gate, "report": str(report),
                        "verdict": "gate_failed"})
    state = {"status": "stopped", "stop_reason": stop_reason,
             "completed": {s: {"fragment": {}} for s in stages},
             "gates": {g: {} for g in passed_gates}}
    return {"state": state, "journal": journal, "state_dir": tmp_path}


FIXTURE = {"scan_ids": [], "w_sha": "0" * 64}


def test_strict_passes_a_gate_stop_the_report_itself_names(tmp_path):
    run = load_run()
    problems, _, stopped_at = run.judge_strict(
        strict_run(tmp_path, BEFORE_G2, "gate_failed", gate="G2", failed_at="G2",
                   passed_gates=("G1",)), FIXTURE)
    assert (problems, stopped_at) == ([], "G2")


def test_strict_fires_on_a_non_gate_stop_or_a_disagreeing_report(tmp_path):
    run = load_run()
    problems, _, _ = run.judge_strict(strict_run(tmp_path, BEFORE_G1, "stage_failed"), FIXTURE)
    assert problems == ["stopped for stage_failed, not a gate verdict"]
    problems, _, _ = run.judge_strict(
        strict_run(tmp_path, BEFORE_G2, "gate_failed", gate="G2", failed_at="G1",
                   passed_gates=("G1",)), FIXTURE)
    assert problems == ["gate_failed at G2, but the report says failed_at G1, status "
                        "evaluated, pass False"]
    problems, _, _ = run.judge_strict(
        strict_run(tmp_path, BEFORE_G1, "gate_failed", gate="G2", failed_at="G2",
                   passed_gates=("G1",)), FIXTURE)
    assert problems == ["rl150: 0 completed, expected 3",
                        "rl150_h_select: 0 completed, expected 3"]


def drive_run(tmp_path, forced, *, stop_reason="final_development_check_failed",
              failed_at="G1"):
    report = tmp_path / "final.json"
    report.write_text(json.dumps({"failed_at": failed_at}))
    state = {"status": "stopped", "stop_reason": stop_reason,
             "completed": {s: {"fragment": {}} for s in EVERY_STAGE}, "gates": {}}
    journal = [{"event": "final_check", "report": str(report), "passed": False}]
    return {"state": state, "journal": journal, "forced": forced, "state_dir": tmp_path}


FIRST = {"gate": "G1", "launcher_verdict": "gate_failed", "evaluator_failed_at": "G1",
         "evaluator_status": "evaluated", "direct_pass": False}
LATER = {"gate": "G2", "launcher_verdict": "gate_not_evaluated", "evaluator_failed_at": "G1",
         "evaluator_status": "not_evaluated", "direct_pass": True}


def test_drive_passes_consistent_forced_gates(tmp_path):
    run = load_run()
    assert run.judge_drive(drive_run(tmp_path, [FIRST, LATER]), FIXTURE)[0] == []


def test_drive_fires_on_an_inconsistent_force_or_wrong_final_stop(tmp_path):
    run = load_run()
    passing_first = {**FIRST, "direct_pass": True}
    problems = run.judge_drive(drive_run(tmp_path, [passing_first]), FIXTURE)[0]
    assert problems == [f"forced gate G1 inconsistent: {passing_first}"]
    problems = run.judge_drive(drive_run(tmp_path, [FIRST], stop_reason="stage_failed"),
                               FIXTURE)[0]
    assert problems == ["stopped as stopped/stage_failed, expected "
                        "final_development_check_failed"]
    problems = run.judge_drive(drive_run(tmp_path, [FIRST], failed_at="G2"), FIXTURE)[0]
    assert problems == ["final report failed_at G2, first forced gate G1"]
    problems = run.judge_drive(drive_run(tmp_path, []), FIXTURE)[0]
    assert problems == ["no gate forced, yet status stopped/final_development_check_failed"]


def test_strict_reports_a_gate_stop_with_no_gate_finished_or_no_report(tmp_path):
    # Before this fix a gate_failed state with no gate_finished event raised TypeError.
    run = load_run()
    problems, counts, stopped_at = run.judge_strict(
        strict_run(tmp_path, BEFORE_G2, "gate_failed"), FIXTURE)
    assert (problems, counts, stopped_at) == (
        ["stopped gate_failed, but the journal's last gate_finished names gate None, not one "
         "of ('G1', 'G2', 'G3a', 'G3b', 'final')"], {}, None)
    missing = strict_run(tmp_path, BEFORE_G2, "gate_failed", gate="G2", failed_at="G2")
    (tmp_path / "report.json").unlink()
    problems, _, stopped_at = run.judge_strict(missing, FIXTURE)
    assert stopped_at == "G2"
    assert len(problems) == 1
    assert problems[0].startswith("stopped gate_failed at G2, but its evaluator report is "
                                  "unreadable: FileNotFoundError")
