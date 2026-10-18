"""scripts/freshpool_report.py over state directories written by the REAL launcher.

The launcher runs its fake-trainer fixture from tests/test_freshpool_launch.py (no model),
so the report is checked against the journal, fragments, gate evidence and gate reports the
launcher itself wrote. This module spawns phase_worker.py processes: tests/conftest.py
refuses it while the production study is running, so it runs in CI, never beside the study.

Expected values are written out from that fixture's design (its RL150 / RL600 / SFT_E /
SFT_T tables and assay counts), never read back from the report.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCH_TESTS = load("freshpool_launch_tests_for_report",
                    ROOT / "tests" / "test_freshpool_launch.py")
REPORT = load("freshpool_report_e2e_under_test", ROOT / "scripts" / "freshpool_report.py")
pytestmark = LAUNCH_TESTS.pytestmark


def report_of(fixture, tmp_path):
    out = tmp_path / "report.json"
    code = REPORT.main(["--state-directory", str(tmp_path / "state"), "--plan",
                        str(fixture.plan_path), "--json", str(out), "--markdown",
                        str(tmp_path / "report.md")])
    assert code == 0
    return json.loads(out.read_text())


def test_report_recomputes_a_full_launcher_run(tmp_path, monkeypatch):
    fixture = LAUNCH_TESTS.Fixture(tmp_path, monkeypatch)
    assert LAUNCH_TESTS.load_launch().main(fixture.argv) == 0
    report = report_of(fixture, tmp_path)
    assert report["study_status"] == "completed"
    assert [(g["gate"], g["verdict"], g["matches_launcher_report"]) for g in report["gates"]] \
        == [(g, "pass", True) for g in ("G1", "G2", "G3a", "G3b")]
    assert (report["gate_runs"][-1]["gate"], report["gate_runs"][-1]["verdict"]) == (
        "final", True)
    assert report["selections"]["rl_star"] == "rl-1e-04-s42-u600"
    assert report["selections_recomputed"] == {
        "rl_continuers": ["rl-5e-05-s42-u150", "rl-1e-04-s42-u150"],
        "rl_star": "rl-1e-04-s42-u600", "sft_e_star": "sft-e-5e-05-s42",
        "sft_t_star": "sft-t-5e-05-s42"}
    assert [s["stage"] for s in report["stages"]] == LAUNCH_TESTS.EXPECTED_ORDER
    assert all(s["status"] == "completed" for s in report["stages"])
    # Fixture assays: W 0 of 5 / 0 of 9; RL* 5 of 5 and 9 of 9; SFT-E* and SFT-T* 6 of 9 on
    # H-gate; EG-SFT 5 of 9. Every image alike, so each interval collapses to its point.
    h = {(r["candidate"], r["split"]): r for r in report["h_accuracy"]}
    assert h[("W", "h_select")]["candidate_accuracy"] == 0.0
    assert h[("rl-1e-04-s42-u600", "h_gate")]["candidate_accuracy"] == pytest.approx(1.0)
    assert h[("rl-1e-04-s42-u600", "h_gate")]["ci95"] == pytest.approx([1.0, 1.0])
    versus = {(r["comparator_role"], r["split"]): r for r in report["rl_star_comparisons"]}
    assert versus[("SFT-E*", "h_gate")]["difference"] == pytest.approx(1 - 6 / 9)
    assert versus[("SFT-T*", "h_gate")]["difference"] == pytest.approx(1 - 6 / 9)
    assert versus[("EG-SFT", "h_gate")]["difference"] == pytest.approx(1 - 5 / 9)
    assert versus[("W", "h_select")]["lb90_one_sided"] == pytest.approx(1.0)
    # Fixture branch receipt: a* .75, D_req .05.
    assert report["target"]["rl_val_target_lower_bound"] == pytest.approx(0.80)


def test_report_of_a_g1_stop(tmp_path, monkeypatch):
    fixture = LAUNCH_TESTS.Fixture(tmp_path, monkeypatch, p0_pass=False)
    assert LAUNCH_TESTS.load_launch().main(fixture.argv) == 2
    report = report_of(fixture, tmp_path)
    assert report["study_status"] == "stopped" and report["stop_reason"] == "gate_failed"
    g1 = report["gates"][0]
    assert (g1["verdict"], g1["matches_launcher_report"]) == ("gate_failed", True)
    # p = .05 for every candidate: u = .1855 < .5, none accepted.
    assert g1["recomputed"]["values"]["acceptance"] == 0.0
    assert [g["state"] for g in report["gates"][1:]] == ["not started"] * 3
    assert report["terminal_event"]["event"] == "study_stopped"


def test_report_after_an_interrupted_and_resumed_arm(tmp_path, monkeypatch):
    fixture = LAUNCH_TESTS.Fixture(tmp_path, monkeypatch)
    marker = tmp_path / "interrupted-once"
    fixture.scenario["rl-1e-05-s42-u150"]["behaviour"] = {"interrupt_once": str(marker)}
    fixture.write()
    command = [sys.executable, "-B", str(ROOT / "scripts" / "freshpool_launch" / "launch.py"),
               *fixture.argv]
    first = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert first.returncode == 3, first.stdout + first.stderr
    middle = report_of(fixture, tmp_path)
    stage = next(s for s in middle["stages"] if s["stage"] == "rl-1e-05-s42-u150")
    assert stage["status"] == "interrupted"
    second = subprocess.run([*command, "--resume"], capture_output=True, text=True,
                            timeout=900)
    assert second.returncode == 0, second.stdout + second.stderr
    report = report_of(fixture, tmp_path)
    stage = next(s for s in report["stages"] if s["stage"] == "rl-1e-05-s42-u150")
    assert (stage["status"], stage["instances"], stage["void_instances"]) == ("completed", 2, 1)
    assert [g["verdict"] for g in report["gates"]] == ["pass"] * 4
