"""Fixture tests for scripts/freshpool_launch: a fake trainer, a fake torch, no model.

Every expected value below is written out from the fixture design (stage order, selected
LRs, stop reasons), never read back from the launcher. The real gate evaluator decides the
gates; a fake evaluator is used only to prove that a missing or non-evaluated gate stops.
"""

import hashlib
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "freshpool_launch"
EVALUATOR = ROOT / "scripts" / "evaluate_freshpool_gates.py"
GiB = 1024**3

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(os, "killpg"), reason="POSIX process groups"
)


def load_launch():
    spec = importlib.util.spec_from_file_location("freshpool_launch_under_test",
                                                  PACKAGE / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


W_SHA = sha_text("W adapter")
LABELS = ["A", "B"]
OTHER = {"A": "B", "B": "A"}
H_SELECT = [("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")]
H_GATE = [("c1", "A"), ("c2", "A"), ("d1", "B"), ("d2", "B")]
COMPLETIONS = {"h_select": 5, "h_gate": 9}
SPLIT_ROWS = {"h_select": H_SELECT, "h_gate": H_GATE}

FAKE_TRAINER = '''
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser()
    for name in ("--stage-id", "--fragment", "--arm-dir"):
        parser.add_argument(name)
    known, _ = parser.parse_known_args(argv)
    with open(os.environ["FAKE_CALLS"], "a") as handle:
        handle.write(json.dumps({"stage": known.stage_id, "argv": argv}) + "\\n")
    scenario = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
    spec = scenario.get(known.stage_id)
    if spec is None:
        return 7
    behaviour = spec.get("behaviour", {})
    arm = Path(known.arm_dir)
    marker = behaviour.get("interrupt_once")
    if marker and not Path(marker).exists():
        (arm / "checkpoint-75").mkdir(parents=True, exist_ok=True)
        Path(marker).write_text("interrupted")
        os.kill(os.getppid(), signal.SIGTERM)
        time.sleep(30)
    if "spawn" in behaviour:
        child = subprocess.Popen(["sleep", str(behaviour["spawn"])])
        Path(behaviour["pid_file"]).write_text(str(child.pid))
    time.sleep(behaviour.get("sleep", 0))
    if "exit" in behaviour:
        return behaviour["exit"]
    payload = dict(spec["payload"])
    if payload.pop("make_checkpoint", False):
        checkpoint = arm / f"checkpoint-{payload['updates']}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        payload["checkpoint"] = str(checkpoint)
    if payload.pop("make_manifests", False):
        for key in ("accepted_manifest", "scanned_manifest"):
            path = arm / f"{known.stage_id}-{key}.jsonl"
            path.write_text("{}\\n")
            payload[key] = str(path)
    payload["argv"] = argv
    Path(known.fragment).write_text(json.dumps(payload))
    return 0
'''

FAKE_TORCH = '''
import os


class _MPS:
    def is_available(self):
        return True

    def recommended_max_memory(self):
        return 48 * 1024**3

    def set_per_process_memory_fraction(self, fraction):
        pass

    def driver_allocated_memory(self):
        return int(os.environ.get("FAKE_DRIVER_BYTES", str(1024**3)))


mps = _MPS()


class _Backends:
    pass


backends = _Backends()
backends.mps = mps


def set_num_threads(count):
    pass
'''

FAKE_EVALUATOR = '''
import json, runpy, sys
_real = runpy.run_path({real!r})
globals().update({{k: v for k, v in _real.items() if not k.startswith("__")}})
if __name__ == "__main__":
    output = sys.argv[sys.argv.index("--output") + 1]
    mode = {mode!r}
    if mode == "awaiting":
        with open(output, "x") as handle:
            json.dump({{"gates": {{"G1": {{"status": "awaiting_evidence"}}}},
                       "failed_at": None, "awaiting": "G1"}}, handle)
    sys.exit(0)
'''


def predictions(split, correct):
    n = COMPLETIONS[split]
    return {i: [label] * correct + [OTHER[label]] * (n - correct)
            for i, label in SPLIT_ROWS[split]}


def assay(run_id, split, adapter, correct):
    return {"payload": {"run_id": run_id, "split": split, "adapter_sha256": adapter,
                        "predictions": predictions(split, correct), "eos_fraction": 1.0}}


def g1_document(p, mixed_groups):
    order = [f"x{i:03d}" for i in range(500)]
    order.sort()
    random.Random(42).shuffle(order)
    u = 1 - p**4 - (1 - p) ** 4
    journal = [{"id": i, "p": p, "u": u, "accepted": u >= 0.5, "scorer_seconds": 1.0,
                "adapter_sha256": W_SHA} for i in order[:400]]
    rollouts, index = [], 0
    for image in order[:64]:
        for seed in (42, 314):
            correct = [True, True, False, False] if index < mixed_groups else [False] * 4
            rollouts.append({"id": image, "seed": seed, "completions": [
                {"correct": c, "valid": True, "eos": True, "canonical_tokenization": True}
                for c in correct]})
            index += 1
    return {"seed": 42, "adapter_sha256": W_SHA, "scan_order": order, "journal": journal,
            "rollouts": rollouts}


def train(run_id, arm, lr, updates, seconds, rl=False):
    payload = {"status": "completed", "run_id": run_id, "arm": arm, "lr": lr, "seed": 42,
               "updates": updates, "committed_active_seconds": seconds,
               "adapter_sha256": sha_text(run_id), "reference_sha256": W_SHA,
               "useful_mixed": [True] * updates if rl else [], "make_checkpoint": True}
    if rl:
        payload["make_manifests"] = True
    return {"payload": payload}


# Hand-designed outcomes. H-select correct completions per image (of 5); H-gate (of 9).
RL150 = {"1e-05": (1e-5, 1), "5e-05": (5e-5, 3), "1e-04": (1e-4, 2)}  # top two: 5e-5, 1e-4
RL600 = {"5e-05": (5e-5, 3), "1e-04": (1e-4, 5)}  # RL* = 1e-4
SFT_E = {"1e-05": (1e-5, 2, 100.0), "5e-05": (5e-5, 4, 100.0), "1e-04": (1e-4, 4, 101.0)}
SFT_T = {"5e-05": (5e-5, 4), "1e-04": (1e-4, 3)}  # SFT-T* = 5e-5
T_RL = 1000.0

EXPECTED_ORDER = [
    "e0", "W-h_select", "W-h_gate", "p0",
    "rl-1e-05-s42-u150", "rl-1e-05-s42-u150-h_select",
    "rl-5e-05-s42-u150", "rl-5e-05-s42-u150-h_select",
    "rl-1e-04-s42-u150", "rl-1e-04-s42-u150-h_select",
    "rl-5e-05-s42-u600", "rl-5e-05-s42-u600-h_select",
    "rl-1e-04-s42-u600", "rl-1e-04-s42-u600-h_select",
    "sft-e-1e-05-s42", "sft-e-1e-05-s42-h_select",
    "sft-e-5e-05-s42", "sft-e-5e-05-s42-h_select",
    "sft-e-1e-04-s42", "sft-e-1e-04-s42-h_select",
    "rl-1e-04-s42-u600-h_gate", "sft-e-5e-05-s42-h_gate",
    "sft-t-5e-05-s42", "sft-t-5e-05-s42-h_select",
    "sft-t-1e-04-s42", "sft-t-1e-04-s42-h_select",
    "sft-t-5e-05-s42-h_gate",
    "egsft-5e-05-s42", "egsft-5e-05-s42-h_gate",
]


def scenario(p0_pass=True):
    s = {
        "e0": {"payload": {"checks": {
            "resume_2_plus_2_identical": True, "resume_1_plus_3_identical": True,
            "reference_unchanged": True, "finite_nonzero_task_gradient": True,
            "scorer_uses_sampler_transform": True},
            "sampler_teacher_max_abs_logprob_difference": 0.0005}},
        "W-h_select": assay("W", "h_select", W_SHA, 0),
        "W-h_gate": assay("W", "h_gate", W_SHA, 0),
        # p = 0.5: u = 0.875, every candidate accepted, 112 of 128 groups mixed.
        # p = 0.05: u = 0.1855, none accepted, so acceptance 0 < 0.30 fails G1.
        "p0": {"payload": g1_document(0.5, 112) if p0_pass else g1_document(0.05, 24)},
    }
    for tag, (lr, correct) in RL150.items():
        run_id = f"rl-{tag}-s42-u150"
        s[run_id] = train(run_id, "RL", lr, 150, 250.0, rl=True)
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    for tag, (lr, correct) in RL600.items():
        run_id = f"rl-{tag}-s42-u600"
        s[run_id] = train(run_id, "RL", lr, 600, T_RL, rl=True)
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    s["rl-1e-04-s42-u600-h_gate"] = assay("rl-1e-04-s42-u600", "h_gate",
                                          sha_text("rl-1e-04-s42-u600"), 9)
    for tag, (lr, correct, seconds) in SFT_E.items():
        run_id = f"sft-e-{tag}-s42"
        s[run_id] = train(run_id, "SFT-E", lr, 600, seconds)
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    s["sft-e-5e-05-s42-h_gate"] = assay("sft-e-5e-05-s42", "h_gate",
                                        sha_text("sft-e-5e-05-s42"), 6)
    for tag, (lr, correct) in SFT_T.items():
        run_id = f"sft-t-{tag}-s42"
        s[run_id] = train(run_id, "SFT-T", lr, 2400, T_RL)
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    s["sft-t-5e-05-s42-h_gate"] = assay("sft-t-5e-05-s42", "h_gate",
                                        sha_text("sft-t-5e-05-s42"), 6)
    s["egsft-5e-05-s42"] = train("egsft-5e-05-s42", "EG-SFT", 5e-5, 600, 100.0)
    s["egsft-5e-05-s42-h_gate"] = assay("egsft-5e-05-s42", "h_gate",
                                        sha_text("egsft-5e-05-s42"), 5)
    return s


COMMON = ["--stage-id", "{stage_id}", "--fragment", "{fragment}", "--arm-dir", "{arm_dir}"]
TRAIN_TAIL = ["--lr", "{lr}", "--cap", "{active_cap}"]
RESUME = ["--resume-from", "{latest_checkpoint}"]


class Fixture:
    def __init__(self, tmp, monkeypatch, *, evaluator=EVALUATOR, p0_pass=True, **overrides):
        self.tmp, self.overrides = tmp, overrides
        pkg = tmp / "pkg"
        pkg.mkdir()
        (pkg / "fake_freshpool.py").write_text(FAKE_TRAINER)
        (pkg / "torch.py").write_text(FAKE_TORCH)
        self.entry = pkg / "fake_freshpool.py"
        self.calls = tmp / "calls.jsonl"
        self.scenario_path = tmp / "scenario.json"
        self.scenario = scenario(p0_pass)
        self.environment = {"PYTORCH_ENABLE_MPS_FALLBACK": "0", "ACCELERATE_TORCH_DEVICE": "mps",
                            "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(pkg),
                            "FAKE_CALLS": str(self.calls),
                            "FAKE_SCENARIO": str(self.scenario_path)}
        for name, value in self.environment.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("FAKE_DRIVER_BYTES", raising=False)
        self.evaluator = evaluator
        self.sentinel = f"freshpool-owner-sentinel-{os.getpid()}-{time.time_ns()}"
        self.g0_files()
        self.write()

    def g0_files(self):
        manifests = self.tmp / "manifests"
        manifests.mkdir()
        files = {}
        for name in ("h_select.jsonl", "h_gate.jsonl", "s_scan_seed42.jsonl"):
            (manifests / name).write_text(f"{name}\n")
            files[name] = sha_file(manifests / name)
        manifest = {"g0_audit_complete": self.overrides.get("audit_complete", True),
                    "forced_fire_passed": True, "files_sha256": files}
        (manifests / "manifest.json").write_text(json.dumps(manifest))
        receipts = {}
        branch = {"a_star": 0.75, "b4_edge_extension": False, "w_valid_eos": 0.99,
                  "w_val_accuracy": self.overrides.get("w_val_accuracy", 0.5), "d_req": 0.05}
        for name, content in (("protocol_revision", "{}"), ("decision_record", "retired"),
                              ("branch", json.dumps(branch))):
            path = self.tmp / f"{name}.json"
            path.write_text(content)
            receipts[name] = {"path": str(path), "sha256": sha_file(path)}
        base = self.tmp / "evidence-base.json"
        base.write_text(json.dumps({
            "labels": LABELS, "a_star": 0.75, "b4_edge_extension": False,
            "rows": {split: [{"id": i, "label": label} for i, label in rows]
                     for split, rows in SPLIT_ROWS.items()}}))
        self.g0 = {"manifest_dir": str(manifests),
                   "manifest_sha256": sha_file(manifests / "manifest.json"),
                   "receipts": receipts}
        self.base = {"path": str(base), "sha256": sha_file(base)}

    def write(self):
        self.scenario_path.write_text(json.dumps(self.scenario))
        entry_sha = sha_file(self.entry)

        def template(kind, extra, resume=False):
            value = {"kind": kind, "entry": "fake_freshpool:main", "entry_sha256": entry_sha,
                     "argv": COMMON + extra}
            if resume:
                value["resume_argv"] = COMMON + extra + RESUME
            return value

        plan = {
            "schema": "freshpool-launch-plan-v1",
            "python": sys.executable,
            "state_directory": str(self.tmp / "state"),
            "evaluator": str(self.evaluator),
            "file_sha256": {str(self.evaluator): sha_file(self.evaluator),
                            str(PACKAGE / "phase_worker.py"): sha_file(
                                PACKAGE / "phase_worker.py"),
                            str(self.entry): entry_sha},
            "environment": self.environment,
            "limits": {"minimum_free_bytes": self.overrides.get("reserve", 5 * GiB),
                       "rss_limit_bytes": self.overrides.get("rss", 64 * GiB),
                       "survivor_grace_seconds": self.overrides.get("grace", 3),
                       "poll_seconds": 0.05},
            "forecast_seconds": {"e0": self.overrides.get("e0_forecast", 600), "p0": 600,
                                 "assay_h_select": 600, "assay_h_gate": 600, "rl": 2000,
                                 "sft_e": 600, "egsft": 600},
            "templates": {
                "e0": template("train", [], resume=True),
                "assay": template("assay", ["--adapter", "{adapter}", "--split", "{split}",
                                            "--manifest", "{split_manifest}"]),
                "p0": template("assay", ["--scan", "{manifest_s_scan_seed42}",
                                         "--adapter", "{w_adapter}"]),
                "rl150": template("train", TRAIN_TAIL + ["--stop-after-steps", "150"],
                                  resume=True),
                "rl600": template("train", TRAIN_TAIL + ["--resume-from",
                                                         "{resume_checkpoint}"], resume=True),
                "sft_e": template("train", TRAIN_TAIL + ["--train", "{accepted_manifest}"],
                                  resume=True),
                "sft_t": template("train", TRAIN_TAIL + [
                    "--train", "{scanned_manifest}", "--continue-from", "{parent_checkpoint}",
                    "--max-active-seconds", "{t_rl}"], resume=True),
                "egsft": template("train", TRAIN_TAIL + ["--train", "{accepted_manifest}"],
                                  resume=True),
            },
            "b4_edge_extension": False,
            "w_endpoint": {"adapter_sha256": W_SHA, "adapter_path": str(self.tmp / "w")},
            "owner_patterns": [self.sentinel],
            "g0": self.g0,
            "evidence_base": self.base,
        }
        self.plan_path = self.tmp / "plan.json"
        self.plan_path.write_text(json.dumps(plan, indent=1))
        self.plan_sha = sha_file(self.plan_path)

    @property
    def argv(self):
        return ["--plan", str(self.plan_path), "--expected-plan-sha256", self.plan_sha]

    def state(self):
        return json.loads((self.tmp / "state" / "state.json").read_text())

    def journal(self):
        path = self.tmp / "state" / "journal.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def stages_called(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]


def first_order(calls):
    seen = []
    for call in calls:
        if call["stage"] not in seen:
            seen.append(call["stage"])
    return seen


def argv_value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_full_sequence_resumes_after_interrupt_and_reaches_g4(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    marker = tmp_path / "interrupted-once"
    fixture.scenario["rl-1e-05-s42-u150"]["behaviour"] = {"interrupt_once": str(marker)}
    fixture.write()
    command = [sys.executable, "-B", str(PACKAGE / "launch.py"), *fixture.argv]
    first = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert first.returncode == 3, first.stdout + first.stderr
    interrupted = fixture.state()
    assert interrupted["status"] == "running"
    wall_before = interrupted["arm_wall_seconds"]["rl-1e-05-s42"]
    assert wall_before > 0
    # A fresh start on the same state is refused: the state directory is single-use.
    again = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert again.returncode == 2 and "state_directory_exists_single_use" in again.stdout

    second = subprocess.run([*command, "--resume"], capture_output=True, text=True,
                            timeout=900)
    assert second.returncode == 0, second.stdout + second.stderr
    state = fixture.state()
    assert state["status"] == "completed"
    calls = fixture.stages_called()
    assert first_order(calls) == EXPECTED_ORDER
    rl_calls = [c["argv"] for c in calls if c["stage"] == "rl-1e-05-s42-u150"]
    assert len(rl_calls) == 2 and "--resume-from" not in rl_calls[0]
    assert argv_value(rl_calls[1], "--resume-from").endswith("checkpoint-75")
    # The arm cap is cumulative over resumes, never reset.
    assert state["arm_wall_seconds"]["rl-1e-05-s42"] >= wall_before
    assert state["selections"] == {
        "rl_continuers": ["rl-5e-05-s42-u150", "rl-1e-04-s42-u150"],
        "rl_star": "rl-1e-04-s42-u600",
        "sft_e_star": "sft-e-5e-05-s42",
        "sft_t_lrs": [5e-05, 1e-04],
        "sft_t_star": "sft-t-5e-05-s42",
    }
    by_stage = {c["stage"]: c["argv"] for c in calls}
    # Continuation to 600 starts from the 150 checkpoint of the same arm.
    assert argv_value(by_stage["rl-1e-04-s42-u600"], "--resume-from").endswith(
        "arms/rl-1e-04-s42/checkpoint-150")
    # SFT-E and EG-SFT train on RL*'s accepted sequence; SFT-T on its full scan, to T_RL.
    accepted = "rl-1e-04-s42-u600-accepted_manifest.jsonl"
    assert argv_value(by_stage["sft-e-1e-05-s42"], "--train").endswith(accepted)
    assert argv_value(by_stage["egsft-5e-05-s42"], "--train").endswith(accepted)
    assert argv_value(by_stage["egsft-5e-05-s42"], "--lr") == "5e-05"
    sft_t = by_stage["sft-t-1e-04-s42"]
    assert argv_value(sft_t, "--train").endswith("rl-1e-04-s42-u600-scanned_manifest.jsonl")
    assert argv_value(sft_t, "--continue-from").endswith("arms/sft-e-1e-04-s42/checkpoint-600")
    assert argv_value(sft_t, "--max-active-seconds") == "1000.0"
    # 1.25x caps: RL 1.25 * 2000, SFT-T 1.25 * T_RL.
    assert argv_value(by_stage["rl-5e-05-s42-u150"], "--cap") == "2500.0"
    assert argv_value(sft_t, "--cap") == "1250.0"
    events = [e["event"] for e in fixture.journal()]
    assert events.count("gate_finished") == 4
    assert {e["gate"]: e["verdict"] for e in fixture.journal()
            if e["event"] == "gate_finished"} == {"G1": "pass", "G2": "pass", "G3a": "pass",
                                                  "G3b": "pass"}
    assert "launch_interrupted" in events and "stage_resumed" in events
    assert events[-1] == "study_development_complete"
    # Resume is refused once the state is terminal.
    third = subprocess.run([*command, "--resume"], capture_output=True, text=True, timeout=120)
    assert third.returncode == 2 and "state_is_terminal" in third.stdout


def test_g1_failure_stops_before_any_rl_and_short_survivor_is_allowed(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, p0_pass=False, grace=5)
    pid_file = tmp_path / "short.pid"
    fixture.scenario["e0"]["behaviour"] = {"spawn": 0.5, "pid_file": str(pid_file)}
    fixture.write()
    assert launch.main(fixture.argv) == 2
    state = fixture.state()
    assert (state["status"], state["stop_reason"]) == ("stopped", "gate_failed")
    assert first_order(fixture.stages_called()) == ["e0", "W-h_select", "W-h_gate", "p0"]
    gate = [e for e in fixture.journal() if e["event"] == "gate_finished"]
    assert [(e["gate"], e["verdict"]) for e in gate] == [("G1", "gate_failed")]
    assert "owned_survivors" not in [e["event"] for e in fixture.journal()]
    # Single use: a stopped study never resumes.
    assert launch.main([*fixture.argv, "--resume"]) == 2


def test_survivor_past_grace_stops_and_is_killed(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, grace=1)
    pid_file = tmp_path / "long.pid"
    fixture.scenario["e0"]["behaviour"] = {"spawn": 60, "pid_file": str(pid_file)}
    fixture.write()
    assert launch.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "stage_exited_with_owned_descendants"
    survivors = [e for e in fixture.journal() if e["event"] == "owned_survivors"]
    pid = int(pid_file.read_text())
    assert survivors and pid in [p["pid"] for p in survivors[0]["processes"]]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        # Gone, or a zombie awaiting its new parent's reap: either way no longer running.
        if not [row for row in launch.processes() if row[0] == pid
                and not row[3].startswith("Z")]:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"survivor {pid} still alive")


@pytest.mark.parametrize("mode,reason", [("silent", "gate_result_missing"),
                                         ("awaiting", "gate_not_evaluated")])
def test_missing_gate_result_stops_never_passes(tmp_path, monkeypatch, mode, reason):
    launch = load_launch()
    fake = tmp_path / f"fake_evaluator_{mode}.py"
    fake.write_text(FAKE_EVALUATOR.format(real=str(EVALUATOR), mode=mode))
    fixture = Fixture(tmp_path, monkeypatch, evaluator=fake)
    assert launch.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == reason
    assert first_order(fixture.stages_called()) == ["e0", "W-h_select", "W-h_gate", "p0"]


def test_disk_reserve_stops_before_any_stage(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, "free_bytes", lambda path: 5 * GiB - 1)
    assert launch.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "disk_reserve"
    assert fixture.stages_called() == []


def test_plan_cannot_lower_the_5_gib_reserve(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, reserve=4 * GiB)
    assert launch.main(fixture.argv) == 2
    assert not (tmp_path / "state").exists()


def test_arm_wall_cap_stops_a_running_stage(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, e0_forecast=0.4)  # cap 0.5 s
    fixture.scenario["e0"]["behaviour"] = {"sleep": 20}
    fixture.write()
    started = time.monotonic()
    assert launch.main(fixture.argv) == 2
    assert time.monotonic() - started < 15
    assert fixture.state()["stop_reason"] == "arm_cap"


def test_fragment_active_seconds_over_cap_stops(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.scenario["rl-1e-05-s42-u150"]["payload"]["committed_active_seconds"] = 2500.5
    fixture.write()
    assert launch.main(fixture.argv) == 2
    state = fixture.state()
    assert state["stop_reason"] == "arm_cap"
    assert first_order(fixture.stages_called())[-1] == "rl-1e-05-s42-u150"


def test_rss_guard_fires(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, rss=1024)  # any python process exceeds 1 KiB
    fixture.scenario["e0"]["behaviour"] = {"sleep": 5}
    fixture.write()
    assert launch.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "process_group_rss_limit"


def test_second_owner_refuses(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                              fixture.sentinel])
    try:
        time.sleep(0.3)
        assert launch.main(fixture.argv) == 2
    finally:
        owner.kill()
        owner.wait()
    assert fixture.state()["stop_reason"] == "second_accelerator_owner"
    assert fixture.stages_called() == []


def test_mps_driver_guard_fails_the_stage(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.scenario["e0"]["behaviour"] = {"sleep": 5}
    fixture.write()
    monkeypatch.setenv("FAKE_DRIVER_BYTES", str(25 * GiB))
    assert launch.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "stage_failed"
    summaries = list((tmp_path / "state" / "guards").glob("e0-*.json"))
    assert summaries
    assert "exceeded 24 GiB" in json.loads(summaries[0].read_text())["stop_reason"]


def test_changed_pinned_entry_is_refused(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.entry.write_text(fixture.entry.read_text() + "\n# edited\n")
    assert launch.main(fixture.argv) == 2
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("override", [{"audit_complete": False}, {"w_val_accuracy": 0.9}])
def test_g0_failure_does_not_launch(tmp_path, monkeypatch, override):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, **override)
    assert launch.main(fixture.argv) == 2
    assert not (tmp_path / "state").exists()
    assert fixture.stages_called() == []


def test_journal_rewrite_is_detected(tmp_path):
    launch = load_launch()
    ledger = launch.Ledger(tmp_path)
    ledger.state = {"journal_bytes": 0, "journal_sha256": ""}
    ledger.append("one")
    ledger.append("two")
    ledger.save()
    ledger.check_journal_prefix()
    ledger.append("three")  # appending keeps the prefix
    ledger.check_journal_prefix()
    lines = ledger.journal_path.read_text().splitlines()
    ledger.journal_path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    with pytest.raises(launch.Stop) as caught:
        ledger.check_journal_prefix()
    assert caught.value.reason == "journal_not_append_only"


def test_interrupt_signal_is_restored(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, "free_bytes", lambda path: 0)
    before = signal.getsignal(signal.SIGTERM)
    launch.main(fixture.argv)
    assert signal.getsignal(signal.SIGTERM) is before
