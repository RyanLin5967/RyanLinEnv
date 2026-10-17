"""Fixture tests for scripts/freshpool_launch/launch.py: a fake trainer, a fake torch, no model.

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
import threading
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
# The S_scan_seed42 manifest: random.Random(42).shuffle(sorted(ids)), written out here.
SCAN = sorted(f"x{i:04d}" for i in range(1300))
random.Random(42).shuffle(SCAN)

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
        time.sleep(behaviour.get("sleep_before_interrupt", 0))
        Path(marker).write_text("interrupted")
        os.kill(os.getppid(), signal.SIGTERM)
        time.sleep(30)
    groups = arm / "trainer" / "reward_groups.jsonl"
    if "live_mixed" in behaviour:
        groups.parent.mkdir(parents=True, exist_ok=True)
        with groups.open("a") as handle:
            for n, mixed in enumerate(behaviour["live_mixed"]):
                handle.write(json.dumps({"id": f"g{n}", "mixed": mixed}) + "\\n")
    if "second_groups_after" in behaviour:
        time.sleep(behaviour["second_groups_after"])
        extra = arm / "second" / "reward_groups.jsonl"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("")
    if "spawn" in behaviour:
        child = subprocess.Popen(["sleep", str(behaviour["spawn"])])
        Path(behaviour["pid_file"]).write_text(str(child.pid))
    time.sleep(behaviour.get("sleep", 0))
    if "exit" in behaviour:
        return behaviour["exit"]
    payload = dict(spec["payload"])
    if payload.get("arm") == "RL" and "live_mixed" not in behaviour:
        groups.parent.mkdir(parents=True, exist_ok=True)
        groups.write_text("".join(json.dumps({"id": f"g{n}", "mixed": m}) + "\\n"
                                  for n, m in enumerate(payload["useful_mixed"])))
    if payload.pop("make_checkpoint", False):
        checkpoint = arm / f"checkpoint-{payload['updates']}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        payload["checkpoint"] = str(checkpoint)
        adapter = arm / "final_adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        payload["adapter_path"] = str(adapter)
    manifests = payload.pop("make_manifests", None)
    if manifests:
        for key in ("accepted", "scanned"):
            path = Path(known.fragment).parent / f"{key}_manifest.jsonl"
            path.write_text("".join(json.dumps({"id": i}) + "\\n" for i in manifests[key]))
            payload[f"{key}_manifest"] = str(path)
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
    u = 1 - p**4 - (1 - p) ** 4
    journal = [{"id": i, "p": p, "u": u, "accepted": u >= 0.5, "scorer_seconds": 1.0,
                "adapter_sha256": W_SHA} for i in SCAN[:400]]
    rollouts, index = [], 0
    for image in SCAN[:64]:
        for seed in (42, 314):
            correct = [True, True, False, False] if index < mixed_groups else [False] * 4
            rollouts.append({"id": image, "seed": seed, "completions": [
                {"correct": c, "valid": True, "eos": True, "canonical_tokenization": True}
                for c in correct]})
            index += 1
    return {"seed": 42, "adapter_sha256": W_SHA, "scan_order": SCAN, "journal": journal,
            "rollouts": rollouts}


def series(*pieces):
    """{step: committed seconds}, linear within each (first, last, s_first, s_last) piece;
    the endpoints are exact."""
    steps = {}
    for first, last, start, stop in pieces:
        for step in range(first, last + 1):
            fraction = (step - first) / max(1, last - first)
            steps[str(step)] = start + (stop - start) * fraction
        steps[str(first)], steps[str(last)] = start, stop
    return steps


def train(run_id, arm, lr, updates, seconds, steps, manifests=None):
    payload = {"status": "completed", "run_id": run_id, "arm": arm, "lr": lr, "seed": 42,
               "updates": updates, "committed_active_seconds": seconds,
               "step_committed_active_seconds": steps,
               "adapter_sha256": sha_text(run_id), "reference_sha256": W_SHA,
               "useful_mixed": [True] * updates if manifests else [], "make_checkpoint": True}
    if manifests:
        payload["make_manifests"] = manifests
    return {"payload": payload}


# Hand-designed outcomes. H-select correct completions per image (of 5); H-gate (of 9).
RL150 = {"1e-05": (1e-5, 1), "5e-05": (5e-5, 3), "1e-04": (1e-4, 2)}  # top two: 5e-5, 1e-4
RL600 = {"5e-05": (5e-5, 3), "1e-04": (1e-4, 5)}  # RL* = 1e-4
SFT_E = {"1e-05": (1e-5, 2, 100.0), "5e-05": (5e-5, 4, 100.0), "1e-04": (1e-4, 4, 101.0)}
SFT_T = {"5e-05": (5e-5, 4), "1e-04": (1e-4, 3)}  # SFT-T* = 5e-5
RL150_SECONDS, T_RL = 250.0, 1000.0
SFT_T_UPDATES = 2400

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
        s[run_id] = train(run_id, "RL", lr, 150, RL150_SECONDS,
                          series((1, 150, 1.0, RL150_SECONDS)),
                          {"accepted": SCAN[:150], "scanned": SCAN[:300]})
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    for tag, (lr, correct) in RL600.items():
        run_id = f"rl-{tag}-s42-u600"
        # Cumulative: passes through the 150 parent's own committed seconds at step 150.
        s[run_id] = train(run_id, "RL", lr, 600, T_RL,
                          series((1, 150, 1.0, RL150_SECONDS), (151, 600, 252.0, T_RL)),
                          {"accepted": SCAN[:600], "scanned": SCAN[:1200]})
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    s["rl-1e-04-s42-u600-h_gate"] = assay("rl-1e-04-s42-u600", "h_gate",
                                          sha_text("rl-1e-04-s42-u600"), 9)
    for tag, (lr, correct, seconds) in SFT_E.items():
        run_id = f"sft-e-{tag}-s42"
        s[run_id] = train(run_id, "SFT-E", lr, 600, seconds, series((1, 600, 0.2, seconds)))
        # Ended at its declared 600-step boundary with max_steps above it: continuable.
        s[run_id]["payload"].update(receipt_status="boundary_reached", requested_boundary={
            "stop_after_steps": 600, "stop_after_active_seconds": None})
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    s["sft-e-5e-05-s42-h_gate"] = assay("sft-e-5e-05-s42", "h_gate",
                                        sha_text("sft-e-5e-05-s42"), 6)
    for tag, (lr, correct) in SFT_T.items():
        run_id = f"sft-t-{tag}-s42"
        parent = SFT_E[tag][2]
        # Continues SFT-E's clock at step 601 and stops at the first step >= T_RL.
        s[run_id] = train(run_id, "SFT-T", lr, SFT_T_UPDATES, T_RL,
                          series((601, SFT_T_UPDATES - 1, parent + 0.5, T_RL - 0.3),
                                 (SFT_T_UPDATES, SFT_T_UPDATES, T_RL, T_RL)))
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", sha_text(run_id), correct)
    s["sft-t-5e-05-s42-h_gate"] = assay("sft-t-5e-05-s42", "h_gate",
                                        sha_text("sft-t-5e-05-s42"), 6)
    s["egsft-5e-05-s42"] = train("egsft-5e-05-s42", "EG-SFT", 5e-5, 600, 100.0,
                                 series((1, 600, 0.2, 100.0)))
    s["egsft-5e-05-s42-h_gate"] = assay("egsft-5e-05-s42", "h_gate",
                                        sha_text("egsft-5e-05-s42"), 5)
    return s


# The PR #4 decision-log entry's shape (docs/decision-log-2026-10-16-claude.md, D11).
RETIRED_RECORD = (
    "# Decisions\n## D11 — design adopted; 16-shot plan retired\n"
    "- **Retired: the unactivated 16-shot prospective TRAIN plan** (seed 2718, 1,600 images,\n"
    "  `docs/fgvc-aircraft-audit-2026-10-16.json`). It was never activated.\n")

COMMON = ["--stage-id", "{stage_id}", "--fragment", "{fragment}", "--arm-dir", "{arm_dir}"]
TRAIN_TAIL = ["--lr", "{lr}", "--cap", "{active_cap}"]
RL_WRAPPER = ["--output", "{arm_dir}/trainer"]
SEQUENTIAL_CYCLE = ["--order", "sequential", "--cycle"]
SFT_E_TAIL = TRAIN_TAIL + SEQUENTIAL_CYCLE + [
    "--train", "{accepted_manifest}", "--stop-after-steps", "600", "--max-steps", "1200"]


def git(directory, *args):
    subprocess.run(["git", "-C", str(directory), "-c", "user.name=fixture",
                    "-c", "user.email=fixture@example.invalid", *args],
                   check=True, capture_output=True, timeout=60)


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


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
        self.extra_e0_argv = []
        self.g0_files()
        self.write()

    def g0_files(self):
        o = self.overrides
        manifests = self.tmp / "manifests"
        manifests.mkdir()
        h_select = list(o.get("h_select", H_SELECT))
        for name, rows in (("h_select", h_select), ("h_gate", H_GATE)):
            jsonl(manifests / f"{name}.jsonl",
                  [{"id": i, "split": "train", "image": f"{i}.jpg", "label": label}
                   for i, label in rows])
        jsonl(manifests / "s_scan_seed42.jsonl",
              [{"id": i, "split": "train", "image": f"{i}.jpg", "label": LABELS[n % 2]}
               for n, i in enumerate(o.get("scan", SCAN))])
        # Pool: 2,919 before the audit, 3 pHash drops, so 2,916 = H (8) + S scan (1,300)
        # + 1,608 others (S drops and the unscanned remainder, written out here).
        drops = o.get("pool_drops", ["drop0", "drop1", "drop2"])
        pool = [i for i, _ in H_SELECT + H_GATE] + SCAN + [f"y{i:04d}" for i in range(1608)]
        pool = o.get("pool", pool)
        jsonl(manifests / "pool.jsonl",
              [{"id": i, "split": "train", "image": f"{i}.jpg", "label": LABELS[n % 2]}
               for n, i in enumerate(pool)])
        pairs = [{"ids": [f"p{n}", f"q{n}"], "hamming": 20} for n in range(50)]
        pairs[0]["hamming"] = o.get("pair_hamming", 20)
        (manifests / "phash_audit.json").write_text(json.dumps({
            "hamming_max": 6, "pool_drops": drops, "forced_fire": {
                "passed": True, "planted_flagged": True, "planted_hamming": 2,
                "distinct_class_pairs": pairs, "distinct_class_pairs_flagged": 0}}))
        files = {name: sha_file(manifests / name) for name in (
            "pool.jsonl", "h_select.jsonl", "h_gate.jsonl", "s_scan_seed42.jsonl",
            "phash_audit.json")}
        manifest = {"g0_audit_complete": o.get("audit_complete", True),
                    "forced_fire_passed": True, "files_sha256": files,
                    "counts": {"h_select": len(h_select), "h_gate": len(H_GATE),
                               "pool_drops": len(drops), "pool_after_audit": len(pool)}}
        (manifests / "manifest.json").write_text(json.dumps(manifest))

        receipts_dir = self.tmp / "receipts"
        receipts_dir.mkdir()
        git(receipts_dir, "init", "-q")
        frozen = {"substantial_gain_gates": {"accuracy": 0.8}, "strong_baseline": {"m": "27B"}}
        candidate = receipts_dir / "candidate.json"
        candidate.write_text(json.dumps({"protocol_id": "candidate", **frozen}))
        protocol = {"protocol_id": "fgvc-aircraft-freshpool-screened-grpo-v1",
                    "activated": o.get("activated", True),
                    "revises": {"path": str(candidate), "sha256": sha_file(candidate)},
                    **frozen, **o.get("protocol_patch", {})}
        branch = {"a_star": 0.75, "b4_edge_extension": False, "w_valid_eos": 0.99,
                  "w_evaluator_refused": o.get("w_evaluator_refused", False),
                  "w_val_accuracy": o.get("w_val_accuracy", 0.5), "d_req": 0.05,
                  "mode": "full"}
        decision = o.get("decision", RETIRED_RECORD)
        contents = {"protocol_revision": json.dumps(protocol), "decision_record": decision,
                    "branch": json.dumps(branch)}
        paths = {name: receipts_dir / f"{name}.json" for name in contents}
        for name, content in contents.items():
            paths[name].write_text(content)
        git(receipts_dir, "add", "candidate.json", *[p.name for p in paths.values()])
        git(receipts_dir, "commit", "-q", "-m", "receipts")
        if o.get("uncommitted_protocol"):
            paths["protocol_revision"].write_text(contents["protocol_revision"] + "\n")
        receipts = {name: {"path": str(path), "sha256": sha_file(path)}
                    for name, path in paths.items()}
        base = self.tmp / "evidence-base.json"
        evidence_rows = {split: [{"id": i, "label": label} for i, label in rows]
                         for split, rows in SPLIT_ROWS.items()}
        if "evidence_h_gate" in o:
            evidence_rows["h_gate"] = o["evidence_h_gate"]
        base.write_text(json.dumps({"labels": LABELS, "a_star": 0.75,
                                    "b4_edge_extension": False, "rows": evidence_rows}))
        self.g0 = {"manifest_dir": str(manifests),
                   "manifest_sha256": sha_file(manifests / "manifest.json"),
                   "receipts": receipts}
        self.base = {"path": str(base), "sha256": sha_file(base)}

    def write(self):
        self.scenario_path.write_text(json.dumps(self.scenario))
        entry_sha = sha_file(self.entry)

        def template(kind, extra):
            return {"kind": kind, "entry": "fake_freshpool:main", "entry_sha256": entry_sha,
                    "argv": COMMON + extra}

        templates = {
            "e0": template("train", self.extra_e0_argv),
            "assay": template("assay", ["--adapter", "{adapter}", "--split", "{split}",
                                        "--manifest", "{split_manifest}"]),
            "p0": template("assay", ["--scan", "{manifest_s_scan_seed42}",
                                     "--adapter", "{w_adapter}"]),
            "rl150": template("train", RL_WRAPPER + ["--"] + TRAIN_TAIL
                              + ["--stop-after-steps", "150"]),
            "rl600": template("train", RL_WRAPPER + ["--continue-output-from",
                                                     "{resume_checkpoint}", "--"] + TRAIN_TAIL),
            "sft_e": template("train", ["--output", "{arm_dir}/trainer", "--"] + SFT_E_TAIL),
            "sft_t": template("train", TRAIN_TAIL + [
                "--train", "{scanned_manifest}", "--continue-from", "{parent_checkpoint}",
                "--stop-after-active-seconds", "{t_rl}"]),
            "egsft": template("train", TRAIN_TAIL + ["--train", "{accepted_manifest}"]),
        }
        for name, patch in self.overrides.get("template_patch", {}).items():
            templates[name] = {**templates[name], **patch}
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
            "templates": templates,
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


def test_interrupted_arm_is_rerun_from_scratch_never_resumed(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    marker = tmp_path / "interrupted-once"
    fixture.scenario["rl-1e-05-s42-u150"]["behaviour"] = {"interrupt_once": str(marker)}
    fixture.write()
    command = [sys.executable, "-B", str(PACKAGE / "launch.py"), *fixture.argv]
    first = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert first.returncode == 3, first.stdout + first.stderr
    assert fixture.state()["status"] == "running"
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
    # The re-run starts from scratch in a new instance; the interrupted one is void and
    # its checkpoint-75 is never handed to the trainer.
    assert len(rl_calls) == 2
    assert argv_value(rl_calls[0], "--arm-dir").endswith("arms/rl-1e-05-s42-u150/instance-1")
    assert argv_value(rl_calls[1], "--arm-dir").endswith("arms/rl-1e-05-s42-u150/instance-2")
    assert not any("checkpoint-75" in part for part in rl_calls[1])
    assert [(v["stage"], v["instance"]) for v in state["void_instances"]] == [
        ("rl-1e-05-s42-u150", 1)]
    void = [e for e in fixture.journal() if e["event"] == "arm_instance_void"]
    assert [(e["stage"], e["instance"]) for e in void] == [("rl-1e-05-s42-u150", 1)]
    completed = state["completed"]["rl-1e-05-s42-u150"]["path"]
    assert completed.endswith("stages/rl-1e-05-s42-u150/instance-2/fragment.json")
    assert state["selections"] == {
        "rl_continuers": ["rl-5e-05-s42-u150", "rl-1e-04-s42-u150"],
        "rl_star": "rl-1e-04-s42-u600",
        "sft_e_star": "sft-e-5e-05-s42",
        "sft_t_lrs": [5e-05, 1e-04],
        "sft_t_star": "sft-t-5e-05-s42",
    }
    by_stage = {c["stage"]: c["argv"] for c in calls}
    # Continuation to 600 starts from the validated 150 endpoint's checkpoint.
    assert argv_value(by_stage["rl-1e-04-s42-u600"], "--continue-output-from").endswith(
        "arms/rl-1e-04-s42-u150/instance-1/checkpoint-150")
    # SFT-E and EG-SFT train on RL*'s accepted sequence; SFT-T on its full scan, to T_RL.
    accepted = "stages/rl-1e-04-s42-u600/instance-1/accepted_manifest.jsonl"
    assert argv_value(by_stage["sft-e-1e-05-s42"], "--train").endswith(accepted)
    assert argv_value(by_stage["egsft-5e-05-s42"], "--train").endswith(accepted)
    assert argv_value(by_stage["egsft-5e-05-s42"], "--lr") == "5e-05"
    sft_t = by_stage["sft-t-1e-04-s42"]
    assert argv_value(sft_t, "--train").endswith(
        "stages/rl-1e-04-s42-u600/instance-1/scanned_manifest.jsonl")
    assert argv_value(sft_t, "--continue-from").endswith(
        "arms/sft-e-1e-04-s42/instance-1/checkpoint-600")
    assert argv_value(sft_t, "--stop-after-active-seconds") == "1000.0"
    # The assays read the endpoint's final adapter, not its optimizer checkpoint.
    assert argv_value(by_stage["sft-t-5e-05-s42-h_gate"], "--adapter").endswith(
        "arms/sft-t-5e-05-s42/instance-1/final_adapter")
    # 1.25x caps: RL 1.25 * 2000, SFT-T 1.25 * T_RL.
    assert argv_value(by_stage["rl-5e-05-s42-u150"], "--cap") == "2500.0"
    assert argv_value(sft_t, "--cap") == "1250.0"
    # Continuations count the parent's committed seconds against the arm cap: no reset.
    started = {e["stage"]: e for e in fixture.journal() if e["event"] == "stage_started"}
    assert started["rl-1e-04-s42-u600"]["inherited_seconds"] == 250.0
    assert started["sft-t-1e-04-s42"]["inherited_seconds"] == 101.0
    assert started["e0"]["owner_poll_seconds"] == 2.0
    # Every RL stage was watched live, and saw every reward-group row it wrote.
    assert [s for s, e in started.items() if e["live_useful_mixed"]] == [
        "rl-1e-05-s42-u150", "rl-5e-05-s42-u150", "rl-1e-04-s42-u150",
        "rl-5e-05-s42-u600", "rl-1e-04-s42-u600"]
    monitored = [(e["stage"], e["rows"]) for e in fixture.journal()
                 if e["event"] == "useful_mixed_monitored"]
    assert monitored == [("rl-1e-05-s42-u150", 150), ("rl-5e-05-s42-u150", 150),
                         ("rl-1e-04-s42-u150", 150), ("rl-5e-05-s42-u600", 600),
                         ("rl-1e-04-s42-u600", 600)]
    events = [e["event"] for e in fixture.journal()]
    assert {e["gate"]: e["verdict"] for e in fixture.journal()
            if e["event"] == "gate_finished"} == {"G1": "pass", "G2": "pass", "G3a": "pass",
                                                  "G3b": "pass"}
    assert "launch_interrupted" in events
    assert events[-1] == "study_development_complete"
    # Resume is refused once the state is terminal.
    third = subprocess.run([*command, "--resume"], capture_output=True, text=True, timeout=120)
    assert third.returncode == 2 and "state_is_terminal" in third.stdout


def test_plan_declaring_a_resume_command_is_refused(tmp_path, monkeypatch, capsys):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, template_patch={
        "rl150": {"resume_argv": COMMON + ["--resume-from", "{latest_checkpoint}"]}})
    assert launch.main(fixture.argv) == 2
    assert "template_declares_resume_argv" in capsys.readouterr().out
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("patch, reason", [
    # rl150/rl600: the wrapper's --output must sit under {arm_dir} so the live tail sees it.
    ({"rl150": COMMON + ["--"] + TRAIN_TAIL}, "template_output_not_under_arm_dir"),
    ({"rl150": COMMON + ["--output", "{stage_dir}/trainer", "--"] + TRAIN_TAIL},
     "template_output_not_under_arm_dir"),
    ({"rl600": COMMON + ["--output", "{arm_dir}/../elsewhere", "--"] + TRAIN_TAIL},
     "template_output_not_under_arm_dir"),
    ({"rl600": COMMON + ["--output", "{arm_dir}x", "--"] + TRAIN_TAIL},
     "template_output_not_under_arm_dir"),
    # An --output after the -- reaches the trainer, not the wrapper (which refuses it).
    ({"rl150": COMMON + ["--", "--output", "{arm_dir}/trainer"] + TRAIN_TAIL},
     "template_output_not_under_arm_dir"),
    ({"rl150": COMMON + RL_WRAPPER + TRAIN_TAIL}, "template_missing_wrapper_separator"),
    # sft_e: --stop-after-steps 600 and --max-steps above 600, both after the --.
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL + ["--max-steps", "1200"]},
     "sft_e_template_stop_after_steps_not_600"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
      + ["--stop-after-steps", "150", "--max-steps", "1200"]},
     "sft_e_template_stop_after_steps_not_600"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
      + ["--stop-after-steps", "600"]}, "sft_e_template_max_steps_not_above_600"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
      + ["--stop-after-steps", "600", "--max-steps", "600"]},
     "sft_e_template_max_steps_not_above_600"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
      + ["--stop-after-steps", "600", "--max-steps", "{t_rl}"]},
     "sft_e_template_max_steps_not_above_600"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--stop-after-steps", "600",
                         "--max-steps", "1200", "--"] + TRAIN_TAIL},
     "sft_e_template_stop_after_steps_not_600"),
    # sft_e: max_steps above the 600-row accepted manifest needs sequential replay that
    # cycles, or the trainer refuses the arm; again only after the --.
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
      + ["--stop-after-steps", "600", "--max-steps", "1200"]},
     "sft_e_template_not_sequential_cycle"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
      + ["--stop-after-steps", "600", "--max-steps", "1200", "--order", "sequential"]},
     "sft_e_template_not_sequential_cycle"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
      + ["--stop-after-steps", "600", "--max-steps", "1200", "--cycle"]},
     "sft_e_template_not_sequential_cycle"),
    ({"sft_e": COMMON + ["--output", "{arm_dir}/t", "--order", "sequential", "--cycle", "--"]
      + TRAIN_TAIL + ["--stop-after-steps", "600", "--max-steps", "1200"]},
     "sft_e_template_not_sequential_cycle"),
])
def test_bad_stage_template_is_refused_at_plan_load(tmp_path, monkeypatch, capsys, patch,
                                                    reason):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, template_patch={
        name: {"argv": argv} for name, argv in patch.items()})
    assert launch.main(fixture.argv) == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (printed["status"], printed["reason"]) == ("refused", reason)
    assert not (tmp_path / "state").exists()
    assert fixture.stages_called() == []


def test_stage_template_equals_form_is_accepted(tmp_path, monkeypatch, capsys):
    """Control for the refusals above: the flag=value spelling passes plan load."""
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, template_patch={
        "rl150": {"argv": COMMON + ["--output={arm_dir}/trainer", "--"] + TRAIN_TAIL},
        "sft_e": {"argv": COMMON + ["--output", "{arm_dir}/t", "--"] + TRAIN_TAIL
                  + ["--stop-after-steps=600", "--max-steps=601", "--order=sequential",
                     "--cycle"]}})
    assert launch.main([*fixture.argv, "--verify-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def run_until_stop(launch, fixture):
    assert launch.main(fixture.argv) == 2
    state = fixture.state()
    assert state["status"] == "stopped"
    assert fixture.journal()[-1]["event"] == "study_stopped"
    return state


def test_partial_w_assay_stops_instead_of_lowering_w(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    del fixture.scenario["W-h_select"]["payload"]["predictions"]["b2"]
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "assay_image_ids_differ_from_manifest"
    assert first_order(fixture.stages_called())[-1] == "W-h_select"


def test_partial_sft_e_h_gate_assay_stops_before_g3a(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    answers = fixture.scenario["sft-e-5e-05-s42-h_gate"]["payload"]["predictions"]
    answers["c1"] = answers["c1"][:8]  # one completion short of greedy + 2 x G4
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "assay_completion_count_differs"
    assert "G3a" not in state["gates"]
    assert first_order(fixture.stages_called())[-1] == "sft-e-5e-05-s42-h_gate"


def test_continuation_only_t_rl_is_refused(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    # A wrapper that reported only the 450 continuation updates' seconds.
    payload = fixture.scenario["rl-5e-05-s42-u600"]["payload"]
    payload["step_committed_active_seconds"] = series((151, 600, 1.0, 750.0))
    payload["committed_active_seconds"] = 750.0
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "continuation_seconds_not_cumulative"
    assert first_order(fixture.stages_called())[-1] == "rl-5e-05-s42-u600"


def test_continuation_must_extend_the_parent_accepted_sequence(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    manifests = fixture.scenario["rl-5e-05-s42-u600"]["payload"]["make_manifests"]
    manifests["accepted"] = SCAN[1:601]
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "continuation_accepted_sequence_not_extended"


@pytest.mark.parametrize("steps,reason", [
    # Restarted the clock instead of continuing SFT-E's.
    (series((1, SFT_T_UPDATES, 0.5, T_RL)), "sft_t_not_cumulative_over_sft_e"),
    # Ran a step past the first one at or above T_RL.
    (series((601, SFT_T_UPDATES - 1, 101.5, T_RL), (SFT_T_UPDATES, SFT_T_UPDATES, 1001.0,
                                                     1001.0)),
     "sft_t_overshoots_first_boundary_at_t_rl"),
])
def test_sft_t_lineage_and_overshoot_are_bounded(tmp_path, monkeypatch, steps, reason):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    payload = fixture.scenario["sft-t-5e-05-s42"]["payload"]
    payload["step_committed_active_seconds"] = steps
    payload["committed_active_seconds"] = steps[str(SFT_T_UPDATES)]
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == reason
    assert first_order(fixture.stages_called())[-1] == "sft-t-5e-05-s42"


def test_g1_failure_stops_before_any_rl_and_short_survivor_is_allowed(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, p0_pass=False, grace=5)
    pid_file = tmp_path / "short.pid"
    fixture.scenario["e0"]["behaviour"] = {"spawn": 0.5, "pid_file": str(pid_file)}
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "gate_failed"
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


def test_unexpected_exception_is_a_journalled_stop(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)

    def broken(self):
        raise TypeError("planted defect")

    monkeypatch.setattr(launch.Launcher, "sequence", broken)
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "unexpected_exception"
    stopped = fixture.journal()[-1]
    assert "planted defect" in stopped["error"] and "TypeError" in stopped["traceback"]
    # Terminal: a stopped study is never re-entered.
    assert launch.main([*fixture.argv, "--resume"]) == 2


def test_malformed_fragment_is_a_stop_not_a_crash(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    # The live reward rows are well formed, so only the fragment itself is malformed.
    fixture.scenario["rl-1e-05-s42-u150"]["behaviour"] = {"live_mixed": [True] * 150}
    fixture.scenario["rl-1e-05-s42-u150"]["payload"]["useful_mixed"] = [None] * 150
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "fragment_useful_mixed_invalid"


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
    state = run_until_stop(launch, fixture)
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


def test_second_owner_appearing_mid_stage_stops_it(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.scenario["e0"]["behaviour"] = {"sleep": 30}
    fixture.write()
    owners = []

    def start_owner():
        deadline = time.monotonic() + 20
        while not fixture.stages_called() and time.monotonic() < deadline:
            time.sleep(0.05)
        owners.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                                        fixture.sentinel]))

    thread = threading.Thread(target=start_owner)
    thread.start()
    started = time.monotonic()
    try:
        assert launch.main(fixture.argv) == 2
    finally:
        thread.join()
        for owner in owners:
            owner.kill()
            owner.wait()
    assert time.monotonic() - started < 20  # the 30 s stage was stopped, not waited out
    assert fixture.state()["stop_reason"] == "second_accelerator_owner"
    seen = [e for e in fixture.journal() if e["event"] == "second_owner_seen"]
    assert seen and owners[0].pid in [p["pid"] for p in seen[0]["processes"]]


def test_own_stage_is_never_a_second_owner(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, p0_pass=False)
    # The launcher's own stage command line names the owner pattern, and runs past
    # several owner polls: it must not be mistaken for a second owner.
    fixture.extra_e0_argv = ["--tag", fixture.sentinel]
    fixture.scenario["e0"]["behaviour"] = {"sleep": 5}
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "gate_failed"


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


PHASE_ENTRY = '''
def main(argv):
    return 0 if argv == ["forwarded"] else 9
'''


@pytest.mark.parametrize("device, code, guard", [
    ("mps", 0, "armed"), ("cpu", 0, "not armed: device cpu"), ("cuda", 1, None), (None, 1, None)])
def test_phase_worker_device_is_the_frozen_accelerate_device(tmp_path, device, code, guard):
    """The MPS driver guard follows ACCELERATE_TORCH_DEVICE, the device Accelerate uses: armed
    on mps, not armed on cpu (the dry run), and any other value refuses before any import.
    The fake torch reports MPS available in every case, so cpu is chosen, not inferred."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "phase_entry.py").write_text(PHASE_ENTRY)
    (pkg / "torch.py").write_text(FAKE_TORCH)
    env = {**os.environ, "PYTHONPATH": str(pkg), "PYTORCH_ENABLE_MPS_FALLBACK": "0",
           "HF_HUB_OFFLINE": "1"}
    env.pop("ACCELERATE_TORCH_DEVICE", None)
    env.pop("FAKE_DRIVER_BYTES", None)
    if device is not None:
        env["ACCELERATE_TORCH_DEVICE"] = device
    guards = tmp_path / "guards"
    result = subprocess.run(
        [sys.executable, "-B", str(PACKAGE / "phase_worker.py"), "--guard-dir", str(guards),
         "--stage", "s", "--kind", "train", "--entry", "phase_entry:main",
         "--entry-sha256", sha_file(pkg / "phase_entry.py"), "--", "forwarded"],
        env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == code, result.stderr
    if guard is None:
        assert "ACCELERATE_TORCH_DEVICE must be one of" in result.stderr
        assert not guards.exists()
        return
    (summary,) = guards.glob("s-*.json")
    stats = json.loads(summary.read_text())
    assert (stats["status"], stats["device"]) == ("completed", device)
    if guard == "armed":
        assert stats["recommended_max_memory_bytes"] == 48 * GiB
        assert "mps_driver_guard" not in stats
    else:
        assert stats["mps_driver_guard"] == guard
        assert "recommended_max_memory_bytes" not in stats


def test_changed_pinned_entry_is_refused(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.entry.write_text(fixture.entry.read_text() + "\n# edited\n")
    assert launch.main(fixture.argv) == 2
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("override,reason", [
    ({"audit_complete": False}, "g0_audit_incomplete"),
    ({"w_val_accuracy": 0.9}, "g0_failed"),
    ({"w_evaluator_refused": True}, "g0_failed"),
    ({"h_select": [("a1", "A"), ("a2", "A"), ("a3", "A"), ("b1", "B")]},
     "g0_holdout_not_2_per_class"),
    ({"evidence_h_gate": [{"id": "c1", "label": "A"}, {"id": "c2", "label": "A"},
                          {"id": "d1", "label": "B"}, {"id": "zz", "label": "B"}]},
     "g0_evidence_rows_differ_from_manifest"),
    ({"scan": sorted(SCAN)}, "g0_scan_not_seed42_order"),
    ({"pair_hamming": 5}, "g0_forced_fire_not_demonstrated"),
    ({"activated": False}, "g0_protocol_revision_not_activated"),
    ({"protocol_patch": {"substantial_gain_gates": {"accuracy": 0.7}}},
     "g0_protocol_frozen_gates_changed"),
    ({"uncommitted_protocol": True}, "g0_protocol_not_committed"),
    ({"decision": "# Decisions\n## D11 — design adopted\n"}, "g0_16_shot_plan_not_retired"),
    # Loose wording that the old 'contains 16-shot and retire' check passed.
    ({"decision": "- the 16-shot plan is NOT retired\n"}, "g0_16_shot_plan_not_retired"),
    ({"decision": "- 16-shot plan: retirement pending\n"}, "g0_16_shot_plan_not_retired"),
    ({"decision": "- Retired: the 16-shot plan (seed 2718, 1,600 images) is not retired\n"},
     "g0_16_shot_plan_not_retired"),
    ({"decision": "- Retired: the 16-shot plan (seed 314, 1,600 images)\n"},
     "g0_16_shot_plan_not_retired"),
    ({"decision": "- Retired: the 16-shot plan (seed 2718, 400 images)\n"},
     "g0_16_shot_plan_not_retired"),
    ({"decision": "- Proposal: Retired: 16-shot plan (seed 2718, 1,600 images)\n"},
     "g0_16_shot_plan_not_retired"),
    # The hedge or a missing fact on a wrapped continuation line of the same bullet.
    ({"decision": "- Retired: the 16-shot plan (seed 2718, 1,600 images,\n"
                  "  docs/audit.json) - retirement pending review\n"},
     "g0_16_shot_plan_not_retired"),
    ({"decision": "- **Retired: the 16-shot plan** (seed 2718,\n  1,600 images); proposed only\n"},
     "g0_16_shot_plan_not_retired"),
    ({"decision": "- Retired: the 16-shot plan (seed 2718,\n- 1,600 images)\n"},
     "g0_16_shot_plan_not_retired"),
    # Pool: 2,919 minus the drops, no drop inside it, H and the scan inside it.
    ({"pool_drops": ["drop0", "drop1"]}, "g0_pool_count_differs"),
    ({"pool": [i for i, _ in H_SELECT + H_GATE] + SCAN + [f"y{i:04d}" for i in range(1605)]
      + ["drop0", "drop1", "drop2"]}, "g0_pool_contains_a_phash_drop"),
    ({"pool": [i for i, _ in H_SELECT + H_GATE] + SCAN[1:] + [f"y{i:04d}" for i in range(1609)]},
     "g0_scan_outside_pool_minus_h"),
    ({"pool": [i for i, _ in H_SELECT[1:] + H_GATE] + SCAN + [f"y{i:04d}" for i in range(1609)]},
     "g0_holdout_outside_pool"),
    ({"pool_drops": "drop0"}, "g0_pool_drop_list_invalid"),
])
def test_g0_failure_does_not_launch(tmp_path, monkeypatch, capsys, override, reason):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch, **override)
    assert launch.main(fixture.argv) == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (printed["status"], printed["reason"]) == ("refused", reason)
    assert not (tmp_path / "state").exists()
    assert fixture.stages_called() == []


def test_g0_fixture_passes_verify_only(tmp_path, monkeypatch, capsys):
    """The control for the G0 refusals above: the unmodified fixture verifies."""
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    assert launch.main([*fixture.argv, "--verify-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def test_git_committed_detects_untracked_and_modified(tmp_path):
    launch = load_launch()
    git(tmp_path, "init", "-q")
    tracked = tmp_path / "tracked.json"
    tracked.write_text("{}")
    git(tmp_path, "add", "tracked.json")
    git(tmp_path, "commit", "-q", "-m", "one")
    untracked = tmp_path / "untracked.json"
    untracked.write_text("{}")
    assert launch.git_committed(tracked) is True
    assert launch.git_committed(untracked) is False
    tracked.write_text("{} ")
    assert launch.git_committed(tracked) is False


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


# Rows 0-59 mixed, then unmixed: the window over rows 21-120 holds 39 mixed (< 40), so the
# stop fires at row 120; with one fewer unmixed row the last window (rows 20-119) holds
# exactly 40, which is not below 0.40 and must not fire.
FIRES = [True] * 60 + [False] * 61
HOLDS = [True] * 60 + [False] * 60


def test_rolling_useful_mixed_stops_a_live_rl_arm(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.scenario["rl-1e-05-s42-u150"]["behaviour"] = {"live_mixed": FIRES, "sleep": 30}
    fixture.write()
    started = time.monotonic()
    state = run_until_stop(launch, fixture)
    assert time.monotonic() - started < 25  # the 30 s stage was stopped, not waited out
    assert state["stop_reason"] == "rolling_useful_mixed_below_0_40"
    finished = [e for e in fixture.journal() if e["event"] == "stage_finished"]
    assert (finished[-1]["stage"], finished[-1]["reason"]) == (
        "rl-1e-05-s42-u150", "rolling_useful_mixed_below_0_40")
    assert first_order(fixture.stages_called())[-1] == "rl-1e-05-s42-u150"


def test_unparseable_live_reward_row_stops_the_arm(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    # mixed: null in the trainer's reward rows; the live guard refuses rather than skips.
    fixture.scenario["rl-1e-05-s42-u150"]["payload"]["useful_mixed"] = [None] * 150
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "reward_groups_unparseable"
    assert first_order(fixture.stages_called())[-1] == "rl-1e-05-s42-u150"


def test_retirement_entry_reads_the_whole_wrapped_bullet():
    launch = load_launch()
    entries = launch.decision_bullets
    lines = ["# D11", "- Retired: the 16-shot plan (seed 2718,", "  1,600 images). Done.",
             "- Other: unrelated", "  continued, pending"]
    joined = entries(lines)
    assert joined == ["# D11", "- Retired: the 16-shot plan (seed 2718, 1,600 images). Done.",
                      "- Other: unrelated continued, pending"]
    assert launch.retirement_entry(joined[1])
    # The first physical line alone lacks '1,600 images'; only the joined bullet passes.
    assert not launch.retirement_entry(lines[1])
    hedged = entries(lines[:3] + ["  Retirement is pending."])
    assert not launch.retirement_entry(hedged[1])
    nested = entries(lines[:3] + ["  - status: pending"])
    assert nested[1].endswith("- status: pending")
    assert not launch.retirement_entry(nested[1])
    assert any(launch.retirement_entry(e) for e in entries(RETIRED_RECORD.splitlines()))


def test_second_reward_groups_after_the_first_stops(tmp_path):
    launch = load_launch()
    arm = tmp_path / "arm"
    first = arm / "trainer" / "reward_groups.jsonl"
    first.parent.mkdir(parents=True)
    monitor = launch.MixedMonitor(arm)
    assert monitor.poll() is None and monitor.path is None
    first.write_text("".join(json.dumps({"mixed": True}) + "\n" for _ in range(5)))
    assert monitor.poll() is None and monitor.path == first
    assert monitor.poll() is None  # the same single file, polled again, is not a stop
    first.open("a").write(json.dumps({"mixed": True}) + "\n")
    assert monitor.poll() is None and len(monitor.flags) == 6
    second = arm / "other" / "reward_groups.jsonl"
    second.parent.mkdir()
    second.write_text("")
    assert monitor.poll() == "reward_groups_ambiguous"


def test_second_reward_groups_mid_stage_stops_a_live_rl_arm(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.scenario["rl-1e-05-s42-u150"]["behaviour"] = {
        "live_mixed": HOLDS, "second_groups_after": 1.0, "sleep": 30}
    fixture.write()
    started = time.monotonic()
    state = run_until_stop(launch, fixture)
    assert time.monotonic() - started < 25
    assert state["stop_reason"] == "reward_groups_ambiguous"
    assert first_order(fixture.stages_called())[-1] == "rl-1e-05-s42-u150"


def test_rolling_useful_mixed_at_exactly_0_40_does_not_stop(tmp_path, monkeypatch):
    launch = load_launch()
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.scenario["rl-1e-05-s42-u150"]["behaviour"] = {"live_mixed": HOLDS, "sleep": 1}
    # The control ends at the next stage, so the RL stage itself must have passed.
    fixture.scenario["rl-1e-05-s42-u150-h_select"]["behaviour"] = {"exit": 9}
    fixture.write()
    state = run_until_stop(launch, fixture)
    assert state["stop_reason"] == "stage_failed"
    assert "rl-1e-05-s42-u150" in state["completed"]
    monitored = [e for e in fixture.journal() if e["event"] == "useful_mixed_monitored"]
    assert [(e["stage"], e["rows"]) for e in monitored] == [("rl-1e-05-s42-u150", 120)]


def test_voided_instance_time_is_charged_to_the_arm_cap(tmp_path, monkeypatch):
    # Cap 1.25 x 4.0 = 5.0 s. Each instance runs >= 3 s: alone under the cap, but the void
    # instance's >= 3 s stays charged, so the re-run reaches the cap. Without the charge
    # the re-run would pass.
    fixture = Fixture(tmp_path, monkeypatch, e0_forecast=4.0)
    marker = tmp_path / "interrupted-once"
    fixture.scenario["e0"]["behaviour"] = {"interrupt_once": str(marker),
                                           "sleep_before_interrupt": 3.0, "sleep": 3.0}
    fixture.write()
    command = [sys.executable, "-B", str(PACKAGE / "launch.py"), *fixture.argv]
    first = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert first.returncode == 3, first.stdout + first.stderr
    second = subprocess.run([*command, "--resume"], capture_output=True, text=True,
                            timeout=120)
    assert second.returncode == 2, second.stdout + second.stderr
    state = fixture.state()
    assert state["stop_reason"] == "arm_cap"
    [void] = state["void_instances"]
    assert (void["stage"], void["arm"], void["instance"]) == ("e0", "e0", 1)
    assert void["instance_wall_seconds"] >= 3.0
    assert state["void_seconds_by_arm"]["e0"] == void["instance_wall_seconds"]
    started = [e for e in fixture.journal() if e["event"] == "stage_started"]
    assert [(e["instance"], e["void_seconds"]) for e in started] == [
        (1, 0.0), (2, void["instance_wall_seconds"])]
    finished = [e for e in fixture.journal() if e["event"] == "stage_finished"]
    assert finished[-1]["reason"] == "arm_cap"
    # The re-run was stopped well before its own 3 s: only the remainder was available.
    assert finished[-1]["elapsed_seconds"] < 2.5
    assert finished[-1]["arm_wall_seconds"] >= 5.0
