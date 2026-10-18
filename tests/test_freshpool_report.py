"""scripts/freshpool_report.py over hand-built synthetic state directories (never a live run).

The state directory is written here the way the launcher writes it (journal events, stage
fragments, gate evidence and reports). The gate evidence documents are assembled by this
file independently of the launcher, so the report's own rebuild is checked against them.
Every expected value below (selections, gate verdicts, accuracies, intervals, target) is
written out from the fixture design, never read back from the report.
"""

import copy
import hashlib
import importlib.util
import json
import random
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_freshpool_gates.py"
GATES = runpy.run_path(str(EVALUATOR))
_spec = importlib.util.spec_from_file_location("freshpool_report_under_test",
                                               ROOT / "scripts" / "freshpool_report.py")
report_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report_module)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


W_SHA = sha_text("W adapter")
LABELS = ["A", "B"]
OTHER = {"A": "B", "B": "A"}
SPLIT_ROWS = {"h_select": [("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")],
              "h_gate": [("c1", "A"), ("c2", "A"), ("d1", "B"), ("d2", "B")]}
COMPLETIONS = {"h_select": 5, "h_gate": 9}
SCAN = sorted(f"x{i:03d}" for i in range(500))
random.Random(42).shuffle(SCAN)
# Every image of a split gets the same number of correct completions, so each paired
# bootstrap draw equals the point difference: the 95% interval and 90% bound are exact.
# Correct completions per image: W 1/5 H-select (.2), 2/9 H-gate (.2222).
# RL at 150 on H-select: 1e-5 .4, 5e-5 .6, 1e-4 .2 -> continuers 5e-5 then 1e-5.
# RL at 600 on H-select: 1e-5 .8, 5e-5 .6 -> RL* = rl-1e-05-s42-u600; its H-gate 7/9.
# SFT-E on H-select: .4, .6, .4 -> SFT-E* 5e-5, alone within 2 pp; its H-gate 5/9.
# SFT-T 5e-5 H-select .6, H-gate 6/9. EG-SFT 5e-5 H-gate 5/9.
RL150 = {"1e-05": (1e-5, 2), "5e-05": (5e-5, 3), "1e-04": (1e-4, 1)}
RL600 = {"5e-05": (5e-5, 3), "1e-05": (1e-5, 4)}
SFT_E = {"1e-05": (1e-5, 2), "5e-05": (5e-5, 3), "1e-04": (1e-4, 2)}
RL_STAR, SFT_E_STAR, SFT_T_STAR = "rl-1e-05-s42-u600", "sft-e-5e-05-s42", "sft-t-5e-05-s42"
EG = "egsft-5e-05-s42"


def predictions(split, correct):
    n = COMPLETIONS[split]
    return {i: [label] * correct + [OTHER[label]] * (n - correct)
            for i, label in SPLIT_ROWS[split]}


def assay(run_id, split, adapter, correct):
    return {"run_id": run_id, "split": split, "adapter_sha256": adapter,
            "predictions": predictions(split, correct), "eos_fraction": 1.0}


def train(run_id, arm, lr, updates, seconds):
    rl = arm == "RL"
    return {"status": "completed", "run_id": run_id, "arm": arm, "lr": lr, "seed": 42,
            "updates": updates, "committed_active_seconds": seconds,
            "adapter_sha256": sha_text(run_id), "reference_sha256": W_SHA,
            "adapter_path": f"/fixture/{run_id}/final_adapter",
            "checkpoint": f"/fixture/{run_id}/checkpoint-{updates}",
            # Trailing 100 of 150 all mixed: G2's rolling fraction 1.0.
            "useful_mixed": [True] * updates if rl else []}


def g1_document():
    # p = .5 (u = .875, accepted) for the first 200, p = .95 (rejected) after: acceptance .5.
    journal = [{"id": i, "p": p, "u": 1 - p**4 - (1 - p) ** 4, "accepted": p == 0.5,
                "scorer_seconds": 1.0, "adapter_sha256": W_SHA}
               for n, i in enumerate(SCAN[:400]) for p in [0.5 if n < 200 else 0.95]]
    rollouts = []
    for image in SCAN[:64]:
        for seed in (42, 314):
            # 112 of 128 groups mixed: observed .875, equal to the predicted mean u .875.
            outcome = [True, True, False, False] if len(rollouts) < 112 else [False] * 4
            rollouts.append({"id": image, "seed": seed, "completions": [
                {"correct": c, "valid": True, "eos": True, "canonical_tokenization": True,
                 "label_only": True}
                for c in outcome]})
    return {"seed": 42, "adapter_sha256": W_SHA, "scan_order": SCAN, "journal": journal,
            "rollouts": rollouts}


W_ENDPOINT = {"run_id": "W", "arm": "W", "lr": None, "seed": None, "updates": 0,
              "committed_active_seconds": 0, "adapter_sha256": W_SHA,
              "reference_sha256": W_SHA, "useful_mixed": [], "splits": {}}
TRAIN_KEYS = ("run_id", "arm", "lr", "seed", "updates", "committed_active_seconds",
              "adapter_sha256", "reference_sha256", "useful_mixed")


class State:
    """A launcher state directory written stage by stage in the launcher's sequence."""

    def __init__(self, tmp, *, receipt_patch=None):
        self.tmp, self.dir = tmp, tmp / "state"
        (self.dir / "stages").mkdir(parents=True)
        (self.dir / "gates").mkdir()
        self.events, self.completed, self.time = [], {}, 1000.0
        base = tmp / "evidence-base.json"
        self.base = {"labels": LABELS, "a_star": 0.75, "b4_edge_extension": False,
                     "rows": {s: [{"id": i, "label": label} for i, label in rows]
                              for s, rows in SPLIT_ROWS.items()}}
        base.write_text(json.dumps(self.base))
        # a* = .75: D_req = max(.05, .2 x .25, .80 - .75) = .05 (full mode, margin .02),
        # so the RL* VAL target is .80.
        receipt = {"a_star": 0.75, "d_req": 0.05, "mode": "full", "g3_margin": 0.02,
                   "branches": {"B2": {"rl_val_needed_lower_bound": 0.80}},
                   **(receipt_patch or {})}
        branch = tmp / "branch.json"
        branch.write_text(json.dumps(receipt))
        self.plan = {"evaluator": str(EVALUATOR),
                     "file_sha256": {str(EVALUATOR): sha_file(EVALUATOR)},
                     "state_directory": str(self.dir),
                     "w_endpoint": {"adapter_sha256": W_SHA, "adapter_path": "/fixture/w"},
                     "evidence_base": {"path": str(base), "sha256": sha_file(base)},
                     "g0": {"receipts": {"branch": {"path": str(branch),
                                                    "sha256": sha_file(branch)}}}}
        self.write_plan()
        self.event("launch_started", pid=1, plan_sha256=self.plan_sha,
                   g0={"fixture": True}, d_req=GATES["d_req"](0.75))

    def write_plan(self):
        self.plan_path = self.tmp / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.plan_sha = sha_file(self.plan_path)

    def event(self, kind, **fields):
        self.time += 1
        self.events.append({"event": kind, "time": self.time, **fields})

    def stage(self, stage_id, payload, *, finish=True):
        directory = self.dir / "stages" / stage_id / "instance-1"
        directory.mkdir(parents=True)
        self.event("stage_started", stage=stage_id, arm=stage_id, instance=1,
                   cap_seconds=750.0, inherited_seconds=0.0, void_seconds=0.0)
        if not finish:
            return
        fragment = directory / "fragment.json"
        fragment.write_text(json.dumps(payload))
        self.event("stage_finished", stage=stage_id, exit_code=0, reason=None,
                   elapsed_seconds=10.0)
        self.completed[stage_id] = payload
        self.event("stage_completed", stage=stage_id, fragment=str(fragment),
                   fragment_sha256=sha_file(fragment))

    def evidence(self):
        """The evaluator's evidence document, assembled here from the fragments."""
        endpoints = {"W": copy.deepcopy(W_ENDPOINT)}
        for payload in self.completed.values():
            if payload.get("arm") in ("RL", "SFT-E", "SFT-T", "EG-SFT"):
                endpoints[payload["run_id"]] = {**{k: payload[k] for k in TRAIN_KEYS},
                                                "splits": {}}
        for payload in self.completed.values():
            if "split" in payload:
                endpoints[payload["run_id"]]["splits"][payload["split"]] = {
                    "predictions": payload["predictions"],
                    "eos_fraction": payload["eos_fraction"]}
        document = {k: self.base[k] for k in ("labels", "rows", "a_star", "b4_edge_extension")}
        document["endpoints"] = list(endpoints.values())
        if "p0" in self.completed:
            document["g1"] = self.completed["p0"]
        return document

    def gate(self, name):
        evidence = self.dir / "gates" / f"{name}-evidence-1.json"
        evidence.write_text(json.dumps(self.evidence(), indent=2))
        self.event("gate_started", gate=name, evidence=str(evidence),
                   evidence_sha256=sha_file(evidence))
        result = GATES["evaluate"](json.loads(evidence.read_text()))
        report = self.dir / "gates" / f"{name}-report-1.json"
        report.write_text(json.dumps(result, indent=2, allow_nan=False))
        if name == "final":
            self.event("final_check", report=str(report), passed=True)
            return result
        gate = result["gates"][name]
        verdict = "pass" if gate.get("status") == "evaluated" and gate.get(
            "pass") is True and result["failed_at"] is None else "gate_failed"
        self.event("gate_finished", gate=name, exit_code=0, report=str(report),
                   report_sha256=sha_file(report), verdict=verdict, stderr="")
        return result

    def select(self, name, value):
        self.event("selection", name=name, value=value)

    def flush(self, status="running"):
        lines = "".join(json.dumps(e) + "\n" for e in self.events)
        (self.dir / "journal.jsonl").write_text(lines)
        state = {"plan_sha256": self.plan_sha, "status": status,
                 "completed": {s: {"fragment": p} for s, p in self.completed.items()}}
        (self.dir / "state.json").write_text(json.dumps(state))

    def build(self):
        return report_module.build(self.dir, self.plan_path)


def through_g1(state):
    state.stage("e0", {"checks": {"resume_2_plus_2_identical": True},
                       "sampler_teacher_max_abs_logprob_difference": 0.0005})
    state.stage("W-h_select", assay("W", "h_select", W_SHA, 1))
    state.stage("W-h_gate", assay("W", "h_gate", W_SHA, 2))
    state.stage("p0", g1_document())
    state.gate("G1")


def full_development(state):
    through_g1(state)
    for tag, (lr, correct) in RL150.items():
        run_id = f"rl-{tag}-s42-u150"
        state.stage(run_id, train(run_id, "RL", lr, 150, 250.0))
        state.stage(f"{run_id}-h_select", assay(run_id, "h_select", sha_text(run_id), correct))
    state.gate("G2")
    state.select("rl_continuers", ["rl-5e-05-s42-u150", "rl-1e-05-s42-u150"])
    for tag, (lr, correct) in RL600.items():
        run_id = f"rl-{tag}-s42-u600"
        state.stage(run_id, train(run_id, "RL", lr, 600, 1000.0))
        state.stage(f"{run_id}-h_select", assay(run_id, "h_select", sha_text(run_id), correct))
    state.select("rl_star", RL_STAR)
    for tag, (lr, correct) in SFT_E.items():
        run_id = f"sft-e-{tag}-s42"
        state.stage(run_id, train(run_id, "SFT-E", lr, 600, 100.0))
        state.stage(f"{run_id}-h_select", assay(run_id, "h_select", sha_text(run_id), correct))
    state.select("sft_e_star", SFT_E_STAR)
    state.stage(f"{RL_STAR}-h_gate", assay(RL_STAR, "h_gate", sha_text(RL_STAR), 7))
    state.stage(f"{SFT_E_STAR}-h_gate", assay(SFT_E_STAR, "h_gate", sha_text(SFT_E_STAR), 5))
    state.gate("G3a")
    state.select("sft_t_lrs", [5e-5])
    state.stage(SFT_T_STAR, train(SFT_T_STAR, "SFT-T", 5e-5, 2400, 1000.0))
    state.stage(f"{SFT_T_STAR}-h_select",
                assay(SFT_T_STAR, "h_select", sha_text(SFT_T_STAR), 3))
    state.select("sft_t_star", SFT_T_STAR)
    state.stage(f"{SFT_T_STAR}-h_gate", assay(SFT_T_STAR, "h_gate", sha_text(SFT_T_STAR), 6))
    state.gate("G3b")
    state.stage(EG, train(EG, "EG-SFT", 5e-5, 600, 100.0))
    state.stage(f"{EG}-h_gate", assay(EG, "h_gate", sha_text(EG), 5))
    state.gate("final")
    state.event("study_development_complete", awaiting="G4")


@pytest.fixture
def full(tmp_path):
    state = State(tmp_path)
    full_development(state)
    state.flush("completed")
    return state


def run(state, *extra):
    return report_module.main(["--state-directory", str(state.dir), "--plan",
                               str(state.plan_path), *extra])


def refusal(capsys):
    lines = capsys.readouterr().out.strip().splitlines()
    return json.loads(lines[-1])


def test_full_development_run_reports_every_gate(full, tmp_path):
    out_json, out_md = tmp_path / "report.json", tmp_path / "report.md"
    assert run(full, "--json", str(out_json), "--markdown", str(out_md)) == 0
    report = json.loads(out_json.read_text())
    assert [(g["gate"], g["verdict"], g["matches_launcher_report"]) for g in report["gates"]] \
        == [("G1", "pass", True), ("G2", "pass", True), ("G3a", "pass", True),
            ("G3b", "pass", True)]
    assert report["gate_runs"][-1]["gate"] == "final"
    assert report["gate_runs"][-1]["verdict"] is True
    assert report["current_evaluation"]["awaiting"] == "G4"
    assert report["current_evaluation"]["failed_at"] is None
    assert report["selections_recomputed"] == {
        "rl_continuers": ["rl-5e-05-s42-u150", "rl-1e-05-s42-u150"], "rl_star": RL_STAR,
        "sft_e_star": SFT_E_STAR, "sft_t_star": SFT_T_STAR}
    g2 = report["gates"][1]["recomputed"]
    assert g2["continue_to_600"] == ["rl-5e-05-s42-u150", "rl-1e-05-s42-u150"]
    assert g2["values"]["h_select_delta_vs_w"] == pytest.approx(0.6 - 0.2)
    g3a = report["gates"][2]["recomputed"]
    assert g3a["values"]["rl_minus_sft_e"] == pytest.approx(7 / 9 - 5 / 9)
    assert report["gates"][3]["recomputed"]["values"]["rl_minus_sft_t"] == pytest.approx(1 / 9)
    assert all(s["status"] == "completed" for s in report["stages"])
    assert len(report["stages"]) == 27
    assert report["e0"]["all_checks_true"] and report["e0"]["logprob_within_0_001"]
    # Target: a* .75 + D_req .05.
    assert report["target"]["rl_val_target_lower_bound"] == pytest.approx(0.80)
    assert report["target"]["d_req"] == pytest.approx(0.05)
    assert report["target"]["rl_star_val_accuracy"] is None
    md = out_md.read_text()
    assert "### G3b: pass" in md and "0.8000" in md


def test_h_accuracies_and_paired_intervals(full):
    report = full.build()
    rows = {(r["candidate"], r["split"]): r for r in report["h_accuracy"]}
    w = rows[("W", "h_select")]
    assert w["candidate_accuracy"] == pytest.approx(0.2) and w["difference"] is None
    assert rows[("W", "h_gate")]["candidate_accuracy"] == pytest.approx(2 / 9)
    rl150 = rows[("rl-1e-04-s42-u150", "h_select")]
    assert rl150["candidate_accuracy"] == pytest.approx(0.2)
    assert rl150["difference"] == pytest.approx(0.0)
    star = rows[(RL_STAR, "h_gate")]
    assert star["candidate_accuracy"] == pytest.approx(7 / 9)
    assert star["difference"] == pytest.approx(5 / 9)
    # Every image of a split scores alike: every draw is the point difference.
    assert star["ci95"] == pytest.approx([5 / 9, 5 / 9])
    assert star["lb90_one_sided"] == pytest.approx(5 / 9)
    assert star["draws"] == 20000 and star["seed"] == 2718
    versus = {(r["comparator_role"], r["split"]): r for r in report["rl_star_comparisons"]}
    assert versus[("W", "h_select")]["difference"] == pytest.approx(0.8 - 0.2)
    assert versus[("SFT-E*", "h_gate")]["difference"] == pytest.approx(2 / 9)
    assert versus[("SFT-E*", "h_gate")]["ci95"] == pytest.approx([2 / 9, 2 / 9])
    assert versus[("SFT-T*", "h_gate")]["difference"] == pytest.approx(1 / 9)
    assert versus[("EG-SFT", "h_gate")]["difference"] == pytest.approx(2 / 9)
    assert versus[("EG-SFT", "h_select")] == {"comparator_role": "EG-SFT", "split": "h_select",
                                               "status": "not yet measured"}
    assert versus[("W", "val")]["status"] == "not yet measured"


def test_interval_is_a_real_interval_when_images_differ(tmp_path):
    """Mixed per-image scores: the two-sided 95% interval brackets the point difference and
    the one-sided 90% bound sits at or above the 95% lower end."""
    state = State(tmp_path)
    through_g1(state)
    run_id = "rl-1e-05-s42-u150"
    state.stage(run_id, train(run_id, "RL", 1e-5, 150, 250.0))
    uneven = assay(run_id, "h_select", sha_text(run_id), 1)
    # a1: 5/5, a2: 1/5, b1: 4/5, b2: 1/5 -> class means .6, .5; mean .55 vs W .2: +.35.
    for image, correct in (("a1", 5), ("b1", 4)):
        label = dict(SPLIT_ROWS["h_select"])[image]
        uneven["predictions"][image] = [label] * correct + [OTHER[label]] * (5 - correct)
    state.stage(f"{run_id}-h_select", uneven)
    state.flush()
    report = state.build()
    row = next(r for r in report["h_accuracy"] if r["candidate"] == run_id)
    assert row["difference"] == pytest.approx(0.35)
    low, high = row["ci95"]
    assert low < 0.35 < high
    assert low <= row["lb90_one_sided"] <= 0.35
    # Two images per class: the class-A mean over resamples ranges .2-1.0, class B .2-.8.
    assert low >= (0.2 + 0.2) / 2 - 0.2 - 1e-9 and high <= (1.0 + 0.8) / 2 - 0.2 + 1e-9


def test_mid_stage_reports_the_running_stage_and_waits(tmp_path):
    state = State(tmp_path)
    through_g1(state)
    run_id = "rl-1e-05-s42-u150"
    state.stage(run_id, train(run_id, "RL", 1e-5, 150, 250.0))
    state.stage(f"{run_id}-h_select", assay(run_id, "h_select", sha_text(run_id), 2))
    state.stage("rl-5e-05-s42-u150", None, finish=False)
    state.flush()
    report = state.build()
    assert [(g["gate"], g.get("verdict", g["state"])) for g in report["gates"]] == [
        ("G1", "pass"), ("G2", "not started"), ("G3a", "not started"),
        ("G3b", "not started")]
    assert report["stages"][-1]["stage"] == "rl-5e-05-s42-u150"
    assert report["stages"][-1]["status"] == "running"
    # One of three RL LRs at 150: the evaluator refuses the partial family; shown, not raised.
    assert report["current_evaluation"] is None
    assert "RL at seed 42 needs exactly one endpoint per LR" in report[
        "current_evaluation_error"]
    assert report["rl_star_comparisons"] == []


def test_partial_trailing_line_is_ignored_and_unadmitted_fragment_listed(full):
    with (full.dir / "journal.jsonl").open("a") as handle:
        handle.write('{"event": "stage_sta')
    stray = full.dir / "stages" / "rl-2e-04-s42-u150" / "instance-1"
    stray.mkdir(parents=True)
    (stray / "fragment.json").write_text("{}")
    report = full.build()
    assert report["journal_partial_tail_bytes"] == len('{"event": "stage_sta')
    assert report["unadmitted_fragments"] == ["stages/rl-2e-04-s42-u150/instance-1/fragment.json"]


def rewrite_journal(state, change):
    path = state.dir / "journal.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events = change(events) or events
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def find(events, kind, **match):
    return next(e for e in events if e["event"] == kind
                and all(e.get(k) == v for k, v in match.items()))


def tamper_fragment(state):
    path = state.dir / "stages" / "W-h_select" / "instance-1" / "fragment.json"
    path.write_text(json.dumps(assay("W", "h_select", W_SHA, 2)))


def tamper_report(state):
    path = state.dir / "gates" / "G2-report-1.json"
    report = json.loads(path.read_text())
    report["gates"]["G2"]["values"]["trailing_useful_mixed"] = 0.61
    path.write_text(json.dumps(report))
    rewrite_journal(state, lambda events: find(events, "gate_finished", gate="G2").update(
        report_sha256=sha_file(path)))


def tamper_evidence(state):
    path = state.dir / "gates" / "G1-evidence-1.json"
    evidence = json.loads(path.read_text())
    evidence["a_star"] = 0.7
    path.write_text(json.dumps(evidence))
    rewrite_journal(state, lambda events: find(events, "gate_started", gate="G1").update(
        evidence_sha256=sha_file(path)))


def tamper_assay_adapter(state):
    stage = "rl-1e-05-s42-u150-h_select"
    path = state.dir / "stages" / stage / "instance-1" / "fragment.json"
    payload = json.loads(path.read_text())
    payload["adapter_sha256"] = sha_text("another adapter")
    path.write_text(json.dumps(payload))
    rewrite_journal(state, lambda events: find(events, "stage_completed", stage=stage).update(
        fragment_sha256=sha_file(path)))
    saved = json.loads((state.dir / "state.json").read_text())
    saved["completed"][stage]["fragment"] = payload
    (state.dir / "state.json").write_text(json.dumps(saved))


def tamper_state_file(state):
    path = state.dir / "state.json"
    saved = json.loads(path.read_text())
    saved["completed"]["W-h_gate"]["fragment"]["eos_fraction"] = 0.5
    path.write_text(json.dumps(saved))


def duplicate_completion(events):
    events.append(dict(find(events, "stage_completed", stage="e0")))


def unparseable_line(state):
    path = state.dir / "journal.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    lines.insert(3, "not json\n")
    path.write_text("".join(lines))


def empty_journal(state):
    (state.dir / "journal.jsonl").write_text("")


@pytest.mark.parametrize("tamper, reason", [
    (lambda s: (s.dir / "stages" / "W-h_select" / "instance-1" / "fragment.json").unlink(),
     "fragment_missing"),
    (tamper_fragment, "fragment_changed_since_completion"),
    (tamper_report, "recomputed_gate_differs"),
    (tamper_evidence, "gate_evidence_differs_from_fragments"),
    (tamper_assay_adapter, "assay_adapter_differs_from_endpoint"),
    (tamper_state_file, "state_fragment_differs"),
    (lambda s: rewrite_journal(s, duplicate_completion), "stage_completed_twice"),
    (lambda s: rewrite_journal(s, lambda e: find(e, "gate_finished", gate="G3a").update(
        verdict="gate_failed")), "recomputed_verdict_differs"),
    (lambda s: rewrite_journal(s, lambda e: find(e, "selection", name="rl_star").update(
        value="rl-5e-05-s42-u600")), "selection_differs"),
    (lambda s: rewrite_journal(s, lambda e: e[0].update(plan_sha256="0" * 64)),
     "journal_plan_differs"),
    (lambda s: rewrite_journal(s, lambda e: e.pop(0) and None),
     "journal_does_not_start_with_launch"),
    (unparseable_line, "journal_line_unparseable"),
    (empty_journal, "journal_empty"),
    (lambda s: (s.dir / "journal.jsonl").unlink(), "journal_missing"),
    (lambda s: (s.dir / "gates" / "G3b-evidence-1.json").unlink(), "gate_evidence_missing"),
])
def test_missing_or_inconsistent_state_refuses(full, capsys, tamper, reason):
    tamper(full)
    assert run(full) == 2
    assert refusal(capsys)["reason"] == reason


def test_inconsistent_branch_target_refuses(tmp_path, capsys):
    state = State(tmp_path, receipt_patch={"branches": {"B2": {
        "rl_val_needed_lower_bound": 0.81}}})
    through_g1(state)
    state.flush()
    assert run(state) == 2
    assert refusal(capsys)["reason"] == "branch_receipt_target_inconsistent"


def test_receipt_d_req_must_equal_the_evaluator(tmp_path, capsys):
    state = State(tmp_path, receipt_patch={"d_req": 0.10})
    through_g1(state)
    state.flush()
    assert run(state) == 2
    assert refusal(capsys)["reason"] == "branch_receipt_disagrees_with_evaluator"


def test_evaluator_differing_from_the_pin_refuses(tmp_path, capsys):
    state = State(tmp_path)
    through_g1(state)
    state.plan["file_sha256"][str(EVALUATOR)] = "0" * 64
    state.write_plan()
    state.events[0]["plan_sha256"] = state.plan_sha
    state.flush()
    assert run(state) == 2
    assert refusal(capsys)["reason"] == "report_code_differs_from_pin"


def test_output_inside_the_state_directory_refuses(full, capsys):
    assert run(full, "--json", str(full.dir / "report.json")) == 2
    assert refusal(capsys)["reason"] == "output_inside_state_directory"
    assert not (full.dir / "report.json").exists()


def test_watch_exits_at_the_terminal_event(full, tmp_path, capsys):
    assert run(full, "--watch", "0.01", "--markdown", str(tmp_path / "r.md")) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["gates"] == {"G1": "pass", "G2": "pass", "G3a": "pass", "G3b": "pass"}
