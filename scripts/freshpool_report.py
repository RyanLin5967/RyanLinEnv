"""Report a freshpool launch state directory: every stage, every gate recomputed, every CI.

  python scripts/freshpool_report.py --state-directory runs/aircraft-freshpool-v1 \\
      [--plan scripts/freshpool_launch/production/plan.json] \\
      [--json report.json] [--markdown report.md] [--watch 60]

Read-only over the state directory: it writes only the --json / --markdown paths (never
inside the state directory) and prints the markdown when no --markdown path is given.
``--watch N`` re-checks every N seconds, re-reports each time the journal grows, and exits
once the study is stopped or its development stages are complete (bound it with timeout).

What it does, in order, and the refusal (exit 2, one JSON line on stdout) for each failure:
  * the plan's own SHA256 must equal every launch_started/launch_resumed plan_sha256 in the
    journal (``journal_plan_differs``); the evaluator, the shared comparison module and the
    launcher that this report imports must equal the plan's pins (``report_code_unpinned``,
    ``report_code_differs_from_pin``); the evidence base and branch receipt must equal their
    pinned SHA256s;
  * the journal must exist, be non-empty, start with launch_started and parse line by line
    (``journal_missing``, ``journal_empty``, ``journal_line_unparseable``). A trailing line
    without its newline is a write in progress: it is ignored and reported, not parsed;
  * every ``stage_completed`` admits exactly one fragment, at stages/<stage>/instance-<k>/
    fragment.json for the instance that stage last started, whose bytes still hash to the
    journalled SHA256 (``fragment_missing``, ``fragment_changed_since_completion``,
    ``stage_completed_twice``, ``stage_completed_without_start``, ...). state.json, when
    present, must hold the same payload for every admitted stage;
  * the evaluator's evidence is rebuilt from the admitted fragments by the launcher's OWN
    ``Launcher.evidence`` (imported, not rewritten), once per ``gate_started`` from the
    fragments completed before it. Each rebuilt document must equal the evidence file the
    launcher wrote (``gate_evidence_differs_from_fragments``); each gate is re-decided by
    the evaluator's own ``evaluate`` and must agree with the launcher's recorded report and
    verdict (``recomputed_gate_differs``, ``recomputed_verdict_differs``);
  * journal selections (continuers, RL*, SFT-E*, SFT-T*) must equal the evaluator's own
    selection functions over the admitted evidence (``selection_differs``).

Then it reports: per-stage status; E0; each gate with checks and values; H-select / H-gate
mean-class accuracy (every completion, greedy included, as the evaluator scores it) of every
endpoint with its paired difference vs W; RL* vs W, SFT-E*, SFT-T* and EG-SFT on H-select,
H-gate and (when present) VAL; and the branch receipt's VAL target a* + D_req.

Uncertainty is the protocol's (docs/aircraft-rl-protocol-freshpool-v1.json
evaluation.uncertainty: paired, 20,000 draws, seed 2718, images resampled within each class),
computed only by ``vision_rl.classification_comparison`` at its defaults, which equal those
values. On H splits that module returns lower bounds only: the 95% interval's lower end is
its 0.025 bound, its upper end is minus the 0.025 bound of the swapped comparison (same
draws, negated), and the one-sided 90% bound is the 0.10 quantile G3a uses. On VAL the
interval is the module's stratified_image_ci95 and the sensitivity class_cluster_ci95.

TEST is never scored here, whatever the evidence holds. H results are native-TRAIN
development evidence; they are not comparable to the VAL target and are not presented so.

Blind spots, stated: it trusts what the fragments say beyond the launcher's own checks (it
cannot see whether predictions came from the stated adapter); it reads a live journal
without a lock, so a report is a snapshot of the complete lines at read time; fragments on
disk that the journal never admitted (void, in flight, or refused by the launcher) are
listed and never used; the current-state evaluation is informational when a stage family is
half done (the evaluator refuses a partial LR family; that refusal is shown, not raised).
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts" / "freshpool_launch" / "launch.py"
DEFAULT_PLAN = ROOT / "scripts" / "freshpool_launch" / "production" / "plan.json"
H_SPLITS = ("h_select", "h_gate")
COMPARISON_SPLITS = (*H_SPLITS, "val")  # never "test"
GATES = ("G1", "G2", "G3a", "G3b")
LATER_GATES = ("G4", "G5", "G6")
SELECTIONS = ("rl_continuers", "rl_star", "sft_e_star", "sft_t_lrs", "sft_t_star")
TERMINAL_EVENTS = ("study_stopped", "study_development_complete")
TWO_SIDED_LOWER = 0.025  # lower end of the protocol's two-sided 95% interval
E0_MAX_LOGPROB_DIFFERENCE = 0.001  # study E0: sampler-teacher agreement <= 0.001
FLOAT_SLACK = 1e-12


class Refuse(Exception):
    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roundtrip(value):
    """What the launcher's files hold for ``value``: JSON, NaN refused, tuples as lists."""
    return json.loads(json.dumps(value, allow_nan=False))


def load_launch():
    spec = importlib.util.spec_from_file_location("freshpool_launch_for_report", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- plan, pins and receipts


def load_plan(plan_path, launch):
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        raise Refuse("plan_missing", plan=str(plan_path))
    plan = json.loads(plan_path.read_text())
    pins = {launch.resolve(name).resolve(): value
            for name, value in (plan.get("file_sha256") or {}).items()}
    evaluator_path = launch.resolve(plan["evaluator"]).resolve()
    if evaluator_path not in pins:
        raise Refuse("report_code_unpinned", file=str(evaluator_path))
    evaluator = launch.load_evaluator(plan)
    comparison = Path(sys.modules[evaluator["compare_classification"].__module__].__file__)
    # Code this report runs must be the code the launcher pinned, where the plan pins it.
    for path in (evaluator_path, LAUNCH.resolve(), comparison.resolve()):
        if path in pins and sha(path) != pins[path]:
            raise Refuse("report_code_differs_from_pin", file=str(path))
    pinned_names = {Path(name).as_posix(): value
                    for name, value in (plan.get("file_sha256") or {}).items()}
    for name, value in pinned_names.items():
        if name.endswith("src/vision_rl/classification_comparison.py") and sha(
                comparison) != value:
            raise Refuse("report_code_differs_from_pin", file=str(comparison))
    return plan, sha(plan_path), evaluator


def pinned_json(launch, entry, reason):
    entry = entry or {}
    path = launch.resolve(entry.get("path", "/nonexistent"))
    if not path.is_file() or sha(path) != entry.get("sha256"):
        raise Refuse(reason, path=str(path))
    return json.loads(path.read_text())


def target(launch, plan, evaluator, base, journal):
    receipt = pinned_json(launch, ((plan.get("g0") or {}).get("receipts") or {}).get("branch"),
                          "branch_receipt_changed")
    a_star = receipt.get("a_star")
    branch = evaluator["d_req"](a_star)
    if (abs(receipt.get("d_req", -1) - branch["d_req"]) > FLOAT_SLACK
            or receipt.get("mode") != branch["mode"] or base.get("a_star") != a_star):
        raise Refuse("branch_receipt_disagrees_with_evaluator", receipt_d_req=receipt.get("d_req"),
                     evaluator_d_req=branch["d_req"])
    value = a_star + branch["d_req"]
    published = (((receipt.get("branches") or {}).get("B2") or {})
                 .get("rl_val_needed_lower_bound"))
    if published is not None and abs(published - value) > FLOAT_SLACK:
        raise Refuse("branch_receipt_target_inconsistent", published=published, computed=value)
    for event in journal:
        if event["event"] in ("launch_started", "launch_resumed") and "d_req" in event:
            if roundtrip(event["d_req"]) != roundtrip(branch):
                raise Refuse("journal_d_req_differs", journal=event["d_req"])
    return {"a_star": a_star, "d_req": branch["d_req"], "mode": branch["mode"],
            "g3_margin": branch["g3_margin"], "rl_val_target_lower_bound": value,
            "published_target": published,
            "note": "Lower bound: G4 also needs +0.05 over the strongest SFT on VAL, whose "
                    "fresh-pool endpoints can raise a*; RL* must also reach 0.80."}


# ---------------------------------------------------------------- journal and fragments


def read_journal(state):
    path = state / "journal.jsonl"
    if not path.is_file():
        raise Refuse("journal_missing", path=str(path))
    data = path.read_bytes()
    pieces = data.split(b"\n")
    partial = pieces.pop()  # b"" when the file ends in a newline
    events = []
    for number, line in enumerate(pieces, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            raise Refuse("journal_line_unparseable", line=number) from None
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise Refuse("journal_line_unparseable", line=number)
        events.append(event)
    if not events:
        raise Refuse("journal_empty", path=str(path))
    if events[0]["event"] != "launch_started":
        raise Refuse("journal_does_not_start_with_launch", first=events[0]["event"])
    return events, len(partial)


def admit(state, journal):
    started, completed = {}, {}
    for index, event in enumerate(journal):
        kind, stage = event["event"], event.get("stage")
        if kind == "stage_started":
            started[stage] = event.get("instance")
        elif kind == "stage_completed":
            if stage in completed:
                raise Refuse("stage_completed_twice", stage=stage)
            instance = started.get(stage)
            if instance is None:
                raise Refuse("stage_completed_without_start", stage=stage)
            tail = ("stages", stage, f"instance-{instance}", "fragment.json")
            if Path(str(event.get("fragment"))).parts[-4:] != tail:
                raise Refuse("fragment_path_unexpected", stage=stage,
                             fragment=event.get("fragment"))
            path = state.joinpath(*tail)
            if not path.is_file():
                raise Refuse("fragment_missing", stage=stage, path=str(path))
            if sha(path) != event.get("fragment_sha256"):
                raise Refuse("fragment_changed_since_completion", stage=stage)
            try:
                payload = json.loads(path.read_text())
            except ValueError:
                raise Refuse("fragment_unparseable", stage=stage) from None
            if not isinstance(payload, dict):
                raise Refuse("fragment_not_an_object", stage=stage)
            completed[stage] = {"fragment": payload, "path": str(path), "instance": instance,
                                "index": index}
    unadmitted = sorted(
        str(path.relative_to(state)) for path in state.glob("stages/*/instance-*/fragment.json")
        if path.parent.parent.name not in completed
        or path.parent.name != f"instance-{completed[path.parent.parent.name]['instance']}")
    return completed, unadmitted


def check_state_file(state, completed, plan_sha):
    path = state / "state.json"
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text())
    except ValueError:
        raise Refuse("state_unparseable") from None
    if saved.get("plan_sha256") != plan_sha:
        raise Refuse("state_belongs_to_another_plan")
    recorded = saved.get("completed") or {}
    for stage, entry in completed.items():
        if stage not in recorded or recorded[stage].get("fragment") != entry["fragment"]:
            raise Refuse("state_fragment_differs", stage=stage)
    return saved


def evidence_for(launch, plan, completed):
    """The launcher's own evidence document over ``completed`` (in completion order)."""
    launcher = object.__new__(launch.Launcher)
    launcher.plan = plan
    launcher.state = {"completed": {stage: {"fragment": entry["fragment"],
                                            "path": entry["path"]}
                                    for stage, entry in completed.items()}}
    try:
        document = launcher.evidence()
    except launch.Stop as stop:
        raise Refuse(stop.reason, **stop.detail) from None
    except (KeyError, TypeError, AttributeError) as exc:
        raise Refuse("fragment_malformed", error=f"{type(exc).__name__}: {exc}") from None
    adapters = {e["run_id"]: e["adapter_sha256"] for e in document["endpoints"]}
    for stage, entry in completed.items():
        f = entry["fragment"]
        if "split" in f and f.get("adapter_sha256") != adapters.get(f.get("run_id")):
            raise Refuse("assay_adapter_differs_from_endpoint", stage=stage)
    return roundtrip(document)


# ---------------------------------------------------------------- gates


def launcher_verdict(name, report):
    """Launcher.gate's verdict rule (and final_check's), applied to a recomputed report."""
    gates = (report or {}).get("gates")
    if name == "final":
        ok = (isinstance(gates, dict) and report.get("failed_at") is None
              and report.get("awaiting") == "G4"
              and all(isinstance(gates.get(g), dict) and gates[g].get("pass") is True
                      for g in GATES))
        return ok
    result = gates.get(name) if isinstance(gates, dict) else None
    if not isinstance(result, dict):
        return "gate_result_missing"
    if result.get("status") != "evaluated":
        return "gate_not_evaluated"
    if result.get("pass") is not True or report.get("failed_at") is not None:
        return "gate_failed"
    return "pass"


def evaluate(evaluator, evidence):
    try:
        return roundtrip(evaluator["evaluate"](evidence)), None
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def gate_runs(launch, plan, evaluator, state, journal, completed):
    runs = []
    for index, event in enumerate(journal):
        if event["event"] != "gate_started":
            continue
        name = event.get("gate")
        earlier = {s: c for s, c in completed.items() if c["index"] < index}
        evidence = evidence_for(launch, plan, earlier)
        written = state / "gates" / Path(str(event.get("evidence"))).name
        if not written.is_file():
            raise Refuse("gate_evidence_missing", gate=name, path=str(written))
        if sha(written) != event.get("evidence_sha256"):
            raise Refuse("gate_evidence_changed", gate=name)
        if json.loads(written.read_text()) != evidence:
            raise Refuse("gate_evidence_differs_from_fragments", gate=name)
        finished = None
        for later in journal[index + 1:]:
            if later["event"] == "gate_started" and later.get("gate") == name:
                break
            if later["event"] == ("final_check" if name == "final" else "gate_finished") and (
                    name == "final" or later.get("gate") == name):
                finished = later
                break
        report, error = evaluate(evaluator, evidence)
        recorded = None
        if finished is not None and finished.get("report"):
            path = state / "gates" / Path(str(finished["report"])).name
            if path.is_file():
                if name != "final" and sha(path) != finished.get("report_sha256"):
                    raise Refuse("gate_report_changed", gate=name)
                recorded = json.loads(path.read_text())
            elif finished.get("report_sha256"):
                raise Refuse("gate_report_missing", gate=name, path=str(path))
        if recorded is not None:
            if report is None:
                raise Refuse("recomputed_gate_differs", gate=name, error=error)
            same = (recorded == report if name == "final" else
                    (recorded.get("gates") or {}).get(name) == report["gates"].get(name)
                    and recorded.get("failed_at") == report["failed_at"])
            if not same:
                raise Refuse("recomputed_gate_differs", gate=name)
        verdict = launcher_verdict(name, report)
        if finished is not None:
            recorded_verdict = finished.get("passed") if name == "final" else finished.get(
                "verdict")
            if recorded_verdict != verdict:
                raise Refuse("recomputed_verdict_differs", gate=name,
                             recorded=recorded_verdict, recomputed=verdict)
        result = (report or {}).get("gates", {}).get(name) if name != "final" else None
        runs.append({"gate": name, "journal_line": index + 1,
                     "state": "finished" if finished else "running",
                     "verdict": verdict, "recomputed": result, "error": error,
                     "report": report if name == "final" else None,
                     "matches_launcher_report": recorded is not None})
    return runs


# ---------------------------------------------------------------- comparisons


def repeated(evaluator, study, candidate, comparator, split):
    """Paired H comparison: point difference, two-sided 95% interval, one-sided 90% LB."""
    compare = evaluator["compare_repeated_classification"]
    args = (study.rows[split], study.labels)
    n = evaluator["COMPLETIONS"][split]
    a, b = (study.predictions(e, split) for e in (candidate, comparator))
    low = compare(args[0], a, b, args[1], completions=n, lower_quantile=TWO_SIDED_LOWER)
    swapped = compare(args[0], b, a, args[1], completions=n, lower_quantile=TWO_SIDED_LOWER)
    lb90 = compare(args[0], a, b, args[1], completions=n,
                   lower_quantile=evaluator["G3"]["lb_quantile"])
    return {"split": split, "candidate": candidate["run_id"], "comparator": comparator["run_id"],
            "candidate_accuracy": low["candidate"]["mean_class_accuracy"],
            "comparator_accuracy": low["comparator"]["mean_class_accuracy"],
            "difference": low["mean_class_accuracy_difference"],
            "ci95": [low["uncertainty"]["lower_bound"],
                     -swapped["uncertainty"]["lower_bound"]],
            "lb90_one_sided": lb90["uncertainty"]["lower_bound"],
            "candidate_valid_fraction": low["candidate"]["valid_fraction"],
            "candidate_eos_fraction": candidate["splits"][split].get("eos_fraction"),
            "draws": low["uncertainty"]["draws"], "seed": low["uncertainty"]["seed"]}


def full_split(evaluator, study, candidate, comparator, split):
    result = evaluator["compare_classification"](
        study.rows[split], study.predictions(candidate, split),
        study.predictions(comparator, split), study.labels)
    return {"split": split, "candidate": candidate["run_id"], "comparator": comparator["run_id"],
            "candidate_accuracy": result["candidate"]["mean_class_accuracy"],
            "comparator_accuracy": result["comparator"]["mean_class_accuracy"],
            "difference": result["mean_class_accuracy_difference"],
            "ci95": result["uncertainty"]["stratified_image_ci95"],
            "class_cluster_ci95": result["uncertainty"]["class_cluster_ci95"],
            "relative_error_reduction": result["relative_error_reduction"],
            "draws": result["uncertainty"]["draws"], "seed": result["uncertainty"]["seed"]}


def pair(evaluator, study, candidate, comparator, split):
    if split not in study.rows or not all(study.has(e, split) for e in (candidate, comparator)):
        return None
    if split in H_SPLITS:
        return repeated(evaluator, study, candidate, comparator, split)
    return full_split(evaluator, study, candidate, comparator, split)


def comparisons(evaluator, study, selections):
    endpoints = [e for e in study.endpoints.values() if e["arm"] != "controlled400-SFT"]
    per_endpoint = []
    for endpoint in endpoints:
        for split in H_SPLITS:
            row = pair(evaluator, study, endpoint, study.w, split)
            if row is not None:
                if endpoint is study.w:
                    row.update(difference=None, ci95=None, lb90_one_sided=None)
                per_endpoint.append({"arm": endpoint["arm"], "lr": endpoint["lr"],
                                     "updates": endpoint["updates"], **row})
    rl_star = study.endpoints.get(selections.get("rl_star"))
    rl_vs = []
    if rl_star is not None:
        eg = study.find("EG-SFT")
        named = [("W", study.w),
                 ("SFT-E*", study.endpoints.get(selections.get("sft_e_star"))),
                 ("SFT-T*", study.endpoints.get(selections.get("sft_t_star"))),
                 ("EG-SFT", eg[0] if len(eg) == 1 else None)]
        for label, comparator in named:
            for split in COMPARISON_SPLITS:
                row = pair(evaluator, study, rl_star, comparator, split) if comparator else None
                rl_vs.append({"comparator_role": label, "split": split, **(row or {
                    "status": "not yet measured"})})
    return per_endpoint, rl_vs


def check_selections(study, selections):
    computed = {}
    for name, select in (
            ("rl_continuers", lambda: [c["run_id"] for c in study.select_rl_continuers()[1]]),
            ("rl_star", lambda: study.select_rl_star()[1]["run_id"]),
            ("sft_e_star", lambda: study.select_sft_e()[1]["run_id"]),
            ("sft_t_star", lambda: study.select_sft_t()[1]["run_id"])):
        if name not in selections:
            continue
        try:
            computed[name] = select()
        except (KeyError, TypeError, ValueError):
            continue
        if computed[name] != selections[name]:
            raise Refuse("selection_differs", selection=name, journal=selections[name],
                         recomputed=computed[name])
    return computed


# ---------------------------------------------------------------- stages


def stage_table(journal, completed):
    stages, order = {}, []
    terminal = None
    for event in journal:
        kind, stage = event["event"], event.get("stage")
        if kind in TERMINAL_EVENTS:
            terminal = event
        if kind in ("launch_interrupted", "study_stopped"):
            for row in stages.values():
                if row["status"] == "running":
                    row["status"] = "interrupted" if kind == "launch_interrupted" else "stopped"
        if stage is None:
            continue
        if stage not in stages and kind == "stage_started":
            order.append(stage)
            stages[stage] = {"stage": stage, "arm": event.get("arm"), "instances": 0,
                             "void_instances": 0, "status": None}
        row = stages.get(stage)
        if row is None:
            continue
        if kind == "stage_started":
            row.update(instances=event.get("instance"), status="running",
                       cap_seconds=event.get("cap_seconds"), started=event.get("time"),
                       charged_before_start_seconds=(event.get("inherited_seconds") or 0)
                       + (event.get("void_seconds") or 0))
        elif kind == "stage_finished":
            row.update(exit_code=event.get("exit_code"), stop=event.get("reason"),
                       elapsed_seconds=event.get("elapsed_seconds"),
                       status="finished, not admitted")
        elif kind == "stage_completed":
            row["status"] = "completed"
        elif kind == "arm_instance_void":
            row["void_instances"] += 1
    for row in stages.values():
        if row["status"] == "completed" and row["stage"] not in completed:
            raise Refuse("stage_completed_but_not_admitted", stage=row["stage"])
    return [stages[s] for s in order], terminal


# ---------------------------------------------------------------- report


def build(state, plan_path):
    state = Path(state)
    launch = load_launch()
    plan, plan_sha, evaluator = load_plan(plan_path, launch)
    journal, partial_bytes = read_journal(state)
    for event in journal:
        if event["event"] in ("launch_started", "launch_resumed") and event.get(
                "plan_sha256") != plan_sha:
            raise Refuse("journal_plan_differs", journal=event.get("plan_sha256"), plan=plan_sha)
    base = pinned_json(launch, plan.get("evidence_base"), "evidence_base_changed")
    goal = target(launch, plan, evaluator, base, journal)
    completed, unadmitted = admit(state, journal)
    saved = check_state_file(state, completed, plan_sha)
    stages, terminal = stage_table(journal, completed)
    runs = gate_runs(launch, plan, evaluator, state, journal, completed)
    evidence = evidence_for(launch, plan, completed)
    current, current_error = evaluate(evaluator, evidence)
    selections = {e["name"]: e["value"] for e in journal if e["event"] == "selection"}
    study = evaluator["Study"](evidence)
    recomputed_selections = check_selections(study, selections)
    per_endpoint, rl_vs = comparisons(evaluator, study, selections)
    latest = {}
    for run in runs:
        latest[run["gate"]] = run
    gates = []
    for name in GATES:
        run = latest.get(name)
        gates.append({"gate": name, **(run or {"state": "not started"})})
    later = [{"gate": name, **((current or {}).get("gates", {}).get(name) or {
        "status": "not evaluated"})} for name in LATER_GATES]
    e0 = (completed.get("e0") or {}).get("fragment")
    rl_val = next((row for row in rl_vs if row.get("split") == "val"
                   and row.get("comparator_role") == "W" and "candidate_accuracy" in row), None)
    goal["rl_star_val_accuracy"] = rl_val["candidate_accuracy"] if rl_val else None
    goal["rl_star_val_meets_target"] = (
        rl_val["candidate_accuracy"] >= goal["rl_val_target_lower_bound"] - FLOAT_SLACK
        if rl_val else None)
    first = journal[0]
    return {
        "schema": "freshpool-report-v1",
        "generated": time.time(),
        "state_directory": str(state),
        "plan_sha256": plan_sha,
        "study_status": (saved or {}).get("status"),
        "stop_reason": (saved or {}).get("stop_reason"),
        "next_action": (saved or {}).get("next_action"),
        "terminal_event": terminal,
        "journal_events": len(journal),
        "journal_partial_tail_bytes": partial_bytes,
        "journal_last_event": {k: journal[-1].get(k) for k in ("event", "stage", "gate",
                                                               "time")},
        "g0": first.get("g0"),
        "target": goal,
        "e0": None if e0 is None else {
            "checks": e0.get("checks"),
            "sampler_teacher_max_abs_logprob_difference": e0.get(
                "sampler_teacher_max_abs_logprob_difference"),
            "all_checks_true": isinstance(e0.get("checks"), dict) and bool(e0["checks"])
            and all(v is True for v in e0["checks"].values()),
            "logprob_within_0_001": isinstance(e0.get(
                "sampler_teacher_max_abs_logprob_difference"), (int, float))
            and e0["sampler_teacher_max_abs_logprob_difference"] <= E0_MAX_LOGPROB_DIFFERENCE},
        "stages": stages,
        "unadmitted_fragments": unadmitted,
        "gates": gates,
        "gate_runs": [{k: v for k, v in run.items() if k != "report"} for run in runs],
        "later_gates": later,
        "selections": selections,
        "selections_recomputed": recomputed_selections,
        "current_evaluation": None if current is None else {
            k: current[k] for k in ("failed_at", "awaiting", "evidence_after_stop",
                                    "substantial_val_and_replication", "final_claim")},
        "current_evaluation_error": current_error,
        "h_accuracy": per_endpoint,
        "rl_star_comparisons": rl_vs,
        "uncertainty": "Protocol evaluation.uncertainty via vision_rl.classification_comparison: "
                       "paired, 20,000 draws, seed 2718, images resampled within class. H: 95% "
                       "interval = [0.025 bound, -(0.025 bound of the swapped pair)], plus the "
                       "one-sided 90% bound G3a uses. VAL: stratified_image_ci95.",
        "scope": "H-select/H-gate are native-TRAIN development evidence, not VAL; TEST is "
                 "never scored by this report.",
    }


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(v, (int, float)) for v in value):
        return f"[{value[0]:.{digits}f}, {value[1]:.{digits}f}]"
    return str(value)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return lines


def markdown(report):
    t = report["target"]
    out = [f"# Freshpool study report: {report['state_directory']}", "",
           f"- Status: **{report['study_status']}**; stop reason: {fmt(report['stop_reason'])}; "
           f"next action: {fmt(report['next_action'])}",
           f"- Journal: {report['journal_events']} events, last "
           f"`{report['journal_last_event']['event']}`"
           + (f" ({report['journal_partial_tail_bytes']} bytes of a line still being written, "
              "ignored)" if report["journal_partial_tail_bytes"] else ""),
           f"- Plan SHA256 `{report['plan_sha256']}`",
           f"- Target (branch receipt B2): a* = {fmt(t['a_star'])}, D_req = {fmt(t['d_req'])} "
           f"({t['mode']} mode, G3 margin {fmt(t['g3_margin'])}), RL* VAL needs at least "
           f"**{fmt(t['rl_val_target_lower_bound'])}** (lower bound). RL* VAL: "
           f"{fmt(t['rl_star_val_accuracy'])}"
           + ("" if t["rl_star_val_meets_target"] is None else
              f", meets the lower bound: {fmt(t['rl_star_val_meets_target'])}"),
           f"- {t['note']}", f"- {report['scope']}", "", "## Stages", ""]
    out += table(["stage", "status", "instance", "void", "elapsed s", "cap s", "exit", "stop"],
                 [[s["stage"], s["status"], s["instances"], s["void_instances"],
                   s.get("elapsed_seconds"), s.get("cap_seconds"), s.get("exit_code"),
                   s.get("stop")] for s in report["stages"]])
    if report["unadmitted_fragments"]:
        out += ["", "Fragments on disk the journal never admitted (not used): "
                + ", ".join(f"`{p}`" for p in report["unadmitted_fragments"])]
    if report["e0"]:
        e0 = report["e0"]
        out += ["", "## E0", "",
                f"All checks true: {fmt(e0['all_checks_true'])}; sampler-teacher max |dlogp| "
                f"{fmt(e0['sampler_teacher_max_abs_logprob_difference'], 8)} (<= 0.001: "
                f"{fmt(e0['logprob_within_0_001'])})"]
    out += ["", "## Gates (recomputed from the fragments by the evaluator)", ""]
    for gate in report["gates"]:
        result = gate.get("recomputed") or {}
        out.append(f"### {gate['gate']}: {gate.get('verdict', gate['state'])}")
        out.append("")
        out.append(f"State: {gate['state']}; evaluator status: {fmt(result.get('status'))}; "
                   f"matches the launcher's report: {fmt(gate.get('matches_launcher_report'))}"
                   + (f"; evaluator error: {gate['error']}" if gate.get("error") else ""))
        checks = result.get("checks") or {}
        values = result.get("values") or {}
        if checks:
            out += [""] + table(["check", "pass"], sorted(checks.items()))
        if values:
            out += [""] + table(["value", ""], sorted(values.items()))
        extras = {k: result[k] for k in ("best", "continue_to_600", "rl_star", "sft_e_star",
                                         "sft_t_star", "sft_t_lrs", "mode", "margin")
                  if k in result}
        if extras:
            out += ["", "; ".join(f"{k}: {fmt(v)}" for k, v in extras.items())]
        out.append("")
    out += ["Later gates (native-VAL package): " + ", ".join(
        f"{g['gate']} {g.get('status')}" for g in report["later_gates"])]
    if report["current_evaluation_error"]:
        out += ["", f"Current-evidence evaluation not computable yet (a stage family is "
                f"incomplete): {report['current_evaluation_error']}"]
    elif report["current_evaluation"]:
        c = report["current_evaluation"]
        out += ["", f"Current evidence: failed at {fmt(c['failed_at'])}, awaiting "
                f"{fmt(c['awaiting'])}"]
    out += ["", "## Selections", ""]
    out += table(["selection", "journal", "recomputed"],
                 [[name, report["selections"].get(name),
                   report["selections_recomputed"].get(name)] for name in SELECTIONS])
    out += ["", "## H-select / H-gate mean-class accuracy vs W", "",
            report["uncertainty"], ""]
    out += table(["endpoint", "arm", "lr", "updates", "split", "accuracy", "W", "delta",
                  "95% CI", "90% LB", "valid", "EOS"],
                 [[r["candidate"], r["arm"], r["lr"], r["updates"], r["split"],
                   r["candidate_accuracy"], r["comparator_accuracy"], r["difference"],
                   r["ci95"], r["lb90_one_sided"], r["candidate_valid_fraction"],
                   r["candidate_eos_fraction"]] for r in report["h_accuracy"]])
    out += ["", "## RL* vs W, SFT-E*, SFT-T*, EG-SFT", ""]
    if not report["rl_star_comparisons"]:
        out.append("RL* not selected yet.")
    else:
        out += table(["comparator", "run", "split", "RL*", "comparator acc", "delta",
                      "95% CI", "90% LB", "class-cluster 95% CI", "RER"],
                     [[r["comparator_role"], r.get("comparator"), r["split"],
                       r.get("candidate_accuracy"), r.get("comparator_accuracy"),
                       r.get("difference"), r.get("ci95"), r.get("lb90_one_sided"),
                       r.get("class_cluster_ci95"), r.get("relative_error_reduction")]
                      if "difference" in r else
                      [r["comparator_role"], "-", r["split"], "not yet measured", None, None,
                       None, None, None, None] for r in report["rl_star_comparisons"]])
    return "\n".join(out) + "\n"


def emit(report, json_path, markdown_path):
    text = markdown(report)
    if json_path:
        Path(json_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if markdown_path:
        Path(markdown_path).write_text(text)
    else:
        print(text)


def outside(state, path):
    if path is None:
        return True
    resolved, root = Path(path).resolve(), Path(state).resolve()
    return resolved != root and root not in resolved.parents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--watch", type=float, help="poll seconds; exit at a terminal event")
    args = parser.parse_args(argv)
    try:
        state = args.state_directory
        if state is None:
            plan = json.loads(Path(args.plan).read_text())
            state = load_launch().resolve(plan["state_directory"])
        if not all(outside(state, p) for p in (args.json, args.markdown)):
            raise Refuse("output_inside_state_directory")
        seen = object()  # never equal to a size, so the first pass always reports
        while True:
            size = (Path(state) / "journal.jsonl").stat().st_size if (
                Path(state) / "journal.jsonl").is_file() else None
            if size != seen:
                seen = size
                report = build(state, args.plan)
                emit(report, args.json, args.markdown)
                if args.watch:
                    print(json.dumps({"reported": report["journal_last_event"],
                                      "gates": {g["gate"]: g.get("verdict", g["state"])
                                                for g in report["gates"]}}), flush=True)
                if not args.watch or report["terminal_event"] is not None:
                    return 0
            time.sleep(args.watch)
    except Refuse as refusal:
        print(json.dumps({"status": "refused", "reason": refusal.reason, **refusal.detail},
                         default=str))
        return 2


if __name__ == "__main__":
    sys.exit(main())
