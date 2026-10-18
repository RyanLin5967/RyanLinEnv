"""Report a freshpool launch state directory: every stage, every gate recomputed, every CI.

  python scripts/freshpool_report.py --state-directory runs/aircraft-freshpool-v1 \\
      [--plan scripts/freshpool_launch/production/plan.json] \\
      [--json report.json] [--markdown report.md] [--watch 60]

Read-only over the state directory. It writes only the --json / --markdown paths, and only
to an allowlist: outside this repository, or under its reports/ directory; never inside the
state directory and never onto a file the plan names (``output_not_allowed``). Without
--markdown it prints the markdown. ``--watch N`` re-checks every N seconds, re-reports each
time the journal grows, and exits at study_stopped / study_development_complete, or as soon
as no live launcher holds the state directory's lock (interrupted, paused, or killed); bound
it with timeout all the same.

What it does, in order, and the refusal (exit 2, one JSON line on stdout) for each failure:
  * code before it runs: the evaluator, launch.py and vision_rl/classification_comparison.py
    must each be pinned by the plan (``report_code_unpinned``) and their bytes must hash to
    the pin BEFORE either is executed (``report_code_differs_from_pin``); the comparison
    module the evaluator actually imported must hash to the same pin;
  * the plan's own SHA256 must equal every launch_started/launch_resumed plan_sha256 in the
    journal (``journal_plan_differs``); the evidence base and branch receipt must equal their
    pinned SHA256s (``evidence_base_changed``, ``branch_receipt_changed``); the receipt's
    D_req and target must equal the evaluator's d_req(a*) and every journalled d_req;
  * the journal must exist, be non-empty, start with launch_started and parse line by line
    (``journal_missing``, ``journal_empty``, ``journal_line_unparseable``). A trailing line
    without its newline is a write in progress: it is ignored and reported, not parsed;
  * every ``stage_completed`` admits exactly one fragment, at stages/<stage>/instance-<k>/
    fragment.json for the instance that stage last started, whose bytes still hash to the
    journalled SHA256 (``fragment_missing``, ``fragment_changed_since_completion``,
    ``stage_completed_twice``, ``stage_completed_without_start``, ...). An admitted assay must
    be on an allowlisted split, H-select or H-gate (``assay_split_not_allowed``): VAL and
    TEST evidence cannot enter this report. state.json, when present, must hold the same
    payload for every admitted stage;
  * the evaluator's evidence is rebuilt from the admitted fragments by the launcher's OWN
    ``Launcher.evidence`` (imported, not rewritten), once per ``gate_started`` from the
    fragments completed before it. Each rebuilt document must equal the evidence file the
    launcher wrote (``gate_evidence_differs_from_fragments``); each gate is re-decided by
    the evaluator's own ``evaluate`` and must agree with the launcher's recorded report and
    verdict (``recomputed_gate_differs``, ``recomputed_verdict_differs``). A gate whose
    evaluator subprocess exited non-zero or timed out has the launcher's verdict
    ``gate_result_missing``, whatever the in-process recomputation says;
  * every journal ``selection`` (continuers, RL*, SFT-E*, SFT-T LRs, SFT-T*) must equal the
    evaluator's own selection function over the evidence admitted before that event
    (``selection_differs``); one the evaluator cannot reproduce from those fragments is
    ``selection_unrecomputable``.
Any other malformed input is ``report_input_malformed`` (exit 2), never a traceback.

Then it reports: per-stage status; E0 as launch.Launcher.e0_validator judges it; each gate
with checks and values; H-select / H-gate mean-class accuracy (every completion, greedy
included, as the evaluator scores it) of every endpoint with its paired difference vs W;
RL* vs W, SFT-E*, SFT-T* and EG-SFT on H-select and H-gate; and the branch receipt's VAL
target a* + D_req, which VAL (G4, the native-VAL package) must reach. No VAL number is
produced here: the launcher assays H splits only.

Uncertainty is the protocol's (docs/aircraft-rl-protocol-freshpool-v1.json
evaluation.uncertainty: paired, 20,000 draws, seed 2718, images resampled within each class),
computed only by ``vision_rl.classification_comparison`` at its defaults, which equal those
values. On H splits that module returns lower bounds only: the 95% interval's lower end is
its 0.025 bound, its upper end is minus the 0.025 bound of the swapped comparison (same
draws, negated), and the one-sided 90% bound is the 0.10 quantile G3a uses.

TEST is never scored here: an admitted fragment on any split but H-select / H-gate is a
refusal, so evaluate() never sees VAL or TEST evidence. H results are native-TRAIN
development evidence; they are not comparable to the VAL target and are not presented so.

Blind spots, stated:
  * it trusts what the fragments say beyond the launcher's own checks (it cannot see whether
    predictions came from the stated adapter);
  * it reads a live journal without a lock, so a report is a snapshot of the complete lines
    at read time; fragments on disk that the journal never admitted (void, in flight, or
    refused by the launcher) are listed and never used;
  * the current-state evaluation is informational when a stage family is half done (the
    evaluator refuses a partial LR family; that refusal is shown, not raised);
  * pins: it checks the three files it executes, not everything launch.verify_pins checks
    (the interpreter link and the import-tree walk describe the launcher's runtime, not this
    process's). Package files imported along the way (vision_rl/__init__.py) are not hashed;
  * verdict rules: launcher_verdict restates Launcher.gate / final_check's rule (including
    their exit-code term) because those methods also run the evaluator subprocess. The
    final_check event records no exit code: a missing or unparseable final report stands in
    for a non-zero one;
  * a launcher window it cannot represent: Launcher.run saves state.json with a stage
    completed BEFORE appending stage_completed. A signal between the two leaves a state
    whose resume returns that stage from state.json, so the journal never admits it; every
    later gate's evidence then holds a fragment the journal does not, and the report refuses
    (``gate_evidence_differs_from_fragments``, naming the state-only stages). It fails
    closed; launch.py is pinned by the live plan, so the order is not changed here;
  * --watch decides "launcher gone" from the lock file (pid plus start time). A launcher
    resumed after the watcher exited needs a new watch.
"""

import argparse
import hashlib
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts" / "freshpool_launch" / "launch.py"
COMPARISON = ROOT / "src" / "vision_rl" / "classification_comparison.py"
DEFAULT_PLAN = ROOT / "scripts" / "freshpool_launch" / "production" / "plan.json"
OUTPUT_ROOTS = (ROOT / "reports",)  # the only places inside the repository it may write
H_SPLITS = ("h_select", "h_gate")
ALLOWED_SPLITS = H_SPLITS  # allowlist: VAL and TEST never enter this report
GATES = ("G1", "G2", "G3a", "G3b")
LATER_GATES = ("G4", "G5", "G6")
TERMINAL_EVENTS = ("study_stopped", "study_development_complete")
TWO_SIDED_LOWER = 0.025  # lower end of the protocol's two-sided 95% interval
FLOAT_SLACK = 1e-12
SELECTORS = {
    "rl_continuers": lambda s: [c["run_id"] for c in s.select_rl_continuers()[1]],
    "rl_star": lambda s: s.select_rl_star()[1]["run_id"],
    "sft_e_star": lambda s: s.select_sft_e()[1]["run_id"],
    "sft_t_lrs": lambda s: s.select_sft_e()[2],
    "sft_t_star": lambda s: s.select_sft_t()[1]["run_id"],
}
SELECTIONS = tuple(SELECTORS)


class Refuse(Exception):
    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roundtrip(value):
    """What the launcher's files hold for ``value``: JSON, NaN refused, tuples as lists."""
    return json.loads(json.dumps(value, allow_nan=False))


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve(path):
    """launch.resolve, restated so nothing from launch.py runs before its hash is checked."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------- plan, pins and receipts


def read_plan(plan_path):
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        raise Refuse("plan_missing", plan=str(plan_path))
    try:
        plan = json.loads(plan_path.read_text())
    except ValueError:
        raise Refuse("plan_unparseable", plan=str(plan_path)) from None
    if not isinstance(plan, dict):
        raise Refuse("plan_unparseable", plan=str(plan_path))
    if not isinstance(plan.get("evaluator"), str):
        raise Refuse("plan_evaluator_missing", plan=str(plan_path))
    if not isinstance(plan.get("file_sha256"), dict):
        raise Refuse("plan_pins_missing", plan=str(plan_path))
    return plan


def load_launch():
    spec = importlib.util.spec_from_file_location("freshpool_launch_for_report", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_code(plan_path):
    """Hash, THEN execute: launch.py, the evaluator and the comparison module it imports."""
    plan = read_plan(plan_path)
    pins = {resolve(name).resolve(): value for name, value in plan["file_sha256"].items()}
    evaluator_path = resolve(plan["evaluator"]).resolve()
    for path in (evaluator_path, LAUNCH.resolve(), COMPARISON.resolve()):
        if path not in pins:
            raise Refuse("report_code_unpinned", file=str(path))
        if not path.is_file() or sha(path) != pins[path]:
            raise Refuse("report_code_differs_from_pin", file=str(path))
    launch = load_launch()
    evaluator = runpy.run_path(str(evaluator_path))
    imported = Path(sys.modules[evaluator["compare_classification"].__module__].__file__)
    if sha(imported) != pins[COMPARISON.resolve()]:
        # The evaluator imported vision_rl from somewhere else (an installed copy, another
        # checkout) whose bytes are not the pinned module's.
        raise Refuse("report_code_differs_from_pin", file=str(imported))
    return launch, plan, sha(plan_path), evaluator


def pinned_json(entry, reason):
    entry = entry if isinstance(entry, dict) else {}
    path = resolve(entry.get("path", "/nonexistent"))
    if not path.is_file() or sha(path) != entry.get("sha256"):
        raise Refuse(reason, path=str(path))
    return json.loads(path.read_text())


def target(plan, evaluator, base, journal):
    receipt = pinned_json(((plan.get("g0") or {}).get("receipts") or {}).get("branch"),
                          "branch_receipt_changed")
    a_star = receipt.get("a_star")
    if not is_number(a_star) or not is_number(receipt.get("d_req")):
        raise Refuse("branch_receipt_malformed", a_star=a_star, d_req=receipt.get("d_req"))
    branch = evaluator["d_req"](a_star)
    if (abs(receipt["d_req"] - branch["d_req"]) > FLOAT_SLACK
            or receipt.get("mode") != branch["mode"] or base.get("a_star") != a_star):
        raise Refuse("branch_receipt_disagrees_with_evaluator", receipt_d_req=receipt["d_req"],
                     evaluator_d_req=branch["d_req"])
    value = a_star + branch["d_req"]
    published = (((receipt.get("branches") or {}).get("B2") or {})
                 .get("rl_val_needed_lower_bound"))
    if published is not None and (not is_number(published)
                                  or abs(published - value) > FLOAT_SLACK):
        raise Refuse("branch_receipt_target_inconsistent", published=published, computed=value)
    for event in journal:
        if event["event"] in ("launch_started", "launch_resumed") and "d_req" in event:
            if roundtrip(event["d_req"]) != roundtrip(branch):
                raise Refuse("journal_d_req_differs", journal=event["d_req"])
    return {"a_star": a_star, "d_req": branch["d_req"], "mode": branch["mode"],
            "g3_margin": branch["g3_margin"], "rl_val_target_lower_bound": value,
            "published_target": published,
            "note": "Lower bound: G4 also needs +0.05 over the strongest SFT on VAL, whose "
                    "fresh-pool endpoints can raise a*; RL* must also reach 0.80. VAL is "
                    "measured by the native-VAL package (G4), not by this report."}


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
            instance = event.get("instance")
            if not isinstance(instance, int) or isinstance(instance, bool) or instance < 1:
                raise Refuse("stage_started_without_instance", stage=stage, line=index + 1)
            started[stage] = instance
        elif kind == "stage_completed":
            if stage in completed:
                raise Refuse("stage_completed_twice", stage=stage)
            if stage not in started:
                raise Refuse("stage_completed_without_start", stage=stage)
            instance = started[stage]
            tail = ("stages", str(stage), f"instance-{instance}", "fragment.json")
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
            if "split" in payload and payload["split"] not in ALLOWED_SPLITS:
                raise Refuse("assay_split_not_allowed", stage=stage, split=payload["split"],
                             allowed=list(ALLOWED_SPLITS))
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
        return None, []
    try:
        saved = json.loads(path.read_text())
    except ValueError:
        raise Refuse("state_unparseable") from None
    if not isinstance(saved, dict):
        raise Refuse("state_unparseable")
    if saved.get("plan_sha256") != plan_sha:
        raise Refuse("state_belongs_to_another_plan")
    recorded = saved.get("completed") or {}
    if not isinstance(recorded, dict):
        raise Refuse("state_unparseable")
    for stage, entry in completed.items():
        held = recorded.get(stage)
        if not isinstance(held, dict) or held.get("fragment") != entry["fragment"]:
            raise Refuse("state_fragment_differs", stage=stage)
    # Completed in state.json, never journalled: in flight between save and append, or the
    # launcher window named in the docstring.
    return saved, sorted(stage for stage in recorded if stage not in completed)


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


def earlier(completed, index):
    return {s: c for s, c in completed.items() if c["index"] < index}


# ---------------------------------------------------------------- gates


def launcher_verdict(name, report, code):
    """Launcher.gate's verdict rule (and final_check's), applied to a recomputed report and
    the evaluator's exit code (None: timed out)."""
    gates = (report or {}).get("gates")
    if name == "final":
        return (code == 0 and isinstance(gates, dict) and report.get("failed_at") is None
                and report.get("awaiting") == "G4"
                and all(isinstance(gates.get(g), dict) and gates[g].get("pass") is True
                        for g in GATES))
    result = gates.get(name) if isinstance(gates, dict) else None
    if code != 0 or not isinstance(result, dict):
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


def gate_runs(launch, plan, evaluator, state, journal, completed, state_only):
    runs = []
    for index, event in enumerate(journal):
        if event["event"] != "gate_started":
            continue
        name = event.get("gate")
        evidence = evidence_for(launch, plan, earlier(completed, index))
        written = state / "gates" / Path(str(event.get("evidence"))).name
        if not written.is_file():
            raise Refuse("gate_evidence_missing", gate=name, path=str(written))
        if sha(written) != event.get("evidence_sha256"):
            raise Refuse("gate_evidence_changed", gate=name)
        if json.loads(written.read_text()) != evidence:
            raise Refuse("gate_evidence_differs_from_fragments", gate=name,
                         completed_in_state_json_only=state_only)
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
                try:
                    recorded = json.loads(path.read_text())
                except ValueError:
                    recorded = None  # the launcher read it as None too (read_json)
                if not isinstance(recorded, dict):
                    recorded = None
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
        if finished is None:
            code = 0  # still running: the verdict the evaluator's report would give
        elif name == "final":
            code = 0 if recorded is not None else None  # final_check records no exit code
        else:
            code = finished.get("exit_code")
        verdict = launcher_verdict(name, report, code)
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
                     "evaluator_exit_code": None if finished is None else code,
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


def pair(evaluator, study, candidate, comparator, split):
    if split not in study.rows or not all(study.has(e, split) for e in (candidate, comparator)):
        return None
    return repeated(evaluator, study, candidate, comparator, split)


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
            for split in H_SPLITS:
                row = pair(evaluator, study, rl_star, comparator, split) if comparator else None
                rl_vs.append({"comparator_role": label, "split": split, **(row or {
                    "status": "not yet measured"})})
    return per_endpoint, rl_vs


def check_selections(launch, plan, evaluator, journal, completed):
    """Each journal selection, recomputed over the evidence admitted before it."""
    computed = {}
    for index, event in enumerate(journal):
        if event["event"] != "selection":
            continue
        name = event.get("name")
        if name not in SELECTORS:
            raise Refuse("selection_unknown", selection=name)
        evidence = evidence_for(launch, plan, earlier(completed, index))
        try:
            value = roundtrip(SELECTORS[name](evaluator["Study"](evidence)))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise Refuse("selection_unrecomputable", selection=name, journal=event.get("value"),
                         error=f"{type(exc).__name__}: {exc}") from None
        if value != roundtrip(event.get("value")):
            raise Refuse("selection_differs", selection=name, journal=event.get("value"),
                         recomputed=value)
        computed[name] = value
    return computed


# ---------------------------------------------------------------- launcher liveness


def launcher_running(launch, state):
    """True while a live process holds the state directory's lock (pid AND start time)."""
    lock = Path(state) / "lock"
    if not lock.is_file():
        return False
    try:
        pid, started = launch.read_lock(lock)
    except (OSError, ValueError):
        return True  # being written, or unreadable: not evidence that the launcher is gone
    return launch.lock_holder_alive(pid, started)


# ---------------------------------------------------------------- stages


def stage_table(journal, completed, running):
    stages, order = {}, []
    terminal = None
    for event in journal:
        kind, stage = event["event"], event.get("stage")
        if kind in TERMINAL_EVENTS:
            terminal = event
        if kind in ("launch_interrupted", "launch_paused", "study_stopped"):
            for row in stages.values():
                if row["status"] == "running":
                    row["status"] = "stopped" if kind == "study_stopped" else "interrupted"
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
        if row["status"] == "running" and not running:
            row["status"] = "orphaned: no live launcher holds the lock"
    return [stages[s] for s in order], terminal


# ---------------------------------------------------------------- report


def e0_summary(launch, completed):
    e0 = (completed.get("e0") or {}).get("fragment")
    if e0 is None:
        return None
    try:
        problem = launch.Launcher.e0_validator(e0, None)
    except Exception as exc:  # a malformed fragment is a failure, never a pass
        problem = f"e0_unvalidatable: {type(exc).__name__}: {exc}"
    return {"checks": e0.get("checks"),
            "sampler_teacher_max_abs_logprob_difference": e0.get(
                "sampler_teacher_max_abs_logprob_difference"),
            "validator": "launch.Launcher.e0_validator", "problem": problem,
            "passes": problem is None}


def build(state, plan_path):
    state = Path(state)
    launch, plan, plan_sha, evaluator = load_code(plan_path)
    journal, partial_bytes = read_journal(state)
    for event in journal:
        if event["event"] in ("launch_started", "launch_resumed") and event.get(
                "plan_sha256") != plan_sha:
            raise Refuse("journal_plan_differs", journal=event.get("plan_sha256"), plan=plan_sha)
    base = pinned_json(plan.get("evidence_base"), "evidence_base_changed")
    goal = target(plan, evaluator, base, journal)
    completed, unadmitted = admit(state, journal)
    saved, state_only = check_state_file(state, completed, plan_sha)
    running = launcher_running(launch, state)
    stages, terminal = stage_table(journal, completed, running)
    runs = gate_runs(launch, plan, evaluator, state, journal, completed, state_only)
    evidence = evidence_for(launch, plan, completed)
    current, current_error = evaluate(evaluator, evidence)
    recomputed_selections = check_selections(launch, plan, evaluator, journal, completed)
    selections = {e["name"]: e["value"] for e in journal if e["event"] == "selection"}
    study = evaluator["Study"](evidence)
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
    first = journal[0]
    return {
        "schema": "freshpool-report-v2",
        "generated": time.time(),
        "state_directory": str(state),
        "plan_sha256": plan_sha,
        "study_status": (saved or {}).get("status"),
        "stop_reason": (saved or {}).get("stop_reason"),
        "next_action": (saved or {}).get("next_action"),
        "terminal_event": terminal,
        "launcher_running": running,
        "journal_events": len(journal),
        "journal_partial_tail_bytes": partial_bytes,
        "journal_last_event": {k: journal[-1].get(k) for k in ("event", "stage", "gate",
                                                               "time")},
        "completed_in_state_json_only": state_only,
        "g0": first.get("g0"),
        "target": goal,
        "e0": e0_summary(launch, completed),
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
                       "one-sided 90% bound G3a uses.",
        "scope": "H-select/H-gate are native-TRAIN development evidence, not VAL; only "
                 "H-split assays are admitted, so neither VAL nor TEST is scored here.",
    }


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            is_number(v) for v in value):
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
           f"next action: {fmt(report['next_action'])}; live launcher: "
           f"{fmt(report['launcher_running'])}",
           f"- Journal: {report['journal_events']} events, last "
           f"`{report['journal_last_event']['event']}`"
           + (f" ({report['journal_partial_tail_bytes']} bytes of a line still being written, "
              "ignored)" if report["journal_partial_tail_bytes"] else ""),
           f"- Plan SHA256 `{report['plan_sha256']}`",
           f"- Target (branch receipt B2): a* = {fmt(t['a_star'])}, D_req = {fmt(t['d_req'])} "
           f"({t['mode']} mode, G3 margin {fmt(t['g3_margin'])}), RL* VAL needs at least "
           f"**{fmt(t['rl_val_target_lower_bound'])}** (lower bound).",
           f"- {t['note']}", f"- {report['scope']}", "", "## Stages", ""]
    out += table(["stage", "status", "instance", "void", "elapsed s", "cap s", "exit", "stop"],
                 [[s["stage"], s["status"], s["instances"], s["void_instances"],
                   s.get("elapsed_seconds"), s.get("cap_seconds"), s.get("exit_code"),
                   s.get("stop")] for s in report["stages"]])
    if report["completed_in_state_json_only"]:
        out += ["", "Completed in state.json but not (yet) journalled: "
                + ", ".join(f"`{s}`" for s in report["completed_in_state_json_only"])]
    if report["unadmitted_fragments"]:
        out += ["", "Fragments on disk the journal never admitted (not used): "
                + ", ".join(f"`{p}`" for p in report["unadmitted_fragments"])]
    if report["e0"]:
        e0 = report["e0"]
        out += ["", "## E0", "",
                f"{e0['validator']}: {'pass' if e0['passes'] else e0['problem']}; "
                f"sampler-teacher max |dlogp| "
                f"{fmt(e0['sampler_teacher_max_abs_logprob_difference'], 8)}"]
    out += ["", "## Gates (recomputed from the fragments by the evaluator)", ""]
    for gate in report["gates"]:
        result = gate.get("recomputed") or {}
        out.append(f"### {gate['gate']}: {gate.get('verdict', gate['state'])}")
        out.append("")
        out.append(f"State: {gate['state']}; evaluator status: {fmt(result.get('status'))}; "
                   f"matches the launcher's report: {fmt(gate.get('matches_launcher_report'))}"
                   + (f"; launcher's evaluator exit code: {fmt(gate['evaluator_exit_code'])}"
                      if gate.get("state") == "finished" and gate.get("evaluator_exit_code")
                      != 0 else "")
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
    out += ["", "## RL* vs W, SFT-E*, SFT-T*, EG-SFT (H splits)", ""]
    if not report["rl_star_comparisons"]:
        out.append("RL* not selected yet.")
    else:
        out += table(["comparator", "run", "split", "RL*", "comparator acc", "delta",
                      "95% CI", "90% LB"],
                     [[r["comparator_role"], r.get("comparator"), r["split"],
                       r.get("candidate_accuracy"), r.get("comparator_accuracy"),
                       r.get("difference"), r.get("ci95"), r.get("lb90_one_sided")]
                      if "difference" in r else
                      [r["comparator_role"], "-", r["split"], "not yet measured", None, None,
                       None, None] for r in report["rl_star_comparisons"]])
    return "\n".join(out) + "\n"


def emit(report, json_path, markdown_path):
    text = markdown(report)
    if json_path:
        Path(json_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if markdown_path:
        Path(markdown_path).write_text(text)
    else:
        print(text)


def named_files(plan, plan_path):
    """Every file the plan names or pins, and the plan itself: never an output."""
    names = [plan_path, plan.get("evaluator"), *(plan.get("file_sha256") or {})]
    for entry in (plan.get("evidence_base"),
                  *(((plan.get("g0") or {}).get("receipts") or {}).values())):
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            names.append(entry["path"])
    return {resolve(name).resolve() for name in names if isinstance(name, (str, Path))}


def output_refusal(path, state, plan, plan_path):
    """Allowlist: outside the repository, or under reports/; never in the state directory
    and never onto a file the plan names. Returns the reason, or None when allowed."""
    resolved = Path(path).resolve()
    state_root = Path(state).resolve()
    if resolved == state_root or state_root in resolved.parents:
        return "output_inside_state_directory"
    if resolved in named_files(plan, plan_path):
        return "output_is_a_plan_file"
    repository = ROOT.resolve()
    if repository in resolved.parents and not any(
            root.resolve() in resolved.parents for root in OUTPUT_ROOTS):
        return "output_inside_repository_outside_reports"
    return None


def run(args):
    plan = read_plan(args.plan)
    state = Path(args.state_directory or resolve(plan.get("state_directory") or "/nonexistent"))
    for path in (args.json, args.markdown):
        reason = path is not None and output_refusal(path, state, plan, args.plan)
        if reason:
            raise Refuse("output_not_allowed", why=reason, path=str(path))
    seen = object()  # never equal to a size, so the first pass always reports
    while True:
        journal = state / "journal.jsonl"
        size = journal.stat().st_size if journal.is_file() else None
        if size != seen:
            seen = size
            report = build(state, args.plan)
            emit(report, args.json, args.markdown)
            if args.watch:
                print(json.dumps({"reported": report["journal_last_event"],
                                  "launcher_running": report["launcher_running"],
                                  "gates": {g["gate"]: g.get("verdict", g["state"])
                                            for g in report["gates"]}}), flush=True)
            if (not args.watch or report["terminal_event"] is not None
                    or not report["launcher_running"]):
                return 0
        time.sleep(args.watch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--watch", type=float,
                        help="poll seconds; exit at a terminal event or when no live "
                             "launcher holds the lock")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Refuse as refusal:
        print(json.dumps({"status": "refused", "reason": refusal.reason, **refusal.detail},
                         default=str))
        return 2
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OSError) as exc:
        # Input this report has no named refusal for: still exit 2 and one JSON line.
        print(json.dumps({"status": "refused", "reason": "report_input_malformed",
                          "error": f"{type(exc).__name__}: {exc}"}, default=str))
        return 2


if __name__ == "__main__":
    sys.exit(main())
