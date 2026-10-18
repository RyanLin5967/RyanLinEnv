"""Sequence aircraft-rl-next-study-v1 (signal-limit v1) seed 42: P0-I, Stage 0, Stage A, Stage B.

Study: docs/aircraft-rl-next-study-v1.json (wins over the .md). The stage order, the init
endpoint of every stage and every update count come from stage_table.py, which refuses the
launch unless it agrees with that JSON. Every stage runs through launch.py's Launcher, so
every freshpool guard applies unchanged: plan identity and pins before every stage, frozen
environment and device, 5 GiB disk reserve and stage headroom, RSS, second accelerator owner,
quiet mode, the 1.25x arm cap charged across voided instances, the live rolling useful-mixed
stop on RL stages, single-use state, the fsynced append-only journal, and an interrupted stage
re-run from scratch, never resumed or quoted. Stage A may instead run its independent arms
concurrently on the UWaterloo hosts (plan ``"stage_a_execution": "uwaterloo"``, uw_stage_a.py);
everything else runs here, serially, on MPS.

Order (study.json stages and gates):
  P0-I at I (G4 rollouts, predicted acceptance and mean u for G in {4,8,16})       -> PI
    (advisory since design-review amendment 1: the evaluator reports its forecast and
    always passes it, so it never stops the study)
  Stage 0 at W: cost probe c(G8), c(G16), each measured over >= 8 screened accepted steps
    at that G, and the beta=0 dr_grpo resume proof at G8 and at G16
    (its validator is the E0 check; c(G) is written to the state before any RL forecast)
  Stage A at W: A1, A2, A3, A5 at every grid LR, H-select each (with constrained greedy);
    A1* (evaluator selection), A4 at A1*'s LR, H-select                              -> GA
  pin I* (GA's i_star, among A1*, A2*, A3*) by run ID and adapter SHA256
  Stage B at I*: P0' (smallest G in {8,16} with predicted acceptance >= a_floor)    -> GB0
    RL x grid to 100 accepted + H-select                                             -> G2
    G2's two continuers to 300 without reset + H-select; RL* (evaluator)
    SFT-E x grid on RL*'s accepted IDs + H-select; SFT-E* (evaluator)
    H-gate RL*, SFT-E*, I*                                                           -> G3a
    SFT-T for G3a's LRs (SFT-E continued to T_RL), LIM-T (the GA winner's objective from
    I* over RL*'s scanned sequence to T_RL at its Stage-A LR), H-select each; SFT-T*
    H-gate SFT-T*, LIM-T                                                             -> G3b
  the evaluator must then report no failure and await G4 (VAL belongs to the native-VAL
  package, sealed TEST to G6; neither is touched here).

Evaluator (plan ``evaluator``: scripts/evaluate_signal_limit_gates.py). It must export GRID,
COMPLETIONS, scan_order, best and Study; ``--evidence E --output R`` writes R with ``gates``
{name: {status, pass, ...}}, ``failed_at`` and ``awaiting``. Read from gate results: GA's
``i_star``, ``winner`` and ``winner_arm``; GB0's ``chosen_group``; G2's ``continue_to_300``;
G3a's ``rl_star``, ``sft_e_star`` and ``sft_t_lrs``. Selections needed before their gate
(A1* for A4's LR, RL*, SFT-E*, SFT-T*) come from the evaluator's own Study, as freshpool's
launcher does. A gate or selection that is missing, unparseable or not passing stops.

Evidence handed to it: ``labels`` and ``rows`` (evidence base), ``s_b_ids`` (every S_b ID, the
scan manifest), ``pi`` and ``p0_prime`` (the P0 documents), ``endpoints`` (W, then every
validated training endpoint with its ``init_sha256`` as this launcher pinned it, the trainer
profile fields of trainer_fragment.py, and ``splits`` h_select / h_gate plus
``h_select_constrained`` / ``h_gate_constrained`` from the constrained greedy pass), and the
look ledger, pins and c(G) for the record.

Look ledger: every H assay is recorded in the state's ``looks`` (endpoint, split, adapter
SHA256, stage) and journalled.

Blind spots, stated: those of launch.py; the init endpoints I and W are identified by their
adapter weights SHA256 (checked against the study text's I hash and W prefix), not by
re-deriving them; the Stage 0 entry (its checks and c(G) measurement) is the plan's template,
judged here only by the fields its fragment reports.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import launch as L  # noqa: E402
import stage_table as T  # noqa: E402
from launch import HEX64, Stop, is_number, jsonl_ids, read_json, sha  # noqa: E402

SCHEMA = "signal-limit-launch-plan-v1"
STAGE_A_TEMPLATES = tuple(f"stage_a_{arm.lower()}" for arm, _, _ in T.STAGE_A_ARMS)
LIM_T_TEMPLATES = tuple(f"lim_t_{arm.lower()}" for arm in T.LIMIT_ARMS)
TRAIN_TEMPLATES = ("stage0", *STAGE_A_TEMPLATES, "rl_screen", "rl_final", "sft_e", "sft_t",
                   *LIM_T_TEMPLATES)
TEMPLATES = (*TRAIN_TEMPLATES, "assay", "p0_i", "p0_prime")
FORECASTS = ("p0", "stage0", "assay_h_select", "assay_h_gate", "sft_e",
             *(f"stage_a_{mode}" for _, mode, _ in T.STAGE_A_ARMS))
GATES = ("PI", "GA", "GB0", "G2", "G3a", "G3b")
EVALUATOR_EXPORTS = ("GRID", "COMPLETIONS", "scan_order", "best", "Study")
G0_MANIFESTS = ("s_a.jsonl", "s_b_scan_seed42.jsonl", "h_select.jsonl", "h_gate.jsonl")
INIT_ROLES = ("I", "W")
# study.json stages P0-I init: "I = controlled400 sft-1e5-time, weights <sha256> (...)".
I_SHA = re.compile(r"weights ([0-9a-f]{64})")
W_PREFIX = re.compile(r"W \(([0-9a-f]{8,})\.\.\.\)")
# Stage 0's resume proof at G8, beta 0, dr_grpo (study.json stages Stage 0: "E0 (4 vs 2+2 vs
# 1+3 byte-identical) for G8, beta=0, dr_grpo, scale_rewards none"), named as e0_gate.py names
# its resume checks, plus what beta 0 changes.
STAGE0_CHECKS = (
    "resume_2_plus_2_identical", "resume_1_plus_3_identical", "finite_nonzero_task_gradient",
    "no_reference_forward_at_beta_0", "loss_type_dr_grpo_recorded",
)
# Design-review amendment 1 (minor finding on cost caps): c(G) is measured, not extrapolated,
# at both G in {8, 16}, over at least this many screened accepted steps each, and E0 runs at
# both G (GB0 may choose either).
STAGE0_MIN_C_STEPS = 8
DIAGNOSTIC_STAGES = ("P0-I", "Stage-0", "P0-prime")  # no endpoint of their own
# trainer_fragment.signal_limit_profile: the RL identity the gates check.
PROFILE_KEYS = ("mode", "loss_type", "scale_rewards", "beta", "group", "reward")
CONTINUERS_KEY = "continue_to_300"  # evaluate_signal_limit_gates gate_g2


# ---------------------------------------------------------------- plan and G0


def study_init_pins(study):
    """(I sha256, W sha256 prefix) as the study text states them."""
    stages = {stage.get("id"): stage for stage in study.get("stages") or []}
    i = I_SHA.search(str((stages.get("P0-I") or {}).get("init")))
    w = W_PREFIX.search(str((stages.get("Stage A") or {}).get("init")))
    return (i.group(1) if i else None), (w.group(1) if w else None)


def verify_counts(plan):
    counts = plan.get("updates")
    if (not isinstance(counts, dict) or set(counts) != set(T.UPDATES)
            or any(type(v) is not int or v < 1 for v in counts.values())
            or counts["rl_screen"] >= counts["final"]):
        raise Stop("plan_update_counts_invalid", updates=counts)
    if counts != T.UPDATES and plan.get("dry_run") is not True:
        # Plan-driven, but a real launch runs the pre-registered counts.
        raise Stop("plan_update_counts_differ_from_study", updates=counts, study=T.UPDATES)
    return counts


def verify_signal_limit_plan(plan_path, expected):
    if sha(plan_path) != expected:
        raise Stop("plan_identity_differs")
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != SCHEMA:
        raise Stop("plan_schema_unknown")
    pinned = plan.get("file_sha256") or {}
    for required in (plan.get("evaluator"), str(HERE / "phase_worker.py"), str(HERE / "launch.py"),
                     str(HERE / "stage_table.py"), str(Path(__file__).resolve())):
        if required is None or not any(L.resolve(p) == L.resolve(required) for p in pinned):
            raise Stop("plan_does_not_pin", file=required)
    L.verify_pins(plan)
    for name, value in (plan.get("environment") or {}).items():
        if L.os.environ.get(name) != value:
            raise Stop("frozen_environment_differs", variable=name)
    L.verify_device(plan)
    study_ref = plan.get("study") or {}
    study_path = L.resolve(study_ref.get("path", "/nonexistent"))
    if not study_path.is_file() or sha(study_path) != study_ref.get("sha256"):
        raise Stop("study_missing_or_changed")
    study = read_json(study_path)
    problems = T.verify_against_study(study if isinstance(study, dict) else {})
    if problems:
        raise Stop("stage_table_differs_from_study", problems=problems)
    counts = verify_counts(plan)
    L.verify_limits(plan, FORECASTS)
    templates = plan.get("templates") or {}
    L.verify_template_shapes(templates, TEMPLATES)
    verify_templates(templates, counts)
    if "init_endpoint" in plan or "w_endpoint" in plan:
        # One plan-wide init would let a stage start from the wrong endpoint.
        raise Stop("init_endpoint_must_be_pinned_per_stage")
    inits = plan.get("init_endpoints") or {}
    for role in INIT_ROLES:
        entry = inits.get(role) or {}
        if not HEX64.fullmatch(str(entry.get("adapter_sha256"))) or not entry.get("adapter_path"):
            raise Stop("init_endpoint_identity_missing", role=role)
    i_sha, w_prefix = study_init_pins(study)
    if plan.get("dry_run") is not True and (
            inits["I"]["adapter_sha256"] != i_sha
            or not w_prefix or not inits["W"]["adapter_sha256"].startswith(w_prefix)):
        raise Stop("init_endpoint_differs_from_study", study_i=i_sha, study_w_prefix=w_prefix)
    trie = plan.get("label_trie") or {}
    trie_path = L.resolve(trie.get("path", "/nonexistent"))
    if not trie_path.is_file() or sha(trie_path) != trie.get("sha256"):
        raise Stop("label_trie_missing_or_changed")
    base = plan.get("evidence_base") or {}
    base_path = L.resolve(base.get("path", "/nonexistent"))
    if not base_path.is_file() or sha(base_path) != base.get("sha256"):
        raise Stop("evidence_base_changed")
    for name in ("data_spec", "model_directory"):
        if not isinstance(plan.get(name), str) or not plan[name]:
            raise Stop("plan_path_missing", field=name)
    execution = plan.get("stage_a_execution", "local")
    if execution not in ("local", "uwaterloo"):
        raise Stop("stage_a_execution_unknown", value=execution)
    if execution == "uwaterloo":
        import uw_stage_a

        uw_stage_a.verify_uw_block(plan)
    elif "uw" in plan:
        raise Stop("uw_block_without_uwaterloo_execution")
    try:
        evaluator = L.load_evaluator(plan)
    except Exception as exc:
        raise Stop("evaluator_unloadable", error=f"{type(exc).__name__}: {exc}") from None
    missing = [name for name in EVALUATOR_EXPORTS if name not in evaluator]
    if missing:
        raise Stop("evaluator_exports_missing", missing=missing)
    if tuple(evaluator["GRID"]) != T.GRID:
        raise Stop("grid_differs_from_evaluator", grid=list(T.GRID), evaluator=evaluator["GRID"])
    if evaluator["COMPLETIONS"] != L.COMPLETIONS:
        raise Stop("completions_differ_from_evaluator", evaluator=evaluator["COMPLETIONS"])
    if not isinstance(plan.get("owner_patterns"), list) or not plan["owner_patterns"]:
        raise Stop("owner_patterns_missing")
    L.verify_quiet_and_headroom(plan)
    return plan, evaluator


def verify_templates(templates, counts):
    """Checked at plan load, before any stage runs (as launch.verify_stage_templates):
    - every template passes ``--device {device}`` exactly once (mps here, cpu on UW);
    - each Stage A template's trainer stops at exactly the plan's Stage A count
      (``--max-steps``), so every arm runs the same 600 updates on the same S_a order;
    - rl_screen/rl_final/sft_e follow launch.py's RL 150 -> 600 and SFT-E rules at the plan's
      counts (100 -> 300 here), and both RL templates pass ``--num-generations {group}``;
    - the assay passes the frozen label trie and its SHA256 (constrained greedy on every H
      assay) and ``--training-group {training_group}``;
    - p0_i predicts G in {4,8,16} at G4; p0_prime chooses G in {8,16} at ``{a_floor}``."""
    for name in TEMPLATES:
        if L.flag_values(templates[name]["argv"], "--device") != ["{device}"]:
            raise Stop("template_device_not_placeholder", template=name)
    for name in STAGE_A_TEMPLATES + LIM_T_TEMPLATES + ("sft_t", "stage0"):
        if (L.wrapper_and_trainer(templates[name]["argv"]) is None
                or not all(isinstance(a, str) for a in templates[name]["argv"])):
            raise Stop("template_missing_wrapper_separator", template=name)
    for name in STAGE_A_TEMPLATES:
        _, trainer = L.wrapper_and_trainer(templates[name]["argv"])
        if L.flag_values(trainer, "--max-steps") != [str(counts["stage_a"])]:
            raise Stop("stage_a_template_max_steps_differs", template=name,
                       max_steps=L.flag_values(trainer, "--max-steps"))
    legacy = {"rl150": templates["rl_screen"], "rl600": templates["rl_final"],
              "sft_e": templates["sft_e"]}
    # launch.py's checks, at this plan's counts. The tokenization rule is freshpool's only.
    L.verify_stage_templates(legacy, {"rl_screen": counts["rl_screen"],
                                      "final": counts["final"]}, tokenized=())
    for name in ("rl_screen", "rl_final"):
        if L.flag_values(templates[name]["argv"], "--num-generations") != ["{group}"]:
            raise Stop("rl_template_group_not_placeholder", template=name)
    argv = templates["assay"]["argv"]
    if (L.flag_values(argv, "--label-trie") != ["{label_trie}"]
            or L.flag_values(argv, "--label-trie-sha256") != ["{label_trie_sha256}"]
            or L.flag_values(argv, "--training-group") != ["{training_group}"]):
        raise Stop("assay_template_without_constrained_greedy")
    p0_i, p0_prime = templates["p0_i"]["argv"], templates["p0_prime"]["argv"]
    if (L.flag_values(p0_i, "--predict-groups") != [",".join(map(str, T.GROUPS))]
            or L.flag_values(p0_i, "--num-generations") != [str(T.P0_GROUP)]
            or L.flag_values(p0_i, "--choose-groups")):
        raise Stop("p0_i_template_differs")
    if (L.flag_values(p0_prime, "--choose-groups") != [",".join(map(str, T.STAGE_B_GROUPS))]
            or L.flag_values(p0_prime, "--a-floor") != ["{a_floor}"]
            or L.flag_values(p0_prime, "--predict-groups")):
        raise Stop("p0_prime_template_differs")


def verify_signal_limit_g0(plan, evaluator):
    """G0, by content: the four manifests hash to the plan's pins, hold unique IDs, do not
    overlap, S_a and S_b are each in their seed-42 order, the evidence base's H rows are the
    H manifests' rows, every image exists and the images hash to the pinned digest.
    a_floor = N x 1.25 / |S_b| is computed here."""
    g0 = plan.get("g0") or {}
    directory = L.resolve(g0.get("manifest_dir", "/nonexistent"))
    files = g0.get("files_sha256") or {}
    if set(files) != set(G0_MANIFESTS):
        raise Stop("g0_manifest_set_differs", files=sorted(files))
    ids, labels = {}, {}
    for name in G0_MANIFESTS:
        path = directory / name
        if not path.is_file() or sha(path) != files[name]:
            raise Stop("g0_manifest_file_differs", file=name)
        found = jsonl_ids(path)
        if not found or len(set(found)) != len(found):
            raise Stop("g0_manifest_invalid", file=name)
        ids[name] = found
        labels[name] = {row["id"]: row.get("label") for row in L.read_jsonl(path)}
    seen = set()
    for name in G0_MANIFESTS:
        if seen & set(ids[name]):
            raise Stop("g0_manifests_overlap", file=name)
        seen |= set(ids[name])
    for name in ("s_a.jsonl", "s_b_scan_seed42.jsonl"):
        if ids[name] != evaluator["scan_order"](ids[name], T.SEED):
            raise Stop("g0_manifest_not_seed42_order", file=name)
    if len(ids["s_a.jsonl"]) < plan["updates"]["stage_a"]:
        raise Stop("g0_s_a_shorter_than_stage_a", s_a=len(ids["s_a.jsonl"]))
    if len(ids["s_b_scan_seed42.jsonl"]) < L.G1_CANDIDATES:
        raise Stop("g0_s_b_shorter_than_p0", s_b=len(ids["s_b_scan_seed42.jsonl"]))
    base = read_json(L.resolve(plan["evidence_base"]["path"])) or {}
    for split in L.SPLITS:
        rows = (base.get("rows") or {}).get(split)
        if not isinstance(rows, list) or {r.get("id"): r.get("label") for r in rows} != labels[
                f"{split}.jsonl"]:
            raise Stop("g0_evidence_rows_differ_from_manifest", split=split)
    L.verify_manifest_images(directory, g0.get("images_sha256"), G0_MANIFESTS)
    floor = T.a_floor(plan["updates"]["final"], len(ids["s_b_scan_seed42.jsonl"]))
    manifests = {f"manifest_{Path(name).stem}": str(directory / name) for name in G0_MANIFESTS}
    return {"checks": {"manifests_hashed": True, "no_overlap": True, "seed42_orders": True,
                       "evidence_rows_match": True, "images_pinned": True},
            "a_floor": floor, "s_b_size": len(ids["s_b_scan_seed42.jsonl"]),
            "manifests": manifests, "scan_ids": ids["s_b_scan_seed42.jsonl"],
            "s_a_ids": ids["s_a.jsonl"],
            "split_ids": {"h_select": ids["h_select.jsonl"], "h_gate": ids["h_gate.jsonl"]}}


def verify(plan_path, expected):
    plan, evaluator = verify_signal_limit_plan(plan_path, expected)
    g0 = verify_signal_limit_g0(plan, evaluator)

    def make(ledger):
        return SignalLimitLauncher(plan, expected, evaluator, ledger, g0)

    return plan, make, {"g0": g0["checks"], "a_floor": g0["a_floor"],
                        "s_b_size": g0["s_b_size"], "grid": list(T.GRID),
                        "updates": plan["updates"],
                        "stage_a_execution": plan.get("stage_a_execution", "local")}


# ---------------------------------------------------------------- the launcher


class SignalLimitLauncher(L.Launcher):
    def __init__(self, plan, plan_sha, evaluator, ledger, g0):
        super().__init__(plan, plan_sha, list(T.GRID), evaluator, ledger, g0)
        self.counts = plan["updates"]
        for key in ("looks", "pinned", "endpoint_init"):
            self.state.setdefault(key, [] if key == "looks" else {})

    # -- identities
    def init(self, role):
        """The pinned init endpoint of a role: I and W from the plan, I* from the state."""
        if role in INIT_ROLES:
            entry = self.plan["init_endpoints"][role]
            return {"run_id": role, "adapter_path": entry["adapter_path"],
                    "adapter_sha256": entry["adapter_sha256"]}
        pinned = self.state["pinned"].get("I*")
        if pinned is None:
            raise Stop("i_star_not_pinned")
        return pinned

    def stage_values(self, role, **values):
        init = self.init(role)
        return {"init_adapter": init["adapter_path"], "init_role": role,
                "w_adapter": self.plan["init_endpoints"]["W"]["adapter_path"],
                "device": self.device(), "data_spec": str(L.resolve(self.plan["data_spec"])),
                "model_directory": str(self.plan["model_directory"]),
                "label_trie": str(L.resolve(self.plan["label_trie"]["path"])),
                "label_trie_sha256": self.plan["label_trie"]["sha256"],
                "a_floor": repr(self.g0["a_floor"]), **values}

    def device(self):
        return self.plan["environment"]["ACCELERATE_TORCH_DEVICE"]

    def pin_endpoint_init(self, run_id, init_sha256):
        """The init this launcher started ``run_id`` from (the evidence's init_sha256)."""
        if self.state["endpoint_init"].get(run_id, init_sha256) != init_sha256:
            raise Stop("endpoint_init_changed_on_resume", run_id=run_id)
        if run_id not in self.state["endpoint_init"]:
            self.state["endpoint_init"][run_id] = init_sha256
            self.ledger.save()

    # -- validators
    def p0_validator(self, role, predict=None, choose=None):
        init = self.init(role)

        def validate(f, cap):
            if not all(k in f for k in ("seed", "adapter_sha256", "scan_order", "journal",
                                        "rollouts")):
                return "p0_document_incomplete"
            if f["seed"] != T.SEED or f["adapter_sha256"] != init["adapter_sha256"]:
                return "p0_not_seed42_under_its_init"
            if f["scan_order"] != self.g0["scan_ids"]:
                return "p0_scan_order_differs_from_manifest"
            if [row.get("id") for row in f["journal"]] != self.g0["scan_ids"][:L.G1_CANDIDATES]:
                return "p0_candidates_not_first_400_of_s_b"
            predicted = f.get("predicted")
            groups = predict or choose
            if not isinstance(predicted, dict) or sorted(predicted) != sorted(
                    str(g) for g in groups):
                return "p0_predictions_missing"
            for entry in predicted.values():
                if not is_number(entry.get("acceptance")) or not 0 <= entry["acceptance"] <= 1:
                    return "p0_predictions_invalid"
            group = T.P0_GROUP
            if choose is not None:
                group = f.get("chosen_group")
                if f.get("a_floor") != self.g0["a_floor"] or f.get("choose_groups") != list(
                        choose):
                    return "p0_prime_floor_or_groups_differ"
                if group is None:
                    # No G qualifies: nothing was sampled; GB0 fails on this document.
                    return None if f["rollouts"] == [] else "p0_prime_rollouts_without_group"
                if group not in choose:
                    return "p0_prime_chosen_group_invalid"
            elif f.get("group") != T.P0_GROUP:
                return "p0_i_not_g4"
            if len(f["rollouts"]) != L.G1_ROLLOUTS or any(
                    len(r.get("completions", [])) != group for r in f["rollouts"]):
                return "p0_rollouts_wrong_size"
            return None

        return validate

    @staticmethod
    def stage0_validator(f, cap):
        """Stage 0: every resume / beta-0 check true at G8 and G16 (``e0_groups``), sampler-
        teacher agreement <= 0.001 (as E0), and c(G) > 0 for G8 and G16, each measured over
        at least STAGE0_MIN_C_STEPS accepted steps (study.json Stage 0 writes)."""
        checks = f.get("checks")
        if (not isinstance(checks, dict) or not set(STAGE0_CHECKS) <= set(checks)
                or any(v is not True for v in checks.values())):
            return "e0_failed"
        if f.get("e0_groups") != list(T.STAGE_B_GROUPS):
            return "stage0_e0_not_at_every_g"
        steps = f.get("c_accepted_steps")
        if not isinstance(steps, dict) or any(
                type(steps.get(str(g))) is not int or steps[str(g)] < STAGE0_MIN_C_STEPS
                for g in T.STAGE_B_GROUPS):
            return "stage0_c_g_not_measured_over_enough_steps"
        agreement = f.get("sampler_teacher_max_abs_logprob_difference")
        if not is_number(agreement) or not 0 <= agreement <= 0.001:
            return "e0_sampler_teacher_agreement_failed"
        costs = f.get("c_seconds_per_accepted_update")
        if not isinstance(costs, dict) or any(
                not is_number(costs.get(str(g))) or costs[str(g)] <= 0
                for g in T.STAGE_B_GROUPS):
            return "stage0_c_g_missing"
        return None

    def stage_a_check(self, arm):
        """Every Stage A arm: its objective's mode, and exactly the first N S_a IDs in S_a's
        order (study.json budget_matching.stage_a: 'the same S_a order')."""
        expected = self.g0["s_a_ids"][: self.counts["stage_a"]]
        mode = dict((a, m) for a, m, _ in T.STAGE_A_ARMS)[arm]

        def check(f):
            if f.get("mode") != mode:
                return "stage_a_mode_differs"
            if f.get("committed_prompt_ids") != expected:
                return "stage_a_exposure_not_s_a_order"
            return None
        return check

    @staticmethod
    def lim_t_check(mode, t_rl, scanned_manifest):
        """LIM-T: the Gate-A winner's objective, from I* over RL*'s scanned sequence from row
        0, cycling, stopping at the first step at or above T_RL (the step before is below)."""
        scanned = jsonl_ids(scanned_manifest)

        def check(f):
            if f.get("mode") != mode:
                return "lim_t_mode_is_not_the_gate_a_winner_objective"
            ids = f.get("committed_prompt_ids")
            if (not scanned or not isinstance(ids, list) or len(ids) != f["updates"]
                    or ids != [scanned[n % len(scanned)] for n in range(len(ids))]):
                return "lim_t_exposure_not_scanned_from_row_0"
            steps = f["step_committed_active_seconds"]
            if min(map(int, steps)) != 1:
                return "lim_t_not_from_step_1"
            if f["committed_active_seconds"] < t_rl - 1e-9:
                return "lim_t_short_of_t_rl"
            if f["updates"] > 1 and steps[str(f["updates"] - 1)] >= t_rl:
                return "lim_t_overshoots_first_boundary_at_t_rl"
            return None
        return check

    def rl_check(self, group, reward, then=None):
        """RL: the profile the study fixes (dr_grpo, scale_rewards none, beta 0) at the chosen
        G with the Gate-A winner's reward, as the trainer recorded it."""
        def check(f):
            profile = {k: f.get(k) for k in ("loss_type", "scale_rewards", "group", "reward")}
            if profile != {"loss_type": "dr_grpo", "scale_rewards": "none", "group": group,
                           "reward": reward} or not (is_number(f.get("beta"))
                                                     and f["beta"] == 0):
                return "rl_profile_differs"
            return then(f) if then else None
        return check

    def assay_validator(self, run_id, split, adapter):
        base = super().assay_validator(run_id, split, adapter)
        expected = set(self.g0["split_ids"][split])
        trie = self.plan["label_trie"]["sha256"]

        def validate(f, cap):
            problem = base(f, cap)
            if problem:
                return problem
            constrained = f.get("constrained_greedy")
            if not isinstance(constrained, dict) or constrained.get("trie_sha256") != trie:
                return "assay_constrained_greedy_missing_or_other_trie"
            predictions = constrained.get("predictions")
            if not isinstance(predictions, dict) or set(predictions) != expected or any(
                    a is not None and not isinstance(a, str) for a in predictions.values()):
                return "assay_constrained_greedy_ids_differ"
            if f.get("assay_group") != T.ASSAY_GROUP:
                return "assay_group_not_4"
            return None

        return validate

    # -- assays and the look ledger
    def assay_values(self, endpoint, split, adapter_path, group):
        """``training_group``: the RL endpoint's G, or 0 (h_assay: not an RL endpoint)."""
        return self.stage_values("W", run_id=endpoint, split=split, adapter=adapter_path,
                                 split_manifest=self.g0["manifests"][f"manifest_{split}"],
                                 training_group=str(group))

    def assay(self, endpoint, split, adapter_path, adapter_sha, group=0):
        stage = f"{endpoint}-{split}"
        fragment = self.run(stage, "assay", stage, self.plan["forecast_seconds"][f"assay_{split}"],
                            self.assay_values(endpoint, split, adapter_path, group),
                            self.assay_validator(endpoint, split, adapter_sha))
        self.record_look(endpoint, split, adapter_sha, stage)
        return fragment

    def assay_fragment(self, fragment, split, group=0):
        return self.assay(fragment["run_id"], split, fragment["adapter_path"],
                          fragment["adapter_sha256"], group)

    def record_look(self, endpoint, split, adapter_sha, stage):
        look = {"endpoint": endpoint, "split": split, "adapter_sha256": adapter_sha,
                "stage": stage}
        if look not in self.state["looks"]:
            self.state["looks"].append(look)
            self.ledger.save()
            self.ledger.append("look", **look)

    # -- Stage A (local or UWaterloo)
    def stage_a_jobs(self, runs):
        """Launcher.run's arguments for each Stage A run, as independent jobs."""
        w = self.init("W")
        jobs = []
        for arm, mode, lr, run_id in runs:
            self.pin_endpoint_init(run_id, w["adapter_sha256"])
            jobs.append({
                "stage_id": run_id, "template": f"stage_a_{arm.lower()}", "arm": run_id,
                "forecast": self.plan["forecast_seconds"][f"stage_a_{mode}"],
                "forecast_key": f"stage_a_{mode}",
                "values": self.stage_values("W", run_id=run_id, lr=repr(lr), arm_label=arm),
                "validate": self.train_validator(run_id, arm, lr, self.counts["stage_a"],
                                                 check=self.stage_a_check(arm),
                                                 init_sha256=w["adapter_sha256"])})
        return jobs

    def assay_jobs(self, run_ids, split="h_select"):
        jobs = []
        for run_id in run_ids:
            f = self.fragment(run_id)
            stage = f"{run_id}-{split}"
            jobs.append({
                "stage_id": stage, "template": "assay", "arm": stage,
                "forecast": self.plan["forecast_seconds"][f"assay_{split}"],
                "forecast_key": f"assay_{split}",
                "values": self.assay_values(run_id, split, f["adapter_path"], 0),
                "validate": self.assay_validator(run_id, split, f["adapter_sha256"]),
                "look": (run_id, split, f["adapter_sha256"], stage)})
        return jobs

    def run_jobs(self, jobs):
        """Independent stages: serially here, or concurrently on UWaterloo hosts."""
        if self.plan.get("stage_a_execution", "local") == "uwaterloo":
            import uw_stage_a

            uw_stage_a.UWRunner(self).run_batch(jobs)
        else:
            for job in jobs:
                self.run(job["stage_id"], job["template"], job["arm"], job["forecast"],
                         job["values"], job["validate"])
        for job in jobs:
            if "look" in job:
                self.record_look(*job["look"])

    def stage_a(self):
        first = T.stage_a_runs()
        self.run_jobs(self.stage_a_jobs(first))
        self.run_jobs(self.assay_jobs([run_id for *_, run_id in first]))
        a1 = {run_id for arm, _, _, run_id in first if arm == "A1"}
        a1_star = self.choose("A1*", a1, lambda s: self.evaluator["best"](s.h_scored(
            s.exact_updates(s.family("A1"), self.counts["stage_a"])), "macro")["run_id"])
        a4 = [run for run in T.stage_a_runs(self.fragment(a1_star)["lr"]) if run[0] == "A4"]
        self.run_jobs(self.stage_a_jobs(a4))
        self.run_jobs(self.assay_jobs([a4[0][3]]))

    def pin_i_star(self, ga):
        candidates = {run_id for arm, _, _, run_id in T.stage_a_runs() if arm in T.I_STAR_FROM}
        run_id = ga.get("i_star")
        if run_id not in candidates or f"{run_id}-h_select" not in self.state["completed"]:
            raise Stop("ga_i_star_missing_or_invalid", i_star=run_id)
        self.record_selection("I*", run_id)
        f = self.fragment(run_id)
        pinned = {"run_id": run_id, "adapter_path": f["adapter_path"],
                  "adapter_sha256": f["adapter_sha256"]}
        if self.state["pinned"].get("I*", pinned) != pinned:
            raise Stop("i_star_changed_on_resume")
        if "I*" not in self.state["pinned"]:
            self.state["pinned"]["I*"] = pinned
            self.ledger.save()
            self.ledger.append("init_endpoint_pinned", role="I*", **pinned)
        return pinned

    def ga_winner(self, ga):
        winners = {run_id: arm for arm, _, _, run_id in T.stage_a_runs() if arm in T.LIMIT_ARMS}
        run_id, arm = ga.get("winner"), ga.get("winner_arm")
        if winners.get(run_id) != arm or f"{run_id}-h_select" not in self.state["completed"]:
            raise Stop("ga_winner_missing_or_invalid", winner=run_id, winner_arm=arm)
        self.record_selection("GA_winner", run_id)
        return run_id, arm

    # -- c(G)
    def freeze_c(self, fragment):
        if "c_seconds_per_accepted_update" not in self.state:
            self.state["c_seconds_per_accepted_update"] = {
                str(g): fragment["c_seconds_per_accepted_update"][str(g)]
                for g in T.STAGE_B_GROUPS}
            self.ledger.save()
            self.ledger.append("c_g_written", **self.state["c_seconds_per_accepted_update"])

    def rl_forecast(self, group):
        """Forecast active seconds of a whole RL arm: c(G) from Stage 0 x N (refusal
        condition 'c(G) missing when an RL forecast is computed')."""
        c = (self.state.get("c_seconds_per_accepted_update") or {}).get(str(group))
        if not is_number(c) or c <= 0:
            raise Stop("c_g_missing_for_rl_forecast", group=group)
        return c * self.counts["final"]

    # -- the study sequence
    def sequence(self):
        forecast = self.plan["forecast_seconds"]
        scan = self.g0["manifests"]["manifest_s_b_scan_seed42"]
        self.run("P0-I", "p0_i", "P0-I", forecast["p0"],
                 self.stage_values("I", run_id="P0-I", scan_manifest=scan),
                 self.p0_validator("I", predict=T.GROUPS))
        self.gate("PI")
        stage0 = self.run("Stage-0", "stage0", "Stage-0", forecast["stage0"],
                          self.stage_values("W", run_id="Stage-0", scan_manifest=scan),
                          self.stage0_validator)
        self.freeze_c(stage0)

        self.stage_a()
        ga = self.gate("GA")
        i_star = self.pin_i_star(ga)
        winner_id, winner = self.ga_winner(ga)
        init = i_star["adapter_sha256"]

        p0 = self.run("P0-prime", "p0_prime", "P0-prime", forecast["p0"],
                      self.stage_values("I*", run_id="P0-prime", scan_manifest=scan),
                      self.p0_validator("I*", choose=T.STAGE_B_GROUPS))
        gb0 = self.gate("GB0")
        group = p0.get("chosen_group")
        if group not in T.STAGE_B_GROUPS or gb0.get("chosen_group") != group:
            raise Stop("gb0_group_disagrees_with_p0_prime", p0=group,
                       gb0=gb0.get("chosen_group"))
        self.record_selection("group", group)
        rl_forecast = self.rl_forecast(group)
        reward = T.RL_REWARDS[winner]

        rl = {}
        for lr in T.GRID:
            arm = f"rl-{T.lr_tag(lr)}-s{T.SEED}"
            run_id = f"{arm}-u{self.counts['rl_screen']}"
            self.pin_endpoint_init(run_id, init)
            rl[run_id] = self.run(
                run_id, "rl_screen", arm, rl_forecast,
                self.stage_values("I*", run_id=run_id, lr=repr(lr), group=str(group),
                                  scan_manifest=scan),
                self.train_validator(run_id, "RL", lr, self.counts["rl_screen"], rl=True,
                                     check=self.rl_check(group, reward), init_sha256=init),
                watch_mixed=True)
            self.assay_fragment(rl[run_id], "h_select", group)
        g2 = self.gate("G2")
        continuers = g2.get(CONTINUERS_KEY)
        if not isinstance(continuers, list) or len(continuers) != 2 or len(
                set(continuers)) != 2 or any(c not in rl for c in continuers):
            raise Stop("g2_continuers_missing", continuers=continuers)
        self.record_selection("rl_continuers", continuers)
        final = {}
        for parent_id in continuers:
            parent = rl[parent_id]
            lr = parent["lr"]
            arm = f"rl-{T.lr_tag(lr)}-s{T.SEED}"
            run_id = f"{arm}-u{self.counts['final']}"
            self.pin_endpoint_init(run_id, init)
            final[run_id] = self.run(
                run_id, "rl_final", arm, rl_forecast,
                self.stage_values("I*", run_id=run_id, lr=repr(lr), group=str(group),
                                  scan_manifest=scan, resume_checkpoint=parent["checkpoint"]),
                self.train_validator(run_id, "RL", lr, self.counts["final"], rl=True,
                                     check=self.rl_check(group, reward,
                                                         self.continuation_check(parent)),
                                     init_sha256=init),
                inherited_seconds=parent["committed_active_seconds"], watch_mixed=True)
            self.assay_fragment(final[run_id], "h_select", group)
        rl_star = self.fragment(self.choose(
            "rl_star", set(final), lambda s: s.select_rl_star()[1]["run_id"]))

        sft_e = {}
        for lr in T.GRID:
            run_id = f"sft-e-{T.lr_tag(lr)}-s{T.SEED}"
            self.pin_endpoint_init(run_id, init)
            sft_e[lr] = self.run(
                run_id, "sft_e", run_id, forecast["sft_e"],
                self.stage_values("I*", run_id=run_id, lr=repr(lr),
                                  accepted_manifest=rl_star["accepted_manifest"]),
                self.train_validator(
                    run_id, "SFT-E", lr, self.counts["final"],
                    check=self.exposure_check(
                        rl_star["accepted_manifest"],
                        self.sft_e_boundary_check(self.counts["final"]), self.counts["final"]),
                    init_sha256=init))
            self.assay_fragment(sft_e[lr], "h_select")
        sft_e_star = self.choose("sft_e_star", {f["run_id"] for f in sft_e.values()},
                                 lambda s: s.select_sft_e()[1]["run_id"])
        self.assay_fragment(rl_star, "h_gate", group)
        self.assay_fragment(self.fragment(sft_e_star), "h_gate")
        self.assay(i_star["run_id"], "h_gate", i_star["adapter_path"], i_star["adapter_sha256"])
        g3a = self.gate("G3a")
        if g3a.get("rl_star") != rl_star["run_id"] or g3a.get("sft_e_star") != sft_e_star:
            raise Stop("g3a_selection_disagrees_with_launch", rl_star=g3a.get("rl_star"),
                       sft_e_star=g3a.get("sft_e_star"))
        qualifying = g3a.get("sft_t_lrs")
        if (not isinstance(qualifying, list) or not qualifying
                or any(lr not in T.GRID for lr in qualifying)):
            raise Stop("g3a_sft_t_lrs_missing", sft_t_lrs=qualifying)
        self.record_selection("sft_t_lrs", qualifying)

        t_rl = rl_star["committed_active_seconds"]
        sft_t = set()
        for lr in qualifying:
            run_id = f"sft-t-{T.lr_tag(lr)}-s{T.SEED}"
            parent = sft_e[lr]
            # The evaluator's SFT-T init is its SFT-E endpoint (it continues without reset).
            self.pin_endpoint_init(run_id, parent["adapter_sha256"])
            fragment = self.run(
                run_id, "sft_t", run_id, t_rl,
                self.stage_values("I*", run_id=run_id, lr=repr(lr), t_rl=repr(t_rl),
                                  scanned_manifest=rl_star["scanned_manifest"],
                                  parent_checkpoint=parent["checkpoint"]),
                self.train_validator(run_id, "SFT-T", lr,
                                     check=self.sft_t_check(parent, t_rl,
                                                            rl_star["scanned_manifest"]),
                                     init_sha256=init),
                inherited_seconds=parent["committed_active_seconds"],
                lineage=(f"sft-e-{T.lr_tag(lr)}-s{T.SEED}",))
            sft_t.add(run_id)
            self.assay_fragment(fragment, "h_select")
        lr = self.fragment(winner_id)["lr"]
        lim_id = f"lim-t-{T.lr_tag(lr)}-s{T.SEED}"
        self.pin_endpoint_init(lim_id, init)
        lim = self.run(
            lim_id, f"lim_t_{winner.lower()}", lim_id, t_rl,
            self.stage_values("I*", run_id=lim_id, lr=repr(lr), t_rl=repr(t_rl),
                              group=str(group), arm_label="LIM-T",
                              scanned_manifest=rl_star["scanned_manifest"]),
            self.train_validator(lim_id, "LIM-T", lr,
                                 check=self.lim_t_check(T.LIM_T_MODES[winner], t_rl,
                                                        rl_star["scanned_manifest"]),
                                 init_sha256=init))
        self.assay_fragment(lim, "h_select")
        sft_t_star = self.choose("sft_t_star", sft_t, lambda s: s.select_sft_t()[1]["run_id"])
        self.assay_fragment(self.fragment(sft_t_star), "h_gate")
        self.assay_fragment(lim, "h_gate")
        self.gate("G3b")
        self.final_check()

    # -- evaluator
    def study(self):
        try:
            return self.evaluator["Study"](self.evidence())
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise Stop("selection_failed", error=f"{type(exc).__name__}: {exc}") from None

    def choose(self, name, allowed, compute):
        """A selection the sequence needs before its gate, from the evaluator's own Study
        (H-select only, as the gates select); recorded once, and never changed on resume."""
        cached = self.state["selections"].get(name)
        if cached is None:
            L.verify_pins(self.plan)
            try:
                cached = compute(self.study())
            except Stop:
                raise
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise Stop("selection_failed", selection=name,
                           error=f"{type(exc).__name__}: {exc}") from None
        if cached not in allowed:
            raise Stop("selection_missing_or_invalid", selection=name, value=cached,
                       allowed=sorted(map(str, allowed)))
        if f"{cached}-h_select" not in self.state["completed"]:
            raise Stop("selection_without_h_select_assay", selection=name, value=cached)
        self.record_selection(name, cached)
        return cached

    def record_selection(self, name, value):
        if self.state["selections"].get(name) == value:
            return
        super().record_selection(name, value)

    def endpoints(self):
        w = self.init("W")
        found = {"W": {"run_id": "W", "arm": "W", "lr": None, "seed": None, "updates": 0,
                       "committed_active_seconds": 0, "adapter_sha256": w["adapter_sha256"],
                       "reference_sha256": w["adapter_sha256"], "useful_mixed": [],
                       "splits": {}}}
        keys = ("run_id", "arm", "lr", "seed", "updates", "committed_active_seconds",
                "adapter_sha256", "reference_sha256", "useful_mixed")
        assays = []
        for stage, entry in self.state["completed"].items():
            f = entry["fragment"]
            if "split" in f:
                assays.append(f)
            elif stage not in DIAGNOSTIC_STAGES:
                found[f["run_id"]] = {
                    **{k: f.get(k) for k in keys},
                    **{k: f[k] for k in PROFILE_KEYS if k in f},
                    "init_sha256": self.state["endpoint_init"].get(f["run_id"]),
                    "splits": {}}
        for f in assays:
            if f["run_id"] not in found:
                raise Stop("assay_for_unknown_endpoint", run_id=f["run_id"])
            splits = found[f["run_id"]]["splits"]
            splits[f["split"]] = {"predictions": f["predictions"],
                                  "eos_fraction": f["eos_fraction"]}
            constrained = f.get("constrained_greedy")
            if constrained is not None:
                splits[f"{f['split']}_constrained"] = {
                    "predictions": constrained["predictions"],
                    "trie_sha256": constrained["trie_sha256"],
                    "eos_fraction": constrained.get("eos_fraction")}
        return found

    def evidence(self):
        base = json.loads(L.resolve(self.plan["evidence_base"]["path"]).read_text())
        document = {k: base[k] for k in ("labels", "rows")}
        document.update(
            s_b_ids=list(self.g0["scan_ids"]), a_floor=self.g0["a_floor"],
            updates=self.counts, endpoints=list(self.endpoints().values()),
            looks=list(self.state["looks"]), pinned=dict(self.state["pinned"]),
            selections=dict(self.state["selections"]),
            c_seconds_per_accepted_update=self.state.get("c_seconds_per_accepted_update"))
        for stage, key in (("P0-I", "pi"), ("Stage-0", "stage0"), ("P0-prime", "p0_prime")):
            if stage in self.state["completed"]:
                document[key] = self.state["completed"][stage]["fragment"]
        return document

    def final_check(self):
        code, _, report, report_path = self.run_evaluator("final")
        gates = (report or {}).get("gates")
        ok = (code == 0 and isinstance(gates, dict) and report.get("failed_at") is None
              and report.get("awaiting") == "G4"
              and all(isinstance(gates.get(g), dict) and gates[g].get("pass") is True
                      for g in GATES))
        self.ledger.append("final_check", report=str(report_path), passed=ok)
        if not ok:
            raise Stop("final_development_check_failed", report=str(report_path))


def main(argv=None):
    return L.launch_main(argv, verify, first_action="P0-I",
                         description=__doc__.splitlines()[0])


if __name__ == "__main__":
    sys.exit(main())
