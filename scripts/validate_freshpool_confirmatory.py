"""Validate the freshpool confirmatory pre-registration and guard its single-use TEST envelope.

Pure logic over committed JSON and recorded evidence: no model, image or accelerator access,
and no file under the sealed TEST directory is ever opened here.

  validate   refuse a malformed plan, or one whose TEST envelope was already used
  authorize  (activated plan only) run the pinned gate evaluator on PRE-TEST evidence; if G4 and
             G5 passed, write the envelope receipt by exclusive create, naming exactly the four
             allowlisted TEST identities
  claim      record one TEST look for one allowlisted role and identity, by exclusive create

Every refusal exits 2 and names its refusal id (plan refusal_conditions).

Blind spots, stated: the guard binds cooperating runners only; a process that opens the sealed
directory without calling ``claim`` is not seen. Envelope history is read from ``git log`` of the
given repository, so a shallow clone sees only its own commits. Prediction provenance (which
adapter produced a TEST file) is trusted, as the gate evaluator states.
"""

import argparse
import copy
import hashlib
import json
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("docs/aircraft-freshpool-confirmatory-v1.json")
ACTIVATED_PLAN = Path("docs/aircraft-freshpool-confirmatory-v1-activated.json")
PLAN_ID = "aircraft-freshpool-confirmatory-v1"
STUDY_ID = "aircraft-freshpool-screened-grpo-v1"
REQUIRED_KEYS = {
    "schema_version", "plan_id", "status", "activated", "activation", "pins",
    "inherited_unchanged", "restated_thresholds", "branch", "entry_condition", "replication",
    "test", "primary_endpoint", "secondary_confirmatory", "multiplicity", "gate_outcomes",
    "refusal_conditions", "test_seal", "activation_prerequisites", "amendments",
}
REQUIRED_PINS = {"protocol_activated", "protocol_revision", "study", "branch_receipt",
                 "named_baseline_plan", "gate_evaluator", "comparison_module"}
INHERITED = {
    "substantial_gain_gates": ("substantial_gain_gates",),
    "strong_baseline": ("strong_baseline",),
    "evaluation_and_seal.uncertainty": ("evaluation_and_seal", "uncertainty"),
    "evaluation_and_seal.final_test": ("evaluation_and_seal", "final_test"),
}
# The TEST allowlist. Nothing outside these roles may ever be named for TEST.
TEST_ROLES = ("rl_star_seed42", "strongest_sft_protocol_pool", "initial_policy_w",
              "named_baseline_27b")
SEQUENCE = ["G4_val_seed42", "G5_val_replication", "G6a_test_rl_vs_strongest_sft",
            "G6b_test_rl_vs_named_baseline"]
TEST_ACCESS_OUTCOME = "G4_pass_G5_pass"
REQUIRED_OUTCOMES = {
    "development_stop_before_G4", "G4_protocol_fail", "G4_protocol_pass_robust_fail",
    "G4_pass_G5_fail", TEST_ACCESS_OUTCOME, "G6a_fail", "G6a_pass_baseline_interface_failure",
    "G6a_pass_G6b_fail", "G6a_pass_G6b_pass", "test_run_incomplete",
}
REQUIRED_REFUSALS = {
    "plan_not_activated", "pin_mismatch", "inherited_block_changed", "seeds_or_n_changed",
    "branch_mismatch", "test_not_sealed", "contender_not_allowlisted", "envelope_already_used",
    "evidence_contains_test", "gates_not_passed", "evidence_branch_mismatch",
    "w_identity_mismatch", "look_not_allowlisted", "look_already_taken",
}
ENVELOPE_DIR = "reports/aircraft-freshpool-confirmatory-v1/"
ACTIVATION_KEYS = {"status", "activated", "activation"}


class Refusal(Exception):
    def __init__(self, refusal_id, detail):
        super().__init__(f"{refusal_id}: {detail}")
        self.refusal_id = refusal_id


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refusal("malformed", f"cannot read JSON {path}: {error}") from error


def dig(document, keys):
    for key in keys:
        document = document[key]
    return document


def envelope_used(repo_root, receipt_rel):
    """True if the receipt exists or ever existed in git history. Unreadable history refuses."""
    if (Path(repo_root) / receipt_rel).exists():
        return True
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "--all", "--format=%H", "--", receipt_rel],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Refusal("envelope_already_used", f"cannot read git history: {error}") from error
    if result.returncode != 0:
        raise Refusal("envelope_already_used",
                      f"cannot read git history of {repo_root}: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def _require(condition, refusal_id, detail):
    if not condition:
        raise Refusal(refusal_id, detail)


def validate_plan(plan, repo_root=ROOT, *, require_activated=False):
    """Return (plan, protocol) or raise Refusal. ``plan`` is a parsed document.

A plan the validator cannot parse (a missing nested field, a wrong type) is refused as
malformed; it never falls through to valid."""
    try:
        return _validate(plan, Path(repo_root), require_activated)
    except Refusal:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise Refusal("malformed", f"{type(error).__name__}: {error}") from error


def _validate(plan, repo_root, require_activated):
    _require(isinstance(plan, dict), "malformed", "plan is not a JSON object")
    missing = REQUIRED_KEYS - plan.keys()
    _require(not missing, "malformed", f"missing keys {sorted(missing)}")
    _require(plan["plan_id"] == PLAN_ID, "malformed", f"plan_id must be {PLAN_ID}")
    _require(plan["schema_version"] == 1, "malformed", "schema_version must be 1")
    _require(type(plan["activated"]) is bool, "malformed", "activated must be true or false")
    _require(isinstance(plan["amendments"], list), "malformed", "amendments must be a list")

    if plan["activated"]:
        _check_activation(plan, repo_root)
    else:
        _require(require_activated is False, "plan_not_activated",
                 "authorize needs the activated copy")
        _require(str(plan["status"]).startswith("UNACTIVATED"), "malformed",
                 "an unactivated plan's status must start with UNACTIVATED")
        _require(plan["activation"] is None, "malformed",
                 "an unactivated plan must have activation null")

    pins = plan["pins"]
    _require(isinstance(pins, dict) and REQUIRED_PINS <= pins.keys(), "malformed",
             f"pins must include {sorted(REQUIRED_PINS)}")
    for name, pin in pins.items():
        path = repo_root / pin["path"]
        _require(path.is_file(), "pin_mismatch", f"{name}: {pin['path']} missing")
        actual = sha256_file(path)
        _require(actual == pin["sha256"], "pin_mismatch",
                 f"{name}: {pin['path']} sha256 {actual} != pinned {pin['sha256']}")

    protocol = load_json(repo_root / pins["protocol_activated"]["path"])
    _require(protocol.get("activated") is True
             and protocol.get("protocol_id") == pins["protocol_activated"]["protocol_id"],
             "pin_mismatch", "protocol_activated must be the activated protocol copy")
    receipt = load_json(repo_root / pins["branch_receipt"]["path"])

    inherited = plan["inherited_unchanged"]
    for name, keys in INHERITED.items():
        _require(inherited.get(name) == canonical_sha256(dig(protocol, keys)),
                 "inherited_block_changed", f"{name} differs from the activated protocol")
    _check_thresholds(plan["restated_thresholds"], protocol)

    _require(str(protocol["data"]["test_status"]).startswith("SEALED"), "test_not_sealed",
             "activated protocol data.test_status is not SEALED")

    staging = protocol["exposure_and_staging"]
    replication = plan["replication"]
    _require(replication["primary_seed"] == staging["primary_seed"]
             and replication["seeds"] == staging["conditional_replication_seeds"]
             and replication["N_accepted_updates"] == staging["N_accepted_updates"],
             "seeds_or_n_changed", "seeds or N differ from the activated protocol")
    _require(sorted(replication["scan_manifests"]) == sorted(str(s) for s in replication["seeds"]),
             "seeds_or_n_changed", "one scan manifest per replication seed")

    branch = plan["branch"]
    _require(receipt["sources"].get("test_consumed") is False, "branch_mismatch",
             "branch receipt does not record test_consumed false")
    for key in ("mode", "d_req", "g3_margin", "b4_edge_extension", "a_star"):
        _require(branch.get(key) == receipt.get(key), "branch_mismatch",
                 f"branch.{key} {branch.get(key)!r} != receipt {receipt.get(key)!r}")

    _check_test(plan["test"], protocol)
    _check_primary(plan["primary_endpoint"], protocol)
    _require(plan["multiplicity"].get("procedure") == "fixed_sequence_conjunctive"
             and plan["multiplicity"].get("sequence") == SEQUENCE, "malformed",
             f"multiplicity must be fixed_sequence_conjunctive over {SEQUENCE}")

    outcomes = plan["gate_outcomes"]
    _require(REQUIRED_OUTCOMES <= outcomes.keys(), "malformed",
             f"gate_outcomes missing {sorted(REQUIRED_OUTCOMES - outcomes.keys())}")
    for name, outcome in outcomes.items():
        _require(outcome.get("test_access") is (name == TEST_ACCESS_OUTCOME), "malformed",
                 f"gate_outcomes.{name}: only {TEST_ACCESS_OUTCOME} may grant TEST access")

    refusals = {item.get("id") for item in plan["refusal_conditions"]}
    _require(REQUIRED_REFUSALS <= refusals, "malformed",
             f"refusal_conditions missing {sorted(REQUIRED_REFUSALS - refusals)}")

    envelope = plan["test_seal"]["envelope"]
    receipt_rel = envelope["receipt_path"]
    _require(receipt_rel.startswith(ENVELOPE_DIR)
             and envelope["looks_dir"].startswith(ENVELOPE_DIR), "malformed",
             f"envelope files must live under {ENVELOPE_DIR}")
    _require(envelope["max_uses"] == 1 and plan["test"]["uses"] == 1, "envelope_already_used",
             "the TEST envelope is single-use: max_uses and test.uses must be 1")
    _require(envelope["state"] == "unused", "envelope_already_used",
             f"envelope state is {envelope['state']!r}")
    _require(not envelope_used(repo_root, receipt_rel), "envelope_already_used",
             f"{receipt_rel} exists or exists in git history")
    return plan, protocol


def _check_activation(plan, repo_root):
    activation = plan["activation"]
    _require(isinstance(activation, dict) and str(plan["status"]).startswith("ACTIVATED"),
             "plan_not_activated", "an activated plan needs status ACTIVATED and an activation")
    for key in ("activated_at", "approved_by", "activates"):
        _require(activation.get(key), "plan_not_activated", f"activation.{key} missing")
    source = activation["activates"]
    source_path = repo_root / source["path"]
    _require(source["path"] == str(PLAN) and source_path.is_file(), "plan_not_activated",
             f"activation.activates.path must be {PLAN}")
    _require(sha256_file(source_path) == source["sha256"], "plan_not_activated",
             "activation.activates.sha256 does not match the unactivated plan")
    unactivated = load_json(source_path)
    validate_plan(unactivated, repo_root)
    for key in (plan.keys() | unactivated.keys()) - ACTIVATION_KEYS:
        _require(plan.get(key) == unactivated.get(key), "plan_not_activated",
                 f"activated copy differs from the unactivated plan in {key}")


def _check_thresholds(restated, protocol):
    gates = protocol["substantial_gain_gates"]
    for block in ("substantial_rl", "named_frontier_comparison"):
        for key, value in restated[block].items():
            _require(gates[block].get(key) == value, "inherited_block_changed",
                     f"restated {block}.{key} {value} != protocol {gates[block].get(key)}")
    rule = gates["substantial_rl"]["validation_replicates"]
    _require(restated["replicate_mean_delta_minimum"] == 0.05 and "at least 0.05" in rule,
             "inherited_block_changed", "replicate mean threshold differs from the protocol")


def _check_test(test, protocol):
    _require(test["images"] == protocol["data"]["native_test_images"]
             and test["split"] == "native_test", "malformed",
             "TEST must be the full native test split")
    contenders = test["contenders"]
    roles = [c.get("role") for c in contenders]
    _require(sorted(roles) == sorted(TEST_ROLES) and len(roles) == len(set(roles)),
             "contender_not_allowlisted", f"TEST roles must be exactly {list(TEST_ROLES)}")
    _require("*" not in json.dumps(contenders), "contender_not_allowlisted",
             "a TEST identity contains a wildcard")
    frozen = protocol["activation"]["frozen_identities"]["strong_baseline"]
    (baseline,) = [c for c in contenders if c["role"] == "named_baseline_27b"]
    _require(baseline.get("repository") == frozen["repository"]
             and baseline.get("revision") == frozen["revision"], "contender_not_allowlisted",
             "named baseline identity differs from the activated protocol")


def _check_primary(primary, protocol):
    uncertainty = protocol["evaluation_and_seal"]["uncertainty"]
    expected = {"draws": uncertainty["bootstrap_draws"], "seed": uncertainty["seed"],
                "confidence_level": uncertainty["confidence_level"]}
    _require(primary.get("bootstrap") == expected, "inherited_block_changed",
             f"primary bootstrap {primary.get('bootstrap')} != protocol {expected}")


def _contains_test(evidence):
    found = []
    if "test" in evidence.get("rows", {}):
        found.append("rows.test")
    if "named_baseline_test" in evidence:
        found.append("named_baseline_test")
    found += [f"endpoints[{e.get('run_id')}].splits.test" for e in evidence.get("endpoints", [])
              if "test" in e.get("splits", {})]
    return found


def authorize(plan_path, evidence, repo_root=ROOT, now=None):
    """Write the single-use envelope receipt; return it. ``evidence`` is the parsed document."""
    repo_root = Path(repo_root)
    raw = Path(plan_path).read_bytes()
    plan, protocol = validate_plan(json.loads(raw), repo_root, require_activated=True)
    found = _contains_test(evidence)
    _require(not found, "evidence_contains_test", f"pre-TEST evidence holds {found}")
    receipt_branch = load_json(repo_root / plan["pins"]["branch_receipt"]["path"])
    _require(evidence.get("a_star") == receipt_branch["a_star"]
             and evidence.get("b4_edge_extension") == receipt_branch["b4_edge_extension"],
             "evidence_branch_mismatch", "evidence a_star / b4_edge_extension != branch receipt")

    evaluator = runpy.run_path(str(repo_root / plan["pins"]["gate_evaluator"]["path"]))
    try:
        report = evaluator["evaluate"](copy.deepcopy(evidence))
    except (KeyError, TypeError, ValueError) as error:
        raise Refusal("gates_not_passed",
                      f"gate evaluator refused the evidence: {error}") from error
    gates = report["gates"]
    passed = all(gates[g].get("status") == "evaluated" and gates[g].get("pass") is True
                 for g in ("G1", "G2", "G3a", "G3b", "G4", "G5"))
    _require(passed and report["failed_at"] is None and report["awaiting"] == "G6"
             and gates["G6"].get("status") == "awaiting_evidence", "gates_not_passed",
             f"failed_at={report['failed_at']} awaiting={report['awaiting']}")

    endpoints = {e["run_id"]: e for e in evidence["endpoints"]}
    (w,) = [e for e in evidence["endpoints"] if e["arm"] == "W"]
    frozen_w = protocol["activation"]["frozen_identities"]["initial_policy_w"]["adapter_sha256"]
    _require(w["adapter_sha256"] == frozen_w, "w_identity_mismatch",
             f"W adapter {w['adapter_sha256']} != frozen {frozen_w}")
    rl = endpoints[gates["G4"]["rl_star"]]
    _require(rl["arm"] == "RL" and rl["seed"] == plan["replication"]["primary_seed"]
             and rl["updates"] == plan["replication"]["N_accepted_updates"], "gates_not_passed",
             "RL* is not the seed-42 600-update RL endpoint")
    sft = endpoints[gates["G4"]["protocol"]["strongest_sft"]]
    (baseline,) = [c for c in plan["test"]["contenders"] if c["role"] == "named_baseline_27b"]

    def actor(role, endpoint):
        return {"role": role, "kind": "actor_adapter", "run_id": endpoint["run_id"],
                "adapter_sha256": endpoint["adapter_sha256"]}

    strongest = actor("strongest_sft_protocol_pool", sft)
    if sft["run_id"] == w["run_id"]:
        # One identity, one TEST run: both roles read W's single prediction file.
        strongest["shares_look_with"] = "initial_policy_w"
    contenders = [actor("rl_star_seed42", rl), strongest, actor("initial_policy_w", w),
                  {"role": "named_baseline_27b", "kind": "named_baseline",
                   "run_id": baseline["repository"],
                   "revision": baseline["revision"]}]
    envelope = plan["test_seal"]["envelope"]
    record = {
        "plan_id": PLAN_ID, "study_id": STUDY_ID,
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_sha256": canonical_sha256(evidence),
        "gate_report_sha256": canonical_sha256(report),
        "authorized_at": (now or datetime.now(UTC)).isoformat(),
        "max_uses": 1, "contenders": contenders,
        "looks_dir": envelope["looks_dir"],
        "rule": "Exactly one TEST look per role, claimed before any sealed file is opened.",
    }
    path = repo_root / envelope["receipt_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise Refusal("envelope_already_used", f"{path} already exists") from error
    return record


def claim(receipt_path, role, identity, repo_root=ROOT, now=None):
    """Record the one TEST look for ``role``; ``identity`` is adapter_sha256 or revision."""
    repo_root = Path(repo_root)
    receipt = load_json(receipt_path)
    matches = [c for c in receipt["contenders"] if c["role"] == role]
    _require(len(matches) == 1, "look_not_allowlisted", f"role {role!r} not in the envelope")
    (contender,) = matches
    _require("shares_look_with" not in contender, "look_not_allowlisted",
             f"{role} is evaluated through {contender.get('shares_look_with')}'s single look")
    expected = contender.get("adapter_sha256") or contender.get("revision")
    _require(identity == expected, "look_not_allowlisted",
             f"identity {identity!r} != envelope {expected!r} for {role}")
    look = repo_root / receipt["looks_dir"] / f"{role}.json"
    look.parent.mkdir(parents=True, exist_ok=True)
    try:
        with look.open("x", encoding="utf-8") as handle:
            json.dump({"role": role, "identity": identity, "run_id": contender["run_id"],
                       "claimed_at": (now or datetime.now(UTC)).isoformat()}, handle, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise Refusal("look_already_taken", f"{look} already exists") from error
    return look


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--plan", type=Path, default=ROOT / PLAN)
    grant = sub.add_parser("authorize")
    grant.add_argument("--plan", type=Path, default=ROOT / ACTIVATED_PLAN)
    grant.add_argument("--evidence", type=Path, required=True)
    look = sub.add_parser("claim")
    look.add_argument("--receipt", type=Path, required=True)
    look.add_argument("--role", required=True, choices=TEST_ROLES)
    look.add_argument("--identity", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            validate_plan(load_json(args.plan))
            print(f"valid: {args.plan} (envelope unused)")
        elif args.command == "authorize":
            record = authorize(args.plan, load_json(args.evidence))
            print(json.dumps(record["contenders"], indent=2))
        else:
            print(claim(args.receipt, args.role, args.identity))
    except Refusal as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    except (KeyError, TypeError, ValueError) as error:
        print(f"refused: malformed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
