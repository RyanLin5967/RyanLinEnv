"""The confirmatory pre-registration refuses malformed or already-used plans and guards TEST.

Expected values are the committed protocol, study and receipt files and the hand-computed gate
fixture of tests/test_evaluate_freshpool_gates.py, never the validator's own output. No TEST
data is read: the fixture's TEST rows are synthetic and are stripped before authorize.
"""

import copy
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(str(REPO / "scripts/validate_freshpool_confirmatory.py"))
validate_plan, authorize, claim, Refusal, main = (
    VALIDATOR[name] for name in ("validate_plan", "authorize", "claim", "Refusal", "main"))
GATE_FIXTURES = runpy.run_path(str(REPO / "tests/test_evaluate_freshpool_gates.py"))
PLAN_REL = "docs/aircraft-freshpool-confirmatory-v1.json"
ACTIVATED_REL = "docs/aircraft-freshpool-confirmatory-v1-activated.json"
RECEIPT_REL = "reports/aircraft-freshpool-confirmatory-v1/test-envelope.json"
# Copied from docs/aircraft-rl-protocol-freshpool-v1-activated.json frozen_identities.
FROZEN_W = "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"
BASELINE_REVISION = "815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@example.invalid",
                    "-c", "user.name=t", *args], check=True, capture_output=True, timeout=60)


@pytest.fixture
def repo(tmp_path):
    """A git repository holding exactly the plan and the files it pins."""
    plan = load(REPO / PLAN_REL)
    for rel in [PLAN_REL, *(pin["path"] for pin in plan["pins"].values())]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, target)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "--", "docs", "scripts", "src")
    git(tmp_path, "commit", "-q", "-m", "fixture")
    return tmp_path


def activate(root):
    source = root / PLAN_REL
    plan = load(source)
    plan.update(status="ACTIVATED 2026-10-18 (test fixture)", activated=True, activation={
        "activates": {"path": PLAN_REL,
                      "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "activated_at": "2026-10-18T00:00:00+00:00", "approved_by": "test"})
    (root / ACTIVATED_REL).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return root / ACTIVATED_REL


def pre_test_evidence():
    """The hand fixture (G1-G5 pass) with TEST removed and W renamed to the frozen adapter."""
    document = GATE_FIXTURES["evidence"]()
    del document["rows"]["test"]
    for endpoint in document["endpoints"]:
        endpoint["splits"].pop("test", None)
    text = json.dumps(document).replace('"w"', f'"{FROZEN_W}"')
    document = json.loads(text)
    document["a_star"] = load(REPO / "docs/aircraft-freshpool-branch-receipt-v1.json")["a_star"]
    return document


def refusal_id(call):
    with pytest.raises(Refusal) as caught:
        call()
    return caught.value.refusal_id


def test_committed_plan_is_valid_and_unactivated():
    plan = load(REPO / PLAN_REL)
    validate_plan(plan, REPO)
    assert plan["activated"] is False and plan["activation"] is None
    assert main(["validate", "--plan", str(REPO / PLAN_REL)]) == 0


def test_plan_seeds_and_thresholds_equal_the_activated_protocol():
    plan = load(REPO / PLAN_REL)
    protocol = load(REPO / "docs/aircraft-rl-protocol-freshpool-v1-activated.json")
    assert plan["replication"]["seeds"] == [314, 2718]
    assert protocol["exposure_and_staging"]["conditional_replication_seeds"] == [314, 2718]
    gates = protocol["substantial_gain_gates"]["substantial_rl"]
    assert gates["minimum_absolute_accuracy"] == 0.8
    assert plan["restated_thresholds"]["substantial_rl"]["minimum_absolute_accuracy"] == 0.8


def mutate(change):
    plan = load(REPO / PLAN_REL)
    change(plan)
    return plan


MALFORMED = [
    ("activated_without_record", lambda p: p.update(activated=True), "plan_not_activated"),
    ("status_word", lambda p: p.update(status="DRAFT"), "malformed"),
    ("missing_key", lambda p: p.pop("multiplicity"), "malformed"),
    ("missing_nested", lambda p: p["test_seal"].pop("envelope"), "malformed"),
    ("pin_sha", lambda p: p["pins"]["study"].update(sha256="0" * 64), "pin_mismatch"),
    ("evaluator_sha", lambda p: p["pins"]["gate_evaluator"].update(sha256="1" * 64),
     "pin_mismatch"),
    ("inherited_hash", lambda p: p["inherited_unchanged"].update(strong_baseline="2" * 64),
     "inherited_block_changed"),
    ("restated_margin", lambda p: p["restated_thresholds"]["substantial_rl"].update(
        minimum_delta_vs_strongest_sft=0.04), "inherited_block_changed"),
    ("bootstrap_draws", lambda p: p["primary_endpoint"]["bootstrap"].update(draws=10000),
     "inherited_block_changed"),
    ("seed_swap", lambda p: p["replication"].update(seeds=[314, 999]), "seeds_or_n_changed"),
    ("n_changed", lambda p: p["replication"].update(N_accepted_updates=400),
     "seeds_or_n_changed"),
    ("mode", lambda p: p["branch"].update(mode="mechanism"), "branch_mismatch"),
    ("extra_contender", lambda p: p["test"]["contenders"].append({"role": "rl_seed314"}),
     "contender_not_allowlisted"),
    ("duplicate_contender", lambda p: p["test"]["contenders"].__setitem__(
        2, dict(p["test"]["contenders"][0])), "contender_not_allowlisted"),
    ("wildcard", lambda p: p["test"]["contenders"][0].update(identity_source="*"),
     "contender_not_allowlisted"),
    ("baseline_revision", lambda p: p["test"]["contenders"][3].update(revision="main"),
     "contender_not_allowlisted"),
    ("partial_test", lambda p: p["test"].update(images=1000), "malformed"),
    ("multiplicity_order", lambda p: p["multiplicity"]["sequence"].reverse(), "malformed"),
    ("g5_fail_grants_test", lambda p: p["gate_outcomes"]["G4_pass_G5_fail"].update(
        test_access=True), "malformed"),
    ("missing_outcome", lambda p: p["gate_outcomes"].pop("G6a_fail"), "malformed"),
    ("missing_refusal", lambda p: p["refusal_conditions"].pop(), "malformed"),
    ("two_uses", lambda p: p["test_seal"]["envelope"].update(max_uses=2),
     "envelope_already_used"),
    ("state_used", lambda p: p["test_seal"]["envelope"].update(state="used"),
     "envelope_already_used"),
    ("receipt_outside", lambda p: p["test_seal"]["envelope"].update(
        receipt_path="runs/x.json"), "malformed"),
]


@pytest.mark.parametrize(("name", "change", "expected"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_each_malformation_is_refused_by_name(name, change, expected):
    assert refusal_id(lambda: validate_plan(mutate(change), REPO)) == expected


def test_cli_refuses_with_exit_code_two(tmp_path, capsys):
    bad = tmp_path / "plan.json"
    bad.write_text(json.dumps(mutate(lambda p: p["replication"].update(seeds=[1, 2]))))
    assert main(["validate", "--plan", str(bad)]) == 2
    assert "seeds_or_n_changed" in capsys.readouterr().err


def test_receipt_on_disk_or_in_history_means_already_used(repo):
    plan = load(repo / PLAN_REL)
    validate_plan(plan, repo)
    receipt = repo / RECEIPT_REL
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    assert refusal_id(lambda: validate_plan(plan, repo)) == "envelope_already_used"
    git(repo, "add", "--", RECEIPT_REL)
    git(repo, "commit", "-q", "-m", "receipt")
    git(repo, "rm", "-q", "--", RECEIPT_REL)
    git(repo, "commit", "-q", "-m", "delete receipt")
    assert not receipt.exists()
    assert refusal_id(lambda: validate_plan(plan, repo)) == "envelope_already_used"


def test_unreadable_history_refuses_instead_of_passing(repo):
    shutil.rmtree(repo / ".git")
    assert refusal_id(lambda: validate_plan(load(repo / PLAN_REL), repo)) == (
        "envelope_already_used")


def test_unactivated_plan_cannot_authorize(repo):
    assert refusal_id(lambda: authorize(repo / PLAN_REL, pre_test_evidence(), repo)) == (
        "plan_not_activated")
    assert not (repo / RECEIPT_REL).exists()


def test_activated_copy_must_match_the_draft(repo):
    path = activate(repo)
    plan = load(path)
    plan["replication"]["order"] = "seed 2718 first"
    path.write_text(json.dumps(plan))
    assert refusal_id(lambda: authorize(path, pre_test_evidence(), repo)) == (
        "plan_not_activated")


def test_authorize_names_exactly_the_hand_computed_contenders_once(repo):
    path = activate(repo)
    record = authorize(path, pre_test_evidence(), repo)
    # Hand fixture: RL* rl600-1e-5; VAL strongest SFT sfte-5e-5 (tie broken on active seconds).
    assert [(c["role"], c["run_id"]) for c in record["contenders"]] == [
        ("rl_star_seed42", "rl600-1e-5"), ("strongest_sft_protocol_pool", "sfte-5e-5"),
        ("initial_policy_w", FROZEN_W),
        ("named_baseline_27b", "mlx-community/Qwen3.8-27B-8bit")]
    assert record["contenders"][3]["revision"] == BASELINE_REVISION
    assert load(repo / RECEIPT_REL) == record
    assert refusal_id(lambda: authorize(path, pre_test_evidence(), repo)) == (
        "envelope_already_used")


def test_evidence_holding_test_content_is_refused(repo):
    path = activate(repo)
    for leak in (lambda d: d["rows"].update(test=[{"id": "x", "label": "A"}]),
                 lambda d: d["endpoints"][0]["splits"].update(test={"predictions": {}}),
                 lambda d: d.update(named_baseline_test={"predictions": {}})):
        document = pre_test_evidence()
        leak(document)
        assert refusal_id(lambda d=document: authorize(path, d, repo)) == (
            "evidence_contains_test")
    assert not (repo / RECEIPT_REL).exists()


def test_failed_or_missing_gates_write_no_receipt(repo):
    path = activate(repo)
    document = pre_test_evidence()
    find = GATE_FIXTURES["find"]
    greedy = GATE_FIXTURES["greedy"]
    # The G5 hand case: seed 2718 delta 0 fails "every seed positive".
    find(document, "SFT-E-5e-05-2718")["splits"]["val"]["predictions"] = greedy(
        {"A": 17, "B": 17})
    assert refusal_id(lambda: authorize(path, document, repo)) == "gates_not_passed"
    early = pre_test_evidence()
    early["endpoints"] = [e for e in early["endpoints"] if e["seed"] not in (314, 2718)]
    assert refusal_id(lambda: authorize(path, early, repo)) == "gates_not_passed"
    assert not (repo / RECEIPT_REL).exists()


def test_branch_and_w_identity_must_match(repo):
    path = activate(repo)
    document = pre_test_evidence()
    document["a_star"] = 0.75
    assert refusal_id(lambda: authorize(path, document, repo)) == "evidence_branch_mismatch"
    text = json.dumps(pre_test_evidence()).replace(FROZEN_W, "e" * 64)
    assert refusal_id(lambda: authorize(path, json.loads(text), repo)) == (
        "w_identity_mismatch")


def test_claim_is_an_allowlist_and_single_use(repo):
    record = authorize(activate(repo), pre_test_evidence(), repo)
    receipt = repo / RECEIPT_REL
    rl = record["contenders"][0]
    look = claim(receipt, "rl_star_seed42", rl["adapter_sha256"], repo)
    assert load(look)["run_id"] == "rl600-1e-5"
    assert refusal_id(lambda: claim(receipt, "rl_star_seed42", rl["adapter_sha256"], repo)) == (
        "look_already_taken")
    assert refusal_id(lambda: claim(receipt, "initial_policy_w", "f" * 64, repo)) == (
        "look_not_allowlisted")
    assert refusal_id(lambda: claim(receipt, "eg_sft", "x", repo)) == "look_not_allowlisted"
    assert claim(receipt, "named_baseline_27b", BASELINE_REVISION, repo).is_file()


def test_w_as_strongest_sft_shares_one_look(repo):
    path = activate(repo)
    document = pre_test_evidence()
    find = GATE_FIXTURES["find"]
    greedy = GATE_FIXTURES["greedy"]
    # W VAL .75 beats every SFT (best .7) yet RL* .9 still passes every point gate.
    find(document, FROZEN_W)["splits"]["val"]["predictions"] = greedy({"A": 15, "B": 15})
    for endpoint in document["endpoints"]:
        if endpoint["seed"] in (314, 2718) and endpoint["arm"] == "RL":
            endpoint["splits"]["val"]["predictions"] = greedy({"A": 19, "B": 19})
    report = GATE_FIXTURES["evaluate"](copy.deepcopy(document))
    if report["gates"]["G4"]["protocol"]["strongest_sft"] != FROZEN_W or report["failed_at"]:
        pytest.fail(f"fixture did not make W strongest with gates passing: {report['failed_at']}")
    record = authorize(path, document, repo)
    strongest = record["contenders"][1]
    assert strongest["run_id"] == FROZEN_W and strongest["shares_look_with"] == "initial_policy_w"
    assert refusal_id(lambda: claim(repo / RECEIPT_REL, "strongest_sft_protocol_pool",
                                    FROZEN_W, repo)) == "look_not_allowlisted"
    assert claim(repo / RECEIPT_REL, "initial_policy_w", FROZEN_W, repo).is_file()


def test_exclusive_create_holds_even_if_the_history_check_misses(repo, monkeypatch):
    """Two authorize calls racing past the validate-time check: the second create fails."""
    path = activate(repo)
    authorize(path, pre_test_evidence(), repo)
    # run_path returns a copy of the module globals; patch the ones authorize reads.
    monkeypatch.setitem(authorize.__globals__, "envelope_used", lambda *args: False)
    first = (repo / RECEIPT_REL).read_bytes()
    assert refusal_id(lambda: authorize(path, pre_test_evidence(), repo)) == (
        "envelope_already_used")
    assert (repo / RECEIPT_REL).read_bytes() == first
