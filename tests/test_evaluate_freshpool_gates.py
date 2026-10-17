import copy
import json
import runpy
from pathlib import Path

import pytest

from vision_rl.classification_comparison import compare_repeated_classification

GATES = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/evaluate_freshpool_gates.py")
)
evaluate, d_req, within, gate_g1 = (
    GATES[name] for name in ("evaluate", "d_req", "within", "gate_g1")
)

LABELS = ["A", "B"]
OTHER = {"A": "B", "B": "A"}
# Two images per class on each H split, twenty per class on VAL and TEST.
H_SELECT = [("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")]
H_GATE = [("c1", "A"), ("c2", "A"), ("d1", "B"), ("d2", "B")]
FULL = [(f"{label}{index:02d}", label) for label in LABELS for index in range(20)]


def repeated(split, counts, completions, invalid=0):
    """Image i gets counts[i] correct answers, then `invalid` Nones, then wrong labels."""
    out = {}
    for (identifier, label), correct in zip(split, counts, strict=True):
        answers = [label] * correct + [None] * invalid
        out[identifier] = answers + [OTHER[label]] * (completions - len(answers))
    return out


def greedy(correct_per_class):
    """The first n images of each class are correct, so smaller sets are nested subsets."""
    return {identifier: label if int(identifier[1:]) < correct_per_class[label] else OTHER[label]
            for identifier, label in FULL}


def endpoint(run_id, arm, lr, seed, updates, active, h_select=None, h_gate=None, val=None,
             test=None, **extra):
    splits = {}
    if h_select is not None:
        splits["h_select"] = {"predictions": repeated(H_SELECT, h_select, 5),
                              "eos_fraction": 1.0}
    if h_gate is not None:
        splits["h_gate"] = {"predictions": repeated(H_GATE, h_gate, 9), "eos_fraction": 1.0}
    if val is not None:
        splits["val"] = {"predictions": greedy(dict(zip(LABELS, val, strict=True)))}
    if test is not None:
        splits["test"] = {"predictions": greedy(dict(zip(LABELS, test, strict=True)))}
    return {"run_id": run_id, "arm": arm, "lr": lr, "seed": seed, "updates": updates,
            "committed_active_seconds": active, "adapter_sha256": run_id,
            "reference_sha256": "w", "splits": splits, **extra}


def journal():
    # p=.5: u = 1 - 2/16 = .875, accepted. p=.95: u = 1 - .81450625 - .00000625, rejected.
    entries = []
    for index in range(400):
        p = 0.5 if index < 200 else 0.95
        entries.append({"id": f"s{index:03d}", "p": p, "u": 1 - p**4 - (1 - p) ** 4,
                        "accepted": p == 0.5, "scorer_seconds": 1.0})
    return entries


def rollouts(mixed_groups=112):
    """64 accepted images x seeds 42/314 = 128 groups; the first `mixed_groups` are mixed."""
    groups = []
    for index in range(64):
        for seed in (42, 314):
            mixed = len(groups) < mixed_groups
            outcomes = [True, False, False, False] if mixed else [False] * 4
            groups.append({"id": f"s{index:03d}", "seed": seed, "completions": [
                {"correct": c, "valid": True, "eos": True, "canonical_tokenization": True}
                for c in outcomes]})
    return groups


def evidence():
    """Hand-computed H-select / H-gate / VAL / TEST outcomes; values are in comments."""
    w = endpoint("w", "W", None, None, 0, 0.0, h_select=[1, 1, 1, 1], h_gate=[2, 2, 2, 2],
                 val=[10, 10], test=[10, 10])  # H-select .2, H-gate 2/9, VAL .5
    flags = [True] * 50 + [False] * 30 + [True] * 70  # trailing 100: 70 mixed -> .70
    endpoints = [
        w,
        # RL at 150 on H-select: .4, .6, .2 -> 5e-5 then 1e-5 continue; 1e-4 stops.
        endpoint("rl150-1e-5", "RL", 1e-5, 42, 150, 3300, h_select=[2, 2, 2, 2],
                 useful_mixed=flags),
        endpoint("rl150-5e-5", "RL", 5e-5, 42, 150, 3300, h_select=[3, 3, 3, 3],
                 useful_mixed=flags),
        endpoint("rl150-1e-4", "RL", 1e-4, 42, 150, 3300, h_select=[1, 1, 1, 1],
                 useful_mixed=flags),
        # At 600 the order flips: 1e-5 .8 beats 5e-5 .6, so RL* = 1e-5.
        endpoint("rl600-1e-5", "RL", 1e-5, 42, 600, 13338, h_select=[4, 4, 4, 4],
                 h_gate=[7, 7, 7, 7], val=[18, 18], test=[18, 18]),
        endpoint("rl600-5e-5", "RL", 5e-5, 42, 600, 13338, h_select=[3, 3, 3, 3]),
        # SFT-E on H-select: .4, .7, .7 -> tie, 5e-5 wins on lower active seconds;
        # both tied LRs are within 2 pp, so SFT-T runs at 5e-5 and 1e-4.
        endpoint("sfte-1e-5", "SFT-E", 1e-5, 42, 600, 3384, h_select=[2, 2, 2, 2],
                 val=[12, 12]),
        endpoint("sfte-5e-5", "SFT-E", 5e-5, 42, 600, 3384, h_select=[4, 3, 4, 3],
                 h_gate=[5, 5, 5, 5], val=[14, 14], test=[14, 14]),
        endpoint("sfte-1e-4", "SFT-E", 1e-4, 42, 600, 3400, h_select=[4, 3, 3, 4],
                 val=[14, 14]),
        # SFT-T on H-select: .75 beats .7. H-gate of SFT-T*: (7+7)/18, (7+6)/18.
        endpoint("sftt-5e-5", "SFT-T", 5e-5, 42, None, 13338, h_select=[4, 4, 4, 3],
                 h_gate=[7, 7, 7, 6], val=[14, 14]),
        endpoint("sftt-1e-4", "SFT-T", 1e-4, 42, None, 13338, h_select=[4, 3, 4, 3],
                 val=[13, 13]),
        endpoint("c400-sft", "controlled400-SFT", 1e-4, None, 400, 2000, val=[13, 13]),
        endpoint("egsft", "EG-SFT", 5e-5, 42, 600, 3384, val=[17, 17]),
    ]
    # Replicates: seed 314 RL .85 vs best SFT .75 (+.10); seed 2718 RL .85 vs .80 (+.05).
    for seed, rl, sft in ((314, 17, 15), (2718, 17, 16)):
        endpoints.append(endpoint(f"rl-{seed}", "RL", 1e-5, seed, 600, 13338, val=[rl, rl]))
        for arm in ("SFT-E", "SFT-T"):
            for lr, correct in ((5e-5, sft), (1e-4, sft - 1)):
                endpoints.append(endpoint(f"{arm}-{lr}-{seed}", arm, lr, seed, 600, 3384,
                                          val=[correct, correct]))
    return {
        "labels": LABELS,
        "rows": {name: [{"id": i, "label": label} for i, label in split]
                 for name, split in (("h_select", H_SELECT), ("h_gate", H_GATE),
                                     ("val", FULL), ("test", FULL))},
        "a_star": 0.75,
        "b4_edge_extension": False,
        "g1": {"journal": journal(), "rollouts": rollouts()},
        "endpoints": endpoints,
    }


def find(document, run_id):
    return next(e for e in document["endpoints"] if e["run_id"] == run_id)


@pytest.mark.parametrize(("a_star", "required", "mode"), [
    (0.90, 0.05, "full"), (0.75, 0.05, "full"),
    (0.70, 0.10, "full"),  # boundary: D_req > 0.10 is strict, so 0.70 stays full mode
    (0.65, 0.15, "mechanism"), (0.20, 0.60, "mechanism"),
])
def test_required_delta_and_branch_mode_match_hand_values(a_star, required, mode):
    result = d_req(a_star)
    assert result["d_req"] == pytest.approx(required)
    assert result["mode"] == mode
    assert result["g3_margin"] == (0.05 if mode == "mechanism" else 0.02)


def test_within_two_points_survives_float_rounding_at_the_boundary():
    candidates = [{"run_id": r, "committed_active_seconds": 1, "macro": m}
                  for r, m in (("x", 0.70), ("y", 0.68), ("z", 0.6799))]
    # 0.70 - 0.68 is 0.020000000000000018 in floating point; still exactly 2 pp.
    window = GATES["SFT_T_WINDOW"]
    assert [c["run_id"] for c in within(candidates, "macro", window)] == ["x", "y"]


def test_repeated_comparison_keeps_completions_of_one_image_together():
    rows = [{"id": i, "label": label} for i, label in H_GATE]
    # Same +5/9 on every image: every bootstrap draw equals the point value.
    flat = compare_repeated_classification(
        rows, repeated(H_GATE, [7] * 4, 9), repeated(H_GATE, [2] * 4, 9), LABELS,
        completions=9, lower_quantile=0.10)
    assert flat["mean_class_accuracy_difference"] == pytest.approx(5 / 9)
    assert flat["uncertainty"]["lower_bound"] == pytest.approx(5 / 9)
    # Per class one image +1, one -1. Class means are -1, 0, 1 w.p. 1/4, 1/2, 1/4;
    # the two-class mean is <= -0.5 w.p. 5/16 > 0.10, and -1 w.p. 1/16 < 0.10.
    split = compare_repeated_classification(
        rows, repeated(H_GATE, [9, 0, 9, 0], 9), repeated(H_GATE, [0, 9, 0, 9], 9), LABELS,
        completions=9, lower_quantile=0.10)
    assert split["mean_class_accuracy_difference"] == 0
    assert split["uncertainty"]["lower_bound"] == -0.5
    missing = repeated(H_GATE, [9, 9, 9, 9], 9)
    del missing["c1"]
    scored = compare_repeated_classification(
        rows, missing, missing, LABELS, completions=9)["candidate"]
    assert scored["invalid_completions_including_missing"] == 9
    assert scored["mean_class_accuracy"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="declared completions"):
        compare_repeated_classification(rows, {"c1": ["A"] * 8}, {}, LABELS, completions=9)


def test_g1_hand_fixture_passes_and_each_threshold_fires():
    result = gate_g1({"journal": journal(), "rollouts": rollouts()})
    assert result["pass"]
    assert result["values"]["acceptance"] == 0.5
    assert result["values"]["mean_u_accepted"] == 0.875
    assert result["values"]["observed_useful_mixed"] == 0.875  # 112 / 128
    assert result["values"]["predicted_useful_mixed"] == 0.875
    # 0.875 - 80/128 = 0.25 > 0.10: miscalibrated screen.
    assert not gate_g1({"journal": journal(), "rollouts": rollouts(80)})["checks"][
        "useful_mixed_calibrated"]
    low = journal()
    for entry in low[100:200]:
        entry.update(p=0.95, u=1 - 0.95**4 - 0.05**4, accepted=False)
    assert gate_g1({"journal": low, "rollouts": rollouts()})["values"]["acceptance"] == 0.25
    slow = journal()
    slow[0]["scorer_seconds"] = 400.0  # mean (399 + 400) / 400 = 1.9975 s: still passes
    assert gate_g1({"journal": slow, "rollouts": rollouts()})["checks"]["scorer_seconds"]
    slow[1]["scorer_seconds"] = 3.0  # mean 2.0025 s > 2.0
    assert not gate_g1({"journal": slow, "rollouts": rollouts()})["checks"]["scorer_seconds"]
    forged = journal()
    forged[300]["accepted"] = True
    with pytest.raises(ValueError, match="disagrees"):
        gate_g1({"journal": forged, "rollouts": rollouts()})


def test_full_hand_fixture_selects_and_gates_as_computed_by_hand():
    report = evaluate(evidence())
    gates = report["gates"]
    assert report["failed_at"] is None and report["awaiting"] is None
    assert gates["G2"]["continue_to_600"] == ["rl150-5e-5", "rl150-1e-5"]
    assert gates["G2"]["values"]["h_select_delta_vs_w"] == pytest.approx(0.4)
    assert gates["G2"]["values"]["trailing_useful_mixed"] == pytest.approx(0.7)
    assert gates["G3a"]["rl_star"] == "rl600-1e-5"
    assert gates["G3a"]["sft_e_star"] == "sfte-5e-5"  # tie at .7, fewer active seconds
    assert gates["G3a"]["sft_t_lrs"] == [5e-5, 1e-4]
    assert gates["G3a"]["values"]["rl_minus_sft_e"] == pytest.approx(2 / 9)
    assert gates["G3a"]["values"]["rl_minus_w_lb90"] == pytest.approx(5 / 9)
    assert gates["G3b"]["sft_t_star"] == "sftt-5e-5"
    assert gates["G3b"]["values"]["rl_minus_sft_t"] == pytest.approx(1 / 36)
    protocol, robust = gates["G4"]["protocol"], gates["G4"]["robust"]
    # VAL tie at .7 among sfte-5e-5 (3384 s), sfte-1e-4 (3400 s), sftt-5e-5 (13338 s).
    assert protocol["strongest_sft"] == "sfte-5e-5"
    assert protocol["values"]["rl_accuracy"] == pytest.approx(0.9)
    assert protocol["values"]["delta_vs_w"] == pytest.approx(0.4)
    assert protocol["values"]["delta_vs_strongest_sft"] == pytest.approx(0.2)
    assert protocol["values"]["relative_error_reduction"] == pytest.approx(2 / 3)
    assert protocol["pass"]
    # EG-SFT .85: point gates hold (+.05, RER 1/3), but each class differs on one image of
    # twenty; both class draws are zero w.p. .95**40 = .129 > .025, so the 95% LB is 0.
    assert robust["strongest_sft"] == "egsft"
    assert robust["values"]["delta_vs_strongest_sft"] == pytest.approx(0.05)
    assert robust["values"]["paired_95_ci"][0] == 0
    assert robust["checks"] == {
        "accuracy_at_least_0_80": True, "delta_vs_w_at_least_0_05": True,
        "delta_vs_strongest_sft_at_least_0_05": True,
        "relative_error_reduction_at_least_0_20": True, "paired_95_lb_positive": False}
    assert gates["G4"]["statement"] is None
    assert gates["G5"]["deltas"] == pytest.approx({"42": 0.2, "314": 0.1, "2718": 0.05})
    assert gates["G5"]["mean"] == pytest.approx(0.35 / 3)
    assert gates["G5"]["pass"]
    assert gates["G6"]["contenders"] == ["rl600-1e-5", "sfte-5e-5", "w"]
    assert gates["G6"]["substantial_rl_test"]["pass"]
    assert gates["G6"]["named_baseline"]["status"] == "incomplete"
    assert report["substantial_val_and_replication"] and report["final_test_gates"]
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_g2_futility_fires_on_each_condition():
    for change, check in (
        (lambda d: find(d, "rl150-5e-5").update(adapter_sha256="w"), "adapter_changed"),
        (lambda d: find(d, "rl150-5e-5").update(reference_sha256="drift"),
         "reference_unchanged"),
        # Trailing 100 hold 41 unmixed then 59 mixed: .59 < .60
        (lambda d: find(d, "rl150-5e-5").update(
            useful_mixed=[True] * 50 + [False] * 41 + [True] * 59),
         "rolling_useful_mixed"),
    ):
        document = evidence()
        change(document)
        report = evaluate(document)
        assert report["failed_at"] == "G2"
        assert not report["gates"]["G2"]["checks"][check]
        assert report["gates"]["G3a"]["status"] == "not_evaluated"
        assert "G4" in report["evidence_after_stop"]
    document = evidence()
    # Three invalid answers per image (12 total) vs W's 0: over the +2 slack.
    find(document, "rl150-5e-5")["splits"]["h_select"]["predictions"] = repeated(
        H_SELECT, [2, 2, 2, 2], 5, invalid=3)
    for run_id in ("rl150-1e-5", "rl150-1e-4"):
        find(document, run_id)["splits"]["h_select"]["predictions"] = repeated(
            H_SELECT, [1, 1, 1, 1], 5)
    gate = evaluate(document)["gates"]["G2"]
    assert not gate["checks"]["invalid_within_w_plus_2"]
    # Every RL LR equal to W on H-select: delta 0 is not > 0.
    for run_id in ("rl150-1e-5", "rl150-5e-5", "rl150-1e-4"):
        find(document, run_id)["splits"]["h_select"]["predictions"] = repeated(
            H_SELECT, [1, 1, 1, 1], 5)
    assert not evaluate(document)["gates"]["G2"]["checks"]["h_select_delta_vs_w_positive"]


def test_mechanism_mode_tightens_g3b_from_two_to_five_points():
    document = evidence()
    document["a_star"] = 0.65  # D_req .15 > .10
    report = evaluate(document)
    assert report["d_req"]["mode"] == "mechanism"
    assert report["gates"]["G3a"]["pass"]  # 2/9 >= .05
    assert report["failed_at"] == "G3b"  # 1/36 < .05
    assert report["gates"]["G4"]["status"] == "not_evaluated"


def test_g3a_stops_on_small_margin_or_uncertain_gain_over_w():
    document = evidence()
    # SFT-E* H-gate equal to RL* (7 of 9 everywhere): lead 0 < .02.
    find(document, "sfte-5e-5")["splits"]["h_gate"]["predictions"] = repeated(H_GATE, [7] * 4, 9)
    report = evaluate(document)
    assert report["failed_at"] == "G3a"
    assert not report["gates"]["G3a"]["checks"]["rl_minus_sft_e_at_least_margin"]
    document = evidence()
    # RL* vs W: per class one image +1 and one -1, so the point delta is 0 and the
    # one-sided 90% bound is -0.5 (hand derivation in the repeated-comparison test).
    find(document, "rl600-1e-5")["splits"]["h_gate"]["predictions"] = repeated(
        H_GATE, [9, 0, 9, 0], 9)
    find(document, "w")["splits"]["h_gate"]["predictions"] = repeated(H_GATE, [0, 9, 0, 9], 9)
    find(document, "sfte-5e-5")["splits"]["h_gate"]["predictions"] = repeated(
        H_GATE, [0, 0, 0, 0], 9)
    gate = evaluate(document)["gates"]["G3a"]
    assert gate["values"]["rl_minus_w"] == 0
    assert gate["values"]["rl_minus_w_lb90"] == -0.5
    assert not gate["checks"]["rl_minus_w_one_sided_90_lb_positive"]


def test_g4_protocol_failure_stops_replication_and_test():
    document = evidence()
    find(document, "sftt-5e-5")["splits"]["val"]["predictions"] = greedy({"A": 17, "B": 17})
    report = evaluate(document)
    assert report["gates"]["G4"]["protocol"]["strongest_sft"] == "sftt-5e-5"
    assert report["failed_at"] == "G4"
    assert report["evidence_after_stop"] == ["G5", "G6"]
    assert not report["substantial_val_and_replication"]


def test_eg_sft_matching_rl_is_stated_as_no_evidence_beyond_reweighting():
    document = evidence()
    find(document, "egsft")["splits"]["val"]["predictions"] = greedy({"A": 18, "B": 18})
    gate = evaluate(document)["gates"]["G4"]
    assert gate["protocol"]["pass"] and not gate["robust"]["pass"]
    assert gate["statement"] == "no evidence RL adds anything beyond reweighting"


def test_g5_needs_every_seed_positive_not_only_the_mean():
    document = evidence()
    # Seed 2718: RL .85 vs best SFT .85 -> delta 0; mean (.2 + .1 + 0)/3 = .1 >= .05.
    find(document, "SFT-E-5e-05-2718")["splits"]["val"]["predictions"] = greedy(
        {"A": 17, "B": 17})
    report = evaluate(document)
    assert report["gates"]["G5"]["mean"] == pytest.approx(0.1)
    assert report["gates"]["G5"]["checks"] == {
        "every_seed_delta_positive": False, "three_seed_mean_at_least_0_05": True}
    assert report["failed_at"] == "G5"
    assert report["gates"]["G6"]["status"] == "not_evaluated"


def test_named_baseline_gates_on_test():
    document = evidence()
    baseline = {"run_id": "27b", "predictions": greedy({"A": 14, "B": 14}),
                "valid_eos_fraction": 0.96}
    document["named_baseline_test"] = baseline
    frontier = evaluate(document)["gates"]["G6"]["named_baseline"]
    assert frontier["values"]["delta"] == pytest.approx(0.2)
    assert frontier["pass"]  # +.2, RER 2/3, LB > .05 for 4-of-20 nested gains per class
    document["named_baseline_test"] = {**baseline, "valid_eos_fraction": 0.94}
    frontier = evaluate(document)["gates"]["G6"]["named_baseline"]
    assert frontier["status"] == "interface_failure" and not frontier["pass"]


def test_missing_or_mismatched_evidence_refuses_instead_of_passing():
    document = evidence()
    del document["b4_edge_extension"]
    with pytest.raises(ValueError, match="b4_edge_extension"):
        evaluate(document)
    document = evidence()
    document["b4_edge_extension"] = True  # grid now needs 2e-4 in every family
    with pytest.raises(ValueError, match="exactly one endpoint per LR"):
        evaluate(document)
    document = evidence()
    find(document, "egsft")["lr"] = 1e-4  # EG-SFT must use the SFT-E* LR
    with pytest.raises(ValueError, match="EG-SFT"):
        evaluate(document)
    document = evidence()
    find(document, "rl-314")["lr"] = 5e-5  # replicates freeze RL*'s LR
    with pytest.raises(ValueError, match="RL at seed 314"):
        evaluate(document)
    document = evidence()
    del find(document, "sfte-5e-5")["splits"]["test"]
    with pytest.raises(ValueError, match="no test predictions"):
        evaluate(document)


def test_waiting_for_evidence_is_not_a_pass():
    document = evidence()
    document["endpoints"] = [e for e in document["endpoints"] if e["arm"] != "EG-SFT"]
    report = evaluate(document)
    assert report["awaiting"] == "G4" and report["failed_at"] is None
    assert report["gates"]["G4"] == {"status": "awaiting_evidence"}
    assert not report["substantial_val_and_replication"]
    early = copy.deepcopy(evidence())
    del early["g1"]
    assert evaluate(early)["awaiting"] == "G1"


def test_robust_point_gate_fails_below_five_points():
    document = evidence()
    # EG-SFT (18 + 17) / 40 = .875; RL* .9 leads by .025 < .05.
    find(document, "egsft")["splits"]["val"]["predictions"] = greedy({"A": 18, "B": 17})
    robust = evaluate(document)["gates"]["G4"]["robust"]
    assert robust["values"]["delta_vs_strongest_sft"] == pytest.approx(0.025)
    assert not robust["checks"]["delta_vs_strongest_sft_at_least_0_05"]
