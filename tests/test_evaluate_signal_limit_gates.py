"""Gate arithmetic for aircraft-rl-next-study-v1, against hand-computed fixtures.

Every expected value below is written out by hand (in comments) or restated from the
study's formulas with literal arithmetic; none is read back from the subject.
"""

import copy
import json
import math
import random
import runpy
from pathlib import Path

import pytest

from vision_rl.classification_comparison import compare_repeated_classification

ROOT = Path(__file__).resolve().parents[1]
GATES = runpy.run_path(str(ROOT / "scripts/evaluate_signal_limit_gates.py"))
FRESH = runpy.run_path(str(ROOT / "scripts/evaluate_freshpool_gates.py"))
evaluate, a_floor, u_g, gate_pi, at_least, p0_values, small_n_lower_bound = (
    GATES[name] for name in ("evaluate", "a_floor", "u_g", "gate_pi", "at_least", "p0_values",
                             "small_n_lower_bound"))
W_ADAPTER, I_ADAPTER = GATES["W_ADAPTER"], GATES["I_ADAPTER"]

LABELS = [f"C{i}" for i in range(10)]
OTHER = {label: LABELS[(i + 1) % 10] for i, label in enumerate(LABELS)}
# Two images per class on each H split (20 images), twenty per class on VAL and TEST.
H_SELECT = [(f"hs{label}{j}", label) for label in LABELS for j in range(2)]
H_GATE = [(f"hg{label}{j}", label) for label in LABELS for j in range(2)]
FULL = [(f"v{label}_{j:02d}", label) for label in LABELS for j in range(20)]
# S_b: 987 IDs (the preview size), scan order random.Random(42).shuffle(sorted(ids)).
S_B = [f"s{index:03d}" for index in range(987)]
SCAN = sorted(S_B)
random.Random(42).shuffle(SCAN)
C400 = [f"sft-{lr}-{match}" for lr in ("1e5", "5e5", "1e4") for match in ("exposure", "time")]
I_STAR = "a2-1e-4"
RL_EXTRA = {"reference_sha256": I_STAR, "loss_type": "dr_grpo", "scale_rewards": "none",
            "beta": 0, "group": 16, "reward": "exact"}


def spread(total):
    """`total` correct completions over the 20 images: 20 x (total // 20), the remainder
    one each on the first images. H-select macro is total / 100, H-gate total / 180."""
    return [total // 20 + (i < total % 20) for i in range(20)]


def repeated(split, counts, completions, invalid=()):
    """Image i gets counts[i] correct answers; images listed in `invalid` get one None."""
    out = {}
    for index, ((identifier, label), correct) in enumerate(zip(split, counts, strict=True)):
        answers = [label] * correct + [None] * (index in invalid)
        out[identifier] = answers + [OTHER[label]] * (completions - len(answers))
    return out


def greedy(correct, split=FULL):
    """The first `correct` images of every class are right, so accuracy is correct / 20."""
    return {i: label if int(i.split("_")[1]) < correct else OTHER[label] for i, label in split}


def constrained(per_class):
    """Constrained greedy on H-select: the first `per_class` of each class's 2 images right."""
    return {i: label if int(i[-1]) < per_class else OTHER[label] for i, label in H_SELECT}


def endpoint(run_id, arm, lr, seed, updates, active, *, init, hs=None, hg=None, val=None,
             test=None, cg=None, adapter=None, **extra):
    splits = {}
    if hs is not None:
        counts = spread(hs) if isinstance(hs, int) else hs
        splits["h_select"] = {"predictions": repeated(H_SELECT, counts, 5), "eos_fraction": 1.0}
    if hg is not None:
        splits["h_gate"] = {"predictions": repeated(H_GATE, spread(hg), 9), "eos_fraction": 1.0}
    if cg is not None:
        splits["h_select_constrained"] = {"predictions": constrained(cg), "trie_sha256": "t"}
    if val is not None:
        splits["val"] = {"predictions": greedy(val)}
    if test is not None:
        splits["test"] = {"predictions": greedy(test)}
    out = {"run_id": run_id, "arm": arm, "lr": lr, "seed": seed, "updates": updates,
           "committed_active_seconds": active, "adapter_sha256": adapter or run_id,
           "splits": splits, **extra}
    if init is not None:
        out["init_sha256"] = init
    return out


def p0_doc(adapter, ps, group, mixed_groups, journal_group=4):
    journal = [{"id": SCAN[i], "p": p,
                "u": 1 - p**journal_group - (1 - p) ** journal_group,
                "accepted": 1 - p**journal_group - (1 - p) ** journal_group >= 0.5,
                "scorer_seconds": 1.0, "adapter_sha256": adapter} for i, p in enumerate(ps)]
    rollouts = []
    for index in range(64):
        for seed in (42, 314):
            mixed = len(rollouts) < mixed_groups
            outcomes = [mixed] + [False] * (group - 1)
            rollouts.append({"id": SCAN[index], "seed": seed, "completions": [
                {"correct": c, "valid": True, "eos": True, "canonical_tokenization": True,
                 "label_only": True} for c in outcomes]})
    return {"seed": 42, "adapter_sha256": adapter, "scan_order": list(SCAN),
            "journal": journal, "rollouts": rollouts}


# P0-I: 100 x p=.5 (u >= .5 at every G), 52 x p=.9 (u4 = .3438 rejected; u8 = .5695 and
# u16 = .8147 accepted), 248 x p=.999 (u16 = 1 - .999**16 = .0159, rejected everywhere).
PI_PS = [0.5] * 100 + [0.9] * 52 + [0.999] * 248
# P0' at I*: 100 x .5, 40 x .9, 20 x .95 (u8 = .3366 rejected, u16 = .5599 accepted),
# 240 x .999. G8 accepts 140/400 = .35 < 375/987; G16 accepts 160/400 = .40.
PRIME_PS = [0.5] * 100 + [0.9] * 40 + [0.95] * 20 + [0.999] * 240


def evidence():
    sa = dict(seed=42, updates=600, active=2000.0, init=W_ADAPTER, cg=1)
    stage_a = [
        # A1 H-select .20, .40, .39 -> A1* = 5e-5; 1e-4 is within 2 pp and joins the pool.
        endpoint("a1-1e-5", "A1", 1e-5, hs=20, val=11, **sa),
        endpoint("a1-5e-5", "A1", 5e-5, hs=40, val=12, **sa),
        endpoint("a1-1e-4", "A1", 1e-4, hs=39, val=12, **sa),
        # A2 .40, .20, .60 -> A2* = 1e-4, the winner and I* (constrained greedy 2 per class).
        endpoint("a2-1e-5", "A2", 1e-5, hs=40, **sa),
        endpoint("a2-5e-5", "A2", 5e-5, hs=20, **sa),
        endpoint(I_STAR, "A2", 1e-4, **{**sa, "cg": 2}, hs=60, hg=80, val=14),
        # A3 .20, .39, .20 -> A3* = 5e-5: a comparator (C* family) and a pool member.
        endpoint("a3-1e-5", "A3", 1e-5, hs=20, **sa),
        endpoint("a3-5e-5", "A3", 5e-5, hs=39, val=15, **sa),
        endpoint("a3-1e-4", "A3", 1e-4, hs=20, **sa),
        endpoint("a4-5e-5", "A4", 5e-5, hs=30, **sa),  # A1*'s LR; reported control
        # A5 .20, .37, .20 -> A5* = 5e-5, C* = max(A1* .40, A3* .39, A5* .37) = A1*.
        endpoint("a5-1e-5", "A5", 1e-5, hs=20, **sa),
        endpoint("a5-5e-5", "A5", 5e-5, hs=37, val=13, **sa),
        endpoint("a5-1e-4", "A5", 1e-4, hs=20, **sa),
    ]
    flags = [False] * 30 + [True] * 70  # 70 of 100 mixed -> .70
    rl = dict(seed=42, init=I_STAR, **RL_EXTRA)
    stage_b = [
        # RL at 100 on H-select: .65, .70, .50 -> 5e-5 then 1e-5 continue.
        endpoint("rl100-1e-5", "RL", 1e-5, updates=100, active=4000.0, hs=65,
                 useful_mixed=flags, **rl),
        endpoint("rl100-5e-5", "RL", 5e-5, updates=100, active=4000.0, hs=70,
                 useful_mixed=flags, **rl),
        endpoint("rl100-1e-4", "RL", 1e-4, updates=100, active=4000.0, hs=50,
                 useful_mixed=flags, **rl),
        # At 300: 1e-5 .80 beats 5e-5 .70 -> RL* = 1e-5. T_RL = 13000 s.
        endpoint("rl300-1e-5", "RL", 1e-5, updates=300, active=13000.0, hs=80, hg=140,
                 val=18, test=18, **rl),
        endpoint("rl300-5e-5", "RL", 5e-5, updates=300, active=13000.0, hs=70, **rl),
        # SFT-E .62, .70, .69 -> SFT-E* = 5e-5; 5e-5 and 1e-4 are within 2 pp.
        endpoint("sfte-1e-5", "SFT-E", 1e-5, 42, 300, 1000.0, init=I_STAR, hs=62, val=12),
        endpoint("sfte-5e-5", "SFT-E", 5e-5, 42, 300, 1000.0, init=I_STAR, hs=70, hg=100,
                 val=14),
        endpoint("sfte-1e-4", "SFT-E", 1e-4, 42, 300, 1000.0, init=I_STAR, hs=69, val=14),
        # SFT-T .72, .70 -> SFT-T* = 5e-5; both reach T_RL.
        endpoint("sftt-5e-5", "SFT-T", 5e-5, 42, 3900, 13000.0, init="sfte-5e-5", hs=72,
                 hg=120, val=14),
        endpoint("sftt-1e-4", "SFT-T", 1e-4, 42, 3900, 13050.0, init="sfte-1e-4", hs=70,
                 val=13),
        # LIM-T at the winner's Stage-A LR (A2*, 1e-4).
        endpoint("limt", "LIM-T", 1e-4, 42, 3700, 13010.0, init=I_STAR, hg=100, val=16,
                 test=16),
    ]
    endpoints = [endpoint("w", "W", None, None, 0, 0.0, init=None, adapter=W_ADAPTER, val=10,
                          test=10), *stage_a, *stage_b]
    for index, run_id in enumerate(C400):  # VAL .55-.65, below every pool maximum
        endpoints.append(endpoint(run_id, "controlled400-SFT", 1e-4, None, 400, 500.0 + index,
                                  init=None, val=11 + index % 3,
                                  adapter=I_ADAPTER if run_id == "sft-1e5-time" else None))
    # Replicates: seed 314 RL .85 vs SFT max .75 (+.10); seed 2718 RL .85 vs .80 (+.05).
    # Each seed's LIM-T (at A2*'s LR 1e-4) sits .05 under that seed's SFT maximum.
    for seed, sft in ((314, 15), (2718, 16)):
        endpoints.append(endpoint(f"rl-{seed}", "RL", 1e-5, seed, 300, 13000.0, init=I_STAR,
                                  val=17, **RL_EXTRA))
        endpoints.append(endpoint(f"limt-{seed}", "LIM-T", 1e-4, seed, 3700, 13010.0,
                                  init=I_STAR, val=sft - 1))
        for lr, correct in ((5e-5, sft), (1e-4, sft - 1)):
            endpoints.append(endpoint(f"sfte-{lr}-{seed}", "SFT-E", lr, seed, 300, 1000.0,
                                      init=I_STAR, val=correct))
            endpoints.append(endpoint(f"sftt-{lr}-{seed}", "SFT-T", lr, seed, 3900, 13000.0,
                                      init=f"sfte-{lr}-{seed}", val=correct))
    return {
        "labels": LABELS,
        "rows": {name: [{"id": i, "label": label} for i, label in split]
                 for name, split in (("h_select", H_SELECT), ("h_gate", H_GATE),
                                     ("val", FULL), ("test", FULL))},
        "s_b_ids": list(S_B),
        "pi": p0_doc(I_ADAPTER, PI_PS, 4, 112),
        "p0_prime": p0_doc(I_STAR, PRIME_PS, 16, 128, journal_group=16),
        "endpoints": endpoints,
    }


def find(document, run_id):
    return next(e for e in document["endpoints"] if e["run_id"] == run_id)


def until(document, gate):
    """Drop the evidence that triggers gates after `gate`, so the run waits there."""
    later = {"GA": ("p0_prime",), "GB0": ()}
    for key in later.get(gate, ()):
        document.pop(key)
    return document


def test_u_generalised_to_g_matches_hand_values():
    assert u_g(0.5, 4) == 0.875  # 1 - 2/16
    assert u_g(0.5, 8) == 1 - 2 / 256
    assert u_g(0.9, 16) == pytest.approx(1 - 0.1853020188851841 - 1e-16, abs=1e-15)
    assert u_g(0.95, 8) < 0.5 < u_g(0.95, 16)


def test_a_floor_boundary_at_preview_s_b():
    floor = a_floor(987)
    assert floor == pytest.approx(375 / 987)  # 0.379939...
    assert not at_least(0.379, floor)
    assert at_least(0.380, floor)
    with pytest.raises(ValueError):
        a_floor(0)


def test_pi_passes_at_152_accepted_and_refuses_at_151():
    result = gate_pi(p0_doc(I_ADAPTER, PI_PS, 4, 112), S_B)
    by_group = result["values"]["by_group"]
    assert by_group["4"]["predicted_acceptance"] == 0.25  # 100 / 400
    assert by_group["8"]["predicted_acceptance"] == 0.38  # 152 / 400
    assert by_group["16"]["predicted_acceptance"] == 0.38
    assert by_group["4"]["p_histogram"] == {"below_band": 0, "in_band": 100, "above_band": 300}
    assert result["values"]["observed_useful_mixed"] == 0.875  # 112 / 128
    assert result["values"]["predicted_useful_mixed"] == 0.875
    assert result["values"]["journal_groups"] == [4]
    assert result["pass"] and result["advisory"] and result["forecast_pass"]
    low = [0.5] * 100 + [0.9] * 51 + [0.999] * 249  # 151 / 400 = .3775 < .37994
    failed = gate_pi(p0_doc(I_ADAPTER, low, 4, 112), S_B)
    assert not failed["checks"]["max_predicted_acceptance_g8_g16_at_least_a_floor"]
    assert failed["checks"]["eos"] and failed["checks"]["scorer_seconds"]
    # Advisory since amendment 1: the forecast fails, the gate never stops the study.
    assert failed["pass"] and not failed["forecast_pass"]


def test_pi_forecast_includes_sampled_validity():
    # 7 of the 512 rollout completions invalid: .98633 >= .95 passes; 26 invalid: .94922 fails.
    for invalid, ok in ((7, True), (26, False)):
        doc = p0_doc(I_ADAPTER, PI_PS, 4, 112)
        completions = [c for g in doc["rollouts"] for c in g["completions"]]
        for c in completions[:invalid]:
            c["valid"] = False
        result = gate_pi(doc, S_B)
        assert result["values"]["valid"] == pytest.approx((512 - invalid) / 512)
        assert result["checks"]["sampled_valid"] is ok and result["forecast_pass"] is ok
        assert result["pass"]


def test_pi_refuses_wrong_adapter_wrong_group_forged_journal_and_foreign_scan():
    with pytest.raises(ValueError, match="adapter"):
        gate_pi(p0_doc(W_ADAPTER, PI_PS, 4, 112), S_B)
    with pytest.raises(ValueError, match="G4"):
        gate_pi(p0_doc(I_ADAPTER, PI_PS, 8, 112), S_B)
    forged = p0_doc(I_ADAPTER, PI_PS, 4, 112)
    forged["journal"][300]["accepted"] = True  # p=.999 claimed accepted
    with pytest.raises(ValueError, match="no G"):
        gate_pi(forged, S_B)
    with pytest.raises(ValueError, match="exactly S_b"):
        gate_pi(p0_doc(I_ADAPTER, PI_PS, 4, 112), S_B[:-1] + ["foreign"])


def test_p0_values_at_g4_equal_the_legacy_g1_values():
    doc = p0_doc(I_ADAPTER, PI_PS, 4, 112)
    legacy = FRESH["gate_g1"](copy.deepcopy(doc), I_ADAPTER)["values"]
    ours = p0_values(doc, I_ADAPTER, S_B)
    for key in ("observed_useful_mixed", "predicted_useful_mixed", "valid", "eos",
                "scorer_seconds_per_candidate", "non_canonical_share_of_correct"):
        assert ours[key] == legacy[key], key
    assert ours["by_group"]["4"]["predicted_acceptance"] == legacy["acceptance"]


def test_full_hand_fixture_selects_and_gates_as_computed_by_hand():
    report = evaluate(evidence())
    gates = report["gates"]
    assert report["a_floor"] == pytest.approx(375 / 987) and report["s_b_size"] == 987
    assert gates["PI"]["pass"]
    ga = gates["GA"]
    assert (ga["c_star"], ga["winner"], ga["winner_arm"], ga["i_star"]) == (
        "a1-5e-5", I_STAR, "A2", I_STAR)
    assert ga["values"]["winner_minus_c_star"] == pytest.approx(0.2)  # .60 - .40
    assert ga["values"]["paired_lb"] == pytest.approx(0.2)  # every image +1/5
    assert ga["values"]["paired_lb_uncorrected"] == pytest.approx(0.2)  # no spread to rescale
    assert ga["values"]["small_n_scale"] == pytest.approx(math.sqrt(2))  # n = 2 per class
    assert ga["values"]["lb_quantile"] == pytest.approx(0.05 / 3)
    assert ga["values"]["constrained_winner_minus_c_star"] == pytest.approx(0.5)  # 1 - .5
    assert ga["pass"]
    gb0 = gates["GB0"]
    assert gb0["chosen_group"] == 16
    assert gb0["values"]["by_group"]["8"]["predicted_acceptance"] == 0.35
    assert gb0["values"]["by_group"]["16"]["predicted_acceptance"] == 0.40
    assert gb0["values"]["journal_groups"] == [16]
    mean_u = (100 * (1 - 2 * 0.5**16) + 40 * (1 - 0.9**16 - 0.1**16)
              + 20 * (1 - 0.95**16 - 0.05**16)) / 160
    assert gb0["values"]["by_group"]["16"]["predicted_mean_u_accepted"] == pytest.approx(
        mean_u, abs=1e-12)
    assert gb0["pass"]
    assert gates["G2"]["continue_to_300"] == ["rl100-5e-5", "rl100-1e-5"]
    assert gates["G2"]["values"]["h_select_delta_vs_i_star"] == pytest.approx(0.1)
    assert gates["G2"]["values"]["trailing_useful_mixed"] == pytest.approx(0.7)
    assert gates["G3a"]["rl_star"] == "rl300-1e-5"
    assert gates["G3a"]["sft_e_star"] == "sfte-5e-5"
    assert gates["G3a"]["sft_t_lrs"] == [5e-5, 1e-4]
    assert gates["G3a"]["values"]["rl_minus_sft_e"] == pytest.approx(2 / 9)  # 7/9 - 5/9
    assert gates["G3a"]["values"]["rl_minus_i_star_lb90"] == pytest.approx(3 / 9)
    assert gates["G3b"]["sft_t_star"] == "sftt-5e-5"
    assert gates["G3b"]["values"]["rl_minus_sft_t"] == pytest.approx(1 / 9)
    assert gates["G3b"]["lim_t"]["rl_minus_lim_t"] == pytest.approx(2 / 9)  # 7/9 - 5/9
    assert gates["G3b"]["lim_t"]["gating"] and gates["G3b"]["pass"]
    protocol, robust = gates["G4"]["protocol"], gates["G4"]["robust"]
    # Protocol pool: A3* (H-select .39, the A3 family best) joins as a comparator and holds
    # the pool maximum, VAL .75; RL* .90 is +.15 ahead, RER 1 - .10/.25 = .6.
    assert protocol["strongest_sft"] == "a3-5e-5"
    assert {e["run_id"] for e in protocol["pool"]} == {
        "w", *C400, "a1-5e-5", "a1-1e-4", "a3-5e-5", "a5-5e-5", I_STAR, "sfte-1e-5",
        "sfte-5e-5", "sfte-1e-4", "sftt-5e-5", "sftt-1e-4"}
    assert protocol["values"]["delta_vs_strongest_sft"] == pytest.approx(0.15)
    assert protocol["values"]["relative_error_reduction"] == pytest.approx(0.6)
    # Robust pool adds LIM-T .80 (A2* is I*, already in): LIM-T is strongest; +.10, RER .5.
    assert robust["strongest_sft"] == "limt"
    assert {e["run_id"] for e in robust["pool"]} - {e["run_id"] for e in protocol["pool"]} == {
        "limt"}
    assert robust["values"]["delta_vs_strongest_sft"] == pytest.approx(0.1)
    assert robust["values"]["relative_error_reduction"] == pytest.approx(0.5)
    # Each class differs on 2 of 20 images; a zero draw needs all 10 classes zero
    # (.9**20 each, about 1e-9 jointly), so the 95% LB is positive.
    assert robust["pass"] and protocol["pass"] and gates["G4"]["pass"]
    assert gates["G4"]["d_req_realised"]["d_req"] == pytest.approx(0.05)  # a* = .80
    assert gates["G5"]["deltas"] == pytest.approx({"42": 0.1, "314": 0.1, "2718": 0.05})
    assert gates["G5"]["mean"] == pytest.approx(0.25 / 3)
    assert gates["G5"]["pass"]
    assert gates["G6"]["contenders"] == ["rl300-1e-5", "limt", "w"]
    assert gates["G6"]["substantial_rl_test"]["pass"]
    assert gates["G6"]["status"] == "partially_evaluated" and not gates["G6"]["pass"]
    assert report["awaiting"] == "G6" and report["failed_at"] is None
    assert report["substantial_val_and_replication"] and report["rl_test_gates"]
    assert not report["final_claim"]
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_pi_forecast_failure_does_not_stop_stage_a():
    document = evidence()
    document["pi"] = p0_doc(I_ADAPTER, [0.5] * 100 + [0.9] * 51 + [0.999] * 249, 4, 112)
    report = evaluate(document)
    assert report["gates"]["PI"]["forecast_pass"] is False
    assert report["failed_at"] is None and report["awaiting"] == "G6"
    assert report["gates"]["GA"]["status"] == "evaluated" and report["gates"]["GA"]["pass"]


def gate_a_after(change):
    document = until(evidence(), "GA")
    change(document)
    return evaluate(document)


def test_gate_a_margin_fires_alone():
    # A2* = .42: class C0's two images +1/5 each, every other image equal to C*: mean
    # +.02 < .03, and every bootstrap draw is .02, so the LB (.02) still passes.
    def change(d):
        find(d, I_STAR)["splits"]["h_select"]["predictions"] = repeated(
            H_SELECT, spread(42), 5)
    report = gate_a_after(change)
    ga = report["gates"]["GA"]
    assert report["failed_at"] == "GA"
    assert ga["values"]["winner_minus_c_star"] == pytest.approx(0.02)
    assert ga["values"]["paired_lb"] == pytest.approx(0.02)
    assert [k for k, v in ga["checks"].items() if not v] == ["winner_minus_c_star_at_least_0_03"]


def test_gate_a_uses_the_bonferroni_quantile_not_0_10():
    # C* (A1*): C0, C1 at (1, 1), C2, C3 at (3, 3), the rest (2, 2): total 40 = .40.
    # Winner: C0, C1 at (5, 0), the rest as C*: total 46 = .46, mean +.06.
    # Class C0 / C1 deltas (+.8, -.2): resampled class means -.2, +.3, +.8 w.p. 1/4, 1/2,
    # 1/4. The overall draw (sum / 10) is -.04 w.p. 1/16 = .0625 and +.01 w.p. 1/4.
    # So the .05/3 = .0167 quantile is -.04 (fails) while the .10 quantile is +.01. With the
    # small-n rescale (n = 2, sqrt 2) the gated bound is .06 + sqrt(2) x (-.04 - .06).
    c_star = [1, 1, 1, 1, 3, 3, 3, 3] + [2] * 12
    winner = [5, 0, 5, 0, 3, 3, 3, 3] + [2] * 12

    def change(d):
        find(d, "a1-5e-5")["splits"]["h_select"]["predictions"] = repeated(H_SELECT, c_star, 5)
        find(d, I_STAR)["splits"]["h_select"]["predictions"] = repeated(H_SELECT, winner, 5)
    report = gate_a_after(change)
    ga = report["gates"]["GA"]
    assert ga["values"]["winner_minus_c_star"] == pytest.approx(0.06)
    assert ga["values"]["lb_quantile"] == pytest.approx(0.05 / 3)
    assert ga["values"]["paired_lb_uncorrected"] == pytest.approx(-0.04)
    assert ga["values"]["paired_lb"] == pytest.approx(0.06 - math.sqrt(2) * 0.1)
    assert [k for k, v in ga["checks"].items() if not v] == ["bonferroni_paired_lb_positive"]
    rows = [{"id": i, "label": label} for i, label in H_SELECT]
    at_10 = compare_repeated_classification(
        rows, repeated(H_SELECT, winner, 5), repeated(H_SELECT, c_star, 5), LABELS,
        completions=5, lower_quantile=0.10)
    assert at_10["uncertainty"]["lower_bound"] == pytest.approx(0.01)  # fixture sanity


@pytest.mark.parametrize(("change", "check"), [
    # Constrained greedy: winner 1 per class = C*'s .5, so the difference 0 is not > 0.
    (lambda d: find(d, I_STAR)["splits"]["h_select_constrained"].update(
        predictions=constrained(1)), "constrained_greedy_winner_minus_c_star_positive"),
    # One invalid completion in 100: winner validity .99 < C*'s 1.0.
    (lambda d: find(d, I_STAR)["splits"]["h_select"].update(
        predictions=repeated(H_SELECT, spread(60), 5, invalid=(0,))),
     "winner_validity_at_least_c_star"),
    (lambda d: find(d, "a3-5e-5").update(init_sha256="not-w"), "init_matches"),
    (lambda d: find(d, "a5-1e-4").update(adapter_sha256=W_ADAPTER), "adapter_changed"),
])
def test_gate_a_each_other_condition_fires(change, check):
    report = gate_a_after(change)
    assert report["failed_at"] == "GA"
    assert [k for k, v in report["gates"]["GA"]["checks"].items() if not v] == [check]


def test_gate_a_refuses_missing_trie_identity_and_wrong_update_counts():
    document = evidence()
    find(document, "a1-5e-5")["splits"]["h_select_constrained"]["trie_sha256"] = "other"
    with pytest.raises(ValueError, match="trie_sha256"):
        evaluate(document)
    document = evidence()
    find(document, "a2-1e-5")["updates"] = 599
    with pytest.raises(ValueError, match="exactly 600"):
        evaluate(document)


def test_gate_b0_chooses_the_smallest_qualifying_g():
    document = evidence()
    # 152 / 400 = .38 at both G8 and G16 -> G8, the smaller; rollouts must then be G8.
    document["p0_prime"] = p0_doc(I_STAR, PI_PS, 8, 128, journal_group=8)
    report = evaluate(until(document, "GB0"))
    assert report["gates"]["GB0"]["chosen_group"] == 8
    document["p0_prime"] = p0_doc(I_STAR, PI_PS, 16, 128, journal_group=16)
    with pytest.raises(ValueError, match="chosen G is 8"):
        evaluate(document)
    # Nothing clears a_floor: G8 = G16 = 100/400 = .25.
    document["p0_prime"] = p0_doc(I_STAR, [0.5] * 100 + [0.999] * 300, 16, 128)
    report = evaluate(document)
    assert report["failed_at"] == "GB0"
    assert report["gates"]["GB0"]["checks"] == {"g_qualifies": False}


def test_gate_b0_calibration_and_mean_u_boundaries():
    # Predicted mixed at G16 on p=.5 images = 1 - 2/65536 = .99997. 116/128 = .90625
    # (gap .0937) passes; 115/128 = .8984 (gap .1015) fails.
    document = until(evidence(), "GB0")
    document["p0_prime"] = p0_doc(I_STAR, PRIME_PS, 16, 116, journal_group=16)
    assert evaluate(document)["gates"]["GB0"]["pass"]
    document["p0_prime"] = p0_doc(I_STAR, PRIME_PS, 16, 115, journal_group=16)
    report = evaluate(document)
    assert report["failed_at"] == "GB0"
    assert [k for k, v in report["gates"]["GB0"]["checks"].items() if not v] == [
        "useful_mixed_calibrated"]
    # Only p=.95 accepted (160 of 400 at G16): mean u = .5599 < .60. The 64 rollout
    # images are p=.95, predicted mixed .5599; 72/128 = .5625 observed.
    ps = [0.95] * 160 + [0.999] * 240
    document["p0_prime"] = p0_doc(I_STAR, ps, 16, 72, journal_group=16)
    gb0 = evaluate(document)["gates"]["GB0"]
    assert gb0["values"]["by_group"]["16"]["predicted_mean_u_accepted"] == pytest.approx(
        1 - 0.95**16 - 0.05**16)
    assert [k for k, v in gb0["checks"].items() if not v] == ["mean_u_accepted"]


def test_gate_b0_refuses_an_adapter_other_than_i_star():
    document = evidence()
    document["p0_prime"]["adapter_sha256"] = "a1-5e-5"
    with pytest.raises(ValueError, match="adapter"):
        evaluate(document)


@pytest.mark.parametrize(("change", "check"), [
    (lambda d: find(d, "rl100-5e-5").update(adapter_sha256=I_STAR), "adapter_changed"),
    (lambda d: find(d, "rl100-5e-5").update(reference_sha256=W_ADAPTER),
     "rl_reference_is_init"),
    (lambda d: find(d, "rl100-1e-4").update(beta=0.04), "rl_profile_recorded"),
    (lambda d: find(d, "rl100-1e-4").update(loss_type="grpo"), "rl_profile_recorded"),
    (lambda d: find(d, "rl100-1e-4").update(group=8), "rl_profile_recorded"),
    (lambda d: find(d, "rl100-1e-4").update(reward="exact_hn"), "rl_profile_recorded"),
    (lambda d: find(d, "rl100-1e-4").update(init_sha256=W_ADAPTER), "init_matches"),
    # 59 of the 100 mixed: .59 < .60.
    (lambda d: find(d, "rl100-5e-5").update(useful_mixed=[False] * 41 + [True] * 59),
     "rolling_useful_mixed"),
])
def test_g2_fires_on_each_condition(change, check):
    document = evidence()
    change(document)
    report = evaluate(document)
    assert report["failed_at"] == "G2"
    assert not report["gates"]["G2"]["checks"][check]
    assert report["gates"]["G3a"]["status"] == "not_evaluated"


def test_g2_delta_and_invalid_conditions():
    document = evidence()
    for run_id in ("rl100-1e-5", "rl100-5e-5", "rl100-1e-4"):
        find(document, run_id)["splits"]["h_select"]["predictions"] = repeated(
            H_SELECT, spread(60), 5)  # equal to I* (.60): delta 0 is not > 0
    assert not evaluate(document)["gates"]["G2"]["checks"]["h_select_delta_vs_i_star_positive"]
    # Slack max(2, 2 sqrt(a + b)) with I* at b = 0: four invalid -> slack 2 x 2 = 4, passes
    # at the boundary; five -> 2 sqrt 5 = 4.47 < 5, fails.
    for images, ok in (((0, 1, 2, 3), True), ((0, 1, 2, 3, 4), False)):
        document = evidence()
        find(document, "rl100-5e-5")["splits"]["h_select"]["predictions"] = repeated(
            H_SELECT, spread(70), 5, invalid=images)
        g2 = evaluate(document)["gates"]["G2"]
        assert g2["values"]["invalid"] == len(images) and g2["values"]["i_star_invalid"] == 0
        assert g2["values"]["invalid_slack"] == pytest.approx(max(2, 2 * math.sqrt(len(images))))
        assert g2["checks"]["invalid_within_i_star_plus_slack"] is ok


def test_g2_invalid_slack_is_two_poisson_sds_of_the_difference():
    slack = GATES["invalid_slack"]
    assert slack(0, 0) == 2  # floor
    assert slack(50, 50) == pytest.approx(20.0)  # 2 x sqrt(100)
    assert slack(1, 0) == 2  # 2 x 1 = 2


def test_g3a_requires_five_points_over_sft_e():
    document = evidence()
    # SFT-E* H-gate 7/9, equal to RL*: RL* - SFT-E* = 0 < .05.
    find(document, "sfte-5e-5")["splits"]["h_gate"]["predictions"] = repeated(
        H_GATE, spread(140), 9)
    report = evaluate(document)
    assert report["failed_at"] == "G3a"
    assert [k for k, v in report["gates"]["G3a"]["checks"].items() if not v] == [
        "rl_minus_sft_e_at_least_0_05"]


def test_g3b_requires_five_points_over_sft_t():
    document = evidence()
    # SFT-T* H-gate 132/180: RL* 140/180 - 132/180 = 8/180 = .044 < .05.
    find(document, "sftt-5e-5")["splits"]["h_gate"]["predictions"] = repeated(
        H_GATE, spread(132), 9)
    report = evaluate(document)
    assert report["failed_at"] == "G3b"
    assert report["gates"]["G3b"]["values"]["rl_minus_sft_t"] == pytest.approx(8 / 180)
    assert [k for k, v in report["gates"]["G3b"]["checks"].items() if not v] == [
        "rl_minus_sft_t_at_least_margin"]


def test_g3b_gates_on_lim_t_too():
    document = evidence()
    # LIM-T H-gate 132/180: RL* - LIM-T = 8/180 = .044 < .05 (amendment 1: it gates).
    find(document, "limt")["splits"]["h_gate"]["predictions"] = repeated(H_GATE, spread(132), 9)
    report = evaluate(document)
    assert report["failed_at"] == "G3b"
    assert report["gates"]["G3b"]["lim_t"]["rl_minus_lim_t"] == pytest.approx(8 / 180)
    assert [k for k, v in report["gates"]["G3b"]["checks"].items() if not v] == [
        "rl_minus_lim_t_at_least_margin"]
    assert report["gates"]["G4"]["status"] == "not_evaluated"


def test_g4_fails_on_the_robust_verdict_alone():
    document = evidence()
    find(document, "limt")["splits"]["val"]["predictions"] = greedy(18)  # equals RL*
    report = evaluate(document)
    g4 = report["gates"]["G4"]
    assert g4["protocol"]["pass"] and not g4["robust"]["pass"]
    assert g4["claim_rests_on"] == "robust"
    assert report["failed_at"] == "G4"


def test_g4_pool_carries_stage_a_ce_endpoints():
    # A1 at 1e-4 (H-select .39, within 2 pp of A1*'s .40) at VAL .85 becomes the protocol
    # strongest SFT, so RL* .90 is only +.05 ahead of it.
    document = evidence()
    find(document, "a1-1e-4")["splits"]["val"]["predictions"] = greedy(17)
    for e in document["endpoints"]:
        e["splits"].pop("test", None)  # stop before G6
    g4 = evaluate(document)["gates"]["G4"]
    assert g4["protocol"]["strongest_sft"] == "a1-1e-4"
    assert g4["protocol"]["values"]["delta_vs_strongest_sft"] == pytest.approx(0.05)


def test_g5_pool_includes_i_star_and_every_seed_must_be_positive():
    document = evidence()
    find(document, "a3-5e-5")["splits"]["val"]["predictions"] = greedy(13)  # A3* to .65
    for seed in (314,):
        for lr in (5e-5, 1e-4):
            for arm in ("sfte", "sftt"):
                find(document, f"{arm}-{lr}-{seed}")["splits"]["val"]["predictions"] = greedy(13)
        find(document, f"limt-{seed}")["splits"]["val"]["predictions"] = greedy(13)
    g5 = evaluate(document)["gates"]["G5"]
    assert g5["seeds"]["314"]["strongest_sft"] == I_STAR  # I* .70 > every other .65
    assert g5["deltas"]["314"] == pytest.approx(0.15)
    document = evidence()
    for lr in (5e-5, 1e-4):
        for arm in ("sfte", "sftt"):
            find(document, f"{arm}-{lr}-2718")["splits"]["val"]["predictions"] = greedy(17)
    report = evaluate(document)
    assert report["failed_at"] == "G5"
    assert report["gates"]["G5"]["deltas"]["2718"] == pytest.approx(0.0)
    assert not report["gates"]["G5"]["checks"]["every_seed_delta_positive"]


def test_g5_uses_the_robust_pool_per_seed():
    # Seed 314's own LIM-T at .85 equals RL; seed-independent A3* (.75) and seed-42 robust
    # members stay in the pool. Delta 0 at seed 314 fails the replication.
    document = evidence()
    find(document, "limt-314")["splits"]["val"]["predictions"] = greedy(17)
    report = evaluate(document)
    g5 = report["gates"]["G5"]
    assert g5["seeds"]["314"]["strongest_sft"] == "limt-314"
    assert g5["deltas"]["314"] == pytest.approx(0.0)
    assert report["failed_at"] == "G5"
    pool = {item["run_id"] for item in g5["seeds"]["2718"]["pool"]}
    assert {"a3-5e-5", I_STAR, "w", "limt-2718"} <= pool
    assert not pool & {"limt", "sfte-5e-5", "sftt-5e-5", "limt-314"}
    document = evidence()
    document["endpoints"] = [e for e in document["endpoints"] if e["run_id"] != "limt-2718"]
    with pytest.raises(ValueError, match="LIM-T at seed 2718"):
        evaluate(document)


def test_small_n_correction_rescales_about_the_mean_and_flips_a_marginal_bound():
    # Unit: n = 2 -> scale sqrt 2; mean .10, plain bound .02 -> .10 - sqrt 2 x .08.
    rows = [{"id": i, "label": label} for i, label in H_SELECT]
    comparison = {"mean_class_accuracy_difference": 0.10, "uncertainty": {"lower_bound": 0.02}}
    corrected, plain, scale = small_n_lower_bound(comparison, rows)
    assert (plain, scale) == (0.02, math.sqrt(2))
    assert corrected == pytest.approx(0.10 - math.sqrt(2) * 0.08)  # -.0131
    with pytest.raises(ValueError, match="same n"):
        small_n_lower_bound(comparison, rows[:-1])
    # Gate A forced fire. Per image correct counts (classes C0..C9, two images each):
    # C* (A1*): C0 (0, 3), C1-C3 (2, 2), C4 (3, 2), C5-C9 (2, 2): 40 = .40.
    # Winner A2*: C0 (5, 0), C1-C3 (3, 3), C4 (4, 3), C5-C9 as C*: 50 = .50.
    # Deltas: C0 (+1, -.6), C1-C4 both +.2, C5-C9 0; mean (.2 + 4 x .2) / 10 = .10 >= .03.
    # C0's resampled mean is -.6 with probability 1/4 > .0167, so the plain bound is
    # (-.6 + .8) / 10 = .02 > 0, and the corrected one .10 + sqrt 2 (.02 - .10) < 0.
    c_star = [0, 3] + [2] * 6 + [3, 2] + [2] * 10
    winner = [5, 0] + [3] * 6 + [4, 3] + [2] * 10

    def change(d):
        find(d, "a1-5e-5")["splits"]["h_select"]["predictions"] = repeated(H_SELECT, c_star, 5)
        find(d, I_STAR)["splits"]["h_select"]["predictions"] = repeated(H_SELECT, winner, 5)
    report = gate_a_after(change)
    ga = report["gates"]["GA"]
    assert ga["values"]["winner_minus_c_star"] == pytest.approx(0.10)
    assert ga["values"]["paired_lb_uncorrected"] == pytest.approx(0.02)
    assert ga["values"]["paired_lb"] == pytest.approx(0.10 - math.sqrt(2) * 0.08)
    assert [k for k, v in ga["checks"].items() if not v] == ["bonferroni_paired_lb_positive"]


def test_c_star_may_be_a3_and_a3_never_wins_gate_a():
    # A3* at .70 beats A2* (.60): A3 is a comparator, so it becomes C* and Gate A fails;
    # the winner is still A2*, and A3* becomes I* (H-select best of A1*/A2*/A3*).
    def change(d):
        find(d, "a3-5e-5")["splits"]["h_select"]["predictions"] = repeated(
            H_SELECT, spread(70), 5)
    ga = gate_a_after(change)["gates"]["GA"]
    assert (ga["c_star"], ga["winner"], ga["winner_arm"], ga["i_star"]) == (
        "a3-5e-5", I_STAR, "A2", "a3-5e-5")
    assert ga["values"]["winner_minus_c_star"] == pytest.approx(-0.1)
    assert not ga["pass"]


def test_refusals_for_identities_the_evaluator_pins():
    document = evidence()
    find(document, "w")["adapter_sha256"] = "other"
    with pytest.raises(ValueError, match="W endpoint"):
        evaluate(document)
    document = evidence()
    find(document, "sft-1e5-time")["adapter_sha256"] = "other"
    with pytest.raises(ValueError, match="I's weights"):
        evaluate(document)
    document = evidence()
    document["s_b_ids"] = S_B + [S_B[0]]
    with pytest.raises(ValueError, match="s_b_ids"):
        evaluate(document)


def test_i_star_is_chosen_among_a1_a3_only_and_c_star_may_be_a5():
    # A5* at .70 beats every other Stage-A endpoint: it becomes C*, so the winner A2*
    # (.60) trails by .10 and Gate A fails; I* is still the best of A1*/A2*/A3* (A2*).
    def change(d):
        find(d, "a5-5e-5")["splits"]["h_select"]["predictions"] = repeated(
            H_SELECT, spread(70), 5)
    ga = gate_a_after(change)["gates"]["GA"]
    assert ga["c_star"] == "a5-5e-5" and ga["i_star"] == I_STAR
    assert ga["values"]["winner_minus_c_star"] == pytest.approx(-0.1)
    assert not ga["pass"]
