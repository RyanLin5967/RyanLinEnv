"""Evaluate aircraft-rl-next-study-v1 (aircraft-signal-limit-v1) gates PI, GA, GB0, G2-G6.

Pure evaluation of already recorded outcomes: no model, image or accelerator access.
Thresholds are copied from docs/aircraft-rl-next-study-v1.json (gates, bootstrap,
budget_matching) and the unchanged protocol ``substantial_gain_gates``; none is a
parameter. Gates run in order and stop at the first failure; evidence for a later gate
after a stop is listed, never evaluated. G0 and E0 are not evaluated here (G0 is the
launcher's manifest/trie/ledger check, E0 is scripts/freshpool_launch/e0_gate.py); this
script publishes a_floor, which G0 requires.

Reuse: selection and gate arithmetic come from scripts/evaluate_freshpool_gates.py
(at_least, above, best, ranked, within, d_req, scan_order and its Study class:
find/family/h_scored/val/bound_to_t_rl/select_sft_e/select_sft_t/_h_gate_difference/
gate_g3b/substantial/substantial_against/gate_g6), u_G(p) from
vision_rl.vlm_local_training.group_useful_probability, and every comparison from
vision_rl.classification_comparison.

Evidence document (JSON):
  labels       ordered canonical vocabulary
  rows         {"h_select"|"h_gate"|"val"|"test": [{"id", "label"}]}
  s_b_ids      every S_b ID (|S_b| sets a_floor = 300 x 1.25 / |S_b|; P0 scan sets must equal it)
  pi           P0-I document at I (scripts/freshpool_launch/p0_screen.py g1 shape):
               {"seed": 42, "adapter_sha256", "scan_order", "journal": [400 x {"id", "p",
               "u", "accepted", "scorer_seconds", "adapter_sha256"}], "rollouts": [64 images
               x seeds 42, 314 x {"id", "seed", "completions": [{"correct", "valid", "eos",
               "canonical_tokenization", "label_only"}]}]}; rollout groups of 4 completions
  p0_prime     the same shape at I*, the same 400 candidates; rollout groups of the chosen G
  endpoints    [{"run_id", "arm", "lr", "seed", "updates", "committed_active_seconds",
                 "adapter_sha256", "init_sha256", "splits": {...}, ...}]
    arm: W, controlled400-SFT, A1-A5 (Stage A), RL, SFT-E, SFT-T, LIM-T (Stage B).
    W and controlled400-SFT need no init_sha256. RL also needs reference_sha256, loss_type,
    scale_rewards, beta, group, reward and (at 100 updates) useful_mixed flags.
    splits.h_select / h_gate: {"predictions": {id: [completions]}, "eos_fraction"}
    splits.h_select_constrained: {"predictions": {id: label | null}, "trie_sha256"}
    splits.val / test: {"predictions": {id: label | null}}
  named_baseline_test   optional {"run_id", "predictions", "valid_eos_fraction"}

Interpretations (the study text admits more than one reading; each is a choice made here):
  * "Sampled macro" is the freshpool reading ratified in D24: mean-class accuracy over every
    completion per image, greedy included (5 on H-select, 9 on H-gate).
  * P0 journals may record u at any G in {4, 8, 16}; the G is recovered from the entries
    (u == u_G(p) and accepted == u >= 0.5 for every entry) and refused if none fits.
    Predicted acceptance and mean u for every G are recomputed from p. The rollout G is
    the common completion count of the groups: 4 at PI, the chosen G at GB0.
  * Gate A (ii) bounds the winner A2* (the screened Dr-GRPO limit, the only RL limit after
    design-review amendment 1) against C* = the H-select best of A1*, A3*, A5*; the
    Bonferroni quantile 0.05/3 pays for the 3 (objective, LR) candidates A2 x 3 it was
    picked from. Gate A (iii) compares the same two endpoints under constrained greedy, both
    decoded with one trie (equal trie_sha256 required). Gate A (iv) compares unconstrained
    H-select validity.
  * H-split lower bounds (Gate A (ii), G3a) carry the small-n correction of the study's
    bootstrap block: with n images in every class, each resampled class mean is rescaled by
    sqrt(n / (n - 1)) about the observed one, so the bootstrap variance is the unbiased one
    (the plain bootstrap gives (n - 1) / n of it, half at n = 2). With equal n the rescale is
    one affine map of every draw, so the corrected bound is mean + s x (plain bound - mean);
    unequal class sizes refuse.
  * PI is advisory (amendment 1): its checks are a forecast published before Stage A, and
    its failure no longer stops the study; GB0 at I* is the binding acceptance gate.
  * G2's rolling useful-mixed fraction is over the top RL arm's last 100 accepted updates
    (all of them at 100). G2's invalid check allows the top arm max(2, 2 sqrt(a + b)) more
    invalid completions than I* (a, b the two invalid counts): two Poisson SDs of the
    difference, instead of the fixed +2 that fired under the null. G3 uses the study's fixed
    mechanism margin 0.05, not d_req; G3b gates on LIM-T as well as SFT-T*.
  * Gate pass at G4 needs both verdicts (the robust pool contains the protocol pool, and the
    claim rests on the robust verdict). The seed-42 delta that enters G5's replication mean
    is the robust one, each replicate seed's pool is the robust one (every seed-independent
    member of the seed-42 robust pool plus that seed's SFT-E, SFT-T and LIM-T), and G6's
    strongest SFT is the robust one. The protocol verdict is reported alongside.
  * A3 (HN-SFT) is not the limit of any RL reward (amendment 1): it is a supervised
    comparator in C*, an I* candidate, and a protocol-pool member like A1/A5; RL always
    uses the exact reward and LIM-T always continues A2's objective.
  * W stays in the G4 pools and in each G5 replicate pool next to I*: strongest_sft says
    "the pool only grows" and freshpool's pool held W. Adding a candidate can only raise the
    strongest SFT, so this is the conservative reading.

Blind spots, stated: identities (adapter, init, reference) and committed active seconds are
trusted as recorded; nothing here sees whether a look was the first one (the look ledger is
the phase worker's), whether SFT-E trained on exactly RL*'s accepted IDs in order, whether
Stage A arms shared one S_a order, or whether SFT-T / LIM-T overshot T_RL by more than one
update. Run IDs, not adapter bytes, identify controlled400 endpoints, except that I
(sft-1e5-time) must carry its activated weights hash.
"""

import argparse
import json
import math
import runpy
from collections import Counter
from pathlib import Path

from vision_rl.classification_comparison import (
    compare_classification,
    compare_repeated_classification,
)
from vision_rl.vlm_local_training import group_useful_probability

FRESH = runpy.run_path(str(Path(__file__).with_name("evaluate_freshpool_gates.py")))
at_least, above, best, ranked, within, d_req, scan_order = (
    FRESH[name] for name in ("at_least", "above", "best", "ranked", "within", "d_req",
                             "scan_order"))
FreshStudy = FRESH["Study"]
EPSILON, GRID, COMPLETIONS = FRESH["EPSILON"], FRESH["GRID"], FRESH["COMPLETIONS"]
PRIMARY_SEED, REPLICATE_SEEDS = FRESH["PRIMARY_SEED"], FRESH["REPLICATE_SEEDS"]
CONTROLLED400_SFT, SFT_T_WINDOW = FRESH["CONTROLLED400_SFT"], FRESH["SFT_T_WINDOW"]

# docs/aircraft-native-val-activation-v1/activated-declaration.json weights_sha256.
W_ADAPTER = "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"
I_RUN_ID = "sft-1e5-time"
I_ADAPTER = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
GROUPS = (4, 8, 16)
G_CHOICES = (8, 16)
PI_GROUP = 4
N_ACCEPTED, CAP = 300, 1.25  # a_floor = N x 1.25 / |S_b|
P0 = {"candidates": 400, "rollout_images": 64, "rollout_seeds": (42, 314), "u_threshold": 0.5,
      "scorer_seconds": 2.0, "eos": 0.95, "valid": 0.95, "mixed_tolerance": 0.10,
      "mean_u_accepted": 0.60}
STAGE_A_UPDATES = 600
STAGE_A_ARMS = ("A1", "A2", "A3", "A4", "A5")
LIMIT_ARMS = ("A2",)  # Gate A candidates: the screened Dr-GRPO limit at every grid LR
COMPARATOR_ARMS = ("A1", "A3", "A5")  # C* = their H-select best (A3 joined in amendment 1)
I_STAR_FROM = ("A1", "A2", "A3")
GATE_A = {"margin": 0.03, "candidates": 3, "lb_quantile": 0.05 / 3}
POOL_WINDOW = 0.02  # Stage-A A1/A3/A5 endpoints within 2 pp of their family best: pool
G2 = {"updates": 100, "min_invalid_slack": 2, "invalid_sds": 2.0, "rolling_window": 100,
      "rolling_mixed": 0.6}
FINAL_UPDATES = N_ACCEPTED
G3 = {"margin": 0.05, "lb_quantile": 0.10, "valid": 0.95, "eos": 0.95}
RL_PROFILE = {"loss_type": "dr_grpo", "scale_rewards": "none"}
RL_REWARD = "exact"  # the A2 limit's reward; exact_hn is not used by this study
SEED_BOUND_ARMS = ("SFT-E", "SFT-T", "LIM-T")  # rerun per replicate seed at G5
ARMS = {"W", "controlled400-SFT", *STAGE_A_ARMS, "RL", "SFT-E", "SFT-T", "LIM-T"}


def small_n_lower_bound(comparison, rows):
    """The plain paired bootstrap bound rescaled for n images per class (study bootstrap
    block, design-review finding 1): mean + sqrt(n / (n - 1)) x (bound - mean)."""
    sizes = set(Counter(row["label"] for row in rows).values())
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError("the small-n correction needs the same n >= 2 images in every class")
    (n,) = sizes
    mean = comparison["mean_class_accuracy_difference"]
    plain = comparison["uncertainty"]["lower_bound"]
    return mean + math.sqrt(n / (n - 1)) * (plain - mean), plain, math.sqrt(n / (n - 1))


def invalid_slack(candidate_invalid, comparator_invalid):
    """G2: two Poisson SDs of the difference of the two invalid counts, at least 2."""
    return max(G2["min_invalid_slack"],
               G2["invalid_sds"] * math.sqrt(candidate_invalid + comparator_invalid))


def u_g(p, group):
    """u_G(p) = 1 - p^G - (1-p)^G, the existing training helper."""
    return group_useful_probability(p, group)


def a_floor(s_b_size):
    """Refusal arithmetic: the scan must accept N = 300 within 1.25 x |S_b| candidates."""
    if type(s_b_size) is not int or s_b_size < 1:
        raise ValueError("|S_b| must be a positive integer")
    return N_ACCEPTED * CAP / s_b_size


def p0_values(doc, adapter, s_b_ids, allow_no_rollouts=False):
    """Validate a P0 document and compute its values for every G in {4, 8, 16}.

    ``allow_no_rollouts`` (P0' only): when no G clears a_floor, p0_screen samples nothing
    and writes ``rollouts: []``; the journal is still validated and the rollout-derived
    values are None, so GB0 records its failure instead of crashing."""
    if doc.get("seed") != PRIMARY_SEED or doc.get("adapter_sha256") != adapter:
        raise ValueError(f"P0 must run at seed 42 under adapter {adapter}")
    order = doc.get("scan_order") or []
    if len(set(order)) != len(order) or order != scan_order(order, PRIMARY_SEED):
        raise ValueError("P0 scan_order is not random.Random(42).shuffle(sorted(ids))")
    if set(order) != set(s_b_ids):
        raise ValueError("P0 scan_order must cover exactly S_b")
    journal, rollouts = doc["journal"], doc["rollouts"]
    if len(journal) != P0["candidates"] or [e["id"] for e in journal] != order[:len(journal)]:
        raise ValueError("P0 needs the first 400 scanned S_b candidates in scan order")
    if any(e.get("adapter_sha256") != adapter for e in journal):
        raise ValueError("P0 journal entries must be scored under the P0 adapter")
    for entry in journal:
        p = entry["p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError(f"journal entry {entry['id']} needs p in [0, 1]")
    journal_groups = [
        g for g in GROUPS
        if all(abs(e["u"] - u_g(e["p"], g)) <= EPSILON
               and e["accepted"] == (e["u"] >= P0["u_threshold"]) for e in journal)]
    if not journal_groups:
        raise ValueError("P0 journal u/accepted match u_G(p) >= 0.5 for no G in {4, 8, 16}")
    scored = {e["id"]: e for e in journal}
    by_group = {}
    for g in GROUPS:
        us = [u_g(e["p"], g) for e in journal]
        accepted = [x for x in us if x >= P0["u_threshold"]]
        by_group[str(g)] = {
            "predicted_acceptance": len(accepted) / len(journal),
            "predicted_mean_u_accepted": sum(accepted) / len(accepted) if accepted else 0.0,
            "p_histogram": {
                "below_band": sum(x < P0["u_threshold"] and e["p"] < 0.5
                                  for x, e in zip(us, journal)),
                "in_band": len(accepted),
                "above_band": sum(x < P0["u_threshold"] and e["p"] > 0.5
                                  for x, e in zip(us, journal)),
            },
        }
    if allow_no_rollouts and rollouts == []:
        return {"journal_groups": journal_groups, "rollout_group": None,
                "journal_ids": [e["id"] for e in journal], "by_group": by_group,
                "observed_useful_mixed": None, "predicted_useful_mixed": None,
                "valid": None, "eos": None,
                "scorer_seconds_per_candidate": sum(e["scorer_seconds"] for e in journal)
                / len(journal),
                "non_canonical_share_of_correct": None}
    images = {g["id"] for g in rollouts}
    sizes = {len(g["completions"]) for g in rollouts}
    expected = {(i, s) for i in images for s in P0["rollout_seeds"]}
    if (len(images) != P0["rollout_images"] or images - set(scored)
            or {(g["id"], g["seed"]) for g in rollouts} != expected
            or len(rollouts) != len(expected) or len(sizes) != 1 or not sizes <= set(GROUPS)):
        raise ValueError("P0 needs equal-size groups (G in {4, 8, 16}) at seeds 42 and 314 "
                         "on 64 scanned candidates")
    (rollout_group,) = sizes
    completions = [c for g in rollouts for c in g["completions"]]
    correct = [c for c in completions if c["correct"]]
    if any(type(c.get("label_only")) is not bool for c in correct):
        raise ValueError("P0 correct completions must state label_only (bare label or not)")
    mixed = sum(len({c["correct"] for c in g["completions"]}) == 2 for g in rollouts)
    return {
        "journal_groups": journal_groups,
        "rollout_group": rollout_group,
        "journal_ids": [e["id"] for e in journal],
        "by_group": by_group,
        "observed_useful_mixed": mixed / len(rollouts),
        "predicted_useful_mixed": sum(u_g(scored[i]["p"], rollout_group) for i in images)
        / len(images),
        "valid": sum(c["valid"] for c in completions) / len(completions),
        "eos": sum(c["eos"] for c in completions) / len(completions),
        "scorer_seconds_per_candidate": sum(e["scorer_seconds"] for e in journal) / len(journal),
        "non_canonical_share_of_correct": (
            sum(c["label_only"] and not c["canonical_tokenization"] for c in correct)
            / len(correct) if correct else None),
    }


def gate_pi(doc, s_b_ids):
    """PI: P0-I at I, advisory since amendment 1. Its checks forecast GB0 (max over G in
    {8, 16} of predicted acceptance >= a_floor, scorer, EOS, and sampled validity, the most
    likely GB0 failure); ``forecast_pass`` is published, and the gate never stops the study,
    because I only proxies I* and Stage A's own result does not depend on it."""
    values = p0_values(doc, I_ADAPTER, s_b_ids)
    if values["rollout_group"] != PI_GROUP:
        raise ValueError("P0-I rollouts must be G4 groups")
    floor = a_floor(len(s_b_ids))
    best_acceptance = max(values["by_group"][str(g)]["predicted_acceptance"] for g in G_CHOICES)
    checks = {
        "max_predicted_acceptance_g8_g16_at_least_a_floor": at_least(best_acceptance, floor),
        "scorer_seconds": not above(values["scorer_seconds_per_candidate"],
                                    P0["scorer_seconds"]),
        "eos": at_least(values["eos"], P0["eos"]),
        "sampled_valid": at_least(values["valid"], P0["valid"]),
    }
    return {"pass": True, "advisory": True, "forecast_pass": all(checks.values()),
            "checks": checks, "a_floor": floor,
            "label": "advisory forecast of GB0 at the proxy I; never stops the study",
            "values": values}


def gate_b0(doc, s_b_ids, i_star_adapter, pi_journal_ids):
    """GB0: P0' at I*. G is the smallest in {8, 16} whose predicted acceptance clears a_floor."""
    values = p0_values(doc, i_star_adapter, s_b_ids, allow_no_rollouts=True)
    if values["journal_ids"] != pi_journal_ids:
        raise ValueError("P0' must score the same 400 S_b candidates as P0-I")
    floor = a_floor(len(s_b_ids))
    # The same rule P0' sampled by (scripts/freshpool_launch/p0_screen.py choose_group: exact
    # >=, no tolerance), so the gate and the launch can never name different G.
    chosen = next((g for g in G_CHOICES
                   if values["by_group"][str(g)]["predicted_acceptance"] >= floor), None)
    checks = {"g_qualifies": chosen is not None}
    if chosen is not None and values["rollout_group"] is None:
        raise ValueError(f"P0' chose no rollouts although G{chosen} qualifies")
    if chosen is not None:
        if values["rollout_group"] != chosen:
            raise ValueError(f"P0' rollouts are G{values['rollout_group']} but the chosen G is "
                             f"{chosen}")
        gap = abs(values["observed_useful_mixed"] - values["predicted_useful_mixed"])
        checks |= {
            "valid": at_least(values["valid"], P0["valid"]),
            "eos": at_least(values["eos"], P0["eos"]),
            "useful_mixed_calibrated": not above(gap, P0["mixed_tolerance"]),
            "mean_u_accepted": at_least(
                values["by_group"][str(chosen)]["predicted_mean_u_accepted"],
                P0["mean_u_accepted"]),
            "scorer_seconds": not above(values["scorer_seconds_per_candidate"],
                                        P0["scorer_seconds"]),
        }
    return {"pass": all(checks.values()), "checks": checks, "a_floor": floor,
            "chosen_group": chosen, "values": values}


def _zero(value):
    return type(value) in (int, float) and value == 0


class Study(FreshStudy):
    """The freshpool Study with this study's arms, budgets, initialisations and I* comparator."""

    def __init__(self, evidence):
        self.evidence, self.labels, self.rows = evidence, evidence["labels"], evidence["rows"]
        self.grid = set(GRID)
        self.branch = {"mode": "mechanism", "g3_margin": G3["margin"]}
        s_b = evidence.get("s_b_ids")
        if not isinstance(s_b, list) or not s_b or len(set(s_b)) != len(s_b):
            raise ValueError("s_b_ids must list every S_b ID once")
        self.s_b_ids, self.a_floor = s_b, a_floor(len(s_b))
        self.endpoints = {}
        for endpoint in evidence.get("endpoints", []):
            if endpoint["arm"] not in ARMS or endpoint["run_id"] in self.endpoints:
                raise ValueError(f"unknown arm or duplicate run ID {endpoint['run_id']}")
            self.endpoints[endpoint["run_id"]] = endpoint
        warm = self.find("W", seed=None)
        if len(warm) != 1 or warm[0]["adapter_sha256"] != W_ADAPTER:
            raise ValueError(f"evidence needs exactly one W endpoint with adapter {W_ADAPTER}")
        self.w, self._h, self._val, self._cmp, self._stage_a = warm[0], {}, {}, {}, None

    # Budgets and identities.
    def exact_updates(self, endpoints, updates=FINAL_UPDATES):
        for e in endpoints:
            if e["updates"] != updates:
                raise ValueError(f"{e['arm']} {e['run_id']} needs exactly {updates} updates, "
                                 f"has {e['updates']}")
        return endpoints

    def controlled400(self):
        found = super().controlled400()
        (i,) = [e for e in found if e["run_id"] == I_RUN_ID]
        if i["adapter_sha256"] != I_ADAPTER:
            raise ValueError(f"controlled400 {I_RUN_ID} must carry I's weights {I_ADAPTER}")
        return found

    def expected_init(self, endpoint):
        arm = endpoint["arm"]
        if arm in STAGE_A_ARMS:
            return W_ADAPTER
        if arm in ("RL", "SFT-E", "LIM-T"):
            return self.i_star()["adapter_sha256"]
        if arm == "SFT-T":
            sft_e = [e for e in self.find("SFT-E", endpoint["seed"])
                     if e["lr"] == endpoint["lr"]]
            if len(sft_e) != 1:
                raise ValueError(f"SFT-T {endpoint['run_id']} needs its SFT-E at the same "
                                 "seed and LR")
            return sft_e[0]["adapter_sha256"]
        return W_ADAPTER  # controlled400 endpoints only need to differ from W

    def identity(self, endpoints):
        checks = {"adapter_changed": True, "init_matches": True,
                  "rl_reference_is_init": True, "rl_profile_recorded": True}
        for e in endpoints:
            init = self.expected_init(e)
            checks["adapter_changed"] &= e["adapter_sha256"] != init
            if e["arm"] != "controlled400-SFT":
                checks["init_matches"] &= e.get("init_sha256") == init
            if e["arm"] == "RL":
                checks["rl_reference_is_init"] &= e.get("reference_sha256") == init
                checks["rl_profile_recorded"] &= (
                    all(e.get(k) == v for k, v in RL_PROFILE.items()) and _zero(e.get("beta"))
                    and e.get("group") == self.chosen_group()
                    and e.get("reward") == RL_REWARD)
        return checks

    # Comparisons.
    def compare(self, candidate, comparator, split, quantile=G3["lb_quantile"]):
        key = (candidate["run_id"], comparator["run_id"], split, quantile)
        if key not in self._cmp:
            self._cmp[key] = compare_repeated_classification(
                self.rows[split], self.predictions(candidate, split),
                self.predictions(comparator, split), self.labels,
                completions=COMPLETIONS[split], lower_quantile=quantile)
        return self._cmp[key]

    def h(self, endpoint, split, quantile=G3["lb_quantile"]):
        """The endpoint's own H score (h_scored reads only the candidate side)."""
        return self.compare(endpoint, endpoint, split, quantile)

    def constrained_macro(self, endpoint):
        result = compare_classification(
            self.rows["h_select"], self.predictions(endpoint, "h_select_constrained"),
            self.predictions(endpoint, "h_select_constrained"), self.labels, resamples=100)
        return result["candidate"]["mean_class_accuracy"]

    # Stage A selection (H-select only).
    def stage_a(self):
        if self._stage_a is None:
            table = {}
            for arm in ("A1", "A2", "A3", "A5"):
                scored = self.h_scored(self.exact_updates(self.family(arm), STAGE_A_UPDATES))
                table[arm] = {"h_select": scored, "star": best(scored, "macro")}
            a4 = self.exact_updates(self.family("A4", lrs=[table["A1"]["star"]["lr"]]),
                                    STAGE_A_UPDATES)
            table["A4"] = {"h_select": self.h_scored(a4), "role": "reported control"}
            c_star = best([table[arm]["star"] for arm in COMPARATOR_ARMS], "macro")
            winner = best([table[arm]["star"] for arm in LIMIT_ARMS], "macro")
            i_star = best([table[arm]["star"] for arm in I_STAR_FROM], "macro")
            arm_of = {r: self.endpoints[r]["arm"] for r in (winner["run_id"], c_star["run_id"])}
            self._stage_a = {"table": table, "c_star": c_star["run_id"],
                                  "winner": winner["run_id"],
                                  "winner_arm": arm_of[winner["run_id"]],
                                  "i_star": i_star["run_id"]}
        return self._stage_a

    def i_star(self):
        return self.endpoints[self.stage_a()["i_star"]]

    def chosen_group(self):
        if "p0_prime" not in self.evidence:
            raise ValueError("RL endpoints need P0' evidence for the chosen G")
        result = self.gate_b0()
        if result["chosen_group"] is None:
            raise ValueError("no G qualified at P0'; RL must not have run")
        return result["chosen_group"]

    # Stage B selection (H-select only).
    def select_rl_continuers(self):
        scored = self.h_scored(self.family("RL", updates=G2["updates"]))
        return scored, ranked(scored, "macro")[:2]

    def select_rl_star(self):
        _, continuers = self.select_rl_continuers()
        final = self.family("RL", updates=FINAL_UPDATES, lrs=[c["lr"] for c in continuers])
        scored = self.h_scored(final)
        return scored, self.endpoints[best(scored, "macro")["run_id"]]

    def select_lim_t(self):
        _, rl = self.select_rl_star()
        winner = self.endpoints[self.stage_a()["winner"]]
        (lim,) = self.bound_to_t_rl(self.family("LIM-T", lrs=[winner["lr"]]), rl)
        return lim

    # Gates.
    def gate_pi(self):
        return gate_pi(self.evidence["pi"], self.s_b_ids)

    def gate_b0(self):
        if "pi" not in self.evidence:
            raise ValueError("P0' needs the P0-I document (same 400 candidates)")
        pi_ids = [e["id"] for e in self.evidence["pi"]["journal"]]
        return gate_b0(self.evidence["p0_prime"], self.s_b_ids, self.i_star()["adapter_sha256"],
                       pi_ids)

    def gate_a(self):
        selection = self.stage_a()
        winner, c_star = (self.endpoints[selection[k]] for k in ("winner", "c_star"))
        comparison = self.compare(winner, c_star, "h_select", GATE_A["lb_quantile"])
        tries = {e["splits"].get("h_select_constrained", {}).get("trie_sha256")
                 for e in (winner, c_star)}
        if len(tries) != 1 or None in tries:
            raise ValueError("Gate A constrained greedy needs one recorded trie_sha256 for both")
        constrained = self.constrained_macro(winner) - self.constrained_macro(c_star)
        lb, plain_lb, scale = small_n_lower_bound(comparison, self.rows["h_select"])
        stage_a = [self.endpoints[c["run_id"]] for arm in selection["table"].values()
                   for c in arm["h_select"]]
        checks = {
            "winner_minus_c_star_at_least_0_03": at_least(
                comparison["mean_class_accuracy_difference"], GATE_A["margin"]),
            "bonferroni_paired_lb_positive": above(lb, 0),
            "constrained_greedy_winner_minus_c_star_positive": above(constrained, 0),
            "winner_validity_at_least_c_star": at_least(
                comparison["candidate"]["valid_fraction"],
                comparison["comparator"]["valid_fraction"]),
            **self.identity(stage_a),
        }
        del checks["rl_reference_is_init"], checks["rl_profile_recorded"]  # no RL in Stage A
        return {"pass": all(checks.values()), "checks": checks,
                "c_star": c_star["run_id"], "winner": winner["run_id"],
                "winner_arm": selection["winner_arm"], "i_star": selection["i_star"],
                "stage_a": selection["table"],
                "values": {"winner_minus_c_star": comparison["mean_class_accuracy_difference"],
                           "lb_quantile": GATE_A["lb_quantile"], "paired_lb": lb,
                           "paired_lb_uncorrected": plain_lb, "small_n_scale": scale,
                           "constrained_winner_minus_c_star": constrained,
                           "winner_valid": comparison["candidate"]["valid_fraction"],
                           "c_star_valid": comparison["comparator"]["valid_fraction"]}}

    def gate_g2(self):
        scored, continuers = self.select_rl_continuers()
        top, i_star = self.endpoints[continuers[0]["run_id"]], self.i_star()
        comparison = self.compare(top, i_star, "h_select")
        arms = [self.endpoints[item["run_id"]] for item in scored]
        if len(top["useful_mixed"]) != G2["updates"]:
            raise ValueError("G2 needs one useful-mixed flag per accepted update")
        window = top["useful_mixed"][-G2["rolling_window"]:]
        rolling = sum(window) / len(window)
        invalid = comparison["candidate"]["invalid_completions_including_missing"]
        invalid_i = comparison["comparator"]["invalid_completions_including_missing"]
        slack = invalid_slack(invalid, invalid_i)
        checks = {
            "h_select_delta_vs_i_star_positive": above(
                comparison["mean_class_accuracy_difference"], 0),
            "invalid_within_i_star_plus_slack": invalid <= invalid_i + slack,
            **self.identity(arms),
            "rolling_useful_mixed": at_least(rolling, G2["rolling_mixed"]),
        }
        return {"label": "futility stop, not evidence", "pass": all(checks.values()),
                "checks": checks, "best": top["run_id"], "h_select": scored,
                "continue_to_300": [c["run_id"] for c in continuers],
                "values": {"h_select_delta_vs_i_star":
                           comparison["mean_class_accuracy_difference"],
                           "invalid": invalid, "i_star_invalid": invalid_i,
                           "invalid_slack": slack, "trailing_useful_mixed": rolling}}

    def gate_g3a(self):
        rl_scored, rl = self.select_rl_star()
        sft_scored, sft, qualifying = self.select_sft_e()
        difference = self._h_gate_difference(rl, sft)
        versus_i = self.compare(rl, self.i_star(), "h_gate", G3["lb_quantile"])
        lb_i, plain_lb_i, _ = small_n_lower_bound(versus_i, self.rows["h_gate"])
        valid = versus_i["candidate"]["valid_fraction"]
        eos = rl["splits"]["h_gate"]["eos_fraction"]
        checks = {
            "rl_minus_sft_e_at_least_0_05": at_least(difference, G3["margin"]),
            "rl_minus_i_star_one_sided_90_lb_positive": above(lb_i, 0),
            "valid_held": at_least(valid, G3["valid"]),
            "eos_held": at_least(eos, G3["eos"]),
            **self.identity([self.endpoints[c["run_id"]] for c in rl_scored]
                            + [self.endpoints[c["run_id"]] for c in sft_scored]),
        }
        return {"pass": all(checks.values()), "checks": checks, "margin": G3["margin"],
                "rl_star": rl["run_id"], "sft_e_star": sft["run_id"],
                "rl_h_select_at_300": rl_scored, "sft_e_h_select": sft_scored,
                "sft_t_lrs": qualifying,
                "values": {"rl_minus_sft_e": difference,
                           "rl_minus_i_star": versus_i["mean_class_accuracy_difference"],
                           "rl_minus_i_star_lb90": lb_i,
                           "rl_minus_i_star_lb90_uncorrected": plain_lb_i,
                           "rl_valid": valid, "rl_eos": eos}}

    def gate_g3b(self):
        """SFT-T* as freshpool's G3b, and (amendment 1, design-review finding 5) LIM-T at the
        same mechanism margin: the claim rests on the robust verdict, so LIM-T gates too."""
        result = super().gate_g3b()
        _, rl = self.select_rl_star()
        lim = self.select_lim_t()
        difference = self._h_gate_difference(rl, lim)
        result["checks"]["rl_minus_lim_t_at_least_margin"] = at_least(
            difference, self.branch["g3_margin"])
        # One identity verdict over every endpoint G3b first uses (SFT-T and LIM-T).
        result["checks"] |= self.identity(
            [*(self.endpoints[c["run_id"]] for c in result["sft_t_h_select"]), lim])
        result["pass"] = all(result["checks"].values())
        result["lim_t"] = {"run_id": lim["run_id"], "gating": True,
                           "rl_minus_lim_t": difference}
        result["values"]["rl_minus_lim_t"] = difference
        return result

    def protocol_pool(self):
        table = self.stage_a()["table"]
        _, _, qualifying = self.select_sft_e()
        close = [self.endpoints[c["run_id"]] for arm in COMPARATOR_ARMS
                 for c in within(table[arm]["h_select"], "macro", POOL_WINDOW)]
        pool = [self.w, *self.controlled400(), *close, self.i_star(),
                *self.exact_updates(self.family("SFT-E")),
                *self.bound_to_t_rl(self.family("SFT-T", lrs=qualifying),
                                    self.select_rl_star()[1])]
        return list({e["run_id"]: e for e in pool}.values())

    def robust_pool(self):
        table = self.stage_a()["table"]
        extra = [self.select_lim_t(),
                 *(self.endpoints[table[arm]["star"]["run_id"]] for arm in LIMIT_ARMS)]
        return list({e["run_id"]: e for e in [*self.protocol_pool(), *extra]}.values())

    def gate_g4(self):
        _, rl = self.select_rl_star()
        protocol = self.substantial(rl, self.protocol_pool(), "val")
        robust = self.substantial(rl, self.robust_pool(), "val")
        identity = self.identity([e for e in self.robust_pool() if e["arm"] != "W"] + [rl])
        realised = max(item["accuracy"] for item in robust["pool"])
        return {"pass": protocol["pass"] and robust["pass"] and all(identity.values()),
                "claim_rests_on": "robust", "identity": identity,
                "protocol": protocol, "robust": robust, "rl_star": rl["run_id"],
                "d_req_realised": d_req(realised)}

    def gate_g5(self, g4):
        _, rl_star = self.select_rl_star()
        _, _, qualifying = self.select_sft_e()
        deltas = {str(PRIMARY_SEED): g4["robust"]["values"]["delta_vs_strongest_sft"]}
        # The robust pool per seed (amendment 1, design-review finding 5): every member of
        # the seed-42 robust pool that no replicate reruns (W, controlled400, Stage-A
        # endpoints incl. I* and A2*), plus the seed's own SFT-E, SFT-T and LIM-T.
        fixed = [e for e in self.robust_pool() if e["arm"] not in SEED_BOUND_ARMS]
        winner = self.endpoints[self.stage_a()["winner"]]
        per_seed, used = {}, []
        for seed in REPLICATE_SEEDS:
            (rl,) = self.family("RL", seed, FINAL_UPDATES, lrs=[rl_star["lr"]])
            sft = [*self.exact_updates(self.family("SFT-E", seed, lrs=qualifying)),
                   *self.bound_to_t_rl(self.family("SFT-T", seed, lrs=qualifying), rl),
                   *self.bound_to_t_rl(self.family("LIM-T", seed, lrs=[winner["lr"]]), rl)]
            used += [rl, *sft]
            pool = [*fixed, *sft]
            scored = [{"run_id": e["run_id"],
                       "committed_active_seconds": e["committed_active_seconds"],
                       "accuracy": self.val(e)} for e in pool]
            strongest = best(scored, "accuracy")
            deltas[str(seed)] = self.val(rl) - strongest["accuracy"]
            per_seed[str(seed)] = {"rl": rl["run_id"], "strongest_sft": strongest["run_id"],
                                   "pool": scored}
        mean = sum(deltas.values()) / len(deltas)
        checks = {"every_seed_delta_positive": all(above(d, 0) for d in deltas.values()),
                  "three_seed_mean_at_least_0_05": at_least(mean, FRESH["SUBSTANTIAL"][
                      "replicate_mean"]),
                  **self.identity(used)}
        return {"pass": all(checks.values()), "checks": checks, "deltas": deltas,
                "mean": mean, "seeds": per_seed, "sft_lrs": qualifying}

    def gate_g6(self, g4):
        return super().gate_g6({"rl_star": g4["rl_star"],
                                "protocol": {"strongest_sft": g4["robust"]["strongest_sft"]}})


def evaluate(evidence):
    """Run the gates in order. The first failure stops the study; the first gate with no
evidence is where it waits. Later evidence is listed, never evaluated."""
    study = Study(evidence)
    endpoints = list(study.endpoints.values())
    order = [
        ("PI", "pi" in evidence, lambda _: study.gate_pi()),
        ("GA", any(e["arm"] in STAGE_A_ARMS for e in endpoints), lambda _: study.gate_a()),
        ("GB0", "p0_prime" in evidence, lambda _: study.gate_b0()),
        ("G2", bool(study.find("RL", updates=G2["updates"])), lambda _: study.gate_g2()),
        ("G3a", any(study.has(e, "h_gate") for e in study.find("SFT-E")),
         lambda _: study.gate_g3a()),
        ("G3b", any(study.has(e, "h_gate") for e in study.find("SFT-T")),
         lambda _: study.gate_g3b()),
        ("G4", any(study.has(e, "val") for e in study.find("LIM-T")),
         lambda _: study.gate_g4()),
        ("G5", any(e["arm"] == "RL" and e["seed"] in REPLICATE_SEEDS for e in endpoints),
         lambda done: study.gate_g5(done["G4"])),
        ("G6", any(study.has(e, "test") for e in endpoints),
         lambda done: study.gate_g6(done["G4"])),
    ]
    gates, failed, waiting, after = {}, None, None, []
    for name, present, run in order:
        if failed or waiting:
            gates[name] = {"status": "not_evaluated",
                           "reason": f"failed at {failed}" if failed else f"waiting at {waiting}"}
            after += [name] if present else []
        elif not present:
            gates[name], waiting = {"status": "awaiting_evidence"}, name
        else:
            result = run(gates)
            if result.get("awaiting_evidence"):
                gates[name], waiting = {"status": "partially_evaluated", **result}, name
            else:
                gates[name] = {"status": "evaluated", **result}
                failed = None if result["pass"] else name
    passed = {name: gates[name].get("pass", False) for name in gates}
    rl_test = (passed["G4"] and passed["G5"]
               and gates["G6"].get("substantial_rl_test", {}).get("pass", False))
    return {"study_id": "aircraft-rl-next-study-v1", "design_id": "aircraft-signal-limit-v1",
            "a_floor": study.a_floor, "s_b_size": len(study.s_b_ids),
            "gates": gates, "failed_at": failed, "awaiting": waiting,
            "evidence_after_stop": after,
            "substantial_val_and_replication": passed["G4"] and passed["G5"],
            "rl_test_gates": rl_test,
            "final_claim": passed["G4"] and passed["G5"] and passed["G6"],
            "scope": "Gate arithmetic over recorded predictions only; H results are native-TRAIN "
                     "development evidence; thresholds are the frozen study and protocol values"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.evidence.read_text()))
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"failed_at": report["failed_at"], "awaiting": report["awaiting"]}))


if __name__ == "__main__":
    main()
