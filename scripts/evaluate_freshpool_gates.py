"""Evaluate aircraft-freshpool-screened-grpo-v1 selections and gates G1-G6 from evidence.

Pure evaluation of already recorded outcomes: no model, image or accelerator access.
Thresholds are copied from runs/rl-next-study-design-v1/study.json and the unchanged
protocol ``substantial_gain_gates``; none is a parameter. Gates run in order and stop
at the first failure. Evidence for a later gate after a stop is reported, never used.

Evidence document (JSON):
  labels                ordered canonical vocabulary
  rows                  {"h_select"|"h_gate"|"val"|"test": [{"id", "label"}]}
  a_star                max controlled400 VAL accuracy (B2: decides D_req and G3 margin)
  b4_edge_extension     bool; true adds 2e-4 to both LR grids
  g1                    {"seed": 42, "adapter_sha256", "scan_order": [S_scan_seed42 ids],
                         "journal": [...400], "rollouts": [...]} (see gate_g1)
  endpoints             [{"run_id", "arm", "lr", "seed", "updates",
                          "committed_active_seconds", "adapter_sha256",
                          "reference_sha256", "useful_mixed", "splits": {...}}]
    arm is one of W, RL, SFT-E, SFT-T, EG-SFT, controlled400-SFT. The controlled400 SFT
    endpoints must be exactly the six run IDs in CONTROLLED400_SFT (B3: all enter the pool).
    splits.h_select/h_gate: {"predictions": {id: [completions]}, "eos_fraction"}
    splits.val/test:        {"predictions": {id: label | null}}
  named_baseline_test   optional {"run_id", "predictions", "valid_eos_fraction"}

Budgets refused on mismatch: SFT-E and EG-SFT exactly 600 updates; SFT-T at least T_RL,
the committed active seconds of the same seed's RL endpoint at RL*'s LR and 600 updates.
Every non-W endpoint must have an adapter different from W's, and every RL endpoint must
keep W's adapter as its KL reference; each gate checks the endpoints it first uses.

Interpretations to ratify before activation (the study text admits more than one reading;
each is a choice made here, not a quotation):
  * "Sampled macro" on H-select / H-gate is the mean-class accuracy over EVERY completion
    per image, greedy included (5 on H-select: greedy + one G4 seed; 9 on H-gate: greedy +
    two G4 seeds), each image's completions resampled together. The other reading, only the
    sampled G4 completions without greedy, would change G2, G3a, G3b and the SFT-T window.
  * G2's rolling useful-mixed fraction is over the last 100 accepted updates.
  * G3a "validity and EOS held" reuses G1's 0.90 and 0.95.
  * G1's 64 rollout images are any 64 of the first 400 scanned candidates, as the study
    says ("64 of them"); they are not required to be accepted ones.

Blind spots, stated: this cannot see whether a VAL/TEST look was the first one (the
look ledger is the phase worker's), whether predictions came from the stated adapter,
or whether ``committed_active_seconds`` is honest. It trusts those identities. It checks
that G1's ``scan_order`` is random.Random(42).shuffle(sorted(ids)) of its own ID set, not
that the set equals the S_scan_seed42 manifest (the manifest SHA256 is the phase
worker's). SFT-T is bounded below by T_RL only: overshoot past the first checkpoint at or
after T_RL is not visible here. Run IDs, not adapter bytes, identify controlled400 endpoints.
"""

import argparse
import json
import random
from pathlib import Path

from vision_rl.classification_comparison import (
    compare_classification,
    compare_repeated_classification,
)

# Absolute slack for float comparisons, far below one image in any split here.
EPSILON = 1e-9
GRID = (1e-5, 5e-5, 1e-4)
B4_LR = 2e-4
PRIMARY_SEED, REPLICATE_SEEDS = 42, (314, 2718)
COMPLETIONS = {"h_select": 5, "h_gate": 9}  # greedy + 1 G4 seed; greedy + 2 G4 seeds
ARMS = {"W", "RL", "SFT-E", "SFT-T", "EG-SFT", "controlled400-SFT"}
G1 = {"candidates": 400, "rollout_images": 64, "rollout_seeds": (42, 314), "group": 4,
      "acceptance": 0.30, "mean_u_accepted": 0.60, "mixed_tolerance": 0.10,
      "valid": 0.90, "eos": 0.95, "scorer_seconds": 2.0, "u_threshold": 0.5}
G2 = {"updates": 150, "invalid_slack": 2, "rolling_window": 100, "rolling_mixed": 0.6}
G3 = {"full_margin": 0.02, "mechanism_margin": 0.05, "lb_quantile": 0.10,
      "valid": G1["valid"], "eos": G1["eos"]}
SFT_T_WINDOW = 0.02
SUBSTANTIAL = {"accuracy": 0.80, "vs_init": 0.05, "vs_sft": 0.05, "rer": 0.20,
               "lb_quantile": 0.025, "replicate_mean": 0.05}
FRONTIER = {"delta": 0.10, "rer": 0.25, "lb": 0.05, "valid_eos": 0.95}
FINAL_UPDATES = 600
# runs/aircraft-controlled400-v1-plan: answer-SFT at three LRs, exposure- and time-matched.
CONTROLLED400_SFT = frozenset(f"sft-{lr}-{match}" for lr in ("1e5", "5e5", "1e4")
                              for match in ("exposure", "time"))


def at_least(value, threshold):
    return value >= threshold - EPSILON


def above(value, threshold):
    return value > threshold + EPSILON


def u(p):
    """Probability that a G4 group of exact-match rewards is mixed."""
    return 1 - p**4 - (1 - p) ** 4


def d_req(a_star):
    """Required VAL delta and the B2 mode it implies; a lower bound on the real D_req."""
    if not isinstance(a_star, (int, float)) or isinstance(a_star, bool) or not 0 <= a_star <= 1:
        raise ValueError("a_star must be an accuracy in [0, 1]")
    required = max(0.05, 0.2 * (1 - a_star), 0.80 - a_star)
    mechanism = above(required, 0.10)
    return {"a_star": a_star, "d_req": required,
            "mode": "mechanism" if mechanism else "full",
            "g3_margin": G3["mechanism_margin"] if mechanism else G3["full_margin"]}


def best(candidates, key):
    """Protocol tie rule: highest key, then lower active compute, then run ID."""
    if not candidates:
        raise ValueError("no candidates to select from")
    top = max(item[key] for item in candidates)
    tied = [item for item in candidates if at_least(item[key], top)]
    return min(tied, key=lambda item: (item["committed_active_seconds"], item["run_id"]))


def ranked(candidates, key):
    remaining, order = list(candidates), []
    while remaining:
        order.append(best(remaining, key))
        remaining.remove(order[-1])
    return order


def within(candidates, key, window):
    """Every candidate within ``window`` of the best, best included."""
    top = best(candidates, key)[key]
    return [item for item in candidates if at_least(item[key], top - window)]


def scan_order(ids, seed):
    """S_scan rule from study.json: random.Random(seed).shuffle(sorted(ids))."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order


def gate_g1(g1, w_adapter):
    journal, rollouts = g1["journal"], g1["rollouts"]
    if g1.get("seed") != PRIMARY_SEED or g1.get("adapter_sha256") != w_adapter:
        raise ValueError("G1 must run at seed 42 under the W adapter")
    order = g1.get("scan_order") or []
    if len(set(order)) != len(order) or order != scan_order(order, PRIMARY_SEED):
        raise ValueError("G1 scan_order is not random.Random(42).shuffle(sorted(ids))")
    if [e["id"] for e in journal] != order[:G1["candidates"]] or len(journal) != G1[
            "candidates"]:
        raise ValueError("G1 needs the first 400 scanned candidates in scan order")
    if any(e.get("adapter_sha256") != w_adapter for e in journal):
        raise ValueError("G1 journal entries must be scored under the W adapter")
    for entry in journal:
        if abs(entry["u"] - u(entry["p"])) > EPSILON or entry["accepted"] != (
            entry["u"] >= G1["u_threshold"]
        ):
            raise ValueError(f"journal entry {entry['id']} disagrees with u(p) >= 0.5")
    scored = {entry["id"]: entry for entry in journal}
    images = {group["id"] for group in rollouts}
    expected = {(i, s) for i in images for s in G1["rollout_seeds"]}
    if (
        len(images) != G1["rollout_images"]
        or images - set(scored)
        or {(g["id"], g["seed"]) for g in rollouts} != expected
        or len(rollouts) != len(expected)
        or any(len(g["completions"]) != G1["group"] for g in rollouts)
    ):
        raise ValueError("G1 needs G4 groups at seeds 42 and 314 on 64 scanned candidates")
    accepted = [entry for entry in journal if entry["accepted"]]
    completions = [c for g in rollouts for c in g["completions"]]
    correct = [c for c in completions if c["correct"]]
    mixed = sum(len({c["correct"] for c in g["completions"]}) == 2 for g in rollouts)
    values = {
        "acceptance": len(accepted) / len(journal),
        "mean_u_accepted": sum(e["u"] for e in accepted) / len(accepted) if accepted else 0.0,
        "observed_useful_mixed": mixed / len(rollouts),
        "predicted_useful_mixed": sum(u(scored[i]["p"]) for i in images) / len(images),
        "valid": sum(c["valid"] for c in completions) / len(completions),
        "eos": sum(c["eos"] for c in completions) / len(completions),
        "scorer_seconds_per_candidate": sum(e["scorer_seconds"] for e in journal) / len(journal),
        "non_canonical_share_of_correct": (
            sum(not c["canonical_tokenization"] for c in correct) / len(correct)
            if correct else None
        ),
    }
    gap = abs(values["observed_useful_mixed"] - values["predicted_useful_mixed"])
    checks = {
        "acceptance": at_least(values["acceptance"], G1["acceptance"]),
        "mean_u_accepted": at_least(values["mean_u_accepted"], G1["mean_u_accepted"]),
        "useful_mixed_calibrated": not above(gap, G1["mixed_tolerance"]),
        "valid": at_least(values["valid"], G1["valid"]),
        "eos": at_least(values["eos"], G1["eos"]),
        "scorer_seconds": not above(values["scorer_seconds_per_candidate"],
                                    G1["scorer_seconds"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "values": values}


class Study:
    def __init__(self, evidence):
        self.evidence, self.labels, self.rows = evidence, evidence["labels"], evidence["rows"]
        if type(evidence.get("b4_edge_extension")) is not bool:
            raise ValueError("b4_edge_extension must be stated as true or false")
        self.grid = set(GRID) | ({B4_LR} if evidence["b4_edge_extension"] else set())
        self.branch = d_req(evidence["a_star"])
        self.endpoints = {}
        for endpoint in evidence.get("endpoints", []):
            if endpoint["arm"] not in ARMS or endpoint["run_id"] in self.endpoints:
                raise ValueError(f"unknown arm or duplicate run ID {endpoint['run_id']}")
            self.endpoints[endpoint["run_id"]] = endpoint
        warm = self.find("W", seed=None)
        if len(warm) != 1:
            raise ValueError("evidence needs exactly one W endpoint")
        self.w, self._h, self._val = warm[0], {}, {}

    def find(self, arm, seed=PRIMARY_SEED, updates=None):
        """Endpoints of one arm; ``seed=None`` means any seed (W, controlled400)."""
        return [e for e in self.endpoints.values() if e["arm"] == arm
                and (seed is None or e["seed"] == seed)
                and (updates is None or e["updates"] == updates)]

    def family(self, arm, seed=PRIMARY_SEED, updates=None, lrs=None):
        found = self.find(arm, seed, updates)
        lrs = self.grid if lrs is None else set(lrs)
        if sorted(e["lr"] for e in found) != sorted(lrs):
            raise ValueError(
                f"{arm} at seed {seed} needs exactly one endpoint per LR {sorted(lrs)}")
        return found

    def identity(self, endpoints):
        """Per-arm adapter/reference checks over every endpoint a gate first uses."""
        return {
            "adapter_changed": all(e["adapter_sha256"] != self.w["adapter_sha256"]
                                   for e in endpoints),
            "reference_unchanged": all(e["reference_sha256"] == self.w["adapter_sha256"]
                                       for e in endpoints if e["arm"] == "RL"),
        }

    def exact_updates(self, endpoints):
        for e in endpoints:
            if e["updates"] != FINAL_UPDATES:
                raise ValueError(f"{e['arm']} {e['run_id']} needs exactly {FINAL_UPDATES} "
                                 f"updates, has {e['updates']}")
        return endpoints

    def bound_to_t_rl(self, endpoints, rl):
        """SFT-T runs to T_RL: RL's committed active seconds at the same seed."""
        for e in endpoints:
            if type(e["updates"]) is not int or e["updates"] < 1 or not at_least(
                    e["committed_active_seconds"], rl["committed_active_seconds"]):
                raise ValueError(f"SFT-T {e['run_id']} must reach T_RL "
                                 f"{rl['committed_active_seconds']} s of {rl['run_id']}")
        return endpoints

    def controlled400(self):
        found = self.find("controlled400-SFT", seed=None)
        if {e["run_id"] for e in found} != CONTROLLED400_SFT:
            raise ValueError(f"protocol pool needs exactly the controlled400 SFT endpoints "
                             f"{sorted(CONTROLLED400_SFT)}")
        return found

    def has(self, endpoint, split):
        return split in endpoint.get("splits", {})

    def predictions(self, endpoint, split):
        if not self.has(endpoint, split):
            raise ValueError(f"{endpoint['run_id']} has no {split} predictions")
        return endpoint["splits"][split]["predictions"]

    def h(self, endpoint, split, quantile=G3["lb_quantile"]):
        """Repeated-completion comparison of an endpoint against W on an H split."""
        key = (endpoint["run_id"], split, quantile)
        if key not in self._h:
            result = compare_repeated_classification(
                self.rows[split], self.predictions(endpoint, split),
                self.predictions(self.w, split), self.labels,
                completions=COMPLETIONS[split], lower_quantile=quantile)
            self._h[key] = result
        return self._h[key]

    def h_scored(self, endpoints, split="h_select"):
        return [{"run_id": e["run_id"], "lr": e["lr"],
                 "committed_active_seconds": e["committed_active_seconds"],
                 "macro": self.h(e, split)["candidate"]["mean_class_accuracy"]}
                for e in endpoints]

    def val(self, endpoint, split="val"):
        """Full-split mean-class accuracy, scored by the shared comparison module."""
        key = (endpoint["run_id"], split)
        if key not in self._val:
            result = compare_classification(
                self.rows[split], self.predictions(endpoint, split),
                self.predictions(self.w, split), self.labels, resamples=100)
            self._val[key] = result["candidate"]["mean_class_accuracy"]
        return self._val[key]

    # Selection on H-select only.
    def select_rl_continuers(self):
        scored = self.h_scored(self.family("RL", updates=G2["updates"]))
        return scored, ranked(scored, "macro")[:2]

    def select_rl_star(self):
        _, continuers = self.select_rl_continuers()
        final = self.family("RL", updates=FINAL_UPDATES, lrs=[c["lr"] for c in continuers])
        scored = self.h_scored(final)
        return scored, self.endpoints[best(scored, "macro")["run_id"]]

    def select_sft_e(self, seed=PRIMARY_SEED):
        scored = self.h_scored(self.exact_updates(self.family("SFT-E", seed)))
        star = best(scored, "macro")
        qualifying = sorted(c["lr"] for c in within(scored, "macro", SFT_T_WINDOW))
        return scored, self.endpoints[star["run_id"]], qualifying

    def select_sft_t(self):
        _, _, qualifying = self.select_sft_e()
        _, rl = self.select_rl_star()
        scored = self.h_scored(self.bound_to_t_rl(self.family("SFT-T", lrs=qualifying), rl))
        return scored, self.endpoints[best(scored, "macro")["run_id"]]

    # Gates.
    def gate_g2(self):
        scored, continuers = self.select_rl_continuers()
        top = self.endpoints[continuers[0]["run_id"]]
        comparison = self.h(top, "h_select")
        arms = [self.endpoints[item["run_id"]] for item in scored]
        if len(top["useful_mixed"]) != G2["updates"]:
            raise ValueError("G2 needs one useful-mixed flag per accepted update")
        window = top["useful_mixed"][-G2["rolling_window"]:]
        rolling = sum(window) / len(window)
        checks = {
            "h_select_delta_vs_w_positive": above(comparison["mean_class_accuracy_difference"], 0),
            "invalid_within_w_plus_2": comparison["candidate"][
                "invalid_completions_including_missing"]
            <= comparison["comparator"]["invalid_completions_including_missing"]
            + G2["invalid_slack"],
            **self.identity(arms),
            "rolling_useful_mixed": at_least(rolling, G2["rolling_mixed"]),
        }
        return {"label": "futility stop, not evidence", "pass": all(checks.values()),
                "checks": checks, "best": top["run_id"], "h_select": scored,
                "continue_to_600": [c["run_id"] for c in continuers],
                "values": {"h_select_delta_vs_w": comparison["mean_class_accuracy_difference"],
                           "trailing_useful_mixed": rolling}}

    def _h_gate_difference(self, rl, sft):
        a = self.h(rl, "h_gate")["candidate"]["mean_class_accuracy"]
        b = self.h(sft, "h_gate")["candidate"]["mean_class_accuracy"]
        return a - b

    def gate_g3a(self):
        rl_scored, rl = self.select_rl_star()
        sft_scored, sft, qualifying = self.select_sft_e()
        margin = self.branch["g3_margin"]
        difference = self._h_gate_difference(rl, sft)
        versus_w = self.h(rl, "h_gate", G3["lb_quantile"])
        valid = versus_w["candidate"]["valid_fraction"]
        eos = rl["splits"]["h_gate"]["eos_fraction"]
        checks = {
            "rl_minus_sft_e_at_least_margin": at_least(difference, margin),
            "rl_minus_w_one_sided_90_lb_positive": above(
                versus_w["uncertainty"]["lower_bound"], 0),
            "valid_held": at_least(valid, G3["valid"]),
            "eos_held": at_least(eos, G3["eos"]),
            **self.identity([self.endpoints[c["run_id"]] for c in rl_scored]
                            + [self.endpoints[c["run_id"]] for c in sft_scored]),
        }
        return {"pass": all(checks.values()), "checks": checks, "mode": self.branch["mode"],
                "margin": margin, "rl_star": rl["run_id"], "sft_e_star": sft["run_id"],
                "rl_h_select_at_600": rl_scored, "sft_e_h_select": sft_scored,
                "sft_t_lrs": qualifying,
                "values": {"rl_minus_sft_e": difference,
                           "rl_minus_w": versus_w["mean_class_accuracy_difference"],
                           "rl_minus_w_lb90": versus_w["uncertainty"]["lower_bound"],
                           "rl_valid": valid, "rl_eos": eos}}

    def gate_g3b(self):
        _, rl = self.select_rl_star()
        scored, sft = self.select_sft_t()
        difference = self._h_gate_difference(rl, sft)
        margin = self.branch["g3_margin"]
        checks = {"rl_minus_sft_t_at_least_margin": at_least(difference, margin),
                  **self.identity([self.endpoints[c["run_id"]] for c in scored])}
        return {"pass": all(checks.values()), "checks": checks, "mode": self.branch["mode"],
                "margin": margin, "sft_t_star": sft["run_id"], "sft_t_h_select": scored,
                "values": {"rl_minus_sft_t": difference}}

    def protocol_pool(self):
        _, _, qualifying = self.select_sft_e()
        return [self.w, *self.family("SFT-E"), *self.family("SFT-T", lrs=qualifying),
                *self.controlled400()]

    def substantial(self, rl, pool, split):
        scored = [{"run_id": e["run_id"],
                   "committed_active_seconds": e["committed_active_seconds"],
                   "accuracy": self.val(e, split)} for e in pool]
        strongest = self.endpoints[best(scored, "accuracy")["run_id"]]
        return self.substantial_against(rl, strongest, split) | {"pool": scored}

    def substantial_against(self, rl, strongest, split):
        result = compare_classification(
            self.rows[split], self.predictions(rl, split),
            self.predictions(strongest, split), self.labels)
        accuracy = result["candidate"]["mean_class_accuracy"]
        versus_w = accuracy - self.val(self.w, split)
        delta, rer = result["mean_class_accuracy_difference"], result["relative_error_reduction"]
        lb = result["uncertainty"]["stratified_image_ci95"]
        checks = {
            "accuracy_at_least_0_80": at_least(accuracy, SUBSTANTIAL["accuracy"]),
            "delta_vs_w_at_least_0_05": at_least(versus_w, SUBSTANTIAL["vs_init"]),
            "delta_vs_strongest_sft_at_least_0_05": at_least(delta, SUBSTANTIAL["vs_sft"]),
            "relative_error_reduction_at_least_0_20": rer is not None
            and at_least(rer, SUBSTANTIAL["rer"]),
            "paired_95_lb_positive": lb is not None and above(lb[0], 0),
        }
        return {"pass": all(checks.values()), "checks": checks,
                "strongest_sft": strongest["run_id"],
                "values": {"rl_accuracy": accuracy, "delta_vs_w": versus_w,
                           "delta_vs_strongest_sft": delta, "relative_error_reduction": rer,
                           "paired_95_ci": lb,
                           "all_predictions_recorded": result["all_predictions_recorded"]}}

    def gate_g4(self):
        _, rl = self.select_rl_star()
        _, sft_e, _ = self.select_sft_e()
        (eg,) = self.exact_updates(self.family("EG-SFT", lrs=[sft_e["lr"]]))
        protocol = self.substantial(rl, self.protocol_pool(), "val")
        robust = self.substantial(rl, [*self.protocol_pool(), eg], "val")
        eg_accuracy = self.val(eg)
        identity = self.identity([eg, *self.controlled400()])
        return {"pass": protocol["pass"] and all(identity.values()), "identity": identity,
                "protocol": protocol, "robust": robust,
                "rl_star": rl["run_id"],
                "eg_sft_at_least_rl": at_least(eg_accuracy, protocol["values"]["rl_accuracy"]),
                "statement": "no evidence RL adds anything beyond reweighting"
                if at_least(eg_accuracy, protocol["values"]["rl_accuracy"]) else None}

    def gate_g5(self, g4):
        _, rl_star = self.select_rl_star()
        _, _, qualifying = self.select_sft_e()
        deltas = {str(PRIMARY_SEED): g4["protocol"]["values"]["delta_vs_strongest_sft"]}
        per_seed, used = {}, []
        for seed in REPLICATE_SEEDS:
            (rl,) = self.family("RL", seed, FINAL_UPDATES, lrs=[rl_star["lr"]])
            sft = [*self.exact_updates(self.family("SFT-E", seed, lrs=qualifying)),
                   *self.bound_to_t_rl(self.family("SFT-T", seed, lrs=qualifying), rl)]
            used += [rl, *sft]
            pool = [self.w, *sft]
            scored = [{"run_id": e["run_id"],
                       "committed_active_seconds": e["committed_active_seconds"],
                       "accuracy": self.val(e)} for e in pool]
            strongest = best(scored, "accuracy")
            deltas[str(seed)] = self.val(rl) - strongest["accuracy"]
            per_seed[str(seed)] = {"rl": rl["run_id"], "strongest_sft": strongest["run_id"],
                                   "pool": scored}
        mean = sum(deltas.values()) / len(deltas)
        checks = {"every_seed_delta_positive": all(above(d, 0) for d in deltas.values()),
                  "three_seed_mean_at_least_0_05": at_least(mean, SUBSTANTIAL["replicate_mean"]),
                  **self.identity(used)}
        return {"pass": all(checks.values()), "checks": checks, "deltas": deltas,
                "mean": mean, "seeds": per_seed, "sft_lrs": qualifying}

    def gate_g6(self, g4):
        rl = self.endpoints[g4["rl_star"]]
        strongest = self.endpoints[g4["protocol"]["strongest_sft"]]
        rl_gates = self.substantial_against(rl, strongest, "test")
        baseline = self.evidence.get("named_baseline_test")
        waiting = None
        if baseline is None:
            frontier = {"status": "incomplete", "pass": False,
                        "reason": "no named-baseline TEST predictions; no frontier claim"}
            waiting = "named_baseline_test" if rl_gates["pass"] else None
        else:
            result = compare_classification(self.rows["test"], self.predictions(rl, "test"),
                                             baseline["predictions"], self.labels)
            rer, ci = result["relative_error_reduction"], result["uncertainty"][
                "stratified_image_ci95"]
            interface = at_least(baseline["valid_eos_fraction"], FRONTIER["valid_eos"])
            checks = {
                "baseline_valid_eos_at_least_0_95": interface,
                "delta_at_least_0_10": at_least(result["mean_class_accuracy_difference"],
                                                FRONTIER["delta"]),
                "relative_error_reduction_at_least_0_25": rer is not None
                and at_least(rer, FRONTIER["rer"]),
                "paired_95_lb_above_0_05": ci is not None and above(ci[0], FRONTIER["lb"]),
            }
            frontier = {"status": "complete" if interface else "interface_failure",
                        "pass": all(checks.values()), "checks": checks,
                        "baseline": baseline["run_id"],
                        "values": {"delta": result["mean_class_accuracy_difference"],
                                   "relative_error_reduction": rer, "paired_95_ci": ci}}
        # protocol final_claim: native-test RL gates AND named-baseline gates.
        return {"pass": rl_gates["pass"] and frontier["pass"], "awaiting_evidence": waiting,
                "substantial_rl_test": rl_gates, "named_baseline": frontier,
                "contenders": [rl["run_id"], strongest["run_id"], self.w["run_id"]]}


def evaluate(evidence):
    """Run the gates in order. The first failure stops the study; the first gate with
no evidence is where it waits. Later evidence is listed, never evaluated."""
    study = Study(evidence)
    endpoints = list(study.endpoints.values())
    order = [
        ("G1", "g1" in evidence,
         lambda _: gate_g1(evidence["g1"], study.w["adapter_sha256"])),
        ("G2", bool(study.find("RL", updates=G2["updates"])), lambda _: study.gate_g2()),
        ("G3a", any(study.has(e, "h_gate") for e in study.find("SFT-E")),
         lambda _: study.gate_g3a()),
        ("G3b", any(study.has(e, "h_gate") for e in study.find("SFT-T")),
         lambda _: study.gate_g3b()),
        ("G4", any(study.has(e, "val") for e in study.find("EG-SFT")),
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
                # Part of the gate evaluated and passed; the rest has no evidence yet.
                gates[name], waiting = {"status": "partially_evaluated", **result}, name
            else:
                gates[name] = {"status": "evaluated", **result}
                failed = None if result["pass"] else name
    passed = {name: gates[name].get("pass", False) for name in gates}
    rl_test = (passed["G4"] and passed["G5"]
               and gates["G6"].get("substantial_rl_test", {}).get("pass", False))
    return {"study_id": "aircraft-freshpool-screened-grpo-v1", "d_req": study.branch,
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
