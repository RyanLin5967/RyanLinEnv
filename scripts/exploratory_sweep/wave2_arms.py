"""Concrete wave-2 rung-1 arm table for exploratory sweep v2 (plan amendment 4).

  python3 -I scripts/exploratory_sweep/wave2_arms.py [--check]

Writes docs/aircraft-exploratory-sweep-v2/arms-wave2-rung1.json, the arms that use the
wave-2 trainer features built and merged after the plan (RFT/expert iteration, DAPO-style
GRPO knobs, closed-set prompt, invalid-answer penalty), from I, at the plan's rung-1 budget
(150 updates, seed 42, declared total 1,500), under the root RL corrections of 2026-10-18
(runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md in the root checkout), with:

- a declared device per arm: `device` (cpu|cuda), `dtype`, `gpu` (cuda only) and
  `device_class`; runner.py launches an arm only on a host whose receipt has that device
  kind and GPU model;
- one matched SFT-from-I control per RL arm on the RL arm's own host and device
  (examples- and time-matched endpoints, as in the plan), at an LR that gives each family's
  controls the same LR count as its RL arms;
- W/I device controls: an assay-only eval of W and of I on every device class the table uses,
  so a GPU arm is compared with I (and W) measured under the same numerics.

Stdlib only; the arm builder is design.py's (same BASE keys and defaults). `--check`
rebuilds in memory and refuses unless the file on disk is byte-identical. Never reads
native VAL or the sealed TEST: nothing here reads data at all.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
OUT = HERE.parents[2] / "docs" / "aircraft-exploratory-sweep-v2" / "arms-wave2-rung1.json"
HIERARCHY = "@payload/code/scripts/exploratory_sweep/label-hierarchy-train-v1.json"


def _design():
    spec = importlib.util.spec_from_file_location("sweep_design", HERE.parent / "design.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _design()

# Device classes. One class per trajectory, start to end (corrections M6). A paired
# comparison (an arm vs its matched control, vs I) stays inside one class. Promotion ranks
# every rl unit together, as the plan's rule says; the device controls measure each class's
# offset from CPU fp32 (wave 1), so a cross-class ranking can be read against it.
# GRPO runs only in float32 (corrections C2, M9): bf16 failed sampler-teacher parity (0.028
# against a 0.02 tolerance) and fp16 has no parity evidence. RFT is completion-only SFT on
# the policy's own samples (no importance ratio), and bf16 is allowed for SFT, so RFT keeps
# the RTX 3070 (sm_86) in bfloat16.
DEVICES = {
    "cpu-float32": {"device": "cpu", "dtype": "float32", "gpu": None, "threads": 16,
                    "features": []},
    "rtx2080ti-float32": {"device": "cuda", "dtype": "float32", "gpu": "rtx2080ti",
                          "threads": 4, "features": ["cuda"]},
    "rtx3070-bfloat16": {"device": "cuda", "dtype": "bfloat16", "gpu": "rtx3070", "threads": 4,
                         "features": ["cuda"]},
}

# Every wave-2 family starts from I (corrections: GRPO-W loses to I by 5-9 pp on dev800).
INIT = "I"
# GRPO-I: beta 0 (C12); LR {1e-5, 2e-5, 5e-5} (C3), Dr-GRPO also 3e-5 (C17); one 3e-6
# forgetting guard for the whole table (C3).
GRPO_LRS = (1e-5, 2e-5, 5e-5)
BETA = 0.0
# Matched SFT-from-I controls cover the same number of DISTINCT LRs as their family's RL
# arms (C4): the family's k-th smallest RL LR gets the k-th of the SFT grid below, so a
# family's best-of-k compares with its controls' best-of-k at the same k. Three LRs use
# {5e-6, 1e-5, 2e-5}; Dr-GRPO's four extend it with 3e-5 (an SFT LR wave 1 already runs);
# the single 3e-6 guard gets 5e-6.
CONTROL_GRID = {1: (5e-6,), 3: (5e-6, 1e-5, 2e-5), 4: (5e-6, 1e-5, 2e-5, 3e-5)}


def control_lrs(lrs) -> dict:
    """{RL LR: matched SFT-from-I control LR}, rank for rank, all distinct."""
    ordered = sorted(lrs)
    return dict(zip(ordered, CONTROL_GRID[len(ordered)]))


# (tag, device class, overrides, features, extra trainer args, LRs). Dropped as refuted
# (C7): hierarchical partial credit (contrast in 1 of 30-50 groups) and both CoT formats
# (valid 94.9% -> 82.8%, accuracy 70.7% -> 55.5%).
GRPO_FAMILIES = [
    # exact reward, minus 0.5 for a non-EOS or out-of-vocabulary answer (no partial credit;
    # the hierarchy file only supplies the label set, with both credits 0)
    ("fmtpen", "cpu-float32", {}, ["hierarchical_reward", "invalid_answer_penalty"],
     ["--label-hierarchy", HIERARCHY, "--family-credit", "0", "--manufacturer-credit", "0",
      "--invalid-reward", "-0.5"], GRPO_LRS),
    ("closedset", "cpu-float32", {"response_format": "closed_set_answer_only"},
     ["closed_set_prompt"], [], GRPO_LRS),
    ("drgrpo", "cpu-float32", {}, ["dr_grpo"],
     ["--loss-type", "dr_grpo", "--scale-rewards", "none"], GRPO_LRS + (3e-5,)),
    ("mu2", "cpu-float32", {}, ["num_iterations"], ["--num-iterations", "2"], GRPO_LRS),
    # DAPO: token-level loss, clip-higher (binds only once a batch is reused, see mu2) and
    # dynamic sampling, >= 8 draws (C5): all-same reward groups dropped and redrawn
    ("dapo", "rtx2080ti-float32", {}, ["dapo", "dynamic_sampling"],
     ["--loss-type", "dapo", "--epsilon-high", "0.28", "--dynamic-sampling-max-draws", "8"],
     GRPO_LRS),
    # plain GRPO-I at a low LR: how much a small step forgets
    ("guard", "cpu-float32", {}, [], [], (3e-6,)),
]
RFT = {"rounds": 400, "prompts_per_round": 32, "epochs_per_round": 1, "max_keep_per_prompt": 1,
       "skip_all_correct": False}
# Controls that must replay EVERY row their RL arm scored (C4), not only its updates: a
# dynamic-sampling arm scores up to 8 groups per update, an RFT round scores all its 32
# prompts. Their control step cap is the most rows the RL run can score at its total.
DYNAMIC_DRAWS = 8
RFT_DEVICE = "rtx3070-bfloat16"
RFT_LRS = GRPO_LRS


def on_device(arm: dict, cls: str) -> dict:
    d = DEVICES[cls]
    arm.update(device=d["device"], dtype=d["dtype"], threads=d["threads"], device_class=cls,
               features=sorted(set(arm.get("features", [])) | set(d["features"])))
    if d["gpu"]:
        arm["gpu"] = d["gpu"]
    return arm


def wave2_arms() -> list:
    arms = []
    for init in ("W", "I"):  # every arm starts from I; W stays as the wave-1 reference
        for cls in DEVICES:
            arms.append(on_device({
                "id": f"wave2-dev-eval-{init}-{cls}", "wave": 2, "rung": 0,
                "track": "baseline", "family": "device-control", "init": init,
                "mode": "eval", "seed": 42, **D.BASE}, cls))
    for tag, cls, overrides, features, extra, lrs in GRPO_FAMILIES:
        ctl_lr = control_lrs(lrs)
        for lr in lrs:
            pairing = {}
            if "dynamic_sampling" in features:
                pairing = {"control_replays": "scanned",
                           "control_max_steps_total": D.RUNG_UPDATES[3] * DYNAMIC_DRAWS}
            if overrides.get("response_format", "answer_only") != "answer_only":
                # the control trains and assays on the same prompt, so the pair isolates RL
                pairing["control_response_format"] = overrides["response_format"]
            a = D.arm(f"wave2-{tag}-grpo-{INIT}-{D.lr_tag(lr)}-g8-b0-{cls}", wave=2,
                      track="rl", family=f"wave2-{tag}", init=INIT, mode="grpo", lr=lr,
                      num_generations=8, beta=BETA, features=features,
                      extra_trainer_args=extra, matched_control_lr=ctl_lr[lr], **pairing,
                      **overrides)
            arms.append(on_device(a, cls))
    ctl_lr = control_lrs(RFT_LRS)
    for lr in RFT_LRS:
        a = D.arm(f"wave2-rft-{INIT}-{D.lr_tag(lr)}-k8-{RFT_DEVICE}", wave=2, track="rl",
                  family="wave2-rft", init=INIT, mode="rft", lr=lr, num_generations=8,
                  features=["rejection_sampling_ft", "rft_step_boundary"], rft=dict(RFT),
                  matched_control_lr=ctl_lr[lr], control_replays="scanned",
                  control_max_steps_total=RFT["rounds"] * RFT["prompts_per_round"])
        a["temperature"] = 1.0  # RFT samples K = 8 per prompt at T 1.0, like GRPO
        arms.append(on_device(a, RFT_DEVICE))
    for a in arms:
        if a["mode"] != "eval":
            a.setdefault("stops", [a["updates"]])
    controls = []
    rls = [a for a in arms if a["mode"] in ("grpo", "rft")]
    made = D.controls_for(arms)
    if len(made) != len(rls):
        raise ValueError("one control per RL arm")
    for c, rl in zip(made, rls):
        if c["control_of"] != rl["id"]:
            raise ValueError("control order differs from its RL arms")
        c.update(lr=rl["matched_control_lr"],
                 features=list(DEVICES[rl["device_class"]]["features"]),
                 response_format=rl.get("control_response_format", c["response_format"]),
                 max_steps_total=rl.get("control_max_steps_total", c["max_steps_total"]))
        if rl.get("control_replays"):
            c["control_replays"] = rl["control_replays"]
        controls.append(on_device(c, rl["device_class"]))
    out = arms + controls
    ids = [a["id"] for a in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate arm id")
    return out


def table() -> dict:
    arms = wave2_arms()
    return {"study_id": D.STUDY_ID, "schema_version": 1, "rung": 1, "wave": 2,
            "amendment": 4, "inits": D.INITS, "rung_updates": D.RUNG_UPDATES,
            "rung_seeds": {str(k): list(v) for k, v in D.RUNG_SEEDS.items()},
            "control_lr": D.CONTROL_LR,
            "control_grid": {str(k): list(v) for k, v in CONTROL_GRID.items()},
            "device_classes": DEVICES, "rft": RFT, "arms": arms}


def render() -> bytes:
    return (json.dumps(table(), indent=1, sort_keys=True) + "\n").encode()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    raw = render()
    if a.check:
        if not OUT.is_file() or OUT.read_bytes() != raw:
            print(f"wave-2 table drift: {OUT}", file=sys.stderr)
            return 1
        print(f"wave-2 table check ok: {len(json.loads(raw)['arms'])} arms")
        return 0
    OUT.write_bytes(raw)
    arms = json.loads(raw)["arms"]
    by = {}
    for x in arms:
        key = f"{x['track']}/{x['device_class']}"
        by[key] = by.get(key, 0) + 1
    print(json.dumps({"path": str(OUT), "arms": len(arms), "by_track_device": by}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
