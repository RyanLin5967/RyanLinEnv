"""Concrete wave-2 rung-1 arm table for exploratory sweep v2 (plan amendment 3).

  python3 -I scripts/exploratory_sweep/wave2_arms.py [--check]

Writes docs/aircraft-exploratory-sweep-v2/arms-wave2-rung1.json, the arms that use the
wave-2 trainer features built and merged after the plan (hierarchical partial-credit reward,
RFT/expert iteration, DAPO-style GRPO knobs, closed-set and tagged think/answer formats), at
the plan's rung-1 budget (150 updates, seed 42, declared total 1,500), with:

- a declared device per arm: `device` (cpu|cuda), `dtype`, `gpu` (cuda only) and
  `device_class`; runner.py launches an arm only on a host whose receipt has that device
  kind and GPU model;
- one matched SFT control per RL arm on the RL arm's own host and device (examples- and
  time-matched endpoints, as in the plan);
- W/I device controls: an assay-only eval of W and of I on every device class the table
  uses, so a GPU arm is compared with W/I measured under the same numerics.

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
HIERARCHY = "@payload/code/src/vision_rl/aircraft_assets/label-hierarchy-train-v1.json"


def _design():
    spec = importlib.util.spec_from_file_location("sweep_design", HERE.parent / "design.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _design()

# Device classes. A comparison stays inside one class; the device controls tie each class
# back to CPU fp32 (wave 1). bf16 training needs compute capability >= 8.0 (RTX 3070, sm_86);
# the RTX 2080 Ti (sm_75) trains in float16 with GradScaler loss scaling.
DEVICES = {
    "cpu-float32": {"device": "cpu", "dtype": "float32", "gpu": None, "threads": 16,
                    "features": []},
    "rtx3070-bfloat16": {"device": "cuda", "dtype": "bfloat16", "gpu": "rtx3070", "threads": 4,
                         "features": ["cuda"]},
    "rtx2080ti-float16": {"device": "cuda", "dtype": "float16", "gpu": "rtx2080ti",
                          "threads": 4, "features": ["cuda", "cuda_float16"]},
}

# (tag, device class, overrides, features, extra trainer args). GRPO-W, G 8, LR {5e-6, 1e-5}
# per family (the plan's wave-2 rung-1 rule); RFT at LR {1e-5, 3e-5}.
GRPO_FAMILIES = [
    ("hier", "cpu-float32", {}, ["hierarchical_reward"],
     ["--label-hierarchy", HIERARCHY, "--family-credit", "0.5", "--manufacturer-credit",
      "0.25"]),
    # exact reward, minus 0.5 for a non-EOS or out-of-vocabulary answer (no partial credit)
    ("fmtpen", "cpu-float32", {}, ["hierarchical_reward", "invalid_answer_penalty"],
     ["--label-hierarchy", HIERARCHY, "--family-credit", "0", "--manufacturer-credit", "0",
      "--invalid-reward", "-0.5"]),
    ("closedset", "cpu-float32", {"response_format": "closed_set_answer_only"},
     ["closed_set_prompt"], []),
    ("drgrpo", "cpu-float32", {}, ["dr_grpo"], ["--loss-type", "dr_grpo", "--scale-rewards",
                                                "none"]),
    ("mu2", "cpu-float32", {}, ["num_iterations"], ["--num-iterations", "2"]),
    # DAPO: token-level loss, clip-higher (binds only once a batch is reused, see mu2) and
    # dynamic sampling (all-same reward groups dropped and redrawn, up to 4 per update)
    ("dapo", "rtx3070-bfloat16", {}, ["dapo", "dynamic_sampling"],
     ["--loss-type", "dapo", "--epsilon-high", "0.28", "--dynamic-sampling-max-draws", "4"]),
    ("cot", "rtx2080ti-float16", {"response_format": "think_answer"}, ["cot_answer_reward"], []),
    ("closedset-cot", "rtx2080ti-float16", {"response_format": "closed_set_think_answer"},
     ["closed_set_prompt", "cot_answer_reward"], []),
]
RFT = {"rounds": 400, "prompts_per_round": 32, "epochs_per_round": 1, "max_keep_per_prompt": 1,
       "skip_all_correct": False}
RFT_DEVICE = "rtx3070-bfloat16"


def on_device(arm: dict, cls: str) -> dict:
    d = DEVICES[cls]
    arm.update(device=d["device"], dtype=d["dtype"], threads=d["threads"], device_class=cls,
               features=sorted(set(arm.get("features", [])) | set(d["features"])))
    if d["gpu"]:
        arm["gpu"] = d["gpu"]
    return arm


def wave2_arms() -> list:
    arms = []
    for init in ("W", "I"):
        for cls in DEVICES:
            arms.append(on_device({
                "id": f"wave2-dev-eval-{init}-{cls}", "wave": 2, "rung": 0, "track": "baseline",
                "family": "device-control", "init": init, "mode": "eval", "seed": 42,
                **D.BASE}, cls))
    for tag, cls, overrides, features, extra in GRPO_FAMILIES:
        for lr in (5e-6, 1e-5):
            a = D.arm(f"wave2-{tag}-grpo-W-{D.lr_tag(lr)}-g8-{cls}", wave=2, track="rl",
                      family=f"wave2-{tag}", init="W", mode="grpo", lr=lr, num_generations=8,
                      features=features, extra_trainer_args=extra, **overrides)
            arms.append(on_device(a, cls))
    for lr in (1e-5, 3e-5):
        a = D.arm(f"wave2-rft-W-{D.lr_tag(lr)}-k8-{RFT_DEVICE}", wave=2, track="rl",
                  family="wave2-rft", init="W", mode="rft", lr=lr, num_generations=8,
                  features=["rejection_sampling_ft", "rft_step_boundary"], rft=dict(RFT))
        a["temperature"] = 1.0  # RFT samples K = 8 per prompt at T 1.0, like GRPO
        arms.append(on_device(a, RFT_DEVICE))
    for a in arms:
        if a["mode"] != "eval":
            a.setdefault("stops", [a["updates"]])
    controls = []
    for c, rl in zip(D.controls_for(arms), [a for a in arms if a["mode"] in ("grpo", "rft")]):
        if c["control_of"] != rl["id"]:
            raise ValueError("control order differs from its RL arms")
        cls = DEVICES[rl["device_class"]]
        c.update(features=[f for f in cls["features"]])
        controls.append(on_device(c, rl["device_class"]))
    out = arms + controls
    ids = [a["id"] for a in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate arm id")
    return out


def table() -> dict:
    arms = wave2_arms()
    return {"study_id": D.STUDY_ID, "schema_version": 1, "rung": 1, "wave": 2,
            "amendment": 3, "inits": D.INITS, "rung_updates": D.RUNG_UPDATES,
            "rung_seeds": {str(k): list(v) for k, v in D.RUNG_SEEDS.items()},
            "control_lr": D.CONTROL_LR, "device_classes": DEVICES, "rft": RFT, "arms": arms}


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
