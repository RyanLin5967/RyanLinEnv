"""Corrected GRPO-from-I arm table for exploratory sweep v2 (plan amendment 5, root D37).

  python3 -I scripts/exploratory_sweep/d37_arms.py [--check]

Writes docs/aircraft-exploratory-sweep-v2/arms-d37-grpo-I.json. It replaces the rung-1
grpo-I arms of arms-rung1.json (LR 1e-6, 5e-6, 1e-5; retired, never launched), which
runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md (root checkout) rules out (C3). D37 (PR #40)
released GRPO-I under these rules, all encoded below:

- fp32 only (C2, M9), on CPU hosts with AVX-512 only (`need_avx512`: the MKL fp32 kernels
  die with SIGILL elsewhere); one device class per trajectory (M6);
- LR {1e-5, 2e-5, 5e-5} plus one 3e-6 forgetting guard (C3); beta 0 (C12); G 8;
- sampling temperature T {1.0, 1.3} (the P0-I pass@8 was measured at both);
- dynamic sampling with >= 8 draws (C5): all-same reward groups are dropped and redrawn,
  and the discarded generation seconds count as RL time (the trainer's active clock);
- 600 updates, assayed (dev800) at 150, 300 and 600: a fixed schedule, no early stop (C11);
- one matched SFT-from-I control per RL arm on its host, LR-gridded rank for rank over
  {5e-6, 1e-5, 2e-5} (C4; the guard's control is 5e-6), replaying EVERY scanned row of its
  RL arm (`control_replays` scanned, up to 600 x 8 rows) and time-matched at each RL
  checkpoint (`control_time_match` per_checkpoint).

Trainer features the arms need (receipt-gated by runner.valid_receipt, so an arm can only
launch on a payload whose trainer has them): dynamic_sampling for the RL arms,
replay_sequence for the controls. Stdlib only; the arm builder is design.py's. `--check`
rebuilds in memory and refuses unless the file on disk is byte-identical. Reads no data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
OUT = HERE.parents[2] / "docs" / "aircraft-exploratory-sweep-v2" / "arms-d37-grpo-I.json"


def _design():
    spec = importlib.util.spec_from_file_location("sweep_design", HERE.parent / "design.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _design()

FAMILY = "d37-grpo-I"
GUARD_FAMILY = "d37-guard"
INIT = "I"
GRPO_LRS = (1e-5, 2e-5, 5e-5)
GUARD_LR = 3e-6
TEMPERATURES = (1.0, 1.3)
BETA = 0.0
GENERATIONS = 8
DYNAMIC_DRAWS = 8
UPDATES = 600
STOPS = [150, 300, 600]
CONTROL_GRID = {1: (5e-6,), 3: (5e-6, 1e-5, 2e-5)}
DEVICE = {"device": "cpu", "dtype": "float32", "threads": 16,
          "device_class": "cpu-float32-avx512", "need_avx512": True}
# arms-rung1.json grpo-I specs (G4 and G8 at LR 1e-6, 5e-6, 1e-5) and their matched
# controls: never launched (held by amendment 3), retired here, replaced by this table
_WHY_LR = {"1e-6": "LR 1e-6 is ruled out (C3: no 1e-6)",
           "5e-6": "LR 5e-6 is outside {1e-5, 2e-5, 5e-5} + one 3e-6 guard (C3)",
           "1e-5": "LR 1e-5 is re-run under D37 here"}
_WHY_ALL = ("beta 0.04 (C12: beta 0), no dynamic sampling (C5), T 1.0 only, 150-update rung "
            "schedule instead of 600 assayed at 150/300/600; replaced by arms-d37-grpo-I.json "
            "(root D37, PR #40)")
RETIRED = {}
for _lr in ("1e-6", "5e-6", "1e-5"):
    for _g in ("g4", "g8"):
        _id = f"grpo-I-lr{_lr}-{_g}"
        RETIRED[_id] = f"{_WHY_LR[_lr]}; {_WHY_ALL}"
        RETIRED[f"{_id}~ctl"] = f"control of retired {_id}"


def control_lrs(lrs) -> dict:
    ordered = sorted(set(lrs))
    return dict(zip(ordered, CONTROL_GRID[len(ordered)]))


def rl_arm(family, lr, temperature, ctl_lr):
    t_tag = f"t{temperature:.1f}".replace(".", "_")
    a = D.arm(f"{family}-{D.lr_tag(lr)}-g8-b0-{t_tag}-ds8", track="rl", family=family,
              init=INIT, mode="grpo", lr=lr, num_generations=GENERATIONS, beta=BETA,
              temperature=temperature, updates=UPDATES, max_steps_total=UPDATES,
              stops=list(STOPS), features=["dynamic_sampling"],
              # beta 0 must name its aggregation (main's trainer rule): "grpo" is the legacy
              # aggregation and leaves the settings identity unchanged
              extra_trainer_args=["--loss-type", "grpo",
                                  "--dynamic-sampling-max-draws", str(DYNAMIC_DRAWS)],
              matched_control_lr=ctl_lr, control_replays="scanned",
              control_max_steps_total=UPDATES * DYNAMIC_DRAWS, amendment=5, **DEVICE)
    return a


def arms() -> list:
    out = []
    grid = control_lrs(GRPO_LRS)
    for temperature in TEMPERATURES:
        for lr in GRPO_LRS:
            out.append(rl_arm(FAMILY, lr, temperature, grid[lr]))
    out.append(rl_arm(GUARD_FAMILY, GUARD_LR, 1.0, control_lrs((GUARD_LR,))[GUARD_LR]))
    controls = D.controls_for(out)
    if [c["control_of"] for c in controls] != [a["id"] for a in out]:
        raise ValueError("one control per RL arm, in order")
    for c, rl in zip(controls, out):
        c.update(lr=rl["matched_control_lr"], features=["replay_sequence"],
                 max_steps_total=rl["control_max_steps_total"],
                 control_replays=rl["control_replays"], control_time_match="per_checkpoint",
                 amendment=5, **DEVICE)
    out += controls
    ids = [a["id"] for a in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate arm id")
    return out


def table() -> dict:
    return {"study_id": D.STUDY_ID, "schema_version": 1, "rung": 1, "amendment": 5,
            "decision": "D37", "inits": D.INITS,
            "rung_updates": {str(k): v for k, v in D.RUNG_UPDATES.items()},
            "rung_seeds": {str(k): list(v) for k, v in D.RUNG_SEEDS.items()},
            "control_lr": D.CONTROL_LR,
            "control_grid": {str(k): list(v) for k, v in CONTROL_GRID.items()},
            "retires": RETIRED, "arms": arms()}


def render() -> bytes:
    return (json.dumps(table(), indent=1, sort_keys=True) + "\n").encode()


def _check_dynamic_draws():
    """wave2_arms.check_dynamic_draws (CORRECTIONS C5: >= 8 draws), one implementation."""
    spec = importlib.util.spec_from_file_location("sweep_wave2_arms",
                                                  HERE.parent / "wave2_arms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_dynamic_draws


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    check_draws = _check_dynamic_draws()
    raw = render()
    check_draws(json.loads(raw)["arms"])  # never write or accept a row below 8 draws
    if a.check:
        if OUT.is_file():  # the file on disk, not only the rebuilt one, must satisfy C5
            try:
                check_draws(json.loads(OUT.read_text())["arms"])
            except ValueError as exc:
                print(f"D37 table refused: {exc}", file=sys.stderr)
                return 1
        if not OUT.is_file() or OUT.read_bytes() != raw:
            print(f"D37 table drift: {OUT}", file=sys.stderr)
            return 1
        print(f"D37 table check ok: {len(json.loads(raw)['arms'])} arms")
        return 0
    OUT.write_bytes(raw)
    print(json.dumps({"path": str(OUT), "arms": len(json.loads(raw)["arms"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
