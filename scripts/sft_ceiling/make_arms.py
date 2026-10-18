"""Write the lane's arm specs (arm_exec.py format, payload 9d9a09f) into runs/aircraft-sft-ceiling-v1/launch/<id>/.

  python3 -I scripts/sft_ceiling/make_arms.py

Each launch dir gets arm.json and run.sh (finds this host's r226lin-sftceil root, runs the sweep's
unchanged arm_exec.py). Arms:
  I-seed{1,2,3,42}: I's exact recipe (controlled400 sft-1e5-time): init W, train400 in file order,
    SFT LR 1e-5 constant, batch 1, max_steps 6000, stopped at I's 2464 updates; only the seed differs.
    Seed 42 on CPU is I's own seed: it separates the device effect (I was trained on MPS) from seed.
  cont-all-lr{5e-6,1e-5,2e-5}: SFT from I on the clean all_train pool
    (docs/aircraft-sft-ceiling-v1/pool_all_train_clean.json), seed 42, stops 300/600/1200/2400/4800/7200 (7200 SFT updates ~ active time of 600 GRPO-G8 updates).
Every stop is assayed on dev800 (sha b01175a1...) by arm_exec's own assay.
"""
import json
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
OUT = WT / "runs/aircraft-sft-ceiling-v1/launch"
TRAIN400 = Path("/Users/idide/RyanLinEnv/data/fgvc-aircraft-2013b/candidates/fourshot-seed791/train400.jsonl")
POOL = WT / "docs/aircraft-sft-ceiling-v1/pool_all_train_clean.json"
COMMON = {"adapter_profile": "legacy-qv-v1", "beta": None, "device": "cpu", "dtype": "float32",
          "features": [], "max_completion_length": 256, "max_pixels": 200704, "min_pixels": 50176,
          "mode": "sft", "num_generations": 4, "order": "random", "response_format": "answer_only",
          # 1.0, as eval-I's spec: arm_exec.assay validates a Profile from the spec (None raises);
          # the dev800 assay is greedy and SFT training never passes --temperature
          "temperature": 1.0, "threads": 16, "save_steps": 100,
          # a stop, not a trajectory parameter (constant LR); large enough never to bind on CPU
          "max_active_seconds": 172800.0, "study": "aircraft-sft-ceiling-v1"}
RUN_SH = """#!/bin/bash
set -euo pipefail
for R in /dev/shm/r226lin-sftceil /var/tmp/r226lin-sftceil; do
  [ -f "$R/RECEIPT.json" ] && break
done
[ -f "$R/RECEIPT.json" ] || { echo "REFUSE: no r226lin-sftceil root on $(hostname)" >&2; exit 3; }
echo "root=$R host=$(hostname -s)"
exec "$R/venv/bin/python" -I "$R/payload/code/scripts/exploratory_sweep/arm_exec.py" \\
  --root "$R" --arm-json arm.json --results results
"""


def main():
    t400 = [json.loads(x)["id"] for x in TRAIN400.read_text().splitlines() if x.strip()]
    pool = json.loads(POOL.read_text())["ids"]
    assert len(t400) == 400 and len(pool) == 1452
    arms = []
    for seed in (1, 2, 3, 4, 5, 42):
        arms.append(dict(COMMON, id=f"sftceil-I-seed{seed}", family="I-seed", init="W", lr=1e-5,
                         seed=seed, train_ids=t400, stops=[2464], max_steps_total=6000))
    for lr in (3e-6, 5e-6, 1e-5, 2e-5):
        tag = f"{lr:.0e}".replace("e-0", "e-")
        arms.append(dict(COMMON, id=f"sftceil-cont-all-lr{tag}", family="cont-all", init="I",
                         lr=lr, seed=42, train_ids=pool, stops=[300, 600, 1200, 2400, 4800, 7200],
                         max_steps_total=7200))
    for arm in arms:
        d = OUT / arm["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "arm.json").write_text(json.dumps(arm, indent=1, sort_keys=True) + "\n")
        (d / "run.sh").write_text(RUN_SH)
        print(arm["id"], arm["init"], arm["lr"], arm["seed"], len(arm["train_ids"]), arm["stops"])


main()
