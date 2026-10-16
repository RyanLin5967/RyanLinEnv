"""Assert the actual Harbor verifier result; CLI exit status alone is insufficient."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job", type=Path)
parser.add_argument("--min-reward", type=float, default=0.0)
parser.add_argument("--max-reward", type=float, default=1.0)
args = parser.parse_args()
trials = sorted(args.job.glob("*/result.json"))
if not trials:
    raise SystemExit("No trial results found")
for path in trials:
    result = json.loads(path.read_text())
    if result.get("exception_info"):
        raise SystemExit(f"Trial failed: {path.parent.name}")
    reward = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
    if reward is None or not args.min_reward <= reward <= args.max_reward:
        raise SystemExit(f"Unexpected reward {reward} in {path.parent.name}")
    print(f"{path.parent.name}: reward={reward:.4f}")
