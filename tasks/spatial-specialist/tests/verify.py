"""Runs in Harbor's separate verifier image; never imports code from the submission."""

import secrets
from pathlib import Path

from vision_rl.data import write_json
from vision_rl.evaluation import evaluate
from vision_rl.spatial import generate_split

logs = Path("/logs/verifier")
try:
    # Per-trial holdout prevents a policy from memorizing a public fixed test seed.
    seed = secrets.randbelow(2**31)
    manifest = generate_split(Path("/tmp/holdout"), "test", 512, seed)
    result = evaluate(
        manifest, logs / "evaluation.json", checkpoint=Path("/logs/artifacts/training/policy.npz")
    )
    write_json(
        logs / "holdout.json",
        {"seed": seed, "examples": 512, "dataset_fingerprint": result["dataset_fingerprint"]},
    )
    (logs / "reward.txt").write_text(str(result["accuracy"]) + "\n")
except Exception as exc:
    # Invalid/missing model => zero reward, never stale reward from a prior run.
    write_json(logs / "error.json", {"type": type(exc).__name__, "message": str(exc)})
    (logs / "reward.txt").write_text("0\n")
