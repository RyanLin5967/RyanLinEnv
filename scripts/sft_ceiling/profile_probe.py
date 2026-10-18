"""Forced-fire probe of arm_exec.assay's Profile construction on a host's own code root.

  ROOT/venv/bin/python -I profile_probe.py ROOT profile-probe.jsonl

Each line is the PROFILE_KEYS slice of a spec. Builds Profile(**keys, dtype=float32) exactly as
arm_exec.assay does and calls validate() (what predict_train_subset calls). Prints OK or the error.
"""
import json
import sys

root = sys.argv[1]
sys.path.insert(0, root + "/payload/code/src")
from vision_rl import vlm_local_training as vlt  # noqa: E402

for line in open(sys.argv[2]):
    keys = json.loads(line)
    try:
        vlt.Profile(**keys, dtype="float32").validate()
        print("OK", keys["temperature"])
    except Exception as exc:  # report every failure type
        print("FAIL", keys["temperature"], type(exc).__name__, exc)
