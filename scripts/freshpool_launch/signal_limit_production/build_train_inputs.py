"""Build ``inputs/`` for measure_units.sh and smoke_stage0.sh: native-TRAIN rows only.

  python3 -I build_train_inputs.py DEST [ROWS]   # DEST/inputs/{train,train8,train24}.jsonl,
                                                 # images/, data_spec.json, w/ (W's run)

Every 6th row of train400 (classes spread), at most ROWS (default 64); each row must be
split "train" with its image under development/images/train/, so no H split, native VAL or
sealed TEST image can be copied. Light work: file copies only.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/Users/idide/RyanLinEnv")
CANDIDATES = ROOT / "data/fgvc-aircraft-2013b/candidates/fourshot-seed791"
W_RUN = ROOT / "runs/aircraft-warmstart-v1"


def main(argv):
    dest = Path(argv[0]) / "inputs"
    limit = int(argv[1]) if len(argv) > 1 else 64
    (dest / "images").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (CANDIDATES / "train400.jsonl").read_text().splitlines()]
    picked, out = rows[::6][:limit], []
    for row in picked:
        source = (CANDIDATES / row["image"]).resolve()
        if row.get("split") != "train" or "/development/images/train/" not in str(source):
            raise SystemExit(f"not a native TRAIN image: {row.get('id')}")
        shutil.copy2(source, dest / "images" / source.name)
        out.append({**row, "image": f"images/{source.name}"})
    for name, part in (("train.jsonl", out), ("train8.jsonl", out[:8]),
                       ("train24.jsonl", out[:24])):
        (dest / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in part))
    shutil.copy2(CANDIDATES / "data_spec.json", dest / "data_spec.json")
    shutil.copytree(W_RUN, dest / "w", dirs_exist_ok=True)
    print(json.dumps({"rows": len(out), "dest": str(dest)}))


if __name__ == "__main__":
    main(sys.argv[1:])
