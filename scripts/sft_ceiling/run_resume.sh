#!/bin/bash
# New-host side of a move: place the packed trainer state (pack/) into this host's root, then run
# arm_exec with the resume spec (arm.json carries resume_stop = packed checkpoint, the remaining
# stops, and prior_endpoints). The trainer's own resume check refuses unless model/data/profile/
# settings are identical, which includes the root path, so the new root must equal the packed one.
set -euo pipefail
for R in /dev/shm/r226lin-sftceil /var/tmp/r226lin-sftceil; do
  [ -f "$R/RECEIPT.json" ] && break
done
[ -f "$R/RECEIPT.json" ] || { echo "REFUSE: no r226lin-sftceil root on $(hostname)" >&2; exit 3; }
ARM=$(python3 -c 'import json;print(json.load(open("arm.json"))["id"])')
PROOT=$(python3 -c 'import json;print(json.load(open("pack/PACK.json"))["root"])')
[ "$PROOT" = "$R" ] || { echo "REFUSE: packed root $PROOT != this root $R (resume identity)" >&2; exit 4; }
python3 - <<'PY'
import hashlib, json, os
p = json.load(open("pack/PACK.json"))
bad = [f for f, h in p["files"].items()
       if hashlib.sha256(open(os.path.join("pack", f), "rb").read()).hexdigest() != h]
if bad:
    raise SystemExit("REFUSE: pack files differ from PACK.json: %s" % bad[:5])
print("pack verified", len(p["files"]), "files, checkpoint", p["checkpoint_step"])
PY
A=$R/arms/$ARM
if [ -d "$A" ]; then mv "$A" "$A.superseded-$(date +%s)"; fi
mkdir -p "$A"
cp -a pack/train "$A/train"
[ -d pack/kept ] && cp -a pack/kept "$A/kept"
mkdir -p results
[ -d pack/prior_results ] && cp -a pack/prior_results/. results/
echo "root=$R host=$(hostname -s) resumed $ARM"
exec "$R/venv/bin/python" -I "$R/payload/code/scripts/exploratory_sweep/arm_exec.py" \
  --root "$R" --arm-json arm.json --results results
