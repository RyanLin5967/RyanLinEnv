#!/bin/bash
# Host side of a move: pack a STOPPED arm's trainer state for resumption on another host.
# Run via uwrun on the OLD host after `uwrun stop` of the arm's run:
#   uwrun run --host OLD --threads 1 --sync <dir holding this script> --name mvpack-ARM-N -- bash pack_arm.sh ARM OLDRUN
# Output in work/pack/: train/ (run record, journals, audit, the latest COMPLETE checkpoint only),
# kept/ (kept endpoint adapters), prior_results/ (the old run's results dir), PACK.json (sha256 of all).
set -euo pipefail
ARM=$1; OLDRUN=$2
for R in /dev/shm/r226lin-sftceil /var/tmp/r226lin-sftceil; do [ -d "$R/arms/$ARM" ] && break; done
A=$R/arms/$ARM; export R
[ -d "$A/train" ] || { echo "REFUSE: no $A/train" >&2; exit 2; }
if pgrep -u "$(id -u)" -f "arms/$ARM/train" >/dev/null; then echo "REFUSE: $ARM still has a live trainer" >&2; exit 3; fi
best=""
for c in $(ls -d "$A"/train/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  ok=1; for f in trainer_state.json optimizer.pt scheduler.pt rng_state.pth adapter_model.safetensors active_time.json; do
    [ -f "$c/$f" ] || ok=0; done
  [ $ok = 1 ] && best=$c
done
[ -n "$best" ] || { echo "REFUSE: no complete checkpoint" >&2; exit 4; }
mkdir -p pack/train
rsync -a --exclude 'checkpoint-*' --exclude final_adapter "$A"/train/ pack/train/
cp -a "$best" pack/train/
[ -d "$A/kept" ] && cp -a "$A/kept" pack/kept
[ -d "$HOME/runs/$OLDRUN/work/results" ] && cp -a "$HOME/runs/$OLDRUN/work/results" pack/prior_results
python3 - "$best" <<'PY'
import hashlib, json, os, sys
step = int(os.path.basename(sys.argv[1]).split("-")[1])
files = {}
for d, _, fs in os.walk("pack"):
    for f in fs:
        p = os.path.join(d, f)
        files[os.path.relpath(p, "pack")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"checkpoint_step": step, "root": os.environ.get("R", ""), "files": files},
          open("pack/PACK.json", "w"), indent=1, sort_keys=True)
print("packed checkpoint", step, len(files), "files")
PY
