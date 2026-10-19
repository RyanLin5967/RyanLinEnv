#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# deploy_restart.sh HOST OLD NEW: after HOST's worker exited on STOP (so no arm is mid-run
# on the old trainer), delta-bootstrap payload NEW onto it, clear STOP, start a new worker.
H=$1; OLD=$2; NEW=$3
WT=$(cd "$(dirname "$0")/../../.." && pwd)
if timeout 60 ~/.claude/bin/uwrun ssh "$H" 'pgrep -u r226lin -f "[g]pu_worker.sh" >/dev/null && echo busy' | grep -q busy; then
  echo "REFUSE: a worker still runs on $H"; exit 2
fi
out=$(bash "$WT/scripts/exploratory_sweep/gpu_ops/boot_delta.sh" "$OLD" "$NEW" "$H")
echo "$out"
name=$(echo "$out" | grep rlegpu-boot | tail -1)
until timeout 60 ~/.claude/bin/uwrun status "$name" | grep -qE "DONE|LOST"; do sleep 20; done
timeout 60 ~/.claude/bin/uwrun status "$name"
timeout 60 ~/.claude/bin/uwrun ssh "$H" 'rm -f /var/tmp/r226lin-rlesweep-gpu/gpuq/STOP'
timeout 600 python3 -I "$WT/scripts/exploratory_sweep/gpu_lane.py" worker --host "$H"
