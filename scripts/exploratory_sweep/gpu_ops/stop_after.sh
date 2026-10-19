#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# stop_after.sh HOST WORKER KEY LABEL [MAXSEC]: block until assays/LABEL.json exists in the
# worker's results dir for queue key KEY, then kill that arm (its arm_exec and trainer
# processes, matched by the spec file name), so the worker files it as failed and moves on.
# Root decisions to cut a run short use this (e.g. P1 stopped after step 300).
H=$1; W=$2; KEY=$3; LABEL=$4; MAX=${5:-540}
F=/home/r226lin/runs/$W/work/results/$KEY/assays/$LABEL.json
end=$(( $(date +%s) + MAX ))
while [ "$(date +%s)" -lt "$end" ]; do
  if timeout 60 ~/.claude/bin/uwrun ssh "$H" "test -s $F" 2>/dev/null; then
    timeout 60 ~/.claude/bin/uwrun ssh "$H" "pkill -u r226lin -f '[a]rm_exec.py .*$KEY.json'; sleep 5; pkill -u r226lin -f '[v]lm_local_training.*/arms/.*$(echo "$KEY" | sed 's/^[0-9]*-//')/'; sleep 3; pgrep -u r226lin -af '$(echo "$KEY" | sed 's/^[0-9]*-//')' | head -3; echo stopped $KEY after $LABEL"
    exit 0
  fi
  sleep 60
done
echo "not yet: $LABEL for $KEY"; exit 1
