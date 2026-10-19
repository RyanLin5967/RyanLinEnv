#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# Mac-side: print what is ACTUALLY running on each lane host, read from the host's
# gpuq/running* dirs (never from queue order), with load1 and GPU memory.
for h in ecetesla2 ecetesla4 eceubuntu2 iccad0 iccad13 iccad8 iccad10 iccad11 iccad12 iccad7; do
  case $h in ece*) R=/var/tmp/r226lin-rlesweep-gpu;; *) R=/dev/shm/r226lin-rlesweep-gpulane;; esac
  out=$(timeout 60 ~/.claude/bin/uwrun ssh $h "cd $R/gpuq 2>/dev/null && for d in running*; do for f in \$d/*.json; do [ -e \"\$f\" ] && echo \"  \$f\"; done; done; echo \"  load1 \$(cut -d' ' -f1 /proc/loadavg) gpu \$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)\"" 2>&1)
  echo "$h:"; echo "$out"
done
