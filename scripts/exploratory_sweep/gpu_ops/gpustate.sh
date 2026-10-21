#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# Mac-side: per GPU host, read gpuq/running, whether a worker process is alive, GPU memory,
# compute apps, the next 3 pending specs and the 2 most recent failures (from the host).
for h in ${@:-ecetesla2 ecetesla4 eceubuntu2}; do
  echo "== $h"
  timeout 90 ~/.claude/bin/uwrun ssh "$h" 'cd /var/tmp/r226lin-rlesweep-gpu/gpuq; echo "running: $(ls running)"; echo "worker: $(pgrep -u r226lin -f "[g]pu_worker.sh" | head -1)"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; echo "pending:"; ls pending | sort | head -3; echo "recent failed:"; ls -t failed | head -2'
done
