#!/bin/bash
# Launch the lane bootstrap (bootstrap_host.sh) on each named host. Usage: boot_all.sh N HOST...
set -u
STAGE=${STAGE:-/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/runs/aircraft-sft-ceiling-v1/stage}
N=$1; shift
for h in "$@"; do
  timeout 600 ~/.claude/bin/uwrun run --host "$h" --threads 4 --mem 10 --timeout 5400 --disk 3 --scratch "${SCRATCH:-disk}" \
    --sync "$STAGE" --name "sftceil-boot-$h-$N" -- bash sftceil/bootstrap_host.sh 2>&1 | tail -2
done
