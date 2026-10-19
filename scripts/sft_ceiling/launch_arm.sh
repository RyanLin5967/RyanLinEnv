#!/bin/bash
# Launch one lane arm: launch_arm.sh ARM_ID HOST N [disk|shm]
# 16 threads (truthful: arm_exec sets OMP/MKL/torch threads to the spec's 16); --mem 17 = measured
# ~13.6 GB max RSS of the fp32 2B LoRA trainer (runner.py ARM_RSS_GB note) + 20%, rounded up.
set -euo pipefail
ARM=$1; HOST=$2; N=$3; SCRATCH=${4:-disk}
DIR=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/runs/aircraft-sft-ceiling-v1/launch/$ARM
[ -f "$DIR/arm.json" ] || { echo "REFUSE: no $DIR/arm.json" >&2; exit 2; }
timeout 900 ~/.claude/bin/uwrun run --host "$HOST" --threads 16 --mem 17 --timeout 259200 --disk 3 \
  --scratch "$SCRATCH" --sync "$DIR" --name "$ARM-$N" -- bash run.sh
