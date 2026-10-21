#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# One overnight tick (Mac-side): fetch every worker, rebuild the report, print new dev800
# endpoints, auto-promote any EXPLORATORY-bf16 arm at >= 0.795, copy raw results and commit
# them (explicit paths only). Bounded; safe to re-run.
WT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$WT" || exit 2
L="python3 -I scripts/exploratory_sweep/gpu_lane.py"
timeout 1500 $L fetch > /dev/null 2>&1
bash scripts/exploratory_sweep/gpu_ops/report_uw.sh > /dev/null || echo "REPORT FAILED (UW)"
python3 -I scripts/exploratory_sweep/gpu_ops/new_endpoints.py
timeout 900 $L promote
timeout 300 $L raw > /dev/null
git add docs/aircraft-exploratory-sweep-v2-gpu/raw docs/aircraft-exploratory-sweep-v2-gpu/promoted-arms.json 2>/dev/null
git diff --cached --quiet || git commit -q -m "GPU lane raw: overnight tick $(date -u +%FT%TZ)"
timeout 120 git push -q 2>&1 | tail -1
