#!/bin/bash
# Fetch one run's small results (summary, assays with per-image predictions, logs, receipts) into
# docs/aircraft-sft-ceiling-v1/raw/<ARM>/results, replacing the previous copy only after a complete
# fetch. Usage: fetch_one.sh RUN ARM
set -euo pipefail
RUN=$1; ARM=$2
RAW=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/docs/aircraft-sft-ceiling-v1/raw/$ARM
TMP=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/runs/aircraft-sft-ceiling-v1/fetch/$RUN-$(date +%s)
mkdir -p "$(dirname "$TMP")"
timeout 900 ~/.claude/bin/uwrun fetch "$RUN" work/results/ "$TMP"
[ -f "$TMP/summary.json" ] || { echo "REFUSE: $RUN has no summary.json yet" >&2; exit 2; }
rm -rf "$RAW.new"; mkdir -p "$RAW.new"; cp -a "$TMP" "$RAW.new/results"
echo "$RUN" > "$RAW.new/RUN"
rm -rf "$RAW"; mv "$RAW.new" "$RAW"
ls "$RAW/results/assays"
