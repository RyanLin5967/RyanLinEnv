#!/bin/bash
# Wait for an earlier run on this host (its rc file), then probe the given configs.
while [ -n "${WAITFOR:-}" ] && [ ! -f "$HOME/runs/$WAITFOR/rc" ]; do sleep 60; done
RUNS="$RUNS" IDS="0447760 1158113 0090868 0489934" DECODE=16 MIN_AVAIL_GB=${MIN_AVAIL_GB:-200} NICE=${NICE:-0} exec bash run.sh
