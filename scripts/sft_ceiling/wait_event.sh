#!/bin/bash
# Block until something changes: any lane run leaves RUNNING, or any run's results/summary.json
# gains an endpoint. Prints the change and exits 0. Polls every POLL seconds (default 600).
# State (last seen) in runs/aircraft-sft-ceiling-v1/wait_event.state. Read-only on the hosts.
POLL=${1:-600}
STATE=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/runs/aircraft-sft-ceiling-v1/wait_event.state
RUNS="sftceil-cont-all-lr1e-5-1:ecetesla0 sftceil-cont-all-lr2e-5-2:iccad6 sftceil-cont-all-lr5e-6-1:iccad7
sftceil-I-seed1-1:iccad8 sftceil-I-seed2-1:iccad10 sftceil-I-seed3-1:iccad13 sftceil-I-seed42-1:iccad5"
snapshot() {
  for pair in $RUNS; do
    run=${pair%%:*}; host=${pair##*:}
    st=$(timeout 90 ~/.claude/bin/uwrun status "$run" 2>/dev/null | tail -1 | awk '{print $3,$4}')
    case "$st" in UNREACHABLE*|"") st=RUNNING;; RUNNING*) st=RUNNING;; esac
    eps=$(timeout 90 ~/.claude/bin/uwrun ssh "$host" "grep -o '\"label\": \"[a-z0-9]*\"' ~/runs/$run/work/results/summary.json 2>/dev/null | wc -l" 2>/dev/null | tail -1)
    [[ "$eps" =~ ^[0-9]+$ ]] || eps=keep
    echo "$run $st eps=$eps"
  done
}
[ -f "$STATE" ] || snapshot > "$STATE"
while true; do
  now=$(snapshot)
  # an unreadable endpoint count keeps the previous value (flaky link), never a false change
  merged=$(paste -d' ' <(echo "$now") "$STATE" | awk '{ if ($3=="eps=keep") $3=$6; print $1, $2, $3 }')
  if [ "$merged" != "$(cat "$STATE")" ]; then
    diff <(cat "$STATE") <(echo "$merged") | grep '^>' ; echo "$merged" > "$STATE"; exit 0
  fi
  sleep "$POLL"
done
