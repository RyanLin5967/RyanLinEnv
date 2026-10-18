#!/bin/bash
# Block until ARM on HOST has >= N optimizer-audit lines, or its uwrun run is no longer RUNNING.
# Usage: wait_steps.sh ARM HOST N [POLL_SECONDS]
ARM=$1; HOST=$2; N=$3; POLL=${4:-60}
while true; do
  n=$(timeout 60 ~/.claude/bin/uwrun ssh "$HOST" "cat /dev/shm/r226lin-sftceil/arms/$ARM/train/optimizer_audit.jsonl /var/tmp/r226lin-sftceil/arms/$ARM/train/optimizer_audit.jsonl 2>/dev/null | wc -l" 2>/dev/null | tail -1)
  [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge "$N" ] && { echo "$ARM reached $n"; exit 0; }
  st=$(timeout 60 ~/.claude/bin/uwrun status "$ARM-1" 2>/dev/null | tail -1)
  case "$st" in *RUNNING*|*UNREACHABLE*|"") ;; *) echo "$ARM stopped: $st"; exit 1;; esac
  sleep "$POLL"
done
