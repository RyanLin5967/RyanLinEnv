#!/bin/bash
# Block until ARM on HOST appends a new optimizer-audit line (any change of the last line), or its
# run RUN stops. Prints the new last line. Usage: wait_new_step.sh ARM HOST RUN
ARM=$1; HOST=$2; RUN=$3
last() { timeout 60 ~/.claude/bin/uwrun ssh "$HOST" "cat /dev/shm/r226lin-sftceil/arms/$ARM/train/optimizer_audit.jsonl /var/tmp/r226lin-sftceil/arms/$ARM/train/optimizer_audit.jsonl 2>/dev/null | tail -1 | cut -c1-120" 2>/dev/null | tail -1; }
start=$(last)
while true; do
  sleep 60
  now=$(last)
  [ -n "$now" ] && [ "$now" != "$start" ] && { echo "$ARM new: $now"; exit 0; }
  st=$(timeout 60 ~/.claude/bin/uwrun status "$RUN" 2>/dev/null | tail -1)
  case "$st" in *RUNNING*|*UNREACHABLE*|"") ;; *) echo "$ARM run stopped: $st"; exit 1;; esac
done
