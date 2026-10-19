#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# Mac-side: block up to $1 seconds (default 480) until any GPU-lane worker log gains a new
# "arm ... rc=" or "worker end" line, then print the new lines. Seen lines persist in
# runs/aircraft-exploratory-sweep-v2-gpu/seen.txt. Every uwrun call is bounded.
WT=$(cd "$(dirname "$0")/../../.." && pwd)
S=$WT/runs/aircraft-exploratory-sweep-v2-gpu/state.json
SEEN=$WT/runs/aircraft-exploratory-sweep-v2-gpu/seen.txt
touch "$SEEN"
end=$(( $(date +%s) + ${1:-480} ))
while [ "$(date +%s)" -lt "$end" ]; do
  new=""
  for w in $(python3 -c "import json;d=json.load(open('$S'));print(' '.join(x['name'] for v in d['workers'].values() for x in v))"); do
    lines=$(timeout 60 ~/.claude/bin/uwrun logs "$w" 2>/dev/null | grep -E "rc=|worker end|OOM" | sed "s|^|$w: |")
    while IFS= read -r l; do
      [ -z "$l" ] && continue
      grep -qxF "$l" "$SEEN" || { echo "$l" >> "$SEEN"; new="$new$l"$'\n'; }
    done <<< "$lines"
  done
  if [ -n "$new" ]; then printf '%s' "$new"; exit 0; fi
  sleep 60
done
echo "no new events"
