#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# Run `gpu_lane.py report` on a UW CPU host instead of the Mac (the paired bootstraps are
# CPU-heavy pure Python and the Mac's fan guard freezes them), then copy report.json and the
# bootstrap cache back into runs/<study>/. Ships only the scripts, dev800 gold, the arm table
# and the fetched result dirs (assays/summary/arm.json). Exit non-zero if no report came back.
WT=$(cd "$(dirname "$0")/../../.." && pwd)
D=runs/aircraft-exploratory-sweep-v2-gpu
T=$(mktemp -d /tmp/rlegpu-report.XXXXXX)
mkdir -p "$T/scripts/exploratory_sweep" "$T/docs/aircraft-exploratory-sweep-v2/manifests" "$T/$D"
cp "$WT"/scripts/exploratory_sweep/*.py "$T/scripts/exploratory_sweep/"
cp "$WT/docs/aircraft-exploratory-sweep-v2/manifests/dev800.jsonl" "$T/docs/aircraft-exploratory-sweep-v2/manifests/"
cp -R "$WT/docs/aircraft-exploratory-sweep-v2-gpu" "$T/docs/"
[ -f "$WT/$D/bootstrap-cache.json" ] && cp "$WT/$D/bootstrap-cache.json" "$T/$D/"
(cd "$WT/$D" && rsync -a --include='*/' --include='arm.json' --include='summary.json' \
  --include='worker.json' --include='assays/*.json' --exclude='*' results/ "$T/$D/results/")
printf '%s\n' 'cd "$(pwd -P)"' \
  '~/.local/bin/uv run -q -p 3.12 --no-project python -I scripts/exploratory_sweep/gpu_lane.py report > report.out 2>&1; echo rc=$?' \
  'mkdir -p out && cp runs/aircraft-exploratory-sweep-v2-gpu/report.json runs/aircraft-exploratory-sweep-v2-gpu/bootstrap-cache.json report.out out/ 2>/dev/null' \
  > "$T/run.sh"
name=rlegpu-report-$(date +%s)
timeout 1500 ~/.claude/bin/uwrun run --pool cpu --threads 1 --timeout 1200 --sync "$T" --name "$name" --wait -- bash run.sh > /dev/null 2>&1
R=$HOME/.claude/uw/results/$name
tail -1 "$R/run.log"
if [ -s "$R/work/out/report.json" ]; then
  cp "$R/work/out/report.json" "$R/work/out/bootstrap-cache.json" "$WT/$D/"
  rm -rf "$T"; exit 0
fi
tail -5 "$R/work/report.out" 2>/dev/null; rm -rf "$T"; exit 2
