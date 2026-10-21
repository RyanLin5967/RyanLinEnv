#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# Mac-side: stage the lane scripts + tests and run pytest on a UW CPU host (never on the Mac,
# quiet mode). Prints the pytest tail from the fetched run.log. tests that import the full
# trainer (vision_rl + torch) fail here for environment reasons only.
WT=$(cd "$(dirname "$0")/../../.." && pwd)
T=$(mktemp -d /tmp/rlegpu-lanetest.XXXXXX)
mkdir -p "$T/scripts/exploratory_sweep" "$T/tests" "$T/docs" "$T/src/vision_rl"
cp "$WT"/scripts/exploratory_sweep/*.py "$T/scripts/exploratory_sweep/"
cp "$WT/tests/test_gpu_lane.py" "$WT/tests/test_exploratory_sweep.py" "$T/tests/"
cp "$WT"/scripts/exploratory_sweep/*.json "$T/scripts/exploratory_sweep/"
cp -R "$WT/docs/aircraft-exploratory-sweep-v2" "$WT/docs/aircraft-exploratory-sweep-v2-gpu" "$T/docs/"
printf '%s\n' 'cd "$(pwd -P)"' \
  '~/.local/bin/uv run -q -p 3.12 --no-project --with pytest python -m pytest -q -p no:cacheprovider tests/test_gpu_lane.py tests/test_exploratory_sweep.py 2>&1 | tail -25' > "$T/run.sh"
name=rlegpu-lanetest-$(date +%s)
timeout 900 ~/.claude/bin/uwrun run --pool cpu --threads 2 --timeout 900 --sync "$T" --name "$name" --wait -- bash run.sh >/dev/null 2>&1
echo "uwrun rc=$? ($name)"
tail -8 "$HOME/.claude/uw/results/$name/run.log"
rm -rf "$T"
