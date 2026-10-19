#!/bin/bash
# Usage (uwrun, cwd = work/): RUNS="pinned:32 unpinned:32" IDS="0447760 ..." NICE=0 bash run.sh
set -uo pipefail
W=/dev/shm/r226lin-qwen38-27b-1d4bf0f2
VENV="$(cd .. && pwd)/venv"
echo "host=$(hostname) start=$(date -u +%FT%TZ) runs=$RUNS ids=$IDS nice=${NICE:-0}"
free -g | head -2; cat /proc/loadavg
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q -p 3.12.15 "$VENV" || exit 3
  VIRTUAL_ENV="$VENV" uv pip install -q --no-deps --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1+cpu torchvision==0.29.1+cpu || exit 3
  grep -v -E '^(torch|torchvision)==' docs/qwen3.8-27b-hf-cpu-freeze.txt > freeze-rest.txt
  VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -r freeze-rest.txt || exit 3
fi
VIRTUAL_ENV="$VENV" uv pip freeze > freeze-observed.txt
if diff docs/qwen3.8-27b-hf-cpu-freeze.txt freeze-observed.txt; then echo FREEZE_MATCH; else echo FREEZE_DIFFERS; exit 4; fi
"$VENV/bin/python" -VV
if [ ! -f "$W/model-00018-of-00018.safetensors" ]; then
  mkdir -p "$W"
  "$VENV/bin/hf" download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --local-dir "$W" 2>&1 | tail -3
fi
du -sh "$W"
mkdir -p out
for spec in $RUNS; do
  cfg=${spec%%:*}; threads=${spec##*:}
  tag="$cfg-t$threads-$(date -u +%H%M%S)"
  avail=$(free -g | awk '/Mem:/{print $7}')
  echo "=== $tag $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) availGB=$avail"
  if [ "$avail" -lt "${MIN_AVAIL_GB:-200}" ]; then echo "REFUSED: availGB $avail < ${MIN_AVAIL_GB:-200}"; exit 5; fi
  OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads nice -n "${NICE:-0}" \
    "$VENV/bin/python" -I kernel_probe.py --config "$cfg" --threads "$threads" --ids $IDS \
    --decode-tokens "${DECODE:-16}" --weights "$W" --out "out/$tag.json" 2> "out/$tag.err" \
    | tail -1
  echo "rc=$? end=$(date -u +%FT%TZ)"; tail -2 "out/$tag.err"
done
echo "ALL_DONE $(date -u +%FT%TZ)"
