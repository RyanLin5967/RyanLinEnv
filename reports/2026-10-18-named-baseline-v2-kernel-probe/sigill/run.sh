#!/bin/bash
# EW=<n elementwise subprocesses> MODEL=<n model subprocesses> bash run.sh
set -uo pipefail
VENV="$(cd .. && pwd)/venv"
uv venv -q -p 3.12.15 "$VENV" || exit 3
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1+cpu torchvision==0.29.1+cpu || exit 3
grep -v -E '^(torch|torchvision)==' docs/qwen3.8-27b-hf-cpu-freeze.txt > ../freeze-rest.txt
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -r ../freeze-rest.txt || exit 3
echo "host=$(hostname) start=$(date -u +%FT%TZ) $(grep -m1 'model name' /proc/cpuinfo)"
crash=0; ok=0
for i in $(seq 1 "${EW:-200}"); do
  "$VENV/bin/python" -I ew_probe.py 32 "$i" > ../ew-$i.out 2> ../ew-$i.err; rc=$?
  if [ $rc = 0 ]; then ok=$((ok+1)); else crash=$((crash+1)); echo "EW run $i rc=$rc"; tail -2 ../ew-$i.err; fi
done
echo "EW_DONE ok=$ok crash=$crash $(date -u +%FT%TZ)"
ids=(0090868 0447760 0489934 1158113)
mok=0; mcrash=0
for i in $(seq 1 "${MODEL:-50}"); do
  id=${ids[$(( (i-1) % 4 ))]}
  "$VENV/bin/python" -I model_probe.py /dev/shm/r226lin-qwen38-27b-1d4bf0f2 "$id" \
    > ../model-$i.out 2> ../model-$i.err; rc=$?
  if [ $rc = 0 ]; then mok=$((mok+1)); echo "MODEL run $i ok $(tail -1 ../model-$i.out)";
  else mcrash=$((mcrash+1)); echo "MODEL run $i rc=$rc"; tail -3 ../model-$i.err; fi
done
echo "MODEL_DONE ok=$mok crash=$mcrash $(date -u +%FT%TZ)"
