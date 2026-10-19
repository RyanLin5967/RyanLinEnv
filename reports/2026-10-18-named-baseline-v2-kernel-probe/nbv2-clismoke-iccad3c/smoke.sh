#!/bin/bash
# The documented invocation, end to end: lock venv only (no project install), python -I, no PYTHONPATH.
set -uo pipefail
VENV="$(cd .. && pwd)/venv"
uv venv -q -p 3.12.15 "$VENV" || exit 3
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1+cpu torchvision==0.29.1+cpu || exit 3
grep -v -E '^(torch|torchvision)==' docs/qwen3.8-27b-hf-cpu-freeze.txt > ../freeze-rest.txt
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -r ../freeze-rest.txt || exit 3
env | grep -E "PYTHONPATH|ONEDNN|MKL_CBWR|ATEN" || echo "no kernel or path env set"
"$VENV/bin/python" -I scripts/aircraft_named_baseline_v2.py smoke \
  --plan docs/aircraft-named-baseline-plan-v2.json --weights /dev/shm/r226lin-qwen38-27b-1d4bf0f2 \
  --ids 0447760 --threads 32 --dtype float32 --out ../smoke.json --repository . 2> ../smoke.err | tail -1
echo "smoke_rc=${PIPESTATUS[0]}"
tail -3 ../smoke.err
jq -c '{host, kernel_fingerprint, answers: [.results[].text], latency: [.results[].latency_seconds], numerics: .load_info.device.cpu_numerics}' ../smoke.json
