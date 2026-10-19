#!/bin/bash
set -uo pipefail
VENV="$(cd .. && pwd)/venv"
uv venv -q -p 3.12.15 "$VENV" || exit 3
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1+cpu torchvision==0.29.1+cpu || exit 3
grep -v -E '^(torch|torchvision)==' docs/qwen3.8-27b-hf-cpu-freeze.txt > freeze-rest.txt
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -r freeze-rest.txt || exit 3
for c in pinned dnnavx unpinned; do
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 nice -n 10 "$VENV/bin/python" -I fp_probe.py "$c" 2>&1 | tail -1
done
