#!/bin/bash
# Workaround matrix for the MKL VML AVX-512 SIGILL; N subprocesses per variant.
set -uo pipefail
VENV="$(cd .. && pwd)/venv"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q -p 3.12.15 "$VENV" || exit 3
  VIRTUAL_ENV="$VENV" uv pip install -q --no-deps --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1+cpu torchvision==0.29.1+cpu || exit 3
  grep -v -E '^(torch|torchvision)==' docs/qwen3.8-27b-hf-cpu-freeze.txt > ../freeze-rest.txt
  VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -r ../freeze-rest.txt || exit 3
fi
N=${N:-100}
variant() {  # name, threads, extra env...
  local name=$1 threads=$2; shift 2
  local crash=0
  for i in $(seq 1 "$N"); do
    env "$@" "$VENV/bin/python" -I ew_probe.py "$threads" "${SEED:-$i}" ${WARM:-} > /dev/null 2>&1
    [ $? = 132 ] && crash=$((crash+1))
  done
  echo "VARIANT $name threads=$threads crashes=$crash/$N $(date -u +%T)"
}
variant seed4-fixed 32 SEED_UNUSED=1
SEED=4 variant same-seed-4 32 X=1
variant threads1 1 X=1
variant mkl-threads1 32 MKL_NUM_THREADS=1
WARM=warm variant warm-up 32 X=1
echo ALL_DONE
