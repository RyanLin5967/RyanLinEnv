#!/bin/bash
# Host environment for the RL-campaign-I pool scorer. Adapted from
# scripts/exploratory_sweep/bootstrap_host.sh (branch exploratory-sweep-v2.noindex, 9d9a09f) with
# two deliberate differences:
#   * ROOT is /var/tmp/r226lin-pools (never the sweep's r226lin-rlesweep root), and
#   * the receipt is NOT copied to ~/rlesweep/receipts, so the sweep runner never places arms here.
# Run by uwrun with the sweep payload (runs/aircraft-exploratory-sweep-v2/payload-9d9a09f) as --sync:
#   uwrun run --host H --threads 8 --timeout 5400 --sync PAYLOAD --name rlpools-setup-H -- \
#       bash ~/rlpools/v1/bin/setup_host.sh [PAYLOAD_DIR]
# PAYLOAD_DIR (default: the cwd) may instead be a verified payload copy staged on the NFS home,
# e.g. ~/rlpools/payload-c44802e copied from a sweep host (its vlm_local_training.py, all_train and
# adapters are byte-identical to 9d9a09f's; host_verify checks it against its own PAYLOAD.json).
# (this script is staged on the NFS home, outside the payload, because host_verify refuses any
# file the payload does not list)
# The payload's own host_verify.py verifies every payload file, the package pins and the model,
# and writes ROOT/RECEIPT.json. Anything wrong refuses (non-zero exit).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
PAY=$(cd "${1:-.}" && pwd); cd "$PAY"
HOST=$(hostname -s | tr 'A-Z' 'a-z')
ROOT=/var/tmp/r226lin-pools
free_gb() { df -P -BG "$1" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}'; }
if [ ! -f "$ROOT/RECEIPT.json" ] && [ "$(free_gb /var/tmp)" -lt 20 ]; then
  echo "REFUSE: < 20 GB free in /var/tmp on $HOST" >&2; exit 3; fi
mkdir -p "$ROOT"; echo "ROOT=$ROOT host=$HOST"
exec 9>"$ROOT/.setup.lock"; flock -n 9 || { echo "REFUSE: another setup holds $ROOT" >&2; exit 4; }
grep -q -w avx2 /proc/cpuinfo || { echo "REFUSE: $HOST has no AVX2" >&2; exit 5; }
PY=$(sed -n 's/^ *"python": "\([0-9.]*\)".*/\1/p' PAYLOAD.json | head -1)
[ -n "$PY" ] || { echo "REFUSE: no python pin in PAYLOAD.json" >&2; exit 6; }
export UV_CACHE_DIR=$ROOT/uvcache UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python
V=$ROOT/venv
INDEX=https://download.pytorch.org/whl/cpu
if [ ! -x $V/bin/python ] || ! $V/bin/python -c 'import torch,trl,transformers,peft' 2>/dev/null; then
  rm -rf "$V"; timeout 900 uv venv -q -p "$PY" "$V"
  timeout 1800 uv pip install -q --python $V/bin/python --index-url $INDEX torch==2.14.1 torchvision==0.29.1
  timeout 1800 uv pip install -q --python $V/bin/python --no-deps --require-hashes -r code/requirements-vlm-local.lock
fi
rm -rf "$UV_CACHE_DIR"
mkdir -p "$ROOT/payload"
rsync -a --delete --exclude '__pycache__' "$PAY"/ "$ROOT/payload"/
rm -f "$ROOT/payload"/{cmd,rc,pid,started,ended,host,launch.sh,run.log} 2>/dev/null || true
timeout 3600 $V/bin/python -I "$ROOT/payload/code/scripts/exploratory_sweep/host_verify.py" \
  --root "$ROOT" --kind cpu --host "$HOST"
echo "POOLS SETUP OK $HOST $ROOT"
