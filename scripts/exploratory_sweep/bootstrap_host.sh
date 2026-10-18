#!/bin/bash
# Idempotent per-host environment for the exploratory sweep. Run by uwrun with the staged
# payload (payload.py --out DIR) as its synced work dir:
#
#   uwrun run --host HOST --threads 8 --timeout 3600 --sync PAYLOAD_DIR \
#       --name rlesweep-boot-HOST-N -- bash code/scripts/exploratory_sweep/bootstrap_host.sh
#
# Builds, under ROOT (host-local; never the 16 GB NFS home):
#   ROOT/venv      uv venv -p <reference python>, torch/torchvision from the official CPU
#                  (or CUDA 13.0, when nvidia-smi sees a GPU) index, the rest from
#                  requirements-vlm-local.lock (--require-hashes); versions must equal the
#                  reference venv's (.venv-detection) for every shared package
#   ROOT/model     Qwen2-VL-2B-Instruct at the pinned revision, verified file by file
#   ROOT/payload   code + TRAIN-only data + W/I adapters, every file sha256-verified, no extras
#   ROOT/RECEIPT.json   everything above, hashed; copied to ~/rlesweep/receipts/HOST.json
# ROOT is /var/tmp/r226lin-rlesweep when /var/tmp has >= 14 GB free (or already holds ROOT),
# else /dev/shm/r226lin-rlesweep when shm has >= 14 GB and MemAvailable >= 64 GB; otherwise
# it refuses. Each step re-verifies and skips when already correct, so re-running is cheap.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
PAY=$(pwd)
HOST=$(hostname -s | tr 'A-Z' 'a-z')
NEED_GB=14
free_gb() { df -P -BG "$1" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}'; }
pick_root() {
  for r in /var/tmp/r226lin-rlesweep /dev/shm/r226lin-rlesweep; do
    [ -f "$r/RECEIPT.json" ] && { echo "$r"; return; }
  done
  if [ "$(free_gb /var/tmp)" -ge $NEED_GB ]; then echo /var/tmp/r226lin-rlesweep; return; fi
  local avail; avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ -d /dev/shm ] && [ "$(free_gb /dev/shm)" -ge $NEED_GB ] && [ "$avail" -ge 64 ]; then
    echo /dev/shm/r226lin-rlesweep; return; fi
  echo "REFUSE: no local root with ${NEED_GB} GB free on $HOST" >&2; exit 3
}
ROOT=$(pick_root); mkdir -p "$ROOT"; echo "ROOT=$ROOT host=$HOST"
exec 9>"$ROOT/.bootstrap.lock"; flock -n 9 || { echo "REFUSE: another bootstrap holds $ROOT" >&2; exit 4; }
grep -q -w avx2 /proc/cpuinfo || { echo "REFUSE: $HOST has no AVX2; torch $HOST CPU wheel dies with SIGILL (rc 132)" >&2; exit 5; }
PY=$(python3 -c 'import json;print(json.load(open("PAYLOAD.json"))["pins"]["python"])' 2>/dev/null || \
     grep -o '"python": "[0-9.]*"' PAYLOAD.json | head -1 | grep -o '[0-9.]*$')
export UV_CACHE_DIR=$ROOT/uvcache UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python
V=$ROOT/venv
if nvidia-smi -L >/dev/null 2>&1; then KIND=cuda; INDEX=https://download.pytorch.org/whl/cu130
else KIND=cpu; INDEX=https://download.pytorch.org/whl/cpu; fi
if [ ! -x $V/bin/python ] || ! $V/bin/python -c 'import torch,trl,transformers,peft' 2>/dev/null; then
  rm -rf "$V"; timeout 900 uv venv -q -p "$PY" "$V"
  timeout 1800 uv pip install -q --python $V/bin/python --index-url $INDEX torch==2.14.1 torchvision==0.29.1
  timeout 1800 uv pip install -q --python $V/bin/python --no-deps --require-hashes -r code/requirements-vlm-local.lock
fi
rm -rf "$UV_CACHE_DIR"
# Payload: exact mirror (rsync --delete), then a full sha256 check with no extra files.
mkdir -p "$ROOT/payload"
rsync -a --delete --exclude '__pycache__' "$PAY"/ "$ROOT/payload"/
rm -f "$ROOT/payload"/{cmd,rc,pid,started,ended,host,launch.sh,run.log} 2>/dev/null || true
timeout 3600 $V/bin/python -I "$ROOT/payload/code/scripts/exploratory_sweep/host_verify.py" \
  --root "$ROOT" --kind $KIND --host "$HOST"
mkdir -p "$HOME/rlesweep/receipts"
cp "$ROOT/RECEIPT.json" "$HOME/rlesweep/receipts/$HOST.json"
echo "BOOTSTRAP OK $HOST $ROOT"
