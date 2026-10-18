#!/bin/bash
# Per-host root for the SFT-ceiling lane (aircraft-sft-ceiling-v1), separate from the CPU sweep.
# Derived from scripts/exploratory_sweep/bootstrap_host.sh (branch exploratory-sweep-v2.noindex,
# 9d9a09f); differences, and only these:
#   * ROOT is r226lin-sftceil (never the sweep's r226lin-rlesweep), receipt goes to
#     ~/sftceil/receipts/HOST.json (never ~/rlesweep/receipts, which the sweep runner reads).
#   * When the sweep's root on this host holds a verified venv/model, they are HARDLINKED
#     (cp -al) into our root on the same filesystem: zero extra disk/RAM, and the sweep's files
#     are never written (the two model meta files host_verify rewrites are unlinked first, and
#     rsync of payload is into our own dir). The sweep refreshing its payload cannot reach us.
#   * The synced dir carries this script under sftceil/, excluded from ROOT/payload so the
#     payload verifier still sees exactly PAYLOAD.json's files.
# Run:  uwrun run --host H --threads 4 --sync STAGE --name sftceil-boot-H-N -- bash sftceil/bootstrap_host.sh
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
PAY=$(pwd)
HOST=$(hostname -s | tr 'A-Z' 'a-z')
NEED_GB=14
free_gb() { df -P -BG "$1" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}'; }
pick_root() {
  for r in /var/tmp/r226lin-sftceil /dev/shm/r226lin-sftceil; do
    [ -f "$r/RECEIPT.json" ] && { echo "$r"; return; }
  done
  # same filesystem as an existing sweep root first: hardlinks cost nothing
  for base in /dev/shm /var/tmp; do
    if [ -f "$base/r226lin-rlesweep/RECEIPT.json" ] && [ -x "$base/r226lin-rlesweep/venv/bin/python" ]; then
      echo "$base/r226lin-sftceil"; return; fi
  done
  if [ "$(free_gb /var/tmp)" -ge $NEED_GB ]; then echo /var/tmp/r226lin-sftceil; return; fi
  local avail; avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ -d /dev/shm ] && [ "$(free_gb /dev/shm)" -ge $NEED_GB ] && [ "$avail" -ge 64 ]; then
    echo /dev/shm/r226lin-sftceil; return; fi
  echo "REFUSE: no local root with ${NEED_GB} GB free on $HOST" >&2; exit 3
}
ROOT=$(pick_root); mkdir -p "$ROOT"; echo "ROOT=$ROOT host=$HOST"
exec 9>"$ROOT/.bootstrap.lock"; flock -n 9 || { echo "REFUSE: another bootstrap holds $ROOT" >&2; exit 4; }
# No-AVX2 hosts only on explicit opt-in (coordinator 2026-10-18: the trainer ran on iccad3 without
# SIGILL; numerics check against an AVX2 host pending, so such results are tagged until it passes).
if ! grep -q -w avx2 /proc/cpuinfo; then
  [ "${ALLOW_NO_AVX2:-0}" = 1 ] || { echo "REFUSE: $HOST has no AVX2" >&2; exit 5; }
  echo "WARNING-TAG: $HOST has no AVX2 (ALLOW_NO_AVX2=1): numerics pending"
fi
SWEEP=$(dirname "$ROOT")/r226lin-rlesweep
if [ -f "$SWEEP/RECEIPT.json" ]; then
  if [ ! -d "$ROOT/venv" ] && "$SWEEP/venv/bin/python" -c 'import torch,trl,transformers,peft' 2>/dev/null; then
    cp -al "$SWEEP/venv" "$ROOT/venv"; echo "venv hardlinked from $SWEEP"; fi
  if [ ! -d "$ROOT/model" ] && [ -d "$SWEEP/model" ]; then
    cp -al "$SWEEP/model" "$ROOT/model"
    # host_verify copy2()s these two onto the model dir: unlink so the sweep's inodes are never written
    rm -f "$ROOT"/model/*/model-source.json "$ROOT"/model/*/.pinned-download.json
    echo "model hardlinked from $SWEEP"; fi
fi
PY=$(python3 -c 'import json;print(json.load(open("PAYLOAD.json"))["pins"]["python"])' 2>/dev/null || \
     grep -o '"python": "[0-9.]*"' PAYLOAD.json | head -1 | grep -o '[0-9.]*$')
export UV_CACHE_DIR=$ROOT/uvcache UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python
V=$ROOT/venv
KIND=cpu; INDEX=https://download.pytorch.org/whl/cpu
if [ ! -x $V/bin/python ] || ! $V/bin/python -c 'import torch,trl,transformers,peft' 2>/dev/null; then
  rm -rf "$V"; timeout 900 uv venv -q -p "$PY" "$V"
  timeout 1800 uv pip install -q --python $V/bin/python --index-url $INDEX torch==2.14.1 torchvision==0.29.1
  timeout 1800 uv pip install -q --python $V/bin/python --no-deps --require-hashes -r code/requirements-vlm-local.lock
fi
rm -rf "$UV_CACHE_DIR"
if [ -f "$PAY/sftceil/DELTA" ]; then
  # Delta stage (slow uplink): the synced dir holds PAYLOAD.json plus only the files that differ
  # from the sweep payload already on this host. Seed from the sweep payload by hardlink, overlay
  # the delta (rsync writes temp+rename: new inodes, the sweep's files are never written), unlink
  # anything PAYLOAD.json does not list. host_verify below re-hashes every file and refuses on any
  # mismatch, so a host whose sweep payload is a different version fails loudly, never silently.
  [ -d "$SWEEP/payload" ] || { echo "REFUSE: delta stage needs $SWEEP/payload" >&2; exit 6; }
  [ -d "$ROOT/payload" ] || cp -al "$SWEEP/payload" "$ROOT/payload"
  rsync -a --exclude 'sftceil/' --exclude tmp/ "$PAY"/ "$ROOT/payload"/
  $V/bin/python -I - "$ROOT/payload" <<'PYEOF'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1]); keep = set(json.loads((root / "PAYLOAD.json").read_text())["files"]) | {"PAYLOAD.json"}
for p in sorted(root.rglob("*")):
    if (p.is_file() or p.is_symlink()) and str(p.relative_to(root)) not in keep:
        p.unlink(); print("unlinked extra", p.relative_to(root))
PYEOF
else
  mkdir -p "$ROOT/payload"
  rsync -a --delete --exclude '__pycache__' --exclude 'sftceil/' \
    --exclude cmd --exclude rc --exclude pid --exclude started --exclude ended --exclude host \
    --exclude launch.sh --exclude run.log --exclude tmp/ "$PAY"/ "$ROOT/payload"/
fi
timeout 3600 $V/bin/python -I "$ROOT/payload/code/scripts/exploratory_sweep/host_verify.py" \
  --root "$ROOT" --kind $KIND --host "$HOST"
mkdir -p "$HOME/sftceil/receipts"
cp "$ROOT/RECEIPT.json" "$HOME/sftceil/receipts/$HOST.json"
echo "BOOTSTRAP OK $HOST $ROOT"
