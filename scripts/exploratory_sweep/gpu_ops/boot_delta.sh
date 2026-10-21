#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# Mac-side: re-bootstrap lane hosts with payload NEW, shipping only files that are new or
# changed versus payload OLD (the Mac's link to UW can be a slow phone hotspot). On the host,
# every other file listed in NEW's PAYLOAD.json is hardlinked (or copied across filesystems)
# from the host's existing payload OLDPAY; bootstrap_host.sh then mirrors it and
# host_verify.py re-hashes EVERY file against NEW's PAYLOAD.json and refuses extras, so a
# stale or missing file refuses, never passes.
#   boot_delta.sh OLD NEW HOST...
# OLD/NEW: payload tags under runs/aircraft-exploratory-sweep-v2-gpu/payload-<tag>, or absolute dirs.
# Environment (defaults = the GPU lane):
#   OLDPAY    host path of the payload OLD describes   (/var/tmp/r226lin-rlesweep-gpu/payload)
#   ROOTNAME  the lane's root name on the host          (r226lin-rlesweep-gpu)
#   RECEIPTS  receipts dir on the NFS home              ($HOME/rlesweep-gpu/receipts)
#   SCRATCH   uwrun run-dir location: disk or shm       (disk)
OLD=$1; NEW=$2; shift 2
WT=$(cd "$(dirname "$0")/../../.." && pwd)
D=$WT/runs/aircraft-exploratory-sweep-v2-gpu
case "$OLD" in /*) OLDDIR=$OLD;; *) OLDDIR=$D/payload-$OLD;; esac
case "$NEW" in /*) NEWDIR=$NEW;; *) NEWDIR=$D/payload-$NEW;; esac
TAG=$(basename "$NEWDIR" | sed 's/^payload-//')
OLDPAY=${OLDPAY:-/var/tmp/r226lin-rlesweep-gpu/payload}
ROOTNAME=${ROOTNAME:-r226lin-rlesweep-gpu}
SCRATCH=${SCRATCH:-disk}
S=$(mktemp -d /tmp/rlegpu-delta-$TAG.XXXXXX)
mkdir -p "$S/ship"
python3 -I "$WT/scripts/exploratory_sweep/gpu_ops/delta_stage.py" "$OLDDIR" "$NEWDIR" "$S/ship" || exit 2
cp "$WT/scripts/exploratory_sweep/gpu_ops/delta_assemble.py" "$S/"
printf '%s\n' 'set -e' "python3 delta_assemble.py '$OLDPAY' ship pay" 'cd pay' \
  "RLE_ROOT_NAME=$ROOTNAME RLE_RECEIPTS=\${RECEIPTS:-\$HOME/rlesweep-gpu/receipts} bash code/scripts/exploratory_sweep/bootstrap_host.sh" \
  > "$S/boot.sh"
for h in "$@"; do
  timeout 1200 ~/.claude/bin/uwrun run --host "$h" --threads 4 --timeout 5400 --scratch "$SCRATCH" \
    --disk 3 --sync "$S" --name "rlegpu-boot-$h-$TAG-$(date +%s)" -- bash boot.sh 2>&1 | tail -1
done
rm -rf "$S"
