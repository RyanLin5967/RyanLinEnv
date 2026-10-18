#!/bin/bash
# GPU lane worker: run queued arm specs on this host's single GPU, one at a time.
#
#   uwrun run --host H --threads 4 --timeout 86400 --name NAME -- \
#       bash -c 'R=/var/tmp/r226lin-rlesweep-gpu; exec bash "$R/payload/code/scripts/exploratory_sweep/gpu_worker.sh" "$R"'
#
# Queue (host-local): ROOT/gpuq/{pending,running,done,failed}/<NN-id>.json. The worker takes
# the lexically first pending spec, moves it to running/ (mv is atomic on one filesystem, so
# two workers on one host cannot take the same spec), runs arm_exec.py under a hard timeout,
# then files the spec under done/ or failed/ by exit code. Results for each arm go to
# $PWD/results/<NN-id>/ (the uwrun run dir's work/results, fetched from the Mac).
# It exits when ROOT/gpuq/STOP exists (checked between arms) or after IDLE_MAX seconds
# with an empty queue. A second worker on the same host refuses (flock on ROOT/gpuq/.worker).
# Per arm it records nvidia-smi before, a 10 s GPU memory/utilisation trace, and worker.json
# (rc, start, end). Blind spot: other users' GPU processes are recorded, not prevented.
set -u
R=$1
RES=${2:-$PWD/results}
Q=$R/gpuq
IDLE_MAX=${IDLE_MAX:-21600}
ARM_TIMEOUT=${ARM_TIMEOUT:-43200}
mkdir -p "$Q/pending" "$Q/running" "$Q/done" "$Q/failed" "$RES"
[ -f "$R/RECEIPT.json" ] || { echo "REFUSE: $R has no RECEIPT.json" >&2; exit 2; }
exec 9>"$Q/.worker"; flock -n 9 || { echo "REFUSE: another worker holds $Q" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=0
PY=$R/venv/bin/python
EXEC=$R/payload/code/scripts/exploratory_sweep/arm_exec.py
# A spec left in running/ by a dead worker goes back to the front of the queue.
for f in "$Q"/running/*.json; do [ -e "$f" ] && mv "$f" "$Q/pending/"; done
idle=0
waited=0
echo "worker start $(date -u +%FT%TZ) host=$(hostname -s) root=$R results=$RES"
while :; do
  if [ -f "$Q/STOP" ]; then echo "STOP file present; exiting"; break; fi
  next=$(ls "$Q/pending" 2>/dev/null | grep '\.json$' | sort | head -1)
  if [ -z "$next" ]; then
    if [ "$idle" -ge "$IDLE_MAX" ]; then echo "idle ${idle}s; exiting"; break; fi
    sleep 60; idle=$((idle + 60)); continue
  fi
  idle=0
  # Shared GPUs: another user's job can hold the card (eceubuntu2, 2026-10-18: a 7 GB
  # Qwen3.5 eval OOMed four queued arms in six minutes). Start an arm only when the card
  # has NEED_MIB free; otherwise wait, logging once per 10 minutes.
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -lt "${NEED_MIB:-6500}" ]; then
    [ $((waited % 600)) -eq 0 ] && echo "gpu busy: ${free} MiB free < ${NEED_MIB:-6500}; $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') $(date -u +%FT%TZ)"
    sleep 60; waited=$((${waited:-0} + 60)); continue
  fi
  waited=0
  mv "$Q/pending/$next" "$Q/running/$next" || continue
  key=${next%.json}
  out=$RES/$key
  mkdir -p "$out"
  nvidia-smi > "$out/nvsmi-before.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 \
    > "$out/gpu-trace.csv" 2>/dev/null &
  trace=$!
  start=$(date +%s)
  echo "arm $key start $(date -u +%FT%TZ)"
  timeout -k 60 "$ARM_TIMEOUT" "$PY" -I "$EXEC" --root "$R" --arm-json "$Q/running/$next" \
    --results "$out" > "$out/arm_exec.out" 2> "$out/arm_exec.err"
  rc=$?
  kill "$trace" 2>/dev/null; wait "$trace" 2>/dev/null
  end=$(date +%s)
  printf '{"key": "%s", "rc": %d, "start": %d, "end": %d, "host": "%s"}\n' \
    "$key" "$rc" "$start" "$end" "$(hostname -s)" > "$out/worker.json"
  if [ "$rc" -eq 0 ]; then mv "$Q/running/$next" "$Q/done/$next"
  else mv "$Q/running/$next" "$Q/failed/$next"; fi
  echo "arm $key rc=$rc wall=$((end - start))s"
done
echo "worker end $(date -u +%FT%TZ)"
