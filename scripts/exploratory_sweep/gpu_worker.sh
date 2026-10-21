#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# GPU lane worker: run queued arm specs on this host's single GPU, one at a time.
# (The RL campaign, scripts/rl_campaign/campaign.py, runs the same worker on CPU hosts too.)
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
# Per arm it records nvidia-smi before, a 10 s GPU memory/utilisation trace, a 10 s trace of
# the arm's process-tree resident memory (rss-trace.tsv; its peak goes into worker.json as
# peak_rss_kb) and worker.json (rc, start, end).
# Blind spot: other users' GPU processes are recorded, not prevented.
#
# Optional environment (defaults keep the GPU lane's behaviour exactly):
#   WAIT_KIND=gpu|cpu  gpu (default): start an arm only when the card has NEED_MIB free.
#                      cpu: start an arm only when the host has NEED_CORES free physical cores
#                      (physical cores minus load1); no nvidia-smi is called.
#   SLOT=N             several workers on one host share the queue; each holds its own lock
#                      (.workerN) and its own running-N/ dir, and slot N first waits N*240 s
#                      so load1 can show the arms of the slots before it.
set -u
R=$1
RES=${2:-$PWD/results}
Q=$R/gpuq
IDLE_MAX=${IDLE_MAX:-21600}
ARM_TIMEOUT=${ARM_TIMEOUT:-43200}
WAIT_KIND=${WAIT_KIND:-gpu}
SLOT=${SLOT:-}
RUN=$Q/running${SLOT:+-$SLOT}
case "$WAIT_KIND" in gpu|cpu) ;; *) echo "REFUSE: WAIT_KIND must be gpu or cpu" >&2; exit 2;; esac
mkdir -p "$Q/pending" "$RUN" "$Q/done" "$Q/failed" "$RES"
[ -f "$R/RECEIPT.json" ] || { echo "REFUSE: $R has no RECEIPT.json" >&2; exit 2; }
exec 9>"$Q/.worker$SLOT"; flock -n 9 || { echo "REFUSE: another worker holds $Q slot '$SLOT'" >&2; exit 3; }
[ "$WAIT_KIND" = gpu ] && export CUDA_VISIBLE_DEVICES=0
PY=$R/venv/bin/python
EXEC=$R/payload/code/scripts/exploratory_sweep/arm_exec.py
# A spec left in this slot's running dir by a dead worker goes back to the front of the queue.
for f in "$RUN"/*.json; do [ -e "$f" ] && mv "$f" "$Q/pending/"; done
idle=0
waited=0
PHYS=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
echo "worker start $(date -u +%FT%TZ) host=$(hostname -s) root=$R results=$RES wait=$WAIT_KIND slot='$SLOT' physical_cores=$PHYS"
[ -n "$SLOT" ] && sleep $((SLOT * 240))
# Resident memory (KB) of a process and all its descendants.
tree_rss() {
  ps -e -o pid=,ppid=,rss= | awk -v root="$1" '
    { parent[$1] = $2; rss[$1] = $3 }
    END { for (p in rss) { q = p; while (q != "" && q != 0 && q != 1) {
            if (q == root) { s += rss[p]; break } q = parent[q] } } print s + 0 }'
}
while :; do
  if [ -f "$Q/STOP" ]; then echo "STOP file present; exiting"; break; fi
  next=$(ls "$Q/pending" 2>/dev/null | grep '\.json$' | sort | head -1)
  if [ -z "$next" ]; then
    if [ "$idle" -ge "$IDLE_MAX" ]; then echo "idle ${idle}s; exiting"; break; fi
    sleep 60; idle=$((idle + 60)); continue
  fi
  idle=0
  if [ "$WAIT_KIND" = gpu ]; then
    # Shared GPUs: another user's job can hold the card (eceubuntu2, 2026-10-18: a 7 GB
    # Qwen3.5 eval OOMed four queued arms in six minutes). Start an arm only when the card
    # has NEED_MIB free; otherwise wait, logging once per 10 minutes.
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -lt "${NEED_MIB:-6500}" ]; then
      [ $((waited % 600)) -eq 0 ] && echo "gpu busy: ${free} MiB free < ${NEED_MIB:-6500}; $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ') $(date -u +%FT%TZ)"
      sleep 60; waited=$((${waited:-0} + 60)); continue
    fi
  else
    # Shared CPU hosts: start an arm only when NEED_CORES physical cores are free.
    load=$(cut -d' ' -f1 /proc/loadavg)
    free=$(awk -v p="$PHYS" -v l="$load" 'BEGIN { printf "%d", p - l }')
    if [ "$free" -lt "${NEED_CORES:-16}" ]; then
      [ $((waited % 600)) -eq 0 ] && echo "cpu busy: ${free} free physical cores < ${NEED_CORES:-16} (load1 $load of $PHYS) $(date -u +%FT%TZ)"
      sleep 60; waited=$((${waited:-0} + 60)); continue
    fi
  fi
  waited=0
  mv "$Q/pending/$next" "$RUN/$next" || continue
  key=${next%.json}
  out=$RES/$key
  mkdir -p "$out"
  trace=
  if [ "$WAIT_KIND" = gpu ]; then
    nvidia-smi > "$out/nvsmi-before.txt" 2>&1
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 \
      > "$out/gpu-trace.csv" 2>/dev/null &
    trace=$!
  fi
  start=$(date +%s)
  echo "arm $key start $(date -u +%FT%TZ)"
  timeout -k 60 "$ARM_TIMEOUT" "$PY" -I "$EXEC" --root "$R" --arm-json "$RUN/$next" \
    --results "$out" > "$out/arm_exec.out" 2> "$out/arm_exec.err" &
  armpid=$!
  peak=0
  printf 'epoch\trss_kb\n' > "$out/rss-trace.tsv"
  while kill -0 "$armpid" 2>/dev/null; do
    rss=$(tree_rss "$armpid")
    [ "$rss" -gt "$peak" ] && peak=$rss
    printf '%s\t%s\n' "$(date +%s)" "$rss" >> "$out/rss-trace.tsv"
    sleep 10
  done
  wait "$armpid"
  rc=$?
  [ -n "$trace" ] && { kill "$trace" 2>/dev/null; wait "$trace" 2>/dev/null; }
  end=$(date +%s)
  printf '{"key": "%s", "rc": %d, "start": %d, "end": %d, "host": "%s", "peak_rss_kb": %d}\n' \
    "$key" "$rc" "$start" "$end" "$(hostname -s)" "$peak" > "$out/worker.json"
  echo "arm $key rc=$rc wall=$((end - start))s peak_rss_kb=$peak"
  # Another user's job taking the shared GPU mid-start OOMs the arm's FIRST stage. Such an
  # arm has no endpoint and no resume point, so it is restarted from scratch: its work dir
  # is wiped, its results kept aside as <key>.oomN, and the spec goes back to pending (at
  # most OOM_RETRIES times). Anything else (a later stage, a continuation, any other error)
  # is filed under failed/ for a person to read.
  fresh=$("$PY" -I -c 'import json,sys; s=json.load(open(sys.argv[1])); print("" if s.get("resume_stop") else s.get("work_id", s["id"]))' "$RUN/$next" 2>/dev/null)
  tries=$(ls -d "$out".oom* 2>/dev/null | wc -l)
  # Only a CO-TENANT OOM is retried: torch's OOM message names every other process holding
  # the card ("Process <pid> has <n> GiB memory in use"); a self-OOM (the arm alone does not
  # fit, e.g. P=4 on a 3070, 2026-10-19) has no such clause and is filed as failed.
  # ... and only when those other processes hold >= 1 GiB: a 222 MiB co-tenant beside our own
  # 10.2 GiB is a self-OOM that a retry cannot fix (2080 Ti, 2026-10-19, 10 futile retries).
  cotenant=$(grep -h "OutOfMemoryError" "$out/arm_exec.err" "$out"/*.log.txt 2>/dev/null \
             | grep -oE "Process [0-9]+ has [0-9.]+ [GM]iB memory in use" \
             | awk '{v = $4; if ($5 == "MiB") v = v / 1024; s += v} END {print (s >= 1.0) ? 1 : 0}')
  if [ "$rc" -ne 0 ] && [ -n "$fresh" ] && [ ! -f "$out/summary.json" ] \
     && [ "${cotenant:-0}" -gt 0 ] && [ "$tries" -lt "${OOM_RETRIES:-20}" ]; then
    rm -rf "$R/arms/$fresh"; mv "$out" "$out.oom$tries"
    mv "$RUN/$next" "$Q/pending/$next"
    echo "arm $key OOM in its first stage; wiped and requeued (retry $((tries + 1)))"
    sleep 120
    continue
  fi
  if [ "$rc" -eq 0 ]; then mv "$RUN/$next" "$Q/done/$next"
  else mv "$RUN/$next" "$Q/failed/$next"; fi
done
echo "worker end $(date -u +%FT%TZ)"
