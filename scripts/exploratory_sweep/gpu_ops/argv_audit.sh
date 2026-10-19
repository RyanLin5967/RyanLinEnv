#!/bin/bash
export UWRUN_OWNER=ryanlinenv  # uwrun v5.11 owner maps: identify this lane
# Mac-side: print the ACTUAL trainer argv flags that matter (--train, --dtype, --device,
# --learning-rate, --dynamic-sampling-max-draws, --loss-type, --temperature, --seed,
# --init-adapter) of every running vlm_local_training process of ours on each lane host,
# read from the host's process table, plus every --flag passed more than once.
for h in ecetesla2 ecetesla4 eceubuntu2 iccad0 iccad13 iccad8 iccad10 "$@"; do
  echo "== $h"
  timeout 60 ~/.claude/bin/uwrun ssh "$h" 'for p in $(pgrep -u r226lin -f "[v]ision_rl.vlm_local_training"); do
    tr "\0" "\n" < /proc/$p/cmdline > /tmp/.argv.$p 2>/dev/null || continue
    python3 - /tmp/.argv.$p <<EOF
import sys
a = open(sys.argv[1]).read().split("\n")
keys = ("--mode", "--train", "--dtype", "--device", "--learning-rate", "--dynamic-sampling-max-draws",
        "--loss-type", "--temperature", "--seed", "--beta", "--output")
out = {k: [a[i + 1] for i, x in enumerate(a[:-1]) if x == k] for k in keys}
dups = sorted({x for x in a if x.startswith("--") and a.count(x) > 1})
print({k: (v if len(v) != 1 else v[0]) for k, v in out.items() if v}, "DUPLICATED:", dups)
EOF
    rm -f /tmp/.argv.$p; done' 2>&1
done
