#!/bin/bash
# Per arm: host, uwrun status, last committed optimizer step, active seconds, sec/update over the
# last 20 updates, endpoints assayed so far (with accuracy). Read-only.
# arm:host:run-suffix (lr2e-5 moved off ecetesla1 at step 1, 2026-10-18 21:5xZ: ~120 s/update there)
ARMS="sftceil-cont-all-lr3e-6:iccad11:10
sftceil-cont-all-lr5e-6:iccad7:10
sftceil-cont-all-lr1e-5:iccad11:10
sftceil-cont-all-lr2e-5:iccad12:10
sftceil-I-seed1:ecetesla0:10
sftceil-I-seed2:iccad10:10
sftceil-I-seed3:iccad13:10
sftceil-I-seed4:iccad12:10
sftceil-I-seed5:iccad6:10
sftceil-I-seed42:iccad5:10
sftceil-cont-all-lr1e-5-s314:iccad0:1
sftceil-I-seed6:iccad7:3"
for triple in $ARMS; do
  arm=${triple%%:*}; rest=${triple#*:}; host=${rest%%:*}; n=${rest##*:}
  st=$(timeout 90 ~/.claude/bin/uwrun status "$arm-$n" 2>&1 | tail -1 | awk '{print $3, $4}')
  info=$(timeout 90 ~/.claude/bin/uwrun ssh "$host" "for R in /dev/shm/r226lin-sftceil /var/tmp/r226lin-sftceil; do [ -d \$R/arms/$arm ] && break; done; A=\$R/arms/$arm/train/optimizer_audit.jsonl; python3 - \$A \$HOME/runs/$arm-$n/work/results/summary.json <<'PY'
import json, sys, os
rows = [json.loads(l) for l in open(sys.argv[1])] if os.path.exists(sys.argv[1]) else []
s = ''
if rows:
    last = rows[-1]; w = rows[-21:]
    rate = (w[-1]['active_seconds'] - w[0]['active_seconds']) / max(1, len(w) - 1)
    s = 'step=%s act=%.0fs %.1fs/upd' % (last.get('step', len(rows)), last['active_seconds'], rate)
eps = ''
if os.path.exists(sys.argv[2]):
    eps = ' '.join('%s:%.4f' % (e['label'], e['metrics']['accuracy']) for e in json.load(open(sys.argv[2]))['endpoints'])
print(s, eps)
PY" 2>&1 | tail -1)
  printf '%-26s %-10s %-18s %s\n' "$arm" "$host" "$st" "$info"
done
