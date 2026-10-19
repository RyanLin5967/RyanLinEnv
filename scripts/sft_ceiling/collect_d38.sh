#!/bin/bash
# One-off (D38): collect partial raw for every arm stopped by root for RL capacity.
C=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/scripts/sft_ceiling/collect_stopped.sh
L="stopped by root for RL capacity (D38)"
while read -r arm host run; do
  [ -z "$arm" ] && continue
  bash "$C" "$arm" "$host" "$run" "$L" || echo "FAILED $arm"
done <<'EOF'
sftceil-I-seed1 iccad8 sftceil-I-seed1-11
sftceil-I-seed2 iccad10 sftceil-I-seed2-10
sftceil-I-seed3 iccad13 sftceil-I-seed3-10
sftceil-I-seed4 iccad12 sftceil-I-seed4-10
sftceil-I-seed5 iccad7 sftceil-I-seed5-11
sftceil-I-seed6 iccad7 sftceil-I-seed6-3
sftceil-I-seed42 iccad5 sftceil-I-seed42-10
sftceil-cont-all-lr1e-5-s314 iccad0 sftceil-cont-all-lr1e-5-s314-1
sftceil-cont-all-lr1e-5-s2718 iccad8 sftceil-cont-all-lr1e-5-s2718-1
sftceil-cont-all-lr5e-6-s314 iccad10 sftceil-cont-all-lr5e-6-s314-2
sftceil-cont-all-lr3e-6 iccad11 sftceil-cont-all-lr3e-6-10
EOF
