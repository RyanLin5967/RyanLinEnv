#!/bin/bash
# One-off (2026-10-18): restart every live lane arm in place with temperature 1.0 in its spec.
# New run names end in -10. Continues past a failure, reports each.
set -u
P=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/scripts/sft_ceiling/restart_inplace.py
while read -r arm host old; do
  [ -z "$arm" ] && continue
  timeout 900 python3 -I "$P" "$arm" "$host" "$old" 10 || echo "FAILED $arm"
done <<'EOF'
sftceil-cont-all-lr5e-6 iccad7 sftceil-cont-all-lr5e-6-1
sftceil-cont-all-lr1e-5 iccad11 sftceil-cont-all-lr1e-5-2
sftceil-cont-all-lr2e-5 iccad12 sftceil-cont-all-lr2e-5-3
sftceil-cont-all-lr3e-6 iccad11 sftceil-cont-all-lr3e-6-1
sftceil-I-seed1 ecetesla0 sftceil-I-seed1-2
sftceil-I-seed2 iccad10 sftceil-I-seed2-1
sftceil-I-seed3 iccad13 sftceil-I-seed3-1
sftceil-I-seed4 iccad12 sftceil-I-seed4-1
sftceil-I-seed5 iccad6 sftceil-I-seed5-1
sftceil-I-seed42 iccad5 sftceil-I-seed42-1
EOF
