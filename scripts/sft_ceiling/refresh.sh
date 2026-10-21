#!/bin/bash
# Fetch the three live seed-42 all_train arms' results (those with a summary), commit raw, run report.
WT=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497
F=$WT/scripts/sft_ceiling/fetch_one.sh
for pair in sftceil-cont-all-lr5e-6:sftceil-cont-all-lr5e-6-10 sftceil-cont-all-lr1e-5:sftceil-cont-all-lr1e-5-10 \
            sftceil-cont-all-lr2e-5:sftceil-cont-all-lr2e-5-10; do
  arm=${pair%%:*}; run=${pair##*:}
  bash "$F" "$run" "$arm" >/dev/null 2>&1 && echo "fetched $arm" || echo "no summary yet: $arm"
done
git -C "$WT" add docs/aircraft-sft-ceiling-v1/raw
git -C "$WT" diff --cached --quiet || git -C "$WT" commit -q -m "SFT-ceiling raw: fetched dev800 stops"
python3 -I "$WT/scripts/sft_ceiling/report.py"
