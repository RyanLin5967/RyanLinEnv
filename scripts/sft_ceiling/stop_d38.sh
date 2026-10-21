#!/bin/bash
# One-off (D38, 2026-10-19): stop the arms root reallocated to RL capacity. Checkpoints, kept adapters
# and scored assays stay on the hosts; only processes stop.
for run in sftceil-I-seed1-11 sftceil-I-seed2-10 sftceil-I-seed3-10 sftceil-I-seed4-10 sftceil-I-seed5-11 \
           sftceil-I-seed6-3 sftceil-I-seed42-10 sftceil-cont-all-lr1e-5-s314-1 sftceil-cont-all-lr1e-5-s2718-1 \
           sftceil-cont-all-lr5e-6-s314-2 sftceil-cont-all-lr3e-6-10; do
  echo "== $run: $(timeout 300 ~/.claude/bin/uwrun stop "$run" 2>&1 | tr '\n' ' ')"
done
