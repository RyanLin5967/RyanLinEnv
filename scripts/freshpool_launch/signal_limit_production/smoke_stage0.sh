#!/bin/bash
# UW smoke of the signal-limit Stage 0 entry: the real stage0_gate.py through the real
# phase_worker.py --remote-cpu on a provisioned host, at G8 only, E0 over 2 steps
# (2 uninterrupted vs 1+1 resumed) and a 1-accepted-update cost probe. NOT a study stage and
# NOT a look: TRAIN rows only (a train400 subset), no H split, no VAL, no TEST; the study's
# own Stage 0 (G8 and G16, 4 steps, >= 8 probe updates) is refused unless run at its defaults.
#
#   ~/.claude/bin/uwrun run --host H --threads 16 --sync DIR --timeout 5400 -- \
#       bash scripts/freshpool_launch/signal_limit_production/smoke_stage0.sh
# DIR = git archive of the commit under test + inputs/ (measure_units.sh's layout). Writes
# out/stage/fragment.json (the entry's fragment), out/guards/ and out/smoke.json.
set -uo pipefail
cd "$(pwd -P)"
PY=/var/tmp/r226lin-slenv-v1/bin/python
MODEL=/var/tmp/r226lin-slmodel-v1/Qwen2-VL-2B-Instruct-895c3a49
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
export ACCELERATE_TORCH_DEVICE=cpu TOKENIZERS_PARALLELISM=false
export VISION_RL_TORCH_THREADS="${UWRUN_THREADS:?}"
mkdir -p out/stage out/guards
TRAINER_SHA=$(sha256sum src/vision_rl/vlm_local_training.py | cut -d' ' -f1)
ENTRY_SHA=$(sha256sum scripts/freshpool_launch/stage0_gate.py | cut -d' ' -f1)
started=$(date +%s)
"$PY" -B scripts/freshpool_launch/phase_worker.py --remote-cpu --guard-dir out/guards \
    --stage smoke-stage0 --kind train --entry stage0_gate:main --entry-sha256 "$ENTRY_SHA" -- \
    --fragment out/stage/fragment.json --stage-dir out/stage --trainer-sha256 "$TRAINER_SHA" \
    --groups 8 --e0-steps 2 --probe-accepted 1 -- \
    --mode grpo --train inputs/train.jsonl --data-spec inputs/data_spec.json \
    --model-directory "$MODEL" --model Qwen/Qwen2-VL-2B-Instruct --device cpu --train-only \
    --response-format answer_only --dtype float32 --min-pixels 50176 --max-pixels 200704 \
    --max-prompt-tokens 2048 --max-completion-length 256 --lora-rank 8 --temperature 1.0 \
    --init-adapter inputs/w/final_adapter --learning-rate 0.0001 --seed 42 \
    --beta 0 --loss-type dr_grpo --scale-rewards none > out/stage.log 2>&1
rc=$?
echo "{\"phase_worker_exit_code\": $rc, \"wall_seconds\": $(( $(date +%s) - started )), \"host\": \"$(hostname)\", \"cpu\": \"$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\", \"threads\": \"$UWRUN_THREADS\"}" > out/smoke.json
cat out/smoke.json
[ -f out/stage/fragment.json ] && "$PY" -c 'import json; f=json.load(open("out/stage/fragment.json")); print(json.dumps({k: f.get(k) for k in ("e0_groups","e0_steps","probe_accepted","checks","c_accepted_steps","c_seconds_per_accepted_update","sampler_teacher_max_abs_logprob_difference")}, indent=1))'
exit $rc
