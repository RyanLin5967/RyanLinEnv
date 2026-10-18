#!/bin/bash
# Measure the signal-limit study's unit costs on one provisioned UW host (CPU FP32).
#
# TRAIN rows only (a train400 subset): no H split, no VAL, no TEST image is read, so nothing
# here is a study look. Run through uwrun with --sync pointing at a directory that holds the
# repository (git archive) plus inputs/:
#   inputs/w/                  W's trainer run (final_adapter + local_training.json)
#   inputs/train.jsonl         64 TRAIN rows with images/ beside it (relative paths), and
#   inputs/train8.jsonl, inputs/train24.jsonl   its first 8 and 24 rows
#   inputs/data_spec.json
#   ~/.claude/bin/uwrun run --host H --threads 16 --sync DIR --timeout 7200 -- \
#       bash scripts/freshpool_launch/signal_limit_production/measure_units.sh
# Each measurement is one trainer (or entry) process; out/units.jsonl gets one line each
# with the instrument named: per-update seconds from the trainer's optimizer_audit.jsonl
# (committed active seconds per step), inference seconds from a monotonic clock around the
# call, in-process.
set -uo pipefail
cd "$(pwd -P)"
PY=/var/tmp/r226lin-slenv-v1/bin/python
MODEL=/var/tmp/r226lin-slmodel-v1/Qwen2-VL-2B-Instruct-895c3a49
export PYTHONPATH="$PWD/src:$PWD/scripts/freshpool_launch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0 ACCELERATE_TORCH_DEVICE=cpu TOKENIZERS_PARALLELISM=false
export VISION_RL_TORCH_THREADS="${UWRUN_THREADS:?}"
mkdir -p out
PROFILE=(--response-format answer_only --dtype float32 --min-pixels 50176 --max-pixels 200704
         --max-prompt-tokens 2048 --max-completion-length 256 --lora-rank 8 --temperature 1.0)
COMMON=(--data-spec inputs/data_spec.json --model-directory "$MODEL"
        --model Qwen/Qwen2-VL-2B-Instruct --device cpu --train-only "${PROFILE[@]}"
        --init-adapter inputs/w/final_adapter --learning-rate 1e-4 --seed 42 --save-steps 100)
uname -a > out/host.txt; grep -m1 "model name" /proc/cpuinfo >> out/host.txt; nproc >> out/host.txt

train() {  # name, extra trainer args...
  local name=$1; shift
  local started=$(date +%s.%N)
  "$PY" -s -P -m vision_rl.vlm_local_training --output "out/$name" "${COMMON[@]}" "$@" \
      > "out/$name.log" 2>&1
  local rc=$?
  "$PY" -s -P - "$name" "$rc" "$started" >> out/units.jsonl <<'PY'
import json, statistics, sys, time
from pathlib import Path
name, rc, started = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
audit = Path("out") / name / "optimizer_audit.jsonl"
seconds = {}
if audit.is_file():
    for line in audit.read_text().splitlines():
        row = json.loads(line)
        seconds[row["step"]] = row.get("completed_state_active_seconds")
steps = [seconds[s] for s in sorted(seconds)]
per = [b - a for a, b in zip([0.0, *steps], steps)]
screen = Path("out") / name / "screen.jsonl"
scanned = len(screen.read_text().splitlines()) if screen.is_file() else None
print(json.dumps({"unit": name, "rc": rc, "instrument": "optimizer_audit.jsonl "
                  "completed_state_active_seconds per step", "per_update_seconds": per,
                  "median_after_first": statistics.median(per[1:]) if len(per) > 1 else None,
                  "max_after_first": max(per[1:]) if len(per) > 1 else None,
                  "screen_rows": scanned, "wall_seconds": time.time() - started}))
PY
}

# UNITS (environment, space-separated) picks a subset; default: every unit.
UNITS=${UNITS:-"sft egsft hn_sft hls_sft cce_sft grpo_g8 grpo_g16 inference"}
want() { case " $UNITS " in *" $1 "*) return 0;; *) return 1;; esac; }

want sft && train sft --mode sft --order sequential --train inputs/train.jsonl --max-steps 6
want egsft && train egsft --mode egsft --order sequential --train inputs/train.jsonl --max-steps 6 \
    --egsft-scale none --egsft-band-group 8
want hn_sft && train hn_sft --mode hn_sft --order sequential --train inputs/train.jsonl --max-steps 6 \
    --confusion-map data/fgvc-aircraft-2013b/audit/p0-confusion-map.json
want hls_sft && train hls_sft --mode hls_sft --order sequential --train inputs/train.jsonl --max-steps 6 \
    --taxonomy data/fgvc-aircraft-2013b/audit/taxonomy-train-only.json
want cce_sft && train cce_sft --mode cce_sft --order sequential --train inputs/train.jsonl --max-steps 6 \
    --label-trie data/fgvc-aircraft-2013b/audit/label-trie-qwen2vl2b.json
for g in 8 16; do
  want "grpo_g$g" || continue
  train "grpo_g$g" --mode grpo --train inputs/train.jsonl --num-generations "$g" --beta 0 \
      --loss-type dr_grpo --scale-rewards none --screen-threshold 0.5 --max-steps 3 \
      --target-accepted 3 --screen-journal "out/grpo_g$g/screen.jsonl"
done

# Inference units: p scoring (P0 and the screen), greedy and one G4 group per image (assays).
want inference && "$PY" -s -P - "$MODEL" >> out/units.jsonl 2> out/inference.log <<'PY'
import json, sys, time
from dataclasses import replace
from pathlib import Path
import h_assay, p0_screen
from vision_rl import vlm_local_training as vlt
model = sys.argv[1]
rows = h_assay.read_rows(Path("inputs/train.jsonl"))[:8]
profile = replace(h_assay.PROFILE, num_generations=4)
started = time.monotonic()
scored = p0_screen.score_candidates(rows, adapter=Path("inputs/w/final_adapter"),
                                    data_spec="inputs/data_spec.json", model_directory=model,
                                    device="cpu", profile=profile)
print(json.dumps({"unit": "scorer", "instrument": "p0_screen.score_candidates scorer_seconds "
                  "(monotonic around the no-grad forward)", "n": len(scored),
                  "per_candidate_seconds": [s for _, s, _ in scored],
                  "call_wall_seconds": time.monotonic() - started}))
for name, flag in (("greedy", "--predict-train"), ("g4_group", "--sample-train")):
    walls = {}
    for n in (8, 24):  # per image = (wall(24) - wall(8)) / 16: the model load cancels
        started = time.monotonic()
        code = vlt.main([flag, f"inputs/train{n}.jsonl", "--output", f"out/{name}{n}",
                         "--data-spec", "inputs/data_spec.json", "--model-directory", model,
                         "--device", "cpu", "--init-adapter", "inputs/w/final_adapter",
                         *sum(([f"--{k.replace('_', '-')}", str(v)] for k, v in
                               vars(profile).items()), [])])
        walls[n] = (code, time.monotonic() - started)
    print(json.dumps({"unit": name, "rc": [walls[8][0], walls[24][0]],
                      "instrument": "monotonic wall around one vlm_local_training call at 8 "
                                    "and at 24 TRAIN images; per image = difference / 16",
                      "wall_seconds": {str(n): w for n, (_, w) in walls.items()},
                      "per_image_seconds": (walls[24][1] - walls[8][1]) / 16}), flush=True)
PY
echo "units done"; cat out/units.jsonl
