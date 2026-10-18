#!/bin/bash
# FP32 per-update cost probe. usage (from the synced tree): bash fp32cost.sh DEVICE ASSETS PY CASE...
#   CASE = sft:STEPS:TIMEOUT | sgrpo:G:TIMEOUT    (all TRAIN rows; never VAL/TEST)
set -u
cd "$(pwd -P)"
DEV=$1; A=$2; PY=$3; shift 3
H=$(hostname -s)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$PWD/hf_home PYTHONPATH=$PWD/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== host $H $(date -Is) device $DEV assets $A py $PY VISION_RL_TORCH_THREADS=${VISION_RL_TORCH_THREADS:-unset} OMP=${OMP_NUM_THREADS:-unset}"
uptime
[ "$DEV" = cuda ] && nvidia-smi && nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
$PY -c "import torch,transformers,trl,peft;print('versions torch',torch.__version__,'cuda',torch.version.cuda,transformers.__version__,trl.__version__,peft.__version__,'threads',torch.get_num_threads())"
[ "$(ls $A/train/images/train | wc -l)" -ge 1 ] || { echo "ASSETS MISSING"; exit 9; }
# S scan (seed 42, file order) and its first 64 rows, image paths absolute; TRAIN split only
$PY - "$A" <<'PYEOF'
import json, sys
A = sys.argv[1]
rows = [json.loads(l) for l in open(f"{A}/train/s_scan_seed42.jsonl")]
assert rows and all(r["split"] == "train" and r["image"].startswith("images/train/") for r in rows)
for r in rows: r["image"] = f"{A}/train/" + r["image"]
open("scan.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
open("train64.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows[:64]))
print("manifests scan", len(rows), "train64", 64)
PYEOF
cat > wrap.py <<'PYEOF'
import sys
from vision_rl.vlm_local_training import main
rc = 1
try:
    rc = main(sys.argv[1:])
finally:
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        print("PEAK_CUDA max_allocated_MiB=%.1f max_reserved_MiB=%.1f" % (
            torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20), file=sys.stderr)
sys.exit(rc)
PYEOF
PROFILE="--response-format answer_only --dtype float32 --min-pixels 50176 --max-pixels 200704 --max-prompt-tokens 2048 --max-completion-length 256 --lora-rank 8 --temperature 1.0"
COMMON="--data-spec $A/spec/data_spec.json --model-directory $A/model/Qwen2-VL-2B-Instruct-895c3a49 --model Qwen/Qwen2-VL-2B-Instruct --device $DEV --train-only --init-adapter $A/adapters/warmstart-v1/final_adapter --learning-rate 1e-5 --seed 42 --save-steps 100 $PROFILE"
mkdir -p out
for c in "$@"; do
  IFS=: read kind n to <<<"$c"
  case $kind in
    sft) out=out/$H-$DEV-sft-s$n
      CMD="$PY wrap.py --mode sft --train train64.jsonl --max-steps $n --num-generations 4 $COMMON --output $out";;
    sgrpo) out=out/$H-$DEV-sgrpo-g$n
      CMD="$PY wrap.py --mode grpo --beta 0 --loss-type dr_grpo --scale-rewards none --num-generations $n --screen-threshold 0.5 --target-accepted 3 --max-steps 3 --screen-journal $out/screen.jsonl --train scan.jsonl $COMMON --output $out";;
  esac
  rm -rf $out; L=$out.; echo "== RUN $c $(date -Is)"; echo "CMD $CMD"
  SMI=""
  if [ "$DEV" = cuda ]; then nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 250 > ${L}nvsmi.txt & SMI=$!; fi
  t0=$(date +%s.%N); timeout $to /usr/bin/time -f "TIME user_s=%U sys_s=%S wall_s=%e maxrss_KB=%M" $CMD > ${L}stdout.txt 2> ${L}stderr.txt; rc=$?; t1=$(date +%s.%N)
  [ -n "$SMI" ] && { kill $SMI 2>/dev/null; wait $SMI 2>/dev/null; }
  peak=""; [ -n "$SMI" ] && peak=$(cut -d, -f1 ${L}nvsmi.txt | sort -n | tail -1)
  echo "== END $c rc=$rc wall_s=$(echo "$t1-$t0"|bc) nvsmi_peak_MiB_total_incl_other_users=$peak"
  grep -E "^TIME|^PEAK_CUDA" ${L}stderr.txt
  grep -v -E "UserWarning|warnings.warn|^\s*$" ${L}stderr.txt | grep -i -E "error|Error|OOM|out of memory|Traceback|local VLM" | tail -6
  $PY - "$out" <<'PYEOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
p = out / "optimizer_audit.jsonl"
if p.is_file():
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    a = [x["active_seconds"] for x in rows]
    d = [a[0]] + [a[i] - a[i - 1] for i in range(1, len(a))]
    print("AUDIT", p, "steps", len(d), "per_update_active_s", [round(x, 2) for x in d], "cumulative", round(a[-1], 2), "grad_norms", [round(x.get("latest_gradient_norm", 0), 4) for x in rows])
j = out / "screen.jsonl"
if j.is_file():
    e = [json.loads(l) for l in j.read_text().splitlines()]
    print("SCREEN", j, "scanned", len(e), "accepted", sum(x["accepted"] for x in e), "scorer_s_total", round(sum(x["scorer_seconds"] for x in e), 2), "scorer_s_per_candidate", round(sum(x["scorer_seconds"] for x in e) / max(1, len(e)), 3))
    for s in sorted({x["global_step"] for x in e}):
        g = [x for x in e if x["global_step"] == s]
        print("SCREEN step", s, "scanned", len(g), "accepted", sum(x["accepted"] for x in g), "scorer_s", round(sum(x["scorer_seconds"] for x in g), 2), "p", [round(x["p"], 3) for x in g][:40])
r = out / "local_training.json"
if r.is_file():
    v = json.loads(r.read_text())
    print("RECORD", r, "status", v.get("status"), "optimizer_audit", json.dumps({k: v.get("optimizer_audit", {}).get(k) for k in ("optimizer_steps_observed", "active_seconds", "active_time")})[:1500])
    if v.get("status") == "failed": print("RECORD error", v.get("error_type"), v.get("error"))
PYEOF
  rm -rf $out/checkpoint-*/optimizer.pt
done
echo "== ALL DONE $(date -Is)"
