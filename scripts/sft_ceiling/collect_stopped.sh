#!/bin/bash
# Collect a STOPPED arm's small raw files into docs/aircraft-sft-ceiling-v1/raw/<ARM>/ (partial):
# the uwrun results dir (arm.json, logs, any assays/summary), the trainer's optimizer_audit.jsonl,
# local_training.json, and the latest complete checkpoint's trainer_state.json + active_time.json,
# plus STOPPED.json {run, host, latest_checkpoint, last_audit_step, label}. Weights stay on the host.
# Usage: collect_stopped.sh ARM HOST RUN LABEL
set -euo pipefail
ARM=$1; HOST=$2; RUN=$3; LABEL=$4
RAW=/Users/idide/wt/RyanLinEnv-agent-ae7d7eaaefb389497/docs/aircraft-sft-ceiling-v1/raw/$ARM
mkdir -p "$RAW"
timeout 600 ~/.claude/bin/uwrun ssh "$HOST" "for R in /dev/shm/r226lin-sftceil /var/tmp/r226lin-sftceil; do [ -d \$R/arms/$ARM ] && break; done; cd \$R/arms/$ARM; best=0; for c in train/checkpoint-*; do [ -f \$c/active_time.json ] && [ -f \$c/optimizer.pt ] && s=\${c##*-} && [ \$s -gt \$best ] && best=\$s; done; files='train/optimizer_audit.jsonl train/local_training.json'; [ \$best -gt 0 ] && files=\"\$files train/checkpoint-\$best/trainer_state.json train/checkpoint-\$best/active_time.json\"; tar czf - \$files -C \$HOME/runs/$RUN/work results 2>/dev/null | base64" > "$RAW/partial.tgz.b64"
base64 -d -i "$RAW/partial.tgz.b64" | tar xzf - -C "$RAW"
rm "$RAW/partial.tgz.b64"
python3 -I - "$RAW" "$ARM" "$HOST" "$RUN" "$LABEL" <<'PY'
import json, sys
from pathlib import Path
raw, arm, host, run, label = sys.argv[1:]
raw = Path(raw)
ck = sorted(raw.glob("train/checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
audit = [json.loads(l) for l in (raw / "train/optimizer_audit.jsonl").read_text().splitlines() if l.strip()]
summary = raw / "results/summary.json"
eps = json.loads(summary.read_text())["endpoints"] if summary.exists() else []
out = {"arm": arm, "host": host, "run": run, "label": label,
       "latest_checkpoint": int(ck[-1].name.split("-")[1]) if ck else 0,
       "committed_active_seconds": (json.loads((ck[-1] / "active_time.json").read_text())
                                    .get("committed_active_seconds") if ck else None),
       "last_audit_step": audit[-1]["step"] if audit else 0,
       "scored_endpoints": [{"label": e["label"], "steps": e["steps"],
                             "accuracy": e["metrics"]["accuracy"]} for e in eps]}
(raw / "STOPPED.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
PY
