"""Write synthetic score_worker shard results for all_train (fixture for build_pools.py tests).

  python3 -I make_fixture.py --payload-data DIR --audit DIR --out RESULTS_DIR [--seed 1]

Values are random (p_gold uniform, ~25% greedy wrong); image_sha256 is the real TRAIN audit
sha256 so the clean build passes. Never mix these with real results: every shard says
"host": "FIXTURE".
"""
import argparse
import json
import random
from pathlib import Path

I = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
A = "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e"

p = argparse.ArgumentParser()
p.add_argument("--payload-data", type=Path, required=True)
p.add_argument("--audit", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
rng = random.Random(a.seed)
rows = [json.loads(l) for l in (a.payload_data / "manifests/all_train.jsonl").read_text().splitlines() if l.strip()]
audit = {e["image_id"]: e["encoded_sha256"] for e in json.loads((a.audit / "train-images.json").read_text())}
a.out.mkdir(parents=True, exist_ok=True)
for k in range(0, len(rows), 32):
    out = []
    for r in rows[k:k + 32]:
        ok = rng.random() > 0.25
        out.append({"id": r["id"], "label": r["label"], "image_sha256": audit[r["id"]],
                    "p_gold": rng.random(), "greedy_answer": r["label"] if ok else None,
                    "greedy_correct": ok, "terminated": ok, "greedy_response": ""})
    doc = {"shard": k // 32, "ids": [x["id"] for x in out], "rows": out, "host": "FIXTURE",
           "adapter_sha256": I, "all_train_sha256": A}
    (a.out / f"shard-{k // 32:03d}.json").write_text(json.dumps(doc))
print("fixture shards", (len(rows) + 31) // 32)
