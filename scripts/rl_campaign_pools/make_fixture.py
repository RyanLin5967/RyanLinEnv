"""Write synthetic score_worker unit results for the eligible rows (fixture for build_pools.py).

  python3 -I make_fixture.py --payload-data DIR --audit DIR --exclusions FILE --out RESULTS_DIR

Values are random (p_gold ~ U(0,1), ~25% greedy wrong); image_sha256 is the real TRAIN audit
sha256 so the clean build passes. Every unit says "host": "FIXTURE"; never mix with real results.
"""

import argparse
import json
import random
from pathlib import Path

PINS = {
    "adapter_sha256": "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7",
    "all_train_sha256": "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e",
    "trie_sha256": "656dc02fbb5b121f4cd9a9d7ea10364532639b35b647f0dc9010ad5d80a37eb0",
    "trainer_sha256": "b6e18a8b7c502ed71f3178ff3891ff0276261aa2cbdc16f9ce0ca869d116a5c0",
}

p = argparse.ArgumentParser()
p.add_argument("--payload-data", type=Path, required=True)
p.add_argument("--audit", type=Path, required=True)
p.add_argument("--exclusions", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
rng = random.Random(a.seed)
ex = json.loads(a.exclusions.read_text())
drop = set(ex["train400"]) | set(ex["val_test_phash_near_duplicates"])
rows = [
    json.loads(line)
    for line in (a.payload_data / "manifests/all_train.jsonl").read_text().splitlines()
    if line.strip()
]
rows = [r for r in rows if r["id"] not in drop]
audit = {
    e["image_id"]: e["encoded_sha256"]
    for e in json.loads((a.audit / "train-images.json").read_text())
}
a.out.mkdir(parents=True, exist_ok=True)
for k in range(0, len(rows), 32):
    out = []
    for r in rows[k : k + 32]:
        ok = rng.random() > 0.25
        pg = rng.random()
        out.append(
            {
                "id": r["id"],
                "label": r["label"],
                "image_sha256": audit[r["id"]],
                "p_gold": pg,
                "p_canonical": pg * 0.97,
                "trie_paths": 2,
                "p_seconds": 1.0,
                "greedy_seconds": 1.0,
                "greedy_answer": r["label"] if ok else None,
                "greedy_correct": ok,
                "terminated": ok,
                "greedy_response": "",
            }
        )
    doc = dict(
        PINS,
        unit=f"score-{k // 32:03d}",
        ids=[x["id"] for x in out],
        rows=out,
        host="FIXTURE",
        validation={"max_abs_diff": 0.0, "paths": 2},
        total_seconds=64.0,
        greedy_seconds=32.0,
    )
    (a.out / f"score-{k // 32:03d}.json").write_text(json.dumps(doc))
print("fixture units", (len(rows) + 31) // 32)
