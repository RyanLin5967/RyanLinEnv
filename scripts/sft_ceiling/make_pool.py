"""Build the lane's SFT-from-I label pool: all_train - train400 - native VAL/TEST pHash near-duplicates.

  python3 -I scripts/sft_ceiling/make_pool.py --out docs/aircraft-sft-ceiling-v1/pool_all_train_clean.json

Inputs are pinned by sha256 and the script refuses on any mismatch or an unexpected result.
Near-duplicates use the same rule as the sweep's design.py (phash_audit.json
external_near_duplicates with reference_split in test/validation). Output ids keep all_train order.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ALL_TRAIN = Path("/Users/idide/wt/RyanLinEnv-exploratory-sweep-v2.noindex/docs/"
                 "aircraft-exploratory-sweep-v2/manifests/all_train.jsonl")
ALL_TRAIN_SHA = "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e"
TRAIN400 = Path("/Users/idide/RyanLinEnv/data/fgvc-aircraft-2013b/candidates/fourshot-seed791/train400.jsonl")
TRAIN400_SHA = "b4e5b821b851189d5dda0963f99f98e6b790a51c5b99b07f6af00843bf2a41f0"
DEV800 = Path("/Users/idide/wt/RyanLinEnv-exploratory-sweep-v2.noindex/docs/"
              "aircraft-exploratory-sweep-v2/manifests/dev800.jsonl")
DEV800_SHA = "b01175a1ea66985384c7d6c615305c9956c12978a1af3528f0e1c0a484be25b5"
PHASH = Path("/Users/idide/RyanLinEnv/runs/aircraft-freshpool-g0-v1/manifests/phash_audit.json")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for p, d in ((ALL_TRAIN, ALL_TRAIN_SHA), (TRAIN400, TRAIN400_SHA), (DEV800, DEV800_SHA)):
        if sha(p) != d:
            sys.exit(f"REFUSE: {p} sha differs")
    audit = json.loads(PHASH.read_text())
    near = {x["pool_id"] for x in audit["external_near_duplicates"]
            if x["reference_split"] in ("test", "validation")}
    allrows = rows(ALL_TRAIN)
    allt = [r["id"] for r in allrows]
    t400 = {r["id"] for r in rows(TRAIN400)}
    dev = {r["id"] for r in rows(DEV800)}
    near_in = near & set(allt)
    if len(near_in) != 477:
        sys.exit(f"REFUSE: expected 477 VAL/TEST near-duplicates in all_train, found {len(near_in)}")
    if dev & set(allt):
        sys.exit("REFUSE: all_train overlaps dev800")
    pool = [i for i in allt if i not in t400 and i not in near]
    if not pool:
        sys.exit("REFUSE: empty pool")
    label = {r["id"]: r["label"] for r in allrows}
    out = {"rule": "all_train - train400 - phash VAL/TEST near-duplicates",
           "inputs_sha256": {"all_train": ALL_TRAIN_SHA, "train400": TRAIN400_SHA,
                             "phash_audit": sha(PHASH), "dev800": DEV800_SHA},
           "all_train": len(allt), "train400_removed": len(t400 & set(allt)),
           "near_dup_removed": len(near_in), "overlap_train400_neardup": len(t400 & near_in),
           "n": len(pool), "classes": len({label[i] for i in pool}), "ids": pool}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print({k: v for k, v in out.items() if k != "ids"})


main()
