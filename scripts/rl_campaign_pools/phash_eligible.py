"""pHash every eligible pool row against native VAL, sealed TEST and train400 (pools v3 input).

  python -I phash_eligible.py --producer prepare_aircraft_freshpool.py --payload-data DIR \
      --audit DIR --val-dir DIR --test-phashes FILE --phash-audit FILE --train400 FILE \
      --exclusions FILE --out FILE

Why: runs/aircraft-freshpool-g0-v1/manifests/phash_audit.json hashed only its 2,919-image pool,
which leaves out train400 and the 15 "probe16_outside_train400" rows. All 15 are pool-eligible
here, so v2 could not prove them free of VAL/TEST near-duplicates.

Method: the producer's own phash64, near_pairs, planted_copy and HAMMING_MAX (imported from
scripts/prepare_aircraft_freshpool.py, not reimplemented), on vision_rl's load_aircraft_rgb.
TEST is never opened: its pHashes come from the root-authorised test-phashes.json, the same file
the original audit used, pinned by sha256.

Refuses (exit 2) when: an input differs from its pinned sha256; an eligible row's image bytes
differ from the native TRAIN audit; any eligible id is left without a pHash; the reference sets
are not VAL 3,333 / TEST 3,333 / train400 400; for the rows the original audit DID hash, the
recomputed pHash or the near-duplicate pairs differ from it (method parity); the forced fire
(a planted VAL copy and a planted eligible copy must be flagged, distinct-class pairs must not)
fails.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PINS = {
    "producer": "019a57d4a5ff88c40a928addb7238f52c422107c83d0d39847576e7634102e51",
    "test_phashes": "edbf711c788680c92fbaf0377e1bda69f0f1f35224f94aaafc787ebe3ef7a457",
    "phash_audit": "67863f31d00e442a84eaad048a4f5681f3234917de0d3c70ce11c99ee1ca16f7",
    "train400": "b4e5b821b851189d5dda0963f99f98e6b790a51c5b99b07f6af00843bf2a41f0",
    "exclusions": "00a433eab22002764d2d91c02888623b2d5a4b37ee90ba3e1bb0eb19b482380b",
    "validation_jsonl": "27d5cd01bd7e8e007211623ac2186f94ecb308afa4a02745ce39c92fb504b9cb",
    "all_train": "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e",
}
ELIGIBLE_ROWS = 1452
PROBE16_OUTSIDE_TRAIN400 = 15


class Refuse(Exception):
    pass


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def pinned(path: Path, key: str) -> Path:
    got = sha_file(path)
    if got != PINS[key]:
        raise Refuse(f"{path} sha256 {got} is not the pinned {key} {PINS[key]}")
    return path


def load_producer(path: Path):
    spec = importlib.util.spec_from_file_location("freshpool_producer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_rows(pairs, split: str) -> list[dict]:
    return [
        {"pool_id": q, "reference_split": split, "reference_id": r, "hamming": int(h)}
        for q, r, h in pairs
    ]


def run(a) -> dict:
    fp = load_producer(pinned(a.producer, "producer"))
    from vision_rl.aircraft_data import load_aircraft_rgb

    man = a.payload_data / "manifests"
    all_train = read_jsonl(pinned(man / "all_train.jsonl", "all_train"))
    ex = json.loads(pinned(a.exclusions, "exclusions").read_text())
    dropped = set(ex["train400"]) | set(ex["val_test_phash_near_duplicates"])
    eligible = [r for r in all_train if r["id"] not in dropped]
    if len(eligible) != ELIGIBLE_ROWS:
        raise Refuse(f"eligible rows {len(eligible)} != {ELIGIBLE_ROWS}")
    train400 = read_jsonl(pinned(a.train400, "train400"))
    train_sha = {
        e["image_id"]: e["encoded_sha256"]
        for e in json.loads((a.audit / "train-images.json").read_text())
    }

    def train_hash(row_id: str) -> int:
        path = man / "images" / "train" / f"{row_id}.jpg"
        if sha_file(path) != train_sha.get(row_id):
            raise Refuse(f"{row_id}: image bytes are not the native TRAIN audit's")
        with load_aircraft_rgb(path) as image:
            return fp.phash64(image)

    eligible_hashes = {r["id"]: train_hash(r["id"]) for r in eligible}
    missing = sorted({r["id"] for r in eligible} - set(eligible_hashes))
    if missing:
        raise Refuse(f"eligible ids without a pHash: {missing[:10]}")
    t4_hashes = {r["id"]: train_hash(r["id"]) for r in train400}

    val_rows = read_jsonl(pinned(a.val_dir / "validation.jsonl", "validation_jsonl"))
    val_hashes = {}
    for r in val_rows:
        with load_aircraft_rgb(a.val_dir / r["image"]) as image:
            val_hashes[r["id"]] = fp.phash64(image)
    raw_test = json.loads(pinned(a.test_phashes, "test_phashes").read_text())
    test_ids = [e["image_id"] for e in json.loads((a.audit / "test-images.json").read_text())]
    test_hashes = fp.load_test_phashes(raw_test, test_ids)
    counts = {"validation": len(val_hashes), "test": len(test_hashes), "train400": len(t4_hashes)}
    if counts != {"validation": 3333, "test": 3333, "train400": 400}:
        raise Refuse(f"reference counts wrong: {counts}")

    refs = {"validation": val_hashes, "test": test_hashes, "train400": t4_hashes}
    near = []
    for split, hashes in sorted(refs.items()):
        near += pair_rows(fp.near_pairs(eligible_hashes, hashes), split)
    near.sort(key=lambda x: (x["pool_id"], x["reference_split"], x["reference_id"]))

    # Method parity against the original audit, on the rows it did hash.
    old = json.loads(pinned(a.phash_audit, "phash_audit").read_text())
    old_hashes = old["pool_phashes"]
    audited = sorted(set(eligible_hashes) & set(old_hashes))
    unaudited = sorted(set(eligible_hashes) - set(old_hashes))
    if len(unaudited) != PROBE16_OUTSIDE_TRAIN400:
        raise Refuse(f"expected {PROBE16_OUTSIDE_TRAIN400} unaudited eligible rows, {unaudited}")
    hash_mismatch = [i for i in audited if f"{eligible_hashes[i]:016x}" != old_hashes[i]]
    audited_set = set(audited)
    key = lambda x: (x["pool_id"], x["reference_split"], x["reference_id"], x["hamming"])  # noqa: E731
    old_pairs = sorted(
        key(x) for x in old["external_near_duplicates"] if x["pool_id"] in audited_set
    )
    new_pairs = sorted(key(x) for x in near if x["pool_id"] in audited_set)
    parity = {
        "audited_rows": len(audited),
        "hash_mismatches": len(hash_mismatch),
        "old_pairs_on_audited_rows": len(old_pairs),
        "new_pairs_on_audited_rows": len(new_pairs),
        "pairs_identical": old_pairs == new_pairs,
    }
    if hash_mismatch or old_pairs != new_pairs:
        raise Refuse(f"method parity with the original audit failed: {parity} {hash_mismatch[:5]}")

    # Forced fire: a planted VAL copy must hit its VAL source; a planted eligible copy must hit
    # its own hash; the producer's distinct-class check runs over the eligible rows.
    val_src = sorted(val_hashes)[len(val_hashes) // 2]
    with load_aircraft_rgb(a.val_dir / f"images/validation/{val_src}.jpg") as image:
        planted_val = fp.phash64(fp.planted_copy(image))
    val_fire = fp.near_pairs({"planted": planted_val}, {val_src: val_hashes[val_src]})
    forced = fp.forced_fire(
        eligible,
        eligible_hashes,
        lambda i: load_aircraft_rgb(man / "images" / "train" / f"{i}.jpg"),
    )
    fire = {
        "planted_val_source": val_src,
        "planted_val_hamming": (planted_val ^ val_hashes[val_src]).bit_count(),
        "planted_val_flagged": bool(val_fire),
        "eligible": forced,
    }
    if not (val_fire and forced["passed"]):
        raise Refuse(f"forced fire failed: {json.dumps(fire)[:1500]}")

    flagged = sorted({x["pool_id"] for x in near})
    return {
        "schema": "rl-campaign-I eligible pHash audit v3",
        "method": "64-bit orthonormal DCT-II pHash; 32px LANCZOS grayscale; 8x8 incl. DC; median",
        "image_transform": "vision_rl.aircraft_data.load_aircraft_rgb (bottom20 banner removed)",
        "hamming_max": fp.HAMMING_MAX,
        "inputs_sha256": dict(PINS),
        "reference_counts": counts,
        "eligible_rows": len(eligible_hashes),
        "eligible_phashes": {i: f"{h:016x}" for i, h in sorted(eligible_hashes.items())},
        "unaudited_in_original": {
            i: {
                "phash": f"{eligible_hashes[i]:016x}",
                "pairs": [x for x in near if x["pool_id"] == i],
            }
            for i in unaudited
        },
        "near_duplicates": near,
        "flagged_ids": flagged,
        "flagged_by_split": {
            s: sorted({x["pool_id"] for x in near if x["reference_split"] == s}) for s in refs
        },
        "parity_with_original_audit": parity,
        "forced_fire": fire,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in (
        "producer",
        "payload-data",
        "audit",
        "val-dir",
        "test-phashes",
        "phash-audit",
        "train400",
        "exclusions",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args(argv)
    try:
        doc = run(a)
    except Refuse as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    raw = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    a.out.write_bytes(raw)
    summary = {
        k: doc[k] for k in ("reference_counts", "parity_with_original_audit", "flagged_by_split")
    }
    summary["sha256"] = hashlib.sha256(raw).hexdigest()
    summary["unaudited_pairs"] = {k: v["pairs"] for k, v in doc["unaudited_in_original"].items()}
    summary["forced_fire"] = {
        "val": doc["forced_fire"]["planted_val_hamming"],
        "eligible_passed": doc["forced_fire"]["eligible"]["passed"],
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
