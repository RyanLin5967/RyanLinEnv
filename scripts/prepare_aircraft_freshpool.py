"""Emit the fresh-pool holdout and per-seed scan manifests, after a pHash near-duplicate audit.

Study: aircraft-freshpool-screened-grpo-v1 (runs/rl-next-study-design-v1/study.json, section
data_manifests_to_create). Reads JPEGs and metadata only: no model, no MPS, no network.

Pool = native TRAIN - train400 - probe16 images outside train400 - pHash drops.
H-select = per class, SHA256 ranks 1 and 3 of compact JSON
[protocol_id, "train-holdout-screen", 791, label, id]; H-gate = ranks 2 and 4.
S = pool - H - S members of within-pool near-duplicate pairs that straddle H and S.
Scan order per seed = random.Random(seed).shuffle(sorted(ids)).

Blind spots, stated rather than hidden:
- TEST stays sealed, so this script never opens sealed/. Pool-vs-TEST coverage is an exact
  encoded-byte SHA256 match against audit/test-images.json plus, only when root supplies one,
  a precomputed TEST pHash file. Without that file the audit records TEST pHash as NOT_CHECKED
  and g0_audit_complete is false.
- Hamming <= 6 is pre-registered and unverified as optimal. A shared background can make two
  different aircraft collide, and a heavy crop can make a true duplicate miss.
- Within-pool pairs that sit both in S, or both in H, are reported but not dropped (the study
  rule names only H/S straddles).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HAMMING_MAX = 6
HOLDOUT_SALT = "train-holdout-screen"
HOLDOUT_SEED = 791
SCAN_SEEDS = (42, 314, 2718)
FORCED_FIRE_DISTINCT_PAIRS = 50
# The real-data contract from runs/aircraft-train-screen-review-v1/analysis.json.
REAL_CONTRACT = {
    "classes": 100,
    "train400": 400,
    "probe16_outside_train400": 15,
    "pool_before_audit": 2919,
}


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical(row) + b"\n" for row in rows)


def phash64(image: Image.Image) -> int:
    """64-bit DCT pHash: 32px LANCZOS grayscale, orthonormal DCT-II, top-left 8x8 incl. DC.

    Bit i is set when coefficient i (row-major) exceeds the median of all 64 coefficients.
    Same basis as scripts/audit_dataset.py perceptual_hash, which drops DC to give 63 bits;
    the study pre-registers 64 bits.
    """
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
    )
    frequency = np.arange(8)[:, None]
    position = np.arange(32)[None, :]
    basis = np.cos(np.pi * (2 * position + 1) * frequency / 64) * np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    coefficients = (basis @ pixels @ basis.T).ravel()
    median = np.median(coefficients)
    return sum(int(value > median) << i for i, value in enumerate(coefficients))


def near_pairs(queries: dict[str, int], references: dict[str, int], within: bool = False):
    """Return (query_id, reference_id, hamming) for every pair at Hamming <= HAMMING_MAX.

    With within=True, queries and references are the same set and each unordered pair of
    distinct IDs is reported once.
    """
    if not queries or not references:
        return []
    query_ids, reference_ids = sorted(queries), sorted(references)
    query = np.array([queries[i] for i in query_ids], dtype=np.uint64)
    reference = np.array([references[i] for i in reference_ids], dtype=np.uint64)
    pairs = []
    for row, value in enumerate(query):
        distances = np.bitwise_count(np.bitwise_xor(reference, value))
        for column in np.flatnonzero(distances <= HAMMING_MAX):
            first, second = query_ids[row], reference_ids[int(column)]
            if within and first >= second:
                continue
            pairs.append((first, second, int(distances[column])))
    return pairs


def build_pool(train_rows, train400_rows, probe_rows):
    train_ids = [row["id"] for row in train_rows]
    if len(set(train_ids)) != len(train_ids) or any(r["split"] != "train" for r in train_rows):
        raise ValueError("Native TRAIN rows must be unique and split=train")
    by_id = {row["id"]: row for row in train_rows}
    train400 = {row["id"] for row in train400_rows}
    if len(train400) != len(train400_rows) or not train400 <= set(by_id):
        raise ValueError("train400 must be unique native TRAIN IDs")
    for row in [*train400_rows, *probe_rows]:
        if row["id"] not in by_id or by_id[row["id"]]["label"] != row["label"]:
            raise ValueError(f"Label or membership disagrees with native TRAIN for {row['id']}")
    probe_outside = {row["id"] for row in probe_rows} - train400
    pool = [row for row in train_rows if row["id"] not in train400 | probe_outside]
    return pool, {
        "native_train": len(train_rows),
        "train400": len(train400),
        "probe16_outside_train400": len(probe_outside),
        "pool_before_audit": len(pool),
    }


def holdout_key(protocol_id: str, label: str, image_id: str) -> str:
    payload = [protocol_id, HOLDOUT_SALT, HOLDOUT_SEED, label, image_id]
    return sha(json.dumps(payload, separators=(",", ":")).encode())


def split_holdout(pool_rows, protocol_id):
    by_label = defaultdict(list)
    for row in pool_rows:
        by_label[row["label"]].append(row)
    select, gate = [], []
    for label in sorted(by_label):
        ranked = sorted(by_label[label], key=lambda r: holdout_key(protocol_id, label, r["id"]))
        if len(ranked) < 4:
            raise ValueError(f"Class {label!r} has fewer than 4 audited pool images")
        select += [ranked[0], ranked[2]]
        gate += [ranked[1], ranked[3]]
    return select, gate


def scan_order(ids, seed: int) -> list[str]:
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order


def make_manifests(
    train_rows,
    train400_rows,
    probe_rows,
    protocol_id: str,
    pool_hashes: dict[str, int],
    reference_hashes: dict[str, dict[str, int]],
    exact_matches: dict[str, list[str]] | None = None,
    contract: dict | None = None,
):
    """Build every manifest from metadata and precomputed hashes; no image IO here.

    reference_hashes maps a reference split name (validation, train400, test) to id->pHash.
    exact_matches maps a reference split name to pool IDs whose encoded bytes match it exactly.
    """
    if not protocol_id:
        raise ValueError("protocol_id is required; there is no default")
    pool, counts = build_pool(train_rows, train400_rows, probe_rows)
    labels = {row["label"] for row in train_rows}
    if contract is not None:
        observed = {**counts, "classes": len(labels)}
        mismatched = {k: (v, observed[k]) for k, v in contract.items() if observed[k] != v}
        if mismatched:
            raise ValueError(f"Fresh-pool contract changed (expected, observed): {mismatched}")
    if set(pool_hashes) != {row["id"] for row in pool}:
        raise ValueError("pHashes must cover exactly the pre-audit pool")
    external = []
    for split, hashes in sorted(reference_hashes.items()):
        for pool_id, ref_id, distance in near_pairs(pool_hashes, hashes):
            external.append(
                {
                    "pool_id": pool_id,
                    "reference_split": split,
                    "reference_id": ref_id,
                    "hamming": distance,
                }
            )
    for split, pool_ids in sorted((exact_matches or {}).items()):
        external += [
            {
                "pool_id": i,
                "reference_split": split,
                "reference_id": None,
                "hamming": 0,
                "exact_encoded_sha256": True,
            }
            for i in sorted(pool_ids)
        ]
    pool_drops = sorted({row["pool_id"] for row in external})
    audited = [row for row in pool if row["id"] not in set(pool_drops)]
    h_select, h_gate = split_holdout(audited, protocol_id)
    holdout = {row["id"] for row in [*h_select, *h_gate]}
    within = near_pairs(
        {r["id"]: pool_hashes[r["id"]] for r in audited},
        {r["id"]: pool_hashes[r["id"]] for r in audited},
        within=True,
    )
    within_rows, s_drops = [], set()
    for first, second, distance in within:
        straddles = (first in holdout) != (second in holdout)
        if straddles:
            s_drops.add(second if first in holdout else first)
        within_rows.append(
            {"ids": [first, second], "hamming": distance, "straddles_h_and_s": straddles}
        )
    s_rows = [row for row in audited if row["id"] not in holdout | s_drops]
    by_id = {row["id"]: row for row in s_rows}
    scans = {seed: [by_id[i] for i in scan_order(by_id, seed)] for seed in SCAN_SEEDS}
    audit = {
        "method": "64-bit orthonormal DCT-II pHash; 32px LANCZOS grayscale; 8x8 incl. DC; median",
        "image_transform": "vision_rl.aircraft_data.load_aircraft_rgb (bottom20 banner removed)",
        "hamming_max": HAMMING_MAX,
        "hamming_max_status": "pre-registered; unverified as optimal",
        "reference_splits_phash": sorted(reference_hashes),
        "reference_splits_exact_bytes": sorted(exact_matches or {}),
        "external_near_duplicates": sorted(
            external, key=lambda r: (r["pool_id"], r["reference_split"], str(r["reference_id"]))
        ),
        "pool_drops": pool_drops,
        "within_pool_pairs": within_rows,
        "s_drops": sorted(s_drops),
        "counts": {
            **counts,
            "pool_drops": len(pool_drops),
            "pool_after_audit": len(audited),
            "h_select": len(h_select),
            "h_gate": len(h_gate),
            "within_pool_pairs": len(within_rows),
            "within_pool_pairs_straddling": sum(r["straddles_h_and_s"] for r in within_rows),
            "s_drops": len(s_drops),
            "s": len(s_rows),
        },
    }
    return {
        "pool": audited,
        "h_select": h_select,
        "h_gate": h_gate,
        "s": s_rows,
        "scans": scans,
        "audit": audit,
        "class_counts": {
            name: dict(sorted(Counter(r["label"] for r in rows).items()))
            for name, rows in (
                ("pool", audited),
                ("h_select", h_select),
                ("h_gate", h_gate),
                ("s", s_rows),
            )
        },
    }


def planted_copy(image: Image.Image) -> Image.Image:
    """Resize to 60% with a different filter, then JPEG-recompress at quality 70."""
    small = image.convert("RGB").resize(
        (max(1, image.width * 3 // 5), max(1, image.height * 3 // 5)), Image.Resampling.BILINEAR
    )
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def forced_fire(pool_rows, pool_hashes: dict[str, int], load_image, seed: int = HOLDOUT_SEED):
    """The detector must flag a planted near-duplicate and must not flag distinct-class pairs.

    Raises on failure, so a run whose audit cannot fire never emits manifests.
    """
    rng = random.Random(seed)
    ids = sorted(row["id"] for row in pool_rows)
    labels = {row["id"]: row["label"] for row in pool_rows}
    planted_id = rng.choice(ids)
    planted_hash = phash64(planted_copy(load_image(planted_id)))
    planted_flagged = near_pairs({"planted": planted_hash}, {planted_id: pool_hashes[planted_id]})
    pairs, seen = [], set()
    per_class = Counter(labels.values())
    cross = (len(ids) ** 2 - sum(n * n for n in per_class.values())) // 2
    if cross < FORCED_FIRE_DISTINCT_PAIRS:
        raise ValueError(f"Forced fire needs {FORCED_FIRE_DISTINCT_PAIRS} distinct-class pairs")
    while len(pairs) < FORCED_FIRE_DISTINCT_PAIRS:
        first, second = sorted(rng.sample(ids, 2))
        if labels[first] == labels[second] or (first, second) in seen:
            continue
        seen.add((first, second))
        distance = (pool_hashes[first] ^ pool_hashes[second]).bit_count()
        pairs.append({"ids": [first, second], "hamming": distance})
    spurious = [pair for pair in pairs if pair["hamming"] <= HAMMING_MAX]
    report = {
        "planted_source_id": planted_id,
        "planted_transform": "resize 60% BILINEAR + JPEG quality 70",
        "planted_hamming": (planted_hash ^ pool_hashes[planted_id]).bit_count(),
        "planted_flagged": bool(planted_flagged),
        "distinct_class_pairs": pairs,
        "distinct_class_pairs_flagged": len(spurious),
        "passed": bool(planted_flagged) and not spurious,
    }
    if not report["passed"]:
        raise ValueError(f"pHash forced fire failed: {json.dumps(report)[:2000]}")
    return report


def write_outputs(output: Path, result, forced, provenance):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty manifest directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "pool.jsonl": jsonl_bytes(result["pool"]),
        "h_select.jsonl": jsonl_bytes(result["h_select"]),
        "h_gate.jsonl": jsonl_bytes(result["h_gate"]),
        "s.jsonl": jsonl_bytes(result["s"]),
        **{f"s_scan_seed{seed}.jsonl": jsonl_bytes(rows) for seed, rows in result["scans"].items()},
        "phash_audit.json": json.dumps(
            {**result["audit"], "forced_fire": forced}, indent=2, sort_keys=True
        ).encode()
        + b"\n",
    }
    for name, content in files.items():
        (output / name).write_bytes(content)
    manifest = {
        "schema_version": 1,
        "study_id": "aircraft-freshpool-screened-grpo-v1",
        "status": "manifests_emitted_before_any_model_output",
        **provenance,
        "rules": {
            "pool": "native TRAIN - train400 - probe16 outside train400 - pHash drops",
            "h_select": "per class, ascending SHA256 of compact JSON "
            f"[protocol_id,{HOLDOUT_SALT!r},{HOLDOUT_SEED},label,id]; ranks 1 and 3",
            "h_gate": "same hash; ranks 2 and 4",
            "s": "pool - H - S members of within-pool pairs straddling H and S",
            "scan": "random.Random(seed).shuffle(sorted(ids))",
        },
        "counts": result["audit"]["counts"],
        "class_counts": result["class_counts"],
        "forced_fire_passed": forced["passed"],
        "files_sha256": {name: sha(content) for name, content in sorted(files.items())},
    }
    temporary = output / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "manifest.json")
    return manifest


def main():
    from vision_rl.aircraft_data import load_aircraft_rgb

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path("data/fgvc-aircraft-2013b")
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="protocol JSON whose protocol_id salts the holdout hash",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        required=True,
        help="runs/aircraft-shared-probe16/metadata-selection.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path, default=root / "development")
    parser.add_argument(
        "--train400", type=Path, default=root / "candidates/fourshot-seed791/train400.jsonl"
    )
    parser.add_argument("--test-audit", type=Path, default=root / "audit/test-images.json")
    parser.add_argument(
        "--test-phashes",
        type=Path,
        help="JSON {id: 16-hex pHash} for TEST, computed by an authorised step",
    )
    args = parser.parse_args()
    if "sealed" in {part for p in (args.development, args.train400) for part in p.parts}:
        raise ValueError("This script never reads sealed/")
    protocol = json.loads(args.protocol.read_text())
    protocol_id = protocol["protocol_id"]
    train_rows = read_jsonl(args.development / "train.jsonl")
    val_rows = read_jsonl(args.development / "validation.jsonl")
    train400_rows = read_jsonl(args.train400)
    probe_rows = json.loads(args.probe.read_text())["selected"]
    pool, _ = build_pool(train_rows, train400_rows, probe_rows)
    paths = {("train", r["id"]): args.development / r["image"] for r in train_rows}
    paths |= {("validation", r["id"]): args.development / r["image"] for r in val_rows}

    def hashes(split, ids):
        return {i: phash64(load_aircraft_rgb(paths[(split, i)])) for i in ids}

    pool_hashes = hashes("train", [r["id"] for r in pool])
    references = {
        "validation": hashes("validation", [r["id"] for r in val_rows]),
        "train400": hashes("train", [r["id"] for r in train400_rows]),
    }
    test_status = "NOT_CHECKED"
    if args.test_phashes:
        references["test"] = {
            k: int(v, 16) for k, v in json.loads(args.test_phashes.read_text()).items()
        }
        test_status = "checked"
    test_bytes = {row["encoded_sha256"] for row in json.loads(args.test_audit.read_text())}
    pool_bytes = {r["id"]: sha(paths[("train", r["id"])].read_bytes()) for r in pool}
    exact = {"test": [i for i, digest in pool_bytes.items() if digest in test_bytes]}
    result = make_manifests(
        train_rows,
        train400_rows,
        probe_rows,
        protocol_id,
        pool_hashes,
        references,
        exact,
        contract=REAL_CONTRACT,
    )
    forced = forced_fire(pool, pool_hashes, lambda i: load_aircraft_rgb(paths[("train", i)]))
    inputs = [
        args.protocol,
        args.probe,
        args.development / "train.jsonl",
        args.development / "validation.jsonl",
        args.train400,
        args.test_audit,
    ]
    if args.test_phashes:
        inputs.append(args.test_phashes)
    provenance = {
        "protocol_id": protocol_id,
        "inputs_sha256": {str(p): sha(p.read_bytes()) for p in inputs},
        "test_phash_reference": test_status,
        "g0_audit_complete": test_status == "checked" and forced["passed"],
    }
    result["audit"]["pool_phashes"] = {i: f"{h:016x}" for i, h in sorted(pool_hashes.items())}
    manifest = write_outputs(args.output, result, forced, provenance)
    print(
        json.dumps(
            {"counts": manifest["counts"], "g0_audit_complete": manifest["g0_audit_complete"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
