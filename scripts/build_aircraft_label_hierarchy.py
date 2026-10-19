"""Build the TRAIN-only FGVC-Aircraft variant -> family -> manufacturer hierarchy.

Reads the audited native-TRAIN image records (data/fgvc-aircraft-2013b/audit/
train-images.json, written by the image audit from the pinned author annotations) and
writes the JSON that vlm_local_training.load_label_hierarchy accepts. It refuses any
record outside the TRAIN split, any variant seen with two families/manufacturers, and
any family seen with two manufacturers. VAL and TEST annotations are never opened.

  python scripts/build_aircraft_label_hierarchy.py \
    --train-records data/fgvc-aircraft-2013b/audit/train-images.json \
    --output scripts/exploratory_sweep/label-hierarchy-train-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "vision-rl-label-hierarchy-v1"


def build(records: list[dict], source_path: str, source_sha256: str) -> dict:
    if not records:
        raise ValueError("no TRAIN records")
    variants: dict[str, tuple[str, str]] = {}
    owners: dict[str, str] = {}
    for record in records:
        if record.get("split") != "train":
            raise ValueError(f"record {record.get('image_id')} is not native TRAIN")
        key = (record["family"], record["manufacturer"])
        if variants.setdefault(record["variant"], key) != key:
            raise ValueError(f"variant {record['variant']!r} has two family/manufacturer pairs")
        if owners.setdefault(record["family"], record["manufacturer"]) != record["manufacturer"]:
            raise ValueError(f"family {record['family']!r} has two manufacturers")
    return {
        "schema": SCHEMA,
        "source": {
            "split": "train",
            "path": source_path,
            "sha256": source_sha256,
            "records": len(records),
            "fields": ["variant", "family", "manufacturer"],
        },
        "levels": ["variant", "family", "manufacturer"],
        "counts": {
            "variant": len(variants),
            "family": len(owners),
            "manufacturer": len(set(owners.values())),
        },
        "labels": {
            variant: {"family": family, "manufacturer": manufacturer}
            for variant, (family, manufacturer) in sorted(variants.items())
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.train_records.read_bytes()
    value = build(
        json.loads(raw),
        args.train_records.as_posix(),
        hashlib.sha256(raw).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
