"""Publish native Aircraft development manifests after the complete integrity gate.

Original JPEGs stay in their existing paths; the shared lazy transform handles
the required bottom20 crop. No test manifest enters the development directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_aircraft import atomic_json, file_hash, verify_completed
from prepare_aircraft import audit
from verify_aircraft_clearance import require_clearance

from vision_rl.aircraft_data import PREPROCESSING_ID


def development_rows(records, split):
    expected = {"train": "train", "val": "validation"}[split]
    rows, seen = [], set()
    for record in records:
        if record["split"] != split or record["image_id"] in seen:
            raise ValueError("Native development membership changed or IDs repeated")
        image_id = record["image_id"]
        if record["path"] != f"development/images/{expected}/{image_id}.jpg":
            raise ValueError("Development image path does not match its audited native split")
        rows.append(
            {
                "id": image_id,
                "split": expected,
                "image": f"images/{expected}/{image_id}.jpg",
                "label": record["variant"],
            }
        )
        seen.add(image_id)
    return rows


def write_jsonl(path, rows):
    value = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text() != value:
        raise ValueError("Existing development manifest differs; refusing silent replacement")
    path.write_text(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/fgvc-aircraft-2013b"))
    parser.add_argument(
        "--metadata-archive",
        type=Path,
        default=Path("runs/aircraft-audit/reproduced/annotations.tar.gz"),
    )
    parser.add_argument("--probe", type=Path, default=Path("runs/aircraft-shared-probe16/prepared"))
    args = parser.parse_args()
    complete = verify_completed(args.root, args.metadata_archive)
    if complete is None or not complete["training_integrity_clear"]:
        raise ValueError("Full original image audit must pass before publishing training manifests")
    require_clearance(args.root, args.metadata_archive, args.probe, complete)
    metadata, _ = audit(args.metadata_archive.read_bytes())
    output = args.root / "development"
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    all_train = []
    for native_split, split in (("train", "train"), ("val", "validation")):
        records = json.loads((args.root / f"audit/{native_split}-images.json").read_text())
        rows = development_rows(records, native_split)
        write_jsonl(output / f"{split}.jsonl", rows)
        counts[split] = len(rows)
        if split == "train":
            all_train = rows
    fewshot_ids = set(metadata["prospective_train_fewshot"]["selected_image_ids"])
    fewshot = [row for row in all_train if row["id"] in fewshot_ids]
    if len(fewshot) != 1600:
        raise ValueError("Frozen prospective16-shot-per-class plan changed")
    write_jsonl(output / "train-16shot.jsonl", fewshot)
    spec = {
        "schema_version": 1,
        "benchmark": "FGVC-Aircraft 2013b author release,100variants/native splits",
        "question": "Identify the exact aircraft variant shown in this image.",
        "labels": metadata["canonical_labels"]["variant"],
        "split_policy": (
            "Original3334train/3333validation/3333test retained; test sealed separately. "
            "Optional train-16shot manifest uses frozen train-only16/class seed2718. "
            "Training protocol and budget must identify which training manifest is used."
        ),
        "preprocessing": "Original JPEG decode RGB then remove exactly bottom20 rows lazily",
        "preprocessing_id": PREPROCESSING_ID,
        "sources": [
            {"url": source["url"], "sha256": source["sha256"]}
            for source in (complete["source"], complete["metadata_source"])
        ],
    }
    spec_path = output / "data_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Existing data specification differs; refusing silent replacement")
    atomic_json(spec_path, spec)
    evidence = {
        "status": "development_manifests_prepared_experiment_not_activated",
        "counts": counts,
        "optional_train_16shot": len(fewshot),
        "source_image_audit_sha256": file_hash(args.root / "audit/image-audit.json"),
        "author_equivalence_receipt_sha256": file_hash(
            args.root / "audit/author-equivalence-v1.json"
        ),
        "files": {
            p.name: file_hash(p)
            for p in [
                output / "train.jsonl",
                output / "validation.jsonl",
                output / "train-16shot.jsonl",
                spec_path,
            ]
        },
        "test_rows_emitted": 0,
        "model_calls": 0,
    }
    atomic_json(args.root / "audit/development-manifests.json", evidence)
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
