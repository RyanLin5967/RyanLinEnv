"""Fetch only a prospectively selected native-TRAIN probe from a pinned mirror.

This does not establish original author image-byte equivalence or benchmark
performance. The complete author download and cross-split audit are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from extract_aircraft import atomic_json, check_space, file_hash
from prepare_aircraft import SOURCE_SHA256, SOURCE_URL, audit

from vision_rl import aircraft_data

REVISION = "16c6368266640c62cafa0cd59b9abb01291b7072"
REPO = "Shiym/ViT-FineTune"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
PREFIX = "fgvc-aircraft-2013b/data/images/"
MAX_BYTES = 200 * 1024**2


def validate_selection(selection, metadata, labels):
    by_id = {row["image_id"]: row for row in metadata}
    selected = selection["selected"]
    if len(selected) != 16 or len({row["id"] for row in selected}) != 16:
        raise ValueError("This bounded probe requires exactly16 unique prospectively selected IDs")
    if selection["canonical_labels"] != labels:
        raise ValueError("Probe must retain the full official canonical class vocabulary")
    for row in selected:
        official = by_id[row["id"]]
        if row["split"] != "train" or official["split"] != "train":
            raise ValueError("Probe image selection is restricted to native TRAIN")
        if row["label"] != official["variant"]:
            raise ValueError("Selected label differs from original author metadata")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-archive",
        type=Path,
        default=Path("runs/aircraft-audit/reproduced/annotations.tar.gz"),
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    report, metadata = audit(args.metadata_archive.read_bytes())
    selected = validate_selection(selection, metadata, report["canonical_labels"]["variant"])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [PREFIX + row["id"] + ".jpg" for row in selected]
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/paths-info/{REVISION}",
        data=json.dumps({"paths": paths}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        info = json.loads(response.read(1024**2))
    if {row["path"] for row in info} != set(paths) or len(info) != len(paths):
        raise ValueError("Pinned mirror omitted or added selected image IDs")
    if sum(row["size"] for row in info) > MAX_BYTES:
        raise ValueError("Selected probe exceeds the authorized200MiB transfer bound")
    info = {row["path"]: row for row in info}
    records, manifest = [], []
    for selected_row, remote in zip(selected, paths, strict=True):
        item = info[remote]
        expected_sha = item["lfs"]["oid"]
        destination = args.output / "original-jpeg" / (selected_row["id"] + ".jpg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            check_space(args.output, item["size"], 10 * 1024**3)
            with urllib.request.urlopen(BASE + remote, timeout=60) as response:
                content = response.read(item["size"] + 1)
            if len(content) != item["size"] or hashlib.sha256(content).hexdigest() != expected_sha:
                raise ValueError("Mirror JPEG does not match immutable revision LFS SHA256")
            destination.write_bytes(content)
        if destination.stat().st_size != item["size"] or file_hash(destination) != expected_sha:
            raise ValueError("Existing probe image differs from pinned mirror bytes")
        with aircraft_data.load_source_rgb(destination) as full:
            full_hash = aircraft_data.rgb_pixel_sha256(full)
            source_size = list(full.size)
        with aircraft_data.load_aircraft_rgb(destination) as cropped:
            model_hash = aircraft_data.rgb_pixel_sha256(cropped)
            model_size = list(cropped.size)
        manifest.append(
            {
                **selected_row,
                "image": os.path.relpath(destination, args.output),
            }
        )
        records.append(
            {
                **selected_row,
                "url": BASE + remote,
                "bytes": item["size"],
                "encoded_sha256": expected_sha,
                "decoded_rgb_sha256": full_hash,
                "model_rgb_sha256": model_hash,
                "source_image_size": source_size,
                "model_image_size": model_size,
            }
        )
    if len({row["model_rgb_sha256"] for row in records}) != len(records):
        raise ValueError("Selected mirror TRAIN probe contains duplicate model-visible pixels")
    spec = {
        "schema_version": 1,
        "benchmark": "FGVC-Aircraft 2013b fixed16 TRAIN mirror feasibility probe",
        "question": "Identify the exact aircraft variant shown in this image.",
        "labels": report["canonical_labels"]["variant"],
        "split_policy": (
            "Prospective16 nativeTRAIN IDs across16 classes, seed791; "
            "all100 labels; no held-out accuracy or official benchmark claim"
        ),
        "preprocessing": "Original JPEG decode RGB then remove exactly bottom20 rows lazily",
        "preprocessing_id": aircraft_data.PREPROCESSING_ID,
        "sources": [{"url": SOURCE_URL, "sha256": SOURCE_SHA256}]
        + [{"url": row["url"], "sha256": row["encoded_sha256"]} for row in records],
    }
    atomic_json(args.output / "data_spec.json", spec)
    (args.output / "train.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    )
    evidence = {
        "status": "fixed_train_probe_ready_original_author_image_equivalence_pending",
        "selection_sha256": file_hash(args.selection),
        "selection_path": str(args.selection),
        "mirror_repo": REPO,
        "mirror_revision": REVISION,
        "images": records,
        "downloaded_image_bytes": sum(row["bytes"] for row in records),
        "preprocessing_id": aircraft_data.PREPROCESSING_ID,
        "preprocessing_source_sha256": file_hash(Path(aircraft_data.__file__)),
        "full_dataset_cross_split_pixel_audit_complete": False,
        "author_original_image_byte_equivalence_verified": False,
        "test_images_accessed": 0,
        "model_calls": 0,
        "manual_image_views": 0,
        "image_terms": report["image_terms"],
        "train_manifest_sha256": file_hash(args.output / "train.jsonl"),
        "data_spec_sha256": file_hash(args.output / "data_spec.json"),
    }
    atomic_json(args.output / "provenance.json", evidence)
    print(
        json.dumps(
            {
                "images": len(records),
                "bytes": evidence["downloaded_image_bytes"],
                "status": evidence["status"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
