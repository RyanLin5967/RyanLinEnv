"""Verify pinned HF shards, preserve encoded images/labels, and audit all splits.

Only development/coco/train and development/coco/valid may be exposed to models.
sealed/test is decoded exclusively by this integrity auditor until final freeze.
No image is displayed or re-encoded. Completed receipts permit safe resumption.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
from prepare_wheat import HF_REPO, HF_REVISION, WILDS_SOURCE_HASHES, digest

BENCHMARK = "gwhd2021-hf-domain-ada-v1"
PINNED = {
    "train-00000-of-00002.parquet": (
        350373033,
        "3ada1fb45d9a0c4c9eca032f47acee6da4c5e7caa97df178b39209e2043b1bd4",
    ),
    "train-00001-of-00002.parquet": (
        349049990,
        "21a834eab7ca5a2b4bd2e2163dbde893d49b1fd9ec28d9964ae8f5b498258051",
    ),
    "validation-00000-of-00001.parquet": (
        263224769,
        "703d391e06743ca4341644acef4867e11ac804b4400a42deedd7c666d4723cd3",
    ),
    "test-00000-of-00001.parquet": (
        298290385,
        "74a937e2c8fb4b7a22287e4b49ee73be4f8a0f507fa99cd170879c9aa6d48412",
    ),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def hash_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_space(root, additional, reserve):
    available = shutil.disk_usage(root).free
    if available - additional < reserve:
        raise RuntimeError(
            f"Disk reserve guard: free={available}, planned={additional}, reserve={reserve}"
        )


def download_shard(name, destination, reserve):
    expected_size, expected_hash = PINNED[name]
    if destination.exists():
        if destination.stat().st_size != expected_size or hash_file(destination) != expected_hash:
            raise ValueError("Existing shard differs from pinned source; refusing to overwrite")
        return
    check_space(destination.parent, expected_size * 2, reserve)
    temporary = destination.with_suffix(".parquet.part")
    if temporary.exists():
        raise ValueError("Incomplete shard exists; preserve or explicitly remove it before retry")
    url = (
        f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/data/{name}?download=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "vision-rl-wheat-audit/1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=90) as response,
            temporary.open("xb") as target,
        ):
            if response.status != 200:
                raise ValueError("Expected complete pinned shard response")
            hasher, total, next_log = hashlib.sha256(), 0, 64 * 1024**2
            while content := response.read(4 * 1024**2):
                total += len(content)
                if total > expected_size:
                    raise ValueError("Shard response exceeds pinned size")
                check_space(destination.parent, len(content), reserve)
                target.write(content)
                hasher.update(content)
                if total >= next_log:
                    print(json.dumps({"shard": name, "downloaded_bytes": total}), flush=True)
                    next_log += 64 * 1024**2
            target.flush()
            os.fsync(target.fileno())
            if total != expected_size or hasher.hexdigest() != expected_hash:
                raise ValueError("Full shard size/SHA256 verification failed")
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Pinned public shard download failed; partial bytes retained") from None
    temporary.replace(destination)


def image_integrity(content, expected_size=(1024, 1024)):
    """Decode for integrity only, preserving the original encoded payload and orientation."""
    with Image.open(io.BytesIO(content)) as image:
        if image.size != expected_size:
            raise ValueError(
                f"Unexpected source image dimensions {image.size}; labels not rescaled"
            )
        orientation = image.getexif().get(274)
        rgb = image.convert("RGB")
        rgb.load()
        pixel = hashlib.sha256()
        pixel.update(canonical({"width": image.width, "height": image.height, "mode": "RGB"}))
        pixel.update(b"\0")
        pixel.update(rgb.tobytes())
        return {
            "width": image.width,
            "height": image.height,
            "encoded_sha256": digest(content),
            "decoded_rgb_sha256": pixel.hexdigest(),
            "encoded_bytes": len(content),
            "source_format": image.format,
            "source_mode": image.mode,
            "exif_orientation": orientation,
        }


def coco_row(row, image_id, annotation_start, sessions):
    original = Path(row["image"]["path"]).stem
    if not re.fullmatch(r"[0-9a-f]{64}", original):
        raise ValueError("Unexpected source image identity")
    categories, boxes = row["objects"]["categories"], row["objects"]["boxes"]
    if len(boxes) != len(categories) or any(type(c) is not int or c != 0 for c in categories):
        raise ValueError("Expected native category zero for every box")
    image = {
        "id": image_id,
        "file_name": original + ".jpg",
        "original_image_id": original,
        "width": 1024,
        "height": 1024,
        "domain": row["domain"],
        "session_id": sessions.index(row["domain"]),
        "country": row["country"],
        "location": row["location"],
        "development_stage": row["development_stage"],
    }
    annotations = []
    for index, box in enumerate(boxes):
        if len(box) != 4 or any(type(v) is not int for v in box) or min(box[2:]) <= 0:
            raise ValueError("Malformed native integer XYWH box")
        annotations.append(
            {
                "id": annotation_start + index,
                "image_id": image_id,
                "category_id": 1,
                "bbox": list(box),
                "area": box[2] * box[3],
                "iscrowd": 0,
                "source_annotation_index": index,
            }
        )
    return image, annotations


def duplicate_groups(records, field):
    grouped = collections.defaultdict(list)
    for row in records:
        grouped[row[field]].append(
            {
                "split": row["split"],
                "image_id": row["image_id"],
                "original_image_id": row["original_image_id"],
            }
        )
    groups = [
        {"sha256": sha, "members": members}
        for sha, members in sorted(grouped.items())
        if len(members) > 1
    ]
    return {
        "all": groups,
        "cross_split": [
            group for group in groups if len({row["split"] for row in group["members"]}) > 1
        ],
    }


def extract_dataset(metadata_dir, root, *, reserve_bytes, remove_verified_shards=False):
    import pyarrow.parquet as parquet

    code = metadata_dir / "wilds-globalwheat_dataset.py"
    if hash_file(code) != WILDS_SOURCE_HASHES["GlobalWheatDataset"]:
        raise ValueError("Domain catalogue source checksum mismatch")
    sessions = next(
        ast.literal_eval(statement.value)
        for statement in ast.parse(code.read_text()).body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SESSIONS" for target in statement.targets
        )
    )
    metadata = {
        name: json.loads((metadata_dir / (name + ".metadata.json")).read_text()) for name in PINNED
    }
    all_names = [Path(row["image"]["path"]).stem for rows in metadata.values() for row in rows]
    if len(all_names) != 6512 or len(set(all_names)) != 6512:
        raise ValueError("Expected 6512 unique source images")
    image_ids = {name: index + 1 for index, name in enumerate(sorted(all_names))}
    root.mkdir(parents=True, exist_ok=True)
    receipts_dir = root / "audit" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    downloads = root / "downloads"
    downloads.mkdir(exist_ok=True)
    folders = {
        "train": root / "development/coco/train",
        "validation": root / "development/coco/valid",
        "test": root / "sealed/test",
    }
    documents = {
        split: {
            "info": {
                "benchmark_id": BENCHMARK,
                "hf_revision": HF_REVISION,
                "native_split": split,
                "license": "CC-BY-4.0",
            },
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "wheat_head", "supercategory": "plant"}],
        }
        for split in folders
    }
    all_integrity, source_receipts, annotation_id = [], [], 1
    for name, rows in metadata.items():
        split = name.split("-")[0]
        folder, document = folders[split], documents[split]
        folder.mkdir(parents=True, exist_ok=True)
        receipt_path = receipts_dir / (name + ".json")
        metadata_hash = digest(canonical(rows))
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["source_sha256"] != PINNED[name][1]
                or receipt["metadata_sha256"] != metadata_hash
            ):
                raise ValueError("Existing extraction receipt source mismatch")
            integrity = receipt["images"]
            if len(integrity) != len(rows):
                raise ValueError("Existing extraction receipt row mismatch")
            for row, entry in zip(rows, integrity, strict=True):
                original = Path(row["image"]["path"]).stem
                if (
                    entry["original_image_id"] != original
                    or hash_file(folder / (original + ".jpg")) != entry["encoded_sha256"]
                ):
                    raise ValueError("Extracted image differs from completed receipt")
            print(
                json.dumps({"shard": name, "resumed_verified_images": len(integrity)}), flush=True
            )
        else:
            destination = downloads / name
            download_shard(name, destination, reserve_bytes)
            integrity, position = [], 0
            source = parquet.ParquetFile(destination)
            for batch in source.iter_batches(batch_size=8, use_threads=False):
                for row in batch.to_pylist():
                    content = row["image"].pop("bytes")
                    if position >= len(rows) or row != rows[position]:
                        raise ValueError("Downloaded annotations differ from audited metadata")
                    original = Path(row["image"]["path"]).stem
                    result = image_integrity(content)
                    if result["exif_orientation"] not in (None, 1):
                        raise ValueError(
                            "Unexpected EXIF orientation; refusing implicit geometry transform"
                        )
                    result.update(
                        {
                            "split": split,
                            "image_id": image_ids[original],
                            "original_image_id": original,
                            "domain": row["domain"],
                        }
                    )
                    target = folder / (original + ".jpg")
                    if target.exists():
                        if hash_file(target) != result["encoded_sha256"]:
                            raise ValueError("Existing image differs from pinned source")
                    else:
                        check_space(root, len(content), reserve_bytes)
                        with target.open("xb") as handle:
                            handle.write(content)
                    integrity.append(result)
                    position += 1
                if position % 256 == 0:
                    print(json.dumps({"shard": name, "audited_images": position}), flush=True)
            if position != len(rows):
                raise ValueError("Downloaded shard ended before all metadata rows")
            receipt = {
                "source_path": "data/" + name,
                "source_revision": HF_REVISION,
                "source_sha256": PINNED[name][1],
                "source_bytes": PINNED[name][0],
                "full_shard_checksum_verified": True,
                "metadata_sha256": metadata_hash,
                "images": integrity,
                "automated_audit_only": split == "test",
            }
            atomic_json(receipt_path, receipt)
            if remove_verified_shards:
                destination.unlink()  # Only this downloaded, SHA-verified temporary payload.
        all_integrity.extend(integrity)
        source_receipts.append({key: value for key, value in receipt.items() if key != "images"})
        for row in rows:
            original = Path(row["image"]["path"]).stem
            image, annotations = coco_row(row, image_ids[original], annotation_id, sessions)
            image["source_shard"] = "data/" + name
            image["native_split"] = split
            document["images"].append(image)
            document["annotations"].extend(annotations)
            annotation_id += len(annotations)
    file_duplicates = duplicate_groups(all_integrity, "encoded_sha256")
    pixel_duplicates = duplicate_groups(all_integrity, "decoded_rgb_sha256")
    cross_domains = {
        f"{first}:{second}": sorted(
            {item["domain"] for item in documents[first]["images"]}
            & {item["domain"] for item in documents[second]["images"]}
        )
        for first, second in itertools.combinations(documents, 2)
    }
    split_summaries = {}
    for split, document in documents.items():
        annotation_path = folders[split] / "_annotations.coco.json"
        atomic_json(annotation_path, document)
        records = [row for row in all_integrity if row["split"] == split]
        manifest = root / "audit" / (split + "-images.json")
        atomic_json(manifest, records)
        counts = collections.Counter(item["image_id"] for item in document["annotations"])
        outside = [
            item
            for item in document["annotations"]
            if item["bbox"][0] < 0
            or item["bbox"][1] < 0
            or item["bbox"][0] + item["bbox"][2] > 1024
            or item["bbox"][1] + item["bbox"][3] > 1024
        ]
        split_summaries[split] = {
            "images": len(document["images"]),
            "boxes": len(document["annotations"]),
            "empty_images": sum(counts[image["id"]] == 0 for image in document["images"]),
            "domains": dict(
                sorted(collections.Counter(item["domain"] for item in records).items())
            ),
            "annotation_path": str(annotation_path),
            "annotation_sha256": hash_file(annotation_path),
            "image_manifest_sha256": hash_file(manifest),
            "image_integrity_fingerprint": digest(canonical(records)),
            "encoded_bytes": sum(row["encoded_bytes"] for row in records),
            "out_of_frame_native_boxes_preserved": outside,
        }
    atomic_json(
        root / "audit/domain-map.json", {domain: index for index, domain in enumerate(sessions)}
    )
    report = {
        "benchmark_id": BENCHMARK,
        "source_revision": HF_REVISION,
        "license": "CC-BY-4.0",
        "source_shards": source_receipts,
        "splits": split_summaries,
        "file_duplicates": file_duplicates,
        "decoded_pixel_duplicates": pixel_duplicates,
        "cross_split_domain_overlaps": cross_domains,
        "all_dimensions": [1024, 1024],
        "decoded_pixel_mode": "RGB, no EXIF transpose",
        "images_visually_inspected": 0,
        "model_inference_calls": 0,
        "annotation_modifications": 0,
        "image_reencoding_operations": 0,
        "test_access": "automated integrity audit only; sealed/test is excluded from development",
        "native_labels_and_membership_preserved": True,
        "ready_for_training": not (
            file_duplicates["cross_split"]
            or pixel_duplicates["cross_split"]
            or any(cross_domains.values())
        ),
        "near_duplicate_and_pretraining_contamination_audit_performed": False,
        "original_zenodo_decoded_pixel_equivalence_verified": False,
    }
    atomic_json(root / "audit/extraction.json", report)
    print(
        json.dumps(
            {
                "audit": str(root / "audit/extraction.json"),
                "ready_for_training": report["ready_for_training"],
                "images": {key: value["images"] for key, value in split_summaries.items()},
                "cross_split_pixel_duplicate_groups": len(pixel_duplicates["cross_split"]),
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("runs/wheat-audit"))
    parser.add_argument("--root", type=Path, default=Path("data") / BENCHMARK)
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=16,
        help="Free-space floor including outstanding shared downloads",
    )
    parser.add_argument("--remove-verified-shards", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    extract_dataset(
        args.metadata_dir,
        args.root,
        reserve_bytes=int(args.reserve_gib * 1024**3),
        remove_verified_shards=args.remove_verified_shards,
    )
    print(json.dumps({"elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
