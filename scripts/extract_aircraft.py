"""Bounded full-author FGVC image integrity audit with sealed native test paths.

Preserve one original JPEG copy; compute raw and lazy-banner-cropped pixel hashes.
No image display, model invocation, lossy re-encoding or full transformed cache.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

from prepare_aircraft import audit, canonical, read_archive, sha

from vision_rl.aircraft_data import (
    PREPROCESSING_ID,
    load_source_rgb,
    remove_banner,
    rgb_pixel_sha256,
)

URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
SOURCE_BYTES = 2753340328
DECLARED_MD5 = "32eca553f897e747706144f3bb444df0"
MAX_IMAGE_BYTES = int(3.5 * 1024**3)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path, name="sha256"):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, name).hexdigest()


def check_space(root, planned_bytes, reserve_bytes):
    available = shutil.disk_usage(root).free
    if available - planned_bytes < reserve_bytes:
        raise RuntimeError(
            f"Aircraft disk guard: free={available}, planned={planned_bytes}, "
            f"reserve={reserve_bytes}"
        )


def acquire(root, reserve):
    archive_path = root / "downloads/fgvc-aircraft-2013b.tar.gz"
    receipt_path = root / "audit/archive.json"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            archive_path.stat().st_size != SOURCE_BYTES
            or file_hash(archive_path) != receipt["sha256"]
        ):
            raise ValueError("Previously downloaded archive changed")
        return archive_path, receipt
    if archive_path.exists():
        raise ValueError("Archive without receipt exists; preserve it for explicit recovery")
    temporary = archive_path.with_suffix(".gz.part")
    if temporary.exists():
        raise ValueError("Partial archive exists; preserve it for explicit recovery")
    check_space(root, SOURCE_BYTES + MAX_IMAGE_BYTES, reserve)
    sha256, md5, total, next_log = hashlib.sha256(), hashlib.md5(), 0, 128 * 1024**2
    with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("xb") as target:
        if (
            response.status != 200
            or int(response.headers.get("Content-Length", -1)) != SOURCE_BYTES
        ):
            raise ValueError("Author archive size/status differs from reviewed resource plan")
        while chunk := response.read(4 * 1024**2):
            total += len(chunk)
            if total > SOURCE_BYTES:
                raise ValueError("Author archive exceeds declared byte size")
            check_space(root, SOURCE_BYTES - total + MAX_IMAGE_BYTES + len(chunk), reserve)
            target.write(chunk)
            sha256.update(chunk)
            md5.update(chunk)
            if total >= next_log:
                print(
                    json.dumps({"downloaded_bytes": total, "total_bytes": SOURCE_BYTES}), flush=True
                )
                next_log += 128 * 1024**2
    if total != SOURCE_BYTES:
        raise ValueError("Truncated author archive")
    receipt = {
        "url": URL,
        "bytes": total,
        "sha256": sha256.hexdigest(),
        "observed_md5": md5.hexdigest(),
        "declared_md5": DECLARED_MD5,
        "declared_md5_matches": md5.hexdigest() == DECLARED_MD5,
        "transport": "HTTPS with standard certificate verification",
        "hash_verification": "Complete received bytes hashed; declared MD5 comparison retained",
    }
    temporary.replace(archive_path)
    atomic_json(receipt_path, receipt)
    return archive_path, receipt


def groups(records, key):
    by_hash = collections.defaultdict(list)
    for row in records:
        by_hash[row[key]].append({"image_id": row["image_id"], "split": row["split"]})
    duplicates = [
        {"sha256": value, "members": members}
        for value, members in sorted(by_hash.items())
        if len(members) > 1
    ]
    return {
        "all": duplicates,
        "cross_split": [
            group for group in duplicates if len({row["split"] for row in group["members"]}) > 1
        ],
    }


def extract(root, archive, receipt, metadata_archive, reserve):
    metadata_content = metadata_archive.read_bytes()
    metadata_report, rows = audit(metadata_content)
    metadata_files = read_archive(metadata_content)
    by_id = {row["image_id"]: row for row in rows}
    seen, records, metadata_received, extracted_bytes = set(), [], {}, 0
    directories = {
        "train": "development/images/train",
        "val": "development/images/validation",
        "test": "sealed/images/test",
    }
    for folder in directories.values():
        (root / folder).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "fgvc-aircraft-2013b":
                raise ValueError("Unexpected full archive member path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Links or special entries are forbidden")
            relative = str(Path(*path.parts[1:]))
            if relative.startswith("data/images/"):
                if not re.fullmatch(r"data/images/[0-9]{7}\.jpg", relative):
                    raise ValueError("Unexpected image filename")
                image_id = path.stem
                if image_id not in by_id or image_id in seen or member.size > 20 * 1024**2:
                    raise ValueError("Unknown/duplicate image ID or oversized JPEG")
                extracted_bytes += member.size
                if extracted_bytes > MAX_IMAGE_BYTES:
                    raise ValueError("Expanded JPEG payload exceeds reviewed storage bound")
                content = source.extractfile(member).read()
                if len(content) != member.size:
                    raise ValueError("Truncated JPEG member")
                row = by_id[image_id]
                image_path = root / directories[row["split"]] / path.name
                encoded = sha(content)
                if image_path.exists():
                    if file_hash(image_path) != encoded:
                        raise ValueError("Existing original JPEG differs from author archive")
                else:
                    check_space(root, len(content), reserve)
                    with image_path.open("xb") as target:
                        target.write(content)
                with load_source_rgb(image_path) as original, remove_banner(original) as cropped:
                    record = {
                        **row,
                        "path": str(image_path.relative_to(root)),
                        "encoded_bytes": len(content),
                        "encoded_sha256": encoded,
                        "source_image_size": [original.width, original.height],
                        "model_image_size": [cropped.width, cropped.height],
                        "decoded_rgb_sha256": rgb_pixel_sha256(original),
                        "model_rgb_sha256": rgb_pixel_sha256(cropped),
                        "preprocessing_id": PREPROCESSING_ID,
                    }
                records.append(record)
                seen.add(image_id)
                if len(seen) % 500 == 0:
                    print(
                        json.dumps(
                            {"audited_images": len(seen), "extracted_bytes": extracted_bytes}
                        ),
                        flush=True,
                    )
            else:
                if relative not in metadata_files or member.size > 2 * 1024**2:
                    raise ValueError("Unexpected non-image member in author archive")
                content = source.extractfile(member).read()
                metadata_received[relative] = sha(content)
    if seen != set(by_id):
        raise ValueError("Full archive does not contain exactly every native metadata image ID")
    if set(metadata_received) != set(metadata_files):
        raise ValueError("Full archive metadata files differ from annotation archive")
    differences = {
        name: {"full_archive": metadata_received[name], "annotation_archive": sha(content)}
        for name, content in metadata_files.items()
        if sha(content) != metadata_received[name]
    }
    critical_mismatches = [
        name for name in differences if name.startswith("data/") or name.endswith(".m")
    ]
    if critical_mismatches:
        raise ValueError(
            "Full image archive annotations/evaluator differ from pinned author metadata"
        )
    records.sort(key=lambda row: row["image_id"])
    duplicates = {
        key: groups(records, key)
        for key in ("encoded_sha256", "decoded_rgb_sha256", "model_rgb_sha256")
    }
    split_reports = {}
    for split in directories:
        selected = [row for row in records if row["split"] == split]
        manifest = root / "audit" / (split + "-images.json")
        atomic_json(manifest, selected)
        split_reports[split] = {
            "images": len(selected),
            "classes": len({row["variant"] for row in selected}),
            "encoded_bytes": sum(row["encoded_bytes"] for row in selected),
            "metadata_image_manifest_sha256": file_hash(manifest),
            "canonical_records_sha256": sha(canonical(selected)),
        }
    report = {
        "status": "fallback_integrity_prepared_experiment_not_activated",
        "release": "FGVC-Aircraft 2013b current author release (100 variants)",
        "source": receipt,
        "metadata_source": metadata_report["source"],
        "metadata_archive_differences": differences,
        "annotation_and_evaluator_exact_matches": not critical_mismatches,
        "splits": split_reports,
        "images": len(records),
        "original_jpeg_bytes": extracted_bytes,
        "duplicates": duplicates,
        "preprocessing_id": PREPROCESSING_ID,
        "preprocessing": (
            "original JPEG -> RGB -> remove bottom20 pixel rows lazily, no resize/reencode"
        ),
        "source_manifest_fields_excluded_from_prompts": [
            "variant",
            "family",
            "manufacturer",
            "source_bbox_xyxy",
        ],
        "training_integrity_clear": not any(value["cross_split"] for value in duplicates.values()),
        "near_duplicate_or_pretraining_contamination_excluded": False,
        "model_inference_calls": 0,
        "images_visually_inspected": 0,
        "test_access": (
            "Automated integrity decoding only; sealed/images/test excluded from development"
        ),
        "image_reencoding_operations": 0,
        "persistent_transformed_image_copies": 0,
        "image_terms": metadata_report["image_terms"],
        "native_labels_and_membership_preserved": True,
    }
    atomic_json(root / "audit/image-audit.json", report)
    return report


def verify_completed(root, metadata_archive):
    """Check retained originals and manifests without fetching or re-decoding images."""
    report_path = root / "audit/image-audit.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    metadata, native_rows = audit(metadata_archive.read_bytes())
    if report["metadata_source"] != metadata["source"]:
        raise ValueError("Completed image audit references different author annotations")
    receipt = json.loads((root / "audit/archive.json").read_text())
    if report["source"] != receipt or report["preprocessing_id"] != PREPROCESSING_ID:
        raise ValueError("Completed image audit source/preprocessing identity changed")
    native = {row["image_id"]: row for row in native_rows}
    seen = set()
    total_bytes = 0
    for split, info in report["splits"].items():
        manifest = root / "audit" / (split + "-images.json")
        if file_hash(manifest) != info["metadata_image_manifest_sha256"]:
            raise ValueError("Completed image manifest changed")
        rows = json.loads(manifest.read_text())
        if len(rows) != info["images"] or sha(canonical(rows)) != info["canonical_records_sha256"]:
            raise ValueError("Completed image manifest count/content changed")
        for row in rows:
            image_id = row["image_id"]
            if image_id in seen or image_id not in native or native[image_id]["split"] != split:
                raise ValueError("Completed image manifest native membership changed")
            original = native[image_id]
            if any(
                row[key] != value for key, value in original.items() if key != "source_image_size"
            ):
                raise ValueError("Completed image manifest native annotations changed")
            expected_folder = {
                "train": "development/images/train",
                "val": "development/images/validation",
                "test": "sealed/images/test",
            }[split]
            expected_path = f"{expected_folder}/{image_id}.jpg"
            if row["path"] != expected_path:
                raise ValueError("Completed image manifest contains an unexpected image path")
            image_path = root / expected_path
            if (
                image_path.stat().st_size != row["encoded_bytes"]
                or file_hash(image_path) != row["encoded_sha256"]
            ):
                raise ValueError("Retained original JPEG changed after its integrity audit")
            total_bytes += row["encoded_bytes"]
            seen.add(image_id)
    if seen != set(native) or total_bytes != report["original_jpeg_bytes"]:
        raise ValueError("Completed image audit does not preserve the complete original dataset")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/fgvc-aircraft-2013b"))
    parser.add_argument(
        "--metadata-archive",
        type=Path,
        default=Path("runs/aircraft-audit/reproduced/annotations.tar.gz"),
    )
    parser.add_argument("--reserve-gib", type=float, default=10)
    parser.add_argument("--remove-verified-archive", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = verify_completed(args.root, args.metadata_archive)
    reused = report is not None
    archive = args.root / "downloads/fgvc-aircraft-2013b.tar.gz"
    if report is None:
        archive, receipt = acquire(args.root, int(args.reserve_gib * 1024**3))
        report = extract(
            args.root, archive, receipt, args.metadata_archive, int(args.reserve_gib * 1024**3)
        )
    if args.remove_verified_archive and archive.exists():
        # Only this downloaded payload; its full hash and all original JPEGs remain.
        if file_hash(archive) != report["source"]["sha256"]:
            raise ValueError("Refusing to delete a changed compressed archive")
        archive.unlink()
    print(
        json.dumps(
            {
                "images": report["images"],
                "integrity_clear": report["training_integrity_clear"],
                "archive_md5_matches": report["source"]["declared_md5_matches"],
                "reused_verified_originals_without_download": reused,
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
