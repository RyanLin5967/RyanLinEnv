"""Audit COCO count-task splits without training or selecting on held-out labels.

Perceptual matches are candidates, not proof of duplicate samples: a shared
background can dominate pHash even when a small target object has changed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "validation", "test")
SOURCE_SPLITS = {"train": "train", "validation": "valid", "test": "test"}


def source_group(filename: str) -> str:
    """Group Roboflow augmentation/export filenames by their original stem."""
    return Path(filename).name.split(".rf.", 1)[0]


def perceptual_hash(image: Image.Image) -> int:
    """63-bit DCT hash: 32px grayscale, top 8x8 frequencies, omit DC.

    Uses the orthonormal DCT-II basis. The median excludes DC, so exactly 31
    bits usually exceed the median. No learned model or task labels are used.
    """
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    frequency = np.arange(8)[:, None]
    position = np.arange(32)[None, :]
    basis = np.cos(np.pi * (2 * position + 1) * frequency / 64) * np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    coefficients = (basis @ pixels @ basis.T).ravel()[1:]
    return sum(int(value > np.median(coefficients)) << i for i, value in enumerate(coefficients))


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1, width1, height1 = first
    x2, y2, width2, height2 = second
    intersection = max(0, min(x1 + width1, x2 + width2) - max(x1, x2)) * max(
        0, min(y1 + height1, y2 + height2) - max(y1, y2)
    )
    union = width1 * height1 + width2 * height2 - intersection
    return intersection / union if union > 0 else 0.0


def group_report(records: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    for row in records:
        if row.get(key) is None:
            missing += 1
        else:
            groups[str(row[key])].append(row)
    overlaps = []
    for value, members in sorted(groups.items()):
        split_counts = dict(sorted(Counter(row["split"] for row in members).items()))
        if len(split_counts) > 1:
            overlaps.append({"value": value, "split_counts": split_counts})
    return {
        "unique_groups": len(groups),
        "missing_rows": missing,
        "within_split_repeated_groups": {
            split: sum(
                sum(row["split"] == split for row in members) > 1 for members in groups.values()
            )
            for split in SPLITS
        },
        "cross_split_groups": len(overlaps),
        "cross_split_rows": sum(sum(item["split_counts"].values()) for item in overlaps),
        "overlap_examples": overlaps[:10],
    }


def audit(source_root: Path, manifest_dir: Path) -> dict:
    records = []
    annotation_summary = {}
    errors: list[dict] = []
    source_checksums = {}
    manifest_checksums = {}
    for split in SPLITS:
        source_path = source_root / SOURCE_SPLITS[split] / "_annotations.coco.json"
        source_bytes = source_path.read_bytes()
        source_checksums[split] = hashlib.sha256(source_bytes).hexdigest()
        source = json.loads(source_bytes)
        categories = {item["id"]: item["name"] for item in source["categories"]}
        target_ids = {
            key for key, name in categories.items() if name in {"black-pawn", "white-pawn"}
        }
        if len(target_ids) != 2:
            raise ValueError(f"{split}: expected distinct black-pawn and white-pawn categories")
        annotations: dict[int, list[dict]] = defaultdict(list)
        for item in source["annotations"]:
            annotations[item["image_id"]].append(item)
        images = {item["file_name"]: item for item in source["images"]}
        manifest_path = manifest_dir / f"{split}.jsonl"
        manifest_bytes = manifest_path.read_bytes()
        manifest_checksums[split] = hashlib.sha256(manifest_bytes).hexdigest()
        rows = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
        counts: Counter[int] = Counter()
        cross_color_overlap = []
        repeated_boxes = []
        for row in rows:
            path = Path(row["image"])
            if not path.is_absolute():
                path = manifest_dir / path
            source_image = images[path.name]
            source_annotations = annotations[source_image["id"]]
            targets = [item for item in source_annotations if item["category_id"] in target_ids]
            target_count = len(targets)
            counts[target_count] += 1
            answer_index = ord(row["answer"]) - ord("A")
            answer = int(row["choices"][answer_index])
            if answer != target_count:
                errors.append({"id": row["id"], "error": "target_count_mismatch"})
            if row["split"] != split:
                errors.append({"id": row["id"], "error": "split_mismatch"})
            for first, second in itertools.combinations(targets, 2):
                if first["bbox"] == second["bbox"]:
                    repeated_boxes.append(row["id"])
                if first["category_id"] != second["category_id"]:
                    overlap = box_iou(first["bbox"], second["bbox"])
                    if overlap >= 0.9:
                        cross_color_overlap.append({"id": row["id"], "iou": overlap})
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                phash = perceptual_hash(image)
                small = np.asarray(image.resize((64, 64), Image.Resampling.LANCZOS), dtype=float)
            if row.get("image_sha256") != pixel_hash:
                errors.append({"id": row["id"], "error": "stored_pixel_hash_mismatch"})
            metadata = source_image.get("extra", {}).get("user_metadata", {})
            records.append(
                {
                    "split": split,
                    "id": row["id"],
                    "pixel_sha256": pixel_hash,
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_filename_group": source_group(path.name),
                    "source_original_name": source_image.get("extra", {}).get("name"),
                    "coveragecv_image_id": metadata.get("coveragecv_image_id"),
                    "coveragecv_view": metadata.get("coveragecv_view"),
                    "phash": phash,
                    "small_rgb": small,
                }
            )
        annotation_summary[split] = {
            "source_images": len(source["images"]),
            "manifest_images": len(rows),
            "source_annotations": len(source["annotations"]),
            "annotations_per_category": dict(
                sorted(
                    Counter(
                        categories[item["category_id"]] for item in source["annotations"]
                    ).items()
                )
            ),
            "target_count_distribution": dict(sorted(counts.items())),
            "exact_repeated_target_box_pairs": len(repeated_boxes),
            "cross_color_target_box_iou_at_least_0_9": len(cross_color_overlap),
            "cross_color_overlap_examples": cross_color_overlap[:10],
        }
    similarity = {}
    for first_split, second_split in itertools.combinations(SPLITS, 2):
        first_rows = [row for row in records if row["split"] == first_split]
        second_rows = [row for row in records if row["split"] == second_split]
        distances = []
        for first, second in itertools.product(first_rows, second_rows):
            distances.append(
                {
                    "first_id": first["id"],
                    "second_id": second["id"],
                    "phash_hamming_63bit": (first["phash"] ^ second["phash"]).bit_count(),
                    "rgb64_mean_absolute_error_0_to_1": float(
                        np.abs(first["small_rgb"] - second["small_rgb"]).mean() / 255
                    ),
                }
            )
        similarity[f"{first_split}:{second_split}"] = {
            "pair_count": len(distances),
            "phash_hamming_at_most": {
                str(threshold): sum(row["phash_hamming_63bit"] <= threshold for row in distances)
                for threshold in (0, 4, 8)
            },
            "rgb64_mae_at_most": {
                str(threshold): sum(
                    row["rgb64_mean_absolute_error_0_to_1"] <= threshold for row in distances
                )
                for threshold in (0.005, 0.01, 0.02)
            },
            "second_split_images_with_candidate": {
                f"phash_le_{threshold}": len(
                    {
                        row["second_id"]
                        for row in distances
                        if row["phash_hamming_63bit"] <= threshold
                    }
                )
                for threshold in (0, 4, 8)
            },
            "lowest_pixel_difference_examples": sorted(
                distances,
                key=lambda row: (row["rgb64_mean_absolute_error_0_to_1"], row["first_id"]),
            )[:10],
        }
    return {
        "schema_version": 1,
        "method": {
            "pixel_hash": "sha256(str(RGB_size) + decoded_RGB_bytes)",
            "phash": "63-bit orthonormal DCT-II; 32px LANCZOS grayscale; 8x8 excluding DC; median",
            "pixel_similarity": "mean RGB absolute difference after 64x64 LANCZOS resize / 255",
            "source_filename_group": "Roboflow filename before .rf. export/augmentation suffix",
            "target_semantics": "sum annotations with category name black-pawn or white-pawn",
            "near_duplicate_interpretation": "similarity candidates, not verified duplicates",
        },
        "source_coco_file_sha256": source_checksums,
        "manifest_file_sha256": manifest_checksums,
        "annotation_consistency": annotation_summary,
        "errors": errors,
        "groups": {
            key: group_report(records, key)
            for key in (
                "pixel_sha256",
                "file_sha256",
                "source_filename_group",
                "source_original_name",
                "coveragecv_image_id",
                "coveragecv_view",
            )
        },
        "cross_split_similarity": similarity,
        "limitations": [
            "Perceptual and resized pixel similarity can miss small target-object changes.",
            "Low distances can reflect a shared background rather than duplicate images.",
            "No capture/session independence is established by this audit.",
            "coveragecv_view semantics are unverified; it may not identify a camera/session.",
            "Annotation arithmetic and box checks do not establish visually correct ground truth.",
            "This audit cannot rule out benchmark contamination in pretrained models.",
            "No model training, inference, test selection, or data modification is performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/roboflow-smoke/source"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/roboflow-pawns/manifests"))
    parser.add_argument("--output", type=Path, default=Path("runs/data-audit.json"))
    args = parser.parse_args()
    report = audit(args.source_root, args.manifest_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}; annotation/hash/split errors: {len(report['errors'])}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
