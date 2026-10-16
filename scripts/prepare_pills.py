"""Prepare and audit the native RF100-VL countingpills v2 COCO benchmark.

No model predictions are consumed. Native validation/test membership is immutable.
Only exact decoded-image matches to either heldout split are removed from train;
the rule is fixed here before any model is evaluated. Capture-hour overlap is
reported, not silently converted into a different benchmark. Test images never
appear in the generated visual-review sheets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SPLITS = {"train": "train", "validation": "valid", "test": "test"}
PROVENANCE = {
    "benchmark": "RF100-VL countingpills-exf0r-gfkt",
    "workspace": "rf100-vl",
    "project": "countingpills-exf0r-gfkt",
    "version": 2,
    "export_format": "coco",
    "export_metadata_endpoint": "https://api.roboflow.com/rf100-vl/countingpills-exf0r-gfkt/2/coco",
    "dataset_url": "https://universe.roboflow.com/rf-100-vl/countingpills-exf0r-gfkt",
    "benchmark_repository": "https://github.com/roboflow/rf100-vl",
    "license": "MIT",
    "license_url": "https://choosealicense.com/licenses/mit/",
}
PURGE_RULE = (
    "Remove train rows whose decoded RGB image hash exactly equals a validation or test "
    "image hash. Preserve native validation/test membership and all remaining train rows. "
    "Do not remove shared capture sessions or use model outputs to choose exclusions."
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def original_name(image: dict) -> str:
    return str(image.get("extra", {}).get("name") or image["file_name"].split(".rf.")[0])


def capture_hour(name: str) -> str | None:
    """Infer a candidate capture hour only from explicit Windows camera timestamps."""
    matched = re.match(r"^WIN_(\d{8})_(\d{2})_\d{2}_\d{2}_Pro(?:[._]|$)", name)
    return f"WIN:{matched[1]}:{matched[2]}" if matched else None


def _groups(rows: list[dict], field: str) -> dict:
    members: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            members[str(row[field])].append(row)
    overlaps = []
    for key, items in sorted(members.items()):
        counts = dict(sorted(Counter(item["split"] for item in items).items()))
        if len(counts) > 1:
            overlaps.append(
                {"group": key, "split_counts": counts, "ids": [item["id"] for item in items]}
            )
    return {
        "unique_groups": len(members),
        "rows_missing_group": sum(row.get(field) is None for row in rows),
        "cross_split_groups": len(overlaps),
        "cross_split_rows": sum(sum(item["split_counts"].values()) for item in overlaps),
        "within_split_repeated_groups": {
            split: sum(sum(r["split"] == split for r in items) > 1 for items in members.values())
            for split in SPLITS
        },
        "overlaps": overlaps,
    }


def fingerprint(rows: list[dict]) -> str:
    """Stable task-content identity independent of download/output location."""
    canonical = [
        {
            key: row[key]
            for key in (
                "id",
                "image_id",
                "split",
                "pixel_sha256",
                "file_sha256",
                "width",
                "height",
                "count",
                "bboxes",
                "category_ids",
                "source_original_name",
            )
        }
        for row in sorted(rows, key=lambda row: row["id"])
    ]
    return sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())


def _validate_box(box: object, width: int, height: int) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("COCO bbox must contain exactly four numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
        raise ValueError("COCO bbox must contain finite numbers")
    x, y, w, h = map(float, box)
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        raise ValueError("COCO bbox must contain finite numbers")
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("COCO bbox is degenerate or exceeds image bounds")
    return [x, y, w, h]


def inspect_source(source: Path) -> tuple[dict[str, list[dict]], dict]:
    source = source.resolve()
    records: dict[str, list[dict]] = {}
    summaries = {}
    all_source_files: list[tuple[str, str]] = []
    for split, dirname in SPLITS.items():
        split_dir = source / dirname
        annotation_path = split_dir / "_annotations.coco.json"
        content = annotation_path.read_bytes()
        coco = json.loads(content)
        category_rows = coco["categories"]
        categories = {item["id"]: item["name"] for item in category_rows}
        if len(categories) != len(category_rows):
            raise ValueError(f"{split}: duplicate category IDs")
        targets = {key for key, value in categories.items() if value == "pill"}
        if len(targets) != 1:
            raise ValueError(f"{split}: expected exactly one pill category")
        image_ids = {image["id"] for image in coco["images"]}
        if len(image_ids) != len(coco["images"]):
            raise ValueError(f"{split}: duplicate source image IDs")
        if len({image["file_name"] for image in coco["images"]}) != len(coco["images"]):
            raise ValueError(f"{split}: duplicate source image filenames")
        annotations: dict[int, list[dict]] = defaultdict(list)
        annotation_ids = set()
        for item in coco["annotations"]:
            if item["id"] in annotation_ids:
                raise ValueError(f"{split}: duplicate annotation IDs")
            annotation_ids.add(item["id"])
            if item["image_id"] not in image_ids or item["category_id"] not in categories:
                raise ValueError(f"{split}: dangling image/category annotation reference")
            if item.get("iscrowd", 0) != 0:
                raise ValueError(f"{split}: crowd annotations need a separate counting contract")
            annotations[item["image_id"]].append(item)
        rows = []
        duplicate_box_images = []
        metadata_keys: Counter[str] = Counter()
        for item in coco["images"]:
            original_path = split_dir / item["file_name"]
            path = original_path.resolve()
            if not path.is_relative_to(split_dir.resolve()) or original_path.is_symlink():
                raise ValueError(f"{split}: unsafe image path")
            file_hash = sha256(path.read_bytes())
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                if (item["width"], item["height"]) != (width, height):
                    raise ValueError(f"{split}: recorded dimensions disagree with image pixels")
                pixel_hash = sha256(str(image.size).encode() + image.tobytes())
            boxes, category_ids = [], []
            for annotation in annotations[item["id"]]:
                box = _validate_box(annotation["bbox"], width, height)
                if annotation["category_id"] not in targets:
                    raise ValueError(
                        f"{split}: non-pill annotations need an explicit target filter"
                    )
                boxes.append(box)
                category_ids.append(annotation["category_id"])
            row_id = f"{split}:{item['id']}:{pixel_hash[:16]}"
            if len({tuple(box) for box in boxes}) != len(boxes):
                duplicate_box_images.append(row_id)
            name = original_name(item)
            metadata_keys.update(item.get("extra", {}).keys())
            rows.append(
                {
                    "id": row_id,
                    "image_id": item["id"],
                    "split": split,
                    "image": str(path),
                    "file_name": item["file_name"],
                    "width": width,
                    "height": height,
                    "count": len(boxes),
                    "bboxes": boxes,
                    "category_ids": category_ids,
                    "pixel_sha256": pixel_hash,
                    "file_sha256": file_hash,
                    "source_original_name": name,
                    "capture_hour": capture_hour(name),
                    "prompt": "Locate every pill in the image with a bounding box.",
                }
            )
            all_source_files.append((f"{dirname}/{item['file_name']}", file_hash))
        all_source_files.append((f"{dirname}/_annotations.coco.json", sha256(content)))
        named = {row["file_name"] for row in rows}
        files = {
            path.name
            for path in split_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        }
        if files != named:
            raise ValueError(f"{split}: disk images and COCO image membership disagree")
        records[split] = rows
        summaries[split] = {
            "images": len(rows),
            "pill_boxes": sum(row["count"] for row in rows),
            "categories": categories,
            "count_distribution": dict(sorted(Counter(row["count"] for row in rows).items())),
            "count_min": min((row["count"] for row in rows), default=0),
            "count_max": max((row["count"] for row in rows), default=0),
            "dimensions": dict(
                sorted(Counter(f"{r['width']}x{r['height']}" for r in rows).items())
            ),
            "annotation_sha256": sha256(content),
            "fingerprint": fingerprint(rows),
            "duplicate_box_images": duplicate_box_images,
            "image_extra_metadata_keys": dict(metadata_keys),
            "source_info": coco.get("info"),
            "licenses": coco.get("licenses"),
        }
    for path in sorted(source.glob("README*")):
        all_source_files.append((path.name, sha256(path.read_bytes())))
    flattened = [row for rows in records.values() for row in rows]
    readme_path = source / "README.roboflow.txt"
    declared = (
        re.search(r"dataset includes (\d+) images", readme_path.read_text())
        if readme_path.exists()
        else None
    )
    report = {
        "schema_version": 1,
        "provenance": PROVENANCE,
        "source_tree_sha256": sha256(json.dumps(sorted(all_source_files)).encode()),
        "source_file_count": len(all_source_files),
        "source_file_checksums": dict(sorted(all_source_files)),
        "export_image_count": {
            "readme_declared": int(declared[1]) if declared else None,
            "actual_archive_and_coco": len(flattened),
            "unlisted_or_missing_image_files": 0,
            "caveat": "Use actual COCO/export membership. Metadata totals may disagree; "
            "the reason is not established and omitted images are not inferred as negatives.",
        },
        "splits": summaries,
        "groups": {
            key: _groups(flattened, key)
            for key in ("file_sha256", "pixel_sha256", "source_original_name", "capture_hour")
        },
        "geometry": {
            "all_boxes_finite_positive_and_in_bounds": True,
            "all_recorded_dimensions_match_pixels": True,
        },
        "label_review": {
            "status": "unreviewed",
            "automated_checks_are_not_a_human_annotation_audit": True,
            "test_images_visually_inspected": 0,
        },
        "session_generalization": {
            "claim_supported": False,
            "group_definition": "WIN_YYYYMMDD_HH from original filename; other names are unknown",
            "caveat": "Native random image splits share inferred capture hours. "
            "Export date_captured is not acquisition time. Unknown camera/session "
            "metadata cannot be reconstructed.",
        },
    }
    return records, report


def purge_train(records: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    heldout_hashes = {
        row["pixel_sha256"] for split in ("validation", "test") for row in records[split]
    }
    removed = [row for row in records["train"] if row["pixel_sha256"] in heldout_hashes]
    result = dict(records)
    result["train"] = [row for row in records["train"] if row["pixel_sha256"] not in heldout_hashes]
    return result, {
        "rule": PURGE_RULE,
        "removed_train_ids": [row["id"] for row in removed],
        "removed_train_count": len(removed),
        "before_counts": {split: len(rows) for split, rows in records.items()},
        "after_counts": {split: len(rows) for split, rows in result.items()},
        "fingerprints_after": {split: fingerprint(rows) for split, rows in result.items()},
    }


def write_clean_coco(source: Path, output: Path, rows: dict[str, list[dict]]) -> None:
    """Copy native COCO metadata, hardlink pixels; source files are never modified."""
    if output.exists():
        raise ValueError("Clean COCO output already exists; choose a new path")
    output.mkdir(parents=True)
    for split, dirname in SPLITS.items():
        directory = output / dirname
        directory.mkdir()
        coco = json.loads((source / dirname / "_annotations.coco.json").read_text())
        retained_ids = {row["image_id"] for row in rows[split]}
        coco["images"] = [row for row in coco["images"] if row["id"] in retained_ids]
        coco["annotations"] = [
            row for row in coco["annotations"] if row["image_id"] in retained_ids
        ]
        if split == "train":
            (directory / "_annotations.coco.json").write_text(json.dumps(coco) + "\n")
        else:
            shutil.copy2(
                source / dirname / "_annotations.coco.json", directory / "_annotations.coco.json"
            )
        for row in rows[split]:
            target = directory / row["file_name"]
            try:
                os.link(row["image"], target)
            except OSError:
                shutil.copy2(row["image"], target)


def contact_sheet(rows: list[dict], output: Path, seed: int = 42) -> list[str]:
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Test contact sheets are deliberately disabled")
    chosen = random.Random(seed).sample(sorted(rows, key=lambda row: row["id"]), min(12, len(rows)))
    cell_width, cell_height = 400, 340
    sheet = Image.new("RGB", (4 * cell_width, math.ceil(len(chosen) / 4) * cell_height), "white")
    font = ImageFont.load_default(size=14)
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
        image.thumbnail((cell_width - 12, cell_height - 54))
        scale_x, scale_y = image.width / row["width"], image.height / row["height"]
        image_draw = ImageDraw.Draw(image)
        for x, y, width, height in row["bboxes"]:
            image_draw.rectangle(
                (x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y),
                outline="#ff3030",
                width=2,
            )
        left, top = (index % 4) * cell_width + 6, (index // 4) * cell_height + 6
        sheet.paste(image, (left, top + 42))
        draw.text(
            (left, top),
            f"{row['split']} source ID {row['image_id']} | {row['count']} pills",
            fill="black",
            font=font,
        )
        draw.text((left, top + 19), row["source_original_name"][:42], fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return [row["id"] for row in chosen]


def prepare(
    source: Path, output: Path, clean_coco: Path | None = None, review_dir: Path | None = None
) -> dict:
    if output.exists():
        raise ValueError("Audit output already exists; choose a new path")
    records, report = inspect_source(source)
    clean, purge = purge_train(records)
    report["train_purge"] = purge
    output.mkdir(parents=True)
    for split, rows in clean.items():
        manifest_dir = output / ("sealed" if split == "test" else "development")
        manifest_dir.mkdir(exist_ok=True)
        (manifest_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    if clean_coco is not None:
        write_clean_coco(source, clean_coco, clean)
    if review_dir is not None:
        report["label_review"]["contact_sheets"] = {
            split: {
                "seed": 42,
                "selected_ids": contact_sheet(clean[split], review_dir / f"{split}-12.png"),
            }
            for split in ("train", "validation")
        }
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/rf100vl-pills/source"))
    parser.add_argument("--output", type=Path, default=Path("data/rf100vl-pills/prepared"))
    parser.add_argument("--clean-coco", type=Path, default=Path("data/rf100vl-pills/clean-coco"))
    parser.add_argument("--review-dir", type=Path, default=Path("runs/pills-audit"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        from dotenv import load_dotenv

        from vision_rl.roboflow import download_dataset

        load_dotenv()
        download_dataset("rf100-vl", "countingpills-exf0r-gfkt", 2, "coco", args.source)
    report = prepare(args.source, args.output, args.clean_coco, args.review_dir)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_tree_sha256": report["source_tree_sha256"],
                "counts": report["train_purge"]["after_counts"],
                "purged_train": report["train_purge"]["removed_train_count"],
                "cross_split_groups": {
                    key: value["cross_split_groups"] for key, value in report["groups"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
