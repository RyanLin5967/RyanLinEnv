"""Audit native two-class RF100-VL stomata v1 before any model is evaluated.

Uses the exact-image purge and immutable heldout-split contract from prepare_pills,
but validates the stomata class taxonomy and source-frame filename structure
separately. The shared helpers do not reinterpret either benchmark's labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import runpy
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COMMON = runpy.run_path(str(Path(__file__).with_name("prepare_pills.py")))
SPLITS = COMMON["SPLITS"]
sha256 = COMMON["sha256"]
fingerprint = COMMON["fingerprint"]
PROVENANCE = {
    "benchmark": "RF100-VL stomata-cells-upfae-ngum",
    "workspace": "rf100-vl",
    "project": "stomata-cells-upfae-ngum",
    "version": 1,
    "export_format": "coco",
    "export_metadata_endpoint": "https://api.roboflow.com/rf100-vl/stomata-cells-upfae-ngum/1/coco",
    "dataset_url": "https://universe.roboflow.com/rf100-vl/stomata-cells-upfae-ngum",
    "benchmark_repository": "https://github.com/roboflow/rf100-vl",
    "license": "MIT",
    "license_url": "https://choosealicense.com/licenses/mit/",
    "class_contract": {
        "close": "Completely shut stomata with no visible opening; encompass the entire stoma. "
        "Do not assign partially open stomata to close.",
        "open": "Stomata with a clear visible central opening; encompass the opening and "
        "surrounding edges. Avoid ambiguous or indistinct openings.",
    },
}


def acquisition_groups(name: str) -> dict:
    """Record explicit timestamps; a prefixed filename may be a source-frame crop."""
    matched = re.fullmatch(r"(?:(\d+)-(\d+)_)?(\d{8})-(\d{6})_jpg", name)
    if not matched:
        return {"source_frame": None, "capture_hour": None, "tile_prefix": None}
    try:
        datetime.strptime(f"{matched[3]}-{matched[4]}", "%Y%m%d-%H%M%S")
    except ValueError:
        return {"source_frame": None, "capture_hour": None, "tile_prefix": None}
    return {
        "source_frame": f"{matched[3]}-{matched[4]}",
        "capture_hour": f"{matched[3]}-{matched[4][:2]}",
        "tile_prefix": f"{matched[1]}-{matched[2]}" if matched[1] else None,
    }


def cross_scale_overlap(records: dict[str, list[dict]]) -> dict:
    train_full = {
        row["source_frame"]
        for row in records["train"]
        if row["source_frame"] is not None and row["tile_prefix"] is None
    }
    train_tiles = {
        row["source_frame"]
        for row in records["train"]
        if row["source_frame"] is not None and row["tile_prefix"] is not None
    }
    return {
        split: {
            "crops_with_same_timestamp_fullframe_in_train": sum(
                row["tile_prefix"] is not None and row["source_frame"] in train_full
                for row in records[split]
            ),
            "fullframes_with_same_timestamp_crops_in_train": sum(
                row["tile_prefix"] is None and row["source_frame"] in train_tiles
                for row in records[split]
            ),
            "total_crops": sum(row["tile_prefix"] is not None for row in records[split]),
            "total_fullframes": sum(row["tile_prefix"] is None for row in records[split]),
        }
        for split in ("validation", "test")
    }


def inspect_source(source: Path) -> tuple[dict[str, list[dict]], dict]:
    source = source.resolve()
    records, summaries = {}, {}
    checksums = {}
    for split, dirname in SPLITS.items():
        directory = source / dirname
        annotation_path = directory / "_annotations.coco.json"
        content = annotation_path.read_bytes()
        coco = json.loads(content)
        categories = {item["id"]: item["name"] for item in coco["categories"]}
        if len(categories) != len(coco["categories"]):
            raise ValueError(f"{split}: duplicate category IDs")
        targets = {key: value for key, value in categories.items() if value in {"open", "close"}}
        if len(targets) != 2 or set(targets.values()) != {"open", "close"}:
            raise ValueError(f"{split}: expected exactly open and close target categories")
        image_ids = {image["id"] for image in coco["images"]}
        if len(image_ids) != len(coco["images"]):
            raise ValueError(f"{split}: duplicate source image IDs")
        if len({image["file_name"] for image in coco["images"]}) != len(coco["images"]):
            raise ValueError(f"{split}: duplicate source image filenames")
        annotations = defaultdict(list)
        annotation_ids = set()
        for item in coco["annotations"]:
            if item["id"] in annotation_ids:
                raise ValueError(f"{split}: duplicate annotation IDs")
            annotation_ids.add(item["id"])
            if item["image_id"] not in image_ids or item["category_id"] not in targets:
                raise ValueError(f"{split}: dangling image or unsupported target category")
            if item.get("iscrowd", 0) != 0:
                raise ValueError(f"{split}: crowd annotations require a distinct metric contract")
            annotations[item["image_id"]].append(item)
        rows, duplicate_boxes, duplicate_crossclass_boxes = [], [], []
        for item in coco["images"]:
            original_path = directory / item["file_name"]
            path = original_path.resolve()
            if not path.is_relative_to(directory.resolve()) or original_path.is_symlink():
                raise ValueError(f"{split}: unsafe image path")
            file_hash = sha256(path.read_bytes())
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                if (item["width"], item["height"]) != (width, height):
                    raise ValueError(f"{split}: image dimensions disagree with metadata")
                pixel_hash = sha256(str(image.size).encode() + image.tobytes())
            boxes, category_ids = [], []
            for annotation in annotations[item["id"]]:
                boxes.append(COMMON["_validate_box"](annotation["bbox"], width, height))
                category_ids.append(annotation["category_id"])
            name = COMMON["original_name"](item)
            row_id = f"{split}:{item['id']}:{pixel_hash[:16]}"
            seen_boxes = defaultdict(set)
            for box, category_id in zip(boxes, category_ids, strict=True):
                seen_boxes[tuple(box)].add(category_id)
            if len(seen_boxes) != len(boxes):
                duplicate_boxes.append(row_id)
            if any(len(ids) > 1 for ids in seen_boxes.values()):
                duplicate_crossclass_boxes.append(row_id)
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
                    "category_names": [categories[key] for key in category_ids],
                    "class_counts": {
                        name: category_ids.count(key) for key, name in targets.items()
                    },
                    "pixel_sha256": pixel_hash,
                    "file_sha256": file_hash,
                    "source_original_name": name,
                    **acquisition_groups(name),
                    "prompt": "Locate each stoma and classify its state as open or close.",
                }
            )
            checksums[f"{dirname}/{item['file_name']}"] = file_hash
        checksums[f"{dirname}/_annotations.coco.json"] = sha256(content)
        named = {row["file_name"] for row in rows}
        files = {
            path.name
            for path in directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        }
        if named != files:
            raise ValueError(f"{split}: COCO and disk image membership disagree")
        records[split] = rows
        summaries[split] = {
            "images": len(rows),
            "object_boxes": sum(row["count"] for row in rows),
            "categories": categories,
            "annotations_per_class": {
                name: sum(row["class_counts"][name] for row in rows) for name in targets.values()
            },
            "count_distribution": dict(sorted(Counter(row["count"] for row in rows).items())),
            "count_min": min((row["count"] for row in rows), default=0),
            "count_max": max((row["count"] for row in rows), default=0),
            "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
            "annotation_sha256": sha256(content),
            "fingerprint": fingerprint(rows),
            "duplicate_box_images": duplicate_boxes,
            "identical_box_multiple_classes_images": duplicate_crossclass_boxes,
            "source_info": coco.get("info"),
            "licenses": coco.get("licenses"),
        }
    for path in sorted(source.glob("README*")):
        checksums[path.name] = sha256(path.read_bytes())
    rows = [row for split_rows in records.values() for row in split_rows]
    readme_path = source / "README.roboflow.txt"
    declared = (
        re.search(r"dataset includes (\d+) images", readme_path.read_text())
        if readme_path.exists()
        else None
    )
    report = {
        "schema_version": 1,
        "provenance": PROVENANCE,
        "source_tree_sha256": sha256(json.dumps(sorted(checksums.items())).encode()),
        "source_file_checksums": dict(sorted(checksums.items())),
        "export_image_count": {
            "readme_declared": int(declared[1]) if declared else None,
            "actual_archive_and_coco": len(rows),
            "unlisted_or_missing_image_files": 0,
        },
        "splits": summaries,
        "groups": {
            key: COMMON["_groups"](rows, key)
            for key in (
                "file_sha256",
                "pixel_sha256",
                "source_original_name",
                "source_frame",
                "capture_hour",
            )
        },
        "geometry": {
            "all_boxes_finite_positive_and_in_bounds": True,
            "all_recorded_dimensions_match_pixels": True,
        },
        "label_review": {
            "status": "unreviewed",
            "test_images_visually_inspected": 0,
            "automated_checks_are_not_a_human_annotation_audit": True,
        },
        "session_generalization": {
            "claim_supported": False,
            "source_frame_definition": "YYYYMMDD-HHMMSS timestamp after optional N-N_ prefix",
            "cross_scale_overlap_with_train": cross_scale_overlap(records),
            "caveat": "Prefixed filenames may be different crops/views of a common source frame. "
            "Repeated timestamps are recorded as candidate source-frame groups. "
            "Native random image splits do not establish unseen-frame generalization.",
        },
    }
    return records, report


def contact_sheet(rows: list[dict], output: Path, seed: int = 42) -> list[str]:
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Test contact sheets are deliberately disabled")
    chosen = random.Random(seed).sample(sorted(rows, key=lambda row: row["id"]), min(12, len(rows)))
    width, height = 400, 440
    sheet = Image.new("RGB", (4 * width, math.ceil(len(chosen) / 4) * height), "white")
    font, draw = ImageFont.load_default(size=14), ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
        image.thumbnail((width - 12, height - 54))
        sx, sy = image.width / row["width"], image.height / row["height"]
        boxes = ImageDraw.Draw(image)
        for (x, y, w, h), name in zip(row["bboxes"], row["category_names"], strict=True):
            color = "#00ff30" if name == "open" else "#ff00dd"
            boxes.rectangle((x * sx, y * sy, (x + w) * sx, (y + h) * sy), outline=color, width=2)
            boxes.text((x * sx + 2, y * sy + 2), name, font=font, fill=color)
        left, top = (index % 4) * width + 6, (index // 4) * height + 6
        sheet.paste(image, (left, top + 42))
        draw.text(
            (left, top),
            f"{row['split']} ID {row['image_id']} | "
            f"open {row['class_counts']['open']}, close {row['class_counts']['close']}",
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
        raise ValueError("Audit output exists; choose a new path")
    records, report = inspect_source(source)
    cleaned, purge = COMMON["purge_train"](records)
    report["train_purge"] = purge
    output.mkdir(parents=True)
    for split, rows in cleaned.items():
        directory = output / ("sealed" if split == "test" else "development")
        directory.mkdir(exist_ok=True)
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    if clean_coco is not None:
        COMMON["write_clean_coco"](source, clean_coco, cleaned)
    if review_dir is not None:
        report["label_review"]["contact_sheets"] = {
            split: {
                "seed": 42,
                "selected_ids": contact_sheet(cleaned[split], review_dir / f"{split}-12.png"),
            }
            for split in ("train", "validation")
        }
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def allocate_groups(rows: list[dict], seed: int = 2718) -> dict[str, list[str]]:
    """Shuffle explicit source-frame IDs, then use fixed 70/15/15 largest remainders.

    Ties in fractional remainders are resolved train, validation, test. Allocation
    consumes no class labels, images, model predictions, or visual-review outcomes.
    Unknown/malformed source groups fail closed instead of becoming singleton IDs.
    """
    if any(row.get("source_frame") is None for row in rows):
        raise ValueError(
            "Grouped holdout requires known, valid source-frame timestamps for every row"
        )
    for row in rows:
        try:
            datetime.strptime(row["source_frame"], "%Y%m%d-%H%M%S")
        except (ValueError, TypeError):
            raise ValueError("Grouped holdout contains an invalid source-frame timestamp") from None
    groups = sorted({row["source_frame"] for row in rows})
    if len(groups) < 7:
        raise ValueError("Grouped holdout requires at least seven source-frame groups")
    random.Random(seed).shuffle(groups)
    ideal = [len(groups) * weight for weight in (0.7, 0.15, 0.15)]
    sizes = [math.floor(value) for value in ideal]
    by_remainder = sorted(range(3), key=lambda index: (-(ideal[index] - sizes[index]), index))
    for index in by_remainder[: len(groups) - sum(sizes)]:
        sizes[index] += 1
    result, offset = {}, 0
    for split, size in zip(SPLITS, sizes, strict=True):
        result[split] = sorted(groups[offset : offset + size])
        offset += size
    return result


def prepare_group_heldout(
    source: Path,
    output: Path,
    seed: int = 2718,
    native_audit: Path | None = None,
) -> dict:
    """Build a separately named source-frame-heldout benchmark without editing native data."""
    if output.exists():
        raise ValueError("Grouped benchmark output exists; choose a new path")
    source, output = source.resolve(), output.resolve()
    native_rows, native_report = inspect_source(source)
    flattened = [row for split in SPLITS for row in native_rows[split]]
    allocation = allocate_groups(flattened, seed)
    group_to_split = {group: split for split, groups in allocation.items() for group in groups}
    protocol = {
        "benchmark": "rf100vl-stomata-group-heldout-v1",
        "official_native_score": False,
        "source_tree_sha256": native_report["source_tree_sha256"],
        "source_provenance": PROVENANCE,
        "seed": seed,
        "allocation": "Python random.Random(seed).shuffle(sorted(unique source_frame timestamps)); "
        "70/15/15 group counts by largest remainder; ties train,validation,test; "
        "consecutive shuffled groups assigned in that order.",
        "target_group_proportions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "source_frame_groups": allocation,
        "label_dependent_stratification": False,
        "model_predictions_consumed": False,
        "preserve_all_source_images_and_annotations": True,
        "native_files_modified": False,
        "evaluation_scope": "Held out source frames from the same five capture hours; "
        "does not establish unseen-session, plant, or laboratory generalization.",
    }
    protocol_hash = sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode())
    source_coco, source_images, source_annotations = {}, {}, {}
    for split, dirname in SPLITS.items():
        coco = json.loads((source / dirname / "_annotations.coco.json").read_text())
        source_coco[split] = coco
        source_images[split] = {item["id"]: item for item in coco["images"]}
        grouped = defaultdict(list)
        for annotation in coco["annotations"]:
            grouped[annotation["image_id"]].append(annotation)
        source_annotations[split] = grouped
    first_coco = source_coco["train"]
    if any(coco["categories"] != first_coco["categories"] for coco in source_coco.values()):
        raise ValueError(
            "Native splits have inconsistent category IDs; do not silently remap labels"
        )
    derivative_coco = {
        split: {
            "info": {
                "description": protocol["benchmark"],
                "version": "1",
                "source_tree_sha256": protocol["source_tree_sha256"],
                "protocol_sha256": protocol_hash,
            },
            "licenses": copy.deepcopy(first_coco.get("licenses", [])),
            "categories": copy.deepcopy(first_coco["categories"]),
            "images": [],
            "annotations": [],
        }
        for split in SPLITS
    }
    derived_rows = {split: [] for split in SPLITS}
    annotation_id = 0
    for image_id, row in enumerate(
        sorted(
            flattened,
            key=lambda item: (item["source_frame"], item["source_original_name"], item["id"]),
        ),
        start=1,
    ):
        original_split = row["split"]
        split = group_to_split[row["source_frame"]]
        filename = f"{original_split}__{row['image_id']}__{row['file_name']}"
        provenance = {
            "original_split": original_split,
            "original_image_id": row["image_id"],
            "original_row_id": row["id"],
            "original_file_name": row["file_name"],
            "source_frame": row["source_frame"],
        }
        image_item = copy.deepcopy(source_images[original_split][row["image_id"]])
        image_item.update(id=image_id, file_name=filename)
        image_item["extra"] = {**image_item.get("extra", {}), "derived_provenance": provenance}
        derivative_coco[split]["images"].append(image_item)
        for original in source_annotations[original_split][row["image_id"]]:
            annotation_id += 1
            annotation = copy.deepcopy(original)
            annotation.update(id=annotation_id, image_id=image_id)
            annotation["derived_provenance"] = {
                "original_split": original_split,
                "original_annotation_id": original["id"],
                "original_image_id": row["image_id"],
            }
            derivative_coco[split]["annotations"].append(annotation)
        derived_rows[split].append(
            {
                **row,
                "id": f"{split}:{image_id}:{row['pixel_sha256'][:16]}",
                "image_id": image_id,
                "split": split,
                "file_name": filename,
                "image": str(output / SPLITS[split] / filename),
                "provenance": provenance,
            }
        )
    combined = [row for rows in derived_rows.values() for row in rows]
    groups = {
        key: COMMON["_groups"](combined, key)
        for key in (
            "file_sha256",
            "pixel_sha256",
            "source_original_name",
            "source_frame",
            "capture_hour",
        )
    }
    if any(
        groups[key]["cross_split_groups"]
        for key in ("file_sha256", "pixel_sha256", "source_original_name", "source_frame")
    ):
        raise ValueError(
            "Grouped allocation still contains duplicate content or shared source frames"
        )
    reviewed_ids = set()
    if native_audit is not None and native_audit.exists():
        existing = json.loads(native_audit.read_text())
        if existing.get("source_tree_sha256") != native_report["source_tree_sha256"]:
            raise ValueError("Prior visual-review audit belongs to a different source dataset")
        for sheet in existing.get("label_review", {}).get("contact_sheets", {}).values():
            reviewed_ids.update(sheet["selected_ids"])
    report = {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "splits": {
            split: {
                "source_frame_groups": len(allocation[split]),
                "images": len(rows),
                "object_boxes": sum(row["count"] for row in rows),
                "annotations_per_class": {
                    name: sum(row["class_counts"][name] for row in rows)
                    for name in ("close", "open")
                },
                "fingerprint": fingerprint(rows),
                "original_split_counts": dict(
                    Counter(row["provenance"]["original_split"] for row in rows)
                ),
            }
            for split, rows in derived_rows.items()
        },
        "groups": groups,
        "geometry": native_report["geometry"],
        "label_review": {
            "status": "unreviewed",
            "new_grouped_test_images_visually_inspected": 0,
            "prior_native_train_validation_spot_check_ids": sorted(reviewed_ids),
            "prior_review_ids_by_new_split": {
                split: sorted(
                    row["provenance"]["original_row_id"]
                    for row in rows
                    if row["provenance"]["original_row_id"] in reviewed_ids
                )
                for split, rows in derived_rows.items()
            },
            "assignment_modified_based_on_prior_review": False,
            "caveat": "Some randomly assigned heldout images may have appeared in native train/val "
            "contact sheets before this derivative existed. No test model outputs were "
            "used to define or select the split, and source annotations remain unchanged.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".grouped-", dir=output.parent) as temp:
        staged = Path(temp) / "benchmark"
        staged.mkdir()
        for split, dirname in SPLITS.items():
            directory = staged / dirname
            directory.mkdir()
            for row in derived_rows[split]:
                provenance = row["provenance"]
                original = (
                    source / SPLITS[provenance["original_split"]] / provenance["original_file_name"]
                )
                target = directory / row["file_name"]
                try:
                    os.link(original, target)
                except OSError:
                    shutil.copy2(original, target)
            content = json.dumps(derivative_coco[split], sort_keys=True) + "\n"
            (directory / "_annotations.coco.json").write_text(content)
            report["splits"][split]["annotation_sha256"] = sha256(content.encode())
            manifest_directory = staged / ("sealed" if split == "test" else "development")
            manifest_directory.mkdir(exist_ok=True)
            (manifest_directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in derived_rows[split])
            )
        (staged / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        (staged / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        staged.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/rf100vl-stomata/source"))
    parser.add_argument("--output", type=Path, default=Path("data/rf100vl-stomata/prepared"))
    parser.add_argument("--clean-coco", type=Path, default=Path("data/rf100vl-stomata/clean-coco"))
    parser.add_argument("--review-dir", type=Path, default=Path("runs/stomata-audit"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--group-heldout",
        action="store_true",
        help="Create the separately named source-frame-heldout benchmark",
    )
    parser.add_argument(
        "--group-output", type=Path, default=Path("data/rf100vl-stomata/grouped-coco")
    )
    args = parser.parse_args()
    if args.download:
        from dotenv import load_dotenv

        from vision_rl.roboflow import download_dataset

        load_dotenv()
        download_dataset("rf100-vl", "stomata-cells-upfae-ngum", 1, "coco", args.source)
    if args.group_heldout:
        report = prepare_group_heldout(
            args.source, args.group_output, native_audit=args.output / "audit.json"
        )
        print(
            json.dumps(
                {
                    "output": str(args.group_output),
                    "protocol_sha256": report["protocol_sha256"],
                    "splits": report["splits"],
                    "prior_review_ids_by_new_split": report["label_review"][
                        "prior_review_ids_by_new_split"
                    ],
                },
                indent=2,
            )
        )
        return
    report = prepare(args.source, args.output, args.clean_coco, args.review_dir)
    print(
        json.dumps(
            {
                "counts": report["train_purge"]["after_counts"],
                "purged_train": report["train_purge"]["removed_train_count"],
                "source_tree_sha256": report["source_tree_sha256"],
                "cross_split_groups": {
                    key: value["cross_split_groups"] for key, value in report["groups"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
