"""Metadata-only GWHD2021 audit; never download full image columns or train a model.

The author Hugging Face release is compared to the original competition CSVs
inside the Zenodo archive using bounded HTTP byte ranges. This is deliberately
not a COCO converter: original annotation issues and evaluator ambiguity must be
preserved and resolved by an explicit experiment protocol before image training.
PyArrow is needed only for --download-metadata; local audits use the standard library.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HF_REPO = "Etienne-David/GlobalWheatHeadDataset2021"
HF_REVISION = "d223a227ee21fca3dbab9605d3fbcdcc2e33abc5"
ZENODO_URL = "https://zenodo.org/records/5092309/files/gwhd_2021.zip?download=1"
ZENODO_SIZE = 10_215_190_132
ZENODO_MD5 = "22b4b542c9ae7e056d7fcdeae9ecaed5"
SPLITS = {
    "train": "competition_train.csv",
    "validation": "competition_val.csv",
    "test": "competition_test.csv",
}
CSV_SHA256 = {
    "competition_train.csv": "1af1d8e386643afe8cd1b41d89ca2634d3212908bdbe58a9d7cd5b32e4048176",
    "competition_val.csv": "1e7ed76fdf67c4f01de1b3ac30ddeddb62577c580c3b0f882101b99e4b24ff2d",
    "competition_test.csv": "8c801e485a34ccf2b9ae2e05bc3bcb2e2043a4df572921d4bf35e1de2b273baa",
    "metadata_dataset.csv": "1fb0a498a4a2d3fff2424b65ecbc41f0ca10d5fc26137b09ffc42c8f3f92d49a",
}
WILDS_COMMIT = "472677590de351857197a9bf24958838c39c272b"
WILDS_SOURCE_HASHES = {
    "DetectionAccuracy": "4fe677510f6e7e11005e0e73f16189c5a5c8a5664fe126055426dcbe5c873d72",
    "GlobalWheatDataset": "25948859e68e6cb4759709bc0158db060c5ab0d15aa1a467ac1f41383e7fcc1e",
}
WILDS_FILES = {
    "train": "official_train.csv",
    "validation": "official_val.csv",
    "test": "official_test.csv",
    "id_validation": "fixed_train_val.csv",
    "id_test": "fixed_train_test.csv",
}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class RangeReader(io.RawIOBase):
    """Fail closed if a server ignores Range or metadata transfer exceeds 64 MiB."""

    def __init__(self, url: str, size: int, budget: dict):
        self.url, self.size, self.budget, self.position = url, size, budget, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Unsupported seek mode")
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative range offset")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 16 * 1024**2 or self.budget["bytes"] + size > 64 * 1024**2:
            raise ValueError("Metadata transfer budget exceeded; full image downloads are disabled")
        start = self.position
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + size - 1}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {start}-{start + size - 1}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range", "").split("/")[0] != expected
                ):
                    raise ValueError("Server ignored or changed the requested metadata range")
                content = response.read(size + 1)
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Public dataset metadata range request failed") from None
        if len(content) != size:
            raise ValueError("Truncated metadata byte range")
        self.position += size
        self.budget["bytes"] += size
        return content

    def readinto(self, buffer):
        content = self.read(len(buffer))
        buffer[: len(content)] = content
        return len(content)


def fetch_metadata(output: Path) -> dict:
    import pyarrow.parquet as parquet

    output.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}?recursive=true&expand=false"
    with urllib.request.urlopen(tree_url, timeout=60) as response:
        content = response.read(1024**2 + 1)
    if len(content) > 1024**2:
        raise ValueError("Unexpectedly large dataset tree metadata")
    tree = json.loads(content)
    (output / "hf-tree.json").write_text(json.dumps(tree, indent=2) + "\n")
    budget = {"bytes": len(content)}
    storage, footer_bytes = [], 0
    for entry in tree:
        if not entry["path"].endswith(".parquet"):
            continue
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{entry['path']}"
        remote = RangeReader(url, entry["size"], budget)
        before_footer = budget["bytes"]
        source = parquet.ParquetFile(remote, pre_buffer=False)
        footer_bytes += budget["bytes"] - before_footer
        image_bytes_bound = sum(
            source.metadata.row_group(group).column(column).total_uncompressed_size
            for group in range(source.metadata.num_row_groups)
            for column in range(source.metadata.row_group(group).num_columns)
            if source.metadata.row_group(group).column(column).path_in_schema == "image.bytes"
        )
        storage.append(
            {
                "path": entry["path"],
                "download_bytes": entry["size"],
                "image_bytes_column_uncompressed_upper_bound": image_bytes_bound,
            }
        )
        # Crucial: image.path is a leaf column; selecting image would fetch image.bytes.
        rows = source.read(
            columns=["image.path", "domain", "country", "location", "development_stage", "objects"],
            use_threads=False,
        ).to_pylist()
        destination = output / (Path(entry["path"]).name + ".metadata.json")
        destination.write_text(json.dumps(rows) + "\n")
        print(f"Fetched metadata for {entry['path']}: {len(rows)} rows", flush=True)
    storage_report = {
        "files": storage,
        "total_download_bytes": sum(item["download_bytes"] for item in storage),
        "extracted_encoded_images_upper_bound": sum(
            item["image_bytes_column_uncompressed_upper_bound"] for item in storage
        ),
        "parquet_plus_extracted_upper_bound": sum(
            item["download_bytes"] + item["image_bytes_column_uncompressed_upper_bound"]
            for item in storage
        ),
        "footer_network_bytes": footer_bytes,
        "full_image_columns_read": 0,
    }
    (output / "hf-storage-metadata.json").write_text(json.dumps(storage_report, indent=2) + "\n")
    remote = RangeReader(ZENODO_URL, ZENODO_SIZE, budget)
    with zipfile.ZipFile(remote) as archive:
        index = [
            {
                "filename": entry.filename,
                "size": entry.file_size,
                "compressed_size": entry.compress_size,
                "crc32": entry.CRC,
            }
            for entry in archive.infolist()
        ]
        (output / "zenodo-zip-index.json").write_text(json.dumps(index) + "\n")
        for filename, expected in CSV_SHA256.items():
            entries = [
                entry for entry in archive.infolist() if Path(entry.filename).name == filename
            ]
            if len(entries) != 1 or entries[0].file_size > 8 * 1024**2:
                raise ValueError("Missing, duplicate, or oversized original CSV")
            content = archive.read(entries[0])
            if digest(content) != expected:
                raise ValueError("Original competition CSV checksum mismatch")
            (output / filename).write_bytes(content)
    transfer = {
        "network_bytes_metadata_only": budget["bytes"],
        "hf_revision": HF_REVISION,
        "full_images_downloaded": 0,
        "image_pixel_arrays_decoded": 0,
    }
    (output / "transfer.json").write_text(json.dumps(transfer, indent=2) + "\n")
    return transfer


def csv_xyxy_to_xywh(text: str) -> list[list[int]]:
    if text == "no_box":
        return []
    result = []
    for value in text.split(";"):
        parts = value.split()
        if len(parts) != 4:
            raise ValueError("CSV box must have exactly four coordinates")
        coordinates = [float(part) for part in parts]
        if not all(math.isfinite(part) and part.is_integer() for part in coordinates):
            raise ValueError("CSV coordinates must be finite integers")
        x, y, right, bottom = map(int, coordinates)
        if right <= x or bottom <= y:
            raise ValueError("CSV box has nonpositive area")
        result.append([x, y, right - x, bottom - y])
    return result


def audit_rows(rows: list[dict], originals: list[dict]) -> dict:
    by_name = collections.defaultdict(list)
    for row in originals:
        by_name[Path(row["image_name"]).stem].append(row)
    names = [Path(row["image"]["path"]).stem for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Hugging Face split contains repeated image IDs")
    missing, extra = sorted(set(by_name) - set(names)), sorted(set(names) - set(by_name))
    exact, duplicates, box_errors, domain_errors = 0, [], [], []
    out_of_frame_boxes, repeat_boxes, counts = [], [], []
    for row in rows:
        name = Path(row["image"]["path"]).stem
        boxes, categories = row["objects"]["boxes"], row["objects"]["categories"]
        if len(boxes) != len(categories) or any(category != 0 for category in categories):
            raise ValueError("Expected one category 0 per wheat-head box")
        candidates = by_name.get(name, [])
        matches = [
            index
            for index, original in enumerate(candidates)
            if csv_xyxy_to_xywh(original["BoxesString"]) == boxes
            and original["domain"] == row["domain"]
        ]
        exact += bool(matches)
        if not matches:
            box_errors.append(name)
        if row["domain"] not in {original["domain"] for original in candidates}:
            domain_errors.append(name)
        if len(candidates) > 1:
            duplicates.append(
                {
                    "image_id": name,
                    "source_row_count": len(candidates),
                    "hf_matches_original_indices": matches,
                    "source_counts": [
                        len(csv_xyxy_to_xywh(item["BoxesString"])) for item in candidates
                    ],
                }
            )
        counts.append(len(boxes))
        repeated = [
            {"bbox_xywh": list(box), "multiplicity": count}
            for box, count in collections.Counter(map(tuple, boxes)).items()
            if count > 1
        ]
        if repeated:
            repeat_boxes.append({"image_id": name, "domain": row["domain"], "boxes": repeated})
        for box in boxes:
            if len(box) != 4 or any(type(value) is not int for value in box):
                raise ValueError("Expected four integer XYWH box coordinates")
            x, y, width, height = box
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1024 or y + height > 1024:
                out_of_frame_boxes.append(
                    {"image_id": name, "domain": row["domain"], "bbox_xywh": box}
                )
    sorted_counts = sorted(counts)

    def quantile(q):
        position = (len(sorted_counts) - 1) * q
        left, right = math.floor(position), math.ceil(position)
        return sorted_counts[left] + (sorted_counts[right] - sorted_counts[left]) * (
            position - left
        )

    return {
        "images": len(rows),
        "source_csv_rows": len(originals),
        "source_unique_ids": len(by_name),
        "missing_original_ids": missing,
        "extra_hf_ids": extra,
        "exact_original_annotation_matches": exact,
        "annotation_mismatch_ids": box_errors,
        "domain_mismatch_ids": domain_errors,
        "duplicate_original_rows": duplicates,
        "domains": dict(sorted(collections.Counter(row["domain"] for row in rows).items())),
        "countries": dict(sorted(collections.Counter(row["country"] for row in rows).items())),
        "boxes": sum(counts),
        "empty_images": counts.count(0),
        "count_min": min(counts),
        "count_max": max(counts),
        "count_quantiles": {str(q): quantile(q) for q in (0.5, 0.9, 0.95, 0.99)},
        "images_exceeding_query_cap": {
            str(cap): sum(count > cap for count in counts) for cap in (100, 128, 200, 300, 500)
        },
        "boxes_outside_declared_1024_dimensions": out_of_frame_boxes,
        "repeated_exact_boxes": repeat_boxes,
        "metadata_sha256": digest(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()),
    }


def audit_directory(source: Path) -> dict:
    records, summaries, source_hashes = {}, {}, {}
    for filename, expected in CSV_SHA256.items():
        source_hashes[filename] = digest((source / filename).read_bytes())
        if source_hashes[filename] != expected:
            raise ValueError("Original competition CSV checksum mismatch")
    for split, csv_file in SPLITS.items():
        rows = [
            row
            for path in sorted(source.glob(f"{split}-*.parquet.metadata.json"))
            for row in json.loads(path.read_text())
        ]
        if not rows:
            raise ValueError(f"Missing {split} metadata; run --download-metadata first")
        with (source / csv_file).open() as handle:
            originals = list(csv.DictReader(handle))
        records[split], summaries[split] = rows, audit_rows(rows, originals)
    cross_split = {}
    for first, second in itertools.combinations(SPLITS, 2):

        def values(split, field):
            if field == "image_id":
                return {Path(row["image"]["path"]).stem for row in records[split]}
            return {row[field] for row in records[split]}

        cross_split[f"{first}:{second}"] = {
            field: sorted(values(first, field) & values(second, field))
            for field in ("image_id", "domain", "country")
        }
    image_index = [
        entry
        for entry in json.loads((source / "zenodo-zip-index.json").read_text())
        if Path(entry["filename"]).suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    all_ids = {Path(row["image"]["path"]).stem for rows in records.values() for row in rows}
    crc_groups = collections.defaultdict(list)
    for entry in image_index:
        crc_groups[(entry["size"], entry["crc32"])].append(entry["filename"])
    report = {
        "schema_version": 1,
        "provenance": {
            "hf_repository": HF_REPO,
            "hf_revision": HF_REVISION,
            "hf_url": f"https://huggingface.co/datasets/{HF_REPO}/tree/{HF_REVISION}",
            "zenodo_doi": "10.5281/zenodo.5092309",
            "zenodo_version": "1.0",
            "zenodo_declared_archive_md5_not_fully_verified": ZENODO_MD5,
            "license": "CC-BY-4.0",
            "csv_sha256": source_hashes,
            "hf_parquet_files": [
                {
                    "path": entry["path"],
                    "size": entry["size"],
                    "declared_lfs_sha256_not_fully_verified": entry.get("lfs", {}).get("oid"),
                }
                for entry in json.loads((source / "hf-tree.json").read_text())
                if entry["path"].endswith(".parquet")
            ],
        },
        "splits": summaries,
        "cross_split_overlaps": cross_split,
        "archive_unreferenced_images": [
            entry["filename"]
            for entry in image_index
            if Path(entry["filename"]).stem not in all_ids
        ],
        "archive_same_size_crc32_duplicate_candidates": [
            names for names in crc_groups.values() if len(names) > 1
        ],
        "pixel_hash_audit_performed": False,
        "images_visually_inspected": 0,
        "annotation_modifications": 0,
        "limitations": [
            "No complete image bytes or decoded pixels audited; filenames are not pixel hashes.",
            "CRC32/size equality identifies candidates, not confirmed image duplicates.",
            "Out-of-frame boxes are preserved and are not automatically label errors.",
            "All-image dimensions still require header checks before conversion.",
            "Official 2021 evaluator matching semantics are not verified.",
        ],
        "conversion": {
            "hf_boxes": "XYWH",
            "original_csv_boxes": "XYXY",
            "hf_category_id": 0,
            "proposed_coco_category_id": 1,
            "preserve_repeated_and_out_of_bounds_boxes": True,
            "preserve_empty_images": True,
            "preserve_native_split_and_domain": True,
        },
        "metric": {
            "name": "Average Domain Accuracy",
            "formula": "mean_domain(mean_image(TP/(TP+FP+FN)))",
            "iou_threshold_documented": ">0.5",
            "empty_gt_empty_pred": 1,
            "empty_gt_nonempty_pred": 0,
            "nonempty_gt_empty_pred": 0,
            "official_2021_assignment_algorithm_verified": False,
            "do_not_claim_hungarian_or_wilds_matcher_is_exact_official_evaluator": True,
        },
    }
    dimensions_file = source / "suspect-original-dimensions.json"
    if dimensions_file.exists():
        evidence = json.loads(dimensions_file.read_text())
        report["verified_original_header_dimensions"] = evidence["dimensions"]
    if (source / "wilds-globalwheat_dataset.py").exists():
        report["wilds_comparison"] = audit_wilds(source, records)
    storage_file = source / "hf-storage-metadata.json"
    if storage_file.exists():
        report["hf_storage_metadata"] = json.loads(storage_file.read_text())
    return report


def audit_wilds(source: Path, hf_records: dict[str, list[dict]]) -> dict:
    """Audit available native WILDS CSVs without substituting competition split IDs."""
    code_path = source / "wilds-globalwheat_dataset.py"
    if digest(code_path.read_bytes()) != WILDS_SOURCE_HASHES["GlobalWheatDataset"]:
        raise ValueError("Unexpected WILDS dataset source checksum")
    sessions = next(
        ast.literal_eval(statement.value)
        for statement in ast.parse(code_path.read_text()).body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SESSIONS" for target in statement.targets
        )
    )
    hf_by_id = {
        Path(row["image"]["path"]).stem: (split, row)
        for split, rows in hf_records.items()
        for row in rows
    }
    splits, split_ids = {}, {}
    for split, name in WILDS_FILES.items():
        path = source / f"wilds-{name}.response"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        names = [Path(row["image_name"]).stem for row in rows]
        split_ids[split] = set(names)
        differences, original_split_counts, counts = [], collections.Counter(), []
        domains = collections.Counter()
        for row, image_id in zip(rows, names, strict=True):
            boxes = csv_xyxy_to_xywh(row["BoxesString"])
            counts.append(len(boxes))
            domain = sessions[int(row["domain"])]
            domains[domain] += 1
            if image_id not in hf_by_id:
                differences.append({"image_id": image_id, "reason": "not_in_hf"})
                continue
            original_split, hf_row = hf_by_id[image_id]
            original_split_counts[original_split] += 1
            if boxes != hf_row["objects"]["boxes"] or domain != hf_row["domain"]:
                differences.append(
                    {
                        "image_id": image_id,
                        "wilds_domain": domain,
                        "hf_domain": hf_row["domain"],
                        "wilds_boxes": len(boxes),
                        "hf_boxes": len(hf_row["objects"]["boxes"]),
                        "coordinates_equal": boxes == hf_row["objects"]["boxes"],
                    }
                )
        splits[split] = {
            "csv_sha256": digest(path.read_bytes()),
            "csv_rows": len(rows),
            "unique_image_ids": len(set(names)),
            "duplicate_ids": {
                image_id: count
                for image_id, count in collections.Counter(names).items()
                if count > 1
            },
            "domains": dict(sorted(domains.items())),
            "hf_split_membership_by_row": dict(original_split_counts),
            "annotation_or_domain_differences_from_hf": differences,
            "boxes": sum(counts),
            "empty_rows": counts.count(0),
            "maximum_boxes_per_row": max(counts),
            "rows_exceeding_query_cap": {
                str(cap): sum(count > cap for count in counts) for cap in (100, 128, 200, 300, 500)
            },
        }
    return {
        "dataset_version": "1.1",
        "library_commit": WILDS_COMMIT,
        "bundle_uuid": "0x36e16907b7254571b708b725f8beae52",
        "split_scheme": "official",
        "is_same_split_as_hf_competition_release": False,
        "missing_native_csvs": [name for split, name in WILDS_FILES.items() if split not in splits],
        "native_split_audit_complete": len(splits) == len(WILDS_FILES),
        "splits": splits,
        "available_split_image_id_overlaps": {
            f"{first}:{second}": sorted(split_ids[first] & split_ids[second])
            for first, second in itertools.combinations(splits, 2)
        },
        "metadata_transport_caveat": (
            "CodaLab certificate verification failed (expired certificate). These public, "
            "unauthenticated CSV probes disabled certificate verification. No credentials or "
            "images were transmitted; hashes pin the received bytes but do not authenticate them. "
            "Revalidate via a trusted transport before freezing a native WILDS experiment."
        ),
    }


def verify_wilds_metric() -> dict:
    """Execute the unchanged official library on synthetic boxes, never images."""
    import inspect
    from importlib.metadata import version

    import torch
    import torchvision
    import wilds
    from wilds.common.metrics.all_metrics import DetectionAccuracy
    from wilds.datasets.globalwheat_dataset import SESSIONS, GlobalWheatDataset

    hashes = {
        cls.__name__: digest(Path(inspect.getfile(cls)).read_bytes())
        for cls in (DetectionAccuracy, GlobalWheatDataset)
    }
    if hashes != WILDS_SOURCE_HASHES or wilds.__version__ != "2.0.0":
        raise ValueError("Official metric verification requires the pinned WILDS 2.0.0 source")
    cases = [
        (
            "ambiguous_overlap",
            [[0, 0, 10, 10], [3, 0, 13, 10]],
            [[1, 0, 11, 10], [0, 0, 10, 10]],
            [1.0, 1.0],
            1 / 3,
        ),
        ("exact_iou_half", [[0, 0, 3, 3]], [[1, 0, 4, 3]], [1.0], 1.0),
        ("score_exact_half", [[0, 0, 3, 3]], [[0, 0, 3, 3]], [0.5], 0.0),
        ("score_above_half", [[0, 0, 3, 3]], [[0, 0, 3, 3]], [0.5001], 1.0),
        ("duplicate_predictions", [[0, 0, 3, 3]], [[0, 0, 3, 3], [0, 0, 3, 3]], [1.0, 1.0], 0.5),
        (
            "repeated_native_gt",
            [[0, 0, 3, 3], [0, 0, 3, 3]],
            [[0, 0, 3, 3], [0, 0, 3, 3]],
            [1.0, 1.0],
            1 / 3,
        ),
        ("out_of_frame_native_gt", [[1000, 0, 1040, 40]], [[1000, 0, 1040, 40]], [1.0], 1.0),
        ("empty_both", [], [], [], 1.0),
        ("empty_gt", [], [[0, 0, 3, 3]], [1.0], 0.0),
        ("empty_pred", [[0, 0, 3, 3]], [], [], 0.0),
    ]
    scores = {}
    metric = DetectionAccuracy()
    for name, truth, predictions, confidence, expected in cases:
        actual = float(
            metric._compute_element_wise(
                [
                    {
                        "boxes": torch.tensor(predictions, dtype=torch.float32).reshape(-1, 4),
                        "scores": torch.tensor(confidence, dtype=torch.float32),
                    }
                ],
                [{"boxes": torch.tensor(truth, dtype=torch.float32).reshape(-1, 4)}],
            )[0]
        )
        if not math.isclose(actual, expected, abs_tol=1e-7):
            raise AssertionError(f"Official metric disagrees for synthetic case {name}")
        scores[name] = {"actual": actual, "expected": expected}

    with tempfile.TemporaryDirectory(prefix="wheat-synthetic-") as temporary:
        folder = Path(temporary) / "globalwheat_v1.1"
        folder.mkdir()
        (folder / "RELEASE_v1.1.txt").write_text("Synthetic fixture, no benchmark images.\n")
        with (folder / "metadata_domain.csv").open("w") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["name", "country", "location", "development_stage"])
            writer.writerows([[session, "UK", "Rothamsted", "Filling"] for session in SESSIONS])
        for filename in WILDS_FILES.values():
            with (folder / filename).open("w") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_name", "BoxesString", "domain"])
                if filename == "official_val.csv":
                    writer.writerows(
                        [[f"{index}.png", "0 0 3 3", 0 if index < 3 else 1] for index in range(4)]
                    )
                else:
                    writer.writerow([filename + ".png", "0 0 3 3", 0])
        dataset = GlobalWheatDataset(
            version="1.1", root_dir=temporary, download=False, split_scheme="official"
        )
        subset = dataset.get_subset("val")
        # Native labels are a list; avoid indexing that list with a numpy array.
        truth = [dataset.y_array[int(index)] for index in subset.indices]
        predictions = [
            {
                "boxes": row["boxes"].clone() if index < 3 else torch.empty(0, 4),
                "scores": torch.ones(1) if index < 3 else torch.empty(0),
            }
            for index, row in enumerate(truth)
        ]
        result, _ = subset.eval(predictions, truth, subset.metadata_array)
        domain_average = float(result["detection_acc_avg_dom"])
        image_average = float(result["detection_acc_avg"])
        if not math.isclose(domain_average, 0.5) or not math.isclose(image_average, 0.75):
            raise AssertionError("Native WILDS equal-domain averaging changed")
    return {
        "library": "wilds",
        "version": wilds.__version__,
        "source_file_sha256": hashes,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_scatter": version("torch-scatter"),
        "synthetic_cases": scores,
        "native_dataset_eval_synthetic": {
            "domain_image_counts": [3, 1],
            "equal_domain_average": domain_average,
            "image_average": image_average,
            "native_constructor_used": True,
        },
        "image_loads": 0,
        "model_inference_calls": 0,
        "competition_2021_equivalence_claimed": False,
    }


def build_hf_protocol(audit: dict) -> dict:
    """Produce a reviewable protocol only after real metadata and metric verification."""
    verification = audit.get("wilds_official_metric_verification", {})
    grouping = verification.get("native_dataset_eval_synthetic", {})
    if (
        verification.get("source_file_sha256") != WILDS_SOURCE_HASHES
        or grouping.get("equal_domain_average") != 0.5
        or grouping.get("image_average") != 0.75
    ):
        raise ValueError("Verify the unchanged official per-image and grouped metric first")
    if audit["provenance"]["hf_revision"] != HF_REVISION:
        raise ValueError("Unexpected author dataset revision")
    for split, expected in {"train": 3655, "validation": 1476, "test": 1381}.items():
        summary = audit["splits"][split]
        if (
            summary["images"] != expected
            or summary["exact_original_annotation_matches"] != expected
            or summary["annotation_mismatch_ids"]
            or summary["domain_mismatch_ids"]
        ):
            raise ValueError("Native author annotations are not fully verified")
    if any(
        value["image_id"] or value["domain"] for value in audit["cross_split_overlaps"].values()
    ):
        raise ValueError("Cross-split image ID or domain overlap")
    protocol = {
        "schema_version": 1,
        "benchmark_id": "gwhd2021-hf-domain-ada-v1",
        "status": "reviewable_metadata_verified_images_not_downloaded",
        "definition": (
            "Author HF release with unchanged native splits and annotations, scored by the "
            "unchanged WILDS 2.0.0 equal-domain detection accuracy implementation."
        ),
        "competition_2021_leaderboard_equivalence_claimed": False,
        "wilds_native_dataset_equivalence_claimed": False,
        "original_image_decoded_pixel_equivalence_verified": False,
        "dataset": audit["provenance"],
        "splits": {
            split: {
                "role": "sealed_final_evaluation" if split == "test" else split,
                "images": values["images"],
                "boxes": values["boxes"],
                "empty_images": values["empty_images"],
                "domains": values["domains"],
                "metadata_sha256": values["metadata_sha256"],
            }
            for split, values in audit["splits"].items()
        },
        "metric": {
            "primary_key": "detection_acc_avg_dom",
            "library": "wilds==2.0.0",
            "library_commit": WILDS_COMMIT,
            "source_file_sha256": WILDS_SOURCE_HASHES,
            "entry_point": "wilds.datasets.globalwheat_dataset.GlobalWheatDataset.eval",
            "formula": "mean_domain(mean_image(TP/(TP+FP+FN)))",
            "prediction_score_test": ">0.5",
            "matching_iou_test": ">=0.5",
            "assignment": "torchvision Matcher best GT per prediction; unique matched GT counts TP",
            "empty_both": 1,
            "empty_only_one_side": 0,
            "coco_ap_role": "supplemental_only",
        },
        "data_contract": {
            "hf_boxes": "XYWH absolute pixels",
            "metric_boxes": "XYXY absolute pixels",
            "hf_category_id": 0,
            "coco_category_id": 1,
            "domain_mapping": "Map author domain strings through pinned WILDS SESSIONS list",
            "preserve_empty_images": True,
            "preserve_repeated_gt_boxes": True,
            "preserve_out_of_frame_gt_boxes": True,
            "do_not_substitute_wilds_native_rows": True,
            "model_without_confidence_scores": "Assign 1.0 to every retained prediction",
            "maximum_predictions_per_image": 500,
            "refinement_selection_cap": "Unselected base predictions must remain intact",
        },
        "selection_and_sealing": {
            "train": "Only native training images and labels may update model parameters",
            "validation": "Model, prompt, postprocessing, and score calibration selection allowed",
            "test": (
                "Images and model outputs remain sealed until model, baseline prompts, "
                "postprocessing, calibration, and experiment selection are frozen"
            ),
            "test_metadata_access_so_far": (
                "Dataset provenance, annotation geometry, domain membership and density audit only"
            ),
            "training_gt_transform_policy": (
                "Any training-only clipping or augmentation must be explicit; "
                "native evaluation GT is never clipped, deduplicated or dropped"
            ),
            "comparisons": "Same complete native test split and exact evaluator for all models",
            "reporting": "Report per-domain scores and equal-domain primary score",
        },
        "remaining_before_experiment": [
            "Full image download deferred by root while local Qwen download has priority",
            "Verify shard checksums and all decoded image dimensions",
            "Audit cross-split complete-file and decoded-pixel duplicates before model inference",
            "Freeze any response to a newly discovered duplicate as a separately named protocol",
            "Record baseline model identities, prompts, confidence policy, and output truncation",
        ],
        "storage": audit.get("hf_storage_metadata"),
    }
    protocol["protocol_sha256_excluding_this_field"] = digest(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    )
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("runs/wheat-audit"))
    parser.add_argument("--download-metadata", action="store_true")
    parser.add_argument("--verify-wilds-metric", action="store_true")
    parser.add_argument("--protocol-output", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/wheat-audit/audit.json"))
    args = parser.parse_args()
    if args.download_metadata:
        fetch_metadata(args.metadata_dir)
    report = audit_directory(args.metadata_dir)
    if args.verify_wilds_metric:
        report["wilds_official_metric_verification"] = verify_wilds_metric()
    if args.protocol_output:
        protocol = build_hf_protocol(report)
        args.protocol_output.parent.mkdir(parents=True, exist_ok=True)
        args.protocol_output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": {split: values["images"] for split, values in report["splits"].items()},
                "full_images_downloaded": 0,
                "official_metric_assignment_verified": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
