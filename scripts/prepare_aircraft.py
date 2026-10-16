"""Bounded author FGVC-Aircraft metadata audit; never download or open images.

The current author annotation archive differs from the site's declared MD5.
Pin its observed HTTPS SHA-256 explicitly and report the discrepancy. Generated
metadata/few-shot plans are not image-ready training manifests.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import itertools
import json
import re
import tarfile
import urllib.request
from pathlib import Path

SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.tar.gz"
SOURCE_SHA256 = "50e4419bd330b68a2e85dbcb32339d31120190efe6944682e87c9b431f5f202a"
SOURCE_SIZE = 407927
DECLARED_MD5 = "e3799900175f63abbc48a5bc5b16427e"
EVALUATOR_SHA256 = "4e25f8db8a0d0163b10b167299c04c7a437a5f0de637f1e79628066ec297676a"
SPLITS = ("train", "val", "test")
LEVELS = {"variant": "variants", "family": "families", "manufacturer": "manufacturers"}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_archive(content):
    if len(content) != SOURCE_SIZE or sha(content) != SOURCE_SHA256:
        raise ValueError("Author annotation archive does not match the explicitly pinned SHA256")
    files, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "fgvc-aircraft-2013b":
                raise ValueError("Unexpected annotation archive path")
            if member.isdir():
                continue
            if not member.isfile() or path.suffix not in (".txt", ".m", ".md", ".html"):
                raise ValueError("Metadata archive contains a link, image or unexpected file")
            total += member.size
            if member.size > 2 * 1024**2 or total > 8 * 1024**2:
                raise ValueError("Annotation archive expansion exceeds metadata budget")
            relative = str(Path(*path.parts[1:]))
            if relative in files:
                raise ValueError("Duplicate archive member")
            files[relative] = archive.extractfile(member).read()
    if sha(files["evaluation.m"]) != EVALUATOR_SHA256:
        raise ValueError("Official evaluator source differs from pinned source")
    return files


def download_metadata(destination):
    if destination.exists():
        content = destination.read_bytes()
        read_archive(content)
        return content
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        content = response.read(SOURCE_SIZE + 1)
    read_archive(content)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return content


def parse_labeled_rows(content, labels):
    result = []
    seen = set()
    for line in content.decode().splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9]{7}", parts[0]):
            raise ValueError("Malformed official image ID or label row")
        image_id, label = parts
        if image_id in seen or label not in labels:
            raise ValueError("Duplicate image ID or unknown canonical label")
        seen.add(image_id)
        result.append((image_id, label))
    return result


def select_fewshot(rows, labels, shots=16, seed=2718):
    """Order-independent seeded SHA ranking, native train only; no validation fitting."""
    if type(shots) is not int or shots <= 0 or type(seed) is not int:
        raise ValueError("shots must be positive and seed must be an integer")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Few-shot selection accepts only native training rows")
    by_class = collections.defaultdict(list)
    ids = set()
    for row in rows:
        if row["image_id"] in ids or row["variant"] not in labels:
            raise ValueError("Duplicate training ID or unknown class")
        ids.add(row["image_id"])
        by_class[row["variant"]].append(row)
    selected = []
    for label in labels:
        if len(by_class[label]) < shots:
            raise ValueError("Not enough native training examples in every class")

        def ranking(row):
            return (
                sha(canonical(["fgvc-aircraft-2013b", seed, label, row["image_id"]])),
                row["image_id"],
            )

        selected.extend(sorted(by_class[label], key=ranking)[:shots])
    return {
        "status": "prospective_metadata_only_not_image_ready",
        "native_split": "train",
        "shots_per_class": shots,
        "seed": seed,
        "algorithm": "ascending SHA256 of compact JSON [release,seed,variant,image_id]",
        "images": len(selected),
        "classes": len(labels),
        "selected_image_ids": [row["image_id"] for row in selected],
        "selected_rows_sha256": sha(canonical(selected)),
        "uses_validation_or_test": False,
    }


def audit(content):
    files = read_archive(content)
    labels = {
        level: files[f"data/{plural}.txt"].decode().splitlines() for level, plural in LEVELS.items()
    }
    if any(values != sorted(set(values)) for values in labels.values()):
        raise ValueError("Canonical classes differ from sorted unique MATLAB class order")
    boxes = {}
    for line in files["data/images_box.txt"].decode().splitlines():
        image_id, *coordinates = line.split()
        if not re.fullmatch(r"[0-9]{7}", image_id) or len(coordinates) != 4 or image_id in boxes:
            raise ValueError("Invalid bounding-box metadata row")
        box = list(map(int, coordinates))
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("Inverted native bounding box")
        boxes[image_id] = box
    rows, summaries, split_ids = [], {}, {}
    hierarchy = {level: collections.defaultdict(set) for level in ("family", "manufacturer")}
    for split in SPLITS:
        ids = files[f"data/images_{split}.txt"].decode().splitlines()
        annotations = {
            level: parse_labeled_rows(files[f"data/images_{level}_{split}.txt"], names)
            for level, names in labels.items()
        }
        if len(ids) != len(set(ids)) or any(
            [i for i, _ in pairs] != ids for pairs in annotations.values()
        ):
            raise ValueError("Native image lists and hierarchical annotations do not align")
        split_ids[split] = set(ids)
        counts = {
            level: collections.Counter(label for _, label in pairs)
            for level, pairs in annotations.items()
        }
        split_rows = []
        for index, image_id in enumerate(ids):
            variant = annotations["variant"][index][1]
            row = {
                "image_id": image_id,
                "split": split,
                "variant": variant,
                "class_id": labels["variant"].index(variant),
                "family": annotations["family"][index][1],
                "manufacturer": annotations["manufacturer"][index][1],
                "source_bbox_xyxy": boxes[image_id],
                "source_image_size": None,
            }
            split_rows.append(row)
            for level in hierarchy:
                hierarchy[level][variant].add(row[level])
        rows.extend(split_rows)
        summaries[split] = {
            "images": len(ids),
            "unique_image_ids": len(set(ids)),
            "classes": {level: len(values) for level, values in counts.items()},
            "class_counts": {
                level: dict(sorted(values.items())) for level, values in counts.items()
            },
            "metadata_rows_sha256": sha(canonical(split_rows)),
            "variant_count_frequency": dict(
                sorted(collections.Counter(counts["variant"].values()).items())
            ),
            "all_hierarchical_rows_match_official_image_order": True,
        }
    all_ids = set.union(*split_ids.values())
    if set(boxes) != all_ids:
        raise ValueError("Bounding-box IDs do not match all native split IDs")
    overlaps = {
        f"{a}:{b}": sorted(split_ids[a] & split_ids[b])
        for a, b in itertools.combinations(SPLITS, 2)
    }
    if any(overlaps.values()):
        raise ValueError("Official splits contain duplicate image IDs")
    trainval = {}
    for level, names in labels.items():
        combined = parse_labeled_rows(files[f"data/images_{level}_trainval.txt"], names)
        expected = {row["image_id"]: row[level] for row in rows if row["split"] != "test"}
        trainval[level] = dict(combined) == expected
    if not all(trainval.values()):
        raise ValueError("Native trainval differs from the union of train and validation")
    return {
        "schema_version": 1,
        "status": "metadata_verified_images_not_acquired",
        "release": "FGVC-Aircraft 2013b (current author annotations, 2014-01-17 HTTP mtime)",
        "source": {
            "url": SOURCE_URL,
            "bytes": len(content),
            "sha256": sha(content),
            "declared_md5": DECLARED_MD5,
            "observed_md5": hashlib.md5(content).hexdigest(),
            "declared_md5_matches": hashlib.md5(content).hexdigest() == DECLARED_MD5,
            "declared_md5_url": (
                "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
                "fgvc-aircraft-2013b-annotations.html"
            ),
            "transport": "HTTPS with standard certificate verification",
        },
        "full_image_archive_not_acquired": {
            "url": (
                "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
                "fgvc-aircraft-2013b.tar.gz"
            ),
            "declared_md5": "32eca553f897e747706144f3bb444df0",
            "checksum_verified": False,
        },
        "source_file_sha256": {name: sha(value) for name, value in sorted(files.items())},
        "website_claims": {"images": 10200, "variants": 102, "families": 70, "manufacturers": 41},
        "actual_counts": {"images": len(all_ids), **{LEVELS[k]: len(v) for k, v in labels.items()}},
        "canonical_labels": labels,
        "splits": summaries,
        "image_id_overlaps": overlaps,
        "native_trainval_is_train_union_validation": trainval,
        "hierarchy": {
            level: {variant: sorted(parents) for variant, parents in sorted(values.items())}
            for level, values in hierarchy.items()
        },
        "bounding_box_notes": {
            "documented_coordinates": "XYXY; author prose says top-left pixel is (1,1)",
            "rows_with_zero_coordinate": [image_id for image_id, box in boxes.items() if 0 in box],
            "inverted_boxes": 0,
            "source_boxes_modified": False,
            "image_dimensions_in_metadata": False,
            "do_not_use_as_an_unverified_crop_policy": True,
        },
        "required_image_preprocessing": (
            "Decode RGB and remove the bottom 20 pixel rows before any model sees the image; "
            "retain full remaining image, no GT-box crop"
        ),
        "image_terms": (
            "Non-commercial research only; photographers retain image copyright; "
            "other use requires their authorization"
        ),
        "evaluator_code_terms": "evaluation.m declares public domain; not an image license",
        "evaluator": {
            "entry_point": "evaluation(data_dir, 'variant_test', images, labels, scores)",
            "source_sha256": EVALUATOR_SHA256,
            "primary": "mean(diag(row_normalized_confusion))",
            "micro_accuracy": "supplemental; not identical because native class counts are 33 or34",
            "class_order": "MATLAB unique(labels0), sorted; matches variants.txt",
            "missing_predictions": "Unclassified and counted incorrect in the class denominator",
            "score_ties": "First class in sorted order; all -Inf chooses unclassified sentinel",
            "duplicate_image_class_pair": (
                "Indexed score assignment overwrites; prohibit duplicates in validated inputs"
            ),
            "unknown_image_bug": (
                "Source checks y==0 instead of x==0; reject unknown IDs before invoking"
            ),
            "unknown_label_bug": (
                "Source filters x/y but not scores; reject malformed labels "
                "rather than relying on this path"
            ),
            "matlab_or_octave_executed": False,
            "competition_2013_equivalence_claimed": False,
            "prospective_parity_cases": [
                {"case": "perfect two-class", "expected_mean_class_accuracy": 1.0},
                {"case": "all incorrect", "expected_mean_class_accuracy": 0.0},
                {
                    "case": "3 correct A images and 1 incorrect B image",
                    "expected_mean_class_accuracy": 0.5,
                    "expected_micro_accuracy": 0.75,
                },
                {"case": "one A correct and one B missing", "expected_mean_class_accuracy": 0.5},
                {"case": "no triplets", "expected_mean_class_accuracy": 0.0},
                {"case": "equal finite A/B scores", "expected_prediction": "A"},
                {"case": "all scores -Inf", "expected_prediction": "unclassified"},
                {
                    "case": "finite negative score for only correct class",
                    "expected_prediction": "correct class",
                },
                {
                    "case": "unknown ID or label or duplicate image-class pair",
                    "wrapper_policy": "reject malformed input before official evaluator",
                },
            ],
        },
        "prospective_train_fewshot": select_fewshot(
            [row for row in rows if row["split"] == "train"], labels["variant"]
        ),
        "trainer_contract": {
            "row_fields": {
                "id": "official seven-digit string",
                "split": "train or validation",
                "image": (
                    "relative path to original JPEG; apply the shared lazy RGB "
                    "bottom20-row removal exactly once"
                ),
                "label": "exact canonical variant string",
            },
            "metadata_sidecar_fields": [
                "source_bbox_xyxy",
                "source_image_size",
                "family",
                "manufacturer",
            ],
            "test_rows_excluded_from_trainer": True,
            "training_jsonl_emitted": False,
            "data_spec_required_fields": [
                "schema_version",
                "benchmark",
                "question",
                "labels",
                "split_policy",
                "preprocessing",
                "sources",
            ],
        },
        "limitations": [
            "The author's declared MD5 does not match the current served annotation archive; "
            "actual HTTPS bytes are separately SHA256 pinned.",
            "No images downloaded, viewed, decoded or used by a model; "
            "pixel duplicates and image dimensions are unverified.",
            "Metadata IDs are disjoint; acquisition sessions and near-duplicate leakage "
            "are not identified by these fields.",
            "No native MATLAB/Octave execution yet; evaluator behavior is source-audited only.",
            "The non-commercial research image terms do not establish permission "
            "for a commercial product or redistribution.",
        ],
        "benchmark_images_downloaded": 0,
        "model_inference_calls": 0,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("runs/aircraft-audit/fgvc-aircraft-2013b-annotations.tar.gz"),
    )
    parser.add_argument("--download-metadata", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/aircraft-audit/audit.json"))
    parser.add_argument("--metadata-rows-output", type=Path)
    args = parser.parse_args()
    content = (
        download_metadata(args.archive) if args.download_metadata else args.archive.read_bytes()
    )
    report, rows = audit(content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.metadata_rows_output:
        args.metadata_rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_rows_output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "counts": report["actual_counts"],
                "declared_md5_matches": report["source"]["declared_md5_matches"],
                "images_downloaded": 0,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
