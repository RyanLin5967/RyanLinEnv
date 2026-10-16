"""Reconcile completed author integrity evidence with immutable mirror candidates.

No downloads, image decoding, model calls or changes to candidate input files.
Retained original-byte hashes are verified by the existing completed-audit check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_aircraft import atomic_json, file_hash, groups, verify_completed
from prepare_aircraft import audit, canonical, sha

from vision_rl import aircraft_data


def compare_candidate(author_rows, candidate, expected_ids):
    authoritative = {row["image_id"]: row for row in author_rows}
    records = candidate["images"]
    if len(records) != len(expected_ids) or {row["id"] for row in records} != set(expected_ids):
        raise ValueError("Candidate IDs differ from the frozen metadata selection")
    fields = (
        "encoded_sha256",
        "decoded_rgb_sha256",
        "model_rgb_sha256",
        "source_image_size",
        "model_image_size",
    )
    matches = []
    for record in records:
        image_id = record["id"]
        source = authoritative[image_id]
        if source["split"] != "train" or record["split"] != "train":
            raise ValueError("Mirror candidate unexpectedly contains held-out membership")
        if source["variant"] != record["label"]:
            raise ValueError("Mirror candidate class differs from original author annotation")
        for field in fields:
            if source[field] != record[field]:
                raise ValueError(f"Author/mirror {field} differs for image {image_id}")
        matches.append({"image_id": image_id, **{field: source[field] for field in fields}})
    return {
        "images": len(matches),
        "all_encoded_bytes_match": True,
        "all_full_and_model_pixels_match": True,
        "all_native_labels_and_splits_match": True,
        "matched_records_sha256": sha(canonical(matches)),
    }


def validate_model_manifest(
    author_rows, candidate, plan, manifest, spec, labels, directory, image_directory
):
    """Validate model-visible membership/targets independently of frozen file hashes."""
    if spec.get("schema_version") != 1 or spec.get("labels") != labels:
        raise ValueError("Model specification canonical vocabulary differs from author metadata")
    if (
        spec.get("preprocessing_id") != aircraft_data.PREPROCESSING_ID
        or candidate.get("preprocessing_id") != aircraft_data.PREPROCESSING_ID
    ):
        raise ValueError("Model preprocessing differs from the shared original-JPEG contract")
    if plan.get("canonical_labels") != labels:
        raise ValueError("Frozen plan canonical vocabulary differs from author metadata")
    source = {row["image_id"]: row for row in author_rows}
    acquired = {row["id"]: row for row in candidate["images"]}
    selected = plan["selected"]
    expected_ids = [row["id"] for row in selected]
    if (
        len(set(expected_ids)) != len(expected_ids)
        or [row.get("id") for row in manifest] != expected_ids
        or [row["id"] for row in candidate["images"]] != expected_ids
    ):
        raise ValueError("Model manifest/provenance order differs from the frozen selection")
    for row, frozen in zip(manifest, selected, strict=True):
        if set(row) != {"id", "split", "image", "label"}:
            raise ValueError("Model manifest requires exactly id/split/image/label")
        image_id = row["id"]
        original, recorded = source[image_id], acquired[image_id]
        if original["split"] != "train" or any(
            item["split"] != "train" for item in (row, frozen, recorded)
        ):
            raise ValueError("Model manifest native split differs from author TRAIN membership")
        if any(item["label"] != original["variant"] for item in (row, frozen, recorded)):
            raise ValueError("Model manifest label differs from author/provenance/plan")
        if not isinstance(row["image"], str) or Path(row["image"]).is_absolute():
            raise ValueError("Model manifest must reference a relative original JPEG path")
        actual = (directory / row["image"]).resolve()
        expected = (image_directory / f"{image_id}.jpg").resolve()
        if actual != expected:
            raise ValueError("Model manifest image path differs from its declared original JPEG")
        if file_hash(actual) != recorded["encoded_sha256"]:
            raise ValueError("A model's retained original JPEG differs from its frozen hash")


def build_clearance(root, metadata_archive, probe, completed_report=None):
    """Recompute evidence; publisher compares it with the previously issued receipt."""
    root, metadata_archive, probe = (
        Path(root).resolve(),
        Path(metadata_archive).resolve(),
        Path(probe).resolve(),
    )
    report = completed_report
    if report is None:
        report = verify_completed(root, metadata_archive)
    if report is None:
        raise ValueError("Original author extraction has not completed; clearance remains pending")
    metadata, _ = audit(metadata_archive.read_bytes())
    labels = metadata["canonical_labels"]["variant"]
    rows, split_counts = [], {}
    for split, expected in (("train", 3334), ("val", 3333), ("test", 3333)):
        records = json.loads((root / f"audit/{split}-images.json").read_text())
        if len(records) != expected or len({row["variant"] for row in records}) != 100:
            raise ValueError("Complete native split count or class coverage changed")
        rows.extend(records)
        split_counts[split] = {"images": len(records), "classes": 100}
    duplicates = {
        field: groups(sorted(rows, key=lambda row: row["image_id"]), field)
        for field in ("encoded_sha256", "decoded_rgb_sha256", "model_rgb_sha256")
    }
    if duplicates != report["duplicates"]:
        raise ValueError("Stored duplicate report differs from the complete manifest records")
    if not report["training_integrity_clear"] or any(x["cross_split"] for x in duplicates.values()):
        raise ValueError("Cross-split exact duplicates block held-out clearance; no silent purge")
    if not report["annotation_and_evaluator_exact_matches"]:
        raise ValueError("Author annotations/evaluator differ from the pinned metadata")
    outputs = {}
    for name, directory, plan_path, manifest_name, image_directory in (
        (
            "probe16",
            probe,
            Path("docs/fgvc-aircraft-probe16-plan-v1.json"),
            "train.jsonl",
            probe / "original-jpeg",
        ),
        (
            "fourshot400",
            root / "candidates/fourshot-seed791",
            Path("docs/fgvc-aircraft-fourshot-plan-v1.json"),
            "train400.jsonl",
            root / "development/images/train",
        ),
    ):
        provenance = directory / "provenance.json"
        candidate = json.loads(provenance.read_text())
        plan = json.loads(plan_path.read_text())
        if (
            file_hash(directory / manifest_name) != candidate["train_manifest_sha256"]
            or file_hash(directory / "data_spec.json") != candidate["data_spec_sha256"]
            or file_hash(plan_path) != candidate["selection_sha256"]
            or file_hash(Path(aircraft_data.__file__)) != candidate["preprocessing_source_sha256"]
        ):
            raise ValueError(
                "Frozen model input manifest, specification or shared transform changed"
            )
        manifest_rows = [
            json.loads(line) for line in (directory / manifest_name).read_text().splitlines()
        ]
        validate_model_manifest(
            rows,
            candidate,
            plan,
            manifest_rows,
            json.loads((directory / "data_spec.json").read_text()),
            labels,
            directory,
            image_directory,
        )
        outputs[name] = {
            **compare_candidate(rows, candidate, [row["id"] for row in plan["selected"]]),
            "candidate_provenance_sha256": file_hash(provenance),
            "plan_sha256": file_hash(plan_path),
            "manifest_sha256": file_hash(directory / manifest_name),
            "data_spec_sha256": file_hash(directory / "data_spec.json"),
            "candidate_input_files_modified": False,
        }
    evidence = {
        "schema_version": 1,
        "verifier_source_sha256": file_hash(Path(__file__)),
        "preprocessing_source_sha256": file_hash(Path(aircraft_data.__file__)),
        "metadata_archive_sha256": file_hash(metadata_archive),
        "status": "author_bytes_and_exact_cross_split_integrity_verified",
        "author_archive": report["source"],
        "image_audit_sha256": file_hash(root / "audit/image-audit.json"),
        "native_splits": split_counts,
        "candidate_equivalence": outputs,
        "exact_cross_split_duplicate_groups": {
            field: len(x["cross_split"]) for field, x in duplicates.items()
        },
        "near_duplicates_or_pretraining_contamination_excluded": False,
        "test_access": "Retained encoded-file hashes and automated audit metadata only",
        "test_model_or_manual_view": False,
        "test_native_manifest": str(root / "audit/test-images.json"),
        "test_original_image_directory": str(root / "sealed/images/test"),
        "validation_manifest_next_step": (
            "Run prepare_aircraft_manifests.py after this check passes"
        ),
    }
    return evidence


def require_clearance(root, metadata_archive, probe, completed_report=None):
    """Require a current receipt and independently reconstruct its semantic evidence."""
    receipt_path = root / "audit/author-equivalence-v1.json"
    if not receipt_path.is_file():
        raise ValueError("Author-equivalence receipt required before native validation publication")
    receipt = json.loads(receipt_path.read_text())
    current = build_clearance(root, metadata_archive, probe, completed_report)
    if receipt != current:
        raise ValueError("Author-equivalence receipt is stale or not bound to current inputs")
    return receipt


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
    evidence = build_clearance(args.root, args.metadata_archive, args.probe)
    output = args.root / "audit/author-equivalence-v1.json"
    if output.exists() and json.loads(output.read_text()) != evidence:
        raise ValueError("Existing author-equivalence evidence changed; refusing replacement")
    atomic_json(output, evidence)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "output": str(output),
                "splits": evidence["native_splits"],
            }
        )
    )


if __name__ == "__main__":
    main()
