"""Acquire only the frozen400 nativeTRAIN candidate JPEGs, pending author audit."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.request
from pathlib import Path

from extract_aircraft import atomic_json, check_space, file_hash, groups
from prepare_aircraft import audit, sha
from prepare_aircraft_manifests import write_jsonl
from prepare_aircraft_probe import BASE, PREFIX, REPO, REVISION

from vision_rl import aircraft_data

PLAN_SHA256 = "44be6969eb2582073f12f7a84594ee0bf1e3c211fa6ac82170a0efbf7302f423"
EXPECTED_IMAGE_BYTES = 110960896
TRANSFER_CAP = 200 * 1024**2
RESERVE_BYTES = 10 * 1024**3


def validate_plan(content, metadata_content):
    if sha(content) != PLAN_SHA256:
        raise ValueError("Four-shot selection differs from the prospectively frozen plan")
    plan = json.loads(content)
    source, rows = audit(metadata_content)
    native = {row["image_id"]: row for row in rows}
    if plan["canonical_labels"] != source["canonical_labels"]["variant"]:
        raise ValueError("Canonical class vocabulary changed")
    for row in plan["selected"]:
        if row["split"] != "train" or native[row["id"]]["split"] != "train":
            raise ValueError("Candidate acquisition is restricted to native TRAIN")
        if row["label"] != native[row["id"]]["variant"]:
            raise ValueError("Candidate labels differ from original author annotations")
    return plan


def validated_paths(plan):
    paths = [PREFIX + row["id"] + ".jpg" for row in plan["selected"]]
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/paths-info/{REVISION}",
        data=json.dumps({"paths": paths}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        records = json.loads(response.read(2 * 1024**2))
    if len(records) != 400 or {row["path"] for row in records} != set(paths):
        raise ValueError("Pinned mirror does not contain the exact frozen400 IDs")
    total = sum(row["size"] for row in records)
    if total != EXPECTED_IMAGE_BYTES or total > TRANSFER_CAP:
        raise ValueError("Selected mirror payload differs from the authorized byte budget")
    for row in records:
        if row["type"] != "file" or row["lfs"]["size"] != row["size"]:
            raise ValueError("Mirror metadata does not expose consistent immutable LFS objects")
    return {row["path"]: row for row in records}


def acquire_one(row, info, root, probe):
    remote = PREFIX + row["id"] + ".jpg"
    item = info[remote]
    expected_sha = item["lfs"]["oid"]
    destination = root / "development/images/train" / (row["id"] + ".jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prior = probe / "original-jpeg" / destination.name
    acquisition = "verified_existing_original"
    downloaded_bytes = 0
    if not destination.exists() and prior.exists():
        if prior.stat().st_size != item["size"] or file_hash(prior) != expected_sha:
            raise ValueError("Existing probe JPEG differs from this pinned mirror object")
        os.link(prior, destination)
        acquisition = "verified_probe_hardlink_no_payload_copy"
    if not destination.exists():
        check_space(root, EXPECTED_IMAGE_BYTES, RESERVE_BYTES)
        with urllib.request.urlopen(BASE + remote, timeout=60) as response:
            content = response.read(item["size"] + 1)
        if len(content) != item["size"] or sha(content) != expected_sha:
            raise ValueError("Received candidate JPEG does not match pinned mirror LFS SHA256")
        temporary = destination.with_suffix(".jpg.part")
        with temporary.open("xb") as handle:
            handle.write(content)
        temporary.replace(destination)
        downloaded_bytes = len(content)
        acquisition = "downloaded_pinned_mirror_original"
    if destination.stat().st_size != item["size"] or file_hash(destination) != expected_sha:
        raise ValueError("Retained candidate original differs from pinned mirror bytes")
    with aircraft_data.load_source_rgb(destination) as original:
        decoded = aircraft_data.rgb_pixel_sha256(original)
        source_size = list(original.size)
    with aircraft_data.load_aircraft_rgb(destination) as cropped:
        model_hash = aircraft_data.rgb_pixel_sha256(cropped)
        model_size = list(cropped.size)
    return {
        **row,
        "image_id": row["id"],
        "url": BASE + remote,
        "path": str(destination.relative_to(root)),
        "encoded_bytes": item["size"],
        "encoded_sha256": expected_sha,
        "decoded_rgb_sha256": decoded,
        "model_rgb_sha256": model_hash,
        "source_image_size": source_size,
        "model_image_size": model_size,
        "acquisition": acquisition,
        "downloaded_bytes": downloaded_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/fgvc-aircraft-2013b"))
    parser.add_argument(
        "--plan", type=Path, default=Path("docs/fgvc-aircraft-fourshot-plan-v1.json")
    )
    parser.add_argument(
        "--metadata-archive",
        type=Path,
        default=Path("runs/aircraft-audit/reproduced/annotations.tar.gz"),
    )
    parser.add_argument("--probe", type=Path, default=Path("runs/aircraft-shared-probe16/prepared"))
    args = parser.parse_args()
    plan = validate_plan(args.plan.read_bytes(), args.metadata_archive.read_bytes())
    args.root.mkdir(parents=True, exist_ok=True)
    check_space(args.root, EXPECTED_IMAGE_BYTES, RESERVE_BYTES)
    info = validated_paths(plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(acquire_one, row, info, args.root, args.probe) for row in plan["selected"]
        ]
        records = [future.result() for future in futures]
    duplicate_report = {
        field: groups(records, field)
        for field in ("encoded_sha256", "decoded_rgb_sha256", "model_rgb_sha256")
    }
    if any(value["all"] for value in duplicate_report.values()):
        atomic_json(args.root / "audit/fourshot-duplicate-failure.json", duplicate_report)
        raise ValueError(
            "Frozen400 TRAIN candidate contains exact duplicate images; no silent purge"
        )
    output = args.root / "candidates/fourshot-seed791"
    output.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "id": row["id"],
            "split": "train",
            "label": row["label"],
            "image": os.path.relpath(args.root / row["path"], output),
        }
        for row in records
    ]
    write_jsonl(output / "train400.jsonl", manifest)
    spec = {
        "schema_version": 1,
        "benchmark": (
            "FGVC-Aircraft nativeTRAIN four-shot candidate,mirror image equivalence pending"
        ),
        "question": "Identify the exact aircraft variant shown in this image.",
        "labels": plan["canonical_labels"],
        "split_policy": (
            "Frozen4 nativeTRAIN images/class,100classes,seed791; no validation/test images here. "
            "Original-author JPEG equivalence and full cross-split integrity audit remain pending."
        ),
        "preprocessing": "Original JPEG decode RGB then remove exactly bottom20 rows lazily",
        "preprocessing_id": aircraft_data.PREPROCESSING_ID,
        "sources": [{"url": plan["source"]["url"], "sha256": plan["source"]["sha256"]}]
        + [{"url": row["url"], "sha256": row["encoded_sha256"]} for row in records],
    }
    if (output / "data_spec.json").exists():
        if json.loads((output / "data_spec.json").read_text()) != spec:
            raise ValueError("Existing candidate data specification changed")
    atomic_json(output / "data_spec.json", spec)
    evidence = {
        "status": "candidate_prepared_author_equivalence_and_cross_split_audit_pending",
        "selection_sha256": PLAN_SHA256,
        "mirror_repo": REPO,
        "mirror_revision": REVISION,
        "images": records,
        "image_count": len(records),
        "classes": len({row["label"] for row in records}),
        "source_payload_bytes": sum(row["encoded_bytes"] for row in records),
        "downloaded_image_bytes_this_invocation": sum(row["downloaded_bytes"] for row in records),
        "hardlinked_existing_probe_images": sum(
            "hardlink" in row["acquisition"] for row in records
        ),
        "preprocessing_id": aircraft_data.PREPROCESSING_ID,
        "preprocessing_source_sha256": file_hash(Path(aircraft_data.__file__)),
        "duplicates_within_candidate": duplicate_report,
        "author_original_image_byte_equivalence_verified": False,
        "full_dataset_cross_split_pixel_audit_complete": False,
        "test_images_accessed": 0,
        "model_calls": 0,
        "manual_image_views": 0,
        "persistent_transformed_image_copies": 0,
        "image_terms": plan["image_terms"],
        "train_manifest_sha256": file_hash(output / "train400.jsonl"),
        "data_spec_sha256": file_hash(output / "data_spec.json"),
    }
    # Retain the first acquisition receipt when a later invocation reuses its files.
    receipt = output / "provenance.json"
    atomic_json(receipt if not receipt.exists() else output / "reuse-verification.json", evidence)
    print(
        json.dumps(
            {
                "images": len(records),
                "bytes": evidence["source_payload_bytes"],
                "downloaded_bytes": evidence["downloaded_image_bytes_this_invocation"],
                "reused_probe_images": evidence["hardlinked_existing_probe_images"],
                "output": str(output),
                "status": evidence["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
