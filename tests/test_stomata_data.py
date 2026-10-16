"""Protect the two-class/native-split contract of the stomata benchmark audit."""

import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

STOMATA = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "prepare_stomata.py"))


def fixture_source(root: Path) -> Path:
    for index, directory in enumerate(("train", "valid", "test")):
        folder = root / directory
        folder.mkdir(parents=True)
        name = f"{index}-1_20170828-141404_jpg"
        filename = f"{name}.rf.abcdef.png"
        Image.new("RGB", (48, 32), (40 * index, 20, 90)).save(folder / filename)
        coco = {
            "images": [
                {"id": 1, "file_name": filename, "width": 48, "height": 32, "extra": {"name": name}}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 10, 8]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 4, 9, 9]},
            ],
            "categories": [
                {"id": 0, "name": "dataset"},
                {"id": 1, "name": "close"},
                {"id": 2, "name": "open"},
            ],
        }
        (folder / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


def test_native_classes_and_common_source_frame_are_preserved(tmp_path):
    source = fixture_source(tmp_path / "source")
    rows, audit = STOMATA["inspect_source"](source)
    assert audit["groups"]["pixel_sha256"]["cross_split_groups"] == 0
    assert audit["groups"]["source_original_name"]["cross_split_groups"] == 0
    assert audit["groups"]["source_frame"]["cross_split_groups"] == 1
    assert audit["groups"]["capture_hour"]["cross_split_groups"] == 1
    assert rows["train"][0]["class_counts"] == {"close": 1, "open": 1}
    assert rows["test"][0]["category_ids"] == [1, 2]
    assert rows["test"][0]["category_names"] == ["close", "open"]
    assert audit["session_generalization"]["claim_supported"] is False


def test_no_silent_class_taxonomy_remapping(tmp_path):
    source = fixture_source(tmp_path / "source")
    path = source / "train" / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    coco["categories"][1]["name"] = "closed"
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="exactly open and close"):
        STOMATA["inspect_source"](source)


def test_prepare_seals_test_and_preserves_heldout_annotation_bytes(tmp_path):
    source = fixture_source(tmp_path / "source")
    output, clean = tmp_path / "prepared", tmp_path / "clean"
    report = STOMATA["prepare"](source, output, clean, tmp_path / "review")
    assert report["train_purge"]["removed_train_count"] == 0
    assert (output / "sealed" / "test.jsonl").is_file()
    assert not (output / "development" / "test.jsonl").exists()
    assert not (tmp_path / "review" / "test-12.png").exists()
    for directory in ("valid", "test"):
        assert (clean / directory / "_annotations.coco.json").read_bytes() == (
            source / directory / "_annotations.coco.json"
        ).read_bytes()


def test_acquisition_groups_do_not_invent_unknown_metadata():
    assert STOMATA["acquisition_groups"]("2-1_20170828-141404_jpg") == {
        "source_frame": "20170828-141404",
        "capture_hour": "20170828-14",
        "tile_prefix": "2-1",
    }
    assert STOMATA["acquisition_groups"]("20170828-141404_jpg")["tile_prefix"] is None
    assert STOMATA["acquisition_groups"]("unknown.jpg")["source_frame"] is None


def test_visual_test_review_requires_separate_explicit_workflow(tmp_path):
    with pytest.raises(ValueError, match="Test contact sheets"):
        STOMATA["contact_sheet"]([{"split": "test"}], tmp_path / "test.png")


def test_cross_scale_source_frame_leakage_is_explicit():
    full = {"source_frame": "frame-a", "tile_prefix": None}
    crop = {"source_frame": "frame-a", "tile_prefix": "1-0"}
    rows = {"train": [full, crop], "validation": [crop], "test": [full, crop]}
    result = STOMATA["cross_scale_overlap"](rows)
    assert result["test"]["crops_with_same_timestamp_fullframe_in_train"] == 1
    assert result["test"]["fullframes_with_same_timestamp_crops_in_train"] == 1
    assert result["validation"]["crops_with_same_timestamp_fullframe_in_train"] == 1


def grouped_fixture_source(root: Path) -> Path:
    """Ten frames, each with one distinct view in every native split."""
    for native_index, directory in enumerate(("train", "valid", "test")):
        folder = root / directory
        folder.mkdir(parents=True)
        images, annotations = [], []
        for group in range(10):
            name = f"{native_index}-0_20170828-14{group:02d}00_jpg"
            filename = f"{name}.rf.abcdef.png"
            Image.new("RGB", (48, 32), (group * 20, native_index * 30, 90)).save(folder / filename)
            images.append(
                {
                    "id": group,
                    "file_name": filename,
                    "width": 48,
                    "height": 32,
                    "extra": {"name": name},
                }
            )
            annotations.extend(
                [
                    {"id": group * 2, "image_id": group, "category_id": 1, "bbox": [2, 3, 10, 8]},
                    {
                        "id": group * 2 + 1,
                        "image_id": group,
                        "category_id": 2,
                        "bbox": [20, 4, 9, 9],
                    },
                ]
            )
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 0, "name": "dataset"},
                {"id": 1, "name": "close"},
                {"id": 2, "name": "open"},
            ],
        }
        (folder / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


def test_group_holdout_preserves_content_and_provenance_without_cross_split_frames(tmp_path):
    source = grouped_fixture_source(tmp_path / "source")
    native_bytes = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "grouped"
    report = STOMATA["prepare_group_heldout"](source, output)
    assert report["protocol"]["official_native_score"] is False
    assert report["protocol"]["seed"] == 2718
    assert report["groups"]["source_frame"]["cross_split_groups"] == 0
    assert report["groups"]["pixel_sha256"]["cross_split_groups"] == 0
    assert {split: row["images"] for split, row in report["splits"].items()} == {
        "train": 21,
        "validation": 6,
        "test": 3,
    }
    image_ids, annotation_ids = [], []
    for directory in ("train", "valid", "test"):
        coco = json.loads((output / directory / "_annotations.coco.json").read_text())
        image_ids.extend(image["id"] for image in coco["images"])
        annotation_ids.extend(annotation["id"] for annotation in coco["annotations"])
        for image in coco["images"]:
            provenance = image["extra"]["derived_provenance"]
            old_directory = {"validation": "valid"}.get(
                provenance["original_split"], provenance["original_split"]
            )
            assert (output / directory / image["file_name"]).read_bytes() == (
                source / old_directory / provenance["original_file_name"]
            ).read_bytes()
        assert {annotation["category_id"] for annotation in coco["annotations"]} == {1, 2}
    assert len(image_ids) == len(set(image_ids)) == 30
    assert len(annotation_ids) == len(set(annotation_ids)) == 60
    assert all(path.read_bytes() == content for path, content in native_bytes.items())
    assert not (output / "development" / "test.jsonl").exists()
    assert (output / "sealed" / "test.jsonl").is_file()


def test_group_holdout_is_reproducible_and_independent_of_output_location(tmp_path):
    source = grouped_fixture_source(tmp_path / "source")
    first = STOMATA["prepare_group_heldout"](source, tmp_path / "first")
    second = STOMATA["prepare_group_heldout"](source, tmp_path / "second")
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert first["splits"] == second["splits"]
    for directory in ("train", "valid", "test"):
        assert (tmp_path / "first" / directory / "_annotations.coco.json").read_bytes() == (
            tmp_path / "second" / directory / "_annotations.coco.json"
        ).read_bytes()


def test_group_allocation_rejects_unknown_and_malformed_timestamps():
    with pytest.raises(ValueError, match="known, valid"):
        STOMATA["allocate_groups"]([{"source_frame": None}])
    with pytest.raises(ValueError, match="invalid source-frame timestamp"):
        STOMATA["allocate_groups"]([{"source_frame": "20179999-250099"}])
    assert STOMATA["acquisition_groups"]("1-0_20179999-250099_jpg")["source_frame"] is None


def test_largest_remainder_allocation_has_fixed_tie_break_and_ignores_input_order():
    from datetime import datetime, timedelta

    rows = [
        {
            "source_frame": (datetime(2017, 8, 28) + timedelta(seconds=index)).strftime(
                "%Y%m%d-%H%M%S"
            )
        }
        for index in range(117)
    ]
    groups = STOMATA["allocate_groups"](rows)
    assert {split: len(items) for split, items in groups.items()} == {
        "train": 82,
        "validation": 18,
        "test": 17,
    }
    assert groups == STOMATA["allocate_groups"](list(reversed(rows)))
