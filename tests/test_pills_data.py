"""Check that benchmark preparation preserves heldout data and rejects invalid labels."""

import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

PREPARE = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "prepare_pills.py"))


def source_fixture(root: Path, *, repeated_pixels: bool = False) -> Path:
    for index, dirname in enumerate(("train", "valid", "test")):
        directory = root / dirname
        directory.mkdir(parents=True)
        name = f"WIN_20240621_10_26_0{index}_Pro_jpg.rf.{index}.png"
        image = Image.new("RGB", (32, 24), (index * 60, 40, 20))
        if repeated_pixels and dirname in ("train", "test"):
            image = Image.new("RGB", (32, 24), (0, 40, 20))
        image.save(directory / name, compress_level=index)
        coco = {
            "images": [
                {
                    "id": 0,
                    "file_name": name,
                    "width": 32,
                    "height": 24,
                    "extra": {"name": name.split(".rf.")[0]},
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [4, 5, 8, 9],
                    "area": 72,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 0, "name": "dataset"}, {"id": 1, "name": "pill"}],
        }
        (directory / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


def test_reencoded_cross_split_duplicates_purge_train_only(tmp_path):
    source = source_fixture(tmp_path / "source", repeated_pixels=True)
    rows, report = PREPARE["inspect_source"](source)
    assert report["groups"]["pixel_sha256"]["cross_split_groups"] == 1
    assert report["groups"]["file_sha256"]["cross_split_groups"] == 0
    assert report["groups"]["capture_hour"]["cross_split_groups"] == 1
    clean, purge = PREPARE["purge_train"](rows)
    assert purge["after_counts"] == {"train": 0, "validation": 1, "test": 1}
    assert clean["test"] is rows["test"]
    assert clean["validation"] is rows["validation"]


def test_preparation_separates_test_and_preserves_native_coco(tmp_path):
    source = source_fixture(tmp_path / "source")
    output, clean = tmp_path / "prepared", tmp_path / "clean"
    report = PREPARE["prepare"](source, output, clean, tmp_path / "review")
    assert report["train_purge"]["removed_train_count"] == 0
    assert (output / "development" / "train.jsonl").is_file()
    assert (output / "development" / "validation.jsonl").is_file()
    assert not (output / "development" / "test.jsonl").exists()
    assert (output / "sealed" / "test.jsonl").is_file()
    for dirname in ("valid", "test"):
        assert (source / dirname / "_annotations.coco.json").read_bytes() == (
            clean / dirname / "_annotations.coco.json"
        ).read_bytes()
    assert (tmp_path / "review" / "train-12.png").is_file()
    assert not (tmp_path / "review" / "test-12.png").exists()
    assert report["label_review"]["status"] == "unreviewed"
    assert report["session_generalization"]["claim_supported"] is False


@pytest.mark.parametrize(
    "bad_box",
    [[31, 0, 2, 2], [0, 0, 0, 2], [-1, 0, 2, 2], [0, 0, float("nan"), 2], [False, 0, 2, 2]],
)
def test_out_of_bounds_or_nonfinite_annotations_are_rejected(tmp_path, bad_box):
    source = source_fixture(tmp_path / "source")
    path = source / "train" / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    coco["annotations"][0]["bbox"] = bad_box
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="bbox"):
        PREPARE["inspect_source"](source)


def test_test_contact_sheets_are_blocked(tmp_path):
    with pytest.raises(ValueError, match="Test contact sheets"):
        PREPARE["contact_sheet"]([{"split": "test"}], tmp_path / "test.png")


def test_capture_hours_are_explicit_not_guessed_from_export_dates():
    assert PREPARE["capture_hour"]("WIN_20240621_10_26_38_Pro_jpg") == "WIN:20240621:10"
    assert PREPARE["capture_hour"]("IMG_7661_jpg") is None
    assert PREPARE["capture_hour"]("2025-03-14T03:01:19+00:00") is None


def test_fingerprint_survives_relocation(tmp_path):
    rows, _ = PREPARE["inspect_source"](source_fixture(tmp_path / "source"))
    original = PREPARE["fingerprint"](rows["test"])
    changed = [dict(row, image="/a/different/root/image.png") for row in rows["test"]]
    assert PREPARE["fingerprint"](changed) == original
