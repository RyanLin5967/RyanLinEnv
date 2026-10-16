import io
import runpy
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
EXTRACT = runpy.run_path(str(Path(__file__).parents[1] / "scripts/extract_wheat.py"))


def test_encoded_and_pixel_duplicates_are_distinct_checks():
    first, second = io.BytesIO(), io.BytesIO()
    image = Image.new("RGB", (8, 8), (31, 41, 59))
    image.save(first, format="PNG", compress_level=0)
    image.save(second, format="PNG", compress_level=9)
    a = EXTRACT["image_integrity"](first.getvalue(), (8, 8))
    b = EXTRACT["image_integrity"](second.getvalue(), (8, 8))
    assert a["encoded_sha256"] != b["encoded_sha256"]
    assert a["decoded_rgb_sha256"] == b["decoded_rgb_sha256"]
    with pytest.raises(ValueError, match="dimensions"):
        EXTRACT["image_integrity"](first.getvalue())


def test_cross_split_duplicate_report_keeps_membership():
    rows = [
        {"split": split, "image_id": i, "original_image_id": str(i), "pixel": "same"}
        for i, split in enumerate(("train", "validation"))
    ]
    report = EXTRACT["duplicate_groups"](rows, "pixel")
    assert len(report["cross_split"]) == 1
    assert [row["split"] for row in report["cross_split"][0]["members"]] == ["train", "validation"]


def test_conversion_preserves_native_duplicates_outside_extent_and_domain():
    row = {
        "image": {"path": "a" * 64 + ".jpg"},
        "domain": "session",
        "country": "country",
        "location": "location",
        "development_stage": "stage",
        "objects": {"boxes": [[1000, 0, 40, 40]] * 2, "categories": [0, 0]},
    }
    image, annotations = EXTRACT["coco_row"](row, 7, 31, ["session"])
    assert image["domain"] == "session"
    assert image["session_id"] == 0
    assert [item["id"] for item in annotations] == [31, 32]
    assert [item["bbox"] for item in annotations] == [[1000, 0, 40, 40]] * 2


def test_download_rejects_checksum_failure_and_preserves_partial(monkeypatch, tmp_path):
    import prepare_wheat  # noqa: F401

    globals_ = EXTRACT["download_shard"].__globals__
    monkeypatch.setitem(globals_["PINNED"], "fake.parquet", (4, "0" * 64))

    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"fake"))
    with pytest.raises(ValueError, match="verification"):
        EXTRACT["download_shard"]("fake.parquet", tmp_path / "fake.parquet", 0)
    assert not (tmp_path / "fake.parquet").exists()
    assert (tmp_path / "fake.parquet.part").read_bytes() == b"fake"


def test_disk_guard_fails_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network"))
    with pytest.raises(RuntimeError, match="reserve"):
        EXTRACT["download_shard"](
            "train-00000-of-00002.parquet", tmp_path / "source.parquet", 10**20
        )
