"""Protect native wheat annotations and bound metadata-only network reads."""

import copy
import io
import runpy
from pathlib import Path

import pytest

PREPARE = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "prepare_wheat.py"))


def test_box_conversion_preserves_repetition_and_truncated_extent():
    convert = PREPARE["csv_xyxy_to_xywh"]
    assert convert("no_box") == []
    assert convert("1000 10 1040 60;1000 10 1040 60") == [
        [1000, 10, 40, 50],
        [1000, 10, 40, 50],
    ]


@pytest.mark.parametrize("text", ["0 0 0 4", "0 nan 4 5", "0 0 1.5 5", "0 0 2"])
def test_invalid_source_coordinates_are_rejected_without_eval(text):
    with pytest.raises(ValueError):
        PREPARE["csv_xyxy_to_xywh"](text)


def test_audit_reports_original_duplicate_resolution_without_mutating_labels():
    originals = [
        {"image_name": "a.png", "BoxesString": "1 2 3 4", "domain": "session"},
        {"image_name": "a.png", "BoxesString": "1000 1 1040 3;1000 1 1040 3", "domain": "session"},
        {"image_name": "b.png", "BoxesString": "no_box", "domain": "session"},
    ]
    rows = [
        {
            "image": {"path": "a.jpg"},
            "objects": {"boxes": [[1000, 1, 40, 2], [1000, 1, 40, 2]], "categories": [0, 0]},
            "domain": "session",
            "country": "country",
        },
        {
            "image": {"path": "b.jpg"},
            "objects": {"boxes": [], "categories": []},
            "domain": "session",
            "country": "country",
        },
    ]
    before = copy.deepcopy(rows)
    report = PREPARE["audit_rows"](rows, originals)
    assert rows == before
    assert report["exact_original_annotation_matches"] == 2
    assert report["duplicate_original_rows"][0]["hf_matches_original_indices"] == [1]
    assert report["empty_images"] == 1
    assert report["boxes"] == 2
    assert len(report["boxes_outside_declared_1024_dimensions"]) == 2
    assert report["repeated_exact_boxes"][0]["boxes"][0]["multiplicity"] == 2


def test_box_and_domain_must_match_the_same_original_duplicate_row():
    originals = [
        {"image_name": "a.png", "BoxesString": "0 0 10 10", "domain": "early"},
        {"image_name": "a.png", "BoxesString": "0 0 20 20", "domain": "later"},
    ]
    rows = [
        {
            "image": {"path": "a.jpg"},
            "objects": {"boxes": [[0, 0, 10, 10]], "categories": [0]},
            "domain": "later",
            "country": "country",
        }
    ]
    report = PREPARE["audit_rows"](rows, originals)
    assert report["exact_original_annotation_matches"] == 0
    assert report["annotation_mismatch_ids"] == ["a"]


def test_protocol_requires_official_group_averaging_evidence():
    audit = {
        "wilds_official_metric_verification": {
            "source_file_sha256": PREPARE["WILDS_SOURCE_HASHES"],
            "native_dataset_eval_synthetic": {"equal_domain_average": 0.75, "image_average": 0.75},
        }
    }
    with pytest.raises(ValueError, match="grouped metric"):
        PREPARE["build_hf_protocol"](audit)


def test_range_reader_refuses_full_download_before_network_request(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Oversized request must be rejected before network access")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    reader = PREPARE["RangeReader"](
        "https://example.org/large.parquet", 1_000_000_000, {"bytes": 0}
    )
    with pytest.raises(ValueError, match="budget"):
        reader.read()


def test_range_reader_rejects_server_ignoring_range_without_reading_body(monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, count):
            pytest.fail("Ignored Range body must never be downloaded")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    reader = PREPARE["RangeReader"]("https://example.org/source", 100, {"bytes": 0})
    with pytest.raises(ValueError, match="ignored"):
        reader.read(8)


def test_range_reader_seek_and_budget_match_actual_transfer(monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 92-99/100"}

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"abcdefgh"))
    budget = {"bytes": 0}
    reader = PREPARE["RangeReader"]("https://example.org/source", 100, budget)
    assert reader.seek(-8, 2) == 92
    assert reader.read(20) == b"abcdefgh"
    assert reader.tell() == 100
    assert budget == {"bytes": 8}
    assert reader.read(1) == b""
