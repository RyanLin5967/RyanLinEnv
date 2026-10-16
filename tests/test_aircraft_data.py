import runpy
from pathlib import Path

import pytest

PREPARE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_aircraft.py"))


def rows():
    return [
        {
            "image_id": f"{index:07d}",
            "split": "train",
            "variant": label,
            "source_bbox_xyxy": [0, 1, 40, 50],
            "source_image_size": None,
        }
        for index, label in enumerate(["A"] * 8 + ["B"] * 8)
    ]


def test_fewshot_is_balanced_deterministic_and_independent_of_source_order():
    selected = PREPARE["select_fewshot"](rows(), ["A", "B"], shots=3)
    reversed_selection = PREPARE["select_fewshot"](list(reversed(rows())), ["A", "B"], shots=3)
    assert selected == reversed_selection
    assert selected["images"] == 6
    assert len(set(selected["selected_image_ids"])) == 6
    by_id = {row["image_id"]: row for row in rows()}
    assert [by_id[i]["variant"] for i in selected["selected_image_ids"]] == ["A"] * 3 + ["B"] * 3
    assert selected["status"] == "prospective_metadata_only_not_image_ready"
    assert not selected["uses_validation_or_test"]


@pytest.mark.parametrize("split", ["val", "validation", "test"])
def test_fewshot_refuses_heldout_rows(split):
    mixed = rows()
    mixed[0]["split"] = split
    with pytest.raises(ValueError, match="only native training"):
        PREPARE["select_fewshot"](mixed, ["A", "B"], shots=3)


def test_fewshot_refuses_incomplete_classes_and_duplicate_ids():
    with pytest.raises(ValueError, match="Not enough"):
        PREPARE["select_fewshot"](rows(), ["A", "B"], shots=9)
    with pytest.raises(ValueError, match="Duplicate"):
        PREPARE["select_fewshot"](rows() + rows()[:1], ["A", "B"], shots=3)


def test_labels_preserve_spaces_and_leading_zero_image_ids():
    parse = PREPARE["parse_labeled_rows"]
    assert parse(b"0000001 Model With Spaces\n", ["Model With Spaces"]) == [
        ("0000001", "Model With Spaces")
    ]
    with pytest.raises(ValueError, match="unknown"):
        parse(b"0000001 model with spaces\n", ["Model With Spaces"])
    with pytest.raises(ValueError, match="Malformed"):
        parse(b"1 Model With Spaces\n", ["Model With Spaces"])


def test_archive_pin_rejects_wrong_bytes_before_extraction():
    with pytest.raises(ValueError, match="pinned SHA256"):
        PREPARE["read_archive"](b"untrusted archive")


def test_changed_seed_changes_selection_without_drawing_from_other_splits():
    first = PREPARE["select_fewshot"](rows(), ["A", "B"], shots=3, seed=2718)
    second = PREPARE["select_fewshot"](rows(), ["A", "B"], shots=3, seed=2719)
    assert first["selected_image_ids"] != second["selected_image_ids"]
    assert set(second["selected_image_ids"]) <= {row["image_id"] for row in rows()}
