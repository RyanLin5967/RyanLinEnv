import io
import json
import runpy
import sys
import tarfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
EXTRACT = runpy.run_path(str(Path(__file__).parents[1] / "scripts/extract_aircraft.py"))
PROBE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_aircraft_probe.py"))
MANIFEST = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_aircraft_manifests.py"))


def test_archive_disk_reserve_checked_before_any_network(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("network opened"))
    with pytest.raises(RuntimeError, match="reserve"):
        EXTRACT["acquire"](tmp_path, 10**30)


def test_partial_download_is_preserved_without_starting_another(monkeypatch, tmp_path):
    partial = tmp_path / "downloads/fgvc-aircraft-2013b.tar.gz.part"
    partial.parent.mkdir()
    partial.write_bytes(b"incomplete original evidence")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("network opened"))
    with pytest.raises(ValueError, match="Partial archive"):
        EXTRACT["acquire"](tmp_path, 0)
    assert partial.read_bytes() == b"incomplete original evidence"


def test_original_bytes_sealed_test_and_duplicate_gate(tmp_path, monkeypatch):
    rows = [
        {"image_id": f"{index:07d}", "split": split, "variant": "A"}
        for index, split in enumerate(("train", "val", "test"), 1)
    ]
    globals_ = EXTRACT["extract"].__globals__
    metadata_files = {"data/images_train.txt": b"0000001\n", "evaluation.m": b"source"}
    monkeypatch.setitem(globals_, "audit", lambda _: ({"source": {}, "image_terms": "test"}, rows))
    monkeypatch.setitem(globals_, "read_archive", lambda _: metadata_files)
    metadata = tmp_path / "metadata"
    metadata.write_bytes(b"synthetic fixture")
    encoded = io.BytesIO()
    Image.new("RGB", (10, 30), (30, 40, 50)).save(encoded, format="JPEG")
    content = encoded.getvalue()
    archive = tmp_path / "source.tar.gz"
    files = {
        **metadata_files,
        **{f"data/images/{row['image_id']}.jpg": content for row in rows},
    }
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in files.items():
            member = tarfile.TarInfo("fgvc-aircraft-2013b/" + name)
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))
    output = tmp_path / "dataset"
    report = EXTRACT["extract"](output, archive, {}, metadata, 0)
    assert not report["training_integrity_clear"]
    assert report["images"] == 3
    assert report["native_labels_and_membership_preserved"]
    assert (output / "sealed/images/test/0000003.jpg").read_bytes() == content
    assert not (output / "development/images/test").exists()
    assert len(list(output.rglob("*.jpg"))) == 3
    assert report["persistent_transformed_image_copies"] == 0
    assert len(report["duplicates"]["model_rgb_sha256"]["cross_split"]) == 1
    manifest = json.loads((output / "audit/test-images.json").read_text())
    assert manifest[0]["source_image_size"] == [10, 30]
    assert manifest[0]["model_image_size"] == [10, 10]
    assert report["test_access"].startswith("Automated integrity decoding only")
    (output / "audit/archive.json").write_text("{}")
    monkeypatch.setitem(globals_, "load_source_rgb", lambda _: pytest.fail("unneeded decode"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("network opened"))
    assert EXTRACT["verify_completed"](output, metadata) == report
    original_path = output / "development/images/train/0000001.jpg"
    original_path.write_bytes(content[:-1] + b"X")
    with pytest.raises(ValueError, match="Retained original JPEG changed"):
        EXTRACT["verify_completed"](output, metadata)
    assert original_path.read_bytes() == content[:-1] + b"X"


def probe_selection():
    metadata = [
        {"image_id": f"{index:07d}", "split": "train", "variant": f"class{index}"}
        for index in range(16)
    ]
    labels = [row["variant"] for row in metadata] + ["unselected class"]
    selection = {
        "canonical_labels": labels,
        "selected": [
            {"id": row["image_id"], "label": row["variant"], "split": "train"} for row in metadata
        ],
    }
    return metadata, labels, selection


def test_probe_preserves_selection_order_and_full_label_vocabulary():
    metadata, labels, selection = probe_selection()
    actual = PROBE["validate_selection"](selection, list(reversed(metadata)), labels)
    assert [row["id"] for row in actual] == [row["image_id"] for row in metadata]
    selection["canonical_labels"] = labels[:-1]
    with pytest.raises(ValueError, match="full official"):
        PROBE["validate_selection"](selection, metadata, labels)


@pytest.mark.parametrize("invalid_split", ["val", "test"])
def test_probe_cannot_select_heldout_native_image(invalid_split):
    metadata, labels, selection = probe_selection()
    metadata[-1]["split"] = invalid_split
    with pytest.raises(ValueError, match="native TRAIN"):
        PROBE["validate_selection"](selection, metadata, labels)


def test_probe_refuses_incorrect_label_or_repeated_id():
    metadata, labels, selection = probe_selection()
    selection["selected"][0]["label"] = "incorrect"
    with pytest.raises(ValueError, match="original author"):
        PROBE["validate_selection"](selection, metadata, labels)
    selection["selected"][0] = selection["selected"][1]
    with pytest.raises(ValueError, match="unique"):
        PROBE["validate_selection"](selection, metadata, labels)


def test_development_manifest_points_to_originals_without_exposing_metadata():
    record = {
        "image_id": "0001234",
        "split": "val",
        "variant": "exact variant",
        "path": "development/images/validation/0001234.jpg",
        "manufacturer": "hidden",
        "source_bbox_xyxy": [1, 2, 3, 4],
    }
    assert MANIFEST["development_rows"]([record], "val") == [
        {
            "id": "0001234",
            "split": "validation",
            "image": "images/validation/0001234.jpg",
            "label": "exact variant",
        }
    ]
    record["split"] = "test"
    with pytest.raises(ValueError, match="membership"):
        MANIFEST["development_rows"]([record], "val")


def test_manifest_publication_never_silently_replaces_existing_experiment_input(tmp_path):
    path = tmp_path / "train.jsonl"
    MANIFEST["write_jsonl"](path, [{"id": "original"}])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="silent replacement"):
        MANIFEST["write_jsonl"](path, [{"id": "changed"}])
    assert path.read_bytes() == before
