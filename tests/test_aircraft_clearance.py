import copy
import json
import runpy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
VERIFY = runpy.run_path(str(Path(__file__).parents[1] / "scripts/verify_aircraft_clearance.py"))
PUBLISH = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_aircraft_manifests.py"))


def fixture():
    fields = {
        "encoded_sha256": "a" * 64,
        "decoded_rgb_sha256": "b" * 64,
        "model_rgb_sha256": "c" * 64,
        "source_image_size": [100, 80],
        "model_image_size": [100, 60],
    }
    author = [{"image_id": "0000001", "split": "train", "variant": "variant", **fields}]
    candidate = {"images": [{"id": "0000001", "split": "train", "label": "variant", **fields}]}
    return author, candidate


def test_equivalence_retains_frozen_candidate_and_matches_every_pixel_contract():
    author, candidate = fixture()
    before = copy.deepcopy(candidate)
    result = VERIFY["compare_candidate"](author, candidate, ["0000001"])
    assert result["all_encoded_bytes_match"]
    assert result["all_full_and_model_pixels_match"]
    assert candidate == before


@pytest.mark.parametrize("field", ["encoded_sha256", "model_rgb_sha256", "source_image_size"])
def test_original_equivalence_rejects_byte_pixel_or_dimension_mismatch(field):
    author, candidate = fixture()
    candidate["images"][0][field] = "changed"
    with pytest.raises(ValueError, match=field):
        VERIFY["compare_candidate"](author, candidate, ["0000001"])


def test_original_equivalence_rejects_wrong_native_split_or_frozen_ids():
    author, candidate = fixture()
    with pytest.raises(ValueError, match="frozen metadata"):
        VERIFY["compare_candidate"](author, candidate, ["0000002"])
    author[0]["split"] = "test"
    with pytest.raises(ValueError, match="held-out"):
        VERIFY["compare_candidate"](author, candidate, ["0000001"])


def model_fixture(tmp_path, monkeypatch):
    author, candidate = fixture()
    preprocessing = VERIFY["aircraft_data"].PREPROCESSING_ID
    candidate["preprocessing_id"] = preprocessing
    plan = {
        "canonical_labels": ["variant"],
        "selected": [{"id": "0000001", "split": "train", "label": "variant"}],
    }
    manifest = [{**plan["selected"][0], "image": "original-jpeg/0000001.jpg"}]
    spec = {"schema_version": 1, "labels": ["variant"], "preprocessing_id": preprocessing}
    monkeypatch.setitem(
        VERIFY["validate_model_manifest"].__globals__, "file_hash", lambda _: "a" * 64
    )
    return [
        author,
        candidate,
        plan,
        manifest,
        spec,
        ["variant"],
        tmp_path,
        tmp_path / "original-jpeg",
    ]


def test_model_manifest_semantics_match_author_and_preserve_frozen_inputs(tmp_path, monkeypatch):
    args = model_fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(args)
    VERIFY["validate_model_manifest"](*args)
    assert args == before


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("label", "wrong", "label"),
        ("split", "test", "native split"),
        ("image", "original-jpeg/0000002.jpg", "image path"),
    ],
)
def test_consistently_hashed_bad_manifest_still_fails_semantics(
    tmp_path, monkeypatch, field, value, error
):
    args = model_fixture(tmp_path, monkeypatch)
    args[3][0][field] = value
    # This validator runs after the frozen byte hashes pass; semantics remain mandatory.
    with pytest.raises(ValueError, match=error):
        VERIFY["validate_model_manifest"](*args)


@pytest.mark.parametrize("field,value", [("labels", ["wrong"]), ("preprocessing_id", "wrong")])
def test_semantic_spec_cannot_change_full_vocabulary_or_pixel_contract(
    tmp_path, monkeypatch, field, value
):
    args = model_fixture(tmp_path, monkeypatch)
    args[4][field] = value
    with pytest.raises(ValueError, match="vocabulary|preprocessing"):
        VERIFY["validate_model_manifest"](*args)


def test_publisher_without_equivalence_receipt_cannot_emit_validation(tmp_path, monkeypatch):
    globals_ = PUBLISH["main"].__globals__
    monkeypatch.setitem(
        globals_, "verify_completed", lambda *args: {"training_integrity_clear": True}
    )
    monkeypatch.setitem(globals_, "audit", lambda *args: pytest.fail("publication began"))
    monkeypatch.setattr(sys, "argv", ["publish", "--root", str(tmp_path)])
    with pytest.raises(ValueError, match="receipt required"):
        PUBLISH["main"]()
    assert not (tmp_path / "development").exists()


def test_receipt_must_equal_freshly_recomputed_evidence(tmp_path, monkeypatch):
    path = tmp_path / "audit/author-equivalence-v1.json"
    path.parent.mkdir()
    receipt = {"status": "verified", "input_hash": "original", "verifier_hash": "current"}
    path.write_text(json.dumps(receipt))
    globals_ = VERIFY["require_clearance"].__globals__
    monkeypatch.setitem(globals_, "build_clearance", lambda *args: copy.deepcopy(receipt))
    assert (
        VERIFY["require_clearance"](tmp_path, tmp_path / "metadata", tmp_path / "probe") == receipt
    )
    changed = {**receipt, "input_hash": "changed-after-receipt"}
    monkeypatch.setitem(globals_, "build_clearance", lambda *args: changed)
    with pytest.raises(ValueError, match="stale or not bound"):
        VERIFY["require_clearance"](tmp_path, tmp_path / "metadata", tmp_path / "probe")
