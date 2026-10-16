import hashlib
import io
import runpy
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
PREPARE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_aircraft_fourshot.py"))
FETCH = runpy.run_path(str(Path(__file__).parents[1] / "scripts/fetch_aircraft_fourshot.py"))


def test_fourshot_plan_is_balanced_train_only_and_order_independent(monkeypatch):
    labels = [f"variant {index:03d}" for index in range(100)]
    rows = [
        {"image_id": f"{class_id * 10 + shot:07d}", "variant": label, "split": "train"}
        for class_id, label in enumerate(labels)
        for shot in range(8)
    ]
    rows += [
        {"image_id": "9000001", "variant": labels[0], "split": "val"},
        {"image_id": "9000002", "variant": labels[0], "split": "test"},
    ]
    report = {
        "canonical_labels": {"variant": labels},
        "source": {"url": "pinned-author", "sha256": "a" * 64},
        "source_file_sha256": {"data/images_variant_train.txt": "b" * 64},
        "splits": {"train": {"metadata_rows_sha256": "c" * 64}},
        "image_terms": "Non-commercial research",
    }
    globals_ = PREPARE["make_plan"].__globals__
    monkeypatch.setitem(globals_, "audit", lambda _: (report, rows))
    plan = PREPARE["make_plan"](b"synthetic source")
    assert len(plan["selected"]) == 400
    assert Counter(row["label"] for row in plan["selected"]) == dict.fromkeys(labels, 4)
    assert {row["split"] for row in plan["selected"]} == {"train"}
    assert not {"9000001", "9000002"} & {row["id"] for row in plan["selected"]}
    assert plan["selection"]["seed"] == 791
    assert plan["canonical_labels"] == labels
    assert plan["uses_model_outputs_or_rewards"] is False
    assert plan["existing16_image_probe_modified"] is False
    assert plan["existing1600_image_plan_activated"] is False
    rows.reverse()
    assert PREPARE["make_plan"](b"synthetic source") == plan


def test_acquisition_rejects_selection_change_before_reading_metadata():
    with pytest.raises(ValueError, match="prospectively frozen"):
        FETCH["validate_plan"](b"changed plan", b"not read")


def test_matching_probe_image_reuses_one_inode_and_exact_lazy_pixels(tmp_path, monkeypatch):
    probe = tmp_path / "probe"
    prior = probe / "original-jpeg/0000001.jpg"
    prior.parent.mkdir(parents=True)
    Image.new("RGB", (40, 42), (20, 30, 40)).save(prior)
    content = prior.read_bytes()
    row = {"id": "0000001", "label": "variant", "split": "train"}
    info = {
        FETCH["PREFIX"] + "0000001.jpg": {
            "size": len(content),
            "lfs": {"oid": hashlib.sha256(content).hexdigest()},
        }
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("network opened"))
    record = FETCH["acquire_one"](row, info, tmp_path / "dataset", probe)
    target = tmp_path / "dataset" / record["path"]
    assert target.stat().st_ino == prior.stat().st_ino
    assert target.read_bytes() == content
    assert record["downloaded_bytes"] == 0
    assert record["model_image_size"] == [40, 22]
    assert record["acquisition"] == "verified_probe_hardlink_no_payload_copy"


def test_wrong_mirror_bytes_never_become_a_candidate_image(tmp_path, monkeypatch):
    globals_ = FETCH["acquire_one"].__globals__
    monkeypatch.setitem(globals_, "check_space", lambda *args: None)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"bad"))
    info = {FETCH["PREFIX"] + "0000001.jpg": {"size": 3, "lfs": {"oid": "0" * 64}}}
    with pytest.raises(ValueError, match="LFS SHA256"):
        FETCH["acquire_one"](
            {"id": "0000001", "label": "variant", "split": "train"},
            info,
            tmp_path / "dataset",
            tmp_path / "probe",
        )
    assert not list((tmp_path / "dataset").rglob("*.jpg"))
