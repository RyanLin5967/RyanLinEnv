"""Optional torch tests for training a classifier on a user's own labeled image folder."""

import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from vision_rl import vision_policy as vp  # noqa: E402
from vision_rl.data import load_manifest  # noqa: E402


def _image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _labeled(root, per_label=6):
    for k, name in enumerate(("cat", "dog")):
        for i in range(per_label):
            _image(root / name / f"{i}.png", (k * 100 + i, i, 50))
    return root


def test_split_keeps_every_label_in_every_split_and_drops_pixel_copies(tmp_path):
    root = _labeled(tmp_path / "photos")
    _image(root / "cat" / "copy.png", (0, 0, 50))  # same pixels as cat/0.png
    result = vp.build_dataset(root, tmp_path / "data")
    assert result["classes"] == ["cat", "dog"]
    assert result["duplicates_dropped"] == 1
    splits = {s: load_manifest(tmp_path / "data" / f"{s}.jsonl", s) for s in result["manifests"]}
    for rows in splits.values():
        assert {r["answer"] for r in rows} == {"A", "B"}
        assert all(r["category"] == "classification" for r in rows)
    hashes = [{r["image_sha256"] for r in rows} for rows in splits.values()]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    assert sum(len(r) for r in splits.values()) == 12


def test_split_refuses_too_few_images_or_labels(tmp_path):
    root = tmp_path / "few"
    _image(root / "cat" / "a.png", (1, 1, 1))
    _image(root / "cat" / "b.png", (2, 2, 2))
    for i in range(3):
        _image(root / "dog" / f"{i}.png", (9, i, 9))
    with pytest.raises(ValueError, match="'cat' has 2 distinct images"):
        vp.build_dataset(root, tmp_path / "out")
    one = tmp_path / "one"
    for i in range(3):
        _image(one / "cat" / f"{i}.png", (i, 0, 0))
    with pytest.raises(ValueError, match="need 2 to 26 label folders"):
        vp.build_dataset(one, tmp_path / "out2")


def test_classification_reward_has_no_partial_credit_for_neighbouring_labels():
    actions, targets = torch.tensor([[0, 1, 2]]), torch.tensor([[1]])
    assert vp.exact_reward(actions, targets, 3).tolist() == [[0.0, 1.0, 0.0]]
    # counting keeps its distance-shaped reward
    assert vp.shaped_reward(actions, targets, 3).tolist() == [[-0.5, 1.0, -0.5]]


def test_task_schema_requires_one_consistent_label_set():
    row = {"category": "classification", "choices": ["cat", "dog"], "prompt": "q"}
    assert vp._task_schema([row, dict(row)]) == (["cat", "dog"], "classification")
    with pytest.raises(ValueError, match="consistent label set"):
        vp._task_schema([row, {**row, "choices": ["dog", "cat"]}])


def _fake_features(rows, encoder, device, *, augment=False, batch_size=24):
    # Mean pixel colour standing in for ResNet features (no model download in tests).
    base = []
    for r in rows:
        with Image.open(r["image"]) as im:
            rgb = torch.tensor(np.asarray(im.convert("RGB").resize((4, 4)), dtype=np.float32))
        rgb = rgb.reshape(-1, 3)
        base.append(torch.cat([rgb.mean(0) / 255, torch.zeros(1021)]))
    x = torch.stack(base)
    return x.repeat(4, 1) if augment else x


def test_train_folder_then_predict_end_to_end(tmp_path, monkeypatch):
    encoder = torch.nn.Linear(2, 2)
    monkeypatch.setattr(vp, "_encoder", lambda path=None: encoder)
    monkeypatch.setattr(vp, "_features", _fake_features)
    photos = _labeled(tmp_path / "photos", per_label=8)
    summary = vp.train_folder(photos, tmp_path / "run", device="cpu", quick=True)
    assert summary["labels"] == ["cat", "dog"]
    assert set(summary["test_accuracy"]) == {"majority_label_baseline", "sft", "rl"}
    assert summary["deployed_model"] in ("sft", "rl")
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert saved == json.loads(json.dumps(summary))
    labels = vp.predict(tmp_path / "run", [photos / "cat"], device="cpu")
    assert len(labels) == 8
    assert all(p["label"] in ("cat", "dog") and 0 <= p["confidence"] <= 1 for p in labels)
    with pytest.raises(ValueError, match="must be new or empty"):
        vp.train_folder(photos, tmp_path / "run", device="cpu", quick=True)


def test_make_example_writes_three_labels(tmp_path):
    out = vp.make_example(tmp_path / "shapes", per_label=3)
    assert sorted(p.name for p in out.iterdir()) == ["circle", "square", "triangle"]
    assert all(len(list(p.glob("*.png"))) == 3 for p in out.iterdir())


def test_existing_train_valid_test_folders_are_kept_as_given(tmp_path):
    root = tmp_path / "export"
    for split, offset in (("train", 0), ("valid", 50), ("test", 100)):
        for k, name in enumerate(("cat", "dog")):
            _image(root / split / name / "a.png", (offset + k, 3, 3))
    result = vp.build_dataset(root, tmp_path / "data")
    assert result["source"] == "existing train/validation/test split"
    assert result["counts"] == {"train": 2, "validation": 2, "test": 2}


def test_partial_split_export_refuses_instead_of_training_on_split_names(tmp_path):
    root = tmp_path / "export"
    for split in ("train", "valid"):
        for i in range(3):
            _image(root / split / "cat" / f"{i}.png", (i, 5, 5))
    with pytest.raises(ValueError, match="looks pre-split"):
        vp.build_dataset(root, tmp_path / "data")


def test_hidden_files_and_macos_zip_debris_are_skipped(tmp_path):
    root = _labeled(tmp_path / "photos", per_label=3)
    (root / "cat" / "._0.png").write_bytes(b"not an image")
    (root / ".DS_Store").write_bytes(b"x")
    _image(root / "__MACOSX" / "cat" / "a.png", (7, 7, 7))
    result = vp.build_dataset(root, tmp_path / "data")
    assert result["classes"] == ["cat", "dog"]
    assert sum(result["counts"].values()) == 6


def test_same_image_under_two_labels_refuses(tmp_path):
    root = _labeled(tmp_path / "photos", per_label=3)
    _image(root / "dog" / "mislabelled.png", (0, 0, 50))  # cat/0.png's pixels
    with pytest.raises(ValueError, match="labelled both 'cat'"):
        vp.build_dataset(root, tmp_path / "data")


def test_predict_checks_the_encoder_and_accepts_any_model_path(tmp_path, monkeypatch):
    encoder = torch.nn.Linear(2, 2)
    monkeypatch.setattr(vp, "_encoder", lambda path=None: encoder)
    monkeypatch.setattr(vp, "_features", _fake_features)
    photos = _labeled(tmp_path / "photos", per_label=4)
    vp.train_folder(photos, tmp_path / "run", device="cpu", quick=True)
    image = [photos / "cat" / "0.png"]
    for model in (tmp_path / "run", tmp_path / "run" / "model", tmp_path / "run/model/best.pt"):
        assert len(vp.predict(model, image, device="cpu")) == 1
    (tmp_path / "run" / "model" / "encoder.pt").write_bytes(b"swapped")
    with pytest.raises(ValueError, match="encoder checksum mismatch"):
        vp.predict(tmp_path / "run", image, device="cpu")
