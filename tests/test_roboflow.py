import io
import json
import stat
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from vision_rl.data import assert_disjoint, load_manifest
from vision_rl.roboflow import (
    RoboflowError,
    _get_json,
    _safe_extract,
    download_dataset,
    import_classification,
    import_coco_counts,
    list_projects,
)


def image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def classification_fixture(root):
    for index, split in enumerate(("train", "valid", "test")):
        for number, category in enumerate(("cat", "dog")):
            image(root / split / category / "sample.png", (index * 70, number * 90, 10))
    return root


def coco_fixture(root):
    for index, split in enumerate(("train", "valid", "test")):
        image(root / split / "sample.png", (index * 70, 50, 10))
        data = {
            "categories": [{"id": 1, "name": "chess-piece"}],
            "images": [{"id": 1, "file_name": "sample.png"}],
            "annotations": [
                {"id": number, "image_id": 1, "category_id": 1, "iscrowd": 0}
                for number in range(index + 1)
            ],
        }
        (root / split / "_annotations.coco.json").write_text(json.dumps(data))
    return root


def test_list_projects_discovers_workspace_without_exposing_other_metadata(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-secret")
    monkeypatch.delenv("ROBOFLOW_WORKSPACE", raising=False)
    responses = [
        {"workspace": "demo"},
        {
            "workspace": {
                "projects": [
                    {
                        "id": "demo/chess",
                        "type": "object-detection",
                        "images": 200,
                        "versions": 2,
                        "api_key": "test-secret",
                        "private": "omitted",
                    }
                ],
                "members": ["not-returned"],
            }
        },
    ]
    with patch("vision_rl.roboflow._get_json", side_effect=responses) as request:
        result = list_projects()
    assert result["workspace"] == "demo"
    assert result["project_count"] == 1
    assert result["projects"][0]["images"] == 200
    assert "test-secret" not in json.dumps(result)
    assert "members" not in json.dumps(result)
    assert request.call_args_list[0].args == ("/", "test-secret")


def test_api_errors_do_not_echo_key_url_or_response():
    secret = "test-secret"
    error = urllib.error.HTTPError(
        f"https://api.roboflow.com/?api_key={secret}",
        403,
        secret,
        {},
        io.BytesIO(secret.encode()),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(RoboflowError) as caught:
            _get_json("/", secret)
    assert "403" in str(caught.value)
    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__


def test_key_required_and_slugs_checked_before_requests(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RoboflowError, match="ROBOFLOW_API_KEY"):
        list_projects()
    with pytest.raises(ValueError, match="slug"):
        download_dataset("../workspace", "project", 1, "coco", tmp_path / "out")


@pytest.mark.parametrize("filename", ["../outside.txt", "/absolute.txt", "a/../../b", "a\\b"])
def test_zip_rejects_traversal(tmp_path, filename):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(filename, "bad")
    target = tmp_path / "extract"
    target.mkdir()
    with pytest.raises(RoboflowError, match="unsafe"):
        _safe_extract(archive, target)
    assert list(target.iterdir()) == []


def test_zip_rejects_symlinks_and_size_limit(tmp_path, monkeypatch):
    archive = tmp_path / "link.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(member, "../../elsewhere")
    with pytest.raises(RoboflowError, match="unsafe"):
        _safe_extract(archive, tmp_path)
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("large.txt", "123456")
    monkeypatch.setattr("vision_rl.roboflow.MAX_EXTRACT_BYTES", 5)
    with pytest.raises(RoboflowError, match="size limit"):
        _safe_extract(archive, tmp_path)


def test_download_uses_version_export_then_signed_link(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-secret")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("train/example.txt", "downloaded")
    buffer.seek(0)
    with (
        patch(
            "vision_rl.roboflow._get_json",
            return_value={
                "export": {
                    "link": "https://storage.roboflow.com/signed-dataset?key=download-token"
                },
            },
        ) as metadata,
        patch("urllib.request.urlopen", return_value=buffer) as download,
    ):
        result = download_dataset("workspace", "project", 2, "coco", tmp_path / "dataset")
    metadata.assert_called_once_with("/workspace/project/2/coco", "test-secret")
    assert "test-secret" not in download.call_args.args[0]
    assert (result / "train/example.txt").read_text() == "downloaded"
    assert not list(tmp_path.glob(".roboflow-*"))


def test_download_failure_leaves_no_output_or_signed_url(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-secret")
    signed_url = "https://storage.roboflow.com/dataset?key=download-token"
    with (
        patch("vision_rl.roboflow._get_json", return_value={"export": {"link": signed_url}}),
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError(signed_url)),
        pytest.raises(RoboflowError) as caught,
    ):
        download_dataset("workspace", "project", 2, "coco", tmp_path / "dataset")
    assert "download-token" not in str(caught.value)
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".roboflow-*"))


def test_classification_preserves_splits_and_train_vocabulary(tmp_path):
    source = classification_fixture(tmp_path / "source")
    result = import_classification(source, tmp_path / "manifests")
    assert result["classes"] == ["cat", "dog"]
    assert result["counts"] == {"train": 2, "validation": 2, "test": 2}
    rows = [load_manifest(result["manifests"][split], split) for split in result["counts"]]
    assert_disjoint(*rows)
    for split_rows in rows:
        assert [row["answer"] for row in split_rows] == ["A", "B"]
        assert all(row["category"] == "classification" for row in split_rows)
    assert Path(rows[1][0]["image"]).parent.parent.name == "valid"


def test_classification_rejects_unseen_test_class_before_writing(tmp_path):
    source = classification_fixture(tmp_path / "source")
    image(source / "test" / "zebra" / "sample.png", "orange")
    with pytest.raises(ValueError, match="classes must all occur"):
        import_classification(source, tmp_path / "manifests")
    assert not (tmp_path / "manifests").exists()


def test_classification_rejects_renamed_pixel_leakage(tmp_path):
    source = classification_fixture(tmp_path / "source")
    with Image.open(source / "train/cat/sample.png") as sample:
        sample.save(source / "test/cat/copy.bmp")
    with pytest.raises(ValueError, match="leakage"):
        import_classification(source, tmp_path / "manifests")
    assert not (tmp_path / "manifests").exists()


def test_coco_count_import_preserves_counts_and_splits(tmp_path):
    source = coco_fixture(tmp_path / "source")
    result = import_coco_counts(source, tmp_path / "manifests", "chess-piece", max_count=4)
    for index, split in enumerate(("train", "validation", "test")):
        row = load_manifest(result["manifests"][split], split)[0]
        assert row["answer"] == chr(66 + index)
        assert row["choices"] == ["0", "1", "2", "3", "4"]
        assert row["category"] == "counting"
        assert "chess-piece" in row["prompt"]


def test_coco_rejects_counts_outside_choice_range(tmp_path):
    source = coco_fixture(tmp_path / "source")
    with pytest.raises(ValueError, match="exceeds max_count"):
        import_coco_counts(source, tmp_path / "manifests", max_count=2)
    assert not (tmp_path / "manifests").exists()


def test_coco_rejects_crowd_counts_and_path_traversal(tmp_path):
    source = coco_fixture(tmp_path / "source")
    annotations = source / "train/_annotations.coco.json"
    data = json.loads(annotations.read_text())
    data["annotations"][0]["iscrowd"] = 1
    annotations.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="crowd"):
        import_coco_counts(source, tmp_path / "manifests")
    data["annotations"][0]["iscrowd"] = 0
    data["images"][0]["file_name"] = "../valid/sample.png"
    annotations.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        import_coco_counts(source, tmp_path / "manifests")
