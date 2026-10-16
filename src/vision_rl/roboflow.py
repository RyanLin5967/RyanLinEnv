"""Read-only Roboflow dataset access and split-preserving benchmark imports.

The REST routes follow Roboflow's official Python SDK (adapters/rfapi.py).
No training, upload, model deployment, or workspace mutation endpoints are used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .data import assert_disjoint, image_hash

API_URL = "https://api.roboflow.com"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_DOWNLOAD_BYTES = 2 * 1024**3
MAX_EXTRACT_BYTES = 4 * 1024**3
MAX_MEMBER_BYTES = 512 * 1024**2
MAX_ARCHIVE_MEMBERS = 100_000


class RoboflowError(RuntimeError):
    """An actionable error that does not include API keys or signed URLs."""


def _api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise RoboflowError("Set ROBOFLOW_API_KEY in your environment or local .env file.")
    return key


def _slug(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError(f"{label} must be a Roboflow URL slug, not a URL or path")
    return value


def _get_json(path: str, key: str) -> dict:
    url = API_URL + path + "?" + urllib.parse.urlencode({"api_key": key})
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            # Workspace/project/export metadata should be small; bound unexpected responses.
            body = response.read(8 * 1024**2 + 1)
            if len(body) > 8 * 1024**2:
                raise RoboflowError("Roboflow metadata response exceeds the size limit.")
            payload = json.loads(body)
            if response.status == 202:
                raise RoboflowError("Dataset export is preparing. Retry the download shortly.")
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        hint = " Check your API key and workspace access." if status in (401, 403) else ""
        raise RoboflowError(f"Roboflow API returned HTTP {status}.{hint}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RoboflowError(
            "Cannot reach the Roboflow API; check your connection and retry."
        ) from None
    except (ValueError, UnicodeDecodeError):
        raise RoboflowError("Roboflow returned invalid JSON metadata.") from None
    if not isinstance(payload, dict):
        raise RoboflowError("Roboflow returned unexpected metadata.")
    return payload


def list_projects(workspace: str | None = None) -> dict:
    """List safe project metadata; discover the key's workspace when omitted."""
    key = _api_key()
    workspace = workspace or os.environ.get("ROBOFLOW_WORKSPACE")
    if not workspace:
        workspace = _get_json("/", key).get("workspace")
        if isinstance(workspace, dict):
            workspace = workspace.get("url") or workspace.get("id")
        if not isinstance(workspace, str):
            raise RoboflowError("Cannot determine workspace; provide its URL slug explicitly.")
    workspace = _slug(workspace, "workspace")
    payload = _get_json(f"/{workspace}", key)
    metadata = payload.get("workspace", {})
    projects = metadata.get("projects", []) if isinstance(metadata, dict) else None
    if not isinstance(projects, list) or any(not isinstance(p, dict) for p in projects):
        raise RoboflowError("Roboflow returned an unexpected project list.")
    # Avoid returning credentials, signed download URLs, user profiles, or member data.
    return {
        "workspace": workspace,
        "project_count": len(projects),
        "projects": [
            {k: p[k] for k in ("id", "type", "images", "versions") if k in p} for p in projects
        ],
    }


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract into a new staging directory after validating the entire ZIP index."""
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise RoboflowError("Dataset archive has too many files.")
        seen, total = set(), 0
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            mode = member.external_attr >> 16
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or ":" in name
                or not path.parts
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
            ):
                raise RoboflowError("Dataset archive contains an unsafe path or file type.")
            normalized = str(path).casefold()
            if normalized in seen:
                raise RoboflowError("Dataset archive contains duplicate paths.")
            seen.add(normalized)
            total += member.file_size
            if member.file_size > MAX_MEMBER_BYTES or total > MAX_EXTRACT_BYTES:
                raise RoboflowError("Dataset archive exceeds the extraction size limit.")
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def download_dataset(
    workspace: str,
    project: str,
    version: int,
    format: str,
    output: str | Path,
) -> Path:
    """Download an existing version in an export format such as ``coco`` or ``folder``.

    The destination must not exist. Failed downloads leave no partial dataset.
    A pending export returns an actionable retry error, without launching training.
    """
    workspace, project = _slug(workspace, "workspace"), _slug(project, "project")
    format = _slug(format, "format")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive integer")
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Download destination already exists; choose a new directory.")
    metadata = _get_json(f"/{workspace}/{project}/{version}/{format}", _api_key())
    if metadata.get("ready") is False:
        raise RoboflowError("Dataset export is preparing. Retry the download shortly.")
    export = metadata.get("export", {})
    link = export.get("link") if isinstance(export, dict) else None
    if not isinstance(link, str):
        raise RoboflowError("Roboflow did not return a dataset download link.")
    parsed = urllib.parse.urlsplit(link)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RoboflowError("Roboflow returned an invalid dataset download link.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".roboflow-", dir=destination.parent) as temp:
            archive_path = Path(temp) / "dataset.zip"
            with (
                urllib.request.urlopen(link, timeout=60) as response,
                archive_path.open("wb") as out,
            ):
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise RoboflowError("Dataset download exceeds the 2 GiB size limit.")
                    out.write(block)
            staged = Path(temp) / "extracted"
            staged.mkdir()
            _safe_extract(archive_path, staged)
            staged.rename(destination)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise RoboflowError(f"Dataset download returned HTTP {status}; retry the export.") from None
    except (urllib.error.URLError, TimeoutError):
        raise RoboflowError("Dataset download failed; check your connection and retry.") from None
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        if isinstance(exc, RoboflowError):
            raise
        raise RoboflowError("Cannot download or extract the dataset archive.") from None
    return destination.resolve()


def _split_folders(folder: str | Path) -> dict[str, Path]:
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError("Dataset folder does not exist")
    valid = [p for p in (root / "valid", root / "validation", root / "val") if p.is_dir()]
    if len(valid) != 1:
        raise ValueError("Dataset must have exactly one valid, validation, or val directory")
    splits = {"train": root / "train", "validation": valid[0], "test": root / "test"}
    if any(not p.is_dir() or p.is_symlink() for p in splits.values()):
        raise ValueError("Dataset must contain train, validation, and test directories")
    return splits


def _image_path(path: Path, split_dir: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(split_dir.resolve()) or path.is_symlink():
        raise ValueError("Dataset image path escapes its split directory or uses a symlink")
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Dataset references a missing or unsupported image")
    return resolved


def _row(image: Path, split: str, choices: list[str], answer: int, prompt: str, category: str):
    digest = image_hash(image)
    return {
        "id": digest,
        "split": split,
        "image": str(image),
        "image_sha256": digest,
        "prompt": prompt,
        "choices": choices,
        "answer": chr(65 + answer),
        "category": category,
    }


def _write_import(rows: dict[str, list[dict]], output: str | Path) -> dict:
    for split, items in rows.items():
        if not items:
            raise ValueError(f"{split} split has no images; all three splits are required")
        if len({r["id"] for r in items}) != len(items):
            raise ValueError(f"{split} split contains duplicate image pixels")
    assert_disjoint(*rows.values())
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Import output already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=output.parent) as temp:
        staged = Path(temp) / "manifests"
        staged.mkdir()
        for split, items in rows.items():
            serialized = []
            for row in items:
                row = dict(row)
                row["image"] = os.path.relpath(row["image"], output)
                serialized.append(json.dumps(row, ensure_ascii=False))
            (staged / f"{split}.jsonl").write_text("\n".join(serialized) + "\n")
        staged.rename(output)
    return {
        "output": str(output),
        "counts": {split: len(items) for split, items in rows.items()},
        "manifests": {split: str(output / f"{split}.jsonl") for split in rows},
    }


def import_classification(folder: str | Path, output: str | Path) -> dict:
    """Import a single-label folder export using class vocabulary from training only."""
    splits = _split_folders(folder)
    classes = sorted(p.name for p in splits["train"].iterdir() if p.is_dir())
    if not 2 <= len(classes) <= 26:
        raise ValueError("Classification requires between 2 and 26 training classes")
    rows = {}
    for split, split_dir in splits.items():
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if any(p.is_symlink() for p in class_dirs):
            raise ValueError("Classification class directories cannot be symlinks")
        unseen = {p.name for p in class_dirs} - set(classes)
        if unseen:
            raise ValueError("Validation/test classes must all occur in the training split")
        rows[split] = []
        for class_dir in class_dirs:
            images = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            if split == "train" and not images:
                raise ValueError("Every training class must have at least one image")
            for path in images:
                image = _image_path(path, split_dir)
                rows[split].append(
                    _row(
                        image,
                        split,
                        classes,
                        classes.index(class_dir.name),
                        "Which category best describes this image?",
                        "classification",
                    )
                )
    result = _write_import(rows, output)
    result["classes"] = classes
    return result


def import_coco_counts(
    folder: str | Path,
    output: str | Path,
    target_class: str | None = None,
    max_count: int = 9,
    question: str | None = None,
) -> dict:
    """Turn COCO instance annotations into count MCQs, preserving all existing splits.

    This creates a counting task, not a detector or a classification dataset.
    Counts use annotations, so annotation completeness must be checked before use.
    Crowds and counts above the configured choice range are rejected, never truncated.
    """
    if not isinstance(max_count, int) or isinstance(max_count, bool) or not 1 <= max_count <= 25:
        raise ValueError("max_count must be between 1 and 25")
    splits = _split_folders(folder)
    choices = [str(n) for n in range(max_count + 1)]
    rows, training_classes = {}, None
    for split, split_dir in splits.items():
        annotation_path = split_dir / "_annotations.coco.json"
        if annotation_path.is_symlink():
            raise ValueError("COCO annotation files cannot be symlinks")
        payload = json.loads(annotation_path.read_text())
        categories = {category["id"]: category["name"] for category in payload["categories"]}
        if split == "train":
            training_classes = set(categories.values())
            if target_class is not None and target_class not in training_classes:
                raise ValueError("target_class is absent from training categories")
        elif set(categories.values()) - training_classes:
            raise ValueError("Validation/test COCO categories must occur in training categories")
        images = {image["id"]: image for image in payload["images"]}
        if len(images) != len(payload["images"]):
            raise ValueError("COCO contains duplicate image IDs")
        counts = dict.fromkeys(images, 0)
        for annotation in payload["annotations"]:
            if annotation["image_id"] not in images or annotation["category_id"] not in categories:
                raise ValueError("COCO annotation references an unknown image or category")
            if target_class is not None and categories[annotation["category_id"]] != target_class:
                continue
            if annotation.get("iscrowd", 0):
                raise ValueError("COCO crowd annotations cannot be converted into exact counts")
            counts[annotation["image_id"]] += 1
        rows[split] = []
        prompt = question or (
            f'How many objects of category "{target_class}" are visible in this image?'
            if target_class is not None
            else "How many annotated objects are in this image?"
        )
        for image_id, item in sorted(images.items(), key=lambda pair: str(pair[0])):
            if counts[image_id] > max_count:
                raise ValueError("An object count exceeds max_count; increase the choice range")
            image = _image_path(split_dir / item["file_name"], split_dir)
            rows[split].append(_row(image, split, choices, counts[image_id], prompt, "counting"))
    result = _write_import(rows, output)
    result.update({"target_class": target_class, "max_count": max_count, "question": question})
    return result
