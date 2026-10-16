"""Explicit dataset splits and content identity for image multiple-choice tasks."""

import hashlib
import json
from pathlib import Path

from PIL import Image

SPLITS = {"train", "validation", "test"}


def image_hash(path: str | Path) -> str:
    # Hash decoded pixels so a renamed/re-encoded image cannot cross splits unnoticed.
    with Image.open(path) as source:
        im = source.convert("RGB")
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def load_manifest(path: str | Path, expected_split: str | None = None) -> list[dict]:
    path = Path(path).resolve()
    rows, ids = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            for field in ("id", "split", "image", "prompt", "answer", "category"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{field} must be a nonempty string")
            if row["id"] in ids:
                raise ValueError("duplicate id")
            if row["split"] not in SPLITS:
                raise ValueError("split must be train, validation, or test")
            if expected_split and row["split"] != expected_split:
                raise ValueError(f"expected {expected_split} split")
            choices = row.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
                raise ValueError("choices must contain 2 to 26 strings")
            if any(not isinstance(c, str) or not c.strip() for c in choices):
                raise ValueError("choices must contain nonempty strings")
            if len(set(choices)) != len(choices):
                raise ValueError("choices must be unique")
            if row["answer"] not in [chr(65 + i) for i in range(len(choices))]:
                raise ValueError("answer must be a valid uppercase choice letter")
            row["image"] = str((path.parent / row["image"]).resolve())
            row["image_sha256"] = image_hash(row["image"])
            ids.add(row["id"])
            rows.append(row)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from None
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def assert_disjoint(*datasets: list[dict]) -> None:
    ids, pixels = set(), set()
    for rows in datasets:
        new_ids = {r["id"] for r in rows}
        new_pixels = {r.get("image_sha256") or image_hash(r["image"]) for r in rows}
        if ids & new_ids or pixels & new_pixels:
            raise ValueError("dataset leakage: IDs or image pixels overlap across splits")
        ids.update(new_ids)
        pixels.update(new_pixels)


def fingerprint(rows: list[dict]) -> str:
    fields = ("id", "split", "prompt", "choices", "answer", "category", "image_sha256")
    canonical = [{k: r[k] for k in fields} for r in sorted(rows, key=lambda r: r["id"])]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
