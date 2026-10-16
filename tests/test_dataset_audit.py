"""Regression coverage for audit findings used to qualify benchmark claims."""

import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image

from vision_rl.data import image_hash

AUDIT = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "audit_dataset.py"))


def test_audit_detects_reencoded_content_and_wrong_target_count(tmp_path):
    source = tmp_path / "source"
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[5:15, 7:22] = [180, 90, 40]
    for index, (split, directory) in enumerate(AUDIT["SOURCE_SPLITS"].items()):
        folder = source / directory
        folder.mkdir(parents=True)
        # Different encoded PNG bytes and filenames, identical decoded RGB pixels.
        path = folder / f"shared_jpg.rf.variant{index}.png"
        Image.fromarray(pixels).save(path, compress_level=index)
        coco = {
            "images": [{"id": 1, "file_name": path.name}],
            "categories": [
                {"id": 0, "name": "pawns"},
                {"id": 1, "name": "black-pawn"},
                {"id": 2, "name": "white-pawn"},
            ],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]},
                {"image_id": 1, "category_id": 2, "bbox": [1, 1, 4, 4]},
            ],
        }
        (folder / "_annotations.coco.json").write_text(json.dumps(coco))
        row = {
            "id": f"row-{index}",
            "image": str(path),
            "image_sha256": image_hash(path),
            "split": split,
            "choices": ["0", "1", "2"],
            "answer": "A" if split == "test" else "C",
        }
        (manifests / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    report = AUDIT["audit"](source, manifests)
    assert report["groups"]["pixel_sha256"]["cross_split_groups"] == 1
    assert report["groups"]["file_sha256"]["cross_split_groups"] == 0
    assert report["groups"]["source_filename_group"]["cross_split_groups"] == 1
    assert report["errors"] == [{"id": "row-2", "error": "target_count_mismatch"}]
    for split in AUDIT["SPLITS"]:
        summary = report["annotation_consistency"][split]
        assert summary["cross_color_target_box_iou_at_least_0_9"] == 1
        assert summary["exact_repeated_target_box_pairs"] == 1
    assert report["cross_split_similarity"]["train:test"]["phash_hamming_at_most"]["0"] == 1
