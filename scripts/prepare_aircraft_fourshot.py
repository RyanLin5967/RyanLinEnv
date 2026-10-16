"""Freeze four native TRAIN examples per Aircraft variant without image/model access."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_aircraft import audit, canonical, select_fewshot, sha


def make_plan(content):
    report, rows = audit(content)
    labels = report["canonical_labels"]["variant"]
    training = [row for row in rows if row["split"] == "train"]
    selection = select_fewshot(training, labels, shots=4, seed=791)
    by_id = {row["image_id"]: row for row in training}
    selected = [
        {"id": image_id, "label": by_id[image_id]["variant"], "split": "train"}
        for image_id in selection["selected_image_ids"]
    ]
    counts = dict(sorted(Counter(row["label"] for row in selected).items()))
    if len(selected) != 400 or len(labels) != 100 or set(counts.values()) != {4}:
        raise ValueError("Reviewed native400-image/100-class/four-shot contract changed")
    return {
        "schema_version": 1,
        "plan_id": "fgvc-aircraft-native-train-fourshot-seed791-v1",
        "status": "prospective_metadata_frozen_images_and_training_not_activated",
        "source": report["source"],
        "source_variant_train_sha256": report["source_file_sha256"][
            "data/images_variant_train.txt"
        ],
        "source_native_train_rows_sha256": report["splits"]["train"]["metadata_rows_sha256"],
        "selection": selection,
        "selected": selected,
        "selected_id_label_split_rows_sha256": sha(canonical(selected)),
        "canonical_labels": labels,
        "class_counts": counts,
        "selection_inputs": ["native training IDs", "canonical variant labels", "seed791"],
        "uses_image_availability": False,
        "uses_model_outputs_or_rewards": False,
        "uses_validation_or_test_for_selection": False,
        "images_downloaded_by_this_preparation": 0,
        "model_calls": 0,
        "native_validation": {"images": 3333, "classes": 100, "membership_modified": False},
        "native_test": {"images": 3333, "classes": 100, "sealed": True},
        "existing16_image_probe_modified": False,
        "existing1600_image_plan_activated": False,
        "matching_requirement": (
            "All compared training arms receive the same fixed400 IDs "
            "and full100-label vocabulary; "
            "optimization and evaluation protocols require a separate freeze before fits"
        ),
        "image_terms": report["image_terms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-archive",
        type=Path,
        default=Path("runs/aircraft-audit/reproduced/annotations.tar.gz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/fgvc-aircraft-fourshot-plan-v1.json")
    )
    args = parser.parse_args()
    plan = make_plan(args.metadata_archive.read_bytes())
    content = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text() != content:
        raise ValueError("Existing frozen four-shot plan differs; refusing replacement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(
        json.dumps(
            {
                "images": len(plan["selected"]),
                "classes": len(plan["canonical_labels"]),
                "plan_sha256": sha(content.encode()),
                "selected_rows_sha256": plan["selected_id_label_split_rows_sha256"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
