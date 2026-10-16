"""Canonical-label metric parity with retained actual unchanged Octave output.

These tests replay recorded native evidence; they do not install/run Octave.
Score-triplet selection (ties/duplicates) was exercised by the native fixture;
the Python scorer consumes its corresponding single-label/missing responses.
"""

import json
from pathlib import Path

import pytest

from vision_rl.vlm_local_training import classification_metrics

EVIDENCE = json.loads(
    (Path(__file__).parents[1] / "docs/fgvc-aircraft-evaluator-parity-2026-10-16.json").read_text()
)
NATIVE = {row["name"]: row for row in EVIDENCE["official_parity"]["synthetic_cases"]}


@pytest.mark.parametrize(
    "name,truth,predicted",
    [
        ("perfect", "AB", "AB"),
        ("all_wrong", "AB", "BA"),
        ("imbalanced_classes", "AAAB", "AAAA"),
        ("missing_prediction", "AB", "A?"),
        ("no_predictions", "AB", "??"),
        ("finite_ties_sorted_class", "AB", "AA"),
        ("all_negative_infinity", "AB", "??"),
        ("negative_finite_scores", "AB", "AB"),
        ("last_duplicate_pair_overwrites", "AB", "BB"),
    ],
)
def test_label_metric_matches_actual_unchanged_official_evaluator(name, truth, predicted):
    assert EVIDENCE["official_parity"]["all_cases_passed"]
    assert EVIDENCE["official_parity"]["official_source_modified"] is False
    assert EVIDENCE["source_sha256"]["evaluation.m"] == (
        "4e25f8db8a0d0163b10b167299c04c7a437a5f0de637f1e79628066ec297676a"
    )
    rows = [{"id": f"{index + 1:07d}", "label": label} for index, label in enumerate(truth)]
    predictions = {row["id"]: label if label != "?" else "" for row, label in zip(rows, predicted)}
    metrics = classification_metrics(rows, predictions, ["A", "B"])
    assert metrics["mean_class_accuracy"] == NATIVE[name]["mean_class_accuracy"]
    assert metrics["invalid_answers"] == predicted.count("?")
    if name == "imbalanced_classes":
        assert metrics["accuracy"] == 0.75
        assert metrics["mean_class_accuracy"] == 0.5
