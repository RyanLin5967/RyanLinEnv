"""Offline, paired exact-label comparison without model or image access.

Predictions are labels already produced by a frozen parser, or None for a
failed/invalid/truncated answer. This module does not parse, repair or generate
answers. The caller must retain raw outputs, parser identity and model receipts.
No numerical result from this module by itself establishes a benchmark win.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _inputs(rows, predictions, labels):
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or not labels
        or any(
            not isinstance(label, str) or not label or label != label.strip() for label in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("labels must be an ordered nonempty unique canonical vocabulary")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("evaluation rows must be a nonempty sequence")
    identifiers, vocabulary = set(), set(labels)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("evaluation rows must be mappings")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("evaluation requires unique nonempty string IDs")
        if not isinstance(row.get("label"), str) or row["label"] not in vocabulary:
            raise ValueError("evaluation row contains an unknown canonical label")
        identifiers.add(identifier)
    for prediction in predictions:
        if not isinstance(prediction, Mapping) or set(prediction) - identifiers:
            raise ValueError("prediction IDs must belong to the fixed evaluation rows")
        if any(value is not None and not isinstance(value, str) for value in prediction.values()):
            raise ValueError("predictions must contain canonical label strings or None")
    return sorted(rows, key=lambda row: row["id"])


def _score(rows, predictions, labels):
    index = {label: number for number, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    counts, correct, invalid = Counter(), Counter(), Counter()
    correctness, missing = [], 0
    for row in rows:
        gold, identifier = row["label"], row["id"]
        predicted = predictions.get(identifier)
        valid = predicted in index
        counts[gold] += 1
        invalid[gold] += int(not valid)
        missing += int(identifier not in predictions)
        correct[gold] += int(predicted == gold)
        correctness.append(int(predicted == gold))
        if valid:
            confusion[index[gold], index[predicted]] += 1
    represented = [label for label in labels if counts[label]]
    per_class = {
        label: {
            "n": counts[label],
            "correct": correct[label],
            "invalid": invalid[label],
            "accuracy": correct[label] / counts[label],
        }
        for label in represented
    }
    return {
        "n": len(rows),
        "correct": sum(correct.values()),
        "micro_accuracy": sum(correct.values()) / len(rows),
        "mean_class_accuracy": sum(item["accuracy"] for item in per_class.values())
        / len(represented),
        "invalid_answers_including_missing": sum(invalid.values()),
        "missing_prediction_records": missing,
        "valid_fraction": (len(rows) - sum(invalid.values())) / len(rows),
        "per_class": per_class,
        "confusion_class_order": list(labels),
        "confusion": confusion.tolist(),
        "confusion_denominator": "Native gold-class counts, including invalid/missing answers",
    }, np.asarray(correctness, dtype=np.int8)


def _paired_bootstrap(deltas, resamples, seed):
    """Multinomial counts are exactly paired image resampling within a class.

Each paired correctness difference is -1, 0 or 1. Sampling counts of those
three outcomes is equivalent in distribution to drawing their image indices,
while avoiding an images-by-resamples allocation. It is not an unpaired test.
"""
    classes = len(deltas)
    child_seeds = np.random.SeedSequence(seed).spawn(2)
    image_rng, class_rng = (np.random.default_rng(value) for value in child_seeds)
    class_means = np.asarray([float(values.mean()) for values in deltas])
    image_draws = np.zeros(resamples, dtype=np.float64)
    for values in deltas:
        counts = np.bincount(values.astype(np.int64) + 1, minlength=3)
        draws = image_rng.multinomial(len(values), counts / len(values), size=resamples)
        image_draws += (draws[:, 2] - draws[:, 0]) / len(values) / classes
    cluster_draws = np.empty(resamples, dtype=np.float64)
    # Bounded allocation also when a caller has many canonical classes.
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        selected = class_rng.integers(0, classes, size=(stop - start, classes))
        cluster_draws[start:stop] = class_means[selected].mean(axis=1)
    image_available = all(len(values) >= 2 for values in deltas)
    return {
        "draws": resamples,
        "seed": seed,
        "confidence_level": 0.95,
        "quantile_method": "linear percentile",
        "stratified_image_ci95": np.quantile(image_draws, [0.025, 0.975]).tolist()
        if image_available
        else None,
        "stratified_image_method": (
            "Paired empirical image bootstrap independently within each fixed class, "
            "then equal-class mean; exact ternary-outcome multinomial sampling"
        ),
        "stratified_image_unavailable_reason": None
        if image_available
        else "At least one represented class has fewer than two images",
        "class_cluster_ci95": np.quantile(cluster_draws, [0.025, 0.975]).tolist()
        if classes >= 2
        else None,
        "class_cluster_method": "Paired bootstrap of complete class effects, equal class weight",
        "caveat": (
            "Intervals describe empirical image/class sampling uncertainty. They do not "
            "remove validation selection bias, training-seed uncertainty, acquisition "
            "dependence or possible pretraining contamination."
        ),
    }


def compare_classification(
    rows: Sequence[Mapping],
    candidate: Mapping[str, str | None],
    comparator: Mapping[str, str | None],
    labels: Sequence[str],
    *,
    require_all_classes: bool = True,
    resamples: int = 20_000,
    seed: int = 2718,
) -> dict:
    """Compare fixed canonical predictions on the same complete denominator.

Missing records remain incorrect and make ``all_predictions_recorded`` false;
they never vanish through an inner join. A recorded None is a completed failed
answer, not an omitted record. Unknown strings are invalid, never fuzzy-matched.
The default rejects a subset that omits any declared class. Even all classes
represented does not establish full native-split coverage: bind its manifest
and declared count in the experiment's independent evaluation receipt.
"""
    if type(resamples) is not int or not 100 <= resamples <= 1_000_000:
        raise ValueError("resamples must be an integer in [100, 1000000]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(require_all_classes) is not bool:
        raise ValueError("require_all_classes must be a boolean")
    rows = _inputs(rows, (candidate, comparator), labels)
    class_indices = {label: [] for label in labels}
    for index, row in enumerate(rows):
        class_indices[row["label"]].append(index)
    represented = [label for label in labels if class_indices[label]]
    if require_all_classes and len(represented) != len(labels):
        raise ValueError("evaluation must represent every declared class")
    candidate_score, a = _score(rows, candidate, labels)
    comparator_score, b = _score(rows, comparator, labels)
    delta = a - b
    class_deltas = [delta[class_indices[label]] for label in represented]
    macro_delta = float(np.mean([values.mean() for values in class_deltas]))
    denominator = 1.0 - comparator_score["mean_class_accuracy"]
    canonical_inputs = {
        "labels": list(labels),
        "rows": [
            {
                "id": row["id"],
                "gold": row["label"],
                "candidate": candidate.get(row["id"]),
                "candidate_recorded": row["id"] in candidate,
                "comparator": comparator.get(row["id"]),
                "comparator_recorded": row["id"] in comparator,
            }
            for row in rows
        ],
    }
    return {
        "schema_version": 1,
        "metric": "exact_label_equal_class_accuracy",
        "candidate": candidate_score,
        "comparator": comparator_score,
        "mean_class_accuracy_difference": macro_delta,
        "relative_error_reduction": macro_delta / denominator if denominator > 0 else None,
        "per_class_difference": {
            label: float(values.mean())
            for label, values in zip(represented, class_deltas, strict=True)
        },
        "declared_classes": len(labels),
        "represented_classes": len(represented),
        "full_vocabulary_covered": len(represented) == len(labels),
        "all_predictions_recorded": not (
            candidate_score["missing_prediction_records"]
            or comparator_score["missing_prediction_records"]
        ),
        "uncertainty": _paired_bootstrap(class_deltas, resamples, seed),
        "input_sha256": hashlib.sha256(
            json.dumps(canonical_inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "input_identity_scope": "Gold labels and parsed canonical outcomes; not raw model outputs",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Offline comparison only; protocol, split, model and parser evidence required",
    }


def compare_repeated_classification(
    rows: Sequence[Mapping],
    candidate: Mapping[str, Sequence[str | None]],
    comparator: Mapping[str, Sequence[str | None]],
    labels: Sequence[str],
    *,
    completions: int,
    lower_quantile: float = 0.025,
    resamples: int = 20_000,
    seed: int = 2718,
) -> dict:
    """Paired equal-class comparison when every image has a fixed number of completions.

Each image's score is its fraction of exactly correct completions (greedy and
sampled alike). The image is the resampling unit: its completions are never
resampled apart, so within-image correlation cannot narrow the interval. A
missing image record counts as ``completions`` invalid, incorrect answers.
``lower_quantile`` is the one-sided bound: 0.025 is the lower end of a two-sided
95% interval, 0.10 a one-sided 90% bound. Every declared class must be present
with at least two images, otherwise the bound is undefined and this refuses.
"""
    if type(completions) is not int or completions < 1:
        raise ValueError("completions must be a positive integer")
    if type(resamples) is not int or not 100 <= resamples <= 1_000_000:
        raise ValueError("resamples must be an integer in [100, 1000000]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(lower_quantile, float) or not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must be a float in (0, 0.5)")
    rows = _inputs(rows, (), labels)
    identifiers = {row["id"] for row in rows}
    for predictions in (candidate, comparator):
        if not isinstance(predictions, Mapping) or set(predictions) - identifiers:
            raise ValueError("prediction IDs must belong to the fixed evaluation rows")
        for answers in predictions.values():
            if (
                not isinstance(answers, Sequence)
                or isinstance(answers, (str, bytes))
                or len(answers) != completions
                or any(value is not None and not isinstance(value, str) for value in answers)
            ):
                raise ValueError("each image needs exactly the declared completions")
    vocabulary = set(labels)
    class_indices = {label: [] for label in labels}
    for index, row in enumerate(rows):
        class_indices[row["label"]].append(index)
    if any(len(indices) < 2 for indices in class_indices.values()):
        raise ValueError("every declared class needs at least two images")

    def score(predictions):
        image_scores, invalid = [], 0
        for row in rows:
            answers = predictions.get(row["id"], [None] * completions)
            invalid += sum(answer not in vocabulary for answer in answers)
            image_scores.append(sum(answer == row["label"] for answer in answers) / completions)
        values = np.asarray(image_scores, dtype=np.float64)
        per_class = {label: float(values[class_indices[label]].mean()) for label in labels}
        return values, {
            "images": len(rows),
            "completions_per_image": completions,
            "mean_class_accuracy": float(np.mean(list(per_class.values()))),
            "per_class": per_class,
            "invalid_completions_including_missing": invalid,
            "missing_image_records": sum(row["id"] not in predictions for row in rows),
            "valid_fraction": (len(rows) * completions - invalid) / (len(rows) * completions),
        }

    a, candidate_score = score(candidate)
    b, comparator_score = score(comparator)
    delta = a - b
    class_deltas = [delta[class_indices[label]] for label in labels]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    draws = np.zeros(resamples, dtype=np.float64)
    for values in class_deltas:
        selected = rng.integers(0, len(values), size=(resamples, len(values)))
        draws += values[selected].mean(axis=1) / len(labels)
    return {
        "schema_version": 1,
        "metric": "exact_label_equal_class_accuracy_over_repeated_completions",
        "candidate": candidate_score,
        "comparator": comparator_score,
        "mean_class_accuracy_difference": float(np.mean([v.mean() for v in class_deltas])),
        "per_class_difference": {
            label: float(values.mean()) for label, values in zip(labels, class_deltas, strict=True)
        },
        "uncertainty": {
            "draws": resamples,
            "seed": seed,
            "lower_quantile": lower_quantile,
            "quantile_method": "linear percentile",
            "lower_bound": float(np.quantile(draws, lower_quantile)),
            "method": (
                "Paired image bootstrap independently within each fixed class, image as "
                "the unit (its completions kept together), then equal-class mean"
            ),
        },
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
