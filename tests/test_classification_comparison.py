import pytest

from vision_rl.classification_comparison import compare_classification


def rows(labels):
    return [{"id": str(index), "label": label} for index, label in enumerate(labels)]


def test_macroaccuracy_preserves_native_denominators_and_invalid_answers():
    gold = rows(["A", "A", "A", "B"])
    result = compare_classification(
        gold,
        {"0": "A", "1": "A", "2": "A", "3": None},
        {"0": "A", "1": "wrong", "2": "B", "3": "B"},
        ["A", "B"],
        resamples=1000,
    )
    assert result["candidate"]["micro_accuracy"] == 0.75
    assert result["candidate"]["mean_class_accuracy"] == 0.5
    assert result["comparator"]["mean_class_accuracy"] == pytest.approx(2 / 3)
    assert result["mean_class_accuracy_difference"] == pytest.approx(-1 / 6)
    assert result["candidate"]["confusion"] == [[3, 0], [0, 0]]
    assert result["candidate"]["invalid_answers_including_missing"] == 1
    assert result["all_predictions_recorded"]
    # One image in B cannot estimate that class's image-sampling uncertainty.
    assert result["uncertainty"]["stratified_image_ci95"] is None


def test_missing_records_are_errors_not_an_inner_join_or_completed_run():
    gold = rows(["A", "A", "B", "B"])
    partial = {"0": "A", "2": "B"}
    explicit = {**partial, "1": None, "3": None}
    complete = {str(i): label for i, label in enumerate(["A", "A", "B", "B"])}
    result = compare_classification(gold, partial, complete, ["A", "B"], resamples=1000)
    other = compare_classification(gold, explicit, complete, ["A", "B"], resamples=1000)
    assert result["candidate"]["n"] == 4
    assert result["candidate"]["mean_class_accuracy"] == 0.5
    assert result["candidate"]["missing_prediction_records"] == 2
    assert not result["all_predictions_recorded"]
    assert other["all_predictions_recorded"]
    assert result["mean_class_accuracy_difference"] == other["mean_class_accuracy_difference"]
    assert result["input_sha256"] != other["input_sha256"]


def test_pairing_eliminates_spurious_uncertainty_for_identical_models():
    gold = rows(["A", "A", "B", "B"])
    predictions = {"0": "A", "1": "B", "2": "A", "3": "B"}
    result = compare_classification(gold, predictions, predictions, ["A", "B"], resamples=1000)
    assert result["mean_class_accuracy_difference"] == 0
    assert result["uncertainty"]["stratified_image_ci95"] == [0, 0]
    assert result["uncertainty"]["class_cluster_ci95"] == [0, 0]


def test_bootstrap_respects_equal_class_weight_and_distinct_sensitivity():
    gold = rows(["A", "A", "B", "B"])
    a = {"0": "A", "1": "A", "2": "A", "3": "A"}
    b = {"0": "B", "1": "B", "2": "B", "3": "B"}
    result = compare_classification(gold, a, b, ["A", "B"], resamples=1000)
    assert result["per_class_difference"] == {"A": 1, "B": -1}
    # All images inside a class have the same effect, so only class resampling varies.
    assert result["uncertainty"]["stratified_image_ci95"] == [0, 0]
    assert result["uncertainty"]["class_cluster_ci95"] == [-1, 1]


def test_stratified_distribution_has_known_small_sample_endpoints():
    gold = rows(["A", "A", "B", "B"])
    a = {"0": "A", "1": "B", "2": "B", "3": "A"}
    b = dict.fromkeys(a)
    result = compare_classification(gold, a, b, ["A", "B"], resamples=20_000)
    # Each of the four independent draws succeeds with p=.5: P(all/none)=1/16.
    assert result["mean_class_accuracy_difference"] == 0.5
    assert result["uncertainty"]["stratified_image_ci95"] == [0, 1]
    assert result["uncertainty"]["class_cluster_ci95"] == [0.5, 0.5]


def test_row_order_is_irrelevant_and_evidence_hashes_are_reproducible():
    gold = rows(["A", "A", "B", "B"])
    a = {"0": "A", "1": "B", "2": "B", "3": None}
    b = {"0": "A", "1": "A", "2": "A", "3": "B"}
    expected = compare_classification(gold, a, b, ["A", "B"], resamples=1000)
    actual = compare_classification(gold[::-1], a, b, ["A", "B"], resamples=1000)
    assert actual == expected


@pytest.mark.parametrize(
    "gold,prediction,labels,match",
    [
        (rows(["A", "A"]), {}, ["A", "B"], "every declared class"),
        ([{"id": "1", "label": "A"}] * 2, {}, ["A"], "unique"),
        (rows(["A"]), {}, ["A", "A"], "unique canonical"),
        (rows(["X"]), {}, ["A", "B"], "unknown canonical"),
        (rows(["A"]), {"extra": "A"}, ["A"], "prediction IDs"),
        (rows(["A"]), {"0": True}, ["A"], "label strings or None"),
        (rows(["A", "B"]), {}, "AB", "ordered nonempty"),
        (rows(["A", "B"]), {}, {"A", "B"}, "ordered nonempty"),
        (rows([" A"]), {}, [" A"], "canonical vocabulary"),
        (iter(rows(["A"])), {}, ["A"], "nonempty sequence"),
    ],
)
def test_mismatched_data_or_invalid_contract_is_rejected(gold, prediction, labels, match):
    with pytest.raises(ValueError, match=match):
        compare_classification(gold, prediction, prediction, labels, resamples=1000)


def test_partial_vocabulary_is_explicit_and_perfect_control_has_no_error_reduction():
    result = compare_classification(
        rows(["A", "A"]),
        {"0": "A", "1": "A"},
        {"0": "A", "1": "A"},
        ["A", "B"],
        require_all_classes=False,
        resamples=1000,
    )
    assert not result["full_vocabulary_covered"]
    assert result["represented_classes"] == 1
    assert result["relative_error_reduction"] is None
    assert result["uncertainty"]["class_cluster_ci95"] is None


def test_labels_are_exact_and_never_repaired_or_case_folded():
    result = compare_classification(
        rows(["A", "A", "B", "B"]),
        {"0": " A", "1": "a", "2": "<answer>B</answer>", "3": "B"},
        dict.fromkeys(["0", "1", "2", "3"]),
        ["A", "B"],
        resamples=1000,
    )
    assert result["candidate"]["correct"] == 1
    assert result["candidate"]["invalid_answers_including_missing"] == 3
