import json
from pathlib import Path

import numpy as np
import pytest

from vision_rl.data import assert_disjoint, fingerprint, load_manifest
from vision_rl.evaluation import compare, evaluate, parse_answer, score
from vision_rl.spatial import generate, generate_split
from vision_rl.training import load_policy, save_policy, train, train_agent


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "data"
    generate(folder, train=128, validation=64, test=128)
    return folder


def test_training_updates_weights_and_generalizes(dataset, tmp_path):
    result = train(dataset / "train.jsonl", dataset / "validation.jsonl", tmp_path / "train")
    assert result["weights_changed"]
    initial = evaluate(
        dataset / "test.jsonl", tmp_path / "initial.json", checkpoint=tmp_path / "train/initial.npz"
    )
    trained = evaluate(
        dataset / "test.jsonl", tmp_path / "trained.json", checkpoint=tmp_path / "train/policy.npz"
    )
    assert trained["accuracy"] > 0.93
    assert trained["accuracy"] - initial["accuracy"] > 0.4
    comparison = compare(
        tmp_path / "trained.json", tmp_path / "initial.json", tmp_path / "diff.json"
    )
    assert comparison["positive_difference_ci"]


def test_agent_selects_only_by_validation(dataset, tmp_path):
    # No test manifest needs to exist for training/selection.
    (dataset / "test.jsonl").unlink()
    result = train_agent(
        dataset / "train.jsonl",
        dataset / "validation.jsonl",
        tmp_path / "agent",
        candidates=2,
        steps=100,
    )
    best = max(result["experiments"], key=lambda r: r["best_validation_accuracy"])
    assert result["selected_candidate"] == best["candidate"]
    assert result["selection_split"] == "validation"


def test_split_leakage_detects_renamed_images(dataset, tmp_path):
    train_rows = load_manifest(dataset / "train.jsonl")
    validation = load_manifest(dataset / "validation.jsonl")
    assert_disjoint(train_rows, validation)
    validation[0]["image_sha256"] = train_rows[0]["image_sha256"]
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(train_rows, validation)
    with pytest.raises(ValueError, match="expected train split"):
        train(dataset / "test.jsonl", dataset / "validation.jsonl", tmp_path / "leak")


def test_seed_and_fingerprint_reproduce_pixels(dataset, tmp_path):
    other = tmp_path / "other"
    generate(other, train=128, validation=64, test=128)
    assert fingerprint(load_manifest(dataset / "test.jsonl")) == fingerprint(
        load_manifest(other / "test.jsonl")
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        ("<answer>B</answer>", "B"),
        ("I think A", None),
        ("A or B", None),
        ("<answer>A</answer><answer>B</answer>", None),
        ("Z", None),
        ("", None),
    ],
)
def test_strict_answer_scoring(text, expected):
    assert parse_answer(text, ["yes", "no"]) == expected


def test_missing_answers_count_as_errors(dataset):
    rows = load_manifest(dataset / "test.jsonl")
    results = score(rows, {rows[0]["id"]: rows[0]["answer"]}, {"name": "partial"})
    assert results["n"] == 128
    assert results["accuracy"] == 1 / 128
    assert results["errors"] == 127
    assert results["invalid_answers"] == 127


def test_comparison_rejects_different_datasets(dataset, tmp_path):
    policy = tmp_path / "policy.npz"
    save_policy(policy, np.zeros((3, 4)))
    evaluate(dataset / "test.jsonl", tmp_path / "a.json", checkpoint=policy)
    evaluate(dataset / "validation.jsonl", tmp_path / "b.json", checkpoint=policy)
    with pytest.raises(ValueError, match="different"):
        compare(tmp_path / "a.json", tmp_path / "b.json", tmp_path / "diff.json")


@pytest.mark.parametrize(
    "weights",
    [
        np.zeros((4, 4)),
        np.full((3, 4), np.nan),
        np.zeros((3, 4), dtype=object),
        np.full((3, 4), 1e30),
    ],
)
def test_verifier_rejects_invalid_weights(tmp_path, weights):
    policy = tmp_path / "bad.npz"
    np.savez(policy, weights=weights)
    with pytest.raises(ValueError):
        load_policy(policy)


def test_duplicate_manifest_ids_fail(dataset):
    path = dataset / "test.jsonl"
    first = path.read_text().splitlines()[0]
    path.write_text(first + "\n" + first + "\n")
    with pytest.raises(ValueError, match="duplicate id"):
        load_manifest(path)


def test_response_file_duplicates_fail(dataset, tmp_path):
    pred = tmp_path / "predictions.jsonl"
    row = json.dumps({"id": "a", "response": "A"})
    pred.write_text(row + "\n" + row)
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate(dataset / "test.jsonl", tmp_path / "eval.json", predictions=pred)


def test_endpoint_does_not_send_labels_and_obeys_budget(dataset, monkeypatch):
    import httpx

    from vision_rl.evaluation import predict_endpoint, prompt_text

    rows = load_manifest(dataset / "test.jsonl")[:2]
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["messages"][0]["content"][0]["text"] == prompt_text(rows[len(requests) - 1])
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "<answer>A</answer>"}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    with pytest.raises(ValueError, match="budget"):
        predict_endpoint(rows, "https://example.invalid/v1", "model", max_requests=1)
    assert requests == []
    predictions = predict_endpoint(rows, "https://example.invalid/v1", "model", max_requests=2)
    assert len(predictions) == 2


def test_harbor_task_schema():
    pytest.importorskip("harbor")
    from harbor.models.task.task import Task

    task = Task(Path("tasks/spatial-specialist"))
    assert task.config.verifier.environment_mode.value == "separate"
    assert task.config.environment.docker_image == "vision-rl:local"
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    definition = resolve_verifier_environment_definition(task.config, task.paths)
    assert not definition.bundled_tests  # Tests are uploaded after the agent finishes.


def test_generate_does_not_overwrite(dataset):
    with pytest.raises(ValueError, match="empty"):
        generate(dataset)


def test_holdout_seed_is_separate(dataset, tmp_path):
    holdout = generate_split(tmp_path / "holdout", "test", 32, 9821231)
    assert_disjoint(load_manifest(dataset / "train.jsonl"), load_manifest(holdout))
