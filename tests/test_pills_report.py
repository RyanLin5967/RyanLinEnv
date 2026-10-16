"""Report gates must distinguish development evidence, failed APIs and frozen tests."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("pycocotools")
pytest.importorskip("matplotlib")
from vision_rl.detection_eval import evaluate_coco  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pills_report", Path(__file__).resolve().parents[1] / "scripts/pills_report.py"
)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def evidence(tmp_path):
    annotations = {
        "images": [{"id": i, "width": 100, "height": 100} for i in range(4)],
        "categories": [{"id": 1, "name": "pill"}],
        "annotations": [
            {"id": i + 1, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20]}
            for i in range(4)
        ],
    }
    path = write_json(tmp_path / "annotations.json", annotations)
    predictions = [
        {"image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9} for i in range(4)
    ]
    evaluate_coco(path, predictions, output=tmp_path / "specialist.json", model={"name": "trained"})
    evaluate_coco(path, [], output=tmp_path / "frontier.json", model={"name": "frontier"})
    write_json(tmp_path / "groups.json", {"0": "a", "1": "a", "2": "b", "3": "b"})
    index = {
        "workspace": str(tmp_path),
        "title": "Measured study",
        "evaluations": [
            {
                "id": "trained",
                "label": "Trained specialist",
                "path": "specialist.json",
                "phase": "validation",
                "role": "specialist",
                "groups_path": "groups.json",
            },
            {
                "id": "frontier",
                "label": "Named frontier",
                "path": "frontier.json",
                "phase": "validation",
                "role": "frontier",
                "groups_path": "groups.json",
            },
        ],
        "comparisons": [{"a": "trained", "b": "frontier"}],
    }
    return tmp_path, index


def test_validation_can_never_be_reported_as_final_win(evidence):
    root, index = evidence
    result = reporter.build_report(
        write_json(root / "index.json", index), root / "out", bootstrap_samples=20
    )
    assert result["final_gate"]["status"] == "not_evaluated"
    assert result["comparisons"][0]["margin_target_met"]
    text = (root / "out/README.md").read_text()
    assert "Frozen test gate: not evaluated" in text
    assert "Validation — selection only" in text
    assert "No measured reports supplied" in text
    assert (root / "out/measured-results.svg").is_file()
    assert "match_cache" not in (root / "out/evidence/trained.json").read_text()
    csv = (root / "out/predictions/trained.csv").read_text()
    assert len(csv.splitlines()) == 5
    checksums = json.loads((root / "out/checksums.json").read_text())
    assert all(reporter._sha(root / "out" / name) == digest for name, digest in checksums.items())
    assert str(root) not in (root / "out/index.json").read_text()


def test_final_gate_rejects_validation_declarations(evidence):
    root, index = evidence
    index["final_gate"] = {"specialist": "trained", "frontiers": ["frontier"]}
    with pytest.raises(ValueError, match="test specialist"):
        reporter.build_report(
            write_json(root / "index.json", index), root / "out", bootstrap_samples=10
        )


@pytest.mark.parametrize("failures", [0, 1])
def test_actual_test_gate_requires_zero_api_failures(evidence, failures):
    root, index = evidence
    for evaluation in index["evaluations"]:
        evaluation["phase"] = "test"
    index["final_gate"] = {"specialist": "trained", "frontiers": ["frontier"]}
    path = root / "frontier.json"
    frontier = json.loads(path.read_text())
    frontier["errors"] = failures
    write_json(path, frontier)
    result = reporter.build_report(
        write_json(root / "index.json", index), root / "out", bootstrap_samples=10
    )
    assert result["final_gate"]["status"] == ("not_met" if failures else "met")


def test_incompatible_subsets_are_not_compared(evidence):
    root, index = evidence
    path = root / "frontier.json"
    frontier = json.loads(path.read_text())
    frontier["dataset_fingerprint"] = "different-image-subset"
    write_json(path, frontier)
    with pytest.raises(ValueError, match="fingerprint"):
        reporter.build_report(
            write_json(root / "index.json", index), root / "out", bootstrap_samples=10
        )


def test_group_manifest_pools_unknown_captures_conservatively(tmp_path):
    path = tmp_path / "groups.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"image_id": 1, "capture_hour": "capture-hour"},
                {"image_id": 2, "capture_hour": None},
                {"image_id": 3, "capture_hour": "unknown"},
            ]
        )
    )
    assert reporter.load_groups(path, [1, 2, 3]) == {
        1: "capture-hour",
        2: "unknown-capture",
        3: "unknown-capture",
    }
    with pytest.raises(ValueError, match="cover"):
        reporter.load_groups(path, [4])


def test_history_missing_entries_are_skipped_not_invented(evidence):
    root, index = evidence
    (root / "history.csv").write_text("epoch,val/AP,loss\n0,,2\n0,0.25,\n1,,1\n1,0.5,\n")
    index["histories"] = [
        {"label": "SFT", "path": "history.csv", "x": "epoch", "y": "val/AP", "x_offset": 1}
    ]
    result = reporter.build_report(
        write_json(root / "index.json", index), root / "out", bootstrap_samples=10
    )
    assert result["histories"][0]["points"] == [
        {"x": 1.0, "AP": 0.25, "training_phase": "supervised"},
        {"x": 2.0, "AP": 0.5, "training_phase": "supervised"},
    ]


def test_supplementary_evidence_removes_secret_fields(evidence):
    root, index = evidence
    write_json(
        root / "provider.json",
        {
            "model": "measured-model",
            "api_key": "DO-NOT-COPY",
            "nested": {"authorization": "DO-NOT-COPY"},
            "checked_evidence_hashes": {str(root / "private-model.pt"): "verified-digest"},
        },
    )
    index["supplementary_evidence"] = [{"id": "provider", "path": "provider.json"}]
    reporter.build_report(
        write_json(root / "index.json", index), root / "out", bootstrap_samples=10
    )
    assert "DO-NOT-COPY" not in (root / "out/evidence/provider.json").read_text()
    published = json.loads((root / "out/evidence/provider.json").read_text())
    assert published["checked_evidence_hashes"] == {
        "$WORKSPACE/private-model.pt": "verified-digest"
    }


def test_annotation_oracle_bound_cannot_enter_final_gate(evidence):
    root, index = evidence
    for evaluation in index["evaluations"]:
        evaluation["phase"] = "test"
    index["final_gate"] = {"specialist": "trained", "frontiers": ["frontier"]}
    path = root / "frontier.json"
    data = json.loads(path.read_text())
    data["synthetic_oracle_bound"] = True
    write_json(path, data)
    with pytest.raises(ValueError, match="oracle"):
        reporter.build_report(
            write_json(root / "index.json", index), root / "out", bootstrap_samples=10
        )


def test_stale_score_does_not_enter_chart_or_tables(evidence):
    root, index = evidence
    path = root / "specialist.json"
    data = json.loads(path.read_text())
    data["AP"] = 0.75
    write_json(path, data)
    with pytest.raises(ValueError, match="cached"):
        reporter.build_report(
            write_json(root / "index.json", index), root / "out", bootstrap_samples=10
        )


def test_compressed_predictions_and_journals_preserve_evidence(evidence):
    root, index = evidence
    index["compress_predictions"] = True
    index["overview"] = ["This development result is not an RL success."]
    (root / "attempts.jsonl").write_text('{"response":"native answer","api_key":"secret"}\n')
    (root / "checks.log").write_text("12 passed, 2 skipped\n")
    index["supplementary_evidence"] = [
        {"id": "attempts", "path": "attempts.jsonl", "compress": True},
        {"id": "checks", "path": "checks.log"},
    ]
    reporter.build_report(
        write_json(root / "index.json", index), root / "out", bootstrap_samples=10
    )
    with gzip.open(root / "out/predictions/trained.csv.gz", "rt") as handle:
        assert len(handle.read().splitlines()) == 5
    with gzip.open(root / "out/evidence/attempts.jsonl.gz", "rt") as handle:
        assert json.loads(handle.read()) == {"response": "native answer"}
    assert (root / "out/evidence/checks.log").read_text() == "12 passed, 2 skipped\n"
    assert "not an RL success" in (root / "out/README.md").read_text()
    checksums = json.loads((root / "out/checksums.json").read_text())
    assert all(reporter._sha(root / "out" / name) == digest for name, digest in checksums.items())
