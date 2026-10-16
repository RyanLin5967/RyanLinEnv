import json

import pytest

from vision_rl.data import load_manifest, write_json
from vision_rl.evaluation import compare, score
from vision_rl.review import make_review
from vision_rl.spatial import generate_split


def test_review_preserves_image_evidence_and_escapes_model_output(tmp_path):
    manifest = generate_split(tmp_path / "data", "test", 8, 42)
    rows = load_manifest(manifest)
    predictions = {r["id"]: r["answer"] for r in rows}
    predictions[rows[0]["id"]] = "<script>alert('wrong')</script>"
    report = tmp_path / "eval.json"
    write_json(report, score(rows, predictions, {"name": "test"}))
    output = tmp_path / "review.html"
    result = make_review(manifest, [report], output)
    assert result["disagreements"] == 1
    text = output.read_text()
    assert "data:image/png;base64," in text
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert rows[0]["id"] in text


def test_label_audit_blocks_invalid_capability_claim(tmp_path):
    manifest = generate_split(tmp_path / "data", "test", 16, 42)
    rows = load_manifest(manifest)
    good = score(rows, {r["id"]: r["answer"] for r in rows}, {"name": "perfect"})
    bad = score(rows, {r["id"]: "Z" for r in rows}, {"name": "bad"})
    a, b, audit, output = [tmp_path / f"{name}.json" for name in ("a", "b", "audit", "out")]
    write_json(a, good)
    write_json(b, bad)
    write_json(
        audit, {"dataset_fingerprint": good["dataset_fingerprint"], "status": "issues_found"}
    )
    result = compare(a, b, output, label_audit=audit)
    assert result["accuracy_difference"] == 1.0
    assert not result["capability_comparison_valid"]
    assert not result["positive_difference_ci"]
    data = json.loads(audit.read_text())
    data["dataset_fingerprint"] = "different"
    write_json(audit, data)
    with pytest.raises(ValueError, match="different dataset"):
        compare(a, b, output, label_audit=audit)


def test_transport_failures_are_not_model_capability_wins(tmp_path):
    manifest = generate_split(tmp_path / "data", "test", 8, 42)
    rows = load_manifest(manifest)
    a, b, output = [tmp_path / f"{name}.json" for name in ("a", "b", "out")]
    write_json(a, score(rows, {r["id"]: r["answer"] for r in rows}, {"name": "good"}))
    write_json(b, score(rows, {}, {"name": "failed API"}))
    result = compare(a, b, output)
    assert not result["positive_difference_ci"]
    assert not result["capability_comparison_valid"]
