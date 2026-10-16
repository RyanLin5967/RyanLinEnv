import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

from vision_rl.aircraft_data import PREPROCESSING_ID, aircraft_pixel_sha256
from vision_rl.local_vlm import prepare_local, sha_file


@pytest.fixture
def aircraft_probe():
    return runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "scripts/aircraft_local_probe.py"
    ))


@pytest.fixture
def aircraft_fixture(tmp_path):
    root = tmp_path / "aircraft"
    images = root / "development/images/train"
    images.mkdir(parents=True)
    audit = root / "audit"
    audit.mkdir()
    labels = [f"Aircraft{index:03}" for index in range(100)]
    rows = []
    for index, label in enumerate(labels):
        image = images / f"{index:07}.jpg"
        Image.new("RGB", (24, 48), (index, 0, 0)).save(image)
        rows.append({"image_id": f"{index:07}", "variant": label, "split": "train",
                     "path": str(image.relative_to(root)), "encoded_sha256": sha_file(image),
                     "model_rgb_sha256": aircraft_pixel_sha256(image),
                     "model_image_size": [24, 28]})
    (audit / "train-images.json").write_text(json.dumps(rows))
    (audit / "image-audit.json").write_text(json.dumps({
        "training_integrity_clear": True, "annotation_and_evaluator_exact_matches": True,
        "preprocessing_id": PREPROCESSING_ID,
        "splits": {"train": {"metadata_image_manifest_sha256": sha_file(
            audit / "train-images.json")}},
    }))
    spec = root / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": 1, "benchmark": "FGVC100", "question": "Which variant is shown?",
        "labels": labels, "split_policy": "native", "preprocessing": "lazy crop bottom20",
        "preprocessing_id": PREPROCESSING_ID,
        "sources": [{"url": "https://example.test/fixture", "sha256": "1" * 64}],
    }))
    return root, spec, tmp_path / "probe", rows, labels


def test_shared16_hash_selection_is_order_independent_and_train_only(
    aircraft_probe, aircraft_fixture,
):
    _, _, _, rows, labels = aircraft_fixture
    select = aircraft_probe["select_probe"]
    first = select(rows, labels)
    assert select(rows[::-1], labels[::-1]) == first
    assert len(first) == len({row["variant"] for row in first}) == 16
    assert select(rows, labels, seed=792) != first
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        select(rows, labels)


def test_shared_preparation_binds_crop_prompt_full_vocabulary_and_audit(
    aircraft_probe, aircraft_fixture,
):
    root, spec, output, _, labels = aircraft_fixture
    protocol = aircraft_probe["prepare"](root, spec, output)
    assert aircraft_probe["verify"](output) == protocol
    assert len(protocol["rows"]) == 16
    assert protocol["task_contract"]["labels"] == labels
    assert len({row["prompt"] for row in protocol["rows"]}) == 1
    assert all(set(row) == {"id", "split", "image", "prompt"} for row in protocol["rows"])
    assert all(row["height"] == 28 and row["source_height"] == 48
               for row in protocol["prepared_images"])
    shared = [json.loads(line) for line in (
        output / "shared-train16.jsonl").read_text().splitlines()]
    assert {row["id"] for row in shared} == {row["id"] for row in protocol["rows"]}
    assert all(set(row) == {"id", "split", "image", "label"} for row in shared)
    with pytest.raises(ValueError, match="immutable"):
        aircraft_probe["prepare"](root, spec, output)


def test_preparation_refuses_uncleared_audit_and_pixel_changes(aircraft_probe, aircraft_fixture):
    root, spec, output, _, _ = aircraft_fixture
    audit = root / "audit/image-audit.json"
    content = json.loads(audit.read_text())
    content["training_integrity_clear"] = False
    audit.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="complete clean audit"):
        aircraft_probe["prepare"](root, spec, output)
    assert not output.exists()
    content["training_integrity_clear"] = True
    audit.write_text(json.dumps(content))
    protocol = aircraft_probe["prepare"](root, spec, output)
    Image.new("RGB", (24, 48), "blue").save(protocol["rows"][0]["image"])
    with pytest.raises(ValueError, match="pixels or prompt"):
        aircraft_probe["verify"](output)


def test_local_preparation_hashes_lazy_cropped_pixels_without_copy(tmp_path):
    image = tmp_path / "source.jpg"
    Image.new("RGB", (30, 50), "red").save(image)
    record = prepare_local({"id": "a", "image": str(image), "prompt": "All classes."},
                           PREPROCESSING_ID)
    assert record["height"] == 30 and record["source_height"] == 50
    assert record["pixel_sha256"] == aircraft_pixel_sha256(image)
    assert record["image_sha256"] == sha_file(image)
    assert len(list(tmp_path.iterdir())) == 1


@pytest.fixture
def mirror_fixture(
    aircraft_probe, aircraft_fixture, tmp_path,
):
    root, spec, _, rows, labels = aircraft_fixture
    selected = aircraft_probe["select_probe"](rows, labels)
    source = tmp_path / "mirror"
    originals = source / "original-jpeg"
    originals.mkdir(parents=True)
    manifest, audit_images = [], []
    for row in selected:
        name = f"{row['image_id']}.jpg"
        (originals / name).write_bytes((root / row["path"]).read_bytes())
        manifest.append({"id": row["image_id"], "split": "train", "label": row["variant"],
                         "image": f"original-jpeg/{name}"})
        audit_images.append({"id": row["image_id"], "label": row["variant"],
                             "encoded_sha256": row["encoded_sha256"],
                             "model_rgb_sha256": row["model_rgb_sha256"],
                             "model_image_size": row["model_image_size"]})
    (source / "data_spec.json").write_bytes(spec.read_bytes())
    (source / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    selection = source / "selection.json"
    selection.write_text(json.dumps({
        "seed": 791, "selection": "prospective hash", "canonical_labels": labels,
        "selected": [{k: row[k] for k in ("id", "label", "split")} for row in manifest],
    }))
    from vision_rl import aircraft_data

    (source / "provenance.json").write_text(json.dumps({
        "selection_path": str(selection), "selection_sha256": sha_file(selection),
        "status": "fixed_train_probe_ready_original_author_image_equivalence_pending",
        "train_manifest_sha256": sha_file(source / "train.jsonl"),
        "data_spec_sha256": sha_file(source / "data_spec.json"),
        "preprocessing_source_sha256": sha_file(aircraft_data.__file__),
        "preprocessing_id": PREPROCESSING_ID, "test_images_accessed": 0,
        "images": audit_images, "mirror_repo": "fixture", "mirror_revision": "1" * 40,
        "author_original_image_byte_equivalence_verified": False,
        "full_dataset_cross_split_pixel_audit_complete": False,
    }))
    return source, manifest


def test_mirror_probe_keeps_pending_equivalence_and_exact_frozen_ids(
    aircraft_probe, mirror_fixture, tmp_path,
):
    source, manifest = mirror_fixture
    output = tmp_path / "mirror-probe"
    result = aircraft_probe["prepare_mirror"](source, output)
    assert aircraft_probe["verify"](output) == result
    assert "author equivalence pending" in result["scope"]
    assert not result["image_provenance_scope"]["author_original_image_byte_equivalence_verified"]
    assert [row["id"] for row in result["rows"]] == [row["id"] for row in manifest]
    manifest.reverse()
    (source / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    with pytest.raises(ValueError, match="provenance changed"):
        aircraft_probe["prepare_mirror"](source, tmp_path / "changed")


@pytest.fixture
def shared_v2_plan(aircraft_probe, mirror_fixture, tmp_path):
    source, manifest = mirror_fixture
    directory = tmp_path / "shared-v2-plan"
    directory.mkdir()
    spec = json.loads((source / "data_spec.json").read_text())
    prompt = (spec["question"] + "\nAllowed labels (the complete class list):\n"
              + "\n".join(spec["labels"]) + "\n" + aircraft_probe["V2_INSTRUCTION"])
    (directory / "prompt.txt").write_text(prompt)
    grammar = aircraft_probe["parser_identity"]()
    plan = {
        "ids": [row["id"] for row in manifest],
        "model_visible_pixels": {row["id"]: aircraft_pixel_sha256(source / row["image"])
                                 for row in manifest},
        "prompt_version": "aircraft-required-brief-think-v2",
        "output_grammar": grammar["name"],
        "parser_source_sha256": grammar["parser_source_sha256"],
        "prompt_sha256": sha_file(directory / "prompt.txt"),
        "inputs": {str(source / name): sha_file(source / name)
                   for name in ("data_spec.json", "train.jsonl", "provenance.json")},
    }
    path = directory / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_frozen_v2_prompt_binds_shared_plan_and_ignores_future_prompt_edits(
    aircraft_probe, mirror_fixture, shared_v2_plan, tmp_path, monkeypatch,
):
    source, _ = mirror_fixture
    output = tmp_path / "baseline-v2"
    protocol = aircraft_probe["prepare_mirror"](source, output, shared_v2_plan)
    assert protocol["schema_version"] == 2
    assert protocol["task_contract"]["grammar"] == aircraft_probe["parser_identity"]()
    assert protocol["prompt_contract"]["plan_sha256"] == sha_file(shared_v2_plan)
    assert protocol["rows"][0]["prompt"] == (shared_v2_plan.parent / "prompt.txt").read_text()
    # Main actor code may grow new prompt candidates; this preparation keeps the
    # approved text and unchanged parser instead of silently following the edit.
    monkeypatch.setitem(aircraft_probe["verify"].__globals__, "prompt_text",
                        lambda *args: "A future different prompt")
    assert aircraft_probe["verify"](output) == protocol
    (shared_v2_plan.parent / "prompt.txt").write_text("Changed approved prompt")
    with pytest.raises(ValueError, match="source audit"):
        aircraft_probe["verify"](output)


@pytest.mark.parametrize("change", ["ids", "pixels", "parser", "hint"])
def test_shared_plan_rejects_ids_pixels_parser_or_answer_hints(
    aircraft_probe, mirror_fixture, shared_v2_plan, tmp_path, change,
):
    source, _ = mirror_fixture
    plan = json.loads(shared_v2_plan.read_text())
    if change == "ids":
        plan["ids"].reverse()
    elif change == "pixels":
        plan["model_visible_pixels"][plan["ids"][0]] = "0" * 64
    elif change == "parser":
        plan["parser_source_sha256"] = "0" * 64
    else:
        prompt = shared_v2_plan.parent / "prompt.txt"
        prompt.write_text(prompt.read_text() + "\nThe first answer is Aircraft000.")
        plan["prompt_sha256"] = sha_file(prompt)
    shared_v2_plan.write_text(json.dumps(plan))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Shared prompt"):
        aircraft_probe["prepare_mirror"](source, output, shared_v2_plan)
    assert not output.exists()


def test_chunk_reports_preserve_full_scope_and_only_score_completed_scope(
    aircraft_probe, aircraft_fixture, monkeypatch, tmp_path,
):
    from types import SimpleNamespace

    root, spec, prepared, _, _ = aircraft_fixture
    protocol = aircraft_probe["prepare"](root, spec, prepared)
    shared = [json.loads(line) for line in (
        prepared / "shared-train16.jsonl").read_text().splitlines()]
    output = tmp_path / "results"
    output.mkdir()
    calls = []

    def fake_local(rows, destination, **kwargs):
        assert len(rows) == 16 and destination == output
        assert kwargs["stream_progress"]
        calls.append(kwargs["stop_after_completions"])
        count = 4 if len(calls) == 1 else 16
        with (output / "executions.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": "invocation_started"}) + "\n")
            handle.write(json.dumps({
                "event": "invocation_finished", "expected": 16, "reserved": count,
                "completed": count, "unknown": 0, "unattempted": 16 - count,
                "hard_stop": False, "cooperative_stop": count < 16,
                "completed_this_invocation": 4 if count == 4 else 12,
                "all_requests_completed": count == 16, "identity_sha256": "frozen",
            }) + "\n")
        return {
            row["id"]: ({"classification_response": "<think>Visible features.</think>" +
                         row["label"], "finish_reason": "stop"} if index < count else {
                             "error": "not_requested_cooperative_stop", "response": ""})
            for index, row in enumerate(shared)
        }

    monkeypatch.setitem(aircraft_probe["run"].__globals__, "run_local", fake_local)
    args = SimpleNamespace(input=prepared, output=output, model_directory=tmp_path / "model",
                           integrity=tmp_path / "integrity", seed=1729, memory_limit_gib=32,
                           resume=False, stream_progress=True, stop_after_completions=4)
    partial = aircraft_probe["run"](args)
    assert partial["metric"] is None and partial["coverage"]["unattempted"] == 12
    assert partial["request_errors"] == 0
    assert not partial["incomplete_scope_scored"] and not partial["benchmark_claim"]
    preserved = (output / "summary-invocation-0001.json").read_bytes()
    args.resume, args.stop_after_completions = True, None
    complete = aircraft_probe["run"](args)
    assert complete["scope"] == protocol["scope"]
    assert complete["coverage"]["all_requests_completed"] and complete["metric"] is not None
    assert (output / "summary-invocation-0001.json").read_bytes() == preserved
    assert (output / "summary-invocation-0002.json").exists()
    assert calls == [4, None]


def test_instruct_is_a_separate_frozen_profile_with_same_scope_and_parser(
    aircraft_probe, mirror_fixture, shared_v2_plan, tmp_path,
):
    source, _ = mirror_fixture
    v2_bytes = shared_v2_plan.read_bytes()
    planned = tmp_path / "instruct-plan"
    plan = aircraft_probe["derive_instruct_plan"](shared_v2_plan, planned)
    output = tmp_path / "instruct-inputs"
    protocol = aircraft_probe["prepare_mirror"](
        source, output, planned / "plan.json", profile="instruct")
    assert aircraft_probe["verify"](output) == protocol
    assert protocol["schema_version"] == 3 and protocol["profile"] == "instruct"
    assert protocol["native_profile"] == "qwen3.8-native-instruct-v1"
    assert protocol["task_contract"]["response_format"] == "answer_only"
    assert protocol["generation"]["max_tokens"] == 16384
    assert protocol["generation"]["temperature"] == 0.7
    assert protocol["generation"]["presence_penalty"] == 1.5
    assert protocol["generation"]["enable_thinking"] is False
    assert protocol["reasoning_effort"] is None
    assert [row["id"] for row in protocol["rows"]] == plan["ids"]
    assert {row["id"]: row["pixel_sha256"] for row in protocol["prepared_images"]} == (
        plan["model_visible_pixels"])
    assert len(protocol["task_contract"]["labels"]) == 100
    assert all(row["prompt"].endswith(aircraft_probe["INSTRUCT_INSTRUCTION"])
               and not row["prompt"].endswith(aircraft_probe["V2_INSTRUCTION"])
               for row in protocol["rows"])
    assert shared_v2_plan.read_bytes() == v2_bytes
    with pytest.raises(ValueError, match="immutable"):
        aircraft_probe["derive_instruct_plan"](shared_v2_plan, planned)


@pytest.mark.parametrize("invalid_plan", [None, "thinking"])
def test_instruct_cannot_reuse_required_thinking_prompt(
    aircraft_probe, mirror_fixture, shared_v2_plan, tmp_path, invalid_plan,
):
    source, _ = mirror_fixture
    output = tmp_path / "bad-mode"
    with pytest.raises(ValueError, match="prompt plan"):
        aircraft_probe["prepare_mirror"](
            source, output, shared_v2_plan if invalid_plan else None, profile="instruct")
    assert not output.exists()


@pytest.mark.parametrize("change", ["sampling", "profile", "response_format"])
def test_prepared_instruct_rejects_mode_or_generation_mutation(
    aircraft_probe, mirror_fixture, shared_v2_plan, tmp_path, change,
):
    source, _ = mirror_fixture
    planned, output = tmp_path / "instruct-plan", tmp_path / "instruct-inputs"
    aircraft_probe["derive_instruct_plan"](shared_v2_plan, planned)
    protocol = aircraft_probe["prepare_mirror"](
        source, output, planned / "plan.json", profile="instruct")
    if change == "sampling":
        protocol["generation"]["temperature"] = 1
    elif change == "profile":
        protocol["profile"] = "xhigh"
    else:
        protocol["task_contract"]["response_format"] = "think_then_answer"
    (output / "inputs.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        aircraft_probe["verify"](output)


@pytest.mark.parametrize("text,valid", [
    ("Saab 340", True), ("<answer>Saab 340</answer>", True),
    ("This is Saab 340", False), ("Explanation.\nSaab 340", False),
    ("<think>ignored</think>Unknown", False),
])
def test_instruct_uses_unchanged_canonical_parser_without_prose_repair(
    aircraft_probe, text, valid,
):
    from vision_rl.local_vlm import final_answer

    result = final_answer(text, "stop", profile="instruct")
    assert result["response"] == text
    assert (aircraft_probe["parse_label"](result["response"], ["Saab 340", "Metroliner"])
            is not None) is valid
