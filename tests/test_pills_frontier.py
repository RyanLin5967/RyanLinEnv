"""Final-test protocol gates must fail before making any hosted request."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_rl import frontier

WRAPPER = runpy.run_path(str(Path(__file__).parents[1] / "scripts/pills_frontier.py"))


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / "source"
    for split_index, split in enumerate(("train", "valid", "test")):
        directory = source / split
        directory.mkdir(parents=True)
        images, annotations = [], []
        for index in range(2):
            name = f"image-{index}.png"
            Image.new("RGB", (32, 24), (30 * split_index, 50 * index, 80)).save(directory / name)
            images.append({"id": index, "file_name": name, "width": 32, "height": 24})
            annotations.append(
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [4, 3, 8, 6],
                    "area": 48,
                    "iscrowd": 0,
                }
            )
        (directory / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": 1, "name": "pill"}],
                }
            )
        )
    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"actual frozen checkpoint bytes")
    policy = tmp_path / "policy.json"
    policy.write_text('{"choice": 1}')
    frozen_path = tmp_path / "freeze.json"
    settings = [
        {
            "provider": "gemini",
            "model": "gemini-4-argon",
            "effort": "high",
            "max_output_tokens": 16384,
            "prompt": "rich",
            "seed": 1729,
            "examples": 1,
        }
    ]
    record = WRAPPER["create_freeze"](
        source, checkpoint, settings, {"AP_margin_min": 0.2}, frozen_path, [policy]
    )
    prepare = SimpleNamespace(
        source=source,
        output=tmp_path / "prepared",
        split="test",
        count=None,
        seed=1729,
        prompt="rich",
        examples=1,
        freeze=frozen_path,
    )
    run = SimpleNamespace(
        input=prepare.output,
        output=tmp_path / "results",
        provider="gemini",
        model="gemini-4-argon",
        effort="high",
        max_tokens=16384,
        concurrency=2,
        resume=False,
    )
    return source, checkpoint, policy, frozen_path, record, prepare, run


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def infer(rows, output, **kwargs):
        calls.append({"rows": rows, **kwargs})
        if not output.exists():
            output.mkdir(parents=True)
        return {
            row["id"]: {"response": '[{"label":"pill","box_2d":[125,125,375,375]}]'} for row in rows
        }

    monkeypatch.setattr(frontier, "run_frontier", infer)
    return calls


def prepared(experiment):
    *_, preparation, args = experiment
    WRAPPER["prepare"](preparation)
    return args


def mutate_protocol(args, mutator):
    path = args.input / "inputs.json"
    protocol = json.loads(path.read_text())
    mutator(protocol)
    path.write_text(json.dumps(protocol))


def test_freeze_pins_checkpoint_policy_models_and_exact_test_contents(experiment):
    source, checkpoint, policy, _, record, _, _ = experiment
    assert record["checkpoint_sha256"] == WRAPPER["sha"](checkpoint)
    assert {item["path"] for item in record["artifacts"]} == {str(checkpoint), str(policy)}
    assert record["test_annotation_sha256"] == WRAPPER["sha"](
        source / "test/_annotations.coco.json"
    )
    entry = record["baselines"][0]
    assert len(entry["protocol"]["rows"]) == 2
    assert len(entry["protocol"]["examples"]) == 1
    assert "image" not in entry["protocol"]["rows"][0]
    assert entry["config"]["effort"] == "high"
    assert len(record["implementation"]) == 3


def test_final_exact_protocol_scores_all_native_images(experiment, transport):
    pytest.importorskip("pycocotools")
    args = prepared(experiment)
    WRAPPER["run"](args)
    assert len(transport) == 1
    report = json.loads((args.output / "evaluation.json").read_text())
    assert report["n"] == 2 and report["AP"] == pytest.approx(1)
    assert report["errors"] == 0
    assert report["model"]["frozen_experiment"]["sha256"] == WRAPPER["sha"](experiment[3])


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "anthropic"),
        ("model", "other-model"),
        ("effort", "low"),
        ("max_tokens", 32768),
        ("concurrency", 1),
        ("max_transient_retries", 1),
        ("retry_delay_seconds", 3),
    ],
)
def test_final_settings_must_match_freeze_before_any_api_call(experiment, transport, field, value):
    args = prepared(experiment)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="not frozen"):
        WRAPPER["run"](args)
    assert transport == []
    assert not args.output.exists()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.update(prompt_mode="minimal"),
        lambda p: p["rows"][0].update(prompt="different prompt"),
        lambda p: p["rows"].reverse(),
        lambda p: p["examples"][0].update(response="[]"),
        lambda p: p.update(examples=[]),
        lambda p: p.update(sampling_seed=99),
    ],
)
def test_prepared_test_edits_are_rejected(experiment, transport, mutator):
    args = prepared(experiment)
    mutate_protocol(args, mutator)
    with pytest.raises(ValueError, match="not frozen"):
        WRAPPER["run"](args)
    assert transport == []


def test_missing_test_image_cannot_shrink_denominator(experiment, transport):
    args = prepared(experiment)
    mutate_protocol(args, lambda p: p["rows"].pop())
    with pytest.raises(ValueError, match="every native test image"):
        WRAPPER["run"](args)
    assert transport == []


def test_freeze_cannot_be_removed_to_bypass_final_gate(experiment, transport):
    args = prepared(experiment)
    mutate_protocol(args, lambda p: p.update(frozen_experiment=None))
    with pytest.raises(ValueError, match="requires a verified freeze"):
        WRAPPER["run"](args)
    assert transport == []


@pytest.mark.parametrize("index", [1, 2])
def test_changed_specialist_checkpoint_or_policy_blocks_frontier_calls(
    experiment, transport, index
):
    args = prepared(experiment)
    experiment[index].write_bytes(b"changed after freeze")
    with pytest.raises(ValueError, match="checkpoint changed|artifact or evaluator"):
        WRAPPER["run"](args)
    assert transport == []


def test_prepare_rejects_changed_fewshot_selection(experiment):
    *_, preparation, _ = experiment
    preparation.examples = 0
    with pytest.raises(ValueError, match="differ from freeze"):
        WRAPPER["prepare"](preparation)
    assert not preparation.output.exists()


def test_prepare_requires_real_matching_checkpoint_bytes(experiment):
    _, checkpoint, _, _, _, preparation, _ = experiment
    checkpoint.write_bytes(b"replacement model")
    with pytest.raises(ValueError, match="checkpoint changed"):
        WRAPPER["prepare"](preparation)


def test_same_frozen_baseline_cannot_use_another_output_for_a_repeat(experiment, transport):
    pytest.importorskip("pycocotools")
    args = prepared(experiment)
    WRAPPER["run"](args)
    original = args.output
    args.output = original.with_name("retry-in-new-directory")
    with pytest.raises(ValueError, match="already reserved"):
        WRAPPER["run"](args)
    assert len(transport) == 1
    args.output = original
    with pytest.raises(ValueError, match="already reserved"):
        WRAPPER["run"](args)
    args.resume = True
    WRAPPER["run"](args)
    assert len(transport) == 2 and transport[-1]["resume"] is True


def test_final_transport_and_parse_errors_remain_in_native_denominator(experiment, monkeypatch):
    pytest.importorskip("pycocotools")
    args = prepared(experiment)

    def failed(rows, output, **kwargs):
        output.mkdir()
        return {
            rows[0]["id"]: {"error": "HTTP 429"},
            rows[1]["id"]: {"response": "invalid bounding boxes"},
        }

    monkeypatch.setattr(frontier, "run_frontier", failed)
    WRAPPER["run"](args)
    report = json.loads((args.output / "evaluation.json").read_text())
    assert report["n"] == 2 and report["AP"] == 0
    assert report["errors"] == 2
    assert len(report["request_or_parse_failures"]) == 2


def test_protocol_changes_during_requests_cannot_change_scoring(experiment, monkeypatch):
    args = prepared(experiment)

    def modified(rows, output, **kwargs):
        output.mkdir()
        mutate_protocol(args, lambda p: p.update(prompt_mode="minimal"))
        return {}

    monkeypatch.setattr(frontier, "run_frontier", modified)
    with pytest.raises(ValueError, match="changed while frontier requests"):
        WRAPPER["run"](args)
    assert not (args.output / "evaluation.json").exists()


def test_annotation_changes_during_requests_cannot_change_scoring(experiment, monkeypatch):
    args = prepared(experiment)

    def modified(rows, output, **kwargs):
        output.mkdir()
        path = experiment[0] / "test/_annotations.coco.json"
        dataset = json.loads(path.read_text())
        dataset["annotations"][0]["bbox"] = [3, 3, 8, 6]
        path.write_text(json.dumps(dataset))
        return {}

    monkeypatch.setattr(frontier, "run_frontier", modified)
    with pytest.raises(ValueError, match="annotations changed"):
        WRAPPER["run"](args)
    assert not (args.output / "evaluation.json").exists()


def test_validation_does_not_require_or_reserve_final_freeze(experiment, transport):
    pytest.importorskip("pycocotools")
    *_, preparation, args = experiment
    preparation.split, preparation.freeze, preparation.count = "validation", None, 1
    WRAPPER["prepare"](preparation)
    WRAPPER["run"](args)
    assert len(transport) == 1 and len(transport[0]["rows"]) == 1
    assert not experiment[3].with_name("freeze.json.runs.json").exists()


def test_native_image_paths_cannot_escape_split(experiment):
    source, *_, preparation, _ = experiment
    annotation = source / "test/_annotations.coco.json"
    dataset = json.loads(annotation.read_text())
    dataset["images"][0]["file_name"] = "../valid/image-0.png"
    annotation.write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="inside its native split"):
        WRAPPER["prepare"](preparation)


@pytest.mark.parametrize("retry_count,delay", [(0, 0), (1, 2), (2, 60)])
def test_declared_retry_policy_is_forwarded_and_all_attempts_are_budgeted(
    experiment, transport, retry_count, delay
):
    pytest.importorskip("pycocotools")
    source, checkpoint, policy, frozen_path, record, preparation, args = experiment
    config = {
        **record["baselines"][0]["config"],
        "max_transient_retries": retry_count,
        "retry_delay_seconds": delay,
    }
    preparation.freeze = frozen_path.with_name("retry-freeze.json")
    frozen = WRAPPER["create_freeze"](
        source, checkpoint, [config], {"AP_margin_min": 0.2}, preparation.freeze, [policy]
    )
    assert frozen["baselines"][0]["config"]["max_transient_retries"] == retry_count
    assert frozen["baselines"][0]["config"]["retry_delay_seconds"] == delay
    WRAPPER["prepare"](preparation)
    args.max_transient_retries, args.retry_delay_seconds = retry_count, delay
    WRAPPER["run"](args)
    assert transport[0]["max_requests"] == 2 * (1 + retry_count)
    assert transport[0]["max_transient_retries"] == retry_count
    assert transport[0]["retry_delay_seconds"] == delay
    report = json.loads((args.output / "evaluation.json").read_text())
    assert report["n"] == 2
    assert report["model"]["request_budget"] == 2 * (1 + retry_count)
    assert report["model"]["max_transient_retries"] == retry_count


@pytest.mark.parametrize("value", [-1, 3, True, 1.5])
def test_freeze_rejects_invalid_retry_counts(value):
    with pytest.raises(ValueError, match="max_transient_retries"):
        WRAPPER["baseline_config"](
            {"provider": "gemini", "model": "gemini-4-argon", "max_transient_retries": value}
        )


@pytest.mark.parametrize("value", [-0.01, 60.01, float("nan"), float("inf"), True, "2"])
def test_freeze_rejects_invalid_retry_delays(value):
    with pytest.raises(ValueError, match="retry_delay_seconds"):
        WRAPPER["baseline_config"](
            {"provider": "gemini", "model": "gemini-4-argon", "retry_delay_seconds": value}
        )


def test_invalid_runtime_retry_policy_fails_before_any_network_request(experiment, transport):
    args = prepared(experiment)
    args.max_transient_retries = 3
    with pytest.raises(ValueError, match="max_transient_retries"):
        WRAPPER["run"](args)
    assert transport == []


def astra_validation(experiment, examples=1):
    *_, preparation, args = experiment
    preparation.split = "validation"
    preparation.freeze = None
    preparation.count = 2
    preparation.examples = examples
    preparation.provider = "openai"
    args.provider, args.model, args.effort = "openai", "gpt-6-astra", "high"
    WRAPPER["prepare"](preparation)
    return args, json.loads((preparation.output / "inputs.json").read_text())


def astra_answers(protocol):
    return {
        row["id"]: {
            "response": '{"detections":[{"label":"pill","box_2d":[4,3,12,9]}]}'
        }
        for row in protocol["rows"]
    }


def test_astra_protocol_preserves_selected_images_but_uses_native_pixel_contract(experiment):
    *_, preparation, _ = experiment
    preparation.split, preparation.freeze, preparation.count = "validation", None, 2
    legacy = WRAPPER["build_protocol"](preparation)
    _, native = astra_validation(experiment)
    assert [row["id"] for row in native["rows"]] == [row["id"] for row in legacy["rows"]]
    assert legacy["schema_version"] == 1
    assert not {"provider", "coordinate_format", "image_detail", "response_format"} & legacy.keys()
    assert json.loads(legacy["examples"][0]["response"])[0]["box_2d"] == [125, 125, 375, 375]
    assert native["schema_version"] == 2
    assert native["provider"] == "openai"
    assert native["coordinate_format"] == "xyxy_pixels"
    assert native["image_detail"] == "original"
    assert "32x24 pixel image" in native["rows"][0]["prompt"]
    assert json.loads(native["examples"][0]["response"]) == {
        "detections": [{"label": "pill", "box_2d": [4, 3, 12, 9]}]
    }
    schema = native["response_format"]["format"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["detections"]["items"]["properties"]["label"]["enum"] == [
        "pill"
    ]


def test_astra_predictions_can_be_rescored_offline_with_exact_coordinate_protocol(experiment):
    pytest.importorskip("pycocotools")
    _, protocol = astra_validation(experiment)
    predictions = astra_answers(protocol)
    report = WRAPPER["score_predictions"](protocol, predictions, {"provider": "openai"})
    assert report["n"] == 2 and report["AP"] == pytest.approx(1)
    assert report["errors"] == 0 and len(report["detections"]) == 2
    predictions[protocol["rows"][0]["id"]] = {"response": "invalid JSON"}
    report = WRAPPER["score_predictions"](protocol, predictions, {"provider": "openai"})
    assert report["n"] == 2 and report["errors"] == 1
    assert report["AP"] < 0.6
    with pytest.raises(ValueError, match="provider"):
        WRAPPER["score_predictions"](protocol, predictions, {"provider": "gemini"})


def test_astra_run_forwards_original_detail_and_strict_schema(experiment, monkeypatch):
    pytest.importorskip("pycocotools")
    args, protocol = astra_validation(experiment)
    calls = []

    def infer(rows, output, **kwargs):
        output.mkdir()
        calls.append(kwargs)
        return astra_answers(protocol)

    monkeypatch.setattr(frontier, "run_frontier", infer)
    WRAPPER["run"](args)
    assert calls[0]["image_detail"] == "original"
    assert calls[0]["response_format"] == protocol["response_format"]
    assert calls[0]["examples"] == protocol["examples"]
    report = json.loads((args.output / "evaluation.json").read_text())
    assert report["AP"] == pytest.approx(1)
    assert report["model"]["coordinate_format"] == "xyxy_pixels"


def test_wrong_provider_cannot_score_a_different_coordinate_contract(experiment, transport):
    args = prepared(experiment)
    args.provider, args.model = "openai", "gpt-6-astra"
    with pytest.raises(ValueError, match="prepare with --provider openai"):
        WRAPPER["run"](args)
    assert transport == []


@pytest.mark.parametrize(
    "field,value", [("image_detail", "high"), ("coordinate_format", "yxyx_1000")]
)
def test_astra_frozen_protocol_rejects_coordinate_or_image_detail_changes(
    experiment, transport, field, value
):
    source, checkpoint, policy, frozen_path, _, preparation, args = experiment
    preparation.provider = "openai"
    preparation.freeze = frozen_path.with_name("astra-freeze.json")
    args.provider, args.model, args.effort = "openai", "gpt-6-astra", "high"
    frozen = WRAPPER["create_freeze"](
        source,
        checkpoint,
        [{"provider": "openai", "model": "gpt-6-astra", "examples": 1}],
        {"AP_margin_min": 0.2},
        preparation.freeze,
        [policy],
    )
    config = frozen["baselines"][0]["config"]
    assert config["image_detail"] == "original" and config["coordinate_format"] == "xyxy_pixels"
    WRAPPER["prepare"](preparation)
    mutate_protocol(args, lambda p: p.update({field: value}))
    with pytest.raises(ValueError, match="not frozen|Invalid explicit OpenAI"):
        WRAPPER["run"](args)
    assert transport == []


def test_astra_runtime_image_detail_must_match_prepared_protocol(experiment, transport):
    args, _ = astra_validation(experiment)
    args.image_detail = "high"
    with pytest.raises(ValueError, match="Image detail differs"):
        WRAPPER["run"](args)
    assert transport == []


def test_astra_response_schema_cannot_be_changed_after_preparation(experiment, transport):
    args, _ = astra_validation(experiment)
    mutate_protocol(args, lambda p: p["response_format"]["format"].update(strict=False))
    with pytest.raises(ValueError, match="Invalid explicit OpenAI"):
        WRAPPER["run"](args)
    assert transport == []


def test_gemini_structured_protocol_matches_native_schema_and_preserves_legacy(experiment):
    *_, preparation, _ = experiment
    preparation.split, preparation.freeze = "validation", None
    legacy = WRAPPER["build_protocol"](preparation)
    preparation.structured_output = False
    assert WRAPPER["build_protocol"](preparation) == legacy
    preparation.structured_output = True
    native = WRAPPER["build_protocol"](preparation)
    assert native["schema_version"] == 2 and native["provider"] == "gemini"
    assert native["coordinate_format"] == "yxyx_1000"
    assert native["structured_output"] is True
    assert "confidence" not in native["prompt"] and "box_2d" in native["prompt"]
    assert native["response_schema"]["items"]["required"] == ["box_2d", "label"]
    assert native["response_schema"]["items"]["properties"]["label"]["enum"] == ["pill"]
    example = json.loads(native["examples"][0]["response"])[0]
    assert list(example) == ["box_2d", "label"]
    assert example["box_2d"] == [125, 125, 375, 375]
    assert all(type(v) is int for v in example["box_2d"])
    assert [row["id"] for row in native["rows"]] == [row["id"] for row in legacy["rows"]]


def test_gemini_structured_final_freeze_and_native_requests(experiment, transport):
    pytest.importorskip("pycocotools")
    source, checkpoint, policy, frozen_path, record, preparation, args = experiment
    preparation.structured_output = True
    preparation.freeze = frozen_path.with_name("gemini-structured-freeze.json")
    config = {**record["baselines"][0]["config"], "structured_output": True}
    frozen = WRAPPER["create_freeze"](
        source, checkpoint, [config], {"AP_margin_min": 0.2}, preparation.freeze, [policy]
    )
    assert frozen["baselines"][0]["config"]["coordinate_format"] == "yxyx_1000"
    WRAPPER["prepare"](preparation)
    WRAPPER["run"](args)
    assert transport[0]["response_schema"] == WRAPPER["gemini_response_schema"](
        [{"id": 1, "name": "pill"}]
    )
    report = json.loads((args.output / "evaluation.json").read_text())
    assert report["AP"] == pytest.approx(1)
    assert report["model"]["structured_output"] is True


def test_gemini_structured_settings_and_frozen_schema_cannot_silently_change(experiment, transport):
    *_, preparation, args = experiment
    preparation.split, preparation.freeze = "validation", None
    preparation.structured_output = True
    WRAPPER["prepare"](preparation)
    args.provider = "anthropic"
    with pytest.raises(ValueError, match="provider-specific"):
        WRAPPER["run"](args)
    args.provider = "gemini"
    mutate_protocol(args, lambda p: p["response_schema"]["items"].update(required=[]))
    with pytest.raises(ValueError, match="Invalid explicit Gemini"):
        WRAPPER["run"](args)
    assert transport == []


@pytest.mark.parametrize(
    "provider,model", [("openai", "gpt-6-astra"), ("anthropic", "claude-opus-5-5")]
)
def test_gemini_structured_flag_cannot_configure_other_providers(provider, model):
    with pytest.raises(ValueError, match="Gemini-only"):
        WRAPPER["baseline_config"](
            {"provider": provider, "model": model, "structured_output": True}
        )


@pytest.fixture(params=["gemini", "openai"])
def complement(experiment, request, monkeypatch):
    *_, preparation, args = experiment
    preparation.provider = request.param
    preparation.structured_output = request.param == "gemini"
    preparation.split, preparation.freeze, preparation.count = "validation", None, 1
    WRAPPER["prepare"](preparation)
    original = json.loads((preparation.output / "inputs.json").read_text())
    native = (
        {
            "modelVersion": "gemini-4-argon", "responseId": "test-native",
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "[]"}]}}],
        }
        if request.param == "gemini" else {
            "model": "gpt-6-astra", "id": "test-native", "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
        }
    )
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-complement-key")
    monkeypatch.setattr(frontier, "_request", lambda *a: (200, native, {}))
    frontier.run_frontier(
        original["rows"], args.output, provider=request.param,
        examples=original["examples"], max_requests=3, max_transient_retries=2,
        image_detail=original.get("image_detail"), response_format=original.get("response_format"),
        response_schema=original.get("response_schema"),
    )
    command = SimpleNamespace(
        input=preparation.output, from_run=args.output,
        output=preparation.output.with_name("complement-inputs"),
    )
    WRAPPER["prepare_complement"](command)
    args.input, args.output = command.output, args.output.with_name("complement-results")
    args.provider, args.model = request.param, frontier.PROVIDERS[request.param][0]
    args.max_transient_retries = 2
    return command, args, original


def test_complement_derives_exact_remaining_images_and_unchanged_settings(complement):
    command, _, original = complement
    protocol = json.loads((command.output / "inputs.json").read_text())
    assert len(protocol["rows"]) == 1
    assert {r["id"] for r in original["rows"]}.isdisjoint(r["id"] for r in protocol["rows"])
    assert len({r["id"] for r in [*original["rows"], *protocol["rows"]]}) == 2
    assert protocol["examples"] == original["examples"]
    assert WRAPPER["complement_settings"](protocol) == WRAPPER["complement_settings"](original)
    assert protocol["validation_complement"]["excluded_ids"] == [original["rows"][0]["id"]]
    assert WRAPPER["verify_protocol"](protocol) is None


@pytest.mark.parametrize(
    "field,value", [("effort", "low"), ("max_tokens", 32768), ("concurrency", 1)]
)
def test_complement_runtime_cannot_change_original_model_settings(
    complement, transport, field, value
):
    _, args, _ = complement
    setattr(args, field, value)
    with pytest.raises(ValueError, match="runtime settings"):
        WRAPPER["run"](args)
    assert transport == []


def test_complement_run_uses_original_native_schema_and_new_attempt_budget(complement, transport):
    pytest.importorskip("pycocotools")
    _, args, _ = complement
    WRAPPER["run"](args)
    assert transport[0]["max_requests"] == 3
    assert transport[0]["max_transient_retries"] == 2
    assert len(transport[0]["rows"]) == 1
    assert transport[0]["timeout_seconds"] == 240
    assert transport[0]["response_format"] or transport[0]["response_schema"]


@pytest.mark.parametrize("mutation", ["overlap", "prompt", "excluded", "settings", "source"])
def test_complement_rejects_drift_before_hosted_calls(complement, transport, mutation):
    command, args, original = complement
    if mutation == "source":
        path = command.input / "inputs.json"
        path.write_text(path.read_text() + " ")
    else:
        def change(p):
            if mutation == "overlap":
                p["rows"] = original["rows"]
            elif mutation == "prompt":
                p["rows"][0]["prompt"] += " changed"
            elif mutation == "excluded":
                p["validation_complement"]["excluded_ids"] = []
            else:
                p["validation_complement"]["request_config"]["effort"] = "low"
        mutate_protocol(args, change)
    with pytest.raises(ValueError):
        WRAPPER["run"](args)
    assert transport == []


def test_complement_cannot_be_prepared_from_test_or_nested_complement(experiment, complement):
    command, _, _ = complement
    command.input = command.output
    command.output = command.output.with_name("nested")
    with pytest.raises(ValueError, match="original validation discovery"):
        WRAPPER["prepare_complement"](command)
    *_, preparation, _ = experiment
    original = json.loads((preparation.output / "inputs.json").read_text())
    original["split"] = "test"
    (preparation.output / "inputs.json").write_text(json.dumps(original))
    command.input = preparation.output
    with pytest.raises(ValueError, match="original validation discovery"):
        WRAPPER["prepare_complement"](command)


@pytest.mark.parametrize("provider", ["qwen", "kimi"])
def test_openrouter_prepare_examples_and_scoring_use_explicit_provider_contract(
    experiment, monkeypatch, provider
):
    pytest.importorskip("pycocotools")
    *_, preparation, args = experiment
    preparation.split, preparation.freeze, preparation.count = "validation", None, 2
    preparation.provider = provider
    args.provider = provider
    args.model = frontier.PROVIDERS[provider][0]
    args.effort = frontier.DEFAULT_EFFORTS[provider]
    WRAPPER["prepare"](preparation)
    protocol = json.loads((preparation.output / "inputs.json").read_text())
    example = json.loads(protocol["examples"][0]["response"])
    expected = (
        [{"label": "pill", "box_2d": [125, 125, 375, 375]}]
        if provider == "qwen" else {
            "detections": [{"class_name": "pill", "x_min": 0.125, "y_min": 0.125,
                            "x_max": 0.375, "y_max": 0.375, "confidence": 1.0}]
        }
    )
    assert example == expected
    assert protocol["coordinate_format"] == (
        "xyxy_1000" if provider == "qwen" else "named_normalized"
    )
    assert protocol["privacy_level"] == "deny"
    calls = []

    def infer(rows, output, **kwargs):
        output.mkdir()
        calls.append(kwargs)
        return {r["id"]: {"response": json.dumps(expected)} for r in rows}

    monkeypatch.setattr(frontier, "run_frontier", infer)
    WRAPPER["run"](args)
    report = json.loads((args.output / "evaluation.json").read_text())
    assert report["AP"] == pytest.approx(1) and report["errors"] == 0
    assert calls[0]["privacy_level"] == "deny"
    assert calls[0]["quantizations"] == (["fp8", "bf16", "fp32"] if provider == "qwen" else None)


@pytest.mark.parametrize("provider", ["qwen", "kimi"])
def test_openrouter_frozen_native_contract_is_model_specific(experiment, provider):
    source, checkpoint, policy, frozen_path, _, preparation, _ = experiment
    config = {"provider": provider, "model": frontier.PROVIDERS[provider][0], "examples": 1}
    frozen = WRAPPER["create_freeze"](
        source, checkpoint, [config], {"AP_margin_min": 0.2},
        frozen_path.with_name(provider + "-freeze.json"), [policy],
    )
    settings = frozen["baselines"][0]["config"]
    assert settings["effort"] == frontier.DEFAULT_EFFORTS[provider]
    assert settings["coordinate_format"] == WRAPPER["ROUTER_COORDINATES"][provider]
    assert settings["privacy_level"] == "deny"
    assert frozen["baselines"][0]["protocol"]["provider"] == provider
    with pytest.raises(ValueError, match="routing"):
        WRAPPER["baseline_config"]({**config, "privacy_level": "allow"})


def complete_complement(complement):
    command, args, _ = complement
    protocol = json.loads((args.input / "inputs.json").read_text())
    frontier.run_frontier(
        protocol["rows"], args.output, provider=args.provider,
        examples=protocol["examples"], max_requests=3, max_transient_retries=2,
        image_detail=protocol.get("image_detail"), response_format=protocol.get("response_format"),
        response_schema=protocol.get("response_schema"),
    )
    merge = runpy.run_path(str(Path(__file__).parents[1] / "scripts/merge_frontier_validation.py"))
    params = {
        "discovery_input": command.input, "discovery_run": command.from_run,
        "complement_input": args.input, "complement_run": args.output,
        "output": args.output.with_name("merged-validation"),
    }
    return merge["merge"], params


def test_validation_union_replays_saved_journals_without_any_hosted_request(
    complement, monkeypatch
):
    pytest.importorskip("pycocotools")
    merge, params = complete_complement(complement)

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline merge must never send a hosted request")

    monkeypatch.setattr(frontier, "_request", forbidden)
    report = merge(**params)
    assert report["n"] == 2 and set(report["image_ids"]) == {0, 1}
    record = json.loads((params["output"] / "merge.json").read_text())
    assert record["partition_sizes"] == [1, 1]
    assert record["hosted_requests_made_by_merge"] == 0
    assert len((params["output"] / "predictions.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="immutable"):
        merge(**params)


def test_validation_union_rejects_replaced_completed_answers(complement):
    merge, params = complete_complement(complement)
    path = params["complement_run"] / "predictions.jsonl"
    prediction = json.loads(path.read_text())
    prediction["response"] = "replaced answer"
    path.write_text(json.dumps(prediction) + "\n")
    with pytest.raises(ValueError, match="final journaled attempt"):
        merge(**params)
    assert not params["output"].exists()


def test_validation_union_rejects_overlapping_ids(complement):
    merge, params = complete_complement(complement)
    params["complement_input"] = params["discovery_input"]
    params["complement_run"] = params["discovery_run"]
    with pytest.raises(ValueError, match="overlap"):
        merge(**params)
    assert not params["output"].exists()


def test_validation_union_rejects_credit_stopped_confirmation(complement, monkeypatch):
    monkeypatch.setattr(
        frontier, "_request",
        lambda *args: (429, {"error": "Roboflow credits exceeded"}, {"retry_after": None}),
    )
    merge, params = complete_complement(complement)
    with pytest.raises(ValueError, match="hard-stopped"):
        merge(**params)
    assert not params["output"].exists()


def rescore_fixture(complement):
    command, _, protocol = complement
    source = command.from_run
    predictions = {
        row["id"]: row for row in (
            json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()
        )
    }
    config = json.loads((source / "run.json").read_text())["identity"]["config"]
    old = WRAPPER["score_predictions"](
        protocol, predictions, {"name": "original-evaluator", "provider": config["provider"]}
    )
    (source / "evaluation.json").write_text(json.dumps(old) + "\n")
    amendment = source.parent / "evaluator-amendment.json"
    amendment.write_text(json.dumps({"change": "Explicit validation-only evaluator revision"}))
    api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/rescore_frontier.py"))
    return api, [(command.input, source)], source.with_name("revision"), amendment


def test_rescore_keeps_native_evidence_and_prior_scores_byte_identical(complement, monkeypatch):
    api, candidates, output, amendment = rescore_fixture(complement)
    original = candidates[0][1]
    before = {path.name: path.read_bytes() for path in original.glob("*.json*")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Re-scoring must never send a provider request")

    monkeypatch.setattr(frontier, "_request", forbidden)
    record = api["rescore"](candidates, output, amendment)
    assert record["complete"] and record["new_hosted_calls"] == 0
    for name, contents in before.items():
        assert (original / name).read_bytes() == contents
    for name in api["NATIVE_FILES"]:
        assert (output / original.name / name).read_bytes() == before[name]
    result = json.loads((output / original.name / "evaluation.json").read_text())
    assert result["evaluation_revision"]["original_result"] == str(original)
    assert result["model"]["request_config"]["provider"] in {"gemini", "openai"}
    with pytest.raises(ValueError, match="immutable"):
        api["rescore"](candidates, output, amendment)


def test_rescore_cannot_open_final_test_protocol(complement):
    api, candidates, output, amendment = rescore_fixture(complement)
    path = candidates[0][0] / "inputs.json"
    protocol = json.loads(path.read_text())
    protocol["split"] = "test"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="only re-score validation"):
        api["rescore"](candidates, output, amendment)
    assert not output.exists()
