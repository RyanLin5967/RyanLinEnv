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
