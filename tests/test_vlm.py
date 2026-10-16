"""Contract tests for optional VLM code; no GPU packages or downloads required."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_rl import vlm
from vision_rl.data import load_manifest


def manifest(tmp_path, split, color, *, identifier=None):
    image = tmp_path / f"{split}.png"
    Image.new("RGB", (8, 8), color).save(image)
    row = {
        "id": identifier or split,
        "split": split,
        "image": image.name,
        "prompt": "Which object is closer?",
        "choices": ["circle", "square"],
        "answer": "B",
        "category": "spatial",
    }
    path = tmp_path / f"{split}.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


@pytest.fixture
def manifests(tmp_path):
    return (manifest(tmp_path, "train", "red"), manifest(tmp_path, "validation", "blue"))


def test_reward_rejects_ambiguous_and_injected_completions():
    completions = [
        "<answer>B</answer>",
        "B",
        "I choose B",
        "<answer>B</answer> extra",
        "<answer>B</answer><answer>A</answer>",
        "C",
        "b",
        "",
        [{"role": "assistant", "content": "<answer>B</answer>"}],
        [{"role": "user", "content": "B"}],
        [{"role": "assistant", "content": "B"}, {"role": "assistant", "content": "B"}],
        [{"role": "assistant", "content": [{"type": "text", "text": "B"}]}],
        [{"role": "assistant", "content": [{"type": "image", "text": "B"}]}],
    ]
    assert vlm.answer_reward(
        completions, ["B"] * len(completions), [["first", "second"]] * len(completions)
    ) == [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
    ]


def test_reward_batch_mismatch_fails():
    with pytest.raises(ValueError, match="lengths must match"):
        vlm.answer_reward(["A"], [], [["a", "b"]])


def test_labels_are_not_embedded_in_conversation(manifests):
    rows = load_manifest(manifests[0])
    records = vlm._training_records(rows)
    assert records[0]["prompt"] == [{"role": "user", "content": vlm.prompt_text(rows[0])}]
    assert records[0]["answer"] == "B"
    assert "<answer>B</answer>" not in records[0]["prompt"][0]["content"]
    assert Path(records[0]["image"]).is_absolute()
    assert "split" not in records[0]


def test_no_model_access_before_cuda_check(monkeypatch):
    calls = []
    torch = SimpleNamespace(version=SimpleNamespace(cuda=None))

    def importing(name):
        calls.append(name)
        assert name == "torch", "CPU preflight must stop before model dependencies"
        return torch

    monkeypatch.setattr(vlm, "import_module", importing)
    with pytest.raises(vlm.VLMPreflightError, match="CUDA GPU is required"):
        vlm.preflight()
    assert calls == ["torch"]


def test_missing_torch_has_actionable_error(monkeypatch):
    def importing(_):
        raise ImportError("torch is absent")

    monkeypatch.setattr(vlm, "import_module", importing)
    with pytest.raises(vlm.VLMPreflightError, match="requirements-vlm.txt"):
        vlm.preflight()


def test_preflight_reports_gpu_and_checks_pinned_trl(monkeypatch):
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            is_bf16_supported=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(
                name="test GPU", total_memory=24 * 1024**3
            ),
        ),
    )
    monkeypatch.setattr(vlm, "import_module", lambda _: torch)
    monkeypatch.setattr(vlm.metadata, "version", lambda name: "1.16.0" if name == "trl" else "test")
    result = vlm.preflight()
    assert result["precision"] == "bfloat16"
    assert result["gpu_memory_gib"] == 24.0
    monkeypatch.setattr(vlm.metadata, "version", lambda _: "0.19.0")
    with pytest.raises(vlm.VLMPreflightError, match="targets TRL 1.16.0"):
        vlm.preflight()


def test_multi_process_rejected_before_import(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(vlm, "import_module", lambda _: pytest.fail("must not import"))
    with pytest.raises(vlm.VLMPreflightError, match="one training process"):
        vlm.preflight()


def test_test_split_is_never_training_input(tmp_path, manifests, monkeypatch):
    test = manifest(tmp_path, "test", "green")
    monkeypatch.setattr(vlm, "preflight", lambda: pytest.fail("must reject before CUDA access"))
    with pytest.raises(ValueError, match="expected train split"):
        vlm.train_vlm(test, manifests[1], tmp_path / "out", "test-model")
    with pytest.raises(ValueError, match="expected validation split"):
        vlm.train_vlm(manifests[0], test, tmp_path / "out", "test-model")


def test_cross_split_image_leakage_rejected(tmp_path, monkeypatch):
    train = manifest(tmp_path, "train", "red")
    validation = manifest(tmp_path, "validation", "red")
    monkeypatch.setattr(vlm, "preflight", lambda: pytest.fail("must reject before CUDA access"))
    with pytest.raises(ValueError, match="overlap"):
        vlm.train_vlm(train, validation, tmp_path / "out", "test-model")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_generations": 1},
        {"learning_rate": float("nan")},
        {"max_steps": 0},
        {"lora_rank": True},
        {"max_completion_length": -1},
        {"seed": -1},
    ],
)
def test_invalid_hyperparameters_fail_before_io(tmp_path, kwargs):
    with pytest.raises(ValueError):
        vlm.train_vlm(
            tmp_path / "absent", tmp_path / "absent2", tmp_path / "out", "model", **kwargs
        )


@pytest.fixture
def fake_training_stack(monkeypatch):
    observed = {}

    class Processor:
        image_processor = object()

        def save_pretrained(self, path):
            observed["processor_saved"] = path

    class Trainer:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def train(self, **kwargs):
            observed["resume"] = kwargs["resume_from_checkpoint"]
            return SimpleNamespace(metrics={"train_loss": 0.125})

        def evaluate(self):
            return {"eval_reward": 0.75}

        def save_model(self, path):
            observed["model_saved"] = path

        def save_state(self):
            observed["state_saved"] = True

    modules = {
        "transformers": SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(
                    model_type="qwen2_5_vl", _commit_hash="abc123"
                )
            ),
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: Processor()),
        ),
        "trl": SimpleNamespace(GRPOConfig=lambda **kw: kw, GRPOTrainer=Trainer),
        "peft": SimpleNamespace(LoraConfig=lambda **kw: kw),
    }
    monkeypatch.setattr(vlm, "import_module", lambda name: modules[name])
    monkeypatch.setattr(vlm, "preflight", lambda: {"precision": "bfloat16", "device": "cuda:0"})
    monkeypatch.setattr(vlm, "_as_dataset", vlm._training_records)
    return observed


def test_training_wires_grpo_validation_checkpoints_and_adapter(
    tmp_path, manifests, fake_training_stack
):
    output = tmp_path / "run"
    result = vlm.train_vlm(*manifests, output, "example/vlm", max_steps=3)
    call = fake_training_stack
    assert call["args"]["gradient_accumulation_steps"] % call["args"]["num_generations"] == 0
    assert call["args"]["num_generations_eval"] == 1
    assert call["args"]["remove_unused_columns"] is False
    assert call["args"]["report_to"] == "none"
    assert call["args"]["push_to_hub"] is False
    assert call["args"]["model_init_kwargs"]["revision"] == "abc123"
    assert [r["id"] for r in call["train_dataset"]] == ["train"]
    assert [r["id"] for r in call["eval_dataset"]] == ["validation"]
    assert call["reward_funcs"] is vlm.answer_reward
    assert call["peft_config"]["target_modules"] == ["q_proj", "v_proj"]
    assert call["model_saved"] == str(output / "final_adapter")
    assert call["state_saved"] is True
    persisted = json.loads((output / "training_metadata.json").read_text())
    assert persisted == result
    assert persisted["status"] == "completed"
    assert persisted["validation_metrics"] == {"eval_reward": 0.75}


def test_training_failure_preserves_metadata(tmp_path, manifests, fake_training_stack, monkeypatch):
    def failing_dataset(_):
        raise RuntimeError("example training failure")

    monkeypatch.setattr(vlm, "_as_dataset", failing_dataset)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="example training failure"):
        vlm.train_vlm(*manifests, output, "example/vlm")
    record = json.loads((output / "training_metadata.json").read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"


def test_resume_requires_original_data_and_hyperparameters(
    tmp_path, manifests, fake_training_stack
):
    output = tmp_path / "run"
    vlm.train_vlm(*manifests, output, "example/vlm")
    checkpoint = output / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}")
    with pytest.raises(ValueError, match="original lora_rank"):
        vlm.train_vlm(
            *manifests, output, "example/vlm", lora_rank=8, resume_from_checkpoint=checkpoint
        )
    vlm.train_vlm(
        *manifests, output, "example/vlm", max_steps=200, resume_from_checkpoint=checkpoint
    )
    assert fake_training_stack["resume"] == str(checkpoint.resolve())
    data = json.loads(manifests[0].read_text())
    data["answer"] = "A"
    manifests[0].write_text(json.dumps(data) + "\n")
    with pytest.raises(ValueError, match="fingerprints must match"):
        vlm.train_vlm(*manifests, output, "example/vlm", resume_from_checkpoint=checkpoint)


def test_nonempty_output_is_not_overwritten(tmp_path, manifests, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "important").write_text("preserve")
    monkeypatch.setattr(vlm, "preflight", lambda: pytest.fail("must reject before CUDA access"))
    with pytest.raises(ValueError, match="not empty"):
        vlm.train_vlm(*manifests, output, "example/vlm")
    assert (output / "important").read_text() == "preserve"


def test_predict_checks_adapter_base_before_loading(tmp_path, manifests, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "other/model"})
    )
    monkeypatch.setattr(vlm, "preflight", lambda: pytest.fail("must reject before CUDA access"))
    with pytest.raises(ValueError, match="base model does not match"):
        vlm.predict_vlm(load_manifest(manifests[0]), "example/model", adapter=adapter)


def test_preflight_cli_needs_no_manifest(monkeypatch, capsys):
    monkeypatch.setattr(vlm, "preflight", lambda: {"gpu": "test GPU"})
    assert vlm.main(["--preflight"]) == 0
    assert json.loads(capsys.readouterr().out)["gpu"] == "test GPU"


def test_predict_rejects_pickle_adapter_before_gpu_access(tmp_path, manifests, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "example/model"})
    )
    (adapter / "adapter_model.bin").write_bytes(b"untrusted pickle payload")
    monkeypatch.setattr(vlm, "preflight", lambda: pytest.fail("must reject before CUDA access"))
    with pytest.raises(ValueError, match="pickle is not accepted"):
        vlm.predict_vlm(load_manifest(manifests[0]), "example/model", adapter=adapter)


@pytest.mark.parametrize(
    "override",
    [
        {"r": 2**40},
        {"r": True},
        {"target_modules": "all-linear"},
        {"target_modules": ["q_proj", "lm_head"]},
        {"layer_replication": [[0, 100000]]},
        {"rank_pattern": {"q_proj": 2**40}},
        {"modules_to_save": ["lm_head"]},
        {"lora_alpha": float("nan")},
        {"init_lora_weights": "loftq"},
    ],
)
def test_untrusted_lora_allocations_rejected_before_gpu_access(
    tmp_path, manifests, monkeypatch, override
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {
        "base_model_name_or_path": "example/model",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        **override,
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    (adapter / "adapter_model.safetensors").write_bytes(b"small mock tensor artifact")
    monkeypatch.setattr(vlm, "preflight", lambda: pytest.fail("reject before GPU/model allocation"))
    with pytest.raises(ValueError, match="adapter"):
        vlm.predict_vlm(load_manifest(manifests[0]), "example/model", adapter=adapter)


def test_valid_lora_config_only_forwards_bounded_parameters():
    config = {
        "base_model_name_or_path": "example/model",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
        "init_lora_weights": True,
        "arbitrary_unknown_field": "must not be forwarded",
    }
    result = vlm._validated_lora_config(config)
    assert result == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "init_lora_weights": False,
    }
