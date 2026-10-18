"""Offline data/reward/control contracts and optional released-TRL CPU checks."""

import hashlib
import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_rl import vlm_local_training as local
from vision_rl.data import assert_disjoint


@pytest.fixture
def diagnostic(tmp_path):
    root = tmp_path / "data"
    local.make_probe_data(root)
    return root


def load_diagnostic(root):
    spec = local.load_spec(root / "data_spec.json")
    train = local.load_classification(root / "train.jsonl", "train", spec["labels"])
    validation = local.load_classification(root / "validation.jsonl", "validation", spec["labels"])
    return spec, train, validation


def test_probe_data_is_explicitly_diagnostic_reproducible_and_disjoint(diagnostic, tmp_path):
    second = tmp_path / "second"
    local.make_probe_data(second)
    spec, train, validation = load_diagnostic(diagnostic)
    assert spec["diagnostic_only"] is True
    assert len(train) == 8 and len(validation) == 4
    assert_disjoint(train, validation)
    assert (diagnostic / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert (diagnostic / "train-0.png").read_bytes() == (second / "train-0.png").read_bytes()


def test_full_class_vocabulary_and_source_hash_are_required(diagnostic):
    path = diagnostic / "data_spec.json"
    spec = json.loads(path.read_text())
    spec["labels"] = [f"aircraft {n}" for n in range(100)]
    path.write_text(json.dumps(spec))
    assert len(local.load_spec(path)["labels"]) == 100
    spec["sources"][0]["sha256"] = "main"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="immutable sha256"):
        local.load_spec(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"split": "test"}, "expected train"),
        ({"prompt": "leaked label"}, "require only"),
        ({"label": "unknown"}, "unknown canonical label"),
    ],
)
def test_training_manifest_rejects_test_unknown_label_and_row_prompt(diagnostic, mutation, match):
    path = diagnostic / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(mutation)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        local.load_classification(path, "train", ["blue", "red"])


def test_missing_class_and_cross_split_reencoded_image_are_rejected(diagnostic):
    spec, train, validation = load_diagnostic(diagnostic)
    path = diagnostic / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["label"] == "red"))
    with pytest.raises(ValueError, match="every declared class"):
        local.load_classification(path, "train", spec["labels"])
    with Image.open(train[0]["image"]) as image:
        image.save(validation[0]["image"])
    validation = local.load_classification(
        diagnostic / "validation.jsonl", "validation", spec["labels"]
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(train, validation)


def test_label_only_enters_sft_completion_and_reward_column(diagnostic):
    spec, rows, _ = load_diagnostic(diagnostic)
    grpo = local.training_records(rows, spec, "grpo")
    sft = local.training_records(rows, spec, "sft")
    assert grpo[0]["prompt"] == grpo[1]["prompt"] == sft[0]["prompt"]
    assert "<answer>blue</answer>" not in grpo[0]["prompt"][0]["content"]
    assert "completion" not in grpo[0]
    assert sft[0]["completion"] == [{"role": "assistant", "content": "blue"}]
    assert "<think>" not in sft[0]["completion"][0]["content"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<answer>blue</answer>", "blue"),
        ("  <answer>blue</answer>\n", "blue"),
        (
            "<think>Compare the squares.\nBlue is more frequent.</think><answer>blue</answer>",
            "blue",
        ),
        ("<think>red</think>\n<answer>blue</answer>", "blue"),
        ("blue", "blue"),
        ("<think>Compare the squares.</think>blue", "blue"),
        ("Blue", None),
        ("The label is blue", None),
        ("<answer>Blue</answer>", None),
        ("<answer> blue </answer>", None),
        ("<answer>green</answer>", None),
        ("Reasoning <answer>blue</answer>", None),
        ("<answer>blue</answer> extra", None),
        ("<answer>red</answer><answer>blue</answer>", None),
        ("<think><answer>red</answer></think><answer>blue</answer>", None),
        ("<think>one</think><think>two</think><answer>blue</answer>", None),
        ("<think>" + "x" * 32769 + "</think><answer>blue</answer>", None),
        ([{"role": "assistant", "content": "<answer>blue</answer>"}], "blue"),
        ([{"role": "user", "content": "<answer>blue</answer>"}], None),
    ],
)
def test_exact_label_parser_accepts_only_one_bounded_reasoning_block(text, expected):
    assert local.parse_label(text, ["blue", "red"]) == expected


def test_answer_only_profile_is_distinct_from_reasoning():
    text = "<think>reason</think><answer>blue</answer>"
    assert local.parse_label(text, ["blue", "red"], allow_reasoning=False) is None
    assert local.Profile().response_format == "think_then_answer"
    assert local.Profile().max_completion_length == 256
    with pytest.raises(ValueError):
        replace(local.Profile(), max_completion_length=513).validate()


def test_reward_requires_eos_and_journals_true_mixed_groups(tmp_path):
    path = tmp_path / "groups.jsonl"
    reward = local.make_reward(["blue", "red"], 4, [99], path)
    completions = [
        "<think>Count the squares.</think><answer>blue</answer>",
        "<answer>red</answer>",
        "<answer>blue</answer>",
        "<answer>blue</answer> ignored",
    ]
    assert reward(
        completions, ["blue"] * 4, ["item"] * 4, [[10, 99], [11, 99], [10], [10, 99]]
    ) == [1, 0, 0, 0]
    journal = json.loads(path.read_text())
    assert journal["rewards"] == [1, 0, 0, 0]
    assert reward.stats["mixed_groups"] == 1
    assert reward.stats["complete_think_blocks"] == 1
    assert reward.stats["task_correct_with_think"] == 1
    assert reward.stats["nonterminated_completions"] == 1
    assert reward.stats["unique_prompt_ids"] == 1
    assert reward.stats["sampled_completions"] == 4
    assert reward.stats["recognition_mixed_groups"] == 1
    assert journal["canonical_predictions"] == ["blue", "red", None, None]
    assert reward(["bad"] * 4, ["blue"] * 4, ["item"] * 4, [[99]] * 4) == [0] * 4
    assert reward.stats["all_zero_groups"] == 1


def test_correct_vs_invalid_only_group_is_not_recognition_support(tmp_path):
    reward = local.make_reward(["blue", "red"], 2, [99], tmp_path / "groups.jsonl")
    assert reward(["blue", "bad"], ["blue"] * 2, ["i"] * 2, [[99]] * 2) == [1, 0]
    assert reward.stats["mixed_groups"] == 1
    assert reward.stats["recognition_mixed_groups"] == 0


def test_train_only_accepts_public_train_subset_and_never_opens_validation(
    diagnostic, tmp_path, monkeypatch
):
    spec_path = diagnostic / "data_spec.json"
    spec = json.loads(spec_path.read_text())
    spec.pop("diagnostic_only")
    spec_path.write_text(json.dumps(spec))
    train = diagnostic / "train.jsonl"
    train.write_text(train.read_text().splitlines()[0] + "\n")
    monkeypatch.setattr(
        local, "preflight", lambda *_: (_ for _ in ()).throw(RuntimeError("ready before model"))
    )
    args = dict(
        train=train,
        validation=None,
        data_spec=spec_path,
        output=tmp_path / "out",
        model_directory=tmp_path / "model",
        max_steps=4,
        train_only=True,
    )
    with pytest.raises(RuntimeError, match="ready before model"):
        local.train_local(**args)
    with pytest.raises(ValueError, match="forbids validation"):
        local.train_local(**dict(args, validation=tmp_path / "never-open-this"))
    with pytest.raises(ValueError, match="400 steps"):
        local.train_local(**dict(args, max_steps=401))
    with pytest.raises(ValueError, match="original finite active-time cap"):
        local.train_local(**dict(args, mode="sft", max_steps=401))
    with pytest.raises(RuntimeError, match="ready before model"):
        local.train_local(**dict(args, mode="sft", max_steps=3600, max_active_seconds=7200))
    assert local.TRAIN_ONLY_STEP_CAPS == {"grpo": 400, "sft": None}
    row = json.loads(train.read_text())
    row["split"] = "test"
    train.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="expected train"):
        local.train_local(**args)


def test_train_only_cli_requires_explicit_scope(monkeypatch, capsys):
    def train(**kwargs):
        assert kwargs["train_only"] is True
        assert kwargs["validation"] is None
        assert kwargs["max_steps"] == 4
        return {"held_out_benchmark_improvement": "unmeasured"}

    monkeypatch.setattr(local, "train_local", train)
    args = [
        "--train",
        "train.jsonl",
        "--data-spec",
        "spec.json",
        "--output",
        "out",
        "--max-steps",
        "4",
    ]
    assert local.main(args + ["--train-only"]) == 0
    assert json.loads(capsys.readouterr().out)["held_out_benchmark_improvement"] == "unmeasured"
    with pytest.raises(SystemExit):
        local.main(args)


def test_reward_rejects_mismatched_lengths_and_mixed_items(tmp_path):
    reward = local.make_reward(["blue", "red"], 2, [99], tmp_path / "groups.jsonl")
    with pytest.raises(ValueError, match="lengths"):
        reward(["bad"], [], [], [])
    with pytest.raises(ValueError, match="complete groups"):
        reward(["bad"], ["blue"], ["i"], [[99]])
    with pytest.raises(ValueError, match="different examples"):
        reward(["bad"] * 2, ["blue"] * 2, ["a", "b"], [[99]] * 2)


def test_mean_class_accuracy_and_cutoff_semantics():
    rows = [{"id": "a", "label": "blue"}, {"id": "b", "label": "blue"}, {"id": "c", "label": "red"}]
    predictions = {
        "a": "<answer>blue</answer>",
        "b": "<answer>blue</answer>",
        "c": {"response": "<answer>red</answer>", "terminated": False},
    }
    metrics = local.classification_metrics(rows, predictions, ["blue", "red"])
    assert metrics["accuracy"] == 2 / 3
    assert metrics["mean_class_accuracy"] == 0.5
    assert metrics["invalid_answers"] == 1
    with pytest.raises(ValueError, match="exactly"):
        local.classification_metrics(rows, {"unknown": "bad"}, ["blue", "red"])
    sampled = local.classification_metrics(
        rows, predictions, ["blue", "red"], decoding_protocol="native-sampling/pass@1"
    )
    assert sampled["protocol"].endswith("/native-sampling/pass@1")
    assert sampled["mean_class_accuracy"] == metrics["mean_class_accuracy"]


def test_control_configs_match_image_and_optimizer_budget(tmp_path):
    profile = local.Profile()
    kwargs = dict(device="cpu", max_steps=4, learning_rate=1e-5, seed=42, save_steps=1)
    grpo = local.trainer_kwargs("grpo", tmp_path, profile, **kwargs)
    sft = local.trainer_kwargs("sft", tmp_path, profile, **kwargs)
    for name in (
        "max_steps",
        "seed",
        "learning_rate",
        "weight_decay",
        "optim",
        "gradient_checkpointing",
        "bf16",
        "fp16",
    ):
        assert grpo[name] == sft[name]
    assert grpo["gradient_accumulation_steps"] == grpo["generation_batch_size"] == 4
    assert grpo["num_iterations"] == 1 and grpo["loss_type"] == "grpo"
    assert grpo["mask_truncated_completions"] is True and grpo["beta"] == 0.04
    assert grpo["sync_ref_model"] is False
    assert local.trainer_kwargs("grpo", tmp_path, profile, beta=0, **kwargs)["beta"] == 0
    assert grpo["use_vllm"] is False and grpo["top_p"] == 1 and grpo["top_k"] == 0
    assert sft["gradient_accumulation_steps"] == 1 and sft["completion_only_loss"] is True
    assert sft["max_length"] is None and sft["packing"] is False


def test_data_and_output_fail_before_import_or_model_access(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must fail before dependencies"))
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep").write_text("preserve")
    with pytest.raises(ValueError, match="not empty"):
        local.train_local(
            diagnostic / "train.jsonl",
            diagnostic / "validation.jsonl",
            diagnostic / "data_spec.json",
            output,
            model_directory=tmp_path / "model",
        )
    with pytest.raises(ValueError, match="complete optimizer"):
        local.train_local(
            diagnostic / "train.jsonl",
            diagnostic / "validation.jsonl",
            diagnostic / "data_spec.json",
            output,
            model_directory=tmp_path / "model",
            resume_from_checkpoint=output / "checkpoint-1",
        )


def test_train_subset_inference_keeps_full_vocabulary_without_requiring_all_classes(
    diagnostic, tmp_path, monkeypatch
):
    path = diagnostic / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="every declared class"):
        local.load_classification(path, "train", ["blue", "red"])

    def ready(device):
        assert device == "cpu"
        raise RuntimeError("validated before model access")

    monkeypatch.setattr(local, "preflight", ready)
    with pytest.raises(RuntimeError, match="validated before model access"):
        local.predict_train_subset(
            path,
            diagnostic / "data_spec.json",
            tmp_path / "out",
            model_directory=tmp_path / "model",
            device="cpu",
        )
    rows[0]["split"] = "test"
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="expected train"):
        local.predict_train_subset(
            path,
            diagnostic / "data_spec.json",
            tmp_path / "out",
            model_directory=tmp_path / "model",
            device="cpu",
        )


def test_train_subset_inference_budget_and_existing_output_fail_before_model(
    diagnostic, tmp_path, monkeypatch
):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must fail before model"))
    path = diagnostic / "train.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    path.write_text("".join(json.dumps(dict(row, id=f"id-{i}")) + "\n" for i in range(33)))
    with pytest.raises(ValueError, match="limited to 32"):
        local.predict_train_subset(
            path,
            diagnostic / "data_spec.json",
            tmp_path / "out",
            model_directory=tmp_path / "model",
        )
    path.write_text(json.dumps(row) + "\n")
    output = tmp_path / "out"
    output.mkdir()
    (output / "preserve").write_text("old evidence")
    with pytest.raises(ValueError, match="not empty"):
        local.predict_train_subset(
            path, diagnostic / "data_spec.json", output, model_directory=tmp_path / "model"
        )


def test_train_subset_cli_is_inference_only(monkeypatch, capsys):
    def predict(manifest, **kwargs):
        assert manifest == Path("train.jsonl")
        assert kwargs["profile"].response_format == "think_then_answer"
        assert kwargs["profile"].max_completion_length == 256
        return {"optimizer_steps": 0}

    monkeypatch.setattr(local, "predict_train_subset", predict)
    argv = [
        "--predict-train",
        "train.jsonl",
        "--data-spec",
        "spec.json",
        "--output",
        "out",
        "--model-directory",
        "model",
    ]
    assert local.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["optimizer_steps"] == 0
    with pytest.raises(SystemExit):
        local.main(argv + ["--train", "train.jsonl"])


def test_saved_adapter_sampling_cli_preserves_explicit_frozen_policy(monkeypatch, capsys):
    def sample(manifest, **kwargs):
        assert kwargs["init_adapter"] == Path("warm/final_adapter")
        assert kwargs["seed"] == 314
        return {"optimizer_steps": 0, "policy_identity": "saved_adapter"}

    monkeypatch.setattr(local, "sample_train_groups", sample)
    assert (
        local.main(
            [
                "--sample-train",
                "train.jsonl",
                "--data-spec",
                "spec.json",
                "--output",
                "out",
                "--model-directory",
                "model",
                "--init-adapter",
                "warm/final_adapter",
                "--seed",
                "314",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["optimizer_steps"] == 0


def test_sampled_train_cli_has_no_optimizer_or_validation_input(monkeypatch, capsys):
    def sample(manifest, **kwargs):
        assert manifest == Path("train.jsonl")
        assert kwargs["profile"].num_generations == 4
        assert kwargs["seed"] == 791
        return {"optimizer_steps": 0, "reward_audit": {"mixed_groups": 0}}

    monkeypatch.setattr(local, "sample_train_groups", sample)
    argv = [
        "--sample-train",
        "train.jsonl",
        "--data-spec",
        "spec.json",
        "--output",
        "out",
        "--model-directory",
        "model",
        "--seed",
        "791",
    ]
    assert local.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["optimizer_steps"] == 0
    with pytest.raises(SystemExit):
        local.main(argv + ["--validation", "validation.jsonl"])
    with pytest.raises(SystemExit):
        local.main(argv + ["--predict-train", "train.jsonl"])


def test_task_sampling_keeps_eos_mask_and_real_group_rewards(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)

    class Inputs(dict):
        def to(self, device):
            return self

    monkeypatch.setattr(local, "_inputs", lambda *_: Inputs(input_ids=torch.tensor([[8, 8]])))
    outputs = iter([[10, 99, 0], [11, 99], [10, 10], [12, 99]])

    class Policy:
        def eval(self):
            return self

        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            return torch.tensor([[8, 8] + next(outputs)])

    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            decode=lambda tokens, **_: {10: "blue", 11: "red", 12: "bad"}[tokens[0]]
        )
    )
    reward = local.make_reward(["blue", "red"], 4, [99], tmp_path / "reward_groups.jsonl")
    groups = local._sample_task_groups(
        Policy(),
        processor,
        [{"id": "one", "label": "blue"}],
        {},
        local.Profile(),
        "cpu",
        tmp_path,
        reward,
        SimpleNamespace(eos_token_id=[99]),
    )
    assert groups["one"][0]["completion_ids"] == [10, 99]
    assert groups["one"][2]["terminated"] is False
    assert reward.stats["mixed_groups"] == 1
    assert reward.stats["sampled_completions"] == 4
    assert json.loads((tmp_path / "reward_groups.jsonl").read_text())["rewards"] == [1, 0, 0, 0]
    assert len(json.loads((tmp_path / "sampled_completions.json").read_text())["one"]) == 4


def test_preflight_checks_versions_before_import(monkeypatch):
    monkeypatch.setattr(local.metadata, "version", lambda _: "wrong")
    monkeypatch.setattr(local, "import_module", lambda _: pytest.fail("must not import"))
    with pytest.raises(local.LocalVLMError, match="Expected torch"):
        local.preflight("cpu")


def test_no_distributed_or_hidden_cpu_fallback(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(local.LocalVLMError, match="one local process"):
        local.preflight("mps")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(local.LocalVLMError, match="Disable"):
        local.preflight("mps")


def test_snapshot_verification_detects_changed_weights(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    files = {
        "weights.safetensors": b"mock bytes",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"x": "weights.safetensors"}}
        ).encode(),
    }
    spec = {
        "model_repository": local.DEFAULT_MODEL,
        "revision": local.MODEL_REVISIONS[local.DEFAULT_MODEL],
        "files": [],
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
        spec["files"].append(
            {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    source = directory / "model-source.json"
    source.write_text(json.dumps(spec))
    (directory / ".pinned-download.json").write_text(
        json.dumps(
            {
                "repository": local.DEFAULT_MODEL,
                "revision": spec["revision"],
                "manifest_sha256": local._file_sha256(source),
            }
        )
    )
    assert local._verified_model_directory(directory, local.DEFAULT_MODEL)["total_bytes"] > 0
    (directory / "weights.safetensors").write_bytes(b"fake bytes")
    with pytest.raises(ValueError, match="hash differs"):
        local._verified_model_directory(directory, local.DEFAULT_MODEL)


def test_warm_start_can_pin_common_format_data_without_demanding_same_task(tmp_path):
    identity = {
        "model": local.DEFAULT_MODEL,
        "model_revision": "pin",
        "processor_model": "processor",
        "processor_revision": "pin",
        "profile": asdict(local.Profile()),
        "train": "task train",
        "validation": "task validation",
        "data_spec_sha256": "task spec",
    }
    parent = dict(
        identity, train="warmup train", validation="warmup validation", data_spec_sha256="warmup"
    )
    parent["profile"] = dict(
        identity["profile"], response_format="answer_only", max_completion_length=32
    )
    (tmp_path / "local_training.json").write_text(
        json.dumps({"status": "completed", "control_identity": parent})
    )
    adapter = tmp_path / "final_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"mock weights")
    info = local._adapter_identity(adapter, identity, local.Profile())
    assert info["parent_data_identity"]["train"] == "warmup train"
    bf16 = replace(local.Profile(), dtype="bfloat16")
    bf16_identity = dict(identity, profile=asdict(bf16))
    with pytest.raises(ValueError, match="identical model/processor/profile"):
        local._adapter_identity(adapter, bf16_identity, bf16)
    transferred = local._adapter_identity(
        adapter, bf16_identity, bf16, precision_transfer="float32-to-bfloat16"
    )
    assert transferred["weights_sha256"] == info["weights_sha256"]
    assert transferred["precision_transfer"]["initial_policy_numerics_changed"] is True
    with pytest.raises(ValueError, match="only explicit"):
        local._adapter_identity(
            adapter, identity, local.Profile(), precision_transfer="float32-to-bfloat16"
        )
    parent["data_spec_sha256"] = identity["data_spec_sha256"]
    (tmp_path / "local_training.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "control_identity": parent,
                "final_adapter": str(adapter),
                "final_adapter_sha256": info["weights_sha256"],
            }
        )
    )
    assert (
        local._inference_adapter_identity(adapter, identity, local.Profile())["weights_sha256"]
        == info["weights_sha256"]
    )
    with pytest.raises(ValueError, match="same pinned task"):
        local._inference_adapter_identity(
            adapter, dict(identity, data_spec_sha256="other"), local.Profile()
        )
    (adapter / "adapter_model.safetensors").write_bytes(b"altered weights")
    with pytest.raises(ValueError, match="completed endpoint identity"):
        local._inference_adapter_identity(adapter, identity, local.Profile())


def released_trl():
    pytest.importorskip("torch")
    trl = pytest.importorskip("trl")
    if trl.__version__ != "1.16.0":
        pytest.skip("requires the reviewed TRL release")
    return trl


def test_actual_peft_preserves_saved_float32_adapter_on_bfloat16_base(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(2)
    config = LlamaConfig(
        vocab_size=5,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    original = peft.get_peft_model(
        LlamaForCausalLM(config),
        peft.LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ),
    )
    with torch.no_grad():
        for name, parameter in original.named_parameters():
            if "lora_" in name:
                parameter.uniform_(0.123456, 0.234567)
    original.save_pretrained(tmp_path)
    loaded = peft.PeftModel.from_pretrained(
        LlamaForCausalLM(config).to(torch.bfloat16), str(tmp_path), is_trainable=True
    )
    result = local._verify_initial_adapter_tensors(loaded, tmp_path)
    assert result == {"exact_saved_tensors": True, "tensor_count": 4, "dtype": "float32"}
    assert {p.dtype for p in loaded.parameters() if p.requires_grad} == {torch.float32}
    assert {p.dtype for p in loaded.parameters() if not p.requires_grad} == {torch.bfloat16}
    with torch.no_grad():
        next(p for p in loaded.parameters() if p.requires_grad).add_(0.00001)
    with pytest.raises(local.LocalVLMError, match="exact saved FP32"):
        local._verify_initial_adapter_tensors(loaded, tmp_path)


def test_released_configs_instantiate_on_cpu(tmp_path):
    trl = released_trl()
    for mode, cls in (("grpo", trl.GRPOConfig), ("sft", trl.SFTConfig)):
        config = cls(
            **local.trainer_kwargs(
                mode,
                tmp_path,
                local.Profile(),
                device="cpu",
                max_steps=2,
                learning_rate=1e-5,
                seed=42,
                save_steps=1,
            )
        )
        assert config.device.type == "cpu"


def test_released_dataset_builds_and_decodes_images_in_isolated_install(diagnostic):
    pytest.importorskip("datasets")
    spec, rows, _ = load_diagnostic(diagnostic)
    for mode in ("grpo", "sft"):
        dataset = local._dataset(rows, spec, mode, local.Profile())
        assert len(dataset) == 8
        assert dataset[0]["image"].mode == "RGB"
        assert dataset[0]["label"] == "blue"


def test_aircraft_lazy_transform_is_shared_by_dataset_input_and_pixel_identity(diagnostic):
    pytest.importorskip("datasets")
    spec, _, _ = load_diagnostic(diagnostic)
    spec["preprocessing_id"] = local.PREPROCESSING_ID
    rows = local.load_classification(
        diagnostic / "train.jsonl", "train", spec["labels"], preprocessing_id=local.PREPROCESSING_ID
    )
    original = (diagnostic / "train-0.png").read_bytes()
    dataset = local._dataset(rows, spec, "grpo", local.Profile())
    assert dataset[0]["image"].size == (224, 204)
    from vision_rl.aircraft_data import rgb_pixel_sha256

    assert rgb_pixel_sha256(dataset[0]["image"]) == rows[0]["image_sha256"]
    assert local._load_rgb(rows[0]["image"], spec).tobytes() == dataset[0]["image"].tobytes()
    assert (diagnostic / "train-0.png").read_bytes() == original


def test_native_processor_and_released_collator_consume_lazy_cropped_dataset(diagnostic):
    released_trl()
    model = Path(__file__).parents[1] / "data/models/Qwen2-VL-2B-Instruct-895c3a49"
    if not (model / "preprocessor_config.json").is_file():
        pytest.skip("requires the separately acquired pinned processor; never downloads")
    import torch
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    torch.set_num_threads(2)
    spec, rows, _ = load_diagnostic(diagnostic)
    spec["preprocessing_id"] = local.PREPROCESSING_ID
    profile = replace(local.Profile(), min_pixels=50176, max_pixels=50176)
    processor = local._processor(model, profile)
    dataset = local._dataset(rows, spec, "sft", profile)
    batch = DataCollatorForVisionLanguageModeling(
        processor, max_length=None, completion_only_loss=True
    )([dataset[0]])
    supervised = batch["labels"][0][batch["labels"][0] != -100]
    assert processor.tokenizer.decode(supervised) == "blue<|im_end|>\n"
    assert batch["pixel_values"].shape[0] > 0
    assert (batch["labels"] == -100).sum() > len(supervised)


def test_unknown_lazy_transform_rejected(diagnostic):
    path = diagnostic / "data_spec.json"
    spec = json.loads(path.read_text())
    spec["preprocessing_id"] = "silently-use-ground-truth-box"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unknown preprocessing_id"):
        local.load_spec(path)


def test_released_vlm_collator_masks_prompt_tokens():
    released_trl()
    import torch
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    torch.set_num_threads(2)

    class Processor:
        def __call__(self, text, **kwargs):
            ids = [[11, 12, 13] if x == "prompt" else [21, 22] for x in text]
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.ones((len(ids), len(ids[0])), dtype=torch.long),
            }

    batch = DataCollatorForVisionLanguageModeling(
        Processor(), max_length=None, completion_only_loss=True
    )([{"prompt": "prompt", "completion": "gold", "image": object()}])
    assert batch["labels"].tolist() == [[-100, -100, -100, 21, 22]]


def test_gradient_audit_measures_updates_and_rejects_nonfinite_gradients(tmp_path):
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_B = torch.nn.Parameter(torch.tensor([1.0]))

    model = Model()
    state, control = SimpleNamespace(global_step=1), SimpleNamespace()
    audit = local._audit_callback(tmp_path, None)
    audit.on_train_begin(None, state, control, model=model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    audit.on_step_begin(None, state, control)
    model.lora_B.square().sum().backward()
    audit.on_pre_optimizer_step(None, state, control, model=model)
    optimizer.step()
    audit.on_step_end(None, state, control)
    audit.on_train_end(None, state, control, model=model)
    assert audit.summary["nonzero_gradient_steps"] == 1
    assert audit.summary["changed_tensors"] == 1 and audit.summary["parameter_delta_l2"] > 0
    model.lora_B.grad = torch.tensor([float("nan")])
    with pytest.raises(local.LocalVLMError, match="nonfinite gradient"):
        audit.on_pre_optimizer_step(None, state, control, model=model)


def test_mps_rng_extension_preserves_parent_rng_and_uses_safe_cpu_load(tmp_path, monkeypatch):
    calls = []

    class Base:
        def _save_rng_state(self, path):
            calls.append(("parent_save", path))

        def _load_rng_state(self, path):
            calls.append(("parent_load", path))

    def save(state, path):
        assert state == "mps state"
        calls.append(("mps_save", path))

    def load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        assert path == tmp_path / "mps_rng_state.pth"
        return "restored state"

    fake = SimpleNamespace(
        save=save,
        load=load,
        mps=SimpleNamespace(
            get_rng_state=lambda: "mps state",
            set_rng_state=lambda state: calls.append(("mps_restore", state)),
        ),
    )
    monkeypatch.setattr(local, "import_module", lambda _: fake)
    assert local._trainer_with_mps_rng(Base, "cpu") is Base
    trainer = local._trainer_with_mps_rng(Base, "mps")()
    trainer._save_rng_state(tmp_path)
    trainer._load_rng_state(tmp_path)
    assert calls == [
        ("parent_save", tmp_path),
        ("mps_save", tmp_path / "mps_rng_state.pth"),
        ("parent_load", tmp_path),
        ("mps_restore", "restored state"),
    ]


def test_active_time_binds_checkpoint_and_separates_discarded_work(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    for name in local.ACTIVE_CHECKPOINT_FILES:
        (checkpoint / name).write_text(
            '{"global_step": 1}' if name == "trainer_state.json" else name
        )
    clock = iter([100.0, 110.0, 200.0, 205.0, 300.0, 303.0])
    monkeypatch.setattr(local.time, "monotonic", lambda: next(clock))
    timer = local._ActiveStepTimer(tmp_path, 40.0, "cpu")
    timer.begin()
    timer.end()
    timer.save(checkpoint, 1)
    saved = local._load_active_time(checkpoint, 40.0)
    assert saved["committed_active_seconds"] == 10.0
    timer.begin()
    timer.end()
    timer.begin()
    timer.interrupt()
    report = timer.summary()
    assert report["completed_state_active_seconds"] == 15.0
    assert report["committed_active_seconds"] == 10.0
    assert report["observed_physical_active_seconds"] == 18.0
    assert report["observed_discarded_or_uncheckpointed_active_seconds"] == 8.0
    assert report["completed_not_checkpointed_active_seconds"] == 5.0
    with pytest.raises(local.LocalVLMError, match="budget/state/optimizer"):
        local._load_active_time(checkpoint, 50.0)
    (checkpoint / "optimizer.pt").write_text("different optimizer")
    with pytest.raises(local.LocalVLMError, match="budget/state/optimizer"):
        local._load_active_time(checkpoint, 40.0)
    (checkpoint / "active_time.json").unlink()
    with pytest.raises(local.LocalVLMError, match="requires checkpoint active_time"):
        local._load_active_time(checkpoint, 40.0)


def test_exhausted_budget_prevents_another_step_and_stale_marker_is_invalidated(tmp_path):
    timer = local._ActiveStepTimer(
        tmp_path, 1.0, "cpu", {"committed_active_seconds": 1.0, "global_step": 3}
    )
    with pytest.raises(local.LocalVLMError, match="already exhausted"):
        timer.begin()
    assert timer.started is None and not timer.path.exists()
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    marker = checkpoint / "active_time.json"
    marker.write_text("old marker")

    class Base:
        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(global_step=3)

        def _save_checkpoint(self, model, trial):
            assert not marker.exists()
            raise OSError("interrupted native write")

    with pytest.raises(OSError, match="interrupted native write"):
        local._trainer_with_active_time(Base)()._save_checkpoint(None, None)
    assert not marker.exists()


def test_active_timer_excludes_journal_io_with_explicit_device_sync(tmp_path, monkeypatch):
    clock = iter([100.0, 110.0, 115.0, 120.0])
    monkeypatch.setattr(local.time, "monotonic", lambda: next(clock))
    syncs = []
    monkeypatch.setattr(
        local,
        "import_module",
        lambda _: SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: syncs.append("sync"))),
    )
    timer = local._ActiveStepTimer(tmp_path, 40.0, "mps")
    timer.begin()
    with timer.exclude_journal_io():
        pass
    timer.end()
    assert syncs == ["sync", "sync", "sync"]
    assert timer.completed == 15.0
    assert timer.summary()["observed_reward_journal_io_seconds"] == 5.0


def test_actual_exposure_lineage_excludes_discarded_events_and_sft_strips_ids(tmp_path):
    ledger = local._ExposureLedger(tmp_path, ["a", "b"])
    ledger.begin(1)
    ledger.observe("a")
    ledger.end(1)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    ledger.save(checkpoint, 1)
    ledger.begin(2)
    ledger.observe("b")  # Observed physical work, not committed to the checkpoint.
    restored = local._ExposureLedger(
        tmp_path, ["a", "b"], json.loads((checkpoint / "exposure.json").read_text())
    )
    restored.begin(2)
    seen = []

    class Base:
        def training_step(self, model, inputs, num_items_in_batch):
            assert inputs == {"input_ids": "native batch"}
            seen.append("native-forward-backward")
            return "loss"

    batch = local._sft_collator_with_ids(lambda _: {"input_ids": "native batch"})([{"id": "a"}])
    trainer = local._trainer_with_sft_exposure(
        Base, SimpleNamespace(observe_prompt=restored.observe)
    )()
    assert trainer.training_step(None, batch) == "loss"
    restored.end(2)
    assert [entry["id"] for entry in restored.entries] == ["a", "a"]
    assert len((tmp_path / "physical_prompt_events.jsonl").read_text().splitlines()) == 3
    assert seen == ["native-forward-backward"]


def test_cooperative_stage_boundaries_do_not_change_total_budget(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(local.time, "monotonic", lambda: next(clock))
    audit = local._audit_callback(tmp_path, 10.0, stop_after_steps=2)
    state, control = SimpleNamespace(global_step=1), SimpleNamespace()
    audit.on_step_begin(None, state, control)
    state.global_step = 2
    audit.on_step_end(None, state, control)
    assert control.should_training_stop and control.should_save
    assert audit.timer.total_budget == 10.0
    assert audit.timer.committed == 2.0


def test_finalize_only_requires_checkpoint_before_model(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("no model access"))
    with pytest.raises(ValueError, match="explicit existing checkpoint"):
        local.train_local(
            diagnostic / "train.jsonl",
            None,
            diagnostic / "data_spec.json",
            tmp_path / "out",
            train_only=True,
            finalize_only=True,
            model_directory=tmp_path / "model",
        )


def test_explicit_endpoint_restore_does_not_train_or_step(tmp_path):
    pytest.importorskip("transformers")
    (tmp_path / "trainer_state.json").write_text('{"global_step": 4}')
    calls = []

    class Trainer:
        def _load_from_checkpoint(self, path):
            calls.append("model")

        def _load_rng_state(self, path):
            calls.append("rng")

        def train(self, **kwargs):
            pytest.fail("finalization cannot train")

    trainer = Trainer()
    audit = local._audit_callback(
        tmp_path, 1.0, resume_state={"committed_active_seconds": 1.0, "global_step": 4}
    )
    local._restore_endpoint_without_training(trainer, tmp_path, audit)
    assert calls == ["model", "rng"] and trainer.state.global_step == 4
    assert audit.summary["optimizer_steps_observed"] == 0
    assert audit.summary["active_time"]["committed_active_seconds"] == 1.0


def test_released_trl_reference_is_initial_adapter_and_both_adapters_restore(tmp_path):
    trl = released_trl()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from trl.trainer.utils import RepeatSampler, use_adapter

    torch.set_num_threads(2)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3, "a": 4}, unk_token="[UNK]"
            )
        ),
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    )
    policy = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=5,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                pad_token_id=0,
                eos_token_id=1,
                bos_token_id=2,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    # Nonzero initial B is a warm-start contract, not pretrained task learning.
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B.default." in name:
                parameter.fill_(0.125)
    args = trl.GRPOConfig(
        **local.trainer_kwargs(
            "grpo",
            tmp_path,
            replace(local.Profile(), num_generations=2),
            device="cpu",
            max_steps=1,
            learning_rate=1e-5,
            seed=42,
            save_steps=1,
        )
    )
    trainer = local._trainer_with_grpo_resume(trl.GRPOTrainer)(
        model=policy,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"prompt": ["a", "a"]}),
        reward_funcs=lambda completions, **_: [0.0] * len(completions),
    )
    policy.config.image_token_id = 3  # Tiny text fixture: config parity only, no sampled image.
    sampled_config = local._task_group_generation_config(
        policy, SimpleNamespace(tokenizer=tokenizer), replace(local.Profile(), num_generations=2)
    )
    for key in (
        "max_new_tokens",
        "do_sample",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "cache_implementation",
    ):
        assert getattr(sampled_config, key) == getattr(trainer.generation_config, key)
    assert sampled_config.suppress_tokens == [3]
    sampler = trainer._get_train_sampler()
    native = RepeatSampler(
        **{
            key: getattr(sampler, key)
            for key in (
                "data_source",
                "mini_repeat_count",
                "batch_size",
                "repeat_count",
                "shuffle",
                "seed",
            )
        }
    )
    first, second = list(native), list(native)
    sampler.set_epoch(0)
    assert list(sampler) == first
    restarted = trainer._get_train_sampler()
    restarted.set_epoch(1)
    assert list(restarted) == second
    reference = local._reference_identity(
        trainer, init_adapter={"weights_sha256": "fixture"}, model=local.DEFAULT_MODEL
    )
    assert reference["kind"] == "initial_saved_adapter"
    assert reference["separate_base_model_allocated"] is False
    assert reference["adapter"]["trainable_tensors"] == 0
    with torch.no_grad(), use_adapter(policy, "ref"):
        assert policy(input_ids=torch.tensor([[2, 4, 1]])).logits.requires_grad is False
    assert policy.active_adapters == ["default"]
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B.default." in name:
                parameter.fill_(0.25)
    saved_default = local._adapter_tensor_identity(policy, "default")
    saved_reference = local._adapter_tensor_identity(policy, "ref")
    checkpoint = tmp_path / "checkpoint"
    trainer.save_model(str(checkpoint))
    assert (checkpoint / "adapter_model.safetensors").is_file()
    assert (checkpoint / "ref/adapter_model.safetensors").is_file()
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B.default." in name:
                parameter.fill_(0.75)
            if "lora_B.ref." in name:
                parameter.fill_(0.9)
    trainer._load_from_checkpoint(str(checkpoint))
    assert local._adapter_tensor_identity(policy, "default") == saved_default
    assert local._adapter_tensor_identity(policy, "ref") == saved_reference
    audit = local._audit_callback(tmp_path, None)
    audit.expected_reference = saved_reference
    audit.on_train_begin(None, SimpleNamespace(), SimpleNamespace(), model=policy)
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B.ref." in name:
                parameter.add_(0.1)
    with pytest.raises(local.LocalVLMError, match="reference adapter changed"):
        audit.on_train_end(None, SimpleNamespace(), SimpleNamespace(), model=policy)
    with pytest.raises(local.LocalVLMError, match="pinned initial anchor"):
        audit.on_train_begin(None, SimpleNamespace(), SimpleNamespace(), model=policy)
    corrupt_checkpoint = tmp_path / "altered-reference-checkpoint"
    trainer.save_model(str(corrupt_checkpoint))
    with pytest.raises(local.LocalVLMError, match="pinned initial anchor"):
        trainer._load_from_checkpoint(str(corrupt_checkpoint))


# --- fresh-pool screened GRPO, EG-SFT and sequential SFT (study aircraft-freshpool-v1) ---


def _hand_w4(p):
    """w_4(p) written out term by term from TRL's advantage (Bessel std, eps 1e-4)."""
    s13, s2 = 0.5, 3**-0.5  # unbiased std of a 4-group with 1 or 3 correct, and with 2
    return (
        4 * p * (1 - p) ** 2 * 1 * 3 / (4 * (s13 + 1e-4))
        + 6 * p**2 * (1 - p) * 2 * 2 / (4 * (s2 + 1e-4))
        + 4 * p**3 * 3 * 1 / (4 * (s13 + 1e-4))
    )


def test_acceptance_band_matches_preregistered_p_interval():
    assert local.group_useful_probability(0.5) == 1 - 2 / 16
    # u(p) >= 0.5 is p in [0.1594, 0.8406] (study.json screening.accept).
    for p in (0.159, 0.841):
        assert local.group_useful_probability(p) < 0.5
    for p in (0.16, 0.5, 0.84):
        assert local.group_useful_probability(p) >= 0.5
    assert local.group_useful_probability(0.3, 2) == pytest.approx(2 * 0.3 * 0.7)
    with pytest.raises(ValueError):
        local.group_useful_probability(1.01)


def test_w4_enumeration_matches_hand_expansion_and_trl_monte_carlo():
    torch = pytest.importorskip("torch")
    released_trl()
    from trl.trainer.utils import nanstd

    torch.set_num_threads(2)
    for p in (0.0, 0.1, 0.3, 0.5, 0.8, 0.95):
        assert local.expected_gradient_weight(p) == pytest.approx(_hand_w4(p), rel=1e-12)
    assert local.egsft_weight(0.5) == 1.0
    assert local.egsft_weight(0.3) == pytest.approx(_hand_w4(0.3) / _hand_w4(0.5))
    generator = torch.Generator().manual_seed(791)
    groups = 200_000
    for p in (0.2, 0.5, 0.8):
        # Three-way softmax policy, gold = token 0. d log pi(y) / d logits = onehot(y) - pi.
        pi = torch.tensor([p, (1 - p) / 2, (1 - p) / 2], dtype=torch.float64)
        samples = torch.multinomial(pi, groups * 4, replacement=True, generator=generator)
        rewards = (samples == 0).to(torch.float64).view(groups, 4)
        # Released TRL 1.16 scale_rewards="group" advantage.
        mean = rewards.mean(dim=1, keepdim=True)
        std = nanstd(rewards, dim=1).unsqueeze(1)
        advantages = (rewards - mean) / (std + 1e-4)
        scores = torch.nn.functional.one_hot(samples, 3).to(torch.float64) - pi
        per_group = (advantages.view(-1, 1) * scores).view(groups, 4, 3).sum(dim=1)
        estimate = per_group.mean(dim=0)
        standard_error = per_group.std(dim=0) / groups**0.5
        expected = _hand_w4(p) * (torch.tensor([1.0, 0, 0], dtype=torch.float64) - pi)
        assert ((estimate - expected).abs() <= 4 * standard_error + 1e-9).all(), (p, estimate)
        assert local.expected_gradient_weight(p) == pytest.approx(_hand_w4(p))


class _BigramPolicy:
    """Next-token probabilities depend only on the current token: an exact fixture."""

    def __init__(self, table, vocab):
        torch = pytest.importorskip("torch")
        self.torch = torch
        logits = torch.full((vocab, vocab), -torch.inf)
        for token, row in table.items():
            for target, probability in row.items():
                logits[token, target] = math.log(probability)
        self.parameter = torch.nn.Parameter(logits.clone().nan_to_num(neginf=-1e9))
        self.training = True
        self.modes = []

    def eval(self):
        self.training = False
        self.modes.append("eval")
        return self

    def train(self, mode=True):
        self.training = mode
        self.modes.append(f"train={mode}")
        return self

    def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
        assert use_cache is False and bool((attention_mask == 1).all())
        logits = self.parameter[input_ids][:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


# Tokens: 0 pad, 1 EOS, 2 image (suppressed by the sampler), 3/4/5 prompt markers, 6 gold, 7 tail.
_EOS, _IMAGE, _GOLD, _TAIL = 1, 2, 6, 7
_PLANTED = {
    3: {_GOLD: 0.45, _IMAGE: 0.5, 4: 0.05},  # all-correct after suppression: P(gold)=0.9
    4: {_GOLD: 0.05, _IMAGE: 0.5, 5: 0.45},  # all-wrong after suppression: P(gold)=0.1
    5: {_GOLD: 0.25, _IMAGE: 0.5, 4: 0.25},  # mixed after suppression: P(gold)=0.5
    _GOLD: {_EOS: 0.495, _IMAGE: 0.5, 4: 0.005},  # P(EOS | gold)=0.99 after suppression
    _EOS: {_TAIL: 1.0},
}


def _gold_batch(marker, tail=True):
    torch = pytest.importorskip("torch")
    # Chat templates supervise a token after the assistant EOS; sampling stops at EOS.
    completion = [_GOLD, _EOS] + ([_TAIL] if tail else [])
    ids = torch.tensor([[0 + 3, marker] + completion])
    labels = torch.tensor([[-100, -100] + completion])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def test_gold_probability_uses_sampler_suppression_and_stops_at_first_eos():
    policy = _BigramPolicy(_PLANTED, 8)
    kwargs = dict(eos_token_ids=[_EOS], suppress_tokens=[_IMAGE], temperature=1.0)
    for marker, expected in ((3, 0.9 * 0.99), (4, 0.1 * 0.99), (5, 0.5 * 0.99)):
        p, canonical = local._batch_gold_probability(policy, _gold_batch(marker), **kwargs)
        assert p == pytest.approx(expected, rel=1e-6)
        assert canonical == [_GOLD, _EOS]
    raw, _ = local._batch_gold_probability(
        policy, _gold_batch(3), eos_token_ids=[_EOS], suppress_tokens=[], temperature=1.0
    )
    assert raw == pytest.approx(0.45 * 0.495, rel=1e-6)
    # Without the sampler's suppression the planted all-correct image would be accepted.
    assert local.group_useful_probability(raw) >= 0.5 > local.group_useful_probability(0.9 * 0.99)
    alternative, _ = local._batch_gold_probability(
        policy, _gold_batch(5), alternatives=[[4, _EOS], [_GOLD, _EOS]], **kwargs
    )
    # P(4 | marker 5) = 0.5 after suppression; P(EOS | 4) is 0 in the fixture.
    assert alternative == pytest.approx(0.5 * 0.99, rel=1e-6)
    table = {**_PLANTED, 4: {_EOS: 1.0}}
    alternative, _ = local._batch_gold_probability(
        _BigramPolicy(table, 8), _gold_batch(5), alternatives=[[4, _EOS]], **kwargs
    )
    assert alternative == pytest.approx(0.5 * 0.99 + 0.5 * 1.0, rel=1e-6)
    with pytest.raises(local.LocalVLMError, match="only EOS"):
        local._batch_gold_probability(policy, _gold_batch(5), alternatives=[[4]], **kwargs)
    batch = _gold_batch(5)
    batch["labels"][0, 3] = 99
    with pytest.raises(local.LocalVLMError, match="differ from the supervised"):
        local._batch_gold_probability(policy, batch, **kwargs)


def test_gold_probability_reads_the_row_through_the_collator_and_restores_mode(diagnostic):
    spec, rows, _ = load_diagnostic(diagnostic)
    seen = []

    def collator(records):
        seen.append(records[0])
        return _gold_batch(4)

    policy = _BigramPolicy(_PLANTED, 8)
    p = local.gold_probability(
        policy,
        None,
        rows[0],
        spec,
        replace(local.Profile(), response_format="answer_only"),
        "cpu",
        eos_token_ids=[_EOS],
        suppress_tokens=[_IMAGE],
        collator=collator,
    )
    assert p == pytest.approx(0.1 * 0.99, rel=1e-6)
    assert seen[0]["completion"] == [{"role": "assistant", "content": rows[0]["label"]}]
    assert seen[0]["image"].mode == "RGB"
    assert policy.modes == ["eval", "train=True"] and policy.training


def test_screen_rejects_planted_all_correct_and_all_wrong_accepts_mixed_and_refuses_exhaustion(
    tmp_path,
):
    policy = _BigramPolicy(_PLANTED, 8)
    markers = {"correct": 3, "wrong": 4, "mixed": 5}
    ids = ["correct", "wrong", "mixed", "correct"]

    def score(model, index):
        p, _ = local._batch_gold_probability(
            model,
            _gold_batch(markers[ids[index]]),
            eos_token_ids=[_EOS],
            suppress_tokens=[_IMAGE],
            temperature=1.0,
        )
        return p

    journal = tmp_path / "screen.jsonl"
    screen = local._ScreenedScan(
        journal,
        ids,
        threshold=0.5,
        target_accepted=2,
        group_size=4,
        score=score,
        adapter_identity=lambda model: "fixture-adapter",
    )
    assert screen.next_accepted(policy, 1) == 2
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["correct", "wrong", "mixed"]
    assert [row["accepted"] for row in rows] == [False, False, True]
    assert [row["p"] for row in rows] == pytest.approx([0.891, 0.099, 0.495], rel=1e-6)
    assert {row["adapter_sha256"] for row in rows} == {"fixture-adapter"}
    assert all(row["scorer_seconds"] >= 0 and row["global_step"] == 1 for row in rows)
    with pytest.raises(local.LocalVLMError, match="exhausted S"):
        screen.next_accepted(policy, 2)
    assert screen.position == 4 and screen.accepted_ids == ["mixed"]
    with pytest.raises(local.LocalVLMError, match="differs from committed prompt exposure"):
        screen.save(tmp_path, [])
    screen.save(tmp_path, [{"id": "mixed"}])
    state = json.loads((tmp_path / "screen.json").read_text())
    restored = local._ScreenedScan(
        journal, ids, threshold=0.5, target_accepted=2, group_size=4, resume_state=state
    )
    assert restored.position == 4 and restored.accepted_ids == ["mixed"]
    for change in ({"threshold": 0.4}, {"scan_ids_sha256": "other"}, {"position": 3}):
        with pytest.raises(local.LocalVLMError, match="scan identity"):
            local._ScreenedScan(
                journal,
                ids,
                threshold=0.5,
                target_accepted=2,
                group_size=4,
                resume_state=dict(state, **change),
            )


def test_scorer_probability_matches_sampled_exact_match_frequency():
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(2)
    torch.manual_seed(2718)
    config = LlamaConfig(
        vocab_size=6,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        initializer_range=0.6,
    )
    model = LlamaForCausalLM(config).eval()
    # At this prompt the fixture model puts ~0.51 on token 0, which plays the image token:
    # the sampler suppresses it, so P(EOS first) is ~0.41 suppressed and ~0.20 raw.
    prompt, suppressed = [2, 5, 4], 0
    samples = 2000
    with torch.no_grad():
        generated = model.generate(
            input_ids=torch.tensor([prompt] * samples),
            attention_mask=torch.ones(samples, len(prompt), dtype=torch.long),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            suppress_tokens=[suppressed],
            eos_token_id=[1],
            pad_token_id=0,
            max_new_tokens=3,
        )[:, len(prompt) :].tolist()
    completions = []
    for tokens in generated:
        stop = tokens.index(1) if 1 in tokens else len(tokens) - 1
        completions.append(tokens[: stop + 1])
    assert not any(suppressed in tokens for tokens in completions)  # the sampler suppressed
    for gold in ([1], [3, 1], [5, 1]):
        ids = torch.tensor([prompt + gold + [4]])
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([[-100] * len(prompt) + gold + [4]]),
        }
        with torch.no_grad():
            p, _ = local._batch_gold_probability(
                model, batch, eos_token_ids=[1], suppress_tokens=[suppressed], temperature=1.0
            )
        frequency = sum(tokens == gold for tokens in completions) / samples
        standard_error = (p * (1 - p) / samples) ** 0.5
        assert abs(frequency - p) <= 3 * standard_error, (gold, frequency, p)


def test_egsft_weight_scales_sft_loss_and_carries_no_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    theta = torch.nn.Parameter(torch.tensor([0.5, -1.0]))

    class Base:
        state = SimpleNamespace(global_step=6)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            assert "labels" in inputs
            loss = (theta * torch.tensor([2.0, 3.0])).sum()
            return (loss, "outputs") if return_outputs else loss

    policy = _BigramPolicy(_PLANTED, 8)
    trainer = local._trainer_with_egsft(
        Base,
        eos_token_ids=[_EOS],
        suppress_tokens=[_IMAGE],
        temperature=1.0,
        group_size=4,
        journal=tmp_path / "egsft.jsonl",
    )()
    trainer._vision_rl_prompt_id = "row-5"
    loss = trainer.compute_loss(policy, _gold_batch(5))
    p = 0.5 * 0.99
    weight = _hand_w4(p) / _hand_w4(0.5)
    assert float(loss.detach()) == pytest.approx(weight * (0.5 * 2 - 1.0 * 3), rel=1e-6)
    loss.backward()
    assert theta.grad.tolist() == pytest.approx([2 * weight, 3 * weight], rel=1e-6)
    assert policy.parameter.grad is None  # p never enters the gradient
    row = json.loads((tmp_path / "egsft.jsonl").read_text())
    assert row == {
        "global_step": 7,
        "id": "row-5",
        "p": pytest.approx(p, rel=1e-6),
        "weight": pytest.approx(weight, rel=1e-6),
        "invocation_id": row["invocation_id"],
    }
    assert row["invocation_id"].isdigit()
    loss, outputs = trainer.compute_loss(policy, _gold_batch(3), return_outputs=True)
    assert outputs == "outputs"
    assert float(loss.detach()) == pytest.approx(_hand_w4(0.891) / _hand_w4(0.5) * -2.0, rel=1e-6)


def test_scorer_time_is_a_sub_account_inside_the_active_step(tmp_path, monkeypatch):
    plain = local._ActiveStepTimer(tmp_path / "plain", 40.0, "cpu")
    plain.begin()
    assert "observed_scorer_seconds" not in json.loads(plain.path.read_text())
    plain.end()
    assert not any("scorer" in key for key in plain.summary())
    with pytest.raises(local.LocalVLMError, match="open active optimizer step"):
        plain.charge_scorer(1.0)
    clock = iter([100.0, 110.0])
    monkeypatch.setattr(local.time, "monotonic", lambda: next(clock))
    timer = local._ActiveStepTimer(tmp_path / "screened", 40.0, "cpu")
    timer.enable_scorer()
    with pytest.raises(local.LocalVLMError, match="open active optimizer step"):
        timer.charge_scorer(1.0)
    timer.begin()
    timer.charge_scorer(3.0)
    timer.end()
    report = timer.summary()
    assert timer.completed == 10.0  # scorer seconds are already inside the step span
    assert report["invocation_scorer_seconds"] == 3.0
    assert report["observed_scorer_seconds_within_active"] == 3.0
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    for name in local.ACTIVE_CHECKPOINT_FILES:
        (checkpoint / name).write_text(
            '{"global_step": 1}' if name == "trainer_state.json" else name
        )
    (checkpoint / "screen.json").write_text("{}")
    timer.save(checkpoint, 1)
    assert local._load_active_time(checkpoint, 40.0)["screen_sha256"]
    (checkpoint / "screen.json").write_text('{"changed": true}')
    with pytest.raises(local.LocalVLMError, match="screen identity"):
        local._load_active_time(checkpoint, 40.0)
    (checkpoint / "screen.json").unlink()
    with pytest.raises(local.LocalVLMError, match="screen identity"):
        local._load_active_time(checkpoint, 40.0)


def _tiny_text_policy(seed=0):
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(seed)
    vocab = {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3, "a": 4, "b": 5, "c": 6}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    )
    tokenizer.backend_tokenizer.pre_tokenizer = import_pre_tokenizer()
    policy = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=7,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                pad_token_id=0,
                eos_token_id=1,
                bos_token_id=2,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    return tokenizer, policy


def import_pre_tokenizer():
    from tokenizers import pre_tokenizers

    return pre_tokenizers.WhitespaceSplit()


# Scan fixture: "R" rows have a five-token gold (p ~ 1e-4, rejected), "A" rows have the
# one-token gold EOS (p ~ 1/7, accepted at threshold 0.05 with G=2). Four accepted
# updates therefore take the A rows at scan positions 1, 3, 4, 6 and stop at position 7.
_SCAN = [("s0", "R"), ("s1", "A"), ("s2", "R"), ("s3", "A"), ("s4", "A"), ("s5", "R")]
_SCAN += [("s6", "A"), ("s7", "A")]
_GOLDS = {"A": [1], "R": [4, 5, 6, 4, 1]}


def _screened_grpo_run(output, *, stop_after_steps=None, resume=None):
    trl = released_trl()
    import torch
    from datasets import Dataset

    torch.set_num_threads(2)
    tokenizer, policy = _tiny_text_policy()
    dataset = Dataset.from_dict(
        {
            "prompt": ["a b", "b c", "c a", "a a", "b b", "c c", "a c", "b a"],
            "id": [identifier for identifier, _ in _SCAN],
            "label": ["a"] * len(_SCAN),
        }
    )
    profile = replace(local.Profile(), num_generations=2, max_completion_length=4)
    args = trl.GRPOConfig(
        **local.trainer_kwargs(
            "grpo",
            output,
            profile,
            device="cpu",
            max_steps=4,
            learning_rate=5e-2,
            seed=42,
            save_steps=1,
        )
    )
    active, exposure_state, screen_state = None, None, None
    if resume is not None:
        active = local._load_active_time(resume, None)
        exposure_state = json.loads((resume / "exposure.json").read_text())
        screen_state = json.loads((resume / "screen.json").read_text())
    ids = [identifier for identifier, _ in _SCAN]
    exposure = local._ExposureLedger(output, ids, exposure_state)
    screen = local._ScreenedScan(
        output / "screen_journal.jsonl",
        ids,
        threshold=0.05,
        target_accepted=4,
        group_size=2,
        resume_state=screen_state,
    )
    audit = local._audit_callback(
        output,
        None,
        resume_state=active,
        exposure=exposure,
        stop_after_steps=stop_after_steps,
        screen=screen,
    )

    def reward(completions, completion_ids, id, **_):
        # Length parity: a varied reward, so updates change the adapter and later scores.
        values = [float(len(tokens) % 2) for tokens in completion_ids]
        for start in range(0, len(id), 2):
            audit.observe_prompt(id[start])
            with (output / "reward_groups.jsonl").open("a") as handle:
                group = {
                    "id": id[start],
                    "rewards": values[start : start + 2],
                    "completion_ids": completion_ids[start : start + 2],
                }
                handle.write(json.dumps(group) + "\n")
        return values

    constructor = local._trainer_with_active_time(
        local._trainer_with_screening(
            local._trainer_with_grpo_resume(trl.GRPOTrainer), screen, dataset
        )
    )
    trainer = constructor(
        model=policy,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[audit],
        reward_funcs=reward,
    )
    transform = local._sampler_transform(trainer)
    assert transform == {"temperature": 1.0, "suppress_tokens": [], "eos_token_ids": [1]}

    def score(model, index):
        prompt = tokenizer(dataset[index]["prompt"], add_special_tokens=False)["input_ids"]
        gold = _GOLDS[_SCAN[index][1]]
        ids = torch.tensor([prompt + gold])
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([[-100] * len(prompt) + gold]),
        }
        was_training = model.training
        model.eval()
        with torch.no_grad():
            p, _ = local._batch_gold_probability(
                model,
                batch,
                eos_token_ids=transform["eos_token_ids"],
                suppress_tokens=transform["suppress_tokens"],
                temperature=transform["temperature"],
            )
        model.train(was_training)
        return p

    screen.score = score
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return trainer


def _committed(entries):
    return [
        {
            k: v
            for k, v in entry.items()
            if k not in {"scorer_seconds", "adapter_hash_seconds", "invocation_id"}
        }
        for entry in entries
    ]


def test_screened_grpo_resume_is_identical_to_uninterrupted(tmp_path):
    released_trl()
    runs = {}
    straight = tmp_path / "straight"
    _screened_grpo_run(straight)
    runs["4"] = straight
    for first in (2, 1):
        output = tmp_path / f"split-{first}"
        _screened_grpo_run(output, stop_after_steps=first)
        assert not (output / "checkpoint-4").exists()
        _screened_grpo_run(output, resume=output / f"checkpoint-{first}")
        runs[f"{first}+{4 - first}"] = output
    final = {}
    for name, output in runs.items():
        checkpoint = output / "checkpoint-4"
        screen = json.loads((checkpoint / "screen.json").read_text())
        exposure = json.loads((checkpoint / "exposure.json").read_text())
        groups = [
            json.loads(line) for line in (output / "reward_groups.jsonl").read_text().splitlines()
        ]
        physical = (output / "screen_journal.jsonl").read_text().splitlines()
        assert len(physical) == screen["position"] == 7  # nothing scored twice or discarded
        assert local._load_active_time(checkpoint, None)["screen_sha256"]
        final[name] = {
            "adapter": (checkpoint / "adapter_model.safetensors").read_bytes(),
            "position": screen["position"],
            "journal": _committed(screen["entries"]),
            "exposure": [entry["id"] for entry in exposure["entries"]],
            "groups": [(g["id"], g["rewards"], g["completion_ids"]) for g in groups],
        }
    assert final["4"]["exposure"] == ["s1", "s3", "s4", "s6"]
    assert [entry["accepted"] for entry in final["4"]["journal"]] == [
        kind == "A" for _, kind in _SCAN[:7]
    ]
    assert [entry["global_step"] for entry in final["4"]["journal"]] == [1, 1, 2, 2, 3, 4, 4]
    # Each update's candidates are scored under the adapter that update starts from.
    hashes = [entry["adapter_sha256"] for entry in final["4"]["journal"]]
    assert hashes[0] == hashes[1] and hashes[2] == hashes[3] and hashes[5] == hashes[6]
    assert len({hashes[1], hashes[3], hashes[4], hashes[6]}) == 4
    assert any(rewards[0] != rewards[1] for _, rewards, _ in final["4"]["groups"])
    assert final["2+2"] == final["4"]
    assert final["1+3"] == final["4"]


def test_screened_trainer_refuses_scan_exhaustion_inside_training(tmp_path):
    released_trl()
    global _SCAN
    saved = _SCAN
    _SCAN = [
        (identifier, "R") if identifier != "s1" else (identifier, "A") for identifier, _ in saved
    ]
    try:
        with pytest.raises(local.LocalVLMError, match="exhausted S"):
            _screened_grpo_run(tmp_path / "out")
    finally:
        _SCAN = saved
    journal = (tmp_path / "out" / "screen_journal.jsonl").read_text().splitlines()
    assert len(journal) == 8  # the whole scan was read before refusing


def test_sequential_sft_replays_given_order_cycles_and_egsft_trains(tmp_path):
    trl = released_trl()
    import torch
    from datasets import Dataset

    torch.set_num_threads(2)
    assert "train_sampling_strategy" not in local.trainer_kwargs(
        "sft",
        tmp_path,
        local.Profile(),
        device="cpu",
        max_steps=1,
        learning_rate=1e-5,
        seed=42,
        save_steps=1,
    )
    with pytest.raises(ValueError, match="sequential for SFT"):
        local.trainer_kwargs(
            "grpo",
            tmp_path,
            local.Profile(),
            device="cpu",
            max_steps=1,
            learning_rate=1e-5,
            seed=42,
            save_steps=1,
            order="sequential",
        )
    order = ["r2", "r0", "r1"]
    for mode in ("sft", "egsft"):
        output = tmp_path / mode
        tokenizer, policy = _tiny_text_policy()
        dataset = Dataset.from_dict(
            {"id": order, "prompt": ["a b", "b c", "c a"], "completion": [" a", " b", " c"]}
        )
        args = trl.SFTConfig(
            **local.trainer_kwargs(
                "sft",
                output,
                local.Profile(),
                device="cpu",
                max_steps=5,
                learning_rate=1e-2,
                seed=42,
                save_steps=5,
                order="sequential",
            )
        )
        exposure = local._ExposureLedger(output, order)
        audit = local._audit_callback(output, None, exposure=exposure)
        constructor = trl.SFTTrainer
        if mode == "egsft":
            constructor = local._trainer_with_egsft(
                constructor,
                eos_token_ids=[1],
                suppress_tokens=[],
                temperature=1.0,
                group_size=4,
                journal=output / "egsft_weights.jsonl",
                journal_io_context=audit.timer.exclude_journal_io,
            )
        constructor = local._trainer_with_active_time(
            local._trainer_with_sft_exposure(constructor, audit)
        )
        trainer = constructor(
            model=policy,
            args=args,
            processing_class=tokenizer,
            train_dataset=dataset,
            callbacks=[audit],
        )
        trainer.data_collator = local._sft_collator_with_ids(trainer.data_collator)
        trainer.train()
        committed = json.loads((output / "checkpoint-5" / "exposure.json").read_text())
        assert [entry["id"] for entry in committed["entries"]] == order + order[:2]
        assert audit.summary["nonzero_gradient_steps"] == 5
    weights = [
        json.loads(line)
        for line in (tmp_path / "egsft" / "egsft_weights.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in weights] == order + order[:2]
    for row in weights:
        assert 0 < row["p"] < 1
        assert row["weight"] == pytest.approx(_hand_w4(row["p"]) / _hand_w4(0.5), rel=1e-9)


def _settings_after_model_refusal(diagnostic, tmp_path, monkeypatch, name, **kwargs):
    """Run train_local until it opens the (absent) model directory; return its settings."""
    monkeypatch.setattr(
        local,
        "preflight",
        lambda *_: {"adapter_source_sha256": local._file_sha256(Path(local.__file__))},
    )
    output = tmp_path / name
    with pytest.raises((FileNotFoundError, ImportError, ModuleNotFoundError)):
        local.train_local(
            diagnostic / "train.jsonl",
            None,
            diagnostic / "data_spec.json",
            output,
            model_directory=tmp_path / "absent-model",
            device="cpu",
            max_steps=4,
            train_only=True,
            **kwargs,
        )
    record = json.loads((output / "local_training.json").read_text())
    assert record["status"] == "failed"
    return record


def test_new_options_leave_existing_settings_identity_unchanged(diagnostic, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    record = _settings_after_model_refusal(diagnostic, tmp_path, monkeypatch, "plain")
    assert record["settings"] == {
        "mode": "grpo",
        "max_steps": 4,
        "learning_rate": 1e-5,
        "beta": 0.04,
        "seed": 42,
        "save_steps": 8,
        "max_active_seconds": None,
        "probe": False,
        "train_only": True,
        "init_adapter": None,
        "device": "cpu",
        "initial_adapter_precision_transfer": None,
    }
    assert record["algorithm"] == "TRL-GRPO" and "screen_sampler_transform" not in record
    sft = _settings_after_model_refusal(
        diagnostic,
        tmp_path,
        monkeypatch,
        "sequential",
        mode="egsft",
        order="sequential",
        cycle=True,
    )
    assert sft["settings"]["order"] == "sequential" and sft["settings"]["cycle"] is True
    assert sft["settings"]["egsft"]["advantage_eps"] == 1e-4
    assert sft["algorithm"] == "TRL-EGSFT" and sft["settings"]["beta"] is None
    screened = _settings_after_model_refusal(
        diagnostic,
        tmp_path,
        monkeypatch,
        "screened",
        screen_journal=tmp_path / "screened" / "screen.jsonl",
        target_accepted=4,
        screen_threshold=0.5,
    )
    assert screened["settings"]["screening"]["journal"] == "screen.jsonl"
    assert screened["settings"]["screening"]["target_accepted"] == 4


def test_screen_and_order_options_refuse_before_model_access(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must refuse before model"))
    output = tmp_path / "out"
    base = dict(
        train=diagnostic / "train.jsonl",
        validation=None,
        data_spec=diagnostic / "data_spec.json",
        output=output,
        model_directory=tmp_path / "model",
        max_steps=4,
        train_only=True,
    )
    journal = output / "screen.jsonl"
    for kwargs, match in (
        ({"order": "sequential"}, "sequential for sft/egsft"),
        ({"mode": "sft", "cycle": True}, "cycle requires sequential"),
        ({"mode": "sft", "screen_journal": journal}, "only to GRPO"),
        ({"screen_threshold": 0.4}, "require screen_journal"),
        ({"target_accepted": 4}, "require screen_journal"),
        ({"screen_journal": journal}, "equal the fixed target_accepted"),
        ({"screen_journal": journal, "target_accepted": 4, "screen_threshold": 1.0}, "(0, 1)"),
        ({"screen_journal": journal, "max_steps": 9, "target_accepted": 9}, "exhaust S"),
        ({"screen_journal": tmp_path / "elsewhere.jsonl", "target_accepted": 4}, "inside output"),
        ({"mode": "sft", "order": "sequential", "max_steps": 9}, "cannot exceed"),
        ({"screen_tokenizations": tmp_path / "t.json"}, "screening or EG-SFT"),
        ({"mode": "nope"}, "sft/egsft/grpo"),
    ):
        with pytest.raises(ValueError, match=match.replace("(", r"\(").replace(")", r"\)")):
            local.train_local(**dict(base, **kwargs))
    assert not output.exists()


def test_screened_resume_requires_matching_checkpoint_screen_state(
    diagnostic, tmp_path, monkeypatch
):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must refuse before model"))
    output = tmp_path / "out"
    checkpoint = output / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    for name in (*local.ACTIVE_CHECKPOINT_FILES, "rng_state.pth"):
        (checkpoint / name).write_text(
            '{"global_step": 1}' if name == "trainer_state.json" else name
        )
    (checkpoint / "ref").mkdir()
    (checkpoint / "ref" / "adapter_model.safetensors").write_text("ref")
    kwargs = dict(
        model_directory=tmp_path / "model",
        max_steps=4,
        train_only=True,
        screen_journal=output / "screen.jsonl",
        target_accepted=4,
        resume_from_checkpoint=checkpoint,
    )
    spec_sha = local._file_sha256(diagnostic / "data_spec.json")
    rows = local.load_classification(
        diagnostic / "train.jsonl", "train", ["blue", "red"], require_all_classes=False
    )
    identity = {
        "model": local.DEFAULT_MODEL,
        "model_revision": local.MODEL_REVISIONS[local.DEFAULT_MODEL],
        "processor_model": local.DEFAULT_MODEL,
        "processor_revision": local.MODEL_REVISIONS[local.DEFAULT_MODEL],
        "profile": asdict(local.Profile()),
        "output_grammar": local.OUTPUT_GRAMMAR,
        "data_spec_sha256": spec_sha,
        "train": local._classification_identity(diagnostic / "train.jsonl", rows),
        "validation": None,
        "image_transform_source_sha256": None,
    }
    settings = {
        "mode": "grpo",
        "max_steps": 4,
        "learning_rate": 1e-5,
        "beta": 0.04,
        "seed": 42,
        "save_steps": 8,
        "max_active_seconds": None,
        "probe": False,
        "train_only": True,
        "init_adapter": None,
        "device": "mps",
        "initial_adapter_precision_transfer": None,
        "screening": {
            "journal": "screen.jsonl",
            "threshold": 0.5,
            "target_accepted": 4,
            "group_size": 4,
            "acceptance": "u(p) = 1 - p^G - (1 - p)^G >= threshold",
            "scan_order": "TRAIN manifest file order",
            "scope": local.SCREEN_SCOPE,
        },
    }
    (output / "local_training.json").write_text(
        json.dumps({"control_identity": identity, "settings": settings})
    )
    (checkpoint / "mps_rng_state.pth").write_text("mps")
    ids = [row["id"] for row in rows]
    (checkpoint / "exposure.json").write_text(
        json.dumps({"schema_version": 1, "entries": [{"global_step": 1, "id": ids[1]}]})
    )
    timer = local._ActiveStepTimer(output, None, "mps")

    def save(screen_ids_accepted):
        entries = [
            {
                "scan_position": i,
                "id": ids[i],
                "p": 0.5 if accepted else 0.01,
                "u": 0.875 if accepted else 0.0394,
                "accepted": accepted,
                "global_step": 1,
                "adapter_sha256": "x",
                "scorer_seconds": 0.1,
                "invocation_id": "1",
            }
            for i, accepted in enumerate(screen_ids_accepted)
        ]
        state = local._ScreenedScan(
            output / "screen.jsonl", ids, threshold=0.5, target_accepted=4, group_size=4
        )
        state.entries = entries
        (checkpoint / "screen.json").write_text(json.dumps(state.state()))
        timer.save(checkpoint, 1)

    save([False, False])  # accepted nothing, but exposure committed ids[1]
    with pytest.raises(local.LocalVLMError, match="differs from exposure"):
        local.train_local(
            diagnostic / "train.jsonl", None, diagnostic / "data_spec.json", output, **kwargs
        )
    (checkpoint / "screen.json").unlink()
    timer.save(checkpoint, 1)
    with pytest.raises(local.LocalVLMError, match="exactly a checkpoint screen state"):
        local.train_local(
            diagnostic / "train.jsonl", None, diagnostic / "data_spec.json", output, **kwargs
        )
    save([False, True] + [False] * (len(ids) - 2))
    with pytest.raises(local.LocalVLMError, match="already exhausted S"):
        local.train_local(
            diagnostic / "train.jsonl", None, diagnostic / "data_spec.json", output, **kwargs
        )
    save([False, True])
    with pytest.raises(pytest.fail.Exception, match="must refuse before model"):
        local.train_local(
            diagnostic / "train.jsonl", None, diagnostic / "data_spec.json", output, **kwargs
        )


def test_cli_passes_new_training_options_and_keeps_inference_clean(monkeypatch, capsys):
    def train(**kwargs):
        assert kwargs["mode"] == "egsft" and kwargs["order"] == "sequential"
        assert kwargs["cycle"] is True and kwargs["screen_journal"] is None
        assert kwargs["screen_threshold"] == 0.5 and kwargs["target_accepted"] == 600
        return {"ok": True}

    monkeypatch.setattr(local, "train_local", train)
    args = ["--train", "t.jsonl", "--data-spec", "s.json", "--output", "o", "--train-only"]
    assert local.main(args + ["--mode", "egsft", "--order", "sequential", "--cycle"]) == 0
    capsys.readouterr()

    def screened(**kwargs):
        assert kwargs["screen_journal"] == Path("o/screen.jsonl")
        assert kwargs["target_accepted"] == 150 and kwargs["screen_threshold"] == 0.5
        return {"ok": True}

    monkeypatch.setattr(local, "train_local", screened)
    assert (
        local.main(args + ["--screen-journal", "o/screen.jsonl", "--target-accepted", "150"]) == 0
    )
    capsys.readouterr()
    monkeypatch.setattr(local, "predict_train_subset", lambda *a, **k: pytest.fail("refuse"))
    predict = [
        "--predict-train",
        "t.jsonl",
        "--data-spec",
        "s.json",
        "--output",
        "o",
        "--model-directory",
        "m",
    ]
    for extra in (["--screen-journal", "j"], ["--order", "sequential"], ["--target-accepted", "5"]):
        with pytest.raises(SystemExit):
            local.main(predict + extra)


# --- SFT-T continuation and EG-SFT committed lineage (review fixes) ---

_E = (["e0", "e1", "e2"], ["a b", "b c", "c a"], [" a", " b", " c"])
_T = (["t0", "t1", "t2", "t3"], ["a a", "b b", "c c", "a c"], [" c", " a", " b", " a"])


def _seq_sft_run(
    output,
    rows,
    *,
    max_steps,
    save_steps,
    mode="sft",
    parent=None,
    resume=None,
    stop_after_steps=None,
    reset_from=None,
):
    """Tiny CPU sequential SFT through the module's wrappers, as train_local assembles them."""
    trl = released_trl()
    import torch
    from datasets import Dataset
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    ids, prompts, completions = rows
    tokenizer, policy = _tiny_text_policy()
    if reset_from is not None:  # control: parent weights only, fresh optimizer
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(policy, load_file(reset_from / "adapter_model.safetensors"))
    dataset = Dataset.from_dict({"id": ids, "prompt": prompts, "completion": completions})
    args = trl.SFTConfig(
        **local.trainer_kwargs(
            "sft",
            output,
            local.Profile(),
            device="cpu",
            max_steps=max_steps,
            learning_rate=1e-2,
            seed=42,
            save_steps=save_steps,
            order="sequential",
        )
    )
    start = resume or parent
    active = local._load_active_time(start, None) if start else None
    exposure_state = json.loads((start / "exposure.json").read_text()) if start else None
    inherited = json.loads((parent / "exposure.json").read_text())["entries"] if parent else None
    egsft_state = (
        json.loads((resume / "egsft.json").read_text()) if resume and mode == "egsft" else None
    )
    exposure = local._ExposureLedger(output, ids, exposure_state, inherited=inherited)
    audit = local._audit_callback(
        output, None, resume_state=active, exposure=exposure, stop_after_steps=stop_after_steps
    )
    constructor = trl.SFTTrainer
    if mode == "egsft":
        constructor = local._trainer_with_egsft(
            constructor,
            eos_token_ids=[1],
            suppress_tokens=[],
            temperature=1.0,
            group_size=4,
            journal=output / "egsft_weights.jsonl",
            journal_io_context=audit.timer.exclude_journal_io,
            resume_state=egsft_state,
        )
    constructor = local._trainer_with_sft_exposure(constructor, audit)
    if parent is not None:
        step = json.loads((parent / "trainer_state.json").read_text())["global_step"]
        constructor = local._trainer_with_continuation(constructor, step)
    constructor = local._trainer_with_active_time(constructor)
    trainer = constructor(
        model=policy,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[audit],
    )
    trainer.data_collator = local._sft_collator_with_ids(trainer.data_collator)
    trainer.train(resume_from_checkpoint=str(start) if start else None)
    return trainer


def _optimizer_state(checkpoint):
    import torch

    state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    return state["state"]


def test_sft_t_continuation_carries_optimizer_state_onto_new_manifest(tmp_path):
    torch = pytest.importorskip("torch")

    parent = tmp_path / "sft-e"
    _seq_sft_run(parent, _E, max_steps=3, save_steps=3)
    parent_checkpoint = parent / "checkpoint-3"
    # Fixture: one uninterrupted optimizer over E then T from its first row, cycling
    # (t0..t3, t0, t1). A continuation without reset must reproduce it exactly.
    reference = tmp_path / "reference"
    sequence = [_E[i] + _T[i] + _T[i][:2] for i in range(3)]
    _seq_sft_run(reference, sequence, max_steps=9, save_steps=9)
    expected_ids = ["e0", "e1", "e2", "t0", "t1", "t2", "t3", "t0", "t1"]

    straight = tmp_path / "sft-t"
    _seq_sft_run(straight, _T, max_steps=9, save_steps=3, parent=parent_checkpoint)
    split = tmp_path / "sft-t-split"
    _seq_sft_run(
        split, _T, max_steps=9, save_steps=3, parent=parent_checkpoint, stop_after_steps=6
    )
    assert not (split / "checkpoint-9").exists()
    _seq_sft_run(
        split,
        _T,
        max_steps=9,
        save_steps=3,
        parent=parent_checkpoint,
        resume=split / "checkpoint-6",
    )
    expected_state = _optimizer_state(reference / "checkpoint-9")
    assert set(expected_state) and all(
        float(value["step"]) == 9 for value in expected_state.values()
    )
    for run in (straight, split):
        checkpoint = run / "checkpoint-9"
        exposure = json.loads((checkpoint / "exposure.json").read_text())["entries"]
        assert [entry["id"] for entry in exposure] == expected_ids
        assert local._load_active_time(checkpoint, None)["global_step"] == 9
        assert (checkpoint / "adapter_model.safetensors").read_bytes() == (
            reference / "checkpoint-9" / "adapter_model.safetensors"
        ).read_bytes()
        state = _optimizer_state(checkpoint)
        assert set(state) == set(expected_state)
        for key, value in expected_state.items():
            for name in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(state[key][name], value[name]), (key, name)
    # The parent's optimizer moments are what was carried: three steps in, not zero.
    assert all(float(v["step"]) == 3 for v in _optimizer_state(parent_checkpoint).values())
    # Control: parent weights with a reset optimizer on the same data must differ,
    # so the equality above is evidence of carry-over, not of an insensitive check.
    reset = tmp_path / "reset"
    _seq_sft_run(
        reset,
        [_T[i] + _T[i][:2] for i in range(3)],
        max_steps=6,
        save_steps=6,
        reset_from=parent_checkpoint,
    )
    assert (reset / "checkpoint-6" / "adapter_model.safetensors").read_bytes() != (
        reference / "checkpoint-9" / "adapter_model.safetensors"
    ).read_bytes()
    assert all(float(v["step"]) == 6 for v in _optimizer_state(reset / "checkpoint-6").values())


def _finish_endpoint(output, identity, settings, step, *, stop_after_steps, stop_seconds=None):
    """Complete a fixture run's record and publish its endpoint receipt at checkpoint-{step}."""
    checkpoint = output / f"checkpoint-{step}"
    adapter = output / "final_adapter"
    adapter.mkdir(exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copyfile(checkpoint / name, adapter / name)
    (output / "adapter-source.py").write_text("# fixture adapter source\n")
    record = {
        "status": "completed",
        "stage_id": "sft-e-fixture",
        "control_identity": identity,
        "settings": settings,
        "environment": {"adapter_source_sha256": local._file_sha256(output / "adapter-source.py")},
        "optimizer_audit": {"active_time": {"fixture": True}},
        "final_adapter_sha256": local._file_sha256(adapter / "adapter_model.safetensors"),
        "reference": None,
        "requested_boundary": {
            "stop_after_steps": stop_after_steps,
            "stop_after_active_seconds": stop_seconds,
        },
    }
    (output / "local_training.json").write_text(json.dumps(record))
    local._publish_endpoint_receipt(output, record, step)
    return record


def test_continuation_identity_records_old_and_new_train_and_refuses_silent_changes(
    diagnostic, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    parent = tmp_path / "parent"
    # SFT-E fixture: declared total 6 steps, stopped at its declared boundary 3.
    _seq_sft_run(parent, _E, max_steps=6, save_steps=1, stop_after_steps=3)
    assert not (parent / "checkpoint-4").exists()
    checkpoint = parent / "checkpoint-3"
    spec = local.load_spec(diagnostic / "data_spec.json")
    rows = local.load_classification(diagnostic / "train.jsonl", "train", spec["labels"])
    subset = diagnostic / "accepted.jsonl"
    subset.write_text(
        "".join(
            line + "\n" for line in (diagnostic / "train.jsonl").read_text().splitlines()[:-1]
        )
    )
    subset_rows = local.load_classification(
        subset, "train", spec["labels"], require_all_classes=False
    )
    identity = {
        "model": local.DEFAULT_MODEL,
        "model_revision": local.MODEL_REVISIONS[local.DEFAULT_MODEL],
        "processor_model": local.DEFAULT_MODEL,
        "processor_revision": local.MODEL_REVISIONS[local.DEFAULT_MODEL],
        "profile": asdict(local.Profile()),
        "output_grammar": local.OUTPUT_GRAMMAR,
        "data_spec_sha256": local._file_sha256(diagnostic / "data_spec.json"),
        "train": local._classification_identity(subset, subset_rows),
        "validation": None,
        "image_transform_source_sha256": None,
    }
    settings = {
        "mode": "sft",
        "max_steps": 6,
        "learning_rate": 1e-5,
        "beta": None,
        "seed": 42,
        "save_steps": 3,
        "max_active_seconds": None,
        "probe": False,
        "train_only": True,
        "init_adapter": None,
        "device": "cpu",
        "initial_adapter_precision_transfer": None,
        "order": "sequential",
        "cycle": False,
    }
    _finish_endpoint(parent, identity, settings, 3, stop_after_steps=3)
    assert json.loads((parent / "endpoint_receipt.json").read_text())["status"] == (
        "boundary_reached"
    )
    monkeypatch.setattr(
        local,
        "preflight",
        lambda *_: {"adapter_source_sha256": local._file_sha256(Path(local.__file__))},
    )
    base = dict(
        train=diagnostic / "train.jsonl",
        validation=None,
        data_spec=diagnostic / "data_spec.json",
        model_directory=tmp_path / "absent-model",
        device="cpu",
        mode="sft",
        order="sequential",
        cycle=True,
        max_steps=7,
        train_only=True,
        continue_from_checkpoint=checkpoint,
    )
    output = tmp_path / "sft-t"
    with pytest.raises((FileNotFoundError, ImportError, ModuleNotFoundError)):
        local.train_local(output=output, **base)
    record = json.loads((output / "local_training.json").read_text())
    continuation = record["settings"]["continuation"]
    new_train = local._classification_identity(diagnostic / "train.jsonl", rows)
    assert continuation["parent_train"]["manifest_sha256"] == local._file_sha256(subset)
    assert continuation["train"]["manifest_sha256"] == local._file_sha256(
        diagnostic / "train.jsonl"
    )
    assert continuation["train"] == new_train == record["control_identity"]["train"]
    assert continuation["parent_global_step"] == 3
    assert continuation["parent_checkpoint_file_sha256"]["optimizer.pt"] == hashlib.sha256(
        (checkpoint / "optimizer.pt").read_bytes()
    ).hexdigest()
    assert not any(continuation["reset"].values())
    assert continuation["parent_endpoint_status"] == "boundary_reached"
    assert continuation["parent_endpoint_receipt_sha256"] == hashlib.sha256(
        (parent / "endpoint_receipt.json").read_bytes()
    ).hexdigest()

    def refuses(match, error=ValueError, **kwargs):
        target = tmp_path / f"refused-{len(list(tmp_path.iterdir()))}"
        with pytest.raises(error, match=match):
            local.train_local(output=target, **dict(base, **kwargs))

    refuses("only to sequential SFT", mode="egsft")
    refuses("only to sequential SFT", order="random", cycle=False)
    refuses("must exceed the parent", max_steps=3)
    refuses("cumulative past the parent", stop_after_steps=3)
    refuses("optimizer settings", learning_rate=2e-5)
    refuses("equals the parent's; use resume", train=subset)
    # Intermediate checkpoint of a finished endpoint: refused.
    refuses("final endpoint checkpoint", error=local.LocalVLMError,
            continue_from_checkpoint=parent / "checkpoint-2")
    # Total-budget exhausted at step 3 (declared max_steps 3, no boundary): refused.
    _finish_endpoint(parent, identity, dict(settings, max_steps=3), 3, stop_after_steps=None)
    assert json.loads((parent / "endpoint_receipt.json").read_text())["status"] == (
        "total_budget_exhausted"
    )
    refuses("boundary_reached at its declared stop_after_steps", error=local.LocalVLMError)
    # Declared boundary equals the total budget (both 3): the receipt is budget-exhausted
    # at global_step == declared, so only the status check can refuse it.
    _finish_endpoint(parent, identity, dict(settings, max_steps=3), 3, stop_after_steps=3)
    exhausted = json.loads((parent / "endpoint_receipt.json").read_text())
    assert exhausted["status"] == "total_budget_exhausted"
    assert exhausted["global_step"] == exhausted["requested_boundary"]["stop_after_steps"] == 3
    refuses("boundary_reached at its declared stop_after_steps", error=local.LocalVLMError)
    # Stopped by an active-seconds boundary, not its declared stop_after_steps: refused.
    _finish_endpoint(parent, identity, settings, 3, stop_after_steps=None, stop_seconds=0.0)
    refuses("boundary_reached at its declared stop_after_steps", error=local.LocalVLMError)
    # No receipt at all: refused; a receipt that no longer verifies: refused.
    (parent / "endpoint_receipt.json").unlink()
    refuses("finished SFT-E endpoint with a receipt", error=local.LocalVLMError)
    _finish_endpoint(parent, identity, settings, 3, stop_after_steps=3)
    (parent / "final_adapter" / "adapter_config.json").write_text("{}")
    refuses("saved adapter bytes differ", error=local.LocalVLMError)
    record = _finish_endpoint(parent, identity, settings, 3, stop_after_steps=3)
    (parent / "local_training.json").write_text(
        json.dumps(dict(record, control_identity=dict(identity, data_spec_sha256="0" * 64)))
    )
    refuses("only the TRAIN manifest identity")
    _finish_endpoint(parent, identity, settings, 3, stop_after_steps=3)
    with (checkpoint / "optimizer.pt").open("ab") as handle:
        handle.write(b"tampered")
    refuses("does not match budget/state/optimizer", error=local.LocalVLMError)


def test_egsft_checkpoint_commits_one_weight_row_per_step_across_resume(tmp_path):
    straight = tmp_path / "straight"
    _seq_sft_run(straight, _T, max_steps=5, save_steps=2, mode="egsft")
    split = tmp_path / "split"
    _seq_sft_run(split, _T, max_steps=5, save_steps=2, mode="egsft")
    # Re-run steps 3..5 from checkpoint-2: the physical journal now holds discarded rows.
    for name in ("checkpoint-4", "checkpoint-5"):
        shutil.rmtree(split / name)
    _seq_sft_run(split, _T, max_steps=5, save_steps=2, mode="egsft", resume=split / "checkpoint-2")
    physical = [
        json.loads(line) for line in (split / "egsft_weights.jsonl").read_text().splitlines()
    ]
    assert [row["global_step"] for row in physical] == [1, 2, 3, 4, 5, 3, 4, 5]
    assert len({row["invocation_id"] for row in physical}) == 2
    expected = json.loads((straight / "checkpoint-5" / "egsft.json").read_text())["entries"]
    committed = json.loads((split / "checkpoint-5" / "egsft.json").read_text())["entries"]
    assert [row["global_step"] for row in committed] == [1, 2, 3, 4, 5]
    assert [row["id"] for row in committed] == ["t0", "t1", "t2", "t3", "t0"]
    assert [row["invocation_id"] for row in committed[:2]] == [physical[0]["invocation_id"]] * 2
    assert [row["invocation_id"] for row in committed[2:]] == [physical[5]["invocation_id"]] * 3
    def strip(rows):
        return [{k: v for k, v in row.items() if k != "invocation_id"} for row in rows]

    assert strip(committed) == strip(expected)
    assert local._load_active_time(split / "checkpoint-5", None)["egsft_sha256"]
    # The endpoint receipt covers egsft.json and verifies it against active_time.json.
    settings = {"mode": "egsft", "max_steps": 5, "max_active_seconds": None}
    _finish_endpoint(split, {"fixture": "egsft"}, settings, 5, stop_after_steps=None)
    receipt = local.read_endpoint_receipt(split)
    assert receipt["checkpoint"]["sha256"]["egsft.json"] == hashlib.sha256(
        (split / "checkpoint-5" / "egsft.json").read_bytes()
    ).hexdigest()
    with pytest.raises(local.LocalVLMError, match="EG-SFT weight lineage"):
        _finish_endpoint(split, {"fixture": "egsft"}, dict(settings, mode="sft"), 5,
                         stop_after_steps=None)
    _finish_endpoint(split, {"fixture": "egsft"}, settings, 5, stop_after_steps=None)
    original = (split / "checkpoint-5" / "egsft.json").read_bytes()
    tampered = json.loads(original)
    tampered["entries"][0]["weight"] = 0.123
    (split / "checkpoint-5" / "egsft.json").write_text(json.dumps(tampered))
    with pytest.raises(local.LocalVLMError, match="saved adapter bytes differ"):
        local.read_endpoint_receipt(split)  # receipt hash catches changed bytes
    with pytest.raises(local.LocalVLMError, match="EG-SFT weight lineage"):
        # A receipt re-issued over the changed file still fails the active-time binding.
        _finish_endpoint(split, {"fixture": "egsft"}, settings, 5, stop_after_steps=None)
    (split / "checkpoint-5" / "egsft.json").write_bytes(original)
    (split / "checkpoint-5" / "egsft.json").write_text('{"schema_version": 1, "entries": []}')
    with pytest.raises(local.LocalVLMError, match="EG-SFT weight identity"):
        local._load_active_time(split / "checkpoint-5", None)


def test_screen_hashes_adapter_once_per_update_outside_scorer_seconds(tmp_path):
    hashed = []

    def slow_hash(model):
        hashed.append(model)
        return "fixture-adapter"

    scan = local._ScreenedScan(
        tmp_path / "screen.jsonl",
        ["r0", "r1", "r2"],
        threshold=0.5,
        target_accepted=1,
        group_size=4,
        score=lambda model, index: [0.001, 0.002, 0.5][index],
        adapter_identity=slow_hash,
    )
    assert scan.next_accepted("model", 1) == 2
    assert hashed == ["model"]  # one hash for the update's three candidates
    rows = [json.loads(line) for line in (tmp_path / "screen.jsonl").read_text().splitlines()]
    assert [row["adapter_sha256"] for row in rows] == ["fixture-adapter"] * 3
    assert rows[0]["adapter_hash_seconds"] >= 0
    assert [row["adapter_hash_seconds"] for row in rows[1:]] == [0.0, 0.0]
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "screen.json").write_text(json.dumps(scan.state()))
    (checkpoint / "active_time.json").write_text(
        json.dumps({"screen_sha256": local._file_sha256(checkpoint / "screen.json")})
    )
    assert local.committed_screen_entries(checkpoint) == scan.entries
    (checkpoint / "screen.json").write_text(json.dumps(dict(scan.state(), entries=[])))
    with pytest.raises(local.LocalVLMError, match="not bound"):
        local.committed_screen_entries(checkpoint)


def test_exposure_ledger_inherits_parent_prefix_verbatim(tmp_path):
    parent = [{"global_step": 1, "id": "e0"}, {"global_step": 2, "id": "e1"}]
    own = parent + [{"global_step": 3, "id": "t0"}]
    # Parent IDs come from the parent's TRAIN and need not be in this one.
    ledger = local._ExposureLedger(tmp_path, ["t0", "t1"], {"entries": own}, inherited=parent)
    assert ledger.entries == own
    with pytest.raises(local.LocalVLMError, match="TRAIN identity"):
        local._ExposureLedger(tmp_path, ["t0", "t1"], {"entries": own})
    swapped = [dict(parent[0], id="e9"), parent[1], own[2]]
    with pytest.raises(local.LocalVLMError, match="continuation parent's"):
        local._ExposureLedger(tmp_path, ["t0"], {"entries": swapped}, inherited=parent)
    with pytest.raises(local.LocalVLMError, match="TRAIN identity"):
        local._ExposureLedger(
            tmp_path,
            ["t0"],
            {"entries": parent + [{"global_step": 3, "id": "e0"}]},
            inherited=parent,
        )


def test_gold_nll_improved_is_none_without_a_pre_score():
    # A train-only SFT-T continuation records no before_train_gold_nll (its pre-score is the
    # parent endpoint); the probe summary must say "not measured", not raise KeyError.
    assert local.gold_nll_improved({"before_note": "continuation"}, 1.25) is None
    assert local.gold_nll_improved({"before_train_gold_nll": 2.0}, 1.25) is True
    assert local.gold_nll_improved({"before_train_gold_nll": 1.25}, 1.25) is False


# ---------------------------------------------------------------------------------------
# Signal-limit study trainer options (docs/aircraft-rl-next-study-v1.json work list).
# Every expected value below is hand-computed from the fixture, never read off the subject.


@pytest.mark.parametrize("group", [4, 8, 16])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_dr_grpo_nostd_weight_is_g_minus_one_times_p(group, p):
    hand = {4: 3, 8: 7, 16: 15}[group] * p
    assert abs(local.expected_gradient_weight(p, group, scale_rewards="none") - hand) <= 1e-12
    # Normalised at p = 0.5 the no-std limit is 2p for every G.
    assert local.egsft_weight(p, group, scale_rewards="none") == pytest.approx(2 * p, abs=1e-12)
    assert local.expected_gradient_weight(1.0, group, scale_rewards="none") == group - 1
    assert local.expected_gradient_weight(0.0, group, scale_rewards="none") == 0
    with pytest.raises(ValueError, match="group or none"):
        local.expected_gradient_weight(p, group, scale_rewards="batch")


def test_dr_grpo_nostd_weight_matches_monte_carlo_without_std():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(2026)
    groups = 200_000
    for p, group, hand in ((0.3, 8, 2.1), (0.7, 4, 2.1)):
        pi = torch.tensor([p, (1 - p) / 2, (1 - p) / 2], dtype=torch.float64)
        samples = torch.multinomial(pi, groups * group, replacement=True, generator=generator)
        rewards = (samples == 0).to(torch.float64).view(groups, group)
        advantages = rewards - rewards.mean(dim=1, keepdim=True)  # scale_rewards="none"
        scores = torch.nn.functional.one_hot(samples, 3).to(torch.float64) - pi
        per_group = (advantages.view(-1, 1) * scores).view(groups, group, 3).sum(dim=1)
        estimate = per_group.mean(dim=0)
        standard_error = per_group.std(dim=0) / groups**0.5
        expected = hand * (torch.tensor([1.0, 0, 0], dtype=torch.float64) - pi)
        assert ((estimate - expected).abs() <= 4 * standard_error + 1e-9).all(), (p, estimate)


def test_egsft_band_is_the_screen_acceptance_set():
    # u_8(p) = 1 - p^8 - (1-p)^8: 0.92^8 = 0.5132 (out), 0.91^8 = 0.4703 (in), symmetric.
    for p, inside in ((0.08, False), (0.09, True), (0.5, True), (0.91, True), (0.92, False)):
        assert local.egsft_in_band(p, 8) is inside
    # G = 4 keeps the pre-registered [0.1594, 0.8406] band.
    for p, inside in ((0.159, False), (0.16, True), (0.84, True), (0.841, False)):
        assert local.egsft_in_band(p, 4) is inside


def test_egsft_nostd_band_scales_loss_by_2p_and_zeroes_outside(tmp_path):
    torch = pytest.importorskip("torch")
    theta = torch.nn.Parameter(torch.tensor([0.5, -1.0]))

    class Base:
        state = SimpleNamespace(global_step=0)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return (theta * torch.tensor([2.0, 3.0])).sum()

    def trainer(band_group):
        return local._trainer_with_egsft(
            Base,
            eos_token_ids=[_EOS],
            suppress_tokens=[_IMAGE],
            temperature=1.0,
            group_size=8,
            journal=tmp_path / f"egsft-{band_group}.jsonl",
            scale_rewards="none",
            band_group=band_group,
        )()

    policy = _BigramPolicy(_PLANTED, 8)
    base_loss = 0.5 * 2 - 1.0 * 3
    # Marker 5: p = 0.5 * 0.99 = 0.495, inside B_4 and B_8; weight 2p = 0.99.
    four = trainer(4)
    four._vision_rl_prompt_id = "mixed"
    loss = four.compute_loss(policy, _gold_batch(5))
    assert float(loss.detach()) == pytest.approx(0.99 * base_loss, rel=1e-6)
    # Marker 4: p = 0.1 * 0.99 = 0.099; u_4 = 1 - 0.901^4 - 0.099^4 = 0.341 < 0.5: out of B_4.
    four._vision_rl_prompt_id = "hard"
    loss = four.compute_loss(policy, _gold_batch(4))
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert theta.grad.tolist() == [0.0, 0.0]  # zero loss, and the optimizer step is skipped
    assert four._vision_rl_skip_step is True
    rows = [json.loads(x) for x in (tmp_path / "egsft-4.jsonl").read_text().splitlines()]
    assert [row["in_band"] for row in rows] == [True, False]
    assert [row["optimizer_step"] for row in rows] == [True, False]
    assert rows[1]["weight"] == 0.0 and rows[1]["p"] == pytest.approx(0.099, rel=1e-6)
    # u_8(0.099) = 1 - 0.901^8 - 0.099^8 = 0.566: a larger G widens the band.
    eight = trainer(8)
    eight._vision_rl_prompt_id = "hard"
    loss = eight.compute_loss(policy, _gold_batch(4))
    assert float(loss.detach()) == pytest.approx(2 * 0.099 * base_loss, rel=1e-6)


def test_egsft_band_skips_the_optimizer_step_and_leaves_adamw_state_untouched(tmp_path):
    torch = pytest.importorskip("torch")
    theta = torch.nn.Parameter(torch.tensor([0.5, -1.0]))

    class Base:
        state = SimpleNamespace(global_step=0)
        optimizer = None

        def create_optimizer(self):
            self.optimizer = torch.optim.AdamW([theta], lr=0.1, weight_decay=0.0)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return (theta * torch.tensor([2.0, 3.0])).sum()

    def build(band_group):
        trainer = local._trainer_with_egsft(
            Base, eos_token_ids=[_EOS], suppress_tokens=[_IMAGE], temperature=1.0,
            group_size=8, journal=tmp_path / f"guard-{band_group}.jsonl",
            scale_rewards="none", band_group=band_group)()
        optimizer = trainer.create_optimizer()
        # torch's LR scheduler rebinds optimizer.step.__func__; the guard must survive it.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return trainer, optimizer, scheduler

    def update(trainer, optimizer, scheduler, marker, identifier):
        trainer._vision_rl_prompt_id = identifier
        optimizer.zero_grad()
        trainer.compute_loss(_BigramPolicy(_PLANTED, 8), _gold_batch(marker)).backward()
        optimizer.step()
        scheduler.step()

    trainer, optimizer, scheduler = build(4)
    # Marker 5: p = .495, in B_4 -> a real AdamW step moves theta and creates its state.
    update(trainer, optimizer, scheduler, 5, "mixed")
    after_first = theta.detach().clone()
    assert not torch.equal(after_first, torch.tensor([0.5, -1.0]))
    state = {k: v.clone() for k, v in optimizer.state[theta].items()}
    assert int(state["step"]) == 1
    # Marker 4: p = .099, outside B_4: AdamW would still move theta on its first moment
    # (m = .9 m_prev, nonzero) if it stepped; the guard must leave theta and state as they are.
    update(trainer, optimizer, scheduler, 4, "hard")
    assert torch.equal(theta.detach(), after_first)
    assert all(torch.equal(optimizer.state[theta][k], v) for k, v in state.items())
    # Back in band: the step happens again and AdamW's count resumes at 2.
    update(trainer, optimizer, scheduler, 5, "mixed")
    assert not torch.equal(theta.detach(), after_first)
    assert int(optimizer.state[theta]["step"]) == 2
    # Without a band nothing is wrapped: the legacy EG-SFT optimizer is the plain one.
    plain, plain_optimizer, _ = build(None)
    assert not getattr(plain_optimizer.step, "_vision_rl_band_guard", False)


def test_grpo_loss_type_and_scale_rewards_reach_trainer_kwargs(tmp_path):
    profile = local.Profile()
    kwargs = dict(device="cpu", max_steps=4, learning_rate=1e-5, seed=42, save_steps=1)
    legacy = local.trainer_kwargs("grpo", tmp_path, profile, **kwargs)
    assert local.trainer_kwargs("grpo", tmp_path, profile, loss_type="grpo", **kwargs) == legacy
    dr = local.trainer_kwargs(
        "grpo", tmp_path, profile, beta=0, loss_type="dr_grpo", scale_rewards="none", **kwargs
    )
    assert (dr["loss_type"], dr["scale_rewards"], dr["beta"]) == ("dr_grpo", "none", 0)
    assert {k: v for k, v in dr.items() if k not in {"loss_type", "scale_rewards", "beta"}} == {
        k: v for k, v in legacy.items() if k not in {"loss_type", "scale_rewards", "beta"}
    }
    with pytest.raises(ValueError, match="only to GRPO"):
        local.trainer_kwargs("sft", tmp_path, profile, loss_type="grpo", **kwargs)
    with pytest.raises(ValueError, match="grpo or dr_grpo"):
        local.trainer_kwargs("grpo", tmp_path, profile, loss_type="dapo", **kwargs)


def test_released_grpo_config_accepts_dr_grpo_without_std(tmp_path):
    trl = released_trl()
    config = trl.GRPOConfig(
        **local.trainer_kwargs(
            "grpo",
            tmp_path,
            local.Profile(),
            device="cpu",
            max_steps=2,
            learning_rate=1e-5,
            seed=42,
            save_steps=1,
            beta=0.0,
            loss_type="dr_grpo",
            scale_rewards="none",
        )
    )
    assert (config.loss_type, config.scale_rewards, config.beta) == ("dr_grpo", "none", 0.0)


def test_beta_zero_refuses_without_explicit_loss_type(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must refuse before model"))
    with pytest.raises(ValueError, match="explicit loss_type"):
        local.train_local(
            diagnostic / "train.jsonl",
            None,
            diagnostic / "data_spec.json",
            tmp_path / "out",
            model_directory=tmp_path / "model",
            max_steps=4,
            train_only=True,
            beta=0.0,
        )
    assert not (tmp_path / "out").exists()


def test_signal_limit_options_enter_settings_only_when_non_default(
    diagnostic, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    plain = _settings_after_model_refusal(diagnostic, tmp_path, monkeypatch, "plain")
    explicit = _settings_after_model_refusal(
        diagnostic,
        tmp_path,
        monkeypatch,
        "explicit-defaults",
        loss_type="grpo",
        scale_rewards="group",
        reward="exact",
    )
    assert explicit["settings"] == plain["settings"]
    beta_zero = _settings_after_model_refusal(
        diagnostic, tmp_path, monkeypatch, "beta-zero-legacy", beta=0.0, loss_type="grpo"
    )
    assert beta_zero["settings"] == dict(plain["settings"], beta=0.0)
    assert beta_zero["regularization_profile"] == "beta-zero-ablation"
    dr = _settings_after_model_refusal(
        diagnostic,
        tmp_path,
        monkeypatch,
        "dr",
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards="none",
    )
    assert dr["settings"] == dict(
        plain["settings"], beta=0.0, loss_type="dr_grpo", scale_rewards="none"
    )
    egsft = _settings_after_model_refusal(
        diagnostic, tmp_path, monkeypatch, "egsft-plain", mode="egsft"
    )
    limit = _settings_after_model_refusal(
        diagnostic,
        tmp_path,
        monkeypatch,
        "egsft-limit",
        mode="egsft",
        egsft_scale="none",
        egsft_band_group=8,
    )
    assert set(egsft["settings"]["egsft"]) == {"group_size", "advantage_eps", "weight"}
    assert limit["settings"]["egsft"]["scale_rewards"] == "none"
    assert limit["settings"]["egsft"]["band"]["group_size"] == 8
    assert limit["settings"]["egsft"]["band"]["threshold"] == 0.5


def _write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def _confusion_doc(confusers):
    # scripts/build_label_trie.py's map format; only "confusers" is read by the trainer.
    return {"schema_version": 1, "map": {
        gold: {"confusers": options, "family": [], "observed": options}
        for gold, options in confusers.items()}}


def _taxonomy_doc(variants):
    # scripts/build_label_trie.py's taxonomy format; only "variants" is read by the trainer.
    return {"schema_version": 1, "variants": variants, "families": {}, "manufacturers": {}}


def _trie_doc(paths, eos_ids):
    # scripts/build_label_trie.py's trie format: EOS-free paths per label plus the EOS IDs,
    # with its content hash written out from the builder's stated rule (compact, sorted,
    # ASCII JSON without sha256/provenance).
    document = {"schema_version": 1, "eos_ids": eos_ids, "labels": list(paths),
                "entries": {label: {"paths": options} for label, options in paths.items()}}
    body = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {**document, "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "provenance": {"note": "fixture"}}


def test_label_loss_modes_bind_their_frozen_resources(diagnostic, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    confusion = _write_json(tmp_path / "confusion.json", _confusion_doc({"blue": ["red"]}))
    taxonomy = _write_json(
        tmp_path / "taxonomy.json",
        _taxonomy_doc(
            {
                "blue": {"family": "F", "manufacturer": "M"},
                "red": {"family": "F", "manufacturer": "M"},
            }
        ),
    )
    trie = _write_json(tmp_path / "trie.json", _trie_doc({"blue": [[5]], "red": [[6]]}, [1]))
    expected = {
        "hn_sft": ("confusion_map", confusion, "completion-only-SFT-plus-confuser-unlikelihood"),
        "hls_sft": ("taxonomy", taxonomy, "hierarchical-label-smoothed-completion-only-SFT"),
        "cce_sft": ("label_trie", trie, "label-trie-constrained-completion-only-SFT"),
    }
    for mode, (name, path, profile_name) in expected.items():
        record = _settings_after_model_refusal(
            diagnostic, tmp_path, monkeypatch, mode, mode=mode, **{name: path}
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = {"confusion_map": "confusion_map_sha256", "taxonomy": "taxonomy_sha256"}.get(
            name, "label_trie_sha256"
        )
        assert record["settings"][key] == digest
        assert record["regularization_profile"] == profile_name
        assert record["algorithm"] == f"TRL-{mode.upper()}"
        assert record["settings"]["beta"] is None
    hn_grpo = _settings_after_model_refusal(
        diagnostic, tmp_path, monkeypatch, "hn-grpo", reward="exact_hn", confusion_map=confusion
    )
    assert hn_grpo["settings"]["reward"] == {"name": "exact_hn", "confuser_reward": -0.5}


def test_signal_limit_options_refuse_before_model_access(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must refuse before model"))
    output = tmp_path / "out"
    base = dict(
        train=diagnostic / "train.jsonl",
        validation=None,
        data_spec=diagnostic / "data_spec.json",
        output=output,
        model_directory=tmp_path / "model",
        max_steps=4,
        train_only=True,
    )
    self_map = _write_json(tmp_path / "self.json", _confusion_doc({"blue": ["blue"]}))
    for kwargs, match in (
        ({"mode": "sft", "loss_type": "dr_grpo"}, "only to GRPO"),
        ({"mode": "sft", "scale_rewards": "none"}, "only to GRPO"),
        ({"loss_type": "dapo"}, "grpo or dr_grpo"),
        ({"reward": "exact_hn"}, "confusion_map is required"),
        ({"mode": "sft", "reward": "exact_hn", "confusion_map": self_map}, "exact_hn for GRPO"),
        ({"mode": "hn_sft"}, "confusion_map is required"),
        ({"mode": "hls_sft"}, "taxonomy is required"),
        ({"mode": "cce_sft"}, "label_trie is required"),
        ({"mode": "sft", "label_trie": tmp_path / "t.json"}, "label_trie is required"),
        ({"mode": "sft", "egsft_scale": "none"}, "only to EG-SFT"),
        ({"mode": "egsft", "egsft_band_group": 5}, "4, 8 or 16"),
        ({"mode": "hn_sft", "confusion_map": self_map}, "own confuser"),
    ):
        with pytest.raises(ValueError, match=match):
            local.train_local(**dict(base, **kwargs))
    assert not output.exists()


def test_exact_hn_reward_is_separately_named_and_penalises_only_map_confusers(tmp_path):
    labels = ["blue", "red", "green"]
    path = tmp_path / "hn.jsonl"
    reward = local.make_reward(
        labels, 4, [99], path, reward="exact_hn", confusers={"blue": ["red"]}
    )
    assert reward.__name__ == "exact_hn_reward"
    assert local.make_reward(labels, 4, [99], tmp_path / "x").__name__ == "exact_label_accuracy"
    # gold, map confuser, non-map wrong label, unterminated confuser (invalid -> 0).
    rewards = reward(
        ["blue", "red", "green", "red"], ["blue"] * 4, ["i"] * 4, [[99], [99], [99], [7]]
    )
    assert rewards == [1.0, -0.5, 0.0, 0.0]
    row = json.loads(path.read_text())
    assert row["rewards"] == [1.0, -0.5, 0.0, 0.0]
    assert row["exact"] == [1.0, 0.0, 0.0, 0.0]
    assert row["confuser"] == [False, True, False, False]
    assert reward.stats["confuser_completions"] == 1 and reward.stats["mixed_groups"] == 1
    # All confusers: mixed by reward = no; no exact gold, so it counts as an all-zero group.
    assert reward(["red"] * 4, ["blue"] * 4, ["j"] * 4, [[99]] * 4) == [-0.5] * 4
    assert reward.stats["mixed_groups"] == 1 and reward.stats["all_zero_groups"] == 1
    assert reward.stats["all_correct_groups"] == 0
    with pytest.raises(ValueError, match="frozen confusion map"):
        local.make_reward(labels, 4, [99], path, reward="exact_hn")
    legacy = local.make_reward(labels, 4, [99], tmp_path / "legacy.jsonl")
    legacy(["blue", "red", "green", "red"], ["blue"] * 4, ["i"] * 4, [[99], [99], [99], [7]])
    assert "confuser_completions" not in legacy.stats
    assert set(json.loads((tmp_path / "legacy.jsonl").read_text())) == {
        "id", "label", "rewards", "mixed", "recognition_mixed", "canonical_predictions",
        "completions", "completion_ids", "completion_tokens", "terminated",
        "complete_think_block",
    }


# Label fixture: 0 pad, 1 EOS, 3 prompt start, 5 marker, 6 gold "g", 4 confuser "a",
# 7 confuser "b" (and the post-EOS template tail), 8 second token of label "gx".
_LABELS = {
    5: {6: 0.5, 4: 0.3, 7: 0.2},
    6: {1: 0.9, 8: 0.05, 2: 0.05},
    4: {1: 1.0},
    7: {1: 1.0},
    8: {1: 1.0},
    1: {7: 1.0},
}
_LABEL_IDS = {"g": [6], "a": [4], "b": [7], "gx": [6, 8]}


def _label_batch():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[3, 5, 6, 1, 7]])
    labels = torch.tensor([[-100, -100, 6, 1, 7]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _label_trainer(tmp_path, mode, base_loss=1.0, label_token_ids=_LABEL_IDS, **resources):
    torch = pytest.importorskip("torch")
    anchor = torch.nn.Parameter(torch.tensor(base_loss))

    class Base:
        state = SimpleNamespace(global_step=2)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            assert "labels" in inputs
            return (anchor * 1.0, "outputs") if return_outputs else anchor * 1.0

    trainer = local._trainer_with_label_loss(
        Base,
        mode=mode,
        journal=tmp_path / f"{mode}.jsonl",
        labels_by_id={"row": "g"},
        eos_token_ids=[1],
        label_token_ids=label_token_ids,
        **resources,
    )()
    trainer._vision_rl_prompt_id = "row"
    return trainer


def test_hn_sft_picks_top_live_confuser_and_its_term_lowers_pi_c(tmp_path):
    torch = pytest.importorskip("torch")
    policy = _BigramPolicy(_LABELS, 9)
    trainer = _label_trainer(tmp_path, "hn_sft", confusers={"g": ["b", "a"]})
    loss = trainer.compute_loss(policy, _label_batch())
    # "a" (0.3) beats "b" (0.2) although listed second; term = -log(1 - 0.3).
    assert float(loss.detach()) == pytest.approx(1.0 + 0.5 * -math.log(0.7), rel=1e-6)
    row = json.loads((tmp_path / "hn_sft.jsonl").read_text())
    assert row["confuser"] == "a" and row["global_step"] == 3
    assert row["confuser_probability"] == pytest.approx(0.3, rel=1e-6)
    loss.backward()
    # d/dz[5,4] of 0.5 * -log(1 - pi_a) = 0.5 * pi_a = 0.15; d/dz[5,6] = 0.5 * -pi_a pi_g / 0.7.
    assert float(policy.parameter.grad[5, 4]) == pytest.approx(0.15, rel=1e-5)
    assert float(policy.parameter.grad[5, 6]) == pytest.approx(-0.5 * 0.3 * 0.5 / 0.7, rel=1e-5)
    assert float(policy.parameter.grad[4, 1]) == pytest.approx(0.0, abs=1e-9)
    prompt = {"input_ids": torch.tensor([[3, 5]]), "attention_mask": torch.ones(1, 2)}

    def pi_a():
        return math.exp(
            local._completion_logprob(policy, prompt, [4, 1], temperature=1.0, suppress_tokens=())
        )

    before = pi_a()
    with torch.no_grad():
        policy.parameter -= 1.0 * policy.parameter.grad
    assert pi_a() < before


def test_hn_sft_without_confuser_is_plain_ce_and_refuses_bad_resources(tmp_path):
    policy = _BigramPolicy(_LABELS, 9)
    trainer = _label_trainer(tmp_path, "hn_sft", confusers={"a": ["g"]})
    assert float(trainer.compute_loss(policy, _label_batch()).detach()) == 1.0
    assert json.loads((tmp_path / "hn_sft.jsonl").read_text())["confuser"] is None
    trainer = _label_trainer(tmp_path, "hn_sft", confusers={"g": ["g"]})
    with pytest.raises(local.LocalVLMError, match="equals the gold"):
        trainer.compute_loss(policy, _label_batch())
    with pytest.raises(ValueError, match="own confuser"):
        local.load_confusion_map(
            _write_json(tmp_path / "m.json", _confusion_doc({"g": ["g"]})),
            ["g", "a"],
        )
    mismatched = _label_trainer(
        tmp_path, "hn_sft", label_token_ids=dict(_LABEL_IDS, g=[7]), confusers={"g": ["a"]}
    )
    with pytest.raises(local.LocalVLMError, match="differs from collated gold"):
        mismatched.compute_loss(policy, _label_batch())


def test_hls_targets_and_loss_follow_the_taxonomy(tmp_path):
    taxonomy = {
        "A1": {"family": "F1", "manufacturer": "M"},
        "A2": {"family": "F1", "manufacturer": "M"},
        "B1": {"family": "F2", "manufacturer": "M"},
        "C1": {"family": "F3", "manufacturer": "N"},
    }
    assert local.hls_targets("A1", taxonomy) == pytest.approx({"A1": 0.9, "A2": 0.05, "B1": 0.05})
    assert local.hls_targets("B1", taxonomy) == pytest.approx(
        {"B1": 0.95, "A1": 0.025, "A2": 0.025}
    )
    assert local.hls_targets("C1", taxonomy) == {"C1": 1.0}
    with pytest.raises(ValueError, match="every declared label"):
        local.load_taxonomy(
            _write_json(tmp_path / "t.json", _taxonomy_doc(taxonomy)),
            ["A1", "A2", "B1", "C1", "D1"],
        )
    policy = _BigramPolicy(_LABELS, 9)
    labels = {
        "g": {"family": "F", "manufacturer": "M"},
        "a": {"family": "F", "manufacturer": "M"},
        "b": {"family": "X", "manufacturer": "N"},
        "gx": {"family": "Y", "manufacturer": "N"},
    }
    trainer = _label_trainer(tmp_path, "hls_sft", base_loss=2.0, taxonomy=labels)
    loss = trainer.compute_loss(policy, _label_batch())
    # q_g = 0.95 (no manufacturer tier) on the gold's own label sequence [6, EOS], mean NLL
    # -(ln .5 + ln .9)/2, NOT TRL's CE (2.0, which also averages the post-EOS tail);
    # q_a = 0.05 on mean NLL of [4, EOS] = -(ln .3 + 0)/2.
    gold = -(math.log(0.5) + math.log(0.9)) / 2
    assert float(loss.detach()) == pytest.approx(0.95 * gold + 0.05 * -math.log(0.3) / 2,
                                                 rel=1e-6)


def test_trie_masked_logprob_and_cce_loss(tmp_path):
    torch = pytest.importorskip("torch")
    policy = _BigramPolicy(_LABELS, 9)
    sequences = {"g": [[6, 1]], "a": [[4, 1]], "gx": [[6, 8, 1]]}
    trie = local.label_trie(sequences)
    assert local.trie_allowed(trie, []) == [4, 6]
    assert local.trie_allowed(trie, [6]) == [1, 8]
    assert local.trie_allowed(trie, [6, 1]) is None and local.trie_allowed(trie, [7]) is None
    mask = local.trie_position_mask(trie, [6, 1, 7])
    assert mask == [[4, 6], [1, 8], None]
    prompt = {"input_ids": torch.tensor([[3, 5]]), "attention_mask": torch.ones(1, 2)}
    score = dict(temperature=1.0, suppress_tokens=())
    unmasked = local._completion_logprob(policy, prompt, [6, 1, 7], **score)
    assert unmasked == pytest.approx(math.log(0.5) + math.log(0.9), rel=1e-6)
    masked = local._completion_logprob(policy, prompt, [6, 1, 7], allowed_tokens=mask, **score)
    # Constrained: P(6 | {4, 6}) = .5/.8, P(EOS | {1, 8}) = .9/.95, tail unrestricted (1.0).
    assert masked == pytest.approx(math.log(0.5 / 0.8) + math.log(0.9 / 0.95), rel=1e-6)
    # Forced fire: a planted off-trie token scores -inf under the mask.
    assert local._completion_logprob(
        policy, prompt, [7, 1], allowed_tokens=[[4, 6], None], **score
    ) == -math.inf
    trainer = _label_trainer(tmp_path, "cce_sft", trie=trie)
    loss = trainer.compute_loss(policy, _label_batch())
    hand = -(math.log(0.5 / 0.8) + math.log(0.9 / 0.95) + 0.0) / 3
    assert float(loss.detach()) == pytest.approx(hand, rel=1e-6)
    assert json.loads((tmp_path / "cce_sft.jsonl").read_text())["constrained_positions"] == 2
    loss.backward()
    assert float(policy.parameter.grad[5, 7]) == 0.0  # masked logit gets no gradient
    # With no restriction the same path is TRL's completion-only CE (mean over 3 tokens).
    free = local._completion_token_logps(
        policy, prompt, [6, 1, 7], allowed_tokens=[None] * 3, **score
    )
    assert float(local.constrained_nll(free[0], 3).detach()) == pytest.approx(
        -(math.log(0.5) + math.log(0.9)) / 3, rel=1e-6
    )
    off = _label_trainer(tmp_path, "cce_sft", trie=local.label_trie({"a": [[4, 1]]}))
    with pytest.raises(local.LocalVLMError, match="leaves the label trie"):
        off.compute_loss(policy, _label_batch())
    with pytest.raises(ValueError, match="proper prefix"):
        local.label_trie({"g": [[6, 1]], "h": [[6, 1, 8, 1]]})


def test_label_trie_logits_processor_masks_off_trie_tokens_for_constrained_greedy():
    torch = pytest.importorskip("torch")
    trie = local.label_trie({"g": [[6, 1]], "a": [[4, 1]], "gx": [[6, 8, 1]]})
    processor = local.label_trie_logits_processor(trie, 2)
    assert processor.scope.startswith("assay-only")
    input_ids = torch.tensor([[3, 5, 9], [3, 5, 6], [3, 5, 4]])
    scores = torch.zeros(3, 10)
    scores[:, 7] = 5.0  # planted off-trie favourite
    out = processor(input_ids[:, :2], scores.clone())
    for row in range(3):
        assert torch.isfinite(out[row]).nonzero().flatten().tolist() == [4, 6]
    out = processor(input_ids, scores.clone())
    assert out[0].tolist() == scores[0].tolist()  # off-trie prefix: left unrestricted
    assert torch.isfinite(out[1]).nonzero().flatten().tolist() == [1, 8]
    assert torch.isfinite(out[2]).nonzero().flatten().tolist() == [1]
    transformers = pytest.importorskip("transformers")
    chained = transformers.LogitsProcessorList([processor])(input_ids[:, :2], scores.clone())
    assert int(chained[0].argmax()) == 4  # greedy can no longer pick the planted token 7


def test_cuda_and_float16_options_validate_without_hardware(monkeypatch):
    local.Profile(dtype="float16").validate()
    with pytest.raises(ValueError, match="float16"):
        local.Profile(dtype="int8").validate()
    monkeypatch.setattr(local.metadata, "version", lambda _: "wrong")
    with pytest.raises(local.LocalVLMError, match="Expected torch"):
        local.preflight("cuda")  # cuda passes the device gate, then the stack check runs
    with pytest.raises(local.LocalVLMError, match="mps, cpu or cuda"):
        local.preflight("tpu")
    card = {"name": "RTX 2080 Ti", "capability": [7, 5], "native_bfloat16": False}
    with pytest.raises(local.LocalVLMError, match="no native bfloat16"):
        local._refuse_unsupported_dtype({"cuda_device": card}, local.Profile(dtype="bfloat16"))
    local._refuse_unsupported_dtype({"cuda_device": card}, local.Profile(dtype="float16"))
    local._refuse_unsupported_dtype({}, local.Profile(dtype="bfloat16"))
    # float16 is inference-only: no GradScaler (fp16=False), so training refuses it anywhere.
    for environment in ({"cuda_device": card}, {}):
        with pytest.raises(local.LocalVLMError, match="float16 training"):
            local._refuse_unsupported_dtype(
                environment, local.Profile(dtype="float16"), training=True)
    with pytest.raises(local.LocalVLMError, match="use float32"):
        local._refuse_unsupported_dtype(
            {"cuda_device": card}, local.Profile(dtype="bfloat16"), training=True)
    local._refuse_unsupported_dtype({}, local.Profile(), training=True)  # float32 trains


def test_cli_passes_signal_limit_options_and_inference_takes_only_the_trie(monkeypatch, capsys):
    seen = {}

    def train(**kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(local, "train_local", train)
    args = ["--train", "t.jsonl", "--data-spec", "s.json", "--output", "o", "--train-only"]
    argv = args + ["--beta", "0", "--loss-type", "dr_grpo", "--scale-rewards", "none",
                   "--reward", "exact_hn", "--confusion-map", "m.json", "--device", "cuda",
                   "--dtype", "float16"]
    assert local.main(argv) == 0
    assert (seen["loss_type"], seen["scale_rewards"], seen["reward"]) == (
        "dr_grpo", "none", "exact_hn"
    )
    assert seen["confusion_map"] == Path("m.json") and seen["device"] == "cuda"
    assert seen["profile"].dtype == "float16"
    assert local.main(args + ["--mode", "egsft", "--egsft-scale", "none",
                              "--egsft-band-group", "8"]) == 0
    assert (seen["egsft_scale"], seen["egsft_band_group"]) == ("none", 8)
    assert local.main(args + ["--mode", "cce_sft", "--label-trie", "trie.json"]) == 0
    assert seen["mode"] == "cce_sft" and seen["label_trie"] == Path("trie.json")
    capsys.readouterr()

    def predict(manifest, **kwargs):
        assert kwargs["label_trie"] == Path("trie.json")
        return {"optimizer_steps": 0}

    monkeypatch.setattr(local, "predict_train_subset", predict)
    inference = ["--predict-train", "train.jsonl", "--data-spec", "spec.json", "--output",
                 "out", "--model-directory", "model"]
    assert local.main(inference + ["--label-trie", "trie.json"]) == 0
    for extra in (["--loss-type", "dr_grpo"], ["--scale-rewards", "none"],
                  ["--egsft-band-group", "8"], ["--taxonomy", "t.json"]):
        with pytest.raises(SystemExit):
            local.main(inference + extra)
    monkeypatch.setattr(local, "sample_train_groups", lambda *a, **k: pytest.fail("refuse"))
    sample = ["--sample-train", "train.jsonl"] + inference[2:]
    with pytest.raises(SystemExit):
        local.main(sample + ["--label-trie", "trie.json"])


def test_constrained_greedy_is_refused_for_sampling(diagnostic, tmp_path):
    with pytest.raises(ValueError, match="greedy-only"):
        local.predict_train_subset(
            diagnostic / "train.jsonl",
            diagnostic / "data_spec.json",
            tmp_path / "out",
            model_directory=tmp_path / "model",
            sample_groups=True,
            label_trie=tmp_path / "trie.json",
        )
