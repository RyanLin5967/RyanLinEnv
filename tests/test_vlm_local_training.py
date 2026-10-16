"""Offline data/reward/control contracts and optional released-TRL CPU checks."""

import hashlib
import json
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
