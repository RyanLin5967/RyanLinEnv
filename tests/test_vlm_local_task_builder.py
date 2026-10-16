"""Numeric submission and train/grader isolation checks; no model is loaded."""

import copy
import hashlib
import json
import os
import shutil
import tomllib
from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image

from vision_rl import vlm_local_task_builder as builder

safe = pytest.importorskip("safetensors.numpy")


@pytest.fixture(scope="module")
def numeric_bytes():
    return safe.save(
        {name: np.zeros(shape, np.float32) for name, shape in builder.adapter_shapes().items()}
    )


@pytest.fixture
def fixture(tmp_path, numeric_bytes):
    tmp_path = tmp_path.resolve()  # macOS /var is an OS alias, not submitted input.
    spec = {
        "schema_version": 1,
        "benchmark": "integration-fixture",
        "labels": ["blue", "red"],
        "question": "Which color?",
        "split_policy": "disjoint development fixture",
        "preprocessing": "RGB",
        "sources": [{"url": "generated://fixture", "sha256": "0" * 64}],
    }
    spec_path = tmp_path / "data_spec.json"
    spec_path.write_text(json.dumps(spec))
    for split, colors in (("train", ("red", "blue")), ("validation", ("green", "yellow"))):
        rows = []
        for i, color in enumerate(colors):
            filename = f"{split}-{i}.png"
            Image.new("RGB", (28, 28), color).save(tmp_path / filename)
            rows.append(
                {
                    "id": f"{split}-{i}",
                    "image": filename,
                    "split": split,
                    "label": spec["labels"][i],
                }
            )
        (tmp_path / (split + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    model = tmp_path / "model"
    model.mkdir()
    (model / "model-source.json").write_text(
        json.dumps(
            {
                "model_repository": builder.trainer.DEFAULT_MODEL,
                "revision": builder.trainer.MODEL_REVISIONS[builder.trainer.DEFAULT_MODEL],
            }
        )
    )
    (model / ".pinned-download.json").write_text("{}")
    initial = tmp_path / "warmup/final_adapter"
    initial.mkdir(parents=True)
    (initial / "adapter_model.safetensors").write_bytes(numeric_bytes)
    (initial / "adapter_config.json").write_text(
        json.dumps({**builder._fixed_lora_config(), "peft_type": "LORA"})
    )
    parent = {
        "status": "completed",
        "final_adapter_sha256": hashlib.sha256(numeric_bytes).hexdigest(),
        "control_identity": {
            "model": builder.trainer.DEFAULT_MODEL,
            "model_revision": builder.trainer.MODEL_REVISIONS[builder.trainer.DEFAULT_MODEL],
            "processor_model": builder.trainer.DEFAULT_MODEL,
            "processor_revision": builder.trainer.MODEL_REVISIONS[builder.trainer.DEFAULT_MODEL],
            "profile": asdict(builder.trainer.Profile()),
            "data_spec_sha256": builder._digest(spec_path),
            "train": {},
            "validation": None,
        },
    }
    (initial.parent / "local_training.json").write_text(json.dumps(parent))
    return dict(
        train=tmp_path / "train.jsonl",
        verification=tmp_path / "validation.jsonl",
        data_spec=spec_path,
        initial_adapter=initial,
        model_directory=model,
        output=tmp_path / "task",
    )


def test_numeric_adapter_exact_contract(tmp_path, numeric_bytes):
    path = tmp_path.resolve() / "model.safetensors"
    path.write_bytes(numeric_bytes)
    values, digest = builder.read_adapter(path)
    assert len(values) == 112
    assert sum(value.size for value in values.values()) == 1_089_536
    assert digest == hashlib.sha256(numeric_bytes).hexdigest()


@pytest.mark.parametrize("mutation", ["extra", "missing", "shape", "dtype", "nan", "inf"])
def test_rejects_numeric_contract_tampering(tmp_path, numeric_bytes, mutation):
    values = safe.load(numeric_bytes)
    name = next(iter(values))
    if mutation == "extra":
        values["visual.untrusted.weight"] = np.zeros(1, np.float32)
    elif mutation == "missing":
        values.pop(name)
    elif mutation == "shape":
        values[name] = values[name].reshape(-1)
    elif mutation == "dtype":
        values[name] = values[name].astype(np.float16)
    else:
        values[name].flat[0] = float(mutation)
    path = tmp_path.resolve() / "model.safetensors"
    safe.save_file(values, str(path))
    with pytest.raises(ValueError):
        builder.read_adapter(path)


@pytest.mark.parametrize("kind", ["file_link", "directory_link", "fifo", "empty", "oversize"])
def test_submission_boundary_rejects_filesystem_tricks(tmp_path, numeric_bytes, kind):
    tmp_path = tmp_path.resolve()
    source = tmp_path / "source"
    source.mkdir()
    path = source / "model.safetensors"
    path.write_bytes(numeric_bytes)
    if kind == "file_link":
        alias = tmp_path / "link.safetensors"
        alias.symlink_to(path)
        path = alias
    elif kind == "directory_link":
        alias = tmp_path / "link"
        alias.symlink_to(source, target_is_directory=True)
        path = alias / path.name
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "empty":
        path.write_bytes(b"")
    else:
        with path.open("wb") as stream:
            stream.truncate(builder.ADAPTER_LIMIT + 1)
    with pytest.raises((ValueError, OSError)):
        builder.read_adapter(path)


def test_task_isolates_verifier_and_mounts_base_readonly(fixture, monkeypatch):
    monkeypatch.setattr(
        builder.trainer, "preflight", lambda *_: pytest.fail("No runtime imports while packaging")
    )
    monkeypatch.setattr(
        builder.trainer,
        "_verified_model_directory",
        lambda *_: pytest.fail("No base hashing while packaging"),
    )
    result = builder.make_task(**fixture)
    task = fixture["output"]
    environment = task / "environment"
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["environment"]["memory_mb"] == 24576
    assert config["environment"]["network_mode"] == "no-network"
    main = json.loads((environment / "docker-compose.yaml").read_text())["services"]["main"]
    assert main["network_mode"] == "none"
    assert main["volumes"] == [
        {
            "type": "bind",
            "source": str(fixture["model_directory"]),
            "target": "/opt/pinned-model",
            "read_only": True,
            "bind": {"create_host_path": False},
        }
    ]
    assert result["counts"] == {"train": 2, "verification": 2}
    assert result["runtime_executed"] is False
    assert not (environment / "tests").exists()
    assert not (environment / "data/verification.jsonl").exists()
    assert "tests" not in (environment / "Dockerfile").read_text()
    assert result["agent"].endswith(":DetectionTrainingAgent")
    assert json.loads((task / "tests/contract.json").read_text()) == result["contract"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_steps", 5),
        ("max_steps", True),
        ("control_step_cap", 33),
        ("beta", 0),
        ("learning_rate", float("nan")),
        ("timeout_sec", 86401),
    ],
)
def test_task_rejects_unbounded_work(fixture, option, value):
    with pytest.raises(ValueError):
        builder.make_task(**fixture, **{option: value})
    assert not fixture["output"].exists()


def test_task_rejects_pixel_leakage(fixture):
    image = fixture["train"].parent
    shutil.copyfile(image / "train-0.png", image / "validation-0.png")
    with pytest.raises(ValueError, match="leakage"):
        builder.make_task(**fixture)


def test_task_never_accepts_sealed_test_manifest(fixture):
    fixture["verification"].write_text(
        fixture["verification"].read_text().replace('"validation"', '"test"')
    )
    with pytest.raises(ValueError, match="expected validation"):
        builder.make_task(**fixture)


def test_snapshot_revision_and_export_identity_are_fixed(fixture):
    source = fixture["model_directory"] / "model-source.json"
    value = json.loads(source.read_text())
    value["revision"] = "main"
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="pinned"):
        builder.make_task(**fixture)


@pytest.mark.parametrize("tamper", ["source", "runtime", "profile", "unknown"])
def test_private_contract_fails_closed(fixture, tamper):
    result = builder.make_task(**fixture)
    contract = copy.deepcopy(result["contract"])
    if tamper == "source":
        contract["source_sha256"]["vlm_local_training.py"] = "0" * 64
    elif tamper == "runtime":
        contract["runtime"]["trl"] = "0.0.1"
    elif tamper == "profile":
        contract["profile"]["lora_rank"] = 64
    else:
        contract["submitted_code"] = "evil.py"
    with pytest.raises(ValueError):
        builder._validate_contract(contract)


def _verify(fixture, monkeypatch, numeric_bytes, predictions):
    task = fixture["output"]
    artifacts = task.parent / "submission"
    artifacts.mkdir()
    (artifacts / "adapter_model.safetensors").write_bytes(numeric_bytes)
    # The verifier must never parse submitted configuration, metadata or code.
    (artifacts / "adapter_config.json").write_text("not json")
    (artifacts / "model.py").write_text("raise AssertionError('executed submitted code')")
    monkeypatch.setattr(builder, "_check_runtime", lambda: dict(builder.RUNTIME))
    monkeypatch.setattr(builder.trainer, "preflight", lambda *_: {})
    monkeypatch.setattr(
        builder, "_check_base", lambda *_: {"file_sha256": {"actual-base": "a" * 64}}
    )

    def predict(path, rows, spec, profile, model):
        assert all(set(row) == {"id", "image"} for row in rows)
        assert path.parent != artifacts
        assert builder.read_adapter(path)[1] == hashlib.sha256(numeric_bytes).hexdigest()
        return predictions, {"adapter_sha256": hashlib.sha256(numeric_bytes).hexdigest()}

    monkeypatch.setattr(builder, "predict_images", predict)
    return builder.verify_submission(
        task / "tests/contract.json",
        artifacts,
        task / "tests/data/verification.jsonl",
        task / "tests/data_spec.json",
        task / "logs",
        fixture["model_directory"],
    )


def test_fresh_grader_keeps_labels_out_and_uses_shared_optional_think_grammar(
    fixture, monkeypatch, numeric_bytes
):
    builder.make_task(
        **fixture, profile={**asdict(builder.trainer.Profile()), "response_format": "answer_only"}
    )
    result = _verify(
        fixture,
        monkeypatch,
        numeric_bytes,
        {
            "validation-0": {"response": "<think>visible detail</think>\nblue", "terminated": True},
            "validation-1": {"response": "red", "terminated": False},
        },
    )
    assert result["reward"] == 0.5
    assert result["metrics"]["invalid_answers"] == 1
    assert result["submitted_code_executed"] is False
    assert result["test_consumed"] is False
    assert result["model_source"]["file_sha256"]["actual-base"] == "a" * 64


def test_grader_detects_changed_pixels_before_model_load(fixture, monkeypatch, numeric_bytes):
    builder.make_task(**fixture)
    Image.new("RGB", (28, 28), "purple").save(fixture["output"] / "tests/data/image-0.png")
    result = _verify(fixture, monkeypatch, numeric_bytes, {})
    assert result["reward"] == 0
    assert "pixels changed" in result["error"]["message"]


def test_invalid_adapter_fails_before_runtime_import_or_base_load(fixture, monkeypatch):
    builder.make_task(**fixture)
    task = fixture["output"]
    artifacts = task.parent / "bad-submission"
    artifacts.mkdir()
    (artifacts / "adapter_model.safetensors").write_bytes(b"bad numeric artifact")
    monkeypatch.setattr(builder, "_check_runtime", lambda: pytest.fail("runtime reached"))
    result = builder.verify_submission(
        task / "tests/contract.json",
        artifacts,
        task / "tests/data/verification.jsonl",
        task / "tests/data_spec.json",
        task / "logs",
        fixture["model_directory"],
    )
    assert result["reward"] == 0
    assert (task / "logs/reward.txt").read_text() == "0\n"


def test_mounted_identity_hashes_actual_files(monkeypatch, tmp_path):
    (tmp_path / "model-source.json").write_text("{}")
    (tmp_path / ".pinned-download.json").write_text("{}")
    contract = {
        "model_manifest_sha256": builder._digest(tmp_path / "model-source.json"),
        "model_marker_sha256": builder._digest(tmp_path / ".pinned-download.json"),
        "model": builder.trainer.DEFAULT_MODEL,
    }
    seen = []
    monkeypatch.setattr(
        builder.trainer,
        "_verified_model_directory",
        lambda *args: seen.append(args) or {"file_sha256": {"model": "f" * 64}},
    )
    assert builder._check_base(contract, tmp_path)["file_sha256"] == {"model": "f" * 64}
    assert seen == [(tmp_path.resolve(), builder.trainer.DEFAULT_MODEL)]
    (tmp_path / "model-source.json").write_text("tampered")
    with pytest.raises(ValueError, match="provenance"):
        builder._check_base(contract, tmp_path)


@pytest.mark.parametrize("sft_exposure_seconds", [2.0, 5.0])
def test_controls_preserve_exposure_endpoint_and_resume_same_budget(
    fixture, monkeypatch, numeric_bytes, sft_exposure_seconds
):
    builder.make_task(**fixture)
    app = fixture["output"] / "environment"
    output = fixture["output"] / "training"
    calls, receipts = [], {}
    monkeypatch.setattr(builder, "_check_runtime", lambda: dict(builder.RUNTIME))
    monkeypatch.setattr(builder, "_check_base", lambda *_: {"file_sha256": {}})

    def train(train, spec, work, **options):
        assert options["train_only"] is True
        assert options["device"] == "cpu"
        calls.append(options)
        step = 4 if options.get("stop_after_steps") else 7
        elapsed = 4.0 if options["mode"] == "grpo" else sft_exposure_seconds if step == 4 else 4.5
        adapter = work / "final_adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        # Changing bytes proves that time continuation cannot overwrite the
        # independently archived exposure endpoint, without loading any model.
        payload = numeric_bytes if step == 4 else numeric_bytes + b"time-endpoint-fixture"
        (adapter / "adapter_model.safetensors").write_bytes(payload)
        (adapter / "adapter_config.json").write_text("{}")
        (work / "local_training.json").write_text(json.dumps({"stage": options["stage_id"]}))
        record = {"optimizer_audit": {"changed_tensors": 1}, "reward_audit": None}
        receipts[work] = {
            "global_step": step,
            "committed_active_seconds": elapsed,
            "checkpoint": {"path": f"checkpoint-{step}"},
            "stage_id": options["stage_id"],
            "final_adapter": {"weights_sha256": hashlib.sha256(payload).hexdigest()},
        }
        return record

    def receipt(work, expected_stage_id):
        assert receipts[work]["stage_id"] == expected_stage_id
        return copy.deepcopy(receipts[work])

    monkeypatch.setattr(builder, "_invoke_training", train)
    monkeypatch.setattr(builder.trainer, "read_endpoint_receipt", receipt)
    result = builder.run_training(
        app / "contract.json", output, app=app, model_directory=fixture["model_directory"]
    )
    assert result["status"] == "completed"
    assert result["held_out_benchmark_gain_measured"] is False
    assert len(calls) == (3 if sft_exposure_seconds < 4 else 2)
    assert all(call["max_steps"] == 32 and call["max_active_seconds"] == 14400 for call in calls)
    assert calls[0]["init_adapter"] == calls[1]["init_adapter"]
    assert (
        output / "endpoints/sft-exposure/final_adapter/adapter_model.safetensors"
    ).read_bytes() == numeric_bytes
    assert (output / "adapter_model.safetensors").read_bytes() == numeric_bytes
    if len(calls) == 3:
        assert calls[2]["resume_from_checkpoint"] == output / "sft/checkpoint-4"
        assert calls[2]["stop_after_active_seconds"] == 4.0
        assert "stop_after_steps" not in calls[2]


def test_training_rechecks_pixels_before_native_trainer(fixture, monkeypatch):
    builder.make_task(**fixture)
    app = fixture["output"] / "environment"
    Image.new("RGB", (28, 28), "purple").save(app / "data/image-0.png")
    monkeypatch.setattr(builder, "_check_runtime", lambda: {})
    monkeypatch.setattr(builder, "_check_base", lambda *_: {})
    monkeypatch.setattr(
        builder.trainer, "train_local", lambda *_args, **_kwargs: pytest.fail("Trainer reached")
    )
    with pytest.raises(ValueError, match="TRAIN pixels"):
        builder.run_training(
            app / "contract.json",
            fixture["output"] / "training",
            app=app,
            model_directory=fixture["model_directory"],
        )


def test_actual_harbor_schema_resolves_fresh_verifier(fixture):
    pytest.importorskip("harbor")
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    builder.make_task(**fixture)
    task = Task(fixture["output"])
    verifier = resolve_verifier_environment_definition(task.config, task.paths)
    assert verifier.directory == fixture["output"] / "environment"
    assert verifier.bundled_tests is False
    assert verifier.config.memory_mb == 24576


def test_initialization_cannot_silently_change_lora_scale(fixture):
    config = fixture["initial_adapter"] / "adapter_config.json"
    value = json.loads(config.read_text())
    value["lora_alpha"] = 8
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="LoRA scale"):
        builder.make_task(**fixture)


def test_each_native_invocation_is_fresh_process_with_original_resume_budgets(
    tmp_path, monkeypatch
):
    work = tmp_path / "trajectory"
    work.mkdir()
    (work / "local_training.json").write_text('{"status":"completed"}')
    seen = []
    monkeypatch.setattr(
        builder.subprocess, "run", lambda command, **kwargs: seen.append((command, kwargs))
    )
    result = builder._invoke_training(
        tmp_path / "train.jsonl",
        tmp_path / "data_spec.json",
        work,
        profile=builder.trainer.Profile(),
        mode="sft",
        device="cpu",
        train_only=True,
        max_steps=32,
        max_active_seconds=14400,
        stop_after_active_seconds=17.5,
        resume_from_checkpoint=work / "checkpoint-4",
        stage_id="sft-time",
    )
    command, options = seen[0]
    assert command[:4] == [builder.sys.executable, "-I", "-m", "vision_rl.vlm_local_training"]
    assert options == {"check": True}
    assert "--train-only" in command and "--validation" not in command
    for flag, value in (
        ("--max-steps", "32"),
        ("--max-active-seconds", "14400"),
        ("--stop-after-active-seconds", "17.5"),
        ("--resume-from-checkpoint", str(work / "checkpoint-4")),
    ):
        assert command[command.index(flag) + 1] == value
    assert result == {"status": "completed"}
