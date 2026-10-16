import asyncio
import json
import shlex
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_rl.data import assert_disjoint, fingerprint, load_manifest
from vision_rl.task_builder import make_vlm_task

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture
def source_manifests(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    paths = []
    for split, color in (("train", "red"), ("validation", "green"), ("test", "blue")):
        image = root / f"source-{split}-label-B.png"
        Image.new("RGB", (10, 10), color).save(image)
        row = {
            "id": f"private-{split}",
            "split": split,
            "image": image.name,
            "prompt": f"Question for {split}",
            "choices": ["circle", "square"],
            "answer": "B",
            "category": "spatial",
            "extra_secret": "do-not-bundle",
        }
        path = root / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        paths.append(path)
    return paths


def test_generated_task_keeps_test_data_out_of_image(tmp_path, source_manifests, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "never-copy-this-secret")
    output = tmp_path / "task"
    result = make_vlm_task(*source_manifests, output, MODEL, max_steps=2, max_tokens=32)
    assert result["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["test_in_build_context"] is False
    environment = output / "environment"
    assert not (environment / "data/test.jsonl").exists()
    assert not (environment / "data/images/test").exists()
    assert (output / "tests/data/test.jsonl").is_file()
    text = "\n".join(p.read_text() for p in output.rglob("*") if p.is_file() and p.suffix != ".png")
    assert "never-copy-this-secret" not in text
    assert "do-not-bundle" not in text
    public_text = "\n".join(
        p.read_text() for p in environment.rglob("*") if p.is_file() and p.suffix != ".png"
    )
    assert "private-test" not in public_text
    assert "Question for test" not in public_text
    assert str(source_manifests[0].parent) not in text
    bundled = [
        load_manifest(environment / "data/train.jsonl", "train"),
        load_manifest(environment / "data/validation.jsonl", "validation"),
        load_manifest(output / "tests/data/test.jsonl", "test"),
    ]
    assert_disjoint(*bundled)
    for source, rows in zip(source_manifests, bundled, strict=True):
        assert fingerprint(load_manifest(source)) == fingerprint(rows)
    # The task remains self-contained after deleting the original source images.
    for image in source_manifests[0].parent.glob("*.png"):
        image.unlink()
    assert len(load_manifest(output / "tests/data/test.jsonl")) == 1


def test_bundled_images_strip_source_metadata(tmp_path, source_manifests):
    image_path = source_manifests[0].parent / json.loads(source_manifests[0].read_text())["image"]
    exif = Image.Exif()
    exif[270] = "hidden answer: B; private source metadata"
    Image.new("RGB", (10, 10), "red").save(image_path, exif=exif)
    output = tmp_path / "task"
    make_vlm_task(*source_manifests, output, MODEL)
    bundled = load_manifest(output / "environment/data/train.jsonl")
    with Image.open(bundled[0]["image"]) as image:
        assert not image.getexif()


def test_generated_harbor_schema_uses_built_context_and_fresh_gpu(tmp_path, source_manifests):
    output = tmp_path / "task"
    make_vlm_task(*source_manifests, output, MODEL)
    config = tomllib.loads((output / "task.toml").read_text())
    assert config["schema_version"] == "1.4"
    assert config["environment"]["gpus"] == 1
    assert config["verifier"]["environment_mode"] == "separate"
    assert "docker_image" not in config["environment"]
    assert "environment" not in config["verifier"]
    assert "env" not in config["environment"]
    dockerfile = (output / "environment/Dockerfile").read_text()
    assert "FROM vision-rl:gpu" in dockerfile
    assert "ENTRYPOINT" not in dockerfile
    assert "COPY ." not in dockerfile
    assert "tests" not in dockerfile
    script = (output / "environment/train.sh").read_text()
    args = shlex.split(script.splitlines()[-1])
    assert args[args.index("--model") + 1] == MODEL
    assert args[args.index("--output") + 1] == "/logs/artifacts/training"
    assert "--test" not in args
    assert "python -I /tests/verify.py" in (output / "tests/test.sh").read_text()
    compile((output / "tests/verify.py").read_text(), "verify.py", "exec")
    pytest.importorskip("harbor")
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    task = Task(output)
    verifier = resolve_verifier_environment_definition(task.config, task.paths)
    assert verifier.directory == output / "environment"
    assert verifier.bundled_tests is False  # Harbor uploads tests only at verifier time.
    assert verifier.config.gpus == 1


@pytest.mark.parametrize(
    "model",
    [
        "org/model; touch /tmp/bad",
        "org/model'$(id)'",
        "$(id)",
        "org/a\ncommand",
        "/host/model",
        "../model",
        "a/b/c",
        "https://huggingface.co/org/model",
        "",
    ],
)
def test_hostile_or_nonportable_model_ids_rejected(tmp_path, source_manifests, model):
    output = tmp_path / "task"
    with pytest.raises(ValueError, match="repository ID"):
        make_vlm_task(*source_manifests, output, model)
    assert not output.exists()


def test_split_leakage_rejected_before_writing(tmp_path, source_manifests):
    test = json.loads(source_manifests[2].read_text())
    test["image"] = json.loads(source_manifests[0].read_text())["image"]
    source_manifests[2].write_text(json.dumps(test))
    output = tmp_path / "task"
    with pytest.raises(ValueError, match="leakage"):
        make_vlm_task(*source_manifests, output, MODEL)
    assert not output.exists()


def test_wrong_split_and_existing_output_rejected(tmp_path, source_manifests):
    output = tmp_path / "task"
    with pytest.raises(ValueError, match="expected test split"):
        make_vlm_task(*source_manifests[:2], source_manifests[0], output, MODEL)
    output.mkdir()
    marker = output / "preserve"
    marker.write_text("mine")
    with pytest.raises(ValueError, match="already exists"):
        make_vlm_task(*source_manifests, output, MODEL)
    assert marker.read_text() == "mine"


def test_bundle_failure_is_atomic(tmp_path, source_manifests, monkeypatch):
    import vision_rl.task_builder as builder

    def fail(*args):
        raise OSError("disk is full")

    monkeypatch.setattr(builder, "_bundle", fail)
    output = tmp_path / "task"
    with pytest.raises(OSError, match="disk is full"):
        make_vlm_task(*source_manifests, output, MODEL)
    assert not output.exists()
    assert not list(tmp_path.glob(".vlm-task-*"))


@pytest.mark.parametrize("malformed", [False, True])
def test_generated_verifier_uses_only_adapter_weights(
    tmp_path, source_manifests, monkeypatch, malformed
):
    import vision_rl.evaluation as evaluation

    output = tmp_path / "task"
    make_vlm_task(*source_manifests, output, MODEL, max_tokens=27)
    logs = tmp_path / "verifier"
    source = tmp_path / "training/final_adapter"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": MODEL, "peft_type": "LORA"})
    )
    (source / "adapter_model.safetensors").write_bytes(
        b"fake numeric weights for mocked evaluation"
    )
    (source / "processor_config.json").write_text("untrusted processor config")
    (source / "evil.py").write_text("raise RuntimeError('must never execute')")
    (source.parent / "training_metadata.json").write_text('{"model_revision":"evil"}')
    if malformed:
        (source / "adapter_model.safetensors").unlink()
    calls = []

    def fake_evaluate(manifest, destination, **kwargs):
        calls.append(kwargs)
        assert kwargs["vlm_model"] == MODEL
        assert kwargs["max_tokens"] == 27
        adapter = kwargs["adapter"]
        assert {p.name for p in adapter.iterdir()} == {
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        assert not (adapter.parent / "training_metadata.json").exists()
        assert manifest == Path("/tests/data/test.jsonl")
        return {"accuracy": 0.75}

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    # Relocate the two fixed container paths for a local verifier contract test.
    script = (output / "tests/verify.py").read_text()
    script = script.replace('Path("/logs/verifier")', f"Path({str(logs)!r})")
    script = script.replace(
        'Path("/logs/artifacts/training/final_adapter")', f"Path({str(source)!r})"
    )
    exec(compile(script, "generated-verifier.py", "exec"), {})
    if malformed:
        assert calls == []
        assert (logs / "reward.txt").read_text() == "0\n"
        assert json.loads((logs / "error.json").read_text()) == {"type": "ValueError"}
    else:
        assert len(calls) == 1
        assert (logs / "reward.txt").read_text() == "0.75\n"


def test_vlm_training_agent_runs_task_script_and_surfaces_failure(tmp_path):
    pytest.importorskip("harbor")
    from vision_rl.harbor_agent import VLMTrainingAgent

    calls = []

    class Environment:
        code = 0

        async def exec(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(stdout="trained", stderr="", return_code=self.code)

    environment = Environment()
    agent = VLMTrainingAgent(logs_dir=tmp_path / "agent-logs")
    asyncio.run(agent.run("ignored", environment, None))
    assert calls == [{"command": "bash /app/train.sh", "timeout_sec": 3500}]
    assert (tmp_path / "agent-logs/training.log").read_text() == "trained"
    environment.code = 2
    with pytest.raises(RuntimeError, match="status 2"):
        asyncio.run(agent.run("ignored", environment, None))
