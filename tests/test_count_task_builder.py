import asyncio
import json
import shlex
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_rl.count_task_builder import make_count_task
from vision_rl.data import assert_disjoint, fingerprint, load_manifest


@pytest.fixture
def count_manifests(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    paths = []
    for split, color in (("train", "red"), ("validation", "green"), ("test", "blue")):
        image = source / f"{split}-answer-1.png"
        Image.new("RGB", (10, 10), color).save(image)
        path = source / f"{split}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": f"example-{split}",
                    "split": split,
                    "image": image.name,
                    "prompt": "How many screws are visible?",
                    "choices": ["0", "1", "2"],
                    "answer": "B",
                    "category": "counting",
                }
            )
            + "\n"
        )
        paths.append(path)
    return paths


def test_count_task_contains_only_training_data_in_build_context(tmp_path, count_manifests):
    output = tmp_path / "count-task"
    result = make_count_task(*count_manifests, output, sft_epochs=2, rl_steps=3)
    assert result["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["test_in_build_context"] is False
    assert result["agent"] == "vision_rl.harbor_agent:CountTrainingAgent"
    assert not (output / "environment/data/test.jsonl").exists()
    rows = [
        load_manifest(output / "environment/data/train.jsonl", "train"),
        load_manifest(output / "environment/data/validation.jsonl", "validation"),
        load_manifest(output / "tests/data/test.jsonl", "test"),
    ]
    assert_disjoint(*rows)
    for original, bundled in zip(count_manifests, rows, strict=True):
        assert fingerprint(load_manifest(original)) == fingerprint(bundled)
    public_text = "\n".join(
        p.read_text()
        for p in (output / "environment").rglob("*")
        if p.is_file() and p.suffix != ".png"
    )
    assert "example-test" not in public_text
    assert str(count_manifests[0].parent) not in public_text
    script = (output / "environment/train.sh").read_text()
    arguments = shlex.split(script.splitlines()[-1])
    assert arguments[:4] == ["python", "-m", "vision_rl.vision_policy", "train"]
    for flag, expected in (
        ("--encoder-weights", "/opt/encoder.pt"),
        ("--device", "cpu"),
        ("--sft-epochs", "2"),
        ("--rl-steps", "3"),
    ):
        assert arguments[arguments.index(flag) + 1] == expected
    assert "--test" not in arguments
    assert "bash /app/train.sh" in (output / "solution/solve.sh").read_text()


def test_count_task_schema_is_offline_and_uploads_tests_only_for_verifier(
    tmp_path, count_manifests
):
    output = tmp_path / "count-task"
    make_count_task(*count_manifests, output)
    config = tomllib.loads((output / "task.toml").read_text())
    assert config["environment"]["network_mode"] == "no-network"
    assert config["environment"]["memory_mb"] == 3072
    assert config["environment"]["cpus"] == 2
    assert "docker_image" not in config["environment"]
    assert "environment" not in config["verifier"]
    assert config["verifier"]["environment_mode"] == "separate"
    dockerfile = (output / "environment/Dockerfile").read_text()
    assert "FROM vision-rl:vision" in dockerfile
    assert "ENTRYPOINT" not in dockerfile
    assert "tests" not in dockerfile
    assert "COPY ." not in dockerfile
    assert "python -I /tests/verify.py" in (output / "tests/test.sh").read_text()
    pytest.importorskip("harbor")
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    task = Task(output)
    verifier = resolve_verifier_environment_definition(task.config, task.paths)
    assert verifier.directory == output / "environment"
    assert verifier.bundled_tests is False


@pytest.mark.parametrize("budget", [{"sft_epochs": 0}, {"rl_steps": -1}, {"rl_steps": True}])
def test_invalid_count_training_budget_rejected(tmp_path, count_manifests, budget):
    with pytest.raises(ValueError, match="positive integer"):
        make_count_task(*count_manifests, tmp_path / "task", **budget)


def test_count_task_rejects_inconsistent_question_and_overlap(tmp_path, count_manifests):
    test = json.loads(count_manifests[2].read_text())
    test["prompt"] = "A different counting question"
    count_manifests[2].write_text(json.dumps(test))
    output = tmp_path / "task"
    with pytest.raises(ValueError, match="consistent counting question"):
        make_count_task(*count_manifests, output)
    assert not output.exists()
    test["image"] = json.loads(count_manifests[0].read_text())["image"]
    count_manifests[2].write_text(json.dumps(test))
    with pytest.raises(ValueError, match="leakage"):
        make_count_task(*count_manifests, output)


def test_count_task_does_not_overwrite(tmp_path, count_manifests):
    output = tmp_path / "task"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        make_count_task(*count_manifests, output)
    assert (output / "keep").read_text() == "keep"


@pytest.mark.parametrize("policy_kind", ["valid", "missing", "oversized", "symlink"])
def test_count_verifier_stages_only_policy_and_uses_trusted_encoder(
    tmp_path, count_manifests, monkeypatch, policy_kind
):
    import vision_rl.vision_policy as policy

    output = tmp_path / "task"
    make_count_task(*count_manifests, output)
    training = tmp_path / "training"
    training.mkdir()
    source = training / "policy.pt"
    if policy_kind == "valid":
        source.write_bytes(b"mocked numeric policy")
    elif policy_kind == "oversized":
        with source.open("wb") as handle:
            handle.truncate(5 * 1024**2 + 1)
    elif policy_kind == "symlink":
        (training / "other.pt").write_bytes(b"not a regular submitted file")
        source.symlink_to(training / "other.pt")
    (training / "encoder.pt").write_bytes(b"untrusted replacement encoder")
    (training / "evil.py").write_text("raise RuntimeError('must not execute')")
    logs = tmp_path / "logs"
    calls = []

    def evaluate(manifest, checkpoint, destination, **kwargs):
        calls.append(kwargs)
        assert manifest == Path("/tests/data/test.jsonl")
        assert kwargs == {"device": "cpu", "trusted_encoder": Path("/opt/encoder.pt")}
        assert [p.name for p in checkpoint.parent.iterdir()] == ["policy.pt"]
        assert checkpoint.read_bytes() == b"mocked numeric policy"
        return {"accuracy": 0.625}

    monkeypatch.setattr(policy, "evaluate_count_policy", evaluate)
    script = (output / "tests/verify.py").read_text()
    script = script.replace('Path("/logs/verifier")', f"Path({str(logs)!r})")
    script = script.replace('Path("/logs/artifacts/training/policy.pt")', f"Path({str(source)!r})")
    exec(compile(script, "count-verifier.py", "exec"), {})
    if policy_kind == "valid":
        assert len(calls) == 1
        assert (logs / "reward.txt").read_text() == "0.625\n"
    else:
        assert calls == []
        assert (logs / "reward.txt").read_text() == "0\n"
        assert json.loads((logs / "error.json").read_text()) == {"type": "ValueError"}


def test_count_agent_runs_generated_script(tmp_path):
    pytest.importorskip("harbor")
    from vision_rl.harbor_agent import CountTrainingAgent

    calls = []

    class Environment:
        async def exec(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(stdout="count trained", stderr="", return_code=0)

    agent = CountTrainingAgent(logs_dir=tmp_path / "agent")
    asyncio.run(agent.run("task", Environment(), None))
    assert calls == [{"command": "bash /app/train.sh", "timeout_sec": 1150}]
    assert (tmp_path / "agent/training.log").read_text() == "count trained"
