"""Package private image splits as a self-contained Harbor GPU training task."""

from __future__ import annotations

import json
import re
import shlex
import tempfile
from pathlib import Path

from PIL import Image

from .data import assert_disjoint, fingerprint, load_manifest


def _model_id(value: str) -> str:
    # Only Hub repository IDs are portable into a generated container. A host
    # pathname or shell expression is not a model reference for this interface.
    if (
        not isinstance(value, str)
        or len(value) > 96
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?", value)
        or ".." in value
        or "--" in value
        or any(part.endswith(".") for part in value.split("/"))
    ):
        raise ValueError(
            "model must be a Hugging Face repository ID, such as Qwen/Qwen2.5-VL-3B-Instruct"
        )
    return value


def _bundle(rows: list[dict], destination: Path, split: str) -> None:
    """Copy pixels and the explicit question schema; omit source paths/metadata."""
    image_dir = destination / "images" / split
    image_dir.mkdir(parents=True)
    serialized = []
    for index, row in enumerate(rows):
        image_path = image_dir / f"{index:08d}.png"
        with Image.open(row["image"]) as source:
            pixels = source.convert("RGB")
            pixels.info.clear()
            pixels.save(image_path)
        item = {key: row[key] for key in ("id", "split", "prompt", "choices", "answer", "category")}
        item["image"] = image_path.relative_to(destination).as_posix()
        serialized.append(json.dumps(item, ensure_ascii=False))
    (destination / f"{split}.jsonl").write_text("\n".join(serialized) + "\n", encoding="utf-8")


def _verifier(model: str, max_tokens: int) -> str:
    return f'''"""Runs in a fresh image, using fixed evaluator code and held-out data."""

import json
import shutil
import tempfile
from pathlib import Path

from vision_rl.data import write_json
from vision_rl.evaluation import evaluate

MODEL = {model!r}
MAX_TOKENS = {max_tokens!r}
logs = Path("/logs/verifier")
logs.mkdir(parents=True, exist_ok=True)
(logs / "reward.txt").write_text("0\\n")
try:
    source = Path("/logs/artifacts/training/final_adapter")
    # Copy only numeric safetensors and its configuration. Agent-supplied Python,
    # chat templates, processor files, and training metadata never enter inference.
    with tempfile.TemporaryDirectory(prefix="trusted-adapter-") as temporary:
        adapter = Path(temporary) / "adapter"
        adapter.mkdir()
        for name, limit in (("adapter_config.json", 65536),
                            ("adapter_model.safetensors", 2 * 1024**3)):
            path = source / name
            if (not path.is_file() or path.is_symlink()
                    or not path.resolve().is_relative_to(source.resolve())
                    or path.stat().st_size > limit):
                raise ValueError("Missing, unsafe, or oversized adapter artifact")
            shutil.copyfile(path, adapter / name)
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != MODEL or config.get("peft_type") != "LORA":
            raise ValueError("Submission must be a LoRA adapter for the configured base model")
        result = evaluate(Path("/tests/data/test.jsonl"), logs / "evaluation.json",
                          vlm_model=MODEL, adapter=adapter, max_tokens=MAX_TOKENS)
        (logs / "reward.txt").write_text(str(result["accuracy"]) + "\\n")
except Exception as exc:
    write_json(logs / "error.json", {{"type": type(exc).__name__}})
    (logs / "reward.txt").write_text("0\\n")
'''


def make_vlm_task(
    train: Path,
    validation: Path,
    test: Path,
    output: Path,
    model: str,
    max_steps: int = 100,
    max_tokens: int = 64,
) -> dict:
    """Create an offline task bundle; this never downloads models or starts jobs.

    Build ``vision-rl:gpu`` on an NVIDIA Linux host first. Harbor builds the task
    image from ``environment/``; test data lives exclusively in ``tests/`` and is
    uploaded to a separate verifier after the training agent finishes.
    """
    model = _model_id(model)
    for name, value in (("max_steps", max_steps), ("max_tokens", max_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Task output already exists; choose a new directory")
    datasets = {
        split: load_manifest(path, expected_split=split)
        for split, path in (("train", train), ("validation", validation), ("test", test))
    }
    assert_disjoint(*datasets.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vlm-task-", dir=output.parent) as temporary:
        staged = Path(temporary) / "task"
        environment = staged / "environment"
        tests = staged / "tests"
        solution = staged / "solution"
        solution.mkdir(parents=True)
        _bundle(datasets["train"], environment / "data", "train")
        _bundle(datasets["validation"], environment / "data", "validation")
        _bundle(datasets["test"], tests / "data", "test")
        (environment / "Dockerfile").write_text(
            "FROM vision-rl:gpu\n"
            "COPY data/ /app/data/\n"
            "COPY train.sh /app/train.sh\n"
            "RUN chmod +x /app/train.sh\n"
            'CMD ["sleep", "infinity"]\n',
            encoding="utf-8",
        )
        command = shlex.join(
            [
                "python",
                "-m",
                "vision_rl.vlm",
                "--train",
                "/app/data/train.jsonl",
                "--validation",
                "/app/data/validation.jsonl",
                "--output",
                "/logs/artifacts/training",
                "--model",
                model,
                "--max-steps",
                str(max_steps),
                "--max-completion-length",
                str(max_tokens),
            ]
        )
        (environment / "train.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + command + "\n")
        (solution / "solve.sh").write_text("#!/bin/bash\nset -euo pipefail\nbash /app/train.sh\n")
        (tests / "test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
            "echo 0 > /logs/verifier/reward.txt\npython -I /tests/verify.py\n"
        )
        (tests / "verify.py").write_text(_verifier(model, max_tokens), encoding="utf-8")
        for script in (environment / "train.sh", solution / "solve.sh", tests / "test.sh"):
            script.chmod(0o755)
        (staged / "task.toml").write_text(
            """schema_version = "1.4"

[task]
name = "vision-rl/custom-vlm-specialist"
version = "0.1.0"
description = "Train a visual specialist and evaluate its adapter on held-out images"

[metadata]
category = "machine-learning"
purpose = "specialist-training"

[agent]
timeout_sec = 3600.0

[verifier]
timeout_sec = 900.0
environment_mode = "separate"

[environment]
cpus = 4
memory_mb = 32768
storage_mb = 65536
gpus = 1
network_mode = "public"
""",
            encoding="utf-8",
        )
        (staged / "instruction.md").write_text(
            "# Train a vision-language specialist\n\n"
            f"Fine-tune `{model}` with GRPO and LoRA using the labeled images in "
            "`/app/data/train.jsonl`. Use `/app/data/validation.jsonl` for model selection. "
            "Run `bash /app/train.sh` for the provided bounded training workflow.\n\n"
            "Submit the final adapter under `/logs/artifacts/training/final_adapter/`, "
            "including `adapter_config.json` and `adapter_model.safetensors`. "
            "Only LoRA adapters for the configured base model are accepted. "
            "Supported adapters target q_proj/v_proj, use rank 1 to 256, and retain "
            "the standard LoRA settings from the provided trainer. The processor "
            "and chat template are fixed by that base model; submitted processor code, "
            "templates, and Python files are ignored. Keep the configuration under 64 KiB "
            "and adapter weights under 2 GiB.\n\n"
            "Harbor evaluates in a fresh GPU container using private test images. "
            "Reward is strict multiple-choice answer accuracy from 0 to 1. "
            "No test data is included in the training image. Keep training logs and "
            "metadata beside the adapter for review. A task reward alone does not "
            "establish a win against a frontier model; compare frozen models on the "
            "same held-out examples and inference protocol.\n",
            encoding="utf-8",
        )
        bundle = {
            "schema_version": 1,
            "model": model,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "counts": {k: len(v) for k, v in datasets.items()},
            "fingerprints": {k: fingerprint(v) for k, v in datasets.items()},
            "test_in_build_context": False,
        }
        (staged / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return {
        **bundle,
        "task": str(output.resolve()),
        "agent": "vision_rl.harbor_agent:VLMTrainingAgent",
    }
