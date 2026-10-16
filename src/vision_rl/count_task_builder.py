"""Build offline Harbor tasks for a learned policy on real counting images."""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path

from .data import assert_disjoint, fingerprint, load_manifest
from .task_builder import _bundle
from .vision_policy import ENCODER_ID, _count_schema

_VERIFY = '''"""Grade numeric policy weights with the pristine image's frozen encoder."""

import shutil
import tempfile
from pathlib import Path

from vision_rl.data import write_json
from vision_rl.vision_policy import evaluate_count_policy

logs = Path("/logs/verifier")
logs.mkdir(parents=True, exist_ok=True)
(logs / "reward.txt").write_text("0\\n")
try:
    source = Path("/logs/artifacts/training/policy.pt")
    if (not source.is_file() or source.is_symlink()
            or source.stat().st_size > 5 * 1024**2):
        raise ValueError("Missing, unsafe, or oversized policy checkpoint")
    with tempfile.TemporaryDirectory(prefix="trusted-count-policy-") as temporary:
        checkpoint = Path(temporary) / "policy.pt"
        shutil.copyfile(source, checkpoint)
        # Only the small policy is staged. Agent-supplied encoder weights, code,
        # preprocessing, and metadata files are never used by the verifier.
        result = evaluate_count_policy(
            Path("/tests/data/test.jsonl"), checkpoint, logs / "evaluation.json",
            device="cpu", trusted_encoder=Path("/opt/encoder.pt"),
        )
        (logs / "reward.txt").write_text(str(result["accuracy"]) + "\\n")
except Exception as exc:
    write_json(logs / "error.json", {"type": type(exc).__name__})
    (logs / "reward.txt").write_text("0\\n")
'''


def make_count_task(
    train: Path,
    validation: Path,
    test: Path,
    output: Path,
    sft_epochs: int = 160,
    rl_steps: int = 800,
) -> dict:
    """Package three disjoint real-image counting splits without starting a run.

    The base image ``vision-rl:vision`` must contain the installed optional vision
    dependencies and immutable ResNet18 encoder at ``/opt/encoder.pt``. Harbor
    builds the task image from its public ``environment/`` directory, then uploads
    ``tests/`` only to a separate verifier after the training agent finishes.
    """
    for name, value in (("sft_epochs", sft_epochs), ("rl_steps", rl_steps)):
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
    choices = _count_schema([row for rows in datasets.values() for row in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count-task-", dir=output.parent) as temporary:
        staged = Path(temporary) / "task"
        environment, tests, solution = (
            staged / "environment",
            staged / "tests",
            staged / "solution",
        )
        solution.mkdir(parents=True)
        _bundle(datasets["train"], environment / "data", "train")
        _bundle(datasets["validation"], environment / "data", "validation")
        _bundle(datasets["test"], tests / "data", "test")
        (environment / "Dockerfile").write_text(
            "FROM vision-rl:vision\n"
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
                "vision_rl.vision_policy",
                "train",
                "--train",
                "/app/data/train.jsonl",
                "--validation",
                "/app/data/validation.jsonl",
                "--output",
                "/logs/artifacts/training",
                "--device",
                "cpu",
                "--encoder-weights",
                "/opt/encoder.pt",
                "--sft-epochs",
                str(sft_epochs),
                "--rl-steps",
                str(rl_steps),
            ]
        )
        (environment / "train.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + command + "\n")
        (solution / "solve.sh").write_text("#!/bin/bash\nset -euo pipefail\nbash /app/train.sh\n")
        (tests / "test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
            "echo 0 > /logs/verifier/reward.txt\npython -I /tests/verify.py\n"
        )
        (tests / "verify.py").write_text(_VERIFY, encoding="utf-8")
        for script in (environment / "train.sh", solution / "solve.sh", tests / "test.sh"):
            script.chmod(0o755)
        (staged / "task.toml").write_text(
            """schema_version = "1.4"

[task]
name = "vision-rl/real-image-count-specialist"
version = "0.1.0"
description = "Train a count policy on image pixels and evaluate with an immutable encoder"

[metadata]
category = "machine-learning"
purpose = "specialist-training"

[agent]
timeout_sec = 1200.0

[verifier]
timeout_sec = 1200.0
environment_mode = "separate"

[environment]
cpus = 2
memory_mb = 3072
storage_mb = 8192
network_mode = "no-network"
""",
            encoding="utf-8",
        )
        (staged / "instruction.md").write_text(
            "# Train a real-image counting specialist\n\n"
            "Train a count policy using the images and labels in `/app/data/train.jsonl`. "
            "Use `/app/data/validation.jsonl` for choosing a checkpoint. The provided "
            "`bash /app/train.sh` workflow warms up a small policy with supervised learning, "
            "then updates it with sampled REINFORCE rewards.\n\n"
            f"The fixed encoder is `{ENCODER_ID}` at `/opt/encoder.pt`. "
            "Only the head and its feature normalization are submitted: write "
            "`/logs/artifacts/training/policy.pt`, smaller than 5 MiB, using the supported "
            "`vision_rl.vision_policy` checkpoint schema. The verifier reads tensor weights "
            "with restricted loading, checks their shapes and metadata, and uses its own "
            "pristine encoder. Submitted Python, encoder replacements, and preprocessing "
            "code are ignored.\n\n"
            "Test images are available only in the separate verifier. Reward is exact "
            "count accuracy from 0 to 1. Keep experiment history and the SFT checkpoint "
            "beside your policy so the contribution of RL can be measured separately. "
            "A held-out task score alone does not establish a frontier-model win.\n",
            encoding="utf-8",
        )
        bundle = {
            "schema_version": 1,
            "encoder": ENCODER_ID,
            "choices": choices,
            "sft_epochs": sft_epochs,
            "rl_steps": rl_steps,
            "counts": {key: len(rows) for key, rows in datasets.items()},
            "fingerprints": {key: fingerprint(rows) for key, rows in datasets.items()},
            "test_in_build_context": False,
        }
        (staged / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return {
        **bundle,
        "task": str(output.resolve()),
        "agent": "vision_rl.harbor_agent:CountTrainingAgent",
    }
