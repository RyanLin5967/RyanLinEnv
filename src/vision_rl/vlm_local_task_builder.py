"""Offline Harbor integration for the pinned local Qwen2-VL training path.

This first CPU task is an integration diagnostic, never a benchmark result.
Only numeric rank-eight language Q/V LoRA weights cross the grading boundary.
The verifier owns the base snapshot, processor, prompt, labels and generation
profile. No submitted configuration, tokenizer, Python or pickle is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from . import vlm_local_training as trainer
from .data import assert_disjoint

ADAPTER_LIMIT = 16 * 1024**2
JSON_LIMIT = 1024**2
PACKAGE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "vlm_local_task_builder.py",
    "vlm_local_training.py",
    "vlm.py",
    "aircraft_data.py",
    "data.py",
)
RUNTIME = {**trainer.STACK, "numpy": "2.5.4", "pillow": "12.3.0", "safetensors": "0.8.0"}
MODEL_PATH = Path("/opt/pinned-model")


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sources():
    return {name: _digest(PACKAGE / name) for name in SOURCE_FILES}


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _bounded_bytes(path: Path, maximum: int) -> bytes:
    """Open each component without following links, including FIFO-safe final open."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Artifact path cannot contain parent traversal")
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as stream:
            attributes = os.fstat(stream.fileno())
            if not stat.S_ISREG(attributes.st_mode) or not 0 < attributes.st_size <= maximum:
                raise ValueError("Artifact must be a bounded nonempty regular file")
            value = stream.read(maximum + 1)
            if len(value) != attributes.st_size or len(value) > maximum:
                raise ValueError("Artifact changed size while reading")
            return value
    finally:
        os.close(directory)


def adapter_shapes():
    """Exact public Qwen2-VL-2B architecture, independent of submitted metadata."""
    shapes = {}
    for layer in range(28):
        for projection, width in (("q_proj", 1536), ("v_proj", 256)):
            prefix = f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}"
            shapes[prefix + ".lora_A.weight"] = (8, 1536)
            shapes[prefix + ".lora_B.weight"] = (width, 8)
    return shapes


def read_adapter(path: Path):
    """Return validated small numeric arrays and the digest of exactly those bytes."""
    import numpy as np
    from safetensors.numpy import load

    raw = _bounded_bytes(path, ADAPTER_LIMIT)
    tensors = load(raw)
    shapes = adapter_shapes()
    if set(tensors) != set(shapes):
        raise ValueError("Adapter tensor names differ from fixed language Q/V rank-eight LoRA")
    for name, value in tensors.items():
        if value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Adapter requires exact finite FP32 tensor shapes")
    return tensors, hashlib.sha256(raw).hexdigest()


def _pixel_identity(rows):
    return hashlib.sha256(
        json.dumps({row["id"]: row["image_sha256"] for row in rows}, sort_keys=True).encode()
    ).hexdigest()


def _fixed_lora_config():
    # Construct from the trusted profile, never from a submitted adapter_config.json.
    return trainer._validated_lora_config(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        }
    )


def _validate_contract(contract):
    expected = {
        "schema_version",
        "scope",
        "model",
        "model_revision",
        "model_manifest_sha256",
        "model_marker_sha256",
        "data_spec_sha256",
        "profile",
        "source_sha256",
        "runtime",
        "train_sha256",
        "verification_sha256",
        "train_pixels_sha256",
        "verification_pixels_sha256",
        "initial_weights_sha256",
        "initial_record_sha256",
        "max_steps",
        "control_step_cap",
        "max_active_seconds",
        "learning_rate",
        "beta",
        "seed",
    }
    if not isinstance(contract, dict) or set(contract) != expected:
        raise ValueError("Unexpected trusted VLM task contract fields")
    if (
        type(contract["schema_version"]) is not int
        or contract["schema_version"] != 1
        or contract["scope"] != "integration_only"
    ):
        raise ValueError("This task is explicitly an integration diagnostic")
    if (
        contract["model"] != trainer.DEFAULT_MODEL
        or contract["model_revision"] != (trainer.MODEL_REVISIONS[trainer.DEFAULT_MODEL])
    ):
        raise ValueError("Only the pinned Qwen2-VL-2B snapshot is supported")
    for name in (
        "model_manifest_sha256",
        "model_marker_sha256",
        "data_spec_sha256",
        "train_sha256",
        "verification_sha256",
        "train_pixels_sha256",
        "verification_pixels_sha256",
        "initial_weights_sha256",
        "initial_record_sha256",
    ):
        if not isinstance(contract[name], str) or not re.fullmatch(r"[0-9a-f]{64}", contract[name]):
            raise ValueError("Contract requires immutable SHA256 identities")
    if contract["source_sha256"] != _sources() or contract["runtime"] != RUNTIME:
        raise ValueError("Installed implementation differs from the trusted task")
    if not isinstance(contract["profile"], dict) or set(contract["profile"]) != set(
        asdict(trainer.Profile())
    ):
        raise ValueError("Freeze every generation/profile field")
    profile = trainer.Profile(**contract["profile"])
    profile.validate()
    if profile.dtype != "float32" or profile.lora_rank != 8 or profile.num_generations != 4:
        raise ValueError("CPU integration fixes FP32, rank-eight Q/V and four generations")
    for name, minimum, maximum in (
        ("max_steps", 1, 4),
        ("control_step_cap", 1, 32),
        ("seed", 0, 2**32 - 1),
    ):
        if type(contract[name]) is not int or not minimum <= contract[name] <= maximum:
            raise ValueError(f"Invalid bounded {name}")
    if contract["control_step_cap"] < contract["max_steps"]:
        raise ValueError("Control cap must permit equal-exposure updates")
    for name, maximum in (("max_active_seconds", 14400), ("learning_rate", 0.01), ("beta", 1)):
        value = contract[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"Invalid bounded {name}")
    return profile


def _read_contract(path):
    contract = json.loads(_bounded_bytes(Path(path), JSON_LIMIT))
    return contract, _validate_contract(contract)


def _check_runtime():
    versions = {name: metadata.version(name) for name in RUNTIME}
    if any(versions[name].split("+")[0] != version for name, version in RUNTIME.items()):
        raise ValueError("Installed runtime differs from the task's exact package versions")
    return versions


def _check_base(contract, directory):
    directory = Path(directory).resolve()
    if _digest(directory / "model-source.json") != contract["model_manifest_sha256"] or (
        _digest(directory / ".pinned-download.json") != contract["model_marker_sha256"]
    ):
        raise ValueError("Mounted base snapshot provenance changed")
    # Hash every actual mounted file; the mount name is never model identity.
    return trainer._verified_model_directory(directory, contract["model"])


def predict_images(adapter, image_rows, spec, profile, model_directory):
    """Image-only trusted inference. The caller keeps evaluation labels separate."""
    if not image_rows or any(set(row) != {"id", "image"} for row in image_rows):
        raise ValueError("Prediction accepts only nonempty image-only id/image records")
    if len({row["id"] for row in image_rows}) != len(image_rows):
        raise ValueError("Prediction IDs must be unique")
    tensors, digest = read_adapter(adapter)  # Reject before allocating the base model.
    import peft
    import torch
    import transformers

    torch.set_num_threads(2)
    processor = trainer._processor(model_directory, profile)
    processor_audit = trainer._audit_processor(processor, image_rows, spec, profile)
    base = transformers.AutoModelForImageTextToText.from_pretrained(
        str(model_directory),
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    if base.config.model_type != "qwen2_vl":
        raise ValueError("Mounted model is not the trusted Qwen2-VL architecture")
    policy = peft.get_peft_model(base, peft.LoraConfig(**_fixed_lora_config()))
    expected = peft.get_peft_model_state_dict(policy)
    if set(expected) != set(tensors) or any(
        tuple(expected[k].shape) != v.shape for k, v in tensors.items()
    ):
        raise ValueError("Installed model adapter layout differs from fixed numeric contract")
    values = {key: torch.from_numpy(value) for key, value in tensors.items()}
    peft.set_peft_model_state_dict(policy, values)
    if any(
        not torch.equal(value, values[key])
        for key, value in peft.get_peft_model_state_dict(policy).items()
    ):
        raise ValueError("Adapter load did not preserve all submitted numeric tensors")
    policy.requires_grad_(False).to("cpu").eval()
    eos = policy.generation_config.eos_token_id
    eos = set(
        ([eos] if isinstance(eos, int) else list(eos or [])) + [processor.tokenizer.eos_token_id]
    )
    predictions = {}
    with torch.inference_mode():
        for row in image_rows:
            inputs = trainer._inputs(processor, row, spec, profile).to("cpu")
            tokens = policy.generate(
                **inputs,
                max_new_tokens=profile.max_completion_length,
                do_sample=False,
                use_cache=True,
            )
            tail = tokens[:, inputs["input_ids"].shape[-1] :]
            predictions[row["id"]] = {
                "response": processor.batch_decode(tail, skip_special_tokens=True)[0],
                "raw_response": processor.batch_decode(tail, skip_special_tokens=False)[0],
                "terminated": bool(tail.shape[-1] and int(tail[0, -1]) in eos),
                "completion_ids": tail[0].tolist(),
            }
    return predictions, {"adapter_sha256": digest, "processor_audit": processor_audit}


def verify_submission(
    contract_path=Path("/tests/contract.json"),
    artifacts=Path("/logs/artifacts/training"),
    verification=Path("/tests/data/verification.jsonl"),
    data_spec=Path("/tests/data_spec.json"),
    logs=Path("/logs/verifier"),
    model_directory=MODEL_PATH,
):
    """Fresh offline grader; every failure leaves zero reward and a typed error."""
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text("0\n")
    try:
        contract, profile = _read_contract(contract_path)
        if (
            _digest(data_spec) != contract["data_spec_sha256"]
            or _digest(verification) != contract["verification_sha256"]
        ):
            raise ValueError("Verifier-owned task or image manifest identity changed")
        spec = trainer.load_spec(data_spec)
        rows = trainer.load_classification(
            verification,
            "validation",
            spec["labels"],
            preprocessing_id=spec.get("preprocessing_id"),
            require_all_classes=False,
        )
        if _pixel_identity(rows) != contract["verification_pixels_sha256"]:
            raise ValueError("Verifier image pixels changed")
        if len(rows) > 4:
            raise ValueError("Integration grader permits at most four development images")
        # Copy the exact checked bytes into a verifier-owned path, closing races.
        with tempfile.TemporaryDirectory(prefix="trusted-vlm-") as temporary:
            path = Path(temporary).resolve() / "adapter_model.safetensors"
            path.write_bytes(_bounded_bytes(Path(artifacts) / path.name, ADAPTER_LIMIT))
            read_adapter(path)
            versions = _check_runtime()
            trainer.preflight("cpu")
            base_identity = _check_base(contract, model_directory)
            predictions, identity = predict_images(
                path,
                [{"id": r["id"], "image": r["image"]} for r in rows],
                spec,
                profile,
                model_directory,
            )
        metrics = trainer.classification_metrics(
            rows, predictions, spec["labels"], allow_reasoning=True
        )
        report = {
            "scope": "integration_only",
            "held_out_benchmark_gain_measured": False,
            "test_consumed": False,
            "metrics": metrics,
            "reward": metrics["mean_class_accuracy"],
            "model_source": base_identity,
            "runtime": versions,
            "source_sha256": _sources(),
            "contract_sha256": _digest(contract_path),
            "submitted_code_executed": False,
            "prediction_input_fields": ["id", "image"],
            **identity,
        }
        _json(logs / "predictions.json", predictions)
        _json(logs / "report.json", report)
        (logs / "reward.txt").write_text(f"{report['reward']:.17g}\n")
        return report
    except Exception as exc:
        report = {"reward": 0.0, "error": {"type": type(exc).__name__, "message": str(exc)}}
        _json(logs / "error.json", report)
        return report


def _invoke_training(train, data_spec, work, **options):
    """Release the entire base allocation at each native trainer process boundary."""
    command = [
        sys.executable,
        "-I",
        "-m",
        "vision_rl.vlm_local_training",
        "--train",
        str(train),
        "--data-spec",
        str(data_spec),
        "--output",
        str(work),
    ]
    profile = options.pop("profile")
    for name, value in {**options, **asdict(profile)}.items():
        if value is None or value is False:
            continue
        command.append("--" + name.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    # No shell and no inherited Python path overrides; the installed pinned
    # source owns training, native checkpoints, RNG and endpoint receipts.
    subprocess.run(command, check=True)
    return json.loads((work / "local_training.json").read_text())


def run_training(
    contract_path=Path("/app/contract.json"),
    output=Path("/logs/artifacts/training"),
    *,
    app=Path("/app"),
    model_directory=MODEL_PATH,
):
    """Call the existing trainer for RL and its declared supervised controls."""
    contract, profile = _read_contract(contract_path)
    _check_runtime()
    base_identity = _check_base(contract, model_directory)
    data_spec, train = app / "data_spec.json", app / "data/train.jsonl"
    initial = app / "initial/final_adapter"
    if (
        _digest(train) != contract["train_sha256"]
        or _digest(data_spec) != contract["data_spec_sha256"]
    ):
        raise ValueError("Frozen training inputs changed")
    spec = trainer.load_spec(data_spec)
    rows = trainer.load_classification(
        train,
        "train",
        spec["labels"],
        preprocessing_id=spec.get("preprocessing_id"),
        require_all_classes=False,
    )
    if len(rows) > 4 or _pixel_identity(rows) != contract["train_pixels_sha256"]:
        raise ValueError("Frozen TRAIN pixels or integration count changed")
    _, digest = read_adapter(initial / "adapter_model.safetensors")
    if (
        digest != contract["initial_weights_sha256"]
        or _digest(initial.parent / "local_training.json") != contract["initial_record_sha256"]
    ):
        raise ValueError("Frozen initialization changed")
    output.mkdir(parents=True, exist_ok=True)
    common = dict(
        model_directory=model_directory,
        device="cpu",
        profile=profile,
        train_only=True,
        max_steps=contract["control_step_cap"],
        max_active_seconds=contract["max_active_seconds"],
        learning_rate=contract["learning_rate"],
        beta=contract["beta"],
        seed=contract["seed"],
        save_steps=1,
        init_adapter=initial,
    )
    evidence = {
        "scope": "integration_only",
        "status": "running",
        "model_source": base_identity,
        "arms": {},
    }
    for name, mode in (("rl", "grpo"), ("sft", "sft")):
        work = output / name
        record = _invoke_training(
            train,
            data_spec,
            work,
            mode=mode,
            stop_after_steps=contract["max_steps"],
            stage_id=name + "-exposure",
            **common,
        )
        receipt = trainer.read_endpoint_receipt(work, expected_stage_id=name + "-exposure")
        if receipt["global_step"] != contract["max_steps"]:
            raise ValueError("Arm exhausted time before the exact equal-exposure boundary")
        evidence["arms"][name + "-exposure"] = {
            "receipt": receipt,
            "optimizer_audit": record["optimizer_audit"],
            "reward_audit": record["reward_audit"],
        }
        endpoint = output / "endpoints" / (name + "-exposure")
        endpoint.mkdir(parents=True)
        shutil.copytree(work / "final_adapter", endpoint / "final_adapter")
        shutil.copyfile(work / "local_training.json", endpoint / "local_training.json")
        _json(endpoint / "receipt.json", receipt)
        if (
            _digest(endpoint / "final_adapter/adapter_model.safetensors")
            != receipt["final_adapter"]["weights_sha256"]
        ):
            raise ValueError("Adapter changed while archiving immutable equal-exposure endpoint")
        if name == "rl":
            # Preserve the submission before any later control failure.
            shutil.copyfile(
                work / "final_adapter/adapter_model.safetensors",
                output / "adapter_model.safetensors",
            )
        _json(output / "training-evidence.json", evidence)
    target = evidence["arms"]["rl-exposure"]["receipt"]["committed_active_seconds"]
    receipt = evidence["arms"]["sft-exposure"]["receipt"]
    if receipt["committed_active_seconds"] < target:
        work = output / "sft"
        _invoke_training(
            train,
            data_spec,
            work,
            mode="sft",
            stop_after_active_seconds=target,
            stage_id="sft-time",
            resume_from_checkpoint=work / receipt["checkpoint"]["path"],
            **common,
        )
        receipt = trainer.read_endpoint_receipt(work, expected_stage_id="sft-time")
    if receipt["committed_active_seconds"] < target:
        raise ValueError("Supervised control exhausted its cap before matching RL active time")
    evidence["arms"]["sft-time"] = {"receipt": receipt, "target_seconds": target}
    evidence["status"] = "completed"
    evidence["held_out_benchmark_gain_measured"] = False
    _json(output / "training-evidence.json", evidence)
    return evidence


def _bundle_rows(rows, destination, split, filename):
    destination.mkdir(parents=True)
    packed = []
    for index, row in enumerate(rows):
        suffix = Path(row["image"]).suffix.lower()
        name = f"image-{index}{suffix}"
        shutil.copyfile(row["image"], destination / name)
        packed.append({"id": row["id"], "image": name, "label": row["label"], "split": split})
    path = destination / filename
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in packed))
    return path


def make_task(
    train: Path,
    verification: Path,
    data_spec: Path,
    initial_adapter: Path,
    model_directory: Path,
    output: Path,
    *,
    profile: dict | None = None,
    max_steps: int = 4,
    control_step_cap: int = 32,
    max_active_seconds: float = 14400,
    learning_rate: float = 1e-5,
    beta: float = 0.04,
    seed: int = 42,
    base_image: str = "vision-rl:vlm-local",
    timeout_sec: int = 86400,
) -> dict:
    """Package <=4 TRAIN and <=4 disjoint validation-derived integration images.

    The trusted local base is mounted read-only into both fresh environments.
    Packaging reads small provenance files but never loads or hashes base tensors;
    each actual runtime verifies all mounted bytes before model construction.
    """
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/:@-]*", base_image):
        raise ValueError("Invalid base image reference")
    if type(timeout_sec) is not int or not 1 <= timeout_sec <= 86400:
        raise ValueError("Timeout must be an integer in 1..86400")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Task output must be new")
    model_directory = Path(model_directory).resolve()
    if any(character in str(model_directory) for character in ("$", "\n", "\r")):
        raise ValueError("Model mount path cannot contain Compose interpolation")
    spec = trainer.load_spec(data_spec)
    sets = [
        trainer.load_classification(
            path,
            split,
            spec["labels"],
            preprocessing_id=spec.get("preprocessing_id"),
            require_all_classes=False,
        )
        for path, split in ((train, "train"), (verification, "validation"))
    ]
    if any(len(rows) > 4 for rows in sets):
        raise ValueError("Integration task permits at most four images per split")
    assert_disjoint(*sets)
    initial_adapter = Path(initial_adapter).resolve()
    tensors, initial_digest = read_adapter(initial_adapter / "adapter_model.safetensors")
    del tensors
    initial_config = trainer._validated_lora_config(
        json.loads((initial_adapter / "adapter_config.json").read_text())
    )
    expected_config = _fixed_lora_config()
    if any(
        initial_config[key] != expected_config[key]
        for key in ("r", "lora_alpha", "lora_dropout", "task_type", "bias")
    ) or set(initial_config["target_modules"]) != set(expected_config["target_modules"]):
        raise ValueError("Initialization LoRA scale or layout differs from the fixed profile")
    parent = json.loads((initial_adapter.parent / "local_training.json").read_text())
    if parent.get("final_adapter_sha256") != initial_digest or parent["control_identity"][
        "data_spec_sha256"
    ] != _digest(data_spec):
        raise ValueError("Initialization must match its completed export and exact task data spec")
    source = json.loads((model_directory / "model-source.json").read_text())
    if (
        source["model_repository"] != trainer.DEFAULT_MODEL
        or source["revision"] != trainer.MODEL_REVISIONS[trainer.DEFAULT_MODEL]
    ):
        raise ValueError("Base snapshot must use the pinned Qwen2-VL-2B revision")
    contract = {
        "schema_version": 1,
        "scope": "integration_only",
        "model": trainer.DEFAULT_MODEL,
        "model_revision": trainer.MODEL_REVISIONS[trainer.DEFAULT_MODEL],
        "model_manifest_sha256": _digest(model_directory / "model-source.json"),
        "model_marker_sha256": _digest(model_directory / ".pinned-download.json"),
        "data_spec_sha256": _digest(data_spec),
        "profile": profile or asdict(trainer.Profile()),
        "source_sha256": _sources(),
        "runtime": RUNTIME,
        "train_sha256": "0" * 64,
        "verification_sha256": "0" * 64,
        "train_pixels_sha256": _pixel_identity(sets[0]),
        "verification_pixels_sha256": _pixel_identity(sets[1]),
        "initial_weights_sha256": initial_digest,
        "initial_record_sha256": _digest(initial_adapter.parent / "local_training.json"),
        "max_steps": max_steps,
        "control_step_cap": control_step_cap,
        "max_active_seconds": max_active_seconds,
        "learning_rate": learning_rate,
        "beta": beta,
        "seed": seed,
    }
    selected_profile = _validate_contract(contract)
    trainer._adapter_identity(
        initial_adapter,
        {
            "model": contract["model"],
            "model_revision": contract["model_revision"],
            "processor_model": contract["model"],
            "processor_revision": contract["model_revision"],
            "profile": contract["profile"],
        },
        selected_profile,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vlm-task-", dir=output.parent) as temporary:
        staged = Path(temporary) / "task"
        environment, tests, solution = (
            staged / name for name in ("environment", "tests", "solution")
        )
        solution.mkdir(parents=True)
        train_path = _bundle_rows(sets[0], environment / "data", "train", "train.jsonl")
        verification_path = _bundle_rows(
            sets[1], tests / "data", "validation", "verification.jsonl"
        )
        contract.update(
            train_sha256=_digest(train_path), verification_sha256=_digest(verification_path)
        )
        for directory in (environment, tests):
            shutil.copyfile(data_spec, directory / "data_spec.json")
            _json(directory / "contract.json", contract)
        initial = environment / "initial/final_adapter"
        initial.mkdir(parents=True)
        shutil.copyfile(
            initial_adapter / "adapter_model.safetensors", initial / "adapter_model.safetensors"
        )
        shutil.copyfile(
            initial_adapter.parent / "local_training.json", initial.parent / "local_training.json"
        )
        # Trusted author constructs the only config that the training initializer sees.
        _json(
            initial / "adapter_config.json",
            {
                **_fixed_lora_config(),
                "peft_type": "LORA",
                "base_model_name_or_path": str(MODEL_PATH),
            },
        )
        (environment / "train.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\npython -I -m vision_rl.vlm_local_task_builder train\n"
        )
        (environment / "Dockerfile").write_text(
            f"FROM {base_image}\nCOPY data/ /app/data/\nCOPY initial/ /app/initial/\n"
            "COPY contract.json data_spec.json train.sh /app/\nRUN chmod +x /app/train.sh\n"
            'CMD ["sleep", "infinity"]\n'
        )
        # JSON is valid YAML and prevents shell/YAML injection in host mount paths.
        _json(
            environment / "docker-compose.yaml",
            {
                "services": {
                    "main": {
                        "network_mode": "none",
                        "volumes": [
                            {
                                "type": "bind",
                                "source": str(model_directory),
                                "target": str(MODEL_PATH),
                                "read_only": True,
                                "bind": {"create_host_path": False},
                            }
                        ],
                    }
                }
            },
        )
        (solution / "solve.sh").write_text("#!/bin/bash\nset -euo pipefail\nbash /app/train.sh\n")
        (tests / "test.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
            "echo 0 > /logs/verifier/reward.txt\n"
            "python -I -m vision_rl.vlm_local_task_builder verify\n"
        )
        for script in (environment / "train.sh", solution / "solve.sh", tests / "test.sh"):
            script.chmod(0o755)
        (staged / "task.toml").write_text(
            'schema_version = "1.4"\n\n[task]\nname = "vision-rl/local-vlm-integration"\n'
            'version = "0.1.0"\ndescription = "CPU Qwen2-VL GRPO numeric adapter integration"\n\n'
            '[metadata]\ncategory = "machine-learning"\npurpose = "integration-only"\n\n'
            f"[agent]\ntimeout_sec = {float(timeout_sec)}\n\n"
            '[verifier]\ntimeout_sec = 14400.0\nenvironment_mode = "separate"\n\n'
            "[environment]\ncpus = 2\nmemory_mb = 24576\nstorage_mb = 8192\n"
            'network_mode = "no-network"\n'
        )
        (staged / "instruction.md").write_text(
            "# Train a local VLM adapter\n\nRun `bash /app/train.sh`. The installed trainer runs "
            "four-or-fewer real GRPO updates and declared supervised controls from a common "
            "initialization. Training diagnostics use TRAIN only. Submit only "
            "`/logs/artifacts/training/adapter_model.safetensors`. The fresh offline verifier "
            "accepts fixed FP32 rank-eight Q/V LoRA tensors and owns all other inference inputs. "
            "Reward is greedy mean-class accuracy on at most four separate development images; "
            "this is an integration check, not a benchmark or frontier-model claim.\n\n"
            "The CPU runtime needs a 24 GiB Docker memory allocation. The existing 4 GiB demo VM "
            "cannot run this task. GPU execution is not implemented by this task.\n"
        )
        bundle = {
            "scope": "integration_only",
            "counts": {"train": len(sets[0]), "verification": len(sets[1])},
            "contract": contract,
            "test_consumed": False,
            "verification_in_build_context": False,
            "model_copied_into_image": False,
            "runtime_executed": False,
        }
        _json(staged / "bundle.json", bundle)
        staged.rename(output)
    return {
        **bundle,
        "task": str(output.resolve()),
        "agent": "vision_rl.harbor_agent:DetectionTrainingAgent",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    commands.add_parser("verify")
    build = commands.add_parser("build", help="Package a tiny development integration task")
    for name in (
        "train",
        "verification",
        "data-spec",
        "initial-adapter",
        "model-directory",
        "output",
    ):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument("--profile", type=Path, help="Trusted author's full Profile JSON")
    build.add_argument("--max-steps", type=int, default=4)
    build.add_argument("--control-step-cap", type=int, default=32)
    build.add_argument("--max-active-seconds", type=float, default=14400)
    build.add_argument("--learning-rate", type=float, default=1e-5)
    build.add_argument("--beta", type=float, default=0.04)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--base-image", default="vision-rl:vlm-local")
    build.add_argument("--timeout-sec", type=int, default=86400)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "build":
        if args["profile"] is not None:
            args["profile"] = json.loads(_bounded_bytes(args["profile"].resolve(), JSON_LIMIT))
        result = make_task(**args)
    else:
        result = run_training() if command == "train" else verify_submission()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
