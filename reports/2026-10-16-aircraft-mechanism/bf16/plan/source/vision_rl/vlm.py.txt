"""Optional CUDA GRPO training and local inference for vision specialists.

Importing this module does not import PyTorch or download model weights. Install
``requirements-vlm.txt`` on a CUDA host, then run ``python -m vision_rl.vlm
--preflight`` before scheduling a training job. This is model-weight training;
Harbor separately runs and grades the agent that conducts experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from .data import assert_disjoint, load_manifest
from .evaluation import parse_answer, prompt_text

SUPPORTED_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl", "gemma3", "llava_next", "smolvlm"})
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_DEPENDENCIES = ("torch", "torchvision", "trl", "transformers", "peft", "datasets", "accelerate")


class VLMPreflightError(RuntimeError):
    """A local dependency or hardware check failed before model downloads."""


def preflight() -> dict[str, Any]:
    """Check local CUDA and dependency imports without contacting a model hub."""
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise VLMPreflightError(
            "This starter supports one training process on one CUDA GPU. "
            "Run with python directly, without a distributed launcher."
        )
    try:
        torch = import_module("torch")
    except (ImportError, OSError) as exc:
        raise VLMPreflightError(
            "VLM training requires optional dependencies: install requirements-vlm.txt "
            "on a Linux host with an NVIDIA GPU and compatible CUDA drivers."
        ) from exc
    if not torch.version.cuda or not torch.cuda.is_available():
        raise VLMPreflightError(
            "An NVIDIA CUDA GPU is required for VLM GRPO and local VLM inference. "
            "CUDA is unavailable; no model weights have been downloaded. "
            "Use the CPU demo here or run this command on a CUDA host."
        )
    versions = {}
    try:
        for package in _DEPENDENCIES:
            versions[package] = metadata.version(package)
            import_module(package)
    except (ImportError, OSError, RuntimeError, metadata.PackageNotFoundError) as exc:
        raise VLMPreflightError(
            "The optional VLM dependencies are missing or incompatible. Install the "
            "pinned requirements-vlm.txt stack and rerun --preflight."
        ) from exc
    if versions["trl"].split("+")[0] != "1.16.0":
        raise VLMPreflightError(
            "This integration targets TRL 1.16.0; install requirements-vlm.txt. "
            f"Found TRL {versions['trl']}. No model weights have been downloaded."
        )
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": f"cuda:{device}",
        "gpu": properties.name,
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
        "cuda": torch.version.cuda,
        "precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "python": sys.version.split()[0],
        "package_versions": versions,
    }


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or len(completion) != 1:
        return ""
    message = completion[0]
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        for block in content
    ):
        return "".join(block["text"] for block in content)
    return ""


def answer_reward(
    completions: list[Any], answer: list[str], choices: list[list[str]], **_: Any
) -> list[float]:
    """Return binary exact-answer reward; malformed or verbose answers earn zero.

    TRL repeats dataset columns for every sampled completion. Length mismatch is
    an integration error, never a reason to silently drop samples with ``zip``.
    """
    if not (len(completions) == len(answer) == len(choices)):
        raise ValueError("Completion, answer, and choices batch lengths must match")
    return [
        float(parse_answer(_completion_text(completion), options) == expected)
        for completion, expected, options in zip(completions, answer, choices, strict=True)
    ]


def _training_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep labels in reward columns, never in the model's question text."""
    return [
        {
            "id": row["id"],
            "prompt": [{"role": "user", "content": prompt_text(row)}],
            "image": str(row["image"]),
            "answer": row["answer"],
            "choices": row["choices"],
            "category": row["category"],
        }
        for row in rows
    ]


def _as_dataset(rows: list[dict[str, Any]]) -> Any:
    datasets = import_module("datasets")
    return datasets.Dataset.from_list(_training_records(rows)).cast_column(
        "image", datasets.Image(mode="RGB")
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_identity(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    images = {row["id"]: _file_sha256(Path(row["image"])) for row in rows}
    return {
        "manifest_sha256": _file_sha256(path),
        "image_sha256": hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest(),
        "count": len(rows),
    }


def _write_metadata(output: Path, record: dict[str, Any]) -> None:
    temporary = output / ".training_metadata.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "training_metadata.json")


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validated_lora_config(config: dict[str, Any]) -> dict[str, Any]:
    """Bound adapter allocations before PEFT constructs any trainable layers.

    Only the q/v-projection LoRA profile produced by ``train_vlm`` is supported.
    Never hand arbitrary artifact configuration through to PEFT: features such
    as layer replication can allocate far more memory than the weights file.
    """
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("adapter must use the supported causal LoRA profile")
    rank = config.get("r")
    if type(rank) is not int or not 1 <= rank <= 256:
        raise ValueError("adapter LoRA rank must be between 1 and 256")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(target not in ("q_proj", "v_proj") for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("adapter target_modules must contain only q_proj and/or v_proj")
    for field in (
        "rank_pattern",
        "alpha_pattern",
        "layer_replication",
        "modules_to_save",
        "trainable_token_indices",
        "target_parameters",
        "layers_pattern",
        "layers_to_transform",
        "use_dora",
        "use_rslora",
        "alora_invocation_tokens",
        "fan_in_fan_out",
        "lora_bias",
    ):
        if config.get(field):
            raise ValueError(f"adapter {field} is outside the supported LoRA profile")
    if (
        config.get("bias", "none") != "none"
        or type(config.get("init_lora_weights", True)) is not bool
    ):
        raise ValueError("adapter uses unsupported bias or initialization settings")
    alpha, dropout = config.get("lora_alpha"), config.get("lora_dropout", 0.0)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 < alpha <= 4096
        or isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise ValueError("adapter LoRA scale or dropout is invalid")
    return {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
        "bias": "none",
        # All weights will be loaded from safetensors, so no initializer is needed.
        "init_lora_weights": False,
    }


def _check_model(model: str, revision: str) -> tuple[Any, str]:
    transformers = import_module("transformers")
    config = transformers.AutoConfig.from_pretrained(
        model, revision=revision, trust_remote_code=False
    )
    if config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported VLM architecture {config.model_type!r}. Supported model types: "
            + ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        )
    return config, getattr(config, "_commit_hash", None) or revision


def train_vlm(
    train: Path,
    validation: Path,
    output: Path,
    model: str,
    max_steps: int = 100,
    seed: int = 42,
    *,
    num_generations: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    max_completion_length: int = 64,
    save_steps: int = 25,
    resume_from_checkpoint: Path | str | None = None,
    model_revision: str = "main",
) -> dict[str, Any]:
    """Train a LoRA VLM policy with real GRPO; validation never updates weights.

    Test manifests are rejected. Checkpoints retain optimizer state for resume;
    ``final_adapter`` contains the compact adapter and processor for inference.
    Passing a nonempty output directory requires a compatible local checkpoint.
    """
    for name, value in (
        ("max_steps", max_steps),
        ("lora_rank", lora_rank),
        ("max_completion_length", max_completion_length),
        ("save_steps", save_steps),
    ):
        _positive_integer(name, value)
    _positive_integer("num_generations", num_generations, minimum=2)
    _positive_integer("seed", seed, minimum=0)
    if lora_rank > 256:
        raise ValueError("lora_rank cannot exceed the supported inference limit of 256")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a Hugging Face model ID or local model directory")
    train, validation, output = Path(train), Path(validation), Path(output)
    train_rows = load_manifest(train, expected_split="train")
    validation_rows = load_manifest(validation, expected_split="validation")
    if not train_rows or not validation_rows:
        raise ValueError("Training and validation manifests must both be nonempty")
    assert_disjoint(train_rows, validation_rows)
    identity = {
        "train": _dataset_identity(train, train_rows),
        "validation": _dataset_identity(validation, validation_rows),
    }
    previous = None
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = Path(resume_from_checkpoint).resolve()
        if (
            not checkpoint.is_relative_to(output.resolve())
            or not (checkpoint / "trainer_state.json").is_file()
        ):
            raise ValueError("Resume requires a trainer checkpoint inside the output directory")
        previous = json.loads((output / "training_metadata.json").read_text(encoding="utf-8"))
        if previous["model"] != model or previous["datasets"] != identity:
            raise ValueError("Resume model and dataset fingerprints must match the original run")
        for name, value in {
            "seed": seed,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "max_completion_length": max_completion_length,
        }.items():
            if previous["hyperparameters"][name] != value:
                raise ValueError(f"Resume must use the original {name}")
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output is not empty; use a new directory or --resume-from-checkpoint")

    environment = preflight()  # No model-hub access occurs before this line succeeds.
    if previous:
        model_revision = previous["model_revision"]
    config, resolved_revision = _check_model(model, model_revision)
    transformers = import_module("transformers")
    trl = import_module("trl")
    peft = import_module("peft")
    processor = transformers.AutoProcessor.from_pretrained(
        model,
        revision=resolved_revision,
        trust_remote_code=False,
        padding_side="left",
    )
    if not hasattr(processor, "image_processor"):
        raise ValueError("The selected model does not provide a vision-language image processor")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "algorithm": "GRPO",
        "adapter": "LoRA",
        "status": "running",
        "model": model,
        "model_type": config.model_type,
        "model_revision": resolved_revision,
        "created_at": previous["created_at"]
        if previous
        else datetime.now(timezone.utc).isoformat(),
        "datasets": identity,
        "environment": environment,
        "hyperparameters": {
            "max_steps": max_steps,
            "seed": seed,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "max_completion_length": max_completion_length,
            "save_steps": save_steps,
        },
        "reward": "exact_answer_accuracy",
        "evaluation": "validation sampled reward; run independent held-out benchmarking separately",
        "resumed_from": str(checkpoint) if checkpoint else None,
    }
    _write_metadata(output, record)
    try:
        args = trl.GRPOConfig(
            output_dir=str(output),
            max_steps=max_steps,
            seed=seed,
            data_seed=seed,
            learning_rate=learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=num_generations,
            num_generations=num_generations,
            per_device_eval_batch_size=1,
            num_generations_eval=1,
            max_completion_length=max_completion_length,
            bf16=environment["precision"] == "bfloat16",
            fp16=environment["precision"] == "float16",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps",
            eval_steps=save_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            logging_steps=1,
            report_to="none",
            push_to_hub=False,
            remove_unused_columns=False,
            use_vllm=False,
            beta=0.0,
            model_init_kwargs={
                "dtype": environment["precision"],
                "revision": resolved_revision,
                "attn_implementation": "sdpa",
                "trust_remote_code": False,
            },
        )
        trainer = trl.GRPOTrainer(
            model=model,
            args=args,
            processing_class=processor,
            reward_funcs=answer_reward,
            train_dataset=_as_dataset(train_rows),
            eval_dataset=_as_dataset(validation_rows),
            peft_config=peft.LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        validation_metrics = trainer.evaluate()
        adapter = output / "final_adapter"
        trainer.save_model(str(adapter))
        processor.save_pretrained(str(adapter))
        trainer.save_state()
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            train_metrics=result.metrics,
            validation_metrics=validation_metrics,
            final_adapter=str(adapter),
        )
        _write_metadata(output, record)
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
        )
        _write_metadata(output, record)
        raise
    return record


def predict_vlm(
    rows: list[dict[str, Any]],
    model: str,
    adapter: Path | str | None = None,
    max_new_tokens: int = 64,
    *,
    model_revision: str = "main",
) -> dict[str, str]:
    """Generate deterministic answers from a base VLM plus optional local adapter.

    ``rows`` must come from ``load_manifest``. Labels and split names are never
    passed to the model. This function does not calculate or select test scores.
    """
    _positive_integer("max_new_tokens", max_new_tokens)
    if not rows:
        return {}
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Prediction rows must have unique IDs")
    adapter_path = Path(adapter) if adapter is not None else None
    adapter_parameters = None
    if adapter_path is not None:
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.is_file():
            raise ValueError("adapter must be a local PEFT adapter directory")
        if adapter_config.is_symlink() or adapter_config.stat().st_size > 65536:
            raise ValueError("adapter configuration must be a regular JSON file under 64 KiB")
        adapter_info = json.loads(adapter_config.read_text(encoding="utf-8"))
        if not isinstance(adapter_info, dict):
            raise ValueError("adapter configuration must be a JSON object")
        if adapter_info.get("base_model_name_or_path") != model:
            raise ValueError("The adapter's base model does not match the selected model")
        safe_weights = adapter_path / "adapter_model.safetensors"
        if not safe_weights.is_file() or safe_weights.is_symlink():
            raise ValueError(
                "adapter requires regular adapter_model.safetensors weights; pickle is not accepted"
            )
        if safe_weights.stat().st_size > 2 * 1024**3:
            raise ValueError("adapter safetensors weights exceed the 2 GiB limit")
        adapter_parameters = _validated_lora_config(adapter_info)
        run_metadata = adapter_path.parent / "training_metadata.json"
        if run_metadata.is_file():
            model_revision = json.loads(run_metadata.read_text(encoding="utf-8"))["model_revision"]
    environment = preflight()
    _, resolved_revision = _check_model(model, model_revision)
    torch = import_module("torch")
    transformers = import_module("transformers")
    pil_image = import_module("PIL.Image")
    # Training never changes the processor. Use the fixed base processor so an
    # untrusted adapter cannot replace the evaluation chat template.
    processor = transformers.AutoProcessor.from_pretrained(
        model,
        revision=resolved_revision,
        trust_remote_code=False,
    )
    policy = transformers.AutoModelForImageTextToText.from_pretrained(
        model,
        revision=resolved_revision,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=getattr(torch, environment["precision"]),
        attn_implementation="sdpa",
    )
    if adapter_path:
        peft = import_module("peft")
        policy = peft.PeftModel.from_pretrained(
            policy, str(adapter_path), config=peft.LoraConfig(**adapter_parameters)
        )
    policy = policy.to(environment["device"]).eval()
    predictions = {}
    for row in rows:
        with pil_image.open(row["image"]) as source:
            image = source.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text(row)},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=[prompt], return_tensors="pt").to(
            environment["device"]
        )
        with torch.inference_mode():
            tokens = policy.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completion = tokens[:, inputs["input_ids"].shape[1] :]
        predictions[row["id"]] = processor.batch_decode(completion, skip_special_tokens=True)[0]
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight", action="store_true", help="Check CUDA/dependencies without downloads"
    )
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if not args.preflight and not all((args.train, args.validation, args.output)):
        parser.error("--train, --validation, and --output are required unless --preflight is used")
    try:
        if args.preflight:
            result = preflight()
        else:
            values = vars(args)
            values.pop("preflight")
            result = train_vlm(**values)
        print(json.dumps(result, indent=2))
        return 0
    except (VLMPreflightError, ValueError, OSError) as exc:
        print(f"vision-rl VLM: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
