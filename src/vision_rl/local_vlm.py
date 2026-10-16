"""Journaled local Qwen vision inference with an explicit pinned MLX runtime."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import os
import threading
import time
from pathlib import Path

from PIL import Image

from .frontier import _append, _digest, _now, _prepare, _public

REPOSITORY = "mlx-community/Qwen3.8-27B-8bit"
REVISION = "815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9"
PACKAGES = (
    "mlx-vlm", "mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers",
    "pillow", "numpy", "safetensors",
)
GENERATION = {
    "max_tokens": 16384, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
    "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
    "enable_thinking": True,
}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions["mlx-vlm"] != "0.6.8":
        raise ValueError("Use the pinned mlx-vlm0.6.8 isolated runtime")
    return versions


def verify_snapshot(directory, integrity):
    """Rehash the sole local snapshot before loading, including processor and tokenizer."""
    directory = Path(directory).resolve()
    record = json.loads(Path(integrity).read_text())
    if record.get("model_repository") != REPOSITORY or record.get("revision") != REVISION:
        raise ValueError("Local model is not the pinned Qwen8bit snapshot")
    if record.get("quantization") != {"bits": 8, "mode": "affine", "group_size": 64}:
        raise ValueError("Unexpected local quantization")
    files = {}
    for entry in record["files"]:
        original = directory / entry["path"]
        path = original.resolve()
        if not path.is_relative_to(directory) or original.is_symlink():
            raise ValueError("Model file escapes the pinned snapshot")
        if path.stat().st_size != entry["bytes"] or sha_file(path) != entry["sha256"]:
            raise ValueError(f"Local model integrity mismatch: {entry['path']}")
        files[entry["path"]] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "preprocessor_config.json", "processor_config.json"}
    if not required.issubset(files) or len([p for p in files if p.endswith(".safetensors")]) != 6:
        raise ValueError("Incomplete model, tokenizer, or processor snapshot")
    return {"repository": REPOSITORY, "revision": REVISION,
            "quantization": record["quantization"], "files": files}


def final_answer(raw_text, finish_reason):
    """Separate the explicit reasoning close marker without repairing the answer."""
    result = {"response": "", "finish_reason": finish_reason}
    if finish_reason != "stop":
        result["error"] = "Local generation did not finish normally"
    elif not isinstance(raw_text, str) or raw_text.count("</think>") != 1:
        result["error"] = "Missing or ambiguous explicit reasoning close marker"
    else:
        result["response"] = raw_text.split("</think>", 1)[1].strip()
        if not result["response"]:
            result["error"] = "Local model returned no final answer"
    return result


class MLXBackend:
    """Import and allocate MLX only when the caller actually starts local inference."""

    def __init__(self, directory, config):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx.core as mx
        from mlx_vlm import load

        self.mx, self.config = mx, config
        previous_memory_limit = mx.set_memory_limit(config["memory_limit_bytes"])
        previous_cache_limit = mx.set_cache_limit(config["cache_limit_bytes"])
        start = time.monotonic()
        self.model, self.processor = load(
            str(directory), strict=True, trust_remote_code=False, local_files_only=True,
        )
        self.load_info = {
            "load_seconds": time.monotonic() - start, "device": mx.device_info(),
            "active_memory_bytes_after_load": mx.get_active_memory(),
            "peak_memory_bytes_after_load": mx.get_peak_memory(),
            "memory_policy": {
                "requested_limit_bytes": config["memory_limit_bytes"],
                "reported_limit_bytes": mx.get_memory_limit(),
                "previous_limit_bytes": previous_memory_limit,
                "hard_allocation_cap": False,
                "semantics": "MLX graph-evaluation guideline; not a process or RAM cap",
                "requested_cache_limit_bytes": config["cache_limit_bytes"],
                "previous_cache_limit_bytes": previous_cache_limit,
                "cache_semantics": "Free cache above this limit is reclaimed on next allocation",
            },
        }
        self.telemetry_stop = threading.Event()
        self.telemetry_thread = threading.Thread(target=self._telemetry, daemon=True)
        self.telemetry_thread.start()

    def _telemetry(self):
        with Path(self.config["memory_telemetry_path"]).open("a") as handle:
            while not self.telemetry_stop.is_set():
                _append(handle, {
                    "created_at": _now(), "active_bytes": self.mx.get_active_memory(),
                    "cache_bytes": self.mx.get_cache_memory(),
                    "peak_bytes_since_last_image_reset": self.mx.get_peak_memory(),
                })
                self.telemetry_stop.wait(1.0)

    def close(self):
        self.telemetry_stop.set()
        self.telemetry_thread.join(timeout=2)

    def infer(self, row, seed):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        mx = self.mx
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.random.seed(seed)
        prompt = apply_chat_template(
            self.processor, self.model.config, row["prompt"], num_images=1,
            enable_thinking=True, reasoning_effort="xhigh", preserve_thinking=True,
        )
        if "Reasoning effort is set to xhigh." not in prompt or not prompt.endswith("<think>\n"):
            raise ValueError("Pinned template did not enable xhigh thinking")
        # MLX's path loader applies EXIF transposition. Native COCO boxes and the
        # shared pixel hash use encoded pixel order, so pass that exact RGB image.
        with Image.open(row["image_path"]) as source, source.convert("RGB") as rgb:
            inputs = prepare_inputs(
                self.processor, images=[rgb], prompts=prompt,
                image_token_index=getattr(self.model.config, "image_token_index", None),
                resize_shape=None, add_special_tokens=True,
            )
        metadata = {
            "formatted_prompt": prompt, "formatted_prompt_sha256": _digest(prompt),
            "input_ids_shape": list(inputs["input_ids"].shape),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "implicit_external_resize": False,
            "exif_orientation_transform": False,
        }
        extra = {key: value for key, value in inputs.items()
                 if key not in {"input_ids", "pixel_values", "attention_mask"}}
        result = generate(
            self.model, self.processor, prompt, verbose=False,
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            mask=inputs.get("attention_mask"), seed=seed, **extra,
            **self.config["generation"],
        )
        # The complete generated text and native usage are retained. A vocabulary-sized
        # last-token logprob array is not part of this detection protocol.
        native = {
            field.name: getattr(result, field.name) for field in dataclasses.fields(result)
            if field.name != "logprobs"
        }
        native["logprobs_retained"] = False
        native["peak_memory_unit"] = "decimal GB, as returned by mlx-vlm"
        return {"native_response": native, "preprocessing": metadata,
                "peak_memory_bytes": mx.get_peak_memory(),
                "active_memory_bytes_after_inference": mx.get_active_memory(),
                "cache_memory_bytes_after_inference": mx.get_cache_memory()}


def run_local(
    rows, output, *, model_directory, integrity, phase, seed=1729,
    memory_limit_bytes=32 * 1024**3, resume=False, backend_factory=None,
):
    """One image at a time; completed and ambiguous attempts are never repeated."""
    if phase not in {"train_probe", "validation"}:
        raise ValueError("Local development runner cannot access final-test images")
    expected_parent = "train" if phase == "train_probe" else "valid"
    if not rows or any(Path(row["image"]).parent.name != expected_parent for row in rows):
        raise ValueError("Every local image must belong to its declared native split")
    if phase == "train_probe" and len(rows) != 2:
        raise ValueError("Initial local resource probe requires exactly two training images")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned32-bit integer")
    if type(memory_limit_bytes) is not int or not 1024**3 <= memory_limit_bytes <= 64 * 1024**3:
        raise ValueError("Memory limit must be explicit and between1GiB and64GiB")
    prepared = [_prepare(row) for row in rows]
    if len({row["id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate local request IDs")
    model = verify_snapshot(model_directory, integrity)
    config = {
        "phase": phase, "seed": seed, "generation": GENERATION,
        "reasoning_effort": "xhigh", "enable_thinking": True, "preserve_thinking": True,
        "memory_limit_bytes": memory_limit_bytes, "cache_limit_bytes": 512 * 1024**2,
        "runtime_versions": runtime_versions(), "coordinate_format": "xyxy_1000",
        "concurrency": 1, "fewshot_training_images": 0, "resize_shape": None,
        "source_orientation": "encoded_pixel_order_without_exif_transposition",
    }
    identity = {
        "schema_version": 1, "kind": "local_mlx_vision", "model": model,
        "config": config, "rows": [_public(row) for row in prepared],
        "adapter_sha256": sha_file(__file__),
    }
    identity_hash = _digest(identity)
    output = Path(output)
    if resume:
        if not output.is_dir():
            raise ValueError("Cannot resume a missing local run")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = output / "run.json"
        if resume:
            recorded = json.loads(run_path.read_text())
            if recorded["identity"] != identity or recorded["identity_sha256"] != identity_hash:
                raise ValueError("Local resume identity differs")
        else:
            with run_path.open("x") as handle:
                json.dump({"created_at": _now(), "identity": identity,
                           "identity_sha256": identity_hash}, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
        return _run_locked(prepared, output, config, identity_hash,
                           model_directory, backend_factory or MLXBackend)


def _run_locked(prepared, output, config, identity_hash, model_directory, backend_factory):
    journal = output / "attempts.jsonl"
    reserved, completed = {}, {}
    expected = {row["id"]: _public(row) for row in prepared}
    if journal.exists():
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            identifier = event["id"]
            if event["identity_sha256"] != identity_hash or identifier not in expected:
                raise ValueError("Local journal identity differs")
            if event["event"] == "reserved":
                if identifier in reserved or event["request"] != expected[identifier]:
                    raise ValueError("Invalid or duplicate local reservation")
                reserved[identifier] = event
            elif event["event"] == "completed":
                if identifier not in reserved or identifier in completed:
                    raise ValueError("Invalid local completion")
                completed[identifier] = event
            else:
                raise ValueError("Unknown local journal event")
    pending = [row for row in prepared if row["id"] not in reserved]
    stopped = any(event.get("hard_stop") for event in completed.values())
    backend = None
    if pending and not stopped:
        with (output / "runtime.jsonl").open("a") as runtime_journal:
            _append(runtime_journal, {"event": "load_started", "created_at": _now(),
                                     "identity_sha256": identity_hash})
            try:
                backend = backend_factory(model_directory, {
                    **config, "memory_telemetry_path": str(output / "mlx-memory.jsonl"),
                })
            except (RuntimeError, ValueError, TypeError, OSError, MemoryError) as exc:
                failure = {"event": "load_failed", "created_at": _now(),
                           "identity_sha256": identity_hash,
                           "exception_type": type(exc).__name__, "exception_message": str(exc)}
                _append(runtime_journal, failure)
                (output / "runtime.json").write_text(json.dumps(failure, indent=2) + "\n")
                raise
            _append(runtime_journal, {"event": "load_completed", "created_at": _now(),
                                     "identity_sha256": identity_hash, **backend.load_info})
            (output / "runtime.json").write_text(json.dumps(backend.load_info, indent=2) + "\n")
    with journal.open("a") as handle:
        for row in pending:
            if stopped:
                break
            item_seed = int(_digest([config["seed"], row["id"]])[:8], 16)
            reservation = {"event": "reserved", "id": row["id"], "created_at": _now(),
                           "identity_sha256": identity_hash, "request": _public(row),
                           "seed": item_seed, "attempt": len(reserved) + 1}
            _append(handle, reservation)
            reserved[row["id"]] = reservation
            start = time.monotonic()
            try:
                if _public(_prepare({"id": row["id"], "image": row["image_path"],
                                     "prompt": row["prompt"]})) != expected[row["id"]]:
                    raise ValueError("Image changed after the local run was recorded")
                native = backend.infer(row, item_seed)
                result = native["native_response"]
                prediction = final_answer(result["text"], result.get("finish_reason"))
                prediction["usage"] = {key: result.get(key) for key in (
                    "prompt_tokens", "generation_tokens", "total_tokens",
                    "prompt_tps", "generation_tps", "peak_memory",
                )}
            except (RuntimeError, ValueError, KeyError, TypeError, OSError, MemoryError) as exc:
                native = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
                prediction = {"response": "", "error": type(exc).__name__}
                stopped = True
            prediction["latency_seconds"] = time.monotonic() - start
            event = {"event": "completed", "id": row["id"], "created_at": _now(),
                     "identity_sha256": identity_hash, "prediction": prediction,
                     "hard_stop": stopped, **native}
            _append(handle, event)
            completed[row["id"]] = event
    predictions = {}
    for row in prepared:
        identifier = row["id"]
        if identifier in completed:
            prediction = completed[identifier]["prediction"]
        else:
            error = (
                "unknown_after_interruption"
                if identifier in reserved else "not_requested_hard_stop"
            )
            prediction = {"response": "", "error": error}
        predictions[identifier] = prediction
    (output / "predictions.jsonl").write_text("".join(
        json.dumps({"id": identifier, **prediction}, allow_nan=False) + "\n"
        for identifier, prediction in predictions.items()
    ))
    if backend is not None and hasattr(backend, "close"):
        backend.close()
    return predictions
