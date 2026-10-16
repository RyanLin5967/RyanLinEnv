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
from .local_vlm_execution import missing_error, plan_chunk

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
INSTRUCT_GENERATION = {
    **GENERATION, "temperature": 0.7, "top_p": 0.8,
    "presence_penalty": 1.5, "enable_thinking": False,
}


def native_profile(profile="xhigh"):
    """Return an explicit author-supported profile; never mutate the frozen defaults."""
    if profile not in {"xhigh", "instruct"}:
        raise ValueError("Native profile must be xhigh or instruct")
    thinking = profile == "xhigh"
    return {
        "profile": profile, "native_profile": f"qwen3.8-native-{profile}-v1",
        "generation": dict(GENERATION if thinking else INSTRUCT_GENERATION),
        "enable_thinking": thinking, "reasoning_effort": "xhigh" if thinking else None,
        "preserve_thinking": True,
        "assistant_prefill": "<think>\n" if thinking else "<think>\n\n</think>\n\n",
    }


def model_image(path, preprocessing_id=None):
    """Return owned RGB pixels, using the exact shared dataset transform when declared."""
    if preprocessing_id is not None:
        from .aircraft_data import PREPROCESSING_ID, load_aircraft_rgb

        if preprocessing_id != PREPROCESSING_ID:
            raise ValueError("Unknown local image preprocessing contract")
        return load_aircraft_rgb(path)
    with Image.open(path) as source:
        return source.convert("RGB")


def prepare_local(row, preprocessing_id=None):
    prepared = _prepare(row)
    if preprocessing_id is not None:
        from .aircraft_data import rgb_pixel_sha256

        with model_image(row["image"], preprocessing_id) as rgb:
            prepared.update(
                source_pixel_sha256=prepared["pixel_sha256"],
                source_file_sha256=sha_file(row["image"]),
                source_width=prepared["width"], source_height=prepared["height"],
                pixel_sha256=rgb_pixel_sha256(rgb), width=rgb.width, height=rgb.height,
                preprocessing_id=preprocessing_id,
                pixel_hash_contract="aircraft_data.rgb_pixel_sha256",
            )
    return prepared


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


def final_answer(raw_text, finish_reason, *, profile="xhigh"):
    """Separate the explicit reasoning close marker without repairing the answer."""
    mode = native_profile(profile)
    result = {"response": "", "finish_reason": finish_reason}
    if finish_reason != "stop":
        result["error"] = "Local generation did not finish normally"
    elif not mode["enable_thinking"]:
        if not isinstance(raw_text, str) or not raw_text.strip():
            result["error"] = "Local model returned no final answer"
        else:
            # The native template already closed its empty thinking block.
            # Preserve all generated prose; the shared task parser decides validity.
            result["response"] = raw_text.strip()
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
        mode = native_profile(self.config.get("profile", "xhigh"))
        template_kwargs = {"enable_thinking": mode["enable_thinking"],
                           "preserve_thinking": True}
        if mode["reasoning_effort"] is not None:
            template_kwargs["reasoning_effort"] = mode["reasoning_effort"]
        prompt = apply_chat_template(
            self.processor, self.model.config, row["prompt"], num_images=1,
            **template_kwargs,
        )
        if not prompt.endswith("<|im_start|>assistant\n" + mode["assistant_prefill"]):
            raise ValueError("Pinned template does not match the native profile prefill")
        if mode["enable_thinking"] != ("Reasoning effort is set to xhigh." in prompt):
            raise ValueError("Pinned template does not match the native reasoning profile")
        # MLX's path loader applies EXIF transposition. Native COCO boxes and the
        # shared pixel hash use encoded pixel order, so pass that exact RGB image.
        with model_image(row["image_path"], self.config.get("preprocessing_id")) as rgb:
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
            "preprocessing_id": self.config.get("preprocessing_id"),
            "native_profile": mode["native_profile"], "template_kwargs": template_kwargs,
            "assistant_prefill": mode["assistant_prefill"],
        }
        extra = {key: value for key, value in inputs.items()
                 if key not in {"input_ids", "pixel_values", "attention_mask"}}
        generation_kwargs = {
            "verbose": False, "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"], "mask": inputs.get("attention_mask"),
            "seed": seed, **extra, **self.config["generation"],
        }
        if self.config.get("stream_progress"):
            from .local_vlm_execution import generate_with_progress

            with Path(self.config["generation_progress_path"]).open("a") as handle:
                def progress(observation):
                    _append(handle, {
                        "event": "generation_progress", "created_at": _now(),
                        "id": row["id"], "seed": seed,
                        "identity_sha256": self.config["identity_sha256"], **observation,
                    })

                result = generate_with_progress(
                    self.model, self.processor, prompt, progress=progress, **generation_kwargs,
                )
        else:
            result = generate(self.model, self.processor, prompt, **generation_kwargs)
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
    image_root=None, task_contract=None, preprocessing_id=None,
    stop_after_completions=None, stream_progress=False, profile="xhigh",
):
    """One image at a time; completed and ambiguous attempts are never repeated."""
    if phase not in {"train_probe", "validation"}:
        raise ValueError("Local development runner cannot access final-test images")
    if (Path(output) / "cancelled.json").exists():
        raise ValueError("Cancelled local runs cannot be resumed or retried")
    mode = native_profile(profile)
    if type(stream_progress) is not bool:
        raise ValueError("stream_progress must be explicit boolean")
    # Validate before model verification, loading, or output mutation. The limit
    # controls an invocation boundary; it never changes the fixed study scope.
    plan_chunk([str(row["id"]) for row in rows], {}, {}, stop_after_completions)
    expected_parent = "train" if phase == "train_probe" else "valid"
    if image_root is None:
        if not rows or any(Path(row["image"]).parent.name != expected_parent for row in rows):
            raise ValueError("Every local image must belong to its declared native split")
    else:
        root = Path(image_root).resolve(strict=True)
        expected_split = "train" if phase == "train_probe" else "validation"
        if not rows or any(
            row.get("split") != expected_split or not Path(row["image"]).resolve(
                strict=True).is_relative_to(root) for row in rows
        ):
            raise ValueError("Generic local images must remain within their declared split root")
    if task_contract is not None:
        if not isinstance(task_contract, dict) or not task_contract.get("task"):
            raise ValueError("An explicit task contract must declare its task")
        # Reject non-JSON identities before any model load or output mutation.
        task_contract = json.loads(json.dumps(task_contract, allow_nan=False))
    expected_probe = 2
    if task_contract and task_contract.get("task") == "classification":
        expected_probe = task_contract.get("probe_images", 2)
    if expected_probe not in {2, 16} or (phase == "train_probe" and len(rows) != expected_probe):
        raise ValueError("Local training probe requires its explicitly declared2or16 images")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned32-bit integer")
    if type(memory_limit_bytes) is not int or not 1024**3 <= memory_limit_bytes <= 64 * 1024**3:
        raise ValueError("Memory limit must be explicit and between1GiB and64GiB")
    prepared = [prepare_local(row, preprocessing_id) for row in rows]
    if len({row["id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate local request IDs")
    model = verify_snapshot(model_directory, integrity)
    from . import local_vlm_execution

    config = {
        "phase": phase, "seed": seed, **mode,
        "memory_limit_bytes": memory_limit_bytes, "cache_limit_bytes": 512 * 1024**2,
        "runtime_versions": runtime_versions(), "coordinate_format": "xyxy_1000",
        "concurrency": 1, "fewshot_training_images": 0, "resize_shape": None,
        "source_orientation": "encoded_pixel_order_without_exif_transposition",
        "execution_protocol": "sequential-completion-boundary-v1",
        "execution_source_sha256": sha_file(local_vlm_execution.__file__),
        "stream_progress": stream_progress,
    }
    if image_root is not None:
        config["image_root"] = str(root)
        config["declared_split"] = expected_split
    if task_contract is not None:
        config["task_contract"] = task_contract
        config["coordinate_format"] = task_contract.get("coordinate_format")
    if preprocessing_id is not None:
        from . import aircraft_data

        config["preprocessing_id"] = preprocessing_id
        config["preprocessing_source_sha256"] = sha_file(aircraft_data.__file__)
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
                           model_directory, backend_factory or MLXBackend,
                           stop_after_completions=stop_after_completions)


def _run_locked(prepared, output, config, identity_hash, model_directory, backend_factory,
                *, stop_after_completions=None):
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
    plan = plan_chunk(expected, reserved, completed, stop_after_completions)
    selected = set(plan["selected_ids"])
    pending = [row for row in prepared if row["id"] in selected]
    stopped = any(event.get("hard_stop") for event in completed.values())
    with (output / "executions.jsonl").open("a") as execution:
        _append(execution, {"event": "invocation_started", "created_at": _now(),
                            "identity_sha256": identity_hash, "plan": plan})
    backend = None
    if pending and not stopped:
        with (output / "runtime.jsonl").open("a") as runtime_journal:
            _append(runtime_journal, {"event": "load_started", "created_at": _now(),
                                     "identity_sha256": identity_hash})
            try:
                backend = backend_factory(model_directory, {
                    **config, "memory_telemetry_path": str(output / "mlx-memory.jsonl"),
                    "generation_progress_path": str(output / "generation-progress.jsonl"),
                    "identity_sha256": identity_hash,
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
                if _public(prepare_local(
                    {"id": row["id"], "image": row["image_path"], "prompt": row["prompt"]},
                    config.get("preprocessing_id"),
                )) != expected[row["id"]]:
                    raise ValueError("Image changed after the local run was recorded")
                native = backend.infer(row, item_seed)
                result = native["native_response"]
                prediction = final_answer(result["text"], result.get("finish_reason"),
                                          profile=config["profile"])
                if config.get("task_contract", {}).get("task") == "classification":
                    # The native template prefills this opener. Retain the complete
                    # completion for the same bounded shared grammar as other arms.
                    prediction["classification_response"] = (
                        "<think>" + result["text"] if config["enable_thinking"]
                        else result["text"]
                    )
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
            error = missing_error(identifier, reserved, hard_stop=stopped,
                                  cooperative_stop=bool(plan["deferred_ids"]))
            prediction = {"response": "", "error": error}
        predictions[identifier] = prediction
    (output / "predictions.jsonl").write_text("".join(
        json.dumps({"id": identifier, **prediction}, allow_nan=False) + "\n"
        for identifier, prediction in predictions.items()
    ))
    if backend is not None and hasattr(backend, "close"):
        backend.close()
    with (output / "executions.jsonl").open("a") as execution:
        _append(execution, {
            "event": "invocation_finished", "created_at": _now(),
            "identity_sha256": identity_hash,
            "expected": len(prepared), "reserved": len(reserved), "completed": len(completed),
            "unknown": len(set(reserved) - set(completed)),
            "unattempted": len(set(expected) - set(reserved)), "hard_stop": stopped,
            "cooperative_stop": bool(plan["deferred_ids"]) and not stopped,
            "completed_this_invocation": len(completed) - plan["completed_before"],
            "all_requests_completed": len(completed) == len(prepared),
        })
    return predictions
