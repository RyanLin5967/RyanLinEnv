"""CPU transformers engine for the named baseline: the original Hugging Face weights.

The MLX engine (``local_vlm``) is Apple-only and pinned by the v1 plan's source hashes,
so it is not edited. This module adds a second engine that plugs into the SAME journaled
core, ``local_vlm._run_locked`` (durable reservation before generation, an interrupted
reservation is an unknown outcome and is never regenerated, completed answers are never
repeated), the same ``final_answer`` and ``prepare_local`` pixel identity, the same
native-instruct profile and the same shared aircraft transform. What differs, and is
stated rather than hidden:

  weights   Qwen/Qwen3.8-27B at a pinned revision in bfloat16 (the author dtype), not the
            mlx-community 8-bit affine conversion; every file is rehashed before each load
  numerics  PyTorch CPU kernels instead of MLX Metal; sampled tokens are NOT bit-identical
  sampler   the mlx-vlm 0.6.8 native-instruct sampler re-expressed in torch (presence
            penalty over the last 20 context tokens, nucleus filter on the untempered
            distribution, top-k, then temperature), driven by a seeded torch.Generator

Importing this module loads no model and imports no torch. It lives beside the runner,
not in src/vision_rl, because the freshpool production plan pins every tracked
src/vision_rl module and must not change.

Blind spots, stated on purpose: the sampler equivalence is checked against a written
re-statement of mlx-vlm 0.6.8 (tests), not by running MLX; the RSS telemetry is the
process's own resident set, so another tenant's memory pressure on a shared host is not
seen; thread count comes from the environment (uwrun --threads) and is recorded, not
enforced.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import resource
import socket
import threading
import time
from pathlib import Path

from vision_rl import local_vlm
from vision_rl.frontier import _append, _digest, _now, _public
from vision_rl.local_vlm import model_image, native_profile, prepare_local, sha_file
from vision_rl.local_vlm_execution import plan_chunk

REPOSITORY = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WEIGHT_SHARDS = 18
PACKAGES = (
    "torch", "torchvision", "transformers", "tokenizers", "pillow", "numpy",
    "safetensors", "huggingface-hub",
)
DTYPES = ("bfloat16", "float32")
PRESENCE_CONTEXT = 20  # mlx-vlm 0.6.8 DEFAULT_REPETITION_CONTEXT_SIZE
REQUIRED_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "preprocessor_config.json", "vocab.json", "merges.txt",
    "model.safetensors.index.json",
}


def git_blob_sha1(path):
    """The Hub's blob id for a non-LFS file: sha1 of 'blob <size>\\0' + bytes."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _snapshot_files(directory):
    directory = Path(directory)
    return sorted(
        str(path.relative_to(directory)) for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    )


def build_integrity(directory, metadata):
    """Hash a downloaded snapshot against the Hub's own revision metadata.

    LFS files must match the Hub's sha256; small files must match the Hub's git blob id.
    Missing, extra or symlinked files are refused. Returns the integrity record.
    """
    directory = Path(directory).resolve()
    if metadata.get("sha") != REVISION or metadata.get("id") != REPOSITORY:
        raise ValueError("Hub metadata is not the pinned repository revision")
    expected = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    present = _snapshot_files(directory)
    if set(present) != set(expected):
        raise ValueError(f"Snapshot files differ from the Hub listing: "
                         f"missing {sorted(set(expected) - set(present))}, "
                         f"extra {sorted(set(present) - set(expected))}")
    files = []
    for name in present:
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"Snapshot file is a symlink: {name}")
        entry, digest = expected[name], sha_file(path)
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Size differs from the Hub: {name}")
        lfs = entry.get("lfs")
        if lfs is not None:
            if digest != lfs["sha256"]:
                raise ValueError(f"LFS sha256 differs from the Hub: {name}")
        elif git_blob_sha1(path) != entry["blobId"]:
            raise ValueError(f"Git blob id differs from the Hub: {name}")
        files.append({"path": name, "bytes": entry["size"], "sha256": digest})
    return {"model_repository": REPOSITORY, "revision": REVISION, "weights_dtype": "bfloat16",
            "instrument": "scripts/aircraft_named_baseline_hf.py build_integrity against "
                          "the Hub API revision listing (?blobs=true)", "files": files}


def verify_snapshot(directory, integrity, integrity_sha256):
    """Rehash every snapshot file before loading; the record itself is pinned by sha256."""
    directory = Path(directory).resolve()
    if sha_file(integrity) != integrity_sha256:
        raise ValueError("Model integrity record differs from the plan pin")
    record = json.loads(Path(integrity).read_text())
    if record.get("model_repository") != REPOSITORY or record.get("revision") != REVISION:
        raise ValueError("Local model is not the pinned Qwen3.8-27B snapshot")
    listed = {entry["path"] for entry in record["files"]}
    if set(_snapshot_files(directory)) != listed:
        raise ValueError("Snapshot directory holds files outside the integrity record")
    files = {}
    for entry in record["files"]:
        original = directory / entry["path"]
        path = original.resolve()
        if not path.is_relative_to(directory) or original.is_symlink():
            raise ValueError("Model file escapes the pinned snapshot")
        if path.stat().st_size != entry["bytes"] or sha_file(path) != entry["sha256"]:
            raise ValueError(f"Local model integrity mismatch: {entry['path']}")
        files[entry["path"]] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
    weights = [p for p in files if p.endswith(".safetensors")]
    if not REQUIRED_FILES.issubset(files) or len(weights) != WEIGHT_SHARDS:
        raise ValueError("Incomplete model, tokenizer, or processor snapshot")
    return {"repository": REPOSITORY, "revision": REVISION, "weights_dtype": "bfloat16",
            "files": files}


def runtime_versions(pins):
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != dict(pins):
        raise ValueError(f"Runtime differs from the plan pins: {versions}")
    return versions


def eos_token_ids(directory):
    """The stopping set mlx-vlm used in v1, read from the author snapshot.

    v1 stopped on the mlx-community config's top-level eos_token_id, [248046, 248044],
    which that conversion copied from the author's generation_config.json. The author
    config.json has no top-level field and its text_config names only 248044, which would
    not stop at <|im_end|>; so the generation config is the source, and it must contain the
    text config's EOS.
    """
    directory = Path(directory)
    generation = json.loads((directory / "generation_config.json").read_text())
    value = generation.get("eos_token_id")
    value = [value] if isinstance(value, int) else value
    if not value or any(type(token) is not int for token in value):
        raise ValueError("Generation config declares no integer EOS token")
    text = json.loads((directory / "config.json").read_text()).get("text_config", {})
    if text.get("eos_token_id") is not None and text["eos_token_id"] not in value:
        raise ValueError("Generation EOS set omits the text config's EOS")
    return sorted(set(value))


def mlx_instruct_probs(logits, context_ids, generation):
    """The sampling distribution of mlx-vlm 0.6.8 generate_step for these settings.

    ``logits`` is a 1-D float tensor; ``context_ids`` the prompt plus generated tokens so
    far (mlx-vlm concatenates the full prompt on the first step). Order, as released:
    presence penalty (subtract once per distinct token in the last 20 context tokens),
    log-softmax, nucleus on the UNtempered distribution (keep a token when the ascending
    cumulative probability up to and including it exceeds 1 - top_p), min-p (disabled at
    0), top-k, then softmax(logprobs / temperature). Temperature 0 is refused here because
    mlx-vlm then takes the argmax instead of sampling.
    """
    import torch

    if generation["repetition_penalty"] not in (None, 1.0) or generation["min_p"] != 0.0:
        raise ValueError("Only the native instruct sampler (no repetition, no min-p) is "
                         "re-stated here")
    if not generation["temperature"] > 0:
        raise ValueError("Greedy decoding is not the native instruct profile")
    logits = logits.detach().to(torch.float32).clone()
    penalty = generation["presence_penalty"]
    if penalty:
        recent = torch.unique(context_ids[-PRESENCE_CONTEXT:])
        logits[recent] -= penalty
    logprobs = torch.log_softmax(logits, dim=-1)
    top_p = generation["top_p"]
    if 0 < top_p < 1.0:
        probs = logprobs.exp()
        order = torch.argsort(logprobs, stable=True)
        cumulative = torch.empty_like(probs)
        cumulative[order] = torch.cumsum(probs[order], dim=-1)
        logprobs = torch.where(cumulative > 1 - top_p, logprobs,
                               torch.full_like(logprobs, -float("inf")))
    top_k = generation["top_k"]
    if top_k > 0:
        if not top_k < logprobs.shape[-1]:
            raise ValueError("top_k must be below the vocabulary size, as in mlx-vlm")
        keep = torch.topk(logprobs, top_k).indices
        masked = torch.full_like(logprobs, -float("inf"))
        masked[keep] = logprobs[keep]
        logprobs = masked
    return torch.softmax(logprobs / generation["temperature"], dim=-1)


def mlx_instruct_sample(logits, context_ids, generation, generator):
    """One token drawn from ``mlx_instruct_probs`` with a seeded torch.Generator."""
    import torch

    probs = mlx_instruct_probs(logits, context_ids, generation)
    return int(torch.multinomial(probs, 1, generator=generator))


def _rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux reports KiB


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


class TransformersCPUBackend:
    """Load the pinned snapshot with transformers on CPU, strict and offline."""

    def __init__(self, directory, config):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.config = torch, config
        threads = config["torch_threads"]
        torch.set_num_threads(threads)
        dtype = getattr(torch, config["dtype"])
        start = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
        )
        self.model, loading = AutoModelForImageTextToText.from_pretrained(
            str(directory), dtype=dtype, local_files_only=True, trust_remote_code=False,
            output_loading_info=True,
        )
        problems = {key: list(value) for key, value in loading.items() if value}
        if problems:
            raise ValueError(f"Non-strict checkpoint load: {problems}")
        self.model.eval()
        self.eos = eos_token_ids(directory)
        self.pad = self.processor.tokenizer.pad_token_id
        if self.pad is None:
            self.pad = self.eos[0]
        self.load_info = {
            "load_seconds": time.monotonic() - start,
            "device": {"type": "cpu", "host": socket.gethostname(), "cpu_model": _cpu_model(),
                       "torch_threads": torch.get_num_threads(), "dtype": config["dtype"],
                       "os_cpu_count": os.cpu_count()},
            "rss_bytes_after_load": _rss_bytes(), "peak_rss_bytes_after_load": _peak_rss_bytes(),
            "eos_token_ids": self.eos,
            "eos_source": "generation_config.json eos_token_id (v1's MLX stopping set)",
            "strict_load": "output_loading_info has no missing, unexpected or mismatched keys",
        }
        self.telemetry_stop = threading.Event()
        self.telemetry_thread = threading.Thread(target=self._telemetry, daemon=True)
        self.telemetry_thread.start()

    def _telemetry(self):
        with Path(self.config["memory_telemetry_path"]).open("a") as handle:
            while not self.telemetry_stop.is_set():
                _append(handle, {"created_at": _now(), "rss_bytes": _rss_bytes(),
                                 "peak_rss_bytes": _peak_rss_bytes()})
                self.telemetry_stop.wait(5.0)

    def close(self):
        self.telemetry_stop.set()
        self.telemetry_thread.join(timeout=6)

    def formatted_prompt(self, text):
        mode = native_profile(self.config["profile"])
        template_kwargs = {"enable_thinking": mode["enable_thinking"], "preserve_thinking": True}
        if mode["reasoning_effort"] is not None:
            template_kwargs["reasoning_effort"] = mode["reasoning_effort"]
        # mlx-vlm formats qwen3_5 as LIST_WITH_IMAGE_FIRST: one image entry, then the text.
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
        if not prompt.endswith("<|im_start|>assistant\n" + mode["assistant_prefill"]):
            raise ValueError("Pinned template does not match the native profile prefill")
        if mode["enable_thinking"] != ("Reasoning effort is set to xhigh." in prompt):
            raise ValueError("Pinned template does not match the native reasoning profile")
        expected = self.config.get("expected_formatted_prompt_sha256")
        if expected is not None and _digest(prompt) != expected:
            raise ValueError("Formatted prompt differs from the MLX engine's recorded prompt")
        return prompt, template_kwargs, mode

    def infer(self, row, seed):
        from transformers import (
            GenerationConfig,
            LogitsProcessor,
            LogitsProcessorList,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        torch = self.torch
        prompt, template_kwargs, mode = self.formatted_prompt(row["prompt"])
        with model_image(row["image_path"], self.config.get("preprocessing_id")) as rgb:
            inputs = self.processor(text=[prompt], images=[rgb], return_tensors="pt")
        metadata = {
            "formatted_prompt": prompt, "formatted_prompt_sha256": _digest(prompt),
            "input_ids_shape": list(inputs["input_ids"].shape),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "implicit_external_resize": False, "exif_orientation_transform": False,
            "preprocessing_id": self.config.get("preprocessing_id"),
            "native_profile": mode["native_profile"], "template_kwargs": template_kwargs,
            "assistant_prefill": mode["assistant_prefill"],
        }
        generation = self.config["generation"]
        generator = torch.Generator().manual_seed(seed)
        eos, started = self.eos, time.monotonic()
        prompt_tokens = int(inputs["input_ids"].shape[1])
        timing = {"first_token_at": None, "last_emit": None}
        progress_handle = None
        if self.config.get("stream_progress"):
            progress_handle = Path(self.config["generation_progress_path"]).open("a")

        class Sampler(LogitsProcessor):
            """Draw the native-instruct sample; hand greedy decoding a one-hot choice."""

            def __call__(self, input_ids, scores):
                if input_ids.shape[0] != 1:
                    raise ValueError("The engine decodes one image at a time")
                token = mlx_instruct_sample(scores[0], input_ids[0], generation, generator)
                chosen = torch.full_like(scores, -float("inf"))
                chosen[0, token] = 0.0
                return chosen

        config = self.config

        class Progress(StoppingCriteria):
            """Observe each appended token; never stops generation itself."""

            def __call__(self, input_ids, scores, **kwargs):
                now = time.monotonic()
                produced = input_ids.shape[1] - prompt_tokens
                if timing["first_token_at"] is None:
                    timing["first_token_at"] = now
                if progress_handle is not None and (
                    timing["last_emit"] is None or produced % 64 == 0
                    or now - timing["last_emit"] >= 5.0 or int(input_ids[0, -1]) in eos
                ):
                    _append(progress_handle, {
                        "event": "generation_progress", "created_at": _now(),
                        "id": row["id"], "seed": seed,
                        "identity_sha256": config["identity_sha256"],
                        "elapsed_seconds": now - started, "generation_tokens": produced,
                        "prompt_tokens": prompt_tokens, "rss_bytes": _rss_bytes(),
                    })
                    timing["last_emit"] = now
                return torch.zeros(input_ids.shape[0], dtype=torch.bool)

        # The library's own decode loop, cache and multimodal positions; greedy over the
        # one-hot sample, so no transformers sampling warper ever touches the logits.
        settings = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=generation["max_tokens"],
            eos_token_id=eos, pad_token_id=self.pad, temperature=None, top_p=None,
            top_k=None, min_p=None, repetition_penalty=None, use_cache=True,
        )
        try:
            with torch.inference_mode():
                sequences = self.model.generate(
                    **inputs, generation_config=settings,
                    logits_processor=LogitsProcessorList([Sampler()]),
                    stopping_criteria=StoppingCriteriaList([Progress()]),
                )
        finally:
            if progress_handle is not None:
                progress_handle.close()
        if not torch.equal(sequences[0, :prompt_tokens], inputs["input_ids"][0]):
            raise ValueError("Generated sequence does not extend the formatted prompt")
        produced = [int(token) for token in sequences[0, prompt_tokens:]]
        finish_reason = "stop" if produced and produced[-1] in eos else "length"
        if finish_reason == "length" and len(produced) != generation["max_tokens"]:
            raise ValueError("Generation ended without EOS before the token ceiling")
        first_token_at = timing["first_token_at"]
        finished = time.monotonic()
        text_ids = produced[:-1] if finish_reason == "stop" else produced
        text = self.processor.tokenizer.decode(text_ids, skip_special_tokens=False)
        ttft = (first_token_at or finished) - started
        decode = finished - (first_token_at or finished)
        native = {
            "text": text, "finish_reason": finish_reason, "token_ids": produced,
            "prompt_tokens": prompt_tokens, "generation_tokens": len(produced),
            "total_tokens": prompt_tokens + len(produced),
            "prompt_tps": prompt_tokens / ttft if ttft > 0 else None,
            "generation_tps": (len(produced) - 1) / decode if decode > 0 else None,
            "time_to_first_token_seconds": ttft,
            "peak_memory": _peak_rss_bytes() / 1e9,
            "peak_memory_unit": "decimal GB, process peak RSS (ru_maxrss)",
            "generation_tokens_semantics": "every sampled token, including the EOS that "
                                           "stopped generation; text excludes that EOS",
            "sampler": "aircraft_named_baseline_hf.mlx_instruct_sample (seeded torch.Generator)",
        }
        return {"native_response": native, "preprocessing": metadata,
                "peak_memory_bytes": _peak_rss_bytes(), "rss_bytes_after_inference": _rss_bytes()}


def run_local_hf(
    rows, output, *, model_directory, integrity, integrity_sha256, runtime_pins, phase,
    seed=1729, dtype="bfloat16", torch_threads=None, resume=False, backend_factory=None,
    image_root=None, task_contract=None, preprocessing_id=None,
    stop_after_completions=None, stream_progress=False, profile="instruct",
    expected_formatted_prompt_sha256=None,
):
    """The ``local_vlm.run_local`` contract with the transformers CPU engine.

    Validation mirrors ``run_local`` (which cannot be edited without breaking the v1
    plan's source pin); execution is ``local_vlm._run_locked`` itself.
    """
    if phase not in {"train_probe", "validation"}:
        raise ValueError("Local development runner cannot access final-test images")
    if (Path(output) / "cancelled.json").exists():
        raise ValueError("Cancelled local runs cannot be resumed or retried")
    if dtype not in DTYPES:
        raise ValueError("dtype must be bfloat16 or float32")
    mode = native_profile(profile)
    if type(stream_progress) is not bool:
        raise ValueError("stream_progress must be explicit boolean")
    plan_chunk([str(row["id"]) for row in rows], {}, {}, stop_after_completions)
    if image_root is None:
        raise ValueError("The transformers engine requires an explicit split image root")
    root = Path(image_root).resolve(strict=True)
    expected_split = "train" if phase == "train_probe" else "validation"
    if not rows or any(
        row.get("split") != expected_split
        or not Path(row["image"]).resolve(strict=True).is_relative_to(root) for row in rows
    ):
        raise ValueError("Generic local images must remain within their declared split root")
    if not isinstance(task_contract, dict) or task_contract.get("task") != "classification":
        raise ValueError("The transformers engine runs the classification contract only")
    task_contract = json.loads(json.dumps(task_contract, allow_nan=False))
    if phase == "train_probe" and (
        task_contract.get("probe_images") not in {2, 4, 16}
        or len(rows) != task_contract["probe_images"]
    ):
        raise ValueError("Local training probe requires its explicitly declared 2, 4 or 16 "
                         "images")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned32-bit integer")
    if torch_threads is None:
        torch_threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count()
    if type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    prepared = [prepare_local(row, preprocessing_id) for row in rows]
    if len({row["id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate local request IDs")
    model = verify_snapshot(model_directory, integrity, integrity_sha256)
    from vision_rl import local_vlm_execution

    config = {
        "phase": phase, "seed": seed, **mode, "engine": "transformers-cpu", "dtype": dtype,
        "runtime_versions": runtime_versions(runtime_pins), "concurrency": 1,
        "fewshot_training_images": 0, "resize_shape": None,
        "source_orientation": "encoded_pixel_order_without_exif_transposition",
        "execution_protocol": "sequential-completion-boundary-v1",
        "execution_source_sha256": sha_file(local_vlm_execution.__file__),
        "journal_source_sha256": sha_file(local_vlm.__file__),
        "stream_progress": stream_progress, "image_root": str(root),
        "declared_split": expected_split, "task_contract": task_contract,
        "coordinate_format": task_contract.get("coordinate_format"),
        "expected_formatted_prompt_sha256": expected_formatted_prompt_sha256,
    }
    if preprocessing_id is not None:
        from vision_rl import aircraft_data

        config["preprocessing_id"] = preprocessing_id
        config["preprocessing_source_sha256"] = sha_file(aircraft_data.__file__)
    identity = {
        "schema_version": 1, "kind": "local_transformers_cpu_vision", "model": model,
        "config": config, "rows": [_public(row) for row in prepared],
        "adapter_sha256": sha_file(__file__),
    }
    identity_hash = _digest(identity)
    # Thread count is a host execution detail, not part of what is computed; it is bound
    # to the backend and recorded in runtime.jsonl, but kept out of the resume identity so a
    # shard may resume on a host with a different core count.
    runtime_config = {**config, "torch_threads": torch_threads}
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
        return local_vlm._run_locked(
            prepared, output, runtime_config, identity_hash, model_directory,
            backend_factory or TransformersCPUBackend,
            stop_after_completions=stop_after_completions,
        )


def main(argv=None):
    """``python -I scripts/aircraft_named_baseline_hf.py integrity --snapshot DIR
    --metadata JSON --write OUT``"""
    import argparse

    parser = argparse.ArgumentParser(description="Hash a downloaded snapshot against the Hub")
    commands = parser.add_subparsers(dest="command", required=True)
    integrity = commands.add_parser("integrity")
    integrity.add_argument("--snapshot", type=Path, required=True)
    integrity.add_argument("--metadata", type=Path, required=True)
    integrity.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build_integrity(args.snapshot, json.loads(args.metadata.read_text()))
    with args.write.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"files": len(record["files"]), "written": str(args.write),
                      "sha256": sha_file(args.write)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
