"""CPU Hugging Face transformers engine for the journaled local Qwen vision runner.

Lives in scripts/, not in the vision_rl package, on purpose: several pre-registered plans
and closures hash every module of src/vision_rl, so a new module there would change their
pinned source sets.

``vision_rl.local_vlm`` is the validated journaled runner, but its preamble is tied to the
8-bit MLX snapshot (``verify_snapshot`` pins that repository, ``runtime_versions`` requires
mlx-vlm 0.6.8) and its source hash is pinned by the paused v1 named-baseline run. So this
module adds an engine beside it instead of editing it: it rebuilds only the preamble
(snapshot verification, runtime pins, run identity) for the author bf16 weights and then
hands over to ``local_vlm._run_locked``, the same reservation/completion journal loop, with
the same ``final_answer`` rule and the same ``classification_response`` field.

Sampling replicates mlx-vlm 0.6.8 (``mlx_vlm.sample_utils``) instead of Hugging Face's
warpers, whose order differs: presence penalty over the last 20 generated tokens, then
log-softmax, then top-p on the untempered distribution, then top-k, then a categorical draw
at the declared temperature. The draw happens inside the logits processor with a per-image
``torch.Generator``; it hands generate() a one-hot score row and generate() runs greedy
(``do_sample=False``), so no Hugging Face warper or model generation default can change the
token. After generation the emitted tokens are compared with the drawn ones and any
difference is refused. The random stream is torch's, not MLX's: the distribution is the
same definition, the individual draws are not.

Blind spots, stated on purpose: integrity is checked against the Hub's published LFS sha256
and git blob ids fetched once into a committed record, so a record edited together with the
files is not detected; the engine does not detect other processes competing for the CPU,
which only changes latency.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import resource
import time
from pathlib import Path

from vision_rl import aircraft_data, local_vlm, local_vlm_execution
from vision_rl.frontier import _digest, _now, _public
from vision_rl.local_vlm_execution import plan_chunk

REPOSITORY = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PACKAGES = ("torch", "torchvision", "transformers", "tokenizers", "pillow", "numpy", "safetensors")
DTYPES = ("bfloat16", "float32")
PRESENCE_CONTEXT = 20
REQUIRED_FILES = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "preprocessor_config.json", "generation_config.json", "model.safetensors.index.json",
}


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_snapshot(directory, integrity):
    """Rehash every file of the author snapshot against the committed Hub record."""
    directory = Path(directory).resolve()
    record = json.loads(Path(integrity).read_text())
    if record.get("model_repository") != REPOSITORY or record.get("revision") != REVISION:
        raise ValueError("Local model is not the pinned author Qwen3.8-27B snapshot")
    files = {}
    for entry in record["files"]:
        original = directory / entry["path"]
        if original.is_symlink() or not original.resolve().is_relative_to(directory):
            raise ValueError("Model file escapes the pinned snapshot")
        if original.stat().st_size != entry["bytes"]:
            raise ValueError(f"Local model size mismatch: {entry['path']}")
        if "sha256" in entry:
            if local_vlm.sha_file(original) != entry["sha256"]:
                raise ValueError(f"Local model integrity mismatch: {entry['path']}")
        elif git_blob_sha1(original) != entry["git_blob_id"]:
            raise ValueError(f"Local model integrity mismatch: {entry['path']}")
        files[entry["path"]] = {key: entry[key] for key in ("bytes", "sha256", "git_blob_id")
                                if key in entry}
    shards = [p for p in files if p.endswith(".safetensors")]
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if not REQUIRED_FILES.issubset(files) or set(index["weight_map"].values()) != set(shards):
        raise ValueError("Incomplete model, tokenizer, or processor snapshot")
    return {"repository": REPOSITORY, "revision": REVISION, "quantization": None,
            "files": files}


def runtime_versions(expected):
    """Installed versions of every package the engine imports; must equal the plan pins."""
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != expected:
        raise ValueError(f"Runtime differs from the pinned versions: {versions}")
    return versions


def mlx_filter(scores, generated, *, temperature, top_p, top_k, min_p, presence_penalty,
               presence_context=PRESENCE_CONTEXT):
    """mlx-vlm 0.6.8's processor and sampler chain on one row; returns tempered logits."""
    import torch

    if min_p != 0.0:
        raise ValueError("min_p other than 0 is not implemented by this engine")
    if not (temperature > 0 and 0 < top_p <= 1 and top_k > 0):
        raise ValueError("Sampling needs temperature > 0, 0 < top_p <= 1 and top_k > 0")
    scores = scores.float().clone()
    if presence_penalty and len(generated):
        recent = torch.as_tensor(list(generated[-presence_context:]), dtype=torch.long)
        scores[..., recent.unique()] -= presence_penalty
    logprobs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
    if top_p < 1.0:
        order = torch.argsort(logprobs, dim=-1)
        cumulative = torch.cumsum(torch.exp(logprobs).gather(-1, order), dim=-1)
        keep = torch.empty_like(cumulative, dtype=torch.bool)
        keep.scatter_(-1, order, cumulative > 1 - top_p)
        logprobs = logprobs.masked_fill(~keep, float("-inf"))
    if top_k < logprobs.shape[-1]:
        kth = torch.topk(logprobs, top_k, dim=-1).indices
        mask = torch.ones_like(logprobs, dtype=torch.bool)
        mask.scatter_(-1, kth, False)
        logprobs = logprobs.masked_fill(mask, float("-inf"))
    return logprobs / temperature


class MLXEquivalentSampler:
    """Logits processor that draws the token itself and forces generate() to emit it."""

    def __init__(self, generation, seed, prompt_length, progress=None):
        import torch

        self.torch, self.generation, self.prompt_length = torch, generation, prompt_length
        self.generator = torch.Generator().manual_seed(seed)
        self.drawn, self.progress, self.step_times = [], progress, []

    def __call__(self, input_ids, scores):
        torch = self.torch
        if input_ids.shape[0] != 1:
            raise ValueError("The engine generates one image at a time")
        generated = input_ids[0, self.prompt_length:].tolist()
        if generated != self.drawn:
            raise ValueError("generate() emitted a token this sampler did not draw")
        g = self.generation
        logits = mlx_filter(scores, generated, temperature=g["temperature"], top_p=g["top_p"],
                            top_k=g["top_k"], min_p=g["min_p"],
                            presence_penalty=g["presence_penalty"])
        token = int(torch.multinomial(torch.softmax(logits, dim=-1)[0], 1,
                                      generator=self.generator))
        self.drawn.append(token)
        self.step_times.append(time.monotonic())
        if self.progress is not None and (len(self.drawn) == 1 or len(self.drawn) % 32 == 0):
            self.progress({"generated_tokens": len(self.drawn)})
        forced = torch.full_like(scores, float("-inf"))
        forced[0, token] = 0.0
        return forced


class TransformersBackend:
    """Author weights on CPU through transformers; imported only when inference starts."""

    def __init__(self, directory, config):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        engine = config["engine"]
        if engine["dtype"] not in DTYPES:
            raise ValueError("Engine dtype must be bfloat16 or float32")
        self.torch, self.config = torch, config
        torch.set_num_threads(engine["threads"])
        start = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(directory), dtype=getattr(torch, engine["dtype"]), local_files_only=True,
            trust_remote_code=False,
        ).eval()
        author = self.model.generation_config
        eos = author.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        from transformers import GenerationConfig

        # Replace the author defaults so generate() merges nothing into the neutral greedy
        # config: no default processor may touch the scores before the sampler sees them.
        self.neutral = GenerationConfig(
            do_sample=False, eos_token_id=sorted(self.eos), pad_token_id=sorted(self.eos)[0])
        self.model.generation_config = self.neutral
        self.load_info = {
            "load_seconds": time.monotonic() - start,
            "device": {"type": "cpu", "torch_threads": torch.get_num_threads(),
                       "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                       "host": os.uname().nodename},
            "dtype": engine["dtype"], "eos_token_ids": sorted(self.eos),
            "max_rss_bytes_after_load": _max_rss(),
            "author_generation_config_replaced": author.to_dict(),
        }

    def close(self):
        pass

    def infer(self, row, seed):
        torch = self.torch
        mode = local_vlm.native_profile(self.config.get("profile", "xhigh"))
        template_kwargs = {"enable_thinking": mode["enable_thinking"], "preserve_thinking": True}
        if mode["reasoning_effort"] is not None:
            template_kwargs["reasoning_effort"] = mode["reasoning_effort"]
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        if not prompt.endswith("<|im_start|>assistant\n" + mode["assistant_prefill"]):
            raise ValueError("Pinned template does not match the native profile prefill")
        expected = self.config["engine"].get("formatted_prompt_sha256")
        if expected is not None and _digest(prompt) != expected:
            raise ValueError("Formatted prompt differs from the pinned MLX formatted prompt")
        with local_vlm.model_image(row["image_path"], self.config.get("preprocessing_id")) as rgb:
            inputs = self.processor(text=[prompt], images=[rgb], return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[1]
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
        progress = None
        if self.config.get("stream_progress"):
            path = Path(self.config["generation_progress_path"])

            def progress(observation):
                with path.open("a") as handle:
                    local_vlm._append(handle, {
                        "event": "generation_progress", "created_at": _now(), "id": row["id"],
                        "seed": seed, "identity_sha256": self.config["identity_sha256"],
                        **observation})

        generation = self.config["generation"]
        sampler = MLXEquivalentSampler(generation, seed, prompt_length, progress)
        from transformers import LogitsProcessorList

        neutral = copy.deepcopy(self.neutral)
        neutral.max_new_tokens = generation["max_tokens"]
        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=neutral,
                                         logits_processor=LogitsProcessorList([sampler]))
        elapsed = time.monotonic() - start
        tokens = output[0, prompt_length:].tolist()
        if tokens != sampler.drawn:
            raise ValueError("generate() output differs from the tokens the sampler drew")
        stopped = bool(tokens) and tokens[-1] in self.eos
        text_tokens = tokens[:-1] if stopped else tokens
        text = self.processor.tokenizer.decode(text_tokens, skip_special_tokens=False)
        native = {
            "text": text, "token_ids": tokens,
            "finish_reason": "stop" if stopped else "length",
            "prompt_tokens": prompt_length, "generation_tokens": len(tokens),
            "total_tokens": prompt_length + len(tokens),
            "prompt_tps": None, "generation_tps": len(tokens) / elapsed if elapsed else None,
            "peak_memory": _max_rss() / 1e9, "generate_seconds": elapsed,
            # Prefill (vision + prompt) ends when the first token's scores arrive.
            "prefill_seconds": sampler.step_times[0] - start if sampler.step_times else None,
            "peak_memory_unit": "decimal GB, process max RSS (getrusage)",
            "logprobs_retained": False,
        }
        return {"native_response": native, "preprocessing": metadata,
                "peak_memory_bytes": _max_rss()}


def _max_rss():
    # Linux reports KiB, macOS bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def run_local_hf(
    rows, output, *, model_directory, integrity, engine, runtime_pins, phase, seed=1729,
    resume=False, backend_factory=None, image_root, task_contract, preprocessing_id,
    stop_after_completions=None, stream_progress=False, profile="instruct",
):
    """``local_vlm.run_local`` for the CPU transformers engine; same journal, new identity."""
    if phase != "validation":
        raise ValueError("The transformers engine journal runs native validation only")
    if (Path(output) / "cancelled.json").exists():
        raise ValueError("Cancelled local runs cannot be resumed or retried")
    mode = local_vlm.native_profile(profile)
    if type(stream_progress) is not bool:
        raise ValueError("stream_progress must be explicit boolean")
    plan_chunk([str(row["id"]) for row in rows], {}, {}, stop_after_completions)
    root = Path(image_root).resolve(strict=True)
    if not rows or any(
        row.get("split") != "validation"
        or not Path(row["image"]).resolve(strict=True).is_relative_to(root) for row in rows
    ):
        raise ValueError("Generic local images must remain within their declared split root")
    if not isinstance(task_contract, dict) or task_contract.get("task") != "classification":
        raise ValueError("The transformers engine runs the classification contract only")
    task_contract = json.loads(json.dumps(task_contract, allow_nan=False))
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned32-bit integer")
    if engine.get("dtype") not in DTYPES or "threads" in engine:
        # Threads change latency, not outputs: they are passed to the backend factory
        # and recorded by the caller, never part of the run identity.
        raise ValueError("Engine identity needs an explicit dtype and no thread count")
    prepared = [local_vlm.prepare_local(row, preprocessing_id) for row in rows]
    if len({row["id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate local request IDs")
    model = verify_snapshot(model_directory, integrity)
    config = {
        "phase": phase, "seed": seed, **mode, "engine": engine,
        "runtime_versions": runtime_versions(runtime_pins),
        "coordinate_format": task_contract.get("coordinate_format"),
        "concurrency": 1, "fewshot_training_images": 0, "resize_shape": None,
        "source_orientation": "encoded_pixel_order_without_exif_transposition",
        "execution_protocol": "sequential-completion-boundary-v1",
        "execution_source_sha256": local_vlm.sha_file(local_vlm_execution.__file__),
        "journal_source_sha256": local_vlm.sha_file(local_vlm.__file__),
        "stream_progress": stream_progress, "image_root": str(root),
        "declared_split": "validation", "task_contract": task_contract,
        "preprocessing_id": preprocessing_id,
        "preprocessing_source_sha256": local_vlm.sha_file(aircraft_data.__file__),
    }
    identity = {
        "schema_version": 1, "kind": "local_hf_transformers_cpu_vision", "model": model,
        "config": config, "rows": [_public(row) for row in prepared],
        "adapter_sha256": local_vlm.sha_file(__file__),
    }
    identity_hash = _digest(identity)
    output = Path(output)
    if resume:
        if not output.is_dir():
            raise ValueError("Cannot resume a missing local run")
    else:
        output.mkdir(parents=True, exist_ok=False)
    import fcntl

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
            prepared, output, config, identity_hash, model_directory,
            backend_factory or TransformersBackend,
            stop_after_completions=stop_after_completions)
