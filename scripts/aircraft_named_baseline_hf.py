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
warpers, whose order differs: presence penalty over the unique tokens among the last 20
entries of mlx-vlm's token history (the prompt, or only its last token when the prompt is
prefilled in 2048-token chunks, followed by the generated tokens), then log-softmax, then
top-p on the untempered distribution, then top-k, then a categorical draw at the declared
temperature. The draw happens inside the logits processor with a per-image
``torch.Generator``; it hands generate() a one-hot score row and generate() runs greedy
(``do_sample=False``), so no Hugging Face warper or model generation default can change the
token. After generation the emitted tokens are compared with the drawn ones and any
difference is refused. The random stream is torch's, not MLX's: the distribution is the
same definition, the individual draws are not.

Blind spots, stated on purpose: integrity is checked against the Hub's published LFS sha256
and git blob ids fetched once into a committed record, so a record edited together with the
files is not detected; the engine does not detect other processes competing for the CPU,
which only changes latency; MKL's chosen code path cannot be read back (torch links MKL
statically), so the kernel pin is enforced by setting it before MKL can initialise and is
checked only through kernel_fingerprint, which detects a difference on its own fixed
operations, not every possible difference inside the model.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
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
# mlx_vlm.generate.ar.DEFAULT_PREFILL_STEP_SIZE; qwen3_5's chunked_prefill_policy enables it.
PREFILL_STEP_SIZE = 2048
REQUIRED_FILES = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "preprocessor_config.json", "generation_config.json", "model.safetensors.index.json",
}


def pin_cpu_numerics(numerics):
    """Hold every shard host to one CPU kernel path; returns the observed path.

    ATen's capability is read from ATEN_CPU_CAPABILITY when torch is imported, MKL's code
    path from MKL_CBWR when MKL initialises and oneDNN's ISA cap from ONEDNN_MAX_CPU_ISA at
    its first dispatch, so all are set before the first torch import; oneDNN is then
    enabled or disabled as the plan says. The pool mixes Ivy Bridge (AVX only), Broadwell
    (AVX2 + FMA) and Skylake-SP (AVX-512); without a pin their default kernels differ
    (kernel_fingerprint, all three), and Ivy Bridge and Broadwell outputs differ (measured:
    reports/2026-10-18-named-baseline-v2-kernel-probe).
    """
    import sys

    # oneDNN reads ONEDNN_MAX_CPU_ISA once, at its first JIT dispatch; None means unset.
    # DNNL_MAX_CPU_ISA is its legacy alias and must never be set beside it.
    wanted = {"ATEN_CPU_CAPABILITY": numerics["aten_cpu_capability_env"],
              "MKL_CBWR": numerics["mkl_cbwr"],
              "ONEDNN_MAX_CPU_ISA": numerics.get("onednn_max_cpu_isa"),
              "DNNL_MAX_CPU_ISA": None}
    loaded = "torch" in sys.modules or any(
        marker in library for library in _mapped_libraries()
        for marker in ("libtorch_cpu", "libmkl", "libiomp"))
    if loaded:
        # MKL reads MKL_CBWR once, when it initialises; if torch or any MKL was already
        # in the process, an environment that differs now cannot be the one in force.
        if any(os.environ.get(k) != v for k, v in wanted.items()):
            raise ValueError("torch or MKL was loaded before the CPU kernel path was pinned")
    else:
        for key, value in wanted.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    import torch

    torch.backends.mkldnn.enabled = numerics["mkldnn_enabled"]
    observed = {
        "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "mkl_available": torch.backends.mkl.is_available(),
        "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA"),
    }
    expected = {"aten_cpu_capability": numerics["aten_cpu_capability"],
                "mkldnn_enabled": numerics["mkldnn_enabled"], "mkl_cbwr": numerics["mkl_cbwr"],
                "mkl_available": numerics["mkl_available"],
                "onednn_max_cpu_isa": numerics.get("onednn_max_cpu_isa")}
    if observed != expected:
        raise ValueError(f"CPU kernel path is not the pre-registered one: {observed}")
    return observed


def _mapped_libraries():
    """Shared objects mapped into this process (Linux); empty where unreadable.

    Blind spot: torch links MKL statically and exports no MKL CBWR query, so the code path
    MKL actually chose cannot be read back; this guard only proves MKL could not have
    initialised before MKL_CBWR was set.
    """
    try:
        return {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                if "/" in line}
    except OSError:
        return set()


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return os.uname().machine


def host_identity():
    """The machine a shard runs on; part of its run identity, so a shard never changes host."""
    return {"hostname": os.uname().nodename, "cpu_model": cpu_model()}


def kernel_fingerprint(threads):
    """SHA256 of fixed float32 computations through the kernels the model uses.

    Covers a matrix product (prefill), a one-row linear (decode), softmax, layer norm,
    scaled-dot-product attention and the vision tower's strided 3-D convolution, on
    seeded inputs at the shard's thread count. It is recorded in the run identity, so the
    merge refuses shards whose kernels computed these differently. Blind spot: equal
    fingerprints do not prove equal model outputs; they rule out a difference these
    operations expose.
    """
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(threads)
    offset = [0]

    def draw(*shape):
        # Inputs from exact integer arithmetic and one correctly rounded division, never
        # torch.randn: ATen's normal sampler picks an AVX2 fill by CPUID whatever the pinned
        # capability, so seeded randn differs between Ivy Bridge and AVX2 hosts (measured:
        # equal model logits, different randn-based fingerprints on iccad3 and iccad5).
        count = math.prod(shape)
        index = torch.arange(offset[0], offset[0] + count, dtype=torch.int64)
        offset[0] += count
        values = (index * 2654435761 + 12345) % 1000003
        return (values.to(torch.float32) / 1000003.0 - 0.5).reshape(shape)

    a, w, x = draw(64, 2048), draw(2048, 2048), draw(1, 2048)
    q, k, v = draw(1, 8, 96, 128), draw(1, 8, 96, 128), draw(1, 8, 96, 128)
    video, kernel = draw(1, 3, 2, 64, 64), draw(32, 3, 2, 16, 16)
    with torch.inference_mode():
        outputs = [a @ w, F.linear(x, w), torch.softmax(a, dim=-1),
                   F.layer_norm(a, (2048,)), F.scaled_dot_product_attention(q, k, v),
                   F.conv3d(video, kernel, stride=(2, 16, 16))]
    digest = hashlib.sha256()
    for output in outputs:
        digest.update(output.contiguous().numpy().tobytes())
    return digest.hexdigest()


# hf download --local-dir keeps its own bookkeeping under .cache/huggingface; transformers
# never reads it. Exactly these names, and only for files the record lists, are tolerated.
HUB_BOOKKEEPING = re.compile(
    r"\.cache/huggingface/(?:CACHEDIR\.TAG|\.gitignore|trees/(?P<tree>[0-9a-f]{40})\.json"
    r"|download/(?P<file>.+)\.(?:lock|metadata))")


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
    # transformers loads optional files it finds (processor_config.json, added_tokens.json,
    # a custom chat template, ...), and the directory sits in shared /dev/shm: anything the
    # record does not list is refused, not ignored.
    def unreadable(error):
        raise ValueError(f"Snapshot directory cannot be listed completely: {error}")

    for parent, directories, names in os.walk(directory, onerror=unreadable):
        for name in directories + names:
            path = Path(parent) / name
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise ValueError(f"Model file escapes the pinned snapshot: {relative}")
            if path.is_dir() or relative in files:
                continue
            match = HUB_BOOKKEEPING.fullmatch(relative)
            if (match is None or (match["tree"] is not None and match["tree"] != REVISION)
                    or (match["file"] is not None and match["file"] not in files)):
                raise ValueError(f"Snapshot holds a file the integrity record does not list: "
                                 f"{relative}")
    return {"repository": REPOSITORY, "revision": REVISION, "quantization": None,
            "files": files}


def _distribution_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_freeze(text):
    """``uv pip freeze`` output as {normalised name: version}; anything else is refused."""
    pins = {}
    for line in text.splitlines():
        name, separator, version = line.strip().partition("==")
        if not separator or not name or not version or _distribution_name(name) in pins:
            raise ValueError(f"Runtime lock line is not a unique name==version: {line!r}")
        pins[_distribution_name(name)] = version
    if not pins:
        raise ValueError("Runtime lock is empty")
    return pins


def installed_distributions():
    installed = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            # A broken install cannot be compared with the lock; refuse it, never skip it.
            raise ValueError(f"Installed distribution without a name at "
                             f"{getattr(distribution, '_path', 'an unknown path')}")
        installed[_distribution_name(name)] = distribution.version
    return installed


def runtime_versions(expected, lock=None):
    """Installed versions of every package the engine imports must equal the plan pins, and
    the whole environment must equal the committed ``uv pip freeze`` lock (transitive
    dependencies such as huggingface_hub included) on the locked Python version."""
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != expected:
        raise ValueError(f"Runtime differs from the pinned versions: {versions}")
    if lock is None:
        raise ValueError("A runtime lock (freeze text and Python version) is required")
    wanted, installed = parse_freeze(lock["freeze"]), installed_distributions()
    if installed != wanted:
        differ = sorted(set(wanted.items()) ^ set(installed.items()))
        raise ValueError(f"Environment differs from the runtime lock: {differ[:10]}")
    if platform.python_version() != lock["python"]:
        raise ValueError(f"Python {platform.python_version()} is not the locked "
                         f"{lock['python']}")
    return versions


def presence_history(prompt_ids, generated, prefill_step_size=PREFILL_STEP_SIZE):
    """The token history mlx-vlm 0.6.8 hands its logits processors.

    generate_step starts ``tokens`` empty and appends the ids its first _step receives:
    the whole prompt, or, when the prompt is longer than the prefill step and is
    prefilled in chunks, only the last prompt token. Generated tokens follow.
    """
    prompt_ids = list(prompt_ids)
    prefix = prompt_ids[-1:] if len(prompt_ids) > prefill_step_size else prompt_ids
    return prefix + list(generated)


def mlx_filter(scores, history, *, temperature, top_p, top_k, min_p, presence_penalty,
               presence_context=PRESENCE_CONTEXT):
    """mlx-vlm 0.6.8's processor and sampler chain on one row; returns tempered logits.

    ``history`` is presence_history(...): the penalty applies to the unique tokens among
    its last ``presence_context`` entries, prompt tokens included, exactly as mlx-lm.
    """
    import torch

    if min_p != 0.0:
        raise ValueError("min_p other than 0 is not implemented by this engine")
    if not (temperature > 0 and 0 < top_p <= 1 and top_k > 0):
        raise ValueError("Sampling needs temperature > 0, 0 < top_p <= 1 and top_k > 0")
    scores = scores.float().clone()
    if presence_penalty and len(history):
        recent = torch.as_tensor(list(history[-presence_context:]), dtype=torch.long)
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


SAMPLING_KEYS = {"max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "repetition_penalty", "enable_thinking"}


def check_generation(generation):
    """Refuse any sampling setting mlx-vlm would apply that mlx_filter does not."""
    unknown = set(generation) - SAMPLING_KEYS
    if unknown:
        raise ValueError(f"Sampling settings this engine does not implement: {sorted(unknown)}")
    if generation.get("repetition_penalty") not in (None, 0, 1.0):
        raise ValueError("repetition_penalty other than 1.0 is not implemented by this engine")


class MLXEquivalentSampler:
    """Logits processor that draws the token itself and forces generate() to emit it."""

    def __init__(self, generation, seed, prompt_length, progress=None):
        import torch

        self.torch, self.generation, self.prompt_length = torch, generation, prompt_length
        self.generator = torch.Generator().manual_seed(seed)
        self.drawn, self.progress, self.step_times = [], progress, []
        self.prompt_ids = None

    def __call__(self, input_ids, scores):
        torch = self.torch
        if input_ids.shape[0] != 1:
            raise ValueError("The engine generates one image at a time")
        generated = input_ids[0, self.prompt_length:].tolist()
        if generated != self.drawn:
            raise ValueError("generate() emitted a token this sampler did not draw")
        if self.prompt_ids is None:
            check_generation(self.generation)
            self.prompt_ids = input_ids[0, :self.prompt_length].tolist()
        history = presence_history(self.prompt_ids, generated)
        g = self.generation
        logits = mlx_filter(scores, history, temperature=g["temperature"], top_p=g["top_p"],
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
        engine = config["engine"]
        if engine["dtype"] not in DTYPES:
            raise ValueError("Engine dtype must be bfloat16 or float32")
        if not engine.get("formatted_prompt_sha256"):
            raise ValueError("Engine needs the pinned formatted-prompt SHA256")
        numerics = pin_cpu_numerics(engine["cpu_numerics"])
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

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
                       "cpu_numerics": numerics, "cpu_model": cpu_model(),
                       "host": os.uname().nodename},
            "dtype": engine["dtype"], "eos_token_ids": sorted(self.eos),
            "max_rss_bytes_after_load": _max_rss(),
            "author_generation_config_replaced": author.to_dict(),
        }

    def close(self):
        pass

    def infer(self, row, seed):
        torch = self.torch
        mode = local_vlm.native_profile(self.config["profile"])
        template_kwargs = {"enable_thinking": mode["enable_thinking"], "preserve_thinking": True}
        if mode["reasoning_effort"] is not None:
            template_kwargs["reasoning_effort"] = mode["reasoning_effort"]
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        if not prompt.endswith("<|im_start|>assistant\n" + mode["assistant_prefill"]):
            raise ValueError("Pinned template does not match the native profile prefill")
        if _digest(prompt) != self.config["engine"]["formatted_prompt_sha256"]:
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
        # Prefill (vision + prompt) ends when the first token's scores arrive.
        prefill = sampler.step_times[0] - start if sampler.step_times else None
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
            # mlx-vlm's meanings: prompt tokens per prefill second, and generated tokens
            # per decode second (after the first token).
            "prompt_tps": prompt_length / prefill if prefill else None,
            "generation_tps": (len(tokens) - 1) / (elapsed - prefill)
            if prefill is not None and len(tokens) > 1 and elapsed > prefill else None,
            "peak_memory": _max_rss() / 1e9, "generate_seconds": elapsed,
            "prefill_seconds": prefill,
            "peak_memory_unit": "decimal GB, process max RSS (getrusage)",
            "logprobs_retained": False,
        }
        return {"native_response": native, "preprocessing": metadata,
                "peak_memory_bytes": _max_rss()}


def _max_rss():
    # Linux reports KiB, macOS bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


MACHINE_KEYS = ("observed_cpu_numerics", "kernel_fingerprint", "host", "runtime_versions",
                "runtime_lock_sha256", "python")


def machine_preflight(model_directory, integrity, engine, runtime_pins, runtime_lock):
    """Everything about this machine that enters a run identity, checked before any charge.

    Rehashes and checks the snapshot (no unlisted file), the runtime lock, the kernel pin,
    and computes the kernel fingerprint and host. Raises on any refusal.
    """
    model = verify_snapshot(model_directory, integrity)
    versions = runtime_versions(runtime_pins, runtime_lock)
    observed = pin_cpu_numerics(engine["cpu_numerics"])
    return {"model": model, "observed_cpu_numerics": observed,
            "kernel_fingerprint": kernel_fingerprint(engine["threads"]),
            "host": host_identity(), "runtime_versions": versions,
            "runtime_lock_sha256": _digest(runtime_lock), "python": platform.python_version()}


def run_local_hf(
    rows, output, *, model_directory, integrity, engine, runtime_pins, phase, seed=1729,
    runtime_lock=None, preflight=None,
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
    if engine.get("dtype") not in DTYPES or type(engine.get("threads")) is not int:
        # float32 reductions (ATen parallel loops, MKL without CBWR STRICT) depend on the
        # thread count, so it is part of the identity and fixed by the plan.
        raise ValueError("Engine identity needs an explicit dtype and thread count")
    if not engine.get("formatted_prompt_sha256") or "cpu_numerics" not in engine:
        raise ValueError("Engine identity needs the formatted-prompt SHA256 and CPU numerics")
    prepared = [local_vlm.prepare_local(row, preprocessing_id) for row in rows]
    if len({row["id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate local request IDs")
    # run_shard passes the preflight it made before charging the invocation; it is the same
    # function on the same process, so the snapshot is not rehashed twice.
    checked = preflight or machine_preflight(model_directory, integrity, engine, runtime_pins,
                                             runtime_lock)
    if (checked["runtime_versions"] != runtime_pins
            or checked["runtime_lock_sha256"] != _digest(runtime_lock)):
        raise ValueError("The preflight was made for another runtime or lock")
    model = checked["model"]
    config = {
        "phase": phase, "seed": seed, **mode, "engine": engine,
        # observed_cpu_numerics equals the plan's or the pin raised; it is a record. The
        # fingerprint is the check that can differ between hosts.
        **{key: checked[key] for key in MACHINE_KEYS},
        "coordinate_format": task_contract.get("coordinate_format"),
        "concurrency": 1, "fewshot_training_images": 0, "resize_shape": None,
        "source_orientation": "encoded_pixel_order_without_exif_transposition",
        "execution_protocol": "sequential-completion-boundary-v1",
        "execution_source_sha256": local_vlm.sha_file(local_vlm_execution.__file__),
        "journal_source_sha256": local_vlm.sha_file(local_vlm.__file__),
        # Where the tree was staged is not identity (each uwrun run syncs to a new directory,
        # and hosts differ); which split directory it is, and every image's hashes, are.
        "stream_progress": stream_progress, "image_root_suffix": "/".join(root.parts[-4:]),
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
        # A directory left by an invocation killed before run.json was written holds at most
        # the lock file; anything else is history and is never overwritten.
        output.mkdir(parents=True, exist_ok=True)
        if any(child.name != ".run.lock" for child in output.iterdir()):
            raise ValueError("A new local run directory must be empty")
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
