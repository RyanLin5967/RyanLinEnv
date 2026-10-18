"""Offline, single-device TRL token-GRPO and matched completion-only VLM SFT.

This adapter does not implement an RL loss. Released TRL owns sampling, token
log-probabilities, group advantages, clipping, masks, and optimizer checkpoints.
Importing this module neither imports torch nor loads/downloads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from . import expanded_contract as _expanded
from .aircraft_data import PREPROCESSING_ID, aircraft_pixel_sha256, load_aircraft_rgb
from .data import assert_disjoint, image_hash
from .vlm import _completion_text, _dataset_identity, _file_sha256, _validated_lora_config

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_GRAMMAR = "canonical-label-v2/optional-tags"
# Exploratory sweep options (default-off: the legacy formats keep their exact prompt,
# grammar and identity). `closed_set_*` lists the labels grouped by manufacturer and
# family with an explicit closed-set instruction; `*think_answer` asks for
# <think>...</think><answer>LABEL</answer> and is scored on the answer tag only.
LEGACY_RESPONSE_FORMATS = ("answer_only", "think_then_answer")
TAGGED_RESPONSE_FORMATS = ("think_answer", "closed_set_think_answer")
CLOSED_SET_RESPONSE_FORMATS = ("closed_set_answer_only", "closed_set_think_answer")
RESPONSE_FORMATS = LEGACY_RESPONSE_FORMATS + ("closed_set_answer_only",) + TAGGED_RESPONSE_FORMATS
TAGGED_OUTPUT_GRAMMAR = "think-answer-tags-v1/answer-only"
# DAPO/Dr.GRPO/GSPO knobs that released TRL 1.16 GRPOConfig implements, with their
# legacy values in `trainer_kwargs`. Only explicitly passed knobs enter settings.
GRPO_KNOBS = {
    "epsilon": 0.2,
    "epsilon_high": None,
    "delta": None,
    "loss_type": "grpo",
    "scale_rewards": "group",
    "importance_sampling_level": "token",
    "mask_truncated_completions": True,
    "num_iterations": 1,
    "top_entropy_quantile": 1.0,
}
GRPO_LOSS_TYPES = ("grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo", "luspo", "vespo")
# FGVC-Aircraft 2013b taxonomy for the closed-set prompt. Every (manufacturer, family,
# variants) row was derived from the author release's native TRAIN annotations only
# (images_{variant,family,manufacturer}_train.txt, SHA-256 29ac66fb..., 9bc9b649...,
# ff5d8c11...): each of the 100 variants maps to exactly one family and manufacturer
# there. No VAL or TEST file was read. Rows follow the data spec's label order. It lives
# here, not in a new module, because frozen study plans pin every vision_rl module.
AIRCRAFT_TAXONOMY_ID = "fgvc-aircraft-2013b-train-taxonomy-v1"
AIRCRAFT_FAMILY_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('Boeing', 'Boeing 707', ('707-320',)),
    ('Boeing', 'Boeing 727', ('727-200',)),
    ('Boeing', 'Boeing 737', (
        '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900',
    )),
    ('Boeing', 'Boeing 747', ('747-100', '747-200', '747-300', '747-400')),
    ('Boeing', 'Boeing 757', ('757-200', '757-300')),
    ('Boeing', 'Boeing 767', ('767-200', '767-300', '767-400')),
    ('Boeing', 'Boeing 777', ('777-200', '777-300')),
    ('Airbus', 'A300', ('A300B4',)),
    ('Airbus', 'A310', ('A310',)),
    ('Airbus', 'A320', ('A318', 'A319', 'A320', 'A321')),
    ('Airbus', 'A330', ('A330-200', 'A330-300')),
    ('Airbus', 'A340', ('A340-200', 'A340-300', 'A340-500', 'A340-600')),
    ('Airbus', 'A380', ('A380',)),
    ('ATR', 'ATR-42', ('ATR-42',)),
    ('ATR', 'ATR-72', ('ATR-72',)),
    ('Antonov', 'An-12', ('An-12',)),
    ('British Aerospace', 'BAE 146', ('BAE 146-200', 'BAE 146-300')),
    ('British Aerospace', 'BAE-125', ('BAE-125',)),
    ('Beechcraft', 'Beechcraft 1900', ('Beechcraft 1900',)),
    ('Boeing', 'Boeing 717', ('Boeing 717',)),
    ('Lockheed Corporation', 'C-130', ('C-130',)),
    ('Douglas Aircraft Company', 'C-47', ('C-47',)),
    ('Canadair', 'CRJ-200', ('CRJ-200',)),
    ('Canadair', 'CRJ-700', ('CRJ-700', 'CRJ-900')),
    ('Cessna', 'Cessna 172', ('Cessna 172',)),
    ('Cessna', 'Cessna 208', ('Cessna 208',)),
    ('Cessna', 'Cessna Citation', ('Cessna 525', 'Cessna 560')),
    ('Canadair', 'Challenger 600', ('Challenger 600',)),
    ('McDonnell Douglas', 'DC-10', ('DC-10',)),
    ('Douglas Aircraft Company', 'DC-3', ('DC-3',)),
    ('Douglas Aircraft Company', 'DC-6', ('DC-6',)),
    ('Douglas Aircraft Company', 'DC-8', ('DC-8',)),
    ('McDonnell Douglas', 'DC-9', ('DC-9-30',)),
    ('de Havilland', 'DH-82', ('DH-82',)),
    ('de Havilland', 'DHC-1', ('DHC-1',)),
    ('de Havilland', 'DHC-6', ('DHC-6',)),
    ('de Havilland', 'Dash 8', ('DHC-8-100', 'DHC-8-300')),
    ('Robin', 'DR-400', ('DR-400',)),
    ('Dornier', 'Dornier 328', ('Dornier 328',)),
    ('Embraer', 'Embraer E-Jet', ('E-170', 'E-190', 'E-195')),
    ('Embraer', 'EMB-120', ('EMB-120',)),
    ('Embraer', 'Embraer ERJ 145', ('ERJ 135', 'ERJ 145')),
    ('Embraer', 'Embraer Legacy 600', ('Embraer Legacy 600',)),
    ('Eurofighter', 'Eurofighter Typhoon', ('Eurofighter Typhoon',)),
    ('Lockheed Martin', 'F-16', ('F-16A/B',)),
    ('McDonnell Douglas', 'F/A-18', ('F/A-18',)),
    ('Dassault Aviation', 'Falcon 2000', ('Falcon 2000',)),
    ('Dassault Aviation', 'Falcon 900', ('Falcon 900',)),
    ('Fokker', 'Fokker 100', ('Fokker 100',)),
    ('Fokker', 'Fokker 50', ('Fokker 50',)),
    ('Fokker', 'Fokker 70', ('Fokker 70',)),
    ('Bombardier Aerospace', 'Global Express', ('Global Express',)),
    ('Gulfstream Aerospace', 'Gulfstream', ('Gulfstream IV', 'Gulfstream V')),
    ('British Aerospace', 'Hawk T1', ('Hawk T1',)),
    ('Ilyushin', 'Il-76', ('Il-76',)),
    ('Lockheed Corporation', 'L-1011', ('L-1011',)),
    ('McDonnell Douglas', 'MD-11', ('MD-11',)),
    ('McDonnell Douglas', 'MD-80', ('MD-80', 'MD-87')),
    ('McDonnell Douglas', 'MD-90', ('MD-90',)),
    ('Fairchild', 'Metroliner', ('Metroliner',)),
    ('Beechcraft', 'King Air', ('Model B200',)),
    ('Piper', 'PA-28', ('PA-28',)),
    ('Cirrus Aircraft', 'SR-20', ('SR-20',)),
    ('Saab', 'Saab 2000', ('Saab 2000',)),
    ('Saab', 'Saab 340', ('Saab 340',)),
    ('Supermarine', 'Spitfire', ('Spitfire',)),
    ('Panavia', 'Tornado', ('Tornado',)),
    ('Tupolev', 'Tu-134', ('Tu-134',)),
    ('Tupolev', 'Tu-154', ('Tu-154',)),
    ('Yakovlev', 'Yak-42', ('Yak-42',)),
)


def aircraft_grouped_label_lines(spec_labels: list[str]) -> list[str]:
    """One line per family, manufacturers in first-appearance order; refuses other label sets."""
    known = [label for _, _, variants in AIRCRAFT_FAMILY_GROUPS for label in variants]
    if len(set(spec_labels)) != len(spec_labels) or sorted(spec_labels) != sorted(known):
        raise ValueError("closed-set grouping requires exactly the 100 FGVC-Aircraft variants")
    order: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for manufacturer, family, variants in AIRCRAFT_FAMILY_GROUPS:
        order.setdefault(manufacturer, []).append((family, variants))
    return [
        f"{manufacturer} / {family}: " + " | ".join(variants)
        for manufacturer, families in order.items()
        for family, variants in families
    ]

# Study controllers freeze finite original budgets before execution. The TRAIN
# diagnostic bounds RL updates, while a matched-time SFT trajectory may need
# more updates. None means no additional step cap, not an unbounded run.
TRAIN_ONLY_STEP_CAPS = {"grpo": 400, "sft": None}
# Screened RL scans, and SFT replaying a given sequence, read up to the whole fresh
# pool (2,919 native-TRAIN images outside train400 and probe16).
SCAN_ROW_CAP = 2919
# Exploratory sweep on TRAIN-derived holdouts only (docs/aircraft-exploratory-sweep-v2.md).
# Naming this exact id (with --train-only) lifts the 400-update GRPO and 400-row random-order
# caps up to the native TRAIN size; the id enters settings, so every such artifact says so.
EXPLORATORY_SWEEP_ID = "aircraft-exploratory-sweep-v2"
NATIVE_TRAIN_ROWS = 3334


def _torch_threads() -> int:
    """Intra-op threads: VLM_TORCH_THREADS when set (1..256), else the historical 2."""
    raw = os.environ.get("VLM_TORCH_THREADS")
    if raw is None:
        return 2
    if not raw.isdigit() or not 1 <= int(raw) <= 256:
        raise ValueError("VLM_TORCH_THREADS must be an integer in 1..256")
    return int(raw)
MODEL_REVISIONS = {
    DEFAULT_MODEL: "895c3a49bc3fa70a340399125c650a463535e71c",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
}
STACK = {
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "trl": "1.16.0",
    "transformers": "5.20.0",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
    "datasets": "5.1.0",
    "fsspec": "2026.7.0",
}
TRL_SOURCES = {
    "GRPOConfig": "fa6d620785f63f6e8c6e42ec41f38faf249198f0eb182bf0808dde8cd76a6819",
    "GRPOTrainer": "1c3dc2253fb3653269c11dfce4f946f74201a4cb63e0887a37e8e3bb4a3f5503",
    "SFTConfig": "4a18512fbdc83c9d16be09ee68529cd4cc07e4243c9ad733686fd3cb70984d6b",
    "SFTTrainer": "a93001470796c6502a4cf8886a13abbd8280e653357b5a95eb1abd53c6868205",
}


DEVICES = ("mps", "cpu", "cuda")
DEFAULT_THREADS = 2
PRECISION_TRANSFERS = ("float32-to-bfloat16", "float32-to-float16")  # warm-start training
BASE_DTYPES = ("float32", "bfloat16", "float16")
# Frozen-adapter TRAIN inference may declare any parent->target base precision change.
INFERENCE_PRECISION_TRANSFERS = tuple(
    f"{a}-to-{b}" for a in BASE_DTYPES for b in BASE_DTYPES if a != b
)
DEFAULT_LOSS_SCALE_INIT = 65536.0  # torch.amp.GradScaler's own default
LOSS_SCALE_GROWTH_INTERVAL = 2000  # torch.amp.GradScaler's own default
# 2**16 / 2**16 = 1: past this many back-to-back overflows the scale has fallen below 1,
# so the overflow is not scaling but a nonfinite forward/backward.
MAX_CONSECUTIVE_LOSS_SCALE_SKIPS = 16


# RFT runs on every trainer device (CUDA from sweep-feat-cuda.noindex).
RFT_DEVICES = DEVICES


class LocalVLMError(RuntimeError):
    """An offline dependency, data, or device contract failed."""


@dataclass(frozen=True)
class Profile:
    response_format: str = "think_then_answer"
    dtype: str = "float32"
    min_pixels: int = 64 * 28 * 28
    max_pixels: int = 256 * 28 * 28
    max_prompt_tokens: int = 2048
    max_completion_length: int = 256
    num_generations: int = 4
    lora_rank: int = 8
    temperature: float = 1.0

    def validate(self) -> None:
        if self.response_format not in RESPONSE_FORMATS:
            raise ValueError("response_format must be one of " + ", ".join(RESPONSE_FORMATS))
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16 or float16")
        for name, value in asdict(self).items():
            if name not in {"dtype", "temperature", "response_format", "adapter_profile"}:
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.num_generations <= 16 or not 1 <= self.lora_rank <= 64:
            raise ValueError("num_generations must be 2..16 and lora_rank 1..64")
        if not 28 * 28 <= self.min_pixels <= self.max_pixels <= 1024 * 28 * 28:
            raise ValueError("invalid bounded Qwen image pixel profile")
        if not 4 <= self.max_completion_length <= 512 or self.max_prompt_tokens > 8192:
            raise ValueError("invalid bounded token profile")
        if not math.isfinite(self.temperature) or not 0 < self.temperature <= 2:
            raise ValueError("temperature must be finite and in (0, 2]")


@dataclass(frozen=True)
class ExpandedProfile(Profile):
    """Separately versioned language-seven rank-8 FP32 LoRA profile.

    A subclass, so every legacy Q/V `asdict(Profile())` identity stays byte-identical.
    Its artifacts go only through `expanded_contract`, never through the 112-tensor grader.
    """

    adapter_profile: str = _expanded.PROFILE_ID

    def validate(self) -> None:
        _expanded.validate_profile(self)
        super().validate()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _archive_source(output: Path, expected_sha256: str) -> None:
    source = Path(__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise LocalVLMError("adapter source changed after preflight")
    (output / "adapter-source.py").write_bytes(source)


def preflight(device: str = "mps") -> dict[str, Any]:
    """Import the pinned stack and inspect availability; no tensors/model/hub IO."""
    if device not in DEVICES or os.environ.get("WORLD_SIZE", "1") != "1":
        raise LocalVLMError("Use one local process with --device mps, cpu or cuda")
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise LocalVLMError("Disable PYTORCH_ENABLE_MPS_FALLBACK to expose unsupported operations")
    try:
        versions = {package: metadata.version(package) for package in STACK}
    except metadata.PackageNotFoundError as exc:
        raise LocalVLMError(
            "Missing local VLM dependencies; see requirements-vlm-local.in"
        ) from exc
    for package, expected in STACK.items():
        if versions[package].split("+")[0] != expected:
            raise LocalVLMError(f"Expected {package}=={expected}, found {versions[package]}")
    try:
        torch = import_module("torch")
        for package in STACK:
            import_module(package)
        trl = import_module("trl")
        sources = {
            name: _file_sha256(Path(inspect.getfile(getattr(trl, name)))) for name in TRL_SOURCES
        }
        import_module("datasets").Dataset.from_dict({"preflight": ["local-only"]})
    except (ImportError, OSError, RuntimeError, pickle.PicklingError) as exc:
        raise LocalVLMError(f"Pinned stack import failed: {type(exc).__name__}: {exc}") from exc
    if sources != TRL_SOURCES:
        raise LocalVLMError("Installed TRL trainer/config source differs from reviewed v1.16.0")
    if device == "mps" and not torch.backends.mps.is_available():
        raise LocalVLMError("MPS is unavailable; no model files have been opened")
    if device == "cuda" and not torch.cuda.is_available():
        raise LocalVLMError("CUDA is unavailable; no model files have been opened")
    environment = {
        "device": device,
        "python": sys.version.split()[0],
        "package_versions": versions,
        "trl_source_sha256": sources,
        "adapter_source_sha256": _file_sha256(Path(__file__)),
        "model_execution": "not tested by preflight",
        "model_downloads": False,
    }
    if device == "cuda":
        # CUDA-only key, so mps/cpu environments (and their resume equality) are unchanged.
        if torch.cuda.device_count() != 1 and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            raise LocalVLMError(
                "several CUDA devices are visible; pin one with CUDA_VISIBLE_DEVICES"
            )
        environment["cuda"] = {
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda": torch.version.cuda,
            "bf16_native": torch.cuda.get_device_capability(0) >= (8, 0),
        }
    return environment


def check_device_dtype(device: str, dtype: str, environment: dict, *, training: bool) -> None:
    """Refuse device/precision pairs that cannot run here, before any output is written.

    float16 is CUDA-only: its training path depends on CUDA autocast and GradScaler loss
    scaling. bfloat16 training needs compute capability >= 8.0, because TRL 1.16's
    chunked log-prob kernel (`torch.addmm(..., out_dtype=float32)` on bf16 inputs) refuses
    older GPUs (measured on an RTX 2080 Ti, sm_75, 2026-10-18).
    """
    if device not in DEVICES or dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("unknown device or dtype")
    if dtype == "float16" and device != "cuda":
        raise ValueError("float16 runs only on --device cuda (CUDA loss scaling)")
    if device == "cuda" and dtype == "bfloat16" and training:
        if not environment.get("cuda", {}).get("bf16_native"):
            raise ValueError(
                "bfloat16 training needs CUDA compute capability >= 8.0; use float16 here"
            )


def _set_threads(torch, threads: int) -> None:
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)


def _cuda_memory(device: str) -> dict | None:
    if device != "cuda":
        return None
    torch = import_module("torch")
    return {
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "scope": "torch caching allocator peaks for this process; excludes CUDA context",
    }


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("classification data spec requires schema_version 1")
    for key in ("benchmark", "question", "split_policy", "preprocessing"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"data spec requires {key}")
    if spec.get("preprocessing_id") not in (None, PREPROCESSING_ID):
        raise ValueError("unknown preprocessing_id; use the shared audited image transform")
    labels = spec.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 1000:
        raise ValueError("data spec requires the full ordered list of 2..1000 labels")
    if any(
        not isinstance(x, str)
        or x != x.strip()
        or not 1 <= len(x) <= 128
        or re.search(r"[<>\r\n\t]", x)
        for x in labels
    ) or len(set(labels)) != len(labels):
        raise ValueError("labels must be unique, trimmed canonical strings without markup")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data spec requires pinned sources")
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("url"), str)
            or not source["url"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))
        ):
            raise ValueError("every source requires url and immutable sha256")
    return spec


def load_classification(
    path: Path,
    split: str,
    labels: list[str],
    *,
    preprocessing_id: str | None = None,
    require_all_classes: bool = True,
) -> list[dict[str, Any]]:
    """Only immutable task-wide text enters the prompt; rows cannot override it."""
    path = Path(path).resolve()
    if preprocessing_id not in (None, PREPROCESSING_ID):
        raise ValueError("unknown preprocessing_id")
    rows, seen = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"id", "split", "image", "label"}:
            raise ValueError(f"{path.name}:{number}: require only id, split, image, label")
        if row["split"] != split or split not in {"train", "validation"}:
            raise ValueError(f"expected {split} split; training adapter never opens test")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("classification IDs must be unique nonempty strings")
        if row["label"] not in labels or not isinstance(row["image"], str):
            raise ValueError("unknown canonical label or invalid image path")
        row["image"] = str((path.parent / row["image"]).resolve())
        row["image_sha256"] = (
            aircraft_pixel_sha256(row["image"])
            if preprocessing_id == PREPROCESSING_ID
            else image_hash(row["image"])
        )
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("manifest is empty")
    if require_all_classes and {row["label"] for row in rows} != set(labels):
        raise ValueError(f"{split} must represent every declared class")
    return rows


def prompt_text(spec: dict[str, Any], response_format: str = "think_then_answer") -> str:
    if response_format in CLOSED_SET_RESPONSE_FORMATS or response_format == "think_answer":
        return _sweep_prompt_text(spec, response_format)
    if response_format == "answer_only":
        instruction = (
            "Return one label copied exactly from the allowed list on its own line. "
            "Do not add a prefix, quotes, explanations, or any text after that label."
        )
    elif response_format == "think_then_answer":
        instruction = (
            "Inspect the aircraft’s visible distinguishing features. First give a brief "
            "explanation of the visible details, enclosed by <think> and </think>. Then "
            "write one label copied exactly from the allowed list on its own line. Do not "
            "add a prefix, quotes, or any text after that label."
        )
    else:
        raise ValueError("unknown response_format")
    return (
        spec["question"]
        + "\nAllowed labels (the complete class list):\n"
        + "\n".join(spec["labels"])
        + "\n"
        + instruction
    )


def _sweep_prompt_text(spec: dict[str, Any], response_format: str) -> str:
    if response_format in TAGGED_RESPONSE_FORMATS:
        instruction = (
            "Inspect the aircraft's visible distinguishing features. First reason briefly "
            "inside <think> and </think>. Then give the final answer as <answer>LABEL</answer>, "
            "where LABEL is one label copied exactly from the allowed list. Write nothing "
            "after </answer>."
        )
    else:
        instruction = (
            "Return one label copied exactly from the allowed list on its own line. "
            "Do not add a prefix, quotes, explanations, or any text after that label."
        )
    if response_format in CLOSED_SET_RESPONSE_FORMATS:
        listing = (
            f"This is a closed-set task: the answer must be exactly one of the "
            f"{len(spec['labels'])} labels below, copied character for character. Labels are "
            "grouped as 'manufacturer / family: label | label'. A manufacturer name, a family "
            "name, or any aircraft name not in the list is never a valid answer; if the "
            "aircraft looks like an unlisted model or version, answer with the closest listed "
            "label.\nAllowed labels by manufacturer and family:\n"
            + "\n".join(aircraft_grouped_label_lines(spec["labels"]))
        )
    else:
        listing = "Allowed labels (the complete class list):\n" + "\n".join(spec["labels"])
    return spec["question"] + "\n" + listing + "\n" + instruction


def _parse_tagged_answer(text: str, labels: list[str]) -> str | None:
    """Answer-only grammar: free text, then exactly one final <answer>LABEL</answer>."""
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    match = re.fullmatch(r"(?s)(.*)<answer>([^<>\r\n]{1,256})</answer>", text)
    answer = match[2].strip() if match else None
    return answer if answer in labels else None


def output_grammar(response_format: str | None) -> str:
    return TAGGED_OUTPUT_GRAMMAR if response_format in TAGGED_RESPONSE_FORMATS else OUTPUT_GRAMMAR


def parse_answer(
    completion: Any,
    labels: list[str],
    *,
    allow_reasoning: bool = True,
    response_format: str | None = None,
) -> str | None:
    """`response_format=None` or a non-tagged format is exactly `parse_label` (whose
    source the named-baseline plan pins, so it stays untouched); a tagged format scores
    only its single final answer tag."""
    if response_format in TAGGED_RESPONSE_FORMATS:
        return _parse_tagged_answer(_completion_text(completion).strip(), labels)
    if response_format is not None and response_format not in RESPONSE_FORMATS:
        raise ValueError("unknown response_format")
    return parse_label(completion, labels, allow_reasoning=allow_reasoning)


def parse_label(completion: Any, labels: list[str], *, allow_reasoning: bool = True) -> str | None:
    # Angle brackets inside thinking are forbidden: no nested answers, injected
    # markup, multiple blocks, text after the answer, or answer-repair fallback.
    thinking = r"(?:<think>[^<>]{0,32768}</think>\s*)?" if allow_reasoning else ""
    match = re.fullmatch(
        thinking + r"(?:<answer>([^<>\r\n]+)</answer>|([^<>\r\n]+))",
        _completion_text(completion).strip(),
    )
    answer = (match[1] or match[2]) if match else None
    return answer if answer in labels else None


def training_records(
    rows: list[dict], spec: dict, mode: str, response_format: str = "think_then_answer"
) -> list[dict]:
    prompt = prompt_text(spec, response_format)
    records = []
    for row in rows:
        record = {
            "id": row["id"],
            "image": row["image"],
            "label": row["label"],
            "prompt": [{"role": "user", "content": prompt}],
        }
        if mode == "sft":
            target = (
                f"<answer>{row['label']}</answer>"
                if response_format in TAGGED_RESPONSE_FORMATS
                else row["label"]
            )
            record["completion"] = [{"role": "assistant", "content": target}]
        records.append(record)
    return records


def _dataset(rows: list[dict], spec: dict, mode: str, profile: Profile, *, lazy_decode=True) -> Any:
    datasets = import_module("datasets")
    dataset = datasets.Dataset.from_list(
        training_records(rows, spec, mode, profile.response_format)
    ).cast_column("image", datasets.Image(mode="RGB", decode=False))
    if not lazy_decode:
        return dataset

    def decode(batch):
        batch["image"] = [_load_rgb(entry["path"], spec) for entry in batch["image"]]
        return batch

    return dataset.with_transform(decode, columns=["image"], output_all_columns=True)


def _load_rgb(path: str | Path, spec: dict):
    """One image path for audit, rollout, SFT, and inference; no transformed cache."""
    if spec.get("preprocessing_id") == PREPROCESSING_ID:
        return load_aircraft_rgb(path)
    pil = import_module("PIL.Image")
    with pil.open(path) as source:
        return source.convert("RGB")


def _classification_identity(path: Path, rows: list[dict]) -> dict:
    identity = _dataset_identity(path, rows)
    pixels = {row["id"]: row["image_sha256"] for row in rows}
    identity["model_visible_pixel_sha256"] = hashlib.sha256(
        json.dumps(pixels, sort_keys=True).encode()
    ).hexdigest()
    return identity


LABEL_HIERARCHY_SCHEMA = "vision-rl-label-hierarchy-v1"


def load_label_hierarchy(path: Path, labels: list[str]) -> dict[str, Any]:
    """TRAIN-derived variant -> family -> manufacturer map for partial-credit rewards.

    Refuses anything not built from the TRAIN split alone, any label set that differs
    from the task vocabulary, and any family claimed by two manufacturers.
    """
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get("schema") != LABEL_HIERARCHY_SCHEMA:
        raise ValueError(f"label hierarchy requires schema {LABEL_HIERARCHY_SCHEMA}")
    source = value.get("source")
    if not isinstance(source, dict) or source.get("split") != "train":
        raise ValueError("label hierarchy must be derived from TRAIN annotations only")
    if value.get("levels") != ["variant", "family", "manufacturer"]:
        raise ValueError("label hierarchy levels must be variant, family, manufacturer")
    mapping = value.get("labels")
    if not isinstance(mapping, dict) or set(mapping) != set(labels):
        raise ValueError("label hierarchy must cover exactly the task's label vocabulary")
    family_owner: dict[str, str] = {}
    for label in labels:
        entry = mapping[label]
        if not isinstance(entry, dict) or set(entry) != {"family", "manufacturer"}:
            raise ValueError("each hierarchy entry requires exactly family and manufacturer")
        family, manufacturer = entry["family"], entry["manufacturer"]
        if not all(isinstance(x, str) and x.strip() == x and x for x in (family, manufacturer)):
            raise ValueError("hierarchy family/manufacturer must be trimmed nonempty strings")
        if family_owner.setdefault(family, manufacturer) != manufacturer:
            raise ValueError(f"family {family!r} belongs to more than one manufacturer")
    return {
        "family": {label: mapping[label]["family"] for label in labels},
        "manufacturer": {label: mapping[label]["manufacturer"] for label in labels},
        "sha256": _file_sha256(path),
    }


def reward_shaping(
    hierarchy: dict[str, Any],
    *,
    family_credit: float,
    manufacturer_credit: float,
    invalid_reward: float = 0.0,
) -> dict[str, Any]:
    """Hierarchical partial credit: exact 1, same family, same manufacturer, wrong 0, invalid."""
    for name, value in (
        ("family_credit", family_credit),
        ("manufacturer_credit", manufacturer_credit),
        ("invalid_reward", invalid_reward),
    ):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= manufacturer_credit <= family_credit < 1:
        raise ValueError("require 0 <= manufacturer_credit <= family_credit < 1")
    if not -1 <= invalid_reward <= 0:
        raise ValueError("invalid_reward must be in [-1, 0]")
    return {
        "kind": "hierarchical-partial-credit-v1",
        "family": hierarchy["family"],
        "manufacturer": hierarchy["manufacturer"],
        "hierarchy_sha256": hierarchy["sha256"],
        "family_credit": float(family_credit),
        "manufacturer_credit": float(manufacturer_credit),
        "invalid_reward": float(invalid_reward),
    }


def shaping_identity(shaping: dict[str, Any] | None) -> dict[str, Any] | None:
    """The settings-identity view of a shaping (no per-label maps; the file hash binds them)."""
    if shaping is None:
        return None
    return {
        key: shaping[key]
        for key in (
            "kind",
            "hierarchy_sha256",
            "family_credit",
            "manufacturer_credit",
            "invalid_reward",
        )
    }


def _shaping_from_options(
    spec, label_hierarchy, family_credit, manufacturer_credit, invalid_reward, *, mode
) -> dict[str, Any] | None:
    if label_hierarchy is None:
        if (family_credit, manufacturer_credit, invalid_reward) != (None, None, None):
            raise ValueError("reward credits require label_hierarchy")
        return None
    if mode != "grpo":
        raise ValueError("hierarchical reward shaping applies only to GRPO")
    if family_credit is None or manufacturer_credit is None:
        raise ValueError("label_hierarchy requires explicit family_credit and manufacturer_credit")
    return reward_shaping(
        load_label_hierarchy(Path(label_hierarchy), spec["labels"]),
        family_credit=family_credit,
        manufacturer_credit=manufacturer_credit,
        invalid_reward=0.0 if invalid_reward is None else invalid_reward,
    )


def shaped_reward(predicted: str | None, expected: str, shaping: dict[str, Any]) -> float:
    """Partial credit for one completion; ``predicted`` is the exact-EOS canonical answer."""
    if predicted is None:
        return shaping["invalid_reward"]
    if predicted == expected:
        return 1.0
    if shaping["family"][predicted] == shaping["family"][expected]:
        return shaping["family_credit"]
    if shaping["manufacturer"][predicted] == shaping["manufacturer"][expected]:
        return shaping["manufacturer_credit"]
    return 0.0


def _complete_think_block(text: str, response_format: str | None) -> bool:
    if response_format in TAGGED_RESPONSE_FORMATS:
        text = text.strip()
        return text.startswith("<think>") and 0 <= text.find("</think>") < text.find("<answer>")
    return bool(re.match(r"<think>[^<>]{0,32768}</think>", text.strip()))


def make_reward(
    labels: list[str],
    group_size: int,
    eos_ids: list[int],
    journal: Path,
    *,
    allow_reasoning: bool = True,
    journal_io_context=None,
    observe_prompt=None,
    shaping: dict[str, Any] | None = None,
    response_format: str | None = None,
):
    """Exact task reward with journaled groups; never inject synthetic variance.

    ``shaping`` (default off) replaces the returned reward with hierarchical partial
    credit (see ``shaped_reward``). Exact-match statistics stay exact; ``mixed_groups``
    counts groups whose returned rewards differ, i.e. that carry a GRPO advantage.
    """
    if shaping is not None:
        if set(shaping["family"]) != set(labels):
            raise ValueError("reward shaping hierarchy must cover exactly the labels")
        stats_extra = {"exact_mixed_groups": 0, "shaped_reward_sum": 0.0}
    else:
        stats_extra = {}
    stats = {"groups": 0, "mixed_groups": 0, "all_correct_groups": 0, "all_zero_groups": 0}
    stats.update(
        unique_prompt_ids=0,
        prompt_presentations=0,
        sampled_completions=0,
        generated_tokens=0,
        terminated_completions=0,
        nonterminated_completions=0,
        complete_think_blocks=0,
        task_correct_with_think=0,
        task_correct_without_think=0,
        canonical_completions=0,
        recognition_mixed_groups=0,
    )
    stats.update(stats_extra)
    seen_ids = set()

    def exact_label_accuracy(completions, label, id, completion_ids, **_):
        if not len(completions) == len(label) == len(id) == len(completion_ids):
            raise ValueError("reward batch lengths differ")
        if len(completions) % group_size:
            raise ValueError("reward batch does not contain complete groups")
        exact = [
            float(
                bool(tokens)
                and tokens[-1] in eos_ids
                and parse_answer(
                    text,
                    labels,
                    allow_reasoning=allow_reasoning,
                    response_format=response_format,
                )
                == expected
            )
            for text, expected, tokens in zip(completions, label, completion_ids, strict=True)
        ]
        if shaping is None:
            rewards = exact
        else:
            rewards = [
                shaped_reward(
                    parse_answer(
                        text,
                        labels,
                        allow_reasoning=allow_reasoning,
                        response_format=response_format,
                    )
                    if tokens and tokens[-1] in eos_ids
                    else None,
                    expected,
                    shaping,
                )
                for text, expected, tokens in zip(completions, label, completion_ids, strict=True)
            ]
        serialized = []
        for start in range(0, len(rewards), group_size):
            stop = start + group_size
            if len(set(id[start:stop])) != 1 or len(set(label[start:stop])) != 1:
                raise ValueError("TRL reward group mixes different examples")
            if observe_prompt is not None:
                observe_prompt(id[start])
            group = exact[start:stop]
            returned = rewards[start:stop]
            mixed = min(returned) != max(returned)
            texts = [_completion_text(x) for x in completions[start:stop]]
            terminated = [
                bool(tokens) and tokens[-1] in eos_ids for tokens in completion_ids[start:stop]
            ]
            thinking = [_complete_think_block(text, response_format) for text in texts]
            canonical = [
                parse_answer(
                    text,
                    labels,
                    allow_reasoning=allow_reasoning,
                    response_format=response_format,
                )
                if ended
                else None
                for text, ended in zip(texts, terminated, strict=True)
            ]
            recognition_mixed = any(group) and any(
                value is not None and value != label[start] for value in canonical
            )
            lengths = [len(tokens) for tokens in completion_ids[start:stop]]
            seen_ids.add(id[start])
            stats["groups"] += 1
            stats["mixed_groups"] += int(mixed)
            stats["all_correct_groups"] += int(all(group))
            stats["all_zero_groups"] += int(not any(group))
            stats["unique_prompt_ids"] = len(seen_ids)
            stats["prompt_presentations"] += 1
            stats["sampled_completions"] += len(group)
            stats["generated_tokens"] += sum(lengths)
            stats["terminated_completions"] += sum(terminated)
            stats["nonterminated_completions"] += len(group) - sum(terminated)
            stats["complete_think_blocks"] += sum(thinking)
            stats["canonical_completions"] += sum(value is not None for value in canonical)
            stats["recognition_mixed_groups"] += int(recognition_mixed)
            stats["task_correct_with_think"] += sum(
                int(value) for value, thought in zip(group, thinking, strict=True) if thought
            )
            stats["task_correct_without_think"] += sum(
                int(value) for value, thought in zip(group, thinking, strict=True) if not thought
            )
            extra = {}
            if shaping is not None:
                stats["exact_mixed_groups"] += int(min(group) != max(group))
                stats["shaped_reward_sum"] += sum(returned)
                extra = {"exact_rewards": group}
            serialized.append(
                json.dumps(
                    {
                        "id": id[start],
                        "label": label[start],
                        "rewards": returned,
                        **extra,
                        "mixed": mixed,
                        "recognition_mixed": recognition_mixed,
                        "canonical_predictions": canonical,
                        "completions": texts,
                        "completion_ids": completion_ids[start:stop],
                        "completion_tokens": lengths,
                        "terminated": terminated,
                        "complete_think_block": thinking,
                    }
                )
                + "\n"
            )
        with (
            journal_io_context() if journal_io_context else nullcontext(),
            journal.open("a", encoding="utf-8") as handle,
        ):
            handle.write("".join(serialized))
            handle.flush()
        return rewards

    exact_label_accuracy.stats = stats
    return exact_label_accuracy


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_grpo_overrides(overrides: dict | None) -> dict:
    """Allowlisted TRL 1.16 GRPOConfig knobs; refuse unknown names and unsafe values.

    Note: with num_iterations == 1 every update is on-policy (ratio == 1), so
    epsilon/epsilon_high/delta clipping only binds when num_iterations > 1.
    """
    if overrides is None:
        return {}
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("grpo_overrides must be a nonempty dict or None")
    unknown = sorted(set(overrides) - set(GRPO_KNOBS))
    if unknown:
        raise ValueError("unknown GRPO knob(s): " + ", ".join(unknown))
    merged = {**GRPO_KNOBS, **overrides}
    epsilon = merged["epsilon"]
    if not _finite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and in (0, 1)")
    high = merged["epsilon_high"]
    if high is not None and (
        not _finite(high)
        or not 0 < high <= 10
        or (merged["loss_type"] != "cispo" and high < epsilon)
    ):
        raise ValueError("epsilon_high must be finite, in (0, 10] and >= epsilon (clip-higher)")
    if merged["delta"] is not None and (not _finite(merged["delta"]) or merged["delta"] <= 1):
        raise ValueError("delta must be finite and > 1")
    if merged["loss_type"] not in GRPO_LOSS_TYPES:
        raise ValueError("loss_type must be one of " + ", ".join(GRPO_LOSS_TYPES))
    if merged["scale_rewards"] not in ("group", "batch", "none"):
        raise ValueError("scale_rewards must be group, batch or none")
    if merged["importance_sampling_level"] not in ("token", "sequence"):
        raise ValueError("importance_sampling_level must be token or sequence")
    if merged["loss_type"] == "luspo" and merged["importance_sampling_level"] != "sequence":
        raise ValueError("luspo requires importance_sampling_level=sequence")
    if merged["loss_type"] == "vespo" and merged["importance_sampling_level"] != "token":
        raise ValueError("vespo requires importance_sampling_level=token")
    if type(merged["mask_truncated_completions"]) is not bool:
        raise ValueError("mask_truncated_completions must be a bool")
    if type(merged["num_iterations"]) is not int or not 1 <= merged["num_iterations"] <= 8:
        raise ValueError("num_iterations must be an integer in 1..8")
    quantile = merged["top_entropy_quantile"]
    if not _finite(quantile) or not 0 < quantile <= 1:
        raise ValueError("top_entropy_quantile must be finite and in (0, 1]")
    return {key: overrides[key] for key in sorted(overrides)}


def trainer_kwargs(
    mode: str,
    output: Path,
    profile: Profile,
    *,
    device: str,
    max_steps: int,
    learning_rate: float,
    seed: int,
    save_steps: int,
    beta: float = 0.04,
    order: str = "random",
    grpo_overrides: dict | None = None,
) -> dict:
    """Explicit comparable defaults; one unique prompt per optimizer update.

    `grpo_overrides` replaces allowlisted legacy GRPO values (see GRPO_KNOBS)."""
    if order not in {"random", "sequential"} or (order == "sequential" and mode != "sft"):
        raise ValueError("order must be random, or sequential for SFT")
    overrides = validate_grpo_overrides(grpo_overrides)
    if overrides and mode != "grpo":
        raise ValueError("grpo_overrides apply only to GRPO")
    common = dict(
        output_dir=str(output),
        max_steps=max_steps,
        seed=seed,
        data_seed=seed,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=False,
        # float16 (CUDA only) turns on accelerate's fp16 autocast + GradScaler loss scaling.
        fp16=profile.dtype == "float16",
        use_cpu=device == "cpu",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=False,
        logging_steps=1,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        torch_compile=False,
        disable_tqdm=True,
    )
    if mode == "grpo":
        common.update(
            gradient_accumulation_steps=profile.num_generations,
            generation_batch_size=profile.num_generations,
            num_generations=profile.num_generations,
            num_generations_eval=1,
            num_iterations=1,
            max_completion_length=profile.max_completion_length,
            temperature=profile.temperature,
            top_p=1.0,
            top_k=0,
            use_vllm=False,
            use_transformers_continuous_batching=False,
            beta=beta,
            sync_ref_model=False,
            loss_type="grpo",
            scale_rewards="group",
            epsilon=0.2,
            importance_sampling_level="token",
            mask_truncated_completions=True,
            disable_dropout=True,
        )
        common.update(overrides)
    elif mode == "sft":
        common.update(
            gradient_accumulation_steps=1,
            max_length=None,
            packing=False,
            padding_free=False,
            completion_only_loss=True,
            assistant_only_loss=False,
            loss_type="nll",
        )
        if order == "sequential":
            # Replay the given manifest order; no data_seed shuffle. Further epochs cycle it.
            common["train_sampling_strategy"] = "sequential"
    else:
        raise ValueError("mode must be grpo or sft")
    return common


def _verified_model_directory(directory: Path, model: str) -> dict:
    """Verify the single-copy downloader's immutable manifest, sizes, and file hashes."""
    source_path = directory / "model-source.json"
    marker = json.loads((directory / ".pinned-download.json").read_text())
    spec = json.loads(source_path.read_text())
    if (
        marker["repository"] != model
        or marker["revision"] != MODEL_REVISIONS[model]
        or marker["manifest_sha256"] != _file_sha256(source_path)
        or spec["model_repository"] != model
        or spec["revision"] != MODEL_REVISIONS[model]
    ):
        raise ValueError("local snapshot provenance differs from the pinned model")
    observed = {}
    for entry in spec["files"]:
        name = entry["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("snapshot file escapes model directory")
        path = directory / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"snapshot file size/type differs: {name}")
        digest = _file_sha256(path)
        if entry.get("sha256"):
            matches = digest == entry["sha256"]
        else:
            if path.stat().st_size > 32 * 1024**2:
                raise ValueError("large snapshot files require SHA256 pins")
            header = f"blob {entry['size_bytes']}\0".encode()
            matches = hashlib.sha1(header + path.read_bytes()).hexdigest() == entry["git_blob_id"]
        if not matches:
            raise ValueError(f"snapshot hash differs: {name}")
        observed[name] = digest
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()).issubset(observed):
        raise ValueError("snapshot index refers to unverified weights")
    return {
        "manifest_sha256": _file_sha256(source_path),
        "file_sha256": observed,
        "total_bytes": sum(entry["size_bytes"] for entry in spec["files"]),
    }


def _processor(model_directory: Path, profile: Profile):
    transformers = import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_directory),
        trust_remote_code=False,
        local_files_only=True,
        padding_side="left",
        min_pixels=profile.min_pixels,
        max_pixels=profile.max_pixels,
    )
    if not hasattr(processor, "image_processor"):
        raise LocalVLMError("selected processor has no image processor")
    return processor


def _inputs(processor, row: dict, spec: dict, profile: Profile):
    image = _load_rgb(row["image"], spec)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text(spec, profile.response_format)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(images=[image], text=[text], return_tensors="pt", add_special_tokens=False)


def _audit_processor(processor, rows: list[dict], spec: dict, profile: Profile) -> dict:
    # Fail on long prompts instead of truncating away image tokens. CPU processing only.
    lengths = [int(_inputs(processor, row, spec, profile)["input_ids"].shape[-1]) for row in rows]
    gold_lengths = [
        len(processor.tokenizer.encode(f"<answer>{label}</answer>", add_special_tokens=False)) + 1
        for label in spec["labels"]
    ]
    if max(lengths) > profile.max_prompt_tokens:
        raise ValueError("untruncated prompt exceeds max_prompt_tokens; revise frozen profile")
    if max(gold_lengths) > profile.max_completion_length:
        raise ValueError("a canonical answer plus EOS exceeds max_completion_length")
    return {
        "max_prompt_tokens_observed": max(lengths),
        "max_gold_tokens_plus_eos": max(gold_lengths),
        "chat_template_sha256": hashlib.sha256(
            json.dumps(processor.chat_template, sort_keys=True).encode()
        ).hexdigest(),
        "image_processor": processor.image_processor.to_dict(),
    }


def classification_metrics(
    rows: list[dict],
    predictions: dict[str, Any],
    labels: list[str],
    *,
    allow_reasoning: bool = True,
    decoding_protocol: str = "greedy/pass@1",
    response_format: str | None = None,
) -> dict:
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("evaluation requires nonempty unique IDs")
    if any(row["label"] not in labels for row in rows):
        raise ValueError("evaluation contains an unknown canonical label")
    if not isinstance(decoding_protocol, str) or not decoding_protocol.strip():
        raise ValueError("decoding_protocol must describe the actual generation protocol")
    if set(predictions) != {r["id"] for r in rows}:
        raise ValueError("predictions must cover exactly the evaluation IDs")
    counts, correct = Counter(), Counter()
    invalid = 0
    for row in rows:
        response = predictions[row["id"]]
        if isinstance(response, dict):
            predicted = (
                parse_answer(
                    response["response"],
                    labels,
                    allow_reasoning=allow_reasoning,
                    response_format=response_format,
                )
                if response["terminated"]
                else None
            )
        else:
            predicted = parse_answer(
                response, labels, allow_reasoning=allow_reasoning, response_format=response_format
            )
        counts[row["label"]] += 1
        correct[row["label"]] += int(predicted == row["label"])
        invalid += int(predicted is None)
    per_class = {label: correct[label] / counts[label] for label in counts}
    return {
        "n": len(rows),
        "accuracy": sum(correct.values()) / len(rows),
        "mean_class_accuracy": sum(per_class.values()) / len(per_class),
        "invalid_answers": invalid,
        "per_class_accuracy": per_class,
        "protocol": output_grammar(response_format) + "/" + decoding_protocol,
    }


def _format_kwargs(profile: Profile) -> dict:
    """Legacy formats pass nothing, so their calls stay exactly as before."""
    if profile.response_format in LEGACY_RESPONSE_FORMATS:
        return {}
    return {"response_format": profile.response_format}


def _evaluate(policy, processor, rows, spec, profile, device, *, journal=None) -> tuple[dict, dict]:
    torch = import_module("torch")
    predictions = {}
    was_training = policy.training
    policy.eval()
    eos = policy.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    eos = set(eos + [processor.tokenizer.eos_token_id])
    with torch.inference_mode():
        for row in rows:
            started = time.monotonic()
            inputs = _inputs(processor, row, spec, profile).to(device)
            tokens = policy.generate(
                **inputs,
                max_new_tokens=profile.max_completion_length,
                do_sample=False,
                use_cache=True,
            )
            tail = tokens[:, inputs["input_ids"].shape[1] :]
            predictions[row["id"]] = {
                "response": processor.batch_decode(tail, skip_special_tokens=True)[0],
                "raw_response": processor.batch_decode(tail, skip_special_tokens=False)[0],
                "terminated": bool(tail.shape[-1] and int(tail[0, -1].item()) in eos),
                "completion_ids": tail[0].tolist(),
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(tail.shape[-1]),
                "latency_seconds": time.monotonic() - started,
            }
            if journal is not None:
                _write(journal, predictions)
    policy.train(was_training)
    return classification_metrics(
        rows,
        predictions,
        spec["labels"],
        allow_reasoning=True,
        **_format_kwargs(profile),
    ), predictions


def _task_group_generation_config(policy, processor, profile):
    """Released TRL ordinary-generate settings, for frozen-policy diagnostics only."""
    eos = policy.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if processor.tokenizer.eos_token_id not in eos:
        eos.insert(0, processor.tokenizer.eos_token_id)
    return import_module("transformers").GenerationConfig(
        max_new_tokens=profile.max_completion_length,
        do_sample=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=eos,
        temperature=profile.temperature,
        top_p=1.0,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        cache_implementation=None,
        suppress_tokens=[policy.config.image_token_id],
        disable_compile=True,
    )


def _sample_task_groups(policy, processor, rows, spec, profile, device, output, reward, config):
    """Serial independent draws; no weights, optimizer, advantage or loss update."""
    torch = import_module("torch")
    groups = {}
    policy.eval()
    with torch.inference_mode():
        for row in rows:
            inputs = _inputs(processor, row, spec, profile).to(device)
            samples = []
            groups[row["id"]] = samples
            for _ in range(profile.num_generations):
                started = time.monotonic()
                tokens = policy.generate(**inputs, generation_config=config)
                tail = tokens[0, inputs["input_ids"].shape[-1] :].tolist()
                # Match released TRL's first-EOS-inclusive completion mask.
                for index, token in enumerate(tail):
                    if token in config.eos_token_id:
                        tail = tail[: index + 1]
                        break
                samples.append(
                    {
                        "response": processor.tokenizer.decode(tail, skip_special_tokens=True),
                        "raw_response": processor.tokenizer.decode(tail, skip_special_tokens=False),
                        "completion_ids": tail,
                        "terminated": bool(tail and tail[-1] in config.eos_token_id),
                        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                        "completion_tokens": len(tail),
                        "latency_seconds": time.monotonic() - started,
                    }
                )
                _write(output / "sampled_completions.json", groups)
            reward(
                completions=[sample["response"] for sample in samples],
                completion_ids=[sample["completion_ids"] for sample in samples],
                label=[row["label"]] * profile.num_generations,
                id=[row["id"]] * profile.num_generations,
            )
    return groups


def predict_train_subset(
    manifest: Path,
    data_spec: Path,
    output: Path,
    *,
    model_directory: Path,
    model: str = DEFAULT_MODEL,
    device: str = "mps",
    profile: Profile = Profile(),
    sample_groups: bool = False,
    seed: int = 42,
    init_adapter: Path | None = None,
    expanded_initial_path: Path | None = None,
    threads: int = DEFAULT_THREADS,
    precision_transfer: str | None = None,
) -> dict:
    """Frozen-policy pass@1 or sampled groups on <=32 TRAIN images; no fit/test.

    This permits a prospectively selected subset of classes while retaining the
    complete task vocabulary. Its mean-class score covers only represented
    classes and is explicitly not a held-out benchmark result.

    ``expanded_initial_path`` is this machine's copy of the approved expanded INIT,
    used to re-verify a registered expanded endpoint (default: the trainer's path).
    """
    profile.validate()
    if expanded_initial_path is not None and not _expanded.is_expanded(profile):
        raise ValueError("expanded_initial_path applies only to the expanded profile")
    if type(sample_groups) is not bool or type(seed) is not int or seed < 0:
        raise ValueError("sample_groups must be boolean and seed a nonnegative integer")
    if model not in MODEL_REVISIONS:
        raise ValueError("choose one of the explicitly pinned Qwen models")
    spec = load_spec(data_spec)
    rows = load_classification(
        manifest,
        "train",
        spec["labels"],
        preprocessing_id=spec.get("preprocessing_id"),
        require_all_classes=False,
    )
    if len(rows) > 32:
        raise ValueError("TRAIN feasibility inference is limited to 32 images")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if precision_transfer is not None and (
        not init_adapter or precision_transfer not in INFERENCE_PRECISION_TRANSFERS
    ):
        raise ValueError("precision_transfer needs a saved adapter and a known transfer")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output is not empty; use a new directory")
    environment = preflight(device)
    check_device_dtype(device, profile.dtype, environment, training=False)
    model_directory = Path(model_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prompt = prompt_text(spec, profile.response_format)
    record = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_revision": MODEL_REVISIONS[model],
        "processor_model": model,
        "processor_revision": MODEL_REVISIONS[model],
        "profile": asdict(profile),
        "output_grammar": output_grammar(profile.response_format),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "data_spec_sha256": _file_sha256(Path(data_spec)),
        "data_identity": _classification_identity(Path(manifest), rows),
        "image_transform_source_sha256": (
            _file_sha256(Path(inspect.getfile(load_aircraft_rgb)))
            if spec.get("preprocessing_id") == PREPROCESSING_ID
            else None
        ),
        "declared_classes": len(spec["labels"]),
        "represented_classes": len({row["label"] for row in rows}),
        "sources": spec["sources"],
        "environment": environment,
        "generation": {
            "do_sample": sample_groups,
            "max_new_tokens": profile.max_completion_length,
            "samples_per_image": profile.num_generations if sample_groups else 1,
            "seed": seed if sample_groups else None,
            "concurrent_generation_batch_size": 1,
        },
        "claim_scope": "TRAIN subset feasibility only; no training or held-out benchmark claim",
        "optimizer_steps": 0,
    }
    if threads != DEFAULT_THREADS:
        record["torch_threads"] = threads
    if _expanded.is_expanded(profile) and not init_adapter:
        raise ValueError("expanded inference requires an approved numeric artifact")
    initial = (
        _inference_adapter_identity(
            Path(init_adapter).resolve(),
            record,
            profile,
            initial_path=expanded_initial_path,
            **({"precision_transfer": precision_transfer} if precision_transfer else {}),
        )
        if init_adapter
        else None
    )
    record["initial_adapter"] = initial
    record["policy_identity"] = "saved_adapter" if initial else "pinned_base"
    metadata_path = output / (
        "train_subset_groups.json" if sample_groups else "train_subset_inference.json"
    )
    _write(metadata_path, record)
    _archive_source(output, environment["adapter_source_sha256"])
    (output / "prompt.txt").write_text(prompt + "\n")
    reward = None
    try:
        torch = import_module("torch")
        # An explicit --threads wins; otherwise VLM_TORCH_THREADS (or the historical 2).
        _set_threads(torch, threads if threads != DEFAULT_THREADS else _torch_threads())
        transformers = import_module("transformers")
        if sample_groups:
            transformers.set_seed(seed)
        record["model_source"] = _verified_model_directory(model_directory, model)
        processor = _processor(model_directory, profile)
        record["processor_audit"] = _audit_processor(processor, rows, spec, profile)
        _write(metadata_path, record)
        policy = transformers.AutoModelForImageTextToText.from_pretrained(
            str(model_directory),
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            dtype=getattr(torch, profile.dtype),
            attn_implementation="sdpa",
        )
        if policy.config.model_type not in {"qwen2_vl", "qwen2_5_vl"}:
            raise LocalVLMError("unexpected cached model architecture")
        if initial:
            policy = (
                _expanded.attach_numeric(
                    policy,
                    Path(init_adapter).resolve(),
                    expected_weights_sha256=initial["weights_sha256"],
                    is_trainable=False,
                    initial_path=expanded_initial_path,
                )
                if _expanded.is_expanded(profile)
                else import_module("peft").PeftModel.from_pretrained(
                    policy, str(init_adapter), is_trainable=False
                )
            )
        policy.requires_grad_(False).to(device).eval()
        if sample_groups:
            config = _task_group_generation_config(policy, processor, profile)
            record["generation_config"] = config.to_dict()
            record["sampling_scope"] = (
                "serial frozen-policy draws with released-TRL generation settings; "
                "no proof of batched generation memory or optimizer execution"
            )
            _write(metadata_path, record)
            reward = make_reward(
                spec["labels"],
                profile.num_generations,
                config.eos_token_id,
                output / "reward_groups.jsonl",
                allow_reasoning=True,
                **_format_kwargs(profile),
            )
            _sample_task_groups(
                policy, processor, rows, spec, profile, device, output, reward, config
            )
            record["reward_audit"] = reward.stats
        else:
            metrics, _ = _evaluate(
                policy,
                processor,
                rows,
                spec,
                profile,
                device,
                journal=output / "predictions.json",
            )
            record["metrics"] = metrics
        record.update(status="completed", elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        if sample_groups:
            sampled_path = output / "sampled_completions.json"
            samples = json.loads(sampled_path.read_text()) if sampled_path.exists() else {}
            record["unfinished_samples_per_id"] = {
                row["id"]: profile.num_generations - len(samples.get(row["id"], [])) for row in rows
            }
            record["reward_audit"] = reward.stats if reward else None
        else:
            predictions_path = output / "predictions.json"
            predictions = (
                json.loads(predictions_path.read_text()) if predictions_path.exists() else {}
            )
            record["completed_prediction_ids"] = list(predictions)
            for row in rows:
                predictions.setdefault(
                    row["id"],
                    {
                        "response": "",
                        "terminated": False,
                        "error": f"Generation not completed: {type(exc).__name__}",
                    },
                )
            _write(predictions_path, predictions)
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        _write(metadata_path, record)
        raise
    _write(metadata_path, record)
    return record


def sample_train_groups(manifest: Path, **kwargs) -> dict:
    """Inference-only real-task reward support; always use a fresh output path."""
    return predict_train_subset(manifest, sample_groups=True, **kwargs)


def _gold_nll(policy, processor, rows, spec, profile, device) -> float:
    torch = import_module("torch")
    collator = import_module("trl.trainer.sft_trainer").DataCollatorForVisionLanguageModeling(
        processor,
        completion_only_loss=True,
        max_length=None,
    )
    total_loss, total_tokens = 0.0, 0
    was_training = policy.training
    policy.eval()
    with torch.inference_mode():
        for record in training_records(rows, spec, "sft", profile.response_format):
            record["image"] = _load_rgb(record["image"], spec)
            batch = {k: v.to(device) for k, v in collator([record]).items()}
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            loss = float(policy(**batch).loss.item())
            if not count or not math.isfinite(loss):
                raise LocalVLMError("nonfinite gold loss or empty completion mask")
            total_loss += loss * count
            total_tokens += count
    policy.train(was_training)
    return total_loss / total_tokens


# Released TRL 1.16 group advantage: (r - mean) / (nanstd + 1e-4), where nanstd is
# Bessel-corrected (count / (count - 1)). The expected-gradient weight below uses both.
TRL_ADVANTAGE_EPS = 1e-4
_TOKEN_ALIGNED_KEYS = ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids")
_NOT_MODEL_INPUT_KEYS = ("labels", "completion_mask", "use_cache")


def group_useful_probability(p: float, group_size: int = 4) -> float:
    """u(p): probability that a group of G exact-match rewards is mixed (nonzero advantage)."""
    if not math.isfinite(p) or not 0 <= p <= 1 or type(group_size) is not int or group_size < 2:
        raise ValueError("p must be in [0, 1] and group_size an integer >= 2")
    return 1.0 - p**group_size - (1.0 - p) ** group_size


def expected_gradient_weight(
    p: float, group_size: int = 4, eps: float = TRL_ADVANTAGE_EPS
) -> float:
    """w_G(p): expected GRPO update on an exact-match reward as a multiple of the gold SFT gradient.

    Enumerates k = 0..G correct samples under TRL's group-std advantage. A correct
    sample's score is the gold gradient g; a wrong sample's conditional expected score
    is -p g / (1 - p). Summing A_i over the group gives, per k,
    C(G,k) p^k (1-p)^(G-k-1) k (G-k) / (G (s_k + eps)); k = 0 and k = G have zero
    advantage. KL, token-length normalization and the 1/G loss mean are ignored; the
    last cancels in the EG-SFT ratio w_G(p) / w_G(0.5).
    """
    if not math.isfinite(p) or not 0 <= p <= 1 or type(group_size) is not int or group_size < 2:
        raise ValueError("p must be in [0, 1] and group_size an integer >= 2")
    total = 0.0
    for k in range(1, group_size):
        mean = k / group_size
        std = math.sqrt(group_size / (group_size - 1) * mean * (1 - mean))
        total += (
            math.comb(group_size, k)
            * p**k
            * (1 - p) ** (group_size - k - 1)
            * k
            * (group_size - k)
            / (group_size * (std + eps))
        )
    return total


def egsft_weight(p: float, group_size: int = 4) -> float:
    """EG-SFT per-sample loss multiplier w_G(p) / w_G(0.5); a constant, never differentiated."""
    return expected_gradient_weight(p, group_size) / expected_gradient_weight(0.5, group_size)


def _completion_logprob(policy, prompt_inputs, completion_ids, *, temperature, suppress_tokens):
    """Sum log-probability of one completion under the sampler's logit transforms.

    Same order as `_sampler_parity`: divide by temperature, then remove suppressed
    tokens (TRL suppresses the image token while sampling) and renormalize.
    """
    torch = import_module("torch")
    prompt_ids = prompt_inputs["input_ids"]
    if prompt_ids.shape[0] != 1 or not completion_ids:
        raise LocalVLMError("gold scoring requires one prompt and a nonempty completion")
    completion = torch.tensor(
        [list(completion_ids)], dtype=prompt_ids.dtype, device=prompt_ids.device
    )
    forward = dict(prompt_inputs)
    forward["input_ids"] = torch.cat((prompt_ids, completion), dim=1)
    forward["attention_mask"] = torch.cat(
        (prompt_inputs["attention_mask"], torch.ones_like(completion)), dim=1
    )
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in forward:
            forward[key] = torch.cat((forward[key], torch.zeros_like(completion)), dim=1)
    logits = policy(**forward, use_cache=False, logits_to_keep=completion.shape[1] + 1).logits
    logits = logits[:, :-1].float() / temperature
    for token in suppress_tokens:
        logits[..., token] = -torch.inf
    logps = logits.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)
    value = float(logps.sum().item())
    if math.isnan(value) or value > 0:
        raise LocalVLMError("invalid gold completion log-probability")
    return value


def _batch_gold_probability(
    policy, batch, *, eos_token_ids, suppress_tokens, temperature, alternatives=()
) -> tuple[float, list[int]]:
    """P(gold through its first EOS) from one collated completion-only SFT batch.

    The collator supervises the chat template's tokens after the assistant EOS too;
    sampling stops at the first EOS, so the canonical sequence is cut there.
    `alternatives` adds other exact token sequences (each ending at its only EOS).
    """
    torch = import_module("torch")
    ids, labels = batch["input_ids"], batch["labels"]
    if ids.shape[0] != 1 or not bool((batch["attention_mask"] == 1).all()):
        raise LocalVLMError("gold scoring requires one unpadded example")
    supervised = (labels[0] != -100).nonzero().flatten().tolist()
    if not supervised or supervised != list(range(supervised[0], ids.shape[1])):
        raise LocalVLMError("gold completion must be one contiguous supervised suffix")
    first = supervised[0]
    if not torch.equal(labels[0, first:], ids[0, first:]):
        raise LocalVLMError("gold labels differ from the supervised input tokens")
    eos = set(eos_token_ids)
    completion = ids[0, first:].tolist()
    stop = next((i for i, token in enumerate(completion) if token in eos), None)
    if stop is None:
        raise LocalVLMError("gold completion has no EOS token")
    canonical = completion[: stop + 1]
    sequences = [canonical]
    for sequence in alternatives:
        sequence = [int(token) for token in sequence]
        if not sequence or sequence[-1] not in eos or any(t in eos for t in sequence[:-1]):
            raise LocalVLMError("alternative gold tokenization must end at its only EOS")
        if sequence not in sequences:
            sequences.append(sequence)
    prompt = {
        key: value[:, :first] if key in _TOKEN_ALIGNED_KEYS else value
        for key, value in batch.items()
        if key not in _NOT_MODEL_INPUT_KEYS
    }
    probability = sum(
        math.exp(
            _completion_logprob(
                policy,
                prompt,
                sequence,
                temperature=temperature,
                suppress_tokens=suppress_tokens,
            )
        )
        for sequence in sequences
    )
    return min(probability, 1.0), canonical


def gold_probability(
    policy,
    processor,
    row,
    spec,
    profile,
    device,
    *,
    eos_token_ids,
    suppress_tokens,
    alternatives=(),
    collator=None,
) -> float:
    """p = P(exact gold + EOS) under the live policy and the sampler's transforms.

    Built on the `_gold_nll` collator path, but summed over gold tokens, not averaged.
    """
    torch = import_module("torch")
    if collator is None:
        collator = import_module("trl.trainer.sft_trainer").DataCollatorForVisionLanguageModeling(
            processor, completion_only_loss=True, max_length=None
        )
    record = training_records([row], spec, "sft", profile.response_format)[0]
    record["image"] = _load_rgb(record["image"], spec)
    batch = {key: value.to(device) for key, value in collator([record]).items()}
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            probability, _ = _batch_gold_probability(
                policy,
                batch,
                eos_token_ids=eos_token_ids,
                suppress_tokens=suppress_tokens,
                temperature=profile.temperature,
                alternatives=alternatives,
            )
    finally:
        policy.train(was_training)
    return probability


def _load_tokenizations(path: Path | None) -> dict[str, list[list[int]]]:
    """Pre-registered alternate gold tokenizations.

    Format: {"schema_version": 1, "sequences": {label: [[token ids ending at EOS], ...]}}.
    """
    if path is None:
        return {}
    value = json.loads(Path(path).read_text())
    sequences = value.get("sequences") if isinstance(value, dict) else None
    if (
        not isinstance(sequences, dict)
        or value.get("schema_version") != 1
        or any(
            not isinstance(label, str)
            or not isinstance(options, list)
            or any(
                not isinstance(option, list)
                or not option
                or any(type(token) is not int or token < 0 for token in option)
                for option in options
            )
            for label, options in sequences.items()
        )
    ):
        raise ValueError(
            "screen tokenizations require schema_version 1 and label -> token-id lists"
        )
    return sequences


SCREEN_SCOPE = (
    "live-policy gold-probability screen: candidates scored lazily in scan order inside the "
    "committed optimizer step that consumes the next accepted candidate"
)


class _ScreenedScan:
    """Scan-order screen; the checkpoint binds scan position and the committed journal.

    Every scored candidate is appended to the physical journal. A checkpoint commits
    the candidates scored by completed updates; their accepted IDs must equal the
    committed prompt exposure. Scoring uses no RNG, so a resumed scan rescoring
    discarded candidates under the restored adapter reproduces the same decisions.

    Canonical lineage: the physical `screen_journal` jsonl is append-only and also holds
    candidates scored in spans a resume later discarded and rescored. Acceptance rate,
    scorer seconds per candidate and every other G1 quantity must be computed from the
    checkpoint's `screen.json` entries (read them with `committed_screen_entries`), never
    from the raw jsonl. `scorer_seconds` times only the gold-probability forward; the
    adapter hash that labels the update is timed separately as `adapter_hash_seconds` and
    excluded from active time like other audit IO.
    """

    def __init__(
        self,
        journal,
        ids,
        *,
        threshold,
        target_accepted,
        group_size,
        score=None,
        adapter_identity=None,
        resume_state=None,
    ):
        self.journal = Path(journal)
        self.ids = list(ids)
        self.threshold = threshold
        self.target_accepted = target_accepted
        self.group_size = group_size
        self.score = score
        self.adapter_identity = adapter_identity or (
            lambda model: _adapter_tensor_identity(model, "default")["sha256"]
        )
        self.timer = None
        self.invocation_id = str(time.time_ns())
        self.scan_sha256 = _object_sha256(self.ids)
        self.entries = []
        if resume_state is not None:
            if (
                resume_state.get("schema_version") != 1
                or resume_state.get("scope") != SCREEN_SCOPE
                or resume_state.get("threshold") != threshold
                or resume_state.get("target_accepted") != target_accepted
                or resume_state.get("group_size") != group_size
                or resume_state.get("scan_ids_sha256") != self.scan_sha256
                or resume_state.get("position") != len(resume_state.get("entries", []))
            ):
                raise LocalVLMError("checkpoint screen state differs from the scan identity")
            self.entries = list(resume_state["entries"])
        for position, entry in enumerate(self.entries):
            if (
                entry["scan_position"] != position
                or position >= len(self.ids)
                or entry["id"] != self.ids[position]
                or entry["accepted"] != (entry["u"] >= threshold)
            ):
                raise LocalVLMError("checkpoint screen journal differs from the scan order")
        if len(self.accepted_ids) > target_accepted:
            raise LocalVLMError("checkpoint screen accepted more than the fixed target")

    @property
    def position(self):
        return len(self.entries)

    @property
    def accepted_ids(self):
        return [entry["id"] for entry in self.entries if entry["accepted"]]

    def _sync(self):
        if self.timer is not None:
            self.timer._sync()

    def next_accepted(self, model, global_step) -> int:
        """Score under the current adapter until one candidate is accepted; refuse on exhaustion."""
        if self.score is None:
            raise LocalVLMError("screen has no scorer")
        if self.timer is not None and self.timer.started is None:
            raise LocalVLMError("screen scoring must run inside a committed active optimizer step")
        if len(self.accepted_ids) >= self.target_accepted:
            raise LocalVLMError("screen already reached its fixed accepted-update target")
        # Every candidate of one update is scored under the same adapter: hash it once,
        # outside the scorer span, as audit overhead rather than scorer or active time.
        with self.timer.exclude_journal_io() if self.timer is not None else nullcontext():
            hash_started = time.monotonic()
            adapter = self.adapter_identity(model)
            hash_seconds = time.monotonic() - hash_started
        while self.position < len(self.ids):
            index = self.position
            self._sync()
            started = time.monotonic()
            probability = float(self.score(model, index))
            self._sync()
            seconds = time.monotonic() - started
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise LocalVLMError("screen gold probability is not in [0, 1]")
            useful = group_useful_probability(probability, self.group_size)
            entry = {
                "scan_position": index,
                "id": self.ids[index],
                "p": probability,
                "u": useful,
                "accepted": useful >= self.threshold,
                "global_step": global_step,
                "adapter_sha256": adapter,
                "scorer_seconds": seconds,
                "adapter_hash_seconds": hash_seconds,
                "invocation_id": self.invocation_id,
            }
            hash_seconds = 0.0  # charged once, to the update's first candidate
            self.entries.append(entry)
            if self.timer is not None:
                self.timer.charge_scorer(seconds)
            with self.timer.exclude_journal_io() if self.timer is not None else nullcontext():
                with self.journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
            if entry["accepted"]:
                return index
        raise LocalVLMError(
            "screened scan exhausted S before the fixed accepted-update target; arm incomplete"
        )

    def state(self):
        return {
            "schema_version": 1,
            "scope": SCREEN_SCOPE,
            "threshold": self.threshold,
            "target_accepted": self.target_accepted,
            "group_size": self.group_size,
            "scan_ids_sha256": self.scan_sha256,
            "position": self.position,
            "accepted": len(self.accepted_ids),
            "committed_scorer_seconds": sum(entry["scorer_seconds"] for entry in self.entries),
            "entries": self.entries,
        }

    def save(self, checkpoint, exposure_entries):
        if self.accepted_ids != [entry["id"] for entry in exposure_entries]:
            raise LocalVLMError("screen accepted sequence differs from committed prompt exposure")
        _write(Path(checkpoint) / "screen.json", self.state())


def committed_screen_entries(checkpoint: Path) -> list[dict]:
    """The canonical screen journal: entries committed by a verified checkpoint.

    Use this, not the physical jsonl, for acceptance a and scorer cost: the jsonl also
    holds candidates from discarded spans that a resume rescored.
    """
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint / "active_time.json").read_text())
    if "screen_sha256" not in saved or _file_sha256(checkpoint / "screen.json") != saved[
        "screen_sha256"
    ]:
        raise LocalVLMError("checkpoint screen state is not bound to its active-time marker")
    return json.loads((checkpoint / "screen.json").read_text())["entries"]


def _sampler_parity(trainer, processor, row, spec, profile, device) -> dict:
    """Diagnostic only: compare real generated-token scores with teacher forcing.

    TRL suppresses the image token during sampling but its teacher probabilities
    cover the full vocabulary. Report that normalization difference separately.
    This function never constructs or modifies the training loss.
    """
    import copy

    torch = import_module("torch")
    policy = trainer.model
    config = copy.deepcopy(trainer.generation_config)
    config.max_new_tokens = min(16, profile.max_completion_length)
    inputs = _inputs(processor, row, spec, profile).to(device)
    prompt_length = inputs["input_ids"].shape[-1]
    was_training = policy.training
    policy.eval()
    with torch.inference_mode():
        generated = policy.generate(
            **inputs, generation_config=config, return_dict_in_generate=True, output_scores=True
        )
        tokens = generated.sequences[:, prompt_length:]
        forward = dict(inputs)
        forward["input_ids"] = generated.sequences
        forward["attention_mask"] = torch.cat(
            (inputs["attention_mask"], torch.ones_like(tokens)), dim=1
        )
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in forward:
                forward[key] = torch.cat((forward[key], torch.zeros_like(tokens)), dim=1)
        logits = policy(**forward, use_cache=False).logits[:, prompt_length - 1 : -1].float()
        logits = logits / profile.temperature
        raw_logps = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        for token in config.suppress_tokens or []:
            logits[..., token] = -torch.inf
        processed_logps = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        sampled_logps = torch.stack(
            [
                score.float().log_softmax(-1).gather(-1, tokens[:, index : index + 1]).squeeze(-1)
                for index, score in enumerate(generated.scores)
            ],
            dim=1,
        )
        error = float((processed_logps - sampled_logps).abs().max().item())
        raw_error = float((raw_logps - sampled_logps).abs().max().item())
        reference_error, reference_kl = None, None
        if "ref" in getattr(policy, "peft_config", {}):
            use_adapter = import_module("trl.trainer.utils").use_adapter
            with use_adapter(policy, "ref"):
                reference_logits = (
                    policy(**forward, use_cache=False).logits[:, prompt_length - 1 : -1].float()
                    / profile.temperature
                )
                reference_logps = (
                    reference_logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
                )
            difference = reference_logps - raw_logps
            reference_error = float(difference.abs().max().item())
            reference_kl = float((difference.exp() - difference - 1).mean().item())
    policy.train(was_training)
    if not math.isfinite(error):
        raise LocalVLMError("nonfinite sampler/teacher log-probability diagnostic")
    return {
        "sampled_tokens": tokens[0].tolist(),
        "token_count": tokens.shape[-1],
        "max_abs_logprob_error_after_same_suppression": error,
        "max_abs_raw_teacher_vs_sampler_logprob_error": raw_error,
        "sampling_suppressed_token_ids": config.suppress_tokens,
        "initial_reference_max_abs_logprob_difference": reference_error,
        "initial_reference_sampled_kl": reference_kl,
        "tolerance": 0.001 if profile.dtype == "float32" else 0.02,
        "within_tolerance": error <= (0.001 if profile.dtype == "float32" else 0.02),
    }


def _adapter_tensor_identity(model, name: str) -> dict:
    """Fingerprint only a named LoRA adapter, with names normalized for comparison."""
    torch = import_module("torch")
    digest, count, size, trainable = hashlib.sha256(), 0, 0, 0
    for key, parameter in sorted(model.named_parameters()):
        if "lora_" not in key or f".{name}." not in key:
            continue
        value = parameter.detach().contiguous().cpu()
        raw = value.view(torch.uint8).numpy().tobytes()
        header = {
            "name": key.replace(f".{name}.", ".adapter."),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        digest.update(json.dumps(header, sort_keys=True).encode() + b"\0" + raw)
        count += 1
        size += len(raw)
        trainable += int(parameter.requires_grad)
    if not count:
        raise LocalVLMError(f"missing {name} LoRA tensors")
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": count,
        "bytes": size,
        "trainable_tensors": trainable,
    }


def _reference_identity(trainer, *, init_adapter: dict | None, model: str) -> dict:
    """Verify released TRL's small copied reference; never create another base."""
    policy = trainer.model
    if trainer.ref_model is not None or set(policy.peft_config) != {"default", "ref"}:
        raise LocalVLMError("expected one shared base with default and copied ref adapters")
    reference = _adapter_tensor_identity(policy, "ref")
    current = _adapter_tensor_identity(policy, "default")
    if reference["sha256"] != current["sha256"] or reference["trainable_tensors"]:
        raise LocalVLMError("reference must be the frozen exact initial adapter")
    if policy.active_adapters != ["default"]:
        raise LocalVLMError("only the default adapter may be active outside reference inference")
    if init_adapter is None:
        torch = import_module("torch")
        if any(
            torch.count_nonzero(p).item()
            for n, p in policy.named_parameters()
            if "lora_B.default." in n
        ):
            raise LocalVLMError("fresh reference requires zero-initialized LoRA B")
    return {
        "kind": "initial_saved_adapter" if init_adapter else "pinned_base_with_zero_lora",
        "model": model,
        "model_revision": MODEL_REVISIONS[model],
        "initial_adapter_weights_sha256": init_adapter["weights_sha256"] if init_adapter else None,
        "adapter": reference,
        "separate_base_model_allocated": False,
        "reference_name": "ref",
        "beta": trainer.beta,
    }


def _assert_reference_anchor(policy, expected: dict) -> dict:
    """Reject rebasing, activation or training of an immutable initial adapter."""
    if set(policy.peft_config) != {"default", "ref"} or policy.active_adapters != ["default"]:
        raise LocalVLMError("reference checkpoint requires default active and frozen ref")
    actual = _adapter_tensor_identity(policy, "ref")
    if actual != expected or actual["trainable_tensors"]:
        raise LocalVLMError("reference adapter differs from the pinned initial anchor")
    return actual


ACTIVE_TIME_SCOPE = (
    "synchronized accumulated optimizer step: generation/reward/reference/forward/backward/"
    "optimizer included; data prefetch/decode, evaluation, external logging/reward journal "
    "and checkpoint IO excluded"
)
ACTIVE_CHECKPOINT_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "adapter_model.safetensors",
)


def _load_active_time(
    checkpoint: Path, total_budget_seconds: float | None, *, allow_exhausted: bool = False
) -> dict:
    """Bind committed compute to the exact complete optimizer checkpoint."""
    path = checkpoint / "active_time.json"
    if not path.is_file():
        raise LocalVLMError("resume requires checkpoint active_time.json")
    saved = json.loads(path.read_text())
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    used = saved.get("committed_active_seconds")
    if (
        saved.get("schema_version") != 1
        or saved.get("scope") != ACTIVE_TIME_SCOPE
        or saved.get("total_budget_seconds") != total_budget_seconds
        or saved.get("global_step") != state["global_step"]
        or type(used) not in (int, float)
        or not math.isfinite(used)
        or used < 0
        or saved.get("checkpoint_file_sha256")
        != {name: _file_sha256(checkpoint / name) for name in ACTIVE_CHECKPOINT_FILES}
    ):
        raise LocalVLMError("checkpoint active time does not match budget/state/optimizer files")
    if total_budget_seconds is not None and used >= total_budget_seconds and not allow_exhausted:
        raise LocalVLMError("checkpoint has already exhausted the committed active-time budget")
    if saved.get("exposure_sha256") is not None and (
        not (checkpoint / "exposure.json").is_file()
        or _file_sha256(checkpoint / "exposure.json") != saved["exposure_sha256"]
    ):
        raise LocalVLMError("checkpoint exposure identity differs")
    if ("screen_sha256" in saved) != (checkpoint / "screen.json").is_file() or (
        "screen_sha256" in saved
        and _file_sha256(checkpoint / "screen.json") != saved["screen_sha256"]
    ):
        raise LocalVLMError("checkpoint screen identity differs")
    if ("egsft_sha256" in saved) != (checkpoint / "egsft.json").is_file() or (
        "egsft_sha256" in saved
        and _file_sha256(checkpoint / "egsft.json") != saved["egsft_sha256"]
    ):
        raise LocalVLMError("checkpoint EG-SFT weight identity differs")
    return saved


class _ExposureLedger:
    """Actual execution events; checkpoint ancestry selects committed presentations."""

    def __init__(self, output, allowed_ids, resume_state=None, *, inherited=None):
        """`inherited`: a continuation parent's committed entries, which must prefix the
        lineage verbatim; they were checked against the parent's TRAIN, not this one."""
        self.output = Path(output)
        self.allowed = set(allowed_ids)
        self.invocation_id = str(time.time_ns())
        self.entries = list((resume_state or {}).get("entries", []))
        self.pending = None
        self.target_step = None
        inherited = list(inherited or [])
        if self.entries[: len(inherited)] != inherited:
            raise LocalVLMError("exposure lineage does not start with the continuation parent's")
        for step, entry in enumerate(self.entries, 1):
            if entry["global_step"] != step or (
                step > len(inherited) and entry["id"] not in self.allowed
            ):
                raise LocalVLMError("checkpoint exposure sequence differs from TRAIN identity")

    def begin(self, step):
        if step != len(self.entries) + 1:
            raise LocalVLMError("exposure step does not continue committed checkpoint ancestry")
        self.target_step, self.pending = step, None

    def observe(self, identifier):
        if self.target_step is None or identifier not in self.allowed or self.pending is not None:
            raise LocalVLMError("expected exactly one actual TRAIN prompt per optimizer update")
        self.pending = {
            "invocation_id": self.invocation_id,
            "global_step": self.target_step,
            "id": identifier,
        }
        with (self.output / "physical_prompt_events.jsonl").open("a") as handle:
            handle.write(json.dumps(self.pending) + "\n")

    def end(self, step):
        if self.pending is None or self.pending["global_step"] != step:
            raise LocalVLMError("optimizer update lacks an actual prompt exposure event")
        self.entries.append(self.pending)
        self.target_step, self.pending = None, None

    def save(self, checkpoint, step):
        if len(self.entries) != step or self.pending is not None:
            raise LocalVLMError("cannot save an incomplete prompt exposure lineage")
        _write(checkpoint / "exposure.json", {"schema_version": 1, "entries": self.entries})


def _sft_collator_with_ids(collator):
    def with_ids(examples):
        batch = collator(examples)
        batch["_vision_rl_prompt_ids"] = [example["id"] for example in examples]
        return batch

    return with_ids


def _trainer_with_sft_exposure(base, audit):
    class ExposureTrainer(base):
        def training_step(self, model, inputs, num_items_in_batch=None):
            identifiers = inputs.pop("_vision_rl_prompt_ids")
            if len(identifiers) != 1:
                raise LocalVLMError("SFT exposure contract requires one actual prompt per update")
            audit.observe_prompt(identifiers[0])
            self._vision_rl_prompt_id = identifiers[0]
            return super().training_step(model, inputs, num_items_in_batch)

    return ExposureTrainer


def _trainer_with_egsft(
    base,
    *,
    eos_token_ids,
    suppress_tokens,
    temperature,
    group_size,
    journal,
    tokenizations=None,
    labels_by_id=None,
    journal_io_context=None,
    resume_state=None,
):
    """EG-SFT: completion-only SFT loss times stopgrad[w_G(p) / w_G(0.5)].

    Rows carry an invocation_id. The physical `journal` is append-only and keeps rows
    of steps a resume later discarded and re-ran; the canonical weights are the
    checkpoint's `egsft.json` (one row per committed global step, hash-bound by the
    active-time marker), restored from `resume_state` on resume.

    p is the screen's gold probability, computed without gradient on the identical
    collated batch. Released TRL's SFT forward is fused and returns only label
    log-probs, not the image-token mass the sampler's suppression renormalizes, so p
    takes one extra no-grad teacher-forced forward rather than reusing that one.
    """
    torch = import_module("torch")
    invocation_id = str(time.time_ns())
    committed = list((resume_state or {}).get("entries", []))
    if (resume_state is not None and resume_state.get("schema_version") != 1) or [
        row["global_step"] for row in committed
    ] != list(range(1, len(committed) + 1)):
        raise LocalVLMError("checkpoint EG-SFT weights are not one row per committed step")

    class ExpectedGradientTrainer(base):
        _vision_rl_egsft_entries = committed

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            step = self.state.global_step
            rows = [row for row in self._vision_rl_egsft_entries if row["global_step"] <= step]
            if [row["global_step"] for row in rows] != list(range(1, step + 1)):
                raise LocalVLMError("EG-SFT weights do not cover every committed step once")
            # Written before the callbacks' on_save, so the active-time marker binds it.
            _write(
                Path(self.args.output_dir) / f"checkpoint-{step}" / "egsft.json",
                {"schema_version": 1, "entries": rows},
            )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            identifier = getattr(self, "_vision_rl_prompt_id", None)
            label = None
            if tokenizations:
                label = labels_by_id[identifier]
            with torch.no_grad():
                probability, _ = _batch_gold_probability(
                    model,
                    inputs,
                    eos_token_ids=eos_token_ids,
                    suppress_tokens=suppress_tokens,
                    temperature=temperature,
                    alternatives=(tokenizations or {}).get(label, ()),
                )
            weight = egsft_weight(probability, group_size)
            # Pass return_outputs through: TRL recomputes full logits with an extra
            # forward whenever outputs are requested.
            result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            loss, outputs = result if return_outputs else (result, None)
            loss = loss * torch.tensor(weight, dtype=loss.dtype, device=loss.device)
            if loss.requires_grad is False and torch.is_grad_enabled():
                raise LocalVLMError("EG-SFT loss lost its gradient path")
            row = {
                "global_step": self.state.global_step + 1,
                "id": identifier,
                "p": probability,
                "weight": weight,
                "invocation_id": invocation_id,
            }
            if torch.is_grad_enabled():
                # Checkpoint save refuses anything but one row per committed step.
                self._vision_rl_egsft_entries.append(row)
            with journal_io_context() if journal_io_context else nullcontext():
                with Path(journal).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            return (loss, outputs) if return_outputs else loss

    return ExpectedGradientTrainer


class _ActiveStepTimer:
    """Commit complete update spans; preserve observed discarded work separately."""

    def __init__(self, output, total_budget_seconds, device, resume_state=None):
        self.output = Path(output)
        self.total_budget = total_budget_seconds
        self.device = device
        self.initial_committed = (resume_state or {}).get("committed_active_seconds", 0.0)
        self.durable_committed = self.initial_committed
        self.durable_global_step = (resume_state or {}).get("global_step", 0)
        self.completed = 0.0
        self.interrupted = 0.0
        self.journal_io = 0.0
        self.step_journal_io = 0.0
        self.started = None
        self.path = self.output / "active_time" / f"invocation-{time.time_ns()}.json"
        self.in_progress = False
        # Screened runs only: scorer forwards run inside the step span, so they are
        # already committed active time; this sub-account reports their share.
        self.scorer = None

    @property
    def committed(self):
        return self.initial_committed + self.completed

    def _sync(self):
        if self.device == "mps":
            import_module("torch").mps.synchronize()
        elif self.device == "cuda":
            import_module("torch").cuda.synchronize()

    def _persist(self):
        value = {
            "schema_version": 1,
            "scope": ACTIVE_TIME_SCOPE,
            "device": self.device,
            "resume_committed_active_seconds": self.initial_committed,
            "completed_active_seconds": self.completed,
            "observed_interrupted_active_seconds": self.interrupted,
            "observed_journal_io_seconds": self.journal_io,
            "in_progress": self.in_progress,
        }
        if self.scorer is not None:
            value["observed_scorer_seconds"] = self.scorer
        _write(self.path, value)

    def enable_scorer(self):
        self.scorer = 0.0 if self.scorer is None else self.scorer

    def charge_scorer(self, seconds):
        """Record scorer time spent inside the open step span (already active time)."""
        if self.scorer is None or self.started is None:
            raise LocalVLMError("scorer seconds must fall inside an open active optimizer step")
        self.scorer += seconds

    def check_remaining(self):
        if self.total_budget is not None and self.committed >= self.total_budget:
            raise LocalVLMError("committed active-time budget is already exhausted")

    def begin(self):
        self.check_remaining()
        if self.started is not None:
            raise LocalVLMError("active optimizer step was not closed")
        self.in_progress = True
        self._persist()
        self._sync()
        self.started = time.monotonic()
        self.step_journal_io = 0.0

    @contextmanager
    def exclude_journal_io(self):
        if self.started is None:
            yield
            return
        self._sync()
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self.step_journal_io += elapsed
            self.journal_io += elapsed

    def end(self):
        if self.started is None:
            raise LocalVLMError("active optimizer step was not started")
        self._sync()
        self.completed += time.monotonic() - self.started - self.step_journal_io
        self.started = None
        self.in_progress = False
        self._persist()

    def interrupt(self):
        # A hard process kill can leave in_progress=true; no duration is invented.
        if self.started is not None:
            self._sync()
            self.interrupted += time.monotonic() - self.started - self.step_journal_io
            self.started = None
            self.in_progress = False
            self._persist()

    def save(self, checkpoint: Path, global_step: int):
        if self.started is not None:
            raise LocalVLMError("cannot checkpoint a partial active optimizer step")
        _write(
            checkpoint / "active_time.json",
            {
                "schema_version": 1,
                "scope": ACTIVE_TIME_SCOPE,
                "device": self.device,
                "global_step": global_step,
                "total_budget_seconds": self.total_budget,
                "committed_active_seconds": self.committed,
                "checkpoint_file_sha256": {
                    name: _file_sha256(checkpoint / name) for name in ACTIVE_CHECKPOINT_FILES
                },
                "exposure_sha256": _file_sha256(checkpoint / "exposure.json")
                if (checkpoint / "exposure.json").is_file()
                else None,
                **(
                    {"screen_sha256": _file_sha256(checkpoint / "screen.json")}
                    if (checkpoint / "screen.json").is_file()
                    else {}
                ),
                **(
                    {"egsft_sha256": _file_sha256(checkpoint / "egsft.json")}
                    if (checkpoint / "egsft.json").is_file()
                    else {}
                ),
            },
        )
        self.durable_committed = self.committed
        self.durable_global_step = global_step

    def summary(self):
        invocations = [
            json.loads(path.read_text()) for path in (self.output / "active_time").glob("*.json")
        ]
        observed = sum(
            item["completed_active_seconds"] + item["observed_interrupted_active_seconds"]
            for item in invocations
        )
        scorer = (
            {
                "invocation_scorer_seconds": self.scorer,
                "observed_scorer_seconds_within_active": sum(
                    item.get("observed_scorer_seconds", 0.0) for item in invocations
                ),
            }
            if self.scorer is not None
            else {}
        )
        return {
            **scorer,
            "scope": ACTIVE_TIME_SCOPE,
            "total_budget_seconds": self.total_budget,
            "resumed_committed_active_seconds": self.initial_committed,
            "invocation_active_seconds": self.completed,
            "completed_state_active_seconds": self.committed,
            "committed_active_seconds": self.durable_committed,
            "committed_global_step": self.durable_global_step,
            "completed_not_checkpointed_active_seconds": self.committed - self.durable_committed,
            "observed_physical_active_seconds": observed,
            "observed_reward_journal_io_seconds": sum(
                item.get("observed_journal_io_seconds", 0.0) for item in invocations
            ),
            "observed_discarded_or_uncheckpointed_active_seconds": max(
                0.0, observed - self.durable_committed
            ),
            "unknown_partial_invocations": sum(bool(item["in_progress"]) for item in invocations),
            "hard_total_physical_compute_cap": False,
        }


def _audit_callback(
    output: Path,
    max_active_seconds: float | None,
    *,
    device: str = "cpu",
    resume_state: dict | None = None,
    exposure: _ExposureLedger | None = None,
    stop_after_steps: int | None = None,
    stop_after_active_seconds: float | None = None,
    screen: _ScreenedScan | None = None,
    loss_scaled: bool = False,
):
    torch = import_module("torch")
    base = import_module("transformers").TrainerCallback

    class Audit(base):
        def __init__(self):
            self.initial = {}
            self.expected_reference = None
            self.timer = _ActiveStepTimer(output, max_active_seconds, device, resume_state)
            if screen is not None:
                if exposure is None:
                    raise LocalVLMError("screened training requires the prompt exposure ledger")
                self.timer.enable_scorer()
                screen.timer = self.timer
            self.summary = {
                "optimizer_steps_observed": 0,
                "nonzero_gradient_steps": 0,
                "active_seconds": 0.0,
                "max_gradient_norm_observed": 0.0,
            }
            if loss_scaled:
                # float16 only: a GradScaler overflow skips the optimizer step, so the
                # prompt is consumed without an update. Counted, never silent.
                self.summary.update(loss_scale_skipped_steps=0, loss_scale=None)
                self.overflow = False
                self.consecutive_skips = 0

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.timer.check_remaining()
            self.initial = {
                n: p.detach().float().cpu().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            if not self.initial or any(
                "lora_" not in n or ".visual." in n or ".ref." in n for n in self.initial
            ):
                raise LocalVLMError("only language LoRA parameters may be trainable")
            self.reference = (
                _adapter_tensor_identity(model, "ref")
                if "ref" in getattr(model, "peft_config", {})
                else None
            )
            if self.expected_reference is not None:
                self.reference = _assert_reference_anchor(model, self.expected_reference)

        def on_step_begin(self, args, state, control, **kwargs):
            if exposure is not None:
                exposure.begin(state.global_step + 1)
            self.timer.begin()

        def observe_prompt(self, identifier):
            if exposure is not None:
                with self.timer.exclude_journal_io():
                    exposure.observe(identifier)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            norm_sq = 0.0
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    norm_sq += float(parameter.grad.detach().float().square().sum().item())
            if not math.isfinite(norm_sq):
                if not loss_scaled:
                    raise LocalVLMError("nonfinite gradient; optimizer step rejected")
                self.overflow = True
                self.summary["latest_gradient_norm"] = 0.0
                return
            if loss_scaled:
                self.overflow = False
            norm = math.sqrt(norm_sq)
            self.summary["latest_gradient_norm"] = norm
            self.summary["nonzero_gradient_steps"] += int(norm > 0)
            self.summary["max_gradient_norm_observed"] = max(
                self.summary["max_gradient_norm_observed"], norm
            )

        def on_step_end(self, args, state, control, model=None, **kwargs):
            self.timer.end()
            if exposure is not None:
                exposure.end(state.global_step)
            if loss_scaled:
                optimizer = kwargs.get("optimizer")
                scaler = getattr(optimizer, "scaler", None)
                if scaler is None:
                    raise LocalVLMError("float16 training lacks an active GradScaler")
                skipped = bool(getattr(optimizer, "step_was_skipped", False))
                if self.overflow != skipped:
                    raise LocalVLMError("loss-scaler skip disagrees with the observed gradients")
                self.consecutive_skips = self.consecutive_skips + 1 if skipped else 0
                self.summary["loss_scale_skipped_steps"] += int(skipped)
                self.summary["loss_scale"] = float(scaler.get_scale())
                if self.consecutive_skips > MAX_CONSECUTIVE_LOSS_SCALE_SKIPS:
                    raise LocalVLMError(
                        "float16 gradients overflowed on every recent step; not loss-scale "
                        "underflow but a nonfinite forward/backward"
                    )
            self.summary["optimizer_steps_observed"] += 1
            self.summary["active_seconds"] = self.timer.completed
            self.summary["completed_state_active_seconds"] = self.timer.committed
            if (
                model is not None
                and self.summary.get("latest_gradient_norm", 0) > 0
                and "first_nonzero_update" not in self.summary
            ):
                delta_sq, changed = 0.0, 0
                for name, parameter in model.named_parameters():
                    if name in self.initial:
                        delta = parameter.detach().float().cpu() - self.initial[name]
                        delta_sq += float(delta.square().sum().item())
                        changed += int(bool(torch.count_nonzero(delta).item()))
                self.summary["first_nonzero_update"] = {
                    "global_step": state.global_step,
                    "gradient_norm": self.summary["latest_gradient_norm"],
                    "changed_tensors_since_invocation_start": changed,
                    "parameter_delta_l2_since_invocation_start": math.sqrt(delta_sq),
                }
            with (output / "optimizer_audit.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": state.global_step, **self.summary}) + "\n")
            if max_active_seconds is not None and self.timer.committed >= max_active_seconds:
                control.should_training_stop = True
                control.should_save = True
            if (stop_after_steps is not None and state.global_step >= stop_after_steps) or (
                stop_after_active_seconds is not None
                and self.timer.committed >= stop_after_active_seconds
            ):
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            if exposure is not None:
                exposure.save(
                    Path(args.output_dir) / f"checkpoint-{state.global_step}", state.global_step
                )
            if screen is not None:
                screen.save(
                    Path(args.output_dir) / f"checkpoint-{state.global_step}", exposure.entries
                )
            self.timer.save(
                Path(args.output_dir) / f"checkpoint-{state.global_step}", state.global_step
            )

        def on_train_end(self, args, state, control, model=None, **kwargs):
            delta_sq, changed = 0.0, 0
            for name, parameter in model.named_parameters():
                if name in self.initial:
                    delta = parameter.detach().float().cpu() - self.initial[name]
                    delta_sq += float(delta.square().sum().item())
                    changed += int(bool(torch.count_nonzero(delta).item()))
            if not math.isfinite(delta_sq):
                raise LocalVLMError("nonfinite final adapter parameters")
            self.summary.update(parameter_delta_l2=math.sqrt(delta_sq), changed_tensors=changed)
            if self.reference is not None:
                final_reference = _adapter_tensor_identity(model, "ref")
                if final_reference != self.reference or final_reference["trainable_tensors"]:
                    raise LocalVLMError("reference adapter changed or became trainable")
                self.summary["reference_unchanged"] = True
                self.summary["reference_sha256"] = final_reference["sha256"]
            self.initial.clear()
            self.summary["active_time"] = self.timer.summary()

    return Audit()


def _adapter_identity(
    path: Path,
    identity: dict,
    profile: Profile,
    *,
    precision_transfer: str | None = None,
    inference: bool = False,
) -> dict:
    if _expanded.is_expanded(profile):
        if precision_transfer is not None:
            raise ValueError("expanded FP32 profile does not permit precision transfer")
        return _expanded.artifact_identity(path, identity, profile)
    parent = json.loads((path.parent / "local_training.json").read_text())
    source = parent["control_identity"]
    matching_keys = ("model", "model_revision", "processor_model", "processor_revision")
    profile_keys = ("dtype", "min_pixels", "max_pixels", "lora_rank")
    transfer = None
    if precision_transfer is not None:
        if inference:
            if (
                precision_transfer not in INFERENCE_PRECISION_TRANSFERS
                or precision_transfer != f"{source['profile']['dtype']}-to-{profile.dtype}"
            ):
                raise ValueError(
                    "inference precision transfer must name the parent's and this profile's "
                    "base dtypes"
                )
        elif (
            precision_transfer not in PRECISION_TRANSFERS
            or source["profile"]["dtype"] != "float32"
            or profile.dtype != precision_transfer.removeprefix("float32-to-")
        ):
            raise ValueError(
                "only explicit float32-to-bfloat16/float16 base precision transfer is supported"
            )
        profile_keys = tuple(key for key in profile_keys if key != "dtype")
        transfer = {
            "profile": precision_transfer,
            "parent_base_dtype": source["profile"]["dtype"],
            "target_base_dtype": profile.dtype,
            "adapter_dtype": "float32",
            "adapter_weights_changed": False,
            "initial_policy_numerics_changed": True,
        }
    if (
        parent["status"] != "completed"
        or any(source[k] != identity[k] for k in matching_keys)
        or any(source["profile"][k] != identity["profile"][k] for k in profile_keys)
    ):
        raise ValueError("warm start requires completed run with identical model/processor/profile")
    config = json.loads((path / "adapter_config.json").read_text())
    validated = _validated_lora_config(config)
    if validated["r"] != profile.lora_rank or set(validated["target_modules"]) != {
        "q_proj",
        "v_proj",
    }:
        raise ValueError("warm start LoRA profile differs")
    weights = path / "adapter_model.safetensors"
    if not weights.is_file() or weights.is_symlink() or weights.stat().st_size > 2 * 1024**3:
        raise ValueError("warm start requires regular bounded safetensors adapter")
    result = {
        "path": str(path),
        "weights_sha256": _file_sha256(weights),
        "config_sha256": _file_sha256(path / "adapter_config.json"),
        "parent_profile": source["profile"],
        "parent_data_identity": {k: source[k] for k in ("data_spec_sha256", "train", "validation")},
    }
    if transfer is not None:
        result["precision_transfer"] = transfer
    return result


def _verify_initial_adapter_tensors(policy, path: Path) -> dict:
    """Verify PEFT preserves each saved FP32 adapter value before any update."""
    torch = import_module("torch")
    expected = import_module("safetensors.torch").load_file(
        str(path / "adapter_model.safetensors"), device="cpu"
    )
    actual = import_module("peft").get_peft_model_state_dict(policy, adapter_name="default")
    if set(expected) != set(actual) or any(
        expected[name].dtype != torch.float32
        or actual[name].dtype != torch.float32
        or not torch.equal(expected[name], actual[name].detach().cpu())
        for name in expected
    ):
        raise LocalVLMError("loaded initial adapter differs from exact saved FP32 tensors")
    return {"exact_saved_tensors": True, "tensor_count": len(expected), "dtype": "float32"}


def _inference_adapter_identity(
    path: Path,
    identity: dict,
    profile: Profile,
    *,
    initial_path: Path | None = None,
    precision_transfer: str | None = None,
) -> dict:
    """Only a hash-bound completed endpoint may be used for a TRAIN assay.

    `precision_transfer` ("<parent dtype>-to-<this dtype>") explicitly runs the frozen
    adapter on a different base precision than it was trained with; it is recorded.
    """
    if _expanded.is_expanded(profile):
        if precision_transfer is not None:
            raise ValueError("expanded FP32 profile does not permit precision transfer")
        return _expanded.artifact_identity(
            path, identity, profile, inference=True, initial_path=initial_path
        )
    if initial_path is not None:
        raise ValueError("initial_path applies only to the expanded profile")
    initial = _adapter_identity(
        path, identity, profile, precision_transfer=precision_transfer, inference=True
    )
    parent = json.loads((path.parent / "local_training.json").read_text())
    if initial["parent_data_identity"]["data_spec_sha256"] != identity["data_spec_sha256"]:
        raise ValueError("saved-adapter TRAIN inference requires the same pinned task data spec")
    if (
        parent.get("final_adapter_sha256") != initial["weights_sha256"]
        or Path(parent.get("final_adapter", "")).name != path.name
    ):
        raise ValueError("saved adapter differs from the completed endpoint identity")
    return initial


def _trainer_with_loss_scaling(base, init_scale: float):
    """Start accelerate's fp16 GradScaler at `init_scale` instead of torch's 2**16."""
    grad_scaler_kwargs = import_module("accelerate.utils").GradScalerKwargs

    class LossScaledTrainer(base):
        def _build_accelerator_args(self, **kwargs):
            args = super()._build_accelerator_args(**kwargs)
            if args.get("mixed_precision") != "fp16":
                raise LocalVLMError("float16 training must use accelerate fp16 mixed precision")
            handlers = [
                h for h in args.get("kwargs_handlers", []) if not isinstance(h, grad_scaler_kwargs)
            ]
            args["kwargs_handlers"] = handlers + [
                grad_scaler_kwargs(
                    init_scale=init_scale, growth_interval=LOSS_SCALE_GROWTH_INTERVAL
                )
            ]
            return args

    return LossScaledTrainer


def _trainer_with_mps_rng(base, device: str):
    """Transformers 5.20 saves CPU/CUDA RNG but omits MPS; preserve it explicitly."""
    if device != "mps":
        return base
    torch = import_module("torch")

    class MPSRNGTrainer(base):
        def _save_rng_state(self, output_dir):
            super()._save_rng_state(output_dir)
            torch.save(torch.mps.get_rng_state(), Path(output_dir) / "mps_rng_state.pth")

        def _load_rng_state(self, checkpoint):
            super()._load_rng_state(checkpoint)
            if checkpoint is not None:
                state = torch.load(
                    Path(checkpoint) / "mps_rng_state.pth", map_location="cpu", weights_only=True
                )
                torch.mps.set_rng_state(state)

    return MPSRNGTrainer


def _trainer_with_active_time(base):
    """Publish checkpoint time only after all native checkpoint writes succeed."""

    class ActiveTimeTrainer(base):
        def _save_checkpoint(self, model, trial):
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            (checkpoint / "active_time.json").unlink(missing_ok=True)
            # The audit callback publishes a new hash-bound marker in on_save.
            return super()._save_checkpoint(model, trial)

    return ActiveTimeTrainer


def _trainer_with_grpo_resume(base, *, expected_reference: dict | None = None):
    """Restore both PEFT adapters and the released sampler's epoch permutation.

    RepeatSampler owns a local generator without set_epoch; restarting Trainer
    otherwise repeats epoch zero. Advance that same generator through the prior
    native permutations, preserving the uninterrupted released sampler order.
    """
    torch = import_module("torch")
    repeat_sampler = import_module("trl.trainer.utils").RepeatSampler

    class EpochRepeatSampler(repeat_sampler):
        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            if self.shuffle:
                self.generator.manual_seed(self.seed)
                for _ in range(getattr(self, "epoch", 0)):
                    torch.randperm(self.num_samples, generator=self.generator)
            return super().__iter__()

    class ResumeTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._initial_reference = None
            if "ref" in getattr(self.model, "peft_config", {}):
                self._initial_reference = (
                    dict(expected_reference)
                    if expected_reference is not None
                    else _adapter_tensor_identity(self.model, "ref")
                )
                _assert_reference_anchor(self.model, self._initial_reference)
            elif expected_reference is not None:
                raise LocalVLMError("pinned reference adapter is missing")

        def _get_train_sampler(self, dataset=None):
            sampler = super()._get_train_sampler(dataset)
            if not isinstance(sampler, repeat_sampler) or sampler.seed is None:
                raise LocalVLMError("expected the pinned seeded TRL RepeatSampler")
            return EpochRepeatSampler(
                **{
                    name: getattr(sampler, name)
                    for name in (
                        "data_source",
                        "mini_repeat_count",
                        "batch_size",
                        "repeat_count",
                        "shuffle",
                        "seed",
                    )
                }
            )

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            policy = self.model if model is None else model
            if "ref" in policy.peft_config:
                if set(policy.peft_config) != {"default", "ref"}:
                    raise LocalVLMError("reference checkpoint requires default and ref adapters")
                policy.load_adapter(resume_from_checkpoint, "default", is_trainable=True)
                policy.set_adapter("default")
                _assert_reference_anchor(policy, self._initial_reference)

    return ResumeTrainer


def _sampler_transform(trainer) -> dict:
    """The released sampler's logit transforms; refuse any the gold scorer does not apply."""
    config = trainer.generation_config
    if (
        config.top_p not in (None, 1.0)
        or config.top_k not in (None, 0)
        or config.min_p is not None
        or config.repetition_penalty not in (None, 1.0)
        or not math.isfinite(trainer.temperature)
        or config.temperature != trainer.temperature
    ):
        raise LocalVLMError(
            "gold scorer supports only temperature and token suppression transforms"
        )
    return {
        "temperature": trainer.temperature,
        "suppress_tokens": list(config.suppress_tokens or []),
        "eos_token_ids": sorted(trainer.eos_token_ids),
    }


def _trainer_with_screening(base, screen: _ScreenedScan, records):
    """Resolve each GRPO update to the next accepted scan candidate under the live adapter.

    The dataloader cannot score: Trainer prefetches an update's batches and
    Accelerate reads one batch ahead, so a scoring sampler would score the next
    candidate before the current update is applied. The sampler therefore yields one
    placeholder slot per accepted update (its length fixes the epoch at the target),
    and the slot is resolved where TRL generates, inside the active step span.
    """
    torch = import_module("torch")

    class _ScreenedRepeatSampler(torch.utils.data.Sampler):
        def __init__(self, slots, repeats):
            self.slots, self.repeats = slots, repeats

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            for _ in range(self.slots * self.repeats):
                yield 0

        def __len__(self):
            return self.slots * self.repeats

    class ScreenedTrainer(base):
        def _get_train_sampler(self, dataset=None):
            native = super()._get_train_sampler(dataset)
            if native.batch_size != 1 or self.num_iterations != 1:
                raise LocalVLMError(
                    "screening requires one prompt per generation and one iteration"
                )
            return _ScreenedRepeatSampler(
                screen.target_accepted, native.mini_repeat_count * native.repeat_count
            )

        def _generate_and_score_completions(self, inputs):
            if self.model.training:
                index = screen.next_accepted(self.model, self.state.global_step + 1)
                inputs = [dict(records[index]) for _ in inputs]
            return super()._generate_and_score_completions(inputs)

    ScreenedTrainer._ScreenedRepeatSampler = _ScreenedRepeatSampler
    return ScreenedTrainer


class _PromptRelay:
    """The one route from the reward's per-group observation to the exposure ledger.

    Dynamic sampling defers observations until it keeps a group, then commits only the
    kept prompt; policy reuse (num_iterations > 1) re-observes that kept prompt for the
    updates that optimize on the same buffered generation.
    """

    def __init__(self, observe):
        self.observe = observe
        self.last_id = None
        self.deferred = None

    def note(self, identifier):
        self.last_id = identifier
        if self.deferred is None:
            self.observe(identifier)
        else:
            self.deferred.append(identifier)

    def defer(self):
        if self.deferred is not None:
            raise LocalVLMError("prompt relay is already deferring")
        self.deferred = []

    def abandon(self):
        self.deferred = None

    def commit(self):
        identifiers, self.deferred = self.deferred, None
        if not identifiers:
            raise LocalVLMError("dynamic sampling scored no group")
        self.last_id = identifiers[-1]
        self.observe(identifiers[-1])
        return identifiers


def dynamic_replacement_index(seed: int, global_step: int, draw: int, size: int) -> int:
    """Deterministic replacement prompt for draw >= 1 of an update (resume-stable, no RNG state)."""
    import random

    if draw < 1 or size < 1:
        raise ValueError("replacement draws start at 1 over a nonempty pool")
    return random.Random(f"dynamic-sampling:{seed}:{global_step}:{draw}").randrange(size)


def _trainer_with_dynamic_sampling(
    base, relay, records, *, max_draws, seed, journal, journal_io_context=None, summary=None
):
    """DAPO dynamic sampling, which released TRL 1.16 lacks, for one prompt per update.

    A generated group whose advantages are all zero (every reward equal) is dropped and
    replaced by a deterministic draw from the TRAIN pool, up to `max_draws` groups per
    update. If every draw is uninformative the last one is kept (it carries only the KL
    term) and the update is journaled as exhausted. Every draw's generation runs inside
    the active step span, so its cost is charged to active time. Only the kept prompt
    enters the exposure ledger; every draw is journaled.
    """
    torch = import_module("torch")
    summary = summary if summary is not None else {}
    summary.update(updates=0, draws=0, dropped_groups=0, exhausted_updates=0)

    class DynamicSamplingTrainer(base):
        def _generate_and_score_completions(self, inputs):
            if not self.model.training:
                return super()._generate_and_score_completions(inputs)
            step = self.state.global_step + 1
            entries = []
            relay.defer()
            try:
                for draw in range(max_draws):
                    if draw:
                        index = dynamic_replacement_index(seed, step, draw, len(records))
                        inputs = [dict(records[index]) for _ in inputs]
                    output = super()._generate_and_score_completions(inputs)
                    informative = bool(torch.count_nonzero(output["advantages"]).item())
                    entries.append({"global_step": step, "draw": draw, "informative": informative})
                    if informative:
                        break
            except BaseException:
                relay.abandon()
                raise
            identifiers = relay.commit()
            if len(identifiers) != len(entries):
                raise LocalVLMError("dynamic sampling expected one reward group per draw")
            for entry, identifier in zip(entries, identifiers, strict=True):
                entry["id"] = identifier
                entry["kept"] = entry is entries[-1]
            exhausted = not entries[-1]["informative"]
            entries[-1]["exhausted"] = exhausted
            with journal_io_context() if journal_io_context else nullcontext():
                with Path(journal).open("a", encoding="utf-8") as handle:
                    handle.write("".join(json.dumps(entry) + "\n" for entry in entries))
            summary["updates"] += 1
            summary["draws"] += len(entries)
            summary["dropped_groups"] += len(entries) - 1
            summary["exhausted_updates"] += int(exhausted)
            return output

    return DynamicSamplingTrainer


def _trainer_with_policy_reuse(base, relay):
    """num_iterations > 1: TRL optimizes several updates on one buffered generation.

    Those reuse updates generate nothing, so the reward never observes a prompt; observe
    the buffered (kept) prompt at the first micro-step of each reuse update instead.
    """

    class PolicyReuseTrainer(base):
        def _prepare_inputs(self, generation_batch):
            if self.model.training:
                every = self.args.steps_per_generation
                regenerate = (
                    self._step % (every * self.num_iterations) == 0
                    or self._buffered_inputs is None
                )
                if not regenerate and self._step % every == 0:
                    if relay.last_id is None:
                        raise LocalVLMError("policy reuse has no buffered prompt to observe")
                    relay.observe(relay.last_id)
            return super()._prepare_inputs(generation_batch)

    return PolicyReuseTrainer


def _restore_endpoint_without_training(trainer, checkpoint, audit):
    """Restore an already-complete optimization boundary; never call train or step."""
    trainer._load_from_checkpoint(str(checkpoint))
    trainer.state = import_module("transformers.trainer_callback").TrainerState.load_from_json(
        str(checkpoint / "trainer_state.json")
    )
    trainer._load_rng_state(str(checkpoint))
    if audit.expected_reference is not None:
        restored = _assert_reference_anchor(trainer.model, audit.expected_reference)
        audit.summary.update(reference_unchanged=True, reference_sha256=restored["sha256"])
    audit.summary.update(
        changed_tensors=0,
        parameter_delta_l2=0.0,
        active_time=audit.timer.summary(),
        endpoint_recovery_only=True,
    )


def _object_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _checkpoint_receipt_files(checkpoint):
    required = list(ACTIVE_CHECKPOINT_FILES) + [
        "active_time.json",
        "exposure.json",
        "rng_state.pth",
        "adapter_config.json",
    ]
    for optional in (
        "screen.json",
        "egsft.json",
        "mps_rng_state.pth",
        "scaler.pt",
        "ref/adapter_model.safetensors",
        "ref/adapter_config.json",
    ):
        if (checkpoint / optional).is_file():
            required.append(optional)
    return {name: _file_sha256(checkpoint / name) for name in required}


def _publish_endpoint_receipt(output, record, global_step):
    checkpoint = output / f"checkpoint-{global_step}"
    active = json.loads((checkpoint / "active_time.json").read_text())
    exposures = json.loads((checkpoint / "exposure.json").read_text())
    settings = record["settings"]
    exhausted = global_step >= settings["max_steps"] or (
        settings["max_active_seconds"] is not None
        and active["committed_active_seconds"] >= settings["max_active_seconds"]
    )
    adapter = output / "final_adapter"
    receipt = {
        "schema_version": 1,
        "stage_id": record["stage_id"],
        "status": "total_budget_exhausted" if exhausted else "boundary_reached",
        "global_step": global_step,
        "original_budgets": {k: settings[k] for k in ("max_steps", "max_active_seconds")},
        "requested_boundary": record["requested_boundary"],
        "committed_active_seconds": active["committed_active_seconds"],
        "committed_prompt_presentations": len(exposures["entries"]),
        "committed_prompt_ids": [entry["id"] for entry in exposures["entries"]],
        "checkpoint": {"path": checkpoint.name, "sha256": _checkpoint_receipt_files(checkpoint)},
        "final_adapter": {
            "path": "final_adapter",
            "weights_sha256": _file_sha256(adapter / "adapter_model.safetensors"),
            "config_sha256": _file_sha256(adapter / "adapter_config.json"),
        },
        "reference_sha256": record["reference"]["adapter"]["sha256"]
        if record["reference"]
        else None,
        "control_identity_sha256": _object_sha256(record["control_identity"]),
        "settings_sha256": _object_sha256(settings),
        "source_sha256": record["environment"]["adapter_source_sha256"],
        "physical_work": record["optimizer_audit"]["active_time"],
    }
    _write(output / "endpoint_receipt.json", receipt)
    read_endpoint_receipt(output, expected_stage_id=record["stage_id"])


def read_endpoint_receipt(output: Path, *, expected_stage_id: str | None = None) -> dict:
    """Verify a completed staged endpoint without model allocation or native deserialization."""
    output = Path(output).resolve()
    receipt = json.loads((output / "endpoint_receipt.json").read_text())
    record = json.loads((output / "local_training.json").read_text())
    if (
        receipt.get("schema_version") != 1
        or record["status"] != "completed"
        or receipt["stage_id"] != record["stage_id"]
        or (expected_stage_id is not None and receipt["stage_id"] != expected_stage_id)
        or receipt["control_identity_sha256"] != _object_sha256(record["control_identity"])
        or receipt["settings_sha256"] != _object_sha256(record["settings"])
        or receipt["source_sha256"] != _file_sha256(output / "adapter-source.py")
        or receipt["source_sha256"] != record["environment"]["adapter_source_sha256"]
        or receipt["physical_work"] != record["optimizer_audit"]["active_time"]
    ):
        raise LocalVLMError("endpoint receipt differs from completed run identity")
    checkpoint = (output / receipt["checkpoint"]["path"]).resolve()
    adapter = (output / receipt["final_adapter"]["path"]).resolve()
    if not checkpoint.is_relative_to(output) or not adapter.is_relative_to(output):
        raise LocalVLMError("endpoint artifact escapes trajectory directory")
    if (
        receipt["checkpoint"]["sha256"] != _checkpoint_receipt_files(checkpoint)
        or receipt["final_adapter"]["weights_sha256"]
        != _file_sha256(adapter / "adapter_model.safetensors")
        or receipt["final_adapter"]["config_sha256"]
        != _file_sha256(adapter / "adapter_config.json")
        or receipt["final_adapter"]["weights_sha256"] != record["final_adapter_sha256"]
    ):
        raise LocalVLMError("endpoint checkpoint or saved adapter bytes differ")
    active = json.loads((checkpoint / "active_time.json").read_text())
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    exposures = json.loads((checkpoint / "exposure.json").read_text())
    entries = exposures["entries"]
    original = {k: record["settings"][k] for k in ("max_steps", "max_active_seconds")}
    if (
        active["scope"] != ACTIVE_TIME_SCOPE
        or active["global_step"] != state["global_step"]
        or active["global_step"] != receipt["global_step"]
        or active["committed_active_seconds"] != receipt["committed_active_seconds"]
        or active["total_budget_seconds"] != original["max_active_seconds"]
        or receipt["original_budgets"] != original
        or receipt["requested_boundary"] != record["requested_boundary"]
        or active["exposure_sha256"] != _file_sha256(checkpoint / "exposure.json")
        or active["checkpoint_file_sha256"]
        != {name: _file_sha256(checkpoint / name) for name in ACTIVE_CHECKPOINT_FILES}
        or len(entries) != receipt["global_step"]
        or len(entries) != receipt["committed_prompt_presentations"]
        or [entry["global_step"] for entry in entries] != list(range(1, len(entries) + 1))
        or [entry["id"] for entry in entries] != receipt["committed_prompt_ids"]
    ):
        raise LocalVLMError("endpoint checkpoint time/exposure lineage differs")
    egsft = checkpoint / "egsft.json"
    if (
        (record["settings"]["mode"] == "egsft") != egsft.is_file()
        or ("egsft_sha256" in active) != egsft.is_file()
        or (
            egsft.is_file()
            and (
                active["egsft_sha256"] != _file_sha256(egsft)
                or [row["id"] for row in json.loads(egsft.read_text())["entries"]]
                != receipt["committed_prompt_ids"]
            )
        )
    ):
        raise LocalVLMError("endpoint EG-SFT weight lineage differs")
    boundary = receipt["requested_boundary"]
    exhausted = receipt["global_step"] >= original["max_steps"] or (
        original["max_active_seconds"] is not None
        and receipt["committed_active_seconds"] >= original["max_active_seconds"]
    )
    reached = (
        boundary["stop_after_steps"] is not None
        and receipt["global_step"] >= boundary["stop_after_steps"]
    ) or (
        boundary["stop_after_active_seconds"] is not None
        and receipt["committed_active_seconds"] >= boundary["stop_after_active_seconds"]
    )
    if receipt["status"] != ("total_budget_exhausted" if exhausted else "boundary_reached") or not (
        exhausted or reached
    ):
        raise LocalVLMError("endpoint did not reach a declared boundary or total budget")
    expected_ref = record["reference"]["adapter"]["sha256"] if record["reference"] else None
    if receipt["reference_sha256"] != expected_ref:
        raise LocalVLMError("endpoint reference identity differs")
    return receipt


CONTINUATION_SCOPE = (
    "SFT-T: continue a verified SFT checkpoint's adapter, optimizer, scheduler, RNG, global "
    "step and committed active time without reset, on a declared different TRAIN manifest "
    "replayed sequentially from its first row at the parent's next global step"
)
_CONTINUATION_SETTINGS = (
    "mode",
    "learning_rate",
    "beta",
    "seed",
    "device",
    "probe",
    "train_only",
    "init_adapter",
    "initial_adapter_precision_transfer",
)
_CONTINUATION_FILES = ACTIVE_CHECKPOINT_FILES + (
    "rng_state.pth",
    "exposure.json",
    "active_time.json",
)


def _continuation_identity(parent_checkpoint: Path, identity: dict, settings: dict) -> dict:
    """Verify an SFT parent checkpoint and record the declared TRAIN identity change.

    Everything except TRAIN must match the parent's control identity and optimizer
    settings; TRAIN must differ (an unchanged TRAIN is a resume, not a continuation).
    The record binds the parent's old identity, the new manifest identity and the
    hashes of every restored state file, so the change is never silent.
    """
    checkpoint = Path(parent_checkpoint).resolve()
    if any(not (checkpoint / name).is_file() for name in _CONTINUATION_FILES):
        raise ValueError("continuation requires a complete SFT checkpoint with time/exposure")
    if settings["device"] == "mps" and not (checkpoint / "mps_rng_state.pth").is_file():
        raise ValueError("MPS continuation requires the parent's mps_rng_state.pth")
    parent = json.loads((checkpoint.parent / "local_training.json").read_text())
    old, previous = parent["control_identity"], parent["settings"]
    if previous.get("continuation") is not None:
        raise ValueError("continuation parent must be an uncontinued SFT run")
    if any(previous.get(key) != settings[key] for key in _CONTINUATION_SETTINGS) or (
        settings["mode"] != "sft"
    ):
        raise ValueError("continuation must keep the parent's SFT optimizer settings")
    if {k: v for k, v in old.items() if k != "train"} != {
        k: v for k, v in identity.items() if k != "train"
    }:
        raise ValueError("continuation may change only the TRAIN manifest identity")
    if old["train"] == identity["train"]:
        raise ValueError("continuation TRAIN equals the parent's; use resume_from_checkpoint")
    active = _load_active_time(checkpoint, previous["max_active_seconds"], allow_exhausted=True)
    exposure = json.loads((checkpoint / "exposure.json").read_text())
    if [entry["global_step"] for entry in exposure["entries"]] != list(
        range(1, active["global_step"] + 1)
    ):
        raise LocalVLMError("continuation parent exposure differs from its global step")
    # SFT-T continues a FINISHED SFT-E endpoint: a verified receipt that stopped at its
    # declared stop_after_steps (not total-budget exhaustion), from its final checkpoint.
    if not (checkpoint.parent / "endpoint_receipt.json").is_file():
        raise LocalVLMError("continuation parent must be a finished SFT-E endpoint with a receipt")
    receipt = read_endpoint_receipt(checkpoint.parent)
    declared = receipt["requested_boundary"]["stop_after_steps"]
    if (
        receipt["status"] != "boundary_reached"
        or declared is None
        or receipt["global_step"] != declared
    ):
        raise LocalVLMError(
            "continuation parent endpoint must be boundary_reached at its declared "
            "stop_after_steps, not budget-exhausted"
        )
    if (
        (checkpoint.parent / receipt["checkpoint"]["path"]).resolve() != checkpoint
        or receipt["global_step"] != active["global_step"]
    ):
        raise LocalVLMError("continuation must start from the parent's final endpoint checkpoint")
    return {
        "scope": CONTINUATION_SCOPE,
        "parent_output": str(checkpoint.parent),
        "parent_checkpoint": checkpoint.name,
        "parent_global_step": active["global_step"],
        "parent_committed_active_seconds": active["committed_active_seconds"],
        "parent_control_identity_sha256": _object_sha256(old),
        "parent_settings_sha256": _object_sha256(previous),
        "parent_endpoint_receipt_sha256": _file_sha256(checkpoint.parent / "endpoint_receipt.json"),
        "parent_endpoint_status": receipt["status"],
        "parent_train": old["train"],
        "train": identity["train"],
        "parent_checkpoint_file_sha256": {
            name: _file_sha256(checkpoint / name) for name in _CONTINUATION_FILES
        },
        "reset": {"adapter": False, "optimizer": False, "scheduler": False, "rng": False},
        "data_start": "TRAIN row 0 at global step parent_global_step + 1, sequential, cycling",
    }


def gold_nll_improved(record: dict, after_nll: float) -> bool | None:
    """Whether the TRAIN-diagnostic gold NLL fell, or None when there is no pre-score.

    A continuation (SFT-T) records no ``before_train_gold_nll``, so for it this is None:
    nothing scores a pre-training gold NLL, and nothing compares it with the parent. The
    parent's own ``probe.after_train_gold_nll`` is not a substitute, because the 8
    diagnostic rows are drawn from each run's own TRAIN (the parent's accepted manifest,
    SFT-T's scanned one), so the two NLLs are over different rows. Indexing the key raised
    KeyError in every train-only continuation after it had trained and saved its adapter,
    before the endpoint receipt, so SFT-T could never finish; found by the freshpool CPU
    dry run (scripts/freshpool_dryrun).
    """
    before = record.get("before_train_gold_nll")
    return None if before is None else after_nll < before


def _trainer_with_continuation(base, parent_global_step: int):
    """Replay the new TRAIN from row 0 at the parent's next step, across resumes.

    Trainer fast-forwards a resumed dataloader to position global_step mod L. Rotating
    the sequential order by the parent's step maps that position to row
    (global_step - parent_global_step) mod L: row 0 first, then cycling.
    """
    torch = import_module("torch")

    class _RotatedSequentialSampler(torch.utils.data.Sampler):
        def __init__(self, length):
            self.length = length

        def __iter__(self):
            return iter((i - parent_global_step) % self.length for i in range(self.length))

        def __len__(self):
            return self.length

    class ContinuationTrainer(base):
        def _get_train_sampler(self, dataset=None):
            native = super()._get_train_sampler(dataset)
            if not isinstance(native, torch.utils.data.SequentialSampler) or (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
                != 1
            ):
                raise LocalVLMError("continuation requires sequential one-row updates")
            return _RotatedSequentialSampler(len(native))

    return ContinuationTrainer


def train_local(
    train: Path,
    validation: Path | None,
    data_spec: Path,
    output: Path,
    *,
    mode: str = "grpo",
    model: str = DEFAULT_MODEL,
    device: str = "mps",
    model_directory: Path | None = None,
    profile: Profile = Profile(),
    max_steps: int = 64,
    learning_rate: float = 1e-5,
    beta: float = 0.04,
    seed: int = 42,
    save_steps: int = 8,
    init_adapter: Path | None = None,
    resume_from_checkpoint: Path | None = None,
    max_active_seconds: float | None = None,
    probe: bool = False,
    train_only: bool = False,
    stop_after_steps: int | None = None,
    stop_after_active_seconds: float | None = None,
    stage_id: str | None = None,
    finalize_only: bool = False,
    initial_adapter_precision_transfer: str | None = None,
    order: str = "random",
    cycle: bool = False,
    screen_journal: Path | None = None,
    screen_threshold: float = 0.5,
    target_accepted: int = 600,
    screen_tokenizations: Path | None = None,
    continue_from_checkpoint: Path | None = None,
    exploratory_sweep: str | None = None,
    threads: int = DEFAULT_THREADS,
    loss_scale_init: float | None = None,
    label_hierarchy: Path | None = None,
    family_credit: float | None = None,
    manufacturer_credit: float | None = None,
    invalid_reward: float | None = None,
    grpo_overrides: dict | None = None,
    dynamic_sampling_max_draws: int | None = None,
) -> dict:
    """Run offline training. Validation is greedy inference only, never trainer input.

    `device="cuda"` runs on one CUDA GPU. `profile.dtype="float16"` (CUDA only) trains with
    fp16 autocast and a dynamic GradScaler starting at `loss_scale_init` (default 2**16);
    an overflowed step is skipped by the scaler and counted in the optimizer audit.
    `threads` sets torch's intra-op CPU threads (default 2).

    `mode="egsft"` is SFT weighted by stopgrad[w_G(p) / w_G(0.5)]. `order="sequential"`
    replays the TRAIN manifest order for SFT/EG-SFT, and `cycle` lets further epochs
    repeat it. `screen_journal` turns GRPO into the screened scan of the TRAIN
    manifest in file order: each update takes the next candidate whose live-policy
    u(p) >= `screen_threshold`, for exactly `target_accepted` updates, and refuses
    when the scan is exhausted first.

    `continue_from_checkpoint` (SFT-T) continues a different, completed-or-checkpointed
    SFT run's checkpoint without resetting optimizer/scheduler/RNG/step/active time, on
    this declared TRAIN manifest from its first row (sequential, cycling as declared).
    `max_steps` and `max_active_seconds` stay cumulative over the parent's lineage. The
    old and new identities and the restored file hashes enter settings["continuation"];
    resuming the continuation itself passes the same continue_from_checkpoint.

    `label_hierarchy` (GRPO only, default off) swaps the exact reward for hierarchical
    partial credit; `family_credit` and `manufacturer_credit` must then be explicit, and
    `invalid_reward` (default 0) scores a non-EOS or out-of-vocabulary answer.

    Exploratory GRPO options (default-off; absent from settings unless passed):
    `grpo_overrides` sets allowlisted TRL GRPOConfig knobs (GRPO_KNOBS: clip-higher
    epsilon_high, loss_type, scale_rewards, mask_truncated_completions, num_iterations,
    ...). `dynamic_sampling_max_draws` drops all-same reward groups and redraws a
    deterministic TRAIN replacement, up to that many groups per update.
    """
    profile.validate()
    if exploratory_sweep is not None and (
        exploratory_sweep != EXPLORATORY_SWEEP_ID or train_only is not True
    ):
        raise ValueError(f"exploratory_sweep must be {EXPLORATORY_SWEEP_ID!r} with train_only")
    overrides = validate_grpo_overrides(grpo_overrides)
    if (overrides or dynamic_sampling_max_draws is not None) and mode != "grpo":
        raise ValueError("grpo_overrides and dynamic sampling apply only to GRPO")
    if dynamic_sampling_max_draws is not None and (
        type(dynamic_sampling_max_draws) is not int or not 2 <= dynamic_sampling_max_draws <= 64
    ):
        raise ValueError("dynamic_sampling_max_draws must be an integer in 2..64")
    policy_reuse = overrides.get("num_iterations", 1) > 1
    if screen_journal is not None and (policy_reuse or dynamic_sampling_max_draws is not None):
        raise ValueError("screening cannot combine with dynamic sampling or num_iterations > 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if profile.dtype == "float16":
        loss_scale_init = DEFAULT_LOSS_SCALE_INIT if loss_scale_init is None else loss_scale_init
        if (
            type(loss_scale_init) not in (int, float)
            or not math.isfinite(loss_scale_init)
            or loss_scale_init < 1
        ):
            raise ValueError("loss_scale_init must be a finite number >= 1")
        loss_scale_init = float(loss_scale_init)
    elif loss_scale_init is not None:
        raise ValueError("loss_scale_init applies only to float16 training")
    for name, value in (("max_steps", max_steps), ("save_steps", save_steps)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a valid nonnegative/positive integer")
    if mode not in {"grpo", "sft", "egsft"} or model not in MODEL_REVISIONS:
        raise ValueError("choose sft/egsft/grpo and one of the explicitly pinned Qwen models")
    if order not in {"random", "sequential"} or (order == "sequential" and mode == "grpo"):
        raise ValueError("order must be random, or sequential for sft/egsft")
    if type(cycle) is not bool or (cycle and order != "sequential"):
        raise ValueError("cycle requires sequential order")
    screened = screen_journal is not None
    if screened:
        if mode != "grpo":
            raise ValueError("screening applies only to GRPO")
        if (
            type(screen_threshold) not in (int, float)
            or not math.isfinite(screen_threshold)
            or not 0 < screen_threshold < 1
        ):
            raise ValueError("screen_threshold must be finite and in (0, 1)")
        if type(target_accepted) is not int or target_accepted < 1:
            raise ValueError("target_accepted must be a positive integer")
        if max_steps != target_accepted:
            raise ValueError("screened max_steps must equal the fixed target_accepted")
    elif screen_threshold != 0.5 or target_accepted != 600:
        raise ValueError("screen_threshold/target_accepted require screen_journal")
    if screen_tokenizations is not None and not (screened or mode == "egsft"):
        raise ValueError("screen_tokenizations apply only to screening or EG-SFT")
    if continue_from_checkpoint is not None and (mode != "sft" or order != "sequential"):
        raise ValueError("continue_from_checkpoint applies only to sequential SFT")
    if model_directory is None:
        raise ValueError("model_directory must point to a verified local snapshot")
    model_directory = Path(model_directory).resolve()
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-2:
        raise ValueError("learning_rate must be finite and in (0, 0.01]")
    if type(beta) not in (int, float) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0, 1]")
    if max_active_seconds is not None and (
        not math.isfinite(max_active_seconds) or max_active_seconds <= 0
    ):
        raise ValueError("max_active_seconds must be finite and positive")
    if stop_after_steps is not None and (
        type(stop_after_steps) is not int or not 1 <= stop_after_steps <= max_steps
    ):
        raise ValueError("stop_after_steps must be within the original total step budget")
    if stop_after_active_seconds is not None and (
        not math.isfinite(stop_after_active_seconds)
        or stop_after_active_seconds <= 0
        or max_active_seconds is None
        or stop_after_active_seconds > max_active_seconds
    ):
        raise ValueError("stop_after_active_seconds must be within the original total time budget")
    if stage_id is not None and (not isinstance(stage_id, str) or not stage_id.strip()):
        raise ValueError("stage_id must be a nonempty string")
    if type(finalize_only) is not bool or (finalize_only and not resume_from_checkpoint):
        raise ValueError("finalize_only requires an explicit existing checkpoint")
    if initial_adapter_precision_transfer is not None and not init_adapter:
        raise ValueError("initial adapter precision transfer requires a pinned saved adapter")
    spec = load_spec(data_spec)
    shaping = _shaping_from_options(
        spec,
        label_hierarchy,
        family_credit,
        manufacturer_credit,
        invalid_reward,
        mode=mode,
    )
    if type(train_only) is not bool:
        raise ValueError("train_only must be boolean")
    if train_only and (validation is not None or probe):
        raise ValueError("TRAIN-only diagnostic forbids validation/probe")
    if train_only:
        step_cap = (
            None
            if screened or exploratory_sweep
            else TRAIN_ONLY_STEP_CAPS["sft" if mode == "egsft" else mode]
        )
        if step_cap is not None and max_steps > step_cap:
            raise ValueError(f"TRAIN-only {mode} diagnostic exceeds {step_cap} steps")
        if mode != "grpo" and max_steps > 400 and max_active_seconds is None:
            raise ValueError("extended TRAIN-only SFT requires an original finite active-time cap")
    if not train_only and validation is None:
        raise ValueError("validation is required unless train_only is explicit")
    if probe and (not spec.get("diagnostic_only") or max_steps > 8):
        raise ValueError("probe requires diagnostic_only data and at most eight optimizer steps")
    train_rows = load_classification(
        train,
        "train",
        spec["labels"],
        preprocessing_id=spec.get("preprocessing_id"),
        require_all_classes=not train_only,
    )
    row_cap = (
        NATIVE_TRAIN_ROWS
        if exploratory_sweep
        else SCAN_ROW_CAP
        if screened or order == "sequential"
        else 400
    )
    if train_only and len(train_rows) > row_cap:
        raise ValueError(f"TRAIN-only diagnostic is limited to {row_cap} images")
    if screened and len(train_rows) < target_accepted:
        raise ValueError("screen scan is shorter than target_accepted; it would exhaust S")
    parent_step = 0
    if continue_from_checkpoint is not None:
        parent_state = Path(continue_from_checkpoint) / "trainer_state.json"
        if not parent_state.is_file():
            raise ValueError("continuation requires a complete SFT checkpoint with time/exposure")
        parent_step = json.loads(parent_state.read_text())["global_step"]
        if max_steps <= parent_step:
            raise ValueError("continuation max_steps is cumulative and must exceed the parent's")
    if order == "sequential" and not cycle and max_steps - parent_step > len(train_rows):
        raise ValueError("sequential replay without cycle cannot exceed the given sequence")
    tokenizations = _load_tokenizations(screen_tokenizations)
    validation_rows = (
        load_classification(
            validation, "validation", spec["labels"], preprocessing_id=spec.get("preprocessing_id")
        )
        if validation is not None
        else []
    )
    if validation_rows:
        assert_disjoint(train_rows, validation_rows)
    diagnostic_rows = (
        sorted(
            train_rows,
            key=lambda row: hashlib.sha256(
                f"train-diagnostic-v1:{seed}:{row['id']}".encode()
            ).hexdigest(),
        )[:8]
        if train_only
        else train_rows
    )
    evaluation_rows = diagnostic_rows if train_only else validation_rows
    evaluation_name = "train_diagnostic" if train_only else "validation"
    output = Path(output).resolve()
    identity = {
        "model": model,
        "model_revision": MODEL_REVISIONS[model],
        "processor_model": model,
        "processor_revision": MODEL_REVISIONS[model],
        "profile": asdict(profile),
        "output_grammar": output_grammar(profile.response_format),
        "data_spec_sha256": _file_sha256(Path(data_spec)),
        "train": _classification_identity(Path(train), train_rows),
        "validation": _classification_identity(Path(validation), validation_rows)
        if validation is not None
        else None,
        "image_transform_source_sha256": _file_sha256(Path(inspect.getfile(load_aircraft_rgb)))
        if spec.get("preprocessing_id") == PREPROCESSING_ID
        else None,
    }
    if _expanded.is_expanded(profile) and (
        not init_adapter
        or mode not in {"grpo", "sft"}
        or (mode == "grpo" and beta != 0.04)
        or continue_from_checkpoint is not None
    ):
        raise ValueError(
            "expanded training requires the approved init, SFT or GRPO with beta .04, "
            "and no continuation"
        )
    initial = (
        _adapter_identity(
            Path(init_adapter).resolve(),
            identity,
            profile,
            precision_transfer=initial_adapter_precision_transfer,
        )
        if init_adapter
        else None
    )
    settings = {
        "mode": mode,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "beta": beta if mode == "grpo" else None,
        "seed": seed,
        "save_steps": save_steps,
        "max_active_seconds": max_active_seconds,
        "probe": probe,
        "train_only": train_only,
        "init_adapter": initial,
        "device": device,
        "initial_adapter_precision_transfer": initial_adapter_precision_transfer,
    }
    # New options enter settings only when used, so existing runs keep their identity.
    if order != "random":
        settings.update(order=order, cycle=cycle)
    if threads != DEFAULT_THREADS:
        settings["torch_threads"] = threads
    if profile.dtype == "float16":
        settings["loss_scaling"] = {
            "scaler": "torch.amp.GradScaler via accelerate mixed_precision=fp16",
            "autocast": "float16",
            "init_scale": loss_scale_init,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": LOSS_SCALE_GROWTH_INTERVAL,
            "max_consecutive_skips": MAX_CONSECUTIVE_LOSS_SCALE_SKIPS,
        }
    if overrides:
        settings["grpo_overrides"] = overrides
    if dynamic_sampling_max_draws is not None:
        settings["dynamic_sampling"] = {
            "max_draws": dynamic_sampling_max_draws,
            "drop": "groups whose TRL advantages are all zero (all rewards equal)",
            "replacement": "dynamic_replacement_index(seed, global_step, draw, |TRAIN|)",
            "exhausted": "keep the last draw (KL-only update), journaled",
            "journal": "dynamic_sampling.jsonl",
        }
    if screened:
        screen_journal = Path(screen_journal).resolve()
        if not screen_journal.is_relative_to(output) or screen_journal == output:
            raise ValueError("screen_journal must be a file inside output")
        settings["screening"] = {
            "journal": str(screen_journal.relative_to(output)),
            "threshold": screen_threshold,
            "target_accepted": target_accepted,
            "group_size": profile.num_generations,
            "acceptance": "u(p) = 1 - p^G - (1 - p)^G >= threshold",
            "scan_order": "TRAIN manifest file order",
            "scope": SCREEN_SCOPE,
        }
    if mode == "egsft":
        settings["egsft"] = {
            "group_size": profile.num_generations,
            "advantage_eps": TRL_ADVANTAGE_EPS,
            "weight": "stopgrad[w_G(p) / w_G(0.5)], TRL Bessel-corrected group std",
        }
    if exploratory_sweep is not None:
        settings["exploratory_sweep"] = exploratory_sweep
    if screen_tokenizations is not None:
        settings["gold_tokenizations_sha256"] = _file_sha256(Path(screen_tokenizations))
    if shaping is not None:
        settings["reward_shaping"] = shaping_identity(shaping)
    continuation = None
    if continue_from_checkpoint is not None:
        continuation = _continuation_identity(Path(continue_from_checkpoint), identity, settings)
        if continuation["parent_global_step"] != parent_step:
            raise LocalVLMError("continuation parent changed while it was being verified")
        if Path(continuation["parent_output"]).is_relative_to(output) or output.is_relative_to(
            Path(continuation["parent_output"])
        ):
            raise ValueError("continuation output must be separate from the parent's output")
        settings["continuation"] = continuation
    checkpoint, previous, active_state, exposure_state = None, None, None, None
    screen_state, egsft_state, inherited_exposure = None, None, None
    if continuation is not None:
        parent_checkpoint = Path(continue_from_checkpoint).resolve()
        inherited_exposure = json.loads((parent_checkpoint / "exposure.json").read_text())[
            "entries"
        ]
        if stop_after_steps is not None and stop_after_steps <= parent_step:
            raise ValueError("continuation stop_after_steps is cumulative past the parent's")
        if not resume_from_checkpoint:
            active_state = _load_active_time(
                parent_checkpoint,
                json.loads((parent_checkpoint.parent / "local_training.json").read_text())[
                    "settings"
                ]["max_active_seconds"],
                allow_exhausted=True,
            )
            if max_active_seconds is not None and (
                active_state["committed_active_seconds"] >= max_active_seconds
            ):
                raise LocalVLMError("continuation parent already used the cumulative time budget")
            exposure_state = {"entries": inherited_exposure}
    if resume_from_checkpoint:
        checkpoint = Path(resume_from_checkpoint).resolve()
        if not checkpoint.is_relative_to(output) or any(
            not (checkpoint / name).is_file()
            for name in (
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "adapter_model.safetensors",
            )
        ):
            raise ValueError(
                "resume requires complete optimizer/scheduler/RNG checkpoint in output"
            )
        if device == "mps" and not (checkpoint / "mps_rng_state.pth").is_file():
            raise ValueError("MPS resume requires the saved mps_rng_state.pth")
        if profile.dtype == "float16" and not (checkpoint / "scaler.pt").is_file():
            raise ValueError("float16 resume requires the saved loss-scaler scaler.pt")
        if (
            mode == "grpo"
            and beta > 0
            and not (checkpoint / "ref/adapter_model.safetensors").is_file()
        ):
            raise ValueError("positive-beta resume requires the frozen ref adapter checkpoint")
        if _expanded.is_expanded(profile):
            _expanded.validate_checkpoint(checkpoint, initial, mode)
        previous = json.loads((output / "local_training.json").read_text())
        if previous["control_identity"] != identity or previous["settings"] != settings:
            raise ValueError("resume must preserve exact model/data/profile/settings")
        active_state = _load_active_time(
            checkpoint, max_active_seconds, allow_exhausted=finalize_only
        )
        if active_state.get("exposure_sha256") is None:
            raise LocalVLMError("resume requires checkpoint-bound actual prompt exposure")
        exposure_state = json.loads((checkpoint / "exposure.json").read_text())
        if len(exposure_state.get("entries", [])) != active_state["global_step"]:
            raise LocalVLMError("checkpoint exposure count differs from optimizer global step")
        if screened != (checkpoint / "screen.json").is_file():
            raise LocalVLMError("screened resume requires exactly a checkpoint screen state")
        if screened:
            screen_state = json.loads((checkpoint / "screen.json").read_text())
        if (mode == "egsft") != (checkpoint / "egsft.json").is_file():
            raise LocalVLMError("EG-SFT resume requires exactly a checkpoint weight journal")
        if mode == "egsft":
            egsft_state = json.loads((checkpoint / "egsft.json").read_text())
            if [row["id"] for row in egsft_state["entries"]] != [
                entry["id"] for entry in exposure_state["entries"]
            ]:
                raise LocalVLMError("checkpoint EG-SFT weights differ from committed exposure")
        boundary_reached = (
            (stop_after_steps is not None and active_state["global_step"] >= stop_after_steps)
            or (
                stop_after_active_seconds is not None
                and active_state["committed_active_seconds"] >= stop_after_active_seconds
            )
            or active_state["global_step"] >= max_steps
            or (
                max_active_seconds is not None
                and active_state["committed_active_seconds"] >= max_active_seconds
            )
        )
        if boundary_reached and not finalize_only:
            raise LocalVLMError("requested cumulative stage boundary is already reached")
        if finalize_only and not boundary_reached:
            raise LocalVLMError("finalize_only requires an already-reached stage boundary or cap")
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output is not empty; use a new directory or complete checkpoint")
    exposure = _ExposureLedger(
        output, [row["id"] for row in train_rows], exposure_state, inherited=inherited_exposure
    )
    screen = (
        _ScreenedScan(
            screen_journal,
            [row["id"] for row in train_rows],
            threshold=screen_threshold,
            target_accepted=target_accepted,
            group_size=profile.num_generations,
            resume_state=screen_state,
        )
        if screened
        else None
    )
    if screen is not None:
        if screen.accepted_ids != [entry["id"] for entry in exposure.entries]:
            raise LocalVLMError("checkpoint screen accepted sequence differs from exposure")
        if screen.position == len(train_rows) and len(screen.accepted_ids) < target_accepted:
            raise LocalVLMError("screened scan already exhausted S before target_accepted")
    environment = preflight(device)
    check_device_dtype(device, profile.dtype, environment, training=True)
    if _expanded.is_expanded(profile):
        # Binds the whole package (contract, converter, grader, siblings) to the endpoint.
        environment["expanded_source_closure"] = _expanded.source_closure()
    if previous and previous["environment"] != environment:
        raise ValueError("resume requires the original environment and adapter source")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "running",
        "algorithm": f"TRL-{mode.upper()}",
        "regularization_profile": ("initial-policy-KL" if beta else "beta-zero-ablation")
        if mode == "grpo"
        else "completion-only-SFT"
        if mode == "sft"
        else "expected-gradient-weighted-completion-only-SFT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "control_identity": identity,
        "settings": settings,
        "environment": environment,
        "sources": spec["sources"],
        "resumed_from": str(checkpoint) if checkpoint else None,
        "claim_scope": "TRAIN learning diagnostic only; no held-out result"
        if train_only
        else "diagnostic only"
        if probe
        else "development validation only",
        "train_diagnostic_ids": [row["id"] for row in diagnostic_rows] if train_only else None,
        "held_out_benchmark_improvement": "unmeasured",
        "audit_scope": "current invocation; append-only journals retain earlier invocations",
        "reference": previous.get("reference") if previous else None,
        "stage_id": stage_id,
        "finalize_only": finalize_only,
        "requested_boundary": {
            "stop_after_steps": stop_after_steps,
            "stop_after_active_seconds": stop_after_active_seconds,
        },
    }
    _write(output / "local_training.json", record)
    _archive_source(output, environment["adapter_source_sha256"])
    audit = None
    try:
        torch = import_module("torch")
        # An explicit --threads wins; otherwise VLM_TORCH_THREADS (or the historical 2).
        _set_threads(torch, threads if threads != DEFAULT_THREADS else _torch_threads())
        transformers, peft, trl = (import_module(x) for x in ("transformers", "peft", "trl"))
        transformers.set_seed(seed)
        record["model_source"] = _verified_model_directory(model_directory, model)
        processor = _processor(model_directory, profile)
        record["processor_audit"] = _audit_processor(
            processor, train_rows + validation_rows, spec, profile
        )
        record_mode = "sft" if mode == "egsft" else mode
        dataset = _dataset(train_rows, spec, record_mode, profile)
        policy = transformers.AutoModelForImageTextToText.from_pretrained(
            str(model_directory),
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            dtype=getattr(torch, profile.dtype),
            attn_implementation="sdpa",
        )
        if policy.config.model_type not in {"qwen2_vl", "qwen2_5_vl"}:
            raise LocalVLMError("unexpected cached model architecture")
        lora = peft.LoraConfig(
            r=profile.lora_rank,
            lora_alpha=2 * profile.lora_rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        policy = (
            _expanded.attach_numeric(
                policy,
                Path(init_adapter).resolve(),
                expected_weights_sha256=initial["weights_sha256"],
                is_trainable=True,
            )
            if _expanded.is_expanded(profile)
            else peft.PeftModel.from_pretrained(policy, str(init_adapter), is_trainable=True)
            if init_adapter
            else peft.get_peft_model(policy, lora)
        )
        policy.to(device)
        if initial:
            record["initial_adapter_tensor_audit"] = _verify_initial_adapter_tensors(
                policy, Path(init_adapter)
            )
        record["parameter_dtypes"] = {
            "trainable": dict(
                Counter(
                    str(parameter.dtype)
                    for parameter in policy.parameters()
                    if parameter.requires_grad
                )
            ),
            "frozen": dict(
                Counter(
                    str(parameter.dtype)
                    for parameter in policy.parameters()
                    if not parameter.requires_grad
                )
            ),
        }
        for layer in policy.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.0
        kwargs = trainer_kwargs(
            record_mode,
            output,
            profile,
            device=device,
            max_steps=max_steps,
            learning_rate=learning_rate,
            seed=seed,
            save_steps=save_steps,
            beta=beta,
            **({"order": order} if order != "random" else {}),
            **({"grpo_overrides": overrides} if overrides else {}),
        )
        args = (trl.GRPOConfig if mode == "grpo" else trl.SFTConfig)(**kwargs)
        if args.device.type != device:
            raise LocalVLMError(f"Accelerate selected {args.device}, expected {device}")
        audit = _audit_callback(
            output,
            max_active_seconds,
            device=device,
            resume_state=active_state,
            exposure=exposure,
            stop_after_steps=stop_after_steps,
            stop_after_active_seconds=stop_after_active_seconds,
            **({"screen": screen} if screen is not None else {}),
            **({"loss_scaled": True} if profile.dtype == "float16" else {}),
        )
        eos = policy.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        eos = sorted(set(eos + [processor.tokenizer.eos_token_id]))
        relay = (
            _PromptRelay(audit.observe_prompt)
            if mode == "grpo" and (policy_reuse or dynamic_sampling_max_draws is not None)
            else None
        )
        reward = make_reward(
            spec["labels"],
            profile.num_generations,
            eos,
            output / "reward_groups.jsonl",
            allow_reasoning=True,
            journal_io_context=audit.timer.exclude_journal_io,
            observe_prompt=relay.note if relay is not None else audit.observe_prompt,
            **({"shaping": shaping} if shaping is not None else {}),
            **_format_kwargs(profile),
        )
        dynamic_summary = {} if dynamic_sampling_max_draws is not None else None
        constructor = trl.GRPOTrainer if mode == "grpo" else trl.SFTTrainer
        if mode == "grpo":
            constructor = _trainer_with_grpo_resume(
                constructor,
                expected_reference=previous["reference"]["adapter"]
                if previous and beta > 0
                else None,
            )
            if screen is not None:
                constructor = _trainer_with_screening(constructor, screen, dataset)
            if policy_reuse:
                constructor = _trainer_with_policy_reuse(constructor, relay)
            if dynamic_sampling_max_draws is not None:
                constructor = _trainer_with_dynamic_sampling(
                    constructor,
                    relay,
                    dataset,
                    max_draws=dynamic_sampling_max_draws,
                    seed=seed,
                    journal=output / "dynamic_sampling.jsonl",
                    journal_io_context=audit.timer.exclude_journal_io,
                    summary=dynamic_summary,
                )
        else:
            if mode == "egsft":
                constructor = _trainer_with_egsft(
                    constructor,
                    eos_token_ids=eos,
                    suppress_tokens=[policy.config.image_token_id],
                    temperature=profile.temperature,
                    group_size=profile.num_generations,
                    journal=output / "egsft_weights.jsonl",
                    tokenizations=tokenizations,
                    labels_by_id={row["id"]: row["label"] for row in train_rows},
                    journal_io_context=audit.timer.exclude_journal_io,
                    resume_state=egsft_state,
                )
            constructor = _trainer_with_sft_exposure(constructor, audit)
            if continuation is not None:
                constructor = _trainer_with_continuation(
                    constructor, continuation["parent_global_step"]
                )
        constructor = _trainer_with_mps_rng(constructor, device)
        if profile.dtype == "float16":
            constructor = _trainer_with_loss_scaling(constructor, loss_scale_init)
        constructor = _trainer_with_active_time(constructor)
        trainer = constructor(
            model=policy,
            args=args,
            processing_class=processor,
            # Screened updates resolve slots from `dataset`; placeholders are never decoded.
            train_dataset=_dataset(train_rows, spec, mode, profile, lazy_decode=False)
            if screen is not None
            else dataset,
            callbacks=[audit],
            **({"reward_funcs": reward} if mode == "grpo" else {}),
        )
        if screen is not None:
            transform = _sampler_transform(trainer)
            if transform["suppress_tokens"] != [policy.config.image_token_id]:
                raise LocalVLMError("screen expects TRL to suppress exactly the image token")

            def score(model, index):
                row = train_rows[index]
                return gold_probability(
                    model,
                    processor,
                    row,
                    spec,
                    profile,
                    device,
                    eos_token_ids=transform["eos_token_ids"],
                    suppress_tokens=transform["suppress_tokens"],
                    alternatives=tokenizations.get(row["label"], ()),
                )

            screen.score = score
            record["screen_sampler_transform"] = transform
        if mode != "grpo":
            trainer.data_collator = _sft_collator_with_ids(trainer.data_collator)
        reference = (
            _reference_identity(trainer, init_adapter=initial, model=model)
            if mode == "grpo" and beta > 0
            else None
        )
        if previous and reference != previous["reference"]:
            raise LocalVLMError("resume reference identity differs from the original run")
        record["reference"] = reference
        audit.expected_reference = reference["adapter"] if reference else None
        record["trainable_parameters"] = {
            "count": sum(
                parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
            ),
            "names": [
                name for name, parameter in policy.named_parameters() if parameter.requires_grad
            ],
        }
        # Trainer owns restoration. Avoid labeling fresh initialization as a resumed pre-score.
        if continuation is not None and not checkpoint:
            # The policy is still the fresh LoRA here, so no pre-score is taken; the probe's
            # gold_nll_improved is None for a continuation (see gold_nll_improved).
            record["before_note"] = "continuation: pre-score is the parent run's endpoint"
        elif not checkpoint:
            before, predictions = _evaluate(
                policy, processor, evaluation_rows, spec, profile, device
            )
            record[f"before_{evaluation_name}"] = before
            _write(output / f"before_{evaluation_name}_predictions.json", predictions)
            if probe or train_only:
                record["before_train_gold_nll"] = _gold_nll(
                    policy, processor, diagnostic_rows, spec, profile, device
                )
                if mode == "grpo":
                    record["sampler_teacher_logprob_parity"] = _sampler_parity(
                        trainer, processor, train_rows[0], spec, profile, device
                    )
        else:
            for key in (
                "before_validation",
                "before_train_diagnostic",
                "before_train_gold_nll",
                "sampler_teacher_logprob_parity",
            ):
                if key in previous:
                    record[key] = previous[key]
        _write(output / "local_training.json", record)
        if finalize_only:
            _restore_endpoint_without_training(trainer, checkpoint, audit)
            train_metrics = {
                "endpoint_recovery_only": True,
                "optimizer_steps_this_invocation": 0,
                "global_step": trainer.state.global_step,
            }
        else:
            start = checkpoint or (
                Path(continue_from_checkpoint).resolve() if continuation is not None else None
            )
            trained = trainer.train(resume_from_checkpoint=str(start) if start else None)
            train_metrics = trained.metrics
        after, predictions = _evaluate(policy, processor, evaluation_rows, spec, profile, device)
        _write(output / f"after_{evaluation_name}_predictions.json", predictions)
        adapter = output / "final_adapter"
        trainer.save_model(str(adapter))
        processor.save_pretrained(str(adapter))
        trainer.save_state()
        if device == "cuda":
            record["cuda_memory"] = _cuda_memory(device)
        record.update(
            status="completed",
            train_metrics=train_metrics,
            optimizer_audit=audit.summary,
            reward_audit=reward.stats if mode == "grpo" else None,
            **({"dynamic_sampling": dynamic_summary} if dynamic_summary is not None else {}),
            **(
                {
                    "screen": {k: v for k, v in screen.state().items() if k != "entries"},
                    "screen_journal_lineage": "canonical = final checkpoint screen.json "
                    "(committed_screen_entries); the physical jsonl includes discarded spans",
                }
                if screen is not None
                else {}
            ),
            final_adapter=str(adapter),
            final_adapter_sha256=_file_sha256(adapter / "adapter_model.safetensors"),
        )
        record[f"after_{evaluation_name}"] = after
        if probe or train_only:
            after_nll = _gold_nll(policy, processor, diagnostic_rows, spec, profile, device)
            # Reload the saved adapter into the same frozen base to verify persistence without
            # allocating a second VLM. The optimizer is no longer used after this point.
            base = policy.unload()
            reloaded = (
                _expanded._attach_values(
                    base,
                    _expanded.read_numeric(adapter / "adapter_model.safetensors")[0],
                    layout=_expanded.active_layout(),
                    is_trainable=False,
                )
                if _expanded.is_expanded(profile)
                else peft.PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
            )
            reload_nll = _gold_nll(reloaded, processor, diagnostic_rows, spec, profile, device)
            record["probe"] = {
                "after_train_gold_nll": after_nll,
                "reloaded_train_gold_nll": reload_nll,
                "reload_nll_agrees": math.isclose(
                    after_nll, reload_nll, rel_tol=1e-5, abs_tol=1e-5
                ),
                "gold_nll_improved": gold_nll_improved(record, after_nll),
                "gradient_and_weight_change": audit.summary["nonzero_gradient_steps"] > 0
                and audit.summary["changed_tensors"] > 0,
                "mixed_reward_groups": reward.stats["mixed_groups"] if mode == "grpo" else None,
                "sampler_teacher_logprob_parity": record.get("sampler_teacher_logprob_parity"),
                "held_out_benchmark_gain": "not measured by this diagnostic",
            }
        _write(output / "local_training.json", record)
        _publish_endpoint_receipt(output, record, trainer.state.global_step)
    except BaseException as exc:
        if audit is not None:
            try:
                audit.timer.interrupt()
                record["active_time_audit"] = audit.timer.summary()
            except Exception as timing_exc:
                record["active_time_error"] = str(timing_exc)
        if device == "cuda":
            try:
                record["cuda_memory"] = _cuda_memory(device)
            except Exception as memory_exc:
                record["cuda_memory_error"] = str(memory_exc)
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _write(output / "local_training.json", record)
        raise
    if _expanded.is_expanded(profile):
        # Outside the try: a refusal here (e.g. vision_rl source changed during the run)
        # must not rewrite the completed native record that the receipt already binds.
        # The verifier's pin is the SHA-256 of the expanded_endpoint.json written here.
        _expanded.publish_endpoint(output)
    return record


RFT_SCOPE = "TRAIN-only rejection-sampling fine-tuning (expert iteration); no held-out claim"


def rft_round_prompt_ids(
    ids: list[str], *, seed: int, round_index: int, per_round: int
) -> list[str]:
    """Deterministic prompt slice for one on-policy round.

    Positions ``round_index * per_round ...`` walk a seeded permutation of the pool, one
    fresh permutation per pass, so rounds see new prompts until the pool is exhausted.
    """
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("RFT pool requires unique nonempty IDs")
    if type(per_round) is not int or not 1 <= per_round <= len(ids):
        raise ValueError("prompts_per_round must be in 1..pool size")
    permutations: dict[int, list[str]] = {}
    chosen = []
    for position in range(round_index * per_round, (round_index + 1) * per_round):
        epoch, offset = divmod(position, len(ids))
        if epoch not in permutations:
            permutations[epoch] = sorted(
                ids,
                key=lambda identifier: hashlib.sha256(
                    f"rft-order-v1:{seed}:{epoch}:{identifier}".encode()
                ).hexdigest(),
            )
        chosen.append(permutations[epoch][offset])
    return chosen


def rft_select(
    samples: list[dict],
    label: str,
    labels: list[str],
    *,
    max_keep: int,
    skip_all_correct: bool = False,
) -> tuple[list[str], dict]:
    """Keep exact-EOS correct completions (distinct texts, sample order), at most max_keep."""
    if not samples:
        raise ValueError("RFT selection requires sampled completions")
    correct = [
        sample["response"]
        for sample in samples
        if sample["terminated"]
        and parse_label(sample["response"], labels, allow_reasoning=True) == label
    ]
    stats = {
        "samples": len(samples),
        "correct": len(correct),
        "pass_rate": len(correct) / len(samples),
        "skipped_all_correct": False,
    }
    if skip_all_correct and len(correct) == len(samples):
        stats["skipped_all_correct"] = True
        return [], stats
    kept = list(dict.fromkeys(correct))[:max_keep]
    stats["kept"] = len(kept)
    return kept, stats


def rft_records(
    kept: dict[str, list[str]], rows_by_id: dict[str, dict], spec: dict, response_format: str
) -> list[dict]:
    """SFT records whose completion is the policy's own correct sample, not the gold label."""
    prompt = prompt_text(spec, response_format)
    records = []
    for identifier, completions in kept.items():
        row = rows_by_id[identifier]
        for index, text in enumerate(completions):
            records.append(
                {
                    "id": f"{identifier}#{index}",
                    "image": row["image"],
                    "label": row["label"],
                    "prompt": [{"role": "user", "content": prompt}],
                    "completion": [{"role": "assistant", "content": text}],
                }
            )
    return records


def _sample_groups_batched(policy, processor, rows, spec, profile, device, output, reward, config):
    """K draws per prompt in ONE generate call (num_return_sequences=K); no update.

    Same released-TRL sampler settings as `_sample_task_groups`, batched over the group,
    so RFT rounds sample K-fold faster. Each completion is cut after its first EOS, as
    TRL's completion mask does. Journals one JSON line per prompt.
    """
    torch = import_module("torch")
    groups = {}
    policy.eval()
    journal = Path(output) / "sampled_completions.jsonl"
    with torch.inference_mode():
        for row in rows:
            started = time.monotonic()
            inputs = _inputs(processor, row, spec, profile).to(device)
            prompt_length = int(inputs["input_ids"].shape[-1])
            tokens = policy.generate(
                **inputs,
                generation_config=config,
                num_return_sequences=profile.num_generations,
            )
            latency = time.monotonic() - started
            sequences = tokens[:, prompt_length:].tolist()
            if len(sequences) != profile.num_generations:
                raise LocalVLMError("batched sampler returned the wrong number of completions")
            samples = []
            for tail in sequences:
                for index, token in enumerate(tail):
                    if token in config.eos_token_id:
                        tail = tail[: index + 1]
                        break
                samples.append(
                    {
                        "response": processor.tokenizer.decode(tail, skip_special_tokens=True),
                        "raw_response": processor.tokenizer.decode(tail, skip_special_tokens=False),
                        "completion_ids": tail,
                        "terminated": bool(tail and tail[-1] in config.eos_token_id),
                        "prompt_tokens": prompt_length,
                        "completion_tokens": len(tail),
                        "group_latency_seconds": latency,
                    }
                )
            groups[row["id"]] = samples
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": row["id"], "samples": samples}) + "\n")
            reward(
                completions=[sample["response"] for sample in samples],
                completion_ids=[sample["completion_ids"] for sample in samples],
                label=[row["label"]] * profile.num_generations,
                id=[row["id"]] * profile.num_generations,
            )
    return groups


def _rft_dataset(records: list[dict], spec: dict) -> Any:
    datasets = import_module("datasets")
    dataset = datasets.Dataset.from_list(records).cast_column(
        "image", datasets.Image(mode="RGB", decode=False)
    )

    def decode(batch):
        batch["image"] = [_load_rgb(entry["path"], spec) for entry in batch["image"]]
        return batch

    return dataset.with_transform(decode, columns=["image"], output_all_columns=True)


class _RFTClock:
    """Cumulative sampling + optimizer wall seconds across rounds and invocations."""

    def __init__(self, committed: float, budget: float | None):
        self.committed, self.budget, self._started = committed, budget, None

    def begin(self):
        self._started = time.monotonic()

    def elapsed(self) -> float:
        running = time.monotonic() - self._started if self._started is not None else 0.0
        return self.committed + running

    def end(self) -> float:
        seconds = time.monotonic() - self._started
        self.committed += seconds
        self._started = None
        return seconds

    def exhausted(self) -> bool:
        return self.budget is not None and self.elapsed() >= self.budget


def _rft_sft_round(
    policy,
    processing_class,
    dataset,
    profile: Profile,
    output: Path,
    *,
    device: str,
    steps: int,
    learning_rate: float,
    seed: int,
    clock: _RFTClock | None = None,
) -> dict:
    """One completion-only SFT pass over this round's kept samples (fresh AdamW)."""
    trl = import_module("trl")
    transformers = import_module("transformers")
    kwargs = trainer_kwargs(
        "sft",
        output,
        profile,
        device=device,
        max_steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        save_steps=steps,
    )
    # Rounds commit their own adapters; trainer checkpoints would only duplicate them.
    kwargs.update(save_strategy="no")
    args = trl.SFTConfig(**kwargs)
    if args.device.type != device:
        raise LocalVLMError(f"Accelerate selected {args.device}, expected {device}")

    class Budget(transformers.TrainerCallback):
        stopped_by_budget = False
        skipped = consecutive = 0

        def on_step_end(self, args, state, control, optimizer=None, **_):
            if profile.dtype == "float16":
                # A GradScaler overflow skips the optimizer step but still advances
                # global_step: count it, and refuse a nonfinite run as train_local does.
                if getattr(optimizer, "step_was_skipped", False):
                    Budget.skipped += 1
                    Budget.consecutive += 1
                    if Budget.consecutive > MAX_CONSECUTIVE_LOSS_SCALE_SKIPS:
                        raise LocalVLMError(
                            "float16 gradients overflowed on every recent step; not loss-scale "
                            "warm-up but a nonfinite forward/backward"
                        )
                else:
                    Budget.consecutive = 0
            if clock is not None and clock.exhausted():
                Budget.stopped_by_budget = True
                control.should_training_stop = True

    constructor = trl.SFTTrainer
    if profile.dtype == "float16":
        # CUDA float16: accelerate fp16 autocast + GradScaler, as train_local does.
        constructor = _trainer_with_loss_scaling(constructor, DEFAULT_LOSS_SCALE_INIT)
    trainer = constructor(
        model=policy,
        args=args,
        processing_class=processing_class,
        train_dataset=dataset,
        callbacks=[Budget()],
    )
    result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    out = {
        "optimizer_steps": trainer.state.global_step,
        "planned_steps": steps,
        "stopped_by_time_budget": Budget.stopped_by_budget,
        "train_loss": result.training_loss,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
    }
    if profile.dtype == "float16":  # presentations stay optimizer_steps; skips are explicit
        out["loss_scale_skipped_steps"] = Budget.skipped
    return out


def _peft_weights_sha256(adapter: Path) -> str:
    return _file_sha256(adapter / "adapter_model.safetensors")


def _load_round_adapter(policy, adapter: Path) -> None:
    """Overwrite the live trainable LoRA with a committed round adapter, exactly."""
    torch = import_module("torch")
    peft = import_module("peft")
    saved = import_module("safetensors.torch").load_file(
        str(adapter / "adapter_model.safetensors"), device="cpu"
    )
    peft.set_peft_model_state_dict(policy, saved, adapter_name="default")
    actual = peft.get_peft_model_state_dict(policy, adapter_name="default")
    if set(saved) != set(actual) or any(
        not torch.equal(saved[name], actual[name].detach().cpu().to(saved[name].dtype))
        for name in saved
    ):
        raise LocalVLMError("restored round adapter differs from its committed tensors")


def rft_local(
    train: Path,
    data_spec: Path,
    output: Path,
    *,
    model_directory: Path,
    rounds: int,
    prompts_per_round: int,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    profile: Profile = Profile(),
    init_adapter: Path | None = None,
    initial_adapter_precision_transfer: str | None = None,
    learning_rate: float = 1e-5,
    seed: int = 42,
    epochs_per_round: int = 1,
    max_keep_per_prompt: int = 1,
    skip_all_correct: bool = False,
    max_steps: int | None = None,
    max_active_seconds: float | None = None,
    stop_after_rounds: int | None = None,
    stop_after_steps: int | None = None,
    threads: int = DEFAULT_THREADS,
    _sample=None,
    _train_round=None,
) -> dict:
    """Rejection-sampling fine-tuning / expert iteration on TRAIN only.

    Each round r (on-policy: the CURRENT adapter samples):
      1. take `prompts_per_round` TRAIN prompts (`rft_round_prompt_ids`),
      2. draw K = profile.num_generations completions each with released-TRL sampler
         settings (`_task_group_generation_config`, temperature from the profile),
      3. keep exact-EOS correct completions (distinct, <= `max_keep_per_prompt`; with
         `skip_all_correct`, drop prompts the policy already solves K/K),
      4. run completion-only SFT on the kept (prompt, own completion) pairs for
         `epochs_per_round` passes, one example per update, fresh AdamW, constant LR,
      5. commit round-NN/adapter and round-NN/round.json.
    Stops after `rounds`, or when the cumulative optimizer-update cap `max_steps` or the
    cumulative sampling+training wall budget `max_active_seconds` is reached; both are
    for fairness against step- or time-matched SFT. Restarting with identical arguments
    resumes after the last committed round; an uncommitted round is moved aside and
    redone from its seeded start. The endpoint is output/final_adapter, bound by
    output/local_training.json like a train_local endpoint, so the TRAIN assay and a
    later GRPO warm start accept it.

    `stop_after_steps` (default off; not part of the settings identity) is a staged
    boundary: before starting a round, stop once the cumulative optimizer updates reach
    it, and publish that endpoint (status "completed", stop_reason "stop_after_steps").
    Rounds are never cut, so the trajectory does not depend on where it was stopped; a
    later call with the same settings and a larger (or no) boundary continues it.
    `device="cuda"` runs on one GPU under the same device/dtype rules as train_local
    (bfloat16 training needs compute capability >= 8.0; float16 uses GradScaler).
    """
    profile.validate()
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if _expanded.is_expanded(profile):
        raise ValueError("RFT supports the legacy Q/V LoRA profile only")
    if profile.response_format not in LEGACY_RESPONSE_FORMATS:
        raise ValueError("RFT supports the legacy response formats only")
    if model not in MODEL_REVISIONS:
        raise ValueError("choose one of the explicitly pinned Qwen models")
    for name, value in (
        ("rounds", rounds),
        ("prompts_per_round", prompts_per_round),
        ("epochs_per_round", epochs_per_round),
        ("max_keep_per_prompt", max_keep_per_prompt),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(skip_all_correct) is not bool:
        raise ValueError("skip_all_correct must be boolean")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer when given")
    if max_active_seconds is not None and (
        not math.isfinite(max_active_seconds) or max_active_seconds <= 0
    ):
        raise ValueError("max_active_seconds must be finite and positive")
    if stop_after_rounds is not None and (
        type(stop_after_rounds) is not int or not 1 <= stop_after_rounds <= rounds
    ):
        raise ValueError("stop_after_rounds must be within rounds")
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError("stop_after_steps must be a positive integer when given")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-2:
        raise ValueError("learning_rate must be finite and in (0, 0.01]")
    if initial_adapter_precision_transfer is not None and not init_adapter:
        raise ValueError("initial adapter precision transfer requires a pinned saved adapter")
    if device not in RFT_DEVICES:
        raise ValueError(f"RFT device must be one of {RFT_DEVICES}")
    spec = load_spec(data_spec)
    rows = load_classification(
        train,
        "train",
        spec["labels"],
        preprocessing_id=spec.get("preprocessing_id"),
        require_all_classes=False,
    )
    if len(rows) > SCAN_ROW_CAP:
        raise ValueError(f"RFT pool is limited to {SCAN_ROW_CAP} TRAIN images")
    if prompts_per_round > len(rows):
        raise ValueError("prompts_per_round exceeds the TRAIN pool")
    if model_directory is None:
        raise ValueError("model_directory must point to a verified local snapshot")
    model_directory = Path(model_directory).resolve()
    output = Path(output).resolve()
    identity = {
        "model": model,
        "model_revision": MODEL_REVISIONS[model],
        "processor_model": model,
        "processor_revision": MODEL_REVISIONS[model],
        "profile": asdict(profile),
        "output_grammar": OUTPUT_GRAMMAR,
        "data_spec_sha256": _file_sha256(Path(data_spec)),
        "train": _classification_identity(Path(train), rows),
        "validation": None,
        "image_transform_source_sha256": _file_sha256(Path(inspect.getfile(load_aircraft_rgb)))
        if spec.get("preprocessing_id") == PREPROCESSING_ID
        else None,
    }
    initial = (
        _adapter_identity(
            Path(init_adapter).resolve(),
            identity,
            profile,
            precision_transfer=initial_adapter_precision_transfer,
        )
        if init_adapter
        else None
    )
    settings = {
        "mode": "rft",
        "rounds": rounds,
        "prompts_per_round": prompts_per_round,
        "samples_per_prompt": profile.num_generations,
        "epochs_per_round": epochs_per_round,
        "max_keep_per_prompt": max_keep_per_prompt,
        "skip_all_correct": skip_all_correct,
        "keep_rule": "exact-EOS canonical label == gold (the GRPO exact reward)",
        "max_steps": max_steps,
        "max_active_seconds": max_active_seconds,
        "learning_rate": learning_rate,
        "seed": seed,
        "init_adapter": initial,
        "initial_adapter_precision_transfer": initial_adapter_precision_transfer,
        "device": device,
        "optimizer": "fresh AdamW per round, constant LR, one example per update",
        "prompt_order": "rft-order-v1 seeded permutation per pass over the TRAIN pool",
        "sampler": "released-TRL settings, num_return_sequences=K per prompt",
    }
    if profile.dtype == "float16":
        settings["loss_scaling"] = {
            "scaler": "torch.amp.GradScaler via accelerate mixed_precision=fp16",
            "autocast": "float16",
            "init_scale": DEFAULT_LOSS_SCALE_INIT,
            "growth_interval": LOSS_SCALE_GROWTH_INTERVAL,
        }
    previous = None
    if (output / "local_training.json").is_file():
        previous = json.loads((output / "local_training.json").read_text())
        if previous.get("settings") != settings or previous.get("control_identity") != identity:
            raise ValueError("RFT resume must preserve exact model/data/profile/settings")
        if previous.get("status") == "completed" and previous.get("stop_reason") != (
            "stop_after_steps"
        ):
            raise LocalVLMError("RFT run already completed")
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output is not empty; use a new directory or resume the same RFT run")
    environment = preflight(device)
    check_device_dtype(device, profile.dtype, environment, training=True)
    if previous and previous["environment"] != environment:
        raise ValueError("RFT resume requires the original environment and adapter source")
    output.mkdir(parents=True, exist_ok=True)
    committed = []
    for index in range(rounds):
        round_dir = output / f"round-{index:02d}"
        receipt = round_dir / "round.json"
        if receipt.is_file() and json.loads(receipt.read_text()).get("status") == "completed":
            committed.append(json.loads(receipt.read_text()))
            continue
        if round_dir.exists():
            discarded = 0
            while (output / f"round-{index:02d}.discarded-{discarded}").exists():
                discarded += 1
            round_dir.rename(output / f"round-{index:02d}.discarded-{discarded}")
        break
    for later in range(len(committed) + 1, rounds):
        if (output / f"round-{later:02d}" / "round.json").is_file():
            raise LocalVLMError("committed rounds are not a contiguous prefix")
    record = {
        "schema_version": 1,
        "status": "running",
        "algorithm": "RFT-expert-iteration",
        "regularization_profile": "rejection-sampled-self-completion-SFT",
        "created_at": previous["created_at"]
        if previous
        else datetime.now(timezone.utc).isoformat(),
        "control_identity": identity,
        "settings": settings,
        "environment": environment,
        "sources": spec["sources"],
        "claim_scope": RFT_SCOPE,
        "held_out_benchmark_improvement": "unmeasured",
        "rounds": committed,
        "resumed_after_round": len(committed) - 1 if committed else None,
    }
    if threads != DEFAULT_THREADS:
        record["torch_threads"] = threads
    _write(output / "local_training.json", record)
    _archive_source(output, environment["adapter_source_sha256"])
    rows_by_id = {row["id"]: row for row in rows}
    ids = [row["id"] for row in rows]
    clock = _RFTClock(
        sum(entry["sampling_seconds"] + entry["training_seconds"] for entry in committed),
        max_active_seconds,
    )
    total_steps = sum(entry["optimizer_steps"] for entry in committed)
    try:
        torch = import_module("torch")
        # An explicit threads wins; otherwise VLM_TORCH_THREADS (or the historical 2).
        _set_threads(torch, threads if threads != DEFAULT_THREADS else _torch_threads())
        transformers, peft = (import_module(x) for x in ("transformers", "peft"))
        record["model_source"] = _verified_model_directory(model_directory, model)
        processor = _processor(model_directory, profile)
        record["processor_audit"] = _audit_processor(processor, rows, spec, profile)
        policy = transformers.AutoModelForImageTextToText.from_pretrained(
            str(model_directory),
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            dtype=getattr(torch, profile.dtype),
            attn_implementation="sdpa",
        )
        if policy.config.model_type not in {"qwen2_vl", "qwen2_5_vl"}:
            raise LocalVLMError("unexpected cached model architecture")
        policy = (
            peft.PeftModel.from_pretrained(policy, str(init_adapter), is_trainable=True)
            if init_adapter
            else peft.get_peft_model(
                policy,
                peft.LoraConfig(
                    r=profile.lora_rank,
                    lora_alpha=2 * profile.lora_rank,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "v_proj"],
                    task_type="CAUSAL_LM",
                    bias="none",
                ),
            )
        )
        policy.to(device)
        if initial and not committed:
            record["initial_adapter_tensor_audit"] = _verify_initial_adapter_tensors(
                policy, Path(init_adapter)
            )
        for layer in policy.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.0
        if committed:
            _load_round_adapter(policy, output / f"round-{len(committed) - 1:02d}" / "adapter")
        config = _task_group_generation_config(policy, processor, profile)
        record["generation_config"] = config.to_dict()
        _write(output / "local_training.json", record)
        sample = _sample or (
            lambda chosen, round_dir, reward: _sample_groups_batched(
                policy, processor, chosen, spec, profile, device, round_dir, reward, config
            )
        )
        train_round = _train_round or (
            lambda records, round_dir, steps, round_seed: _rft_sft_round(
                policy,
                processor,
                _rft_dataset(records, spec),
                profile,
                round_dir / "trainer",
                device=device,
                steps=steps,
                learning_rate=learning_rate,
                seed=round_seed,
                clock=clock,
            )
        )
        stop_reason = None
        for index in range(len(committed), rounds):
            if stop_after_rounds is not None and index >= stop_after_rounds:
                stop_reason = "stop_after_rounds"
                break
            if stop_after_steps is not None and total_steps >= stop_after_steps:
                stop_reason = "stop_after_steps"
                break
            if max_steps is not None and total_steps >= max_steps:
                stop_reason = "max_steps"
                break
            if clock.exhausted():
                stop_reason = "max_active_seconds"
                break
            round_dir = output / f"round-{index:02d}"
            round_dir.mkdir()
            round_seed = int(
                hashlib.sha256(f"rft-round-v1:{seed}:{index}".encode()).hexdigest()[:8], 16
            )
            transformers.set_seed(round_seed)
            chosen = [
                rows_by_id[identifier]
                for identifier in rft_round_prompt_ids(
                    ids, seed=seed, round_index=index, per_round=prompts_per_round
                )
            ]
            reward = make_reward(
                spec["labels"],
                profile.num_generations,
                config.eos_token_id,
                round_dir / "reward_groups.jsonl",
                allow_reasoning=True,
            )
            clock.begin()
            groups = sample(chosen, round_dir, reward)
            sampling_seconds = clock.end()
            kept, per_prompt = {}, {}
            for row in chosen:
                texts, stats = rft_select(
                    groups[row["id"]],
                    row["label"],
                    spec["labels"],
                    max_keep=max_keep_per_prompt,
                    skip_all_correct=skip_all_correct,
                )
                per_prompt[row["id"]] = stats
                if texts:
                    kept[row["id"]] = texts
            records = rft_records(kept, rows_by_id, spec, profile.response_format)
            _write(
                round_dir / "kept.json",
                {"kept": kept, "per_prompt": per_prompt},
            )
            steps = epochs_per_round * len(records)
            if max_steps is not None:
                steps = min(steps, max_steps - total_steps)
            training = {"optimizer_steps": 0, "planned_steps": steps}
            training_seconds = 0.0
            if steps > 0 and not clock.exhausted():
                clock.begin()
                training = train_round(records, round_dir, steps, round_seed)
                training_seconds = clock.end()
            total_steps += training["optimizer_steps"]
            adapter = round_dir / "adapter"
            policy.save_pretrained(str(adapter))
            entry = {
                "status": "completed",
                "round": index,
                "round_seed": round_seed,
                "prompt_ids": [row["id"] for row in chosen],
                "sampled_completions": sum(s["samples"] for s in per_prompt.values()),
                "correct_completions": sum(s["correct"] for s in per_prompt.values()),
                "mean_pass_rate": sum(s["pass_rate"] for s in per_prompt.values())
                / len(per_prompt),
                "prompts_all_correct": sum(
                    s["correct"] == s["samples"] for s in per_prompt.values()
                ),
                "prompts_none_correct": sum(s["correct"] == 0 for s in per_prompt.values()),
                "prompts_kept": len(kept),
                "kept_completions": len(records),
                "reward_audit": reward.stats,
                **training,
                "cumulative_optimizer_steps": total_steps,
                "sampling_seconds": sampling_seconds,
                "training_seconds": training_seconds,
                "adapter_sha256": _peft_weights_sha256(adapter),
            }
            _write(round_dir / "round.json", entry)
            committed.append(entry)
            record["rounds"] = committed
            _write(output / "local_training.json", record)
        else:
            stop_reason = "rounds"
        if stop_reason == "stop_after_rounds":
            record.update(status="stopped", stop_reason=stop_reason)
            _write(output / "local_training.json", record)
            return record
        adapter = output / "final_adapter"
        policy.save_pretrained(str(adapter))
        processor.save_pretrained(str(adapter))
        if stop_reason == "stop_after_steps":
            record["boundary"] = {"stop_after_steps": stop_after_steps}
        record.update(
            status="completed",
            stop_reason=stop_reason,
            optimizer_steps=total_steps,
            active_seconds=clock.committed,
            final_adapter=str(adapter),
            final_adapter_sha256=_peft_weights_sha256(adapter),
        )
        _write(output / "local_training.json", record)
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _write(output / "local_training.json", record)
        raise
    return record


def make_probe_data(output: Path) -> dict:
    """Deterministic synthetic diagnostic, explicitly excluded from benchmark claims."""
    import random

    pil = import_module("PIL.Image")
    draw_module = import_module("PIL.ImageDraw")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("probe data output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "benchmark": "synthetic-diagnostic-color-majority-v1",
        "diagnostic_only": True,
        "labels": ["blue", "red"],
        "question": "Which color occupies more squares in the colored grid?",
        "split_policy": "8 diagnostic train images, 4 diagnostic validation images; no test",
        "preprocessing": "224x224 RGB grid; 36 versus 28 squares; no external images",
        "sources": [
            {
                "url": "generated://vision_rl.vlm_local_training/make_probe_data",
                "sha256": _file_sha256(Path(__file__)),
            }
        ],
    }
    for split, count, seed in (("train", 8, 1042), ("validation", 4, 2042)):
        rng, rows = random.Random(seed), []
        for index in range(count):
            label = spec["labels"][index % 2]
            other = spec["labels"][(index + 1) % 2]
            colors = [label] * 36 + [other] * 28
            rng.shuffle(colors)
            image = pil.new("RGB", (224, 224), "gray")
            draw = draw_module.Draw(image)
            for cell, color in enumerate(colors):
                x, y = 16 + cell % 8 * 24, 16 + cell // 8 * 24
                draw.rectangle((x, y, x + 23, y + 23), fill=color)
            identifier = f"{split}-{index}"
            image.save(output / f"{identifier}.png")
            rows.append(
                {"id": identifier, "split": split, "image": f"{identifier}.png", "label": label}
            )
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _write(output / "data_spec.json", spec)
    return {"path": str(output.resolve()), "diagnostic_only": True}


def _rft_main(parser, args: dict, rft_args: dict) -> int:
    """CLI for --mode rft: TRAIN pool only; GRPO/SFT-only options are refused."""
    if not all(args[k] for k in ("train", "data_spec", "output", "model_directory")):
        parser.error("RFT requires --train, --data-spec, --output and --model-directory")
    if rft_args["rft_rounds"] is None or rft_args["rft_prompts_per_round"] is None:
        parser.error("RFT requires --rft-rounds and --rft-prompts-per-round")
    refused = [
        key
        for key in (
            "validation",
            "resume_from_checkpoint",
            "screen_journal",
            "screen_tokenizations",
            "continue_from_checkpoint",
            "label_hierarchy",
            "family_credit",
            "manufacturer_credit",
            "invalid_reward",
            "stop_after_steps",
            "stop_after_active_seconds",
            "stage_id",
            "probe",
            "finalize_only",
            "cycle",
            "grpo_overrides",
            "dynamic_sampling_max_draws",
            "loss_scale_init",
        )
        if args.get(key) not in (None, False)
    ]
    if refused or (
        args["order"],
        args["screen_threshold"],
        args["target_accepted"],
        args["beta"],
        args["save_steps"],
    ) != ("random", 0.5, 600, 0.04, 8):
        parser.error(f"RFT refuses GRPO/SFT-only options: {refused or 'order/screen/beta/save'}")
    try:
        result = rft_local(
            args["train"],
            args["data_spec"],
            args["output"],
            model_directory=args["model_directory"],
            rounds=rft_args["rft_rounds"],
            prompts_per_round=rft_args["rft_prompts_per_round"],
            model=args["model"],
            device=args["device"],
            profile=args["profile"],
            init_adapter=args["init_adapter"],
            initial_adapter_precision_transfer=args["initial_adapter_precision_transfer"],
            learning_rate=args["learning_rate"],
            seed=args["seed"],
            epochs_per_round=rft_args["rft_epochs_per_round"] or 1,
            max_keep_per_prompt=rft_args["rft_max_keep_per_prompt"] or 1,
            skip_all_correct=rft_args["rft_skip_all_correct"],
            max_steps=args["max_steps"],
            max_active_seconds=args["max_active_seconds"],
            stop_after_rounds=rft_args["rft_stop_after_rounds"],
            **(
                {"stop_after_steps": rft_args["rft_stop_after_steps"]}
                if rft_args["rft_stop_after_steps"] is not None
                else {}
            ),
            threads=args.pop("threads"),
        )
    except (LocalVLMError, ValueError, OSError) as exc:
        print(f"local VLM: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _cli_bool(value: str) -> bool:
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--make-probe-data", type=Path)
    parser.add_argument(
        "--predict-train",
        type=Path,
        help="Frozen-base inference on at most 32 TRAIN images; never fits",
    )
    parser.add_argument(
        "--sample-train",
        type=Path,
        help="Frozen-base sampled reward groups on at most 32 TRAIN images; never fits",
    )
    parser.add_argument("--mode", choices=("grpo", "sft", "egsft", "rft"), default="grpo")
    for name in (
        "train",
        "validation",
        "data-spec",
        "output",
        "model-directory",
        "init-adapter",
        "resume-from-checkpoint",
        "screen-journal",
        "screen-tokenizations",
        "continue-from-checkpoint",
        "label-hierarchy",
    ):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_REVISIONS), default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=DEVICES, default="mps")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="torch intra-op CPU threads (default 2)",
    )
    parser.add_argument(
        "--loss-scale-init",
        type=float,
        help="float16 only: initial GradScaler loss scale (default 65536)",
    )
    # None means "not given": legacy modes then use 64; RFT treats it as an optional cap.
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="GRPO KL coefficient; 0 explicitly selects the no-KL ablation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=8)
    parser.add_argument("--max-active-seconds", type=float)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--stop-after-active-seconds", type=float)
    parser.add_argument("--stage-id")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--initial-adapter-precision-transfer",
        choices=INFERENCE_PRECISION_TRANSFERS,
        help="training: float32-to-bfloat16|float16; TRAIN inference: <parent>-to-<dtype>",
    )
    parser.add_argument("--order", choices=("random", "sequential"), default="random")
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--screen-threshold", type=float, default=0.5)
    parser.add_argument("--target-accepted", type=int, default=600)
    parser.add_argument("--probe", action="store_true")
    for name, value in GRPO_KNOBS.items():
        if type(value) is bool:
            kind = _cli_bool
        elif name in ("loss_type", "scale_rewards", "importance_sampling_level"):
            kind = str
        else:
            kind = int if type(value) is int else float
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=kind,
            default=None,
            help=f"exploratory GRPO knob; omitted keeps the legacy value {value!r}",
        )
    parser.add_argument(
        "--dynamic-sampling-max-draws",
        type=int,
        help="drop all-same GRPO groups and redraw, up to this many groups per update",
    )
    parser.add_argument(
        "--exploratory-sweep",
        choices=(EXPLORATORY_SWEEP_ID,),
        help="TRAIN-holdout exploratory sweep; lifts the TRAIN-only step/row caps",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Bounded TRAIN learning diagnostic; no held-out data",
    )
    parser.add_argument("--family-credit", type=float, help="GRPO hierarchical reward")
    parser.add_argument("--manufacturer-credit", type=float, help="GRPO hierarchical reward")
    parser.add_argument("--invalid-reward", type=float, help="GRPO hierarchical reward")
    for name in ("rounds", "prompts-per-round", "epochs-per-round", "max-keep-per-prompt"):
        parser.add_argument(f"--rft-{name}", type=int)
    parser.add_argument("--rft-stop-after-rounds", type=int)
    parser.add_argument("--rft-stop-after-steps", type=int)
    parser.add_argument("--rft-skip-all-correct", action="store_true")
    for name, value in asdict(Profile()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    rft_args = {
        key: args.pop(key)
        for key in (
            "rft_rounds",
            "rft_prompts_per_round",
            "rft_epochs_per_round",
            "rft_max_keep_per_prompt",
            "rft_stop_after_rounds",
            "rft_stop_after_steps",
            "rft_skip_all_correct",
        )
    }
    knobs = {key: args.pop(key) for key in GRPO_KNOBS}
    knobs = {key: value for key, value in knobs.items() if value is not None}
    if knobs:
        args["grpo_overrides"] = knobs
    if args["dynamic_sampling_max_draws"] is None:
        args.pop("dynamic_sampling_max_draws")
    try:
        if args.pop("preflight"):
            result = preflight(args["device"])
        elif path := args.pop("make_probe_data"):
            result = make_probe_data(path)
        else:
            predict_train = args.pop("predict_train")
            sample_train = args.pop("sample_train")
            if predict_train is not None and sample_train is not None:
                parser.error("choose --predict-train or --sample-train")
            args["profile"] = Profile(**{key: args.pop(key) for key in asdict(Profile())})
            if args["mode"] == "rft":
                if predict_train is not None or sample_train is not None:
                    parser.error("RFT cannot be combined with TRAIN inference")
                return _rft_main(parser, args, rft_args)
            if any(value not in (None, False) for value in rft_args.values()):
                parser.error("--rft-* options require --mode rft")
            if args["max_steps"] is None:
                args["max_steps"] = 64
            if predict_train is not None or sample_train is not None:
                if not all(args[k] for k in ("data_spec", "output", "model_directory")):
                    parser.error(
                        "TRAIN inference requires --data-spec, --output, --model-directory"
                    )
                if any(
                    args[k]
                    for k in (
                        "train",
                        "validation",
                        "resume_from_checkpoint",
                        "probe",
                        "train_only",
                        "exploratory_sweep",
                        "stop_after_steps",
                        "stop_after_active_seconds",
                        "stage_id",
                        "finalize_only",
                        "cycle",
                        "screen_journal",
                        "screen_tokenizations",
                        "continue_from_checkpoint",
                        "label_hierarchy",
                        "family_credit",
                        "manufacturer_credit",
                        "invalid_reward",
                        "grpo_overrides",
                        "dynamic_sampling_max_draws",
                    )
                    if k in args
                ) or (
                    args["order"],
                    args["screen_threshold"],
                    args["target_accepted"],
                    args["loss_scale_init"],
                ) != (
                    "random",
                    0.5,
                    600,
                    None,
                ):
                    parser.error("TRAIN inference cannot be combined with training inputs")
                function = sample_train_groups if sample_train is not None else predict_train_subset
                result = function(
                    sample_train if sample_train is not None else predict_train,
                    **({"seed": args["seed"]} if sample_train is not None else {}),
                    **{
                        key: args[key]
                        for key in (
                            "data_spec",
                            "output",
                            "model_directory",
                            "model",
                            "device",
                            "profile",
                            "init_adapter",
                            "threads",
                        )
                    },
                    **(
                        {"precision_transfer": args["initial_adapter_precision_transfer"]}
                        if args["initial_adapter_precision_transfer"]
                        else {}
                    ),
                )
                print(json.dumps(result, indent=2, allow_nan=False))
                return 0
            if not all(args[k] for k in ("train", "data_spec", "output")) or (
                not args["train_only"] and not args["validation"]
            ):
                parser.error(
                    "--train, --data-spec, --output and --validation or --train-only required"
                )
            result = train_local(**args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (LocalVLMError, ValueError, OSError) as exc:
        print(f"local VLM: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
