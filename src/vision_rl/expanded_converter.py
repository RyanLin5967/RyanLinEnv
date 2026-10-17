"""Unactivated, numeric-only language LoRA expansion; never loads a base model.

Preserve every existing Q/V tensor byte and alpha/r. Add seeded A and zero B
for language K/O/gate/up/down. Output is a conversion artifact, not a completed
training run. Runtime training/grading support is deliberately not installed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from importlib import metadata
from pathlib import Path
import tempfile

import numpy as np
from safetensors.numpy import load, save

from vision_rl.vlm_local_task_builder import _bounded_bytes

FORMAT = "policy-preserving-language-lora-expansion-v1"
QWEN_PARENT_WEIGHTS = "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"
QWEN_PARENT_CONFIG = "30fc0ca5a8b5fa497c5b2be675d6efe7dbf257886ed843efe67ed94a6186386c"
MAX_ADAPTER_BYTES = 64 * 1024**2
QV = ("self_attn.q_proj", "self_attn.v_proj")
PROJECTIONS = (*QV, "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


@dataclass(frozen=True)
class Layout:
    name: str
    model: str
    revision: str
    layers: int
    hidden: int
    intermediate: int
    kv_width: int
    layer_prefix: str
    rank: int = 8
    alpha: int = 16


QWEN = Layout(
    "qwen2vl2b-language-seven-r8-v1", "Qwen/Qwen2-VL-2B-Instruct",
    "895c3a49bc3fa70a340399125c650a463535e71c", 28, 1536, 8960, 256,
    "model.language_model.layers",
)
TINY = Layout(
    "synthetic-llama-language-seven-r8-v1", "random-tiny-Llama", "seed-42",
    2, 16, 32, 8, "model.layers",
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_digest(path):
    return digest(Path(path).read_bytes())


def targets(layout, projections=PROJECTIONS):
    if layout not in (QWEN, TINY):
        raise ValueError("Only the explicit prospective Qwen or tiny-fixture layout is supported")
    return sorted(f"{layout.layer_prefix}.{layer}.{projection}"
                  for layer in range(layout.layers) for projection in projections)


def shapes(layout, projections=PROJECTIONS):
    widths = {
        "self_attn.q_proj": (layout.hidden, layout.hidden),
        "self_attn.k_proj": (layout.kv_width, layout.hidden),
        "self_attn.v_proj": (layout.kv_width, layout.hidden),
        "self_attn.o_proj": (layout.hidden, layout.hidden),
        "mlp.gate_proj": (layout.intermediate, layout.hidden),
        "mlp.up_proj": (layout.intermediate, layout.hidden),
        "mlp.down_proj": (layout.hidden, layout.intermediate),
    }
    result = {}
    for target in targets(layout, projections):
        out_width, in_width = widths[".".join(target.split(".")[-2:])]
        prefix = "base_model.model." + target
        result[prefix + ".lora_A.weight"] = (layout.rank, in_width)
        result[prefix + ".lora_B.weight"] = (out_width, layout.rank)
    return result


def validate_config(config, layout):
    """Reject every nonstandard transform; never forward caller config to PEFT."""
    exact = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": layout.rank,
             "lora_alpha": layout.alpha, "lora_dropout": 0.0, "bias": "none"}
    if any(type(config.get(k)) is bool or config.get(k) != v for k, v in exact.items()):
        raise ValueError("Source LoRA rank/scale/dropout/bias/type differs")
    chosen = config.get("target_modules")
    if (not isinstance(chosen, list) or len(set(chosen)) != len(chosen)
            or set(chosen) not in ({"q_proj", "v_proj"}, set(targets(layout, QV)))):
        raise ValueError("Source must contain exactly the trusted language Q/V targets")
    metadata = {"base_model_name_or_path", "revision", "peft_version", "inference_mode",
                "init_lora_weights", "megatron_core", "qalora_group_size"}
    inactive = {
        "alora_invocation_tokens", "alpha_pattern", "arrow_config", "auto_mapping",
        "corda_config", "ensure_weight_tying", "eva_config", "exclude_modules",
        "fan_in_fan_out", "kasa_config", "layer_replication", "layers_pattern",
        "layers_to_transform", "loftq_config", "lora_bias", "lora_ga_config",
        "megatron_config", "modules_to_save", "monteclora_config", "rank_pattern",
        "target_parameters", "trainable_token_indices", "use_bdlora", "use_dora",
        "use_qalora", "use_rslora", "velora_config",
    }
    if set(config) - set(exact) - {"target_modules"} - metadata - inactive:
        raise ValueError("Unknown adapter configuration field")
    if any(config.get(key) not in (None, False, {}, []) for key in inactive):
        raise ValueError("Source enables a transform outside plain LoRA")
    if type(config.get("init_lora_weights", True)) is not bool:
        raise ValueError("Special source initialization is unsupported")


def validate_tensors(values, expected):
    if set(values) != set(expected):
        raise ValueError("Exact adapter tensor names differ; no vision/head/extra tensors allowed")
    for name, value in values.items():
        if value.dtype != np.float32 or value.shape != expected[name] or not np.isfinite(value).all():
            raise ValueError("Require exact shapes and finite FP32 tensors")


def tensor_records(values):
    return {name: {"shape": list(value.shape), "dtype": "float32",
                   "data_sha256": digest(value.tobytes(order="C"))}
            for name, value in sorted(values.items())}


def trusted_config(layout, projections=PROJECTIONS):
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": layout.rank,
            "lora_alpha": layout.alpha, "lora_dropout": 0.0, "bias": "none",
            "target_modules": targets(layout, projections), "init_lora_weights": False,
            "inference_mode": True, "base_model_name_or_path": layout.model,
            "revision": layout.revision, "fan_in_fan_out": False,
            "use_rslora": False, "use_dora": False, "lora_bias": False}


def expand_adapter(source, output, *, layout, seed, expected_weights_sha256,
                   expected_config_sha256, provenance):
    """Read bounded numeric tensors, publish a new immutable conversion artifact."""
    source, output = Path(source), Path(output)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Expansion seed must be a uint32")
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be new; never overwrite an adapter")
    raw = _bounded_bytes(source / "adapter_model.safetensors", MAX_ADAPTER_BYTES)
    config_raw = _bounded_bytes(source / "adapter_config.json", 1024**2)
    if digest(raw) != expected_weights_sha256 or digest(config_raw) != expected_config_sha256:
        raise ValueError("Source weights/config differ from the explicit immutable pins")
    config = json.loads(config_raw)
    validate_config(config, layout)
    current = load(raw)
    validate_tensors(current, shapes(layout, QV))
    expanded = {name: value.copy() for name, value in current.items()}
    new_seeds = {}
    for target in targets(layout):
        prefix = "base_model.model." + target
        a, b = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
        if a in current:
            continue
        module_seed = int.from_bytes(hashlib.sha256(
            f"{FORMAT}:{seed}:{target}".encode()).digest()[:8], "little")
        new_seeds[target] = module_seed
        dimensions = shapes(layout)
        bound = 1 / math.sqrt(dimensions[a][1])
        rng = np.random.Generator(np.random.PCG64(module_seed))
        expanded[a] = rng.uniform(-bound, bound, dimensions[a]).astype(np.float32)
        expanded[b] = np.zeros(dimensions[b], dtype=np.float32)
        if not np.count_nonzero(expanded[a]):
            raise ValueError("New A must be nonzero; no silent dead adapter")
    validate_tensors(expanded, shapes(layout))
    preserved = tensor_records(current)
    if any(tensor_records({key: expanded[key]})[key] != value for key, value in preserved.items()):
        raise ValueError("A preserved Q/V tensor changed")
    new_names = sorted(set(expanded) - set(current))
    if any(expanded[name].tobytes() != bytes(expanded[name].nbytes)
           for name in new_names if ".lora_B." in name):
        raise ValueError("Every newly added B must be exact positive zero")
    result_raw = save(expanded, metadata={"format": "pt"})
    if len(result_raw) > MAX_ADAPTER_BYTES:
        raise ValueError("Expanded artifact exceeds the prospective 64 MiB numeric bound")
    output_config = encoded(trusted_config(layout))
    receipt = {
        "schema_version": 1, "artifact_kind": FORMAT, "activated": False,
        "layout": asdict(layout), "source_sha256": file_digest(__file__),
        "runtime": {"numpy": np.__version__, "safetensors": metadata.version("safetensors")},
        "source": {"weights_sha256": digest(raw), "config_sha256": digest(config_raw),
                   "provenance": provenance},
        "expansion_seed": seed, "new_module_seeds": new_seeds,
        "initialization": "Independent per-module PCG64 uniform(-1/sqrt(fan_in),+1/sqrt(fan_in)) FP32 A; exact zero FP32 B",
        "preserved_tensor_count": len(current), "added_tensor_count": len(new_names),
        "preserved_tensors": preserved, "all_tensors": tensor_records(expanded),
        "parameter_count": sum(value.size for value in expanded.values()),
        "raw_tensor_bytes": sum(value.nbytes for value in expanded.values()),
        "output": {"weights_sha256": digest(result_raw), "config_sha256": digest(output_config)},
        "scaling": {"rank": layout.rank, "alpha": layout.alpha, "alpha_over_rank": layout.alpha / layout.rank},
        "reference_contract": "Released TRL must copy this exact expanded adapter to frozen ref; never bare-base disable_adapter",
        "optimizer_state_transferred": False, "training_performed": False,
        "mathematical_policy_identity": True,
        "pretrained_numeric_logit_equivalence": "unmeasured",
        "application_gain_claim": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expanded-incomplete-", dir=output.parent) as temporary:
        staging = Path(temporary) / "adapter"
        staging.mkdir()
        (staging / "adapter_model.safetensors").write_bytes(result_raw)
        (staging / "adapter_config.json").write_bytes(output_config)
        (staging / "expansion_receipt.json").write_bytes(encoded(receipt))
        (staging / "complete.json").write_bytes(encoded({
            "schema_version": 1, "kind": FORMAT,
            "file_sha256": {path.name: file_digest(path) for path in sorted(staging.iterdir())},
        }))
        staging.rename(output)
    verify_expansion(output, source, layout=layout)
    return receipt


def verify_expansion(output, source, *, layout):
    output, source = Path(output), Path(source)
    marker = json.loads(_bounded_bytes(output / "complete.json", 1024**2))
    expected_files = {"adapter_model.safetensors", "adapter_config.json", "expansion_receipt.json"}
    if (marker.get("kind") != FORMAT or set(marker.get("file_sha256", {})) != expected_files
            or any(digest(_bounded_bytes(output / name, MAX_ADAPTER_BYTES)) != expected
                   for name, expected in marker["file_sha256"].items())):
        raise ValueError("Expanded artifact completion marker does not match its files")
    receipt = json.loads(_bounded_bytes(output / "expansion_receipt.json", 1024**2))
    values = load(_bounded_bytes(output / "adapter_model.safetensors", MAX_ADAPTER_BYTES))
    original_raw = _bounded_bytes(source / "adapter_model.safetensors", MAX_ADAPTER_BYTES)
    original = load(original_raw)
    validate_tensors(original, shapes(layout, QV))
    validate_tensors(values, shapes(layout))
    if (receipt["artifact_kind"] != FORMAT or receipt["layout"] != asdict(layout)
            or receipt["schema_version"] != 1 or receipt["activated"] is not False
            or receipt["source_sha256"] != file_digest(__file__)
            or receipt["runtime"] != {"numpy": np.__version__, "safetensors": metadata.version("safetensors")}
            or receipt["source"]["weights_sha256"] != digest(original_raw)
            or receipt["source"]["config_sha256"] != digest(_bounded_bytes(source / "adapter_config.json", 1024**2))
            or receipt["all_tensors"] != tensor_records(values)
            or receipt["preserved_tensors"] != tensor_records(original)
            or receipt["preserved_tensor_count"] != len(original)
            or receipt["added_tensor_count"] != len(values) - len(original)
            or receipt["parameter_count"] != sum(value.size for value in values.values())
            or receipt["raw_tensor_bytes"] != sum(value.nbytes for value in values.values())
            or receipt["output"] != {"weights_sha256": marker["file_sha256"]["adapter_model.safetensors"],
                                      "config_sha256": marker["file_sha256"]["adapter_config.json"]}
            or json.loads((output / "adapter_config.json").read_text()) != trusted_config(layout)):
        raise ValueError("Expanded layout/source/config/tensor receipt differs")
    for name, value in original.items():
        if value.tobytes() != values[name].tobytes():
            raise ValueError("Original Q/V tensor bytes changed")
    for name in set(values) - set(original):
        if ".lora_B." in name and values[name].tobytes() != bytes(values[name].nbytes):
            raise ValueError("New B changed from exact zero")
        if ".lora_A." in name:
            target = name.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            seed = int.from_bytes(hashlib.sha256(
                f"{FORMAT}:{receipt['expansion_seed']}:{target}".encode()).digest()[:8], "little")
            bound = 1 / math.sqrt(values[name].shape[1])
            expected = np.random.Generator(np.random.PCG64(seed)).uniform(
                -bound, bound, values[name].shape).astype(np.float32)
            if (receipt["new_module_seeds"].get(target) != seed
                    or expected.tobytes() != values[name].tobytes()):
                raise ValueError("New A differs from deterministic per-module initialization")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    parent_path = args.source_adapter.parent / "local_training.json"
    parent_raw = _bounded_bytes(parent_path, 1024**2)
    parent = json.loads(parent_raw)
    if (parent.get("status") != "completed" or parent.get("final_adapter_sha256") != QWEN_PARENT_WEIGHTS
            or parent["control_identity"]["model"] != QWEN.model
            or parent["control_identity"]["model_revision"] != QWEN.revision
            or parent["control_identity"]["profile"]["dtype"] != "float32"):
        raise ValueError("Only the pinned completed common FP32 warm export is supported")
    receipt = expand_adapter(
        args.source_adapter, args.output, layout=QWEN, seed=args.seed,
        expected_weights_sha256=QWEN_PARENT_WEIGHTS, expected_config_sha256=QWEN_PARENT_CONFIG,
        provenance={"kind": "completed_training_export", "record_sha256": digest(parent_raw),
                    "control_identity": parent["control_identity"], "settings": parent["settings"]},
    )
    print(json.dumps({"output": str(args.output), "activated": False,
                      "parameter_count": receipt["parameter_count"], **receipt["output"]}, indent=2))


if __name__ == "__main__":
    main()
