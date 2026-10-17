"""Verifier-owned language-seven LoRA contracts; no model import at module load.

INIT is an externally pinned zero-update conversion. ENDPOINT is an actual
native training boundary; a verifier additionally pins its receipt, either
explicitly or through the source-owned registry below. Legacy Q/V artifacts
continue through the unchanged native trainer and 112-tensor grader contracts.

Trust root: ``APPROVAL_SHA256`` (source) pins ``expanded_assets/approval.json``,
which pins the warm parent record, the conversion plan, the warm adapter bytes,
the converter source and the converted INIT files. The warm
``adapter_model.safetensors`` is not committed (``*.safetensors`` is ignored);
without it every expanded path refuses.

Blind spots, stated here on purpose: the endpoint binds the exact ``vision_rl``
package source and the native receipt, but not the installed third-party stack
beyond what native ``preflight`` records, and not the operator's data files
beyond the identities the native record carries. Because the binding is exact,
an endpoint verifies only against the ``vision_rl`` source that produced it: a
verifier checks out that commit (any later edit to any package module refuses).
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import asdict
from pathlib import Path

PROFILE_ID = "qwen2vl2b-language-seven-r8-v1"
LEGACY_PROFILE = "legacy-qv-v1"
APPROVAL_SHA256 = "765ab91136b40ce02512cf0c9eb9499810690b466bc42c648e5a9f2eda9cdbc3"
ASSETS = Path(__file__).with_name("expanded_assets")
# Name of the converter layout the contract enforces. Only the converter's explicit
# layouts exist: QWEN for production, TINY for the random CPU test fixture.
LAYOUT = "QWEN"
ENDPOINT_KIND = "native-expanded-trained-endpoint-v2"
# Verifier-owned registry of accepted expanded_endpoint.json SHA-256 pins, kept in
# ASSETS (a reviewed data file, not a .py, so adding a pin leaves the source closure
# every endpoint binds unchanged). Empty means none is accepted where no explicit
# verifier pin is supplied (TRAIN assay, numeric attach).
ENDPOINT_PINS_FILE = "endpoint-pins.json"
MAX_BYTES = 64 * 1024**2
JSON_BYTES = 1024**2
RECORD_BYTES = 64 * 1024**2
SOURCE_BYTES = 4 * 1024**2
MATCHED_ARMS = ("exposure", "time")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bounded(path, maximum=JSON_BYTES):
    from .vlm_local_task_builder import _bounded_bytes

    return _bounded_bytes(Path(path), maximum)


def read_json(path, maximum=JSON_BYTES):
    return json.loads(bounded(path, maximum))


def converter():
    from . import expanded_converter

    return expanded_converter


def active_layout():
    if LAYOUT not in ("QWEN", "TINY"):
        raise ValueError("unknown expanded adapter layout")
    return getattr(converter(), LAYOUT)


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits.lower() for character in value)
    )


def is_expanded(profile):
    return getattr(profile, "adapter_profile", LEGACY_PROFILE) == PROFILE_ID


def portable_initial(identity):
    """INIT identity without its machine-local path; every byte-binding field stays."""
    return {key: value for key, value in identity.items() if key != "path"}


def source_closure():
    """SHA-256 of every module of the installed vision_rl package, including this one."""
    package = Path(__file__).resolve().parent
    files = {
        path.name: sha(bounded(path, SOURCE_BYTES)) for path in sorted(package.glob("*.py"))
    }
    if not {"expanded_contract.py", "expanded_converter.py", "expanded_grader.py",
            "vlm_local_training.py", "vlm_local_task_builder.py"} <= set(files):
        raise ValueError("expanded source closure is incomplete")
    return {"package": "vision_rl", "files": files,
            "sha256": sha(json.dumps(files, sort_keys=True).encode())}


def approved_lineage():
    """The approval hash is owned by source, never supplied by an artifact."""
    raw = bounded(ASSETS / "approval.json")
    if sha(raw) != APPROVAL_SHA256:
        raise ValueError("verifier-owned expansion approval changed")
    approval = json.loads(raw)
    for name, expected in approval["trusted_files"].items():
        if sha(bounded(ASSETS / name, MAX_BYTES)) != expected:
            raise ValueError("approved warm/conversion lineage changed")
    converter_path = Path(converter().__file__).resolve()
    if sha(bounded(converter_path, JSON_BYTES)) != approval["converter_sha256"]:
        raise ValueError("approved converter source changed")
    plan = read_json(ASSETS / "conversion-plan.json")
    parent = read_json(ASSETS / "warm-local-training.json")
    if (
        approval["profile_id"] != PROFILE_ID
        or approval["expansion_seed"] != 42
        or plan["target_profile"] != PROFILE_ID
        or plan["expansion_seed"] != 42
        or plan["converter_sha256"] != approval["converter_sha256"]
        or plan["parent_weights_sha256"] != parent["final_adapter_sha256"]
        or parent["status"] != "completed"
        or parent["settings"]["mode"] != "sft"
    ):
        raise ValueError("approved conversion does not match actual warm SFT lineage")
    return approval, parent


def validate_profile(profile):
    if getattr(profile, "adapter_profile", LEGACY_PROFILE) not in (LEGACY_PROFILE, PROFILE_ID):
        raise ValueError("unknown trusted adapter profile")
    if hasattr(profile, "adapter_profile") and profile.adapter_profile != PROFILE_ID:
        raise ValueError("the expanded profile class carries only the language-seven profile")
    if is_expanded(profile) and (profile.lora_rank != 8 or profile.dtype != "float32"):
        raise ValueError("language-seven profile requires rank8 FP32")


def read_numeric(path, *, layout=None):
    """Only bounded numeric arrays; configs, code and pickle never enter PEFT."""
    from safetensors.numpy import load

    module = converter()
    chosen = active_layout() if layout is None else layout
    raw = bounded(path, MAX_BYTES)
    values = load(raw)
    module.validate_tensors(values, module.shapes(chosen))
    return values, sha(raw)


def validate_saved_config(path, *, layout=None):
    """Validate native serialization metadata, but never instantiate it."""
    module = converter()
    chosen = active_layout() if layout is None else layout
    config = read_json(path)
    targets = config.get("target_modules")
    if not isinstance(targets, list) or len(targets) != len(set(targets)) or set(targets) != set(
        module.targets(chosen)
    ):
        raise ValueError("expanded configuration must name exactly the language projections")
    module.validate_config({**config, "target_modules": ["q_proj", "v_proj"]}, chosen)
    return config


def tensor_fingerprint(values):
    """Same normalized per-tensor fingerprint as the native frozen-ref audit."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        normalized = name.replace(".lora_A.weight", ".lora_A.adapter.weight").replace(
            ".lora_B.weight", ".lora_B.adapter.weight"
        )
        header = {"name": normalized, "shape": list(value.shape), "dtype": "torch.float32"}
        digest.update(json.dumps(header, sort_keys=True).encode() + b"\0" + value.tobytes())
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(values),
        "bytes": sum(value.nbytes for value in values.values()),
        "trainable_tensors": 0,
    }


def validate_initial(path):
    """Verify the exact approved112→392 conversion, including external lineage."""
    path = Path(path)
    approval, parent = approved_lineage()
    for name, expected in approval["initial_files"].items():
        if sha(bounded(path / name, MAX_BYTES)) != expected:
            raise ValueError("initialization differs from externally approved conversion bytes")
    module = converter()
    receipt = module.verify_expansion(path, ASSETS / "warm", layout=active_layout())
    provenance = receipt["source"]["provenance"]
    if (
        receipt["expansion_seed"] != approval["expansion_seed"]
        or provenance != {
            "kind": "completed_training_export",
            "record_sha256": approval["trusted_files"]["warm-local-training.json"],
            "control_identity": parent["control_identity"],
            "settings": parent["settings"],
        }
        or receipt["training_performed"] is not False
        or receipt["optimizer_state_transferred"] is not False
    ):
        raise ValueError("conversion provenance differs from approved warm source")
    values, weights = read_numeric(path / "adapter_model.safetensors")
    if weights != approval["initial_files"]["adapter_model.safetensors"]:
        raise ValueError("initialization changed while it was being verified")
    return {
        "kind": "approved_policy_preserving_conversion",
        "path": str(path.resolve()),
        "adapter_profile": PROFILE_ID,
        "weights_sha256": weights,
        "config_sha256": approval["initial_files"]["adapter_config.json"],
        "conversion_receipt_sha256": approval["initial_files"]["expansion_receipt.json"],
        "conversion_complete_sha256": approval["initial_files"]["complete.json"],
        "approval_sha256": APPROVAL_SHA256,
        "conversion_plan_sha256": approval["trusted_files"]["conversion-plan.json"],
        "parent_record_sha256": approval["trusted_files"]["warm-local-training.json"],
        "expansion_seed": approval["expansion_seed"],
        "parent_profile": parent["control_identity"]["profile"],
        "parent_data_identity": {
            k: parent["control_identity"][k] for k in ("data_spec_sha256", "train", "validation")
        },
        "initial_reference": tensor_fingerprint(values),
        "optimizer_state_transferred": False,
        "conversion_optimizer_steps": 0,
    }


def check_task(identity, profile):
    validate_profile(profile)
    if not is_expanded(profile):
        raise ValueError("expanded contract requires the language-seven profile")
    _, parent = approved_lineage()
    source = parent["control_identity"]
    for key in (
        "model", "model_revision", "processor_model", "processor_revision", "data_spec_sha256"
    ):
        if identity.get(key) != source[key]:
            raise ValueError("expanded artifact task/model/processor differs from warm source")
    for key in ("dtype", "min_pixels", "max_pixels", "lora_rank"):
        if asdict(profile)[key] != source["profile"][key]:
            raise ValueError("expanded profile changes approved pixel or precision contract")


def endpoint_receipt_pins():
    """The registry file; a data asset, so listing a pin never changes the source closure."""
    registry = read_json(ASSETS / ENDPOINT_PINS_FILE)
    if not isinstance(registry, dict):
        raise ValueError("verifier-owned endpoint pin registry is malformed")
    pins = registry.get("pins")
    if (registry.get("schema_version") != 1 or not isinstance(pins, list)
            or not all(is_sha256(pin) for pin in pins) or len(set(pins)) != len(pins)):
        raise ValueError("verifier-owned endpoint pin registry is malformed")
    return frozenset(pins)


def registered_pin(output):
    """Receipt pin from the verifier-owned registry; an unlisted endpoint is refused."""
    digest = sha(bounded(Path(output) / "expanded_endpoint.json"))
    if digest not in endpoint_receipt_pins():
        raise ValueError("expanded endpoint receipt is not in the verifier-owned pin registry")
    return digest


def endpoint_identity(path, *, expected_receipt_sha256=None, initial_path=None):
    """Bind a loadable endpoint to the exact final_adapter bytes its receipt names."""
    path = Path(path)
    if path.name != "final_adapter" or path.is_symlink() or not path.is_dir():
        raise ValueError("an expanded endpoint loads only from its own final_adapter directory")
    output = path.parent
    pin = expected_receipt_sha256 if expected_receipt_sha256 is not None else registered_pin(output)
    endpoint = validate_endpoint(output, expected_receipt_sha256=pin, initial_path=initial_path)
    weights = sha(bounded(path / "adapter_model.safetensors", MAX_BYTES))
    config = sha(bounded(path / "adapter_config.json"))
    if weights != endpoint["weights_sha256"] or config != endpoint["config_sha256"]:
        raise ValueError("final_adapter bytes differ from the pinned endpoint identity")
    return {
        "kind": "expanded_trained_endpoint",
        "path": str(path.resolve()),
        "adapter_profile": PROFILE_ID,
        "weights_sha256": weights,
        "config_sha256": config,
        "training_receipt_sha256": endpoint["native_receipt_sha256"],
        "expanded_receipt_sha256": pin,
        "approval_sha256": APPROVAL_SHA256,
        "parent_profile": endpoint["control_identity"]["profile"],
        "parent_data_identity": {
            k: endpoint["control_identity"][k] for k in ("data_spec_sha256", "train", "validation")
        },
    }


def artifact_identity(path, identity, profile, *, inference=False, initial_path=None):
    check_task(identity, profile)
    if (Path(path) / "expansion_receipt.json").exists():
        return validate_initial(path)
    if not inference:
        raise ValueError("fresh expanded training requires the approved converted initialization")
    return endpoint_identity(path, initial_path=initial_path)


def _attach_values(base, values, *, layout, is_trainable):
    """Construct trusted PEFT layers, then load exactly the validated arrays."""
    import peft
    import torch

    module = converter()
    module.validate_tensors(values, module.shapes(layout))
    config = module.trusted_config(layout)
    config["inference_mode"] = not is_trainable
    policy = peft.get_peft_model(base, peft.LoraConfig(**config))
    expected = peft.get_peft_model_state_dict(policy)
    if set(expected) != set(values) or any(
        tuple(expected[k].shape) != v.shape for k, v in values.items()
    ):
        raise ValueError("actual model does not match exact language adapter layout")
    numeric = {name: torch.from_numpy(value.copy()) for name, value in values.items()}
    peft.set_peft_model_state_dict(policy, numeric)
    loaded = peft.get_peft_model_state_dict(policy)
    if any(value.dtype != torch.float32 or not torch.equal(value.detach().cpu(), numeric[name])
           for name, value in loaded.items()):
        raise ValueError("numeric adapter load changed FP32 tensor bytes")
    if not is_trainable:
        policy.requires_grad_(False)
    assert_runtime(policy, layout=layout, is_trainable=is_trainable)
    return policy


def attach_numeric(base, path, *, expected_weights_sha256, is_trainable, initial_path=None):
    """Re-validate after base allocation and attach only the bytes that were validated.

    The SHA of the bytes actually deserialized must equal both the caller's earlier
    identity and this second validation, so a swap between checks is refused.
    """
    path = Path(path)
    if not is_sha256(expected_weights_sha256):
        raise ValueError("attach requires the weights SHA-256 from the validated identity")
    if (path / "expansion_receipt.json").exists():
        identity = validate_initial(path)
    else:
        identity = endpoint_identity(path, initial_path=initial_path)
    values, weights = read_numeric(path / "adapter_model.safetensors")
    if not weights == identity["weights_sha256"] == expected_weights_sha256:
        raise ValueError("adapter bytes changed between validation and numeric attach")
    return _attach_values(base, values, layout=active_layout(), is_trainable=is_trainable)


def assert_runtime(policy, *, layout=None, is_trainable=True):
    module = converter()
    chosen = active_layout() if layout is None else layout
    expected = set(module.targets(chosen))
    layers = {name.removeprefix("base_model.model."): layer
              for name, layer in policy.named_modules() if hasattr(layer, "lora_A")}
    if set(layers) != expected or any(layer.scaling["default"] != 2 for layer in layers.values()):
        raise ValueError("runtime target paths/scaling differ; vision adapters are forbidden")
    wanted = {name.replace(".weight", ".default.weight") for name in module.shapes(chosen)}
    trainable = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    if set(trainable) != (wanted if is_trainable else set()):
        raise ValueError("only exact default language LoRA tensors may be trainable")
    return {"tensor_count": len(wanted), "parameters": sum(p.numel() for p in trainable.values()),
            "vision_and_base_frozen": True}


def validate_checkpoint(checkpoint, initial, mode, *, layout=None):
    """Validate adapter numeric/config state before native pickle/RNG restoration."""
    checkpoint = Path(checkpoint)
    read_numeric(checkpoint / "adapter_model.safetensors", layout=layout)
    validate_saved_config(checkpoint / "adapter_config.json", layout=layout)
    if mode == "grpo":
        reference, _ = read_numeric(checkpoint / "ref/adapter_model.safetensors", layout=layout)
        validate_saved_config(checkpoint / "ref/adapter_config.json", layout=layout)
        if tensor_fingerprint(reference) != initial["initial_reference"]:
            raise ValueError("expanded checkpoint rebased the approved immutable reference")


def endpoint_payload(output, *, initial_path=None):
    """Author-owned native trajectory only; zero-update conversion is not a fit.

    ``initial_path`` is where this verifier holds the approved INIT; it defaults to
    the path the trainer recorded. Only hashes are compared, never the path.
    """
    from . import vlm_local_training as native

    output = Path(output)
    receipt = native.read_endpoint_receipt(output)
    record = read_json(output / "local_training.json", RECORD_BYTES)
    fields = record["control_identity"]["profile"]
    if fields.get("adapter_profile") != PROFILE_ID or receipt["global_step"] < 1:
        raise ValueError("expanded endpoint requires a real positive native optimizer boundary")
    profile = native.ExpandedProfile(**fields)
    profile.validate()
    check_task(record["control_identity"], profile)
    mode = record["settings"]["mode"]
    if mode not in ("grpo", "sft"):
        raise ValueError("expanded endpoints are GRPO or SFT only")
    recorded = record["settings"]["init_adapter"]
    initial = validate_initial(Path(recorded["path"] if initial_path is None else initial_path))
    if portable_initial(recorded) != portable_initial(initial):
        raise ValueError("endpoint initialization differs from approved expanded lineage")
    native_source = sha(bounded(Path(native.__file__).resolve(), SOURCE_BYTES))
    if record["environment"]["adapter_source_sha256"] != native_source:
        raise ValueError("endpoint was not produced by this reviewed native implementation")
    closure = source_closure()
    if record["environment"].get("expanded_source_closure") != closure:
        raise ValueError("endpoint was not produced by this exact vision_rl source closure")
    checkpoint = output / receipt["checkpoint"]["path"]
    validate_checkpoint(checkpoint, initial, mode)
    values, weights = read_numeric(output / "final_adapter/adapter_model.safetensors")
    validate_saved_config(output / "final_adapter/adapter_config.json")
    committed, _ = read_numeric(checkpoint / "adapter_model.safetensors")
    if weights != receipt["final_adapter"]["weights_sha256"] or any(
        value.tobytes() != committed[name].tobytes() for name, value in values.items()
    ):
        raise ValueError("final expanded adapter differs from committed optimizer state")
    if mode == "grpo":
        reference = record["reference"]
        if (reference is None
                or reference["adapter"] != initial["initial_reference"]
                or reference["initial_adapter_weights_sha256"] != initial["weights_sha256"]
                or record["optimizer_audit"].get("reference_unchanged") is not True):
            raise ValueError("expanded training did not preserve the approved frozen reference")
    return {
        "schema_version": 2, "kind": ENDPOINT_KIND,
        "adapter_profile": PROFILE_ID, "approval_sha256": APPROVAL_SHA256,
        "weights_sha256": weights,
        "config_sha256": sha(bounded(output / "final_adapter/adapter_config.json")),
        "native_receipt_sha256": sha(bounded(output / "endpoint_receipt.json")),
        "native_record_sha256": sha(bounded(output / "local_training.json", RECORD_BYTES)),
        "native_source_sha256": native_source,
        "source_closure": closure,
        "initial": portable_initial(initial),
        "control_identity": record["control_identity"], "settings": record["settings"],
        "status": receipt["status"], "original_budgets": receipt["original_budgets"],
        "global_step": receipt["global_step"],
        "committed_active_seconds": receipt["committed_active_seconds"],
        "actual_prompt_ids": receipt["committed_prompt_ids"],
        "reference": record["reference"], "heldout_gain_claim": False,
    }


def publish_endpoint(output):
    """Write expanded_endpoint.json; return its SHA-256, the pin a verifier records."""
    from .vlm_local_training import _write

    path = Path(output) / "expanded_endpoint.json"
    _write(path, endpoint_payload(output))
    return sha(bounded(path))


def validate_endpoint(output, *, expected_receipt_sha256, initial_path=None):
    """No unpinned path: the verifier supplies the receipt SHA it already holds."""
    if not is_sha256(expected_receipt_sha256):
        raise ValueError("a verifier-owned expected endpoint receipt SHA-256 is required")
    output = Path(output)
    raw = bounded(output / "expanded_endpoint.json")
    if sha(raw) != expected_receipt_sha256:
        raise ValueError("endpoint differs from verifier-owned receipt pin")
    payload = json.loads(raw)
    if payload != endpoint_payload(output, initial_path=initial_path):
        raise ValueError("expanded endpoint metadata differs from actual native provenance")
    return payload


def validate_submission(path, *, endpoint, expected_receipt_sha256, initial_path=None):
    """Only weights are submitted; endpoint/receipt pin belong to the verifier."""
    approved = validate_endpoint(endpoint, expected_receipt_sha256=expected_receipt_sha256,
                                 initial_path=initial_path)
    values, weights = read_numeric(path)
    if weights != approved["weights_sha256"]:
        raise ValueError("submitted numeric weights differ from genuine approved trained endpoint")
    return values, approved


def check_matched_controls(rl, controls, *, initial_path=None):
    """Matched SFT controls for one expanded GRPO endpoint, every budget rule explicit.

    ``rl`` is ``{"endpoint": path, "receipt_sha256": pin}``. Each control adds
    ``"arm"``: ``"exposure"`` (the same number of optimizer steps and of prompt
    presentations as RL; which prompts is the sampler's, as in the controlled400
    design) or ``"time"`` (``max_active_seconds`` equal to RL's committed active
    seconds, and reached), and ``"learning_rate"``, its declared LR (a positive
    float). ``"accepted_subset": True`` declares an SFT on a different TRAIN
    manifest whose every committed prompt is one RL actually committed
    (useful-group control); otherwise the task identity must be identical. Every
    endpoint must have exhausted its declared total budget.
    """
    def require_fields(entry, required, optional=frozenset()):
        if (not isinstance(entry, dict) or not required <= set(entry)
                or set(entry) - required - optional
                or not isinstance(entry["endpoint"], (str, Path))
                or not is_sha256(entry["receipt_sha256"])):
            raise ValueError("each endpoint needs exactly its declared fields and receipt pin")
        return entry

    def load(entry):
        payload = validate_endpoint(entry["endpoint"], expected_receipt_sha256=entry[
            "receipt_sha256"], initial_path=initial_path)
        if payload["status"] != "total_budget_exhausted":
            raise ValueError("a matched endpoint must have exhausted its declared total budget")
        return payload

    if not isinstance(controls, list) or not controls:
        raise ValueError("require one GRPO endpoint and at least one declared SFT control")
    require_fields(rl, {"endpoint", "receipt_sha256"})
    for entry in controls:
        require_fields(entry, {"endpoint", "receipt_sha256", "arm", "learning_rate"},
                 {"accepted_subset"})
        rate, subset = entry["learning_rate"], entry.get("accepted_subset", False)
        if (entry["arm"] not in MATCHED_ARMS or type(rate) is not float or not rate > 0
                or rate == float("inf") or type(subset) is not bool):
            raise ValueError("each control declares an exposure/time arm and a positive LR")
    if len({(entry["arm"], entry["learning_rate"]) for entry in controls}) != len(controls):
        raise ValueError("each control declares a distinct exposure/time arm and LR")
    pins = [rl["receipt_sha256"], *(entry["receipt_sha256"] for entry in controls)]
    paths = [Path(entry["endpoint"]).resolve() for entry in (rl, *controls)]
    if len(set(pins)) != len(pins) or len(set(paths)) != len(paths):
        raise ValueError("every matched endpoint must be distinct")
    rl_payload = load(rl)
    if rl_payload["settings"]["mode"] != "grpo":
        raise ValueError("require GRPO and at least one declared expanded SFT control")
    arms = []
    rl_ids = set(rl_payload["actual_prompt_ids"])
    for entry in controls:
        control = load(entry)
        arm, rate, subset = entry["arm"], entry["learning_rate"], entry.get(
            "accepted_subset", False)
        settings = control["settings"]
        if (settings["mode"] != "sft" or settings["learning_rate"] != rate
                or settings["seed"] != rl_payload["settings"]["seed"]
                or control["initial"] != rl_payload["initial"]
                or control["control_identity"]["profile"]
                != rl_payload["control_identity"]["profile"]):
            raise ValueError("expanded controls differ in initialization/profile/seed/mode/LR")
        task = {k: v for k, v in control["control_identity"].items() if k != "train"}
        rl_task = {k: v for k, v in rl_payload["control_identity"].items() if k != "train"}
        same_train = control["control_identity"]["train"] == rl_payload["control_identity"][
            "train"]
        if task != rl_task or (not subset and not same_train) or (subset and same_train):
            raise ValueError("expanded control task differs without a declared accepted subset")
        if subset and not set(control["actual_prompt_ids"]) <= rl_ids:
            raise ValueError("accepted-subset control trained on prompts RL never committed")
        if arm == "exposure" and (
            control["global_step"] != rl_payload["global_step"]
            or settings["max_steps"] != rl_payload["global_step"]
            or len(control["actual_prompt_ids"]) != len(rl_payload["actual_prompt_ids"])
        ):
            raise ValueError("exposure-matched control differs in optimizer steps/presentations")
        if arm == "time" and (
            settings["max_active_seconds"] != rl_payload["committed_active_seconds"]
            or control["committed_active_seconds"] < settings["max_active_seconds"]
        ):
            raise ValueError("time-matched control did not reach RL's committed active seconds")
        arms.append({"arm": arm, "learning_rate": rate, "accepted_subset": subset,
                     "receipt_sha256": entry["receipt_sha256"],
                     "global_step": control["global_step"],
                     "committed_active_seconds": control["committed_active_seconds"]})
    return {"matched_initialization_and_task": True, "rl_receipt_sha256": rl["receipt_sha256"],
            "rl_global_step": rl_payload["global_step"],
            "rl_committed_active_seconds": rl_payload["committed_active_seconds"],
            "controls": arms, "performance_comparison": "not performed"}
