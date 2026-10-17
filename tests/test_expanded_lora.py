"""Expanded language-seven LoRA: trust root, endpoint binding, fair controls, tiny training.

Every expanded-contract test runs on a tiny random trust root built here (TINY layout),
because the production warm adapter weights are not committed. The production trust
root is checked only for the committed bytes it pins.
"""

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision_rl import expanded_contract as e
from vision_rl import vlm_local_task_builder as legacy_grader
from vision_rl import vlm_local_training as native

save = pytest.importorskip("safetensors.numpy").save

INITIAL_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "complete.json", "expansion_receipt.json"
)
TRUSTED_FILES = (
    "conversion-plan.json", "warm-local-training.json",
    "warm/adapter_config.json", "warm/adapter_model.safetensors",
)
WARM_CONFIG = {
    "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16,
    "lora_dropout": 0.0, "bias": "none", "target_modules": ["q_proj", "v_proj"],
}
PROMPT_IDS = ["train-a", "train-b", "train-c", "train-d"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def numeric(values):
    return save(values, metadata={"format": "pt"})


@pytest.fixture
def trust(tmp_path, monkeypatch):
    """A complete tiny approval chain: warm SFT record → plan → INIT → approval pin."""
    conv = e.converter()
    layout = conv.TINY
    assets = tmp_path / "assets"
    rng = np.random.default_rng(7)
    warm = {name: rng.uniform(-0.5, 0.5, shape).astype(np.float32)
            for name, shape in conv.shapes(layout, conv.QV).items()}
    (assets / "warm").mkdir(parents=True)
    (assets / "warm/adapter_model.safetensors").write_bytes(numeric(warm))
    write_json(assets / "warm/adapter_config.json", WARM_CONFIG)
    revision = native.MODEL_REVISIONS[native.DEFAULT_MODEL]
    parent_identity = {
        "model": native.DEFAULT_MODEL, "model_revision": revision,
        "processor_model": native.DEFAULT_MODEL, "processor_revision": revision,
        "profile": asdict(native.Profile()), "data_spec_sha256": "d" * 64,
        "train": {"rows": 4, "sha256": "1" * 64}, "validation": None,
    }
    parent = {"status": "completed", "settings": {"mode": "sft", "seed": 42},
              "final_adapter_sha256": sha(assets / "warm/adapter_model.safetensors"),
              "control_identity": parent_identity}
    write_json(assets / "warm-local-training.json", parent)
    converter_sha = sha(Path(conv.__file__).resolve())
    write_json(assets / "conversion-plan.json", {
        "target_profile": e.PROFILE_ID, "expansion_seed": 42, "converter_sha256": converter_sha,
        "parent_weights_sha256": parent["final_adapter_sha256"]})
    initial = tmp_path / "initial"
    conv.expand_adapter(
        assets / "warm", initial, layout=layout, seed=42,
        expected_weights_sha256=sha(assets / "warm/adapter_model.safetensors"),
        expected_config_sha256=sha(assets / "warm/adapter_config.json"),
        provenance={"kind": "completed_training_export",
                    "record_sha256": sha(assets / "warm-local-training.json"),
                    "control_identity": parent_identity, "settings": parent["settings"]})
    write_json(assets / "approval.json", {
        "profile_id": e.PROFILE_ID, "expansion_seed": 42, "converter_sha256": converter_sha,
        "initial_files": {name: sha(initial / name) for name in INITIAL_FILES},
        "trusted_files": {name: sha(assets / name) for name in TRUSTED_FILES}})
    monkeypatch.setattr(e, "ASSETS", assets)
    monkeypatch.setattr(e, "APPROVAL_SHA256", sha(assets / "approval.json"))
    monkeypatch.setattr(e, "LAYOUT", "TINY")
    profile = native.ExpandedProfile(response_format="answer_only", max_completion_length=8)
    return SimpleNamespace(
        root=tmp_path, assets=assets, initial=initial, profile=profile,
        identity={**parent_identity, "profile": asdict(profile)},
        initial_identity=e.validate_initial(initial),
        values=e.read_numeric(initial / "adapter_model.safetensors")[0],
    )


def make_endpoint(trust, output, *, mode="sft", steps=2, max_steps=2, max_active_seconds=None,
                  committed=10.0, learning_rate=1e-4, seed=42, ids=None, train=None,
                  delta=0.01):
    """Model-free native trajectory written through the real native receipt functions."""
    output = Path(output)
    ids = list(ids or PROMPT_IDS[:steps])
    assert len(ids) == steps
    checkpoint = output / f"checkpoint-{steps}"
    checkpoint.mkdir(parents=True)
    values = {name: value.copy() for name, value in trust.values.items()}
    name = sorted(key for key in values if ".k_proj.lora_B." in key)[0]
    values[name] += np.float32(delta)
    raw = numeric(values)
    config = (trust.initial / "adapter_config.json").read_bytes()
    for directory in (checkpoint, output / "final_adapter"):
        directory.mkdir(exist_ok=True)
        (directory / "adapter_model.safetensors").write_bytes(raw)
        (directory / "adapter_config.json").write_bytes(config)
    if mode == "grpo":
        (checkpoint / "ref").mkdir()
        for file in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(trust.initial / file, checkpoint / "ref" / file)
    write_json(checkpoint / "trainer_state.json", {"global_step": steps})
    for file in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / file).write_bytes(b"synthetic " + file.encode())
    write_json(checkpoint / "exposure.json", {
        "entries": [{"global_step": index + 1, "id": key} for index, key in enumerate(ids)]})
    write_json(checkpoint / "active_time.json", {
        "schema_version": 1, "scope": native.ACTIVE_TIME_SCOPE,
        "total_budget_seconds": max_active_seconds, "global_step": steps,
        "committed_active_seconds": committed,
        "exposure_sha256": sha(checkpoint / "exposure.json"),
        "checkpoint_file_sha256": {
            file: sha(checkpoint / file) for file in native.ACTIVE_CHECKPOINT_FILES}})
    environment = {"adapter_source_sha256": sha(Path(native.__file__)),
                   "expanded_source_closure": e.source_closure()}
    identity = dict(trust.identity)
    if train is not None:
        identity["train"] = train
    record = {
        "status": "completed", "stage_id": "synthetic",
        "settings": {"mode": mode, "max_steps": max_steps,
                     "max_active_seconds": max_active_seconds, "learning_rate": learning_rate,
                     "seed": seed, "beta": 0.04 if mode == "grpo" else None,
                     "init_adapter": trust.initial_identity},
        "requested_boundary": {"stop_after_steps": None, "stop_after_active_seconds": None},
        "control_identity": identity, "environment": environment,
        "reference": {
            "kind": "initial_saved_adapter",
            "adapter": trust.initial_identity["initial_reference"],
            "initial_adapter_weights_sha256": trust.initial_identity["weights_sha256"],
        } if mode == "grpo" else None,
        "optimizer_audit": {"active_time": {"synthetic": True},
                            **({"reference_unchanged": True} if mode == "grpo" else {})},
        "final_adapter_sha256": sha(output / "final_adapter/adapter_model.safetensors"),
    }
    native._archive_source(output, environment["adapter_source_sha256"])
    native._write(output / "local_training.json", record)
    native._publish_endpoint_receipt(output, record, steps)
    return e.publish_endpoint(output)


def changed_weights(trust, path, amount=1.0):
    values = {name: value.copy() for name, value in trust.values.items()}
    values[sorted(values)[0]] += np.float32(amount)
    Path(path).write_bytes(numeric(values))


# Production trust root and legacy contracts.


def test_committed_production_trust_root_matches_its_source_pins():
    assets = Path(e.__file__).with_name("expanded_assets")
    assert sha(assets / "approval.json") == e.APPROVAL_SHA256
    approval = json.loads((assets / "approval.json").read_text())
    assert approval["profile_id"] == e.PROFILE_ID and approval["activation"] is False
    assert approval["converter_sha256"] == sha(Path(e.converter().__file__))
    for name in ("conversion-plan.json", "warm-local-training.json", "warm/adapter_config.json"):
        assert sha(assets / name) == approval["trusted_files"][name]
    assert e.LAYOUT == "QWEN" and e.ENDPOINT_RECEIPT_PINS == frozenset()


def test_legacy_profile_identity_and_112_tensor_grader_are_unchanged():
    assert set(asdict(native.Profile())) == {
        "response_format", "dtype", "min_pixels", "max_pixels", "max_prompt_tokens",
        "max_completion_length", "num_generations", "lora_rank", "temperature"}
    assert not hasattr(native.Profile(), "adapter_profile")
    assert not e.is_expanded(native.Profile())
    assert len(legacy_grader.adapter_shapes()) == 112
    assert legacy_grader.ADAPTER_LIMIT == 16 * 1024**2


def test_expanded_profile_is_versioned_and_strict():
    profile = native.ExpandedProfile()
    profile.validate()
    assert asdict(profile)["adapter_profile"] == e.PROFILE_ID and e.is_expanded(profile)
    for kwargs in ({"adapter_profile": "all-linear"}, {"adapter_profile": e.LEGACY_PROFILE},
                   {"lora_rank": 16}, {"dtype": "bfloat16"}):
        with pytest.raises(ValueError):
            native.ExpandedProfile(**kwargs).validate()


def test_legacy_grader_rejects_the_expanded_initialization(trust):
    with pytest.raises(ValueError):
        legacy_grader.read_adapter(trust.initial / "adapter_model.safetensors")


# INIT trust root.


def test_tiny_approved_conversion_has_external_lineage(trust):
    assert trust.initial_identity["conversion_optimizer_steps"] == 0
    assert trust.initial_identity["initial_reference"]["tensor_count"] == 28
    assert e.artifact_identity(trust.initial, trust.identity, trust.profile) == (
        trust.initial_identity)
    assert native._adapter_identity(trust.initial, trust.identity, trust.profile) == (
        trust.initial_identity)


def test_rehashed_fabricated_conversion_provenance_is_rejected(trust):
    target = trust.root / "forged"
    shutil.copytree(trust.initial, target)
    receipt = e.read_json(target / "expansion_receipt.json")
    receipt["source"]["provenance"]["record_sha256"] = "0" * 64
    (target / "expansion_receipt.json").write_bytes(e.converter().encoded(receipt))
    marker = e.read_json(target / "complete.json")
    marker["file_sha256"]["expansion_receipt.json"] = sha(target / "expansion_receipt.json")
    (target / "complete.json").write_bytes(e.converter().encoded(marker))
    e.converter().verify_expansion(target, e.ASSETS / "warm", layout=e.active_layout())
    with pytest.raises(ValueError, match="externally approved"):
        e.validate_initial(target)


def test_rehashed_approval_or_changed_parent_is_rejected(trust):
    record = e.read_json(trust.assets / "warm-local-training.json")
    record["settings"]["seed"] = 999
    write_json(trust.assets / "warm-local-training.json", record)
    with pytest.raises(ValueError, match="lineage changed"):
        e.approved_lineage()
    approval = e.read_json(trust.assets / "approval.json")
    approval["trusted_files"]["warm-local-training.json"] = sha(
        trust.assets / "warm-local-training.json")
    write_json(trust.assets / "approval.json", approval)
    with pytest.raises(ValueError, match="approval changed"):
        e.approved_lineage()


def test_exact_numeric_layout_precision_finiteness_and_size(trust):
    name = sorted(trust.values)[0]
    path = trust.root / "numeric.safetensors"
    for change in ({name: trust.values[name].astype(np.float64)},
                   {name: np.full(trust.values[name].shape, np.nan, np.float32)},
                   {name: trust.values[name][:1]},
                   {"base_model.model.visual.q_proj.lora_A.weight": np.ones((8, 16), np.float32)}):
        path.write_bytes(save({**trust.values, **change}))
        with pytest.raises(ValueError):
            e.read_numeric(path)
    with path.open("wb") as handle:
        handle.truncate(e.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="bounded"):
        e.read_numeric(path)


def test_symlink_and_fifo_are_refused_before_deserialization(trust):
    link = trust.root / "link"
    link.symlink_to(trust.initial / "adapter_model.safetensors")
    with pytest.raises(OSError):
        e.read_numeric(link)
    pipe = trust.root / "fifo"
    os.mkfifo(pipe)
    with pytest.raises(ValueError, match="regular"):
        e.read_numeric(pipe)


def test_task_mismatch_and_unapproved_initialization_are_refused(trust):
    for key in ("model", "model_revision", "data_spec_sha256"):
        with pytest.raises(ValueError):
            e.artifact_identity(trust.initial, {**trust.identity, key: "changed"}, trust.profile)
    empty = trust.root / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="approved converted"):
        e.artifact_identity(empty, trust.identity, trust.profile)
    with pytest.raises(ValueError, match="precision transfer"):
        native._adapter_identity(trust.initial, trust.identity, trust.profile,
                                 precision_transfer="float32-to-bfloat16")


def test_strict_expanded_config(trust):
    config = e.read_json(trust.initial / "adapter_config.json")
    path = trust.root / "adapter_config.json"
    write_json(path, config)
    e.validate_saved_config(path)
    for change in ({"lora_alpha": 32}, {"target_modules": "all-linear"},
                   {"modules_to_save": ["lm_head"]}, {"use_dora": True},
                   {"unknown_future_feature": True}):
        write_json(path, {**config, **change})
        with pytest.raises(ValueError):
            e.validate_saved_config(path)


def test_reference_rebase_is_rejected_numerically(trust):
    checkpoint = trust.root / "checkpoint"
    for base in (checkpoint, checkpoint / "ref"):
        base.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(trust.initial / name, base / name)
    e.validate_checkpoint(checkpoint, trust.initial_identity, "grpo")
    changed_weights(trust, checkpoint / "ref/adapter_model.safetensors")
    with pytest.raises(ValueError, match="rebased"):
        e.validate_checkpoint(checkpoint, trust.initial_identity, "grpo")


# D6: verifier-owned pins.


def test_endpoint_verification_requires_a_verifier_pin(trust):
    output = trust.root / "run"
    pin = make_endpoint(trust, output)
    for missing in (None, "", "0" * 63, "G" * 64):
        with pytest.raises(ValueError, match="verifier-owned"):
            e.validate_endpoint(output, expected_receipt_sha256=missing)
    with pytest.raises(TypeError):
        e.validate_endpoint(output)
    with pytest.raises(ValueError, match="receipt pin"):
        e.validate_endpoint(output, expected_receipt_sha256="0" * 64)
    payload = e.validate_endpoint(output, expected_receipt_sha256=pin)
    assert payload["kind"] == e.ENDPOINT_KIND and payload["global_step"] == 2
    assert payload["weights_sha256"] == sha(output / "final_adapter/adapter_model.safetensors")
    # The self-consistent tree above is not loadable until the registry lists its pin.
    with pytest.raises(ValueError, match="registry"):
        e.artifact_identity(output / "final_adapter", trust.identity, trust.profile,
                            inference=True)


def test_tampered_endpoint_files_are_rejected(trust):
    output = trust.root / "run"
    pin = make_endpoint(trust, output, mode="grpo")
    mutations = {
        "final": lambda root: changed_weights(
            trust, root / "final_adapter/adapter_model.safetensors"),
        "checkpoint": lambda root: changed_weights(
            trust, root / "checkpoint-2/adapter_model.safetensors"),
        "reference": lambda root: changed_weights(
            trust, root / "checkpoint-2/ref/adapter_model.safetensors"),
        "record": lambda root: (root / "local_training.json").write_text(
            (root / "local_training.json").read_text().replace('"seed": 42', '"seed": 43')),
    }
    for name, mutate in mutations.items():
        copy = trust.root / f"copy-{name}"
        shutil.copytree(output, copy)
        e.validate_endpoint(copy, expected_receipt_sha256=pin)  # the copy itself verifies
        mutate(copy)
        with pytest.raises((ValueError, RuntimeError)):
            e.validate_endpoint(copy, expected_receipt_sha256=pin)


# D1 / D1b: loaded bytes bound to the endpoint's final_adapter.


def test_endpoint_identity_binds_only_final_adapter_bytes(trust, monkeypatch):
    output = trust.root / "run"
    pin = make_endpoint(trust, output)
    monkeypatch.setattr(e, "ENDPOINT_RECEIPT_PINS", frozenset({pin}))
    identity = e.artifact_identity(output / "final_adapter", trust.identity, trust.profile,
                                   inference=True)
    assert identity["weights_sha256"] == sha(output / "final_adapter/adapter_model.safetensors")
    assert identity["expanded_receipt_sha256"] == pin
    assert native._inference_adapter_identity(
        output / "final_adapter", trust.identity, trust.profile) == identity
    planted = output / "planted"
    shutil.copytree(output / "final_adapter", planted)
    changed_weights(trust, planted / "adapter_model.safetensors")
    for path in (planted, output / "checkpoint-2"):
        with pytest.raises(ValueError, match="final_adapter"):
            e.artifact_identity(path, trust.identity, trust.profile, inference=True)
    link = output / "linked"
    link.symlink_to(output / "final_adapter", target_is_directory=True)
    with pytest.raises(ValueError, match="final_adapter"):
        e.artifact_identity(link, trust.identity, trust.profile, inference=True)
    with pytest.raises(ValueError, match="approved converted"):
        native._adapter_identity(output / "final_adapter", trust.identity, trust.profile)


def test_attach_refuses_bytes_other_than_the_validated_ones(trust, monkeypatch):
    output = trust.root / "run"
    pin = make_endpoint(trust, output)
    monkeypatch.setattr(e, "ENDPOINT_RECEIPT_PINS", frozenset({pin}))
    final = output / "final_adapter"
    for expected in (None, "0" * 64, trust.initial_identity["weights_sha256"]):
        with pytest.raises(ValueError):
            e.attach_numeric(object(), final, expected_weights_sha256=expected,
                             is_trainable=False)
    genuine = sha(final / "adapter_model.safetensors")
    original = e.endpoint_identity

    def swap_after_validation(path, **kwargs):
        identity = original(path, **kwargs)
        changed_weights(trust, Path(path) / "adapter_model.safetensors")
        return identity

    monkeypatch.setattr(e, "endpoint_identity", swap_after_validation)
    with pytest.raises(ValueError, match="between validation and numeric attach"):
        e.attach_numeric(object(), final, expected_weights_sha256=genuine, is_trainable=False)


# D3: source closure.


def test_endpoint_binds_the_whole_package_source_closure(trust, monkeypatch):
    closure = e.source_closure()
    package = Path(e.__file__).resolve().parent
    assert set(closure["files"]) == {path.name for path in package.glob("*.py")}
    for name in ("expanded_contract.py", "expanded_converter.py", "expanded_grader.py",
                 "vlm_local_training.py", "vlm_local_task_builder.py", "vlm.py"):
        assert closure["files"][name] == sha(package / name)
    output = trust.root / "run"
    pin = make_endpoint(trust, output)
    assert e.validate_endpoint(output, expected_receipt_sha256=pin)["source_closure"] == closure
    altered = {**closure, "files": {**closure["files"], "expanded_grader.py": "0" * 64}}
    monkeypatch.setattr(e, "source_closure", lambda: altered)
    with pytest.raises(ValueError, match="source closure"):
        e.validate_endpoint(output, expected_receipt_sha256=pin)


# D5: path-independent initialization identity.


def test_endpoint_verifies_after_moving_to_another_machine_layout(trust):
    output = trust.root / "run"
    pin = make_endpoint(trust, output, mode="grpo")
    moved = trust.root / "elsewhere"
    shutil.copytree(output, moved / "run")
    shutil.copytree(trust.initial, moved / "initial")
    shutil.rmtree(trust.initial)
    with pytest.raises((OSError, ValueError)):
        e.validate_endpoint(moved / "run", expected_receipt_sha256=pin)
    payload = e.validate_endpoint(moved / "run", expected_receipt_sha256=pin,
                                  initial_path=moved / "initial")
    assert "path" not in payload["initial"]
    forged = trust.root / "forged-initial"
    shutil.copytree(moved / "initial", forged)
    (forged / "adapter_config.json").write_text(
        (forged / "adapter_config.json").read_text() + " ")
    with pytest.raises(ValueError, match="externally approved"):
        e.validate_endpoint(moved / "run", expected_receipt_sha256=pin, initial_path=forged)


# D2: fair controls.


@pytest.fixture
def controls(trust):
    root = trust.root / "study"
    rl = {"endpoint": root / "rl", "receipt_sha256": make_endpoint(
        trust, root / "rl", mode="grpo", steps=4, max_steps=4, committed=100.0)}

    def control(name, *, arm="exposure", learning_rate=1e-4, accepted_subset=None, **kwargs):
        defaults = {"steps": 4, "max_steps": 4, "committed": 30.0}
        if arm == "time":
            defaults = {"steps": 4, "max_steps": 400, "max_active_seconds": 100.0,
                        "committed": 100.5}
        pin = make_endpoint(trust, root / name, learning_rate=learning_rate,
                            **{**defaults, **kwargs})
        entry = {"endpoint": root / name, "receipt_sha256": pin, "arm": arm,
                 "learning_rate": learning_rate}
        if accepted_subset is not None:
            entry["accepted_subset"] = accepted_subset
        return entry

    return SimpleNamespace(rl=rl, control=control)


def test_matched_controls_accept_exposure_and_time_arms(controls):
    exposure = controls.control("sft-exposure")
    time = controls.control("sft-time", arm="time")
    result = e.check_matched_controls(controls.rl, [exposure, time])
    assert [arm["arm"] for arm in result["controls"]] == ["exposure", "time"]
    assert result["rl_global_step"] == 4 and result["rl_committed_active_seconds"] == 100.0


def test_matched_controls_reject_unmatched_budgets(controls):
    one_step = {"steps": 1, "max_steps": 1, "committed": 1.0}
    for arm in ("exposure", "time"):
        entry = controls.control(f"short-{arm}", arm=arm, **one_step)
        with pytest.raises(ValueError, match="matched"):
            e.check_matched_controls(controls.rl, [entry])
    short_time = controls.control("short-time-cap", arm="time", max_active_seconds=50.0,
                                  committed=50.0)
    with pytest.raises(ValueError, match="time-matched"):
        e.check_matched_controls(controls.rl, [short_time])


def test_matched_controls_reject_duplicates_and_missing_or_wrong_pins(controls):
    entry = controls.control("sft")
    with pytest.raises(ValueError, match="distinct"):
        e.check_matched_controls(controls.rl, [entry, entry])
    with pytest.raises(ValueError, match="distinct"):
        e.check_matched_controls(controls.rl, [entry, {**entry, "arm": "time"}])
    unpinned = {key: value for key, value in entry.items() if key != "receipt_sha256"}
    with pytest.raises(ValueError):
        e.check_matched_controls(controls.rl, [unpinned])
    with pytest.raises(ValueError, match="receipt pin"):
        e.check_matched_controls(controls.rl, [{**entry, "receipt_sha256": "0" * 64}])
    with pytest.raises(ValueError, match="receipt pin"):
        e.check_matched_controls({**controls.rl, "receipt_sha256": "0" * 64}, [entry])
    with pytest.raises(ValueError):
        e.check_matched_controls(entry | {"arm": None}, [entry])
    with pytest.raises(ValueError):
        e.check_matched_controls(controls.rl, [])


def test_matched_controls_reject_changed_seed_lr_or_arm(controls):
    with pytest.raises(ValueError, match="seed"):
        e.check_matched_controls(controls.rl, [controls.control("seed", seed=7)])
    wrong_rate = {**controls.control("rate"), "learning_rate": 5e-5}
    with pytest.raises(ValueError, match="LR"):
        e.check_matched_controls(controls.rl, [wrong_rate])
    first = controls.control("lr-a")
    second = controls.control("lr-b", delta=0.02)
    with pytest.raises(ValueError, match="distinct exposure/time arm"):
        e.check_matched_controls(controls.rl, [first, second])
    with pytest.raises(ValueError, match="arm"):
        e.check_matched_controls(controls.rl, [{**controls.control("odd"), "arm": "steps"}])


def test_accepted_subset_control_is_explicit(controls):
    other_train = {"rows": 2, "sha256": "2" * 64}
    subset_ids = ["train-a", "train-b", "train-a", "train-c"]
    accepted = controls.control("accepted", accepted_subset=True, train=other_train,
                                ids=subset_ids)
    result = e.check_matched_controls(controls.rl, [accepted])
    assert result["controls"][0]["accepted_subset"] is True
    undeclared = controls.control("undeclared", train=other_train, ids=subset_ids, delta=0.02)
    with pytest.raises(ValueError, match="accepted subset"):
        e.check_matched_controls(controls.rl, [undeclared])
    outside = controls.control("outside", accepted_subset=True, train=other_train,
                               ids=["train-a", "train-z", "train-a", "train-c"], delta=0.03)
    with pytest.raises(ValueError, match="never committed"):
        e.check_matched_controls(controls.rl, [outside])
    same_train = controls.control("same", accepted_subset=True, delta=0.04)
    with pytest.raises(ValueError, match="accepted subset"):
        e.check_matched_controls(controls.rl, [same_train])


def test_matched_controls_require_a_grpo_rl_endpoint(controls):
    sft = controls.control("as-rl")
    other = controls.control("other", delta=0.02, learning_rate=5e-5)
    rl = {key: sft[key] for key in ("endpoint", "receipt_sha256")}
    with pytest.raises(ValueError, match="GRPO"):
        e.check_matched_controls(rl, [other])


# D4: a real tiny-model training run publishes and verifies an endpoint.


def tiny_base(torch):
    from transformers import LlamaConfig, LlamaForCausalLM, set_seed

    set_seed(42)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=6, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, eos_token_id=1,
        bos_token_id=2, attention_dropout=0.0))
    # Vision decoys: the language-only layout must never adapt them.
    base.visual = torch.nn.Module()
    base.visual.q_proj = torch.nn.Linear(16, 16)
    base.visual.gate_proj = torch.nn.Linear(16, 32)
    return base


def train_tiny(trust, work, mode, torch):
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    backend = Tokenizer(models.WordLevel(
        {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3, "a": 4, "b": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]")
    model = e.attach_numeric(tiny_base(torch), trust.initial, is_trainable=True,
                             expected_weights_sha256=trust.initial_identity["weights_sha256"])
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    settings = native.trainer_kwargs(mode, work, trust.profile, device="cpu", max_steps=2,
                                     learning_rate=0.001, seed=42, save_steps=1, beta=0.04)
    settings["gradient_checkpointing"] = False
    ids = ["synthetic-a", "synthetic-b", "synthetic-c"]
    audit = native._audit_callback(work, 600.0, exposure=native._ExposureLedger(work, ids))
    data = {"id": ids, "prompt": ["a", "b", "a b"]}
    if mode == "sft":
        data["completion"] = [" b [EOS]", " a [EOS]", " a [EOS]"]

    def reward(completions, completion_ids, id, **_):
        assert len(set(id)) == 1
        audit.observe_prompt(id[0])
        return [float(bool(tokens) and tokens[-1] == 1 and "a" in text.split())
                for text, tokens in zip(completions, completion_ids, strict=True)]

    constructor = (native._trainer_with_grpo_resume(GRPOTrainer) if mode == "grpo"
                   else native._trainer_with_sft_exposure(SFTTrainer, audit))
    trainer = native._trainer_with_active_time(constructor)(
        model=model, args=(GRPOConfig if mode == "grpo" else SFTConfig)(**settings),
        processing_class=tokenizer, train_dataset=Dataset.from_dict(data), callbacks=[audit],
        **({"reward_funcs": reward} if mode == "grpo" else {}))
    reference = None
    if mode == "sft":
        trainer.data_collator = native._sft_collator_with_ids(trainer.data_collator)
    else:
        reference = native._reference_identity(
            trainer, init_adapter=trust.initial_identity, model=native.DEFAULT_MODEL)
        assert reference["adapter"] == trust.initial_identity["initial_reference"]
        audit.expected_reference = reference["adapter"]
    trainer.train()
    assert trainer.state.global_step == 2
    trainer.save_model(str(work / "final_adapter"))
    environment = native.preflight("cpu")
    environment["expanded_source_closure"] = e.source_closure()
    record = {
        "status": "completed", "stage_id": f"tiny-{mode}",
        "settings": {"mode": mode, "max_steps": 2, "max_active_seconds": 600.0,
                     "learning_rate": 0.001, "seed": 42,
                     "beta": 0.04 if mode == "grpo" else None,
                     "init_adapter": trust.initial_identity},
        "requested_boundary": {"stop_after_steps": None, "stop_after_active_seconds": None},
        "control_identity": trust.identity, "environment": environment,
        "reference": reference, "optimizer_audit": audit.summary,
        "final_adapter_sha256": sha(work / "final_adapter/adapter_model.safetensors"),
    }
    native._archive_source(work, environment["adapter_source_sha256"])
    native._write(work / "local_training.json", record)
    native._publish_endpoint_receipt(work, record, trainer.state.global_step)
    changed = [name for name, value in model.named_parameters()
               if name in initial and not torch.equal(value, initial[name])]
    model.eval()
    return e.publish_endpoint(work), model, changed


@pytest.mark.parametrize("mode", ["sft", "grpo"])
def test_tiny_model_training_publishes_a_verifiable_endpoint(trust, monkeypatch, mode):
    torch = pytest.importorskip("torch")
    for module in ("peft", "trl", "datasets", "tokenizers", "transformers"):
        pytest.importorskip(module)
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cpu")
    torch.set_num_threads(1)
    work = trust.root / f"tiny-{mode}"
    work.mkdir()
    pin, model, changed = train_tiny(trust, work, mode, torch)
    assert all(".default." in name and "lora_" in name for name in changed)
    if mode == "sft":
        assert any(".q_proj." not in name and ".v_proj." not in name for name in changed)

    payload = e.validate_endpoint(work, expected_receipt_sha256=pin)
    final = work / "final_adapter/adapter_model.safetensors"
    assert payload["global_step"] == 2 and payload["weights_sha256"] == sha(final)
    assert payload["status"] == "total_budget_exhausted"
    assert len(payload["actual_prompt_ids"]) == 2
    if mode == "grpo":
        assert payload["reference"]["adapter"] == trust.initial_identity["initial_reference"]
    _, approved = e.validate_submission(final, endpoint=work, expected_receipt_sha256=pin)
    assert approved == payload

    monkeypatch.setattr(e, "ENDPOINT_RECEIPT_PINS", frozenset({pin}))
    identity = native._inference_adapter_identity(
        work / "final_adapter", trust.identity, trust.profile)
    loaded = e.attach_numeric(tiny_base(torch), work / "final_adapter", is_trainable=False,
                              expected_weights_sha256=identity["weights_sha256"]).eval()
    started = e.attach_numeric(tiny_base(torch), trust.initial, is_trainable=False,
                               expected_weights_sha256=trust.initial_identity["weights_sha256"])
    sample = torch.tensor([[2, 4, 5, 1]])
    with torch.no_grad():
        trained_logits = model(sample).logits
        torch.testing.assert_close(loaded(sample).logits, trained_logits, rtol=1e-6, atol=1e-6)
        if mode == "sft":
            assert not torch.equal(started.eval()(sample).logits, trained_logits)

    planted = work / "planted"
    shutil.copytree(work / "final_adapter", planted)
    with pytest.raises(ValueError, match="final_adapter"):
        native._inference_adapter_identity(planted, trust.identity, trust.profile)
    with pytest.raises(ValueError, match="receipt pin"):
        e.validate_endpoint(work, expected_receipt_sha256="0" * 64)
    forged = trust.root / f"forged-initial-{mode}"
    shutil.copytree(trust.initial, forged)
    (forged / "complete.json").write_text((forged / "complete.json").read_text() + " ")
    with pytest.raises(ValueError, match="externally approved"):
        e.validate_endpoint(work, expected_receipt_sha256=pin, initial_path=forged)
    tampered = {
        "final": "final_adapter/adapter_model.safetensors",
        "checkpoint": "checkpoint-2/adapter_model.safetensors",
    }
    if mode == "grpo":
        tampered["reference"] = "checkpoint-2/ref/adapter_model.safetensors"
    for name, relative in tampered.items():
        copy = trust.root / f"tamper-{mode}-{name}"
        shutil.copytree(work, copy)
        e.validate_endpoint(copy, expected_receipt_sha256=pin)
        changed_weights(trust, copy / relative)
        with pytest.raises((ValueError, RuntimeError)):
            e.validate_endpoint(copy, expected_receipt_sha256=pin)
