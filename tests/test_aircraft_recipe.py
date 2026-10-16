"""Controller contract fixtures, deliberately not model or learning evidence."""

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_rl import aircraft_recipe as recipe
from vision_rl import experiment_loop as loop
from vision_rl import vlm_local_training as native


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def declaration():
    return {
        "schema_version": 1,
        "study_id": "contract-fixture",
        "activated": True,
        "scope": recipe.SCOPE,
        "device": "cpu",
        "profile": asdict(native.Profile()),
        "seed": 42,
        "max_steps": 8,
        "max_active_seconds": 100.0,
        "stop_after_steps": 2,
        "save_steps": 1,
        "rl_learning_rate": 1e-5,
        "beta": 0.04,
        "sft_controls": [
            {"id": "matched", "learning_rate": 1e-5, "response_format": "think_then_answer"}
        ],
        "assay_seeds": [42, 314],
        "budget": {"max_wall_seconds": 600, "command_timeout_seconds": 60},
    }


class FixtureTrainer:
    """Deterministic file fixtures; no tensors, model, gradient or reward training."""

    Profile = native.Profile
    load_spec = staticmethod(native.load_spec)
    load_classification = staticmethod(native.load_classification)
    _classification_identity = staticmethod(native._classification_identity)

    def __init__(self):
        self.calls = []
        self.finalizations = []
        self.assays = []
        self.fail_after_commit = False

    def train_local(self, train, validation, data_spec, output, *, finalize_only=False, **settings):
        assert validation is None and settings["train_only"]
        (self.finalizations if finalize_only else self.calls).append(settings)
        output.mkdir(parents=True, exist_ok=True)
        mode = settings["mode"]
        seconds_per_step = 3 if mode == "grpo" else 1
        steps = settings["stop_after_steps"] or int(settings["stop_after_active_seconds"])
        steps = min(steps, settings["max_steps"])
        if finalize_only:
            steps = int(settings["resume_from_checkpoint"].name.split("-")[-1])
        if self.fail_after_commit:
            steps = max(1, steps - 1)
        checkpoint = output / f"checkpoint-{steps}"
        if not finalize_only:
            checkpoint.mkdir(exist_ok=True)
            write(checkpoint / "active_time.json", {"global_step": steps})
            write(checkpoint / "exposure.json", {})
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise InterruptedError("fixture failure after durable commit")
        adapter = output / "final_adapter"
        adapter.mkdir(exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(f"fixture-{mode}-{steps}".encode())
        write(adapter / "adapter_config.json", {"fixture": True})
        initial = loop._hash_path(settings["init_adapter"] / "adapter_model.safetensors")
        record = {
            "status": "completed",
            "final_adapter_sha256": loop._hash_path(adapter / "adapter_model.safetensors"),
            "settings": {
                name: settings[name] for name in ("mode", "learning_rate", "device", "seed")
            },
            "control_identity": {"profile": asdict(settings["profile"])},
            "reference": {"initial_adapter_weights_sha256": initial},
            "optimizer_audit": {"reference_unchanged": True},
        }
        record["settings"]["init_adapter"] = {"weights_sha256": initial}
        record["settings"]["beta"] = settings["beta"] if mode == "grpo" else None
        record["control_identity"]["data_spec_sha256"] = loop._hash_path(data_spec)
        rows = native.load_classification(train, "train", ["red", "blue"])
        record["control_identity"]["train"] = native._classification_identity(train, rows)
        write(output / "local_training.json", record)
        receipt = {
            "schema_version": 1,
            "stage_id": settings["stage_id"],
            "status": "total_budget_exhausted"
            if steps == settings["max_steps"]
            else "boundary_reached",
            "original_budgets": {
                name: settings[name] for name in ("max_steps", "max_active_seconds")
            },
            "global_step": steps,
            "committed_active_seconds": steps * seconds_per_step,
            "committed_prompt_presentations": steps,
            "committed_prompt_ids": ["a", "b"] * (steps // 2) + (["a"] if steps % 2 else []),
            "source_sha256": loop._hash_path(Path(native.__file__)),
            "requested_boundary": {
                name: settings[name] for name in ("stop_after_steps", "stop_after_active_seconds")
            },
            "final_adapter": {
                "weights_sha256": record["final_adapter_sha256"],
                "config_sha256": loop._hash_path(adapter / "adapter_config.json"),
            },
            "physical_work": {
                "observed_physical_active_seconds": steps * seconds_per_step,
                "observed_discarded_or_uncheckpointed_active_seconds": 0,
            },
        }
        write(output / "endpoint_receipt.json", receipt)
        return record

    @staticmethod
    def read_endpoint_receipt(output, *, expected_stage_id=None):
        result = recipe.read(output / "endpoint_receipt.json")
        if result["stage_id"] != expected_stage_id:
            raise ValueError("fixture stage identity mismatch")
        if result["final_adapter"]["weights_sha256"] != loop._hash_path(
            output / "final_adapter" / "adapter_model.safetensors"
        ):
            raise ValueError("fixture altered adapter")
        return result

    def predict_train_subset(
        self,
        manifest,
        *,
        output,
        profile,
        init_adapter,
        data_spec,
        device,
        model_directory,
        seed=None,
        sample=False,
    ):
        self.assays.append((str(output), sample, seed))
        output.mkdir(parents=True)
        record = {
            "status": "completed",
            "profile": asdict(profile),
            "initial_adapter": {
                "weights_sha256": loop._hash_path(init_adapter / "adapter_model.safetensors")
            },
            "data_spec_sha256": loop._hash_path(data_spec),
            "data_identity": native._classification_identity(
                manifest, native.load_classification(manifest, "train", ["red", "blue"])
            ),
            "environment": {
                "device": device,
                "adapter_source_sha256": loop._hash_path(Path(native.__file__)),
            },
            "generation": {"do_sample": sample, "seed": seed if sample else None},
            "elapsed_seconds": 1.0,
        }
        raw = {
            "a": {"response": "red", "terminated": True},
            "b": {"response": "invalid", "terminated": True},
        }
        if sample:
            raw = {key: [value] * profile.num_generations for key, value in raw.items()}
        write(output / ("sampled_completions.json" if sample else "predictions.json"), raw)
        write(
            output / ("train_subset_groups.json" if sample else "train_subset_inference.json"),
            record,
        )
        return record

    def sample_train_groups(self, manifest, **kwargs):
        return self.predict_train_subset(manifest, sample=True, **kwargs)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    fake = FixtureTrainer()
    monkeypatch.setattr(recipe, "trainer_api", lambda: fake)
    spec = {
        "schema_version": 1,
        "benchmark": "unit fixture, not Aircraft evidence",
        "question": "Color?",
        "labels": ["red", "blue"],
        "split_policy": "TRAIN only",
        "preprocessing": "unit fixture",
        "sources": [{"url": "fixture://test", "sha256": "0" * 64}],
    }
    write(tmp_path / "spec.json", spec)
    rows = []
    for identifier, label in zip(["a", "b"], spec["labels"], strict=True):
        Image.new("RGB", (4, 4), label).save(tmp_path / (identifier + ".png"))
        rows.append(
            {"id": identifier, "label": label, "image": identifier + ".png", "split": "train"}
        )
    for name in ("train", "assay"):
        (tmp_path / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    initial = tmp_path / "warmup" / "final_adapter"
    initial.mkdir(parents=True)
    (initial / "adapter_model.safetensors").write_bytes(b"numeric fixture stand-in; never loaded")
    write(initial / "adapter_config.json", {"fixture": True})
    train_rows = native.load_classification(tmp_path / "train.jsonl", "train", spec["labels"])
    write(
        initial.parent / "local_training.json",
        {
            "status": "completed",
            "final_adapter_sha256": loop._hash_path(initial / "adapter_model.safetensors"),
            "settings": {"max_steps": 400},
            "optimizer_audit": {},
            "control_identity": {
                "data_spec_sha256": loop._hash_path(tmp_path / "spec.json"),
                "train": native._classification_identity(tmp_path / "train.jsonl", train_rows),
            },
        },
    )
    model = tmp_path / "model"
    write(model / "model-source.json", {"files": [{"path": "weights", "size_bytes": 7}]})
    write(model / ".pinned-download.json", {})
    (model / "weights").write_bytes(b"fixture")
    write(tmp_path / "study.json", declaration())
    plan_path = recipe.prepare(
        tmp_path / "train.jsonl",
        tmp_path / "assay.jsonl",
        tmp_path / "spec.json",
        initial,
        model,
        tmp_path / "study.json",
        tmp_path / "plan",
    )
    state = tmp_path / "state"
    state.mkdir()
    return SimpleNamespace(
        fake=fake,
        root=tmp_path,
        plan=recipe.read(plan_path),
        context=recipe.read(plan_path.parent / "context.json"),
        state=state,
    )


def invoke(h, kind, arm=None, endpoint=None):
    identifier = {"common": "common", "compare": "comparison"}.get(kind)
    if kind == "assay":
        identifier = "initial-assay" if arm == "initial" else f"{arm}-{endpoint}-assay"
    if kind == "train":
        identifier = f"{arm}-{endpoint}"
    output = h.state / "setup" / identifier
    output.mkdir(parents=True, exist_ok=True)
    return recipe.stage(h.root / "plan" / "context.json", h.state, output, kind, arm, endpoint)


def test_plan_reuses_runner_and_freezes_inputs(harness):
    plan = harness.plan
    assert plan["evaluation_mode"] == "development"
    assert [stage["id"] for stage in plan["setup"]] == [
        "common",
        "initial-assay",
        "rl-exposure",
        "rl-exposure-assay",
        "matched-exposure",
        "matched-exposure-assay",
        "matched-time",
        "matched-time-assay",
        "comparison",
    ]
    assert str(harness.root / "model" / "weights") in plan["inputs"]
    for transitive in ("evaluation.py", "spatial.py", "training.py"):
        assert str(recipe.PACKAGE / transitive) in plan["inputs"]
    assert all("evaluate" not in candidate for candidate in plan["candidates"])
    assert not harness.fake.calls


def test_prepare_and_common_stage_use_same_isolated_package(harness):
    """Real subprocesses import a frozen package; no model load or mocked native API."""
    h = harness
    isolated = h.root / "isolated" / "source" / "vision_rl"
    shutil.copytree(recipe.PACKAGE, isolated, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    environment = {**os.environ, "PYTHONPATH": str(isolated.parent)}
    plan_dir = h.root / "isolated-plan"
    command = [sys.executable, "-B", "-m", "vision_rl.aircraft_recipe", "prepare"]
    for flag, path in {
        "train": h.root / "train.jsonl",
        "assay": h.root / "assay.jsonl",
        "data-spec": h.root / "spec.json",
        "initial-adapter": h.root / "warmup" / "final_adapter",
        "model-directory": h.root / "model",
        "study": h.root / "study.json",
        "output": plan_dir,
    }.items():
        command += ["--" + flag, str(path)]
    subprocess.run(command, env=environment, cwd=h.root, check=True, capture_output=True)
    context = recipe.read(plan_dir / "context.json")
    assert all(Path(path).is_relative_to(isolated) for path in context["source_hashes"])
    state = h.root / "isolated-state"
    command = [
        sys.executable,
        "-B",
        "-m",
        "vision_rl.aircraft_recipe",
        "stage",
        "--context",
        str(plan_dir / "context.json"),
        "--state-dir",
        str(state),
        "--output",
        str(state / "setup" / "common"),
        "--kind",
        "common",
    ]
    subprocess.run(command, env=environment, cwd=h.root, check=True, capture_output=True)
    assert (
        recipe.read(state / "setup" / "common" / "published" / "receipt.json")[
            "training_performed_by_recipe"
        ]
        is False
    )
    with (isolated / "evaluation.py").open("a") as handle:
        handle.write("\n# altered transitive dependency\n")
    failed = subprocess.run(command, env=environment, cwd=h.root, capture_output=True, text=True)
    assert failed.returncode == 2 and "implementation changed" in failed.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("activated", False),
        ("scope", "full_validation"),
        ("max_steps", 1200),
        ("stop_after_steps", 9),
        ("max_active_seconds", float("nan")),
        ("assay_seeds", [42, 42]),
        ("sft_controls", []),
    ],
)
def test_invalid_or_unactivated_plan_rejected(field, value):
    spec = declaration()
    spec[field] = value
    with pytest.raises(ValueError):
        recipe.normalize_study(spec)


def test_missing_actual_receipt_api_fails_explicitly(monkeypatch):
    monkeypatch.delattr(native, "read_endpoint_receipt", raising=False)
    with pytest.raises(ValueError, match="no model or training"):
        recipe.trainer_api()


def test_larger_sft_horizon_requires_native_capability_before_any_work(monkeypatch):
    spec = declaration()
    spec.update(schema_version=2, sft_max_steps=3600, sft_max_active_seconds=200)
    monkeypatch.delattr(native, "TRAIN_ONLY_STEP_CAPS", raising=False)
    with pytest.raises(ValueError, match="current native capability"):
        recipe.normalize_study(spec)
    monkeypatch.setattr(native, "TRAIN_ONLY_STEP_CAPS", {"grpo": 400, "sft": 3600}, raising=False)
    assert recipe.normalize_study(spec)["sft_max_steps"] == 3600
    monkeypatch.setattr(native, "TRAIN_ONLY_STEP_CAPS", {"grpo": 400, "sft": None})
    assert recipe.normalize_study(spec)["sft_max_steps"] == 3600
    spec["max_steps"] = 401
    with pytest.raises(ValueError, match="400 RL"):
        recipe.normalize_study(spec)


def test_separate_sft_original_horizon_survives_time_continuation(harness):
    h = harness
    h.context["study"].update(schema_version=2, sft_max_steps=400, sft_max_active_seconds=200)
    write(h.root / "plan" / "context.json", h.context)
    invoke(h, "common")
    invoke(h, "train", "rl", "exposure")
    invoke(h, "train", "matched", "exposure")
    invoke(h, "train", "matched", "time")
    rl, sft, resumed_sft = h.fake.calls
    assert rl["max_steps"] == 8 and rl["max_active_seconds"] == 100
    assert sft["max_steps"] == resumed_sft["max_steps"] == 400
    assert sft["max_active_seconds"] == resumed_sft["max_active_seconds"] == 200
    assert resumed_sft["stop_after_active_seconds"] == 6
    assert resumed_sft["resume_from_checkpoint"].name == "checkpoint-2"


def test_equal_time_continues_original_trajectory_and_budgets(harness):
    h = harness
    invoke(h, "common")
    invoke(h, "train", "rl", "exposure")
    invoke(h, "train", "matched", "exposure")
    prior = recipe._endpoint(h.state, "matched", "exposure")
    old_hash = loop._hash_path(prior)
    invoke(h, "train", "matched", "time")
    first, second, continuation = h.fake.calls
    assert first["max_steps"] == second["max_steps"] == continuation["max_steps"] == 8
    assert continuation["max_active_seconds"] == 100
    assert continuation["stop_after_steps"] is None
    assert continuation["stop_after_active_seconds"] == 6
    assert continuation["resume_from_checkpoint"].name == "checkpoint-2"
    assert loop._hash_path(prior) == old_hash
    assert recipe.read(prior / "trainer-receipt.json")["global_step"] == 2


@pytest.mark.parametrize("cap", [2, 4])
def test_capped_sft_retains_explicit_unmet_time_control(harness, cap):
    h = harness
    h.context["study"]["max_steps"] = cap
    write(h.root / "plan" / "context.json", h.context)
    invoke(h, "common")
    invoke(h, "train", "rl", "exposure")
    invoke(h, "train", "matched", "exposure")
    with pytest.raises(ValueError, match="equal-time control is incomplete"):
        invoke(h, "train", "matched", "time")
    output = h.state / "setup" / "matched-time"
    report = recipe.read(output / "unmet-time-control.json")
    assert not report["equal_time_control_complete"]
    assert not report["primary_requirement_satisfied"]
    assert report["target_committed_active_seconds"] == 6
    assert report["actual_committed_active_seconds"] == cap
    assert report["shortfall_seconds"] == 6 - cap
    assert report["original_budgets"]["max_steps"] == cap
    assert not (output / "published").exists()
    assert len(h.fake.calls) == (2 if cap == 2 else 3)


def test_completed_endpoint_publication_retry_never_retrains(harness, monkeypatch):
    h = harness
    invoke(h, "common")
    original = recipe.shutil.copytree
    monkeypatch.setattr(
        recipe.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(InterruptedError())
    )
    with pytest.raises(InterruptedError):
        invoke(h, "train", "rl", "exposure")
    monkeypatch.setattr(recipe.shutil, "copytree", original)
    invoke(h, "train", "rl", "exposure")
    assert len(h.fake.calls) == 1
    assert list((h.state / "setup" / "rl-exposure").glob("incomplete-*"))


def test_interrupted_arm_uses_durable_checkpoint_not_fresh_budget(harness):
    h = harness
    invoke(h, "common")
    h.fake.fail_after_commit = True
    with pytest.raises(InterruptedError):
        invoke(h, "train", "rl", "exposure")
    invoke(h, "train", "rl", "exposure")
    assert h.fake.calls[1]["resume_from_checkpoint"].name == "checkpoint-1"
    assert h.fake.calls[1]["max_active_seconds"] == h.fake.calls[0]["max_active_seconds"]


def test_no_durable_checkpoint_never_restarts(harness):
    h = harness
    invoke(h, "common")
    (h.state / "work" / "rl").mkdir(parents=True)
    with pytest.raises(ValueError, match="do not restart"):
        invoke(h, "train", "rl", "exposure")
    assert not h.fake.calls


def test_boundary_checkpoint_without_endpoint_finalizes_without_retraining(harness):
    h = harness
    invoke(h, "common")
    checkpoint = h.state / "work" / "rl" / "checkpoint-2"
    write(checkpoint / "active_time.json", {"global_step": 2, "committed_active_seconds": 6})
    write(checkpoint / "exposure.json", {})
    checkpoint_before = loop._hash_path(checkpoint)
    invoke(h, "train", "rl", "exposure")
    assert not h.fake.calls
    assert len(h.fake.finalizations) == 1
    assert loop._hash_path(checkpoint) == checkpoint_before
    receipt = recipe.read(recipe._endpoint(h.state, "rl", "exposure") / "receipt.json")
    assert receipt["zero_update_finalization_performed"]
    assert not receipt["training_performed_in_publishing_invocation"]


def test_total_cap_checkpoint_finalizes_but_does_not_claim_unmet_time(harness):
    h = harness
    h.context["study"]["max_steps"] = 4
    write(h.root / "plan" / "context.json", h.context)
    invoke(h, "common")
    invoke(h, "train", "rl", "exposure")
    invoke(h, "train", "matched", "exposure")
    checkpoint = h.state / "work" / "matched" / "checkpoint-4"
    write(checkpoint / "active_time.json", {"global_step": 4, "committed_active_seconds": 4})
    write(checkpoint / "exposure.json", {})
    with pytest.raises(ValueError, match="equal-time control is incomplete"):
        invoke(h, "train", "matched", "time")
    assert len(h.fake.calls) == 2 and len(h.fake.finalizations) == 1
    assert not (h.state / "setup" / "matched-time" / "published").exists()


def test_missing_finalization_capability_fails_before_model(harness):
    h = harness
    invoke(h, "common")
    checkpoint = h.state / "work" / "rl" / "checkpoint-2"
    write(checkpoint / "active_time.json", {"global_step": 2, "committed_active_seconds": 6})
    write(checkpoint / "exposure.json", {})
    h.fake.train_local = lambda *args, **kwargs: pytest.fail("must not load/train a model")
    with pytest.raises(ValueError, match="zero-update trainer endpoint finalization"):
        invoke(h, "train", "rl", "exposure")


def test_exact_final_step_is_valid_even_when_total_budget_is_exhausted(harness):
    h = harness
    h.context["study"]["max_steps"] = 2
    write(h.root / "plan" / "context.json", h.context)
    invoke(h, "common")
    invoke(h, "train", "rl", "exposure")
    receipt = recipe.read(recipe._endpoint(h.state, "rl", "exposure") / "trainer-receipt.json")
    assert receipt["status"] == "total_budget_exhausted"


def test_budget_exhausted_before_requested_exposure_does_not_publish(harness):
    h = harness
    invoke(h, "common")
    original = h.fake.read_endpoint_receipt

    def incomplete(*args, **kwargs):
        receipt = original(*args, **kwargs)
        receipt.update(
            status="total_budget_exhausted",
            global_step=1,
            committed_prompt_presentations=1,
            committed_prompt_ids=["a"],
        )
        return receipt

    h.fake.read_endpoint_receipt = incomplete
    with pytest.raises(ValueError, match="unexpected optimizer-step"):
        invoke(h, "train", "rl", "exposure")


def test_unknown_committed_train_id_rejected(harness):
    h = harness
    invoke(h, "common")
    original = h.fake.read_endpoint_receipt

    def changed(*args, **kwargs):
        receipt = original(*args, **kwargs)
        receipt["committed_prompt_ids"][0] = "sealed-test-id"
        return receipt

    h.fake.read_endpoint_receipt = changed
    with pytest.raises(ValueError, match="TRAIN lineage"):
        invoke(h, "train", "rl", "exposure")


def test_assay_completion_reused_but_partial_never_resampled(harness):
    h = harness
    invoke(h, "common")
    invoke(h, "assay", "initial", "initial")
    count = len(h.fake.assays)
    invoke(h, "assay", "initial", "initial")
    assert len(h.fake.assays) == count
    path = (
        h.state
        / "setup"
        / "initial-assay"
        / "work"
        / "think_then_answer-greedy"
        / "train_subset_inference.json"
    )
    record = recipe.read(path)
    record["status"] = "failed"
    write(path, record)
    with pytest.raises(ValueError, match="never automatically resample"):
        invoke(h, "assay", "initial", "initial")


def test_assay_releases_each_new_model_before_next_load(harness, monkeypatch):
    h = harness
    released = []
    monkeypatch.setattr(
        recipe,
        "_release_inference_model",
        lambda device: released.append((device, len(h.fake.assays))),
    )
    invoke(h, "common")
    invoke(h, "assay", "initial", "initial")
    assert released == [("cpu", 1), ("cpu", 2), ("cpu", 3)]
    invoke(h, "assay", "initial", "initial")
    assert len(released) == 3


def test_failed_assay_releases_model_and_preserves_partial_outputs(harness, monkeypatch):
    h = harness
    released = []
    monkeypatch.setattr(recipe, "_release_inference_model", lambda device: released.append(device))

    def fail(manifest, **kwargs):
        write(kwargs["output"] / "partial.json", {"error": "fixture generation failed"})
        raise RuntimeError("fixture generation failed")

    monkeypatch.setattr(h.fake, "predict_train_subset", fail)
    invoke(h, "common")
    with pytest.raises(RuntimeError, match="fixture generation failed"):
        invoke(h, "assay", "initial", "initial")
    assert released == ["cpu"]
    stage = h.state / "setup" / "initial-assay"
    assert list(stage.glob("work/*/partial.json"))
    assert not (stage / "published").exists()


def test_precision_transfer_is_explicitly_disabled(harness):
    spec = declaration()
    spec["initial_adapter_precision_transfer"] = None
    assert recipe.normalize_study(spec)["initial_adapter_precision_transfer"] is None
    spec["initial_adapter_precision_transfer"] = "float32-to-bfloat16"
    with pytest.raises(ValueError, match="not authorized"):
        recipe.normalize_study(spec)
    h = harness
    h.context["study"]["initial_adapter_precision_transfer"] = None
    write(h.root / "plan" / "context.json", h.context)
    invoke(h, "common")
    invoke(h, "train", "rl", "exposure")
    assert h.fake.calls[0]["initial_adapter_precision_transfer"] is None


def test_source_or_input_change_rejects_before_worker(harness):
    h = harness
    (h.root / "a.png").write_bytes(b"altered")
    with pytest.raises(ValueError, match="Frozen Aircraft input changed"):
        invoke(h, "common")
    assert not h.fake.calls and not h.fake.assays


def test_full_contract_fixture_retains_failures_and_has_no_achievement(harness):
    h = harness
    invoke(h, "common")
    invoke(h, "assay", "initial", "initial")
    for arm in ("rl", "matched"):
        invoke(h, "train", arm, "exposure")
        invoke(h, "assay", arm, "exposure")
    invoke(h, "train", "matched", "time")
    invoke(h, "assay", "matched", "time")
    invoke(h, "compare")
    report = recipe.read(h.state / "setup" / "comparison" / "published" / "comparison.json")
    assert (
        report["comparisons_vs_common_initialization"]["rl-exposure"]["candidate"][
            "invalid_answers_including_missing"
        ]
        == 1
    )
    assert report["sft_search_total_committed_active_seconds"] == 6
    assert report["warmup"]["warmup_exposure_accounting"].startswith("configuration-derived")
    assert not report["primary_requirement_satisfied"] and not report["test_consumed"]
    assert report["split"] == "train"


def test_every_declared_sft_is_retained_in_compiled_plan(harness):
    spec = declaration()
    spec["sft_controls"].extend(
        [
            {"id": "answer-low", "learning_rate": 1e-5, "response_format": "answer_only"},
            {"id": "answer-high", "learning_rate": 1e-4, "response_format": "answer_only"},
        ]
    )
    normalized = recipe.normalize_study(copy.deepcopy(spec))
    assert len(recipe._arms(normalized)) == 4
    assert normalized["sft_controls"] == spec["sft_controls"]


def test_compiled_controller_terminal_resume_does_no_work(harness, monkeypatch):
    """Execute the real ordered controller with file-only trainer contract fixtures."""
    h = harness
    calls = []

    def execute(runner, candidate, attempt, stage):
        argv = [runner.substitute(arg, candidate, attempt) for arg in candidate[stage]]
        assert argv[1:3] == ["-m", "vision_rl.aircraft_recipe"]
        calls.append(argv)
        result = recipe.main(argv[3:])
        return {"returncode": result, "reason": "exited", "seconds": 0.0}

    monkeypatch.setattr(loop._Runner, "command", execute)
    loop.initialize(h.root / "plan" / "plan.json", h.state)
    result = loop.run_loop(h.state)
    assert result["status"] == "awaiting_final_evaluation"
    assert not result["test_consumed"]
    assert all(record["status"] == "completed" for record in result["setup"].values())
    calls_before = len(calls)
    bytes_before = {name: (h.state / name).read_bytes() for name in ("state.json", "events.jsonl")}
    artifacts_before = loop._hash_path(h.state / "setup")
    loop.run_loop(h.state)
    assert len(calls) == calls_before == 11
    assert len(h.fake.calls) == 3
    assert all((h.state / name).read_bytes() == value for name, value in bytes_before.items())
    assert loop._hash_path(h.state / "setup") == artifacts_before


def test_receipt_rejects_changed_operational_stop_target(harness):
    h = harness
    invoke(h, "common")
    original = h.fake.read_endpoint_receipt

    def changed(*args, **kwargs):
        receipt = original(*args, **kwargs)
        receipt["requested_boundary"]["stop_after_steps"] = 1
        return receipt

    h.fake.read_endpoint_receipt = changed
    with pytest.raises(ValueError, match="operational stop target"):
        invoke(h, "train", "rl", "exposure")


def test_completed_assay_with_wrong_seed_cannot_be_reused(harness):
    h = harness
    invoke(h, "common")
    invoke(h, "assay", "initial", "initial")
    path = (
        h.state
        / "setup"
        / "initial-assay"
        / "work"
        / "think_then_answer-sample-42"
        / "train_subset_groups.json"
    )
    record = recipe.read(path)
    record["generation"]["seed"] = 99
    write(path, record)
    count = len(h.fake.assays)
    with pytest.raises(ValueError, match="identity differs"):
        invoke(h, "assay", "initial", "initial")
    assert len(h.fake.assays) == count


@pytest.mark.parametrize("response_format", ["answer_only", "think_then_answer"])
def test_both_prompt_styles_use_identical_optional_think_grammar(tmp_path, response_format):
    context = {
        "labels": ["red", "blue"],
        "assay_rows": [{"id": "a", "label": "red"}],
        "study": {"assay_seeds": [42]},
    }
    correct = {"response": "<think>Visible red.</think>\nred", "terminated": True}
    wrong = {"response": "<think>Visible blue.</think>\nblue", "terminated": True}
    write(tmp_path / "greedy.json", {"a": correct})
    assert recipe._canonical_predictions(context, tmp_path / "greedy.json", response_format) == {
        "a": "red"
    }
    name = response_format + "-sample-42"
    write(tmp_path / "endpoint" / name / "sampled_completions.json", {"a": [correct, *[wrong] * 3]})
    write(tmp_path / "initial" / name / "sampled_completions.json", {"a": [wrong] * 4})
    report = recipe._sample_comparison(
        context, tmp_path / "endpoint", tmp_path / "initial", response_format
    )
    assert report["completions"] == 4
    assert report["exact_accuracy"] == report["valid_pair_correctness_delta"] == 0.25
    assert report["both_valid_fraction"] == 1
    assert report["useful_mixed_groups"] == 1


def test_optional_think_does_not_relax_eos_or_exact_answer(tmp_path):
    context = {
        "labels": ["red", "blue"],
        "assay_rows": [{"id": "a", "label": "red"}, {"id": "b", "label": "blue"}],
    }
    write(
        tmp_path / "greedy.json",
        {
            "a": {"response": "<think>Red.</think>\nred", "terminated": False},
            "b": {"response": "<think>Blue.</think>\nblue extra text", "terminated": True},
        },
    )
    assert recipe._canonical_predictions(context, tmp_path / "greedy.json", "answer_only") == {
        "a": None,
        "b": None,
    }
