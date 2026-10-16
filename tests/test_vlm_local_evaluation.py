"""No-model evaluation tests: journals, clearance, native scope and endpoint roles."""

import hashlib
import json
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_rl import vlm_local_evaluation as evaluation
from vision_rl import vlm_local_training as native


def study():
    return {
        "schema_version": 1,
        "activated": True,
        "scope": evaluation.SCOPE,
        "profile": {**asdict(native.Profile()), "response_format": "answer_only"},
        "device": "cpu",
        "arms": [
            {
                "id": name,
                "adapter": "/unopened/endpoint",
                "weights_sha256": "a" * 64,
                "record_sha256": "b" * 64,
            }
            for name in evaluation.ARM_IDS
        ],
        "budget": {
            "max_wall_seconds": 86400,
            "max_invocations": 512,
            "chunk_images": 1,
            "chunk_seconds": 1800,
            "image_seconds": 120,
            "max_output_bytes": 2 * 1024**3,
        },
        "bootstrap": {"resamples": 100, "seed": 2718},
    }


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Two synthetic colored JPEGs; never mislabeled as actual native evidence."""
    root = tmp_path.resolve()
    monkeypatch.setattr(evaluation, "NATIVE_IMAGES", 2)
    monkeypatch.setattr(evaluation, "NATIVE_CLASSES", 2)
    monkeypatch.setattr(evaluation, "sources", lambda: {"fixture": "fixed-source"})
    monkeypatch.setattr(evaluation, "_deadline", lambda seconds: nullcontext())
    clearance_calls = []

    def clearance(*args):
        clearance_calls.append(args)
        return {"fixture": True, "status": "not_actual_author_clearance"}

    monkeypatch.setattr(evaluation, "_require_clearance", clearance)
    spec = {
        "schema_version": 1,
        "benchmark": "unit fixture only",
        "question": "Which color?",
        "labels": ["red", "blue"],
        "split_policy": "fixture",
        "preprocessing": "fixture",
        "preprocessing_id": evaluation.PREPROCESSING_ID,
        "sources": [{"url": "fixture://generated", "sha256": "a" * 64}],
    }
    evaluation.write(root / "spec.json", spec)
    images = []
    for identifier, color in (("a", "red"), ("b", "blue")):
        path = root / f"{identifier}.jpg"
        Image.new("RGB", (32, 48), color).save(path)
        images.append(
            {
                "id": identifier,
                "image": str(path),
                "encoded_sha256": evaluation._sha(path),
                "model_rgb_sha256": evaluation.aircraft_pixel_sha256(path),
            }
        )
    evaluation.write(root / "images.json", images)
    evaluation.write(
        root / "gold.json", [{"id": "a", "label": "red"}, {"id": "b", "label": "blue"}]
    )
    plan = {
        "schema_version": 1,
        "scope": evaluation.SCOPE,
        "study": study(),
        "paths": {
            "data_spec": str(root / "spec.json"),
            "root": "fixture-root",
            "metadata_archive": "fixture-metadata",
            "probe": "fixture-probe",
        },
        "source_sha256": evaluation.sources(),
        "input_sha256": {
            str(root / name): evaluation._sha(root / name)
            for name in ("spec.json", "images.json", "gold.json")
        },
        "runtime": evaluation.safe.RUNTIME,
        "base": {"fixture_no_model": True},
        "prompt_sha256": hashlib.sha256(
            native.prompt_text(spec, "answer_only").encode()
        ).hexdigest(),
        "clearance": clearance(),
        "labels": spec["labels"],
    }

    def persist():
        evaluation.write(root / "plan.json", plan)
        (root / "plan.sha256").write_text(evaluation._sha(root / "plan.json") + "\n")

    persist()
    calls, closed = [], []

    class FakeBackend:
        def __init__(self, plan, arm):
            self.arm = arm["id"]
            self.provenance = {
                "fixture_no_model": True,
                "environment": {"device": plan["study"]["device"]},
                "runtime": plan["runtime"],
                "adapter_sha256": arm["weights_sha256"],
                "profile": plan["study"]["profile"],
                "prompt_sha256": plan["prompt_sha256"],
                "generation_kwargs": evaluation._generation_kwargs(plan["study"]["profile"]),
                "generation_config": {"fixture_no_model": True},
                "trainable_parameters": 0,
                "prediction_fields": ["id", "image"],
            }

        def predict(self, row):
            assert set(row) == {"id", "image"}
            calls.append((self.arm, row["id"]))
            return {
                "response": "<think>red details</think>\nred"
                if row["id"] == "a"
                else "not a canonical variant",
                "raw_response": "fixture raw",
                "terminated": True,
                "completion_ids": [10, 2],
                "prompt_tokens": 12,
            }

        def close(self):
            closed.append(self.arm)

    monkeypatch.setattr(evaluation, "Backend", FakeBackend)
    return SimpleNamespace(
        root=root,
        images=images,
        plan=plan,
        persist=persist,
        calls=calls,
        closed=closed,
        clearance_calls=clearance_calls,
        backend=FakeBackend,
    )


def test_unactivated_proposal_fails_before_clearance_or_output(tmp_path, monkeypatch):
    declaration = study()
    declaration["activated"] = False
    evaluation.write(tmp_path / "study.json", declaration)
    monkeypatch.setattr(
        evaluation, "_require_clearance", lambda *a: pytest.fail("must not read data")
    )
    with pytest.raises(ValueError, match="unactivated"):
        evaluation.prepare(
            *([tmp_path / "unopened"] * 6),
            study=tmp_path / "study.json",
            output=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_actual_missing_clearance_cannot_be_replaced_by_boolean(tmp_path):
    with pytest.raises(ValueError, match="receipt required"):
        evaluation._require_clearance(tmp_path, tmp_path / "metadata", tmp_path / "probe")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(device="cuda"),
        lambda s: s["profile"].update(dtype="bfloat16"),
        lambda s: s["arms"].pop(),
        lambda s: s["arms"][1].update(weights_sha256=None),
        lambda s: s["budget"].update(image_seconds=float("nan")),
        lambda s: s["budget"].update(max_invocations=0),
        lambda s: s["bootstrap"].update(seed=True),
    ],
)
def test_invalid_prospective_contract_fails(mutation):
    value = study()
    mutation(value)
    with pytest.raises(ValueError):
        evaluation.declaration(value)


def test_bounded_resume_never_repeats_and_terminal_loads_no_model(bundle):
    h = bundle
    first = evaluation.run_chunk(h.root, "common")
    assert first["remaining"] == 1
    second = evaluation.run_chunk(h.root, "common")
    assert second["status"] == "completed"
    before = (h.root / "arms/common/predictions.jsonl").read_bytes()
    state = evaluation.read(h.root / "execution.json")
    evaluation.run_chunk(h.root, "common")
    assert h.calls == [("common", "a"), ("common", "b")]
    assert len(h.closed) == 2
    assert evaluation.read(h.root / "execution.json") == state
    assert (h.root / "arms/common/predictions.jsonl").read_bytes() == before
    assert second["metric"] is None


def test_interrupted_reservation_becomes_unknown_without_regeneration(bundle, monkeypatch):
    h = bundle
    original = h.backend.predict

    def interrupt(self, row):
        raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(h.backend, "predict", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluation.run_chunk(h.root, "common")
    monkeypatch.setattr(h.backend, "predict", original)
    report = evaluation.run_chunk(h.root, "common")
    assert h.calls == [("common", "b")]
    assert report["counts"]["unknown_after_interruption"] == 1
    assert not report["primary_requirement_satisfied"]
    events = [
        json.loads(line)
        for line in (h.root / "arms/common/predictions.jsonl").read_text().splitlines()
    ]
    assert events[1]["invocation"] == events[2]["invocation"]


def test_exception_is_retained_and_not_retried(bundle, monkeypatch):
    h = bundle
    original = h.backend.predict

    def error(self, row):
        if row["id"] == "a":
            h.calls.append((self.arm, row["id"]))
            raise RuntimeError("fixture generation failure")
        return original(self, row)

    monkeypatch.setattr(h.backend, "predict", error)
    with pytest.raises(RuntimeError, match="fixture"):
        evaluation.run_chunk(h.root, "common")
    assert h.closed == ["common"]
    result = evaluation.run_chunk(h.root, "common")
    assert result["counts"]["generation_error"] == 1
    assert h.calls == [("common", "a"), ("common", "b")]


def test_original_wall_budget_survives_resume(bundle):
    h = bundle
    evaluation.write(h.root / "execution.json", {"created_at": 0, "invocations": 1})
    with pytest.raises(ValueError, match="budget exhausted"):
        evaluation.run_chunk(h.root, "common")
    assert not h.calls


def test_max_invocations_cannot_reset_on_new_arm(bundle):
    h = bundle
    h.plan["study"]["budget"]["max_invocations"] = 1
    h.persist()
    evaluation.run_chunk(h.root, "common")
    with pytest.raises(ValueError, match="budget exhausted"):
        evaluation.run_chunk(h.root, "rl-exposure")


def test_changed_clearance_rejects_before_model(bundle, monkeypatch):
    monkeypatch.setattr(evaluation, "_require_clearance", lambda *a: {"changed": True})
    with pytest.raises(ValueError, match="clearance changed"):
        evaluation.run_chunk(bundle.root, "common")
    assert not bundle.calls


def test_changed_source_or_image_fails_before_model(bundle, monkeypatch):
    monkeypatch.setattr(evaluation, "sources", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="executed source"):
        evaluation.run_chunk(bundle.root, "common")
    assert not bundle.calls


def test_changed_native_pixels_fail_before_reservation(bundle):
    h = bundle
    Path(h.images[0]["image"]).write_bytes(b"changed encoded JPEG")
    with pytest.raises(ValueError, match="native image changed"):
        evaluation.run_chunk(h.root, "common")
    assert not h.calls
    events = (h.root / "arms/common/predictions.jsonl").read_text().splitlines()
    assert [json.loads(line)["event"] for line in events] == ["header"]


def test_concurrent_evaluator_lock_rejects(bundle):
    with evaluation._lock(bundle.root), pytest.raises(ValueError, match="already owns"):
        evaluation.run_chunk(bundle.root, "common")


def test_journal_rejects_duplicate_reordered_modified_and_torn(bundle):
    h = bundle
    evaluation.run_chunk(h.root, "common")
    arm = h.plan["study"]["arms"][0]
    journal = evaluation._journal(h.root, h.plan, arm, h.images)
    path = journal.path
    reference = json.loads(path.read_text().splitlines()[1])["invocation"]
    with pytest.raises(ValueError, match="reordered"):
        journal.append("reserved", "a", invocation=reference)
    text = path.read_text()
    path.write_text(text.replace('"id":"a"', '"id":"b"'))
    with pytest.raises(ValueError, match="changed"):
        evaluation._journal(h.root, h.plan, arm, h.images)
    path.write_text(text + '{"partial":')
    with pytest.raises(ValueError):
        evaluation._journal(h.root, h.plan, arm, h.images)


@pytest.mark.parametrize("copy_head", [False, True])
def test_common_journal_cannot_be_copied_into_other_arms(bundle, copy_head):
    h = bundle
    evaluation.run_chunk(h.root, "common")
    evaluation.run_chunk(h.root, "common")
    for name in evaluation.ARM_IDS[1:]:
        shutil.copytree(h.root / "arms/common", h.root / "arms" / name)
        if copy_head:
            shutil.copyfile(
                h.root / "journal-heads/common.json", h.root / "journal-heads" / (name + ".json")
            )
    with pytest.raises(ValueError, match="head|different frozen arm"):
        evaluation.compare(h.root)
    with pytest.raises(ValueError, match="head|different frozen arm"):
        evaluation.run_chunk(h.root, "rl-exposure")
    assert h.calls == [("common", "a"), ("common", "b")]


@pytest.mark.parametrize("mutation", ["delete", "empty", "prefix", "missing_head"])
def test_consumed_journal_cannot_be_deleted_or_rewound(bundle, mutation):
    h = bundle
    evaluation.run_chunk(h.root, "common")
    path = h.root / "arms/common/predictions.jsonl"
    if mutation == "delete":
        path.unlink()
    elif mutation == "empty":
        path.write_bytes(b"")
    elif mutation == "prefix":
        path.write_text(path.read_text().splitlines()[0] + "\n")
    else:
        (h.root / "journal-heads/common.json").unlink()
    with pytest.raises(ValueError):
        evaluation.run_chunk(h.root, "common")
    assert h.calls == [("common", "a")]


@pytest.mark.parametrize("mutation", ["weights", "missing", "metadata", "receipt"])
def test_backend_invocation_identity_is_required_on_resume_and_compare(bundle, mutation):
    h = bundle
    evaluation.run_chunk(h.root, "common")
    evaluation.run_chunk(h.root, "common")
    path = h.root / "arms/common/invocation-00001.identity.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "metadata":
        metadata = path.with_name("invocation-00001.json")
        value = evaluation.read(metadata)
        value["backend"]["adapter_sha256"] = "c" * 64
        evaluation.write(metadata, value)
    else:
        value = evaluation.read(path)
        if mutation == "weights":
            value["backend"]["adapter_sha256"] = "c" * 64
        else:
            value["identity"]["arm"]["id"] = "rl-exposure"
        evaluation.write(path, value)
    for operation in (
        lambda: evaluation.run_chunk(h.root, "common"),
        lambda: evaluation.compare(h.root),
    ):
        with pytest.raises((ValueError, OSError)):
            operation()
    assert h.calls == [("common", "a"), ("common", "b")]


def test_frozen_plan_change_cannot_reuse_existing_journal(bundle):
    evaluation.run_chunk(bundle.root, "common")
    bundle.plan["study"]["bootstrap"]["seed"] += 1
    bundle.persist()
    with pytest.raises(ValueError, match="different frozen arm/plan"):
        evaluation.run_chunk(bundle.root, "common")


def test_backend_mismatch_is_rejected_before_reservation(bundle, monkeypatch):
    original = bundle.backend.__init__

    def changed(self, plan, arm):
        original(self, plan, arm)
        self.provenance["adapter_sha256"] = "c" * 64

    monkeypatch.setattr(bundle.backend, "__init__", changed)
    with pytest.raises(ValueError, match="Backend receipt"):
        evaluation.run_chunk(bundle.root, "common")
    assert not bundle.calls


@pytest.mark.parametrize("event_kind", ["reserved", "outcome"])
def test_fsynced_event_before_head_commit_recovers_without_retry(bundle, monkeypatch, event_kind):
    h = bundle
    original = evaluation.write
    failed = False
    target_sequence = 2 if event_kind == "reserved" else 3

    def fail_head(path, value):
        nonlocal failed
        if (
            Path(path) == h.root / "journal-heads/common.json"
            and value["sequence"] == target_sequence
            and not failed
        ):
            failed = True
            raise OSError("fixture crash after journal fsync")
        return original(path, value)

    monkeypatch.setattr(evaluation, "write", fail_head)
    with pytest.raises(OSError, match="fixture crash"):
        evaluation.run_chunk(h.root, "common")
    monkeypatch.setattr(evaluation, "write", original)
    result = evaluation.run_chunk(h.root, "common")
    assert result["status"] == "completed"
    assert result["counts"]["unknown_after_interruption"] == (event_kind == "reserved")
    assert h.calls == (
        [("common", "b")] if event_kind == "reserved" else [("common", "a"), ("common", "b")]
    )


def test_outcome_cannot_change_its_reserved_invocation(bundle, monkeypatch):
    original = bundle.backend.predict
    monkeypatch.setattr(
        bundle.backend, "predict", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        evaluation.run_chunk(bundle.root, "common")
    monkeypatch.setattr(bundle.backend, "predict", original)
    journal = evaluation._journal(
        bundle.root, bundle.plan, bundle.plan["study"]["arms"][0], bundle.images
    )
    wrong = {**journal.pending_invocation, "number": 999}
    with pytest.raises((ValueError, OSError)):
        journal.append("outcome", "a", invocation=wrong, result={"status": "completed"})


def test_all_arms_native_denominator_and_invalids_retained_offline(bundle):
    h = bundle
    for name in evaluation.ARM_IDS:
        evaluation.run_chunk(h.root, name)
        evaluation.run_chunk(h.root, name)
    result = evaluation.compare(h.root)
    assert len(result["versus_common_initialization"]) == 7
    assert len(result["rl_versus_each_sft"]) == 6
    metric = result["versus_common_initialization"]["rl-exposure"]
    assert metric["candidate"]["mean_class_accuracy"] == 0.5
    assert metric["candidate"]["invalid_answers_including_missing"] == 1
    assert result["all_inference_completed_without_unknown_or_errors"]
    assert not result["primary_requirement_satisfied"] and not result["test_consumed"]


def test_incomplete_arm_cannot_disappear_from_comparison(bundle):
    evaluation.run_chunk(bundle.root, "common")
    with pytest.raises(ValueError, match="complete denominator"):
        evaluation.compare(bundle.root)


def test_native_validation_manifest_cannot_relabel_train_or_labels(bundle):
    h = bundle
    root = h.root / "dataset"
    directory = root / "development/images/validation"
    directory.mkdir(parents=True)
    authors, rows = [], []
    for item, label in zip(h.images, ["red", "blue"], strict=True):
        image = directory / (item["id"] + ".jpg")
        image.write_bytes(Path(item["image"]).read_bytes())
        authors.append(
            {
                "image_id": item["id"],
                "split": "val",
                "variant": label,
                "path": str(image.relative_to(root)),
                "encoded_sha256": item["encoded_sha256"],
                "model_rgb_sha256": item["model_rgb_sha256"],
            }
        )
        rows.append(
            {
                "id": item["id"],
                "split": "validation",
                "image": f"images/validation/{item['id']}.jpg",
                "label": label,
            }
        )
    evaluation.write(root / "audit/val-images.json", authors)
    manifest = root / "development/validation.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    images, gold = evaluation._validation_rows(root, manifest, {"labels": ["red", "blue"]})
    assert len(images) == len(gold) == 2
    rows[0]["label"] = "blue"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="author evidence"):
        evaluation._validation_rows(root, manifest, {"labels": ["red", "blue"]})
    rows[0].update(label="red", split="train")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="author evidence"):
        evaluation._validation_rows(root, manifest, {"labels": ["red", "blue"]})


def test_inference_backend_rejects_target_fields_without_model():
    backend = evaluation.Backend.__new__(evaluation.Backend)
    with pytest.raises(ValueError, match="image-only"):
        backend.predict({"id": "x", "image": "unopened", "label": "leaked"})


@pytest.fixture
def endpoints(tmp_path):
    arms = study()["arms"]
    identity = {"train": {"n": 400}, "validation": None, "profile": study()["profile"]}
    records, receipts = {}, {}
    for arm in arms:
        name = arm["id"]
        arm["adapter"] = str(tmp_path / name / "final_adapter")
        if name == "common":
            record = {
                "final_adapter_sha256": "warm",
                "settings": {"max_steps": 400, "train_only": True},
                "control_identity": identity,
            }
        else:
            rl = name == "rl-exposure"
            count = 800 if name.endswith("time") else 400
            seconds = 10 if rl or name.endswith("time") else 5
            settings = {
                "init_adapter": {"weights_sha256": "warm"},
                "mode": "grpo" if rl else "sft",
                "learning_rate": 1e-5 if rl or "1e5" in name else 5e-5 if "5e5" in name else 1e-4,
                "seed": 42,
                "train_only": True,
                "max_steps": 400 if rl else 6000,
                "max_active_seconds": 10800,
                "beta": 0.04 if rl else None,
            }
            record = {
                "settings": settings,
                "control_identity": identity,
                "environment": {"adapter_source_sha256": "fixed"},
                "reference": {"initial_adapter_weights_sha256": "warm"},
                "optimizer_audit": {
                    "reference_unchanged": True,
                    "active_time": {"committed": seconds},
                },
            }
            receipt = {
                "settings_sha256": native._object_sha256(settings),
                "control_identity_sha256": native._object_sha256(identity),
                "final_adapter": {"weights_sha256": arm["weights_sha256"]},
                "source_sha256": "fixed",
                "global_step": count,
                "committed_prompt_presentations": count,
                "committed_prompt_ids": [f"train-{i}" for i in range(400)] * (count // 400),
                "committed_active_seconds": seconds,
                "physical_work": record["optimizer_audit"]["active_time"],
                "requested_boundary": {
                    "stop_after_steps": None if name.endswith("time") else 400,
                    "stop_after_active_seconds": 10 if name.endswith("time") else None,
                },
            }
            receipts[name] = receipt
            evaluation.write(Path(arm["adapter"]).parent / "trainer-receipt.json", receipt)
        records[name] = record
        evaluation.write(Path(arm["adapter"]).parent / "local_training.json", record)
    return SimpleNamespace(arms=arms, records=records, receipts=receipts)


def test_controlled_endpoint_roles_are_verified_from_actual_receipts(endpoints):
    assert len(evaluation._controlled_endpoints(endpoints.arms)) == 7


@pytest.mark.parametrize("case", ["init", "lr", "short_time", "prefix", "reset_budget", "source"])
def test_incomplete_or_unmatched_endpoint_cannot_be_relabelled(endpoints, case):
    h = endpoints
    name = "sft-1e5-time" if case in {"short_time", "prefix", "reset_budget"} else "rl-exposure"
    record, receipt = h.records[name], h.receipts[name]
    if case == "init":
        record["settings"]["init_adapter"]["weights_sha256"] = "different"
    elif case == "lr":
        record["settings"]["learning_rate"] = 1e-3
    elif case == "short_time":
        receipt["committed_active_seconds"] = 9
        receipt["physical_work"]["committed"] = 9
    elif case == "prefix":
        receipt["committed_prompt_ids"][0] = "different image"
    elif case == "reset_budget":
        record["settings"]["max_steps"] = 10000
    else:
        receipt["source_sha256"] = "changed"
    directory = Path(next(a["adapter"] for a in h.arms if a["id"] == name)).parent
    evaluation.write(directory / "local_training.json", record)
    evaluation.write(directory / "trainer-receipt.json", receipt)
    with pytest.raises(ValueError):
        evaluation._controlled_endpoints(h.arms)


def test_real_no_model_deadline_interrupts_its_own_worker():
    script = """
import time
from vision_rl.vlm_local_evaluation import _deadline
try:
    with _deadline(.05):
        time.sleep(30)
except KeyboardInterrupt:
    print('owned worker interrupted')
else:
    raise AssertionError('deadline was ignored')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0 and "owned worker interrupted" in result.stdout
