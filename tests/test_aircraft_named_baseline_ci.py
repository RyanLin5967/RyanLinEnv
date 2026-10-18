"""Fail-closed boundaries of the sharded named-baseline CI runner. CPU only, no model."""

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = _load("aircraft_named_baseline_ci", "scripts/aircraft_named_baseline_ci.py")
export = _load("export_native_val_endpoints", "scripts/export_native_val_endpoint_predictions.py")
PLAN = ROOT / "docs/aircraft-named-baseline-ci-plan-v1.json"
RUN = {"GITHUB_RUN_ID": "77", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "abc",
       "GITHUB_REPOSITORY": "RyanLin5967/RyanLinEnv"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def activated(tmp_path):
    plan = json.loads(PLAN.read_text())
    plan.update(activated=True, status="ACTIVATED_FOR_TEST",
                activation={"activated_at": "2026-10-18T00:00:00Z", "approved_by": "test"})
    path = tmp_path / "activated.json"
    path.write_text(json.dumps(plan))
    return path, sha(path)


def record(ids, shards, index, *, identity=None, attempt="1", status="completed"):
    mine = ci.partition(ids, shards, index)
    return {
        "plan_id": ci.PLAN_ID, "plan_sha256": "p", "stage": "discovery", "shards": shards,
        "index": index, "status": status,
        "run": {"run_id": "77", "run_attempt": attempt, "sha": "abc"},
        "identity": identity or {"model": "m"},
        "ids": mine,
        "results": {i: {"status": "completed", "canonical_label": None, "terminated": False,
                        "response": "", "latency_seconds": 1.0} for i in mine},
        "timing": {"setup_seconds": 10.0, "elapsed_seconds": 20.0},
    }


IDS = [f"{n:07d}" for n in range(10)]


def verify(records, shards=3, **kwargs):
    return ci.verify_shards(records, IDS, shards, stage="discovery", plan_sha256="p",
                            run={"run_id": "77", "sha": "abc"}, **kwargs)


def test_committed_plan_is_unactivated_and_every_val_command_refuses_it():
    plan = json.loads(PLAN.read_text())
    assert plan["activated"] is False and plan["activation"] is None
    for stage in ("discovery", "full"):
        with pytest.raises(ValueError, match="unactivated"):
            ci.matrix(PLAN, sha(PLAN), stage, 4)


def test_activated_copy_needs_its_exact_sha256(activated):
    path, digest = activated
    with pytest.raises(ValueError, match="exact SHA256"):
        ci.matrix(path, "0" * 64, "discovery", 4)
    assert ci.matrix(path, digest, "discovery", 4) == {"shard": [0, 1, 2, 3]}


@pytest.mark.parametrize("shards", [0, 21, -1])
def test_discovery_shard_count_is_bounded(activated, shards):
    path, digest = activated
    with pytest.raises(ValueError, match="shard count"):
        ci.matrix(path, digest, "discovery", shards)


def test_full_val_needs_a_discovery_receipt(activated):
    path, digest = activated
    with pytest.raises(ValueError, match="discovery receipt"):
        ci.matrix(path, digest, "full", 20)


def test_pinned_inputs_and_manifests(activated):
    plan = json.loads(PLAN.read_text())
    for name in ("val_images", "validation", "data_spec"):
        ci.pinned(plan["inputs"][name])
    ci.pinned(plan["evaluator_identity"]["plan"])
    ci.pinned(plan["analysis"]["endpoint_predictions"])
    ci.pinned(plan["analysis"]["endpoint_report"])
    ci.pinned(plan["baseline_identity"]["manifest"])
    discovery = ci.stage_images(plan, "discovery")
    full = ci.stage_images(plan, "full")
    assert len(discovery) == 100 and len(full) == 3333
    assert {row["id"] for row in discovery} <= {row["id"] for row in full}
    assert all(row["path"].startswith("development/images/validation/") for row in full)
    plumbing = ci.stage_images(plan, "plumbing")
    assert [row["path"].split("/")[2] for row in plumbing] == ["train"] * 3
    assert not {row["id"] for row in plumbing} & {row["id"] for row in full}
    assert len(ci.frozen_sources(plan)) == 41


def test_model_manifest_is_the_pinned_revision():
    plan = json.loads(PLAN.read_text())
    manifest = json.loads(ci.pinned(plan["baseline_identity"]["manifest"]).read_text())
    assert manifest["model_repository"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert manifest["revision"] == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    weights = sum(e["size_bytes"] for e in manifest["files"] if e["path"].endswith(".safetensors"))
    assert weights == plan["baseline_identity"]["weights_bytes"] == 3982649232 + 3526688744


@pytest.mark.parametrize("shards", [1, 2, 3, 7, 20, 256])
def test_partition_scores_each_image_exactly_once(shards):
    ids = [f"{n:07d}" for n in range(3333)]
    parts = [ci.partition(ids, shards, index) for index in range(shards)]
    flat = [identifier for part in parts for identifier in part]
    assert sorted(flat) == ids and len(flat) == len(set(flat)) == 3333


def test_complete_shards_merge_in_author_order():
    merged, identity = verify({i: record(IDS, 3, i) for i in range(3)})
    assert list(merged) == IDS and identity == {"model": "m"}


def test_merge_refuses_a_missing_shard():
    records = {i: record(IDS, 3, i) for i in (0, 2)}
    with pytest.raises(ValueError, match=r"missing \[1\]"):
        verify(records)
    records[1] = None  # artifact present but no shard record (job killed)
    with pytest.raises(ValueError, match=r"missing \[1\]"):
        verify(records)


def test_merge_refuses_an_unexpected_shard():
    records = {i: record(IDS, 3, i) for i in range(3)}
    records[3] = record(IDS, 4, 3)
    with pytest.raises(ValueError, match=r"unexpected \[3\]"):
        verify(records)


def test_merge_refuses_duplicated_or_misplaced_images():
    records = {i: record(IDS, 3, i) for i in range(3)}
    records[1]["ids"] = records[1]["ids"] + [IDS[0]]
    with pytest.raises(ValueError, match="wrong images"):
        verify(records)


def test_merge_refuses_a_gap_inside_a_shard():
    records = {i: record(IDS, 3, i) for i in range(3)}
    records[2]["results"].pop(records[2]["ids"][-1])
    with pytest.raises(ValueError, match="gap"):
        verify(records)


@pytest.mark.parametrize("status", ["generation_error", "unknown_after_interruption"])
def test_merge_refuses_a_non_completed_outcome(status):
    records = {i: record(IDS, 3, i) for i in range(3)}
    records[0]["results"][IDS[0]]["status"] = status
    with pytest.raises(ValueError, match="non-completed"):
        verify(records)


@pytest.mark.parametrize("status", ["partial", "started", "generation_error"])
def test_merge_refuses_an_incomplete_shard(status):
    records = {i: record(IDS, 3, i) for i in range(3)}
    records[1]["status"] = status
    with pytest.raises(ValueError, match="did not complete"):
        verify(records)


def test_merge_refuses_a_rerun_shard_another_run_or_identity():
    records = {i: record(IDS, 3, i) for i in range(3)}
    records[1] = record(IDS, 3, 1, attempt="2")
    with pytest.raises(ValueError, match="re-run"):
        verify(records)
    records[1] = record(IDS, 3, 1)
    records[1]["run"]["run_id"] = "78"
    with pytest.raises(ValueError, match="another workflow run"):
        verify(records)
    records[1] = record(IDS, 3, 1, identity={"model": "other"})
    with pytest.raises(ValueError, match="different model"):
        verify(records)


def write_artifacts(root, stage, records):
    for index, value in records.items():
        directory = root / f"named-baseline-{stage}-shard-{index}"
        directory.mkdir(parents=True)
        (directory / "shard.json").write_text(json.dumps(value))


def test_merge_command_refuses_a_missing_shard_before_any_scoring(
    activated, tmp_path, monkeypatch, capsys
):
    path, digest = activated
    for key, value in RUN.items():
        monkeypatch.setenv(key, value)
    plan = json.loads(path.read_text())
    ids = [row["id"] for row in ci.stage_images(plan, "discovery")]
    records = {}
    for index in (0, 2):
        value = record(ids, 3, index)
        value["plan_sha256"] = digest
        records[index] = value
    write_artifacts(tmp_path / "artifacts", "discovery", records)
    code = ci.main(["merge", "--stage", "discovery", "--shards", "3", "--plan", str(path),
                    "--plan-sha256", digest, "--artifacts", str(tmp_path / "artifacts"),
                    "--work", str(tmp_path / "work"), "--output", str(tmp_path / "out")])
    assert code == 2
    assert "missing [1]" in capsys.readouterr().err
    assert not (tmp_path / "work").exists() and not (tmp_path / "out").exists()


def test_merge_command_refuses_no_artifacts_and_foreign_artifacts(activated, tmp_path, monkeypatch):
    path, digest = activated
    for key, value in RUN.items():
        monkeypatch.setenv(key, value)
    args = ["merge", "--stage", "discovery", "--shards", "2", "--plan", str(path),
            "--plan-sha256", digest, "--artifacts", str(tmp_path / "artifacts"),
            "--work", str(tmp_path / "work"), "--output", str(tmp_path / "out")]
    assert ci.main(args) == 2  # nothing downloaded: every shard is missing
    (tmp_path / "artifacts" / "named-baseline-full-shard-0").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unexpected artifact"):
        ci.read_shard_records(tmp_path / "artifacts", "discovery", 2)


def test_plumbing_refuses_any_plan_but_the_committed_one(activated):
    path, digest = activated
    with pytest.raises(ValueError, match="committed plan"):
        ci.require_stage_plan(path, digest, "plumbing")


def receipt(digest, **changes):
    value = {"plan_id": ci.PLAN_ID, "plan_sha256": digest, "stage": "discovery",
             "status": "completed", "counts": {"completed": 100}, "valid_fraction": 0.95,
             "timing": {"mean_image_latency_seconds": 60.0, "max_setup_seconds": 900.0}}
    value.update(changes)
    return value


def test_full_shard_rule_matches_hand_arithmetic(activated):
    path, digest = activated
    plan = json.loads(path.read_text())
    # usable = 19800 - 900 - 900 = 18000 s; 1.5 x 60 s = 90 s per image; 200 per shard.
    assert ci.required_full_shards(plan, receipt(digest)) == 17  # ceil(3333 / 200)


def test_discovery_gate_refuses_failures(activated, tmp_path, monkeypatch):
    path, digest = activated
    plan = json.loads(path.read_text())
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    cases = {
        "another plan": receipt("0" * 64),
        "not a completed": receipt(digest, status="partial"),
        "every image": receipt(digest, counts={"completed": 99}),
        "valid-answer": receipt(digest, valid_fraction=0.5),
        "at least 17 shards": receipt(digest),
    }
    for reason, value in cases.items():
        (tmp_path / "receipt.json").write_text(json.dumps(value))
        with pytest.raises(ValueError, match=reason):
            ci.discovery_gate(plan, digest, "receipt.json", 16)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt(digest)))
    ci.discovery_gate(plan, digest, "receipt.json", 17)


def test_endpoint_export_reproduces_the_committed_report():
    plan = json.loads(PLAN.read_text())
    value = json.loads(ci.pinned(plan["analysis"]["endpoint_predictions"]).read_text())
    report = json.loads(ci.pinned(plan["analysis"]["endpoint_report"]).read_text())
    labels = json.loads(ci.pinned(plan["evaluator_identity"]["plan"]).read_text())["labels"]
    gold = [{"id": r["id"], "label": r["label"]}
            for r in map(json.loads, ci.pinned(plan["inputs"]["validation"]).read_text()
                         .splitlines())]
    arms = value["arms"]
    assert set(arms) == {"common", "rl-exposure", "sft-1e5-time"}
    for arm in plan["analysis"]["reproduce_against_report"]:
        expected = report["versus_common_initialization"][arm]["input_sha256"]
        assert export.canonical_input_sha256(labels, gold, arms[arm], arms["common"]) == expected
        planted = copy.deepcopy(arms[arm])
        planted[gold[0]["id"]] = None if planted[gold[0]["id"]] else gold[0]["label"]
        assert export.canonical_input_sha256(labels, gold, planted, arms["common"]) != expected


def _has_commit(commit):
    try:
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
                       check=True, capture_output=True, timeout=30)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _has_commit("85f0865920015fd7e637d2b997bd9d7597dc1e83"),
                    reason="shallow checkout lacks the frozen evaluator commit")
def test_frozen_evaluator_materialises_byte_for_byte(tmp_path):
    plan = json.loads(PLAN.read_text())
    source = ci.materialize_evaluator(plan, tmp_path / "frozen")
    assert (source / "vision_rl/vlm_local_evaluation.py").is_file()
    written = [p for p in (tmp_path / "frozen").rglob("*.py")]
    assert len(written) == 41
    with pytest.raises(ValueError, match="Never reuse"):
        ci.materialize_evaluator(plan, tmp_path / "frozen")
