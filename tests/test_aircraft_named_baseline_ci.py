"""Fail-closed boundaries of the sharded named-baseline CI runner. CPU only, no model."""

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
       "GITHUB_REPOSITORY": "RyanLin5967/RyanLinEnv", "GITHUB_REF": "refs/heads/main"}
RUNNER = ROOT / "scripts/aircraft_named_baseline_ci.py"
WORKFLOW = ROOT / ".github/workflows/aircraft-named-baseline-ci.yml"
FROZEN_COMMIT = "85f0865920015fd7e637d2b997bd9d7597dc1e83"


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
    # Expected membership written out as i mod N, never taken from ci.partition.
    mine = [identifier for position, identifier in enumerate(ids) if position % shards == index]
    return {
        "plan_id": ci.PLAN_ID, "plan_sha256": "p", "stage": "discovery", "shards": shards,
        "index": index, "status": status,
        "run": {"run_id": "77", "run_attempt": attempt, "sha": "abc"},
        "identity": identity or {"model": "m"},
        "ids": mine,
        "results": {i: {"status": "completed", "canonical_label": None, "terminated": False,
                        "response": "", "latency_seconds": 1.0} for i in mine},
        "timing": {"setup_seconds": 10.0, "elapsed_seconds": 20.0,
                   "fixed_setup_seconds": 8.0, "image_fetch_seconds": 2.0},
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


@pytest.mark.parametrize("shards", [0, 3, 5, 21, -1])
def test_discovery_shard_count_is_the_declared_four(activated, shards):
    path, digest = activated
    with pytest.raises(ValueError, match="shard count"):
        ci.matrix(path, digest, "discovery", shards)


@pytest.mark.parametrize("field, value", [
    ("generation", {"profile": {}}),
    ("sharding", {}),
    ("amendments", ["relax the gate"]),
    ("analysis", {}),
])
def test_activated_copy_may_change_only_activation_fields(tmp_path, field, value):
    plan = json.loads(PLAN.read_text())
    plan.update(activated=True, status="ACTIVATED",
                activation={"activated_at": "2026-10-18T00:00:00Z", "approved_by": "test"})
    plan[field] = value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="changes pre-registered fields"):
        ci.load_plan(path, sha(path))


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
    for index, part in enumerate(parts):
        assert part == [ids[p] for p in range(3333) if p % shards == index]


def test_partition_is_interleaved_author_order():
    assert ci.partition(IDS, 3, 0) == ["0000000", "0000003", "0000006", "0000009"]
    assert ci.partition(IDS, 3, 1) == ["0000001", "0000004", "0000007"]
    assert ci.partition(IDS, 3, 2) == ["0000002", "0000005", "0000008"]


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
    for index in (0, 1, 3):
        value = record(ids, 4, index)
        value["plan_sha256"] = digest
        records[index] = value
    write_artifacts(tmp_path / "artifacts", "discovery", records)
    code = ci.main(["merge", "--stage", "discovery", "--shards", "4", "--plan", str(path),
                    "--plan-sha256", digest, "--artifacts", str(tmp_path / "artifacts"),
                    "--work", str(tmp_path / "work"), "--output", str(tmp_path / "out")])
    assert code == 2
    assert "missing [2]" in capsys.readouterr().err
    assert not (tmp_path / "work").exists() and not (tmp_path / "out").exists()


def test_merge_command_refuses_no_artifacts_and_foreign_artifacts(
    activated, tmp_path, monkeypatch, capsys
):
    path, digest = activated
    for key, value in RUN.items():
        monkeypatch.setenv(key, value)
    args = ["merge", "--stage", "discovery", "--shards", "4", "--plan", str(path),
            "--plan-sha256", digest, "--artifacts", str(tmp_path / "artifacts"),
            "--work", str(tmp_path / "work"), "--output", str(tmp_path / "out")]
    assert ci.main(args) == 2  # nothing downloaded: every shard is missing
    assert "missing [0, 1, 2, 3]" in capsys.readouterr().err
    (tmp_path / "artifacts" / "named-baseline-full-shard-0").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unexpected artifact"):
        ci.read_shard_records(tmp_path / "artifacts", "discovery", 2)


def test_plumbing_refuses_any_plan_but_the_committed_one(activated):
    path, digest = activated
    with pytest.raises(ValueError, match="committed plan"):
        ci.require_stage_plan(path, digest, "plumbing")


# The discovery sample's compact-JSON ID hash, as pre-registered (c8d73a4c...).
DISCOVERY_IDS_SHA256 = "c8d73a4cd59107e65df631a8b5af476b9916aa1e0ec1a6bf0642741b06963865"


def receipt(digest, **changes):
    run = {"run_id": "90", "run_attempt": "1", "sha": "def", "ref": "refs/heads/main",
           "repository": "RyanLin5967/RyanLinEnv"}
    value = {"plan_id": ci.PLAN_ID, "plan_sha256": digest, "stage": "discovery",
             "status": "completed", "ids_sha256": DISCOVERY_IDS_SHA256,
             "counts": {"completed": 100, "denominator": 100}, "valid_fraction": 0.95,
             "run": run,
             "identity": {"plan_sha256": digest, "stage": "discovery", "run_id": "90",
                          "sha": "def", "runner_sha256": sha(RUNNER),
                          "workflow_sha256": sha(WORKFLOW)},
             "timing": {"mean_image_latency_seconds": 40.0,
                        "max_shard_mean_image_latency_seconds": 55.0,
                        "mean_image_fetch_seconds": 5.0, "max_fixed_setup_seconds": 900.0}}
    value.update(changes)
    return value


def test_full_shard_rule_matches_hand_arithmetic(activated):
    path, digest = activated
    plan = json.loads(path.read_text())
    # usable = 19800 - 900 - 900 = 18000 s. The SLOWEST shard's mean (55 s, not the 40 s
    # overall mean) + 5 s fetch, x 3.0 = 180 s, below the 330 s floor: 330 s per image,
    # floor(18000 / 330) = 54 per shard.
    assert ci.required_full_shards(plan, receipt(digest)) == 62  # ceil(3333 / 54)
    slow = receipt(digest)
    slow["timing"]["max_shard_mean_image_latency_seconds"] = 115.0
    # 3.0 x (115 + 5) = 360 s > 330 s; floor(18000 / 360) = 50 per shard.
    assert ci.required_full_shards(plan, slow) == 67  # ceil(3333 / 50)


def test_shard_rule_survives_the_measured_runner_spread(activated):
    """Plumbing measured 113.07 s and 297.80 s per image on two free runners. Shards sized
    from an all-fast discovery must still fit the start limit on the slow host."""
    path, digest = activated
    plan = json.loads(path.read_text())
    fast = receipt(digest)
    fast["timing"].update(max_shard_mean_image_latency_seconds=113.07,
                          mean_image_fetch_seconds=1.07, max_fixed_setup_seconds=65.4)
    shards = ci.required_full_shards(plan, fast)
    assert shards == 61  # 18834.6 / (3.0 x 114.14) = 55.0 -> 55 per shard
    per_shard = -(-3333 // shards)  # busiest shard: 55 images
    assert 65.4 + per_shard * (297.80 + 1.07) <= 19800 - 900
    # A discovery that lands only on very fast hosts (9.4 s measured) is held by the floor.
    fast["timing"]["max_shard_mean_image_latency_seconds"] = 9.4
    shards = ci.required_full_shards(plan, fast)
    assert shards == 59  # floor(18834.6 / 330) = 57 per shard
    assert 65.4 + -(-3333 // shards) * (297.80 + 1.07) <= 19800 - 900


class FakeTorch:
    def __init__(self, capability):
        self.backends = SimpleNamespace(
            cpu=SimpleNamespace(get_cpu_capability=lambda: capability),
            mkldnn=SimpleNamespace(enabled=True))


def test_cpu_kernel_path_is_pinned_and_checked(monkeypatch):
    plan = json.loads(PLAN.read_text())
    torch = FakeTorch("AVX2")
    assert ci.check_cpu_numerics(plan, torch) == {"aten_cpu_capability": "AVX2",
                                                  "mkldnn_enabled": False}
    assert torch.backends.mkldnn.enabled is False
    with pytest.raises(ValueError, match="not the pre-registered"):
        ci.check_cpu_numerics(plan, FakeTorch("AVX512"))
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(ValueError, match="imported before"):
        ci.pin_cpu_numerics(plan)
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    ci.pin_cpu_numerics(plan)
    assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
    monkeypatch.delenv("ATEN_CPU_CAPABILITY")


@pytest.mark.parametrize("responses, ok", [
    (["DC-8", "Global Express", "DC-8"], True),
    (["DC-8", "DC-8", "DC-8"], False),  # what the AVX-512 bf16 host answered
])
def test_plumbing_canary_requires_the_preregistered_answers(tmp_path, responses, ok):
    ids = ["0450014", "0056978", "2223757"]

    def shard(*args):
        (tmp_path / "out").mkdir(exist_ok=True)
        return {"ids": ids, "host": {"cpu_model": "x"}, "status": "completed",
                "results": {i: {"response": r} for i, r in zip(ids, responses, strict=True)}}

    if ok:
        assert ci.plumbing(tmp_path / "w", tmp_path / "out", shard=shard)["canary"]["match"]
    else:
        with pytest.raises(ValueError, match="differ from the pre-registered"):
            ci.plumbing(tmp_path / "w", tmp_path / "out", shard=shard)
    written = json.loads((tmp_path / "out" / "shard.json").read_text())
    assert written["canary"]["match"] is ok


def gate_cases(digest):
    other_code = receipt(digest)
    other_code["identity"]["runner_sha256"] = "0" * 64
    other_identity = receipt(digest)
    other_identity["identity"]["run_id"] = "91"
    rerun = receipt(digest)
    rerun["run"]["run_attempt"] = "2"
    branch = receipt(digest)
    branch["run"]["ref"] = "refs/heads/feature"
    no_timing = receipt(digest)
    del no_timing["timing"]["max_shard_mean_image_latency_seconds"]
    return {
        "another plan": receipt("0" * 64),
        "not a completed": receipt(digest, status="partial"),
        "other images": receipt(digest, ids_sha256="0" * 64),
        "every image": receipt(digest, counts={"completed": 99, "denominator": 100}),
        "first-attempt run on main": rerun,
        "run on main": branch,
        "identity does not match": other_identity,
        "other runner or workflow code": other_code,
        "valid-answer": receipt(digest, valid_fraction=0.5),
        "not a number": receipt(digest, valid_fraction=None),
        "timing": no_timing,
        "at least 62 shards": receipt(digest),
    }


def test_discovery_gate_refuses_failures(activated, tmp_path, monkeypatch):
    path, digest = activated
    plan = json.loads(path.read_text())
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    for reason, value in gate_cases(digest).items():
        (tmp_path / "receipt.json").write_text(json.dumps(value))
        with pytest.raises(ValueError, match=reason):
            ci.discovery_gate(plan, digest, "receipt.json", 61)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt(digest)))
    ci.discovery_gate(plan, digest, "receipt.json", 62)


def test_matrix_command_refuses_a_malformed_receipt_with_code_2(
    activated, tmp_path, monkeypatch, capsys
):
    path, digest = activated
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    for value in (receipt(digest, valid_fraction=None), ["not", "an", "object"],
                  receipt(digest, counts=None, run="x", identity=7)):
        (tmp_path / "receipt.json").write_text(json.dumps(value))
        code = ci.main(["matrix", "--stage", "full", "--shards", "40", "--plan", str(path),
                        "--plan-sha256", digest, "--discovery-receipt", "receipt.json"])
        assert code == 2
        assert "discovery gate" in capsys.readouterr().err


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


def test_frozen_evaluator_materialises_byte_for_byte(tmp_path):
    if not _has_commit("85f0865920015fd7e637d2b997bd9d7597dc1e83"):
        if os.environ.get("REQUIRE_FROZEN_COMMIT") == "1":
            pytest.fail("The frozen evaluator commit is required here but absent")
        pytest.skip("shallow checkout lacks the frozen evaluator commit; the named-baseline "
                    "workflow's tests job runs this with full history")
    plan = json.loads(PLAN.read_text())
    source = ci.materialize_evaluator(plan, tmp_path / "frozen")
    assert (source / "vision_rl/vlm_local_evaluation.py").is_file()
    written = [p for p in (tmp_path / "frozen").rglob("*.py")]
    assert len(written) == 41
    with pytest.raises(ValueError, match="Never reuse"):
        ci.materialize_evaluator(plan, tmp_path / "frozen")


# ------------------------------------------------------------ run identity and host facts


def test_val_stages_refuse_any_ref_but_main(monkeypatch):
    for key, value in RUN.items():
        monkeypatch.setenv(key, value)
    assert ci.run_identity("discovery")["ref"] == "refs/heads/main"
    monkeypatch.setenv("GITHUB_REF", "refs/heads/edited-plan-and-runner")
    for stage in ("discovery", "full"):
        with pytest.raises(ValueError, match="only from refs/heads/main"):
            ci.run_identity(stage)
    monkeypatch.setenv("GITHUB_REF", "refs/pull/27/merge")
    assert ci.run_identity("plumbing")["ref"] == "refs/pull/27/merge"
    monkeypatch.delenv("GITHUB_REF")
    with pytest.raises(ValueError, match="inside a GitHub Actions run"):
        ci.run_identity("plumbing")


def test_host_facts_record_the_cpu_model_and_flags(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: AMD EPYC 7763 64-Core Processor\n"
                       "flags\t\t: sse avx2 fma\n\nprocessor\t: 1\nmodel name\t: other\n")
    assert ci.cpu_facts(cpuinfo) == {"cpu_model": "AMD EPYC 7763 64-Core Processor",
                                     "cpu_flags": ["avx2", "fma", "sse"]}
    assert ci.cpu_facts(tmp_path / "absent") == {"cpu_model": None, "cpu_flags": None}


# ------------------------------------------------- run_shard fail-closed, no model, no net


LABELS = ["A320", "DC-8", "Global Express"]


class FakeBackend:
    def __init__(self, behaviour, clock):
        self.spec = {"labels": LABELS}
        self.behaviour, self.clock, self.calls = behaviour, clock, 0

    def predict(self, row):
        self.calls += 1
        return self.behaviour(self.calls, self.clock)


def answer(text="DC-8", ids=2):
    return {"response": text, "raw_response": text, "terminated": True,
            "completion_ids": list(range(ids)), "prompt_tokens": 5}


def run_fake_shard(tmp_path, monkeypatch, behaviour, *, pixels="px"):
    for key, value in RUN.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GITHUB_REF", "refs/pull/27/merge")
    clock = {"t": 0.0}
    backend = FakeBackend(behaviour, clock)

    def fetch_images(images, root):
        for row in images:
            image = Path(root) / f"{row['id']}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(row["id"].encode())
            row.update(image=str(image), encoded_sha256=sha(image), model_rgb_sha256="px")

    frozen = {
        "aircraft_data": SimpleNamespace(aircraft_pixel_sha256=lambda path: pixels),
        "vlm_local_training": SimpleNamespace(
            parse_label=lambda text, labels, allow_reasoning: text if text in labels else None),
        "vlm_local_evaluation": SimpleNamespace(_deadline=lambda s: contextlib.nullcontext()),
    }
    monkeypatch.setattr(ci, "materialize_evaluator", lambda plan, where: where)
    monkeypatch.setattr(ci, "load_frozen", lambda plan, root: frozen)
    monkeypatch.setattr(ci, "fetch_images", fetch_images)
    monkeypatch.setattr(ci, "fetch_model", lambda plan, where: None)
    monkeypatch.setattr(ci, "load_backend", lambda plan, fz, where: (backend, {"model": "m"}))
    output = tmp_path / "out"
    try:
        result = ci.run_shard(PLAN, sha(PLAN), "plumbing", 1, 0, tmp_path / "work", output,
                              clock=lambda: clock["t"])
    except (ValueError, RuntimeError) as exc:
        result = exc
    return result, json.loads((output / "shard.json").read_text()), backend


def test_run_shard_completes_three_images(tmp_path, monkeypatch):
    result, record, backend = run_fake_shard(tmp_path, monkeypatch, lambda n, c: answer())
    assert isinstance(result, dict) and record["status"] == "completed"
    assert list(record["results"]) == ["0450014", "0056978", "2223757"]
    assert {r["canonical_label"] for r in record["results"].values()} == {"DC-8"}
    assert record["host"]["cpu_count"] == os.cpu_count() and "cpu_model" in record["host"]
    assert record["identity"]["runner_sha256"] == sha(RUNNER)
    assert record["identity"]["workflow_sha256"] == sha(WORKFLOW)


def test_run_shard_generation_error_stops_and_fails(tmp_path, monkeypatch):
    def behaviour(n, clock):
        if n == 2:
            raise RuntimeError("out of memory")
        return answer()
    result, record, backend = run_fake_shard(tmp_path, monkeypatch, behaviour)
    assert isinstance(result, ValueError) and "ended generation_error" in str(result)
    assert record["status"] == "generation_error" and backend.calls == 2
    assert record["results"]["0056978"]["error"] == "RuntimeError: out of memory"


@pytest.mark.parametrize("bad", [
    {"response": 7}, {"terminated": 1}, {"completion_ids": list(range(257))},
    {"response": "x" * 65537},
])
def test_run_shard_malformed_output_is_a_generation_error(tmp_path, monkeypatch, bad):
    result, record, _ = run_fake_shard(tmp_path, monkeypatch, lambda n, c: {**answer(), **bad})
    assert record["status"] == "generation_error"
    assert "Malformed or oversized" in record["results"]["0450014"]["error"]
    assert isinstance(result, ValueError)


def test_run_shard_interruption_is_unknown_and_never_regenerated(tmp_path, monkeypatch):
    def behaviour(n, clock):
        raise KeyboardInterrupt  # what the frozen _deadline raises on its SIGALRM
    result, record, backend = run_fake_shard(tmp_path, monkeypatch, behaviour)
    assert record["status"] == "unknown_after_interruption" and backend.calls == 1
    assert record["results"] == {"0450014": {"status": "unknown_after_interruption",
                                             "canonical_label": None}}
    assert isinstance(result, ValueError)


def test_run_shard_budget_stop_is_partial(tmp_path, monkeypatch):
    def behaviour(n, clock):
        clock["t"] += 18_901.0  # 18,901 + 900 > 19,800: the next image may not start
        return answer()
    result, record, backend = run_fake_shard(tmp_path, monkeypatch, behaviour)
    assert record["status"] == "partial" and backend.calls == 1
    assert list(record["results"]) == ["0450014"]
    assert isinstance(result, ValueError) and "ended partial" in str(result)


def test_run_shard_escaped_exception_leaves_failed(tmp_path, monkeypatch):
    result, record, backend = run_fake_shard(tmp_path, monkeypatch, lambda n, c: answer(),
                                             pixels="changed")
    assert isinstance(result, ValueError) and "Model-visible image changed" in str(result)
    assert record["status"] == "failed" and backend.calls == 0 and record["results"] == {}


# --------------------------------------- frozen-code paths (tests job only: full history)


def _require_frozen():
    if os.environ.get("REQUIRE_FROZEN_COMMIT") != "1":
        pytest.skip("imports the frozen evaluator; runs in the named-baseline workflow's "
                    "tests job (REQUIRE_FROZEN_COMMIT=1), never on the study Mac")
    if not _has_commit(FROZEN_COMMIT):
        pytest.fail("The frozen evaluator commit is required here but absent")


def _merge_in_subprocess(tmp_path, plan_path, digest, stage, records):
    """Each merge imports the frozen package once, so each runs in its own interpreter."""
    write_artifacts(tmp_path / "artifacts", stage, records)
    env = {**os.environ, **RUN}
    completed = subprocess.run(
        [sys.executable, "-B", str(RUNNER), "merge", "--stage", stage, "--shards",
         str(len(records)), "--plan", str(plan_path), "--plan-sha256", digest,
         "--artifacts", str(tmp_path / "artifacts"), "--work", str(tmp_path / "work"),
         "--output", str(tmp_path / "out")],
        env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    out = tmp_path / "out"
    return (json.loads((out / f"{stage}-receipt.json").read_text()),
            json.loads((out / f"{stage}-predictions.json").read_text()))


def _shard_records(ids, shards, digest, stage, outcome):
    records = {}
    for index in range(shards):
        value = record(ids, shards, index)
        value.update(plan_sha256=digest, stage=stage, host={"cpu_model": f"cpu-{index}"})
        value["timing"] = {"setup_seconds": 60.0 + index, "fixed_setup_seconds": 50.0 + index,
                           "image_fetch_seconds": 0.5 * len(value["ids"]),
                           "elapsed_seconds": 1000.0}
        value["results"] = {i: {"status": "completed", "latency_seconds": 10.0 * (index + 1),
                                **outcome(i)} for i in value["ids"]}
        records[index] = value
    return records


def test_discovery_merge_success_path_with_the_frozen_code(activated, tmp_path):
    _require_frozen()
    path, digest = activated
    plan = json.loads(path.read_text())
    ids = [row["id"] for row in ci.stage_images(plan, "discovery")]
    gold = {r["id"]: r["label"] for r in ci.gold_rows(plan)}
    labels = json.loads(ci.pinned(plan["evaluator_identity"]["plan"]).read_text())["labels"]
    wrong = {label: labels[(n + 1) % len(labels)] for n, label in enumerate(labels)}

    def outcome(identifier):
        position = ids.index(identifier)
        if position < 37:  # correct
            text = gold[identifier]
        elif position < 50:  # valid but wrong
            text = wrong[gold[identifier]]
        else:  # unterminated: never parsed
            return {"response": "...", "raw_response": "...", "terminated": False,
                    "canonical_label": None}
        return {"response": text, "raw_response": text + "<|im_end|>", "terminated": True,
                "canonical_label": text}

    receipt_value, predictions = _merge_in_subprocess(
        tmp_path, path, digest, "discovery", _shard_records(ids, 4, digest, "discovery", outcome))
    assert receipt_value["status"] == "completed"
    assert receipt_value["ids_sha256"] == DISCOVERY_IDS_SHA256
    assert receipt_value["counts"] == {"completed": 100, "denominator": 100}
    assert receipt_value["valid_fraction"] == 0.5 and receipt_value["unterminated"] == 50
    assert receipt_value["metric"]["correct"] == 37
    assert receipt_value["metric"]["mean_class_accuracy"] == pytest.approx(0.37)
    timing = receipt_value["timing"]
    assert timing["mean_image_latency_seconds"] == pytest.approx(25.0)  # 25 each at 10..40
    assert timing["max_shard_mean_image_latency_seconds"] == 40.0
    assert timing["min_shard_mean_image_latency_seconds"] == 10.0
    assert timing["max_fixed_setup_seconds"] == 53.0 and timing["max_setup_seconds"] == 63.0
    assert timing["mean_image_fetch_seconds"] == pytest.approx(0.5)
    assert receipt_value["hosts"]["3"] == {"cpu_model": "cpu-3",
                                           "mean_image_latency_seconds": 40.0}
    assert receipt_value["hoeffding"]["half_width"] == pytest.approx(0.13581015157406195)
    assert receipt_value["run"]["ref"] == "refs/heads/main"
    assert sum(v is not None for v in predictions["predictions"].values()) == 50
    first = ids[0]
    assert predictions["raw"][first]["raw_response"] == gold[first] + "<|im_end|>"


def test_full_merge_success_path_compares_against_the_endpoints(activated, tmp_path):
    _require_frozen()
    path, digest = activated
    plan = json.loads(path.read_text())
    ids = [row["id"] for row in ci.stage_images(plan, "full")]
    arms = json.loads(ci.pinned(plan["analysis"]["endpoint_predictions"]).read_text())["arms"]
    report = json.loads(ci.pinned(plan["analysis"]["endpoint_report"]).read_text())

    def outcome(identifier):  # the candidate replays W's canonical outcomes
        label = arms["common"][identifier]
        if label is None:
            return {"response": "", "raw_response": "", "terminated": False,
                    "canonical_label": None}
        return {"response": label, "raw_response": label, "terminated": True,
                "canonical_label": label}

    receipt_value, _ = _merge_in_subprocess(
        tmp_path, path, digest, "full", _shard_records(ids, 4, digest, "full", outcome))
    comparisons = receipt_value["comparisons"]
    assert set(comparisons) == {"W", "RL400", "strongest_sft"}
    assert comparisons["W"]["mean_class_accuracy_difference"] == 0.0
    for role, arm in (("RL400", "rl-exposure"), ("strongest_sft", "sft-1e5-time")):
        committed = report["versus_common_initialization"][arm]
        assert (comparisons[role]["mean_class_accuracy_difference"]
                == -committed["mean_class_accuracy_difference"])
        assert (comparisons[role]["candidate"]["mean_class_accuracy"]
                == committed["comparator"]["mean_class_accuracy"])


EXPORT_CHECK = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("ci", sys.argv[1])
ci = importlib.util.module_from_spec(spec); spec.loader.exec_module(ci)
plan = json.loads(ci.PLAN.read_text())
frozen = ci.load_frozen(plan, ci.materialize_evaluator(plan, Path(sys.argv[2])))
cases = json.loads(Path(sys.argv[3]).read_text())
out = [frozen["classification_comparison"].compare_classification(
    cases["gold"], case, cases["common"], cases["labels"], resamples=100, seed=1)["input_sha256"]
    for case in cases["candidates"]]
print(json.dumps(out))
"""


def test_exporter_hash_equals_the_frozen_input_sha256(tmp_path):
    """The stdlib exporter hash must equal the frozen code's, on the export and a flip."""
    _require_frozen()
    plan = json.loads(PLAN.read_text())
    arms = json.loads(ci.pinned(plan["analysis"]["endpoint_predictions"]).read_text())["arms"]
    labels = json.loads(ci.pinned(plan["evaluator_identity"]["plan"]).read_text())["labels"]
    gold = ci.gold_rows(plan)
    planted = copy.deepcopy(arms["rl-exposure"])
    planted[gold[0]["id"]] = None if planted[gold[0]["id"]] else gold[0]["label"]
    candidates = [arms["rl-exposure"], arms["sft-1e5-time"], planted]
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"gold": gold, "labels": labels, "common": arms["common"],
                                 "candidates": candidates}))
    completed = subprocess.run(
        [sys.executable, "-B", "-c", EXPORT_CHECK, str(RUNNER), str(tmp_path / "frozen"),
         str(cases)], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
    frozen_hashes = json.loads(completed.stdout)
    mine = [export.canonical_input_sha256(labels, gold, c, arms["common"]) for c in candidates]
    assert frozen_hashes == mine
    assert frozen_hashes[2] != frozen_hashes[0]
