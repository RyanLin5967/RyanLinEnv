"""CPU-only tests for the named-baseline v2 runner (engine option, shards, merge).

The model is a fake backend; no torch is imported. Expected values are derived from the
fixture by hand, never by calling the runner.
"""

import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest
from PIL import Image

from vision_rl import local_vlm
from vision_rl.aircraft_data import aircraft_pixel_sha256

REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY / "docs/aircraft-named-baseline-plan-v2.json"
V1_PLAN = REPOSITORY / "docs/aircraft-named-baseline-plan-v1.json"
INSTRUCTION = (
    "Return one label copied exactly from the allowed list on its own line. "
    "Do not add a prefix, quotes, explanations, or any text after that label."
)
PER_CLASS = 3
LOAD_SECONDS = 2.0


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local_hf = load_script("aircraft_named_baseline_hf")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_list_sha(ids):
    return hashlib.sha256(("[" + ",".join(f'"{i}"' for i in ids) + "]").encode()).hexdigest()


def independent_discovery(author):
    best = {}
    for row in author:
        text = ('["fgvc-aircraft-2013b","fgvc-aircraft-freshpool-screened-grpo-v1",'
                f'"baseline-discovery",791,"{row["variant"]}","{row["image_id"]}"]')
        key = hashlib.sha256(text.encode()).hexdigest()
        if row["variant"] not in best or key < best[row["variant"]][0]:
            best[row["variant"]] = (key, row["image_id"])
    chosen = {value[1] for value in best.values()}
    return [row["image_id"] for row in author if row["image_id"] in chosen]


def latency_of(identifier):
    return float(3 + int(identifier) % 11)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def runner(monkeypatch, clock):
    module = load_script("aircraft_named_baseline_v2")
    fake_time = types.SimpleNamespace(monotonic=clock.monotonic, time=time.time)
    monkeypatch.setattr(module, "time", fake_time)
    monkeypatch.setattr(local_vlm, "time", fake_time)
    monkeypatch.setattr(module.v1, "NATIVE_IMAGES", 100 * PER_CLASS)
    monkeypatch.setattr(module.local_hf, "runtime_versions", lambda pins, lock=None: dict(pins))
    # The kernel pin imports torch; the fake backend never computes, so record a fixed path.
    monkeypatch.setattr(module.local_hf, "pin_cpu_numerics",
                        lambda numerics: {"aten_cpu_capability": "FIXTURE"})
    monkeypatch.setattr(module.local_hf, "kernel_fingerprint", lambda threads: "FIXTURE-FP")
    # Tests move the runner between allowlisted fixture hosts by editing fixture_host.
    module.fixture_host = dict(HOSTS[0])
    monkeypatch.setattr(module.local_hf, "host_identity", lambda: dict(module.fixture_host))
    return module


HOSTS = [{"hostname": "fixture-a", "cpu_model": "Fixture CPU A"},
         {"hostname": "fixture-b", "cpu_model": "Fixture CPU B"},
         {"hostname": "fixture-c", "cpu_model": "Fixture CPU C"}]


def write_snapshot(root):
    """Fake author snapshot plus a Hub-style record: LFS sha256 and git blob SHA-1."""
    model = root / "weights"
    model.mkdir()
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "preprocessor_config.json", "generation_config.json",
             "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    files = []
    for name in names:
        data = name.encode()
        (model / name).write_bytes(data)
        entry = {"path": name, "bytes": len(data),
                 "git_blob_id": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
        if name.endswith(".safetensors") or name == "tokenizer.json":
            entry["sha256"] = hashlib.sha256(data).hexdigest()
        files.append(entry)
    index = json.dumps({"weight_map": {"a": names[-2], "b": names[-1]}}).encode()
    (model / "model.safetensors.index.json").write_bytes(index)
    files.append({"path": "model.safetensors.index.json", "bytes": len(index),
                  "git_blob_id": hashlib.sha1(b"blob %d\0" % len(index) + index).hexdigest()})
    return model, {"model_repository": "Qwen/Qwen3.8-27B",
                   "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "files": files}


@pytest.fixture
def world(tmp_path):
    repo = tmp_path / "repo"
    dataset = repo / "data/fgvc-aircraft-2013b"
    images = dataset / "development/images/validation"
    images.mkdir(parents=True)
    (dataset / "development/images/train").mkdir()
    (dataset / "audit").mkdir()
    labels = [f"Variant {index:03}" for index in range(100)]
    author, published = [], []
    for number in range(100 * PER_CLASS):
        label = labels[(number * 7) % 100]
        identifier = f"{1000000 + number:07}"
        image = images / f"{identifier}.jpg"
        Image.new("RGB", (24, 48), (number % 256, number // 256, 9)).save(image)
        author.append({
            "image_id": identifier, "variant": label, "split": "val",
            "path": f"development/images/validation/{identifier}.jpg",
            "encoded_sha256": sha(image), "model_rgb_sha256": aircraft_pixel_sha256(image),
        })
        published.append({"id": identifier, "image": f"images/validation/{identifier}.jpg",
                          "label": label, "split": "validation"})
    (dataset / "audit/val-images.json").write_text(json.dumps(author))
    (dataset / "development/validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in published))
    spec_path = dataset / "candidates/fourshot-seed791/data_spec.json"
    spec_path.parent.mkdir(parents=True)
    question = "Identify the exact aircraft variant shown in this image."
    spec_path.write_text(json.dumps({
        "schema_version": 1, "benchmark": "fixture", "question": question, "labels": labels,
        "split_policy": "native", "preprocessing": "bottom20",
        "preprocessing_id": "fgvc-aircraft-rgb-remove-bottom20-v1",
        "sources": [{"url": "https://example.test/x", "sha256": "1" * 64}],
    }))
    prompt = (question + "\nAllowed labels (the complete class list):\n" + "\n".join(labels)
              + "\n" + INSTRUCTION)
    weights, record = write_snapshot(tmp_path)
    plan = json.loads(PLAN.read_text())
    integrity = repo / plan["model"]["integrity"]
    integrity.parent.mkdir(parents=True)
    integrity.write_text(json.dumps(record))
    plan["model"]["integrity_sha256"] = sha(integrity)
    plan["activated"] = True
    plan["activation"] = {"activated_at": "2026-10-18T00:00:00Z", "approved_by": "test"}
    for name, path in (("val_images", dataset / "audit/val-images.json"),
                       ("validation", dataset / "development/validation.jsonl"),
                       ("data_spec", spec_path)):
        plan["inputs"][name]["sha256"] = sha(path)
    plan["prompt_and_parser"]["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    discovery = independent_discovery(author)
    plan["stages"]["discovery"]["ids"] = discovery
    plan["stages"]["discovery"]["ids_sha256"] = id_list_sha(discovery)
    plan["stages"]["full"]["ids_sha256"] = id_list_sha([row["image_id"] for row in author])
    plan["stages"]["full"]["images"] = 100 * PER_CLASS
    plan["stages"]["discovery"]["budget"] = {
        "chunk_images": 20, "image_seconds": 120, "load_allowance_seconds": 600,
        "long_generation_allowance_seconds": 2100, "max_invocations": 3,
        "max_active_seconds": 26100, "max_calendar_seconds": 604800,
        "max_output_bytes": 1 << 30}
    plan["stages"]["full"]["budget"] = {
        "chunk_images": 50, "image_seconds": 10, "load_allowance_seconds": 10,
        "long_generation_allowance_seconds": 10, "max_invocations": 4,
        "max_calendar_seconds": 604800, "max_output_bytes": 1 << 30}
    plan["stages"]["full"]["gate"]["max_projected_shard_active_seconds"] = 172800
    plan["stages"]["full"]["gate"]["max_shard_active_seconds"] = 345600
    plan["sharding"]["shards"] = {"discovery": 3, "full": 3}
    plan["sharding"]["hosts"] = {host["hostname"]: host["cpu_model"] for host in HOSTS}
    freeze = repo / plan["model"]["runtime_lock"]["freeze"]
    freeze.write_text("fixture-package==1.0\n")
    plan["model"]["runtime_lock"]["freeze_sha256"] = sha(freeze)
    plan["sources_pinned"] = {name: sha(REPOSITORY / name) for name in plan["sources_pinned"]}
    gold = {row["image_id"]: row["variant"] for row in author}
    return {"repo": repo, "plan": plan, "gold": gold, "author": author, "labels": labels,
            "discovery": discovery, "out": tmp_path / "out", "tmp": tmp_path,
            "weights": weights, "record": record, "integrity": integrity}


def write_plan(world, plan=None):
    path = world["tmp"] / f"activated-{len(list(world['tmp'].glob('activated-*')))}.json"
    path.write_text(json.dumps(plan or world["plan"]))
    return path, sha(path)


def fake_backend(answers, calls, *, finish="stop", fail=None, clock=None, seen=None, scale=1):
    class Backend:
        load_info = {"load_seconds": LOAD_SECONDS, "device": {"fake": True}}

        def __init__(self, directory, config):
            assert config["profile"] == "instruct" and config["enable_thinking"] is False
            assert config["engine"]["dtype"] == "float32"
            if seen is not None:
                seen.append(config["engine"])
            if clock is not None:
                clock.now += LOAD_SECONDS

        def infer(self, row, seed):
            calls.append(row["id"])
            if clock is not None:
                clock.now += scale * latency_of(row["id"])
            if fail is not None and row["id"] == fail[0]:
                raise fail[1]
            return {"native_response": {
                "text": answers[row["id"]], "finish_reason": finish, "prompt_tokens": 11,
                "generation_tokens": 3, "total_tokens": 14, "prompt_tps": None,
                "generation_tps": 1.0, "peak_memory": 0.1,
            }, "preprocessing": {}, "peak_memory_bytes": 1}

        def close(self):
            pass

    return Backend


EVALUATOR_ARMS = {"W": "common", "RL400": "rl-exposure", "strongest_SFT": "sft-1e5-time"}


def exported_arm(world, name, predictions, **override):
    """An export-arm file, written by hand from the arm schema."""
    return {"schema": "named-baseline-v2-arm-predictions-v1", "plan_arm": name,
            "evaluator_arm": EVALUATOR_ARMS[name],
            "comparison_json_sha256":
                world["plan"]["comparison"]["arm_source"]["comparison_json_sha256"],
            "prediction_journal_sha256": "0" * 64, "predictions": predictions, **override}


def answers_for(world, wrong=()):
    labels = world["labels"]
    return {i: (labels[(labels.index(g) + 1) % 100] if i in wrong else g)
            for i, g in world["gold"].items()}


def run_all(runner, world, stage, answers, calls, clock, path, digest, *, shards=range(3),
            scale=1):
    report = None
    for shard in shards:
        for _ in range(4):
            report = runner.run_shard(path, digest, stage, shard, world["repo"], world["out"],
                                      world["weights"],
                                      backend_factory=fake_backend(answers, calls, clock=clock,
                                                                   scale=scale))
            journal = world["out"] / f"stage-{stage}/shard-{shard}-of-3/run/attempts.jsonl"
            done = sum('"completed"' in line for line in journal.read_text().splitlines())
            assigned = len(world["discovery" if stage == "discovery" else "gold"]) // 3
            if done >= assigned:
                break
    return report


def test_committed_plan_is_unactivated_and_refused(runner, tmp_path):
    plan = json.loads(PLAN.read_text())
    assert plan["activated"] is False and plan["activation"] is None
    assert "UNACTIVATED" in plan["status"]
    with pytest.raises(ValueError, match="unactivated"):
        runner.load_plan(PLAN, sha(PLAN))
    assert runner.main(["run", "--plan", str(PLAN), "--expected-plan-sha256", sha(PLAN),
                        "--stage", "discovery", "--shard", "0", "--weights", str(tmp_path),
                        "--output", str(tmp_path / "never")]) == 2
    assert not (tmp_path / "never").exists()


def test_activated_plan_with_unfilled_source_pin_is_refused(runner, world):
    plan = json.loads(json.dumps(world["plan"]))
    plan["sources_pinned"]["scripts/aircraft_named_baseline_hf.py"] = None
    path, digest = write_plan(world, plan)
    with pytest.raises(ValueError, match="pin every runner source"):
        runner.load_plan(path, digest)


def test_changed_pinned_source_is_refused_before_any_output(runner, world):
    plan = json.loads(json.dumps(world["plan"]))
    plan["sources_pinned"]["scripts/aircraft_named_baseline_hf.py"] = "0" * 64
    path, digest = write_plan(world, plan)
    with pytest.raises(ValueError, match="pinned runner source changed"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend({}, []))
    assert not world["out"].exists()


def test_committed_plan_pins_match_this_tree_and_v1(runner):
    plan, v1 = json.loads(PLAN.read_text()), json.loads(V1_PLAN.read_text())
    for name, digest in plan["sources_pinned"].items():
        if digest is not None:
            assert sha(REPOSITORY / name) == digest, name
    # v1's paused run hashes these files; v2 must leave them byte-identical.
    for name, digest in v1["sources_pinned"].items():
        assert sha(REPOSITORY / name) == digest, name
    assert plan["model"]["integrity_sha256"] == sha(REPOSITORY / plan["model"]["integrity"])
    record = json.loads((REPOSITORY / plan["model"]["integrity"]).read_text())
    assert record["revision"] == plan["baseline_identity"]["revision"]
    assert len(record["files"]) == 32
    assert sum(f["path"].endswith(".safetensors") for f in record["files"]) == 18
    assert plan["stages"]["discovery"]["ids"] == v1["stages"]["discovery"]["ids"]
    assert plan["stages"]["full"]["ids_sha256"] == v1["stages"]["full"]["ids_sha256"]
    assert plan["generation"]["sampling"] == local_vlm.INSTRUCT_GENERATION
    assert plan["prompt_and_parser"]["prompt_sha256"] == v1["prompt_and_parser"]["prompt_sha256"]
    assert plan["sharding"]["shards"] == {"discovery": 10, "full": 60}
    assert plan["comparison"]["resamples"] == 20000 and plan["comparison"]["seed"] == 2718
    assert [a["name"] for a in plan["comparison"]["arms"]] == ["W", "RL400", "strongest_SFT"]


def test_shards_partition_the_manifest_by_position(runner):
    ids = [f"{n:03}" for n in range(10)]
    parts = [runner.shard_ids(ids, k, 3) for k in range(3)]
    assert parts == [["000", "003", "006", "009"], ["001", "004", "007"],
                     ["002", "005", "008"]]
    with pytest.raises(ValueError, match="outside"):
        runner.shard_ids(ids, 3, 3)


def test_sharded_discovery_merges_and_scores(runner, world, clock):
    path, digest = write_plan(world)
    discovery = world["discovery"]
    wrong = set(discovery[:13])
    answers = answers_for(world, wrong)
    answers[discovery[20]] = "Variant 020 because of the tail"
    calls, seen = [], []
    partial = runner.run_shard(path, digest, "discovery", 1, world["repo"], world["out"],
                               world["weights"],
                               backend_factory=fake_backend(answers, calls, clock=clock,
                                                            seen=seen))
    assert seen[0]["threads"] == 32  # fixed by the plan, part of the run identity
    assert calls == discovery[1::3][:20] and partial["status"] == "partial"
    assert partial["metric"] is None
    report = run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    assert sorted(calls) == sorted(discovery) and len(calls) == 100
    merged = runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)
    assert merged["status"] == "completed" == report["status"]
    assert merged["metric"]["correct"] == 100 - 13 - 1
    assert merged["metric"]["mean_class_accuracy"] == pytest.approx(0.86)
    assert merged["metric"]["image_accuracy"] == pytest.approx(0.86)
    assert merged["invalid_or_unterminated_answers"] == 1
    assert [s["assigned"] for s in merged["per_shard"]] == [34, 33, 33]
    assert merged["predictions"][discovery[0]] != world["gold"][discovery[0]]
    decision = runner.discovery_decision(world["plan"], merged)
    assert decision["decision"] == "proceed", decision
    # Per shard: 100 full images (300 / 3) in 2 chunks of 50; the fixture's full wall cap is
    # 10 + 50 x 10 + 10 = 520 s; overhead per invocation is the fake 2 s load.
    latency = sum(latency_of(i) for i in discovery) / 100
    projection = 100 * latency + 2 * LOAD_SECONDS
    assert decision["projected_shard_active_seconds"] == pytest.approx(projection)
    assert decision["shard_max_active_seconds"] == pytest.approx(2 * projection + 520)
    tight = json.loads(json.dumps(world["plan"]))
    tight["stages"]["full"]["gate"]["max_projected_shard_active_seconds"] = 10
    stopped = runner.discovery_decision(tight, merged)
    assert stopped["decision"] == "stop" and "unaffordable" in stopped["reasons"][0]


def test_merge_refuses_missing_shard(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest, shards=[0, 2])
    report = runner.merged_report(world["plan"], "discovery", world["repo"], world["out"])
    assert report["status"] == "partial" and report["missing_images"] == 33
    with pytest.raises(ValueError, match="Merge refused: 33 missing"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)
    with pytest.raises(ValueError, match="discovery gate"):
        runner.run_shard(path, digest, "full", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))


def test_merge_refuses_duplicate_or_foreign_images(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    stage = world["out"] / "stage-discovery"
    donor = (stage / "shard-0-of-3/run/attempts.jsonl").read_text().splitlines()[:2]
    victim = stage / "shard-1-of-3/run/attempts.jsonl"
    original = victim.read_text()
    victim.write_text(original + "\n".join(donor) + "\n")
    with pytest.raises(ValueError, match="outside its shard"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)
    victim.write_text(original)
    runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)
    (stage / "shard-0-of-2").mkdir()
    with pytest.raises(ValueError, match="Unexpected shard directory"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)


def test_merge_refuses_shards_with_different_identity(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    run_json = world["out"] / "stage-discovery/shard-2-of-3/run/run.json"
    recorded = json.loads(run_json.read_text())
    recorded["identity"]["config"]["engine"]["dtype"] = "bfloat16"
    run_json.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="different run identity"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)


def test_full_runs_after_discovery_and_compares_with_declared_arms(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world, wrong=set(list(world["gold"])[:30])), []
    with pytest.raises(ValueError, match="discovery shard 0 never ran"):
        runner.run_shard(path, digest, "full", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    calls.clear()
    run_all(runner, world, "full", answers, calls, clock, path, digest)
    assert sorted(calls) == sorted(world["gold"])
    merged = runner.merge(world["plan"], "full", world["repo"], world["out"], digest)
    assert merged["metric"]["correct"] == 270
    perfect = {i: g for i, g in world["gold"].items()}
    arms = {}
    for name in ("W", "RL400", "strongest_SFT"):
        arms[name] = world["tmp"] / f"{name}.json"
        arms[name].write_text(json.dumps(exported_arm(world, name, perfect)))
    result = runner.compare(world["plan"], world["repo"], world["out"],
                            {k: str(v) for k, v in arms.items()}, digest)
    image = result["against"]["W"]["image"]
    assert image["image_accuracy_difference"] == pytest.approx(-30 / 300)
    assert image["comparator_image_accuracy"] == 1.0 and image["draws"] == 20000
    # 30 of 300 paired images lose: the resampled loss count is Binomial(300, 0.1), so the
    # 95% interval of the difference sits well inside (-0.15, -0.05) and excludes zero.
    low, high = image["ci95"]
    assert -0.15 < low < -0.1 < high < -0.05
    equal = result["against"]["RL400"]["equal_class"]
    assert equal["comparator"]["mean_class_accuracy"] == 1.0
    assert equal["uncertainty"]["seed"] == 2718
    with pytest.raises(ValueError, match="declared arms"):
        runner.compare(world["plan"], world["repo"], world["out"], {"W": str(arms["W"])},
                       digest)
    short = world["tmp"] / "short.json"
    short.write_text(json.dumps(exported_arm(world, "W", dict(list(perfect.items())[:-1]))))
    with pytest.raises(ValueError, match="does not cover"):
        runner.compare(world["plan"], world["repo"], world["out"],
                       {**{k: str(v) for k, v in arms.items()}, "W": str(short)}, digest)


def test_source_change_after_first_invocation_is_refused(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                     world["weights"],
                     backend_factory=fake_backend(answers, calls, clock=clock))
    ledger = world["out"] / "stage-discovery/shard-0-of-3/ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    events[0]["sources"]["scripts/aircraft_named_baseline_hf.py"] = "0" * 64
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(ValueError, match="changed since this shard's first invocation"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))


def test_snapshot_verification_refuses_changed_or_linked_files(world):
    integrity = world["integrity"]
    local_hf.verify_snapshot(world["weights"], integrity)
    shard = world["weights"] / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"X" * len(shard.read_bytes()))
    with pytest.raises(ValueError, match="integrity mismatch"):
        local_hf.verify_snapshot(world["weights"], integrity)
    shard.write_bytes(b"model-00001-of-00002.safetensors")
    config = world["weights"] / "config.json"
    config.write_bytes(b"CONFIG.json")
    with pytest.raises(ValueError, match="integrity mismatch: config.json"):
        local_hf.verify_snapshot(world["weights"], integrity)
    config.unlink()
    os.symlink(world["weights"] / "tokenizer.json", config)
    with pytest.raises(ValueError, match="escapes"):
        local_hf.verify_snapshot(world["weights"], integrity)
    record = dict(world["record"], revision="0" * 40)
    integrity.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not the pinned"):
        local_hf.verify_snapshot(world["weights"], integrity)


def test_engine_journal_refuses_non_validation_phase(world, tmp_path):
    with pytest.raises(ValueError, match="native validation only"):
        local_hf.run_local_hf([], tmp_path / "x", model_directory=world["weights"],
                              integrity=world["integrity"], engine={}, runtime_pins={},
                              phase="train_probe", image_root=tmp_path, task_contract={},
                              preprocessing_id=None)
    assert not (tmp_path / "x").exists()


def test_engine_identity_needs_dtype_and_thread_count(world, tmp_path):
    identifier = world["author"][0]["image_id"]
    root = world["repo"] / "data/fgvc-aircraft-2013b/development/images/validation"
    rows = [{"id": identifier, "image": str(root / f"{identifier}.jpg"), "split": "validation",
             "prompt": "x"}]
    for engine in ({"name": "transformers-cpu", "dtype": "float32"},
                   {"name": "transformers-cpu", "dtype": "float32", "threads": "32"},
                   {"name": "transformers-cpu", "dtype": "float16", "threads": 32},
                   {"name": "transformers-cpu", "threads": 32}):
        with pytest.raises(ValueError, match="explicit dtype and thread count"):
            local_hf.run_local_hf(rows, tmp_path / "x", model_directory=world["weights"],
                                  integrity=world["integrity"], engine=engine, runtime_pins={},
                                  phase="validation", image_root=root,
                                  task_contract={"task": "classification"},
                                  preprocessing_id=None)
    assert not (tmp_path / "x").exists()


def test_smoke_reads_train_images_only(runner, world):
    val_id = world["author"][0]["image_id"]
    with pytest.raises(OSError):
        runner.smoke(world["plan"], world["repo"], world["weights"], [val_id], 1, "float32",
                     backend_factory=fake_backend({}, []))


def test_unpinned_integrity_record_is_refused(runner, world):
    record = json.loads(world["integrity"].read_text())
    record["files"][0]["bytes"] += 0  # same content, different serialization
    world["integrity"].write_text(json.dumps(record, indent=1))
    path, digest = write_plan(world)
    with pytest.raises(ValueError, match="integrity record differs"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend({}, []))
    with pytest.raises(ValueError, match="integrity record differs"):
        runner.smoke(world["plan"], world["repo"], world["weights"], ["x"], 1, "float32",
                     backend_factory=fake_backend({}, []))


def test_repeated_image_within_a_shard_journal_is_refused(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    journal = world["out"] / "stage-discovery/shard-1-of-3/run/attempts.jsonl"
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines + lines[:2]) + "\n")
    with pytest.raises(ValueError, match="Duplicate reserved event"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)


def test_shards_under_different_plans_are_refused(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    ledger = world["out"] / "stage-discovery/shard-2-of-3/ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    events[0]["plan_sha256"] = "f" * 64
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    # Without a given plan the shards must still agree with each other ...
    with pytest.raises(ValueError, match="different plan or runner source"):
        runner.merged_report(world["plan"], "discovery", world["repo"], world["out"])
    # ... and merge also binds every shard to the activated plan it is given.
    with pytest.raises(ValueError, match="plan other than the one given"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)


def test_merge_command_output_does_not_block_the_full_stage(runner, world, clock, capsys):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    assert runner.main(["merge", "--plan", str(path), "--stage", "discovery",
                        "--expected-plan-sha256", digest, "--output", str(world["out"]),
                        "--repository", str(world["repo"])]) == 0
    merged = json.loads((world["out"] / "merged-discovery-predictions.json").read_text())
    assert len(merged) == 100
    assert runner.main(["report", "--plan", str(path), "--stage", "discovery",
                        "--output", str(world["out"]),
                        "--repository", str(world["repo"])]) == 0
    report = runner.run_shard(path, digest, "full", 0, world["repo"], world["out"],
                              world["weights"],
                              backend_factory=fake_backend(answers, calls, clock=clock))
    assert report["per_shard"][0]["completed"] == 50


def test_committed_plan_pins_every_hashed_source(runner, world):
    plan = json.loads(PLAN.read_text())
    assert set(plan["sources_pinned"]) == set(runner.source_hashes())
    partial = json.loads(json.dumps(world["plan"]))
    partial["sources_pinned"].pop("src/vision_rl/frontier.py")
    path, digest = write_plan(world, partial)
    with pytest.raises(ValueError, match="name exactly the runner's hashed sources"):
        runner.load_plan(path, digest)


def test_engine_requires_prompt_pin_and_numerics(world, tmp_path):
    identifier = world["author"][0]["image_id"]
    root = world["repo"] / "data/fgvc-aircraft-2013b/development/images/validation"
    rows = [{"id": identifier, "image": str(root / f"{identifier}.jpg"), "split": "validation",
             "prompt": "x"}]
    numerics = {"aten_cpu_capability_env": "default"}
    for engine in ({"name": "transformers-cpu", "dtype": "float32", "threads": 32,
                    "cpu_numerics": numerics},
                   {"name": "transformers-cpu", "dtype": "float32", "threads": 32,
                    "formatted_prompt_sha256": "a" * 64}):
        with pytest.raises(ValueError, match="formatted-prompt SHA256 and CPU numerics"):
            local_hf.run_local_hf(rows, tmp_path / "x", model_directory=world["weights"],
                                  integrity=world["integrity"], engine=engine, runtime_pins={},
                                  phase="validation", image_root=root,
                                  task_contract={"task": "classification"},
                                  preprocessing_id=None)
    assert not (tmp_path / "x").exists()


def test_hard_stop_blocks_merge(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    # The last image of shard 1 fails: every image is then journaled, none is missing or
    # unknown, and only the hard stop stands between the shard and a "complete" merge.
    victim = world["discovery"][1::3][-1]
    run_all(runner, world, "discovery", answers, calls, clock, path, digest, shards=[0, 2])
    for _ in range(2):
        runner.run_shard(path, digest, "discovery", 1, world["repo"], world["out"],
                         world["weights"],
                         backend_factory=fake_backend(answers, calls, clock=clock,
                                                      fail=(victim, RuntimeError("boom"))))
    report = runner.merged_report(world["plan"], "discovery", world["repo"], world["out"])
    assert report["hard_stop_errors"] == [victim] and report["status"] == "partial"
    assert report["missing_images"] == 0 and not report["unknown_after_interruption"]
    with pytest.raises(ValueError, match="0 missing, 0 unknown and 1 hard-stop"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)


def test_merge_refuses_a_plan_other_than_the_one_run(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    with pytest.raises(ValueError, match="plan other than the one given"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], "e" * 64)
    assert runner.main(["merge", "--plan", str(PLAN), "--stage", "discovery",
                        "--expected-plan-sha256", sha(PLAN), "--output", str(world["out"]),
                        "--repository", str(world["repo"])]) == 2


def test_hidden_files_in_a_stage_directory_are_ignored(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    (world["out"] / "stage-discovery/.DS_Store").write_text("x")
    (world["out"] / "stage-discovery/.shard-0-of-3.tmp123").mkdir()
    assert runner.merge(world["plan"], "discovery", world["repo"], world["out"],
                        digest)["status"] == "completed"


def test_run_directory_left_without_run_json_is_started_fresh(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run = world["out"] / "stage-discovery/shard-0-of-3/run"
    run.mkdir(parents=True)
    (run / ".run.lock").write_text("")
    report = runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                              world["weights"],
                              backend_factory=fake_backend(answers, calls, clock=clock))
    assert report["per_shard"][0]["completed"] == 20  # one discovery chunk
    stale = world["out"] / "stage-discovery/shard-1-of-3/run"
    stale.mkdir(parents=True)
    (stale / "attempts.jsonl").write_text("")
    with pytest.raises(ValueError, match="must be empty"):
        runner.run_shard(path, digest, "discovery", 1, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))


def test_sampling_settings_the_engine_lacks_are_refused():
    base = {"max_tokens": 16384, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.0, "enable_thinking": False}
    local_hf.check_generation(base)
    with pytest.raises(ValueError, match="repetition_penalty"):
        local_hf.check_generation(dict(base, repetition_penalty=1.05))
    with pytest.raises(ValueError, match="frequency_penalty"):
        local_hf.check_generation(dict(base, frequency_penalty=0.5))


def test_kernel_pin_refuses_an_already_loaded_torch_or_mkl(monkeypatch):
    numerics = {"aten_cpu_capability_env": "default", "mkl_cbwr": "AVX"}
    monkeypatch.delenv("MKL_CBWR", raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    with pytest.raises(ValueError, match="loaded before the CPU kernel path"):
        local_hf.pin_cpu_numerics(numerics)
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(local_hf, "_mapped_libraries",
                        lambda: {"/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so"})
    with pytest.raises(ValueError, match="loaded before the CPU kernel path"):
        local_hf.pin_cpu_numerics(numerics)


def test_budget_semantics_match_the_stage_budgets():
    plan = json.loads(PLAN.read_text())
    walls = []
    for stage in ("discovery", "full"):
        b = plan["stages"][stage]["budget"]
        walls.append(b["load_allowance_seconds"] + b["chunk_images"] * b["image_seconds"]
                     + b["long_generation_allowance_seconds"])
    text = plan["budget_semantics"]["invocation_wall_seconds"]
    assert f"discovery {walls[0]} s, full {walls[1]} s" in text
    assert walls == [1338520, 1478920]
    assert plan["engine"]["threads"] == 32


# --- Guards added after the third review; each is made to fire on purpose. ---------------


def test_gate_projects_full_shards_at_the_slowest_discovery_host(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    # Shards 0 and 1 run on host A at the fixture latency, shard 2 on host B at 3x.
    run_all(runner, world, "discovery", answers, calls, clock, path, digest, shards=[0, 1])
    runner.fixture_host = dict(HOSTS[1])
    run_all(runner, world, "discovery", answers, calls, clock, path, digest, shards=[2],
            scale=3)
    report = runner.merged_report(world["plan"], "discovery", world["repo"], world["out"])
    discovery = world["discovery"]
    slow = discovery[2::3]
    slow_mean = 3 * sum(latency_of(i) for i in slow) / len(slow)
    pooled = (sum(latency_of(i) for i in discovery) + 2 * sum(latency_of(i) for i in slow)) / 100
    assert [s["host"]["hostname"] for s in report["per_shard"]] == [
        "fixture-a", "fixture-a", "fixture-b"]
    assert report["per_host"]["fixture-b"]["mean_image_latency_seconds"] == pytest.approx(
        slow_mean)
    assert report["mean_image_latency_seconds"] == pytest.approx(pooled)
    assert pooled < slow_mean
    decision = runner.discovery_decision(world["plan"], report)
    assert decision["decision"] == "proceed"
    assert decision["slowest_host"]["hostname"] == "fixture-b"
    # A full shard is 300 / 3 = 100 images in 2 chunks of 50; overhead is the 2 s fake load.
    projection = 100 * slow_mean + 2 * LOAD_SECONDS
    assert decision["projected_shard_active_seconds"] == pytest.approx(projection)
    assert decision["shard_max_active_seconds"] == pytest.approx(2 * projection + 520)
    budget = runner.stage_budget(world["plan"], "full", world["repo"], world["out"], digest,
                                 HOSTS[1])
    assert budget["max_active_seconds"] == pytest.approx(2 * projection + 520)
    # Host C is allowlisted but ran no discovery image: the cap does not cover it.
    runner.fixture_host = dict(HOSTS[2])
    with pytest.raises(ValueError, match="this host ran no discovery images"):
        runner.run_shard(path, digest, "full", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))
    assert not (world["out"] / "stage-full").exists()
    runner.fixture_host = dict(HOSTS[1])
    report = runner.run_shard(path, digest, "full", 0, world["repo"], world["out"],
                              world["weights"],
                              backend_factory=fake_backend(answers, calls, clock=clock))
    assert report["per_shard"][0]["completed"] == 50


def test_host_off_the_allowlist_is_refused_before_any_output(runner, world):
    path, digest = write_plan(world)
    for host in ({"hostname": "fixture-z", "cpu_model": "Fixture CPU A"},
                 {"hostname": "fixture-a", "cpu_model": "Fixture CPU B"}):
        runner.fixture_host = host
        with pytest.raises(ValueError, match="not on the plan's shard-host allowlist"):
            runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                             world["weights"], backend_factory=fake_backend({}, []))
    assert not world["out"].exists()


def test_a_shard_never_changes_host(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                     world["weights"], backend_factory=fake_backend(answers, calls, clock=clock))
    run_json = world["out"] / "stage-discovery/shard-0-of-3/run/run.json"
    assert json.loads(run_json.read_text())["identity"]["config"]["host"] == HOSTS[0]
    runner.fixture_host = dict(HOSTS[1])
    ledger = world["out"] / "stage-discovery/shard-0-of-3/ledger.jsonl"
    before = ledger.read_text()
    with pytest.raises(ValueError, match=r"recorded run identity in \['host'\]"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))
    assert len(calls) == 20  # nothing ran on the second host
    assert ledger.read_text() == before  # and no invocation was charged


def test_merge_refuses_a_shard_whose_kernels_computed_differently(runner, world, clock,
                                                                   monkeypatch):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest, shards=[0, 1])
    monkeypatch.setattr(runner.local_hf, "kernel_fingerprint", lambda threads: "OTHER-FP")
    # The shard runs (its own identity is consistent), but its post-run summary already
    # reads the stage's other shards and refuses; so does the merge.
    with pytest.raises(ValueError, match="Shard 2 ran under a different run identity"):
        runner.run_shard(path, digest, "discovery", 2, world["repo"], world["out"],
                         world["weights"],
                         backend_factory=fake_backend(answers, calls, clock=clock))
    assert '"completed"' in (world["out"] / "stage-discovery/shard-2-of-3/run/attempts.jsonl"
                             ).read_text()
    with pytest.raises(ValueError, match="Shard 2 ran under a different run identity"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)


def test_merge_refuses_a_shard_without_its_ledger_or_identity(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    shard = world["out"] / "stage-discovery/shard-1-of-3"
    ledger = (shard / "ledger.jsonl").read_text()
    (shard / "ledger.jsonl").unlink()  # as if only run/ had been fetched from its host
    with pytest.raises(ValueError, match="Shard 1 has journaled images but no budget ledger"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)
    with pytest.raises(ValueError, match="no budget ledger"):
        runner.merged_report(world["plan"], "discovery", world["repo"], world["out"])
    (shard / "ledger.jsonl").write_text(ledger)
    identity = (shard / "run/run.json").read_text()
    (shard / "run/run.json").unlink()
    with pytest.raises(ValueError, match="no budget ledger or run identity"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)
    (shard / "run/run.json").write_text(identity)
    assert runner.merge(world["plan"], "discovery", world["repo"], world["out"],
                        digest)["status"] == "completed"


def test_compare_refuses_arms_without_pinned_provenance_or_outside_the_vocabulary(
        runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    run_all(runner, world, "full", answers, calls, clock, path, digest)
    perfect = dict(world["gold"])
    good = {}
    for name in EVALUATOR_ARMS:
        good[name] = world["tmp"] / f"good-{name}.json"
        good[name].write_text(json.dumps(exported_arm(world, name, perfect)))
    runner.compare(world["plan"], world["repo"], world["out"],
                   {k: str(v) for k, v in good.items()}, digest)
    unknown_label = dict(perfect, **{next(iter(perfect)): "Not A Variant"})
    cases = [
        ("strongest_SFT", exported_arm(world, "strongest_SFT", perfect,
                                       evaluator_arm="sft-1e4-time"), "not export-arm output"),
        ("W", exported_arm(world, "W", perfect, comparison_json_sha256="f" * 64),
         "not export-arm output"),
        ("RL400", exported_arm(world, "W", perfect), "not export-arm output"),
        ("W", perfect, "not export-arm output"),
        ("RL400", exported_arm(world, "RL400", unknown_label), "outside the label vocabulary"),
    ]
    for name, content, message in cases:
        bad = world["tmp"] / "bad.json"
        bad.write_text(json.dumps(content))
        with pytest.raises(ValueError, match=message):
            runner.compare(world["plan"], world["repo"], world["out"],
                           {**{k: str(v) for k, v in good.items()}, name: str(bad)}, digest)


def write_bundle(world):
    """Minimal evaluator bundle: plan.json, one arm journal file and comparison.json."""
    bundle = world["tmp"] / "bundle"
    journal = bundle / "arms/sft-1e5-time/predictions.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"fixture": 1}\n')
    (bundle / "arms/common").mkdir()
    (bundle / "arms/common/predictions.jsonl").write_text('{"fixture": 0}\n')
    (bundle / "plan.json").write_text(json.dumps(
        {"labels": world["labels"],
         "study": {"arms": [{"id": "common"}, {"id": "sft-1e5-time"}]}}))
    comparison = {"plan_sha256": sha(bundle / "plan.json"),
                  "strongest_declared_sft": "sft-1e5-time",
                  "prediction_journal_sha256": {"common": "0" * 64,
                                                "sft-1e5-time": sha(journal)}}
    return bundle, comparison


def pin_comparison(world, bundle, comparison):
    (bundle / "comparison.json").write_text(json.dumps(comparison))
    plan = json.loads(json.dumps(world["plan"]))
    plan["comparison"]["arm_source"]["comparison_json_sha256"] = sha(bundle / "comparison.json")
    return plan


def test_export_arm_refuses_an_unpinned_comparison_or_a_different_arm(runner, world):
    bundle, comparison = write_bundle(world)
    plan = pin_comparison(world, bundle, comparison)
    with pytest.raises(ValueError, match="not one of the declared arms"):
        runner.export_arm(plan, bundle, "sft-1e5-time")
    with pytest.raises(ValueError, match="comparison.json differs from the plan's pinned"):
        runner.export_arm(world["plan"], bundle, "strongest_SFT")
    other = pin_comparison(world, bundle,
                           dict(comparison, strongest_declared_sft="sft-1e4-time"))
    with pytest.raises(ValueError, match="strongest declared SFT is sft-1e4-time"):
        runner.export_arm(other, bundle, "strongest_SFT")
    other = pin_comparison(world, bundle, dict(comparison, plan_sha256="e" * 64))
    with pytest.raises(ValueError, match="plan.json is not the plan"):
        runner.export_arm(other, bundle, "strongest_SFT")
    plan = pin_comparison(world, bundle, comparison)
    with pytest.raises(ValueError, match="common journal differs"):
        runner.export_arm(plan, bundle, "W")
    (bundle / "arms/sft-1e5-time/predictions.jsonl").write_text('{"fixture": 2}\n')
    with pytest.raises(ValueError, match="sft-1e5-time journal differs"):
        runner.export_arm(plan, bundle, "strongest_SFT")


def test_plan_consistency_rules_refuse_a_bad_plan(runner, world):
    runner.load_plan(PLAN, require_activated=False)  # the committed plan satisfies them
    full = world["plan"]["stages"]["full"]
    wall = runner.invocation_wall_seconds(full["budget"])
    cases = []
    plan = json.loads(json.dumps(world["plan"]))
    plan["stages"]["full"]["gate"]["max_shard_active_seconds"] = (
        full["gate"]["max_projected_shard_active_seconds"] + wall - 1)
    cases.append((plan, "gate is inconsistent"))
    plan = json.loads(json.dumps(world["plan"]))
    plan["comparison"]["arms"][2]["evaluator_arm"] = "sft-1e4-time"
    cases.append((plan, "comparison arms must be exactly"))
    plan = json.loads(json.dumps(world["plan"]))
    plan["sharding"]["hosts"] = {}
    cases.append((plan, "allowlist shard hosts"))
    plan = json.loads(json.dumps(world["plan"]))
    plan["sharding"]["shards"] = {"discovery": 3}
    cases.append((plan, "shard count for full"))
    for plan, message in cases:
        path, digest = write_plan(world, plan)
        with pytest.raises(ValueError, match=message):
            runner.load_plan(path, digest)


def test_report_command_never_scores(runner, world, clock, capsys):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    assert runner.merge(world["plan"], "discovery", world["repo"], world["out"],
                        digest)["metric"] is not None
    capsys.readouterr()
    assert runner.main(["report", "--plan", str(path), "--stage", "discovery",
                        "--output", str(world["out"]), "--repository", str(world["repo"])]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "completed" and printed["metric"] is None
    assert "predictions" not in printed and printed["decision"]["decision"] == "proceed"


def test_snapshot_refuses_files_the_record_does_not_list(world):
    weights, integrity = world["weights"], world["integrity"]
    hub = weights / ".cache/huggingface"
    (hub / "download").mkdir(parents=True)
    (hub / "trees").mkdir()
    for name in ("CACHEDIR.TAG", ".gitignore", "download/config.json.metadata",
                 "download/model-00001-of-00002.safetensors.lock",
                 "trees/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.json"):
        (hub / name).write_text("bookkeeping")
    local_hf.verify_snapshot(weights, integrity)  # hf download's own bookkeeping only
    for extra, message in (
            ("processor_config.json", "does not list: processor_config.json"),
            ("sub/added_tokens.json", "does not list: sub/added_tokens.json"),
            (".cache/huggingface/download/evil.json.metadata", "does not list"),
            (".cache/huggingface/trees/" + "0" * 40 + ".json", "does not list"),
            (".cache/huggingface/other.txt", "does not list")):
        path = weights / extra
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        with pytest.raises(ValueError, match=message):
            local_hf.verify_snapshot(weights, integrity)
        path.unlink()
    os.symlink(weights / "config.json", weights / "linked.json")
    with pytest.raises(ValueError, match="escapes the pinned snapshot: linked.json"):
        local_hf.verify_snapshot(weights, integrity)


def test_runtime_lock_refuses_any_other_environment(monkeypatch):
    import importlib.metadata as metadata

    pins = {name: "1.0" for name in local_hf.PACKAGES}
    monkeypatch.setattr(metadata, "version", lambda name: pins[name])
    installed = {"torch": "1.0", "huggingface-hub": "2.2.0"}
    monkeypatch.setattr(local_hf, "installed_distributions", lambda: dict(installed))
    monkeypatch.setattr(local_hf.platform, "python_version", lambda: "3.12.15")
    lock = {"freeze": "torch==1.0\nhuggingface_hub==2.2.0\n", "python": "3.12.15"}
    assert local_hf.runtime_versions(pins, lock) == pins
    with pytest.raises(ValueError, match="runtime lock .* is required"):
        local_hf.runtime_versions(pins, None)
    for freeze in ("torch==1.0\nhuggingface_hub==2.3.0\n", "torch==1.0\n",
                   "torch==1.0\nhuggingface_hub==2.2.0\nextra==1\n"):
        with pytest.raises(ValueError, match="differs from the runtime lock"):
            local_hf.runtime_versions(pins, dict(lock, freeze=freeze))
    with pytest.raises(ValueError, match="not the locked 3.12.14"):
        local_hf.runtime_versions(pins, dict(lock, python="3.12.14"))
    for freeze in ("torch>=1.0\n", "torch==1.0\nTorch==1.0\n", ""):
        with pytest.raises(ValueError, match="Runtime lock"):
            local_hf.parse_freeze(freeze)


def test_changed_runtime_lock_file_is_refused_before_any_output(runner, world):
    path, digest = write_plan(world)
    freeze = world["repo"] / world["plan"]["model"]["runtime_lock"]["freeze"]
    freeze.write_text("fixture-package==2.0\n")
    with pytest.raises(ValueError, match="Runtime lock differs from the plan's pinned"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend({}, []))
    assert not world["out"].exists()


def test_committed_runtime_lock_is_the_pinned_freeze():
    plan = json.loads(PLAN.read_text())
    lock = plan["model"]["runtime_lock"]
    assert sha(REPOSITORY / lock["freeze"]) == lock["freeze_sha256"]
    pins = local_hf.parse_freeze((REPOSITORY / lock["freeze"]).read_text())
    for name, version in plan["model"]["runtime_pins"].items():
        assert pins[name] == version, name
    assert pins["huggingface-hub"] == "2.2.0" and lock["python"] == "3.12.15"


def test_committed_plan_runs_the_canary_checked_kernel_path_on_allowlisted_hosts():
    plan = json.loads(PLAN.read_text())
    numerics = plan["engine"]["cpu_numerics"]
    assert (numerics["aten_cpu_capability_env"], numerics["mkl_cbwr"],
            numerics["mkldnn_enabled"], numerics["onednn_max_cpu_isa"]) == (
        "default", "AVX", True, "AVX")
    assert set(plan["sharding"]["hosts"]) == {
        "ICCAD1", "ICCAD3", "iccad5", "iccad10", "iccad12", "iccad13"}


def test_shards_staged_at_different_paths_merge_and_resume(runner, world, clock):
    # Each uwrun run syncs the tree to a new directory, and hosts differ: where the tree
    # was staged must not enter the run identity.
    import shutil

    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    elsewhere = world["tmp"] / "staged-elsewhere"
    shutil.copytree(world["repo"], elsewhere)
    runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                     world["weights"], backend_factory=fake_backend(answers, calls, clock=clock))
    for repository, shards in ((elsewhere, [0, 1]), (world["repo"], [2])):
        for shard in shards:
            for _ in range(3):
                runner.run_shard(path, digest, "discovery", shard, repository, world["out"],
                                 world["weights"],
                                 backend_factory=fake_backend(answers, calls, clock=clock))
    merged = runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)
    assert merged["status"] == "completed" and len(calls) == 100


def test_machine_refusals_are_not_charged_as_invocations(runner, world, clock):
    path, digest = write_plan(world)
    (world["weights"] / "processor_config.json").write_text("{}")
    with pytest.raises(ValueError, match="does not list: processor_config.json"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend({}, []))
    assert not (world["out"] / "stage-discovery/shard-0-of-3/ledger.jsonl").exists()


def test_merge_refuses_a_ledger_copied_from_another_shard(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    stage = world["out"] / "stage-discovery"
    (stage / "shard-1-of-3/ledger.jsonl").write_text(
        (stage / "shard-0-of-3/ledger.jsonl").read_text())
    with pytest.raises(ValueError, match="Shard 1 ledger belongs to another stage, shard"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"], digest)


def test_unlistable_snapshot_directory_or_nameless_package_is_refused(world, monkeypatch):
    import importlib.metadata as metadata

    hidden = world["weights"] / "hidden"
    hidden.mkdir()
    (hidden / "added_tokens.json").write_text("{}")
    hidden.chmod(0)
    try:
        if os.access(hidden, os.R_OK):
            pytest.skip("running as a user who can read mode-000 directories")
        with pytest.raises(ValueError, match="cannot be listed completely"):
            local_hf.verify_snapshot(world["weights"], world["integrity"])
    finally:
        hidden.chmod(0o700)
    nameless = types.SimpleNamespace(metadata={"Name": None}, version="1.0")
    monkeypatch.setattr(metadata, "distributions", lambda: [nameless])
    with pytest.raises(ValueError, match="without a name"):
        local_hf.installed_distributions()


def test_a_charged_invocation_that_failed_before_run_json_still_binds_the_host(
        runner, world, clock, monkeypatch):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    real = runner.local_hf.run_local_hf

    def staging_error(*args, **kwargs):
        raise ValueError("staged image missing")

    monkeypatch.setattr(runner.local_hf, "run_local_hf", staging_error)
    with pytest.raises(ValueError, match="staged image missing"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))
    shard = world["out"] / "stage-discovery/shard-0-of-3"
    assert not (shard / "run/run.json").exists()
    ledger = (shard / "ledger.jsonl").read_text()
    monkeypatch.setattr(runner.local_hf, "run_local_hf", real)
    runner.fixture_host = dict(HOSTS[1])
    with pytest.raises(ValueError, match="ledger started on another host"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], backend_factory=fake_backend(answers, calls))
    assert (shard / "ledger.jsonl").read_text() == ledger and calls == []
    runner.fixture_host = dict(HOSTS[0])
    report = runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                              world["weights"],
                              backend_factory=fake_backend(answers, calls, clock=clock))
    assert report["per_shard"][0]["completed"] == 20


def test_engine_refuses_a_preflight_made_for_another_runtime(world, tmp_path):
    identifier = world["author"][0]["image_id"]
    root = world["repo"] / "data/fgvc-aircraft-2013b/development/images/validation"
    rows = [{"id": identifier, "image": str(root / f"{identifier}.jpg"), "split": "validation",
             "prompt": "x"}]
    engine = {"name": "transformers-cpu", "dtype": "float32", "threads": 32,
              "formatted_prompt_sha256": "a" * 64, "cpu_numerics": {}}
    lock = {"freeze": "a==1\n", "python": "3.12.15"}
    preflight = {"model": {}, "runtime_versions": {"torch": "0"},
                 "runtime_lock_sha256": "0" * 64, **{key: None for key in (
                     "observed_cpu_numerics", "kernel_fingerprint", "host", "python")}}
    with pytest.raises(ValueError, match="preflight was made for another runtime"):
        local_hf.run_local_hf(rows, tmp_path / "x", model_directory=world["weights"],
                              integrity=world["integrity"], engine=engine,
                              runtime_pins={"torch": "1"}, runtime_lock=lock,
                              preflight=preflight, phase="validation", image_root=root,
                              task_contract={"task": "classification"},
                              preprocessing_id=None)
    assert not (tmp_path / "x").exists()
