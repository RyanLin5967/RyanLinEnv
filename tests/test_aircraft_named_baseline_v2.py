"""CPU-only tests for the named-baseline v2 runner (engine option, shards, merge).

The model is a fake backend; no torch is imported. Expected values are derived from the
fixture by hand, never by calling the runner.
"""

import hashlib
import importlib.util
import json
import os
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
    monkeypatch.setattr(module.local_hf, "runtime_versions", lambda pins: dict(pins))
    return module


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
    plan["sources_pinned"] = {name: sha(REPOSITORY / name) for name in plan["sources_pinned"]}
    gold = {row["image_id"]: row["variant"] for row in author}
    return {"repo": repo, "plan": plan, "gold": gold, "author": author, "labels": labels,
            "discovery": discovery, "out": tmp_path / "out", "tmp": tmp_path,
            "weights": weights, "record": record, "integrity": integrity}


def write_plan(world, plan=None):
    path = world["tmp"] / f"activated-{len(list(world['tmp'].glob('activated-*')))}.json"
    path.write_text(json.dumps(plan or world["plan"]))
    return path, sha(path)


def fake_backend(answers, calls, *, finish="stop", fail=None, clock=None, seen=None):
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
                clock.now += latency_of(row["id"])
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


def answers_for(world, wrong=()):
    labels = world["labels"]
    return {i: (labels[(labels.index(g) + 1) % 100] if i in wrong else g)
            for i, g in world["gold"].items()}


def run_all(runner, world, stage, answers, calls, clock, path, digest, *, shards=range(3)):
    report = None
    for shard in shards:
        for _ in range(4):
            report = runner.run_shard(path, digest, stage, shard, world["repo"], world["out"],
                                      world["weights"], 7,
                                      backend_factory=fake_backend(answers, calls, clock=clock))
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
                        "--threads", "1", "--output", str(tmp_path / "never")]) == 2
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
                         world["weights"], 1, backend_factory=fake_backend({}, []))
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
    assert plan["sharding"]["shards"] == 3
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
                               world["weights"], 5,
                               backend_factory=fake_backend(answers, calls, clock=clock,
                                                            seen=seen))
    assert seen[0]["threads"] == 5
    assert calls == discovery[1::3][:20] and partial["status"] == "partial"
    assert partial["metric"] is None
    report = run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    assert sorted(calls) == sorted(discovery) and len(calls) == 100
    merged = runner.merge(world["plan"], "discovery", world["repo"], world["out"])
    assert merged["status"] == "completed" == report["status"]
    assert merged["metric"]["correct"] == 100 - 13 - 1
    assert merged["metric"]["mean_class_accuracy"] == pytest.approx(0.86)
    assert merged["metric"]["image_accuracy"] == pytest.approx(0.86)
    assert merged["invalid_or_unterminated_answers"] == 1
    assert [s["assigned"] for s in merged["per_shard"]] == [34, 33, 33]
    assert merged["predictions"][discovery[0]] != world["gold"][discovery[0]]
    decision = runner.discovery_decision(world["plan"], merged)
    assert decision["decision"] == "proceed", decision


def test_merge_refuses_missing_shard(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest, shards=[0, 2])
    report = runner.merged_report(world["plan"], "discovery", world["repo"], world["out"])
    assert report["status"] == "partial" and report["missing_images"] == 33
    with pytest.raises(ValueError, match="Merge refused: 33 missing"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"])
    with pytest.raises(ValueError, match="discovery gate"):
        runner.run_shard(path, digest, "full", 0, world["repo"], world["out"],
                         world["weights"], 1, backend_factory=fake_backend(answers, calls))


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
        runner.merge(world["plan"], "discovery", world["repo"], world["out"])
    victim.write_text(original)
    runner.merge(world["plan"], "discovery", world["repo"], world["out"])
    (stage / "shard-0-of-2").mkdir()
    with pytest.raises(ValueError, match="Unexpected shard directory"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"])


def test_merge_refuses_shards_with_different_identity(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    run_json = world["out"] / "stage-discovery/shard-2-of-3/run/run.json"
    recorded = json.loads(run_json.read_text())
    recorded["identity"]["config"]["engine"]["dtype"] = "bfloat16"
    run_json.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="different run identity"):
        runner.merge(world["plan"], "discovery", world["repo"], world["out"])


def test_full_runs_after_discovery_and_compares_with_declared_arms(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world, wrong=set(list(world["gold"])[:30])), []
    with pytest.raises(ValueError, match="discovery shard 0 never ran"):
        runner.run_shard(path, digest, "full", 0, world["repo"], world["out"],
                         world["weights"], 1, backend_factory=fake_backend(answers, calls))
    run_all(runner, world, "discovery", answers, calls, clock, path, digest)
    calls.clear()
    run_all(runner, world, "full", answers, calls, clock, path, digest)
    assert sorted(calls) == sorted(world["gold"])
    merged = runner.merge(world["plan"], "full", world["repo"], world["out"])
    assert merged["metric"]["correct"] == 270
    perfect = {i: g for i, g in world["gold"].items()}
    arms = {}
    for name in ("W", "RL400", "strongest_SFT"):
        arms[name] = world["tmp"] / f"{name}.json"
        arms[name].write_text(json.dumps(perfect))
    result = runner.compare(world["plan"], world["repo"], world["out"],
                            {k: str(v) for k, v in arms.items()})
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
        runner.compare(world["plan"], world["repo"], world["out"], {"W": str(arms["W"])})
    short = world["tmp"] / "short.json"
    short.write_text(json.dumps(dict(list(perfect.items())[:-1])))
    with pytest.raises(ValueError, match="does not cover"):
        runner.compare(world["plan"], world["repo"], world["out"],
                       {**{k: str(v) for k, v in arms.items()}, "W": str(short)})


def test_source_change_after_first_invocation_is_refused(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                     world["weights"], 1,
                     backend_factory=fake_backend(answers, calls, clock=clock))
    ledger = world["out"] / "stage-discovery/shard-0-of-3/ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    events[0]["sources"]["scripts/aircraft_named_baseline_hf.py"] = "0" * 64
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(ValueError, match="changed since this shard's first invocation"):
        runner.run_shard(path, digest, "discovery", 0, world["repo"], world["out"],
                         world["weights"], 1, backend_factory=fake_backend(answers, calls))


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


def test_engine_identity_needs_dtype_and_excludes_threads(world, tmp_path):
    identifier = world["author"][0]["image_id"]
    root = world["repo"] / "data/fgvc-aircraft-2013b/development/images/validation"
    rows = [{"id": identifier, "image": str(root / f"{identifier}.jpg"), "split": "validation",
             "prompt": "x"}]
    for engine in ({"name": "transformers-cpu", "dtype": "bfloat16", "threads": 4},
                   {"name": "transformers-cpu", "dtype": "float16"},
                   {"name": "transformers-cpu"}):
        with pytest.raises(ValueError, match="explicit dtype and no thread count"):
            local_hf.run_local_hf(rows, tmp_path / "x", model_directory=world["weights"],
                                  integrity=world["integrity"], engine=engine, runtime_pins={},
                                  phase="validation", image_root=root,
                                  task_contract={"task": "classification"},
                                  preprocessing_id=None)
    assert not (tmp_path / "x").exists()


def test_smoke_reads_train_images_only(runner, world):
    val_id = world["author"][0]["image_id"]
    with pytest.raises(OSError):
        runner.smoke(world["repo"], world["weights"], world["integrity"], [val_id],
                     world["repo"] / "data/fgvc-aircraft-2013b/candidates/fourshot-seed791/"
                     "data_spec.json", 1, "float32", {}, None,
                     backend_factory=fake_backend({}, []))
