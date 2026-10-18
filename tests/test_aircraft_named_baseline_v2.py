"""CPU-only tests for plan v2: transformers engine wiring, shards and their merge.

The model is a fake backend; expected values come from the fixture's own answers and
latencies, never from the runner under test. Sampler tests need torch and skip without it
(the CI local-vlm job, which installs torch, runs this file).
"""

import copy
import hashlib
import importlib.util
import json
import math
import time
import types
from pathlib import Path

import pytest
from PIL import Image

from vision_rl import local_vlm
from vision_rl.aircraft_data import aircraft_pixel_sha256

REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY / "docs/aircraft-named-baseline-plan-v2.json"
PLAN_V1 = REPOSITORY / "docs/aircraft-named-baseline-plan-v1.json"
INSTRUCTION = (
    "Return one label copied exactly from the allowed list on its own line. "
    "Do not add a prefix, quotes, explanations, or any text after that label."
)
PER_CLASS = 3
LOAD_SECONDS = 2.0
SHARDS = 3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local_vlm_hf = _load("aircraft_named_baseline_hf",
                     REPOSITORY / "scripts/aircraft_named_baseline_hf.py")
V1_TESTS = _load("named_baseline_v1_tests", Path(__file__).with_name(
    "test_aircraft_named_baseline.py"))
fake_backend, latency_of = V1_TESTS.fake_backend, V1_TESTS.latency_of
independent_discovery, id_list_sha = V1_TESTS.independent_discovery, V1_TESTS.id_list_sha


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    module = _load("aircraft_named_baseline", REPOSITORY / "scripts/aircraft_named_baseline.py")
    fake_time = types.SimpleNamespace(monotonic=clock.monotonic, time=time.time)
    monkeypatch.setattr(module, "time", fake_time)
    monkeypatch.setattr(local_vlm, "time", fake_time)
    monkeypatch.setattr(module, "NATIVE_IMAGES", 100 * PER_CLASS)
    monkeypatch.setattr(module.local_vlm_hf, "runtime_versions", lambda pins: dict(pins))
    return module


def fake_snapshot(directory):
    names = sorted(local_vlm_hf.REQUIRED_FILES) + [
        f"model-{i:05d}-of-00018.safetensors" for i in range(1, 19)]
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(name.encode())
    return names


def fake_hub_metadata(directory, names, lfs=()):
    siblings = []
    for name in names:
        data = (directory / name).read_bytes()
        entry = {"rfilename": name, "size": len(data),
                 "blobId": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
        if name in lfs:
            entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        siblings.append(entry)
    return {"id": local_vlm_hf.REPOSITORY, "sha": local_vlm_hf.REVISION, "siblings": siblings}


@pytest.fixture
def world(tmp_path):
    repo = tmp_path / "repo"
    dataset = repo / "data/fgvc-aircraft-2013b"
    images = dataset / "development/images/validation"
    train_images = dataset / "development/images/train"
    images.mkdir(parents=True)
    train_images.mkdir(parents=True)
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
    train = []
    for number in range(4):
        identifier = f"{2000000 + number:07}"
        image = train_images / f"{identifier}.jpg"
        Image.new("RGB", (32, 40), (number, 50, 90)).save(image)
        train.append({"image_id": identifier, "variant": labels[number], "split": "train",
                      "path": f"development/images/train/{identifier}.jpg",
                      "encoded_sha256": sha(image)})
    (dataset / "audit/train-images.json").write_text(json.dumps(train))
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
    plan = json.loads(PLAN.read_text())
    model = tmp_path / "host-local-snapshot"
    names = fake_snapshot(model)
    record = local_vlm_hf.build_integrity(model, fake_hub_metadata(model, names))
    integrity = repo / plan["model"]["integrity"]
    integrity.parent.mkdir(parents=True, exist_ok=True)
    integrity.write_text(json.dumps(record))
    plan["activated"] = True
    plan["activation"] = {"activated_at": "2026-10-18T00:00:00Z", "approved_by": "test"}
    plan["model"]["integrity_sha256"] = sha(integrity)
    for name, path in (("val_images", dataset / "audit/val-images.json"),
                       ("validation", dataset / "development/validation.jsonl"),
                       ("data_spec", spec_path)):
        plan["inputs"][name]["sha256"] = sha(path)
    plan["smoke"]["train_ids"] = [row["image_id"] for row in train]
    plan["smoke"]["train_images"]["sha256"] = sha(dataset / "audit/train-images.json")
    plan["prompt_and_parser"]["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    plan["prompt_and_parser"]["formatted_prompt_sha256"] = None  # the fake has no template
    discovery = independent_discovery(author)
    plan["stages"]["discovery"]["ids"] = discovery
    plan["stages"]["discovery"]["ids_sha256"] = id_list_sha(discovery)
    # Fixture-scale budgets (fake latencies are 3-13 s), so the gate and the hard ceiling
    # are exercised at this scale rather than at the plan's CPU-hour scale.
    small = {"image_seconds": 10, "load_allowance_seconds": 10,
             "long_generation_allowance_seconds": 10, "max_calendar_seconds": 10**7,
             "max_output_bytes": 10**9}
    plan["stages"]["discovery"]["budget"].update(small, chunk_images=20, max_invocations=6,
                                                 max_active_seconds=10**6)
    plan["stages"]["full"]["budget"].update(small, chunk_images=40, max_invocations=60)
    plan["stages"]["full"]["gate"]["max_projected_active_seconds"] = 10**6
    plan["stages"]["full"]["ids_sha256"] = id_list_sha([row["image_id"] for row in author])
    plan["stages"]["full"]["images"] = 100 * PER_CLASS
    gold = {row["image_id"]: row["variant"] for row in author}
    return {"repo": repo, "plan": plan, "gold": gold, "author": author, "labels": labels,
            "discovery": discovery, "out": tmp_path / "out", "tmp": tmp_path, "model": model,
            "train": train}


def write_plan(world, plan=None):
    path = world["tmp"] / f"activated-{len(list(world['tmp'].glob('activated-*')))}.json"
    path.write_text(json.dumps(plan or world["plan"]))
    return path, sha(path)


def answers_for(world, wrong=()):
    labels = world["labels"]
    return {
        identifier: (labels[(labels.index(label) + 1) % 100] if identifier in wrong else label)
        for identifier, label in world["gold"].items()
    }


def run_shard(runner, world, stage, k, answers, calls, clock, path=None, digest=None):
    if path is None:
        path, digest = write_plan(world)
    return runner.run_stage(path, digest, stage, world["repo"], world["out"],
                            backend_factory=fake_backend(answers, calls, clock=clock),
                            shard=(k, SHARDS), model_directory=world["model"])


def run_all(runner, world, stage, answers, clock, path, digest):
    calls = {k: [] for k in range(SHARDS)}
    for k in range(SHARDS):
        while True:
            report = run_shard(runner, world, stage, k, answers, calls[k], clock, path, digest)
            unit = report_unit(report, k)
            if unit["completed"] == unit["images"]:
                break
    return report, calls


def report_unit(report, k):
    return next(u for u in report["shards"] if u["unit"] == f"shard-{k}-of-{SHARDS}")


def test_committed_v2_plan_is_unactivated_and_refused(runner, tmp_path):
    plan = json.loads(PLAN.read_text())
    assert plan["activated"] is False and plan["activation"] is None
    assert "UNACTIVATED" in plan["status"]
    with pytest.raises(ValueError, match="unactivated"):
        runner.run_stage(PLAN, sha(PLAN), "discovery", REPOSITORY, tmp_path / "never",
                         shard=(0, SHARDS))
    assert not (tmp_path / "never").exists()


def test_committed_v2_plan_pins_match_this_tree_and_v1(runner):
    plan, v1 = json.loads(PLAN.read_text()), json.loads(PLAN_V1.read_text())
    assert plan["model"]["engine"] == "transformers-cpu"
    assert plan["sharding"] == {**plan["sharding"], "shards": SHARDS,
                                "assignment": runner.SHARD_ASSIGNMENT}
    for name, digest in plan["sources_pinned"].items():
        assert sha(REPOSITORY / name) == digest, name
    assert "scripts/aircraft_named_baseline_hf.py" in plan["sources_pinned"]
    # Same prompt, parser, inputs, discovery sample, full manifest and sampling as v1.
    for key in ("prompt_sha256", "parser_source_sha256", "output_grammar", "response_format"):
        assert plan["prompt_and_parser"][key] == v1["prompt_and_parser"][key]
    assert plan["inputs"] == v1["inputs"]
    assert plan["stages"]["discovery"]["ids"] == v1["stages"]["discovery"]["ids"]
    assert plan["stages"]["discovery"]["ids_sha256"] == v1["stages"]["discovery"]["ids_sha256"]
    assert plan["stages"]["full"]["ids_sha256"] == v1["stages"]["full"]["ids_sha256"]
    assert plan["generation"]["sampling"] == local_vlm.INSTRUCT_GENERATION
    assert plan["generation"]["seed"] == v1["generation"]["seed"]
    assert plan["generation"]["dtype"] in local_vlm_hf.DTYPES
    assert plan["comparison"]["resamples"] == 20000 and plan["comparison"]["seed"] == 2718
    assert {e["arm"] for e in plan["comparison"]["endpoints"].values()} == {
        "common", "rl-exposure", "sft-1e5-time"}
    assert plan["baseline_identity"]["revision"] == local_vlm_hf.REVISION
    # The committed integrity record is pinned, and its weights match the Hub's own sha256.
    integrity = REPOSITORY / plan["model"]["integrity"]
    assert sha(integrity) == plan["model"]["integrity_sha256"]
    hub = json.loads((REPOSITORY / plan["model"]["hub_metadata"]).read_text())
    assert sha(REPOSITORY / plan["model"]["hub_metadata"]) == plan["model"]["hub_metadata_sha256"]
    lfs = {s["rfilename"]: s["lfs"]["sha256"] for s in hub["siblings"] if s.get("lfs")}
    record = {f["path"]: f["sha256"] for f in json.loads(integrity.read_text())["files"]}
    assert lfs and all(record[name] == digest for name, digest in lfs.items())
    assert set(record) == {s["rfilename"] for s in hub["siblings"]}
    assert sum(name.endswith(".safetensors") for name in record) == 18


def test_shard_argument_must_match_the_plan(runner, world, clock):
    path, digest = write_plan(world)
    for shard, message in ((None, "pass --shard"), ((0, 2), "pass --shard")):
        with pytest.raises(ValueError, match=message):
            runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                             backend_factory=fake_backend(answers_for(world), []), shard=shard)
    with pytest.raises(ValueError, match="k/N"):
        runner.parse_shard("3/3")
    assert runner.parse_shard("2/3") == (2, 3)
    assert not world["out"].exists()


def test_sharded_discovery_runs_round_robin_and_merges_to_the_unsharded_score(
        runner, world, clock):
    path, digest = write_plan(world)
    discovery = world["discovery"]
    wrong = set(discovery[::9])  # 12 wrong, spread over all shards
    answers = answers_for(world, wrong)
    report, calls = run_all(runner, world, "discovery", answers, clock, path, digest)
    for k in range(SHARDS):
        assert calls[k] == discovery[k::SHARDS]  # each shard, in author order, once
    assert report["status"] == "completed"
    expected = 100 - len(wrong)
    assert report["metric"]["correct"] == expected
    assert report["metric"]["mean_class_accuracy"] == pytest.approx(expected / 100)
    assert report["completed"] == 100 and report["unattempted"] == 0
    # 34/33/33 images, chunks of 20: two invocations per shard.
    assert [u["invocations"] for u in report["shards"]] == [2, 2, 2]
    assert report["invocations"] == 6
    assert all(u["hosts"] for u in report["shards"])
    latency = sum(latency_of(i) for i in discovery) / 100
    assert report["mean_image_latency_seconds"] == pytest.approx(latency)


def test_missing_shard_blocks_metric_merge_and_full(runner, world, clock, capsys):
    path, digest = write_plan(world)
    answers = answers_for(world)
    for k in (0, 1):
        for _ in range(2):
            run_shard(runner, world, "discovery", k, answers, [], clock, path, digest)
    report = runner.stage_report(world["plan"], "discovery", world["repo"], world["out"])
    assert report["status"] == "partial" and report["metric"] is None
    assert report["unattempted"] == len(world["discovery"][2::SHARDS])
    target = world["tmp"] / "merged.json"
    assert runner.main(["merge", "--plan", str(path), "--stage", "discovery",
                        "--output", str(world["out"]), "--repository", str(world["repo"]),
                        "--write", str(target)]) == 2
    assert "merge refused" in capsys.readouterr().err and not target.exists()
    with pytest.raises(ValueError, match="discovery never ran \\(shard-2-of-3\\)"):
        runner.run_stage(path, digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, []), shard=(2, SHARDS),
                         model_directory=world["model"])
    assert not (world["out"] / "stage-full").exists()


def test_misplaced_duplicate_or_stray_shard_records_are_refused(runner, world, clock):
    path, digest = write_plan(world)
    run_all(runner, world, "discovery", answers_for(world), clock, path, digest)
    base = world["out"] / "stage-discovery"
    victim = world["discovery"][0]  # owned by shard 0
    source = (base / "shard-0-of-3/run/attempts.jsonl").read_text().splitlines()
    copied = [line for line in source if json.loads(line)["id"] == victim]
    assert len(copied) == 2
    target = base / "shard-1-of-3/run/attempts.jsonl"
    original = target.read_text()
    target.write_text(original + "\n".join(copied) + "\n")
    with pytest.raises(ValueError, match="does not own"):
        runner.stage_report(world["plan"], "discovery", world["repo"], world["out"])
    target.write_text(original)
    assert runner.stage_report(world["plan"], "discovery", world["repo"],
                               world["out"])["status"] == "completed"
    (base / "shard-3-of-4").mkdir()
    with pytest.raises(ValueError, match="outside the plan"):
        runner.stage_report(world["plan"], "discovery", world["repo"], world["out"])
    (base / "shard-3-of-4").rmdir()
    run_json = base / "shard-2-of-3/run/run.json"
    recorded = json.loads(run_json.read_text())
    recorded["identity"]["rows"] = recorded["identity"]["rows"][:-1]
    run_json.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="other than its assignment"):
        runner.stage_report(world["plan"], "discovery", world["repo"], world["out"])


def test_gate_projects_one_shard_at_the_worst_unit(runner, world):
    plan = world["plan"]
    full = plan["stages"]["full"]
    per_shard = math.ceil(full["images"] / SHARDS)
    chunks = math.ceil(per_shard / full["budget"]["chunk_images"])

    def unit(latency, load, charged, generated, invocations=2):
        return {"mean_image_latency_seconds": latency, "mean_load_seconds": load,
                "charged_active_seconds": charged, "generation_seconds_total": generated,
                "invocations": invocations}

    report = {
        "status": "completed", "unknown_after_interruption": [], "hard_stop_errors": [],
        "completed": 100, "invalid_or_unterminated_answers": 0,
        "shards": [unit(100.0, 50.0, 4000.0, 3400.0), unit(160.0, 80.0, 5400.0, 5300.0),
                   unit(120.0, 60.0, 4500.0, 4000.0)],
    }
    decision = runner.discovery_decision(plan, report)
    # Worst latency 160 s; worst per-invocation overhead max(80, 300, 250) = 300 s.
    assert decision["per_invocation_overhead_seconds"] == 300.0
    assert decision["projected_full_active_seconds"] == pytest.approx(
        per_shard * 160.0 + chunks * 300.0)
    assert decision["projection_scope"] == "per shard (parallel units)"
    report["shards"][1]["mean_load_seconds"] = None
    assert "no completed model load" in " ".join(
        runner.discovery_decision(plan, report)["reasons"])


def write_bundle(world, arms):
    bundle = world["tmp"] / "bundle"
    pins = {}
    for arm, labels in arms.items():
        directory = bundle / "arms" / arm
        directory.mkdir(parents=True)
        lines = [json.dumps({"event": "header", "id": None})]
        for identifier, label in labels.items():
            lines.append(json.dumps({"event": "reserved", "id": identifier}))
            lines.append(json.dumps({"event": "outcome", "id": identifier, "result": {
                "status": "completed", "canonical_label": label}}))
        (directory / "predictions.jsonl").write_text("\n".join(lines) + "\n")
        pins[arm] = sha(directory / "predictions.jsonl")
    return bundle, pins


def test_full_sharded_then_compare_against_pinned_endpoints(runner, world, clock, capsys):
    plan = copy.deepcopy(world["plan"])
    gold = world["gold"]
    labels = world["labels"]
    wrong_endpoint = set(list(gold)[:30])
    arms = {
        "common": {i: (labels[(labels.index(g) + 1) % 100] if i in wrong_endpoint else g)
                   for i, g in gold.items()},
        "rl-exposure": dict(gold),
        "sft-1e5-time": {i: (None if i in wrong_endpoint else g) for i, g in gold.items()},
    }
    bundle, pins = write_bundle(world, arms)
    for endpoint in plan["comparison"]["endpoints"].values():
        endpoint["prediction_journal_sha256"] = pins[endpoint["arm"]]
    path, digest = write_plan(world, plan)
    world["plan"] = plan
    baseline_wrong = set(list(gold)[100:115])
    answers = answers_for(world, baseline_wrong)
    run_all(runner, world, "discovery", answers, clock, path, digest)
    report, calls = run_all(runner, world, "full", answers, clock, path, digest)
    order = [row["image_id"] for row in world["author"]]
    for k in range(SHARDS):
        assert calls[k] == order[k::SHARDS]
    assert report["status"] == "completed" and report["completed"] == 300
    assert report["metric"]["correct"] == 300 - 15
    target = world["tmp"] / "merged.json"
    assert runner.main(["merge", "--plan", str(path), "--stage", "full",
                        "--output", str(world["out"]), "--repository", str(world["repo"]),
                        "--write", str(target)]) == 0
    capsys.readouterr()
    merged = json.loads(target.read_text())
    assert set(merged) == set(gold) and sum(merged[i] == gold[i] for i in gold) == 285
    assert runner.main(["compare-endpoints", "--plan", str(path), "--stage", "full",
                        "--output", str(world["out"]), "--repository", str(world["repo"]),
                        "--bundle", str(bundle)]) == 0
    result = json.loads(capsys.readouterr().out)
    by_arm = {plan["comparison"]["endpoints"][name]["arm"]: value
              for name, value in result["comparisons"].items()}
    # Hand values: each class has 3 images, so a wrong image costs 1/300 of macro accuracy.
    assert by_arm["rl-exposure"]["mean_class_accuracy_difference"] == pytest.approx(-15 / 300)
    assert by_arm["common"]["mean_class_accuracy_difference"] == pytest.approx((30 - 15) / 300)
    assert by_arm["sft-1e5-time"]["mean_class_accuracy_difference"] == pytest.approx(
        (30 - 15) / 300)
    assert all(v["uncertainty"]["seed"] == 2718 and v["uncertainty"]["draws"] == 20000
               for v in result["comparisons"].values())
    # A comparator journal that is not the pinned one is refused.
    journal = bundle / "arms/common/predictions.jsonl"
    journal.write_text(journal.read_text() + "\n")
    assert runner.main(["compare-endpoints", "--plan", str(path), "--stage", "full",
                        "--output", str(world["out"]), "--repository", str(world["repo"]),
                        "--bundle", str(bundle)]) == 2
    assert "differs from the plan pin" in capsys.readouterr().err


def test_snapshot_integrity_refuses_tampering_extras_and_unpinned_records(world, tmp_path):
    plan = world["plan"]
    integrity = world["repo"] / plan["model"]["integrity"]
    pin = plan["model"]["integrity_sha256"]
    verified = local_vlm_hf.verify_snapshot(world["model"], integrity, pin)
    assert len(verified["files"]) == len(local_vlm_hf.REQUIRED_FILES) + 18
    with pytest.raises(ValueError, match="plan pin"):
        local_vlm_hf.verify_snapshot(world["model"], integrity, "0" * 64)
    extra = world["model"] / "pytorch_model.bin"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="outside the integrity record"):
        local_vlm_hf.verify_snapshot(world["model"], integrity, pin)
    extra.unlink()
    shard = world["model"] / "model-00007-of-00018.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1] + b"!")
    with pytest.raises(ValueError, match="integrity mismatch"):
        local_vlm_hf.verify_snapshot(world["model"], integrity, pin)
    # build_integrity checks LFS sha256 and small-file git blob ids against the Hub.
    other = tmp_path / "other"
    names = fake_snapshot(other)
    metadata = fake_hub_metadata(other, names, lfs={names[-1]})
    assert len(local_vlm_hf.build_integrity(other, metadata)["files"]) == len(names)
    metadata["siblings"][-1]["lfs"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="LFS sha256"):
        local_vlm_hf.build_integrity(other, metadata)
    metadata = fake_hub_metadata(other, names)
    metadata["siblings"][0]["blobId"] = "0" * 40
    with pytest.raises(ValueError, match="blob id"):
        local_vlm_hf.build_integrity(other, metadata)
    (other / names[0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        local_vlm_hf.build_integrity(other, fake_hub_metadata(other, names[1:]) | {
            "siblings": fake_hub_metadata(other, names[1:])["siblings"] + [
                {"rfilename": names[0], "size": 1, "blobId": "0" * 40}]})


def test_smoke_runs_pinned_train_images_only_through_the_same_journal(runner, world, clock):
    plan = world["plan"]
    train = {row["image_id"]: row["variant"] for row in world["train"]}
    calls = []
    answers = dict(train)
    answers[world["train"][1]["image_id"]] = "not a label"
    report = runner.smoke_train(plan, world["repo"], world["out"],
                                model_directory=world["model"],
                                backend_factory=fake_backend(answers, calls, clock=clock))
    assert calls == list(train)
    assert report["completed"] == 4 and report["val_consumed"] is False
    assert [image["correct_on_train"] for image in report["images"]] == [
        True, False, True, True]
    assert report["mean_seconds_per_image"] == pytest.approx(
        sum(latency_of(i) for i in train) / 4)
    assert not (world["out"] / "stage-discovery").exists()
    v1 = json.loads(PLAN_V1.read_text())
    with pytest.raises(ValueError, match="transformers engine only"):
        runner.smoke_train(v1, world["repo"], world["tmp"] / "v1-smoke")


def test_mlx_sampler_matches_an_independent_restatement():
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    generation = {**local_vlm.INSTRUCT_GENERATION, "top_k": 3}

    def expected(logits, context):
        # Written from mlx-vlm 0.6.8 sample_utils/generate.ar, independently of the module.
        values = numpy.array(logits, dtype=numpy.float64)
        for token in set(context[-20:]):
            values[token] -= 1.5
        logprobs = values - numpy.log(numpy.exp(values - values.max()).sum()) - values.max()
        probs = numpy.exp(logprobs)
        order = numpy.argsort(logprobs, kind="stable")
        cumulative = numpy.empty_like(probs)
        cumulative[order] = numpy.cumsum(probs[order])
        logprobs = numpy.where(cumulative > 1 - 0.8, logprobs, -numpy.inf)
        keep = numpy.argsort(-logprobs, kind="stable")[:3]
        masked = numpy.full_like(logprobs, -numpy.inf)
        masked[keep] = logprobs[keep]
        scaled = numpy.exp((masked - masked.max()) / 0.7)
        return scaled / scaled.sum()

    rng = numpy.random.default_rng(5)
    for _ in range(50):
        logits = rng.normal(0, 2, 12).tolist()
        context = rng.integers(0, 12, 30).tolist()
        got = local_vlm_hf.mlx_instruct_probs(torch.tensor(logits), torch.tensor(context),
                                              generation)
        assert numpy.allclose(got.numpy(), expected(logits, context), atol=1e-6)
    # Hand case: token 0 dominates after the penalty on token 1; nucleus keeps {0, 1}.
    probs = local_vlm_hf.mlx_instruct_probs(
        torch.tensor([3.0, 3.0, 0.0, -2.0]), torch.tensor([1, 1, 1]), generation)
    assert probs[2] == 0 and probs[3] == 0 and probs[0] > probs[1] > 0
    first = local_vlm_hf.mlx_instruct_sample(
        torch.tensor(logits), torch.tensor(context), generation, torch.Generator().manual_seed(9))
    again = local_vlm_hf.mlx_instruct_sample(
        torch.tensor(logits), torch.tensor(context), generation, torch.Generator().manual_seed(9))
    assert first == again
    with pytest.raises(ValueError, match="Greedy"):
        local_vlm_hf.mlx_instruct_probs(torch.tensor(logits), torch.tensor(context),
                                        {**generation, "temperature": 0.0})


def test_eos_set_is_v1s_mlx_stopping_set_from_the_generation_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 248044}}))
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": [248046, 248044]}))
    # The mlx-community config v1 stopped on: top-level eos_token_id [248046, 248044].
    assert local_vlm_hf.eos_token_ids(tmp_path) == [248044, 248046]
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [248046]}))
    with pytest.raises(ValueError, match="omits the text config"):
        local_vlm_hf.eos_token_ids(tmp_path)
