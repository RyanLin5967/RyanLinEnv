"""CPU-only tests for the named-baseline runner. The model is a fake backend."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
from PIL import Image

from vision_rl import local_vlm
from vision_rl.aircraft_data import aircraft_pixel_sha256

REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY / "docs/aircraft-named-baseline-plan-v1.json"
INSTRUCTION = (
    "Return one label copied exactly from the allowed list on its own line. "
    "Do not add a prefix, quotes, explanations, or any text after that label."
)
PER_CLASS = 3


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "aircraft_named_baseline", REPOSITORY / "scripts/aircraft_named_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "NATIVE_IMAGES", 100 * PER_CLASS)
    monkeypatch.setattr(local_vlm, "runtime_versions", lambda: {"mlx-vlm": "0.6.8"})
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_key(label, identifier):
    # Written out by hand, not through the runner's serializer.
    text = (
        '["fgvc-aircraft-2013b","fgvc-aircraft-freshpool-screened-grpo-v1",'
        f'"baseline-discovery",791,"{label}","{identifier}"]'
    )
    return hashlib.sha256(text.encode()).hexdigest()


def independent_discovery(author):
    best = {}
    for row in author:
        key = independent_key(row["variant"], row["image_id"])
        if row["variant"] not in best or key < best[row["variant"]][0]:
            best[row["variant"]] = (key, row["image_id"])
    chosen = {value[1] for value in best.values()}
    return [row["image_id"] for row in author if row["image_id"] in chosen]


def id_list_sha(ids):
    return hashlib.sha256(("[" + ",".join(f'"{i}"' for i in ids) + "]").encode()).hexdigest()


@pytest.fixture
def world(tmp_path):
    repo = tmp_path / "repo"
    dataset = repo / "data/fgvc-aircraft-2013b"
    images = dataset / "development/images/validation"
    images.mkdir(parents=True)
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
    model = repo / "data/models/Qwen3.8-27B-8bit"
    model.mkdir(parents=True)
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "preprocessor_config.json", "processor_config.json"]
    names += [f"model-{i:05d}-of-00006.safetensors" for i in range(1, 7)]
    files = []
    for name in names:
        (model / name).write_bytes(name.encode())
        files.append({"path": name, "bytes": len(name.encode()),
                      "sha256": hashlib.sha256(name.encode()).hexdigest()})
    integrity = repo / "runs/local-qwen8bit/model-integrity.json"
    integrity.parent.mkdir(parents=True)
    integrity.write_text(json.dumps({
        "model_repository": local_vlm.REPOSITORY, "revision": local_vlm.REVISION,
        "quantization": {"bits": 8, "mode": "affine", "group_size": 64}, "files": files,
    }))
    plan = json.loads(PLAN.read_text())
    plan["activated"] = True
    plan["activation"] = {"activated_at": "2026-10-17T00:00:00Z", "approved_by": "test"}
    for name, path in (("val_images", dataset / "audit/val-images.json"),
                       ("validation", dataset / "development/validation.jsonl"),
                       ("data_spec", spec_path)):
        plan["inputs"][name]["sha256"] = sha(path)
    plan["prompt_and_parser"]["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    discovery = independent_discovery(author)
    plan["stages"]["discovery"]["ids"] = discovery
    plan["stages"]["discovery"]["ids_sha256"] = id_list_sha(discovery)
    plan["stages"]["full"]["ids_sha256"] = id_list_sha([row["image_id"] for row in author])
    gold = {row["image_id"]: row["variant"] for row in author}
    return {"repo": repo, "plan": plan, "gold": gold, "author": author, "labels": labels,
            "discovery": discovery, "out": tmp_path / "out", "tmp": tmp_path}


def write_plan(world, plan=None):
    path = world["tmp"] / f"activated-{len(list(world['tmp'].glob('activated-*')))}.json"
    path.write_text(json.dumps(plan or world["plan"]))
    return path, sha(path)


def fake_backend(answers, calls, *, finish="stop", fail=None):
    class Backend:
        load_info = {"load_seconds": 2.0, "device": {"fake": True}}

        def __init__(self, directory, config):
            assert config["profile"] == "instruct" and config["enable_thinking"] is False
            assert config["generation"]["temperature"] == 0.7
            self.config = config

        def infer(self, row, seed):
            identifier = row["id"]
            calls.append(identifier)
            if fail is not None and identifier == fail[0]:
                raise fail[1]
            return {"native_response": {
                "text": answers[identifier], "finish_reason": finish, "prompt_tokens": 11,
                "generation_tokens": 3, "total_tokens": 14, "prompt_tps": 1.0,
                "generation_tps": 1.0, "peak_memory": 0.1,
            }, "preprocessing": {}, "peak_memory_bytes": 1}

        def close(self):
            pass

    return Backend


def answers_for(world, wrong=()):
    labels = world["labels"]
    return {
        identifier: (labels[(labels.index(label) + 1) % 100] if identifier in wrong else label)
        for identifier, label in world["gold"].items()
    }


def test_committed_plan_is_unactivated_and_refused(runner, tmp_path):
    plan = json.loads(PLAN.read_text())
    assert plan["activated"] is False and plan["activation"] is None
    assert "UNACTIVATED" in plan["status"]
    with pytest.raises(ValueError, match="unactivated"):
        runner.load_plan(PLAN, sha(PLAN))
    with pytest.raises(ValueError, match="unactivated"):
        runner.run_stage(PLAN, sha(PLAN), "discovery", REPOSITORY, tmp_path / "never")
    assert not (tmp_path / "never").exists()
    assert runner.main(["run", "--plan", str(PLAN), "--expected-plan-sha256", sha(PLAN),
                        "--stage", "discovery", "--output", str(tmp_path / "never")]) == 2


def test_committed_plan_pins_match_this_tree(runner):
    plan = json.loads(PLAN.read_text())
    ids = plan["stages"]["discovery"]["ids"]
    assert len(ids) == len(set(ids)) == 100
    assert plan["stages"]["discovery"]["ids_sha256"] == id_list_sha(ids)
    assert plan["prompt_and_parser"]["parser_source_sha256"] == runner.parser_source_sha256()
    for name, digest in plan["sources_pinned"].items():
        assert sha(REPOSITORY / name) == digest, name
    assert plan["generation"]["sampling"] == local_vlm.INSTRUCT_GENERATION
    assert plan["model"]["runtime_pins"]["mlx-vlm"] == "0.6.8"
    half = plan["stages"]["discovery"]["uncertainty"]["half_width"]
    assert half == pytest.approx(math.sqrt(math.log(40) / 200), rel=1e-12)
    assert runner.hoeffding_half_width(100, 0.95) == pytest.approx(half, rel=1e-12)


def test_discovery_selection_matches_independent_hashing(runner, world):
    selected = runner.select_discovery(world["author"], world["labels"])
    assert selected == world["discovery"]
    assert len({world["gold"][i] for i in selected}) == 100
    assert runner.select_discovery(world["author"][::-1], world["labels"]) == selected[::-1]
    with pytest.raises(ValueError, match="native VAL"):
        runner.select_discovery([{**world["author"][0], "split": "test"}] + world["author"][1:],
                                world["labels"])


def test_discovery_runs_in_chunks_resumes_and_scores_with_evaluator_rules(runner, world):
    path, digest = write_plan(world)
    discovery = world["discovery"]
    wrong = set(discovery[:13])
    answers = answers_for(world, wrong)
    answers[discovery[20]] = "Variant 020 because of the tail"  # prose: invalid, never repaired
    calls = []
    first = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                             backend_factory=fake_backend(answers, calls))
    assert first["status"] == "partial" and first["completed"] == 50 and first["metric"] is None
    assert calls == discovery[:50]
    second = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                              backend_factory=fake_backend(answers, calls))
    assert calls == discovery and second["status"] == "completed"
    again = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                             backend_factory=fake_backend(answers, calls))
    assert calls == discovery and again["metric"] == second["metric"]
    expected_correct = 100 - 13 - 1
    assert second["metric"]["correct"] == expected_correct
    assert second["metric"]["mean_class_accuracy"] == pytest.approx(expected_correct / 100)
    assert second["metric"]["micro_accuracy"] == pytest.approx(expected_correct / 100)
    assert second["invalid_or_unterminated_answers"] == 1
    low, high = second["metric"]["hoeffding"]["interval"]
    assert low == pytest.approx(0.86 - 0.13581015157406195)
    assert high == pytest.approx(min(1.0, 0.86 + 0.13581015157406195))
    assert second["decision"]["decision"] == "proceed"
    ledger = runner.ledger_state(world["out"] / "stage-discovery/ledger.jsonl")
    assert ledger["invocations"] == 2 and not ledger["unfinished"]


def test_unterminated_answers_are_invalid_and_block_full(runner, world):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    for _ in range(2):
        report = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                                  backend_factory=fake_backend(answers, calls, finish="length"))
    assert report["metric"]["correct"] == 0
    assert report["invalid_or_unterminated_answers"] == 100
    assert report["decision"]["decision"] == "stop"
    with pytest.raises(ValueError, match="discovery gate"):
        runner.run_stage(path, digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls))


def test_interruption_is_unknown_never_regenerated_and_halts(runner, world):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    victim = world["discovery"][3]
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls,
                                                      fail=(victim, KeyboardInterrupt())))
    ledger = runner.ledger_state(world["out"] / "stage-discovery/ledger.jsonl")
    assert ledger["invocations"] == 1 and not ledger["unfinished"]
    with pytest.raises(ValueError, match="unknown or hard-stop"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls))
    assert calls.count(victim) == 1
    report = runner.stage_report(world["plan"], "discovery", world["repo"], world["out"])
    assert report["unknown_after_interruption"] == [victim] and report["metric"] is None
    with pytest.raises(ValueError, match="discovery gate"):
        runner.run_stage(path, digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls))


def test_backend_error_hard_stops_and_halts(runner, world):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                     backend_factory=fake_backend(answers, calls,
                                                  fail=(world["discovery"][0], MemoryError())))
    with pytest.raises(ValueError, match="unknown or hard-stop"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls))
    assert calls == world["discovery"][:1]


def test_unfinished_invocation_is_charged_its_cap_and_blocks(runner, world):
    path, digest = write_plan(world)
    ledger = world["out"] / "stage-discovery/ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"event": "started", "invocation": 1, "time": 0,
                                  "wall_cap_seconds": 8700, "plan_sha256": digest,
                                  "sources": runner.source_hashes()}) + "\n")
    state = runner.ledger_state(ledger)
    assert state["charged_active_seconds"] == 8700 and state["unfinished"] == [1]
    with pytest.raises(ValueError, match="never recorded its finish"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers_for(world), []))


def test_budgets_refuse_new_invocations(runner, world):
    plan = copy.deepcopy(world["plan"])
    plan["stages"]["discovery"]["budget"]["max_invocations"] = 1
    path, digest = write_plan(world, plan)
    calls = []
    runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                     backend_factory=fake_backend(answers_for(world), calls))
    with pytest.raises(ValueError, match="invocation budget exhausted"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers_for(world), calls))
    assert len(calls) == 50


def test_calendar_budget_survives_resume(runner, world):
    path, digest = write_plan(world)
    calls = []
    runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                     backend_factory=fake_backend(answers_for(world), calls))
    with pytest.raises(ValueError, match="calendar budget exhausted"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers_for(world), calls),
                         clock=lambda: 10**12)


def test_plan_hash_and_inputs_are_enforced(runner, world):
    path, digest = write_plan(world)
    with pytest.raises(ValueError, match="exact SHA256"):
        runner.run_stage(path, "0" * 64, "discovery", world["repo"], world["out"])
    runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                     backend_factory=fake_backend(answers_for(world), []))
    other = copy.deepcopy(world["plan"])
    other["generation"]["seed"] = 7
    other_path, other_digest = write_plan(world, other)
    with pytest.raises(ValueError, match="Plan or source changed"):
        runner.run_stage(other_path, other_digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers_for(world), []))
    spec = world["repo"] / "data/fgvc-aircraft-2013b/candidates/fourshot-seed791/data_spec.json"
    spec.write_text(spec.read_text().replace("Identify", "Name"))
    with pytest.raises(ValueError, match="Pinned input changed: data_spec"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"])


def test_wrong_prompt_or_discovery_ids_are_refused(runner, world):
    plan = copy.deepcopy(world["plan"])
    plan["prompt_and_parser"]["prompt_sha256"] = "0" * 64
    path, digest = write_plan(world, plan)
    with pytest.raises(ValueError, match="eight-endpoint evaluator"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"])
    plan = copy.deepcopy(world["plan"])
    plan["stages"]["discovery"]["ids"] = plan["stages"]["discovery"]["ids"][::-1]
    path, digest = write_plan(world, plan)
    with pytest.raises(ValueError, match="pre-registered IDs"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"])


def test_full_requires_discovery_then_runs_with_derived_cap(runner, world):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    with pytest.raises(ValueError, match="discovery gate"):
        runner.run_stage(path, digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls))
    assert calls == []
    for _ in range(2):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls))
    report = runner.stage_report(world["plan"], "discovery", world["repo"], world["out"])
    decision = runner.discovery_decision(world["plan"], report)
    latency = report["mean_image_latency_seconds"]
    projection = 3333 * latency + math.ceil(3333 / 128) * 2.0
    assert decision["projected_full_active_seconds"] == pytest.approx(projection)
    assert decision["full_max_active_seconds"] == pytest.approx(min(172800, 2 * projection))
    full_calls = []
    for _ in range(3):
        result = runner.run_stage(path, digest, "full", world["repo"], world["out"],
                                  backend_factory=fake_backend(answers, full_calls))
    assert full_calls == [row["image_id"] for row in world["author"]]
    assert result["status"] == "completed" and result["metric"]["mean_class_accuracy"] == 1.0


def test_unaffordable_profile_stops_before_full(runner, world):
    plan = copy.deepcopy(world["plan"])
    plan["stages"]["full"]["gate"]["max_projected_active_seconds"] = 1
    path, digest = write_plan(world, plan)
    answers, calls = answers_for(world), []
    for _ in range(2):
        report = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                                  backend_factory=fake_backend(answers, calls))
    assert report["decision"]["decision"] == "stop"
    assert "unaffordable" in report["decision"]["reasons"][0]
    with pytest.raises(ValueError, match="discovery gate"):
        runner.run_stage(path, digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls))


def test_compare_uses_the_shared_paired_comparator(runner, world, capsys):
    path, digest = write_plan(world)
    discovery = world["discovery"]
    answers = answers_for(world, set(discovery[:10]))
    for _ in range(2):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, []))
    against = world["tmp"] / "endpoint.json"
    against.write_text(json.dumps({i: world["gold"][i] for i in discovery}))
    assert runner.main(["compare", "--plan", str(path), "--stage", "discovery",
                        "--output", str(world["out"]), "--repository", str(world["repo"]),
                        "--against", str(against)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mean_class_accuracy_difference"] == pytest.approx(-0.10)
    assert result["uncertainty"]["seed"] == 2718 and result["uncertainty"]["draws"] == 20000
