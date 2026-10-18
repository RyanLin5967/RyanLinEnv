"""CPU-only tests for the named-baseline runner. The model is a fake backend."""

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
PLAN = REPOSITORY / "docs/aircraft-named-baseline-plan-v1.json"
INSTRUCTION = (
    "Return one label copied exactly from the allowed list on its own line. "
    "Do not add a prefix, quotes, explanations, or any text after that label."
)
PER_CLASS = 3


class FakeClock:
    """Monotonic time that only the fake backend advances, so every latency is recorded."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


def latency_of(identifier):
    # The backend's recorded per-image latency; tests derive expected values from this.
    return float(3 + int(identifier) % 11)


LOAD_SECONDS = 2.0


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def runner(monkeypatch, clock):
    spec = importlib.util.spec_from_file_location(
        "aircraft_named_baseline", REPOSITORY / "scripts/aircraft_named_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_time = types.SimpleNamespace(monotonic=clock.monotonic, time=time.time)
    monkeypatch.setattr(module, "time", fake_time)
    monkeypatch.setattr(local_vlm, "time", fake_time)
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


def fake_backend(answers, calls, *, finish="stop", fail=None, clock=None):
    class Backend:
        load_info = {"load_seconds": LOAD_SECONDS, "device": {"fake": True}}

        def __init__(self, directory, config):
            assert config["profile"] == "instruct" and config["enable_thinking"] is False
            assert config["generation"]["temperature"] == 0.7
            self.config = config
            if clock is not None:
                clock.now += LOAD_SECONDS

        def infer(self, row, seed):
            identifier = row["id"]
            calls.append(identifier)
            if clock is not None:
                clock.now += latency_of(identifier)
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


def test_discovery_runs_in_chunks_resumes_and_scores_with_evaluator_rules(runner, world, clock):
    path, digest = write_plan(world)
    discovery = world["discovery"]
    wrong = set(discovery[:13])
    answers = answers_for(world, wrong)
    answers[discovery[20]] = "Variant 020 because of the tail"  # prose: invalid, never repaired
    # Real latencies: a zero-latency discovery projects below one full wall cap, and the
    # hard ceiling would refuse every full invocation, so the gate must stop it.
    calls = []
    first = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                             backend_factory=fake_backend(answers, calls, clock=clock))
    assert first["status"] == "partial" and first["completed"] == 50 and first["metric"] is None
    assert calls == discovery[:50]
    second = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                              backend_factory=fake_backend(answers, calls, clock=clock))
    assert calls == discovery and second["status"] == "completed"
    again = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                             backend_factory=fake_backend(answers, calls, clock=clock))
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


def test_full_requires_discovery_then_runs_with_derived_cap(runner, world, clock):
    path, digest = write_plan(world)
    answers, calls = answers_for(world), []
    with pytest.raises(ValueError, match="discovery gate"):
        runner.run_stage(path, digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls, clock=clock))
    assert calls == []
    for _ in range(2):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls, clock=clock))
    # Expected values come from the backend's recorded latencies, not the runner's report.
    latency = sum(latency_of(i) for i in world["discovery"]) / 100
    charged = 2 * LOAD_SECONDS + sum(latency_of(i) for i in world["discovery"])
    report = runner.stage_report(world["plan"], "discovery", world["repo"], world["out"])
    assert report["mean_image_latency_seconds"] == pytest.approx(latency)
    assert report["mean_load_seconds"] == LOAD_SECONDS
    assert report["charged_active_seconds"] == pytest.approx(charged)
    decision = runner.discovery_decision(world["plan"], report)
    projection = 3333 * latency + math.ceil(3333 / 128) * LOAD_SECONDS
    assert decision["projected_full_active_seconds"] == pytest.approx(projection)
    assert decision["full_max_active_seconds"] == pytest.approx(min(172800, 2 * projection))
    full_calls = []
    for _ in range(3):
        result = runner.run_stage(path, digest, "full", world["repo"], world["out"],
                                  backend_factory=fake_backend(answers, full_calls, clock=clock))
    assert full_calls == [row["image_id"] for row in world["author"]]
    assert result["status"] == "completed" and result["metric"]["mean_class_accuracy"] == 1.0
    full_charged = 3 * LOAD_SECONDS + sum(latency_of(r["image_id"]) for r in world["author"])
    assert result["charged_active_seconds"] == pytest.approx(full_charged)
    assert full_charged <= decision["full_max_active_seconds"]


def test_full_refuses_discovery_from_another_plan_or_source(runner, world, clock):
    path, digest = write_plan(world)
    answers = answers_for(world)
    for _ in range(2):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, [], clock=clock))
    relaxed = copy.deepcopy(world["plan"])
    relaxed["stages"]["full"]["gate"]["minimum_valid_answer_fraction"] = 0.0
    relaxed_path, relaxed_digest = write_plan(world, relaxed)
    calls = []
    with pytest.raises(ValueError, match="different plan or source"):
        runner.run_stage(relaxed_path, relaxed_digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls, clock=clock))
    ledger = world["out"] / "stage-discovery/ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    for event in events:
        if event["event"] == "started":
            event["sources"] = {**event["sources"], "src/vision_rl/frontier.py": "0" * 64}
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="different plan or source"):
        runner.run_stage(path, digest, "full", world["repo"], world["out"],
                         backend_factory=fake_backend(answers, calls, clock=clock))
    assert calls == [] and not (world["out"] / "stage-full").exists()


def discovery_charge(world, count):
    return LOAD_SECONDS + sum(latency_of(i) for i in world["discovery"][:count])


@pytest.mark.parametrize("slack, expect_halt", [(0.0, False), (-1.0, True)])
def test_active_seconds_cap_is_a_hard_ceiling(runner, world, clock, slack, expect_halt):
    wall = 600 + 50 * 120 + 2100
    plan = copy.deepcopy(world["plan"])
    # Room for exactly one charged chunk plus one more full wall cap (or 1 s less).
    plan["stages"]["discovery"]["budget"]["max_active_seconds"] = (
        discovery_charge(world, 50) + wall + slack)
    path, digest = write_plan(world, plan)
    calls = []
    runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                     backend_factory=fake_backend(answers_for(world), calls, clock=clock))
    assert len(calls) == 50
    if expect_halt:
        with pytest.raises(ValueError, match="active-seconds budget exhausted"):
            runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                             backend_factory=fake_backend(answers_for(world), calls, clock=clock))
        assert len(calls) == 50
    else:
        report = runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                                  backend_factory=fake_backend(answers_for(world), calls,
                                                               clock=clock))
        assert report["status"] == "completed"
        assert report["charged_active_seconds"] <= plan["stages"]["discovery"]["budget"][
            "max_active_seconds"]


def test_active_seconds_cap_refuses_a_first_invocation_that_cannot_fit(runner, world, clock):
    plan = copy.deepcopy(world["plan"])
    plan["stages"]["discovery"]["budget"]["max_active_seconds"] = 600 + 50 * 120 + 2100 - 1
    path, digest = write_plan(world, plan)
    calls = []
    with pytest.raises(ValueError, match="active-seconds budget exhausted"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers_for(world), calls, clock=clock))
    assert calls == []
    assert not (world["out"] / "stage-discovery/ledger.jsonl").exists()


def test_output_bytes_budget_halts(runner, world, clock):
    plan = copy.deepcopy(world["plan"])
    plan["stages"]["discovery"]["budget"]["max_output_bytes"] = 1
    path, digest = write_plan(world, plan)
    calls = []
    runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                     backend_factory=fake_backend(answers_for(world), calls, clock=clock))
    with pytest.raises(ValueError, match="output-bytes budget exhausted"):
        runner.run_stage(path, digest, "discovery", world["repo"], world["out"],
                         backend_factory=fake_backend(answers_for(world), calls, clock=clock))
    assert len(calls) == 50


def gate_report(**changes):
    report = {
        "status": "completed", "unknown_after_interruption": [], "hard_stop_errors": [],
        "completed": 100, "invalid_or_unterminated_answers": 0,
        "mean_image_latency_seconds": 10.0, "mean_load_seconds": 0.0, "invocations": 2,
        "charged_active_seconds": 1000.0, "generation_seconds_total": 1000.0,
    }
    return {**report, **changes}


def test_projection_uses_measured_load_and_overhead(runner, world):
    plan = world["plan"]
    # A measured 0.0 s load is used as measured, never replaced by the 600 s allowance.
    decision = runner.discovery_decision(plan, gate_report())
    assert decision["projected_full_active_seconds"] == 33330.0
    assert decision["full_max_active_seconds"] == 66660.0
    # Charged seconds not spent generating (load, rehash) count per invocation.
    decision = runner.discovery_decision(plan, gate_report(charged_active_seconds=1300.0))
    assert decision["per_invocation_overhead_seconds"] == 150.0
    assert decision["projected_full_active_seconds"] == 33330.0 + 27 * 150.0
    decision = runner.discovery_decision(plan, gate_report(mean_load_seconds=None))
    assert decision["decision"] == "stop" and decision["full_max_active_seconds"] is None


FULL_WALL = 600 + 128 * 120 + 2100  # 18,060 s, written out independently of the runner


def ceiling_halts_before_last_chunk(latency, cap):
    # Independent replay of the hard-ceiling rule for a run costing exactly the projection
    # (zero overhead): 26 chunks of 128, then 5 images; each start needs charged + wall <= cap.
    charged = 0.0
    for size in [128] * 26 + [5]:
        if charged + FULL_WALL > cap:
            return True
        charged += size * latency
    return False


def gate_for_projection(runner, plan, projection):
    latency = projection / 3333
    return runner.discovery_decision(plan, gate_report(
        mean_image_latency_seconds=latency, generation_seconds_total=100 * latency,
        charged_active_seconds=100 * latency))


def test_accurate_projection_under_the_old_limit_that_would_halt_is_stopped(runner, world):
    # P = 160,000 s is within the old 172,800 s gate, and a run costing exactly P reaches its
    # last start with 3,328 x P / 3,333 charged: + 18,060 s wall cap exceeds 172,800 s.
    projection = 160000.0
    assert projection <= 172800
    assert ceiling_halts_before_last_chunk(projection / 3333, 172800)
    decision = gate_for_projection(runner, world["plan"], projection)
    assert decision["projected_full_active_seconds"] == pytest.approx(projection)
    assert decision["full_max_active_seconds"] == 172800
    assert decision["decision"] == "stop"
    assert any("invocation wall cap" in reason for reason in decision["reasons"])
    # A zero-latency profile: the 2x term binds far below one wall cap.
    assert ceiling_halts_before_last_chunk(0.0, 2 * 54.0)


@pytest.mark.parametrize("projection, cap, expect_stop", [
    (172800 - FULL_WALL + 1, 172800, True),      # 1 s over P + wall <= cap
    (172800 - FULL_WALL, 172800, False),         # exactly fits
    (FULL_WALL - 1, 2 * (FULL_WALL - 1), True),  # 2x term binds and is 1 s short
    (FULL_WALL, 2 * FULL_WALL, False),
])
def test_gate_boundary_is_projection_plus_one_wall_cap(runner, world, projection, cap,
                                                       expect_stop):
    decision = gate_for_projection(runner, world["plan"], projection)
    assert decision["full_max_active_seconds"] == pytest.approx(cap)
    assert (decision["decision"] == "stop") is expect_stop
    if not expect_stop:
        assert not ceiling_halts_before_last_chunk(projection / 3333, cap)


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
