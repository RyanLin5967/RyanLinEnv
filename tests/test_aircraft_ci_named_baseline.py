"""CPU-only tests for the CI named-baselines runner. The model is a fake backend.

The shard loop, journal, record, merge, parser re-check, scoring and refusals run for real
over the pre-registered 100-image discovery manifest with synthetic JPEGs.
"""

import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest
from PIL import Image

from vision_rl import ci_shards
from vision_rl.aircraft_data import aircraft_pixel_sha256

REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = REPOSITORY / "docs/aircraft-ci-named-baselines-plan-v1.json"
INPUTS = REPOSITORY / "docs/aircraft-ci-named-baseline-v1/inputs/fgvc-aircraft-2013b"
MODEL = "qwen2-vl-2b-instruct"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location(
        "aircraft_ci_named_baseline", REPOSITORY / "scripts/aircraft_ci_named_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def world(tmp_path, monkeypatch, runner):
    plan = json.loads(PLAN.read_text())
    ids = plan["stages"]["discovery"]["ids"]
    gold = {}
    for line in (INPUTS / "development/validation.jsonl").read_text().splitlines():
        row = json.loads(line)
        gold[row["id"]] = row["label"]
    labels = json.loads((INPUTS / "candidates/fourshot-seed791/data_spec.json").read_text())[
        "labels"
    ]
    repo = tmp_path / "repo"
    images = repo / "data/fgvc-aircraft-2013b/development/images/validation"
    images.mkdir(parents=True)
    author = []
    for number, identifier in enumerate(ids):
        path = images / f"{identifier}.jpg"
        Image.new("RGB", (64, 64 + 20), (number, 100, 200)).save(path, quality=90)
        author.append(
            {
                "image_id": identifier,
                "encoded_sha256": sha(path),
                "model_rgb_sha256": aircraft_pixel_sha256(path),
            }
        )
    (repo / "data/fgvc-aircraft-2013b/audit").mkdir(parents=True)
    (repo / "data/fgvc-aircraft-2013b/audit/val-images.json").write_text(json.dumps(author))
    rows = [{"id": i, "image": str(images / f"{i}.jpg"), "split": "validation"} for i in ids]
    inputs = {"rows": rows, "gold": [{"id": i, "label": gold[i]} for i in ids], "labels": labels}
    stub = types.SimpleNamespace(
        stage_inputs=lambda plan, stage, repository: inputs,
        hoeffding_half_width=lambda n, c: 0.1358,
    )
    monkeypatch.setattr(runner, "named_baseline", lambda: stub)
    monkeypatch.setattr(runner, "verify_model_directory", lambda *a: None)
    activated = dict(plan, activated=True, activation={"activated_at": "t", "approved_by": "r"})
    plan_path = tmp_path / "activated.json"
    plan_path.write_text(json.dumps(activated))
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    return types.SimpleNamespace(
        plan=activated,
        plan_path=plan_path,
        plan_sha=sha(plan_path),
        ids=ids,
        gold=gold,
        labels=labels,
        inputs=inputs,
        repo=repo,
        out=tmp_path / "out",
    )


def correct(identifier, ids):
    # Fake model rule, written independently of the runner: every third image is answered
    # correctly; every fifth is cut off (unterminated); the rest answer gibberish.
    return ids.index(identifier) % 3 == 0 and ids.index(identifier) % 5 != 0


class FakeBackend:
    def __init__(self, world):
        self.world = world
        self.provenance = {"model": MODEL, "fake": True}

    def predict(self, row):
        ids, index = self.world.ids, self.world.ids.index(row["id"])
        answer = self.world.gold[row["id"]] if correct(row["id"], ids) else "not an aircraft"
        return {
            "response": answer,
            "raw_response": answer,
            "terminated": index % 5 != 0,
            "completion_ids": [1, 2, 3],
            "prompt_tokens": 10,
        }

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.5
        return self.now


def run_all(runner, world, shards=(0, 1, 2, 3)):
    for shard in shards:
        runner.run_shard(
            world.plan_path,
            world.plan_sha,
            MODEL,
            "discovery",
            shard,
            4,
            world.repo / "m",
            world.out,
            world.repo,
            backend_factory=lambda *a: FakeBackend(world),
            monotonic=Clock(),
        )


def merge(runner, world):
    return runner.merge(
        world.plan, world.plan_sha, MODEL, "discovery", 4, world.out, world.inputs, world.repo
    )


def test_committed_plan_is_unactivated_and_refused(runner):
    plan = json.loads(PLAN.read_text())
    assert plan["activated"] is False and plan["activation"] is None
    with pytest.raises(ValueError, match="unactivated"):
        runner.load_plan(PLAN, sha(PLAN))
    code = runner.main(
        [
            "merge",
            "--plan",
            str(PLAN),
            "--expected-plan-sha256",
            sha(PLAN),
            "--model",
            MODEL,
            "--stage",
            "discovery",
            "--shards",
            "4",
            "--records",
            "x",
            "--output",
            "y",
        ]
    )
    assert code == 2


def test_committed_inputs_and_endpoints_match_their_pins(runner):
    plan = json.loads(PLAN.read_text())
    for name in ("val_images", "validation", "data_spec"):
        assert sha(INPUTS / plan["inputs"][name]["path"]) == plan["inputs"][name]["sha256"]
    endpoints = plan["analysis"]["endpoint_predictions"]
    assert sha(REPOSITORY / endpoints["path"]) == endpoints["sha256"]
    assert runner.stage_order(plan, "full", REPOSITORY)[:2] == ["0034309", "0034958"]
    assert len(runner.stage_order(plan, "discovery", REPOSITORY)) == 100


def test_full_shards_score_every_image_once(runner, world):
    run_all(runner, world)
    report = merge(runner, world)
    expected_correct = sum(correct(i, world.ids) for i in world.ids)
    # One image per class, so equal-class accuracy equals the fraction correct.
    assert report["metric"]["correct"] == expected_correct
    assert report["metric"]["mean_class_accuracy"] == pytest.approx(expected_correct / 100)
    assert report["unterminated"] == 20
    assert report["n"] == 100 and list(report["predictions"]) == world.ids
    # Clock ticks 0.5 s per call; each image spans exactly one tick between its reads.
    assert report["latencies_seconds"] == [0.5] * 100
    assert report["decision"]["decision"] == "proceed"


def test_missing_shard_is_refused(runner, world):
    run_all(runner, world, shards=(0, 1, 3))
    with pytest.raises(ci_shards.ShardGapError, match=r"missing shard\(s\) \[2\] of 4"):
        merge(runner, world)


def test_missing_shard_refused_by_cli(runner, world, tmp_path, monkeypatch):
    run_all(runner, world, shards=(0, 1, 3))
    monkeypatch.setattr(runner, "load_plan", lambda *a, **k: world.plan)
    output = tmp_path / "merged.json"
    code = runner.main(
        [
            "merge",
            "--plan",
            str(world.plan_path),
            "--expected-plan-sha256",
            world.plan_sha,
            "--model",
            MODEL,
            "--stage",
            "discovery",
            "--shards",
            "4",
            "--records",
            str(world.out),
            "--output",
            str(output),
            "--repository",
            str(world.repo),
        ]
    )
    assert code == 2 and not output.exists()


def test_tampered_label_is_refused(runner, world):
    run_all(runner, world)
    path = world.out / "shard-000.json"
    record = json.loads(path.read_text())
    first = world.ids[1]
    record["results"][first]["canonical_label"] = world.gold[first]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs from the frozen parser"):
        merge(runner, world)


def test_rerun_of_one_shard_in_another_attempt_is_refused(runner, world, monkeypatch):
    run_all(runner, world, shards=(0, 1, 2))
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    run_all(runner, world, shards=(3,))
    with pytest.raises(ci_shards.ShardGapError, match="run_attempt"):
        merge(runner, world)


def test_shard_refuses_wrong_count_and_reuse(runner, world):
    with pytest.raises(ValueError, match="pre-registered 4"):
        runner.run_shard(
            world.plan_path,
            world.plan_sha,
            MODEL,
            "discovery",
            0,
            5,
            world.repo / "m",
            world.out,
            world.repo,
            backend_factory=lambda *a: FakeBackend(world),
        )
    run_all(runner, world, shards=(0,))
    with pytest.raises(ValueError, match="prior output exists"):
        run_all(runner, world, shards=(0,))


def test_shard_outside_actions_is_refused(runner, world, monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID")
    with pytest.raises(ValueError, match="GitHub Actions"):
        run_all(runner, world, shards=(0,))


def test_wall_cap_stops_a_shard_without_a_record(runner, world):
    world.plan["execution"]["shard_wall_seconds"] = 600 + 3
    world.plan_path.write_text(json.dumps(world.plan))
    world.plan_sha = sha(world.plan_path)
    with pytest.raises(ValueError, match="partial_wall_cap"):
        run_all(runner, world, shards=(0,))
    assert not (world.out / "shard-000.json").exists()
    assert (world.out / "shard-000.journal.jsonl").exists()


def test_full_stage_needs_committed_discovery(runner, world):
    with pytest.raises(ValueError, match="discovery report is not committed"):
        runner.expected_shards(world.plan, MODEL, "full", world.repo)
