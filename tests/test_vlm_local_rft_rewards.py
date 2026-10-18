"""Hierarchical partial-credit reward and RFT / expert-iteration contracts on fake data.

Nothing here loads a model or needs torch; the trainer's heavy pieces are replaced by
fakes. The released-TRL SFT round and the batched sampler are checked in
tests/test_vlm_local_training.py (CI local-vlm job).
"""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision_rl import vlm_local_training as local

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/vision_rl/aircraft_assets/label-hierarchy-train-v1.json"
LABELS = ["737-800", "737-900", "A321", "MD-11"]
HIERARCHY = {
    "737-800": {"family": "Boeing 737", "manufacturer": "Boeing"},
    "737-900": {"family": "Boeing 737", "manufacturer": "Boeing"},
    "A321": {"family": "A320", "manufacturer": "Airbus"},
    "MD-11": {"family": "MD-11", "manufacturer": "McDonnell Douglas"},
}


def write_hierarchy(path, labels=None, split="train", **overrides):
    value = {
        "schema": local.LABEL_HIERARCHY_SCHEMA,
        "source": {"split": split, "path": "fake", "sha256": "0" * 64},
        "levels": ["variant", "family", "manufacturer"],
        "labels": labels if labels is not None else HIERARCHY,
        **overrides,
    }
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def shaping(tmp_path):
    hierarchy = local.load_label_hierarchy(write_hierarchy(tmp_path / "h.json"), LABELS)
    return local.reward_shaping(hierarchy, family_credit=0.5, manufacturer_credit=0.2,
                                invalid_reward=-0.1)


def test_hierarchy_loader_accepts_train_and_refuses_val_mismatch_and_two_owners(tmp_path):
    loaded = local.load_label_hierarchy(write_hierarchy(tmp_path / "ok.json"), LABELS)
    assert loaded["family"]["737-900"] == "Boeing 737"
    assert loaded["manufacturer"]["A321"] == "Airbus"
    assert len(loaded["sha256"]) == 64
    with pytest.raises(ValueError, match="TRAIN"):
        local.load_label_hierarchy(write_hierarchy(tmp_path / "v.json", split="validation"),
                                   LABELS)
    with pytest.raises(ValueError, match="exactly the task"):
        local.load_label_hierarchy(write_hierarchy(tmp_path / "m.json"), LABELS + ["A330-200"])
    two = dict(HIERARCHY, **{"737-900": {"family": "Boeing 737", "manufacturer": "Airbus"}})
    with pytest.raises(ValueError, match="more than one manufacturer"):
        local.load_label_hierarchy(write_hierarchy(tmp_path / "t.json", labels=two), LABELS)
    with pytest.raises(ValueError, match="schema"):
        local.load_label_hierarchy(write_hierarchy(tmp_path / "s.json", schema="x"), LABELS)


def test_committed_aircraft_hierarchy_is_train_only_and_consistent():
    value = json.loads(ASSET.read_text())
    labels = sorted(value["labels"])
    loaded = local.load_label_hierarchy(ASSET, labels)
    assert value["source"]["split"] == "train"
    assert value["source"]["path"] == "data/fgvc-aircraft-2013b/audit/train-images.json"
    assert value["counts"] == {"variant": 100, "family": 70, "manufacturer": 30}
    assert len(set(loaded["family"].values())) == 70
    assert len(set(loaded["manufacturer"].values())) == 30
    # Spot values typed from the FGVC-Aircraft hierarchy, not read back from the subject.
    assert value["labels"]["737-800"] == {"family": "Boeing 737", "manufacturer": "Boeing"}
    assert value["labels"]["A321"] == {"family": "A320", "manufacturer": "Airbus"}
    assert value["labels"]["MD-11"] == {"family": "MD-11", "manufacturer": "McDonnell Douglas"}


def test_builder_refuses_non_train_and_inconsistent_records(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_hierarchy", ROOT / "scripts/build_aircraft_label_hierarchy.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    rows = [
        {"split": "train", "image_id": "1", "variant": "737-800", "family": "Boeing 737",
         "manufacturer": "Boeing"},
        {"split": "train", "image_id": "2", "variant": "A321", "family": "A320",
         "manufacturer": "Airbus"},
    ]
    built = builder.build(rows, "x.json", "0" * 64)
    assert built["counts"] == {"variant": 2, "family": 2, "manufacturer": 2}
    path = write_hierarchy(tmp_path / "b.json", labels=built["labels"])
    assert local.load_label_hierarchy(path, ["737-800", "A321"])["family"]["A321"] == "A320"
    with pytest.raises(ValueError, match="not native TRAIN"):
        builder.build(rows + [dict(rows[0], split="val", image_id="3")], "x", "0" * 64)
    with pytest.raises(ValueError, match="two family"):
        builder.build(rows + [dict(rows[0], family="Boeing 757", image_id="3")], "x", "0" * 64)
    with pytest.raises(ValueError, match="two manufacturers"):
        builder.build(rows + [{"split": "train", "image_id": "4", "variant": "A320",
                               "family": "A320", "manufacturer": "Boeing"}], "x", "0" * 64)


def test_shaped_reward_levels_and_validation(tmp_path, shaping):
    assert local.shaped_reward("737-800", "737-800", shaping) == 1.0
    assert local.shaped_reward("737-900", "737-800", shaping) == 0.5
    assert local.shaped_reward("A321", "737-800", shaping) == 0.0
    assert local.shaped_reward(None, "737-800", shaping) == -0.1
    hierarchy = local.load_label_hierarchy(write_hierarchy(tmp_path / "h2.json"), LABELS)
    same_maker = dict(HIERARCHY, **{"MD-11": {"family": "MD-11", "manufacturer": "Boeing"}})
    other = local.load_label_hierarchy(write_hierarchy(tmp_path / "h3.json", labels=same_maker),
                                       LABELS)
    shaped = local.reward_shaping(other, family_credit=0.5, manufacturer_credit=0.2)
    assert local.shaped_reward("MD-11", "737-800", shaped) == 0.2
    for bad in ({"family_credit": 1.0, "manufacturer_credit": 0.1},
                {"family_credit": 0.2, "manufacturer_credit": 0.3},
                {"family_credit": 0.5, "manufacturer_credit": -0.1},
                {"family_credit": 0.5, "manufacturer_credit": 0.1, "invalid_reward": 0.1},
                {"family_credit": float("nan"), "manufacturer_credit": 0.1}):
        with pytest.raises(ValueError):
            local.reward_shaping(hierarchy, **bad)


def test_make_reward_returns_partial_credit_but_keeps_exact_statistics(tmp_path, shaping):
    path = tmp_path / "groups.jsonl"
    reward = local.make_reward(LABELS, 4, [99], path, shaping=shaping)
    completions = ["737-800", "737-900", "Boeing 737-800", "737-800"]
    ids = [[1, 99], [2, 99], [3, 99], [1]]  # last one never reached EOS
    assert reward(completions, ["737-800"] * 4, ["x"] * 4, ids) == [1.0, 0.5, -0.1, -0.1]
    line = json.loads(path.read_text())
    assert line["rewards"] == [1.0, 0.5, -0.1, -0.1]
    assert line["exact_rewards"] == [1.0, 0.0, 0.0, 0.0]
    assert reward.stats["mixed_groups"] == 1 and reward.stats["exact_mixed_groups"] == 1
    assert reward.stats["all_correct_groups"] == 0
    # No exact answer, but family credit still separates the group: a GRPO signal that the
    # exact reward lacks.
    assert reward(["737-900", "A321", "MD-11", "x"], ["737-800"] * 4, ["y"] * 4,
                  [[1, 99]] * 4) == [0.5, 0.0, 0.0, -0.1]
    assert reward.stats["mixed_groups"] == 2 and reward.stats["exact_mixed_groups"] == 1
    assert reward.stats["all_zero_groups"] == 1
    assert reward.stats["shaped_reward_sum"] == pytest.approx(1.0 + 0.5 - 0.2 + 0.5 - 0.1)


def test_legacy_reward_journal_and_stats_keys_are_unchanged(tmp_path):
    path = tmp_path / "groups.jsonl"
    reward = local.make_reward(["blue", "red"], 2, [99], path)
    assert reward(["blue", "red"], ["blue"] * 2, ["i"] * 2, [[99]] * 2) == [1.0, 0.0]
    assert list(json.loads(path.read_text())) == [
        "id", "label", "rewards", "mixed", "recognition_mixed", "canonical_predictions",
        "completions", "completion_ids", "completion_tokens", "terminated",
        "complete_think_block",
    ]
    assert "exact_mixed_groups" not in reward.stats
    assert "shaped_reward_sum" not in reward.stats


def _refused_settings(diagnostic, tmp_path, monkeypatch, name, **kwargs):
    monkeypatch.setattr(
        local, "preflight",
        lambda *_: {"adapter_source_sha256": local._file_sha256(Path(local.__file__))},
    )
    output = tmp_path / name
    with pytest.raises((FileNotFoundError, ImportError, ModuleNotFoundError)):
        local.train_local(diagnostic / "train.jsonl", None, diagnostic / "data_spec.json",
                          output, model_directory=tmp_path / "absent", device="cpu",
                          max_steps=4, train_only=True, **kwargs)
    return json.loads((output / "local_training.json").read_text())


@pytest.fixture
def diagnostic(tmp_path):
    root = tmp_path / "data"
    local.make_probe_data(root)
    return root


def color_hierarchy(tmp_path):
    return write_hierarchy(tmp_path / "colors.json", labels={
        "blue": {"family": "cool", "manufacturer": "paint"},
        "red": {"family": "warm", "manufacturer": "paint"},
    })


def test_train_local_shaping_enters_identity_only_when_used(diagnostic, tmp_path, monkeypatch):
    plain = _refused_settings(diagnostic, tmp_path, monkeypatch, "plain")
    assert "reward_shaping" not in plain["settings"] and "torch_threads" not in plain
    shaped = _refused_settings(
        diagnostic, tmp_path, monkeypatch, "shaped", label_hierarchy=color_hierarchy(tmp_path),
        family_credit=0.5, manufacturer_credit=0.25,
    )
    assert shaped["settings"]["reward_shaping"] == {
        "kind": "hierarchical-partial-credit-v1",
        "hierarchy_sha256": local._file_sha256(tmp_path / "colors.json"),
        "family_credit": 0.5,
        "manufacturer_credit": 0.25,
        "invalid_reward": 0.0,
    }
    assert {k: v for k, v in shaped["settings"].items() if k != "reward_shaping"} == plain[
        "settings"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"family_credit": 0.5, "manufacturer_credit": 0.2}, "require label_hierarchy"),
        ({"invalid_reward": -0.1}, "require label_hierarchy"),
        ({"label_hierarchy": "H"}, "explicit family_credit"),
        ({"label_hierarchy": "H", "family_credit": 0.5, "manufacturer_credit": 0.2,
          "mode": "sft"}, "only to GRPO"),
        ({"torch_threads": 0}, "torch_threads"),
    ],
)
def test_shaping_and_thread_options_refuse_before_any_output(
    diagnostic, tmp_path, monkeypatch, kwargs, match
):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must refuse first"))
    if kwargs.get("label_hierarchy") == "H":
        kwargs["label_hierarchy"] = color_hierarchy(tmp_path)
    with pytest.raises(ValueError, match=match):
        local.train_local(diagnostic / "train.jsonl", None, diagnostic / "data_spec.json",
                          tmp_path / "out", model_directory=tmp_path / "m", device="cpu",
                          max_steps=4, train_only=True, **kwargs)
    assert not (tmp_path / "out").exists()


def test_rft_prompt_slices_are_seeded_cover_the_pool_and_reshuffle_each_pass():
    ids = [f"r{i}" for i in range(10)]
    first = [local.rft_round_prompt_ids(ids, seed=7, round_index=r, per_round=5)
             for r in range(4)]
    assert sorted(first[0] + first[1]) == sorted(ids)
    assert sorted(first[2] + first[3]) == sorted(ids)
    assert first[0] + first[1] != first[2] + first[3]
    assert first[0] == local.rft_round_prompt_ids(ids, seed=7, round_index=0, per_round=5)
    assert first[0] != local.rft_round_prompt_ids(ids, seed=8, round_index=0, per_round=5)
    for bad in (0, 11):
        with pytest.raises(ValueError):
            local.rft_round_prompt_ids(ids, seed=7, round_index=0, per_round=bad)


def sample(text, terminated=True):
    return {"response": text, "terminated": terminated}


def test_rft_select_keeps_only_distinct_terminated_exact_answers():
    samples = [sample("blue"), sample("blue", terminated=False), sample("red"),
               sample("<think>grid</think>\nblue"), sample("blue")]
    kept, stats = local.rft_select(samples, "blue", ["blue", "red"], max_keep=5)
    assert kept == ["blue", "<think>grid</think>\nblue"]
    assert stats["correct"] == 3 and stats["pass_rate"] == 3 / 5 and stats["kept"] == 2
    kept, _ = local.rft_select(samples, "blue", ["blue", "red"], max_keep=1)
    assert kept == ["blue"]
    easy = [sample("blue")] * 4
    kept, stats = local.rft_select(easy, "blue", ["blue", "red"], max_keep=1,
                                   skip_all_correct=True)
    assert kept == [] and stats["skipped_all_correct"] is True
    kept, _ = local.rft_select(easy, "blue", ["blue", "red"], max_keep=1)
    assert kept == ["blue"]
    kept, stats = local.rft_select([sample("red")] * 2, "blue", ["blue", "red"], max_keep=1)
    assert kept == [] and stats["correct"] == 0


def test_rft_records_train_on_the_policy_sample_not_the_gold_label(diagnostic):
    spec = local.load_spec(diagnostic / "data_spec.json")
    rows = {"a": {"id": "a", "image": "/x.png", "label": "blue"}}
    records = local.rft_records({"a": ["<think>more blue</think>\nblue", "blue"]}, rows, spec,
                                "think_then_answer")
    assert [r["completion"][0]["content"] for r in records] == [
        "<think>more blue</think>\nblue", "blue"]
    assert [r["id"] for r in records] == ["a#0", "a#1"]
    assert records[0]["prompt"][0]["content"] == local.prompt_text(spec, "think_then_answer")


class FakePolicy:
    """Stands in for a PEFT VLM: an integer 'weight' that each SFT update increments."""

    def __init__(self):
        self.weight = 0
        self.config = SimpleNamespace(model_type="qwen2_vl", image_token_id=5)

    def to(self, device):
        return self

    def modules(self):
        return []

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_text(f"weight={self.weight}")


def fake_stack(monkeypatch, policies):
    class Dropout:
        pass

    def from_pretrained(*_, **__):
        policies.append(FakePolicy())
        return policies[-1]

    modules = {
        "torch": SimpleNamespace(set_num_threads=lambda n: None, float32="f32",
                                 nn=SimpleNamespace(Dropout=Dropout)),
        "transformers": SimpleNamespace(
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=from_pretrained),
            set_seed=lambda seed: None,
        ),
        "peft": SimpleNamespace(get_peft_model=lambda policy, config: policy,
                                LoraConfig=lambda **kw: kw),
    }
    monkeypatch.setattr(local, "import_module", lambda name: modules[name])
    monkeypatch.setattr(local, "preflight", lambda *_: {
        "adapter_source_sha256": local._file_sha256(Path(local.__file__))})
    monkeypatch.setattr(local, "_verified_model_directory", lambda *_: {"fake": True})
    monkeypatch.setattr(local, "_processor", lambda *_: SimpleNamespace(
        save_pretrained=lambda path: None))
    monkeypatch.setattr(local, "_audit_processor", lambda *_: {"fake": True})
    monkeypatch.setattr(local, "_task_group_generation_config", lambda *_: SimpleNamespace(
        eos_token_id=[99], to_dict=lambda: {"fake": True}))

    def load_round(policy, adapter):
        policy.weight = int((adapter / "adapter_model.safetensors").read_text().split("=")[1])

    monkeypatch.setattr(local, "_load_round_adapter", load_round)


def make_sampler(calls, labels_by_id):
    """K=4 draws: the first `correct` are the gold label, the rest the other color."""

    def sampler(chosen, round_dir, reward):
        calls.append([row["id"] for row in chosen])
        groups = {}
        for row in chosen:
            other = "red" if row["label"] == "blue" else "blue"
            correct = int(row["id"].split("-")[1]) % 5  # 0..4 correct of 4 (4 = all)
            texts = [row["label"]] * min(correct, 4) + [other] * (4 - min(correct, 4))
            groups[row["id"]] = [sample(text) for text in texts]
            reward(completions=[s["response"] for s in groups[row["id"]]],
                   completion_ids=[[1, 99]] * 4, label=[row["label"]] * 4,
                   id=[row["id"]] * 4)
        return groups

    return sampler


def make_trainer(policies, seen):
    def train_round(records, round_dir, steps, round_seed):
        seen.append({"records": records, "steps": steps, "seed": round_seed})
        policies[-1].weight += steps
        return {"optimizer_steps": steps, "planned_steps": steps,
                "stopped_by_time_budget": False}

    return train_round


def run_rft(diagnostic, output, policies, calls, seen, **kwargs):
    defaults = dict(model_directory=diagnostic / "model", rounds=3, prompts_per_round=4,
                    profile=replace(local.Profile(), response_format="answer_only"),
                    seed=3)
    defaults.update(kwargs)
    return local.rft_local(
        diagnostic / "train.jsonl", diagnostic / "data_spec.json", output,
        _sample=make_sampler(calls, None), _train_round=make_trainer(policies, seen),
        **defaults,
    )


def test_rft_rounds_keep_correct_samples_train_on_them_and_bind_the_endpoint(
    diagnostic, tmp_path, monkeypatch
):
    policies, calls, seen = [], [], []
    fake_stack(monkeypatch, policies)
    record = run_rft(diagnostic, tmp_path / "rft", policies, calls, seen)
    assert record["status"] == "completed" and record["stop_reason"] == "rounds"
    assert [len(c) for c in calls] == [4, 4, 4]
    assert calls[0] + calls[1] != []  # rounds sample through the seeded permutation
    expected_kept = [sum(int(i.split("-")[1]) % 5 > 0 for i in round_ids) for round_ids in calls]
    assert [len(s["records"]) for s in seen if s] == [k for k in expected_kept if k]
    for entry in seen:
        for item in entry["records"]:
            assert item["completion"][0]["content"] == item["label"]
    assert record["optimizer_steps"] == sum(expected_kept) == policies[-1].weight
    final = tmp_path / "rft" / "final_adapter" / "adapter_model.safetensors"
    assert final.read_text() == f"weight={sum(expected_kept)}"
    saved = json.loads((tmp_path / "rft" / "local_training.json").read_text())
    assert saved["final_adapter_sha256"] == local._file_sha256(final)
    assert saved["control_identity"]["validation"] is None
    assert saved["settings"]["mode"] == "rft" and saved["claim_scope"] == local.RFT_SCOPE
    assert [r["round"] for r in saved["rounds"]] == [0, 1, 2]
    assert saved["rounds"][0]["prompt_ids"] == calls[0]
    assert saved["rounds"][0]["reward_audit"]["sampled_completions"] == 16


def test_rft_max_steps_cap_truncates_the_last_round_and_stops(diagnostic, tmp_path,
                                                              monkeypatch):
    policies, calls, seen = [], [], []
    fake_stack(monkeypatch, policies)
    record = run_rft(diagnostic, tmp_path / "rft", policies, calls, seen, rounds=10,
                     epochs_per_round=3, max_steps=7)
    assert record["optimizer_steps"] == 7 and record["stop_reason"] == "max_steps"
    assert sum(entry["steps"] for entry in seen) == 7


def test_rft_resume_continues_from_committed_round_and_discards_partial(
    diagnostic, tmp_path, monkeypatch
):
    policies, calls, seen = [], [], []
    fake_stack(monkeypatch, policies)
    output = tmp_path / "rft"
    stopped = run_rft(diagnostic, output, policies, calls, seen, stop_after_rounds=1)
    assert stopped["status"] == "stopped" and not (output / "final_adapter").exists()
    weight_after_one = policies[-1].weight
    (output / "round-01").mkdir()  # an uncommitted, interrupted round
    (output / "round-01" / "junk").write_text("partial")
    resumed = run_rft(diagnostic, output, policies, calls, seen)
    assert (output / "round-01.discarded-0" / "junk").is_file()
    assert resumed["status"] == "completed" and len(resumed["rounds"]) == 3
    # The fresh policy object was restored from round 0's committed adapter, then trained.
    assert policies[-1].weight == resumed["optimizer_steps"] >= weight_after_one
    uninterrupted_calls, uninterrupted_seen, others = [], [], []
    fake_stack(monkeypatch, others)
    straight = run_rft(diagnostic, tmp_path / "straight", others, uninterrupted_calls,
                       uninterrupted_seen)
    assert calls == uninterrupted_calls
    assert [r["prompt_ids"] for r in resumed["rounds"]] == [
        r["prompt_ids"] for r in straight["rounds"]]
    assert [s["seed"] for s in seen] == [s["seed"] for s in uninterrupted_seen]
    with pytest.raises(local.LocalVLMError, match="already completed"):
        run_rft(diagnostic, output, policies, calls, seen)
    with pytest.raises(ValueError, match="preserve"):
        run_rft(diagnostic, tmp_path / "straight", others, [], [], prompts_per_round=3)


def test_rft_skip_all_correct_drops_solved_prompts(diagnostic, tmp_path, monkeypatch):
    policies, calls, seen = [], [], []
    fake_stack(monkeypatch, policies)
    record = run_rft(diagnostic, tmp_path / "rft", policies, calls, seen,
                     skip_all_correct=True, rounds=2)
    kept_ids = {item["id"].split("#")[0] for entry in seen for item in entry["records"]}
    assert all(int(i.split("-")[1]) % 5 not in (0, 4) for i in kept_ids)
    assert record["settings"]["skip_all_correct"] is True


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rounds": 0}, "rounds"),
        ({"prompts_per_round": 9}, "exceeds"),
        ({"max_steps": 0}, "max_steps"),
        ({"stop_after_rounds": 4}, "stop_after_rounds"),
        ({"epochs_per_round": 0}, "epochs_per_round"),
        ({"torch_threads": 999}, "torch_threads"),
    ],
)
def test_rft_refuses_bad_options_before_any_output(diagnostic, tmp_path, monkeypatch, kwargs,
                                                   match):
    monkeypatch.setattr(local, "preflight", lambda *_: pytest.fail("must refuse first"))
    with pytest.raises(ValueError, match=match):
        run_rft(diagnostic, tmp_path / "out", [], [], [], **kwargs)
    assert not (tmp_path / "out").exists()


def test_rft_refuses_validation_manifest(diagnostic, tmp_path):
    with pytest.raises(ValueError, match="expected train split"):
        local.rft_local(diagnostic / "validation.jsonl", diagnostic / "data_spec.json",
                        tmp_path / "o", model_directory=tmp_path / "m", rounds=1,
                        prompts_per_round=1)


def test_cli_routes_rft_and_keeps_legacy_max_steps_default(monkeypatch, capsys, tmp_path):
    captured = {}
    def capture(name):
        def record(*_, **kwargs):
            captured[name] = kwargs
            return {"ok": 1}

        return record

    monkeypatch.setattr(local, "rft_local", capture("rft"))
    monkeypatch.setattr(local, "train_local", capture("train"))
    base = ["--train", "t.jsonl", "--data-spec", "s.json", "--output", "o", "--device", "cpu"]
    assert local.main(base + ["--mode", "rft", "--model-directory", "m", "--rft-rounds", "2",
                              "--rft-prompts-per-round", "8", "--torch-threads", "16"]) == 0
    rft = captured["rft"]
    assert (rft["rounds"], rft["prompts_per_round"], rft["max_steps"]) == (2, 8, None)
    assert (rft["epochs_per_round"], rft["max_keep_per_prompt"]) == (1, 1)
    assert rft["torch_threads"] == 16
    assert local.main(base + ["--train-only"]) == 0
    train = captured["train"]
    assert train["max_steps"] == 64 and train["torch_threads"] == 2
    assert train["label_hierarchy"] is None and train["family_credit"] is None
    for bad in (base + ["--rft-rounds", "2", "--train-only"],
                base + ["--mode", "rft", "--model-directory", "m", "--rft-rounds", "1",
                        "--rft-prompts-per-round", "1", "--validation", "v.jsonl"],
                base + ["--mode", "rft", "--model-directory", "m", "--rft-rounds", "1",
                        "--rft-prompts-per-round", "1", "--beta", "0"],
                base + ["--mode", "rft", "--model-directory", "m", "--rft-rounds", "1"],
                base + ["--mode", "rft", "--model-directory", "m", "--rft-rounds", "1",
                        "--rft-prompts-per-round", "1", "--label-hierarchy", "h.json"]):
        with pytest.raises(SystemExit):
            local.main(bad)
