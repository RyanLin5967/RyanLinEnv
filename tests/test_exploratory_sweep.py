"""Exploratory sweep v2 driver logic: bootstrap statistics, promotion, controls, scoring.

Pure-Python parts only (no model, no UW host): the runner's paired bootstrap and its
successive-halving promotion, matched-control construction, and arm_exec's scoring and
holdout guard. Expected values are computed by hand here, never by the subject.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "exploratory_sweep" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("runner")
arm_exec = load("arm_exec")


def test_binomial_draws_match_moments():
    binom = runner.Binomial(random.Random(7))
    n, p, draws = 800, 0.1, 20000
    xs = [binom.draw(n, p) for _ in range(draws)]
    mean = sum(xs) / draws
    var = sum((x - mean) ** 2 for x in xs) / draws
    assert abs(mean - 80) < 0.3  # sd of the mean ~ sqrt(72/20000) = 0.06
    assert abs(var - 72) < 4
    assert binom.draw(0, 0.5) == 0 and binom.draw(5, 0.0) == 0 and binom.draw(5, 1.0) == 5


def test_paired_bootstrap_point_and_planted_cases():
    gold = {f"i{k}": "A" for k in range(100)}
    a = {k: "A" for k in gold}
    b = dict(a)
    for k in list(gold)[:30]:
        b[k] = "B"  # b wrong on 30 items a gets right
    for k in list(gold)[30:40]:
        a[k] = None  # a wrong on 10 items b gets right
    out = runner.paired_bootstrap(a, b, gold, 4000, seed=1)
    assert out["delta"] == pytest.approx((30 - 10) / 100)
    assert out["wins"] == 30 and out["losses"] == 10
    assert out["ci95"][0] > 0 and out["ci95"][0] < 0.2 < out["ci95"][1]
    same = runner.paired_bootstrap(a, dict(a), gold, 1000, seed=1)
    assert same["delta"] == 0 and same["ci95"] == [0, 0] and same["p_le_0"] == 1.0
    assert runner.paired_bootstrap(a, b, gold, 1000, seed=3) == runner.paired_bootstrap(
        a, b, gold, 1000, seed=3
    )


def test_arm_exec_score_is_equal_class_mean():
    gold = {"1": "x", "2": "x", "3": "x", "4": "y"}
    pred = {"1": "x", "2": None, "3": "y", "4": "y"}
    m = arm_exec.score(pred, gold)
    assert m["accuracy"] == 0.5
    assert m["valid_fraction"] == 0.75
    assert m["equal_class_accuracy"] == pytest.approx((1 / 3 + 1) / 2)


def test_arm_exec_refuses_holdout_training_manifest(tmp_path):
    (tmp_path / "RECEIPT.json").write_text("{}")
    for name in ("dev800.jsonl", "h_gate_final.jsonl", "h_select.jsonl", "../x.jsonl"):
        arm = arm_exec.Arm(tmp_path, {"id": "a", "train_manifest": name}, tmp_path / "r")
        with pytest.raises(SystemExit, match="not a training manifest"):
            arm.manifest()


def make_table(arms):
    return {
        "study_id": runner.STUDY,
        "rung_seeds": {"1": [42], "2": [42, 314, 2718], "3": [42, 314, 2718]},
        "rung_updates": {"1": 150, "2": 600, "3": 1500},
        "control_lr": {"W": 1e-5, "I": 5e-6},
        "arms": arms,
    }


def base_arm(arm_id, track):
    return {
        "id": arm_id, "track": track, "rung": 1, "seed": 42, "init": "W", "mode": "grpo",
        "lr": 1e-5, "stops": [150], "max_steps_total": 1500, "wave": 1,
        "train_manifest": "s_train_scan_seed42.jsonl", "features": [],
    }


def finish(tmp_path, state, arm_id, accuracy, steps=150, ids=None):
    res = tmp_path / "results" / arm_id
    res.mkdir(parents=True)
    metrics = {"accuracy": accuracy, "equal_class_accuracy": accuracy, "valid_fraction": 1.0}
    (res / "summary.json").write_text(json.dumps({"endpoints": [{
        "label": f"step{steps}", "steps": steps, "metrics": metrics,
        "active_seconds": 123.0}]}))
    ids = ids or [f"p{k}" for k in range(steps)]
    (res / "endpoint_receipt.json").write_text(json.dumps(
        {"global_step": steps, "committed_prompt_ids": ids}))
    state["arms"][arm_id] = {"status": "DONE", "results_dir": str(res), "host": "h1"}


def setup(tmp_path, monkeypatch, state):
    monkeypatch.setattr(runner, "SWEEP_DIR", tmp_path / "sweep")
    monkeypatch.setattr(runner, "DESIGN", tmp_path / "design")
    (tmp_path / "sweep").mkdir()
    (tmp_path / "design").mkdir()
    path = tmp_path / "sweep" / "state.json"
    path.write_text(json.dumps(state))
    return path


def test_promote_keeps_top_third_per_track_and_continues_seed42(tmp_path, monkeypatch):
    rl = [base_arm(f"rl{k}", "rl") for k in range(6)]
    sft = [dict(base_arm(f"sft{k}", "sft"), mode="sft") for k in range(3)]
    gated = dict(base_arm("w2-never", "rl"), wave=2)
    state = {"arms": {}, "hosts": {}}
    for k, arm in enumerate(rl):
        finish(tmp_path, state, arm["id"], 0.60 + k / 100)  # rl5 best, rl4 second
    for k, arm in enumerate(sft):
        finish(tmp_path, state, arm["id"], 0.70 - k / 100)  # sft0 best
    path = setup(tmp_path, monkeypatch, state)
    (tmp_path / "design" / "arms-rung1.json").write_text(
        json.dumps(make_table(rl + sft + [gated])))

    class A:
        from_rung, late, overwrite = 1, False, False

    assert runner.cmd_promote(A, path) == 0
    out = json.loads((tmp_path / "sweep" / "arms-rung2.json").read_text())
    assert out["promotion"] == {"sft": ["sft0"], "rl": ["rl5", "rl4"]}
    ids = [a["id"] for a in out["arms"]]
    assert "rl5@r2-s42" in ids and "rl5@r2-s42~ctl" in ids and "sft0@r2-s42~ctl" not in ids
    cont = next(a for a in out["arms"] if a["id"] == "rl5@r2-s42")
    assert (cont["work_id"], cont["resume_stop"], cont["same_host_as"]) == ("rl5", 150, "rl5")
    assert cont["stops"] == [600]
    fresh = next(a for a in out["arms"] if a["id"] == "rl5@r2-s314")
    assert fresh["train_manifest"] == "s_train_scan_seed314.jsonl"
    assert fresh["work_id"] == "rl5@r2-s314" and "resume_stop" not in fresh
    saved = json.loads(path.read_text())["promotions"]["1->2"]
    assert saved["not_entered"] == ["w2-never"]

    # A late wave-2 unit that beats the lowest promoted rl unit (rl4, 0.64) joins.
    state = json.loads(path.read_text())
    finish(tmp_path, state, "w2-never", 0.65)
    path.write_text(json.dumps(state))

    class Late(A):
        late = True

    assert runner.cmd_promote(Late, path) == 0
    out = json.loads((tmp_path / "sweep" / "arms-rung2.json").read_text())
    assert "w2-never@r2-s2718" in [a["id"] for a in out["arms"]]


def test_promote_refuses_while_a_launched_arm_runs(tmp_path, monkeypatch):
    arms = [base_arm("rl0", "rl"), base_arm("rl1", "rl")]
    state = {"arms": {"rl1": {"status": "RUNNING"}}, "hosts": {}}
    finish(tmp_path, state, "rl0", 0.6)
    path = setup(tmp_path, monkeypatch, state)
    (tmp_path / "design" / "arms-rung1.json").write_text(json.dumps(make_table(arms)))

    class A:
        from_rung, late, overwrite = 1, False, False

    with pytest.raises(SystemExit, match="unfinished"):
        runner.cmd_promote(A, path)


def test_control_spec_matches_consumed_sequence_and_time(tmp_path):
    state = {"arms": {}}
    ids = ["b", "a", "b", "c"]
    finish(tmp_path, state, "rl", 0.6, steps=4, ids=ids)
    ctl = {"id": "rl~ctl", "control_of": "rl", "max_steps_total": 1500}
    spec = runner.control_spec(state, ctl)
    assert spec["train_ids"] == ids and spec["stops"] == [4]
    assert spec["time_budget_seconds"] == 123.0
    with pytest.raises(SystemExit, match="needs a finished"):
        runner.control_spec({"arms": {}}, ctl)
