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


def _save_steps(argv):
    return argv[argv.index("--save-steps") + 1]


def test_arm_exec_save_steps_is_constant_across_stages(tmp_path):
    # save_steps is in the trainer's resume identity: every stage of one run must pass the same
    base = {"id": "a", "init": "W", "mode": "sft", "lr": 1e-5, "seed": 42,
            "response_format": "answer_only", "num_generations": 4, "min_pixels": 1,
            "max_pixels": 2, "max_completion_length": 8, "max_active_seconds": 100.0}
    m = tmp_path / "m.jsonl"
    two = arm_exec.Arm(tmp_path, dict(base, stops=[2, 4]), tmp_path / "r")
    first = two.trainer_args(m, 10, 2, None)
    second = two.trainer_args(m, 10, 4, tmp_path / "checkpoint-2")
    assert _save_steps(first) == _save_steps(second) == "2"
    promoted = arm_exec.Arm(tmp_path, dict(base, stops=[600], resume_stop=150), tmp_path / "r")
    assert _save_steps(promoted.trainer_args(m, 1500, 600, tmp_path / "c")) == "150"
    epochs = arm_exec.Arm(tmp_path, dict(base, stops=[2329, 4658]), tmp_path / "r")
    assert {_save_steps(epochs.trainer_args(m, 4658, k, None)) for k in (2329, 4658)} == {"2329"}
    ctl = arm_exec.Arm(tmp_path, dict(base, stops=[150]), tmp_path / "r")
    step = ctl.trainer_args(m, 1500, 150, None)
    timed = ctl.trainer_args(m, 1500, None, tmp_path / "c", stop_seconds=50.0)
    assert _save_steps(step) == _save_steps(timed) == "150"


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


def test_promote_records_screened_unit_infeasible_at_rung3(tmp_path, monkeypatch):
    # plan amendment 2: a screened unit declares 600 updates, so it cannot run rung 3
    arms = []
    for unit, total, acc in (("scr", 600, 0.80), ("plain", 1500, 0.70)):
        for seed in (42, 314, 2718):
            a = dict(base_arm(f"{unit}@r2-s{seed}", "rl"), unit=unit, rung=2, seed=seed,
                     stops=[600], max_steps_total=total)
            arms.append(a)
    state = {"arms": {}, "hosts": {}}
    for a in arms:
        finish(tmp_path, state, a["id"], 0.80 if a["unit"] == "scr" else 0.70, steps=600)
    path = setup(tmp_path, monkeypatch, state)
    (tmp_path / "sweep" / "arms-rung2.json").write_text(json.dumps(make_table(arms)))

    class A:
        from_rung, late, overwrite = 2, False, False

    assert runner.cmd_promote(A, path) == 0
    out = json.loads((tmp_path / "sweep" / "arms-rung3.json").read_text())
    assert out["promotion"]["rl"] == ["scr", "plain"]
    assert out["infeasible"] == ["scr"]
    ids = [a["id"] for a in out["arms"]]
    assert not [i for i in ids if i.startswith("scr")]
    assert "plain@r3-s42" in ids and "plain@r3-s42~ctl" in ids
    assert json.loads(path.read_text())["promotions"]["2->3"]["infeasible"] == ["scr"]


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


def test_poll_finds_summary_under_rsync_subpath_semantics(tmp_path, monkeypatch):
    """`uwrun fetch NAME SUB DEST` is rsync: SUB without a trailing slash lands in
    DEST/<basename>. The fake below copies the way rsync does, so a fetch of
    `work/results` (no slash) leaves summary.json one level down and the arm FAILS."""
    import shutil

    remote = tmp_path / "remote" / "work" / "results"
    remote.mkdir(parents=True)
    (remote / "summary.json").write_text(json.dumps({"endpoints": []}))

    def fake_uw(*args, timeout=300, check=True):
        if args[0] == "status":
            return "rlesweep-x iccad13 DONE rc=0\n"
        if args[0] == "fetch":
            sub, dest = args[2], Path(args[3])
            src = tmp_path / "remote" / sub.rstrip("/")
            target = dest if sub.endswith("/") else dest / src.name
            shutil.copytree(src, target, dirs_exist_ok=True)
            return ""
        return "log tail"

    state = {"arms": {"a1": {"status": "RUNNING", "run": "rlesweep-x", "host": "iccad13"}},
             "hosts": {}}
    path = setup(tmp_path, monkeypatch, state)
    monkeypatch.setattr(runner, "uw", fake_uw)
    assert runner.cmd_poll(None, path) == 0
    arm = json.loads(path.read_text())["arms"]["a1"]
    assert arm["status"] == "DONE"
    assert (Path(arm["results_dir"]) / "summary.json").is_file()


def test_save_steps_stays_constant_through_rung2_and_rung3_promotion():
    # Rung budgets 150 -> 600 -> 1500. The trajectory's first stage saved every 150 (seed 42)
    # or every 600 (a seed that enters at rung 2); each later stage must pass the same value.
    # The gcd of a rung-3 arm's own stops and resume_stop is gcd(1500, 600) = 300: wrong.
    table = make_table([])
    r1 = dict(base_arm("u", "sft"), mode="sft")
    rung2 = {a["seed"]: a for a in runner.promoted_arms("u", [r1], "sft", 1, table)}
    assert rung2[42]["save_steps"] == 150 and "save_steps" not in rung2[314]
    rung3 = {a["seed"]: a for a in runner.promoted_arms(
        "u", list(rung2.values()), "sft", 2, table)}
    assert (rung3[42]["resume_stop"], rung3[42]["save_steps"]) == (600, 150)
    assert (rung3[314]["resume_stop"], rung3[314]["save_steps"]) == (600, 600)
    m = Path("m.jsonl")
    for seed, want in ((42, "150"), (314, "600"), (2718, "600")):
        arm = arm_exec.Arm(Path("/nonexistent"), dict(rung3[seed], response_format="x",
                           num_generations=1, min_pixels=1, max_pixels=2,
                           max_completion_length=8), Path("/nonexistent/r"))
        assert _save_steps(arm.trainer_args(m, 1500, 1500, Path("c"))) == want
    for seed in (314, 2718):  # rung-2 entrants: their own first stage
        arm = arm_exec.Arm(Path("/nonexistent"), dict(rung2[seed], response_format="x",
                           num_generations=1, min_pixels=1, max_pixels=2,
                           max_completion_length=8), Path("/nonexistent/r"))
        assert _save_steps(arm.trainer_args(m, 1500, 600, None)) == "600"


UWRUN_V5_HOSTS = """\
host         load1 nproc cores availGB ourThr localGB rootfs  shmGB  avx glibc speed freeCores
iccad13      46.59    72    36     235      0       8    yes    129  512  2.28  12.3         0
iccad4        0.02    24    24     121      0       0    yes     62    2  2.28  10.6        23
"""


def test_host_table_reads_uwrun_v5_columns_by_header():
    t = runner.host_table(UWRUN_V5_HOSTS)
    assert t["iccad13"] == {"load1": 46.59, "nproc": 72, "cores": 36, "avail_gb": 235.0,
                            "local_gb": 8.0}
    assert t["iccad4"]["avail_gb"] == 121.0 and t["iccad4"]["cores"] == 24
    old = "host load1 nproc availGB localGB\niccad4 0.0 24 121 0\n"
    with pytest.raises(SystemExit, match="header lacks"):
        runner.host_table(old)


def test_capacity_counts_physical_cores():
    t = runner.host_table(UWRUN_V5_HOSTS)
    # iccad4: 0.75 * 24 cores - 0.02 load = 17.98 -> one 16-thread arm; 0.75 * 121 GB -> 5
    assert runner.capacity({"arms": {}}, t, "iccad4", 16) == 1
    # iccad13: 0.75 * 36 = 27 cores < load 46.59 -> none (by nproc 72 it would be 0.3 -> 0
    # too, so plant a lighter load where the two rules disagree: 0.75*72-20=34 vs 27-20=7)
    t["iccad13"]["load1"] = 20.0
    assert runner.capacity({"arms": {}}, t, "iccad13", 16) == 0


def test_launch_skips_a_control_on_a_stale_host_and_launches_the_rest(tmp_path, monkeypatch):
    rl = base_arm("rl0", "rl")
    other = base_arm("rl1", "rl")
    ctl = runner.control_for(rl, {"W": 1e-5})
    state = {"arms": {}, "hosts": {
        "old": {"receipt": {"valid": True, "payload_sha256": "OLD", "root": "/r",
                            "features": ["exploratory_sweep"]}},
        "new": {"receipt": {"valid": True, "payload_sha256": "NEW", "root": "/r",
                            "features": ["exploratory_sweep"]}}},
        "payload": {"sha256": "NEW"}, "uwrun_runs": []}
    finish(tmp_path, state, "rl0", 0.5)
    state["arms"]["rl0"]["host"] = "old"
    path = setup(tmp_path, monkeypatch, state)
    table = tmp_path / "t.json"
    table.write_text(json.dumps(make_table([rl, ctl, other])))
    rows = runner.host_table("host load1 nproc cores availGB ourThr localGB\n"
                             "old 0 72 36 200 0 50\nnew 0 72 36 200 0 50\n")
    monkeypatch.setattr(runner, "host_table", lambda: rows)
    launched = []
    monkeypatch.setattr(runner, "launch_one",
                        lambda sp, spec, host, root, timeout, dry: launched.append(
                            (spec["id"], host)))

    class A:
        ids = wave = rung = hosts = None
        max_launch, timeout, dry_run = 1000, 60, False

    A.table = str(table)
    assert runner.cmd_launch(A, path) == 0
    assert launched == [("rl1", "new")]


def test_drained_stale_hosts_never_include_a_host_with_a_live_arm():
    state = {"payload": {"sha256": "NEW"}, "arms": {
        "a": {"status": "RUNNING", "host": "busy"},
        "b": {"status": "DONE", "host": "drained"}}, "hosts": {
        "busy": {"receipt": {"payload_sha256": "OLD"}},
        "drained": {"receipt": {"payload_sha256": "OLD"}},
        "booting": {"receipt": {"payload_sha256": "OLD"}, "boot_payload_sha256": "NEW"},
        "current": {"receipt": {"payload_sha256": "NEW"}},
        "never": {"receipt": None}}}
    assert runner.drained_stale_hosts(state) == ["drained"]


def test_continuation_resumes_sftall_epoch2_in_place_with_its_epoch1_endpoint():
    spec = {"id": "sftall-W-1e-5", "stops": [2329, 4658], "max_steps_total": 4658,
            "mode": "sft"}
    e1 = {"label": "step2329", "steps": 2329, "metrics": {"accuracy": 0.5}}
    cont = runner.continuation_spec(spec, {"endpoints": [e1]})
    assert (cont["id"], cont["stops"], cont["resume_stop"]) == ("sftall-W-1e-5", [4658], 2329)
    # epoch 1 ran (under the old and the new rule alike) with save_steps 2329
    assert (cont["save_steps"], cont["work_id"], cont["prior_endpoints"]) == (
        2329, "sftall-W-1e-5", [e1])
    for bad in ([], [dict(e1, steps=2000)], [e1, dict(e1, label="step4658", steps=4658)]):
        with pytest.raises(SystemExit, match="proper prefix"):
            runner.continuation_spec(spec, {"endpoints": bad})


def test_bootstrap_refuses_a_host_whose_var_tmp_is_full(tmp_path, monkeypatch):
    # iccad4 2026-10-18: / (holding /var/tmp) at 100%, 0 GB localGB; the sync would fill it
    rows = runner.host_table(UWRUN_V5_HOSTS)
    monkeypatch.setattr(runner, "host_table", lambda: rows)
    calls = []
    monkeypatch.setattr(runner, "uw", lambda *x, **k: calls.append(x))

    class A:
        payload, host, repin = "/nonexistent", ["iccad13", "iccad4"], False

    with pytest.raises(SystemExit, match=r"\['iccad4'\] have < 3.0 GB"):
        runner.cmd_bootstrap(A, setup(tmp_path, monkeypatch, {"arms": {}, "hosts": {}}))
    assert calls == []
    assert runner.low_disk_hosts(rows, ["iccad13", "iccad4", "unlisted"]) == ["iccad4", "unlisted"]


def test_bootstrap_refuses_a_host_running_our_arm(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch, {"hosts": {}, "arms": {
        "a": {"status": "RUNNING", "host": "iccad13"}, "b": {"status": "DONE", "host": "x"}}})
    calls = []
    monkeypatch.setattr(runner, "uw", lambda *x, **k: calls.append(x))

    class A:
        payload, host, repin = "/nonexistent", ["iccad13", "x"], True

    with pytest.raises(SystemExit, match=r"\['iccad13'\] run our arms"):
        runner.cmd_bootstrap(A, path)
    assert calls == []


def test_a_continuation_can_itself_be_continued():
    spec = {"id": "s", "stops": [2329, 4658, 6987], "max_steps_total": 6987, "mode": "sft"}
    e1 = {"label": "step2329", "steps": 2329}
    e2 = {"label": "step4658", "steps": 4658}
    first = runner.continuation_spec(spec, {"endpoints": [e1]})
    assert first["stops"] == [4658, 6987]
    second = runner.continuation_spec(first, {"endpoints": [e1, e2]})
    assert (second["stops"], second["resume_stop"], second["save_steps"]) == ([6987], 4658, 2329)


def test_continue_restores_the_failed_record_when_the_launch_fails(tmp_path, monkeypatch):
    spec = {"id": "s", "stops": [2329, 4658], "max_steps_total": 4658, "mode": "sft",
            "features": []}
    res = tmp_path / "res"
    res.mkdir()
    (res / "summary.json").write_text(json.dumps(
        {"endpoints": [{"label": "step2329", "steps": 2329}]}))
    (tmp_path / "arm.json").write_text(json.dumps(spec))
    state = {"payload": {"sha256": "P"}, "uwrun_runs": [], "hosts": {"h": {"receipt": {
        "valid": True, "payload_sha256": "P", "root": "/r", "features": ["exploratory_sweep"]}}},
        "arms": {"s": {"status": "FAILED", "host": "h", "run": "r1", "kind": "train",
                       "spec_path": str(tmp_path / "arm.json"), "results_dir": str(res)}}}
    path = setup(tmp_path, monkeypatch, state)
    rows = runner.host_table("host load1 nproc cores availGB ourThr localGB\nh 0 72 36 200 0 50\n")
    monkeypatch.setattr(runner, "host_table", lambda: rows)

    def boom(*a, **k):
        raise SystemExit("REFUSE: uwrun run rc=255")

    monkeypatch.setattr(runner, "launch_one", boom)

    class A:
        arm, reason, timeout, dry_run = "s", "epoch-2 refusal", 60, False

    with pytest.raises(SystemExit, match="rc=255"):
        runner.cmd_continue(A, path)
    arms = json.loads(path.read_text())["arms"]
    assert list(arms) == ["s"] and arms["s"]["status"] == "FAILED"
    assert "retired_reason" not in arms["s"]


# ---- wave 2 (plan amendment 3): concrete arms reach the trainer with their features ------
wave2 = load("wave2_arms")
HIERARCHY_ASSET = ROOT / "src/vision_rl/aircraft_assets/label-hierarchy-train-v1.json"


def _fake_root(tmp_path):
    root = tmp_path / "root"
    adapter = root / "payload" / "adapters" / "W" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter.parent / "local_training.json").write_text(
        json.dumps({"control_identity": {"profile": {"dtype": "float32"}}}))
    asset = root / "payload/code/src/vision_rl/aircraft_assets" / HIERARCHY_ASSET.name
    asset.parent.mkdir(parents=True)
    asset.write_bytes(HIERARCHY_ASSET.read_bytes())
    return root, asset


def _trainer_call(local, monkeypatch, argv):
    """Run the trainer's own CLI on arm_exec's argv; return (function, kwargs) it would call."""
    seen = []
    monkeypatch.setattr(local, "train_local", lambda **k: seen.append(("train", k)) or {})
    monkeypatch.setattr(local, "rft_local", lambda *a, **k: seen.append(("rft", k)) or {})
    assert argv[:2] == ["-m", "vision_rl.vlm_local_training"]
    assert local.main(argv[2:]) == 0
    assert len(seen) == 1
    return seen[0]


def test_wave2_table_is_reproducible_and_every_arm_declares_its_device():
    assert wave2.OUT.read_bytes() == wave2.render()
    arms = json.loads(wave2.OUT.read_text())["arms"]
    assert len(arms) == 42
    for a in arms:
        assert a["device"] in ("cpu", "cuda") and a["device_class"] in wave2.DEVICES
        assert (a["device"] == "cuda") == ("cuda" in a["features"]) == ("gpu" in a)
        assert a["wave"] == 2 and a["id"].startswith("wave2-")
    rl = [a for a in arms if a["track"] == "rl"]
    assert {a["updates"] for a in rl} == {150} and {tuple(a["stops"]) for a in rl} == {(150,)}
    by_id = {a["id"]: a for a in arms}
    for c in (a for a in arms if a["track"] == "control"):
        parent = by_id[c["control_of"]]
        assert (c["device"], c["dtype"], c.get("gpu")) == (
            parent["device"], parent["dtype"], parent.get("gpu"))
    used = {a["device_class"] for a in rl}
    evals = {(a["init"], a["device_class"]) for a in arms if a["mode"] == "eval"}
    assert evals == {(i, c) for i in ("W", "I") for c in used}  # W/I control per device


def test_wave2_arms_reach_the_trainer_with_their_features(tmp_path, monkeypatch):
    from vision_rl import vlm_local_training as local

    root, asset = _fake_root(tmp_path)
    arms = json.loads(wave2.OUT.read_text())["arms"]
    checked = 0
    for a in arms:
        if a["mode"] == "eval":
            continue
        arm = arm_exec.Arm(root, a, tmp_path / "r")
        argv = arm.trainer_args(tmp_path / "m.jsonl", int(a["max_steps_total"]), 150, None)
        kind, kw = _trainer_call(local, monkeypatch, argv)
        assert kw["device"] == a["device"] and kw["profile"].dtype == a["dtype"]
        assert kw["profile"].response_format == a["response_format"]
        if a["device"] == "cuda":
            assert kw["threads"] == 4
            assert kw["initial_adapter_precision_transfer"] == f"float32-to-{a['dtype']}"
        if a["mode"] == "rft":
            assert kind == "rft" and kw["stop_after_steps"] == 150 and kw["max_steps"] == 1500
            assert (kw["rounds"], kw["prompts_per_round"]) == (400, 32)
            assert kw["profile"].num_generations == 8 and kw["profile"].temperature == 1.0
        else:
            assert kind == "train" and kw["stop_after_steps"] == 150
        fam = a["family"]
        if fam in ("wave2-hier", "wave2-fmtpen"):
            assert kw["label_hierarchy"] == asset.resolve()
        if fam == "wave2-hier":
            assert (kw["family_credit"], kw["manufacturer_credit"]) == (0.5, 0.25)
        if fam == "wave2-fmtpen":
            assert (kw["family_credit"], kw["invalid_reward"]) == (0.0, -0.5)
        if fam == "wave2-dapo":
            assert kw["grpo_overrides"] == {"loss_type": "dapo", "epsilon_high": 0.28}
            assert kw["dynamic_sampling_max_draws"] == 4
        if fam == "wave2-drgrpo":
            assert kw["grpo_overrides"] == {"loss_type": "dr_grpo", "scale_rewards": "none"}
        if fam == "wave2-mu2":
            assert kw["grpo_overrides"] == {"num_iterations": 2}
        if a["track"] == "control" or fam in ("wave2-closedset", "wave2-cot"):
            assert "grpo_overrides" not in kw and kw.get("label_hierarchy") is None
        checked += 1
    assert checked == 36  # 18 RL arms + 18 controls


def test_payload_args_refuse_paths_outside_the_payload(tmp_path):
    root, _ = _fake_root(tmp_path)
    arm = arm_exec.Arm(root, {"id": "a"}, tmp_path / "r")
    (root / "outside.json").write_text("{}")
    for bad in ("@payload/../outside.json", "@payload/code/missing.json"):
        with pytest.raises(SystemExit, match="not a file in the payload"):
            arm.payload_arg(bad)
    assert arm.payload_arg("--x") == "--x"


def _rft_out(tmp_path, rounds, stop_reason="stop_after_steps", epochs=1):
    out = tmp_path / "root" / "arms" / "a" / "train"
    (out / "final_adapter").mkdir(parents=True)
    (out / "final_adapter" / "adapter_model.safetensors").write_bytes(b"w")
    entries = []
    for i, (kept, steps) in enumerate(rounds):
        (out / f"round-{i:02d}").mkdir()
        (out / f"round-{i:02d}" / "kept.json").write_text(json.dumps({"kept": kept}))
        entries.append({"round": i, "optimizer_steps": steps})
    total = sum(s for _, s in rounds)
    (out / "local_training.json").write_text(json.dumps({
        "status": "completed", "stop_reason": stop_reason, "optimizer_steps": total,
        "active_seconds": 12.5, "settings": {"epochs_per_round": epochs},
        "final_adapter_sha256": arm_exec.sha_file(out / "final_adapter" /
                                                  "adapter_model.safetensors"),
        "rounds": entries}))
    return arm_exec.Arm(tmp_path / "root", {"id": "a", "mode": "rft"}, tmp_path / "r")


def test_rft_receipt_lists_one_prompt_per_update_in_round_order(tmp_path):
    arm = _rft_out(tmp_path, [({"p1": ["x"], "p2": ["y", "z"]}, 3), ({"p3": ["x"]}, 1)])
    r = arm.rft_receipt()
    assert r["committed_prompt_ids"] == ["p1", "p2", "p2", "p3"]
    assert r["global_step"] == 4 and r["status"] == "boundary_reached"
    assert r["committed_active_seconds"] == 12.5
    two = _rft_out(tmp_path / "e", [({"p1": ["x"], "p2": ["y"]}, 3)], "max_steps", epochs=2)
    got = two.rft_receipt()
    assert got["committed_prompt_ids"] == ["p1", "p2", "p1"]  # 2 epochs, capped at 3 updates
    assert got["status"] == "total_budget_exhausted"


def test_rft_receipt_refuses_unbound_or_unaccounted_endpoints(tmp_path):
    arm = _rft_out(tmp_path, [({"p1": ["x"]}, 2)])  # 2 updates, 1 kept example, 1 epoch
    with pytest.raises(SystemExit, match="do not account"):
        arm.rft_receipt()
    arm = _rft_out(tmp_path / "b", [({"p1": ["x"]}, 1)])
    (arm.out / "final_adapter" / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differs from its record"):
        arm.rft_receipt()
    arm = _rft_out(tmp_path / "c", [({"p1": ["x"]}, 1)], stop_reason="stop_after_rounds")
    with pytest.raises(SystemExit, match="RFT stage ended"):
        arm.rft_receipt()


def test_launch_refuses_a_host_whose_device_differs_from_the_arm(tmp_path):
    def state(device):
        return {"payload": {"sha256": "P"}, "hosts": {"h": {"receipt": {
            "valid": True, "payload_sha256": "P", "features": ["exploratory_sweep", "cuda"],
            "device": device}}}}

    cpu = {"kind": "cpu"}
    ampere = {"kind": "cuda", "gpu": "NVIDIA GeForce RTX 3070"}
    turing = {"kind": "cuda", "gpu": "NVIDIA GeForce RTX 2080 Ti"}
    gpu_arm = {"id": "g", "device": "cuda", "gpu": "rtx3070"}
    runner.valid_receipt(state(ampere), "h", ["cuda"], gpu_arm)  # listed feature: accepted
    with pytest.raises(SystemExit, match=r"lacks features \['dapo'\]"):  # only the absent one
        runner.valid_receipt(state(ampere), "h", ["cuda", "dapo"], gpu_arm)
    for device, match in ((cpu, "declares cuda"), (turing, "declares GPU rtx3070")):
        with pytest.raises(SystemExit, match=match):
            runner.valid_receipt(state(device), "h", ["cuda"], gpu_arm)
    with pytest.raises(SystemExit, match="declares cpu"):
        runner.valid_receipt(state(ampere), "h", [], {"id": "c", "device": "cpu"})
    runner.valid_receipt(state(cpu), "h", [], {"id": "c"})  # a wave-1 arm: cpu by default


def test_rft_continuation_accepts_a_round_boundary_past_its_stop_never_short():
    spec = {"id": "r", "mode": "rft", "stops": [150, 600], "max_steps_total": 1500}
    past = {"label": "step150", "steps": 163}
    cont = runner.continuation_spec(spec, {"endpoints": [past]})
    assert (cont["stops"], cont["resume_stop"]) == ([600], 150)
    with pytest.raises(SystemExit, match="proper prefix"):
        runner.continuation_spec(spec, {"endpoints": [dict(past, steps=140)]})
    sft = dict(spec, mode="sft")  # every other mode: steps must equal the stop
    with pytest.raises(SystemExit, match="proper prefix"):
        runner.continuation_spec(sft, {"endpoints": [past]})


def test_one_gpu_arm_per_host():
    hosts = {"h": {"load1": 0.0, "nproc": 72, "cores": 36, "avail_gb": 200.0}}
    busy = {"arms": {"g": {"host": "h", "status": "RUNNING", "device": "cuda", "threads": 4,
                           "launched_ts": 0}}}
    assert runner.capacity(busy, hosts, "h", 4, "cuda") == 0
    assert runner.capacity(busy, hosts, "h", 4, "cpu") > 0
    assert runner.capacity({"arms": {}}, hosts, "h", 4, "cuda") > 0
