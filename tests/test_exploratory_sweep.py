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
        "id": arm_id, "track": track, "rung": 1, "seed": 42, "init": "I", "mode": "grpo",
        "family": "d37-grpo-I",
        "lr": 1e-5, "stops": [150], "max_steps_total": 1500, "wave": 1, "num_generations": 8,
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
                            "local_gb": 8.0, "avx": "512"}
    assert t["iccad4"]["avx"] == "2"
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
        "payload": {"sha256": "NEW"}, "uwrun_runs": [], "gate0": {"passed": True},
        "d37_release": {"verified": True}}
    finish(tmp_path, state, "rl0", 0.5)
    state["arms"]["rl0"]["host"] = "old"
    path = setup(tmp_path, monkeypatch, state)
    table = tmp_path / "t.json"
    table.write_text(json.dumps(make_table([rl, ctl, other])))
    rows = runner.host_table("host load1 nproc cores availGB ourThr localGB\n"
                             "old 0 72 36 200 0 50\nnew 0 72 36 200 0 50\n")
    monkeypatch.setattr(runner, "host_table", lambda: rows)
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)
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
            "features": [], "family": "sftall-W"}
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
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)

    class A:
        arm, reason, timeout, dry_run = "s", "epoch-2 refusal", 60, False

    with pytest.raises(SystemExit, match="rc=255"):
        runner.cmd_continue(A, path)
    arms = json.loads(path.read_text())["arms"]
    assert list(arms) == ["s"] and arms["s"]["status"] == "FAILED"
    assert "retired_reason" not in arms["s"]


# ---- wave 2 (plan amendment 4): concrete arms reach the trainer with their features ------
wave2 = load("wave2_arms")
HIERARCHY_ASSET = ROOT / "scripts/exploratory_sweep/label-hierarchy-train-v1.json"


def _fake_root(tmp_path):
    root = tmp_path / "root"
    for init in ("W", "I"):
        adapter = root / "payload" / "adapters" / init / "final_adapter"
        adapter.mkdir(parents=True)
        (adapter.parent / "local_training.json").write_text(
            json.dumps({"control_identity": {"profile": {"dtype": "float32"}}}))
    asset = root / "payload/code/scripts/exploratory_sweep" / HIERARCHY_ASSET.name
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
    # 5 GRPO families x 3 LRs + Dr-GRPO's 3e-5 + the 3e-6 guard + RFT x 3 = 20 RL arms,
    # 20 matched controls, W and I evaluated on each of the 3 device classes = 6
    assert len(arms) == 46
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


def test_wave2_table_follows_the_root_rl_corrections():
    """runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md (root checkout), binding 2026-10-18."""
    arms = json.loads(wave2.OUT.read_text())["arms"]
    trained = [a for a in arms if a["mode"] != "eval"]
    assert {a["init"] for a in trained} == {"I"}  # GRPO-W loses to I by 5-9 pp
    fams = {a["family"] for a in arms if a["track"] == "rl"}
    assert fams == {"wave2-fmtpen", "wave2-closedset", "wave2-drgrpo", "wave2-mu2",
                    "wave2-dapo", "wave2-guard", "wave2-rft"}  # hier and both CoT dropped
    assert not any("think" in a["response_format"] for a in arms)
    grpo = [a for a in arms if a["mode"] == "grpo"]
    assert {a["beta"] for a in grpo} == {0.0}
    assert {a["dtype"] for a in grpo} == {"float32"}  # GRPO in fp32 only
    lrs = {}
    for a in grpo + [a for a in arms if a["mode"] == "rft"]:
        lrs.setdefault(a["family"], set()).add(a["lr"])
    main = {1e-5, 2e-5, 5e-5}
    assert lrs == {"wave2-fmtpen": main, "wave2-closedset": main, "wave2-mu2": main,
                   "wave2-dapo": main, "wave2-rft": main, "wave2-drgrpo": main | {3e-5},
                   "wave2-guard": {3e-6}}
    dapo = [a for a in grpo if a["family"] == "wave2-dapo"]
    assert all(a["extra_trainer_args"][-2:] == ["--dynamic-sampling-max-draws", "8"]
               for a in dapo)
    by_id = {a["id"]: a for a in arms}
    # per family, rank for rank and all DISTINCT: same LR count as the RL arms (C4)
    three = {1e-5: 5e-6, 2e-5: 1e-5, 5e-5: 2e-5}
    want = {f: three for f in ("wave2-fmtpen", "wave2-closedset", "wave2-mu2", "wave2-dapo",
                               "wave2-rft")}
    want["wave2-drgrpo"] = {1e-5: 5e-6, 2e-5: 1e-5, 3e-5: 2e-5, 5e-5: 3e-5}
    want["wave2-guard"] = {3e-6: 5e-6}
    per_family = {}
    for c in (a for a in arms if a["track"] == "control"):
        rl = by_id[c["control_of"]]
        expected = want[rl["family"]][rl["lr"]]
        assert c["init"] == "I" and c["mode"] == "sft" and c["lr"] == expected
        per_family.setdefault(rl["family"], []).append(c["lr"])
        promoted = runner.control_for(rl, {"W": 1e-5, "I": 5e-6})  # rung 2/3 control
        assert promoted["lr"] == expected
        for key in ("response_format", "max_steps_total"):
            assert promoted[key] == c[key], key
        assert promoted.get("control_replays") == c.get("control_replays")
    assert all(len(set(v)) == len(v) for v in per_family.values())  # distinct per family
    assert len(per_family["wave2-drgrpo"]) == 4
    # the closed-set pair isolates RL: its control trains and assays on the closed-set prompt
    for c in (a for a in arms if a["track"] == "control"):
        rl = by_id[c["control_of"]]
        assert c["response_format"] == rl["response_format"]
    # every scored row is replayed (C4) where RL scores more rows than it commits
    replay = {by_id[c["control_of"]]["family"] for c in arms
              if c["track"] == "control" and c.get("control_replays") == "scanned"}
    assert replay == {"wave2-dapo", "wave2-rft"}


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
            assert kw["initial_adapter_precision_transfer"] == (
                None if a["dtype"] == "float32" else f"float32-to-{a['dtype']}")
        if a["mode"] == "rft":
            assert kind == "rft" and kw["stop_after_steps"] == 150 and kw["max_steps"] == 1500
            assert (kw["rounds"], kw["prompts_per_round"]) == (400, 32)
            assert kw["profile"].num_generations == 8 and kw["profile"].temperature == 1.0
        else:
            assert kind == "train" and kw["stop_after_steps"] == 150
        fam = a["family"]
        if fam == "wave2-fmtpen":
            assert kw["label_hierarchy"] == asset.resolve()
            assert (kw["family_credit"], kw["invalid_reward"]) == (0.0, -0.5)
        if fam == "wave2-dapo":
            assert kw["grpo_overrides"] == {"loss_type": "dapo", "epsilon_high": 0.28}
            assert kw["dynamic_sampling_max_draws"] == 8
        if fam == "wave2-drgrpo":  # main's explicit Dr-GRPO parameters
            assert (kw["loss_type"], kw["scale_rewards"]) == ("dr_grpo", "none")
            assert "grpo_overrides" not in kw
        if fam == "wave2-mu2":
            assert kw["grpo_overrides"] == {"num_iterations": 2}
        if fam in ("wave2-fmtpen", "wave2-closedset", "wave2-mu2", "wave2-guard"):
            # beta 0 names its aggregation: the legacy "grpo" (settings unchanged)
            assert kw["loss_type"] == "grpo" and kw["scale_rewards"] == "group"
        if fam == "wave2-dapo":
            assert kw["loss_type"] is None  # DAPO's loss_type is the knob above
        if a["mode"] == "grpo":
            assert kw["beta"] == 0.0 and kw["profile"].dtype == "float32"
        if a["track"] == "control" or fam in ("wave2-closedset", "wave2-guard"):
            assert "grpo_overrides" not in kw and kw.get("label_hierarchy") is None
        checked += 1
    assert checked == 40  # 20 RL arms + 20 controls


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
        # every real round record lists all the prompts it sampled (kept or not)
        entries.append({"round": i, "optimizer_steps": steps, "prompt_ids": list(kept)})
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


def _assay_specs():
    """Every arm the runner can hand to arm_exec: both tables, plus every rung-2 arm and
    promoted control (runner.promoted_arms -> control_for) built from their RL units."""
    tables = [json.loads((ROOT / "docs/aircraft-exploratory-sweep-v2/arms-rung1.json")
                         .read_text()), json.loads(wave2.OUT.read_text())]
    specs = []
    for table in tables:
        specs += table["arms"]
        for a in table["arms"]:
            if a["track"] in ("rl", "sft") and a["mode"] != "eval":
                specs += runner.promoted_arms(a["id"], [a], a["track"], 1, table)
    return specs


def test_every_arm_and_promoted_control_builds_a_valid_assay_profile(tmp_path):
    from vision_rl import vlm_local_training as local

    specs = _assay_specs()
    promoted_controls = [a for a in specs if "@r2" in a["id"] and a["track"] == "control"]
    assert promoted_controls and all(a["temperature"] is None for a in promoted_controls)
    for a in specs:
        profile = arm_exec.Arm(tmp_path, a, tmp_path / "r").assay_profile(local)
        profile.validate()  # predict_train_subset runs this first; None raised TypeError
        # literal Profile defaults (never read from the subject): temperature 1.0, G 4
        defaults = {"temperature": 1.0, "num_generations": 4}
        for k in arm_exec.PROFILE_KEYS:  # the spec's value, or the default if absent/None
            want = a.get(k)
            assert getattr(profile, k) == (defaults[k] if want is None else want), (a["id"], k)
    controls = sum(a["track"] == "control" and "@r" not in a["id"] for a in specs)
    assert controls == 40 + 29 + 20  # wave-1 and pre-registered wave-2 controls, and ours


def test_receipt_gpu_matches_whole_model_numbers():
    names = {"NVIDIA GeForce RTX 3070": "rtx3070", "NVIDIA GeForce RTX 2080 Ti": "rtx2080ti",
             "NVIDIA A40": "a40", "NVIDIA RTX A4000": "a4000", "NVIDIA A100-SXM4-40GB": "a100"}
    for name, want in names.items():
        assert runner.receipt_gpu({"kind": "cuda", "gpu": name}) == want
    assert runner.receipt_gpu({"kind": "cuda", "gpu": "RTX 3070 Ti"}) != "rtx3070"
    assert runner.receipt_gpu({"kind": "cpu"}) is None


def test_rft_stage_reuses_a_completed_endpoint_only_for_its_own_stop(tmp_path):
    arm = _rft_out(tmp_path, [({"p1": ["x"]}, 1)])  # boundary endpoint, boundary unset
    rec = json.loads((arm.out / "local_training.json").read_text())
    (arm.out / "local_training.json").write_text(json.dumps(
        dict(rec, boundary={"stop_after_steps": 150})))
    assert arm.rft_endpoint_done(150) and not arm.rft_endpoint_done(600)
    got = arm.run_stage("train-150", ["-m", "x"], 150)  # no trainer subprocess is started
    assert got["global_step"] == 1 and arm.log[-1]["reused_completed_endpoint"] is True
    done = _rft_out(tmp_path / "t", [({"p1": ["x"]}, 1)], stop_reason="max_steps")
    assert done.rft_endpoint_done(600)  # a terminal endpoint ends the run at any stop
    (done.out / "local_training.json").write_text(json.dumps(dict(rec, status="failed")))
    assert not done.rft_endpoint_done(600)


def test_one_gpu_arm_per_host():
    hosts = {"h": {"load1": 0.0, "nproc": 72, "cores": 36, "avail_gb": 200.0}}
    busy = {"arms": {"g": {"host": "h", "status": "RUNNING", "device": "cuda", "threads": 4,
                           "launched_ts": 0}}}
    assert runner.capacity(busy, hosts, "h", 4, device="cuda") == 0
    assert runner.capacity(busy, hosts, "h", 4, device="cpu") > 0
    assert runner.capacity({"arms": {}}, hosts, "h", 4, device="cuda") > 0


# uwrun v5.4's own marker line (~/.claude/bin/uwrun memory guard), values planted
HOST_OOM = ("uwrun: memory guard: the run holds 14000 MB (--mem 19 GB), host RAM free 3000 MB "
            "(floor 4096 MB): stopped")
CAP_OOM = ("uwrun: memory guard: the run holds 20000 MB (--mem 19 GB), host RAM free 90000 MB "
           "(floor 4096 MB): stopped")


def test_failure_reason_separates_host_oom_from_our_cap_and_from_code():
    s = "rlesweep-x iccad8 DONE rc=137 oom-guard"
    assert runner.failure_reason(s, "...\n" + HOST_OOM) == "oom-guard:host-ram"
    assert runner.failure_reason(s, CAP_OOM) == "oom-guard:mem-cap"
    assert runner.failure_reason(s, "no guard line") == "oom-guard:unparsed"
    assert runner.failure_reason("x h DONE rc=143 timedout", "") == "timedout"
    assert runner.failure_reason("x h DONE rc=2", "resume must preserve") == "code"


def test_poll_marks_oom_guard_failed_and_retires_a_host_kill_for_relaunch(tmp_path, monkeypatch):
    state = {"hosts": {}, "arms": {
        "a": {"status": "RUNNING", "run": "ra", "host": "h"},
        "b": {"status": "RUNNING", "run": "rb", "host": "h"}}}
    path = setup(tmp_path, monkeypatch, state)

    def fake_uw(*args, **k):
        if args[0] == "status":
            return f"{args[1]} h DONE rc=137 oom-guard\n"
        if args[0] == "logs":
            return HOST_OOM if args[1] == "ra" else CAP_OOM
        if args[0] == "fetch":  # a stage-less summary exists (rc=0 would not make it DONE)
            Path(args[3]).mkdir(parents=True)
            (Path(args[3]) / "summary.json").write_text(json.dumps({"endpoints": []}))
        return ""

    monkeypatch.setattr(runner, "uw", fake_uw)
    runner.cmd_poll(None, path)
    arms = json.loads(path.read_text())["arms"]
    assert "a" not in arms and arms["a#fail1"]["failure_reason"] == "oom-guard:host-ram"
    assert arms["a#fail1"]["retired_from"] == "a"
    assert arms["b"]["status"] == "FAILED" and arms["b"]["failure_reason"] == "oom-guard:mem-cap"


def test_host_oom_retire_keeps_a_resumable_arm_and_stops_after_two():
    def st(endpoints, prior):
        arms = {"s": {"status": "FAILED", "failure_reason": "oom-guard:host-ram",
                      "results_dir": "/nonexistent"}}
        for k in range(prior):
            arms[f"s#fail{k + 1}"] = {"retired_from": "s", "failure_reason": "oom-guard:host-ram"}
        return {"arms": arms}
    assert runner.retire_if_host_oom(st([], 0), "s") == "s#fail1"
    assert runner.retire_if_host_oom(st([], 1), "s") == "s#fail2"
    assert runner.retire_if_host_oom(st([], 2), "s") is None


def test_host_oom_with_a_completed_stop_stays_failed_for_continue(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(
        {"endpoints": [{"label": "step2329", "steps": 2329}]}))
    state = {"arms": {"s": {"status": "FAILED", "failure_reason": "oom-guard:host-ram",
                            "results_dir": str(tmp_path)}}}
    assert runner.retire_if_host_oom(state, "s") is None
    assert state["arms"]["s"]["status"] == "FAILED"


def test_fresh_work_id_only_for_from_scratch_relaunches():
    state = {"arms": {"a#fail1": {"retired_from": "a"}, "a#fail2": {"retired_from": "a"}}}
    assert runner.fresh_work_id(state, {"id": "a"})["work_id"] == "a~a3"
    assert "work_id" not in runner.fresh_work_id(state, {"id": "new"})
    assert runner.fresh_work_id(state, {"id": "a", "resume_stop": 150}) == {
        "id": "a", "resume_stop": 150}
    assert runner.fresh_work_id(state, {"id": "a", "work_id": "w"})["work_id"] == "w"


def test_mem_gb_is_measured_peak_plus_20_percent_and_refuses_unmeasured(monkeypatch):
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", None)
    with pytest.raises(SystemExit, match="assay peak RSS is unmeasured"):
        runner.mem_gb({"mode": "sft"})
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)
    assert runner.mem_gb({"mode": "sft"}) == 16          # ceil(1.2 * 12.97 = 15.564)
    assert runner.mem_gb({"mode": "grpo", "num_generations": 16}) == 19  # 18.264
    assert runner.mem_gb({"mode": "grpo", "num_generations": 8}) == 17   # 16.272
    assert runner.mem_gb({"mode": "eval"}) == 12                         # 12.0
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 16.0)
    assert runner.mem_gb({"mode": "sft"}) == 20          # assay dominates: ceil(19.2)
    assert runner.mem_gb({"mode": "grpo", "num_generations": 32}) is None  # unmeasured: no cap
    assert runner.mem_gb({"mode": "egsft", "num_generations": 4}) is None


def test_launch_one_passes_mem_and_a_fresh_work_dir(tmp_path, monkeypatch):
    state = {"hosts": {"h": {"receipt": {"payload_sha256": "P"}}}, "uwrun_runs": [],
             "arms": {"a#fail1": {"retired_from": "a"}}}
    path = setup(tmp_path, monkeypatch, state)
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)
    calls = []
    monkeypatch.setattr(runner, "uw", lambda *x, **k: calls.append(x))
    name = runner.launch_one(path, {"id": "a", "mode": "sft"}, "h", "/r", 60, False)
    argv = calls[0]
    assert argv[argv.index("--mem") + 1] == "16"
    spec = json.loads((tmp_path / "sweep" / "launch" / name / "arm.json").read_text())
    assert spec["work_id"] == "a~a2"
    assert json.loads(path.read_text())["arms"]["a"]["mem_gb"] == 16


def test_launch_one_omits_mem_for_an_unmeasured_kind(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch, {"hosts": {"h": {"receipt": {"payload_sha256": "P"}}},
                                         "uwrun_runs": [], "arms": {}})
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)
    calls = []
    monkeypatch.setattr(runner, "uw", lambda *x, **k: calls.append(x))
    runner.launch_one(path, {"id": "e", "mode": "egsft", "num_generations": 4}, "h", "/r", 60,
                      False)
    assert "--mem" not in calls[0]
    assert json.loads(path.read_text())["arms"]["e"]["mem_gb"] is None


def test_amendment3_launch_allowlist_holds_w_init_and_gates_grpo_i():
    # amendment 5: the allowlisted family is the D37 table's; the old grpo-I is retired
    table = {"g": {"id": "g", "family": "d37-grpo-I"}, "w": {"id": "w", "family": "grpo-W"},
             "old": {"id": "old", "family": "grpo-I"}}
    closed, opened = {}, {"gate0": {"passed": True}, "d37_release": {"verified": True}}
    unverified = {"gate0": {"passed": True}}
    for state in (closed, opened):
        for spec in ({"id": "w", "family": "grpo-W", "track": "rl"},
                     {"id": "w~ctl", "family": "control", "track": "control", "control_of": "w"},
                     {"id": "s", "family": "sft-I", "track": "sft"},
                     {"id": "x", "family": "sftall-rl", "track": "rl"},
                     {"id": "old", "family": "grpo-I", "track": "rl"},
                     {"id": "old~ctl", "family": "control", "track": "control",
                      "control_of": "old"}):
            assert "allowlist" in runner.launch_hold(state, spec, table)
    g = {"id": "g", "family": "d37-grpo-I", "track": "rl"}
    gc = {"id": "g~ctl", "family": "control", "track": "control", "control_of": "g"}
    assert "GATE 0" in runner.launch_hold(closed, g, table)
    assert "GATE 0" in runner.launch_hold({"gate0": {"passed": "yes"}}, gc, table)
    assert runner.launch_hold(opened, g, table) is None
    assert runner.launch_hold(opened, gc, table) is None
    assert "d37_release" in runner.launch_hold(unverified, g, table)
    assert "d37_release" in runner.launch_hold(unverified, gc, table)


def test_amendment3_promotion_never_ranks_w_init_units(tmp_path, monkeypatch):
    w = [dict(base_arm(f"w{k}", "rl"), init="W", family="grpo-W") for k in range(3)]
    i = [base_arm(f"i{k}", "rl") for k in range(3)]
    state = {"arms": {}, "hosts": {}}
    for k, arm in enumerate(w):
        finish(tmp_path, state, arm["id"], 0.90 + k / 100)  # W would win if ranked
    for k, arm in enumerate(i):
        finish(tmp_path, state, arm["id"], 0.60 + k / 100)
    path = setup(tmp_path, monkeypatch, state)
    (tmp_path / "design" / "arms-rung1.json").write_text(json.dumps(make_table(w + i)))

    class A:
        from_rung, late, overwrite = 1, False, False

    assert runner.cmd_promote(A, path) == 0
    out = json.loads((tmp_path / "sweep" / "arms-rung2.json").read_text())
    assert out["promotion"]["rl"] == ["i2"] and out["held_w_init"] == ["w0", "w1", "w2"]
    assert not [a for a in out["arms"] if a["id"].startswith("w")]


def test_amendment3_continue_only_for_sftall(tmp_path, monkeypatch):
    (tmp_path / "arm.json").write_text(json.dumps({"id": "g", "family": "grpo-W"}))
    path = setup(tmp_path, monkeypatch, {"hosts": {}, "arms": {"g": {
        "status": "FAILED", "spec_path": str(tmp_path / "arm.json"), "results_dir": "/x"}}})

    class A:
        arm, reason, timeout, dry_run = "g", "r", 60, True

    with pytest.raises(SystemExit, match="amendment 3"):
        runner.cmd_continue(A, path)


def test_amendment3_refresh_targets_only_continuable_hosts_while_gate0_open(tmp_path):
    res = tmp_path / "r"
    res.mkdir()
    (res / "summary.json").write_text(json.dumps({"endpoints": [{"label": "step2329"}]}))
    (tmp_path / "a.json").write_text(json.dumps({"family": "sftall-W"}))
    (tmp_path / "b.json").write_text(json.dumps({"family": "grpo-W"}))
    state = {"arms": {
        "s": {"status": "FAILED", "host": "iccad13", "spec_path": str(tmp_path / "a.json"),
              "results_dir": str(res)},
        "g": {"status": "FAILED", "host": "iccad8", "spec_path": str(tmp_path / "b.json"),
              "results_dir": str(res)}}}
    assert runner.continuable_hosts(state) == {"iccad13"}


# ---- review fixes (PR #36 at f6b95d6): each test fails on that code -------------------
def _dapo_out(tmp_path, lines, committed):
    arm = arm_exec.Arm(tmp_path / "root", {"id": "d", "mode": "grpo"}, tmp_path / "r")
    arm.out.mkdir(parents=True)
    (arm.out / "dynamic_sampling.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines))
    return arm, {"global_step": len(committed), "committed_prompt_ids": committed}


def _draws(step, ids):
    return [{"global_step": step, "draw": k, "informative": k == len(ids) - 1, "id": i,
             "kept": k == len(ids) - 1} for k, i in enumerate(ids)]


def test_dynamic_sampling_control_replays_every_scored_group_in_scan_order(tmp_path):
    # update 2 ran twice (a resume): its LAST block counts; update 3 was never committed
    lines = (_draws(1, ["a", "b", "c"]) + _draws(2, ["d", "e"]) + _draws(2, ["d", "f", "g"])
             + _draws(3, ["h"]))
    arm, receipt = _dapo_out(tmp_path, lines, ["c", "g"])
    doc = arm.scanned_ids(receipt)
    assert doc["scanned_prompt_ids"] == ["a", "b", "c", "d", "f", "g"]
    assert json.loads((arm.out / "scanned_ids.json").read_text()) == doc
    for committed in (["c", "e"], ["c", "g", "h", "i"]):  # kept id differs / update 4 missing
        bad, receipt = _dapo_out(tmp_path / str(len(committed)) / committed[-1], lines,
                                 committed)
        with pytest.raises(SystemExit, match="does not account"):
            bad.scanned_ids(receipt)


def test_rft_control_replays_every_sampled_prompt_not_only_kept_completions(tmp_path):
    arm = _rft_out(tmp_path, [({"p1": ["x"]}, 1), ({"p3": ["y"]}, 1)])
    rec = json.loads((arm.out / "local_training.json").read_text())
    rec["rounds"][0]["prompt_ids"] = ["p1", "p2"]
    rec["rounds"][1]["prompt_ids"] = ["p3", "p4"]
    (arm.out / "local_training.json").write_text(json.dumps(rec))
    receipt = arm.rft_receipt()
    assert receipt["committed_prompt_ids"] == ["p1", "p3"]  # kept examples only
    assert arm.scanned_ids(receipt)["scanned_prompt_ids"] == ["p1", "p2", "p3", "p4"]
    plain = arm_exec.Arm(tmp_path / "x", {"id": "g", "mode": "grpo"}, tmp_path / "r")
    assert plain.scanned_ids({"global_step": 1, "committed_prompt_ids": ["a"]}) is None


def test_scanned_replay_control_uses_scanned_ids_and_refuses_without_them(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "summary.json").write_text(json.dumps({"endpoints": [
        {"label": "step2", "steps": 2, "active_seconds": 9.0}]}))
    (res / "endpoint_receipt.json").write_text(json.dumps(
        {"global_step": 2, "committed_prompt_ids": ["c", "g"]}))
    state = {"arms": {"d": {"status": "DONE", "results_dir": str(res)}}}
    ctl = {"id": "d~ctl", "control_of": "d", "max_steps_total": 100,
           "control_replays": "scanned"}
    with pytest.raises(SystemExit, match="no scanned_ids.json"):
        runner.control_spec(state, ctl)
    (res / "scanned_ids.json").write_text(json.dumps(
        {"global_step": 1, "scanned_prompt_ids": ["a", "c"]}))
    with pytest.raises(SystemExit, match="another step"):
        runner.control_spec(state, ctl)
    (res / "scanned_ids.json").write_text(json.dumps(
        {"global_step": 2, "scanned_prompt_ids": ["a", "b", "c", "d", "f", "g"]}))
    spec = runner.control_spec(state, ctl)
    assert spec["train_ids"] == ["a", "b", "c", "d", "f", "g"] and spec["stops"] == [6]
    plain = runner.control_spec(state, {"id": "d~ctl", "control_of": "d", "max_steps_total": 100})
    assert plain["train_ids"] == ["c", "g"]  # every other control: unchanged


def test_wave2_launch_refuses_until_gate0_passed(tmp_path, monkeypatch):
    receipt = {"valid": True, "payload_sha256": "P", "root": "/r", "device": {"kind": "cpu"},
               "features": sorted(set(f for a in json.loads(wave2.OUT.read_text())["arms"]
                                      for f in a["features"]) | {"exploratory_sweep"})}
    rows = runner.host_table(
        "host load1 nproc cores availGB ourThr localGB\nh 0 72 36 500 0 50\n")
    monkeypatch.setattr(runner, "host_table", lambda: rows)
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)

    class A:
        table, ids, wave, rung, hosts = str(wave2.OUT), None, None, None, None
        max_launch, timeout, dry_run = 100, 60, False

    # n 0-5 vary GATE 0 with d37_release verified (the gate is the only variable); n 6 has
    # GATE 0 passed but d37_release unverified: wave-2 controls are PR #36's machinery too
    for n, gate in enumerate((None, {}, {"passed": False}, {"passed": "true"}, {"passed": 1},
                              {"passed": True}, {"passed": True})):
        calls = []
        monkeypatch.setattr(runner, "launch_one",
                            lambda path, spec, *a, **k: calls.append(spec["id"]) or "run")
        state = {"payload": {"sha256": "P"}, "hosts": {"h": {"receipt": receipt}},
                 "arms": {}, "uwrun_runs": []}
        if n < 6:
            state["d37_release"] = {"verified": True}
        if gate is not None:
            state["gate0"] = gate
        (tmp_path / f"g{n}").mkdir()
        path = setup(tmp_path / f"g{n}", monkeypatch, state)
        runner.cmd_launch(A, path)
        if n == 5:  # {"passed": True}; not `gate == ...`: {"passed": 1} == {"passed": True}
            # the gate, not the allowlist, was the only thing holding them: CPU arms launch
            assert any(i.startswith("wave2-") and "cpu-float32" in i and "~ctl" not in i
                       for i in calls), calls
        else:
            assert calls == [], (gate, calls)
    held = runner.launch_hold({}, {"id": "x", "family": "wave2-rft", "track": "rl"}, {})
    assert "GATE 0" in held


# ---- delta review of 0f0f430: replay manifests with repeated ids, stricter checks ------
def _replay_payload(tmp_path):
    """A probe TRAIN pool of 8 real images, placed as a payload's training manifest."""
    from vision_rl import vlm_local_training as local

    data = tmp_path / "probe"
    local.make_probe_data(data)
    manifests = tmp_path / "root" / "payload" / "data" / "manifests"
    manifests.mkdir(parents=True)
    rows = [json.loads(x) for x in (data / "train.jsonl").read_text().splitlines()]
    for r in rows:
        r["image"] = str((data / r["image"]).resolve())
    (manifests / "s_train_probe.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return local, data, [r["id"] for r in rows]


@pytest.mark.parametrize("kind", ["dapo-repeats", "rung3-1500", "dapo-rung3-4000"])
def test_replay_manifest_with_repeats_loads_and_trains_in_order(tmp_path, monkeypatch, kind):
    local, data, pool = _replay_payload(tmp_path)
    if kind == "dapo-repeats":  # replacement draws pick the same row again
        ids = [pool[0], pool[3], pool[3], pool[1], pool[0], pool[5], pool[3]]
    else:  # a rung-3 control (1,500 committed ids) or a DAPO one past the 3,334-row cap
        n = 1500 if kind == "rung3-1500" else 4000
        ids = [pool[(k * 3) % len(pool)] for k in range(n)]
    spec = {"id": f"x~ctl-{kind}", "mode": "sft", "train_ids": ids, "lr": 1e-5, "seed": 42,
            "init": "I", "order": "sequential", "cycle": True, "response_format": "answer_only",
            "num_generations": 4, "min_pixels": 1, "max_pixels": 2, "max_completion_length": 8,
            "max_active_seconds": 100.0}
    arm = arm_exec.Arm(tmp_path / "root", spec, tmp_path / "r")
    path = arm.manifest()
    argv = arm.trainer_args(path, 12000, len(ids), None)
    assert "--replay-sequence" in argv
    with pytest.raises(ValueError, match="must be unique"):  # the crash this fixes
        local.load_classification(path, "train", ["blue", "red"], require_all_classes=False)
    rows = local.load_classification(path, "train", ["blue", "red"], require_all_classes=False,
                                     allow_repeated_ids=True)
    assert [r["id"] for r in rows] == ids  # every presentation, in replay order
    # train_local accepts it (past loader, row cap and validation) and reaches preflight
    monkeypatch.setattr(local, "preflight", lambda *_: (_ for _ in ()).throw(
        RuntimeError("reached preflight")))
    common = dict(mode="sft", train_only=True, exploratory_sweep=local.EXPLORATORY_SWEEP_ID,
                  order="sequential", cycle=True, max_steps=12000, max_active_seconds=100.0,
                  learning_rate=1e-5, model_directory=tmp_path / "model")
    with pytest.raises(RuntimeError, match="reached preflight"):
        local.train_local(path, None, data / "data_spec.json", tmp_path / "out",
                          replay_sequence=True, **common)
    with pytest.raises(ValueError, match="must be unique"):
        local.train_local(path, None, data / "data_spec.json", tmp_path / "out2", **common)


def test_replay_sequence_is_control_only_and_repeats_must_be_exact(tmp_path):
    local, data, pool = _replay_payload(tmp_path)
    with pytest.raises(ValueError, match="sequential SFT control"):
        local.train_local(data / "train.jsonl", None, data / "data_spec.json", tmp_path / "o",
                          mode="grpo", train_only=True, replay_sequence=True)
    rows = [json.loads(x) for x in (data / "train.jsonl").read_text().splitlines()]
    bad = dict(rows[0], label="red" if rows[0]["label"] == "blue" else "blue")
    (data / "bad.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [rows[0], bad]))
    with pytest.raises(ValueError, match="must repeat its row"):
        local.load_classification(data / "bad.jsonl", "train", ["blue", "red"],
                                  require_all_classes=False, allow_repeated_ids=True)
    distinct = arm_exec.Arm(tmp_path / "root", {"id": "c", "mode": "sft", "train_ids": pool,
                                                "lr": 1e-5, "seed": 1, "init": "I",
                                                "response_format": "answer_only",
                                                "num_generations": 4, "min_pixels": 1,
                                                "max_pixels": 2, "max_completion_length": 8},
                            tmp_path / "r")
    assert "--replay-sequence" not in distinct.trainer_args(tmp_path / "m", 8, 8, None)


def test_assay_profile_refuses_a_missing_key_other_than_temperature(tmp_path):
    from vision_rl import vlm_local_training as local

    # the sweep's real pixel bounds: assay_profile now also runs Profile.validate()
    base = {"id": "a", "mode": "grpo", "response_format": "answer_only", "min_pixels": 50176,
            "max_pixels": 200704, "max_completion_length": 8, "num_generations": 8,
            "temperature": None}
    assert arm_exec.Arm(tmp_path, base, tmp_path / "r").assay_profile(local).temperature == 1.0
    for key in ("response_format", "max_pixels", "num_generations"):
        with pytest.raises(SystemExit, match=key):
            arm_exec.Arm(tmp_path, dict(base, **{key: None}), tmp_path / "r").assay_profile(local)
    sft = dict(base, mode="sft", num_generations=None)  # SFT never samples: default 4
    assert arm_exec.Arm(tmp_path, sft, tmp_path / "r").assay_profile(local).num_generations == 4


def test_scanned_replay_refuses_committed_ids_outside_the_scan_order(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "summary.json").write_text(json.dumps({"endpoints": [
        {"label": "step2", "steps": 2, "active_seconds": 9.0}]}))
    state = {"arms": {"d": {"status": "DONE", "results_dir": str(res)}}}
    ctl = {"id": "d~ctl", "control_of": "d", "max_steps_total": 100,
           "control_replays": "scanned"}

    def write(committed, scanned, schema=None):
        (res / "endpoint_receipt.json").write_text(json.dumps(
            {"global_step": 2, "committed_prompt_ids": committed,
             **({"schema": schema} if schema else {})}))
        (res / "scanned_ids.json").write_text(json.dumps(
            {"global_step": 2, "scanned_prompt_ids": scanned}))

    write(["c", "g"], ["a", "c", "d", "g"])
    assert runner.control_spec(state, ctl)["train_ids"] == ["a", "c", "d", "g"]
    for committed in (["g", "c"], ["c", "z"]):  # out of order / never scanned
        write(committed, ["a", "c", "d", "g"])
        with pytest.raises(SystemExit, match="not within its scanned ids"):
            runner.control_spec(state, ctl)
    rft = "arm_exec-rft-endpoint-v1"
    write(["g", "c"], ["a", "c", "d", "g"], rft)  # RFT: kept order is canonical, subset
    assert runner.control_spec(state, ctl)["train_ids"] == ["a", "c", "d", "g"]
    write(["c", "z"], ["a", "c", "d", "g"], rft)
    with pytest.raises(SystemExit, match="not within its scanned ids"):
        runner.control_spec(state, ctl)


def test_rft_scan_refuses_a_kept_prompt_the_round_never_sampled(tmp_path):
    arm = _rft_out(tmp_path, [({"p1": ["x"]}, 1)])
    rec = json.loads((arm.out / "local_training.json").read_text())
    rec["rounds"][0]["prompt_ids"] = ["p2", "p3"]
    (arm.out / "local_training.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="did not sample"):
        arm.scanned_ids({"global_step": 1, "committed_prompt_ids": ["p1"]})


def test_truncated_dynamic_sampling_journal_refuses_clearly(tmp_path):
    lines = _draws(1, ["a", "b"])
    arm, receipt = _dapo_out(tmp_path, lines, ["b"])
    with (arm.out / "dynamic_sampling.jsonl").open("a") as handle:
        handle.write('{"global_step": 2, "dra')  # a crash mid-write
    with pytest.raises(SystemExit, match="line 3 of 3 is not JSON"):
        arm.scanned_ids(receipt)


def _vlt():
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from vision_rl import vlm_local_training as vlt
    return vlt


PROFILE_BASE = {"response_format": "answer_only", "min_pixels": 50176, "max_pixels": 200704,
                "max_completion_length": 256, "num_generations": 4}


def test_assay_profile_accepts_controls_without_temperature_and_sft_with_none(tmp_path):
    vlt = _vlt()
    control = dict(PROFILE_BASE, id="c")                     # design.py controls: no key
    sft = dict(PROFILE_BASE, id="s", temperature=None)        # every SFT-family arm
    grpo = dict(PROFILE_BASE, id="g", temperature=0.7)
    # forced fire: the old construction fails exactly as reported, for both shapes
    with pytest.raises(KeyError, match="temperature"):
        vlt.Profile(**{k: control[k] for k in arm_exec.PROFILE_KEYS})
    with pytest.raises(TypeError):
        vlt.Profile(**{k: sft[k] for k in arm_exec.PROFILE_KEYS}).validate()
    for spec, temp in ((control, 1.0), (sft, 1.0), (grpo, 0.7)):
        p = arm_exec.Arm(tmp_path, spec, tmp_path / "r").assay_profile(vlt)
        assert p.temperature == temp and p.max_pixels == 200704  # 1.0 = Profile default


def test_valid_receipt_feature_gate_admits_a_present_feature():
    state = {"payload": {"sha256": "P"}, "hosts": {"h": {"receipt": {
        "valid": True, "payload_sha256": "P", "root": "/r",
        "features": ["exploratory_sweep", "cuda"]}}}}
    assert runner.valid_receipt(state, "h", ["cuda"])["root"] == "/r"  # old code refused
    with pytest.raises(SystemExit, match=r"lacks features \['expanded_lora'\]"):
        runner.valid_receipt(state, "h", ["cuda", "expanded_lora"])


def test_continuation_of_an_arm_that_died_in_its_first_assay():
    spec = {"id": "sftall-W-1e-5", "stops": [2329, 4658], "max_steps_total": 4658,
            "mode": "sft", "family": "sftall-W"}
    with pytest.raises(SystemExit, match="proper prefix"):
        runner.continuation_spec(spec, {"endpoints": []})
    cont = runner.continuation_spec(spec, {"endpoints": []}, first_stage_ran=True)
    assert (cont["stops"], cont["resume_stop"], cont["reuse_kept"], cont["save_steps"]) == (
        [2329, 4658], None, True, 2329)


def _kept(work, stop, step, sha=None):
    import hashlib
    k = work / "kept" / f"step{stop}"
    (k / "final_adapter").mkdir(parents=True)
    (k / "final_adapter" / "adapter_model.safetensors").write_bytes(b"w%d" % stop)
    real = hashlib.sha256(b"w%d" % stop).hexdigest()
    (k / "endpoint_receipt.json").write_text(json.dumps(
        {"global_step": step, "status": "stop_after_steps", "committed_active_seconds": 9.0,
         "final_adapter": {"weights_sha256": sha or real}}))


def test_run_reassays_a_kept_stop_then_trains_the_next_from_its_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "RECEIPT.json").write_text("{}")
    spec = {"id": "a", "mode": "sft", "stops": [2329, 4658], "max_steps_total": 4658,
            "reuse_kept": True, "resume_stop": None, "prior_endpoints": []}
    arm = arm_exec.Arm(tmp_path, spec, tmp_path / "res")
    _kept(arm.work, 2329, 2329)
    (arm.out / "checkpoint-2329").mkdir(parents=True)
    stages, assays = [], []
    monkeypatch.setattr(arm, "manifest", lambda: tmp_path / "m.jsonl")
    monkeypatch.setattr(arm, "trainer_args", lambda m, total, stop, resume, **k: [stop, resume])
    monkeypatch.setattr(arm, "run_stage", lambda label, args: stages.append((label, args)) or {
        "status": "max_steps", "global_step": 4658, "committed_active_seconds": 20.0})
    monkeypatch.setattr(arm, "keep_adapter", lambda label: tmp_path / "fa")
    monkeypatch.setattr(arm, "copy_small", lambda: None)
    monkeypatch.setattr(arm, "assay", lambda adapter, label, m: assays.append((adapter, label))
                        or {"accuracy": 0.5})
    summary = arm_exec.run(arm, False)
    assert stages == [("train-4658", [4658, arm.out / "checkpoint-2329"])]
    assert assays[0] == (arm.work / "kept" / "step2329" / "final_adapter", "step2329")
    assert [e["label"] for e in summary["endpoints"]] == ["step2329", "step4658"]
    assert summary["endpoints"][0]["reassayed_kept"] is True


def test_kept_endpoint_is_used_only_when_it_is_exactly_that_stop(tmp_path):
    arm = arm_exec.Arm(tmp_path, {"id": "a"}, tmp_path / "r")
    assert arm.kept_endpoint(4658) is None                 # nothing kept: it trains
    _kept(arm.work, 4658, 4000)
    assert arm.kept_endpoint(4658) is None                 # wrong step: it trains
    _kept(arm.work, 2329, 2329)
    assert arm.kept_endpoint(2329)[0]["global_step"] == 2329
    done = arm_exec.Arm(tmp_path, {"id": "a", "prior_endpoints": [{"label": "step2329"}]},
                        tmp_path / "r")
    assert done.kept_endpoint(2329) is None                # already scored: not re-assayed


def test_sftall_that_died_in_its_first_assay_is_continuable_and_its_host_refreshable(
        tmp_path, monkeypatch):
    res = tmp_path / "res"
    res.mkdir()
    (res / "train-2329.log.txt").write_text("trained\n")  # no summary.json: died in assay
    spec = {"id": "sftall-W-1e-5", "family": "sftall-W", "stops": [2329, 4658],
            "max_steps_total": 4658, "mode": "sft", "features": []}
    (tmp_path / "arm.json").write_text(json.dumps(spec))
    x = {"status": "FAILED", "host": "iccad13", "run": "r", "kind": "train",
         "spec_path": str(tmp_path / "arm.json"), "results_dir": str(res)}
    state = {"payload": {"sha256": "P"}, "uwrun_runs": [], "arms": {"sftall-W-1e-5": x},
             "hosts": {"iccad13": {"receipt": {"valid": True, "payload_sha256": "P",
                                               "root": "/r", "features": ["exploratory_sweep"]}}}}
    assert runner.continuable_hosts(state) == {"iccad13"}
    path = setup(tmp_path, monkeypatch, state)
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)
    sent = []
    monkeypatch.setattr(runner, "launch_one", lambda sp, spec, host, root, t, dry:
                        sent.append((spec, host, dry)))

    class A:
        arm, reason, timeout, dry_run = "sftall-W-1e-5", "assay TypeError", 60, True

    assert runner.cmd_continue(A, path) == 0
    cont, host, _ = sent[0]
    assert (host, cont["stops"], cont["reuse_kept"], cont["resume_stop"]) == (
        "iccad13", [2329, 4658], True, None)


def test_kept_adapter_whose_bytes_differ_from_its_receipt_refuses(tmp_path):
    arm = arm_exec.Arm(tmp_path, {"id": "a"}, tmp_path / "r")
    _kept(arm.work, 2329, 2329, sha="0" * 64)
    with pytest.raises(SystemExit, match="kept step2329 adapter sha"):
        arm.kept_endpoint(2329)


def test_a_reuse_kept_continuation_that_dies_again_stays_continuable(tmp_path):
    res = tmp_path / "res2"
    res.mkdir()  # the re-assay run: no training log, no summary
    spec = {"id": "s", "family": "sftall-W", "stops": [2329, 4658], "stops_all": [2329, 4658],
            "reuse_kept": True, "max_steps_total": 4658, "mode": "sft"}
    (tmp_path / "arm.json").write_text(json.dumps(spec))
    x = {"status": "FAILED", "host": "iccad13", "spec_path": str(tmp_path / "arm.json"),
         "results_dir": str(res), "failure_reason": "oom-guard:host-ram"}
    state = {"arms": {"s": x}}
    assert runner.continuable_hosts(state) == {"iccad13"}
    assert runner.retire_if_host_oom(state, "s") is None and state["arms"]["s"] is x
    cont = runner.continuation_spec(spec, runner.read_summary(x),
                                    first_stage_ran=runner.first_stage_ran(spec, x))
    assert cont["stops"] == [2329, 4658] and cont["reuse_kept"] is True


def test_host_oom_in_a_first_assay_is_not_retired(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "train-2329.log.txt").write_text("x")
    (tmp_path / "arm.json").write_text(json.dumps({"id": "s", "stops": [2329, 4658]}))
    state = {"arms": {"s": {"status": "FAILED", "failure_reason": "oom-guard:host-ram",
                            "results_dir": str(res), "spec_path": str(tmp_path / "arm.json")}}}
    assert runner.retire_if_host_oom(state, "s") is None


def test_launch_places_an_avx512_arm_only_on_an_avx512_host(tmp_path, monkeypatch):
    arm = dict(base_arm("d", "rl"), need_avx512=True)
    rec = {"valid": True, "payload_sha256": "P", "root": "/r", "features": ["exploratory_sweep"]}
    state = {"arms": {}, "uwrun_runs": [], "payload": {"sha256": "P"}, "gate0": {"passed": True},
             "d37_release": {"verified": True},
             "hosts": {"iccad5": {"receipt": dict(rec)}, "iccad7": {"receipt": dict(rec)},
                       "iccad9": {"receipt": dict(rec)}}}
    path = setup(tmp_path, monkeypatch, state)
    table = tmp_path / "t.json"
    table.write_text(json.dumps(make_table([arm])))
    rows = runner.host_table(
        "host load1 nproc cores availGB ourThr localGB avx\n"
        "iccad5 0 40 20 400 0 50 2\niccad7 0 72 36 200 0 50 512\n"
        "iccad9 0 48 24 900 0 50 512\n")
    rows["iccad9"]["load1"] = 30.0  # busy: iccad7 is the only AVX-512 host with room
    monkeypatch.setattr(runner, "host_table", lambda: rows)
    monkeypatch.setattr(runner, "ASSAY_PEAK_GIB", 10.0)
    sent = []
    monkeypatch.setattr(runner, "launch_one",
                        lambda sp, spec, host, root, t, dry: sent.append(host))

    class A:
        ids = wave = rung = hosts = None
        max_launch, timeout, dry_run = 1000, 60, False

    A.table = str(table)
    runner.cmd_launch(A, path)
    assert sent == ["iccad7"]  # iccad5 (AVX2) has the most free memory but never qualifies
    rows["iccad7"]["avx"] = None  # a host whose AVX width is unknown is refused too
    sent.clear()
    runner.cmd_launch(A, path)
    assert sent == []


def test_d37_table_is_corrections_compliant_and_retires_every_old_grpo_i_spec():
    import importlib.util as iu
    spec = iu.spec_from_file_location("d37", ROOT / "scripts" / "exploratory_sweep" / "d37_arms.py")
    d37 = iu.module_from_spec(spec)
    spec.loader.exec_module(d37)
    t = json.loads((ROOT / "docs" / "aircraft-exploratory-sweep-v2" / "arms-d37-grpo-I.json")
                   .read_text())
    assert t == json.loads(d37.render())  # the committed table is the generator's
    rl = [a for a in t["arms"] if a["track"] == "rl"]
    ctl = [a for a in t["arms"] if a["track"] == "control"]
    main = [a for a in rl if a["family"] == "d37-grpo-I"]
    assert sorted((a["lr"], a["temperature"]) for a in main) == [
        (1e-5, 1.0), (1e-5, 1.3), (2e-5, 1.0), (2e-5, 1.3), (5e-5, 1.0), (5e-5, 1.3)]
    assert [(a["lr"], a["temperature"]) for a in rl if a["family"] == "d37-guard"] == [(3e-6, 1.0)]
    for a in rl:
        assert (a["init"], a["beta"], a["num_generations"], a["dtype"], a["device"]) == (
            "I", 0.0, 8, "float32", "cpu")
        assert a["need_avx512"] is True and a["stops"] == [150, 300, 600]
        assert a["max_steps_total"] == 600 and "dynamic_sampling" in a["features"]
        i = a["extra_trainer_args"].index("--dynamic-sampling-max-draws")
        assert int(a["extra_trainer_args"][i + 1]) >= 8
    assert sorted({c["lr"] for c in ctl}) == [5e-6, 1e-5, 2e-5]
    assert {c["control_of"] for c in ctl} == {a["id"] for a in rl}
    for c in ctl:
        assert (c["init"], c["mode"], c["dtype"], c["need_avx512"]) == ("I", "sft", "float32", True)
        assert "replay_sequence" in c["features"] and c["control_replays"] == "scanned"
    rung1 = json.loads((ROOT / "docs" / "aircraft-exploratory-sweep-v2" / "arms-rung1.json")
                       .read_text())
    old = {a["id"] for a in rung1["arms"]
           if a["family"] == "grpo-I" or str(a.get("control_of", "")).startswith("grpo-I-")}
    assert len(old) == 12 and set(t["retires"]) == old
    held = {a["id"]: runner.launch_hold({"gate0": {"passed": True}}, a,
                                        {x["id"]: x for x in rung1["arms"]})
            for a in rung1["arms"] if a["id"] in old}
    assert all("allowlist" in v for v in held.values())


GUARDED_BASE = ["-m", "vision_rl.vlm_local_training", "--train", "m.jsonl", "--dtype", "float32",
                "--learning-rate", "1e-05", "--max-steps", "600"]


@pytest.mark.parametrize("extras,why", [
    (["--dtype", "bfloat16"], "not allowlisted"),             # override of a guarded flag
    (["--learning-rate", "1"], "not allowlisted"),
    (["--learning", "1"], "not allowlisted"),                 # argparse abbreviation
    (["--train", "x.jsonl"], "may not pass --train"),
    (["--validation", "v.jsonl"], "may not pass --validation"),
    (["--train=x.jsonl"], "may not pass"),
    (["--dynamic-sampling-max-draws=8"], "not allowlisted"),  # `=` form
    (["--dynamic-sampling-max-draws", "4"], "integer >= 8"),  # C5
    (["--dynamic-sampling-max-draws", "--dtype"], "integer >= 8"),
    (["--dynamic-sampling-max-draws"], "no value"),
    (["--dynamic-sampling-max-draws", "8", "--dynamic-sampling-max-draws", "9"], "twice"),
])
def test_extra_trainer_args_refuse_overrides_and_unlisted_flags(extras, why):
    with pytest.raises(SystemExit, match=why):
        arm_exec.checked_extras(extras, GUARDED_BASE)


def test_extra_trainer_args_refuse_a_flag_arm_exec_already_passes(monkeypatch):
    monkeypatch.setitem(arm_exec.EXTRA_FLAGS, "--dtype", lambda v: None)  # even if listed
    with pytest.raises(SystemExit, match="override arm_exec's own value"):
        arm_exec.checked_extras(["--dtype", "bfloat16"], GUARDED_BASE)


def test_trainer_args_end_with_the_checked_d37_extras_and_refuse_a_bad_table_row(tmp_path):
    t = json.loads((ROOT / "docs" / "aircraft-exploratory-sweep-v2" / "arms-d37-grpo-I.json")
                   .read_text())
    rl = next(a for a in t["arms"] if a["track"] == "rl")
    arm = arm_exec.Arm(tmp_path, rl, tmp_path / "r")
    argv = arm.trainer_args(tmp_path / "m.jsonl", 600, 150, None)
    assert argv[-2:] == ["--dynamic-sampling-max-draws", "8"]
    assert argv.count("--dtype") == 1 and argv.count("--learning-rate") == 1
    bad = arm_exec.Arm(tmp_path, dict(rl, extra_trainer_args=["--dtype", "bfloat16"]),
                       tmp_path / "r")
    with pytest.raises(SystemExit, match="not allowlisted"):
        bad.trainer_args(tmp_path / "m.jsonl", 600, 150, None)


# ---- wave-2 extras (amendment 4): each table flag has an exact-name entry and a validator
@pytest.mark.parametrize("extras,why", [
    (["--loss-type", "cispo"], "one of"),                       # outside the plan's values
    (["--loss-type=dapo"], "not allowlisted"),                  # `=` form
    (["--loss", "dapo"], "not allowlisted"),                    # argparse abbreviation
    (["--loss-type", "dapo", "--loss-type", "grpo"], "twice"),  # last value would win
    (["--scale-rewards", "batch"], "one of"),
    (["--epsilon-high", "0.1"], "outside"),                     # below epsilon 0.2
    (["--epsilon-high", "nan"], "outside"),
    (["--num-iterations", "9"], "1..8"),
    (["--family-credit", "1"], "outside"),
    (["--manufacturer-credit", "-0.1"], "outside"),
    (["--invalid-reward", "0.5"], "outside"),
    (["--invalid-reward", "x"], "a number"),
    (["--label-hierarchy", "/etc/passwd"], "label-hierarchy-train-v1.json"),
    (["--label-hierarchy", "label-hierarchy-train-v1.json"], "label-hierarchy-train-v1.json"),
    (["--beta", "0.04"], "not allowlisted"),                    # arm_exec's own GRPO flag
    (["--response-format", "think_answer"], "not allowlisted"),
])
def test_wave2_extra_trainer_args_refuse_bad_values_and_forms(extras, why):
    with pytest.raises(SystemExit, match=why):
        arm_exec.checked_extras(extras, GUARDED_BASE)


def test_every_wave2_table_extra_passes_the_allowlist_and_a_tampered_row_refuses(tmp_path):
    root, asset = _fake_root(tmp_path)
    arms = [a for a in json.loads(wave2.OUT.read_text())["arms"] if a.get("extra_trainer_args")]
    assert {a["family"] for a in arms} == {"wave2-fmtpen", "wave2-closedset", "wave2-drgrpo",
                                           "wave2-mu2", "wave2-dapo", "wave2-guard"}
    for a in arms:  # resolved @payload/ path, then the allowlist: every row passes as is
        argv = arm_exec.Arm(root, a, tmp_path / "r").trainer_args(
            tmp_path / "m.jsonl", int(a["max_steps_total"]), 150, None)
        assert argv[-len(a["extra_trainer_args"]):][::2] == a["extra_trainer_args"][::2]
    dapo = next(a for a in arms if a["family"] == "wave2-dapo")
    for tamper in (["--learning-rate", "1e-2"], ["--train", "x.jsonl"],
                   ["--dynamic-sampling-max-draws", "4"]):
        bad = arm_exec.Arm(root, dict(dapo, extra_trainer_args=dapo["extra_trainer_args"]
                                      + tamper), tmp_path / "r")
        with pytest.raises(SystemExit, match="REFUSE"):
            bad.trainer_args(tmp_path / "m.jsonl", 1500, 150, None)
    rft = next(a for a in json.loads(wave2.OUT.read_text())["arms"] if a["mode"] == "rft")
    bad = arm_exec.Arm(root, dict(rft, extra_trainer_args=["--learning-rate", "1"]),
                       tmp_path / "r")
    with pytest.raises(SystemExit, match="not allowlisted"):  # RFT argv is checked too
        bad.trainer_args(tmp_path / "m.jsonl", 1500, 150, None)
