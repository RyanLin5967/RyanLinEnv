"""RL campaign launcher (scripts/rl_campaign/campaign.py) and the reused sweep pieces it extends.

Expected values are written by hand from the fixtures, never computed by the subject.
No uwrun call is made: every test works on specs, files and pure functions.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camp = load("rl_campaign", "scripts/rl_campaign/campaign.py")
arm_exec = camp.arm_exec


def table(**defaults):
    return {"study_id": camp.STUDY, "defaults": defaults, "arms": []}


def grpo_row(**kw):
    row = {"id": "a", "device_class": "cpu-fp32", "mode": "grpo", "lr": 1e-5, "track": "smoke",
           "stops": [50, 100], "max_steps_total": 200}
    row.update(kw)
    return row


# ---- spec building ------------------------------------------------------------------------
def test_spec_starts_from_I_and_carries_save_steps():
    spec = camp.build_spec(table(), grpo_row())
    assert spec["init"] == "I" and spec["device"] == "cpu" and spec["dtype"] == "float32"
    assert spec["threads"] == 16 and spec["save_steps"] == 50  # gcd(50, 100)
    assert spec["extra_trainer_args"] == []  # beta 0.04: no explicit loss type needed
    gpu = camp.build_spec(table(), grpo_row(device_class="gtx1070-fp16", mode="sft"))
    assert (gpu["device"], gpu["dtype"], gpu["features"]) == ("cuda", "float16", ["cuda"])


def test_beta_zero_grpo_declares_the_legacy_loss_type():
    spec = camp.build_spec(table(), grpo_row(beta=0.0))
    assert spec["extra_trainer_args"] == ["--loss-type", "grpo"]
    kept = camp.build_spec(table(), grpo_row(beta=0.0, extra_trainer_args=["--loss-type",
                                                                           "dr_grpo"]))
    assert kept["extra_trainer_args"] == ["--loss-type", "dr_grpo"]


@pytest.mark.parametrize("row", [
    grpo_row(init="W"),                       # not from I
    grpo_row(stops=[100, 50]),                # not increasing
    grpo_row(stops=[50, 300]),                # past max_steps_total
    grpo_row(device_class="rtx3070-bfloat16"),  # GPU lane's class
    grpo_row(mode="rft"),
    grpo_row(stops=[60, 100], save_steps=40),  # 60 is not a multiple of the carried 40
    grpo_row(device_class="gtx1070-fp16"),    # C2: GRPO only in fp32
    grpo_row(device_class="gtx1070-fp32", mode="sft"),  # measured: fp32 weights exceed 8 GB
    grpo_row(extra_trainer_args=["--dynamic-sampling-max-draws", "4"]),  # C5: < 8 draws
    grpo_row(track="rl", group="g", rung=1),  # table has no fixed rung_stops (C11)
    grpo_row(track="rl", rung=1),             # an RL arm needs a best-of-k group
])
def test_spec_refusals(row):
    with pytest.raises(SystemExit):
        camp.build_spec(table(), row)


def test_gpu_lane_and_no_avx2_hosts_are_refused():
    for host in ("ecetesla2", "ecetesla4", "eceubuntu2", "iccad1", "iccad4"):
        with pytest.raises(SystemExit):
            camp.refuse_hosts([host])
    camp.refuse_hosts(["iccad7", "ecetesla3"])


def test_device_classes_from_receipts():
    assert camp.device_classes_for({"device": {"kind": "cpu"}, "avx2": True}) == ["cpu-fp32"]
    assert camp.device_classes_for({"device": {"kind": "cpu"}, "avx2": False}) == []
    card = {"device": {"kind": "cuda", "gpu": "NVIDIA GeForce GTX 1070"}}
    assert camp.device_classes_for(card) == ["gtx1070-fp16", "gtx1070-fp32"]
    assert camp.device_classes_for({"device": {"kind": "cuda", "gpu": "RTX 3070"}}) == []


def test_cpu_worker_mem_needs_a_measured_peak_or_an_explicit_value():
    with pytest.raises(SystemExit, match="no measured peak"):
        camp.worker_mem_gb({"measured": {}}, "cpu-fp32", None)
    assert camp.worker_mem_gb({}, "cpu-fp32", 30) == 30.0
    # 13.6 GB measured -> 13.6 * 1.2 = 16.32, rounded up to 16.4
    assert camp.worker_mem_gb({"measured": {"cpu-fp32": {"peak_rss_gb": 13.6}}},
                              "cpu-fp32", None) == 16.4
    assert camp.worker_mem_gb({}, "gtx1070-fp32", None) == 24.0


# ---- controls -------------------------------------------------------------------------------
def test_control_is_sft_from_I_time_matched_at_every_rl_checkpoint():
    rl = camp.build_spec(table(), grpo_row())
    summary = {"endpoints": [{"label": "step50", "steps": 50, "active_seconds": 900.5},
                             {"label": "step100", "steps": 100, "active_seconds": 1850.0}]}
    receipt = {"global_step": 100, "committed_prompt_ids": [f"p{i}" for i in range(100)]}
    ctl = camp.control_spec(rl, summary, receipt)
    assert ctl["id"] == "a~ctl" and ctl["mode"] == "sft" and ctl["init"] == "I"
    assert ctl["lr"] == 5e-6 and ctl["order"] == "sequential" and ctl["cycle"] is True
    assert ctl["time_stops"] == [900.5, 1850.0] and ctl["stops"] == []
    assert ctl["matched_steps"] == [50, 100] and ctl["train_ids"][0] == "p0"
    assert ctl["device_class"] == "cpu-fp32" and ctl["save_steps"] == 100000
    assert "same_host_as" not in ctl and ctl["control_of"] == "a"
    assert arm_exec.time_endpoints(ctl) == [("time1", 900.5), ("time2", 1850.0)]
    with pytest.raises(SystemExit):
        camp.control_spec(rl, summary, dict(receipt, global_step=99))


def test_time_endpoints_keep_the_legacy_single_label_and_refuse_bad_lists():
    assert arm_exec.time_endpoints({"time_budget_seconds": 12.5}) == [("time", 12.5)]
    assert arm_exec.time_endpoints({}) == []
    for bad in ({"time_stops": [5, 5]}, {"time_stops": [0, 3]},
                {"time_stops": [3], "time_budget_seconds": 3}):
        with pytest.raises(SystemExit):
            arm_exec.time_endpoints(bad)


def test_time_only_control_argv_has_no_resume_and_carries_save_steps(tmp_path):
    (tmp_path / "payload/adapters/I/final_adapter").mkdir(parents=True)
    (tmp_path / "payload/adapters/I/local_training.json").write_text(
        json.dumps({"control_identity": {"profile": {"dtype": "float32"}}}))
    rl = camp.build_spec(table(), grpo_row())
    ctl = camp.control_spec(rl, {"endpoints": [{"label": "step50", "steps": 50,
                                                "active_seconds": 10.0}]},
                            {"global_step": 50, "committed_prompt_ids": ["x"]})
    arm = arm_exec.Arm(tmp_path, ctl, tmp_path / "r")
    argv = arm.trainer_args(tmp_path / "m.jsonl", 100000, None, None, stop_seconds=10.0)
    assert argv[argv.index("--save-steps") + 1] == "100000"
    assert argv[argv.index("--stop-after-active-seconds") + 1] == "10.0"
    assert "--resume-from-checkpoint" not in argv and argv[argv.index("--mode") + 1] == "sft"


# ---- report ---------------------------------------------------------------------------------
GOLD = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}


def test_stratified_bootstrap_hand_values():
    same = {"a1": "A", "a2": "B", "b1": "B", "b2": "A"}
    r = camp.stratified_bootstrap(same, same, GOLD, 200)
    assert r["delta"] == 0 and r["ci95"] == [0, 0] and r["p_le_0"] == 1.0
    better = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    worse = {"a1": "B", "a2": "B", "b1": "A", "b2": "A"}
    r = camp.stratified_bootstrap(better, worse, GOLD, 200)
    # every item is a win, inside every class: no resample can change it
    assert r["delta"] == 1.0 and r["ci95"] == [1.0, 1.0] and r["p_le_0"] == 0.0
    assert (r["wins"], r["losses"], r["classes"]) == (4, 0, 2)


def test_stratification_keeps_class_counts():
    # class A: one win of two; class B: no difference. Stratified replicates can only move
    # class A's win (0, 1 or 2 of 2 draws), so every replicate delta is in {0, .25, .5}.
    a = {"a1": "A", "a2": "B", "b1": "B", "b2": "B"}
    b = {"a1": "B", "a2": "B", "b1": "B", "b2": "B"}
    r = camp.stratified_bootstrap(a, b, GOLD, 500)
    assert r["delta"] == 0.25
    assert r["ci95"][0] in (0.0, 0.25, 0.5) and r["ci95"][1] in (0.0, 0.25, 0.5)
    assert 0.15 < r["p_le_0"] < 0.35  # P(class A draws no win) = 1/4


def _arm_dir(base, arm_id, spec, host, endpoints, assays):
    d = base / arm_id
    (d / "assays").mkdir(parents=True)
    (d / "arm.json").write_text(json.dumps(dict(spec, id=arm_id)))
    (d / "summary.json").write_text(json.dumps({"host": host, "endpoints": endpoints}))
    for label, pred in assays.items():
        correct = sum(pred[i] == g for i, g in GOLD.items()) / 4
        (d / "assays" / f"{label}.json").write_text(json.dumps({
            "metrics": {"accuracy": correct, "valid_fraction": 1.0,
                        "equal_class_accuracy": correct}, "predictions": pred}))
    return arm_id, d


def test_report_per_checkpoint_vs_same_host_I_and_matched_control(tmp_path):
    right = dict(GOLD)
    half = {"a1": "A", "a2": "B", "b1": "B", "b2": "A"}
    wrong = {"a1": "B", "a2": "B", "b1": "A", "b2": "A"}
    cls = {"device_class": "cpu-fp32"}
    dirs = [
        _arm_dir(tmp_path, "eval-I-h1", dict(cls, mode="eval"), "h1.uwaterloo.ca",
                 [{"label": "init", "steps": 0}], {"init": half}),
        _arm_dir(tmp_path, "rl", dict(cls, mode="grpo"), "h1",
                 [{"label": "step1", "steps": 1, "active_seconds": 5.0},
                  {"label": "step2", "steps": 2, "active_seconds": 9.0}],
                 {"step1": half, "step2": right}),
        _arm_dir(tmp_path, "rl~ctl", dict(cls, mode="sft", track="control", matched_steps=[1, 2]),
                 "h1",
                 [{"label": "time1", "steps": 7, "active_seconds": 5.1},
                  {"label": "time2", "steps": 13, "active_seconds": 9.2}],
                 {"time1": wrong, "time2": half}),
    ]
    out = camp.build_report(camp.collect(dirs), GOLD, 100)
    assert out["missing"] == [] and out["baselines"]["host/h1"]["arm"] == "eval-I-h1"
    step1, step2 = out["checkpoints"]
    assert (step1["label"], step1["vs_I"]["delta"], step1["vs_control"]["delta"]) == (
        "step1", 0.0, 0.5)
    assert (step2["vs_I"]["delta"], step2["vs_control"]["delta"]) == (0.5, 0.5)
    assert step2["control"]["steps"] == 13 and step2["control"]["label"] == "time2"


def test_report_names_an_arm_without_a_same_device_baseline(tmp_path):
    dirs = [_arm_dir(tmp_path, "rl", {"device_class": "gtx1070-fp16", "mode": "grpo"}, "h2",
                     [{"label": "step1", "steps": 1}], {"step1": dict(GOLD)})]
    out = camp.build_report(camp.collect(dirs), GOLD, 50)
    assert out["checkpoints"] == [] and out["missing"] == [
        "rl: no I baseline on h2 or gtx1070-fp16"]


# ---- root corrections (runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md) -----------------
def rl_table(**kw):
    t = {"study_id": camp.STUDY, "defaults": {}, "rung_stops": {"1": [50, 100], "2": [200]},
         "arms": []}
    t.update(kw)
    return t


def test_rl_arm_needs_a_group_and_the_fixed_rung_stops():
    ok = camp.build_spec(rl_table(), grpo_row(track="rl", group="g"))
    assert ok["stops"] == [50, 100] and ok["group"] == "g"
    for row in (grpo_row(track="rl"), grpo_row(track="rl", group="g", stops=[25, 100])):
        with pytest.raises(SystemExit):
            camp.build_spec(rl_table(), row)


def test_control_lrs_pair_by_rank_and_need_equal_k():
    rows = [grpo_row(id=f"r{lr}", lr=lr, track="rl", group="g") for lr in (5e-5, 1e-5, 2e-5)]
    got = camp.assign_control_lrs(rl_table(arms=rows))
    # lowest RL LR <-> lowest control LR (5e-6, 1e-5, 2e-5 by hand)
    assert got == {"r1e-05": {"control_lr": 5e-6}, "r2e-05": {"control_lr": 1e-5},
                   "r5e-05": {"control_lr": 2e-5}}
    with pytest.raises(SystemExit, match="best-of-k"):
        camp.assign_control_lrs(rl_table(arms=rows[:2]))
    plac = {"id": "p", "track": "placebo", "mode": "sft", "lr": 1e-5, "seed": 42,
            "placebo_control_seed": 314}
    assert camp.assign_control_lrs(rl_table(arms=[plac]))["p"] == {"control_lr": 1e-5,
                                                                    "control_seed": 314}
    with pytest.raises(SystemExit):
        camp.assign_control_lrs(rl_table(arms=[dict(plac, placebo_control_seed=42)]))


def test_control_trains_on_scanned_rows_charges_pool_seconds_and_takes_the_paired_lr():
    rl = dict(camp.build_spec(table(), grpo_row()), control_lr=2e-5, pool_scoring_seconds=100.0)
    summary = {"endpoints": [{"label": "step50", "steps": 50, "active_seconds": 900.0},
                             {"label": "step100", "steps": 100, "active_seconds": 1800.0}]}
    receipt = {"global_step": 100, "committed_prompt_ids": ["c1", "c2"]}
    ctl = camp.control_spec(rl, summary, receipt, scanned=["s1", "s2", "s3"])
    assert ctl["train_ids"] == ["s1", "s2", "s3"] and ctl["lr"] == 2e-5
    assert ctl["time_stops"] == [1000.0, 1900.0] and ctl["pool_scoring_seconds_charged"] == 100
    placebo = camp.control_spec(dict(rl, control_seed=314, track="placebo"), summary, receipt)
    assert placebo["seed"] == 314 and placebo["placebo"] is True
    assert placebo["train_ids"] == ["c1", "c2"]


def test_report_refuses_a_time_match_under_different_cotenancy(tmp_path):
    cls = {"device_class": "cpu-fp32"}
    dirs = [
        _arm_dir(tmp_path, "eval-I", dict(cls, mode="eval"), "h1", [{"label": "init"}],
                 {"init": dict(GOLD)}),
        _arm_dir(tmp_path, "rl", dict(cls, mode="grpo"), "h1",
                 [{"label": "step1", "steps": 1}], {"step1": dict(GOLD)}),
        _arm_dir(tmp_path, "rl~ctl", dict(cls, mode="sft", track="control", matched_steps=[1]),
                 "h1", [{"label": "time1", "steps": 3}], {"time1": dict(GOLD)}),
    ]
    (tmp_path / "rl~ctl" / "stages.json").write_text(json.dumps(
        [{"stage": "train-time1", "cotenancy": {"shared": True}}]))
    out = camp.build_report(camp.collect(dirs), GOLD, 50)
    assert "vs_control" not in out["checkpoints"][0]
    assert out["refused"] == ["rl step1: co-tenancy differs from its control (rl False, ctl True)"]


def test_report_quotes_a_continuation_only_after_a_passed_e0(tmp_path):
    cls = {"device_class": "cpu-fp32"}
    dirs = [
        _arm_dir(tmp_path, "eval-I", dict(cls, mode="eval"), "h1", [{"label": "init"}],
                 {"init": dict(GOLD)}),
        _arm_dir(tmp_path, "rl", dict(cls, mode="grpo", resume_stop=100, rung=2), "h1",
                 [{"label": "step200", "steps": 200}], {"step200": dict(GOLD)}),
    ]
    out = camp.build_report(camp.collect(dirs), GOLD, 50)
    assert out["checkpoints"] == [] and "E0" in out["refused"][0]
    out = camp.build_report(camp.collect(dirs), GOLD, 50, e0_passed={"cpu-fp32"})
    assert [e["label"] for e in out["checkpoints"]] == ["step200"]


def _entry(arm, group, track, acc, cacc, pred, cpred, label="step1"):
    return {"arm": arm, "group": group, "track": track, "rung": 1, "label": label,
            "accuracy": acc, "equal_class_accuracy": acc, "vs_control": {},
            "control": {"arm": arm + "~ctl", "label": "time1", "accuracy": cacc},
            "_pred": pred, "_cpred": cpred}


def test_best_of_k_takes_the_best_on_each_side_with_equal_k():
    right, wrong = dict(GOLD), {"a1": "B", "a2": "B", "b1": "A", "b2": "A"}
    half = {"a1": "A", "a2": "B", "b1": "B", "b2": "A"}
    es = [_entry("x", "g", "rl", 0.5, 0.0, half, wrong),
          _entry("y", "g", "rl", 1.0, 0.5, right, half)]
    (g,) = camp.best_of_k(es, GOLD, 50)
    # best RL = y (1.0); best control = y~ctl (0.5): delta 1.0 - 0.5 by hand
    assert g["k"] == 2 and g["best_rl"]["arm"] == "y" and g["best_control"]["arm"] == "y~ctl"
    assert g["best_vs_best"]["delta"] == 0.5


def _group(track, group, delta, rung=1, acc=0.8):
    return {"track": track, "group": group, "rung": rung, "best_vs_best": {"delta": delta},
            "best_rl": {"accuracy": acc}}


def test_promotion_bar_is_the_placebo_maximum_and_refuses_without_one():
    groups = [_group("rl", f"g{i}", d) for i, d in enumerate((0.04, 0.03, 0.01, -0.01, 0.0,
                                                              0.02))]
    with pytest.raises(SystemExit, match="placebo"):
        camp.promotion(groups, 1)
    placebo = [_group("placebo", "p1", 0.012), _group("placebo", "p2", 0.035)]
    out = camp.promotion(groups + placebo, 1)
    # 6 units -> top third = 2 (g0 0.04, g1 0.03); bar = 0.035 -> only g0 clears it
    assert out["bar_placebo_max"] == 0.035 and out["kept"] == ["g0"]
    assert out["top_third_below_bar"] == ["g1"]


def test_arm_exec_refuses_bf16_rl_and_few_dynamic_draws_but_allows_bf16_sft():
    arm_exec.refuse_spec({"mode": "sft", "dtype": "bfloat16"})
    arm_exec.refuse_spec({"mode": "grpo", "dtype": "float32",
                          "extra_trainer_args": ["--dynamic-sampling-max-draws", "8"]})
    for bad in ({"mode": "grpo", "dtype": "bfloat16"}, {"mode": "rft", "dtype": "float16"},
                {"mode": "grpo", "dtype": "float32",
                 "extra_trainer_args": ["--dynamic-sampling-max-draws", "7"]}):
        with pytest.raises(SystemExit):
            arm_exec.refuse_spec(bad)


def test_arm_exec_refuses_parity_outside_tolerance(tmp_path):
    rec = tmp_path / "local_training.json"
    # the measured bf16 3070 value (C2): 0.02814 against 0.02
    rec.write_text(json.dumps({"sampler_teacher_logprob_parity": {
        "within_tolerance": False, "tolerance": 0.02,
        "max_abs_raw_teacher_vs_sampler_logprob_error": 0.02814}}))
    with pytest.raises(SystemExit, match="parity"):
        arm_exec.check_parity(rec)
    rec.write_text(json.dumps({}))
    with pytest.raises(SystemExit, match="parity"):
        arm_exec.check_parity(rec)
    rec.write_text(json.dumps({"sampler_teacher_logprob_parity": {
        "within_tolerance": True, "tolerance": 0.001,
        "max_abs_raw_teacher_vs_sampler_logprob_error": 9.6e-7}}))
    assert arm_exec.check_parity(rec)["within_tolerance"] is True


def test_scanned_ids_for_dynamic_sampling_keep_dropped_draws_in_order(tmp_path):
    spec = camp.build_spec(table(), grpo_row())
    arm = arm_exec.Arm(tmp_path, spec, tmp_path / "r")
    arm.out.mkdir(parents=True)
    rows = [(1, 0, "a"), (1, 1, "b"), (2, 0, "c"), (2, 0, "c2"), (3, 0, "late")]
    (arm.out / "dynamic_sampling.jsonl").write_text("".join(
        json.dumps({"global_step": s, "draw": d, "id": i}) + "\n" for s, d, i in rows))
    # step 2's draw 0 was redone after a resume ("c2" wins); step 3 is past the endpoint
    assert arm.scanned_ids({"global_step": 2}) == ["a", "b", "c2"]
    assert arm.scanned_ids({"global_step": 3}) == ["a", "b", "c2", "late"]
    (arm.out / "dynamic_sampling.jsonl").write_text(json.dumps(
        {"global_step": 2, "draw": 0, "id": "x"}) + "\n")
    with pytest.raises(SystemExit, match="every step"):
        arm.scanned_ids({"global_step": 2})


def test_cotenancy_summary_flags_shared_only_when_a_foreign_process_was_seen():
    c = arm_exec.CoTenancy(cuda=False)
    c.sample(1)
    s = c.summary()
    assert s["device"] == "cpu" and s["samples"] == 1 and s["shared"] is False
    c = arm_exec.CoTenancy(cuda=True)
    c.samples, c.hits, c.foreign = 4, 1, {"999": {"used_mib": "7000", "name": "python"}}
    assert c.summary()["shared"] is True and c.summary()["samples_with_cotenant"] == 1

# ---- worker -------------------------------------------------------------------------------
WORKER = ROOT / "scripts/exploratory_sweep/gpu_worker.sh"


def test_worker_refuses_an_unknown_wait_kind(tmp_path):
    p = subprocess.run(["bash", str(WORKER), str(tmp_path)], env={"WAIT_KIND": "tpu",
                       "PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=30)
    assert p.returncode == 2 and "WAIT_KIND" in p.stderr


def test_cpu_worker_runs_a_queued_spec_records_peak_rss_and_files_it_done(tmp_path):
    """A fake root whose 'python' sleeps while holding memory: the worker must take the spec
    from pending/ (NEED_CORES=-1000: never waits on a busy CI runner), record a non-zero
    peak RSS, and file the spec under done/ with rc 0."""
    root = tmp_path / "root"
    (root / "venv/bin").mkdir(parents=True)
    (root / "RECEIPT.json").write_text("{}")
    py = root / "venv/bin/python"
    py.write_text("#!/bin/bash\nexec python3 -c 'import time; x = bytearray(30 * 2**20); "
                  "time.sleep(12)'\n")
    py.chmod(0o755)
    (root / "gpuq/pending").mkdir(parents=True)
    (root / "gpuq/pending/01-smoke.json").write_text("{}")
    res = tmp_path / "res"
    env = {"WAIT_KIND": "cpu", "NEED_CORES": "-1000", "IDLE_MAX": "0", "PATH": "/usr/bin:/bin"}
    p = subprocess.run(["bash", str(WORKER), str(root), str(res)], env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    assert (root / "gpuq/done/01-smoke.json").is_file()
    doc = json.loads((res / "01-smoke/worker.json").read_text())
    assert doc["rc"] == 0 and doc["peak_rss_kb"] > 20 * 1024  # holds a 30 MiB buffer
    assert not (res / "01-smoke/gpu-trace.csv").exists()  # no nvidia-smi on a CPU worker


def test_s_per_update_uses_cumulative_endpoint_seconds_across_resumed_stages(tmp_path):
    # the fp16 smoke's shape: audit seconds restart per stage; endpoints are cumulative
    (tmp_path / "summary.json").write_text(json.dumps({"endpoints": [
        {"label": "step1", "steps": 1, "active_seconds": 20.0},
        {"label": "step3", "steps": 3, "active_seconds": 50.0}]}))
    m = camp.arm_measurements(tmp_path)
    # by hand: 20 / 1 = 20; (50 - 20) / 2 = 15; mean 50 / 3
    assert m["s_per_update_by_checkpoint"] == [20.0, 15.0] and m["updates"] == 3
    assert abs(m["s_per_update_mean"] - 50 / 3) < 1e-12
    (tmp_path / "arm.json").write_text(json.dumps({"resume_stop": 1}))
    m = camp.arm_measurements(tmp_path)
    assert m["s_per_update_by_checkpoint"] == [15.0] and m["s_per_update_mean"] == 15.0
