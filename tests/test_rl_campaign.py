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
    cpu = {"device": {"kind": "cpu"}, "avx2": True, "avx512f": True}
    assert camp.device_classes_for(cpu) == ["cpu-fp32"]
    assert camp.device_classes_for(dict(cpu, avx512f=False)) == []  # MKL SIGILL on AVX2-only
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
    with pytest.raises(SystemExit, match="scanned"):  # an RL arm must name the rows it read
        camp.control_spec(rl, summary, receipt)
    ctl = camp.control_spec(rl, summary, receipt, scanned=[f"p{i}" for i in range(100)])
    assert ctl["id"] == "a~ctl" and ctl["mode"] == "sft" and ctl["init"] == "I"
    assert ctl["lr"] == 5e-6 and ctl["order"] == "sequential" and ctl["cycle"] is True
    assert ctl["time_stops"] == [900.5, 1850.0] and ctl["stops"] == []
    assert ctl["matched_steps"] == [50, 100] and ctl["train_ids"][0] == "p0"
    assert ctl["device_class"] == "cpu-fp32" and ctl["save_steps"] == 100000
    assert "same_host_as" not in ctl and ctl["control_of"] == "a"
    assert arm_exec.time_endpoints(ctl) == [("time1", 900.5), ("time2", 1850.0)]
    with pytest.raises(SystemExit):
        camp.control_spec(rl, summary, dict(receipt, global_step=99), scanned=["p0"])


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
                            {"global_step": 50, "committed_prompt_ids": ["x"]}, scanned=["x"])
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


def _arm_dir(base, arm_id, spec, host, endpoints, assays, scanned=None):
    d = base / arm_id
    (d / "assays").mkdir(parents=True)
    if scanned is not None:  # {label: number of rows the RL arm read}
        (d / "scanned").mkdir()
        for label, n in scanned.items():
            (d / "scanned" / f"{label}.json").write_text(json.dumps(
                {"label": label, "ids": [f"r{i}" for i in range(n)]}))
    (d / "arm.json").write_text(json.dumps(dict(spec, id=arm_id)))
    (d / "summary.json").write_text(json.dumps({"host": host, "endpoints": endpoints}))
    stages = [{"stage": "train-" + (e["label"][4:] if e["label"].startswith("step")
                                    else e["label"]), "cotenancy": {"shared": False}}
              for e in endpoints if e["label"].startswith(("step", "time"))]
    if stages:  # every train stage unshared unless a test overwrites stages.json
        (d / "stages.json").write_text(json.dumps(stages))
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
        _arm_dir(tmp_path, "rl", dict(cls, mode="grpo", stops=[1, 2]), "h1",
                 [{"label": "step1", "steps": 1, "active_seconds": 5.0},
                  {"label": "step2", "steps": 2, "active_seconds": 9.0}],
                 {"step1": half, "step2": right}, scanned={"step1": 1, "step2": 2}),
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
    placebo = camp.control_spec(dict(rl, control_seed=314, track="placebo", mode="sft"),
                                summary, receipt)
    assert placebo["seed"] == 314 and placebo["placebo"] is True
    assert placebo["train_ids"] == ["c1", "c2"]


def test_report_refuses_a_time_match_under_different_cotenancy(tmp_path):
    cls = {"device_class": "cpu-fp32"}
    dirs = [
        _arm_dir(tmp_path, "eval-I", dict(cls, mode="eval"), "h1", [{"label": "init"}],
                 {"init": dict(GOLD)}),
        _arm_dir(tmp_path, "rl", dict(cls, mode="grpo", stops=[1]), "h1",
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


def _group(track, group, delta, rung=1, acc=0.8, k=3):
    return {"track": track, "group": group, "rung": rung, "best_vs_best": {"delta": delta},
            "best_rl": {"accuracy": acc}, "k": k}


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
    # a placebo from another rung or with another k never sets the bar
    with pytest.raises(SystemExit, match="placebo"):
        camp.promotion(groups + [_group("placebo", "p", 0.0, rung=2),
                                 _group("placebo", "q", 0.0, k=2)], 1)
    other = camp.promotion(groups + placebo + [_group("placebo", "big", 0.5, rung=2)], 1)
    assert other["bar_placebo_max"] == 0.035


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
    fp32 = {"control_identity": {"profile": {"dtype": "float32"}}}
    rec.write_text(json.dumps({}))  # no recorded dtype: refuses before parity
    with pytest.raises(SystemExit, match="dtype"):
        arm_exec.check_parity(rec)
    rec.write_text(json.dumps(fp32))  # fp32 but no parity record
    with pytest.raises(SystemExit, match="parity"):
        arm_exec.check_parity(rec)
    # the measured bf16 3070 value (C2): 0.02814 against 0.02
    rec.write_text(json.dumps(dict(fp32, sampler_teacher_logprob_parity={
        "within_tolerance": False, "tolerance": 0.02,
        "max_abs_raw_teacher_vs_sampler_logprob_error": 0.02814})))
    with pytest.raises(SystemExit, match="parity"):
        arm_exec.check_parity(rec)
    ok = dict(fp32, sampler_teacher_logprob_parity={
        "within_tolerance": True, "tolerance": 0.001,
        "max_abs_raw_teacher_vs_sampler_logprob_error": 9.6e-7})
    rec.write_text(json.dumps(ok))
    assert arm_exec.check_parity(rec)["within_tolerance"] is True
    # the RECORDED dtype decides, not the spec: a bf16 run inside bf16's 0.02 tolerance refuses
    rec.write_text(json.dumps(dict(ok, control_identity={"profile": {"dtype": "bfloat16"}})))
    with pytest.raises(SystemExit, match="bfloat16"):
        arm_exec.check_parity(rec)


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
    c = arm_exec.CoTenancy(cuda=False, threads=10**6)  # no load can exceed this
    c.sample(1)
    s = c.summary()
    assert s["device"] == "cpu" and s["samples"] == 1 and s["shared"] is False
    busy = arm_exec.CoTenancy(cuda=False, threads=-2)  # any load1 >= 0 exceeds -2 + 1
    busy.sample(1)
    assert busy.summary()["shared"] is True and busy.summary()["samples_with_cotenant"] == 1
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


def test_time_stops_accept_increasing_positive_lists_and_refuse_the_rest():
    # cases from the GPU lane's fix (6205487); the old `not 0 < a < b` refused all of them
    assert arm_exec.time_endpoints({"time_stops": [5, 7.5]}) == [("time1", 5.0), ("time2", 7.5)]
    for bad in ([0, 10], [10, 10], [20, 10], [-1]):
        with pytest.raises(SystemExit):
            arm_exec.time_endpoints({"time_stops": bad})


# ---- PR #38 review: table extras cannot override guarded trainer flags ------------------
def _fake_payload(tmp_path):
    (tmp_path / "payload/adapters/I/final_adapter").mkdir(parents=True)
    (tmp_path / "payload/adapters/I/local_training.json").write_text(
        json.dumps({"control_identity": {"profile": {"dtype": "float32"}}}))


@pytest.mark.parametrize("extra", [
    ["--dtype", "bfloat16"],                                   # bf16 GRPO under fp32 guards
    ["--dynamic-sampling-max-draws", "2"],                     # C5 below 8
    ["--dynamic-sampling-max-draws", "8", "--dynamic-sampling-max-draws", "2"],  # last wins
    ["--learning-rate", "1"],                                  # breaks the control LR pairing
    ["--train", "@payload/data/manifests/dev800.jsonl"],       # selection holdout as TRAIN
    ["--validation", "@payload/data/manifests/dev800.jsonl"],
    ["--train=@payload/data/manifests/dev800.jsonl"],          # = form
    ["--dtype=bfloat16"],
    ["--dty", "bfloat16"],                                     # argparse abbreviation
    ["--learning-rat", "1"],
    ["--seed", "7"],
    ["--label-hierarchy", "@payload/data/manifests/dev800.jsonl"],
    ["--loss-type"],                                           # no value
])
def test_extras_that_override_or_escape_the_guards_refuse(tmp_path, extra):
    with pytest.raises(SystemExit):
        camp.build_spec(table(), grpo_row(extra_trainer_args=extra))
    _fake_payload(tmp_path)
    spec = dict(camp.build_spec(table(), grpo_row()), extra_trainer_args=extra)
    arm = arm_exec.Arm(tmp_path, spec, tmp_path / "r")
    with pytest.raises(SystemExit):
        arm.trainer_args(tmp_path / "m.jsonl", 200, 50, None)


def test_allowlisted_extras_reach_the_trainer_once(tmp_path):
    _fake_payload(tmp_path)
    extra = ["--dynamic-sampling-max-draws", "8", "--loss-type", "dr_grpo",
             "--skip-zero-advantage-steps", "--prompts-per-update", "4"]
    spec = camp.build_spec(table(), grpo_row(extra_trainer_args=extra))
    argv = arm_exec.Arm(tmp_path, spec, tmp_path / "r").trainer_args(
        tmp_path / "m.jsonl", 200, 50, None)
    assert argv[-7:] == extra and argv.count("--dtype") == 1
    with pytest.raises(SystemExit, match="repeats"):
        arm_exec.refuse_repeated_flags(["--dtype", "float32", "--x", "--dtype", "bfloat16"])


def _pair_dirs(tmp_path, rl_stages, ctl_stages, rl_rows, ctl_steps):
    cls = {"device_class": "cpu-fp32"}
    dirs = [
        _arm_dir(tmp_path, "eval-I", dict(cls, mode="eval"), "h1", [{"label": "init"}],
                 {"init": dict(GOLD)}),
        _arm_dir(tmp_path, "rl", dict(cls, mode="grpo", stops=[1, 2]), "h1",
                 [{"label": "step1", "steps": 1}, {"label": "step2", "steps": 2}],
                 {"step1": dict(GOLD), "step2": dict(GOLD)}, scanned=rl_rows),
        _arm_dir(tmp_path, "rl~ctl", dict(cls, mode="sft", track="control",
                                          matched_steps=[1, 2]), "h1",
                 [{"label": "time1", "steps": ctl_steps[0]},
                  {"label": "time2", "steps": ctl_steps[1]}],
                 {"time1": dict(GOLD), "time2": dict(GOLD)}),
    ]
    for name, stages in (("rl", rl_stages), ("rl~ctl", ctl_stages)):
        (tmp_path / name / "stages.json").write_text(json.dumps(
            [{"stage": st, "cotenancy": {"shared": sh}} for st, sh in stages]))
    return camp.build_report(camp.collect(dirs), GOLD, 50)


def test_cotenancy_is_compared_stage_by_stage(tmp_path):
    # both arms were shared SOMEWHERE (the old any-stage flags agree), but checkpoint 1's
    # RL stage was alone while its control stage was shared
    out = _pair_dirs(tmp_path, [("train-1", False), ("train-2", True)],
                     [("train-time1", True), ("train-time2", True)],
                     {"step1": 8, "step2": 16}, (40, 80))
    assert out["refused"] == ["rl step1: co-tenancy differs from its control (rl False, ctl True)"]
    assert [("vs_control" in e) for e in out["checkpoints"]] == [False, True]


def test_control_must_have_seen_at_least_the_rows_its_rl_arm_read(tmp_path):
    out = _pair_dirs(tmp_path, [("train-1", False), ("train-2", False)],
                     [("train-time1", False), ("train-time2", False)],
                     {"step1": 8, "step2": 16}, (8, 15))
    assert out["refused"] == ["rl step2: control saw 15 rows < 16 the RL arm read"]


def test_report_refuses_an_rl_checkpoint_without_scanned_rows(tmp_path):
    out = _pair_dirs(tmp_path, [("train-1", False), ("train-2", False)],
                     [("train-time1", False), ("train-time2", False)],
                     {"step1": 8}, (40, 80))
    assert out["refused"] == ["rl step2: no scanned/ rows recorded"]


def test_cotenancy_counts_every_earlier_stage_on_both_sides(tmp_path):
    # only the control's FIRST stage was shared; at step2 the producing stages agree (both
    # unshared) but the control's time2 stop includes time1's shared seconds
    out = _pair_dirs(tmp_path, [("train-1", False), ("train-2", False)],
                     [("train-time1", True), ("train-time2", False)],
                     {"step1": 8, "step2": 16}, (40, 80))
    assert out["refused"] == [
        "rl step1: co-tenancy differs from its control (rl False, ctl True)",
        "rl step2: co-tenancy differs from its control (rl False, ctl True)"]


def test_a_missing_stage_record_refuses_instead_of_falling_back(tmp_path):
    out = _pair_dirs(tmp_path, [("train-1", False), ("train-2", False)],
                     [("train-time1", False)], {"step1": 8, "step2": 16}, (40, 80))
    assert out["refused"] == ["rl step2: no ctl stage record for time2"]


@pytest.mark.parametrize("extra", [
    ["--epsilon-high", "nan"], ["--family-credit", "inf"], ["--invalid-reward", "-inf"],
    ["--label-hierarchy", "@payload/code/../data/x.json"],
    ["--label-hierarchy", "@payload/code/scripts/../../x.json"],
    ["--label-hierarchy", "@payload/code//x.json"],
    ["--label-hierarchy", "@payload/code/./x.json"],
    ["--label-hierarchy", "@payload/data/x.json"],
])
def test_extra_values_must_be_finite_and_stay_inside_payload_code(extra):
    for resolved in (False, True):  # the raw table value and the resolved path both refuse
        with pytest.raises(SystemExit):
            arm_exec.checked_extras(extra, [], resolved=resolved)


def test_extra_values_that_are_fine_pass():
    ok = ["--epsilon-high", "0.28", "--label-hierarchy",
          "@payload/code/scripts/exploratory_sweep/label-hierarchy-train-v1.json"]
    assert arm_exec.checked_extras(ok, [], resolved=False) == ok


def test_stage_record_refuses_stale_files_foreign_stage_ids_and_non_fp32(tmp_path):
    import os
    rec = tmp_path / "local_training.json"
    good = {"stage_id": "s2", "control_identity": {"profile": {"dtype": "float32"}},
            "sampler_teacher_logprob_parity": {"within_tolerance": True, "tolerance": 0.001}}
    rec.write_text(json.dumps(good))
    os.utime(rec, (1000.0, 1000.0))
    assert arm_exec.check_parity(rec, 999.0, "s2")["within_tolerance"] is True
    with pytest.raises(SystemExit, match="stale"):   # written before this stage started
        arm_exec.check_parity(rec, 1001.0, "s2")
    with pytest.raises(SystemExit, match="stage_id"):  # the previous stage's record
        arm_exec.check_parity(rec, 999.0, "s3")
    # RFT: same record rules, no parity
    assert arm_exec.stage_record(rec, 999.0)["stage_id"] == "s2"
    rec.write_text(json.dumps(dict(good, control_identity={"profile": {"dtype": "float16"}})))
    with pytest.raises(SystemExit, match="float16"):
        arm_exec.stage_record(rec)


def test_a_synthetic_four_draw_dynamic_sampling_row_refuses_before_the_trainer(tmp_path):
    # C5 (>= 8 draws), on a row built here rather than a committed table
    _fake_payload(tmp_path)
    row = grpo_row(extra_trainer_args=["--dynamic-sampling-max-draws", "4"])
    with pytest.raises(SystemExit, match="integer >= 8"):
        camp.build_spec(table(), row)
    spec = dict(camp.build_spec(table(), grpo_row()), extra_trainer_args=[
        "--dynamic-sampling-max-draws", "4"])
    with pytest.raises(SystemExit, match="integer >= 8"):
        arm_exec.refuse_spec(spec)
    with pytest.raises(SystemExit, match="integer >= 8"):
        arm_exec.Arm(tmp_path, spec, tmp_path / "r").trainer_args(
            tmp_path / "m.jsonl", 200, 50, None)
    ok = dict(spec, extra_trainer_args=["--dynamic-sampling-max-draws", "8"])
    argv = arm_exec.Arm(tmp_path, ok, tmp_path / "r").trainer_args(
        tmp_path / "m.jsonl", 200, 50, None)
    assert argv[-2:] == ["--dynamic-sampling-max-draws", "8"]


# ---- stage ids chosen by arm_exec, through the real trainer_args / rft_args argv -------
FAKE_TRAINER = r'''
import hashlib, json, os, sys
argv = sys.argv[1:]
mode = open(os.environ["FAKE_MODE_FILE"]).read().strip()
def opt(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None
out, sid, rft = opt("--output"), opt("--stage-id"), opt("--mode") == "rft"
stop = opt("--rft-stop-after-steps") if rft else opt("--stop-after-steps")
stop = int(stop) if stop is not None else None
os.makedirs(out + "/final_adapter", exist_ok=True)
weights = b"adapter-" + str(stop).encode() + mode.encode()
open(out + "/final_adapter/adapter_model.safetensors", "wb").write(weights)
sha = hashlib.sha256(weights).hexdigest()
open(out + "/argv.json", "w").write(json.dumps(argv))
dtype = "bfloat16" if mode == "bf16" else "float32"
record = {"control_identity": {"profile": {"dtype": dtype}}}
if rft:
    record.update(status="completed", final_adapter_sha256=sha, rounds=[], optimizer_steps=0,
                  active_seconds=0.0, settings={"epochs_per_round": 1})
    if mode == "rft_rounds":   # the run ran out of rounds before this stage's stop
        record.update(stop_reason="rounds")
    else:
        record.update(stop_reason="stop_after_steps",
                      boundary={"stop_after_steps": 1 if mode == "rft_wrong_boundary" else stop})
else:
    recorded = None if mode == "ignore_id" else sid
    record.update(stage_id=recorded, sampler_teacher_logprob_parity={
        "within_tolerance": True, "tolerance": 1e-3})
    if mode != "stale":
        open(out + "/endpoint_receipt.json", "w").write(json.dumps({
            "status": "boundary_reached", "global_step": stop, "stage_id": recorded,
            "committed_prompt_ids": ["p"] * (stop or 0), "committed_active_seconds": 1.0,
            "final_adapter": {"path": "final_adapter", "weights_sha256": sha}}))
if mode != "stale":
    open(out + "/local_training.json", "w").write(json.dumps(record))
'''


def _stage_arm(tmp_path, mode, spec):
    root = tmp_path / "root"
    (root / "venv/bin").mkdir(parents=True)
    _fake_payload(root)
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(FAKE_TRAINER)
    py = root / "venv/bin/python"
    # arm_exec runs `python -s -P -m vision_rl.vlm_local_training ...`
    py.write_text(f"#!/bin/bash\nshift 4\nexec python3 {fake} \"$@\"\n")
    py.chmod(0o755)
    mode_file = tmp_path / "mode"
    mode_file.write_text(mode)
    arm = arm_exec.Arm(root, spec, tmp_path / "results")
    arm.env["FAKE_MODE_FILE"] = str(mode_file)
    arm.results.mkdir(parents=True)
    arm.out.mkdir(parents=True)
    return arm, mode_file


def test_run_stage_passes_its_own_stage_id_and_requires_it_back(tmp_path):
    spec = camp.build_spec(table(), grpo_row(id="arm1"))
    arm, mode = _stage_arm(tmp_path, "honest", spec)
    m = tmp_path / "m.jsonl"
    arm.run_stage("train-50", arm.trainer_args(m, 200, 50, None))
    sent = json.loads((arm.out / "argv.json").read_text())
    assert sent[sent.index("--stage-id") + 1] == "arm1-train-50"  # chosen by arm_exec
    assert json.loads((arm.out / "local_training.json").read_text())["stage_id"] == (
        "arm1-train-50")
    for bad, match in (("ignore_id", "stage_id"), ("bf16", "bfloat16")):
        mode.write_text(bad)
        with pytest.raises(SystemExit, match=match):
            arm.run_stage("train-100", arm.trainer_args(m, 200, 100, None))
    # a stage that exits 0 without rewriting its receipt and record: both are stale
    mode.write_text("stale")
    with pytest.raises(SystemExit, match="stale"):
        arm.run_stage("train-150", arm.trainer_args(m, 200, 150, None))
    with pytest.raises(SystemExit, match="arm_exec's"):
        arm.run_stage("x", arm.trainer_args(m, 200, 50, None) + ["--stage-id", "mine"])


def test_sft_stage_checks_the_id_but_not_float32(tmp_path):
    spec = camp.build_spec(table(), grpo_row(id="s1", mode="sft", device_class="gtx1070-fp16"))
    arm, mode = _stage_arm(tmp_path, "bf16", spec)
    m = tmp_path / "m.jsonl"
    arm.run_stage("train-50", arm.trainer_args(m, 200, 50, None))
    mode.write_text("ignore_id")
    with pytest.raises(SystemExit, match="stage_id"):
        arm.run_stage("train-100", arm.trainer_args(m, 200, 100, None))


def _rft_spec():
    return dict(camp.BASE, id="r1", mode="rft", lr=1e-5, device="cpu", dtype="float32",
                threads=1, stops=[150, 300], max_steps_total=1500,
                rft={"rounds": 400, "prompts_per_round": 32})


def test_rft_stage_requires_its_own_stop_boundary_and_float32(tmp_path):
    arm, mode = _stage_arm(tmp_path, "honest", _rft_spec())
    m = tmp_path / "m.jsonl"
    argv = arm.trainer_args(m, 1500, 150, None)  # the real rft_args path
    assert argv[argv.index("--rft-stop-after-steps") + 1] == "150"
    receipt = arm.run_stage("train-150", argv, 150)
    assert receipt["status"] == "boundary_reached"
    assert "--stage-id" not in json.loads((arm.out / "argv.json").read_text())  # RFT CLI
    for bad, match in (("rft_wrong_boundary", "boundary"), ("bf16", "bfloat16"),
                       ("stale", "stale")):
        mode.write_text(bad)
        with pytest.raises(SystemExit, match=match):
            arm.run_stage("train-300", arm.trainer_args(m, 1500, 300, None), 300)
    with pytest.raises(SystemExit, match="ran with"):  # the stop that ran must be the stop
        arm.run_stage("train-300", arm.trainer_args(m, 1500, 300, None), 450)


def test_rft_stage_that_runs_out_of_rounds_ends_the_run(tmp_path):
    arm, mode = _stage_arm(tmp_path, "rft_rounds", _rft_spec())
    receipt = arm.run_stage("train-150", arm.trainer_args(tmp_path / "m.jsonl", 1500, 150,
                                                          None), 150)
    assert receipt["status"] == "total_budget_exhausted" and receipt["stop_reason"] == "rounds"


def test_label_hierarchy_symlink_out_of_payload_code_refuses(tmp_path):
    root = tmp_path / "root"
    (root / "payload/code/assets").mkdir(parents=True)
    (root / "payload/data").mkdir(parents=True)
    (root / "payload/data/x.json").write_text("{}")
    (root / "payload/code/assets/real.json").write_text("{}")
    (root / "payload/code/assets/link.json").symlink_to(root / "payload/data/x.json")
    arm = arm_exec.Arm(root, {"id": "a", "mode": "grpo"}, tmp_path / "r")
    assert arm.payload_arg("@payload/code/assets/real.json").endswith("real.json")
    with pytest.raises(SystemExit, match="outside payload/code"):
        arm.payload_arg("@payload/code/assets/link.json")


# ---- C16 needs every stage of the trajectory, on both sides -----------------------------
def test_rl_side_counts_its_earlier_stages_too(tmp_path):
    # RL train-1 shared, train-2 not; control shared throughout. Cumulative: both True at
    # step2, so it is quotable; an own-stage-only RL lookup would refuse it.
    out = _pair_dirs(tmp_path, [("train-1", True), ("train-2", False)],
                     [("train-time1", True), ("train-time2", True)],
                     {"step1": 8, "step2": 16}, (40, 80))
    assert out["refused"] == [] and all("vs_control" in e for e in out["checkpoints"])


def test_a_continuation_with_an_unfetched_earlier_rung_refuses(tmp_path):
    cls = {"device_class": "cpu-fp32"}
    dirs = [
        _arm_dir(tmp_path, "eval-I", dict(cls, mode="eval"), "h1", [{"label": "init"}],
                 {"init": dict(GOLD)}),
        # only the rung-2 dir arrived: its stages hold train-4, not rung 1's train-1/train-2
        _arm_dir(tmp_path, "rl", dict(cls, mode="grpo", resume_stop=2, rung=2,
                                      trajectory_stops=[1, 2, 4]), "h1",
                 [{"label": "step4", "steps": 4}], {"step4": dict(GOLD)}, scanned={"step4": 32}),
        _arm_dir(tmp_path, "rl~ctl@r2", dict(cls, mode="sft", track="control", rung=2,
                                             matched_steps=[1, 2, 4]), "h1",
                 [{"label": f"time{i}", "steps": 100 * i} for i in (1, 2, 3)],
                 {f"time{i}": dict(GOLD) for i in (1, 2, 3)}),
    ]
    out = camp.build_report(camp.collect(dirs), GOLD, 50, e0_passed={"cpu-fp32"})
    assert out["refused"] == ["rl step4: no rl stage record for step1"]  # names the gap
    assert camp.required_stages({"trajectory_stops": [1, 2, 4]}, "step4") == [
        "step1", "step2", "step4"]
    assert camp.required_stages({"resume_stop": 2, "stops": [4]}, "step4") is None
    assert camp.required_stages({}, "step4") is None  # unknown trajectory: refuse
