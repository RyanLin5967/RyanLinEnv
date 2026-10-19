"""GPU lane (exploratory sweep v2): save_steps carried through in-place continuations.

save_steps is part of the trainer's resume identity, so every stage of one trajectory must
pass the value the trajectory STARTED with. Expected values are written by hand (150: the gcd
of the table's first-launch stops 150/300/600), never computed by the subject.
"""

from __future__ import annotations

import importlib.util
import json
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


lane = load("gpu_lane")
arm_exec = load("arm_exec")


def _save_steps(argv):
    return argv[argv.index("--save-steps") + 1]


def _fake_root(tmp_path):
    w = tmp_path / "payload" / "adapters" / "W"
    (w / "final_adapter").mkdir(parents=True)
    (w / "local_training.json").write_text(
        json.dumps({"control_identity": {"profile": {"dtype": "float32"}}}))
    return tmp_path


def _stage_args(root, spec, stop, resume):
    arm = arm_exec.Arm(root, spec, root / "r")
    return arm.trainer_args(root / "m.jsonl", int(spec["max_steps_total"]), stop, resume)


def test_150_started_arm_keeps_save_steps_150_at_300_600_and_1200(tmp_path):
    root = _fake_root(tmp_path)
    table = lane.grpo("rtx3070-bfloat16", 1e-5, 0.0, False)
    assert table["stops"] == [150, 300, 600] and table["max_steps_total"] == 1200
    seen = [_save_steps(_stage_args(root, table, 150, None))]
    for resume, extra, stop in ((150, [], 300), (300, [], 600), (600, [1200], 1200)):
        spec = lane.continue_spec(table, resume, extra)
        assert spec["resume_stop"] == resume and spec["work_id"] == table["id"]
        seen.append(_save_steps(_stage_args(root, spec, stop, root / f"checkpoint-{resume}")))
    assert seen == ["150", "150", "150", "150"]


def test_gcd_alone_would_have_drifted(tmp_path):
    # the failure the carried value prevents: resume 600 -> [1200] by gcd gives 600
    root = _fake_root(tmp_path)
    table = lane.grpo("rtx3070-bfloat16", 1e-5, 0.0, False)
    drifted = dict(table, stops=[1200], resume_stop=600, work_id=table["id"])
    assert _save_steps(_stage_args(root, drifted, 1200, root / "c")) == "600"
    carried = lane.continue_spec(table, 600, [1200])
    assert _save_steps(_stage_args(root, carried, 1200, root / "c")) == "150"


def test_bf16_cuda_arm_declares_precision_transfer_every_stage(tmp_path):
    root = _fake_root(tmp_path)
    table = lane.grpo("rtx3070-bfloat16", 1e-5, 0.0, False)
    for spec, stop, resume in ((table, 150, None),
                               (lane.continue_spec(table, 150, []), 300, root / "c")):
        argv = _stage_args(root, spec, stop, resume)
        i = argv.index("--initial-adapter-precision-transfer")
        assert argv[i + 1] == "float32-to-bfloat16"


@pytest.mark.parametrize("resume,extra", [(150, [1250]), (600, []), (600, [1300])])
def test_continue_refuses_bad_stops(resume, extra):
    table = lane.grpo("rtx3070-bfloat16", 1e-5, 0.0, False)
    with pytest.raises(SystemExit):
        lane.continue_spec(table, resume, extra)


def test_continue_refuses_a_spec_that_is_already_a_continuation():
    table = lane.grpo("rtx3070-bfloat16", 1e-5, 0.0, False)
    with pytest.raises(SystemExit):
        lane.continue_spec(lane.continue_spec(table, 150, []), 300, [])


def _rl_dir(tmp_path, times, draws):
    d = tmp_path / "rl"
    d.mkdir()
    eps = [{"label": f"step{s}", "steps": s, "active_seconds": t, "metrics": {}}
           for s, t in zip((150, 300, 600), times)]
    (d / "summary.json").write_text(json.dumps({"endpoints": eps}))
    (d / "dynamic_sampling.jsonl").write_text("".join(json.dumps(r) + "\n" for r in draws))
    return d


def test_control2_trains_on_every_scanned_id_in_scan_order_and_matches_each_checkpoint(tmp_path):
    rl = lane.corrected_grpo_i()[0]
    # journal written out of order on purpose; scan order is (global_step, draw)
    draws = [{"global_step": 2, "draw": 0, "id": "c", "kept": True},
             {"global_step": 1, "draw": 1, "id": "b", "kept": True},
             {"global_step": 1, "draw": 0, "id": "a", "kept": False}]
    d = _rl_dir(tmp_path, [100.0, 210.5, 400.0], draws)
    specs = lane.control2_specs(rl, d, [5e-6, 1e-5, 2e-5])
    assert [s["lr"] for s in specs] == [5e-6, 1e-5, 2e-5]
    for s in specs:
        assert s["train_ids"] == ["a", "b", "c"]  # the discarded draw "a" is included
        assert s["time_stops"] == [100.0, 210.5, 400.0]
        assert s["control_checkpoints"] == ["step150", "step300", "step600"]
        assert s["init"] == "I" and s["mode"] == "sft" and s["dtype"] == "bfloat16"
        assert arm_exec.time_endpoints(s) == [("time1", 100.0), ("time2", 210.5),
                                              ("time3", 400.0)]


def test_control2_lists_each_scanned_id_once_in_first_read_order(tmp_path):
    rl = lane.corrected_grpo_i()[0]
    draws = [{"global_step": 1, "draw": 0, "id": "a"}, {"global_step": 1, "draw": 1, "id": "b"},
             {"global_step": 2, "draw": 0, "id": "a"}, {"global_step": 2, "draw": 1, "id": "c"}]
    d = _rl_dir(tmp_path, [100.0, 200.0, 300.0], draws)
    assert lane.control2_specs(rl, d, [1e-5])[0]["train_ids"] == ["a", "b", "c"]


def test_control2_refuses_non_increasing_checkpoint_seconds(tmp_path):
    rl = lane.corrected_grpo_i()[0]
    d = _rl_dir(tmp_path, [100.0, 90.0, 400.0], [{"global_step": 1, "draw": 0, "id": "a"}])
    with pytest.raises(SystemExit):
        lane.control2_specs(rl, d, [1e-5])


@pytest.mark.parametrize("extras", [
    ["--dtype", "bfloat16"],                       # not allowlisted (arm_exec's own flag)
    ["--learning-rate", "1"],
    ["--train", "x.jsonl"], ["--train=x.jsonl"], ["--validation", "v.jsonl"],
    ["--dynamic-sampling-max-draws=8"],            # = form
    ["--dynamic-sampling-max"],                    # abbreviation
    ["--dynamic-sampling-max-draws", "4"],         # below C5's 8
    ["--dynamic-sampling-max-draws", "8", "--dynamic-sampling-max-draws", "16"],
    ["--loss-type"], ["--loss-type", "--x"],       # no value
    ["--loss-type", "bnpo"],
])
def test_extras_refuse_overrides_and_unvalidated_values(tmp_path, extras):
    root = _fake_root(tmp_path)
    spec = dict(lane.corrected_grpo_i()[0], extra_trainer_args=extras, device="cpu")
    with pytest.raises(SystemExit):
        _stage_args(root, spec, 150, None)


def test_extras_refuse_a_flag_arm_exec_already_passes():
    with pytest.raises(SystemExit):
        arm_exec.checked_extras(["--loss-type", "grpo"], ["--loss-type", "grpo"])


def test_lane_extras_pass_and_land_once(tmp_path):
    root = _fake_root(tmp_path)
    # every table arm (GRPO, RFT, multi-image): each declared extra flag lands exactly once
    arms = lane.corrected_grpo_i() + lane.amendment7()
    assert any(a["mode"] == "rft" for a in arms) and any(
        "--prompts-per-update" in a.get("extra_trainer_args", []) for a in arms)
    for arm in arms:
        argv = _stage_args(root, dict(arm, device="cpu"), 150, None)
        for flag in [x for x in arm.get("extra_trainer_args", []) if x.startswith("--")]:
            assert argv.count(flag) == 1, (arm["id"], flag)
        assert argv.count("--dtype") == 1
        assert argv.count("--learning-rate") == 1 and argv.count("--train") == 1


def _recorded(tmp_path, **over):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    man = tmp_path / "m.jsonl"
    man.write_text('{"id": "a"}\n')
    rec = {"settings": {"device": "cpu", "learning_rate": 2e-5,
                        "dynamic_sampling": {"max_draws": 8}},
           "control_identity": {"profile": {"dtype": "float32"},
                                "train": {"manifest_sha256": arm_exec.sha_file(man)}}}
    for k, v in over.items():
        if k == "dtype":
            rec["control_identity"]["profile"]["dtype"] = v
        elif k == "manifest_sha256":
            rec["control_identity"]["train"]["manifest_sha256"] = v
        elif k == "max_draws":
            rec["settings"]["dynamic_sampling"]["max_draws"] = v
        else:
            rec["settings"][k] = v
    (out / "local_training.json").write_text(json.dumps(rec))
    return out, man


@pytest.mark.parametrize("over", [{"dtype": "bfloat16"}, {"learning_rate": 1e-5},
                                  {"device": "cuda"}, {"manifest_sha256": "0" * 64},
                                  {"max_draws": 4}])
def test_check_record_refuses_a_run_other_than_the_spec(tmp_path, over):
    spec = dict(lane.corrected_grpo_i()[0], device="cpu")
    arm = arm_exec.Arm(tmp_path, spec, tmp_path / "r")
    arm.out, man = _recorded(tmp_path, **over)
    with pytest.raises(SystemExit):
        arm.check_record(["--train", str(man)])


def test_check_record_accepts_the_spec(tmp_path):
    spec = dict(lane.corrected_grpo_i()[0], device="cpu")
    arm = arm_exec.Arm(tmp_path, spec, tmp_path / "r")
    arm.out, man = _recorded(tmp_path)
    assert arm.check_record(["--train", str(man)])["dynamic_sampling_max_draws"] == 8


def test_promotion_specs_keep_the_recipe_and_change_only_device_or_seed():
    for arm in lane.exploratory_bf16() + lane.pool_arms():
        fp32, s314, s2718 = lane.promotion_specs(arm)
        assert fp32["dtype"] == "float32" and fp32["device_class"] == "rtx2080ti-float32"
        assert "label" not in fp32  # the fp32 replicate is not exploratory-bf16
        for k in ("lr", "temperature", "num_generations", "beta", "extra_trainer_args",
                  "train_manifest", "seed", "stops", "init"):
            assert fp32[k] == arm[k]
        for s, seed in ((s314, 314), (s2718, 2718)):
            assert s["seed"] == seed and s["dtype"] == "bfloat16"
            want = (f"s_train_scan_seed{seed}.jsonl" if arm["train_manifest"].startswith("s_train")
                    else arm["train_manifest"])
            assert s["train_manifest"] == want
    with pytest.raises(SystemExit):
        lane.promotion_specs(lane.corrected_grpo_i()[0])


def _start(tmp_path, data=b"weights"):
    d = tmp_path / "start" / "sftall-lr5e-6-s300" / "final_adapter"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(data)
    (d.parent / "local_training.json").write_text("{}")
    return arm_exec.sha_file(d / "adapter_model.safetensors")


def test_start_adapter_init_needs_its_exact_pinned_weights(tmp_path):
    sha = _start(tmp_path)
    good = {"start": "sftall-lr5e-6-s300", "weights_sha256": sha}
    arm = arm_exec.Arm(tmp_path, {"id": "x", "init": good, "mode": "grpo"}, tmp_path / "r")
    assert arm.init_adapter() == tmp_path / "start" / "sftall-lr5e-6-s300" / "final_adapter"
    for bad in ({"start": "sftall-lr5e-6-s300", "weights_sha256": "0" * 64},
                {"start": "sftall-lr5e-6-s300"},
                {"start": "missing", "weights_sha256": sha},
                {"start": "../x", "weights_sha256": sha}):
        with pytest.raises(SystemExit):
            arm_exec.Arm(tmp_path, {"id": "x", "init": bad, "mode": "grpo"},
                         tmp_path / "r").init_adapter()


@pytest.mark.parametrize("stops", [[0.0, 10.0], [10.0, 10.0], [20.0, 10.0], [-1.0]])
def test_time_stops_refuse_non_positive_or_non_increasing(stops):
    with pytest.raises(SystemExit):
        arm_exec.time_endpoints({"time_stops": stops})


def test_time_stops_accept_a_positive_increasing_list():
    assert arm_exec.time_endpoints({"time_stops": [5, 7.5]}) == [("time1", 5.0), ("time2", 7.5)]


def test_corrected_grpo_i_arms_follow_the_root_corrections():
    arms = lane.corrected_grpo_i()
    assert [(a["lr"], a["temperature"]) for a in arms] == [
        (2e-5, 1.3), (2e-5, 1.0), (5e-5, 1.3), (1e-5, 1.3), (3e-6, 1.3)]
    for a in arms:
        assert a["dtype"] == "float32" and a["device_class"] == "rtx2080ti-float32"
        assert a["beta"] == 0.0 and a["screen_threshold"] is None and a["init"] == "I"
        x = a["extra_trainer_args"]
        assert int(x[x.index("--dynamic-sampling-max-draws") + 1]) >= 8
        assert x[x.index("--loss-type") + 1] == "grpo"
        assert a["stops"] == [150, 300, 600]
