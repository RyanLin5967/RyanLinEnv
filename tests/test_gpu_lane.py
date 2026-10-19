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
