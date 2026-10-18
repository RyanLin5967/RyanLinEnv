"""CPU-only checks of the Harbor dogfood CI checker: fixtures only, no Docker, no model.

The passing fixture is written from the plan's documented receipt schema with literal values
(the local PASS numbers), never by calling the checker. Each planted defect must flip exactly
the criterion it targets and make the CLI exit non-zero.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/harbor_dogfood_ci.py"
SPEC = importlib.util.spec_from_file_location("harbor_dogfood_ci", SCRIPT)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)

GIB = 1024**3
INITIAL = "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"
SUBMITTED = "ee0e7d0f" + "0" * 56
MANIFEST = "2c20d30e3b0c27b2da711fa1b977d7dbb47049f22bef9d5b2af2d47e75e6bfcc"
REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
PROBE = "aircraft-harbor-dogfood-local-sft-probe-v4"
MAIN = "aircraft-harbor-dogfood-local-v4"


@pytest.fixture(scope="module")
def plan_dir(tmp_path_factory):
    return ci.extract_plan(tmp_path_factory.mktemp("plan") / "plan")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        path.write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        path.write_text(json.dumps(value))


def guard():
    return {
        "status": "completed",
        "exit_code": 0,
        "stop_reason": None,
        "peak_tree_rss_bytes": 12_235_571_200,
        "cgroup_memory": {"memory.max": 13 * GIB, "memory.swap.max": 0, "memory.swap.current": 0},
    }


def resources():
    memory = {"memory.max": 13 * GIB, "memory.swap.max": 0, "memory.swap.current": 0}
    return [{"tree_rss_bytes": 11_000_000_000, "cgroup_memory": memory} for _ in range(3)]


def loader(policy):
    return dict(
        policy,
        status="completed",
        model_directory="/opt/pinned-model",
        aliases_restored=True,
        environment_restored=True,
        runtime_sources=policy["source_sha256"],
        opens=[
            {"file": "model-00001-of-00002.safetensors", "device": "cpu", "backend": "pread"},
            {"file": "model-00002-of-00002.safetensors", "device": "cpu", "backend": "pread"},
        ],
    )


def passing_tree(root, plan_dir):
    policy = json.loads((plan_dir / "plan.json").read_text())["loader"]
    put(
        root / "prepared/bundle/task/environment/contract.json",
        {"model_revision": REVISION, "model_manifest_sha256": MANIFEST},
    )
    probe = root / "output/jobs" / PROBE / "sft-resource-task__a/artifacts/logs/artifacts"
    put(probe / "runtime/sft-resource-probe/guard.json", guard())
    put(probe / "runtime/sft-resource-probe/loader.json", loader(policy))
    put(probe / "runtime/sft-resource-probe/resources.jsonl", resources())
    put(probe / "sft-resource-probe.json", {"receipt": {"global_step": 1}})
    trial = root / "output/jobs" / MAIN / "task__b"
    artifacts = trial / "artifacts/logs/artifacts"
    for stage in ("rl-exposure", "sft-exposure", "sft-time"):
        put(artifacts / "runtime" / stage / "guard.json", guard())
        put(artifacts / "runtime" / stage / "loader.json", loader(policy))
        put(artifacts / "runtime" / stage / "resources.jsonl", resources())
    put(
        artifacts / "training/training-evidence.json",
        {
            "status": "completed",
            "model_source": {"manifest_sha256": MANIFEST},
            "arms": {
                "rl-exposure": {
                    "receipt": {"global_step": 1, "final_adapter": {"weights_sha256": SUBMITTED}},
                    "optimizer_audit": {"parameter_delta_l2": 0.01042},
                },
                "sft-exposure": {"receipt": {"global_step": 1}},
                "sft-time": {
                    "receipt": {"committed_active_seconds": 545.38},
                    "target_seconds": 530.31,
                },
            },
        },
    )
    put(
        artifacts / "training/rl/optimizer_audit.jsonl",
        [
            {
                "step": 1,
                "optimizer_steps_observed": 1,
                "nonzero_gradient_steps": 1,
                "first_nonzero_update": {"changed_tensors_since_invocation_start": 112},
            }
        ],
    )
    put(
        trial / "verifier/report.json",
        {
            "scope": "integration_only",
            "submitted_code_executed": False,
            "adapter_sha256": SUBMITTED,
            "reward": 1.0,
        },
    )
    put(trial / "verifier/loader.json", loader(policy))
    put(trial / "verifier/runtime/guard.json", guard())
    put(trial / "verifier/runtime/resources.jsonl", resources())
    put(
        trial / "result.json",
        {"trial_name": "task__b", "verifier_environment_mode": "separate", "exception_info": None},
    )
    return root


def edit(path, change):
    if path.suffix == ".jsonl":
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        change(rows)
        put(path, rows)
    else:
        value = json.loads(path.read_text())
        change(value)
        put(path, value)


def run_checker(root, plan_dir):
    result = root / "ci-criteria.json"
    code = ci.main(
        ["criteria", "--root", str(root), "--plan-dir", str(plan_dir), "--result", str(result)]
    )
    return code, json.loads(result.read_text())


def failed(result):
    return {name for name, row in result["criteria"].items() if not row["pass"]}


ARTIFACTS = f"output/jobs/{MAIN}/task__b/artifacts/logs/artifacts"
PLANTS = {
    "C1_mmap_backend": (
        f"{ARTIFACTS}/runtime/rl-exposure/loader.json",
        lambda v: v["opens"][0].update(backend="mmap"),
        "C1_pretrained_model_loaded",
    ),
    "C1_wrong_manifest": (
        f"{ARTIFACTS}/training/training-evidence.json",
        lambda v: v["model_source"].update(manifest_sha256="0" * 64),
        "C1_pretrained_model_loaded",
    ),
    "C2_uniform_reward_zero_gradient": (
        f"{ARTIFACTS}/training/rl/optimizer_audit.jsonl",
        lambda rows: rows[0].update(nonzero_gradient_steps=0),
        "C2_real_optimizer_update",
    ),
    "C2_adapter_unchanged": (
        f"{ARTIFACTS}/training/training-evidence.json",
        lambda v: v["arms"]["rl-exposure"]["receipt"]["final_adapter"].update(
            weights_sha256=INITIAL
        ),
        "C2_real_optimizer_update",
    ),
    "C3_shared_verifier_environment": (
        f"output/jobs/{MAIN}/task__b/result.json",
        lambda v: v.update(verifier_environment_mode="shared"),
        "C3_separate_verifier_scored_submission",
    ),
    "C3_submitted_code_executed": (
        f"output/jobs/{MAIN}/task__b/verifier/report.json",
        lambda v: v.update(submitted_code_executed=True),
        "C3_separate_verifier_scored_submission",
    ),
    "C4_container_swapped": (
        f"output/jobs/{PROBE}/sft-resource-task__a/artifacts/logs/artifacts/"
        "runtime/sft-resource-probe/resources.jsonl",
        lambda rows: rows[1]["cgroup_memory"].update({"memory.swap.current": 4096}),
        "C4_resource_guards_held",
    ),
    "C4_rss_row_over_12_gib": (
        f"{ARTIFACTS}/runtime/sft-time/resources.jsonl",
        lambda rows: rows[2].update(tree_rss_bytes=12 * GIB + 1),
        "C4_resource_guards_held",
    ),
}


def test_plan_archive_matches_pins_and_refuses_tampering(plan_dir, tmp_path):
    assert ci.sha(plan_dir / "plan.json") == ci.PLAN_SHA
    sums = tmp_path / "sums"
    lines = ci.PLAN_SUMS.read_text().splitlines()
    sums.write_text("\n".join(["0" * 64 + lines[0][64:]] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="differs from its pin"):
        ci.extract_plan(tmp_path / "bad", sums_path=sums)


def test_passing_fixture_passes(plan_dir, tmp_path):
    code, result = run_checker(passing_tree(tmp_path / "run", plan_dir), plan_dir)
    assert failed(result) == set()
    assert result["driver_error"] is None
    assert (code, result["ci_checker"]["pass"]) == (0, True)


@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_planted_failure_flips_its_criterion(plan_dir, tmp_path, plant):
    root = passing_tree(tmp_path / "run", plan_dir)
    relative, change, criterion = PLANTS[plant]
    edit(root / relative, change)
    code, result = run_checker(root, plan_dir)
    assert code == 1
    assert criterion in failed(result)
    assert result["pass"] is False and result["ci_checker"]["pass"] is False


def test_empty_tree_fails_every_criterion(plan_dir, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    code, result = run_checker(root, plan_dir)
    assert code == 1
    assert failed(result) == set(ci.CRITERIA)


def test_missing_probe_job_fails(plan_dir, tmp_path):
    root = passing_tree(tmp_path / "run", plan_dir)
    for path in sorted((root / "output/jobs" / PROBE).rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    code, result = run_checker(root, plan_dir)
    assert code == 1
    assert "C4_resource_guards_held" in failed(result)
    assert result["driver_error"]


def test_sft_time_short_of_rl_boundary_fails_although_c1_c4_pass(plan_dir, tmp_path):
    root = passing_tree(tmp_path / "run", plan_dir)
    edit(
        root / f"{ARTIFACTS}/training/training-evidence.json",
        lambda v: v["arms"]["sft-time"]["receipt"].update(committed_active_seconds=530.0),
    )
    code, result = run_checker(root, plan_dir)
    assert failed(result) == set()
    assert "RL active-time boundary" in result["driver_error"]
    assert code == 1


def test_cli_exit_status_is_nonzero_on_planted_failure(plan_dir, tmp_path):
    root = passing_tree(tmp_path / "run", plan_dir)
    edit(root / f"output/jobs/{MAIN}/task__b/verifier/report.json", lambda v: v.update(error="x"))
    command = [sys.executable, "-I", str(SCRIPT), "criteria", "--root", str(root)]
    command += ["--plan-dir", str(plan_dir)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "C3_separate_verifier_scored_submission: FAIL" in completed.stdout


def test_ci_gate_refuses_off_runner(plan_dir):
    plan = json.loads((plan_dir / "plan.json").read_text())
    runner = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "RyanLin5967/RyanLinEnv",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    ci.ci_launch_gate(plan, environ=runner, platform="linux", cwd=ci.ROOT)
    with pytest.raises(ValueError, match="recreated plan root"):
        ci.ci_launch_gate(plan, environ=runner, platform="linux", cwd=REPO)
    for environ, platform in (
        (runner, "darwin"),
        (dict(runner, GITHUB_RUN_ATTEMPT="2"), "linux"),
        (dict(runner, GITHUB_EVENT_NAME="push"), "linux"),
        ({}, "linux"),
    ):
        with pytest.raises(ValueError):
            ci.ci_launch_gate(plan, environ=environ, platform=platform, cwd=ci.ROOT)
    with pytest.raises(ValueError, match="unreviewed"):
        ci.ci_launch_gate(dict(plan, local_launch_reviewed=False), environ=runner, platform="linux")


def test_collect_uploads_text_receipts_only(plan_dir, tmp_path):
    root = passing_tree(tmp_path / "run", plan_dir)
    artifacts = root / ARTIFACTS / "training"
    for name in (
        "rl/final_adapter/adapter_model.safetensors",
        "rl/final_adapter/tokenizer.json",
        "rl/checkpoint-1/trainer_state.json",
        "adapter_model.safetensors",
        "rl/adapter-source.py",
        "image-0.jpg",
    ):
        (artifacts / name).parent.mkdir(parents=True, exist_ok=True)
        (artifacts / name).write_bytes(b"x")
    destination = tmp_path / "evidence"
    assert ci.main(["collect", "--root", str(root), "--dest", str(destination)]) == 0
    uploaded = {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()}
    assert not [p for p in uploaded if p.endswith((".safetensors", ".jpg", ".py"))]
    assert not [p for p in uploaded if "final_adapter" in p or "checkpoint-" in p]
    assert f"{ARTIFACTS}/training/training-evidence.json" in uploaded
    manifest = json.loads((destination / "evidence-manifest.json").read_text())
    assert len(manifest["skipped"]) == 6


def test_workflow_is_manual_free_runner_and_bounded():
    text = (REPO / ".github/workflows/aircraft-harbor-dogfood.yml").read_text()
    assert "workflow_dispatch" in text and "\n  push" not in text and "pull_request" not in text
    assert "runs-on: ubuntu-24.04" in text
    minutes = [int(line.split(":")[1]) for line in text.splitlines() if "timeout-minutes" in line]
    assert minutes and max(minutes) <= 360
    assert "scripts/harbor_dogfood_ci.py criteria" in text
    assert "path: evidence/" in text
