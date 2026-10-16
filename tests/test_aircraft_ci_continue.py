"""Light, no-model tests for exact CPU continuation provenance and bounded handoffs."""

import copy
import importlib.util
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "aircraft_continue", Path(__file__).parents[1] / "scripts/aircraft_ci_continue.py"
)
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)
ci = continuation.ci


@pytest.fixture
def plan(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    return {
        "schema_version": 1,
        "repository": ci.REPOSITORY,
        "parent_run_id": 100,
        "parent_commit": "a" * 40,
        "first4_files": {name: "c" * 64 for name in continuation.FIRST4_FILES},
        "artifacts": {
            name: {"id": index, "sha256": "d" * 64, "bytes": 100}
            for index, name in enumerate(continuation.ARMS, 1)
        },
    }


@pytest.mark.parametrize("mutation", ["commit", "one_arm", "same_artifact", "oversize", "unknown"])
def test_exact_prospective_envelope(plan, mutation):
    assert continuation.envelope(plan) == plan
    if mutation == "commit":
        plan["parent_commit"] = "main"
    elif mutation == "one_arm":
        del plan["artifacts"]["grpo"]
    elif mutation == "same_artifact":
        plan["artifacts"]["grpo"]["id"] = 2
    elif mutation == "oversize":
        plan["artifacts"]["grpo"]["bytes"] = 3 * ci.GIB
    else:
        plan["restart"] = True
    with pytest.raises(ValueError):
        continuation.envelope(plan)


def first4(root, arm, plan):
    for prefix in (f"work/{arm}", f"endpoints/{arm}-first4"):
        ci.write(
            root / prefix / "endpoint_receipt.json",
            {
                "global_step": 4,
                "original_budgets": {"max_steps": 64, "max_active_seconds": 7200},
                "checkpoint": {"path": "checkpoint-4"},
            },
        )
        for name in (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "active_time.json",
            "exposure.json",
        ):
            # Plain fixture bytes, never unpickled or called a real model checkpoint.
            target = root / prefix / "checkpoint-4" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("no-model fixture")
    ci.write(
        root / f"phases/{arm}-first4/guard.json", {"status": "completed", "elapsed_seconds": 12}
    )
    ci.write(
        root / f"phases/{arm}-first4/result.json",
        {"committed_steps": 4, "hardware": {"passed": True, "gates": {"fixture": True}}},
    )
    state = {
        "status": "completed",
        "group": "first4",
        "arm": arm,
        "commit": plan["parent_commit"],
        "run_id": str(plan["parent_run_id"]),
        "descriptor_sha256": plan["first4_files"][str(continuation.DESCRIPTOR)],
        "original_plan_sha256": ci.PLAN_SHA,
        "file_sha256": continuation.inventory(root),
    }
    ci.write(root / "ci-state.json", state)
    return root


def warm_output(tmp_path, plan):
    sources = {arm: first4(tmp_path / arm, arm, plan) for arm in continuation.ARMS}
    output = tmp_path / "warm"
    continuation.merge_first4(output, sources, plan)
    for suffix in continuation.ASSAYS:
        ci.write(
            output / f"phases/warm-{suffix}/guard.json",
            {"status": "completed", "elapsed_seconds": 100},
        )
    allocation = continuation.remaining_allocation(output)
    continuation.write_handoff(output, plan, "warm", allocation=allocation)
    return output


@pytest.mark.parametrize(
    "case",
    [
        "failed",
        "wrong_arm",
        "different_commit",
        "modified_optimizer",
        "missing_rng",
        "false_gate",
        "reset_budget",
    ],
)
def test_bad_parent_artifact_rejected_without_pickle_or_model(tmp_path, plan, case):
    root = first4(tmp_path, "grpo", plan)
    state = ci.read(root / "ci-state.json")
    if case in {"failed", "wrong_arm", "different_commit"}:
        state[{"failed": "status", "wrong_arm": "arm", "different_commit": "commit"}[case]] = (
            "changed"
        )
        ci.write(root / "ci-state.json", state)
    elif case == "modified_optimizer":
        (root / "work/grpo/checkpoint-4/optimizer.pt").write_text("changed")
    elif case == "missing_rng":
        (root / "work/grpo/checkpoint-4/rng_state.pth").unlink()
    else:
        path = root / (
            "phases/grpo-first4/result.json"
            if case == "false_gate"
            else "work/grpo/endpoint_receipt.json"
        )
        value = ci.read(path)
        if case == "false_gate":
            value["hardware"]["passed"] = False
        else:
            value["original_budgets"]["max_active_seconds"] = 8000
        ci.write(path, value)
        state["file_sha256"] = continuation.inventory(root, exclude=("ci-state.json",))
        ci.write(root / "ci-state.json", state)
    with pytest.raises((ValueError, OSError)):
        continuation.verify_first4(root, "grpo", plan)


@pytest.mark.parametrize(
    "name,mode",
    [
        ("../escape", 0),
        ("/absolute", 0),
        ("a/../b", 0),
        ("link", stat.S_IFLNK | 0o777),
        ("pipe", stat.S_IFIFO | 0o644),
    ],
)
def test_archive_rejects_paths_and_links_before_extraction(tmp_path, name, mode):
    path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo(name)
        info.external_attr = mode << 16
        archive.writestr(info, "fixture")
    with pytest.raises(ValueError):
        continuation.extract_zip(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_oversize_zip_rejected_before_extraction(tmp_path, monkeypatch):
    path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", "1234")
    monkeypatch.setitem(ci.LIMITS, "output_bytes", 3)
    with pytest.raises(ValueError, match="budget"):
        continuation.extract_zip(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_remaining_budget_is_shared_once_and_never_reset(tmp_path, plan):
    output = warm_output(tmp_path, plan)
    record = continuation.verify_handoff(output, plan, "warm")
    allocation = record["allocation"]
    assert allocation["prior_shared_seconds"] == 324
    assert (
        allocation["prior_shared_seconds"] + sum(allocation["reserved_seconds"].values()) <= 43200
    )
    ci.write(
        output / "phases/warm-greedy/guard.json", {"status": "completed", "elapsed_seconds": 0}
    )
    with pytest.raises(ValueError):
        continuation.verify_handoff(output, plan, "warm")


def test_no_continuation_without_all_three_fresh_warm_assays(tmp_path, plan):
    output = warm_output(tmp_path, plan)
    (output / "phases/warm-sample314/guard.json").unlink()
    with pytest.raises(ValueError):
        continuation.remaining_allocation(output)


def test_exact_stage_reuse_without_restarting_first_four(tmp_path, plan, monkeypatch):
    output = warm_output(tmp_path, plan)
    monkeypatch.setattr(continuation.sys, "platform", "linux")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("EXPERIMENT_UID", str(__import__("os").geteuid()))
    monkeypatch.setenv("EXPERIMENT_GID", str(__import__("os").getegid()))
    monkeypatch.setattr(ci, "ORIGINAL", tmp_path)
    monkeypatch.setattr(ci, "OUTPUT_REL", Path("warm"))
    runner = SimpleNamespace(_require_hardware_pair=lambda: None, _require_warm_assay=lambda: None)
    monkeypatch.setattr(ci, "check_inputs", lambda: (runner, {}))
    calls = []
    monkeypatch.setattr(
        ci, "supervise_phase", lambda phase, limits: calls.append((phase, copy.deepcopy(limits)))
    )
    plan_path = tmp_path / "plan.json"
    ci.write(plan_path, plan)
    continuation.container_stage("grpo", plan_path)
    assert [phase for phase, _ in calls] == [
        "grpo-full",
        "grpo-greedy",
        "grpo-sample42",
        "grpo-sample314",
    ]
    original = ci.read(output / "handoff.json")["allocation"]
    assert all(
        limits["all_phase_wall_seconds"]
        == original["prior_shared_seconds"] + original["reserved_seconds"]["grpo"]
        for _, limits in calls
    )
    assert not any("first4" in phase for phase, _ in calls)


def test_docker_inherits_guards_and_contains_no_host_credentials(tmp_path, plan):
    command = continuation.docker_command(
        tmp_path / "inputs", tmp_path / "output", "sha256:fixed", "grpo", tmp_path / "plan.json"
    )
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cpus") + 1] == "2"
    assert command[command.index("--memory") + 1] == str(13 * ci.GIB)
    assert "--init" in command and "--read-only" in command
    assert not any("TOKEN" in part for part in command)
    assert command[-6:] == [
        "/ci/continue.py",
        "container-stage",
        "--group",
        "grpo",
        "--plan",
        "/ci/continuation-plan.json",
    ]


def test_single_use_parent_rejects_prior_failed_continuation(plan, monkeypatch):
    monkeypatch.setattr(
        ci,
        "api_json",
        lambda path: {
            "workflow_runs": [
                {
                    "id": 199,
                    "display_title": "aircraft-cpu-continue parent-100",
                    "conclusion": "failure",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="consumed"):
        continuation.require_single_use(plan)


def test_foreign_or_failed_parent_rejected_before_any_artifact_download(plan, monkeypatch):
    calls = []

    def api(path):
        calls.append(path)
        if not path:
            return {"private": False, "full_name": ci.REPOSITORY}
        return {
            "id": 100,
            "head_sha": "a" * 40,
            "path": continuation.FIRST4_WORKFLOW,
            "event": "workflow_dispatch",
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "failure",
            "head_repository": {"full_name": ci.REPOSITORY},
        }

    monkeypatch.setattr(ci, "api_json", api)
    with pytest.raises(ValueError, match="successful"):
        continuation.verify_parent(plan)
    assert calls == ["", "actions/runs/100"]


def test_workflow_is_finite_public_manual_and_warm_precedes_parallel_arms():
    text = (Path(__file__).parents[1] / continuation.CONTINUATION_WORKFLOW).read_text()
    assert "workflow_dispatch:" in text and "pull_request:" not in text and "schedule:" not in text
    assert text.count("timeout-minutes: 360") == 4
    assert text.count("needs: warm") == 2
    assert "needs: [grpo, dr_grpo]" in text
    assert text.count("retention-days: 1") == 4
    assert "github.event.repository.private == false" in text
    assert "self-hosted" not in text and "runs/aircraft-ci-continuation/output/" in text


def completed_pair(tmp_path, plan):
    warm = warm_output(tmp_path / "fixture", plan)
    allocation = ci.read(warm / "handoff.json")["allocation"]
    sources = {}
    for arm in continuation.ARMS:
        import shutil

        output = tmp_path / (arm + "-completed")
        shutil.copytree(warm, output)
        for relative in continuation._owned_subtrees(arm):
            target = output / relative
            target.mkdir(parents=True, exist_ok=True)
            (target / "fixture.txt").write_text(arm)
        # The continued work has rotated its old checkpoint; it must not reappear on merge.
        shutil.rmtree(output / f"work/{arm}/checkpoint-4")
        for phase in (f"{arm}-full", *(f"{arm}-{a}" for a in continuation.ASSAYS)):
            ci.write(
                output / "phases" / phase / "guard.json",
                {"status": "completed", "elapsed_seconds": 10},
            )
        continuation.write_handoff(output, plan, arm, allocation=allocation)
        sources[arm] = output
    return sources


def test_parallel_merge_counts_shared_phases_once_and_preserves_exact_work(tmp_path, plan):
    sources = completed_pair(tmp_path, plan)
    output = tmp_path / "merged"
    continuation.merge_completed(output, sources, plan)
    assert len(ci.completed_phases(output)) == 13
    assert sum(v["elapsed_seconds"] for v in ci.completed_phases(output).values()) == 404
    for arm in continuation.ARMS:
        assert continuation.inventory(output / "work" / arm) == continuation.inventory(
            sources[arm] / "work" / arm
        )
        assert not (output / f"work/{arm}/checkpoint-4").exists()


def test_different_shared_warm_result_cannot_be_merged(tmp_path, plan):
    sources = completed_pair(tmp_path, plan)
    changed = sources["dr_grpo"]
    ci.write(
        changed / "phases/warm-greedy/guard.json", {"status": "completed", "elapsed_seconds": 99}
    )
    allocation = ci.read(changed / "handoff.json")["allocation"]
    continuation.write_handoff(changed, plan, "dr_grpo", allocation=allocation)
    with pytest.raises(ValueError, match="Shared"):
        continuation.merge_completed(tmp_path / "merged", sources, plan)


def test_parent_success_requires_current_and_historical_file_bytes(plan, monkeypatch, tmp_path):
    import base64

    monkeypatch.chdir(tmp_path)
    for name in continuation.FIRST4_FILES:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed fixture source")
        plan["first4_files"][name] = ci.digest(path)

    def api(path):
        if not path:
            return {"private": False, "full_name": ci.REPOSITORY}
        if path.startswith("actions/runs/"):
            return {
                "id": 100,
                "head_sha": "a" * 40,
                "path": continuation.FIRST4_WORKFLOW,
                "event": "workflow_dispatch",
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
                "head_repository": {"full_name": ci.REPOSITORY},
            }
        return {
            "encoding": "base64",
            "content": base64.b64encode(b"reviewed fixture source").decode(),
        }

    monkeypatch.setattr(ci, "api_json", api)
    assert continuation.verify_parent(plan)["id"] == 100
    Path(continuation.FIRST4_FILES[0]).write_text("changed later helper")
    with pytest.raises(ValueError, match="changed a frozen"):
        continuation.verify_parent(plan)
