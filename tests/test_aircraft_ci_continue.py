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
ROOT = Path(__file__).parents[1]


@pytest.fixture
def plan(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setattr(continuation, "descriptor_path", lambda: ROOT / continuation.DESCRIPTOR)
    return {
        "schema_version": 1,
        "repository": ci.REPOSITORY,
        "parent_run_id": 100,
        "parent_commit": "a" * 40,
        "first4_files": {name: ci.digest(ROOT / name) for name in continuation.FIRST4_FILES},
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


def completed_loader_phase(root, phase, elapsed):
    """Synthetic execution records only: no Transformers import or model call."""
    identity = ci.loader_identity()
    contract = ci.read(continuation.descriptor_path())
    ci.write(
        root / "phases" / phase / "guard.json",
        {
            "phase": phase,
            "status": "completed",
            "exit_code": 0,
            "elapsed_seconds": elapsed,
            "loader": identity,
        },
    )
    ci.write(
        root / "phases" / phase / "loader.json",
        {
            **identity,
            "status": "completed",
            "runtime_versions": identity["versions"],
            "runtime_sources": identity["source_sha256"],
            "model_directory": continuation.loader_model_directory(identity, contract),
            "opens": [
                {"file": name, "device": "cpu", "backend": "pread"} for name in identity["shards"]
            ],
            "installed_files_modified": False,
            "global_safetensors_alias_modified": False,
            "aliases_restored": True,
            "environment_restored": True,
        },
    )


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
    completed_loader_phase(root, f"{arm}-first4", 12)
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
        "loader": ci.loader_identity(),
        "file_sha256": continuation.inventory(root),
    }
    ci.write(root / "ci-state.json", state)
    return root


def warm_output(tmp_path, plan):
    sources = {arm: first4(tmp_path / arm, arm, plan) for arm in continuation.ARMS}
    output = tmp_path / "warm"
    continuation.merge_first4(output, sources, plan)
    for suffix in continuation.ASSAYS:
        completed_loader_phase(output, f"warm-{suffix}", 100)
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
    model_directory = continuation.loader_model_directory(
        ci.loader_identity(), ci.read(continuation.descriptor_path())
    )
    monkeypatch.setattr(continuation, "loader_model_directory", lambda *_: model_directory)
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
    assert "HF_DEACTIVATE_ASYNC_LOAD=1" in command
    assert f"AIRCRAFT_LOADER_SHA256={ci.loader_identity()['sha256']}" in command
    assert any(part.endswith(":/ci/aircraft_cpu_loader.py:ro") for part in command)
    assert any(part.endswith(":/ci/experiment-plan.json:ro") for part in command)
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
            completed_loader_phase(output, phase, 10)
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
    path = changed / "phases/warm-greedy/guard.json"
    ci.write(path, ci.read(path) | {"elapsed_seconds": 99})
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
        path.write_bytes((ROOT / name).read_bytes())
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
            "content": base64.b64encode(
                (ROOT / path.removeprefix("contents/").split("?ref=")[0]).read_bytes()
            ).decode(),
        }

    monkeypatch.setattr(ci, "api_json", api)
    assert continuation.verify_parent(plan)["id"] == 100
    Path(continuation.FIRST4_FILES[0]).write_text("changed later helper")
    with pytest.raises(ValueError, match="changed a frozen"):
        continuation.verify_parent(plan)


@pytest.mark.parametrize("entrypoint", ["parent", "host"])
def test_altered_loader_only_rejected_before_api_or_restoration(
    tmp_path, plan, monkeypatch, entrypoint
):
    # All three previously pinned files remain unchanged. Only the independently
    # mounted loader changes, reproducing the old transitive-identity hole.
    source = tmp_path / "aircraft_cpu_loader.py"
    source.write_bytes((ROOT / continuation.LOADER_FILE).read_bytes() + b"\n# changed bootstrap\n")
    spec = importlib.util.spec_from_file_location("changed_loader_fixture", source)
    changed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(changed)
    monkeypatch.setattr(ci, "loader_module", lambda: changed)
    calls = []
    monkeypatch.setattr(ci, "api_json", lambda path: calls.append(path))
    monkeypatch.setattr(continuation, "artifact_download", lambda *a, **k: calls.append("download"))
    monkeypatch.setattr(continuation, "setup_runtime", lambda *a: calls.append("runtime"))
    monkeypatch.setattr(continuation.sys, "platform", "linux")
    monkeypatch.setenv("GITHUB_REPOSITORY", ci.REPOSITORY)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("PARENT_RUN_ID", "100")
    with pytest.raises(ValueError, match="loader identity"):
        if entrypoint == "parent":
            continuation.verify_parent(plan)
        else:
            continuation.host_run(tmp_path / "never-created", plan, "warm")
    assert calls == []
    assert not (tmp_path / "never-created").exists()


def test_legacy_envelope_without_loader_pin_has_no_fallback(plan):
    del plan["first4_files"][continuation.LOADER_FILE]
    with pytest.raises(ValueError, match="Pin both"):
        continuation.envelope(plan)


@pytest.mark.parametrize("mutation", ["missing", "policy", "source"])
def test_descriptor_loader_must_match_actual_and_direct_pin(tmp_path, plan, monkeypatch, mutation):
    contract = ci.read(continuation.descriptor_path())
    if mutation == "missing":
        del contract["loader"]
    elif mutation == "policy":
        contract["loader"]["backend"] = "mmap"
    else:
        plan["first4_files"][continuation.LOADER_FILE] = "0" * 64
    descriptor = tmp_path / "descriptor.json"
    ci.write(descriptor, contract)
    plan["first4_files"][str(continuation.DESCRIPTOR)] = ci.digest(descriptor)
    monkeypatch.setattr(continuation, "descriptor_path", lambda: descriptor)
    with pytest.raises(ValueError, match="loader identity"):
        continuation.checked_loader(plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "state_missing",
        "state_changed",
        "guard_missing",
        "guard_changed",
        "wrong_phase",
        "nonzero_exit",
        "receipt_missing",
        "receipt_source",
        "receipt_active",
        "receipt_error",
        "alias_not_restored",
        "environment_not_restored",
        "installed_modified",
        "global_modified",
        "wrong_runtime_source",
        "wrong_runtime_version",
        "wrong_base",
        "wrong_backend",
        "missing_shard",
        "extra_shard",
        "bad_open_type",
        "too_many_opens",
    ],
)
def test_parent_loader_execution_receipts_are_required_even_with_valid_inventory(
    tmp_path, plan, mutation
):
    root = first4(tmp_path, "grpo", plan)
    state_path = root / "ci-state.json"
    state = ci.read(state_path)
    if mutation.startswith("state_"):
        if mutation == "state_missing":
            del state["loader"]
        else:
            state["loader"]["sha256"] = "0" * 64
    elif mutation in {"guard_missing", "guard_changed", "wrong_phase", "nonzero_exit"}:
        path = root / "phases/grpo-first4/guard.json"
        guard = ci.read(path)
        if mutation == "guard_missing":
            del guard["loader"]
        elif mutation == "guard_changed":
            guard["loader"]["sha256"] = "0" * 64
        elif mutation == "wrong_phase":
            guard["phase"] = "dr_grpo-first4"
        else:
            guard["exit_code"] = 1
        ci.write(path, guard)
    else:
        path = root / "phases/grpo-first4/loader.json"
        record = ci.read(path)
        if mutation == "receipt_missing":
            path.unlink()
        else:
            mutations = {
                "receipt_source": {"sha256": "0" * 64},
                "receipt_active": {"status": "active"},
                "receipt_error": {"error_type": "RuntimeError"},
                "alias_not_restored": {"aliases_restored": False},
                "environment_not_restored": {"environment_restored": False},
                "installed_modified": {"installed_files_modified": True},
                "global_modified": {"global_safetensors_alias_modified": True},
                "wrong_runtime_source": {"runtime_sources": {}},
                "wrong_runtime_version": {
                    "runtime_versions": record["runtime_versions"] | {"torch": "2.0"}
                },
                "wrong_base": {"model_directory": "/other/base"},
                "wrong_backend": {
                    "opens": [item | {"backend": "mmap"} for item in record["opens"]]
                },
                "missing_shard": {"opens": record["opens"][:1]},
                "extra_shard": {
                    "opens": record["opens"]
                    + [{"file": "other", "device": "cpu", "backend": "pread"}]
                },
                "bad_open_type": {"opens": [None]},
                "too_many_opens": {"opens": record["opens"] * 33},
            }
            ci.write(path, record | mutations[mutation])
    state["file_sha256"] = continuation.inventory(root, exclude=("ci-state.json",))
    ci.write(state_path, state)
    with pytest.raises((ValueError, OSError)):
        continuation.verify_first4(root, "grpo", plan)


def test_continued_handoff_cannot_drop_loader_receipt_with_rehashed_inventory(tmp_path, plan):
    output = warm_output(tmp_path, plan)
    path = output / "phases/warm-sample314/loader.json"
    path.unlink()
    allocation = ci.read(output / "handoff.json")["allocation"]
    continuation.write_handoff(output, plan, "warm", allocation=allocation)
    with pytest.raises(OSError):
        continuation.verify_handoff(output, plan, "warm")
