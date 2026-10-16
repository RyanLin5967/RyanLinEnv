"""Light tests of the manual CPU job's packaging and fail-closed boundaries."""

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "aircraft_ci", Path(__file__).parents[1] / "scripts/aircraft_ci_experiment.py"
)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


def archive(tmp_path, entries):
    path = tmp_path / "fixture.tar"
    with tarfile.open(path, "w") as target:
        for name, content, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(content)
            target.addfile(info, io.BytesIO(content))
    return path


@pytest.mark.parametrize("name", ["/outside", "../outside", "a/../outside", "a//b", "./a", ""])
def test_archive_paths_cannot_escape(name):
    with pytest.raises(ValueError):
        ci.safe_name(name)


def test_exact_numeric_bundle_roundtrip(tmp_path):
    content = b"numeric fixture only"
    expected = {"folder/adapter.safetensors": hashlib.sha256(content).hexdigest()}
    path = archive(tmp_path, [(next(iter(expected)), content, tarfile.REGTYPE)])
    ci.extract_tar(path, tmp_path / "output", expected, limit=100)
    assert (tmp_path / "output/folder/adapter.safetensors").read_bytes() == content


@pytest.mark.parametrize(
    "case", ["symlink", "hardlink", "duplicate", "unknown", "size", "hash", "missing"]
)
def test_rejects_archive_boundary_violations(tmp_path, case):
    content = b"fixture"
    kind = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE}.get(case, tarfile.REGTYPE)
    entries = [("a", content, kind)]
    if case == "duplicate":
        entries *= 2
    expected = {"a": hashlib.sha256(content).hexdigest()}
    if case == "unknown":
        expected = {}
    if case == "missing":
        expected["b"] = expected["a"]
    if case == "hash":
        expected["a"] = "0" * 64
    with pytest.raises(ValueError):
        ci.extract_tar(
            archive(tmp_path, entries),
            tmp_path / "out",
            expected,
            limit=1 if case == "size" else 100,
        )


def test_regular_rejects_parent_symlink(tmp_path):
    directory = tmp_path / "real"
    directory.mkdir()
    (directory / "file").write_text("data")
    (tmp_path / "alias").symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        ci.regular(tmp_path / "alias/file")


def test_failed_phase_never_becomes_continuation(tmp_path):
    ci.write(tmp_path / "phases/grpo-first4/guard.json", {"status": "failed"})
    with pytest.raises(ValueError, match="terminal"):
        ci.completed_phases(tmp_path)


def test_cgroup_attribution_distinguishes_file_anon_and_swap_without_guard_changes(tmp_path):
    (tmp_path / "memory.current").write_text("8192\n")
    (tmp_path / "memory.max").write_text(str(13 * ci.GIB) + "\n")
    (tmp_path / "memory.swap.current").write_text("0\n")
    (tmp_path / "memory.swap.max").write_text("max\n")
    (tmp_path / "memory.stat").write_text("anon 4096\nfile 2048\nkernel 1024\n")
    observed = ci.cgroup_memory(tmp_path)
    assert observed["memory.current"] == 8192
    assert observed["memory.max"] == 13 * ci.GIB
    assert observed["memory.peak"] is None
    assert observed["memory.swap.current"] == 0
    assert observed["memory.swap.max"] == "max"
    assert observed["memory.stat"] == {"anon": 4096, "file": 2048, "kernel": 1024}


@pytest.mark.parametrize("host_swap", [4096, 20480])
def test_unrelated_host_swap_does_not_authorize_or_invalidate_container_swap(host_swap):
    # Both observed failed runners had nonzero host swap and zero container swap.
    # Only an exact kernel-enforced container contract can pass the new check.
    memory = {
        "memory.current": 9 * ci.GIB,
        "memory.max": 13 * ci.GIB,
        "memory.swap.max": 0,
        "memory.swap.current": 0,
        "host_swap_growth_bytes": host_swap,
    }
    ci.require_cgroup_memory(memory, ci.LIMITS)
    memory["memory.swap.current"] = 4096
    with pytest.raises(RuntimeError, match="contract"):
        ci.require_cgroup_memory(memory, ci.LIMITS)


@pytest.mark.parametrize(
    "key,value",
    [
        ("memory.max", None),
        ("memory.max", "max"),
        ("memory.max", 14 * ci.GIB),
        ("memory.swap.max", None),
        ("memory.swap.max", "max"),
        ("memory.swap.max", 4096),
        ("memory.swap.max", False),
        ("memory.swap.current", None),
        ("memory.swap.current", 4096),
        ("memory.current", None),
        ("memory.current", -1),
        ("memory.current", 14 * ci.GIB),
    ],
)
def test_cgroup_contract_rejects_missing_unbounded_or_changed_kernel_limits(key, value):
    memory = {
        "memory.current": ci.GIB,
        "memory.max": 13 * ci.GIB,
        "memory.swap.max": 0,
        "memory.swap.current": 0,
    }
    memory[key] = value
    with pytest.raises(RuntimeError):
        ci.require_cgroup_memory(memory, ci.LIMITS)


def test_supervisor_requires_container_contract_before_spawning_worker(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace())
    monkeypatch.setattr(ci, "ORIGINAL", tmp_path)
    monkeypatch.setattr(ci, "OUTPUT_REL", Path("output"))
    monkeypatch.setattr(ci, "check_inputs", lambda: (object(), {}))
    monkeypatch.setattr(ci, "cgroup_memory", lambda: {})
    monkeypatch.setattr(ci.subprocess, "Popen", lambda *a, **k: pytest.fail("worker started"))
    with pytest.raises(RuntimeError, match="contract"):
        ci.supervise_phase("grpo-first4", ci.LIMITS)


@pytest.mark.parametrize("container_swap", [0, 4096])
def test_supervisor_records_host_swap_but_stops_actual_container_swap(
    tmp_path, monkeypatch, container_swap
):
    monkeypatch.setattr(ci, "ORIGINAL", tmp_path)
    monkeypatch.setattr(ci, "OUTPUT_REL", Path("output"))
    host_swap = iter([0, 20480])
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            swap_memory=lambda: SimpleNamespace(used=next(host_swap)),
            virtual_memory=lambda: SimpleNamespace(available=5 * ci.GIB),
            Process=lambda _: object(),
        ),
    )
    runner = SimpleNamespace(
        _tree_rss=lambda _: ci.GIB,
        output_size=lambda: 0,
        append=lambda path, row: path.write_text(json.dumps(row) + "\n"),
    )
    monkeypatch.setattr(ci, "check_inputs", lambda: (runner, {}))
    initial = {
        "memory.current": ci.GIB,
        "memory.max": 13 * ci.GIB,
        "memory.swap.max": 0,
        "memory.swap.current": 0,
    }
    samples = iter([initial, initial | {"memory.swap.current": container_swap}])
    monkeypatch.setattr(ci, "cgroup_memory", lambda: next(samples))
    child = SimpleNamespace(pid=123, returncode=None)
    child.poll = lambda: child.returncode
    child.wait = lambda **_: child.returncode
    monkeypatch.setattr(ci.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(ci.time, "sleep", lambda _: setattr(child, "returncode", 0))
    kills = []

    def stop(pid, signum):
        kills.append((pid, signum))
        child.returncode = -signum

    monkeypatch.setattr(ci.os, "killpg", stop)
    if container_swap:
        with pytest.raises(RuntimeError, match="Phase failed"):
            ci.supervise_phase("warm-greedy", ci.LIMITS)
    else:
        ci.supervise_phase("warm-greedy", ci.LIMITS)
    phase = tmp_path / "output/phases/warm-greedy"
    guard = ci.read(phase / "guard.json")
    sample = ci.read(phase / "resources.jsonl")
    assert sample["host_swap_growth_bytes"] == 20480
    assert sample["cgroup_memory"]["memory.swap.current"] == container_swap
    assert guard["status"] == ("failed" if container_swap else "completed")
    assert guard["initial_cgroup_memory"] == initial
    assert kills == ([(123, ci.signal.SIGINT)] if container_swap else [])


def test_run_cannot_execute_locally_or_without_manual_event(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(ValueError, match="manual attempt"):
        ci.host_run(tmp_path / "run", tmp_path / "missing.json", "grpo")
    assert not (tmp_path / "run").exists()


def test_docker_preserves_worker_parent_and_resource_contract(tmp_path):
    command = ci.docker_command(tmp_path / "inputs", tmp_path / "output", "sha256:fixed", "grpo")
    assert command[:3] == ["docker", "run", "--init"]
    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert f"EXPERIMENT_UID={os.getuid()}" in command
    assert f"EXPERIMENT_GID={os.getgid()}" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--memory") + 1] == str(13 * ci.GIB)
    assert command[command.index("--memory-swap") + 1] == str(13 * ci.GIB)
    assert command[command.index("--cpus") + 1] == "2"
    assert "HF_DEACTIVATE_ASYNC_LOAD=1" in command
    assert f"AIRCRAFT_LOADER_SHA256={ci.loader_identity()['sha256']}" in command
    assert any(item.endswith(":/ci/aircraft_cpu_loader.py:ro") for item in command)
    assert "--read-only" in command
    assert not any("TOKEN" in item or "docker.sock" in item for item in command)
    assert command[-4:] == ["/ci/experiment.py", "container-stage", "--arm", "grpo"]


def test_filesystem_preflight_exercises_actual_output_before_model(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_UID", str(os.geteuid()))
    monkeypatch.setenv("EXPERIMENT_GID", str(os.getegid()))
    receipt = ci.filesystem_preflight(tmp_path)
    assert receipt["effective_uid"] == tmp_path.stat().st_uid
    assert receipt["create_write_fsync_read_unlink_passed"] is True
    assert receipt["model_calls"] == 0 and receipt["worker_started"] is False
    assert not (tmp_path / ".write-preflight").exists()
    assert ci.read(tmp_path / "filesystem-preflight.json") == receipt


def test_filesystem_preflight_rejects_wrong_identity_without_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_UID", str(os.geteuid() + 1))
    monkeypatch.setenv("EXPERIMENT_GID", str(os.getegid()))
    with pytest.raises(RuntimeError, match="identity"):
        ci.filesystem_preflight(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_container_never_reaches_inputs_or_worker_when_output_not_writable(monkeypatch):
    monkeypatch.setattr(ci.sys, "platform", "linux")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")

    def reject(_):
        raise PermissionError("actual output mount")

    monkeypatch.setattr(ci, "filesystem_preflight", reject)
    monkeypatch.setattr(ci, "check_inputs", lambda: pytest.fail("must not reach model inputs"))
    with pytest.raises(PermissionError, match="actual output"):
        ci.container_stage("grpo")


def test_contract_retains_exact_science_and_public_payload_bounds():
    contract = json.loads(
        (Path(__file__).parents[1] / "docs/aircraft-cpu-experiment-plan.json").read_text()
    )
    assert contract["original_plan_sha256"] == ci.PLAN_SHA
    assert contract["limits"] == ci.LIMITS
    assert contract["scientific_choices_changed"] is False
    assert contract["loader"] == ci.loader_identity()
    assert len([r for r in contract["downloads"] if r["path"].endswith(".jpg")]) == 96
    assert all(
        "/development/images/train/" in r["path"]
        for r in contract["downloads"]
        if r["path"].endswith(".jpg")
    )
    assert contract["model_bytes"] == 4429621382
    assert contract["jpeg_bytes"] == 27237237
    assert contract["bundle"]["bytes"] < 4 * 1024**2
    assert len([name for name in contract["files"] if "/source/vision_rl/" in name]) == 37
    assert not any(
        name.endswith((".jpg", ".safetensors", ".pt", ".pth")) for name in contract["files"]
    )
