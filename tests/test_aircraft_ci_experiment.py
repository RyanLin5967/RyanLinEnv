"""Light tests of the manual CPU job's packaging and fail-closed boundaries."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

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


def test_run_cannot_execute_locally_or_without_manual_event(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(ValueError, match="manual attempt"):
        ci.host_run(tmp_path / "run", tmp_path / "missing.json", "grpo")
    assert not (tmp_path / "run").exists()


def test_docker_preserves_worker_parent_and_resource_contract(tmp_path):
    command = ci.docker_command(tmp_path / "inputs", tmp_path / "output", "sha256:fixed", "grpo")
    assert command[:3] == ["docker", "run", "--init"]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--memory") + 1] == str(13 * ci.GIB)
    assert command[command.index("--memory-swap") + 1] == str(13 * ci.GIB)
    assert command[command.index("--cpus") + 1] == "2"
    assert "--read-only" in command
    assert not any("TOKEN" in item or "docker.sock" in item for item in command)
    assert command[-4:] == ["/ci/experiment.py", "container-stage", "--arm", "grpo"]


def test_contract_retains_exact_science_and_public_payload_bounds():
    contract = json.loads(
        (Path(__file__).parents[1] / "docs/aircraft-cpu-experiment-plan.json").read_text()
    )
    assert contract["original_plan_sha256"] == ci.PLAN_SHA
    assert contract["limits"] == ci.LIMITS
    assert contract["scientific_choices_changed"] is False
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
