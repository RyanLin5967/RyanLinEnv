"""Fixture-only checks for the local plan. Never starts Colima/Docker or loads a model.

The only non-fixture reads are hashing/packaging the 7 TRAIN JPEGs and small model
provenance JSON files (no tensors), exactly as the draft's preview did.
"""

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent / "aircraft-harbor-dogfood-v1-plan"
AUDIT = DRAFT / "claude-audit-v1"
GIB = 1024**3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local = load("local_plan_run", HERE / "local_run.py")
derive_plan = load("local_plan_derive", HERE / "derive_plan.py")
draft_run = load("local_plan_draft_run", DRAFT / "run.py")
fixed_run = load("local_plan_fixed_run", AUDIT / "fixed_run.py")
PLAN = json.loads((HERE / "plan.json").read_text())
DRAFT_PLAN = json.loads((DRAFT / "plan.json").read_text())
WARM = "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- the draft is untouched and the copies are exact ---------------------------------


def test_draft_untouched():
    evidence = json.loads((DRAFT / "preparation-evidence.json").read_text())
    for name, value in evidence["files_sha256"].items():
        assert sha(DRAFT / name) == value, name
    assert sha(DRAFT / "plan.json") == derive_plan.DRAFT_PLAN_SHA


@pytest.mark.parametrize("name", ["aircraft_cpu_loader.py", "profile.json"])
def test_copies_byte_identical(name):
    assert (HERE / name).read_bytes() == (DRAFT / name).read_bytes()
    assert PLAN["copies"][name] == {
        "sha256": sha(DRAFT / name),
        "identical_to_draft": True,
        "change": None,
    }


RUNTIME_PATCHES = ["runtime-systemexit.patch", "runtime-stage-walls.patch"]
PREPARE_PATCHES = ["prepare-task-agent-timeout.patch"]


def apply_patches(tmp_path, name, patches):
    target = tmp_path / name
    target.write_bytes((DRAFT / name).read_bytes())
    for patch in patches:
        subprocess.run(["patch", "-s", str(target), str(HERE / patch)], check=True, timeout=30)
    return target.read_bytes()


@pytest.mark.parametrize(
    "name, patches", [("runtime.py", RUNTIME_PATCHES), ("prepare_task.py", PREPARE_PATCHES)]
)
def test_patched_file_is_draft_plus_declared_patches(tmp_path, name, patches):
    """Each patched file is exactly the draft plus its declared patches, in order."""
    assert apply_patches(tmp_path, name, patches) == (HERE / name).read_bytes()
    assert (HERE / name).read_bytes() != (DRAFT / name).read_bytes()
    copy_ = PLAN["copies"][name]
    assert copy_["identical_to_draft"] is False
    assert copy_["change"] == patches
    assert copy_["draft_sha256"] == sha(DRAFT / name)
    assert copy_["patch_sha256"] == {patch: sha(HERE / patch) for patch in patches}


def test_runtime_keeps_the_systemexit_patch():
    assert any(c["field"] == "runtime.py" and "SystemExit" in c["why"] for c in PLAN["changes"])
    assert "except SystemExit as exc:" in (HERE / "runtime.py").read_text()
    assert "except SystemExit" not in (DRAFT / "runtime.py").read_text()


def test_rederive_refuses_an_undeclared_runtime_edit(tmp_path, monkeypatch):
    fake = tmp_path / "here"
    fake.mkdir()
    for name in derive_plan.LOCAL_FILES:
        (fake / name).write_bytes((HERE / name).read_bytes())
    (fake / "runtime.py").write_text((HERE / "runtime.py").read_text() + "# undeclared\n")
    monkeypatch.setattr(derive_plan, "HERE", fake)
    with pytest.raises(ValueError, match="draft plus patch runtime.py"):
        derive_plan.derive(False)


def test_v3_paths_are_new_and_consistent():
    assert PLAN["name"] == "aircraft-harbor-dogfood-local-v4"
    assert local.DEFAULT_ROOT == local.REPOSITORY_ROOT / PLAN["local"]["root"]
    assert PLAN["local"]["root"] == "runs/aircraft-harbor-dogfood-local-v4"
    assert (local.PROBE_JOB, local.MAIN_JOB) == (
        "aircraft-harbor-dogfood-local-sft-probe-v4",
        "aircraft-harbor-dogfood-local-v4",
    )
    assert (PLAN["local"]["probe_job"], PLAN["local"]["main_job"]) == (
        local.PROBE_JOB,
        local.MAIN_JOB,
    )
    assert PLAN["local"]["result"] == PLAN["local"]["root"] + "/output/" + local.RESULT_NAME
    text = json.dumps({k: PLAN[k] for k in ("local", "gates", "name")})
    for old in ("local-v1", "probe-v1", "local-v2", "probe-v2"):
        assert old not in text, old
    assert not (local.DEFAULT_ROOT / "output").exists()


def test_previous_attempt_is_the_preserved_failed_v2():
    previous = PLAN["previous_attempt"]["v2"]
    assert previous["plan_sha256"].startswith("a2cd71df")
    assert sha(derive_plan.V2_PLAN) == previous["plan_sha256"]
    assert sha(derive_plan.V2_RESULT) == previous["result_sha256"]
    for key in (
        "probe_loader_receipt",
        "probe_guard_receipt",
        "rl_exposure_loader_receipt",
        "rl_exposure_guard_receipt",
    ):
        assert sha(derive_plan.ROOT / previous[key]["path"]) == previous[key]["sha256"], key
    assert previous["probe_loader_receipt"]["status"] == "completed"
    guard = previous["rl_exposure_guard_receipt"]
    assert (guard["status"], guard["exit_code"], guard["wall_seconds"]) == ("failed", -2, 1800)
    assert guard["stop_reason"] == "Disk/output/wall resource guard"
    assert 1801 < guard["elapsed_seconds"] < 1802
    assert "a2cd71df" in PLAN["status"] and "1801.29" in PLAN["status"]


def test_previous_attempt_is_the_preserved_failed_v1():
    previous = PLAN["previous_attempt"]["v1"]
    assert previous["plan_sha256"].startswith("277bb9b2")
    assert sha(derive_plan.V1_PLAN) == previous["plan_sha256"]
    assert sha(derive_plan.V1_RESULT) == previous["result_sha256"]
    for key in ("probe_loader_receipt", "probe_guard_receipt"):
        assert sha(derive_plan.ROOT / previous[key]["path"]) == previous[key]["sha256"]
    assert previous["probe_loader_receipt"]["error_type"] == "SystemExit"
    assert "277bb9b2" in PLAN["status"] and "SystemExit" in PLAN["status"]


# ---- plan.json ------------------------------------------------------------------------


def test_plan_local_fields():
    assert PLAN["training"] == dict(DRAFT_PLAN["training"], max_steps=1)
    # Shipped false; only the reviewer flips it (derive_plan.py --local-launch-reviewed).
    assert type(PLAN["local_launch_reviewed"]) is bool
    assert "launch_reviewed" not in PLAN
    assert "local" in PLAN["scope"] and "not held-out" in PLAN["scope"]
    assert PLAN["status"].startswith("prepared locally and unexecuted")
    assert PLAN["profile"] == json.loads((HERE / "profile.json").read_text())
    assert PLAN["model_identity"]["initial_weights_sha256"] == WARM
    assert PLAN["model_identity"]["revision"] == "895c3a49bc3fa70a340399125c650a463535e71c"
    for key in (
        "source_sha256",
        "loader",
        "provenance",
        "selection",
        "host_source_sha256",
        "data_spec_sha256",
    ):
        assert PLAN[key] == DRAFT_PLAN[key], key


def test_support_pins_match_every_local_file():
    assert set(PLAN["support_sha256"]) == set(derive_plan.LOCAL_FILES)
    for name, value in PLAN["support_sha256"].items():
        assert sha(HERE / name) == value, name


def test_plan_rederives_from_live_files():
    """Every pin is recomputed from live files; only the date may differ."""
    a, b = copy.deepcopy(derive_plan.derive(PLAN["local_launch_reviewed"])), copy.deepcopy(PLAN)
    for value in (a, b):
        value.pop("created_at")
        value["pin_verification"].pop("date")
    assert a == b


def test_rederive_refuses_a_changed_pin(monkeypatch):
    original = derive_plan.sha

    def tampered(path):
        value = original(path)
        return "0" * 64 if Path(path).name == "data_spec.json" else value

    monkeypatch.setattr(derive_plan, "sha", tampered)
    with pytest.raises(ValueError, match="data_spec"):
        derive_plan.derive(False)


# ---- reuse of run.py and the race patch ------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["sha", "stop_owned_containers", "artifact_file", "assert_runtime_receipts", "assert_probe"],
)
def test_functions_verbatim_from_draft(name):
    assert inspect.getsource(getattr(local, name)) == inspect.getsource(getattr(draft_run, name))


@pytest.mark.parametrize("name", ["tree_bytes", "run_harbor"])
def test_race_patch_applied(name):
    assert inspect.getsource(getattr(local, name)) == inspect.getsource(getattr(fixed_run, name))
    if name == "run_harbor":
        assert inspect.getsource(local.run_harbor) != inspect.getsource(draft_run.run_harbor)


def test_patch_reproduces_fixed_run(tmp_path):
    target = tmp_path / "run.py"
    target.write_bytes((DRAFT / "run.py").read_bytes())
    subprocess.run(
        ["patch", "-s", str(target), str(AUDIT / "run-output-size-race.patch")],
        check=True,
        timeout=30,
    )
    assert target.read_bytes() == (AUDIT / "fixed_run.py").read_bytes()


def test_harbor_command_same_flags_as_draft():
    for probe in (False, True):
        ours = local.harbor_command(Path("t"), Path("j"), "n", probe=probe)
        theirs = draft_run.harbor_command(Path("t"), Path("j"), "n", probe=probe)
        assert ours[0] == "/Users/idide/RyanLinEnv/.venv/bin/harbor"
        assert ours[1:] == theirs[4:] and theirs[:4] == ["uv", "run", "--no-sync", "harbor"]


def test_tree_bytes_vanished_entry_counts_zero(tmp_path, monkeypatch):
    (tmp_path / "keep").write_bytes(b"x" * 10)
    (tmp_path / "gone").write_bytes(b"x" * 99)
    real = os.lstat

    def racing(path, *a, **k):
        if str(path).endswith("gone"):
            raise FileNotFoundError(path)
        return real(path, *a, **k)

    monkeypatch.setattr(local.os, "lstat", racing)
    assert local.tree_bytes(tmp_path) == 10
    with pytest.raises(FileNotFoundError):
        local.tree_bytes(tmp_path / "missing-root")


# ---- run_harbor caps (real child process; docker stubbed) -----------------------------


def sleeper(seconds):
    return [sys.executable, "-I", "-c", f"import time; time.sleep({seconds})"]


@pytest.fixture
def stopped(monkeypatch):
    calls = []
    monkeypatch.setattr(local, "stop_owned_containers", calls.append)
    return calls


def test_wall_cap_kills_and_stops_owned(tmp_path, stopped):
    with pytest.raises(RuntimeError, match="wall"):
        local.run_harbor(sleeper(60), tmp_path, timeout=1, env=dict(os.environ), plan_sha="abc")
    assert stopped == ["abc"]


def test_output_cap_two_gib(tmp_path, stopped):
    with (tmp_path / "sparse").open("wb") as stream:
        stream.truncate(2 * GIB + 1)
    with pytest.raises(RuntimeError, match="output/disk"):
        local.run_harbor(sleeper(60), tmp_path, timeout=60, env=dict(os.environ), plan_sha="abc")
    assert stopped == ["abc"]


def test_disk_floor_one_gib(tmp_path, stopped, monkeypatch):
    monkeypatch.setattr(local.shutil, "disk_usage", lambda _: SimpleNamespace(free=GIB - 1))
    with pytest.raises(RuntimeError, match="output/disk"):
        local.run_harbor(sleeper(60), tmp_path, timeout=60, env=dict(os.environ), plan_sha="abc")
    assert stopped == ["abc"]


def test_clean_exit_passes_and_still_stops_owned(tmp_path, stopped):
    local.run_harbor(sleeper(0), tmp_path, timeout=60, env=dict(os.environ), plan_sha="abc")
    assert stopped == ["abc"]


def test_stop_filters_by_plan_label(monkeypatch):
    seen = []
    monkeypatch.setattr(local.subprocess, "check_output", lambda c, **k: seen.append(c) or "id1\n")
    monkeypatch.setattr(local.subprocess, "run", lambda c, **k: seen.append(c))
    local.stop_owned_containers("PLANSHA")
    assert seen == [
        ["docker", "ps", "-q", "--filter", "label=vision-rl.harbor-plan=PLANSHA"],
        ["docker", "stop", "--time", "20", "id1"],
    ]


# ---- launch gate ------------------------------------------------------------------------


REVIEWED = dict(PLAN, local_launch_reviewed=True)
GOOD = dict(
    environ={"DOCKER_HOST": local.DOCKER_HOST}, platform="darwin", cwd=local.REPOSITORY_ROOT
)


def test_gate_accepts_only_exact_local_setup():
    local.require_local_launch(REVIEWED, **GOOD)


@pytest.mark.parametrize(
    "plan,override,match",
    [
        (dict(PLAN, local_launch_reviewed=False), {}, "unreviewed"),
        (dict(REVIEWED, cgroup_policy_reviewed=False), {}, "unreviewed"),
        (dict(REVIEWED, training=dict(PLAN["training"], max_steps=4)), {}, "max_steps"),
        (REVIEWED, {"platform": "linux"}, "darwin"),
        (REVIEWED, {"environ": {}}, "DOCKER_HOST"),
        (REVIEWED, {"environ": {"DOCKER_HOST": "unix:///var/run/docker.sock"}}, "DOCKER_HOST"),
        (
            REVIEWED,
            {"environ": {"DOCKER_HOST": local.DOCKER_HOST, "DOCKER_CONTEXT": "x"}},
            "CONTEXT",
        ),
        (REVIEWED, {"cwd": "/tmp"}, "repository root"),
    ],
)
def test_gate_refuses(plan, override, match):
    with pytest.raises(ValueError, match=match):
        local.require_local_launch(plan, **{**GOOD, **override})


def test_unreviewed_execute_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(local, "read_plan", lambda: dict(PLAN, local_launch_reviewed=False))
    for name in ("verify_pins", "verify_bundle", "image_id", "run_harbor", "sha"):
        monkeypatch.setattr(local, name, lambda *a, **k: pytest.fail("acted before the gate"))
    monkeypatch.setattr(local.subprocess, "Popen", lambda *a, **k: pytest.fail("process"))
    with pytest.raises(ValueError, match="unreviewed"):
        local.execute(tmp_path / "root")
    assert not (tmp_path / "root").exists()


def test_verify_pins_passes_on_live_files():
    local.verify_pins(PLAN)


# ---- fixture Harbor job trees for success criteria -------------------------------------


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def guard(**over):
    memory = {"memory.max": 13 * GIB, "memory.swap.max": 0, "memory.swap.current": 0}
    return {
        "status": "completed",
        "exit_code": 0,
        "peak_tree_rss_bytes": 11 * GIB,
        "cgroup_memory": memory,
        **over,
    }


def loader():
    policy = PLAN["loader"]
    return {
        **policy,
        "status": "completed",
        "aliases_restored": True,
        "environment_restored": True,
        "model_directory": "/opt/pinned-model",
        "runtime_sources": policy["source_sha256"],
        "opens": [{"file": s, "device": "cpu", "backend": "pread"} for s in policy["shards"]],
    }


def phase(directory):
    write(directory / "guard.json", guard())
    write(directory / "loader.json", loader())
    row = {"tree_rss_bytes": 10 * GIB, "cgroup_memory": guard()["cgroup_memory"]}
    (directory / "resources.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")


def fixture_run(root):
    jobs = root / "output/jobs"
    probe = jobs / local.PROBE_JOB / "trial__p"
    phase(probe / "artifacts/runtime/sft-resource-probe")
    write(probe / "artifacts/sft-resource-probe.json", {"receipt": {"global_step": 1}})
    trial = jobs / local.MAIN_JOB / "trial__m"
    for stage in ("rl-exposure", "sft-exposure", "sft-time"):
        phase(trial / "artifacts/runtime" / stage)
    adapter = "a" * 64
    receipt = {
        "global_step": 1,
        "committed_active_seconds": 600,
        "final_adapter": {"weights_sha256": adapter},
    }
    write(
        trial / "artifacts/training/training-evidence.json",
        {
            "status": "completed",
            "model_source": {"manifest_sha256": PLAN["model_identity"]["manifest_sha256"]},
            "arms": {
                "rl-exposure": {
                    "receipt": receipt,
                    "optimizer_audit": {"parameter_delta_l2": 0.01, "changed_tensors": 112},
                },
                "sft-exposure": {
                    "receipt": dict(receipt, final_adapter={"weights_sha256": "b" * 64})
                },
                "sft-time": {
                    "receipt": dict(receipt, committed_active_seconds=620),
                    "target_seconds": 600,
                },
            },
        },
    )
    audit = {
        "step": 1,
        "optimizer_steps_observed": 1,
        "nonzero_gradient_steps": 1,
        "first_nonzero_update": {"global_step": 1, "changed_tensors_since_invocation_start": 112},
    }
    (trial / "artifacts/training/rl").mkdir(parents=True)
    (trial / "artifacts/training/rl/optimizer_audit.jsonl").write_text(json.dumps(audit) + "\n")
    (trial / "artifacts/training/sft").mkdir(parents=True)
    (trial / "artifacts/training/sft/optimizer_audit.jsonl").write_text(
        json.dumps(dict(audit, nonzero_gradient_steps=0)) + "\n"
    )
    write(
        trial / "verifier/report.json",
        {
            "scope": "integration_only",
            "submitted_code_executed": False,
            "adapter_sha256": adapter,
            "reward": 0.0,
        },
    )
    write(trial / "verifier/loader.json", loader())
    write(trial / "verifier/runtime/guard.json", guard())
    row = {"tree_rss_bytes": 9 * GIB, "cgroup_memory": guard()["cgroup_memory"]}
    (trial / "verifier/runtime/resources.jsonl").write_text(json.dumps(row) + "\n")
    write(
        trial / "result.json",
        {"trial_name": "trial__m", "verifier_environment_mode": "separate", "exception_info": None},
    )
    write(jobs / local.MAIN_JOB / "result.json", {"n_total_trials": 1})
    return jobs


CONTRACT = {
    "model_revision": PLAN["model_identity"]["revision"],
    "model_manifest_sha256": PLAN["model_identity"]["manifest_sha256"],
}
CRITERIA = (
    "C1_pretrained_model_loaded",
    "C2_real_optimizer_update",
    "C3_separate_verifier_scored_submission",
    "C4_resource_guards_held",
)


def evaluate(root):
    return local.evaluate(root, PLAN, CONTRACT, plan_sha="x")


def test_fixture_run_passes_everything_with_zero_accuracy(tmp_path):
    jobs = fixture_run(tmp_path)
    local.assert_probe(jobs / local.PROBE_JOB)
    local.assert_complete(jobs / local.MAIN_JOB, 1)
    result = evaluate(tmp_path)
    assert result["pass"], result["criteria"]
    assert result["verifier_reward_recorded_not_judged"] == 0.0
    assert result["accuracy_is_pass_criterion"] is False


def test_empty_run_is_not_a_pass(tmp_path):
    result = evaluate(tmp_path)
    assert not result["pass"]
    assert not any(result["criteria"][c]["pass"] for c in CRITERIA)


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


M = Path("output/jobs") / local.MAIN_JOB / "trial__m"
P = Path("output/jobs") / local.PROBE_JOB / "trial__p"
EVIDENCE = M / "artifacts/training/training-evidence.json"


def rl(value):
    return value["arms"]["rl-exposure"]


MUTATIONS = {
    "rl loader missing": (
        "C1",
        lambda r: (r / M / "artifacts/runtime/rl-exposure/loader.json").unlink(),
    ),
    "rl loader mmap": (
        "C1",
        lambda r: edit(
            r / M / "artifacts/runtime/rl-exposure/loader.json",
            lambda v: v["opens"][0].update(backend="mmap"),
        ),
    ),
    "wrong base manifest": (
        "C1",
        lambda r: edit(r / EVIDENCE, lambda v: v["model_source"].update(manifest_sha256="0" * 64)),
    ),
    "uniform reward zero gradient": (
        "C2",
        lambda r: (r / M / "artifacts/training/rl/optimizer_audit.jsonl").write_text(
            json.dumps({"step": 1, "optimizer_steps_observed": 1, "nonzero_gradient_steps": 0})
            + "\n"
        ),
    ),
    "adapter unchanged from warm": (
        "C2",
        lambda r: edit(
            r / EVIDENCE, lambda v: rl(v)["receipt"]["final_adapter"].update(weights_sha256=WARM)
        ),
    ),
    "zero parameter delta": (
        "C2",
        lambda r: edit(
            r / EVIDENCE, lambda v: rl(v)["optimizer_audit"].update(parameter_delta_l2=0.0)
        ),
    ),
    "verifier error": (
        "C3",
        lambda r: edit(r / M / "verifier/report.json", lambda v: v.update(error={"type": "X"})),
    ),
    "verifier graded other adapter": (
        "C3",
        lambda r: edit(r / M / "verifier/report.json", lambda v: v.update(adapter_sha256="c" * 64)),
    ),
    "shared verifier env": (
        "C3",
        lambda r: edit(
            r / M / "result.json", lambda v: v.update(verifier_environment_mode="shared")
        ),
    ),
    "probe guard stopped": (
        "C4",
        lambda r: edit(
            r / P / "artifacts/runtime/sft-resource-probe/guard.json",
            lambda v: v.update(stop_reason="Memory resource guard"),
        ),
    ),
    "swap in one row": (
        "C4",
        lambda r: (
            (r / M / "artifacts/runtime/sft-time/resources.jsonl")
            .open("a")
            .write(
                json.dumps(
                    {
                        "tree_rss_bytes": 1,
                        "cgroup_memory": {"memory.swap.current": 4096, "memory.swap.max": 0},
                    }
                )
                + "\n"
            )
        ),
    ),
    "rss above 12 GiB": (
        "C4",
        lambda r: edit(
            r / M / "artifacts/runtime/sft-exposure/guard.json",
            lambda v: v.update(peak_tree_rss_bytes=12 * GIB + 1),
        ),
    ),
    "probe never ran": (
        "C4",
        lambda r: __import__("shutil").rmtree(r / "output/jobs" / local.PROBE_JOB),
    ),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_criterion_fires_on_its_defect(tmp_path, name):
    target, mutate = MUTATIONS[name]
    fixture_run(tmp_path)
    mutate(tmp_path)
    result = evaluate(tmp_path)
    failed = {c.split("_")[0] for c in CRITERIA if not result["criteria"][c]["pass"]}
    assert target in failed and not result["pass"]


def test_assert_complete_expects_one_step(tmp_path):
    jobs = fixture_run(tmp_path)
    edit(tmp_path / EVIDENCE, lambda v: v["arms"]["sft-exposure"]["receipt"].update(global_step=4))
    with pytest.raises(ValueError, match="exactly one update"):
        local.assert_complete(jobs / local.MAIN_JOB, 1)
    with pytest.raises(ValueError):
        local.assert_complete(jobs / local.MAIN_JOB, 4)


# ---- execute ordering with every external action stubbed ---------------------------------


def stub_execute(monkeypatch, tmp_path, probe_error=None):
    calls = []
    monkeypatch.setattr(local, "read_plan", lambda: REVIEWED)
    monkeypatch.setattr(local, "require_local_launch", lambda plan: calls.append("gate"))
    monkeypatch.setattr(local, "verify_pins", lambda plan: calls.append("pins"))
    monkeypatch.setattr(local, "verify_bundle", lambda *a: calls.append("bundle") or CONTRACT)
    monkeypatch.setattr(local, "image_id", lambda: "sha256:img")

    def fake_harbor(command, output, *, timeout, env, plan_sha):
        calls.append(
            (
                "harbor",
                command[command.index("--job-name") + 1],
                timeout,
                "--disable-verification" in command,
            )
        )
        assert env["PYTHONPATH"] == str(local.FROZEN_SOURCE) and plan_sha == local.sha(
            HERE / "plan.json"
        )
        if probe_error and len(calls) == 4:
            raise RuntimeError(probe_error)

    monkeypatch.setattr(local, "run_harbor", fake_harbor)
    monkeypatch.setattr(local, "assert_probe", lambda job: calls.append("assert_probe"))
    monkeypatch.setattr(
        local, "assert_complete", lambda job, steps: calls.append(("assert_complete", steps))
    )
    return calls


def test_execute_runs_probe_then_main_with_caps(monkeypatch, tmp_path):
    calls = stub_execute(monkeypatch, tmp_path)
    result = local.execute(tmp_path)
    assert calls == [
        "gate",
        "pins",
        "bundle",
        ("harbor", local.PROBE_JOB, 2300, True),
        "assert_probe",
        ("harbor", local.MAIN_JOB, 12000, False),
        ("assert_complete", 1),
    ]
    written = json.loads((tmp_path / "output" / local.RESULT_NAME).read_text())
    assert written == result and result["driver_error"] is None
    assert not result["pass"]  # nothing real ran: criteria cannot pass on an empty tree
    with pytest.raises(FileExistsError):
        local.execute(tmp_path)  # single use


def test_execute_probe_failure_skips_main(monkeypatch, tmp_path):
    calls = stub_execute(monkeypatch, tmp_path, probe_error="Harbor host physical wall guard")
    result = local.execute(tmp_path)
    assert [c for c in calls if isinstance(c, tuple)] == [("harbor", local.PROBE_JOB, 2300, True)]
    assert "wall" in result["driver_error"] and not result["pass"]


# ---- packaging preview (no Docker, no tensors) and build context ------------------------


# .venv has no safetensors; .venv-detection does. Packaging imports safetensors.numpy
# only (checked with -X importtime: no torch import), so no MPS or model work.
PACKAGER = "/Users/idide/RyanLinEnv/.venv-detection/bin/python"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("prepared") / "bundle"
    subprocess.run(
        [
            PACKAGER,
            "-I",
            "-B",
            str(HERE / "prepare_task.py"),
            "--root",
            "/Users/idide/RyanLinEnv",
            "--output",
            str(output),
            "--base-image",
            local.IMAGE,
        ],
        check=True,
        timeout=240,
        capture_output=True,
    )
    return output


def test_packaged_bundle_matches_this_plan(bundle):
    contract = local.verify_bundle(bundle, sha(HERE / "plan.json"), PLAN)
    assert contract["max_steps"] == 1 and contract["control_step_cap"] == 32
    assert "sft-probe" in (bundle / "sft-resource-task/environment/train.sh").read_text()
    assert "runtime.py train" in (bundle / "task/environment/train.sh").read_text()


def test_packaged_bundle_rejected_for_other_plan(bundle):
    with pytest.raises(ValueError, match="different plan"):
        local.verify_bundle(bundle, "0" * 64, PLAN)


def test_packaged_agent_timeout_is_the_raised_one(bundle):
    for task in ("task", "sft-resource-task"):
        text = (bundle / task / "task.toml").read_text()
        assert "[agent]\ntimeout_sec = 9900.0\n" in text, task


def test_packaged_bundle_rejected_for_other_agent_timeout(bundle):
    other = copy.deepcopy(PLAN)
    other["local"]["agent_timeout_seconds"] = 7200
    with pytest.raises(ValueError, match="agent timeout"):
        local.verify_bundle(bundle, sha(HERE / "plan.json"), other)


# ---- v3 walls (timing.json from the failed v2 run) ------------------------------------

TIMING = json.loads((HERE / "timing.json").read_text())
V2_RL = (
    derive_plan.ROOT
    / "runs/aircraft-harbor-dogfood-local-v2/output/jobs/aircraft-harbor-dogfood-local-v2"
)


def test_wall_values():
    """The declared v3 walls, literally; each change is declared in plan.json changes."""
    assert PLAN["local"]["stage_wall_seconds"] == {
        "sft-resource-probe": 2000,
        "rl-exposure": 4000,
        "sft-exposure": 2000,
        "sft-time": 3600,
    }
    assert PLAN["local"]["verify_wall_seconds"] == 1800
    assert PLAN["local"]["agent_timeout_seconds"] == 9900
    assert PLAN["local"]["probe_wall_seconds"] == 2300
    assert PLAN["local"]["main_wall_seconds"] == 12000
    assert PLAN["local"]["outer_timeout_seconds"] == 14600
    fields = " ".join(c["field"] for c in PLAN["changes"])
    for stage in ("rl-exposure", "sft-exposure", "sft-resource-probe"):
        assert f"STAGE_WALL_SECONDS[{stage}]" in fields, stage
    assert "timeout_sec" in fields and "main_wall_seconds" in fields
    assert isinstance(PLAN["local_launch_reviewed"], bool)  # flipped by the documented review step


def test_runtime_stage_walls_match_the_plan():
    runtime = load("runtime_walls_under_test", HERE / "runtime.py")
    assert runtime.STAGE_WALL_SECONDS == PLAN["local"]["stage_wall_seconds"]
    source = inspect.getsource(runtime.controller)
    assert "STAGE_WALL_SECONDS[stage]" in source and "1800" not in source
    main = inspect.getsource(runtime.main)
    assert 'Path("/logs/verifier/runtime"), 1800' in main  # verify wall unchanged
    draft = load("runtime_draft_walls", DRAFT / "runtime.py")
    assert "3600 if stage == \"sft-time\" else 1800" in inspect.getsource(draft.controller)


def test_timing_json_matches_the_v2_receipts():
    """Spot values read independently of derive_plan from the preserved v2 receipts."""
    (guard_path,) = V2_RL.rglob("runtime/rl-exposure/guard.json")
    guard = json.loads(guard_path.read_text())
    (record_path,) = V2_RL.rglob("training/rl/local_training.json")
    active = json.loads(record_path.read_text())["active_time_audit"]["committed_active_seconds"]
    rl = TIMING["rl-exposure"]
    assert rl["guard_elapsed_seconds"] == round(guard["elapsed_seconds"], 2) == 1801.29
    assert rl["committed_active_seconds"] == round(active, 2) == 752.76
    assert rl["peak_tree_rss_bytes"] == 12138778624
    assert TIMING["sft-resource-probe"]["guard_elapsed_seconds"] == 921.26
    assert TIMING["sft-resource-probe"]["committed_active_seconds"] == 83.13
    marks = rl["phase_marks_seconds_since_guard_start"]
    assert marks["before_train_diagnostic_written"] < marks["step_1_checkpoint_committed"]
    assert marks["step_1_checkpoint_committed"] < marks["after_train_diagnostic_written"]
    assert marks["after_train_diagnostic_written"] < rl["guard_elapsed_seconds"]
    assert rl["guard_elapsed_seconds"] < rl["projected_need_seconds"]
    assert TIMING["sft-time"]["steps_to_reach_target_at_probe_step"] <= 32
    assert TIMING["sft-time"]["target_active_seconds"] == 752.76


def test_every_wall_is_at_least_twice_its_need():
    walls = PLAN["local"]["stage_wall_seconds"]
    for stage, row in TIMING["walls"].items():
        assert walls[stage] >= 2 * row["need_seconds"], stage
    # 3600 s, the first proposal for rl-exposure, is below 2x even the lower bound.
    assert 3600 < 2 * TIMING["rl-exposure"]["guard_elapsed_seconds"]
    assert PLAN["local"]["verify_wall_seconds"] >= 2 * TIMING["verify"]["need_bound_seconds"]
    assert PLAN["local"]["agent_timeout_seconds"] >= 2 * TIMING["harbor_agent_timeout"]["need_seconds"]


def test_outer_walls_enclose_inner_walls():
    local.require_wall_nesting(PLAN)
    w = PLAN["local"]
    stages = w["stage_wall_seconds"]
    trained = stages["rl-exposure"] + stages["sft-exposure"] + stages["sft-time"]
    assert w["agent_timeout_seconds"] >= trained + 300
    assert w["probe_wall_seconds"] >= stages["sft-resource-probe"] + 300
    assert w["main_wall_seconds"] >= w["agent_timeout_seconds"] + w["verify_wall_seconds"] + 300
    assert w["outer_timeout_seconds"] >= w["probe_wall_seconds"] + w["main_wall_seconds"] + 300


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("agent_timeout_seconds", 7200, "agent timeout"),
        ("probe_wall_seconds", 2100, "Probe driver wall"),
        ("main_wall_seconds", 9300, "Main driver wall"),
    ],
)
def test_wall_nesting_refuses_the_v2_values(key, value, match):
    other = copy.deepcopy(PLAN)
    other["local"][key] = value
    with pytest.raises(ValueError, match=match):
        local.require_wall_nesting(other)


def test_execute_refuses_unnested_walls_before_any_launch(monkeypatch, tmp_path):
    calls = stub_execute(monkeypatch, tmp_path)
    other = copy.deepcopy(REVIEWED)
    other["local"]["main_wall_seconds"] = 9300
    monkeypatch.setattr(local, "read_plan", lambda: other)
    with pytest.raises(ValueError, match="Main driver wall"):
        local.execute(tmp_path)
    assert calls == ["gate"] and not (tmp_path / "output").exists()


def test_readme_outer_timeout_matches_the_plan():
    text = (HERE / "README.md").read_text()
    assert f"timeout {PLAN['local']['outer_timeout_seconds']} " in text
    assert "timeout 11700" not in text


def test_rederive_refuses_a_stale_timing_json(tmp_path, monkeypatch):
    fake = tmp_path / "here"
    fake.mkdir()
    for name in derive_plan.LOCAL_FILES:
        (fake / name).write_bytes((HERE / name).read_bytes())
    timing = json.loads((fake / "timing.json").read_text())
    timing["rl-exposure"]["projected_need_seconds"] = 1800.0
    (fake / "timing.json").write_text(json.dumps(timing))
    monkeypatch.setattr(derive_plan, "HERE", fake)
    with pytest.raises(ValueError, match="timing.json"):
        derive_plan.derive(False)


def test_rederive_refuses_a_wall_below_twice_the_need(monkeypatch):
    monkeypatch.setitem(derive_plan.STAGE_WALLS, "rl-exposure", 3600)
    with pytest.raises(ValueError, match="rl-exposure wall 3600 < 2x"):
        derive_plan.measure_v2_timing()


def test_packaged_task_separate_verifier_and_train_isolation(bundle):
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    task = Task(bundle / "task")
    definition = resolve_verifier_environment_definition(task.config, task.paths)
    assert definition.directory == bundle / "task/environment" and not definition.bundled_tests
    training = (bundle / "task/environment/data/train.jsonl").read_text()
    checkers = (bundle / "task/tests/data/verification.jsonl").read_text()
    for number in PLAN["selection"]["checker_train"]:
        assert number not in training and number in checkers
    assert not (bundle / "task/environment/tests").exists()


def test_build_context_exact_pinned_set(tmp_path):
    context = local.build_context(tmp_path / "ctx")
    files = {str(p.relative_to(context)) for p in context.rglob("*") if p.is_file()}
    assert files == set(PLAN["build_files"]) | {"src/vision_rl/" + n for n in PLAN["source_sha256"]}
    with pytest.raises(ValueError, match="new"):
        local.build_context(context)


# ---- regression: the native trainer's SystemExit vs the scoped loader receipt ------------
#
# v1 (plan 277bb9b2) failed here: the frozen trainer ends with `raise SystemExit(main())`,
# so a successful probe raised SystemExit(0) through scoped_loader and the receipt read
# status failed / error_type SystemExit. These run runtime.child() for real, against a stub
# vision_rl.vlm_local_training on sys.path, with no model, Docker or Linux worker.

STUB_TRAINER = """
import transformers.modeling_utils as modeling
from pathlib import Path
import os
for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
    modeling.safe_open(str(Path(os.environ["STUB_MODEL"]) / name), framework="pt")
if __name__ == "__main__":
    raise {exit}
"""


def stub_loader_module():
    """A loader with the pinned loader's receipt contract and no environment demands."""
    from contextlib import contextmanager

    @contextmanager
    def scoped_loader(model_directory, receipt_path, *, expected_sha256):
        receipt = {"status": "active", "opens": []}
        modeling = sys.modules["transformers.modeling_utils"]
        original = modeling.safe_open

        def opened(filename, **kwargs):
            receipt["opens"].append(Path(filename).name)
            return original(filename, **kwargs)

        modeling.safe_open = opened
        try:
            yield receipt
            receipt["status"] = "completed"
        except BaseException as exc:
            receipt.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            modeling.safe_open = original
            Path(receipt_path).write_text(json.dumps(receipt))

    return SimpleNamespace(scoped_loader=scoped_loader)


def pinned_loader_module(runtime):
    """The real pinned aircraft_cpu_loader.py with only its Linux/version probes stubbed."""
    module = runtime.loader_module()
    module.sys = SimpleNamespace(platform="linux")
    module._runtime = lambda: ({"stub": "versions"}, {"stub": "sources"})
    return module


LOADERS = {"stub": lambda runtime: stub_loader_module(), "pinned": pinned_loader_module}


@pytest.fixture
def native_child(tmp_path, monkeypatch):
    """Return run(runtime_path, exit_expr, loader) -> (raised SystemExit or None, receipt)."""
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in ("vision_rl", "transformers", "safetensors")}
    for name in saved:
        monkeypatch.delitem(sys.modules, name)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (model / name).write_bytes(b"stub")
    opened = []
    safe_open = lambda filename, **kwargs: opened.append(Path(filename).name)  # noqa: E731
    safetensors = SimpleNamespace(safe_open=safe_open)
    modeling = SimpleNamespace(safe_open=safe_open)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(modeling_utils=modeling))
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling)
    monkeypatch.setenv("STUB_MODEL", str(model))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HARBOR_MODEL_PARENT", str(os.getppid()))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    count = [0]

    def run(runtime_path, exit_expr, loader, stage="native"):
        count[0] += 1
        package = tmp_path / f"stub{count[0]}" / "vision_rl"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "vlm_local_training.py").write_text(STUB_TRAINER.format(exit=exit_expr))
        monkeypatch.syspath_prepend(str(package.parent))
        for name in [k for k in sys.modules if k.split(".")[0] == "vision_rl"]:
            del sys.modules[name]
        runtime = load(f"runtime_under_test_{count[0]}", runtime_path)
        runtime.HERE = HERE  # the pinned loader file lives next to the v2 runtime.py
        artifacts = tmp_path / f"artifacts{count[0]}"
        monkeypatch.setattr(runtime, "ARTIFACTS", artifacts)
        monkeypatch.setattr(runtime, "MODEL", model)
        monkeypatch.setattr(runtime.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None))
        pins = {"contract_sha256": "c" * 64, "loader": {"sha256": sha(HERE / "aircraft_cpu_loader.py")}}
        def verify_submission():
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
                sys.modules["transformers.modeling_utils"].safe_open(str(model / name), framework="pt")
            return {"reward": 0.0}

        builder = SimpleNamespace(
            _read_contract=lambda path: ({}, None), verify_submission=verify_submission
        )
        monkeypatch.setattr(runtime, "checked_builder", lambda: (builder, pins))
        real_sha = runtime.sha
        monkeypatch.setattr(
            runtime, "sha", lambda path: "c" * 64 if str(path) in ("/app/contract.json", "/tests/contract.json") else real_sha(path)
        )
        monkeypatch.setattr(runtime, "checked_native_argv", lambda *a, **k: "sft-resource-probe")
        module = LOADERS[loader](runtime)
        monkeypatch.setattr(runtime, "loader_module", lambda: module)
        raised = None
        verifier = tmp_path / f"verifier{count[0]}" / "loader.json"
        if stage == "verify-child":
            real_path = runtime.Path
            monkeypatch.setattr(
                runtime,
                "Path",
                lambda p, *a: verifier if str(p) == "/logs/verifier/loader.json" else real_path(p, *a),
            )
        try:
            argv = ["--stage-id", "sft-resource-probe"] if stage == "native" else []
            runtime.child(stage, argv)
        except SystemExit as exc:
            raised = exc
        receipt = artifacts / "runtime/sft-resource-probe/loader.json"
        if stage == "verify-child":
            receipt = verifier
        return raised, json.loads(receipt.read_text())

    yield run
    for name in [k for k in sys.modules if k.split(".")[0] == "vision_rl"]:
        del sys.modules[name]


@pytest.mark.parametrize("loader", sorted(LOADERS))
@pytest.mark.parametrize("exit_expr", ["SystemExit(0)", "SystemExit(None)", "SystemExit()"])
def test_clean_trainer_exit_completes_loader_receipt(native_child, loader, exit_expr):
    raised, receipt = native_child(HERE / "runtime.py", exit_expr, loader)
    assert raised is None
    assert receipt["status"] == "completed" and "error_type" not in receipt
    assert len(receipt["opens"]) == 2


@pytest.mark.parametrize("loader", sorted(LOADERS))
@pytest.mark.parametrize("code", [3, 1, "fatal"])
def test_failed_trainer_exit_propagates_and_fails_receipt(native_child, loader, code):
    raised, receipt = native_child(HERE / "runtime.py", f"SystemExit({code!r})", loader)
    assert raised is not None and raised.code == code
    assert receipt["status"] == "failed" and receipt["error_type"] == "SystemExit"


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_other_trainer_errors_still_fail_receipt(native_child, loader):
    with pytest.raises(RuntimeError, match="boom"):
        native_child(HERE / "runtime.py", "RuntimeError('boom')", loader)


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_regression_fires_on_unpatched_draft_runtime(native_child, loader):
    """The same scenario against the draft runtime.py reproduces the v1 failure."""
    raised, receipt = native_child(DRAFT / "runtime.py", "SystemExit(0)", loader)
    assert raised is not None and raised.code == 0
    assert receipt["status"] == "failed" and receipt["error_type"] == "SystemExit"


def test_pinned_loader_receipt_passes_assert_probe_shape(native_child, tmp_path):
    """With the patch, the real loader's receipt clears the draft's completion check."""
    raised, receipt = native_child(HERE / "runtime.py", "SystemExit(0)", "pinned")
    assert raised is None
    assert receipt["status"] == "completed"
    assert receipt["aliases_restored"] is True and receipt["environment_restored"] is True
    assert {c["file"] for c in receipt["opens"]} == set(PLAN["loader"]["shards"])


@pytest.mark.parametrize("loader", sorted(LOADERS))
@pytest.mark.parametrize("runtime_path", [HERE / "runtime.py", DRAFT / "runtime.py"])
def test_verify_child_never_had_the_systemexit_defect(native_child, loader, runtime_path):
    """verify_submission returns a report; the verifier receipt completes in both versions."""
    raised, receipt = native_child(runtime_path, "SystemExit(0)", loader, stage="verify-child")
    assert raised is None and receipt["status"] == "completed" and len(receipt["opens"]) == 2
