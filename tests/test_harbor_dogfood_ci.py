"""CPU-only tests of the GitHub Harbor dogfood gate: planted failing evidence must fail it.

No Docker, Harbor, network or model. The passing fixture is built from the committed receipts of
the local PASS (reports/2026-10-16-harbor-pretrained-dogfood/evidence); each mutation plants one
defect and must flip the named criterion and the process exit code.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts/harbor_dogfood_ci.py"
EVIDENCE = REPO / "reports/2026-10-16-harbor-pretrained-dogfood/evidence"
SPEC = importlib.util.spec_from_file_location("harbor_dogfood_ci", SCRIPT)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)

PROBE_JOB = "aircraft-harbor-dogfood-local-sft-probe-v4"
MAIN_JOB = "aircraft-harbor-dogfood-local-v4"
RL_ADAPTER = "ee0e7d0faa79405006c1c41b4cacbdbbc47e6fd4025000b5e304aead7108a3bc"
MANIFEST = "2c20d30e3b0c27b2da711fa1b977d7dbb47049f22bef9d5b2af2d47e75e6bfcc"
REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
GIB = 1024**3
CRITERIA = (
    "C1_pretrained_model_loaded",
    "C2_real_optimizer_update",
    "C3_separate_verifier_scored_submission",
    "C4_resource_guards_held",
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        path.write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        path.write_text(json.dumps(value))


def committed(name):
    return json.loads((EVIDENCE / name).read_text())


def rows(n=3):
    return [
        {
            "elapsed_seconds": float(i),
            "tree_rss_bytes": 9 * GIB,
            "cgroup_memory": {"memory.swap.current": 0, "memory.swap.max": 0},
        }
        for i in range(n)
    ]


def stage(directory, guard, loader):
    put(directory / "guard.json", committed(guard))
    put(directory / "loader.json", committed(loader))
    put(directory / "resources.jsonl", rows())


@pytest.fixture
def passing(tmp_path):
    root = tmp_path / "ci"
    root.mkdir()
    ci.extract_plan(REPO, root)
    put(
        root / "prepared/bundle/task/environment/contract.json",
        {"model_revision": REVISION, "model_manifest_sha256": MANIFEST, "max_steps": 1},
    )
    jobs = root / "output/jobs"
    probe = jobs / PROBE_JOB / "sft-resource-task__a"
    probe_art = probe / "artifacts/logs/artifacts"
    stage(
        probe_art / "runtime/sft-resource-probe",
        "guard-sft-resource-probe.json",
        "loader-sft-resource-probe.json",
    )
    put(probe_art / "sft-resource-probe.json", {"receipt": {"global_step": 1}})
    put(jobs / PROBE_JOB / "result.json", {"id": "job", "stats": {}})
    trial = jobs / MAIN_JOB / "task__b"
    art = trial / "artifacts/logs/artifacts"
    for name in ("rl-exposure", "sft-exposure", "sft-time"):
        stage(art / "runtime" / name, f"guard-{name}.json", f"loader-{name}.json")
    put(trial / "verifier/report.json", committed("verifier-report.json"))
    put(trial / "verifier/loader.json", committed("loader-verifier.json"))
    put(trial / "verifier/runtime/guard.json", committed("guard-verifier.json"))
    put(trial / "verifier/runtime/resources.jsonl", rows())
    audit = [
        json.loads(x) for x in (EVIDENCE / "rl-optimizer-audit.jsonl").read_text().split("\n") if x
    ]
    put(art / "training/rl/optimizer_audit.jsonl", audit)
    put(
        art / "training/training-evidence.json",
        {
            "status": "completed",
            "model_source": {"manifest_sha256": MANIFEST},
            "arms": {
                "rl-exposure": {
                    "receipt": {"global_step": 1, "final_adapter": {"weights_sha256": RL_ADAPTER}},
                    "optimizer_audit": {"parameter_delta_l2": 0.0104217373739981},
                },
                "sft-exposure": {"receipt": {"global_step": 1}},
                "sft-time": {
                    "receipt": {"global_step": 7, "committed_active_seconds": 545.3751104729995},
                    "target_seconds": 530.3061584569998,
                },
            },
        },
    )
    put(
        trial / "result.json",
        {"trial_name": "task__b", "verifier_environment_mode": "separate", "exception_info": None},
    )
    put(jobs / MAIN_JOB / "result.json", {"id": "job", "stats": {}})
    return root


def cli(root):
    process = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "criteria", "--root", str(root)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return process.returncode, json.loads(process.stdout)


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def edit_rows(path, change):
    values = [json.loads(x) for x in path.read_text().splitlines() if x]
    change(values)
    put(path, values)


MAIN = Path("output/jobs") / MAIN_JOB / "task__b"
ART = MAIN / "artifacts/logs/artifacts"
PROBE = Path("output/jobs") / PROBE_JOB / "sft-resource-task__a/artifacts/logs/artifacts"


def test_committed_pass_evidence_passes(passing):
    code, result = cli(passing)
    assert (code, result["pass"], result["driver_error"]) == (0, True, None)
    assert result["criteria"] == {name: True for name in CRITERIA}


def set_backend(value):
    value["opens"][0]["backend"] = "mmap"


def uniform_reward(rows_):
    rows_[0]["nonzero_gradient_steps"] = 0


def other_adapter(value):
    value["adapter_sha256"] = "0" * 64


def swap_in_row(rows_):
    rows_[-1]["cgroup_memory"]["memory.swap.current"] = 4096


def rss_over(value):
    value["peak_tree_rss_bytes"] = 12 * GIB + 1


def joint_env(value):
    value["verifier_environment_mode"] = "shared"


def same_adapter(value):
    value["arms"]["rl-exposure"]["receipt"]["final_adapter"]["weights_sha256"] = (
        "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"
    )


PLANTED = [
    ("mmap-loader", ART / "runtime/rl-exposure/loader.json", edit, set_backend, "C1"),
    ("uniform-reward", ART / "training/rl/optimizer_audit.jsonl", edit_rows, uniform_reward, "C2"),
    ("unchanged-adapter", ART / "training/training-evidence.json", edit, same_adapter, "C2"),
    ("wrong-adapter", MAIN / "verifier/report.json", edit, other_adapter, "C3"),
    ("shared-verifier", MAIN / "result.json", edit, joint_env, "C3"),
    (
        "probe-swap",
        PROBE / "runtime/sft-resource-probe/resources.jsonl",
        edit_rows,
        swap_in_row,
        "C4",
    ),
    ("rss-over", ART / "runtime/sft-time/guard.json", edit, rss_over, "C4"),
]


@pytest.mark.parametrize("case", PLANTED, ids=[c[0] for c in PLANTED])
def test_planted_failure_fails_its_criterion_and_exit(passing, case):
    _, relative, editor, change, criterion = case
    editor(passing / relative, change)
    code, result = cli(passing)
    assert code == 1 and result["pass"] is False
    failed = {name for name, ok in result["criteria"].items() if not ok}
    assert any(name.startswith(criterion) for name in failed)


def test_short_sft_time_control_fails_although_c1_to_c4_pass(passing):
    def short(value):
        value["arms"]["sft-time"]["receipt"]["committed_active_seconds"] = 100.0

    edit(passing / ART / "training/training-evidence.json", short)
    code, result = cli(passing)
    assert code == 1 and result["pass"] is False
    assert result["criteria"] == {name: True for name in CRITERIA}
    assert "RL active-time boundary" in result["driver_error"]


def test_missing_verifier_loader_fails(passing):
    (passing / MAIN / "verifier/loader.json").unlink()
    code, result = cli(passing)
    assert code == 1 and not result["criteria"]["C3_separate_verifier_scored_submission"]


def test_empty_run_fails_every_criterion(tmp_path):
    root = tmp_path / "ci"
    root.mkdir()
    ci.extract_plan(REPO, root)
    code, result = cli(root)
    assert code == 1 and result["pass"] is False
    assert result["criteria"] == {name: False for name in CRITERIA}


def test_tampered_plan_tarball_is_refused(tmp_path):
    fake = tmp_path / "repo"
    report = fake / "reports/2026-10-16-harbor-pretrained-dogfood"
    shutil.copytree(REPO / "reports/2026-10-16-harbor-pretrained-dogfood", report)
    with (report / "plan-source.tar.gz").open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ValueError, match="differs from the reviewed tarball"):
        ci.extract_plan(fake, tmp_path / "out")


GOOD_ENV = {
    "GITHUB_REPOSITORY": "RyanLin5967/RyanLinEnv",
    "GITHUB_EVENT_NAME": "workflow_dispatch",
    "GITHUB_RUN_ATTEMPT": "1",
    "RUNNER_ENVIRONMENT": "github-hosted",
}


@pytest.fixture
def plan(tmp_path):
    return json.loads((ci.extract_plan(REPO, tmp_path) / "plan.json").read_text())


def test_launch_gate_accepts_first_manual_hosted_attempt(plan):
    ci.require_ci_launch(plan, environ=dict(GOOD_ENV), platform="linux")


@pytest.mark.parametrize(
    "change",
    [
        {"platform": "darwin"},
        {"GITHUB_EVENT_NAME": "push"},
        {"GITHUB_RUN_ATTEMPT": "2"},
        {"GITHUB_REPOSITORY": "someone/fork"},
        {"RUNNER_ENVIRONMENT": "self-hosted"},
        {"DOCKER_HOST": "tcp://10.0.0.1:2375"},
        {"DOCKER_CONTEXT": "colima"},
    ],
)
def test_launch_gate_refuses(plan, change):
    env = dict(GOOD_ENV, **change)
    platform = env.pop("platform", "linux")
    with pytest.raises(ValueError):
        ci.require_ci_launch(plan, environ=env, platform=platform)


def test_launch_gate_refuses_unreviewed_plan(plan):
    plan["local_launch_reviewed"] = False
    with pytest.raises(ValueError, match="unreviewed"):
        ci.require_ci_launch(plan, environ=dict(GOOD_ENV), platform="linux")


def test_evidence_copy_is_an_allowlist(passing, tmp_path):
    planted = passing / ART / "training/rl"
    (planted / "fixture.jpg").write_bytes(b"\xff\xd8 not shipped")
    (planted / "adapter_model.safetensors").write_bytes(b"weights not shipped")
    (planted / "tokenizer.json").write_text("{}")
    (planted / "checkpoint-1").mkdir()
    (planted / "checkpoint-1/optimizer.pt").write_bytes(b"state")
    (planted / "checkpoint-1/trainer_state.json").write_text("{}")
    (planted / "linked.json").symlink_to(passing / ART / "training/training-evidence.json")
    put(passing / "prepared/bundle/task/environment/data/train/x.jpg", {})
    destination = tmp_path / "evidence"
    copied = ci.collect_evidence(passing, destination)
    names = {Path(p).name for p in copied}
    assert "training-evidence.json" in names and "report.json" in names
    assert not names & {
        "fixture.jpg",
        "adapter_model.safetensors",
        "tokenizer.json",
        "optimizer.pt",
        "trainer_state.json",
        "linked.json",
        "x.jpg",
    }
    shipped = {p.name for p in destination.rglob("*") if p.is_file()}
    assert shipped == names | {"MANIFEST.json"}
    assert not any(p.is_symlink() for p in destination.rglob("*"))
