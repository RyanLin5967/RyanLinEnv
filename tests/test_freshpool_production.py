"""Tests for the freshpool production launch package (scripts/freshpool_launch/production).

What runs in CI: the committed receipts against launch.py's own G0 checks (the activated
protocol copy, the decision record), the branch receipt against comparison.json read
independently, the plan's templates against launch.py's plan-load rules and the study's fixed
profile, and start.py / chain.py / status.py on temporary directories with a fake launcher.
What cannot run in CI: anything under DATA_ROOT (manifests, model, runtime pins); start.py's
verify-only covers those at launch. Expected values are written out from the study, the
report or the fixture design, never read back from prepare.py.
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "scripts" / "freshpool_launch" / "production"
sys.path.insert(0, str(PRODUCTION))

import start  # noqa: E402
import status  # noqa: E402

PROTOCOL_ID = "fgvc-aircraft-freshpool-screened-grpo-v1"
PLAN = json.loads((PRODUCTION / "plan.json").read_text())
REVISION = json.loads((ROOT / "docs" / "aircraft-rl-protocol-freshpool-v1.json").read_text())
ACTIVATED = json.loads(
    (ROOT / "docs" / "aircraft-rl-protocol-freshpool-v1-activated.json").read_text())
BRANCH = json.loads((ROOT / "docs" / "aircraft-freshpool-branch-receipt-v1.json").read_text())
COMPARISON = json.loads(
    (ROOT / "reports" / "2026-10-17-controlled400-native-val" / "comparison.json").read_text())


def load_launch():
    spec = importlib.util.spec_from_file_location(
        "freshpool_launch_production_test", ROOT / "scripts" / "freshpool_launch" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flag(argv, name):
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == name]


# ---------------------------------------------------------------- receipts


def test_committed_activated_protocol_passes_launch_g0():
    """launch.py's own check, on the committed files (the CI checkout has them committed)."""
    launch = load_launch()
    launch.verify_protocol(ROOT / "docs" / "aircraft-rl-protocol-freshpool-v1-activated.json")
    assert ACTIVATED["protocol_id"] != PROTOCOL_ID and ACTIVATED["activated"] is True
    assert ACTIVATED["data"]["train_holdout"]["hash_protocol_id"] == PROTOCOL_ID
    assert ACTIVATED["revises"]["baseline_discovery_salt"]["literal"] == PROTOCOL_ID
    differing = sorted(k for k in set(ACTIVATED) | set(REVISION)
                       if ACTIVATED.get(k) != REVISION.get(k))
    assert differing == ["activated", "activation", "protocol_id", "status"]
    assert REVISION["activated"] is False  # the revision itself stays unactivated


def test_committed_decision_record_retires_the_16_shot_plan():
    load_launch().verify_decision_record(ROOT / "docs" / "decision-log-2026-10-16-claude.md")


def test_branch_receipt_matches_the_native_val_comparison():
    versus = COMPARISON["versus_common_initialization"]
    w = versus["sft-1e5-time"]["comparator"]
    best = versus["sft-1e5-time"]["candidate"]
    # Written out from the report (reports/2026-10-17-controlled400-native-val/README.md).
    assert round(BRANCH["w_val_accuracy"], 4) == 0.6784 == round(w["mean_class_accuracy"], 4)
    assert round(BRANCH["w_valid_eos"], 3) == 0.973
    assert BRANCH["a_star"] == best["mean_class_accuracy"]
    assert round(BRANCH["a_star"], 4) == 0.7539
    assert all(v["candidate"]["mean_class_accuracy"] <= BRANCH["a_star"]
               for v in versus.values())
    assert (BRANCH["d_req"], BRANCH["mode"], BRANCH["g3_margin"]) == (0.05, "full", 0.02)
    assert BRANCH["b4_edge_extension"] is False and BRANCH["w_evaluator_refused"] is False
    branches = BRANCH["branches"]
    assert branches["B2"]["a_star_endpoint"] == "sft-1e5-time"
    assert round(branches["B2"]["rl_val_needed_lower_bound"], 4) == 0.8039
    assert sorted(branches["B3"]["pool"]) == sorted(
        f"sft-{lr}-{m}" for lr in ("1e5", "5e5", "1e4") for m in ("exposure", "time"))
    assert branches["B5"]["match"] is True and branches["B0"]["outcome"] == "pass"
    assert round(branches["B6"]["rl400_val_mean_class_accuracy"], 4) == 0.6982
    evaluator = load_launch().load_evaluator({"evaluator": str(
        ROOT / "scripts" / "evaluate_freshpool_gates.py")})
    assert evaluator["d_req"](BRANCH["a_star"])["d_req"] == BRANCH["d_req"]


def test_evidence_base_matches_branch_and_design():
    base = json.loads((PRODUCTION / "evidence-base.json").read_text())
    assert base["a_star"] == BRANCH["a_star"] and base["b4_edge_extension"] is False
    assert len(base["labels"]) == 100 == len(set(base["labels"]))
    for split in ("h_select", "h_gate"):
        rows = base["rows"][split]
        assert len(rows) == 200 and len({r["id"] for r in rows}) == 200
        assert sorted({r["label"] for r in rows}) == sorted(base["labels"])
    assert not {r["id"] for r in base["rows"]["h_select"]} & {
        r["id"] for r in base["rows"]["h_gate"]}


# ---------------------------------------------------------------- plan


def test_plan_templates_pass_launch_plan_load_rules():
    launch = load_launch()
    launch.verify_stage_templates(PLAN["templates"])
    for name in launch.TEMPLATES:
        assert ":" in PLAN["templates"][name]["entry"]
        assert "resume_argv" not in PLAN["templates"][name]
    assert set(PLAN["forecast_seconds"]) == set(launch.FORECASTS)


def test_plan_is_mps_production_not_dry_run():
    assert "dry_run" not in PLAN
    assert PLAN["environment"]["ACCELERATE_TORCH_DEVICE"] == "mps"
    assert PLAN["environment"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert PLAN["environment"]["PYTHONPATH"].split(":")[0] == (
        "/Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex/src")
    assert PLAN["limits"]["minimum_free_bytes"] == 5 * 1024**3
    for template in PLAN["templates"].values():
        assert set(flag(template["argv"], "--device")) <= {"mps"}


STUDY_PROFILE = {"--response-format": "answer_only", "--dtype": "float32",
                 "--min-pixels": "50176", "--max-pixels": "200704",
                 "--max-completion-length": "256", "--num-generations": "4",
                 "--lora-rank": "8", "--temperature": "1.0"}


@pytest.mark.parametrize("name", ["e0", "assay", "p0", "rl150", "rl600", "sft_e", "sft_t",
                                  "egsft"])
def test_every_template_carries_the_study_fixed_profile(name):
    argv = PLAN["templates"][name]["argv"]
    for key, value in STUDY_PROFILE.items():
        assert flag(argv, key) == [value], (name, key)


def test_rl_and_sft_templates_carry_the_study_settings():
    t = PLAN["templates"]
    for name in ("rl150", "rl600"):
        argv = t[name]["argv"]
        assert flag(argv, "--beta") == ["0.04"] and flag(argv, "--max-steps") == ["600"]
        assert flag(argv, "--target-accepted") == ["600"]
        assert flag(argv, "--screen-threshold") == ["0.5"]
        assert flag(argv, "--train") == ["{manifest_s_scan_seed42}"]
        assert flag(argv, "--init-adapter") == ["{w_adapter}"]
        assert flag(argv, "--max-active-seconds") == ["{arm_cap}"]
    assert flag(t["rl150"]["argv"], "--stop-after-steps") == ["150"]
    assert flag(t["sft_e"]["argv"], "--stop-after-steps") == ["600"]
    assert flag(t["sft_e"]["argv"], "--train") == ["{accepted_manifest}"]
    assert flag(t["sft_t"]["argv"], "--train") == ["{scanned_manifest}"]
    assert flag(t["sft_t"]["argv"], "--stop-after-active-seconds") == ["{t_rl}"]
    assert flag(t["egsft"]["argv"], "--mode") == ["egsft"]
    assert flag(t["egsft"]["argv"], "--max-steps") == ["600"]
    e0 = t["e0"]["argv"]
    assert t["e0"]["entry"] == "e0_gate:main"
    assert flag(e0, "--beta") == ["0.04"] and flag(e0, "--learning-rate") == ["0.0001"]
    # The E0 entry pins the trainer it runs: the plan's pinned trainer source.
    assert flag(e0, "--trainer-sha256") == [
        PLAN["file_sha256"]["src/vision_rl/vlm_local_training.py"]]


def test_plan_pins_every_launcher_file_and_vision_rl_module():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "src/vision_rl"],
                             capture_output=True, text=True, check=True, timeout=60).stdout.split()
    for path in [*tracked, "scripts/evaluate_freshpool_gates.py",
                 *(f"scripts/freshpool_launch/{m}.py" for m in (
                     "launch", "phase_worker", "e0_gate", "trainer_fragment", "h_assay",
                     "p0_screen"))]:
        assert path in PLAN["file_sha256"], path
    receipts = PLAN["g0"]["receipts"]
    assert receipts["protocol_revision"]["path"] == (
        "docs/aircraft-rl-protocol-freshpool-v1-activated.json")
    assert receipts["branch"]["path"] == "docs/aircraft-freshpool-branch-receipt-v1.json"


def test_plan_sha256_file_names_the_plan():
    digest = start.sha(PRODUCTION / "plan.json")
    assert (PRODUCTION / "plan.sha256").read_text() == digest + "\n"


def test_forecasts_are_the_stated_measurements():
    f = PLAN["forecast_seconds"]
    assert f["rl"] == 13338.0 and f["sft_e"] == 3384.0 and f["egsft"] == 3966.0
    assert f["assay_h_select"] == 2513.0 and f["assay_h_gate"] == 4608.8
    assert (f["e0"], f["p0"]) == (7200.0, 3564.0)


# ---------------------------------------------------------------- start.py


def test_quiet_mode_refuses(tmp_path):
    quiet = tmp_path / "QUIET"
    start.check_quiet(quiet)
    quiet.write_text("")
    with pytest.raises(start.Refused) as caught:
        start.check_quiet(quiet)
    assert caught.value.reason == "quiet_mode_on"


def test_disk_requirement_refuses(tmp_path):
    start.check_disk(tmp_path, required=1)
    with pytest.raises(start.Refused) as caught:
        start.check_disk(tmp_path, required=10**18)
    assert caught.value.reason == "disk_below_reserve_plus_study_output"


def test_state_rules(tmp_path):
    state = tmp_path / "state"
    assert start.check_state(state, resume=False) is None
    with pytest.raises(start.Refused, match="no_state_to_resume"):
        start.check_state(state, resume=True)
    state.mkdir()
    with pytest.raises(start.Refused, match="state_directory_exists_single_use"):
        start.check_state(state, resume=False)
    (state / "state.json").write_text(json.dumps({"status": "stopped"}))
    with pytest.raises(start.Refused, match="state_is_terminal"):
        start.check_state(state, resume=True)
    (state / "state.json").write_text(json.dumps({"status": "running"}))
    (state / "lock").write_text("")  # created, pid not yet written: ask, never allow
    with pytest.raises(start.Refused, match="lock_unreadable"):
        start.check_state(state, resume=True)
    (state / "lock").write_text(str(os.getpid()))
    with pytest.raises(start.Refused, match="launcher_still_running"):
        start.check_state(state, resume=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (state / "lock").write_text(str(dead.pid))
    assert start.check_state(state, resume=True)["status"] == "running"


def test_images_link(tmp_path):
    manifests, target, other = tmp_path / "m", tmp_path / "images", tmp_path / "other"
    manifests.mkdir()
    with pytest.raises(start.Refused, match="development_images_missing"):
        start.images_link(manifests, target)
    target.mkdir()
    assert start.images_link(manifests, target) == "created"
    assert (manifests / "images").resolve() == target.resolve()
    assert start.images_link(manifests, target) == "present"
    other.mkdir()
    with pytest.raises(start.Refused, match="manifest_images_link_elsewhere"):
        start.images_link(manifests, other)
    (manifests / "images").unlink()
    (manifests / "images").mkdir()
    with pytest.raises(start.Refused, match="manifest_images_not_a_link"):
        start.images_link(manifests, target)


def git(directory, *args):
    subprocess.run(["git", "-C", str(directory), "-c", "user.name=t",
                    "-c", "user.email=t@example.invalid", *args],
                   check=True, capture_output=True, timeout=60)


def test_frozen_worktree_rules(tmp_path):
    repo, frozen = tmp_path / "repo", tmp_path / "frozen.noindex"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "a.txt").write_text("one")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "one")
    with pytest.raises(start.Refused, match="frozen_worktree_missing"):
        start.frozen_worktree(repo, frozen, resume=True)
    first = start.frozen_worktree(repo, frozen, resume=False)
    assert (frozen / "a.txt").read_text() == "one"
    assert start.frozen_worktree(repo, frozen, resume=False) == first  # reused, same commit
    (repo / "a.txt").write_text("two")
    git(repo, "commit", "-q", "-am", "two")
    with pytest.raises(start.Refused, match="frozen_worktree_at_another_commit"):
        start.frozen_worktree(repo, frozen, resume=False)
    # A resume keeps the frozen commit even though the checkout moved on.
    assert start.frozen_worktree(repo, frozen, resume=True) == first
    (frozen / "a.txt").write_text("edited in place")
    with pytest.raises(start.Refused, match="frozen_worktree_modified"):
        start.frozen_worktree(repo, frozen, resume=True)


def test_plan_identity_refuses_a_changed_plan(tmp_path):
    production = tmp_path / "scripts" / "freshpool_launch" / "production"
    production.mkdir(parents=True)
    (production / "plan.json").write_text("{}\n")
    (production / "plan.sha256").write_text(start.sha(production / "plan.json") + "\n")
    assert start.plan_identity(tmp_path)[1] == start.sha(production / "plan.json")
    (production / "plan.json").write_text("{ }\n")
    with pytest.raises(start.Refused, match="plan_differs_from_plan_sha256"):
        start.plan_identity(tmp_path)


def test_environment_is_the_plan_environment_plus_a_minimal_base(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"environment": {"ACCELERATE_TORCH_DEVICE": "mps", "X": "1"}}))
    env = start.environment(plan)
    assert env["ACCELERATE_TORCH_DEVICE"] == "mps" and env["X"] == "1"
    assert "/usr/bin" in env["PATH"].split(":") and env["HOME"] == str(Path.home())
    assert "PYTHONPATH" not in env  # only when the plan sets it


# ---------------------------------------------------------------- chain.py and status.py

FAKE_LAUNCHER = '''
import json, signal, sys, time
from pathlib import Path
state = Path(sys.argv[sys.argv.index("--plan") + 1]).parent / "state"
state.mkdir(exist_ok=True)
mode = Path(sys.argv[sys.argv.index("--plan") + 1]).read_text().strip()
def interrupted(*_):
    (state / "state.json").write_text(json.dumps({"status": "running", "next_action": "e0"}))
    sys.exit(3)
signal.signal(signal.SIGTERM, interrupted)
if mode == "finish":
    (state / "state.json").write_text(json.dumps({"status": "completed",
                                                  "next_action": "native-VAL package (G4)"}))
    sys.exit(0)
time.sleep(60)
'''


def run_chain(tmp_path, mode, *extra):
    launcher = tmp_path / "scripts" / "freshpool_launch" / "launch.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(FAKE_LAUNCHER)
    plan = tmp_path / "plan.json"
    plan.write_text(mode)
    record = tmp_path / "chain.json"
    command = [sys.executable, "-B", str(PRODUCTION / "chain.py"), "--launcher", str(launcher),
               "--plan", str(plan), "--expected-plan-sha256", "0" * 64, "--state-directory",
               str(tmp_path / "state"), "--chain-file", str(record), *extra]
    return command, record


def test_chain_records_a_finished_study(tmp_path):
    command, record = run_chain(tmp_path, "finish")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    finished = json.loads(record.read_text())
    assert (finished["status"], finished["launcher_exit_code"], finished["study_status"]) == (
        "finished", 0, "completed")


def test_chain_forwards_sigterm_as_a_resumable_interrupt(tmp_path):
    command, record = run_chain(tmp_path, "hang")
    process = subprocess.Popen(command, start_new_session=True)
    deadline = time.monotonic() + 30
    while not record.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(1)
    os.kill(process.pid, signal.SIGTERM)
    assert process.wait(timeout=60) == 3
    finished = json.loads(record.read_text())
    assert (finished["launcher_exit_code"], finished["study_status"]) == (3, "running")
    assert finished["signals_forwarded"] == [signal.SIGTERM]


def test_chain_bounds_the_run(tmp_path):
    command, record = run_chain(tmp_path, "hang", "--max-seconds", "1")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 3, result.stderr
    assert json.loads(record.read_text())["max_seconds_reached"] is True


IGNORES_TERM = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(120)
"""


def test_chain_kills_a_launcher_that_ignores_the_bound(tmp_path):
    command, record = run_chain(tmp_path, "hang", "--max-seconds", "1",
                                "--kill-after-seconds", "2")
    (tmp_path / "scripts" / "freshpool_launch" / "launch.py").write_text(IGNORES_TERM)
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == -signal.SIGKILL, result.stderr
    assert time.monotonic() - started < 30
    finished = json.loads(record.read_text())
    assert finished["max_seconds_reached"] is True and finished["killed_after_bound"] is True


def test_status_reads_a_running_stage(tmp_path):
    state = tmp_path / "state"
    stage_log = state / "stages" / "p0" / "instance-1" / "stage.log"
    stage_log.parent.mkdir(parents=True)
    stage_log.write_text("line one\nline two\n")
    now = time.time()
    journal = [{"event": "launch_started", "time": now - 100},
               {"event": "stage_started", "stage": "e0", "arm": "e0", "instance": 1,
                "time": now - 90, "cap_seconds": 9000.0, "inherited_seconds": 0.0,
                "void_seconds": 0.0},
               {"event": "stage_finished", "stage": "e0", "time": now - 50},
               {"event": "stage_started", "stage": "p0", "arm": "p0", "instance": 1,
                "time": now - 40, "cap_seconds": 4455.0, "inherited_seconds": 0.0,
                "void_seconds": 0.0}]
    (state / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in journal))
    (state / "state.json").write_text(json.dumps({"status": "running", "next_action": "p0",
                                                  "completed": {"e0": {}}, "gates": {}}))
    (state / "lock").write_text(str(os.getpid()))
    report = status.summary(state, events=3, log_lines=1)
    assert report["status"] == "running" and report["launcher_alive"] is True
    running = report["running_stage"]
    assert running["stage"] == "p0" and running["cap_seconds"] == 4455.0
    assert running["stage_log_tail"] == ["line two"]
    assert 39 <= running["elapsed_wall_seconds"] < 120
    assert [e["event"] for e in report["last_events"]] == [
        "stage_started", "stage_finished", "stage_started"]
    assert status.summary(tmp_path / "absent")["status"] == "no state"
