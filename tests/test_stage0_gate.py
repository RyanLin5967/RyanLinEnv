"""Fixture tests for scripts/freshpool_launch/stage0_gate.py: the signal-limit Stage 0 entry.

No model runs. Every trainer invocation goes through the real phase_worker.py in its own
process, with a fake torch (MPS guard path) and a fake beta-0 dr_grpo trainer that writes the
files train_local writes and publishes a real endpoint receipt; the entry reads them back
through the real vision_rl.vlm_local_training verifiers. Expected values are written out by
hand from the fixture design below, never read back from the entry:

  scan        16 rows s00..s15; scan position n is accepted unless n % 3 == 2, so the
              accepted IDs are s00 s01 s03 s04 s06 s07 s09 s10 ...
  seconds     committed active seconds after step k are k * 1.25 * G (G8: 10 per update,
              G16: 20), so c(G8) = 10.0 and c(G16) = 20.0 s per accepted update
  rewards     G rewards per group, 1 at both ends and 0 between (a mixed group)
  tokens      every completion is 2 tokens: G8 E0 generates 4 * 8 * 2 = 64, G16 128
  parity      0.0004 at G8, 0.0007 at G16, so the reported maximum is 0.0007
"""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "freshpool_launch"
sys.path.insert(0, str(PACKAGE))

stage0_gate = importlib.import_module("stage0_gate")
signal_limit = importlib.import_module("signal_limit_launch")

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")

FAKE_TORCH = '''
class _MPS:
    def is_available(self):
        return True

    def recommended_max_memory(self):
        return 48 * 1024**3

    def set_per_process_memory_fraction(self, fraction):
        pass

    def driver_allocated_memory(self):
        return 3 * 1024**3

    def current_allocated_memory(self):
        return 2 * 1024**3


mps = _MPS()


class _Backends:
    pass


backends = _Backends()
backends.mps = mps


def set_num_threads(count):
    pass
'''

# A beta-0 dr_grpo trainer stand-in. SCENARIO keys apply at every G unless "at_group" names
# one G: diverge (a resumed process samples other tokens), drop_loss_type (the record lacks
# loss_type), kl_logged (a kl metric in log_history), ref_adapter (a ref adapter directory in
# every checkpoint), reference_record (a KL reference in the record and receipt),
# parity_reference (the parity diagnostic ran a reference forward), scorer_drift.
FAKE_TRAINER = r'''
import hashlib, json, os, sys
from pathlib import Path

from vision_rl import vlm_local_training as vlt

SCENARIO = json.loads(os.environ.get("FAKE_STAGE0", "{}"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv):
    value = {f: argv[argv.index(f) + 1] for f in (
        "--output", "--resume-from-checkpoint", "--stop-after-steps", "--max-steps",
        "--train", "--init-adapter", "--num-generations", "--save-steps", "--beta",
        "--loss-type", "--scale-rewards") if f in argv}
    group = int(value["--num-generations"])
    on = SCENARIO.get("at_group") in (None, group)
    def fire(key):
        return SCENARIO.get(key) if on else None
    output = Path(value["--output"])
    steps = int(value["--max-steps"])
    stop = int(value.get("--stop-after-steps", steps))
    save = int(value["--save-steps"])
    w_sha = sha(Path(value["--init-adapter"]) / "adapter_model.safetensors")
    scan = [json.loads(l)["id"] for l in Path(value["--train"]).read_text().splitlines()]
    accepted = [i for n, i in enumerate(scan) if n % 3 != 2]
    start, previous = 0, None
    resumed = "--resume-from-checkpoint" in value
    if resumed:
        checkpoint = Path(value["--resume-from-checkpoint"])
        start = int(checkpoint.name.split("-")[1])
        previous = json.loads((output / "local_training.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    settings = {"mode": "grpo", "max_steps": steps, "max_active_seconds": None,
                "beta": float(value["--beta"]), "init_adapter": {"weights_sha256": w_sha},
                "screening": {"group_size": group, "target_accepted": steps}}
    if not fire("drop_loss_type"):
        settings["loss_type"] = value["--loss-type"]
    settings["scale_rewards"] = value["--scale-rewards"]
    reference = ({"kind": "initial_saved_adapter", "initial_adapter_weights_sha256": w_sha,
                  "adapter": {"sha256": "identity-of-W", "trainable_tensors": 0}}
                 if fire("reference_record") else None)
    record = {
        "status": "running", "stage_id": "stage0-fixture",
        "control_identity": {"fixture": True, "profile": {"temperature": 1.0}},
        "settings": settings,
        "environment": {"adapter_source_sha256": None},
        "reference": reference,
        "screen_sampler_transform": {"temperature": 1.0, "suppress_tokens": [151655],
                                     "eos_token_ids": [151645]},
        "requested_boundary": {"stop_after_steps": stop if stop < steps else None,
                               "stop_after_active_seconds": None},
    }
    record["sampler_teacher_logprob_parity"] = (previous or {}).get(
        "sampler_teacher_logprob_parity") or {
        "max_abs_logprob_error_after_same_suppression": 0.0004 if group == 8 else 0.0007,
        "sampling_suppressed_token_ids": [151655],
        "initial_reference_max_abs_logprob_difference":
            0.01 if fire("parity_reference") else None,
        "initial_reference_sampled_kl": 0.001 if fire("parity_reference") else None}
    (output / "local_training.json").write_text(json.dumps(record))
    entries = []
    history = []
    if resumed:
        entries = json.loads((checkpoint / "exposure.json").read_text())["entries"]
        history = json.loads((checkpoint / "trainer_state.json").read_text())["log_history"]
    for step in range(start + 1, stop + 1):
        seconds = step * 1.25 * group
        rewards = [1.0] + [0.0] * (group - 2) + [1.0]
        ids = [[5, 2]] + [[9, 2]] * (group - 2) + [[5, 2]]
        if resumed and fire("diverge"):
            ids = [[5, 2], [8, 2]] + [[9, 2]] * (group - 3) + [[5, 2]]
        if fire("short_rewards"):  # G - 1 rewards: a group not at this G
            rewards, ids = rewards[:-1], ids[:-1]
        row = {"step": step, "completed_state_active_seconds": seconds,
               "latest_gradient_norm": 0.75}
        if fire("audit_row_reference"):
            row["reference_sha256"] = "identity-of-W"
        with (output / "optimizer_audit.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        with (output / "reward_groups.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "id": accepted[step - 1], "rewards": rewards, "completion_ids": ids,
                "recognition_mixed": True, "completion_tokens": [2] * group}) + "\n")
        entries.append({"invocation_id": "i", "global_step": step, "id": accepted[step - 1]})
        log = {"step": step, "loss": 0.125, "reward": 0.25}
        if fire("kl_logged"):
            log["kl"] = 0.0
        if fire("drop_loss_at_step") == step:
            del log["loss"]
        history.append(log)
        if step % save and step != stop:
            continue
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        if fire("ref_adapter"):
            (checkpoint / "ref").mkdir()
            (checkpoint / "ref" / "adapter_model.safetensors").write_bytes(b"ref")
        for name, content in (("trainer_state.json", json.dumps({
                                   "global_step": step, "log_history": history})),
                              ("optimizer.pt", "opt"), ("scheduler.pt", "sched"),
                              ("rng_state.pth", "rng"), ("adapter_config.json", "{}")):
            (checkpoint / name).write_text(content)
        (checkpoint / "adapter_model.safetensors").write_bytes(f"weights {step}".encode())
        (checkpoint / "exposure.json").write_text(json.dumps({"schema_version": 1,
                                                              "entries": entries}))
        last = scan.index(accepted[step - 1])
        screen = [{"scan_position": n, "id": scan[n], "p": 0.5 if n % 3 != 2 else 0.99,
                   "u": 0.875 if n % 3 != 2 else 0.0394, "accepted": n % 3 != 2,
                   "adapter_sha256": "a", "scorer_seconds": 0.25, "global_step": 1 + len([
                       i for i in scan[:n] if i in accepted])} for n in range(last + 1)]
        (checkpoint / "screen.json").write_text(json.dumps({"entries": screen}))
        (checkpoint / "active_time.json").write_text(json.dumps({
            "schema_version": 1, "scope": vlt.ACTIVE_TIME_SCOPE, "device": "mps",
            "global_step": step, "total_budget_seconds": None,
            "committed_active_seconds": seconds,
            "checkpoint_file_sha256": {n: sha(checkpoint / n)
                                       for n in vlt.ACTIVE_CHECKPOINT_FILES},
            "exposure_sha256": sha(checkpoint / "exposure.json"),
            "screen_sha256": sha(checkpoint / "screen.json")}))
    adapter = output / "final_adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(f"weights {stop}".encode())
    (adapter / "adapter_config.json").write_text("{}")
    (output / "adapter-source.py").write_text("# fixture\n")
    record.update(status="completed", final_adapter_sha256=sha(
        adapter / "adapter_model.safetensors"),
        optimizer_audit={"active_time": {
            "fixture": True, "observed_scorer_seconds_within_active":
                0.25 * len(screen) + (SCENARIO.get("scorer_drift", 0.0) if on else 0.0)},
                         "changed_tensors": 2,
                         **({"reference_unchanged": True}
                            if fire("audit_reference_keys") else {})})
    record["environment"]["adapter_source_sha256"] = sha(output / "adapter-source.py")
    (output / "local_training.json").write_text(json.dumps(record))
    vlt._publish_endpoint_receipt(output, record, stop)
    return 0
'''

# Written out, not read from the entry: the launcher's five plus e0_gate's beta-0 checks.
CHECK_NAMES = (
    "resume_2_plus_2_identical", "resume_1_plus_3_identical", "finite_nonzero_task_gradient",
    "no_reference_forward_at_beta_0", "loss_type_dr_grpo_recorded",
    "scan_follows_train_manifest", "initial_adapter_is_w", "useful_mixed_group",
    "lora_parameter_changed", "scorer_uses_sampler_transform",
    "active_time_ledger_cumulative", "scorer_seconds_within_active_time", "group_size_is_g",
)
TRUE = dict.fromkeys(CHECK_NAMES, True)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


class Stage0Fixture:
    def __init__(self, tmp, monkeypatch, scenario=None):
        pkg = tmp / "pkg"
        pkg.mkdir()
        (pkg / "torch.py").write_text(FAKE_TORCH)
        (pkg / "fake_stage0_trainer.py").write_text(FAKE_TRAINER)
        monkeypatch.setattr(stage0_gate, "ENTRY", "fake_stage0_trainer:main")
        monkeypatch.delenv("VISION_RL_TORCH_THREADS", raising=False)
        for name, value in {"PYTHONPATH": f"{pkg}{os.pathsep}{ROOT / 'src'}",
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_HUB_OFFLINE": "1",
                            "ACCELERATE_TORCH_DEVICE": "mps",
                            "FAKE_STAGE0": json.dumps(scenario or {})}.items():
            monkeypatch.setenv(name, value)
        self.w = tmp / "w"
        self.w.mkdir()
        (self.w / "adapter_model.safetensors").write_bytes(b"W weights")
        self.scan = tmp / "train400.jsonl"
        self.scan_ids = [f"s{i:02d}" for i in range(16)]
        self.scan.write_text("".join(json.dumps({"id": i}) + "\n" for i in self.scan_ids))
        self.stage = tmp / "stage"
        self.stage.mkdir()
        self.fragment = self.stage / "fragment.json"
        self.trainer_sha = sha_bytes((pkg / "fake_stage0_trainer.py").read_bytes())

    def trainer(self, drop=()):
        argv = ["--mode", "grpo", "--train", str(self.scan), "--init-adapter", str(self.w),
                "--device", "mps", "--beta", "0", "--loss-type", "dr_grpo",
                "--scale-rewards", "none", "--train-only"]
        for flag in drop:
            at = argv.index(flag)
            argv = argv[:at] + argv[at + (1 if flag == "--train-only" else 2):]
        return argv

    def argv(self, options=(), trainer=None):
        return ["--fragment", str(self.fragment), "--stage-dir", str(self.stage),
                "--trainer-sha256", self.trainer_sha, *options, "--",
                *(self.trainer() if trainer is None else trainer)]

    def run(self, options=()):
        assert stage0_gate.main(self.argv(options)) == 0
        return json.loads(self.fragment.read_text())


def test_study_defaults_measure_c_and_prove_e0_at_g8_and_g16(tmp_path, monkeypatch):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    fragment = fixture.run()
    assert fragment["study_parameters"] is True
    assert fragment["e0_groups"] == [8, 16]
    assert fragment["e0_steps"] == 4
    assert fragment["probe_accepted"] == 8
    assert fragment["e0_splits"] == {"uninterrupted": [4], "resume_2_plus_2": [2, 2],
                                     "resume_1_plus_3": [1, 3]}
    assert fragment["checks"] == {"8": TRUE, "16": TRUE}
    assert fragment["c_accepted_steps"] == {"8": 8, "16": 8}
    assert fragment["c_seconds_per_accepted_update"] == {"8": 10.0, "16": 20.0}
    assert fragment["sampler_teacher_max_abs_logprob_difference"] == 0.0007
    g8, g16 = fragment["measurements"]["8"], fragment["measurements"]["16"]
    assert [i["label"] for i in g8["invocations"]] == [
        "g8-probe", "g8-uninterrupted-part1", "g8-resume_2_plus_2-part1",
        "g8-resume_2_plus_2-part2", "g8-resume_1_plus_3-part1", "g8-resume_1_plus_3-part2"]
    assert [i["exit_code"] for i in g8["invocations"] + g16["invocations"]] == [0] * 12
    assert {i["execution"] for i in g8["invocations"]} == {"mps"}
    assert g8["invocations"][0]["peak_mps_driver_allocated_bytes"] == 3 * 1024**3
    assert g8["probe_committed_active_seconds"] == 80.0
    assert g16["probe_committed_active_seconds"] == 160.0
    # probe at 8 accepted: the 8th accepted ID s10 is scan position 10, so 11 scored rows.
    assert g8["probe_committed_scorer_seconds"] == 11 * 0.25
    assert g8["update_seconds"] == [10.0, 10.0, 10.0, 10.0]
    assert g16["update_seconds"] == [20.0, 20.0, 20.0, 20.0]
    assert g8["generated_tokens"] == 64 and g16["generated_tokens"] == 128
    # the probe saves only at its last accepted update
    assert sorted(g8["probe_checkpoint_bytes"]) == ["checkpoint-8"]
    # the launcher's own validator accepts the real fragment
    assert signal_limit.SignalLimitLauncher.stage0_validator(fragment, None) is None


@pytest.mark.parametrize("scenario, failed", [
    # A resumed process that samples different tokens is not identical.
    ({"diverge": True}, {"resume_2_plus_2_identical", "resume_1_plus_3_identical"}),
    # The trainer did not record loss_type dr_grpo.
    ({"drop_loss_type": True}, {"loss_type_dr_grpo_recorded"}),
    # Each observable of a reference forward fires the beta-0 check on its own.
    ({"kl_logged": True}, {"no_reference_forward_at_beta_0"}),
    ({"ref_adapter": True}, {"no_reference_forward_at_beta_0"}),
    ({"reference_record": True}, {"no_reference_forward_at_beta_0"}),
    ({"parity_reference": True}, {"no_reference_forward_at_beta_0"}),
    ({"audit_reference_keys": True}, {"no_reference_forward_at_beta_0"}),
    ({"audit_row_reference": True}, {"no_reference_forward_at_beta_0"}),
    # A log_history without a logged loss at step 2 does not prove the kl metric's absence.
    ({"drop_loss_at_step": 2}, {"no_reference_forward_at_beta_0"}),
    ({"short_rewards": True}, {"group_size_is_g"}),
    ({"scorer_drift": 0.5}, {"scorer_seconds_within_active_time"}),
])
def test_each_forced_fire_is_written_false_at_its_own_g(tmp_path, monkeypatch, scenario,
                                                        failed):
    fixture = Stage0Fixture(tmp_path, monkeypatch, {**scenario, "at_group": 16})
    fragment = fixture.run()
    assert fragment["checks"]["8"] == TRUE
    assert {k for k, v in fragment["checks"]["16"].items() if v is not True} == failed
    assert signal_limit.SignalLimitLauncher.stage0_validator(fragment, None) == "e0_failed"


def test_scorer_outside_active_time_withholds_c(tmp_path, monkeypatch):
    fixture = Stage0Fixture(tmp_path, monkeypatch, {"scorer_drift": 0.5, "at_group": 8})
    fragment = fixture.run()
    assert fragment["c_seconds_per_accepted_update"] == {"8": None, "16": 20.0}
    assert fragment["c_accepted_steps"] == {"8": 8, "16": 8}
    assert "scorer" in fragment["detail"]["8"]["c_refused"]


def test_smoke_sizes_are_recorded_and_refused_by_the_launcher(tmp_path, monkeypatch):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    fragment = fixture.run(["--groups", "8", "--e0-steps", "2", "--probe-accepted", "3"])
    assert fragment["study_parameters"] is False
    assert (fragment["e0_groups"], fragment["e0_steps"], fragment["probe_accepted"]) == (
        [8], 2, 3)
    assert fragment["e0_splits"] == {"uninterrupted": [2], "resume_2_plus_2": [1, 1],
                                     "resume_1_plus_3": [1, 1]}
    assert fragment["checks"] == {"8": TRUE}
    assert fragment["c_accepted_steps"] == {"8": 3}
    assert fragment["c_seconds_per_accepted_update"] == {"8": 10.0}  # 30 s / 3
    assert (signal_limit.SignalLimitLauncher.stage0_validator(fragment, None)
            == "stage0_not_at_study_e0_steps_or_probe_size")


def test_evaluate_refuses_true_without_evidence(tmp_path, monkeypatch):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    fixture.run(["--groups", "8", "--e0-steps", "2", "--probe-accepted", "2"])
    root = fixture.stage / "g8"
    runs = {name: stage0_gate.E.read_run(root / name, name)
            for name in ("uninterrupted", "resume_2_plus_2", "resume_1_plus_3")}
    probe = stage0_gate.E.read_run(root / "probe", "probe")
    w = sha_bytes(b"W weights")
    checks, _ = stage0_gate.evaluate(runs, probe, 8, 2, w, fixture.scan_ids)
    assert checks == TRUE
    # The record no longer says reference None (the key is absent): not proven.
    del runs["uninterrupted"]["record"]["reference"]
    checks, _ = stage0_gate.evaluate(runs, probe, 8, 2, w, fixture.scan_ids)
    assert checks["no_reference_forward_at_beta_0"] is False
    # A malformed run makes every check false rather than raising.
    checks, detail = stage0_gate.evaluate({}, probe, 8, 2, w, fixture.scan_ids)
    assert checks == dict.fromkeys(CHECK_NAMES, False) and "error" in detail
    # A run at another G cannot be keyed under this one.
    checks, _ = stage0_gate.evaluate(runs, probe, 16, 2, w, fixture.scan_ids)
    assert checks["group_size_is_g"] is False
    # A different W is not the init.
    checks, _ = stage0_gate.evaluate(runs, probe, 8, 2, sha_bytes(b"other"), fixture.scan_ids)
    assert checks["initial_adapter_is_w"] is False


def read_smoke(fixture):
    root = fixture.stage / "g8"
    runs = {name: stage0_gate.E.read_run(root / name, name)
            for name in ("uninterrupted", "resume_2_plus_2", "resume_1_plus_3")}
    return runs, stage0_gate.E.read_run(root / "probe", "probe")


def failed_after(change, fixture):
    runs, probe = read_smoke(fixture)
    change(runs, probe)
    checks, _ = stage0_gate.evaluate(runs, probe, 8, 2, sha_bytes(b"W weights"),
                                     fixture.scan_ids)
    return {k for k, v in checks.items() if v is not True}


def set_receipt_reference(runs, probe):
    runs["resume_1_plus_3"]["receipt"]["reference_sha256"] = "identity-of-W"


def decreasing_ledger(runs, probe):
    runs["resume_2_plus_2"]["audit_rows"][1]["completed_state_active_seconds"] = 5.0


def ledger_end_not_receipt(runs, probe):
    probe["receipt"]["committed_active_seconds"] = 21.0


@pytest.mark.parametrize("change, failed", [
    (set_receipt_reference, {"no_reference_forward_at_beta_0"}),
    (decreasing_ledger, {"active_time_ledger_cumulative"}),
    # 21 s != the probe's last ledger value 20 s; the scorer check still holds (0.5 <= 21).
    (ledger_end_not_receipt, {"active_time_ledger_cumulative"}),
])
def test_evaluate_fires_on_one_field(tmp_path, monkeypatch, change, failed):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    fixture.run(["--groups", "8", "--e0-steps", "2", "--probe-accepted", "2"])
    assert failed_after(lambda runs, probe: None, fixture) == set()
    assert failed_after(change, fixture) == failed


def test_cost_requires_every_condition(tmp_path, monkeypatch):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    fixture.run(["--groups", "8", "--e0-steps", "2", "--probe-accepted", "2"])
    _, probe = read_smoke(fixture)
    # probe of 2 at G8: 2 * 10 s committed over 2 accepted updates
    assert stage0_gate.cost(probe, 2) == (10.0, 2, None)
    assert stage0_gate.cost(probe, 3)[0] is None  # target 3, 2 accepted
    shorter = dict(probe, exposure=probe["exposure"][:1])
    assert stage0_gate.cost(shorter, 2)[0] is None
    other_step = dict(probe, global_step=3)
    assert stage0_gate.cost(other_step, 2)[0] is None
    zero = dict(probe, receipt=dict(probe["receipt"], committed_active_seconds=0.0))
    assert stage0_gate.cost(zero, 2)[:2] == (None, 2)
    assert "seconds" in stage0_gate.cost(zero, 2)[2]


@pytest.mark.parametrize("change, message", [
    (lambda a: a + ["--num-generations", "8"], "owns"),
    (lambda a: a + ["--save-steps", "1"], "owns"),
    (lambda a: a + ["--output=/tmp/x"], "owns"),
    (lambda a: [x if x != "0" else "0.04" for x in a], "beta 0"),
    (lambda a: [x if x != "dr_grpo" else "grpo" for x in a], "dr_grpo"),
    (lambda a: [x if x != "none" else "group" for x in a], "dr_grpo"),
])
def test_wrong_or_owned_trainer_flags_are_refused(tmp_path, monkeypatch, change, message):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match=message):
        stage0_gate.main(fixture.argv(trainer=change(fixture.trainer())))
    assert not (fixture.stage / "g8").exists()


@pytest.mark.parametrize("flag", ["--loss-type", "--beta", "--scale-rewards", "--train",
                                  "--init-adapter", "--device"])
def test_missing_required_trainer_flag_is_refused(tmp_path, monkeypatch, flag):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match=flag):
        stage0_gate.main(fixture.argv(trainer=fixture.trainer(drop=[flag])))
    assert not (fixture.stage / "g8").exists()


def test_missing_train_only_is_refused(tmp_path, monkeypatch):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="TRAIN-only"):
        stage0_gate.main(fixture.argv(trainer=fixture.trainer(drop=["--train-only"])))


def test_a_failed_trainer_part_raises_and_writes_no_fragment(tmp_path, monkeypatch):
    fixture = Stage0Fixture(tmp_path, monkeypatch)
    (tmp_path / "pkg" / "fake_stage0_trainer.py").write_text(
        FAKE_TRAINER + "\n_main = main\n"
        "def main(argv):\n    return 5 if '--resume-from-checkpoint' in argv else _main(argv)\n")
    fixture.trainer_sha = sha_bytes((tmp_path / "pkg" / "fake_stage0_trainer.py").read_bytes())
    with pytest.raises(stage0_gate.E.E0Error):
        stage0_gate.main(fixture.argv(["--groups", "8", "--e0-steps", "2",
                                       "--probe-accepted", "2"]))
    assert not fixture.fragment.exists()


@pytest.mark.parametrize("environ, flags", [
    ({"ACCELERATE_TORCH_DEVICE": "mps"}, []),
    ({"ACCELERATE_TORCH_DEVICE": "cpu"}, ["--dry-run"]),
    ({"ACCELERATE_TORCH_DEVICE": "cpu", "VISION_RL_TORCH_THREADS": "6"}, ["--remote-cpu"]),
    ({"ACCELERATE_TORCH_DEVICE": "cuda", "VISION_RL_TORCH_THREADS": "6"}, ["--remote-cuda"]),
    ({"ACCELERATE_TORCH_DEVICE": "cuda"}, ["--remote-cuda"]),
])
def test_child_flag_follows_the_parent_execution_path(environ, flags):
    assert stage0_gate.child_execution_flags(environ) == flags


def test_remote_cpu_parent_forwards_remote_cpu_to_every_child(tmp_path, monkeypatch):
    commands = []

    class Completed:
        returncode = 0

    def fake_run(command, timeout):
        commands.append(command)
        return Completed()

    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cpu")
    monkeypatch.setenv("VISION_RL_TORCH_THREADS", "6")
    monkeypatch.setattr(stage0_gate.subprocess, "run", fake_run)
    runner = stage0_gate.Runner(tmp_path, "0" * 64)
    assert runner("g8-probe", ["--device", "cpu"], 8, 8) == 0
    worker, trainer = commands[0][:commands[0].index("--")], commands[0][
        commands[0].index("--") + 1:]
    assert "--remote-cpu" in worker and "--dry-run" not in worker
    assert trainer == ["--device", "cpu"]
    invocation = runner.invocations[0]
    assert invocation["execution"] == "remote-cpu"
    assert invocation["timeout_seconds"] == 14400.0  # 3600 * (8 * 8 / 16)
    # off MPS the worker's zero-initialised MPS peaks are not reported as measurements
    assert invocation["peak_mps_driver_allocated_bytes"] is None
    assert invocation["peak_mps_tensor_allocated_bytes"] is None


@pytest.mark.parametrize("group, updates, seconds", [
    (4, 4, 3600.0),     # e0_gate's own size: unchanged
    (2, 1, 3600.0),     # never below it
    (8, 4, 7200.0),     # 32 / 16 = 2
    (16, 8, 28800.0),   # the G16 probe: 128 / 16 = 8
])
def test_child_wall_scales_with_group_and_updates(group, updates, seconds):
    assert stage0_gate.child_timeout_seconds(group, updates) == seconds


def test_a_child_past_its_wall_fails_the_stage_naming_it(tmp_path, monkeypatch):
    def too_slow(command, timeout):
        raise stage0_gate.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "mps")
    monkeypatch.delenv("VISION_RL_TORCH_THREADS", raising=False)
    monkeypatch.setattr(stage0_gate.subprocess, "run", too_slow)
    runner = stage0_gate.Runner(tmp_path, "0" * 64)
    with pytest.raises(stage0_gate.E.E0Error, match="g16-probe exceeded its 28800 s"):
        runner("g16-probe", ["--device", "mps"], 16, 8)


NOOP_ENTRY = "def main(argv):\n    return 0\n"
FAKE_CUDA = '''

class _Cuda:
    def is_available(self):
        return True

    def device_count(self):
        return 1

    def get_device_name(self, index):
        return "fake"

    def max_memory_allocated(self):
        return 5


cuda = _Cuda()
'''


@pytest.mark.parametrize("environ", [
    {"ACCELERATE_TORCH_DEVICE": "mps"},
    {"ACCELERATE_TORCH_DEVICE": "cpu"},
    {"ACCELERATE_TORCH_DEVICE": "cpu", "VISION_RL_TORCH_THREADS": "6", "UWRUN_THREADS": "6"},
    {"ACCELERATE_TORCH_DEVICE": "cuda", "VISION_RL_TORCH_THREADS": "6", "UWRUN_THREADS": "6",
     "CUDA_VISIBLE_DEVICES": "0"},
])
def test_every_child_flag_is_accepted_by_the_real_phase_worker(tmp_path, monkeypatch, environ):
    """Each flag child_execution_flags returns, through the real phase_worker.py, in the
    environment that selects it: a flag the worker does not define exits non-zero."""
    import subprocess

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "torch.py").write_text(FAKE_TORCH + FAKE_CUDA)
    (pkg / "noop_stage0_entry.py").write_text(NOOP_ENTRY)
    env = {k: v for k, v in os.environ.items()
           if k not in ("VISION_RL_TORCH_THREADS", "UWRUN_THREADS", "CUDA_VISIBLE_DEVICES")}
    env.update(PYTHONPATH=str(pkg), PYTORCH_ENABLE_MPS_FALLBACK="0", HF_HUB_OFFLINE="1",
               **environ)
    flags = stage0_gate.child_execution_flags(env)
    command = [sys.executable, "-B", str(PACKAGE / "phase_worker.py"),
               "--guard-dir", str(tmp_path / "guards"), "--stage", "flag-check",
               "--kind", "train", "--entry", "noop_stage0_entry:main",
               "--entry-sha256", sha_bytes(NOOP_ENTRY.encode()), *flags, "--",
               "--device", environ["ACCELERATE_TORCH_DEVICE"]]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    # Forced fire: the same command with a flag the worker does not define is refused.
    bogus = [part if part not in ("--remote-cpu", "--remote-cuda", "--dry-run")
             else "--remote-bogus" for part in command]
    if flags:
        refused = subprocess.run(bogus, env=env, capture_output=True, text=True, timeout=120)
        assert refused.returncode != 0 and "--remote-bogus" in refused.stderr


def test_entry_checks_cover_the_launchers_stage0_checks():
    assert stage0_gate.CHECKS == CHECK_NAMES
    # The launcher requires every check the entry computes (PR #34 review N6), not the
    # study's five named ones only: the hand-written list above is the specification.
    assert signal_limit.STAGE0_CHECKS == CHECK_NAMES
    assert signal_limit.STAGE0_E0_STEPS == stage0_gate.STUDY_E0_STEPS == 4
    assert signal_limit.STAGE0_MIN_C_STEPS == stage0_gate.STUDY_PROBE_ACCEPTED == 8
