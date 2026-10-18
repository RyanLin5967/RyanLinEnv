"""Fixture tests for scripts/freshpool_launch/e0_gate.py: the production E0 entry.

No model runs. Each trainer invocation goes through the real phase_worker.py in its own
process, as in a launch, with a fake torch (MPS guard path) and a fake trainer that writes the
files train_local writes and publishes a real endpoint receipt; the entry reads them back
through the real vision_rl.vlm_local_training verifiers. Expected values are written out from
the fixture design, never read back from the entry.
"""

import hashlib
import importlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "freshpool_launch"
sys.path.insert(0, str(PACKAGE))

e0_gate = importlib.import_module("e0_gate")

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX processes")

ANCHOR = "reference adapter differs from the pinned initial anchor"

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

# A trainer stand-in: writes what train_local writes for a screened GRPO run (per-step
# audit and reward rows, one checkpoint per step, exposure/screen/active-time markers, a
# KL-reference adapter), restores from --resume-from-checkpoint, and refuses a checkpoint
# whose reference adapter bytes differ from the run's (the trainer's anchor check).
FAKE_TRAINER = r'''
import hashlib, json, os, sys
from pathlib import Path

from vision_rl import vlm_local_training as vlt

SCENARIO = json.loads(os.environ.get("FAKE_E0", "{}"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safetensor(values):
    data = b"".join(__import__("struct").pack("<f", v) for v in values)
    header = json.dumps({"base_model.a.lora_A.weight": {
        "dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}).encode()
    return len(header).to_bytes(8, "little") + header + data


def main(argv):
    value = {f: argv[argv.index(f) + 1] for f in (
        "--output", "--resume-from-checkpoint", "--stop-after-steps", "--max-steps",
        "--train", "--init-adapter", "--screen-journal") if f in argv}
    output = Path(value["--output"])
    steps = int(value["--max-steps"])
    stop = int(value.get("--stop-after-steps", steps))
    w_sha = sha(Path(value["--init-adapter"]) / "adapter_model.safetensors")
    scan = [json.loads(l)["id"] for l in Path(value["--train"]).read_text().splitlines()]
    accepted = [i for n, i in enumerate(scan) if n % 3 != 2]
    reference = safetensor([0.25, -0.5, 0.125])
    start, previous = 0, None
    resumed = "--resume-from-checkpoint" in value
    if resumed:
        checkpoint = Path(value["--resume-from-checkpoint"])
        start = int(checkpoint.name.split("-")[1])
        previous = json.loads((output / "local_training.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "running", "stage_id": "e0-fixture",
        "control_identity": {"fixture": True, "profile": {"temperature": 1.0}},
        "settings": {"mode": "grpo", "max_steps": steps, "max_active_seconds": None,
                     "init_adapter": {"weights_sha256": w_sha}},
        "environment": {"adapter_source_sha256": None},
        "reference": {"kind": "initial_saved_adapter", "initial_adapter_weights_sha256": w_sha,
                      "adapter": {"sha256": "identity-of-W", "trainable_tensors": 0}},
        "screen_sampler_transform": {"temperature": 1.0, "suppress_tokens": [151655],
                                     "eos_token_ids": [151645]},
        "requested_boundary": {"stop_after_steps": stop if stop < steps else None,
                               "stop_after_active_seconds": None},
    }
    record["sampler_teacher_logprob_parity"] = (previous or {}).get(
        "sampler_teacher_logprob_parity") or {
        "max_abs_logprob_error_after_same_suppression": SCENARIO.get("parity", 0.0004),
        "max_abs_raw_teacher_vs_sampler_logprob_error": 0.9,
        "sampling_suppressed_token_ids": [151655]}
    (output / "local_training.json").write_text(json.dumps(record))
    if resumed and sha(checkpoint / "ref" / "adapter_model.safetensors") != hashlib.sha256(
            reference).hexdigest() and not SCENARIO.get("ignore_tamper"):
        record.update(status="failed", error_type="LocalVLMError",
                      error="reference adapter differs from the pinned initial anchor")
        (output / "local_training.json").write_text(json.dumps(record))
        print("local VLM: reference adapter differs from the pinned initial anchor",
              file=sys.stderr)
        return 2
    if SCENARIO.get("fail_stop") == stop and resumed == SCENARIO.get("fail_resumed", False):
        return 5
    entries = []
    if resumed:
        entries = json.loads((checkpoint / "exposure.json").read_text())["entries"]
    for step in range(start + 1, stop + 1):
        seconds = 10.0 * step
        if resumed and SCENARIO.get("reset_clock"):
            seconds = 10.0 * (step - start)
        rewards = [1.0, 1.0, 1.0, 1.0] if SCENARIO.get("kl_only") else [1.0, 0.0, 0.0, 1.0]
        ids = [[5, 2], [9, 2], [9, 2], [5, 2]]
        if resumed and SCENARIO.get("diverge"):
            ids = [[5, 2], [8, 2], [9, 2], [5, 2]]
        with (output / "optimizer_audit.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, "completed_state_active_seconds": seconds,
                                     "latest_gradient_norm": SCENARIO.get("norm", 0.75)}) + "\n")
        with (output / "reward_groups.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "id": accepted[step - 1], "rewards": rewards, "completion_ids": ids,
                "mixed": min(rewards) != max(rewards),
                "recognition_mixed": min(rewards) != max(rewards),
                "completion_tokens": [2, 2, 2, 2]}) + "\n")
        entries.append({"invocation_id": "i", "global_step": step, "id": accepted[step - 1]})
        checkpoint = output / f"checkpoint-{step}"
        (checkpoint / "ref").mkdir(parents=True)
        weights = f"weights {step}".encode()
        for name, content in (("trainer_state.json", json.dumps({"global_step": step})),
                              ("optimizer.pt", "opt"), ("scheduler.pt", "sched"),
                              ("rng_state.pth", "rng"), ("adapter_config.json", "{}")):
            (checkpoint / name).write_text(content)
        (checkpoint / "adapter_model.safetensors").write_bytes(weights)
        (checkpoint / "ref" / "adapter_model.safetensors").write_bytes(reference)
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
                0.25 * len(screen) + SCENARIO.get("scorer_drift", 0.0)},
                         "reference_unchanged": True,
                         "changed_tensors": 0 if SCENARIO.get("frozen") else 2})
    record["environment"]["adapter_source_sha256"] = sha(output / "adapter-source.py")
    (output / "local_training.json").write_text(json.dumps(record))
    vlt._publish_endpoint_receipt(output, record, stop)
    return 0
'''


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


class E0Fixture:
    def __init__(self, tmp, monkeypatch, scenario=None):
        pkg = tmp / "pkg"
        pkg.mkdir()
        (pkg / "torch.py").write_text(FAKE_TORCH)
        (pkg / "fake_e0_trainer.py").write_text(FAKE_TRAINER)
        monkeypatch.setattr(e0_gate, "ENTRY", "fake_e0_trainer:main")
        for name, value in {"PYTHONPATH": f"{pkg}{os.pathsep}{ROOT / 'src'}",
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_HUB_OFFLINE": "1",
                            "ACCELERATE_TORCH_DEVICE": "mps",
                            "FAKE_E0": json.dumps(scenario or {})}.items():
            monkeypatch.setenv(name, value)
        self.w = tmp / "w"
        self.w.mkdir()
        (self.w / "adapter_model.safetensors").write_bytes(b"W weights")
        self.scan = tmp / "scan.jsonl"
        self.scan_ids = [f"s{i:02d}" for i in range(12)]
        self.scan.write_text("".join(json.dumps({"id": i}) + "\n" for i in self.scan_ids))
        self.stage = tmp / "stage"
        self.stage.mkdir()
        self.fragment = self.stage / "fragment.json"
        self.trainer_sha = sha_bytes((pkg / "fake_e0_trainer.py").read_bytes())

    def argv(self, extra=()):
        return ["--fragment", str(self.fragment), "--stage-dir", str(self.stage),
                "--trainer-sha256", self.trainer_sha, "--",
                "--mode", "grpo", "--train", str(self.scan), "--init-adapter", str(self.w),
                "--train-only", "--device", "mps", "--beta", "0.04", *extra]

    def run(self, extra=()):
        assert e0_gate.main(self.argv(extra)) == 0
        return json.loads(self.fragment.read_text())


def test_identical_trajectories_pass_every_check(tmp_path, monkeypatch):
    fixture = E0Fixture(tmp_path, monkeypatch)
    fragment = fixture.run()
    assert fragment["checks"] == dict.fromkeys(e0_gate.CHECKS, True)
    assert fragment["sampler_teacher_max_abs_logprob_difference"] == 0.0004
    # uninterrupted (1) + 2+2 (2) + the altered-reference resume (1) + 1+3 (2), in order.
    labels = [i["label"] for i in fragment["measurements"]["invocations"]]
    assert labels == ["uninterrupted-part1", "resume_2_plus_2-part1", "altered_reference",
                      "resume_2_plus_2-part2", "resume_1_plus_3-part1",
                      "resume_1_plus_3-part2"]
    codes = [i["exit_code"] for i in fragment["measurements"]["invocations"]]
    assert codes == [0, 0, 2, 0, 0, 0]
    invocation = fragment["measurements"]["invocations"][0]
    assert invocation["peak_mps_driver_allocated_bytes"] == 3 * 1024**3
    assert invocation["peak_mps_tensor_allocated_bytes"] == 2 * 1024**3
    assert invocation["peak_rss_bytes"] > 0
    assert fragment["measurements"]["update_seconds"] == [10.0, 10.0, 10.0, 10.0]
    assert fragment["measurements"]["generated_tokens"] == 4 * 8
    assert sorted(fragment["measurements"]["checkpoint_bytes"]) == [
        "checkpoint-1", "checkpoint-2", "checkpoint-3", "checkpoint-4"]
    assert fragment["detail"]["task_gradient_steps"] == [1, 2, 3, 4]
    # The corrupted copy is the 2+2 split checkpoint with one bit of one value flipped.
    altered = fragment["altered_reference"]
    assert altered["file"].endswith("altered_reference/checkpoint-2/ref/adapter_model.safetensors")
    assert altered["sha256_before"] != altered["sha256_after"]
    data = Path(altered["file"]).read_bytes()
    length = int.from_bytes(data[:8], "little")
    assert struct.unpack("<f", data[8 + length:12 + length])[0] != 0.25
    assert ANCHOR in altered["error"]
    # The real split runs resumed from their own checkpoints at 2 and 1.
    records = sorted(p.name for p in fixture.stage.glob("*-record.json"))
    assert records == ["resume_1_plus_3-part1-record.json", "resume_1_plus_3-part2-record.json",
                       "resume_2_plus_2-part1-record.json", "resume_2_plus_2-part2-record.json",
                       "uninterrupted-part1-record.json"]


@pytest.mark.parametrize("scenario, failed", [
    # A resumed process that samples different tokens is not identical.
    ({"diverge": True}, {"resume_2_plus_2_identical", "resume_1_plus_3_identical"}),
    # A trainer that accepts an altered reference checkpoint fails the forced fire.
    ({"ignore_tamper": True}, {"altered_reference_checkpoint_refused"}),
    # All-equal rewards: zero advantage, so a nonzero norm is KL-only movement.
    ({"kl_only": True}, {"useful_mixed_group", "finite_nonzero_task_gradient"}),
    ({"norm": 0.0}, {"finite_nonzero_task_gradient"}),
    ({"frozen": True}, {"lora_parameter_changed"}),
    # A resume that restarts the active-time clock is not cumulative.
    ({"reset_clock": True}, {"active_time_ledger_cumulative"}),
    # Scorer seconds the trainer counted inside active time that the committed screen
    # journal does not hold (unledgered or discarded scorer work) fail the scorer check.
    ({"scorer_drift": 0.5}, {"scorer_seconds_within_active_time"}),
])
def test_each_failure_is_written_false_not_raised(tmp_path, monkeypatch, scenario, failed):
    fixture = E0Fixture(tmp_path, monkeypatch, scenario)
    fragment = fixture.run()
    assert {k for k, v in fragment["checks"].items() if v is not True} == failed


def test_parity_is_reported_for_the_launcher_to_bound(tmp_path, monkeypatch):
    fixture = E0Fixture(tmp_path, monkeypatch, {"parity": 0.002})
    fragment = fixture.run()
    assert fragment["sampler_teacher_max_abs_logprob_difference"] == 0.002
    assert all(fragment["checks"].values())  # the 0.001 bound is the launcher's


def test_scan_order_w_identity_and_refusal_reason_are_checked(tmp_path, monkeypatch):
    fixture = E0Fixture(tmp_path, monkeypatch)
    fixture.run()
    runs = {name: e0_gate.read_run(fixture.stage / name, name) for name in e0_gate.SPLITS}
    refused = {"exit_code": 2, "record_status": "failed", "error": ANCHOR}
    w = sha_bytes(b"W weights")
    checks, _, _ = e0_gate.evaluate(runs, refused, w, fixture.scan_ids)
    assert all(checks.values())
    checks, _, _ = e0_gate.evaluate(runs, refused, w, list(reversed(fixture.scan_ids)))
    assert checks["scan_follows_s_scan_seed42"] is False
    checks, _, _ = e0_gate.evaluate(runs, refused, sha_bytes(b"another adapter"),
                                    fixture.scan_ids)
    assert not checks["initial_adapter_is_w"] and not checks["reference_unchanged"]
    # A corrupted resume that failed for another reason did not show the guard firing.
    other = {"exit_code": 2, "record_status": "failed", "error": "OSError: disk full"}
    checks, _, _ = e0_gate.evaluate(runs, other, w, fixture.scan_ids)
    assert checks["altered_reference_checkpoint_refused"] is False


def test_a_failed_trainer_part_raises_and_writes_no_fragment(tmp_path, monkeypatch):
    fixture = E0Fixture(tmp_path, monkeypatch, {"fail_stop": 4, "fail_resumed": True})
    with pytest.raises(e0_gate.E0Error):
        e0_gate.main(fixture.argv())
    assert not fixture.fragment.exists()


@pytest.mark.parametrize("extra, message", [
    (["--save-steps", "8"], "owns"),
    (["--output", "/tmp/x"], "owns"),
    (["--beta", "0"], "beta 0.04"),
])
def test_owned_or_wrong_trainer_flags_are_refused(tmp_path, monkeypatch, extra, message):
    fixture = E0Fixture(tmp_path, monkeypatch)
    argv = fixture.argv()
    if extra[0] == "--beta":
        argv = argv[:argv.index("--beta")] + extra
    else:
        argv = argv + extra
    with pytest.raises(SystemExit, match=message):
        e0_gate.main(argv)
    assert not (fixture.stage / "uninterrupted").exists()


def test_launcher_requires_exactly_this_check_set():
    launch = importlib.import_module("runpy").run_path(str(PACKAGE / "launch.py"))
    assert tuple(launch["E0_CHECKS"]) == e0_gate.CHECKS
