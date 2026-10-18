"""E0: the protocol's real_learning_and_resume_gate for the freshpool screened sampler.

Launcher entry ``e0_gate:main`` (scripts/freshpool_launch/launch.py, template ``e0``).
Usage, after the phase worker's ``--``:

  --fragment F --stage-dir DIR --trainer-sha256 HEX
  -- <vision_rl.vlm_local_training CLI arguments for a screened GRPO run from W>

The trainer arguments must name ``--mode grpo``, ``--train`` (the S_scan_seed42 manifest),
``--init-adapter`` (W), ``--train-only`` and ``--device``; the entry owns ``--output``,
``--max-steps``, ``--target-accepted``, ``--screen-journal``, ``--stop-after-steps``,
``--resume-from-checkpoint`` and ``--save-steps``.

Protocol real_learning_and_resume_gate ("From W, run four real accepted updates
uninterrupted and compare with two-plus-two and one-plus-three resumed from complete
optimizer-boundary checkpoints: identical scan position, screening journal, sampled tokens,
rewards and adapter bytes"). Each trainer invocation runs in its OWN process through
phase_worker.py, the way every real stage runs: the 2+2 and 1+3 resumes restore from a
checkpoint in a fresh process (MPS, CPU and Python RNG, optimizer, scheduler, both adapters,
sampler position and the active-time ledger), each child carries the 24 GiB MPS driver guard,
and its memory is returned to the system when it exits. A fourth, deliberately corrupted
resume (the 2+2 split checkpoint with one reference-adapter value changed) must be REFUSED by
the trainer: "Reject an altered-reference checkpoint".

Checks (every one must be true; the launcher's E0 validator requires exactly this set):
  resume_2_plus_2_identical, resume_1_plus_3_identical
      global step, final adapter bytes, the committed screen journal (scan position, ID, p,
      u, accepted, adapter hash, step), the reward groups (ID, rewards, completion token
      IDs) and the committed prompt exposure, each equal to the uninterrupted run's
  scan_follows_s_scan_seed42      the committed scanned IDs are a prefix of the --train
                                  manifest, in its order, and u >= 0.5 exactly when accepted
  initial_adapter_is_w            every run started from W's weights (the --init-adapter file)
  reference_unchanged             every run's KL reference is W's saved adapter, identical
                                  across runs, and every run's audit reports it unchanged
  altered_reference_checkpoint_refused
                                  the corrupted resume failed with the trainer's anchor error
  useful_mixed_group              at least one group with a valid correct AND a valid wrong
                                  answer (reward journal ``recognition_mixed``)
  finite_nonzero_task_gradient    every pre-clip gradient norm is finite, and some update
                                  whose group rewards are not all equal (nonzero GRPO task
                                  advantage) has a nonzero norm; KL-only movement does not
                                  count (study: "KL-only movement does not qualify")
  lora_parameter_changed          the final adapter bytes differ from W's and the audit saw
                                  changed LoRA tensors
  scorer_uses_sampler_transform   the screen's scorer took the sampler's image-token
                                  suppression, EOS set and temperature
  active_time_ledger_cumulative   in every run the committed active seconds per step are
                                  1..4, nondecreasing across the resume boundary, and end at
                                  the receipt's committed seconds
plus ``sampler_teacher_max_abs_logprob_difference``: the largest of the three runs' trainer
parity (sampled vs teacher-forced token log-probabilities after the same suppression,
float32), which the launcher bounds by 0.001.

Measurements (reported, not gated; protocol "Measure peak RSS and MPS driver/tensor
allocations separately, synchronized update time, generation tokens and checkpoint bytes"):
per invocation the child's peak RSS, peak MPS driver and tensor allocation (phase worker
telemetry) and wall seconds; per update the committed active seconds; generated tokens; and
the bytes of every checkpoint of the uninterrupted run.

A check that fails is written as false, never raised, so the launcher's E0 validator is what
stops the study. A trainer invocation that fails (other than the corrupted resume, which must
fail) raises: no fragment is written and the stage fails.

Blind spots, stated: the KL-free task gradient is inferred from a nonzero norm on a step whose
group rewards differ (an all-equal group has zero advantage, so its gradient is KL-only); the
two are not separated numerically. The parity is the trainer's own measurement on the first
scan row, up to 16 tokens.
"""

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = 4  # protocol: four real accepted updates
SPLITS = {"uninterrupted": [4], "resume_2_plus_2": [2, 2], "resume_1_plus_3": [1, 3]}
ENTRY = "vision_rl.vlm_local_training:main"
OWNED = ("--output", "--max-steps", "--target-accepted", "--screen-journal",
         "--stop-after-steps", "--resume-from-checkpoint", "--save-steps")
REQUIRED = ("--mode", "--train", "--init-adapter", "--device")
SCREEN_KEYS = ("scan_position", "id", "p", "u", "accepted", "adapter_sha256", "global_step")
GROUP_KEYS = ("id", "rewards", "completion_ids")
COMPARED = ("global_step", "adapter_sha256", "screen", "reward_groups", "exposure")
ANCHOR_ERROR = "reference adapter differs from the pinned initial anchor"
CHILD_TIMEOUT_SECONDS = 3600
CHECKS = (
    "resume_2_plus_2_identical", "resume_1_plus_3_identical", "scan_follows_s_scan_seed42",
    "initial_adapter_is_w", "reference_unchanged", "altered_reference_checkpoint_refused",
    "useful_mixed_group", "finite_nonzero_task_gradient", "lora_parameter_changed",
    "scorer_uses_sampler_transform", "active_time_ledger_cumulative",
)


class E0Error(RuntimeError):
    """A trainer invocation that had to succeed did not."""


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def flag_values(argv, flag):
    values = []
    for n, part in enumerate(argv):
        if part == flag:
            values.append(argv[n + 1] if n + 1 < len(argv) else None)
        elif part.startswith(flag + "="):
            values.append(part[len(flag) + 1:])
    return values


def single(argv, flag):
    values = flag_values(argv, flag)
    if len(values) != 1 or values[0] is None:
        raise SystemExit(f"the trainer arguments need exactly one {flag}")
    return values[0]


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


class Runner:
    """Runs one trainer invocation in its own guarded process, as every real stage does."""

    def __init__(self, stage_dir, trainer_sha256):
        self.guard_dir = stage_dir / "guards"
        self.trainer_sha256 = trainer_sha256
        self.dry_run = os.environ.get("ACCELERATE_TORCH_DEVICE") == "cpu"
        self.invocations = []

    def __call__(self, label, argv):
        before = set(self.guard_dir.glob("*.json")) if self.guard_dir.is_dir() else set()
        command = [sys.executable, "-B", str(HERE / "phase_worker.py"),
                   "--guard-dir", str(self.guard_dir), "--stage", f"e0-{label}",
                   "--kind", "train", "--entry", ENTRY, "--entry-sha256", self.trainer_sha256,
                   *(["--dry-run"] if self.dry_run else []), "--", *argv]
        print(json.dumps({"e0_invocation": label}), flush=True)
        completed = subprocess.run(command, timeout=CHILD_TIMEOUT_SECONDS)
        guards = sorted(set(self.guard_dir.glob("*.json")) - before)
        guard = json.loads(guards[-1].read_text()) if guards else {}
        # ru_maxrss is bytes on macOS and KiB on Linux; it is the largest child so far.
        maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        record = {"label": label, "exit_code": completed.returncode,
                  "elapsed_seconds": guard.get("elapsed_seconds"),
                  "peak_mps_driver_allocated_bytes": guard.get("peak_driver_allocated_bytes"),
                  "peak_mps_tensor_allocated_bytes": guard.get("peak_tensor_allocated_bytes"),
                  "children_peak_rss_bytes": maxrss if sys.platform == "darwin"
                  else maxrss * 1024}
        self.invocations.append(record)
        return completed.returncode


def trainer_argv(common, output, target, done):
    argv = [*common, "--max-steps", str(STEPS), "--target-accepted", str(STEPS),
            "--save-steps", "1", "--screen-journal", str(output / "screen.jsonl"),
            "--output", str(output)]
    if target < STEPS:
        argv += ["--stop-after-steps", str(target)]
    if done:
        argv += ["--resume-from-checkpoint", str(output / f"checkpoint-{done}")]
    return argv


def trajectory(run, common, name, splits, directory, after_first_part=None):
    done = 0
    for number, part in enumerate(splits):
        target = done + part
        code = run(f"{name}-part{number + 1}", trainer_argv(common, directory, target, done))
        if code != 0:
            raise E0Error(f"trainer exited {code} on {name} part {number + 1}")
        shutil.copyfile(directory / "local_training.json",
                        directory.parent / f"{name}-part{number + 1}-record.json")
        done = target
        if number == 0 and after_first_part is not None:
            after_first_part(directory, done)
    return read_run(directory, name)


def read_run(directory, name):
    from vision_rl import vlm_local_training as vlt

    receipt = vlt.read_endpoint_receipt(directory)
    record = json.loads((directory / "local_training.json").read_text())
    checkpoint = directory / receipt["checkpoint"]["path"]
    groups = read_jsonl(directory / "reward_groups.jsonl")
    seconds = {}
    for entry in read_jsonl(directory / "optimizer_audit.jsonl"):
        seconds[entry["step"]] = entry  # last row per step: the committed one
    parts = sorted(directory.parent.glob(f"{name}-part*-record.json"))
    return {
        "directory": str(directory),
        "receipt": receipt,
        "record": record,
        "part_records": [json.loads(p.read_text()) for p in parts],
        "global_step": receipt["global_step"],
        "adapter_sha256": receipt["final_adapter"]["weights_sha256"],
        "screen": [{k: e[k] for k in SCREEN_KEYS}
                   for e in vlt.committed_screen_entries(checkpoint)],
        "reward_groups": [{k: g[k] for k in GROUP_KEYS} for g in groups],
        "groups": groups,
        "exposure": receipt["committed_prompt_ids"],
        "audit_rows": [seconds[s] for s in sorted(seconds)],
        "checkpoint_bytes": {p.name: directory_bytes(p)
                             for p in sorted(directory.glob("checkpoint-*"))},
    }


def corrupt_reference(source, destination, step):
    """Copy a run directory at its split checkpoint and change one reference value.

    safetensors layout: an 8-byte little-endian header length, a JSON header naming each
    tensor's dtype and data_offsets, then the raw data. Flipping the lowest bit of the
    first byte of the alphabetically first tensor changes one float32 value by one unit in
    the last place (little-endian low mantissa byte) and leaves the file loadable, so the
    trainer must refuse it on identity, not on a parse error."""
    shutil.copytree(source, destination, symlinks=True)
    path = destination / f"checkpoint-{step}" / "ref" / "adapter_model.safetensors"
    data = bytearray(path.read_bytes())
    length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + length])
    name = sorted(key for key in header if key != "__metadata__")[0]
    if header[name]["dtype"] != "F32":
        raise E0Error(f"reference tensor {name} is {header[name]['dtype']}, expected F32")
    offset = 8 + length + header[name]["data_offsets"][0]
    before = file_sha256(path)
    data[offset] ^= 1
    path.write_bytes(bytes(data))
    return {"tensor": name, "file": str(path), "sha256_before": before,
            "sha256_after": file_sha256(path)}


def identical(first, second):
    return all(first[key] == second[key] for key in COMPARED)


def ledger_cumulative(run):
    rows = run["audit_rows"]
    values = [row.get("completed_state_active_seconds") for row in rows]
    return ([row["step"] for row in rows] == list(range(1, STEPS + 1))
            and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)
            and all(b >= a for a, b in zip(values, values[1:]))
            and values[-1] == run["receipt"]["committed_active_seconds"])


def evaluate(runs, tamper, w_sha, scan_ids):
    base = runs["uninterrupted"]
    every = list(runs.values())
    records = [r["record"] for r in every] + [p for r in every for p in r["part_records"]]
    audits = [r.get("optimizer_audit") or {} for r in records]
    references = [r["record"].get("reference") or {} for r in every]
    transform = base["record"].get("screen_sampler_transform") or {}
    profile = base["record"]["control_identity"]["profile"]
    rows, groups = base["audit_rows"], base["groups"]
    norms = [row.get("latest_gradient_norm") for row in rows]
    finite = len(norms) == STEPS and all(
        type(n) in (int, float) and math.isfinite(n) for n in norms)
    task_steps = [row["step"] for row, group in zip(rows, groups, strict=False)
                  if min(group["rewards"]) != max(group["rewards"])
                  and type(row.get("latest_gradient_norm")) in (int, float)
                  and row["latest_gradient_norm"] > 0]
    scanned = [entry["id"] for entry in base["screen"]]
    checks = {
        "resume_2_plus_2_identical": identical(base, runs["resume_2_plus_2"]),
        "resume_1_plus_3_identical": identical(base, runs["resume_1_plus_3"]),
        "scan_follows_s_scan_seed42": scanned == scan_ids[: len(scanned)]
        and len(groups) == STEPS
        and [e["id"] for e in base["screen"] if e["accepted"]] == base["exposure"]
        and all((e["u"] >= 0.5) == e["accepted"] for e in base["screen"]),
        "initial_adapter_is_w": all(
            ((r.get("settings") or {}).get("init_adapter") or {}).get("weights_sha256")
            == w_sha for r in records),
        "reference_unchanged": all(ref.get("initial_adapter_weights_sha256") == w_sha
                                   for ref in references)
        and all(ref.get("adapter") == references[0].get("adapter") for ref in references)
        and all(a.get("reference_unchanged") is True for a in audits),
        "altered_reference_checkpoint_refused": tamper["exit_code"] not in (0, None)
        and tamper["record_status"] == "failed" and ANCHOR_ERROR in tamper["error"],
        "useful_mixed_group": any(g.get("recognition_mixed") is True for g in groups),
        "finite_nonzero_task_gradient": finite and bool(task_steps),
        "lora_parameter_changed": base["adapter_sha256"] != w_sha
        and (base["record"].get("optimizer_audit") or {}).get("changed_tensors", 0) > 0,
        # The trainer hands the scorer exactly this transform and refuses any suppression
        # but the image token; the parity diagnostic sampled with the same suppression.
        "scorer_uses_sampler_transform": isinstance(transform.get("suppress_tokens"), list)
        and len(transform["suppress_tokens"]) == 1 and bool(transform.get("eos_token_ids"))
        and transform.get("temperature") == profile["temperature"]
        and all(p.get("sampling_suppressed_token_ids") == transform["suppress_tokens"]
                for p in (r["record"].get("sampler_teacher_logprob_parity") or {}
                          for r in every)),
        "active_time_ledger_cumulative": all(ledger_cumulative(r) for r in every),
    }
    parities = [r["record"].get("sampler_teacher_logprob_parity") or {} for r in every]
    errors = [p.get("max_abs_logprob_error_after_same_suppression") for p in parities]
    agreement = (max(errors) if all(type(e) in (int, float) and math.isfinite(e)
                                    for e in errors) else None)
    return checks, agreement, {"task_gradient_steps": task_steps, "gradient_norms": norms,
                               "parity": parities}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("trainer arguments must follow a literal --")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--trainer-sha256", required=True)
    args = parser.parse_args(argv[:split])
    common = argv[split + 1:]
    if any(a == flag or a.startswith(flag + "=") for a in common for flag in OWNED):
        raise SystemExit(f"the E0 entry owns {', '.join(OWNED)}")
    for flag in REQUIRED:
        single(common, flag)
    if single(common, "--mode") != "grpo" or "--train-only" not in common:
        raise SystemExit("E0 is a TRAIN-only screened GRPO run")
    if flag_values(common, "--beta") not in ([], ["0.04"]):
        raise SystemExit("E0 needs the study's beta 0.04 KL reference")
    w_sha = file_sha256(Path(single(common, "--init-adapter")) / "adapter_model.safetensors")
    scan_ids = [row["id"] for row in read_jsonl(single(common, "--train"))]
    stage = args.stage_dir.resolve()
    run = Runner(stage, args.trainer_sha256)
    tamper = {}

    def tamper_after_split(directory, step):
        target = stage / "altered_reference"
        tamper.update(corrupt_reference(directory, target, step))
        code = run("altered_reference", trainer_argv(common, target, STEPS, step))
        record = json.loads((target / "local_training.json").read_text())
        tamper.update(exit_code=code, record_status=record.get("status"),
                      error=str(record.get("error", "")))

    runs = {}
    for name, splits in SPLITS.items():
        hook = tamper_after_split if name == "resume_2_plus_2" else None
        runs[name] = trajectory(run, common, name, splits, stage / name, hook)
    checks, agreement, detail = evaluate(runs, tamper, w_sha, scan_ids)
    base = runs["uninterrupted"]
    rows = base["audit_rows"]
    seconds = [row["completed_state_active_seconds"] for row in rows]
    fragment = {
        "scope": "protocol real_learning_and_resume_gate (study E0), engineering compute",
        "steps": STEPS,
        "checks": checks,
        "sampler_teacher_max_abs_logprob_difference": agreement,
        "detail": detail,
        "altered_reference": tamper,
        "measurements": {
            "invocations": run.invocations,
            "update_committed_active_seconds": dict(zip(
                [str(r["step"]) for r in rows], seconds)),
            "update_seconds": [b - a for a, b in zip([0.0, *seconds], seconds)],
            "generated_tokens": sum(sum(g.get("completion_tokens", [])) for g in base["groups"]),
            "checkpoint_bytes": base["checkpoint_bytes"],
        },
        "trajectories": {
            name: {"directory": r["directory"], "global_step": r["global_step"],
                   "adapter_sha256": r["adapter_sha256"],
                   **{f"{key}_sha256": digest(r[key])
                      for key in ("screen", "reward_groups", "exposure")}}
            for name, r in runs.items()},
    }
    temporary = args.fragment.with_suffix(".tmp")
    temporary.write_text(json.dumps(fragment, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
