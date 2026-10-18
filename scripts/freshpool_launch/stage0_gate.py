"""Stage 0 of the signal-limit study: the c(G) cost probe and the beta-0 dr_grpo E0, per G.

Launcher entry ``stage0_gate:main`` (scripts/freshpool_launch/signal_limit_launch.py,
template ``stage0``). Usage, after the phase worker's ``--``:

  --fragment F --stage-dir DIR --trainer-sha256 HEX
  [--groups 8,16] [--e0-steps 4] [--probe-accepted 8]
  -- <vision_rl.vlm_local_training CLI arguments for a screened GRPO run from W>

The trainer arguments must name ``--mode grpo``, ``--train`` (train400, the scan manifest),
``--init-adapter`` (W), ``--device``, ``--beta 0``, ``--loss-type dr_grpo``,
``--scale-rewards none`` and ``--train-only``; the entry owns ``--num-generations``,
``--output``, ``--max-steps``, ``--target-accepted``, ``--screen-journal``,
``--stop-after-steps``, ``--resume-from-checkpoint`` and ``--save-steps``.

Study (docs/aircraft-rl-next-study-v1.json stages "Stage 0"): "cost probe at W on train400
images: screened beta=0 dr_grpo at G8 and at G16, each until at least 8 accepted updates, with
the screened sampler -> c(G8), c(G16) per accepted update (scorer included), both MEASURED;
then E0 (4 vs 2+2 vs 1+3 byte-identical) for beta=0, dr_grpo, scale_rewards none at G8 AND at
G16". Per G, in this order:
  probe   one uninterrupted screened run to ``--probe-accepted`` accepted updates;
          c(G) = the receipt's committed active seconds / the accepted updates the committed
          screen journal holds. Committed active time includes the scorer; c(G) is reported
          only when the run's scorer seconds are proven inside it (e0_gate's
          scorer_within_active) and accepted == global step == committed exposures ==
          ``--probe-accepted``; otherwise it is null and the launcher refuses.
  E0      N = ``--e0-steps`` uninterrupted vs N/2 + (N - N/2) vs 1 + (N - 1), each part
          resumed in a fresh process from the previous part's optimizer-boundary
          checkpoint, compared as e0_gate compares (global step, final adapter bytes,
          committed screen journal, reward groups, committed exposure).
Every trainer invocation runs in its own phase_worker.py process (e0_gate.Runner's shape);
when this entry's own worker is a UW path, each child gets that path's flag, never
``--dry-run`` (child_execution_flags: ACCELERATE_TORCH_DEVICE cuda -> ``--remote-cuda``;
VISION_RL_TORCH_THREADS set -> ``--remote-cpu``; cpu otherwise -> ``--dry-run``; MPS ->
none).

Checks, keyed by G (``checks["8"]``, ``checks["16"]``), each false unless proven:
  resume_2_plus_2_identical, resume_1_plus_3_identical
      the N/2 + (N - N/2) and 1 + (N - 1) resumes equal the uninterrupted run (at the study's
      N = 4 these are exactly 2+2 and 1+3; a smoke's actual splits are in ``e0_splits``)
  finite_nonzero_task_gradient    every pre-clip gradient norm of the uninterrupted run is
                                  finite and some update with unequal group rewards has a
                                  nonzero norm (at beta 0 there is no KL term at all)
  no_reference_forward_at_beta_0  for the probe and every E0 run, all of: every record (final
      and each part) has settings beta == 0 and no ``reference``; the receipt's
      reference_sha256 is null; no checkpoint holds a ``ref`` adapter directory (TRL 1.16
      adds the "ref" adapter, computes ref_per_token_logps and logs ``kl`` only under
      ``beta != 0.0``); the endpoint checkpoint's trainer_state log_history has a logged
      ``loss`` for every step 1..global step and no ``kl`` metric anywhere; no optimizer
      audit row or record carries reference_sha256 / reference_unchanged; and every run's
      sampler-teacher parity reports its initial-reference fields as null (the parity
      diagnostic runs the ref adapter whenever one exists). Missing evidence is false.
  loss_type_dr_grpo_recorded      every record of the probe and E0 runs: mode grpo,
                                  loss_type dr_grpo, scale_rewards none, beta 0
plus e0_gate's checks that apply at beta 0 (its reference_unchanged and
altered_reference_checkpoint_refused presuppose a KL reference and do not):
  scan_follows_train_manifest, initial_adapter_is_w, useful_mixed_group,
  lora_parameter_changed, scorer_uses_sampler_transform, active_time_ledger_cumulative,
  scorer_seconds_within_active_time (e0_gate's definitions, over N steps; the ledger and
  scorer checks also cover the probe)
and group_size_is_g: every record's screening group_size is G and every reward group holds G
rewards (a run at the wrong G cannot be keyed under this one).

Fragment: e0_groups, e0_steps, e0_splits, probe_accepted, study_parameters (true only at
the study's 8,16 / 4 / 8; anything else exists only for a smoke and the launcher's
stage0_validator refuses it), checks, c_accepted_steps, c_seconds_per_accepted_update,
sampler_teacher_max_abs_logprob_difference (the largest over every probe and E0 run at every
G; null if any run lacks it), detail and measurements (per G: per invocation peak RSS, MPS
driver and tensor bytes and wall seconds; probe committed seconds and scorer seconds; update
seconds; generated tokens; checkpoint bytes).

A check that fails is written false, never raised. A trainer invocation that fails raises: no
fragment is written and the stage fails.

Blind spots, stated: the absence of a reference forward is inferred from what TRL 1.16 does
and records under ``beta != 0.0`` (no "ref" adapter, no ``kl`` metric, no reference record,
null parity reference fields); a forward through some other path that records nothing would
not be seen. The task gradient is inferred as in e0_gate. c(G) is one run's ratio, not a mean
over repeats.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e0_gate as E  # noqa: E402

STUDY_GROUPS = (8, 16)
STUDY_E0_STEPS = 4
STUDY_PROBE_ACCEPTED = 8
ENTRY = E.ENTRY
OWNED = (*E.OWNED, "--num-generations")
REQUIRED = ("--mode", "--train", "--init-adapter", "--device", "--beta", "--loss-type",
            "--scale-rewards")
CHECKS = (
    # signal_limit_launch.STAGE0_CHECKS, written out (the launcher does not import this file)
    "resume_2_plus_2_identical", "resume_1_plus_3_identical", "finite_nonzero_task_gradient",
    "no_reference_forward_at_beta_0", "loss_type_dr_grpo_recorded",
    # e0_gate's checks that apply at beta 0
    "scan_follows_train_manifest", "initial_adapter_is_w", "useful_mixed_group",
    "lora_parameter_changed", "scorer_uses_sampler_transform",
    "active_time_ledger_cumulative", "scorer_seconds_within_active_time",
    "group_size_is_g",
)
KL_KEY = re.compile(r"(^|[/_])kl($|[/_])")
PARITY_REFERENCE_KEYS = ("initial_reference_max_abs_logprob_difference",
                         "initial_reference_sampled_kl")


def splits(steps):
    half = steps // 2
    return {"uninterrupted": [steps], "resume_2_plus_2": [half, steps - half],
            "resume_1_plus_3": [1, steps - 1]}


def child_execution_flags(environ):
    """The phase-worker flag that reproduces this entry's own execution path in a child:
    ACCELERATE_TORCH_DEVICE cuda -> ``--remote-cuda`` (UW GPU path); else
    VISION_RL_TORCH_THREADS set (phase_worker allows it only under a remote flag) ->
    ``--remote-cpu``; else cpu -> ``--dry-run``; else (local MPS) none, as e0_gate."""
    device = environ.get("ACCELERATE_TORCH_DEVICE")
    if device == "cuda":
        return ["--remote-cuda"]
    if environ.get("VISION_RL_TORCH_THREADS") is not None:
        return ["--remote-cpu"]
    if device == "cpu":
        return ["--dry-run"]
    return []


class Runner(E.Runner):
    """e0_gate.Runner, labelled per G and carrying the parent's execution path
    (child_execution_flags) instead of e0_gate's cpu-means-dry-run rule."""

    def __init__(self, stage_dir, trainer_sha256):
        super().__init__(stage_dir, trainer_sha256)
        self.mode = child_execution_flags(os.environ)
        self.dry_run = self.mode == ["--dry-run"]

    def __call__(self, label, argv):
        before = set(self.guard_dir.glob("*.json")) if self.guard_dir.is_dir() else set()
        mode = self.mode
        command = [sys.executable, "-B", str(HERE / "phase_worker.py"),
                   "--guard-dir", str(self.guard_dir), "--stage", f"stage0-{label}",
                   "--kind", "train", "--entry", ENTRY, "--entry-sha256", self.trainer_sha256,
                   *mode, "--", *argv]
        print(json.dumps({"stage0_invocation": label}), flush=True)
        completed = subprocess.run(command, timeout=E.CHILD_TIMEOUT_SECONDS)
        guards = sorted(set(self.guard_dir.glob("*.json")) - before)
        guard = json.loads(guards[-1].read_text()) if guards else {}
        self.invocations.append({
            "label": label, "exit_code": completed.returncode,
            "execution": mode[0][2:] if mode else "mps",
            "elapsed_seconds": guard.get("elapsed_seconds"),
            "peak_rss_bytes": guard.get("peak_rss_bytes"),
            "peak_mps_driver_allocated_bytes": guard.get("peak_driver_allocated_bytes"),
            "peak_mps_tensor_allocated_bytes": guard.get("peak_tensor_allocated_bytes")})
        return completed.returncode


def trainer_argv(common, group, steps, output, target, done, save_steps=1):
    argv = [*common, "--num-generations", str(group), "--max-steps", str(steps),
            "--target-accepted", str(steps), "--save-steps", str(save_steps),
            "--screen-journal", str(output / "screen.jsonl"), "--output", str(output)]
    if target < steps:
        argv += ["--stop-after-steps", str(target)]
    if done:
        argv += ["--resume-from-checkpoint", str(output / f"checkpoint-{done}")]
    return argv


def trajectory(run, common, group, steps, name, parts, directory):
    done = 0
    for number, part in enumerate(parts):
        target = done + part
        label = f"g{group}-{name}-part{number + 1}"
        code = run(label, trainer_argv(common, group, steps, directory, target, done))
        if code != 0:
            raise E.E0Error(f"trainer exited {code} on {label}")
        (directory.parent / f"{name}-part{number + 1}-record.json").write_bytes(
            (directory / "local_training.json").read_bytes())
        done = target
    return E.read_run(directory, name)


def probe(run, common, group, accepted, directory):
    code = run(f"g{group}-probe", trainer_argv(common, group, accepted, directory, accepted, 0,
                                               save_steps=accepted))
    if code != 0:
        raise E.E0Error(f"trainer exited {code} on the G{group} cost probe")
    (directory.parent / "probe-part1-record.json").write_bytes(
        (directory / "local_training.json").read_bytes())
    return E.read_run(directory, "probe")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def cost(run, accepted):
    """(c(G), accepted count, reason): c only when every condition holds."""
    receipt = run["receipt"]
    count = sum(1 for entry in run["screen"] if entry["accepted"] is True)
    seconds = receipt.get("committed_active_seconds")
    if not (count == run["global_step"] == len(run["exposure"]) == accepted):
        return None, count, "accepted updates differ from global step, exposure or target"
    if not number(seconds) or seconds <= 0:
        return None, count, "committed active seconds missing or not positive"
    if not E.scorer_within_active(run):
        return None, count, "scorer seconds not proven inside committed active time"
    return seconds / count, count, None


def records_of(run):
    return [run["record"], *run["part_records"]]


def ledger_cumulative(run, steps):
    rows = run["audit_rows"]
    values = [row.get("completed_state_active_seconds") for row in rows]
    return ([row["step"] for row in rows] == list(range(1, steps + 1))
            and all(number(v) and v > 0 for v in values)
            and all(b >= a for a, b in zip(values, values[1:]))
            and values[-1] == run["receipt"]["committed_active_seconds"])


def is_zero(value):
    return number(value) and value == 0


def no_reference_forward(run):
    """Every observable of a reference adapter or forward that the trainer and TRL leave,
    each required to be present and to show none. Missing evidence returns False."""
    directory = Path(run["directory"])
    receipt = run["receipt"]
    for record in records_of(run):
        if not is_zero((record.get("settings") or {}).get("beta")):
            return False
        if "reference" not in record or record["reference"] is not None:
            return False
        audit = record.get("optimizer_audit") or {}
        if "reference_sha256" in audit or "reference_unchanged" in audit:
            return False
    if "reference_sha256" not in receipt or receipt["reference_sha256"] is not None:
        return False
    checkpoints = sorted(directory.glob("checkpoint-*"))
    if not checkpoints or any((c / "ref").exists() for c in checkpoints):
        return False
    state = json.loads((directory / receipt["checkpoint"]["path"] / "trainer_state.json")
                       .read_text())
    history = state.get("log_history")
    if not isinstance(history, list) or not all(isinstance(h, dict) for h in history):
        return False
    logged = {h.get("step") for h in history if number(h.get("loss"))}
    if not set(range(1, run["global_step"] + 1)) <= logged:
        return False
    if any(KL_KEY.search(str(key)) for entry in history for key in entry):
        return False
    if any("reference_sha256" in row or "reference_unchanged" in row
           for row in run["audit_rows"]):
        return False
    parity = run["record"].get("sampler_teacher_logprob_parity")
    return (isinstance(parity, dict)
            and all(key in parity and parity[key] is None for key in PARITY_REFERENCE_KEYS))


def dr_grpo_recorded(run):
    for record in records_of(run):
        settings = record.get("settings") or {}
        if (settings.get("mode") != "grpo" or settings.get("loss_type") != "dr_grpo"
                or settings.get("scale_rewards") != "none" or not is_zero(settings.get("beta"))):
            return False
    return True


def group_is(run, group, steps):
    return (all(((r.get("settings") or {}).get("screening") or {}).get("group_size") == group
                for r in records_of(run))
            and len(run["groups"]) == steps
            and all(len(g.get("rewards", ())) == group for g in run["groups"]))


def evaluate(runs, probe_run, group, steps, w_sha, scan_ids):
    """Every check under one guard: a malformed trainer output makes every check false
    (the error in ``detail``), never a crash."""
    try:
        return _evaluate(runs, probe_run, group, steps, w_sha, scan_ids)
    except Exception as exc:
        return dict.fromkeys(CHECKS, False), {"error": f"{type(exc).__name__}: {exc}"}


def _evaluate(runs, probe_run, group, steps, w_sha, scan_ids):
    base = runs["uninterrupted"]
    every = list(runs.values())
    everything = [probe_run, *every]
    records = [r for run in every for r in records_of(run)]
    transform = base["record"].get("screen_sampler_transform") or {}
    profile = base["record"]["control_identity"]["profile"]
    rows, groups = base["audit_rows"], base["groups"]
    norms = [row.get("latest_gradient_norm") for row in rows]
    finite = len(norms) == steps and all(number(n) for n in norms)
    task_steps = [row["step"] for row, g in zip(rows, groups, strict=False)
                  if min(g["rewards"]) != max(g["rewards"])
                  and number(row.get("latest_gradient_norm")) and row["latest_gradient_norm"] > 0]
    scanned = [entry["id"] for entry in base["screen"]]
    checks = {
        "resume_2_plus_2_identical": E.identical(base, runs["resume_2_plus_2"]),
        "resume_1_plus_3_identical": E.identical(base, runs["resume_1_plus_3"]),
        "finite_nonzero_task_gradient": finite and bool(task_steps),
        "no_reference_forward_at_beta_0": all(no_reference_forward(r) for r in everything),
        "loss_type_dr_grpo_recorded": all(dr_grpo_recorded(r) for r in everything),
        "scan_follows_train_manifest": bool(scanned) and scanned == scan_ids[: len(scanned)]
        and len(groups) == steps
        and [e["id"] for e in base["screen"] if e["accepted"]] == base["exposure"]
        and all((e["u"] >= 0.5) == e["accepted"] for e in base["screen"]),
        "initial_adapter_is_w": all(
            ((r.get("settings") or {}).get("init_adapter") or {}).get("weights_sha256")
            == w_sha for r in records + records_of(probe_run)),
        "useful_mixed_group": any(g.get("recognition_mixed") is True for g in groups),
        "lora_parameter_changed": base["adapter_sha256"] != w_sha
        and (base["record"].get("optimizer_audit") or {}).get("changed_tensors", 0) > 0,
        "scorer_uses_sampler_transform": isinstance(transform.get("suppress_tokens"), list)
        and len(transform["suppress_tokens"]) == 1 and bool(transform.get("eos_token_ids"))
        and transform.get("temperature") == profile["temperature"]
        and all((r["record"].get("sampler_teacher_logprob_parity") or {}).get(
            "sampling_suppressed_token_ids") == transform["suppress_tokens"]
            for r in everything),
        "active_time_ledger_cumulative": all(ledger_cumulative(r, steps) for r in every)
        and ledger_cumulative(probe_run, probe_run["global_step"]),
        "scorer_seconds_within_active_time": all(E.scorer_within_active(r)
                                                 for r in everything),
        "group_size_is_g": all(group_is(r, group, steps) for r in every)
        and group_is(probe_run, group, probe_run["global_step"]),
    }
    return checks, {"task_gradient_steps": task_steps, "gradient_norms": norms}


def parity_errors(runs):
    return [(r["record"].get("sampler_teacher_logprob_parity") or {}).get(
        "max_abs_logprob_error_after_same_suppression") for r in runs]


def measurements(base, probe_run):
    rows = base["audit_rows"]
    seconds = [row.get("completed_state_active_seconds") for row in rows]
    return {
        "probe_committed_active_seconds": probe_run["receipt"].get("committed_active_seconds"),
        "probe_committed_scorer_seconds": probe_run["committed_scorer_seconds"],
        "probe_global_step": probe_run["global_step"],
        "probe_generated_tokens": sum(sum(g.get("completion_tokens", []))
                                      for g in probe_run["groups"]),
        "update_committed_active_seconds": dict(zip([str(r["step"]) for r in rows], seconds)),
        "update_seconds": [b - a if number(a) and number(b) else None
                           for a, b in zip([0.0, *seconds], seconds)],
        "generated_tokens": sum(sum(g.get("completion_tokens", [])) for g in base["groups"]),
        "checkpoint_bytes": base["checkpoint_bytes"],
        "probe_checkpoint_bytes": probe_run["checkpoint_bytes"],
    }


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def group_list(text):
    values = [int(part) for part in text.split(",")]
    if not values or len(set(values)) != len(values) or any(v < 2 for v in values):
        raise argparse.ArgumentTypeError("groups must be distinct integers >= 2")
    return values


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("trainer arguments must follow a literal --")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--trainer-sha256", required=True)
    parser.add_argument("--groups", type=group_list, default=list(STUDY_GROUPS))
    parser.add_argument("--e0-steps", type=positive_int, default=STUDY_E0_STEPS)
    parser.add_argument("--probe-accepted", type=positive_int, default=STUDY_PROBE_ACCEPTED)
    args = parser.parse_args(argv[:split])
    if args.e0_steps < 2:
        raise SystemExit("E0 needs at least 2 steps to resume across a boundary")
    common = argv[split + 1:]
    if any(a == flag or a.startswith(flag + "=") for a in common for flag in OWNED):
        raise SystemExit(f"the Stage 0 entry owns {', '.join(OWNED)}")
    for flag in REQUIRED:
        E.single(common, flag)
    if E.single(common, "--mode") != "grpo" or "--train-only" not in common:
        raise SystemExit("Stage 0 is a TRAIN-only screened GRPO run")
    try:
        beta = float(E.single(common, "--beta"))
    except ValueError:
        beta = None
    if beta != 0:
        raise SystemExit("Stage 0 runs at beta 0")
    if E.single(common, "--loss-type") != "dr_grpo" or E.single(
            common, "--scale-rewards") != "none":
        raise SystemExit("Stage 0 runs dr_grpo with scale_rewards none")
    w_sha = E.file_sha256(Path(E.single(common, "--init-adapter")) / "adapter_model.safetensors")
    scan_ids = [row["id"] for row in E.read_jsonl(E.single(common, "--train"))]
    stage = args.stage_dir.resolve()
    run = Runner(stage, args.trainer_sha256)
    parts = splits(args.e0_steps)

    checks, costs, counts, detail, measured, parity = {}, {}, {}, {}, {}, []
    for group in args.groups:
        root = stage / f"g{group}"
        root.mkdir(parents=True, exist_ok=True)
        first = len(run.invocations)
        probe_run = probe(run, common, group, args.probe_accepted, root / "probe")
        runs = {name: trajectory(run, common, group, args.e0_steps, name, p, root / name)
                for name, p in parts.items()}
        key = str(group)
        checks[key], detail[key] = evaluate(runs, probe_run, group, args.e0_steps, w_sha,
                                            scan_ids)
        costs[key], counts[key], reason = cost(probe_run, args.probe_accepted)
        detail[key]["c_refused"] = reason
        parity += parity_errors([probe_run, *runs.values()])
        measured[key] = {"invocations": run.invocations[first:],
                         **measurements(runs["uninterrupted"], probe_run)}
        detail[key]["trajectories"] = {
            name: {"directory": r["directory"], "global_step": r["global_step"],
                   "adapter_sha256": r["adapter_sha256"],
                   **{f"{k}_sha256": E.digest(r[k]) for k in ("screen", "reward_groups",
                                                               "exposure")}}
            for name, r in {"probe": probe_run, **runs}.items()}
    study = (args.groups == list(STUDY_GROUPS) and args.e0_steps == STUDY_E0_STEPS
             and args.probe_accepted == STUDY_PROBE_ACCEPTED)
    fragment = {
        "scope": "signal-limit study Stage 0 at W: c(G) cost probe and beta-0 dr_grpo E0",
        "study_parameters": study,
        "e0_groups": args.groups,
        "e0_steps": args.e0_steps,
        "e0_splits": parts,
        "probe_accepted": args.probe_accepted,
        "checks": checks,
        "c_accepted_steps": counts,
        "c_seconds_per_accepted_update": costs,
        "sampler_teacher_max_abs_logprob_difference": max(parity)
        if parity and all(number(e) for e in parity) else None,
        "detail": detail,
        "measurements": measured,
    }
    temporary = args.fragment.with_suffix(".tmp")
    temporary.write_text(json.dumps(fragment, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
