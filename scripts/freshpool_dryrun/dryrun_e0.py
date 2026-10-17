"""Dry-run E0 entry: screened-GRPO resume identity through the trainer's own CLI.

Launcher entry ``dryrun_e0:main`` for the freshpool CPU dry run (run.py) ONLY. The production
E0 entry is still an open dependency (scripts/freshpool_launch/README.md). This file is not
it: nobody has reviewed it against the protocol's real_learning_and_resume_gate, and it runs on
whatever device its trainer arguments name.

Usage, after the phase worker's ``--``:

  --fragment F --stage-dir DIR --steps N -- <vision_rl.vlm_local_training CLI arguments for
  a screened GRPO run, without the flags in OWNED>

Three trajectories of N accepted updates, each in its own new directory under DIR:
uninterrupted, 2 + (N - 2) and 1 + (N - 1), the split ones stopping at the boundary with
``--stop-after-steps`` and resuming from that checkpoint with ``--resume-from-checkpoint``.
Each is read back only through verified outputs (``read_endpoint_receipt``,
``committed_screen_entries``) plus its reward journal, and compared on: global step, final
adapter bytes, the committed screen journal (scan position, ID, p, u, accepted, adapter hash,
step), the reward groups (ID, rewards, completion token IDs) and the committed prompt exposure.

Fragment: the launcher's E0 ``checks`` and ``sampler_teacher_max_abs_logprob_difference`` (the
trainer's own sampler/teacher parity, recorded by the uninterrupted run), plus per-trajectory
digests. A failed check is written as false, never raised, so the launcher's E0 validator is
what stops the study.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from vision_rl import vlm_local_training as vlt

TRAINER = vlt.main
OWNED = ("--output", "--max-steps", "--target-accepted", "--screen-journal",
         "--stop-after-steps", "--resume-from-checkpoint")
SCREEN_KEYS = ("scan_position", "id", "p", "u", "accepted", "adapter_sha256", "global_step")
GROUP_KEYS = ("id", "rewards", "completion_ids")
COMPARED = ("global_step", "adapter_sha256", "screen", "reward_groups", "exposure")


class E0Error(RuntimeError):
    """A trajectory did not complete through the trainer."""


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def trajectory(common, steps, splits, directory):
    done = 0
    for number, part in enumerate(splits):
        target = done + part
        argv = [*common, "--max-steps", str(steps), "--target-accepted", str(steps),
                "--screen-journal", str(directory / "screen.jsonl"), "--output", str(directory)]
        if target < steps:
            argv += ["--stop-after-steps", str(target)]
        if number:
            argv += ["--resume-from-checkpoint", str(directory / f"checkpoint-{done}")]
        code = TRAINER(argv)
        if code not in (0, None):
            raise E0Error(f"trainer exited {code} on {directory.name} part {number + 1}")
        done = target
    receipt = vlt.read_endpoint_receipt(directory)
    record = json.loads((directory / "local_training.json").read_text())
    checkpoint = directory / receipt["checkpoint"]["path"]
    return {
        "directory": str(directory),
        "global_step": receipt["global_step"],
        "adapter_sha256": receipt["final_adapter"]["weights_sha256"],
        "screen": [{k: e[k] for k in SCREEN_KEYS}
                   for e in vlt.committed_screen_entries(checkpoint)],
        "reward_groups": [{k: g[k] for k in GROUP_KEYS}
                          for g in read_jsonl(directory / "reward_groups.jsonl")],
        "exposure": receipt["committed_prompt_ids"],
        "record": record,
    }


def identical(first, second):
    return all(first[key] == second[key] for key in COMPARED)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("trainer arguments must follow a literal --")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv[:split])
    common = argv[split + 1:]
    if any(a == flag or a.startswith(flag + "=") for a in common for flag in OWNED):
        raise SystemExit(f"the E0 entry owns {', '.join(OWNED)}")
    if args.steps < 3:
        raise SystemExit("E0 needs at least 3 updates for its 2 + rest and 1 + rest splits")
    runs = {}
    for name, splits in (("uninterrupted", [args.steps]),
                         ("resume_2_plus_rest", [2, args.steps - 2]),
                         ("resume_1_plus_rest", [1, args.steps - 1])):
        runs[name] = trajectory(common, args.steps, splits, args.stage_dir / name)
    base = runs["uninterrupted"]
    record = base["record"]
    audit = record.get("optimizer_audit") or {}
    references = [run["record"].get("reference") for run in runs.values()]
    transform = record.get("screen_sampler_transform") or {}
    parity = record.get("sampler_teacher_logprob_parity") or {}
    norm = audit.get("max_gradient_norm_observed")
    checks = {
        "resume_2_plus_2_identical": identical(base, runs["resume_2_plus_rest"]),
        "resume_1_plus_3_identical": identical(base, runs["resume_1_plus_rest"]),
        "reference_unchanged": references[0] is not None
        and all(reference == references[0] for reference in references)
        and all((run["record"].get("optimizer_audit") or {}).get("reference_unchanged") is True
                for run in runs.values()),
        "finite_nonzero_task_gradient": (audit.get("nonzero_gradient_steps") or 0) >= 1
        and isinstance(norm, (int, float)) and math.isfinite(norm) and norm > 0,
        "scorer_uses_sampler_transform": bool(transform.get("suppress_tokens"))
        and transform.get("temperature") == record["control_identity"]["profile"]["temperature"],
    }
    fragment = {
        "scope": "dry-run E0 on the tiny random model; not the production E0 entry",
        "steps": args.steps,
        "checks": checks,
        "sampler_teacher_max_abs_logprob_difference": parity.get(
            "max_abs_logprob_error_after_same_suppression"),
        "trajectories": {
            name: {"directory": run["directory"], "global_step": run["global_step"],
                   "adapter_sha256": run["adapter_sha256"],
                   **{f"{key}_sha256": digest(run[key])
                      for key in ("screen", "reward_groups", "exposure")}}
            for name, run in runs.items()},
    }
    temporary = args.fragment.with_suffix(".tmp")
    temporary.write_text(json.dumps(fragment, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
