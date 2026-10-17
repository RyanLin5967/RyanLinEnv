"""Run one freshpool training stage with the pinned trainer and write the launcher's fragment.

Launcher entry ``trainer_fragment:main`` (scripts/freshpool_launch/launch.py reads the
fragment). Usage, after the phase worker's ``--``:

  --run-id ID --arm {RL,SFT-E,SFT-T,EG-SFT} --lr LR --seed 42 --fragment F --output DIR
  [--scan-manifest S_scan_seed42.jsonl] [--continue-output-from PARENT_CHECKPOINT]
  -- <vision_rl.vlm_local_training CLI arguments, without --output>

The wrapper adds ``--output DIR`` to the trainer arguments and calls the trainer's own
``main``. ``--continue-output-from`` is the RL 150 -> 600 continuation (study.json arms
RL(lr): "continue (no reset)"): the parent's VALIDATED output directory is copied into the
fresh DIR and the trainer resumes the copied checkpoint there, so the parent endpoint that
G2 judged is never written to, and an interrupted continuation leaves nothing a re-run
could inherit.

After the trainer exits 0 the wrapper reads ONLY verified results:
``read_endpoint_receipt`` (adapter, checkpoint, exposure and active-time lineage) and, for
RL, ``committed_screen_entries`` of the receipt's checkpoint (never the physical journal).
Fragment fields:

  status, run_id, arm, lr, seed                  identity, checked against the trainer record
  updates, committed_active_seconds              receipt global_step and CUMULATIVE seconds
  step_committed_active_seconds                  {step: committed seconds} from
                                                 optimizer_audit.jsonl, last value per step;
                                                 its final entry must equal the receipt's
  adapter_sha256, adapter_path                   final_adapter weights file SHA256 and dir
  reference_sha256                               RL: the KL reference's initial adapter
                                                 weights SHA256; SFT arms: their initial
                                                 adapter's. Both must be W's.
  checkpoint                                     the receipt's checkpoint directory
  useful_mixed                                   RL: reward_groups.jsonl "mixed" per update,
                                                 whose IDs must equal the committed exposure
  accepted_manifest, scanned_manifest            RL: committed accepted IDs and every scanned
                                                 ID in scan order, as trainer-loadable rows
  receipt_status, requested_boundary             from the receipt

Blind spots, stated: the per-step seconds and the useful-mixed flags come from append-only
physical journals (optimizer_audit.jsonl, reward_groups.jsonl). They are exact only because
an interrupted stage is never resumed (it is re-run in a new directory), so no discarded
span can sit in them; a line count or ID mismatch refuses rather than guesses.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

from vision_rl import vlm_local_training as vlt

ARM_MODES = {"RL": "grpo", "SFT-E": "sft", "SFT-T": "sft", "EG-SFT": "egsft"}
# The trainer entry; module-level so tests substitute a fake, never selectable by a plan.
TRAINER = vlt.main


class FragmentError(RuntimeError):
    """The trainer's verified outputs do not support a fragment."""


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def manifest_rows(path):
    """Rows of a trainer manifest with image paths made absolute (load_classification
    resolves a relative image against the manifest's own directory)."""
    path = Path(path).resolve()
    rows = {}
    for row in read_jsonl(path):
        rows[row["id"]] = {**row, "image": str((path.parent / row["image"]).resolve())}
    return rows


def write_rows(path, ids, rows):
    missing = [i for i in ids if i not in rows]
    if missing:
        raise FragmentError(f"{len(missing)} IDs are not in the scan manifest: {missing[:3]}")
    Path(path).write_text("".join(json.dumps(rows[i], sort_keys=True) + "\n" for i in ids))


def step_seconds(output):
    path = output / "optimizer_audit.jsonl"
    if not path.is_file():
        raise FragmentError("trainer wrote no optimizer_audit.jsonl")
    steps = {}
    for entry in read_jsonl(path):
        value = entry.get("completed_state_active_seconds")
        if type(entry.get("step")) is not int or type(value) not in (int, float) or (
                not math.isfinite(value)):
            raise FragmentError("optimizer_audit.jsonl entry lacks step or committed seconds")
        steps[entry["step"]] = value
    return {str(step): steps[step] for step in sorted(steps)}


def build_fragment(args, output):
    receipt = vlt.read_endpoint_receipt(output)
    record = json.loads((output / "local_training.json").read_text())
    settings = record["settings"]
    if settings["mode"] != ARM_MODES[args.arm]:
        raise FragmentError(f"arm {args.arm} needs trainer mode {ARM_MODES[args.arm]}, "
                            f"trainer ran {settings['mode']}")
    if settings["learning_rate"] != args.lr or settings["seed"] != args.seed:
        raise FragmentError("trainer learning rate or seed differs from the stage identity")
    initial = settings.get("init_adapter") or {}
    if args.arm == "RL":
        reference = record.get("reference") or {}
        if receipt["reference_sha256"] is None or reference.get("kind") != (
                "initial_saved_adapter"):
            raise FragmentError("RL endpoint has no saved-adapter KL reference")
        reference_sha = reference["initial_adapter_weights_sha256"]
    else:
        reference_sha = initial.get("weights_sha256")
    if reference_sha is None or reference_sha != initial.get("weights_sha256"):
        raise FragmentError("reference is not the run's initial adapter")
    checkpoint = (output / receipt["checkpoint"]["path"]).resolve()
    steps = step_seconds(output)
    if steps.get(str(receipt["global_step"])) != receipt["committed_active_seconds"]:
        raise FragmentError("optimizer audit's final committed seconds differ from the receipt")
    fragment = {
        "status": "completed",
        "run_id": args.run_id,
        "arm": args.arm,
        "lr": args.lr,
        "seed": args.seed,
        "updates": receipt["global_step"],
        "committed_active_seconds": receipt["committed_active_seconds"],
        "step_committed_active_seconds": steps,
        "adapter_sha256": receipt["final_adapter"]["weights_sha256"],
        "adapter_path": str((output / receipt["final_adapter"]["path"]).resolve()),
        "reference_sha256": reference_sha,
        "checkpoint": str(checkpoint),
        "useful_mixed": [],
        "receipt_status": receipt["status"],
        "requested_boundary": receipt["requested_boundary"],
        "trainer_output": str(output),
    }
    if args.arm == "RL":
        if args.scan_manifest is None:
            raise FragmentError("RL fragments need --scan-manifest")
        groups = read_jsonl(output / "reward_groups.jsonl")
        if [group["id"] for group in groups] != receipt["committed_prompt_ids"]:
            raise FragmentError("reward groups differ from the committed prompt exposure")
        fragment["useful_mixed"] = [bool(group["mixed"]) for group in groups]
        entries = vlt.committed_screen_entries(checkpoint)
        accepted = [entry["id"] for entry in entries if entry["accepted"]]
        if accepted != receipt["committed_prompt_ids"]:
            raise FragmentError("committed screen accepted IDs differ from the exposure")
        rows = manifest_rows(args.scan_manifest)
        directory = Path(args.fragment).resolve().parent
        for key, ids in (("accepted_manifest", accepted),
                         ("scanned_manifest", [entry["id"] for entry in entries])):
            path = directory / f"{key}.jsonl"
            write_rows(path, ids, rows)
            fragment[key] = str(path)
    return fragment


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("trainer arguments must follow a literal --")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_MODES), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-manifest", type=Path)
    parser.add_argument("--continue-output-from", type=Path)
    args = parser.parse_args(argv[:split])
    trainer_args = argv[split + 1:]
    if any(a == "--output" or a.startswith("--output=") or a.startswith(
            "--resume-from-checkpoint") for a in trainer_args):
        raise SystemExit("the wrapper owns --output and --resume-from-checkpoint")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("trainer output must be a new directory (stages are never resumed)")
    if args.continue_output_from is not None:
        parent = args.continue_output_from.resolve()
        if not (parent.parent / "endpoint_receipt.json").is_file():
            raise SystemExit("continuation parent must be a finished endpoint with a receipt")
        vlt.read_endpoint_receipt(parent.parent)
        if output.exists():
            output.rmdir()
        shutil.copytree(parent.parent, output, symlinks=True)
        trainer_args += ["--resume-from-checkpoint", str(output / parent.name)]
    code = TRAINER([*trainer_args, "--output", str(output)])
    if code not in (0, None):
        return code
    try:
        fragment = build_fragment(args, output)
    except (FragmentError, vlt.LocalVLMError, KeyError, OSError, ValueError) as exc:
        print(f"trainer fragment: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    temporary = args.fragment.with_suffix(".tmp")
    temporary.write_text(json.dumps(fragment, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
