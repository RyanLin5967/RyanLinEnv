"""H-select / H-gate assay of one endpoint: greedy plus G4 sampled groups on every image.

Launcher entry ``h_assay:main`` (scripts/freshpool_launch/launch.py, template ``assay``).
Usage, after the phase worker's ``--``:

  --run-id ID --split {h_select,h_gate} --manifest H.jsonl --adapter ADAPTER_DIR
  --fragment F --stage-dir DIR --data-spec SPEC --model-directory MODEL [--device mps]
  [profile options, default answer-only as study.json fixed_profile]

Completions per image (study.json; scripts/evaluate_freshpool_gates.py COMPLETIONS):
  h_select: greedy + one G4 group at seed 42           = 5
  h_gate:   greedy + G4 groups at seeds 42 and 314     = 9
in that order. Every completion is scored the way the GRPO reward scores it (exact-EOS
canonical): the recorded answer is ``parse_label`` of the response only when the completion
ended at EOS, else null (invalid).

Generation reuses vision_rl.vlm_local_training's own ``predict_train_subset`` (greedy) and
``sample_train_groups`` (G4, released-TRL sampler settings), unchanged. Those refuse more
than 32 images per call, so the split is assayed in consecutive chunks of at most 32 rows in
manifest order, each in its own output directory. Each sampled chunk calls
``transformers.set_seed(seed)`` at its start: draws are deterministic given the manifest and
the chunking, which is fixed here (CHUNK) and recorded in the fragment.

The fragment holds ``predictions`` {id: [answer|null] * completions}, ``eos_fraction`` over
every completion, and the assayed adapter's weights SHA256, which the launcher checks against
the endpoint's. A chunk that returns anything but a completed record with one prediction per
row fails the stage: nothing is padded or guessed.

Signal-limit v1 (docs/aircraft-rl-next-study-v1.json fixed_profile.assay and
secondary_decoder; implementation_work_list h_assay):
  * the assay's G is ASSAY_GROUP = 4 whatever the training G; ``--training-group G``
    (4, 8 or 16; 0 for an endpoint that is not RL, recorded as null) only records the
    endpoint's training G in the fragment (``training_group``, with ``assay_group`` 4);
  * ``--label-trie T --label-trie-sha256 H`` adds a label-trie constrained greedy pass per
    chunk, a format-separation diagnostic that is never the primary evaluator: the fragment
    gains ``constrained_greedy`` {trie_sha256, predictions {id: answer|null}, eos_fraction},
    scored exactly like the unconstrained greedy pass. The trie file must hash to H, and
    vision_rl.vlm_local_training.predict_train_subset must accept ``label_trie`` (the
    label_trie_logits_processor work item); otherwise the assay refuses before any model call.
Without those options the fragment is byte-identical to the freshpool one.
"""

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

from vision_rl import vlm_local_training as vlt

CHUNK = 32  # vlm_local_training refuses TRAIN inference on more than 32 images per call
SEEDS = {"h_select": (42,), "h_gate": (42, 314)}
COMPLETIONS = {"h_select": 5, "h_gate": 9}
PROFILE = vlt.Profile(response_format="answer_only")  # study.json fixed_profile prompt
ASSAY_GROUP = 4  # signal-limit fixed_profile.assay: "assay G stays 4 regardless of training G"
TRAINING_GROUPS = (0, 4, 8, 16)  # 0: not an RL endpoint (no training G); recorded as null


def constrained_predict(manifest, *, label_trie, **kwargs):
    """Greedy decoding restricted to the label trie, through the trainer's own TRAIN-subset
    inference (one call per chunk, like the unconstrained greedy pass)."""
    return vlt.predict_train_subset(manifest, label_trie=label_trie, **kwargs)


def constrained_supported():
    return "label_trie" in inspect.signature(vlt.predict_train_subset).parameters


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    path = Path(path).resolve()
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            # Trainer manifests carry exactly id, split, image, label; images resolve
            # against the manifest's directory, so make them absolute for the chunk files.
            rows.append({"id": row["id"], "split": row["split"], "label": row["label"],
                         "image": str((path.parent / row["image"]).resolve())})
    if len({row["id"] for row in rows}) != len(rows) or not rows:
        raise ValueError("assay manifest must hold unique, nonempty rows")
    return rows


def answer(completion, labels):
    """Exact-EOS canonical scoring, as the GRPO reward: a label only if terminated."""
    if not completion.get("terminated"):
        return None, False
    return vlt.parse_label(completion.get("response", ""), labels, allow_reasoning=True), True


def assay(rows, split, adapter, stage_dir, *, data_spec, model_directory, device, profile,
          label_trie=None):
    labels = vlt.load_spec(Path(data_spec))["labels"]
    adapter_sha = file_sha256(Path(adapter) / "adapter_model.safetensors")
    answers = {row["id"]: [] for row in rows}
    terminated = total = 0
    common = {"data_spec": Path(data_spec), "model_directory": Path(model_directory),
              "device": device, "profile": profile, "init_adapter": Path(adapter)}
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    constrained = {} if label_trie is not None else None
    constrained_terminated = 0
    for number, chunk in enumerate(chunks):
        manifest = stage_dir / "chunks" / f"chunk-{number:02d}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in chunk))
        if label_trie is not None:
            output = stage_dir / f"constrained-{number:02d}"
            record = constrained_predict(manifest, output=output, label_trie=Path(label_trie),
                                         **common)
            if record.get("status") != "completed":
                raise RuntimeError(f"constrained greedy chunk {number} did not complete")
            if (record.get("initial_adapter") or {}).get("weights_sha256") != adapter_sha:
                raise RuntimeError("generation ran under a different adapter than assayed")
            found = json.loads((output / "predictions.json").read_text())
            for row in chunk:
                if row["id"] not in found:
                    raise RuntimeError(f"{row['id']}: no constrained greedy completion")
                value, ended = answer(found[row["id"]], labels)
                constrained[row["id"]] = value
                constrained_terminated += ended
        passes = [("greedy", None), *(("seed", seed) for seed in SEEDS[split])]
        for kind, seed in passes:
            output = stage_dir / (f"greedy-{number:02d}" if seed is None
                                  else f"seed{seed}-{number:02d}")
            if seed is None:
                record = vlt.predict_train_subset(manifest, output=output, **common)
                found = json.loads((output / "predictions.json").read_text())
                per_row = {i: [found[i]] if i in found else [] for i in answers}
            else:
                record = vlt.sample_train_groups(manifest, output=output, seed=seed, **common)
                found = json.loads((output / "sampled_completions.json").read_text())
                per_row = {i: found.get(i, []) for i in answers}
            if record.get("status") != "completed":
                raise RuntimeError(f"{kind} chunk {number} did not complete")
            initial = record.get("initial_adapter") or {}
            if initial.get("weights_sha256") != adapter_sha:
                raise RuntimeError("generation ran under a different adapter than assayed")
            expected = 1 if seed is None else profile.num_generations
            for row in chunk:
                completions = per_row[row["id"]]
                if len(completions) != expected:
                    raise RuntimeError(f"{row['id']}: {len(completions)} completions, "
                                       f"expected {expected}")
                for completion in completions:
                    value, ended = answer(completion, labels)
                    answers[row["id"]].append(value)
                    terminated += ended
                    total += 1
    if any(len(v) != COMPLETIONS[split] for v in answers.values()):
        raise RuntimeError("assay did not produce the declared completions per image")
    result = {"predictions": answers, "eos_fraction": terminated / total,
              "adapter_sha256": adapter_sha, "chunks": len(chunks), "chunk_size": CHUNK}
    if constrained is not None:
        if set(constrained) != set(answers):
            raise RuntimeError("constrained greedy did not cover every image")
        result["constrained_greedy"] = {
            "trie_sha256": file_sha256(label_trie), "predictions": constrained,
            "eos_fraction": constrained_terminated / len(constrained)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=tuple(SEEDS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--data-spec", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--training-group", type=int, choices=TRAINING_GROUPS)
    parser.add_argument("--label-trie", type=Path)
    parser.add_argument("--label-trie-sha256")
    for name, value in asdict(PROFILE).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    profile = vlt.Profile(**{key: args.pop(key) for key in asdict(PROFILE)})
    if profile.num_generations != ASSAY_GROUP:
        raise SystemExit("study.json fixes G = 4")
    trie, trie_sha = args["label_trie"], args["label_trie_sha256"]
    if (trie is None) != (trie_sha is None):
        raise SystemExit("--label-trie and --label-trie-sha256 go together")
    if trie is not None:
        if not trie.is_file() or file_sha256(trie) != trie_sha:
            raise SystemExit("label trie is missing or does not hash to --label-trie-sha256")
        if not constrained_supported():
            raise SystemExit("vision_rl.vlm_local_training.predict_train_subset has no "
                             "label_trie parameter: constrained greedy is not available")
    rows = read_rows(args["manifest"])
    result = assay(rows, args["split"], args["adapter"], args["stage_dir"],
                   data_spec=args["data_spec"], model_directory=args["model_directory"],
                   device=args["device"], profile=profile, label_trie=trie)
    fragment = {"run_id": args["run_id"], "split": args["split"],
                "manifest_sha256": file_sha256(args["manifest"]),
                "seeds": list(SEEDS[args["split"]]), "completion_order":
                "greedy, then each seed's G4 group in seed order", **result}
    if args["training_group"] is not None:
        fragment.update(training_group=args["training_group"] or None, assay_group=ASSAY_GROUP)
    temporary = args["fragment"].with_suffix(".tmp")
    temporary.write_text(json.dumps(fragment, indent=2, allow_nan=False) + "\n")
    temporary.replace(args["fragment"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
