"""P0 -> G1: score the first 400 S_scan_seed42 candidates under W and sample G4 groups.

Launcher entry ``p0_screen:main`` (scripts/freshpool_launch/launch.py, template ``p0``).
Usage, after the phase worker's ``--``:

  --scan-manifest S_scan_seed42.jsonl --adapter W_ADAPTER_DIR --fragment F --stage-dir DIR
  --data-spec SPEC --model-directory MODEL [--device mps] [profile options]

study.json gate G1 ("seed 42 at W"): "score first 400 S candidates; G4 at seeds 42,314 on 64
of them". Pre-registered here:
  * candidates: the first 400 rows of the scan manifest, in its order (the manifest IS the
    random.Random(42).shuffle(sorted(ids)) order; the launcher checks that at G0);
  * rollout images: the first 64 of those 400, every one regardless of acceptance (the
    evaluator's reading of "64 of them");
  * p: P(exact gold + EOS), one teacher-forced forward under W with the sampler's exact
    transforms (temperature, image-token suppression, EOS set of
    ``_task_group_generation_config``), via the trainer's ``_batch_gold_probability`` --
    the same function its screened scan uses. No alternate tokenizations: P0 MEASURES the
    non-canonical share that decides the study's alt-tokenization switch;
  * scorer_seconds: that forward only, MPS-synchronized on both sides;
  * u, accepted: ``group_useful_probability(p, 4)`` and u >= 0.5;
  * each sampled completion: eos = ended at EOS; valid = ended at EOS and parses to a
    vocabulary label; correct = valid and equal to the gold label (exact-EOS canonical, as
    the GRPO reward); canonical_tokenization = its token IDs equal the gold+EOS sequence the
    scorer used for that image.
Sampling reuses ``sample_train_groups`` unchanged, in chunks of at most 32 rows (its limit),
each chunk seeded with ``transformers.set_seed(seed)`` at its start.

The fragment is the evaluator's ``g1`` document: seed, adapter_sha256 (W's weights file
SHA256), scan_order (every manifest ID), journal (400 entries) and rollouts (128 groups).
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import h_assay
from h_assay import CHUNK, PROFILE, file_sha256, read_rows

from vision_rl import vlm_local_training as vlt

SEED = 42
CANDIDATES, ROLLOUT_IMAGES, ROLLOUT_SEEDS = 400, 64, (42, 314)
THRESHOLD = 0.5
GROUP = 4


def score_candidates(rows, *, adapter, data_spec, model_directory, device, profile,
                     model=vlt.DEFAULT_MODEL):
    """[(p, scorer_seconds, canonical gold+EOS token IDs)] for each row, under ``adapter``."""
    torch = import_module("torch")
    transformers = import_module("transformers")
    spec = vlt.load_spec(Path(data_spec))
    vlt.preflight(device)
    model_directory = Path(model_directory).resolve()
    vlt._verified_model_directory(model_directory, model)
    processor = vlt._processor(model_directory, profile)
    policy = transformers.AutoModelForImageTextToText.from_pretrained(
        str(model_directory), trust_remote_code=False, local_files_only=True,
        use_safetensors=True, dtype=getattr(torch, profile.dtype), attn_implementation="sdpa")
    policy = import_module("peft").PeftModel.from_pretrained(policy, str(adapter),
                                                             is_trainable=False)
    policy.requires_grad_(False).to(device).eval()
    config = vlt._task_group_generation_config(policy, processor, profile)
    eos, suppress = sorted(config.eos_token_id), list(config.suppress_tokens or [])
    collator = import_module("trl.trainer.sft_trainer").DataCollatorForVisionLanguageModeling(
        processor, completion_only_loss=True, max_length=None)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    results = []
    try:
        for row in rows:
            record = vlt.training_records([row], spec, "sft", profile.response_format)[0]
            record["image"] = vlt._load_rgb(record["image"], spec)
            batch = {key: value.to(device) for key, value in collator([record]).items()}
            sync()
            started = time.monotonic()
            with torch.no_grad():
                p, canonical = vlt._batch_gold_probability(
                    policy, batch, eos_token_ids=eos, suppress_tokens=suppress,
                    temperature=profile.temperature)
            sync()
            results.append((float(p), time.monotonic() - started, list(canonical)))
    finally:
        del policy
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    return results


def sample_groups(rows, seed, stage_dir, *, adapter, data_spec, model_directory, device,
                  profile):
    """{id: [sampled completions]} from the trainer's own sampler, chunked."""
    found = {}
    for number in range(0, len(rows), CHUNK):
        chunk = rows[number:number + CHUNK]
        manifest = stage_dir / "chunks" / f"rollout-seed{seed}-{number // CHUNK:02d}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in chunk))
        output = stage_dir / f"rollout-seed{seed}-{number // CHUNK:02d}"
        record = vlt.sample_train_groups(
            manifest, output=output, seed=seed, data_spec=Path(data_spec),
            model_directory=Path(model_directory), device=device, profile=profile,
            init_adapter=Path(adapter))
        if record.get("status") != "completed":
            raise RuntimeError(f"rollout chunk seed {seed} #{number // CHUNK} did not complete")
        if (record.get("initial_adapter") or {}).get("weights_sha256") != file_sha256(
                Path(adapter) / "adapter_model.safetensors"):
            raise RuntimeError("rollouts ran under a different adapter than W")
        found.update(json.loads((output / "sampled_completions.json").read_text()))
    return found


def g1_document(rows, scores, samples, adapter_sha, labels):
    candidates = rows[:CANDIDATES]
    journal = []
    for row, (p, seconds, _) in zip(candidates, scores, strict=True):
        u = vlt.group_useful_probability(p, GROUP)
        journal.append({"id": row["id"], "p": p, "u": u, "accepted": u >= THRESHOLD,
                        "scorer_seconds": seconds, "adapter_sha256": adapter_sha})
    canonical = {row["id"]: score[2] for row, score in zip(candidates, scores, strict=True)}
    rollouts = []
    for row in candidates[:ROLLOUT_IMAGES]:
        for seed in ROLLOUT_SEEDS:
            group = samples[seed].get(row["id"], [])
            if len(group) != GROUP:
                raise RuntimeError(f"{row['id']} seed {seed}: {len(group)} completions")
            completions = []
            for completion in group:
                value, ended = h_assay.answer(completion, labels)
                completions.append({
                    "correct": value == row["label"], "valid": value is not None,
                    "eos": ended,
                    "canonical_tokenization": list(completion.get("completion_ids", []))
                    == canonical[row["id"]]})
            rollouts.append({"id": row["id"], "seed": seed, "completions": completions})
    return {"seed": SEED, "adapter_sha256": adapter_sha,
            "scan_order": [row["id"] for row in rows], "journal": journal,
            "rollouts": rollouts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scan-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--data-spec", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    for name, value in asdict(PROFILE).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    profile = vlt.Profile(**{key: args.pop(key) for key in asdict(PROFILE)})
    if profile.num_generations != GROUP:
        raise SystemExit("study.json fixes G = 4")
    rows = read_rows(args["scan_manifest"])
    if len(rows) < CANDIDATES:
        raise SystemExit(f"scan manifest holds {len(rows)} rows, fewer than {CANDIDATES}")
    labels = vlt.load_spec(args["data_spec"])["labels"]
    adapter_sha = file_sha256(args["adapter"] / "adapter_model.safetensors")
    common = {"adapter": args["adapter"], "data_spec": args["data_spec"],
              "model_directory": args["model_directory"], "device": args["device"],
              "profile": profile}
    scores = score_candidates(rows[:CANDIDATES], **common)
    if len(scores) != CANDIDATES:
        raise RuntimeError("scorer did not return one result per candidate")
    samples = {seed: sample_groups(rows[:ROLLOUT_IMAGES], seed, args["stage_dir"], **common)
               for seed in ROLLOUT_SEEDS}
    document = g1_document(rows, scores, samples, adapter_sha, labels)
    temporary = args["fragment"].with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(args["fragment"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
