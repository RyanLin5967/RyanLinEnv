"""Freshpool CPU dry run: the real launch pipeline end to end on a tiny random Qwen2-VL.

Purpose: catch integration bugs between scripts/prepare_aircraft_freshpool.py,
scripts/freshpool_launch/ (launch.py, phase_worker.py, trainer_fragment.py, h_assay.py,
p0_screen.py), src/vision_rl/vlm_local_training.py and scripts/evaluate_freshpool_gates.py.
It says nothing about learning: the model is random, the images synthetic, the labels fake.

What is real: G0 manifests come from the preparer's own make_manifests / forced_fire /
write_outputs on synthetic images; launch.py (loaded here, constants scaled) runs G0 and every
stage through phase_worker.py; the stage entries are the real wrappers, which call the real
trainer CLI and inference functions; every gate and selection is the real evaluator's
(dryrun_evaluator.py scales only its update counts). Nothing is mocked.

Deviations, pre-registered in dryrun_scale.py and here:
  * counts: RL 150 -> 3 and 600 -> 6 updates (launcher, evaluator, templates); CAP_FACTOR
    1.25 -> 100 (the cap is launcher wall time, which process start-up dominates for a tiny
    arm); pool 2,919 -> 442 before the audit. P0 (400 candidates, 64 x 2 groups), H (2 per
    class) and the useful-mixed window (100) are NOT scaled;
  * device cpu: each plan carries the explicit marker "dry_run": true, which is the only way
    launch.py accepts ACCELERATE_TORCH_DEVICE=cpu (it requires mps otherwise); the launcher
    then passes --dry-run to phase_worker.py, which requires cpu, requires every forwarded
    --device to be cpu, and arms no MPS guard;
  * model: a 1-layer random Qwen2-VL with a 14-token word-level tokenizer, saved with a
    model-source manifest naming the pinned repository and revision so the trainer's snapshot
    check accepts it; its token embeddings are one-hot and its LM head a bigram table, so after
    "assistant" it answers alpha/bravo/charlie (~0.33 each) or almost never delta, then EOS.
    That puts p in the screening band for three classes and near 0 for the fourth, so the
    screen both accepts and rejects. LoRA learning on it is negligible by construction;
  * W: a 2-step SFT of that model on the 16-image train400 stand-in;
  * receipts: a byte copy of docs/aircraft-rl-protocol-freshpool-v1.json, a separately
    identified activated copy of it (the form launch.py requires) and a branch receipt,
    committed in a throwaway git repository; the decision record is the real one;
  * E0: the production entry scripts/freshpool_launch/e0_gate.py, unchanged (4 vs 2+2 vs 1+3
    plus the altered-reference resume), each trainer part through phase_worker.py --dry-run.

Two launches, each from its own plan and fresh state directory:
  drive  - a gate the evaluator fails is journalled (dryrun_gate_forced) and then passed with
           the evaluator's own gate result, so every later stage still runs. It must run every
           stage, then stop at the final check with final_development_check_failed and the
           final report's failed_at equal to the first forced gate (or complete, if none was
           forced). Each forced gate must be consistent: the first a real evaluated failure,
           every later one not evaluated because of it.
  strict - the launcher as shipped (scaled counts only). It must complete, or stop with
           gate_failed at a gate the evaluator's report itself names as failed_at, having
           completed exactly the stages before that gate.
Any other stop - a stage failure, a refused fragment, an unexpected exception - fails the dry
run. Exit 0 only when both pass; work/summary.json holds the verdicts.

Blind spots, stated: random-model gate outcomes exercise the stop path, not the pass path, of
any gate they fail; the forced pass is the dry run's, never the launcher's; selections among
tied H-select scores fall to active seconds, so the two launches may choose differently.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAUNCH_DIR = ROOT / "scripts" / "freshpool_launch"
REAL_EVALUATOR = ROOT / "scripts" / "evaluate_freshpool_gates.py"
EVALUATOR = HERE / "dryrun_evaluator.py"
TRAINER_SOURCE = ROOT / "src" / "vision_rl" / "vlm_local_training.py"
sys.path.insert(0, str(HERE))

import dryrun_scale as scale  # noqa: E402

ENVIRONMENT = {"PYTORCH_ENABLE_MPS_FALLBACK": "0", "ACCELERATE_TORCH_DEVICE": "cpu",
               "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(HERE)}

SEED = scale.SEED
LABELS = list(scale.LABELS)
PER_CLASS = 115
TRAIN400_PER_CLASS = 4
PROBE_OUTSIDE = ("alpha-004", "bravo-004")  # probe16 stand-in: two outside train400 ...
PROBE_INSIDE = ("charlie-000",)  # ... and one inside it
PLANTED = ("delta-114", "delta-000")  # pool image re-encoded from a train400 image
POOL_BEFORE_AUDIT = len(LABELS) * PER_CLASS - len(LABELS) * TRAIN400_PER_CLASS - len(
    PROBE_OUTSIDE)  # 460 - 16 - 2 = 442
VALIDATION_IMAGES = 8
GRID = (1e-5, 5e-5, 1e-4)
SPLIT_IMAGES = len(LABELS) * 2
PROFILE_ARGS = ["--response-format", "answer_only", "--min-pixels", "784", "--max-pixels",
                "784", "--max-completion-length", "4"]
# Bigram design (see the module docstring): logits after "assistant" and after a label.
LABEL_LOGIT, RARE_LOGIT, EOS_LOGIT, RESIDUAL_SCALE = 5.0, -2.0, 8.0, 0.05
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "<|video_pad|>"]
VOCAB = ["[PAD]", "[UNK]", *SPECIALS, "user", "assistant", *LABELS]
TOKEN = {token: index for index, token in enumerate(VOCAB)}
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
    "{% for item in message['content'] %}{% if item['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif item['type'] == 'text' %}"
    "{{ item['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
GiB = 1024**3


class DryRunError(RuntimeError):
    """The fixture could not be built as designed."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(directory, *args):
    subprocess.run(["git", "-C", str(directory), "-c", "user.name=freshpool-dryrun",
                    "-c", "user.email=freshpool-dryrun@example.invalid", *args],
                   check=True, capture_output=True, timeout=60)


# ---------------------------------------------------------------- fixture


def build_images(root):
    """460 native-TRAIN stand-ins (115 per class) and 8 validation references: random
    16x16 colour fields upsampled to 64x64; every 8x8 DCT frequency the pHash reads carries
    energy, so hashes of distinct images are far apart and a re-encoded copy stays close."""
    import numpy as np
    from PIL import Image

    images = root / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(SEED)

    def make(name):
        base = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(base).resize((64, 64), Image.Resampling.BICUBIC).save(
            images / f"{name}.png")

    rows = []
    for label in LABELS:
        for n in range(PER_CLASS):
            image_id = f"{label}-{n:03d}"
            make(image_id)
            rows.append({"id": image_id, "split": "train", "label": label,
                         "image": f"../images/{image_id}.png"})
    for n in range(VALIDATION_IMAGES):
        make(f"validation-{n}")
    copy, source = PLANTED
    with Image.open(images / f"{source}.png") as original:
        original.convert("RGB").resize((60, 60), Image.Resampling.BILINEAR).save(
            images / f"{copy}.png")
    return images, rows


def build_manifests(root, images, rows, protocol):
    """G0 manifests through the real preparer (no FGVC data, no real-data contract)."""
    from PIL import Image

    preparer = load_module("prepare_aircraft_freshpool_dryrun",
                           ROOT / "scripts" / "prepare_aircraft_freshpool.py")

    def load(image_id):
        with Image.open(images / f"{image_id}.png") as image:
            return image.convert("RGB")

    train400 = [r for r in rows if int(r["id"][-3:]) < TRAIN400_PER_CLASS]
    by_id = {r["id"]: r for r in rows}
    probe = [{"id": i, "label": by_id[i]["label"]} for i in PROBE_OUTSIDE + PROBE_INSIDE]
    pool, counts = preparer.build_pool(rows, train400, probe)
    if counts["pool_before_audit"] != POOL_BEFORE_AUDIT:
        raise DryRunError(f"pool before audit {counts['pool_before_audit']}, "
                          f"designed {POOL_BEFORE_AUDIT}")
    pool_hashes = {r["id"]: preparer.phash64(load(r["id"])) for r in pool}
    references = {
        "validation": {f"validation-{n}": preparer.phash64(load(f"validation-{n}"))
                       for n in range(VALIDATION_IMAGES)},
        "train400": {r["id"]: preparer.phash64(load(r["id"])) for r in train400},
    }
    hash_protocol_id = preparer.holdout_salt(protocol)
    result = preparer.make_manifests(rows, train400, probe, hash_protocol_id, pool_hashes,
                                     references)
    forced = preparer.forced_fire(pool, pool_hashes, load)
    manifest = preparer.write_outputs(root / "manifests", result, forced, {
        "protocol_id": protocol["protocol_id"], "hash_protocol_id": hash_protocol_id,
        "g0_audit_complete": True,
        "fixture": "freshpool CPU dry run: synthetic images, not FGVC-Aircraft"})
    write_jsonl(root / "data" / "train400.jsonl", train400)
    return manifest, result


def build_model(directory):
    """The tiny Qwen2-VL and its processor, saved as a pinned-snapshot look-alike."""
    import torch
    from safetensors import safe_open
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import (
        GenerationConfig,
        PreTrainedTokenizerFast,
        Qwen2VLConfig,
        Qwen2VLForConditionalGeneration,
        Qwen2VLImageProcessor,
        Qwen2VLProcessor,
        Qwen2VLVideoProcessor,
        set_seed,
    )

    from vision_rl import vlm_local_training as vlt

    backend = Tokenizer(models.WordLevel(TOKEN, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.add_special_tokens([AddedToken(t, special=True, normalized=False)
                                for t in SPECIALS + ["[PAD]", "[UNK]"]])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="<|im_end|>",
        padding_side="left", extra_special_tokens=SPECIALS)
    processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=784, max_pixels=784, patch_size=14,
                                              temporal_patch_size=1, merge_size=1),
        tokenizer=tokenizer, video_processor=Qwen2VLVideoProcessor(),
        chat_template=CHAT_TEMPLATE)
    hidden = 24
    config = Qwen2VLConfig(
        text_config={"vocab_size": len(VOCAB), "hidden_size": hidden, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 3,
                     "num_key_value_heads": 3, "max_position_embeddings": 512,
                     "eos_token_id": TOKEN["<|im_end|>"], "pad_token_id": TOKEN["[PAD]"],
                     "tie_word_embeddings": False,
                     "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "embed_dim": 16, "hidden_size": hidden, "mlp_ratio": 2,
                       "num_heads": 2, "patch_size": 14, "temporal_patch_size": 1,
                       "spatial_merge_size": 1},
        image_token_id=TOKEN["<|image_pad|>"], video_token_id=TOKEN["<|video_pad|>"],
        vision_start_token_id=TOKEN["<|vision_start|>"],
        vision_end_token_id=TOKEN["<|vision_end|>"], tie_word_embeddings=False)
    set_seed(SEED)
    model = Qwen2VLForConditionalGeneration(config).eval()
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    if embed.data_ptr() == head.data_ptr():
        raise DryRunError("tiny model ties its LM head to the embeddings; bigram needs both")
    with torch.no_grad():
        embed.zero_()
        for index in range(len(VOCAB)):
            embed[index, index] = 1.0
        head.zero_()
        scale_ = hidden ** -0.5  # final RMSNorm maps a one-hot residual to sqrt(hidden)
        for label in LABELS:
            logit = RARE_LOGIT if label == LABELS[-1] else LABEL_LOGIT
            head[TOKEN[label], TOKEN["assistant"]] = logit * scale_
            head[TOKEN["<|im_end|>"], TOKEN[label]] = EOS_LOGIT * scale_
        for name, module in model.named_modules():
            if "visual" not in name and name.endswith(("o_proj", "down_proj")):
                module.weight.mul_(RESIDUAL_SCALE)
    model.generation_config = GenerationConfig(
        eos_token_id=TOKEN["<|im_end|>"], pad_token_id=TOKEN["[PAD]"])
    directory.mkdir(parents=True)
    model.save_pretrained(str(directory))
    processor.save_pretrained(str(directory))
    weights = directory / "model.safetensors"
    with safe_open(str(weights), "pt") as handle:
        keys = sorted(handle.keys())
    write_json(directory / "model.safetensors.index.json", {
        "metadata": {"total_size": weights.stat().st_size},
        "weight_map": {key: "model.safetensors" for key in keys}})
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    source = {
        "model_repository": vlt.DEFAULT_MODEL, "revision": vlt.MODEL_REVISIONS[vlt.DEFAULT_MODEL],
        "fixture": "freshpool CPU dry run: a RANDOM tiny Qwen2-VL, not the pinned weights",
        "files": [{"path": str(p.relative_to(directory)), "size_bytes": p.stat().st_size,
                   "sha256": sha(p)} for p in files]}
    write_json(directory / "model-source.json", source)
    write_json(directory / ".pinned-download.json", {
        "repository": vlt.DEFAULT_MODEL, "revision": vlt.MODEL_REVISIONS[vlt.DEFAULT_MODEL],
        "manifest_sha256": sha(directory / "model-source.json"),
        "fixture": "freshpool CPU dry run only"})


def check_model(directory, spec_path, rows_by_label):
    """Fail fast, with the numbers, if the saved model or processor is not as designed."""
    import torch
    import transformers

    from vision_rl import vlm_local_training as vlt

    profile = vlt.Profile(response_format="answer_only", min_pixels=784, max_pixels=784,
                          max_completion_length=4)
    vlt._verified_model_directory(directory, vlt.DEFAULT_MODEL)
    processor = vlt._processor(directory, profile)
    tokenizer = processor.tokenizer
    observed = tokenizer("assistant\nalpha<|im_end|>\n", add_special_tokens=False)["input_ids"]
    expected = [TOKEN["assistant"], TOKEN["alpha"], TOKEN["<|im_end|>"]]
    if observed != expected or tokenizer.eos_token_id != TOKEN["<|im_end|>"]:
        raise DryRunError(f"reloaded tokenizer: {observed} for {expected}, "
                          f"eos {tokenizer.eos_token_id}")
    policy = transformers.AutoModelForImageTextToText.from_pretrained(
        str(directory), local_files_only=True, use_safetensors=True, dtype=torch.float32,
        attn_implementation="sdpa").eval()
    spec = vlt.load_spec(spec_path)
    probabilities = {}
    for label, row in rows_by_label.items():
        probabilities[label] = vlt.gold_probability(
            policy, processor, row, spec, profile, "cpu",
            eos_token_ids=[TOKEN["<|im_end|>"]], suppress_tokens=[TOKEN["<|image_pad|>"]])
    common = [probabilities[label] for label in LABELS[:-1]]
    if not all(0.25 <= p <= 0.40 for p in common) or not probabilities[LABELS[-1]] < 0.01:
        raise DryRunError(f"gold probabilities off design: {probabilities}")
    return probabilities


def train_w(root, spec_path, model_dir):
    from vision_rl import vlm_local_training as vlt

    output = root / "w"
    code = vlt.main(["--mode", "sft", "--train", str(root / "data" / "train400.jsonl"),
                     "--train-only", "--data-spec", str(spec_path), "--model-directory",
                     str(model_dir), "--device", "cpu", "--max-steps", "2", "--learning-rate",
                     "1e-4", "--seed", str(SEED), "--save-steps", "1", "--output", str(output),
                     *PROFILE_ARGS])
    if code != 0:
        raise DryRunError(f"W training exited {code}")
    return output / "final_adapter"


def build_fixture(root):
    started = time.monotonic()
    root.mkdir(parents=True)
    protocol = json.loads((ROOT / "docs" / "aircraft-rl-protocol-freshpool-v1.json").read_text())
    images, rows = build_images(root)
    manifest, result = build_manifests(root, images, rows, protocol)
    spec_path = root / "data_spec.json"
    write_json(spec_path, {
        "schema_version": 1, "benchmark": "freshpool-cpu-dryrun-synthetic-v1",
        "diagnostic_only": True, "labels": LABELS,
        "question": "Which synthetic class is shown?",
        "split_policy": "synthetic native-TRAIN stand-in; no validation or test is read",
        "preprocessing": "64x64 RGB colour fields upsampled from 16x16 random; no external images",
        "sources": [{"url": "generated://scripts/freshpool_dryrun/run.py",
                     "sha256": sha(Path(__file__))}]})
    model_dir = root / "model"
    build_model(model_dir)
    first = {}
    for row in rows:
        first.setdefault(row["label"], {**row, "image": str(images / f"{row['id']}.png")})
    probabilities = check_model(model_dir, spec_path, first)
    w_adapter = train_w(root, spec_path, model_dir)

    receipts = root / "receipts"
    receipts.mkdir()
    git(receipts, "init", "-q")
    revision = receipts / "revision.json"
    revision.write_bytes((ROOT / "docs" / "aircraft-rl-protocol-freshpool-v1.json").read_bytes())
    activated = {**protocol, "protocol_id": protocol["protocol_id"] + "-dryrun-activated",
                 "activated": True,
                 "status": "DRY-RUN FIXTURE: an activated copy for the CPU dry run only",
                 "activation": {"activates": {"path": str(revision), "sha256": sha(revision)}}}
    write_json(receipts / "protocol_revision.json", activated)
    write_json(receipts / "branch.json", {
        "a_star": 0.75, "b4_edge_extension": False, "w_valid_eos": 0.99,
        "w_evaluator_refused": False, "w_val_accuracy": 0.5, "d_req": 0.05, "mode": "full",
        "fixture": "freshpool CPU dry run only"})
    git(receipts, "add", "revision.json", "protocol_revision.json", "branch.json")
    git(receipts, "commit", "-q", "-m", "dry-run receipts")
    decision = ROOT / "docs" / "decision-log-2026-10-16-claude.md"
    evidence = root / "evidence-base.json"
    write_json(evidence, {
        "labels": LABELS, "a_star": 0.75, "b4_edge_extension": False,
        "rows": {split: [{"id": r["id"], "label": r["label"]} for r in result[split]]
                 for split in ("h_select", "h_gate")}})
    return {
        "root": root, "spec": spec_path, "model": model_dir, "w_adapter": w_adapter,
        "w_sha": sha(w_adapter / "adapter_model.safetensors"),
        "manifest_dir": root / "manifests",
        "scan_ids": [r["id"] for r in read_jsonl(root / "manifests" / "s_scan_seed42.jsonl")],
        "receipts": {name: {"path": str(path), "sha256": sha(path)} for name, path in (
            ("protocol_revision", receipts / "protocol_revision.json"),
            ("decision_record", decision), ("branch", receipts / "branch.json"))},
        "evidence": {"path": str(evidence), "sha256": sha(evidence)},
        "summary": {"counts": manifest["counts"],
                    "pool_drops": result["audit"]["pool_drops"],
                    "planted_duplicate_dropped": PLANTED[0] in result["audit"]["pool_drops"],
                    "gold_probability_at_base": probabilities,
                    "seconds": time.monotonic() - started},
    }


# ---------------------------------------------------------------- plan


def templates(fixture):
    # The E0 entry owns --save-steps (it checkpoints every update), so it takes the rest.
    unsaved = ["--data-spec", str(fixture["spec"]), "--model-directory", str(fixture["model"]),
               "--device", "cpu", "--train-only", *PROFILE_ARGS]
    trainer = [*unsaved, "--save-steps", "1"]
    inference = ["--data-spec", str(fixture["spec"]), "--model-directory",
                 str(fixture["model"]), "--device", "cpu", *PROFILE_ARGS]
    identity = ["--init-adapter", "{w_adapter}", "--learning-rate", "{lr}", "--seed", "{seed}",
                "--stage-id", "{stage_id}"]
    tokenizations = ["--screen-tokenizations", "{screen_tokenizations}"]
    # {arm_cap}, not {active_cap}: rl600 resumes rl150's checkpoint with identical settings.
    rl = ["--mode", "grpo", "--train", "{manifest_s_scan_seed42}", *trainer, *identity,
          "--max-steps", str(scale.FINAL_UPDATES), "--target-accepted",
          str(scale.FINAL_UPDATES), "--screen-journal", "{arm_dir}/trainer/screen.jsonl",
          "--max-active-seconds", "{arm_cap}", *tokenizations]

    def wrapper(arm, *extra):
        return ["--run-id", "{run_id}", "--arm", arm, "--lr", "{lr}", "--seed", "{seed}",
                "--fragment", "{fragment}", "--output", "{arm_dir}/trainer", *extra, "--"]

    sft = ["--mode", "sft", "--order", "sequential", "--cycle", *trainer, *identity,
           "--max-steps", str(scale.SFT_MAX_STEPS), "--max-active-seconds", "{active_cap}"]
    scan = ["--scan-manifest", "{manifest_s_scan_seed42}"]
    argv = {
        "e0": ["--fragment", "{fragment}", "--stage-dir", "{stage_dir}", "--trainer-sha256",
               sha(TRAINER_SOURCE), "--", "--mode", "grpo", "--train",
               "{manifest_s_scan_seed42}", *unsaved, "--init-adapter", "{w_adapter}",
               "--learning-rate", "0.0001", "--seed", "{seed}", "--beta", "0.04"],
        "assay": ["--run-id", "{run_id}", "--split", "{split}", "--manifest",
                  "{split_manifest}", "--adapter", "{adapter}", "--fragment", "{fragment}",
                  "--stage-dir", "{stage_dir}", *inference],
        "p0": ["--scan-manifest", "{manifest_s_scan_seed42}", "--adapter", "{w_adapter}",
               "--fragment", "{fragment}", "--stage-dir", "{stage_dir}", *inference],
        "rl150": [*wrapper("RL", *scan), *rl, "--stop-after-steps",
                  str(scale.RL_SCREEN_UPDATES)],
        "rl600": [*wrapper("RL", *scan, "--continue-output-from", "{resume_checkpoint}"), *rl],
        "sft_e": [*wrapper("SFT-E"), "--train", "{accepted_manifest}", *sft,
                  "--stop-after-steps", str(scale.FINAL_UPDATES)],
        "sft_t": [*wrapper("SFT-T"), "--train", "{scanned_manifest}", *sft,
                  "--continue-from-checkpoint", "{parent_checkpoint}",
                  "--stop-after-active-seconds", "{t_rl}"],
        "egsft": [*wrapper("EG-SFT"), "--mode", "egsft", "--order", "sequential", "--train",
                  "{accepted_manifest}", *trainer, *identity, "--max-steps",
                  str(scale.FINAL_UPDATES), "--max-active-seconds", "{active_cap}",
                  *tokenizations],
    }
    entries = {"e0": (LAUNCH_DIR / "e0_gate.py", "e0_gate:main", "train"),
               "assay": (LAUNCH_DIR / "h_assay.py", "h_assay:main", "assay"),
               "p0": (LAUNCH_DIR / "p0_screen.py", "p0_screen:main", "assay")}
    fragment = (LAUNCH_DIR / "trainer_fragment.py", "trainer_fragment:main", "train")
    result = {}
    for name, parts in argv.items():
        path, entry, kind = entries.get(name, fragment)
        result[name] = {"kind": kind, "entry": entry, "entry_sha256": sha(path), "argv": parts}
    return result


def write_plan(fixture, directory, sentinel):
    directory.mkdir(parents=True)
    pinned = [EVALUATOR, REAL_EVALUATOR, HERE / "dryrun_scale.py", LAUNCH_DIR / "e0_gate.py",
              LAUNCH_DIR / "phase_worker.py", LAUNCH_DIR / "trainer_fragment.py",
              LAUNCH_DIR / "h_assay.py", LAUNCH_DIR / "p0_screen.py", TRAINER_SOURCE]
    manifest = fixture["manifest_dir"] / "manifest.json"
    plan = {
        "schema": "freshpool-launch-plan-v1",
        "python": sys.executable,
        "state_directory": str(directory / "state"),
        "evaluator": str(EVALUATOR),
        "environment": ENVIRONMENT,
        "dry_run": True,
        "file_sha256": {str(path): sha(path) for path in pinned},
        "limits": {"minimum_free_bytes": 5 * GiB, "rss_limit_bytes": 12 * GiB,
                   "survivor_grace_seconds": 10, "poll_seconds": 0.25},
        # Caps are scale.CAP_FACTOR x these wall seconds.
        "forecast_seconds": {"e0": 60, "p0": 30, "assay_h_select": 20, "assay_h_gate": 20,
                             "rl": 30, "sft_e": 20, "egsft": 20},
        "templates": templates(fixture),
        "b4_edge_extension": False,
        "w_endpoint": {"adapter_sha256": fixture["w_sha"],
                       "adapter_path": str(fixture["w_adapter"])},
        "owner_patterns": [sentinel],
        "g0": {"manifest_dir": str(fixture["manifest_dir"]), "manifest_sha256": sha(manifest),
               "receipts": fixture["receipts"]},
        "evidence_base": fixture["evidence"],
    }
    path = directory / "plan.json"
    write_json(path, plan)
    return path


# ---------------------------------------------------------------- launches


def load_launch():
    launch = load_module("freshpool_launch_dryrun", LAUNCH_DIR / "launch.py")
    launch.RL_SCREEN_UPDATES = scale.RL_SCREEN_UPDATES
    launch.FINAL_UPDATES = scale.FINAL_UPDATES
    launch.CAP_FACTOR = scale.CAP_FACTOR
    launch.POOL_BEFORE_AUDIT = POOL_BEFORE_AUDIT
    return launch


def forcing_gate(launch, real, forced):
    """Record a gate the evaluator fails, then pass it with the evaluator's own result for
    that gate, so the drive launch reaches every later stage."""

    def gate(self, name):
        try:
            return real(self, name)
        except launch.Stop as stop:
            if stop.reason not in ("gate_failed", "gate_not_evaluated"):
                raise
            report = json.loads(Path(stop.detail["report"]).read_text())
            if name == "G1":
                result = self.evaluator["gate_g1"](
                    self.evidence()["g1"], self.plan["w_endpoint"]["adapter_sha256"])
            else:
                study = self.study()
                result = {"G2": study.gate_g2, "G3a": study.gate_g3a,
                          "G3b": study.gate_g3b}[name]()
            entry = {"gate": name, "launcher_verdict": stop.reason,
                     "evaluator_failed_at": report.get("failed_at"),
                     "evaluator_status": (report.get("gates") or {}).get(name, {}).get("status"),
                     "direct_pass": result.get("pass"), "direct_checks": result.get("checks"),
                     "report": stop.detail["report"]}
            forced.append(entry)
            self.ledger.append("dryrun_gate_forced", **entry)
            self.state["gates"][name] = {"report": stop.detail["report"], "result": result,
                                         "forced_by_dry_run": True}
            self.ledger.save()
            return result

    return gate


def run_launch(launch, plan_path, force):
    forced, real = [], launch.Launcher.gate
    if force:
        launch.Launcher.gate = forcing_gate(launch, real, forced)
    started = time.monotonic()
    try:
        code = launch.main(["--plan", str(plan_path), "--expected-plan-sha256", sha(plan_path)])
    finally:
        launch.Launcher.gate = real
    state_dir = Path(json.loads(plan_path.read_text())["state_directory"])
    state = json.loads((state_dir / "state.json").read_text()) if (
        state_dir / "state.json").is_file() else None
    journal = read_jsonl(state_dir / "journal.jsonl") if (
        state_dir / "journal.jsonl").is_file() else []
    return {"code": code, "forced": forced, "state": state, "journal": journal,
            "state_dir": state_dir, "seconds": time.monotonic() - started}


# ---------------------------------------------------------------- verdicts

CATEGORIES = {
    "e0": r"e0", "w_assay": r"W-h_(select|gate)", "p0": r"p0",
    "rl150": rf"rl-.+-s{SEED}-u{scale.RL_SCREEN_UPDATES}",
    "rl150_h_select": rf"rl-.+-s{SEED}-u{scale.RL_SCREEN_UPDATES}-h_select",
    "rl600": rf"rl-.+-s{SEED}-u{scale.FINAL_UPDATES}",
    "rl600_h_select": rf"rl-.+-s{SEED}-u{scale.FINAL_UPDATES}-h_select",
    "rl_star_h_gate": rf"rl-.+-s{SEED}-u{scale.FINAL_UPDATES}-h_gate",
    "sft_e": rf"sft-e-.+-s{SEED}", "sft_e_h_select": rf"sft-e-.+-s{SEED}-h_select",
    "sft_e_star_h_gate": rf"sft-e-.+-s{SEED}-h_gate",
    "sft_t": rf"sft-t-.+-s{SEED}", "sft_t_h_select": rf"sft-t-.+-s{SEED}-h_select",
    "sft_t_star_h_gate": rf"sft-t-.+-s{SEED}-h_gate",
    "egsft": rf"egsft-.+-s{SEED}", "egsft_h_gate": rf"egsft-.+-s{SEED}-h_gate",
}
# Stage counts the launcher must have completed when it stops at (or after) each point,
# written out from the study sequence; sft_t is k in 1..3 (G3a's qualifying LRs).
BEFORE = {
    "G1": {"e0": 1, "w_assay": 2, "p0": 1},
    "G2": {"rl150": 3, "rl150_h_select": 3},
    "G3a": {"rl600": 2, "rl600_h_select": 2, "sft_e": 3, "sft_e_h_select": 3,
            "rl_star_h_gate": 1, "sft_e_star_h_gate": 1},
    "G3b": {"sft_t": "k", "sft_t_h_select": "k", "sft_t_star_h_gate": 1},
    "final": {"egsft": 1, "egsft_h_gate": 1},
}
ORDER = ("G1", "G2", "G3a", "G3b", "final")


def categorise(stage_ids):
    counts, unknown = dict.fromkeys(CATEGORIES, 0), []
    for stage in stage_ids:
        names = [name for name, pattern in CATEGORIES.items() if re.fullmatch(pattern, stage)]
        if len(names) == 1:
            counts[names[0]] += 1
        else:
            unknown.append(stage)
    return counts, unknown


def structure_problems(stage_ids, through):
    """Problems if the completed stages are not exactly those before ``through``."""
    counts, unknown = categorise(stage_ids)
    expected = dict.fromkeys(CATEGORIES, 0)
    for point in ORDER[:ORDER.index(through) + 1]:
        expected.update(BEFORE[point])
    k = counts["sft_t"]
    problems = [f"unclassified stage {s}" for s in unknown]
    for name, value in expected.items():
        want = k if value == "k" else value
        if counts[name] != want or (value == "k" and not 1 <= k <= len(GRID)):
            problems.append(f"{name}: {counts[name]} completed, expected {value}")
    return problems, counts


def fragment_problems(state, fixture):
    """Independent re-reads of what each completed stage left on disk."""
    problems = []
    scan = fixture["scan_ids"]
    for stage, entry in sorted(state["completed"].items()):
        f = entry["fragment"]
        if "split" in f:
            predictions = f.get("predictions") or {}
            per_image = {"h_select": 5, "h_gate": 9}[f["split"]]
            if len(predictions) != SPLIT_IMAGES or any(
                    len(v) != per_image for v in predictions.values()):
                problems.append(f"{stage}: assay shape")
            continue
        if "arm" not in f:
            continue
        adapter = Path(f["adapter_path"]) / "adapter_model.safetensors"
        if not adapter.is_file() or sha(adapter) != f["adapter_sha256"]:
            problems.append(f"{stage}: adapter bytes differ from the fragment")
        if f["reference_sha256"] != fixture["w_sha"]:
            problems.append(f"{stage}: reference/initial adapter is not W")
        steps = sorted(int(k) for k in f["step_committed_active_seconds"])
        if steps != list(range(steps[0], steps[-1] + 1)) or steps[-1] != f["updates"]:
            problems.append(f"{stage}: step series {steps[:3]}..{steps[-3:]}")
        expected = {"SFT-E": scale.FINAL_UPDATES, "EG-SFT": scale.FINAL_UPDATES}.get(f["arm"])
        if f["arm"] == "RL":
            expected = (scale.RL_SCREEN_UPDATES if stage.endswith(
                f"-u{scale.RL_SCREEN_UPDATES}") else scale.FINAL_UPDATES)
            accepted = [r["id"] for r in read_jsonl(f["accepted_manifest"])]
            scanned = [r["id"] for r in read_jsonl(f["scanned_manifest"])]
            screen = json.loads((Path(f["checkpoint"]) / "screen.json").read_text())["entries"]
            if (len(accepted) != f["updates"] or scanned != scan[:len(scanned)]
                    or [e["id"] for e in screen] != scanned
                    or [e["id"] for e in screen if e["accepted"]] != accepted
                    or any((e["u"] >= 0.5) != e["accepted"] for e in screen)):
                problems.append(f"{stage}: screen, accepted and scanned manifests disagree")
        if f["arm"] == "SFT-T" and (steps[0] != scale.FINAL_UPDATES + 1
                                    or f["updates"] <= scale.FINAL_UPDATES):
            problems.append(f"{stage}: SFT-T does not continue SFT-E's step count")
        if expected is not None and f["updates"] != expected:
            problems.append(f"{stage}: {f['updates']} updates, expected {expected}")
    return problems


def last(journal, event):
    found = [e for e in journal if e["event"] == event]
    return found[-1] if found else None


def judge_drive(run, fixture):
    state, journal, forced = run["state"], run["journal"], run["forced"]
    if state is None:
        return ["no state: the launcher refused before starting"], {}
    problems, counts = structure_problems(state["completed"], "final")
    problems += fragment_problems(state, fixture)
    final = last(journal, "final_check")
    if forced:
        first = forced[0]["gate"]
        report = json.loads(Path(final["report"]).read_text()) if final else {}
        if (state["status"], state.get("stop_reason")) != (
                "stopped", "final_development_check_failed"):
            problems.append(f"stopped as {state['status']}/{state.get('stop_reason')}, "
                            "expected final_development_check_failed")
        if report.get("failed_at") != first:
            problems.append(f"final report failed_at {report.get('failed_at')}, first forced "
                            f"gate {first}")
        for n, entry in enumerate(forced):
            consistent = (
                (entry["launcher_verdict"], entry["evaluator_failed_at"], entry["direct_pass"])
                == ("gate_failed", entry["gate"], False) if n == 0 else
                (entry["launcher_verdict"], entry["evaluator_failed_at"],
                 entry["evaluator_status"]) == ("gate_not_evaluated", first, "not_evaluated"))
            if not consistent:
                problems.append(f"forced gate {entry['gate']} inconsistent: {entry}")
    elif state["status"] != "completed" or not (final and final.get("passed")):
        problems.append(f"no gate forced, yet status {state['status']}/"
                        f"{state.get('stop_reason')}")
    return problems, counts


def judge_strict(run, fixture):
    state, journal = run["state"], run["journal"]
    if state is None:
        return ["no state: the launcher refused before starting"], {}, None
    if state["status"] == "completed":
        problems, counts = structure_problems(state["completed"], "final")
        return problems + fragment_problems(state, fixture), counts, "completed"
    if state.get("stop_reason") != "gate_failed":
        return [f"stopped for {state.get('stop_reason')}, not a gate verdict"], {}, None
    finished = last(journal, "gate_finished") or {}
    gate = finished.get("gate")
    if gate not in ORDER:
        return [f"stopped gate_failed, but the journal's last gate_finished names gate "
                f"{gate!r}, not one of {ORDER}"], {}, None
    try:
        report = json.loads(Path(finished["report"]).read_text())
    except (KeyError, TypeError, OSError, ValueError) as exc:
        return [f"stopped gate_failed at {gate}, but its evaluator report is unreadable: "
                f"{type(exc).__name__}: {exc}"], {}, gate
    result = (report.get("gates") or {}).get(gate) or {}
    problems = []
    if (report.get("failed_at"), result.get("status"), result.get("pass")) != (
            gate, "evaluated", False):
        problems.append(f"gate_failed at {gate}, but the report says failed_at "
                        f"{report.get('failed_at')}, status {result.get('status')}, "
                        f"pass {result.get('pass')}")
    passed = [g for g in ORDER[:ORDER.index(gate)]]
    if sorted(state["gates"]) != sorted(passed):
        problems.append(f"gates recorded {sorted(state['gates'])}, expected {passed}")
    structure, counts = structure_problems(state["completed"], gate)
    return problems + structure + fragment_problems(state, fixture), counts, gate


def diagnostics(run):
    """What a reader needs when a launch stopped for the wrong reason."""
    lines = []
    stop = last(run["journal"], "study_stopped")
    if stop:
        lines.append("study_stopped: " + json.dumps(stop, default=str)[:4000])
    started = last(run["journal"], "stage_started")
    if started:
        log = (run["state_dir"] / "stages" / started["stage"]
               / f"instance-{started['instance']}" / "stage.log")
        if log.is_file():
            tail = log.read_text(errors="replace").splitlines()[-150:]
            lines.append(f"--- tail of {log} ---")
            lines.extend(tail)
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True, help="new directory for everything")
    args = parser.parse_args(argv)
    work = args.work.resolve()
    if work.exists():
        raise SystemExit(f"{work} exists; the dry run needs a new directory")
    if "torch" in sys.modules:
        raise SystemExit("torch was imported before the frozen environment was set")
    os.environ.update(ENVIRONMENT)  # before any torch, transformers or accelerate import
    fixture = build_fixture(work / "fixture")
    print(json.dumps({"fixture": fixture["summary"]}, indent=2, default=str), flush=True)
    launch = load_launch()
    sentinel = f"freshpool-dryrun-owner-sentinel-{uuid.uuid4().hex}"
    runs, verdicts = {}, {}
    for name, force in (("drive", True), ("strict", False)):
        plan = write_plan(fixture, work / name, sentinel)
        runs[name] = run_launch(launch, plan, force)
        print(json.dumps({name: {"code": runs[name]["code"],
                                 "status": (runs[name]["state"] or {}).get("status"),
                                 "stop_reason": (runs[name]["state"] or {}).get("stop_reason"),
                                 "seconds": runs[name]["seconds"]}}), flush=True)
    problems, counts = judge_drive(runs["drive"], fixture)
    verdicts["drive"] = {"passed": not problems, "problems": problems, "stage_counts": counts,
                         "forced_gates": runs["drive"]["forced"]}
    problems, counts, stopped_at = judge_strict(runs["strict"], fixture)
    verdicts["strict"] = {"passed": not problems, "problems": problems, "stage_counts": counts,
                          "stopped_at": stopped_at}
    summary = {
        "passed": all(v["passed"] for v in verdicts.values()),
        "verdicts": verdicts,
        "runs": {name: {"code": run["code"], "seconds": run["seconds"],
                        "status": (run["state"] or {}).get("status"),
                        "stop_reason": (run["state"] or {}).get("stop_reason"),
                        "selections": (run["state"] or {}).get("selections"),
                        "state_directory": str(run["state_dir"])}
                 for name, run in runs.items()},
        "fixture": fixture["summary"],
        "scaled": {"rl_screen_updates": scale.RL_SCREEN_UPDATES,
                   "final_updates": scale.FINAL_UPDATES, "cap_factor": scale.CAP_FACTOR,
                   "pool_before_audit": POOL_BEFORE_AUDIT, "e0_steps": 4},
    }
    write_json(work / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    for name, verdict in verdicts.items():
        if not verdict["passed"]:
            print(f"=== {name}: FAILED ===", flush=True)
            print("\n".join(diagnostics(runs[name])), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
