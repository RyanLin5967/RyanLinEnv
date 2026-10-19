"""Real-image count policies: frozen ResNet18, SFT warmup, then sampled REINFORCE.

Install the optional torch/torchvision stack before using this module. The encoder
sees resized image pixels only; labels are consumed by supervised loss/reward and
validation scoring. No test manifest is accepted by the training function.
"""

import argparse
import copy
import hashlib
import json
import platform
import random
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from vision_rl.data import assert_disjoint, fingerprint, image_hash, load_manifest, write_json

ENCODER_ID = "torchvision/resnet18/IMAGENET1K_V1"
REWARD = "1[action == target] - abs(action - target) / (num_actions - 1)"
CLASSIFICATION_REWARD = "1[action == target]"
TASKS = ("counting", "classification")


def state_fingerprint(state: dict) -> str:
    """Serialization-independent identity of a frozen encoder's exact tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _stack():
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("Install requirements-vision.txt to use the real-image policy") from exc
    return torch, torchvision


def select_device(requested: str = "auto") -> str:
    torch, _ = _stack()
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this host")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this host")
    return requested


def _count_schema(rows: list[dict]) -> list[str]:
    choices = rows[0]["choices"]
    if choices != [str(i) for i in range(len(choices))]:
        raise ValueError("count policy requires choices in order 0, 1, ..., N")
    if any(r["choices"] != choices or r["category"] != "counting" for r in rows):
        raise ValueError("count policy requires a consistent counting task")
    if any(r["prompt"] != rows[0]["prompt"] for r in rows):
        raise ValueError("count policy requires one consistent counting question")
    return choices


def _task_schema(rows: list[dict]) -> tuple[list[str], str]:
    """(choices, task) for one consistent counting or single-label classification task."""
    if rows[0]["category"] == "counting":
        return _count_schema(rows), "counting"
    if rows[0]["category"] != "classification":
        raise ValueError("policy supports counting or classification tasks")
    choices = rows[0]["choices"]
    if any(r["choices"] != choices or r["category"] != "classification" for r in rows):
        raise ValueError("classification policy requires one consistent label set")
    if any(r["prompt"] != rows[0]["prompt"] for r in rows):
        raise ValueError("classification policy requires one consistent question")
    return choices, "classification"


def exact_reward(actions, targets, num_actions: int):
    """Classification reward: labels have no order, so only an exact match earns credit."""
    if num_actions < 2:
        raise ValueError("at least two label actions are required")
    return (actions == targets).float()


def shaped_reward(actions, targets, num_actions: int):
    """Bounded count reward; labels never enter the policy's observations."""
    if num_actions < 2:
        raise ValueError("at least two count actions are required")
    return (actions == targets).float() - (actions - targets).abs().float() / (num_actions - 1)


def reinforce_loss(
    logits, targets, *, samples=8, entropy_weight=0.005, reference=None, kl=0.02, reward=None
):
    """Sampled policy gradient with an unbiased leave-one-out group baseline."""
    torch, _ = _stack()
    if samples < 2:
        raise ValueError("at least two independent samples are required")
    log_probs = logits.log_softmax(-1)
    probs = log_probs.exp()
    actions = torch.multinomial(probs, samples, replacement=True)
    rewards = (reward or shaped_reward)(actions, targets[:, None], logits.shape[-1])
    baseline = (rewards.sum(1, keepdim=True) - rewards) / (samples - 1)
    advantage = (rewards - baseline).detach()
    policy_loss = -(advantage * log_probs.gather(1, actions)).mean()
    entropy = -(probs * log_probs).sum(-1).mean()
    divergence = logits.new_tensor(0.0)
    if reference is not None:
        divergence = (probs * (log_probs - reference.log_softmax(-1))).sum(-1).mean()
    loss = policy_loss - entropy_weight * entropy + kl * divergence
    return loss, {
        "sampled_reward": float(rewards.mean().detach()),
        "entropy": float(entropy.detach()),
        "kl_to_sft": float(divergence.detach()),
    }


def _encoder(weights_path: Path | None = None):
    torch, torchvision = _stack()
    weights = None if weights_path else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False)


def _features(rows, encoder, device, *, augment=False, batch_size=24):
    """Global mean and spatial standard deviation of frozen convolution features.

    Whole-frame resize preserves all objects. Training adds horizontal/vertical
    flips, while validation and inference each use exactly one unmodified view.
    """
    torch, torchvision = _stack()
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224), antialias=True),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensors = []
    views = (
        [
            None,
            Image.Transpose.FLIP_LEFT_RIGHT,
            Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.ROTATE_180,
        ]
        if augment
        else [None]
    )
    for transpose in views:
        for row in rows:
            with Image.open(row["image"]) as source:
                im = source.convert("RGB")
                if transpose is not None:
                    im = im.transpose(transpose)
                tensors.append(transform(im))
    trunk = torch.nn.Sequential(*list(encoder.children())[:-2]).to(device).eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            grid = trunk(torch.stack(tensors[start : start + batch_size]).to(device))
            result.append(torch.cat([grid.mean((2, 3)), grid.std((2, 3))], dim=1).cpu())
    trunk.cpu()
    return torch.cat(result).float()


def _head(inputs: int, actions: int, hidden: int):
    torch, _ = _stack()
    if hidden:
        return torch.nn.Sequential(
            torch.nn.Linear(inputs, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, actions),
        )
    return torch.nn.Linear(inputs, actions)


def _metrics(logits, targets, task="counting"):
    predictions = logits.argmax(-1)
    if task == "classification":
        return {"accuracy": float((predictions == targets).float().mean()), "mae": None}
    nonzero = targets != 0
    return {
        "accuracy": float((predictions == targets).float().mean()),
        "mae": float((predictions - targets).abs().float().mean()),
        "nonzero_accuracy": float((predictions[nonzero] == targets[nonzero]).float().mean())
        if nonzero.any()
        else None,
    }


def _rank(metrics):
    return metrics["accuracy"], -(metrics["mae"] or 0.0)


def _save_policy(path, head, metadata, mean, scale):
    torch, _ = _stack()
    torch.save(
        {
            "schema_version": 1,
            "metadata": metadata,
            "state_dict": copy.deepcopy(head.cpu().state_dict()),
            "feature_mean": mean.cpu(),
            "feature_scale": scale.cpu(),
        },
        path,
    )


def train_count_policy(
    train_manifest: Path,
    validation_manifest: Path,
    output: Path,
    *,
    device="auto",
    seeds=(7, 19),
    hidden_sizes=(0, 128),
    sft_epochs=160,
    rl_steps=800,
    encoder_weights: Path | None = None,
    augment=True,
) -> dict:
    """Run a bounded experiment search; choose checkpoint using validation only."""
    torch, torchvision = _stack()
    if not seeds or not hidden_sizes or sft_epochs < 1 or rl_steps < 1:
        raise ValueError("provide seeds, architectures, and positive SFT/RL budgets")
    if any(type(h) is not int or not 0 <= h <= 512 for h in hidden_sizes):
        raise ValueError("hidden sizes must be integers between 0 and 512")
    train_rows = load_manifest(train_manifest, expected_split="train")
    val_rows = load_manifest(validation_manifest, expected_split="validation")
    assert_disjoint(train_rows, val_rows)
    choices, task = _task_schema(train_rows + val_rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    selected_device = select_device(device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(0)
    encoder = _encoder(encoder_weights)
    torch.save(encoder.state_dict(), output / "encoder.pt")
    encoder_sha = hashlib.sha256((output / "encoder.pt").read_bytes()).hexdigest()
    x = _features(train_rows, encoder, selected_device, augment=augment)
    v = _features(val_rows, encoder, selected_device)
    mean, scale = x.mean(0), x.std(0).clamp_min(0.05)
    x, v = (x - mean) / scale, (v - mean) / scale
    y = torch.tensor([ord(r["answer"]) - 65 for r in train_rows] * (4 if augment else 1))
    vy = torch.tensor([ord(r["answer"]) - 65 for r in val_rows])
    metadata = {
        "encoder": ENCODER_ID,
        "encoder_sha256": encoder_sha,
        "encoder_state_sha256": state_fingerprint(encoder.state_dict()),
        "encoder_file": "encoder.pt",
        "choices": choices,
        "prompt": train_rows[0]["prompt"],
        "feature_dim": x.shape[1],
        "train_fingerprint": fingerprint(train_rows),
        "validation_fingerprint": fingerprint(val_rows),
        "selection_split": "validation",
        "task": task,
        "reward": REWARD if task == "counting" else CLASSIFICATION_REWARD,
        "image_transform": "RGB resize224x224 + ImageNet normalize; mean/std layer4",
        "train_views": 4 if augment else 1,
        "optimizer_device": "cpu",
        "encoder_device": selected_device,
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
    }
    plan = {
        "seeds": list(seeds),
        "hidden_sizes": list(hidden_sizes),
        "sft_epochs": sft_epochs,
        "rl_steps": rl_steps,
        "sft_lr": 0.003,
        "rl_lr": 0.0005,
        "weight_decay": 0.05,
        "batch_size": 128,
        "samples_per_observation": 8,
        "entropy_weight": 0.005,
        "kl_weight": 0.02,
        "selection": "maximum validation exact accuracy; then minimum validation MAE"
        if task == "counting"
        else "maximum validation exact accuracy",
        "test_access": False,
    }
    write_json(output / "plan.json", {**plan, **metadata})
    candidates = []
    history_path = output / "history.jsonl"
    selected = None
    for hidden in hidden_sizes:
        for seed in seeds:
            torch.manual_seed(seed)
            candidate = f"h{hidden}-s{seed}"
            head = _head(x.shape[1], len(choices), hidden)
            candidate_meta = {**metadata, "hidden": hidden, "seed": seed, "candidate": candidate}
            _save_policy(
                output / f"{candidate}-initial.pt",
                head,
                {**candidate_meta, "phase": "initial"},
                mean,
                scale,
            )
            optimizer = torch.optim.AdamW(head.parameters(), lr=plan["sft_lr"], weight_decay=0.05)
            best_sft, sft_state = None, None
            for epoch in range(1, sft_epochs + 1):
                head.train()
                for indices in torch.randperm(len(x)).split(128):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(
                        head(x[indices]), y[indices], label_smoothing=0.05
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                if epoch % 5 == 0 or epoch == sft_epochs:
                    head.eval()
                    with torch.no_grad():
                        metrics = _metrics(head(v), vy, task)
                        train_metrics = _metrics(head(x), y, task)
                    event = {
                        "candidate": candidate,
                        "phase": "sft",
                        "step": epoch,
                        "validation": metrics,
                        "train": train_metrics,
                        "loss": float(loss.detach()),
                    }
                    with history_path.open("a") as f:
                        f.write(json.dumps(event) + "\n")
                    if best_sft is None or _rank(metrics) > _rank(best_sft):
                        best_sft, sft_state = metrics, copy.deepcopy(head.state_dict())
                        sft_epoch = epoch
            head.load_state_dict(sft_state)
            _save_policy(
                output / f"{candidate}-sft.pt",
                head,
                {**candidate_meta, "phase": "sft", "selected_step": sft_epoch},
                mean,
                scale,
            )
            reference = copy.deepcopy(head).eval().requires_grad_(False)
            optimizer = torch.optim.AdamW(head.parameters(), lr=plan["rl_lr"], weight_decay=0.05)
            best_rl, rl_state = None, None
            for step in range(1, rl_steps + 1):
                indices = torch.randint(len(x), (min(128, len(x)),))
                # Keep dropout disabled: sampled categorical actions provide exploration.
                head.eval()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    reference_logits = reference(x[indices])
                loss, reward_metrics = reinforce_loss(
                    head(x[indices]),
                    y[indices],
                    reference=reference_logits,
                    reward=shaped_reward if task == "counting" else exact_reward,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                if step % 20 == 0 or step == rl_steps:
                    with torch.no_grad():
                        metrics = _metrics(head(v), vy, task)
                        train_metrics = _metrics(head(x), y, task)
                    event = {
                        "candidate": candidate,
                        "phase": "rl",
                        "step": step,
                        "validation": metrics,
                        "train": train_metrics,
                        **reward_metrics,
                    }
                    with history_path.open("a") as f:
                        f.write(json.dumps(event) + "\n")
                    if best_rl is None or _rank(metrics) > _rank(best_rl):
                        best_rl, rl_state = metrics, copy.deepcopy(head.state_dict())
                        rl_step = step
            head.load_state_dict(rl_state)
            _save_policy(
                output / f"{candidate}-rl.pt",
                head,
                {**candidate_meta, "phase": "rl", "selected_step": rl_step},
                mean,
                scale,
            )
            entry = {
                "candidate": candidate,
                "sft_validation": best_sft,
                "rl_validation": best_rl,
                "sft_epoch": sft_epoch,
                "rl_step": rl_step,
            }
            candidates.append(entry)
            # Select RL policy within the RL arm; keep independently best SFT ablation.
            if selected is None or _rank(best_rl) > _rank(selected["rl_validation"]):
                selected = entry
            print(json.dumps(entry), flush=True)
    best_sft_candidate = max(candidates, key=lambda r: _rank(r["sft_validation"]))
    import shutil

    for name, source in {
        "policy.pt": f"{selected['candidate']}-rl.pt",
        "sft.pt": f"{best_sft_candidate['candidate']}-sft.pt",
        "initial.pt": f"{selected['candidate']}-initial.pt",
    }.items():
        shutil.copyfile(output / source, output / name)
    majority = Counter(y.tolist()).most_common(1)[0][0]
    report = {
        **metadata,
        "plan": plan,
        "experiments": candidates,
        "selected_candidate": selected["candidate"],
        "sft_candidate": best_sft_candidate["candidate"],
        "majority_count_from_train": majority,
        "train_n": len(train_rows),
        "validation_n": len(val_rows),
        "elapsed_seconds": time.monotonic() - start,
        "platform": platform.platform(),
        "checkpoint_sha256": hashlib.sha256((output / "policy.pt").read_bytes()).hexdigest(),
    }
    write_json(output / "training.json", report)
    return report


def _load_head_checkpoint(checkpoint: Path):
    torch, _ = _stack()
    if checkpoint.stat().st_size > 5 * 1024**2:
        raise ValueError("policy checkpoint exceeds 5 MiB")
    with zipfile.ZipFile(checkpoint) as archive:
        if sum(item.file_size for item in archive.infolist()) > 8 * 1024**2:
            raise ValueError("policy checkpoint expanded size exceeds 8 MiB")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != 1:
        raise ValueError("unsupported count-policy checkpoint schema")
    meta = saved["metadata"]
    if meta.get("encoder") != ENCODER_ID or meta.get("feature_dim") != 1024:
        raise ValueError("unsupported encoder or feature dimensions")
    hidden = meta.get("hidden")
    if type(hidden) is not int or not 0 <= hidden <= 512:
        raise ValueError("policy hidden size must be between 0 and 512")
    choices = meta.get("choices")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
        raise ValueError("policy must have 2 to 26 choices")
    if meta.get("task", "counting") not in TASKS:
        raise ValueError("unsupported policy task")
    for name in ("feature_mean", "feature_scale"):
        value = saved[name]
        if not isinstance(value, torch.Tensor) or value.shape != (1024,):
            raise ValueError("invalid feature normalization shape")
        if not torch.isfinite(value).all() or (name == "feature_scale" and (value <= 0).any()):
            raise ValueError("invalid feature normalization values")
    expected = _head(1024, len(choices), hidden).state_dict()
    state = saved["state_dict"]
    if set(state) != set(expected) or any(
        not isinstance(state[k], torch.Tensor)
        or state[k].shape != expected[k].shape
        or not torch.isfinite(state[k]).all()
        for k in expected
    ):
        raise ValueError("invalid policy state tensor shapes or values")
    return saved


def _checked_encoder(checkpoint: Path, meta: dict, trusted_encoder: Path | None = None):
    """The checkpoint's sibling encoder.pt after its checksum, or a trusted encoder after its
    tensor fingerprint."""
    if trusted_encoder is not None:
        encoder_path = Path(trusted_encoder)
    else:
        if meta["encoder_file"] != "encoder.pt":
            raise ValueError("unsafe encoder path in checkpoint")
        encoder_path = checkpoint.parent / "encoder.pt"
        if hashlib.sha256(encoder_path.read_bytes()).hexdigest() != meta["encoder_sha256"]:
            raise ValueError("encoder checksum mismatch")
    encoder = _encoder(encoder_path)
    if trusted_encoder is not None and (
        state_fingerprint(encoder.state_dict()) != meta.get("encoder_state_sha256")
    ):
        raise ValueError("trusted encoder tensor checksum mismatch")
    return encoder


def predict_count_policy(
    rows: list[dict], checkpoint: Path, *, device="auto", trusted_encoder: Path | None = None
) -> dict:
    """Run a saved pixel-only policy without exposing answers to the encoder/head."""
    torch, _ = _stack()
    checkpoint = Path(checkpoint)
    saved = _load_head_checkpoint(checkpoint)
    meta = saved["metadata"]
    choices, task = _task_schema(rows)
    if (
        choices != meta["choices"]
        or rows[0]["prompt"] != meta["prompt"]
        or task != meta.get("task", "counting")
    ):
        raise ValueError("evaluation question/choices differ from the trained task")
    encoder = _checked_encoder(checkpoint, meta, trusted_encoder)
    x = _features(rows, encoder, select_device(device))
    x = (x - saved["feature_mean"]) / saved["feature_scale"]
    head = _head(meta["feature_dim"], len(meta["choices"]), meta["hidden"])
    head.load_state_dict(saved["state_dict"])
    head.eval()
    with torch.inference_mode():
        labels = head(x).argmax(-1).tolist()
    return {
        r["id"]: {"response": f"<answer>{chr(65 + label)}</answer>"}
        for r, label in zip(rows, labels, strict=True)
    }


def evaluate_count_policy(
    manifest: Path,
    checkpoint: Path,
    output: Path,
    *,
    device="auto",
    trusted_encoder: Path | None = None,
):
    from vision_rl.evaluation import score

    rows = load_manifest(manifest)
    started = time.monotonic()
    predictions = predict_count_policy(
        rows, checkpoint, device=device, trusted_encoder=trusted_encoder
    )
    task = _load_head_checkpoint(Path(checkpoint))["metadata"].get("task", "counting")
    result = score(
        rows,
        predictions,
        {
            "name": f"resnet18-{'count' if task == 'counting' else 'classification'}-policy",
            "backend": "torch",
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        },
    )
    result["elapsed_seconds"] = time.monotonic() - started
    if task == "counting":
        result["mean_absolute_count_error"] = float(
            np.mean([abs(ord(item["answer"]) - ord(item["expected"])) for item in result["items"]])
        )
    write_json(output, result)
    return result


# ---------------------------------------------------------------------------
# Train on your own labeled photos: vision-rl train-folder / predict / make-example.
# Folder in (one subfolder per label), tested classifier out. Same SFT-then-RL recipe.

FOLDER_PROMPT = "Which category best describes this image?"
MIN_PER_LABEL = 3  # one image each for train, validation and test
SPLIT_NAMES = {"train", "test", "valid", "validation", "val"}


def _visible(path: Path, root: Path) -> bool:
    """Skip hidden files and folders and macOS zip debris (__MACOSX, ._name.jpg)."""
    parts = path.relative_to(root).parts
    return not any(p.startswith(".") or p == "__MACOSX" for p in parts)


def _label_dirs(root: Path) -> list[Path]:
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and _visible(p, root))
    if any(p.is_symlink() for p in dirs):
        raise ValueError("label folders cannot be symlinks")
    if not 2 <= len(dirs) <= 26:
        raise ValueError(f"need 2 to 26 label folders, found {len(dirs)} in {root}")
    return dirs


def build_dataset(
    images: str | Path,
    output: str | Path,
    *,
    seed: int = 0,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict:
    """Write train/validation/test manifests from a labeled image folder.

    Each label is split separately, so every split sees every label. Pixel-identical copies
    are dropped before splitting; the same pixels under two labels refuse."""
    from vision_rl.roboflow import (
        IMAGE_SUFFIXES,
        _image_path,
        _row,
        _write_import,
        import_classification,
    )

    root = Path(images).resolve()
    if not root.is_dir():
        raise ValueError(f"image folder does not exist: {root}")
    top = {p.name.lower() for p in root.iterdir() if p.is_dir()}
    if top & SPLIT_NAMES:
        # An export that is already split: use it exactly as given, or refuse. Never treat
        # "train"/"valid" as labels.
        if not ("train" in top and "test" in top and top & {"valid", "validation", "val"}):
            raise ValueError(
                "folder looks pre-split but needs train/, test/ and one of valid/, "
                "validation/ or val/ (or use one subfolder per label instead)"
            )
        result = import_classification(root, output)
        result["source"] = "existing train/validation/test split"
        return result
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("split fractions must be between 0 and 1")
    labels = [p.name for p in _label_dirs(root)]
    owner, duplicates, per_label = {}, 0, {}
    for index, name in enumerate(labels):
        label_dir = root / name
        paths = sorted(
            p
            for p in label_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and _visible(p, root)
        )
        unique = []
        for path in paths:
            digest = image_hash(_image_path(path, label_dir))
            if digest in owner:
                if owner[digest][0] != name:
                    raise ValueError(
                        f"the same image is labelled both {owner[digest][0]!r} "
                        f"({owner[digest][1]}) and {name!r} ({path}); keep one"
                    )
                duplicates += 1
                continue
            owner[digest] = (name, path)
            unique.append(path.resolve())
        if len(unique) < MIN_PER_LABEL:
            raise ValueError(
                f"label {name!r} has {len(unique)} distinct images; need at least {MIN_PER_LABEL}"
            )
        per_label[index] = unique
    rng = random.Random(seed)
    rows = {"train": [], "validation": [], "test": []}
    for index, paths in per_label.items():
        paths = paths[:]
        rng.shuffle(paths)
        n_test = max(1, round(len(paths) * test_fraction))
        n_val = max(1, round(len(paths) * validation_fraction))
        if len(paths) - n_test - n_val < 1:
            n_test, n_val = 1, 1
        parts = {
            "test": paths[:n_test],
            "validation": paths[n_test : n_test + n_val],
            "train": paths[n_test + n_val :],
        }
        for split, items in parts.items():
            rows[split] += [
                _row(p, split, labels, index, FOLDER_PROMPT, "classification") for p in items
            ]
    result = _write_import(rows, output)
    result.update(classes=labels, source="split by label", seed=seed, duplicates_dropped=duplicates)
    return result


def _test_accuracy(manifest: Path, checkpoint: Path, device: str) -> dict:
    out = checkpoint.with_name(checkpoint.stem + "-test.json")
    result = evaluate_count_policy(manifest, checkpoint, out, device=device)
    return {"accuracy": result["accuracy"], "correct": result["correct"], "n": result["n"]}


def train_folder(
    images: str | Path,
    output: str | Path,
    *,
    device: str = "auto",
    quick: bool = False,
    seed: int = 0,
    encoder_weights: Path | None = None,
) -> dict:
    """Prepare data, train (SFT then RL), select on validation, then score once on test."""
    import shutil

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output folder must be new or empty: {output}")
    data = build_dataset(images, output / "data", seed=seed)
    manifests = {k: Path(v) for k, v in data["manifests"].items()}
    model = output / "model"
    report = train_count_policy(
        manifests["train"],
        manifests["validation"],
        model,
        device=device,
        seeds=(7,) if quick else (7, 19),
        hidden_sizes=(0,) if quick else (0, 128),
        sft_epochs=40 if quick else 160,
        rl_steps=200 if quick else 800,
        encoder_weights=encoder_weights,
    )
    experiments = {e["candidate"]: e for e in report["experiments"]}
    sft_val = experiments[report["sft_candidate"]]["sft_validation"]
    rl_val = experiments[report["selected_candidate"]]["rl_validation"]
    # Deploy whichever stage validation prefers (a tie keeps RL); test is read only after.
    chosen = "rl" if rl_val["accuracy"] >= sft_val["accuracy"] else "sft"
    shutil.copyfile(model / ("policy.pt" if chosen == "rl" else "sft.pt"), model / "best.pt")
    test = manifests["test"]
    scores = {
        "sft": _test_accuracy(test, model / "sft.pt", device),
        "rl": _test_accuracy(test, model / "policy.pt", device),
    }
    train_labels = Counter(r["answer"] for r in load_manifest(manifests["train"]))
    test_rows = load_manifest(test)
    majority = train_labels.most_common(1)[0][0]
    baseline = sum(r["answer"] == majority for r in test_rows) / len(test_rows)
    summary = {
        "labels": data["classes"],
        "images": data["counts"],
        "duplicates_dropped": data.get("duplicates_dropped", 0),
        "validation_accuracy": {"sft": sft_val["accuracy"], "rl": rl_val["accuracy"]},
        "test_accuracy": {
            "majority_label_baseline": baseline,
            "sft": scores["sft"]["accuracy"],
            "rl": scores["rl"]["accuracy"],
        },
        "deployed_model": chosen,
        "model": str(model / "best.pt"),
        "quick": quick,
        "device": report["encoder_device"],
    }
    write_json(output / "summary.json", summary)
    return summary


def predict(model: str | Path, images: list[str | Path], *, device: str = "auto") -> list[dict]:
    """Label each image with a train-folder model: top label and its probability."""
    from vision_rl.roboflow import IMAGE_SUFFIXES

    torch, _ = _stack()
    checkpoint = Path(model)
    for candidate in (checkpoint / "model" / "best.pt", checkpoint / "best.pt"):
        if checkpoint.is_dir() and candidate.is_file():
            checkpoint = candidate
            break
    if not checkpoint.is_file():
        raise ValueError(f"no model at {model}; pass the train-folder output folder")
    saved = _load_head_checkpoint(checkpoint)
    meta = saved["metadata"]
    if meta.get("task") != "classification":
        raise ValueError("predict needs a model trained by train-folder (a classification task)")
    encoder = _checked_encoder(checkpoint, meta)
    paths = []
    for item in images:
        path = Path(item).resolve()
        if path.is_dir():
            paths += sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and _visible(p, path)
            )
        else:
            paths.append(path)
    if not paths:
        raise ValueError("no images to label")
    x = _features([{"image": str(p)} for p in paths], encoder, select_device(device))
    x = (x - saved["feature_mean"]) / saved["feature_scale"]
    head = _head(meta["feature_dim"], len(meta["choices"]), meta["hidden"])
    head.load_state_dict(saved["state_dict"])
    head.eval()
    with torch.inference_mode():
        probs = head(x).softmax(-1)
    best = probs.argmax(-1).tolist()
    return [
        {"image": str(p), "label": meta["choices"][k], "confidence": round(float(probs[i, k]), 4)}
        for i, (p, k) in enumerate(zip(paths, best, strict=True))
    ]


def make_example(output: str | Path, per_label: int = 24, seed: int = 0) -> Path:
    """A tiny synthetic dataset (circles, squares, triangles) to try train-folder offline."""
    from PIL import ImageDraw

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"example folder must be new or empty: {output}")
    rng = random.Random(seed)
    for shape in ("circle", "square", "triangle"):
        folder = output / shape
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_label):
            image = Image.new("RGB", (128, 128), tuple(rng.randrange(150, 256) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            size = rng.randrange(30, 60)
            cx, cy = rng.randrange(size, 128 - size), rng.randrange(size, 128 - size)
            color = tuple(rng.randrange(0, 120) for _ in range(3))
            box = (cx - size, cy - size, cx + size, cy + size)
            if shape == "circle":
                draw.ellipse(box, fill=color)
            elif shape == "square":
                draw.rectangle(box, fill=color)
            else:
                points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
                draw.polygon(points, fill=color)
            image.save(folder / f"{shape}-{i:03d}.png")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    train.add_argument("--hidden-sizes", type=int, nargs="+", default=[0, 128])
    train.add_argument("--sft-epochs", type=int, default=160)
    train.add_argument("--rl-steps", type=int, default=800)
    train.add_argument("--encoder-weights", type=Path)
    train.add_argument("--no-augment", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    evaluate.add_argument("--trusted-encoder", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        result = train_count_policy(
            args.train,
            args.validation,
            args.output,
            device=args.device,
            seeds=args.seeds,
            hidden_sizes=args.hidden_sizes,
            sft_epochs=args.sft_epochs,
            rl_steps=args.rl_steps,
            encoder_weights=args.encoder_weights,
            augment=not args.no_augment,
        )
    else:
        result = evaluate_count_policy(
            args.manifest,
            args.checkpoint,
            args.output,
            device=args.device,
            trusted_encoder=args.trusted_encoder,
        )
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
