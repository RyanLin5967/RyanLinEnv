"""A genuine, small REINFORCE policy and a bounded training experiment agent."""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from vision_rl.data import assert_disjoint, fingerprint, load_manifest, write_json
from vision_rl.spatial import features


def probabilities(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = x @ weights
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def save_policy(path: Path, weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=weights)


def load_policy(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.stat().st_size > 16384:
        raise ValueError("invalid CPU policy: checkpoint exceeds 16 KiB")
    # Bound expansion and validate the NPY header before NumPy allocates an array.
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != 1 or entries[0].filename != "weights.npy":
                raise ValueError("invalid CPU policy fields")
            if entries[0].file_size > 4096:
                raise ValueError("invalid CPU policy expanded size")
            with archive.open(entries[0]) as data:
                version = np.lib.format.read_magic(data)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(data)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(data)
                else:
                    raise ValueError("unsupported CPU policy array format")
                if shape != (3, 4) or dtype.kind != "f":
                    raise ValueError("invalid CPU policy array header")
    except (zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("invalid CPU policy archive") from exc
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"weights"}:
            raise ValueError("invalid CPU policy fields")
        weights = artifact["weights"]
    if weights.shape != (3, 4) or weights.dtype.kind != "f" or not np.isfinite(weights).all():
        raise ValueError("invalid CPU policy weights")
    if np.abs(weights).max() > 10000:
        raise ValueError("invalid CPU policy magnitude")
    return weights


def train(
    train: Path,
    validation: Path,
    output: Path,
    steps: int = 800,
    learning_rate: float = 0.03,
    seed: int = 42,
    checkpoint: Path | None = None,
) -> dict:
    if steps < 1 or not 0 < learning_rate <= 1:
        raise ValueError("steps must be positive and learning rate in (0, 1]")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"training output must be empty: {output}")
    train_rows = load_manifest(train, "train")
    val_rows = load_manifest(validation, "validation")
    assert_disjoint(train_rows, val_rows)
    x, vx = features(train_rows), features(val_rows)
    y = np.array([ord(row["answer"]) - 65 for row in train_rows])
    vy = np.array([ord(row["answer"]) - 65 for row in val_rows])
    rng = np.random.default_rng(seed)
    weights = load_policy(checkpoint) if checkpoint else np.zeros((3, 4), dtype=np.float64)
    initial = weights.copy()
    initial_accuracy = float((np.argmax(vx @ weights, axis=1) == vy).mean())
    best, best_score, best_step = weights.copy(), initial_accuracy, 0
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    baseline = 0.25
    history = []
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(x), size=64)
        batch = x[indices]
        probs = probabilities(batch, weights)
        actions = (rng.random((64, 1)) > probs.cumsum(axis=1)).sum(axis=1).clip(0, 3)
        # The update receives only the reward for each sampled action, not a gold action target.
        rewards = (actions == y[indices]).astype(float)
        advantage = rewards - baseline
        score_gradient = np.eye(4)[actions] - probs
        gradient = batch.T @ (advantage[:, None] * score_gradient) / 64
        # Adam ascent on the policy-gradient reward objective.
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        weights += learning_rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        baseline = 0.95 * baseline + 0.05 * float(rewards.mean())
        if step == 1 or step % 50 == 0 or step == steps:
            score = float((np.argmax(vx @ weights, axis=1) == vy).mean())
            history.append(
                {
                    "step": step,
                    "sampled_train_reward": float(rewards.mean()),
                    "validation_accuracy": score,
                    "reward_baseline": baseline,
                }
            )
            if score > best_score:
                best, best_score, best_step = weights.copy(), score, step
    output.mkdir(parents=True, exist_ok=True)
    save_policy(output / "initial.npz", initial)
    save_policy(output / "policy.npz", best)
    metadata = {
        "algorithm": "REINFORCE with EMA baseline and Adam",
        "encoder": "frozen red/blue centroid differences from image pixels",
        "policy": "3x4 linear softmax",
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": 64,
        "initial_validation_accuracy": initial_accuracy,
        "best_validation_accuracy": best_score,
        "selected_step": best_step,
        "train_fingerprint": fingerprint(train_rows),
        "validation_fingerprint": fingerprint(val_rows),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "weights_changed": bool(not np.array_equal(initial, best)),
    }
    write_json(output / "training.json", metadata)
    (output / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in history))
    return metadata


def train_agent(
    train_path: Path,
    validation_path: Path,
    output: Path,
    steps: int = 800,
    candidates: int = 3,
    seed: int = 42,
) -> dict:
    """A deterministic experiment controller, not an LLM or a trainable agent policy."""
    if not 1 <= candidates <= 3:
        raise ValueError("candidate budget must be between 1 and 3")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"agent output must be empty: {output}")
    runs = []
    for index, rate in enumerate((0.01, 0.03, 0.08)[:candidates]):
        run = train(
            train_path,
            validation_path,
            output / f"candidate-{index}",
            steps=steps,
            learning_rate=rate,
            seed=seed + index,
        )
        runs.append({"candidate": index, **run})
    winner = max(runs, key=lambda run: run["best_validation_accuracy"])
    selected = output / f"candidate-{winner['candidate']}"
    shutil.copyfile(selected / "policy.npz", output / "policy.npz")
    shutil.copyfile(selected / "initial.npz", output / "initial.npz")
    result = {
        "agent": "bounded-experiment-controller",
        "budget": {"candidates": candidates, "steps_per_candidate": steps},
        "selection_split": "validation",
        "selected_candidate": winner["candidate"],
        "experiments": runs,
    }
    write_json(output / "agent.json", result)
    return result
