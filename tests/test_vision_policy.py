"""Optional torch tests for reward gradients, label isolation, and training splits."""

import hashlib
import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from vision_rl import vision_policy as vp  # noqa: E402


def test_reinforce_gradient_matches_exact_expected_reward():
    torch.manual_seed(917)
    logits = torch.tensor([[0.3, -0.1, 0.7]], requires_grad=True)
    target = torch.tensor([1])
    loss, _ = vp.reinforce_loss(logits, target, samples=100000, entropy_weight=0, kl=0)
    sampled_gradient = torch.autograd.grad(loss, logits)[0]
    actions = torch.arange(3)[None, :]
    exact_loss = -(logits.softmax(-1) * vp.shaped_reward(actions, target[:, None], 3)).sum()
    exact_gradient = torch.autograd.grad(exact_loss, logits)[0]
    assert torch.allclose(sampled_gradient, exact_gradient, atol=0.005)
    assert sampled_gradient[0, 1] < 0  # Gradient descent increases the correct action.


def test_reward_prioritizes_exact_count_and_penalizes_distance():
    actions = torch.tensor([[0, 1, 2, 3]])
    assert vp.shaped_reward(actions, torch.tensor([[1]]), 4).tolist()[0] == pytest.approx(
        [-1 / 3, 1, -1 / 3, -2 / 3]
    )


def test_count_policy_inference_does_not_read_answer(tmp_path, monkeypatch):
    encoder = torch.nn.Linear(2, 2)
    torch.save(encoder.state_dict(), tmp_path / "encoder.pt")
    metadata = {
        "choices": ["0", "1"],
        "prompt": "How many pawns?",
        "encoder_file": "encoder.pt",
        "encoder_sha256": hashlib.sha256((tmp_path / "encoder.pt").read_bytes()).hexdigest(),
        "encoder_state_sha256": vp.state_fingerprint(encoder.state_dict()),
        "encoder": vp.ENCODER_ID,
        "feature_dim": 1024,
        "hidden": 0,
    }
    head = torch.nn.Linear(1024, 2)
    vp._save_policy(tmp_path / "policy.pt", head, metadata, torch.zeros(1024), torch.ones(1024))
    monkeypatch.setattr(vp, "_encoder", lambda path: encoder)
    monkeypatch.setattr(vp, "_features", lambda rows, enc, device: torch.full((1, 1024), 0.5))
    # Intentionally omit answer entirely from the observation. Inference must work.
    observations = [
        {
            "id": "x",
            "choices": ["0", "1"],
            "category": "counting",
            "prompt": "How many pawns?",
            "image": "irrelevant-in-this-unit-test",
        }
    ]
    prediction = vp.predict_count_policy(observations, tmp_path / "policy.pt", device="cpu")
    observations[0]["answer"] = "do not read this"
    assert vp.predict_count_policy(observations, tmp_path / "policy.pt", device="cpu") == prediction
    (tmp_path / "encoder.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        vp.predict_count_policy(observations, tmp_path / "policy.pt", device="cpu")
    # Verifier uses its trusted encoder even when the submitted sibling is corrupt.
    assert (
        vp.predict_count_policy(
            observations,
            tmp_path / "policy.pt",
            device="cpu",
            trusted_encoder=tmp_path / "trusted.pt",
        )
        == prediction
    )


def test_oversized_hidden_metadata_rejected_before_allocation(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": {
                "encoder": vp.ENCODER_ID,
                "feature_dim": 1024,
                "hidden": 2**40,
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="hidden size"):
        vp._load_head_checkpoint(path)


def test_training_rejects_test_manifest_before_model_download(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image)
    row = {
        "id": "x",
        "image": str(image),
        "split": "test",
        "prompt": "Count",
        "choices": ["0", "1"],
        "answer": "A",
        "category": "counting",
    }
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(vp, "_encoder", lambda _: pytest.fail("must not download a model"))
    with pytest.raises(ValueError, match="expected train"):
        vp.train_count_policy(manifest, manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_count_schema_rejects_mismatched_question_and_non_count_choices():
    row = {"choices": ["0", "1"], "category": "counting", "prompt": "Count pawns"}
    with pytest.raises(ValueError, match="question"):
        vp._count_schema([row, {**row, "prompt": "Count kings"}])
    with pytest.raises(ValueError, match="order"):
        vp._count_schema([{**row, "choices": ["1", "0"]}])
