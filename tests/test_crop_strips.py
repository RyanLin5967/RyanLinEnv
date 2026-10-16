"""Side-strip geometry, high-IoU curriculum, immutable replay, and resumability."""

import fcntl
import json
import os

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from vision_rl import crop_strips as cs  # noqa: E402


def toy(tmp_path, *, name="a.png", value=200):
    image = np.zeros((80, 100, 3), np.uint8)
    image[20:60, 25:75] = [value, 80, 40]
    Image.fromarray(image).save(tmp_path / name)
    info = {"id": 7, "file_name": name, "width": 100, "height": 80}
    target = {"id": 1, "image_id": 7, "category_id": 1, "bbox": [25, 20, 50, 40],
              "area": 2000, "iscrowd": 0}
    predictions = [
        {"image_id": 7, "category_id": 1, "bbox": [24.3, 19.7, 50, 40], "score": 0.9},
        {"image_id": 7, "category_id": 1, "bbox": [24, 20, 50, 40], "score": 0.7},
        {"image_id": 7, "category_id": 1, "bbox": [-3, 4, 10, 10], "score": 0.001},
    ]
    return info, target, predictions


def small_config(**kwargs):
    return cs.StripConfig(normal_samples=128, tangent_samples=32, width=16,
                           cpu_threads=2, image_workers=1, **kwargs)


def test_every_side_samples_original_pixels_with_inward_normal():
    y, x = np.meshgrid(np.arange(80), np.arange(100), indexing="ij")
    image = Image.fromarray(np.stack((x, y, x * 0), -1).astype(np.uint8))
    config = small_config()
    box = np.array([25, 20, 75, 60], np.float32)
    strips = cs.side_strips(image, box, config)
    assert strips.shape == (4, 32, 128, 3)
    middle = config.normal_samples // 2
    # Left/top increase into the box; right/bottom decrease into it.
    for side, channel, boundary, direction in [(0, 0, 25, 1), (1, 1, 20, 1),
                                                (2, 0, 75, -1), (3, 1, 60, -1)]:
        profile = strips[side, 16, :, channel].astype(float)
        assert abs(profile[middle] - boundary) <= 1
        assert direction * (profile[-1] - profile[0]) > 10


def test_fractional_edge_encode_decode_preserves_inward_signs():
    boxes = np.array([[10.7, 20.2, 111.3, 92.9]], np.float32)
    targets = boxes + np.array([[1.3, -0.6, -2.4, 3.7]], np.float32)
    encoded = cs.edge_displacements(boxes, targets)
    assert encoded[0, 0] > 0 and encoded[0, 1] < 0
    assert encoded[0, 2] > 0 and encoded[0, 3] < 0
    np.testing.assert_allclose(cs.decode_edges(boxes, encoded), targets, atol=1e-5)


def test_distribution_bins_align_with_actual_sampled_pixel_centers():
    config = small_config()
    model = cs.make_model(config)
    centers = ((torch.arange(config.normal_samples) + 0.5)
               * (2 * config.normal_span / config.normal_samples) - config.normal_span)
    assert torch.equal(model.axis, centers)
    indices = cs.distribution_targets(centers, config) * (config.normal_samples - 1)
    assert torch.allclose(indices, torch.arange(config.normal_samples).float(), atol=1e-5)


def test_actual_proposals_duplicates_and_resampled_jitter_are_separate(tmp_path):
    info, target, predictions = toy(tmp_path)
    config = small_config()
    bank = cs.frame_bank(tmp_path, info, predictions, [0, 1], config,
                          annotations=[target], epoch=0)
    replay = cs.frame_bank(tmp_path, info, predictions, [0, 1], config,
                            annotations=[target], epoch=0)
    changed = cs.frame_bank(tmp_path, info, predictions, [0, 1], config,
                             annotations=[target], epoch=1)
    assert bank["index"].tolist() == [0, 1, -1]
    assert bank["unique"].tolist() == [True, False, True]
    assert bank["context"][1, 1] > 0.9  # Duplicate has higher-ranked overlapping evidence.
    assert bank["context"][0, 0] == pytest.approx(0.9)
    np.testing.assert_array_equal(bank["proposal"], replay["proposal"])
    np.testing.assert_array_equal(bank["proposal"][:2], changed["proposal"][:2])
    assert not np.array_equal(bank["proposal"][2], changed["proposal"][2])
    assert cs.jitter_scales(0, 12)[1] == [0.2, 0.5, 0.3]
    assert cs.jitter_scales(11, 12)[1] == [0.05, 0.5, 0.45]


def test_full_proposal_list_survives_refinement_and_identity_is_exact(tmp_path):
    info, _, predictions = toy(tmp_path)
    bank = cs.frame_bank(tmp_path, info, predictions, [0], small_config())
    delta = np.array([[0.01, -0.02, 0.03, -0.01]], np.float32)
    quality = np.array([0.8], np.float32)
    assert cs.apply_refinement(predictions, bank["index"], bank["proposal"], delta, quality,
                                blend=0, quality_blend=0) == predictions
    refined = cs.apply_refinement(predictions, bank["index"], bank["proposal"], delta, quality)
    assert len(refined) == len(predictions)
    assert refined[1:] == predictions[1:]
    assert refined[0]["bbox"] != predictions[0]["bbox"]


def test_model_exposes_native_spatial_features_and_learns_boundary_loss(tmp_path):
    torch.set_num_threads(2)
    info, target, predictions = toy(tmp_path)
    config = small_config()
    bank = cs.frame_bank(tmp_path, info, predictions, [0], config, annotations=[target])
    model = cs.make_model(config)
    pixels, context = cs._batch(bank, "cpu")
    first = model(pixels, context)
    assert first["logits"].shape == (2, 4, 128)
    assert first["side_features"].shape == (2, 4, 32)
    assert first["displacements"].abs().max() < 1e-6
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    before = cs.supervised_loss(first, bank, config).item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = cs.supervised_loss(model(pixels, context), bank, config)
        loss.backward()
        optimizer.step()
    assert cs.supervised_loss(model(pixels, context), bank, config).item() < before
    assert model.attention.weight.grad.abs().sum() > 0


def test_bound_numeric_loader_exact_replay_hash_pin_and_nonregular_rejection(tmp_path):
    pytest.importorskip("safetensors")
    torch.set_num_threads(2)
    config = small_config()
    model = cs.make_model(config).eval()
    path = tmp_path / "model.safetensors"
    cs.save_model(model, path, config, {"training_detector_sha256": "a" * 64})
    restored, _, _ = cs.load_model(path, expected_sha256=cs._sha(path))
    pixels, context = torch.randn(1, 4, 3, 32, 128), torch.randn(1, 6)
    with torch.inference_mode():
        before, after = model(pixels, context), restored(pixels, context)
    assert all(torch.equal(before[key], after[key]) for key in before)
    with pytest.raises(ValueError, match="SHA differs"):
        cs.load_model(path, expected_sha256="b" * 64)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        cs.load_model(fifo)
    link = tmp_path / "link.safetensors"
    link.symlink_to(path)
    with pytest.raises(OSError):
        cs.load_model(link)
    parent_link = tmp_path / "linked"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        cs.load_model(parent_link / "model.safetensors")


def test_image_only_prediction_pins_composition_without_relabeling_training(tmp_path):
    pytest.importorskip("safetensors")
    info, _, predictions = toy(tmp_path)
    config = small_config()
    path = tmp_path / "model.safetensors"
    cs.save_model(cs.make_model(config), path, config,
                   {"training_detector_sha256": "a" * 64, "blend": 0.0})
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({"images": [info]}))
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps(predictions))
    composition = {"checkpoint_sha256": cs._sha(path), "proposals_sha256": cs._sha(proposals),
                   "manifest_sha256": cs._sha(manifest), "deployment_detector_sha256": "b" * 64}
    output = tmp_path / "output.json"
    result = cs.refine_manifest(path, manifest, tmp_path, proposals, output,
                                composition=composition)
    assert json.loads(output.read_text()) == predictions
    assert result["training_detector_sha256"] == "a" * 64
    assert result["deployment_detector_sha256"] == "b" * 64
    assert not result["labels_available_to_predictor"]
    composition["proposals_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Frozen proposals"):
        cs.refine_manifest(path, manifest, tmp_path, proposals, output, composition=composition)
    composition["proposals_sha256"] = cs._sha(proposals)
    composition["image_tree_sha256"] = result["image_tree_sha256"]
    with Image.open(tmp_path / "a.png") as source:
        changed = source.copy()
    changed.putpixel((0, 0), (1, 2, 3))
    changed.save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="Frozen image tree"):
        cs.refine_manifest(path, manifest, tmp_path, proposals, output, composition=composition)


def test_train_output_lock_rejects_concurrent_writer(tmp_path):
    with (tmp_path / ".train.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another trainer"):
            cs.train(tmp_path, tmp_path, tmp_path, tmp_path, "a" * 64, small_config())


def test_decoded_pixel_fingerprint_ignores_different_container_bytes(tmp_path):
    info, _, _ = toy(tmp_path)
    from PIL import PngImagePlugin

    image = Image.open(tmp_path / "a.png")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("variant", "different compressed container metadata")
    image.save(tmp_path / "b.png", pnginfo=metadata)
    other = {**info, "file_name": "b.png"}
    files, pixels = cs._image_hashes(tmp_path, [info, other])
    assert files[0] != files[1]
    assert pixels[0] == pixels[1]


def test_epoch_resume_commits_best_model_and_matches_uninterrupted_weights(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    pytest.importorskip("pycocotools")
    source = tmp_path / "data"
    proposals = []
    for split, color in [("train", 180), ("valid", 220)]:
        root = source / split
        root.mkdir(parents=True)
        info, target, predictions = toy(root, name=f"{split}.png", value=color)
        (root / "_annotations.coco.json").write_text(json.dumps({
            "images": [info], "annotations": [target], "categories": [{"id": 1, "name": "pill"}],
        }))
        proposal = tmp_path / f"{split}-proposals.json"
        proposal.write_text(json.dumps(predictions))
        proposals.append(proposal)
    config = small_config(epochs=2, batch_size=4, evaluate_every=1,
                           blend_candidates=(0.5,), quality_candidates=(0.0,))
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    cs.train(source, *proposals, complete, "a" * 64, config)
    original = cs.os.replace

    def fail_after_epoch_commit(source_path, destination):
        original(source_path, destination)
        if destination == interrupted / "trainer-state.pt":
            raise RuntimeError("simulated crash after epoch commit before best publication")

    monkeypatch.setattr(cs.os, "replace", fail_after_epoch_commit)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cs.train(source, *proposals, interrupted, "a" * 64, config)
    monkeypatch.setattr(cs.os, "replace", original)
    cs.train(source, *proposals, interrupted, "a" * 64, config, resume=True)
    expected, _, _ = cs.load_model(complete / "last.safetensors")
    replay, _, _ = cs.load_model(interrupted / "last.safetensors")
    assert all(torch.equal(tensor, replay.state_dict()[name])
               for name, tensor in expected.state_dict().items())
    expected_best, _, _ = cs.load_model(complete / "best.safetensors")
    replay_best, _, _ = cs.load_model(interrupted / "best.safetensors")
    assert all(torch.equal(tensor, replay_best.state_dict()[name])
               for name, tensor in expected_best.state_dict().items())
    report = json.loads((interrupted / "best-evaluation.json").read_text())
    metadata = json.loads((interrupted / "best.json").read_text())
    assert report["AP"] == metadata["AP"]
