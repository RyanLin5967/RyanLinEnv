"""Pixel geometry, label isolation, safe replay, and real gradient checks."""

import json
import os
from dataclasses import asdict, replace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from vision_rl import crop_refinement as cr  # noqa: E402


def toy_data(tmp_path):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 25:75] = [220, 80, 40]
    Image.fromarray(image).save(tmp_path / "a.png")
    images = [{"id": 7, "file_name": "a.png", "width": 100, "height": 80}]
    gt = [{"image_id": 7, "category_id": 1, "bbox": [25, 20, 50, 40]}]
    proposals = [
        {"image_id": 7, "category_id": 1, "bbox": [23.2, 21.7, 50, 38], "score": 0.9},
        {"image_id": 7, "category_id": 1, "bbox": [24, 20, 50, 40], "score": 0.7},
        {"image_id": 7, "category_id": 1, "bbox": [-3, 4, 10, 10], "score": 0.001},
    ]
    return images, gt, proposals


def test_fractional_crop_coordinates_roundtrip_and_native_pixels():
    box = np.array([-2.7, 8.1, 93.7, 49.2], np.float32)
    extent = cr.crop_extent(box, 0.30)
    np.testing.assert_allclose(cr.absolute_box(cr.relative_box(box, extent), extent), box,
                               atol=1e-5)
    image = Image.fromarray(np.tile(np.arange(100, dtype=np.uint8)[None, :, None], (80, 1, 3)))
    crop = cr.crop_pixels(image, np.array([20, 10, 60, 50], np.float32), 40)
    assert crop.shape == (40, 40, 3)
    assert crop[20, 0, 0] == 20
    assert crop[20, -1, 0] == 59


def test_training_assigns_one_to_one_quality_and_deterministic_jitters(tmp_path):
    images, gt, proposals = toy_data(tmp_path)
    config = cr.CropConfig(crop_size=64, width=16, jitter_per_gt=2)
    bank = cr.build_bank(tmp_path, images, proposals, config, annotations=gt, training=True)
    assert bank["index"].tolist() == [0, 1, -1, -1]
    assert bank["positive"].tolist() == [True] * 4
    assert bank["unique"].tolist() == [True, False, True, True]
    replay = cr.build_bank(tmp_path, images, proposals, config, annotations=gt, training=True)
    for key in bank:
        np.testing.assert_array_equal(bank[key], replay[key])


def test_budget_retains_all_unrefined_low_confidence_proposals(tmp_path):
    images, _, proposals = toy_data(tmp_path)
    config = cr.CropConfig(crop_size=64, width=16, max_per_image=1)
    bank = cr.build_bank(tmp_path, images, proposals, config)
    assert bank["index"].tolist() == [0]
    assert not bank["positive"].any()
    extent = bank["extent"][0]
    target = np.array([25, 20, 75, 60], np.float32)
    refined = cr.apply_refinement(proposals, bank, cr.relative_box(target, extent)[None],
                                  np.ones(1), blend=1)
    assert len(refined) == len(proposals)
    np.testing.assert_allclose(refined[0]["bbox"], [25, 20, 50, 40], atol=1e-5)
    assert refined[1:] == proposals[1:]
    assert cr.apply_refinement(proposals, bank, cr.relative_box(target, extent)[None],
                               np.ones(1), blend=0) == proposals


def test_jitter_training_bank_cannot_be_submitted_as_predictions(tmp_path):
    images, gt, proposals = toy_data(tmp_path)
    config = cr.CropConfig(crop_size=64, width=16)
    bank = cr.build_bank(tmp_path, images, proposals, config, annotations=gt, training=True)
    with pytest.raises(ValueError, match="training-only"):
        cr.apply_refinement(proposals, bank, bank["target"], np.ones(len(bank["index"])))


def test_distribution_loss_has_correct_fractional_gradient():
    logits = torch.zeros(1, 1, 5, requires_grad=True)
    target = torch.tensor([[0.3]])  # bins 1 and 2 carry .8/.2 mass.
    loss = cr.distribution_loss(logits, target)
    loss.backward()
    assert logits.grad.flatten().tolist() == pytest.approx([0.2, -0.6, 0, 0.2, 0.2], abs=1e-6)


def test_spatial_model_identity_prior_and_real_localization_gradients():
    torch.set_num_threads(2)
    config = cr.CropConfig(crop_size=64, width=16)
    model = cr.make_model(config)
    pixels = torch.randn(2, 3, 64, 64)
    first = config.padding / (1 + 2 * config.padding)
    expected = torch.tensor([first, first, 1 - first, 1 - first])
    initial = model(pixels)
    assert torch.allclose(initial["edges"], expected[None].expand(2, 4), atol=1e-4)
    targets = expected[None].repeat(2, 1)
    targets[:, [0, 2]] += 0.03
    pos = torch.ones(2, dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    before, _ = cr.supervised_loss(initial, targets, pos, pos)
    for _ in range(5):
        loss, _ = cr.supervised_loss(model(pixels), targets, pos, pos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after, _ = cr.supervised_loss(model(pixels), targets, pos, pos)
    assert after.item() < before.item()
    # At least one spatial encoder parameter received a true localization gradient.
    assert model.stem[0].weight.grad.abs().sum() > 0


def test_empty_positive_loss_remains_finite_and_trains_quality():
    torch.set_num_threads(2)
    model = cr.make_model(cr.CropConfig(crop_size=64, width=16))
    output = model(torch.zeros(2, 3, 64, 64))
    positive = torch.zeros(2, dtype=torch.bool)
    loss, _ = cr.supervised_loss(output, torch.zeros(2, 4), positive, positive)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.quality[-1].bias.grad > 0


def test_numeric_export_exact_replay_and_detector_identity(tmp_path):
    pytest.importorskip("safetensors")
    torch.set_num_threads(2)
    config = cr.CropConfig(crop_size=64, width=16)
    model = cr.make_model(config).eval()
    checkpoint = tmp_path / "crop.safetensors"
    cr.save_model(model, checkpoint, config, {"detector_sha256": "a" * 64})
    replay, _, _ = cr.load_model(checkpoint, expected_detector_sha256="a" * 64)
    pixels = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        for key, value in model(pixels).items():
            assert torch.equal(value, replay(pixels)[key])
    with pytest.raises(ValueError, match="different detector"):
        cr.load_model(checkpoint, expected_detector_sha256="b" * 64)


def test_public_prediction_never_opens_annotation_file(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    images, _, proposals = toy_data(tmp_path)
    config = cr.CropConfig(crop_size=64, width=16)
    checkpoint = tmp_path / "crop.safetensors"
    cr.save_model(cr.make_model(config), checkpoint, config,
                  {"detector_sha256": "a" * 64, "blend": 0.0})
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({"images": images}))
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(json.dumps(proposals))
    from pathlib import Path

    original = Path.open

    def guarded_open(path, *args, **kwargs):
        assert "annotation" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    output = tmp_path / "refined.json"
    report = cr.refine_manifest(checkpoint, manifest, tmp_path, proposal_path, output)
    assert json.loads(output.read_text()) == proposals
    assert report["proposals"] == 3
    assert report["refined_proposals"] == 2
    assert not report["labels_available_to_predictor"]


def test_label_bearing_manifest_rejected(tmp_path):
    pytest.importorskip("safetensors")
    images, gt, proposals = toy_data(tmp_path)
    checkpoint = tmp_path / "crop.safetensors"
    cr.save_model(cr.make_model(cr.CropConfig(crop_size=64, width=16)), checkpoint,
                  cr.CropConfig(crop_size=64, width=16), {"detector_sha256": "a" * 64})
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({"images": images, "annotations": gt}))
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(json.dumps(proposals))
    with pytest.raises(ValueError, match="without labels"):
        cr.refine_manifest(checkpoint, manifest, tmp_path, proposal_path, tmp_path / "out.json")


def test_predictions_reject_nonfinite_boxes_and_unknown_images(tmp_path):
    images, _, proposals = toy_data(tmp_path)
    path = tmp_path / "proposals.json"
    proposals[0]["score"] = float("nan")
    path.write_text(json.dumps(proposals))
    with pytest.raises(ValueError, match="finite"):
        cr._load_predictions(path, images)
    proposals[0]["score"] = 0.9
    proposals[0]["image_id"] = 99
    path.write_text(json.dumps(proposals))
    with pytest.raises(ValueError, match="unknown"):
        cr._load_predictions(path, images)


def test_epoch_resume_replays_exact_weights_and_rejects_changed_inputs(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    pytest.importorskip("pycocotools")
    source = tmp_path / "data"
    predictions = []
    for split in ("train", "valid"):
        root = source / split
        root.mkdir(parents=True)
        images, annotations, proposals = toy_data(root)
        filename = f"{split}.png"
        (root / "a.png").rename(root / filename)
        images[0]["file_name"] = filename
        annotations[0].update(id=1, area=2000, iscrowd=0)
        (root / "_annotations.coco.json").write_text(json.dumps({
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "pill"}],
        }))
        prediction_path = tmp_path / f"{split}-proposals.json"
        prediction_path.write_text(json.dumps(proposals))
        predictions.append(prediction_path)
    config = cr.CropConfig(crop_size=64, width=16, epochs=2, batch_size=4,
                           cpu_threads=2, evaluate_every=1, jitter_per_gt=1,
                           blend_candidates=(0.5,), quality_candidates=(0.0,))
    complete = tmp_path / "complete"
    interrupted = tmp_path / "interrupted"
    cr.train(source, *predictions, complete, "a" * 64, config)
    original = cr.os.replace

    def interrupt_after_checkpoint(source_path, destination):
        original(source_path, destination)
        if destination == interrupted / "trainer-state.pt":
            raise RuntimeError("simulated interruption after atomic checkpoint")

    monkeypatch.setattr(cr.os, "replace", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cr.train(source, *predictions, interrupted, "a" * 64, config)
    monkeypatch.setattr(cr.os, "replace", original)
    cr.train(source, *predictions, interrupted, "a" * 64, config, resume=True)
    expected, _, _ = cr.load_model(complete / "last.safetensors")
    replay, _, _ = cr.load_model(interrupted / "last.safetensors")
    assert all(torch.equal(tensor, replay.state_dict()[key])
               for key, tensor in expected.state_dict().items())
    proposals = json.loads(predictions[0].read_text())
    proposals[0]["bbox"][0] += 1
    predictions[0].write_text(json.dumps(proposals))
    with pytest.raises(ValueError, match="Resume inputs"):
        cr.train(source, *predictions, interrupted, "a" * 64, config, resume=True)


def test_numeric_loader_rejects_fifo_symlink_components_and_wrong_hash(tmp_path):
    pytest.importorskip("safetensors")
    config = cr.CropConfig(crop_size=64, width=16)
    checkpoint = tmp_path / "crop.safetensors"
    cr.save_model(cr.make_model(config), checkpoint, config, {})
    with pytest.raises(ValueError, match="SHA"):
        cr.load_model(checkpoint, expected_sha256="0" * 64)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        cr.load_model(fifo)
    link = tmp_path / "linked.safetensors"
    link.symlink_to(checkpoint)
    with pytest.raises(OSError):
        cr.load_model(link)
    directory_link = tmp_path / "linked-directory"
    directory_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        cr.load_model(directory_link / checkpoint.name)
    with pytest.raises(ValueError, match="regular"):
        cr.load_model(tmp_path)


@pytest.mark.parametrize("config_change", [
    {"batch_size": 1000000}, {"cpu_threads": True}, {"width": 1000000},
    {"max_per_image": 1000000}, {"padding": float("nan")},
])
def test_numeric_loader_rejects_unbounded_artifact_configuration(tmp_path, config_change):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    config = cr.CropConfig(crop_size=64, width=16)
    checkpoint = tmp_path / "crop.safetensors"
    save_file(cr.make_model(config).state_dict(), checkpoint, metadata={
        "vision_rl": json.dumps({"format": "native-pixel-cascade-v1",
                                  "config": asdict(replace(config, **config_change))}),
    })
    with pytest.raises(ValueError):
        cr.load_model(checkpoint)


@pytest.mark.parametrize("corruption", ["shape", "dtype", "nonfinite"])
def test_numeric_loader_requires_fixed_finite_tensor_schema(tmp_path, corruption):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    config = cr.CropConfig(crop_size=64, width=16)
    state = cr.make_model(config).state_dict()
    if corruption == "shape":
        state["axis"] = state["axis"][:-1].contiguous()
    elif corruption == "dtype":
        state["axis"] = state["axis"].double()
    else:
        state["axis"][0] = float("nan")
    checkpoint = tmp_path / "crop.safetensors"
    save_file(state, checkpoint, metadata={"vision_rl": json.dumps({
        "format": "native-pixel-cascade-v1", "config": asdict(config),
    })})
    with pytest.raises(ValueError, match="architecture|nonfinite"):
        cr.load_model(checkpoint)


def test_frozen_transfer_preserves_training_identity_and_cpu_budget(tmp_path):
    pytest.importorskip("safetensors")
    images, _, proposals = toy_data(tmp_path)
    config = cr.CropConfig(crop_size=64, width=16, cpu_threads=4)
    checkpoint = tmp_path / "crop.safetensors"
    cr.save_model(cr.make_model(config), checkpoint, config, {
        "detector_sha256": "a" * 64, "train_proposals_sha256": "b" * 64,
        "source_code_sha256": "c" * 64, "blend": 0.75, "quality_blend": 0.0,
    })
    original_bytes = checkpoint.read_bytes()
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({"images": images}))
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(json.dumps(proposals))
    composition = {"checkpoint_sha256": cr._sha(checkpoint),
                   "manifest_sha256": cr._sha(manifest),
                   "proposals_sha256": cr._sha(proposal_path),
                   "deployment_detector_sha256": "d" * 64,
                   "blend": 0.75, "quality_blend": 0.0}
    output = tmp_path / "output.json"
    report = cr.refine_manifest(checkpoint, manifest, tmp_path, proposal_path, output,
                                cpu_threads=2, composition=composition)
    assert report["training_detector_sha256"] == "a" * 64
    assert report["training_proposals_sha256"] == "b" * 64
    assert report["training_source_sha256"] == "c" * 64
    assert report["deployment_detector_sha256"] == "d" * 64
    assert report["cpu_threads"] == torch.get_num_threads() == 2
    assert report["proposals"] == len(proposals)
    assert checkpoint.read_bytes() == original_bytes
    replay = json.loads(output.read_text())
    assert [p["score"] for p in replay] == [p["score"] for p in proposals]
    assert replay[-1] == proposals[-1]
    assert all(report[key] >= 0 for key in ("load_seconds", "crop_seconds", "prediction_seconds"))
    with pytest.raises(ValueError, match="Conflicting"):
        cr.refine_manifest(checkpoint, manifest, tmp_path, proposal_path, output,
                            blend=0.5, composition=composition)
    with pytest.raises(ValueError, match="proposals_sha256"):
        cr.refine_manifest(checkpoint, manifest, tmp_path, proposal_path, output,
                            composition={**composition, "proposals_sha256": "0" * 64})
    with pytest.raises(ValueError, match="image tree"):
        cr.refine_manifest(checkpoint, manifest, tmp_path, proposal_path, output,
                            composition={**composition, "image_tree_sha256": "0" * 64})


@pytest.mark.parametrize("changed_input", ["image", "proposals"])
def test_image_only_prediction_rejects_input_mutation_before_publication(
    tmp_path, monkeypatch, changed_input
):
    pytest.importorskip("safetensors")
    images, _, proposals = toy_data(tmp_path)
    config = cr.CropConfig(crop_size=64, width=16)
    checkpoint = tmp_path / "crop.safetensors"
    cr.save_model(cr.make_model(config), checkpoint, config, {})
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({"images": images}))
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(json.dumps(proposals))
    original_predict = cr.predict_bank

    def mutate_after_predict(*args, **kwargs):
        output = original_predict(*args, **kwargs)
        if changed_input == "image":
            Image.new("RGB", (100, 80), "white").save(tmp_path / "a.png")
        else:
            proposal_path.write_text(json.dumps(proposals) + "\n")
        return output

    monkeypatch.setattr(cr, "predict_bank", mutate_after_predict)
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="changed during"):
        cr.refine_manifest(checkpoint, manifest, tmp_path, proposal_path, output, cpu_threads=2)
    assert not output.exists()
