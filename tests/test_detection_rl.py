"""Numerical RL checks, strict split isolation, and resumable execution."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from vision_rl import detection_rl as rl  # noqa: E402


def cache(split="train", n=4, d=3):
    boxes = torch.tensor([[[0.2, 0.2, 0.6, 0.6], [0.21, 0.2, 0.61, 0.6]]]).repeat(n, 1, 1)
    generator = torch.Generator().manual_seed(200 if split == "train" else 400)
    return {
        "version": 1,
        "split": split,
        "ids": [f"{split}-{i}" for i in range(n)],
        "boxes": boxes,
        "logits": torch.tensor([[1.0, 0.3]]).repeat(n, 1),
        "embeddings": torch.randn(n, 2, d, generator=generator),
        "targets": [torch.tensor([[0.22, 0.2, 0.62, 0.6]]) for _ in range(n)],
        "image_sizes": torch.tensor([[640, 480]]).repeat(n, 1),
        "coco_image_ids": list(range(n)),
        "metadata": {"model_sha256": "a" * 64, "category_id": 1},
        "pixel_sha256": [f"{split}-pixel-{i}" for i in range(n)],
    }


def test_identity_preserves_out_of_image_detector_boxes_exactly():
    boxes = torch.tensor([[-0.1, 0.2, 1.3, 0.8]])
    assert torch.equal(rl.refine_boxes(boxes, torch.tensor([40])), boxes)
    refined = rl.refine_boxes(boxes[:, None], torch.arange(81)[None])
    assert refined.shape == (1, 81, 4)
    assert torch.all(refined[..., 2:] > refined[..., :2])
    assert torch.equal(rl.action_table()[40], torch.zeros(4))


def test_reward_unique_matching_penalizes_duplicates_and_missing_objects():
    target = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
    boxes = target[None].repeat(3, 2, 1)
    keep = torch.tensor([[True, False], [True, True], [False, False]])
    reward, stats = rl.matching_rewards(boxes, keep, torch.tensor([0.9, 0.8]), target)
    assert stats["AP"].tolist() == pytest.approx([1, 1, 0])
    assert stats["tp_fraction"].tolist() == [[1, 0], [1, 0], [0, 0]]
    assert reward[:, 0].tolist() == pytest.approx([2, 1 + 2 / 3, 0])
    # Every action gets the true team utility, including displacement of later queries.
    assert torch.equal(reward[:, 0], reward[:, 1])


def test_empty_targets_reward_false_positives_without_nan():
    boxes = torch.tensor([[[0.2, 0.2, 0.8, 0.8]]]).repeat(2, 1, 1)
    reward, stats = rl.matching_rewards(
        boxes, torch.tensor([[True], [False]]), torch.tensor([0.9]), torch.empty(0, 4)
    )
    assert reward[:, 0].tolist() == pytest.approx([-1 / 3, 0])
    assert stats["AP"].tolist() == [0, 0]
    assert torch.isfinite(reward).all()


def test_equal_iou_ties_match_later_annotation_like_native_coco():
    targets = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]])
    boxes = torch.tensor([[[0.0, 0.0, 20.0, 10.0], [0.0, 0.0, 10.0, 10.0]]])
    _, statistics = rl.matching_rewards(
        boxes, torch.ones(1, 2, dtype=torch.bool), torch.tensor([0.9, 0.8]), targets
    )
    # Straddler ties at IoU .5; assigning it to the later GT preserves both
    # matches at .5. At the other nine thresholds only the second query hits.
    expected = (1 + 9 * (0.5 * 51 / 101)) / 10
    assert statistics["AP"].item() == pytest.approx(expected, abs=1e-7)
    assert statistics["tp_fraction"].tolist()[0] == pytest.approx([0.1, 1.0])


def test_leave_one_out_excludes_current_sample():
    rewards = torch.tensor([[1.0, 3], [2, 6], [3, 9]])
    assert rl.leave_one_out_advantage(rewards).tolist() == [[-1.5, -4.5], [0, 0], [1.5, 4.5]]


@pytest.mark.parametrize("temperature", [0.5, 1.0])
def test_sampled_joint_gradient_matches_enumerated_two_query_objective(temperature):
    # Enumerate every geometry/retain action for two competing queries. This tests
    # cross-query credit, not just the easier single-object categorical identity.
    torch.set_num_threads(2)
    geometry = torch.tensor([[[0.3, -0.1], [-0.2, 0.4]]], requires_grad=True)
    keep_logits = torch.tensor([[0.5, -0.3]], requires_grad=True)
    boxes = torch.tensor([[[0.25, 0.25, 0.75, 0.75], [0.22, 0.22, 0.72, 0.72]]])
    targets = [torch.tensor([[0.2, 0.2, 0.8, 0.8]])]
    config = rl.RefinementConfig(
        groups=8000,
        entropy_weight=0,
        kl_weight=0,
        iou_shaping=0,
        keep_sampling_temperature=temperature,
    )
    loss, _ = rl.reinforce_loss(
        geometry,
        keep_logits,
        boxes,
        targets,
        torch.Generator().manual_seed(17),
        config,
        geometry.detach(),
        keep_logits.detach(),
    )
    sampled_gradient = torch.autograd.grad(loss, (geometry, keep_logits))
    probabilities = geometry.softmax(-1)[0]
    keep_prob = (keep_logits / temperature).sigmoid()[0]
    expected = torch.tensor(0.0)
    for actions in itertools.product(range(2), repeat=2):
        for keep in itertools.product((0, 1), repeat=2):
            action = torch.tensor([actions])
            kept = torch.tensor([keep], dtype=torch.bool)
            predicted = rl.refine_boxes(boxes[0][None], action, config.geometry_step)
            reward, _ = rl.matching_rewards(predicted, kept, keep_prob.detach(), targets[0])
            probability = torch.tensor(1.0)
            for query in range(2):
                probability = probability * probabilities[query, actions[query]]
                probability = probability * (
                    keep_prob[query] if keep[query] else 1 - keep_prob[query]
                )
            expected = expected + probability * reward[0, 0]
    exact_gradient = torch.autograd.grad(-expected, (geometry, keep_logits))
    for sampled, exact in zip(sampled_gradient, exact_gradient, strict=True):
        assert torch.allclose(sampled, exact, atol=0.025, rtol=0.1), (sampled, exact)


def test_training_rejects_test_split_and_cross_split_pixels_before_output(tmp_path):
    config = rl.RefinementConfig(hidden=8, warmup_epochs=0, rl_epochs=1)
    with pytest.raises(ValueError, match="expected train"):
        rl.train_refinement(cache("test"), cache("validation"), tmp_path / "out", config)
    assert not (tmp_path / "out").exists()
    train, val = cache(), cache("validation")
    val["pixel_sha256"][0] = train["pixel_sha256"][0]
    with pytest.raises(ValueError, match="pixel_sha256 overlap"):
        rl.train_refinement(train, val, tmp_path / "out", config)


def _evaluator(validation):
    def evaluate(predictions):
        values = [
            rl.box_iou(prediction["boxes"], target).max().item()
            for prediction, target in zip(predictions, validation["targets"], strict=True)
        ]
        return {"AP": sum(values) / len(values)}

    return evaluate


def test_interrupted_resume_matches_uninterrupted_weights_and_optimizer(tmp_path, monkeypatch):
    train, val = cache(), cache("validation", n=2)
    config = rl.RefinementConfig(
        hidden=8,
        warmup_epochs=1,
        rl_epochs=2,
        groups=4,
        batch_size=2,
    )
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "resumed"
    rl.train_refinement(train, val, uninterrupted, config, evaluator=_evaluator(val))
    save = rl._atomic_save

    def fail_after_first_checkpoint(payload, path):
        save(payload, path)
        if path.name == "last.pt" and payload["completed_epochs"] == 1:
            raise InterruptedError("simulated process interruption after durable checkpoint")

    monkeypatch.setattr(rl, "_atomic_save", fail_after_first_checkpoint)
    with pytest.raises(InterruptedError):
        rl.train_refinement(train, val, interrupted, config, evaluator=_evaluator(val))
    monkeypatch.setattr(rl, "_atomic_save", save)
    rl.train_refinement(train, val, interrupted, config, resume=True, evaluator=_evaluator(val))
    full = torch.load(uninterrupted / "last.pt", weights_only=True)
    resumed = torch.load(interrupted / "last.pt", weights_only=True)
    assert full["history"] == resumed["history"]
    assert torch.equal(full["generator_state"], resumed["generator_state"])
    for key in full["state_dict"]:
        assert torch.equal(full["state_dict"][key], resumed["state_dict"][key])
    for parameter, state in full["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, resumed["optimizer"]["state"][parameter][key])


def test_prediction_ignores_labels_and_checks_frozen_detector_identity(tmp_path):
    config = rl.RefinementConfig(hidden=8)
    observation = cache("test", n=1)
    mean, scale = torch.zeros(3), torch.ones(3)
    policy = rl.QueryRefinementPolicy(18, config)
    path = tmp_path / "policy.pt"
    rl._atomic_save(
        rl._policy_payload(policy, mean, scale, config, {"model_sha256": "a" * 64}), path
    )
    expected = rl.predict_refinement(observation, path)
    observation["targets"] = "this is deliberately not a target tensor"
    actual = rl.predict_refinement(observation, path)
    assert torch.equal(actual[0]["boxes"], expected[0]["boxes"])
    assert torch.equal(actual[0]["scores"], expected[0]["scores"])
    observation["metadata"]["model_sha256"] = "different detector"
    with pytest.raises(ValueError, match="different frozen detector"):
        rl.predict_refinement(observation, path)


def test_coco_export_preserves_source_ids_and_unclipped_coordinates():
    source = cache(n=1)
    source["coco_image_ids"] = [287]
    prediction = [
        {
            "image_id": "train-0",
            "boxes": torch.tensor([[-0.1, 0.2, 1.1, 0.6]]),
            "scores": torch.tensor([0.7]),
        }
    ]
    exported = rl.coco_predictions(source, prediction)
    assert exported[0]["image_id"] == 287
    assert exported[0]["category_id"] == 1
    assert exported[0]["bbox"] == pytest.approx([-64, 96, 768, 192], abs=0.0001)


def test_resume_changed_config_rejected(tmp_path):
    train, val = cache(), cache("validation", n=2)
    config = rl.RefinementConfig(hidden=8, warmup_epochs=0, rl_epochs=1, groups=2)
    rl.train_refinement(train, val, tmp_path, config, evaluator=_evaluator(val))
    different = rl.RefinementConfig(hidden=8, warmup_epochs=0, rl_epochs=2, groups=2)
    with pytest.raises(ValueError, match="identical caches and configuration"):
        rl.train_refinement(train, val, tmp_path, different, resume=True, evaluator=_evaluator(val))


def test_safe_refiner_roundtrip_and_model_binding(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    config = rl.RefinementConfig()
    observation = cache("test", n=1, d=256)
    mean, scale = torch.zeros(256), torch.ones(256)
    policy = rl.QueryRefinementPolicy(524, config)
    checkpoint, safe = tmp_path / "policy.pt", tmp_path / "refiner.safetensors"
    rl._atomic_save(
        rl._policy_payload(policy, mean, scale, config, {"model_sha256": "a" * 64}), checkpoint
    )
    rl.export_refinement_safetensors(checkpoint, safe)
    ordinary = rl.predict_refinement(observation, checkpoint)
    protected = rl.predict_refinement_safetensors(observation, safe)
    assert torch.equal(ordinary[0]["boxes"], protected[0]["boxes"])
    assert torch.equal(ordinary[0]["scores"], protected[0]["scores"])
    state = load_file(str(safe))
    state["model_sha256"][0] = 0
    save_file(state, str(safe))
    with pytest.raises(ValueError, match="different frozen detector"):
        rl.predict_refinement_safetensors(observation, safe)


def test_safe_refiner_rejects_extra_tensor_and_bad_normalization(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    observation = cache("test", n=1, d=256)
    config = rl.RefinementConfig()
    policy = rl.QueryRefinementPolicy(524, config)
    checkpoint, safe = tmp_path / "policy.pt", tmp_path / "refiner.safetensors"
    rl._atomic_save(
        rl._policy_payload(
            policy, torch.zeros(256), torch.ones(256), config, {"model_sha256": "a" * 64}
        ),
        checkpoint,
    )
    rl.export_refinement_safetensors(checkpoint, safe)
    original = load_file(str(safe))
    save_file({**original, "unexpected": torch.zeros(1)}, str(safe))
    with pytest.raises(ValueError, match="unexpected or missing"):
        rl.predict_refinement_safetensors(observation, safe)
    original["feature_scale"][0] = 0
    save_file(original, str(safe))
    with pytest.raises(ValueError, match="at least .05"):
        rl.predict_refinement_safetensors(observation, safe)


def test_zero_residual_preserves_extreme_detector_confidence_ordering():
    config = rl.RefinementConfig(hidden=8)
    policy = rl.QueryRefinementPolicy(18, config)
    logits = torch.tensor([[9.0, 10.0, -10.0]])
    geometry, keep = policy(torch.randn(1, 3, 18), logits)
    assert torch.equal(keep, logits)
    assert torch.equal(keep.sigmoid(), logits.sigmoid())
    assert geometry.argmax(-1).tolist() == [[40, 40, 40]]


def test_joint_policy_regularizers_scale_with_query_count():
    config = rl.RefinementConfig(
        groups=4,
        entropy_weight=0.003,
        kl_weight=0.02,
        iou_shaping=0,
    )
    geometry = torch.randn(1, 3, 81, requires_grad=True)
    keep = torch.tensor([[1.0, 0.5, -0.2]], requires_grad=True)
    boxes = torch.tensor([[[0.1, 0.1, 0.4, 0.4]]]).repeat(1, 3, 1)
    loss, stats = rl.reinforce_loss(
        geometry,
        keep,
        boxes,
        [torch.empty(0, 4)],
        torch.Generator().manual_seed(8),
        config,
        torch.zeros_like(geometry),
        torch.zeros_like(keep),
    )
    expected = stats["policy_loss"] + 3 * (
        -config.entropy_weight * stats["entropy"] + config.kl_weight * stats["kl"]
    )
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_crash_after_uncommitted_best_update_restores_committed_warmup(tmp_path, monkeypatch):
    train, val = cache(), cache("validation", n=2)
    config = rl.RefinementConfig(
        hidden=8,
        warmup_epochs=1,
        rl_epochs=2,
        groups=2,
        batch_size=2,
        learning_rate=0.01,
        rl_learning_rate=0.01,
    )

    def controlled_policy_objective(geometry, keep, *args, **kwargs):
        # Deterministic optimizer trajectory isolates transaction consistency.
        return -keep.mean(), {}

    def evaluate(predictions):
        return {"AP": torch.stack([row["scores"].mean() for row in predictions]).mean().item()}

    monkeypatch.setattr(rl, "reinforce_loss", controlled_policy_objective)
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "interrupted"
    rl.train_refinement(train, val, uninterrupted, config, evaluator=evaluate)
    save = rl._atomic_save
    crashed = False

    def interrupt_between_best_and_commit(payload, path):
        nonlocal crashed
        save(payload, path)
        last = path.parent / "last.pt"
        if path.name == "best.pt" and last.exists() and not crashed:
            committed = torch.load(last, weights_only=True)
            if committed["completed_epochs"] == 1:
                crashed = True
                raise InterruptedError("best artifact updated, epoch not yet committed")

    monkeypatch.setattr(rl, "_atomic_save", interrupt_between_best_and_commit)
    with pytest.raises(InterruptedError):
        rl.train_refinement(train, val, interrupted, config, evaluator=evaluate)
    monkeypatch.setattr(rl, "_atomic_save", save)
    rl.train_refinement(train, val, interrupted, config, resume=True, evaluator=evaluate)
    full = torch.load(uninterrupted / "last.pt", weights_only=True)
    resumed = torch.load(interrupted / "last.pt", weights_only=True)
    assert full["history"] == resumed["history"]
    for key in full["state_dict"]:
        assert torch.equal(full["state_dict"][key], resumed["state_dict"][key])
        assert torch.equal(
            full["best_payload"]["state_dict"][key], resumed["best_payload"]["state_dict"][key]
        )


def test_native_annotation_pin_rejects_changed_labels_before_scoring(tmp_path):
    import json

    validation = cache("validation", n=1)
    boxes = []
    for index, box in enumerate(validation["targets"][0]):
        x1, y1, x2, y2 = map(float, box)
        boxes.append(
            {
                "id": index,
                "image_id": 0,
                "category_id": 1,
                "bbox": [640 * x1, 480 * y1, 640 * (x2 - x1), 480 * (y2 - y1)],
            }
        )
    document = {"images": [{"id": 0, "width": 640, "height": 480}], "annotations": boxes}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document))
    validation["metadata"]["annotation_path"] = str(path)
    pinned = rl.pin_native_annotations(validation)
    document["annotations"][0]["bbox"][0] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="targets differ"):
        rl.pin_native_annotations(validation)
    with pytest.raises(ValueError, match="changed during refinement"):
        rl._evaluate(pinned, [], None)


def test_mean_action_keeps_identity_and_supports_fine_interpolation():
    config = rl.RefinementConfig(hidden=8, geometry_inference="mean")
    policy = rl.QueryRefinementPolicy(18, config)
    geometry, _ = policy(torch.randn(1, 2, 18), torch.zeros(1, 2))
    boxes = torch.tensor([[[-0.1, 0.2, 0.8, 0.9], [0.2, 0.2, 0.8, 0.8]]])
    assert torch.equal(rl.mean_refine_boxes(boxes, geometry), boxes)
    logits = torch.full((1, 2, 81), -100.0)
    logits[..., 40] = 0
    logits[..., 67] = 0  # dx=+.03, all other components zero.
    refined = rl.mean_refine_boxes(boxes, logits)
    expected = boxes.clone()
    expected[..., [0, 2]] += (boxes[..., 2] - boxes[..., 0])[..., None] * 0.015
    assert torch.allclose(refined, expected, atol=1e-7)


def test_safe_mean_action_roundtrip_and_invalid_mode_rejected(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    observation = cache("test", n=1, d=256)
    config = rl.RefinementConfig(geometry_inference="mean")
    policy = rl.QueryRefinementPolicy(524, config)
    with torch.no_grad():
        policy.network[-1].bias[67] = 5
    checkpoint, safe = tmp_path / "policy.pt", tmp_path / "refiner.safetensors"
    rl._atomic_save(
        rl._policy_payload(
            policy, torch.zeros(256), torch.ones(256), config, {"model_sha256": "a" * 64}
        ),
        checkpoint,
    )
    rl.export_refinement_safetensors(checkpoint, safe)
    ordinary = rl.predict_refinement(observation, checkpoint)
    protected = rl.predict_refinement_safetensors(observation, safe)
    assert torch.equal(ordinary[0]["boxes"], protected[0]["boxes"])
    state = load_file(str(safe))
    state["geometry_inference"][0] = 2
    save_file(state, str(safe))
    with pytest.raises(ValueError, match="safe geometry inference mode"):
        rl.predict_refinement_safetensors(observation, safe)
