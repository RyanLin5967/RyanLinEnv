"""Real score-function checks, immutable observations, and resumable optimization."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from vision_rl import crop_rl as crl  # noqa: E402


def toy_cache(split="train", n=3, dim=4):
    generator = torch.Generator().manual_seed(13 if split == "train" else 42)
    return {
        "version": 1,
        "split": split,
        "images": [
            {
                "id": i,
                "width": 100,
                "height": 80,
                "file_name": f"{split}-{i}.png",
                "pixel_sha256": f"{split}-{i}",
            }
            for i in range(n)
        ],
        "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0], [70.0, 60.0, 85.0, 70.0]]).repeat(n, 1),
        "scores": torch.tensor([0.9, 0.01]).repeat(n),
        "image_ids": torch.arange(n).repeat_interleave(2),
        "category_ids": torch.ones(2 * n, dtype=torch.int64),
        "selected_indices": torch.arange(n) * 2,
        "side_features": torch.randn(n, 4, dim, generator=generator),
        "side_logits": torch.randn(n, 4, 17, generator=generator),
        "targets": [torch.tensor([[21.2, 20.0, 61.2, 60.0]]) for _ in range(n)],
        "metadata": {
            "detector_sha256": "a" * 64,
            "supervised_sha256": "b" * 64,
            "proposal_pipeline_sha256": "c" * 64,
        },
    }


def test_small_native_pixel_actions_and_exact_identity():
    boxes = torch.tensor([[20.0, 10.0, 60.0, 50.0], [-10.0, 5.0, 400.0, 800.0]])
    config = crl.PixelRLConfig()
    scales = crl.offset_scales(boxes, config)
    assert torch.allclose(scales, torch.tensor([[0.8] * 4, [4.0] * 4]))
    identity = torch.full((2, 4), 2)
    assert torch.equal(crl.action_boxes(boxes, identity, config), boxes)
    candidates = crl.action_boxes(boxes[:, None], crl.JOINT_ACTIONS[None], config)
    assert candidates.shape == (2, 625, 4)
    assert torch.equal(candidates[:, 312], boxes)
    assert (candidates[..., 2:] > candidates[..., :2]).all()


def test_precomputed_action_iou_matches_direct_geometry_for_all_625_actions():
    cache = toy_cache(n=1)
    config = crl.PixelRLConfig()
    table = crl.prepare_rewards(cache, config)[0]
    direct = crl.box_iou(
        crl.action_boxes(cache["boxes"][:1, None], crl.JOINT_ACTIONS[None], config),
        cache["targets"][0],
    )
    assert torch.equal(table.candidate_ious, direct)
    actions = crl.JOINT_ACTIONS[:, None]
    _, from_table = table.rewards(actions, config.recall_bonus)
    all_boxes = cache["boxes"][None].repeat(625, 1, 1)
    all_boxes[:, 0] = crl.action_boxes(cache["boxes"][:1], crl.JOINT_ACTIONS, config)
    ious = crl.box_iou(all_boxes, cache["targets"][0])
    _, direct_reward = crl.match_iou_reward(ious, cache["scores"])
    assert torch.equal(from_table["AP"], direct_reward["AP"])


def test_sparse_matching_preserves_all_false_positive_ranks():
    ious = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]])
    scores = torch.tensor([0.99, 0.9, 0.8, 0.7])
    _, stats = crl.match_iou_reward(ious, scores)
    # First prediction is an FP, second matches the only object; duplicates lose.
    assert stats["AP"].item() == pytest.approx(0.5)
    assert stats["AP95"].item() == pytest.approx(0.5)
    assert stats["recall"].item() == 1


def test_exact_target_tie_matches_coco_later_target_convention():
    # If query0 picks later target1, query1 still matches target0 => recall1.
    ious = torch.tensor([[[0.9, 0.9], [0.9, 0.1]]])
    _, stats = crl.match_iou_reward(ious, torch.tensor([0.9, 0.8]))
    assert stats["recall"].item() == pytest.approx(0.9)


def test_factorized_joint_score_gradient_matches_enumeration():
    torch.set_num_threads(1)
    # Two boxes, two choices for each of four sides => 256 joint configurations.
    logits = torch.tensor([[[0.2, -0.3]] * 4, [[-0.1, 0.4]] * 4], requires_grad=True)
    combinations = torch.tensor(list(itertools.product(range(2), repeat=8))).reshape(-1, 2, 4)
    # Coupled reward: choosing the same first side on both boxes causes a duplicate.
    reward = (combinations[:, 0].sum(-1) == 2).float()
    reward += 0.7 * (combinations[:, 1].sum(-1) == 1).float()
    reward -= 0.9 * (combinations[:, 0, 0] == combinations[:, 1, 0]).float()
    logprob = logits.log_softmax(-1)
    chosen_logprob = (
        logprob[None]
        .expand(256, -1, -1, -1)
        .gather(-1, combinations[..., None])
        .squeeze(-1)
        .sum((-1, -2))
    )
    expected = -(chosen_logprob.exp() * reward).sum()
    exact = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actions = crl.sample_actions(logits, 100000, torch.Generator().manual_seed(13))
    sampled_reward = (actions[:, 0].sum(-1) == 2).float()
    sampled_reward += 0.7 * (actions[:, 1].sum(-1) == 1).float()
    sampled_reward -= 0.9 * (actions[:, 0, 0] == actions[:, 1, 0]).float()
    config = crl.PixelRLConfig(kl_weight=0, entropy_weight=0)
    loss, _ = crl.policy_gradient(logits, actions, sampled_reward, logprob.detach(), config)
    sampled = torch.autograd.grad(loss, logits)[0]
    assert torch.allclose(sampled, exact, atol=0.006, rtol=0.1), (sampled, exact)


def test_train_only_pilot_detects_real_geometry_headroom():
    cache = toy_cache()
    config = crl.PixelRLConfig(pilot_groups=128, max_offset_fraction=0.05)
    report = crl.exploration_pilot(cache, crl.prepare_rewards(cache, config), config)
    assert report["split"] == "train"
    assert report["not_a_model_score"]
    assert report["mean_best_of_samples_AP_gain"] > 0
    assert report["go"]
    cache["split"] = "validation"
    with pytest.raises(ValueError, match="training split"):
        crl.prepare_rewards(cache, config)


def test_prediction_ignores_targets_and_preserves_frozen_proposals_and_scores(tmp_path):
    cache = toy_cache("test")
    config = crl.PixelRLConfig(hidden=8)
    mean, scale = torch.zeros(4), torch.ones(4)
    features = crl.observation_features(cache, mean, scale)
    policy = crl.PixelSidePolicy(features.shape[-1], config)
    checkpoint = tmp_path / "pixel.safetensors"
    crl.export_policy(crl._payload(policy, mean, scale, config, cache["metadata"]), checkpoint)
    first = crl.predict_cache(cache, checkpoint)
    cache["targets"] = "not even a tensor; inference must not read this"
    assert crl.predict_cache(cache, checkpoint) == first
    assert len(first) == len(cache["boxes"])
    for i, row in enumerate(first):
        x1, y1, x2, y2 = map(float, cache["boxes"][i])
        assert row["bbox"] == [x1, y1, x2 - x1, y2 - y1]
        assert row["score"] == float(cache["scores"][i])
    cache["metadata"]["supervised_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="supervised_sha256 differs"):
        crl.predict_cache(cache, checkpoint)


def _evaluator(cache):
    def evaluate(predictions):
        total = 0
        for image, targets in zip(cache["images"], cache["targets"], strict=True):
            records = [row for row in predictions if row["image_id"] == image["id"]]
            boxes = torch.tensor(
                [
                    [
                        r["bbox"][0],
                        r["bbox"][1],
                        r["bbox"][0] + r["bbox"][2],
                        r["bbox"][1] + r["bbox"][3],
                    ]
                    for r in records
                ]
            )
            total += float(crl.box_iou(boxes, targets).amax())
        return {"AP": total / len(cache["images"])}

    return evaluate


def test_exact_resume_after_commit_before_artifact_publication(tmp_path, monkeypatch):
    train, validation = toy_cache(), toy_cache("validation")
    config = crl.PixelRLConfig(
        hidden=8, epochs=3, batch_images=2, groups=4, minimum_pilot_gain=0, max_offset_fraction=0.05
    )
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    crl.train(train, validation, full, config, evaluator=_evaluator(validation))
    save = crl._save

    def fail_after_commit(path, payload):
        save(path, payload)
        if path.name == "last.pt" and payload["epoch"] == 1:
            raise InterruptedError("epoch committed; artifacts not yet materialized")

    monkeypatch.setattr(crl, "_save", fail_after_commit)
    with pytest.raises(InterruptedError):
        crl.train(train, validation, interrupted, config, evaluator=_evaluator(validation))
    monkeypatch.setattr(crl, "_save", save)
    crl.train(train, validation, interrupted, config, evaluator=_evaluator(validation), resume=True)
    state_a = torch.load(full / "last.pt", weights_only=True)
    state_b = torch.load(interrupted / "last.pt", weights_only=True)
    assert state_a["history"] == state_b["history"]
    assert torch.equal(state_a["rng"], state_b["rng"])
    for key in state_a["state"]:
        assert torch.equal(state_a["state"][key], state_b["state"][key])
    for parameter, values in state_a["optimizer"]["state"].items():
        for key, value in values.items():
            assert torch.equal(value, state_b["optimizer"]["state"][parameter][key])


def test_resume_changed_native_pixel_features_rejected(tmp_path):
    train, validation = toy_cache(), toy_cache("validation")
    config = crl.PixelRLConfig(hidden=8, epochs=1, minimum_pilot_gain=0)
    crl.train(train, validation, tmp_path, config, evaluator=_evaluator(validation))
    train["side_features"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="identical data, source, and configuration"):
        crl.train(
            train, validation, tmp_path, config, evaluator=_evaluator(validation), resume=True
        )


def test_split_and_pixel_overlap_rejected_before_training(tmp_path):
    with pytest.raises(ValueError, match="expected train"):
        crl.train(toy_cache("test"), toy_cache("validation"), tmp_path / "bad")
    train, val = toy_cache(), toy_cache("validation")
    val["images"][0]["pixel_sha256"] = train["images"][0]["pixel_sha256"]
    with pytest.raises(ValueError, match="decoded pixels overlap"):
        crl.train(train, val, tmp_path / "overlap")
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "overlap").exists()


def native_fixture(tmp_path):
    import json

    import numpy as np
    from PIL import Image

    from vision_rl import crop_strips as strips

    pixels = np.zeros((80, 100, 3), dtype=np.uint8)
    pixels[20:60, 25:75] = [220, 80, 40]
    Image.fromarray(pixels).save(tmp_path / "a.png")
    images = [{"id": 7, "file_name": "a.png", "width": 100, "height": 80}]
    predictions = [
        {
            "image_id": 7,
            "category_id": 1,
            "bbox": [24.3, 19.7, 50.4, 40.2],
            "score": 0.900000000001,
        },
        {
            "image_id": 7,
            "category_id": 1,
            "bbox": [-3.123456789, 2.2, 10.3, 10.4],
            "score": 0.000100000001,
        },
    ]
    manifest, proposals = tmp_path / "images.json", tmp_path / "proposals.json"
    manifest.write_text(json.dumps({"images": images}))
    proposals.write_text(json.dumps(predictions))
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": images,
                "categories": [{"id": 1, "name": "pill"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 7,
                        "category_id": 1,
                        "bbox": [25, 20, 50, 40],
                        "area": 2000,
                        "iscrowd": 0,
                    }
                ],
            }
        )
    )
    config = strips.StripConfig(
        normal_samples=128, tangent_samples=32, width=16, cpu_threads=1, image_workers=1
    )
    torch.manual_seed(71)
    model = strips.make_model(config)
    checkpoint = tmp_path / "crop.safetensors"
    strips.save_model(model, checkpoint, config, {"blend": 0.75, "quality_blend": 0.25})
    return checkpoint, manifest, proposals, annotations


def test_native_pixel_export_exactly_matches_supervised_stage_and_image_only_replay(tmp_path):
    import json

    from vision_rl import crop_strips as strips

    supervised, manifest, proposals, annotations = native_fixture(tmp_path)
    cache = crl.export_pixel_cache(
        supervised,
        manifest,
        tmp_path,
        proposals,
        tmp_path / "train.pt",
        split="train",
        detector_sha256="a" * 64,
        proposal_pipeline_sha256="c" * 64,
        annotation_path=annotations,
    )
    output = tmp_path / "supervised.json"
    strips.refine_manifest(supervised, manifest, tmp_path, proposals, output)
    baseline = json.loads(output.read_text())
    assert cache["base_predictions"] == baseline
    assert cache["side_features"].shape == (1, 4, 32)
    assert cache["side_logits"].shape == (1, 4, 128)
    assert cache["selected_indices"].tolist() == [0]
    assert cache["scores"].dtype == torch.float64
    assert cache["scores"].tolist() == [row["score"] for row in baseline]
    assert baseline[1] == json.loads(proposals.read_text())[1]
    mean, scale = torch.zeros(32), torch.ones(32)
    config = crl.PixelRLConfig(hidden=8)
    policy = crl.PixelSidePolicy(111, config)
    policy_path = tmp_path / "pixel.safetensors"
    crl.export_policy(crl._payload(policy, mean, scale, config, cache["metadata"]), policy_path)
    assert crl.predict_cache(cache, policy_path) == baseline
    # Validation reports AP95 from the exact native matching cache.
    report = crl._evaluate(cache, baseline, None)
    assert 0 <= report["AP95"] <= 1
    annotations.unlink()  # The public predictor must require no labels whatsoever.
    report = crl.refine_manifest(
        policy_path,
        supervised,
        manifest,
        tmp_path,
        proposals,
        tmp_path / "rl.json",
        detector_sha256="a" * 64,
        proposal_pipeline_sha256="c" * 64,
    )
    assert not report["labels_available_to_predictor"]
    assert json.loads((tmp_path / "rl.json").read_text()) == baseline
    with pytest.raises(ValueError, match="supervised_composition_sha256 differs"):
        crl.refine_manifest(
            policy_path,
            supervised,
            manifest,
            tmp_path,
            proposals,
            tmp_path / "wrong.json",
            detector_sha256="a" * 64,
            proposal_pipeline_sha256="c" * 64,
            composition={"blend": 0.5},
        )


def test_native_labels_do_not_change_observations_and_changed_annotations_fail(tmp_path):
    import json

    supervised, manifest, proposals, annotations = native_fixture(tmp_path)
    kwargs = {
        "split": "train",
        "detector_sha256": "a" * 64,
        "proposal_pipeline_sha256": "c" * 64,
        "annotation_path": annotations,
    }
    first = crl.export_pixel_cache(supervised, manifest, tmp_path, proposals, **kwargs)
    document = json.loads(annotations.read_text())
    document["annotations"][0]["bbox"] = [30, 20, 45, 40]
    document["annotations"][0]["area"] = 1800
    annotations.write_text(json.dumps(document))
    second = crl.export_pixel_cache(supervised, manifest, tmp_path, proposals, **kwargs)
    for key in ("boxes", "scores", "side_features", "side_logits", "selected_indices"):
        assert torch.equal(first[key], second[key])
    assert not torch.equal(first["targets"][0], second["targets"][0])
    with pytest.raises(ValueError, match="annotations differ"):
        crl._annotation_identity(first)
    with pytest.raises(ValueError, match="only train/validation"):
        crl.export_pixel_cache(
            supervised,
            manifest,
            tmp_path,
            proposals,
            detector_sha256="a" * 64,
            proposal_pipeline_sha256="c" * 64,
            annotation_path=annotations,
        )


def test_float64_scores_retain_native_ranking_and_identity(tmp_path):
    cache = toy_cache(n=1)
    cache["boxes"] = cache["boxes"].double()
    cache["scores"] = torch.tensor([0.900000000001, 0.9], dtype=torch.float64)
    cache["base_predictions"] = [
        {
            "image_id": 0,
            "category_id": 1,
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "score": float(s),
        }
        for b, s in zip(cache["boxes"], cache["scores"], strict=True)
    ]
    config = crl.PixelRLConfig(hidden=8)
    policy = crl.PixelSidePolicy(55, config)
    result = crl.predictions_from_policy(cache, policy, torch.zeros(4), torch.ones(4), config)
    assert result == cache["base_predictions"]
    ious = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)
    _, metrics = crl.match_iou_reward(ious, cache["scores"])
    assert metrics["AP"].item() == 1


def test_safe_policy_rejects_changed_prior_and_parent_symlink(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    config = crl.PixelRLConfig(hidden=8)
    policy = crl.PixelSidePolicy(55, config)
    path = tmp_path / "pixel.safetensors"
    crl.export_policy(
        crl._payload(policy, torch.zeros(4), torch.ones(4), config, toy_cache()["metadata"]), path
    )
    with safe_open(str(path), framework="pt") as source:
        metadata = source.metadata()
    tensors = load_file(str(path))
    tensors["policy.reference_logprob"][0] += 0.01
    save_file(tensors, str(tmp_path / "tampered.safetensors"), metadata=metadata)
    with pytest.raises(ValueError, match="reference prior"):
        crl.load_policy(tmp_path / "tampered.safetensors")
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        crl.load_policy(tmp_path / "alias" / "pixel.safetensors")
    with pytest.raises(ValueError, match="SHA differs"):
        crl.load_policy(path, expected_sha256="a" * 64)


def test_training_output_single_writer_lock(tmp_path):
    import fcntl

    with (tmp_path / ".train.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another pixel RL trainer"):
            crl.train(toy_cache(), toy_cache("validation"), tmp_path)


@pytest.mark.parametrize("seed", [1, 7, 19])
def test_image_reward_matches_native_coco_ap_and_ap95(tmp_path, seed):
    import json

    from vision_rl.detection_eval import _cached_ap, _prepared_cache, evaluate_coco

    generator = torch.Generator().manual_seed(seed)
    target = torch.tensor(
        [[10.0, 10.0, 40.0, 40.0], [30.0, 30.0, 60.0, 60.0], [10.0, 50.0, 35.0, 75.0]],
        dtype=torch.float64,
    )
    boxes = target.repeat(4, 1) + torch.randn(12, 4, generator=generator).double() * 2
    scores = torch.rand(12, generator=generator).double()
    scores[0:2] = 0.9  # Native score tie order remains stable.
    predictions = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "score": float(s),
        }
        for b, s in zip(boxes, scores, strict=True)
    ]
    document = {
        "images": [{"id": 1, "file_name": "unused.png", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "pill"}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "area": float((b[2] - b[0]) * (b[3] - b[1])),
                "iscrowd": 0,
            }
            for i, b in enumerate(target)
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document))
    native = evaluate_coco(path, predictions)
    _, result = crl.match_iou_reward(crl.box_iou(boxes, target)[None], scores)
    assert result["AP"].item() == pytest.approx(native["AP"], abs=1e-7)
    prepared = _prepared_cache(native["match_cache"])
    prepared["iou_thresholds"] = prepared["iou_thresholds"][-1:]
    for image in prepared["items"].values():
        for category in image.values():
            category["matches"] = category["matches"][-1:]
            category["ignored_detections"] = category["ignored_detections"][-1:]
    assert result["AP95"].item() == pytest.approx(_cached_ap(prepared, [1]), abs=1e-7)


def test_leave_one_out_gradient_exactly_matches_all_paired_groups():
    # Exhaustively integrate both independent rollouts, including equal-action groups.
    # This checks the estimator algebra without Monte Carlo tolerance.
    logits = torch.tensor([[[0.2, -0.3]], [[-0.1, 0.4]]], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor(list(itertools.product(range(2), repeat=2))).reshape(4, 2, 1)
    rewards = torch.tensor([0.3, 0.8, -0.5, 0.2], dtype=torch.float64)
    logprob = logits.log_softmax(-1)
    joint_logprob = (
        logprob[None].expand(4, -1, -1, -1).gather(-1, actions[..., None]).squeeze(-1).sum((-1, -2))
    )
    probability = joint_logprob.exp()
    exact = torch.autograd.grad(-(probability * rewards).sum(), logits, retain_graph=True)[0]
    config = crl.PixelRLConfig(kl_weight=0, entropy_weight=0)
    integrated = logits.sum() * 0
    for first, second in itertools.product(range(4), repeat=2):
        loss, _ = crl.policy_gradient(
            logits, actions[[first, second]], rewards[[first, second]], logprob.detach(), config
        )
        integrated = integrated + loss * (probability[first] * probability[second]).detach()
    estimated = torch.autograd.grad(integrated, logits)[0]
    assert torch.allclose(estimated, exact, atol=1e-12, rtol=1e-12)
