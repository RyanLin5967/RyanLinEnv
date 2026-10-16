"""Direct native-detector policy math and training-state contracts."""

import itertools
import math

import pytest

torch = pytest.importorskip("torch")
from vision_rl import detector_policy_training as dpt  # noqa:E402


def test_detached_gaussian_score_function_gradient_matches_analytic_polynomial():
    mean = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64, requires_grad=True)
    scale = torch.full_like(mean, 0.6)
    actions = dpt.sample_latents(mean, scale, 200000, torch.Generator().manual_seed(9))
    assert not actions.requires_grad
    rewards = actions[:, 0, 0] ** 2 + 2 * actions[:, 0, 1] + actions[:, 1, 0] * actions[:, 1, 1]
    loss = dpt.leave_one_out_loss(dpt.factor_log_probability(mean, scale, actions), rewards)
    actual = torch.autograd.grad(loss, mean)[0]
    expected = -torch.tensor([[0.4, 2], [0.1, 0.7]], dtype=torch.float64)
    assert torch.allclose(actual, expected, atol=0.025, rtol=0.03), (actual, expected)
    with pytest.raises(ValueError, match="detached"):
        dpt.factor_log_probability(mean, scale, actions.requires_grad_())


def test_reference_scale_and_analytic_joint_kl_have_no_hidden_query_group_factor():
    reference = torch.tensor(
        [[0.5, 0.5, -2.0, -3.0, 2.0], [0.3, 0.6, -1.0, -2.0, -4.0]], dtype=torch.float64
    )
    mean = reference.clone().requires_grad_()
    config = dpt.PolicyTrainingConfig()
    scale = dpt.policy_scales(mean, (1000, 500), config)
    assert not scale.requires_grad
    changed = mean + scale
    kl = dpt.gaussian_kl(changed, reference, scale)
    assert kl.tolist() == pytest.approx([2.5, 2.5])
    assert kl.sum().item() == pytest.approx(5, abs=1e-12)
    assert (scale[:, :2] <= reference[:, 2:4].exp() * 0.05).all()


def test_counterfactual_factor_importance_gradient_exact_enumeration():
    logits = torch.tensor([[0.2, -0.3], [-0.1, 0.4]], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor(list(itertools.product(range(2), repeat=2)))
    reward = torch.tensor([0.3, 0.8, -0.5, 0.2], dtype=torch.float64)
    selected_logprob = logits.log_softmax(-1)[torch.arange(2)[None], actions]
    probability = selected_logprob.sum(-1).exp()
    exact = torch.autograd.grad(-(probability * reward).sum(), logits, retain_graph=True)[0]
    qprob = torch.tensor([0.85, 0.15], dtype=torch.float64)
    integrated = logits.sum() * 0
    for a in range(4):
        for q in range(2):
            reset = actions[a].clone()
            reset[q] = 0
            index = int(reset[0] * 2 + reset[1])
            loss = dpt.counterfactual_loss(
                selected_logprob[a : a + 1],
                reward[a : a + 1],
                reward[index].reshape(1, 1),
                torch.tensor([q]),
                qprob,
            )
            integrated = integrated + loss * probability[a].detach() * qprob[q]
    estimated = torch.autograd.grad(integrated, logits)[0]
    assert torch.allclose(estimated, exact, atol=1e-12, rtol=1e-12)


def test_with_replacement_duplicate_factors_keep_correct_weights():
    logprob = torch.tensor([[0.1, 0.3], [0.2, 0.4]], requires_grad=True)
    reward = torch.tensor([1.0, 0.5])
    indices = torch.tensor([0, 0, 1])
    cf = torch.tensor([[0.5, 0.5, 0.2], [0.2, 0.2, 0.3]])
    probability = torch.tensor([0.75, 0.25])
    actual = dpt.counterfactual_loss(logprob, reward, cf, indices, probability)
    manual = (
        -sum(
            (reward[g] - cf[g, k]) * logprob[g, indices[k]] / probability[indices[k]]
            for g in range(2)
            for k in range(3)
        )
        / 6
    )
    assert torch.allclose(actual, manual, atol=1e-7, rtol=1e-7)


def test_each_sample_reranks_scores_before_unique_matching():
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    boxes = torch.tensor(
        [[[2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 1.0, 1.0]]], dtype=torch.float64
    ).repeat(2, 1, 1)
    scores = torch.tensor([[0.9, 0.8], [0.8, 0.9]], dtype=torch.float64)
    reward, stats = dpt.image_reward(boxes, scores, target)
    assert stats["AP"].tolist() == [0.5, 1]
    assert reward.tolist() == [0.5, 1]


def test_native_coco_float64_recall_boundary_is_preserved():
    targets = torch.tensor(
        [[i * 2.0, 0.0, i * 2.0 + 1.0, 1.0] for i in range(10)], dtype=torch.float64
    )
    _, stats = dpt.image_reward(targets[None, :7], torch.ones(1, 7), targets)
    # NumPy/COCO linspace(.70) lies just above representable7/10;71/101 is wrong here.
    assert stats["AP"].item() == pytest.approx(70 / 101, abs=1e-12)


def test_counterfactuals_replace_only_selected_query_and_rerank():
    mean = torch.tensor([[0.5, 0.5, 0.0, 0.0, 2.0], [2.5, 2.5, 0.0, 0.0, 1.0]], dtype=torch.float64)
    samples = mean[None].repeat(2, 1, 1)
    samples[0, 0, 4] = -2
    samples[1, 1, :2] = 0.5
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    selected = torch.tensor([0, 1, 0])
    actual = dpt.counterfactual_rewards(samples, mean, target, selected, 0.2)
    for g in range(2):
        for k, q in enumerate(selected):
            replacement = samples[g : g + 1].clone()
            replacement[0, q] = mean[q]
            boxes, scores = dpt.decode_latents(replacement)
            expected, _ = dpt.image_reward(boxes, scores, target)
            assert actual[g, k] == expected[0]


def test_policy_objective_backpropagates_into_live_mean_with_reference_frozen():
    mean = torch.tensor(
        [[0.48, 0.5, -0.7, -0.7, 2.0], [0.8, 0.8, -2.0, -2.0, -3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference = mean.detach().clone()
    target = torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float64)
    config = dpt.PolicyTrainingConfig(groups=32, query_samples=16, pixel_std=2.0)
    loss, metrics = dpt.policy_objective(
        mean, reference, target, (100, 100), config, torch.Generator().manual_seed(81)
    )
    loss.backward()
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum() > 0
    assert metrics["joint_KL"] == 0


def test_native_outputs_latent_mean_preserves_differentiability_and_boxes():
    boxes = torch.tensor([[[0.3, 0.4, 0.2, 0.1]]], requires_grad=True)
    logits = torch.tensor([[[2.0, -1.0]]], requires_grad=True)
    mean = dpt.latent_mean({"pred_boxes": boxes, "pred_logits": logits})
    decoded, scores = dpt.decode_latents(mean)
    assert torch.allclose(decoded, torch.tensor([[[0.2, 0.35, 0.4, 0.45]]], dtype=torch.float64))
    assert scores.item() == pytest.approx(torch.sigmoid(logits.double())[0, 0, 0].item())
    mean.sum().backward()
    assert boxes.grad.abs().sum() > 0 and logits.grad[0, 0, 0] == 1 and logits.grad[0, 0, 1] == 0


@pytest.fixture
def tiny_runtime(tmp_path, monkeypatch):
    """Exercise the real training/checkpoint/control engine with a small native-shaped net."""
    import json
    from types import SimpleNamespace

    from safetensors.torch import load_file, save_file

    class TinyDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bbox_embed = torch.nn.Linear(3, 4)
            self.class_embed = torch.nn.Linear(3, 2)
            with torch.no_grad():
                self.bbox_embed.weight.fill_(0.01)
                self.bbox_embed.bias.copy_(torch.tensor([-0.4, 0.0, -0.4, -0.4]))
                self.class_embed.weight.fill_(0.02)
                self.class_embed.bias.copy_(torch.tensor([3.0, -3.0]))
            self.register_buffer("offset", torch.linspace(0, 0.4, 300).reshape(1, 300, 1))

        def forward(self, images):
            features = images.mean((-1, -2))
            boxes = self.bbox_embed(features).sigmoid()[:, None].expand(-1, 300, -1)
            boxes = boxes + torch.cat(
                (self.offset, self.offset, self.offset * 0, self.offset * 0), -1
            )
            logits = self.class_embed(features)[:, None].expand(-1, 300, -1) - self.offset * 20
            return {"pred_boxes": boxes, "pred_logits": logits}

    source = tmp_path / "data"
    for split in ("train", "valid"):
        (source / split).mkdir(parents=True)
        (source / split / "_annotations.coco.json").write_text(json.dumps({"fixture": split}))

    class TinyData:
        def __init__(self, source, split, resolution):
            assert split in ("train", "valid")
            self.annotation_path = source / split / "_annotations.coco.json"
            self.identity = {
                "fingerprint": split,
                "annotation_sha256": dpt.sha(self.annotation_path),
                "image_sha256": [split + str(i) for i in range(3)],
            }
            self.category_id = 1
            self.images = [{"id": i, "width": 100, "height": 100} for i in range(3)]
            self.targets = [
                torch.tensor([[0.2, 0.3, 0.6, 0.7]], dtype=torch.float64) for _ in range(3)
            ]

        def __len__(self):
            return 3

        def get(self, index, flip=False):
            image = torch.full((3, 8, 8), 0.2 + index * 0.1)
            native = self.targets[index].clone()
            target = {"boxes": torch.tensor([[0.4, 0.5, 0.4, 0.4]]), "labels": torch.tensor([0])}
            if flip:
                target["boxes"][:, 0] = 1 - target["boxes"][:, 0]
                native[:, [0, 2]] = 1 - native[:, [2, 0]]
            return image, target, native

        def verify(self):
            if dpt.sha(self.annotation_path) != self.identity["annotation_sha256"]:
                raise ValueError("native annotation bytes changed")

    checkpoint = tmp_path / "initial.safetensors"
    save_file(TinyDetector().state_dict(), str(checkpoint))

    def load_model(path, config):
        model = TinyDetector()
        model.load_state_dict(load_file(str(path)))
        return SimpleNamespace(model=SimpleNamespace(model=model), model_config=None)

    class Criterion(torch.nn.Module):
        weight_dict = {"loss_bbox": 1.0}

        def forward(self, raw, targets):
            expected = torch.stack([row["boxes"][0] for row in targets])
            return {"loss_bbox": (raw["pred_boxes"][:, 0] - expected).square().mean()}

    @torch.no_grad()
    def evaluate(model, data, config):
        batch = torch.stack([data.get(i)[0] for i in range(len(data))])
        raw = model(batch)
        error = (raw["pred_boxes"][:, 0] - torch.tensor([0.4, 0.5, 0.4, 0.4])).square().mean()
        ap = float(1 - error)
        return {"AP": ap, "AP95": ap**2}, dpt._native_predictions(raw, data.images, 1)

    monkeypatch.setattr(dpt, "NativeData", TinyData)
    monkeypatch.setattr(dpt, "_model", load_model)
    monkeypatch.setattr(dpt, "_criterion", lambda *args: Criterion().eval())
    monkeypatch.setattr(dpt, "evaluate_model", evaluate)
    # Full native drift/reward audit is covered by the separate actual-model dogfood.
    # A six-parameter SFT fixture would otherwise run hundreds of artificial epochs
    # merely to consume the real 300-query reward auditing overhead.
    monkeypatch.setattr(dpt, "monitor_reference_drift", lambda *args: {"passed": True})
    monkeypatch.setattr(
        dpt, "_gate", lambda *args: {"go": True, "fixture": "synthetic-engine-test"}
    )
    config = dpt.PolicyTrainingConfig(
        epochs=3,
        batch_images=2,
        groups=4,
        query_samples=4,
        cpu_threads=1,
        pixel_std=1.0,
        learning_rate=0.0001,
    )
    return source, checkpoint, config


def test_full_engine_cpu_resume_restores_native_parameters_optimizer_scheduler_rng(
    tmp_path, tiny_runtime, monkeypatch
):
    source, checkpoint, config = tiny_runtime
    full, tmp = tmp_path / "full", tmp_path / "interrupted"
    dpt.train(source, checkpoint, full, config)
    save = dpt.save_state

    def interrupt(path, state):
        save(path, state)
        if state["optimizer_steps"] == 2:
            raise InterruptedError("durableepochcommit beforeselectedpublication")

    monkeypatch.setattr(dpt, "save_state", interrupt)
    with pytest.raises(InterruptedError):
        dpt.train(source, checkpoint, tmp, config)
    monkeypatch.setattr(dpt, "save_state", save)
    dpt.train(source, checkpoint, tmp, config, resume=True)
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(tmp / "last.pt", weights_only=True)
    assert a["history"] == b["history"]
    assert a["selections"] == b["selections"]
    assert a["scheduler"] == b["scheduler"]
    assert a["optimizer_steps"] == b["optimizer_steps"] == 6
    for name, value in a["model"].items():
        assert torch.equal(value, b["model"][name])
    for name, value in a["rng"]["generators"].items():
        assert torch.equal(value, b["rng"]["generators"][name])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["rng"]["python"] == b["rng"]["python"] and a["rng"]["numpy"] == b["rng"]["numpy"]
    for index, values in a["optimizer"]["state"].items():
        for key, value in values.items():
            assert torch.equal(value, b["optimizer"]["state"][index][key])


def test_controls_match_init_images_scope_update_count_and_actual_wall_budget(
    tmp_path, tiny_runtime
):
    from dataclasses import replace

    source, checkpoint, config = tiny_runtime
    rl = tmp_path / "rl"
    rl_result = dpt.train(source, checkpoint, rl, config)
    equal = tmp_path / "equal-updates"
    equal_result = dpt.train(source, checkpoint, equal, config, mode="sft_updates", control_run=rl)
    assert equal_result["optimizer_steps"] == rl_result["optimizer_steps"]
    assert equal_result["reference_sha256"] == rl_result["reference_sha256"]
    wall = tmp_path / "equal-walltime"
    wall_result = dpt.train(source, checkpoint, wall, config, mode="sft_walltime", control_run=rl)
    assert wall_result["optimization_seconds"] >= rl_result["optimization_seconds"]
    assert wall_result["matched_control"]["RL_updates"] == rl_result["optimizer_steps"]
    with pytest.raises(ValueError, match="control configuration differs"):
        dpt.train(
            source,
            checkpoint,
            tmp_path / "bad",
            replace(config, learning_rate=0.0002),
            mode="sft_updates",
            control_run=rl,
        )


def test_resume_rejects_changed_native_annotation_identity(tmp_path, tiny_runtime):
    source, checkpoint, config = tiny_runtime
    output = tmp_path / "fit"
    dpt.train(source, checkpoint, output, config)
    (source / "train/_annotations.coco.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identical source/configuration/data"):
        dpt.train(source, checkpoint, output, config, resume=True)


def test_failed_pilot_produces_no_trained_checkpoint(tmp_path, tiny_runtime, monkeypatch):
    source, checkpoint, config = tiny_runtime
    monkeypatch.setattr(dpt, "_gate", lambda *args: {"go": False})
    output = tmp_path / "no-go"
    result = dpt.train(source, checkpoint, output, config)
    assert result["status"] == "no_go_action_space" and result["optimizer_steps"] == 0
    assert not list(output.glob("*.safetensors"))


def test_native_training_single_writer_lock(tmp_path, tiny_runtime):
    import fcntl

    source, checkpoint, config = tiny_runtime
    output = tmp_path / "locked"
    output.mkdir()
    with (output / ".training.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another trainer"):
            dpt.train(source, checkpoint, output, config)


def test_equal_domain_sampling_is_identical_across_training_arms():
    class Data:
        images = [{"domain": "A"}, {"domain": "B"}, {"domain": "B"}, {"domain": "B"}]

        def __len__(self):
            return len(self.images)

    data = Data()
    config = dpt.PolicyTrainingConfig(reward_backend="wheat_ada", domain_balanced_sampling=True)
    probabilities = dpt._sampling_probabilities(data, config)
    assert probabilities.tolist() == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    order1 = dpt._draw_order(data, config, torch.Generator().manual_seed(5), 100)
    order2 = dpt._draw_order(data, config, torch.Generator().manual_seed(5), 100)
    assert order1 == order2 and len(order1) == 100
    assert set(order1) == {0, 1, 2, 3}


def test_wheat_native_reward_uses_official_pixel_geometry_and_primary_metric():
    pytest.importorskip("wilds")
    from vision_rl.wheat_eval import official_image_rewards

    config = dpt.PolicyTrainingConfig(reward_backend="wheat_ada", domain_balanced_sampling=True)
    boxes = torch.tensor(
        [
            [[0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]],
            [[0.1, 0.1, 0.3, 0.3], [0.7, 0.7, 0.8, 0.8]],
        ],
        dtype=torch.float64,
    )
    scores = torch.tensor([[0.8, 0.9], [0.8, 0.5]], dtype=torch.float64)
    targets = torch.tensor([[0.1, 0.1, 0.3, 0.3]], dtype=torch.float64)
    actual, stats = dpt.native_reward(boxes, scores, targets, config, (1024, 1024))
    expected = official_image_rewards(boxes * 1024, scores, targets * 1024)
    assert torch.equal(actual, expected)
    assert actual.tolist() == pytest.approx([0.5, 1])
    assert set(stats) == {"ADA"}


def test_wheat_calibration_has_symmetric_fixed_grid_and_deterministic_tie(tmp_path):
    import json

    pytest.importorskip("wilds")
    annotation = tmp_path / "wheat.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 100, "height": 100, "domain": "Arvalis_1"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}
                ],
                "categories": [{"id": 1, "name": "wheat"}],
            }
        )
    )
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.4}]
    report, calibrated = dpt.calibrated_wheat_evaluation(annotation, predictions)
    assert report["calibration"]["foreground_logit_offset"] == 0.5
    assert report["calibration"]["raw_zero_offset_ADA"] == 0
    assert report["metrics"]["detection_acc_avg_dom"] == 1
    assert len(report["calibration"]["grid"]) == 7
    assert predictions[0]["score"] == 0.4
    assert calibrated[0]["bbox"] == predictions[0]["bbox"]
    assert len(calibrated) == len(predictions)


def test_native_trust_region_backtracks_parameter_delta_and_rolls_back_rejected_state():
    from dataclasses import replace

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.center = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, image):
            x = self.center
            return {
                "pred_boxes": torch.stack((x, x * 0 + 0.5, x * 0 + 0.2, x * 0 + 0.2)).reshape(
                    1, 1, 4
                ),
                "pred_logits": (x * 0).expand(1, 1, 2),
            }

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    batch = torch.zeros(1)
    mean = dpt.latent_mean(model(batch))
    reference = mean.detach().clone()
    model.center.grad = torch.tensor(1.0)
    config = dpt.PolicyTrainingConfig(trust_region_max_kl=0.01, pixel_std=1)
    metrics = dpt.backtracked_adam_step(
        model, optimizer, batch, mean, reference, [(1000, 1000)], config
    )
    assert 0 < metrics["trust_region_step_fraction"] < 1
    assert metrics["trust_region_max_image_KL_per_query"] <= 0.01
    assert model.center.item() != 0.5
    assert optimizer.state[model.center]["step"].item() == 1
    old = model.center.detach().clone()
    state = {k: v.clone() for k, v in optimizer.state[model.center].items()}
    mean = dpt.latent_mean(model(batch))
    model.center.grad = torch.tensor(1.0)
    with pytest.raises(FloatingPointError, match="No native Adam proposal"):
        dpt.backtracked_adam_step(
            model,
            optimizer,
            batch,
            mean,
            reference,
            [(1000, 1000)],
            replace(config, trust_region_backtracks=0),
        )
    assert torch.equal(model.center, old)
    for key, value in state.items():
        assert torch.equal(optimizer.state[model.center][key], value)


def test_lazy_setup_rng_side_effects_do_not_change_committed_training_stream(
    tmp_path, tiny_runtime, monkeypatch
):
    import random

    import numpy as np

    source, checkpoint, config = tiny_runtime
    original = dpt._model
    calls = 0

    def first_process_import(path, cfg):
        nonlocal calls
        calls += 1
        if calls == 1:
            random.seed(1994)
            np.random.seed(2994)
            torch.manual_seed(3994)
        return original(path, cfg)

    monkeypatch.setattr(dpt, "_model", first_process_import)
    dpt.train(source, checkpoint, tmp_path / "cold", config)
    dpt.train(source, checkpoint, tmp_path / "warm", config)
    a = torch.load(tmp_path / "cold/last.pt", weights_only=True)
    b = torch.load(tmp_path / "warm/last.pt", weights_only=True)
    assert a["rng"]["python"] == b["rng"]["python"]
    assert a["rng"]["numpy"] == b["rng"]["numpy"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["history"] == b["history"]


def test_wheat_gate_cannot_silently_fall_back_to_pills_ap_threshold():
    import json
    from dataclasses import replace
    from pathlib import Path

    protocol = json.loads(Path("docs/wheat-direct-rl-protocol-v1.json").read_text())
    config = dpt.PolicyTrainingConfig(reward_backend="wheat_ada", domain_balanced_sampling=True)
    with pytest.raises(ValueError, match="threshold differs"):
        dpt._validate_protocol(config, protocol)
    dpt._validate_protocol(
        replace(config, pilot_min_reward_gain=0.05), protocol, pixel_scales=[1, 2, 4]
    )
    with pytest.raises(ValueError, match="action scales differ"):
        dpt._validate_protocol(
            replace(config, pilot_min_reward_gain=0.05), protocol, pixel_scales=[0.25, 0.5, 1]
        )


@pytest.mark.parametrize("field", ["source_sha256", "software", "trainable_scope"])
def test_matched_control_rejects_changed_implementation_provenance_before_update(
    tmp_path, tiny_runtime, field
):
    import json
    from dataclasses import replace

    source, checkpoint, config = tiny_runtime
    config = replace(config, epochs=1, max_steps=1)
    rl = tmp_path / "rl"
    dpt.train(source, checkpoint, rl, config)
    path = rl / "plan.json"
    plan = json.loads(path.read_text())
    plan[field] = {"corrupted": True}
    path.write_text(json.dumps(plan))
    output = tmp_path / "control"
    with pytest.raises(ValueError, match="differs from the RL implementation"):
        dpt.train(source, checkpoint, output, config, mode="sft_updates", control_run=rl)
    assert not (output / "last.pt").exists()


def test_native_train_rejects_identical_pixels_with_different_file_bytes(tmp_path):
    import json

    from PIL import Image

    source = tmp_path / "data"
    for split, compression in [("train", 0), ("valid", 9)]:
        directory = source / split
        directory.mkdir(parents=True)
        Image.new("RGB", (16, 16), (21, 71, 91)).save(
            directory / "image.png", compress_level=compression
        )
        (directory / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "image.png", "width": 16, "height": 16}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [1, 1, 8, 8],
                            "area": 64,
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [{"id": 1, "name": "object"}],
                }
            )
        )
    assert dpt.sha(source / "train/image.png") != dpt.sha(source / "valid/image.png")
    with pytest.raises(ValueError, match="decoded pixels overlap"):
        dpt.train(source, tmp_path / "no-model.safetensors", tmp_path / "fit")


def test_sentinel_covers_each_domain_without_consuming_training_generator():
    class Data:
        images = [{"id": i, "domain": "rare" if i == 0 else "common"} for i in range(20)]

        def __len__(self):
            return len(self.images)

    config = dpt.PolicyTrainingConfig(
        reward_backend="wheat_ada", domain_balanced_sampling=True, drift_sentinel_images=4
    )
    generator = torch.Generator().manual_seed(100)
    before = generator.get_state().clone()
    indices = dpt._sentinel_indices(Data(), config)
    assert len(indices) == 4 and len(set(indices)) == 4 and 0 in indices
    assert indices == dpt._sentinel_indices(Data(), config)
    assert torch.equal(before, generator.get_state())


def test_wheat_comparison_requires_both_controls_trace_and_strongest_sft(tmp_path):
    import json
    from dataclasses import asdict

    pytest.importorskip("wilds")
    from vision_rl.wheat_eval import calibrated_wheat_evaluation

    annotation = tmp_path / "annotations.json"
    domains = ["Arvalis_1", "Arvalis_2", "Arvalis_3"]
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "width": 100, "height": 100, "domain": domain}
                    for i, domain in enumerate(domains)
                ],
                "annotations": [
                    {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20]}
                    for i in range(3)
                ],
                "categories": [{"id": 1, "name": "wheat"}],
            }
        )
    )
    predictions = [
        {"image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99} for i in range(3)
    ]
    best, _ = calibrated_wheat_evaluation(annotation, predictions)
    low, _ = calibrated_wheat_evaluation(annotation, [])
    protocol = {
        "benchmark_id": "gwhd2021-hf-domain-ada-v1",
        "data": {"validation_images": 3},
        "RL_acceptance": {"minimum_absolute_ADA_gain_over_initial_and_each_matched_SFT": 0.05},
    }
    config = asdict(
        dpt.PolicyTrainingConfig(
            reward_backend="wheat_ada", domain_balanced_sampling=True, pilot_min_reward_gain=0.05
        )
    )
    common = {
        "configuration": config,
        "reference_sha256": "initial",
        "train_identity": {"id": "train"},
        "validation_identity": {"id": "validation"},
        "trainable_scope": {"parameters": 1},
        "source_sha256": {"code": "frozen"},
        "software": {"torch": "same"},
        "drift_sentinel_image_ids": [1],
    }
    paths = {mode: tmp_path / mode for mode in ("rl", "sft_updates", "sft_walltime")}
    for path in paths.values():
        path.mkdir()
    dpt.write(paths["rl"] / "protocol.json", protocol)
    common["protocol_sha256"] = dpt.sha(paths["rl"] / "protocol.json")
    for mode, path in paths.items():
        plan = {**common, "mode": mode, "integration_only": False}
        summary = {
            "mode": mode,
            "status": "completed",
            "optimizer_steps": 2,
            "optimization_seconds": 1.0,
            "draw_chain_history": ["first", "second"],
            "optimization_timing_scope": "committed active",
            "selections": {"best-trained": {"sha256": None}},
        }
        if mode != "rl":
            match = {
                "run_plan_sha256": dpt.sha(paths["rl"] / "plan.json"),
                "run_summary_sha256": dpt.sha(paths["rl"] / "summary.json"),
                "RL_updates": 2,
                "RL_optimization_seconds": 1.0,
            }
            plan["matched_control"] = summary["matched_control"] = match
        dpt.write(path / "initial-evaluation.json", {"wheat": low})
        dpt.write(path / "trained-evaluation.json", {"wheat": best if mode == "rl" else low})
        dpt.write(path / "trained-predictions.coco.json", predictions if mode == "rl" else [])
        (path / "best-trained.safetensors").write_bytes(b"test-only-artifact-hash-fixture")
        summary["selections"]["best-trained"]["sha256"] = dpt.sha(path / "best-trained.safetensors")
        for field, name in [
            ("initial_evaluation_sha256", "initial-evaluation.json"),
            ("trained_evaluation_sha256", "trained-evaluation.json"),
            ("trained_predictions_sha256", "trained-predictions.coco.json"),
        ]:
            summary[field] = dpt.sha(path / name)
        dpt.write(path / "plan.json", plan)
        dpt.write(path / "summary.json", summary)
    strongest = tmp_path / "strongest.json"
    dpt.write(strongest, low)
    args = (*paths.values(), strongest, tmp_path / "comparison.json")
    result = dpt.compare_wheat_controls(*args)
    assert result["primary_requirement_satisfied"]
    assert set(result["comparisons"]) == {
        "initial",
        "sft_updates",
        "sft_walltime",
        "strongest_available_sft",
    }
    dpt.write(strongest, best)
    assert not dpt.compare_wheat_controls(*args)["primary_requirement_satisfied"]
    path = paths["sft_walltime"] / "summary.json"
    summary = json.loads(path.read_text())
    summary["draw_chain_history"][0] = "different-image"
    dpt.write(path, summary)
    with pytest.raises(ValueError, match="image/augmentation sequence differs"):
        dpt.compare_wheat_controls(*args)


def test_actual_pixel_scale_diagnostics_expose_collapsed_small_box_grid():
    from dataclasses import replace

    reference = torch.tensor(
        [
            [0.5, 0.5, math.log(0.004), math.log(0.004), 3.0],
            [0.8, 0.8, math.log(0.2), math.log(0.2), -3.0],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([[0.4985, 0.498, 0.5025, 0.502]], dtype=torch.float64)
    config = dpt.PolicyTrainingConfig()
    actual = []
    for pixels in (1.0, 2.0, 4.0):
        values, counts = dpt.scale_observations(
            reference, (1000, 1000), targets, replace(config, pixel_std=pixels)
        )
        report = dpt.summarize_scales(values)
        actual.append(report["foreground_center_SD_pixels"]["quantiles"]["0.5"])
        assert report["foreground_center_relative_cap_indicator"]["fraction"] == 1
        assert report["foreground_log_size_upper_cap_indicator"]["fraction"] == 1
        assert counts["matched_targets_at_IoU50"] == 1
        assert report["matched_center_error_in_SD"]["quantiles"]["1"] == pytest.approx(2.5)
    assert actual == pytest.approx([0.2, 0.2, 0.2])


def test_common_noise_training_audit_is_reproducible_and_separates_confidence_geometry():
    pytest.importorskip("wilds")
    reference = torch.tensor([[0.3, 0.3, math.log(0.2), math.log(0.2), -0.1]], dtype=torch.float64)
    targets = torch.tensor([[0.2, 0.2, 0.4, 0.4]], dtype=torch.float64)
    config = dpt.PolicyTrainingConfig(
        reward_backend="wheat_ada", domain_balanced_sampling=True, audit_groups=32
    )
    generator = torch.Generator().manual_seed(71)
    before = generator.get_state().clone()
    baseline = dpt.paired_training_reward_audit(
        reference, reference, targets, (1024, 1024), config, 18
    )
    assert baseline["paired_expected_reward_gain"] == 0
    assert baseline["paired_noise_standard_error"] == 0
    mean = reference.clone()
    mean[:, 4] = 2
    report = dpt.paired_training_reward_audit(mean, reference, targets, (1024, 1024), config, 18)
    assert report == dpt.paired_training_reward_audit(
        mean, reference, targets, (1024, 1024), config, 18
    )
    assert report["paired_expected_reward_gain"] > 0
    assert report["deterministic_reward"] == {
        "reference": 0.0,
        "current": 1.0,
        "geometry_only": 0.0,
        "confidence_only": 1.0,
    }
    assert torch.equal(before, generator.get_state())


def test_pilot_scale_selects_best_passing_then_smallest_scale_on_tie():
    rows = {
        "4.0": {"go": False, "mean_best_of_samples_gain": 0.2},
        "2.0": {"go": True, "mean_best_of_samples_gain": 0.07},
        "1.0": {"go": True, "mean_best_of_samples_gain": 0.07},
    }
    assert dpt.select_pilot_scale(rows) == "1.0"
    for row in rows.values():
        row["go"] = False
    assert dpt.select_pilot_scale(rows) == "4.0"
