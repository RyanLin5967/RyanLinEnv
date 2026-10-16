"""Independent native-score and policy-gradient audits on synthetic CPU data."""

import itertools
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from vision_rl.detection_eval import evaluate_coco  # noqa: E402
from vision_rl.detector_policy_training import (  # noqa: E402
    counterfactual_loss,
    counterfactual_rewards,
    decode_latents,
    factor_log_probability,
    gaussian_kl,
    image_reward,
    sample_policy,
)


def native_score(tmp_path, targets, boxes, scores):
    annotations = [
        {
            "id": i + 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, x2 - x, y2 - y],
            "area": (x2 - x) * (y2 - y),
            "iscrowd": 0,
        }
        for i, (x, y, x2, y2) in enumerate(targets)
    ]
    document = {
        "images": [{"id": 1, "file_name": "synthetic.png", "width": 1000, "height": 1000}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": annotations,
    }
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(document))
    predictions = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, x2 - x, y2 - y],
            "score": float(score),
        }
        for (x, y, x2, y2), score in zip(boxes, scores, strict=True)
    ]
    return evaluate_coco(path, predictions)["AP"]


@pytest.mark.parametrize(("count", "found"), [(10, 7), (20, 13), (25, 21)])
def test_native_float64_recall_boundaries(tmp_path, count, found):
    targets = np.array([[20.0 * i, 0.0, 20.0 * i + 10, 10.0] for i in range(count)])
    boxes = targets[:found].copy()
    scores = np.linspace(1, 0.1, found)
    _, actual = image_reward(
        torch.tensor(boxes)[None], torch.tensor(scores)[None], torch.tensor(targets)
    )
    assert actual["AP"].item() == pytest.approx(
        native_score(tmp_path, targets, boxes, scores), abs=1e-14
    )


def test_native_matching_ties_and_independent_group_rankings(tmp_path):
    # At IoU .5 the first proposal ties both GTs. Native COCO picks the later
    # annotation, leaving the left GT for the second proposal.
    targets = np.array([[0.0, 0.0, 10, 10], [10.0, 0.0, 20, 10]])
    boxes = np.array([[0.0, 0.0, 20, 10], [0.0, 0.0, 10, 10], [0.0, 0.0, 10, 10]])
    group_scores = np.array([[0.9, 0.8, 0.1], [0.1, 0.8, 0.9], [0.9, 0.9, 0.9]])
    _, actual = image_reward(
        torch.tensor(boxes)[None].expand(3, -1, -1),
        torch.tensor(group_scores),
        torch.tensor(targets),
    )
    for index, scores in enumerate(group_scores):
        assert actual["AP"][index].item() == pytest.approx(
            native_score(tmp_path, targets, boxes, scores), abs=1e-14
        )


def test_native_random_geometry_and_score_parity(tmp_path):
    rng = np.random.default_rng(491)
    xy = rng.uniform(20, 300, (7, 2))
    targets = np.concatenate((xy, xy + rng.uniform(20, 60, (7, 2))), axis=-1)
    boxes = np.concatenate(
        [targets + rng.normal(0, scale, targets.shape) for scale in (0, 1, 5, 10)]
    )
    group_scores = rng.uniform(0, 1, (5, len(boxes)))
    _, actual = image_reward(
        torch.tensor(boxes)[None].expand(5, -1, -1),
        torch.tensor(group_scores),
        torch.tensor(targets),
    )
    for index, scores in enumerate(group_scores):
        assert actual["AP"][index].item() == pytest.approx(
            native_score(tmp_path, targets, boxes, scores), abs=1e-14
        )


def test_all_500_queries_remain_in_native_precision_denominator(tmp_path):
    targets = np.array([[0.0, 0.0, 10.0, 10.0]])
    boxes = np.tile([30.0, 30.0, 40.0, 40.0], (500, 1))
    boxes[-1] = targets[0]
    scores = np.linspace(1.0, 0.01, 500)
    _, actual = image_reward(
        torch.tensor(boxes)[None], torch.tensor(scores)[None], torch.tensor(targets)
    )
    expected = native_score(tmp_path, targets, boxes, scores)
    assert expected == pytest.approx(1 / 500, abs=1e-14)
    assert actual["AP"].item() == pytest.approx(expected, abs=1e-14)
    with pytest.raises(ValueError, match="500"):
        image_reward(torch.zeros(1, 501, 4), torch.zeros(1, 501), torch.tensor(targets))


def test_empty_ground_truth_uses_explicit_background_convention():
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]])
    scores = torch.tensor([[0.25, 0.75]])
    reward, statistics = image_reward(boxes, scores, torch.empty(0, 4))
    assert reward.item() == -0.5
    assert statistics["AP"].item() == statistics["AP95"].item() == 0
    assert statistics["empty_ground_truth"]


def test_no_positive_detections_scores_zero_against_native(tmp_path):
    targets = np.array([[0.0, 0.0, 10.0, 10.0]])
    boxes = np.array([[20.0, 20.0, 30.0, 30.0]])
    scores = np.array([0.99])
    reward, statistics = image_reward(
        torch.tensor(boxes)[None], torch.tensor(scores)[None], torch.tensor(targets)
    )
    assert reward.item() == statistics["AP"].item() == 0
    assert native_score(tmp_path, targets, boxes, scores) == 0


def smooth_reward(actions):
    x, y = actions[..., 0, 0], actions[..., 1, 0]
    return (x + y).sin() + 0.25 * x * y + 0.1 * x.square()


def test_actual_factor_loss_matches_analytic_gradient_by_gaussian_quadrature():
    points, weights = np.polynomial.hermite.hermgauss(32)
    grid = np.stack(np.meshgrid(points, points, indexing="ij"), -1).reshape(-1, 2) * np.sqrt(2)
    probabilities = torch.tensor(np.multiply.outer(weights, weights).reshape(-1) / np.pi)
    mean = torch.zeros(2, 5, dtype=torch.float64)
    mean[:, 0] = torch.tensor([0.4, -0.2])
    mean.requires_grad_()
    reference = mean.detach().clone()
    reference[:, 0] = torch.tensor([-0.3, 0.6])
    sigma = torch.ones_like(mean)
    sigma[:, 0] = torch.tensor([0.35, 0.6])
    samples = mean.detach()[None].repeat(len(grid), 1, 1)
    samples[..., 0] += torch.tensor(grid) * sigma[:, 0]
    log_probability = factor_log_probability(mean, sigma, samples)
    rewards = smooth_reward(samples)
    baselines = []
    for query in range(2):
        modified = samples.clone()
        modified[:, query] = reference[query]
        baselines.append(smooth_reward(modified))
    baselines = torch.stack(baselines, -1)
    selection = torch.tensor([0.85, 0.15], dtype=torch.float64)
    # Integrate over both action quadrature and independent sampled query index.
    expected_loss = sum(
        selection[q]
        * counterfactual_loss(
            log_probability,
            rewards * probabilities * len(grid),
            baselines[:, q : q + 1] * probabilities[:, None] * len(grid),
            torch.tensor([q]),
            selection,
        )
        for q in range(2)
    )
    gradient = -torch.autograd.grad(expected_loss, mean)[0][:, 0]
    x, y = mean.detach()[:, 0]
    common = torch.exp(-0.5 * sigma[:, 0].square().sum()) * torch.cos(x + y)
    expected = common + torch.stack((0.25 * y + 0.2 * x, 0.25 * x))
    torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)


def test_duplicate_query_samples_keep_their_importance_contributions():
    log_probability = torch.tensor([[0.2, 0.7], [-0.4, 0.3]], requires_grad=True)
    rewards = torch.tensor([0.8, 0.1])
    indices = torch.tensor([0, 0, 1])
    baselines = torch.tensor([[0.1, 0.1, 0.2], [0.4, 0.4, 0.9]])
    probabilities = torch.tensor([0.9, 0.1])
    loss = counterfactual_loss(log_probability, rewards, baselines, indices, probabilities)
    actual = torch.autograd.grad(loss, log_probability)[0]
    expected = torch.zeros_like(log_probability)
    for group in range(2):
        for position, query in enumerate(indices):
            expected[group, query] -= (
                (rewards[group] - baselines[group, position]) / probabilities[query] / 6
            )
    torch.testing.assert_close(actual, expected)


def test_counterfactual_changes_only_selected_factor_and_reuses_duplicates():
    generator = torch.Generator().manual_seed(727)
    reference = torch.tensor(
        [[0.2, 0.2, -2.0, -2.0, 0.9], [0.7, 0.6, -2.0, -1.5, 1.2]], dtype=torch.float64
    )
    samples = reference[None] + 0.1 * torch.randn(3, 2, 5, generator=generator, dtype=torch.float64)
    targets = decode_latents(reference[None])[0][0]
    indices = torch.tensor([1, 0, 1])
    actual = counterfactual_rewards(samples, reference, targets, indices, 0.2)
    for position, query in enumerate(indices):
        modified = samples.clone()
        modified[:, query] = reference[query]
        expected, _ = image_reward(*decode_latents(modified), targets, ap95_weight=0.2)
        torch.testing.assert_close(actual[:, position], expected, atol=0, rtol=0)


def test_analytic_kl_matches_torch_distribution_and_gradient():
    mean = torch.randn(
        3, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64, requires_grad=True
    )
    reference = torch.ones_like(mean)
    scale = torch.linspace(0.1, 0.5, 5, dtype=torch.float64).expand_as(mean)
    actual = gaussian_kl(mean, reference, scale)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean, scale), torch.distributions.Normal(reference, scale)
    ).sum(-1)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), mean)[0]
    torch.testing.assert_close(gradient, (mean.detach() - reference) / scale.square())


def test_sparse_mixture_factor_gradient_matches_independent_analytic_expectation():
    points, weights = np.polynomial.hermite.hermgauss(32)
    grid = np.stack(np.meshgrid(points, points, indexing="ij"), -1).reshape(-1, 2)
    grid = torch.tensor(grid * np.sqrt(2))
    quadrature = torch.tensor(np.multiply.outer(weights, weights).reshape(-1) / np.pi)
    mean = torch.zeros(2, 5, dtype=torch.float64)
    mean[:, 0] = torch.tensor([0.4, -0.2], dtype=torch.float64)
    mean.requires_grad_()
    reference = mean.detach().clone()
    reference[:, 0] = torch.tensor([-0.3, 0.6], dtype=torch.float64)
    scale = torch.ones_like(mean)
    scale[:, 0] = torch.tensor([0.35, 0.6], dtype=torch.float64)
    selection = torch.tensor([0.85, 0.15], dtype=torch.float64)
    rho = 0.25
    expected_loss = mean.sum() * 0
    for gates in itertools.product((False, True), repeat=2):
        active = torch.tensor(gates)[None].expand(len(grid), -1)
        samples = mean.detach()[None].repeat(len(grid), 1, 1)
        samples[..., 0] += grid * scale[:, 0]
        samples = torch.where(active[..., None], samples, reference[None])
        gate_probability = rho ** sum(gates) * (1 - rho) ** (2 - sum(gates))
        log_probability = factor_log_probability(mean, scale, samples, active, rho)
        reward = smooth_reward(samples)
        for query in range(2):
            replaced = samples.clone()
            replaced[:, query] = reference[query]
            baseline = smooth_reward(replaced)
            expected_loss = expected_loss + gate_probability * selection[query] * (
                counterfactual_loss(
                    log_probability,
                    reward * quadrature * len(grid),
                    baseline[:, None] * quadrature[:, None] * len(grid),
                    torch.tensor([query]),
                    selection,
                )
            )
    actual = -torch.autograd.grad(expected_loss, mean)[0][:, 0]
    # Independent analytic expectation via the Gaussian characteristic function.
    mu = mean.detach().numpy()[:, 0]
    ref = reference.numpy()[:, 0]
    sigma = scale.numpy()[:, 0]
    gaussian_characteristic = np.exp(1j * mu - 0.5 * sigma**2)
    characteristic = (1 - rho) * np.exp(1j * ref) + rho * gaussian_characteristic
    expected_coordinate = (1 - rho) * ref + rho * mu
    expected = torch.tensor([
        rho * (gaussian_characteristic[0] * characteristic[1]).real
        + 0.25 * rho * expected_coordinate[1] + 0.2 * rho * mu[0],
        rho * (gaussian_characteristic[1] * characteristic[0]).real
        + 0.25 * rho * expected_coordinate[0],
    ])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_sparse_sampler_has_frozen_anchor_and_inactive_factors_have_zero_gradient():
    mean = torch.ones(2, 5, dtype=torch.float64, requires_grad=True)
    reference = torch.zeros_like(mean)
    scale = torch.full_like(mean, 0.2)
    samples, active = sample_policy(
        mean, reference, scale, 64, torch.Generator().manual_seed(91), 0.25
    )
    assert active.any() and (~active).any()
    assert not samples.requires_grad
    assert torch.equal(samples[~active], reference[None].expand_as(samples)[~active])
    inactive_log_probability = factor_log_probability(mean, scale, samples, active, 0.25)[~active]
    gradient = torch.autograd.grad(inactive_log_probability.sum(), mean)[0]
    assert torch.count_nonzero(gradient) == 0
