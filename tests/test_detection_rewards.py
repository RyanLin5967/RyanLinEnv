"""Exact reward equivalence, including ties and incomplete recall."""

import pytest

torch = pytest.importorskip("torch")

from vision_rl.detection_rewards import interpolated_ap  # noqa: E402


def brute_force_ap(hits, keep, num_targets):
    if num_targets == 0 or hits.shape[-1] == 0:
        return torch.zeros(hits.shape[0], device=hits.device)
    cumulative_hits = hits.float().cumsum(-1)
    precision = cumulative_hits / keep[:, None].cumsum(-1).clamp_min(1)
    recall = cumulative_hits / num_targets
    eligible = recall[..., None] >= torch.linspace(0, 1, 101, device=hits.device)
    return precision[..., None].masked_fill(~eligible, 0).amax(-2).mean((-1, -2))


@pytest.mark.parametrize("queries", [0, 1, 2, 11, 128, 500])
def test_matches_brute_force_for_random_retained_detections(queries):
    generator = torch.Generator().manual_seed(queries + 17)
    keep = torch.rand((8, queries), generator=generator) > 0.4
    # Independent IoU thresholds exercise arbitrary precision envelopes. Only
    # retained queries can match, and unique matches cannot exceed target count.
    raw_hits = (torch.rand((8, 10, queries), generator=generator) > 0.65) & keep[:, None]
    num_targets = max(queries // 3, 1)
    hits = raw_hits & (raw_hits.cumsum(-1) <= num_targets)
    expected = brute_force_ap(hits, keep, num_targets)
    torch.testing.assert_close(interpolated_ap(hits, keep, num_targets), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("matches", "retained", "targets", "expected"),
    [
        ([1, 1], [1, 1], 2, 1.0),
        ([0, 1], [1, 1], 1, 0.5),
        ([0, 1], [0, 1], 1, 1.0),
        ([1, 0], [1, 1], 1, 1.0),
        ([0, 0], [0, 0], 1, 0.0),
        ([0, 0], [1, 1], 0, 0.0),
        # Recall 0.5 contributes at 51 of the 101 recall thresholds.
        ([1, 0, 0], [1, 0, 1], 2, 51 / 101),
        # A late second match raises the interpolated precision at earlier recall.
        ([0, 1, 1], [1, 1, 1], 2, 2 / 3),
        ([0, 1, 0, 1], [1, 1, 1, 1], 2, 0.5),
    ],
)
def test_adversarial_precision_recall_curves(matches, retained, targets, expected):
    hits = torch.tensor([[matches]], dtype=torch.bool)
    keep = torch.tensor([retained], dtype=torch.bool)
    actual = interpolated_ap(hits, keep, targets)
    assert actual.item() == pytest.approx(expected, abs=1e-7)
    torch.testing.assert_close(actual, brute_force_ap(hits, keep, targets), rtol=0, atol=0)


def test_noncontiguous_views_and_empty_group():
    hits = torch.tensor([[[1, 0, 0, 1, 0, 1]]], dtype=torch.bool).expand(4, 10, -1)[..., ::2]
    keep = torch.ones(4, 6, dtype=torch.bool)[:, ::2]
    torch.testing.assert_close(
        interpolated_ap(hits, keep, 2), brute_force_ap(hits, keep, 2), rtol=0, atol=0
    )
    assert interpolated_ap(hits[:0], keep[:0], 2).shape == (0,)


def test_bad_contracts_fail_explicitly():
    hits = torch.zeros(2, 10, 3, dtype=torch.bool)
    keep = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="hits"):
        interpolated_ap(hits, keep[:1], 2)
    with pytest.raises(ValueError, match="boolean"):
        interpolated_ap(hits.float(), keep, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        interpolated_ap(hits, keep, -1)
    with pytest.raises(ValueError, match="IoU threshold"):
        interpolated_ap(hits[:, :0], keep, 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
def test_mps_matches_cpu():
    hits = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.bool).expand(2, 10, -1)
    keep = torch.ones(2, 4, dtype=torch.bool)
    expected = interpolated_ap(hits, keep, 2)
    torch.testing.assert_close(interpolated_ap(hits.to("mps"), keep.to("mps"), 2).cpu(), expected)
