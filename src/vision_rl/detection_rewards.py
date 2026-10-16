"""Small, exact reward kernels for detection policy optimization."""

from __future__ import annotations

import torch


@torch.no_grad()
def interpolated_ap(
    hits: torch.Tensor, keep: torch.Tensor, num_targets: int
) -> torch.Tensor:
    """COCO's 101-point interpolated AP for each sampled detection set.

    ``hits[G,H,Q]`` contains unique-match indicators at ``H`` IoU thresholds,
    already ordered by decreasing detection confidence. ``keep[G,Q]`` marks
    retained queries. Matching and handling ignored/crowd annotations belong to
    the caller; this function only integrates the resulting precision/recall.
    Empty ground truth returns zero as a training reward convention (not COCO's
    undefined-category sentinel).

    A suffix maximum is the precision envelope. Binary search finds its first
    eligible position at each recall threshold, including the leftmost position
    when recall is repeated. This is equivalent to maximizing over all eligible
    queries but avoids materializing a ``[G,H,Q,101]`` tensor.
    """
    if hits.ndim != 3 or keep.shape != (hits.shape[0], hits.shape[2]):
        raise ValueError("expected hits[G,H,Q] and keep[G,Q]")
    if hits.dtype != torch.bool or keep.dtype != torch.bool:
        raise ValueError("matching and retention indicators must be boolean")
    if hits.device != keep.device:
        raise ValueError("matching and retention indicators must share a device")
    if not isinstance(num_targets, int) or isinstance(num_targets, bool) or num_targets < 0:
        raise ValueError("num_targets must be a nonnegative integer")
    groups, thresholds, queries = hits.shape
    if thresholds < 1:
        raise ValueError("at least one IoU threshold is required")
    if num_targets == 0 or queries == 0 or groups == 0:
        return torch.zeros(groups, dtype=torch.float32, device=hits.device)

    cumulative_hits = hits.float().cumsum(-1)
    precision = cumulative_hits / keep[:, None].cumsum(-1).clamp_min(1)
    recall = cumulative_hits / num_targets
    envelope = precision.flip(-1).cummax(-1).values.flip(-1)
    recall_grid = torch.linspace(0, 1, 101, device=hits.device)
    indices = torch.searchsorted(
        recall.contiguous(),
        recall_grid.expand(groups, thresholds, -1).contiguous(),
        right=False,
    )
    # Binary search returns Q for unattained recall. The sentinel gives it zero.
    envelope = torch.cat((envelope, torch.zeros_like(envelope[..., :1])), -1)
    return envelope.gather(-1, indices).mean((-1, -2))
