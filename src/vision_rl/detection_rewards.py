"""Small, exact reward kernels for detection policy optimization."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=512)
def _recall_counts(num_targets: int, queries: int) -> tuple[int, ...]:
    """Integer TP thresholds reproducing pycocotools' float64 recall grid.

    Construct this small lookup on CPU: MPS has no float64 tensors. Comparing
    float32 recalls to a float32 linspace can move an entire recall bin (1/101
    AP), e.g. seven exact matches against ten targets. Native COCO's float64
    grid also has boundary rounding, so rational ceil(grid * targets) is not
    interchangeable with its actual searchsorted convention.
    """
    recall = np.arange(queries + 1, dtype=np.float64) / num_targets
    return tuple(np.searchsorted(recall, np.linspace(0, 1, 101), side="left").tolist())


@torch.no_grad()
def interpolated_ap(hits: torch.Tensor, keep: torch.Tensor, num_targets: int) -> torch.Tensor:
    """COCO's 101-point interpolated AP for each sampled detection set.

    ``hits[G,H,Q]`` contains unique-match indicators at ``H`` IoU thresholds,
    already ordered by decreasing detection confidence. ``keep[G,Q]`` marks
    retained queries. Matching and handling ignored/crowd annotations belong to
    the caller; this function only integrates the resulting precision/recall.
    Empty ground truth returns zero as a training reward convention (not COCO's
    undefined-category sentinel).

    A suffix maximum is the precision envelope. Binary search finds its first
    eligible position at each recall threshold, including the leftmost position
    when recall is repeated. Recall eligibility matches native COCO float64 on
    CPU and MPS; precision integration returns float32. This avoids materializing
    a ``[G,H,Q,101]`` tensor.
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

    cumulative_hits = hits.cumsum(-1)
    precision = cumulative_hits.float() / keep[:, None].cumsum(-1).clamp_min(1)
    envelope = precision.flip(-1).cummax(-1).values.flip(-1)
    required_hits = torch.tensor(_recall_counts(num_targets, queries), device=hits.device)
    indices = torch.searchsorted(
        cumulative_hits.contiguous(),
        required_hits.expand(groups, thresholds, -1).contiguous(),
        right=False,
    )
    # Binary search returns Q for unattained recall. The sentinel gives it zero.
    envelope = torch.cat((envelope, torch.zeros_like(envelope[..., :1])), -1)
    return envelope.gather(-1, indices).mean((-1, -2))
