"""Shard coverage rules: every image exactly once, or no score. Standard library only."""

import copy
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ci_shards", Path(__file__).resolve().parents[1] / "scripts/ci_shards.py"
)
ci_shards = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_shards)

IDS = [f"{n:07d}" for n in range(10)]
IDENTITY = {
    "plan_id": "p",
    "plan_sha256": "a" * 64,
    "model": "m",
    "stage": "discovery",
    "shards": 3,
    "run_id": "11",
    "run_attempt": "1",
}
# Hand-written slices of 10 IDs into 3 contiguous shards: [0,3), [3,6), [6,10).
SLICES = [IDS[0:3], IDS[3:6], IDS[6:10]]


def record(index, ids=None):
    ids = SLICES[index] if ids is None else ids
    return {
        **IDENTITY,
        "shard": index,
        "status": "completed",
        "ids": ids,
        "ids_sha256": ci_shards.ids_sha256(ids),
        "results": {i: {"status": "completed", "canonical_label": "x"} for i in ids},
    }


def records():
    return [record(k) for k in range(3)]


def test_slices_are_contiguous_and_cover_once():
    assert [ci_shards.shard_ids(IDS, k, 3) for k in range(3)] == SLICES
    assert ci_shards.shard_bounds(10, 3) == [(0, 3), (3, 6), (6, 10)]


def test_complete_records_merge_in_stage_order():
    merged = ci_shards.merge_shard_records(list(reversed(records())), IDS, IDENTITY)
    assert list(merged) == IDS


def test_missing_shard_is_refused():
    found = [r for r in records() if r["shard"] != 1]
    with pytest.raises(ci_shards.ShardGapError, match=r"missing shard\(s\) \[1\] of 3"):
        ci_shards.merge_shard_records(found, IDS, IDENTITY)


def test_no_records_is_refused():
    with pytest.raises(ci_shards.ShardGapError, match=r"missing shard\(s\) \[0, 1, 2\]"):
        ci_shards.merge_shard_records([], IDS, IDENTITY)


def test_duplicate_shard_is_refused():
    with pytest.raises(ci_shards.ShardGapError, match="shard 2 recorded more than once"):
        ci_shards.merge_shard_records(records() + [record(2)], IDS, IDENTITY)


def test_dropped_image_inside_a_shard_is_refused():
    found = records()
    del found[2]["results"][IDS[9]]
    with pytest.raises(ci_shards.ShardGapError, match="results do not cover its slice"):
        ci_shards.merge_shard_records(found, IDS, IDENTITY)


def test_shifted_slice_is_refused():
    found = records()
    found[1] = record(1, IDS[2:6])
    with pytest.raises(ci_shards.ShardGapError, match="differs from its pre-registered slice"):
        ci_shards.merge_shard_records(found, IDS, IDENTITY)


@pytest.mark.parametrize("field", ci_shards.IDENTITY_FIELDS)
def test_foreign_identity_is_refused(field):
    found = records()
    found[0][field] = "other"
    with pytest.raises(ci_shards.ShardGapError, match=f"shard 0 has {field}="):
        ci_shards.merge_shard_records(found, IDS, IDENTITY)


def test_partial_shard_and_unknown_outcome_are_refused():
    found = records()
    found[0]["status"] = "partial_wall_cap"
    found[1]["results"][IDS[4]] = {"status": "unknown_after_interruption"}
    with pytest.raises(ci_shards.ShardGapError) as caught:
        ci_shards.merge_shard_records(found, IDS, IDENTITY)
    assert "shard 0 status" in str(caught.value)
    assert f"image {IDS[4]} has no completed outcome" in str(caught.value)


def test_shard_count_bounds():
    with pytest.raises(ValueError):
        ci_shards.shard_bounds(10, 0)
    with pytest.raises(ValueError):
        ci_shards.shard_bounds(10, 11)
    with pytest.raises(ValueError):
        ci_shards.shard_bounds(3333, 257)
    assert len(ci_shards.shard_bounds(3333, 256)) == 256


def test_full_shard_count_rule():
    # usable = 19800 - 300 - 600 = 18900 s; 18900 / (1.5 * 60) = 210 images per shard;
    # ceil(3333 / 210) = 16 shards.
    count = ci_shards.full_shard_count(
        3333,
        mean_latency=60.0,
        overhead=300.0,
        shard_wall_seconds=19800,
        image_seconds=600,
        safety_factor=1.5,
    )
    assert count == 16
    # 18900 / (1.5 * 4000) = 3 per shard -> 1111 shards > 256: no fit.
    too_slow = copy.deepcopy(
        dict(overhead=300.0, shard_wall_seconds=19800, image_seconds=600, safety_factor=1.5)
    )
    assert ci_shards.full_shard_count(3333, mean_latency=4000.0, **too_slow) is None
    assert ci_shards.full_shard_count(3333, mean_latency=0.0, **too_slow) is None
