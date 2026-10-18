"""Shard bookkeeping for matrix-sharded CI evaluations; standard library only.

A stage's ordered image IDs are cut into contiguous, deterministic shards. Each shard job
writes one record. ``merge_shard_records`` accepts the records only if together they cover
every image exactly once, under one plan, one model, one stage, one workflow run and one
run attempt. Any gap, duplicate, foreign image, incomplete shard or identity mismatch is a
refusal, never a partial score.

Blind spots, stated on purpose: a record is a JSON file produced by a CI job, not a signed
witness; this module checks consistency between records, not that the job really ran the
model. Workflow artifacts and the run ID are the external evidence for that.
"""

from __future__ import annotations

import hashlib
import json
import math

MAX_SHARDS = 256  # GitHub Actions matrix limit per job.
IDENTITY_FIELDS = ("plan_id", "plan_sha256", "model", "stage", "shards", "run_id", "run_attempt")


class ShardGapError(ValueError):
    """The shard records do not cover the stage exactly once; nothing may be scored."""


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def ids_sha256(ids):
    return hashlib.sha256(compact(list(ids)).encode()).hexdigest()


def shard_bounds(total, shards):
    if type(total) is not int or total < 1:
        raise ValueError("A stage needs at least one image")
    if type(shards) is not int or not 1 <= shards <= min(total, MAX_SHARDS):
        raise ValueError(f"Shard count must be an integer in 1..{min(total, MAX_SHARDS)}")
    return [(k * total // shards, (k + 1) * total // shards) for k in range(shards)]


def shard_ids(ids, shard, shards):
    """Contiguous slice of the author-ordered IDs; every shard is nonempty."""
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Stage IDs must be unique")
    bounds = shard_bounds(len(ids), shards)
    if type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("Shard index out of range")
    start, stop = bounds[shard]
    return ids[start:stop]


def merge_shard_records(records, ids, expected):
    """Return {id: result} in stage order, or raise ShardGapError.

    ``records`` are the parsed shard JSON files that were found, in any order. ``expected``
    fixes every identity field in IDENTITY_FIELDS; each record must match all of them.
    """
    ids = list(ids)
    shards = expected["shards"]
    bounds = shard_bounds(len(ids), shards)
    problems, by_shard = [], {}
    for record in records:
        index = record.get("shard")
        if type(index) is not int or not 0 <= index < shards:
            problems.append(f"record with out-of-range shard index {index!r}")
            continue
        if index in by_shard:
            problems.append(f"shard {index} recorded more than once")
            continue
        by_shard[index] = record
        for field in IDENTITY_FIELDS:
            if record.get(field) != expected[field]:
                problems.append(
                    f"shard {index} has {field}={record.get(field)!r}, expected {expected[field]!r}"
                )
        if record.get("status") != "completed":
            problems.append(f"shard {index} status is {record.get('status')!r}, not completed")
        start, stop = bounds[index]
        want = ids[start:stop]
        if record.get("ids") != want or record.get("ids_sha256") != ids_sha256(want):
            problems.append(f"shard {index} image list differs from its pre-registered slice")
        results = record.get("results")
        if not isinstance(results, dict) or set(results) != set(want):
            problems.append(f"shard {index} results do not cover its slice exactly")
            continue
        for identifier in want:
            result = results[identifier]
            if not isinstance(result, dict) or result.get("status") != "completed":
                problems.append(f"shard {index} image {identifier} has no completed outcome")
    missing = sorted(set(range(shards)) - set(by_shard))
    if missing:
        problems.append(f"missing shard(s) {missing} of {shards}")
    if problems:
        raise ShardGapError("Refusing to score: " + "; ".join(problems))
    merged = {}
    for index in range(shards):
        for identifier, result in by_shard[index]["results"].items():
            if identifier in merged:
                raise ShardGapError(f"Refusing to score: image {identifier} scored twice")
            merged[identifier] = result
    if list(sorted(merged)) != sorted(ids) or len(merged) != len(ids):
        raise ShardGapError("Refusing to score: merged images differ from the stage")
    return {identifier: merged[identifier] for identifier in ids}


def full_shard_count(
    total, *, mean_latency, overhead, shard_wall_seconds, image_seconds, safety_factor
):
    """Pre-registered sizing rule; None when no shard count can fit the wall cap."""
    usable = shard_wall_seconds - overhead - image_seconds
    if not (math.isfinite(mean_latency) and mean_latency > 0) or usable <= 0:
        return None
    per_shard = math.floor(usable / (safety_factor * mean_latency))
    if per_shard < 1:
        return None
    count = math.ceil(total / per_shard)
    return count if count <= min(total, MAX_SHARDS) else None
