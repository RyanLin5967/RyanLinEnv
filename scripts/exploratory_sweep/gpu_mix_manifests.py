"""Write the replay/mix training manifests for the OVERNIGHT re-plan (root, 2026-10-19 ~06:45Z).

  python3 -I scripts/exploratory_sweep/gpu_mix_manifests.py [--check]

Inputs, all pinned by sha256 (refuses on any mismatch):
  pool_mixedv3_454.jsonl   rl-campaign-pools ad54cfb pools v3 mixed     052a2953...
  s_train.jsonl            the sweep's clean TRAIN pool (no holdout, train400 or near-duplicate)
  scores-v3.jsonl          ad54cfb pools v3 scores.jsonl (I's greedy answer per row)  c0df5d61...
Outputs (docs/aircraft-exploratory-sweep-v2-gpu/manifests/):
  pool_r1_mixedv3_replay_908.jsonl  R1: mixed-v3 (454) + 454 s_train rows NOT in mixed whose
                                    I-greedy answer is correct (replay of I's own correct answers)
  pool_r3_mixedv3_strain_908.jsonl  R3: mixed-v3 (454) + 454 s_train rows NOT in mixed (any)
Selection by sha256(json[study, tag, id]) rank. Refuses on a duplicate id, a row outside
s_train u mixed, or fewer than 454 eligible s_train rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
M = WT / "docs" / "aircraft-exploratory-sweep-v2-gpu" / "manifests"
S_TRAIN = WT / "docs" / "aircraft-exploratory-sweep-v2" / "manifests" / "s_train.jsonl"
SCORES = WT / "docs" / "aircraft-exploratory-sweep-v2-gpu" / "pools-ref" / "scores-v3.jsonl"
PINS = {"pool_mixedv3_454.jsonl": "052a2953d235", "scores-v3.jsonl": "c0df5d610282"}
STUDY = "aircraft-exploratory-sweep-v2-gpu"


def rows(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rank(tag: str, i: str) -> str:
    return hashlib.sha256(json.dumps([STUDY, tag, i]).encode()).hexdigest()


def build() -> dict:
    for name, pin, path in (("pool_mixedv3_454.jsonl", PINS["pool_mixedv3_454.jsonl"],
                             M / "pool_mixedv3_454.jsonl"),
                            ("scores-v3.jsonl", PINS["scores-v3.jsonl"], SCORES)):
        if not sha(path).startswith(pin):
            raise SystemExit(f"REFUSE: {name} differs from its pin {pin}")
    mixed = rows(M / "pool_mixedv3_454.jsonl")
    s_train = rows(S_TRAIN)
    mixed_ids = {r["id"] for r in mixed}
    greedy_ok = {r["id"] for r in rows(SCORES) if r.get("greedy_correct") is True}
    outside = [r for r in s_train if r["id"] not in mixed_ids]
    out = {}
    for name, tag, pool in (("pool_r1_mixedv3_replay_908.jsonl", "r1",
                             [r for r in outside if r["id"] in greedy_ok]),
                            ("pool_r3_mixedv3_strain_908.jsonl", "r3", outside)):
        if len(pool) < len(mixed):
            raise SystemExit(f"REFUSE: {name}: only {len(pool)} eligible s_train rows")
        extra = sorted(pool, key=lambda r: rank(tag, r["id"]))[: len(mixed)]
        xs = sorted(mixed + extra, key=lambda r: rank(tag + "-order", r["id"]))
        ids = [r["id"] for r in xs]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"REFUSE: {name} repeats an id")
        out[name] = xs
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    for name, xs in build().items():
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in xs)
        if a.check:
            if (M / name).read_text() != text:
                raise SystemExit(f"REFUSE: {name} differs from a rebuild")
        else:
            (M / name).write_text(text)
        print(name, len(xs), hashlib.sha256(text.encode()).hexdigest()[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
