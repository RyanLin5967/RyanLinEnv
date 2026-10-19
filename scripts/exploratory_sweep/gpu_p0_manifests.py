"""Write the P0-I probe manifests (GPU lane, root corrections C1/C14). Light, Mac-side.

  python3 -I scripts/exploratory_sweep/gpu_p0_manifests.py [--check]

p0_poolA200.jsonl / p0_poolB200.jsonl: two disjoint 2-per-class halves of the clean pool
(see build()).
                      2 rows per class, chosen by sha256(json[study, "s", label, id]).
p0_alltrain200.jsonl: all_train minus s_train minus train400 minus every pool id the G0 pHash
                      audit flags as a near-duplicate of a VAL/TEST image; 2 rows per class
                      where the class has them, the shortfall filled by the next-ranked rows
                      of any class (rank by sha256(json[study, "a", label, id])).
Deterministic; --check refuses if the committed files differ from a rebuild. Refuses on a
dev/h_ holdout id, a train400 id or a flagged id in either output, or on fewer than 190 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
SWEEP = WT / "docs" / "aircraft-exploratory-sweep-v2" / "manifests"
OUT = WT / "docs" / "aircraft-exploratory-sweep-v2-gpu" / "manifests"
REPO = Path("/Users/idide/RyanLinEnv")
PHASH = REPO / "runs" / "aircraft-freshpool-g0-v1" / "manifests" / "phash_audit.json"
TRAIN400 = (REPO / "data" / "fgvc-aircraft-2013b" / "candidates" / "fourshot-seed791"
            / "train400.jsonl")
STUDY = "aircraft-exploratory-sweep-v2-gpu"


def rows(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def rank(tag: str, r: dict) -> str:
    return hashlib.sha256(json.dumps([STUDY, tag, r["label"], r["id"]]).encode()).hexdigest()


def pick(pool: list, tag: str, per_class: int, total: int) -> list:
    by = {}
    for r in sorted(pool, key=lambda r: rank(tag, r)):
        by.setdefault(r["label"], []).append(r)
    chosen = [r for label in sorted(by) for r in by[label][:per_class]]
    rest = sorted((r for label in by for r in by[label][per_class:]), key=lambda r: rank(tag, r))
    chosen += rest[: max(0, total - len(chosen))]
    return sorted(chosen, key=lambda r: rank(tag, r))


def build() -> dict:
    near = {x["pool_id"] for x in json.loads(PHASH.read_text())["external_near_duplicates"]}
    t400 = {r["id"] for r in rows(TRAIN400)}
    hold = set()
    for m in SWEEP.glob("*.jsonl"):
        if m.name.startswith(("dev", "h_")):
            hold |= {r["id"] for r in rows(m)}
    s_train = rows(SWEEP / "s_train.jsonl")
    s_ids = {r["id"] for r in s_train}
    extra = [r for r in rows(SWEEP / "all_train.jsonl")
             if r["id"] not in s_ids and r["id"] not in near and r["id"] not in t400]
    # Measured 2026-10-18: all_train minus s_train minus train400 minus flagged ids is only
    # 15 rows (all_train's other 890 are train400 and VAL/TEST near-duplicates), so the clean
    # all_train pool is s_train plus 15. P0 therefore probes that clean pool as two disjoint
    # halves of 2 rows per class each: A, and B (which takes the 15 extras first).
    first = pick(s_train, "s", 2, 200)
    taken = {r["id"] for r in first}
    rest = [r for r in s_train if r["id"] not in taken]
    second = extra + [r for r in pick(rest, "a", 2, 200)
                      if sum(x["label"] == r["label"] for x in extra) < 2][: 200 - len(extra)]
    second = sorted(second, key=lambda r: rank("a", r))[:200]
    out = {"p0_poolA200.jsonl": first, "p0_poolB200.jsonl": second}
    # Extra placebo seeds (backlog: SFT-from-I seeds 7, 11): s_train in the design's scan rule,
    # random.Random(seed).shuffle(sorted(ids)) (design.py), for seeds the design did not cut.
    import random
    lookup = {r["id"]: r for r in s_train}
    for seed in (7, 11):
        ids = sorted(lookup)
        random.Random(seed).shuffle(ids)
        out[f"s_train_scan_seed{seed}.jsonl"] = [lookup[i] for i in ids]
    if {r["id"] for r in first} & {r["id"] for r in second}:
        raise SystemExit("REFUSE: P0 halves overlap")
    for name, xs in out.items():
        ids = {r["id"] for r in xs}
        if len(xs) < 190 or ids & (near | t400 | hold) or len(ids) != len(xs):
            raise SystemExit(f"REFUSE: {name} has {len(xs)} rows or a forbidden id")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    out = build()
    for name, xs in out.items():
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in xs)
        path = OUT / name
        if a.check:
            if path.read_text() != text:
                raise SystemExit(f"REFUSE: {path} differs from a rebuild")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(name, len(xs), "classes", len({r["label"] for r in xs}),
              hashlib.sha256(text.encode()).hexdigest()[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
