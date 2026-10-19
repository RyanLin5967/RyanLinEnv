"""Reduce the I scores into hard/mid/mixed RL pools plus an unreachable list; prove disjointness.

  python3 -I build_pools.py --results DIR --payload-data DIR --audit DIR --phash FILE \
      --train400 FILE --exclusions FILE --out DIR
  python3 -I build_pools.py ... --self-test        # planted-overlap fixture; must refuse each plant

Standard library only. Inputs:
  --results       score_worker.py unit results (results/score-NNN.json)
  --payload-data  the sweep payload's data/ (manifests/*.jsonl, images under manifests/images/train,
                  splits.json): all_train and every TRAIN-derived holdout
  --audit         data/fgvc-aircraft-2013b/audit: train-/val-/test-images.json. From the VAL and
                  TEST files only image_id and encoded_sha256 are read, only to prove disjointness.
  --phash         runs/aircraft-freshpool-g0-v1/manifests/phash_audit.json
                  (VAL/TEST near-duplicates)
  --train400      data/fgvc-aircraft-2013b/candidates/fourshot-seed791/train400.jsonl
  --exclusions    the exclusions file score_worker.py pinned; refused unless it equals the sets
                  re-derived here from --train400 and --phash

Eligible rows = all_train - train400 - the 477 all_train images flagged as pHash near-duplicates
(Hamming <= 6) of a native VAL or sealed TEST image (CORRECTIONS.md C9/M8). 1,452 rows.

Pools (rules fixed by CORRECTIONS.md 2026-10-18, before any corrected score was read):
  p_gold      = P(gold label + EOS) under I at temperature 1.0, summed over every label-trie
                tokenisation (score_worker.py; checked against the trainer per unit)
  hard        = I greedy answer != gold (wrong or invalid)  AND  p_gold >= 0.02
  mid         = 0.1 <= p_gold <= 0.9
  mixed       = hard UNION mid
  unreachable = p_gold < 0.02 (listed separately, never pooled; with per-class counts, and the
                classes whose every eligible image is unreachable)
Pool manifests are trainer format ({id, image, label, split}, images/train/<id>.jpg relative to the
manifest directory, the payload's data/manifests layout) and CLASS-STRATIFIED: round r takes every
class's r-th member, classes in sorted label order, members within a class ordered by
sha256(json([STUDY, pool, SEED, label, id])); every prefix is as class-balanced as the pool allows.

Refuses (exit 2) when: a unit is missing, or two results for one id disagree, or a unit carries a
different adapter/payload/trie/profile, or failed its trainer cross-check; the scored ids are not
exactly the eligible rows; any pool or unreachable row's id or image content sha256 is in dev800,
dev600, h_select, h_gate_final, native VAL, sealed TEST, train400 or the near-duplicate set; a row's
image bytes differ from the native TRAIN audit's encoded_sha256 for its id.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

STUDY = "aircraft-rl-campaign-I-pools-v2"
SEED = 20261018
I_WEIGHTS_SHA256 = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
ALL_TRAIN_SHA256 = "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e"
TRIE_SHA256 = "656dc02fbb5b121f4cd9a9d7ea10364532639b35b647f0dc9010ad5d80a37eb0"
TRAINER_SHA256 = "b6e18a8b7c502ed71f3178ff3891ff0276261aa2cbdc16f9ce0ca869d116a5c0"
EXCLUSIONS_SHA256 = "00a433eab22002764d2d91c02888623b2d5a4b37ee90ba3e1bb0eb19b482380b"
TRAIN400_SHA256 = "b4e5b821b851189d5dda0963f99f98e6b790a51c5b99b07f6af00843bf2a41f0"
ELIGIBLE_ROWS = 1452
SHARD = 32
P_REACH, P_LO, P_HI = 0.02, 0.1, 0.9
HOLDOUTS = ("dev800", "dev600", "h_select", "h_gate_final")


class Refuse(Exception):
    pass


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha(Path(path).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def jsonl(rows) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def id_set_sha(ids) -> str:
    return sha("\n".join(sorted(ids)).encode())


def exclusion_sets(a, all_train_ids: set) -> tuple[set, set, dict]:
    if sha_file(a.train400) != TRAIN400_SHA256:
        raise Refuse("train400.jsonl differs from the pinned sha256")
    t4 = {r["id"] for r in read_jsonl(a.train400)}
    phash = json.loads(a.phash.read_text())
    near = {
        x["pool_id"]
        for x in phash["external_near_duplicates"]
        if x["reference_split"] in ("test", "validation")
    } & all_train_ids
    if len(t4) != 400 or len(near) != 477 or t4 & near or not t4 <= all_train_ids:
        raise Refuse(f"exclusions re-derived wrong: train400 {len(t4)}, near {len(near)}")
    if sha_file(a.exclusions) != EXCLUSIONS_SHA256:
        raise Refuse("exclusions file differs from the pinned sha256")
    ex = json.loads(a.exclusions.read_text())
    if set(ex["train400"]) != t4 or set(ex["val_test_phash_near_duplicates"]) != near:
        raise Refuse(
            "the worker's exclusions file disagrees with train400.jsonl / phash_audit.json"
        )
    return (
        t4,
        near,
        {
            "train400_sha256": TRAIN400_SHA256,
            "phash_audit_sha256": sha_file(a.phash),
            "exclusions_sha256": EXCLUSIONS_SHA256,
        },
    )


def load_scores(results: Path, eligible: list[dict]) -> tuple[dict, dict]:
    expected = (len(eligible) + SHARD - 1) // SHARD
    seen, units, timing, checks = {}, set(), collections.Counter(), []
    for f in sorted(results.glob("score-*.json")):
        doc = json.loads(f.read_text())
        for key, want in (
            ("adapter_sha256", I_WEIGHTS_SHA256),
            ("all_train_sha256", ALL_TRAIN_SHA256),
            ("trie_sha256", TRIE_SHA256),
            ("trainer_sha256", TRAINER_SHA256),
        ):
            if doc.get(key) != want:
                raise Refuse(f"{f.name}: {key} {doc.get(key)} is not the pinned value")
        k = int(doc["unit"].split("-")[1])
        if doc["ids"] != [r["id"] for r in eligible[SHARD * k : SHARD * k + SHARD]] or doc[
            "ids"
        ] != [r["id"] for r in doc["rows"]]:
            raise Refuse(f"{f.name}: does not hold eligible rows {SHARD * k}..")
        v = doc["validation"]
        if v["max_abs_diff"] > 1e-4:
            raise Refuse(f"{f.name}: trainer cross-check failed")
        checks.append(v)
        units.add(k)
        timing["unit_seconds"] += doc["total_seconds"]
        timing["greedy_seconds"] += doc["greedy_seconds"]
        timing["p_seconds"] += sum(r["p_seconds"] for r in doc["rows"])
        timing[f"host:{doc['host']}"] += doc["total_seconds"]
        for r in doc["rows"]:
            r = dict(r, host=doc["host"], unit=doc["unit"])
            prev = seen.get(r["id"])
            if prev and (
                prev["greedy_answer"] != r["greedy_answer"]
                or abs(prev["p_gold"] - r["p_gold"]) > 1e-4
            ):
                raise Refuse(f"{r['id']}: two unit results disagree")
            seen.setdefault(r["id"], r)
    if units != set(range(expected)):
        raise Refuse(f"missing units: {sorted(set(range(expected)) - units)[:10]}")
    gold = {r["id"]: r["label"] for r in eligible}
    if set(seen) != set(gold):
        raise Refuse("scored ids are not exactly the eligible rows")
    for i, r in seen.items():
        if r["label"] != gold[i] or r["greedy_correct"] != (r["greedy_answer"] == gold[i]):
            raise Refuse(f"{i}: label or correctness inconsistent")
        if not (
            isinstance(r["p_gold"], float)
            and 0.0 <= r["p_gold"] <= 1.0
            and r["p_canonical"] <= r["p_gold"] + 1e-9
        ):
            raise Refuse(f"{i}: p_gold invalid")
    cross = {
        "units_checked": len(checks),
        "max_abs_diff": max(c["max_abs_diff"] for c in checks),
        "max_paths_checked": max(c["paths"] for c in checks),
    }
    return seen, {
        "timing": {k: round(v, 1) for k, v in sorted(timing.items())},
        "cross_check": cross,
    }


def references(payload_data: Path, audit: Path, t4: set, near: set) -> dict:
    """{name: {"ids": set, "sha": set}} for every set a pool must never touch."""
    man = payload_data / "manifests"
    splits = json.loads((payload_data / "splits.json").read_text())
    refs = {}
    for name in HOLDOUTS:
        path = man / f"{name}.jsonl"
        if sha_file(path) != splits["manifests_sha256"][f"manifests/{name}.jsonl"]:
            raise Refuse(f"{name}.jsonl differs from splits.json")
        rows = read_jsonl(path)
        refs[name] = {
            "ids": {r["id"] for r in rows},
            "sha": {sha_file(man / r["image"]) for r in rows},
            "manifest_sha256": sha_file(path),
        }
    for name, file in (("native_val", "val-images.json"), ("sealed_test", "test-images.json")):
        raw = (audit / file).read_bytes()
        entries = json.loads(raw)
        refs[name] = {
            "ids": {e["image_id"] for e in entries},
            "sha": {e["encoded_sha256"] for e in entries},
            "audit_file_sha256": sha(raw),
        }
        if len(refs[name]["ids"]) != 3333:
            raise Refuse(f"{name}: expected 3,333 ids, found {len(refs[name]['ids'])}")
    train_sha = {
        e["image_id"]: e["encoded_sha256"]
        for e in json.loads((audit / "train-images.json").read_text())
    }
    refs["train400"] = {"ids": set(t4), "sha": {train_sha[i] for i in t4}}
    refs["val_test_phash_near_duplicates"] = {"ids": set(near), "sha": {train_sha[i] for i in near}}
    for name in refs:
        refs[name]["id_set_sha256"] = id_set_sha(refs[name]["ids"])
        refs[name]["n"] = len(refs[name]["ids"])
    return refs


def check_disjoint(pool_name: str, rows: list[dict], refs: dict) -> dict:
    ids = {r["id"] for r in rows}
    shas = {r["image_sha256"] for r in rows}
    out = {}
    for name, ref in refs.items():
        by_id, by_sha = sorted(ids & ref["ids"]), sorted(shas & ref["sha"])
        if by_id or by_sha:
            raise Refuse(f"{pool_name} overlaps {name}: ids {by_id[:5]} content {by_sha[:3]}")
        out[name] = {"id_overlap": 0, "content_overlap": 0}
    return out


def stratify(pool: str, rows: list[dict]) -> list[dict]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    for label in by:
        by[label].sort(key=lambda r: sha(json.dumps([STUDY, pool, SEED, label, r["id"]]).encode()))
    out, depth = [], max((len(v) for v in by.values()), default=0)
    for i in range(depth):
        for label in sorted(by):
            if i < len(by[label]):
                out.append(by[label][i])
    return out


def build(scores: dict, eligible: list[dict], refs: dict, train_audit: dict, labels: list[str]):
    rows = [scores[r["id"]] for r in eligible]
    for r in rows:
        if train_audit.get(r["id"]) != r["image_sha256"]:
            raise Refuse(f"{r['id']}: image bytes are not the native TRAIN audit's")
    wrong = lambda r: not r["greedy_correct"]  # noqa: E731
    pools = {
        "hard": [r for r in rows if wrong(r) and r["p_gold"] >= P_REACH],
        "mid": [r for r in rows if P_LO <= r["p_gold"] <= P_HI],
        "mixed": [
            r for r in rows if (wrong(r) and r["p_gold"] >= P_REACH) or P_LO <= r["p_gold"] <= P_HI
        ],
        "unreachable": [r for r in rows if r["p_gold"] < P_REACH],
    }
    disjoint = {"eligible": check_disjoint("eligible", rows, refs)}
    ordered = {}
    for name, members in pools.items():
        disjoint[name] = check_disjoint(name, members, refs)
        ordered[name] = stratify(name, members)
    bands = {
        "p<0.02": lambda r: r["p_gold"] < P_REACH,
        "0.02<=p<0.1": lambda r: P_REACH <= r["p_gold"] < P_LO,
        "0.1<=p<=0.9": lambda r: P_LO <= r["p_gold"] <= P_HI,
        "p>0.9": lambda r: r["p_gold"] > P_HI,
    }
    cov = {"bands": {}, "per_class": {}}
    for name, f in bands.items():
        sel = [r for r in rows if f(r)]
        cov["bands"][name] = {
            "n": len(sel),
            "greedy_wrong": sum(wrong(r) for r in sel),
            "greedy_invalid": sum(r["greedy_answer"] is None for r in sel),
        }
    pc = cov["per_class"]
    for label in labels:
        cls = [r for r in rows if r["label"] == label]
        pc[label] = {
            "eligible": len(cls),
            **{k: sum(r["label"] == label for r in v) for k, v in pools.items()},
            "greedy_acc": round(sum(r["greedy_correct"] for r in cls) / len(cls), 4)
            if cls
            else None,
            "mean_p": round(sum(r["p_gold"] for r in cls) / len(cls), 4) if cls else None,
        }
    cov["totals"] = {
        "eligible": len(rows),
        "greedy_correct": sum(r["greedy_correct"] for r in rows),
        "greedy_wrong": sum(wrong(r) for r in rows),
        "greedy_invalid": sum(r["greedy_answer"] is None for r in rows),
        "greedy_wrong_unreachable": sum(wrong(r) and r["p_gold"] < P_REACH for r in rows),
        **{k: len(v) for k, v in pools.items()},
        "hard_and_mid": sum(wrong(r) and P_LO <= r["p_gold"] <= P_HI for r in rows),
        "mean_noncanonical_share": round(
            sum(r["p_gold"] - r["p_canonical"] for r in rows)
            / max(sum(r["p_gold"] for r in rows), 1e-12),
            5,
        ),
    }
    cov["unreachable_classes"] = sorted(
        lab
        for lab in labels
        if pc[lab]["eligible"] and pc[lab]["unreachable"] == pc[lab]["eligible"]
    )
    cov["classes_majority_unreachable"] = sorted(
        lab
        for lab in labels
        if pc[lab]["eligible"] and pc[lab]["unreachable"] * 2 > pc[lab]["eligible"]
    )
    cov["classes_without_eligible_rows"] = sorted(lab for lab in labels if not pc[lab]["eligible"])
    cov["class_coverage"] = {k: sum(pc[lab][k] > 0 for lab in labels) for k in pools}
    cov["class_count_min_max"] = {
        k: [min(pc[lab][k] for lab in labels), max(pc[lab][k] for lab in labels)] for k in pools
    }
    return rows, ordered, disjoint, cov


def trainer_row(r: dict) -> dict:
    return {
        "id": r["id"],
        "image": f"images/train/{r['id']}.jpg",
        "label": r["label"],
        "split": "train",
    }


def coverage_md(cov: dict, timing: dict) -> str:
    t = cov["totals"]
    lines = [
        "# I on the eligible TRAIN rows: RL pool coverage",
        "",
        f"Eligible = all_train (2,329) - train400 (400) - VAL/TEST pHash near-duplicates (477) = "
        f"{t['eligible']} rows. I greedy correct {t['greedy_correct']} "
        f"({t['greedy_correct'] / t['eligible']:.4f}); wrong {t['greedy_wrong']} "
        f"(invalid {t['greedy_invalid']}); wrong and unreachable (p < 0.02) "
        f"{t['greedy_wrong_unreachable']}.",
        f"p_gold summed over the label trie; non-canonical share of total p mass "
        f"{t['mean_noncanonical_share']}.",
        f"Pool-scoring seconds (charge pro rata to pool arms): {timing['unit_seconds']} "
        f"(greedy {timing['greedy_seconds']}, p {timing['p_seconds']}), shared CPU hosts.",
        "",
        "| set | rows | classes covered | min/max per class |",
        "|---|---|---|---|",
    ]
    for k in ("hard", "mid", "mixed", "unreachable"):
        mn, mx = cov["class_count_min_max"][k]
        lines.append(f"| {k} | {t[k]} | {cov['class_coverage'][k]} / 100 | {mn} / {mx} |")
    lines += [
        "",
        f"hard AND mid: {t['hard_and_mid']}",
        "Unreachable classes (every eligible image p < 0.02): "
        f"{cov['unreachable_classes'] or 'none'}",
        "Classes with a majority of images unreachable: "
        f"{cov['classes_majority_unreachable'] or 'none'}",
        "",
        "| p_gold band | rows | greedy wrong | greedy invalid |",
        "|---|---|---|---|",
    ]
    for b, v in cov["bands"].items():
        lines.append(f"| {b} | {v['n']} | {v['greedy_wrong']} | {v['greedy_invalid']} |")
    lines += [
        "",
        "| class | eligible | hard | mid | mixed | unreachable | I greedy acc | mean p_gold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, v in cov["per_class"].items():
        lines.append(
            f"| {label} | {v['eligible']} | {v['hard']} | {v['mid']} | {v['mixed']} | "
            f"{v['unreachable']} | {v['greedy_acc']} | {v['mean_p']} |"
        )
    return "\n".join(lines) + "\n"


def prepare(a):
    man = a.payload_data / "manifests"
    if sha_file(man / "all_train.jsonl") != ALL_TRAIN_SHA256:
        raise Refuse("all_train.jsonl is not the pinned pool")
    all_train = read_jsonl(man / "all_train.jsonl")
    t4, near, ex_meta = exclusion_sets(a, {r["id"] for r in all_train})
    eligible = [r for r in all_train if r["id"] not in t4 | near]
    if len(eligible) != ELIGIBLE_ROWS:
        raise Refuse(f"eligible rows {len(eligible)} != {ELIGIBLE_ROWS}")
    refs = references(a.payload_data, a.audit, t4, near)
    train_audit = {
        e["image_id"]: e["encoded_sha256"]
        for e in json.loads((a.audit / "train-images.json").read_text())
    }
    labels = json.loads((a.payload_data / "data_spec.json").read_text())["labels"]
    return eligible, refs, train_audit, labels, ex_meta


def run(a) -> dict:
    eligible, refs, train_audit, labels, ex_meta = prepare(a)
    scores, meta = load_scores(a.results, eligible)
    if a.self_test:
        return self_test(scores, eligible, refs, train_audit, labels)
    rows, ordered, disjoint, cov = build(scores, eligible, refs, train_audit, labels)
    out = a.out
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    files = {}
    for name, members in ordered.items():
        raw = jsonl(trainer_row(r) for r in members)
        (out / "manifests" / f"{name}.jsonl").write_bytes(raw)
        files[f"manifests/{name}.jsonl"] = {
            "sha256": sha(raw),
            "rows": len(members),
            "id_set_sha256": id_set_sha(r["id"] for r in members),
        }
    keep = (
        "id",
        "label",
        "image_sha256",
        "p_gold",
        "p_canonical",
        "trie_paths",
        "greedy_answer",
        "greedy_correct",
        "terminated",
        "greedy_response",
        "p_seconds",
        "greedy_seconds",
        "host",
        "unit",
    )
    raw = jsonl({k: r[k] for k in keep} for r in rows)
    (out / "scores.jsonl").write_bytes(raw)
    files["scores.jsonl"] = {"sha256": sha(raw), "rows": len(rows)}
    raw = (json.dumps(cov, indent=1, sort_keys=True) + "\n").encode()
    (out / "coverage.json").write_bytes(raw)
    files["coverage.json"] = {"sha256": sha(raw)}
    raw = coverage_md(cov, meta["timing"]).encode()
    (out / "coverage.md").write_bytes(raw)
    files["coverage.md"] = {"sha256": sha(raw)}
    receipt = {
        "study": STUDY,
        "seed": SEED,
        "rule": {
            "eligible": "all_train - train400 - VAL/TEST pHash near-duplicates (477)",
            "p_gold": "P(gold + EOS) at T 1.0 under I, summed over label-trie tokenisations",
            "hard": f"I greedy answer != gold AND p_gold >= {P_REACH}",
            "mid": f"{P_LO} <= p_gold <= {P_HI}",
            "mixed": "hard OR mid",
            "unreachable": f"p_gold < {P_REACH} (never pooled)",
            "order": "class-stratified round robin, hashed within class",
        },
        "adapter_I_sha256": I_WEIGHTS_SHA256,
        "all_train_sha256": ALL_TRAIN_SHA256,
        "trie_sha256": TRIE_SHA256,
        "trainer_sha256": TRAINER_SHA256,
        "splits_json_sha256": sha_file(a.payload_data / "splits.json"),
        "exclusions": ex_meta,
        "references": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("ids", "sha")} for k, v in refs.items()
        },
        "disjointness": disjoint,
        "files": files,
        "pool_scoring": meta["timing"],
        "trainer_cross_check": meta["cross_check"],
        "images_note": "manifests reference images/train/<id>.jpg; place them in the payload's "
        "data/manifests directory (same layout as all_train.jsonl) to train",
    }
    raw = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode()
    (out / "POOLS.json").write_bytes(raw)
    return {
        "POOLS.json": sha(raw),
        **{k: v["sha256"] for k, v in files.items()},
        "counts": {k: v.get("rows") for k, v in files.items()},
    }


def self_test(scores, eligible, refs, train_audit, labels) -> dict:
    """Clean build passes; every planted overlap (by id, and by content) must refuse."""
    build(scores, eligible, refs, train_audit, labels)
    outcomes = {}
    victim = eligible[0]["id"]
    for name, ref in refs.items():
        planted = sorted(ref["ids"])[0]
        # (1) planted by id into the eligible set, forced into hard
        s = copy.deepcopy(scores)
        s[planted] = dict(s[victim], id=planted, greedy_correct=False, p_gold=0.5)
        el = list(eligible) + [dict(eligible[0], id=planted)]
        try:
            build(s, el, refs, dict(train_audit, **{planted: s[planted]["image_sha256"]}), labels)
            outcomes[f"{name}/id"] = "NOT REFUSED"
        except Refuse as exc:
            outcomes[f"{name}/id"] = f"refused: {str(exc)[:110]}"
        # (2) planted by content: an eligible row carrying a reference image's bytes
        s = copy.deepcopy(scores)
        s[victim] = dict(s[victim], image_sha256=sorted(ref["sha"])[0], greedy_correct=False)
        try:
            build(
                s, eligible, refs, dict(train_audit, **{victim: s[victim]["image_sha256"]}), labels
            )
            outcomes[f"{name}/content"] = "NOT REFUSED"
        except Refuse as exc:
            outcomes[f"{name}/content"] = f"refused: {str(exc)[:110]}"
        # (3) the pool-level check alone, on a pool list holding one planted row
        try:
            check_disjoint("hard", [scores[victim], dict(scores[victim], id=planted)], refs)
            outcomes[f"{name}/pool_level"] = "NOT REFUSED"
        except Refuse as exc:
            outcomes[f"{name}/pool_level"] = f"refused: {str(exc)[:110]}"
    s = copy.deepcopy(scores)
    s[victim] = dict(s[victim], image_sha256="0" * 64)
    try:
        build(s, eligible, refs, train_audit, labels)
        outcomes["forged_train_bytes"] = "NOT REFUSED"
    except Refuse as exc:
        outcomes["forged_train_bytes"] = f"refused: {str(exc)[:110]}"
    bad = [k for k, v in outcomes.items() if not v.startswith("refused")]
    if bad:
        raise Refuse(f"self-test: planted overlaps passed: {bad}")
    return {"self_test": "passed", "plants": outcomes}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("results", "payload-data", "audit", "phash", "train400", "exclusions"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if not a.self_test and not a.out:
        p.error("--out is required unless --self-test")
    try:
        print(json.dumps(run(a), indent=1, sort_keys=True))
    except Refuse as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
