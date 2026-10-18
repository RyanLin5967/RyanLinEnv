"""Reduce the I-on-all_train scores into hard / mid / mixed RL pools, and prove disjointness.

  python3 -I build_pools.py --results DIR --payload-data DIR --audit DIR [--phash FILE] --out DIR
  python3 -I build_pools.py ... --self-test        # planted-overlap fixture; must refuse each plant

Standard library only. Inputs:
  --results       score_worker.py shard files (results/shard-NNN.json)
  --payload-data  the sweep payload's data/ (manifests/*.jsonl, images under manifests/images/train,
                  splits.json): the all_train pool and every TRAIN-derived holdout
  --audit         data/fgvc-aircraft-2013b/audit: train-/val-/test-images.json. ONLY two fields are
                  read from the VAL and TEST files, image_id and encoded_sha256, and only to prove
                  disjointness. No label, prediction or score of VAL or TEST is read or produced.
  --phash         optional runs/aircraft-freshpool-g0-v1/manifests/phash_audit.json: rows flagged
                  as pHash near-duplicates (Hamming <= 6) of a VAL/TEST image are ANNOTATED, not
                  dropped (design.py keeps them in all_train on purpose; the SFT baseline uses them).

Pools (pre-registered here, before any score was read):
  hard  = greedy answer of I is not the gold label (wrong OR invalid: no EOS / no vocabulary label)
  mid   = 0.1 <= p_gold <= 0.9   (p_gold = P(exact gold + EOS) under I at temperature 1.0)
  mixed = hard UNION mid
Each pool manifest is in trainer format ({id, image, label, split} with images/train/<id>.jpg
relative to the manifest directory, the same layout as the payload's data/manifests) and is
CLASS-STRATIFIED: rows are interleaved round-robin over classes (round r takes every class's r-th
member; classes in sorted label order; members within a class ordered by
sha256(json([STUDY, pool, SEED, label, id]))), so every prefix is as class-balanced as the pool allows.

Refuses (exit 2, reason on stderr) when: a shard is missing, duplicated with a disagreeing value,
or carries a different adapter/payload/profile; a scored id set is not exactly all_train; any pool
row's id, or its image content sha256, is in dev800, dev600, h_select, h_gate_final, native VAL or
sealed TEST; a pool row's image bytes differ from the native TRAIN audit's encoded_sha256 for its id.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

STUDY = "aircraft-rl-campaign-I-pools-v1"
SEED = 20261018
I_WEIGHTS_SHA256 = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
ALL_TRAIN_SHA256 = "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e"
P_LO, P_HI = 0.1, 0.9
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


def load_scores(results: Path, all_train: list[dict]) -> dict:
    files = sorted(results.glob("shard-*.json"))
    expected = (len(all_train) + 31) // 32
    seen = {}
    shards = set()
    for f in files:
        doc = json.loads(f.read_text())
        if doc.get("adapter_sha256") != I_WEIGHTS_SHA256:
            raise Refuse(f"{f.name}: adapter {doc.get('adapter_sha256')} is not I")
        if doc.get("all_train_sha256") != ALL_TRAIN_SHA256:
            raise Refuse(f"{f.name}: scored a different all_train")
        if doc["ids"] != [r["id"] for r in doc["rows"]]:
            raise Refuse(f"{f.name}: ids and rows disagree")
        k = doc["shard"]
        if doc["ids"] != [r["id"] for r in all_train[32 * k:32 * k + 32]]:
            raise Refuse(f"{f.name}: shard {k} does not hold all_train rows {32 * k}..")
        shards.add(k)
        for r in doc["rows"]:
            r = dict(r, host=doc["host"], shard=k)
            prev = seen.get(r["id"])
            if prev and (prev["greedy_answer"] != r["greedy_answer"]
                         or abs(prev["p_gold"] - r["p_gold"]) > 1e-4):
                raise Refuse(f"{r['id']}: two shard results disagree")
            seen.setdefault(r["id"], r)
    if shards != set(range(expected)):
        raise Refuse(f"missing shards: {sorted(set(range(expected)) - shards)[:10]}")
    gold = {r["id"]: r["label"] for r in all_train}
    if set(seen) != set(gold):
        raise Refuse("scored ids are not exactly all_train")
    for i, r in seen.items():
        if r["label"] != gold[i] or r["greedy_correct"] != (r["greedy_answer"] == gold[i]):
            raise Refuse(f"{i}: label or correctness inconsistent")
        if not (isinstance(r["p_gold"], float) and 0.0 <= r["p_gold"] <= 1.0):
            raise Refuse(f"{i}: p_gold invalid")
    return seen


def references(payload_data: Path, audit: Path) -> dict:
    """{name: {"ids": set, "sha": set}} for every set a pool must never touch."""
    man = payload_data / "manifests"
    splits = json.loads((payload_data / "splits.json").read_text())
    refs = {}
    for name in HOLDOUTS:
        path = man / f"{name}.jsonl"
        if sha_file(path) != splits["manifests_sha256"][f"manifests/{name}.jsonl"]:
            raise Refuse(f"{name}.jsonl differs from splits.json")
        rows = read_jsonl(path)
        refs[name] = {"ids": {r["id"] for r in rows},
                      "sha": {sha_file(man / r["image"]) for r in rows},
                      "manifest_sha256": sha_file(path)}
    for name, file in (("native_val", "val-images.json"), ("sealed_test", "test-images.json")):
        raw = (audit / file).read_bytes()
        entries = json.loads(raw)
        refs[name] = {"ids": {e["image_id"] for e in entries},
                      "sha": {e["encoded_sha256"] for e in entries},
                      "audit_file_sha256": sha(raw)}
        if len(refs[name]["ids"]) != 3333:
            raise Refuse(f"{name}: expected 3,333 ids, found {len(refs[name]['ids'])}")
    for name in refs:
        refs[name]["id_set_sha256"] = id_set_sha(refs[name]["ids"])
        refs[name]["n"] = len(refs[name]["ids"])
    return refs


def check_disjoint(pool_name: str, rows: list[dict], refs: dict) -> dict:
    ids = {r["id"] for r in rows}
    shas = {r["image_sha256"] for r in rows}
    out = {}
    for name, ref in refs.items():
        by_id = sorted(ids & ref["ids"])
        by_sha = sorted(shas & ref["sha"])
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


def build(scores: dict, all_train: list[dict], refs: dict, train_audit: dict, phash: dict | None):
    rows = [scores[r["id"]] for r in all_train]
    for r in rows:
        want = train_audit.get(r["id"])
        if want is None or want != r["image_sha256"]:
            raise Refuse(f"{r['id']}: image bytes are not the native TRAIN audit's")
    pools = {
        "hard": [r for r in rows if not r["greedy_correct"]],
        "mid": [r for r in rows if P_LO <= r["p_gold"] <= P_HI],
    }
    pools["mixed"] = [r for r in rows if not r["greedy_correct"] or P_LO <= r["p_gold"] <= P_HI]
    disjoint = {"all_train": check_disjoint("all_train", rows, refs)}
    ordered = {}
    for name, members in pools.items():
        disjoint[name] = check_disjoint(name, members, refs)
        ordered[name] = stratify(name, members)
    near = {}
    if phash:
        for x in phash["external_near_duplicates"]:
            if x["reference_split"] in ("test", "validation"):
                near.setdefault(x["pool_id"], set()).add(x["reference_split"])
    labels = sorted({r["label"] for r in all_train})
    coverage = {"bands": {}, "per_class": {}}
    bands = {
        "p<0.1": lambda r: r["p_gold"] < P_LO, "0.1<=p<=0.9": lambda r: P_LO <= r["p_gold"] <= P_HI,
        "p>0.9": lambda r: r["p_gold"] > P_HI,
    }
    for name, f in bands.items():
        sel = [r for r in rows if f(r)]
        coverage["bands"][name] = {"n": len(sel), "greedy_wrong": sum(not r["greedy_correct"] for r in sel),
                                   "greedy_invalid": sum(r["greedy_answer"] is None for r in sel)}
    coverage["totals"] = {"all_train": len(rows),
                          "greedy_correct": sum(r["greedy_correct"] for r in rows),
                          "greedy_invalid": sum(r["greedy_answer"] is None for r in rows),
                          **{k: len(v) for k, v in pools.items()},
                          "hard_and_mid": sum((not r["greedy_correct"]) and P_LO <= r["p_gold"] <= P_HI
                                              for r in rows),
                          "phash_flagged_near_val_test": {k: sum(r["id"] in near for r in v)
                                                          for k, v in pools.items()} if phash else None}
    for label in labels:
        cls = [r for r in rows if r["label"] == label]
        coverage["per_class"][label] = {
            "all_train": len(cls), **{k: sum(r["label"] == label for r in v) for k, v in pools.items()},
            "greedy_acc": round(sum(r["greedy_correct"] for r in cls) / len(cls), 4),
            "mean_p": round(sum(r["p_gold"] for r in cls) / len(cls), 4)}
    pc = coverage["per_class"]
    coverage["class_coverage"] = {k: sum(pc[l][k] > 0 for l in labels) for k in pools}
    coverage["class_count_min_max"] = {k: [min(pc[l][k] for l in labels), max(pc[l][k] for l in labels)]
                                       for k in pools}
    return rows, ordered, disjoint, coverage, near


def trainer_row(r: dict) -> dict:
    return {"id": r["id"], "image": f"images/train/{r['id']}.jpg", "label": r["label"], "split": "train"}


def coverage_md(cov: dict) -> str:
    t = cov["totals"]
    lines = ["# I on all_train: pool coverage", "",
             f"all_train {t['all_train']} rows; I greedy correct {t['greedy_correct']} "
             f"({t['greedy_correct'] / t['all_train']:.4f}); invalid {t['greedy_invalid']}.", "",
             "| pool | rows | classes covered | min/max per class |", "|---|---|---|---|"]
    for k in ("hard", "mid", "mixed"):
        mn, mx = cov["class_count_min_max"][k]
        lines.append(f"| {k} | {t[k]} | {cov['class_coverage'][k]} / 100 | {mn} / {mx} |")
    lines += ["", f"hard AND mid: {t['hard_and_mid']}", "",
              "| p_gold band | rows | greedy wrong | greedy invalid |", "|---|---|---|---|"]
    for b, v in cov["bands"].items():
        lines.append(f"| {b} | {v['n']} | {v['greedy_wrong']} | {v['greedy_invalid']} |")
    lines += ["", "| class | all_train | hard | mid | mixed | I greedy acc | mean p_gold |",
              "|---|---|---|---|---|---|---|"]
    for label, v in cov["per_class"].items():
        lines.append(f"| {label} | {v['all_train']} | {v['hard']} | {v['mid']} | {v['mixed']} | "
                     f"{v['greedy_acc']} | {v['mean_p']} |")
    return "\n".join(lines) + "\n"


def run(a) -> dict:
    man = a.payload_data / "manifests"
    if sha_file(man / "all_train.jsonl") != ALL_TRAIN_SHA256:
        raise Refuse("all_train.jsonl is not the pinned pool")
    all_train = read_jsonl(man / "all_train.jsonl")
    for r in all_train:
        r["_abs"] = str(man / r["image"])
    refs = references(a.payload_data, a.audit)
    train_audit = {e["image_id"]: e["encoded_sha256"]
                   for e in json.loads((a.audit / "train-images.json").read_text())}
    phash = json.loads(a.phash.read_text()) if a.phash else None
    scores = load_scores(a.results, [{k: v for k, v in r.items() if k != "_abs"} for r in all_train])
    if a.self_test:
        return self_test(scores, all_train, refs, train_audit, phash)
    rows, ordered, disjoint, coverage, near = build(scores, all_train, refs, train_audit, phash)
    out = a.out
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    files = {}
    for name, members in ordered.items():
        raw = jsonl(trainer_row(r) for r in members)
        (out / "manifests" / f"{name}.jsonl").write_bytes(raw)
        files[f"manifests/{name}.jsonl"] = {"sha256": sha(raw), "rows": len(members),
                                            "id_set_sha256": id_set_sha(r["id"] for r in members)}
    score_rows = [{k: r[k] for k in ("id", "label", "image_sha256", "p_gold", "greedy_answer",
                                     "greedy_correct", "terminated", "greedy_response", "host", "shard")}
                  | {"phash_near_val_test": sorted(near.get(r["id"], [])) if phash else None}
                  for r in rows]
    raw = jsonl(score_rows)
    (out / "scores.jsonl").write_bytes(raw)
    files["scores.jsonl"] = {"sha256": sha(raw), "rows": len(score_rows)}
    raw = (json.dumps(coverage, indent=1, sort_keys=True) + "\n").encode()
    (out / "coverage.json").write_bytes(raw)
    files["coverage.json"] = {"sha256": sha(raw)}
    raw = coverage_md(coverage).encode()
    (out / "coverage.md").write_bytes(raw)
    files["coverage.md"] = {"sha256": sha(raw)}
    receipt = {
        "study": STUDY, "seed": SEED, "rule": {"hard": "I greedy answer != gold (wrong or invalid)",
                                              "mid": f"{P_LO} <= p_gold <= {P_HI}", "mixed": "hard OR mid"},
        "adapter_I_sha256": I_WEIGHTS_SHA256, "all_train_sha256": ALL_TRAIN_SHA256,
        "splits_json_sha256": sha_file(a.payload_data / "splits.json"),
        "references": {k: {kk: vv for kk, vv in v.items() if kk not in ("ids", "sha")}
                       for k, v in refs.items()},
        "disjointness": disjoint, "files": files,
        "hosts": dict(collections.Counter(r["host"] for r in rows)),
        "images_dir_note": "manifests reference images/train/<id>.jpg; copy them next to the payload's "
                           "data/manifests (same layout) to train",
    }
    raw = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode()
    (out / "POOLS.json").write_bytes(raw)
    return {"POOLS.json": sha(raw), **{k: v["sha256"] for k, v in files.items()},
            "counts": {k: v.get("rows") for k, v in files.items()}}


def self_test(scores, all_train, refs, train_audit, phash) -> dict:
    """Clean build passes; every planted overlap (by id, and by content) must refuse."""
    build(scores, all_train, refs, train_audit, phash)  # clean: must not raise
    outcomes = {}
    victim = all_train[0]["id"]
    for name, ref in refs.items():
        for kind in ("id", "content"):
            s = copy.deepcopy(scores)
            planted = sorted(ref["ids"])[0] if kind == "id" else None
            if kind == "id":
                r = dict(s[victim], id=planted, greedy_correct=False)  # force it into hard
                s[planted] = r
                at = [dict(x) for x in all_train] + [dict(all_train[0], id=planted)]
                ta = dict(train_audit, **{planted: r["image_sha256"]})
            else:
                s[victim] = dict(s[victim], image_sha256=sorted(ref["sha"])[0], greedy_correct=False)
                at, ta = all_train, dict(train_audit, **{victim: s[victim]["image_sha256"]})
            try:
                build(s, at, refs, ta, phash)
                outcomes[f"{name}/{kind}"] = "NOT REFUSED"
            except Refuse as exc:
                outcomes[f"{name}/{kind}"] = f"refused: {str(exc)[:120]}"
    # The pool-level check on its own (the plants above are already caught by the all_train
    # check, which runs first): a pool list holding one planted row must refuse.
    for name, ref in refs.items():
        planted = dict(scores[victim], id=sorted(ref["ids"])[0])
        try:
            check_disjoint("hard", [scores[victim], planted], refs)
            outcomes[f"pool_level/{name}"] = "NOT REFUSED"
        except Refuse as exc:
            outcomes[f"pool_level/{name}"] = f"refused: {str(exc)[:120]}"
    bad = [k for k, v in outcomes.items() if not v.startswith("refused")]
    if bad:
        raise Refuse(f"self-test: planted overlaps passed: {bad}")
    # The non-stratified sanity: a forged image (bytes differ from the TRAIN audit) must refuse.
    s = copy.deepcopy(scores)
    s[victim] = dict(s[victim], image_sha256="0" * 64)
    try:
        build(s, all_train, refs, train_audit, phash)
        raise Refuse("self-test: forged TRAIN image bytes passed")
    except Refuse as exc:
        if "self-test" in str(exc):
            raise
        outcomes["forged_train_bytes"] = f"refused: {str(exc)[:120]}"
    return {"self_test": "passed", "plants": outcomes}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--payload-data", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--phash", type=Path)
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
