"""Emit the exploratory sweep v2 splits and rung-1 arm table, deterministically.

Exploratory only: every number this design produces comes from TRAIN-derived holdouts.
Native VAL and the sealed TEST are never read, listed or copied here.

  python3 -I scripts/exploratory_sweep/design.py [--freshpool DIR] [--native-train JSONL]
      [--out docs/aircraft-exploratory-sweep-v2] [--check]

Writes, under --out:
  manifests/*.jsonl    trainer manifests (id, image, label, split), images relative to the
                       manifest directory as images/train/<id>.jpg
  splits.json          sizes, construction rule, sha256 of every manifest
  arms-rung1.json      the rung-1 arm table the runner reads (wave 1 runnable, wave 2 gated)
--check rebuilds in memory and refuses unless every file on disk is byte-identical.
Only the standard library is used, so it runs anywhere in well under a second.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

STUDY_ID = "aircraft-exploratory-sweep-v2"
SPLIT_SEED = 20261018
SCAN_SEEDS = (42, 314, 2718)
DEV_PER_CLASS = 6
ROOT_CHECKOUT = Path("/Users/idide/RyanLinEnv")
FRESHPOOL = ROOT_CHECKOUT / "runs/aircraft-freshpool-g0-v1/manifests"
NATIVE_TRAIN = ROOT_CHECKOUT / "data/fgvc-aircraft-2013b/development/train.jsonl"
NATIVE_TRAIN_SHA256 = "909ea0c93f72feece41ef0ea8896bc74a751f4753d2f4ec1ac642b2c84783eec"
FRESHPOOL_MANIFEST_SHA256 = None  # recorded, and checked against every file hash it lists

# Initial adapters (weights sha256). W: aircraft-warmstart-v1 final; I: controlled400
# sft-1e5-time published final (native VAL 0.7539).
INITS = {
    "W": {"path": "adapters/W", "weights_sha256":
          "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c",
          "source": "runs/aircraft-warmstart-v1/final_adapter"},
    "I": {"path": "adapters/I", "weights_sha256":
          "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7",
          "source": "runs/aircraft-controlled400-v1/setup/sft-1e5-time/published/final_adapter"},
}
RUNG_UPDATES = {1: 150, 2: 600, 3: 1500}
ALL_TRAIN_ROWS = 2329  # checked against the built all_train manifest in emit()
RUNG_SEEDS = {1: (42,), 2: (42, 314, 2718), 3: (42, 314, 2718)}
CONTROL_LR = {"W": 1e-5, "I": 5e-6}
# The trainer requires a finite original active-time budget for any TRAIN-only SFT run past
# 400 updates (every sweep run declares the rung-3 total). 72 h is a ceiling, never a target:
# stage boundaries are update counts, or, for a time-matched control, the RL arm's seconds.
SFT_ACTIVE_CAP_SECONDS = 259200.0
BASE = {"beta": 0.04, "temperature": 1.0, "num_generations": 4, "screen_threshold": None,
        "response_format": "answer_only", "max_pixels": 200704, "min_pixels": 50176,
        "max_completion_length": 256, "adapter_profile": "legacy-qv-v1", "threads": 16,
        "device": "cpu", "dtype": "float32", "features": []}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl(rows) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def row(r: dict) -> dict:
    if r.get("split") != "train" or r["image"] != f"images/train/{r['id']}.jpg":
        raise ValueError(f"not a native TRAIN row: {r}")
    return {"id": r["id"], "image": r["image"], "label": r["label"], "split": "train"}


def key(*parts) -> str:
    return sha(json.dumps([STUDY_ID, *parts], separators=(",", ":")).encode())


def build_splits(freshpool: Path, native_train: Path):
    manifest_raw = (freshpool / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    for name, digest in manifest["files_sha256"].items():
        if sha((freshpool / name).read_bytes()) != digest:
            raise ValueError(f"freshpool file changed since G0: {name}")
    if sha(native_train.read_bytes()) != NATIVE_TRAIN_SHA256:
        raise ValueError("native TRAIN manifest differs from the pinned development/train.jsonl")
    native = {r["id"]: row(r) for r in read_jsonl(native_train)}
    if len(native) != 3334:
        raise ValueError("native TRAIN must hold 3,334 rows")
    s = [row(r) for r in read_jsonl(freshpool / "s.jsonl")]
    h_select = [row(r) for r in read_jsonl(freshpool / "h_select.jsonl")]
    h_gate = [row(r) for r in read_jsonl(freshpool / "h_gate.jsonl")]
    audit = json.loads((freshpool / "phash_audit.json").read_text())
    for r in s + h_select + h_gate:
        if native.get(r["id"]) != r:
            raise ValueError(f"freshpool row is not the native TRAIN row: {r['id']}")

    # dev600: per class, the 6 S members with the smallest key(dev600, seed, label, id).
    by_class = collections.defaultdict(list)
    for r in s:
        by_class[r["label"]].append(r)
    if len(by_class) != 100 or min(map(len, by_class.values())) < DEV_PER_CLASS + 4:
        raise ValueError("every class needs at least 10 S members to carve dev600")
    dev600 = []
    for label in sorted(by_class):
        ranked = sorted(by_class[label], key=lambda r: key("dev600", SPLIT_SEED, label, r["id"]))
        dev600.extend(ranked[:DEV_PER_CLASS])
    dev600.sort(key=lambda r: r["id"])

    holdout_ids = {r["id"] for r in h_select + h_gate + dev600}
    # Near-duplicate partners (freshpool within-pool pHash pairs, Hamming <= 6) of any
    # holdout image never train. Pool images that near-duplicate native VAL/TEST stay in
    # all_train: they are legitimate native TRAIN, and the strongest SFT baseline uses them.
    partners = set()
    for pair in audit["within_pool_pairs"]:
        a, b = pair["ids"]
        if a in holdout_ids and b not in holdout_ids:
            partners.add(b)
        if b in holdout_ids and a not in holdout_ids:
            partners.add(a)
    val_test_near = {x["pool_id"] for x in audit["external_near_duplicates"]
                     if x["reference_split"] in ("test", "validation")}
    excluded = holdout_ids | partners
    s_train = sorted((r for r in s if r["id"] not in excluded), key=lambda r: r["id"])
    all_train = sorted((r for r in native.values() if r["id"] not in excluded),
                       key=lambda r: r["id"])
    dev800 = sorted(h_select + dev600, key=lambda r: r["id"])
    out = {
        "dev600": dev600, "dev800": dev800, "h_select": sorted(h_select, key=lambda r: r["id"]),
        "h_gate_final": sorted(h_gate, key=lambda r: r["id"]), "s_train": s_train,
        "all_train": all_train,
    }
    for seed in SCAN_SEEDS:
        for name, rows in (("s_train", s_train), ("all_train", all_train)):
            ids = sorted(r["id"] for r in rows)
            random.Random(seed).shuffle(ids)
            lookup = {r["id"]: r for r in rows}
            out[f"{name}_scan_seed{seed}"] = [lookup[i] for i in ids]
    # Invariants, refused rather than warned.
    trains = {"s_train", "all_train"}
    held = {"dev800", "h_gate_final", "dev600", "h_select"}
    for t in trains:
        tid = {r["id"] for r in out[t]}
        for h in held:
            if tid & {r["id"] for r in out[h]}:
                raise ValueError(f"{t} overlaps {h}")
        if tid & partners:
            raise ValueError(f"{t} holds a near-duplicate of a holdout")
    if {r["id"] for r in out["dev800"]} & {r["id"] for r in out["h_gate_final"]}:
        raise ValueError("dev800 overlaps the final-check holdout")
    counts = collections.Counter(r["label"] for r in dev600)
    if set(counts.values()) != {DEV_PER_CLASS} or len(counts) != 100:
        raise ValueError("dev600 is not 6 per class over 100 classes")
    meta = {
        "freshpool_manifest_sha256": sha(manifest_raw),
        "native_train_sha256": NATIVE_TRAIN_SHA256,
        "near_duplicate_partners_excluded": sorted(partners),
        "val_test_near_duplicates_kept_in_all_train": len(val_test_near - excluded),
    }
    return out, meta


def fractional_grpo_w():
    """GRPO-from-W fractional design: full LR x G grid, twice, nuisance factors balanced.

    Every LR x G cell runs twice with complementary (beta, temperature, screening) levels;
    among 4,000 seeded candidate assignments the one with the smallest maximum absolute
    correlation between any two coded factor columns (balance enforced) is kept.
    """
    lrs, gs = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4), (4, 8, 16)
    betas, temps, screens = (0.0, 0.04, 0.1), (0.7, 1.0), (None, 0.5)
    cells = list(itertools.product(range(5), range(3)))
    rng = random.Random(SPLIT_SEED)
    best = None
    for _ in range(4000):
        beta_cols = [rng.sample(range(3), 2) for _ in cells]  # two distinct betas per cell
        temp_first = [rng.randrange(2) for _ in cells]
        screen_first = [rng.randrange(2) for _ in cells]
        runs = []
        for (i, j), b, t, sc in zip(cells, beta_cols, temp_first, screen_first):
            runs.append((i, j, b[0], t, sc))
            runs.append((i, j, b[1], 1 - t, 1 - sc))
        cnt = collections.Counter(r[2] for r in runs)
        if sorted(cnt.values()) != [10, 10, 10]:
            continue
        cols = list(zip(*runs))
        def corr(x, y):
            n = len(x); mx, my = sum(x) / n, sum(y) / n
            vx = sum((a - mx) ** 2 for a in x); vy = sum((b - my) ** 2 for b in y)
            return abs(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (vx * vy) ** 0.5)
        worst = max(corr(cols[a], cols[b]) for a, b in itertools.combinations(range(5), 2))
        if best is None or worst < best[0]:
            best = (worst, runs)
    worst, runs = best
    arms = []
    for i, j, b, t, sc in runs:
        arms.append({"lr": lrs[i], "num_generations": gs[j], "beta": betas[b],
                     "temperature": temps[t], "screen_threshold": screens[sc]})
    return arms, round(worst, 4)


def arm(arm_id, *, track, family, init, mode, lr, wave=1, **overrides):
    a = {"id": arm_id, "wave": wave, "rung": 1, "track": track, "family": family,
         "init": init, "mode": mode, "lr": lr, "seed": 42, "updates": RUNG_UPDATES[1],
         "max_steps_total": RUNG_UPDATES[3], "train_manifest": "s_train_scan_seed42.jsonl",
         **BASE}
    a.update(overrides)
    if mode != "grpo":
        for k in ("beta", "temperature", "screen_threshold"):
            a[k] = None
        if mode == "sft":
            a["num_generations"] = BASE["num_generations"]
        a["max_active_seconds"] = SFT_ACTIVE_CAP_SECONDS
    return a


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def rung1_arms():
    arms = []
    for init in ("W", "I"):
        arms.append({"id": f"eval-{init}", "wave": 1, "rung": 0, "track": "baseline",
                     "family": "baseline", "init": init, "mode": "eval", "seed": 42,
                     **BASE})
    for lr in (5e-6, 1e-5, 3e-5, 1e-4):
        arms.append(arm(f"sft-W-{lr_tag(lr)}", track="sft", family="sft-W", init="W",
                        mode="sft", lr=lr))
    for lr in (1e-6, 5e-6, 1e-5):
        arms.append(arm(f"sft-I-{lr_tag(lr)}", track="sft", family="sft-I", init="I",
                        mode="sft", lr=lr))
    for lr in (1e-5, 3e-5):
        arms.append(arm(f"egsft-W-{lr_tag(lr)}", track="sft", family="egsft-W", init="W",
                        mode="egsft", lr=lr))
    design, worst = fractional_grpo_w()
    for n, d in enumerate(design):
        sc = "s" if d["screen_threshold"] else "n"
        arms.append(arm(
            f"grpo-W-{n:02d}-lr{lr_tag(d['lr'])}-g{d['num_generations']}-b{d['beta']}"
            f"-t{d['temperature']}-{sc}", track="rl", family="grpo-W", init="W", mode="grpo",
            lr=d["lr"], num_generations=d["num_generations"], beta=d["beta"],
            temperature=d["temperature"], screen_threshold=d["screen_threshold"]))
    for lr, g in itertools.product((1e-6, 5e-6, 1e-5), (4, 8)):
        arms.append(arm(f"grpo-I-lr{lr_tag(lr)}-g{g}", track="rl", family="grpo-I", init="I",
                        mode="grpo", lr=lr, num_generations=g))
    # Data scale: SFT on all native TRAIN minus holdouts, two epochs, assayed per epoch.
    # SFT(all) -> GRPO arms are conditional (created at promotion, see the plan).
    n_all = ALL_TRAIN_ROWS
    for init, lr in (("W", 1e-5), ("W", 3e-5), ("I", 5e-6)):
        arms.append(arm(f"sftall-{init}-{lr_tag(lr)}", track="sftall", family=f"sftall-{init}",
                        init=init, mode="sft", lr=lr, updates=n_all, stops=[n_all, 2 * n_all],
                        max_steps_total=2 * n_all,
                        train_manifest="all_train_scan_seed42.jsonl"))
    # Additions beyond the brief (docs/aircraft-exploratory-sweep-v2.md, "Additions").
    for seed in (314, 2718):
        arms.append(arm(f"noise-sft-W-1e-5-s{seed}", track="noise", family="noise-sft",
                        init="W", mode="sft", lr=1e-5, seed=seed,
                        train_manifest=f"s_train_scan_seed{seed}.jsonl"))
        arms.append(arm(f"noise-grpo-W-1e-5-g4-s{seed}", track="noise", family="noise-grpo",
                        init="W", mode="grpo", lr=1e-5, seed=seed,
                        train_manifest=f"s_train_scan_seed{seed}.jsonl"))
    # Resolution: the trainer binds an init adapter to its parent's pixel profile, so a
    # higher-resolution arm from W needs a declared transfer (wave 2, gated).
    arms.append(arm("w2-res-sft-W-1e-5-px401k", wave=2, track="sft", family="w2-res-sft-W",
                    init="W", mode="sft", lr=1e-5, max_pixels=401408,
                    features=["resolution_transfer"]))
    arms.append(arm("w2-res-grpo-W-1e-5-g8-px401k", wave=2, track="rl", family="w2-res-grpo-W",
                    init="W", mode="grpo", lr=1e-5, num_generations=8, max_pixels=401408,
                    features=["resolution_transfer"]))
    for lr in (5e-6, 1e-5):
        arms.append(arm(f"think-grpo-W-{lr_tag(lr)}-g8", track="rl", family="think-grpo-W",
                        init="W", mode="grpo", lr=lr, num_generations=8,
                        response_format="think_then_answer"))
    # Wave 2: gated on a trainer feature the host receipt must list (runner refuses else).
    wave2 = [
        ("drgrpo", "dr_grpo", {}), ("trie", "trie_constrained_decoding", {}),
        ("confuser", "confuser_unlikelihood", {}), ("ccE", "constrained_ce", {}),
        ("hier", "hierarchical_reward", {}), ("closedset", "closed_set_prompt", {}),
        ("cot", "cot_answer_reward", {"response_format": "think_then_answer"}),
        ("dapo", "dapo", {}), ("ppu4", "prompts_per_update", {}),
        ("mu2", "num_iterations", {}), ("cosine", "lr_schedule", {}),
        ("auxnll", "rl_aux_nll", {}), ("fmtpen", "invalid_answer_penalty", {}),
    ]
    for tag, feature, extra in wave2:
        for lr in (5e-6, 1e-5):
            arms.append(arm(f"w2-{tag}-grpo-W-{lr_tag(lr)}-g8", wave=2, track="rl",
                            family=f"w2-{tag}", init="W", mode="grpo", lr=lr,
                            num_generations=8, features=[feature], **extra))
    for lr in (1e-5, 3e-5):
        arms.append(arm(f"w2-rft-W-{lr_tag(lr)}", wave=2, track="rl", family="w2-rft",
                        init="W", mode="rft", lr=lr, num_generations=8,
                        features=["rejection_sampling_ft"]))
        arms.append(arm(f"w2-aug-sft-W-{lr_tag(lr)}", wave=2, track="sft",
                        family="w2-aug-sft", init="W", mode="sft", lr=lr,
                        features=["sft_augmentation"]))
    for a in arms:
        if a["mode"] in ("sft", "egsft") and a["track"] != "baseline":
            a["order"] = "random"
        if a["mode"] != "eval":
            a.setdefault("stops", [a["updates"]])
    return arms, worst


def controls_for(arms):
    """One matched SFT control per RL arm: same init, its consumed prompt sequence.

    E = examples-matched endpoint (same updates); T = active-time-matched endpoint (the
    same SFT run continued, cycling the sequence, to the RL arm's active seconds), run on
    the RL arm's own host right after it.
    """
    out = []
    for a in arms:
        if a["mode"] in ("grpo", "rft"):
            out.append({
                "id": f"{a['id']}~ctl", "wave": a["wave"], "rung": a["rung"],
                "track": "control", "family": "control", "init": a["init"], "mode": "sft",
                "lr": CONTROL_LR[a["init"]], "seed": a["seed"], "control_of": a["id"],
                "depends_on": [a["id"]], "same_host_as": a["id"], "order": "sequential",
                "cycle": True, "train_manifest": f"@consumed:{a['id']}",
                **{k: v for k, v in BASE.items() if k not in ("beta", "temperature",
                   "screen_threshold")},
                "max_pixels": a["max_pixels"], "response_format": "answer_only",
                "max_active_seconds": SFT_ACTIVE_CAP_SECONDS,
                "features": [f for f in a["features"] if f == "cuda"],
            })
    return out


def emit(out_dir: Path, freshpool: Path, native_train: Path) -> dict[str, bytes]:
    splits, meta = build_splits(freshpool, native_train)
    files = {f"manifests/{name}.jsonl": jsonl(rows) for name, rows in sorted(splits.items())}
    if len(splits["all_train"]) != ALL_TRAIN_ROWS:
        raise ValueError("ALL_TRAIN_ROWS differs from the built all_train manifest")
    arms, worst = rung1_arms()
    arms += controls_for(arms)
    ids = [a["id"] for a in arms]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate arm id")
    splits_doc = {
        "study_id": STUDY_ID, "split_seed": SPLIT_SEED, **meta,
        "sizes": {name: len(rows) for name, rows in sorted(splits.items())},
        "manifests_sha256": {k: sha(v) for k, v in files.items()},
        "rules": {
            "dev600": "from S (freshpool s.jsonl): per class the 6 rows with the smallest "
                      "sha256(json [study_id,'dev600',20261018,label,id]); 600 = 6 x 100",
            "dev800": "h_select (200) + dev600 (600): the selection holdout",
            "h_gate_final": "freshpool h_gate (200): touched once, by rung-3 finalists only",
            "s_train": "S - dev600 - pHash partners (within-pool Hamming <= 6) of any holdout",
            "all_train": "native TRAIN - h_select - h_gate - dev600 - holdout pHash partners "
                         "(pool images the G0 audit flagged as near-duplicates of native "
                         "VAL/TEST are kept: legitimate native TRAIN; strongest baseline)",
            "scan": "random.Random(seed).shuffle(sorted(ids)) for seeds 42, 314, 2718",
        },
    }
    files["splits.json"] = (json.dumps(splits_doc, indent=1, sort_keys=True) + "\n").encode()
    table = {"study_id": STUDY_ID, "schema_version": 1, "rung": 1,
             "fractional_design_max_abs_factor_correlation": worst,
             "inits": INITS, "rung_updates": RUNG_UPDATES,
             "rung_seeds": {str(k): list(v) for k, v in RUNG_SEEDS.items()},
             "control_lr": CONTROL_LR, "arms": arms}
    files["arms-rung1.json"] = (json.dumps(table, indent=1, sort_keys=True) + "\n").encode()
    return files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--freshpool", type=Path, default=FRESHPOOL)
    p.add_argument("--native-train", type=Path, default=NATIVE_TRAIN)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[2] / "docs" / STUDY_ID)
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    files = emit(a.out, a.freshpool, a.native_train)
    if a.check:
        bad = [n for n, raw in files.items()
               if not (a.out / n).is_file() or (a.out / n).read_bytes() != raw]
        if bad:
            print(f"design drift: {bad}", file=sys.stderr)
            return 1
        print(f"design check ok: {len(files)} files")
        return 0
    for name, raw in files.items():
        path = a.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    table = json.loads(files["arms-rung1.json"])
    by = collections.Counter((x["wave"], x["track"]) for x in table["arms"])
    print(json.dumps({"files": len(files), "arms": len(table["arms"]),
                      "by_wave_track": {f"{w}/{t}": n for (w, t), n in sorted(by.items())},
                      "sizes": json.loads(files["splits.json"])["sizes"],
                      "max_abs_corr": table["fractional_design_max_abs_factor_correlation"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
