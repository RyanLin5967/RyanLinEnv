"""Reduce the lane's fetched dev800 assays into docs/aircraft-sft-ceiling-v1/report.json.

  python3 -I scripts/sft_ceiling/report.py

Reads docs/aircraft-sft-ceiling-v1/raw/<run>/results/{summary.json,assays/*.json} (raw, committed
before this runs) and I's dev800 assay from the CPU sweep (eval-I). Paired bootstrap is the sweep
runner's own paired_bootstrap (imported from the 9d9a09f payload, not reimplemented), 20000
resamples, seed 20261018. Refuses on any assay that is not exactly dev800 (sha b01175a1...).
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
DOCS = WT / "docs/aircraft-sft-ceiling-v1"
RAW = DOCS / "raw"
PAYLOAD = Path("/Users/idide/RyanLinEnv/runs/aircraft-exploratory-sweep-v2/payload-9d9a09f")
DEV800 = PAYLOAD / "data/manifests/dev800.jsonl"
DEV800_SHA = "b01175a1ea66985384c7d6c615305c9956c12978a1af3528f0e1c0a484be25b5"
I_ASSAY = Path("/Users/idide/RyanLinEnv/runs/aircraft-exploratory-sweep-v2/results/"
               "eval-I-1792328724/assays/init.json")
I_SHA = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
RESAMPLES, SEED = 20000, 20261018


def runner():
    path = PAYLOAD / "code/scripts/exploratory_sweep/runner.py"
    spec = importlib.util.spec_from_file_location("sweep_runner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sweep_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def load(doc_path, gold):
    doc = json.loads(doc_path.read_text())
    m = doc["metrics"]
    if m["manifest_sha256"] != DEV800_SHA or set(doc["predictions"]) != set(gold):
        sys.exit(f"REFUSE: {doc_path} is not a dev800 assay")
    acc = sum(doc["predictions"][i] == g for i, g in gold.items()) / len(gold)
    if abs(acc - m["accuracy"]) > 1e-12:
        sys.exit(f"REFUSE: {doc_path} accuracy does not recompute from its predictions")
    return m, doc["predictions"]


def main():
    r = runner()
    gold = {json.loads(x)["id"]: json.loads(x)["label"] for x in DEV800.read_text().splitlines()}
    i_metrics, i_pred = load(I_ASSAY, gold)
    if i_metrics["adapter_sha256"] != I_SHA:
        sys.exit("REFUSE: eval-I assay is not adapter I")
    rows = []
    for summary_path in sorted(RAW.glob("*/results/summary.json")):
        s = json.loads(summary_path.read_text())
        arm = json.loads((summary_path.parent / "arm.json").read_text())
        for ep in s["endpoints"]:
            m, pred = load(summary_path.parent / "assays" / f"{ep['label']}.json", gold)
            row = {"arm": s["arm"], "family": arm["family"], "lr": arm["lr"], "seed": arm["seed"],
                   "init": arm["init"], "label": ep["label"], "updates": ep["steps"],
                   "active_seconds": ep.get("active_seconds"), "host": s["host"],
                   "accuracy": m["accuracy"], "valid_fraction": m["valid_fraction"],
                   "eos_fraction": m["eos_fraction"],
                   "equal_class_accuracy": m["equal_class_accuracy"],
                   "adapter_sha256": m["adapter_sha256"],
                   "vs_I": r.paired_bootstrap(pred, i_pred, gold, RESAMPLES, SEED)}
            rows.append(row)
    if not rows:
        sys.exit("REFUSE: no fetched endpoints under raw/")
    seeds = [x for x in rows if x["family"] == "I-seed"]
    seed_accs = [i_metrics["accuracy"]] + [x["accuracy"] for x in seeds]
    out = {"holdout": "dev800", "dev800_sha256": DEV800_SHA, "bootstrap": {
               "resamples": RESAMPLES, "seed": SEED, "impl": "runner.paired_bootstrap @9d9a09f"},
           "I": {"adapter_sha256": I_SHA, "accuracy": i_metrics["accuracy"],
                 "trained_on": "mps (controlled400)"},
           "I_seed_spread": {
               "accuracies": {"I(seed42,mps)": i_metrics["accuracy"],
                              **{f"seed{x['seed']}(cpu)": x["accuracy"] for x in seeds}},
               "n": len(seed_accs), "mean": statistics.mean(seed_accs),
               "sd": statistics.stdev(seed_accs) if len(seed_accs) > 1 else None,
               "range": max(seed_accs) - min(seed_accs)},
           "endpoints": rows}
    (DOCS / "report.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"I {i_metrics['accuracy']:.4f}")
    for x in rows:
        b = x["vs_I"]
        print(f"{x['arm']:<28} {x['label']:<9} upd={x['updates']:<5} acc={x['accuracy']:.4f} "
              f"dI={b['delta']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] "
              f"p<=0={b['p_le_0']:.3f}")
    print("seed spread", json.dumps(out["I_seed_spread"]))


main()
