"""Export the canonical native-VAL labels of the declared comparison endpoints.

Local preparation only, standard library only: no model, no image, no TEST file. It reads
the completed eight-endpoint evaluator bundle (runs/aircraft-native-val-activation-v1/bundle,
evaluator frozen at 85f0865), checks every prediction journal against the SHA256 that the
committed raw report (reports/2026-10-17-controlled400-native-val/comparison.json) recorded,
and writes {arm: {image id: canonical label or null}} for the arms the named-baseline CI
plan compares against.

The export is accepted only if it reproduces the committed comparison exactly: for every
exported arm other than the warm initialization, the canonical-input SHA256 of
(gold, arm, common) must equal ``versus_common_initialization[arm].input_sha256`` in that
report. That hash covers the gold labels and both arms' canonical outcomes for all 3,333
images, so a wrong, reordered or partial export cannot pass. The CI merge job repeats the
check with the frozen ``compare_classification`` itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/2026-10-17-controlled400-native-val/comparison.json"
ARMS = ("common", "rl-exposure", "sft-1e5-time")
NATIVE_IMAGES = 3333


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_input_sha256(labels, gold, candidate, comparator):
    """The frozen compare_classification input identity, recomputed independently.

    Rows are ordered by image ID, as the frozen ``_inputs`` returns them."""
    rows = [
        {
            "id": row["id"],
            "gold": row["label"],
            "candidate": candidate.get(row["id"]),
            "candidate_recorded": row["id"] in candidate,
            "comparator": comparator.get(row["id"]),
            "comparator_recorded": row["id"] in comparator,
        }
        for row in sorted(gold, key=lambda row: row["id"])
    ]
    payload = {"labels": list(labels), "rows": rows}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def journal_labels(path, identifiers):
    """Canonical outcome per image from an evaluator journal, in denominator order."""
    outcomes = {}
    for line in Path(path).read_text().splitlines():
        event = json.loads(line)
        if event["event"] != "outcome":
            continue
        if event["id"] in outcomes:
            raise ValueError(f"Repeated outcome in {path}")
        if event["result"]["status"] != "completed":
            raise ValueError(f"Non-completed outcome in {path}: {event['id']}")
        outcomes[event["id"]] = event["result"]["canonical_label"]
    if list(outcomes) != list(identifiers):
        raise ValueError(f"Journal does not cover the native denominator in order: {path}")
    return outcomes


def export(bundle, output):
    bundle = Path(bundle)
    report = json.loads(REPORT.read_text())
    plan = json.loads((bundle / "plan.json").read_text())
    if sha_file(bundle / "plan.json") != report["plan_sha256"]:
        raise ValueError("Evaluator bundle plan differs from the committed report")
    gold = json.loads((bundle / "gold.json").read_text())
    identifiers = [row["id"] for row in gold]
    if len(identifiers) != NATIVE_IMAGES or len(set(identifiers)) != NATIVE_IMAGES:
        raise ValueError("Gold must hold all 3333 native VAL images once")
    if report["strongest_declared_sft"] != ARMS[2]:
        raise ValueError("The committed strongest declared SFT changed")
    arms, journals = {}, {}
    for arm in ARMS:
        path = bundle / "arms" / arm / "predictions.jsonl"
        journals[arm] = sha_file(path)
        if journals[arm] != report["prediction_journal_sha256"][arm]:
            raise ValueError(f"Prediction journal differs from the committed report: {arm}")
        arms[arm] = journal_labels(path, identifiers)
    for arm in ARMS[1:]:
        expected = report["versus_common_initialization"][arm]["input_sha256"]
        observed = canonical_input_sha256(plan["labels"], gold, arms[arm], arms["common"])
        if observed != expected:
            raise ValueError(f"Export does not reproduce the committed comparison: {arm}")
    value = {
        "schema_version": 1,
        "scope": "FGVC-Aircraft 2013b native VAL canonical outcomes; no TEST",
        "evaluator_commit": "85f0865920015fd7e637d2b997bd9d7597dc1e83",
        "report": "reports/2026-10-17-controlled400-native-val/comparison.json",
        "report_sha256": sha_file(REPORT),
        "bundle_plan_sha256": report["plan_sha256"],
        "gold_sha256": sha_file(bundle / "gold.json"),
        "prediction_journal_sha256": journals,
        "roles": {"W": "common", "RL400": "rl-exposure", "strongest_sft": ARMS[2]},
        "reproduces": {
            arm: report["versus_common_initialization"][arm]["input_sha256"] for arm in ARMS[1:]
        },
        "arms": arms,
    }
    Path(output).write_text(json.dumps(value, indent=1, sort_keys=False) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = export(args.bundle, args.output)
    print(json.dumps({"exported": list(value["arms"]), "images": NATIVE_IMAGES}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
