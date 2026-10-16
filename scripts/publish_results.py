"""Build the public evidence bundle from measured local runs, without dataset images or keys."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

from vision_rl.data import write_json
from vision_rl.evaluation import compare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/2026-10-16-pawn-counting"
SPECIALIST = ROOT / "runs/real-specialist"
BASELINE = ROOT / "runs/real-baseline/gemini-4-argon-test"


def sanitized(value):
    if isinstance(value, dict):
        return {k: sanitized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitized(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "$WORKSPACE")
    return value


def copy_json(source, name):
    value = sanitized(json.loads(source.read_text()))
    write_json(OUT / name, value)
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = {}
    for name in ("rl", "sft", "initial", "majority"):
        models[name] = copy_json(SPECIALIST / f"test-{name}.json", f"{name}-evaluation.json")
    models["argon"] = copy_json(BASELINE / "evaluation.json", "argon-evaluation.json")
    copy_json(BASELINE / "summary.json", "argon-usage.json")
    copy_json(ROOT / "runs/data-audit.json", "data-audit.json")
    copy_json(ROOT / "runs/mvp-data-audit.json", "additional-data-audit.json")
    source_audit = copy_json(
        ROOT / "runs/real-baseline/annotation-audit.json", "annotation-audit.json"
    )
    copy_json(ROOT / "runs/real-baseline/provenance.json", "api-provenance.json")
    audit = {
        "dataset_fingerprint": models["argon"]["dataset_fingerprint"],
        "status": "issues_found",
        "conclusion": source_audit["conclusion"],
        "finding_statuses": [f["status"] for f in source_audit["findings"]],
    }
    write_json(OUT / "label-audit.json", audit)
    for name in ("rl", "sft"):
        compare(
            OUT / f"{name}-evaluation.json",
            OUT / "argon-evaluation.json",
            OUT / f"{name}-vs-argon.json",
            label_audit=OUT / "label-audit.json",
        )
    compare(
        OUT / "rl-evaluation.json",
        OUT / "sft-evaluation.json",
        OUT / "rl-vs-sft.json",
        label_audit=OUT / "label-audit.json",
    )
    copy_json(SPECIALIST / "training.json", "training.json")
    copy_json(SPECIALIST / "plan.json", "training-plan.json")
    copy_json(SPECIALIST / "heldout-evaluation-plan.json", "heldout-evaluation-plan.json")
    copy_json(
        SPECIALIST / "published-checkpoint-verification.json",
        "checkpoint-replay.json",
    )
    shutil.copyfile(SPECIALIST / "history.jsonl", OUT / "training-history.jsonl")
    # Native provider response body, model ID, token usage, and request settings.
    # Adapter never writes authorization headers or request image bytes into this file.
    shutil.copyfile(BASELINE / "raw-responses.jsonl", OUT / "argon-native-responses.jsonl")
    native = [
        json.loads(line) for line in (OUT / "argon-native-responses.jsonl").read_text().splitlines()
    ]
    if len(native) != 30 or any(row["status_code"] != 200 for row in native):
        raise ValueError("Native response evidence must contain the 30 completed requests")
    with (OUT / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        names = ("majority", "initial", "sft", "rl", "argon")
        writer.writerow(["image_id", "imported_count", *[f"{n}_count" for n in names]])
        indexed = {n: {i["id"]: i for i in models[n]["items"]} for n in names}
        for item in models["argon"]["items"]:
            writer.writerow(
                [
                    item["id"],
                    ord(item["expected"]) - 65,
                    *[ord(indexed[n][item["id"]]["answer"]) - 65 for n in names],
                ]
            )
    checkpoints = OUT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for source, target in (("policy.pt", "rl.pt"), ("sft.pt", "sft.pt")):
        shutil.copyfile(SPECIALIST / source, checkpoints / target)
    harbor = next((ROOT / "jobs/real-count-dogfood").glob("*/result.json"))
    copy_json(harbor, "harbor-trial.json")
    copy_json(harbor.parent / "verifier/evaluation.json", "harbor-evaluation.json")
    controls = {}
    for name, job in (
        ("reference", "oracle-smoke-v3"),
        ("training_agent", "custom-agent-smoke"),
        ("empty_agent", "nop-smoke"),
    ):
        trial = next((ROOT / "jobs" / job).glob("*/result.json"))
        controls[name] = sanitized(json.loads(trial.read_text()))
    write_json(OUT / "harbor-control-trials.json", controls)
    copy_json(ROOT / "runs/docker-demo/summary.json", "docker-demo.json")
    files = {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    write_json(OUT / "checksums.json", files)
    print(f"Published {len(files)} measured evidence files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
