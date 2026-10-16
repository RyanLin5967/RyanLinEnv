"""Re-score validation answers offline into an immutable, provenance-bound snapshot."""

import argparse
import json
import runpy
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE_FILES = ("run.json", "attempts.jsonl", "predictions.jsonl", "summary.json")


def rescore(candidates, output, amendment):
    """Keep original evidence intact while applying one evaluator revision to all candidates."""
    api = runpy.run_path(str(ROOT / "scripts/pills_frontier.py"))
    loop = runpy.run_path(str(ROOT / "scripts/pills_loop.py"))
    output, amendment = Path(output).resolve(), Path(amendment).resolve()
    if output.exists():
        raise ValueError("Re-scored evidence is immutable; choose a new output directory")
    decision = json.loads(amendment.read_text())
    if not isinstance(decision, dict) or not decision:
        raise ValueError("An explicit evaluator amendment is required")
    if not candidates:
        raise ValueError("At least one validation candidate is required")
    source_hashes = {str(amendment): api["sha"](amendment)}
    evaluator_hashes = {
        str(path): api["sha"](path) for path in (
            Path(__file__).resolve(), ROOT / "src/vision_rl/detection_eval.py",
            ROOT / "scripts/pills_frontier.py", ROOT / "scripts/pills_loop.py",
            ROOT / "src/vision_rl/frontier.py",
        )
    }
    results, names = [], set()
    for input_dir, result_dir in candidates:
        input_dir, result_dir = Path(input_dir).resolve(), Path(result_dir).resolve()
        candidate = json.loads((input_dir / "inputs.json").read_text())
        if candidate.get("split") != "validation":
            raise ValueError("Evaluator revision may only re-score validation evidence")
        if result_dir.name in names:
            raise ValueError("Candidate result directory names must be unique")
        names.add(result_dir.name)
        protocol, config, report, hashes = loop["checked_saved_responses"](
            input_dir, result_dir
        )
        old = json.loads((result_dir / "evaluation.json").read_text())
        for name in (*NATIVE_FILES, "evaluation.json"):
            path = result_dir / name
            hashes[str(path)] = api["sha"](path)
        source_hashes.update(hashes)
        report["model"].update(
            coordinate_format=protocol.get("coordinate_format", "yxyx_1000"),
            image_detail=protocol.get("image_detail", "provider-default"),
            structured_output=protocol.get("structured_output", False),
            request_config=config,
        )
        report["evaluation_revision"] = {
            "amendment": str(amendment), "amendment_sha256": api["sha"](amendment),
            "original_result": str(result_dir), "source_hashes": hashes,
            "evaluator_hashes": evaluator_hashes, "new_hosted_calls": 0,
        }
        events = [
            json.loads(line) for line in (result_dir / "attempts.jsonl").read_text().splitlines()
        ]
        hard_stopped = any(event.get("hard_stop") for event in events)
        results.append((result_dir, report, {
            "original_input": str(input_dir), "original_result": str(result_dir),
            "new_result": str(output / result_dir.name), "images": report["n"],
            "provider": config["provider"], "model": config["model"],
            "old_metrics": {key: old[key] for key in ("AP", "AP50", "errors")},
            "new_metrics": {key: report[key] for key in ("AP", "AP50", "errors")},
            "hard_stopped": hard_stopped,
            "scope": "Incomplete; not a model-comparison result" if hard_stopped else "validation",
        }))
    loop["check_hashes"](source_hashes)
    loop["check_hashes"](evaluator_hashes)
    output.mkdir(parents=True)
    for original, report, _ in results:
        destination = output / original.name
        destination.mkdir()
        for name in NATIVE_FILES:
            shutil.copyfile(original / name, destination / name)
            if api["sha"](destination / name) != source_hashes[str(original / name)]:
                raise ValueError("Native evidence changed while copying")
        detections = report.pop("detections")
        for name, value in (("evaluation.json", report), ("detections.json", detections)):
            (destination / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    # The manifest is written last: a partial snapshot is never marked complete.
    loop["check_hashes"](source_hashes)
    loop["check_hashes"](evaluator_hashes)
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": True, "scope": "Offline validation evaluator revision; no final-test access",
        "amendment": decision, "source_hashes": source_hashes,
        "evaluator_hashes": evaluator_hashes, "new_hosted_calls": 0,
        "native_evidence_copied_without_changes": True,
        "candidates": [record for _, _, record in results],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs=2, action="append", required=True,
                        metavar=("INPUT_DIRECTORY", "RESULT_DIRECTORY"))
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = rescore(args.candidate, args.output, args.amendment)
    print(json.dumps({"candidates": len(report["candidates"]), "new_hosted_calls": 0}))


if __name__ == "__main__":
    main()
