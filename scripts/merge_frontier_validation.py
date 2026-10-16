"""Pool disjoint discovery and confirmation answers offline; never make an API call."""

import argparse
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def merge(discovery_input, discovery_run, complement_input, complement_run, output):
    """Validate every saved answer against its journal and score the full validation set."""
    wrapper = runpy.run_path(str(ROOT / "scripts/pills_frontier.py"))
    loop = runpy.run_path(str(ROOT / "scripts/pills_loop.py"))
    output = Path(output)
    if output.exists():
        raise ValueError("Merged evidence is immutable; choose a new output directory")
    inputs = [Path(discovery_input).resolve(), Path(complement_input).resolve()]
    runs = [Path(discovery_run).resolve(), Path(complement_run).resolve()]
    protocols, configurations, predictions, sources = [], [], {}, {}
    for input_dir, result_dir in zip(inputs, runs, strict=True):
        candidate = json.loads((input_dir / "inputs.json").read_text())
        if candidate.get("split") != "validation":
            raise ValueError("Only validation evidence may be merged")
        protocol, config, _, hashes = loop["checked_saved_responses"](input_dir, result_dir)
        events = [
            json.loads(line) for line in (result_dir / "attempts.jsonl").read_text().splitlines()
        ]
        if any(event.get("hard_stop") for event in events):
            raise ValueError("Incomplete hard-stopped validation cannot form a pooled comparison")
        sources.update(hashes)
        protocols.append(protocol)
        configurations.append(config)
        for line in (result_dir / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("error", "").startswith(("not_requested_", "unknown_after_interruption")):
                raise ValueError("Incomplete validation answers cannot form a pooled comparison")
            if row["id"] in predictions:
                raise ValueError("Discovery and confirmation IDs overlap")
            predictions[row["id"]] = row
    original, complement = protocols
    if original.get("validation_complement") or not complement.get("validation_complement"):
        raise ValueError("Merge requires original discovery followed by its explicit complement")
    reference = complement["validation_complement"]
    for key, path in (
        ("source_inputs", inputs[0] / "inputs.json"), ("source_run", runs[0] / "run.json")
    ):
        if (
            Path(reference[key]["path"]).resolve() != path
            or reference[key]["sha256"] != wrapper["sha"](path)
        ):
            raise ValueError("Complement does not reference this exact discovery experiment")
    settings = [{k: v for k, v in c.items() if k != "max_requests"} for c in configurations]
    if settings[0] != settings[1]:
        raise ValueError("Discovery and confirmation request settings differ")
    if wrapper["complement_settings"](original) != wrapper["complement_settings"](complement):
        raise ValueError("Discovery and confirmation prompt, schema, or examples differ")
    config = configurations[0]
    full = wrapper["build_protocol"](SimpleNamespace(
        source=Path(original["annotation"]).parent.parent,
        split="validation", count=None, seed=original["sampling_seed"],
        prompt=original["prompt_mode"], examples=len(original["examples"]),
        provider=config["provider"], image_detail=original.get("image_detail"),
        structured_output=original.get("structured_output", False),
    ))
    if wrapper["complement_settings"](full) != wrapper["complement_settings"](original):
        raise ValueError("Full validation protocol differs from original discovery")
    row_map = {row["id"]: row for protocol in protocols for row in protocol["rows"]}
    if set(row_map) != {row["id"] for row in full["rows"]}:
        raise ValueError("Merged evidence does not cover every native validation image")
    if any(row_map[row["id"]] != row for row in full["rows"]):
        raise ValueError("Merged image bytes or prompts differ from full validation protocol")
    record = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "validation confirmation; not a final-test result",
        "source_hashes": sources, "request_settings": settings[0],
        "partition_sizes": [len(protocol["rows"]) for protocol in protocols],
        "disjoint_ids": True, "all_native_validation_images": True,
        "scoring": "Pooled native COCO AP from every saved answer; never averaged partition AP.",
        "hosted_requests_made_by_merge": 0,
    }
    report = wrapper["score_predictions"](full, predictions, {
        "provider": config["provider"], "name": config["model"], "effort": config["effort"],
        "max_output_tokens": config["max_output_tokens"],
        "coordinate_format": full.get("coordinate_format", "yxyx_1000"),
        "image_detail": full.get("image_detail", "provider-default"),
        "validation_union": record,
    })
    # Check again after scoring, before writing any publishable result.
    loop["check_hashes"](sources)
    output.mkdir(parents=True)
    for name, value in (
        ("inputs.json", full), ("merge.json", record),
        ("detections.json", report.pop("detections")), ("evaluation.json", report),
    ):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (output / "predictions.jsonl").write_text("".join(
        json.dumps(predictions[row["id"]], allow_nan=False) + "\n" for row in full["rows"]
    ))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "discovery-input", "discovery-run", "complement-input", "complement-run", "output"
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = merge(**vars(args))
    print(json.dumps({key: report[key] for key in ("n", "AP", "AP50", "errors")}))


if __name__ == "__main__":
    main()
