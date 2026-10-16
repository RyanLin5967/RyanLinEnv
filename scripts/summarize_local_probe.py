"""Derive native usage and resource evidence from a finished local probe; never infer."""

import argparse
import json
import statistics
from pathlib import Path

from vision_rl.local_vlm import sha_file


def summarize(result, supervision):
    result, supervision = Path(result), Path(supervision)
    status = json.loads(supervision.read_text())
    if "finished_at" not in status:
        raise ValueError("A live probe cannot be summarized as complete")
    run = json.loads((result / "run.json").read_text())
    identity = run["identity"]
    expected = {row["id"]: row for row in identity["rows"]}
    reserved, completed = {}, {}
    for line in (result / "attempts.jsonl").read_text().splitlines():
        event = json.loads(line)
        identifier = event["id"]
        if event["identity_sha256"] != run["identity_sha256"] or identifier not in expected:
            raise ValueError("Native journal identity differs")
        if event["event"] == "reserved":
            if identifier in reserved or event["request"] != expected[identifier]:
                raise ValueError("Duplicate or changed native reservation")
            reserved[identifier] = event
        elif event["event"] == "completed":
            if identifier not in reserved or identifier in completed:
                raise ValueError("Duplicate or unreserved native completion")
            completed[identifier] = event
        else:
            raise ValueError("Unexpected native journal event")
    items, totals, durations = [], {}, []
    for identifier, request in expected.items():
        item = {"id": identifier, "input_pixels": request["width"] * request["height"],
                "model_pixel_sha256": request["pixel_sha256"]}
        if identifier not in completed:
            item["error"] = ("unknown_after_interruption" if identifier in reserved
                             else "not_requested")
        else:
            event = completed[identifier]
            native = event.get("native_response", {})
            item.update(
                finish_reason=native.get("finish_reason"), error=event["prediction"].get("error"),
                latency_seconds=event["prediction"]["latency_seconds"],
                peak_mlx_memory_bytes=event.get("peak_memory_bytes"),
                explicit_thinking_closes=native.get("text", "").count("</think>"),
                usage={key: native.get(key) for key in (
                    "prompt_tokens", "generation_tokens", "total_tokens",
                    "prompt_tps", "generation_tps",
                )},
                image_grid_thw=event.get("preprocessing", {}).get("image_grid_thw"),
                input_ids_shape=event.get("preprocessing", {}).get("input_ids_shape"),
            )
            durations.append(item["latency_seconds"])
            for key in ("prompt_tokens", "generation_tokens", "total_tokens"):
                if native.get(key) is not None:
                    totals[key] = totals.get(key, 0) + native[key]
        items.append(item)
    normal = sum(item.get("finish_reason") == "stop" and not item.get("error") for item in items)
    return {
        "scope": "Training feasibility probe only; no held-out benchmark claim",
        "model": {key: identity["model"][key]
                  for key in ("repository", "revision", "quantization")},
        "configuration": identity["config"], "images": items,
        "coverage": {"expected": len(expected), "reserved": len(reserved),
                     "completed": len(completed), "normal_finish": normal,
                     "unknown": len(set(reserved) - set(completed)),
                     "unattempted": len(set(expected) - set(reserved)),
                     "all_finished_normally": normal == len(expected)},
        "known_native_token_totals": totals,
        "usage_unknown_completed_attempts": sum(
            event.get("native_response", {}).get("total_tokens") is None
            for event in completed.values()),
        "latency_seconds": {"sum": sum(durations),
                            "median": statistics.median(durations) if durations else None,
                            "min": min(durations) if durations else None,
                            "max": max(durations) if durations else None},
        "resource_supervision": status,
        "sources_sha256": {str(path): sha_file(path) for path in (
            result / "run.json", result / "attempts.jsonl", supervision,
        )},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.result, args.supervision)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["coverage"]))


if __name__ == "__main__":
    main()
