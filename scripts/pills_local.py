"""Prepare a two-training-image resource probe or run an explicit local validation protocol."""

import argparse
import json
import runpy
from pathlib import Path

from vision_rl.frontier import _prepare, _public
from vision_rl.local_vlm import run_local, sha_file

ROOT = Path(__file__).resolve().parents[1]


def prepare_probe(source, output):
    wrapper = runpy.run_path(str(ROOT / "scripts/pills_frontier.py"))
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Prepared local probes are immutable")
    annotation = source / "train/_annotations.coco.json"
    data = json.loads(annotation.read_text())
    ordered = sorted(data["images"], key=lambda r: (r["width"] * r["height"], r["id"]))
    if len(ordered) < 2:
        raise ValueError("Need at least two native training images")
    largest, median = ordered[-1], ordered[len(ordered) // 2]
    if largest["id"] == median["id"]:
        median = next(row for row in ordered if row["id"] != largest["id"])
    prompt = wrapper["router_prompt"](data["categories"], "qwen", rich=False)
    rows = [{"id": str(row["id"]), "image": str(wrapper["image_path"](
        source / "train", row)), "prompt": prompt} for row in (largest, median)]
    protocol = {
        "schema_version": 1, "phase": "train_probe", "split": "train",
        "scope": "Resource feasibility only; these are training images, not benchmark evidence",
        "selection": "Largest training image by pixels, then median; ID breaks area ties",
        "annotation": str(annotation), "annotation_sha256": sha_file(annotation),
        "coordinate_format": "xyxy_1000", "prompt_mode": "minimal", "examples": [],
        "rows": rows, "prepared_images": [_public(_prepare(row)) for row in rows],
    }
    output.mkdir(parents=True)
    (output / "inputs.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def run(args):
    wrapper = runpy.run_path(str(ROOT / "scripts/pills_frontier.py"))
    path = args.input / "inputs.json"
    protocol = json.loads(path.read_text())
    if protocol.get("phase") == "train_probe":
        if Path(protocol["annotation"]).parent.name != "train" or (
            sha_file(protocol["annotation"]) != protocol["annotation_sha256"]
        ):
            raise ValueError("Training probe annotation identity changed")
        if [_public(_prepare(row)) for row in protocol["rows"]] != protocol["prepared_images"]:
            raise ValueError("Training probe image or prompt changed")
        phase = "train_probe"
    elif protocol.get("split") == "validation" and protocol.get("provider") == "qwen":
        wrapper["verify_protocol"](protocol)
        if protocol["examples"]:
            raise ValueError("This local protocol has no few-shot examples")
        phase = "validation"
    else:
        raise ValueError("Local run accepts only a prepared training probe or Qwen validation")
    predictions = run_local(
        protocol["rows"], args.output, model_directory=args.model_directory,
        integrity=args.integrity, phase=phase, seed=args.seed,
        memory_limit_bytes=int(args.memory_limit_gib * 1024**3), resume=args.resume,
    )
    summary = {
        "phase": phase, "images": len(protocol["rows"]), "protocol_sha256": sha_file(path),
        "errors": sum(bool(row.get("error")) for row in predictions.values()),
        "scope": protocol.get("scope", "Local quantized model validation only"),
    }
    if phase == "validation":
        identity = json.loads((args.output / "run.json").read_text())["identity"]
        report = wrapper["score_predictions"](protocol, predictions, {
            "provider": "qwen", "name": identity["model"]["repository"],
            "revision": identity["model"]["revision"], "runtime": "local MLX",
            "quantization": identity["model"]["quantization"], "effort": "xhigh",
            "protocol_sha256": sha_file(path), "coordinate_format": "xyxy_1000",
        })
        detections = report.pop("detections")
        for name, value in (("evaluation.json", report), ("detections.json", detections)):
            (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
        summary.update(AP=report["AP"], errors=report["errors"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-probe")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("run")
    for name in ("input", "output", "model-directory", "integrity"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--seed", type=int, default=1729)
    infer.add_argument("--memory-limit-gib", type=float, default=32)
    infer.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = (
        prepare_probe(args.source, args.output) if args.command == "prepare-probe" else run(args)
    )
    print(json.dumps({key: result[key] for key in ("phase", "scope") if key in result}))


if __name__ == "__main__":
    main()
