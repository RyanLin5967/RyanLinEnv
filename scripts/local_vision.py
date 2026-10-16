"""Prepare immutable local vision development inputs without model or test access."""

import argparse
import collections
import json
from pathlib import Path

from vision_rl.detection_eval import parse_detection_response
from vision_rl.frontier import _prepare, _public
from vision_rl.local_vlm import GENERATION, REPOSITORY, REVISION, run_local, sha_file

WHEAT_TASK = {
    "task": "object_detection", "benchmark_id": "gwhd2021-hf-domain-ada-v1",
    "coordinate_format": "xyxy_1000", "categories": [{"id": 1, "name": "wheat_head"}],
    "confidence_policy": "Retain native confidence when present; otherwise assign1.0",
    "prediction_score_test": ">0.5", "matching_iou_test": ">=0.5",
    "maximum_predictions_per_image": 500,
    "postprocessing": "Strict shared parser; no clipping, deduplication, repair or NMS",
    "primary_metric": "WILDS2.0.0 detection_acc_avg_dom",
    "errors": "Preserve every image and failure; incomplete scopes cannot support a win",
}
WHEAT_PROMPT = (
    "Locate every wheat head in this image, including partially visible wheat heads. "
    "Return one JSON array with one object per wheat head: "
    '{"bbox_2d":[xmin,ymin,xmax,ymax],"label":"wheat_head"}. '
    "Coordinates must be integers normalized to0..1000 relative to the entire image, "
    "in xmin,ymin,xmax,ymax order. Use the label wheat_head for every detection. "
    "Return at most500 detections. Return [] if no wheat heads are present."
)


def immutable_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prepare_wheat_probe(source, output):
    """Use TRAIN density only to choose one stress image and one median image."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Prepared local inputs are immutable")
    annotation = source / "train/_annotations.coco.json"
    document = json.loads(annotation.read_text())
    if [{"id": x["id"], "name": x["name"]} for x in document["categories"]] != (
        WHEAT_TASK["categories"]
    ):
        raise ValueError("Wheat category vocabulary differs from the frozen task")
    counts = collections.Counter(x["image_id"] for x in document["annotations"])
    images = document["images"]
    if len(images) < 2 or len({row["id"] for row in images}) != len(images):
        raise ValueError("Need at least two distinct native training images")
    if any(row.get("native_split", "train") != "train" for row in images):
        raise ValueError("Probe metadata must contain training images only")
    ordered = sorted(images, key=lambda row: (counts[row["id"]], str(row["id"])))
    largest, median = ordered[-1], ordered[len(ordered) // 2]
    if largest["id"] == median["id"]:
        median = ordered[0]
    rows = []
    for image in (largest, median):
        path = (source / "train" / image["file_name"]).resolve(strict=True)
        if not path.is_relative_to(source / "train"):
            raise ValueError("Training image escapes its declared split")
        rows.append({"id": str(image["id"]), "image": str(path), "split": "train",
                     "prompt": WHEAT_PROMPT})
    protocol = {
        "schema_version": 1, "kind": "generic_local_vision", "phase": "train_probe",
        "scope": "Two TRAIN images for resource and geometry feasibility; no benchmark score",
        "selection": "Maximum training box count then median count; lexical ID breaks ties",
        "source": str(source), "image_root": str(source / "train"),
        "annotation": str(annotation), "annotation_sha256": sha_file(annotation),
        "task_contract": json.loads(json.dumps(WHEAT_TASK)), "rows": rows,
        "prepared_images": [_public(_prepare(row)) for row in rows],
        "selection_audit": [{"id": str(image["id"]), "domain": image.get("domain"),
                             "training_box_count": counts[image["id"]]}
                            for image in (largest, median)],
        "model": {"repository": REPOSITORY, "revision": REVISION,
                  "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}},
        "generation": dict(GENERATION), "reasoning_effort": "xhigh",
        "native_processor": True, "external_resize": None, "examples": [],
    }
    immutable_write(output / "inputs.json", protocol)
    return protocol


def verify_inputs(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("kind") != "generic_local_vision" or protocol.get("phase") not in {
        "train_probe", "validation",
    }:
        raise ValueError("Generic development inputs cannot open sealed test images")
    expected_model = {"repository": REPOSITORY, "revision": REVISION,
                      "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}}
    if (protocol.get("model") != expected_model or protocol.get("generation") != GENERATION
            or protocol.get("reasoning_effort") != "xhigh"
            or protocol.get("native_processor") is not True
            or protocol.get("external_resize") is not None or protocol.get("examples") != []):
        raise ValueError("Prepared native model, processor or generation contract is unsupported")
    root = Path(protocol["image_root"]).resolve(strict=True)
    split = "train" if protocol["phase"] == "train_probe" else "validation"
    for row in protocol["rows"]:
        if row.get("split") != split or not Path(row["image"]).resolve(
            strict=True
        ).is_relative_to(root):
            raise ValueError("Image split or root differs from the prepared contract")
    if sha_file(protocol["annotation"]) != protocol["annotation_sha256"]:
        raise ValueError("Annotation identity changed")
    if [_public(_prepare(row)) for row in protocol["rows"]] != protocol["prepared_images"]:
        raise ValueError("Prepared image or prompt identity changed")
    return protocol


def run(args):
    path = args.input / "inputs.json"
    protocol = verify_inputs(path)
    # Bind the exact complete preparation, including selection and scoring policy.
    task = {**protocol["task_contract"], "prepared_protocol_sha256": sha_file(path)}
    predictions = run_local(
        protocol["rows"], args.output, model_directory=args.model_directory,
        integrity=args.integrity, phase=protocol["phase"], seed=args.seed,
        memory_limit_bytes=int(args.memory_limit_gib * 1024**3), resume=args.resume,
        image_root=protocol["image_root"], task_contract=task,
    )
    geometry = []
    for image in protocol["prepared_images"]:
        prediction = predictions[image["id"]]
        record = {"id": image["id"], "error": prediction.get("error")}
        if task["task"] == "object_detection" and not record["error"]:
            try:
                boxes = parse_detection_response(
                    prediction["response"], {**image, "id": int(image["id"])},
                    task["categories"], coordinate_format=task["coordinate_format"],
                )
                if len(boxes) > task["maximum_predictions_per_image"]:
                    raise ValueError("Response exceeds the frozen prediction count limit")
                record["valid_detection_count"] = len(boxes)
            except (ValueError, TypeError, KeyError) as exc:
                record["error"] = str(exc)
        geometry.append(record)
    summary = {"scope": protocol["scope"], "phase": protocol["phase"],
               "protocol_sha256": sha_file(path), "geometry": geometry,
               "errors": sum(bool(row["error"]) for row in geometry),
               "accuracy_score_computed": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-wheat-probe")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("run")
    for name in ("input", "output", "model-directory", "integrity"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--seed", type=int, default=1729)
    infer.add_argument("--memory-limit-gib", type=float, default=32)
    infer.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = (prepare_wheat_probe(args.source, args.output)
              if args.command == "prepare-wheat-probe" else run(args))
    print(json.dumps({"phase": result["phase"], "scope": result["scope"]}))


if __name__ == "__main__":
    main()
