"""Command-line entry points. Network work is always explicitly selected."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from vision_rl.data import assert_disjoint, load_manifest, write_json
from vision_rl.evaluation import compare, evaluate
from vision_rl.spatial import generate
from vision_rl.training import train, train_agent


def demo(output: Path, steps: int = 800) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"demo output must be empty: {output}")
    dataset = output / "data"
    generate(dataset)
    assert_disjoint(
        *(load_manifest(dataset / f"{s}.jsonl", s) for s in ("train", "validation", "test"))
    )
    train_agent(dataset / "train.jsonl", dataset / "validation.jsonl", output / "training", steps)
    initial = evaluate(
        dataset / "test.jsonl",
        output / "initial.json",
        checkpoint=output / "training/initial.npz",
        model="untrained-spatial-policy",
    )
    trained = evaluate(
        dataset / "test.jsonl",
        output / "specialist.json",
        checkpoint=output / "training/policy.npz",
        model="trained-spatial-policy",
    )
    comparison = compare(
        output / "specialist.json", output / "initial.json", output / "comparison.json"
    )
    result = {
        "task": "synthetic-spatial-onboarding",
        "frontier_comparison": False,
        "initial_accuracy": initial["accuracy"],
        "trained_accuracy": trained["accuracy"],
        "test_examples": trained["n"],
        "paired_ci95": comparison["paired_bootstrap_ci95"],
        "artifacts": str(output.resolve()),
    }
    write_json(output / "summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train specialist vision models and measure capability gaps."
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Run the complete local CPU RL learning experiment")
    d.add_argument("--output", type=Path, default=Path("runs/demo"))
    d.add_argument("--steps", type=int, default=800)
    g = sub.add_parser(
        "generate", help="Generate independent synthetic train/validation/test splits"
    )
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--seed", type=int, default=42)
    for name, count in (("train", 1024), ("validation", 256), ("test", 512)):
        g.add_argument(f"--{name}", type=int, default=count)
    for command in ("train", "train-agent"):
        t = sub.add_parser(command, help="Train the CPU spatial policy using reward feedback")
        t.add_argument("--train", type=Path, required=True)
        t.add_argument("--validation", type=Path, required=True)
        t.add_argument("--output", type=Path, required=True)
        t.add_argument("--steps", type=int, default=800)
        t.add_argument("--seed", type=int, default=42)
        if command == "train":
            t.add_argument("--learning-rate", type=float, default=0.03)
            t.add_argument("--checkpoint", type=Path)
        else:
            t.add_argument("--candidates", type=int, default=3)
    e = sub.add_parser("evaluate", help="Score a CPU checkpoint, multimodal API, or predictions")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    source = e.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--endpoint", help="Chat-completions base URL, usually ending in /v1")
    source.add_argument("--predictions", type=Path, help="JSONL with id and raw response")
    source.add_argument("--vlm-model", help="Hugging Face VLM ID or directory; requires CUDA")
    e.add_argument("--adapter", type=Path, help="Local LoRA adapter for --vlm-model")
    e.add_argument("--model")
    e.add_argument("--max-requests", type=int, default=100)
    e.add_argument("--max-tokens", type=int, default=128)
    c = sub.add_parser(
        "compare", help="Compare results on identical examples using paired bootstrap"
    )
    c.add_argument("specialist", type=Path)
    c.add_argument("baseline", type=Path)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--label-audit", type=Path, help="Dataset-specific annotation audit JSON")
    review = sub.add_parser("review", help="Build a local image-by-image disagreement report")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--reports", nargs="+", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    a = sub.add_parser(
        "audit", help="Validate manifests and reject overlapping image pixels or IDs"
    )
    a.add_argument("manifests", nargs="+", type=Path)
    task = sub.add_parser("make-vlm-task", help="Package a specialist GPU training task for Harbor")
    for split in ("train", "validation", "test"):
        task.add_argument(f"--{split}", type=Path, required=True)
    task.add_argument("--output", type=Path, required=True)
    task.add_argument("--model", required=True)
    task.add_argument("--max-steps", type=int, default=100)
    task.add_argument("--max-tokens", type=int, default=64)
    count_task = sub.add_parser(
        "make-count-task", help="Package a real-image CPU training Harbor task"
    )
    for split in ("train", "validation", "test"):
        count_task.add_argument(f"--{split}", type=Path, required=True)
    count_task.add_argument("--output", type=Path, required=True)
    count_task.add_argument("--sft-epochs", type=int, default=160)
    count_task.add_argument("--rl-steps", type=int, default=800)
    detection_task = sub.add_parser(
        "make-detection-task", help="Package a native COCO detector and RL refinement task"
    )
    detection_task.add_argument("--source", type=Path, required=True)
    detection_task.add_argument("--output", type=Path, required=True)
    detection_task.add_argument("--detector-epochs", type=int, default=40)
    detection_task.add_argument("--rl-epochs", type=int, default=30)
    detection_task.add_argument("--warmup-epochs", type=int, default=8)
    detection_task.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    detection_task.add_argument("--batch-size", type=int, default=1)
    detection_task.add_argument("--grad-accum-steps", type=int, default=4)
    detection_task.add_argument("--preflight-batches", type=int, default=0)
    detection_task.add_argument("--queries", type=int, default=128)
    detection_task.add_argument("--base-image", default="vision-rl:detection")
    detection_task.add_argument("--timeout-sec", type=int, default=86460)
    sub.add_parser(
        "doctor", help="Check Python, Docker and credential presence without showing secrets"
    )
    rf = sub.add_parser("roboflow", help="Read workspace projects and import versioned datasets")
    rsub = rf.add_subparsers(dest="rf_command", required=True)
    rlist = rsub.add_parser("list")
    rlist.add_argument("--workspace")
    rd = rsub.add_parser("download")
    rd.add_argument("--workspace", required=True)
    rd.add_argument("--project", required=True)
    rd.add_argument("--version", type=int, required=True)
    rd.add_argument("--format", default="coco")
    rd.add_argument("--output", type=Path, required=True)
    ri = rsub.add_parser("import-classification")
    ri.add_argument("--folder", type=Path, required=True)
    ri.add_argument("--output", type=Path, required=True)
    rc = rsub.add_parser("import-coco")
    rc.add_argument("--folder", type=Path, required=True)
    rc.add_argument("--output", type=Path, required=True)
    rc.add_argument("--category", dest="target_class", help="COCO class to count; omit for all")
    rc.add_argument("--max-count", type=int, default=10)
    rc.add_argument("--question", help="Explicit visual question, e.g. count black and white pawns")
    return p


def main() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)
    args = vars(parser().parse_args())
    command = args.pop("command")
    try:
        if command == "demo":
            result = demo(**args)
        elif command == "generate":
            result = generate(**args)
        elif command == "train":
            result = train(**args)
        elif command == "train-agent":
            args["train_path"] = args.pop("train")
            args["validation_path"] = args.pop("validation")
            result = train_agent(**args)
        elif command == "evaluate":
            full = evaluate(**args)
            result = {k: v for k, v in full.items() if k != "items"}
        elif command == "compare":
            result = compare(**args)
        elif command == "review":
            from vision_rl.review import make_review

            result = make_review(**args)
        elif command == "audit":
            datasets = [load_manifest(path) for path in args["manifests"]]
            assert_disjoint(*datasets)
            result = {"valid": True, "examples": [len(rows) for rows in datasets]}
        elif command == "make-vlm-task":
            from vision_rl.task_builder import make_vlm_task

            result = make_vlm_task(**args)
        elif command == "make-count-task":
            from vision_rl.count_task_builder import make_count_task

            result = make_count_task(**args)
        elif command == "make-detection-task":
            from vision_rl.detection_task_builder import make_detection_task

            result = make_detection_task(**args)
        elif command == "doctor":
            import importlib.util
            import os
            import platform
            from importlib.metadata import PackageNotFoundError, version

            try:
                process = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                docker = process.stdout.strip() if process.returncode == 0 else "not reachable"
            except (OSError, subprocess.TimeoutExpired):
                docker = "not reachable"
            result = {
                "python": platform.python_version(),
                "docker": docker,
                "roboflow_key_present": bool(os.getenv("ROBOFLOW_API_KEY")),
                "model_key_present": bool(os.getenv("MODEL_API_KEY")),
            }
            result["packages"] = {}
            for package in ("harbor", "torch", "rfdetr", "pycocotools", "safetensors"):
                try:
                    result["packages"][package] = version(package)
                except PackageNotFoundError:
                    result["packages"][package] = None
            result["devices"] = ["cpu"]
            if importlib.util.find_spec("torch"):
                import torch

                if torch.cuda.is_available():
                    result["devices"].append("cuda")
                if torch.backends.mps.is_available():
                    result["devices"].append("mps")
        else:
            from vision_rl import roboflow

            rf_command = args.pop("rf_command")
            fn = {
                "list": roboflow.list_projects,
                "download": roboflow.download_dataset,
                "import-classification": roboflow.import_classification,
                "import-coco": roboflow.import_coco_counts,
            }[rf_command]
            result = fn(**args)
        print(json.dumps(result, indent=2, default=str, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
