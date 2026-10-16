"""Label-free RF-DETR inference, auditable TTA, and measured throughput.

The inference manifest contains images only; category IDs come from the frozen
training mapping. Numeric checkpoints use the same trusted architecture loader as
the Harbor verifier. Alternative resolutions use RF-DETR's eager positional
interpolation, never a resized or partially loaded checkpoint. TTA is a candidate
to validate, not a promised improvement.

Official implementation: https://github.com/roboflow/rf-detr
Resolution/backend guidance: https://rfdetr.roboflow.com/latest/learn/run/detection/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class InferenceConfig:
    resolutions: tuple[int, ...] = (384,)
    horizontal_flip: bool = False
    fusion_iou: float = 0.55
    max_detections: int = 500
    batch_size: int = 4
    cpu_threads: int = 2
    device: str = "cpu"
    architecture: str = "nano"
    checkpoint_resolution: int | None = None

    def validate(self) -> None:
        if self.architecture not in {"nano", "small"}:
            raise ValueError("Architecture must be nano or small")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Device must be cpu, mps, or cuda")
        if self.checkpoint_resolution is not None and (
            type(self.checkpoint_resolution) is not int
            or self.checkpoint_resolution not in (384, 512)
        ):
            raise ValueError("Trusted checkpoint resolution must be 384 or 512")
        if not self.resolutions or len(self.resolutions) > 4:
            raise ValueError("Use between one and four resolutions")
        if len(set(self.resolutions)) != len(self.resolutions):
            raise ValueError("Duplicate resolutions would overweight one view")
        if any(type(r) is not int or r < 128 or r > 1024 or r % 32 for r in self.resolutions):
            raise ValueError("Resolutions must be multiples of 32 between 128 and 1024")
        if not math.isfinite(self.fusion_iou) or not 0 < self.fusion_iou <= 1:
            raise ValueError("Fusion IoU must be in (0, 1]")
        if type(self.max_detections) is not int or not 1 <= self.max_detections <= 500:
            raise ValueError("At most 500 detections are permitted")
        if not 1 <= self.batch_size <= 64 or not 1 <= self.cpu_threads <= 64:
            raise ValueError("Batch size and CPU threads must be in 1..64")

    @property
    def views(self) -> list[tuple[int, bool]]:
        return [
            (r, f)
            for r in self.resolutions
            for f in ((False, True) if self.horizontal_flip else (False,))
        ]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_image_manifest(path: Path, image_root: Path) -> list[dict]:
    """Require a separate image-only manifest; never open annotation files."""
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or set(document) != {"images"}:
        raise ValueError("Inference manifest must contain only an images array, without labels")
    if not isinstance(document["images"], list) or not document["images"]:
        raise ValueError("Inference manifest must contain images")
    root, ids, rows = image_root.resolve(), set(), []
    for row in document["images"]:
        if not isinstance(row, dict) or set(row) != {"id", "file_name", "width", "height"}:
            raise ValueError("Image rows require exactly id, file_name, width, height")
        if type(row["id"]) is not int or row["id"] in ids:
            raise ValueError("Image IDs must be unique integers")
        if any(type(row[axis]) is not int or row[axis] <= 0 for axis in ("width", "height")):
            raise ValueError("Image dimensions must be positive integers")
        if not isinstance(row["file_name"], str):
            raise ValueError("Image filename must be text")
        image_path = (root / row["file_name"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError("Image path must be a file within the image root")
        ids.add(row["id"])
        rows.append({**row, "path": image_path})
    return rows


def _categories(category_ids: tuple[int, ...]) -> None:
    if (
        not category_ids
        or len(category_ids) > 256
        or any(type(c) is not int or not 0 <= c < 2**63 for c in category_ids)
        or len(set(category_ids)) != len(category_ids)
    ):
        raise ValueError("Category IDs must be a nonempty, unique training-order mapping")


def decode_queries(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    category_ids: tuple[int, ...],
    *,
    flipped: bool = False,
    view_id: int = 0,
    max_detections: int = 500,
) -> list[dict]:
    """Keep every foreground query score up to the frozen 500-detection limit.

    RF-DETR's unused background logit is ignored. Sparse COCO category IDs are
    mapped explicitly in training order. Coordinates remain unconstrained, just
    like the official postprocessor; clipping would change COCO box overlap.
    """
    _categories(category_ids)
    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 4
        or logits.ndim != 2
        or logits.shape[0] != len(boxes)
        or logits.shape[1] < len(category_ids)
    ):
        raise ValueError("Decoder boxes and logits have incompatible shapes")
    if len(boxes) > 1000 or not 1 <= max_detections <= 500:
        raise ValueError("Decoder output exceeds the bounded query budget")
    if not np.isfinite(boxes).all() or not np.isfinite(logits).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Decoder output must be finite with positive box dimensions")
    if type(view_id) is not int or view_id < 0:
        raise ValueError("View ID must be a nonnegative integer")
    foreground = logits[:, : len(category_ids)]
    # This stable sigmoid avoids overflow for extreme but finite numeric outputs.
    scores = np.exp(-np.logaddexp(0, -foreground))
    order = np.argsort(-scores.ravel(), kind="stable")[:max_detections]
    xyxy = np.concatenate(
        (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), axis=1
    )
    if flipped:
        xyxy[:, [0, 2]] = 1 - xyxy[:, [2, 0]]
    results = []
    for flat in order:
        query, label = divmod(int(flat), len(category_ids))
        results.append(
            {
                "box": xyxy[query].tolist(),
                "score": float(scores[query, label]),
                "category_id": category_ids[label],
                "view_id": view_id,
                "query_id": query,
            }
        )
    return results


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    overlap = np.maximum(
        0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])
    ).prod(axis=1)
    union = (box[2:] - box[:2]).prod() + (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - overlap
    return overlap / np.maximum(union, np.finfo(np.float64).tiny)


def weighted_box_fusion(
    views: list[list[dict]], *, iou_threshold: float = 0.55, max_detections: int = 500
) -> list[dict]:
    """Confidence-weight coordinates across views without same-view suppression.

    Each cluster accepts at most one query from a given view, keeping adjacent
    objects (even overlapping boxes) distinct. Cross-view matching is greedy in
    stable confidence order. Confidence averages over *all* requested views, so
    missing-view support has an explicit zero contribution. Every output retains
    its contributing view/query IDs; fusion has no ground-truth access.
    """
    if not 1 <= len(views) <= 8 or not 0 < iou_threshold <= 1:
        raise ValueError("Fusion requires 1..8 views and IoU in (0, 1]")
    if not 1 <= max_detections <= 500 or any(len(view) > 500 for view in views):
        raise ValueError("At most 500 detections per view/output are permitted")
    candidates = []
    for view_id, view in enumerate(views):
        seen = set()
        for row in view:
            box = np.asarray(row["box"], dtype=np.float64)
            score, category, query = row["score"], row["category_id"], row["query_id"]
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or (box[2:] <= box[:2]).any()
                or not math.isfinite(score)
                or not 0 <= score <= 1
                or type(category) is not int
                or not 0 <= category < 2**63
                or type(query) is not int
                or query < 0
                or row["view_id"] != view_id
                or (query, category) in seen
            ):
                raise ValueError("Invalid or duplicate query provenance in fusion input")
            seen.add((query, category))
            candidates.append((float(score), view_id, query, category, box))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    if len(views) == 1:
        # A native pass needs neither quadratic matching nor suppression. Keep
        # exact coordinates and confidences, including overlapping DETR queries.
        result = [
            {
                "box": box.tolist(),
                "score": score,
                "category_id": category,
                "contributors": [[view, query]],
            }
            for score, view, query, category, box in candidates
        ]
        return sorted(result, key=lambda r: (-r["score"], r["category_id"], r["contributors"]))[
            :max_detections
        ]
    # Fixed-size numeric arrays avoid repeatedly walking Python cluster objects
    # and materializing their boxes for every query in a multi-view pass.
    capacity = len(candidates)
    cluster_boxes = np.empty((capacity, 4), dtype=np.float64)
    cluster_categories = np.empty(capacity, dtype=np.int64)
    cluster_views = np.zeros(capacity, dtype=np.uint16)
    cluster_scores = np.zeros(capacity, dtype=np.float64)
    cluster_weights = np.zeros(capacity, dtype=np.float64)
    contributors: list[list[list[int]]] = []
    count = 0
    for score, view_id, query, category, box in candidates:
        view_bit = 1 << view_id
        eligible = np.flatnonzero(
            (cluster_categories[:count] == category) & ((cluster_views[:count] & view_bit) == 0)
        )
        match = None
        if len(eligible):
            overlaps = _iou(box, cluster_boxes[eligible])
            best = int(overlaps.argmax())
            if overlaps[best] >= iou_threshold:
                match = int(eligible[best])
        weight = max(score, np.finfo(np.float64).eps)
        if match is None:
            cluster_boxes[count] = box
            cluster_categories[count] = category
            cluster_views[count] = view_bit
            cluster_scores[count] = score
            cluster_weights[count] = weight
            contributors.append([[view_id, query]])
            count += 1
        else:
            cluster_boxes[match] = (
                cluster_boxes[match] * cluster_weights[match] + box * weight
            ) / (cluster_weights[match] + weight)
            cluster_weights[match] += weight
            cluster_scores[match] += score
            cluster_views[match] |= view_bit
            contributors[match].append([view_id, query])
    results = [
        {
            "box": cluster_boxes[i].tolist(),
            "score": float(cluster_scores[i] / len(views)),
            "category_id": int(cluster_categories[i]),
            "contributors": sorted(contributors[i]),
        }
        for i in range(count)
    ]
    return sorted(results, key=lambda r: (-r["score"], r["category_id"], r["contributors"]))[
        :max_detections
    ]


def _synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _memory(device: str) -> dict[str, int]:
    import resource

    import torch

    # ru_maxrss is a process high-water mark, not an attributable model allocation.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "process_rss_high_water_bytes": int(rss if platform.system() == "Darwin" else rss * 1024)
    }
    if device == "mps":
        result.update(
            mps_allocated_bytes=torch.mps.current_allocated_memory(),
            mps_driver_bytes=torch.mps.driver_allocated_memory(),
        )
    elif device == "cuda":
        result.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    return result


def _forward(model, batch, *, capture_embeddings: bool):
    if not capture_embeddings:
        return model(batch), None
    captured = []

    def capture(_module, inputs, _output):
        captured.append(inputs[0][-1].detach())

    hook = model.class_embed.register_forward_hook(capture)
    try:
        raw = model(batch)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape[:2] != raw["pred_logits"].shape[:2]:
        raise RuntimeError("RF-DETR decoder embedding contract changed")
    return raw, captured[0]


def _refine_view(
    raw,
    embeddings,
    images,
    checkpoint: Path,
    detector_sha256: str,
    category_id: int,
    *,
    flipped: bool,
    view_id: int,
) -> list[list[dict]]:
    """Refine in the observed view's coordinates, then invert the augmentation."""
    import torch
    from torchvision.ops import box_convert

    from vision_rl.detection_rl import predict_refinement_safetensors

    logits = raw["pred_logits"][..., 0].detach().cpu()
    ordering = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :128]
    boxes = box_convert(raw["pred_boxes"].detach().cpu(), in_fmt="cxcywh", out_fmt="xyxy")
    features = embeddings.detach().cpu()
    cache = {
        "version": 1,
        "split": "inference",
        "ids": [str(r["id"]) for r in images],
        "boxes": torch.gather(boxes, 1, ordering[..., None].expand(-1, -1, 4)),
        "logits": torch.gather(logits, 1, ordering),
        "embeddings": torch.gather(
            features, 1, ordering[..., None].expand(-1, -1, features.shape[-1])
        ),
        "image_sizes": torch.tensor([[r["width"], r["height"]] for r in images]),
        "metadata": {"model_sha256": detector_sha256},
    }
    refined = predict_refinement_safetensors(cache, checkpoint, device="cpu")
    results = []
    if len(refined) != len(images):
        raise RuntimeError("Refiner returned a different number of images")
    for index, prediction in enumerate(refined):
        if prediction["image_id"] != cache["ids"][index]:
            raise RuntimeError("Refiner changed image identity")
        xyxy = prediction["boxes"].numpy().astype(np.float64)
        if flipped:
            xyxy[:, [0, 2]] = 1 - xyxy[:, [2, 0]]
        if len(xyxy) != len(ordering[index]) or len(prediction["scores"]) != len(xyxy):
            raise RuntimeError("Refiner changed the fixed query contract")
        results.append(
            [
                {
                    "box": box.tolist(),
                    "score": float(score),
                    "category_id": category_id,
                    "view_id": view_id,
                    "query_id": int(query),
                }
                for box, score, query in zip(
                    xyxy, prediction["scores"], ordering[index], strict=True
                )
            ]
        )
    return results


def infer_images(
    detector,
    images: list[dict],
    category_ids: tuple[int, ...],
    config: InferenceConfig,
    *,
    refinement: Path | None = None,
    detector_sha256: str | None = None,
) -> tuple[list[dict], dict]:
    """Run bounded batches, returning COCO boxes plus auditable view provenance."""
    import torch
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64

    config.validate()
    _categories(category_ids)
    if refinement is not None:
        if len(category_ids) != 1:
            raise ValueError("The trusted refiner supports exactly one foreground category")
        if (
            not isinstance(detector_sha256, str)
            or len(detector_sha256) != 64
            or any(c not in "0123456789abcdef" for c in detector_sha256)
        ):
            raise ValueError("Refinement requires the exact frozen detector SHA256")
    if not images:
        raise ValueError("At least one image is required")
    torch.set_num_threads(config.cpu_threads)
    model = detector.model.model.eval()
    if config.checkpoint_resolution is not None and (
        detector.model_config.resolution != config.checkpoint_resolution
    ):
        raise ValueError("Loaded checkpoint resolution differs from the frozen inference profile")
    block_size = detector.model_config.patch_size * detector.model_config.num_windows
    if any(resolution % block_size for resolution in config.resolutions):
        raise ValueError("Resolution is incompatible with the trusted backbone window size")
    transforms = {r: make_coco_transforms_square_div_64("val", r) for r in config.resolutions}
    records, batch_times, preprocessing, fusion, forward = [], [], 0.0, 0.0, 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), config.batch_size):
            batch_images = images[start : start + config.batch_size]
            batch_started = time.perf_counter()
            per_image: list[list[list[dict]]] = [[] for _ in batch_images]
            decode_started = time.perf_counter()
            decoded = []
            for row in batch_images:
                with Image.open(row["path"]) as source:
                    if source.size != (row["width"], row["height"]):
                        raise ValueError("Image dimensions differ from the frozen manifest")
                    decoded.append(source.convert("RGB"))
            preprocessing += time.perf_counter() - decode_started
            previous_resolution, normalized_batch = None, None
            for view_id, (resolution, flip) in enumerate(config.views):
                prep_started = time.perf_counter()
                if resolution != previous_resolution:
                    normalized_batch = torch.stack(
                        [transforms[resolution](im, None)[0] for im in decoded]
                    ).to(config.device)
                    previous_resolution = resolution
                # Adjacent flip views share decode, resize, normalization and
                # host-device transfer. Only the inexpensive tensor flip repeats.
                batch = normalized_batch.flip(-1) if flip else normalized_batch
                _synchronize(config.device)
                preprocessing += time.perf_counter() - prep_started
                forward_started = time.perf_counter()
                raw, embeddings = _forward(model, batch, capture_embeddings=refinement is not None)
                _synchronize(config.device)
                forward += time.perf_counter() - forward_started
                post_started = time.perf_counter()
                if refinement is not None:
                    proposals = _refine_view(
                        raw,
                        embeddings,
                        batch_images,
                        refinement,
                        detector_sha256,
                        category_ids[0],
                        flipped=flip,
                        view_id=view_id,
                    )
                    for index, rows in enumerate(proposals):
                        per_image[index].append(rows)
                else:
                    boxes = raw["pred_boxes"].cpu().numpy()
                    logits = raw["pred_logits"].cpu().numpy()
                    for index in range(len(batch_images)):
                        per_image[index].append(
                            decode_queries(
                                boxes[index],
                                logits[index],
                                category_ids,
                                flipped=flip,
                                view_id=view_id,
                                max_detections=config.max_detections,
                            )
                        )
                fusion += time.perf_counter() - post_started
            fusion_started = time.perf_counter()
            for row, views in zip(batch_images, per_image, strict=True):
                for detection in weighted_box_fusion(
                    views, iou_threshold=config.fusion_iou, max_detections=config.max_detections
                ):
                    x1, y1, x2, y2 = detection.pop("box")
                    records.append(
                        {
                            "image_id": row["id"],
                            **detection,
                            "bbox": [
                                x1 * row["width"],
                                y1 * row["height"],
                                (x2 - x1) * row["width"],
                                (y2 - y1) * row["height"],
                            ],
                        }
                    )
            fusion += time.perf_counter() - fusion_started
            batch_times.append(time.perf_counter() - batch_started)
    elapsed = time.perf_counter() - started
    timing = {
        "elapsed_seconds": elapsed,
        "images": len(images),
        "images_per_second": len(images) / elapsed,
        "batch_seconds": batch_times,
        "preprocessing_seconds": preprocessing,
        "model_forward_seconds": forward,
        "postprocessing_seconds": fusion,
        "memory_at_end": _memory(config.device),
        "prediction_sha256": _digest(records),
    }
    return records, timing


def prediction_difference(reference: list[dict], candidate: list[dict]) -> dict:
    """Measure numerical changes between batch sizes without silently asserting parity."""

    def indexed(rows):
        return {
            (r["image_id"], r["category_id"], tuple(map(tuple, r["contributors"]))): r for r in rows
        }

    left, right = indexed(reference), indexed(candidate)
    common = left.keys() & right.keys()
    return {
        "exact_prediction_match": reference == candidate,
        "matched_detection_identities": len(common),
        "missing_detection_identities": len(left.keys() - right.keys()),
        "added_detection_identities": len(right.keys() - left.keys()),
        "maximum_score_difference": max(
            (abs(left[k]["score"] - right[k]["score"]) for k in common), default=None
        ),
        "maximum_bbox_pixel_difference": max(
            (float(np.max(np.abs(np.asarray(left[k]["bbox"]) - right[k]["bbox"]))) for k in common),
            default=None,
        ),
    }


def benchmark_detector(
    detector,
    images: list[dict],
    category_ids: tuple[int, ...],
    config: InferenceConfig,
    *,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    warmup: int = 2,
    repeats: int = 5,
    refinement: Path | None = None,
    detector_sha256: str | None = None,
) -> dict:
    """Measure warm and end-to-end execution; include every run and prediction delta."""
    if not batch_sizes or any(type(b) is not int or not 1 <= b <= 64 for b in batch_sizes):
        raise ValueError("Benchmark batch sizes must be in 1..64")
    if not 1 <= warmup <= 20 or not 2 <= repeats <= 100:
        raise ValueError("Benchmark requires 1..20 warmups and 2..100 measured repetitions")
    if len(images) < max(batch_sizes):
        raise ValueError("Benchmark requires at least as many images as the largest batch")
    baseline, results = None, []
    refinement_arguments = (
        {"refinement": refinement, "detector_sha256": detector_sha256}
        if refinement is not None
        else {}
    )
    for batch_size in batch_sizes:
        run_config = replace(config, batch_size=batch_size)
        warmup_start = time.perf_counter()
        for _ in range(warmup):
            infer_images(
                detector, images[:batch_size], category_ids, run_config, **refinement_arguments
            )
        warmup_elapsed = time.perf_counter() - warmup_start
        runs, differences = [], []
        for _ in range(repeats):
            predictions, timing = infer_images(
                detector, images, category_ids, run_config, **refinement_arguments
            )
            if baseline is None:
                baseline = predictions
            differences.append(prediction_difference(baseline, predictions))
            runs.append(timing)
        batch_seconds = [s for run in runs for s in run["batch_seconds"]]
        results.append(
            {
                "batch_size": batch_size,
                "warmup_runs": warmup,
                "warmup_elapsed_seconds": warmup_elapsed,
                "measured_runs": runs,
                "prediction_differences": differences,
                "median_batch_latency_ms": float(np.median(batch_seconds) * 1000),
                "p95_batch_latency_ms": float(np.percentile(batch_seconds, 95) * 1000),
                "pooled_images_per_second": sum(run["images"] for run in runs)
                / sum(run["elapsed_seconds"] for run in runs),
            }
        )
    return {
        "format_version": 1,
        "configuration": asdict(config),
        "measurements": results,
        "timing_scope": "full RGB decode, validation preprocessing, transfer, synchronized "
        "eager forward, CPU query decoding and fusion; model loading excluded",
        "memory_scope": "process lifetime RSS high-water and sampled device memory; "
        "MPS samples are not peak allocations",
        "compiler": "eager; no unverified compiler speedup claimed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "benchmark"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--refinement",
        type=Path,
        help="Optional numeric-only refiner bound to this detector; top128 queries",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-ids", type=int, nargs="+", required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[384])
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--fusion-iou", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--architecture", choices=("nano", "small"), default="nano")
    parser.add_argument(
        "--checkpoint-resolution",
        type=int,
        choices=(384, 512),
        help="Trusted model positional grid; distinct from TTA input resolutions",
    )
    parser.add_argument("--benchmark-batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    config = InferenceConfig(
        resolutions=tuple(args.resolutions),
        horizontal_flip=args.horizontal_flip,
        fusion_iou=args.fusion_iou,
        batch_size=args.batch_size,
        cpu_threads=args.cpu_threads,
        device=args.device,
        architecture=args.architecture,
        checkpoint_resolution=args.checkpoint_resolution,
    )
    config.validate()
    category_ids = tuple(args.category_ids)
    _categories(category_ids)
    if args.output.exists():
        raise ValueError("Inference evidence is immutable; choose a fresh output directory")
    images = load_image_manifest(args.manifest, args.image_root)
    import torch

    from vision_rl.detection_training import load_detector_safetensors

    torch.set_num_threads(config.cpu_threads)
    load_started = time.perf_counter()
    detector = load_detector_safetensors(
        args.checkpoint,
        architecture=config.architecture,
        num_classes=len(category_ids),
        device=config.device,
        **(
            {"resolution": config.checkpoint_resolution}
            if config.checkpoint_resolution is not None
            else {}
        ),
    )
    load_seconds = time.perf_counter() - load_started
    detector_sha256 = _hash_file(args.checkpoint)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": detector_sha256,
        "refinement_sha256": _hash_file(args.refinement) if args.refinement else None,
        "query_policy": "top128 per view, then refinement before flip inversion"
        if args.refinement
        else "all foreground query-class scores, at most500",
        "manifest_sha256": _hash_file(args.manifest),
        "images": [{"id": r["id"], "sha256": _hash_file(r["path"])} for r in images],
        "category_ids_in_training_order": category_ids,
        "configuration": asdict(config),
        "views": [
            {"id": i, "resolution": r, "horizontal_flip": f}
            for i, (r, f) in enumerate(config.views)
        ],
        "host": platform.platform(),
        "model_parameters": sum(p.numel() for p in detector.model.model.parameters()),
        "model_dtype": str(next(detector.model.model.parameters()).dtype),
        "model_load_seconds": load_seconds,
        "software": {p: version(p) for p in ("torch", "rfdetr", "numpy", "Pillow")},
        "label_access": False,
        "preprocessing": "RF-DETR val square resize; "
        "full RGB decode, original pixel coordinate system; no confidence cutoff",
    }
    args.output.mkdir(parents=True)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.command == "predict":
        records, result = infer_images(
            detector,
            images,
            category_ids,
            config,
            refinement=args.refinement,
            detector_sha256=detector_sha256,
        )
        predictions = [{k: v for k, v in row.items() if k != "contributors"} for row in records]
        (args.output / "predictions.coco.json").write_text(json.dumps(predictions) + "\n")
        (args.output / "fusion-provenance.json").write_text(json.dumps(records) + "\n")
    else:
        result = benchmark_detector(
            detector,
            images,
            category_ids,
            config,
            batch_sizes=tuple(args.benchmark_batch_sizes),
            warmup=args.warmup,
            repeats=args.repeats,
            refinement=args.refinement,
            detector_sha256=detector_sha256,
        )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
