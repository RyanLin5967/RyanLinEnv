"""Prepare, execute, and score reproducible native RF100-VL detection baselines.

Discovery samples validation only. A test run requires a frozen experiment record;
it never searches prompts or changes model settings after observing test scores.
"""

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from vision_rl.data import write_json

BENCHMARK = "RF100-VL/countingpills-exf0r-gfkt/v2"
ROUTER_COORDINATES = {"qwen": "xyxy_1000", "kimi": "named_normalized"}


def router_protocol_fields(provider):
    return {
        "coordinate_format": ROUTER_COORDINATES[provider], "privacy_level": "deny",
        "quantizations": ["fp8", "bf16", "fp32"] if provider == "qwen" else None,
    }


def router_prompt(categories, provider, rich=True):
    if provider == "qwen":
        return gemini_structured_prompt(categories, rich).replace(
            "ymin,xmin,ymax,xmax", "xmin,ymin,xmax,ymax"
        )
    labels = [item["name"] for item in categories]
    detail = (
        " Detect every visible individual pill/tablet. Include its full visible extent; "
        "for partly obscured pills estimate the full shape from visible portions. "
        "Exclude shadows, reflections, and objects too indistinct to identify as pills."
        if rich and labels == ["pill"] else ""
    )
    return (
        f"Locate every object belonging to these classes: {json.dumps(labels)}.{detail} "
        'Return JSON {"detections":[{"x_min":0.1,"y_min":0.2,"x_max":0.3,"y_max":0.4,'
        '"class_name":"class name","confidence":0.95}]}. '
        "Coordinates must be normalized to 0..1 relative to the entire image: "
        "x_min/y_min are top-left and x_max/y_max are bottom-right. "
        "Confidence is from 0 to 1. Use only the listed labels. "
        'Return {"detections":[]} if no objects are present.'
    )


def gemini_response_schema(categories):
    """Pinned Roboflow Gemini v6 native detection schema (no confidence field)."""
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "box_2d": {
                    "type": "ARRAY",
                    "items": {"type": "INTEGER", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "label": {"type": "STRING", "enum": [c["name"] for c in categories]},
            },
            "required": ["box_2d", "label"],
            "propertyOrdering": ["box_2d", "label"],
        },
    }


def gemini_structured_prompt(categories, rich=True):
    """Keep the native normalized integer box contract aligned with the schema."""
    from vision_rl.detection_eval import detection_prompt

    prompt = detection_prompt(categories, rich=rich)
    return prompt.replace(
        '{"label":"class name","box":[ymin,xmin,ymax,xmax],"score":0.95}.',
        '{"box_2d":[ymin,xmin,ymax,xmax],"label":"class name"}.',
    ).replace(
        "Coordinates must be normalized to 0..1000",
        "Coordinates must be integers normalized to 0..1000",
    ).replace(" Score is your confidence from 0 to 1.", "")


def astra_response_format(categories):
    """Roboflow's native Astra detection shape, with exact allowed class labels."""
    return {
        "format": {
            "type": "json_schema",
            "name": "detections",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": [c["name"] for c in categories],
                                },
                                "box_2d": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                            },
                            "required": ["label", "box_2d"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["detections"],
                "additionalProperties": False,
            },
        }
    }


def astra_prompt(categories, width, height, rich=True):
    labels = [item["name"] for item in categories]
    detail = ""
    if rich and [label.casefold() for label in labels] == ["pill"]:
        detail = (
            " Detect every visible individual pill/tablet. Include its full visible extent; "
            "for partly obscured pills estimate the full shape from visible portions. "
            "Exclude shadows, reflections, and objects too indistinct to identify as pills."
        )
    return (
        "Detect all objects in this image. Output JSON with the key detections holding a list "
        "of objects with label and box_2d. Each box_2d must be [x_min,y_min,x_max,y_max]: "
        f"top-left and bottom-right corners in absolute pixel coordinates of this {width}x{height} "
        f"pixel image. Only use these labels: {json.dumps(labels)}.{detail} "
        'Return {"detections":[]} if no objects are present.'
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def image_path(directory, item):
    from PIL import Image

    original = directory / item["file_name"]
    path = original.resolve()
    if original.is_symlink() or not path.is_relative_to(directory.resolve()):
        raise ValueError("Image path must remain inside its native split")
    with Image.open(path) as image:
        if image.size != (item["width"], item["height"]):
            raise ValueError("Image dimensions disagree with COCO annotations")
    return path


def build_protocol(args):
    """Describe fixed image bytes and prompts without running inference."""
    from vision_rl.detection_eval import detection_prompt, load_coco
    from vision_rl.frontier import PROVIDERS

    source = args.source.resolve()
    split = "valid" if args.split == "validation" else args.split
    annotation = source / split / "_annotations.coco.json"
    dataset = load_coco(annotation)
    provider = getattr(args, "provider", "gemini")
    if provider not in PROVIDERS:
        raise ValueError("Unsupported protocol provider")
    detail = getattr(args, "image_detail", None)
    structured = getattr(args, "structured_output", False)
    if type(structured) is not bool or (structured and provider != "gemini"):
        raise ValueError("structured_output is an explicit Gemini-only boolean setting")
    if provider != "openai" and detail is not None:
        raise ValueError("image_detail is only configurable for the OpenAI protocol")
    if provider == "openai" and detail not in {None, "low", "high", "original", "auto"}:
        raise ValueError("Unsupported OpenAI image_detail")
    if args.count is not None and not 1 <= args.count <= len(dataset["images"]):
        raise ValueError("Invalid discovery sample size")
    ordered = sorted(
        dataset["images"],
        key=lambda row: hashlib.sha256(f"{args.seed}:{row['file_name']}".encode()).hexdigest(),
    )
    selected = ordered[: args.count] if args.count else ordered
    prompt = detection_prompt(dataset["categories"], rich=args.prompt == "rich")
    if structured:
        prompt = gemini_structured_prompt(dataset["categories"], rich=args.prompt == "rich")
    if provider == "openai":
        prompt = astra_prompt(dataset["categories"], "{width}", "{height}", args.prompt == "rich")
    elif provider in ROUTER_COORDINATES:
        prompt = router_prompt(dataset["categories"], provider, args.prompt == "rich")

    def item_prompt(item):
        if provider == "openai":
            return astra_prompt(
                dataset["categories"], item["width"], item["height"], args.prompt == "rich"
            )
        return prompt

    examples = []
    if args.examples:
        train_file = source / "train/_annotations.coco.json"
        train = load_coco(train_file)
        if args.examples > len(train["images"]):
            raise ValueError("Requested more few-shot examples than native training images")
        candidates = sorted(
            train["images"],
            key=lambda row: hashlib.sha256(
                f"examples:{args.seed}:{row['file_name']}".encode()
            ).hexdigest(),
        )[: args.examples]
        names = {item["id"]: item["name"] for item in train["categories"]}
        for item in candidates:
            boxes = []
            for ann in train["annotations"]:
                if ann["image_id"] != item["id"]:
                    continue
                x, y, w, h = ann["bbox"]
                if provider == "kimi":
                    boxes.append({
                        "x_min": round(x / item["width"], 6),
                        "y_min": round(y / item["height"], 6),
                        "x_max": round((x + w) / item["width"], 6),
                        "y_max": round((y + h) / item["height"], 6),
                        "class_name": names[ann["category_id"]], "confidence": 1.0,
                    })
                    continue
                coordinates = (
                    [round(x), round(y), round(x + w), round(y + h)]
                    if provider == "openai"
                    else [
                        round(1000 * y / item["height"], 2),
                        round(1000 * x / item["width"], 2),
                        round(1000 * (y + h) / item["height"], 2),
                        round(1000 * (x + w) / item["width"], 2),
                    ]
                )
                if structured:
                    coordinates = [round(value) for value in coordinates]
                elif provider == "qwen":
                    coordinates = [round(coordinates[i]) for i in (1, 0, 3, 2)]
                boxes.append(
                    {"box_2d": coordinates, "label": names[ann["category_id"]]}
                    if structured
                    else {"label": names[ann["category_id"]], "box_2d": coordinates}
                )
            path = image_path(source / "train", item)
            examples.append(
                {
                    "id": f"train:{item['id']}",
                    "split": "train",
                    "image": str(path),
                    "prompt": item_prompt(item),
                    "response": json.dumps(
                        {"detections": boxes} if provider in {"openai", "kimi"} else boxes
                    ),
                    "file_sha256": sha(path),
                }
            )
    rows = []
    for item in selected:
        image = image_path(source / split, item)
        rows.append(
            {
                "id": str(item["id"]),
                "image": str(image),
                "prompt": item_prompt(item),
                "file_sha256": sha(image),
                "width": item["width"],
                "height": item["height"],
            }
        )
    protocol = {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "split": args.split,
        "annotation": str(annotation),
        "annotation_sha256": sha(annotation),
        "sampling_seed": args.seed,
        "selection": "all native images" if not args.count else "seeded filename-hash sample",
        "prompt_mode": args.prompt,
        "prompt": prompt,
        "examples": examples,
        "rows": rows,
        "frozen_experiment": None,
    }
    if provider == "openai":
        protocol.update(
            schema_version=2,
            provider=provider,
            coordinate_format="xyxy_pixels",
            image_detail=detail or "original",
            response_format=astra_response_format(dataset["categories"]),
        )
    elif structured:
        protocol.update(
            schema_version=2,
            provider=provider,
            coordinate_format="yxyx_1000",
            structured_output=True,
            response_schema=gemini_response_schema(dataset["categories"]),
        )
    elif provider in ROUTER_COORDINATES:
        protocol.update(schema_version=2, provider=provider, **router_protocol_fields(provider))
    return protocol


def protocol_identity(protocol):
    """Pin task contents and ordering while allowing an identical dataset relocation."""
    result = {
        key: value
        for key, value in protocol.items()
        if key not in {"annotation", "frozen_experiment", "rows", "examples"}
    }
    for name in ("rows", "examples"):
        result[name] = [
            {key: value for key, value in row.items() if key != "image"} for row in protocol[name]
        ]
    return result


def baseline_config(config):
    from vision_rl.frontier import DEFAULT_EFFORTS, PROVIDERS, SUPPORTED_EFFORTS

    allowed = {
        "provider",
        "model",
        "effort",
        "max_output_tokens",
        "prompt",
        "seed",
        "examples",
        "concurrency",
        "max_transient_retries",
        "retry_delay_seconds",
        "image_detail",
        "coordinate_format",
        "structured_output",
        "privacy_level",
        "quantizations",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown frozen baseline configuration field")
    provider = config.get("provider")
    if provider not in PROVIDERS or config.get("model") != PROVIDERS[provider][0]:
        raise ValueError("Frozen baseline must name a supported provider and exact model")
    structured = config.get("structured_output", False)
    if type(structured) is not bool or (structured and provider != "gemini"):
        raise ValueError("structured_output is an explicit Gemini-only boolean setting")
    result = {
        "effort": DEFAULT_EFFORTS[provider],
        "max_output_tokens": 16384,
        "prompt": "rich",
        "seed": 1729,
        "examples": 0,
        "concurrency": 2,
        "max_transient_retries": 0,
        "retry_delay_seconds": 2.0,
        **config,
    }
    if result["effort"] not in SUPPORTED_EFFORTS[provider]:
        raise ValueError("Frozen baseline has unsupported effort")
    if provider == "openai":
        result.setdefault("image_detail", "original")
        result.setdefault("coordinate_format", "xyxy_pixels")
        if result["image_detail"] not in {"low", "high", "original", "auto"}:
            raise ValueError("Frozen baseline has unsupported image_detail")
        if result["coordinate_format"] != "xyxy_pixels":
            raise ValueError("Astra requires the explicit xyxy_pixels coordinate protocol")
    elif provider in ROUTER_COORDINATES:
        for key, value in router_protocol_fields(provider).items():
            result.setdefault(key, value)
            if result[key] != value:
                raise ValueError("Frozen OpenRouter routing differs from declared defaults")
        if "image_detail" in config or result["max_output_tokens"] > 16384:
            raise ValueError("Unsupported OpenRouter detail or token ceiling")
    elif "image_detail" in config or (
        "coordinate_format" in config
        and not (structured and config["coordinate_format"] == "yxyx_1000")
    ):
        raise ValueError("OpenAI-specific protocol fields cannot configure another provider")
    if provider not in ROUTER_COORDINATES and any(
        key in config for key in ("privacy_level", "quantizations")
    ):
        raise ValueError("OpenRouter routing cannot configure another provider")
    if structured:
        result["coordinate_format"] = "yxyx_1000"
    else:
        result.pop("structured_output", None)
    for name, low, high in (
        ("max_output_tokens", 16, 128000),
        ("examples", 0, 4),
        ("concurrency", 1, 2),
    ):
        if (
            isinstance(result[name], bool)
            or not isinstance(result[name], int)
            or not (low <= result[name] <= high)
        ):
            raise ValueError(f"Invalid frozen baseline {name}")
    if result["prompt"] not in {"minimal", "rich"} or type(result["seed"]) is not int:
        raise ValueError("Invalid frozen prompt mode or sampling seed")
    result.update(retry_policy(result["max_transient_retries"], result["retry_delay_seconds"]))
    return result


def retry_policy(retries, delay):
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("max_transient_retries must be an integer between 0 and 2")
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 60:
        raise ValueError("retry_delay_seconds must be a finite number between 0 and 60")
    return {"max_transient_retries": retries, "retry_delay_seconds": float(delay)}


def create_freeze(source, checkpoint, baselines, criteria, output, artifacts=()):
    """Freeze actual model bytes and every final baseline before touching test answers.

    Baselines are dictionaries accepted by ``baseline_config``. Additional model
    artifacts (for example an RL policy JSON) are frozen alongside the checkpoint.
    The criteria are recorded here; a separate comparison evaluates those criteria.
    """
    from vision_rl import detection_eval, frontier

    source, checkpoint, output = Path(source).resolve(), Path(checkpoint).resolve(), Path(output)
    if output.exists():
        raise ValueError("Freeze records are immutable; choose a new path")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("Freeze requires at least one explicitly configured baseline")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("Freeze requires nonempty success criteria")
    digest(criteria)  # Reject non-finite JSON before writing anything.
    entries = []
    for raw in baselines:
        config = baseline_config(raw)
        protocol = build_protocol(
            SimpleNamespace(
                source=source,
                split="test",
                count=None,
                seed=config["seed"],
                prompt=config["prompt"],
                examples=config["examples"],
                provider=config["provider"],
                image_detail=config.get("image_detail"),
                structured_output=config.get("structured_output", False),
            )
        )
        identity = protocol_identity(protocol)
        entry = {"config": config, "protocol": identity, "protocol_sha256": digest(identity)}
        entry["id"] = digest({"config": config, "protocol_sha256": entry["protocol_sha256"]})
        entries.append(entry)
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate frozen baseline configuration")
    pinned = [checkpoint, *(Path(path).resolve() for path in artifacts)]
    if len(set(pinned)) != len(pinned) or any(not path.is_file() for path in pinned):
        raise ValueError("Frozen checkpoint/artifacts must be distinct existing files")
    record = {
        "schema_version": 2,
        "benchmark": BENCHMARK,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "artifacts": [{"path": str(path), "sha256": sha(path)} for path in pinned],
        "test_annotation_sha256": sha(source / "test/_annotations.coco.json"),
        "success_criteria": criteria,
        "baselines": entries,
        "implementation": [
            {"path": str(Path(path).resolve()), "sha256": sha(path)}
            for path in (__file__, detection_eval.__file__, frontier.__file__)
        ],
        "adapter_source_commit": frontier.SOURCE_COMMIT,
        "test_policy": (
            "One output journal per frozen baseline and one completed answer per image. "
            "Only declared HTTP 502/503/504 retries are allowed; unknown reservations never retry."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def check_freeze(path):
    frozen = json.loads(Path(path).read_text())
    if frozen.get("schema_version") != 2 or frozen.get("benchmark") != BENCHMARK:
        raise ValueError("Use a v2 freeze record generated by this wrapper's freeze command")
    if not frozen.get("baselines") or not frozen.get("success_criteria"):
        raise ValueError("Freeze record lacks baseline configurations or success criteria")
    if sha(frozen["checkpoint_path"]) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen specialist checkpoint changed")
    for item in [*frozen["artifacts"], *frozen["implementation"]]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Frozen specialist artifact or evaluator implementation changed")
    for entry in frozen["baselines"]:
        if digest(entry["protocol"]) != entry["protocol_sha256"] or entry["id"] != digest(
            {
                "config": entry["config"],
                "protocol_sha256": entry["protocol_sha256"],
            }
        ):
            raise ValueError("Frozen baseline identity is inconsistent")
    return frozen


def prepare(args):
    if args.output.exists():
        raise ValueError("Choose a new output directory; prepared experiments are immutable")
    if args.split == "test" and (args.count is not None or args.freeze is None):
        raise ValueError("Test requires --freeze and must include the entire native test split")
    if args.freeze and args.split != "test":
        raise ValueError("A final-test freeze cannot be attached to validation")
    protocol = build_protocol(args)
    if args.freeze:
        frozen = check_freeze(args.freeze)
        identity = digest(protocol_identity(protocol))
        if frozen["test_annotation_sha256"] != protocol["annotation_sha256"] or not any(
            item["protocol_sha256"] == identity for item in frozen["baselines"]
        ):
            raise ValueError(
                "Prepared test images, prompts, examples, or ordering differ from freeze"
            )
        protocol["frozen_experiment"] = {
            "path": str(args.freeze.resolve()),
            "sha256": sha(args.freeze),
        }
    args.output.mkdir(parents=True)
    write_json(args.output / "inputs.json", protocol)
    print(
        json.dumps(
            {
                "prepared": str(args.output),
                "split": args.split,
                "images": len(protocol["rows"]),
                "examples": len(protocol["examples"]),
            }
        )
    )


def complement_settings(protocol):
    """Fields that must remain identical when extending a validation sample."""
    return {
        key: value for key, value in protocol_identity(protocol).items()
        if key not in {"rows", "selection", "validation_complement"}
    }


def checked_discovery(input_path, run_path):
    """Bind an existing validation protocol to the actual original request identity."""
    from vision_rl import frontier

    protocol = json.loads(Path(input_path).read_text())
    if protocol.get("split") != "validation" or protocol.get("validation_complement"):
        raise ValueError("Complement source must be an original validation discovery protocol")
    verify_protocol(protocol)
    run = json.loads(Path(run_path).read_text())
    if run.get("identity_sha256") != frontier._digest(run["identity"]):
        raise ValueError("Discovery run identity checksum changed")
    for name in ("rows", "examples"):
        prepared = [
            frontier._public(frontier._prepare(row, example=name == "examples"))
            for row in protocol[name]
        ]
        if prepared != run["identity"][name]:
            raise ValueError("Discovery run images, prompts, or examples disagree with protocol")
    config = run["identity"]["config"]
    if (
        config.get("response_schema") != protocol.get("response_schema")
        or config.get("response_format") != protocol.get("response_format")
        or config.get("image_detail") != protocol.get("image_detail")
        or config.get("privacy_level") != protocol.get("privacy_level")
        or config.get("quantizations") != protocol.get("quantizations")
        or config["max_requests"] != len(protocol["rows"]) * (
            1 + config.get("max_transient_retries", 0)
        )
    ):
        raise ValueError("Discovery request settings disagree with prepared protocol")
    if protocol.get("provider") and config["provider"] != protocol["provider"]:
        raise ValueError("Discovery provider disagrees with protocol")
    return protocol, config


def prepare_complement(args):
    """Extend a fixed discovery protocol to all remaining native validation images."""
    if args.output.exists():
        raise ValueError("Choose a new output directory; prepared experiments are immutable")
    input_path = (args.input / "inputs.json").resolve()
    run_path = (args.from_run / "run.json").resolve()
    discovery, config = checked_discovery(input_path, run_path)
    full = build_protocol(
        SimpleNamespace(
            source=Path(discovery["annotation"]).parent.parent,
            split="validation", count=None, seed=discovery["sampling_seed"],
            prompt=discovery["prompt_mode"], examples=len(discovery["examples"]),
            provider=config["provider"], image_detail=discovery.get("image_detail"),
            structured_output=discovery.get("structured_output", False),
        )
    )
    if complement_settings(full) != complement_settings(discovery):
        raise ValueError("Reconstructed protocol differs from discovery settings or examples")
    expected = {row["id"]: row for row in full["rows"]}
    if any(row != expected.get(row["id"]) for row in discovery["rows"]):
        raise ValueError("Reconstructed discovery image prompts or hashes differ")
    excluded = [row["id"] for row in discovery["rows"]]
    full["rows"] = [row for row in full["rows"] if row["id"] not in set(excluded)]
    if not full["rows"]:
        raise ValueError("Discovery already includes every validation image")
    full["selection"] = "all native validation images outside pinned discovery sample"
    full["validation_complement"] = {
        "source_inputs": {"path": str(input_path), "sha256": sha(input_path)},
        "source_run": {"path": str(run_path), "sha256": sha(run_path)},
        "excluded_ids": excluded,
        "request_config": config,
        "scope": "Development confirmation only; never a final-test result or final win.",
    }
    verify_protocol(full)
    args.output.mkdir(parents=True)
    write_json(args.output / "inputs.json", full)
    print(json.dumps({
        "prepared": str(args.output), "split": "validation", "images": len(full["rows"]),
        "excluded": len(excluded),
    }))


def verify_complement(protocol, images):
    """Prevent changing settings, sampling a subset, or re-answering discovery images."""
    reference = protocol["validation_complement"]
    if protocol["split"] != "validation":
        raise ValueError("Complements are validation-only")
    for key in ("source_inputs", "source_run"):
        source = reference[key]
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Pinned complement discovery source changed")
    discovery, config = checked_discovery(
        reference["source_inputs"]["path"], reference["source_run"]["path"]
    )
    if reference["request_config"] != config:
        raise ValueError("Complement request settings differ from discovery run")
    if complement_settings(protocol) != complement_settings(discovery):
        raise ValueError("Complement protocol settings or examples differ from discovery")
    excluded = [row["id"] for row in discovery["rows"]]
    if reference["excluded_ids"] != excluded:
        raise ValueError("Complement excluded IDs differ from pinned discovery")
    expected = sorted(
        [item for identifier, item in images.items() if identifier not in set(excluded)],
        key=lambda row: hashlib.sha256(
            f"{protocol['sampling_seed']}:{row['file_name']}".encode()
        ).hexdigest(),
    )
    if [row["id"] for row in protocol["rows"]] != [str(item["id"]) for item in expected]:
        raise ValueError("Complement must contain every remaining validation image in fixed order")
    for row in protocol["rows"]:
        prompt = protocol["prompt"]
        if protocol.get("provider") == "openai":
            prompt = prompt.replace("{width}", str(row["width"])).replace(
                "{height}", str(row["height"])
            )
        if row["prompt"] != prompt:
            raise ValueError("Complement image prompt differs from pinned template")


def verify_protocol(protocol):
    from vision_rl.detection_eval import load_coco

    annotation = Path(protocol["annotation"])
    split_dir = "valid" if protocol["split"] == "validation" else "test"
    if protocol["split"] not in {"validation", "test"} or annotation.parent.name != split_dir:
        raise ValueError("Protocol split does not match its native annotation directory")
    if sha(annotation) != protocol["annotation_sha256"]:
        raise ValueError("Source annotations changed after experiment preparation")
    dataset = load_coco(annotation)
    if protocol.get("provider") == "openai":
        if (
            protocol.get("schema_version") != 2
            or protocol.get("coordinate_format") != "xyxy_pixels"
            or protocol.get("image_detail") not in {"low", "high", "original", "auto"}
            or protocol.get("response_format") != astra_response_format(dataset["categories"])
            or any(key in protocol for key in (
                "response_schema", "structured_output", "privacy_level", "quantizations"
            ))
        ):
            raise ValueError("Invalid explicit OpenAI coordinate/image/schema protocol")
    elif protocol.get("provider") == "gemini":
        if (
            protocol.get("schema_version") != 2
            or protocol.get("structured_output") is not True
            or protocol.get("coordinate_format") != "yxyx_1000"
            or protocol.get("response_schema") != gemini_response_schema(dataset["categories"])
            or any(key in protocol for key in (
                "image_detail", "response_format", "privacy_level", "quantizations"
            ))
        ):
            raise ValueError("Invalid explicit Gemini structured-output protocol")
    elif protocol.get("provider") in ROUTER_COORDINATES:
        if (
            protocol.get("schema_version") != 2
            or any(protocol.get(key) != value for key, value in router_protocol_fields(
                protocol["provider"]
            ).items())
            or any(key in protocol for key in (
                "image_detail", "response_format", "response_schema", "structured_output"
            ))
        ):
            raise ValueError("Invalid explicit OpenRouter detection/routing protocol")
    elif any(
        key in protocol
        for key in (
            "coordinate_format", "image_detail", "response_format", "response_schema",
            "structured_output",
            "privacy_level", "quantizations",
        )
    ):
        raise ValueError("Unexpected provider-specific protocol fields")
    images = {str(item["id"]): item for item in dataset["images"]}
    ids = [row["id"] for row in protocol["rows"]]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= images.keys():
        raise ValueError("Protocol has missing, duplicate, or unknown evaluation image IDs")
    if protocol["split"] == "test" and set(ids) != images.keys():
        raise ValueError("Final test must contain every native test image exactly once")
    for row in protocol["rows"]:
        image = images[row["id"]]
        expected = image_path(annotation.parent, image)
        if Path(row["image"]).resolve() != expected or (row["width"], row["height"]) != (
            image["width"],
            image["height"],
        ):
            raise ValueError("Protocol image identity disagrees with native annotations")
    for row in [*protocol["rows"], *protocol["examples"]]:
        if sha(row["image"]) != row["file_sha256"]:
            raise ValueError("Image changed after experiment preparation")
    if "validation_complement" in protocol:
        verify_complement(protocol, images)
    reference = protocol.get("frozen_experiment")
    if protocol["split"] == "test" and not reference:
        raise ValueError("Final test requires a verified freeze record")
    if reference:
        if protocol["split"] != "test" or sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Frozen experiment record changed or has the wrong split")
        frozen = check_freeze(reference["path"])
        if frozen["test_annotation_sha256"] != protocol["annotation_sha256"]:
            raise ValueError("Frozen test annotations differ from prepared protocol")
        return frozen
    return None


def frozen_baseline(frozen, protocol, args):
    identity = digest(protocol_identity(protocol))
    settings = {
        "provider": args.provider,
        "model": args.model,
        "effort": args.effort,
        "max_output_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "max_transient_retries": args.max_transient_retries,
        "retry_delay_seconds": args.retry_delay_seconds,
    }
    if args.provider == "openai":
        settings.update(image_detail=protocol["image_detail"], coordinate_format="xyxy_pixels")
    elif protocol.get("structured_output"):
        settings.update(structured_output=True, coordinate_format="yxyx_1000")
    elif args.provider in ROUTER_COORDINATES:
        settings.update(router_protocol_fields(args.provider))
    matches = [
        entry
        for entry in frozen["baselines"]
        if entry["protocol_sha256"] == identity
        and all(entry["config"][key] == value for key, value in settings.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            "Final baseline model, effort, tokens, concurrency, retry policy, "
            "or protocol is not frozen"
        )
    return matches[0]


def reserve_final_output(reference, baseline, output, resume):
    """Bind a frozen baseline to one journal; a different output cannot repeat it."""
    freeze_path = Path(reference["path"])
    journal = freeze_path.with_name(freeze_path.name + ".runs.json")
    output = str(Path(output).resolve())
    with freeze_path.with_name(freeze_path.name + ".runs.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = (
            json.loads(journal.read_text())
            if journal.exists()
            else {
                "freeze_sha256": reference["sha256"],
                "baselines": {},
            }
        )
        if state["freeze_sha256"] != reference["sha256"]:
            raise ValueError("Final-run reservation belongs to a different freeze record")
        existing = state["baselines"].get(baseline["id"])
        if existing:
            if existing["output"] != output or not resume:
                raise ValueError("Final baseline already reserved; only resume its original output")
            return
        if resume:
            raise ValueError("Cannot resume an unreserved final baseline")
        if Path(output).exists():
            raise ValueError("New final output must not already exist")
        state["baselines"][baseline["id"]] = {
            "output": output,
            "reserved_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": baseline["protocol_sha256"],
        }
        temporary = journal.with_suffix(journal.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(state, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, journal)
        directory_fd = os.open(journal.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def score_predictions(protocol, predictions, model):
    """Rescore existing answers offline using the frozen provider coordinate contract.

    Returns the normal COCO report, including request/parse failures and a
    ``detections`` list. No hosted request or answer selection is performed.
    """
    from vision_rl.detection_eval import evaluate_coco, load_coco, parse_detection_response

    verify_protocol(protocol)
    annotation = Path(protocol["annotation"])
    dataset = load_coco(annotation)
    images = {str(item["id"]): item for item in dataset["images"]}
    coordinate_format = protocol.get("coordinate_format", "yxyx_1000")
    provider = model.get("provider")
    if (provider == "openai") != (protocol.get("provider") == "openai") or (
        protocol.get("provider") and protocol["provider"] != provider
    ):
        raise ValueError("Model provider does not match the prepared coordinate protocol")
    boxes, failures = [], []
    for row in protocol["rows"]:
        prediction = predictions.get(row["id"], {"error": "missing prediction"})
        if prediction.get("error"):
            failures.append({"id": row["id"], "error": prediction["error"]})
            continue
        try:
            boxes.extend(
                parse_detection_response(
                    prediction["response"],
                    images[row["id"]],
                    dataset["categories"],
                    coordinate_format=coordinate_format,
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            failures.append({"id": row["id"], "error": type(exc).__name__})
    report = evaluate_coco(
        annotation,
        boxes,
        image_ids=[int(row["id"]) for row in protocol["rows"]],
        model=model,
    )
    report.update(request_or_parse_failures=failures, errors=len(failures), detections=boxes)
    return report


def run(args):
    from vision_rl.frontier import run_frontier

    policy = retry_policy(
        getattr(args, "max_transient_retries", 0), getattr(args, "retry_delay_seconds", 2.0)
    )
    args.max_transient_retries = policy["max_transient_retries"]
    args.retry_delay_seconds = policy["retry_delay_seconds"]
    protocol_path = args.input / "inputs.json"
    protocol_sha256 = sha(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    frozen = verify_protocol(protocol)
    if (args.provider == "openai") != (protocol.get("provider") == "openai"):
        raise ValueError("Model provider does not match protocol; prepare with --provider openai")
    if protocol.get("provider") and args.provider != protocol["provider"]:
        raise ValueError("Model provider does not match the prepared provider-specific protocol")
    if getattr(args, "image_detail", None) is not None and args.image_detail != protocol.get(
        "image_detail"
    ):
        raise ValueError("Image detail differs from prepared protocol")
    complement_config = protocol.get("validation_complement", {}).get("request_config")
    if complement_config:
        actual = {
            "provider": args.provider, "model": args.model, "effort": args.effort,
            "max_output_tokens": args.max_tokens, "max_inflight": args.concurrency,
            "max_transient_retries": args.max_transient_retries,
            "retry_delay_seconds": args.retry_delay_seconds,
        }
        if any(actual[key] != complement_config.get(key) for key in actual):
            raise ValueError("Complement runtime settings differ from pinned discovery run")
    if frozen:
        baseline = frozen_baseline(frozen, protocol, args)
        reserve_final_output(protocol["frozen_experiment"], baseline, args.output, args.resume)
    predictions = run_frontier(
        protocol["rows"],
        args.output,
        provider=args.provider,
        model=args.model,
        effort=args.effort,
        max_output_tokens=args.max_tokens,
        max_requests=len(protocol["rows"]) * (1 + args.max_transient_retries),
        examples=protocol["examples"],
        resume=args.resume,
        max_inflight=args.concurrency,
        image_detail=protocol.get("image_detail"),
        response_format=protocol.get("response_format"),
        response_schema=protocol.get("response_schema"),
        privacy_level=protocol.get("privacy_level"),
        quantizations=protocol.get("quantizations"),
        timeout_seconds=complement_config["timeout_seconds"] if complement_config else 240,
        **policy,
    )
    # Never score against labels or protocol bytes that changed while requests ran.
    if sha(protocol_path) != protocol_sha256:
        raise ValueError("Prepared protocol changed while frontier requests were running")
    verify_protocol(protocol)
    report = score_predictions(
        protocol,
        predictions,
        model={
            "provider": args.provider,
            "name": args.model,
            "effort": args.effort,
            "max_output_tokens": args.max_tokens,
            "max_transient_retries": args.max_transient_retries,
            "retry_delay_seconds": args.retry_delay_seconds,
            "request_budget": len(protocol["rows"]) * (1 + args.max_transient_retries),
            "protocol_sha256": protocol_sha256,
            "frozen_experiment": protocol.get("frozen_experiment"),
            "coordinate_format": protocol.get("coordinate_format", "yxyx_1000"),
            "image_detail": protocol.get("image_detail", "provider-default"),
            "structured_output": bool(protocol.get("structured_output")),
            **(
                {key: protocol[key] for key in ("privacy_level", "quantizations")}
                if args.provider in ROUTER_COORDINATES else {}
            ),
            **(
                {"validation_complement": protocol["validation_complement"]}
                if complement_config else {}
            ),
        },
    )
    write_json(args.output / "detections.json", report.pop("detections"))
    write_json(args.output / "evaluation.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {"items", "matching_cache", "match_cache", "predictions", "ground_truth"}
            },
            indent=2,
        )
    )


def main():
    from vision_rl.frontier import PROVIDERS
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--count", type=int)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--prompt", choices=["minimal", "rich"], default="rich")
    p.add_argument("--examples", type=int, choices=range(0, 5), default=0)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    p.add_argument("--image-detail", choices=["low", "high", "original", "auto"])
    p.add_argument(
        "--structured-output", action="store_true",
        help="Gemini only: native schema-constrained normalized boxes (new explicit protocol)",
    )
    c = sub.add_parser("prepare-complement", help="extend fixed discovery to remaining validation")
    c.add_argument("--input", type=Path, required=True, help="original discovery input directory")
    c.add_argument("--from-run", type=Path, required=True, help="original completed run directory")
    c.add_argument("--output", type=Path, required=True)
    f = sub.add_parser("freeze", help="pin checkpoint bytes and final baseline protocols")
    f.add_argument("--source", type=Path, required=True)
    f.add_argument("--checkpoint", type=Path, required=True)
    f.add_argument("--artifact", type=Path, action="append", default=[])
    f.add_argument(
        "--baselines",
        type=Path,
        required=True,
        help="JSON list of exact model, prompt, example, and gateway-retry configurations",
    )
    f.add_argument("--criteria", type=Path, required=True, help="JSON object of success criteria")
    f.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--provider", choices=list(PROVIDERS), required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--effort", default="high")
    r.add_argument("--max-tokens", type=int, default=16384)
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--image-detail", choices=["low", "high", "original", "auto"])
    r.add_argument(
        "--max-transient-retries",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="additional attempts only for completed HTTP 502/503/504 errors",
    )
    r.add_argument("--retry-delay-seconds", type=float, default=2.0)
    r.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        record = create_freeze(
            args.source,
            args.checkpoint,
            json.loads(args.baselines.read_text()),
            json.loads(args.criteria.read_text()),
            args.output,
            args.artifact,
        )
        print(
            json.dumps(
                {
                    "freeze": str(args.output),
                    "sha256": sha(args.output),
                    "baselines": len(record["baselines"]),
                }
            )
        )
    else:
        commands = {"prepare": prepare, "prepare-complement": prepare_complement, "run": run}
        commands[args.command](args)


if __name__ == "__main__":
    main()
