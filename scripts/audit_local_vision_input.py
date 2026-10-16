"""Replay a recorded TRAIN image through the installed NumPy processor on CPU."""

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.image_processing_utils import ImageProcessingMixin  # noqa: E402

from vision_rl.aircraft_data import load_aircraft_rgb, rgb_pixel_sha256  # noqa: E402
from vision_rl.frontier import _public  # noqa: E402
from vision_rl.local_vlm import prepare_local, sha_file  # noqa: E402


def audit(prepared, result, identifier, model_directory, mlx_package):
    protocol = json.loads((prepared / "inputs.json").read_text())
    row = next(row for row in protocol["rows"] if row["id"] == identifier)
    if protocol["phase"] != "train_probe" or row["split"] != "train":
        raise ValueError("This audit accepts only already attempted TRAIN probe images")
    recorded = next(row for row in protocol["prepared_images"] if row["id"] == identifier)
    events = [json.loads(line) for line in (result / "attempts.jsonl").read_text().splitlines()]
    reserved = [e for e in events if e["id"] == identifier and e["event"] == "reserved"]
    completed = [e for e in events if e["id"] == identifier and e["event"] == "completed"]
    if len(reserved) != 1 or len(completed) != 1:
        raise ValueError("Require one preserved reservation and completion")
    current = _public(prepare_local(row, protocol["preprocessing_id"]))
    assert current == recorded == reserved[0]["request"]
    native = completed[0]["preprocessing"]

    source = mlx_package / "models/qwen3_vl/processing_qwen3_vl.py"
    names = {"_resize_video_frames", "_smart_resize_image", "_to_numpy_image",
             "_load_qwen_vl_json", "_qwen_vl_image_kwargs", "Qwen3VLImageProcessor"}
    selected = [node for node in ast.parse(source.read_text()).body
                if getattr(node, "name", None) in names]
    assert len(selected) == len(names)
    namespace = {
        "np": np, "math": math, "Image": Image, "List": List, "Optional": Optional,
        "Tuple": Tuple, "ImageProcessingMixin": ImageProcessingMixin,
        "_IMAGE_PROCESSOR_KWARGS": {
            "min_pixels", "max_pixels", "resized_height", "resized_width",
        },
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    image_config = namespace["_qwen_vl_image_kwargs"](model_directory)
    image_processor = namespace["Qwen3VLImageProcessor"](**image_config)
    with load_aircraft_rgb(row["image"]) as rgb:
        assert rgb_pixel_sha256(rgb) == current["pixel_sha256"]
        tensors = image_processor(images=[rgb])
    assert tensors["image_grid_thw"].tolist() == native["image_grid_thw"]
    assert list(tensors["pixel_values"].shape) == native["pixel_values_shape"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory, local_files_only=True, trust_remote_code=False,
    )
    tokenizer.chat_template = (model_directory / "chat_template.jinja").read_text()
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                       {"type": "text", "text": row["prompt"]}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
        reasoning_effort="xhigh", preserve_thinking=True,
    )
    assert formatted == native["formatted_prompt"]
    image_tokens = int(tensors["image_grid_thw"].prod() // image_processor.merge_size**2)
    expanded = formatted.replace("<|image_pad|>", "<|image_pad|>" * image_tokens)
    tokens = tokenizer(expanded, padding=True, return_tensors="np")
    assert list(tokens["input_ids"].shape) == native["input_ids_shape"]
    actual_image_tokens = int((tokens["input_ids"] == tokenizer.convert_tokens_to_ids(
        "<|image_pad|>")).sum())
    assert image_tokens == actual_image_tokens
    assert "mlx.core" not in sys.modules and "torch" not in sys.modules
    return {
        "scope": "Read-only completed TRAIN request preprocessing replay; no model call",
        "id": identifier, "source_path": row["image"],
        "source_sha256": current["source_file_sha256"],
        "cropped_rgb_sha256": current["pixel_sha256"],
        "cropped_size": [current["width"], current["height"]],
        "prepared_reserved_current_pixels_equal": True,
        "formatted_prompt_exactly_reproduced": True,
        "processor": "mlx-vlm0.6.8 Qwen3VL NumPy port, patched by qwen3_5 loader",
        "processor_source_sha256": sha_file(source), "image_configuration": image_config,
        "image_grid_thw": native["image_grid_thw"],
        "pixel_values_shape": native["pixel_values_shape"],
        "input_ids_shape": native["input_ids_shape"],
        "image_pad_tokens": actual_image_tokens,
        "current_cpu_pixel_tensor_sha256": hashlib.sha256(
            tensors["pixel_values"].tobytes()).hexdigest(),
        "historical_pixel_tensor_hash_available": False,
        "limitation": (
            "Historical journal records input RGB hash and tensor shapes, not tensor bytes"),
        "model_calls": 0, "mlx_imported": False, "torch_imported": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "result", "model-directory", "mlx-package", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    report = audit(args.prepared, args.result, args.id, args.model_directory, args.mlx_package)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
