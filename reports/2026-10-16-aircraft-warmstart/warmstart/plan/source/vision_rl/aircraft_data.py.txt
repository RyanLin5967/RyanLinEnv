"""One canonical, lazy FGVC-Aircraft image transform shared by all model arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

PREPROCESSING_ID = "fgvc-aircraft-rgb-remove-bottom20-v1"


def rgb_pixel_sha256(image: Image.Image) -> str:
    """Hash exact RGB pixels plus dimensions; no encoding-dependent JPEG/PNG bytes."""
    if image.mode != "RGB":
        raise ValueError("Pixel hashing requires an already decoded RGB image")
    header = json.dumps(
        {"width": image.width, "height": image.height, "mode": "RGB"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    hasher = hashlib.sha256(header + b"\0")
    hasher.update(image.tobytes())
    return hasher.hexdigest()


def load_source_rgb(path: str | Path) -> Image.Image:
    """Decode original pixels without silently applying orientation transforms."""
    with Image.open(path) as source:
        if source.getexif().get(274) not in (None, 1):
            raise ValueError("Unexpected EXIF orientation requires an explicit dataset decision")
        rgb = source.convert("RGB")
        rgb.load()
    return rgb


def remove_banner(image: Image.Image) -> Image.Image:
    if image.mode != "RGB" or image.height <= 20:
        raise ValueError("Aircraft input must be RGB with more than 20 pixel rows")
    return image.crop((0, 0, image.width, image.height - 20))


def load_aircraft_rgb(path: str | Path) -> Image.Image:
    """Return model-visible RGB pixels with the bottom20 banner removed exactly once.

    Input is the original JPEG. No resize, GT crop, EXIF transpose or re-encoding
    occurs here. All baselines and training arms must call this same function.
    """
    with load_source_rgb(path) as original:
        return remove_banner(original)


def aircraft_pixel_sha256(path: str | Path) -> str:
    """Fingerprint the model-visible image produced by the canonical transform."""
    with load_aircraft_rgb(path) as image:
        return rgb_pixel_sha256(image)
