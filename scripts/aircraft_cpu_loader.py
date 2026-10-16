"""Prospective CPU7946 Linux loader amendment; importing allocates no model.

Only Transformers' CPU reads of the two pinned base shards use safetensors'
official pread backend. Conversion, tensors, precision and training stay native.
The process-local alias and serial-loading environment are restored on exit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

POLICY = {
    "schema_version": 1,
    "amendment": "linux-cpu-serial-pread-v1",
    "platform": "linux",
    "device": "cpu",
    "backend": "pread",
    "environment": {"HF_DEACTIVATE_ASYNC_LOAD": "1"},
    "versions": {"torch": "2.14.1", "transformers": "5.20.0", "safetensors": "0.8.0"},
    "torch_cpu_wheel_version": "2.14.1+cpu",
    "source_sha256": {
        "transformers/modeling_utils.py": (
            "553412f58334f65abba617f3b344828698dbf63615e109159fd03dafd3422d38"
        ),
        "transformers/core_model_loading.py": (
            "c365aea4e8e9db375a56d1c80b01d92439554fdf63561a08b0d537fad2eef6b5"
        ),
        "safetensors/__init__.pyi": (
            "88467818dfbb7a6e4e2d1c085efb1d07cee365d87350ebe85d1d76f9ced6325d"
        ),
    },
    "shards": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    "patched_alias": "transformers.modeling_utils.safe_open",
    "scientific_choices_changed": False,
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity():
    return {"file": "scripts/aircraft_cpu_loader.py", "sha256": sha256(__file__), **POLICY}


def _write(path, receipt):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _runtime():
    versions = {name: metadata.version(name) for name in POLICY["versions"]}
    if any(
        versions[name] != version
        and not (name == "torch" and versions[name] == POLICY["torch_cpu_wheel_version"])
        for name, version in POLICY["versions"].items()
    ):
        raise RuntimeError("CPU loader requires its exact reviewed package versions")
    sources = {}
    for name, expected in POLICY["source_sha256"].items():
        distribution = metadata.distribution(name.split("/")[0])
        actual = sha256(distribution.locate_file(name))
        if actual != expected:
            raise RuntimeError("CPU loader implementation source differs from the reviewed version")
        sources[name] = actual
    return versions, sources


def _scoped_open(original, allowed, receipt, receipt_path):
    def open_base(filename, *args, **kwargs):
        path = Path(filename)
        if (
            args
            or set(kwargs) - {"framework", "device", "backend"}
            or kwargs.get("framework") != "pt"
            or kwargs.get("device", "cpu") != "cpu"
            or kwargs.get("backend", "mmap") not in {"mmap", "pread"}
            or path.is_symlink()
            or path.resolve() not in allowed
        ):
            raise RuntimeError("CPU loader amendment only permits the two pinned CPU base shards")
        # This list is an audit of calls, not a new tensor cache or materializer.
        if len(receipt["opens"]) >= 64:
            raise RuntimeError("Unexpected number of base-shard loads")
        receipt["opens"].append({"file": path.name, "device": "cpu", "backend": "pread"})
        _write(receipt_path, receipt)
        return original(filename, **(kwargs | {"backend": "pread"}))

    return open_base


@contextmanager
def scoped_loader(model_directory, receipt_path, *, expected_sha256):
    """Apply only in the guarded Linux CPU worker; never edit installed modules."""
    if (
        sys.platform != "linux"
        or os.environ.get("ACCELERATE_USE_CPU") != "true"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise RuntimeError("CPU loader amendment requires the explicit Linux CPU worker")
    if sha256(__file__) != expected_sha256:
        raise RuntimeError("CPU loader bootstrap source identity changed")
    versions, sources = _runtime()
    directory = Path(model_directory).resolve(strict=True)
    allowed = {directory / name for name in POLICY["shards"]}
    if any(not path.is_file() or path.is_symlink() for path in allowed):
        raise RuntimeError("CPU loader requires both regular pinned base shards")
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise RuntimeError("Preserve the prior loader receipt; never retry a phase")

    import safetensors
    import transformers.modeling_utils as modeling

    original = modeling.safe_open
    if original is not safetensors.safe_open:
        raise RuntimeError("Transformers safe_open alias was already modified")
    previous = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
    receipt = {
        **identity(),
        "status": "active",
        "runtime_versions": versions,
        "runtime_sources": sources,
        "model_directory": str(directory),
        "opens": [],
        "installed_files_modified": False,
        "global_safetensors_alias_modified": False,
    }
    _write(receipt_path, receipt)
    modeling.safe_open = _scoped_open(original, allowed, receipt, receipt_path)
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    try:
        yield receipt
        if not receipt["opens"]:
            raise RuntimeError("Worker completed without exercising the pinned loader")
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        modeling.safe_open = original
        if previous is None:
            os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
        else:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = previous
        receipt["aliases_restored"] = (
            modeling.safe_open is original and safetensors.safe_open is original
        )
        receipt["environment_restored"] = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == previous
        _write(receipt_path, receipt)
