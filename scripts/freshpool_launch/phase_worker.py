"""Run one freshpool stage entry point in its own process, with the MPS driver guard.

Modelled on runs/aircraft-controlled400-launch-v1/phase_worker.py. The launcher starts this
file once per stage in a new session (process group); this file refuses unless the frozen
environment is set, imports exactly one entry ``module:function`` whose source file has the
SHA256 the launcher pinned (an allowlist by content, not by name), samples MPS driver
allocation every 0.5 s for model-owning kinds, and writes a summary receipt.

The device is the frozen ``ACCELERATE_TORCH_DEVICE``, which is what Accelerate (and so the
trainer) actually uses. It must be ``mps``, except that ``--dry-run`` (passed by the launcher
only for a plan carrying ``"dry_run": true``, the CPU dry run in scripts/freshpool_dryrun)
requires ``cpu`` instead; anything else refuses. On ``cpu`` there is no MPS allocation to
guard, so the driver guard is not armed. Every forwarded ``--device`` must equal the device,
and a stage forwarding none is taken at the entries' default, ``mps``, so an inference entry
(which does not go through Accelerate) can never run on a device the guard does not cover.

Blind spots, stated: the hash covers the entry module's own file, not modules it imports
(the launcher pins those through the plan's ``file_sha256``); the driver limit is sampled,
so an allocation spike shorter than one interval is missed; kind ``common`` gets no MPS
guard and must not load a model; the forwarded ``--device`` is read by flag name, so an
entry that chose its device some other way would not be seen.
"""

import argparse
import hashlib
import importlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

DRIVER_LIMIT_BYTES = 24 * 1024**3
SAMPLE_SECONDS = 0.5
KINDS = ("train", "assay", "common")
MODEL_KINDS = ("train", "assay")
FROZEN_ENVIRONMENT = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "HF_HUB_OFFLINE": "1",
}
# ACCELERATE_TORCH_DEVICE: mps for a real launch; cpu only under --dry-run.
DEVICE, DRY_RUN_DEVICE = "mps", "cpu"
ENTRY_DEFAULT_DEVICE = "mps"  # h_assay, p0_screen and the trainer all default --device to mps


def forwarded_devices(forwarded):
    """Every --device value in the forwarded argv (both spellings), or the entries' default."""
    values = [forwarded[i + 1] for i, a in enumerate(forwarded[:-1]) if a == "--device"]
    values += [a.split("=", 1)[1] for a in forwarded if a.startswith("--device=")]
    if forwarded and forwarded[-1] == "--device":
        values.append(None)
    return values or [ENTRY_DEFAULT_DEVICE]


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_entry(entry, expected_sha256):
    module_name, separator, function_name = entry.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"entry must be module:function, got {entry!r}")
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if source is None or sha(source) != expected_sha256:
        raise RuntimeError(f"entry module {module_name} is not the pinned source")
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"entry {entry} is not callable")
    return function, str(Path(source).resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--guard-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--entry-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise ValueError("forwarded entry arguments must follow a literal --")
    split = argv.index("--")
    args, forwarded = parser.parse_args(argv[:split]), argv[split + 1:]
    for name, value in FROZEN_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"{name} must be {value!r} before any torch import")
    device = os.environ.get("ACCELERATE_TORCH_DEVICE")
    required = DRY_RUN_DEVICE if args.dry_run else DEVICE
    if device != required:
        raise RuntimeError(f"ACCELERATE_TORCH_DEVICE must be {required!r} before any torch "
                           f"import (dry run: {args.dry_run}), got {device!r}")
    if args.kind in MODEL_KINDS and set(forwarded_devices(forwarded)) != {device}:
        raise RuntimeError(f"forwarded --device {forwarded_devices(forwarded)} differs from "
                           f"ACCELERATE_TORCH_DEVICE {device!r}")
    args.guard_dir.mkdir(parents=True, exist_ok=True)
    identifier = f"{args.stage}-{os.getpid()}-{time.time_ns()}"
    summary = args.guard_dir / (identifier + ".json")
    started = time.monotonic()
    stats = {
        "pid": os.getpid(),
        "stage": args.stage,
        "kind": args.kind,
        "device": device,
        "dry_run": args.dry_run,
        "entry": args.entry,
        "entry_sha256": args.entry_sha256,
        "argv": forwarded,
        "status": "running",
        "driver_limit_bytes": DRIVER_LIMIT_BYTES,
        "peak_driver_allocated_bytes": 0,
        "sampling_interval_seconds": SAMPLE_SECONDS,
    }
    write(summary, stats)
    done, thread = threading.Event(), None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        function, source = load_entry(args.entry, args.entry_sha256)
        stats["entry_source"] = source
        if args.kind in MODEL_KINDS and device == "cpu":
            import torch

            torch.set_num_threads(2)
            stats["mps_driver_guard"] = "not armed: device cpu"
        elif args.kind in MODEL_KINDS:
            import torch

            torch.set_num_threads(2)
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is unavailable; no CPU fallback")
            recommended = torch.mps.recommended_max_memory()
            fraction = min(1.0, DRIVER_LIMIT_BYTES / recommended)
            torch.mps.set_per_process_memory_fraction(fraction)
            stats.update(recommended_max_memory_bytes=recommended, allocator_fraction=fraction)

            def telemetry():
                try:
                    with (args.guard_dir / (identifier + ".jsonl")).open("x") as handle:
                        while not done.is_set():
                            driver = torch.mps.driver_allocated_memory()
                            stats["peak_driver_allocated_bytes"] = max(
                                stats["peak_driver_allocated_bytes"], driver
                            )
                            handle.write(json.dumps({
                                "elapsed_seconds": time.monotonic() - started,
                                "driver_allocated_bytes": driver,
                            }) + "\n")
                            handle.flush()
                            write(summary, stats)
                            if driver > DRIVER_LIMIT_BYTES:
                                raise RuntimeError("MPS driver allocation exceeded 24 GiB")
                            done.wait(SAMPLE_SECONDS)
                except Exception as exc:
                    stats["stop_reason"] = f"{type(exc).__name__}: {exc}"
                    write(summary, stats)
                    os.kill(os.getpid(), signal.SIGINT)

            thread = threading.Thread(target=telemetry, daemon=True)
            thread.start()
        code = function(forwarded)
        code = 0 if code is None else code
        stats["exit_code"] = code
        stats["status"] = "completed" if code == 0 and "stop_reason" not in stats else "failed"
        return code if stats["status"] == "completed" else (code or 3)
    except BaseException as exc:
        stats.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        done.set()
        if thread is not None:
            thread.join(timeout=2)
        stats["elapsed_seconds"] = time.monotonic() - started
        write(summary, stats)


if __name__ == "__main__":
    raise SystemExit(main())
