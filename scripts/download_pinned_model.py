"""Download one pinned public model copy, retaining partial files and a disk reserve."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_file(path, entry):
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Incorrect byte count: {entry['path']}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {entry['size_bytes']}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            sha256.update(block)
            git_blob.update(block)
    if entry.get("sha256"):
        if sha256.hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch: {entry['path']}")
    elif git_blob.hexdigest() != entry["git_blob_id"]:
        raise ValueError(f"Git blob mismatch: {entry['path']}")
    return {"path": entry["path"], "bytes": entry["size_bytes"], "sha256": sha256.hexdigest()}


def worker(spec, output, integrity):
    # Set before importing the hub. local_dir stores the sole model copy; no Xet cache.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    from huggingface_hub import snapshot_download

    snapshot_download(
        spec["model_repository"], revision=spec["revision"], local_dir=output,
        allow_patterns=[entry["path"] for entry in spec["files"]],
        max_workers=1, token=False,
    )
    verified = []
    for entry in spec["files"]:
        verified.append(verify_file(output / entry["path"], entry))
        print(json.dumps({"verified": entry["path"], "bytes": entry["size_bytes"]}), flush=True)
    write(integrity, {
        "model_repository": spec["model_repository"], "revision": spec["revision"],
        "verified_at": datetime.now(timezone.utc).isoformat(), "files": verified,
        "total_bytes": sum(entry["bytes"] for entry in verified),
        "quantization": spec["quantization"], "model_loaded": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=6)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output, args.status = args.output.resolve(), args.status.resolve()
    source = args.manifest.read_bytes()
    spec = json.loads(source)
    if len(spec["revision"]) != 40 or args.min_free_gib < 1:
        raise ValueError("Require a full model revision and an explicit positive disk reserve")
    for entry in spec["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Model files must remain inside the target directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    integrity = args.status.with_name("model-integrity.json")
    identity = {
        "repository": spec["model_repository"], "revision": spec["revision"],
        "manifest_sha256": hashlib.sha256(source).hexdigest(),
    }
    identity_path = args.output / ".pinned-download.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Refusing to reuse a download directory for a different snapshot")
    else:
        write(identity_path, identity)
    if args.worker:
        worker(spec, args.output, integrity)
        return
    with (args.output / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        downloaded = sum(
            path.stat().st_size for path in args.output.rglob("*")
            if path.is_file() and (path.name.endswith(".incomplete") or path.name in {
                entry["path"] for entry in spec["files"]
            })
        )
        remaining = max(0, sum(entry["size_bytes"] for entry in spec["files"]) - downloaded)
        reserve = int(args.min_free_gib * 1024**3)
        free = shutil.disk_usage(args.output).free
        if free < remaining + reserve + 256 * 1024**2:
            raise ValueError(
                "Insufficient disk for the remaining snapshot plus reserved free space"
            )
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        status = {
            **identity, "supervisor_pid": os.getpid(), "child_pid": child.pid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output": str(args.output), "remaining_bytes_at_start": remaining,
            "minimum_free_bytes": reserve, "single_copy": True,
            "xet_cache_disabled": True, "model_loaded": False,
        }
        write(args.status, status)
        while child.poll() is None:
            if shutil.disk_usage(args.output).free < reserve + 256 * 1024**2:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                status["stop_reason"] = "disk_reserve_reached; partial download retained"
                break
            time.sleep(0.2)
        status.update(
            exit_code=child.wait(), finished_at=datetime.now(timezone.utc).isoformat(),
            free_bytes_at_finish=shutil.disk_usage(args.output).free,
            complete=child.returncode == 0 and integrity.exists(),
        )
        write(args.status, status)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
