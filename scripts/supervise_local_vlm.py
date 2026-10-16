"""Run one local model child with durable RSS telemetry and a bounded safety stop."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def metal_sample(path):
    """Read a bounded tail, tolerating the producer's incomplete final line."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 8192))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
            if "active_bytes" in record and "cache_bytes" in record:
                return record
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def supervise(command, status_path, log_path, rss_limit_bytes, *, interval=0.25,
              metal_telemetry=None, metal_limit_bytes=None):
    if not command or not 1024**3 <= rss_limit_bytes <= 48 * 1024**3:
        raise ValueError("Require a command and an explicit RSS limit between1GiB and48GiB")
    if (metal_telemetry is None) != (metal_limit_bytes is None) or (
        metal_limit_bytes is not None and not 1024**3 <= metal_limit_bytes <= 48 * 1024**3
    ):
        raise ValueError("Native Metal monitoring requires both telemetry and an explicit limit")
    status_path, log_path = Path(status_path), Path(log_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a second supervisor from starting the same probe.
    with status_path.open("x") as handle:
        json.dump({"created_at": now(), "state": "launching"}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    telemetry_path = status_path.with_name("process-memory.jsonl")
    with log_path.open("x") as log, telemetry_path.open("x") as telemetry:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        status = {
            "started_at": now(), "supervisor_pid": os.getpid(), "child_pid": child.pid,
            "command": command, "command_sha256": hashlib.sha256(
                json.dumps(command).encode()).hexdigest(),
            "rss_limit_bytes": rss_limit_bytes, "sample_interval_seconds": interval,
            "limit_semantics": "Observed child-process RSS watchdog, not an OS allocation cap",
            "mlx_memory_limit_semantics": "MLX setting is advisory",
            "scope": "Only this supervisor's own child; no automatic retries",
            "metal_telemetry": str(metal_telemetry) if metal_telemetry else None,
            "metal_active_plus_cache_limit_bytes": metal_limit_bytes,
        }
        write(status_path, status)
        peak, metal_peak = 0, 0
        while child.poll() is None:
            result = subprocess.run(
                ["/bin/ps", "-o", "rss=", "-p", str(child.pid)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                rss = int(result.stdout.strip()) * 1024
                peak = max(peak, rss)
                telemetry.write(json.dumps({"created_at": now(), "pid": child.pid,
                                            "rss_bytes": rss}) + "\n")
                telemetry.flush()
                if rss > rss_limit_bytes:
                    status.update(stop_reason="rss_safety_limit_exceeded",
                                  observed_rss_bytes_at_stop=rss)
            if metal_telemetry is not None and (native := metal_sample(metal_telemetry)):
                allocation = native["active_bytes"] + native["cache_bytes"]
                metal_peak = max(metal_peak, allocation)
                if allocation > metal_limit_bytes:
                    status.update(stop_reason="metal_allocation_safety_limit_exceeded",
                                  metal_sample_at_stop=native)
            if status.get("stop_reason"):
                write(status_path, status)
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
            time.sleep(interval)
        status.update(finished_at=now(), exit_code=child.wait(), peak_observed_rss_bytes=peak,
                      peak_observed_metal_active_plus_cache_bytes=metal_peak)
        write(status_path, status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--rss-limit-gib", type=float, default=36)
    parser.add_argument("--metal-telemetry", type=Path)
    parser.add_argument("--metal-limit-gib", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    status = supervise(
        command, args.status, args.log, int(args.rss_limit_gib * 1024**3),
        metal_telemetry=args.metal_telemetry,
        metal_limit_bytes=(int(args.metal_limit_gib * 1024**3)
                           if args.metal_limit_gib is not None else None),
    )
    raise SystemExit(status["exit_code"])


if __name__ == "__main__":
    main()
