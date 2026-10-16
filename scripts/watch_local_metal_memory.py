"""Contain one already-authorized local probe using its recorded Metal allocation telemetry."""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def identity(pid):
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sample(path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 8192))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
            if "active_bytes" in value and "cache_bytes" in value:
                return value
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--limit-gib", type=float, default=36)
    args = parser.parse_args()
    if not 1 <= args.limit_gib <= 48 or identity(args.pid) != args.expected_identity:
        raise ValueError("Target process identity or explicit resource threshold is invalid")
    status = {
        "started_at": datetime.now(timezone.utc).isoformat(), "guardian_pid": os.getpid(),
        "target_pid": args.pid, "target_identity": args.expected_identity,
        "limit_bytes": int(args.limit_gib * 1024**3),
        "scope": "Authorized probe only; active plus cache Metal bytes; no retries",
        "semantics": "Sampled telemetry cutoff, not an OS allocation cap",
    }
    with args.status.open("x") as handle:
        json.dump(status, handle, indent=2)
    peak = 0
    while identity(args.pid) == args.expected_identity:
        value = sample(args.telemetry)
        if value is not None:
            allocation = value["active_bytes"] + value["cache_bytes"]
            peak = max(peak, allocation)
            if allocation > status["limit_bytes"]:
                status.update(stop_reason="metal_allocation_safety_limit_exceeded", sample=value)
                # Recheck exact start time and command immediately before signaling.
                if identity(args.pid) == args.expected_identity:
                    os.kill(args.pid, signal.SIGTERM)
                    deadline = time.monotonic() + 5
                    while (identity(args.pid) == args.expected_identity
                           and time.monotonic() < deadline):
                        time.sleep(0.1)
                    if identity(args.pid) == args.expected_identity:
                        os.kill(args.pid, signal.SIGKILL)
                break
        time.sleep(0.25)
    status.update(finished_at=datetime.now(timezone.utc).isoformat(),
                  peak_observed_active_plus_cache_bytes=peak)
    temporary = args.status.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(args.status)


if __name__ == "__main__":
    main()
