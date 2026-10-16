"""Resume one failed author archive in place; reject changed or ignored ranges.

This recovery path requires a recorded natural failure and retained-prefix hash.
It never starts from byte zero, replaces a prefix, or downloads a second archive.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from extract_aircraft import (
    DECLARED_MD5,
    MAX_IMAGE_BYTES,
    SOURCE_BYTES,
    URL,
    atomic_json,
    check_space,
    file_hash,
)

HEADER_NAMES = (
    "Date",
    "Server",
    "Content-Length",
    "Content-Range",
    "Content-Type",
    "Content-Encoding",
    "ETag",
    "Last-Modified",
    "Accept-Ranges",
)
CHUNK_BYTES = 64 * 1024


def headers(response):
    return {name: response.headers[name] for name in HEADER_NAMES if name in response.headers}


def request(method="GET", **extra):
    return urllib.request.Request(
        URL, method=method, headers={"Accept-Encoding": "identity", **extra}
    )


def validate_range(response, start, end, entity):
    expected = f"bytes {start}-{end}/{SOURCE_BYTES}"
    if (
        response.status != 206
        or response.headers.get("Content-Range") != expected
        or int(response.headers.get("Content-Length", -1)) != end - start + 1
        or response.headers.get("Content-Encoding", "identity") != "identity"
    ):
        raise ValueError("Server did not honor the exact identity-encoded partial byte range")
    for name, value in entity.items():
        if response.headers.get(name) != value:
            raise ValueError(f"Author entity validator changed or disappeared: {name}")


def resume(root, failure_path, reserve, *, timeout=60, attempts=3):
    if not 1 <= attempts <= 3 or not 1 <= timeout <= 120:
        raise ValueError("Recovery requires one to three attempts and a bounded 1..120s timeout")
    failure = json.loads(failure_path.read_text())
    part = root / "downloads/fgvc-aircraft-2013b.tar.gz.part"
    archive = root / "downloads/fgvc-aircraft-2013b.tar.gz"
    output = root / "audit/range-resume.json"
    if archive.exists() or (root / "audit/archive.json").exists() or output.exists():
        raise ValueError("Existing completed/recovery evidence requires explicit review")
    if (
        failure.get("collected_exit_code") != 1
        or failure.get("old_process_present") is not False
        or Path(failure["partial_path"]).resolve() != part.resolve()
        or failure["expected_complete_bytes"] != SOURCE_BYTES
    ):
        raise ValueError("Recovery requires this archive's recorded natural failure")
    started = time.monotonic()
    evidence = {
        "schema_version": 1,
        "status": "started",
        "failure_evidence_sha256": file_hash(failure_path),
        "resume_source_sha256": file_hash(Path(__file__)),
        "original_prefix_bytes": failure["preserved_prefix_bytes"],
        "original_prefix_sha256": failure["preserved_prefix_sha256"],
        "source_url": URL,
        "expected_complete_bytes": SOURCE_BYTES,
        "timeout_seconds": timeout,
        "maximum_attempts": attempts,
        "requests": [],
        "payload_copy_count": 1,
    }
    with part.open("r+b") as target:
        fcntl.flock(target.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        target.seek(0, 2)
        initial = target.tell()
        if initial != failure["preserved_prefix_bytes"] or not 0 < initial < SOURCE_BYTES:
            raise ValueError("Retained prefix length changed or is not an incomplete archive")
        target.seek(0)
        if hashlib.file_digest(target, "sha256").hexdigest() != failure["preserved_prefix_sha256"]:
            raise ValueError("Retained prefix hash differs from the natural-failure evidence")
        check_space(root, SOURCE_BYTES - initial + MAX_IMAGE_BYTES, reserve)
        try:
            with urllib.request.urlopen(request("HEAD"), timeout=timeout) as response:
                head = headers(response)
                if response.status != 200 or int(head.get("Content-Length", -1)) != SOURCE_BYTES:
                    raise ValueError("Author resource length/status changed before resume")
                if head.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Author resource unexpectedly uses content encoding")
            entity = {name: head[name] for name in ("ETag", "Last-Modified") if name in head}
            strong_etag = entity.get("ETag", "")
            conditional = (
                strong_etag
                if strong_etag and not strong_etag.startswith("W/")
                else entity.get("Last-Modified")
            )
            if not conditional:
                raise ValueError("No usable author entity validator; explicit recovery required")
            evidence["head_headers"] = head
            evidence["if_range"] = conditional
            # The first transfer did not retain server validators. Verify a local tail
            # window before trusting a conditional continuation of today's entity.
            tail_start = max(0, initial - CHUNK_BYTES)
            target.seek(tail_start)
            tail = target.read(initial - tail_start)
            with urllib.request.urlopen(
                request(Range=f"bytes={tail_start}-{initial - 1}", **{"If-Range": conditional}),
                timeout=timeout,
            ) as response:
                validate_range(response, tail_start, initial - 1, entity)
                evidence["tail_verification_headers"] = headers(response)
                if response.read(len(tail) + 1) != tail:
                    raise ValueError("Author resource no longer matches the preserved prefix tail")
            evidence["tail_verified_bytes"] = len(tail)
            evidence["tail_verified_sha256"] = hashlib.sha256(tail).hexdigest()
            evidence["original_transfer_entity_validator_recorded"] = False
            evidence["entity_identity_limit"] = (
                "Current validators and exact retained-tail overlap checked; original request "
                "did not retain validators. Final full hashes and declared MD5 remain mandatory."
            )
            for attempt in range(attempts):
                target.seek(0, 2)
                start = target.tell()
                record = {"attempt": attempt + 1, "start": start, "status": "started"}
                evidence["requests"].append(record)
                atomic_json(output, evidence)
                try:
                    with urllib.request.urlopen(
                        request(
                            Range=f"bytes={start}-{SOURCE_BYTES - 1}", **{"If-Range": conditional}
                        ),
                        timeout=timeout,
                    ) as response:
                        record["headers"] = headers(response)
                        validate_range(response, start, SOURCE_BYTES - 1, entity)
                        next_log = start + 16 * 1024**2
                        while target.tell() < SOURCE_BYTES:
                            remaining = SOURCE_BYTES - target.tell()
                            chunk = response.read(min(CHUNK_BYTES, remaining))
                            if not chunk:
                                raise ConnectionError("Author partial response ended prematurely")
                            if len(chunk) > remaining:
                                raise ValueError(
                                    "Author response exceeds the reviewed archive size"
                                )
                            check_space(root, remaining + MAX_IMAGE_BYTES, reserve)
                            target.write(chunk)
                            target.flush()
                            if target.tell() >= next_log:
                                print(
                                    json.dumps(
                                        {
                                            "downloaded_bytes": target.tell(),
                                            "total_bytes": SOURCE_BYTES,
                                        }
                                    ),
                                    flush=True,
                                )
                                next_log += 16 * 1024**2
                        if response.read(1):
                            raise ValueError("Author response has bytes beyond its declared range")
                    record["status"] = "complete"
                    break
                except (TimeoutError, ConnectionError, OSError) as exc:
                    record.update(
                        status="transport_failure", error=type(exc).__name__ + ": " + str(exc)
                    )
                    target.flush()
                    record["preserved_bytes"] = target.seek(0, 2)
                    atomic_json(output, evidence)
                    if attempt + 1 == attempts or record["preserved_bytes"] == SOURCE_BYTES:
                        raise
                    time.sleep(5)
            target.flush()
            target.seek(0)
            sha256 = hashlib.file_digest(target, "sha256").hexdigest()
            target.seek(0)
            md5 = hashlib.file_digest(target, "md5").hexdigest()
            target.seek(0)
            prefix_hash = hashlib.sha256()
            remaining = initial
            while remaining:
                chunk = target.read(min(1024**2, remaining))
                if not chunk:
                    raise ValueError("Completed archive lost the preserved prefix")
                prefix_hash.update(chunk)
                remaining -= len(chunk)
            if prefix_hash.hexdigest() != failure["preserved_prefix_sha256"]:
                raise ValueError("Recovery modified the original retained prefix")
            receipt = {
                "url": URL,
                "bytes": SOURCE_BYTES,
                "sha256": sha256,
                "observed_md5": md5,
                "declared_md5": DECLARED_MD5,
                "declared_md5_matches": md5 == DECLARED_MD5,
                "transport": (
                    "HTTPS with standard certificate verification; conditional Range resume"
                ),
                "hash_verification": (
                    "Complete received bytes hashed; declared MD5 comparison retained"
                ),
                "resume_evidence_path": str(output),
            }
            part.replace(archive)
            atomic_json(root / "audit/archive.json", receipt)
            evidence.update(status="complete", full_archive=receipt, original_prefix_preserved=True)
            atomic_json(output, evidence)
            print(
                json.dumps(
                    {
                        "resume_complete": True,
                        "archive": str(archive),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            return archive, receipt
        except BaseException as exc:
            target.flush()
            evidence.update(
                status="failed_preserved",
                error=type(exc).__name__ + ": " + str(exc),
                preserved_bytes=target.seek(0, 2),
            )
            atomic_json(output, evidence)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/fgvc-aircraft-2013b"))
    parser.add_argument(
        "--failure-receipt",
        type=Path,
        default=Path("runs/aircraft-audit/original-transfer-failure-2026-10-16.json"),
    )
    parser.add_argument(
        "--metadata-archive",
        type=Path,
        default=Path("runs/aircraft-audit/reproduced/annotations.tar.gz"),
    )
    parser.add_argument("--reserve-gib", type=float, default=10)
    parser.add_argument("--complete-audit", action="store_true")
    args = parser.parse_args()
    resume(args.root, args.failure_receipt, int(args.reserve_gib * 1024**3))
    if args.complete_audit:
        for script, extra in (
            (
                "extract_aircraft.py",
                ["--reserve-gib", str(args.reserve_gib), "--remove-verified-archive"],
            ),
            ("verify_aircraft_clearance.py", []),
            ("prepare_aircraft_manifests.py", []),
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name(script)),
                    "--root",
                    str(args.root),
                    "--metadata-archive",
                    str(args.metadata_archive),
                    *extra,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
