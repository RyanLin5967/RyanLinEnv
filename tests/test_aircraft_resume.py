import fcntl
import hashlib
import io
import json
import runpy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
RECOVER = runpy.run_path(str(Path(__file__).parents[1] / "scripts/resume_aircraft.py"))


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


def fixture(tmp_path, monkeypatch, *, rejected=None, timeout_once=False):
    full = b"preserved" + b"remaining"
    prefix = full[:9]
    root = tmp_path / "dataset"
    partial = root / "downloads/fgvc-aircraft-2013b.tar.gz.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(prefix)
    failure = tmp_path / "failure.json"
    failure.write_text(
        json.dumps(
            {
                "collected_exit_code": 1,
                "old_process_present": False,
                "partial_path": str(partial),
                "expected_complete_bytes": len(full),
                "preserved_prefix_bytes": len(prefix),
                "preserved_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            }
        )
    )
    globals_ = RECOVER["resume"].__globals__
    monkeypatch.setitem(globals_, "SOURCE_BYTES", len(full))
    monkeypatch.setitem(globals_, "MAX_IMAGE_BYTES", 0)
    monkeypatch.setitem(globals_, "CHUNK_BYTES", 4)
    monkeypatch.setitem(globals_, "check_space", lambda *args: None)
    monkeypatch.setattr("time.sleep", lambda _: None)
    requests = []
    entity = {"ETag": '"fixed"', "Last-Modified": "Fri, 17 Jan 2014 17:42:51 GMT"}

    def urlopen(request, *, timeout):
        assert timeout == 60
        headers = dict(request.header_items())
        requests.append({"method": request.method, **headers})
        if request.method == "HEAD":
            return Response(b"", 200, {**entity, "Content-Length": str(len(full))})
        assert headers["If-range"] == entity["ETag"]
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        response_headers = {
            **entity,
            "Content-Range": f"bytes {start}-{end}/{len(full)}",
            "Content-Length": str(end - start + 1),
        }
        status, body = 206, full[start : end + 1]
        if start >= len(prefix):
            if rejected == "ignored":
                status, body = 200, full
            elif rejected == "offset":
                response_headers["Content-Range"] = f"bytes 0-{end}/{len(full)}"
            elif rejected == "entity":
                response_headers["ETag"] = '"changed"'
            if timeout_once and start == len(prefix):
                response = Response(body, status, response_headers)
                original_read = response.read

                def read(length):
                    if response.tell() >= 4:
                        raise TimeoutError("synthetic read timeout")
                    return original_read(length)

                response.read = read
                return response
        elif rejected == "tail":
            body = b"X" * len(body)
        return Response(body, status, response_headers)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return root, partial, failure, full, prefix, requests


def test_resume_preserves_prefix_and_rejects_full_body_duplicate(tmp_path, monkeypatch):
    root, partial, failure, full, prefix, requests = fixture(tmp_path, monkeypatch)
    archive, receipt = RECOVER["resume"](root, failure, 0)
    assert archive.read_bytes() == full and not partial.exists()
    assert archive.read_bytes()[: len(prefix)] == prefix
    assert receipt["sha256"] == hashlib.sha256(full).hexdigest()
    assert [row.get("Range") for row in requests] == [None, "bytes=5-8", "bytes=9-17"]
    evidence = json.loads((root / "audit/range-resume.json").read_text())
    assert evidence["original_prefix_preserved"] and evidence["payload_copy_count"] == 1
    assert len(list((root / "downloads").iterdir())) == 1


@pytest.mark.parametrize("rejected", ["ignored", "offset", "entity", "tail"])
def test_invalid_range_entity_or_overlap_never_appends(tmp_path, monkeypatch, rejected):
    root, partial, failure, _, prefix, _ = fixture(tmp_path, monkeypatch, rejected=rejected)
    with pytest.raises(ValueError):
        RECOVER["resume"](root, failure, 0)
    assert partial.read_bytes() == prefix
    assert not (root / "audit/archive.json").exists()
    assert (
        json.loads((root / "audit/range-resume.json").read_text())["status"] == "failed_preserved"
    )


def test_read_timeout_resumes_from_new_end_without_repeating_appended_bytes(tmp_path, monkeypatch):
    root, _, failure, full, _, requests = fixture(tmp_path, monkeypatch, timeout_once=True)
    archive, _ = RECOVER["resume"](root, failure, 0)
    assert archive.read_bytes() == full
    assert [row.get("Range") for row in requests] == [
        None,
        "bytes=5-8",
        "bytes=9-17",
        "bytes=13-17",
    ]
    evidence = json.loads((root / "audit/range-resume.json").read_text())
    assert evidence["requests"][0]["status"] == "transport_failure"
    assert evidence["requests"][0]["preserved_bytes"] == 13
    assert evidence["requests"][1]["start"] == 13


def test_modified_prefix_fails_before_network(tmp_path, monkeypatch):
    root, partial, failure, _, prefix, requests = fixture(tmp_path, monkeypatch)
    partial.write_bytes(b"X" + prefix[1:])
    with pytest.raises(ValueError, match="prefix hash"):
        RECOVER["resume"](root, failure, 0)
    assert not requests and partial.read_bytes() == b"X" + prefix[1:]


def test_second_recovery_cannot_open_network_while_owner_holds_lock(tmp_path, monkeypatch):
    root, partial, failure, _, prefix, requests = fixture(tmp_path, monkeypatch)
    with partial.open("r+b") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            RECOVER["resume"](root, failure, 0)
    assert not requests and partial.read_bytes() == prefix
