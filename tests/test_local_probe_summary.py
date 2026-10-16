import json
import runpy
from pathlib import Path

import pytest


def test_native_usage_summary_keeps_unknown_attempt_in_denominator(tmp_path):
    summarize = runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "scripts/summarize_local_probe.py"
    ))["summarize"]
    rows = [{"id": str(i), "width": 20, "height": 30, "pixel_sha256": "a" * 64}
            for i in range(2)]
    (tmp_path / "run.json").write_text(json.dumps({
        "identity_sha256": "identity", "identity": {
            "rows": rows, "model": {"repository": "fixture", "revision": "pinned",
                                     "quantization": {"bits": 8}},
            "config": {"phase": "train_probe"},
        },
    }))
    events = [{"event": "reserved", "id": row["id"], "identity_sha256": "identity",
               "request": row} for row in rows]
    events.insert(1, {
        "event": "completed", "id": "0", "identity_sha256": "identity",
        "prediction": {"latency_seconds": 2}, "native_response": {
            "finish_reason": "stop", "text": "thinking</think>A", "prompt_tokens": 10,
            "generation_tokens": 4, "total_tokens": 14,
        },
    })
    journal = tmp_path / "attempts.jsonl"
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"finished_at": "now", "exit_code": -15}))
    result = summarize(tmp_path, status)
    assert result["coverage"] == {"expected": 2, "reserved": 2, "completed": 1,
                                  "normal_finish": 1, "unknown": 1, "unattempted": 0,
                                  "all_finished_normally": False}
    assert result["known_native_token_totals"]["total_tokens"] == 14
    assert result["images"][1]["error"] == "unknown_after_interruption"
    with journal.open("a") as handle:
        handle.write(json.dumps(events[1]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(tmp_path, status)
    status.write_text("{}")
    with pytest.raises(ValueError, match="live probe"):
        summarize(tmp_path, status)
