import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def supervisor():
    return runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "scripts/supervise_local_vlm.py"
    ))["supervise"]


def test_real_child_completion_has_rss_evidence_and_cannot_be_relaunched(supervisor, tmp_path):
    status, log = tmp_path / "status.json", tmp_path / "output.log"
    command = [sys.executable, "-c", "print('completed', flush=True)"]
    result = supervisor(command, status, log, 1024**3, interval=0.01)
    assert result["exit_code"] == 0 and "stop_reason" not in result
    assert log.read_text() == "completed\n"
    assert json.loads(status.read_text())["child_pid"] == result["child_pid"]
    with pytest.raises(FileExistsError):
        supervisor(command, status, log, 1024**3)


def test_watchdog_terminates_own_child_on_observed_excess_without_retry(
    supervisor, tmp_path, monkeypatch,
):
    calls = []

    def sample(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=str(2 * 1024**2), stderr="")

    monkeypatch.setattr(subprocess, "run", sample)
    result = supervisor(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "status.json", tmp_path / "output.log", 1024**3, interval=0.01,
    )
    assert result["stop_reason"] == "rss_safety_limit_exceeded"
    assert result["peak_observed_rss_bytes"] == 2 * 1024**3
    assert result["exit_code"] != 0 and len(calls) == 1
    assert calls[0][-1] == str(result["child_pid"])


def test_metal_guard_reads_last_complete_native_sample_despite_inflight_write(tmp_path):
    module = runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "scripts/watch_local_metal_memory.py"
    ))
    path = tmp_path / "native.jsonl"
    assert module["sample"](path) is None
    path.write_text('{"active_bytes":100,"cache_bytes":20}\n{"active_bytes":')
    assert module["sample"](path) == {"active_bytes": 100, "cache_bytes": 20}


def test_integrated_metal_guard_stops_child_even_when_rss_undercounts(supervisor, tmp_path):
    telemetry = tmp_path / "metal.jsonl"
    telemetry.write_text(json.dumps({"active_bytes": 1024**3, "cache_bytes": 1}) + "\n")
    result = supervisor(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "status.json", tmp_path / "output.log", 2 * 1024**3,
        metal_telemetry=telemetry, metal_limit_bytes=1024**3, interval=0.01,
    )
    assert result["stop_reason"] == "metal_allocation_safety_limit_exceeded"
    assert result["peak_observed_metal_active_plus_cache_bytes"] == 1024**3 + 1
    assert result["exit_code"] != 0
