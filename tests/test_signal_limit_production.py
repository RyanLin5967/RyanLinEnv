"""Refusals of the signal-limit production package (scripts/freshpool_launch/
signal_limit_production): start.py's checks and prepare.py's activated-copy rule.

No model, no uwrun, no state: each check is called on fixtures written here, and each
expected value comes from the fixture (never from the subject)."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "freshpool_launch" / "signal_limit_production"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE.parent))
prepare = importlib.import_module("prepare")
start = importlib.import_module("start")


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- prepare: activated copy


def test_activated_copy_may_differ_only_in_status_activated_activation():
    source = {"study_id": "s", "status": "draft", "activated": None, "activation": None,
              "gates": {"g": 1}}
    good = {**source, "status": "ACTIVATED", "activated": True, "activation": {"by": "root"}}
    assert prepare.check_activated_copy(source, good) == ["activated", "activation", "status"]
    for change in ({"gates": {"g": 2}}, {"study_id": "t"}, {"extra": 1}):
        with pytest.raises(SystemExit, match="differs outside"):
            prepare.check_activated_copy(source, {**good, **change})
    with pytest.raises(SystemExit, match="differs outside"):
        prepare.check_activated_copy(source, {**good, "activated": False})  # not activated
    missing = dict(good)
    del missing["gates"]
    with pytest.raises(SystemExit, match="differs outside"):
        prepare.check_activated_copy(source, missing)
    reordered = {"activation": good["activation"], **{k: v for k, v in good.items()
                                                      if k != "activation"}}
    with pytest.raises(SystemExit, match="reorders"):
        prepare.check_activated_copy(source, reordered)


def test_committed_activated_copies_differ_from_their_sources_only_in_activation_fields():
    for copy, source in ((prepare.STUDY_ACTIVATED, prepare.STUDY),
                         (prepare.PROTOCOL_ACTIVATED, prepare.PROTOCOL)):
        a = json.loads((ROOT / copy).read_text())
        s = json.loads((ROOT / source).read_text())
        assert sorted(k for k in set(a) | set(s) if a.get(k) != s.get(k)) == [
            "activated", "activation", "status"]
        assert a["activation"]["approved_by"] == "root"  # the placeholder, until root's line
        assert a["activation"]["activates"]["sha256"] == hashlib.sha256(
            (ROOT / source).read_bytes()).hexdigest()


def test_committed_plan_hash_and_its_g0_pins():
    plan_bytes = (ROOT / prepare.PLAN).read_bytes()
    assert hashlib.sha256(plan_bytes).hexdigest() == (ROOT / prepare.PLAN_SHA).read_text().strip()
    plan = json.loads(plan_bytes)
    assert plan["execution"] == "uwaterloo" and plan["environment"][
        "ACCELERATE_TORCH_DEVICE"] == plan["uw"]["device"]
    for name, digest in plan["g0"]["files_sha256"].items():
        assert hashlib.sha256((ROOT / plan["g0"]["manifest_dir"] / name).read_bytes()
                              ).hexdigest() == digest
    # TRAIN rows only, in images/train/ (the bundle ships exactly what the rows name).
    for name in plan["g0"]["files_sha256"]:
        for line in (ROOT / plan["g0"]["manifest_dir"] / name).read_text().splitlines():
            row = json.loads(line)
            assert row["split"] == "train" and row["image"].startswith("images/train/")


# ---------------------------------------------------------------- start.py refusals


def test_start_refuses_outside_the_frozen_worktree(tmp_path):
    here = tmp_path / "elsewhere" / "scripts" / "freshpool_launch" / "signal_limit_production"
    here.mkdir(parents=True)
    with pytest.raises(start.Refused) as refused:
        start.frozen_worktree(here=here, frozen=tmp_path / "frozen")
    assert refused.value.reason == "start_not_run_from_frozen_worktree"


def _repo(tmp_path, log_text):
    repo = tmp_path / "repo"
    for path, text in ((prepare.STUDY_ACTIVATED, '{"a": 1}'),
                       (prepare.PROTOCOL_ACTIVATED, '{"p": 1}'),
                       (start.DECISION_LOG, log_text)):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    return repo


def test_start_refuses_without_roots_decision_log_line_naming_both_copies(tmp_path):
    study, protocol = sha_text('{"a": 1}'), sha_text('{"p": 1}')
    for text, missing in (("", 2), (f"D99 study {study}\n", 1),
                          (f"D99 protocol {protocol}\n", 1)):
        with pytest.raises(start.Refused) as refused:
            start.root_activation(_repo(tmp_path / str(len(text)), text))
        assert refused.value.reason == "root_activation_line_missing"
        assert len(refused.value.detail["missing"]) == missing
    found = start.root_activation(_repo(tmp_path / "ok", f"D99 {study} and {protocol}\n"))
    assert found == {prepare.STUDY_ACTIVATED: study, prepare.PROTOCOL_ACTIVATED: protocol}


def test_start_refuses_a_plan_whose_bytes_differ_from_plan_sha256(tmp_path):
    repo = tmp_path / "repo"
    (repo / prepare.PLAN).parent.mkdir(parents=True)
    (repo / prepare.PLAN).write_text('{"schema": "x"}\n')
    (repo / prepare.PLAN_SHA).write_text(sha_text('{"schema": "x"}\n') + "\n")
    assert start.plan_identity(repo)[1] == sha_text('{"schema": "x"}\n')
    (repo / prepare.PLAN).write_text('{"schema": "y"}\n')  # one byte changed, sha kept
    with pytest.raises(start.Refused) as refused:
        start.plan_identity(repo)
    assert refused.value.reason == "plan_differs_from_plan_sha256"


def test_start_refuses_when_prepare_check_reports_stale(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho '{\"status\": \"stale\"}'\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(start.Refused) as refused:
        start.prepare_check(ROOT, str(fake))
    assert refused.value.reason == "prepare_check_failed"


def test_start_refuses_when_no_host_qualifies(tmp_path, monkeypatch):
    uw_stage_a = importlib.import_module("uw_stage_a")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"uw": {"hosts": ["h1", "h2"], "per_host": 2,
                                       "min_free_ram_gb": 32, "min_local_gb": 8,
                                       "min_idle_threads": 16, "device": "cpu"}}))
    # h1: provisioned but 4 idle threads (24 - 20); h2: not provisioned; h3 is not allowlisted.
    monkeypatch.setattr(uw_stage_a.UWRunner, "hosts", lambda self: {
        "h1": (20.0, 24, 100, 0, 400, True), "h2": (0.5, 24, 100, 0, 400, False)})
    with pytest.raises(start.Refused) as refused:
        start.probe_hosts(plan)
    assert refused.value.reason == "no_uw_host_qualifies"
    hosts = refused.value.detail["hosts"]
    assert hosts["h1"]["qualifies"] is False and hosts["h2"]["qualifies"] is False
    monkeypatch.setattr(uw_stage_a.UWRunner, "hosts", lambda self: {
        "h1": (2.0, 24, 100, 1, 400, True)})
    report = start.probe_hosts(plan)
    assert report["h1"]["qualifies"] is True and report["h2"] == {"reachable": False}


def test_dry_run_stops_at_the_first_refusal_and_never_starts_the_chain(monkeypatch, capsys):
    started = []
    monkeypatch.setattr(start, "frozen_worktree", lambda: "c0ffee")
    monkeypatch.setattr(start, "root_activation", lambda repo: (_ for _ in ()).throw(
        start.Refused("root_activation_line_missing")))
    monkeypatch.setattr(start, "start_chain", lambda *a, **k: started.append(a))
    assert start.main(["--dry-run"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "root_activation_line_missing"
    assert started == []
