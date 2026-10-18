"""Drive the exploratory sweep v2 from the Mac: bootstrap hosts, launch arms, fetch, report.

Exploratory only (docs/aircraft-exploratory-sweep-v2.md): every arm trains and is assayed on
TRAIN-derived data. Nothing here reads, lists or copies native VAL or the sealed TEST.

All heavy work runs on UW hosts through ~/.claude/bin/uwrun; this script only launches,
polls and fetches small files, so it is light enough for the Mac under quiet mode
(Python 3.9 standard library only).

  runner.py bootstrap --payload DIR [--repin] [--host H ...]  # no host: pin only
  runner.py receipts                       # read every host's bootstrap receipt into state
  runner.py refresh                        # bootstrap the pinned payload on drained stale hosts
  runner.py continue ARM --reason R        # FAILED multi-stop arm: rest of its stops, in place
  runner.py launch --table T [--ids A,B] [--wave 1] [--rung 1] [--hosts H,...]
                   [--max-launch N] [--timeout S] [--dry-run]
  runner.py bench --host H --kind grpo4|grpo8|sft [--steps N] [--threads 16]
  runner.py poll                           # status, fetch finished arms, resolve controls
  runner.py promote --from-rung 1|2 [--late] # successive halving -> runs/.../arms-rung{2,3}.json
  runner.py derive sftall-seeds|sftall-rl  # data-scale replicas / SFT(all TRAIN) -> GRPO
  runner.py report [--resamples 20000]     # dev800 table + paired bootstrap vs matched SFT

State lives in --state (default runs/aircraft-exploratory-sweep-v2/state.json in the root
checkout) and is rewritten atomically under an fcntl lock after every change, keeping keys
other writers own (e.g. "features").

Refusals (exit non-zero, nothing launched): a host with no receipt, a receipt whose payload
sha256 differs from the payload this sweep pinned, a receipt without the trainer features an
arm needs, an arm whose dependency has not finished, a control whose RL arm summary is
missing, a launch that would leave less than 25% of a host's cores or memory free.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import fcntl
import json
import math
import random
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STUDY = "aircraft-exploratory-sweep-v2"
ROOT_CHECKOUT = Path("/Users/idide/RyanLinEnv")
SWEEP_DIR = ROOT_CHECKOUT / "runs" / STUDY
DEFAULT_STATE = SWEEP_DIR / "state.json"
UWRUN = Path.home() / ".claude" / "bin" / "uwrun"
HERE = Path(__file__).resolve()
DESIGN = HERE.parents[2] / "docs" / STUDY
ARM_EXEC = "payload/code/scripts/exploratory_sweep/arm_exec.py"
WAVE2_TABLE = "arms-wave2-rung1.json"  # scripts/exploratory_sweep/wave2_arms.py
RECEIPTS = "~/rlesweep/receipts"
HEADROOM = 0.75  # leave a quarter of every host's cores and memory free
ARM_RSS_GB = 16.0  # measured max RSS of one fp32 2B LoRA trainer was ~13.6 GB (iccad1, GNU time)
# Peak resident GiB per arm kind, MEASURED: sum of VmHWM over each live run's processes
# (uwrun ssh, /proc/PID/status, UWRUN_NAME), 22 running rung-1 arms, 2026-10-18T20:20Z;
# assay: rlesweep-assay-rss-probe-1 (in-process predict_train_subset, adapter W, iccad5).
# --mem = ceil(1.2 * max(training peak of the kind, assay peak)): the assay runs in the same
# run after training. G4 was not running; it is bounded by G8 (fewer generations), unverified.
PEAK_RSS_GIB = {"sft": 12.97, "grpo4": 13.56, "grpo8": 13.56, "grpo16": 15.22}
# Assay: self ru_maxrss after chunks 1-6 of 25 = 12.9, 13.621, 13.800, 13.800, 14.106,
# 14.362 GiB (rlesweep-assay-rss-probe-full, iccad5, 8 threads; stopped at chunk 6 by
# amendment 3, iccad5 belongs to the GATE 0 lane). It still grows, so the 25-chunk peak is
# EXTRAPOLATED (unverified): largest increment 0.306 GiB x 19 more chunks -> 20.2 GiB.
# Replace it with the measured max of summary.json `resources.self_maxrss_gib` from the
# first full assay under this payload (an sftall continuation).
ASSAY_PEAK_GIB = 20.2
MEM_HEADROOM = 1.2
NAME_OK = re.compile(r"[^A-Za-z0-9_-]")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- state ---------------------------------------------------------------------------
@contextlib.contextmanager
def state_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.is_file() else {}
        state.setdefault("schema_version", 1)
        state.setdefault("study_id", STUDY)
        for key in ("hosts", "arms"):
            state.setdefault(key, {})
        state.setdefault("uwrun_runs", [])
        yield state
        state["updated_at"] = now()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        tmp.replace(path)


def read_state(path: Path) -> dict:
    with state_lock(path) as state:
        return json.loads(json.dumps(state))


# ---- uwrun ---------------------------------------------------------------------------
def uw(*args: str, timeout: int = 300, check: bool = True) -> str:
    proc = subprocess.run([str(UWRUN), *args], capture_output=True, text=True,
                          timeout=timeout)
    if check and proc.returncode != 0:
        raise SystemExit(f"REFUSE: uwrun {' '.join(args[:2])} rc={proc.returncode}: "
                         f"{(proc.stderr or proc.stdout).strip()[-400:]}")
    return proc.stdout


def run_status(name: str) -> str:
    out = uw("status", name, timeout=180, check=False).strip().splitlines()
    return out[-1] if out else "UNKNOWN"


def host_table(text: str | None = None) -> dict:
    """{host: {load1, nproc, cores, avail_gb, local_gb}} from `uwrun hosts all`.

    Columns are found by the header row, never by position: uwrun v5 inserted `cores` and
    `ourThr`, which shifted availGB and localGB, and positional parsing then read physical
    cores as free GB without any error. A missing column refuses. `cores` is physical cores
    (uwrun v5 places by them: a second hardware thread adds 0-3% here, measured by uwrun).
    """
    lines = (uw("hosts", "all", timeout=300) if text is None else text).splitlines()
    header = next((ln.split() for ln in lines if ln.split()[:2] == ["host", "load1"]), None)
    need = ("load1", "nproc", "cores", "availGB", "localGB")
    if header is None or not set(need) <= set(header):
        raise SystemExit(f"REFUSE: `uwrun hosts all` header lacks {need}: {header}")
    col = {k: header.index(k) for k in need}
    hosts = {}
    for line in lines:
        parts = line.split()
        if len(parts) != len(header) or parts[0] == "host":
            continue
        try:
            hosts[parts[0]] = {"load1": float(parts[col["load1"]]),
                               "nproc": int(parts[col["nproc"]]),
                               "cores": int(parts[col["cores"]]),
                               "avail_gb": float(parts[col["availGB"]]),
                               "local_gb": float(parts[col["localGB"]])}
        except ValueError:
            raise SystemExit(f"REFUSE: unparsable `uwrun hosts` row: {line!r}")
    if not hosts:
        raise SystemExit("REFUSE: `uwrun hosts all` returned no parsable host rows")
    return hosts


def safe_name(text: str) -> str:
    return NAME_OK.sub("_", text)[:90]


# ---- bootstrap and receipts ------------------------------------------------------------
# uwrun syncs the ~1 GB payload into the host's /var/tmp, often the shared root fs
MIN_LOCAL_GB = 3.0


def low_disk_hosts(hosts: dict, names) -> list:
    """Hosts whose free /var/tmp (uwrun's localGB) is below MIN_LOCAL_GB, or not listed at all."""
    return sorted(h for h in names if (hosts.get(h) or {}).get("local_gb", 0.0) < MIN_LOCAL_GB)


def cmd_bootstrap(a, state_path: Path) -> int:
    live = sorted({x.get("host") for x in read_state(state_path)["arms"].values()
                   if x.get("status") in ("LAUNCHED", "RUNNING")} & set(a.host))
    if live:
        raise SystemExit(f"REFUSE: {live} run our arms; bootstrap rewrites ROOT/payload under"
                         " them (wait until drained; `refresh` does this)")
    low = low_disk_hosts(host_table(), a.host)
    if low:
        raise SystemExit(f"REFUSE: {low} have < {MIN_LOCAL_GB} GB free in /var/tmp (uwrun hosts"
                         " localGB); the payload sync would fill a shared disk")
    payload = Path(a.payload).resolve()
    doc = json.loads((payload / "PAYLOAD.json").read_bytes())
    import hashlib

    sha = hashlib.sha256((payload / "PAYLOAD.json").read_bytes()).hexdigest()
    if doc.get("code_dirty"):
        raise SystemExit("REFUSE: payload was staged from a dirty tree; commit, then restage")
    with state_lock(state_path) as state:
        pinned = state.get("payload", {}).get("sha256")
        if pinned and pinned != sha and not a.repin:
            raise SystemExit(f"REFUSE: state pins payload {pinned[:12]}; pass --repin to move")
        state["payload"] = {"sha256": sha, "dir": str(payload), "code_head": doc["code_head"]}
    for host in a.host:
        name = safe_name(f"rlesweep-boot-{host}-{doc['code_head'][:7]}-{int(time.time())}")
        uw("run", "--host", host, "--threads", "8", "--timeout", "5400", "--sync",
           str(payload), "--name", name, "--", "bash",
           "code/scripts/exploratory_sweep/bootstrap_host.sh", timeout=1800)
        with state_lock(state_path) as state:
            h = state["hosts"].setdefault(host, {})
            h.update(boot_run=name, boot_payload_sha256=sha, boot_started=now(), receipt=None)
            state["uwrun_runs"].append({"name": name, "host": host, "kind": "bootstrap",
                                        "at": now()})
        print(f"bootstrap {host}: {name}")
    return 0


def cmd_receipts(a, state_path: Path) -> int:
    state = read_state(state_path)
    bad = 0
    for host, h in sorted(state["hosts"].items()):
        status = run_status(h["boot_run"]) if h.get("boot_run") else "NONE"
        raw = uw("ssh", host, f"cat {RECEIPTS}/{host}.json", timeout=180, check=False)
        try:
            receipt = json.loads(raw)
        except json.JSONDecodeError:
            receipt = None
        ok = (receipt is not None and "DONE rc=0" in status
              and receipt["payload"]["payload_sha256"] == state.get("payload", {}).get("sha256"))
        summary = None
        if receipt is not None:
            summary = {"root": receipt["root"], "nproc": receipt["nproc"],
                       "payload_sha256": receipt["payload"]["payload_sha256"],
                       "code_head": receipt["payload"]["code_head"],
                       "features": receipt["features"]["available"],
                       "device": receipt["device"], "cpu_model": receipt["cpu_model"],
                       "created_at": receipt["created_at"], "valid": ok}
        with state_lock(state_path) as st:
            st["hosts"][host].update(boot_status=status, receipt=summary)
        print(f"{host}: {status} receipt={'OK' if ok else 'MISSING/STALE'}")
        bad += not ok
    return 1 if bad else 0


def drained_stale_hosts(state: dict) -> list:
    """Hosts whose receipt is for an older payload, none of our arms LAUNCHED/RUNNING on them,
    and no bootstrap of the pinned payload already issued (one attempt; a failed one shows in
    `receipts` and is not retried here). Re-bootstrapping rewrites ROOT/payload, so it never
    happens under a live arm."""
    pinned = state.get("payload", {}).get("sha256")
    live = {x.get("host") for x in state["arms"].values()
            if x.get("status") in ("LAUNCHED", "RUNNING")}
    return sorted(host for host, h in state["hosts"].items()
                  if (h.get("receipt") or {}).get("payload_sha256") not in (None, pinned)
                  and host not in live and h.get("boot_payload_sha256") != pinned)


def continuable_hosts(state: dict) -> set:
    """Hosts of FAILED arms in CONTINUE_FAMILIES whose summary has a completed step stop."""
    out = set()
    for x in state["arms"].values():
        if x.get("status") != "FAILED" or not x.get("results_dir"):
            continue
        try:
            spec = json.loads(Path(x["spec_path"]).read_text())
            summary = json.loads((Path(x["results_dir"]) / "summary.json").read_text())
        except (OSError, ValueError, KeyError):
            continue
        if spec.get("family") in CONTINUE_FAMILIES and any(
                str(e.get("label", "")).startswith("step") for e in summary.get("endpoints", [])):
            out.add(x["host"])
    return out


def cmd_refresh(a, state_path: Path) -> int:
    """Bootstrap the pinned payload on every drained host whose receipt is stale."""
    state = read_state(state_path)
    hosts = drained_stale_hosts(state)
    if not gate0_passed(state):  # amendment 3: leave freed cores to the GATE 0 SFT lane
        keep = continuable_hosts(state)
        for host in sorted(set(hosts) - keep):
            print(f"refresh: skip {host}: no continuable arm there (amendment 3, GATE 0 open)")
        hosts = [h for h in hosts if h in keep]
    low = low_disk_hosts(host_table(), hosts)
    for host in low:
        print(f"refresh: skip {host}: < {MIN_LOCAL_GB} GB free in /var/tmp")
    hosts = [h for h in hosts if h not in low]
    if not hosts:
        print("refresh: no drained host with a stale receipt")
        return 0
    return cmd_bootstrap(argparse.Namespace(payload=state["payload"]["dir"], host=hosts,
                                            repin=False), state_path)


def receipt_gpu(device: dict) -> str | None:
    """The GPU model class a receipt's device names (None for a CPU receipt)."""
    if device.get("kind") != "cuda":
        return None
    name = str(device.get("gpu", "")).replace(" ", "").lower()
    for model in ("3070", "2080ti", "3090", "4090", "a100", "a6000", "a4000", "a40", "l40"):
        # whole model number only: "a40" must not match an RTX A4000, nor "3070" a 3070ti
        if re.search(re.escape(model) + r"(?![0-9a-z])", name):
            return "rtx" + model if model[0].isdigit() else model
    return name or "unknown"


def valid_receipt(state: dict, host: str, features, spec: dict | None = None) -> dict:
    h = state["hosts"].get(host) or {}
    r = h.get("receipt")
    if not r or not r.get("valid"):
        raise SystemExit(f"REFUSE: {host} has no valid bootstrap receipt (run `receipts`)")
    if r["payload_sha256"] != state.get("payload", {}).get("sha256"):
        raise SystemExit(f"REFUSE: {host} receipt is for another payload")
    # parenthesised: `a | b - c` is `a | (b - c)`, which listed every declared feature as
    # missing, so no feature-gated (wave-2) arm could ever launch
    missing = sorted((set(features) | {"exploratory_sweep"}) - set(r["features"]))
    if missing:
        raise SystemExit(f"REFUSE: {host} trainer lacks features {missing}")
    if spec is not None:
        # The trainer feature "cuda" only says the code supports it; the host's device is
        # the receipt's. An arm runs only where its declared device (and GPU model) is.
        want = spec.get("device", "cpu")
        have = (r.get("device") or {}).get("kind", "cpu")
        if want != have:
            raise SystemExit(f"REFUSE: {spec['id']} declares {want}, {host} is {have}")
        if want == "cuda" and spec.get("gpu") != receipt_gpu(r.get("device") or {}):
            raise SystemExit(f"REFUSE: {spec['id']} declares GPU {spec.get('gpu')}, {host} "
                             f"has {receipt_gpu(r.get('device') or {})}")
    return r


# ---- launching ------------------------------------------------------------------------
def mem_gb(spec: dict):
    """uwrun --mem for this arm: measured peak of its kind (and the assay) plus 20%.
    An unmeasured kind (e.g. egsft) gets None: no per-run cap, never a guessed one (a cap
    too low kills the arm at its assay after hours of training); uwrun's host guard still
    applies, and the arm's summary.json `resources` then measures it. Refuses while the
    assay peak, which every kind includes, is unmeasured."""
    if ASSAY_PEAK_GIB is None:
        raise SystemExit("REFUSE: assay peak RSS is unmeasured; --mem cannot be sized")
    if spec.get("mode") == "eval":
        peak = ASSAY_PEAK_GIB
    else:
        mode = spec.get("mode")
        kind = mode if mode != "grpo" else f"grpo{int(spec.get('num_generations', 0))}"
        if kind not in PEAK_RSS_GIB:
            return None
        peak = max(PEAK_RSS_GIB[kind], ASSAY_PEAK_GIB)
    return math.ceil(MEM_HEADROOM * peak)


def capacity(state: dict, hosts: dict, host: str, threads: int,
             mem: float = ARM_RSS_GB, device: str = "cpu") -> int:
    """Arms of `threads` threads `host` can take now, keeping 25% of physical cores and memory."""
    t = hosts.get(host)
    if t is None:
        return 0
    ours = [x for x in state["arms"].values()
            if x.get("host") == host and x.get("status") in ("LAUNCHED", "RUNNING")]
    if device == "cuda" and any(x.get("device") == "cuda" for x in ours):
        return 0  # one GPU arm per host at a time (one GPU each)
    our_threads = sum(x.get("threads", 16) for x in ours)
    # load1 lags a fresh launch; count our own arms at full weight until load reflects them
    busy = max(t["load1"], our_threads)
    cores = HEADROOM * t["cores"] - busy  # physical cores, as uwrun v5 places
    recent = sum(x.get("mem_gb") or ARM_RSS_GB for x in ours
                 if time.time() - x.get("launched_ts", 0) < 900)
    free_mem = HEADROOM * t["avail_gb"] - recent
    return max(0, min(int(cores // threads), int(free_mem // mem)))


def control_spec(state: dict, ctl: dict) -> dict:
    rl = state["arms"].get(ctl["control_of"]) or {}
    summary_path = Path(rl.get("results_dir", "")) / "summary.json"
    receipt_path = Path(rl.get("results_dir", "")) / "endpoint_receipt.json"
    if not receipt_path.is_file():  # an RFT arm: arm_exec's derived endpoint receipt
        receipt_path = receipt_path.with_name("rft_endpoint_receipt.json")
    if rl.get("status") != "DONE" or not summary_path.is_file() or not receipt_path.is_file():
        raise SystemExit(f"REFUSE: control {ctl['id']} needs a finished {ctl['control_of']}")
    summary = json.loads(summary_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    last = [e for e in summary["endpoints"] if e["label"].startswith("step")][-1]
    ids = receipt["committed_prompt_ids"]
    steps = receipt["global_step"]
    # A prompts_per_update arm commits P prompt ids per global step (P = 1 otherwise).
    per_step = len(ids) // steps if steps else 0
    if not ids or per_step < 1 or len(ids) != steps * per_step:
        raise SystemExit(f"REFUSE: {ctl['control_of']} receipt does not match its last step")
    if ctl.get("control_replays") == "scanned":
        # CORRECTIONS C4: replay EVERY row the RL arm scored, in scan order (arm_exec's
        # scanned_ids.json). Missing is a refusal, never a fall back to the committed ids.
        scanned_path = summary_path.with_name("scanned_ids.json")
        if not scanned_path.is_file():
            raise SystemExit(f"REFUSE: control {ctl['id']} replays every scanned id; "
                             f"{ctl['control_of']} has no scanned_ids.json")
        scanned = json.loads(scanned_path.read_text())
        if scanned.get("global_step") != receipt["global_step"]:
            raise SystemExit(f"REFUSE: {ctl['control_of']} scanned ids are for another step")
        ids = scanned["scanned_prompt_ids"]
    if not ids or steps != last["steps"] or len(ids) > ctl["max_steps_total"] * per_step:
        raise SystemExit(f"REFUSE: {ctl['control_of']} receipt does not match its last step")
    spec = dict(ctl)
    # examples-matched: one SFT update per prompt the RL arm committed (or, for a
    # scanned-replay control, scored), in its order
    spec.update(train_ids=ids, stops=[len(ids)],
                time_budget_seconds=float(last["active_seconds"]))
    if per_step > 1:  # the SFT control's update cap is in images: P x the RL arm's steps
        spec["max_steps_total"] = ctl["max_steps_total"] * per_step
    return spec


def fresh_work_id(state: dict, spec: dict) -> dict:
    """A from-scratch relaunch of a retired arm gets its own work dir: the trainer refuses a
    non-empty output dir, and the old one may hold a killed run's partial checkpoints.
    Continuations (resume_stop) and promoted arms (work_id set) keep theirs."""
    if spec.get("resume_stop") or spec.get("work_id"):
        return spec
    tries = sum(1 for x in state["arms"].values() if x.get("retired_from") == spec["id"])
    return dict(spec, work_id=f"{spec['id']}~a{tries + 1}") if tries else spec


def launch_one(state_path: Path, spec: dict, host: str, root: str, timeout: int,
               dry: bool) -> str:
    mem = mem_gb(spec)
    spec = fresh_work_id(read_state(state_path), spec)
    name = safe_name(f"rlesweep-{spec['id']}-{int(time.time() * 1000) % 10**9}")
    launch_dir = SWEEP_DIR / "launch" / name
    launch_dir.mkdir(parents=True, exist_ok=False)
    (launch_dir / "arm.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    inner = (f"R={shlex.quote(root)}; exec \"$R/venv/bin/python\" -I \"$R/{ARM_EXEC}\" "
             f"--root \"$R\" --arm-json arm.json --results results")
    argv = ["run", "--host", host, "--threads", str(spec.get("threads", 16)), "--timeout",
            str(timeout), *(["--mem", str(mem)] if mem else []), "--sync", str(launch_dir),
            "--name", name, "--", "bash", "-c", inner]
    if dry:
        print("DRY", host, name, " ".join(shlex.quote(x) for x in argv[-3:]))
        return name
    uw(*argv, timeout=900)
    with state_lock(state_path) as state:
        state["arms"][spec["id"]] = {
            "status": "LAUNCHED", "host": host, "run": name, "launched_at": now(),
            "launched_ts": time.time(), "threads": spec.get("threads", 16), "mem_gb": mem,
            "device": spec.get("device", "cpu"),
            "spec_path": str(launch_dir / "arm.json"), "kind": spec.get("kind", "train"),
            "wave": spec.get("wave"), "rung": spec.get("rung"), "track": spec.get("track"),
            "family": spec.get("family"), "control_of": spec.get("control_of"),
            "receipt_payload_sha256": state["hosts"][host]["receipt"]["payload_sha256"]}
        state["uwrun_runs"].append({"name": name, "host": host, "kind": "arm",
                                    "arm": spec["id"], "at": now()})
    print(f"launched {spec['id']} on {host}: {name}")
    return name


# Plan amendment 3 (2026-10-18, exploratory, before any VAL use; RL lessons audit C8/C11):
# GRPO-from-W loses to I by 5-9 pp on dev800 and GATE 0 SFT is starved on our own hosts.
# The 22 running rung-1 arms finish and are recorded as exploratory. New launches are an
# ALLOWLIST: only grpo-I arms and their matched controls, and only once GATE 0 has passed
# (state["gate0"]["passed"] is True, written by whoever records GATE 0). `continue` is
# allowed only for the sftall families. No W-init unit is promoted. Freed cores stay idle
# for the GATE 0 SFT lane, so `refresh` bootstraps only hosts holding a continuable arm.
# Plan amendment 4 (arms-wave2-rung1.json): every wave-2 family starts from I and goes
# through the same GATE 0 check as grpo-I, matched controls and device evals included.
WAVE2_FAMILIES = {"wave2-fmtpen", "wave2-closedset", "wave2-drgrpo", "wave2-mu2",
                  "wave2-dapo", "wave2-guard", "wave2-rft", "device-control"}
LAUNCH_FAMILIES = {"grpo-I"} | WAVE2_FAMILIES
CONTINUE_FAMILIES = {"sftall-I", "sftall-W"}


def gate0_passed(state: dict) -> bool:
    return (state.get("gate0") or {}).get("passed") is True


def launch_hold(state: dict, spec: dict, table_by_id: dict):
    """None if amendment 3 allows launching `spec` now, else the reason it is held."""
    family = spec.get("family")
    if spec.get("track") == "control":
        family = (table_by_id.get(spec.get("control_of")) or {}).get("family")
    if family not in LAUNCH_FAMILIES:
        return f"held (amendment 3): family {family} is not on the launch allowlist"
    if not gate0_passed(state):
        return "held (amendment 3): GATE 0 has not passed"
    return None


def is_w_init(arm: dict) -> bool:
    init = arm.get("init")
    return init == "W" or (isinstance(init, dict) and str(init.get("arm", "")).find("-W") >= 0)


def continuation_spec(spec: dict, summary: dict) -> dict:
    """The rest of a FAILED multi-stop arm, resumed in place from its last completed stop.

    Same id, work dir and host; the completed endpoints are carried into the new summary.
    Refuses unless the completed `step` endpoints are exactly a proper prefix of the stops."""
    # a continuation keeps the arm's full stop list, so it can itself be continued
    stops = [int(x) for x in spec.get("stops_all") or spec.get("stops") or []]
    # the stop each endpoint was cut at (label stepN). Its `steps` must equal that stop, except
    # RFT, whose round-granular boundary may pass it (never fall short)
    ends = [e for e in summary.get("endpoints", []) if str(e.get("label", "")).startswith("step")]
    done = [int(str(e["label"])[4:]) for e in ends]
    rft = spec.get("mode") == "rft"
    reached = all(int(e["steps"]) >= d if rft else int(e["steps"]) == d
                  for e, d in zip(ends, done))
    if not reached or not done or done != stops[:len(done)] or len(done) >= len(stops):
        raise SystemExit(f"REFUSE: completed steps {done} are not a proper prefix of {stops}")
    cont = dict(spec)
    cont.update(stops=stops[len(done):], stops_all=stops, resume_stop=done[-1],
                save_steps=arm_exec_module().save_steps_for(spec, int(spec["max_steps_total"])),
                work_id=spec.get("work_id", spec["id"]),
                prior_endpoints=summary["endpoints"])
    return cont


def cmd_continue(a, state_path: Path) -> int:
    """Retire a FAILED arm's record (ARM#failN) and relaunch its remaining stops in place."""
    state = read_state(state_path)
    x = state["arms"].get(a.arm) or {}
    if x.get("status") != "FAILED":
        raise SystemExit(f"REFUSE: {a.arm} is {x.get('status')}, not FAILED")
    spec = json.loads(Path(x["spec_path"]).read_text())
    if spec.get("family") not in CONTINUE_FAMILIES:
        raise SystemExit(f"REFUSE (amendment 3): only {sorted(CONTINUE_FAMILIES)} continue; "
                         f"{a.arm} is {spec.get('family')}")
    summary = json.loads((Path(x["results_dir"]) / "summary.json").read_text())
    cont = continuation_spec(spec, summary)
    cont["continued_from_run"] = x["run"]
    host = x["host"]
    r = valid_receipt(state, host, cont.get("features", []), cont)
    if capacity(state, host_table(), host, cont.get("threads", 16),
                mem_gb(cont) or ARM_RSS_GB, cont.get("device", "cpu")) < 1 and not a.dry_run:
        raise SystemExit(f"REFUSE: {host} has no free capacity now; retry later")
    if a.dry_run:
        launch_one(state_path, cont, host, r["root"], a.timeout, True)
        return 0
    with state_lock(state_path) as st:
        n = 1
        while f"{a.arm}#fail{n}" in st["arms"]:
            n += 1
        retired = f"{a.arm}#fail{n}"
        rec = st["arms"].pop(a.arm)
        rec.update(retired_reason=a.reason, retired_at=now(), retired_from=a.arm,
                   kind="retired", continued_by=True)
        st["arms"][retired] = rec
    try:
        launch_one(state_path, cont, host, r["root"], a.timeout, False)
    except BaseException:
        # the launch never happened: put the FAILED record back, so a plain `launch` cannot
        # treat the arm as new and restart it from scratch over its kept checkpoints
        with state_lock(state_path) as st:
            if a.arm not in st["arms"]:
                rec = st["arms"].pop(retired)
                for k in ("retired_reason", "retired_at", "retired_from", "kind",
                          "continued_by"):
                    rec.pop(k, None)
                rec["kind"] = x.get("kind", "train")
                st["arms"][a.arm] = rec
        raise
    return 0


def cmd_launch(a, state_path: Path) -> int:
    table = json.loads(Path(a.table).read_text())
    if table.get("study_id") != STUDY:
        raise SystemExit("REFUSE: arm table is for another study")
    state = read_state(state_path)
    wanted = set(a.ids.split(",")) if a.ids else None
    arms = [x for x in table["arms"]
            if (wanted is None or x["id"] in wanted)
            and (a.wave is None or x["wave"] == a.wave)
            and (a.rung is None or x["rung"] == a.rung)
            and state["arms"].get(x["id"], {}).get("status") not in
            ("LAUNCHED", "RUNNING", "DONE")]
    if wanted is not None and len(arms) != len(wanted):
        missing = wanted - {x["id"] for x in arms}
        raise SystemExit(f"REFUSE: ids absent, finished or running: {sorted(missing)}")
    hosts = host_table()
    allowed = a.hosts.split(",") if a.hosts else sorted(
        h for h, v in state["hosts"].items() if (v.get("receipt") or {}).get("valid"))
    if not allowed:
        raise SystemExit("REFUSE: no host has a valid receipt; run bootstrap and receipts")
    launched = 0
    skipped = []
    by_id = {x["id"]: x for x in table["arms"]}
    for spec in arms:
        if launched >= a.max_launch:
            break
        held = launch_hold(state, spec, by_id)
        if held:
            skipped.append(f"{spec['id']}: {held}")
            continue
        if spec["track"] == "control":
            try:
                spec = control_spec(read_state(state_path), spec)
            except SystemExit as exc:
                skipped.append(str(exc))
                continue
            host = read_state(state_path)["arms"][spec["control_of"]]["host"]
            choices = [host]
        elif spec.get("same_host_as"):
            parent = read_state(state_path)["arms"].get(spec["same_host_as"]) or {}
            if parent.get("status") != "DONE":
                skipped.append(f"{spec['id']}: waits for {spec['same_host_as']}")
                continue
            choices = [parent["host"]]
        else:
            choices = allowed
        st = read_state(state_path)
        best = None
        for host in choices:
            try:
                r = valid_receipt(st, host, spec.get("features", []), spec)
            except SystemExit as exc:
                # a control waits for its RL arm's host (refreshed once drained); skipping
                # it still refuses that launch without aborting every other arm's
                skipped.append(str(exc))
                continue
            free = capacity(st, hosts, host, spec.get("threads", 16),
                            mem_gb(spec) or ARM_RSS_GB, spec.get("device", "cpu"))
            if free > 0 and (best is None or free > best[1]):
                best = (host, free, r["root"])
        if best is None:
            skipped.append(f"{spec['id']}: no host with free capacity")
            continue
        launch_one(state_path, spec, best[0], best[2], a.timeout, a.dry_run)
        launched += 1
    for line in skipped:
        print("SKIP", line)
    print(f"launched {launched} of {len(arms)} eligible arms")
    return 0


def cmd_bench(a, state_path: Path) -> int:
    base = json.loads((DESIGN / "arms-rung1.json").read_text())
    template = next(x for x in base["arms"] if x["family"] == "sft-W" and x["lr"] == 1e-5)
    spec = dict(template)
    mode, g = {"grpo4": ("grpo", 4), "grpo8": ("grpo", 8), "sft": ("sft", 4)}[a.kind]
    spec.update(id=f"bench-{a.kind}-{a.host}-t{a.threads}-{int(time.time())}", kind="bench",
                track="bench", family="bench", mode=mode, num_generations=g,
                bench_steps=a.steps, threads=a.threads, stops=None, max_active_seconds=None)
    if mode == "grpo":
        spec.update(beta=0.04, temperature=1.0, screen_threshold=None)
    st = read_state(state_path)
    r = valid_receipt(st, a.host, [])
    launch_one(state_path, spec, a.host, r["root"], a.timeout, a.dry_run)
    return 0


# ---- polling --------------------------------------------------------------------------
def cmd_poll(a, state_path: Path) -> int:
    state = read_state(state_path)
    for arm_id, x in sorted(state["arms"].items()):
        if x.get("status") not in ("LAUNCHED", "RUNNING"):
            continue
        status = run_status(x["run"])
        update = {"run_status": status, "polled_at": now()}
        if "RUNNING" in status:
            update["status"] = "RUNNING"
        elif " DONE " in f" {status} ":
            dest = SWEEP_DIR / "results" / safe_name(arm_id)
            if dest.exists():
                dest = dest.with_name(dest.name + f"-{int(time.time())}")
            # trailing slash: rsync copies the CONTENTS of work/results into dest (without it
            # the files land in dest/results and summary.json is never found)
            uw("fetch", x["run"], "work/results/", str(dest), timeout=900, check=False)
            summary = dest / "summary.json"
            markers = ("oom-guard", "timedout", "overquota", "stopped")
            ok = ("rc=0" in status and summary.is_file()
                  and not any(m in status for m in markers))
            update.update(status="DONE" if ok else "FAILED", results_dir=str(dest),
                          finished_at=now())
            if not ok:
                log = uw("logs", x["run"], timeout=180, check=False)
                update["log_tail"] = log[-1500:]
                update["failure_reason"] = failure_reason(status, log)
        elif "LOST" in status or "UNREACH" in status:
            update["status"] = "LOST" if "LOST" in status else x["status"]
        with state_lock(state_path) as st:
            st["arms"][arm_id].update(update)
            retired = retire_if_host_oom(st, arm_id)
        print(f"{arm_id}: {status}" + (f" {update['failure_reason']}"
                                        if update.get("failure_reason") else "")
              + (f" -> retired as {retired}, relaunches from scratch" if retired else ""))
    return 0


GUARD_LINE = re.compile(r"memory guard: the run holds (\d+) MB \(--mem (\d+) GB\), "
                        r"host RAM free (\d+) MB \(floor (\d+) MB\)")
OOM_RETRIES = 2


def failure_reason(status: str, log: str) -> str:
    """oom-guard:host-ram (the host ran short: infrastructure), oom-guard:mem-cap (the arm
    outgrew its --mem: our sizing, never auto-relaunched), oom-guard:unparsed, timedout,
    overquota, stopped, or code (anything else, e.g. a trainer refusal)."""
    if "oom-guard" in status:
        m = GUARD_LINE.search(log or "")
        if m is None:
            return "oom-guard:unparsed"
        rss, cap = int(m.group(1)), int(m.group(2))
        return "oom-guard:mem-cap" if cap > 0 and rss > cap * 1024 else "oom-guard:host-ram"
    for marker in ("timedout", "overquota", "stopped"):
        if marker in status:
            return marker
    return "code"


def retire_if_host_oom(st: dict, arm_id: str):
    """A host-RAM oom-guard kill is infrastructure: retire the record (ARM#failN) so the
    launch loop relaunches the arm from scratch in a fresh work dir, at most OOM_RETRIES
    times. An arm that already completed a proper prefix of its stops stays FAILED for
    `runner.py continue` instead (its kept checkpoint is worth resuming)."""
    x = st["arms"][arm_id]
    if x.get("status") != "FAILED" or x.get("failure_reason") != "oom-guard:host-ram":
        return None
    if sum(1 for y in st["arms"].values() if y.get("retired_from") == arm_id
           and str(y.get("failure_reason", "")).startswith("oom-guard")) >= OOM_RETRIES:
        return None
    try:
        summary = json.loads((Path(x["results_dir"]) / "summary.json").read_text())
        if any(str(e.get("label", "")).startswith("step") for e in summary.get("endpoints", [])):
            return None
    except (OSError, ValueError, KeyError):
        pass
    n = 1
    while f"{arm_id}#fail{n}" in st["arms"]:
        n += 1
    x.update(retired_reason="oom-guard:host-ram (infrastructure); relaunch from scratch",
             retired_at=now(), retired_from=arm_id, kind="retired")
    st["arms"][f"{arm_id}#fail{n}"] = st["arms"].pop(arm_id)
    return f"{arm_id}#fail{n}"


# ---- promotion (successive halving; the rule is pre-registered in the plan) ------------
def last_step_accuracy(state: dict, arm_id: str):
    x = state["arms"].get(arm_id) or {}
    if x.get("status") != "DONE":
        return None
    summary = json.loads((Path(x["results_dir"]) / "summary.json").read_text())
    steps = [e for e in summary.get("endpoints", []) if e["label"].startswith("step")]
    if not steps:
        return None
    m = steps[-1]["metrics"]
    return (m["accuracy"], m["equal_class_accuracy"], m["valid_fraction"])


def control_for(rl: dict, control_lr: dict) -> dict:
    keep = ("init", "seed", "threads", "device", "dtype", "gpu", "device_class", "max_pixels",
            "min_pixels", "max_completion_length", "adapter_profile", "num_generations", "wave",
            "rung", "max_steps_total")
    ctl = {k: rl[k] for k in keep if k in rl}
    lr = control_lr.get(rl["init"], 5e-6) if isinstance(rl["init"], str) else 5e-6
    if rl.get("matched_control_lr"):  # wave 2: each family's controls span its LR count
        lr = float(rl["matched_control_lr"])
    extra = {k: rl[k] for k in ("control_replays", "control_response_format") if rl.get(k)}
    ctl.update(id=f"{rl['id']}~ctl", track="control", family="control", mode="sft", lr=lr,
               control_of=rl["id"], depends_on=[rl["id"]], same_host_as=rl["id"],
               order="sequential", cycle=True, train_manifest=f"@consumed:{rl['id']}",
               # wave 2: a closed-set RL arm's control trains and assays on the same prompt
               response_format=extra.get("control_response_format", "answer_only"),
               beta=None, temperature=None,
               screen_threshold=None, features=[f for f in rl.get("features", [])
                                                if f == "cuda"],
               max_active_seconds=SFT_ACTIVE_CAP_SECONDS,
               # a scanned-replay control trains on every scored row, more than RL updates
               max_steps_total=int(rl.get("control_max_steps_total", rl["max_steps_total"])))
    if "control_replays" in extra:
        ctl["control_replays"] = extra["control_replays"]
    return ctl


SFT_ACTIVE_CAP_SECONDS = 259200.0


def arm_exec_module():
    """arm_exec.py beside this file, so the runner reuses its save_steps rule, not a copy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("arm_exec", HERE.parent / "arm_exec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def promoted_arms(unit: str, xs: list, track: str, r: int, table: dict) -> list:
    seeds_next = table["rung_seeds"][str(r + 1)]
    updates_next = table["rung_updates"][str(r + 1)]
    base = next(x for x in xs if x["seed"] == 42)
    arms = []
    for seed in seeds_next:
        prev = next((x for x in xs if x["seed"] == seed), None)
        arm = dict(base)
        arm.update(id=f"{unit}@r{r + 1}-s{seed}", unit=unit, rung=r + 1, seed=seed,
                   updates=updates_next, stops=[updates_next])
        arm.pop("resume_stop", None)
        arm.pop("save_steps", None)
        if prev is not None:  # continue this seed's own trajectory in place
            # save_steps is in the trainer's resume identity: keep the trajectory's own value
            arm.update(work_id=prev.get("work_id", prev["id"]),
                       resume_stop=prev["stops"][-1], same_host_as=prev["id"],
                       train_manifest=prev["train_manifest"],
                       save_steps=arm_exec_module().save_steps_for(
                           prev, int(prev["max_steps_total"])))
        else:
            arm.update(work_id=arm["id"], same_host_as=None,
                       train_manifest=base["train_manifest"].replace("seed42", f"seed{seed}"))
        arms.append(arm)
        if track == "rl":
            arms.append(control_for(arm, table["control_lr"]))
    return arms


def cmd_promote(a, state_path: Path) -> int:
    """Rung r -> r+1 (pre-registered rule, docs/aircraft-exploratory-sweep-v2.md).

    Within each track (sft, rl) rank units by dev800 accuracy at their last step endpoint
    (rung 2+: mean over seeds; ties: equal-class accuracy, then valid fraction); rung 1 keeps
    the top third (ceil), rung 2 the top 3. Only units that were launched take part; every
    launched arm must have finished; a FAILED arm ranks last and is never promoted.
    --late (after a promotion exists): units that finished since, and were not ranked then,
    are promoted if they score >= the lowest unit promoted in their track; their arms are
    appended to the existing rung table.
    """
    state = read_state(state_path)
    r = a.from_rung
    src = DESIGN / "arms-rung1.json" if r == 1 else SWEEP_DIR / f"arms-rung{r}.json"
    table = json.loads(src.read_text())
    if r == 1 and (DESIGN / WAVE2_TABLE).is_file():
        # The concrete wave-2 arms (plan amendment 4) rank in the same tracks at rung 1.
        table["arms"] = table["arms"] + json.loads((DESIGN / WAVE2_TABLE).read_text())["arms"]
    units = {}
    for x in table["arms"]:
        if x["track"] in ("sft", "rl") and x["rung"] == r:
            units.setdefault(x.get("unit", x["id"]), []).append(x)
    # amendment 3: no W-init unit is promoted (they are recorded, exploratory, not ranked)
    held_w = sorted(u for u, xs in units.items() if any(is_w_init(x) for x in xs))
    units = {u: xs for u, xs in units.items() if u not in held_w}
    entered = {u: xs for u, xs in units.items()
               if all(x["id"] in state["arms"] for x in xs)}
    pending = [x["id"] for xs in entered.values() for x in xs
               if state["arms"][x["id"]].get("status") not in ("DONE", "FAILED")]
    if pending:
        raise SystemExit(f"REFUSE: {len(pending)} launched rung-{r} arms unfinished, "
                         f"e.g. {pending[:3]}")
    scored = {}
    for unit, xs in entered.items():
        vals = [last_step_accuracy(state, x["id"]) for x in xs]
        ok = [v for v in vals if v is not None]
        scored[unit] = (tuple(sum(v[i] for v in ok) / len(ok) for i in range(3))
                        if ok and len(ok) == len(xs) else None)
    key = f"{r}->{r + 1}"
    path = SWEEP_DIR / f"arms-rung{r + 1}.json"
    previous = state.get("promotions", {}).get(key)
    if a.late:
        if previous is None:
            raise SystemExit(f"REFUSE: --late needs an existing {key} promotion")
        ranked_before = set(previous["ranked"])
        added = {}
        for track in ("sft", "rl"):
            floor = min((tuple(previous["scores"][u]) for u in previous["kept"][track]),
                        default=None)
            added[track] = sorted(
                u for u, m in scored.items()
                if u not in ranked_before and entered[u][0]["track"] == track
                and m is not None and floor is not None and m >= floor)
        out = json.loads(path.read_text())
        for track, us in added.items():
            for u in us:
                out["promotion"][track].append(u)
                if int(entered[u][0]["max_steps_total"]) < int(table["rung_updates"][str(r + 1)]):
                    out.setdefault("infeasible", []).append(u)  # plan amendment 2
                    continue
                out["arms"] += promoted_arms(u, entered[u], track, r, table)
        path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        with state_lock(state_path) as st:
            p = st["promotions"][key]
            p["ranked"] = sorted(set(p["ranked"]) | set(scored))
            p["scores"].update({u: list(m) for u, m in scored.items() if m is not None})
            for track, us in added.items():
                p["kept"][track] += us
            p.setdefault("late", []).append({"at": now(), "added": added})
        print(json.dumps({"late_added": added, "table": str(path)}, indent=1))
        return 0
    if path.exists() and not a.overwrite:
        raise SystemExit(f"REFUSE: {path} exists; pass --overwrite to re-derive it")
    keep, arms, infeasible = {}, [], []
    for track in ("sft", "rl"):
        ranked = sorted((u for u in scored if entered[u][0]["track"] == track),
                        key=lambda u: (scored[u] is not None, scored[u] or ()), reverse=True)
        n = math.ceil(len(ranked) / 3) if r == 1 else min(3, len(ranked))
        keep[track] = [u for u in ranked[:n] if scored[u] is not None]
        for u in keep[track]:
            # plan amendment 2: a unit whose declared total is below the next rung's budget
            # (screened arms, 600) is recorded infeasible there and not replaced
            if int(entered[u][0]["max_steps_total"]) < int(table["rung_updates"][str(r + 1)]):
                infeasible.append(u)
                continue
            arms += promoted_arms(u, entered[u], track, r, table)
    out = dict(table, rung=r + 1, arms=arms, promoted_from=str(src),
               promotion={t: list(keep[t]) for t in keep}, infeasible=infeasible,
               held_w_init=held_w, created_at=now())
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    with state_lock(state_path) as st:
        st.setdefault("promotions", {})[key] = {
            "table": str(path), "kept": keep, "ranked": sorted(scored),
            "infeasible": infeasible,
            "scores": {u: list(m) for u, m in scored.items() if m is not None},
            "not_entered": sorted(set(units) - set(entered)), "at": now()}
    print(json.dumps({"rung": r + 1, "kept": keep, "arms": len(arms), "table": str(path)},
                     indent=1))
    return 0


# ---- data-scale derivations (pre-registered in the plan, "Data scale") ----------------
def cmd_derive(a, state_path: Path) -> int:
    """sftall-seeds: seeds 314 and 2718 of every sftall arm (full budget, rung 1 done).
    sftall-rl: GRPO from the best sftall endpoint by dev800 accuracy (either epoch),
    LR {1e-6, 5e-6, 1e-5}, G 8, beta 0.04, T 1.0 on s_train, each with a matched SFT
    control from the same endpoint (LR 5e-6), all on the host that holds the endpoint."""
    state = read_state(state_path)
    table = json.loads((DESIGN / "arms-rung1.json").read_text())
    sftall = [x for x in table["arms"] if x["track"] == "sftall"]
    unfinished = [x["id"] for x in sftall
                  if (state["arms"].get(x["id"]) or {}).get("status") != "DONE"]
    if unfinished:
        raise SystemExit(f"REFUSE: sftall rung-1 arms not DONE: {unfinished}")
    arms = []
    if a.what == "sftall-seeds":
        for x in sftall:
            for seed in (314, 2718):
                arm = dict(x)
                arm.update(id=f"{x['id']}-s{seed}", unit=x["id"], seed=seed,
                           train_manifest=f"all_train_scan_seed{seed}.jsonl")
                arms.append(arm)
    else:
        best = None
        for x in sftall:
            summary = json.loads((Path(state["arms"][x["id"]]["results_dir"])
                                  / "summary.json").read_text())
            for e in summary["endpoints"]:
                m = e["metrics"]
                score = (m["accuracy"], m["equal_class_accuracy"], m["valid_fraction"])
                if best is None or score > best[0]:
                    best = (score, x, e["label"])
        score, parent, label = best
        template = next(x for x in table["arms"] if x["family"] == "grpo-I")
        for lr in (1e-6, 5e-6, 1e-5):
            arm = dict(template)
            arm.update(id=f"sftall2rl-{parent['id']}-{label}-lr{lr:.0e}-g8".replace("e-0", "e-"),
                       family="sftall-rl", track="rl", init={"arm": parent["id"],
                                                              "label": label},
                       lr=lr, num_generations=8, beta=0.04, temperature=1.0,
                       screen_threshold=None, same_host_as=parent["id"],
                       derived_from={"arm": parent["id"], "label": label,
                                     "dev800": list(score)})
            arms.append(arm)
            arms.append(control_for(arm, table["control_lr"]))
    out = dict(table, rung=1, arms=arms, derived=a.what, created_at=now())
    path = SWEEP_DIR / f"arms-{a.what}.json"
    if path.exists() and not a.overwrite:
        raise SystemExit(f"REFUSE: {path} exists; pass --overwrite to re-derive it")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    with state_lock(state_path) as st:
        st.setdefault("derived", {})[a.what] = {"table": str(path), "arms": len(arms),
                                               "at": now()}
    print(json.dumps({"derived": a.what, "arms": [x["id"] for x in arms],
                      "table": str(path)}, indent=1))
    return 0


# ---- report ---------------------------------------------------------------------------
class Binomial:
    """Exact binomial draws by inverse CDF (tables cached per n); stdlib only."""

    def __init__(self, rng: random.Random):
        self.rng, self.tables = rng, {}

    def draw(self, n: int, p: float) -> int:
        if n == 0 or p <= 0:
            return 0
        if p >= 1:
            return n
        key = (n, p)
        cdf = self.tables.get(key)
        if cdf is None:
            logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log1p(-p) for k in range(n + 1)]
            total, cdf = 0.0, []
            for v in logs:
                total += math.exp(v)
                cdf.append(total)
            self.tables[key] = cdf
        return min(bisect.bisect_left(cdf, self.rng.random() * cdf[-1]), n)


def paired_bootstrap(a: dict, b: dict, gold: dict, resamples: int, seed: int) -> dict:
    """Accuracy(a) - accuracy(b) over the same items, resampled jointly (multinomial)."""
    d = [(a[i] == g) - (b[i] == g) for i, g in gold.items()]
    n = len(d)
    plus, minus = d.count(1), d.count(-1)
    rng = random.Random(seed)
    binom = Binomial(rng)
    deltas = []
    for _ in range(resamples):
        x = binom.draw(n, plus / n)
        rest = n - x
        y = binom.draw(rest, minus / (n - plus)) if n > plus else 0
        deltas.append((x - y) / n)
    deltas.sort()
    lo, hi = deltas[int(0.025 * resamples)], deltas[int(0.975 * resamples) - 1]
    return {"delta": (plus - minus) / n, "ci95": [lo, hi], "n": n, "resamples": resamples,
            "p_le_0": sum(v <= 0 for v in deltas) / resamples, "wins": plus, "losses": minus}


def cmd_report(a, state_path: Path) -> int:
    state = read_state(state_path)
    gold = {}
    for line in (DESIGN / "manifests" / "dev800.jsonl").read_text().splitlines():
        r = json.loads(line)
        gold[r["id"]] = r["label"]

    def endpoint(arm_id, label=None):
        x = state["arms"].get(arm_id) or {}
        if x.get("status") != "DONE":
            return None
        s = json.loads((Path(x["results_dir"]) / "summary.json").read_text())
        eps = s.get("endpoints") or []
        ep = next((e for e in eps if e["label"] == label), None) if label else (
            eps[-1] if eps else None)
        if ep is None:
            return None
        doc = json.loads((Path(x["results_dir"]) / "assays" / f"{ep['label']}.json").read_text())
        if set(doc["predictions"]) != set(gold):
            raise SystemExit(f"REFUSE: {arm_id} assay is not over dev800")
        return ep, doc["predictions"]

    rows, comparisons = [], []
    for arm_id, x in sorted(state["arms"].items()):
        if x.get("kind") == "bench" or x.get("status") != "DONE":
            continue
        found = endpoint(arm_id)
        if found is None:
            continue
        ep, pred = found
        m = ep["metrics"]
        rows.append({"arm": arm_id, "track": x.get("track"), "label": ep["label"],
                     "steps": ep.get("steps"), "accuracy": m["accuracy"],
                     "valid_fraction": m["valid_fraction"],
                     "equal_class_accuracy": m["equal_class_accuracy"]})
        ctl_id = f"{arm_id}~ctl"
        if x.get("track") in ("rl", "noise") and ctl_id in state["arms"]:
            for label in ("step", "time"):
                ctl = None
                cx = state["arms"][ctl_id]
                if cx.get("status") == "DONE":
                    s = json.loads((Path(cx["results_dir"]) / "summary.json").read_text())
                    labels = [e["label"] for e in s["endpoints"]]
                    pick = [x for x in labels if x.startswith(label)]
                    ctl = endpoint(ctl_id, pick[-1]) if pick else None
                if ctl is None:
                    continue
                comparisons.append({"rl": arm_id, "control": ctl_id, "matched": label,
                                    **paired_bootstrap(pred, ctl[1], gold, a.resamples,
                                                       seed=20261018)})
    out = {"study_id": STUDY, "exploratory": True, "holdout": "dev800", "created_at": now(),
           "arms": sorted(rows, key=lambda r: -r["accuracy"]), "rl_vs_matched_sft": comparisons}
    path = SWEEP_DIR / "report.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for r in out["arms"]:
        print(f"{r['accuracy']:.4f} {r['valid_fraction']:.3f} {r['equal_class_accuracy']:.4f} "
              f"{r['track']:>8} {r['arm']} [{r['label']}]")
    for c in comparisons:
        print(f"{c['rl']} vs {c['matched']}-matched SFT: {c['delta']:+.4f} "
              f"CI95 [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--state", type=Path, default=DEFAULT_STATE)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--payload", required=True)
    b.add_argument("--host", action="append", default=[],
                   help="none: only (re)pin the payload; drained hosts follow via `refresh`")
    b.add_argument("--repin", action="store_true")
    sub.add_parser("receipts")
    sub.add_parser("refresh")
    co = sub.add_parser("continue")
    co.add_argument("arm")
    co.add_argument("--reason", required=True)
    co.add_argument("--timeout", type=int, default=86400)
    co.add_argument("--dry-run", action="store_true")
    ln = sub.add_parser("launch")
    ln.add_argument("--table", default=str(DESIGN / "arms-rung1.json"))
    ln.add_argument("--ids")
    ln.add_argument("--wave", type=int)
    ln.add_argument("--rung", type=int)
    ln.add_argument("--hosts")
    ln.add_argument("--max-launch", type=int, default=1000)
    ln.add_argument("--timeout", type=int, default=86400)
    ln.add_argument("--dry-run", action="store_true")
    be = sub.add_parser("bench")
    be.add_argument("--host", required=True)
    be.add_argument("--kind", choices=("grpo4", "grpo8", "sft"), required=True)
    be.add_argument("--steps", type=int, default=6)
    be.add_argument("--threads", type=int, default=16)
    be.add_argument("--timeout", type=int, default=14400)
    be.add_argument("--dry-run", action="store_true")
    sub.add_parser("poll")
    dv = sub.add_parser("derive")
    dv.add_argument("what", choices=("sftall-seeds", "sftall-rl"))
    dv.add_argument("--overwrite", action="store_true")
    pr = sub.add_parser("promote")
    pr.add_argument("--from-rung", type=int, choices=(1, 2), required=True)
    pr.add_argument("--late", action="store_true")
    pr.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--resamples", type=int, default=20000)
    a = p.parse_args(argv)
    fn = {"bootstrap": cmd_bootstrap, "receipts": cmd_receipts, "refresh": cmd_refresh,
          "continue": cmd_continue, "launch": cmd_launch, "bench": cmd_bench,
          "poll": cmd_poll, "report": cmd_report,
          "promote": cmd_promote, "derive": cmd_derive}[a.cmd]
    return fn(a, a.state)


if __name__ == "__main__":
    sys.exit(main())
