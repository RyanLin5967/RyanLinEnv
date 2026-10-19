"""RL campaign from I (SFT -> RL): queue arms on UW hosts, derive matched SFT controls, report.

Exploratory, TRAIN-derived holdouts only: every checkpoint is scored on dev800 (the sweep's
h_select + dev600). Nothing here reads, lists or copies native VAL or the sealed TEST.
Goal (docs/aircraft-rl-campaign-I/README.md): an RL recipe started FROM I (sft-1e5-time,
8f27d0ca...) that beats I continued with SFT for the same measured compute.

Mac side, light (Python 3.9 stdlib). Everything heavy runs through ~/.claude/bin/uwrun.
It reuses, never re-implements:
  * the GPU lane's queue/worker model: gpu_lane.push (host-local pending/ queue),
    gpu_lane.cmd_fetch / arm_dirs, gpu_lane.continue_spec (carried save_steps), and
    gpu_worker.sh (now also WAIT_KIND=cpu with a free-core wait, SLOT=N, RSS trace);
  * the CPU sweep runner's payload pin / receipts / state lock / host table / control rule
    (runner.state_lock, host_table, low_disk_hosts, safe_name, control_for, Binomial) and
    the payload builder + bootstrap_host.sh / host_verify.py (own root and receipts);
  * arm_exec.py on the host (stops, dev800 assays, time stops, save_steps_for).

  campaign.py bootstrap --payload DIR --host H [--host H ...] [--repin]
  campaign.py receipts                      # read receipts, assign device classes
  campaign.py enqueue --table T [--ids A,B] [--host H] [--prio NN]
  campaign.py worker --host H [--slot N] [--mem GB] [--timeout S]
  campaign.py queue --host H
  campaign.py fetch                         # copy every worker's results, update arm status
  campaign.py controls [--of ARM]           # time-matched SFT-from-I control per RL arm
  campaign.py report [--resamples 20000]
  campaign.py promote --table T --rung R    # successive halving by dev800 delta-vs-control

State: runs/aircraft-rl-campaign-I/state.json in the root checkout (runner.state_lock: fcntl
lock, atomic rewrite, other writers' keys kept). Never the CPU sweep's or GPU lane's state.

Root corrections (runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md), enforced here or in arm_exec:
  C2  GRPO only in an fp32 class; a GRPO stage whose sampler-teacher parity is not within
      tolerance refuses (arm_exec.check_parity). bf16/fp16 only for SFT and eval.
  M6  one device class (and host) per trajectory: continuations and controls run where the
      arm ran; a class change refuses.
  C4/C10 controls: k control LRs (default 5e-6, 1e-5, 2e-5) paired with a group's k RL LRs;
      trained on every row the RL arm read (scored / drawn), in scan order; time-matched at
      EACH RL checkpoint; `pool_scoring_seconds` is charged to the RL side.
  C5  dynamic sampling needs --dynamic-sampling-max-draws >= 8; dropped draws are generated
      inside the update span, so they are in the RL arm's active seconds.
  C10/M5/C11 promotion: best-of-k vs best-of-k (same k); the bar is the max placebo delta
      (SFT-from-I vs SFT-from-I at another seed, same code); fixed rung_stops, no early stop.
  C16 per-stage co-tenancy (arm_exec.CoTenancy: foreign GPU process, or CPU load1 > own
      threads + 1); a time comparison with different
      co-tenancy refuses.
  C19 a continuation is quoted only after the device class's E0 (kind e0: N uninterrupted
      vs N/2+N/2 resumed, against the N vs N run-to-run floor) passed.

Refusals (non-zero exit, nothing launched): a host the GPU lane owns or with no AVX2; a host
without a valid receipt for the pinned payload; an arm whose init is not I; a stop that is not
a multiple of the trajectory's save_steps; a CPU worker with no measured peak RSS and no --mem;
a report entry with no same-device I baseline; promotion with unfinished arms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
SWEEP_SCRIPTS = WT / "scripts" / "exploratory_sweep"
STUDY = "aircraft-rl-campaign-I"
ROOT_CHECKOUT = Path("/Users/idide/RyanLinEnv")
CAMPAIGN_DIR = ROOT_CHECKOUT / "runs" / STUDY
STATE = CAMPAIGN_DIR / "state.json"
DOCS = WT / "docs" / STUDY
GOLD = WT / "docs" / "aircraft-exploratory-sweep-v2" / "manifests" / "dev800.jsonl"
ROOT_NAME = "r226lin-rlcampaign"  # host-local root (bootstrap_host.sh RLE_ROOT_NAME)
RECEIPTS = "~/rlcampaign/receipts"
BOOT_SCRIPT = "code/scripts/exploratory_sweep/bootstrap_host.sh"
WORKER = "payload/code/scripts/exploratory_sweep/gpu_worker.sh"
SEED = 20261018
I_SHA256 = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"

# Hosts this campaign never uses: the GPU lane's cards, no-AVX2 hosts, a full root disk.
FORBIDDEN = {"ecetesla2": "GPU lane", "ecetesla4": "GPU lane", "eceubuntu2": "GPU lane",
             "iccad1": "no AVX2", "iccad2": "no AVX2", "iccad3": "no AVX2",
             "iccad4": "root disk full", "eceubuntu1": "shared login box"}
# Device classes: comparisons (vs I, vs control) stay inside one class, preferably one host.
DEVICE_CLASSES = {
    "cpu-fp32": {"device": "cpu", "dtype": "float32", "wait": "cpu", "threads": 16,
                 "gpu": None},
    "gtx1070-fp32": {"device": "cuda", "dtype": "float32", "wait": "gpu", "threads": 4,
                     "gpu": "GTX 1070", "need_mib": 7600, "mem_gb": 24,
                     # measured 2026-10-18 (uwrun run rlc-memprobe-ecetesla3-1792359646): 2.209 B
                     # fp32 weights need 8426 MiB; the card has 8107 MiB; .to("cuda") OOMs at
                     # 7850 MiB allocated, before any generation. G4 and G8 cannot fit.
                     "infeasible": "fp32 Qwen2-VL-2B weights (8426 MiB) exceed the 8107 MiB card"},
    "gtx1070-fp16": {"device": "cuda", "dtype": "float16", "wait": "gpu", "threads": 4,
                     "gpu": "GTX 1070", "need_mib": 7600, "mem_gb": 24},
}
# Every arm's identity apart from the per-arm fields; the same values as the CPU sweep and
# GPU lane (gpu_lane.BASE), so I's profile is unchanged.
BASE = {"adapter_profile": "legacy-qv-v1", "max_completion_length": 256, "max_pixels": 200704,
        "min_pixels": 50176, "response_format": "answer_only", "seed": 42, "wave": 1,
        "rung": 1, "init": "I", "train_manifest": "s_train_scan_seed42.jsonl",
        "num_generations": 8, "temperature": 1.0, "beta": 0.04, "screen_threshold": None}
CONTROL_LR = 5e-6  # CPU sweep / GPU lane rule: SFT control LR 5e-6 from I
CONTROL_MAX_STEPS = 100000
CONTROL_ACTIVE_CAP = 259200.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("sweep_runner", SWEEP_SCRIPTS / "runner.py")
lane = _load("gpu_lane", SWEEP_SCRIPTS / "gpu_lane.py")
arm_exec = _load("arm_exec", SWEEP_SCRIPTS / "arm_exec.py")


def configure(state_path: Path = STATE, campaign_dir: Path = CAMPAIGN_DIR) -> None:
    """Point the reused modules at this campaign: its study id, state, dir and host classes.

    runner.state_lock stamps runner.STUDY; gpu_lane.push writes ROOT/gpuq on the host and
    calls gpu_lane.save_state; gpu_lane.cmd_fetch reads gpu_lane.STATE["workers"] and copies
    into gpu_lane.STUDY_DIR/results. All of them now name this campaign only."""
    runner.STUDY = STUDY
    runner.RECEIPTS = RECEIPTS
    lane.STATE = state_path
    lane.STUDY_DIR = campaign_dir
    lane.load_state = lambda: _read(state_path)

    def save(state: dict) -> None:  # gpu_lane.push hands back its whole dict; keep its key
        with runner.state_lock(state_path) as st:
            st.setdefault("queued", {}).update(state.get("queued", {}))
    lane.save_state = save


def _read(state_path: Path) -> dict:
    state = runner.read_state(state_path)
    for key in ("workers", "queued", "measured"):
        state.setdefault(key, {})
    return state


# ---- hosts ------------------------------------------------------------------------------
def refuse_hosts(hosts) -> None:
    bad = {h: FORBIDDEN[h] for h in hosts if h in FORBIDDEN}
    if bad:
        raise SystemExit(f"REFUSE: hosts this campaign may not use: {bad}")


def device_classes_for(receipt: dict) -> list:
    """Device classes a host's receipt supports. A CPU host needs AVX-512F: training on an
    AVX2-only host dies in MKL with SIGILL (root, 2026-10-19), so it supports no class."""
    device = receipt.get("device") or {}
    if device.get("kind") == "cpu":
        return ["cpu-fp32"] if receipt.get("avx2") and receipt.get("avx512f") else []
    gpu = device.get("gpu") or ""
    return sorted(c for c, v in DEVICE_CLASSES.items() if v["gpu"] and v["gpu"] in gpu)


def cmd_bootstrap(a, state_path: Path) -> int:
    refuse_hosts(a.host)
    payload = Path(a.payload).resolve()
    raw = (payload / "PAYLOAD.json").read_bytes()
    doc = json.loads(raw)
    if doc.get("code_dirty"):
        raise SystemExit("REFUSE: payload was staged from a dirty tree; commit, then restage")
    if doc["files"].get("adapters/I/final_adapter/adapter_model.safetensors") != I_SHA256:
        raise SystemExit("REFUSE: payload's I adapter is not 8f27d0ca...")
    low = runner.low_disk_hosts(runner.host_table(), a.host)
    if low:
        raise SystemExit(f"REFUSE: {low} have < {runner.MIN_LOCAL_GB} GB free in /var/tmp")
    sha = hashlib.sha256(raw).hexdigest()
    with runner.state_lock(state_path) as st:
        pinned = st.get("payload", {}).get("sha256")
        if pinned and pinned != sha and not a.repin:
            raise SystemExit(f"REFUSE: state pins payload {pinned[:12]}; pass --repin to move")
        live = sorted({x.get("host") for x in st["arms"].values()
                       if x.get("status") == "QUEUED"} & set(a.host))
        if live and pinned != sha:
            raise SystemExit(f"REFUSE: {live} have queued arms of the pinned payload")
        st["payload"] = {"sha256": sha, "dir": str(payload), "code_head": doc["code_head"]}
    for host in a.host:
        name = runner.safe_name(f"rlc-boot-{host}-{doc['code_head'][:7]}-{int(time.time())}")
        inner = (f"RLE_ROOT_NAME={ROOT_NAME} RLE_RECEIPTS=$HOME/rlcampaign/receipts "
                 f"exec bash {BOOT_SCRIPT}")
        runner.uw("run", "--host", host, "--threads", "8", "--mem", "16", "--timeout", "5400",
                  "--sync", str(payload), "--name", name, "--", "bash", "-c", inner,
                  timeout=1800)
        with runner.state_lock(state_path) as st:
            st["hosts"].setdefault(host, {}).update(
                boot_run=name, boot_payload_sha256=sha, boot_started=runner.now(), receipt=None)
            st["uwrun_runs"].append({"name": name, "host": host, "kind": "bootstrap",
                                     "at": runner.now()})
        print(f"bootstrap {host}: {name}")
    return 0


def cmd_receipts(a, state_path: Path) -> int:
    state = _read(state_path)
    bad = 0
    for host, h in sorted(state["hosts"].items()):
        status = runner.run_status(h["boot_run"]) if h.get("boot_run") else "NONE"
        raw = runner.uw("ssh", host, f"cat {RECEIPTS}/{host}.json", timeout=180, check=False)
        try:
            receipt = json.loads(raw)
        except json.JSONDecodeError:
            receipt = None
        ok = (receipt is not None and "DONE rc=0" in status
              and receipt["payload"]["payload_sha256"] == state.get("payload", {}).get("sha256")
              and receipt["root"].endswith(ROOT_NAME))
        summary = None
        if receipt is not None:
            summary = {"root": receipt["root"], "nproc": receipt["nproc"],
                       "payload_sha256": receipt["payload"]["payload_sha256"],
                       "features": receipt["features"]["available"], "device": receipt["device"],
                       "cpu_model": receipt["cpu_model"], "avx2": receipt.get("avx2"),
                       "avx512f": receipt.get("avx512f"),
                       "classes": device_classes_for(receipt), "valid": ok}
        with runner.state_lock(state_path) as st:
            st["hosts"][host].update(boot_status=status, receipt=summary)
        print(f"{host}: {status} receipt={'OK' if ok else 'MISSING/STALE'} "
              f"classes={summary and summary['classes']}")
        bad += not ok
    return 1 if bad else 0


def valid_host(state: dict, host: str, device_class: str) -> dict:
    refuse_hosts([host])
    r = runner.valid_receipt(state, host, [])
    if device_class not in r.get("classes", []):
        raise SystemExit(f"REFUSE: {host} does not support {device_class}")
    return r


# ---- arm table ----------------------------------------------------------------------------
def load_table(path: Path) -> dict:
    table = json.loads(Path(path).read_text())
    if table.get("study_id") != STUDY:
        raise SystemExit("REFUSE: arm table is for another study")
    return table


def build_spec(table: dict, arm: dict) -> dict:
    """The full arm_exec spec for one table row (defaults <- table defaults <- row)."""
    spec = dict(BASE)
    spec.update(table.get("defaults", {}))
    spec.update(arm)
    cls = DEVICE_CLASSES.get(spec.get("device_class"))
    if cls is None:
        raise SystemExit(f"REFUSE: {spec['id']}: unknown device_class {spec.get('device_class')}")
    if cls.get("infeasible"):
        raise SystemExit(f"REFUSE: {spec['id']}: {spec['device_class']} is infeasible: "
                         f"{cls['infeasible']}")
    if spec["init"] != "I":
        raise SystemExit(f"REFUSE: {spec['id']}: this campaign starts every arm from I")
    spec.update(device=cls["device"], dtype=cls["dtype"], threads=cls["threads"],
                features=["cuda"] if cls["device"] == "cuda" else [])
    spec.setdefault("track", "eval" if spec.get("mode") == "eval" else "rl")
    spec.setdefault("family", spec["track"])
    if spec.get("mode") == "eval":
        return spec
    if spec.get("mode") not in ("grpo", "sft", "egsft"):
        raise SystemExit(f"REFUSE: {spec['id']}: mode must be grpo, sft, egsft or eval")
    extra = [str(x) for x in spec.get("extra_trainer_args", [])]
    if spec["mode"] == "grpo" and float(spec["beta"]) == 0 and "--loss-type" not in extra:
        # the merged trainer refuses beta=0 GRPO without an explicit loss_type; "grpo" is the
        # aggregation every earlier beta=0 arm ran with
        extra += ["--loss-type", "grpo"]
    spec["extra_trainer_args"] = extra
    if spec["mode"] == "grpo" and spec["dtype"] != "float32":
        # C2: bf16 GRPO failed sampler-teacher parity on every 3070 arm; RL only in fp32
        raise SystemExit(f"REFUSE: {spec['id']}: GRPO runs only in an fp32 device class, "
                         f"not {spec['device_class']}")
    arm_exec.refuse_spec(spec)  # the host-side rules (fp32 RL, dynamic sampling >= 8)
    if spec["track"] in ("rl", "placebo"):
        # C11: a fixed rung schedule declared before any result; stops are its rung's stops
        schedule = table.get("rung_stops") or {}
        want = schedule.get(str(spec.get("rung", 1)))
        if not want or [int(x) for x in spec["stops"]] != [int(x) for x in want]:
            raise SystemExit(f"REFUSE: {spec['id']}: stops must equal the table's fixed "
                             f"rung_stops[{spec.get('rung', 1)}] = {want}")
        if spec["track"] == "rl" and not spec.get("group"):
            raise SystemExit(f"REFUSE: {spec['id']}: an RL arm needs a `group` (best-of-k unit)")
    stops = [int(x) for x in spec["stops"]]
    total = int(spec["max_steps_total"])
    save = arm_exec.save_steps_for(spec, total)
    if not stops or stops != sorted(set(stops)) or stops[-1] > total:
        raise SystemExit(f"REFUSE: {spec['id']}: stops must increase and stay <= max_steps_total")
    if any(x % save for x in stops):
        raise SystemExit(f"REFUSE: {spec['id']}: stops must be multiples of save_steps {save}")
    spec["save_steps"] = save  # carried through every continuation (resume identity)
    return spec


def cmd_enqueue(a, state_path: Path) -> int:
    table = load_table(a.table)
    wanted = set(a.ids.split(",")) if a.ids else None
    rows = [x for x in table["arms"] if wanted is None or x["id"] in wanted]
    if wanted is not None and {x["id"] for x in rows} != wanted:
        raise SystemExit(f"REFUSE: unknown ids {sorted(wanted - {x['id'] for x in rows})}")
    lrs = assign_control_lrs(table)
    for i, row in enumerate(rows):
        spec = build_spec(table, row)
        spec.update(lrs.get(spec["id"], {}))
        state = _read(state_path)
        if spec["id"] in state["arms"]:
            print(f"skip {spec['id']}: already {state['arms'][spec['id']]['status']}")
            continue
        host = a.host or spec.get("host") or pick_host(state, spec["device_class"])
        r = valid_host(state, host, spec["device_class"])
        push(state_path, host, spec, f"{int(a.prio) + i:02d}", r["root"])
    return 0


def assign_control_lrs(table: dict) -> dict:
    """{arm id: {"control_lr", "control_seed"}} for every RL / placebo row (C4, C10, M5).

    RL group: its k arms (k distinct LRs) are paired, by LR rank, with the table's k
    `control_lrs` (default {5e-6, 1e-5, 2e-5}): best-of-k RL is then compared with best-of-k
    SFT with the same k. A group whose size differs from len(control_lrs) refuses.
    Placebo group: SFT-from-I arms; each one's "control" is the same SFT at
    `placebo_control_seed` (same LR), so the identical comparison/promotion code measures
    how large a delta pure seed noise produces."""
    control_lrs = sorted(float(x) for x in table.get("control_lrs", (5e-6, 1e-5, 2e-5)))
    defaults = table.get("defaults", {})
    groups = {}
    for row in table["arms"]:
        r = dict(defaults, **row)
        if r.get("track") in ("rl", "placebo"):
            groups.setdefault((r["track"], r.get("group") or r["id"]), []).append(r)
    out = {}
    for (track, group), rows in sorted(groups.items()):
        if track == "placebo":
            for r in rows:
                if r.get("mode") != "sft" or "placebo_control_seed" not in r:
                    raise SystemExit(f"REFUSE: placebo {r['id']} must be SFT with a "
                                     "placebo_control_seed")
                if int(r["placebo_control_seed"]) == int(r.get("seed", BASE["seed"])):
                    raise SystemExit(f"REFUSE: placebo {r['id']} control seed equals its seed")
                out[r["id"]] = {"control_lr": float(r["lr"]),
                                "control_seed": int(r["placebo_control_seed"])}
            continue
        if len(rows) != len(control_lrs) or len({float(r["lr"]) for r in rows}) != len(rows):
            raise SystemExit(f"REFUSE: group {group} has {len(rows)} arms / LRs; best-of-k needs "
                             f"exactly k = {len(control_lrs)} distinct LRs (control_lrs)")
        for r, lr in zip(sorted(rows, key=lambda r: float(r["lr"])), control_lrs):
            out[r["id"]] = {"control_lr": lr}
    return out


def pick_host(state: dict, device_class: str) -> str:
    """The valid host of this class with the fewest queued arms (ties: name)."""
    hosts = [h for h, v in state["hosts"].items()
             if (v.get("receipt") or {}).get("valid") and device_class in v["receipt"]["classes"]
             and h not in FORBIDDEN]
    if not hosts:
        raise SystemExit(f"REFUSE: no host with a valid receipt for {device_class}")
    load = {h: sum(1 for x in state["arms"].values()
                   if x.get("host") == h and x.get("status") == "QUEUED") for h in hosts}
    return min(hosts, key=lambda h: (load[h], h))


def push(state_path: Path, host: str, spec: dict, prio: str, root: str, suffix: str = "",
         kind: str | None = None) -> None:
    """gpu_lane.push into this host's campaign root, then record the arm."""
    lane.ROOT = root
    lane.DEVICE_CLASS = {host: spec["device_class"]}
    state = _read(state_path)
    lane.push(host, spec, prio, state, suffix=suffix)
    qid = spec["id"] + suffix
    with runner.state_lock(state_path) as st:
        st["arms"][qid] = {"status": "QUEUED", "host": host, "key": f"{prio}-{qid}",
                           "device_class": spec["device_class"], "queued_at": runner.now(),
                           "kind": kind or spec.get("track"), "spec": spec,
                           "control_of": spec.get("control_of")}


# ---- workers ------------------------------------------------------------------------------
def worker_mem_gb(state: dict, device_class: str, given: float | None) -> float:
    """uwrun --mem: --mem when given, else the class's measured peak RSS + 20%."""
    if given is not None:
        return float(given)
    cls = DEVICE_CLASSES[device_class]
    measured = (state.get("measured") or {}).get(device_class, {}).get("peak_rss_gb")
    if measured:
        return math.ceil(measured * 1.2 * 10) / 10
    if cls["device"] == "cuda":
        return float(cls["mem_gb"])
    raise SystemExit(f"REFUSE: no measured peak RSS for {device_class}; pass --mem (run a "
                     "smoke arm first: fetch records the worker's peak)")


def cmd_worker(a, state_path: Path) -> int:
    state = _read(state_path)
    h = state["hosts"].get(a.host) or {}
    classes = (h.get("receipt") or {}).get("classes") or []
    if len(classes) == 0:
        raise SystemExit(f"REFUSE: {a.host} has no valid receipt/device class")
    device_class = classes[0]
    valid_host(state, a.host, device_class)
    cls = DEVICE_CLASSES[device_class]
    mem = worker_mem_gb(state, device_class, a.mem)
    root = h["receipt"]["root"]
    env = {"WAIT_KIND": cls["wait"], "ARM_TIMEOUT": str(a.arm_timeout),
           "IDLE_MAX": str(a.idle_max)}
    if cls["wait"] == "gpu":
        env["NEED_MIB"] = str(cls["need_mib"])
    else:
        # free physical cores required before an arm starts (default: the class's threads);
        # lower it only on a host with fewer physical cores than threads (recorded in state)
        env["NEED_CORES"] = str(a.need_cores if a.need_cores is not None else cls["threads"])
    if a.slot is not None:
        env["SLOT"] = str(a.slot)
    name = runner.safe_name(f"rlc-worker-{a.host}-s{a.slot or 0}-{int(time.time())}")
    inner = (f"R={shlex.quote(root)}; " + " ".join(f"{k}={v}" for k, v in sorted(env.items()))
             + f" exec bash \"$R/{WORKER}\" \"$R\" \"$PWD/results\"")
    runner.uw("run", "--host", a.host, "--threads", str(cls["threads"]), "--mem", f"{mem:g}",
              "--timeout", str(a.timeout), "--name", name, "--", "bash", "-c", inner,
              timeout=600)
    with runner.state_lock(state_path) as st:
        st.setdefault("workers", {}).setdefault(a.host, []).append(
            {"name": name, "at": runner.now(), "slot": a.slot, "mem_gb": mem,
             "need_cores": env.get("NEED_CORES"),
             "threads": cls["threads"], "device_class": device_class})
        st["uwrun_runs"].append({"name": name, "host": a.host, "kind": "worker",
                                 "at": runner.now()})
    print(f"worker {name} on {a.host} ({device_class}, --mem {mem:g})")
    return 0


def cmd_queue(a, state_path: Path) -> int:
    root = (_read(state_path)["hosts"].get(a.host, {}).get("receipt") or {}).get("root")
    if not root:
        raise SystemExit(f"REFUSE: {a.host} has no receipt")
    print(runner.uw("ssh", a.host, f"cd {root}/gpuq && for d in pending running* done failed; "
                    "do echo \"$d: $(ls $d 2>/dev/null | tr '\\n' ' ')\"; done; cat /proc/loadavg;"
                    " nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader "
                    "2>/dev/null || true", timeout=180))
    return 0


# ---- fetch: status, s/update, memory peaks ---------------------------------------------
def arm_measurements(d: Path) -> dict:
    """s/update and memory peaks. Time comes from the summary's step endpoints (each one's
    committed CUMULATIVE active seconds across resumed stages); the optimizer audit's
    active_seconds restarts at every resumed stage, so it is not used for time."""
    out = {}
    s = d / "summary.json"
    steps = step_endpoints(json.loads(s.read_text())) if s.is_file() else []
    steps = [e for e in steps if e.get("active_seconds") is not None]
    if steps:
        pts = [(int(e["steps"]), float(e["active_seconds"])) for e in steps]
        arm = json.loads((d / "arm.json").read_text()) if (d / "arm.json").is_file() else {}
        if not arm.get("resume_stop"):  # a continuation's earlier checkpoints are elsewhere
            pts = [(0, 0.0)] + pts
        per = [(t1 - t0) / (n1 - n0) for (n0, t0), (n1, t1) in zip(pts, pts[1:]) if n1 > n0]
        out.update(updates=pts[-1][0], active_seconds=pts[-1][1],
                   s_per_update_mean=(pts[-1][1] - pts[0][1]) / max(1, pts[-1][0] - pts[0][0]),
                   s_per_update_by_checkpoint=[round(x, 2) for x in per])
    w = d / "worker.json"
    if w.is_file():
        doc = json.loads(w.read_text())
        out.update(rc=doc["rc"], wall_seconds=doc["end"] - doc["start"])
        if doc.get("peak_rss_kb"):
            out["peak_rss_gb"] = round(doc["peak_rss_kb"] / 1024**2, 2)
    trace = d / "gpu-trace.csv"
    if trace.is_file():
        used = [int(x.split(",")[1].split()[0]) for x in trace.read_text().splitlines()
                if x.count(",") >= 2 and x.split(",")[1].split()[0].isdigit()]
        if used:
            out["gpu_mem_peak_mib"] = max(used)
    stages = d / "stages.json"
    if stages.is_file():
        co = [s.get("cotenancy") for s in json.loads(stages.read_text()) if s.get("cotenancy")]
        if co:
            out["cotenancy_shared"] = any(c.get("shared") for c in co)
            out["cotenant_gpu_processes"] = sorted({p for c in co
                                                    for p in c.get("cotenant_gpu_processes", {})})
    lt = d / "local_training.json"
    if lt.is_file():
        rec = json.loads(lt.read_text())
        cuda = rec.get("cuda_memory") or (rec.get("record") or {}).get("cuda_memory")
        if cuda:
            out["cuda_max_allocated_gib"] = round(cuda["max_memory_allocated_bytes"] / 2**30, 2)
    return out


def cmd_fetch(a, state_path: Path) -> int:
    lane.cmd_fetch(a)
    results = {}
    for arm_id, d in lane.arm_dirs():
        w = d / "worker.json"
        if w.is_file():
            results.setdefault(arm_id, []).append(d)
    with runner.state_lock(state_path) as st:
        measured = st.setdefault("measured", {})
        for qid, x in st["arms"].items():
            key = x.get("key")
            dirs = [d for d in results.get(x["spec"]["id"], []) if d.name == key] if key else []
            if not dirs or x.get("status") != "QUEUED":
                continue
            d = dirs[-1]
            m = arm_measurements(d)
            ok = m.get("rc") == 0 and (d / "summary.json").is_file()
            x.update(status="DONE" if ok else "FAILED", results_dir=str(d),
                     finished_at=runner.now(), measurements=m)
            if (d / "summary.json").is_file():
                e0 = json.loads((d / "summary.json").read_text()).get("e0")
                if e0 is not None:  # C19: continuations are quotable only after this passes
                    st.setdefault("e0", {})[x["device_class"]] = dict(
                        e0, arm=qid, host=x["host"], at=runner.now())
            if m.get("peak_rss_gb"):
                cls = measured.setdefault(x["device_class"], {"peak_rss_gb": 0.0, "from": []})
                if m["peak_rss_gb"] > cls["peak_rss_gb"]:
                    cls.update(peak_rss_gb=m["peak_rss_gb"], from_arm=qid)
                cls["from"] = sorted(set(cls["from"]) | {qid})
            print(f"{qid}: {x['status']} {json.dumps(m)[:300]}")
    return 0


# ---- controls -------------------------------------------------------------------------------
def step_endpoints(summary: dict) -> list:
    return [e for e in summary.get("endpoints", []) if str(e["label"]).startswith("step")]


def control_spec(rl: dict, summary: dict, receipt: dict, lr: float | None = None,
                 scanned: list | None = None) -> dict:
    """SFT from I, time-matched at EVERY RL checkpoint (root corrections C4, C10).

    runner.control_for builds the template (same init/seed/device/profile, sequential +
    cycling, SFT). The control trains on `scanned`: every row whose label the RL arm read
    (screened: every SCORED candidate; dynamic sampling: every drawn group, kept or dropped;
    arm_exec writes them per checkpoint under scanned/), in scan order; without it, on the
    committed prompts. One active-time stop per RL checkpoint: the RL arm's committed active
    seconds (which already hold screen scoring and dropped dynamic-sampling generations,
    both timed inside the update span) PLUS the arm's declared `pool_scoring_seconds` (an
    offline pool score that chose its rows, C10). LR: the arm's paired `control_lr`
    (assign_control_lrs); seed: `control_seed` for a placebo pair. Note: the cycled sequence
    holds every row read by the LAST checkpoint, so an early time stop can see later rows
    (favours the control: conservative for an RL win)."""
    steps = step_endpoints(summary)
    if rl.get("mode") in ("grpo", "rft"):
        if not scanned:
            raise SystemExit(f"REFUSE: {rl['id']}: no scanned/ rows for its last checkpoint; a "
                             "control on committed prompts alone would see fewer labels")
        ids = list(scanned)
    else:  # an SFT arm (placebo) reads exactly its committed rows
        ids = list(receipt.get("committed_prompt_ids") or [])
    if not steps or not ids or receipt["global_step"] != steps[-1]["steps"]:
        raise SystemExit(f"REFUSE: {rl['id']} receipt does not match its last step endpoint")
    lr = float(lr if lr is not None else rl.get("control_lr", CONTROL_LR))
    ctl = runner.control_for(rl, {"I": lr})
    for k in ("same_host_as", "depends_on"):
        ctl.pop(k, None)
    pool = float(rl.get("pool_scoring_seconds") or 0.0)
    times = [float(e["active_seconds"]) + pool for e in steps]
    ctl.update(device_class=rl["device_class"], train_ids=ids, stops=[], time_stops=times,
               max_steps_total=CONTROL_MAX_STEPS, save_steps=CONTROL_MAX_STEPS,
               max_active_seconds=max(CONTROL_ACTIVE_CAP, 2 * times[-1]), lr=lr,
               matched_steps=[int(e["steps"]) for e in steps], extra_trainer_args=[],
               pool_scoring_seconds_charged=pool, track="control", rung=rl.get("rung", 1),
               group=rl.get("group"), placebo=rl.get("track") == "placebo")
    if rl.get("control_seed") is not None:
        ctl["seed"] = int(rl["control_seed"])
    return ctl


def cmd_controls(a, state_path: Path) -> int:
    state = _read(state_path)
    made = 0
    merged = collect(lane.arm_dirs())
    for qid, x in sorted(state["arms"].items()):
        if x.get("status") != "DONE" or x.get("kind") not in ("rl", "smoke", "placebo"):
            continue
        if a.of and qid != a.of:
            continue
        rung = int(x["spec"].get("rung", 1))
        suffix = "" if rung == 1 else f"@r{rung}"
        ctl_id = x["spec"]["id"] + "~ctl" + suffix
        if ctl_id in state["arms"]:
            continue
        d = Path(x["results_dir"])
        receipt = json.loads((d / "endpoint_receipt.json").read_text())
        # every checkpoint of the trajectory (a continuation's dir holds only its own stops)
        eps = merged.get(x["spec"]["id"], {}).get("endpoints") or {}
        summary = {"endpoints": sorted(eps.values(), key=lambda e: e.get("steps") or 0)}
        last = step_endpoints(summary)[-1]["label"] if step_endpoints(summary) else None
        scan = d / "scanned" / f"{last}.json"
        scanned = json.loads(scan.read_text())["ids"] if scan.is_file() else None
        ctl = control_spec(x["spec"], summary, receipt, scanned=scanned)
        ctl["id"] = ctl_id
        r = valid_host(state, x["host"], x["device_class"])  # same host, same device class
        push(state_path, x["host"], ctl, a.prio, r["root"], kind="control")
        made += 1
    print(f"queued {made} controls")
    return 0


# ---- report ---------------------------------------------------------------------------------
def stratified_bootstrap(a: dict, b: dict, gold: dict, resamples: int, seed: int = SEED) -> dict:
    """Paired, class-stratified bootstrap of accuracy(a) - accuracy(b).

    Within each gold class the items are resampled with replacement (n_c draws), so every
    replicate keeps dev800's class counts; the pair (a_i, b_i) moves together. Per class the
    counts of +1 / -1 differences are drawn exactly (runner.Binomial), as runner's unstratified
    paired_bootstrap does over the whole set."""
    by = {}
    for i, g in gold.items():
        by.setdefault(g, []).append((a[i] == g) - (b[i] == g))
    n = len(gold)
    rng = random.Random(seed)
    binom = runner.Binomial(rng)
    classes = [(len(d), d.count(1), d.count(-1)) for _, d in sorted(by.items())]
    deltas = []
    for _ in range(resamples):
        total = 0
        for nc, plus, minus in classes:
            x = binom.draw(nc, plus / nc)
            y = binom.draw(nc - x, minus / (nc - plus)) if nc > plus else 0
            total += x - y
        deltas.append(total / n)
    deltas.sort()
    plus = sum(c[1] for c in classes)
    minus = sum(c[2] for c in classes)
    return {"delta": (plus - minus) / n, "ci95": [deltas[int(0.025 * resamples)],
                                                   deltas[int(0.975 * resamples) - 1]],
            "p_le_0": sum(v <= 0 for v in deltas) / resamples, "n": n, "classes": len(classes),
            "resamples": resamples, "wins": plus, "losses": minus, "stratified": "gold class"}


def load_gold() -> dict:
    return {r["id"]: r["label"] for r in map(json.loads, GOLD.read_text().splitlines())}


def collect(results_dirs) -> dict:
    """{arm id: {"spec", "host", "assays": {label: doc}, "endpoints": {label: endpoint},
    "cotenancy_shared"}}; a later dir (continuation, re-assay) adds or replaces labels, and the
    spec of the latest dir wins (a continuation carries its rung and resume_stop)."""
    out = {}
    for arm_id, d in results_dirs:
        if not (d / "summary.json").is_file():
            continue
        summary = json.loads((d / "summary.json").read_text())
        rec = out.setdefault(arm_id, {"host": summary["host"].split(".")[0].lower(),
                                      "assays": {}, "endpoints": {}, "cotenancy_shared": False,
                                      "continued": False, "stage_shared": {}, "rows_read": {}})
        rec["spec"] = json.loads((d / "arm.json").read_text())
        rec["continued"] = rec["continued"] or bool(rec["spec"].get("resume_stop"))
        for e in summary["endpoints"]:
            rec["endpoints"][e["label"]] = e
        for f in sorted((d / "assays").glob("*.json")):
            rec["assays"][f.stem] = json.loads(f.read_text())
        if (d / "stages.json").is_file():
            for st in json.loads((d / "stages.json").read_text()):
                shared = bool((st.get("cotenancy") or {}).get("shared"))
                rec["cotenancy_shared"] = rec["cotenancy_shared"] or shared
                stage = str(st.get("stage", ""))
                if stage.startswith("train-"):  # train-50 -> step50, train-time2 -> time2
                    x = stage[len("train-"):]
                    rec["stage_shared"][x if x.startswith("time") else f"step{x}"] = shared
        for f in sorted((d / "scanned").glob("*.json")):
            rec["rows_read"][f.stem] = len(json.loads(f.read_text())["ids"])
    return out


def control_of(arms: dict, arm_id: str):
    """The arm's latest-rung matched control (id ARM~ctl or ARM~ctl@rN), or None."""
    found = [(int(r["spec"].get("rung", 1)), cid) for cid, r in arms.items()
             if cid == arm_id + "~ctl" or cid.startswith(arm_id + "~ctl@r")]
    return arms[max(found)[1]] if found else None


def control_label(ctl: dict, steps: int):
    """The control's time label matched to RL checkpoint `steps` (its spec's matched_steps)."""
    matched = [int(x) for x in ctl["spec"].get("matched_steps") or []]
    return f"time{matched.index(int(steps)) + 1}" if int(steps) in matched else None


def build_report(arms: dict, gold: dict, resamples: int, e0_passed=()) -> dict:
    for arm_id, rec in arms.items():
        for label, doc in rec["assays"].items():
            if set(doc["predictions"]) != set(gold):
                raise SystemExit(f"REFUSE: {arm_id}/{label} is not over dev800")
    base_host, base_class = {}, {}
    for arm_id, rec in sorted(arms.items()):
        if rec["spec"].get("mode") == "eval" and "init" in rec["assays"]:
            base_host[rec["host"]] = (arm_id, rec["assays"]["init"])
            base_class.setdefault(rec["spec"]["device_class"], (arm_id, rec["assays"]["init"]))
    out = {"study_id": STUDY, "exploratory": True, "holdout": "dev800", "resamples": resamples,
           "seed": SEED, "created_at": runner.now(), "bootstrap": "paired, class-stratified",
           "baselines": {}, "checkpoints": [], "groups": [], "missing": [], "refused": []}
    for key, (arm_id, doc) in sorted(base_host.items()):
        out["baselines"][f"host/{key}"] = {"arm": arm_id, "accuracy": doc["metrics"]["accuracy"]}
    for key, (arm_id, doc) in sorted(base_class.items()):
        out["baselines"][f"class/{key}"] = {"arm": arm_id, "accuracy": doc["metrics"]["accuracy"]}
    for arm_id, rec in sorted(arms.items()):
        spec = rec["spec"]
        if spec.get("mode") == "eval" or spec.get("track") == "control":
            continue
        if rec["continued"] and spec["device_class"] not in e0_passed:
            out["refused"].append(f"{arm_id}: continuation on {spec['device_class']} without a "
                                  "passed E0 resume check (C19)")
            continue
        base = base_host.get(rec["host"]) or base_class.get(spec["device_class"])
        if base is None:
            out["missing"].append(f"{arm_id}: no I baseline on {rec['host']} or "
                                  f"{spec['device_class']}")
            continue
        ctl = control_of(arms, arm_id)
        steps = sorted((e for e in rec["endpoints"].values() if e["label"].startswith("step")),
                       key=lambda e: e["steps"])
        for e in steps:
            doc = rec["assays"].get(e["label"])
            if doc is None:
                continue  # an endpoint kept without an assay (assay: false)
            entry = {"arm": arm_id, "device_class": spec["device_class"], "host": rec["host"],
                     "label": e["label"], "steps": e["steps"], "group": spec.get("group"),
                     "track": spec.get("track"), "rung": spec.get("rung", 1),
                     "active_seconds": e.get("active_seconds"),
                     "accuracy": doc["metrics"]["accuracy"],
                     "valid_fraction": doc["metrics"]["valid_fraction"],
                     "equal_class_accuracy": doc["metrics"]["equal_class_accuracy"],
                     "I_baseline": base[0], "I_accuracy": base[1]["metrics"]["accuracy"],
                     "vs_I": stratified_bootstrap(doc["predictions"], base[1]["predictions"],
                                                  gold, resamples)}
            label = control_label(ctl, e["steps"]) if ctl is not None else None
            if label is not None and label in ctl["assays"]:
                if ctl["spec"]["device_class"] != spec["device_class"]:
                    raise SystemExit(f"REFUSE: {arm_id} and its control ran on different "
                                     "device classes")
                cdoc, cep = ctl["assays"][label], ctl["endpoints"][label]
                entry["control"] = {"arm": ctl["spec"]["id"], "label": label,
                                    "accuracy": cdoc["metrics"]["accuracy"],
                                    "steps": cep.get("steps"),
                                    "active_seconds": cep.get("active_seconds"),
                                    "cotenancy_shared": ctl["cotenancy_shared"]}
                rl_sh = rec["stage_shared"].get(e["label"], rec["cotenancy_shared"])
                ctl_sh = ctl["stage_shared"].get(label, ctl["cotenancy_shared"])
                # rows the RL arm read by this checkpoint (scanned/), else its committed steps
                if spec.get("mode") in ("grpo", "rft"):
                    read = rec["rows_read"].get(e["label"], -1)  # arm_exec writes scanned/
                else:  # an SFT (placebo) arm reads one row per step
                    read = int(e["steps"])
                if rl_sh != ctl_sh:
                    # C16, stage by stage: the RL stage that made this checkpoint and the
                    # control stage that made its time match must share co-tenancy
                    out["refused"].append(f"{arm_id} {e['label']}: co-tenancy differs from "
                                          f"its control (rl {rl_sh}, ctl {ctl_sh})")
                elif read < 0:
                    out["refused"].append(f"{arm_id} {e['label']}: no scanned/ rows recorded")
                elif int(cep.get("steps") or 0) < read:
                    # the control must have seen at least as many rows as the RL arm read
                    out["refused"].append(f"{arm_id} {e['label']}: control saw "
                                          f"{cep.get('steps')} rows < {read} the RL arm read")
                else:
                    entry["vs_control"] = stratified_bootstrap(
                        doc["predictions"], cdoc["predictions"], gold, resamples)
                    entry["_pred"], entry["_cpred"] = doc["predictions"], cdoc["predictions"]
            else:
                out["missing"].append(f"{arm_id} {e['label']}: no matched control endpoint")
            out["checkpoints"].append(entry)
    out["groups"] = best_of_k(out["checkpoints"], gold, resamples)
    for e in out["checkpoints"]:
        e.pop("_pred", None)
        e.pop("_cpred", None)
    return out


def best_of_k(entries: list, gold: dict, resamples: int) -> list:
    """Per group (an RL group, or one placebo pair): best-of-k RL checkpoint vs best-of-k
    matched-control checkpoint by dev800 accuracy, with the same k on both sides (C10): every
    RL candidate counts only when its matched control endpoint exists. The delta of the two
    selected endpoints is bootstrapped (paired, class-stratified)."""
    groups = {}
    for e in entries:
        if "vs_control" not in e:
            continue
        key = (e["track"], e.get("group") or e["arm"], int(e.get("rung", 1)))
        groups.setdefault(key, []).append(e)
    out = []
    for (track, group, rung), es in sorted(groups.items()):
        rl = max(es, key=lambda e: (e["accuracy"], e["equal_class_accuracy"]))
        ctl = max(es, key=lambda e: (e["control"]["accuracy"], e["control"]["label"]))
        out.append({"track": track, "group": group, "rung": rung, "k": len(es),
                    "arms": sorted({e["arm"] for e in es}),
                    "best_rl": {"arm": rl["arm"], "label": rl["label"],
                                "accuracy": rl["accuracy"]},
                    "best_control": {"arm": ctl["control"]["arm"],
                                     "label": ctl["control"]["label"],
                                     "accuracy": ctl["control"]["accuracy"]},
                    "best_vs_best": stratified_bootstrap(rl["_pred"], ctl["_cpred"], gold,
                                                         resamples)})
    return out


def cmd_report(a, state_path: Path) -> int:
    gold = load_gold()
    e0 = {c for c, v in (_read(state_path).get("e0") or {}).items() if v.get("passed")}
    out = build_report(collect(lane.arm_dirs()), gold, a.resamples, e0_passed=e0)
    path = CAMPAIGN_DIR / "report.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for k, v in out["baselines"].items():
        print(f"BASELINE {k}: {v['arm']} acc {v['accuracy']:.4f}")
    for e in out["checkpoints"]:
        c = e["vs_I"]
        line = (f"{e['device_class']:>13} {e['arm']:<40} {e['label']:>8} acc {e['accuracy']:.4f}"
                f" vsI {c['delta']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
        if "vs_control" in e:
            c = e["vs_control"]
            line += (f" vsCTL {c['delta']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]"
                     f" (ctl {e['control']['accuracy']:.4f} @ {e['control']['steps']} SFT steps)")
        print(line)
    for g in out["groups"]:
        c = g["best_vs_best"]
        print(f"GROUP {g['track']}/{g['group']} r{g['rung']} k={g['k']} best-of-k "
              f"{g['best_rl']['accuracy']:.4f} vs {g['best_control']['accuracy']:.4f}: "
              f"{c['delta']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
    for m in out["missing"]:
        print("MISSING", m)
    for m in out["refused"]:
        print("REFUSED", m)
    print(f"wrote {path}")
    return 1 if out["missing"] or out["refused"] or not out["checkpoints"] else 0


# ---- successive halving ---------------------------------------------------------------------
def promotion(groups: list, rung: int) -> dict:
    """Rung r -> r+1 decision (C10, C11, M5). Units are RL groups at rung r, scored by the
    best-of-k RL vs best-of-k control delta. The bar is the MAXIMUM placebo delta (SFT-from-I
    vs SFT-from-I at another seed, scored by the identical best_of_k code): with no finished
    placebo pair, promotion refuses. Kept: the top third (ceil) of units by score, and of
    those only the ones strictly above the placebo maximum. The rung schedule is the table's
    (fixed before any result); nothing here stops early on success."""
    units = sorted((g for g in groups if g["track"] == "rl" and g["rung"] == rung),
                   key=lambda g: (g["best_vs_best"]["delta"], g["best_rl"]["accuracy"]),
                   reverse=True)
    if not units:
        raise SystemExit(f"REFUSE: no scored rung-{rung} RL group")
    ks = {g["k"] for g in units}
    if len(ks) != 1:
        raise SystemExit(f"REFUSE: rung-{rung} RL groups have different k {sorted(ks)}")
    k = ks.pop()
    # same rung (same training length) and same k (best-of-k inflates with k)
    placebo = [g["best_vs_best"]["delta"] for g in groups
               if g["track"] == "placebo" and g["rung"] == rung and g["k"] == k]
    if not placebo:
        raise SystemExit(f"REFUSE: no finished rung-{rung} placebo with k = {k}; the "
                         "promotion bar is the placebo maximum (M5)")
    bar = max(placebo)
    top = units[:math.ceil(len(units) / 3)]
    return {"bar_placebo_max": bar, "placebo_deltas": sorted(placebo),
            "ranked": [[g["group"], g["best_vs_best"]["delta"]] for g in units],
            "kept": [g["group"] for g in top if g["best_vs_best"]["delta"] > bar],
            "top_third_below_bar": [g["group"] for g in top
                                    if g["best_vs_best"]["delta"] <= bar]}


def cmd_promote(a, state_path: Path) -> int:
    """Rung r -> r+1: `promotion` picks the groups; every arm of a kept group continues in
    place (gpu_lane.continue_spec, carried save_steps) on its own host and device class to
    the table's fixed rung_stops[r+1], so best-of-k stays k vs k. Controls follow through
    `controls` once each continuation is done."""
    table = load_table(a.table)
    r = a.rung
    schedule = table.get("rung_stops") or {}
    if str(r + 1) not in schedule:
        raise SystemExit(f"REFUSE: the table's fixed rung schedule has no rung {r + 1}")
    state = _read(state_path)
    rung = {q: x for q, x in state["arms"].items()
            if x.get("kind") in ("rl", "placebo", "control")
            and int(x["spec"].get("rung", 1)) == r}
    pending = [q for q, x in rung.items() if x["status"] != "DONE"]
    for q, x in rung.items():
        if x.get("kind") in ("rl", "placebo"):
            ctl = x["spec"]["id"] + "~ctl" + ("" if r == 1 else f"@r{r}")
            if (state["arms"].get(ctl) or {}).get("status") != "DONE":
                pending.append(ctl)
    if pending:
        raise SystemExit(f"REFUSE: unfinished rung-{r} arms/controls: {sorted(pending)[:5]}")
    e0 = {c for c, v in (state.get("e0") or {}).items() if v.get("passed")}
    report = build_report(collect(lane.arm_dirs()), load_gold(), a.resamples, e0_passed=e0)
    if report["refused"]:
        raise SystemExit(f"REFUSE: report refusals block promotion: {report['refused'][:3]}")
    decision = promotion(report["groups"], r)
    with runner.state_lock(state_path) as st:
        st.setdefault("promotions", {})[f"{r}->{r + 1}"] = dict(decision, at=runner.now())
    stops = [int(x) for x in schedule[str(r + 1)]]
    for q, x in sorted(rung.items()):
        if x.get("kind") != "rl" or x["spec"].get("group") not in decision["kept"]:
            continue
        first = state["arms"][x["spec"]["id"]]["spec"]  # the trajectory's first launch
        if x["device_class"] != first["device_class"]:
            raise SystemExit(f"REFUSE: {q} changed device class mid-trajectory (M6)")
        spec = lane.continue_spec({k: v for k, v in first.items() if k != "save_steps"},
                                  int(x["spec"]["stops"][-1]), stops)
        spec["save_steps"] = first["save_steps"]  # the trajectory's own value
        spec["rung"] = r + 1
        rr = valid_host(state, x["host"], x["device_class"])
        push(state_path, x["host"], spec, a.prio, rr["root"], suffix=f"@r{r + 1}", kind="rl")
    print(json.dumps(decision, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--state", type=Path, default=STATE)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--payload", required=True)
    b.add_argument("--host", action="append", required=True)
    b.add_argument("--repin", action="store_true")
    sub.add_parser("receipts")
    e = sub.add_parser("enqueue")
    e.add_argument("--table", required=True)
    e.add_argument("--ids")
    e.add_argument("--host")
    e.add_argument("--prio", default="50")
    w = sub.add_parser("worker")
    w.add_argument("--host", required=True)
    w.add_argument("--slot", type=int)
    w.add_argument("--mem", type=float)
    w.add_argument("--need-cores", type=int)
    w.add_argument("--timeout", type=int, default=259200)
    w.add_argument("--arm-timeout", type=int, default=172800)
    w.add_argument("--idle-max", type=int, default=21600)
    q = sub.add_parser("queue")
    q.add_argument("--host", required=True)
    sub.add_parser("fetch")
    c = sub.add_parser("controls")
    c.add_argument("--of")
    c.add_argument("--prio", default="05")
    r = sub.add_parser("report")
    r.add_argument("--resamples", type=int, default=20000)
    pr = sub.add_parser("promote")
    pr.add_argument("--table", required=True)
    pr.add_argument("--rung", type=int, required=True)
    pr.add_argument("--prio", default="10")
    pr.add_argument("--resamples", type=int, default=20000)
    a = p.parse_args(argv)
    configure(a.state, a.state.parent)
    fn = {"bootstrap": cmd_bootstrap, "receipts": cmd_receipts, "enqueue": cmd_enqueue,
          "worker": cmd_worker, "queue": cmd_queue, "fetch": cmd_fetch,
          "controls": cmd_controls, "report": cmd_report, "promote": cmd_promote}[a.cmd]
    return fn(a, a.state)


if __name__ == "__main__":
    sys.exit(main())
