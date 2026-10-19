"""GPU lane of exploratory sweep v2 (Mac side, light: specs, queue, fetch, report).

Exploratory only, TRAIN-derived holdouts (dev800); never reads native VAL or sealed TEST.
Study dir: runs/aircraft-exploratory-sweep-v2-gpu/ in this worktree (never the CPU sweep's
state.json). Arms run through gpu_worker.sh, one per GPU at a time, from a host-local queue.

  gpu_lane.py specs                         # write docs/.../gpu-arms.json (the arm table)
  gpu_lane.py enqueue --host H --ids A,B [--prio NN]  # push specs into H's queue
  gpu_lane.py worker --host H               # start H's queue worker through uwrun
  gpu_lane.py queue --host H                # list H's queue
  gpu_lane.py fetch                         # copy every worker's results into the study dir
  gpu_lane.py control --of ARM --host H     # matched SFT control (E + T) from ARM's receipt
  gpu_lane.py report                        # per-arm dev800 curves vs I on the same device

Every arm spec keeps the CPU sweep's keys (arm_exec.py reads them) plus device/dtype.
Refusals exit non-zero: an unknown id, a spec already queued or finished on any host, a
control whose RL arm has no fetched receipt, a report with no I baseline on an arm's device.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
STUDY = "aircraft-exploratory-sweep-v2-gpu"
STUDY_DIR = WT / "runs" / STUDY
DOCS = WT / "docs" / STUDY
TABLE = DOCS / "gpu-arms.json"
STATE = STUDY_DIR / "state.json"
UWRUN = Path.home() / ".claude" / "bin" / "uwrun"
ROOT = "/var/tmp/r226lin-rlesweep-gpu"
GOLD = WT / "docs" / "aircraft-exploratory-sweep-v2" / "manifests" / "dev800.jsonl"
# device classes: the GPU kind and precision an arm ran under. Comparisons stay inside one.
DEVICE_CLASS = {"ecetesla2": "rtx3070-bfloat16", "ecetesla4": "rtx3070-bfloat16",
                "eceubuntu2": "rtx2080ti-float32"}
DTYPE = {"rtx3070-bfloat16": "bfloat16", "rtx2080ti-float32": "float32"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runner():
    return _load("sweep_runner", HERE.parent / "runner.py")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def uw(*args: str, timeout: int = 300) -> str:
    p = subprocess.run([str(UWRUN), *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise SystemExit(f"REFUSE: uwrun {' '.join(args[:2])} rc={p.returncode}: "
                         f"{(p.stderr or p.stdout).strip()[-400:]}")
    return p.stdout


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"study_id": STUDY, "workers": {}, "queued": {}}


def save_state(state: dict) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(STATE)


# ---- arm table ------------------------------------------------------------------------
BASE = {"adapter_profile": "legacy-qv-v1", "max_completion_length": 256, "max_pixels": 200704,
        "min_pixels": 50176, "response_format": "answer_only", "seed": 42, "threads": 4,
        "device": "cuda", "features": ["cuda"], "wave": 1, "init": "W",
        "train_manifest": "s_train_scan_seed42.jsonl"}


def grpo(cls: str, lr: float, beta: float, screened: bool, g: int = 8, t: float = 1.0,
         init: str = "W") -> dict:
    tag = "s" if screened else "n"
    arm_id = f"gpu-{cls.split('-')[0]}-grpo-{init}-lr{lr:g}-g{g}-b{beta:g}-t{t:g}-{tag}"
    arm = dict(BASE, id=arm_id, family=f"grpo-{init}", init=init, track="rl", mode="grpo",
               lr=lr, beta=beta, temperature=t,
               num_generations=g, dtype=DTYPE[cls], device_class=cls,
               screen_threshold=0.5 if screened else None,
               # screened: s_train has 1,424 unique rows, so 600 accepted is the ceiling
               # (CPU sweep amendment 2); unscreened declares 1,200 so a promising arm can
               # continue in place past 600 without changing its settings identity.
               max_steps_total=600 if screened else 1200, stops=[150, 300, 600])
    return arm


def evals(host: str) -> list:
    cls = DEVICE_CLASS[host]
    return [dict(BASE, id=f"gpu-{host}-eval-{w}", init=w, family="baseline",
                 track="baseline", mode="eval", lr=0.0, beta=0.04, temperature=1.0,
                 num_generations=4, dtype=DTYPE[cls], device_class=cls, rung=0,
                 screen_threshold=None) for w in ("W", "I")]


def cmd_specs(a) -> int:
    arms = evals("ecetesla2") + evals("ecetesla4") + evals("eceubuntu2")
    for lr in (5e-6, 1e-5, 2e-5, 5e-5):
        for beta in (0.0, 0.04):
            for screened in (False, True):
                arms.append(grpo("rtx3070-bfloat16", lr, beta, screened))
    for lr, beta, screened in ((1e-5, 0.0, False), (5e-6, 0.04, False), (2e-5, 0.0, False),
                               (1e-5, 0.04, True)):
        arms.append(grpo("rtx2080ti-float32", lr, beta, screened))
    # Amendment 1 (coordinator, after the first W endpoints; before any GRPO-I result):
    # SFT -> RL. GRPO from I (the strongest SFT), extending the CPU sweep's grpo-I family:
    # LR {1e-6, 3e-6, 1e-5} x beta {0, 0.04} x screening {off, on}, G8, T 1.0, on both
    # device classes; each gets a time-matched SFT-continuation-from-I control (LR 5e-6).
    for cls in ("rtx3070-bfloat16", "rtx2080ti-float32"):
        for lr in (1e-6, 3e-6, 1e-5):
            for beta in (0.0, 0.04):
                for screened in (False, True):
                    arms.append(grpo(cls, lr, beta, screened, init="I"))
    ids = [x["id"] for x in arms]
    if len(set(ids)) != len(ids):
        raise SystemExit("REFUSE: duplicate arm ids")
    DOCS.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps({"study_id": STUDY, "parent_study": "aircraft-exploratory-sweep-v2",
                                 "holdout": "dev800", "arms": arms}, indent=1, sort_keys=True)
                     + "\n")
    print(f"wrote {TABLE}: {len(arms)} arms")
    return 0


def table() -> dict:
    return {x["id"]: x for x in json.loads(TABLE.read_text())["arms"]}


def push(host: str, spec: dict, prio: str, state: dict, suffix: str = "") -> None:
    if DEVICE_CLASS[host] != spec["device_class"]:
        raise SystemExit(f"REFUSE: {spec['id']} is {spec['device_class']}, {host} is "
                         f"{DEVICE_CLASS[host]}")
    qid = spec["id"] + suffix
    if qid in state["queued"]:
        raise SystemExit(f"REFUSE: {qid} already queued on {state['queued'][qid]['host']}")
    key = f"{prio}-{qid}"
    raw = base64.b64encode((json.dumps(spec, indent=1, sort_keys=True) + "\n").encode()).decode()
    uw("ssh", host, f"mkdir -p {ROOT}/gpuq/pending && echo {raw} | base64 -d > "
       f"{ROOT}/gpuq/pending/.{key}.tmp && mv {ROOT}/gpuq/pending/.{key}.tmp "
       f"{ROOT}/gpuq/pending/{key}.json && echo queued {key}", timeout=180)
    state["queued"][qid] = {"host": host, "key": key, "at": now()}
    save_state(state)
    print(f"queued {key} on {host}")


def cmd_enqueue(a) -> int:
    t, state = table(), load_state()
    for i, arm_id in enumerate(a.ids.split(",")):
        if arm_id not in t:
            raise SystemExit(f"REFUSE: unknown arm {arm_id}")
        push(a.host, t[arm_id], f"{int(a.prio) + i:02d}", state)
    return 0


def cmd_worker(a) -> int:
    state = load_state()
    name = f"rlegpu-worker-{a.host}-{int(time.time())}"
    need = {"eceubuntu2": 10500}.get(a.host, 6500)  # MiB free before an arm starts
    inner = (f"R={ROOT}; NEED_MIB={need} exec bash "
             "\"$R/payload/code/scripts/exploratory_sweep/gpu_worker.sh\" "
             f"\"$R\" \"$PWD/results\"")
    # --mem: host RSS cap for the uwrun memory guard (a GPU arm's trainer + assay child hold
    # the model in host RAM while loading; 24 GB is headroom over that, not a measurement).
    uw("run", "--host", a.host, "--threads", "4", "--timeout", str(a.timeout), "--mem", "24",
       "--name", name, "--", "bash", "-c", inner, timeout=600)
    state["workers"].setdefault(a.host, []).append({"name": name, "at": now()})
    save_state(state)
    print(f"worker {name}")
    return 0


def cmd_queue(a) -> int:
    print(uw("ssh", a.host, f"cd {ROOT}/gpuq && for d in pending running done failed; do "
             f"echo \"$d: $(ls $d | tr '\\n' ' ')\"; done; nvidia-smi --query-gpu="
             f"memory.used,utilization.gpu --format=csv,noheader", timeout=180))
    return 0


def cmd_fetch(a) -> int:
    state = load_state()
    for host, workers in sorted(state["workers"].items()):
        for w in workers:
            dest = STUDY_DIR / "results" / host / w["name"]
            subprocess.run([str(UWRUN), "fetch", w["name"], "work/results"],
                           capture_output=True, text=True, timeout=1800)
            src = Path.home() / ".claude" / "uw" / "results" / w["name"]
            if src.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", f"{src}/", f"{dest}/"], check=True, timeout=600)
            st = subprocess.run([str(UWRUN), "status", w["name"]], capture_output=True,
                                text=True, timeout=180).stdout.strip()
            print(f"{host} {w['name']}: {st.splitlines()[-1] if st else '?'}")
    for summary in sorted((STUDY_DIR / "results").rglob("summary.json")):
        s = json.loads(summary.read_text())
        eps = [f"{e['label']}={e['metrics']['accuracy']:.4f}" for e in s.get("endpoints", [])]
        print(f"  {s['arm']}: {' '.join(eps)}")
    return 0


def arm_dirs() -> list:
    """[(arm id, results dir)] over every fetched worker, ordered by the worker's start time
    (a continuation's dir comes after the run it continues)."""
    out = []
    for arm_json in (STUDY_DIR / "results").rglob("arm.json"):
        d = arm_json.parent
        w = d / "worker.json"
        start = json.loads(w.read_text())["start"] if w.is_file() else 10**12
        out.append((start, d.name, json.loads(arm_json.read_text())["id"], d))
    return [(i, d) for _, _, i, d in sorted(out)]


def arm_results() -> dict:
    """{arm id: latest results dir}."""
    return dict(arm_dirs())


RAW_FILES = ("arm.json", "summary.json", "worker.json", "stages.json", "endpoint_receipt.json",
             "optimizer_audit.jsonl", "reward_groups.jsonl", "screen_journal.jsonl",
             "gpu-trace.csv", "nvsmi-before.txt", "local_training.json")


def cmd_raw(a) -> int:
    """Copy each fetched arm's small raw files to docs/<study>/raw/<arm>/ (to be committed
    before any interpretation). Copies only the allowlisted names and assays/*.json."""
    import shutil

    n = 0
    for arm_id, d in arm_dirs():
        dest = DOCS / "raw" / d.parent.parent.name / d.name.replace("~", "_")
        for name in RAW_FILES:
            if (d / name).is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(d / name, dest / name)
                n += 1
        for f in sorted((d / "assays").glob("*.json")):
            (dest / "assays").mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / "assays" / f.name)
            n += 1
    if n == 0:
        raise SystemExit("REFUSE: no fetched arm results to copy")
    print(f"copied {n} files under {DOCS / 'raw'}")
    return 0


def cmd_continue(a) -> int:
    """Continue an arm in place on its host from checkpoint-<resume_stop> (same work dir,
    same trajectory); stops after resume_stop only."""
    t, state = table(), load_state()
    extra = [int(x) for x in a.extra_stops.split(",")] if a.extra_stops else []
    spec = continue_spec(t[a.of], a.resume_stop, extra)
    push(a.host, spec, a.prio, state, suffix=f"@c{a.resume_stop}{a.retry}")
    return 0


def continue_spec(original: dict, resume_stop: int, extra_stops: list) -> dict:
    """The arm-table spec continued in place from checkpoint-<resume_stop>. save_steps is
    carried from the trajectory's FIRST launch (the table spec), never recomputed from the
    continuation's own stops: it is part of the trainer's resume identity."""
    arm_exec = _load("arm_exec", HERE.parent / "arm_exec.py")
    if original.get("resume_stop") or original.get("work_id"):
        raise SystemExit("REFUSE: pass the arm-table spec (the trajectory's first launch)")
    stops = sorted({x for x in original["stops"] if x > resume_stop} |
                   {int(x) for x in extra_stops})
    if not stops or min(stops) <= resume_stop or max(stops) > original["max_steps_total"]:
        raise SystemExit("REFUSE: no stop past resume_stop, or a stop past max_steps_total")
    save = arm_exec.save_steps_for(original, original["max_steps_total"])
    if resume_stop % save or any(x % save for x in stops):
        raise SystemExit(f"REFUSE: stops must be multiples of the carried save_steps {save}")
    spec = dict(original)
    spec.update(work_id=original["id"], resume_stop=resume_stop, stops=stops, save_steps=save)
    return spec


def cmd_control(a) -> int:
    """SFT from W on the RL arm's committed prompt sequence: E = same examples, T = same
    active seconds (CPU-sweep control rule, same host/device as the RL arm)."""
    res, t, state = arm_results(), table(), load_state()
    rl = t.get(a.of) or json.loads((res[a.of] / "arm.json").read_text())
    d = res.get(a.of)
    if (d is None or not (d / "summary.json").is_file()
            or not (d / "endpoint_receipt.json").is_file()):
        raise SystemExit(f"REFUSE: {a.of} has no fetched summary/receipt")
    summary = json.loads((d / "summary.json").read_text())
    receipt = json.loads((d / "endpoint_receipt.json").read_text())
    last = [e for e in summary["endpoints"] if e["label"].startswith("step")][-1]
    ids = receipt["committed_prompt_ids"]
    if not ids or receipt["global_step"] != last["steps"]:
        raise SystemExit(f"REFUSE: {a.of} receipt does not match its last step")
    keep = ("adapter_profile", "max_completion_length", "max_pixels", "min_pixels", "seed",
            "threads", "device", "features", "wave", "init", "dtype", "device_class")
    ctl = {k: rl[k] for k in keep}
    # CPU sweep control rule: SFT LR 1e-5 from W, 5e-6 from I
    ctl.update(id=f"{a.of}~ctl", family="control", track="control", mode="sft",
               lr={"W": 1e-5, "I": 5e-6}[rl["init"]],
               control_of=a.of, order="sequential", cycle=True, num_generations=4,
               response_format="answer_only", beta=None, temperature=None,
               screen_threshold=None, max_steps_total=100000, max_active_seconds=259200.0,
               train_manifest=f"@consumed:{a.of}", train_ids=ids, stops=[len(ids)],
               time_budget_seconds=float(last["active_seconds"]))
    push(a.host, ctl, a.prio, state)
    return 0


def cmd_report(a) -> int:
    run = _runner()
    gold = {}
    for line in GOLD.read_text().splitlines():
        r = json.loads(line)
        gold[r["id"]] = r["label"]
    rows, base, host_of = {}, {}, {}
    for arm_id, d in arm_dirs():
        spec = json.loads((d / "arm.json").read_text())
        if (d / "summary.json").is_file():
            host_of[arm_id] = json.loads((d / "summary.json").read_text())["host"].split(".")[0]
        for f in sorted((d / "assays").glob("*.json")):
            doc = json.loads(f.read_text())
            if set(doc["predictions"]) != set(gold):
                raise SystemExit(f"REFUSE: {arm_id}/{f.name} is not over dev800")
            rows[(arm_id, f.stem)] = (spec, f.stem, doc)  # a later dir (re-assay) wins
            if spec["mode"] == "eval":
                # keyed by host (same physical card) and by device class (first host wins)
                base[(host_of.get(arm_id), spec["init"])] = (arm_id, doc)
                base.setdefault((spec["device_class"], spec["init"]), (arm_id, doc))
    out = {"study_id": STUDY, "exploratory": True, "holdout": "dev800", "created_at": now(),
           "resamples": a.resamples, "seed": 20261018, "baselines": {}, "arms": []}
    for (cls, init), (arm_id, doc) in sorted(base.items(), key=str):
        out["baselines"][f"{cls}/{init}"] = {"arm": arm_id, **{k: doc["metrics"][k] for k in (
            "accuracy", "valid_fraction", "equal_class_accuracy")}}
    # Device control: the same W/I adapters assayed on CPU fp32 by the CPU sweep (read only).
    cpu = {}
    cpu_dir = Path("/Users/idide/RyanLinEnv/runs/aircraft-exploratory-sweep-v2/results")
    for init in ("W", "I"):
        found = sorted(cpu_dir.glob(f"eval-{init}-*/assays/init.json"))
        if found:
            cpu[init] = (str(found[-1]), json.loads(found[-1].read_text()))
    out["device_control"] = []
    for (key, init), (arm_id, doc) in sorted(base.items(), key=str):
        if key not in DEVICE_CLASS or init not in cpu:
            continue  # host-keyed rows only, each against the CPU fp32 assay
        cp = cpu[init][1]["predictions"]
        gp = doc["predictions"]
        out["device_control"].append({
            "init": init, "host": key, "gpu_arm": arm_id, "cpu_source": cpu[init][0],
            "gpu_accuracy": doc["metrics"]["accuracy"],
            "cpu_accuracy": cpu[init][1]["metrics"]["accuracy"],
            "same_prediction_fraction": sum(cp[i] == gp[i] for i in gold) / len(gold),
            "gpu_minus_cpu": run.paired_bootstrap(gp, cp, gold, a.resamples, seed=20261018)})
    for spec, label, doc in rows.values():
        if spec["mode"] == "eval":
            continue
        same = host_of.get(spec["id"])
        i = base.get((same, "I")) or base.get((spec["device_class"], "I"))
        if i is None:
            raise SystemExit(f"REFUSE: no I baseline on {spec['device_class']}")
        steps = int(label[4:]) if label.startswith("step") else None
        entry = {"arm": spec["id"], "device_class": spec["device_class"], "label": label,
                 "host": same, "I_baseline": i[0],
                 "steps": steps, **{k: doc["metrics"][k] for k in (
                     "accuracy", "valid_fraction", "equal_class_accuracy")},
                 "vs_I": run.paired_bootstrap(doc["predictions"], i[1]["predictions"], gold,
                                              a.resamples, seed=20261018)}
        # matched SFT control of this RL arm (same device): E = examples-matched endpoint,
        # T = time-matched endpoint; every RL checkpoint is compared with both.
        if not spec["id"].endswith("~ctl"):
            for (cid, clabel), (_, _, cdoc) in rows.items():
                if cid == spec["id"] + "~ctl":
                    which = "T" if clabel == "time" else "E"
                    entry[f"vs_ctl_{which}"] = dict(
                        label=clabel, control_accuracy=cdoc["metrics"]["accuracy"],
                        **run.paired_bootstrap(doc["predictions"], cdoc["predictions"], gold,
                                               a.resamples, seed=20261018))
        w = base.get((same, "W")) or base.get((spec["device_class"], "W"))
        if w is not None:
            entry["vs_W_delta"] = run.paired_bootstrap(doc["predictions"], w[1]["predictions"],
                                                       gold, 2000, seed=20261018)["delta"]
        out["arms"].append(entry)
    out["arms"].sort(key=lambda e: (e["device_class"], e["arm"], e["steps"] or 10**9))
    path = STUDY_DIR / "report.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for c in out["device_control"]:
        d = c["gpu_minus_cpu"]
        print(f"DEVICE {c['init']} {c['host']}: gpu {c['gpu_accuracy']:.4f} cpu "
              f"{c['cpu_accuracy']:.4f} same-pred {c['same_prediction_fraction']:.4f} "
              f"delta {d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]")
    for k, v in out["baselines"].items():
        print(f"BASELINE {k}: acc {v['accuracy']:.4f} valid {v['valid_fraction']:.4f}")
    for e in out["arms"]:
        c = e["vs_I"]
        print(f"{e['device_class']:>18} {e['arm']:<48} {e['label']:>8} acc {e['accuracy']:.4f} "
              f"valid {e['valid_fraction']:.3f} vsI {c['delta']:+.4f} "
              f"[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] P(<=0) {c['p_le_0']:.3f}")
        for k in ("vs_ctl_E", "vs_ctl_T"):
            if k in e:
                c = e[k]
                print(f"{'':>18} {'':<48} {k:>8} ctl {c['control_accuracy']:.4f} delta "
                      f"{c['delta']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("specs")
    e = sub.add_parser("enqueue")
    e.add_argument("--host", required=True, choices=sorted(DEVICE_CLASS))
    e.add_argument("--ids", required=True)
    e.add_argument("--prio", default="50")
    w = sub.add_parser("worker")
    w.add_argument("--host", required=True, choices=sorted(DEVICE_CLASS))
    w.add_argument("--timeout", type=int, default=172800)
    q = sub.add_parser("queue")
    q.add_argument("--host", required=True, choices=sorted(DEVICE_CLASS))
    sub.add_parser("fetch")
    sub.add_parser("raw")
    c = sub.add_parser("control")
    c.add_argument("--of", required=True)
    c.add_argument("--host", required=True, choices=sorted(DEVICE_CLASS))
    c.add_argument("--prio", default="05")
    k = sub.add_parser("continue")
    k.add_argument("--of", required=True)
    k.add_argument("--host", required=True, choices=sorted(DEVICE_CLASS))
    k.add_argument("--resume-stop", type=int, required=True)
    k.add_argument("--extra-stops", default="")
    k.add_argument("--prio", default="04")
    k.add_argument("--retry", default="", help="suffix for a re-queue of the same continuation")
    r = sub.add_parser("report")
    r.add_argument("--resamples", type=int, default=20000)
    a = p.parse_args(argv)
    return {"specs": cmd_specs, "enqueue": cmd_enqueue, "worker": cmd_worker,
            "queue": cmd_queue, "fetch": cmd_fetch, "control": cmd_control,
            "report": cmd_report, "raw": cmd_raw, "continue": cmd_continue}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
