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
import os
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
# uwrun v5.11 enforces owner maps by caller identity; unset counts as another session.
os.environ["UWRUN_OWNER"] = "ryanlinenv"
ROOT = "/var/tmp/r226lin-rlesweep-gpu"
# Lane roots per host. CPU hosts (root decision 2026-10-19: the T1.0 corrected GRPO-I arms and
# their controls on iccad0/iccad13) keep their own root on /dev/shm (iccad0's root fs is full).
ROOTS = {h: "/dev/shm/r226lin-rlesweep-gpulane"
         for h in ("iccad0", "iccad13", "iccad8", "iccad10", "iccad11", "iccad12", "iccad7")}


def root_of(host: str) -> str:
    return ROOTS.get(host, ROOT)
GOLD = WT / "docs" / "aircraft-exploratory-sweep-v2" / "manifests" / "dev800.jsonl"
# device classes: the GPU kind and precision an arm ran under. Comparisons stay inside one.
DEVICE_CLASS = {"ecetesla2": "rtx3070-bfloat16", "ecetesla4": "rtx3070-bfloat16",
                "eceubuntu2": "rtx2080ti-float32", "iccad0": "cpu-float32",
                "iccad13": "cpu-float32",
                "iccad8": "cpu-float32", "iccad10": "cpu-float32", "iccad11": "cpu-float32",
                "iccad12": "cpu-float32", "iccad7": "cpu-float32"}
DTYPE = {"rtx3070-bfloat16": "bfloat16", "rtx2080ti-float32": "float32", "cpu-float32": "float32"}


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


CALIBRATION = ("RL400", "SFT1e5E", "SFT5e5E", "SFT1e4E", "SFT5e5T", "SFT1e4T")
SFT_I_STOPS = [150, 300, 600, 1200, 2400, 4800]


def sft_i(lr: float, seed: int = 42) -> dict:
    """SFT continuation from I on the clean pool (s_train) in its seed's scan order, cycling;
    bf16 on the 3070s (allowed for SFT by the root corrections)."""
    cls = "rtx3070-bfloat16"
    return dict(BASE, id=f"gpu-rtx3070-sft-I-lr{lr:g}-s{seed}", init="I", family="sft-I",
                track="sft", mode="sft", lr=lr, seed=seed, order="sequential", cycle=True,
                train_manifest=f"s_train_scan_seed{seed}.jsonl", num_generations=4,
                beta=None, temperature=1.0, screen_threshold=None, dtype=DTYPE[cls],
                device_class=cls, stops=list(SFT_I_STOPS), max_steps_total=SFT_I_STOPS[-1],
                # extended TRAIN-only SFT needs a finite active-time cap (trainer rule)
                max_active_seconds=259200.0)


def gate0() -> list:
    """Amendment 2 (root RL corrections 2026-10-18, GATE 0; before any of these ran):
    E0 and P0-I in fp32 on the 2080 Ti; dev800 calibration of the native-VAL-scored
    controlled400 endpoints, the SFT-from-I curve and its seed placebo in bf16 on the 3070s."""
    ti = "rtx2080ti-float32"
    e0 = dict(BASE, id="gpu-rtx2080ti-e0-I-grpo-lr1e-05-g8-b0-ds8", init="I", family="gate0",
              track="gate0", mode="e0", lr=1e-5, beta=0.0, temperature=1.0, num_generations=8,
              screen_threshold=None, dtype=DTYPE[ti], device_class=ti, e0_steps=4,
              # beta 0 needs an explicit loss type in the merged trainer
              extra_trainer_args=["--dynamic-sampling-max-draws", "8", "--loss-type", "grpo"])
    p0 = dict(BASE, id="gpu-rtx2080ti-p0-I", init="I", family="gate0", track="gate0",
              mode="p0", lr=0.0, beta=None, temperature=1.0, num_generations=8,
              screen_threshold=None, dtype=DTYPE[ti], device_class=ti,
              probe_manifests=["p0_poolA200.jsonl", "p0_poolB200.jsonl"],
              temperatures=[1.0, 1.3])
    cal = [dict(BASE, id=f"gpu-rtx3070-calib-{n}", init=n, family="calibration",
                track="baseline", mode="eval", lr=0.0, beta=0.04, temperature=1.0,
                num_generations=4, dtype="bfloat16", device_class="rtx3070-bfloat16", rung=0,
                screen_threshold=None) for n in CALIBRATION]
    sft = [sft_i(lr) for lr in (5e-6, 1e-5, 2e-5)]
    placebo = [sft_i(1e-5, seed) for seed in (314, 2718)]
    smoke = dict(p0, id="gpu-rtx2080ti-p0-I-smoke", probe_limit=3, temperatures=[1.0, 1.3])
    # Amendment 3 (backlog, before these ran): placebo seeds 7 and 11; and M9, a 2-seed
    # bf16-vs-fp32 replicate of ONE GRPO-I config (LR 1e-5, beta 0, G8, T 1, dynamic sampling
    # 8, unscreened), each device class its own trajectory, compared with the placebo spread.
    placebo += [sft_i(1e-5, seed) for seed in (7, 11)]
    m9 = []
    for cls in ("rtx3070-bfloat16", "rtx2080ti-float32"):
        for seed in (42, 314):
            arm = grpo(cls, 1e-5, 0.0, False, init="I")
            arm.update(id=f"gpu-{cls.split('-')[0]}-m9-grpo-I-lr1e-05-g8-b0-ds8-s{seed}",
                       family="m9", seed=seed, train_manifest=f"s_train_scan_seed{seed}.jsonl",
                       extra_trainer_args=["--dynamic-sampling-max-draws", "8",
                                           "--loss-type", "grpo"])
            m9.append(arm)
    # Amendment 4 (coordinator; before it ran): P0-I also on the campaign's corrected `mixed`
    # pool (rl-campaign-pools.noindex 4ea173d, docs/aircraft-rl-campaign-I/pools/v2/pools/
    # manifests/mixed.jsonl, sha256 9406af37...; hard = greedy-wrong & p >= 0.02 is a subset).
    # SUPERSEDED before it ran: pools v2 held 0447760, a VAL/TEST near-duplicate. Amendment 12
    # (root, 2026-10-19): P0-I on pools v3 mixed (rl-campaign-pools ad54cfb, sha 052a2953..,
    # 454 rows; hard v3 sha 9707f347.., 207). The v2 spec (gpu-rtx2080ti-p0-I-mixed) was
    # removed from this table and from its queue; it never ran.
    p0_mixed = dict(p0, id="gpu-rtx2080ti-p0-I-mixedv3", probe_manifests=["p0_mixedv3_454.jsonl"])
    return ([e0, smoke, p0] + cal + sft + placebo + m9 + [p0_mixed] + corrected_grpo_i()
            + corrected_grpo_i_cpu() + amendment7())


def cpu_grpo_i(lr: float, t: float, seed: int, arm_id: str, family: str) -> dict:
    arm = grpo("cpu-float32", lr, 0.0, False, t=t, init="I")
    arm.update(id=arm_id, family=family, device="cpu", features=[], threads=16, seed=seed,
               train_manifest=f"s_train_scan_seed{seed}.jsonl",
               extra_trainer_args=["--dynamic-sampling-max-draws", "8", "--loss-type", "grpo"])
    return arm


def amendment7() -> list:
    """Amendment 7 (root D38, 2026-10-19; before these ran): the 2080 Ti is held by another
    user, so the fp32 half of M9 runs on CPU fp32 (CPU fp32 == 2080 Ti fp32 on the device
    control, identical predictions on all 800 items); and CPU copies of the corrected T1.3
    arms lr1e-5 and lr5e-5. Same spec as their GPU twins except device."""
    m9 = [cpu_grpo_i(1e-5, 1.0, seed, f"cpu-m9-grpo-I-lr1e-05-g8-b0-ds8-s{seed}", "m9")
          for seed in (42, 314)]
    t13 = [cpu_grpo_i(lr, 1.3, 42, f"cpu-cgrpo-I-lr{lr:g}-g8-b0-t1.3-ds8", "cgrpo-I")
           for lr in (1e-5, 5e-5)]
    return m9 + t13 + amendment8()


def amendment8() -> list:
    """Amendment 8 (root, 2026-10-19; before these ran): seed replicates on CPU fp32 of the
    2080 Ti's lead config lr2e-5 T1.3 (seeds 314, 2718; the confirmatory plan needs them) and
    of lr5e-5 T1.3 (seed 314)."""
    return [cpu_grpo_i(lr, 1.3, seed, f"cpu-cgrpo-I-lr{lr:g}-g8-b0-t1.3-ds8-s{seed}", "cgrpo-I")
            for lr, seed in ((2e-5, 314), (2e-5, 2718), (5e-5, 314))] + [
        # Amendment 9 (root, before it ran): the lr3e-6 T1.3 forgetting guard on CPU too,
        # since the 2080 Ti is held by another user's job.
        cpu_grpo_i(3e-6, 1.3, 42, "cpu-cgrpo-I-lr3e-06-g8-b0-t1.3-ds8", "cgrpo-I"),
        # Amendment 10 (root, before it ran): seed 314 of lr2e-5 T1.0.
        cpu_grpo_i(2e-5, 1.0, 314, "cpu-cgrpo-I-lr2e-05-g8-b0-t1-ds8-s314", "cgrpo-I")] + [
        # Amendment 11 (root, before they ran): lead-config seed replicates 7 and 11.
        cpu_grpo_i(2e-5, 1.3, seed, f"cpu-cgrpo-I-lr2e-05-g8-b0-t1.3-ds8-s{seed}", "cgrpo-I")
        for seed in (7, 11)] + exploratory_bf16() + pool_arms() + replan_arms() + early_seeds() \
        + t10_replan() + multi_image_arms() + r2_neighbourhood() + sftall_arms()


def exploratory_bf16() -> list:
    """Amendment 13 (root, under CORRECTIONS C2; before these ran): EXPLORATORY-bf16 GRPO-I on
    the 3070s, the fastest search (~7 s/update). Labelled exploratory: any arm >= 0.795 on
    dev800 must replicate in fp32 before it counts."""
    arms = []
    for lr, t in ((2e-5, 1.3), (5e-5, 1.3), (2e-5, 1.0), (5e-5, 1.0)):
        arm = grpo("rtx3070-bfloat16", lr, 0.0, False, t=t, init="I")
        arm.update(id=f"gpu-rtx3070-xbf16-cgrpo-I-lr{lr:g}-g8-b0-t{t:g}-ds8",
                   family="xbf16", label="EXPLORATORY-bf16",
                   extra_trainer_args=["--dynamic-sampling-max-draws", "8",
                                       "--loss-type", "grpo"])
        arms.append(arm)
    return arms


def corrected_grpo_i_cpu() -> list:
    """Amendment 6 (root, 2026-10-19; before these ran): the T1.0 variants of the corrected
    GRPO-I arms on CPU fp32 (iccad0/iccad13, 16 threads), lr2e-5 first. The 2080 Ti keeps
    the T1.3 arms. Their controls also run on CPU fp32 (one device class per trajectory)."""
    arms = []
    for lr in (2e-5, 5e-5, 1e-5, 3e-6):
        arm = grpo("cpu-float32", lr, 0.0, False, t=1.0, init="I")
        arm.update(id=f"cpu-cgrpo-I-lr{lr:g}-g8-b0-t1-ds8", family="cgrpo-I", device="cpu",
                   features=[], threads=16,
                   extra_trainer_args=["--dynamic-sampling-max-draws", "8",
                                       "--loss-type", "grpo"])
        arms.append(arm)
    return arms


def corrected_grpo_i() -> list:
    """Amendment 5 (GATE 0 passed, root decision D37; before any of these ran): GRPO-I per
    the root corrections. fp32 on the 2080 Ti only, beta 0 with an explicit loss type,
    dynamic sampling (8 draws), unscreened, G8, dev800 at 150/300/600. Launch order:
    lr2e-5 T1.3, lr2e-5 T1.0, lr5e-5 T1.3, lr1e-5 T1.3, then the lr3e-6 forgetting guard."""
    arms = []
    for lr, t in ((2e-5, 1.3), (2e-5, 1.0), (5e-5, 1.3), (1e-5, 1.3), (3e-6, 1.3)):
        arm = grpo("rtx2080ti-float32", lr, 0.0, False, t=t, init="I")
        arm.update(id=f"gpu-rtx2080ti-cgrpo-I-lr{lr:g}-g8-b0-t{t:g}-ds8", family="cgrpo-I",
                   extra_trainer_args=["--dynamic-sampling-max-draws", "8",
                                       "--loss-type", "grpo"])
        arms.append(arm)
    return arms


# main's arm_exec.check_feature_flags: a feature and its extra flag appear together, or the
# arm refuses on the host. Mirrors arm_exec.FEATURE_EXTRA_FLAGS.
FEATURE_EXTRA_FLAGS = {"dynamic_sampling": "--dynamic-sampling-max-draws"}


def declare_features(arm: dict) -> dict:
    flags = {str(x) for x in arm.get("extra_trainer_args") or []}
    features = list(arm.get("features") or [])
    for feature, flag in FEATURE_EXTRA_FLAGS.items():
        if flag in flags and feature not in features:
            features.append(feature)
    return dict(arm, features=features)


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
    arms += gate0()
    arms = [declare_features(x) for x in arms]
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
    """The committed arm table plus promoted arms (docs/<study>/promoted-arms.json)."""
    t = {x["id"]: x for x in json.loads(TABLE.read_text())["arms"]}
    if PROMOTED.is_file():
        for x in json.loads(PROMOTED.read_text())["arms"]:
            if x["id"] in t and t[x["id"]] != x:
                raise SystemExit(f"REFUSE: promoted {x['id']} differs from the table")
            t[x["id"]] = x
    return t


PROMOTED = DOCS / "promoted-arms.json"
PROMOTION_BAR = 0.795  # OVERNIGHT plan: +1 pp over the best SFT-from-I (0.785)
FLAG_BAR = 0.790  # root: flag for an early seed and fp32 twin


def promotion_specs(arm: dict) -> list:
    """For an EXPLORATORY-bf16 arm at >= PROMOTION_BAR: its fp32 replicate on the 2080 Ti
    (same recipe, seed 42) and bf16 seeds 314 and 2718 of the same recipe on the 3070s."""
    if arm.get("family") not in ("xbf16", "xbf16-pool"):
        raise SystemExit(f"REFUSE: {arm['id']} is not an EXPLORATORY-bf16 arm")
    base = {k: v for k, v in arm.items() if k != "label"}
    fp32 = dict(base, id=arm["id"].replace("gpu-rtx3070-", "gpu-rtx2080ti-promo-"),
                device_class="rtx2080ti-float32", dtype="float32",
                family="promo-fp32", promoted_from=arm["id"])
    out = [fp32]
    for seed in (314, 2718):
        s = dict(arm, id=f"{arm['id']}-s{seed}", seed=seed, promoted_from=arm["id"])
        if s["train_manifest"].startswith("s_train_scan_seed"):
            s["train_manifest"] = f"s_train_scan_seed{seed}.jsonl"
        out.append(s)
    return out


def cmd_promote(a) -> int:
    """Scan fetched endpoints; for each EXPLORATORY-bf16 arm at >= the bar not yet promoted,
    write its promotion specs to promoted-arms.json and queue: the fp32 replicate at the FRONT
    of the 2080 Ti queue, seeds 314/2718 on the 3070s. (Its bf16 control2 is queued when the
    arm finishes, since time stops need every checkpoint.) Prints nothing when none qualify."""
    t, state = table(), load_state()
    doc = json.loads(PROMOTED.read_text()) if PROMOTED.is_file() else {"arms": [], "log": []}
    done = {x["promoted_from"] for x in doc["arms"]}
    for arm_id, d in arm_dirs():
        spec = t.get(arm_id)
        if not spec or spec.get("family") not in ("xbf16", "xbf16-pool") or arm_id in done:
            continue
        hits = [(f.stem, json.loads(f.read_text())["metrics"]["accuracy"])
                for f in sorted((d / "assays").glob("step*.json"))]
        best = max(hits, key=lambda x: x[1]) if hits else None
        if best and FLAG_BAR <= best[1] < PROMOTION_BAR:
            print(f"FLAG {arm_id} best {best[0]} {best[1]:.4f} "
                  f"(>= {FLAG_BAR}: seed + fp32 twin early)")
        if not best or best[1] < PROMOTION_BAR:
            continue
        specs = promotion_specs(spec)
        doc["arms"] += specs
        doc["log"].append({"arm": arm_id, "endpoint": best[0], "accuracy": best[1], "at": now()})
        PROMOTED.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        done.add(arm_id)
        push("eceubuntu2", specs[0], "000", state)
        push("ecetesla2", specs[1], "05", state)
        push("ecetesla4", specs[2], "05", state)
        print(f"PROMOTED {arm_id} ({best[0]} {best[1]:.4f})")
    return 0


def replan_arms() -> list:
    """Amendment 15 (root re-plan 2026-10-19 ~06:45Z, after P1 @150 = 0.7462; before these
    ran): pool-only GRPO forgets the easy images, so anchor it. EXPLORATORY-bf16, 600 updates.
      R1 RFT / expert iteration from I on mixed-v3 + an equal number of s_train rows I answers
         correctly greedily (replay; pool_r1_mixedv3_replay_908.jsonl): K 8 at T 1.3, keep one
         correct EOS answer per prompt, SFT on them, LR 5e-6 (the merged RFT mode).
      R2 GRPO mixed-v3 lr5e-6, beta 0.04 (KL anchor), T 1.3, DS 8.
      R3 GRPO on mixed-v3 + s_train 50/50 (pool_r3_mixedv3_strain_908.jsonl), lr1e-5, beta 0,
         T 1.3, DS 8."""
    cls = "rtx3070-bfloat16"
    r1 = dict(BASE, id="gpu-rtx3070-xbf16-R1-rft-I-mixedv3replay-lr5e-06-k8-t1.3", init="I",
              family="xbf16-pool", label="EXPLORATORY-bf16 R1", track="rl", mode="rft",
              lr=5e-6, num_generations=8, temperature=1.3, beta=None, screen_threshold=None,
              dtype=DTYPE[cls], device_class=cls, train_manifest="pool_r1_mixedv3_replay_908.jsonl",
              rft={"rounds": 400, "prompts_per_round": 32, "epochs_per_round": 1,
                   "max_keep_per_prompt": 1, "skip_all_correct": False},
              max_steps_total=1200, stops=[150, 300, 600])
    r2 = grpo(cls, 5e-6, 0.04, False, t=1.3, init="I")
    r2.update(id="gpu-rtx3070-xbf16-R2-mixedv3-lr5e-06-g8-b0.04-t1.3-ds8", family="xbf16-pool",
              label="EXPLORATORY-bf16 R2", train_manifest="pool_mixedv3_454.jsonl",
              extra_trainer_args=["--dynamic-sampling-max-draws", "8", "--loss-type", "grpo"])
    r3 = grpo(cls, 1e-5, 0.0, False, t=1.3, init="I")
    r3.update(id="gpu-rtx3070-xbf16-R3-mixedv3strain-lr1e-05-g8-b0-t1.3-ds8", family="xbf16-pool",
              label="EXPLORATORY-bf16 R3", train_manifest="pool_r3_mixedv3_strain_908.jsonl",
              extra_trainer_args=["--dynamic-sampling-max-draws", "8", "--loss-type", "grpo"])
    return [r1, r2, r3]


def early_seeds() -> list:
    """Amendment 16 (root, 2026-10-19 ~07:30Z; before it ran): the best-trending recipe,
    xbf16 GRPO-I lr2e-5 T1.0 on s_train (0.7837 at 600), gets its seed-314 replicate early
    (same spec as a promotion seed)."""
    lead = next(a for a in exploratory_bf16() if a["lr"] == 2e-5 and a["temperature"] == 1.0)
    return [promotion_specs(lead)[1]]


def t10_replan() -> list:
    """Amendment 17 (root, 2026-10-19 ~08:30Z; before these ran): every T1.3 arm declined while
    xbf16 lr2e-5 T1.0 on s_train reached 0.7837, so the anchored variants move to T 1.0.
    P3 (hard-v3 only) retired before step 150. EXPLORATORY-bf16, 600 updates, DS 8."""
    cls, ds = "rtx3070-bfloat16", ["--dynamic-sampling-max-draws", "8", "--loss-type", "grpo"]
    rows = [("R3t1", "pool_r3_mixedv3_strain_908.jsonl", "mixedv3strain", 2e-5, 0.0),
            ("R2t1", "pool_mixedv3_454.jsonl", "mixedv3", 5e-6, 0.04),
            ("P5t1", "s_train_scan_seed42.jsonl", "s_train", 1e-4, 0.0)]
    arms = []
    for tag, man, pool, lr, beta in rows:
        arm = grpo(cls, lr, beta, False, t=1.0, init="I")
        arm.update(id=f"gpu-rtx3070-xbf16-{tag}-{pool}-lr{lr:g}-g8-b{beta:g}-t1-ds8",
                   family="xbf16" if pool == "s_train" else "xbf16-pool",
                   label=f"EXPLORATORY-bf16 {tag}", train_manifest=man, extra_trainer_args=ds)
        arms.append(arm)
    return arms


def multi_image_arms() -> list:
    """Amendment 18 (root, 2026-10-19 ~09:30Z; before these ran): every single-image GRPO-I
    arm sits at I +- noise, so raise the per-update signal with main's M3 (#39/#47): P images
    per update and the zero-advantage step skip. G8, beta 0, --loss-type grpo, T 1.0, s_train,
    600 updates (dynamic sampling cannot combine with P > 1). A 2-update memory probe at P=4
    and P=2 on the 3070 decides which P fits 8 GB in bf16. EXPLORATORY-bf16."""
    cls = "rtx3070-bfloat16"
    arms = []
    for p, lr, steps in ((4, 2e-5, None), (2, 2e-5, None), (4, 5e-5, None), (2, 5e-5, None),
                         (4, 2e-5, 2), (2, 2e-5, 2),
                         # Amendment 20 (root; before it ran): P=2 is the M3 size; add lr1e-5
                         (2, 1e-5, None)):
        arm = grpo(cls, lr, 0.0, False, t=1.0, init="I")
        probe = steps is not None
        arm.update(id=(f"gpu-rtx3070-{'probe' if probe else 'xbf16'}-P{p}-s_train-lr{lr:g}"
                       f"-g8-b0-t1-zskip"),
                   family="probe" if probe else "xbf16", label=f"EXPLORATORY-bf16 P{p}",
                   extra_trainer_args=["--loss-type", "grpo", "--prompts-per-update", str(p),
                                       "--skip-zero-advantage-steps", "true"])
        # main's M3 guard: a P > 1 arm declares the feature the runner's host gate reads
        arm["features"] = [*arm.get("features", []), "prompts_per_update"]
        if probe:
            arm.update(stops=[2], max_steps_total=2)
        arms.append(arm)
    return arms


def r2_neighbourhood() -> list:
    """Amendment 21 (root, 2026-10-19 ~12:30Z; before these ran): explore around R2t1
    (mixed-v3, lr5e-6, beta 0.04, T 1.0; 0.7762 -> 0.7837): its seed 314, lr1e-5 beta 0.04,
    lr5e-6 beta 0.1 (EXPLORATORY-bf16), and its fp32 twin on the 2080 Ti."""
    r2 = next(a for a in t10_replan() if a["id"].startswith("gpu-rtx3070-xbf16-R2t1-"))
    seed = dict(r2, id=r2["id"] + "-s314", seed=314)
    out = [seed]
    for lr, beta in ((1e-5, 0.04), (5e-6, 0.1)):
        a = grpo("rtx3070-bfloat16", lr, beta, False, t=1.0, init="I")
        a.update(id=f"gpu-rtx3070-xbf16-R2t1-mixedv3-lr{lr:g}-g8-b{beta:g}-t1-ds8",
                 family="xbf16-pool", label="EXPLORATORY-bf16 R2t1 neighbourhood",
                 train_manifest="pool_mixedv3_454.jsonl",
                 extra_trainer_args=["--dynamic-sampling-max-draws", "8", "--loss-type", "grpo"])
        out.append(a)
    out.append(promotion_specs(r2)[0])  # fp32 twin, gpu-rtx2080ti-promo-...
    return out


START_ADAPTERS = DOCS / "start-adapters.json"


def sftall_arms() -> list:
    """Amendment 22 (root, CORRECTIONS M2; written before any of these ran): RL from sftall*,
    the all_train lr5e-6 step-300 SFT adapter (dev800 0.785, SFT lane), staged on each GPU
    host at ROOT/start/sftall-lr5e-6-s300/ and pinned by the weights sha256 the SFT lane
    reported (start-adapters.json). EXPLORATORY-bf16 on the 3070s:
      (a) GRPO from sftall*, s_train, lr1e-5, beta 0, T 1.0, DS 8;
      (b) the R2t1 recipe from sftall*: mixed-v3, lr5e-6, beta 0.04, T 1.0, DS 8;
      plus a dev800 eval of sftall* on the 3070 (the comparator is max(sftall*, its
      continued-SFT controls), never I). Controls (c) come from control2 once (a)/(b) finish.
    Empty until start-adapters.json pins the adapter."""
    if not START_ADAPTERS.is_file():
        return []
    pin = json.loads(START_ADAPTERS.read_text())["sftall-lr5e-6-s300"]
    init = {"start": "sftall-lr5e-6-s300", "weights_sha256": pin["weights_sha256"]}
    cls, ds = "rtx3070-bfloat16", ["--dynamic-sampling-max-draws", "8", "--loss-type", "grpo"]
    a = grpo(cls, 1e-5, 0.0, False, t=1.0, init="I")
    a.update(id="gpu-rtx3070-xbf16-sftall-s_train-lr1e-05-g8-b0-t1-ds8", init=init,
             family="xbf16", label="EXPLORATORY-bf16 M2a", extra_trainer_args=ds)
    b = grpo(cls, 5e-6, 0.04, False, t=1.0, init="I")
    b.update(id="gpu-rtx3070-xbf16-sftall-R2t1-mixedv3-lr5e-06-g8-b0.04-t1-ds8", init=init,
             family="xbf16-pool", label="EXPLORATORY-bf16 M2b",
             train_manifest="pool_mixedv3_454.jsonl", extra_trainer_args=ds)
    ev = dict(BASE, id="gpu-rtx3070-eval-sftall-lr5e-6-s300", init=init, family="baseline",
              track="baseline", mode="eval", lr=0.0, beta=0.04, temperature=1.0,
              num_generations=4, dtype="bfloat16", device_class=cls, rung=0,
              screen_threshold=None)
    return [ev, a, b]


def pool_arms() -> list:
    """Amendment 14 (OVERNIGHT plan P1-P7, before they ran): EXPLORATORY-bf16 GRPO-I on the
    corrected v3 pools (rl-campaign-pools ad54cfb: mixed 454 sha 052a2953.., hard 207 sha
    9707f347..), where P0-I found the signal, plus the s_train upper LR bracket and G16.
    (P4, s_train lr5e-5 T1.3, is the amendment-13 arm.)"""
    rows = [("P1", "mixedv3", 2e-5, 1.3, 8), ("P2", "mixedv3", 5e-5, 1.3, 8),
            ("P3", "hardv3", 2e-5, 1.3, 8), ("P5", "s_train", 1e-4, 1.3, 8),
            ("P6", "mixedv3", 2e-5, 1.5, 8), ("P7", "mixedv3", 2e-5, 1.3, 16)]
    manifests = {"mixedv3": "pool_mixedv3_454.jsonl", "hardv3": "pool_hardv3_207.jsonl",
                 "s_train": "s_train_scan_seed42.jsonl"}
    arms = []
    for tag, pool, lr, t, g in rows:
        arm = grpo("rtx3070-bfloat16", lr, 0.0, False, g=g, t=t, init="I")
        arm.update(id=f"gpu-rtx3070-xbf16-{pool}-lr{lr:g}-g{g}-b0-t{t:g}-ds8",
                   family="xbf16-pool" if pool != "s_train" else "xbf16",
                   label=f"EXPLORATORY-bf16 {tag}", train_manifest=manifests[pool],
                   extra_trainer_args=["--dynamic-sampling-max-draws", "8",
                                       "--loss-type", "grpo"])
        arms.append(arm)
    return arms


def push(host: str, spec: dict, prio: str, state: dict, suffix: str = "") -> None:
    if DEVICE_CLASS[host] != spec["device_class"]:
        raise SystemExit(f"REFUSE: {spec['id']} is {spec['device_class']}, {host} is "
                         f"{DEVICE_CLASS[host]}")
    qid = spec["id"] + suffix
    if qid in state["queued"]:
        raise SystemExit(f"REFUSE: {qid} already queued on {state['queued'][qid]['host']}")
    key = f"{prio}-{qid}"
    raw = base64.b64encode((json.dumps(spec, indent=1, sort_keys=True) + "\n").encode()).decode()
    R = root_of(host)
    uw("ssh", host, f"mkdir -p {R}/gpuq/pending && echo {raw} | base64 -d > "
       f"{R}/gpuq/pending/.{key}.tmp && mv {R}/gpuq/pending/.{key}.tmp "
       f"{R}/gpuq/pending/{key}.json && echo queued {key}", timeout=180)
    state["queued"][qid] = {"host": host, "key": key, "at": now()}
    save_state(state)
    print(f"queued {key} on {host}")


def cmd_enqueue(a) -> int:
    t, state = table(), load_state()
    for i, arm_id in enumerate(a.ids.split(",")):
        if arm_id not in t:
            raise SystemExit(f"REFUSE: unknown arm {arm_id}")
        push(a.host, t[arm_id], str(int(a.prio) + i).zfill(max(2, len(a.prio))), state)
    return 0


def cmd_worker(a) -> int:
    state = load_state()
    name = f"rlegpu-worker-{a.host}-{int(time.time())}"
    cpu = DEVICE_CLASS[a.host] == "cpu-float32"
    need = {"eceubuntu2": 10500}.get(a.host, 6500)  # MiB free before an arm starts
    # The worker counts free PHYSICAL cores minus load1 (which counts hyperthreads), so on a
    # busy HT host it never clears 16; root places CPU arms by uwpick, so --need-cores lets
    # the placement decision stand (measured 2026-10-19: iccad0 -3, iccad13 -27 "free").
    wait = f"WAIT_KIND=cpu NEED_CORES={a.need_cores}" if cpu else f"NEED_MIB={need}"
    if a.slot:
        wait += f" SLOT={a.slot}"
        name += f"-s{a.slot}"
    inner = (f"R={root_of(a.host)}; {wait} exec bash "
             "\"$R/payload/code/scripts/exploratory_sweep/gpu_worker.sh\" "
             f"\"$R\" \"$PWD/results\"")
    # --mem: host RSS cap for the uwrun memory guard. GPU: a trainer + assay child hold the
    # model in host RAM while loading (24 GB is headroom, not a measurement). CPU fp32 arm:
    # 25 GB (root's figure; the CPU sweep measured ~13.6 GB max RSS per arm).
    extra = (["--threads", "16", "--mem", "25", "--scratch", "shm", "--disk", "3"] if cpu
             else ["--threads", "4", "--mem", "24"])
    uw("run", "--host", a.host, *extra, "--timeout", str(a.timeout),
       "--name", name, "--", "bash", "-c", inner, timeout=600)
    state["workers"].setdefault(a.host, []).append({"name": name, "at": now()})
    save_state(state)
    print(f"worker {name}")
    return 0


def cmd_queue(a) -> int:
    print(uw("ssh", a.host, f"cd {root_of(a.host)}/gpuq && "
                             "for d in pending running done failed; do "
             f"echo \"$d: $(ls $d | tr '\\n' ' ')\"; done; nvidia-smi --query-gpu="
             f"memory.used,utilization.gpu --format=csv,noheader 2>/dev/null; uptime",
             timeout=180))
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
             "gpu-trace.csv", "nvsmi-before.txt", "local_training.json",
             "dynamic_sampling.jsonl", "e0.json", "p0.json")


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
        for f in sorted(d.glob("p0-*.json")):  # P0-I per-row probe tables
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f.name)
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


def scanned_ids(d: Path) -> list:
    """Every row the RL arm READ, in scan order: dynamic-sampling draws (kept and discarded)
    ordered by (global_step, draw); without that journal, the committed prompt ids."""
    ds = d / "dynamic_sampling.jsonl"
    if ds.is_file():
        rows = [json.loads(x) for x in ds.read_text().splitlines() if x.strip()]
        ids = [r["id"] for r in sorted(rows, key=lambda r: (r["global_step"], r["draw"]))]
    else:
        ids = json.loads((d / "endpoint_receipt.json").read_text())["committed_prompt_ids"]
    # A training manifest must list each id once (the trainer refuses repeats); dynamic
    # sampling's random replacement draws can read a row twice. Keep the first read of each
    # row, in scan order: the same set of labels the RL arm had access to, in the order it
    # first saw them; the SFT control then cycles over that list.
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def control2_specs(rl: dict, d: Path, lrs: list) -> list:
    """Root corrections C4/C10 control for one finished RL arm: SFT from the same init at
    each LR, on all scanned ids in scan order (cycling), time-matched at EACH RL checkpoint
    (time1/time2/... = the RL endpoints' committed active seconds), bf16 on a 3070."""
    summary = json.loads((d / "summary.json").read_text())
    eps = [e for e in summary["endpoints"] if e["label"].startswith("step")]
    times = [float(e["active_seconds"]) for e in eps]
    if not eps or any(not a < b for a, b in zip([0.0] + times, times)):
        raise SystemExit(f"REFUSE: {rl['id']} endpoints lack increasing active seconds")
    ids = scanned_ids(d)
    if not ids:
        raise SystemExit(f"REFUSE: {rl['id']} has no scanned ids")
    # device class must match within a trajectory: GPU RL arms -> 3070 bf16 controls (root
    # decision), CPU RL arms -> CPU fp32 controls.
    cls = "cpu-float32" if rl["device_class"] == "cpu-float32" else "rtx3070-bfloat16"
    dev = ({"device": "cpu", "features": [], "threads": 16} if cls == "cpu-float32" else {})
    out = []
    for lr in lrs:
        out.append(dict(BASE, id=f"{rl['id']}~ctl-lr{lr:g}", init=rl["init"], family="control2",
                        track="control", mode="sft", lr=lr, seed=rl["seed"],
                        order="sequential", cycle=True, num_generations=4, beta=None,
                        temperature=1.0, screen_threshold=None, dtype=DTYPE[cls],
                        device_class=cls, control_of=rl["id"],
                        control_checkpoints=[e["label"] for e in eps],
                        train_manifest=f"@consumed:{rl['id']}", train_ids=ids, stops=[],
                        time_stops=times, max_steps_total=100000,
                        max_active_seconds=259200.0, **dev))
    return out


def cmd_control2(a) -> int:
    res, t, state = arm_results(), table(), load_state()
    d = res.get(a.of)
    if d is None or not (d / "summary.json").is_file():
        raise SystemExit(f"REFUSE: {a.of} has no fetched summary")
    for i, spec in enumerate(control2_specs(t[a.of], d, [float(x) for x in a.lrs.split(",")])):
        push(a.host, spec, f"{int(a.prio) + i:02d}", state)
    return 0


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
               response_format="answer_only", beta=None, temperature=1.0,
               screen_threshold=None, max_steps_total=100000, max_active_seconds=259200.0,
               train_manifest=f"@consumed:{a.of}", train_ids=ids, stops=[len(ids)],
               time_budget_seconds=float(last["active_seconds"]))
    push(a.host, ctl, a.prio, state)
    return 0


# Native VAL macro accuracy (equal-class mean) of the controlled400 endpoints, from the
# committed table in reports/2026-10-17-controlled400-native-val/README.md (Results).
NATIVE_VAL_MACRO = {"W": 0.6784, "RL400": 0.6982, "SFT1e5E": 0.7109, "SFT5e5E": 0.7399,
                    "SFT1e4E": 0.7301, "I": 0.7539, "SFT5e5T": 0.7534, "SFT1e4T": 0.7520}


def _ranks(values: list) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):  # average ranks over ties
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list, y: list) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def cmd_calib(a) -> int:
    """dev800 calibration (root corrections M4): the 8 endpoints with native VAL scores
    (W, RL400, the six SFT controls incl. I) assayed on dev800 on ONE device class; rank
    agreement with native VAL, and the sign of RL400 vs same-LR same-exposure SFT."""
    run = _runner()
    gold = {}
    for line in GOLD.read_text().splitlines():
        r = json.loads(line)
        gold[r["id"]] = r["label"]
    found = {}
    for arm_id, d in arm_dirs():
        spec = json.loads((d / "arm.json").read_text())
        if spec.get("mode") != "eval" or spec.get("device_class") != a.device_class:
            continue
        f = d / "assays" / "init.json"
        if f.is_file():
            found.setdefault(spec["init"], (arm_id, json.loads(f.read_text())))
    missing = [k for k in NATIVE_VAL_MACRO if k not in found]
    names = [k for k in NATIVE_VAL_MACRO if k in found]
    if len(names) < 3:
        raise SystemExit(f"REFUSE: only {names} assayed on {a.device_class}")
    rows = [{"endpoint": k, "arm": found[k][0], "native_val_macro": NATIVE_VAL_MACRO[k],
             **{m: found[k][1]["metrics"][m] for m in ("accuracy", "equal_class_accuracy",
                                                        "valid_fraction")}} for k in names]
    out = {"device_class": a.device_class, "missing": missing, "rows": rows,
           "spearman_native_vs_dev800_equal_class": spearman(
               [r["native_val_macro"] for r in rows], [r["equal_class_accuracy"] for r in rows]),
           "spearman_native_vs_dev800_accuracy": spearman(
               [r["native_val_macro"] for r in rows], [r["accuracy"] for r in rows]),
           "native_source": "reports/2026-10-17-controlled400-native-val/README.md"}
    if "RL400" in found and "SFT1e5E" in found:
        out["rl400_minus_sft1e5E"] = {
            "native_val_macro": NATIVE_VAL_MACRO["RL400"] - NATIVE_VAL_MACRO["SFT1e5E"],
            "dev800": run.paired_bootstrap(found["RL400"][1]["predictions"],
                                           found["SFT1e5E"][1]["predictions"], gold,
                                           a.resamples, seed=20261018)}
    path = STUDY_DIR / f"calibration-{a.device_class}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for r in sorted(rows, key=lambda r: r["native_val_macro"]):
        print(f"{r['endpoint']:>8} native {r['native_val_macro']:.4f}  dev800 acc "
              f"{r['accuracy']:.4f} eqcls {r['equal_class_accuracy']:.4f} "
              f"valid {r['valid_fraction']:.4f}")
    print(f"spearman(native, dev800 eqcls) {out['spearman_native_vs_dev800_equal_class']:+.3f}; "
          f"(native, dev800 acc) {out['spearman_native_vs_dev800_accuracy']:+.3f}; "
          f"missing {missing}")
    if "rl400_minus_sft1e5E" in out:
        c = out["rl400_minus_sft1e5E"]
        print(f"RL400 - SFT1e5E: native {c['native_val_macro']:+.4f}; dev800 "
              f"{c['dev800']['delta']:+.4f} [{c['dev800']['ci95'][0]:+.4f},"
              f"{c['dev800']['ci95'][1]:+.4f}]")
    print(f"wrote {path}")
    return 0


def cmd_placebo(a) -> int:
    """SFT-from-I seed placebo (root corrections M5/C11): the same recipe at several seeds,
    each scored best-of-k over its own checkpoints (the selection a promotion makes). The
    spread of best-of-k across seeds is the floor a promotion threshold must exceed."""
    import itertools

    curves = {}
    for arm_id, d in arm_dirs():
        spec = json.loads((d / "arm.json").read_text())
        if spec.get("family") != "sft-I" or float(spec["lr"]) != a.lr:
            continue
        for f in (d / "assays").glob("step*.json"):
            m = json.loads(f.read_text())["metrics"]
            curves.setdefault(spec["seed"], {})[int(f.stem[4:])] = m["accuracy"]
    stops = sorted(set.intersection(*(set(c) for c in curves.values()))) if curves else []
    if len(curves) < 2 or not stops:
        raise SystemExit(f"REFUSE: need >= 2 seeds with common checkpoints; have {curves}")
    best = {s: max(c[k] for k in stops) for s, c in curves.items()}
    pairs = [{"seeds": [x, y], "best_of_k_delta": best[x] - best[y]}
             for x, y in itertools.combinations(sorted(best), 2)]
    out = {"lr": a.lr, "common_stops": stops, "k": len(stops), "curves": curves,
           "best_of_k": best, "pairs": pairs,
           "max_abs_best_of_k_delta": max(abs(p["best_of_k_delta"]) for p in pairs),
           "per_stop_range": {k: max(c[k] for c in curves.values()) -
                              min(c[k] for c in curves.values()) for k in stops}}
    path = STUDY_DIR / f"placebo-sft-I-lr{a.lr:g}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for s, c in sorted(curves.items()):
        print(f"seed {s:>5}: " + " ".join(f"{k}:{c[k]:.4f}" for k in sorted(c)) +
              f"  best-of-{len(stops)} {best[s]:.4f}")
    print(f"max |best-of-k delta| {out['max_abs_best_of_k_delta']:.4f}; wrote {path}")
    return 0


def cmd_p0sum(a) -> int:
    """P0-I summary: the probe's own aggregate over (1) the clean pool halves A u B ("all"),
    (2) the campaign's `mixed` pool and (3) its `hard` subset (greedy-wrong & p >= 0.02,
    docs/<study>/pools-ref/hard-v3.jsonl, sha256 9707f347..., pools v3), per temperature."""
    probes = _load("gpu_probes", HERE.parent / "gpu_probes.py")
    hard = {json.loads(x)["id"]
            for x in (DOCS / "pools-ref" / "hard-v3.jsonl").read_text().splitlines()
            if x.strip()}
    tables = {}
    for _, d in arm_dirs():
        for f in d.glob("p0-p0_*-T*.json"):
            if "smoke" in d.name:
                continue
            name, t = f.stem.removeprefix("p0-").rsplit("-T", 1)
            tables.setdefault(t, {})[name] = json.loads(f.read_text())
    out = {}
    for t, by in sorted(tables.items()):
        sets = {}
        if "p0_poolA200" in by and "p0_poolB200" in by:
            sets["all_clean_pool_AB400"] = by["p0_poolA200"] + by["p0_poolB200"]
        if "p0_mixedv3_454" in by:
            sets["mixed_v3_454"] = by["p0_mixedv3_454"]
            sets["hard_v3_subset_of_mixed"] = [r for r in by["p0_mixedv3_454"] if r["id"] in hard]
        out[t] = {k: probes.aggregate(v, 8) for k, v in sets.items() if v}
    if not any(out.values()):
        raise SystemExit("REFUSE: no complete P0-I probe table fetched yet")
    path = STUDY_DIR / "p0-summary.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    keys = ("n", "acceptance_u_ge_0_5", "realised_mixed_fraction", "recognition_mixed_fraction",
            "format_only_mixed_fraction", "greedy_wrong", "pass_at_G_among_greedy_wrong",
            "pass_at_G_greedy_wrong_images_fraction_of_all", "p_gold_lt_0_02", "mean_p_gold")
    for t, sets in out.items():
        for k, v in sets.items():
            print(f"T{t} {k}: " + " ".join(
                f"{x}={v[x]:.4f}" if isinstance(v[x], float) else f"{x}={v[x]}" for x in keys))
    print(f"wrote {path}")
    return 0


class _CachedBootstrap:
    """runner.paired_bootstrap, memoised on disk. Its draws depend only on (n, wins, losses,
    resamples, seed) (a multinomial over win/tie/loss), so that tuple is the exact key; the
    full report otherwise re-draws 20,000 resamples for every pair on every tick."""

    def __init__(self, run, path: Path):
        self.run, self.path = run, path
        self.cache = json.loads(path.read_text()) if path.is_file() else {}

    def paired_bootstrap(self, a: dict, b: dict, gold: dict, resamples: int, seed: int) -> dict:
        d = [(a[i] == g) - (b[i] == g) for i, g in gold.items()]
        key = f"{len(d)}:{d.count(1)}:{d.count(-1)}:{resamples}:{seed}"
        if key not in self.cache:
            self.cache[key] = self.run.paired_bootstrap(a, b, gold, resamples, seed)
            self.misses = getattr(self, "misses", 0) + 1
            if self.misses % 10 == 0:
                self.save()  # survive a kill mid-report
        return dict(self.cache[key])

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, sort_keys=True))
        tmp.replace(self.path)


def cmd_report(a) -> int:
    run = _CachedBootstrap(_runner(), STUDY_DIR / "bootstrap-cache.json")
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
    # the best SFT-from-I endpoint (any seed, any checkpoint; first of ties in arm order)
    best_sft = None
    for (cid, clabel), (cspec, _, cdoc) in rows.items():
        if cspec.get("family") == "sft-I" and (
                best_sft is None
                or cdoc["metrics"]["accuracy"] > best_sft[2]["metrics"]["accuracy"]):
            best_sft = (cid, clabel, cdoc, cspec["device_class"])
    if best_sft:
        out["best_sft_I"] = {"arm": best_sft[0], "label": best_sft[1],
                             "accuracy": best_sft[2]["metrics"]["accuracy"],
                             "device_class": best_sft[3]}
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
        # root-corrections control (control2): SFT from the same init at each LR, on every
        # scanned id, time-matched at each RL checkpoint (time<k> <-> the k-th step label)
        for (cid, clabel), (cspec, _, cdoc) in rows.items():
            if (cspec.get("family") == "control2" and cspec.get("control_of") == spec["id"]
                    and clabel.startswith("time")):
                k = int(clabel[4:]) - 1
                if cspec["control_checkpoints"][k] == label:
                    entry.setdefault("vs_ctl2", {})[f"lr{float(cspec['lr']):g}"] = dict(
                        control_arm=cid, control_label=clabel,
                        control_accuracy=cdoc["metrics"]["accuracy"],
                        control_device=cspec["device_class"],
                        **run.paired_bootstrap(doc["predictions"], cdoc["predictions"], gold,
                                               a.resamples, seed=20261018))
        if best_sft and spec.get("family") != "sft-I":
            bs = best_sft
            entry["vs_best_sft_I"] = dict(
                sft_arm=bs[0], sft_label=bs[1], sft_accuracy=bs[2]["metrics"]["accuracy"],
                sft_device=bs[3], same_device=bs[3] == spec["device_class"],
                **run.paired_bootstrap(doc["predictions"], bs[2]["predictions"], gold,
                                       a.resamples, seed=20261018))
        w = base.get((same, "W")) or base.get((spec["device_class"], "W"))
        if w is not None:
            entry["vs_W_delta"] = run.paired_bootstrap(doc["predictions"], w[1]["predictions"],
                                                       gold, 2000, seed=20261018)["delta"]
        out["arms"].append(entry)
    out["arms"].sort(key=lambda e: (e["device_class"], e["arm"], e["steps"] or 10**9))
    run.save()
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
        for k in ("vs_ctl_E", "vs_ctl_T", "vs_best_sft_I"):
            if k in e:
                c = e[k]
                ref = c.get("control_accuracy", c.get("sft_accuracy"))
                print(f"{'':>18} {'':<48} {k:>8} ref {ref:.4f} delta "
                      f"{c['delta']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
        for lr, c in sorted((e.get("vs_ctl2") or {}).items()):
            print(f"{'':>18} {'':<48} ctl2-{lr:>6} {c['control_label']} ref "
                  f"{c['control_accuracy']:.4f} delta {c['delta']:+.4f} "
                  f"[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
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
    w.add_argument("--need-cores", type=int, default=16)
    w.add_argument("--slot", default="", help="second worker on a host shares its queue")
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
    cb = sub.add_parser("calib")
    cb.add_argument("--device-class", default="rtx3070-bfloat16", choices=sorted(DTYPE))
    cb.add_argument("--resamples", type=int, default=20000)
    pl = sub.add_parser("placebo")
    pl.add_argument("--lr", type=float, default=1e-5)
    sub.add_parser("p0sum")
    c2 = sub.add_parser("control2")
    c2.add_argument("--of", required=True)
    c2.add_argument("--host", required=True, choices=sorted(DEVICE_CLASS))
    c2.add_argument("--lrs", default="5e-6,1e-5,2e-5")
    c2.add_argument("--prio", default="05")
    sub.add_parser("promote")
    r = sub.add_parser("report")
    r.add_argument("--resamples", type=int, default=20000)
    a = p.parse_args(argv)
    return {"specs": cmd_specs, "enqueue": cmd_enqueue, "worker": cmd_worker,
            "queue": cmd_queue, "fetch": cmd_fetch, "control": cmd_control,
            "report": cmd_report, "raw": cmd_raw, "calib": cmd_calib, "placebo": cmd_placebo,
            "p0sum": cmd_p0sum, "control2": cmd_control2, "promote": cmd_promote,
            "continue": cmd_continue}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
