"""Run one exploratory-sweep arm on a bootstrapped host: train stages, then the dev assay.

  ROOT/venv/bin/python -I arm_exec.py --root ROOT (--arm-json FILE | --arm-b64 B64) --results DIR

The arm spec (base64 JSON from runner.py) names the init, mode, hyperparameters, manifest,
`stops` (cumulative update counts to stop, assay and keep an adapter at) and, for a matched
SFT control, the consumed RL sequence (`train_ids`) and the RL arm's active seconds
(`time_budget_seconds`). Training runs in ROOT/arms/<arm>/ (host-local); only small files
go to --results, which uwrun fetches back. Every stage runs the trainer as a subprocess with
`--train-only --exploratory-sweep aircraft-exploratory-sweep-v2`; a stage whose endpoint
receipt is missing or wrong stops the arm (non-zero exit), never a silent pass.

Assay: greedy decoding on the 800-image selection holdout (h_select + dev600) through the
trainer's own `predict_train_subset` in chunks of 32 (its TRAIN-inference limit); the answer
is `parse_label(response)` only when the completion ended at EOS, else invalid (None) --
the same exact-EOS canonical scoring as scripts/freshpool_launch/h_assay.py. `--final-check`
assays h_gate_final instead (rung-3 finalists only).
Kinds: eval (assay the init adapter only), train (default), bench (N steps, timing only).
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SWEEP = "aircraft-exploratory-sweep-v2"
CHUNK = 32
PROFILE_KEYS = ("response_format", "min_pixels", "max_pixels", "max_completion_length",
                "num_generations", "temperature")


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


# controlled400 endpoints with native VAL scores (payload.py INITS): eval-only inits for the
# dev800 calibration (root corrections M4). Never a training init.
CALIBRATION = ("RL400", "SFT1e5E", "SFT5e5E", "SFT1e4E", "SFT5e5T", "SFT1e4T")


def _int_at_least(n: int):
    def check(v: str):
        if not re.fullmatch(r"-?[0-9]+", v) or int(v) < n:
            raise SystemExit(f"REFUSE: expected an integer >= {n}, got {v!r}")
    return check


# Table-supplied trainer flags, an ALLOWLIST (exact names; each takes one value checked by
# its validator). argparse keeps the LAST value of a repeated flag and accepts unique
# abbreviations and `--flag=value`, so an extra appended after arm_exec's own flags could
# silently override --dtype, --learning-rate, --train, ... while the spec-field guards
# pass (PR #38 review). Extend this map only with a validator; never add a flag arm_exec
# sets itself. Blind spot: a validator checks the value's form, not the trainer's semantics.
def _float_in(lo: float, hi: float, *, lo_open: bool = False, hi_open: bool = False):
    def check(v: str):
        try:
            x = float(v)
        except ValueError:
            raise SystemExit(f"REFUSE: expected a number, got {v!r}") from None
        ok = (x > lo if lo_open else x >= lo) and (x < hi if hi_open else x <= hi)
        if not ok or x != x:
            raise SystemExit(f"REFUSE: {v!r} is outside {'(' if lo_open else '['}{lo}, {hi}"
                             f"{')' if hi_open else ']'}")
    return check


def _one_of(*values: str):
    def check(v: str):
        if v not in values:
            raise SystemExit(f"REFUSE: expected one of {list(values)}, got {v!r}")
    return check


def _int_between(lo: int, hi: int):
    def check(v: str):
        # ASCII digits only: str.isdigit() also accepts e.g. "²", which int() rejects
        if not re.fullmatch(r"[0-9]+", v) or not lo <= int(v) <= hi:
            raise SystemExit(f"REFUSE: expected an integer in {lo}..{hi}, got {v!r}")
    return check


def _payload_file(name: str):
    """A resolved @payload/ path (arm_exec.payload_arg ran first) to exactly this file."""
    def check(v: str):
        path = Path(v)
        if (not path.is_absolute() or path.resolve() != path or path.name != name
                or not path.is_file()):
            raise SystemExit(f"REFUSE: expected the payload's {name}, got {v!r}")
    return check


def _payload_json(v: str):
    """The RAW table value of a file flag: an `@payload/code/...json` asset that stays inside
    payload/code (no `..`, `.`, empty or absolute parts, no manifest). Checked before
    resolution; `_payload_file` checks the resolved path after Arm.payload_arg."""
    rel = v[len("@payload/"):] if v.startswith("@payload/") else ""
    parts = rel.split("/")
    if (not rel.endswith(".json") or parts[0] != "code" or len(parts) < 2
            or any(x in ("", ".", "..") for x in parts) or "manifests" in parts
            or "\\" in rel or posixpath.normpath(rel) != rel):
        raise SystemExit(f"REFUSE: expected an @payload/code/...json asset, got {v!r}")


MIN_DYNAMIC_DRAWS = 8  # root correction C5: fewer draws leave most G8 groups uninformative
EXTRA_FLAGS = {
    "--dynamic-sampling-max-draws": _int_at_least(MIN_DYNAMIC_DRAWS),  # CORRECTIONS C5
    # wave-2 table (amendment 4); values are the plan's, ranges no wider than the trainer's
    "--loss-type": _one_of("grpo", "dr_grpo", "dapo"),
    "--scale-rewards": _one_of("group", "none"),
    "--epsilon-high": _float_in(0.2, 10.0),  # clip-higher: >= epsilon 0.2
    "--num-iterations": _int_between(1, 8),
    "--label-hierarchy": _payload_file("label-hierarchy-train-v1.json"),
    "--family-credit": _float_in(0.0, 1.0, hi_open=True),
    "--manufacturer-credit": _float_in(0.0, 1.0, hi_open=True),
    "--invalid-reward": _float_in(-1.0, 0.0),
    # correction M3; defaults (P 1, no skip) are never passed, so legacy identity holds
    "--prompts-per-update": _int_between(2, 32),
    "--skip-zero-advantage-steps": _one_of("true"),
}
NEVER_EXTRA = {"--train", "--validation"}
# extras whose value is a file: given as @payload/<rel> only (Arm.resolved_extras)
PAYLOAD_FLAGS = {"--label-hierarchy"}


def checked_extras(extras, base_args, resolved: bool = True) -> list[str]:
    """`extras` as trainer argv, or refuse: only EXTRA_FLAGS, each once, as `--flag value`
    (or a bare flag whose validator is None), never a flag arm_exec already passes, never
    --train/--validation. `resolved` False checks a table's RAW values (refuse_spec, before
    any stage): a file flag (PAYLOAD_FLAGS) then passes `_payload_json` instead of the
    resolved-path check."""
    own = {a for a in base_args if isinstance(a, str) and a.startswith("--")}
    out, seen, i = [], set(), 0
    extras = [str(x) for x in extras or []]
    while i < len(extras):
        flag = extras[i]
        if flag in NEVER_EXTRA or flag.split("=")[0] in NEVER_EXTRA:
            raise SystemExit(f"REFUSE: extra_trainer_args may not pass {flag}")
        if "=" in flag or flag not in EXTRA_FLAGS:
            raise SystemExit(f"REFUSE: extra trainer flag {flag!r} is not allowlisted "
                             f"(exact `--flag value` from {sorted(EXTRA_FLAGS)})")
        if flag in own:
            raise SystemExit(f"REFUSE: extra {flag} would override arm_exec's own value")
        if flag in seen:
            raise SystemExit(f"REFUSE: extra {flag} given twice")
        seen.add(flag)
        if EXTRA_FLAGS[flag] is None:
            out.append(flag)
            i += 1
            continue
        if i + 1 >= len(extras):
            raise SystemExit(f"REFUSE: extra {flag} has no value")
        check = _payload_json if (flag in PAYLOAD_FLAGS and not resolved) else EXTRA_FLAGS[flag]
        check(extras[i + 1])
        out += [flag, extras[i + 1]]
        i += 2
    return out


def save_steps_for(spec: dict, max_steps: int) -> int:
    """--save-steps for every stage of this arm's trajectory.

    save_steps is part of the trainer's resume identity (settings), so it must be the same in
    every stage of one trajectory, across continuations too. A trajectory's first arm uses the
    gcd of its declared stops (and resume_stop); a continuation carries the value its
    trajectory started with in `save_steps` (gpu_lane.continue_spec / runner.promoted_arms set
    it), because the gcd rule alone drifts (resume 300 -> [600] gives 300; resume 600 ->
    [1200] gives 600). A stage boundary always saves its own checkpoint regardless. (It was
    `stop`, which changed between stages and made every multi-stop arm refuse to resume:
    "resume must preserve exact model/data/profile/settings". Same rule as the CPU sweep's
    exploratory-sweep-v2.noindex cb6d7ae.)
    """
    if spec.get("save_steps"):
        return int(spec["save_steps"])
    marks = [int(x) for x in spec.get("stops") or []] + (
        [int(spec["resume_stop"])] if spec.get("resume_stop") else [])
    return math.gcd(*marks) if marks else int(max_steps)


COTENANCY_PERIOD = 15  # seconds between co-tenancy samples during a stage


def refuse_spec(spec: dict) -> None:
    """Binding RL rules (runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md) checked before any
    stage runs. RL (grpo, rft) trains only in float32: bf16 GRPO failed the sampler-teacher
    parity gate on every RTX 3070 arm (0.028 > 0.02, C2). bf16/fp16 stay allowed for SFT and
    eval. Dynamic sampling needs at least MIN_DYNAMIC_DRAWS draws per update (C5).
    `extra_trainer_args` must pass the allowlist (checked_extras)."""
    if spec["mode"] in ("grpo", "rft") and spec.get("dtype", "float32") != "float32":
        raise SystemExit(f"REFUSE: {spec['mode']} runs only in float32 (C2), not "
                         f"{spec.get('dtype')}")
    checked_extras(spec.get("extra_trainer_args"), [], resolved=False)  # before any stage


def refuse_repeated_flags(argv: list) -> None:
    """Outcome guard on the argv that actually runs: no flag may appear twice (argparse keeps
    the last value, so a repeat overrides what the launcher set or checked)."""
    flags = [x for x in argv if str(x).startswith("--")]
    seen = {f for f in flags if flags.count(f) > 1}
    if seen:
        raise SystemExit(f"REFUSE: trainer argv repeats {sorted(seen)}")


def stage_record(record_path: Path, started: float | None = None,
                 stage_id: str | None = None, float32: bool = True) -> dict:
    """The trainer's local_training.json for THIS stage. Refuses a record older than the
    stage's start (a stage that exited 0 without rewriting it), one whose stage_id is not
    `stage_id` (the id arm_exec chose and passed as --stage-id), and, for RL, one whose
    RECORDED dtype is not float32 (C2: the trainer's record decides, not the spec)."""
    path = Path(record_path)
    if started is not None and path.stat().st_mtime < started:
        raise SystemExit(f"REFUSE: {path} predates this stage (stale record)")
    record = json.loads(path.read_text())
    if stage_id is not None and record.get("stage_id") != stage_id:
        raise SystemExit(f"REFUSE: {path} stage_id {record.get('stage_id')!r} is not this "
                         f"stage's {stage_id!r}")
    dtype = ((record.get("control_identity") or {}).get("profile") or {}).get("dtype")
    if float32 and dtype != "float32":
        raise SystemExit(f"REFUSE: the RL stage recorded dtype {dtype!r}, not float32 (C2)")
    return record


def stage_id_for(work: Path, label: str) -> str:
    """The id arm_exec chooses for one stage and passes as --stage-id."""
    return f"{work.name}-{label}"


def check_parity(record_path: Path, started: float | None = None,
                 stage_id: str | None = None) -> dict:
    """A GRPO stage must have run in float32 (stage_record) and record a sampler-teacher
    log-prob parity inside tolerance (C2)."""
    record = stage_record(record_path, started, stage_id)
    parity = record.get("sampler_teacher_logprob_parity")
    if not isinstance(parity, dict) or parity.get("within_tolerance") is not True:
        raise SystemExit(f"REFUSE: sampler-teacher parity not within tolerance: {parity}")
    return {k: parity.get(k) for k in ("within_tolerance", "tolerance",
                                        "max_abs_raw_teacher_vs_sampler_logprob_error")}


class CoTenancy:
    """Who else used this machine while a stage ran (root correction C16).

    CUDA: every `nvidia-smi --query-compute-apps` sample; a process not in the stage's own
    process tree is a co-tenant. CPU: a sample is co-tenant when load1 exceeds the stage's own
    threads + 1 (other work shows only as load). `shared` is True when any sample saw a
    co-tenant, None when any CUDA sample could not be taken (unknown, never "alone");
    the campaign report refuses
    a time comparison between an RL arm and a control whose `shared` values differ.
    Blind spot: a co-tenant that starts and ends between two samples is not seen."""

    def __init__(self, cuda: bool, threads: int = 0):
        self.cuda, self.samples, self.foreign, self.hits, self.load = cuda, 0, {}, 0, []
        self.threads = int(threads)
        self.unmeasured = 0  # CUDA samples where nvidia-smi could not be run

    @staticmethod
    def tree(root: int) -> set | None:
        """The stage's process tree, or None when `ps` cannot be run or parsed (the sample
        is then unmeasured: arm_exec must never die while its trainer runs on)."""
        try:
            out = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True,
                                 text=True, timeout=30).stdout.split()
            parent = dict(zip(map(int, out[0::2]), map(int, out[1::2])))
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        mine = {root}
        for pid in parent:
            p = pid
            while p and p != 1:
                if p == root:
                    mine.add(pid)
                    break
                p = parent.get(p, 0)
        return mine

    def sample(self, root: int) -> None:
        self.samples += 1
        load1 = os.getloadavg()[0]
        self.load.append(load1)
        if not self.cuda:
            self.hits += load1 > self.threads + 1
            return
        try:
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,"
                                "process_name", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.unmeasured += 1  # never abort the stage (its trainer is running)
            return
        if q.returncode != 0:
            self.unmeasured += 1
            return
        mine, hit = self.tree(root), False
        if mine is None:
            self.unmeasured += 1
            return
        for line in q.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) not in mine:
                hit = True
                self.foreign[parts[0]] = {"used_mib": parts[1],
                                          "name": parts[2] if len(parts) > 2 else ""}
        self.hits += hit

    def summary(self) -> dict:
        load = sorted(self.load)
        return {"device": "cuda" if self.cuda else "cpu", "samples": self.samples,
                "period_seconds": COTENANCY_PERIOD, "samples_with_cotenant": self.hits,
                "cotenant_gpu_processes": self.foreign,
                "shared": None if self.unmeasured else self.hits > 0,
                "samples_unmeasured": self.unmeasured,
                "load1_median": load[len(load) // 2] if load else None,
                "load1_max": load[-1] if load else None}


class Arm:
    def __init__(self, root: Path, spec: dict, results: Path):
        self.root, self.spec, self.results = root, spec, results
        self.payload = root / "payload"
        self.model = root / "model" / "Qwen2-VL-2B-Instruct-895c3a49"
        self.spec_path = self.payload / "data" / "data_spec.json"
        # A promoted arm continues its rung-1 run in place: same work dir, same trajectory.
        self.work = root / "arms" / spec.get("work_id", spec["id"])
        self.out = self.work / "train"
        self.python = root / "venv" / "bin" / "python"
        self.env = dict(os.environ, PYTHONPATH=str(self.payload / "code" / "src"),
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                        VLM_TORCH_THREADS=str(spec.get("threads", 16)))
        if spec.get("device") == "cuda":
            # Allocator fragmentation only (no numerics): the fp32 beta>0 GRPO on the 2080 Ti
            # OOMed needing 274 MiB with 547 MiB reserved-but-unallocated (2026-10-19).
            self.env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.env[var] = str(spec.get("threads", 16))
        self.log = []

    # ---- manifests -------------------------------------------------------------------
    def manifest(self) -> Path:
        s = self.spec
        if s.get("train_ids"):
            rows = {}
            for m in sorted((self.payload / "data" / "manifests").glob("*.jsonl")):
                if m.name.startswith(("dev", "h_")):
                    continue  # never train on a holdout row, even by id
                for line in m.read_text().splitlines():
                    r = json.loads(line)
                    rows[r["id"]] = r
            missing = [i for i in s["train_ids"] if i not in rows]
            if missing:
                raise SystemExit(f"REFUSE: consumed ids not in a training manifest: {missing[:5]}")
            path = self.payload / "data" / "manifests" / f"ctl-{s['id'].replace('~', '_')}.jsonl"
            path.write_text("".join(json.dumps(rows[i], sort_keys=True) + "\n"
                                    for i in s["train_ids"]))
            return path
        name = s["train_manifest"]
        if name.startswith(("dev", "h_")) or "/" in name:
            raise SystemExit(f"REFUSE: {name} is not a training manifest")
        return self.payload / "data" / "manifests" / name

    # ---- training --------------------------------------------------------------------
    def payload_arg(self, value) -> str:
        """`@payload/<rel>` names a file shipped in this host's payload (e.g. the TRAIN
        label hierarchy); it may not escape the payload directory."""
        text = str(value)
        if not text.startswith("@payload/"):
            return text
        path = (self.payload / text.removeprefix("@payload/")).resolve()
        if not path.is_relative_to(self.payload.resolve()) or not path.is_file():
            raise SystemExit(f"REFUSE: {text} is not a file in the payload")
        if text.startswith("@payload/code/") and not path.is_relative_to(
                (self.payload / "code").resolve()):  # symlinks resolved first
            raise SystemExit(f"REFUSE: {text} resolves outside payload/code")
        return str(path)

    def resolved_extras(self) -> list[str]:
        """The table's extras with @payload/ values resolved. A file-valued extra
        (PAYLOAD_FLAGS) must be given as @payload/<rel>, so payload_arg confines it to this
        host's payload: an absolute path elsewhere, even with the right file name, refuses."""
        raw = [str(x) for x in self.spec.get("extra_trainer_args") or []]
        for i, flag in enumerate(raw):
            if flag in PAYLOAD_FLAGS and (i + 1 >= len(raw)
                                          or not raw[i + 1].startswith("@payload/")):
                raise SystemExit(f"REFUSE: {flag} must name a payload file as @payload/<rel>")
        return [self.payload_arg(x) for x in raw]

    def cuda_args(self) -> list[str]:
        if self.spec.get("device", "cpu") != "cuda":
            return []
        # CUDA lane: explicit torch CPU threads (the GPU does the math), and a W/I init
        # trained in float32 needs its declared base-precision transfer.
        args = ["--threads", str(self.spec.get("threads", 4))]
        transfer = self.precision_transfer(self.init_adapter())
        if transfer:  # repeated on resume so the run's settings identity is unchanged
            args += ["--initial-adapter-precision-transfer", transfer]
        return args

    def rft_args(self, manifest: Path, max_steps: int, stop: int | None) -> list[str]:
        """--mode rft: rounds of K samples per prompt, keep exact-EOS correct, SFT on them.
        A rung stop is the trainer's round-granular --rft-stop-after-steps boundary (never
        part of the run's identity), so a promoted arm continues the same output dir."""
        s, r = self.spec, self.spec["rft"]
        args = ["-m", "vision_rl.vlm_local_training", "--mode", "rft",
                "--train", str(manifest), "--data-spec", str(self.spec_path),
                "--output", str(self.out), "--model-directory", str(self.model),
                "--device", s.get("device", "cpu"), "--dtype", s.get("dtype", "float32"),
                "--init-adapter", str(self.init_adapter()),
                "--learning-rate", repr(float(s["lr"])), "--seed", str(s["seed"]),
                "--max-steps", str(max_steps), "--lora-rank", "8",
                "--response-format", s["response_format"],
                "--num-generations", str(s["num_generations"]),
                "--temperature", repr(float(s["temperature"])),
                "--min-pixels", str(s["min_pixels"]), "--max-pixels", str(s["max_pixels"]),
                "--max-completion-length", str(s["max_completion_length"]),
                "--rft-rounds", str(int(r["rounds"])),
                "--rft-prompts-per-round", str(int(r["prompts_per_round"])),
                "--rft-epochs-per-round", str(int(r.get("epochs_per_round", 1))),
                "--rft-max-keep-per-prompt", str(int(r.get("max_keep_per_prompt", 1)))]
        if r.get("skip_all_correct"):
            args += ["--rft-skip-all-correct"]
        args += self.cuda_args()
        if stop and stop < max_steps:
            args += ["--rft-stop-after-steps", str(stop)]
        if s.get("max_active_seconds"):
            args += ["--max-active-seconds", repr(float(s["max_active_seconds"]))]
        args = args + checked_extras(self.resolved_extras(), args)
        refuse_repeated_flags(args)  # outcome guard on the argv that actually runs
        return args

    def trainer_args(self, manifest: Path, max_steps: int, stop: int | None,
                     resume: Path | None, stop_seconds: float | None = None) -> list[str]:
        s = self.spec
        multi_image = self.multi_image_args(s["mode"])  # refuses the M3 keys outside GRPO
        if s["mode"] == "rft":
            if resume is not None or stop_seconds is not None:
                raise SystemExit("REFUSE: RFT resumes its own output dir and has no time stop")
            return self.rft_args(manifest, max_steps, stop)
        mode = "sft" if s["mode"] == "sft" else s["mode"]
        args = ["-m", "vision_rl.vlm_local_training", "--mode", mode,
                "--train", str(manifest), "--data-spec", str(self.spec_path),
                "--output", str(self.out), "--model-directory", str(self.model),
                "--device", s.get("device", "cpu"), "--dtype", s.get("dtype", "float32"),
                "--train-only", "--exploratory-sweep", SWEEP,
                "--init-adapter", str(self.init_adapter()),
                "--learning-rate", repr(float(s["lr"])), "--seed", str(s["seed"]),
                "--max-steps", str(max_steps), "--lora-rank", "8",
                "--response-format", s["response_format"],
                "--num-generations", str(s["num_generations"]),
                "--min-pixels", str(s["min_pixels"]), "--max-pixels", str(s["max_pixels"]),
                "--max-completion-length", str(s["max_completion_length"])]
        args += self.cuda_args()
        args += ["--save-steps", str(save_steps_for(s, max_steps))]
        if stop and stop < max_steps:
            args += ["--stop-after-steps", str(stop)]
        args += multi_image
        if mode == "grpo":
            args += ["--beta", repr(float(s["beta"])),
                     "--temperature", repr(float(s["temperature"]))]
            if s.get("screen_threshold"):
                args += ["--screen-journal", str(self.out / "screen_journal.jsonl"),
                         "--screen-threshold", repr(float(s["screen_threshold"])),
                         "--target-accepted", str(max_steps)]
        else:
            args += ["--order", s.get("order", "random")]
            if s.get("cycle"):
                args += ["--cycle"]
            ids = s.get("train_ids") or []
            if len(set(ids)) < len(ids):
                # a control replaying a sequence that presents a row again (dynamic-sampling
                # redraws, an RFT pool wrap, more committed ids than distinct rows): the
                # trainer loads the repeats and trains them in this order
                args += ["--replay-sequence"]
        if s.get("max_active_seconds"):
            args += ["--max-active-seconds", repr(float(s["max_active_seconds"]))]
        if stop_seconds is not None:
            if not s.get("max_active_seconds") or stop_seconds > float(s["max_active_seconds"]):
                raise SystemExit("REFUSE: a time stop needs a larger declared max_active_seconds")
            args += ["--stop-after-active-seconds", repr(float(stop_seconds))]
        if resume is not None:
            args += ["--resume-from-checkpoint", str(resume)]
        args = args + checked_extras(self.resolved_extras(), args)
        refuse_repeated_flags(args)  # outcome guard on the argv that actually runs
        return args

    def scanned_ids(self, receipt: dict) -> dict | None:
        """Every prompt the RL arm scored against its gold label, in scan order, for a
        control that must replay ALL of them (CORRECTIONS C4), not just the committed ones.

        - RFT: every prompt sampled in each committed round (local_training.json
          rounds[].prompt_ids, the trainer's own bound record); K samples of each were
          scored against gold, kept or not.
        - Dynamic sampling (DAPO): every drawn group, in draw order (the trainer's
          dynamic_sampling.jsonl). A resumed update is journaled again, so the LAST block
          per update counts; each block's kept draw must be that update's committed id.
        - Screened GRPO: every candidate the screen SCORED, accepted or not, in scan
          order (the checkpoint's committed screen entries).
        None for any other arm (its committed ids are already everything it scored).
        """
        if self.spec["mode"] == "rft":
            rec = json.loads((self.out / "local_training.json").read_text())
            ids = [pid for entry in rec["rounds"] for pid in entry["prompt_ids"]]
            for entry in rec["rounds"]:  # what a round kept must be prompts it sampled
                kept = json.loads((self.out / f"round-{entry['round']:02d}" / "kept.json")
                                  .read_text())["kept"]
                if not set(kept) <= set(entry["prompt_ids"]):
                    raise SystemExit(f"REFUSE: RFT round {entry['round']} kept prompts it "
                                     f"did not sample")
            source = "local_training.json rounds[].prompt_ids"
        elif (self.out / "dynamic_sampling.jsonl").is_file():
            steps, committed = int(receipt["global_step"]), receipt["committed_prompt_ids"]
            if len(committed) != steps:
                raise SystemExit("REFUSE: dynamic sampling with reused updates is not replayable")
            blocks, block = {}, []
            lines = (self.out / "dynamic_sampling.jsonl").read_text().splitlines()
            for number, line in enumerate(lines, 1):
                try:
                    entry = json.loads(line)
                except ValueError:
                    raise SystemExit(f"REFUSE: dynamic_sampling.jsonl line {number} of "
                                     f"{len(lines)} is not JSON (truncated write?)") from None
                block.append(entry)
                if entry.get("kept"):
                    blocks[int(entry["global_step"])] = block
                    block = []
            ids = []
            for step in range(1, steps + 1):
                got = blocks.get(step)
                if (not got or [e["global_step"] for e in got] != [step] * len(got)
                        or [e["draw"] for e in got] != list(range(len(got)))
                        or got[-1]["id"] != committed[step - 1]):
                    raise SystemExit(f"REFUSE: dynamic_sampling.jsonl does not account for "
                                     f"update {step}")
                ids += [e["id"] for e in got]
            source = "dynamic_sampling.jsonl, every draw, last block per update"
        elif self.spec["mode"] == "grpo" and self.spec.get("screen_threshold"):
            sys.path.insert(0, str(self.payload / "code" / "src"))
            from vision_rl import vlm_local_training as vlt

            ckpt = self.out / f"checkpoint-{int(receipt['global_step'])}"
            read = getattr(vlt, "committed_screen_entries", None)
            if read is None:
                raise SystemExit("REFUSE: this payload's trainer has no "
                                 "committed_screen_entries; screened rows cannot be listed")
            try:
                entries = read(ckpt)
            except (vlt.LocalVLMError, OSError, ValueError, KeyError) as error:
                raise SystemExit(f"REFUSE: screened rows of {ckpt.name} unreadable: "
                                 f"{error}") from None
            ids = [e["id"] for e in sorted(entries, key=lambda e: e["scan_position"])]
            source = "committed screen entries, every scored candidate, scan order"
        else:
            return None
        doc = {"schema": "arm_exec-scanned-ids-v1", "global_step": receipt["global_step"],
               "scanned_prompt_ids": ids, "source": source}
        write_json(self.out / "scanned_ids.json", doc)
        return doc

    def multi_image_args(self, mode: str) -> list[str]:
        """Correction M3 arm keys -> the trainer's GRPO knobs (one route only).

        `prompts_per_update` P: P distinct TRAIN images x G generations per optimizer
        update; the arm's `stops`/`updates` stay in global steps, so a stop at N consumes
        N*P images (the matched control replays all N*P committed prompt ids).
        `skip_zero_advantage_steps`: an update whose whole batch has zero advantage takes no
        optimizer step but still counts as a step of P consumed images (beta 0 only). A
        value equal to the legacy default is not passed, so the run keeps the legacy
        settings identity."""
        s = self.spec
        ppu, skip = s.get("prompts_per_update", 1), s.get("skip_zero_advantage_steps", False)
        if type(ppu) is not int or not 1 <= ppu <= 32 or type(skip) is not bool:
            raise SystemExit("REFUSE: prompts_per_update must be an int in 1..32 and "
                             "skip_zero_advantage_steps a bool")
        # The same knobs may instead come as allowlisted extras (EXTRA_FLAGS validates their
        # form; `=` and abbreviated forms are refused there). One route per knob only.
        extra = [str(x) for x in s.get("extra_trainer_args") or []]
        given = {f: extra[i + 1] for i, f in enumerate(extra[:-1])
                 if f in ("--prompts-per-update", "--skip-zero-advantage-steps")}
        if ("--prompts-per-update" in given and "prompts_per_update" in s) or (
                "--skip-zero-advantage-steps" in given and "skip_zero_advantage_steps" in s):
            raise SystemExit("REFUSE: prompts_per_update/skip_zero_advantage_steps given both "
                             "as an arm key and in extra_trainer_args")
        if "--prompts-per-update" in given:
            EXTRA_FLAGS["--prompts-per-update"](given["--prompts-per-update"])
        if "--skip-zero-advantage-steps" in given:
            EXTRA_FLAGS["--skip-zero-advantage-steps"](given["--skip-zero-advantage-steps"])
        p_eff = int(given.get("--prompts-per-update", ppu))
        skip_eff = skip or "--skip-zero-advantage-steps" in given
        # The runner's host gate reads `features`: a P>1 arm must declare the feature, and an
        # arm that declares it must set P>1 (else a "ppu" arm would silently run P=1).
        if ("prompts_per_update" in s.get("features", [])) != (p_eff != 1):
            raise SystemExit("REFUSE: feature prompts_per_update and prompts_per_update > 1 "
                             "must come together")
        if (p_eff != 1 or skip_eff) and mode != "grpo":
            raise SystemExit(f"REFUSE: prompts_per_update/skip_zero_advantage_steps are GRPO "
                             f"only, not {mode}")
        if skip_eff and float(s["beta"]) != 0:
            raise SystemExit("REFUSE: skip_zero_advantage_steps needs beta 0")
        return ((["--prompts-per-update", str(ppu)] if ppu != 1 else [])
                + (["--skip-zero-advantage-steps", "true"] if skip else []))

    def rft_endpoint_done(self, stop: int | None) -> bool:
        """An RFT run whose bound endpoint already covers this stage (e.g. the arm died in
        the assay after training): the trainer refuses to re-enter a completed run, so the
        stage reuses that endpoint instead of training again. A boundary endpoint counts
        only for its own stop; any other completed stop reason ends the run."""
        path = self.out / "local_training.json"
        if self.spec["mode"] != "rft" or not path.is_file():
            return False
        rec = json.loads(path.read_text())
        if rec.get("status") != "completed":
            return False
        if rec.get("stop_reason") == "stop_after_steps":
            return (rec.get("boundary") or {}).get("stop_after_steps") == stop
        return True

    def run_stage(self, label: str, args: list[str], stop: int | None = None) -> dict:
        if self.rft_endpoint_done(stop):
            # an earlier attempt's completed endpoint (its stage may have been refused after
            # the trainer finished): it must still be float32 (C2) to be reused
            stage_record(self.out / "local_training.json")
            self.log.append({"stage": label, "rc": 0, "reused_completed_endpoint": True,
                             "argv": args})
            receipt = self.rft_receipt()
            self.scanned_ids(receipt)
            self.log[-1].update(receipt_status=receipt["status"],
                                global_step=receipt["global_step"],
                                adapter_sha256=receipt["final_adapter"]["weights_sha256"])
            return receipt
        started = time.time()
        args = list(args)
        sid = None
        flag = "--rft-stop-after-steps"
        if self.spec["mode"] == "rft":
            # the RFT CLI takes no --stage-id: the stop that runs (rft_args' round-granular
            # boundary) is this stage's identity, so it must be the stop the caller declared
            ran = int(args[args.index(flag) + 1]) if flag in args else None
            if stop is not None and ran is not None and ran != stop:
                raise SystemExit(f"REFUSE: RFT stage {label} ran with {flag} {ran}, not {stop}")
        else:
            if "--stage-id" in args:
                raise SystemExit("REFUSE: --stage-id is arm_exec's, not the caller's")
            sid = stage_id_for(self.work, label)
            args += ["--stage-id", sid]
        log = self.work / f"{label}.log"
        cuda = self.spec.get("device", "cpu") == "cuda"
        watch = CoTenancy(cuda, int(self.spec.get("threads") or 0))
        with log.open("w") as handle:
            proc = subprocess.Popen([str(self.python), "-s", "-P", *args], env=self.env,
                                    stdout=handle, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                watch.sample(proc.pid)
                try:
                    proc.wait(timeout=COTENANCY_PERIOD)
                except subprocess.TimeoutExpired:
                    pass
            rc = proc.returncode
        tail = log.read_text().splitlines()[-25:]
        shutil.copy2(log, self.results / f"{label}.log.txt")
        entry = {"stage": label, "rc": rc, "wall_seconds": round(time.time() - started, 1),
                 "argv": args, "cotenancy": watch.summary()}
        self.log.append(entry)
        if rc != 0:
            write_json(self.results / "stages.json", self.log)
            print("\n".join(tail), file=sys.stderr)
            raise SystemExit(f"REFUSE: stage {label} rc={rc}")
        # freshness and identity first: every later check reads these files
        record = self.out / "local_training.json"
        if self.spec["mode"] == "rft":
            # this stage's fresh float32 record (C2) must end at THIS stage's boundary, or
            # end the run early (rounds / max_steps / max_active_seconds:
            # total_budget_exhausted in rft_receipt)
            rec = stage_record(record, started)
            got = (rec.get("boundary") or {}).get("stop_after_steps")
            if rec.get("stop_reason") == "stop_after_steps":
                if ran is None or got != ran:
                    raise SystemExit(f"REFUSE: RFT record boundary {got} is not this stage's "
                                     f"stop {ran} (stale record)")
            elif rec.get("stop_reason") not in ("rounds", "max_steps", "max_active_seconds"):
                raise SystemExit(f"REFUSE: RFT stage {label} ended {rec.get('stop_reason')}")
            elif ran is not None and int(rec.get("optimizer_steps") or 0) > ran:
                # an early end may stop short of this stage's stop, never past it
                raise SystemExit(f"REFUSE: RFT stage {label} ended early at "
                                 f"{rec.get('optimizer_steps')} updates, past its stop {ran}")
            receipt = self.rft_receipt()
        else:
            receipt_file = self.out / "endpoint_receipt.json"
            if receipt_file.stat().st_mtime < started:
                raise SystemExit(f"REFUSE: {receipt_file} predates this stage (stale receipt)")
            receipt = json.loads(receipt_file.read_text())
            if receipt.get("stage_id") != sid:
                raise SystemExit(f"REFUSE: receipt stage_id {receipt.get('stage_id')!r} is not "
                                 f"this stage's {sid!r}")
            stage_record(record, started, sid, float32=self.spec["mode"] == "grpo")
        adapter = self.out / receipt["final_adapter"]["path"]
        weights = receipt["final_adapter"]["weights_sha256"]
        if sha_file(adapter / "adapter_model.safetensors") != weights:
            raise SystemExit(f"REFUSE: stage {label} adapter differs from its receipt")
        entry.update(receipt_status=receipt["status"], global_step=receipt.get("global_step"),
                     adapter_sha256=receipt["final_adapter"]["weights_sha256"])
        if self.spec["mode"] == "grpo":
            entry["sampler_parity"] = check_parity(record, started, sid)
        self.scanned_ids(receipt)
        write_json(self.results / "stages.json", self.log)
        return receipt

    def rft_receipt(self) -> dict:
        """The runner-facing endpoint of an RFT stage, derived from the trainer's own bound
        record (local_training.json: status completed, final_adapter_sha256) and its
        committed rounds (round-NN/kept.json). `committed_prompt_ids` lists the prompt of
        every SFT example in round order, one per optimizer update: what an
        examples-matched SFT control replays. Within a round the trainer shuffles its
        examples, so the order is canonical, not the trainer's; the per-round multiset is
        exact unless the update cap or the active-seconds budget cut the last round, where
        the first k entries of kept.json (sorted by prompt id) stand in for the trained k.
        A rung boundary never cuts a round."""
        rec = json.loads((self.out / "local_training.json").read_text())
        reasons = ("stop_after_steps", "max_steps", "max_active_seconds", "rounds")
        if rec.get("status") != "completed" or rec.get("stop_reason") not in reasons:
            raise SystemExit(f"REFUSE: RFT stage ended {rec.get('status')}/"
                             f"{rec.get('stop_reason')}")
        sha = sha_file(self.out / "final_adapter" / "adapter_model.safetensors")
        if sha != rec["final_adapter_sha256"]:
            raise SystemExit("REFUSE: RFT final adapter differs from its record")
        epochs = int(rec["settings"]["epochs_per_round"])
        ids = []
        for entry in rec["rounds"]:
            kept = json.loads((self.out / f"round-{entry['round']:02d}" / "kept.json")
                              .read_text())["kept"]
            one_pass = [pid for pid, texts in kept.items() for _ in texts]
            ids += (one_pass * epochs)[:entry["optimizer_steps"]]
        if len(ids) != rec["optimizer_steps"]:
            raise SystemExit("REFUSE: RFT kept examples do not account for its updates")
        receipt = {
            "schema": "arm_exec-rft-endpoint-v1",
            "derived_from": "local_training.json rounds + round-NN/kept.json",
            "status": ("boundary_reached" if rec["stop_reason"] == "stop_after_steps"
                       else "total_budget_exhausted"),
            "stop_reason": rec["stop_reason"], "global_step": rec["optimizer_steps"],
            "committed_active_seconds": rec["active_seconds"],
            "committed_prompt_ids": ids, "rounds": len(rec["rounds"]),
            "final_adapter": {"path": "final_adapter", "weights_sha256": sha}}
        write_json(self.out / "rft_endpoint_receipt.json", receipt)
        return receipt

    def init_adapter(self) -> Path:
        """W, I (or an assay-only calibration endpoint) from the payload, or {"arm", "label"}:
        a kept endpoint on this host."""
        init = self.spec["init"]
        if isinstance(init, str):
            if init not in ("W", "I") and not (init in CALIBRATION
                                                and self.spec["mode"] == "eval"):
                raise SystemExit(f"REFUSE: unknown init {init!r}")
            return self.payload / "adapters" / init / "final_adapter"
        if "start" in init:
            # A start adapter staged on this host by another lane (root corrections M2:
            # RL from sftall*): ROOT/start/<name>/final_adapter with its local_training.json
            # beside it. Refuses unless its weights hash equals the sha the spec pins (taken
            # from the producing lane's report), so a missing or different file never trains.
            name = str(init["start"])
            if "/" in name or name.startswith("."):
                raise SystemExit(f"REFUSE: bad start name {name!r}")
            path = self.root / "start" / name / "final_adapter"
            want = init.get("weights_sha256")
            if not want or not (path / "adapter_model.safetensors").is_file():
                raise SystemExit(f"REFUSE: start adapter {path} missing or unpinned")
            if sha_file(path / "adapter_model.safetensors") != want:
                raise SystemExit(f"REFUSE: start adapter {path} differs from its pin {want[:12]}")
            if not (path.parent / "local_training.json").is_file():
                raise SystemExit(f"REFUSE: start adapter {path} has no "
                                 "local_training.json beside it")
            return path
        path = self.root / "arms" / init["arm"] / "kept" / init["label"] / "final_adapter"
        receipt = json.loads((path.parent / "endpoint_receipt.json").read_text())
        weights = receipt["final_adapter"]["weights_sha256"]
        if sha_file(path / "adapter_model.safetensors") != weights:
            raise SystemExit(f"REFUSE: kept init {path} differs from its endpoint receipt")
        return path

    def precision_transfer(self, adapter: Path) -> str | None:
        """'<parent base dtype>-to-<this arm's dtype>' when they differ, else None."""
        parent = json.loads((adapter.parent / "local_training.json").read_text())
        source = parent["control_identity"]["profile"]["dtype"]
        dtype = self.spec.get("dtype", "float32")
        return None if source == dtype else f"{source}-to-{dtype}"

    def keep_adapter(self, label: str) -> Path:
        # kept/<label>/{final_adapter, local_training.json, endpoint_receipt.json}: the
        # trainer binds an init adapter to the run record beside it, so a kept endpoint can
        # seed a later arm on this host (e.g. SFT(all TRAIN) -> GRPO).
        dst = self.work / "kept" / label
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.out / "final_adapter", dst / "final_adapter")
        for name in ("local_training.json", "endpoint_receipt.json",
                     "rft_endpoint_receipt.json"):
            if (self.out / name).is_file():
                shutil.copy2(self.out / name, dst / name)
        return self.out / "final_adapter"

    def copy_small(self) -> None:
        for name in ("local_training.json", "endpoint_receipt.json", "optimizer_audit.jsonl",
                     "physical_prompt_events.jsonl", "reward_groups.jsonl",
                     "screen_journal.jsonl", "egsft_weights.jsonl", "trainer_state.json",
                     "dynamic_sampling.jsonl", "rft_endpoint_receipt.json",
                     "scanned_ids.json"):
            src = self.out / name
            if src.is_file() and src.stat().st_size < 64 * 1024**2:
                shutil.copy2(src, self.results / name)

    # ---- assay -----------------------------------------------------------------------
    def assay_isolated(self, adapter: Path, label: str, manifest_name: str) -> dict:
        """CUDA: assay in a child process, so this process never holds GPU memory (the
        caching allocator and CUDA context) while the next training stage runs. CPU: in-process."""
        if self.spec.get("device", "cpu") != "cuda":
            return self.assay(adapter, label, manifest_name)
        spec_file = self.results / "arm.json"
        log = self.work / f"assay-{label}.log"
        with log.open("w") as handle:
            rc = subprocess.run(
                [str(self.python), "-I", str(Path(__file__).resolve()), "--root", str(self.root),
                 "--arm-json", str(spec_file), "--results", str(self.results),
                 "--assay-adapter", str(adapter), "--assay-label", label,
                 "--assay-manifest", manifest_name],
                stdout=handle, stderr=subprocess.STDOUT).returncode
        shutil.copy2(log, self.results / f"assay-{label}.log.txt")
        if rc != 0:
            raise SystemExit(f"REFUSE: assay {label} rc={rc}")
        doc = json.loads((self.results / "assays" / f"{label}.json").read_text())
        if doc["metrics"]["adapter_sha256"] != sha_file(adapter / "adapter_model.safetensors"):
            raise SystemExit(f"REFUSE: assay {label} scored a different adapter")
        return doc["metrics"]

    def assay_profile(self, vlt):
        """The assay's Profile from the spec. Only `temperature` (any mode) and
        `num_generations` (SFT and eval, which never sample) may be absent or None; they then
        take the Profile default. Table controls have no `temperature` (s[k] raised
        KeyError), and SFT arms and promoted controls (runner.control_for) set it to None,
        which Profile.validate() refuses with a TypeError inside predict_train_subset, so
        every such arm crashed at its first assay, after training. The assay is greedy, so
        neither key changes a prediction. Any other key absent or None refuses: e.g. a None
        response_format would silently assay under the wrong prompt."""
        s = self.spec
        optional = {"temperature"} | ({"num_generations"} if s.get("mode") in (
            "sft", "egsft", "eval") else set())
        missing = [k for k in PROFILE_KEYS if s.get(k) is None and k not in optional]
        if missing:
            raise SystemExit(f"REFUSE: {s.get('id')} has no {missing} for its assay profile")
        profile = vlt.Profile(**{k: s[k] for k in PROFILE_KEYS if s.get(k) is not None},
                              dtype=s.get("dtype", "float32"))
        profile.validate()
        return profile

    def kept_endpoint(self, stop: int):
        """(receipt, adapter) of kept/step<stop> when its endpoint receipt is exactly this stop
        and no endpoint for it was recorded in this run, else None (then the stop trains)."""
        if any(e.get("label") == f"step{stop}" for e in self.spec.get("prior_endpoints") or []):
            return None
        kept = self.work / "kept" / f"step{int(stop)}"
        path = kept / "endpoint_receipt.json"
        if not path.is_file() or not (kept / "final_adapter").is_dir():
            return None
        receipt = json.loads(path.read_text())
        if int(receipt.get("global_step", -1)) != int(stop):
            return None
        want = (receipt.get("final_adapter") or {}).get("weights_sha256")
        got = sha_file(kept / "final_adapter" / "adapter_model.safetensors")
        if want != got:  # a partial copy must never be scored as this stop
            raise SystemExit(f"REFUSE: kept step{stop} adapter sha {got[:12]} != receipt "
                             f"{str(want)[:12]}")
        return receipt, kept / "final_adapter"

    def assay(self, adapter: Path, label: str, manifest_name: str) -> dict:
        sys.path.insert(0, str(self.payload / "code" / "src"))
        os.environ.update({k: v for k, v in self.env.items() if k.isupper()})
        from vision_rl import vlm_local_training as vlt

        s = self.spec
        profile = self.assay_profile(vlt)
        labels = vlt.load_spec(self.spec_path)["labels"]
        manifest = self.payload / "data" / "manifests" / manifest_name
        rows = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
        for r in rows:
            r["image"] = str((manifest.parent / r["image"]).resolve())
        stage = self.work / "assay" / label
        if stage.exists():
            shutil.rmtree(stage)
        predictions, terminated, started = {}, 0, time.time()
        adapter_sha = sha_file(adapter / "adapter_model.safetensors")
        for n in range(0, len(rows), CHUNK):
            chunk = rows[n:n + CHUNK]
            cm = stage / "chunks" / f"chunk-{n // CHUNK:02d}.jsonl"
            cm.parent.mkdir(parents=True, exist_ok=True)
            cm.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in chunk))
            out = stage / f"greedy-{n // CHUNK:02d}"
            extra = {}
            if s.get("device", "cpu") == "cuda":
                extra["threads"] = int(s.get("threads", 4))
                transfer = self.precision_transfer(adapter)
                if transfer:
                    extra["precision_transfer"] = transfer
            record = vlt.predict_train_subset(cm, data_spec=self.spec_path, output=out,
                                              model_directory=self.model,
                                              device=s.get("device", "cpu"), profile=profile,
                                              init_adapter=adapter, **extra)
            if record.get("status") != "completed":
                raise SystemExit(f"REFUSE: assay chunk {n // CHUNK} did not complete")
            if (record.get("initial_adapter") or {}).get("weights_sha256") != adapter_sha:
                raise SystemExit("REFUSE: assay generated under a different adapter")
            found = json.loads((out / "predictions.json").read_text())
            for r in chunk:
                c = found.get(r["id"])
                if c is None:
                    raise SystemExit(f"REFUSE: no prediction for {r['id']}")
                ended = bool(c.get("terminated"))
                terminated += ended
                # Legacy formats: _format_kwargs is {} and parse_answer is parse_label.
                predictions[r["id"]] = (vlt.parse_answer(c.get("response", ""), labels,
                                                         allow_reasoning=True,
                                                         **vlt._format_kwargs(profile))
                                        if ended else None)
        if len(predictions) != len(rows):
            raise SystemExit("REFUSE: assay row count differs from its manifest")
        gold = {r["id"]: r["label"] for r in rows}
        metrics = score(predictions, gold)
        metrics.update(adapter_sha256=adapter_sha, manifest=manifest_name,
                       manifest_sha256=sha_file(manifest), eos_fraction=terminated / len(rows),
                       assay_seconds=round(time.time() - started, 1), label=label)
        write_json(self.results / "assays" / f"{label}.json",
                   {"metrics": metrics, "predictions": predictions})
        return metrics


def score(pred: dict, gold: dict) -> dict:
    n = len(gold)
    correct = sum(pred[i] == gold[i] for i in gold)
    by = collections.defaultdict(list)
    for i, g in gold.items():
        by[g].append(pred[i] == g)
    return {"n": n, "accuracy": correct / n,
            "valid_fraction": sum(pred[i] is not None for i in gold) / n,
            "equal_class_accuracy": sum(sum(v) / len(v) for v in by.values()) / len(by),
            "classes": len(by)}


def endpoint_assay(arm: Arm, label: str, holdout: str):
    """dev800 assay of the stage's adapter; `assay: false` (E0 and timing arms) keeps the
    adapter and records no metrics."""
    if arm.spec.get("assay", True) is False:
        return None
    return arm.assay_isolated(arm.out / "final_adapter", label, holdout)


def record_scanned(arm: Arm, receipt: dict, label: str) -> None:
    """results/scanned/<label>.json for an RL checkpoint (the campaign's controls and its
    rows-read check): every row the arm scored (Arm.scanned_ids), else its committed ids."""
    if arm.spec["mode"] in ("grpo", "rft"):
        doc = arm.scanned_ids(receipt)
        ids = (doc["scanned_prompt_ids"] if doc is not None
               else list(receipt.get("committed_prompt_ids") or []))
        write_json(arm.results / "scanned" / f"{label}.json",
                   {"label": label, "global_step": receipt.get("global_step"), "ids": ids,
                    "source": doc["source"] if doc else "committed_prompt_ids"})


def adapter_max_abs_diff(a: Path, b: Path) -> float:
    from safetensors.torch import load_file

    x, y = load_file(str(a / "adapter_model.safetensors")), load_file(
        str(b / "adapter_model.safetensors"))
    if set(x) != set(y):
        raise SystemExit("REFUSE: E0 adapters have different tensors")
    return max(float((x[k].double() - y[k].double()).abs().max()) for k in x)


def run_e0(arm: Arm, manifest: Path, summary: dict) -> dict:
    """E0 resume check (root correction C19): N updates uninterrupted, twice (A, A2: the
    device's own run-to-run floor), and N/2 + N/2 with a checkpoint resume (B), same seed.
    Passes when |B - A| <= 2 |A2 - A| + 1e-7 elementwise-max over the LoRA tensors (on a
    deterministic device all three are bit-identical). No assay."""
    n = int(arm.spec.get("e0_updates", 4))
    if n < 2 or n % 2:
        raise SystemExit("REFUSE: e0_updates must be an even number >= 2")
    finals = {}
    for tag, stops in (("A", [n]), ("A2", [n]), ("B", [n // 2, n])):
        spec = dict(arm.spec, kind="train", stops=stops, max_steps_total=n, save_steps=n // 2,
                    assay=False, work_id=f"{arm.spec['id']}-{tag}")
        sub = Arm(arm.root, spec, arm.results / tag)
        sub.results.mkdir(parents=True, exist_ok=True)
        sub.work.mkdir(parents=True, exist_ok=True)
        previous = None
        for stop in stops:
            resume = sub.out / f"checkpoint-{previous}" if previous else None
            sub.run_stage(f"train-{stop}", sub.trainer_args(manifest, n, stop, resume))
            previous = stop
        sub.copy_small()
        arm.log += [dict(e, trajectory=tag) for e in sub.log]
        finals[tag] = sub.out / "final_adapter"
    floor = adapter_max_abs_diff(finals["A"], finals["A2"])
    resumed = adapter_max_abs_diff(finals["A"], finals["B"])
    sha = {t: sha_file(p / "adapter_model.safetensors") for t, p in finals.items()}
    summary["e0"] = {"updates": n, "adapter_sha256": sha,
                     "max_abs_diff_A_A2": floor, "max_abs_diff_A_B": resumed,
                     "passed": resumed <= 2 * floor + 1e-7,
                     "device": arm.spec.get("device", "cpu"), "dtype": arm.spec.get("dtype")}
    write_json(arm.results / "stages.json", arm.log)
    write_json(arm.results / "summary.json", summary)
    if not summary["e0"]["passed"]:
        raise SystemExit(f"REFUSE: E0 resume mismatch {resumed:.3g} > floor {floor:.3g}")
    return summary


def write_summary(arm: Arm, summary: dict) -> None:
    """summary.json plus peak resident memory so far (the source for runner.py's --mem table):
    self = this process, which also runs the in-process assay; children = the largest
    trainer subprocess that has exited. Linux ru_maxrss is in KiB."""
    import resource

    def gib(kib):
        return round(kib / 1048576, 3)

    summary["resources"] = {
        "self_maxrss_gib": gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "children_maxrss_gib": gib(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)}
    write_json(arm.results / "summary.json", summary)


def time_endpoints(spec: dict) -> list:
    """[(label, active seconds)] to stop, keep and assay at after the step stops.

    `time_budget_seconds` (one value; the plan's matched control) keeps its label "time".
    `time_stops` (a strictly increasing list; a per-checkpoint time-matched control, D37)
    gives "time1", "time2", ... Both at once refuse. Ported from the GPU lane with its fix
    (6205487): the check read `not 0 < a < b`, which refused EVERY list (the first pair
    compares against 0.0)."""
    one, many = spec.get("time_budget_seconds"), spec.get("time_stops")
    if one and many:
        raise SystemExit("REFUSE: pass time_budget_seconds or time_stops, not both")
    if one:
        return [("time", float(one))]
    if many is not None and not many:
        raise SystemExit("REFUSE: time_stops is empty")
    stops = [float(x) for x in many or []]
    if any(not 0 <= a < b for a, b in zip([0.0] + stops, stops)):
        raise SystemExit("REFUSE: time_stops must be positive and strictly increasing")
    return [(f"time{i + 1}", x) for i, x in enumerate(stops)]


# A trainer feature an arm declares and the extra flag that turns it on travel together
# (review of PR #36): a row that declares dynamic_sampling without the flag would run plain
# GRPO on a host gated for it, and a row with the flag but not the feature would skip the
# receipt gate. Checked on the host before any stage.
FEATURE_EXTRA_FLAGS = {"dynamic_sampling": "--dynamic-sampling-max-draws"}


def check_feature_flags(spec: dict) -> None:
    flags = {str(x) for x in spec.get("extra_trainer_args") or []}
    features = set(spec.get("features") or [])
    for feature, flag in FEATURE_EXTRA_FLAGS.items():
        if (feature in features) != (flag in flags):
            raise SystemExit(f"REFUSE: feature {feature} and extra {flag} must appear "
                             f"together (features {sorted(features)})")


def run(arm: Arm, final_check: bool) -> dict:
    s = arm.spec
    check_feature_flags(s)
    arm.results.mkdir(parents=True, exist_ok=True)
    arm.work.mkdir(parents=True, exist_ok=True)
    write_json(arm.results / "arm.json", s)
    if s["mode"] != "eval":
        refuse_spec(s)
    holdout = "h_gate_final.jsonl" if final_check else "dev800.jsonl"
    summary = {"arm": s["id"], "host": os.uname().nodename, "root": str(arm.root),
               "receipt_sha256": sha_file(arm.root / "RECEIPT.json"),
               # an in-place continuation (runner.py continue) carries its earlier endpoints
               "endpoints": list(s.get("prior_endpoints") or [])}
    if s["mode"] in ("e0", "p0"):  # GPU-lane GATE 0 probes (gpu_probes.py)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import gpu_probes

        fn = gpu_probes.run_e0 if s["mode"] == "e0" else gpu_probes.run_p0
        summary[s["mode"]] = fn(arm, sha_file) if s["mode"] == "p0" else {
            k: v for k, v in fn(arm, sha_file).items()
            if k not in ("reward_groups", "dynamic_sampling")}
        write_json(arm.results / "stages.json", arm.log)
        write_json(arm.results / "summary.json", summary)
        return summary
    if s["mode"] == "eval":
        adapter = arm.init_adapter()
        summary["endpoints"].append({"label": "init", "steps": 0,
                                     "metrics": arm.assay(adapter, "init", holdout)})
        write_summary(arm, summary)
        return summary
    manifest = arm.manifest()
    kind = s.get("kind", "train")
    if kind == "e0":
        return run_e0(arm, manifest, summary)
    if kind == "bench":
        load_start = os.getloadavg()[0]
        arm.run_stage("bench", arm.trainer_args(manifest, s["bench_steps"], None, None))
        audit = (arm.out / "optimizer_audit.jsonl").read_text().splitlines()
        rows = [json.loads(ln) for ln in audit if ln.strip()]
        cum = [r["active_seconds"] for r in rows]
        per = [b - a for a, b in zip([0.0] + cum, cum)]
        steady = sorted(per[1:])
        summary["bench"] = {"per_update_seconds": [round(x, 2) for x in per],
                            "median_after_first": steady[len(steady) // 2] if steady else None,
                            "threads": s.get("threads"), "nproc": os.cpu_count(),
                            "load1_start": load_start, "load1_end": os.getloadavg()[0],
                            "stage_wall_seconds": arm.log[-1]["wall_seconds"]}
        arm.copy_small()
        write_summary(arm, summary)
        return summary
    total = int(s["max_steps_total"])
    previous = s.get("resume_stop")  # promotion: continue this host's own boundary
    for stop in s["stops"]:
        kept = arm.kept_endpoint(stop) if s.get("reuse_kept") else None
        if kept is not None:
            # runner.py continue: this stop trained and its adapter was kept, but the run
            # died before its endpoint was recorded (e.g. in the assay): re-assay the kept
            # adapter instead of training again (the trainer refuses a non-empty output)
            receipt, adapter = kept
            summary["endpoints"].append({
                "label": f"step{stop}", "steps": int(stop),
                "receipt_status": receipt["status"], "reassayed_kept": True,
                "active_seconds": receipt["committed_active_seconds"],
                "metrics": arm.assay(adapter, f"step{stop}", holdout)})
            arm.copy_small()
            write_summary(arm, summary)
            previous = stop
            continue
        # RFT continues its own output dir (round-level resume), never a checkpoint.
        resume = (arm.out / f"checkpoint-{previous}" if previous and s["mode"] != "rft"
                  else None)
        if resume is not None and not resume.is_dir():
            raise SystemExit(f"REFUSE: resume checkpoint {resume} is missing on this host")
        stage_args = arm.trainer_args(manifest, total, stop, resume)
        # only RFT needs the stop (to reuse a completed boundary endpoint); every other mode
        # keeps run_stage's two-argument call
        receipt = (arm.run_stage(f"train-{stop}", stage_args, stop) if s["mode"] == "rft"
                   else arm.run_stage(f"train-{stop}", stage_args))
        arm.keep_adapter(f"step{stop}")
        summary["endpoints"].append({
            "label": f"step{stop}", "steps": receipt.get("global_step", stop),
            "receipt_status": receipt["status"],
            "active_seconds": receipt["committed_active_seconds"],
            "metrics": endpoint_assay(arm, f"step{stop}", holdout)})
        record_scanned(arm, receipt, f"step{stop}")
        arm.copy_small()
        write_summary(arm, summary)
        previous = stop
        if receipt["status"] == "total_budget_exhausted":
            break
    exhausted = bool(summary["endpoints"]) and (
        summary["endpoints"][-1].get("receipt_status") == "total_budget_exhausted")
    # time-matched endpoints (port of the GPU lane's 6205487): `time_budget_seconds` gives
    # one ("time"), `time_stops` one per RL checkpoint ("time1", "time2", ...), each resuming
    # the previous stop's boundary checkpoint on the same SFT trajectory
    wanted = time_endpoints(s)
    resume = arm.out / f"checkpoint-{previous}" if previous else None
    for label, seconds in wanted:
        if exhausted:
            break
        if resume is not None and not resume.is_dir():
            raise SystemExit(f"REFUSE: resume checkpoint {resume} is missing on this host")
        receipt = arm.run_stage(f"train-{label}", arm.trainer_args(
            manifest, total, None, resume, stop_seconds=seconds))
        arm.keep_adapter(label)
        summary["endpoints"].append({
            "label": label, "steps": receipt.get("global_step"),
            "receipt_status": receipt["status"], "target_active_seconds": seconds,
            # matched only if the trainer stopped on the seconds, not on the update budget
            "time_matched": receipt["status"] != "total_budget_exhausted",
            "active_seconds": receipt["committed_active_seconds"],
            "metrics": endpoint_assay(arm, label, holdout)})
        arm.copy_small()
        write_summary(arm, summary)
        resume = arm.out / f"checkpoint-{receipt['global_step']}"
        exhausted = receipt["status"] == "total_budget_exhausted"
    matched = {e["label"] for e in summary["endpoints"] if e.get("time_matched", True)}
    missing = [label for label, _ in wanted if label not in matched]
    if missing and s.get("time_stops"):
        # a per-checkpoint control (D37) that stopped on its update cap before an RL
        # checkpoint's seconds is not time-matched there: fail loudly (summary kept). A plan
        # control's single "time" endpoint stays, flagged time_matched false, beside its valid
        # examples-matched endpoint; report never compares a flagged endpoint.
        write_summary(arm, summary)
        raise SystemExit(f"REFUSE: time-matched endpoints {missing} not reached (update "
                         f"budget {total} exhausted first)")
    arm.copy_small()
    write_json(arm.results / "stages.json", arm.log)
    write_summary(arm, summary)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--arm-b64")
    src.add_argument("--arm-json", type=Path, help="arm spec file (runner.py syncs it)")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--final-check", action="store_true")
    # Internal (assay_isolated): assay one saved adapter in this process and exit.
    p.add_argument("--assay-adapter", type=Path)
    p.add_argument("--assay-label")
    p.add_argument("--assay-manifest", choices=("dev800.jsonl", "h_gate_final.jsonl"))
    a = p.parse_args(argv)
    if not (a.root / "RECEIPT.json").is_file():
        raise SystemExit(f"REFUSE: {a.root} has no bootstrap RECEIPT.json")
    spec = json.loads(a.arm_json.read_text() if a.arm_json else base64.b64decode(a.arm_b64))
    if a.assay_adapter is not None:
        if not (a.assay_label and a.assay_manifest):
            raise SystemExit("REFUSE: --assay-adapter needs --assay-label and --assay-manifest")
        arm = Arm(a.root, spec, a.results)
        metrics = arm.assay(a.assay_adapter, a.assay_label, a.assay_manifest)
        print(json.dumps({"arm": spec["id"], "label": a.assay_label,
                          "accuracy": metrics["accuracy"]}))
        return 0
    summary = run(Arm(a.root, spec, a.results), a.final_check)
    print(json.dumps({"arm": spec["id"], "endpoints": [
        {k: e[k] for k in ("label", "steps")} | {"accuracy": (e["metrics"] or {}).get("accuracy")}
        for e in summary.get("endpoints", [])], "bench": summary.get("bench")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
