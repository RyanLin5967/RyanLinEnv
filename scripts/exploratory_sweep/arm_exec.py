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


def _int_at_least(n: int):
    def check(v: str):
        if not v.lstrip("-").isdigit() or int(v) < n:
            raise SystemExit(f"REFUSE: expected an integer >= {n}, got {v!r}")
    return check


# Table-supplied trainer flags, an ALLOWLIST (exact names; each takes one value checked by
# its validator). argparse keeps the LAST value of a repeated flag and accepts unique
# abbreviations and `--flag=value`, so an extra appended after arm_exec's own flags could
# silently override --dtype, --learning-rate, --train, ... while the spec-field guards
# pass (PR #38 review). Extend this map only with a validator; never add a flag arm_exec
# sets itself. Blind spot: a validator checks the value's form, not the trainer's semantics.
EXTRA_FLAGS = {
    "--dynamic-sampling-max-draws": _int_at_least(8),  # CORRECTIONS C5: >= 8 draws
}
NEVER_EXTRA = {"--train", "--validation"}


def checked_extras(extras, base_args) -> list[str]:
    """`extras` as trainer argv, or refuse: only EXTRA_FLAGS, each once, as `--flag value`,
    never a flag arm_exec already passes, never --train/--validation."""
    own = {a for a in base_args if isinstance(a, str) and a.startswith("--")}
    out, seen, i = [], set(), 0
    extras = [str(x) for x in extras]
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
        if i + 1 >= len(extras):
            raise SystemExit(f"REFUSE: extra {flag} has no value")
        EXTRA_FLAGS[flag](extras[i + 1])
        out += [flag, extras[i + 1]]
        seen.add(flag)
        i += 2
    return out


def save_steps_for(spec: dict, max_steps: int) -> int:
    """--save-steps for every stage of this arm's trajectory.

    save_steps is part of the trainer's resume identity (settings), so it must be the same in
    every stage of one trajectory, across promotions too. A trajectory's first arm uses the
    gcd of its declared stops (and resume_stop); a promoted arm carries the value its
    trajectory started with in `save_steps` (runner.promoted_arms sets it), because the gcd
    rule alone drifts across rungs (150 -> 600 -> 1500 gives 150, 150, then 300). A stage
    boundary always saves its own checkpoint regardless. (It was `stop`, which changed between
    stages and made every multi-stop arm refuse to resume: "resume must preserve exact
    model/data/profile/settings"; ported from GPU lane acdab57.)
    """
    if spec.get("save_steps"):
        return int(spec["save_steps"])
    marks = [int(x) for x in spec.get("stops") or []] + (
        [int(spec["resume_stop"])] if spec.get("resume_stop") else [])
    return math.gcd(*marks) if marks else int(max_steps)


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
    def trainer_args(self, manifest: Path, max_steps: int, stop: int | None,
                     resume: Path | None, stop_seconds: float | None = None) -> list[str]:
        s = self.spec
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
        args += ["--save-steps", str(save_steps_for(s, max_steps))]
        if stop and stop < max_steps:
            args += ["--stop-after-steps", str(stop)]
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
        if s.get("max_active_seconds"):
            args += ["--max-active-seconds", repr(float(s["max_active_seconds"]))]
        if stop_seconds is not None:
            if not s.get("max_active_seconds") or stop_seconds > float(s["max_active_seconds"]):
                raise SystemExit("REFUSE: a time stop needs a larger declared max_active_seconds")
            args += ["--stop-after-active-seconds", repr(float(stop_seconds))]
        if resume is not None:
            args += ["--resume-from-checkpoint", str(resume)]
        return args + checked_extras(s.get("extra_trainer_args") or [], args)

    def run_stage(self, label: str, args: list[str]) -> dict:
        started = time.time()
        log = self.work / f"{label}.log"
        with log.open("w") as handle:
            rc = subprocess.run([str(self.python), "-s", "-P", *args], env=self.env, stdout=handle,
                                stderr=subprocess.STDOUT).returncode
        tail = log.read_text().splitlines()[-25:]
        shutil.copy2(log, self.results / f"{label}.log.txt")
        entry = {"stage": label, "rc": rc, "wall_seconds": round(time.time() - started, 1),
                 "argv": args}
        self.log.append(entry)
        if rc != 0:
            write_json(self.results / "stages.json", self.log)
            print("\n".join(tail), file=sys.stderr)
            raise SystemExit(f"REFUSE: stage {label} rc={rc}")
        receipt = json.loads((self.out / "endpoint_receipt.json").read_text())
        adapter = self.out / receipt["final_adapter"]["path"]
        weights = receipt["final_adapter"]["weights_sha256"]
        if sha_file(adapter / "adapter_model.safetensors") != weights:
            raise SystemExit(f"REFUSE: stage {label} adapter differs from its receipt")
        entry.update(receipt_status=receipt["status"], global_step=receipt.get("global_step"),
                     adapter_sha256=receipt["final_adapter"]["weights_sha256"])
        return receipt

    def init_adapter(self) -> Path:
        """W or I from the payload, or {"arm", "label"}: a kept endpoint on this host."""
        init = self.spec["init"]
        if isinstance(init, str):
            if init not in ("W", "I"):
                raise SystemExit(f"REFUSE: unknown init {init!r}")
            return self.payload / "adapters" / init / "final_adapter"
        path = self.root / "arms" / init["arm"] / "kept" / init["label"] / "final_adapter"
        receipt = json.loads((path.parent / "endpoint_receipt.json").read_text())
        weights = receipt["final_adapter"]["weights_sha256"]
        if sha_file(path / "adapter_model.safetensors") != weights:
            raise SystemExit(f"REFUSE: kept init {path} differs from its endpoint receipt")
        return path

    def keep_adapter(self, label: str) -> Path:
        # kept/<label>/{final_adapter, local_training.json, endpoint_receipt.json}: the
        # trainer binds an init adapter to the run record beside it, so a kept endpoint can
        # seed a later arm on this host (e.g. SFT(all TRAIN) -> GRPO).
        dst = self.work / "kept" / label
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.out / "final_adapter", dst / "final_adapter")
        for name in ("local_training.json", "endpoint_receipt.json"):
            shutil.copy2(self.out / name, dst / name)
        return self.out / "final_adapter"

    def copy_small(self) -> None:
        for name in ("local_training.json", "endpoint_receipt.json", "optimizer_audit.jsonl",
                     "physical_prompt_events.jsonl", "reward_groups.jsonl",
                     "screen_journal.jsonl", "egsft_weights.jsonl", "trainer_state.json"):
            src = self.out / name
            if src.is_file() and src.stat().st_size < 64 * 1024**2:
                shutil.copy2(src, self.results / name)

    # ---- assay -----------------------------------------------------------------------
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

    def assay_profile(self, vlt):
        """The greedy assay's Profile. A key the spec lacks OR sets to None takes the Profile
        default: matched controls have no `temperature` (KeyError at their first assay, found
        by the wave-2 integration, ce099a3) and every SFT-family arm sets temperature None,
        which Profile.validate() rejects (math.isfinite(None): TypeError) inside
        predict_train_subset. Greedy decoding does not sample, so temperature cannot change
        an assay. Any other present value passes exactly as before."""
        s = self.spec
        profile = vlt.Profile(**{k: s[k] for k in PROFILE_KEYS if s.get(k) is not None},
                              dtype=s.get("dtype", "float32"))
        profile.validate()
        return profile

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
            record = vlt.predict_train_subset(cm, data_spec=self.spec_path, output=out,
                                              model_directory=self.model,
                                              device=s.get("device", "cpu"), profile=profile,
                                              init_adapter=adapter)
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
                predictions[r["id"]] = (vlt.parse_label(c.get("response", ""), labels,
                                                        allow_reasoning=True) if ended else None)
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


def write_summary(arm: Arm, summary: dict) -> None:
    """summary.json plus peak resident memory so far (the source for runner.py's --mem table):
    self = this process, which also runs the in-process assay; children = the largest
    trainer subprocess that has exited. Linux ru_maxrss is in KiB."""
    import resource
    gib = lambda kib: round(kib / 1048576, 3)
    summary["resources"] = {
        "self_maxrss_gib": gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "children_maxrss_gib": gib(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)}
    write_json(arm.results / "summary.json", summary)


def run(arm: Arm, final_check: bool) -> dict:
    s = arm.spec
    arm.results.mkdir(parents=True, exist_ok=True)
    arm.work.mkdir(parents=True, exist_ok=True)
    write_json(arm.results / "arm.json", s)
    holdout = "h_gate_final.jsonl" if final_check else "dev800.jsonl"
    summary = {"arm": s["id"], "host": os.uname().nodename, "root": str(arm.root),
               "receipt_sha256": sha_file(arm.root / "RECEIPT.json"),
               # an in-place continuation (runner.py continue) carries its earlier endpoints
               "endpoints": list(s.get("prior_endpoints") or [])}
    if s["mode"] == "eval":
        adapter = arm.init_adapter()
        summary["endpoints"].append({"label": "init", "steps": 0,
                                     "metrics": arm.assay(adapter, "init", holdout)})
        write_summary(arm, summary)
        return summary
    manifest = arm.manifest()
    kind = s.get("kind", "train")
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
        resume = arm.out / f"checkpoint-{previous}" if previous else None
        if resume is not None and not resume.is_dir():
            raise SystemExit(f"REFUSE: resume checkpoint {resume} is missing on this host")
        receipt = arm.run_stage(f"train-{stop}", arm.trainer_args(manifest, total, stop, resume))
        arm.keep_adapter(f"step{stop}")
        summary["endpoints"].append({
            "label": f"step{stop}", "steps": receipt.get("global_step", stop),
            "receipt_status": receipt["status"],
            "active_seconds": receipt["committed_active_seconds"],
            "metrics": arm.assay(arm.out / "final_adapter", f"step{stop}", holdout)})
        arm.copy_small()
        write_summary(arm, summary)
        previous = stop
        if receipt["status"] == "total_budget_exhausted":
            break
    exhausted = summary["endpoints"][-1]["receipt_status"] == "total_budget_exhausted"
    if s.get("time_budget_seconds") and not exhausted:
        resume = arm.out / f"checkpoint-{previous}"
        receipt = arm.run_stage("train-time", arm.trainer_args(
            manifest, total, None, resume, stop_seconds=float(s["time_budget_seconds"])))
        arm.keep_adapter("time")
        summary["endpoints"].append({
            "label": "time", "steps": receipt.get("global_step"),
            "receipt_status": receipt["status"],
            "active_seconds": receipt["committed_active_seconds"],
            "metrics": arm.assay(arm.out / "final_adapter", "time", holdout)})
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
    a = p.parse_args(argv)
    if not (a.root / "RECEIPT.json").is_file():
        raise SystemExit(f"REFUSE: {a.root} has no bootstrap RECEIPT.json")
    spec = json.loads(a.arm_json.read_text() if a.arm_json else base64.b64decode(a.arm_b64))
    summary = run(Arm(a.root, spec, a.results), a.final_check)
    print(json.dumps({"arm": spec["id"], "endpoints": [
        {k: e[k] for k in ("label", "steps")} | {"accuracy": e["metrics"]["accuracy"]}
        for e in summary.get("endpoints", [])], "bench": summary.get("bench")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
