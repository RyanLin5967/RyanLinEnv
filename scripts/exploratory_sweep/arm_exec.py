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
    def payload_arg(self, value) -> str:
        """`@payload/<rel>` names a file shipped in this host's payload (e.g. the TRAIN
        label hierarchy); it may not escape the payload directory."""
        text = str(value)
        if not text.startswith("@payload/"):
            return text
        path = (self.payload / text.removeprefix("@payload/")).resolve()
        if not path.is_relative_to(self.payload.resolve()) or not path.is_file():
            raise SystemExit(f"REFUSE: {text} is not a file in the payload")
        return str(path)

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
        return args + [self.payload_arg(x) for x in s.get("extra_trainer_args", [])]

    def trainer_args(self, manifest: Path, max_steps: int, stop: int | None,
                     resume: Path | None, stop_seconds: float | None = None) -> list[str]:
        s = self.spec
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
        args += [self.payload_arg(x) for x in s.get("extra_trainer_args", [])]
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
        else:
            return None
        doc = {"schema": "arm_exec-scanned-ids-v1", "global_step": receipt["global_step"],
               "scanned_prompt_ids": ids, "source": source}
        write_json(self.out / "scanned_ids.json", doc)
        return doc

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
            self.log.append({"stage": label, "rc": 0, "reused_completed_endpoint": True,
                             "argv": args})
            receipt = self.rft_receipt()
            self.scanned_ids(receipt)
            self.log[-1].update(receipt_status=receipt["status"],
                                global_step=receipt["global_step"],
                                adapter_sha256=receipt["final_adapter"]["weights_sha256"])
            return receipt
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
        receipt = (self.rft_receipt() if self.spec["mode"] == "rft" else
                   json.loads((self.out / "endpoint_receipt.json").read_text()))
        adapter = self.out / receipt["final_adapter"]["path"]
        weights = receipt["final_adapter"]["weights_sha256"]
        if sha_file(adapter / "adapter_model.safetensors") != weights:
            raise SystemExit(f"REFUSE: stage {label} adapter differs from its receipt")
        self.scanned_ids(receipt)
        entry.update(receipt_status=receipt["status"], global_step=receipt.get("global_step"),
                     adapter_sha256=receipt["final_adapter"]["weights_sha256"])
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
        return vlt.Profile(**{k: s[k] for k in PROFILE_KEYS if s.get(k) is not None},
                           dtype=s.get("dtype", "float32"))

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
        # RFT continues its own output dir (round-level resume), never a checkpoint.
        resume = (arm.out / f"checkpoint-{previous}" if previous and s["mode"] != "rft"
                  else None)
        if resume is not None and not resume.is_dir():
            raise SystemExit(f"REFUSE: resume checkpoint {resume} is missing on this host")
        receipt = arm.run_stage(f"train-{stop}", arm.trainer_args(manifest, total, stop, resume),
                                stop)
        arm.keep_adapter(f"step{stop}")
        summary["endpoints"].append({
            "label": f"step{stop}", "steps": receipt.get("global_step", stop),
            "receipt_status": receipt["status"],
            "active_seconds": receipt["committed_active_seconds"],
            "metrics": arm.assay_isolated(arm.out / "final_adapter", f"step{stop}", holdout)})
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
            "metrics": arm.assay_isolated(arm.out / "final_adapter", "time", holdout)})
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
        {k: e[k] for k in ("label", "steps")} | {"accuracy": e["metrics"]["accuracy"]}
        for e in summary.get("endpoints", [])], "bench": summary.get("bench")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
