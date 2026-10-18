"""Score adapter I on every all_train row: p(gold) by teacher forcing, plus greedy correctness.

  ROOT/venv/bin/python -I score_worker.py --root ROOT --state STATE --threads N [--max-shards K]

ROOT is a host root holding venv/, model/ and payload/ (the exploratory-sweep payload 9d9a09f,
verified file by file by host_verify.py into ROOT/RECEIPT.json): either this campaign's own
/var/tmp/r226lin-pools (setup_host.sh) or a sweep host's r226lin-rlesweep root, used READ-ONLY
(nothing is written under ROOT except this worker's own ROOT/../r226lin-pools-work scratch).

The pool is the payload's data/manifests/all_train.jsonl: native TRAIN minus dev800 (h_select +
dev600), minus h_gate_final, minus near-duplicate partners of those holdouts (design.py on
exploratory-sweep-v2.noindex). Its sha256 is checked against the payload's splits.json and the
pinned value below, and I's weights against the pinned sha256, before anything runs.

Work is split into 32-row shards (rows in all_train order, which is sorted by id). Workers on
several hosts share STATE (on the NFS home): a shard is claimed by an atomic mkdir of
STATE/claims/shard-NNN and finished by an atomic rename of STATE/results/shard-NNN.json. A claim
older than STALE_SECONDS with no result is taken over (the result is deterministic, so a rare
duplicate is harmless: the reducer refuses only if two results for one id DISAGREE).

Per row, both reuse the trainer's own validated code paths unchanged:
  * greedy: vision_rl.vlm_local_training.predict_train_subset on the shard (<= 32 rows, its
    limit), scored exactly as scripts/exploratory_sweep/arm_exec.py Arm.assay scores dev800:
    the answer is parse_label(response) only if the completion ended at EOS, else None (invalid);
    correct = answer == gold label.
  * p_gold: vision_rl.vlm_local_training.gold_probability, P(exact gold tokens + EOS) under I at
    the sampler's transforms (temperature 1.0, image-token suppression, the EOS set of
    _task_group_generation_config), as the trainer's screened scan and p0_screen.py compute it.
Profile = the dev800 eval profile (arm.json of eval-I): answer_only, min/max pixels 50176/200704,
max_completion_length 256, temperature 1.0, float32, CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

I_WEIGHTS_SHA256 = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
ALL_TRAIN_SHA256 = "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e"
ALL_TRAIN_ROWS = 2329
SHARD = 32
STALE_SECONDS = 2 * 3600
PROFILE = {"response_format": "answer_only", "min_pixels": 50176, "max_pixels": 200704,
           "max_completion_length": 256, "num_generations": 4, "temperature": 1.0,
           "dtype": "float32"}
MODEL_DIR_NAME = "Qwen2-VL-2B-Instruct-895c3a49"


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_rows(payload: Path) -> list[dict]:
    manifest = payload / "data" / "manifests" / "all_train.jsonl"
    digest = sha_file(manifest)
    splits = json.loads((payload / "data" / "splits.json").read_text())
    if digest != ALL_TRAIN_SHA256 or splits["manifests_sha256"]["manifests/all_train.jsonl"] != digest:
        raise SystemExit(f"REFUSE: all_train.jsonl sha256 {digest} is not the pinned pool")
    rows = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
    if len(rows) != ALL_TRAIN_ROWS or len({r["id"] for r in rows}) != ALL_TRAIN_ROWS:
        raise SystemExit("REFUSE: all_train must hold 2,329 unique rows")
    for r in rows:
        if r["split"] != "train" or r["image"] != f"images/train/{r['id']}.jpg":
            raise SystemExit(f"REFUSE: not a native TRAIN row: {r}")
        r["image"] = str((manifest.parent / r["image"]).resolve())
    return rows


def claim(state: Path, shard: int) -> bool:
    result = state / "results" / f"shard-{shard:03d}.json"
    if result.exists():
        return False
    path = state / "claims" / f"shard-{shard:03d}"
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return False
        if age < STALE_SECONDS or result.exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            return False
    (path / "owner").write_text(f"{socket.gethostname()} {os.getpid()} {time.time():.0f}\n")
    return True


class Scorer:
    def __init__(self, root: Path, threads: int):
        self.root, self.payload = root, root / "payload"
        self.model = root / "model" / MODEL_DIR_NAME
        self.adapter = self.payload / "adapters" / "I" / "final_adapter"
        self.spec_path = self.payload / "data" / "data_spec.json"
        if sha_file(self.adapter / "adapter_model.safetensors") != I_WEIGHTS_SHA256:
            raise SystemExit("REFUSE: adapter I weights differ from the pinned sha256")
        receipt = json.loads((root / "RECEIPT.json").read_text())
        self.receipt_sha256 = sha_file(root / "RECEIPT.json")
        self.payload_sha256 = receipt["payload"]["payload_sha256"]
        if self.payload_sha256 != hashlib.sha256((self.payload / "PAYLOAD.json").read_bytes()).hexdigest():
            raise SystemExit("REFUSE: payload changed since its receipt")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          VLM_TORCH_THREADS=str(threads))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
        sys.path.insert(0, str(self.payload / "code" / "src"))
        from vision_rl import vlm_local_training as vlt
        import torch
        import transformers
        import peft
        import trl.trainer.sft_trainer as sft

        torch.set_num_threads(threads)
        self.vlt, self.torch = vlt, torch
        self.profile = vlt.Profile(**PROFILE)
        self.profile.validate()
        self.spec = vlt.load_spec(self.spec_path)
        self.labels = self.spec["labels"]
        vlt.preflight("cpu")
        vlt._verified_model_directory(self.model, vlt.DEFAULT_MODEL)
        self.processor = vlt._processor(self.model, self.profile)
        policy = transformers.AutoModelForImageTextToText.from_pretrained(
            str(self.model), trust_remote_code=False, local_files_only=True,
            use_safetensors=True, dtype=torch.float32, attn_implementation="sdpa")
        self.policy = peft.PeftModel.from_pretrained(policy, str(self.adapter), is_trainable=False)
        self.policy.requires_grad_(False).to("cpu").eval()
        config = vlt._task_group_generation_config(self.policy, self.processor, self.profile)
        self.eos, self.suppress = sorted(config.eos_token_id), list(config.suppress_tokens or [])
        self.collator = sft.DataCollatorForVisionLanguageModeling(
            self.processor, completion_only_loss=True, max_length=None)
        self.work = root.parent / "r226lin-pools-work"

    def greedy(self, rows: list[dict], shard: int) -> dict:
        vlt = self.vlt
        stage = self.work / f"shard-{shard:03d}-{os.getpid()}"
        shutil.rmtree(stage, ignore_errors=True)
        cm = stage / "chunk.jsonl"
        cm.parent.mkdir(parents=True, exist_ok=True)
        cm.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        out = stage / "greedy"
        record = vlt.predict_train_subset(cm, data_spec=self.spec_path, output=out,
                                          model_directory=self.model, device="cpu",
                                          profile=self.profile, init_adapter=self.adapter)
        if record.get("status") != "completed":
            raise SystemExit(f"REFUSE: greedy shard {shard} did not complete")
        if (record.get("initial_adapter") or {}).get("weights_sha256") != I_WEIGHTS_SHA256:
            raise SystemExit("REFUSE: greedy ran under a different adapter")
        found = json.loads((out / "predictions.json").read_text())
        shutil.rmtree(stage, ignore_errors=True)
        return found

    def p_gold(self, row: dict) -> float:
        with self.torch.no_grad():
            return float(self.vlt.gold_probability(
                self.policy, self.processor, row, self.spec, self.profile, "cpu",
                eos_token_ids=self.eos, suppress_tokens=self.suppress, collator=self.collator))

    def shard(self, rows: list[dict], shard: int) -> dict:
        started = time.time()
        found = self.greedy(rows, shard)
        greedy_seconds = time.time() - started
        out = []
        for r in rows:
            c = found.get(r["id"])
            if c is None or c.get("error"):
                raise SystemExit(f"REFUSE: no greedy prediction for {r['id']}")
            ended = bool(c.get("terminated"))
            answer = (self.vlt.parse_label(c.get("response", ""), self.labels, allow_reasoning=True)
                      if ended else None)
            t0 = time.time()
            p = self.p_gold(r)
            if not 0.0 <= p <= 1.0:
                raise SystemExit(f"REFUSE: p_gold out of range for {r['id']}: {p}")
            out.append({"id": r["id"], "label": r["label"],
                        "image_sha256": sha_file(Path(r["image"])),
                        "p_gold": p, "p_seconds": round(time.time() - t0, 3),
                        "greedy_response": c.get("response", ""), "terminated": ended,
                        "greedy_answer": answer, "greedy_correct": answer == r["label"],
                        "greedy_completion_tokens": c.get("completion_tokens")})
        return {"shard": shard, "ids": [r["id"] for r in rows], "rows": out,
                "host": socket.gethostname(), "pid": os.getpid(),
                "root": str(self.root), "receipt_sha256": self.receipt_sha256,
                "payload_sha256": self.payload_sha256, "adapter_sha256": I_WEIGHTS_SHA256,
                "all_train_sha256": ALL_TRAIN_SHA256, "profile": PROFILE,
                "greedy_seconds": round(greedy_seconds, 1),
                "total_seconds": round(time.time() - started, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--max-shards", type=int, default=0, help="stop after K shards (0 = no limit)")
    a = p.parse_args(argv)
    rows = load_rows(a.root / "payload")
    shards = [rows[i:i + SHARD] for i in range(0, len(rows), SHARD)]
    a.state.mkdir(parents=True, exist_ok=True)
    scorer = None
    done = 0
    for k, chunk in enumerate(shards):
        if a.max_shards and done >= a.max_shards:
            break
        if not claim(a.state, k):
            continue
        if scorer is None:
            scorer = Scorer(a.root, a.threads)
        result = scorer.shard(chunk, k)
        atomic_json(a.state / "results" / f"shard-{k:03d}.json", result)
        done += 1
        print(f"shard {k:03d} done on {socket.gethostname()} in {result['total_seconds']} s "
              f"(greedy {result['greedy_seconds']} s); "
              f"correct {sum(r['greedy_correct'] for r in result['rows'])}/{len(chunk)}",
              flush=True)
    missing = [k for k in range(len(shards))
               if not (a.state / "results" / f"shard-{k:03d}.json").exists()]
    print(f"worker exits: did {done} shards; {len(missing)} of {len(shards)} shards lack results",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
