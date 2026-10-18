"""Score adapter I for RL campaign I: pool scoring of all_train and the P0-I G8 rollouts.

  ROOT/venv/bin/python -I score_worker.py --root ROOT --state STATE --trie TRIE --threads N

ROOT is a host root holding venv/, model/ and payload/ (an exploratory-sweep payload whose trainer,
data and adapters are pinned below; host_verify.py wrote ROOT/RECEIPT.json): this campaign's own
/var/tmp/r226lin-pools (setup_host.sh) or a sweep host's r226lin-rlesweep root used READ-ONLY.
Scratch goes to ROOT/../r226lin-pools-work only.

Units (claimed by atomic mkdir under STATE/claims, finished by atomic rename into STATE/results;
a claim older than STALE_SECONDS without a result is taken over; results are deterministic, so a
rare duplicate is harmless and the reducer refuses only if two results disagree). P0 units first:
  p0-tTT-NN  P0-I: 8 sampled completions (G8) per image at temperature TT/10 through the trainer's
             own sample_train_groups (released-TRL sampler settings, transformers.set_seed(42) at
             each 32-image unit), plus the trie-summed p(gold) at that temperature. Images:
             the first 200 rows of s_train_scan_seed42, then the first 200 rows of
             all_train_scan_seed42 not already taken (membership recorded per image).
  score-NNN  pools: all_train rows 32*NNN.. (sorted by id): I greedy (predict_train_subset, scored
             exactly like arm_exec.Arm.assay: parse_label only when the completion ended at EOS) and
             p(gold) at temperature 1.0 summed over every label-trie tokenisation.

p(gold) = sum over the trie's tokenisations t of the gold label (data/fgvc-aircraft-2013b/audit/
label-trie-qwen2vl2b.json at 552d490: every spelling within canonical length + 2, each followed by
either EOS id) of P(t + EOS) under the sampler's transforms (logits / T, image token suppressed,
renormalised), as vision_rl.vlm_local_training._batch_gold_probability computes it with
`alternatives`. This file computes it with one prompt forward and one batched continuation
forward over the prompt's KV cache (the trainer would run 2 x |paths| full forwards). It is
CHECKED against the trainer's own _batch_gold_probability (canonical alone, and with every trie
alternative) on the first eligible row of every score unit; |diff| > 1e-4 refuses.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path

I_WEIGHTS_SHA256 = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
ALL_TRAIN_SHA256 = "9bed5893ad866fa2b6a5f86c4edc98421699154f435879e9e4867825c64caa9e"
TRAINER_SHA256 = "b6e18a8b7c502ed71f3178ff3891ff0276261aa2cbdc16f9ce0ca869d116a5c0"
TRIE_SHA256 = "656dc02fbb5b121f4cd9a9d7ea10364532639b35b647f0dc9010ad5d80a37eb0"
# train400 (400) + the 477 VAL/TEST pHash near-duplicates kept in all_train (CORRECTIONS.md C9/M8):
# never scored, never pooled. build_pools.py re-derives both sets from their sources and refuses
# if this file disagrees.
EXCLUSIONS_SHA256 = "00a433eab22002764d2d91c02888623b2d5a4b37ee90ba3e1bb0eb19b482380b"
ELIGIBLE_ROWS = 1452
SCAN_SHA256 = {
    "s_train_scan_seed42.jsonl": "8f078caaa7a20cd27fe8758cc57619d8ac2457f768135cf83ba528d37d38c896",
    "all_train_scan_seed42.jsonl": "5fcb65887c482b46ce74c71bbe35af6e5d2bbae08fe29bc2edb1204d04200b26",
}
ALL_TRAIN_ROWS = 2329
SHARD = 32
P0_PER_SOURCE = 200
P0_TEMPS = (10, 13)  # tenths
P0_G = 8
P0_SEED = 42
STALE_SECONDS = 3 * 3600
VALIDATE_MAX_PATHS = 16
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


def read_manifest(payload: Path, name: str, pinned: str) -> list[dict]:
    manifest = payload / "data" / "manifests" / name
    digest = sha_file(manifest)
    if digest != pinned:
        raise SystemExit(f"REFUSE: {name} sha256 {digest} is not the pinned one")
    rows = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
    for r in rows:
        if r["split"] != "train" or r["image"] != f"images/train/{r['id']}.jpg":
            raise SystemExit(f"REFUSE: not a native TRAIN row: {r}")
        r["image"] = str((manifest.parent / r["image"]).resolve())
    return rows


def units(payload: Path, exclusions: Path) -> dict[str, dict]:
    rows = read_manifest(payload, "all_train.jsonl", ALL_TRAIN_SHA256)
    if len(rows) != ALL_TRAIN_ROWS or len({r["id"] for r in rows}) != ALL_TRAIN_ROWS:
        raise SystemExit("REFUSE: all_train must hold 2,329 unique rows")
    if sha_file(exclusions) != EXCLUSIONS_SHA256:
        raise SystemExit("REFUSE: exclusions file differs from the pinned sha256")
    ex = json.loads(exclusions.read_text())
    excluded = set(ex["train400"]) | set(ex["val_test_phash_near_duplicates"])
    eligible = [r for r in rows if r["id"] not in excluded]
    if len(excluded) != 877 or len(eligible) != ELIGIBLE_ROWS:
        raise SystemExit("REFUSE: exclusions must remove 877 rows and leave 1,452")
    s_scan = read_manifest(payload, "s_train_scan_seed42.jsonl", SCAN_SHA256["s_train_scan_seed42.jsonl"])
    a_scan = read_manifest(payload, "all_train_scan_seed42.jsonl", SCAN_SHA256["all_train_scan_seed42.jsonl"])
    s_ids = {r["id"] for r in s_scan[:P0_PER_SOURCE]}
    a_ids = {r["id"] for r in a_scan[:P0_PER_SOURCE]}
    p0 = list(s_scan[:P0_PER_SOURCE]) + [r for r in a_scan[:P0_PER_SOURCE] if r["id"] not in s_ids]
    for r in p0:
        r["p0_sources"] = [n for n, ids in (("s_train", s_ids), ("all_train", a_ids)) if r["id"] in ids]
    out = {}
    for t in P0_TEMPS:
        for k in range(0, len(p0), SHARD):
            out[f"p0-t{t}-{k // SHARD:02d}"] = {"kind": "p0", "temp10": t, "rows": p0[k:k + SHARD]}
    for k in range(0, len(eligible), SHARD):
        out[f"score-{k // SHARD:03d}"] = {"kind": "score", "rows": eligible[k:k + SHARD]}
    return out


def claim(state: Path, name: str) -> bool:
    if (state / "results" / f"{name}.json").exists():
        return False
    path = state / "claims" / name
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return False
        if age < STALE_SECONDS or (state / "results" / f"{name}.json").exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            return False
    (path / "owner").write_text(f"{socket.gethostname()} {os.getpid()} {time.time():.0f}\n")
    return True


class Scorer:
    def __init__(self, root: Path, threads: int, trie_path: Path):
        self.root, self.payload = root, root / "payload"
        self.model = root / "model" / MODEL_DIR_NAME
        self.adapter = self.payload / "adapters" / "I" / "final_adapter"
        self.spec_path = self.payload / "data" / "data_spec.json"
        self.threads = threads
        if sha_file(self.adapter / "adapter_model.safetensors") != I_WEIGHTS_SHA256:
            raise SystemExit("REFUSE: adapter I weights differ from the pinned sha256")
        if sha_file(self.payload / "code/src/vision_rl/vlm_local_training.py") != TRAINER_SHA256:
            raise SystemExit("REFUSE: trainer source differs from the pinned sha256")
        if sha_file(trie_path) != TRIE_SHA256:
            raise SystemExit("REFUSE: label trie differs from the pinned sha256")
        receipt = json.loads((root / "RECEIPT.json").read_text())
        self.receipt_sha256 = sha_file(root / "RECEIPT.json")
        self.payload_sha256 = receipt["payload"]["payload_sha256"]
        if self.payload_sha256 != sha_file(self.payload / "PAYLOAD.json"):
            raise SystemExit("REFUSE: payload changed since its receipt")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLM_TORCH_THREADS=str(threads))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
        sys.path.insert(0, str(self.payload / "code" / "src"))
        from vision_rl import vlm_local_training as vlt
        import torch
        import transformers
        import peft
        import trl.trainer.sft_trainer as sft

        torch.set_num_threads(threads)
        self.vlt, self.torch, self.transformers = vlt, torch, transformers
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
        trie = json.loads(Path(trie_path).read_text())
        if sorted(trie["eos_ids"]) != self.eos or sorted(trie["entries"]) != sorted(self.labels):
            raise SystemExit("REFUSE: trie EOS ids or labels differ from the sampler's")
        self.paths = {label: [list(p) for p in e["paths"]] for label, e in trie["entries"].items()}
        self.canonical = {label: list(e["canonical"]) for label, e in trie["entries"].items()}
        self.work = root.parent / "r226lin-pools-work"

    # ---- p(gold) summed over the trie --------------------------------------------------
    def _batch(self, row):
        vlt = self.vlt
        record = vlt.training_records([row], self.spec, "sft", self.profile.response_format)[0]
        record["image"] = vlt._load_rgb(record["image"], self.spec)
        return {k: v.to("cpu") for k, v in self.collator([record]).items()}

    def _logp(self, logits, temperature):
        logits = logits.float() / temperature
        logits[..., self.suppress] = -self.torch.inf
        return logits.log_softmax(-1)

    def trie_p(self, row, temperature: float) -> dict:
        torch, vlt = self.torch, self.vlt
        batch = self._batch(row)
        ids, labels = batch["input_ids"], batch["labels"]
        supervised = (labels[0] != -100).nonzero().flatten().tolist()
        first = supervised[0]
        completion = ids[0, first:].tolist()
        stop = next(i for i, t in enumerate(completion) if t in self.eos)
        canonical = completion[:stop + 1]
        paths = self.paths[row["label"]]
        if canonical[:-1] != self.canonical[row["label"]] or canonical[:-1] not in paths:
            raise SystemExit(f"REFUSE: {row['id']}: collator canonical is not the trie canonical")
        prompt = {k: (v[:, :first] if k in vlt._TOKEN_ALIGNED_KEYS else v)
                  for k, v in batch.items() if k not in vlt._NOT_MODEL_INPUT_KEYS}
        started = time.time()
        with torch.no_grad():
            out = self.policy(**prompt, use_cache=True, logits_to_keep=1)
            head = self._logp(out.logits[0, -1], temperature)  # predicts each path's first token
            cache = out.past_key_values
            path_logp = []
            for c in range(0, len(paths), 32):
                chunk = paths[c:c + 32]
                width = max(len(p) for p in chunk)
                inp = torch.tensor([p + [p[-1]] * (width - len(p)) for p in chunk], dtype=ids.dtype)
                past = copy.deepcopy(cache)
                past.batch_repeat_interleave(len(chunk))
                logits = self.policy(input_ids=inp, past_key_values=past, use_cache=True).logits
                logp = self._logp(logits, temperature)
                for i, p in enumerate(chunk):
                    value = float(head[p[0]])
                    for k in range(1, len(p)):
                        value += float(logp[i, k - 1, p[k]])
                    eos_lp = logp[i, len(p) - 1, self.eos]
                    path_logp.append((value, [float(x) for x in eos_lp]))
                del past, logits, logp
        total = sum(math.exp(v + e) for v, eos_lps in path_logp for e in eos_lps)
        ci = paths.index(canonical[:-1])
        canon = math.exp(path_logp[ci][0] + path_logp[ci][1][self.eos.index(canonical[-1])])
        if not (0.0 <= canon <= total <= 1.0 + 1e-6) or math.isnan(total):
            raise SystemExit(f"REFUSE: {row['id']}: invalid trie probability {canon} {total}")
        return {"p": min(total, 1.0), "p_canonical": canon, "paths": len(paths),
                "seconds": round(time.time() - started, 3), "_batch": batch, "_canonical": canonical}

    def validate(self, row, mine: dict, temperature: float) -> dict:
        """The trainer's own _batch_gold_probability, canonical-only and with every alternative."""
        vlt, torch = self.vlt, self.torch
        alts = [p + [e] for p in self.paths[row["label"]] for e in self.eos]
        with torch.no_grad():
            canon, _ = vlt._batch_gold_probability(
                self.policy, mine["_batch"], eos_token_ids=self.eos, suppress_tokens=self.suppress,
                temperature=temperature)
            full, _ = vlt._batch_gold_probability(
                self.policy, mine["_batch"], eos_token_ids=self.eos, suppress_tokens=self.suppress,
                temperature=temperature, alternatives=alts)
        diff = max(abs(canon - mine["p_canonical"]), abs(full - mine["p"]))
        if diff > 1e-4:
            raise SystemExit(f"REFUSE: {row['id']}: cached trie p disagrees with the trainer's "
                             f"_batch_gold_probability: mine {mine['p']}/{mine['p_canonical']} "
                             f"trainer {full}/{canon}")
        return {"id": row["id"], "trainer_p_canonical": canon, "trainer_p_trie": full,
                "mine_p_canonical": mine["p_canonical"], "mine_p_trie": mine["p"],
                "max_abs_diff": diff, "paths": mine["paths"]}

    # ---- units --------------------------------------------------------------------------
    def _chunk_manifest(self, rows, name):
        stage = self.work / f"{name}-{os.getpid()}"
        shutil.rmtree(stage, ignore_errors=True)
        cm = stage / "chunk.jsonl"
        cm.parent.mkdir(parents=True, exist_ok=True)
        cm.write_text("".join(json.dumps({k: r[k] for k in ("id", "image", "label", "split")},
                                         sort_keys=True) + "\n" for r in rows))
        return stage, cm

    def score_unit(self, name, rows) -> dict:
        vlt = self.vlt
        started = time.time()
        stage, cm = self._chunk_manifest(rows, name)
        record = vlt.predict_train_subset(cm, data_spec=self.spec_path, output=stage / "greedy",
                                          model_directory=self.model, device="cpu",
                                          profile=self.profile, init_adapter=self.adapter)
        if record.get("status") != "completed" or (record.get("initial_adapter") or {}).get(
                "weights_sha256") != I_WEIGHTS_SHA256:
            raise SystemExit(f"REFUSE: greedy {name} did not complete under I")
        found = json.loads((stage / "greedy" / "predictions.json").read_text())
        greedy_seconds = time.time() - started
        out, validation = [], None
        for r in rows:
            c = found.get(r["id"])
            if c is None or c.get("error"):
                raise SystemExit(f"REFUSE: no greedy prediction for {r['id']}")
            ended = bool(c.get("terminated"))
            answer = vlt.parse_label(c.get("response", ""), self.labels, allow_reasoning=True) \
                if ended else None
            mine = self.trie_p(r, 1.0)
            if validation is None and mine["paths"] <= VALIDATE_MAX_PATHS:
                validation = self.validate(r, mine, 1.0)
            out.append({"id": r["id"], "label": r["label"], "image_sha256": sha_file(Path(r["image"])),
                        "p_gold": mine["p"], "p_canonical": mine["p_canonical"],
                        "trie_paths": mine["paths"], "p_seconds": mine["seconds"],
                        "greedy_response": c.get("response", ""), "terminated": ended,
                        "greedy_answer": answer, "greedy_correct": answer == r["label"],
                        "greedy_seconds": round(float(c.get("latency_seconds", 0.0)), 3),
                        "greedy_completion_tokens": c.get("completion_tokens")})
        if validation is None:
            raise SystemExit(f"REFUSE: {name}: no row with <= {VALIDATE_MAX_PATHS} trie paths to validate")
        shutil.rmtree(stage, ignore_errors=True)
        return {"unit": name, "kind": "score", "ids": [r["id"] for r in rows], "rows": out,
                "validation": validation, "greedy_seconds": round(greedy_seconds, 1),
                "total_seconds": round(time.time() - started, 1)}

    def p0_unit(self, name, rows, temp10) -> dict:
        vlt = self.vlt
        temperature = temp10 / 10
        started = time.time()
        profile = vlt.Profile(**dict(PROFILE, num_generations=P0_G, temperature=temperature))
        stage, cm = self._chunk_manifest(rows, name)
        record = vlt.sample_train_groups(cm, output=stage / "groups", seed=P0_SEED,
                                         data_spec=self.spec_path, model_directory=self.model,
                                         device="cpu", profile=profile, init_adapter=self.adapter)
        if record.get("status") != "completed" or (record.get("initial_adapter") or {}).get(
                "weights_sha256") != I_WEIGHTS_SHA256:
            raise SystemExit(f"REFUSE: sampling {name} did not complete under I")
        samples = json.loads((stage / "groups" / "sampled_completions.json").read_text())
        sample_seconds = time.time() - started
        out = []
        for r in rows:
            group = samples.get(r["id"]) or []
            if len(group) != P0_G:
                raise SystemExit(f"REFUSE: {r['id']}: {len(group)} completions, not {P0_G}")
            comps = []
            for c in group:
                ended = bool(c.get("terminated"))
                answer = vlt.parse_label(c.get("response", ""), self.labels, allow_reasoning=True) \
                    if ended else None
                comps.append({"response": c.get("response", ""), "terminated": ended,
                              "answer": answer, "reward": int(answer == r["label"]),
                              "completion_ids": c.get("completion_ids"),
                              "seconds": round(float(c.get("latency_seconds", 0.0)), 3)})
            mine = self.trie_p(r, temperature)
            out.append({"id": r["id"], "label": r["label"], "p0_sources": r["p0_sources"],
                        "image_sha256": sha_file(Path(r["image"])), "temperature": temperature,
                        "p_gold_trie": mine["p"], "p_canonical": mine["p_canonical"],
                        "p_seconds": mine["seconds"], "completions": comps})
        shutil.rmtree(stage, ignore_errors=True)
        return {"unit": name, "kind": "p0", "temperature": temperature, "G": P0_G, "seed": P0_SEED,
                "ids": [r["id"] for r in rows], "rows": out, "sample_seconds": round(sample_seconds, 1),
                "total_seconds": round(time.time() - started, 1),
                "sampler_record": {k: record.get(k) for k in ("generation", "generation_config",
                                                              "profile", "reward_audit")}}

    def stamp(self, doc: dict) -> dict:
        doc.update(host=socket.gethostname(), pid=os.getpid(), threads=self.threads,
                   root=str(self.root), receipt_sha256=self.receipt_sha256,
                   payload_sha256=self.payload_sha256, adapter_sha256=I_WEIGHTS_SHA256,
                   trainer_sha256=TRAINER_SHA256, trie_sha256=TRIE_SHA256,
                   all_train_sha256=ALL_TRAIN_SHA256, profile=PROFILE)
        return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--trie", type=Path, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--exclusions", type=Path, required=True)
    p.add_argument("--only", choices=("p0", "score"), default="score",
                   help="unit kind to run (default score; P0-I rollouts moved to the GPU lane)")
    p.add_argument("--smoke", type=int, default=0,
                   help="validate the cached trie p against the trainer on N rows (T 1.0 and 1.3), then exit")
    a = p.parse_args(argv)
    todo = units(a.root / "payload", a.exclusions)
    if a.smoke:
        scorer = Scorer(a.root, a.threads, a.trie)
        rows = [r for r in todo["score-000"]["rows"] if len(scorer.paths[r["label"]]) > 1][:a.smoke]
        for r in rows:
            for t in (1.0, 1.3):
                mine = scorer.trie_p(r, t)
                print(json.dumps(dict(scorer.validate(r, mine, t), temperature=t,
                                      mine_seconds=mine["seconds"])), flush=True)
        return 0
    a.state.mkdir(parents=True, exist_ok=True)
    scorer, done = None, 0
    for name, unit in todo.items():
        if a.only and unit["kind"] != a.only:
            continue
        if not claim(a.state, name):
            continue
        if scorer is None:
            scorer = Scorer(a.root, a.threads, a.trie)
        if unit["kind"] == "p0":
            result = scorer.p0_unit(name, unit["rows"], unit["temp10"])
        else:
            result = scorer.score_unit(name, unit["rows"])
        atomic_json(a.state / "results" / f"{name}.json", scorer.stamp(result))
        done += 1
        print(f"{name} done on {socket.gethostname()} in {result['total_seconds']} s", flush=True)
    missing = [n for n in todo if todo[n]["kind"] == a.only and not (a.state / "results" / f"{n}.json").exists()]
    print(f"worker exits: did {done} units; {len(missing)} of {len(todo)} units lack results",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
