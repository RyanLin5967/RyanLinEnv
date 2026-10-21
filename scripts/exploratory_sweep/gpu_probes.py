"""GPU-lane GATE 0 probes (root RL corrections 2026-10-18), run by arm_exec.py on a host.

  mode "e0": CUDA resume equivalence (C19). From the spec's init (I), the same GRPO config
             runs 4 updates uninterrupted twice (A, B: the run-to-run floor on this device)
             and 2 + 2 resumed from checkpoint-2 (C). Every adapter tensor of C is compared
             with A, and A with B (max |diff|, bitwise-equal count). Also the per-step
             prompt ids and reward groups. Writes e0.json. It measures; it does not judge.
  mode "p0": P0-I (C1, C14). At the init (I), frozen, on each probe manifest:
             greedy pass@1 (the assay's exact-EOS parse), then for each temperature T:
             G samples per image (the trainer's own serial sampler and reward) and
             p(gold) = P(any label-trie tokenisation of the gold label + EOS) under the
             sampler's transforms at T (the trainer's gold_probability, which the screened
             scan uses). Writes p0-<manifest>-T<T>.json rows and p0.json aggregates:
             acceptance a(u >= 0.5) with u = 1 - p^G - (1 - p)^G, realised mixed fraction
             split into recognition-mixed (a wrong sample is another valid label) and
             format-only (every wrong sample is unparsable), and pass@G among greedy-wrong.
Exploratory, TRAIN-derived rows only; never native VAL or sealed TEST.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


# ---- E0 ------------------------------------------------------------------------------
def run_e0(arm, sha_file) -> dict:
    import torch
    from safetensors.torch import load_file

    s = arm.spec
    if s.get("device") != "cuda" or s.get("dtype") != "float32" or s["mode"] != "e0":
        raise SystemExit("REFUSE: E0 is the CUDA fp32 resume check")
    total, half = int(s["e0_steps"]), int(s["e0_steps"]) // 2
    if total < 2 or total % 2:
        raise SystemExit("REFUSE: e0_steps must be even and >= 2")
    spec = dict(s, mode="grpo", stops=[half, total], max_steps_total=total)
    arm.spec = spec  # trainer_args reads the GRPO spec; save_steps = gcd(half, total)
    manifest = arm.manifest()
    base = arm.work / "e0"
    outs = {}
    for name in ("A", "B", "C"):
        arm.out = base / name
        if arm.out.exists():
            raise SystemExit(f"REFUSE: {arm.out} exists; E0 needs fresh runs")
        if name in ("A", "B"):
            arm.run_stage(f"e0-{name}", arm.trainer_args(manifest, total, None, None))
        else:
            arm.run_stage("e0-C1", arm.trainer_args(manifest, total, half, None))
            arm.run_stage("e0-C2", arm.trainer_args(manifest, total, total,
                                                    arm.out / f"checkpoint-{half}"))
        outs[name] = arm.out
    arm.spec = s

    def tensors(name):
        return load_file(str(outs[name] / "final_adapter" / "adapter_model.safetensors"))

    def compare(x, y):
        a, b = tensors(x), tensors(y)
        if set(a) != set(b):
            raise SystemExit(f"REFUSE: {x} and {y} adapters have different tensor names")
        diffs = {k: float((a[k].double() - b[k].double()).abs().max()) for k in sorted(a)}
        return {"tensors": len(a), "bitwise_equal": sum(torch.equal(a[k], b[k]) for k in a),
                "max_abs_diff": max(diffs.values()), "per_tensor_max_abs_diff": diffs,
                "weights_sha256": {x: sha_file(outs[x] / "final_adapter" /
                                               "adapter_model.safetensors"),
                                   y: sha_file(outs[y] / "final_adapter" /
                                               "adapter_model.safetensors")}}

    def journal(name, file):
        p = outs[name] / file
        if not p.is_file():
            return None
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    def prompts(name):
        r = json.loads((outs[name] / "endpoint_receipt.json").read_text())
        return r.get("committed_prompt_ids"), r.get("committed_active_seconds")

    result = {"schema": "gpu-lane-e0-v1", "steps": total, "resume_at": half,
              "A_vs_B_uninterrupted_floor": compare("A", "B"),
              "A_vs_C_resumed": compare("A", "C"),
              "prompt_ids": {n: prompts(n)[0] for n in outs},
              "active_seconds": {n: prompts(n)[1] for n in outs},
              "reward_groups_equal": {
                  "A_B": journal("A", "reward_groups.jsonl") == journal("B", "reward_groups.jsonl"),
                  "A_C": (journal("A", "reward_groups.jsonl")
                          == journal("C", "reward_groups.jsonl"))},
              "reward_groups": {n: journal(n, "reward_groups.jsonl") for n in outs},
              "dynamic_sampling": {n: journal(n, "dynamic_sampling.jsonl") for n in outs}}
    _write(arm.results / "e0.json", result)
    return result


# ---- P0-I ----------------------------------------------------------------------------
def run_p0(arm, sha_file) -> dict:
    import sys

    s = arm.spec
    if s["mode"] != "p0" or s.get("dtype") != "float32":
        raise SystemExit("REFUSE: P0-I runs frozen fp32")
    sys.path.insert(0, str(arm.payload / "code" / "src"))
    import os
    os.environ.update({k: v for k, v in arm.env.items() if k.isupper()})
    from importlib import import_module

    from vision_rl import vlm_local_training as vlt

    torch = import_module("torch")
    transformers = import_module("transformers")
    peft = import_module("peft")
    device = s.get("device", "cuda")
    spec = vlt.load_spec(arm.spec_path)
    labels = spec["labels"]
    trie_path = arm.payload / "data" / "label-trie-qwen2vl2b.json"
    sequences, _, trie_sha = vlt.load_label_trie(trie_path)
    if set(sequences) != set(labels):
        raise SystemExit("REFUSE: label trie labels differ from the task labels")
    G = int(s["num_generations"])
    base = vlt.Profile(response_format=s["response_format"], min_pixels=s["min_pixels"],
                       max_pixels=s["max_pixels"],
                       max_completion_length=s["max_completion_length"],
                       num_generations=G, temperature=1.0, dtype=s["dtype"])
    environment = vlt.preflight(device)
    vlt._set_threads(torch, int(s.get("threads", 4)))
    adapter = arm.init_adapter()
    adapter_sha = sha_file(adapter / "adapter_model.safetensors")
    processor = vlt._processor(arm.model, base)
    policy = transformers.AutoModelForImageTextToText.from_pretrained(
        str(arm.model), trust_remote_code=False, local_files_only=True, use_safetensors=True,
        dtype=getattr(torch, s["dtype"]), attn_implementation="sdpa")
    policy = peft.PeftModel.from_pretrained(policy, str(adapter), is_trainable=False)
    policy.requires_grad_(False).to(device).eval()
    out_all = {"schema": "gpu-lane-p0-v1", "init": s["init"], "adapter_sha256": adapter_sha,
               "label_trie_sha256": trie_sha, "G": G, "temperatures": s["temperatures"],
               "environment_device": environment.get("cuda"), "manifests": {}}
    for name in s["probe_manifests"]:
        if not name.startswith("p0_"):
            raise SystemExit(f"REFUSE: {name} is not a P0 probe manifest")
        path = arm.payload / "data" / "manifests" / name
        rows = vlt.load_classification(path, "train", labels,
                                       preprocessing_id=spec.get("preprocessing_id"),
                                       require_all_classes=False)
        if s.get("probe_limit"):  # smoke run: first N rows only, labelled as such
            rows = rows[: int(s["probe_limit"])]
        stage = arm.work / "p0" / name.removesuffix(".jsonl")
        stage.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        _, greedy = vlt._evaluate(policy, processor, rows, spec, base, device)
        greedy_answer = {
            r["id"]: (vlt.parse_answer(greedy[r["id"]]["response"], labels, allow_reasoning=True,
                                       **vlt._format_kwargs(base))
                      if greedy[r["id"]]["terminated"] else None) for r in rows}
        per_t = {}
        for T in s["temperatures"]:
            profile = vlt.Profile(**{**vlt.asdict(base), "temperature": float(T)})
            config = vlt._task_group_generation_config(policy, processor, profile)
            tdir = stage / f"T{T}"
            tdir.mkdir(parents=True, exist_ok=True)
            reward = vlt.make_reward(labels, G, config.eos_token_id, tdir / "reward_groups.jsonl",
                                     allow_reasoning=True, **vlt._format_kwargs(profile))
            transformers.set_seed(int(s["seed"]))
            groups = vlt._sample_task_groups(policy, processor, rows, spec, profile, device,
                                             tdir, reward, config)
            eos = list(config.eos_token_id)
            out_rows = []
            for r in rows:
                p = vlt.gold_probability(policy, processor, r, spec, profile, device,
                                         eos_token_ids=eos,
                                         suppress_tokens=[policy.config.image_token_id],
                                         alternatives=sequences[r["label"]])
                answers = [vlt.parse_answer(x["response"], labels, allow_reasoning=True,
                                            **vlt._format_kwargs(profile))
                           if x["terminated"] else None for x in groups[r["id"]]]
                out_rows.append({"id": r["id"], "label": r["label"], "p_gold": p,
                                 "u": 1 - p ** G - (1 - p) ** G,
                                 "greedy": greedy_answer[r["id"]],
                                 "greedy_correct": greedy_answer[r["id"]] == r["label"],
                                 "samples": answers,
                                 "correct": sum(a == r["label"] for a in answers)})
            _write(arm.results / f"p0-{name.removesuffix('.jsonl')}-T{T}.json", out_rows)
            per_t[str(T)] = aggregate(out_rows, G)
        out_all["manifests"][name] = {"rows": len(rows), "per_temperature": per_t,
                                      "greedy_accuracy": sum(greedy_answer[r["id"]] == r["label"]
                                                             for r in rows) / len(rows),
                                      "seconds": round(time.monotonic() - started, 1)}
        _write(arm.results / "p0.json", out_all)
    return out_all


def aggregate(rows: list, G: int) -> dict:
    n = len(rows)
    mixed = [r for r in rows if 0 < r["correct"] < G]
    recog = [r for r in mixed
             if any(a is not None and a != r["label"] for a in r["samples"])]
    wrong = [r for r in rows if not r["greedy_correct"]]
    ps = sorted(r["p_gold"] for r in rows)
    return {"n": n,
            "acceptance_u_ge_0_5": sum(r["u"] >= 0.5 for r in rows) / n,
            "mean_p_gold": sum(ps) / n, "median_p_gold": ps[n // 2],
            "p_gold_lt_0_02": sum(p < 0.02 for p in ps) / n,
            "realised_mixed_fraction": len(mixed) / n,
            "recognition_mixed_fraction": len(recog) / n,
            "format_only_mixed_fraction": (len(mixed) - len(recog)) / n,
            "all_correct_fraction": sum(r["correct"] == G for r in rows) / n,
            "all_wrong_fraction": sum(r["correct"] == 0 for r in rows) / n,
            "greedy_wrong": len(wrong),
            "pass_at_G_among_greedy_wrong": (sum(r["correct"] > 0 for r in wrong) / len(wrong)
                                             if wrong else None),
            "pass_at_G_greedy_wrong_images_fraction_of_all": (
                sum(r["correct"] > 0 for r in wrong) / n),
            "hard_greedy_wrong_and_p_ge_0_02": sum(
                (not r["greedy_correct"]) and r["p_gold"] >= 0.02 for r in rows) / n}
