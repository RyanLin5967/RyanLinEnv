"""Write (or check) the freshpool production G0 receipts and launch plan.

  python scripts/freshpool_launch/production/prepare.py --write   # regenerate every file
  python scripts/freshpool_launch/production/prepare.py --check   # refuse unless identical

Outputs, all committed:
  docs/aircraft-rl-protocol-freshpool-v1-activated.json   the activated protocol copy
  docs/aircraft-freshpool-branch-receipt-v1.json          controlled400 branch B0-B6 + D_req
  scripts/freshpool_launch/production/evidence-base.json  labels, H rows, a*, B4
  scripts/freshpool_launch/production/plan.json           the launch.py plan
  scripts/freshpool_launch/production/plan.sha256         its SHA256, the --expected value

Deterministic: every value is read from a pinned file (named below) or written out here. It
opens no model, no image and no accelerator. The decision record
(docs/decision-log-2026-10-16-claude.md) is written by hand and only pinned here, so edit it
BEFORE running --write; any later byte change to it, or to any file the plan pins, needs a new
--write and a new review.

Paths: repository files are pinned relative to the repository, and launch.py resolves them
against its own checkout, which start.py makes a detached worktree at FROZEN. Data, model,
warm adapter, manifests, the interpreter and the runtime packages live in DATA_ROOT (the main
checkout's ignored data/ and runs/) and are pinned by absolute path.
"""

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_ROOT = Path("/Users/idide/RyanLinEnv")
FROZEN = Path("/Users/idide/wt/RyanLinEnv-freshpool-launch-v1.noindex")
PYTHON = DATA_ROOT / ".venv-detection" / "bin" / "python"
MODEL = DATA_ROOT / "data" / "models" / "Qwen2-VL-2B-Instruct-895c3a49"
DATA_SPEC = DATA_ROOT / "data" / "fgvc-aircraft-2013b" / "candidates" / "fourshot-seed791" / (
    "data_spec.json")
W_ADAPTER = DATA_ROOT / "runs" / "aircraft-warmstart-v1" / "final_adapter"
MANIFESTS = DATA_ROOT / "runs" / "aircraft-freshpool-g0-v1" / "manifests"
STATE = DATA_ROOT / "runs" / "aircraft-freshpool-v1"
CONTROLLED400_PLAN = DATA_ROOT / "runs" / "aircraft-controlled400-launch-v1" / "launch-plan.json"
BASELINE = DATA_ROOT / "data" / "models" / "Qwen3.8-27B-8bit" / ".pinned-download.json"

PROTOCOL_ID = "fgvc-aircraft-freshpool-screened-grpo-v1"
ACTIVATED_ID = PROTOCOL_ID + "-activated-2026-10-17"
REVISION = "docs/aircraft-rl-protocol-freshpool-v1.json"
ACTIVATED = "docs/aircraft-rl-protocol-freshpool-v1-activated.json"
BRANCH = "docs/aircraft-freshpool-branch-receipt-v1.json"
DECISION = "docs/decision-log-2026-10-16-claude.md"
NATIVE_VAL = "reports/2026-10-17-controlled400-native-val"
EVIDENCE = "scripts/freshpool_launch/production/evidence-base.json"
PLAN = "scripts/freshpool_launch/production/plan.json"
PLAN_SHA = "scripts/freshpool_launch/production/plan.sha256"
W_SHA = "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"  # protocol W

# Repository files the stages execute; launch.py pins them before every stage and gate.
REPO_PINS = (
    "scripts/freshpool_launch/launch.py", "scripts/freshpool_launch/phase_worker.py",
    "scripts/freshpool_launch/e0_gate.py", "scripts/freshpool_launch/trainer_fragment.py",
    "scripts/freshpool_launch/h_assay.py", "scripts/freshpool_launch/p0_screen.py",
    "scripts/evaluate_freshpool_gates.py", "src/vision_rl/__init__.py",
    "src/vision_rl/vlm_local_training.py", "src/vision_rl/vlm.py", "src/vision_rl/data.py",
    "src/vision_rl/evaluation.py", "src/vision_rl/aircraft_data.py",
    "src/vision_rl/classification_comparison.py", "src/vision_rl/spatial.py",
    "src/vision_rl/training.py",
)  # every vision_rl module the trainer and evaluator import (measured with sys.modules)
TRAINER = "src/vision_rl/vlm_local_training.py"

# Owner patterns: Python programs that would take the accelerator (launch.py reads only
# processes whose argv[0] is a Python interpreter).
OWNER_PATTERNS = [
    "vision_rl.vlm_local_training", "vision_rl.aircraft_recipe", "phase_worker.py",
    "freshpool_launch/launch.py", "supervise_chunks.py", "mlx_vlm", "mlx_lm",
    "local_vision.py", "pills_local.py", "aircraft_local_probe.py", "supervise_local_vlm.py",
    "check_mlx_stream_parity.py",
]

# --- forecasts (seconds). Every number is a measurement named here, or the study's own
# extrapolation (docs/aircraft-freshpool-study-v1.json compute_plan_mps_hours, unverified).
GRPO_S = 20.29  # controlled400 RL: 8115.88 active s / 400 updates (optimizer_audit step 400)
SFT_S = 5.64  # controlled400 SFT: 1128.79 s / 200 durable updates
SCORER_S = 0.97  # study assumption for one teacher-forced forward (P0 measures it; G1 <= 2.0)
ACCEPTANCE = 0.5  # study assumption (G1 requires >= 0.30)
# controlled400 TRAIN32 assays (runs/aircraft-controlled400-v1/setup/*-assay receipts),
# slowest observed 32-image chunk: greedy 59.6 s, one G4 seed 299.4 s (RL400 endpoint).
GREEDY_CHUNK_MAX, G4_CHUNK_MAX, CHUNKS = 59.6, 299.4, math.ceil(200 / 32)
FORECAST = {
    "e0": 7200.0,  # study: "E0 resume/real-update gate (engineering ledger)" 2.0 h
    "p0": 3564.0,  # study: "P0 -> G1" 0.99 h
    # Assays are not budget-matched arms: the cap only stops a hang, so the forecast is the
    # slowest observed chunk rate over 7 chunks, not the typical one.
    "assay_h_select": round(CHUNKS * (GREEDY_CHUNK_MAX + G4_CHUNK_MAX), 1),  # 2513.0
    "assay_h_gate": round(CHUNKS * (GREEDY_CHUNK_MAX + 2 * G4_CHUNK_MAX), 1),  # 4608.8
    "rl": round(600 * (GRPO_S + SCORER_S / ACCEPTANCE), 1),  # 13338.0 = study T_RL
    "sft_e": round(600 * SFT_S, 1),  # 3384.0
    # EG-SFT runs one extra no-grad teacher-forced forward per update for w4(p)
    # (vlm_local_training._trainer_with_egsft); the study's 1.42 h omitted it.
    "egsft": round(600 * (SFT_S + SCORER_S), 1),  # 3966.0
}

PROFILE = ["--response-format", "answer_only", "--dtype", "float32", "--min-pixels", "50176",
           "--max-pixels", "200704", "--max-prompt-tokens", "2048",
           "--max-completion-length", "256", "--num-generations", "4", "--lora-rank", "8",
           "--temperature", "1.0"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dumps(value):
    return json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def evaluator():
    return runpy.run_path(str(REPO / "scripts" / "evaluate_freshpool_gates.py"))


# ---------------------------------------------------------------- receipts


def activated_protocol():
    revision = read_json(REPO / REVISION)
    if revision.get("protocol_id") != PROTOCOL_ID or revision.get("activated") is not False:
        raise SystemExit("the revision must be the unactivated freshpool protocol")
    salt = revision["data"]["train_holdout"]["hash_protocol_id"]
    if salt != PROTOCOL_ID or revision["revises"]["baseline_discovery_salt"]["literal"] != (
            PROTOCOL_ID):
        raise SystemExit("the revision's salts are not the pinned literal")
    source = read_json(MODEL / "model-source.json")
    spec = read_json(DATA_SPEC)
    sys.path.insert(0, str(REPO / "src"))
    from vision_rl import vlm_local_training as vlt

    prompt = vlt.prompt_text(spec, "answer_only")
    processor_files = ("chat_template.json", "preprocessor_config.json", "tokenizer.json",
                       "tokenizer_config.json", "vocab.json", "merges.txt",
                       "generation_config.json", "config.json")
    baseline = read_json(BASELINE)
    activation = {
        "activates": {"path": REVISION, "sha256": sha(REPO / REVISION),
                      "protocol_id": PROTOCOL_ID},
        "date": "2026-10-17",
        "authority": "Root's freshpool production workflow (task: activate under a separately "
                     "identified activated copy whose hash_protocol_id salt is unchanged). "
                     "Recorded in docs/decision-log-2026-10-16-claude.md D19.",
        "changed_vs_revision": ["protocol_id", "status", "activated", "activation"],
        "unchanged_vs_revision": "Every other top-level field, byte for byte after JSON "
                                 "parsing; scripts/freshpool_launch/launch.py refuses any "
                                 "difference (g0_activated_copy_differs_from_revision).",
        "salts_unchanged": {"data.train_holdout.hash_protocol_id": salt,
                            "revises.baseline_discovery_salt.literal": salt,
                            "g0_manifest_hash_protocol_id": read_json(
                                MANIFESTS / "manifest.json")["hash_protocol_id"]},
        "scope": "Seed-42 development stages E0 through G3b and EG-SFT, run by "
                 "scripts/freshpool_launch (launch package scripts/freshpool_launch/production). "
                 "Native VAL (G4, G5) and TEST (G6) only through the native-VAL package "
                 "(study work item 6), not by this launch.",
        "frozen_identities": {
            "actor_model": {"repository": source["model_repository"],
                            "revision": source["revision"],
                            "model_source_manifest_sha256": sha(MODEL / "model-source.json")},
            "initial_policy_w": {"adapter_sha256": sha(W_ADAPTER / "adapter_model.safetensors"),
                                 "config_sha256": sha(W_ADAPTER / "adapter_config.json"),
                                 "source_record_sha256": sha(
                                     W_ADAPTER.parent / "local_training.json"),
                                 "path": str(W_ADAPTER.relative_to(DATA_ROOT))},
            "data_spec": {"path": str(DATA_SPEC.relative_to(DATA_ROOT)),
                          "sha256": sha(DATA_SPEC),
                          "preprocessing_id": spec["preprocessing_id"]},
            "prompt": {"response_format": "answer_only",
                       "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "labels_sha256": hashlib.sha256(json.dumps(
                           spec["labels"], separators=(",", ":")).encode()).hexdigest(),
                       "labels": len(spec["labels"])},
            "parser_and_trainer": {"output_grammar": vlt.OUTPUT_GRAMMAR,
                                   "trainer_source_sha256": sha(REPO / TRAINER)},
            "processor_files_sha256": {name: sha(MODEL / name) for name in processor_files},
            "strong_baseline": {
                "repository": baseline["repository"], "revision": baseline["revision"],
                "model_source_manifest_sha256": baseline["manifest_sha256"],
                "provenance": "docs/local-qwen3.8-27b-8bit-provenance.json",
                "native_profile": "Not frozen here: strong_baseline.bounded_prompt_selection "
                                  "selects it on the 100-image VAL discovery sample before any "
                                  "full comparison. Needed only at G6; nothing in this launch "
                                  "runs the baseline."},
        },
    }
    status = ("ACTIVATED 2026-10-17 (separately identified copy of the unactivated revision "
              f"{REVISION}); seed-42 development launch authorized through G3b and EG-SFT")
    return {**revision, "protocol_id": ACTIVATED_ID, "status": status, "activated": True,
            "activation": activation}


def branch_receipt():
    comparison = read_json(REPO / NATIVE_VAL / "comparison.json")
    plan = read_json(REPO / NATIVE_VAL / "plan.json")
    versus = comparison["versus_common_initialization"]
    common = next(iter(versus.values()))["comparator"]
    if any(v["comparator"]["mean_class_accuracy"] != common["mean_class_accuracy"]
           for v in versus.values()):
        raise SystemExit("versus_common_initialization comparators disagree on W")
    sft = {arm: v["candidate"] for arm, v in versus.items() if arm.startswith("sft-")}
    if len(sft) != 6 or not comparison["all_inference_completed_without_unknown_or_errors"]:
        raise SystemExit("expected six completed controlled400 SFT endpoints")
    a_star_arm = max(sorted(sft), key=lambda arm: sft[arm]["mean_class_accuracy"])
    a_star = max(common["mean_class_accuracy"], sft[a_star_arm]["mean_class_accuracy"])
    d = evaluator()["d_req"](a_star)
    lr_of = {"1e5": "1e-05", "5e5": "5e-05", "1e4": "1e-04"}
    best_lr = lr_of[a_star_arm.split("-")[1]]
    best_exposure = max((a for a in sft if a.endswith("exposure")),
                        key=lambda arm: sft[arm]["mean_class_accuracy"])
    w_identity = plan["adapters"]["common"]
    w_match = (w_identity["weights_sha256"] == W_SHA == sha(W_ADAPTER / "adapter_model.safetensors")
               and w_identity["config_sha256"] == sha(W_ADAPTER / "adapter_config.json"))
    rl = versus["rl-exposure"]["candidate"]
    return {
        "schema_version": 1,
        "study_id": "aircraft-freshpool-screened-grpo-v1",
        "receipt": "controlled400 native-VAL branch resolution (study.json "
                   "branches_on_controlled400_val) and D_req publication, both G0 items",
        "sources": {
            "comparison": {"path": f"{NATIVE_VAL}/comparison.json",
                           "sha256": sha(REPO / NATIVE_VAL / "comparison.json")},
            "plan": {"path": f"{NATIVE_VAL}/plan.json",
                     "sha256": sha(REPO / NATIVE_VAL / "plan.json")},
            "report": f"{NATIVE_VAL}/README.md",
            "metric": comparison["metric"], "split": comparison["split"], "n": comparison["n"],
            "test_consumed": comparison["test_consumed"],
        },
        # The launcher's G0 fields.
        "a_star": a_star,
        "b4_edge_extension": best_lr == "1e-04",
        "w_valid_eos": common["valid_fraction"],
        "w_evaluator_refused": not (comparison["all_inference_completed_without_unknown_or_errors"]
                                    and comparison["failures_by_arm"]["common"] == 0),
        "w_val_accuracy": common["mean_class_accuracy"],
        "d_req": d["d_req"],
        "mode": d["mode"],
        "g3_margin": d["g3_margin"],
        "branches": {
            "B0": {"rule": "W interface failure (valid EOS < 0.95 or evaluator refusal) -> "
                           "do not launch",
                   "w_valid_eos_fraction": common["valid_fraction"],
                   "w_invalid_or_missing": common["invalid_answers_including_missing"],
                   "evaluator_failures_common": comparison["failures_by_arm"]["common"],
                   "outcome": "pass" if common["valid_fraction"] >= 0.95 else "do_not_launch"},
            "B1": {"rule": "W VAL outside [0.15, 0.85] -> do not launch",
                   "w_val_mean_class_accuracy": common["mean_class_accuracy"],
                   "outcome": "pass" if 0.15 <= common["mean_class_accuracy"] <= 0.85
                   else "do_not_launch"},
            "B2": {"rule": "publish D_req(a*) = max(0.05, 0.2(1-a*), 0.80-a*) (lower bound); "
                           "D_req > 0.10 -> mechanism mode, else full mode",
                   "a_star": a_star, "a_star_endpoint": a_star_arm,
                   "terms": {"floor": 0.05, "relative_error": 0.2 * (1 - a_star),
                             "absolute_0_80": 0.80 - a_star},
                   "d_req": d["d_req"], "mode": d["mode"],
                   "g3_margin": d["g3_margin"],
                   # a* + D_req = max(a* + 0.05, a* + 0.2(1 - a*), 0.80): the VAL accuracy
                   # an RL arm needs against an unchanged a*.
                   "rl_val_needed_lower_bound": a_star + d["d_req"],
                   "note": "Lower bound: fresh-pool SFT endpoints also enter the strongest-SFT "
                           "pool and can raise a*."},
            "B3": {"rule": "all controlled400 SFT endpoints enter the strongest-SFT pool (no "
                           "new VAL looks)",
                   "pool": {arm: {"val_mean_class_accuracy": sft[arm]["mean_class_accuracy"],
                                  "valid_fraction": sft[arm]["valid_fraction"],
                                  "adapter_sha256": plan["adapters"][arm]["weights_sha256"]}
                            for arm in sorted(sft)}},
            "B4": {"rule": "best controlled400 answer-SFT LR = 1e-4 -> add 2e-4 to both grids",
                   "best_endpoint": a_star_arm, "best_lr": best_lr,
                   "best_exposure_matched_endpoint": best_exposure,
                   "b4_edge_extension": best_lr == "1e-04",
                   "note": "Neither the best endpoint (equal time) nor the best exposure-"
                           "matched endpoint is at 1e-4, so the reading does not matter."},
            "B5": {"rule": "W identity match -> reuse W VAL; else +0.94 h",
                   "controlled400_common_adapter_sha256": w_identity["weights_sha256"],
                   "controlled400_common_config_sha256": w_identity["config_sha256"],
                   "freshpool_w_adapter_sha256": W_SHA,
                   "match": w_match, "reuse_w_val": w_match},
            "B6": {"rule": "RL400 VAL reported as stale-pool reference; RL*-RL400 never "
                           "reported as a pool effect",
                   "rl400_val_mean_class_accuracy": rl["mean_class_accuracy"],
                   "rl400_valid_fraction": rl["valid_fraction"],
                   "rl400_adapter_sha256": plan["adapters"]["rl-exposure"]["weights_sha256"]},
            "invariant": "No branch alters margins, seeds, TEST rules or the strongest-SFT "
                         "definition.",
        },
    }


def evidence_base(a_star, b4):
    spec = read_json(DATA_SPEC)
    rows = {}
    for split in ("h_select", "h_gate"):
        lines = (MANIFESTS / f"{split}.jsonl").read_text().splitlines()
        rows[split] = [{"id": r["id"], "label": r["label"]}
                       for r in (json.loads(line) for line in lines if line.strip())]
    return {"labels": spec["labels"], "rows": rows, "a_star": a_star, "b4_edge_extension": b4,
            "sources": {"labels": str(DATA_SPEC), "rows": str(MANIFESTS), "a_star": BRANCH}}


# ---------------------------------------------------------------- plan


def templates():
    unsaved = ["--data-spec", str(DATA_SPEC), "--model-directory", str(MODEL), "--model",
               "Qwen/Qwen2-VL-2B-Instruct", "--device", "mps", "--train-only", *PROFILE]
    inference = ["--data-spec", str(DATA_SPEC), "--model-directory", str(MODEL),
                 "--device", "mps", *PROFILE]
    identity = ["--init-adapter", "{w_adapter}", "--learning-rate", "{lr}", "--seed", "{seed}",
                "--stage-id", "{stage_id}"]
    tokenizations = ["--screen-tokenizations", "{screen_tokenizations}"]
    rl = ["--mode", "grpo", "--train", "{manifest_s_scan_seed42}", *unsaved, "--save-steps",
          "50", *identity, "--beta", "0.04", "--max-steps", "600", "--target-accepted", "600",
          "--screen-threshold", "0.5", "--screen-journal", "{arm_dir}/trainer/screen.jsonl",
          "--max-active-seconds", "{arm_cap}", *tokenizations]
    sft = ["--mode", "sft", "--order", "sequential", "--cycle", *unsaved, "--save-steps",
           "100", *identity, "--max-steps", "6000", "--max-active-seconds", "{active_cap}"]

    def wrapper(arm, *extra):
        return ["--run-id", "{run_id}", "--arm", arm, "--lr", "{lr}", "--seed", "{seed}",
                "--fragment", "{fragment}", "--output", "{arm_dir}/trainer", *extra, "--"]

    scan = ["--scan-manifest", "{manifest_s_scan_seed42}"]
    argv = {
        # E0 at the grid's largest learning rate, so a real LoRA change is visible in 4 steps.
        "e0": ["--fragment", "{fragment}", "--stage-dir", "{stage_dir}", "--trainer-sha256",
               sha(REPO / TRAINER), "--", "--mode", "grpo", "--train",
               "{manifest_s_scan_seed42}", *unsaved, "--init-adapter", "{w_adapter}",
               "--learning-rate", "0.0001", "--seed", "{seed}", "--beta", "0.04"],
        "assay": ["--run-id", "{run_id}", "--split", "{split}", "--manifest",
                  "{split_manifest}", "--adapter", "{adapter}", "--fragment", "{fragment}",
                  "--stage-dir", "{stage_dir}", *inference],
        "p0": ["--scan-manifest", "{manifest_s_scan_seed42}", "--adapter", "{w_adapter}",
               "--fragment", "{fragment}", "--stage-dir", "{stage_dir}", *inference],
        "rl150": [*wrapper("RL", *scan), *rl, "--stop-after-steps", "150"],
        "rl600": [*wrapper("RL", *scan, "--continue-output-from", "{resume_checkpoint}"), *rl],
        "sft_e": [*wrapper("SFT-E"), "--train", "{accepted_manifest}", *sft,
                  "--stop-after-steps", "600"],
        "sft_t": [*wrapper("SFT-T"), "--train", "{scanned_manifest}", *sft,
                  "--continue-from-checkpoint", "{parent_checkpoint}",
                  "--stop-after-active-seconds", "{t_rl}"],
        "egsft": [*wrapper("EG-SFT"), "--mode", "egsft", "--order", "sequential", "--train",
                  "{accepted_manifest}", *unsaved, "--save-steps", "100", *identity,
                  "--max-steps", "600", "--max-active-seconds", "{active_cap}",
                  *tokenizations],
    }
    entries = {"e0": ("e0_gate", "train"), "assay": ("h_assay", "assay"),
               "p0": ("p0_screen", "assay")}
    result = {}
    for name, parts in argv.items():
        module, kind = entries.get(name, ("trainer_fragment", "train"))
        path = REPO / "scripts" / "freshpool_launch" / f"{module}.py"
        result[name] = {"kind": kind, "entry": f"{module}:main", "entry_sha256": sha(path),
                        "argv": parts}
    return result


def runtime_pins():
    """controlled400's pinned runtime (package metadata, TRL/transformers/peft/accelerate/
    torch sources, the interpreter binary), minus its own run files; each must still match."""
    plan = read_json(CONTROLLED400_PLAN)
    pins = {path: value for path, value in plan["file_sha256"].items()
            if "/runs/aircraft-controlled400" not in path and "release-receipt" not in path}
    changed = [path for path, value in pins.items() if sha(path) != value]
    if changed:
        raise SystemExit(f"controlled400 runtime pins no longer match: {changed}")
    return pins, plan["environment"]


def launch_plan(receipt_paths, evidence_path):
    pins, environment = runtime_pins()
    environment = {**environment,
                   "PYTHONPATH": f"{FROZEN / 'src'}:{DATA_ROOT}/runs/vlm-local-libs/site-packages"}
    data_pins = {str(p): sha(p) for p in (
        MODEL / "model-source.json", MODEL / ".pinned-download.json", DATA_SPEC,
        W_ADAPTER / "adapter_model.safetensors", W_ADAPTER / "adapter_config.json",
        W_ADAPTER.parent / "local_training.json", PYTHON.resolve())}
    manifest = MANIFESTS / "manifest.json"
    return {
        "schema": "freshpool-launch-plan-v1",
        "study": "docs/aircraft-freshpool-study-v1.json",
        "protocol": ACTIVATED,
        "python": str(PYTHON),
        "state_directory": str(STATE),
        "evaluator": "scripts/evaluate_freshpool_gates.py",
        "environment": environment,
        "file_sha256": {**{p: sha(REPO / p) for p in REPO_PINS}, **data_pins, **pins},
        "limits": {"minimum_free_bytes": 5 * 1024**3, "rss_limit_bytes": 24 * 1024**3,
                   "survivor_grace_seconds": 30, "poll_seconds": 5},
        "forecast_seconds": FORECAST,
        "forecast_sources": {
            "grpo_seconds_per_update": GRPO_S, "sft_seconds_per_update": SFT_S,
            "scorer_seconds_per_candidate_assumed": SCORER_S,
            "acceptance_assumed": ACCEPTANCE,
            "assay_slowest_chunk_seconds": {"greedy": GREEDY_CHUNK_MAX, "g4_seed": G4_CHUNK_MAX},
            "rule": "cap = 1.25 x forecast per arm (launch.py CAP_FACTOR); see prepare.py"},
        "templates": templates(),
        "b4_edge_extension": False,
        "w_endpoint": {"adapter_sha256": W_SHA, "adapter_path": str(W_ADAPTER)},
        "owner_patterns": OWNER_PATTERNS,
        "g0": {"manifest_dir": str(MANIFESTS), "manifest_sha256": sha(manifest),
               "receipts": {name: {"path": path, "sha256": sha(REPO / path)}
                            for name, path in receipt_paths.items()}},
        "evidence_base": {"path": evidence_path, "sha256": sha(REPO / evidence_path)},
    }


def build():
    """{repository path: text} for every output, in dependency order."""
    outputs = {ACTIVATED: dumps(activated_protocol())}
    branch = branch_receipt()
    outputs[BRANCH] = dumps(branch)
    outputs[EVIDENCE] = dumps(evidence_base(branch["a_star"], branch["b4_edge_extension"]))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    outputs = build()
    if args.write:
        for path, text in outputs.items():
            (REPO / path).write_text(text)
    else:
        stale = [p for p, text in outputs.items() if (REPO / p).read_text() != text]
        if stale:
            print(json.dumps({"status": "stale", "files": stale}))
            return 1
    receipts = {"protocol_revision": ACTIVATED, "decision_record": DECISION, "branch": BRANCH}
    plan = dumps(launch_plan(receipts, EVIDENCE))
    plan_sha = hashlib.sha256(plan.encode()).hexdigest()
    if args.write:
        (REPO / PLAN).write_text(plan)
        (REPO / PLAN_SHA).write_text(plan_sha + "\n")
    elif (REPO / PLAN).read_text() != plan or (REPO / PLAN_SHA).read_text() != plan_sha + "\n":
        print(json.dumps({"status": "stale", "files": [PLAN, PLAN_SHA]}))
        return 1
    print(json.dumps({"status": "written" if args.write else "current", "plan_sha256": plan_sha,
                      "files": [*outputs, PLAN, PLAN_SHA]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
