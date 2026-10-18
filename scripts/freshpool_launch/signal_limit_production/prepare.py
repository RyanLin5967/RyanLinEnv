"""Write (or check) the signal-limit v1 production package: activated copies, G0, plan.

  python scripts/freshpool_launch/signal_limit_production/prepare.py --write   # regenerate
  python scripts/freshpool_launch/signal_limit_production/prepare.py --check   # refuse unless identical

Light work only (JSON, SHA256 of files, the S_a / S_b split of 2,027 ids): no model, no image
decoding, no accelerator. Every compute stage of the study runs on the UWaterloo hosts
(``"execution": "uwaterloo"`` in the plan; scripts/freshpool_launch/uw_stage_a.py); this
machine only sequences, gates and keeps the journal.

Outputs, committed:
  docs/aircraft-rl-next-study-v1-activated.json              activated study copy
  docs/aircraft-rl-protocol-signal-limit-v1-activated.json   activated protocol copy
  scripts/freshpool_launch/signal_limit_production/evidence-base.json   labels + H rows
  scripts/freshpool_launch/signal_limit_production/plan.json            the launcher plan
  scripts/freshpool_launch/signal_limit_production/plan.sha256          its SHA256
Outputs in DATA_ROOT (ignored data, pinned by SHA256 in the plan), written only when absent
and otherwise required to be byte-identical:
  runs/aircraft-signal-limit-g0-v1/manifests/{s_a,s_b_scan_seed42,h_select,h_gate}.jsonl
  runs/aircraft-signal-limit-g0-v1/manifests/images -> data/fgvc-aircraft-2013b/development/images

Activated copies (study and protocol) differ from the unactivated files ONLY in ``status``,
``activated`` and ``activation``; ``check_activated_copy`` refuses anything else, and the
plan pins both copies and both sources. The activation record names ``approved_by: "root"``
as a placeholder: root approves by committing the activated copies' SHA256 in the decision
log; nothing here can approve. ``protocol_id`` is unchanged in the activated protocol copy
(the split salt ``data.pool_split.hash_protocol_id`` is a pinned literal either way); the copy
is identified by its path, its SHA256 and ``activation.activates``.

Device. The study's fixed profile says "FP32 MPS"; the protocol's activation prerequisites
say "on the chosen device". The activation record declares the chosen device: UWaterloo
hosts through uwrun, FP32, one device class for every stage (``EXECUTION`` below, chosen by
measured cost, receipts in reports/2026-10-18-signal-limit-uw-cost/), and discloses that W, I
and the controlled400 SFT pool were trained on MPS.
"""

import argparse
import hashlib
import json
import random
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_ROOT = Path("/Users/idide/RyanLinEnv")
FROZEN = Path("/Users/idide/wt/RyanLinEnv-signal-limit-launch-v1.noindex")
STATE = DATA_ROOT / "runs" / "aircraft-signal-limit-v1"
PYTHON = DATA_ROOT / ".venv-detection" / "bin" / "python"
MODEL = DATA_ROOT / "data" / "models" / "Qwen2-VL-2B-Instruct-895c3a49"
DATA_SPEC = DATA_ROOT / "data" / "fgvc-aircraft-2013b" / "candidates" / "fourshot-seed791" / (
    "data_spec.json")
TRAIN400 = DATA_SPEC.parent / "train400.jsonl"
W_ADAPTER = DATA_ROOT / "runs" / "aircraft-warmstart-v1" / "final_adapter"
I_ADAPTER = (DATA_ROOT / "runs" / "aircraft-controlled400-v1" / "setup" / "sft-1e5-time"
             / "published" / "final_adapter")
FRESHPOOL_G0 = DATA_ROOT / "runs" / "aircraft-freshpool-g0-v1" / "manifests"
G0_DIR = DATA_ROOT / "runs" / "aircraft-signal-limit-g0-v1" / "manifests"
DEVELOPMENT_IMAGES = DATA_ROOT / "data" / "fgvc-aircraft-2013b" / "development" / "images"
CONTROLLED400_PLAN = DATA_ROOT / "runs" / "aircraft-controlled400-launch-v1" / "launch-plan.json"
UWRUN = Path("/Users/idide/.claude/bin/uwrun")

STUDY = "docs/aircraft-rl-next-study-v1.json"
STUDY_ACTIVATED = "docs/aircraft-rl-next-study-v1-activated.json"
PROTOCOL = "docs/aircraft-rl-protocol-signal-limit-v1.json"
PROTOCOL_ACTIVATED = "docs/aircraft-rl-protocol-signal-limit-v1-activated.json"
EVIDENCE = "scripts/freshpool_launch/signal_limit_production/evidence-base.json"
PLAN = "scripts/freshpool_launch/signal_limit_production/plan.json"
PLAN_SHA = "scripts/freshpool_launch/signal_limit_production/plan.sha256"
COST_REPORT = "reports/2026-10-18-signal-limit-uw-cost/README.md"
SMOKE_REPORT = "reports/2026-10-18-signal-limit-uw-smoke/README.md"
ACTIVATION_KEYS = ("status", "activated", "activation")
ACTIVATION_DATE = "2026-10-18"
TRAINER = "src/vision_rl/vlm_local_training.py"
LABEL_TRIE = "data/fgvc-aircraft-2013b/audit/label-trie-qwen2vl2b.json"
CONFUSION_MAP = "data/fgvc-aircraft-2013b/audit/p0-confusion-map.json"
TAXONOMY = "data/fgvc-aircraft-2013b/audit/taxonomy-train-only.json"
G0_PREVIEW = "reports/2026-10-18-signal-limit-label-trie/g0-preview.json"
G0_MANIFESTS = ("s_a.jsonl", "s_b_scan_seed42.jsonl", "h_select.jsonl", "h_gate.jsonl")
SEED = 42

LAUNCH_PINS = tuple(f"scripts/freshpool_launch/{name}" for name in (
    "launch.py", "phase_worker.py", "signal_limit_launch.py", "stage_table.py",
    "uw_stage_a.py", "trainer_fragment.py", "h_assay.py", "p0_screen.py", "stage0_gate.py",
    "e0_gate.py")) + ("scripts/evaluate_signal_limit_gates.py",
                      "scripts/evaluate_freshpool_gates.py")

# ---------------------------------------------------------------- execution (measured)
# Chosen by measured FP32 cost on the hosts; every number names its receipt.
# See COST_REPORT for the measurements and the choice.
EXECUTION = {
    # CPU FP32 on the Xeon Gold 61xx class (Skylake-SP, AVX-512) at 16 torch threads. The
    # 2080 Ti is ~7x faster per update but cannot run the study: GRPO G16 FP32 is out of
    # memory there and G8 peaks at 10.0 of 10.6 GiB on a GPU another user shares; the 3070s
    # (8 GiB) cannot hold FP32 weights. Gold 6154 and 6136 measured the same at 16 threads.
    "device": "cpu",
    "hosts": ["iccad0", "iccad7", "iccad8", "iccad9", "iccad10", "iccad11", "iccad12",
              "iccad13"],
    "threads": 16,
    "per_host": 4,  # uwrun sessions of anyone on the host, ours included
    "min_idle_threads": 16,  # nproc - load1: the stage's 16 threads are idle when it starts
}
UW_PYTHON = "/var/tmp/r226lin-slenv-v1/bin/python"
UW_MODEL = "/var/tmp/r226lin-slmodel-v1/Qwen2-VL-2B-Instruct-895c3a49"

# Seconds per unit on the chosen device. Filled from COST_REPORT; caps are 1.25 x forecast.
UNIT = {
    "sft_update": None,  # A1/A4/A5/SFT-E per update
    "egsft_update": None,  # A2: SFT + one no-grad p forward
    "hn_sft_update": None,  # A3: SFT + eval-mode confuser scoring
    "scorer": None,  # one teacher-forced p forward (P0, screen)
    "greedy_image": None,  # assay: greedy decode, one image
    "g4_image": None,  # assay / P0: one G4 sampled group, one image
    "grpo_g8_update": None,  # screened dr_grpo beta 0, per accepted update, scorer incl.
    "grpo_g16_update": None,
}

PROFILE = ["--response-format", "answer_only", "--dtype", "float32", "--min-pixels", "50176",
           "--max-pixels", "200704", "--max-prompt-tokens", "2048",
           "--max-completion-length", "256", "--lora-rank", "8", "--temperature", "1.0"]
OWNER_PATTERNS = [
    "vision_rl.vlm_local_training", "vision_rl.aircraft_recipe", "phase_worker.py",
    "supervise_chunks.py", "mlx_vlm", "mlx_lm",
    "local_vision.py", "pills_local.py", "aircraft_local_probe.py", "supervise_local_vlm.py",
    "check_mlx_stream_parity.py", "aircraft_named_baseline.py",
]
QUIET_FILE = "/Users/idide/.claude/QUIET"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dumps(value):
    return json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def scan_order(ids, seed):
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order


# ---------------------------------------------------------------- G0 manifests


def g0_manifests():
    """{name: text} for the four G0 manifests. S_a / S_b: split_signal_limit over the
    freshpool G0's S (2,027 audit survivors) with the protocol's pinned salt; S_a in Stage A's
    order and S_b in the seed-42 scan order (both random.Random(42).shuffle(sorted(ids)));
    H-select / H-gate: the freshpool G0 files, byte for byte."""
    fresh = runpy.run_path(str(REPO / "scripts" / "prepare_aircraft_freshpool.py"))
    protocol = read_json(REPO / PROTOCOL)
    salt = fresh["pool_split_salt"](protocol)
    rows = read_jsonl(FRESHPOOL_G0 / "s.jsonl")
    s_a, s_b = fresh["split_signal_limit"](rows, salt)
    by_id = {row["id"]: row for row in rows}
    out = {}
    for name, part in (("s_a.jsonl", s_a), ("s_b_scan_seed42.jsonl", s_b)):
        order = scan_order([row["id"] for row in part], SEED)
        out[name] = jsonl(by_id[i] for i in order)
    for name in ("h_select.jsonl", "h_gate.jsonl"):
        out[name] = (FRESHPOOL_G0 / name).read_text()
    ids = [set(json.loads(line)["id"] for line in text.splitlines()) for text in out.values()]
    if any(a & b for n, a in enumerate(ids) for b in ids[n + 1:]):
        raise SystemExit("G0 manifests overlap")
    return out, {"s_a": len(s_a), "s_b": len(s_b),
                 "s_b_classes": len(Counter(row["label"] for row in s_b)),
                 "salt": salt}


def write_g0(texts, write):
    """Write absent G0 files (and the images link); refuse any present file that differs."""
    if write:
        G0_DIR.mkdir(parents=True, exist_ok=True)
    differs = []
    for name, text in texts.items():
        path = G0_DIR / name
        if path.exists():
            if path.read_text() != text:
                differs.append(name)
        elif write:
            path.write_text(text)
        else:
            differs.append(name)
    link = G0_DIR / "images"
    if link.is_symlink():
        if Path(link.readlink()) != DEVELOPMENT_IMAGES:
            differs.append("images")
    elif write and not link.exists():
        link.symlink_to(DEVELOPMENT_IMAGES, target_is_directory=True)
    else:
        differs.append("images")
    if differs:
        raise SystemExit(f"G0 files differ or are missing: {differs}")


# ---------------------------------------------------------------- activated copies


def check_activated_copy(source, activated):
    """The activated copy equals its source except ACTIVATION_KEYS; refuse anything else."""
    keys = set(source) | set(activated)
    changed = sorted(k for k in keys if source.get(k) != activated.get(k))
    if not set(changed) <= set(ACTIVATION_KEYS) or activated.get("activated") is not True:
        raise SystemExit(f"activated copy differs outside {ACTIVATION_KEYS}: {changed}")
    if [k for k in source if k in activated] != [k for k in activated if k in source]:
        raise SystemExit("activated copy reorders the source's fields")
    return changed


def frozen_identities():
    sys.path.insert(0, str(REPO / "src"))
    from vision_rl import vlm_local_training as vlt

    spec = read_json(DATA_SPEC)
    source = read_json(MODEL / "model-source.json")
    prompt = vlt.prompt_text(spec, "answer_only")
    processor_files = ("chat_template.json", "preprocessor_config.json", "tokenizer.json",
                       "tokenizer_config.json", "vocab.json", "merges.txt",
                       "generation_config.json", "config.json")
    return {
        "actor_model": {"repository": source["model_repository"],
                        "revision": source["revision"],
                        "model_source_manifest_sha256": sha(MODEL / "model-source.json")},
        "init_w": {"adapter_sha256": sha(W_ADAPTER / "adapter_model.safetensors"),
                   "config_sha256": sha(W_ADAPTER / "adapter_config.json"),
                   "path": str(W_ADAPTER.relative_to(DATA_ROOT))},
        "init_i": {"adapter_sha256": sha(I_ADAPTER / "adapter_model.safetensors"),
                   "config_sha256": sha(I_ADAPTER / "adapter_config.json"),
                   "path": str(I_ADAPTER.relative_to(DATA_ROOT))},
        "data_spec": {"path": str(DATA_SPEC.relative_to(DATA_ROOT)), "sha256": sha(DATA_SPEC),
                      "preprocessing_id": spec["preprocessing_id"]},
        "train400": {"path": str(TRAIN400.relative_to(DATA_ROOT)), "sha256": sha(TRAIN400)},
        "prompt": {"response_format": "answer_only",
                   "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "labels": len(spec["labels"])},
        "parser_and_trainer": {"output_grammar": vlt.OUTPUT_GRAMMAR,
                               "trainer_source_sha256": sha(REPO / TRAINER)},
        "processor_files_sha256": {name: sha(MODEL / name) for name in processor_files},
        "label_resources": {name: {"path": path, "sha256": sha(REPO / path)}
                            for name, path in (("label_trie", LABEL_TRIE),
                                               ("confusion_map", CONFUSION_MAP),
                                               ("taxonomy", TAXONOMY))},
        "g0_manifests_sha256": {name: sha(G0_DIR / name) for name in G0_MANIFESTS},
    }


def execution_record():
    return {
        "executor": "UWaterloo Engineering hosts through ~/.claude/bin/uwrun; this Mac only "
                    "sequences, gates and journals (plan execution: uwaterloo)",
        "device": EXECUTION["device"], "dtype": "float32", "hosts": EXECUTION["hosts"],
        "replaces": "fixed_profile.model's 'FP32 MPS' device. The protocol's activation "
                    "prerequisites name 'the chosen device'; this is it: CPU FP32, Xeon Gold "
                    "61xx (Skylake-SP, AVX-512) hosts, 16 torch threads per stage. dtype stays "
                    "FP32.",
        "why_not_gpu": "The RTX 2080 Ti (the only FP32-capable GPU, 11 GiB) ran SFT and "
                       "GRPO G8 but GRPO G16 FP32 ran out of memory, and Stage 0 must prove "
                       "E0 and measure c(G) at G16 (GB0 may choose it); G8 peaked at 10.0 of "
                       "10.6 GiB beside another user's process. The RTX 3070s (8 GiB) cannot "
                       "hold the 8.8 GB FP32 weights. Receipts: cost_receipts.",
        "one_device_class": "Every stage of the study (P0-I, Stage 0, Stage A, P0', RL, "
                            "SFT-E, SFT-T, LIM-T, every H assay) runs on this one device "
                            "class, so budget matching in active seconds (T_RL, caps) and "
                            "every H comparison is within one device.",
        "disclosed": [
            "W, I and the six controlled400 SFT pool endpoints were trained on MPS FP32; "
            "their native-VAL scores (a*) were measured on MPS. CPU FP32 numerics differ "
            "from MPS FP32 at the ULP level (kernel order); a new endpoint's VAL runs on the "
            "native-VAL package's own device.",
            "Shared hosts: other users' work runs beside a stage. The host probe requires "
            "the stage's 16 threads idle (nproc - load1) when it starts; wall-time caps "
            "include any later contention (an arm that hits its 1.25x cap stops the study, "
            "pre-registered).",
        ],
        "cost_receipts": COST_REPORT,
        "smoke_receipt": SMOKE_REPORT,
    }


def activation(source_path, kind):
    return {
        "activates": {"path": source_path, "sha256": sha(REPO / source_path)},
        "date": ACTIVATION_DATE,
        "approved_by": "root",
        "approval": "PLACEHOLDER: root approves by committing this file's SHA256 in "
                    "docs/decision-log-2026-10-16-claude.md; until then the plan exists but "
                    "the study is not authorized to start (start.py refuses without that "
                    "line).",
        "changed_vs_source": list(ACTIVATION_KEYS),
        "unchanged_vs_source": "every other top-level field, byte for byte after JSON "
                               "parsing (prepare.py check_activated_copy)",
        "scope": "Seed-42 development stages P0-I, Stage 0, Stage A (Gate A), and, only if "
                 "Gate A passes, Stage B through G3b, run by scripts/freshpool_launch/"
                 "signal_limit_launch.py from the plan in scripts/freshpool_launch/"
                 "signal_limit_production. Native VAL (G4, G5) and sealed TEST (G6) only "
                 "through the native-VAL package, never by this launch.",
        "execution": execution_record(),
        "frozen_identities": frozen_identities(),
        "kind": kind,
    }


def activated_copies():
    out = {}
    for source, target, kind, status in (
            (STUDY, STUDY_ACTIVATED, "study",
             "ACTIVATED (pending root's decision-log line) " + ACTIVATION_DATE
             + ": separately identified copy of the unactivated pre-registration "
             + STUDY + " with amendment 1; seed-42 development through G3b"),
            (PROTOCOL, PROTOCOL_ACTIVATED, "protocol",
             "ACTIVATED (pending root's decision-log line) " + ACTIVATION_DATE
             + ": separately identified copy of the unactivated revision " + PROTOCOL)):
        document = read_json(REPO / source)
        if document.get("activated") not in (None, False):
            raise SystemExit(f"{source} is not the unactivated source")
        copy = {key: value for key, value in document.items()}
        copy["status"] = status
        copy["activated"] = True
        copy["activation"] = activation(source, kind)
        check_activated_copy(document, copy)
        out[target] = dumps(copy)
    return out


# ---------------------------------------------------------------- evidence base and plan


def evidence_base():
    spec = read_json(DATA_SPEC)
    rows = {split: [{"id": r["id"], "label": r["label"]}
                    for r in read_jsonl(G0_DIR / f"{split}.jsonl")]
            for split in ("h_select", "h_gate")}
    return {"labels": spec["labels"], "rows": rows,
            "sources": {"labels": str(DATA_SPEC.relative_to(DATA_ROOT)),
                        "rows": str(G0_DIR.relative_to(DATA_ROOT))}}


def templates():
    train = ["--data-spec", "{data_spec}", "--model-directory", "{model_directory}",
             "--model", "Qwen/Qwen2-VL-2B-Instruct", "--device", "{device}", "--train-only",
             *PROFILE]
    inference = ["--data-spec", "{data_spec}", "--model-directory", "{model_directory}",
                 "--device", "{device}", *PROFILE, "--num-generations", "4"]
    identity = ["--init-adapter", "{init_adapter}", "--learning-rate", "{lr}",
                "--seed", "{seed}", "--stage-id", "{stage_id}"]

    def wrapper(arm, *extra):
        return ["--run-id", "{run_id}", "--arm", arm, "--lr", "{lr}", "--seed", "{seed}",
                "--fragment", "{fragment}", "--output", "{arm_dir}/trainer", *extra, "--"]

    sft = ["--order", "sequential", "--cycle", *train, "--save-steps", "100", *identity,
           "--max-steps", "6000", "--max-active-seconds", "{active_cap}"]
    rl = ["--mode", "grpo", "--train", "{scan_manifest}", *train,
          "--num-generations", "{group}", "--save-steps", "50", *identity,
          "--beta", "0", "--loss-type", "dr_grpo", "--scale-rewards", "none",
          "--reward", "exact", "--max-steps", "300", "--target-accepted", "300",
          "--screen-threshold", "0.5", "--screen-journal", "{arm_dir}/trainer/screen.jsonl",
          "--max-active-seconds", "{arm_cap}"]
    scan = ["--scan-manifest", "{scan_manifest}"]
    stage_a_extra = {"a1": [], "a2": ["--egsft-scale", "none", "--egsft-band-group", "8"],
                     "a3": ["--confusion-map", "{confusion_map}"],
                     "a4": ["--taxonomy", "{taxonomy}"], "a5": ["--label-trie", "{label_trie}"]}
    modes = {"a1": "sft", "a2": "egsft", "a3": "hn_sft", "a4": "hls_sft", "a5": "cce_sft"}
    argv = {
        "p0_i": ["--scan-manifest", "{scan_manifest}", "--adapter", "{init_adapter}",
                 "--fragment", "{fragment}", "--stage-dir", "{stage_dir}", *inference,
                 "--predict-groups", "4,8,16"],
        "p0_prime": ["--scan-manifest", "{scan_manifest}", "--adapter", "{init_adapter}",
                     "--fragment", "{fragment}", "--stage-dir", "{stage_dir}", *inference,
                     "--choose-groups", "8,16", "--a-floor", "{a_floor}"],
        # Stage 0 at the grid's largest learning rate, so a real LoRA change shows in 4 steps.
        "stage0": ["--fragment", "{fragment}", "--stage-dir", "{stage_dir}",
                   "--trainer-sha256", sha(REPO / TRAINER), "--", "--mode", "grpo",
                   "--train", "{train400_manifest}", *train, "--init-adapter",
                   "{init_adapter}", "--learning-rate", "0.0001", "--seed", "{seed}",
                   "--beta", "0", "--loss-type", "dr_grpo", "--scale-rewards", "none"],
        "assay": ["--run-id", "{run_id}", "--split", "{split}", "--manifest",
                  "{split_manifest}", "--adapter", "{adapter}", "--fragment", "{fragment}",
                  "--stage-dir", "{stage_dir}", *inference, "--training-group",
                  "{training_group}", "--label-trie", "{label_trie}",
                  "--label-trie-sha256", "{label_trie_sha256}"],
        "rl_screen": [*wrapper("RL", *scan), *rl, "--stop-after-steps", "100"],
        "rl_final": [*wrapper("RL", *scan, "--continue-output-from", "{resume_checkpoint}"),
                     *rl],
        "sft_e": [*wrapper("SFT-E"), "--mode", "sft", "--train", "{accepted_manifest}", *sft,
                  "--stop-after-steps", "300"],
        "sft_t": [*wrapper("SFT-T"), "--mode", "sft", "--train", "{scanned_manifest}", *sft,
                  "--continue-from-checkpoint", "{parent_checkpoint}",
                  "--stop-after-active-seconds", "{t_rl}"],
        "lim_t_a2": [*wrapper("LIM-T"), "--mode", "egsft", "--order", "sequential", "--cycle",
                     "--train", "{scanned_manifest}", *train, "--save-steps", "100",
                     *identity, "--egsft-scale", "none", "--egsft-band-group", "{group}",
                     "--max-steps", "100000", "--max-active-seconds", "{active_cap}",
                     "--stop-after-active-seconds", "{t_rl}"],
    }
    for arm, extra in stage_a_extra.items():
        argv[f"stage_a_{arm}"] = [
            *wrapper("{arm_label}"), "--mode", modes[arm], "--order", "sequential",
            "--train", "{manifest_s_a}", *train, "--save-steps", "100", *identity,
            "--max-steps", "600", "--max-active-seconds", "{active_cap}", *extra]
    entries = {"stage0": ("stage0_gate", "train"), "assay": ("h_assay", "assay"),
               "p0_i": ("p0_screen", "assay"), "p0_prime": ("p0_screen", "assay")}
    result = {}
    for name, parts in argv.items():
        module, kind = entries.get(name, ("trainer_fragment", "train"))
        path = REPO / "scripts" / "freshpool_launch" / f"{module}.py"
        result[name] = {"kind": kind, "entry": f"{module}:main", "entry_sha256": sha(path),
                        "argv": parts}
    return result


def forecasts():
    """Seconds per stage on the chosen device (UNIT x planned units); caps are 1.25x."""
    u = UNIT
    if any(v is None for v in u.values()):
        raise SystemExit(f"measured unit costs missing: {[k for k, v in u.items() if v is None]}")
    h = 200  # images per H split
    return {
        "p0": round(400 * u["scorer"] + 128 * u["g4_image"], 1),
        "stage0": round(sum((8 + 4 + 4 + 4) * u[f"grpo_g{g}_update"] for g in (8, 16)), 1),
        "assay_h_select": round(h * (u["greedy_image"] + u["g4_image"]) * 2, 1),
        "assay_h_gate": round(h * (u["greedy_image"] + 2 * u["g4_image"]) * 2, 1),
        "sft_e": round(300 * u["sft_update"], 1),
        "stage_a_sft": round(600 * u["sft_update"], 1),
        "stage_a_egsft": round(600 * u["egsft_update"], 1),
        "stage_a_hn_sft": round(600 * u["hn_sft_update"], 1),
        "stage_a_hls_sft": round(600 * u["sft_update"], 1),
        "stage_a_cce_sft": round(600 * u["sft_update"], 1),
    }


def repo_pins():
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "src/vision_rl"],
                             capture_output=True, text=True, timeout=60, check=True)
    files = tracked.stdout.split()
    if TRAINER not in files:
        raise SystemExit("git ls-files found no trainer: wrong repository")
    return (*LAUNCH_PINS, STUDY, STUDY_ACTIVATED, PROTOCOL, PROTOCOL_ACTIVATED, LABEL_TRIE,
            CONFUSION_MAP, TAXONOMY, G0_PREVIEW, *files)


def runtime_pins():
    """controlled400's pinned Mac runtime: the gates evaluator imports vision_rl here."""
    plan = read_json(CONTROLLED400_PLAN)
    pins = {path: value for path, value in plan["file_sha256"].items()
            if "/runs/aircraft-controlled400" not in path and "release-receipt" not in path}
    changed = [path for path, value in pins.items() if sha(path) != value]
    if changed:
        raise SystemExit(f"controlled400 runtime pins no longer match: {changed}")
    return pins, plan["environment"]


def launch_plan():
    pins, environment = runtime_pins()
    environment = {**environment, "ACCELERATE_TORCH_DEVICE": EXECUTION["device"],
                   "PYTHONPATH": f"{FROZEN / 'src'}:{DATA_ROOT}/runs/vlm-local-libs/site-packages"}
    data_pins = {str(p): sha(p) for p in (
        MODEL / "model-source.json", MODEL / ".pinned-download.json", DATA_SPEC, TRAIN400,
        W_ADAPTER / "adapter_model.safetensors", W_ADAPTER / "adapter_config.json",
        W_ADAPTER.parent / "local_training.json", I_ADAPTER / "adapter_model.safetensors",
        I_ADAPTER / "adapter_config.json", I_ADAPTER.parent / "local_training.json",
        PYTHON.resolve(), UWRUN)}
    launcher = runpy.run_path(str(REPO / "scripts" / "freshpool_launch" / "launch.py"))
    return {
        "schema": "signal-limit-launch-plan-v1",
        "execution": "uwaterloo",
        "study": {"path": STUDY_ACTIVATED, "sha256": sha(REPO / STUDY_ACTIVATED)},
        "protocol": {"path": PROTOCOL_ACTIVATED, "sha256": sha(REPO / PROTOCOL_ACTIVATED)},
        "python": str(PYTHON), "python_resolved": str(PYTHON.resolve()),
        "import_root": "src", "quiet_file": QUIET_FILE, "state_directory": str(STATE),
        "evaluator": "scripts/evaluate_signal_limit_gates.py",
        "environment": environment,
        "file_sha256": {**{p: sha(REPO / p) for p in repo_pins()}, **data_pins, **pins},
        "updates": {"stage_a": 600, "rl_screen": 100, "final": 300},
        "limits": {"minimum_free_bytes": 5 * 1024**3, "rss_limit_bytes": 24 * 1024**3,
                   "survivor_grace_seconds": 30, "poll_seconds": 5,
                   # Fetched outputs land here: the largest is a trainer run with its
                   # checkpoints (controlled400 work/rl: 107,084 KiB, du -sk 2026-10-17),
                   # plus its bundle; 4 of them for headroom.
                   "stage_headroom_bytes": 4 * 2 * 107084 * 1024},
        "forecast_seconds": forecasts(),
        "forecast_sources": {"unit_seconds": UNIT, "receipts": COST_REPORT,
                             "rule": "forecast = measured unit cost x planned units on the "
                                     "chosen device; cap = 1.25 x forecast (launch.py "
                                     "CAP_FACTOR); RL = c(G) from Stage 0 x 300; SFT-T and "
                                     "LIM-T = T_RL; assays x2 (a cap only guards a hang)"},
        "templates": templates(),
        "init_endpoints": {
            "I": {"adapter_sha256": sha(I_ADAPTER / "adapter_model.safetensors"),
                  "adapter_path": str(I_ADAPTER)},
            "W": {"adapter_sha256": sha(W_ADAPTER / "adapter_model.safetensors"),
                  "adapter_path": str(W_ADAPTER)}},
        "label_trie": {"path": LABEL_TRIE, "sha256": sha(REPO / LABEL_TRIE)},
        "confusion_map": {"path": CONFUSION_MAP, "sha256": sha(REPO / CONFUSION_MAP)},
        "taxonomy": {"path": TAXONOMY, "sha256": sha(REPO / TAXONOMY)},
        "train400": {"path": str(TRAIN400), "sha256": sha(TRAIN400)},
        "evidence_base": {"path": EVIDENCE, "sha256": sha(REPO / EVIDENCE)},
        "data_spec": str(DATA_SPEC),
        "model_directory": str(MODEL),
        "owner_patterns": OWNER_PATTERNS,
        "g0": {"manifest_dir": str(G0_DIR),
               "files_sha256": {name: sha(G0_DIR / name) for name in G0_MANIFESTS},
               "images_sha256": launcher["images_digest"](G0_DIR, G0_MANIFESTS)},
        "uw": {"uwrun": str(UWRUN), "hosts": EXECUTION["hosts"],
               "per_host": EXECUTION["per_host"], "max_concurrent": len(EXECUTION["hosts"]),
               "threads": EXECUTION["threads"], "python": UW_PYTHON,
               "model_directory": UW_MODEL, "min_local_gb": 8, "min_free_ram_gb": 32,
               "poll_seconds": 30, "unreachable_seconds": 1800,
               "device": EXECUTION["device"], "mixed_poll_seconds": 120,
               "min_idle_threads": EXECUTION["min_idle_threads"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    texts, sizes = g0_manifests()
    write_g0(texts, args.write)
    outputs = activated_copies()
    outputs[EVIDENCE] = dumps(evidence_base())
    stale = []
    for path, text in outputs.items():
        if args.write:
            (REPO / path).write_text(text)
        elif not (REPO / path).is_file() or (REPO / path).read_text() != text:
            stale.append(path)
    if stale:
        print(json.dumps({"status": "stale", "files": stale}))
        return 1
    plan = dumps(launch_plan())
    plan_sha = hashlib.sha256(plan.encode()).hexdigest()
    if args.write:
        (REPO / PLAN).write_text(plan)
        (REPO / PLAN_SHA).write_text(plan_sha + "\n")
    elif (REPO / PLAN).read_text() != plan or (REPO / PLAN_SHA).read_text() != plan_sha + "\n":
        print(json.dumps({"status": "stale", "files": [PLAN, PLAN_SHA]}))
        return 1
    print(json.dumps({"status": "written" if args.write else "current", "plan_sha256": plan_sha,
                      "g0": sizes, "a_floor": 300 * 1.25 / sizes["s_b"],
                      "activated_sha256": {p: sha(REPO / p)
                                           for p in (STUDY_ACTIVATED, PROTOCOL_ACTIVATED)},
                      "files": [*outputs, PLAN, PLAN_SHA]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
