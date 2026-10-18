"""Fixture tests for scripts/freshpool_launch/signal_limit_launch.py and uw_stage_a.py.

A fake stage entry, a fake ``torch``, a contract evaluator (or the real
scripts/evaluate_signal_limit_gates.py when the tree has it) and a fake ``uwrun`` that runs
each "remote" stage in its own directory on this machine. No model runs.

Every expected value is written out from the fixture design (stage order, selections, stop
reasons), never read back from the launcher.
"""

import hashlib
import importlib
import json
import os
import random
import shutil
import stat
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "freshpool_launch"
STUDY = ROOT / "docs" / "aircraft-rl-next-study-v1.json"
REAL_EVALUATOR = ROOT / "scripts" / "evaluate_signal_limit_gates.py"
GiB = 1024**3

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(os, "killpg"), reason="POSIX process groups")
sys.path.insert(0, str(PACKAGE))
signal_limit = importlib.import_module("signal_limit_launch")


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# The study's pinned identities (docs/aircraft-rl-next-study-v1.json stages; the gates
# evaluator checks the same two hashes).
I_SHA = "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"
W_SHA = "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"
PYTHON_SHA = sha_file(os.path.realpath(sys.executable))
LABELS = ["A", "B"]
OTHER = {"A": "B", "B": "A"}
H_ROWS = {split: [(f"{prefix}{n:02d}", "A" if n < 10 else "B") for n in range(20)]
          for split, prefix in (("h_select", "h"), ("h_gate", "g"))}
COMPLETIONS = {"h_select": 5, "h_gate": 9}


def seed42(ids):
    order = sorted(ids)
    random.Random(42).shuffle(order)
    return order


S_A = seed42(f"a{n:04d}" for n in range(600))
S_B = seed42(f"b{n:04d}" for n in range(700))
A_FLOOR = 375 / 700  # 300 x 1.25 / |S_b|
TRIE_TEXT = '{"trie": "fixture"}'
TRIE_SHA = sha_text(TRIE_TEXT)
MAP_TEXT = '{"map": "fixture"}'
MAP_SHA = sha_text(MAP_TEXT)
TAXONOMY_TEXT = '{"variants": "fixture"}'
TAXONOMY_SHA = sha_text(TAXONOMY_TEXT)
# What each Stage A arm's trainer records (trainer_fragment.signal_limit_profile): A2 is the
# screened no-std limit at B_8; A3, A4 and A5 hashed the plan's frozen resources.
ARM_IDENTITY = {"A2": {"egsft_scale": "none", "egsft_band_group": 8},
                "A3": {"confusion_map_sha256": MAP_SHA},
                "A4": {"taxonomy_sha256": TAXONOMY_SHA},
                "A5": {"label_trie_sha256": TRIE_SHA}}


def lr_tag(lr):
    return {1e-5: "1e-05", 5e-5: "5e-05", 1e-4: "1e-04"}[lr]


# ---------------------------------------------------------------- the fixture design
# Per endpoint, k = images per class (of 10) answered correctly on every completion; the rest
# are answered with the other label. Selections that follow (written out by hand):
#   A1* = stage-a-a1-5e-05 (k 2) is C* (A3* k 1 and A5* k 1 trail it); A2* =
#   stage-a-a2-5e-05 (k 7) wins Gate A and is I*.
#   G2 continues rl-1e-04 (k 9) then rl-5e-05 (k 8); RL* = rl-1e-04-s42-u300 (k 10).
#   SFT-E 5e-5 and 1e-4 tie at k 7; SFT-E* = 5e-5 (100 s < 101 s); both are SFT-T LRs.
#   SFT-T* = 5e-5 (k 8). LIM-T runs A2's objective (egsft) at A2*'s LR 5e-5.
STAGE_A_K = {"A1": (1, 2, 0), "A2": (5, 7, 6), "A3": (1, 0, 0), "A5": (0, 1, 0)}
A4_K = 2
I_STAR = "stage-a-a2-5e-05-s42"
RL100_K = {1e-5: 6, 5e-5: 8, 1e-4: 9}
RL300_K = {5e-5: 9, 1e-4: 10}
SFT_E = {1e-5: (5, 100.0), 5e-5: (7, 100.0), 1e-4: (7, 101.0)}
SFT_T_K = {5e-5: 8, 1e-4: 7}
H_GATE_K = {I_STAR: 5, "rl-1e-04-s42-u300": 10, "sft-e-5e-05-s42": 6, "sft-t-5e-05-s42": 7,
            "lim-t-5e-05-s42": 6}
LIM_T_K = 6
RL100_SECONDS, T_RL = 250.0, 1000.0
SFT_T_UPDATES, LIM_T_UPDATES = 1000, 800
ACCEPTED_100, SCANNED_100 = S_B[:100], S_B[:200]
ACCEPTED_300, SCANNED_300 = S_B[:300], S_B[:600]  # RL*'s

STAGE_A_RUNS = [f"stage-a-{arm.lower()}-{lr_tag(lr)}-s42" for arm in ("A1", "A2", "A3", "A5")
                for lr in (1e-5, 5e-5, 1e-4)]
EXPECTED_ORDER = [
    "P0-I", "Stage-0", *STAGE_A_RUNS, *(f"{r}-h_select" for r in STAGE_A_RUNS),
    "stage-a-a4-5e-05-s42", "stage-a-a4-5e-05-s42-h_select", "P0-prime",
    "rl-1e-05-s42-u100", "rl-1e-05-s42-u100-h_select",
    "rl-5e-05-s42-u100", "rl-5e-05-s42-u100-h_select",
    "rl-1e-04-s42-u100", "rl-1e-04-s42-u100-h_select",
    "rl-1e-04-s42-u300", "rl-1e-04-s42-u300-h_select",
    "rl-5e-05-s42-u300", "rl-5e-05-s42-u300-h_select",
    "sft-e-1e-05-s42", "sft-e-1e-05-s42-h_select",
    "sft-e-5e-05-s42", "sft-e-5e-05-s42-h_select",
    "sft-e-1e-04-s42", "sft-e-1e-04-s42-h_select",
    "rl-1e-04-s42-u300-h_gate", "sft-e-5e-05-s42-h_gate", f"{I_STAR}-h_gate",
    "sft-t-5e-05-s42", "sft-t-5e-05-s42-h_select",
    "sft-t-1e-04-s42", "sft-t-1e-04-s42-h_select",
    "lim-t-5e-05-s42", "lim-t-5e-05-s42-h_select",
    "sft-t-5e-05-s42-h_gate", "lim-t-5e-05-s42-h_gate",
]
EXPECTED_SELECTIONS = {
    "A1*": "stage-a-a1-5e-05-s42", "I*": I_STAR, "GA_winner": I_STAR, "group": 8,
    "rl_continuers": ["rl-1e-04-s42-u100", "rl-5e-05-s42-u100"],
    "rl_star": "rl-1e-04-s42-u300", "sft_e_star": "sft-e-5e-05-s42",
    "sft_t_lrs": [5e-05, 1e-04], "sft_t_star": "sft-t-5e-05-s42",
}


def predictions(split, k, single=False):
    found = {}
    for split_id, label in H_ROWS[split]:
        index = int(split_id[1:]) % 10
        answer = label if index < k else OTHER[label]
        found[split_id] = answer if single else [answer] * COMPLETIONS[split]
    return found


def assay(run_id, split, k, group=0):
    return {"payload": {
        "run_id": run_id, "split": split, "adapter_sha256": sha_text(run_id),
        "predictions": predictions(split, k), "eos_fraction": 1.0, "chunks": 1,
        "chunk_size": 32, "constrained_greedy": {
            "trie_sha256": TRIE_SHA, "predictions": predictions(split, k, single=True),
            "eos_fraction": 1.0},
        "training_group": group or None, "assay_group": 4}}


def series(*pieces):
    """{step: committed seconds}, linear within each (first, last, s_first, s_last) piece."""
    steps = {}
    for first, last, start, stop in pieces:
        for step in range(first, last + 1):
            steps[str(step)] = start + (stop - start) * (step - first) / max(1, last - first)
        steps[str(first)], steps[str(last)] = start, stop
    return steps


def train(run_id, arm, lr, updates, seconds, steps, exposure, reference, **extra):
    return {"payload": {
        "status": "completed", "run_id": run_id, "arm": arm, "lr": lr, "seed": 42,
        "updates": updates, "committed_active_seconds": seconds,
        "step_committed_active_seconds": steps, "adapter_sha256": sha_text(run_id),
        "reference_sha256": reference, "useful_mixed": [], "make_checkpoint": True,
        "committed_prompt_ids": exposure, **extra}}


def p0_document(adapter, group, **extra):
    u = 1 - 0.5**group - 0.5**group
    journal = [{"id": i, "p": 0.5, "u": u, "accepted": True, "scorer_seconds": 1.0,
                "adapter_sha256": adapter} for i in S_B[:400]]
    rollouts = [{"id": i, "seed": seed, "completions": [
        {"correct": n == 0, "valid": True, "eos": True, "canonical_tokenization": True,
         "label_only": True, "completion_ids": [5, 2] if n == 0 else [3, 2]}
        for n in range(group)]} for i in S_B[:64] for seed in (42, 314)]
    return {"payload": {"seed": 42, "adapter_sha256": adapter, "scan_order": S_B,
                        "journal": journal, "rollouts": rollouts, "group": group, **extra}}


RL_PROFILE = {"mode": "grpo", "loss_type": "dr_grpo", "scale_rewards": "none", "beta": 0.0,
              "group": 8, "reward": "exact"}


def scenario():
    i_star_sha = sha_text(I_STAR)
    s = {
        "P0-I": p0_document(I_SHA, 4, predicted={
            str(g): {"acceptance": 1.0, "mean_u": 1 - 2 * 0.5**g,
                     "mean_u_accepted": 1 - 2 * 0.5**g} for g in (4, 8, 16)}),
        "Stage-0": {"payload": {
            "checks": {g: dict.fromkeys(signal_limit.STAGE0_CHECKS, True) for g in ("8", "16")},
            "sampler_teacher_max_abs_logprob_difference": 0.0005, "e0_groups": [8, 16],
            "e0_steps": 4, "probe_accepted": 8,
            "c_accepted_steps": {"8": 8, "16": 8},
            "c_seconds_per_accepted_update": {"8": 10.0, "16": 20.0}}},
        "P0-prime": p0_document(i_star_sha, 8, predicted={
            str(g): {"acceptance": 1.0, "mean_u": 1 - 2 * 0.5**g,
                     "mean_u_accepted": 1 - 2 * 0.5**g} for g in (8, 16)},
            chosen_group=8, choose_groups=[8, 16], a_floor=A_FLOOR),
    }
    modes = {"A1": "sft", "A2": "egsft", "A3": "hn_sft", "A4": "hls_sft", "A5": "cce_sft"}
    for arm, ks in [*STAGE_A_K.items(), ("A4", (None, A4_K, None))]:
        for lr, k in zip((1e-5, 5e-5, 1e-4), ks):
            if k is None:
                continue
            run_id = f"stage-a-{arm.lower()}-{lr_tag(lr)}-s42"
            s[run_id] = train(run_id, arm, lr, 600, 60.0, series((1, 600, 0.1, 60.0)),
                              S_A, W_SHA, mode=modes[arm], init_sha256=W_SHA,
                              **ARM_IDENTITY.get(arm, {}))
            s[f"{run_id}-h_select"] = assay(run_id, "h_select", k)
    s[f"{I_STAR}-h_gate"] = assay(I_STAR, "h_gate", H_GATE_K[I_STAR])
    for lr, k in RL100_K.items():
        run_id = f"rl-{lr_tag(lr)}-s42-u100"
        s[run_id] = train(run_id, "RL", lr, 100, RL100_SECONDS,
                          series((1, 100, 1.0, RL100_SECONDS)), ACCEPTED_100, i_star_sha,
                          make_manifests={"accepted": ACCEPTED_100, "scanned": SCANNED_100},
                          useful_mixed=[True] * 100, **RL_PROFILE)
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", k, 8)
    for lr, k in RL300_K.items():
        run_id = f"rl-{lr_tag(lr)}-s42-u300"
        s[run_id] = train(run_id, "RL", lr, 300, T_RL,
                          series((1, 100, 1.0, RL100_SECONDS), (101, 300, 252.0, T_RL)),
                          ACCEPTED_300, i_star_sha,
                          make_manifests={"accepted": ACCEPTED_300, "scanned": SCANNED_300},
                          useful_mixed=[True] * 300, **RL_PROFILE)
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", k, 8)
    s["rl-1e-04-s42-u300-h_gate"] = assay("rl-1e-04-s42-u300", "h_gate", 10, 8)
    for lr, (k, seconds) in SFT_E.items():
        run_id = f"sft-e-{lr_tag(lr)}-s42"
        s[run_id] = train(run_id, "SFT-E", lr, 300, seconds, series((1, 300, 0.2, seconds)),
                          ACCEPTED_300, i_star_sha, receipt_status="boundary_reached",
                          requested_boundary={"stop_after_steps": 300,
                                              "stop_after_active_seconds": None})
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", k)
    s["sft-e-5e-05-s42-h_gate"] = assay("sft-e-5e-05-s42", "h_gate", 6)
    for lr, k in SFT_T_K.items():
        run_id = f"sft-t-{lr_tag(lr)}-s42"
        parent = SFT_E[lr][1]
        exposure = ACCEPTED_300 + [SCANNED_300[n % 600] for n in range(SFT_T_UPDATES - 300)]
        s[run_id] = train(run_id, "SFT-T", lr, SFT_T_UPDATES, T_RL,
                          series((301, SFT_T_UPDATES - 1, parent + 0.5, T_RL - 0.3),
                                 (SFT_T_UPDATES, SFT_T_UPDATES, T_RL, T_RL)),
                          exposure, i_star_sha)
        s[f"{run_id}-h_select"] = assay(run_id, "h_select", k)
    s["sft-t-5e-05-s42-h_gate"] = assay("sft-t-5e-05-s42", "h_gate", 7)
    lim = "lim-t-5e-05-s42"
    s[lim] = train(lim, "LIM-T", 5e-5, LIM_T_UPDATES, T_RL,
                   series((1, LIM_T_UPDATES - 1, 1.0, T_RL - 0.5),
                          (LIM_T_UPDATES, LIM_T_UPDATES, T_RL, T_RL)),
                   [SCANNED_300[n % 600] for n in range(LIM_T_UPDATES)], i_star_sha,
                   mode="egsft", init_sha256=i_star_sha, egsft_scale="none",
                   egsft_band_group=8)  # the chosen G (GB0 chose 8)
    s[f"{lim}-h_select"] = assay(lim, "h_select", LIM_T_K)
    s[f"{lim}-h_gate"] = assay(lim, "h_gate", H_GATE_K[lim])
    return s


# ---------------------------------------------------------------- fakes

FAKE_ENTRY = '''
import argparse, json, os, time
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)  # --arm is not --arm-dir
    for name in ("--stage-id", "--fragment", "--arm-dir", "--scan-manifest"):
        parser.add_argument(name)
    known, _ = parser.parse_known_args(argv)
    with open(os.environ["FAKE_CALLS"], "a") as handle:
        handle.write(json.dumps({
            "stage": known.stage_id, "argv": argv, "cwd": os.getcwd(),
            "uwrun_threads": os.environ.get("UWRUN_THREADS"),
            "torch_threads": os.environ.get("VISION_RL_TORCH_THREADS"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "device": os.environ.get("ACCELERATE_TORCH_DEVICE")}) + "\\n")
    spec = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text()).get(known.stage_id)
    if spec is None:
        return 7
    behaviour = spec.get("behaviour", {})
    if os.environ.get("FAKE_INIT_IN_RUN") == "1" and "--init" in argv:
        # The real trainer reads the init adapter's run record: <run>/local_training.json
        # beside <run>/final_adapter.
        init = Path(argv[argv.index("--init") + 1])
        if init.name != "final_adapter" or not (init.parent / "local_training.json").is_file():
            return 11
    time.sleep(behaviour.get("sleep", 0))
    if "exit" in behaviour:
        return behaviour["exit"]
    arm = Path(known.arm_dir).resolve()
    payload = dict(spec["payload"])
    if payload.pop("make_checkpoint", False):
        checkpoint = arm / f"checkpoint-{payload['updates']}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        adapter = arm / "final_adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_text(
            behaviour.get("adapter_text", payload["run_id"]))
        (arm / "local_training.json").write_text("{}")  # the run record, as the trainer
        payload.update(checkpoint=str(checkpoint), adapter_path=str(adapter))
    if payload.get("arm") == "RL":
        groups = arm / "trainer" / "reward_groups.jsonl"
        groups.parent.mkdir(parents=True, exist_ok=True)
        groups.write_text("".join(json.dumps({"id": f"g{n}", "mixed": m}) + "\\n"
                                  for n, m in enumerate(payload["useful_mixed"])))
    manifests = payload.pop("make_manifests", None)
    if manifests:
        images = {}
        if known.scan_manifest:
            scan = Path(known.scan_manifest).resolve()
            for line in scan.read_text().splitlines():
                row = json.loads(line)
                images[row["id"]] = str((scan.parent / row["image"]).resolve())
        for key in ("accepted", "scanned"):
            path = Path(known.fragment).resolve().parent / f"{key}_manifest.jsonl"
            path.write_text("".join(json.dumps(
                {"id": i, **({"image": images[i]} if images else {})}) + "\\n"
                for i in manifests[key]))
            payload[f"{key}_manifest"] = str(path)
    for flag in ("--train",):
        if flag in argv and payload.get("arm") == "SFT-E":
            # SFT-E trains on RL*'s accepted manifest: every image it names must open here.
            manifest = Path(argv[argv.index(flag) + 1]).resolve()
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                if "image" in row:
                    image = Path(row["image"])
                    image = image if image.is_absolute() else manifest.parent / image
                    if image.read_text() != "jpg":
                        return 9
    Path(known.fragment).write_text(json.dumps(payload))
    time.sleep(behaviour.get("linger", 0))
    return 0
'''

FAKE_TORCH = '''
class _MPS:
    def is_available(self):
        return True

    def recommended_max_memory(self):
        return 48 * 1024**3

    def set_per_process_memory_fraction(self, fraction):
        pass

    def driver_allocated_memory(self):
        return 1024**3

    def current_allocated_memory(self):
        return 512 * 1024**2


mps = _MPS()


class _Backends:
    pass


backends = _Backends()
backends.mps = mps


def set_num_threads(count):
    pass


class _Cuda:
    def is_available(self):
        return True

    def device_count(self):
        return 1

    def get_device_name(self, index):
        return "Fake RTX 2080 Ti"

    def max_memory_allocated(self):
        return 3 * 1024**3


cuda = _Cuda()
'''

# The evaluator contract signal_limit_launch.py relies on (its docstring), implemented
# plainly: macro = mean over classes of the fraction of correct completions.
CONTRACT_EVALUATOR = '''
import argparse, json, random
from pathlib import Path

GRID = (1e-5, 5e-5, 1e-4)
COMPLETIONS = {"h_select": 5, "h_gate": 9}
FAIL_AT = __FAIL_AT__


def scan_order(ids, seed):
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order


def best(candidates, key):
    top = max(item[key] for item in candidates)
    tied = [item for item in candidates if item[key] >= top - 1e-9]
    return min(tied, key=lambda item: (item["committed_active_seconds"], item["run_id"]))


class Study:
    def __init__(self, evidence):
        self.evidence, self.rows = evidence, evidence["rows"]
        if not evidence.get("s_b_ids"):
            raise ValueError("s_b_ids missing")
        self.endpoints = {e["run_id"]: e for e in evidence["endpoints"]}
        for e in evidence["endpoints"]:
            if e["arm"] != "W" and len(str(e.get("init_sha256"))) != 64:
                raise ValueError(f"{e['run_id']} has no init_sha256")
            if e["arm"] == "RL" and (e.get("loss_type"), e.get("scale_rewards"),
                                     e.get("beta")) != ("dr_grpo", "none", 0.0):
                raise ValueError(f"{e['run_id']} RL profile missing")

    def family(self, arm, updates=None, lrs=None):
        found = [e for e in self.endpoints.values() if e["arm"] == arm
                 and (updates is None or e["updates"] == updates)]
        if sorted(e["lr"] for e in found) != sorted(GRID if lrs is None else lrs):
            raise ValueError(f"{arm} needs one endpoint per LR")
        return found

    def exact_updates(self, endpoints, updates=300):
        if any(e["updates"] != updates for e in endpoints):
            raise ValueError("updates differ")
        return endpoints

    def macro(self, endpoint, split):
        predictions = endpoint["splits"][split]["predictions"]
        classes = {}
        for row in self.rows[split.replace("_constrained", "")]:
            answers = predictions[row["id"]]
            answers = answers if isinstance(answers, list) else [answers]
            classes.setdefault(row["label"], []).append(
                sum(a == row["label"] for a in answers) / len(answers))
        return sum(sum(v) / len(v) for v in classes.values()) / len(classes)

    def h_scored(self, endpoints, split="h_select"):
        return [{"run_id": e["run_id"], "lr": e["lr"],
                 "committed_active_seconds": e["committed_active_seconds"],
                 "macro": self.macro(e, split)} for e in endpoints]

    def star(self, arm):
        return best(self.h_scored(self.exact_updates(self.family(arm), 600)), "macro")

    def select_rl_continuers(self):
        scored = self.h_scored(self.family("RL", updates=100))
        first = best(scored, "macro")
        second = best([c for c in scored if c is not first], "macro")
        return scored, [first, second]

    def select_rl_star(self):
        _, continuers = self.select_rl_continuers()
        final = self.family("RL", updates=300, lrs=[c["lr"] for c in continuers])
        return None, self.endpoints[best(self.h_scored(final), "macro")["run_id"]]

    def select_sft_e(self):
        scored = self.h_scored(self.exact_updates(self.family("SFT-E")))
        star = best(scored, "macro")
        qualifying = sorted(c["lr"] for c in scored if c["macro"] >= star["macro"] - 0.02)
        return scored, self.endpoints[star["run_id"]], qualifying

    def select_sft_t(self):
        _, _, qualifying = self.select_sft_e()
        scored = self.h_scored(self.family("SFT-T", lrs=qualifying))
        return scored, self.endpoints[best(scored, "macro")["run_id"]]


def evaluate(evidence):
    study = Study(evidence)
    arms = {e["arm"] for e in evidence["endpoints"]}
    has_h_gate = lambda arm: any(e["arm"] == arm and "h_gate" in e["splits"]
                                 for e in evidence["endpoints"])

    def ga():
        stars = {arm: study.star(arm) for arm in ("A1", "A2", "A3", "A5")}
        winner = stars["A2"]  # A3 is a comparator (design-review amendment 1)
        c_star = best([stars["A1"], stars["A3"], stars["A5"]], "macro")
        tries = {study.endpoints[c["run_id"]]["splits"]["h_select_constrained"]["trie_sha256"]
                 for c in (winner, c_star)}
        i_star = best([stars[a] for a in ("A1", "A2", "A3")], "macro")
        return {"pass": len(tries) == 1, "winner": winner["run_id"],
                "winner_arm": study.endpoints[winner["run_id"]]["arm"],
                "c_star": c_star["run_id"], "i_star": i_star["run_id"]}

    def gb0():
        predicted = evidence["p0_prime"]["predicted"]
        floor = 375 / len(evidence["s_b_ids"])
        chosen = next((g for g in (8, 16) if predicted[str(g)]["acceptance"] >= floor), None)
        return {"pass": chosen is not None, "chosen_group": chosen}

    def g3a():
        _, rl = study.select_rl_star()
        _, sft, qualifying = study.select_sft_e()
        return {"pass": True, "rl_star": rl["run_id"], "sft_e_star": sft["run_id"],
                "sft_t_lrs": qualifying}

    order = [
        ("PI", "pi" in evidence, lambda: {"pass": True}),
        ("GA", bool(arms & {"A1", "A2", "A3", "A4", "A5"}), ga),
        ("GB0", "p0_prime" in evidence, gb0),
        ("G2", any(e["arm"] == "RL" and e["updates"] == 100 for e in evidence["endpoints"]),
         lambda: {"pass": True, "continue_to_300": [
             c["run_id"] for c in study.select_rl_continuers()[1]]}),
        ("G3a", has_h_gate("SFT-E"), g3a),
        ("G3b", has_h_gate("SFT-T"), lambda: {"pass": True}),
        ("G4", False, None),
    ]
    gates, failed, waiting = {}, None, None
    for name, present, run in order:
        if failed or waiting:
            gates[name] = {"status": "not_evaluated"}
        elif not present:
            gates[name], waiting = {"status": "awaiting_evidence"}, name
        else:
            result = run()
            if name == FAIL_AT:
                result["pass"] = False
            gates[name] = {"status": "evaluated", **result}
            failed = None if result["pass"] else name
    return {"gates": gates, "failed_at": failed, "awaiting": waiting}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.evidence.read_text()))
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
'''

# A fake uwrun: each "host" is a directory here; a run copies --sync into ROOT/NAME/work and
# runs the bash command there detached, with uwrun's UWRUN_THREADS. Faults are injected per
# run name (first instance only) through FAKE_UW_FAULTS. fetch copies like rsync: a SUBPATH
# with a trailing slash copies its content, without one the directory itself.
FAKE_UWRUN = '''#!{python}
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ["FAKE_UW_ROOT"])
HOSTS = json.loads(os.environ["FAKE_UW_HOSTS"])
FAULTS = json.loads(os.environ.get("FAKE_UW_FAULTS", "{{}}"))


def log(event, **fields):
    with (ROOT / "uwrun-log.jsonl").open("a") as handle:
        handle.write(json.dumps({{"event": event, "time": time.time(), **fields}}) + "\\n")


def meta(name):
    return json.loads((ROOT / name / "meta.json").read_text())


def alive(name):
    try:
        os.kill(int((ROOT / name / "pid").read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def running_on(host):
    return sum(1 for d in ROOT.iterdir() if (d / "meta.json").is_file()
               and meta(d.name)["host"] == host and not (d / "rc").exists())


def fault(name):
    for key, kind in FAULTS.items():
        marker = ROOT / f"fault-{{key}}"
        if key in name and not marker.exists():
            marker.write_text(kind)
            return kind
    return None


def known(name):
    if (ROOT / name / "meta.json").is_file():
        return True
    print(f"uwrun: no run {{name}}", file=sys.stderr)  # real uwrun: die, exit 2
    return False


def main(argv):
    command = argv[0]
    if command in ("fetch", "stop", "rm") and not known(argv[1]):
        return 2
    if command == "ssh":
        host = argv[1]
        if HOSTS.get(host) is None:
            return 255
        if "uwprobe" not in argv[2]:
            # Any other remote command (the useful-mixed mirror): run it with ~/runs/NAME
            # meaning this fake host's run directory.
            log("ssh", host=host)
            done = subprocess.run(["bash", "-c", argv[2].replace("~/runs/", str(ROOT) + "/")],
                                  capture_output=True, text=True)
            sys.stdout.write(done.stdout)
            return done.returncode
        load, nproc, avail, uwruns, local, ready, *rest = HOSTS[host]
        avx2 = rest[0] if rest else 1
        vram = f" {{rest[1]}}" if len(rest) > 1 else ""
        busy = uwruns + running_on(host)
        print(f"uwprobe {{load}} {{nproc}} {{avail}} {{busy}} {{local}} {{ready}} {{avx2}}{{vram}}")
        return 0
    if command == "run":
        options, rest = argv[1:argv.index("--")], argv[argv.index("--") + 1:]
        values = dict(zip(options[::2], options[1::2]))
        name, host = values["--name"], values["--host"]
        run = ROOT / name
        run.mkdir()
        kind = fault(name)
        if kind == "fail_launch":
            return 2
        if kind == "slow_launch":
            time.sleep(2.5)  # a long rsync: the launcher must charge it while it lasts
        shutil.copytree(values["--sync"], run / "work")
        if kind == "tamper_pins":  # pins.json changed on the host after the launcher hashed it
            (run / "work" / "pins.json").write_text("{{}}")
        if kind == "tamper_repo":  # a pinned bundle file changed on the host
            with (run / "work" / "repo" / "scripts" / "freshpool_launch"
                  / "phase_worker.py").open("a") as handle:
                handle.write("\\n# changed on the host\\n")
        # The host's provisioned packages (here: the fake entry and torch).
        shutil.copytree(os.environ["FAKE_UW_PROVISION"], run / "work" / "repo" / "src",
                        dirs_exist_ok=True)
        (run / "meta.json").write_text(json.dumps({{
            "host": host, "threads": values["--threads"], "timeout": values["--timeout"],
            "min_local": values["--min-local"], "fault": kind}}))
        (run / "cmd.json").write_text(json.dumps(rest))
        child = subprocess.Popen([sys.executable, __file__, "_exec", name],
                                 start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (run / "pid").write_text(str(child.pid))
        log("run", name=name, host=host)
        print(name)
        return 0
    if command == "_exec":
        name = argv[1]
        run, info = ROOT / name, meta(argv[1])
        # Overlap barrier: the first run waits (up to 60 s) until a second one exists, so the
        # concurrency the test asserts is observed however fast the fake stages are.
        deadline = time.monotonic() + 60
        while (sum(1 for d in ROOT.iterdir() if (d / "meta.json").is_file()) < 2
               and time.monotonic() < deadline):
            time.sleep(0.05)
        if info["fault"] == "slow":
            time.sleep(2.5)
        env = {{**os.environ, "UWRUN_THREADS": info["threads"], "UWRUN_NAME": name}}
        with (run / "run.log").open("w") as out:
            child = subprocess.Popen(json.loads((run / "cmd.json").read_text()),
                                     cwd=run / "work", env=env, stdout=out, stderr=out)
            try:
                rc = child.wait(timeout=float(info["timeout"]))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                (run / "timedout").write_text("")
                rc = 124
        log("exit", name=name, rc=rc)
        if info["fault"] == "lost":
            return 0  # the session vanished without an rc
        if info["fault"] == "timedout":
            (run / "timedout").write_text("")
        if info["fault"] == "stopped":
            (run / "stopped").write_text("")  # someone else ran uwrun stop on it
        if info["fault"] == "no_fragment":
            (run / "work" / "out" / "stage" / "fragment.json").unlink()
        if info["fault"] == "no_hardware":
            (run / "work" / "out" / "hardware.json").unlink()
        (run / "rc").write_text(str(rc))
        return 0
    if command == "status":
        name = argv[1]
        if not (ROOT / name / "meta.json").is_file():
            print(f"{{name}} - UNKNOWN")
            return 0
        host = meta(name)["host"]
        if (ROOT / name / "rc").exists():
            flags = [f for f in ("timedout", "stopped") if (ROOT / name / f).exists()]
            print(" ".join([name, host, "DONE", "rc=" + (ROOT / name / "rc").read_text(),
                            *flags]))
        elif alive(name):
            print(f"{{name}} {{host}} RUNNING since=now")
        else:
            print(f"{{name}} {{host}} LOST")
        return 0
    if command == "fetch":
        name, sub, destination = argv[1], argv[2], Path(argv[3])
        marker = ROOT / name / "fetch-failed-once"
        if meta(name)["fault"] == "fetch_fail" and not marker.exists():
            marker.write_text("")
            return 2
        if destination.exists() and any(destination.iterdir()):
            return 2
        source = ROOT / name / sub.rstrip("/")
        target = destination if sub.endswith("/") else destination / source.name
        shutil.copytree(source, target, dirs_exist_ok=True)
        print(destination)
        return 0
    if command == "stop":
        name = argv[1]
        try:
            os.killpg(int((ROOT / name / "pid").read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass
        (ROOT / name / "stopped").write_text("")
        if not (ROOT / name / "rc").exists():
            (ROOT / name / "rc").write_text("143")
        log("stop", name=name)
        print("stopped (session empty)")
        return 0
    if command == "rm":
        shutil.rmtree(ROOT / argv[1] / "work", ignore_errors=True)
        log("rm", name=argv[1])
        print("removed")
        return 0
    return 2


sys.exit(main(sys.argv[1:]))
'''


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


COMMON = ["--stage-id", "{stage_id}", "--fragment", "{fragment}", "--arm-dir", "{arm_dir}",
          "--device", "{device}"]
RL_TAIL = ["--lr", "{lr}", "--cap", "{arm_cap}", "--num-generations", "{group}"]


class Fixture:
    def __init__(self, tmp, monkeypatch, *, uw=False, fail_at=None, evaluator=None,
                 hosts=None, faults=None, uw_all=None, **overrides):
        # uw_all: None (Stage A only on UW when uw), "cpu" or "cuda" (every stage on UW).
        uw = uw or uw_all is not None
        self.tmp, self.overrides, self.uw, self.uw_all = tmp, overrides, uw, uw_all
        pkg = tmp / "pkg"
        pkg.mkdir()
        (pkg / "fake_signal_entry.py").write_text(FAKE_ENTRY)
        (pkg / "torch.py").write_text(FAKE_TORCH)
        self.entry = pkg / "fake_signal_entry.py"
        self.calls = tmp / "calls.jsonl"
        self.scenario_path = tmp / "scenario.json"
        self.scenario = scenario()
        self.environment = {"PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "ACCELERATE_TORCH_DEVICE": uw_all or "mps",
                            "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(pkg),
                            "FAKE_CALLS": str(self.calls),
                            "FAKE_SCENARIO": str(self.scenario_path)}
        if overrides.get("init_in_run"):
            self.environment["FAKE_INIT_IN_RUN"] = "1"
        for name, value in self.environment.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("VISION_RL_TORCH_THREADS", raising=False)
        if evaluator is None:
            evaluator = tmp / "contract_evaluator.py"
            evaluator.write_text(CONTRACT_EVALUATOR.replace("__FAIL_AT__", repr(fail_at)))
        self.evaluator = evaluator
        self.sentinel = f"signal-limit-owner-sentinel-{os.getpid()}-{uuid.uuid4().hex}"
        self.g0_files()
        if uw:
            self.uw_files(monkeypatch, hosts, faults)
        self.write()

    def g0_files(self):
        manifests = self.tmp / "manifests"
        images = manifests / "img"
        images.mkdir(parents=True)
        rows = {
            "s_a.jsonl": [(i, LABELS[n % 2]) for n, i in enumerate(S_A)],
            "s_b_scan_seed42.jsonl": [(i, LABELS[n % 2]) for n, i in enumerate(S_B)],
            "h_select.jsonl": H_ROWS["h_select"], "h_gate.jsonl": H_ROWS["h_gate"]}
        jpg = sha_text("jpg")
        listing = ""
        for name, items in rows.items():
            for i, _ in items:
                (images / f"{i}.jpg").write_text("jpg")
                listing += f"{name}\t{i}\t{jpg}\n"
            jsonl(manifests / name, [{"id": i, "split": "train", "image": f"img/{i}.jpg",
                                      "label": label} for i, label in items])
        self.g0 = {"manifest_dir": str(manifests),
                   "files_sha256": {name: sha_file(manifests / name) for name in rows},
                   "images_sha256": sha_text(listing)}
        base = self.tmp / "evidence-base.json"
        base.write_text(json.dumps({"labels": LABELS, "rows": {
            **{split: [{"id": i, "label": label} for i, label in items]
               for split, items in H_ROWS.items()},
            **self.overrides.get("base_extra_rows", {})}}))
        self.base = {"path": str(base), "sha256": sha_file(base)}
        self.init_paths = {}
        for role, text in (("i", "I adapter"), ("w", "W adapter")):
            # init_in_run: as the real W and I, a final_adapter inside its trainer run, whose
            # local_training.json the trainer reads (fake entry rule FAKE_INIT_IN_RUN).
            folder = (self.tmp / f"run-{role}" / "final_adapter"
                      if self.overrides.get("init_in_run") else self.tmp / f"init-{role}")
            folder.mkdir(parents=True)
            (folder / "adapter_model.safetensors").write_text(text)
            if self.overrides.get("init_in_run"):
                (folder.parent / "local_training.json").write_text('{"record": "' + text + '"}')
            self.init_paths[role] = str(folder)
        trie = self.tmp / "label-trie.json"
        trie.write_text(TRIE_TEXT)
        self.trie = {"path": str(trie), "sha256": TRIE_SHA}
        resources = {}
        for name, text, digest in (("confusion_map", MAP_TEXT, MAP_SHA),
                                   ("taxonomy", TAXONOMY_TEXT, TAXONOMY_SHA)):
            path = self.tmp / f"{name}.json"
            path.write_text(text)
            resources[name] = {"path": str(path), "sha256": digest}
        self.resources = resources

    def uw_files(self, monkeypatch, hosts, faults):
        root = self.tmp / "uw-root"
        root.mkdir()
        provision = self.tmp / "provision"
        provision.mkdir()
        shutil.copy(self.entry, provision / "fake_signal_entry.py")
        (provision / "torch.py").write_text(FAKE_TORCH)
        uwrun = self.tmp / "uwrun"
        uwrun.write_text(FAKE_UWRUN.format(python=sys.executable))
        uwrun.chmod(uwrun.stat().st_mode | stat.S_IXUSR)
        # uw.python and uw.model_directory must be host-local /var/tmp paths.
        self.var = Path("/var/tmp") / f"rle-signal-limit-test-{uuid.uuid4().hex[:12]}"
        (self.var / "model").mkdir(parents=True)
        python = self.var / "python"
        python.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
        python.chmod(python.stat().st_mode | stat.S_IXUSR)
        hosts = hosts if hosts is not None else {
            "hosta": [0.1, 28, 100, 0, 100, 1], "hostb": [0.5, 24, 100, 0, 500, 1]}
        monkeypatch.setenv("FAKE_UW_ROOT", str(root))
        monkeypatch.setenv("FAKE_UW_HOSTS", json.dumps(hosts))
        monkeypatch.setenv("FAKE_UW_FAULTS", json.dumps(faults or {}))
        monkeypatch.setenv("FAKE_UW_PROVISION", str(provision))
        self.uw_root, self.uwrun = root, uwrun
        self.uw_block = {
            "uwrun": str(uwrun), "hosts": sorted(hosts), "per_host": 1, "max_concurrent": 4,
            "threads": 6, "python": str(python), "model_directory": str(self.var / "model"),
            "min_local_gb": 5, "min_free_ram_gb": 16, "poll_seconds": 0.05,
            "unreachable_seconds": 2,
            "forecast_seconds": {"assay_h_select": 600, "stage_a_sft": 600,
                                 "stage_a_egsft": 600, "stage_a_hn_sft": 600,
                                 "stage_a_hls_sft": 600, "stage_a_cce_sft": 600}}
        if self.uw_all is not None:
            del self.uw_block["forecast_seconds"]
            self.uw_block.update(device=self.uw_all, mixed_poll_seconds=0.05)
            if self.uw_all == "cuda":
                self.uw_block.update(min_free_vram_mib=10000)

    def cleanup(self):
        if self.uw:
            shutil.rmtree(self.var, ignore_errors=True)

    def templates(self):
        entry_sha = sha_file(self.entry)

        def template(kind, argv):
            return {"kind": kind, "entry": "fake_signal_entry:main", "entry_sha256": entry_sha,
                    "argv": argv}

        wrapper = ["--output", "{arm_dir}/trainer", *COMMON]
        rl_wrapper = [*wrapper, "--scan-manifest", "{scan_manifest}"]
        found = {
            "stage0": template("train", COMMON + ["--init", "{init_adapter}", "--",
                                                  "--num-generations", "8"]),
            "rl_screen": template("train", rl_wrapper + ["--", *RL_TAIL,
                                                         "--stop-after-steps", "100"]),
            "rl_final": template("train", wrapper[:2] + [
                "--continue-output-from", "{resume_checkpoint}", *COMMON,
                "--scan-manifest", "{scan_manifest}", "--", *RL_TAIL]),
            "sft_e": template("train", wrapper + [
                "--", "--lr", "{lr}", "--cap", "{active_cap}", "--order", "sequential",
                "--cycle", "--train", "{accepted_manifest}", "--stop-after-steps", "300",
                "--max-steps", "600"]),
            "sft_t": template("train", COMMON + [
                "--", "--lr", "{lr}", "--train", "{scanned_manifest}", "--continue-from",
                "{parent_checkpoint}", "--stop-after-active-seconds", "{t_rl}"]),
            "assay": template("assay", COMMON + [
                "--adapter", "{adapter}", "--split", "{split}", "--manifest",
                "{split_manifest}", "--label-trie", "{label_trie}", "--label-trie-sha256",
                "{label_trie_sha256}", "--training-group", "{training_group}"]),
            "p0_i": template("assay", COMMON + [
                "--scan", "{scan_manifest}", "--adapter", "{init_adapter}",
                "--predict-groups", "4,8,16", "--num-generations", "4"]),
            "p0_prime": template("assay", COMMON + [
                "--scan", "{scan_manifest}", "--adapter", "{init_adapter}",
                "--choose-groups", "8,16", "--a-floor", "{a_floor}"]),
        }
        for arm in ("a1", "a2", "a3", "a4", "a5"):
            found[f"stage_a_{arm}"] = template("train", COMMON + [
                "--arm", "{arm_label}", "--init", "{init_adapter}", "--train",
                "{manifest_s_a}", "--", "--lr", "{lr}", "--max-steps", "600"])
        for arm in ("a2",):
            found[f"lim_t_{arm}"] = template("train", COMMON + [
                "--arm", "{arm_label}", "--", "--lr", "{lr}", "--init", "{init_adapter}",
                "--train", "{scanned_manifest}", "--stop-after-active-seconds", "{t_rl}",
                "--egsft-band-group", "{group}"])
        return found

    def write(self):
        self.scenario_path.write_text(json.dumps(self.scenario))
        o = self.overrides
        pins = {str(self.evaluator): sha_file(self.evaluator), str(self.entry): sha_file(
            self.entry), os.path.realpath(sys.executable): PYTHON_SHA}
        for name in ("phase_worker.py", "launch.py", "stage_table.py", "signal_limit_launch.py",
                     "uw_stage_a.py"):
            pins[str(PACKAGE / name)] = sha_file(PACKAGE / name)
        if self.uw:
            pins[str(self.uwrun)] = sha_file(self.uwrun)
        for name in o.get("unpin", ()):
            pins.pop(str(self.uwrun) if name == "uwrun" else str(PACKAGE / name))
        templates = self.templates()
        for name, patch in o.get("template_patch", {}).items():
            templates[name] = {**templates[name], **patch}
        plan = {
            "schema": "signal-limit-launch-plan-v1",
            "python": sys.executable, "python_resolved": os.path.realpath(sys.executable),
            "import_root": None, "state_directory": str(self.tmp / "state"),
            "evaluator": str(self.evaluator), "file_sha256": pins,
            "environment": self.environment,
            "study": {"path": str(STUDY), "sha256": o.get("study_sha", sha_file(STUDY))},
            "updates": o.get("updates", {"stage_a": 600, "rl_screen": 100, "final": 300}),
            "quiet_file": str(self.tmp / "QUIET"),
            "limits": {"minimum_free_bytes": 5 * GiB, "rss_limit_bytes": 64 * GiB,
                       "survivor_grace_seconds": 3, "poll_seconds": 0.05,
                       "stage_headroom_bytes": 0},
            "forecast_seconds": {"p0": 600, "stage0": 600, "assay_h_select": 600,
                                 "assay_h_gate": 600, "sft_e": 600, "stage_a_sft": 600,
                                 "stage_a_egsft": 600, "stage_a_hn_sft": 600,
                                 "stage_a_hls_sft": 600, "stage_a_cce_sft": 600},
            "templates": templates,
            "init_endpoints": {
                "I": {"adapter_sha256": o.get("i_sha", I_SHA),
                      "adapter_path": self.init_paths["i"]},
                "W": {"adapter_sha256": W_SHA, "adapter_path": self.init_paths["w"]}},
            "label_trie": self.trie, "evidence_base": self.base,
            **{name: {**entry, "sha256": o.get("resource_sha", {}).get(name, entry["sha256"])}
               for name, entry in self.resources.items()},
            "data_spec": "data/fgvc-aircraft-2013b/spec.json",
            "model_directory": str(self.tmp / "model"),
            "owner_patterns": [self.sentinel], "g0": self.g0,
            "stage_a_execution": "uwaterloo" if self.uw else "local",
        }
        if self.uw_all is not None:
            del plan["stage_a_execution"]
            plan["execution"] = "uwaterloo"
        if self.uw:
            plan["uw"] = {**self.uw_block, **o.get("uw_patch", {})}
        plan.update(o.get("plan_patch", {}))
        self.plan_path = self.tmp / "plan.json"
        self.plan_path.write_text(json.dumps(plan, indent=1))
        self.plan_sha = sha_file(self.plan_path)

    @property
    def argv(self):
        return ["--plan", str(self.plan_path), "--expected-plan-sha256", self.plan_sha]

    def state(self):
        return json.loads((self.tmp / "state" / "state.json").read_text())

    def journal(self):
        path = self.tmp / "state" / "journal.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def stages_called(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def uw_log(self):
        path = self.uw_root / "uwrun-log.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def make(tmp_path, monkeypatch):
    made = []

    def build(**options):
        fixture = Fixture(tmp_path, monkeypatch, **options)
        made.append(fixture)
        return fixture

    yield build
    for fixture in made:
        fixture.cleanup()


def first_order(calls):
    seen = []
    for call in calls:
        if call["stage"] not in seen:
            seen.append(call["stage"])
    return seen


def argv_value(argv, flag):
    return argv[argv.index(flag) + 1]


def final_evidence(fixture):
    gates = fixture.tmp / "state" / "gates"
    (path,) = sorted(gates.glob("final-evidence-*.json"))
    return json.loads(path.read_text())


# ---------------------------------------------------------------- the local sequence


def test_local_sequence_runs_every_stage_in_order_with_pinned_inits(make):
    fixture = make()
    assert signal_limit.main(fixture.argv) == 0
    state = fixture.state()
    assert state["status"] == "completed"
    assert first_order(fixture.stages_called()) == EXPECTED_ORDER
    assert state["selections"] == EXPECTED_SELECTIONS
    assert state["pinned"]["I*"]["adapter_sha256"] == sha_text(I_STAR)
    assert state["c_seconds_per_accepted_update"] == {"8": 10.0, "16": 20.0}
    by_stage = {c["stage"]: c["argv"] for c in fixture.stages_called()}
    # Init endpoint per stage: I for P0-I, W for Stage 0 and Stage A, I* for Stage B.
    assert argv_value(by_stage["P0-I"], "--adapter") == str(fixture.tmp / "init-i")
    assert argv_value(by_stage["Stage-0"], "--init") == str(fixture.tmp / "init-w")
    assert argv_value(by_stage["stage-a-a3-1e-04-s42"], "--init") == str(fixture.tmp / "init-w")
    i_star_adapter = argv_value(by_stage["P0-prime"], "--adapter")
    assert i_star_adapter.endswith(f"arms/{I_STAR}/instance-1/final_adapter")
    assert argv_value(by_stage["lim-t-5e-05-s42"], "--init") == i_star_adapter
    # A4 runs at A1*'s LR; LIM-T runs the Gate-A winner's (A2) template at A2*'s LR.
    assert argv_value(by_stage["stage-a-a4-5e-05-s42"], "--lr") == "5e-05"
    assert argv_value(by_stage["lim-t-5e-05-s42"], "--arm") == "LIM-T"
    assert argv_value(by_stage["P0-prime"], "--a-floor") == repr(A_FLOOR)
    # RL runs at GB0's G with the cap from c(G8) x 300 = 3000 s (1.25 x = 3750).
    assert argv_value(by_stage["rl-1e-05-s42-u100"], "--num-generations") == "8"
    assert argv_value(by_stage["rl-1e-05-s42-u100"], "--cap") == "3750.0"
    # Every H assay passes the trie and the endpoint's training G (RL 8, others 0).
    assert argv_value(by_stage["rl-1e-04-s42-u300-h_gate"], "--training-group") == "8"
    assert argv_value(by_stage["sft-e-5e-05-s42-h_gate"], "--training-group") == "0"
    assert argv_value(by_stage[f"{I_STAR}-h_select"], "--label-trie-sha256") == TRIE_SHA
    evidence = final_evidence(fixture)
    endpoints = {e["run_id"]: e for e in evidence["endpoints"]}
    assert endpoints["W"]["adapter_sha256"] == W_SHA
    assert endpoints["stage-a-a1-1e-05-s42"]["init_sha256"] == W_SHA
    assert endpoints["rl-1e-04-s42-u300"]["init_sha256"] == sha_text(I_STAR)
    assert endpoints["sft-t-1e-04-s42"]["init_sha256"] == sha_text("sft-e-1e-04-s42")
    assert endpoints["lim-t-5e-05-s42"]["arm"] == "LIM-T"
    assert endpoints[I_STAR]["splits"]["h_select_constrained"]["trie_sha256"] == TRIE_SHA
    assert endpoints["rl-1e-05-s42-u100"]["group"] == 8
    assert evidence["s_b_ids"] == S_B and evidence["pi"]["group"] == 4
    # H-select: Stage A 13, RL 3 + 2, SFT-E 3, SFT-T 2, LIM-T 1; H-gate: RL*, SFT-E*, I*,
    # SFT-T*, LIM-T.
    assert len(evidence["looks"]) == 24 + 5
    events = [e["event"] for e in fixture.journal()]
    assert events[-1] == "study_development_complete"
    assert events.count("c_g_written") == 1 and events.count("init_endpoint_pinned") == 1


def test_a_gate_failure_stops_before_stage_b(make):
    fixture = make(fail_at="GA")
    assert signal_limit.main(fixture.argv) == 2
    state = fixture.state()
    assert state["status"] == "stopped" and state["stop_reason"] == "gate_failed"
    assert first_order(fixture.stages_called())[-1] == "stage-a-a4-5e-05-s42-h_select"
    assert "I*" not in state["pinned"]


def test_stage_a_arm_trained_off_the_s_a_order_stops(make):
    fixture = make()
    fixture.scenario["stage-a-a3-5e-05-s42"]["payload"]["committed_prompt_ids"] = S_A[::-1]
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "stage_a_exposure_not_s_a_order"


def test_lim_t_with_the_other_objective_stops(make):
    fixture = make()
    fixture.scenario["lim-t-5e-05-s42"]["payload"]["mode"] = "hn_sft"
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "lim_t_mode_is_not_the_gate_a_winner_objective"


@pytest.mark.parametrize(("stage", "change", "reason"), [
    # A2 run as group-std EG-SFT without the band: not the pre-registered limit.
    ("stage-a-a2-1e-04-s42", {"egsft_scale": "group", "egsft_band_group": None},
     "stage_a_a2_not_the_screened_no_std_limit"),
    ("stage-a-a2-1e-05-s42", {"egsft_band_group": 16},
     "stage_a_a2_not_the_screened_no_std_limit"),
    # Group-std scale at the right band: the scale half of the check fires alone.
    ("stage-a-a2-5e-05-s42", {"egsft_scale": "group"},
     "stage_a_a2_not_the_screened_no_std_limit"),
    # A4 trained against another taxonomy.
    ("stage-a-a4-5e-05-s42", {"taxonomy_sha256": "f" * 64},
     "stage_a_label_resource_differs_from_plan_pin"),
    # A5 trained against another trie, A3 against another map.
    ("stage-a-a5-5e-05-s42", {"label_trie_sha256": "f" * 64},
     "stage_a_label_resource_differs_from_plan_pin"),
    ("stage-a-a3-1e-05-s42", {"confusion_map_sha256": None},
     "stage_a_label_resource_differs_from_plan_pin"),
    # LIM-T at B_16 although GB0 chose G8; LIM-T with group-std scale at the chosen band.
    ("lim-t-5e-05-s42", {"egsft_band_group": 16}, "lim_t_not_the_limit_at_the_chosen_band"),
    ("lim-t-5e-05-s42", {"egsft_scale": "group"}, "lim_t_not_the_limit_at_the_chosen_band"),
])
def test_objective_and_resource_identity_is_checked_on_the_fragment(make, stage, change, reason):
    fixture = make()
    fixture.scenario[stage]["payload"].update(change)
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == reason


def choose_g16(fixture, lim_t_band):
    """GB0 picks G16 (P0' predicts acceptance 0 at G8, 1 at G16; its rollouts are G16):
    every RL fragment records G16 and LIM-T records the band ``lim_t_band``."""
    i_star_sha = sha_text(I_STAR)
    fixture.scenario["P0-prime"] = p0_document(i_star_sha, 16, predicted={
        "8": {"acceptance": 0.0, "mean_u": 0.0, "mean_u_accepted": None},
        "16": {"acceptance": 1.0, "mean_u": 1 - 2 * 0.5**16,
               "mean_u_accepted": 1 - 2 * 0.5**16}},
        chosen_group=16, choose_groups=[8, 16], a_floor=A_FLOOR)
    for stage, entry in fixture.scenario.items():
        if stage.startswith("rl-") and "group" in entry["payload"]:
            entry["payload"]["group"] = 16
    fixture.scenario["lim-t-5e-05-s42"]["payload"]["egsft_band_group"] = lim_t_band
    fixture.write()


def test_g16_branch_runs_rl_and_lim_t_at_the_chosen_g(make):
    # The G16 branch amendment 1 added: GB0's G, not A2's fixed B_8, is LIM-T's band.
    fixture = make()
    choose_g16(fixture, 16)
    assert signal_limit.main(fixture.argv) == 0, fixture.journal()[-1]
    state = fixture.state()
    assert state["status"] == "completed"
    assert state["selections"] == {**EXPECTED_SELECTIONS, "group": 16}
    by_stage = {c["stage"]: c["argv"] for c in fixture.stages_called()}
    assert argv_value(by_stage["rl-1e-05-s42-u100"], "--num-generations") == "16"
    assert argv_value(by_stage["lim-t-5e-05-s42"], "--egsft-band-group") == "16"
    # c(G16) x 300 = 6000 s, cap 1.25 x = 7500.
    assert argv_value(by_stage["rl-1e-05-s42-u100"], "--cap") == "7500.0"


def test_g16_branch_refuses_lim_t_at_a2s_band(make):
    fixture = make()
    choose_g16(fixture, 8)  # A2's Stage-A band, not GB0's G16
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "lim_t_not_the_limit_at_the_chosen_band"


def test_a_stopped_batch_keeps_the_looks_it_already_took(make):
    # The 4th H-select assay of the first Stage A batch stops the study: the three that
    # completed before it (A1 at each LR) are on the look ledger all the same.
    fixture = make()
    del fixture.scenario["stage-a-a2-1e-05-s42-h_select"]["payload"]["constrained_greedy"]
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    state = fixture.state()
    assert state["stop_reason"] == "assay_constrained_greedy_missing_or_other_trie"
    assert [look["stage"] for look in state["looks"]] == [
        f"stage-a-a1-{tag}-s42-h_select" for tag in ("1e-05", "5e-05", "1e-04")]


def test_rl_profile_must_be_dr_grpo_beta_0_at_the_chosen_g(make):
    fixture = make()
    fixture.scenario["rl-1e-05-s42-u100"]["payload"]["group"] = 16
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "rl_profile_differs"


def test_missing_c_g_stops_at_stage_0(make):
    fixture = make()
    del fixture.scenario["Stage-0"]["payload"]["c_seconds_per_accepted_update"]["16"]
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "stage0_c_g_missing"
    assert first_order(fixture.stages_called()) == ["P0-I", "Stage-0"]


FEW_C_STEPS = "stage0_c_g_not_measured_over_enough_steps"
STUDY_SIZES = "stage0_not_at_study_e0_steps_or_probe_size"


@pytest.mark.parametrize(("change", "reason"), [
    (lambda p: p.update(e0_groups=[8]), "stage0_e0_not_at_every_g"),
    # One flat checks dict claimed for both G: nothing ties it to either.
    (lambda p: p.update(checks=dict.fromkeys(signal_limit.STAGE0_CHECKS, True)),
     "stage0_e0_checks_not_keyed_by_g"),
    (lambda p: p["checks"].pop("16"), "stage0_e0_checks_not_keyed_by_g"),
    (lambda p: p["checks"]["16"].update(resume_1_plus_3_identical=False), "e0_failed"),
    (lambda p: p["checks"]["8"].pop("no_reference_forward_at_beta_0"), "e0_failed"),
    (lambda p: p["c_accepted_steps"].update({"16": 7}), FEW_C_STEPS),  # 7 < 8
    (lambda p: p.pop("c_accepted_steps"), FEW_C_STEPS),
    # A smoke's sizes, recorded by stage0_gate.py, never pass the study's Stage 0.
    (lambda p: p.update(e0_steps=2), STUDY_SIZES),
    (lambda p: p.pop("e0_steps"), STUDY_SIZES),
    (lambda p: p.update(e0_steps=4.0), STUDY_SIZES),
    (lambda p: p.update(probe_accepted=7), STUDY_SIZES),  # 7 < 8
    (lambda p: p.pop("probe_accepted"), STUDY_SIZES),
])
def test_stage_0_must_measure_c_g_and_prove_e0_at_both_g(make, change, reason):
    fixture = make()
    change(fixture.scenario["Stage-0"]["payload"])
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == reason
    assert first_order(fixture.stages_called()) == ["P0-I", "Stage-0"]


def test_assay_without_constrained_greedy_stops(make):
    fixture = make()
    del fixture.scenario["stage-a-a1-1e-05-s42-h_select"]["payload"]["constrained_greedy"]
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "assay_constrained_greedy_missing_or_other_trie"


@pytest.mark.skipif(not REAL_EVALUATOR.is_file(),
                    reason="scripts/evaluate_signal_limit_gates.py is not in this tree")
def test_local_sequence_passes_the_real_gates_evaluator(make):
    fixture = make(evaluator=REAL_EVALUATOR)
    code = signal_limit.main(fixture.argv)
    gates = [e for e in fixture.journal() if e["event"] == "gate_finished"]
    assert code == 0, gates[-1:]
    state = fixture.state()
    assert state["status"] == "completed"
    assert state["selections"] == EXPECTED_SELECTIONS
    assert {e["gate"]: e["verdict"] for e in fixture.journal()
            if e["event"] == "gate_finished"} == dict.fromkeys(
                ("PI", "GA", "GB0", "G2", "G3a", "G3b"), "pass")


# ---------------------------------------------------------------- plan refusals


@pytest.mark.parametrize("options, reason", [
    ({"study_sha": "0" * 64}, "study_missing_or_changed"),
    ({"i_sha": "1" * 64}, "init_endpoint_differs_from_study"),
    ({"updates": {"stage_a": 6, "rl_screen": 1, "final": 3}},
     "plan_update_counts_differ_from_study"),
    ({"updates": {"stage_a": 600, "rl_screen": 300, "final": 300}},
     "plan_update_counts_invalid"),
    ({"plan_patch": {"w_endpoint": {"adapter_sha256": W_SHA, "adapter_path": "/w"}}},
     "init_endpoint_must_be_pinned_per_stage"),
    ({"template_patch": {"assay": {"argv": COMMON}}},
     "assay_template_without_constrained_greedy"),
    ({"template_patch": {"stage_a_a2": {"argv": COMMON + ["--", "--max-steps", "599"]}}},
     "stage_a_template_max_steps_differs"),
    ({"template_patch": {"p0_prime": {"argv": COMMON + ["--choose-groups", "8,16",
                                                        "--a-floor", "0.3"]}}},
     "p0_prime_template_differs"),
    ({"unpin": ("stage_table.py",)}, "plan_does_not_pin"),
    ({"plan_patch": {"stage_a_execution": "cloud"}}, "stage_a_execution_unknown"),
    ({"plan_patch": {"uw": {}}}, "uw_block_without_uwaterloo_execution"),
    ({"plan_patch": {"taxonomy": {"path": "/nonexistent", "sha256": TAXONOMY_SHA}}},
     "taxonomy_missing_or_changed"),
    ({"plan_patch": {"confusion_map": {"path": "/nonexistent", "sha256": MAP_SHA}}},
     "confusion_map_missing_or_changed"),
    # The file exists, its bytes differ from the pin.
    ({"resource_sha": {"taxonomy": "0" * 64}}, "taxonomy_missing_or_changed"),
    ({"resource_sha": {"confusion_map": "0" * 64}}, "confusion_map_missing_or_changed"),
    # An evidence base carrying VAL or sealed TEST rows never reaches the evidence.
    ({"base_extra_rows": {"test": [{"id": "t0", "label": "A"}]}},
     "evidence_base_rows_not_h_splits_only"),
    ({"base_extra_rows": {"val": []}}, "evidence_base_rows_not_h_splits_only"),
])
def test_plan_refusals_before_any_state(make, capsys, options, reason):
    fixture = make(**options)
    assert signal_limit.main(fixture.argv) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == reason
    assert not (fixture.tmp / "state").exists()


def test_verify_only_reports_a_floor_and_counts(make, capsys):
    fixture = make()
    assert signal_limit.main([*fixture.argv, "--verify-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["a_floor"] == A_FLOOR and report["s_b_size"] == 700
    assert report["updates"] == {"stage_a": 600, "rl_screen": 100, "final": 300}
    assert report["model_calls"] == 0


# ---------------------------------------------------------------- the UWaterloo path


def uw_stage_a_stages():
    return [*STAGE_A_RUNS, *(f"{r}-h_select" for r in STAGE_A_RUNS), "stage-a-a4-5e-05-s42",
            "stage-a-a4-5e-05-s42-h_select"]


def test_uw_stage_a_runs_on_the_hosts_and_the_rest_locally(make):
    fixture = make(uw=True)
    assert signal_limit.main(fixture.argv) == 0, fixture.journal()[-1]
    state = fixture.state()
    assert state["status"] == "completed"
    assert state["selections"] == EXPECTED_SELECTIONS  # same evidence, same selections
    calls = fixture.stages_called()
    remote = {c["stage"] for c in calls if c["cwd"].startswith(str(fixture.uw_root))}
    assert remote == set(uw_stage_a_stages())
    for call in calls:
        if call["stage"] in remote:
            # cpu, with uwrun's --threads as the trainer's torch threads.
            assert (call["device"], call["uwrun_threads"], call["torch_threads"]) == (
                "cpu", "6", "6")
            assert argv_value(call["argv"], "--device") == "cpu"
            assert not any(part.startswith(str(fixture.tmp)) for part in call["argv"])
        else:
            assert (call["device"], call["torch_threads"]) == ("mps", None)
    journal = fixture.journal()
    hardware = {e["stage"]: e for e in journal if e["event"] == "uw_stage_hardware"}
    assert set(hardware) == set(uw_stage_a_stages())
    for entry in hardware.values():
        assert entry["host"] in ("hosta", "hostb")
        assert entry["hardware"]["pins_failed"] == [] and entry["hardware"]["threads"] == "6"
        # The 5 pinned repository files plus every bundled input file (counted on disk).
        inputs = fixture.tmp / "state" / "uw" / "bundles" / entry["uw_name"] / "inputs"
        assert entry["hardware"]["pins_checked"] == 5 + sum(
            1 for path in inputs.rglob("*") if path.is_file())
        assert entry["hardware"]["pins_checked"] > 5
    started = {e["stage"]: e for e in journal if e["event"] == "stage_started"}
    assert started["stage-a-a1-1e-05-s42"]["executor"] == "uwaterloo"
    assert "executor" not in started["P0-prime"]
    # The fetched Stage A endpoint is what Stage B starts from, by hash.
    assert state["pinned"]["I*"]["adapter_sha256"] == sha_text(I_STAR)
    assert "/remote/arm/final_adapter" in state["pinned"]["I*"]["adapter_path"]
    # per_host 1 on two hosts: never more than two runs at once, never two on one host.
    log, live, peak = fixture.uw_log(), {}, 0
    for event in log:
        if event["event"] == "run":
            live[event["name"]] = event["host"]
        elif event["event"] in ("exit", "stop") and event["name"] in live:
            del live[event["name"]]
        peak = max(peak, len(live))
        assert len(set(live.values())) == len(live)
    assert peak == 2
    assert {e["name"] for e in log if e["event"] == "rm"} == {
        e["uw_name"] for e in hardware.values()}
    assert state["uw_running"] == {}


def test_uw_lost_run_interrupts_and_the_resume_reruns_it_from_scratch(make):
    fixture = make(uw=True, faults={"stage-a-a3-5e-05": "lost"})
    assert signal_limit.main(fixture.argv) == 3
    state = fixture.state()
    assert state["status"] == "running"
    lost = [e for e in fixture.journal() if e["event"] == "uw_run_lost"]
    assert [e["stage"] for e in lost] == ["stage-a-a3-5e-05-s42"]
    assert signal_limit.main([*fixture.argv, "--resume"]) == 0
    state = fixture.state()
    assert state["status"] == "completed" and state["selections"] == EXPECTED_SELECTIONS
    voided = [(v["stage"], v["instance"]) for v in state["void_instances"]]
    assert ("stage-a-a3-5e-05-s42", 1) in voided
    completed = state["completed"]["stage-a-a3-5e-05-s42"]["path"]
    assert completed.endswith("stages/stage-a-a3-5e-05-s42/instance-2/fragment.json")


def test_uw_remote_timeout_is_the_arm_cap(make):
    fixture = make(uw=True, faults={"stage-a-a1-1e-05": "timedout"})
    assert signal_limit.main(fixture.argv) == 2
    state = fixture.state()
    assert state["stop_reason"] == "arm_cap"
    assert state["uw_running"] == {}


def test_uw_fetched_adapter_that_differs_stops(make):
    fixture = make(uw=True)
    fixture.scenario["stage-a-a2-1e-05-s42"]["behaviour"] = {"adapter_text": "other bytes"}
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "uw_fetched_adapter_differs"
    # A stage that stops still has its remote directory removed.
    started = [e for e in fixture.journal() if e["event"] == "stage_started"
               and e["stage"] == "stage-a-a2-1e-05-s42"]
    assert started and started[-1]["uw_name"] in {
        e["name"] for e in fixture.uw_log() if e["event"] == "rm"}


def test_uw_launch_failure_pauses_resumably(make):
    fixture = make(uw=True, faults={"stage-a-a1-1e-05": "fail_launch"})
    assert signal_limit.main(fixture.argv) == 3
    state = fixture.state()
    assert state["status"] == "running"
    assert "uw_launch_failed" in state["next_action"]
    (name,) = state["uw_running"]
    # uwrun never recorded the run: the resume finds nothing to stop and is not wedged.
    assert signal_limit.main([*fixture.argv, "--resume"]) == 0
    state = fixture.state()
    assert state["status"] == "completed" and state["uw_running"] == {}
    assert [e["uw_name"] for e in fixture.journal() if e["event"] == "uw_orphan_absent"] == [
        name]
    assert ("stage-a-a1-1e-05-s42", 1) in [
        (v["stage"], v["instance"]) for v in state["void_instances"]]


def test_uw_fetch_failure_interrupts_and_the_resume_reaps_and_reruns(make):
    fixture = make(uw=True, faults={"stage-a-a5-1e-04": "fetch_fail"})
    assert signal_limit.main(fixture.argv) == 3
    state = fixture.state()
    assert state["status"] == "running"
    (name,) = [n for n, e in state["uw_running"].items() if e["stage"] == "stage-a-a5-1e-04-s42"]
    assert [e["uw_name"] for e in fixture.journal() if e["event"] == "uw_fetch_failed"] == [
        name]
    assert signal_limit.main([*fixture.argv, "--resume"]) == 0
    state = fixture.state()
    assert state["status"] == "completed" and state["uw_running"] == {}
    journal = fixture.journal()
    assert name in {e["uw_name"] for e in journal if e["event"] == "uw_orphan_stopped"}
    assert name in {e["name"] for e in fixture.uw_log() if e["event"] == "rm"}
    assert ("stage-a-a5-1e-04-s42", 1) in [
        (v["stage"], v["instance"]) for v in state["void_instances"]]


@pytest.mark.parametrize(("faults", "patch", "reason"), [
    ({"stage-a-a1-5e-05": "stopped"}, {}, "uw_run_stopped_externally"),
    ({"stage-a-a1-5e-05": "no_fragment"}, {}, "stage_fragment_missing"),
    ({"stage-a-a1-5e-05": "no_hardware"}, {}, "uw_hardware_record_missing_or_pins_failed"),
    ({"stage-a-a1-1e-05": "tamper_pins"}, {}, "stage_failed"),  # rc 95, pins.json
    ({"stage-a-a1-1e-05": "tamper_repo"}, {}, "stage_failed"),  # rc 95, the changed file
])
def test_uw_remote_failures_stop_and_clean_up(make, faults, patch, reason):
    fixture = make(uw=True, faults=faults, uw_patch=patch)
    assert signal_limit.main(fixture.argv) == 2
    state = fixture.state()
    assert state["stop_reason"] == reason
    finished = [e for e in fixture.journal()
                if e["event"] == "stage_finished" and e.get("executor") == "uwaterloo"]
    assert finished
    tamper = {"tamper_pins": ["pins.json"],
              "tamper_repo": ["repo/scripts/freshpool_launch/phase_worker.py"]}.get(
                  next(iter(faults.values()), None))
    if tamper:
        (tampered,) = [e for e in finished if e["stage"] == "stage-a-a1-1e-05-s42"]
        assert tampered["exit_code"] == 95
        hardware = [e for e in fixture.journal() if e["event"] == "uw_stage_hardware"
                    and e["stage"] == "stage-a-a1-1e-05-s42"]
        assert hardware[0]["hardware"]["pins_failed"] == tamper
    removed = {e["name"] for e in fixture.uw_log() if e["event"] == "rm"}
    assert {e["uw_name"] for e in finished} <= removed


def test_uw_arm_wall_time_reaches_state_while_a_stage_runs(make, monkeypatch):
    # The Mac launcher may die mid-stage: the arm clock must already be on disk, or a
    # resume would void the instance at 0 s and refill the arm cap.
    import threading

    uw_stage_a = importlib.import_module("uw_stage_a")
    monkeypatch.setattr(uw_stage_a, "SAVE_SECONDS", 0.2)
    fixture = make(uw=True, faults={"stage-a-a1-1e-05": "slow"})
    seen, done = [], threading.Event()

    def watch():
        path = fixture.tmp / "state" / "state.json"
        while not done.is_set():
            try:
                state = json.loads(path.read_text())
            except (OSError, ValueError):
                state = {}
            running = {e["stage"] for e in (state.get("uw_running") or {}).values()}
            if "stage-a-a1-1e-05-s42" in running:
                seen.append(state["arm_wall_seconds"].get("stage-a-a1-1e-05-s42", 0.0))
            done.wait(0.05)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        assert signal_limit.main(fixture.argv) == 0
    finally:
        done.set()
        watcher.join()
    assert seen and max(seen) >= 1.0  # saved mid-run, not only at launch and finish


def test_uw_arm_wall_time_reaches_state_while_the_upload_runs(make, monkeypatch):
    # A launcher killed during ``uwrun run`` (its rsync can take minutes) must already have
    # charged that time to the arm, or the resume voids the instance at 0 s.
    import threading

    uw_stage_a = importlib.import_module("uw_stage_a")
    monkeypatch.setattr(uw_stage_a, "SAVE_SECONDS", 0.2)
    fixture = make(uw=True, faults={"stage-a-a1-1e-05": "slow_launch"})
    seen, done = [], threading.Event()

    def watch():
        path = fixture.tmp / "state" / "state.json"
        while not done.is_set():
            try:
                state = json.loads(path.read_text())
            except (OSError, ValueError):
                state = {}
            launched = {e["name"] for e in (json.loads(line) for line in (
                (fixture.uw_root / "uwrun-log.jsonl").read_text().splitlines()
                if (fixture.uw_root / "uwrun-log.jsonl").exists() else []))
                if e["event"] == "run"}
            running = {n for n, e in (state.get("uw_running") or {}).items()
                       if e["stage"] == "stage-a-a1-1e-05-s42"}
            if running and not running & launched:  # recorded, but uwrun run not returned
                seen.append(state["arm_wall_seconds"].get("stage-a-a1-1e-05-s42", 0.0))
            done.wait(0.05)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        assert signal_limit.main(fixture.argv) == 0
    finally:
        done.set()
        watcher.join()
    assert seen and max(seen) >= 1.0  # saved during the upload, not only after it


@pytest.mark.parametrize("hosts", [
    {"hosta": [0.1, 28, 100, 0, 100, 0]},  # python / model not provisioned
    {"hosta": [0.1, 28, 100, 0, 100, 1, 0]},  # no AVX2: torch CPU dies with SIGILL
    {"hosta": [0.1, 28, 8, 0, 100, 1]},  # 8 GB free RAM < 16
    {"hosta": [0.1, 28, 100, 0, 4, 1]},  # 4 GB free in /var/tmp < 5
    {"hosta": [0.1, 28, 100, 1, 100, 1]},  # another uwrun run already there (per_host 1)
    {"hosta": None},  # unreachable
])
def test_uw_no_qualifying_host_pauses_without_running_anything(make, hosts):
    fixture = make(uw=True, hosts=hosts)
    assert signal_limit.main(fixture.argv) == 3
    state = fixture.state()
    assert state["status"] == "running" and "uw_no_host_qualifies" in state["next_action"]
    assert not any(c["cwd"].startswith(str(fixture.uw_root)) for c in fixture.stages_called())


@pytest.mark.parametrize("patch, reason", [
    ({"python": "/home/r226lin/venv/bin/python"}, "uw_path_not_host_local"),
    ({"model_directory": "/var/tmp/../home/model"}, "uw_path_not_host_local"),
    ({"min_local_gb": 4}, "uw_disk_reserve_below_5_GiB"),
    ({"hosts": ["iccad12", "iccad12"]}, "uw_hosts_invalid"),
    ({"per_host": 0}, "uw_limit_invalid"),
    ({"forecast_seconds": {}}, "uw_forecast_missing_or_invalid"),
    ({"uwrun": "/nonexistent/uwrun"}, "uwrun_missing"),
    ({"uwrun": "uwrun"}, "uwrun_not_absolute"),
    ({"extra": 1}, "uw_block_invalid"),
    # No host-side setup hook: it would run before the pin check or rewrite unseen after it.
    ({"setup": "true"}, "uw_setup_not_supported"),
])
def test_uw_block_refusals(make, capsys, patch, reason):
    fixture = make(uw=True, uw_patch=patch)
    assert signal_limit.main(fixture.argv) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == reason
    assert not (fixture.tmp / "state").exists()


@pytest.mark.parametrize("unpin", ["uw_stage_a.py", "uwrun"])
def test_uw_path_requires_the_runner_and_uwrun_pinned(make, capsys, unpin):
    fixture = make(uw=True, unpin=(unpin,))
    assert signal_limit.main(fixture.argv) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "plan_does_not_pin"
    assert not (fixture.tmp / "state").exists()


def test_uw_execution_refuses_a_dry_run_plan(make, capsys):
    fixture = make(uw=True, plan_patch={"dry_run": True})
    assert signal_limit.main(fixture.argv) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "uw_execution_with_dry_run"
    assert not (fixture.tmp / "state").exists()


# ---------------------------------------------------------------- every stage on UWaterloo


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_uw_every_stage_runs_on_the_hosts_and_nothing_here(make, device):
    hosts = None
    if device == "cuda":  # (load, nproc, availGB, uwruns, localGB, ready, avx2, free VRAM MiB)
        hosts = {"gpua": [0.1, 20, 80, 0, 300, 1, 1, 11000]}
    fixture = make(uw_all=device, hosts=hosts)
    assert signal_limit.main(fixture.argv) == 0, fixture.journal()[-3:]
    state = fixture.state()
    assert state["status"] == "completed"
    assert state["selections"] == EXPECTED_SELECTIONS  # same evidence, same selections
    calls = fixture.stages_called()
    order = first_order(calls)
    assert sorted(order) == sorted(EXPECTED_ORDER)
    # Stage A's independent stages may interleave across hosts; everything else is serial.
    stage_a = set(uw_stage_a_stages())
    assert [s for s in order if s not in stage_a] == [
        s for s in EXPECTED_ORDER if s not in stage_a]
    # Every stage ran in a fake host's run directory, none in this checkout.
    assert all(c["cwd"].startswith(str(fixture.uw_root)) for c in calls)
    for call in calls:
        assert (call["device"], call["uwrun_threads"], call["torch_threads"]) == (
            device, "6", "6")
        assert argv_value(call["argv"], "--device") == device
        assert call["cuda_visible"] == ("0" if device == "cuda" else None)
        assert not any(part.startswith(str(fixture.tmp)) for part in call["argv"])
    journal = fixture.journal()
    started = [e for e in journal if e["event"] == "stage_started"]
    assert started and all(e["executor"] == "uwaterloo" and e["device"] == device
                           for e in started)
    live = {e["stage"] for e in started if e["live_useful_mixed"]}
    assert live == {s for s in EXPECTED_ORDER if s.startswith("rl-") and "-h_" not in s}
    monitored = {e["stage"] for e in journal if e["event"] == "useful_mixed_monitored"}
    assert monitored == live
    # The continuation started from the fetched parent checkpoint, shipped as an input.
    final = [c for c in calls if c["stage"] == "rl-1e-04-s42-u300"][0]
    assert argv_value(final["argv"], "--continue-output-from").startswith(
        "inputs/resume_checkpoint")
    # RL*'s accepted manifest named REMOTE image paths; SFT-E (which opens every image,
    # fake entry rule) ran on its translated copy, re-bundled with this machine's images.
    rl_star = state["completed"]["rl-1e-04-s42-u300"]["fragment"]
    assert rl_star["accepted_manifest"].endswith("accepted_manifest.local.jsonl")
    rows = [json.loads(line) for line in Path(rl_star["accepted_manifest"]).read_text()
            .splitlines()]
    assert [r["id"] for r in rows] == ACCEPTED_300
    assert all(Path(r["image"]).read_text() == "jpg"
               and str(fixture.tmp / "state" / "uw" / "bundles") in r["image"] for r in rows)
    raw = Path(rl_star["accepted_manifest"]).with_name("accepted_manifest.jsonl")
    assert all(str(fixture.uw_root) in json.loads(line)["image"]
               for line in raw.read_text().splitlines())
    if device == "cuda":
        finished = [e for e in journal if e["event"] == "stage_finished"]
        assert finished and all(
            e["remote_peak_cuda_allocated_bytes"] == 3 * 1024**3 for e in finished)
    # Quiet mode on this machine is not waited out: no stage runs here.
    assert not [e for e in journal if e["event"] == "quiet_mode_wait"]


def test_uw_ships_an_adapter_with_its_run_record(make):
    # W and I are final_adapter directories inside their trainer runs; the trainer reads
    # <run>/local_training.json, so the run directory ships with the adapter (fake rule).
    local = make(init_in_run=True)
    assert signal_limit.main(local.argv) == 0  # the same layout passes here


def test_uw_ships_an_adapter_with_its_run_record_remotely(make):
    fixture = make(uw_all="cpu", init_in_run=True)
    assert signal_limit.main(fixture.argv) == 0, fixture.journal()[-3:]
    calls = {c["stage"]: c for c in fixture.stages_called()}
    assert argv_value(calls["P0-I"]["argv"], "--adapter") == (
        "inputs/init_adapter/run-i/final_adapter")
    stage0 = calls["Stage-0"]["argv"]
    assert argv_value(stage0, "--init") == "inputs/init_adapter/run-w/final_adapter"


def test_uw_every_stage_ignores_local_quiet_mode(make):
    fixture = make(uw_all="cpu")
    (fixture.tmp / "QUIET").write_text("")
    assert signal_limit.main(fixture.argv) == 0
    assert fixture.state()["status"] == "completed"


def test_uw_manifest_translation_is_what_lets_sft_e_open_its_images(make, monkeypatch):
    # Forced fire: without the translation the accepted manifest names REMOTE paths, which
    # do not exist here, so SFT-E cannot be bundled and the study stops.
    uw_stage_a = importlib.import_module("uw_stage_a")
    real = uw_stage_a.UWRunner.import_fragment
    monkeypatch.setattr(uw_stage_a.UWRunner, "import_fragment",
                        staticmethod(lambda destination, local, bundle=None:
                                     real(destination, local, None)))
    fixture = make(uw_all="cpu")
    assert signal_limit.main(fixture.argv) == 2
    state = fixture.state()
    assert state["status"] == "stopped"
    assert "sft-e-1e-05-s42" not in state["completed"]
    assert "rl-1e-04-s42-u300" in state["completed"]


def test_uw_live_useful_mixed_stop_stops_the_remote_rl_stage(make):
    fixture = make(uw_all="cpu")
    run_id = "rl-1e-05-s42-u100"
    payload = fixture.scenario[run_id]["payload"]
    payload["useful_mixed"] = [False] * 100  # 0 of 100 mixed: below 40
    fixture.scenario[run_id]["behaviour"] = {"linger": 30}  # alive after writing its groups
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    state = fixture.state()
    assert state["stop_reason"] == "rolling_useful_mixed_below_0_40"
    journal = fixture.journal()
    (finished,) = [e for e in journal if e["event"] == "stage_finished"
                   and e["stage"] == run_id]
    # Stopped while it ran (no rc of its own), long before its 30 s linger ended.
    assert finished["exit_code"] is None and finished["reason"] == (
        "rolling_useful_mixed_below_0_40")
    assert finished["elapsed_seconds"] < 25
    stopped = {e["name"] for e in fixture.uw_log() if e["event"] == "stop"}
    assert finished["uw_name"] in stopped
    assert any(e["event"] == "ssh" for e in fixture.uw_log())  # the mirror read it there


def test_uw_late_useful_mixed_window_stops_after_the_fetch(make):
    # A stage that ends before any mirror poll: the fetched groups complete the mirror and
    # the launcher's late poll still stops on them (nothing fails open).
    fixture = make(uw_all="cpu", uw_patch={"mixed_poll_seconds": 3600})
    run_id = "rl-1e-05-s42-u100"
    fixture.scenario[run_id]["payload"]["useful_mixed"] = [False] * 100
    fixture.write()
    assert signal_limit.main(fixture.argv) == 2
    assert fixture.state()["stop_reason"] == "rolling_useful_mixed_below_0_40"


def test_uw_busy_host_below_the_idle_thread_floor_is_not_used(make):
    # 28 cores at load 25: 3 idle < 6 required; the other host (load 0.5 of 24) is used.
    fixture = make(uw_all="cpu", uw_patch={"min_idle_threads": 6},
                   hosts={"busy": [25.0, 28, 100, 0, 100, 1], "idle": [0.5, 24, 100, 0, 100, 1]})
    assert signal_limit.main(fixture.argv) == 0
    assert {e["host"] for e in fixture.uw_log() if e["event"] == "run"} == {"idle"}


def test_uw_cuda_host_below_the_vram_floor_is_not_used(make):
    # Another process holds the GPU: 4,000 MiB free < the 10,000 MiB floor.
    fixture = make(uw_all="cuda", hosts={"gpua": [0.1, 20, 80, 0, 300, 1, 1, 4000]})
    assert signal_limit.main(fixture.argv) == 3
    state = fixture.state()
    assert state["status"] == "running" and "uw_no_host_qualifies" in state["next_action"]
    assert fixture.stages_called() == []


@pytest.mark.parametrize("options, reason", [
    ({"uw_all": "cpu", "plan_patch": {"execution": "cloud"}}, "execution_unknown"),
    ({"uw_all": "cpu", "plan_patch": {"stage_a_execution": "local"}},
     "execution_uwaterloo_with_local_stage_a"),
    ({"uw_all": "cpu", "plan_patch": {"dry_run": True}}, "uw_execution_with_dry_run"),
    ({"uw_all": "cpu", "uw_patch": {"forecast_seconds": {"stage_a_sft": 1}}},
     "uw_forecasts_duplicate_plan_forecasts"),
    ({"uw_all": "cpu", "uw_patch": {"device": "cuda", "min_free_vram_mib": 9000}},
     "plan_device_differs_from_uw_device"),
    ({"uw_all": "cpu", "uw_patch": {"device": "tpu"}}, "plan_device_differs_from_uw_device"),
    ({"uw_all": "cuda", "uw_patch": {"min_free_vram_mib": None}},
     "uw_cuda_without_vram_floor"),
    ({"uw_all": "cuda", "uw_patch": {"per_host": 2}}, "uw_cuda_per_host_not_1"),
    ({"uw": True, "uw_patch": {"device": "cuda", "min_free_vram_mib": 9000}},
     "uw_stage_a_only_mode_is_cpu"),
    ({"uw_all": "cpu", "uw_patch": {"mixed_poll_seconds": 0}}, "uw_limit_invalid"),
    ({"uw_all": "cpu", "uw_patch": {"min_idle_threads": 0}}, "uw_limit_invalid"),
])
def test_uw_every_stage_plan_refusals(make, capsys, options, reason):
    fixture = make(**options)
    assert signal_limit.main(fixture.argv) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == reason
    assert not (fixture.tmp / "state").exists()
