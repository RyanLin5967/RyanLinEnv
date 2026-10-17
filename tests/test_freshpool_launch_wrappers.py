"""Fixture tests for the freshpool stage wrappers: trainer_fragment, h_assay and p0_screen.

No model runs. The trainer and the generation functions are fakes that write the files the
real ones write; the wrappers' reading of them goes through the real
vision_rl.vlm_local_training verifiers (read_endpoint_receipt, committed_screen_entries,
parse_label, load_spec). Expected values are written out from the fixture design.
"""

import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "freshpool_launch"
sys.path.insert(0, str(PACKAGE))

vlt = importlib.import_module("vision_rl.vlm_local_training")
h_assay = importlib.import_module("h_assay")
p0_screen = importlib.import_module("p0_screen")
trainer_fragment = importlib.import_module("trainer_fragment")
evaluator = importlib.import_module("runpy").run_path(
    str(ROOT / "scripts" / "evaluate_freshpool_gates.py"))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


W_WEIGHTS = b"W adapter weights"
W_SHA = sha_bytes(W_WEIGHTS)
LABELS = ["Alpha", "Bravo", "Charlie"]
OTHER = {"Alpha": "Bravo", "Bravo": "Charlie", "Charlie": "Alpha"}


def write_spec(path):
    path.write_text(json.dumps({
        "schema_version": 1, "benchmark": "fixture", "question": "Which?",
        "split_policy": "fixture", "preprocessing": "none", "labels": LABELS,
        "sources": [{"url": "https://example.invalid/data", "sha256": "0" * 64}]}))
    return path


def write_manifest(path, ids):
    path.write_text("".join(json.dumps({"id": i, "split": "train", "image": f"img/{i}.jpg",
                                        "label": LABELS[n % 3]}) + "\n"
                            for n, i in enumerate(ids)))
    return path


# ---------------------------------------------------------------- trainer_fragment

SCAN_IDS = [f"s{i:02d}" for i in range(20)]
# The fake screen rejects every third scanned candidate (index % 3 == 2).
ACCEPTED = [i for n, i in enumerate(SCAN_IDS) if n % 3 != 2]
SECONDS_PER_STEP = 10.0


class FakeTrainer:
    """Writes what train_local writes for a staged run, then publishes a real receipt."""

    def __init__(self, mode="grpo", lr=1e-5, seed=42, max_steps=6, corrupt=None, code=0):
        self.mode, self.lr, self.seed, self.max_steps = mode, lr, seed, max_steps
        self.corrupt, self.code, self.calls = corrupt, code, []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if self.code:
            return self.code
        value = {flag: argv[argv.index(flag) + 1] for flag in (
            "--output", "--resume-from-checkpoint", "--stop-after-steps", "--max-steps",
            "--stop-after-active-seconds", "--continue-from-checkpoint", "--train")
            if flag in argv}
        output = Path(value["--output"])
        max_steps = int(value.get("--max-steps", self.max_steps))
        settings = {"mode": self.mode, "max_steps": max_steps, "max_active_seconds": None,
                    "learning_rate": self.lr, "seed": self.seed, "beta": 0.0, "device": "cpu",
                    "probe": None, "train_only": False,
                    "init_adapter": {"weights_sha256": W_SHA},
                    "initial_adapter_precision_transfer": None, "continuation": None}
        identity = {"fixture": True, "train": value.get("--train", "fixture-train")}
        start, inherited = 0, []
        if "--resume-from-checkpoint" in value:
            start = int(Path(value["--resume-from-checkpoint"]).name.split("-")[1])
            assert Path(value["--resume-from-checkpoint"]).is_dir()
        if "--continue-from-checkpoint" in value:
            # SFT-T: the trainer's own parent verification (receipt boundary_reached at its
            # declared stop_after_steps, final checkpoint, unchanged settings, new TRAIN).
            parent = Path(value["--continue-from-checkpoint"])
            settings["continuation"] = vlt._continuation_identity(parent, identity, settings)
            start = settings["continuation"]["parent_global_step"]
            inherited = json.loads((parent / "exposure.json").read_text())["entries"]
        boundary = {"stop_after_steps": None, "stop_after_active_seconds": None}
        if "--stop-after-steps" in value:
            stop = boundary["stop_after_steps"] = int(value["--stop-after-steps"])
        else:
            limit = boundary["stop_after_active_seconds"] = float(
                value["--stop-after-active-seconds"])
            stop = math.ceil(limit / SECONDS_PER_STEP)  # first step at or above it
        assert start < stop <= max_steps
        output.mkdir(parents=True, exist_ok=True)
        grpo = self.mode == "grpo"
        for step in range(start + 1, stop + 1):
            with (output / "optimizer_audit.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, "completed_state_active_seconds":
                                         SECONDS_PER_STEP * step}) + "\n")
            if grpo:
                group_id = ACCEPTED[step - 1] if self.corrupt != "reward_ids" else "zz"
                with (output / "reward_groups.jsonl").open("a") as handle:
                    handle.write(json.dumps({"id": group_id, "mixed": step % 2 == 1}) + "\n")
        checkpoint = output / f"checkpoint-{stop}"
        checkpoint.mkdir()
        weights = f"weights {self.mode} {stop}".encode()
        for name, content in (("trainer_state.json", json.dumps({"global_step": stop})),
                              ("optimizer.pt", "opt"), ("scheduler.pt", "sched"),
                              ("rng_state.pth", "rng"), ("adapter_config.json", "{}")):
            (checkpoint / name).write_text(content)
        (checkpoint / "adapter_model.safetensors").write_bytes(weights)
        entries = inherited + [
            {"invocation_id": "i", "global_step": k,
             "id": SCAN_IDS[(k - start - 1) % len(SCAN_IDS)] if inherited else ACCEPTED[k - 1]}
            for k in range(len(inherited) + 1, stop + 1)]
        (checkpoint / "exposure.json").write_text(json.dumps({"schema_version": 1,
                                                              "entries": entries}))
        active = {"schema_version": 1, "scope": vlt.ACTIVE_TIME_SCOPE, "device": "cpu",
                  "global_step": stop, "total_budget_seconds": None,
                  "committed_active_seconds": SECONDS_PER_STEP * stop,
                  "checkpoint_file_sha256": {
                      name: vlt._file_sha256(checkpoint / name)
                      for name in vlt.ACTIVE_CHECKPOINT_FILES},
                  "exposure_sha256": vlt._file_sha256(checkpoint / "exposure.json")}
        if grpo:
            last = SCAN_IDS.index(ACCEPTED[stop - 1])
            screen = [{"scan_position": n, "id": SCAN_IDS[n], "accepted": n % 3 != 2}
                      for n in range(last + 1)]
            (checkpoint / "screen.json").write_text(json.dumps({"entries": screen}))
            active["screen_sha256"] = vlt._file_sha256(checkpoint / "screen.json")
        (checkpoint / "active_time.json").write_text(json.dumps(active))
        adapter = output / "final_adapter"
        adapter.mkdir(exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(weights)
        (adapter / "adapter_config.json").write_text("{}")
        (output / "adapter-source.py").write_text("# fixture source\n")
        record = {
            "status": "completed", "stage_id": "fixture",
            "control_identity": identity,
            "settings": settings,
            "environment": {"adapter_source_sha256": vlt._file_sha256(
                output / "adapter-source.py")},
            "optimizer_audit": {"active_time": {"fixture": True}},
            "final_adapter_sha256": sha_bytes(weights),
            "reference": {"kind": "initial_saved_adapter",
                          "initial_adapter_weights_sha256": W_SHA,
                          "adapter": {"sha256": "tensor-identity-of-W"}} if grpo else None,
            "requested_boundary": boundary,
        }
        (output / "local_training.json").write_text(json.dumps(record))
        vlt._publish_endpoint_receipt(output, record, stop)
        return 0


def run_wrapper(tmp_path, monkeypatch, trainer, *, arm="RL", run_id="rl-1e-05-s42-u150",
                output="out", stop=3, extra=(), trainer_extra=()):
    monkeypatch.setattr(trainer_fragment, "TRAINER", trainer)
    scan = write_manifest(tmp_path / "scan.jsonl", SCAN_IDS)
    stage = tmp_path / f"stage-{output}"
    stage.mkdir()
    fragment = stage / "fragment.json"
    argv = ["--run-id", run_id, "--arm", arm, "--lr", "1e-05", "--seed", "42",
            "--fragment", str(fragment), "--output", str(tmp_path / output),
            "--scan-manifest", str(scan), *extra, "--",
            "--stop-after-steps", str(stop), *trainer_extra]
    code = trainer_fragment.main(argv)
    return code, (json.loads(fragment.read_text()) if fragment.exists() else None)


def snapshot(directory):
    return {str(p.relative_to(directory)): sha_bytes(p.read_bytes())
            for p in sorted(directory.rglob("*")) if p.is_file()}


def test_rl_fragment_reads_verified_receipt_and_committed_screen(tmp_path, monkeypatch):
    trainer = FakeTrainer()
    code, fragment = run_wrapper(tmp_path, monkeypatch, trainer)
    assert code == 0
    assert trainer.calls[0][-2:] == ["--output", str((tmp_path / "out").resolve())]
    assert fragment["status"] == "completed"
    assert (fragment["run_id"], fragment["arm"], fragment["lr"], fragment["seed"]) == (
        "rl-1e-05-s42-u150", "RL", 1e-05, 42)
    assert fragment["updates"] == 3
    assert fragment["committed_active_seconds"] == 30.0
    assert fragment["step_committed_active_seconds"] == {"1": 10.0, "2": 20.0, "3": 30.0}
    assert fragment["adapter_sha256"] == sha_bytes(b"weights grpo 3")
    assert fragment["reference_sha256"] == W_SHA
    assert fragment["useful_mixed"] == [True, False, True]
    assert fragment["receipt_status"] == "boundary_reached"
    assert Path(fragment["checkpoint"]).name == "checkpoint-3"
    assert Path(fragment["adapter_path"]).name == "final_adapter"
    accepted = [json.loads(line) for line in
                Path(fragment["accepted_manifest"]).read_text().splitlines()]
    scanned = [json.loads(line)["id"] for line in
               Path(fragment["scanned_manifest"]).read_text().splitlines()]
    assert [row["id"] for row in accepted] == ["s00", "s01", "s03"]
    assert scanned == ["s00", "s01", "s02", "s03"]
    # Trainer-loadable rows: exactly id/split/image/label, image made absolute.
    assert set(accepted[0]) == {"id", "split", "image", "label"}
    assert accepted[0]["image"] == str((tmp_path / "img" / "s00.jpg").resolve())


def test_rl_continuation_copies_the_parent_and_stays_cumulative(tmp_path, monkeypatch):
    first = FakeTrainer()
    code, parent = run_wrapper(tmp_path, monkeypatch, first, output="u3", stop=3)
    assert code == 0
    before = snapshot(tmp_path / "u3")
    second = FakeTrainer()
    code, final = run_wrapper(tmp_path, monkeypatch, second, output="u6", stop=6,
                              run_id="rl-1e-05-s42-u600",
                              extra=["--continue-output-from", parent["checkpoint"]])
    assert code == 0
    # The trainer resumes the COPY inside its own new output; the parent is untouched.
    resume = second.calls[0][second.calls[0].index("--resume-from-checkpoint") + 1]
    assert resume == str((tmp_path / "u6" / "checkpoint-3").resolve())
    assert snapshot(tmp_path / "u3") == before
    assert final["updates"] == 6 and final["committed_active_seconds"] == 60.0
    # Cumulative: the series passes through the parent's own committed seconds at step 3.
    assert final["step_committed_active_seconds"]["3"] == parent["committed_active_seconds"]
    assert sorted(map(int, final["step_committed_active_seconds"])) == [1, 2, 3, 4, 5, 6]
    assert final["useful_mixed"] == [True, False, True, False, True, False]
    ids = [json.loads(line)["id"] for line in
           Path(final["accepted_manifest"]).read_text().splitlines()]
    assert ids == ["s00", "s01", "s03", "s04", "s06", "s07"]
    assert final["receipt_status"] == "total_budget_exhausted"


def test_sft_fragment_uses_initial_adapter_and_no_screen(tmp_path, monkeypatch):
    code, fragment = run_wrapper(tmp_path, monkeypatch, FakeTrainer(mode="sft"),
                                 arm="SFT-E", run_id="sft-e-1e-05-s42")
    assert code == 0
    assert fragment["reference_sha256"] == W_SHA
    assert fragment["useful_mixed"] == []
    assert "accepted_manifest" not in fragment and "scanned_manifest" not in fragment


@pytest.mark.parametrize("trainer,arm", [
    (FakeTrainer(corrupt="reward_ids"), "RL"),  # reward journal disagrees with exposure
    (FakeTrainer(mode="sft"), "RL"),            # trainer mode is not the arm's
    (FakeTrainer(lr=5e-5), "RL"),               # trainer LR is not the stage's
])
def test_inconsistent_trainer_output_writes_no_fragment(tmp_path, monkeypatch, trainer, arm):
    code, fragment = run_wrapper(tmp_path, monkeypatch, trainer, arm=arm)
    assert code == 2 and fragment is None


def test_failed_trainer_writes_no_fragment(tmp_path, monkeypatch):
    code, fragment = run_wrapper(tmp_path, monkeypatch, FakeTrainer(code=2))
    assert code == 2 and fragment is None


def test_wrapper_owns_output_and_refuses_reuse(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_wrapper(tmp_path, monkeypatch, FakeTrainer(), trainer_extra=["--output", "x"])
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "leftover").write_text("from an interrupted instance")
    with pytest.raises(SystemExit):
        run_wrapper(tmp_path, monkeypatch, FakeTrainer(), output="used")


def load_launch():
    spec = importlib.util.spec_from_file_location("freshpool_launch_for_wrappers",
                                                  PACKAGE / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sft_t(tmp_path, monkeypatch, parent_checkpoint, t_rl, *, max_steps=100):
    monkeypatch.setattr(trainer_fragment, "TRAINER", FakeTrainer(mode="sft"))
    scan = write_manifest(tmp_path / "scanned.jsonl", SCAN_IDS)
    stage = tmp_path / "stage-sft-t"
    stage.mkdir()
    fragment = stage / "fragment.json"
    code = trainer_fragment.main([
        "--run-id", "sft-t-1e-05-s42", "--arm", "SFT-T", "--lr", "1e-05", "--seed", "42",
        "--fragment", str(fragment), "--output", str(tmp_path / "sft-t"), "--",
        "--mode", "sft", "--train", str(scan), "--continue-from-checkpoint",
        str(parent_checkpoint), "--stop-after-active-seconds", repr(t_rl),
        "--max-steps", str(max_steps)])
    return code, (json.loads(fragment.read_text()) if fragment.exists() else None)


def test_sft_t_through_the_wrapper_continues_sft_e_and_satisfies_sft_t_check(
        tmp_path, monkeypatch):
    launch = load_launch()
    # SFT-E (scaled: 3 steps for 600) stops at its declared boundary below max_steps.
    code, parent = run_wrapper(tmp_path, monkeypatch, FakeTrainer(mode="sft", max_steps=6),
                               arm="SFT-E", run_id="sft-e-1e-05-s42", output="sft-e", stop=3)
    assert code == 0
    assert (parent["receipt_status"], parent["requested_boundary"]["stop_after_steps"]) == (
        "boundary_reached", 3)
    assert launch.Launcher.sft_e_boundary_check(3)(parent) is None
    before = snapshot(tmp_path / "sft-e")
    # T_RL 55 s at 10 s/step: steps 4 (40) and 5 (50) are below it, step 6 (60) reaches it.
    code, fragment = run_sft_t(tmp_path, monkeypatch, parent["checkpoint"], 55.0)
    assert code == 0
    assert snapshot(tmp_path / "sft-e") == before  # the parent endpoint is never written
    assert fragment["arm"] == "SFT-T" and fragment["updates"] == 6
    assert fragment["committed_active_seconds"] == 60.0
    # The audit series starts at the parent's step + 1 (601 at full scale), cumulative.
    assert fragment["step_committed_active_seconds"] == {"4": 40.0, "5": 50.0, "6": 60.0}
    assert fragment["receipt_status"] == "boundary_reached"
    assert fragment["requested_boundary"] == {"stop_after_steps": None,
                                              "stop_after_active_seconds": 55.0}
    assert fragment["reference_sha256"] == W_SHA
    record = json.loads((tmp_path / "sft-t" / "local_training.json").read_text())
    assert record["settings"]["continuation"]["parent_global_step"] == 3
    # The launcher's own SFT-T checks accept the real wrapper's fragment...
    host = SimpleNamespace(plan={"w_endpoint": {"adapter_sha256": W_SHA}}, g0={})
    validate = launch.Launcher.train_validator(
        host, "sft-t-1e-05-s42", "SFT-T", 1e-05, check=launch.Launcher.sft_t_check(parent, 55.0))
    assert validate(fragment, 1000.0) is None
    # ...and fire on it when T_RL says the run went a step too far.
    assert launch.Launcher.sft_t_check(parent, 45.0)(fragment) == (
        "sft_t_overshoots_first_boundary_at_t_rl")


def test_sft_e_run_to_max_steps_is_refused_by_launcher_and_trainer(tmp_path, monkeypatch):
    launch = load_launch()
    code, parent = run_wrapper(tmp_path, monkeypatch, FakeTrainer(mode="sft", max_steps=3),
                               arm="SFT-E", run_id="sft-e-1e-05-s42", output="sft-e", stop=3)
    assert code == 0 and parent["receipt_status"] == "total_budget_exhausted"
    assert launch.Launcher.sft_e_boundary_check(3)(parent) == "sft_e_not_a_continuable_boundary"
    with pytest.raises(vlt.LocalVLMError, match="boundary_reached"):
        run_sft_t(tmp_path, monkeypatch, parent["checkpoint"], 55.0)


# ---------------------------------------------------------------- h_assay

H_IDS = [f"h{i:03d}" for i in range(70)]  # three chunks: 32, 32, 6


def label_of(identifier):
    return LABELS[H_IDS.index(identifier) % 3] if identifier in H_IDS else LABELS[
        int(identifier[1:]) % 3]


class FakeGeneration:
    """predict_train_subset / sample_train_groups stand-ins writing the real file layouts.

    greedy: correct and terminated. seed 42: [correct, other label, unparseable, correct but
    not terminated]. seed 314: [other, correct, correct, other], all terminated.
    """

    def __init__(self, adapter_sha, drop=None):
        self.adapter_sha, self.drop, self.calls = adapter_sha, drop, []

    def rows(self, manifest):
        return [json.loads(line) for line in Path(manifest).read_text().splitlines()]

    def record(self):
        return {"status": "completed", "initial_adapter": {"weights_sha256": self.adapter_sha}}

    def predict(self, manifest, *, output, **kwargs):
        rows = self.rows(manifest)
        assert len(rows) <= 32 and kwargs["profile"].response_format == "answer_only"
        self.calls.append(("greedy", None, len(rows)))
        output.mkdir(parents=True)
        (output / "predictions.json").write_text(json.dumps({
            r["id"]: {"response": r["label"], "terminated": True} for r in rows}))
        return self.record()

    def sample(self, manifest, *, output, seed, **kwargs):
        rows = self.rows(manifest)
        assert len(rows) <= 32
        self.calls.append(("seed", seed, len(rows)))
        output.mkdir(parents=True)
        groups = {}
        for r in rows:
            gold, other = r["label"], OTHER[r["label"]]
            if seed == 42:
                group = [(gold, True), (other, True), ("not a label", True), (gold, False)]
            else:
                group = [(other, True), (gold, True), (gold, True), (other, True)]
            groups[r["id"]] = [{"response": text, "terminated": ended,
                                "completion_ids": [1, 2] if ended else [1]}
                               for text, ended in group]
            if self.drop == r["id"]:
                groups[r["id"]] = groups[r["id"]][:3]
        (output / "sampled_completions.json").write_text(json.dumps(groups))
        return self.record()


def assay_setup(tmp_path, monkeypatch, drop=None, adapter_sha=None):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"endpoint weights")
    fake = FakeGeneration(adapter_sha or sha_bytes(b"endpoint weights"), drop)
    monkeypatch.setattr(vlt, "predict_train_subset", fake.predict)
    monkeypatch.setattr(vlt, "sample_train_groups", fake.sample)
    manifest = tmp_path / "h_gate.jsonl"
    manifest.write_text("".join(json.dumps({"id": i, "split": "train", "image": f"{i}.jpg",
                                            "label": label_of(i)}) + "\n" for i in H_IDS))
    stage = tmp_path / "stage"
    stage.mkdir()
    argv = ["--run-id", "rl-1e-04-s42-u600", "--split", "h_gate", "--manifest", str(manifest),
            "--adapter", str(adapter), "--fragment", str(stage / "fragment.json"),
            "--stage-dir", str(stage), "--data-spec", str(write_spec(tmp_path / "spec.json")),
            "--model-directory", str(tmp_path / "model")]
    return fake, argv, stage


def test_h_gate_assay_scores_greedy_then_two_g4_seeds_in_chunks(tmp_path, monkeypatch):
    fake, argv, stage = assay_setup(tmp_path, monkeypatch)
    assert h_assay.main(argv) == 0
    fragment = json.loads((stage / "fragment.json").read_text())
    assert fake.calls == [(kind, seed, n) for n in (32, 32, 6)
                          for kind, seed in (("greedy", None), ("seed", 42), ("seed", 314))]
    assert (fragment["run_id"], fragment["split"]) == ("rl-1e-04-s42-u600", "h_gate")
    assert fragment["adapter_sha256"] == sha_bytes(b"endpoint weights")
    assert set(fragment["predictions"]) == set(H_IDS)
    gold = label_of("h004")  # Bravo
    other = OTHER[gold]
    assert fragment["predictions"]["h004"] == [gold, gold, other, None, None,
                                               other, gold, gold, other]
    # 9 completions per image, one (seed 42's fourth) not terminated.
    assert fragment["eos_fraction"] == pytest.approx(8 / 9)
    # The real evaluator accepts the fragment's shape: 9 completions per image.
    rows = [{"id": i, "label": label_of(i)} for i in H_IDS]
    result = evaluator["compare_repeated_classification"](
        rows, fragment["predictions"], fragment["predictions"], LABELS, completions=9,
        lower_quantile=0.1, resamples=100)
    # Per image: greedy 1 + seed 42 one (its unterminated gold is invalid) + seed 314 two
    # = 4 correct of 9, in every class.
    assert result["candidate"]["mean_class_accuracy"] == pytest.approx(4 / 9)


def test_h_assay_refuses_a_short_group(tmp_path, monkeypatch):
    _, argv, stage = assay_setup(tmp_path, monkeypatch, drop="h040")
    with pytest.raises(RuntimeError, match="h040"):
        h_assay.main(argv)
    assert not (stage / "fragment.json").exists()


def test_h_assay_refuses_generation_under_another_adapter(tmp_path, monkeypatch):
    _, argv, stage = assay_setup(tmp_path, monkeypatch, adapter_sha="0" * 64)
    with pytest.raises(RuntimeError, match="different adapter"):
        h_assay.main(argv)
    assert not (stage / "fragment.json").exists()


# ---------------------------------------------------------------- p0_screen

P0_IDS = sorted(f"x{i:03d}" for i in range(450))
random.Random(42).shuffle(P0_IDS)


def test_p0_writes_a_g1_document_the_evaluator_scores(tmp_path, monkeypatch):
    adapter = tmp_path / "w"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(W_WEIGHTS)
    canonical = [7, 8, 2]

    def score(rows, **kwargs):
        assert [r["id"] for r in rows] == P0_IDS[:400]
        # p = 0.5 for the first 300 candidates (u = 0.875), p = 0.99 after (u ~ 0.039).
        return [(0.5 if n < 300 else 0.99, 0.25, canonical) for n in range(len(rows))]

    sampled = []

    def sample(manifest, *, output, seed, **kwargs):
        rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
        assert len(rows) <= 32 and kwargs["init_adapter"] == adapter
        sampled.append((seed, [r["id"] for r in rows]))
        output.mkdir(parents=True)
        groups = {r["id"]: [
            {"response": r["label"], "terminated": True, "completion_ids": canonical},
            {"response": r["label"], "terminated": True, "completion_ids": [7, 9, 2]},
            {"response": OTHER[r["label"]], "terminated": True, "completion_ids": [5, 2]},
            {"response": OTHER[r["label"]], "terminated": True, "completion_ids": [5, 2]},
        ] for r in rows}
        (output / "sampled_completions.json").write_text(json.dumps(groups))
        return {"status": "completed", "initial_adapter": {"weights_sha256": W_SHA}}

    monkeypatch.setattr(p0_screen, "score_candidates", score)
    monkeypatch.setattr(vlt, "sample_train_groups", sample)
    stage = tmp_path / "stage"
    stage.mkdir()
    scan = tmp_path / "scan.jsonl"
    scan.write_text("".join(json.dumps({"id": i, "split": "train", "image": f"{i}.jpg",
                                        "label": LABELS[n % 3]}) + "\n"
                            for n, i in enumerate(P0_IDS)))
    assert p0_screen.main([
        "--scan-manifest", str(scan), "--adapter", str(adapter),
        "--fragment", str(stage / "fragment.json"), "--stage-dir", str(stage),
        "--data-spec", str(write_spec(tmp_path / "spec.json")),
        "--model-directory", str(tmp_path / "model")]) == 0
    document = json.loads((stage / "fragment.json").read_text())
    assert (document["seed"], document["adapter_sha256"]) == (42, W_SHA)
    assert document["scan_order"] == P0_IDS
    assert [e["id"] for e in document["journal"]] == P0_IDS[:400]
    assert [e["accepted"] for e in document["journal"]] == [True] * 300 + [False] * 100
    assert document["journal"][0]["u"] == pytest.approx(0.875)
    # Rollouts: the first 64 candidates, seeds 42 and 314, in chunks of 32.
    assert [(seed, len(ids)) for seed, ids in sampled] == [(42, 32), (42, 32),
                                                          (314, 32), (314, 32)]
    assert {g["id"] for g in document["rollouts"]} == set(P0_IDS[:64])
    result = evaluator["gate_g1"](document, W_SHA)
    values = result["values"]
    assert values["acceptance"] == pytest.approx(0.75)
    assert values["mean_u_accepted"] == pytest.approx(0.875)
    assert values["observed_useful_mixed"] == 1.0
    assert values["predicted_useful_mixed"] == pytest.approx(0.875)
    assert values["valid"] == 1.0 and values["eos"] == 1.0
    # Half the correct completions used a non-canonical tokenization.
    assert values["non_canonical_share_of_correct"] == pytest.approx(0.5)
    assert values["scorer_seconds_per_candidate"] == pytest.approx(0.25)
    # Designed to miss calibration: observed 1.0 vs predicted 0.875 is a 0.125 gap.
    assert result["checks"]["useful_mixed_calibrated"] is False
