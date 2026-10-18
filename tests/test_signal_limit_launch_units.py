"""Unit tests for the signal-limit launcher items: the stage table, plan-driven update counts,
init_endpoint, p0_screen's G handling and h_assay's assay G and constrained-greedy pass.

No model runs. Every expected value is written out from the fixture or the study text, never
read back from the code under test.
"""

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts" / "freshpool_launch"
sys.path.insert(0, str(PACKAGE))

launch = importlib.import_module("launch")
stage_table = importlib.import_module("stage_table")
h_assay = importlib.import_module("h_assay")
p0_screen = importlib.import_module("p0_screen")
vlt = importlib.import_module("vision_rl.vlm_local_training")

STUDY = json.loads((ROOT / "docs" / "aircraft-rl-next-study-v1.json").read_text())


# ---------------------------------------------------------------- stage table


def test_stage_table_agrees_with_the_study_json():
    assert stage_table.verify_against_study(STUDY) == []


@pytest.mark.parametrize("mutate, expected", [
    (lambda s: s["stages"][2]["arms"][1].update(lrs=[1e-5, 5e-5]), "A2 LRs"),
    (lambda s: s["stages"][2].update(arm_count=12), "arm count"),
    (lambda s: s["stages"][3]["rl"].update(N_accepted=600), "Stage B RL N"),
    (lambda s: s["stages"].pop(1), "stage order"),
    (lambda s: s["fixed_profile"].update(assay="H-select: greedy"), "assay G"),
    (lambda s: s["stages"][3]["rl"].update(
        reward="exact-EOS 0/1 if A2 won Gate A; exact_hn if A3 won"), "reward rule"),
])
def test_stage_table_refuses_a_study_it_does_not_follow(mutate, expected):
    study = copy.deepcopy(STUDY)
    mutate(study)
    problems = stage_table.verify_against_study(study)
    assert problems and any(expected in p for p in problems), problems


def test_stage_a_runs_are_the_13_preregistered_arms():
    runs = stage_table.stage_a_runs(5e-05)
    assert [r[3] for r in runs] == [
        "stage-a-a1-1e-05-s42", "stage-a-a1-5e-05-s42", "stage-a-a1-1e-04-s42",
        "stage-a-a2-1e-05-s42", "stage-a-a2-5e-05-s42", "stage-a-a2-1e-04-s42",
        "stage-a-a3-1e-05-s42", "stage-a-a3-5e-05-s42", "stage-a-a3-1e-04-s42",
        "stage-a-a5-1e-05-s42", "stage-a-a5-5e-05-s42", "stage-a-a5-1e-04-s42",
        "stage-a-a4-5e-05-s42"]
    assert [r[1] for r in runs if r[0] == "A3"] == ["hn_sft"] * 3
    assert len(stage_table.stage_a_runs()) == 12  # A4 waits for A1*'s LR
    with pytest.raises(ValueError):
        stage_table.stage_a_runs(2e-4)


def test_a_floor_matches_the_study_preview():
    # study.json G0: "a_floor = 375 / |S_b| published (preview 375/987 = 0.380)"
    assert stage_table.a_floor(300, 987) == 375 / 987
    assert round(stage_table.a_floor(300, 987), 3) == 0.380
    with pytest.raises(ValueError):
        stage_table.a_floor(300, 0)


# ---------------------------------------------------------------- launch.py helpers


def test_update_counts_default_to_the_freshpool_values_read_at_call_time(monkeypatch):
    assert launch.update_counts({}) == {"rl_screen": 150, "final": 600}
    monkeypatch.setattr(launch, "FINAL_UPDATES", 6)  # the CPU dry run scales the module values
    assert launch.update_counts({})["final"] == 6
    assert launch.update_counts({"updates": {"rl_screen": 100, "final": 300}}) == {
        "rl_screen": 100, "final": 300}


@pytest.mark.parametrize("updates", [{"rl_screen": 300, "final": 300}, {"final": 300},
                                     {"rl_screen": 100, "final": "300"}, [100, 300]])
def test_invalid_update_counts_refuse(updates):
    with pytest.raises(launch.Stop) as caught:
        launch.update_counts({"updates": updates})
    assert caught.value.reason == "plan_update_counts_invalid"


def test_init_endpoint_aliases_w_endpoint_and_refuses_a_conflict():
    w = {"adapter_sha256": "a" * 64, "adapter_path": "/w"}
    other = {"adapter_sha256": "b" * 64, "adapter_path": "/i"}
    assert launch.init_endpoint({"w_endpoint": w}) == w
    assert launch.init_endpoint({"init_endpoint": other}) == other
    assert launch.init_endpoint({"init_endpoint": w, "w_endpoint": dict(w)}) == w
    with pytest.raises(launch.Stop) as caught:
        launch.init_endpoint({"init_endpoint": other, "w_endpoint": w})
    assert caught.value.reason == "init_endpoint_conflicts_with_w_endpoint"


def templates(screen, final, sft_stop, sft_max):
    wrapper = ["--output", "{arm_dir}/trainer", "--"]
    rl = ["--lr", "{lr}", "--cap", "{arm_cap}", "--screen-tokenizations",
          "{screen_tokenizations}"]
    return {
        "rl150": {"argv": wrapper + rl + ["--stop-after-steps", str(screen)]},
        "rl600": {"argv": wrapper + rl},
        "sft_e": {"argv": wrapper + ["--order", "sequential", "--cycle", "--stop-after-steps",
                                     str(sft_stop), "--max-steps", str(sft_max)]},
        "egsft": {"argv": ["--screen-tokenizations", "{screen_tokenizations}"]},
    }


def test_template_checks_follow_the_plan_counts_not_the_600_literal():
    counts = {"rl_screen": 100, "final": 300}
    launch.verify_stage_templates(templates(100, 300, 300, 600), counts)
    with pytest.raises(launch.Stop) as caught:
        launch.verify_stage_templates(templates(100, 300, 600, 1200), counts)
    assert caught.value.reason == "sft_e_template_stop_after_steps_not_600"
    with pytest.raises(launch.Stop) as caught:
        launch.verify_stage_templates(templates(150, 300, 300, 600), counts)
    assert caught.value.reason == "rl150_template_stop_after_steps_not_150"
    # No counts: the freshpool literals, unchanged.
    launch.verify_stage_templates(templates(150, 600, 600, 1200))


# ---------------------------------------------------------------- p0_screen


def u(p, g):
    return 1 - p**g - (1 - p) ** g


def test_predicted_acceptance_and_mean_u_from_p_only():
    # 2 candidates at p = 0.5 and 2 at p = 0.05. Hand values:
    #   u_4(0.05) = 1 - 0.95^4 - 0.05^4 = 0.1854875 (rejected); u_8(0.05) = 0.33657956...
    #   (rejected); u_16(0.05) = 0.55987333... (accepted); u_G(0.5) = 1 - 2 / 2^G (accepted).
    scores = [(0.5, 1.0, []), (0.5, 1.0, []), (0.05, 1.0, []), (0.05, 1.0, [])]
    found = p0_screen.predicted(scores, (4, 8, 16))
    assert found["4"]["acceptance"] == 0.5 and found["8"]["acceptance"] == 0.5
    assert found["16"]["acceptance"] == 1.0
    assert found["4"]["mean_u"] == pytest.approx((2 * 0.875 + 2 * 0.1854875) / 4)
    assert found["4"]["mean_u_accepted"] == pytest.approx(0.875)
    assert found["16"]["mean_u_accepted"] == pytest.approx(
        (2 * (1 - 2 / 2**16) + 2 * u(0.05, 16)) / 4)
    nothing = p0_screen.predicted([(0.0, 1.0, [])], (8,))
    assert nothing["8"] == {"acceptance": 0.0, "mean_u": 0.0, "mean_u_accepted": None}


def test_group_choice_is_the_smallest_g_at_or_above_a_floor():
    prediction = {"8": {"acceptance": 0.38}, "16": {"acceptance": 0.6}}
    assert p0_screen.choose_group(prediction, (8, 16), 0.38) == 8
    assert p0_screen.choose_group(prediction, (16, 8), 0.381) == 16
    assert p0_screen.choose_group(prediction, (8, 16), 0.61) is None


@pytest.mark.parametrize("text", ["4,5", "8,8", "", "2"])
def test_group_lists_outside_4_8_16_refuse(text):
    with pytest.raises((SystemExit, ValueError)):
        p0_screen.parse_groups(text)


LABELS = ["Alpha", "Bravo"]


def p0_fixture(tmp_path, monkeypatch, scores_p):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": 1, "benchmark": "fixture", "question": "Which?",
        "split_policy": "fixture", "preprocessing": "none", "labels": LABELS,
        "sources": [{"url": "https://example.invalid/data", "sha256": "0" * 64}]}))
    ids = [f"x{i:03d}" for i in range(400)]
    manifest = tmp_path / "scan.jsonl"
    manifest.write_text("".join(json.dumps({"id": i, "split": "train", "image": f"{i}.jpg",
                                            "label": LABELS[n % 2]}) + "\n"
                                for n, i in enumerate(ids)))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"I weights")
    sampled = []

    def fake_score(rows, **common):
        return [(scores_p(n), 0.5, [5, 2]) for n, _ in enumerate(rows)]

    def fake_sample(rows, seed, stage_dir, **common):
        group = common["profile"].num_generations
        sampled.append((seed, group, len(rows)))
        return {row["id"]: [{"response": row["label"], "terminated": True,
                             "completion_ids": [5, 2]}] * group for row in rows}

    monkeypatch.setattr(p0_screen, "score_candidates", fake_score)
    monkeypatch.setattr(p0_screen, "sample_groups", fake_sample)
    argv = ["--scan-manifest", str(manifest), "--adapter", str(adapter), "--fragment",
            str(tmp_path / "fragment.json"), "--stage-dir", str(tmp_path / "stage"),
            "--data-spec", str(spec), "--model-directory", str(tmp_path / "model")]
    return argv, sampled, tmp_path / "fragment.json"


def test_p0_without_new_options_writes_the_freshpool_document(tmp_path, monkeypatch):
    argv, sampled, fragment = p0_fixture(tmp_path, monkeypatch, lambda n: 0.5)
    assert p0_screen.main(argv) == 0
    document = json.loads(fragment.read_text())
    assert sorted(document) == ["adapter_sha256", "journal", "rollouts", "scan_order", "seed"]
    assert sampled == [(42, 4, 64), (314, 4, 64)]
    assert document["journal"][0]["u"] == 0.875  # u_4(0.5) = 1 - 2/16
    assert len(document["rollouts"]) == 128


def test_p0_i_predicts_g_4_8_16_at_g4(tmp_path, monkeypatch):
    argv, sampled, fragment = p0_fixture(tmp_path, monkeypatch,
                                         lambda n: 0.5 if n < 100 else 0.05)
    assert p0_screen.main([*argv, "--predict-groups", "4,8,16"]) == 0
    document = json.loads(fragment.read_text())
    assert sampled == [(42, 4, 64), (314, 4, 64)]
    assert document["group"] == 4
    assert {g: v["acceptance"] for g, v in document["predicted"].items()} == {
        "4": 0.25, "8": 0.25, "16": 1.0}


def test_p0_prime_chooses_g_then_samples_at_it(tmp_path, monkeypatch):
    # 160 of 400 at p = 0.5 (accepted at every G), 240 at p = 0.05 (accepted only at G16):
    # acceptance G8 = 0.4, G16 = 1.0.
    argv, sampled, fragment = p0_fixture(tmp_path, monkeypatch,
                                         lambda n: 0.5 if n < 160 else 0.05)
    assert p0_screen.main([*argv, "--choose-groups", "8,16", "--a-floor", "0.38"]) == 0
    document = json.loads(fragment.read_text())
    assert document["chosen_group"] == 8 and document["group"] == 8
    assert sampled == [(42, 8, 64), (314, 8, 64)]
    assert all(len(r["completions"]) == 8 for r in document["rollouts"])
    assert document["journal"][200]["u"] == pytest.approx(u(0.05, 8))
    assert document["journal"][200]["accepted"] is False

    sampled.clear()
    assert p0_screen.main([*argv, "--choose-groups", "8,16", "--a-floor", "0.41"]) == 0
    assert json.loads(fragment.read_text())["chosen_group"] == 16
    assert sampled == [(42, 16, 64), (314, 16, 64)]



def test_p0_prime_with_no_qualifying_g_samples_nothing(tmp_path, monkeypatch):
    # p = 0.01 everywhere: u_16(0.01) = 1 - 0.99^16 - 0.01^16 = 0.1485..., so no G qualifies.
    argv, sampled, fragment = p0_fixture(tmp_path, monkeypatch, lambda n: 0.01)
    assert p0_screen.main([*argv, "--choose-groups", "8,16", "--a-floor", "0.38"]) == 0
    document = json.loads(fragment.read_text())
    assert document["chosen_group"] is None and document["group"] is None
    assert document["rollouts"] == [] and sampled == []
    assert document["predicted"]["16"]["acceptance"] == 0.0
    assert len(document["journal"]) == 400


@pytest.mark.parametrize("extra", [["--num-generations", "5"],
                                   ["--choose-groups", "8,16"],
                                   ["--a-floor", "0.4"],
                                   ["--choose-groups", "8,16", "--a-floor", "0"],
                                   ["--choose-groups", "8", "--a-floor", "0.4",
                                    "--predict-groups", "4"]])
def test_p0_refuses_bad_group_options(tmp_path, monkeypatch, extra):
    argv, sampled, _ = p0_fixture(tmp_path, monkeypatch, lambda n: 0.5)
    with pytest.raises(SystemExit):
        p0_screen.main([*argv, *extra])
    assert sampled == []


# ---------------------------------------------------------------- h_assay


def h_fixture(tmp_path, monkeypatch, constrained_answer):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": 1, "benchmark": "fixture", "question": "Which?",
        "split_policy": "fixture", "preprocessing": "none", "labels": LABELS,
        "sources": [{"url": "https://example.invalid/data", "sha256": "0" * 64}]}))
    ids = [f"h{i:02d}" for i in range(40)]  # two chunks: 32 + 8
    manifest = tmp_path / "h_select.jsonl"
    manifest.write_text("".join(json.dumps({"id": i, "split": "train", "image": f"{i}.jpg",
                                            "label": LABELS[n % 2]}) + "\n"
                                for n, i in enumerate(ids)))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"endpoint weights")
    adapter_sha = hashlib.sha256(b"endpoint weights").hexdigest()
    trie = tmp_path / "trie.json"
    trie.write_text('{"trie": "fixture"}')
    calls = []

    def rows_of(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    def record():
        return {"status": "completed", "initial_adapter": {"weights_sha256": adapter_sha}}

    def greedy(manifest, *, output, **common):
        calls.append(("greedy", common["profile"].num_generations))
        output.mkdir(parents=True)
        (output / "predictions.json").write_text(json.dumps({
            r["id"]: {"response": r["label"], "terminated": True} for r in rows_of(manifest)}))
        return record()

    def sampled(manifest, *, output, seed, **common):
        calls.append(("seed", seed, common["profile"].num_generations))
        output.mkdir(parents=True)
        (output / "sampled_completions.json").write_text(json.dumps({
            r["id"]: [{"response": "Bravo", "terminated": True}] * 4
            for r in rows_of(manifest)}))
        return record()

    def constrained(manifest, *, output, label_trie, **common):
        calls.append(("constrained", str(label_trie)))
        output.mkdir(parents=True)
        (output / "predictions.json").write_text(json.dumps({
            r["id"]: {"response": constrained_answer, "terminated": True}
            for r in rows_of(manifest)}))
        return record()

    monkeypatch.setattr(vlt, "predict_train_subset", greedy)
    monkeypatch.setattr(vlt, "sample_train_groups", sampled)
    monkeypatch.setattr(h_assay, "constrained_predict", constrained)
    argv = ["--run-id", "E", "--split", "h_select", "--manifest", str(manifest), "--adapter",
            str(adapter), "--fragment", str(tmp_path / "fragment.json"), "--stage-dir",
            str(tmp_path / "stage"), "--data-spec", str(spec), "--model-directory",
            str(tmp_path / "model")]
    trie_sha = hashlib.sha256(b'{"trie": "fixture"}').hexdigest()
    return argv, calls, tmp_path / "fragment.json", trie, trie_sha, ids


def test_assay_without_new_options_is_the_freshpool_fragment(tmp_path, monkeypatch):
    argv, calls, fragment, *_ = h_fixture(tmp_path, monkeypatch, "Alpha")
    assert h_assay.main(argv) == 0
    document = json.loads(fragment.read_text())
    assert "constrained_greedy" not in document and "training_group" not in document
    assert [c[0] for c in calls] == ["greedy", "seed", "greedy", "seed"]


def test_assay_constrained_greedy_and_training_group(tmp_path, monkeypatch):
    argv, calls, fragment, trie, trie_sha, ids = h_fixture(tmp_path, monkeypatch, "Alpha")
    monkeypatch.setattr(h_assay, "constrained_supported", lambda: True)
    assert h_assay.main([*argv, "--label-trie", str(trie), "--label-trie-sha256", trie_sha,
                         "--training-group", "8"]) == 0
    document = json.loads(fragment.read_text())
    # The assay samples G4 whatever the training G (8 here).
    assert [c for c in calls if c[0] == "seed"] == [("seed", 42, 4), ("seed", 42, 4)]
    assert [c[0] for c in calls] == ["constrained", "greedy", "seed"] * 2
    assert document["training_group"] == 8 and document["assay_group"] == 4
    assert document["constrained_greedy"] == {
        "trie_sha256": trie_sha, "predictions": {i: "Alpha" for i in ids},
        "eos_fraction": 1.0}
    # The primary (unconstrained) predictions are untouched: greedy gold + 4 x Bravo.
    assert document["predictions"]["h00"] == ["Alpha", "Bravo", "Bravo", "Bravo", "Bravo"]


def test_assay_training_group_zero_records_null(tmp_path, monkeypatch):
    argv, _, fragment, *_ = h_fixture(tmp_path, monkeypatch, "Alpha")
    assert h_assay.main([*argv, "--training-group", "0"]) == 0
    document = json.loads(fragment.read_text())
    assert document["training_group"] is None and document["assay_group"] == 4


def test_assay_refuses_a_wrong_trie_hash_or_a_trainer_without_trie_support(tmp_path,
                                                                           monkeypatch):
    argv, calls, _, trie, trie_sha, _ = h_fixture(tmp_path, monkeypatch, "Alpha")
    monkeypatch.setattr(h_assay, "constrained_supported", lambda: True)
    with pytest.raises(SystemExit, match="does not hash"):
        h_assay.main([*argv, "--label-trie", str(trie), "--label-trie-sha256", "0" * 64])
    with pytest.raises(SystemExit, match="go together"):
        h_assay.main([*argv, "--label-trie", str(trie)])
    monkeypatch.setattr(h_assay, "constrained_supported", lambda: False)
    with pytest.raises(SystemExit, match="no label_trie parameter"):
        h_assay.main([*argv, "--label-trie", str(trie), "--label-trie-sha256", trie_sha])
    with pytest.raises(SystemExit):
        h_assay.main([*argv, "--num-generations", "8"])
    assert calls == []


def test_constrained_supported_reads_the_trainer_signature(monkeypatch):
    monkeypatch.setattr(vlt, "predict_train_subset", lambda manifest, *, label_trie=None: 0)
    assert h_assay.constrained_supported() is True
    monkeypatch.setattr(vlt, "predict_train_subset", lambda manifest, *, output=None: 0)
    assert h_assay.constrained_supported() is False


# ---------------------------------------------------------------- trainer thread override


def test_torch_threads_default_two_and_env_override(monkeypatch):
    monkeypatch.delenv("VISION_RL_TORCH_THREADS", raising=False)
    assert vlt._torch_threads() == 2
    monkeypatch.setenv("VISION_RL_TORCH_THREADS", "32")
    assert vlt._torch_threads() == 32
    for bad in ("0", "300", "x"):
        monkeypatch.setenv("VISION_RL_TORCH_THREADS", bad)
        with pytest.raises(vlt.LocalVLMError):
            vlt._torch_threads()


def test_stage_table_refuses_a_changed_reward_rule():
    study = copy.deepcopy(STUDY)
    study["stages"][3]["rl"]["reward"] = "exact-EOS 0/1 always"
    assert "Stage B RL reward rule differs" in stage_table.verify_against_study(study)


# ---------------------------------------------------------------- trainer_fragment profile


trainer_fragment = importlib.import_module("trainer_fragment")


def test_signal_limit_profile_only_for_new_arms_and_dr_grpo_rl():
    initial = {"weights_sha256": "c" * 64}
    rl = {"mode": "grpo", "beta": 0.0, "loss_type": "dr_grpo", "scale_rewards": "none",
          "screening": {"group_size": 8}, "reward": {"name": "exact_hn"}}
    assert trainer_fragment.signal_limit_profile("RL", rl, initial) == {
        "mode": "grpo", "init_sha256": "c" * 64, "loss_type": "dr_grpo",
        "scale_rewards": "none", "beta": 0.0, "group": 8, "reward": "exact_hn"}
    # A freshpool RL run records no loss_type: its fragment gains nothing.
    legacy = {"mode": "grpo", "beta": 0.04, "screening": {"group_size": 4}}
    assert trainer_fragment.signal_limit_profile("RL", legacy, initial) == {}
    assert trainer_fragment.signal_limit_profile("SFT-E", {"mode": "sft"}, initial) == {}
    assert trainer_fragment.signal_limit_profile("A3", {"mode": "hn_sft"}, initial) == {
        "mode": "hn_sft", "init_sha256": "c" * 64}
    assert trainer_fragment.SIGNAL_LIMIT_ARMS["LIM-T"] == ("egsft",)
    assert trainer_fragment.ARM_MODES == {"RL": "grpo", "SFT-E": "sft", "SFT-T": "sft",
                                          "EG-SFT": "egsft"}


# ---------------------------------------------------------------- phase_worker --remote-cpu

PHASE_ENTRY = '''
def main(argv):
    return 0
'''
FAKE_TORCH = '''
def set_num_threads(count):
    import os
    os.environ["FAKE_TORCH_THREADS_SET"] = str(count)
'''


@pytest.mark.parametrize("device, flags, uwrun, threads, code, message", [
    ("cpu", ["--remote-cpu"], "6", "6", 0, None),
    ("cpu", ["--remote-cpu"], "6", "2", 1, "VISION_RL_TORCH_THREADS equal to UWRUN_THREADS"),
    ("cpu", ["--remote-cpu"], None, None, 1, "UWRUN_THREADS in 1..256"),
    ("cpu", ["--remote-cpu", "--dry-run"], "6", "6", 1, "cannot also be --dry-run"),
    ("mps", ["--remote-cpu"], "6", "6", 1, "ACCELERATE_TORCH_DEVICE must be"),
    ("cpu", ["--dry-run"], None, "6", 1, "set only by the UW path"),
    ("cpu", ["--dry-run"], None, None, 0, None),
])
def test_phase_worker_remote_cpu(tmp_path, device, flags, uwrun, threads, code, message):
    """--remote-cpu (the UW path) needs cpu, UWRUN_THREADS and the trainer's thread variable
    equal to it, and never --dry-run; outside it the trainer's thread variable refuses."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "phase_entry.py").write_text(PHASE_ENTRY)
    (pkg / "torch.py").write_text(FAKE_TORCH)
    env = {**os.environ, "PYTHONPATH": str(pkg), "PYTORCH_ENABLE_MPS_FALLBACK": "0",
           "HF_HUB_OFFLINE": "1", "ACCELERATE_TORCH_DEVICE": device}
    for name, value in (("UWRUN_THREADS", uwrun), ("VISION_RL_TORCH_THREADS", threads)):
        env.pop(name, None)
        if value is not None:
            env[name] = value
    guards = tmp_path / "guards"
    result = subprocess.run(
        [sys.executable, "-B", str(PACKAGE / "phase_worker.py"), "--guard-dir", str(guards),
         "--stage", "s", "--kind", "train", "--entry", "phase_entry:main", "--entry-sha256",
         hashlib.sha256(PHASE_ENTRY.encode()).hexdigest(), *flags, "--", "--device", "cpu"],
        env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == code, result.stderr
    if code:
        assert message in result.stderr and not guards.exists()
        return
    (summary,) = guards.glob("s-*.json")
    stats = json.loads(summary.read_text())
    assert stats["status"] == "completed"
    assert stats.get("remote_cpu", False) is ("--remote-cpu" in flags)


# ---------------------------------------------------------------- uw_stage_a helpers


uw_stage_a = importlib.import_module("uw_stage_a")


def test_import_fragment_translates_remote_paths_and_checks_the_adapter(tmp_path):
    destination = tmp_path / "remote"
    adapter = destination / "arm" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (destination / "stage").mkdir()
    (destination / "remote_root.txt").write_text("/var/tmp/r226lin-runs/x/work\n")
    remote = "/var/tmp/r226lin-runs/x/work/out"
    fragment = {"adapter_path": f"{remote}/arm/final_adapter",
                "adapter_sha256": hashlib.sha256(b"weights").hexdigest(),
                "checkpoint": f"{remote}/arm/checkpoint-600", "other": "/var/tmp/elsewhere",
                "nested": [{"p": f"{remote}/stage/x.jsonl"}]}
    (destination / "stage" / "fragment.json").write_text(json.dumps(fragment))
    local = tmp_path / "fragment.json"
    assert uw_stage_a.UWRunner.import_fragment(destination, local) is None
    found = json.loads(local.read_text())
    assert found["adapter_path"] == str(destination) + "/arm/final_adapter"
    assert found["checkpoint"] == str(destination) + "/arm/checkpoint-600"
    assert found["other"] == "/var/tmp/elsewhere"
    assert found["nested"] == [{"p": str(destination) + "/stage/x.jsonl"}]
    (adapter / "adapter_model.safetensors").write_bytes(b"changed in transfer")
    assert uw_stage_a.UWRunner.import_fragment(destination, local) == (
        "uw_fetched_adapter_differs")


def test_remote_check_refuses_a_changed_or_missing_pin_and_an_empty_list(tmp_path,
                                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    good = {"a.py": hashlib.sha256(b"x = 1\n").hexdigest()}
    for pins, code in ((good, 0), ({"a.py": "0" * 64}, 1), ({"b.py": "0" * 64}, 1), ({}, 1)):
        (tmp_path / "pins.json").write_text(json.dumps(pins))
        assert uw_stage_a.remote_check("pins.json", "hardware.json") == code
        hardware = json.loads((tmp_path / "hardware.json").read_text())
        assert hardware["pins_checked"] == len(pins)
