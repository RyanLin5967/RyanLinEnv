"""Correction M3 (runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md, critic.md) and the C5 check.

(a) `prompts_per_update` P: P distinct TRAIN images x G generations per optimizer update.
(b) `skip_zero_advantage_steps`: no optimizer step (parameters, AdamW moments and step count
    untouched) when every advantage of the update is zero; the update still counts toward
    max_steps as P consumed images.
(c) `param_delta_l2`: every GRPO optimizer_audit.jsonl row carries the update's L2 change.
(d) C5: dynamic sampling's dropped generations are charged to committed active seconds.
Every expected value below is a hand value or a released-library fact, never the subject's.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_vlm_local_training import _settings_after_model_refusal, _tiny_text_policy, released_trl

from vision_rl import vlm_local_training as local

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = ROOT / "scripts" / "exploratory_sweep" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"m3_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def diagnostic(tmp_path):
    root = tmp_path / "data"
    local.make_probe_data(root)
    return root


def _kwargs(mode="grpo", profile=None, **extra):
    return local.trainer_kwargs(
        mode,
        Path("/out"),
        profile or replace(local.Profile(), num_generations=8),
        device="cpu",
        max_steps=4,
        learning_rate=1e-5,
        seed=42,
        save_steps=1,
        **extra,
    )


# (a) trainer kwargs, validation, settings, CLI -----------------------------------------------


def test_prompts_per_update_sets_generation_and_accumulation_to_p_times_g():
    legacy = _kwargs()
    # Hand values: legacy is one prompt x G 8; P 8 is 8 x 8 = 64 completions per update.
    assert legacy["gradient_accumulation_steps"] == legacy["generation_batch_size"] == 8
    ppu = _kwargs(grpo_overrides={"prompts_per_update": 8})
    assert ppu["gradient_accumulation_steps"] == ppu["generation_batch_size"] == 64
    assert ppu["per_device_train_batch_size"] == 1 and ppu["num_generations"] == 8
    changed = ("gradient_accumulation_steps", "generation_batch_size")
    assert {k: v for k, v in ppu.items() if k not in changed} == {
        k: v for k, v in legacy.items() if k not in changed
    }
    # The module's own knobs never reach GRPOConfig.
    both = _kwargs(beta=0.0, grpo_overrides={"prompts_per_update": 2,
                                             "skip_zero_advantage_steps": True})
    assert "prompts_per_update" not in both and "skip_zero_advantage_steps" not in both
    assert both["gradient_accumulation_steps"] == 16 and both["beta"] == 0.0


def test_multi_image_knobs_validate():
    for bad in (0, 33, True, 2.0, "2"):
        with pytest.raises(ValueError, match="1..32"):
            local.validate_grpo_overrides({"prompts_per_update": bad})
    with pytest.raises(ValueError, match="num_iterations"):
        local.validate_grpo_overrides({"prompts_per_update": 2, "num_iterations": 2})
    for bad in (1, "true", None):
        with pytest.raises(ValueError, match="skip_zero_advantage_steps must be a bool"):
            local.validate_grpo_overrides({"skip_zero_advantage_steps": bad})
    # beta > 0: a KL term has a gradient, so the skip would drop a real update.
    with pytest.raises(ValueError, match="requires beta 0"):
        _kwargs(beta=0.04, grpo_overrides={"skip_zero_advantage_steps": True})
    with pytest.raises(ValueError, match="only to GRPO"):
        _kwargs("sft", grpo_overrides={"prompts_per_update": 2})


def test_released_grpo_config_accepts_eight_prompts_by_eight_generations(tmp_path):
    trl = released_trl()
    args = trl.GRPOConfig(**{**_kwargs(beta=0.0, grpo_overrides={
        "prompts_per_update": 8, "skip_zero_advantage_steps": True}), "output_dir": str(tmp_path)})
    # Hand value 8 x 8 = 64; generation is aligned with the update (one per update).
    assert args.generation_batch_size == args.steps_per_generation == 64
    assert args.gradient_accumulation_steps == 64 and args.num_iterations == 1


def test_train_local_refuses_multi_image_outside_plain_grpo(diagnostic, tmp_path):
    common = dict(train=tmp_path / "t.jsonl", validation=None,
                  data_spec=tmp_path / "s.json", output=tmp_path / "o")
    with pytest.raises(ValueError, match="only to GRPO"):
        local.train_local(**common, mode="sft", grpo_overrides={"prompts_per_update": 2})
    refused = "cannot combine with screening or dynamic sampling"
    with pytest.raises(ValueError, match=refused):
        local.train_local(**common, grpo_overrides={"prompts_per_update": 2},
                          screen_journal=tmp_path / "o" / "screen.jsonl")
    with pytest.raises(ValueError, match=refused):
        local.train_local(**common, grpo_overrides={"prompts_per_update": 2},
                          dynamic_sampling_max_draws=4)
    with pytest.raises(ValueError, match="requires beta 0"):
        local.train_local(**common, grpo_overrides={"skip_zero_advantage_steps": True})
    # The diagnostic TRAIN split has 8 rows (make_probe_data): 9 distinct prompts cannot exist.
    with pytest.raises(ValueError, match="exceeds the TRAIN rows"):
        local.train_local(diagnostic / "train.jsonl", None, diagnostic / "data_spec.json",
                          tmp_path / "o9", model_directory=tmp_path / "absent", device="cpu",
                          max_steps=4, train_only=True,
                          grpo_overrides={"prompts_per_update": 9})


def test_multi_image_knobs_enter_settings_only_when_passed(diagnostic, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    record = _settings_after_model_refusal(
        diagnostic, tmp_path, monkeypatch, "m3", beta=0.0, loss_type="grpo",
        grpo_overrides={"prompts_per_update": 4, "skip_zero_advantage_steps": True})
    assert record["settings"]["grpo_overrides"] == {
        "prompts_per_update": 4, "skip_zero_advantage_steps": True}
    plain = _settings_after_model_refusal(diagnostic, tmp_path, monkeypatch, "plain")
    assert "grpo_overrides" not in plain["settings"]


def test_cli_routes_multi_image_knobs_only_when_given(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(local, "train_local", lambda **k: seen.clear() or seen.update(k) or {})
    base = ["--train", "t", "--data-spec", "s", "--output", "o", "--train-only"]
    assert local.main(base) == 0
    assert "grpo_overrides" not in seen
    argv = base + ["--prompts-per-update", "8", "--skip-zero-advantage-steps", "true"]
    assert local.main(argv) == 0
    assert seen["grpo_overrides"] == {"prompts_per_update": 8, "skip_zero_advantage_steps": True}
    monkeypatch.setattr(local, "predict_train_subset", lambda *a, **k: pytest.fail("ran"))
    predict = ["--predict-train", "m", "--data-spec", "s", "--output", "o",
               "--model-directory", "d"]
    for bad in (["--prompts-per-update", "2"], ["--skip-zero-advantage-steps", "true"]):
        with pytest.raises(SystemExit):
            local.main(predict + bad)
    capsys.readouterr()


def test_host_receipt_detects_the_prompts_per_update_feature(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # features() prepends the payload
    host_verify = _load_script("host_verify")
    assert host_verify.FEATURE_FLAGS["prompts_per_update"] == "--prompts-per-update"
    assert "prompts_per_update" in host_verify.features(tmp_path)["available"]


# (a) exposure lineage and endpoint receipt ---------------------------------------------------


def test_exposure_ledger_commits_p_distinct_prompts_per_update(tmp_path):
    ids = ["a", "b", "c", "d"]
    ledger = local._ExposureLedger(tmp_path, ids, prompts_per_update=2)
    ledger.begin(1)
    ledger.observe("a")
    with pytest.raises(local.LocalVLMError, match="same prompt twice"):
        ledger.observe("a")
    ledger.observe("c")
    with pytest.raises(local.LocalVLMError, match="exactly 2 actual TRAIN prompts"):
        ledger.observe("d")
    ledger.end(1)
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    ledger.save(checkpoint, 1)
    saved = json.loads((checkpoint / "exposure.json").read_text())["entries"]
    assert [(e["global_step"], e["id"]) for e in saved] == [(1, "a"), (1, "c")]
    ledger.begin(2)
    ledger.observe("b")
    with pytest.raises(local.LocalVLMError, match="lacks an actual prompt"):
        ledger.end(2)  # one of the two prompts observed
    restored = local._ExposureLedger(tmp_path, ids, {"entries": saved}, prompts_per_update=2)
    with pytest.raises(local.LocalVLMError, match="ancestry"):
        restored.begin(1)
    restored.begin(2)
    restored.observe("b")
    restored.observe("d")
    restored.end(2)
    assert [(e["global_step"], e["id"]) for e in restored.entries] == [
        (1, "a"), (1, "c"), (2, "b"), (2, "d")]
    # The legacy (P 1) reader refuses a two-prompt lineage; malformed lineages refuse.
    with pytest.raises(local.LocalVLMError, match="TRAIN identity"):
        local._ExposureLedger(tmp_path, ids, {"entries": saved})
    with pytest.raises(local.LocalVLMError, match="partial"):
        local._ExposureLedger(tmp_path, ids, {"entries": saved[:1]}, prompts_per_update=2)
    repeated = [{"global_step": 1, "id": "a"}, {"global_step": 1, "id": "a"}]
    with pytest.raises(local.LocalVLMError, match="repeats a prompt"):
        local._ExposureLedger(tmp_path, ids, {"entries": repeated}, prompts_per_update=2)
    with pytest.raises(local.LocalVLMError, match="continuation"):
        local._ExposureLedger(tmp_path, ids, {"entries": saved}, inherited=saved,
                              prompts_per_update=2)


def _synthetic_endpoint(output, entries, steps, overrides=None):
    """A model-free completed trajectory written through the real receipt functions."""
    output.mkdir(parents=True)
    checkpoint = output / f"checkpoint-{steps}"
    for directory in (checkpoint, output / "final_adapter"):
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"weights")
        (directory / "adapter_config.json").write_bytes(b"{}")
    local._write(checkpoint / "trainer_state.json", {"global_step": steps})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    local._write(checkpoint / "exposure.json", {"schema_version": 1, "entries": entries})
    local._write(checkpoint / "active_time.json", {
        "schema_version": 1, "scope": local.ACTIVE_TIME_SCOPE, "global_step": steps,
        "total_budget_seconds": None, "committed_active_seconds": 3.0,
        "exposure_sha256": local._file_sha256(checkpoint / "exposure.json"),
        "checkpoint_file_sha256": {name: local._file_sha256(checkpoint / name)
                                   for name in local.ACTIVE_CHECKPOINT_FILES}})
    source = local._file_sha256(Path(local.__file__))
    record = {
        "status": "completed", "stage_id": "m3",
        "settings": {"mode": "grpo", "max_steps": steps, "max_active_seconds": None,
                     **({"grpo_overrides": overrides} if overrides else {})},
        "requested_boundary": {"stop_after_steps": None, "stop_after_active_seconds": None},
        "control_identity": {"synthetic": True}, "environment": {"adapter_source_sha256": source},
        "reference": None, "optimizer_audit": {"active_time": {"synthetic": True}},
        "final_adapter_sha256": local._file_sha256(
            output / "final_adapter" / "adapter_model.safetensors"),
    }
    local._archive_source(output, source)
    local._write(output / "local_training.json", record)
    local._publish_endpoint_receipt(output, record, steps)
    return local.read_endpoint_receipt(output)


def test_endpoint_receipt_binds_p_prompts_per_global_step(tmp_path):
    two = [{"global_step": s, "id": i} for s, i in ((1, "a"), (1, "c"), (2, "b"), (2, "d"))]
    receipt = _synthetic_endpoint(tmp_path / "p2", two, 2, {"prompts_per_update": 2})
    assert receipt["global_step"] == 2 and receipt["committed_prompt_presentations"] == 4
    assert receipt["committed_prompt_ids"] == ["a", "c", "b", "d"]
    # The same four entries without the declared P are not a 2-step lineage.
    with pytest.raises(local.LocalVLMError, match="exposure lineage differs"):
        _synthetic_endpoint(tmp_path / "p1", two, 2)
    legacy = [{"global_step": 1, "id": "a"}, {"global_step": 2, "id": "b"}]
    assert _synthetic_endpoint(tmp_path / "legacy", legacy, 2)["committed_prompt_ids"] == [
        "a", "b"]


# (b) the zero-advantage gate and (c) the per-update delta ------------------------------------


def test_zero_advantage_gate_skips_the_step_and_leaves_adamw_state_untouched():
    torch = pytest.importorskip("torch")
    theta = torch.nn.Parameter(torch.tensor([0.5, -1.0]))

    class Base:
        def __init__(self):
            self.model = SimpleNamespace(training=True)
            self.optimizer = None
            self.advantages = None

        def create_optimizer(self):
            self.optimizer = torch.optim.AdamW([theta], lr=0.1, weight_decay=0.0)
            return self.optimizer

        def _generate_and_score_completions(self, inputs):
            return {"advantages": self.advantages}

    gate = local._ZeroAdvantageGate()
    trainer = local._trainer_with_zero_advantage_skip(Base, gate)()
    optimizer = trainer.create_optimizer()
    # torch's LR scheduler rebinds optimizer.step.__func__; the guard must survive it.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def update(advantages):
        trainer.advantages = torch.tensor(advantages)
        trainer._generate_and_score_completions(None)
        optimizer.zero_grad()
        (theta * torch.tensor([2.0, 3.0])).sum().backward()
        optimizer.step()
        scheduler.step()

    update([0.0, 0.0, 0.0, 0.0])
    assert gate.skipped_this_step() is True
    assert torch.equal(theta.detach(), torch.tensor([0.5, -1.0])) and len(optimizer.state) == 0
    # Hand value: AdamW's first step moves each weight by lr * g / (|g| + eps) ~ 0.1 * sign(g).
    update([0.0, 0.5, -0.5, 0.0])
    assert gate.skipped_this_step() is False
    assert theta.detach().tolist() == pytest.approx([0.4, -1.1], abs=1e-6)
    after = theta.detach().clone()
    state = {k: v.clone() for k, v in optimizer.state[theta].items()}
    update([0.0] * 4)
    assert torch.equal(theta.detach(), after)
    assert all(torch.equal(optimizer.state[theta][k], v) for k, v in state.items())
    assert (gate.scored_batches, gate.zero_batches) == (3, 2)
    trainer.model.training = False  # evaluation generations never touch the gate
    trainer.advantages = torch.tensor([1.0])
    trainer._generate_and_score_completions(None)
    assert gate.scored_batches == 3


def test_audit_rows_carry_param_delta_and_refuse_an_unsafe_skip(tmp_path):
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_B = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    model = Model()
    gate = local._ZeroAdvantageGate()
    audit = local._audit_callback(tmp_path, None, param_delta=True, zero_advantage_gate=gate)
    state, control = SimpleNamespace(global_step=0), SimpleNamespace()
    audit.on_train_begin(None, state, control, model=model)

    def step(change, advantages, grad=(0.0, 0.0)):
        gate.observe(torch.tensor(advantages))
        audit.on_step_begin(None, state, control, model=model)
        model.lora_B.grad = torch.tensor(grad)
        audit.on_pre_optimizer_step(None, state, control, model=model)
        with torch.no_grad():
            model.lora_B += torch.tensor(change)
        state.global_step += 1
        audit.on_step_end(None, state, control, model=model)

    step([3.0, 4.0], [1.0, -1.0], grad=(1.0, 0.0))  # hand value: |(3, 4)| = 5
    step([0.0, 0.0], [0.0, 0.0])
    rows = [json.loads(x) for x in (tmp_path / "optimizer_audit.jsonl").read_text().splitlines()]
    assert [r["param_delta_l2"] for r in rows] == [pytest.approx(5.0), 0.0]
    assert [r["zero_advantage_skipped"] for r in rows] == [False, True]
    assert [r["zero_advantage_skipped_steps"] for r in rows] == [0, 1]
    with pytest.raises(local.LocalVLMError, match="skipped zero-advantage step moved"):
        step([1.0, 0.0], [0.0, 0.0])
    audit.timer.interrupt()
    with pytest.raises(local.LocalVLMError, match="nonzero gradient"):
        step([0.0, 0.0], [0.0, 0.0], grad=(0.0, 1e-3))


def _ppu_grpo_run(output, *, rewards, skip=False, prompts=2, steps=2, dynamic=None,
                  outside_sleep=0.0):
    """Tiny CPU GRPO through the module's own wrappers, P prompts x G 2 per update.

    `rewards(call, n)` returns the n rewards of the call-th reward invocation. `dynamic`
    = max_draws turns on the module's dynamic sampling (P must be 1)."""
    trl = released_trl()
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback

    torch.set_num_threads(2)
    tokenizer, policy = _tiny_text_policy()
    ids = ["p0", "p1", "p2", "p3"]
    rows = {"prompt": ["a b", "b c", "c a", "a a"], "id": ids, "label": ["a"] * 4}
    dataset = Dataset.from_dict(rows)
    profile = replace(local.Profile(), num_generations=2, max_completion_length=4)
    overrides = {}
    if prompts != 1:
        overrides["prompts_per_update"] = prompts
    if skip:
        overrides["skip_zero_advantage_steps"] = True
    args = trl.GRPOConfig(**local.trainer_kwargs(
        "grpo", output, profile, device="cpu", max_steps=steps, learning_rate=5e-2, seed=42,
        save_steps=steps, beta=0.0, **({"grpo_overrides": overrides} if overrides else {})))
    exposure = local._ExposureLedger(output, ids, prompts_per_update=prompts)
    gate = local._ZeroAdvantageGate() if skip else None
    audit = local._audit_callback(output, None, exposure=exposure, param_delta=True,
                                  zero_advantage_gate=gate)
    relay = local._PromptRelay(audit.observe_prompt) if dynamic else None
    calls = []

    def reward(completions, completion_ids, id, **_):
        calls.append(list(id))
        for start in range(0, len(id), 2):
            (relay.note if relay else audit.observe_prompt)(id[start])
        return rewards(len(calls) - 1, len(id))

    class SleepOutsideTheSpan(TrainerCallback):
        def on_step_end(self, *a, **k):
            time.sleep(outside_sleep)  # after the audit closed the span: never active time

    constructor = local._trainer_with_grpo_resume(trl.GRPOTrainer)
    advantages = []

    class CaptureAdvantages(constructor):  # TRL's raw advantages per scored batch
        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if self.model.training:
                advantages.append(output["advantages"].detach().cpu().tolist())
            return output

    constructor = CaptureAdvantages
    dynamic_summary = {}
    if dynamic:
        records = [dict(zip(rows, values, strict=True)) for values in zip(*rows.values())]
        constructor = local._trainer_with_dynamic_sampling(
            constructor, relay, records, max_draws=dynamic, seed=42,
            journal=output / "dynamic_sampling.jsonl",
            journal_io_context=audit.timer.exclude_journal_io, summary=dynamic_summary)
    if gate is not None:
        constructor = local._trainer_with_zero_advantage_skip(constructor, gate)
    constructor = local._trainer_with_active_time(constructor)
    initial = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    trainer = constructor(model=policy, args=args, processing_class=tokenizer,
                          train_dataset=dataset, callbacks=[audit, SleepOutsideTheSpan()],
                          reward_funcs=reward)
    trainer.train()
    if gate is not None:  # what train_local checks after trainer.train()
        local._check_zero_gate_coverage(
            gate, audit.summary["optimizer_steps_observed"], args.num_iterations)
    lines = (output / "optimizer_audit.jsonl").read_text().splitlines()
    return SimpleNamespace(trainer=trainer, args=args, calls=calls, initial=initial,
                           rows=[json.loads(x) for x in lines], audit=audit,
                           dynamic=dynamic_summary, advantages=advantages, gate=gate)


def _inner_adamw(trainer):
    inner = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert type(inner).__name__ == "AdamW"
    return inner


def test_released_grpo_update_consumes_p_distinct_prompt_groups(tmp_path):
    # Every group of two is (1, 0): each group carries a nonzero advantage.
    run = _ppu_grpo_run(tmp_path / "ppu", rewards=lambda c, n: [float(i % 2 == 0)
                                                                for i in range(n)])
    # Hand values: P 2 x G 2 = 4 completions, generated once and accumulated per update.
    assert run.args.steps_per_generation == run.args.gradient_accumulation_steps == 4
    assert run.trainer.state.global_step == 2
    assert len(run.calls) == 2 and all(len(batch) == 4 for batch in run.calls)
    for batch in run.calls:
        assert batch[0] == batch[1] and batch[2] == batch[3] and batch[0] != batch[2]
    checkpoint = tmp_path / "ppu" / "checkpoint-2"
    entries = json.loads((checkpoint / "exposure.json").read_text())["entries"]
    assert [e["global_step"] for e in entries] == [1, 1, 2, 2]
    # 4 TRAIN prompts at 2 per update: the two updates are exactly one epoch.
    assert sorted(e["id"] for e in entries) == ["p0", "p1", "p2", "p3"]
    assert all(row["param_delta_l2"] > 0 for row in run.rows)
    final = {n: p.detach() for n, p in run.trainer.model.named_parameters() if p.requires_grad}
    # Total displacement is bounded by the summed per-update deltas (triangle inequality).
    total = math.sqrt(sum(float((final[n] - run.initial[n]).square().sum()) for n in final))
    assert 0 < total <= run.rows[0]["param_delta_l2"] + run.rows[1]["param_delta_l2"] + 1e-6


def test_released_grpo_phantom_step_moves_weights_unless_skipped(tmp_path):
    """B4 forced fire: update 1 is informative, update 2 has zero advantage everywhere."""

    def rewards(call, n):
        return [float(i % 2 == 0) for i in range(n)] if call == 0 else [1.0] * n

    # Control (legacy, no skip): AdamW steps on its momentum, so update 2 moves the weights.
    control = _ppu_grpo_run(tmp_path / "control", rewards=rewards)
    assert control.rows[1]["param_delta_l2"] > 0
    assert {int(s["step"]) for s in _inner_adamw(control.trainer).state.values()} == {2}
    # With the skip: update 2 takes no optimizer step; weights and AdamW state stay put.
    run = _ppu_grpo_run(tmp_path / "skip", rewards=rewards, skip=True)
    assert run.trainer.state.global_step == 2  # still counted against max_steps
    assert [row["zero_advantage_skipped"] for row in run.rows] == [False, True]
    assert run.rows[0]["param_delta_l2"] > 0 and run.rows[1]["param_delta_l2"] == 0.0
    assert run.rows[-1]["zero_advantage_skipped_steps"] == 1
    assert {int(s["step"]) for s in _inner_adamw(run.trainer).state.values()} == {1}
    entries = json.loads((tmp_path / "skip/checkpoint-2/exposure.json").read_text())["entries"]
    assert len(entries) == 4  # the skipped update still consumed and committed its 2 images
    # Same seed, same informative first update: the two runs agree after update 1.
    assert control.rows[0]["param_delta_l2"] == run.rows[0]["param_delta_l2"]


# (d) C5: dynamic sampling's dropped draws are charged to committed active seconds -------------

DROP_SLEEP = 0.75
OUTSIDE_SLEEP = 3.0


def test_dynamic_sampling_dropped_draws_are_charged_to_committed_active_seconds(tmp_path):
    """Each update draws flat, flat, mixed: two dropped groups per update, each sleeping
    DROP_SLEEP inside its reward (inside the draw's generate-and-score). A sleep after the
    audit closes each step must NOT be charged, which proves the clock can tell the two."""

    def rewards(call, n):
        if call % 3 == 2:
            return [1.0, 0.0]
        time.sleep(DROP_SLEEP)
        return [1.0, 1.0]

    run = _ppu_grpo_run(tmp_path / "ds", rewards=rewards, prompts=1, dynamic=3,
                        outside_sleep=OUTSIDE_SLEEP)
    assert run.dynamic == {"updates": 2, "draws": 6, "dropped_groups": 4, "exhausted_updates": 0}
    saved = json.loads((tmp_path / "ds/checkpoint-2/active_time.json").read_text())
    committed = saved["committed_active_seconds"]
    charged_drops = 4 * DROP_SLEEP  # hand value: 4 dropped draws x 0.75 s = 3.0 s
    assert committed >= charged_drops
    # The 2 x 3.0 s spent outside the span is not in it (all real tiny-model work is < 6 s).
    assert committed < charged_drops + 2 * OUTSIDE_SLEEP
    entries = json.loads((tmp_path / "ds/checkpoint-2/exposure.json").read_text())["entries"]
    assert len(entries) == 2  # only the kept prompt of each update is exposure


# arm_exec ------------------------------------------------------------------------------------


LOSS = ["--loss-type", "grpo"]  # beta-0 GRPO must name its loss aggregation (main's rule)


def _arm(**extra):
    return {"id": "m3", "mode": "grpo", "init": "I", "lr": 1e-5, "seed": 42, "beta": 0.0,
            "temperature": 1.0, "response_format": "answer_only", "num_generations": 8,
            "min_pixels": 50176, "max_pixels": 200704, "max_completion_length": 256,
            "train_manifest": "s_train_scan_seed42.jsonl", "stops": [150],
            "extra_trainer_args": list(LOSS), **extra}


def _arm_root(tmp_path):
    root = tmp_path / "root"
    adapter = root / "payload" / "adapters" / "I" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter.parent / "local_training.json").write_text(
        json.dumps({"control_identity": {"profile": {"dtype": "float32"}}}))
    return root


def test_arm_exec_passes_the_multi_image_keys_to_the_trainer(tmp_path, monkeypatch):
    arm_exec = _load_script("arm_exec")
    root = _arm_root(tmp_path)
    seen = []
    monkeypatch.setattr(local, "train_local", lambda **k: seen.append(k) or {})

    def trainer_kwargs_for(spec):
        argv = arm_exec.Arm(root, spec, tmp_path / "r").trainer_args(
            tmp_path / "m.jsonl", 1500, 150, None)
        assert argv[:2] == ["-m", "vision_rl.vlm_local_training"]
        assert local.main(argv[2:]) == 0
        return seen[-1]

    kw = trainer_kwargs_for(_arm(prompts_per_update=8, skip_zero_advantage_steps=True,
                                 features=["prompts_per_update"]))
    assert kw["grpo_overrides"] == {"prompts_per_update": 8, "skip_zero_advantage_steps": True}
    assert kw["beta"] == 0.0 and kw["profile"].num_generations == 8
    # Legacy-valued keys are not passed, so the settings identity stays legacy.
    kw = trainer_kwargs_for(_arm(prompts_per_update=1, skip_zero_advantage_steps=False))
    assert "grpo_overrides" not in kw
    assert "grpo_overrides" not in trainer_kwargs_for(_arm())
    kw = trainer_kwargs_for(_arm(skip_zero_advantage_steps=True))  # P 1 needs no feature
    assert kw["grpo_overrides"] == {"skip_zero_advantage_steps": True}
    feature = {"features": ["prompts_per_update"]}
    # The allowlisted-extras route reaches the trainer identically.
    kw = trainer_kwargs_for(_arm(**feature, extra_trainer_args=LOSS + [
        "--prompts-per-update", "8", "--skip-zero-advantage-steps", "true"]))
    assert kw["grpo_overrides"] == {"prompts_per_update": 8, "skip_zero_advantage_steps": True}
    for bad, match in (
        (_arm(mode="sft", prompts_per_update=8, **feature), "GRPO only"),
        (_arm(mode="rft", skip_zero_advantage_steps=True), "GRPO only"),
        (_arm(beta=0.04, skip_zero_advantage_steps=True), "beta 0"),
        (_arm(prompts_per_update=0), "1..32"),
        (_arm(prompts_per_update=True), "1..32"),
        (_arm(skip_zero_advantage_steps="true"), "bool"),
        (_arm(prompts_per_update=8, extra_trainer_args=["--prompts-per-update", "4"],
              **feature), "both as an arm key"),
        (_arm(extra_trainer_args=["--prompts-per-update=4"]), "not allowlisted"),
        (_arm(extra_trainer_args=["--skip", "true"]), "not allowlisted"),  # argparse prefix
        (_arm(extra_trainer_args=["--prompts-per-update", "1"]), "2..32"),  # default: omit
        (_arm(extra_trainer_args=["--skip-zero-advantage-steps", "false"]), "true"),
        (_arm(extra_trainer_args=["--prompts-per-update", "8"]), "must come together"),
        (_arm(beta=0.04, extra_trainer_args=["--skip-zero-advantage-steps", "true"]),
         "beta 0"),
        (_arm(prompts_per_update=8), "must come together"),  # the host gate would miss it
        (_arm(**feature), "must come together"),  # a "ppu" arm that would run P 1
    ):
        with pytest.raises(SystemExit, match=match):
            arm_exec.Arm(root, bad, tmp_path / "r").trainer_args(
                tmp_path / "m.jsonl", 1500, 150, None)


def test_stale_table_ppu_arms_refuse_instead_of_running_one_prompt(tmp_path):
    """docs/.../arms-rung1.json's w2-ppu4 arms name the feature but set no P. Once a host
    receipt lists prompts_per_update they become launchable, and must not train P=1."""
    arm_exec = _load_script("arm_exec")
    root = _arm_root(tmp_path)
    table = json.loads((ROOT / "docs/aircraft-exploratory-sweep-v2/arms-rung1.json").read_text())
    stale = [a for a in table["arms"] if "prompts_per_update" in a.get("features", [])]
    assert [a["id"] for a in stale] == ["w2-ppu4-grpo-W-5e-6-g8", "w2-ppu4-grpo-W-1e-5-g8"]
    for a in stale:
        with pytest.raises(SystemExit, match="must come together"):
            arm_exec.Arm(root, dict(a, init="I"), tmp_path / "r").trainer_args(
                tmp_path / "m.jsonl", 1500, 150, None)


def test_control_of_a_multi_image_arm_replays_every_committed_image(tmp_path):
    runner = _load_script("runner")
    res = tmp_path / "rl"
    res.mkdir()
    ids = [f"p{k}" for k in range(8)]  # hand value: 4 steps x P 2
    (res / "summary.json").write_text(json.dumps({"endpoints": [
        {"label": "step4", "steps": 4, "active_seconds": 50.0}]}))
    (res / "endpoint_receipt.json").write_text(json.dumps(
        {"global_step": 4, "committed_prompt_ids": ids}))
    arm = tmp_path / "arm.json"
    arm.write_text(json.dumps({"id": "rl", "prompts_per_update": 2}))
    state = {"arms": {"rl": {"status": "DONE", "results_dir": str(res),
                             "spec_path": str(arm)}}}
    ctl = {"id": "rl~ctl", "control_of": "rl", "max_steps_total": 6}
    spec = runner.control_spec(state, ctl)
    assert spec["train_ids"] == ids and spec["stops"] == [8]
    assert spec["max_steps_total"] == 12  # images: 6 RL steps x P 2
    # The same receipt from an arm that declared another P (or none) is refused.
    for declared in ({"id": "rl", "prompts_per_update": 4}, {"id": "rl"}):
        arm.write_text(json.dumps(declared))
        with pytest.raises(SystemExit, match="declares prompts_per_update"):
            runner.control_spec(state, ctl)
    arm.write_text(json.dumps({"id": "rl", "extra_trainer_args": [
        "--loss-type", "grpo", "--prompts-per-update", "2"]}))  # the extras route counts
    assert runner.control_spec(state, ctl)["max_steps_total"] == 12
    no_spec = {"arms": {"rl": {"status": "DONE", "results_dir": str(res)}}}
    with pytest.raises(SystemExit, match="declares prompts_per_update 1"):
        runner.control_spec(no_spec, ctl)
    arm.write_text(json.dumps({"id": "rl"}))
    (res / "endpoint_receipt.json").write_text(json.dumps(
        {"global_step": 4, "committed_prompt_ids": ids[:7]}))  # not P per step
    with pytest.raises(SystemExit, match="does not match"):
        runner.control_spec(state, ctl)
    (res / "endpoint_receipt.json").write_text(json.dumps(
        {"global_step": 4, "committed_prompt_ids": ids[:4]}))  # legacy P 1: unchanged
    assert runner.control_spec(state, ctl)["max_steps_total"] == 6


# Review hardening (PR #39 fresh review) ------------------------------------------------------


def test_zero_gate_coverage_refuses_a_gate_that_missed_updates():
    gate = local._ZeroAdvantageGate()
    gate.scored_batches = 4
    local._check_zero_gate_coverage(gate, 4, 1)
    local._check_zero_gate_coverage(gate, 7, 2)  # hand value: ceil(7 / 2) = 4 <= 4 <= 7
    for steps, iterations in ((5, 1), (3, 1), (9, 2)):
        with pytest.raises(local.LocalVLMError, match="did not see every update"):
            local._check_zero_gate_coverage(gate, steps, iterations)
    gate.scored_batches = 0  # e.g. a TRL change that no longer calls the hooked method
    with pytest.raises(local.LocalVLMError, match="scored 0 batches for 2"):
        local._check_zero_gate_coverage(gate, 2, 1)


def test_released_grpo_normalizes_advantages_per_group_at_p_greater_than_one(tmp_path):
    """P 2 x G 2 rewards (1, 1, 0, 0): per-group normalization gives all-zero advantages
    (each group is flat), whole-batch normalization would give (+, +, -, -). So the update
    must be skipped; under whole-batch normalization it would not be."""
    run = _ppu_grpo_run(tmp_path / "pg", rewards=lambda c, n: [1.0, 1.0, 0.0, 0.0], skip=True)
    assert run.advantages == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    assert [row["zero_advantage_skipped"] for row in run.rows] == [True, True]
    assert run.gate.scored_batches == 2
    # Mixed group next to a flat one: (1, 0, 1, 1). Per group: (+a, -a, 0, 0) with a > 0;
    # whole batch (mean 0.75) would make the flat group's advantages nonzero.
    mixed = _ppu_grpo_run(tmp_path / "mix", rewards=lambda c, n: [1.0, 0.0, 1.0, 1.0])
    for batch in mixed.advantages:
        assert batch[2] == batch[3] == 0.0
        assert batch[0] > 0 and batch[1] == -batch[0]


def test_prompts_per_update_refuses_duplicate_train_ids(tmp_path, monkeypatch):
    """The loader refuses repeats today, except on the replay path; the P>1 guard must
    hold on its own, so the loader is made to admit the repeat here."""
    data = tmp_path / "data"
    local.make_probe_data(data)
    rows = (data / "train.jsonl").read_text().splitlines()
    replay = data / "replay.jsonl"  # beside the images its relative paths name
    replay.write_text("\n".join(rows + rows[:1]) + "\n")  # one TRAIN id twice
    loader = local.load_classification
    monkeypatch.setattr(local, "load_classification",
                        lambda *a, **k: loader(*a, **{**k, "allow_repeated_ids": True}))
    common = dict(validation=None, data_spec=data / "data_spec.json",
                  model_directory=tmp_path / "absent", device="cpu", max_steps=4,
                  train_only=True)
    with pytest.raises(ValueError, match="requires distinct TRAIN ids"):
        local.train_local(replay, output=tmp_path / "o", **common,
                          grpo_overrides={"prompts_per_update": 2})
    # P 1 is unaffected by this guard: it fails later, on the absent model directory (or,
    # on a CI image without the local VLM stack, at preflight).
    with pytest.raises((FileNotFoundError, ImportError, ModuleNotFoundError, ValueError,
                        local.LocalVLMError)) as e:
        local.train_local(replay, output=tmp_path / "o1", **common)
    assert "distinct TRAIN ids" not in str(e.value)


def test_extra_flag_integers_are_ascii_digits_only():
    arm_exec = _load_script("arm_exec")
    check = arm_exec._int_between(2, 32)
    check("8")
    for bad in ("\u00b2", "\u0668", "8 ", "+8", "-8", "", "08.0"):  # superscript 2, Arabic 8
        with pytest.raises(SystemExit, match="expected an integer"):
            check(bad)
    at_least = arm_exec._int_at_least(8)
    at_least("8")
    for bad in ("\u0668", "\u00b9\u2070"):
        with pytest.raises(SystemExit, match="expected an integer"):
            at_least(bad)
