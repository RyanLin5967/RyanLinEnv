"""Exploratory sweep options: closed-set prompt, tagged think/answer format, GRPO knobs,
dynamic sampling. Every option is default-off; the legacy pins below were measured on the
pre-change trainer (origin/main 5cc4b08), never derived from the code under test."""

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision_rl import vlm_local_training as local

ROOT = Path(__file__).resolve().parents[1]


def _data_root():
    """Gitignored data lives in the main checkout; a worktree reaches it via git."""
    try:
        common = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        common = ""
    for root in (ROOT, Path(common).parent if common else None):
        if root is not None and (root / "data" / "fgvc-aircraft-2013b").is_dir():
            return root
    return ROOT


DATA_ROOT = _data_root()
REAL_SPEC = DATA_ROOT / "data" / "fgvc-aircraft-2013b" / "development" / "data_spec.json"
TRAIN_ANNOTATIONS = (
    DATA_ROOT / "runs" / "aircraft-audit" / "source" / "fgvc-aircraft-2013b" / "data"
)
TINY = {"question": "Q?", "labels": ["A", "B", "C"]}

# Measured with origin/main 5cc4b08's vlm_local_training (scratch probe, 2026-10-18).
LEGACY_ANSWER_ONLY_TINY = (
    "Q?\nAllowed labels (the complete class list):\nA\nB\nC\nReturn one label copied exactly "
    "from the allowed list on its own line. Do not add a prefix, quotes, explanations, or any "
    "text after that label."
)
LEGACY_GRPO_KWARGS_SHA = "ab1cb3b509e2e3f6ba4107adedadb803225e5ce3dd39712db591b3cba63c313c"
LEGACY_SFT_KWARGS_SHA = "c2d12d18717257c55d46a9916c8034e2ad68aec8005d767b1bf18d463acd7a7e"
# answer_only on the real spec; equals the sha of the prompt.txt that controlled400 ran with.
LEGACY_REAL_ANSWER_ONLY_SHA = "859dd366a3287a1db0fdc4ff3e3472780e974167e25f8512773cb420d60894e1"
LEGACY_REAL_THINK_SHA = "5a37649dd8f5e33a521e91f64e0a97a38034e9c850bed6dce564b07c66c6700a"
LEGACY_PROFILE = {
    "response_format": "think_then_answer",
    "dtype": "float32",
    "min_pixels": 50176,
    "max_pixels": 200704,
    "max_prompt_tokens": 2048,
    "max_completion_length": 256,
    "num_generations": 4,
    "lora_rank": 8,
    "temperature": 1.0,
}


def kwargs_sha(mode, **extra):
    kwargs = local.trainer_kwargs(
        mode,
        Path("/out"),
        local.Profile(),
        device="cpu",
        max_steps=4,
        learning_rate=1e-5,
        seed=1,
        save_steps=2,
        **extra,
    )
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest(), kwargs


# Legacy byte identity -------------------------------------------------------------------


def test_legacy_prompt_profile_and_kwargs_are_byte_identical():
    assert local.prompt_text(TINY, "answer_only") == LEGACY_ANSWER_ONLY_TINY
    assert asdict(local.Profile()) == LEGACY_PROFILE
    assert kwargs_sha("grpo")[0] == LEGACY_GRPO_KWARGS_SHA
    assert kwargs_sha("sft")[0] == LEGACY_SFT_KWARGS_SHA
    assert kwargs_sha("grpo", grpo_overrides=None)[0] == LEGACY_GRPO_KWARGS_SHA


@pytest.mark.skipif(not REAL_SPEC.is_file(), reason="needs the local FGVC data spec")
def test_legacy_real_prompts_match_the_pre_change_trainer():
    spec = json.loads(REAL_SPEC.read_text())
    for response_format, expected in (
        ("answer_only", LEGACY_REAL_ANSWER_ONLY_SHA),
        ("think_then_answer", LEGACY_REAL_THINK_SHA),
    ):
        text = local.prompt_text(spec, response_format)
        assert hashlib.sha256(text.encode()).hexdigest() == expected


def test_parse_label_source_is_untouched_for_the_named_baseline_pin():
    # The 27B named-baseline plan pins inspect.getsource(parse_label).
    import inspect

    source = inspect.getsource(local.parse_label)
    assert "response_format" not in source
    assert source.startswith(
        "def parse_label(completion: Any, labels: list[str], *, allow_reasoning: bool = True)"
    )


def test_legacy_reward_journal_and_metrics_are_unchanged(tmp_path):
    reward = local.make_reward(["A", "B"], 2, [9], tmp_path / "g.jsonl")
    assert reward(["A", "<answer>A</answer>"], ["A", "A"], ["i", "i"], [[1, 9], [2, 9]]) == [
        1.0,
        1.0,
    ]
    metrics = local.classification_metrics(
        [{"id": "i", "label": "A"}], {"i": "<answer>A</answer>"}, ["A", "B"]
    )
    assert metrics["protocol"] == "canonical-label-v2/optional-tags/greedy/pass@1"


# Closed-set prompt ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_SPEC.is_file(), reason="needs the local FGVC data spec")
@pytest.mark.parametrize("response_format", ["closed_set_answer_only", "closed_set_think_answer"])
def test_closed_set_prompt_lists_every_label_exactly_once(response_format):
    spec = json.loads(REAL_SPEC.read_text())
    text = local.prompt_text(spec, response_format)
    listing = text.split("Allowed labels by manufacturer and family:\n", 1)[1].split("\n")
    lines = [line for line in listing if " / " in line and ": " in line]
    assert len(lines) == 70
    listed = [label for line in lines for label in line.split(": ", 1)[1].split(" | ")]
    assert sorted(listed) == sorted(spec["labels"]) and len(listed) == 100
    assert "Beechcraft / King Air: Model B200" in lines
    assert "Embraer / Embraer ERJ 145: ERJ 135 | ERJ 145" in lines
    assert "closed-set task" in text and "exactly one of the 100 labels" in text
    local.Profile(response_format=response_format).validate()


def test_closed_set_refuses_a_different_label_set():
    with pytest.raises(ValueError, match="100 FGVC-Aircraft"):
        local.prompt_text(TINY, "closed_set_answer_only")
    assert not any("|" in label for _, _, group in local.AIRCRAFT_FAMILY_GROUPS for label in group)


@pytest.mark.skipif(
    not (TRAIN_ANNOTATIONS / "images_variant_train.txt").is_file(),
    reason="needs the author release TRAIN annotations",
)
def test_taxonomy_matches_the_train_annotations():
    def load(kind):
        rows = (TRAIN_ANNOTATIONS / f"images_{kind}_train.txt").read_text().splitlines()
        return dict(line.split(" ", 1) for line in rows if line)

    variant, family, manufacturer = load("variant"), load("family"), load("manufacturer")
    truth = {variant[i]: (manufacturer[i], family[i]) for i in variant}
    table = {
        label: (maker, fam)
        for maker, fam, labels in local.AIRCRAFT_FAMILY_GROUPS
        for label in labels
    }
    assert table == truth and len(table) == 100


# Tagged think/answer format -------------------------------------------------------------

LABELS = ["737-300", "Model B200", "F/A-18"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>wing <b> winglets</think>\n<answer>737-300</answer>", "737-300"),
        ("<answer>Model B200</answer>", "Model B200"),
        ("free reasoning without tags <answer> F/A-18 </answer>  ", "F/A-18"),
        ("<think>x</think><answer>King Air 350</answer>", None),  # out of vocabulary
        ("<think>x</think><answer>737-300</answer> trailing", None),
        ("<answer>737-300</answer><answer>737-300</answer>", None),
        ("<think>x</think>\n737-300", None),  # legacy bare label is not the tagged grammar
        ("<answer>737-300\n</answer>", None),
        ("", None),
    ],
)
def test_tagged_parser_scores_only_the_final_answer_tag(text, expected):
    assert local.parse_answer(text, LABELS, response_format="think_answer") == expected
    assert local.parse_answer(text, LABELS, response_format="closed_set_think_answer") == expected


def test_parse_answer_is_parse_label_for_untagged_formats():
    for text in ("737-300", "<think>a</think>\n737-300", "<answer>737-300</answer>", "x"):
        expected = local.parse_label(text, LABELS, allow_reasoning=True)
        for response_format in (None, "answer_only", "closed_set_answer_only"):
            assert local.parse_answer(text, LABELS, response_format=response_format) == expected
    with pytest.raises(ValueError, match="response_format"):
        local.parse_answer("737-300", LABELS, response_format="bogus")


def test_tagged_reward_is_answer_only_and_requires_eos(tmp_path):
    journal = tmp_path / "groups.jsonl"
    reward = local.make_reward(LABELS, 4, [9], journal, response_format="think_answer")
    texts = [
        "<think>long fuselage</think><answer>737-300</answer>",
        "<answer>737-300</answer>",  # no think: still full reward (answer-only)
        "<think>x</think><answer>Model B200</answer>",  # wrong label
        "<think>x</think><answer>737-300</answer>",  # truncated: no EOS
    ]
    ids = [[1, 9], [2, 9], [3, 9], [4, 5]]
    assert reward(texts, ["737-300"] * 4, ["i"] * 4, ids) == [1.0, 1.0, 0.0, 0.0]
    row = json.loads(journal.read_text())
    assert row["complete_think_block"] == [True, False, True, True]
    assert row["canonical_predictions"] == ["737-300", "737-300", "Model B200", None]
    assert reward.stats["mixed_groups"] == 1 and reward.stats["task_correct_with_think"] == 1


def test_tagged_metrics_and_sft_targets(tmp_path):
    rows = [{"id": "a", "label": "737-300"}, {"id": "b", "label": "F/A-18"}]
    predictions = {
        "a": {"response": "<think>t</think><answer>737-300</answer>", "terminated": True},
        "b": {"response": "F/A-18", "terminated": True},  # bare label is invalid here
    }
    metrics = local.classification_metrics(
        rows, predictions, LABELS, response_format="closed_set_think_answer"
    )
    assert metrics["accuracy"] == 0.5 and metrics["invalid_answers"] == 1
    assert metrics["protocol"].startswith("think-answer-tags-v1/answer-only/")
    profile = local.Profile(response_format="think_answer")
    assert local._format_kwargs(profile) == {"response_format": "think_answer"}
    assert local._format_kwargs(local.Profile(response_format="answer_only")) == {}
    record = local.training_records(
        [{"id": "a", "image": "x.jpg", "label": "737-300"}], TINY, "sft", "think_answer"
    )[0]
    assert record["completion"][0]["content"] == "<answer>737-300</answer>"
    assert local.parse_answer(
        record["completion"][0]["content"], LABELS, response_format="think_answer"
    ) == "737-300"
    legacy = local.training_records(
        [{"id": "a", "image": "x.jpg", "label": "737-300"}], TINY, "sft", "answer_only"
    )[0]
    assert legacy["completion"][0]["content"] == "737-300"


def test_think_answer_prompt_uses_the_flat_list():
    text = local.prompt_text(TINY, "think_answer")
    assert text.startswith("Q?\nAllowed labels (the complete class list):\nA\nB\nC\n")
    assert "<answer>LABEL</answer>" in text and "Write nothing after </answer>." in text


def test_h_assay_scores_with_the_profile_grammar():
    sys.path.insert(0, str(ROOT / "scripts" / "freshpool_launch"))
    h_assay = importlib.import_module("h_assay")
    # Free reasoning outside <think> is invalid in the legacy grammar, valid when tagged.
    completion = {"terminated": True, "response": "looks long <answer>737-300</answer>"}
    assert h_assay.answer(completion, LABELS) == (None, True)  # legacy answer_only profile
    tagged = replace(h_assay.PROFILE, response_format="think_answer")
    assert h_assay.answer(completion, LABELS, tagged) == ("737-300", True)
    assert h_assay.answer({"terminated": True, "response": "737-300"}, LABELS) == (
        "737-300",
        True,
    )


# GRPO knobs ------------------------------------------------------------------------------


def test_grpo_overrides_apply_and_validate():
    sha, kwargs = kwargs_sha(
        "grpo",
        grpo_overrides={
            "epsilon_high": 0.28,
            "loss_type": "dapo",
            "scale_rewards": "none",
            "mask_truncated_completions": False,
            "num_iterations": 2,
        },
    )
    assert sha != LEGACY_GRPO_KWARGS_SHA
    assert kwargs["epsilon_high"] == 0.28 and kwargs["loss_type"] == "dapo"
    assert kwargs["scale_rewards"] == "none" and kwargs["num_iterations"] == 2
    assert kwargs["mask_truncated_completions"] is False and kwargs["epsilon"] == 0.2
    for bad, match in (
        ({"epsilon_high": 0.1}, "epsilon_high"),
        ({"loss_type": "ppo"}, "loss_type"),
        ({"loss_type": "luspo"}, "luspo"),
        ({"scale_rewards": "std"}, "scale_rewards"),
        ({"num_iterations": 0}, "num_iterations"),
        ({"num_iterations": True}, "num_iterations"),
        ({"mask_truncated_completions": 1}, "mask_truncated"),
        ({"delta": 1.0}, "delta"),
        ({"top_entropy_quantile": 0.0}, "top_entropy"),
        ({"beta": 0.0}, "unknown GRPO knob"),
        ({}, "nonempty"),
    ):
        with pytest.raises(ValueError, match=match):
            local.validate_grpo_overrides(bad)
    assert local.validate_grpo_overrides({"loss_type": "cispo", "epsilon_high": 5.0})
    with pytest.raises(ValueError, match="only to GRPO"):
        kwargs_sha("sft", grpo_overrides={"loss_type": "dapo"})


def test_released_grpo_config_accepts_every_knob(tmp_path):
    trl = pytest.importorskip("trl")
    pytest.importorskip("torch")
    kwargs = local.trainer_kwargs(
        "grpo",
        tmp_path,
        local.Profile(),
        device="cpu",
        max_steps=1,
        learning_rate=1e-5,
        seed=1,
        save_steps=1,
        grpo_overrides={
            "epsilon": 0.2,
            "epsilon_high": 0.28,
            "delta": 1.5,
            "loss_type": "dapo",
            "scale_rewards": "batch",
            "importance_sampling_level": "sequence",
            "mask_truncated_completions": True,
            "num_iterations": 2,
            "top_entropy_quantile": 0.2,
        },
    )
    args = trl.GRPOConfig(**kwargs)
    assert (args.epsilon_high, args.loss_type, args.num_iterations) == (0.28, "dapo", 2)
    assert args.steps_per_generation == args.gradient_accumulation_steps == 4


def test_train_local_refuses_knobs_outside_grpo_or_with_screening(tmp_path):
    common = dict(
        train=tmp_path / "t.jsonl",
        validation=None,
        data_spec=tmp_path / "s.json",
        output=tmp_path / "o",
    )
    with pytest.raises(ValueError, match="only to GRPO"):
        local.train_local(**common, mode="sft", grpo_overrides={"loss_type": "dapo"})
    with pytest.raises(ValueError, match="only to GRPO"):
        local.train_local(**common, mode="sft", dynamic_sampling_max_draws=4)
    with pytest.raises(ValueError, match="2..64"):
        local.train_local(**common, dynamic_sampling_max_draws=1)
    with pytest.raises(ValueError, match="screening cannot combine"):
        local.train_local(
            **common,
            dynamic_sampling_max_draws=4,
            screen_journal=tmp_path / "o" / "screen.jsonl",
        )
    with pytest.raises(ValueError, match="screening cannot combine"):
        local.train_local(
            **common,
            grpo_overrides={"num_iterations": 2},
            screen_journal=tmp_path / "o" / "screen.jsonl",
        )


def test_cli_routes_knobs_only_when_given(monkeypatch, capsys):
    seen = {}

    def train(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(local, "train_local", train)
    base = ["--train", "t", "--data-spec", "s", "--output", "o", "--train-only"]
    assert local.main(base) == 0
    assert "grpo_overrides" not in seen and "dynamic_sampling_max_draws" not in seen
    assert local.main(
        base
        + [
            "--epsilon-high",
            "0.28",
            "--loss-type",
            "dapo",
            "--mask-truncated-completions",
            "false",
            "--num-iterations",
            "2",
            "--dynamic-sampling-max-draws",
            "4",
            "--response-format",
            "closed_set_think_answer",
        ]
    ) == 0
    assert seen["grpo_overrides"] == {
        "epsilon_high": 0.28,
        "loss_type": "dapo",
        "mask_truncated_completions": False,
        "num_iterations": 2,
    }
    assert seen["dynamic_sampling_max_draws"] == 4
    assert seen["profile"].response_format == "closed_set_think_answer"
    with pytest.raises(SystemExit):
        local.main(base + ["--mask-truncated-completions", "maybe"])
    capsys.readouterr()


def test_inference_cli_refuses_training_knobs(monkeypatch, capsys):
    monkeypatch.setattr(local, "predict_train_subset", lambda *a, **k: pytest.fail("ran"))
    argv = ["--predict-train", "m", "--data-spec", "s", "--output", "o", "--model-directory", "d"]
    with pytest.raises(SystemExit):
        local.main(argv + ["--loss-type", "dapo"])
    with pytest.raises(SystemExit):
        local.main(argv + ["--dynamic-sampling-max-draws", "4"])
    capsys.readouterr()


# Dynamic sampling and policy reuse ---------------------------------------------------------


def test_replacement_index_is_deterministic_and_spread():
    draws = [local.dynamic_replacement_index(42, 7, d, 2919) for d in range(1, 9)]
    assert draws == [local.dynamic_replacement_index(42, 7, d, 2919) for d in range(1, 9)]
    assert len(set(draws)) > 1
    assert draws != [local.dynamic_replacement_index(43, 7, d, 2919) for d in range(1, 9)]
    with pytest.raises(ValueError):
        local.dynamic_replacement_index(42, 7, 0, 10)


def make_dynamic(tmp_path, rewards_by_id, max_draws=3):
    torch = pytest.importorskip("torch")
    observed, generated = [], []
    relay = local._PromptRelay(observed.append)

    class Base:
        def __init__(self):
            self.model = SimpleNamespace(training=True)
            self.state = SimpleNamespace(global_step=4)

        def _generate_and_score_completions(self, inputs):
            identifier = inputs[0]["id"]
            generated.append(identifier)
            relay.note(identifier)  # what make_reward does once per scored group
            rewards = torch.tensor(rewards_by_id[identifier], dtype=torch.float32)
            return {"advantages": rewards - rewards.mean(), "id": identifier}

    records = [{"id": f"r{i}"} for i in range(50)]
    summary = {}
    trainer = local._trainer_with_dynamic_sampling(
        Base,
        relay,
        records,
        max_draws=max_draws,
        seed=42,
        journal=tmp_path / "dynamic.jsonl",
        summary=summary,
    )()
    return trainer, observed, generated, summary, relay


def test_dynamic_sampling_drops_all_same_groups(tmp_path):
    first = local.dynamic_replacement_index(42, 5, 1, 50)
    second = local.dynamic_replacement_index(42, 5, 2, 50)
    rewards = {f"r{i}": [1.0, 1.0, 1.0, 1.0] for i in range(50)}
    rewards["start"] = [0.0, 0.0, 0.0, 0.0]
    rewards[f"r{second}"] = [1.0, 0.0, 0.0, 1.0]
    if first == second:
        pytest.skip("replacement draws coincide for this seed")
    trainer, observed, generated, summary, _ = make_dynamic(tmp_path, rewards)
    output = trainer._generate_and_score_completions([{"id": "start"}] * 4)
    assert generated == ["start", f"r{first}", f"r{second}"]
    assert output["id"] == f"r{second}"
    assert observed == [f"r{second}"]  # only the kept prompt enters the exposure ledger
    lines = [json.loads(x) for x in (tmp_path / "dynamic.jsonl").read_text().splitlines()]
    assert [(x["id"], x["informative"], x["kept"]) for x in lines] == [
        ("start", False, False),
        (f"r{first}", False, False),
        (f"r{second}", True, True),
    ]
    assert lines[-1]["exhausted"] is False and {x["global_step"] for x in lines} == {5}
    assert summary == {"updates": 1, "draws": 3, "dropped_groups": 2, "exhausted_updates": 0}


def test_dynamic_sampling_keeps_informative_first_draw_and_journals_exhaustion(tmp_path):
    rewards = {f"r{i}": [0.0] * 4 for i in range(50)}
    rewards["mixed"] = [1.0, 0.0, 0.0, 0.0]
    rewards["flat"] = [1.0] * 4
    trainer, observed, generated, summary, _ = make_dynamic(tmp_path, rewards, max_draws=2)
    trainer._generate_and_score_completions([{"id": "mixed"}] * 4)
    assert generated == ["mixed"] and observed == ["mixed"]
    trainer._generate_and_score_completions([{"id": "flat"}] * 4)
    assert len(generated) == 3 and observed[-1] == generated[-1]
    last = json.loads((tmp_path / "dynamic.jsonl").read_text().splitlines()[-1])
    assert last["exhausted"] is True and last["kept"] is True
    assert summary["exhausted_updates"] == 1 and summary["updates"] == 2


def test_dynamic_sampling_passes_evaluation_through(tmp_path):
    trainer, observed, generated, _, _ = make_dynamic(tmp_path, {"e": [0.0] * 4})
    trainer.model.training = False
    trainer._generate_and_score_completions([{"id": "e"}] * 4)
    assert generated == ["e"] and observed == ["e"]
    assert not (tmp_path / "dynamic.jsonl").exists()


def test_policy_reuse_observes_buffered_prompt_on_reuse_updates():
    observed = []
    relay = local._PromptRelay(observed.append)

    class Base:
        def __init__(self):
            self.model = SimpleNamespace(training=True)
            self.args = SimpleNamespace(steps_per_generation=4)
            self.num_iterations = 2
            self._step = 0
            self._buffered_inputs = None

        def _prepare_inputs(self, batch):
            if self._step % 8 == 0 or self._buffered_inputs is None:
                relay.note(batch)  # generation scores one group via the reward
                self._buffered_inputs = [batch]
            self._step += 1
            return self._buffered_inputs[0]

    trainer = local._trainer_with_policy_reuse(Base, relay)()
    for update, prompt in enumerate(["p0", "p0", "p1", "p1"]):
        for _ in range(4):
            trainer._prepare_inputs(prompt)
    # one observation per optimizer update: generation, then reuse
    assert observed == ["p0", "p0", "p1", "p1"]


def test_prompt_relay_refuses_double_deferral_and_empty_commit():
    relay = local._PromptRelay(lambda _: None)
    relay.defer()
    with pytest.raises(local.LocalVLMError):
        relay.defer()
    with pytest.raises(local.LocalVLMError):
        relay.commit()
