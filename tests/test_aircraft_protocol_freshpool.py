"""The freshpool protocol revises the candidate only where the study says it may.

Both files are loaded from disk; expected values are the committed candidate bytes,
never anything derived from the revision itself.
"""

import hashlib
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
CANDIDATE = DOCS / "aircraft-rl-protocol-candidate-v1.json"
FRESHPOOL = DOCS / "aircraft-rl-protocol-freshpool-v1.json"
STUDY = DOCS / "aircraft-freshpool-study-v1.json"
STUDY_MD = DOCS / "aircraft-freshpool-study-v1.md"
# Pinned by root decision (d): the holdout salt is this literal, whatever protocol_id becomes.
HOLDOUT_SALT_LITERAL = "fgvc-aircraft-freshpool-screened-grpo-v1"
BYTE_IDENTICAL = ("substantial_gain_gates", "strong_baseline")
METADATA = {"protocol_id", "status", "created_date", "author_agent", "revises"}
CHANGED = {
    ("data", "train_membership"),
    ("data", "shots_per_class"),
    ("data", "train_holdout"),
    ("data", "train_images"),
    ("data", "claim_scope"),
    ("actor_and_objective", "initial_policy"),
    ("actor_and_objective", "algorithm", "clip_epsilon_status"),
    ("actor_and_objective", "optimizer"),
    ("actor_and_objective", "reasoning_instruction_v2"),
    ("actor_and_objective", "instruction"),
    ("real_learning_and_resume_gate", "before"),
    ("real_learning_and_resume_gate", "actual_task_data"),
    ("real_learning_and_resume_gate", "experiment"),
    ("exposure_and_staging",),
    ("supervised_controls",),
    ("evaluation_and_seal", "validation"),
    ("evaluation_and_seal", "test_freeze_before_access"),
    ("resources", "not_measured"),
    ("resources", "proposed_first_seed_limits"),
    ("activation_prerequisites",),
}
# Inside a wholly revised block, these sub-fields still must equal the candidate.
KEPT_INSIDE_CHANGED = (
    ("supervised_controls", "strongest_definition"),
    ("supervised_controls", "time_accounting"),
    ("supervised_controls", "wall_time_readiness"),
    ("supervised_controls", "strong_answer_only_controls", "learning_rates"),
    ("exposure_and_staging", "primary_seed"),
    ("exposure_and_staging", "conditional_replication_seeds"),
    ("activation_prerequisites", 0),
)


def load(path):
    raw = path.read_text(encoding="utf-8")
    return raw, json.loads(raw)


def block_text(raw, key, value):
    """The exact bytes of a top-level block as it sits in the file."""
    text = f'  "{key}": ' + json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n  ")
    assert raw.count(text) == 1, f"{key} block not found verbatim in file"
    return text


def test_gate_and_baseline_blocks_are_byte_identical():
    cand_raw, cand = load(CANDIDATE)
    fresh_raw, fresh = load(FRESHPOOL)
    for key in BYTE_IDENTICAL:
        assert key in cand and key in fresh
        assert fresh[key] == cand[key]
        assert (
            block_text(fresh_raw, key, fresh[key]).encode()
            == block_text(cand_raw, key, cand[key]).encode()
        )


def leaves(obj, path=()):
    """Yield (path, value) for every leaf; an empty dict is itself a leaf, so adding or
    removing a key whose value is {} changes the leaf set."""
    if isinstance(obj, dict) and not obj:
        yield path, {}
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, (*path, key))
    else:
        yield path, obj


def test_only_declared_fields_change():
    _, cand = load(CANDIDATE)
    _, fresh = load(FRESHPOOL)
    cand_leaves = dict(leaves({k: v for k, v in cand.items() if k not in METADATA}))
    fresh_leaves = dict(leaves({k: v for k, v in fresh.items() if k not in METADATA}))
    changed = {
        path
        for path in cand_leaves.keys() | fresh_leaves.keys()
        if cand_leaves.get(path, object()) != fresh_leaves.get(path, object())
    }
    allowed = {path for path in changed if any(path[: len(c)] == c for c in CHANGED)}
    assert changed == allowed, sorted(changed - allowed)
    # Every declared change actually happened.
    for declared in CHANGED:
        assert any(path[: len(declared)] == declared for path in changed), declared


def test_revision_identity():
    cand_raw, cand = load(CANDIDATE)
    _, fresh = load(FRESHPOOL)
    assert fresh["protocol_id"] != cand["protocol_id"]
    assert fresh["activated"] is False
    assert fresh["revises"]["path"] == "docs/aircraft-rl-protocol-candidate-v1.json"
    assert fresh["revises"]["sha256"] == hashlib.sha256(cand_raw.encode("utf-8")).hexdigest()
    holdout = fresh["data"]["train_holdout"]
    assert holdout["H_select"]["ranks"] == [1, 3] and holdout["H_gate"]["ranks"] == [2, 4]


def test_leaves_sees_added_or_removed_empty_dict_keys():
    assert list(leaves({"a": {}})) == [(("a",), {})]
    assert list(leaves({"a": {"b": {}}})) == [(("a", "b"), {})]
    assert dict(leaves({"x": 1, "e": {}})) != dict(leaves({"x": 1}))
    assert dict(leaves({"x": {"e": {}}})) != dict(leaves({"x": {}}))


def pick(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def test_fields_kept_inside_revised_blocks_equal_the_candidate():
    _, cand = load(CANDIDATE)
    _, fresh = load(FRESHPOOL)
    for path in KEPT_INSIDE_CHANGED:
        assert pick(fresh, path) == pick(cand, path), path


def test_study_design_is_committed_and_pinned_by_hash():
    _, fresh = load(FRESHPOOL)
    revises = fresh["revises"]
    assert revises["study"] == "docs/aircraft-freshpool-study-v1.json"
    assert revises["study_markdown"] == "docs/aircraft-freshpool-study-v1.md"
    assert revises["study_sha256"] == hashlib.sha256(STUDY.read_bytes()).hexdigest()
    assert revises["study_markdown_sha256"] == hashlib.sha256(STUDY_MD.read_bytes()).hexdigest()
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    assert study["study_id"] == "aircraft-freshpool-screened-grpo-v1"


def test_holdout_salt_is_a_pinned_literal_not_the_protocol_id_field():
    _, fresh = load(FRESHPOOL)
    holdout = fresh["data"]["train_holdout"]
    assert holdout["hash_protocol_id"] == HOLDOUT_SALT_LITERAL
    assert "hash_protocol_id" in holdout["hash"]


def test_revision_encodes_the_study_profile_and_budgets():
    _, fresh = load(FRESHPOOL)
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    rl_arm = next(arm for arm in study["arms"] if arm["id"] == "RL(lr)")
    actor = fresh["actor_and_objective"]
    assert "learning_rate" not in actor["optimizer"]
    assert actor["optimizer"]["learning_rate_grid"] == rl_arm["lrs"] == [1e-05, 5e-05, 0.0001]
    assert actor["optimizer"]["distinct_prompts_per_optimizer_update"] == 1
    assert "reasoning_instruction_v2" not in actor
    assert actor["instruction"]["id"] == "answer_only"
    staging = fresh["exposure_and_staging"]
    assert staging["N_accepted_updates"] == study["fixed_profile"]["N_accepted_updates"] == 600
    assert staging["optimizer_update_unit"]["distinct_prompts_per_optimizer_update"] == 1
    assert [s["cumulative_accepted_updates"] for s in staging["stages"]] == [150, 600, 600]
    assert "1.25x" in staging["arm_caps"] and "never quoted" in staging["arm_caps"]
    controls = fresh["supervised_controls"]
    assert set(controls["arms"]) == {"SFT-E", "SFT-T", "EG-SFT"}
    # The retired 400/1,200 presentation budgets are gone from every revised block.
    for block in ("exposure_and_staging", "supervised_controls"):
        text = json.dumps(fresh[block])
        for stale in ("1,200", "1200", "400 IDs", "same 400", 'presentations": 400'):
            assert stale not in text, (block, stale)
        assert "4-prompt" not in text and "fourshot SFT" not in text
    assert fresh["data"]["claim_scope"] != load(CANDIDATE)[1]["data"]["claim_scope"]
    assert "not a 4-shot claim" in fresh["data"]["claim_scope"]
    assert isinstance(fresh["data"]["train_images"], str)


def test_selection_uses_h_select_and_val_only_for_pool_and_final_gates():
    _, fresh = load(FRESHPOOL)
    seal = fresh["evaluation_and_seal"]
    assert "H-select only" in seal["validation"]
    assert "checkpoint and learning-rate selection" not in seal["validation"]
    freeze = " ".join(seal["test_freeze_before_access"])
    assert "H-select only" in freeze
    assert "exposure checkpoint and strongest SFT comparator using validation only" not in freeze
    assert "selected on full validation" not in json.dumps(fresh["supervised_controls"])
