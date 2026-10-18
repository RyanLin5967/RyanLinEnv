"""The signal-limit protocol revises the activated freshpool protocol only where it says.

Both files are loaded from disk; expected values are the committed parent bytes, never
anything derived from the revision itself.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PARENT = DOCS / "aircraft-rl-protocol-freshpool-v1-activated.json"
REVISION = DOCS / "aircraft-rl-protocol-signal-limit-v1.json"
BYTE_IDENTICAL = ("substantial_gain_gates", "strong_baseline")
UNCHANGED = ("evaluation_and_seal", "mechanism_gate", "conditional_common_warm_start",
             "resources", "primary_sources")
METADATA = {"protocol_id", "status", "activated", "created_date", "author_agent", "revises",
            "root_review_status", "activation"}
CHANGED = {
    ("data", "shots_per_class"),
    ("data", "train_images"),
    ("data", "train_membership"),
    ("data", "pool_split"),
    ("data", "claim_scope"),
    ("actor_and_objective", "initial_policy"),
    ("actor_and_objective", "algorithm", "loss_type"),
    ("actor_and_objective", "algorithm", "reward_scaling"),
    ("actor_and_objective", "algorithm", "beta"),
    ("actor_and_objective", "algorithm", "reference"),
    ("actor_and_objective", "sampling", "completions_per_prompt"),
    ("actor_and_objective", "optimizer", "learning_rate_selection"),
    ("actor_and_objective", "reward"),
    ("real_learning_and_resume_gate", "experiment"),
    ("exposure_and_staging",),
    ("supervised_controls",),
    ("activation_prerequisites",),
}
KEPT_INSIDE_CHANGED = (
    ("exposure_and_staging", "primary_seed"),
    ("exposure_and_staging", "conditional_replication_seeds"),
    ("supervised_controls", "strongest_definition"),
    ("supervised_controls", "time_accounting"),
    ("supervised_controls", "wall_time_readiness"),
    ("supervised_controls", "strong_answer_only_controls", "learning_rates"),
    ("activation_prerequisites", 0),
)


def load(path):
    raw = path.read_text(encoding="utf-8")
    return raw, json.loads(raw)


def block_text(raw, key, value):
    """The exact bytes of a top-level block in a one-space-indented file."""
    text = f' "{key}": ' + json.dumps(value, indent=1, ensure_ascii=False).replace("\n", "\n ")
    assert raw.count(text) == 1, f"{key} block not found verbatim in file"
    return text


def leaves(obj, path=()):
    if isinstance(obj, dict) and not obj:
        yield path, {}
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, (*path, key))
    else:
        yield path, obj


def get(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def test_gate_and_baseline_blocks_are_byte_identical():
    parent_raw, parent = load(PARENT)
    revision_raw, revision = load(REVISION)
    for key in BYTE_IDENTICAL:
        assert revision[key] == parent[key]
        assert (block_text(revision_raw, key, revision[key]).encode()
                == block_text(parent_raw, key, parent[key]).encode())
    for key in UNCHANGED:
        assert revision[key] == parent[key], key


def test_only_declared_fields_change_and_every_declared_change_happened():
    _, parent = load(PARENT)
    _, revision = load(REVISION)
    before = dict(leaves({k: v for k, v in parent.items() if k not in METADATA}))
    after = dict(leaves({k: v for k, v in revision.items() if k not in METADATA}))
    missing = object()
    changed = {path for path in before.keys() | after.keys()
               if before.get(path, missing) != after.get(path, missing)}
    allowed = {path for path in changed if any(path[:len(c)] == c for c in CHANGED)}
    assert changed == allowed, sorted(changed - allowed)
    for declared in CHANGED:
        assert any(path[:len(declared)] == declared for path in changed), declared
    for path in KEPT_INSIDE_CHANGED:
        assert get(revision, path) == get(parent, path), path


def test_revision_identity_and_pinned_salts():
    parent_raw, parent = load(PARENT)
    _, revision = load(REVISION)
    assert revision["protocol_id"] == "fgvc-aircraft-signal-limit-v1"
    assert revision["protocol_id"] != parent["protocol_id"]
    assert revision["activated"] is False and "activation" not in revision
    assert revision["revises"]["path"] == "docs/aircraft-rl-protocol-freshpool-v1-activated.json"
    assert revision["revises"]["sha256"] == hashlib.sha256(parent_raw.encode()).hexdigest()
    assert revision["revises"]["study"] == "docs/aircraft-rl-next-study-v1.json"
    # H manifests keep the freshpool salt; the S_a / S_b split has its own pinned literal.
    assert revision["data"]["train_holdout"] == parent["data"]["train_holdout"]
    assert revision["data"]["pool_split"]["hash_protocol_id"] == "fgvc-aircraft-signal-limit-v1"
    algorithm = revision["actor_and_objective"]["algorithm"]
    assert (algorithm["loss_type"], algorithm["reward_scaling"], algorithm["beta"]) == (
        "dr_grpo", "none", 0.0)
    assert revision["exposure_and_staging"]["N_accepted_updates"] == 300
    assert revision["exposure_and_staging"]["stage_a_updates"] == 600


def test_prepare_script_reads_the_committed_split_salt():
    spec = importlib.util.spec_from_file_location(
        "prepare_aircraft_freshpool", ROOT / "scripts/prepare_aircraft_freshpool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, revision = load(REVISION)
    assert module.pool_split_salt(revision) == "fgvc-aircraft-signal-limit-v1"
