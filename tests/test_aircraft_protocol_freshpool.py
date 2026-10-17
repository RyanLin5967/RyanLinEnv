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
BYTE_IDENTICAL = ("substantial_gain_gates", "strong_baseline")
METADATA = {"protocol_id", "status", "created_date", "author_agent", "revises"}
CHANGED = {
    ("data", "train_membership"),
    ("data", "shots_per_class"),
    ("data", "train_holdout"),
    ("supervised_controls", "strong_answer_only_controls", "selection"),
}


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
    if isinstance(obj, dict):
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
