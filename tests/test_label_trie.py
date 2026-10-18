"""Label trie (signal-limit study): hand-built tokenizer fixtures, hand-written expected tries,
forced fires. Every expected value below is written by hand, never computed by the subject."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import label_trie as lt  # noqa: E402

SCRIPT = SCRIPTS / "build_label_trie.py"

# Hand-built byte vocab. "AB-1" spells as [7], [3,4,5] and [1,2,4,5]; "A" as [1] only.
VOCAB = {b"A": 1, b"B": 2, b"AB": 3, b"-": 4, b"1": 5, b"Z": 6, b"AB-1": 7}
CANONICAL = {"AB-1": [7], "A": [1]}
EOS = 99


def script():
    spec = importlib.util.spec_from_file_location("build_label_trie", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_trie(slack=2):
    trie = lt.build_trie(["AB-1", "A"], VOCAB, CANONICAL, [EOS], slack=slack)
    trie["sha256"] = lt.trie_sha256(trie)
    return trie


def test_byte_level_alphabet_matches_gpt2_table():
    table = lt.bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert table[ord("A")] == "A"
    assert table[32] == "Ġ" and table[10] == "Ċ" and table[0] == "Ā"
    assert table[127] == chr(289) and table[173] == chr(323)
    assert lt.byte_vocab({"Ġ1": 5, "AB": 3}) == {b" 1": 5, b"AB": 3}
    with pytest.raises(ValueError):
        lt.byte_vocab({"一": 1})  # not in the byte-level alphabet


def test_enumeration_and_counts_by_hand():
    assert lt.enumerate_tokenizations(b"AB-1", VOCAB, 4) == [(1, 2, 4, 5), (3, 4, 5), (7,)]
    assert lt.enumerate_tokenizations(b"AB-1", VOCAB, 3) == [(3, 4, 5), (7,)]
    assert lt.count_tokenizations(b"AB-1", VOCAB) == {1: 1, 3: 1, 4: 1}
    assert lt.count_tokenizations(b"AC", VOCAB) == {}


def test_hand_written_trie_and_prune_record():
    trie = fixture_trie()
    assert trie["eos_ids"] == [99]
    assert trie["entries"]["AB-1"] == {
        "canonical": [7], "max_tokens": 3, "paths": [[3, 4, 5], [7]],
        "kept": 2, "pruned": 1, "pruned_by_length": {"4": 1},
    }
    assert trie["entries"]["A"] == {
        "canonical": [1], "max_tokens": 3, "paths": [[1]],
        "kept": 1, "pruned": 0, "pruned_by_length": {},
    }
    assert trie["path_count"] == 3
    index = lt.LabelTrie(trie)
    assert index.allowed([]) == {1, 3, 7}
    assert index.allowed([1]) == {99}  # [1,2,4,5] was pruned, so "A" can only end
    assert index.allowed([3]) == {4}
    assert index.allowed([3, 4]) == {5}
    assert index.allowed([3, 4, 5]) == {99}
    assert index.allowed([7, 99]) == frozenset()
    assert index.label([3, 4, 5, 99]) == "AB-1" and index.label([1, 99]) == "A"
    assert index.label([7]) is None  # no EOS: not a full path
    # slack 3 keeps the 4-token spelling
    wide = lt.LabelTrie(fixture_trie(slack=3))
    assert wide.allowed([1]) == {2, 99}


def test_two_eos_ids_double_the_full_paths():
    trie = lt.build_trie(["A"], VOCAB, {"A": [1]}, [99, 98])
    assert trie["eos_ids"] == [98, 99] and trie["path_count"] == 2
    assert lt.LabelTrie(trie).allowed([1]) == {98, 99}


def test_build_refuses_bad_inputs():
    with pytest.raises(ValueError, match="does not spell"):
        lt.build_trie(["A"], VOCAB, {"A": [2]}, [EOS])
    with pytest.raises(ValueError, match="EOS"):
        lt.build_trie(["A"], VOCAB, {"A": [1]}, [5])  # EOS inside the ordinary vocab
    with pytest.raises(ValueError, match="EOS"):
        lt.build_trie(["A"], VOCAB, {"A": [1]}, [])
    with pytest.raises(ValueError, match="canonical"):
        lt.build_trie(["A", "AB-1"], VOCAB, {"A": [1]}, [EOS])


def test_forced_fire_planted_off_trie_token_is_masked():
    index = lt.LabelTrie(fixture_trie())
    row = [0.0] * 100
    row[6] = 50.0  # planted: "Z" is in the vocab but on no label path
    masked = index.mask_row([], row)
    assert masked[6] == -math.inf
    assert [i for i, v in enumerate(masked) if v != -math.inf] == [1, 3, 7]
    assert max(range(100), key=masked.__getitem__) in {1, 3, 7}
    after_a = index.mask_row([1], row)
    assert [i for i, v in enumerate(after_a) if v != -math.inf] == [99]
    with pytest.raises(ValueError, match="off the trie"):
        index.mask_row([6], row)
    with pytest.raises(ValueError, match="outside the logits row"):
        index.mask_row([1], [0.0] * 50)


def test_forced_fire_coverage_refuses():
    index = lt.LabelTrie(fixture_trie())
    good = [("AB-1", [7, 99]), ("AB-1", [3, 4, 5, 99]), ("A", [1, 99])]
    assert lt.coverage(index, good, 3) == {"valid_completions": 3, "in_trie": 3, "expected": 3}
    with pytest.raises(ValueError, match="off T"):  # a pruned tokenisation, absent from T
        lt.coverage(index, good + [("AB-1", [1, 2, 4, 5, 99])], 4)
    with pytest.raises(ValueError, match="off T"):  # in T, but spells another label
        lt.coverage(index, [("AB-1", [1, 99])], 1)
    with pytest.raises(ValueError, match="off T"):  # no EOS
        lt.coverage(index, [("A", [1])], 1)
    with pytest.raises(ValueError, match="expected 4"):
        lt.coverage(index, good, 4)
    with pytest.raises(ValueError, match="zero"):
        lt.coverage(index, [], 0)


def test_sha_is_canonical_json_and_tamper_refuses():
    doc = {"b": [1, 2], "a": "x", "sha256": "ignored", "provenance": {"p": 1}}
    assert lt.trie_sha256(doc) == hashlib.sha256(b'{"a":"x","b":[1,2]}').hexdigest()
    trie = fixture_trie()
    trie["entries"]["A"]["paths"].append([6])
    with pytest.raises(ValueError, match="sha256"):
        lt.LabelTrie(trie)


def test_upper_bound_by_hand():
    per_class = {"x": {"n": 4, "correct": 2, "invalid": 1},
                 "y": {"n": 2, "correct": 0, "invalid": 0}}
    assert lt.upper_bound_macro(per_class) == pytest.approx(0.375)
    with pytest.raises(ValueError):
        lt.upper_bound_macro({"x": {"n": 1, "correct": 1, "invalid": 1}})


TRAIN_ROWS = [
    {"image_id": "1", "split": "train", "variant": "737-300", "family": "Boeing 737",
     "manufacturer": "Boeing"},
    {"image_id": "2", "split": "train", "variant": "737-400", "family": "Boeing 737",
     "manufacturer": "Boeing"},
    {"image_id": "3", "split": "train", "variant": "A320", "family": "A320",
     "manufacturer": "Airbus"},
    {"image_id": "4", "split": "train", "variant": "737-300", "family": "Boeing 737",
     "manufacturer": "Boeing"},
]
LABELS = ["737-300", "737-400", "A320"]


def test_taxonomy_train_only_by_hand():
    tax = lt.taxonomy(TRAIN_ROWS, LABELS)
    assert tax == {
        "train_rows": 4,
        "variants": {"737-300": {"family": "Boeing 737", "manufacturer": "Boeing"},
                     "737-400": {"family": "Boeing 737", "manufacturer": "Boeing"},
                     "A320": {"family": "A320", "manufacturer": "Airbus"}},
        "families": {"A320": ["A320"], "Boeing 737": ["737-300", "737-400"]},
        "manufacturers": {"Airbus": ["A320"], "Boeing": ["737-300", "737-400"]},
    }
    with pytest.raises(ValueError, match="non-TRAIN"):
        lt.taxonomy(TRAIN_ROWS + [{**TRAIN_ROWS[0], "image_id": "9", "split": "val"}], LABELS)
    with pytest.raises(ValueError, match="two parents"):
        lt.taxonomy(TRAIN_ROWS + [{**TRAIN_ROWS[0], "family": "Other"}], LABELS)
    with pytest.raises(ValueError, match="cover"):
        lt.taxonomy(TRAIN_ROWS, LABELS + ["DC-3"])


def test_confusion_map_by_hand():
    tax = lt.taxonomy(TRAIN_ROWS, LABELS)
    observed = [("A320", "737-300"), ("A320", None), ("A320", "A320"), ("737-300", "737-300")]
    assert lt.confusion_map(tax, observed, LABELS) == {
        "737-300": {"family": ["737-400"], "observed": [], "confusers": ["737-400"]},
        "737-400": {"family": ["737-300"], "observed": [], "confusers": ["737-300"]},
        "A320": {"family": [], "observed": ["737-300"], "confusers": ["737-300"]},
    }
    with pytest.raises(ValueError):
        lt.confusion_map(tax, [("A320", "Concorde")], LABELS)


def test_script_confusion_refuses_non_train_p0_image(tmp_path):
    builder = script()
    tax = lt.taxonomy(TRAIN_ROWS, LABELS)
    seed = tmp_path / "rollout-seed42-00"
    seed.mkdir()
    group = {"id": "1", "label": "A320", "canonical_predictions": ["737-300", None]}
    (seed / "reward_groups.jsonl").write_text(json.dumps(group) + "\n")
    out = builder.build_confusion(tmp_path, tax, {"1"}, LABELS)
    assert out["map"]["A320"]["observed"] == ["737-300"]
    assert out["sources"]["groups"] == 1 and out["sources"]["completions"] == 2
    with pytest.raises(SystemExit, match="not a TRAIN image"):
        builder.build_confusion(tmp_path, tax, {"2"}, LABELS)
    with pytest.raises(SystemExit, match="no P0"):
        builder.build_confusion(tmp_path / "empty", tax, {"1"}, LABELS)


def _comparison(per_class, macro, invalid):
    arm = {"per_class": per_class, "mean_class_accuracy": macro,
           "invalid_answers_including_missing": invalid}
    return {"versus_common_initialization": {
        name: {"candidate": arm, "comparator": arm} for name in script().ARMS if name != "common"
    }}


def test_script_upper_bound_table_by_hand():
    builder = script()
    gold = [{"id": "r1", "label": "x"}, {"id": "r2", "label": "x"}, {"id": "r3", "label": "y"}]
    # r1 right, r2 invalid, r3 wrong-but-valid: macro (1/2 + 0)/2; bound (2/2 + 0)/2
    preds = {"r1": ("x", [1]), "r2": (None, [2]), "r3": ("x", [1])}
    completions = {arm: preds for arm in builder.ARMS}
    per_class = {"x": {"n": 2, "correct": 1, "invalid": 1}, "y": {"n": 1, "correct": 0,
                                                                  "invalid": 0}}
    table = builder.upper_bound_table(_comparison(per_class, 0.25, 1), completions, gold,
                                      ["x", "y"])
    assert set(table) == set(builder.ARMS)
    assert table["common"] == {"macro": 0.25, "invalid_or_missing": 1,
                               "upper_bound_macro": 0.5, "headroom_pp": 25.0}
    with pytest.raises(SystemExit, match="does not reproduce"):
        builder.upper_bound_table(_comparison(per_class, 0.3, 1), completions, gold, ["x", "y"])
    lying = {"x": {"n": 2, "correct": 1, "invalid": 0}, "y": per_class["y"]}
    with pytest.raises(SystemExit, match="upper bound"):
        builder.upper_bound_table(_comparison(lying, 0.25, 1), completions, gold, ["x", "y"])


# ---------------------------------------------------------------- trainer loaders agree
# The trainer (src/vision_rl/vlm_local_training.py) reads the builder's committed files; these
# tests pin that the two sides read one format and one trie, so the cce_sft mask and the
# constrained-greedy processor see exactly the T the coverage check certified.

AUDIT = Path(__file__).resolve().parents[1] / "data" / "fgvc-aircraft-2013b" / "audit"


def _local():
    return pytest.importorskip("vision_rl.vlm_local_training")


def test_trainer_reads_the_builders_fixture_trie_as_the_same_tree(tmp_path):
    local = _local()
    path = tmp_path / "trie.json"
    path.write_text(json.dumps(fixture_trie(slack=3)))
    sequences, trie, file_sha = local.load_label_trie(path)
    # Hand-written: "A" -> [1]; "AB-1" -> [7], [3,4,5], [1,2,4,5] (slack 3 keeps all), + EOS.
    assert sequences == {"AB-1": [[1, 2, 4, 5, EOS], [3, 4, 5, EOS], [7, EOS]],
                         "A": [[1, EOS]]}
    assert local.trie_allowed(trie, []) == [1, 3, 7]
    assert local.trie_allowed(trie, [1]) == [2, EOS]
    assert local.trie_allowed(trie, [1, EOS]) is None  # complete
    assert file_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    reader = lt.LabelTrie(fixture_trie(slack=3))
    for prefix in ([], [1], [1, 2], [1, 2, 4], [3], [3, 4], [7]):
        assert set(local.trie_allowed(trie, prefix)) == set(reader.allowed(prefix))


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda d: d["entries"]["A"]["paths"].append([6]), "hash"),  # content changed, sha kept
    (lambda d: d.update(labels=["A", "AB-1"]), "labels matching"),
    (lambda d: d.pop("eos_ids"), "eos_ids"),
    (lambda d: d.update(schema_version=2), "schema_version"),
])
def test_trainer_refuses_a_changed_or_malformed_trie(tmp_path, mutate, match):
    local = _local()
    document = fixture_trie()
    mutate(document)
    path = tmp_path / "trie.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        local.load_label_trie(path)


def test_trainer_refuses_an_eos_inside_a_path_and_the_old_sequences_format(tmp_path):
    local = _local()
    document = fixture_trie()
    document["entries"]["A"]["paths"] = [[1, EOS]]
    document["sha256"] = lt.trie_sha256(document)  # consistent hash, invalid content
    path = tmp_path / "trie.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="EOS-free"):
        local.load_label_trie(path)
    path.write_text(json.dumps({"schema_version": 1, "sequences": {"A": [[1, EOS]]}}))
    with pytest.raises(ValueError, match="eos_ids"):
        local.load_label_trie(path)


def test_committed_artifacts_load_in_the_trainer_with_one_trie():
    local = _local()
    path = AUDIT / "label-trie-qwen2vl2b.json"
    document = json.loads(path.read_text())
    labels = document["labels"]
    assert len(labels) == 100
    sequences, trie, _ = local.load_label_trie(path)
    # g0-preview.json: 1,092 label paths, 2,184 with the two EOS IDs.
    assert sum(len(v) for v in sequences.values()) == 2184 == document["path_count"]
    assert local._label_trie_content_sha256(document) == lt.trie_sha256(document)
    reader = lt.LabelTrie(document)
    for options in sequences.values():
        for full in options:
            # Every certified path is on the trainer's trie, end to end, and both readers
            # allow the same next tokens at every prefix.
            assert local.trie_position_mask(trie, full)[-1] == sorted(reader.allowed(full[:-1]))
            for cut in range(len(full)):
                assert set(local.trie_allowed(trie, full[:cut])) == set(
                    reader.allowed(full[:cut]))
    confusers = local.load_confusion_map(AUDIT / "p0-confusion-map.json", labels)
    assert set(confusers) == set(labels)
    # Hand-checked in the committed map: 707-320's only confuser is DC-8 (observed).
    assert confusers["707-320"] == ["DC-8"]
    taxonomy = local.load_taxonomy(AUDIT / "taxonomy-train-only.json", labels)
    assert taxonomy["737-300"] == {"family": "Boeing 737", "manufacturer": "Boeing"}
