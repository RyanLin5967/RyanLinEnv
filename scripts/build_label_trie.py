"""Build the signal-limit label trie T and its G0 evidence (docs/aircraft-rl-next-study-v1.json,
``label_trie`` and implementation_work_list). CPU only; loads the tokenizer, never the model.

Writes, all sha256-pinned:
  * <audit>/taxonomy-train-only.json   variant -> family/manufacturer from TRAIN annotations only
  * <audit>/p0-confusion-map.json      gold -> confusers = taxonomy family siblings U valid wrong
                                       answers in the freshpool P0 rollouts (TRAIN images only)
  * <audit>/label-trie-qwen2vl2b.json  T: every tokenisation of the 100 labels spelled by the
                                       ordinary vocab, <= canonical length + 2 tokens, + one EOS;
                                       per-label prune record; sha256 over its content
  * <report>                           coverage over the 8 controlled400 native-VAL prediction
                                       journals (must equal --expected-paths, default 26,191) and
                                       the invalid-to-correct upper-bound table

Refuses (exit 1, naming the reason) on: a journal or report whose sha256 differs from the pin
in the committed comparison.json; a P0 rollout image that is not a TRAIN image; a valid VAL
completion off T; a coverage count other than --expected-paths; an upper bound recomputed from
the journals that disagrees with the one from comparison.json; zero rows anywhere.

Never reads sealed TEST data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import label_trie as lt  # noqa: E402

from vision_rl.classification_comparison import _score  # noqa: E402

NATIVE_VAL = "reports/2026-10-17-controlled400-native-val"
ARMS = ("common", "rl-exposure", "sft-1e4-exposure", "sft-1e4-time", "sft-1e5-exposure",
        "sft-1e5-time", "sft-5e5-exposure", "sft-5e5-time")
EXPECTED_PATHS = 26191  # study.json label_trie.coverage_check
SLACK = 2  # study.json label_trie.build: <= canonical length + 2 tokens


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ROOTS = [REPO]  # --source-root appends the checkout that holds the ignored data/ and runs/


def rel(path: Path) -> str:
    """Repo-relative path, so records do not depend on which checkout ran the build."""
    path = Path(path).resolve()
    for root in ROOTS:
        try:
            return str(path.relative_to(root.resolve()))
        except ValueError:
            pass
    return str(path)


def write(path: Path, document) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return sha(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_tokenizer(directory: Path):
    """(vocab bytes->id, canonical encoder, eos ids, pins). EOS = generation_config eos IDs U
    tokenizer eos_token: the same set the native-VAL evaluator stops on."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    vocab = lt.byte_vocab(tokenizer.get_vocab(with_added_tokens=False))
    generation = json.loads((directory / "generation_config.json").read_text())
    config = json.loads((directory / "tokenizer_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.token_to_id(config["eos_token"]))
    pins = {name: {"path": rel(directory / name), "sha256": sha(directory / name)}
            for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json")}

    def encode(label):
        return tokenizer.encode(label, add_special_tokens=False).ids

    return vocab, encode, sorted(eos), pins


def build_taxonomy(annotations: Path, labels):
    rows = json.loads(annotations.read_text())
    document = lt.taxonomy(rows, labels)
    document.update(
        schema_version=1,
        source={"path": rel(annotations), "sha256": sha(annotations),
                "rule": "split == 'train' rows only; refuse any other split"},
    )
    return document, {row["image_id"] for row in rows}


def build_confusion(p0_dir: Path, taxonomy, train_ids, labels):
    files = sorted(p0_dir.glob("rollout-seed*/reward_groups.jsonl"))
    if not files:
        raise SystemExit(f"no P0 reward_groups.jsonl under {p0_dir}")
    observed, groups, sources = [], 0, []
    for path in files:
        for group in read_jsonl(path):
            if group["id"] not in train_ids:
                raise SystemExit(f"P0 rollout image {group['id']} is not a TRAIN image")
            groups += 1
            observed += [(group["label"], p) for p in group["canonical_predictions"]]
        sources.append({"path": rel(path), "sha256": sha(path)})
    if not observed:
        raise SystemExit("P0 rollouts hold zero completions")
    mapping = lt.confusion_map(taxonomy, observed, labels)
    return {
        "schema_version": 1,
        "rule": "confusers(gold) = taxonomy-train-only family siblings U valid wrong answers "
                "observed for that gold label in the freshpool P0 G4 rollouts (TRAIN images); "
                "invalid answers add nothing; frozen, no change after any output is seen",
        "sources": {"taxonomy": "taxonomy-train-only.json (sha256 in this file's pins)",
                    "p0_reward_groups": sources, "groups": groups,
                    "completions": len(observed)},
        "map": mapping,
    }


def val_completions(journals: Path, pins: dict):
    """{arm: {id: (canonical_label|None, completion_ids)}} from sha-pinned journals."""
    out = {}
    for arm in ARMS:
        path = journals / arm / "predictions.jsonl"
        if sha(path) != pins[arm]:
            raise SystemExit(f"{rel(path)} sha256 differs from comparison.json's pin")
        outcomes = {}
        for record in read_jsonl(path):
            if record["event"] == "outcome":
                result = record["result"]
                outcomes[record["id"]] = (result["canonical_label"], result["completion_ids"])
        out[arm] = outcomes
    return out


def upper_bound_table(comparison, completions, gold_rows, labels):
    versus = comparison["versus_common_initialization"]
    per_arm = {"common": next(iter(versus.values()))["comparator"]}
    per_arm.update({arm: value["candidate"] for arm, value in versus.items()})
    if set(per_arm) != set(ARMS):
        raise SystemExit("comparison.json arms differ from the 8 controlled400 arms")
    table = {}
    for arm in ARMS:
        recorded = per_arm[arm]
        predictions = {i: label for i, (label, _) in completions[arm].items()}
        actual, _ = _score(gold_rows, predictions, labels)
        repaired = {row["id"]: predictions.get(row["id"]) or row["label"] for row in gold_rows}
        bound, _ = _score(gold_rows, repaired, labels)
        from_report = lt.upper_bound_macro(recorded["per_class"])
        if abs(actual["mean_class_accuracy"] - recorded["mean_class_accuracy"]) > 1e-12:
            raise SystemExit(f"{arm}: journal macro does not reproduce comparison.json")
        if abs(bound["mean_class_accuracy"] - from_report) > 1e-12:
            raise SystemExit(f"{arm}: upper bound from journals != from comparison.json")
        table[arm] = {
            "macro": recorded["mean_class_accuracy"],
            "invalid_or_missing": recorded["invalid_answers_including_missing"],
            "upper_bound_macro": from_report,
            "headroom_pp": 100 * (from_report - recorded["mean_class_accuracy"]),
        }
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer-dir", type=Path,
                        default=REPO / "data/models/Qwen2-VL-2B-Instruct-895c3a49")
    parser.add_argument("--native-val-report", type=Path, default=REPO / NATIVE_VAL)
    parser.add_argument("--journals", type=Path,
                        default=REPO / "runs/aircraft-native-val-activation-v1/bundle/arms")
    parser.add_argument("--gold", type=Path,
                        default=REPO / "runs/aircraft-native-val-activation-v1/bundle/gold.json")
    parser.add_argument("--train-annotations", type=Path,
                        default=REPO / "data/fgvc-aircraft-2013b/audit/train-images.json")
    parser.add_argument("--p0-dir", type=Path,
                        default=REPO / "runs/aircraft-freshpool-v1/stages/p0/instance-1")
    parser.add_argument("--audit-dir", type=Path, default=REPO / "data/fgvc-aircraft-2013b/audit")
    parser.add_argument("--report", type=Path,
                        default=REPO / "reports/2026-10-18-signal-limit-label-trie/g0-preview.json")
    parser.add_argument("--expected-paths", type=int, default=EXPECTED_PATHS)
    parser.add_argument("--source-root", type=Path, default=REPO,
                        help="checkout holding the ignored inputs; recorded paths are relative")
    args = parser.parse_args(argv)
    ROOTS.append(args.source_root)

    report_dir = args.native_val_report
    comparison = json.loads((report_dir / "comparison.json").read_text())
    labels = json.loads((report_dir / "plan.json").read_text())["labels"]
    sums = dict(reversed(line.split()) for line in (report_dir / "SHA256SUMS").read_text()
                .splitlines() if line.strip())
    for name in ("comparison.json", "plan.json"):
        if sums.get(f"{NATIVE_VAL}/{name}") != sha(report_dir / name):
            raise SystemExit(f"{name} does not match {NATIVE_VAL}/SHA256SUMS")
    if len(labels) != 100 or comparison["test_consumed"] is not False:
        raise SystemExit("native-VAL report is not the 100-label, TEST-unconsumed comparison")

    taxonomy, train_ids = build_taxonomy(args.train_annotations, labels)
    taxonomy_sha = write(args.audit_dir / "taxonomy-train-only.json", taxonomy)
    confusion = build_confusion(args.p0_dir, taxonomy, train_ids, labels)
    confusion["sources"]["taxonomy_sha256"] = taxonomy_sha
    confusion_sha = write(args.audit_dir / "p0-confusion-map.json", confusion)

    vocab, encode, eos, tokenizer_pins = load_tokenizer(args.tokenizer_dir)
    trie = lt.build_trie(labels, vocab, {label: encode(label) for label in labels}, eos,
                         slack=SLACK)
    trie["provenance"] = {"tokenizer": tokenizer_pins,
                          "builder": {"path": "scripts/build_label_trie.py",
                                      "module": "scripts/label_trie.py"}}
    trie["sha256"] = lt.trie_sha256(trie)
    trie_file_sha = write(args.audit_dir / "label-trie-qwen2vl2b.json", trie)
    index = lt.LabelTrie(trie)

    completions = val_completions(args.journals, comparison["prediction_journal_sha256"])
    valid = [(label, ids) for arm in ARMS for label, ids in completions[arm].values()
             if label is not None]
    cover = lt.coverage(index, valid, args.expected_paths)
    gold_rows = json.loads(args.gold.read_text())
    if len(gold_rows) != comparison["n"]:
        raise SystemExit("gold rows do not match the comparison's n")
    table = upper_bound_table(comparison, completions, gold_rows, labels)

    pins = {
        "taxonomy-train-only.json": taxonomy_sha,
        "p0-confusion-map.json": confusion_sha,
        "label-trie-qwen2vl2b.json": trie_file_sha,
    }
    report = {
        "schema_version": 1,
        "study": "aircraft-rl-next-study-v1 (unactivated); G0 items label_trie + confusion map",
        "status": "pre-G0 build from existing artifacts; recomputed and frozen at G0",
        "files_sha256": {rel(args.audit_dir / k): v for k, v in pins.items()},
        "trie": {"content_sha256": trie["sha256"], "slack": SLACK, "eos_ids": eos,
                 "paths_with_eos": trie["path_count"],
                 "label_paths": sum(len(e["paths"]) for e in trie["entries"].values()),
                 "pruned_tokenisations": sum(e["pruned"] for e in trie["entries"].values()),
                 "non_canonical_valid_completions": sum(
                     list(ids[:-1]) != trie["entries"][label]["canonical"]
                     for label, ids in valid)},
        "coverage": {**cover, "arms": list(ARMS),
                     "journals_sha256": comparison["prediction_journal_sha256"]},
        "upper_bound": {"rule": "macro if every invalid/missing answer became the gold label; "
                                "from existing predictions only",
                        "source": {"path": f"{NATIVE_VAL}/comparison.json",
                                   "sha256": sha(report_dir / "comparison.json")},
                        "gold": {"path": rel(args.gold), "sha256": sha(args.gold)},
                        "per_arm": table},
    }
    write(args.report, report)
    print(json.dumps({"files": report["files_sha256"], "trie": report["trie"],
                      "coverage": cover,
                      "upper_bound": {a: round(t["upper_bound_macro"], 4)
                                      for a, t in table.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
