"""Label trie T over token IDs for the signal-limit study (docs/aircraft-rl-next-study-v1.json).

Pure logic, CPU only, no model. T holds every tokenisation of each of the 100 canonical labels
that the tokenizer's ordinary vocabulary can spell byte-for-byte, pruned to at most
``canonical length + slack`` tokens (slack 2 in the study), each followed by one EOS ID.

Blind spots, stated on purpose:
  * Only ordinary (non-added) vocabulary entries spell labels; added/special tokens never do.
  * A tokenisation longer than the prune bound is not in T. The prune record keeps, per label,
    how many tokenisations were dropped, so the bound is auditable, not silent.
  * T says nothing about text AROUND the label: a completion with a leading space or newline is
    off-trie even though ``parse_label`` (which strips) would accept it. The coverage check is
    what proves this never happened on the 8 controlled400 VAL journals.

Lives in scripts/, not src/vision_rl/: the activated freshpool plan pins every tracked
src/vision_rl module (tests/test_freshpool_production.py), and this study is unactivated. A new
vision_rl module would void that plan's pin set; moving it in belongs to the plan re-cut that
activates this study.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
PRUNE_RULE = "keep a tokenisation iff its length (EOS excluded) <= canonical length + slack"


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level alphabet (used by Qwen2's ByteLevel BPE): byte value -> vocab char."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes, extra = printable[:], 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(printable, map(chr, codes), strict=True))


def byte_vocab(vocab: Mapping[str, int]) -> dict[bytes, int]:
    """Byte-level vocab strings -> {raw bytes: id}. Refuses a token outside the byte alphabet."""
    decode = {char: byte for byte, char in bytes_to_unicode().items()}
    out: dict[bytes, int] = {}
    for token, identifier in vocab.items():
        try:
            raw = bytes(decode[char] for char in token)
        except KeyError as error:
            raise ValueError(f"token {token!r} is not byte-level; refusing") from error
        if raw in out:
            raise ValueError(f"two vocab entries spell {raw!r}; refusing")
        out[raw] = identifier
    return out


def _pieces(text: bytes, vocab: Mapping[bytes, int]) -> list[list[tuple[int, int]]]:
    """pieces[i] = [(end, id)] for every vocab token equal to text[i:end]."""
    return [
        [(end, vocab[text[start:end]]) for end in range(start + 1, len(text) + 1)
         if text[start:end] in vocab]
        for start in range(len(text))
    ]


def count_tokenizations(text: bytes, vocab: Mapping[bytes, int]) -> dict[int, int]:
    """{length: number of tokenisations of ``text`` with that many tokens}, unbounded."""
    pieces = _pieces(text, vocab)
    ways: list[dict[int, int]] = [dict() for _ in range(len(text) + 1)]
    ways[len(text)] = {0: 1}
    for start in range(len(text) - 1, -1, -1):
        for end, _ in pieces[start]:
            for length, count in ways[end].items():
                ways[start][length + 1] = ways[start].get(length + 1, 0) + count
    return dict(sorted(ways[0].items()))


def enumerate_tokenizations(
    text: bytes, vocab: Mapping[bytes, int], max_tokens: int
) -> list[tuple[int, ...]]:
    """Every tokenisation of ``text`` with at most ``max_tokens`` tokens, sorted."""
    pieces = _pieces(text, vocab)
    found: list[tuple[int, ...]] = []

    def walk(start: int, path: tuple[int, ...]) -> None:
        if start == len(text):
            found.append(path)
            return
        if len(path) == max_tokens:
            return
        for end, identifier in pieces[start]:
            walk(end, path + (identifier,))

    walk(0, ())
    return sorted(found)


def build_trie(
    labels: Sequence[str],
    vocab: Mapping[bytes, int],
    canonical: Mapping[str, Sequence[int]],
    eos_ids: Iterable[int],
    *,
    slack: int = 2,
) -> dict:
    """The trie document (paths without EOS, per label) plus its prune record."""
    eos = sorted(set(eos_ids))
    if not eos or set(eos) & set(vocab.values()):
        raise ValueError("EOS IDs must be nonempty and outside the ordinary vocabulary")
    if len(set(labels)) != len(labels) or set(canonical) != set(labels):
        raise ValueError("labels must be unique and each needs its canonical tokenisation")
    if slack < 0:
        raise ValueError("slack must be >= 0")
    by_id = {identifier: raw for raw, identifier in vocab.items()}
    entries = {}
    for label in labels:
        text = label.encode("utf-8")
        canon = [int(i) for i in canonical[label]]
        if b"".join(by_id.get(i, b"\0?") for i in canon) != text:
            raise ValueError(f"canonical tokenisation of {label!r} does not spell it")
        bound = len(canon) + slack
        paths = enumerate_tokenizations(text, vocab, bound)
        counts = count_tokenizations(text, vocab)
        kept = sum(c for length, c in counts.items() if length <= bound)
        if kept != len(paths) or tuple(canon) not in set(paths):
            raise ValueError(f"enumeration and count disagree for {label!r}")
        entries[label] = {
            "canonical": canon,
            "max_tokens": bound,
            "paths": [list(path) for path in paths],
            "kept": kept,
            "pruned": sum(counts.values()) - kept,
            "pruned_by_length": {str(n): c for n, c in counts.items() if n > bound},
        }
    owner: dict[tuple[int, ...], str] = {}
    for label, entry in entries.items():
        for path in entry["paths"]:
            if tuple(path) in owner:
                raise ValueError(f"path {path} spells two labels")
            owner[tuple(path)] = label
    return {
        "schema_version": SCHEMA_VERSION,
        "prune_rule": PRUNE_RULE,
        "slack": slack,
        "eos_ids": eos,
        "labels": list(labels),
        "entries": entries,
        "path_count": sum(len(e["paths"]) for e in entries.values()) * len(eos),
    }


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def trie_sha256(trie: Mapping) -> str:
    """SHA256 of the trie document without its own ``sha256`` and ``provenance`` fields."""
    body = {k: v for k, v in trie.items() if k not in ("sha256", "provenance")}
    return hashlib.sha256(canonical_json(body)).hexdigest()


class LabelTrie:
    """Prefix index over T: full paths are label path + one EOS ID."""

    def __init__(self, trie: Mapping):
        if trie.get("sha256") is not None and trie["sha256"] != trie_sha256(trie):
            raise ValueError("trie sha256 does not match its content; refusing")
        self.eos = frozenset(trie["eos_ids"])
        self.label_of: dict[tuple[int, ...], str] = {}
        self.children: dict[tuple[int, ...], set[int]] = {}
        for label, entry in trie["entries"].items():
            for path in entry["paths"]:
                path = tuple(path)
                for eos in self.eos:
                    full = path + (eos,)
                    self.label_of[full] = label
                    for cut in range(len(full)):
                        self.children.setdefault(full[:cut], set()).add(full[cut])

    def allowed(self, prefix: Sequence[int]) -> frozenset[int]:
        """Token IDs allowed after ``prefix``; empty after a full path or off the trie."""
        return frozenset(self.children.get(tuple(prefix), ()))

    def contains(self, completion_ids: Sequence[int]) -> bool:
        return tuple(completion_ids) in self.label_of

    def label(self, completion_ids: Sequence[int]) -> str | None:
        return self.label_of.get(tuple(completion_ids))

    def mask_row(self, prefix: Sequence[int], row: Sequence[float]) -> list[float]:
        """One logits row with every token not allowed after ``prefix`` set to -inf.

        Refuses an off-trie prefix (nothing allowed) rather than returning an all -inf row.
        """
        allowed = self.allowed(prefix)
        if not allowed:
            raise ValueError("prefix is off the trie or already complete; refusing to mask")
        if max(allowed) >= len(row):
            raise ValueError("allowed token ID outside the logits row")
        return [value if i in allowed else float("-inf") for i, value in enumerate(row)]


def coverage(trie: LabelTrie, completions: Iterable[tuple[str, Sequence[int]]],
             expected: int) -> dict:
    """Every (parsed label, completion_ids) must be a T path spelling that same label.

    Refuses (ValueError) on any miss, any label disagreement, or a count != ``expected``.
    """
    seen, misses = 0, []
    for label, ids in completions:
        seen += 1
        if trie.label(ids) != label:
            misses.append({"label": label, "completion_ids": list(ids)})
    if not seen:
        raise ValueError("coverage check saw zero valid completions; refusing")
    if misses:
        raise ValueError(f"{len(misses)} valid completions are off T, e.g. {misses[:3]}")
    if seen != expected:
        raise ValueError(f"coverage counted {seen} valid completions, expected {expected}")
    return {"valid_completions": seen, "in_trie": seen, "expected": expected}


def upper_bound_macro(per_class: Mapping[str, Mapping[str, int]]) -> float:
    """Equal-class macro if every invalid/missing answer had been the gold label."""
    if not per_class:
        raise ValueError("no classes")
    for item in per_class.values():
        if item["n"] <= 0 or item["correct"] + item["invalid"] > item["n"]:
            raise ValueError("per-class counts are inconsistent")
    return sum((v["correct"] + v["invalid"]) / v["n"] for v in per_class.values()) / len(per_class)


def taxonomy(rows: Iterable[Mapping], labels: Sequence[str]) -> dict:
    """variant -> {family, manufacturer} from TRAIN annotation rows only; refuses non-TRAIN rows,
    a variant with two parents, or a label never annotated."""
    parent: dict[str, tuple[str, str]] = {}
    count = 0
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"non-TRAIN row {row.get('image_id')!r}; refusing")
        key = (row["family"], row["manufacturer"])
        if parent.setdefault(row["variant"], key) != key:
            raise ValueError(f"variant {row['variant']!r} has two parents")
        count += 1
    if set(parent) != set(labels):
        raise ValueError("TRAIN annotations do not cover exactly the label vocabulary")
    variants = {label: {"family": parent[label][0], "manufacturer": parent[label][1]}
                for label in labels}

    def group(field):
        out: dict[str, list[str]] = {}
        for label in labels:
            out.setdefault(variants[label][field], []).append(label)
        return dict(sorted(out.items()))

    return {"train_rows": count, "variants": variants,
            "families": group("family"), "manufacturers": group("manufacturer")}


def confusion_map(tax: Mapping, observed: Iterable[tuple[str, str | None]],
                  labels: Sequence[str]) -> dict:
    """gold -> sorted confusers: taxonomy family siblings U valid wrong answers observed.

    ``observed`` is (gold, canonical prediction or None); None and correct answers add nothing.
    """
    vocabulary = set(labels)
    family = {label: tax["variants"][label]["family"] for label in labels}
    siblings = {g: {x for x in labels if x != g and family[x] == family[g]} for g in labels}
    seen: dict[str, set[str]] = {g: set() for g in labels}
    for gold, predicted in observed:
        if gold not in vocabulary or (predicted is not None and predicted not in vocabulary):
            raise ValueError("observation outside the label vocabulary")
        if predicted is not None and predicted != gold:
            seen[gold].add(predicted)
    return {
        g: {"family": sorted(siblings[g]), "observed": sorted(seen[g]),
            "confusers": sorted(siblings[g] | seen[g])}
        for g in labels
    }
