import hashlib
import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_aircraft_freshpool.py"
SPEC = importlib.util.spec_from_file_location("prepare_aircraft_freshpool", SCRIPT)
FRESH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRESH)

PROTOCOL = "proto-test"
DECOY_PROTOCOL_ID = "proto-top-level-decoy"


def row(image_id, label):
    return {
        "id": image_id,
        "image": f"images/train/{image_id}.jpg",
        "label": label,
        "split": "train",
    }


def fixture():
    """Hand-built: A has a01-a07, B has b01-b10; train400 = a01,b01; probe = a01 (in
    train400) and a02 (outside). Expected IDs below were computed by hand-running the
    study's hash/shuffle rules in a separate shell, never by calling the script."""
    train = [row(f"a{i:02d}", "A") for i in range(1, 8)] + [
        row(f"b{i:02d}", "B") for i in range(1, 11)
    ]
    train400 = [
        {"id": "a01", "label": "A", "split": "train"},
        {"id": "b01", "label": "B", "split": "train"},
    ]
    probe = [
        {"id": "a01", "label": "A", "split": "train"},
        {"id": "a02", "label": "A", "split": "train"},
    ]
    pool_ids = [
        "a03",
        "a04",
        "a05",
        "a06",
        "a07",
        "b02",
        "b03",
        "b04",
        "b05",
        "b06",
        "b07",
        "b08",
        "b09",
        "b10",
    ]
    base = {i: random.Random(index).getrandbits(64) for index, i in enumerate(pool_ids)}
    # Fixture sanity (not the subject): unplanted hashes are far apart.
    assert min((x ^ y).bit_count() for x, y in itertools.combinations(base.values(), 2)) > 6
    pool = dict(base)
    pool["b04"] = base["b06"] ^ 0b111  # straddle, S member sorts first: drop b04
    pool["b05"] = base["b03"] ^ 0b1  # straddle, H member sorts first: drop b05
    pool["b10"] = base["b02"] ^ 0b11  # both in S: reported, not dropped
    pool["a05"] = base["a04"] ^ 0b1111  # both in H-gate: reported, not dropped
    pool["a07"] = base["a04"] ^ (0b1111 << 40)  # H-select x H-gate: reported, flagged
    references = {
        "validation": {"v01": base["a03"] ^ 0b111111},  # Hamming 6: boundary, dropped
        "train400": {"b01": base["b08"] ^ 0b1111111},  # Hamming 7: boundary, kept
    }
    exact = {"test": {"b07": ["encoded_sha256", "model_rgb_sha256"]}}
    return train, train400, probe, pool, references, exact


def ids(rows):
    return [r["id"] for r in rows]


def test_pool_holdout_and_scan_manifests_follow_the_registered_rules():
    train, train400, probe, pool, references, exact = fixture()
    out = FRESH.make_manifests(train, train400, probe, PROTOCOL, pool, references, exact)
    audit = out["audit"]
    assert audit["counts"]["pool_before_audit"] == 14
    assert audit["counts"]["probe16_outside_train400"] == 1
    assert audit["pool_drops"] == ["a03", "b07"]
    assert ids(out["pool"]) == [
        "a04",
        "a05",
        "a06",
        "a07",
        "b02",
        "b03",
        "b04",
        "b05",
        "b06",
        "b08",
        "b09",
        "b10",
    ]
    assert ids(out["h_select"]) == ["a07", "a06", "b06", "b08"]
    assert ids(out["h_gate"]) == ["a04", "a05", "b09", "b03"]
    assert audit["s_drops"] == ["b04", "b05"]

    def pair(a, b, hamming, h_s, select_gate=False):
        return {
            "ids": [a, b],
            "hamming": hamming,
            "straddles_h_and_s": h_s,
            "straddles_h_select_and_h_gate": select_gate,
        }

    assert audit["within_pool_pairs"] == [
        pair("a04", "a05", 4, False),
        pair("a04", "a07", 4, False, True),
        pair("b02", "b10", 2, False),
        pair("b03", "b05", 1, True),
        pair("b04", "b06", 3, True),
    ]
    assert audit["counts"]["within_pool_pairs_straddling_h_select_and_h_gate"] == 1
    assert any("H-select and H-gate" in note for note in audit["notes"])
    assert ids(out["s"]) == ["b02", "b10"]
    assert {seed: ids(rows) for seed, rows in out["scans"].items()} == {
        42: ["b10", "b02"],
        314: ["b10", "b02"],
        2718: ["b10", "b02"],
    }
    holdout = set(ids(out["h_select"])) | set(ids(out["h_gate"]))
    assert not holdout & set(ids(out["s"]))
    assert not {"a01", "a02", "b01"} & set(ids(out["pool"]))
    assert audit["external_near_duplicates"] == [
        {"pool_id": "a03", "reference_split": "validation", "reference_id": "v01", "hamming": 6},
        {
            "pool_id": "b07",
            "reference_split": "test",
            "reference_id": None,
            "hamming": 0,
            "exact_match_fields": ["encoded_sha256", "model_rgb_sha256"],
        },
    ]


def test_holdout_depends_on_protocol_id_and_refuses_thin_classes():
    train, train400, probe, pool, references, exact = fixture()
    other = FRESH.make_manifests(train, train400, probe, "proto-other", pool, references, exact)
    assert ids(other["h_select"]) != ["a07", "a06", "b06", "b08"]
    with pytest.raises(ValueError, match="protocol_id"):
        FRESH.make_manifests(train, train400, probe, "", pool, references, exact)
    references["train400"]["b01"] = pool["a04"]  # a second A drop leaves A with 3
    with pytest.raises(ValueError, match="fewer than 4"):
        FRESH.make_manifests(train, train400, probe, PROTOCOL, pool, references, exact)


def test_contract_and_membership_violations_refuse():
    train, train400, probe, pool, references, exact = fixture()
    contract = {"classes": 2, "train400": 2, "probe16_outside_train400": 1, "pool_before_audit": 14}
    FRESH.make_manifests(train, train400, probe, PROTOCOL, pool, references, exact, contract)
    with pytest.raises(ValueError, match="contract changed"):
        FRESH.make_manifests(
            train,
            train400,
            probe,
            PROTOCOL,
            pool,
            references,
            exact,
            {**contract, "pool_before_audit": 2919},
        )
    bad_probe = [{"id": "a02", "label": "B", "split": "train"}]
    with pytest.raises(ValueError, match="disagrees"):
        FRESH.make_manifests(train, train400, bad_probe, PROTOCOL, pool, references, exact)
    with pytest.raises(ValueError, match="cover exactly"):
        FRESH.make_manifests(
            train, train400, probe, PROTOCOL, {**pool, "a01": 0}, references, exact
        )


def test_scan_order_is_seeded_shuffle_of_sorted_ids():
    unsorted = [f"x{i:02d}" for i in reversed(range(10))]
    assert FRESH.scan_order(unsorted, 42) == [
        "x07",
        "x03",
        "x02",
        "x08",
        "x05",
        "x06",
        "x09",
        "x04",
        "x00",
        "x01",
    ]
    assert FRESH.scan_order(unsorted, 314) == [
        "x04",
        "x05",
        "x09",
        "x06",
        "x02",
        "x00",
        "x08",
        "x01",
        "x07",
        "x03",
    ]
    assert FRESH.scan_order(unsorted, 2718) == [
        "x06",
        "x09",
        "x04",
        "x08",
        "x07",
        "x01",
        "x02",
        "x05",
        "x00",
        "x03",
    ]


def smooth(seed, size=(320, 240)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)).resize(
        size, Image.Resampling.BICUBIC
    )


def test_phash_flags_planted_resized_recompressed_copy_and_not_distinct_images():
    original = smooth(1)
    planted = FRESH.planted_copy(original)
    assert planted.size == (192, 144)  # resized, then decoded from JPEG bytes
    near = FRESH.near_pairs({"orig": FRESH.phash64(original)}, {"copy": FRESH.phash64(planted)})
    assert [pair[:2] for pair in near] == [("orig", "copy")]
    distinct = {f"d{s}": FRESH.phash64(smooth(s)) for s in range(2, 12)}
    assert FRESH.near_pairs(distinct, distinct, within=True) == []
    # Hand-built hashes: the threshold is inclusive at 6 and exclusive at 7.
    assert FRESH.near_pairs({"q": 0}, {"six": 0b111111, "seven": 0b1111111}) == [("q", "six", 6)]


def pool_images():
    rows = [row(f"p{i:02d}", "A" if i < 9 else "B") for i in range(18)]
    images = {r["id"]: smooth(100 + i) for i, r in enumerate(rows)}
    return rows, images


def test_forced_fire_passes_on_working_detector():
    rows, images = pool_images()
    hashes = {i: FRESH.phash64(image) for i, image in images.items()}
    report = FRESH.forced_fire(rows, hashes, images.__getitem__)
    assert report["passed"] and report["planted_flagged"]
    assert report["planted_hamming"] <= 6
    assert len(report["distinct_class_pairs"]) == 50
    assert report["distinct_class_pairs_flagged"] == 0
    labels = {r["id"]: r["label"] for r in rows}
    assert all(
        labels[a] != labels[b] for a, b in (p["ids"] for p in report["distinct_class_pairs"])
    )


@pytest.mark.parametrize("threshold", [-1, 64])
def test_forced_fire_refuses_blind_or_trigger_happy_detector(monkeypatch, threshold):
    rows, images = pool_images()
    hashes = {i: FRESH.phash64(image) for i, image in images.items()}
    monkeypatch.setattr(FRESH, "HAMMING_MAX", threshold)
    with pytest.raises(ValueError, match="forced fire failed"):
        FRESH.forced_fire(rows, hashes, images.__getitem__)


def test_forced_fire_refuses_too_few_distinct_class_pairs():
    rows, images = pool_images()
    rows = rows[:9] + rows[9:10]  # 9 x 1 = 9 cross-class pairs < 50
    hashes = {r["id"]: FRESH.phash64(images[r["id"]]) for r in rows}
    with pytest.raises(ValueError, match="distinct-class pairs"):
        FRESH.forced_fire(rows, hashes, images.__getitem__)


def model_rgb_sha256_by_hand(path):
    """Fixture-side: bottom20 crop of decoded RGB, hashed as header + NUL + raw pixels."""
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    cropped = rgb.crop((0, 0, rgb.width, rgb.height - 20))
    header = json.dumps(
        {"width": cropped.width, "height": cropped.height, "mode": "RGB"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + cropped.tobytes()).hexdigest()


def build_tree(tmp_path, monkeypatch):
    dev = tmp_path / "development"
    (dev / "images/train").mkdir(parents=True)
    (dev / "images/validation").mkdir(parents=True)
    labels = ["A", "B"]
    train = [row(f"t{c}{i:02d}", labels[c]) for c in range(2) for i in range(12)]
    val = [
        {
            "id": f"v{c}",
            "image": f"images/validation/v{c}.jpg",
            "label": labels[c],
            "split": "validation",
        }
        for c in range(2)
    ]
    for index, r in enumerate(train):
        smooth(200 + index).save(dev / r["image"], quality=95)
    for index, r in enumerate(val):
        smooth(400 + index).save(dev / r["image"], quality=95)
    # Plant a VAL near-duplicate of pool image t005.
    FRESH.planted_copy(smooth(205)).save(dev / val[0]["image"], quality=95)
    (dev / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (dev / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    train400 = tmp_path / "train400.jsonl"
    train400.write_text(
        "".join(
            json.dumps({"id": i, "label": lab, "split": "train"}) + "\n"
            for i, lab in (("t000", "A"), ("t100", "B"))
        )
    )
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"selected": [{"id": "t001", "label": "A", "split": "train"}]}))
    protocol = tmp_path / "protocol.json"
    # The top-level id is a decoy: the holdout salt is the pinned literal under data.
    protocol.write_text(
        json.dumps(
            {
                "protocol_id": DECOY_PROTOCOL_ID,
                "data": {"train_holdout": {"hash_protocol_id": PROTOCOL}},
            }
        )
    )
    # TEST rows: z1 matches t111 by encoded bytes only (pixel digest is a decoy); z2 matches
    # t110 by model pixels only (encoded digest is a decoy); z3 matches nothing.
    t111 = hashlib.sha256((dev / "images/train/t111.jpg").read_bytes()).hexdigest()
    t110_rgb = model_rgb_sha256_by_hand(dev / "images/train/t110.jpg")
    test_rows = [
        {"image_id": "z1", "encoded_sha256": t111, "model_rgb_sha256": "1" * 64},
        {"image_id": "z2", "encoded_sha256": "2" * 64, "model_rgb_sha256": t110_rgb},
        {"image_id": "z3", "encoded_sha256": "3" * 64, "model_rgb_sha256": "4" * 64},
    ]
    test_audit = tmp_path / "test-images.json"
    test_audit.write_text(json.dumps(test_rows))
    monkeypatch.setattr(
        FRESH,
        "REAL_CONTRACT",
        {"classes": 2, "train400": 2, "probe16_outside_train400": 1, "pool_before_audit": 21},
    )
    monkeypatch.setattr(FRESH, "TEST_IMAGE_COUNT", 3)
    argv = [
        "prepare",
        "--protocol",
        str(protocol),
        "--probe",
        str(probe),
        "--output",
        str(tmp_path / "out"),
        "--development",
        str(dev),
        "--train400",
        str(train400),
        "--test-audit",
        str(test_audit),
    ]
    return dev, test_audit, argv


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    FRESH.main()


def test_main_reads_jpegs_and_writes_hashed_manifests(tmp_path, monkeypatch):
    _, _, argv = build_tree(tmp_path, monkeypatch)
    output = tmp_path / "out"
    run_main(monkeypatch, argv)
    manifest = json.loads((output / "manifest.json").read_text())
    audit = json.loads((output / "phash_audit.json").read_text())
    assert audit["pool_drops"] == ["t005", "t110", "t111"]
    exact = [r for r in audit["external_near_duplicates"] if r["reference_split"] == "test"]
    assert [(r["pool_id"], r["exact_match_fields"]) for r in exact] == [
        ("t110", ["model_rgb_sha256"]),
        ("t111", ["encoded_sha256"]),
    ]
    assert manifest["counts"]["pool_after_audit"] == 18
    assert manifest["test_phash_reference"] == "NOT_CHECKED"
    assert manifest["g0_audit_complete"] is False
    assert audit["forced_fire"]["passed"] is True
    for name, digest in manifest["files_sha256"].items():
        assert FRESH.sha((output / name).read_bytes()) == digest
    assert len((output / "h_select.jsonl").read_text().splitlines()) == 4
    # Each scan file is that seed's shuffle of the sorted S ids, and the seeds differ.
    s_ids = [json.loads(line)["id"] for line in (output / "s.jsonl").read_text().splitlines()]
    assert len(s_ids) == 10
    scans = {}
    for seed in (42, 314, 2718):
        expected = sorted(s_ids)
        random.Random(seed).shuffle(expected)
        lines = (output / f"s_scan_seed{seed}.jsonl").read_text().splitlines()
        scans[seed] = [json.loads(line)["id"] for line in lines]
        assert scans[seed] == expected
    assert len({tuple(order) for order in scans.values()}) == 3
    with pytest.raises(FileExistsError):
        FRESH.main()


def test_main_with_test_phashes_checks_test_and_completes_g0(tmp_path, monkeypatch):
    dev, _, argv = build_tree(tmp_path, monkeypatch)
    # z3's pHash is a planted resized/recompressed copy of pool image t006.
    with Image.open(dev / "images/train/t006.jpg") as source:
        rgb = source.convert("RGB")
    planted = FRESH.planted_copy(rgb.crop((0, 0, rgb.width, rgb.height - 20)))
    phashes = {"z1": "0" * 16, "z2": "f" * 16, "z3": f"{FRESH.phash64(planted):016x}"}
    phash_file = tmp_path / "test-phashes.json"
    phash_file.write_text(json.dumps(phashes))
    run_main(monkeypatch, [*argv, "--test-phashes", str(phash_file)])
    output = tmp_path / "out"
    manifest = json.loads((output / "manifest.json").read_text())
    audit = json.loads((output / "phash_audit.json").read_text())
    assert manifest["test_phash_reference"] == "checked"
    assert manifest["g0_audit_complete"] is True
    assert "test" in audit["reference_splits_phash"]
    assert audit["pool_drops"] == ["t005", "t006", "t110", "t111"]
    test_phash = [r for r in audit["external_near_duplicates"] if r.get("reference_id") == "z3"]
    assert [r["pool_id"] for r in test_phash] == ["t006"]
    assert str(phash_file) in manifest["inputs_sha256"]


@pytest.mark.parametrize(
    "phashes, match",
    [
        ({}, "keys must equal"),
        ({"z1": "0" * 16, "z2": "0" * 16}, "keys must equal"),
        ({"z1": "0" * 16, "z2": "0" * 16, "z3": "0" * 16, "z4": "0" * 16}, "keys must equal"),
        ({"z1": "0" * 16, "z2": "0" * 16, "z3": "0" * 15}, "16 lowercase hex"),
        ({"z1": "0" * 16, "z2": "0" * 16, "z3": "G" * 16}, "16 lowercase hex"),
        ({"z1": "0" * 16, "z2": "0" * 16, "z3": 0}, "16 lowercase hex"),
        ([], "JSON object"),
    ],
)
def test_main_refuses_bad_test_phashes(tmp_path, monkeypatch, phashes, match):
    _, _, argv = build_tree(tmp_path, monkeypatch)
    phash_file = tmp_path / "test-phashes.json"
    phash_file.write_text(json.dumps(phashes))
    with pytest.raises(ValueError, match=match):
        run_main(monkeypatch, [*argv, "--test-phashes", str(phash_file)])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda rows: [], "must list 3 TEST rows, got 0"),
        (lambda rows: rows[:2], "must list 3 TEST rows, got 2"),
        (lambda rows: {"rows": rows}, "must list 3 TEST rows"),
        (lambda rows: [rows[0], rows[0], rows[2]], "unique"),
        (lambda rows: [rows[0], rows[1], {**rows[2], "model_rgb_sha256": None}], "model_rgb"),
        (lambda rows: [rows[0], rows[1], {**rows[2], "encoded_sha256": "x"}], "encoded_sha256"),
    ],
)
def test_main_refuses_bad_test_audit(tmp_path, monkeypatch, mutate, match):
    _, test_audit, argv = build_tree(tmp_path, monkeypatch)
    test_audit.write_text(json.dumps(mutate(json.loads(test_audit.read_text()))))
    with pytest.raises(ValueError, match=match):
        run_main(monkeypatch, argv)
    assert not (tmp_path / "out").exists()


def hand_holdout(salt, rows):
    """Independent re-statement of the registered rule, hashing with hashlib directly."""
    by_label = {}
    for image_id, label in rows:
        key = hashlib.sha256(
            json.dumps(
                [salt, "train-holdout-screen", 791, label, image_id], separators=(",", ":")
            ).encode()
        ).hexdigest()
        by_label.setdefault(label, []).append((key, image_id))
    select = []
    for label in sorted(by_label):
        ranked = [image_id for _, image_id in sorted(by_label[label])]
        select += [ranked[0], ranked[2]]
    return select


def test_main_salts_holdout_with_pinned_hash_protocol_id_not_protocol_id(tmp_path, monkeypatch):
    _, _, argv = build_tree(tmp_path, monkeypatch)
    run_main(monkeypatch, argv)
    output = tmp_path / "out"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["hash_protocol_id"] == PROTOCOL
    assert manifest["protocol_id"] == DECOY_PROTOCOL_ID
    # Audited pool, by hand: TRAIN t000-t011 (A), t100-t111 (B), minus train400 t000/t100,
    # probe t001, and drops t005 (VAL near-dup), t110 and t111 (TEST exact).
    audited = [(f"t0{i:02d}", "A") for i in (2, 3, 4, 6, 7, 8, 9, 10, 11)] + [
        (f"t1{i:02d}", "B") for i in range(1, 10)
    ]
    expected = hand_holdout(PROTOCOL, audited)
    decoy = hand_holdout(DECOY_PROTOCOL_ID, audited)
    assert expected != decoy  # fixture sanity: the two salts select different images
    lines = (output / "h_select.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == expected


@pytest.mark.parametrize(
    "protocol",
    [
        {"protocol_id": PROTOCOL},
        {"protocol_id": PROTOCOL, "data": {"train_holdout": {}}},
        {"protocol_id": PROTOCOL, "data": {"train_holdout": {"hash_protocol_id": ""}}},
        {"protocol_id": PROTOCOL, "data": {"train_holdout": {"hash_protocol_id": 7}}},
        {"protocol_id": PROTOCOL, "data": []},
    ],
)
def test_main_refuses_protocol_without_pinned_hash_protocol_id(tmp_path, monkeypatch, protocol):
    _, _, argv = build_tree(tmp_path, monkeypatch)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="hash_protocol_id"):
        run_main(monkeypatch, argv)
    assert not (tmp_path / "out").exists()


def test_committed_freshpool_protocol_pins_the_holdout_salt():
    protocol = json.loads(
        (SCRIPT.parents[1] / "docs/aircraft-rl-protocol-freshpool-v1.json").read_text()
    )
    assert FRESH.holdout_salt(protocol) == "fgvc-aircraft-freshpool-screened-grpo-v1"
