# Endpoint per-image predictions for named-baseline pairing (native VAL)

`endpoint-predictions.json` holds the canonical per-image labels (or `null` for invalid) of three
controlled400 endpoints on all 3,333 native VAL images. The endpoints are `common` (W),
`rl-exposure` (RL400) and `sft-1e5-time` (the strongest declared SFT). They are the comparators in
[docs/aircraft-ci-named-baselines-plan-v1.json](../../docs/aircraft-ci-named-baselines-plan-v1.json),
which pins this file's SHA256.

The file was extracted on 2026-10-18 by plain Python with no `vision_rl` import. The source was the
`outcome` events (`result.canonical_label`) in
`runs/aircraft-native-val-activation-v1/bundle/arms/<arm>/predictions.jsonl`, which is local and
ignored. That field is the output of the frozen evaluator's parser, and the evaluator's `compare`
command re-checks it. The extraction was checked in three ways:

- Each journal's SHA256 equals `prediction_journal_sha256` in
  [../2026-10-17-controlled400-native-val/comparison.json](../2026-10-17-controlled400-native-val/comparison.json).
- Every outcome is `completed`, and the IDs equal the gold order.
- The macro accuracy recomputed from these labels equals that file's value: 0.67843, 0.69818 and
  0.75391, each within 1e-12. The invalid counts are 91, 77 and 41.

These are VAL predictions only. TEST stays sealed.
