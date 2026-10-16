# Evidence bundle

Measured on 2026-10-16 UTC (2026-10-15 Toronto). Read [the results and annotation caveat](../../docs/results.md) before interpreting the scores.

- `predictions.csv`: every frozen model's count on the same 30 IDs.
- `*-evaluation.json`: per-item outputs, label agreement, dataset fingerprint, and model identity.
- `argon-native-responses.jsonl`: actual provider replies with model version, token usage, timing, and request settings. No keys or image bytes.
- `training-plan.json`, `heldout-evaluation-plan.json`, `training-history.jsonl`, `training.json`: declared search, selected checkpoint, and learning history.
- `checkpoints/`: compact trained heads; reconstruct the public ImageNet encoder with `scripts/prepare_encoder.py`.
- `checkpoint-replay.json`: the published checkpoints reproduced all 60 recorded predictions through the evaluation CLI.
- `annotation-audit.json`, `label-audit.json`, `data-audit.json`, `additional-data-audit.json`: dataset quality evidence and limitations.
- `harbor-trial.json`, `harbor-evaluation.json`: actual Docker/Harbor replay with a separate verifier.
- `harbor-control-trials.json`, `docker-demo.json`: executed reference, training-agent, empty-agent, and Docker onboarding controls.
- `checksums.json`: SHA-256 identities for the generated evidence files.

The dataset images, credentials, large pretrained encoder, and scratch runs are intentionally excluded. Source code is MIT licensed; upstream pretrained weights retain their upstream terms. These checkpoints are educational artifacts trained on one dataset, with no claim of broad generalization.
