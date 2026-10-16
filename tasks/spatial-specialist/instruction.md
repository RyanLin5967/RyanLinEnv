# Train a spatial vision specialist

Train a policy to recognize where a red square is relative to a blue circle.
The four actions are left, right, above, and below. Training examples are in
`/app/data/train.jsonl`, and validation examples are in `/app/data/validation.jsonl`.

Use reward feedback to improve the policy. Inspect `vision-rl train --help` and
`vision-rl train-agent --help` for the available tools. The CPU policy uses a fixed
color-centroid image encoder and a learned 3×4 linear softmax head. You may change
training code and hyperparameters within the five-minute time budget.

Submit `/logs/artifacts/training/policy.npz`, a NumPy archive containing only a
finite floating-point array named `weights` of shape `(3, 4)`. The inference rule
is `argmax(features(image) @ weights)`. Values must have magnitude at most 10,000;
the archive must be smaller than 16 KiB. The fixed encoder computes normalized
red-minus-blue x and y centroid differences followed by a constant bias of 1.

The verifier runs in a fresh container and scores a separately generated test
split. Reward is test accuracy in [0, 1]. Validation is available for choosing
your checkpoint; the final test examples are not available during training.
Save experiment history alongside the checkpoint to make the result reviewable.

This is a synthetic onboarding task. Its score is not an HSS score or evidence
that a model outperforms a frontier model on a real benchmark.
