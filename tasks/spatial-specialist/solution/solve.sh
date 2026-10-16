#!/bin/bash
set -euo pipefail
vision-rl train-agent --train /app/data/train.jsonl \
  --validation /app/data/validation.jsonl --output /logs/artifacts/training
