#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
python -I /tests/verify.py
