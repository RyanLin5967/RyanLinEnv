"""Reconstruct the public pretrained encoder for the small published policy checkpoints."""

import argparse
from pathlib import Path

from vision_rl.vision_policy import _encoder, _stack

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Encoder output already exists; choose a new path")
args.output.parent.mkdir(parents=True, exist_ok=True)
torch, _ = _stack()
torch.save(_encoder().state_dict(), args.output)
print(f"Saved public ResNet18 encoder to {args.output}")
