#!/bin/bash
# Provision one UWaterloo host for the signal-limit study (plan uw.python / uw.model_directory).
#
# Run from the Mac through uwrun, with --sync pointing at a directory that holds:
#   model/Qwen2-VL-2B-Instruct-895c3a49/   the pinned model (data/models/... on the Mac)
#   requirements-vlm-local.lock            the repository's hashed lock
#   requirements-torch-cpu.lock            the repository's hashed CPU torch lock
#   src/                                    the repository's src (for the preflight)
#   provision_host.sh                       this file
#   ~/.claude/bin/uwrun run --host H --sync DIR --timeout 5400 -- bash provision_host.sh cpu|cuda
#
# Writes, host-local (never the 16 GB NFS home):
#   /var/tmp/r226lin-slenv-v1      uv CPython 3.12 venv: the hashed lock, then torch 2.14.1 +
#                                  torchvision 0.29.1 (cpu: the hashed CPU lock; cuda: PyPI's
#                                  default Linux wheel, the cu130 build, which has no hash lock
#                                  in the repository: its installed RECORD hashes are written
#                                  to the receipt instead)
#   /var/tmp/r226lin-slmodel-v1/Qwen2-VL-2B-Instruct-895c3a49   the model, then verified by
#                                  the trainer's own _verified_model_directory
# and prints one JSON receipt line (also out/provision.json): host, device, versions,
# preflight, model check, sha256 of the venv's torch RECORD. Refuses (exit != 0) on any
# step; an existing env or model directory is re-verified, never silently reused unchecked.
set -euo pipefail
device=${1:?cpu or cuda}
case "$device" in cpu|cuda) ;; *) echo "device must be cpu or cuda" >&2; exit 2;; esac
cd "$(pwd -P)"
ENV=/var/tmp/r226lin-slenv-v1
MODEL_ROOT=/var/tmp/r226lin-slmodel-v1
MODEL=$MODEL_ROOT/Qwen2-VL-2B-Instruct-895c3a49
UV=~/.local/bin/uv
export UV_CACHE_DIR=/var/tmp/r226lin-uvcache
mkdir -p out
grep -q -w avx2 /proc/cpuinfo || { echo "no AVX2: torch's CPU wheel dies with SIGILL" >&2; exit 3; }

if [ ! -x "$ENV/bin/python" ]; then
  rm -rf "$ENV.tmp"
  "$UV" venv -q -p 3.12 "$ENV.tmp"
  "$UV" pip install -q -p "$ENV.tmp/bin/python" --no-deps --require-hashes \
      -r requirements-vlm-local.lock
  if [ "$device" = cpu ]; then
    "$UV" pip install -q -p "$ENV.tmp/bin/python" --no-deps --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu -r requirements-torch-cpu.lock
  else
    "$UV" pip install -q -p "$ENV.tmp/bin/python" torch==2.14.1 torchvision==0.29.1
  fi
  mv "$ENV.tmp" "$ENV"
fi

if [ ! -d "$MODEL" ]; then
  mkdir -p "$MODEL_ROOT"
  rm -rf "$MODEL.tmp"
  # same filesystem (/var/tmp): a rename, not a second 4 GiB copy
  mv model/Qwen2-VL-2B-Instruct-895c3a49 "$MODEL.tmp"
  mv "$MODEL.tmp" "$MODEL"
fi

PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$ENV/bin/python" -s -P - \
    "$device" "$MODEL" "$ENV" > out/provision.json <<'PY'
import glob, hashlib, json, os, socket, sys
from pathlib import Path
sys.path.insert(0, os.environ["PYTHONPATH"])
device, model, env = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
from vision_rl import vlm_local_training as vlt
pre = vlt.preflight(device)
vlt._verified_model_directory(model.resolve(), vlt.DEFAULT_MODEL)
records = sorted(glob.glob(f"{env}/lib/python3.12/site-packages/torch-*.dist-info/RECORD"))
import torch
# An existing env is reused, so it must be the build this device needs (the path does not
# encode cpu vs cuda): a CUDA wheel on a cpu plan, or the reverse, refuses.
if (device == "cuda") != (torch.version.cuda is not None):
    sys.exit(f"{env} holds torch {torch.__version__} (cuda {torch.version.cuda}); "
             f"device {device} needs the other build: remove the env and provision again")
receipt = {"host": socket.gethostname(), "device": device, "python": env + "/bin/python",
           "preflight": pre, "model_directory": str(model), "model_verified": True,
           "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "torch_record_sha256": hashlib.sha256(Path(records[0]).read_bytes()).hexdigest()
           if records else None}
if device == "cuda":
    receipt["gpu"] = torch.cuda.get_device_name(0)
    receipt["gpu_total_mib"] = torch.cuda.get_device_properties(0).total_memory // 2**20
print(json.dumps(receipt, sort_keys=True, default=str))
PY
cat out/provision.json
