"""Host side of bootstrap_host.sh: verify payload, pins and model; write ROOT/RECEIPT.json.

Refuses (non-zero exit, reason on stderr) on: a payload file missing, changed or extra; an
installed package whose version differs from the reference venv's; a missing core package;
a model file that fails the trainer's own snapshot verifier after one pinned download; the
trainer's preflight failing. Nothing is warned and continued.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

MODEL_DIR_NAME = "Qwen2-VL-2B-Instruct-895c3a49"
CORE = ("torch", "torchvision", "transformers", "peft", "accelerate", "trl", "datasets")
# The only allowed departure from the reference venv (.venv-detection), each with its reason.
# .venv-detection has no trl/datasets; datasets 5.1.0 requires fsspec<=2026.7.0
# (requirements-vlm-local.in), and W and I were both trained under fsspec 2026.7.0 (their
# local_training.json package_versions). The installed version must equal the value here.
PIN_EXCEPTIONS = {"fsspec": "2026.7.0"}
# Trainer flags that stand for a wave-2 feature; the runner launches a gated arm only on a
# host whose receipt lists every feature the arm needs. Unknown features never match.
# A flag is "--name" (present in --help), ("--option", "choice") (listed in its choices),
# or ("@attr", NAME, member): member of the trainer module's constant NAME (for a value of
# a free-text option such as --response-format, whose choices --help does not list).
# Wave-2 integration (2026-10-18): the built features' real flags replace the placeholder
# names this table was pre-registered with; features nobody built yet keep theirs.
FEATURE_FLAGS = {
    "exploratory_sweep": "--exploratory-sweep",
    "cuda": ("--device", "cuda"),
    "dr_grpo": "--loss-type",
    "dapo": "--epsilon-high",
    "trie_constrained_decoding": "--constrained-decoding",
    "confuser_unlikelihood": "--confuser-unlikelihood",
    "constrained_ce": "--constrained-ce",
    "hierarchical_reward": "--label-hierarchy",
    "closed_set_prompt": ("@attr", "RESPONSE_FORMATS", "closed_set_answer_only"),
    "cot_answer_reward": ("@attr", "RESPONSE_FORMATS", "think_answer"),
    "dynamic_sampling": "--dynamic-sampling-max-draws",
    "prompts_per_update": "--prompts-per-update",
    "num_iterations": "--num-iterations",
    "lr_schedule": "--lr-scheduler",
    "rl_aux_nll": "--aux-nll-coef",
    "invalid_answer_penalty": "--invalid-reward",
    "rejection_sampling_ft": ("--mode", "rft"),
    "rft_step_boundary": "--rft-stop-after-steps",
    "cuda_float16": "--loss-scale-init",
    "sft_augmentation": "--sft-augmentation",
    "resolution_transfer": "--pixel-transfer",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def refuse(msg: str) -> None:
    print(f"REFUSE: {msg}", file=sys.stderr)
    raise SystemExit(2)


def verify_payload(payload: Path) -> dict:
    raw = (payload / "PAYLOAD.json").read_bytes()
    doc = json.loads(raw)
    expected = doc["files"]
    present = {str(p.relative_to(payload)) for p in payload.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.name != "PAYLOAD.json"}
    if present - set(expected):
        refuse(f"extra payload files: {sorted(present - set(expected))[:5]}")
    if set(expected) - present:
        refuse(f"missing payload files: {sorted(set(expected) - present)[:5]}")
    bad = [n for n, d in expected.items() if sha_file(payload / n) != d]
    if bad:
        refuse(f"payload sha256 mismatch: {bad[:5]}")
    images = [n for n in expected if n.startswith("data/manifests/images/")]
    if any(not n.startswith("data/manifests/images/train/") for n in images):
        refuse("a non-TRAIN image path is in the payload")
    return {"payload_sha256": hashlib.sha256(raw).hexdigest(), "files": len(expected),
            "images": len(images), "code_head": doc["code_head"],
            "code_dirty": doc["code_dirty"], "pins": doc["pins"]}


def norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def verify_pins(pins: dict) -> dict:
    installed = {norm(d.metadata["Name"]): d.version for d in metadata.distributions()}
    missing = [p for p in CORE if p not in installed]
    if missing:
        refuse(f"core packages missing: {missing}")
    if platform.python_version() != pins["python"]:
        refuse(f"python {platform.python_version()} != reference {pins['python']}")
    def wanted(n):
        return PIN_EXCEPTIONS.get(n, pins["packages"].get(n))

    differ = {n: (v, wanted(n)) for n, v in installed.items()
              if wanted(n) is not None and v.split("+")[0] != wanted(n).split("+")[0]}
    if differ:
        refuse(f"versions differ from the reference venv: {differ}")
    shared = sorted(n for n in installed if n in pins["packages"])
    return {"python": platform.python_version(), "shared_with_reference": len(shared),
            "not_in_reference": sorted(n for n in installed if n not in pins["packages"]),
            "pin_exceptions": PIN_EXCEPTIONS,
            "core": {n: installed[n] for n in CORE}}


def ensure_model(root: Path, payload: Path) -> dict:
    sys.path.insert(0, str(payload / "code/src"))
    from vision_rl import vlm_local_training as vlt

    model = root / "model" / MODEL_DIR_NAME
    model.mkdir(parents=True, exist_ok=True)
    for name in ("model-source.json", ".pinned-download.json"):
        shutil.copy2(payload / "model-meta" / name, model / name)
    try:
        return {"verified": vlt._verified_model_directory(model, vlt.DEFAULT_MODEL),
                "downloaded": False}
    except (OSError, ValueError, KeyError):
        pass
    spec = json.loads((model / "model-source.json").read_text())
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    snapshot_download(spec["model_repository"], revision=spec["revision"], local_dir=model,
                      allow_patterns=[e["path"] for e in spec["files"]], token=False)
    shutil.rmtree(model / ".cache", ignore_errors=True)
    try:
        verified = vlt._verified_model_directory(model, vlt.DEFAULT_MODEL)
    except (OSError, ValueError, KeyError) as exc:
        refuse(f"model fails the trainer's snapshot verifier after download: {exc}")
    return {"verified": verified, "downloaded": True,
            "download_seconds": round(time.monotonic() - started, 1)}


def features(payload: Path) -> dict:
    sys.path.insert(0, str(payload / "code/src"))
    from vision_rl import vlm_local_training as vlt

    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        vlt.main(["--help"])
    text = " ".join(out.getvalue().split())

    def has(flag):
        if isinstance(flag, str):
            return re.search(re.escape(flag) + r"(?![\w-])", text) is not None
        if flag[0] == "@attr":
            return flag[2] in getattr(vlt, flag[1], ())
        option, choice = flag
        found = re.search(re.escape(option) + r" \{([^}]*)\}", text)
        return bool(found) and choice in found.group(1).split(",")

    return {"available": sorted(f for f, flag in FEATURE_FLAGS.items() if has(flag)),
            "trainer_help_sha256": hashlib.sha256(text.encode()).hexdigest()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kind", choices=("cpu", "cuda"), required=True)
    p.add_argument("--host", required=True)
    a = p.parse_args(argv)
    payload = a.root / "payload"
    pay = verify_payload(payload)
    pins = verify_pins(pay.pop("pins"))
    model = ensure_model(a.root, payload)
    sys.path.insert(0, str(payload / "code/src"))
    from vision_rl import vlm_local_training as vlt

    pre = vlt.preflight("cpu")
    import torch

    device = {"kind": a.kind, "torch": torch.__version__,
              "isa": torch.backends.cpu.get_cpu_capability(),
              "cuda": bool(torch.cuda.is_available())}
    if a.kind == "cuda":
        if not torch.cuda.is_available():
            refuse("nvidia-smi sees a GPU but torch.cuda is unavailable")
        device.update(gpu=torch.cuda.get_device_name(0),
                      capability=list(torch.cuda.get_device_capability(0)),
                      vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    cpuinfo = Path("/proc/cpuinfo").read_text()
    flags = next((ln.split(":", 1)[1].split() for ln in cpuinfo.splitlines()
                  if ln.startswith("flags")), [])
    receipt = {
        "schema_version": 1, "study_id": "aircraft-exploratory-sweep-v2", "host": a.host,
        "root": str(a.root), "created_at": datetime.now(timezone.utc).isoformat(),
        "nproc": os.cpu_count(),
        "cpu_model": next((ln.split(":", 1)[1].strip() for ln in cpuinfo.splitlines()
                           if ln.startswith("model name")), None),
        "avx2": "avx2" in flags, "avx512f": "avx512f" in flags,
        "payload": pay, "packages": pins, "model": {
            "dir": str(a.root / "model" / MODEL_DIR_NAME),
            "manifest_sha256": model["verified"]["manifest_sha256"],
            "file_sha256": model["verified"]["file_sha256"],
            "downloaded_now": model["downloaded"]},
        "preflight": {k: pre[k] for k in ("python", "package_versions", "trl_source_sha256",
                                          "adapter_source_sha256")},
        "device": device, "features": features(payload),
    }
    raw = json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    tmp = a.root / "RECEIPT.json.tmp"
    tmp.write_text(raw)
    tmp.replace(a.root / "RECEIPT.json")
    print(json.dumps({"receipt_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                      "payload_sha256": pay["payload_sha256"], "host": a.host,
                      "root": str(a.root), "features": receipt["features"]["available"],
                      "device": device}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
