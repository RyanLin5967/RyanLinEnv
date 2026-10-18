"""Stage the per-host payload for the exploratory sweep (Mac side, light: links + hashes).

  python3 -I scripts/exploratory_sweep/payload.py --out DIR [--repo ROOT_CHECKOUT]

DIR then holds exactly what a host needs besides the venv and the model weights (which the
host fetches itself at the pinned revision and verifies against model-meta/):

  code/      this worktree's src/vision_rl, scripts/exploratory_sweep, requirements lock
  data/      data_spec.json, manifests/*.jsonl and manifests/images/train/<id>.jpg
  adapters/  W and I: <name>/final_adapter/ plus the run's local_training.json beside it
  model-meta/ model-source.json and .pinned-download.json of the pinned Qwen2-VL-2B snapshot
  PAYLOAD.json  {relative path: sha256} for every file above, plus identities

Allowlist guard (refuses, never warns): an image enters only if its id is a native TRAIN
id of development/train.jsonl, its source is development/images/train/<id>.jpg, and its
bytes match encoded_sha256 in audit/train-images.json. Nothing from images/validation,
sealed/ or any TEST listing can enter, because nothing outside that allowlist is read.
Blind spot: the guard trusts audit/train-images.json and development/train.jsonl as the
definition of native TRAIN; both are pinned by sha256 below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[2]
STUDY = "aircraft-exploratory-sweep-v2"
TRAIN_JSONL_SHA256 = "909ea0c93f72feece41ef0ea8896bc74a751f4753d2f4ec1ac642b2c84783eec"
INITS = {
    "W": ("runs/aircraft-warmstart-v1/final_adapter",
          "f05840be84e1058e8fe8a011b57402ccfab5f5fb751c37ab6d8a6b84fab52c1c"),
    "I": ("runs/aircraft-controlled400-v1/setup/sft-1e5-time/published/final_adapter",
          "8f27d0ca170284a20b1f5672035c914639ddc2b2058d329da87b57a1cec46af7"),
}
MODEL = "data/models/Qwen2-VL-2B-Instruct-895c3a49"
REFERENCE_VENV = ".venv-detection"


def reference_pins(repo: Path) -> dict:
    """{normalised name: version} of every distribution in the reference venv."""
    site = sorted((repo / REFERENCE_VENV / "lib").glob("python3.*/site-packages"))
    if len(site) != 1:
        raise SystemExit("refusing: cannot find exactly one reference site-packages")
    pins = {}
    for d in site[0].glob("*.dist-info"):
        name, _, version = d.name[: -len(".dist-info")].rpartition("-")
        pins[name.lower().replace("_", "-").replace(".", "-")] = version
    if not {"torch", "torchvision", "transformers", "peft", "accelerate"} <= set(pins):
        raise SystemExit("refusing: reference venv lacks the core stack")
    python = (repo / REFERENCE_VENV / "pyvenv.cfg").read_text()
    version = next(l.split("=")[1].strip() for l in python.splitlines()
                   if l.startswith("version_info") or l.startswith("version "))
    return {"python": version, "packages": dict(sorted(pins.items()))}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # same volume: no copy
    except OSError:
        shutil.copy2(src, dst)


def build(out: Path, repo: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing: {out} is not empty (stale files must never ship)")
    dev = repo / "data/fgvc-aircraft-2013b/development"
    train_jsonl = dev / "train.jsonl"
    if sha_file(train_jsonl) != TRAIN_JSONL_SHA256:
        raise SystemExit("refusing: development/train.jsonl differs from the pinned TRAIN list")
    native = {}
    for line in train_jsonl.read_text().splitlines():
        r = json.loads(line)
        native[r["id"]] = r
    audit = {x["image_id"]: x for x in json.loads(
        (repo / "data/fgvc-aircraft-2013b/audit/train-images.json").read_text())}
    if set(audit) != set(native) or any(x["split"] != "train" for x in audit.values()):
        raise SystemExit("refusing: train-images.json does not describe exactly native TRAIN")

    design = WORKTREE / "docs" / STUDY
    manifests = sorted((design / "manifests").glob("*.jsonl"))
    if not manifests:
        raise SystemExit("refusing: no sweep manifests; run design.py first")
    wanted = set()
    for m in manifests:
        place(m, out / "data/manifests" / m.name)
        for line in m.read_text().splitlines():
            r = json.loads(line)
            if r["id"] not in native or r["image"] != f"images/train/{r['id']}.jpg":
                raise SystemExit(f"refusing: {m.name} row {r['id']} is not native TRAIN")
            wanted.add(r["id"])
    # Every native TRAIN image the sweep can read: all manifests' ids (all_train + holdouts).
    for image_id in sorted(wanted):
        src = dev / "images/train" / f"{image_id}.jpg"
        if src.is_symlink() or sha_file(src) != audit[image_id]["encoded_sha256"]:
            raise SystemExit(f"refusing: {src} is not the audited TRAIN image")
        place(src, out / "data/manifests/images/train" / f"{image_id}.jpg")
    place(dev / "data_spec.json", out / "data/data_spec.json")
    place(design / "splits.json", out / "data/splits.json")
    place(design / "arms-rung1.json", out / "data/arms-rung1.json")
    hier = {i: {k: audit[i][k] for k in ("manufacturer", "family", "variant")} for i in native}
    (out / "data/train_hierarchy.json").write_text(json.dumps(hier, sort_keys=True) + "\n")

    for name, (rel, digest) in INITS.items():
        src = repo / rel
        if sha_file(src / "adapter_model.safetensors") != digest:
            raise SystemExit(f"refusing: adapter {name} weights differ from {digest}")
        for f in sorted(src.iterdir()):
            if f.is_file():
                place(f, out / "adapters" / name / "final_adapter" / f.name)
        # The trainer binds an init adapter to its completed run record beside it.
        place(src.parent / "local_training.json", out / "adapters" / name / "local_training.json")
    for f in ("model-source.json", ".pinned-download.json"):
        place(repo / MODEL / f, out / "model-meta" / f)

    code = out / "code"
    tracked = subprocess.run(["git", "-C", str(WORKTREE), "ls-files", "src/vision_rl",
                              "scripts/exploratory_sweep", "scripts/freshpool_launch/h_assay.py",
                              "requirements-vlm-local.lock", "requirements-vlm-local.in"],
                             check=True, capture_output=True, text=True).stdout.split()
    untracked = subprocess.run(["git", "-C", str(WORKTREE), "ls-files", "--others",
                                "--exclude-standard", "scripts/exploratory_sweep"],
                               check=True, capture_output=True, text=True).stdout.split()
    for rel in sorted(set(tracked + untracked)):
        src = WORKTREE / rel
        if src.is_file():
            dst = code / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)  # copy, never link: the worktree keeps changing
    # The expanded-LoRA contract needs the (uncommitted) warm adapter bytes it pins.
    warm = code / "src/vision_rl/expanded_assets/warm/adapter_model.safetensors"
    shutil.copy2(repo / INITS["W"][0] / "adapter_model.safetensors", warm)

    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(out))] = sha_file(path)
    head = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(WORKTREE), "status", "--porcelain", "--",
                            "src", "scripts/exploratory_sweep"], check=True,
                           capture_output=True, text=True).stdout.strip()
    payload = {"study_id": STUDY, "schema_version": 1, "code_head": head,
               "reference_venv": REFERENCE_VENV, "pins": reference_pins(repo),
               "code_dirty": bool(dirty), "images": len(wanted), "files": files}
    raw = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
    (out / "PAYLOAD.json").write_bytes(raw)
    return {"out": str(out), "files": len(files), "images": len(wanted),
            "payload_sha256": hashlib.sha256(raw).hexdigest(), "code_head": head,
            "code_dirty": bool(dirty)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path("/Users/idide/RyanLinEnv"))
    a = p.parse_args(argv)
    print(json.dumps(build(a.out.resolve(), a.repo.resolve()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
